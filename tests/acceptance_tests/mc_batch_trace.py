"""Audit independent MINT owners at the native real provider and consumer.

Run on job-local traces made with MG7_MC_BATCH_TRACE=1. Distinct jobs may reuse
draw and request numbers. Each job must preserve its own issue/commit order.
"""
import argparse
from collections import Counter
import gzip
import json
import math
import struct


class TraceError(ValueError):
    """A trace cannot establish native ownership or physical dispatch."""


def require(condition, message):
    if not condition:
        raise TraceError(message)


def binary64(word):
    require(len(word) == 16, "expected a binary64 word")
    return struct.unpack(">d", bytes.fromhex(word))[0]


def read_job(lines):
    lines = iter(lines)
    require(next(lines, "").strip() == "MG7 MC batch trace v1", "unknown trace format")
    draw = committed = last_request = last_call = 0
    mode = None
    live, requests, calls = {}, {}, {}
    written, surplus, bound = set(), set(), set()
    accepted, rejected = set(), set()
    counts = Counter()
    occupancy = Counter()
    for line_number, line in enumerate(lines, 2):
        fields = line.split()
        try:
            tag, *args = fields
            if tag in ("DRAW", "COMMIT", "WRITE", "SURPLUS", "ACCEPT", "REJECT"):
                phase, owner = map(int, args)
                require(phase in (0, 1, 2) and mode in (None, phase), "changed MINT mode")
                mode = phase
                if tag == "DRAW":
                    require(owner == draw + 1, "lost or repeated draw")
                    draw = owner
                    live[owner] = set()
                elif tag == "COMMIT":
                    require(owner == committed + 1 and owner in live, "out-of-order or repeated commit")
                    require(not live.pop(owner), "point committed before its physical result was consumed")
                    committed = owner
                elif tag in ("ACCEPT", "REJECT"):
                    require(mode == 2 and owner == committed, "unweighting did not follow the candidate commit")
                    require(owner not in accepted and owner not in rejected, "candidate classified twice")
                    (accepted if tag == "ACCEPT" else rejected).add(owner)
                else:
                    require(mode == 2 and 0 < owner <= committed, "event released before commit")
                    if accepted or rejected:
                        require(owner in accepted, "writing rejected or unclassified candidate")
                    require(owner not in written | surplus, "duplicate event output")
                    (written if tag == "WRITE" else surplus).add(owner)
            elif tag == "REQUEST":
                owner, req, fold, sector, flavour = map(int, args[:5])
                sample = tuple(args[5:])
                require(owner in live and req == last_request + 1, "invalid request owner or sequence")
                require(not live[owner], "multiple outstanding requests for one owner")
                require(fold >= 0 and sector > 0 and flavour > 0, "invalid FKS routing")
                require(len(sample) >= 6 and (len(sample) - 2) % 4 == 0, "invalid physical momentum count")
                for word in sample:
                    binary64(word)
                requests[req] = (owner, flavour, sample)
                live[owner].add(req)
                last_request = req
            elif tag == "BIND":
                call, *ids = map(int, args)
                require(call == last_call + 1 and ids and len(ids) == len(set(ids)), "bad packed IDs")
                require(all(req in requests and req not in bound for req in ids), "bound unknown or repeated request")
                bound.update(ids)
                require(len({requests[req][0] for req in ids}) == len(ids), "repeated MINT owner")
                last_call = call
                calls[call] = dict(ids=ids, inputs={}, outputs={}, scatter={}, mapping={})
            elif tag == "CANCEL":
                call = int(args[0])
                state = calls.pop(call)
                require(args[1:] == ["first_use_unavailable"] and not state["outputs"] and not state["scatter"],
                        "cancelling an evaluated real batch")
                bound.difference_update(state["ids"])
            elif tag == "PROVIDER":
                call, n, evaluator = map(int, args[:3])
                backend = args[3]
                state = calls[call]
                require(n == len(state["ids"]) and "backend" not in state, "provider count differs from packed call")
                require(evaluator > 0 and backend in ("fortran", "umami"), "invalid provider")
                require(backend != "fortran" or n == 1, "native scalar dispatch reported as batched")
                state["backend"] = backend
                occupancy[(backend, n)] += 1
            elif tag in ("INPUT", "OUTPUT", "MAP", "SCATTER"):
                call, slot = map(int, args[:2])
                state = calls[call]
                require("backend" in state, "missing physical dispatch")
                if tag == "MAP":
                    global_slot, keep = map(int, args[2:])
                    require(slot > 0 and global_slot > 0 and keep in (0,1), "invalid order projection")
                    require(slot not in state["mapping"], "duplicate order projection")
                    state["mapping"][slot] = global_slot if keep else 0
                    continue
                require(1 <= slot <= len(state["ids"]), "lane outside logical count")
                req = state["ids"][slot-1]
                if tag == "INPUT":
                    flavour = int(args[2])
                    sample = tuple(args[3:])
                    require(slot not in state["inputs"] and (flavour, sample) == requests[req][1:],
                            "provider momentum, couplings or flavour differ from the request")
                    state["inputs"][slot] = sample[2:]
                elif tag == "OUTPUT":
                    require(slot in state["inputs"] and slot not in state["outputs"], "unmatched physical output")
                    state["outputs"][slot] = tuple(args[2:])
                else:
                    require(int(args[2]) == req, "scatter owner differs from submitted lane")
                    require(slot in state["outputs"] and slot not in state["scatter"], "unmatched scatter")
                    target = tuple(args[3:])
                    raw = state["outputs"][slot]
                    if state["backend"] == "fortran":
                        require(target == raw, "native scalar output changed on scatter")
                    else:
                        values = list(map(binary64, raw))
                        require(values and all(math.isfinite(v) for v in values), "nonfinite physical result")
                        require(set(state["mapping"]) == set(range(1,len(values)+1)), "incomplete order projection")
                        projected = [0.] * (len(target)-1)
                        total = 0.
                        threshold = max(map(abs,values)) * 1.e-12
                        for order, value in enumerate(values, 1):
                            dest = state["mapping"][order]
                            if dest:
                                require(dest <= len(projected), "invalid global order slot")
                                total += value
                                if abs(value) > threshold:
                                    projected[dest-1] = value
                        if abs(total) < threshold:
                            total = 0.
                        require(list(map(binary64,target)) == [total]+projected, "physical result scattered to another lane or order")
                    state["scatter"][slot] = target
                    owner, flavour, sample = requests[req]
                    requests[req] = owner, target
                    if len(state["scatter"]) == len(state["ids"]):
                        if state["backend"] == "umami" and len(set(state["inputs"].values())) > 1:
                            counts["distinct_point_calls"] += 1
                        del calls[call]
            elif tag == "CONSUME":
                owner, req = map(int,args[:2])
                require(req in requests and requests[req] == (owner, tuple(args[2:])),
                        "consumer owner or value differs from scattered result")
                live[owner].remove(req)
                del requests[req]
                bound.remove(req)
            else:
                raise TraceError("unknown event "+tag)
            counts[tag] += 1
        except (IndexError, KeyError, ValueError) as exc:
            raise TraceError(f"line {line_number}: {exc}") from exc
    require(draw > 0 and draw == committed and not live and not requests and not calls, "unfinished or missing draw")
    if accepted or rejected:
        require(len(accepted)+len(rejected)==draw, "incomplete hit/miss classification")
        require(accepted==written|surplus, "accepted candidates differ from written events and terminal surplus")
    return dict(mode=mode,counts=dict(counts),occupancy={f"{backend}/{n}":count for (backend,n),count in occupancy.items()})


def audit(paths, require_batch=True, expected_writes=None):
    jobs = []
    for path in paths:
        with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
            jobs.append(read_job(stream))
    check_phases(jobs, require_batch, expected_writes)
    return jobs


def check_phases(jobs, require_batch=True, expected_writes=None):
    for mode in (0,1,2):
        matching = [job["counts"] for job in jobs if job["mode"] == mode]
        require(matching, f"no native mode {mode} job")
        require(any(row.get("CONSUME",0) for row in matching), f"mode {mode} has no consumed real result")
        if require_batch:
            require(any(row.get("distinct_point_calls",0) for row in matching),
                    f"mode {mode} has no single UMAMI call with distinct physical MINT momenta")
    writes=sum(job["counts"].get("WRITE",0) for job in jobs)
    require(writes > 0, "no native event writes")
    if expected_writes is not None:
        require(writes == expected_writes, f"expected {expected_writes} LHE writes, saw {writes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-one-control", action="store_true")
    parser.add_argument("--expected-writes", type=int)
    parser.add_argument("trace", nargs="+")
    args = parser.parse_args()
    print(json.dumps(audit(args.trace, not args.width_one_control,args.expected_writes), indent=2))
