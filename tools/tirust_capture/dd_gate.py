#!/usr/bin/env python3
"""Qualify the bridge's double-double (DD) gate from replays of one trace.

Inputs are per-call JSON lines written by `replay --calls-jsonl` for the same
production trace:

* the production variant (the installed bridge settings). Every call must
  reproduce the recorded bits; its route deltas label which calls used the DD
  tier (delta index 2, Tier106);
* one or more reference variants with a larger DD headroom (for example 40, the
  previous 2^-66 gate, or 105, which leaves no lane in DD). Their differences on
  DD-labelled calls measure the DD error against 256/512-bit evaluation.

Errors are componentwise: |difference| / max |value| of the same tensor degree
over all four Laurent slots (`max_comp_rel` of the replay), so an analytically
vanishing slot is measured against the finite part of that degree. The gate is qualified when no
DD-labelled call exceeds FAIL (default 1e-9); calls above FLAG (1e-12) are listed.

  dd_gate.py --production h26.jsonl --reference h40=h40.jsonl --reference h105=h105.jsonl --json out.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

DD = 2
HIGHER = (3, 4)


def calls(path):
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def error_of(call):
    if call["status"] != call["rec_status"]:
        return math.inf
    value = call.get("max_comp_rel")
    return math.inf if value is None else float(value)


def quantiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
    return {"p50": pick(.5), "p90": pick(.9), "p99": pick(.99), "max": ordered[-1]}


def analyse(production, references, flag=1e-12, fail=1e-9, keep=50):
    labels = {}
    sanity = {"calls": 0, "non_identical": 0, "status_mismatch": 0}
    dd_by_key = defaultdict(int)
    for call in calls(production):
        sanity["calls"] += 1
        sanity["non_identical"] += not call["identical"]
        sanity["status_mismatch"] += call["status"] != call["rec_status"]
        if call["rec_status"] == 0 and call["delta"][DD] > 0:
            labels[call["i"]] = (call["seg"], call["n"], call["rank"], call["check"])
            dd_by_key[f"n{call['n']}:r{call['rank']}:check{call['check']}"] += 1
    result = {"production": str(production), "flag": flag, "fail": fail, "production_replay": sanity,
              "dd_calls": len(labels), "dd_calls_by_key": dict(sorted(dd_by_key.items())), "references": {}}
    qualified = sanity["non_identical"] == 0 and sanity["status_mismatch"] == 0
    for name, path in references:
        errors, flagged, failing = [], [], []
        by_key = defaultdict(list)
        still_dd = unmatched = 0
        seen = 0
        for call in calls(path):
            label = labels.get(call["i"])
            if label is None:
                continue
            seen += 1
            if (call["seg"], call["n"], call["rank"], call["check"]) != label:
                raise ValueError(f"{path}: call {call['i']} does not match the production replay")
            if call["delta"][DD] > 0:
                still_dd += 1
            if not any(call["delta"][k] for k in HIGHER):
                unmatched += 1
                continue
            error = error_of(call)
            errors.append(error)
            by_key[f"n{call['n']}:r{call['rank']}:check{call['check']}"].append(error)
            entry = {"i": call["i"], "seg": call["seg"], "n": call["n"], "rank": call["rank"],
                     "check": call["check"], "error": error if math.isfinite(error) else None}
            if error > fail:
                failing.append(entry)
            elif error > flag:
                flagged.append(entry)
        if seen != len(labels):
            raise ValueError(f"{path}: {seen} of {len(labels)} DD-labelled calls present")
        compared = len(errors)
        report = {"path": str(path), "dd_calls_compared": compared,
                  "dd_calls_still_dd_in_reference": still_dd,
                  "dd_calls_without_higher_tier_in_reference": unmatched,
                  "bit_identical": sum(1 for e in errors if e == 0),
                  "above_flag": len(flagged) + len(failing), "above_fail": len(failing),
                  "error_quantiles": quantiles(errors),
                  "by_key": {k: {"calls": len(v), "max": max(v), "above_flag": sum(e > flag for e in v)}
                             for k, v in sorted(by_key.items())},
                  "failing": sorted(failing, key=lambda e: -(e["error"] or math.inf))[:keep],
                  "flagged": sorted(flagged, key=lambda e: -e["error"])[:keep]}
        result["references"][name] = report
        qualified = qualified and not failing
    result["qualified"] = qualified
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=CALLS.jsonl", required=True)
    parser.add_argument("--flag", type=float, default=1e-12)
    parser.add_argument("--fail", type=float, default=1e-9)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    references = []
    for item in args.reference:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error("--reference expects NAME=PATH")
        references.append((name, Path(path)))
    result = analyse(args.production, references, args.flag, args.fail)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sanity = result["production_replay"]
    print(f"production replay: {sanity['calls']} calls, {sanity['non_identical']} not bit-identical, "
          f"{sanity['status_mismatch']} status mismatches; {result['dd_calls']} DD-labelled calls")
    for name, report in result["references"].items():
        q = report["error_quantiles"]
        print(f"  {name}: compared {report['dd_calls_compared']} (still DD {report['dd_calls_still_dd_in_reference']}, "
              f"no higher tier {report['dd_calls_without_higher_tier_in_reference']}), identical {report['bit_identical']}, "
              f"> flag {report['above_flag']}, > fail {report['above_fail']}, max {q.get('max', 0):.3e}")
    print("qualified" if result["qualified"] else "NOT qualified")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
