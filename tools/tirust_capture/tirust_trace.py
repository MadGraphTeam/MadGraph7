#!/usr/bin/env python3
"""Read TIRust bridge call traces written by shim.c and summarize them.

A segment is the sequence of calls between two cache clears; MadLoop clears the
TIRust numerical caches once per phase-space point, so segment k is point k of
the stream (the bootstrap/initialization point included).

Usage: trace.py TRACE [--json FILE]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import mmap
from pathlib import Path
import struct
import sys

DP, MP, REGISTER, CLEAR, MP_INPUTS = 1, 2, 3, 4, 5
_HEADER = struct.Struct("<7id")


class TraceError(ValueError):
    pass


def coefficient_count(rank):
    """Number of symmetric tensor coefficients up to ``rank`` in four dimensions."""
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def records(path, with_payload=True):
    """Yield one dict per record. Payload arrays are tuples of floats/bytes."""
    with open(path, "rb") as stream:
        size = Path(path).stat().st_size
        if size == 0:
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            yield from _parse(data, size, with_payload)


def parse_bytes(data, with_payload=True):
    return list(_parse(memoryview(data), len(data), with_payload))


def _need(i, n, size, kind):
    if i + n > size:
        raise TraceError(f"truncated record kind {kind} at byte {i}")


def _parse(data, size, with_payload):
    i = 0
    while i < size:
        kind = data[i]
        start = i
        i += 1
        if kind in (DP, MP, MP_INPUTS):
            _need(i, _HEADER.size, size, kind)
            n, rank, poles, finite, check, cache, status, seconds = _HEADER.unpack_from(data, i)
            i += _HEADER.size
            if not 1 <= n <= 64 or not 0 <= rank <= 64:
                raise TraceError(f"implausible n={n} rank={rank} at byte {start}")
            record = dict(kind=kind, n=n, rank=rank, poles=poles, finite=finite, check=check,
                          cache=cache, status=status, seconds=seconds, offset=start)
            if kind == DP:
                _need(i, 1, size, kind)
                record["has_inputs"] = bool(data[i])
                i += 1
                if record["has_inputs"]:
                    lengths = (4 * n, n * n, 2 * n, 1)
                    fields = ("shifts", "invariants", "masses", "mu2")
                    for name, count in zip(fields, lengths):
                        _need(i, 8 * count, size, kind)
                        if with_payload:
                            values = struct.unpack_from(f"<{count}d", data, i)
                            record[name] = values[0] if name == "mu2" else values
                        i += 8 * count
                    _need(i, 8, size, kind)
                    (olen,) = struct.unpack_from("<Q", data, i)
                    i += 8
                    if olen and status != 0:
                        raise TraceError(f"failed DP call carries outputs at byte {start}")
                    if olen not in (0, 4 * coefficient_count(rank)):
                        raise TraceError(f"unexpected DP output length {olen} at byte {start}")
                    _need(i, 16 * olen, size, kind)
                    if with_payload:
                        record["output"] = struct.unpack_from(f"<{2 * olen}d", data, i)
                    record["output_length"] = olen
                    i += 16 * olen
            elif kind == MP_INPUTS:
                _need(i, 32, size, kind)
                lengths = struct.unpack_from("<4Q", data, i)
                i += 32
                for name, count in zip(("shifts", "invariants", "masses", "scale"), lengths):
                    _need(i, count, size, kind)
                    if with_payload:
                        record[name] = bytes(data[i:i + count])
                    i += count
                _need(i, 8, size, kind)
                (olen,) = struct.unpack_from("<Q", data, i)
                i += 8
                if olen and status != 0:
                    raise TraceError(f"failed QP call carries outputs at byte {start}")
                _need(i, olen, size, kind)
                if with_payload:
                    record["output"] = bytes(data[i:i + olen])
                record["output_length"] = olen
                i += olen
            yield record
        elif kind == REGISTER:
            _need(i, 20, size, kind)
            n, rank, status = struct.unpack_from("<3i", data, i)
            (seconds,) = struct.unpack_from("<d", data, i + 12)
            i += 20
            yield dict(kind=kind, n=n, rank=rank, status=status, seconds=seconds, offset=start)
        elif kind == CLEAR:
            _need(i, 8, size, kind)
            (seconds,) = struct.unpack_from("<d", data, i)
            i += 8
            yield dict(kind=kind, seconds=seconds, offset=start)
        else:
            raise TraceError(f"bad record kind {kind} at byte {start}")


def summarize(path_or_records):
    """Aggregate counts, failures and time per segment and per (kind, n, rank, check)."""
    source = records(path_or_records, with_payload=False) if isinstance(path_or_records, (str, Path)) \
        else path_or_records
    segments = [Counter()]
    keys = defaultdict(lambda: {"calls": 0, "failed": 0, "seconds": 0.0})
    registrations = []
    totals = Counter()
    for record in source:
        kind = record["kind"]
        if kind == CLEAR:
            totals["clears"] += 1
            if sum(segments[-1].values()):
                segments.append(Counter())
            continue
        if kind == REGISTER:
            registrations.append([record["n"], record["rank"], record["status"]])
            continue
        label = "dp" if kind == DP else "qp"
        key = f"{label}:n{record['n']}:r{record['rank']}:check{record['check']}"
        entry = keys[key]
        entry["calls"] += 1
        entry["failed"] += record["status"] != 0
        entry["seconds"] += record["seconds"]
        segments[-1][label] += 1
        segments[-1][label + "_failed"] += record["status"] != 0
        totals[label] += 1
        totals[label + "_failed"] += record["status"] != 0
    segments = [dict(s) for s in segments if sum(s.values())]
    return {"totals": dict(totals), "registrations": registrations, "segments": len(segments),
            "segments_with_dp_failures": sum(1 for s in segments if s.get("dp_failed")),
            "segments_with_qp_calls": sum(1 for s in segments if s.get("qp")),
            "by_key": {k: keys[k] for k in sorted(keys)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path, help="write the summary as JSON")
    args = parser.parse_args(argv)
    summary = summarize(args.trace)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    t = summary["totals"]
    print(f"{args.trace}: {summary['segments']} segments, {t.get('dp', 0)} DP calls "
          f"({t.get('dp_failed', 0)} failed), {t.get('qp', 0)} QP calls ({t.get('qp_failed', 0)} failed), "
          f"{len(summary['registrations'])} plan registrations")
    for key, entry in summary["by_key"].items():
        print(f"  {key:24s} {entry['calls']:9d} calls {entry['failed']:7d} failed {entry['seconds']:10.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
