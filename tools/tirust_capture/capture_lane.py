#!/usr/bin/env python3
"""Rerun one campaign numerical lane with the capture shim preloaded.

The worker, cards, points and warm-up stream are exactly those of
`tirust_benchmark_numerical.py --warmup-only` (the same `start_worker` call);
only LD_PRELOAD and the shim's TIRUST_TRACE* variables are added to the worker
environment. The lane's warm-up values are written next to the trace so they can
be compared bit for bit with the campaign lane. Nothing in the campaign report
is modified.

  capture_lane.py --campaign C --case qcd_n3_uux_zgg --shim libtirust_capture.so --out DIR --cpu 17
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tirust_benchmark as bench  # noqa: E402
import tirust_trace  # noqa: E402


def capture(campaign, case_id, shim, out, cpu, backend=9, source="candidate", mp_inputs=False, timeout=None):
    from tirust_benchmark_campaign import activate_campaign
    from tirust_benchmark_warmup import parse_warmup
    activate_campaign(bench, campaign)
    config = bench.load_config()
    case = next(c for c in config["cases"] if c["id"] == case_id)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shim = Path(shim).resolve()
    base_environment = bench.environment

    def environment():
        env = base_environment()
        env.update(LD_PRELOAD=str(shim), TIRUST_TRACE=str(out / "trace"), TIRUST_TRACE_INPUTS="1",
                   TIRUST_SUMMARY=str(out / "summary"))
        if mp_inputs:
            env["TIRUST_TRACE_MP_INPUTS"] = "1"
        return env

    bench.environment = environment
    run = out / "run"
    run.mkdir()
    timeout = bench.case_worker_timeout(case) if timeout is None else timeout
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker, _ = bench.start_worker(stack, source, case, [backend], config, cpu, run, timeout=timeout)
            raw = run / "raw"
            raw.mkdir()
            for name in ("warmup.dat", "dimensions.dat"):
                shutil.copy2(worker.directory / name, raw / name)
            ready = {key: getattr(worker, key) for key in ("points", "ndim", "nso", "nborn", "cold_seconds")}
            bench.dump(raw / "ready.json", ready)
            observation = parse_warmup(raw, [backend], config["points_per_case"], ready)
            bench.dump(raw / "observation.json", observation)
    finally:
        bench.environment = base_environment
    # Helper processes (make, compilers) also load the shim; keep only traces with calls.
    traces = []
    for path in sorted(out.glob("trace.*")):
        if path.stat().st_size == 0:
            path.unlink()
        else:
            traces.append(path)
    for path in out.glob("summary.*"):
        if path.stat().st_size == 0:
            path.unlink()
    if len(traces) != 1:
        raise RuntimeError(f"expected one worker trace, found {[p.name for p in traces]}")
    summary = tirust_trace.summarize(traces[0])
    record = {"campaign": campaign, "case": case_id, "source": source, "backend": backend, "cpu": cpu,
              "trace": traces[0].name, "trace_sha256": bench.digest(traces[0]),
              "shim_sha256": bench.digest(shim), "warmup_sha256": bench.digest(run / "raw/warmup.dat"),
              "trace_summary": summary}
    (out / "capture.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--backend", type=int, default=9)
    parser.add_argument("--source", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--mp-inputs", action="store_true", help="also record QP inputs (record kind 5)")
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args(argv)
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    record = capture(args.campaign, args.case, args.shim, args.out, args.cpu, args.backend, args.source,
                     args.mp_inputs, args.timeout)
    totals = record["trace_summary"]["totals"]
    print(json.dumps({"case": args.case, "trace": record["trace"], "segments": record["trace_summary"]["segments"],
                      "dp": totals.get("dp", 0), "dp_failed": totals.get("dp_failed", 0), "qp": totals.get("qp", 0)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
