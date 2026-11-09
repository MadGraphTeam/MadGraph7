#!/usr/bin/env python3
"""Retain native numerical checks during preparation; never qualify timing."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import shutil

import tirust_benchmark as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--backends", nargs="+", type=int, default=[1, 6, 7, 8, 9])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=3600.)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label) or set(args.backends) - {1, 6, 7, 8, 9}:
        parser.error("Invalid label or backend selection")
    config = bench.load_config()
    case = next((value for value in config["cases"] if value["id"] == args.case), None)
    if case is None:
        parser.error("Unknown process")
    cpu = bench.select_quiet_cpu() if args.cpu is None else args.cpu
    if cpu not in bench.os.sched_getaffinity(0):
        parser.error("CPU is outside the allowed affinity")
    report = bench.REPORT / "experiments/numerical" / args.label
    report.mkdir(parents=True, exist_ok=False)
    protocols = [Path(__file__), bench.ROOT / "tools/tirust_benchmark.py",
                 bench.ROOT / "tools/tirust_benchmark_worker.py", bench.ROOT / "tools/tirust_benchmark_worker.f",
                 bench.ROOT / "tools/tirust_benchmark_compiler.py"]
    for path in protocols:
        shutil.copy2(path, report / path.name)
    shutil.copy2(bench.REPORT / "cases.json", report / "cases.json")
    specification = {"case": case, "performance_eligible": False,
                     "purpose": "native numerical integration check during build preparation",
                     "machine": bench.machine_identity(cpu), "host_load": bench.host_load_snapshot(),
                     "configuration_sha256": bench.digest(report / "cases.json"),
                     "protocols": {path.name: bench.digest(path) for path in protocols}}
    bench.dump(report / "specification.json", specification)
    results, reference = [], None
    for source, backend in [("baseline", 1), *(("candidate", value) for value in args.backends)]:
        run_dir = report / f"{source}-{backend}"
        run_dir.mkdir()
        try:
            with ExitStack() as stack:
                if backend in (8, 9):
                    lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                    fcntl.flock(lock, fcntl.LOCK_EX)
                worker, _ = bench.start_worker(stack, source, case, [backend], config, cpu, run_dir,
                                               timeout=args.timeout)
                observation = bench.observe(worker, [backend], 1, run_dir / "raw",
                                            count=config["points_per_case"])
            if source == "baseline":
                reference = observation
            result = {"source": source, "backend": backend, "status": "complete",
                      "performance_eligible": False, "observation": observation,
                      "comparison": bench.compare_rows(reference["rows"], observation["rows"],
                        config["relative_tolerance"]) if reference is not None else None}
            bench.dump(run_dir / "result.json", result)
        except Exception as error:
            result = {"source": source, "backend": backend, "status": "failed",
                      "performance_eligible": False, "error": repr(error)}
            bench.dump(run_dir / "failure.json", result)
        results.append(result)
        bench.dump(report / "results.json", {"case": case, "performance_eligible": False, "results": results})
        print(json.dumps({key: result[key] for key in ("source", "backend", "status")}), flush=True)
    if any(result["status"] != "complete" or result["comparison"] is None
           or not result["comparison"]["passed"] for result in results):
        raise RuntimeError(f"Unresolved native numerical failures retained in {report}")


if __name__ == "__main__":
    main()
