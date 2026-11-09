#!/usr/bin/env python3
"""Explicit unchecked fixed-QP followups; only a private driver is recompiled."""
from contextlib import ExitStack
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import stat

import sys
sys.path.insert(0, str(Path.cwd() / "tools"))
import tirust_physical_stress as stress

bench, REPORT = stress.bench, stress.REPORT
specification = importlib.util.spec_from_file_location("physical_stress_unchecked_helper",
    REPORT / "protocol-sources/traced-unchecked-helper.py")
helper = importlib.util.module_from_spec(specification)
specification.loader.exec_module(helper)
helper.bench, helper.ROOT, helper.FROZEN_PROTOCOL = bench, bench.ROOT, REPORT / "protocol-sources"
helper.DIAGNOSTIC_TEMPLATE = REPORT / "protocol-sources/trace-worker.f"


def execute(case, backend, cpu, timeout):
    source = "candidate"
    label = f"candidate-{backend}-qp-init-dp-warmup-trace"
    report = REPORT / "diagnostics" / case["id"] / label
    report.mkdir(parents=True, exist_ok=False)
    original = stress.private_output(source, case)
    output = stress.PRIVATE.parent / "unchecked" / case["id"] / label
    def ignore(directory, names):
        return [name for name in names if name in ("build-history", "worker-raw")
                or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]
    shutil.copytree(original, output, symlinks=True, ignore=ignore)
    bench.output_dir = lambda requested_source, requested_case: output
    directory = bench.subprocess_dir(source, case)
    for name in stress.RAW_OUTPUTS:
        (directory / name).unlink(missing_ok=True)
    config = stress.configuration()
    config["madloop_parameters"].update(CTModeRun=4, CTModeInit=1)
    points = REPORT / "points" / case["id"] / "selected.dat"
    shutil.copy2(Path(__file__), report / "runner.py")
    record = {"case": case, "source": source, "backend": backend,
        "precision_request": "qp", "initialization_mode": 1,
        "force_stability_checks": False, "stability_checked": False,
        "performance_eligible": False, "created_utc": bench.utc_now(),
        "phase_timeout_seconds": timeout, "points_sha256": bench.digest(points),
        "runner_sha256": bench.digest(report / "runner.py"),
        "helper_sha256": bench.digest(REPORT / "protocol-sources/traced-unchecked-helper.py"),
        "original_compilation_sha256": bench.digest(original / "compilation.json"),
        "original_executable_sha256": bench.digest(next((original / "SubProcesses").glob("P*_*")) / "check"),
        "cpu": cpu, "rows": {"accepted": [], "unchecked": [], "rejected": []}}
    bench.dump(report / "specification.json", record)
    bench.dump(report / "configuration.json", config)
    phase, interrupted = "driver-only-relink-and-initialization", False
    try:
        with ExitStack() as stack:
            if backend in (8, 9):
                lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            worker, settings = helper.start_unchecked_worker(stack, source, case, backend, config,
                cpu, report, timeout, points)
            record["settings"] = settings
            phase = "single-traversal"
            try:
                worker.run(1)
            finally:
                bench.retain_observation(directory, report / "raw")
            record["observation"] = helper.parse_forced_qp(report, backend, 2)
            phase = "shutdown"
        record["status"] = "unchecked-qp-diagnostic"
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        record.update(status="interrupted" if interrupted else "timeout" if "timeout" in str(error).lower() else "failed",
                      failed_phase=phase, error=repr(error))
    finally:
        if not (report / "raw").exists():
            bench.retain_observation(directory, report / "raw")
        shutil.copy2(output / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
        shutil.copy2(output / "Cards/param_card.dat", report / "param_card.dat")
        try:
            record["rows"] = stress.diagnostic_rows(report / "raw", backend, True)
        except Exception as error:
            record.update(status="failed", postprocessing_error=repr(error))
        if (directory / "warmup.dat").exists():
            shutil.copy2(directory / "warmup.dat", report / "raw/warmup.dat")
        record["finished_utc"] = bench.utc_now()
        bench.dump(report / "result.json", record)
    print(json.dumps({"case": case["id"], "backend": backend, "status": record["status"]}), flush=True)
    if interrupted:
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("non-symbolica", "symbolica"), required=True)
    parser.add_argument("--timeout", type=float, default=180.)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error("Run from the MadGraph repository root")
    cases = {case["id"]: case for case in json.loads((REPORT / "plan.json").read_text())["cases"]}
    requests = [("qcd_n4_uux_zzgg", 9)]
    cpu = bench.select_quiet_cpu()
    for case_id, backend in requests:
        execute(cases[case_id], backend, cpu, args.timeout)


if __name__ == "__main__":
    main()
