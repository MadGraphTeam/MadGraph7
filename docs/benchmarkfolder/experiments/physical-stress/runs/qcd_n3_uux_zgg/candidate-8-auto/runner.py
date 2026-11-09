#!/usr/bin/env python3
"""Bounded real-mass stress checks with frozen workers; never acceptance timing.

Run from the MG root after sourcing tools/tirust_env.sh. Preparation copies
generated process sources/cards and keeps the original immutable library links.
No generated source or compilation provenance is rewritten, and no build runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
from pathlib import Path
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOLS = REPOSITORY / "docs/benchmarkfolder/experiments/physical-stress/protocol-sources"
RUNNER_BYTES = Path(__file__).read_bytes()
if FROZEN_PROTOCOLS.is_dir():
    sys.path.insert(0, str(FROZEN_PROTOCOLS))
import tirust_benchmark as bench
from tirust_benchmark_worker import RAW_OUTPUTS, validate_status

# Frozen protocol copies retain their original implementation; restore only the
# installation-location constants that normally derive from their source path.
bench.ROOT = REPOSITORY
bench.REPORT = REPOSITORY / "docs/benchmarkfolder"
bench.WORK = REPOSITORY / ".benchmark-work/tirust"
bench.BASELINE = bench.WORK / "baseline"
sys.path.insert(0, str(REPOSITORY))

REPORT = bench.REPORT / "experiments/physical-stress"
PRIVATE = bench.ROOT / ".tirust-build/physical-stress/outputs"
SELECTED = {"qcd_n2_gg_ttx": 1, "qcd_n3_uux_zgg": 2,
            "qcd_n4_uux_zzgg": 3, "ew_n3_epem_ttxa": 1}
ORIGINAL_OUTPUT = bench.output_dir


def private_output(source, case):
    return PRIVATE / source / case["id"]


def configuration():
    config = json.loads((REPORT / "cases.json").read_text())
    if config["madloop_parameters"]["MLStabThres"] != 1e-6:
        raise ValueError("This stress campaign requires the frozen strict 1e-6 policy")
    config["points_per_case"] = 2
    return config


def prepare():
    REPORT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(bench.REPORT / "cases.json", REPORT / "cases.json")
    config = configuration()
    protocols = [Path(__file__), *(bench.ROOT / "tools" / name for name in
                 ("tirust_benchmark.py", "tirust_benchmark_worker.py",
                  "tirust_benchmark_worker.f", "tirust_benchmark_compiler.py"))]
    protocol_dir = REPORT / "protocol-sources"
    protocol_dir.mkdir()
    for path in protocols:
        shutil.copy2(path, protocol_dir / path.name)
    cases = [case for case in config["cases"] if case["id"] in SELECTED]
    for case in cases:
        case_id, nexternal = case["id"], case["nfinal"] + 2
        point_dir = REPORT / "points" / case_id
        point_dir.mkdir(parents=True)
        parents, rows, points = [], [], []
        for role, suffix, index in (("ordinary-control", "", 1),
                                     ("stress", "-stress", SELECTED[case_id])):
            path = bench.REPORT / "points" / (case_id + suffix + ".dat")
            metadata = json.loads(path.with_suffix(".json").read_text())
            if bench.digest(path) != metadata["sha256"]:
                raise ValueError("Parent point stream changed")
            lines = path.read_text().splitlines()
            if int(lines[0]) != len(metadata["points"]):
                raise ValueError("Parent DAT point count differs from its actual metadata list")
            start = 1 + (index - 1) * nexternal
            selected = lines[start:start + nexternal]
            if len(selected) != nexternal:
                raise ValueError("Missing selected point")
            rows.extend(selected)
            points.append(metadata["points"][index - 1])
            for item in (path, path.with_suffix(".json")):
                shutil.copy2(item, point_dir / ("parent" + suffix + item.suffix))
            parents.append({"role": role, "source": str(path.relative_to(bench.ROOT)),
                            "sha256": metadata["sha256"], "source_point_index": index,
                            "point_id": metadata["points"][index - 1]["id"]})
        derived = point_dir / "selected.dat"
        derived.write_text("2\n" + "\n".join(rows) + "\n")
        inputs = dict(metadata["inputs"], count=2)
        bench.dump(derived.with_suffix(".json"), {
            "inputs": inputs, "points": points, "sha256": bench.digest(derived),
            "parents": parents, "scope": "Point1 is the ordinary initialization/control anchor; point2 is the sole selected stress point. Exact DAT rows are copied, not regenerated. Parent metadata count is the original generator setting; this derivative records its actual count."})
        for source in ("baseline", "candidate"):
            original, output = ORIGINAL_OUTPUT(source, case), private_output(source, case)
            inputs_dir = REPORT / "inputs" / source / case_id
            inputs_dir.mkdir(parents=True)
            def ignore(directory, names):
                excluded = {"build-history", "worker-raw", "worker-protocol.fifo", "__pycache__"}
                if Path(directory) == original:
                    excluded.add("lib")
                return set(names) & excluded
            shutil.copytree(original, output, symlinks=True, ignore=ignore)
            (output / "lib").symlink_to(original / "lib", target_is_directory=True)
            compilation = json.loads((output / "compilation.json").read_text())
            directory = next((output / "SubProcesses").glob("P*_*"))
            if bench.generated_sources(output) != compilation["sources"]:
                raise ValueError("Copied generated sources differ")
            if bench.digest(directory / "check") != compilation["executable_sha256"]:
                raise ValueError("Copied executable differs")
            if bench.dependency_inputs(directory) != compilation["link_inputs_after"]:
                raise ValueError("Copied link inputs differ")
            for name in ("generation.json", "compilation.json"):
                shutil.copy2(output / name, inputs_dir / name)
            shutil.copy2(output / "Cards/param_card.dat", inputs_dir / "param_card.dat")
            links = {str(path.relative_to(output)): {"target": str(path.readlink()),
                    "resolved": str(path.resolve()),
                    "sha256": bench.digest(path) if path.is_file() else None}
                    for path in output.rglob("*") if path.is_symlink()}
            original_directory = next((original / "SubProcesses").glob("P*_*"))
            original_runtime, private_runtime = bench.runtime_identity(original_directory), bench.runtime_identity(directory)
            if original_runtime["libraries"] != private_runtime["libraries"]:
                raise ValueError("Relocated worker resolves different dynamic libraries")
            bench.dump(inputs_dir / "isolation.json", {"original_output": str(original),
                "private_output": str(output), "copied_executable_sha256": bench.digest(directory / "check"),
                "source_inventory_equal": True, "compilation_record_unchanged": True,
                "library_policy": "Unchanged original lib directory is shared read-only; private cards, filters and worker outputs are independent.",
                "symlinks": links, "original_runtime": original_runtime, "private_runtime": private_runtime})
    bench.dump(REPORT / "plan.json", {"created_utc": bench.utc_now(), "cases": cases,
        "selected_stress_parent_indices": SELECTED,
        "primary_runs": [{"source": "baseline", "backend": 1}] +
                        [{"source": "candidate", "backend": backend} for backend in (1, 6, 7, 8, 9)],
        "points_per_run": 2, "stress_points_per_run": 1, "default_phase_timeout_seconds": 180.,
        "performance_eligible": False, "stability_threshold": 1e-6,
        "coefficient_comparison_tolerance": config["relative_tolerance"],
        "protocol_sha256": {path.name: bench.digest(path) for path in protocols}})


def diagnostic_rows(path, backend, forced_qp):
    accepted, unchecked, rejected = [], [], []
    values = path / "values.dat"
    if not values.exists():
        return {"accepted": accepted, "unchecked": unchecked, "rejected": rejected}
    for line in values.read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Malformed retained numerical row")
        point, order, status = map(int, fields[:3])
        numbers = list(map(float, fields[3:]))
        row = {"point": point, "order": order, "status": status,
               "role": "ordinary-control" if point == 1 else "stress",
               "raw_numbers": fields[3:]}
        if not all(math.isfinite(value) for value in numbers):
            rejected.append(dict(row, reason="nonfinite output"))
            continue
        row.update(coefficients=numbers[:4], accuracy=numbers[-1])
        if forced_qp and status == 120 and numbers[-1] == -1:
            unchecked.append(dict(row, precision="QP", stability_checked=False))
            continue
        try:
            decoded = validate_status(status, [backend])
            if numbers[-1] < 0 or numbers[-1] > 1e-6:
                raise ValueError("Missing or above-policy stability estimate")
        except ValueError as error:
            rejected.append(dict(row, reason=str(error)))
        else:
            accepted.append(dict(row, precision=decoded["precision"], stability_checked=True))
    return {"accepted": accepted, "unchecked": unchecked, "rejected": rejected}


def execute(case, source, backend, precision, timeout, cpu):
    config = configuration()
    if precision == "qp":
        config["madloop_parameters"].update(CTModeRun=4, CTModeInit=4)
    run = REPORT / "runs" / case["id"] / f"{source}-{backend}-{precision}"
    run.mkdir(parents=True, exist_ok=False)
    points = REPORT / "points" / case["id"] / "selected.dat"
    record = {"case": case, "source": source, "backend": backend, "precision_request": precision,
              "created_utc": bench.utc_now(), "performance_eligible": False,
              "phase_timeout_seconds": timeout, "points_sha256": bench.digest(points),
              "configuration_sha256": bench.digest(REPORT / "cases.json"),
              "stability_checked": precision == "auto", "cpu": cpu}
    (run / "runner.py").write_bytes(RUNNER_BYTES)
    record["runner_sha256"] = bench.digest(run / "runner.py")
    record["protocol_sha256"] = {path.name: bench.digest(path) for path in FROZEN_PROTOCOLS.iterdir()
                                 if path.is_file() and path.name != "tirust_physical_stress.py"}
    bench.dump(run / "specification.json", record)
    directory = bench.subprocess_dir(source, case)
    for name in RAW_OUTPUTS:
        (directory / name).unlink(missing_ok=True)
    phase = "startup-validation-and-initialization"
    try:
        with ExitStack() as stack:
            if backend in (8, 9):
                lock = stack.enter_context((bench.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            worker, settings = bench.start_worker(stack, source, case, [backend], config, cpu, run,
                                                  timeout=timeout, points_path=points)
            record["settings"] = settings
            phase = "single-traversal"
            if precision == "auto":
                record["observation"] = bench.observe(worker, [backend], 1, run / "raw", count=2)
            else:
                try:
                    worker.run(1)
                finally:
                    bench.retain_observation(directory, run / "raw")
            phase = "shutdown"
        record["status"] = "complete" if precision == "auto" else "unchecked-qp-diagnostic"
    except Exception as error:
        record.update(status="timeout" if "timeout" in str(error).lower() else "failed",
                      failed_phase=phase, error=repr(error))
    finally:
        if not (run / "raw").exists():
            bench.retain_observation(directory, run / "raw")
        shutil.copy2(private_output(source, case) / "Cards/MadLoopParams.dat", run / "MadLoopParams.dat")
    record["rows"] = {"accepted": [], "unchecked": [], "rejected": []}
    try:
        record["rows"] = diagnostic_rows(run / "raw", backend, precision == "qp")
        if record["status"] == "complete" and record["rows"]["rejected"]:
            record["status"] = "rejected-policy"
        if precision == "qp" and record["status"] == "unchecked-qp-diagnostic":
            dimensions = list(map(int, (run / "raw/dimensions.dat").read_text().split()))
            if len(dimensions) != 3 or dimensions[0] != max(dimensions[1:]):
                raise ValueError("Invalid forced-QP dimensions")
            expected = [(point, order) for point in (1, 2) for order in range(dimensions[0] + 1)]
            if (record["rows"]["accepted"] or record["rows"]["rejected"] or
                    [(row["point"], row["order"]) for row in record["rows"]["unchecked"]] != expected):
                raise ValueError("Forced-QP stream incomplete or not entirely unchecked status120")
    except Exception as error:
        record.update(status="failed", postprocessing_error=repr(error))
    bench.dump(run / "result.json", record)
    print(json.dumps({key: record[key] for key in ("source", "backend", "precision_request", "status")}
                     | {"case": case["id"]}), flush=True)
    return record


def summarize():
    config = configuration()
    results = [json.loads(path.read_text()) for path in sorted((REPORT / "runs").glob("*/*/result.json"))]
    comparisons = []
    for left_index, left in enumerate(results):
        for right in results[left_index + 1:]:
            if left["case"] != right["case"]:
                continue
            if any(record["status"] not in ("complete", "unchecked-qp-diagnostic") for record in (left, right)):
                continue
            dimensions = []
            for record in (left, right):
                label = f'{record["source"]}-{record["backend"]}-{record["precision_request"]}'
                values = list(map(int, (REPORT / "runs" / record["case"]["id"] / label / "raw/dimensions.dat").read_text().split()))
                if len(values) != 3 or values[0] != max(values[1:]):
                    raise ValueError("Invalid retained dimensions")
                dimensions.append(values[0])
            if dimensions[0] != dimensions[1]:
                continue
            for role, point in (("ordinary-control", 1), ("stress", 2)):
                rows_left = [row for category in ("accepted", "unchecked") for row in left["rows"][category] if row["point"] == point]
                rows_right = [row for category in ("accepted", "unchecked") for row in right["rows"][category] if row["point"] == point]
                if not rows_left or not rows_right:
                    continue
                expected = [(point, order) for order in range(dimensions[0] + 1)]
                if any([(row["point"], row["order"]) for row in rows] != expected for rows in (rows_left, rows_right)):
                    continue
                identity = lambda record: f'{record["source"]}-{record["backend"]}-{record["precision_request"]}'
                comparisons.append({"case": left["case"]["id"], "role": role,
                    "reference": identity(left), "candidate": identity(right),
                    "stability_checked_both": left["stability_checked"] and right["stability_checked"],
                    **bench.compare_rows(rows_left, rows_right, config["relative_tolerance"])})
    summary = {"scope": "Four real-mass stress points; one independent ordinary control anchor per process. Timings unqualified. Forced-QP diagnostic coefficients never count as checked stability.",
               "created_utc": bench.utc_now(), "performance_eligible": False,
               "primary_policy": {"MLStabThres": 1e-6, "coefficient_comparison_tolerance": config["relative_tolerance"]},
               "runs": results, "comparisons": comparisons}
    bench.dump(REPORT / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--case", choices=tuple(SELECTED))
    parser.add_argument("--source", choices=("baseline", "candidate"))
    parser.add_argument("--backend", type=int, choices=(1, 6, 7, 8, 9))
    parser.add_argument("--precision", choices=("auto", "qp"), default="auto")
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error("Run from the MG repository root")
    if args.command == "prepare":
        prepare()
        return
    bench.output_dir = private_output
    if args.command == "summarize":
        summarize()
        return
    if args.precision == "qp" and (args.case is None or args.source is None or args.backend is None or args.backend == 7):
        parser.error("Forced-QP diagnostics require one explicit case/source/backend with QP support")
    cpu = bench.select_quiet_cpu() if args.cpu is None else args.cpu
    if cpu not in bench.os.sched_getaffinity(0) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid CPU or timeout")
    plan = json.loads((REPORT / "plan.json").read_text())
    for case in plan["cases"]:
        if args.case and case["id"] != args.case:
            continue
        for specification in plan["primary_runs"]:
            if args.source and specification["source"] != args.source:
                continue
            if args.backend and specification["backend"] != args.backend:
                continue
            execute(case, specification["source"], specification["backend"], args.precision, args.timeout, cpu)
            summarize()


if __name__ == "__main__":
    main()
