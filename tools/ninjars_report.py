#!/usr/bin/env python3
"""Build the NinjaRS report snapshot from retained, measured benchmark inputs.

Run from the repository: python tools/ninjars_report.py --compile
Missing measurements remain null/pending. Calibration runs are never timings.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

try:
    from tools import ninjars_benchmark as benchmark
except ModuleNotFoundError:  # Direct invocation from tools/.
    import ninjars_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/benchmarkfolder"
BACKENDS = {1: "CutTools", 6: "Ninja", 7: "Collier", 8: "NinjaRS"}


def finite(value, label, positive=False):
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    if positive and value <= 0:
        raise ValueError(f"Nonpositive {label}: {value!r}")
    return value


class Inputs:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def read_bytes(self, path, optional=False):
        if not path.exists():
            if optional:
                return None
            raise FileNotFoundError(path)
        content = path.read_bytes()
        try:
            name = str(path.relative_to(self.root))
        except ValueError:
            name = str(path)
        self.files[name] = hashlib.sha256(content).hexdigest()
        return content

    def read(self, path, optional=False):
        content = self.read_bytes(path, optional)
        return json.loads(content) if content is not None else None


def summarize(runs, expected_repetitions, expected_points, report=None,
              protocol_cache=None, reader=None):
    if not runs:
        return {"status": "pending", "repetitions": 0, "median_us": None,
                "cold_seconds": None, "dp": None, "qp": None, "timed_calls": None,
                "status_counts": {}, "accuracy_counts": {}, "cpus": [],
                "minimum_us": None, "maximum_us": None, "mad_us": None}
    complete = len(runs) >= expected_repetitions
    for key in ("executable_sha256", "shared_numerical_settings"):
        identities = {json.dumps(run.get(key), sort_keys=True) for run in runs}
        if len(identities) != 1:
            raise ValueError(f"Mixed {key} across timing repetitions")
        if complete and any(key not in run for run in runs):
            raise ValueError(f"Missing {key} in completed timing repetitions")
    if len({run["cpu"] for run in runs}) != 1:
        raise ValueError("Different pinned CPUs across timing repetitions")
    protocol_fields = ("benchmark_protocol", "protocol_identity_sha256",
                       "protocol_validation_sha256")
    if len({tuple(run.get(key) for key in protocol_fields) for run in runs}) != 1:
        raise ValueError("Mixed benchmark protocols or validation across timing repetitions")
    try:
        for run in runs:
            benchmark.validate_benchmark_protocol(
                run, report, required=complete, minimum_repetitions=expected_repetitions,
                cache=protocol_cache, reader=reader)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    values, cold, timed, accuracy = [], [], Counter(), Counter()
    for run in runs:
        values.append(finite(run["seconds_per_point"], "seconds per point", True) * 1e6)
        cold.append(finite(run["cold_seconds"], "initialization seconds", True))
        timed.update(run["timed_status_counts"])
        accuracy.update(run["status_counts"])
        if sum(run["timed_status_counts"].values()) != run["loops"] * expected_points:
            raise ValueError("Timed histogram count differs from input count")
    dp, qp = 0, 0
    for code, count in timed.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError("Invalid timed status count")
        h, t, u = int(code) // 100, (int(code) // 10) % 10, int(code) % 10
        allowed = {1: {(2, 1), (3, 9)}, 6: {(2, 6), (3, 8)},
                   7: {(2, 7)}, 8: {(2, 8), (3, 2)}}
        if t not in (1, 2) or (h, u) not in allowed[runs[0]["backend"]]:
            raise ValueError(f"Unexpected reducer return code {code}")
        dp += count if h == 2 else 0
        qp += count if h == 3 else 0
    median = statistics.median(values)
    return {
        "status": "measured" if len(runs) >= expected_repetitions else "partial",
        "repetitions": len(runs), "median_us": median if complete else None,
        "minimum_us": min(values), "maximum_us": max(values),
        "mad_us": statistics.median(abs(x - median) for x in values),
        "cold_seconds": statistics.median(cold) if complete else None, "dp": dp, "qp": qp,
        "timed_calls": sum(timed.values()), "status_counts": dict(sorted(timed.items())),
        "accuracy_counts": dict(sorted(accuracy.items())),
        "cpus": sorted({r["cpu"] for r in runs}),
        "executable_sha256": sorted({r["executable_sha256"] for r in runs
                                     if "executable_sha256" in r}),
        "git_revisions": sorted({r["git_revision"] for r in runs if "git_revision" in r}),
    }


def correctness(groups, config):
    """Independently recompute all six pairwise coefficient comparisons."""
    samples = {b: next((r for r in groups.get(("candidate", b), [])
                       if r["repetition"] == 0), None) for b in BACKENDS}
    if not all(samples.values()):
        return {"status": "pending", "points": None, "orders": None,
                "max_tolerance_fraction": None, "failures": None,
                "comparisons": None}
    hashes = {(r["points_sha256"], r["parameter_card_sha256"])
              for r in samples.values()}
    if len(hashes) != 1:
        raise ValueError("Correctness inputs differ across reducers")
    rows = {b: r["rows"] for b, r in samples.items()}
    keys = [(r["point"], r["order"]) for r in rows[1]]
    orders = sorted({o for _, o in keys})
    expected = [(p, o) for p in range(1, config["points_per_case"] + 1)
                for o in orders]
    if not orders or orders != list(range(len(orders))) or keys != expected:
        raise ValueError("Incomplete point/order stream")
    if any([(r["point"], r["order"]) for r in group] != keys for group in rows.values()):
        raise ValueError("Different numerical point/order streams")
    scales = {}
    for group in rows.values():
        for row in group:
            if len(row["coefficients"]) != 4:
                raise ValueError("Expected Born, finite, single and double pole")
            values = [abs(finite(x, "Laurent coefficient")) for x in row["coefficients"]]
            scales[row["point"]] = max(scales.get(row["point"], 0), *values)
    maximum, failures, comparisons = 0.0, 0, 0
    for i, (point, _) in enumerate(keys):
        for left in BACKENDS:
            for right in BACKENDS:
                if left >= right:
                    continue
                for a, b in zip(rows[left][i]["coefficients"], rows[right][i]["coefficients"]):
                    tolerance = (config["relative_tolerance"] * max(abs(a), abs(b)) +
                                 config["scale_tolerance"] * scales[point])
                    difference = abs(a - b)
                    fraction = difference / tolerance if tolerance else 0.0
                    maximum = max(maximum, fraction)
                    failures += difference > tolerance
                    comparisons += 1
    return {"status": "passed" if not failures else "failed",
            "points": config["points_per_case"], "orders": len(orders),
            "max_tolerance_fraction": maximum, "failures": failures,
            "comparisons": comparisons}


def build_snapshot(report, work, timestamp=None):
    inputs = Inputs(ROOT)
    protocol_cache = {}
    config = inputs.read(report / "cases.json")
    provenance = inputs.read(report / "provenance.json")
    cases = []
    for case in config["cases"]:
        case_id = case["id"]
        groups = {}
        for source in ("baseline", "candidate"):
            for backend in BACKENDS:
                runs = []
                for path in (report / "data").glob(f"{source}-{case_id}-{backend}-*.json"):
                    repetition = path.stem.rsplit("-", 1)[-1]
                    if not repetition.isdigit():
                        continue
                    run = inputs.read(path)
                    expected = (source, case_id, backend, int(repetition))
                    actual = tuple(run[k] for k in ("source", "case", "backend", "repetition"))
                    if actual != expected:
                        raise ValueError(f"Run metadata does not match its filename: {path}")
                    runs.append(run)
                runs.sort(key=lambda r: r["repetition"])
                if [r["repetition"] for r in runs] != list(range(len(runs))):
                    raise ValueError(f"Missing repetition in {source}/{case_id}/{backend}")
                groups[source, backend] = runs
        summaries = {
            source: {str(b): summarize(groups[source, b], config["repetitions"],
                                      config["points_per_case"], report,
                                      protocol_cache, inputs.read) for b in BACKENDS}
            for source in ("baseline", "candidate")
        }
        protocols = {
            tuple(run.get(key) for key in ("benchmark_protocol", "protocol_identity_sha256",
                                           "protocol_validation_sha256"))
            for runs in groups.values() for run in runs}
        if len(protocols) > 1:
            raise ValueError(f"Mixed benchmark protocols or validation within {case_id}")
        reference = summaries["baseline"]["6"]["median_us"]
        for b in BACKENDS:
            row = summaries["candidate"][str(b)]
            row["ratio"] = row["median_us"] / reference if reference and row["median_us"] else None
        agreement = correctness(groups, config)
        recorded_agreement = inputs.read(report / "data" / f"agreement-{case_id}.json", optional=True)
        if recorded_agreement and agreement["status"] != "pending":
            if recorded_agreement["passed"] != (agreement["status"] == "passed"):
                raise ValueError(f"Stale correctness summary for {case_id}")
        performance = inputs.read(report / "data" / f"performance-{case_id}.json", optional=True)
        if performance:
            for source, backend in (("baseline", 6), ("candidate", 8)):
                actual = [r["seconds_per_point"] for r in groups[source, backend]]
                if actual != performance["measurements"][source]["samples"]:
                    raise ValueError(f"Stale performance summary for {case_id}/{source}")
                if len(actual) < config["repetitions"]:
                    raise ValueError(f"Incomplete performance repetitions for {case_id}")
                if not all(r.get("paired_baseline_candidate") for r in groups[source, backend]):
                    raise ValueError(f"Unpaired performance runs for {case_id}")
            baseline_runs, candidate_runs = groups["baseline", 6], groups["candidate", 8]
            if len({r["cpu"] for r in baseline_runs + candidate_runs}) != 1:
                raise ValueError(f"Different performance CPUs for {case_id}")
            if len({json.dumps(r["shared_numerical_settings"], sort_keys=True)
                    for r in baseline_runs + candidate_runs}) != 1:
                raise ValueError(f"Different performance numerical settings for {case_id}")
            if ([r.get("measurement_run_id") for r in baseline_runs] !=
                    [r.get("measurement_run_id") for r in candidate_runs]):
                raise ValueError(f"Different paired measurement runs for {case_id}")
            if performance["maximum_ratio"] != config["max_runtime_ratio"]:
                raise ValueError(f"Stale performance threshold for {case_id}")
            finite(performance["runtime_ratio"], "runtime ratio", True)
            for bound in performance["ratio_confidence_interval_95"]:
                finite(bound, "confidence bound", True)
        else:
            performance = {"status": "pending", "runtime_ratio": None,
                           "ratio_confidence_interval_95": None}
        preparation = {}
        for source in ("baseline", "candidate"):
            preparation[source] = {}
            for stage in ("generation", "compilation"):
                retained = report / "inputs" / source / case_id / f"{stage}.json"
                path = retained if retained.exists() else (
                    work / "outputs" / source / case_id / f"{stage}.json")
                preparation[source][stage] = inputs.read(path, optional=True)
        point_info = inputs.read(report / "points" / f"{case_id}.json", optional=True)
        hashes = sorted({r["points_sha256"] for group in groups.values() for r in group})
        cards = sorted({r["parameter_card_sha256"] for group in groups.values() for r in group})
        if len(hashes) > 1 or len(cards) > 1:
            raise ValueError(f"Mixed numerical inputs within {case_id}")
        point_hash = next(iter(hashes), None)
        if point_info is not None:
            point_data = inputs.read_bytes(report / "points" / f"{case_id}.dat")
            retained_hash = hashlib.sha256(point_data).hexdigest()
            if (retained_hash != point_info["sha256"] or
                    (point_hash is not None and point_hash != retained_hash)):
                raise ValueError(f"Retained point stream differs from its evidence for {case_id}")
            if (point_info["points"] != config["points_per_case"] or
                    int(point_data.splitlines()[0]) != config["points_per_case"]):
                raise ValueError(f"Retained point count differs for {case_id}")
            point_hash = retained_hash
        cases.append(dict(case, timings=summaries, agreement=agreement,
                          performance=performance, preparation=preparation,
                          point_info=point_info, point_hash=point_hash,
                          parameter_hash=next(iter(cards), None)))
    stats = {
        "cases": len(cases),
        "correctness_passed": sum(c["agreement"]["status"] == "passed" for c in cases),
        "correctness_failed": sum(c["agreement"]["status"] == "failed" for c in cases),
        "performance_passed": sum(c["performance"]["status"] == "passed" for c in cases),
        "performance_failed": sum(c["performance"]["status"] == "failed" for c in cases),
        "performance_unresolved": sum(c["performance"]["status"] == "unresolved" for c in cases),
        "measured_candidate_cells": sum(c["timings"]["candidate"][str(b)]["status"] == "measured"
                                        for c in cases for b in BACKENDS),
    }
    cpu = next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "Not recorded") if Path("/proc/cpuinfo").exists() else "Not recorded"
    benchmark_cpu = provenance.get("host_cpu", provenance.get("host", {}).get("Model name"))
    return {
        "schema_version": 1,
        "generated_utc": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M UTC"),
        "config": config, "provenance": provenance,
        "host_cpu": benchmark_cpu or cpu + " (report host)",
        "host_cpu_source": "benchmark provenance" if benchmark_cpu else "report-generation host",
        "cases": cases, "summary": stats, "inputs_sha256": dict(sorted(inputs.files.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="also compile benchmark.pdf with Typst")
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".benchmark-work")
    args = parser.parse_args()
    snapshot = build_snapshot(args.report_dir, args.work_dir)
    output = args.report_dir / "report-data.json"
    output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    if args.compile:
        subprocess.run(["typst", "compile", "--root", str(ROOT),
                        str(args.report_dir / "benchmark.typ"),
                        str(args.report_dir / "benchmark.pdf")], cwd=ROOT, check=True)
    print(f"{snapshot['summary']['correctness_passed']}/{len(snapshot['cases'])} "
          f"correctness cases passed; {snapshot['summary']['performance_passed']}/"
          f"{len(snapshot['cases'])} performance gates passed; wrote {output}")


if __name__ == "__main__":
    main()
