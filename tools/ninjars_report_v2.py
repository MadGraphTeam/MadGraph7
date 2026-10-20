#!/usr/bin/env python3
"""Render a validated policy-v2 dataset without overwriting historical reports."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy

ROOT = bench.ROOT
REPORT = ROOT / "docs/benchmarkfolder"
BACKENDS = (1, 6, 7, 8)


def compare(groups, pairs=None):
    """Recompute every comparison independently of the publication summary."""
    keys = list(groups)
    if not keys:
        raise ValueError("Empty numerical cohort")
    stream = [(r["point"], r["order"]) for r in groups[keys[0]]]
    if not stream or len(stream) != len(set(stream)):
        raise ValueError("Empty or duplicated point/order stream")
    scales = defaultdict(float)
    for rows in groups.values():
        if [(r["point"], r["order"]) for r in rows] != stream:
            raise ValueError("Mismatched point/order streams")
        for row in rows:
            if len(row["coefficients"]) != 4 or not all(
                    math.isfinite(x) for x in row["coefficients"]):
                raise ValueError("Invalid Born/Laurent coefficients")
            scales[row["point"]] = max(scales[row["point"]],
                                       *map(abs, row["coefficients"]))
    pairs = list(itertools.combinations(keys, 2) if pairs is None else pairs)
    failures, worst, checks = [], 0., 0
    for left, right in pairs:
        for a, b in zip(groups[left], groups[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                bound = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
                error = abs(x-y)
                fraction = error/bound if bound else 0.
                worst = max(worst, fraction)
                checks += 1
                if error > bound:
                    failures.append({"targets": [left, right], "point": a["point"],
                                     "order": a["order"], "coefficient": coefficient,
                                     "values": [x, y], "tolerance": bound})
    return {"passed": not failures, "pairs": len(pairs), "failures": failures,
            "status": "failed" if failures else "passed",
            "points": len({p for p, _ in stream}),
            "orders": len({o for _, o in stream}), "comparisons": checks,
            "max_tolerance_fraction": worst}


def summary(runs, points, minimum):
    """Summarize already structurally validated records, retaining real usage."""
    if not runs:
        return {"status": "pending", "repetitions": 0, "median_us": None,
                "cold_seconds": None, "dp": None, "qp": None, "timed_calls": None,
                "status_counts": {}, "accuracy_counts": {}, "cpus": [],
                "minimum_us": None, "maximum_us": None, "mad_us": None, "ratio": None}
    if [r["repetition"] for r in runs] != list(range(len(runs))) or len(runs) < minimum:
        raise ValueError("Incomplete timing/reference repetitions")
    values, cold, counts, accuracy = [], [], Counter(), Counter()
    for run in runs:
        for key in ("seconds_per_point", "cold_seconds"):
            if not math.isfinite(run[key]) or run[key] <= 0:
                raise ValueError("Invalid measured time")
        if sum(run["timed_status_counts"].values()) != run["loops"] * points:
            raise ValueError("Incomplete timed status histogram")
        for code, count in run["timed_status_counts"].items():
            if type(count) is not int or count < 0:
                raise ValueError("Invalid timed status count")
            bench.validate_status(int(code), run["backend"])
        values.append(run["seconds_per_point"] * 1e6)
        cold.append(run["cold_seconds"])
        counts.update(run["timed_status_counts"])
        accuracy.update(run["status_counts"])
    median = statistics.median(values)
    return {"status": "measured", "repetitions": len(runs), "median_us": median,
            "cold_seconds": statistics.median(cold),
            "dp": sum(v for k, v in counts.items() if int(k)//100 == 2),
            "qp": sum(v for k, v in counts.items() if int(k)//100 == 3),
            "timed_calls": sum(counts.values()), "status_counts": dict(sorted(counts.items())),
            "accuracy_counts": dict(sorted(accuracy.items())),
            "cpus": sorted({r["cpu"] for r in runs}), "minimum_us": min(values),
            "maximum_us": max(values),
            "mad_us": statistics.median(abs(v-median) for v in values), "ratio": None,
            "executable_sha256": sorted({r["executable_sha256"] for r in runs})}


def compare_recorded(actual, recorded):
    """Publication booleans and mismatch records must agree with raw data."""
    if {k: actual[k] for k in ("passed", "pairs", "failures")} != recorded:
        raise ValueError("Stale or forged numerical dataset summary")


def generation_for_build(record, build):
    """A historical model-correction export is not a later measured export."""
    if record.get("source_identity") != build["generation_source_identity"]:
        return None
    if record.get("parameter_card_sha256") != build["parameter_card_sha256"]:
        raise ValueError("Generation parameter card differs from measured build")
    seconds = record.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid generation duration")
    return record


def build_snapshot(dataset_path, timestamp=None, generation_inputs=None):
    dataset_path = policy.path(dataset_path)
    dataset = policy.validate_dataset(dataset_path)
    inputs = {}

    def read(filename):
        filename = policy.path(filename)
        raw = filename.read_bytes()
        inputs[str(filename.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    read(dataset_path)
    rules = read(policy.check_file(dataset["policy"]))
    config = read(policy.check_file(rules["configuration"]))
    correction = read(policy.check_file(rules["correction"]))
    proof = read(policy.check_file(dataset["representative_proof"]))
    bindings = [read(policy.check_file(dataset[key]))
                for key in ("timing_bindings", "reference_bindings")]
    provenance = read(REPORT / "provenance.json")
    # Historical diagnostics are tied to their old data and must not be shown as
    # findings of this independently validated dataset.
    provenance.pop("outstanding_findings", None)
    groups = defaultdict(list)
    for filename, checksum in dataset["measurements_sha256"].items():
        record = read(filename)
        if inputs[filename] != checksum:
            raise ValueError("Measurement changed after dataset validation")
        matches = [b for b in bindings if policy.path(filename).is_relative_to(policy.path(b["report"]))]
        if len(matches) != 1:
            raise ValueError("Ambiguous timing/reference source role")
        target = matches[0]["targets"][f"{record['case']}-{record['source']}-{record['backend']}"]
        role = target["role"]
        if record["backend"] not in policy.ROLES[role]:
            raise ValueError("Backend is invalid for source role")
        groups[record["case"], role, record["backend"]].append(record)
    for runs in groups.values():
        runs.sort(key=lambda r: r["repetition"])
    cases = []
    for case in config["cases"]:
        case_id = case["id"]
        timings = {
            source: {str(b): summary(groups.get((case_id, role, b), []),
                                    config["points_per_case"], config["repetitions"])
                     for b in BACKENDS}
            for source, role in (("baseline", "original"), ("candidate", "candidate_corrected"))
        }
        references = {str(b): summary(groups.get((case_id, "original_helper_corrected", b), []),
                                     config["points_per_case"], 2) for b in (1, 6, 7)}
        agreement = {"status": "pending", "points": None, "orders": None,
                     "comparisons": None, "max_tolerance_fraction": None}
        original = {"status": "pending", "failures": [], "pairs": 0}
        performance = {"status": "pending", "runtime_ratio": None,
                       "ratio_confidence_interval_95": None}
        point_hash = parameter_hash = None
        if case_id in dataset["completed_cases"]:
            rows = {f"{role}/{b}": groups[case_id, role, b][0]["rows"]
                    for role, backends in policy.ROLES.items() for b in backends}
            corrected = {k: v for k, v in rows.items() if not k.startswith("original/")}
            agreement = compare(corrected)
            pairs = [(f"original/{b}", f"original_helper_corrected/{b}") for b in (1, 6, 7)]
            pairs += [(f"original/{b}", f"candidate_corrected/{b}") for b in (1, 6, 7)]
            pairs += [("original/6", "candidate_corrected/8")]
            original = compare(rows, pairs)
            recorded = dataset["cases"][case_id]
            compare_recorded(agreement, recorded["corrected_agreement"])
            compare_recorded(original, recorded["original_diagnostics"])
            performance = bench.runtime_statistics(
                [r["seconds_per_point"] for r in groups[case_id, "original", 6]],
                [r["seconds_per_point"] for r in groups[case_id, "candidate_corrected", 8]],
                config["max_runtime_ratio"], config["seed"])
            if performance != recorded["performance"]:
                raise ValueError("Stale performance conclusion")
            first = groups[case_id, "candidate_corrected", 8][0]
            point_hash, parameter_hash = first["points_sha256"], first["parameter_card_sha256"]
            for item in timings["candidate"].values():
                item["ratio"] = item["median_us"] / timings["baseline"]["6"]["median_us"]
        preparation = {source: {"generation": None, "compilation": None}
                       for source in ("baseline", "candidate")}
        for source in preparation:
            key = f"{case_id}-{source}-6"
            if key in bindings[0]["targets"]:
                build = read(policy.check_file(bindings[0]["targets"][key]["build"]))
                preparation[source]["compilation"] = build["origin_compilation"]
                c = correction["cases"][case_id]
                record = (c["original_generation"] if source == "baseline" else
                          read(policy.check_file(c["generation_records"][2])))
                retained = (policy.path(generation_inputs) if generation_inputs is not None else
                            policy.path(bindings[0]["report"]) / "inputs") / source / case_id
                if source == "candidate" and (
                        generation_inputs is not None or (retained / "generation.json").is_file()):
                    record = read(retained / "generation.json")
                    manifest = read(retained / "compiled-sources.json")
                    if (bench.content_digest(manifest["files"]) != manifest["files_sha256"] or
                            manifest["files_sha256"] != build["original_generated_sources_sha256"]):
                        raise ValueError("Generation source manifest differs from measured build")
                    if generation_for_build(record, build) is None:
                        raise ValueError("Generation identity differs from measured build")
                preparation[source]["generation"] = generation_for_build(record, build)
        cases.append(dict(case, timings=timings, corrected_reference_usage=references,
                          agreement=agreement, original_diagnostics=original,
                          performance=performance, preparation=preparation,
                          point_hash=point_hash, parameter_hash=parameter_hash))
    counts = {"cases": len(cases)}
    for gate, name in (("agreement", "correctness"), ("performance", "performance")):
        for status in ("passed", "failed", "unresolved"):
            counts[f"{name}_{status}"] = sum(c[gate]["status"] == status for c in cases)
    counts["measured_candidate_cells"] = sum(
        t["status"] == "measured" for c in cases for t in c["timings"]["candidate"].values())
    counts["original_mismatch_records"] = sum(len(c["original_diagnostics"]["failures"]) for c in cases)
    return {"schema_version": 2, "generated_utc": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M UTC"),
            "dataset_identity_sha256": dataset["identity_sha256"],
            "policy_identity_sha256": rules["identity_sha256"],
            "correction_identity_sha256": correction["identity_sha256"],
            "representative_proof_identity_sha256": proof["identity_sha256"],
            "dataset_path": str(dataset_path.relative_to(ROOT)),
            "config": config, "provenance": provenance, "cases": cases, "summary": counts,
            "all_cases_passed": dataset["all_cases_passed"],
            "host_cpu": provenance.get("host_cpu", provenance.get("host", {}).get("Model name", "Not recorded")),
            "inputs_sha256": dict(sorted(inputs.items()))}


def render(dataset_path, destination, timestamp=None, compile_pdf=False,
           generation_inputs=None):
    destination = policy.path(destination)
    if destination.exists():
        raise ValueError("Report destination exists; preserve the previous report")
    snapshot = build_snapshot(dataset_path, timestamp, generation_inputs)
    template = REPORT / "policy-v2/template/benchmark.typ"
    style = REPORT / "style.typ"
    snapshot["render_sources_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), template, style)}
    destination.mkdir(parents=True)
    bench.dump(destination / "report-data.json", snapshot)
    shutil.copy2(template, destination / "benchmark.typ")
    shutil.copy2(style, destination / "style.typ")
    if compile_pdf:
        subprocess.run(["typst", "compile", "benchmark.typ", "benchmark.pdf"],
                       cwd=destination, check=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timestamp")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--generation-inputs", type=Path,
                        help="retained final CLI inputs root (candidate/CASE/generation.json "
                             "and compiled-sources.json), verified against measured builds")
    args = parser.parse_args()
    output = args.output or args.dataset.parent / "report"
    snapshot = render(args.dataset, output, args.timestamp, args.compile,
                      args.generation_inputs)
    print(json.dumps({"report": str(output), "summary": snapshot["summary"],
                      "all_cases_passed": snapshot["all_cases_passed"]}, indent=2))


if __name__ == "__main__":
    main()
