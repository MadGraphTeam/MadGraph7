#!/usr/bin/env python3
"""Recheck retained real data with corrected analysis; never evaluate physics."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_supplement_v2 as supplement
import ninjars_report_v2 as report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze():
    destination = HERE / "actual-data-analysis.json"
    require(not destination.exists(), "Preserve the existing post-fix analysis")
    prior = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-generation-v1/candidate-correctness-diagnostic-v1.json"
    rejection = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-candidate-controls-launch-v1/postcollection-first-rejection.json"
    original = json.loads(prior.read_text())
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in (prior, rejection)}
    tables, settings, annotations = defaultdict(dict), defaultdict(set), Counter()
    runs = []
    for entry in original["source_groups"]:
        binding_path = supplement.checked(entry["bindings"], inputs)
        binding, protocol, records = supplement.checked_bindings(binding_path, inputs)
        require(len(records) == entry["records"], "Changed actual collection inventory")
        runs.append({"run": entry["run"], "binding": entry["bindings"], "records": len(records),
                     "repetitions": protocol["repetitions"]})
        for (case, source, backend, repetition), (_, record) in records.items():
            numerical = supplement.comparable_numerical_settings(record)
            settings[case].add(json.dumps(numerical, sort_keys=True))
            annotation = record["shared_numerical_settings"].get("selected_backend_qp_integrand", "absent")
            annotations[f"{backend}/{annotation}"] += 1
            if repetition == 0:
                role = binding["role_map"][source]
                target = f"{role}/{backend}"
                require(target not in tables[case], "Duplicate actual correctness target")
                tables[case][target] = record["rows"]
    require(all(len(values) == 1 for values in settings.values()),
            "Actual backends changed a common physical/numerical setting")
    required = {f"{role}/{backend}" for role in ("candidate_corrected", "original_helper_corrected")
                for backend in supplement.ROLES[role]}
    cases = {}
    for case, rows in sorted(tables.items()):
        require(set(rows) == required, "Actual seven-target correctness cohort is incomplete")
        agreement = report.compare(rows)
        require(agreement["passed"] and agreement["pairs"] == 21, "Actual cross-backend disagreement")
        previous = original["cases"][case]["agreement"]
        for key in ("passed", "status", "pairs", "points", "orders", "comparisons", "max_tolerance_fraction", "failures"):
            require(agreement[key] == previous[key], "Corrected settings analysis changed physical comparisons: " + key)
        cases[case] = {"agreement": agreement, "stability": report.stability_snapshot(rows)}
    sources = ("tools/ninjars_supplement_v2.py", "tools/ninjars_stress_v2.py",
               "tools/ninjars_report_v2.py", "tools/ninjars_benchmark_policy_v2.py",
               "tools/ninjars_benchmark_workers.py", "tools/ninjars_benchmark_worker.f")
    result = {
        "schema": "madloop-backend-qp-annotation-actual-analysis-v1",
        "analysis_reader_sources_sha256": {name: digest(ROOT / name) for name in sources},
        "analysis_script_sha256": digest(Path(__file__).resolve()),
        "source_groups": runs, "cases": cases,
        "raw_qp_annotation_counts_by_backend_and_value": dict(sorted(annotations.items())),
        "common_settings_equal_after_required_qp_validation": True,
        "all_21_pair_physical_comparisons_unchanged": True,
        "inputs_sha256": dict(sorted(inputs.items())),
        "original_rejection_and_diagnostic_unchanged": True,
        "core_dataset_identity_sha256": None,
        "final_report_validation_passed": False,
        "timing_acceptance_evaluated": False,
        "physics_evaluations": 0,
        "scope": "Read-only postprocessing of completed actual100/100/12-point seven-target cohorts; no invented core identity or final timing/report declaration",
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_groups": len(runs), "cases": {
        case: {key: value["agreement"][key] for key in ("points", "pairs", "comparisons", "max_tolerance_fraction")}
        for case, value in cases.items()}, "final_report_validation_passed": False,
        "physics_evaluations": 0}, indent=2))


if __name__ == "__main__":
    analyze()
