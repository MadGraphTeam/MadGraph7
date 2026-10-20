#!/usr/bin/env python3
"""Verify completed generated controls from retained files; run no amplitudes."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def checks():
    comparison_source = V2 / "final-controls-plan-fixed/compare.py"
    spec = importlib.util.spec_from_file_location("retained_control_comparison", comparison_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparison = module.compare(comparison_source.with_name("comparison-spec.json"))
    if comparison != policy.read(comparison_source.with_name("comparison.json")):
        raise RuntimeError("Retained control comparison differs from raw evidence")
    if not comparison["passed"] or comparison["timing_acceptance"]:
        raise RuntimeError("Controls have failed or are mislabelled as timing acceptance")
    correction = policy.validate_correction(V2 / "model-correction.json")
    runs = {}
    for name in (
        "controls-candidate-abi3-terminal-fixed-hh-complex-20261020",
        "controls-candidate-abi3-terminal-fixed-mixed-legacy-20261020",
        "controls-reference-hh-complex-20261020",
    ):
        binding = policy.checked_bindings(
            V2 / "runtime-bindings-controls-abi3-terminal-fixed" / (name + ".json"),
            correction)
        protocol, records = workers.checked_series(policy.path(binding["report"]), minimum=2)
        if protocol["repetitions"] != 2 or protocol["acceptance_eligible"]:
            raise RuntimeError("Wrong completed-control protocol")
        fresh_parity = {}
        if protocol["fresh_parity"]:
            for key, target in binding["targets"].items():
                report = policy.path(binding["report"])
                first = policy.read(report / "data" / (key + "-0.json"))
                fresh = policy.read(report / "data" / (key + "-fresh.json"))
                if (first["executable_sha256"] != target["executable_sha256"] or
                        fresh["executable_sha256"] != target["fresh_executable_sha256"]):
                    raise RuntimeError("Fresh/worker executable binding differs")
                fresh_parity[key] = workers.numerical_parity(first, fresh, protocol["configuration"])
        runs[name] = {
            "report": binding["report"], "target_count": len(binding["targets"]),
            "persistent_records": len(records),
            "fresh_parity": fresh_parity,
            "status_counts": {
                f"{case}-{source}-{backend}-{rep}": data["status_counts"]
                for (case, source, backend, rep), (_, data) in records.items()
            },
        }
    if sum(run["target_count"] for run in runs.values()) != 17:
        raise RuntimeError("Missing completed control target")
    if sum(run["persistent_records"] for run in runs.values()) != 34:
        raise RuntimeError("Missing completed control repetition")
    return {
        "comparison_identity_sha256": comparison["identity_sha256"],
        "proof": comparison["proof"],
        "cases": {name: {
            "passed": data["passed"], "points": data["points"],
            "corrected_targets": data["corrected_targets"],
            "pairs_per_repetition": data["pairs_per_repetition"],
            "repetitions": len(data["comparisons"]),
        } for name, data in comparison["cases"].items()},
        "new_runs": runs,
        "reused_bindings": [
            item for item in comparison["bindings"]
            if "runtime-bindings-controls-abi3-terminal-fixed/" not in item["path"]
        ],
        "acceptance_evidence": False, "physics_evaluations_started": False,
    }


def main():
    manifest = policy.verified(policy.read(HERE / "manifest.json"),
                               "generated-controls-retention-v2")
    for name, record in manifest["files"].items():
        path = policy.path(name)
        if path.stat().st_size != record["bytes"] or workers.digest(path) != record["sha256"]:
            raise RuntimeError("Changed retained file: " + name)
    result = checks()
    if result != policy.read(HERE / "validation.json"):
        raise RuntimeError("Retained verification summary differs")
    print(json.dumps({"retained_files": len(manifest["files"]), **result}, indent=2))


if __name__ == "__main__":
    main()
