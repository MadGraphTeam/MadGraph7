#!/usr/bin/env python3
"""Recompute all generated-control corrected-cohort comparisons without evaluation."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def compare(specification):
    specification = policy.path(specification)
    spec = policy.read(specification)
    rules, correction, _ = policy.validate_policy(policy.path(spec["policy"]))
    proof = policy.validate_proof(policy.path(spec["proof"]), policy.path(spec["policy"]))
    config = policy.read(policy.path(spec["configuration"]))
    policy.validate_config(config, core=False)
    if set(spec["cases"]) != {"control-gg-h", "control-gg-hh", "control-mixed", "control-complex"}:
        raise RuntimeError("All four complete generated controls are required")
    proof_trees = {entry["binding"]["generation_source_identity"].get("tree_state_sha256")
                   for entry in proof["entries"] if entry["role"] == "candidate_corrected"}
    cache, measurements, results = {}, {}, {}
    expected = {("candidate_corrected", b) for b in (1, 6, 7, 8)} | {
        ("original_helper_corrected", b) for b in (1, 6, 7)}
    for case_id, targets in spec["cases"].items():
        if len(targets) != 7 or {(t["role"], t["backend"]) for t in targets} != expected:
            raise RuntimeError("Incomplete/duplicate corrected control cohort: " + case_id)
        groups = {0: {}, 1: {}}
        inputs, dimensions, statuses = set(), set(), {}
        for target in targets:
            filename = policy.path(target["binding"])
            if filename not in cache:
                binding = policy.checked_bindings(filename, correction)
                policy.validate_proven_runtimes(proof, binding)
                protocol, records = workers.checked_series(policy.path(binding["report"]), minimum=2)
                policy.validate_adapter(protocol, rules)
                if protocol["configuration"] != config or protocol["repetitions"] != 2:
                    raise RuntimeError("Controls require the same complete two-command configuration")
                cache[filename] = binding, records
            binding, records = cache[filename]
            candidates = [(key, item) for key, item in binding["targets"].items()
                          if item["case"] == case_id and item["role"] == target["role"]
                          and item["backend"] == target["backend"]]
            if len(candidates) != 1:
                raise RuntimeError("Missing/ambiguous target binding")
            target_key, item = candidates[0]
            if (target["role"] == "candidate_corrected" and
                    proof_trees != {item["generation_source_identity"].get("tree_state_sha256")}):
                raise RuntimeError("Candidate generator differs from the final proof")
            selected = [(key, value) for key, value in records.items()
                        if f"{key[0]}-{key[1]}-{key[2]}" == target_key
                        and value[1]["executable_sha256"] == item["executable_sha256"]]
            if len(selected) != 2 or {key[3] for key, _ in selected} != {0, 1}:
                raise RuntimeError("Missing/duplicate full control commands")
            label = f"{target['role']}/{target['backend']}"
            statuses[label] = []
            for (_, _, _, repetition), (record_path, data) in selected:
                if len({row["point"] for row in data["rows"]}) != 10:
                    raise RuntimeError("Incomplete ten-point control stream")
                settings = {key: value for key, value in data["shared_numerical_settings"].items()
                            if key != "selected_backend_qp_integrand"}
                if (settings.get("mlstabthres") != 1e-9 or
                        settings.get("collierrequiredaccuracy") != 1e-12 or
                        settings.get("colliercomputeuvpoles") is not True or
                        settings.get("colliercomputeirpoles") is not True or
                        (target["backend"] != 7 and
                         data["shared_numerical_settings"].get("selected_backend_qp_integrand") is not True)):
                    raise RuntimeError("Generated controls use different precision/pole policy")
                inputs.add((data["points_sha256"], data["parameter_card_sha256"],
                            json.dumps(settings, sort_keys=True)))
                dimensions.add(json.dumps(data["answer_dimensions"], sort_keys=True))
                groups[repetition][label] = data["rows"]
                statuses[label].append({"repetition": repetition, "counts": data["status_counts"]})
                measurements[policy.relative(record_path)] = workers.digest(record_path)
        if len(inputs) != 1 or len(dimensions) != 1:
            raise RuntimeError("Different physical inputs, numerical settings or dimensions")
        comparisons = {str(rep): policy.compare_rows(rows) for rep, rows in groups.items()}
        results[case_id] = {
            "passed": all(value["passed"] for value in comparisons.values()),
            "comparisons": comparisons, "status_counts": statuses, "points": 10,
            "corrected_targets": 7, "pairs_per_repetition": 21,
        }
    return policy.seal({
        "schema": "generated-controls-corrected-comparison-v2",
        "specification": policy.bound_file(specification),
        "comparison_source": policy.bound_file(Path(__file__)),
        "policy": policy.bound_file(policy.path(spec["policy"])),
        "proof": policy.bound_file(policy.path(spec["proof"])),
        "bindings": [policy.bound_file(filename) for filename in sorted(cache)],
        "measurements_sha256": measurements, "cases": results,
        "passed": all(value["passed"] for value in results.values()),
        "timing_acceptance": False,
        "ordinary_forced_ninja_failures_remain_failures": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path,
                        default=Path(__file__).resolve().with_name("comparison-spec.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if policy.path(args.output).exists():
        raise RuntimeError("Preserve previous comparison evidence")
    result = compare(args.specification)
    policy.bench.dump(policy.path(args.output), result)
    print(result["identity_sha256"])
    if not result["passed"]:
        raise SystemExit("Generated-control numerical comparison failed; evidence retained")
