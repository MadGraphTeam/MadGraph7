#!/usr/bin/env python3
"""Verify retained candidate representative results without running amplitudes."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def checks():
    proof = policy.validate_proof(
        V2 / "representative-proof-abi3-tensor-factor-v1.json",
        V2 / "policy-runtime-groups.json")
    correction = policy.validate_correction(V2 / "model-correction.json")
    results = {}
    for name in ("ggh", "mixed", "forced-qp", "ninja-near-singular"):
        identifier = f"representative-candidate-abi3-tensor-factor-v1-{name}-20261020"
        binding = policy.checked_bindings(
            V2 / "runtime-bindings-abi3-tensor-factor-v1" / (identifier + ".json"),
            correction)
        report = policy.path(binding["report"])
        protocol, series = workers.checked_series(report, minimum=2)
        assert protocol["acceptance_eligible"] is False
        targets = {}
        for key, target in binding["targets"].items():
            assert target["role"] == "candidate_corrected"
            first = policy.read(report / "data" / (key + "-0.json"))
            fresh = policy.read(report / "data" / (key + "-fresh.json"))
            assert first["executable_sha256"] == target["executable_sha256"]
            assert fresh["executable_sha256"] == target["fresh_executable_sha256"]
            parity = workers.numerical_parity(first, fresh, protocol["configuration"])
            assert parity == {"passed": True, "exact_rows": True,
                              "status_counts_match": True}
            targets[key] = {"status_counts": first["status_counts"],
                            "fresh_parity": parity}
        results[name] = {"targets": targets, "records": len(series)}
    assert sum(len(item["targets"]) for item in results.values()) == 8
    assert sum(item["records"] for item in results.values()) == 16
    return {"proof_identity_sha256": proof["identity_sha256"],
            "coverage": proof["coverage"], "candidate_runs": results,
            "acceptance_evidence": False, "physics_evaluations_started": False}


def main():
    manifest = policy.verified(policy.read(HERE / "manifest.json"),
                               "candidate-representative-retention-v2")
    for name, record in manifest["files"].items():
        path = policy.path(name)
        assert path.stat().st_size == record["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name
    result = checks()
    assert result == policy.read(HERE / "validation.json")
    print(json.dumps({"retained_files": len(manifest["files"]), **result}, indent=2))


if __name__ == "__main__":
    main()
