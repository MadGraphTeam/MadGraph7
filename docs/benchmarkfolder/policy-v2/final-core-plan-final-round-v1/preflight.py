#!/usr/bin/env python3
"""Fail closed until final runtime proof and complete legacy references exist."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def preflight(proof_path, reference_path):
    policy_path = ROOT / "docs/benchmarkfolder/policy-v2/policy-runtime-groups.json"
    frozen, correction, config = policy.validate_policy(policy_path)
    proof = policy.validate_proof(proof_path, policy_path)
    reference = policy.checked_bindings(reference_path, correction, live=True)
    protocol, rows = workers.checked_series(policy.path(reference["report"]), minimum=2)
    policy.validate_adapter(protocol, frozen)
    policy.validate_proven_runtimes(proof, reference)
    required = {(case["id"], "baseline", backend, repetition)
                for case in config["cases"] for backend in (1, 6, 7)
                for repetition in (0, 1)}
    if set(rows) != required or protocol["configuration"] != config:
        raise RuntimeError("Corrected references lack exact complete eight-case coverage")
    if any(item["role"] != "original_helper_corrected" for item in reference["targets"].values()):
        raise RuntimeError("Wrong helper reference source role")
    if protocol["cpu"] != 207:
        raise RuntimeError("Helper references must use final campaign CPU 207")
    return {"schema": "final-core-preflight-v1", "passed": True,
            "reference_records": len(rows), "required_cases": [c["id"] for c in config["cases"]],
            "proof_identity_sha256": proof["identity_sha256"],
            "reference_binding_identity_sha256": reference["identity_sha256"],
            "numerical_jobs_started": [], "final_campaign_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.proof, args.references), indent=2))
