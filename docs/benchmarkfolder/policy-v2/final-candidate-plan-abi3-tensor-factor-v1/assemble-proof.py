#!/usr/bin/env python3
"""Seal representative proof only after all held candidate runs are complete."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy

V2 = ROOT / "docs/benchmarkfolder/policy-v2"
legacy = policy.verified(policy.read(V2 / "legacy-coverage-status.json"),
                         "legacy-representative-coverage-status-v2")
bindings = [policy.check_file(item) for item in legacy["bindings"]]
for case in ("ggh", "mixed", "forced-qp", "ninja-near-singular"):
    bindings.append(V2 / "runtime-bindings-abi3-tensor-factor-v1" /
                    f"representative-candidate-abi3-tensor-factor-v1-{case}-20261020.json")
output = V2 / "representative-proof-abi3-tensor-factor-v1.json"
if output.exists():
    raise RuntimeError("Preserve existing proof; use a new versioned namespace")
proof = policy.create_proof(V2 / "policy-runtime-groups.json", bindings, output)
print(proof["identity_sha256"])
