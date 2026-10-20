#!/usr/bin/env bash
# HELD: root coordinates final proof, generation, references and quiet CPU 207.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "tools")
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

root = Path("docs/benchmarkfolder/policy-v2")
policy_path = root / "policy-runtime-groups.json"
_, correction, config = policy.validate_policy(policy_path)
proof = policy.validate_proof(root / "representative-proof-abi3-tensor-factor-v1.json", policy_path)
reference = policy.checked_bindings(
    root / "runtime-bindings-core-24h-20261020/corrected-original-core-v2-24h-20261020.json",
    correction)
protocol, rows = workers.checked_series(policy.path(reference["report"]), minimum=2)
policy.validate_adapter(protocol, policy.read(policy_path))
policy.validate_proven_runtimes(proof, reference)
required = {(case["id"], "baseline", backend, repetition)
            for case in config["cases"] for backend in (1, 6, 7)
            for repetition in (0, 1)}
if set(rows) != required or protocol["configuration"] != config:
    raise RuntimeError("Corrected references do not cover the exact eight-case core policy")
if any(item["role"] != "original_helper_corrected" for item in reference["targets"].values()):
    raise RuntimeError("Reference binding has a different source role")
print("Refreshed proof and complete corrected-original references validated.", flush=True)
PY
python tools/ninjars_benchmark_references_v2.py \
  --timing-spec docs/benchmarkfolder/policy-v2/final-core-plan-abi3-tensor-factor-v1-24h/spec.json
python tools/ninjars_benchmark_policy_v2.py bind \
  --spec docs/benchmarkfolder/policy-v2/timing-runs/core-abi3-tensor-factor-v1-24h-20261020/binding-spec.json \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --output docs/benchmarkfolder/policy-v2/timing-runs/core-abi3-tensor-factor-v1-24h-20261020/role-bindings.json
python tools/ninjars_benchmark_policy_v2.py runtime-bind \
  --bindings docs/benchmarkfolder/policy-v2/timing-runs/core-abi3-tensor-factor-v1-24h-20261020/role-bindings.json \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --output docs/benchmarkfolder/policy-v2/runtime-bindings-abi3-tensor-factor-v1-24h/core-abi3-tensor-factor-v1-24h-20261020.json
