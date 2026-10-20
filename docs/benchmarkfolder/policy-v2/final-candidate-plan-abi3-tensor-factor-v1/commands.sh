#!/usr/bin/env bash
# HELD: run only after root coordinates final-generation/runtime freeze.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
for representative_case in ggh mixed forced-qp ninja-near-singular; do
  representative_name="representative-candidate-abi3-tensor-factor-v1-${representative_case}-20261020"
  representative_report="docs/benchmarkfolder/policy-v2/representative-runs/${representative_name}"
  python tools/ninjars_benchmark_references_v2.py \
    --timing-spec "docs/benchmarkfolder/policy-v2/final-candidate-plan-abi3-tensor-factor-v1/specs/${representative_case}.json"
  python tools/ninjars_benchmark_policy_v2.py bind \
    --spec "${representative_report}/binding-spec.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "${representative_report}/role-bindings.json"
  python tools/ninjars_benchmark_policy_v2.py runtime-bind \
    --bindings "${representative_report}/role-bindings.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "docs/benchmarkfolder/policy-v2/runtime-bindings-abi3-tensor-factor-v1/${representative_name}.json"
done
python -B docs/benchmarkfolder/policy-v2/final-candidate-plan-abi3-tensor-factor-v1/assemble-proof.py
