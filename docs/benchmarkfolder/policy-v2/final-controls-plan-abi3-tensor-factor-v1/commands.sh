#!/usr/bin/env bash
# HELD: root coordinates corrected generation, final proof and CPU/MP availability.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
for control_set in hh-complex mixed-legacy; do
  control_name="controls-candidate-abi3-tensor-factor-v1-${control_set}-20261020"
  control_report="docs/benchmarkfolder/policy-v2/control-runs/${control_name}"
  python tools/ninjars_benchmark_references_v2.py \
    --timing-spec "docs/benchmarkfolder/policy-v2/final-controls-plan-abi3-tensor-factor-v1/specs/${control_set}.json"
  python tools/ninjars_benchmark_policy_v2.py bind \
    --spec "${control_report}/binding-spec.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "${control_report}/role-bindings.json"
  python tools/ninjars_benchmark_policy_v2.py runtime-bind \
    --bindings "${control_report}/role-bindings.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-tensor-factor-v1/${control_name}.json"
done
python -B docs/benchmarkfolder/policy-v2/final-controls-plan-abi3-tensor-factor-v1/compare.py \
  --output docs/benchmarkfolder/policy-v2/final-controls-plan-abi3-tensor-factor-v1/comparison.json
