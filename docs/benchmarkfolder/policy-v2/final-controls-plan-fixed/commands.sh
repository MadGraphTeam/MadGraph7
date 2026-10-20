#!/usr/bin/env bash
# HELD: root coordinates corrected generation, final proof and CPU/MP availability.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
for control_set in hh-complex mixed-legacy; do
  control_name="controls-candidate-abi3-terminal-fixed-${control_set}-20261020"
  control_report="docs/benchmarkfolder/policy-v2/control-runs/${control_name}"
  python tools/ninjars_benchmark_references_v2.py \
    --timing-spec "docs/benchmarkfolder/policy-v2/final-controls-plan-fixed/specs/${control_set}.json"
  python tools/ninjars_benchmark_policy_v2.py bind \
    --spec "${control_report}/binding-spec.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "${control_report}/role-bindings.json"
  python tools/ninjars_benchmark_policy_v2.py runtime-bind \
    --bindings "${control_report}/role-bindings.json" \
    --correction docs/benchmarkfolder/policy-v2/model-correction.json \
    --output "docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-terminal-fixed/${control_name}.json"
done
python tools/ninjars_benchmark_references_v2.py \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --config docs/benchmarkfolder/policy-v2/controls.json \
  --cases control-gg-hh,control-complex \
  --run-id controls-reference-hh-complex-20261020 \
  --cpu 4 --backends 1,6,7 --representative
python tools/ninjars_benchmark_policy_v2.py runtime-bind \
  --bindings docs/benchmarkfolder/policy-v2/reference-runs/controls-reference-hh-complex-20261020/role-bindings.json \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --output docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-terminal-fixed/controls-reference-hh-complex-20261020.json
python -B docs/benchmarkfolder/policy-v2/final-controls-plan-fixed/compare.py \
  --output docs/benchmarkfolder/policy-v2/final-controls-plan-fixed/comparison.json
