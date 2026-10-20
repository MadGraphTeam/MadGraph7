#!/usr/bin/env bash
# HELD: root must authorize this phase after the committed source and dev NLO validation pass.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
test "$(git branch --show-current)" = madloop_ninjars
# Requires the complete paired timing cohort and all8 corrected references.
python tools/ninjars_benchmark_policy_v2.py publish \
  --policy docs/benchmarkfolder/policy-v2/policy-runtime-groups.json \
  --proof docs/benchmarkfolder/policy-v2/representative-proof-abi3-tensor-factor-v1.json \
  --bindings docs/benchmarkfolder/policy-v2/runtime-bindings-abi3-tensor-factor-v1-24h/core-abi3-tensor-factor-v1-24h-20261020.json \
    docs/benchmarkfolder/policy-v2/runtime-bindings-core-24h-20261020/corrected-original-core-v2-24h-20261020.json \
  --output docs/benchmarkfolder/policy-v2/datasets/core-abi3-tensor-factor-v1-24h-20261020/dataset.json
