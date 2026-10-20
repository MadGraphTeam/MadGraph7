#!/usr/bin/env bash
# Prepared command only: numerical launch is held for root coordination.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
python tools/ninjars_benchmark_references_v2.py \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --config docs/benchmarkfolder/policy-v2/reference-collection-plan/cases.json \
  --cases all \
  --run-id corrected-original-core-v2-20261020 \
  --cpu 207 \
  --backends 1,6,7
