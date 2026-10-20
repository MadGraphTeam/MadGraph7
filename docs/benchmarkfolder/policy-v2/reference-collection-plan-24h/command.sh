#!/usr/bin/env bash
# Full-eight reference retry; root coordinates launch and CPU 207.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python tools/ninjars_benchmark_references_v2.py \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --config docs/benchmarkfolder/policy-v2/reference-collection-plan-24h/cases.json \
  --cases all \
  --run-id corrected-original-core-v2-24h-20261020 \
  --cpu 207 \
  --backends 1,6,7 \
  --timeout 86400
