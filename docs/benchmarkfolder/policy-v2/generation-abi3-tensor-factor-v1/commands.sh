#!/usr/bin/env bash
# HELD: root must authorize this phase after the committed source and dev NLO validation pass.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
test "$(git branch --show-current)" = madloop_ninjars
# Two generation jobs inherit only these CPUs; CPU207 remains reserved for references.
taskset -c "${GENERATION_CPUS:-7,8}" python -B docs/benchmarkfolder/policy-v2/generation-abi3-tensor-factor-v1/regenerate.py
python -B docs/benchmarkfolder/policy-v2/generation-abi3-tensor-factor-v1/summarize.py
python -B docs/benchmarkfolder/policy-v2/final-runtime-audit-abi3-tensor-factor-v1/audit.py \
  --output docs/benchmarkfolder/policy-v2/final-runtime-audit-abi3-tensor-factor-v1/inventory.json
python -B docs/benchmarkfolder/policy-v2/final-runtime-audit-abi3-tensor-factor-v1/check-reuse.py
