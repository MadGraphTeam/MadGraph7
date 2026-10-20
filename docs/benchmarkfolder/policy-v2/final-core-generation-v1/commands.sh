#!/usr/bin/env bash
# HELD until root supplies final identities and releases generation resources.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
test "$(git branch --show-current)" = madloop_ninjars
: "${GENERATION_CPUS:?Set the coordinated generation CPU list after quiet release}"
freeze_file="${FINAL_CORE_FREEZE_FILE:-docs/benchmarkfolder/policy-v2/final-core-generation-v1/freeze.json}"
test -f "$freeze_file"
taskset -c "$GENERATION_CPUS" python -B \
  docs/benchmarkfolder/policy-v2/final-core-generation-v1/regenerate.py \
  --freeze "$freeze_file" --jobs 2
python -B docs/benchmarkfolder/policy-v2/final-core-generation-v1/summarize.py
python -B docs/benchmarkfolder/policy-v2/final-core-generation-v1/audit_core.py \
  --generation docs/benchmarkfolder/policy-v2/final-core-generation-v1 \
  --output docs/benchmarkfolder/policy-v2/final-core-generation-v1/runtime-inventory.json
