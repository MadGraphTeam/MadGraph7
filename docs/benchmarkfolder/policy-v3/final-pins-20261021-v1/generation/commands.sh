#!/usr/bin/env bash
# Root must first supply the exact new source/runtime freeze and CPU release.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
test "$(git branch --show-current)" = madloop_ninjars
: "${GENERATION_CPUS:?Set the root-authorized generation CPU list}"
campaign_dir=docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation
taskset -c "$GENERATION_CPUS" python -B "$campaign_dir/regenerate.py" \
  --freeze "$campaign_dir/freeze.json" --jobs 2
python -B "$campaign_dir/summarize.py"
python -B "$campaign_dir/audit_core.py" --generation "$campaign_dir" \
  --output "$campaign_dir/runtime-inventory.json"
