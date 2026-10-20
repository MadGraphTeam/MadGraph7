#!/usr/bin/env bash
# HELD: root must authorize this phase after the committed source and dev NLO validation pass.
set -euo pipefail
cd /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
test "$(git branch --show-current)" = madloop_ninjars
# Preparation/building may be scheduled separately from the Symbolica runtime slot.
# Run these phases individually only after root coordinates their resources.
case "${1:-}" in
  prepare) python -B docs/benchmarkfolder/policy-v2/final-variant-controls-abi3-tensor-factor-v1/execute.py prepare ;;
  run) python -B docs/benchmarkfolder/policy-v2/final-variant-controls-abi3-tensor-factor-v1/execute.py run --cpu 5 ;;
  compare) python -B docs/benchmarkfolder/policy-v2/final-variant-controls-abi3-tensor-factor-v1/execute.py compare-rotation \
    --binding docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-tensor-factor-v1/controls-candidate-abi3-tensor-factor-v1-hh-complex-20261020.json ;;
  *) printf '%s\n' 'Choose prepare, run, or compare after root coordinates that phase.' >&2; exit 2 ;;
esac
