#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${IN_NIX_SHELL:-}" && -z "${PYAMPLICOL_FFT_NO_NIX:-}" ]]; then
  exec nix develop --quiet --command "$0" "$@"
fi

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

exec "$workspace_root/.venv/bin/python" \
  "$workspace_root/tools/fft_profiling/fft_profiling.py" \
  --output "$workspace_root/fft_profiling" \
  --render \
  "$@"
