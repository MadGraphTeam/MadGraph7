#!/usr/bin/env bash
# Software tests only: no actual numerical campaign or benchmark execution.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 4 python -B -m unittest \
  tests.unit_tests.loop.test_ninjars_benchmark_policy_v3 \
  tests.unit_tests.loop.test_ninjars_benchmark_pair_v3 \
  tests.unit_tests.loop.test_ninjars_report_v3 -v
