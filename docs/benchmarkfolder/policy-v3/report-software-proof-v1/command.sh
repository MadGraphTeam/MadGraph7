#!/usr/bin/env bash
# Software only. Run when root has assigned the CPU4 software window.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export RAYON_NUM_THREADS=1
report_software_run=$(mktemp -d "$PWD/.benchmark-work/report-v3-reproduction.XXXXXX")
export NINJARS_REPORT_V3_SOFTWARE_ARTIFACTS="$report_software_run/rendered"
taskset -c 4 python -B -m unittest tests.unit_tests.loop.test_ninjars_report_v3 -v > "$report_software_run/stdout.log" 2> "$report_software_run/stderr.log"
