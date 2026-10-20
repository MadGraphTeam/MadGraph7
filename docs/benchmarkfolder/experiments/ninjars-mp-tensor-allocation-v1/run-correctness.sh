#!/usr/bin/env bash
set -euo pipefail
test "$PWD" = /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python3 docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/run.py unit --name unit-2
python3 docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/run.py golden --name golden-1
python3 docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/run.py capture-all --name capture-all-1
python3 docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/summarize.py
