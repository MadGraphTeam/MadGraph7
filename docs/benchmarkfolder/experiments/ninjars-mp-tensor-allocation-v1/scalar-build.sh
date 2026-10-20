#!/usr/bin/env bash
set -euo pipefail
test "$PWD" = /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
taskset -c 212 rustc --edition=2024 -O \
  docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/scalar-diagnostic.rs \
  --extern ninjars=.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build/release/deps/libninjars-029811cf3d12c72d.rlib \
  -L dependency=.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build/release/deps \
  -L native=.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build/release/build/gmp-mpfr-sys-fbaf675a2d7bb3d7/out/lib \
  -o .benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/scalar-diagnostic
