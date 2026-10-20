#!/usr/bin/env bash
set -euo pipefail
test "$PWD" = /shared/external/madloop_ninjars/madgraph7_madloop_ninjars
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export CARGO_TARGET_DIR="$PWD/.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/target"
export CARGO_BUILD_BUILD_DIR="$PWD/.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build"
export CARGO_BUILD_JOBS=1
taskset -c 212 cargo test --manifest-path .benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/bridge/Cargo.toml --locked --offline --release --lib --features fenv-test-support --no-run --message-format=json
