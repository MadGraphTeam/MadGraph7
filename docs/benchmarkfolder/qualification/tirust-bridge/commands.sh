#!/usr/bin/env bash
# Documented replay from the MadGraph repository root. Local stage paths are
# deliberate: the checked installer applies the patch to copied pinned sources.
set -euo pipefail
source tools/tirust_env.sh

# The test manifest is vendor/tirust/Cargo.toml with only iregi-rust replaced by
# path="$PWD/.tirust-build/pole-development/crates/iregi-rust". Its transitive
# oneloop and Symbolica git pins are unchanged. The staged TIRust source is the
# pinned checkout with vendor/tirust/patches/0001-direct-poles-only.patch applied
# and vendor/tirust/pole-kernels copied to its madloop-pole-kernels directory.
cargo test --manifest-path .tirust-build/bridge-poles/Cargo.toml --locked --offline \
  --features poles-only -- --test-threads=1

mkdir -p .tirust-build/bridge-qualification .tirust-build/bridge-smoke
cp .tirust-build/cargo-build/debug/deps/libmadloop_tirust.so \
  .tirust-build/bridge-qualification/libmadloop_tirust.so
python vendor/tirust/check_fixtures.py \
  .tirust-build/bridge-qualification/libmadloop_tirust.so \
  ../references/TIRust/crates/iregi-rust/tests/fixtures/gate_passed_lanes.json \
  --dp-limit 1e-5 --qp
python vendor/tirust/check_fixtures.py \
  .tirust-build/bridge-qualification/libmadloop_tirust.so \
  ../references/TIRust/crates/iregi-rust/tests/fixtures/method_choice_lanes.json --qp

"$FC" -O3 -fPIC -J .tirust-build/bridge-smoke -c vendor/tirust/mtirust.f90 \
  -o .tirust-build/bridge-smoke/mtirust.o
"$FC" -O3 -x f95 -I .tirust-build/bridge-smoke -c - \
  -o .tirust-build/bridge-smoke/smoke.o < vendor/tirust/smoke.f90
"$FC" .tirust-build/bridge-smoke/smoke.o .tirust-build/bridge-smoke/mtirust.o \
  -lgcc_s .tirust-build/bridge-qualification/libmadloop_tirust.so \
  -Wl,-rpath,"$PWD/.tirust-build/bridge-qualification" \
  -o .tirust-build/bridge-qualification/smoke
.tirust-build/bridge-qualification/smoke --require-poles
nm -D --defined-only .tirust-build/bridge-qualification/libmadloop_tirust.so
