# ABI3 compensated-DP retry production port

This directory records validation of the production port of the sealed
[`t3-dp-retry`](../t3-dp-retry/README.md) experiment. It is integration evidence,
not a final original-Ninja performance acceptance result.

The production bridge exposes a separate explicit compensated-DP call with
lazy state. Ordinary DP retains the original engine and does not allocate that
state. MadLoop retries only a finite normal DP stability failure from backend 8,
once, using the same directions, rotations and threshold. A failed retry still
reaches the existing backend and genuine high-precision rescue path.

Validation passed:

- All 16 Rust tests, including the frozen 480 callback components, broader guarded
  callback controls, exact ordinary-DP behavior, parameter rebinding, malformed
  buffers, and actual 256/384-bit MP convergence.
- All 13 focused compiled Fortran and installer tests after the final template
  correction.
- Private ABI3 installation and installed native DP → explicit compensated DP →
  native DP → actual binary128-input MP smoke. The installed archive contains
  the environment observer and no test-only FENV mutators.
- All 24 mock control-flow scenarios compiled from a fresh actual CLI export of
  the production templates: rotations 0/2, backend 8 alone or followed by Ninja 6,
  and six ordered consecutive-point scenarios. These mock results establish
  state transitions; they are explicitly not genuine MP numerical evidence.

`validation.json` binds production source and private installed artifacts.
`stage-list.json` lists the proposed files for root review. `mock-flow/origin.json`
binds the actual CLI output, templates, recording driver and frozen scenario
harness. The uninstrumented generated sources are in `generated/`.

The first fresh generated compilation exposed a fixed-form formatting error:
manually continued IF conditions confused the Fortran writer's indentation.
The template now supplies those conditions on one line for the writer to wrap.
`failed-template-format/` preserves the failed generated source and build log;
`failure.json` explains the unavailable first generation transcript and retained
follow-up invocation error. No numerical threshold or control-flow decision was
changed by the formatting fix.

Run from the repository root after entering the pinned development environment:

```bash
source .ninjars-build/dev-env.sh
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
export CARGO_TARGET_DIR="$PWD/.ninjars-build/abi3-private/cargo"
export CARGO_BUILD_BUILD_DIR=/common/build-cache/external/f7/d00a3ee9585371
export MAKEFLAGS="SHELL=$(command -v bash)"
cargo test --manifest-path vendor/ninjars/Cargo.toml --locked --release \
  --features fenv-test-support -- --test-threads=1
python -m unittest tests.unit_tests.loop.test_ninjars_fortran \
  tests.unit_tests.various.test_ninjars_install
python bin/madgraph docs/benchmarkfolder/experiments/t3-dp-retry-port/qcd1.mg5
python docs/benchmarkfolder/experiments/t3-dp-retry-port/production_flow.py
```

The CLI command file uses the validated private ABI3 install. For a new machine,
install the current bridge first and adjust that command file's local library
paths. The flow harness refuses to overwrite prior outputs. Preserve existing
records and choose a fresh namespace before rerunning it. Symbolica tests and
the installation smoke must run without another restricted Symbolica process.
The cache paths above are build reuse settings, not numerical inputs.
