---
title: "Development Documentation"
nav_order: 7
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Development Documentation

This directory contains the stable contracts and decisions that supplement the
source and tests:

- [`API_CONTRACT.md`](API_CONTRACT.md)
- [`CONFIG_CONTRACT.md`](CONFIG_CONTRACT.md)
- [`PACKAGING_CONTRACT.md`](PACKAGING_CONTRACT.md)
- [`PHYSICS_EXTRACTION_CONTRACT.md`](PHYSICS_EXTRACTION_CONTRACT.md)
- [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md)
- [`ON_THE_FLY_MODE_ARCHITECTURE.md`](ON_THE_FLY_MODE_ARCHITECTURE.md)
- [`MODEL_ASSET_PROVENANCE.md`](MODEL_ASSET_PROVENANCE.md) and
  [`model-assets/`](model-assets/README.md)

Historical feature plans, audit trails, milestone ledgers, and duplicated
machine-readable summaries are deliberately not maintained here. Current
behavior is defined by the implementation, tests, and contracts above.
The on-the-fly architecture page is retained because it records the current
production contract and validation boundary; its historical dispositions are
explicitly labelled as such.

## Linux source-test wheels

The contributor installer and `just source-runtime` build a host-compatible
Linux wheel for local tests, rather than claiming compatibility with an older
manylinux platform. They explicitly pass the backend setting
`--config-setting pyamplicol.host-wheel=true` to `python -m build --wheel`.
Published dependencies are unchanged. Normal release builds still target
`manylinux_2_28` and must run in the appropriate build environment; a host wheel
is not a substitute for the audited release artifact.

## One-loop test suite

Loop tests carry one cost marker: `loop_fast` (unit and saved-control checks),
`loop_physics` (generation and provider-reference comparisons) or `loop_nightly`
(the longest generation and precision controls); generic direct-R2 tests are
additionally `deferred_direct_r2` and are deselected unless named in `-m`. Run
`just test-loop loop_fast` (or another marker expression) after staging the
native runtime, with `SYMBOLICA_LICENSE` set; without it loop tests skip rather
than abort. `tests/conftest.py` resolves external inputs once: UFO models and
MadGraph default to the reference checkout/workspace and are overridden by the
documented `PYAMPLICOL_*` variables, saved PACBIN/artifact controls come from
`PYAMPLICOL_LOOP_SAVED_FIXTURES` (default `<workspace>/fixtures`), and provider
references are the committed `tests/fixtures/loop/*.json`, whose absence fails.

## Optimized loop native builds

For long loop generation/validation runs, build the extension and loop SDK
libraries optimized, rather than with the unoptimized Cargo dev profile:
`tools/developer/build_loop_native.py build NEW_DIR` (Cargo `release`, 16
codegen units, no cross-crate LTO unless `CARGO_PROFILE_RELEASE_*` say
otherwise; digest, revision, profile, `RUSTFLAGS` and whether the tree was dirty
are recorded in `NEW_DIR/build-info.json`), then
`tools/developer/build_loop_native.py stage NEW_DIR` into a checkout whose
native build-input digest matches. Cargo builds into a private target directory
(`--target-dir`; by default `cargo-loop-native-<profile>` next to a set
`CARGO_TARGET_DIR`, never the shared one itself), so concurrent dev builds of
other checkouts neither block the build nor replace its `release/` outputs.
