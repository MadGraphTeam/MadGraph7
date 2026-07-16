# Architecture Decisions

## Repository Identity

The canonical case-sensitive repository directory is `pyamplicol`. Historical
guidance that spells the nested directory `pyAmpliCol` refers to the same
macOS checkout but must not be encoded in scripts, manifests, or CI.

Every imported file is sourced from a Git object named by
`SOURCE_BASELINE.toml`. Working-tree copies are ineligible.

## Completion States

**Candidate-complete** means all source, API, physics, native SDK, wheel,
sdist, and isolated-install tests pass against immutable local candidate
dependencies. Candidate artifacts clearly record those revisions and cannot
be published.

**Release-complete** additionally requires exact compatible versions on PyPI
and crates.io, verified Symbolica serialization/runtime compatibility, all
license/provenance gates, and the full supported-platform wheel matrix.

The implementation may be candidate-complete while the strict release gate is
closed solely by an unavailable upstream release.

## Developer-Only Legacy Reference

`--without-legacy-amplicol` belongs to
`dependencies/install_dependencies.py` only. It is never a `pyamplicol` CLI
option and no installed module imports the legacy adapter.

## Reproducibility

Source and unpacked-sdist builds are equivalent when their normalized wheel
contents match after excluding ZIP timestamps and RECORD signatures. Native
binaries must match when built with the same target, Rust toolchain, lockfile,
environment, and reproducible-build flags; cross-toolchain bit identity is
not claimed.

## Supported Platforms

Binary releases target macOS arm64, macOS x86_64, and manylinux x86_64.
Source installation may work elsewhere but is unsupported until the complete
native and physics matrix passes there.
