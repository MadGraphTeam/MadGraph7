# Packaging And Release Contract

The first independent packaging audit approved bootstrap work but found the
release dependency gate closed. This document is normative for build and
release implementation.

## Canonical Build

- Distribution: `pyamplicol==0.1.0`, Python 3.11+, 0BSD.
- In-tree backend: pinned Maturin 1.14.1 through PEP 517
  `backend-path = ["build_backend"]`.
- Extension: `pyamplicol._rusticol` from `rusticol-python` with
  `abi3-py311`.
- Static SDK: `rusticol-capi` depends on the Python-free core and contains no
  PyO3, NumPy, Python symbols, or Python linker dependency.
- Root Cargo workspace metadata is the only release-version source. Python
  reports installed metadata through `importlib.metadata`.

Build hooks create an allowlisted temporary source overlay, use an external
`CARGO_TARGET_DIR`, and never mutate the checkout or unpacked sdist. Candidate
artifacts use
`0.1.0.dev0+candidate.<12-hex-dependency-lock-digest>`, embed
`publishable: false`, and never enter `dist/`.

## SDK Capture

One JSON-message `cargo rustc` invocation builds `rusticol-capi` and asks
rustc for native static libraries. The backend:

1. selects exactly one static archive from Cargo `compiler-artifact` records;
2. accepts only typed, target-specific allowlisted system libraries and
   frameworks;
3. rejects absolute paths, `-L`, RPATHs, compiler-discovered replacements,
   and cross-target SDK builds;
4. scans the archive for Python, PyO3, and NumPy symbols;
5. generates a deterministic, target-filtered CycloneDX 1.5 dependency SBOM;
6. emits relative `metadata.json` and `link.json`, including archive and SBOM
   hashes;
7. stages generated SDK files through `rusticol-python`'s `OUT_DIR` for
   Maturin's generated-file include support.

The installed layout is:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  fortran/rusticol.f90
  lib/librusticol_capi.a
  sboms/rusticol-capi.cyclonedx.json
  config.py
  metadata.json
  link.json
```

Fortran module files are not shipped because they are compiler-specific.
`rusticol-config` reports typed paths and link arguments; it never emits a
machine-specific build-time path.

Wheel audits accept only the `pyamplicol/` package, its one `.dist-info/`
directory, and an optional repair-tool `.libs/` directory. They verify the
release lock, schema resources, exact PEP 639 license inventory, SDK archive,
and SDK SBOM before deployment testing.

## Artifact Parity

Release wheels are built only from the one retained unpacked sdist outside
the parent workspace. A direct-source same-host wheel is discarded after
normalized path, metadata, `RECORD`, package-resource, SDK-hash, and native
binary comparison.

Release targets are:

- macOS 11 arm64, `cp311-abi3`;
- macOS 11 x86_64, `cp311-abi3`;
- manylinux_2_28 x86_64, `cp311-abi3`.

Each compressed wheel is at most 95 MB and is installed and tested on CPython
3.11 and 3.14 without Rust present.

## Closed Release Gates

Publication remains blocked until:

- published Symbolica Python and Rust artifacts contain every required local
  fix and pass serialization compatibility;
- the Symbolica-selected SymJIT release passes the AArch64 regressions;
- `ufo-model-loader` containing revision
  `9cb4deeae40ddd64184049af07ac1d03ce5f6162` is published;
- every bundled model has complete redistributable license provenance.

Candidate mode may consume locally built exact dependency wheels and patched
crates, but public metadata never contains a Git, path, editable, or parent
dependency.

## Publishing

CI checks candidate builds with read-only permissions and no OIDC. Release
artifacts are manually built from a signed tag on the three release targets.
TestPyPI and production receive the unchanged, previously validated bundle.
Production publishing is `workflow_dispatch` only; only the protected `pypi`
job receives `id-token: write`, performs no checkout or build, and uses PyPI
Trusted Publishing.

Every external CI action is pinned by immutable commit SHA. Linux bootstrap
verifies the pinned `rustup-init` archive checksum, and the final artifact
manifest binds every retained file to the signed source tag and full source
commit before a publishing job can consume it.
