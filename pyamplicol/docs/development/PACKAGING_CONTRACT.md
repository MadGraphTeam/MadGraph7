<!-- SPDX-License-Identifier: 0BSD -->

# Packaging And Release Contract

This document is normative for standalone pyAmpliCol package builds and
publication.

## Canonical Build

- Distribution: `pyamplicol==0.1.0`, Python 3.11+, license `0BSD`.
- Backend: the in-tree PEP 517 wrapper delegates to pinned Maturin.
- Python extension: `pyamplicol._rusticol`, built from `rusticol-python` with
  `abi3-py311`.
- Native SDK: `rusticol-capi` links the Python-independent Rusticol core and
  must contain no PyO3, NumPy, Python symbols, or Python linker dependency.
- Version source: root Cargo workspace metadata. Installed Python obtains the
  version from distribution metadata.
- Rust resolution: the committed `Cargo.lock` is authoritative.

Build hooks create a temporary allowlisted source overlay, use an external
`CARGO_TARGET_DIR`, and leave the source tree or unpacked source distribution
unchanged.

## Dependency Modes

Release mode reads `dependencies/release-lock.toml` and accepts only exact
published package/crate versions and compatible ABI metadata. Git, path,
editable, and candidate dependencies are forbidden.

Candidate mode is available only from a full source checkout. It reads the
repository-only contributor contract and produces explicitly non-publishable
artifacts. Contributor dependency setup and local candidate inputs are excluded
from release wheels and source distributions.

Release and candidate Cargo resolution are physically separate. A release
command must fail closed when exact published inputs are unavailable; it must
never fall back to contributor state.

## Wheel Contents

The wheel contains:

- the typed Python package and one native extension;
- model assets with their provenance and content manifest;
- packaged examples and Python/Rust/C++/Fortran API templates;
- direct-JIT f64 self-test data;
- license and third-party notice files;
- the target-specific Rusticol C ABI archive, C/C++ headers, Fortran module
  source, SDK configuration, and relative link metadata.

The installed SDK layout is:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  fortran/rusticol.f90
  lib/librusticol_capi.a
  config.py
  metadata.json
  link.json
```

The backend captures one static archive from Cargo JSON messages, validates the
complete C ABI, rejects non-relocatable link inputs, scans for Python-family
symbols, and records only target-appropriate system libraries/frameworks.
`rusticol-config` exposes typed C/C++ flags, Fortran source, Rust linker flags,
and JSON metadata without a machine-specific build path.

Wheel audits permit only the `pyamplicol/` package, one `.dist-info/` directory,
and an optional platform repair directory. Standard wheel `RECORD` protects
installed-file integrity. Model and process-artifact digests remain scoped to
the package features that verify those payloads against accidental mutation.

## Source Distribution

One retained source distribution is the source of all release wheels. It must
build with `python -m pip install .` using published dependencies and contain:

- Python/Rust/build sources and lockfiles;
- schemas, tests, examples, docs, and release tooling;
- the README, licenses, notices, and TeX report sources.

It excludes dependency checkouts, contributor setup, local candidate inputs,
generated process outputs, build products, caches, and local environments.

`just sdist`, `just wheel-from-sdist`, and the PEP 517 hooks exercise the same
backend. The sdist audit checks required members and forbidden source-checkout
inputs before a wheel is retained.

## Validation Matrix

Release targets are:

- macOS 11 arm64, `cp311-abi3`;
- macOS 11 x86_64, `cp311-abi3`;
- manylinux 2.28 x86_64, `cp311-abi3`.

Each wheel is installed on CPython 3.11 and 3.14 and tested without Rust in the
consumer environment. Validation includes:

- `twine check` and platform/native-dependency audits;
- import, CLI, direct-JIT f64 self-test, and packaged-example checks;
- Python total/resolved runtime behavior with Symbolica imports blocked;
- C++17, Fortran 2008, and generated Rust f64 driver compilation/execution;
- model-parameter card and direct UFO-parameter override behavior;
- the independent physics oracle and required regression gates.

Local entry points are:

```console
just source-gate
just test-deployment
just release-artifacts
just publish-dry-run
```

`publish-dry-run` validates ordinary package files and prints the upload command
without uploading anything.

## Publishing

The release inventory is exactly one source distribution and one wheel per
supported target. Publication never rebuilds these files.

The validated-artifact workflow is manually dispatched with read-only source
permissions. A separate manual publishing workflow downloads the successful
validated inventory, verifies the expected platforms and non-candidate version,
and publishes through a protected TestPyPI or PyPI environment using OIDC
Trusted Publishing. Only that final job receives `id-token: write`.

Publication remains blocked until the release dependency contract marks the
Symbolica/SymJIT combination compatible and every target passes the complete
installed-package matrix.
