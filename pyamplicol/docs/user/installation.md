<!-- SPDX-License-Identifier: 0BSD -->

# Installation

## Released Binary Wheel

After `0.1.0` is published, the normal installation is:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The planned wheels support Python 3.11 and newer through the `cp311-abi3` ABI
on macOS arm64, macOS x86_64, and manylinux x86_64. A binary-wheel install does
not require Rust. `pyamplicol==0.1.0` is not published at this milestone.

## Direct Source Install

```console
git clone https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
python -m pip install .
```

This invokes the same PEP 517 backend used for release artifacts and consumes
published dependencies only. It requires Python 3.11+, Rust 1.89+, and a C/C++
toolchain. Strict release mode currently fails closed while exact dependency
releases, artifact hashes, or platform targets are unverified; it does not
silently use a developer checkout.

## Retained Wheel

```console
just wheel
python -m pip install dist/pyamplicol-*.whl
```

To select an interpreter and let the recipe build a wheel when necessary:

```console
just install-wheel PYTHON=/path/to/python
```

The static Rusticol SDK is staged only into a built wheel. Running
`rusticol-config` directly from an unstaged source tree reports that the SDK is
unavailable.

## Contributor Candidate

```console
just dev-install
PYTHON=.venv/bin/python just dev-test
```

`dev-install` creates the managed `.venv`, checks out immutable candidate
revisions, verifies and applies checked patches, and builds candidate inputs.
Use `--without-legacy-amplicol` when the independent Fortran reference is not
needed. Candidate artifacts record their provenance and are not publishable.

Generation, Python runtime loading, and benchmarking are integrated in
candidate environments. Installed-wheel deployment validation remains
incomplete; check [Release Status](release-status.md) before treating a
candidate wheel as a release artifact.

Useful inventory:

```console
just --list
python dependencies/install_dependencies.py --dry-run --without-legacy-amplicol
```

Do not set `PYTHONPATH` to the parent project or rely on an editable install
when validating release behavior.
