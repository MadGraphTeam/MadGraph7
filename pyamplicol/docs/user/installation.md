<!-- SPDX-License-Identifier: 0BSD -->

# Installation

## Binary Wheel

After `0.1.0` is published, install the release with:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The planned `cp311-abi3` wheels support Python 3.11 and newer on macOS arm64,
macOS x86_64, and manylinux x86_64. Wheel users do not need Rust. A C++ or
Fortran compiler is needed only when compiling a native consumer against the
included Rusticol SDK.

`pyamplicol==0.1.0` is not published at this milestone. See
[Release Status](release-status.md) before treating a locally built artifact
as a release.

## Source Install

```console
git clone https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
python -m pip install .
```

This runs the same in-tree PEP 517/Maturin backend used for release artifacts
and resolves published dependencies only. It requires Python 3.11+, Rust 1.89+
and a C/C++ toolchain. The build checks the release dependency contract and
fails clearly if a required published version or compatibility state is not
ready; it does not substitute contributor inputs.

An unpacked release source distribution supports the same command:

```console
python -m pip download --no-binary pyamplicol pyamplicol
tar -xf pyamplicol-0.1.0.tar.gz
cd pyamplicol-0.1.0
python -m pip install .
```

The source distribution contains the build backend, locked Rust workspace,
tests, docs, examples, and release tooling required for this build. Candidate
dependency setup is intentionally source-checkout-only and is not distributed.

## Retained Local Wheel

```console
just wheel
python -m pip install dist/pyamplicol-*.whl
```

To select an interpreter and build a matching wheel when necessary:

```console
just install-wheel PYTHON=/path/to/python
```

The wheel owns the Python extension and target-specific static SDK. Running
`rusticol-config` from an unstaged source tree therefore reports that the SDK
is unavailable.

Useful release-equivalent checks are:

```console
just check
just test
just sdist
just wheel-from-sdist
just test-deployment
just publish-dry-run
```

`publish-dry-run` builds and checks ordinary Python package files, performs
platform and clean-install smoke tests, and prints the upload command without
publishing.

## Contributor Environment

While exact release dependencies are gated, prepare the isolated managed
environment with:

```console
just dev-install
PYTHON=.venv/bin/python just dev-test
```

This mode uses pinned candidate dependency inputs and marks its wheels
non-publishable. To omit the optional independent legacy-Fortran oracle:

```console
python dependencies/install_dependencies.py --without-legacy-amplicol
PYTHON=.venv/bin/python PYAMPLICOL_BUILD_MODE=candidate just source-gate
```

Contributor-only dependency setup is excluded from release package files.
Release builds remain governed by `dependencies/release-lock.toml`; candidate
state is not a fallback for `python -m pip install .`, `just wheel`, or
`just test-deployment`.

Use `just --list` for the complete recipe inventory. Validation should run in
the managed environment without an inherited `PYTHONPATH` or an editable
installation from another checkout.
