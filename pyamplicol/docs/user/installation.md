<!-- SPDX-License-Identifier: 0BSD -->

# Installation

## Binary Wheel

After `0.1.0` is uploaded to PyPI, install it with:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The validated `cp311-abi3` wheels target Python 3.11 and newer on macOS arm64,
macOS x86_64, and manylinux x86_64. Wheel users do not need Rust. A C, C++,
Fortran, or Rust compiler is needed only when compiling that language's native
consumer against the included Rusticol SDK. pyAmpliCol has no LHAPDF
dependency.

The immutable `v0.1.0` source tag exists, but `pyamplicol==0.1.0` has not yet
been uploaded to PyPI or TestPyPI. See
[Release Status](release-status.md) before treating a locally built artifact
as a release.

An installed wheel can also populate a new or empty profiling campaign:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
```

`--force` replaces template members without deleting unrelated files already
in the destination; choose a new destination for a clean reset.

Its launcher uses installed resources and runs headlessly when the optional
Ratatui bindings are absent. An original-AmpliCol checkout is needed only when
that reference backend is selected. Record its default while copying with
`--local-amplicol /path/to/clean/complete/checkout`, or provide/override it on
the copied launcher's `run` command with `--original-amplicol PATH`.

## Source Install

```console
git clone --branch v0.1.0 --depth 1 https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
python -m pip install .
```

This runs the same in-tree PEP 517/Maturin backend used for release artifacts
and resolves exact published packages/crates plus SymJIT 2.22.0 from its
official repository at the immutable revision recorded in the release lock. It
requires Python 3.11+, Rust 1.89+ and a C/C++ toolchain. The build checks that
release dependency contract and does not substitute contributor inputs.

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

For contributor development, prepare the isolated managed environment with:

```console
just dev-install
PYTHON=.venv/bin/python just dev-test
```

On Nix or NixOS, the repository includes a developer shell:

```console
nix develop
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The flake provides Python 3.11, Rust 1.89, C/C++ and Fortran compilers, native
build libraries, PDF inspection utilities, and the TeX tools used to render a
fresh installed profiling campaign. It intentionally omits pyAmpliCol's Python
runtime, test, and pinned candidate packages: `just dev-install` installs
those into `.venv` from the repository's contributor locks.

The first native `just dev-install` can take several minutes. Later runs reuse
the workspace-local Cargo cache under `.artifacts/dev-install` and are normally
substantially faster.

The same command stages the candidate wheel's Rust extension and native SDK
beside the Python source. Contributor imports verify a lightweight native-source
build ID and the staged extension hash. If Rust or native build inputs change,
or another extension is found, imports fail with `just dev-install` as the
refresh command. This check is limited to source and candidate builds; normal
published-wheel imports are unchanged.

The flake is a source-checkout contributor tool rather than part of the PyPI
source distribution, because that distribution excludes the candidate
dependency installer and accepts only the release-locked published
packages/crates plus the official immutable SymJIT Git revision.

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
