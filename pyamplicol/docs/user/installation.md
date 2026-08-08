---
title: "Installation"
nav_order: 2
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Installation

pyAmpliCol is distributed as a Python package with a Rust-backed runtime. For
most users, installation is a normal `pip` operation: the published wheel
already contains the Python extension and the Rusticol native SDK.

> **Recommended path:** use Python 3.11 or newer in a fresh virtual
> environment and install from PyPI.

## Install from PyPI

### macOS and Linux

```console
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyamplicol
```

Verify the installation without generating a process:

```console
python -c 'import pyamplicol; print(pyamplicol.__version__)'
pyamplicol doctor
pyamplicol examples list
```

`pyamplicol doctor` reports the package, Python, runtime, platform, and
Symbolica environment that pyAmpliCol can see. `examples list` confirms that
the packaged run cards and data files are available.

## Supported binary-wheel platforms

The current release provides `cp311-abi3` wheels for Python 3.11 and newer on:

| Platform | Architecture | Minimum deployment target |
| --- | --- | --- |
| macOS | Apple silicon (`arm64`) | macOS 11 |
| macOS | Intel (`x86_64`) | macOS 11 |
| Linux | `x86_64` | manylinux 2.28 |

Wheel users do **not** need a Rust compiler. pyAmpliCol does not depend on
LHAPDF.

The generated C11, C++17, Fortran 2008, and Rust 2021 examples compile against
the wheel-owned Rusticol SDK. Install the compiler only for the language you
intend to use:

- C/C++ compiler for C11 or C++17 consumers;
- Fortran compiler for Fortran 2008 consumers;
- Rust compiler for the standalone Rust driver.

See [Native APIs](native-apis.md) for the generated driver workflow.

## Create an editable examples workspace

The examples are package resources. Copy them before editing or running them:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

Paths inside the supplied TOML cards are relative to the card, so the copied
workspace can be moved. Keep the environment containing pyAmpliCol activated,
or invoke its executable by absolute path.

Continue with [Quick Start](quick-start.md).

## Install a tagged source release

Use a tagged snapshot when your platform has no compatible wheel or when you
need a source build:

```console
git clone --branch v0.1.4 --depth 1 \
  https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
python -m pip install .
```

A source build requires:

- Python 3.11 or newer;
- Rust 1.89 or newer;
- a C/C++ toolchain;
- network access to the release-locked Python, Cargo, and SymJIT inputs.

The repository-pinned CI toolchain may be newer than the minimum supported
Rust version. A Fortran compiler remains optional unless you build a Fortran
consumer.

To build from the source distribution published on PyPI:

```console
python -m pip download --no-binary pyamplicol pyamplicol
tar -xf pyamplicol-*.tar.gz
cd pyamplicol-*/
python -m pip install .
```

## Contributor installation

For development from a checkout, use the repository-managed candidate
environment rather than an editable installation:

```console
git clone https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The first native build can take several minutes. Later invocations reuse the
workspace-local Cargo cache.

On Nix or NixOS:

```console
nix develop
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The Nix shell supplies Python, Rust, C/C++, Fortran, build libraries, and the
documentation/PDF tools. `just dev-install` installs the pinned Python and
native inputs into `.venv`.

> Candidate wheels are deliberately marked non-publishable. Published builds
> use the release dependency lock and CI workflow instead.

## Symbolica licensing

Generation and Python arbitrary-precision execution use Symbolica. The native
binary64 runtime embedded in generated artifacts does not import Symbolica.
Restricted Symbolica mode may clamp generation resources; pyAmpliCol records
the requested and effective settings separately.

For licensing details and the built-in license-request helpers, see
[Symbolica and Licensing](symbolica-and-licensing.md).

## Common installation issues

### `No matching distribution found`

Check your Python version and platform:

```console
python -VV
python -c 'import platform; print(platform.system(), platform.machine())'
```

If no wheel matches, use the tagged source installation above.

### `pyamplicol: command not found`

Activate the environment where the package was installed:

```console
. .venv/bin/activate
python -m pyamplicol --help
```

### Generated native driver cannot find `rusticol-config`

The SDK discovery command belongs to the installed wheel. Activate that wheel's
environment before invoking `make`, or set `RUSTICOL_CONFIG` to its executable:

```console
export RUSTICOL_CONFIG="$(python -c 'import sys; print(sys.prefix + "/bin/rusticol-config")')"
make -C artifacts/pp_zjj/API/c run
```

### A contributor import says the candidate wheel is stale

The native build inputs changed after the local candidate was staged. Refresh
the same checkout once:

```console
just dev-install
```

Do not work around this by mixing an extension from a different checkout.

## Next steps

- [Quick Start](quick-start.md) — generate, inspect, evaluate, and profile a process.
- [Configuration](configuration.md) — run cards, overrides, color modes, and evaluators.
- [Models and Processes](models-and-processes.md) — built-in, JSON, UFO, and prepared models.
- [Troubleshooting](troubleshooting.md) — focused solutions for runtime and generation failures.

The exact build and publication boundary is recorded in
[Release and Support](release-and-support.md).
