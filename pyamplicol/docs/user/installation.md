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

## Native libraries and glibc

On Linux, every native library records the glibc versions it needs (its ELF
version requirements, `readelf -V`). The wheels above are built in a
manylinux 2.28 container and need glibc 2.28 or newer, one-loop engine
included. The release native libraries built from a checkout with
`tools/developer/build_loop_native.py` — the Python extension `_rusticol.so`
and the loop SDK's shared library `librusticol_loop_sdk.so` — may need more:
they can require `GLIBC_ABI_GNU2_TLS` (TLS descriptors). glibc 2.43 and newer
define that version, and so do glibc 2.42 builds that include the
stable-branch update adding it; the glibc 2.42 release itself does not. Such a
library needs **glibc 2.42 with that update, or glibc 2.43 or newer**. Check the
running glibc with `ldd --version` or
`python -c "import os; print(os.confstr('CS_GNU_LIBC_VERSION'))"`, and whether
it defines the version with
`readelf -V /path/to/libc.so.6 | grep GLIBC_ABI_GNU2_TLS` (`ldd` of any
program shows the `libc.so.6` it uses).

pyAmpliCol checks this when it loads the extension. Before
`pyamplicol._rusticol` is opened, it compares the newest glibc release the file
requires (2.42 for `GLIBC_ABI_GNU2_TLS`) with the running one. An older glibc
fails cleanly at load time with `pyamplicol.NativePlatformError` (a
`DependencyError`) instead of the dynamic loader's raw message:

```text
pyamplicol.api.errors.NativePlatformError: Native library
.../pyamplicol/_rusticol.so requires glibc >= 2.42 (it needs the ELF version
GLIBC_ABI_GNU2_TLS, which glibc 2.43 and newer define, as do glibc 2.42 builds
with the stable-branch update that added it; the glibc 2.42 release does not),
but this process runs glibc 2.40. Run it on a Linux system or container with
glibc 2.43 or newer, or an updated glibc 2.42 build (`ldd --version` shows the
version), or build pyAmpliCol's native libraries from source on this host so
that they link against its glibc (see 'Native libraries and glibc' in
docs/user/installation.md).
```

A release number cannot tell whether a glibc 2.42 build includes the update:
on one without it, the dynamic loader refuses the library in its version check,
before any of its code runs, and that refusal becomes the same
`NativePlatformError`; its message says that this process runs glibc 2.42,
"which does not define it", asks to update glibc, and ends with the loader's
text.

The error's attributes carry the same facts: `library`, `required` (the oldest
glibc release that can provide the version, `"2.42"`), `found` (the running
glibc) and `version_name`. The floor is read from the installed library
itself, so a library built against an older glibc (a wheel, or a source build
on that host) is accepted wherever it can load. A C library that is not glibc
(for example musl) is named as such. If the loader still reports a missing
glibc version, that error is translated into the same `NativePlatformError`
with the loader's text appended; when the library it names is a dependency's
(for instance a locally built Symbolica), the remedy is a build of that
library for this glibc.

C, C++ and Fortran clients of `librusticol_loop_sdk.so` do not see
`NativePlatformError`, which belongs to the Python package: the dynamic loader
stops them before `main` with
`` version `GLIBC_ABI_GNU2_TLS' not found (required by .../librusticol_loop_sdk.so) ``.
C and C++ clients can define `RUSTICOL_VIRTUAL_SHARED` to stop at compile time
instead (release numbers only). An executable that links the static archive
does not carry the floor; a shared object that links it keeps the TLS
descriptors and should be linked on a glibc that defines `GLIBC_ABI_GNU2_TLS`
(see "glibc floor (Linux)" in the loop SDK's
`rust/loop-reduction/sdk/README.md`).

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
git clone --branch v1.0.1 --depth 1 \
  https://github.com/MadGraphTeam/pyamplicol.git
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

For development from a checkout, `just dev-install` creates a repository-managed
environment with published Python dependencies from `pyproject.toml`, including
Symbolica 3.0.0 and ufo-model-loader 1.0.0, and the release-mode native build with
published SymJIT 2.26.4. It does not use an editable installation. Building a new
release wheel requires a clean Git checkpoint and complete release assets;
published-dependency mode does not relax these publication guards.

```console
git clone https://github.com/MadGraphTeam/pyamplicol.git
cd pyamplicol
just dev-install
PYTHON=.venv/bin/python just dev-test
```

Normal setup uses published PyPI dependencies. For dependency development,
`--loader-wheel PATH` explicitly selects a local loader wheel; the installer
does not search for one automatically.

The native build can take several minutes. For a dirty development checkout, add
`--wheel-directory PATH` to reuse an already-built compatible release wheel,
or use `--dependencies-only` to install dependencies without building or
staging pyAmpliCol, leaving the existing native runtime untouched.
`--no-build` skips both runtime dependency installation and the project
build/staging; developer tools and explicitly requested optional references
may still be set up. `just dev-test` includes a fresh release build and therefore
also requires a clean checkpoint; dirty edits can use focused Python tests
against the staged compatible native runtime.

On Nix or NixOS:

```console
nix develop
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The Nix shell supplies Python, Rust, C/C++, Fortran, build libraries, and the
documentation/PDF tools. `just dev-install` prepares the published-dependency
environment in `.venv` and builds and stages the project in release mode.

The original AmpliCol and Reference FFT repositories are optional profiling
inputs and are omitted by default. Request either or both explicitly:

```console
just dev-install --with-legacy-amplicol --with-reference-fft
```

Either opt-in also installs the `fft-profiling` Python extra into `.venv`, so
the profiling driver and PDF renderer are ready without reinstalling the
project in editable mode.

### Historical candidate mode (opt-in)

The old pinned-source dependency machinery is retained behind `--candidate`,
but its pinned APIs are incompatible with the current tensor implementation,
even when the dependencies report the same version numbers. It needs updated,
coherent upstream pins before use and is not a working current-source setup
route. Its candidate wheels are non-publishable. Use the published dependency
lane above; ordinary `just dev-test` uses release mode.

## Symbolica licensing

Generation and Python arbitrary-precision execution use Symbolica. The native
binary64 runtime embedded in generated artifacts does not import Symbolica.
pyAmpliCol attempts to register its package key at startup, including in
spawned workers. A valid package key covers its Symbolica operations without a
personal license. If registration fails synchronously, the existing personal
license and restricted-mode paths remain available; restricted mode clamps
generation to one worker and one Symbolica core.

For details and the built-in personal license-request helpers, see
[Symbolica and Licensing](symbolica-and-licensing.md).

## Common installation issues

### `No matching distribution found`

Check your Python version and platform:

```console
python -VV
python -c 'import platform; print(platform.system(), platform.machine())'
```

If no wheel matches, use the tagged source installation above.

### `NativePlatformError`: a native library needs a newer glibc

The native libraries need a glibc version this system does not provide: an
older glibc release, or (message "which does not define it") a glibc 2.42
build without the stable-branch update that defines `GLIBC_ABI_GNU2_TLS` (see
[Native libraries and glibc](#native-libraries-and-glibc)). Update glibc or use
a system or container with glibc 2.43 or newer (or an updated 2.42), or build
the native libraries from source on this host.

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

The local candidate no longer matches the native build inputs. The historical
candidate pins are also incompatible with the current tensor implementation;
rebuilding them is not a supported recovery route. Switch the same checkout to
the default published-dependency environment:

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
