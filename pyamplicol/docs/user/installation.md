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

### GMP, MPFR and MPC in the wheels

The wheels carry GMP 6.3.0, MPFR 4.2.2 and MPC 1.4.1 (GNU LGPL version 3 or
later, see `THIRD_PARTY_NOTICES.md`) as separate shared libraries, linked
dynamically and replaceable by any interface-compatible build:

| Platform | Python extension | C SDK |
| --- | --- | --- |
| Linux | `pyamplicol.libs/libgmp-pyamplicol-<hash>.so.10`, `libmpfr-pyamplicol-<hash>.so.6` | `pyamplicol/_sdk/lib/libgmp-pyamplicol.so.10`, `libmpfr-pyamplicol.so.6`, `libmpc-pyamplicol.so.3` |
| macOS | `pyamplicol.dylibs/`: GMP and MPFR (and MPC when the extension links it) under the names the wheel repair gives them (hashed, as on Linux; the extension's load commands name those files) | `pyamplicol/_sdk/lib/libgmp.10.dylib`, `libmpfr.6.dylib`, `libmpc.3.dylib` (`@rpath` install names) |

The wheel repair bundles only the libraries the extension links (on Linux,
GMP and MPFR); the C SDK ships all three because its link line names all
three.

The Linux libraries use their own sonames, so neither the extension nor an SDK
consumer can bind a system `libgmp.so.10` by accident. The C SDK archive
`librusticol_capi.a` contains no copy of these libraries. The wheels ship it
xz-compressed (`pyamplicol/_sdk/lib/librusticol_capi.a.xz`); the first
`rusticol-config` query that needs it unpacks it into the per-user cache
`<cache>/sdk/<version>-<sha256>/` (`<cache>` = `$PYAMPLICOL_CACHE_DIR` if set,
otherwise `~/.cache/pyamplicol` on Linux or `~/Library/Caches/pyamplicol` on
macOS; see [The SDK archive cache](native-apis.md#the-sdk-archive-cache)).
`rusticol-config --libs` (which the generated drivers' Makefiles use) prints
everything a C, C++ or Fortran consumer links against: the unpacked archive,
the three shared libraries by path (dependents first), a run path to their
directory, and the target's system libraries. On Linux, for example:

```text
~/.cache/pyamplicol/sdk/<version>-<sha256>/librusticol_capi.a
<site-packages>/pyamplicol/_sdk/lib/libmpc-pyamplicol.so.3
<site-packages>/pyamplicol/_sdk/lib/libmpfr-pyamplicol.so.6
<site-packages>/pyamplicol/_sdk/lib/libgmp-pyamplicol.so.10
-Wl,-rpath,<site-packages>/pyamplicol/_sdk/lib -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc
```

`rusticol-config --rustflags` gives the same flags as `-C link-arg=...` for a
Rust consumer. A consumer that is moved away from the environment it was
linked in needs the three libraries next to it or on its library path.

## Native libraries and glibc

On Linux, every native library records the glibc symbol versions it needs (its
ELF version requirements, `readelf -V`), and the dynamic loader refuses it on a
glibc that does not define one of them. The wheels above are built in a
manylinux 2.28 container and need glibc 2.28 or newer. The release native
libraries built from a checkout with `tools/developer/build_loop_native.py` —
the Python extension `_rusticol.so` and the loop SDK's shared library
`librusticol_loop_sdk.so` — link against the build host's glibc; built with the
repository's toolchain they need **glibc 2.38 or newer** (`GLIBC_2.38`) and no
glibc ABI marker. The build compiles its C dependencies with the traditional
TLS dialect, so the libraries do not need `GLIBC_ABI_GNU2_TLS` (TLS
descriptors), which only glibc builds carrying that marker define. The
`build-info.json` of a native build lists each library's requirements
(`glibc_version_needs`). Check the running glibc with `ldd --version` or
`python -c "import os; print(os.confstr('CS_GNU_LIBC_VERSION'))"`.

pyAmpliCol checks this before it loads the extension: every glibc version
`pyamplicol._rusticol` needs is looked up in the version definitions of the
glibc libraries the Python process has loaded (`libc.so.6`, `libm.so.6`, ...),
the same check the dynamic loader makes. A missing one fails cleanly at load
time with `pyamplicol.NativePlatformError` (a `DependencyError`) instead of the
loader's raw message:

```text
pyamplicol.api.errors.NativePlatformError: Native library
.../pyamplicol/_rusticol.so needs the ELF version GLIBC_2.38, which this
process's glibc 2.36 does not define. Run it on a Linux system or container
whose glibc defines it (glibc 2.38 or newer); `ldd --version` shows the glibc
release. Or build pyAmpliCol's native libraries from source on this host so
that they link against its glibc (see 'Native libraries and glibc' in
docs/user/installation.md).
```

The error's attributes carry the same facts: `library`, `missing` (the version
names the running glibc does not define), `required` (the newest glibc release
among them, `"2.38"`; None if only ABI markers are missing, which no release
number guarantees) and `found` (the running glibc release). The requirements
are read from the installed library itself, so a library built against an
older glibc (a wheel, or a source build on that host) loads wherever the
dynamic loader accepts it. A process that does not run on glibc (for example on
musl) is named as such. When the dynamic loader itself reports a missing glibc
version, its error is translated into the same `NativePlatformError` with the
loader's text appended; when the library it names belongs to a dependency (for
instance a locally built Symbolica, whose own requirements can be higher), the
remedy is a build of that library for this glibc.

C, C++ and Fortran clients of `librusticol_loop_sdk.so` do not see
`NativePlatformError`, which belongs to the Python package. The library's
symbols bind numbered glibc versions (`__isoc23_strtol@GLIBC_2.38`, for
instance), so linking a client executable with a toolchain on an older glibc
fails at link time with
`` librusticol_loop_sdk.so: undefined reference to `__isoc23_strtol@GLIBC_2.38' ``.
A client built on glibc 2.38 or newer and run on an older glibc is stopped by
the dynamic loader before `main` with
`` version `GLIBC_2.38' not found (required by .../librusticol_loop_sdk.so) ``.
C and C++ clients of the release build can define `RUSTICOL_VIRTUAL_SHARED` to
stop at compile time instead (see "glibc floor (Linux)" in the loop SDK's
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

## Build a wheel from a checkout and run the loop examples

These commands build the platform wheel from a checkout with the release
tooling, install that wheel alone in a fresh virtual environment outside the
repository, and run every shipped one-loop example. They are the commands the
release check runs.

**1. Build the wheel.** The toolchain is Rust 1.97.1 (`rust-toolchain.toml`),
a C/C++ compiler, `make` and `m4`, and `gfortran` for the Fortran SDK; the
build also needs network access for the locked Python and Cargo inputs. On
Linux x86-64 the release build runs in the pinned manylinux 2.28 image, as
the release workflow (`.github/workflows/release-artifacts.yml`) does: the
source distribution first, then the wheel built from it (an audited
`cp311-abi3` wheel at most 95,000,000 bytes, below PyPI's default 100 MB
file limit):

```console
git clone https://github.com/MadGraphTeam/pyamplicol.git
cd pyamplicol
docker run --rm -v "$PWD:/io" -w /io -e PYAMPLICOL_BUILD_MODE=release \
  quay.io/pypa/manylinux_2_28_x86_64@sha256:b04887b645dde99b9e955aeae3ff4da414992d0bd88259f046295b56361c5614 \
  /bin/bash -euxo pipefail -c '
    git config --global --add safe.directory /io
    yum install -y gcc-c++ gcc-gfortran make
    curl --proto "=https" --tlsv1.2 -sSf -o /tmp/rustup-init \
      https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init
    chmod +x /tmp/rustup-init
    /tmp/rustup-init -y --profile minimal --default-toolchain 1.97.1
    source "$HOME/.cargo/env"
    PY311=/opt/python/cp311-cp311/bin/python
    "$PY311" -m pip install "build>=1.2,<2" "maturin==1.15.0"
    "$PY311" tools/release/build_release_artifacts.py --sdist-only --output-dir dist/sdist
    "$PY311" tools/release/build_release_artifacts.py \
      --retained-sdist dist/sdist/pyamplicol-*.tar.gz --output-dir dist
  '
```

The build takes about an hour on a large host. The files in `dist/` belong to
the container's root user. On macOS, and on a Linux host that provides
`patchelf` and the toolchain above, run the two
`tools/release/build_release_artifacts.py` commands directly with
`PYAMPLICOL_BUILD_MODE=release` and a Python 3.11 that has `build` and
`maturin==1.15.0` installed.

**2. Install the wheel alone in a fresh environment** outside the repository
(Python 3.11 or newer). Only the wheel comes from the checkout; its
dependencies (Symbolica 3.0.3, ufo-model-loader 1.0.1, NumPy, ...) come from
PyPI:

```console
python3.11 -m venv ~/pyamplicol-1.0.3-venv
. ~/pyamplicol-1.0.3-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "$PWD"/dist/pyamplicol-1.0.3-*.whl
```

On NixOS, PyPI's manylinux wheels of the dependencies (NumPy) need
`libstdc++.so.6` on the loader path, which NixOS does not provide by default:
use `nix-ld`, or export `LD_LIBRARY_PATH` with your GCC runtime library
directory before running `pyamplicol`.

**3. Check the installation and run the loop examples** from a directory
outside the repository. Generation uses Symbolica: pyAmpliCol registers its
package key at import; if you hold a personal Symbolica license, export it as
`SYMBOLICA_LICENSE` first (see [Symbolica and Licensing](symbolica-and-licensing.md)).

```console
mkdir -p ~/pyamplicol-run && cd ~/pyamplicol-run
python -c 'import pyamplicol; print(pyamplicol.__version__)'
pyamplicol self-test
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
make -C native loop
for example in loop_uux_ttxgg loop_uux_ddxepem loop_gg_zgg loop_gg_epemmpmm; do
  pyamplicol "${example}_generate.toml"
  pyamplicol check "${example}_generate.toml"
  pyamplicol "${example}_check.toml"
  pyamplicol "${example}_evaluate.toml"
  pyamplicol "${example}_profile.toml"
  python python/loop_walkthrough.py "${example}_generate.toml" --points 2 --regenerate \
    --native-client native/loop_client
done
```

Each step is described in [the examples guide](../../examples/README.md#one-loop-examples);
`--regenerate` makes the walkthrough generate its artifact again with the
example's correlator declaration (the card's generation above writes it
without one).
Measured from the wheel (2026-11-11, an AMD EPYC host shared with other
jobs, the four examples run side by side): `loop_uux_ttxgg` and
`loop_gg_zgg` generate in under a minute and run their whole sequence in
about 6 minutes (peak 2.6 and 10.8 GiB); the complex-mass examples
`loop_gg_epemmpmm` and `loop_uux_ddxepem` generate in 5 and 7 minutes and
take about 21 and 36 minutes for the whole sequence (peak 5.7 and 19.6 GiB,
the latter in the walkthrough). Each walkthrough generates its artifact again
in a child process.

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
- Rust 1.96 or newer (the workspace `rust-version`);
- a C/C++ toolchain, `make` and `m4` (GMP, MPFR and MPC are compiled from
  the sources of the locked `gmp-mpfr-sys` crate);
- network access to the release-locked Python, Cargo, and SymJIT inputs.

A local source build links GMP, MPFR and MPC statically into its own native
libraries (`gmp-mpfr-sys`'s default); it is not meant for redistribution.
Distribution wheels (`tools/release/build_release_artifacts.py`, which passes
the backend setting `pyamplicol.distribution-wheel=true`) instead build the
three libraries as shared libraries and bundle them as described in
[GMP, MPFR and MPC in the wheels](#gmp-mpfr-and-mpc-in-the-wheels); on Linux
that build runs in a manylinux 2.28 environment and needs `patchelf`.

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
Symbolica 3.0.3 and ufo-model-loader 1.0.1, and the release-mode native build with
SymJIT 2.28.3 (see `dependencies/README.md` for the native pins). It does not use an editable installation. Building a new
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

A native library needs a glibc symbol version that this system's glibc does not
define (see [Native libraries and glibc](#native-libraries-and-glibc)); the
message names the library, the missing versions and the running glibc. Update
glibc or use a system or container whose glibc defines them (glibc 2.38 or
newer for the release native libraries built from a checkout), or build the
native libraries from source on this host.

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
