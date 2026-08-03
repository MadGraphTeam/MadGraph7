<!-- SPDX-License-Identifier: 0BSD -->

# pyAmpliCol

[![Tests](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml/badge.svg)](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml)

pyAmpliCol generates and evaluates color-ordered scattering amplitudes from
built-in, JSON, or UFO models. It provides a typed Python API and CLI, fast
Rust-backed execution, runtime helicity and color-flow selection, and generated
Python, C11, C++17, Fortran 2008, and Rust 2021 interfaces.

## Release status

Version `0.1.0` has been tagged as an immutable archival source snapshot but
has not yet been uploaded to PyPI or TestPyPI. The
[validated release-artifacts workflow](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/release-artifacts.yml)
produces one source distribution and three `cp311-abi3` wheels; publication
uses a successful run whose head SHA is the intended release source:

- macOS 11 or newer on Apple silicon;
- macOS 11 or newer on x86-64;
- manylinux 2.28 x86-64.

Each wheel completed the full installed Python, C11, C++17, Fortran 2008, and
Rust 2021 API deployment on CPython 3.11. CPython 3.14 received a focused abi3
installation, import, metadata, and direct-runtime smoke test. The release
workflow did not run the separate performance campaigns.

See the
[release status](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/release-status.md)
for the remaining upload step.

## Installation

Once the release is uploaded:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The binary wheels include the Rust runtime and native SDK; wheel users do not
need a Rust compiler. pyAmpliCol has no LHAPDF dependency.

To build the tagged source snapshot before the PyPI upload:

```console
git clone --branch v0.1.0 --depth 1 https://github.com/mg5amcnlo/pyamplicol.git
cd pyamplicol
python -m pip install .
```

A source build requires Python 3.11 or newer, Rust 1.89 or newer, and a C/C++
toolchain. A Fortran compiler is required only for Fortran consumers.

Contributor setup uses pinned source dependencies and produces explicitly
non-publishable candidate builds:

```console
nix develop  # optional on Nix/NixOS
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The first `just dev-install` native build can take several minutes. Repeated
installs reuse the workspace-local Cargo cache and are substantially faster.

Full installation details are in the
[installation guide](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/installation.md).

## Quick start

Copy the installed examples into an editable workspace:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

The primary example generates a multiprocess `p p > Z j j` artifact from the
packaged serialized Standard Model, then evaluates and profiles one concrete
subprocess. It retains the 18 model-supported tree-level channels and reports
the omitted loop-induced `g g > Z g g` candidate:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

For direct CLI use:

```console
pyamplicol generate "d d~ > z g" ./artifacts/ddbar_zg \
  --model built-in-sm

pyamplicol inspect ./artifacts/ddbar_zg
```

The same runtime is available from Python:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > z g g")
total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
assert resolved.total() == total
```

See the
[examples guide](https://github.com/mg5amcnlo/pyamplicol/blob/main/examples/README.md)
for complete cards, parameter updates, selector examples, and generated API
drivers.

## Models and execution

pyAmpliCol supports:

- the packaged built-in Standard Model;
- packaged serialized JSON and trusted UFO examples;
- user-supplied JSON or trusted UFO model paths;
- leading-color, contracted next-to-leading-color, and contracted full-color
  calculations;
- recurrence, compiled-DAG, and eager execution modes;
- JIT, C++, and assembly evaluator backends where supported;
- binary64 execution without importing Symbolica, plus precision-controlled
  Python evaluation when exact expressions are retained.

Generated artifacts preserve complete public helicity and color axes. Runtime
calls can select one flow or helicity globally or per phase-space point without
regenerating the artifact.

The public C ABI is version 1. Every generated artifact can include standalone
Python, C11, C++17, Fortran 2008, and dependency-free Rust 2021 drivers backed
by the wheel-owned static Rusticol SDK.

## Profiling campaigns

An installed wheel can populate a self-contained campaign workspace:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
```

Each campaign keeps attempts, prepared artifacts, logs, locks, and leases in
its visible `campaign_artifacts/` directory. Moving or renaming the whole
campaign moves that state with it and never consults legacy repository-level
`.artifacts` state. `--force` resets that local state plus the managed PDF,
summary IDs, measurement lineage, and known LaTeX byproducts while preserving
unrelated destination files and a previously recorded original-AmpliCol
checkout. Stop active campaign processes before resetting their destination.

All pyAmpliCol backends work from installed resources. The optional original
AmpliCol reference backend additionally requires
`--original-amplicol PATH_TO_COMPLETE_CHECKOUT`; it is unavailable when that
checkout is not supplied. Neither pyAmpliCol nor the supported patched
original-AmpliCol comparison checkout requires LHAPDF.

Four rendered report snapshots are retained. Their raw measurements, generated
tables, and report workspaces are intentionally not shipped in the source tree:

- [Consolidated report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/arxiv/pyAmpliCol.pdf)
- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3/pyAmpliCol.pdf)
- [MacBook M3 Z-process subset](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3/z_table/z_table.pdf)
- [x86 EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/x86_EPYC/pyAmpliCol.pdf)

These reports come from separate manual measurement campaigns; they are not
release-CI results.

## Documentation

- [User guide](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/index.md)
- [Configuration](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/configuration.md)
- [Models and processes](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/models.md)
- [Runtime](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/runtime.md)
- [Native SDK](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/native-sdk.md)
- [Symbolica licensing](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/user/symbolica.md)
- [Performance reports](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/README.md)

## Dependencies and license

Release builds use pinned published dependencies plus SymJIT 2.22.0 from an
immutable revision of the official
[symjit-crate repository](https://github.com/siravan/symjit-crate). pyAmpliCol
does not carry a private SymJIT fork or a local SymJIT patch.

pyAmpliCol is distributed under the 0BSD license. Third-party components and
model assets retain their own terms; see
[THIRD_PARTY_NOTICES.md](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).
