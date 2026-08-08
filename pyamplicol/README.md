<!-- SPDX-License-Identifier: 0BSD -->

<p align="center">
  <img src="https://raw.githubusercontent.com/mg5amcnlo/pyamplicol/main/docs/assets/pyamplicol_logo.png" alt="pyAmpliCol" width="760">
</p>

<p align="center">
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/v/pyamplicol.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/pyversions/pyamplicol.svg" alt="Python versions"></a>
  <a href="https://mg5amcnlo.github.io/pyamplicol/"><img src="https://img.shields.io/badge/docs-User%20Guide-2f81f7.svg?logo=githubpages" alt="pyAmpliCol documentation"></a>
  <a href="https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml"><img src="https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/mg5amcnlo/pyamplicol/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-0BSD-blue.svg" alt="License: 0BSD"></a>
</p>

<p align="center"><strong>Fast color-ordered scattering amplitudes from Python and native APIs.</strong></p>

pyAmpliCol generates and evaluates color-ordered scattering amplitudes from
built-in, JSON, or UFO models. It provides a typed Python API and CLI, fast
Rust-backed execution, runtime helicity and color-flow selection, and generated
Python, C11, C++17, Fortran 2008, and Rust 2021 interfaces.

Explore the complete [pyAmpliCol documentation](https://mg5amcnlo.github.io/pyamplicol/)
for guided workflows, API examples, technical reference, and release support.

## Installation

Install the release from PyPI:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The binary wheels include the Rust runtime and native SDK; wheel users do not
need a Rust compiler. pyAmpliCol has no LHAPDF dependency.

After the 0.1.4 release candidate is validated and tagged, build its source
snapshot with:

```console
git clone --branch v0.1.4 --depth 1 https://github.com/mg5amcnlo/pyamplicol.git
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
[documentation](https://mg5amcnlo.github.io/pyamplicol/).

## Quick start

Copy the installed examples into an editable workspace:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

Keep the Python environment containing pyAmpliCol activated while using the
top-level CLI and Python examples, or invoke its executables by explicit path.
For a copy below a source checkout prepared by `just dev-install`, generated
artifact Python and native API drivers also find the nearest checkout `.venv`
automatically; explicit SDK overrides and an active environment take
precedence.

The primary example generates a multiprocess `p p > Z j j` artifact from the
packaged serialized Standard Model, then evaluates and profiles one concrete
subprocess. Its 19 ordered candidates collapse to eight side-permutation
classes; it stores the seven tree-level representatives and reports the
omitted loop-induced `g g > Z g g` class. The card inherits the portable JIT
O2 default, so its process artifact can be moved between supported 64-bit
little-endian macOS arm64, macOS x86_64, and Linux x86_64 hosts:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

For direct CLI use:

```console
pyamplicol generate "d d~ > z g" ./artifacts/builtin_ddbar_to_zg \
  --model built-in-sm

pyamplicol inspect ./artifacts/builtin_ddbar_to_zg
```

Process generation can also be steered directly from Python:

```python
from pyamplicol import GenerationConfig, Generator

generator = Generator(GenerationConfig(workers=4))
plan = generator.plan("d d~ > z g")  # Resolve without writing an artifact.
result = generator.generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    mode="replace",
)
print(result.output)
```

The same runtime is available from Python:

```python
import json
from pathlib import Path

from pyamplicol import Runtime

momenta = json.loads(Path("data/pp_zjj_momenta.json").read_text())
runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
assert resolved.total() == total
```

Concrete process expressions may reorder particles within the incoming side or
within the outgoing side. Rusticol maps momenta, helicities, color flows, and
resolved metadata to that requested order; particles never cross the `>`
boundary. Stable process IDs remain available when more than one generated
representative could match an expression.

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
  calculations, with on-the-fly execution currently limited to leading color;
- recurrence, compiled-DAG, eager, and on-the-fly execution modes;
- JIT, C++, and assembly evaluator backends where supported;
- binary64 execution without importing Symbolica, plus precision-controlled
  Python evaluation when exact expressions are retained. On-the-fly execution
  currently supports native binary64 only.

Generated artifacts preserve complete public helicity and color physics.
Runtime calls can select one flow or helicity globally or per phase-space point
without regenerating the artifact. On-the-fly artifacts keep this contract in a
compact query-local seed rather than materializing the full axes in the
artifact; `inspect` reports their physical census without constructing it.
Recurrence, eager, and on-the-fly execution reuse the same prepared model
kernel bundle.

The public C ABI is version 1. Every generated artifact can include standalone
Python, C11, C++17, Fortran 2008, and dependency-free Rust 2021 drivers backed
by the wheel-owned static Rusticol SDK.

## Profiling campaigns

An installed wheel can populate a self-contained campaign workspace:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
cd ./pyamplicol-profiling-campaign
./steer_performance_campaign.py run \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in
```

That deliberately small real campaign measures only the final-state-
multiplicity-one `d d~ > Z` recurrence cell. Broader campaign selections are
intended for dedicated profiling hosts. The
[documentation](https://mg5amcnlo.github.io/pyamplicol/) covers selection,
continuation, optional original-AmpliCol comparisons, artifact retention, and
PDF generation.

The repository retains only two rendered performance snapshots. Raw JSON,
generated tables, attempts, and campaign workspaces stay untracked:

- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)

These are manual measurement snapshots rather than release-CI results; raw
campaign data remains local and the report format is reproducible from an
installed package.

## Documentation

Read the complete [pyAmpliCol documentation](https://mg5amcnlo.github.io/pyamplicol/).

## Dependencies and license

Release builds use pinned published dependencies plus SymJIT 2.22.0 from an
immutable revision of the official
[symjit-crate repository](https://github.com/siravan/symjit-crate).

pyAmpliCol is distributed under the 0BSD license. Third-party components and
model assets retain their own terms; see
[THIRD_PARTY_NOTICES.md](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).
