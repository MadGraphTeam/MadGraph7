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

After the 0.2.0 release candidate is validated and tagged, build its source
snapshot with:

```console
git clone --branch v0.2.0 --depth 1 https://github.com/mg5amcnlo/pyamplicol.git
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
  calculations;
- recurrence, compiled-DAG, eager, and on-the-fly execution modes;
- JIT, C++, and assembly evaluator backends where supported;
- binary64 execution without importing Symbolica, plus precision-controlled
  Python evaluation when exact expressions are retained. On-the-fly execution
  currently supports native binary64 only.

Reusable artifacts preserve complete public helicity and color physics unless
the request explicitly fixes selectors at generation time. Runtime calls can
then select one flow or helicity globally or per phase-space point without
regenerating the artifact. On-the-fly artifacts always keep selection at
runtime and carry the complete contract in a compact query-local seed rather
than materializing the full axes; `inspect` reports their physical census
without constructing it.
Recurrence, eager, and on-the-fly execution reuse the same prepared model
kernel bundle.

Contracted NLC/full-colour recurrence and on-the-fly execution can use the
exact `symmetric-group-fft` colour contraction. It transforms certified
permutation-orbit blocks and retains unsupported terms as exact direct
residuals. Recurrence artifacts persist one helicity-parametric physical-colour
schedule, its helicity-support masks, and precomputed per-helicity row groups;
loading binds those groups once, so warmed evaluation does not rescan the
masks. On-the-fly execution instead constructs and caches the requested family
on first use, which is why that warm-up belongs to its plotted setup time.

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

The source checkout also contains a thin orchestrator for the dedicated
FullColor FFT comparison. It delegates generation and timing to the existing
pyAmpliCol profiling commands and schedules independent measurement children:

```console
just dev-install --with-legacy-amplicol --with-reference-fft
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --multiplicities 2 3 4 5 \
  --lines reference-fft amplicol pyamplicol-recurrence pyamplicol-otf madgraph \
  --cores 8 --candidate-cores 1 \
  --memory-limit-gib 30 --time-limit-seconds 3600 \
  --amplicol-root /path/to/AmpliCol \
  --reference-fft-root /path/to/AllGluonsMultipletFFT \
  --madgraph-root /path/to/MG5_aMC
```

`--multiplicities` adds the selected values to the persistent fill history and
defaults to `2 ... 9`. `--lines` similarly adds any combination of
`reference-fft`, `amplicol`, `pyamplicol-recurrence`, `pyamplicol-otf`, and
`madgraph`; the two pyAmpliCol groups each schedule their direct and FFT
companion curves together so they reuse the same generation lane. Dependencies
are selected automatically.
Repeating a command against the same output unions both selections, resumes
unfinished cells, and skips completed cells;
`--resume` is an explicit alias for that default. `--cores` is the total
scheduler budget, while `--candidate-cores` is one candidate child's core
claim and evaluator setting. The memory and time limits are strict per-child
cutoffs. `--retry` reruns only failed/skipped cells in the active selection.
`--overwrite` reruns every selected cell and replaces each old result only when
that cell's worker is about to launch; queued or blocked cells retain their old
results. Use `--output PATH` for an independent run directory. `--refresh`
removes only that exact recognized output directory and restarts it, so a
custom output also scopes the refresh; a path that does not exist simply starts
cleanly. Without `--output`, fixed and summed
workloads use separate `IMPLEMENTATION_DOCS/RESULTS/fft-profiling/runs/`
directories named `cluster-fullcolor-n2-n9` and
`cluster-fullcolor-helicity-sum-n2-n9`. Refresh also shares the persistent
MadGraph cache-writer lock and refuses to delete a run while a standalone
MadGraph profiler is using that cache.

Add `--compare-helicity-sums` for the independent complete physical-helicity-
sum workload. The fixed-helicity MadGraph lane selects the shared helicity
through the generated `MATRIX(P,NHEL,IC)` entry point. The summed lane instead
calls the generated `SMATRIX(P,ANS)` with `USERHEL=-1`; MadGraph applies its
native IDEN normalization and may reuse its warmed `GOODHEL` pruning. Fixed-
helicity and summed overlays carry distinct workload identities and cannot be
mixed. `just dev-install` omits the developer-only AmpliCol and Reference FFT
repositories unless they are requested with `--with-legacy-amplicol` and
`--with-reference-fft`; either opt-in also installs the `fft-profiling` Python
extra into `.venv`. Their profiler roots default to
`dependencies/checkouts/legacy-amplicol` and
`dependencies/checkouts/reference-fft`; `--build-amplicol` may build the
AmpliCol probe once. Both paths can be overridden explicitly with
`--amplicol-root` and `--reference-fft-root`. The MadGraph root defaults to
`PYAMPLICOL_MADGRAPH_ROOT` or a recognized developer checkout and may be set
explicitly with `--madgraph-root`.

The published fixed-helicity MadGraph series currently has measured points
through `n=5` for pure gluons and `n=6` for
`d d~ > d d~ + gluons`. Pure-gluon `n=6` retains its measured resource cutoff;
`n=7..9` are explicit protocol-scope not-applicable cells for both families.
The independent helicity-sum MadGraph series has measured points for both
families at `n=2..5`. Every admitted point passed the same-workload numerical
gate before entering the PDFs.

The rolling plot frontiers are per process and implementation, rather than a
claim that every curve reaches the same `n`. Both fixed-helicity and helicity-
sum OTF curves are requested only through final-state `n=6`; beyond that the
publication protocol retains recurrence, AmpliCol, and Reference FFT where
applicable. Within that frontier, cutoffs are annotated rather than hidden or
interpolated. Pure-gluon OTF FFT reached the 3,600 s first-use runtime cap at
`n=6` and retains its measured `n=5` point. The helicity-sum comparison extends
the `d d~ > d d~ + gluons` curves through `n=6` using an authenticated isolated
30 GiB extension. At that point every requested curve measured successfully.
For OTF, direct and FFT setup took 1,725.8 s and 1,607.9 s, while warmed runtime
was 5.321 and 3.759 ms/point respectively (a 1.416x FFT speedup). The measured
peak RSS values were 1.17 and 1.22 GiB.

The plotted *setup time* is deliberately method-specific. pyAmpliCol includes
artifact generation, a fresh load, the first requested evaluation, and OTF
family warm-up where applicable. Reference FFT includes its build,
initialization, and first pass. AmpliCol includes process/color-object
generation for the fixed workload, or process/raw-library generation and build
plus the immutable snapshot for the summed workload. Warmed runtime is measured
separately after those setup boundaries. Resource limits apply individually to
each child. The retained isolated high-frontier extensions use a 30 GiB
process-tree guard.

After the first snapshot is published, rendering never waits for workers and
uses the latest available data. Concurrent renders and refresh publication are
serialized so an older render cannot replace a newer one:

```console
python tools/fft_profiling/fft_profiling.py --render
python tools/fft_profiling/fft_profiling.py --render --compare-helicity-sums
python tools/fft_profiling/fft_profiling.py --render --output /path/to/run
```

For a custom helicity-sum output, repeat `--compare-helicity-sums` on the
render command. Default outputs also refresh the corresponding canonical PDF;
custom outputs keep their PDF inside the selected run directory. During a
scan, the progress display reports the active cell, total live RSS, and
occupied core slots.

An older local MadGraph overlay that predates the node-fingerprint field may
appear only in a nonterminal anytime render, only when its system, machine, and
Python version match the current workstation. Such plots carry an explicit
provenance note; the strict terminal publication merger still requires the
complete host identity produced by a fresh profiler run.

The public performance index retains four selected rendered snapshots. Raw
JSON, generated tables, attempts, and campaign workspaces stay untracked:

- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)
- [FullColor FFT selected-helicity snapshot](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final.pdf)
- [FullColor FFT helicity-sum snapshot](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final_helicity_sum.pdf)

These are manual measurement snapshots rather than release-CI results; raw
campaign data remains local. The general host report format is reproducible
from an installed package, while the FullColor FFT snapshots use the source-
checkout orchestrator above.

## Documentation

Read the complete [pyAmpliCol documentation](https://mg5amcnlo.github.io/pyamplicol/).

## Dependencies and license

Release builds use pinned published dependencies plus SymJIT 2.22.0 from an
immutable revision of the official
[symjit-crate repository](https://github.com/siravan/symjit-crate).

pyAmpliCol is distributed under the 0BSD license. Third-party components and
model assets retain their own terms; see
[THIRD_PARTY_NOTICES.md](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).
