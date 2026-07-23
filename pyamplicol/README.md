<!-- SPDX-License-Identifier: 0BSD -->

# pyAmpliCol

[![Tests](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml/badge.svg)](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/tests.yml)

pyAmpliCol is a generator using current recursion to build fast color-ordered
matrix-element evaluators. It supports built-in and external UFO models, with
generated processes accessible from Python, Rust, C++17, and Fortran 2008.

## Quick Start

The primary workflow uses the packaged serialized Standard Model, expands
`p p > Z j j` into concrete subprocesses, and writes one multiprocess artifact:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
```

`pyamplicol==0.1.0` is not published yet. The commands above are the intended
binary-wheel workflow; use the contributor workflow below for this milestone
checkout. Current publication gates are listed in
[Release Status](docs/user/release-status.md).

The example deliberately defines a two-flavor `p`/`j` set containing `d`,
`d~`, and `g`. The current planner reduces the cartesian request to 19 physical
concrete processes. Select one using its readable concrete expression or its
stable runtime ID. For example, `d d~ > z g g` has the stable ID
`p_p_to_z_j_j_4` inside `artifacts/pp_zjj`.

Inspect the complete output inventory before selecting a process:

```console
pyamplicol inspect artifacts/pp_zjj
```

The human view is a colored table of artifact, process, alias, coverage, and
runtime metadata. Add `--format json` for the corresponding machine-readable
inventory, or `--process 'd d~ > z g g'` for detailed physics metadata for one
entry.

Runtime parameters can be supplied as one atomic JSON update or as direct UFO
external-parameter overrides:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > z g g' \
  --momenta data/pp_zjj_momenta.json \
  --model-parameters data/model_parameters.json
```

The equivalent machine-stable selector is `--process p_p_to_z_j_j_4`.
Case and whitespace are normalized for expression selectors; concrete particle
labels and ordering remain significant.

`data/model_parameters.json` updates the real model inputs `aS` and `MZ`.
Unknown, immutable, or invalid entries reject the complete update.

## Runtime Profiling

Profile an already generated artifact with its deterministic validation point,
or provide phase-space points explicitly with `--momenta`:

```console
pyamplicol profile artifacts/pp_zjj \
  --process p_p_to_z_j_j_4 \
  --target-runtime 1.0 \
  --batch-size 128
```

`--process` accepts either a stable process/alias ID or the exact concrete
expression, such as `d d~ > z g g`. The profiler warms the selected runtime,
calibrates independent timed blocks and repetitions per block toward the target
duration, and reports the mean time per point with standard deviation, standard
error, and relative standard error (standard error divided by the mean). The
result identifies the compiled/eager mode and whether color and helicity axes
are complete or selected. In a terminal it uses a colored progress bar with
live elapsed-time, sampling, and uncertainty metadata, followed by
colorized PrettyTables. Pressing `Ctrl-C` stops sampling and reports a clearly
marked partial result from every fully completed block. Native Rusticol
profiling is a separate paired pass over the same batch and repetition count as
the ordinary wall-time block. It reports exclusive native input, state, source,
momentum, model, stage, amplitude, reduction, materialization, output-copy, and
selector phases; internal leaf-gather/backend/output-gather attribution; and
per-stage detail. Internal attribution is non-additive: full-stage evaluator
envelopes own leaf gathering, while composed selected-chunk input-pack
envelopes own it. A second `Native Work Counters` table reports data movement
and materialization per profiled point plus backend/allocation activity per
runtime call. Repeated profiling uses constant aggregate storage rather than a
profile vector proportional to the repetition count. `--format json` keeps
stdout limited to the typed `BenchmarkResult` payload, including the same
timings and counters; progress and diagnostics remain on stderr.

`pyamplicol benchmark` remains a compatibility alias with the same options and
output. TOML run cards continue to use `action = "benchmark"`, and the Python
interface remains `BenchmarkRunner`/`BenchmarkResult`.

### Reproduce the `q q~ > Z + 6g` benchmark workloads

The packaged examples include the two independent LC workloads used for the
corresponding performance-PDF measurements. Copy the examples, enter that
directory, then generate and profile each artifact from its own card:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples

pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile --card benchmark_z6g_single_flow_helicity_sum.toml

pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile --card benchmark_z6g_all_flows_single_helicity.toml
```

The first card uses the default `topology-replay` LC layout, which is optimized
for selecting one physical color flow at runtime and summing all helicities.
The second explicitly uses `all-flow-union`, which builds a shared cross-flow
recurrence optimized for summing all flows at one runtime-selected helicity.
Both artifacts retain every physical LC flow and helicity, so either can select
any retained component globally or per phase-space point; these are layout
choices, not generation-selected coverage shortcuts. Their outputs are
respectively `artifacts/uubar_z6g_single_flow_helicity_sum` and
`artifacts/uubar_z6g_all_flows_single_helicity`.

Both cards use compiled JIT O3 and native Rusticol wall timing. The default
layout can be overridden directly; for example, the following command creates
an all-flow union without editing the single-flow card:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --lc-flow-layout all-flow-union \
  --set generation.output=artifacts/uubar_z6g_all_flows_override
```

To exercise the same complete selector contract in eager mode, override the
execution mode and choose a distinct output path:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --execution-mode eager \
  --set generation.output=artifacts/uubar_z6g_single_flow_helicity_sum_eager
```

The cards use `u u~ > Z g g g g g g`. Replace `u u~` with `d d~` in both cards
when reproducing a PDF row whose literal process is `d d~ > Z+6g`; the runtime
selector methodology is otherwise unchanged. See
[the packaged-example guide](examples/README.md#reproduce-the-z-ladder-workloads)
for changing the selected flow or helicity and for the generation-specialized
comparison cards.

## Installation

| Goal | Command | Dependency source |
| --- | --- | --- |
| Install a released wheel | `python -m pip install pyamplicol` | PyPI; no Rust compiler |
| Build and install this source tree | `python -m pip install .` | Published packages only |
| Retain a local wheel | `just wheel` | Published packages only; writes `dist/` |
| Install the matching retained wheel | `just install-wheel PYTHON=/path/to/python` | Existing or newly built wheel |
| Enter the complete Nix contributor shell | `nix develop` | Python/Rust/native/Fortran/TeX toolchains |
| Prepare a contributor environment | `just dev-install` | Pinned, non-publishable candidate inputs |

Source builds require Python 3.11 or newer, Rust 1.89 or newer, and a C/C++
toolchain. A Fortran compiler is needed only for Fortran consumers. The static
native SDK is staged into a built wheel, so install that wheel before using
`rusticol-config`.

Contributor setup is source-checkout-only:

```console
nix develop  # optional on Nix/NixOS; supplies every system tool
just dev-install
PYTHON=.venv/bin/python just dev-test
just --list
```

The repository flake supplies Python 3.11, the pinned Rust toolchain, C/C++ and
Fortran compilers, native libraries, PDF utilities, and a complete TeX setup.
`just dev-install` remains responsible for the lock-controlled Python packages
and candidate dependency checkouts; the flake deliberately does not duplicate
them. It also stages the candidate wheel's native extension and SDK beside
`src/pyamplicol` for source-tree tests. Contributor builds record a lightweight
native-source build ID and the staged extension hash. If native sources change,
or a different extension is found, the next import fails with a request to rerun
`just dev-install` instead of silently loading stale code. Published wheels keep
the normal package-manager import path.

Strict source and wheel builds use `dependencies/release-lock.toml` and fail
closed while a required published dependency remains unverified. Contributor
inputs never enter a release wheel or source distribution. See
[Installation](docs/user/installation.md) for the complete build matrix.

## Generate

The primary run card is:

```toml
schema_version = 1
action = "generate"

[model]
source = "models/json/sm/sm.json"
restriction = "default"

[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]

[color]
accuracy = "lc"

[generation]
output = "artifacts/pp_zjj"
emit_api_bundle = true
```

Paths in a card are resolved relative to that card. Equivalent direct steering
uses the same schema fields:

```console
pyamplicol generate "p p > Z j j" artifacts/pp_zjj \
  --model models/json/sm/sm.json \
  --restriction default \
  --multiparticle 'p=d,d~,g' \
  --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 \
  --max-quark-lines 2 \
  --color-accuracy lc \
  --jit-optimization-level 3
```

Configuration precedence is defaults, TOML, dedicated command flags, ordered
`--set dotted.path=value` overrides, then recorded license/resource clamps.
Unknown fields are rejected. `examples/all_options.toml` is the exhaustive
commented schema-v1 reference.

`generate --dry-run` does not write an artifact. Built-in models can be planned
directly. For an external source, first compile it or populate the model cache:

```console
pyamplicol model compile models/json/sm/sm.json models/sm.pyAmplicol-model.json \
  --restriction default
pyamplicol generate "p p > Z j j" artifacts/pp_zjj \
  --model models/sm.pyAmplicol-model.json \
  --multiparticle 'p=d,d~,g' --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2 --dry-run
```

Useful local commands include:

```console
pyamplicol examples list
pyamplicol config template run.toml
pyamplicol config resolve run.toml --format json
pyamplicol model inspect models/json/sm/sm.json --restriction default
pyamplicol doctor
pyamplicol self-test
```

## Prepared Eager Execution

The default `compiled` execution mode builds process-wide stage evaluators.
The opt-in `eager` mode prepares model-local kernels once and lets Rusticol
execute compact process DAG tables. Wheels include built-in-SM JIT O3 packs for
both `x86_64` and `aarch64` and select the host architecture automatically, so
the common path needs no model-preparation command:

```console
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_eager \
  --model built-in-sm \
  --execution-mode eager --color-accuracy nlc
```

Prepare a local bundle when using an external model or selecting a C++/ASM
pack instead:

```console
pyamplicol model compile models/json/sm/sm.json models/ufo-sm-cpp-o3.pyamplicol-model \
  --backend cpp
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_cpp_eager \
  --model models/ufo-sm-cpp-o3.pyamplicol-model --execution-mode eager
```

The `.pyamplicol-model` bundle is self-contained and retains exact expressions
alongside one prepared JIT, ASM, or C++ kernel pack. Eager generation requires
a matching pack and fails with the exact preparation command when one is
absent; it never silently compiles kernels. The pack's backend and optimization
settings are authoritative and any requested adjustment is recorded.
Consequently, `--model built-in-sm` in eager mode resolves to the wheel-owned
JIT O3 pack; pass an explicitly prepared path to select another backend.

Portable model IR and prepared executable state have different compatibility
contracts. `.pyAmplicol-model.json` IR is architecture-independent, whereas a
SymJIT storage-v3 pack is scoped to one architecture class. Its saved state is
tested across operating systems within that class, such as Linux to macOS on
`x86_64`, but is not transferable between `x86_64` and `aarch64`. The loader
rejects an explicit cross-architecture bundle before DAG construction or
SymJIT loading. A future SymJIT storage ABI may permit wider portability without
changing the `.pyamplicol-model` bundle interface.

Rusticol tiles large point batches through a reusable workspace. Configure the
upper bounds with `[evaluator.eager] point_tile_size = 1024` and
`workspace_mib = 256`. The generated artifact supports the same Python, C11,
Rust, C++17, and Fortran 2008 total/resolved APIs and selectors as compiled
mode. JIT f64 evaluation of saved SymJIT applications is Symbolica-free after
generation. Eager generation and non-f64 evaluation retain the normal
Symbolica requirement. ASM and C++ accept batches but remain scalar; only
SymJIT may auto-SIMD.

## Typed Python

The Python API uses the same resolved configuration. External sources are
compiled before non-writing planning:

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config

# Read the same TOML card used by the command-line example. Resolution applies
# defaults and validates all settings before any expensive work begins.
resolution = resolve_config("generate_pp_zjj_from_ufo_sm.toml")
config = resolution.effective

# Preserve the card's named or file-based restriction while compiling the
# selected model. The configured cache avoids repeating this preparation.
model = ModelSource.from_config(config.model).compile(
    cache_dir=config.model.cache_dir,
    use_cache=config.model.cache,
)

# Convert each human-readable process declaration from the card into a typed
# request. Here `process_entry` is one item from the card's [process] entries.
process_requests = []
for process_entry in config.process.entries:
    process_requests.append(
        ProcessRequest.parse(
            process_entry.expression,
            name=process_entry.name,
        )
    )
processes = ProcessSet(tuple(process_requests))

# The generator uses the resolved settings above. Planning expands `p` and `j`
# into concrete subprocesses but does not write or compile a process artifact.
generator = Generator(resolution)
plan = generator.plan(processes, model=model)
print([request.name for request in plan.concrete_processes])

# Generate the artifact at the output location specified in the TOML card.
assert config.generation.output is not None
result = generator.generate(
    processes,
    config.generation.output,
    model=model,
    mode=config.generation.mode,
)
print(result.output)
```

Importing `pyamplicol` does not import Symbolica or execute UFO modules.
Serialized JSON is therefore the preferred portable model input. A UFO
directory is also accepted, but loading it executes trusted Python code.

## Runtime

```python
import json
from pathlib import Path

from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > z g g")
runtime.set_model_parameters({"aS": 0.117, "MZ": 91.188})
momenta = json.loads(Path("data/pp_zjj_momenta.json").read_text())

total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
for summed, explicit in zip(total, resolved.total(), strict=True):
    assert abs(summed - explicit) <= 1.0e-12 * max(1.0, abs(summed))
```

At leading color, resolved values have shape `(point, helicity, color flow)`.
At NLC/full, color is contracted and the final dimension has length one.
Selectors use stable IDs from `runtime.physics`; color-flow selection is valid
only for leading-color artifacts. CLI `--color-flow` also accepts a one-based
ordinal, so `--color-flow 1` selects the first flow advertised by the artifact.
The default LC recurrence layout is `topology-replay`, optimized for a selected
flow with a helicity sum. Use `--lc-flow-layout all-flow-union` when the hot
workload sums all flows at a runtime-selected helicity. The union layout is
rejected for NLC/full and for LC requests with generation-selected color
sectors, generation-selected helicities, or truncated color coverage.
When an axis was not fixed during generation, the same artifact supports a
different physical selector for every point. Rusticol stably groups mixed
selectors before evaluator dispatch, leaves already pooled input in place, and
scatters results back to caller order:

```python
mixed = runtime.evaluate(
    momenta,
    color_flow_by_point=[
        runtime.physics.color_flow_ids[index % 2]
        for index in range(len(momenta))
    ],
)
```

`helicity_by_point` follows the same contract. A batch-global selector and a
per-point selector are mutually exclusive on the same axis. Artifacts generated
with an explicit flow or helicity specialization remain limited to that
generated coverage.

At f64 precision, Rusticol executes either the default direct SymJIT
application or a target-compatible ASM/C++ compiled evaluator without importing
Symbolica or consulting its runtime license state. SymJIT is a separate
MIT-licensed dependency. Compiled payloads must match the runtime target triple,
and every recorded CPU feature must be available. Python precision requests
other than 16 use retained Symbolica evaluator state. Native APIs are f64-only.

Process artifacts are executable inputs. Manifest hashes detect accidental
modification but do not establish who produced an artifact. Generate artifacts
yourself or load them only from a trusted source.

JIT evaluator state is stored in one indexed `evaluators.pacbin` file. Artifact
inspection reports physical files, total on-disk size, and logical evaluator
member count separately; Rusticol maps members directly from the container
instead of materializing thousands of small files.

## Generated APIs

Generation emits one root `API/` directory, shared by every process in the
artifact:

```text
artifacts/pp_zjj/API/
  validation_points.dat
  python/check_standalone.py
  rust/check_standalone.rs
  cpp/check_standalone.cpp
  fortran/check_standalone.f90
```

All four drivers select a concrete process, accept a JSON model-parameter card
and direct overrides, evaluate resolved components, explicitly sum them, and
compare with the optimized total:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > z g g' --set-parameter aS 0.117 0 --json

make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > z g g" --set-parameter aS 0.117 0 --precision 16 --json'
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
```

The generated Rust 2021 program includes the wheel-shipped safe wrapper located
by `rusticol-config --rust-source` and links the installed Rusticol C ABI with
`rusticol-config --rustflags`. Its Makefile uses `rustc`; `rust-script` is an
optional convenience target, not a runtime requirement. No separately
published Rust crate is needed. Rust, C++, and Fortran reject `--precision`
values other than 16; use the generated Python driver for precision-controlled
Symbolica evaluation.

Native build products are placed in a sibling `.pyamplicol-api-build/`
directory, outside the integrity-checked artifact. The installed wheel exposes
the safe Rust source wrapper, C/C++ headers, Fortran module source, static
library, and target-specific link arguments through `rusticol-config`.

## Models And Color

The distribution includes external `sm`, `scalars`, and `scalar_gravity`
models in UFO and serialized JSON forms. `pyamplicol examples copy` materializes
those wheel resources into the copied workspace because the model API currently
accepts filesystem paths rather than packaged model names. The included
`python/copy_packaged_models.py` helper provides the same resource copy for a
separately arranged workspace.

The built-in SM remains available as a compatibility model with its legacy
aliases, isolated under `pyamplicol.models.builtin`. Generic external-model
behavior is driven by compiled particle, color, Lorentz, propagator, coupling,
anti-relation, source, and crossing metadata. External models do not inherit
the complete built-in alias table; define multiparticles explicitly when a
reproducible expansion matters. Release-internal model-hardening gates are
tracked in [Release Status](docs/user/release-status.md).

`color.accuracy = "lc"` stores every physical leading-color flow. `"nlc"` and
`"full"` currently mean the supported contracted SU(3) color calculations,
not arbitrary UFO color groups or representations. Unsupported model features
fail during model preflight with a specific diagnostic. See
[Models And Processes](docs/user/models.md) for the supported subset.

## Symbolica Licensing

pyAmpliCol checks `symbolica.is_licensed()` when generation first needs
Symbolica. Without a valid license it suggests a request command once and, for
eligible non-commercial use, continues in Symbolica's restricted mode with
generation clamped to one process worker and one Symbolica core. Symbolica
limits restricted mode to one instance and one core per device; commercial
work requires the professional license path. pyAmpliCol's clamp does not grant
eligibility or replace Symbolica's terms. Requested and effective settings
remain separately recorded.

```console
pyamplicol request-symbolica-trial-license
pyamplicol request-symbolica-hobbyist-license
export SYMBOLICA_LICENSE='issued-key'
```

`--no-symbolica-suggestion`, `[symbolica] suggest_license = false`, and JSON
output suppress the acquisition reminder and startup banner. These generation
limits do not apply to independent Rusticol handles executing direct SymJIT or
target-compatible ASM/C++ f64 artifacts. See the
[Symbolica Licensing](docs/user/symbolica.md) guide and the
[current upstream terms](https://symbolica.io/docs/get_started.html).

## Documentation And License

- [User Guide](docs/user/index.md)
- [Examples](examples/README.md)
- [Configuration](docs/user/configuration.md)
- [Native SDK](docs/user/native-sdk.md)
- [Release Status](docs/user/release-status.md)

Original pyAmpliCol and Rusticol source is licensed under the BSD Zero Clause
License (`0BSD`). Symbolica and bundled third-party model material retain their
own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
