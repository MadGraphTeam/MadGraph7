---
title: "Configuration"
nav_order: 3
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Configuration

pyAmpliCol uses one typed configuration schema for TOML run cards, direct CLI
options, and Python configuration classes. Unknown fields are rejected rather
than ignored, with a nearest-field suggestion when a likely spelling exists.

The complete field-by-field reference is the shipped
[`examples/all_options.toml`](https://github.com/mg5amcnlo/pyamplicol/blob/main/examples/all_options.toml).
This page focuses on practical configurations.

## Minimal generation card

```toml
schema_version = 1
action = "generate"

[model]
source = "built-in-sm"

[process]
entries = [{ expression = "d d~ > Z g", name = "ddbar_Zg" }]

[color]
accuracy = "lc"

[generation]
output = "artifacts/ddbar_Zg"
```

Run it with:

```console
pyamplicol run.toml
```

Equivalent direct command:

```console
pyamplicol generate 'd d~ > Z g' artifacts/ddbar_Zg \
  --model built-in-sm --color-accuracy lc
```

## Configuration precedence

Values are resolved in this order:

1. schema defaults;
2. TOML card values;
3. dedicated command-line flags;
4. repeated `--set section.field=value` overrides, from left to right;
5. effective license/resource adjustments.

For example:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --set generation.workers=2 \
  --set generation.mode=replace
```

Requested and effective configurations are retained separately in generated
artifact provenance. A restricted Symbolica environment may reduce effective
generation resources without rewriting the requested card.

Inspect resolution without running the action:

```console
pyamplicol config resolve generate_pp_zjj_from_ufo_sm.toml
pyamplicol config resolve generate_pp_zjj_from_ufo_sm.toml --json
```

Create a commented template:

```console
pyamplicol config template run.toml
```

## Paths

Paths in a TOML card are resolved relative to that card, including:

- model sources;
- artifact outputs and inputs;
- model-parameter cards;
- momenta files;
- cache paths.

This makes copied example workspaces movable. Direct CLI paths are resolved
from the current working directory.

Generation output policy is explicit:

| `generation.mode` | Behavior |
| --- | --- |
| `error` | Default; refuse an existing destination. |
| `replace` | Atomically replace the generated artifact. |
| `append` | Add to an existing compatible artifact. |

Use `replace` deliberately while iterating:

```console
pyamplicol generate --card run.toml --set generation.mode=replace
```

## External JSON or UFO model

The primary example uses serialized JSON, which does not execute Python while
loading:

```toml
[model]
source = "models/json/sm/sm.json"
restriction = "default"
simplify = true
cache = true
```

A trusted UFO directory uses the same field:

```toml
[model]
source = "/absolute/path/to/MyUFO"
restriction = "default"
```

UFO modules are Python code and execute during loading. See
[Models and Processes](models-and-processes.md) before using an external model.

## Processes and multiparticles

One card may contain one inclusive multiparticle request or multiple named
requests:

```toml
[process]
entries = [
  { expression = "p p > Z j j" },
  { expression = "u u~ > Z g", name = "uubar_Zg" },
]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

Every model provides the generic `all` multiparticle for valid propagating
physical external particles. It excludes ghosts, Goldstones,
non-propagating records, and auxiliary states. An explicit `all` entry replaces
the default; other user labels are merged over model defaults.

Broad products such as `p p > all all` can expand combinatorially. Prefer a
narrow user-defined label for production scans.

`flavor_scheme`, `max_quark_lines`, and coupling-order settings constrain that
expansion. Coupling-order names come from the selected model: the generic
engine treats them as model-defined filtering and scheduling data and does not
assign arbitrary names fixed QCD or electroweak meanings.

## Color accuracy and LC layout

```toml
[color]
accuracy = "lc"                 # lc, nlc, or full
contraction = "direct"          # direct or symmetric-group-fft
lc_flow_layout = "topology-replay"
```

| Accuracy | Resolved color output |
| --- | --- |
| `lc` | One entry per physical leading-color flow. |
| `nlc` | One contracted color entry per helicity. |
| `full` | One contracted color entry per helicity. |

LC offers two complete-coverage layouts:

| Layout | Best suited to |
| --- | --- |
| `topology-replay` | Default: one selected flow with a helicity sum. |
| `all-flow-union` | All flows with one selected helicity. |

Both preserve all physical flows and helicities for runtime selection.
`all-flow-union` is LC-only and is incompatible with generation-fixed or
truncated color/helicity coverage.

For contracted NLC/full output, `symmetric-group-fft` evaluates the same exact
colour interference as `direct` while Fourier-transforming certified
permutation orbits and retaining all other terms as direct residuals. It is
available for `recurrence` and `on-the-fly`; compiled/eager execution and LC
flows deliberately reject it.

```toml
[color]
accuracy = "full"
contraction = "symmetric-group-fft"

[evaluator]
execution_mode = "recurrence"
```

## Execution mode and evaluator backend

```toml
[evaluator]
execution_mode = "compiled"     # recurrence, compiled, eager, on-the-fly
backend = "jit"                 # jit, asm, cpp
batch_size = 128

[evaluator.jit]
optimization_level = 2
compress = true
```

### Execution modes

| Mode | What is stored in the process artifact |
| --- | --- |
| `recurrence` | Compact current-recursion schedules over prepared local kernels. |
| `compiled` | Process-wide stage evaluators compiled during generation. |
| `eager` | Compact DAG invocation tables over prepared local kernels. |
| `on-the-fly` | A compact process seed; the selected LC query family, or contracted NLC/full family, is constructed on first use. |

`recurrence` is the default. Eager, recurrence, and on-the-fly require a
compatible prepared `.pyamplicol-model` kernel pack; `built-in-sm` selects the
packaged JIT O2 pack automatically. Raw JSON/UFO model IR normally uses
`compiled` unless you first create a prepared bundle.

Recurrence, compiled, and eager may fix
`process.selected_color_sector_ids`, `process.selected_source_helicities`, or
both at generation time when a deliberately specialized artifact is useful.
Omit them to retain reusable runtime selectors. On-the-fly keeps selection at
runtime instead: its last warmed family is retained until another selector is
requested.

For reusable contracted recurrence artifacts, generation persists one shared
all-helicity physical-colour plan plus an exact per-helicity dispatch sidecar.
Cold selector binding copies only the dependency-closed active row groups;
warmed evaluation does not scan helicity-support masks. This companion belongs
to recurrence execution. On-the-fly instead constructs the requested family
from its compact process seed on first use and caches that family; compiled and
eager artifacts retain their own execution layouts.

On-the-fly is native binary64 only. LC retains physical flow and helicity
selection without materializing either LC layout. NLC and full colour expose
one contracted color component, accept helicity selection, and reject LC-flow
selectors. Their cold family is correctness-oriented and may become
impractical at high multiplicity. For on-the-fly generation,
`evaluator.optimization.cores` requests query-construction workers for the cold
warm-up; it does not make steady-state numerical evaluation use that many
threads. See [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
for the distinct lifecycle of each mode.

### Backends

| Backend | Notes |
| --- | --- |
| `jit` | Default direct SymJIT application. Compiled O1/O2 artifacts are portable on supported 64-bit little-endian hosts; prepared packs use exact O2. |
| `asm` | Target-native Symbolica assembly evaluator. |
| `cpp` | Target-native generated C++ evaluator. |

Explicit JIT O0/O3 and ASM/C++ artifacts remain target-native. See
[Artifacts and Portability](artifacts-and-portability.md).

## Generation validation

```toml
[generation.validation]
enabled = true
samples = 2
seed = 12345
relative_tolerance = 1e-12
absolute_tolerance = 1e-300
post_build_validation = false
```

Generation validation checks symbolic/numerical construction before
publication. Structural artifact checks always run. The optional
`post_build_validation` reopens the finished artifact and evaluates it again;
it is disabled by default because its cost can dominate large process builds.
Enable it when you explicitly want that additional runtime smoke:

```console
pyamplicol generate --card run.toml \
  --set generation.validation.post_build_validation=true
```

## Numerical current reuse

Certified current-relation reuse is enabled by default. It prefers exact
structural proofs and may apply independently verified high-precision equal,
opposite, or zero relations. The artifact records the evidence, tolerances,
probe derivation, replay identity, and whether a structural proof was present.

This configurable discovery pass applies to recurrence, compiled, and eager
generation. On-the-fly uses its compact source projection instead and does not
run relation discovery.

`diagnostic` records deterministic candidates and their exact replay without
changing the generated evaluator. `certified-reuse` may apply a relation only
after its independent verification pass succeeds; it emits one warning when
an applied relation has numerical evidence but no structural proof. Malformed,
non-finite, unstable, or stale evidence fails closed and is never reused.

Keep the unoptimized path for a comparison with:

```console
pyamplicol generate --card run.toml --no-numerical-current-reuse
```

Or in TOML:

```toml
[generation.relation_discovery]
mode = "off"                    # off, diagnostic, certified-reuse
```

The exhaustive probe, precision, seed, and tolerance fields are documented in
`all_options.toml`.

## Evaluation card

```toml
schema_version = 1
action = "evaluate"

[evaluation]
artifact = "artifacts/pp_zjj"
process = "d d~ > g z g"
precision = 16
resolved = true
model_parameters = "data/model_parameters.json"
momenta = "data/pp_zjj_momenta.json"

[output]
format = "human"                # programmatic/config provenance default
color = "auto"                  # auto, always, never
progress = "off"                # auto, tty, log, off
```

`helicity_ids` and `color_flow_ids` may be added under `[evaluation]`. Empty
lists mean complete retained coverage.

The model-parameter card is a flat JSON object:

```json
{
  "aS": 0.117,
  "MZ": 91.1876,
  "complex_parameter": [1.0, -0.25]
}
```

Values must be finite real numbers or `[real, imaginary]` pairs. Direct
parameter overrides are applied after the card and win atomically.

## Profiling card

Run cards retain `action = "benchmark"`; the preferred direct CLI spelling is
`pyamplicol profile`:

```toml
action = "benchmark"

[evaluation]
artifact = "artifacts/pp_zjj"
process = "d d~ > g z g"
momenta = "data/pp_zjj_momenta.json"

[benchmark]
target_runtime = 1.0
batch_size = 128
precision = 16
warmup_runs = 2
minimum_samples = 5
color_flow_ids = ["1"]
```

With both benchmark selector lists empty, LC profiling infers the stored
layout's deterministic hot workload. Explicit subsets and selected-axis lists
override that default; a complete summed-axis list is normalized to equivalent
omission. A valid shape outside the optimized layout emits at most one pre-loop
warning per loaded process.

See [Runtime and Selectors](runtime-and-selectors.md) for evaluation and profiling
semantics.

## Output and progress

```toml
[output]
format = "human"
color = "auto"
progress = "auto"
log_level = "info"
```

- The `pyamplicol` CLI always produces aligned colored terminal tables unless
  the invocation includes `--json`. The run-card `format` field remains part
  of the typed configuration used by the Python API and recorded provenance;
  it does not silently switch CLI stdout to JSON.
- `pyamplicol ... --json` reserves standard output for stable, uncolored,
  machine-readable data.
- `auto` progress uses a TTY progress bar interactively and rate-limited log
  messages otherwise.
- `Ctrl-C` during profiling retains completed timing blocks and marks the result
  partial.

## Resolve configuration from Python

```python
from pathlib import Path

from pyamplicol.config import resolve_config

resolution = resolve_config(
    {
        "schema_version": 1,
        "action": "generate",
        "model": {"source": "models/json/sm/sm.json"},
        "process": {"entries": [{"expression": "d d~ > z g"}]},
        "generation": {"output": "artifacts/builtin_ddbar_to_zg"},
    },
    base_dir=Path("pyamplicol-examples"),
    overrides=("generation.workers=2",),
)
print(resolution.effective.generation.output)
```

For a mapping, relative paths use `base_dir`; without it they use the current
working directory. Values supplied through `--set` or `overrides=` use TOML
syntax, so quote strings containing spaces.

## Further reading

- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [LC workloads and execution modes](lc-workloads-and-execution-modes.md)
- [Models and Processes](models-and-processes.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Exhaustive current schema](https://github.com/mg5amcnlo/pyamplicol/blob/main/examples/all_options.toml)
