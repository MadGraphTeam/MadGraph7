---
title: "Generation Modes and Evaluators"
nav_order: 3
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Generation Modes and Evaluators

pyAmpliCol makes two independent choices when it builds a process artifact:

1. the **execution mode** describes how process-wide work is organized;
2. the **evaluator backend** describes how local symbolic expressions become
   executable numerical kernels.

Keeping these axes separate makes it possible to compare algorithms without
changing the model, process, color approximation, or public runtime API.

## Choose an execution mode

Set the mode in TOML:

```toml
[evaluator]
execution_mode = "recurrence"
backend = "jit"
```

or on the command line:

```console
pyamplicol generate "u u~ > Z g g" artifacts/uubar_zgg \
  --model built-in-sm \
  --execution-mode recurrence \
  --backend jit
```

Recurrence, compiled, and eager can deliberately specialize generation to one
or more flow IDs, helicity IDs, or both with
`process.selected_color_sector_ids` and
`process.selected_source_helicities`. This reduces the reusable coverage of
the resulting artifact and is useful when the same narrow workload will be
run repeatedly. Omit those fields when runtime selection is wanted.
On-the-fly does not use generation-time specialization: `warm_up(...)` builds
the runtime-selected family and retains only the most recently selected one.

| Mode | What generation writes | Best starting point | Model requirement |
| --- | --- | --- | --- |
| `recurrence` | Compact current-recursion schedules over prepared local kernels | Default; scalable general-purpose generation and runtime | A compatible prepared model bundle |
| `compiled` | Process-local stage DAG evaluators | Self-contained compiled processes and direct backend experiments | Portable model IR is sufficient |
| `eager` | Compact process DAG invocation tables over prepared local kernels | Compare process DAG scheduling while reusing model kernels | A compatible prepared model bundle |
| `on-the-fly` | A compact process seed; the requested query family is built on first use | High-multiplicity LC single-flow/helicity-sum work with an explicit cold warm-up | A compatible prepared model bundle |

### Recurrence

Recurrence is the global default. Generation writes compact current schedules
and references only the prepared model kernels needed by the process. It is the
recommended first choice for the wheel-owned built-in Standard Model:

```console
pyamplicol generate "u u~ > Z g g g" artifacts/uubar_z3g_recurrence \
  --model built-in-sm
```

The installed wheel supplies portable JIT-O2 prepared bundles for both
`built-in-sm` and `built-in-sm-heft`, so no explicit pack path is needed for
either model. A raw external JSON or UFO model does not have prepared kernels;
compile a `.pyamplicol-model` bundle first.

Recurrence has two runtime tiling controls:

```toml
[evaluator.recurrence]
point_tile_size = 1024
workspace_mib = 256
```

The runtime may reduce the effective point tile to stay within the configured
workspace. It does not increase the requested tile.

### Compiled

Compiled mode lowers a process-wide DAG into evaluator stages while generating
the artifact. It can start from portable compiled model IR and is therefore the
primary shipped JSON-model example:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

That card compiles seven tree-level representatives for `p p > Z j j` using
process-local JIT O2. The resulting all-JIT-O2 process artifact is portable
across the supported 64-bit little-endian release platforms.

Select compiled mode explicitly when overriding another card:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml \
  --execution-mode compiled \
  --set generation.output=artifacts/qq_z6g_compiled_override
```

Always use a different output when comparing modes; a process artifact is an
immutable executable input, not a directory into which unrelated plans should
be mixed.

### Eager

Eager mode writes compact DAG invocation tables and applies prepared local
kernels directly at runtime:

```console
pyamplicol generate "u u~ > Z g g g" artifacts/uubar_z3g_eager \
  --model built-in-sm \
  --execution-mode eager
```

Its workspace controls mirror recurrence:

```toml
[evaluator.eager]
point_tile_size = 1024
workspace_mib = 256
```

An eager process artifact is standalone: it carries the referenced prepared
kernels and compact invocation data. The model bundle used during generation
is not needed to evaluate that artifact later.

### On-the-fly

On-the-fly (OTF) stores a compact process seed and the referenced prepared
kernels instead of materializing a reusable process schedule during
generation. The first explicit `warm_up(...)` or evaluation constructs the
requested query family; later evaluations on the same loaded handle reuse it.
Selecting a different family replaces the previous one.

This trade is designed around LC with one selected flow summed over all
helicities. The same artifact can also build the all-flow sum for one selected
helicity, so OTF does not use the materialized
`topology-replay`/`all-flow-union` layout setting. Contracted NLC and full
colour are available as low-multiplicity correctness capabilities, but their
cold family can grow too quickly for practical high-multiplicity work.

OTF evaluation is native binary64 only (`precision=16`) and requires a
prepared model bundle; `built-in-sm` selects the wheel-owned JIT-O2 bundle.
Plan its cold stage explicitly with exactly one phase-space point:

```python
result = runtime.warm_up(
    (point,),
    precision=16,
    color_flows=(runtime.physics.color_flows[0],),
    progress=progress,
)
```

The optional progress observer reports process preparation, query-family
construction, family finalization, first evaluation, elapsed time, workers,
and resident memory where available. `runtime.clear()` returns a Python OTF
handle to cold state while retaining the loaded artifact and current model
parameters. Native callers close and reload the handle. See
[LC workloads and execution modes](lc-workloads-and-execution-modes.md#the-otf-warm-state-lifecycle)
for the cache lifecycle and all five API spellings.

## Choose an evaluator backend

| Backend | Output | Portability and runtime |
| --- | --- | --- |
| `jit` | Direct SymJIT application | Compiled O1/O2 state can be portable; prepared packs use exact O2; f64 execution is Symbolica-independent |
| `cpp` | Generated and compiled C++ evaluator library | Target-specific executable payload; f64 runtime is Symbolica-independent |
| `asm` | Symbolica assembly evaluator library | Target-specific executable payload; f64 runtime is Symbolica-independent |

### JIT

The public JIT defaults are optimization level 2 with compression enabled:

```toml
[evaluator.jit]
optimization_level = 2
compress = true
```

Prepared JIT kernel bundles used by recurrence, eager, and on-the-fly always
use O2 because that is their cross-architecture storage contract. Process-local
compiled JIT O1 and O2 artifacts are marked `portable-64le` only when every
execution leaf is authenticated as compatible O1/O2 JIT state and no
target-specific capability is present.

Explicit JIT O0 or O3 remains available for host-specific experiments:

```console
pyamplicol generate --card qq_z6g_compiled_jit_o3.toml
```

Those artifacts are target-native. Copying one to another architecture is
expected to fail compatibility checks rather than silently relower it.

Compression factors repeated complex instruction sequences into internal
applets without changing the evaluator ABI or numerical contract. Disable it
only for a deliberate comparison:

```console
pyamplicol generate ... --no-jit-compress
```

### C++

C++ is a process-local compiled backend:

```toml
[evaluator]
execution_mode = "compiled"
backend = "cpp"

[evaluator.cpp]
optimization = "O3"
native_arch = false
```

Portable C++ code generation is the default. `native_arch = true` opts into
host-native instructions and records the required CPU features in the
artifact. Loaders reject an incompatible target before reading executable
state. Extra compiler flags use a restricted allowlist so an unrecorded ISA
requirement cannot be introduced accidentally.

### Assembly

The assembly backend is also target-native and uses the same public Rusticol
runtime surface. Like C++, it is primarily a compiled-mode backend. Generation
requires Symbolica; f64 evaluation of the completed compatible artifact does
not.

## Prepared model bundles

A JSON file such as `models/json/sm/sm.json` is portable model IR. It contains
model structure and expressions but no ready local-kernel backend.

A file ending in `.pyamplicol-model` is a prepared bundle: model IR, exact
expressions, and one compiled local-kernel backend. Recurrence, eager, and
on-the-fly require this prepared boundary.

Prepare an external JIT-O2 bundle:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit \
  --jit-optimization-level 2 \
  --jit-compress
```

Use it in recurrence or eager generation:

```console
pyamplicol generate "d d~ > z g g g" artifacts/ddbar_z3g_ufo \
  --model models/ufo-sm-jit-o2.pyamplicol-model \
  --execution-mode recurrence
```

A missing or incompatible pack fails explicitly; configuration never falls
back from recurrence/eager/on-the-fly to compiled mode.

## Color accuracy and LC flow layout

Execution mode is independent of color accuracy:

| Accuracy | Runtime color axis |
| --- | --- |
| `lc` | One physical leading-color flow per resolved color entry |
| `nlc` | One contracted color entry per helicity |
| `full` | One contracted color entry per helicity |

LC also has two complete-coverage layouts:

| Layout | Optimized workload |
| --- | --- |
| `topology-replay` | Default: one runtime-selected flow, summed helicities |
| `all-flow-union` | All flows, one runtime-selected helicity |

Choose union flow explicitly:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --lc-flow-layout all-flow-union \
  --set generation.output=artifacts/uubar_z6g_all_flow_union
```

Both layouts retain all physical LC flows and helicities and support runtime
selectors. `all-flow-union` is LC-only and is incompatible with a
generation-specialized flow/helicity request or truncated color coverage.
OTF retains the same physical LC selector choices but constructs the requested
family at runtime and therefore ignores this materialized-layout setting.

## Validation and current reuse

Generation has three distinct checks; they should not be confused:

1. **Artifact writing** always validates the schema, confined references,
   declared payload sizes, and digests.
2. **Current relation discovery** uses deterministic high-precision probes and
   independent verification to apply certified equal, opposite, or zero-current
   reuse when structural proof is unavailable.
3. **Post-build validation** optionally reopens the completed artifact and
   compares native f64 optimized and resolved evaluation.

Post-build validation is off by default because it does not change the written
artifact and can be disproportionately expensive for large resolved axes.
Enable it when an immediate runtime smoke is useful:

```console
pyamplicol generate ... --post-build-validation
```

Ordinary generation validation defaults to two deterministic samples. Current
relation discovery defaults to `certified-reuse` and remains independently
high precision. To retain the unoptimized current schedule for a comparison:

```console
pyamplicol generate ... --no-numerical-current-reuse
```

This changes optimization, not the expected physics result.
OTF uses its compact source projection and does not run the configurable
relation-discovery pass.

## Three matched examples

The copied example workspace contains three cards for the same
`u u~ > Z + 6g` LC topology-replay workload:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml
pyamplicol profile  --card qq_z6g_recurrence_jit_o2.toml

pyamplicol generate --card qq_z6g_compiled_jit_o3.toml
pyamplicol profile  --card qq_z6g_compiled_jit_o3.toml

pyamplicol generate --card qq_z6g_eager_jit_o2.toml
pyamplicol profile  --card qq_z6g_eager_jit_o2.toml
```

These are intentionally explicit comparison cards: recurrence JIT O2, compiled
JIT O3, and eager JIT O2. Generation and profiling can be substantial for six
final-state gluons; use the primary Z+jet example for a quick functional
check.

The packaged OTF walkthrough uses `p p > Z j j`, an explicit one-point
warm-up, and the same public profiler:

```console
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py
pyamplicol profile --card otf_pp_zjj.toml
```

## Decision guide

| Goal | Suggested choice |
| --- | --- |
| First built-in-SM artifact | Recurrence + JIT O2 (defaults) |
| Raw JSON/UFO model without a prepared pack | Compiled mode, or prepare a pack first |
| Reuse prepared kernels with process tables | Eager + the pack's backend |
| Keep a high-multiplicity LC artifact compact and repeatedly run one selected flow | On-the-fly + an explicit one-point warm-up |
| Cross-architecture release-host movement | Compiled all-JIT O1/O2 artifact, or eager/recurrence with a prepared JIT O2 pack |
| One-flow/helicity-sum LC runtime | `topology-replay` |
| All-flow/single-helicity LC runtime | `all-flow-union` |
| Host-specific performance experiment | Explicit compiled JIT O3, C++, ASM, or C++ `native_arch` |
| Fast confidence after generation | Normal generation validation; enable post-build validation only when wanted |

## See also

- [Configuration](configuration.md)
- [Models and Processes](models-and-processes.md)
- [Artifacts and Portability](artifacts-and-portability.md)
- [Profiling and Benchmarking](profiling-and-benchmarking.md)
- [LC workloads and execution modes](lc-workloads-and-execution-modes.md)
- [Symbolica and Licensing](symbolica-and-licensing.md)
