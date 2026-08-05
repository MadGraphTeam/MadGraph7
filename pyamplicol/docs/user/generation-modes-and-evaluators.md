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

| Mode | What generation writes | Best starting point | Model requirement |
| --- | --- | --- | --- |
| `recurrence` | Compact current-recursion schedules over prepared local kernels | Default; scalable general-purpose generation and runtime | A compatible prepared model bundle |
| `compiled` | Process-local stage DAG evaluators | Self-contained compiled processes and direct backend experiments | Portable model IR is sufficient |
| `eager` | Compact process DAG invocation tables over prepared local kernels | Compare process DAG scheduling while reusing model kernels | A compatible prepared model bundle |

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

### Recurrence

Recurrence is the global default. Generation writes compact current schedules
and references only the prepared model kernels needed by the process. It is the
recommended first choice for the wheel-owned built-in Standard Model:

```console
pyamplicol generate "u u~ > Z g g g" artifacts/uubar_z3g_recurrence \
  --model built-in-sm
```

The installed wheel supplies a portable built-in-SM JIT-O2 prepared bundle, so
no explicit pack path is needed in this case. A raw external JSON or UFO model
does not have prepared kernels; compile a `.pyamplicol-model` bundle first.

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

Prepared JIT kernel bundles used by recurrence and eager always use O2 because
that is their cross-architecture storage contract. Process-local compiled JIT
O1 and O2 artifacts are marked `portable-64le` only when every execution leaf
is authenticated as compatible O1/O2 JIT state and no target-specific
capability is present.

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
expressions, and one compiled local-kernel backend. Recurrence and eager require
this prepared boundary.

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
back from recurrence/eager to compiled mode.

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

## Decision guide

| Goal | Suggested choice |
| --- | --- |
| First built-in-SM artifact | Recurrence + JIT O2 (defaults) |
| Raw JSON/UFO model without a prepared pack | Compiled mode, or prepare a pack first |
| Reuse prepared kernels with process tables | Eager + the pack's backend |
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
- [Symbolica and Licensing](symbolica-and-licensing.md)
