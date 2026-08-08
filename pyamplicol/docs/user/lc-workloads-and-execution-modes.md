---
title: "LC workloads and execution modes"
nav_order: 1
parent: "Get started: a gentle walkthrough"
---
<!-- SPDX-License-Identifier: 0BSD -->

# LC workloads and execution modes

This page helps with two choices that are easy to mix up:

1. **Which part of the leading-colour result will be evaluated most often?**
   This determines the LC flow layout for a materialized process output.
2. **When should pyAmpliCol construct the numerical representation?** This
   determines the execution mode.

Neither choice changes the requested physics. The LC flow layout reorganizes
the same leading-colour components, while the execution mode reorganizes when
the same current construction is prepared. Use [the gentle
walkthrough](gentle-walkthrough.md) first if “process output,” “flow,” or
“selector” is unfamiliar.

## Two useful slices through an LC result

At one phase-space point, imagine the leading-colour result as a rectangular
table. Its rows are the physical helicity configurations \(h\), and its columns
are the physical LC flows \(f\). A runtime request may sum either axis:

| Workload | Quantity requested repeatedly | Typical physics use |
| --- | --- | --- |
| One flow, helicity sum | Fix one \(f\), sum every retained \(h\) | A flow-resolved calculation or a colour-flow sampling/integration step |
| All flows, one helicity | Fix one \(h\), sum every retained \(f\) | A helicity-resolved calculation or repeated evaluations for sampled helicities |

Asking for neither selector sums both axes. Asking for a set of IDs sums the
selected subset. The two layouts below are performance arrangements for these
requests; they do not add or remove LC terms when their coverage is complete.

### `topology-replay`: one flow, summed helicities

`topology-replay` is the default. It retains complete physical flow and
helicity coverage, but arranges the generated construction so that the compact
topology can be replayed for the requested flow. It is therefore the natural
layout when each evaluation normally selects one flow and omits the helicity
selector, meaning “sum all helicities.”

Use it when:

- a downstream colour sampler asks for one LC flow at a time;
- the matrix element should remain reusable for another flow later; or
- the main observable is a flow-resolved contribution summed over spin.

The explicit card setting is:

```toml
[color]
accuracy = "lc"
lc_flow_layout = "topology-replay"
```

Because it is the default, omitting `lc_flow_layout` has the same effect. A
direct generation override is:

```console
pyamplicol generate --card run.toml \
  --color-accuracy lc \
  --lc-flow-layout topology-replay
```

Once generated, select the first physical flow shown by `inspect` and sum all
helicities by omitting `--helicity`:

```console
pyamplicol evaluate --card run.toml --color-flow 1
pyamplicol profile --card run.toml --color-flow 1
```

CLI flow numbers are **one-based**. A stable flow ID printed by `inspect`, such
as `flow:...`, may be used in place of `1`.

### `all-flow-union`: all flows, one helicity

`all-flow-union` retains the same complete axes, but builds one shared
cross-flow recurrence. It is the complementary layout for repeatedly selecting
one helicity and omitting the flow selector, meaning “sum all physical LC
flows.”

Use it when:

- a helicity sampler asks for one physical helicity at a time;
- every LC flow is needed for each selected helicity; or
- the main observable is a helicity-resolved LC result with colour summed.

Set it in the card:

```toml
[color]
accuracy = "lc"
lc_flow_layout = "all-flow-union"
```

or override the card during generation:

```console
pyamplicol generate --card run.toml \
  --color-accuracy lc \
  --lc-flow-layout all-flow-union
```

Then select one stable helicity ID from `inspect` and omit `--color-flow`:

```console
pyamplicol evaluate --card run.toml \
  --helicity 'h:-1,+1,-1,+1,-1'
pyamplicol profile --card run.toml \
  --helicity 'h:-1,+1,-1,+1,-1'
```

The displayed helicity length depends on the number of external particles, so
copy an ID from the process output rather than copying the illustrative ID
above into an unrelated process.

`all-flow-union` is LC-only. It is available for recurrence, eager, and
compiled outputs. OTF does not materialize either LC layout; its runtime
selector directly determines which query family is constructed.

### Pick the layout from the repeated workload

For a complete materialized output, either layout can still evaluate all
flows, all helicities, or a supported selected subset. “Optimized for” does not
mean “only capable of.” It means that the stored current construction is
organized around one common loop direction.

| If most calls request... | Start with... |
| --- | --- |
| One flow and every helicity | `topology-replay` |
| Every flow and one helicity | `all-flow-union` |
| The completely summed LC result | Benchmark both layouts for the real process and batch size |
| A mixture that changes during a run | Keep complete coverage, then profile the dominant request |

The packaged cards provide a larger-multiplicity comparison without hiding
the settings:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile --card benchmark_z6g_all_flows_single_helicity.toml
```

These are performance examples, so generation can take appreciably longer
than the small `p p > Z j j` tour.

## Runtime selection in Python and native programs

The layout is fixed when the process output is generated. Python and native
programs load that output and select physical IDs at runtime; they cannot
change its generated layout.

### Typed Python

The typed objects in `runtime.physics` avoid copying IDs by hand:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
flow = runtime.physics.color_flows[0]
helicity = runtime.physics.helicities[0]

# One physical flow, summed over every helicity.
flow_values = runtime.evaluate(points, color_flows=(flow,), precision=16)

# One physical helicity, summed over every LC flow.
helicity_values = runtime.evaluate(points, helicities=(helicity,), precision=16)
```

Here `points` is a batch with shape
`[point][external particle][E, px, py, pz]`. The same calls accept stable string
IDs instead of the typed objects. The global selector arguments are sequences:
passing one entry selects one component, passing several selects and sums that
subset, and passing `None` sums the complete retained axis.

Typed generation uses the same schema as a TOML card. For example, this is the
complete choice of LC layout and execution mode in Python:

```python
from pyamplicol import Generator, ModelSource
from pyamplicol.config import resolve_config

settings = resolve_config(
    {
        "schema_version": 1,
        "action": "generate",
        "color": {
            "accuracy": "lc",
            "lc_flow_layout": "all-flow-union",
        },
        "evaluator": {"execution_mode": "recurrence"},
    }
)
Generator(settings).generate(
    "d d~ > g z g",
    "artifacts/ddbar_to_zgg_union",
    model=ModelSource.built_in_sm(),
)
```

Replace `all-flow-union` by `topology-replay`, or `recurrence` by `compiled`,
`eager`, or `on-the-fly` as described below. For OTF, omit
`lc_flow_layout`: no materialized flow layout is used.

### C11

The C ABI uses a null pointer with a zero count for an unselected axis. Given
stable `flow_id` and `helicity_id` strings, the two LC calls are:

```c
const char *one_flow[] = {flow_id};
int status = rusticol_runtime_evaluate_selected_f64(
    handle, momenta, momentum_count, point_count,
    NULL, 0, one_flow, 1,
    NULL, 0, NULL, 0,
    values, point_count);

const char *one_helicity[] = {helicity_id};
status = rusticol_runtime_evaluate_selected_f64(
    handle, momenta, momentum_count, point_count,
    one_helicity, 1, NULL, 0,
    NULL, 0, NULL, 0,
    values, point_count);
```

The arguments after the global string IDs are optional per-point **zero-based**
selector-index arrays. Do not supply both the global and per-point form for the
same axis. End the handle lifetime with `rusticol_runtime_free(handle)`.

### C++17

The C++ wrapper uses empty vectors for an unselected axis:

```cpp
auto flow_values = runtime.evaluate_selected(
    momenta, point_count, {}, {flow_id});
auto helicity_values = runtime.evaluate_selected(
    momenta, point_count, {helicity_id}, {});
```

The final two optional arguments, omitted here, are the per-point helicity and
flow index vectors. The `rusticol::Runtime` destructor releases the handle.

### Fortran 2008

In the Fortran module, omit the optional axis that should be summed:

```fortran
character(len=128) :: flow_ids(1), helicity_ids(1)
real(c_double), allocatable :: values(:)

flow_ids(1) = flow_id
call runtime%evaluate_selected(momenta, point_count, values, &
    color_ids=flow_ids, ierr=ierr)

helicity_ids(1) = helicity_id
call runtime%evaluate_selected(momenta, point_count, values, &
    helicity_ids=helicity_ids, ierr=ierr)
```

Use `call runtime%close()` when the loaded process is no longer needed.

### Rust 2021

The installed safe wrapper represents omitted axes with `Selectors::all()`:

```rust
let one_flow = Selectors::all().with_colors([flow_id]);
let flow_values = runtime.evaluate_selected_f64(
    &momenta, point_count, &one_flow, None, None,
)?;

let one_helicity = Selectors::all().with_helicities([helicity_id]);
let helicity_values = runtime.evaluate_selected_f64(
    &momenta, point_count, &one_helicity, None, None,
)?;
```

The two `None` arguments are the optional per-point zero-based helicity and
flow index slices. Dropping `runtime` releases the native handle.

Complete load, error-handling, metadata, memory-layout, and compiler examples
are in the [Native APIs guide](native-apis.md). Stable IDs can be read from that
metadata or copied from `pyamplicol inspect`; do not infer IDs from their
position except where the CLI explicitly accepts a one-based ordinal.

## Generation specialization is not runtime selection

Complete coverage gives one process output reusable runtime selectors. A
runtime selection chooses among components that are already represented by
that output and can change from call to call.

Generation specialization instead omits unneeded work while constructing a
recurrence, eager, or compiled output. The relevant card fields are:

```toml
[process]
# Flow-specialized packaged example: u u~ > Z g g g g g g.
reference_color_order = [2, 4, 5, 6, 7, 8, 9, 1, 3]
selected_color_sector_ids = [0]
```

or, in the separate helicity-specialized card:

```toml
[process]
# Helicity-specialized packaged example: u u~ > Z g g g g g g.
selected_source_helicities = { "1" = -1, "2" = 1, "3" = -1, "4" = 1, "5" = -1, "6" = 1, "7" = -1, "8" = 1, "9" = -1 }
```

These are low-level generation inputs. `selected_color_sector_ids` are
zero-based construction-sector IDs, not the one-based physical flow ordinals
accepted by `--color-flow`. A specialized-away flow or helicity cannot be
restored by a later runtime call. Use the maintained
`benchmark_z6g_generation_specialized_*.toml` cards as worked examples instead
of guessing sector IDs or a reference order.

For reusable compiled and eager outputs, do not combine generation
specialization with `all-flow-union`; generation rejects that combination.
Recurrence can encode a valid specialized union construction, but its excluded
selectors are still absent, so it should be treated as an advanced,
non-reusable output. `process.max_color_sectors` is incompatible with
`all-flow-union` in every materialized mode.

OTF takes the opposite approach: it rejects the generation-specialization
fields and retains complete compact selector metadata. The first warm-up or
evaluation then constructs only the requested runtime family.

## Four ways to organize the same calculation

The execution mode is selected in `[evaluator]`:

```toml
[evaluator]
execution_mode = "recurrence" # or compiled, eager, on-the-fly
```

or on the generation command line:

```console
pyamplicol generate --card run.toml --execution-mode recurrence
```

| Mode | What generation stores | What happens after loading | Natural starting point |
| --- | --- | --- | --- |
| `recurrence` | Prepared local model kernels and compact current-recursion schedules | The stored schedules are replayed for the selected point batch | General-purpose default with a prepared model |
| `compiled` | Process-wide compiled stage evaluators | The process-local compiled stages evaluate the batch | Larger generation investment and widest process-local optimizer view |
| `eager` | Prepared local kernels and compact invocation/finalization tables | The tables call the local kernels directly | Usually the quickest materialized output to generate and load |
| `on-the-fly` | Prepared local kernels and a compact process seed | The first selected request constructs a query family, which is then retained in memory | Compact high-multiplicity LC output when one selected family dominates |

The choice changes where work and storage occur, not the meaning of LC, NLC,
or full colour. Results for the same supported physics selection should agree
across modes within the stated numerical tolerance.

### Recurrence

Recurrence is the default and the closest stored representation to the
physical current recursion. Generation prepares compact authenticated current
schedules; numerical evaluation replays them through the model's local
kernels. It supports LC, NLC, and full colour, reusable runtime selectors, and
optional generation specialization. Point tiling bounds reusable workspace for
large batches.

Choose recurrence first when a prepared model is available and the process
output should remain broadly reusable.

### Compiled

Compiled mode lowers the complete process DAG into process-wide stage
evaluators during generation. That costs more generation work and gives the
optimizer the broadest view of one process. Unlike recurrence, eager, and OTF,
it can start from raw JSON/UFO model IR without a prepared local-kernel pack.
It supports LC, NLC, full colour, reusable selectors when coverage is complete,
and generation-specialized outputs.

Choose it when process-local compilation is desirable, when comparing a
specialized baseline, or when no prepared model bundle exists.

### Eager

Eager mode stores compact tables of kernel invocations, finalizations, and
closures around the prepared model's local kernels. It moves less structural
work into a process-wide compiler and is usually the quickest materialized
mode to generate and load. Like recurrence and compiled mode, it supports LC,
NLC, full colour, complete runtime selection, and optional generation
specialization.

Choose it when fast materialized generation/load matters and a prepared model
is available.

### On-the-fly

OTF stores a compact process seed rather than every reusable process schedule.
It still needs a prepared model. For LC it can express both selector families:
one flow summed over helicities, or all flows summed at one helicity. It does
not use the `topology-replay`/`all-flow-union` generation setting because the
runtime request itself defines the family.

OTF is native `f64` only (`precision=16`). Its practical performance focus is
LC with one selected flow and a helicity sum. Contracted NLC and full-colour
families are available for low-multiplicity correctness work, but their cold
family grows rapidly and is not a practical high-multiplicity route.

## The OTF warm-state lifecycle

OTF moves a potentially expensive operation from generation to the first use
of a selected family. Treat warm-up as part of job planning, not as a small
extra numerical sample.

### Cold, warm, replaced, and cleared

One loaded OTF runtime follows this lifecycle:

| Action | State afterwards |
| --- | --- |
| Load an OTF process | Cold: compact seed loaded, no selected family retained |
| `warm_up` selection A | A is retained, and exactly one `f64` point has also been evaluated |
| Evaluate selection A again | Reuses A's structural family; ordinary numerical execution remains |
| Successfully warm or evaluate selection B | B replaces A; only the most recent selected family is retained |
| Change model parameters, then evaluate the retained selection | The same structure is retained, and its numerical workspace is refreshed with the latest parameter values before execution |
| Python `runtime.clear()` | Cold again while the process output and current model parameters remain loaded |
| Close/free/drop the native handle | All state owned by that handle is released; a later load starts cold |

The family cache is **handle-local and last-family-only**. A second
`pyamplicol` CLI command loads another handle, so it does not inherit warm state
from the first command. Keep one Python or native runtime alive when
same-selection reuse matters.

There is no public knob for retaining an arbitrary number of OTF families and
no cache-size setting. `evaluator.optimization.cores` controls the requested
cold query-construction worker count; it does not set the number of retained
families or promise that warmed numerical evaluation uses that many threads.

### Explicit warm-up means exactly one binary64 point

Python exposes the structural operation only on OTF runtimes:

```python
flow = runtime.physics.color_flows[0]
one_point = (point,)

result = runtime.warm_up(
    one_point,
    precision=16,
    color_flows=(flow,),
    progress=progress,
)
```

The outer batch must contain exactly one phase-space point, and the precision
must be `16` (native binary64). This point is genuinely evaluated after the
family is prepared. Passing a 128-point profiling batch, or asking for higher
precision, is rejected. A repeated `warm_up` for the same family reuses its
structure but still performs the required one-point evaluation;
`result.already_warm` and `result.warmed_query_count` distinguish reuse from a
cold construction.

The optional progress sink receives four stages:

1. process preparation;
2. query-family construction;
3. family finalization; and
4. the first one-point evaluation.

Progress includes completed/total work, elapsed time, worker count, and current
and peak resident memory when the platform supplies them. Without a callback
or Python progress sink, progress reporting and its memory sampling are not
enabled. The final first-evaluation notification occurs after the warmed state
has been committed and is not cancellable.

The packaged example supplies a coloured terminal progress display and result
table:

```console
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py
```

### The same warm-up in native languages

All native interfaces use one flattened point in
`[external particle][E, px, py, pz]` order and stable selector IDs. Omitting an
axis sums it.

In C, warm one flow and all helicities:

```c
const char *one_flow[] = {flow_id};
RusticolWarmUpResult result = {0};
int status = rusticol_runtime_warm_up_f64(
    handle, point, momentum_count,
    NULL, 0, one_flow, 1,
    report_progress, user_data, &result);
```

The callback has this signature and returns nonzero to continue:

```c
int report_progress(
    const RusticolWarmUpProgressEvent *event,
    void *user_data);
```

The wrappers preserve the same selector order:

```cpp
auto result = runtime.warm_up(point, {}, {flow_id}, report_progress);
```

```fortran
call runtime%warm_up(point, result, color_ids=flow_ids, &
    progress_callback=report_progress, progress_user_data=user_data, ierr=ierr)
```

```rust
let selectors = Selectors::all().with_colors([flow_id]);
let result = runtime.warm_up_f64(
    &point, &selectors, Some(&mut report_progress),
)?;
```

For all flows at one helicity, put one helicity ID in the helicity argument and
leave the colour argument empty. Passing no callback is valid. C++ releases at
destruction, Fortran exposes `runtime%close()`, Rust releases on `drop`, and C
uses `rusticol_runtime_free(handle)`. The native wrappers do not expose
Python's clear-without-unload convenience; close and reload to return to a
fully cold native handle.

### Model-parameter updates do not keep stale numbers

`runtime.set_model_parameters(...)` updates the loaded runtime atomically. On
the next warm-up or evaluation, an already retained OTF family refreshes its
parameter-dependent workspace before executing. Merely changing a numerical
model parameter does not require rebuilding the same structural selector
family, and the old parameter values are not silently reused.

Python `runtime.clear()` discards the OTF family, prepared process state, and
scratch storage while keeping the artifact and the latest parameter values
loaded. In recurrence, compiled, and eager modes, `clear()` is a no-op because
they do not have the corresponding OTF family cache.

## A practical decision sequence

1. Decide the colour accuracy from the physics question: `lc`, `nlc`, or
   `full`.
2. For materialized LC modes, decide which slice is repeated most often:
   one-flow/helicity-sum suggests `topology-replay`; all-flow/one-helicity
   suggests `all-flow-union`.
3. Start with recurrence for a reusable prepared-model output. Use eager when
   materialized generation/load time is the priority, and compiled for
   process-wide compilation or raw model IR.
4. Consider OTF when the compact output and one dominant LC selector family
   outweigh an expensive cold warm-up. Keep the runtime handle alive.
5. Keep generation coverage complete until a measured production workload
   justifies a non-reusable specialized output.
6. Profile the exact process, selector, parameters, batch size, and target
   machine used by the application.

See [Configuration](configuration.md#color-accuracy-and-lc-layout) for every card
field, [Runtime and Selectors](runtime-and-selectors.md) for resolved output and
per-point selectors, and the [Native APIs](native-apis.md) for complete
buildable language examples.
