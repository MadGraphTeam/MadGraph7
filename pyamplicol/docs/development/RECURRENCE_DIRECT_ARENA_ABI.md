# Recurrence Direct-Arena ABI

## Status

This document freezes the implementation boundary for the recurrence runtime
being developed on `codex/recurrence-execution`. The detailed design and
acceptance contract is maintained in
`PREPARE_STANDALONE_PYAMPLICOL/recurrence_direct_arena_contract.md`.

The direct-arena design replaces the pre-release packetized recurrence
runtime. There is no recurrence artifact compatibility requirement and no
v1-to-v2 converter.

## ABI Identities

- Builder input: `pyamplicol-recurrence-builder-input-v2`
- Prepared companion: `pyamplicol-recurrence-direct-template-v1`
- Semantic plan: `pyamplicol-recurrence-plan-v2`
- Runtime layout: `pyamplicol-recurrence-runtime-layout-v2`
- Prepared backend: `rusticol.recurrence-direct-backend.v1`
- Runtime capability: `rusticol.recurrence-direct-arena.complex-f64.v1`
- LC capability: `rusticol.recurrence-color.lc.v1`
- PACBIN framing: `pacbin-v1`
- PACBIN member kind: `RecurrenceDirectPlan`
- PACBIN member path: `plan/recurrence-direct-plan-v2.bin`

The following identities are removed:

- `pyamplicol-recurrence-builder-input-v1`
- `pyamplicol-recurrence-template-v1`
- `pyamplicol-recurrence-plan-v1`
- `pyamplicol-recurrence-runtime-layout-v1`
- `rusticol.recurrence-runtime.complex-f64.v1`
- `RecurrenceRuntimePlan`
- `plan/recurrence-plan-v1.bin`

Old recurrence artifacts fail from their small execution metadata with a
regeneration message. Their PACBIN payload is not decoded.

## Ownership

Python and the model compiler own exact algebra, model-generic proof
contracts, prepared-template construction, fixed-width builder-input columns,
and exact execution.

The Rust recurrence builder owns current/contribution interning, dynamic LC
states, closure-driven construction, liveness, momentum-form interning, arena
slot assignment, direct row grouping, deterministic digests, and PACBIN
serialization.

The prepared model owns model-generic source, contribution, finalization, and
closure semantics plus one direct executor pack for JIT O2, C++ O3, or ASM.
Process generation never compiles a backend evaluator.

Rusticol owns authenticated loading, direct handle resolution, selector plans,
workspace allocation, stable selector grouping, tiled execution, reduction,
and native profiling.

## Runtime Representation

Current values use aligned split-complex, component-major storage:

```text
current_re[(component_base + component) * point_stride + point]
current_im[(component_base + component) * point_stride + point]
```

Momentum forms and amplitude destinations use the same point-contiguous
principle. Semantic current IDs and physical arena component ranges are
distinct. Interval coloring may reuse ranges whose liveness intervals do not
overlap.

The plan contains fixed-width:

- current descriptors;
- source rows;
- contribution rows;
- finalization rows;
- closure rows;
- stage-ordered row-group descriptors;
- momentum forms;
- exact factors;
- replay/selector mappings.
- physical amplitude-destination bindings;
- resolved-helicity source-state and public-helicity catalogs.

Rows are grouped by stage, role, direct executor, and destination operation.
Each loaded row group names one pre-resolved typed function handle. The hot
loop traverses the authenticated plan directly and does not construct a
second packet schedule or perform string or map lookup.

## Direct Backend Rule

A direct executor receives arena views, momentum views, parameter/factor
views, a fixed-width row range, and a point tile. It loads parent planes by
row offsets and writes or accumulates into destination planes.

Identity propagation is one generic Rusticol intrinsic executor. Its prepared
catalog dimensions are upper bounds only; each finalization row supplies the
actual component count. The executor is therefore shared by every
identity-propagator state and never duplicated per process or model state.

The recurrence lane must not use:

- `EagerKernelInput`;
- `EagerKernelCall`;
- `EagerKernelBackend`;
- packed kernel input/output buffers;
- packet output scratch;
- a scatter/attachment pass;
- `PreparedEvaluatorBackend::evaluate_batch`.

Small register or stack temporaries internal to one direct primitive are
allowed. A process- or packet-sized packed buffer is not.

Saved JIT direct templates use portable SymJIT optimization level 2. C++ and
ASM implement the same row/arena ABI but retain scalar batched expectations.

## Layout Semantics

`topology-replay` stores local source-state ancestry and one compact
recurrence per proven flow topology class. Selecting a flow maps sources and
momentum forms to its representative and executes that compact recurrence
once while summing the retained helicity closures.

`all-flow-union` excludes numerical helicity from current identity. Runtime
source dispatch fills one selected helicity and one dynamic-color union
produces all physical-flow closures.

Omitting a flow or helicity at generation retains the complete physical axis.
Explicit generation selection limits coverage. Layout changes organization,
not selector semantics.

## Hot-Loop Invariants

- Points are contiguous and form the JIT SIMD axis.
- Contributions accumulate before exactly one finalization per non-source
  current.
- Closures accumulate directly into amplitude destinations.
- Momentum forms are filled once per tile.
- Stable selector grouping is skipped when the input is already grouped.
- A warmed native `evaluate_f64_into` allocates no heap memory.
- Packet-input bytes, temporary packet-output bytes, and scatter bytes are
  always zero.
- Compiled and eager runtime code is not modified by this lane.

## First End-To-End Gate

Topology replay must pass built-in and UFO-SM:

- `d d~ > Z g`: 31 currents, 34 contributions, 12 closures;
- `d d~ > Z g g`: 69 currents, 126 contributions, 24 closures.

Both physical `Zgg` flows, every retained helicity component, exact execution,
native f64 execution, deterministic PACBIN validation, direct-call counters,
and zero warmed allocations are required. After that gate, all-flow union is
added for `Zgg`, followed by the complete `Z+1g` through `Z+6g` ladder.

An independent review against
`docs/development/RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` is required after
each layout milestone.
