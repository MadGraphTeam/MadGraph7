# Direct-Arena McClintock Validation

This read-only audit validates the current topology-replay recurrence milestone
against:

- `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`;
- `recurrence_direct_arena_contract.md`;
- original AmpliCol's compact mode-1 recurrence.

All-flow union is outside this verdict and remains pending.

## Verdict

The topology-replay implementation has corrected the architectural gaps
identified by the original McClintock audit. It now builds an AmpliCol-like
compact recurrence before `GenericDAG`, retains dynamic LC color state and
local source ancestry in current identity, lowers exact contribution and
closure terms into a Direct-Arena schedule, groups rows by prepared callable,
and executes directly against persistent current and momentum arenas.

No recurrence `KernelPacket`, `Attachment`, `EagerKernelInput`, packed
input/output buffer, scatter step, `EagerKernelBackend`, or
`PreparedEvaluatorBackend::evaluate_batch` route is present in the
topology-replay hot path. This closes the decisive runtime-representation gap
from the original audit.

The built-in and UFO-SM f64 topology canaries have already demonstrated the
required compact structural counts and component agreement for `d d~ > Zg`
and `d d~ > Zgg`. The warmed native scheduler allocation test reports zero
allocations over repeated evaluations.

The topology milestone is not yet ready to commit solely because the new exact
executor and native exact-section binding are still uncommitted and have not
received their final rebuilt-extension integration run. Their static design is
consistent with the Direct-Arena contract and consumes the authenticated
plan-v2 sections and semantic/runtime-layout digests. That final integration
run, followed by review of its result, is the remaining topology-replay gate.

## Audit Findings

### 1. Pre-`GenericDAG` Fork

The recurrence physics/projection path explicitly forks before `GenericDAG`
construction (`src/pyamplicol/generation/recurrence_physics.py:4`), and the
fixed-width recurrence input encoder does not import or construct a
`GenericDAG` (`src/pyamplicol/generation/recurrence_columnar.py:8`).

Direct lowering consumes the validated semantic recurrence program and
prepared Direct-Arena descriptors
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:276-340`). It
does not construct a process evaluator or SymJIT application. Process
generation therefore references prepared model kernels without performing
per-process evaluator construction.

**Result:** pass for topology replay.

### 2. Dynamic LC Color State And Transition Witnesses

`CurrentCoreKey` contains an interned dynamic LC color-state ID
(`rust/crates/rusticol-core/src/recurrence/layout.rs:428-533`) and every
contribution retains its exact quantum-flow witness
(`rust/crates/rusticol-core/src/recurrence/layout.rs:581-649`).

Construction interns source and derived color states
(`rust/crates/rusticol-core/src/recurrence/construct.rs:625-730`,
`858-873`, and `1489-1522`). Lowering authenticates each contribution's
transition witness against the contribution key before accepting it
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:464`).

This avoids both incorrect aliasing of distinct partial color words and
sector-expanded eager rows.

**Result:** pass for topology replay.

### 3. Local Source Ancestry And Helicity Sharing

Topology-replay current identity stores only local source-state ancestry
(`rust/crates/rusticol-core/src/recurrence/layout.rs:157-205`). Construction
combines the parent-local assignments rather than attaching a complete global
helicity assignment to every current
(`rust/crates/rusticol-core/src/recurrence/construct.rs:2162-2166`).

The runtime executes one compact recurrence per selected replay target and
maps the resulting representative helicity destinations back to public
helicity labels. It does not execute one complete recurrence per helicity.

For `d d~ > Zg`, the compact plan has 31 currents and 12 closure destinations,
not 12 independent current graphs. This is the same sharing principle as
AmpliCol mode 1.

**Result:** pass for topology replay.

### 4. Closures And Single Finalization

Construction builds explicit signed closure-term lists
(`rust/crates/rusticol-core/src/recurrence/construct.rs:1672-1824` and
`1985-2026`). Direct lowering authenticates closure templates, parent order,
eligible quantum-flow witnesses, exact factors, and component factors
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:596-702`).

Finalizations are lowered from the semantic program once
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:527-579`), and
the finalization-row map records one row per propagated current
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:737-740`).

This matches AmpliCol's rule: accumulate every contribution to an
unpropagated current, then finalize or propagate that current exactly once.

**Result:** pass for the Zg/Zgg topology milestone. Reflected pure-gluon,
same-flavour reconstruction, and three-open-line closure canaries remain
broader acceptance work.

### 5. Genuine Direct-Arena Runtime

The recurrence runtime imports and invokes `execute_direct_plan` directly
(`rust/crates/rusticol-core/src/recurrence/direct_runtime.rs:14` and
`749`). The plan owns fixed arena assignments and row groups; lowering creates
those groups by stage, role, and prepared executor
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:773-806`).

A static search of the recurrence lowering/runtime lane finds none of the
prohibited packet-runtime types or the generic `evaluate_batch` route.
Parent values and momenta are addressed by offsets in the Direct-Arena plan,
and results accumulate into destination planes.

**Result:** pass.

### 6. Prepared-Call Grouping And Counters

Rows are sorted into homogeneous prepared-executor ranges during lowering
(`rust/crates/rusticol-core/src/recurrence/direct_lowering.rs:773-806`).
`execute_direct_plan` walks those ranges, so backend dispatch is per
homogeneous row group and point tile, not per recurrence edge and not per
helicity.

The runtime carries `DirectExecutionCounters` through this call
(`rust/crates/rusticol-core/src/recurrence/direct_runtime.rs:14` and `749`).
The topology plan therefore has the correct structure for reporting direct
calls, rows, and lanes without reintroducing packet gather/scatter.

**Result:** pass architecturally. Representative real-process counter and SIMD
occupancy measurements remain part of performance qualification.

### 7. Built-In/UFO And Exact/f64 Plan Sharing

Both built-in SM and UFO-SM use the same recurrence projection, semantic
builder, prepared direct-template contract, lowering, and runtime. The current
Zg/Zgg canaries have matching structural counts and f64 resolved components
after explicit model-state mapping; no model-name optimization branch is used.

The exact executor loads plan-v2/runtime-layout-v2 sections through the native
binding and requires both the semantic and runtime-layout digests
(`src/pyamplicol/runtime/recurrence_exact/_plan_v2.py:176-177` and
`260-275`). It interprets the same row groups, finalizations, closures, exact
factors, and replay mappings used by f64
(`src/pyamplicol/runtime/recurrence_exact/_execution.py:66-131`).

**Result:** model-generic f64 parity passes for the milestone. Exact/f64 plan
sharing passes static review but awaits the final rebuilt-native built-in/UFO
integration run before certification.

### 8. Structural Counts And Allocation Evidence

The integration contract fixes:

- `d d~ > Zg`: 31 currents, 34 contributions, 12 closures;
- `d d~ > Zgg`: 69 currents, 126 contributions, 24 closures.

These assertions are recorded in
`tests/integration/test_recurrence_generation_runtime.py:41-42`, and both
built-in and UFO-SM f64 canaries have passed them.

The native allocation regression uses a counting global allocator and repeats
warmed Direct-Arena topology execution
(`rust/crates/rusticol-core/tests/recurrence_direct_arena_allocations.rs`).
Its completed release-mode run observed zero heap allocations and zero
allocated bytes across 32 evaluations, including replay mapping and momentum
filling.

**Result:** pass for the native scheduler core. Public native
`evaluate_f64_into` allocation accounting remains a later full-API gate.

## Remaining Topology-Replay Blocker

Before committing and benchmarking this milestone:

1. finish review of the exact executor/native exact-section binding;
2. rebuild the staged native extension once from the settled source;
3. run the focused built-in and UFO-SM Zg/Zgg exact tests at precisions 32 and
   50, including selectors and parameter updates;
4. rerun the focused f64 canaries and native allocation test against that same
   binary identity;
5. confirm no finding from this audit regressed in the final diff.

No additional architecture redesign is indicated for topology replay.
All-flow union remains pending because its runtime-helicity source-selection
ABI is not yet complete and is deliberately not certified here.
