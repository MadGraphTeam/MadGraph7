<!-- SPDX-License-Identifier: 0BSD -->

# On-the-fly Mode Architecture and Implementation Plan

## Status and decision

This document freezes the architecture selected by the on-the-fly research
prototypes made from source revision
`a08feed4aacf39c00dfedaeedd3a82a9666f1565`. The prototypes remain local,
unmerged research commits. They are evidence and test fixtures, not production
code to merge wholesale.

The selected design is:

> compact algorithmic axes + a compact process/ownership seed + a Rust port of
> the query-local selected-GenericDAG semantics + a structural interpreter,
> followed by a plain 8-byte trace VM with a bounded cache + one
> plan-independent prepared-executor pool.

On-the-fly execution is a distinct runtime lane and artifact kind. It is not a
new `RecurrenceStrategy` layered on `DirectRecurrencePlan`, because constructing
that plan would reintroduce the global materialization that this mode exists to
avoid.

The first production slice is LC, one selected helicity, and one selected
physical flow. Selected-flow helicity streaming and LC all-flow streaming come
after exact structural and numerical parity. NLC/full require a separate
compact contraction proof and are not part of the initial mode.

## Prototype evidence and disposition

| Research commit | Evidence | Decision |
|---|---|---|
| `ce8037749c97ae04bae6cec42161991754ee2078` | Algorithmic helicity and LC-flow rank/unrank codecs address one selector without constructing cardinality-sized tables. | Adopt the algorithms and fixtures; rewrite them as production contracts. Do not impose the prototype's machine-integer selector ceiling. |
| `4c9a1d6cedb082140c5b492171bd2fe7873b0f98` | A plain 8-byte VM was bitwise equal to the synthetic direct interpreter, used no warm allocations, occupied 14--24% of the compared direct hot bytes, and met the 1.4x warm-runtime gate. Generic superinstructions lowered 1.69--1.80x more slowly than direct rows on the pure-like trace. | Adopt the plain VM after the structural interpreter is correct. Reject generic superinstructions from the initial design. |
| `84f2246f0de8f5d9e598eb76f2ff8e85bd0dbd36` | A shared-amplitude color DP is invalid because ordered prefixes carry distinct color-ordered amplitudes. Lazy word generation is exact. LC contraction is a streamable diagonal second moment in all exhaustively checked pure, open-line, and identical-line domains; coherent summation is wrong. | Reject shared-amplitude DP. Retain lazy LC word generation and the diagonal streaming reducer. |
| `2312b1137c7e2f3d6bdf9064a902592c16bac58c` | Candidate A builds a selected GenericDAG query with zero full color-plan, global projection, global feasibility-index, or full-axis calls. Semantic contribution/closure multisets agree with the selected structural oracle. Retained n=5 cases agree with an independent p32 recurrence runtime. | Adopt the selected-GenericDAG semantics and structural oracle. Port the query-local kernel to Rust instead of shipping the Python prototype. |
| `caa500dd7fa67a7e496238e63c6b823aab4b8dd6` | The demand-first memo used synthetic numerical kernels, looped over a helicity axis, and did not match the required structural ownership boundary. Its reverse lookup reduced discovery work. | Reject the evaluator. Retain only the reverse-index idea and the requirement for compact source/closure anchors. |
| `6b4f011d90d5262807d8316d73f28b106ea31fb9` | Candidate C compiled a selector-local trace to the plain VM, had zero warm allocations, kept numeric workspace separate, used an accountable bounded cache, and executed oversize entries ephemerally. Its cross-selector hit relied on synthetic equivalence. | Adopt the VM/cache shape conditionally after a real trace exists. Initially bind cache identity to the exact decoded selector. |
| `dc171ac357c2817465f20baf3ff0efa91086b53a` | Selected-query structural construction remained query-sized through n=10 under a 30 GiB watchdog. The n=10 pure-gluon case used 81 currents, 400 contributions, one closure, and 481 peak rows against a 371,589,120-entry full selector product; the open-line case used 73, 309, one, and 382 against 41,287,680. | Adopt these structural counters and n=10 watchdog cases as acceptance gates. They are not high-n numerical authority. |
| `cf8de3820152be262bafc6f93e5af9a6f423b271` | Compact ownership seeds preserve crossed sources, pairing classes, explicit orders, owner keys, source slots, and parity without public flow or projection tables. The seed was 7,711 bytes/141 rows at n=7 and 10,136 bytes/481 rows at n=10. Ownership still delegates to Candidate A. | Adopt the compact seed concept. Replace its normalization heuristic with the existing authenticated normalization builder and prove ownership in Rust. |
| `e71161cb656ed5eeaa69d18221735b242c60554d` | One 597-handle prepared pool served two compact plans; warm calls allocated zero bytes and the watchdog peak was 0.252 GiB. No direct plan, full axes, global projection, or union schedule was built. The genuine-kernel traces were intentionally shorter than a complete process trace and used hard-coded executor IDs. | Adopt the plan-independent pool seam. Replace manual IDs with semantic-key lookup and place pool ownership in the runtime lane, not cached plans. |

The evidence supports the architecture, but not a claim that full-process
compact numerical parity is already proven. The open questions are recorded
explicitly below.

## Non-negotiable anti-materialization contract

The on-the-fly path must not call, directly or through a fallback:

- Python `build_color_plan`;
- Python `Generator._prepare_process_construction`;
- Python `project_recurrence_process_v1`;
- Python `build_recurrence_builder_input_v1`;
- Python `build_recurrence_physics`;
- Rust `build_replay_targets`;
- Rust `finish_program`; or
- any `DirectRecurrencePlan` constructor, codec, loader, or runtime.

Tests must poison these entry points or count calls. A zero count is an
architecture acceptance condition for this mode, not merely telemetry. There
is no silent fallback to the ordinary recurrence builder.

The mode also obeys these structural invariants:

1. It stores no dense `ProcessPhysics.helicities`, dense
   `ProcessPhysics.color_components`, global color plan, full recurrence
   projection, global feasibility index, replay-target union, or union
   schedule.
2. Resident query memory is proportional to the selected query DAG/trace,
   numeric workspace, and an explicit cache budget. It is not proportional to
   the product of all helicities and all flows.
3. The selector ABI carries decoded values, not a machine-sized flattened
   selector: a public helicity tuple, an LC word and open-line blocks, and the
   pairing/permutation parity. Python may expose arbitrary-precision ranks;
   Rust may stream lexicographic successors. No new `u64` cardinality ceiling
   is permitted merely for implementation convenience.
4. Every owner identity retains crossed source identity, current,
   contribution/closure role, coupling orders, contraction identity, color
   weight, and Fermi sign. Canonicalization may reorder representations; it
   may not merge non-identical physics.
5. Allocations use checked arithmetic and fallible allocation. User-selected
   cache/workspace budgets are allowed. Arbitrary acceptance ceilings are not.
6. The new internal artifact/trace format has no backward-compatibility or
   migration requirement. Unsupported old experimental data fail closed;
   there is no compatibility fallback to a materialized plan.

## Compact descriptor and query contracts

### `OnTheFlyProcessSeedV1`

One compact, immutable process seed is produced at generation time. It contains
only process-wide information that a selected query cannot reconstruct safely:

- schema and algorithm revisions;
- process identity and digest;
- model, prepared-kernel catalog, and semantic direct-executor catalog
  digests;
- for each external leg, the crossed source anchor, public-helicity to source
  state mapping, chirality, spin/statistics, color role, and crossing
  orientation;
- compact color-role/source metadata sufficient to decode an LC word;
- open-line pairing classes, species, source slots, and the exact permutation
  parity contract;
- explicit finalized coupling-order limits;
- any proven reference color order/reflection rule needed by LC decoding;
- the existing exact recurrence-normalization payload and digest;
- algorithmic helicity/flow-axis revisions and arbitrary-size cardinalities
  encoded canonically at the Python boundary; and
- the ownership algorithm revision.

The seed contains no enumerated helicity axis, flow axis, color sectors,
projection, recurrence rows, or prepared executor IDs.

### `DecodedLcQueryV1`

A query contains:

- the seed digest;
- the public helicity values in public leg order;
- one exact LC word;
- canonical open-line blocks or trace orientation;
- pairing/permutation parity;
- the public selector identities/digest used for reporting; and
- an exact external-permutation binding when an artifact alias is used.

Validation first confirms that the decoded query belongs to the seed domain.
Ranks, when accepted at an API boundary, are decoded before they enter the
structural builder.

### Structural trace and workspace

`OnTheFlyStructuralTraceV1` is an immutable query-local sequence of source,
transition, finalization, and closure operations. Each operation references a
semantic executor key. `OnTheFlyWorkspaceLayoutV1` separately describes the
checked numeric workspace required by that trace. Structural proof metadata
records deterministic contribution and closure multisets plus query-local
counters. It is validation data, not another global plan.

## Coupling orders and normalization

Coupling-order limits are resolved once during compact-seed production and are
explicit inputs to every query build. Query execution never infers, relaxes,
or expands them. The first implementation must fail closed when finalized
limits cannot be established without constructing the forbidden global color
plan; it must not materialize that plan as a convenience.

Normalization is not copied from the ownership prototype. Seed production
uses the existing authenticated `build_recurrence_normalization` contract and
binds its payload/digest to the same process, source ordering, color accuracy,
and coupling policy. The query-local Rust port applies each averaging factor,
coupling, LC color weight, identical-particle factor, and Fermi sign exactly
once. No amplitude division, averaging workaround, or process-name branch is
allowed.

## Query-local selected-GenericDAG port

The Rust builder ports the proven semantics of Candidate A, not the current
global recurrence finishing path:

1. Validate one `DecodedLcQueryV1` against its seed.
2. Resolve only the source anchors needed by that query.
3. Discover currents and contributions demand-first using reverse indexes over
   the compact prepared process/kernel catalogs.
4. Derive the exact selected closure/root ownership, preserving open-line
   source slots and permutation parity.
5. Emit deterministic, contiguous query-local registers and a structural
   trace.
6. Compare semantic contribution/closure multisets against the retained
   selected-GenericDAG oracle during development.

The port must not call the global `construct.rs` replay-target/finalization
tail. Shared low-level transition and closure rules may be factored out only
when their inputs are query-local and their behavior remains common to both
builders.

## Plan-independent prepared executor pool

Prepared model/kernel contexts and the semantic transition, propagator,
finalization, and closure catalog are loaded once into a pool independent of
any recurrence plan or process source domain. The runtime lane owns that pool.

Process/query source-dispatch domains are a separate validated binding. The
compact seed supplies crossed-source anchors; an
`OnTheFlySourceDomainBinding` owned by the lane binds those anchors to the
model-level pool without constructing a direct plan. This binding is not part
of the model-level pool because source domains are process- and query-bound.
The currently unresolved construction of complete source-domain anchors is an
explicit stop condition below.

A cached structural/VM plan stores only semantic keys or resolved non-owning
handles whose lifetimes are bounded by the lane. It owns neither the prepared
pool nor the source-domain binding. Lane shutdown drops cache entries and
workspaces first, then source-domain bindings, and finally the model/prepared
pool, so no plan/pool or plan/binding ownership cycle is possible.

`PreparedDirectExecutorCatalog` is the starting point for semantic lookup, but
the production API must resolve a key composed of the authenticated operation
kind, source/current quantum identity, kernel/template identity, coupling
orders, and any required momentum/contraction form. Hard-coded executor IDs
from the seam prototype are forbidden. Missing, ambiguous, or digest-mismatched
lookups fail closed.

Parameter updates mutate the lane's prepared runtime state through the same
existing atomic parameter contract. Structural traces and VM code remain
immutable and reusable when their bound structural identities do not change.

## Execution sequence

### Current production LC family retention

The native LC lane currently declares a deterministic `last-family-only`
policy with a hard limit of one retained selector family. The outer prepared
selection, lane request family, executor row family, handles, and exact
semantic binding map advance together only after the candidate's first
successful evaluation. Preparing or executing a failed candidate discards its
pending rows and bindings without evicting the last successful family. Before
any row owner is replaced or cleared, exposed prepared row tables are
invalidated. `Runtime.clear()` leaves zero retained/pending families,
selections, handles, destinations, and semantic bindings. The private runtime
census authenticates this contract as `family_cache_policy =
"last-family-only"` and `family_cache_limit = 1`.

### Structural interpreter first

The first numeric implementation interprets `OnTheFlyStructuralTraceV1`
directly. This deliberately avoids introducing bytecode and cache behavior
before full-query correctness is proven. Its acceptance requires:

- exact structural-multiset agreement with Candidate A;
- f64 agreement with the current recurrence runtime at the same point and
  selector;
- retained p32 agreement where independent artifacts exist;
- correct parameter-mutation behavior without structural rebuilding; and
- zero forbidden global-builder calls.

### Plain VM and bounded cache second

Only after the interpreter passes does the mode lower the same trace to the
plain 8-byte VM. VM and interpreter must be bitwise equal for deterministic
f64 fixtures. Warm execution allocates zero bytes.

The cache is an accountable byte-budgeted LRU. Its initial key binds:

- seed/process/model/prepared/direct-catalog digests;
- exact decoded helicity and LC flow selector;
- coupling-order limits;
- external permutation;
- structural and VM algorithm revisions; and
- any workspace-layout identity that changes code interpretation.

Oversize entries run ephemerally rather than changing the user budget. Numeric
workspace is owned separately and is never charged ambiguously to the plan
cache. Cross-selector sharing is disabled until an exact equivalence proof
demonstrates it; the synthetic prototype hit is insufficient.

## Streaming evaluation milestones

1. **Single selected query.** Execute one decoded helicity and one physical LC
   flow.
2. **Selected-flow helicity streaming.** Iterate the algorithmic helicity axis
   in canonical public order, evaluating and reducing one query at a time. No
   helicity table or list is retained.
3. **LC all-flow streaming.** For each helicity, lazily generate physical LC
   words/open-line pairings and accumulate the exact diagonal second moment
   `sum_f w_f |A_f|^2`. Never compute a coherent `|sum_f A_f|^2`, and never
   merge amplitude histories merely because their combinatorial prefix state
   matches.
4. **Total evaluation.** Nest the two streams with the existing public
   averaging and normalization contract, keeping only bounded reducer state.

Identical open lines retain endpoint species, source slots, external momentum
permutation, and Fermi parity through every streaming stage.

## Production file and interface map

Paths marked **new** do not yet exist. This map fixes ownership before code is
written; changes to it require an architecture review.

| Path | Planned responsibility |
|---|---|
| `rust/crates/rusticol-core/src/recurrence/on_the_fly.rs` (**new**) | Validated compact seed/query types, selected structural builder, trace/workspace layout, semantic proof counters, and the structural interpreter. |
| `rust/crates/rusticol-core/src/recurrence/construct.rs` | Factor only reusable query-local prepared transition/closure rules. Keep `build_replay_targets`, `finish_program`, and all global finishing inaccessible to the new lane. |
| `rust/crates/rusticol-core/src/recurrence/direct_lowering.rs` | Expose plan-independent semantic executor-key lookup from `PreparedDirectExecutorCatalog`; do not create `DirectRecurrencePlanParts`. |
| `rust/crates/rusticol-core/src/engine/recurrence_backend.rs` | Factor the current seam into a model/prepared executor pool and shared invocation primitives. Remove `DirectRecurrencePlan` and process source-domain requirements from the pool API used by this lane. |
| `rust/crates/rusticol-core/src/engine/on_the_fly_lane.rs` (**new**, later) | Own the model/prepared pool, per-process `OnTheFlySourceDomainBinding`, workspace, structural/VM execution, cache, selector streaming, and LC reducers, with the explicit drop order above. |
| `rust/crates/rusticol-core/src/engine/on_the_fly_manifest.rs` (**new**, later) | Decode and validate only the compact artifact members for this mode. |
| `rust/crates/rusticol-core/src/engine/on_the_fly_load.rs` (**new**, later) | Load the compact seed/catalog members, build the source-domain binding, and construct the lane without recurrence-plan or dense-physics deserialization. |
| `rust/crates/rusticol-core/src/engine/artifact_load.rs` | Dispatch the distinct artifact kind to the on-the-fly loader without recurrence fallback. |
| `rust/crates/rusticol-core/src/engine.rs` | Add the distinct `NativeExecutionLane` variant and top-level lane ownership/dispatch. |
| `rust/crates/rusticol-core/src/engine/native_runtime.rs` | Branch on the outer artifact contract before the current unconditional dense `ProcessPhysicsV1` load; route compact physics and evaluation through the distinct lane while preserving selected-result semantics. |
| `rust/crates/rusticol-core/src/engine/physics.rs` and `rust/crates/rusticol-core/src/metadata.rs` | Add compact algorithmic axis descriptors without requiring dense public physics arrays. Exact placement follows existing type ownership. |
| `rust/crates/rusticol-python/src/recurrence.rs` | Minimal cold-path bridge for seed/query validation and structural parity fixtures; not a Python hot-path evaluator. |
| `src/pyamplicol/generation/on_the_fly_seed.py` (**new**, later) | Build and serialize the compact process seed, finalized coupling limits, axis descriptors, and normalization payload. |
| `src/pyamplicol/generation/recurrence_physics.py` | Supply the existing authenticated `build_recurrence_normalization` output; no duplicate normalization implementation. |
| `src/pyamplicol/generation/service.py` | Add a distinct artifact-builder branch only after the private Rust slice passes. It must not reuse `_construct_recurrence_artifact` or the forbidden builders. |
| `src/pyamplicol/generation/artifact_writer.py` | Write the distinct compact members and outer artifact kind; never project compact physics into dense recurrence members. |
| `src/pyamplicol/_internal/versions.py` | Register the new internal runtime/artifact ABI identity when the compact serialized lane is introduced. |
| `src/pyamplicol/artifacts/inspection.py` | Inspect compact axes, seed identity, and mode-specific members without enumerating axes or loading recurrence data. |
| `src/pyamplicol/config/models.py` | Add the public execution-mode setting only at the final public milestone. |
| `src/pyamplicol/api/results.py` and `src/pyamplicol/runtime/backend.py` | Expose lazy/algorithmic physics selectors while preserving the existing concrete selected-result API. Avoid eager tuple conversion. |
| `rust/crates/rusticol-core/tests/on_the_fly_query.rs` (**new**) | Focused structural, numerical, memory, ownership, VM/cache, streaming, and poison-counter tests. |

The first private vertical slice should touch only the Rust recurrence module,
the executor lookup seam, and focused tests. Public Python configuration,
artifact schema, and APIs wait until the private slice proves the core
contract.

## Staged implementation and acceptance

### Stage 0: freeze fixtures and poison boundaries

- Import the compact-axis and selected-GenericDAG fixtures as test data, not
  runtime Python dependencies.
- Add poison/call counters for every forbidden materialization entry point.
- Define canonical semantic keys for prepared source, transition,
  finalization, and closure executors.

**Gate:** the fixture set is deterministic, and a deliberately wired global
builder fails the test immediately.

### Stage 1: private LC structural interpreter

- Implement `OnTheFlyProcessSeedV1`, `DecodedLcQueryV1`, query-local Rust
  construction, semantic executor lookup, workspace layout, and the direct
  structural interpreter.
- Cover scalar contact, a pure-gluon selected LC query, one open quark line,
  and distinct/identical multi-line ownership.

**Gate:** exact contribution/closure multisets; exact source, coupling, color,
and Fermi identities; f64 parity with current recurrence; retained p32 parity
where available; parameter-update correctness; zero forbidden calls.

### Stage 2: plain VM and cache

- Lower the proven trace to the plain 8-byte instruction format.
- Add the bounded accountable LRU, ephemeral oversize execution, and separate
  reusable workspace.
- Test two compact plans sharing one prepared pool, plan eviction while the
  pool remains live, pool destruction after lane shutdown, and concurrent
  independent lanes.

**Gate:** bitwise interpreter/VM equality, zero warm allocations, no ownership
cycle, exact cache accounting, and warm p50/p95 no worse than 1.4x the
structural interpreter on retained traces. Performance is a regression signal,
not permission to weaken correctness.

### Stage 3: compact artifact and private native lane

- Serialize the seed and any required compact prepared-catalog data.
- Give the outer artifact contract a distinct kind/capability and branch before
  `ProcessPhysicsV1` deserialization.
- Load the distinct lane through `on_the_fly_load.rs` without dense
  `ProcessPhysics` or a direct plan.
- Validate and build the lane-owned process source-domain binding against the
  model/prepared pool.
- Preserve existing selected-result identities and parameter mutation.

**Gate:** artifact load and one selected evaluation never invoke a forbidden
builder/loader; inspection remains lazy; artifact growth tracks seed/catalog
content rather than full selector cardinality.

### Stage 4: helicity and LC-flow streaming

- Add selected-flow helicity streaming, then LC all-flow diagonal streaming.
- Check pure adjoints, one open line, three distinct lines, three identical
  lines, and explicit identical-fermion permutations.

**Gate:** canonical selector order, exact resolved-to-total reduction, bounded
resident reducer state, no coherent-flow cross terms, and no full-axis
enumeration.

### Stage 5: public mode and generality

- Add the configuration, artifact dispatch, lazy physics API, and public
  selected/resolved/total evaluation behavior.
- Extend to built-in and UFO models, aliases, external permutations, electroweak
  and scalar interactions, and mixed open-line classes.

**Gate:** current compiled/recurrence reference ladders agree point-by-point
where applicable. Unsupported compact domains fail during generation with a
specific error. NLC/full remain unsupported until their own compact
contraction architecture is proven.

## Memory and scaling validation

Any research or canary command capable of expanding recurrence state runs
under the existing 30 GiB process-tree watchdog. It records the canonical
watchdog JSON and terminates the case if the bound is crossed. The watchdog is
a development safety rule, not an artifact acceptance ceiling and not a reason
to cap valid selector cardinalities.

The exact structural n=10 gates inherited from `dc171ac...` are:

| Case | currents | contributions | closures | peak construction rows | full selector product |
|---|---:|---:|---:|---:|---:|
| pure gluon, n=10 | 81 | 400 | 1 | 481 | 371,589,120 |
| one open quark line, n=10 | 73 | 309 | 1 | 382 | 41,287,680 |

The production port must reproduce these deterministic structural digests and
counters without global axes/projection/color planning while staying below the
watchdog. Prototype RSS values are recorded evidence, not hard product limits.
High-n numerical acceptance is deferred until an independent authority exists.

## Required test matrix

The focused test suite must include:

- arbitrary-size helicity/flow rank round trips and lexicographic streaming;
- malformed decoded selectors, coupling limits, source anchors, parity, and
  semantic executor keys failing closed;
- scalar contact, pure gluon, one open line, three distinct open lines, three
  identical open lines, and four open lines;
- external aliases/permutations and identical-fermion signs;
- exact normalization and resolved-to-total reduction;
- structural interpreter versus current recurrence f64;
- retained p32 numerical checks at the identical phase-space point;
- parameter mutation with stable structure and changed numeric output;
- direct interpreter versus VM bitwise equality and zero warm allocations;
- cache miss/hit/eviction/ephemeral behavior, exact byte accounting, and no
  pool ownership cycle;
- selected-flow helicity order and all-flow diagonal streaming;
- poison assertions for all forbidden Python and Rust builders/loaders;
- n=7, n=8, and the exact n=10 structural/watchdog cases above; and
- deterministic serialization/digests without dense-axis members.

No test may allocate or enumerate the full n=10 selector product merely to
prove that it was avoided.

## Explicit unknowns and stop conditions

These points are not resolved by the prototypes and must not be obscured by
implementation momentum:

1. **Full-process compact parity is not proven.** The executor-seam trace uses
   genuine prepared kernels but is intentionally shorter than the complete
   selected query DAG.
2. **Source-domain query binding is not proven.** The seam used query-bound
   source domains 46 and 59. The compact seed and lane now own this boundary,
   but constructing complete model/process source anchors without global
   materialization still needs a production contract.
3. **Semantic-key lookup is not implemented.** The prototype used hard-coded
   executor IDs. Production must resolve every operation unambiguously from
   authenticated semantic identity.
4. **Standalone Rust ownership is not proven.** The compact ownership seed
   delegates query closure/current ownership to Candidate A. Conflicting or
   incomplete ownership proofs are a stop condition.
5. **High-n numerical authority is absent.** n=7/n=8/n=10 results are
   structural only. They cannot certify amplitudes.
6. **The compact public physics/artifact ABI is unfinished.** In particular,
   lazy axis discovery must not trigger tuple materialization in Python.
7. **LC diagonal generality is bounded by the tested contract.** A new color
   structure outside the proven classes must fail or obtain a new exact proof.
8. **Cross-selector cache equivalence is unproven.** Exact-selector keys remain
   mandatory initially.
9. **Arbitrary-precision on-the-fly execution is not an initial runtime
   feature.** p32 is an independent development authority, not the f64 hot
   path.
10. **UFO and broad interaction generality remain to be demonstrated.** No
    built-in process names or multiplicities may enter production logic to
    paper over a failed generic rule.

If a stage cannot establish its structural or numerical authority, stop at
the first divergent source/current/contribution/closure identity and report
it. Do not normalize around the mismatch, increase a resource ceiling, or
fall back to a materialized recurrence plan.

## Definition of complete

The mode is ready for public exposure only when it evaluates selected and
streamed LC queries using the compact seed and query-local Rust path; resolves
real prepared executors by semantic identity; agrees with structural, f64,
and available p32 authorities; retains exact normalization and fermion/color
semantics; has bounded accountable cache/workspace memory; passes the n=10
watchdog gates; and proves every forbidden global materialization counter is
zero.

This plan adds no release or publication procedure. Normal project delivery
rules apply only after the architecture and correctness gates above are met.
