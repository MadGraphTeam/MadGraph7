# Recurrence Architecture Audit Against Original AmpliCol

This note preserves the independent read-only audit performed while the
recurrence ABI was being frozen. It is an implementation checklist, not a
claim that recurrence execution is complete or performance-qualified.

## Status At The Audit

The branch has a sound pre-DAG, model-generic foundation:

- recurrence configuration and LC-only preflight;
- a process-independent prepared recurrence-template catalog;
- process/color projection before `GenericDAG` construction;
- exact factors, fixed-width column encoding, and Rust input validation;
- explicit model-owned dynamic flavour/quantum-flow contracts, checked against
  the live model callback;
- the same catalog path for built-in and UFO models.

It does not yet have a recurrence state builder, PACBIN schedule writer/loader,
`RecurrenceExecutionRuntime`, or native recurrence evaluation. The private
binding currently validates identity and returns `schedule_constructed=false`.
Generation intentionally stops before `GenericDAG` until the builder exists.
No AmpliCol-like generation, runtime, memory, or scaling claim is valid before
those parts and the gates below are complete.

## Original AmpliCol Strategy

Original AmpliCol stores compact currents containing particle/current type,
chirality, momentum subset, process-support bits, local external-current
ancestry, ordered color words, and contributing vertices. It constructs by
subset size and child splits, rejecting incompatible pairs before state
materialization.

For its mode 1, one flow with all helicities, current interning includes local
source-current ancestry. Partial currents are shared whenever that local
ancestry agrees; AmpliCol does not copy a complete recurrence for every full
helicity assignment.

For mode 2, all flows with one runtime helicity, source spin is a runtime
placeholder. Current interning uses ordered external color words plus the
current contract and process support. Exact reversal and exchange relations
produce signed aliases. Runtime evaluates each interaction, accumulates every
contribution to a current, then propagates/finalizes that current once.

Closures include signed or reversed pure-gluon closures, multiple-open-line
partner terms, same-flavour reconstruction, and external-fermion exchange
signs. Dead-tree filtering is exact backward reachability. AmpliCol's separate
ten-point numerical helicity-equivalence filter is not a production proof and
does not need to be copied where pyAmpliCol has exact certificates.

Relevant legacy implementation locations:

- `dependencies/checkouts/legacy-amplicol/amplitude_QCD.f03`: current records,
  subset construction, interning, recurrence evaluation, closures, filtering,
  and generated compact arrays;
- `dependencies/checkouts/legacy-amplicol/amplicol_generate.f03`: library
  generation and numerical helicity filtering;
- `dependencies/checkouts/legacy-amplicol/amplicol_color_library_probe.f03`:
  report-side flow mapping and generated-module evaluation;
- `dependencies/checkouts/legacy-amplicol/amplicol_reweight.f03`: direct
  dynamic all-flow mode-2 recurrence.

## Library Comparison Nuance

`--library=create` initializes AmpliCol mode 1 and writes a compact generated
module for each generated group/integral. The report's all-flow library probe
uses a dynamic mode-2 amplitude to enumerate physical rows, then evaluates the
mapped mode-1 generated modules. It is not one universal generated all-flow
union.

Consequently, performance comparisons must keep three contracts separate:

1. pyAmpliCol topology replay versus AmpliCol generated mode-1 modules;
2. pyAmpliCol all-flow union versus AmpliCol's direct mode-2 recurrence;
3. pyAmpliCol all-flow union versus the report's multi-module library probe.

## Required Parity Corrections

### Executable Dynamic LC Color State

The recurrence builder must carry an interned canonical partial-color state,
including ordered color word or open-string tuple. Each transition needs an
executable, model-generic operation and exact witness for inheritance,
concatenation/reversal, singlet movement, and closure. A complete physical
sector ID must not be used as current identity because that destroys sharing.
Omitting partial color state would instead alias distinct currents
incorrectly. This contract must be frozen before the Rust state builder.

### Layout-Specific Helicity Identity

`topology-replay` must share states by local source-state ancestry rather than
by the complete helicity assignment. `all-flow-union` must use runtime-helicity
source placeholders and exclude numerical helicity from internal current
identity. It must not instantiate the graph once per source-helicity template.
The layout policy must be explicit in state construction rather than inferred
from a single unconditional `spin_state` field.

### Compact States, Not Expanded Eager Rows

The stored schedule must scale with unique current states, contribution rows,
finalizations, and signed closure terms. It must contain no
flow-by-helicity-by-edge expansion and no `GenericDAG`. Every non-source
propagated current has exactly one finalization after all contributions have
been accumulated.

### Closure And Reconstruction Semantics

The builder must schedule exact signed term lists for reflected pure-gluon
closures, multiple-open-line partner contractions, same-flavour
reconstruction, and exchange signs. A generic closure copied once per
sector/helicity is neither compact nor sufficient.

### Process-Set Sharing

Process support remains a bitmap attached to a state and is not part of state
identity. Multi-process construction must intern schedules by semantic digest
and demonstrate fewer unique schedules than concrete subprocesses for
`p p > j j j j`, matching the role of AmpliCol's process bitsets.

### Packetized Backend Calls

Prepared kernels must be invoked over homogeneous edge-by-point packets. One
backend call per recurrence edge would make indirect dispatch and
gather/scatter dominate. Runtime profiles must expose rows per call and prove
that batching is effective.

## Scaling Expectations

Topology replay should have one state per compatible subset, local source
ancestry, current contract, and partial color state, without an extra complete
helicity multiplicative factor. All-flow union should have one state per unique
ordered partial color/open-string state and current contract, independent of
helicity. Physical output-flow growth is unavoidable, but internal memory must
not acquire an additional helicity or sector-copy factor.

Prepared model compilation is amortized. Recurrence process generation must
not construct per-process symbolic evaluators or invoke SymJIT, C++, ASM, or a
linker.

## Mandatory Structural Counters

Generation and artifacts must report:

- candidate child pairs and accepted/rejected transitions by reason;
- unique currents by subset, current state, chirality, and dynamic color state;
- contribution fan-in distribution;
- unique open-string/trace states, replay classes, targets, and aliases;
- structural-zero and equivalent-helicity representatives;
- closures, signed closure terms, and residual/unproved sectors;
- process-support density and schedule aliases;
- referenced prepared-kernel IDs;
- rows and bytes for sources, contributions, finalizations, closures, and
  selectors;
- peak live states, peak RSS, Python extraction, Rust builder, and
  serialization time;
- direct comparison with AmpliCol `n_cur`, `n_vert`, and `n_amps` before and
  after filtering/optimization.

Runtime profiles must report source fills, contribution rows, finalizations,
closure terms, backend calls by stage/kernel, rows per call, SIMD occupancy,
selector-group sizes, gather/scatter bytes and time, kernel time,
accumulation/finalization/reduction, stable-reorder decisions, and warmed
allocations.

## Hard Review Checklist

- [x] Executable partial LC color-state ABI is implemented and independently
      validated.
- [x] Topology replay uses local source ancestry rather than complete helicity.
- [x] All-flow union uses runtime-helicity placeholders and is graph-size
      independent of retained helicity count.
- [x] No `GenericDAG` or flow-by-helicity-by-edge materialization exists.
- [x] Stored contribution count equals the sum of each current's fan-in.
- [x] Every non-source propagated current has exactly one finalization.
- [x] Exact closure/reconstruction term coverage is complete.
- [x] Process-set schedule sharing is implemented and measured.
- [x] Direct row-group counters demonstrate typed grouped execution rather than
      packet packing or edge-wise evaluator dispatch.
- [x] Built-in and equivalent UFO physics produce matching state,
      contribution, and closure topology after explicit model-state mapping.
- [ ] A fresh independent reviewer has checked the implementation against this
      audit after the builder and runtime exist.

## Source-Freeze Audit Checklist And Results

This section records the post-Direct-Arena source-freeze audit. It supersedes
the historical implementation-status paragraph above, which is retained to
show what the original audit reviewed. The current recurrence ABI is
Direct-Arena v2; the packetized v1 runtime has been removed.

Audit date: 2026-07-24

Source freeze status: **candidate ready for the final independent re-audit**.
The implementation blockers identified by the original audit are closed.
Approval remains withheld until the exact committed source is rebuilt,
remeasured, and independently reviewed.

### Proven In Current Sources

- [x] Direct-Arena v2 is a separate runtime lane and does not route recurrence
      through eager packets, packed evaluator inputs, attachments, or scatter.
- [x] Dynamic partial LC color state and layout-specific helicity identity are
      part of recurrence state identity.
- [x] Topology replay uses local source ancestry; all-flow union excludes
      numerical helicity from non-source current identity.
- [x] Closure rows bind complete proof groups, including exact signs and
      reconstruction terms.
- [x] Replay source transport fails closed across initial/final crossing roles,
      source-template contracts, momentum signs, and crossing phases.
- [x] Stored final topology-replay and all-flow-union counts agree with the
      corresponding retained AmpliCol counts for the tracked LC ladder.
- [x] Built-in and UFO-SM use the same model-generic recurrence catalog and
      builder path.

### Test And Documentation Closure

- [x] Direct-Arena v2 is documented as the sole current recurrence ABI;
      recurrence v1 identifiers are explicitly removed.
- [x] Built-in and UFO-SM all-flow-union tests pass for folded pure-gluon
      reflection and same-flavour reconstruction.
- [x] Built-in and UFO-SM all-flow-union tests pass multiple-open-line closure
      semantics.
- [x] Native f64 and exact Python agree component-wise for every difficult
      all-flow-union case.
- [x] Recurrence artifacts, rather than compiled artifacts, pass homogeneous,
      alternating, deterministic-random, and pre-grouped per-point selector
      tests for both layouts.

Unchecked items remain source-freeze gates. Test presence alone is not
acceptance evidence.

### Remaining Source-Freeze Gates

- [ ] Rebuild from the exact committed checkpoint and repeat the LC correctness,
      allocation, process-set, and qq_Z6g performance evidence.
- [ ] A fresh independent reviewer rechecks the completed builder, both
      layouts, process-set sharing, and runtime against this audit.

The closure-rooted builder no longer constructs the historical ten-times-larger
forward candidate graph. It streams demanded states backward from physical
closures and validates dependencies forward. Process-set interning is also
implemented: the guarded `p p > j j j j` acceptance has 11 concrete process
bindings, 8 unique schedules, and 3 exact aliases. Representative aliases
match independently generated artifacts component by component.

### Counter Semantics

The final qq_Z6g topology schedule has 1,425 currents, 8,338 contributions,
384 closure terms, and 45 reachable dynamic color states for both built-in SM
and UFO-SM. The earlier count of 72 dynamic color states was the complete
pre-liveness catalog; 27 entries belonged only to discarded candidates.
Closure-rooted construction interns only reachable states, so 45 is the
correct final-schedule count.

The current 69 row groups are not an expansion from the earlier reported 44
kernel packets. The definitions differ. The 69 groups cover every typed
`(stage, direct_executor_id)` range:

- 9 source groups for 19 rows;
- 41 contribution groups for 8,338 rows;
- 17 finalization groups for 858 rows;
- 2 closure groups for 384 rows.

The old packet count omitted source work, identity finalizations, and direct
closures. Direct-Arena executes one typed handle per group and point tile;
there are no packed input/output buffers, attachments, or scatter rows.

### Focused Test Results

Command:

```console
SYMBOLICA_LICENSE=... .venv/bin/python tools/ci/memory_watchdog.py \
  --limit-gib 30 -- .venv/bin/python -m pytest \
  tests/integration/test_recurrence_generation_runtime.py::test_builtin_lc_recurrence_artifact_loads_and_matches_compiled \
  tests/integration/test_recurrence_generation_runtime.py::test_builtin_lc_all_flow_union_recurrence_matches_compiled \
  tests/integration/test_recurrence_prepared_execution.py::test_direct_arena_all_flow_union_difficult_semantics \
  -q
```

Current pre-freeze results on 2026-07-24:

- 112 focused Python unit tests passed, with one guarded process-set test
  skipped from the unguarded unit run;
- 22 public recurrence integration tests passed across built-in/UFO-SM,
  topology replay, all-flow union, pure-gluon reflection, same-flavour
  reconstruction, and three open quark lines;
- the guarded `p p > j j j j` process-set test passed in 16.31 seconds with
  0.719 GiB peak RSS;
- 202 focused Rust recurrence tests passed;
- malformed ragged source-state mappings and truncated support words are
  rejected without panics.

These are pre-freeze results. The exact committed checkpoint must still be
rebuilt and rerun before the independent reviewer may approve the source.
