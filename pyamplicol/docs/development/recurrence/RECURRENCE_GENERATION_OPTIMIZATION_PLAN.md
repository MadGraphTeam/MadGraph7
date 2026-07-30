# pyAmpliCol Recurrence-Generation Optimization

## Summary

Optimize recurrence generation for LC `topology-replay` and `all-flow-union` without changing public APIs, CLI behavior, recurrence plane ABIs, runtime semantics, or runtime performance. The implementation will preserve deterministic canonical output and target topology `d d~ > Z + 8*g` generation below one hour while also materially reducing union generation time.

The working diagnosis is layout-dependent:

- Historic topology n=9 spent 47.0 s of 61.4 s recorded native time lowering, but 814 s end-to-end; lowering was not most total time.
- Historic union n=8 spent 82.3 s in semantic construction versus 10.0 s lowering and 1,263 s end-to-end.
- Current pre-FFI Python preparation is itself material at roughly 49.7 s topology and 23.2 s union for n=9.

Therefore the work will optimize Python preparation and Rust semantic construction, while profiling—but not duplicating—the migration task’s P-kernel lowering and serialization work.

## Implementation

### 1. Bootstrap, ownership, and profiling

- Before any implementation edit, write everything inside this `<proposed_plan>` block verbatim to `docs/development/recurrence/RECURRENCE_GENERATION_OPTIMIZATION_PLAN.md`.
- Immediately afterward create the active goal, without a token budget: “Implement and fully validate the verbatim recurrence-generation optimization plan in `docs/development/recurrence/RECURRENCE_GENERATION_OPTIMIZATION_PLAN.md`.”
- Record the starting commit, dirty-state inventory, toolchain, native build provenance, and isolated baseline artifacts before modifying code.
- Keep all outputs under `.artifacts/recurrence-generation-opt`; never escalate commands or write outside the workspace.
- Wrap every potentially RAM-intensive build, generation, profiling, and validation command with `tools/ci/memory_watchdog.py --limit-gib 30`.
- Extend existing internal phase telemetry, without adding public CLI/API options, to measure:
  - process/model/color preparation;
  - projection, semantic-catalog, crossing, and schedule construction;
  - numerical-current warm-up, baseline certification, each probe, and final generation;
  - Python normalization, serialization, and Python↔Rust transformations;
  - Rust support indexing, candidate enumeration, feasibility rejection, transition matching, color acceptance, current/interaction insertion, closure processing, canonical emission, and serialization;
  - `Evaluator.load`, `get_instructions()`, `repr`, P-kernel translation, SIMD preparation/sealing, and storage serialization.
- Include operation counters and peak RSS alongside time: support buckets, theoretical versus visited pairs, accepted transitions, cache hits/misses, clones/hashes, currents/interactions, serialized bytes, and certification relations.
- Send the measured evaluator/translation/serialization breakdown and concrete upstream candidates to `pyAmpliCol - migrate arena model`.

### 2. Exact-output Python improvements

- Construct each immutable color plan, projection, crossing map, and semantic catalog once per generation and reuse it across warm-up, scheduling, and native calls. Cache only under complete semantic keys and scope caches to the generation session.
- Generate all schedule digests and serialized views from one normalized schedule object instead of repeating normalization and traversal.
- Replace repeated per-record validation and object conversion with a single validated boundary followed by trusted internal bulk construction; retain all existing external signatures and validation behavior.
- Reuse stage/support indexes, exact executor descriptors, warm-up contexts, and candidate spools instead of rebuilding equivalent dictionaries, tuples, and manifests.
- Eliminate repeated Python↔Rust round trips and duplicate materialization where the same catalog or schedule is already available in canonical form.
- Preserve authenticated numerical-reuse behavior exactly. Do not weaken probe count, precision, evidence digests, independent schedules, or failure handling.

### 3. Exact-output Rust semantic-construction improvements

- Represent ordinary external support with compact bitmasks and retain an exact fallback beyond the compact width.
- Build support/stage indexes and enumerate only admissible disjoint support pairs, including every orientation and multiplicity required by existing semantics, instead of scanning Cartesian stage products.
- Hoist decoded transition, witness/source, coupling-slot/order, sector, and closure metadata so accepted-pair processing uses numeric indexes rather than repeated decoding or hashing.
- Add exact forward-feasibility, closure/source, and lane-demand indexes to reject impossible candidates before allocating interactions or cloning metadata.
- Index and memoize color-target acceptance using canonical fragment identities and sector bitmaps, keyed by the complete semantic context.
- Use transient side maps for exact current/interaction hash-consing. Retain borrowed or numeric references while testing candidates and clone only accepted records.
- Replace repeated quadratic cyclic canonicalization with an exact linear minimal-rotation implementation where profiling confirms that path is active.
- Preserve the previous deterministic canonical emission order, signs, duplicates, contribution ordering, IDs, closure mapping, and persisted bytes. Drop construction-only support, color, transition, and hash indexes before lowering.
- Remove any optimization that fails exact-output tests or regresses the corresponding high-multiplicity phase.

The implementation must not introduce `DirectApplication`, `DirectTable`, scalar-plane lowering, recurrence epilogue generation, new row/plane bindings, parameter broadcast ownership, scratch scheduling, or runtime row scheduling. Those remain owned by the SymJIT 2.22 migration and the stable `pyamplicol-symjit-plane-application-v1` / `pyamplicol-recurrence-plane-binding-v1` boundary.

### 4. Original AmpliCol comparison and redesign scouting

Create `docs/development/recurrence/RECURRENCE_GENERATION_OPTIMIZATION_LEGACY_COMPARISON.md` from a read-only audit of pinned legacy revision `79c96cecf2a722e50c3d2030b6894d755f96518a`. Compare legacy and current pyAmpliCol across storage/lifetimes, recurrence ordering, overwrite/reuse, parameters/couplings, batching/vectorization, and generation shortcuts.

Classify these as directly transferable:

- compact support masks and rejection before materialization;
- support/stage buckets, disjoint-pair indexing, closure feasibility, and backward demand;
- exact keyed current lookup and transient hash side indexes;
- construction-versus-persisted lifetime separation and compact side arrays;
- hoisted immutable transition and coupling-slot metadata;
- homogeneous late grouping as a profiling-backed lowering opportunity.

Classify these as model-specific or unsafe:

- single-point tolerance-based current/interaction merging;
- numerical helicity and same-flavour inference;
- fixed-width masks, fixed color limits, and hard-coded QCD symmetries;
- literalized masses, widths, couplings, and fixed QCD/EW power extraction;
- full resident value arrays, generated-Fortran execution, and model-specific vertex grouping.

Do not modify the legacy checkout. If it builds with the existing workspace toolchain without patches, network access, or escalation, benchmark it read-only under the watchdog; otherwise record the static comparison and build limitation. Legacy behavior is inspiration, not a semantic oracle, and cannot justify artifact, ABI, parameter-layout, scratch, or runtime changes.

Create `docs/development/recurrence/RECURRENCE_GENERATION_OPTIMIZATION_SCOUTING.md` for reviewed-only designs. For each, document expected generation impact, RAM implications, runtime risk, implementation depth, ABI consequences, and migration dependencies:

- retained native generation sessions/cross-call caches and a zero-certified-relation fast path;
- batched native arbitrary-precision certification probes;
- shared structural DAGs with per-flow overlays;
- persistent or chunked color arenas;
- lazy/streamed construction and deterministic parallel enumeration;
- structural relation proofs or exact prefilters;
- persisted homogeneous row layouts, liveness scratch reuse, and other plane/runtime changes.

None of these redesigns will be implemented in this round.

## Validation and Acceptance

### Semantic and numerical correctness

- Run all recurrence unit and integration suites, especially exact union, topology schedules, catalog construction, projection, pairing, sharing, preflight, prepared execution, selected-flow consistency, three-line execution, process-set sharing, and numerical-current warm-up.
- Compare baseline and candidate canonical semantic censuses independently of transient enumeration: current identity/support/type/color/helicity, transition and coupling identity, source/witness data, signs, contribution multisets, interaction endpoints, closure maps, selector axes, normalization, and schedule digests.
- Require exact bytes for runtime-bearing recurrence payloads and evaluator/model kernels. Permit differences only in an explicit allowlist of timing/provenance metadata; reject unknown differences.
- Verify numerical parity using identical parameters, phase-space points, selectors, and existing strict tolerances for:
  - `d d~ > Z g g g g`
  - `d d~ > t t~ g g g`
  - `g g > g g g g`
- Cover LC topology, LC all-flow union, NLC, and full color wherever the baseline supports them. Any missing baseline mode must be reported, not silently skipped.

### Generation performance and RAM

Use isolated cold-cache roots, alternate baseline/candidate order, and record phase timing plus peak RSS.

- n=2–4: both LC layouts, three repetitions, five-minute timeout.
- n=5–7: both layouts, three repetitions, fifteen-minute timeout.
- n=8: both layouts, three final repetitions; one-hour topology and two-hour union timeout.
- n=9 topology: authenticated baseline and at least two candidate runs, with a third if variation exceeds 5%; two-hour watchdog timeout and a required candidate result below 3,600 seconds.
- n=9 union: timeout-censored baseline/candidate scouting up to six hours, polled regularly without blocking user updates.
- Use the exact ladder `d d~ > Z + (n-1)*g`.

Acceptance requires:

- topology n=9 completes below one hour and is faster than baseline;
- union n=8 median generation improves materially, targeted at 20% or more;
- no median generation regression above 5% at lower multiplicities;
- peak generation RSS remains below 30 GiB and does not regress by more than 10% for the same case;
- secondary NLC/full-color canaries remain correct and show no material generation regression.

### Runtime no-regression gate

- Benchmark n=6 and n=7 for both LC layouts at batch sizes 1, 128, and 1024, using seven isolated warmed subprocess samples, two warm-ups, and a five-second measurement window.
- If runtime payloads and evaluation implementation are demonstrably byte-identical, record that proof and run load/evaluate canaries.
- Otherwise require the candidate’s 95% confidence upper bound to show no slowdown in every required cell. Expand inconclusive cells to at most 21 samples; reject the runtime-affecting change if still inconclusive.
- Reject any statistically supported runtime slowdown or material runtime RSS increase.

## Final Deliverables and Assumptions

- Public Python APIs, CLI commands/options, native ABI, recurrence plane schemas, parameter semantics, selectors, and runtime behavior remain unchanged.
- Safe implementation changes remain upstream of the migration-owned plane application, binding, scratch, epilogue, and row-scheduling boundary.
- The final report will contain reproducible guarded commands, baseline/candidate commits, phase and counter breakdowns, generation/RSS tables, artifact comparison, numerical/runtime results, retained and rejected optimizations, the separate legacy comparison, and detailed redesign proposals.
- If a profiling command cannot run without escalation, use a non-escalated alternative and record the limitation; never request or attempt escalation.
