# Eager and Compiled Direct-Arena Session Handoff

Status date: 2026-07-25  
Operational worktree:
`/Users/vjhirsch/HEP_programs/pyAmpliCol/PREPARE_STANDALONE_PYAMPLICOL/pyamplicol_eager_and_compiled_arena`  
Feature branch: `eager_and_compiled_arena`

## 2026-07-26 continuation status

This ledger records the state at its original handoff and is intentionally
not rewritten retroactively. The implementation history after that handoff
has superseded several unfinished items below:

- eager and compiled Direct-Arena execution is implemented for LC topology,
  LC all-flow union, NLC contracted color, and full contracted color;
- compiled precision-32 NLC/full contraction and the later report physics
  canaries are fixed;
- exact-source all-JIT, four-independent-quark-line, color, selector, API,
  source, artifact, and runtime-identity gates now exist;
- the complete 168-cell AArch64 matrix and its eight-shard x86-64 equivalent
  have strict numerical, performance, resource, provenance, and postflight
  acceptance logic;
- the authoritative qq_Z6g gate covers built-in/UFO-SM and
  topology/union workloads, compares eager/compiled/recurrence with seven
  paired subprocess rounds, and requires compiled batch-128/1024 performance
  to remain within 15% of recurrence;
- a pinned original-AmpliCol producer authenticates both selected-flow and
  all-flow evidence, including the generated executable row and complete
  generated-library payload.

The migration goal is still active at this update. Remaining work is
execution rather than another runtime design change: freeze one final clean
feature SHA, build its exact candidate, run the full local and x86 acceptance
campaigns, populate the authenticated qq_Z6g comparison, land that SHA on
`main`, and complete the profile-scoped 742-cell report regeneration and
publication audit. The exact repository and evidence manifests remain
authoritative over prose in the older sections below.

## Mandatory restart directive

The next implementation session must begin by:

1. Reading this file and
   `docs/development/arena/EAGER_AND_COMPILED_ARENA_PLAN.md` completely.
2. Fetching remote state and treating the current repository, not remembered
   chat context, as authoritative.
3. Assigning completion of **every unfinished item in this handoff** as its
   active goal with the goal tool. Preserve the original goal statement
   verbatim:

   > Replace the existing eager and compiled-JIT internal runtimes with
   > separate Direct-Arena execution lanes based on the accepted recurrence
   > Direct-Arena substrate, preserving all public Python, CLI, Rust, C, C++,
   > and Fortran APIs and physics semantics while intentionally replacing
   > prior internal artifact and evaluator ABIs, retaining applicable LC,
   > NLC, and full-color optimizations, and demonstrating pointwise parity
   > plus robust portable wall-time gains across complete runtime-selector
   > workloads.

4. Never requesting command escalation. Use the dedicated worktree,
   worktree-local caches, and `/private/tmp`; if the sandbox blocks an
   operation, use a sandbox-safe route. Guard every substantial build,
   generation, or benchmark with
   `tools/ci/memory_watchdog.py --limit-gib 30`.
5. Rebuilding one exact-source candidate before publishing new benchmark
   evidence. Do not combine an older native module, a dirty source tree, or
   timing workers from different commits.

This document is a continuation ledger, not a claim that the full migration
goal has passed its acceptance gates.

## Authoritative Git state at handoff creation

- Accepted recurrence and starting main SHA:
  `671a0dc5f6f486d0207ce87b871e7f964fdea073`.
- That main SHA is an ancestor of the feature branch.
- Last implementation commit before this handoff:
  `25f9770623e3d8466f1690b0da2f22f3feacb7dd`.
- Runtime cutover milestone:
  `15c0596dd3088861d903a2927ea33042191c1402`.
- The remote feature branch and `main` were successfully fast-forwarded
  through the initial handoff commit
  `efe1bb0281c1ec81a83b9a80dfe386a29b0b5f55`. The documentation-only commit
  containing this update is its direct descendant; use `git rev-parse
  origin/main` for that final exact SHA.
- The source tree was restored clean before writing this document. An
  unsuccessful exact-color experiment was deliberately reverted and is not
  part of the feature history.

The coherent feature series after `671a0dc` comprises 42 commits, beginning
with the approved plan/baseline/substrate extraction and ending, before this
handoff, with:

- `d0b90b4` — require compiled plane-arena artifacts.
- `1b872a2` — eliminate compiled LC replay allocations.
- `d804ef1` — require Direct-Arena eager artifacts.
- `ddc7358` — preserve eager Direct descriptors on append.
- `7604e57` — eliminate eager native packing and result allocations.
- `15c0596` — complete native eager and compiled arena cutover.
- `7f35266` — keep process-local eager packs exact-loadable.
- `25f9770` — correctly benchmark compiled O3 when a prepared-model bundle is
  the model source.

Use `git log --reverse origin/main..HEAD` for the complete exact series.

## What is implemented

### Shared Direct-Arena substrate

- Lane-neutral deterministic liveness/range allocation and the aligned
  split-complex, component-major, points-contiguous arena substrate were
  extracted from the accepted recurrence mechanics.
- Recurrence remains behind its own adapter; recurrence-specific currents,
  closure proofs, topology/replay builders, row semantics, and construction
  logic were not imported into eager or compiled execution.
- Eager, compiled, and recurrence retain separate execution lanes.

### Compiled-JIT production lane

- Production compiled artifacts require the new plane-arena capability and
  fail closed on old compiled artifacts with an actionable regeneration
  error. There is no dense production fallback.
- Fused process-specific stages, chunk boundaries, funclet compression, and
  fanout order remain intact.
- JIT and native stage applications bind directly to canonical current and
  amplitude planes through the compiled DirectApplication ABI.
- Stage/leaf input packing, output scatter, and amplitude remapping were
  removed from production arena execution.
- Plane-native totals and resolved reducers retain LC reduction order and the
  compact repeated-color, Hermitian-chain, and Walsh reductions previously
  developed for NLC/full color.
- Deterministic bounded tiling protects parent/stage locality. Earlier
  persistent whole-state and overly broad AoSoA experiments that regressed
  locality were not retained.
- LC replay scratch/result allocation was removed from the warmed native
  path.

### Eager production lane

- Production eager artifacts require `eager-direct-arena-v1` and fail closed
  on old eager artifacts. No packet gather/evaluate/scatter production
  fallback or dual artifact representation remains.
- The eager-specific plan models source initialization, invocation/fanout,
  finalization, closures, coupling/output factors, selectors, and ordered
  multi-destination stores directly against arena planes.
- Whole-plan execution uses preallocated persistent workspace and bounded
  point tiles.
- The public resolved path consumes borrowed flat momentum input and avoids
  nested momentum materialization/crossing/result copies.
- Selector-set caches are validated before insertion and bounded to 64
  entries.

### Native CPP/ASM prepared lanes

- Native eager CPP and ASM model packs publish a genuine plane-native
  row-outer/point-inner DirectTable derived from retained Symbolica
  instructions.
- The DirectTable emitter supports two- and four-wide SIMD, odd scalar tails,
  ordered overwrite/add fanout, and x86 AVX2 four-wide selection.
- A fixed metadata structure authenticates ABI version, flags, strides,
  input/output shape, SIMD width, callable role, target, and the exact
  evaluator-state hash before Rust casts or invokes the function.
- Native compiled CPP/ASM process stages publish only `.direct` process
  libraries plus exact evaluator states. Dense process-stage libraries and
  Python fallback production were removed.
- Both public `cpp` and `asm` configurations use the plane-native callable
  ABI. The new DirectTable body is compiler-generated SIMD C++, not custom
  handwritten/inline assembly; the recurrence raw callable remains separate.

### Artifact cutover and exact-pack publication

- Internal eager and compiled artifact/evaluator ABIs were intentionally
  replaced; old artifacts must be regenerated.
- Process-local eager packs now drop irrelevant model-wide recurrence
  companions. This fixed precision-32 eager loading after process-local kernel
  pruning.
- Recurrence-bearing or mixed artifacts conservatively retain the full
  prepared-kernel inventory required by their model-wide recurrence
  companions.
- Artifact publication reconstructs `PreparedKernelPack` before committing
  the manifest, so dangling kernel/catalog references fail during generation.
- Append mode preserves new direct descriptors and accumulated capabilities.

## Exact candidate and fresh artifact evidence

The most recent exact-source candidate built before this documentation-only
commit was:

- Source revision:
  `25f9770623e3d8466f1690b0da2f22f3feacb7dd`.
- Version: `0.1.0.dev0+candidate.29e71015537c`.
- Native input digest:
  `64356e4d0143bfc484a6d236a7894174e142f298036c9a65200914e6fe55b493`.
- Wheel:
  `/private/tmp/pyamplicol-eager-compiled-arena-current-build/wheelhouse/pyamplicol-0.1.0.dev0+candidate.29e71015537c-cp311-abi3-macosx_11_0_arm64.whl`.
- Wheel SHA-256:
  `a88cd59d1bb3ff1c6fd86ec816e80a8c364fec3bd5ed33b22fde7ec295755515`.
- Wheel size: 6,795,805 bytes.
- Extracted site:
  `/private/tmp/pyamplicol-eager-compiled-arena-current-build/site`.
- Cached rebuild peak RSS after Rust was already built: 0.066 GiB.
- The original clean `15c0596` release/LTO build took about 19m55s and peaked
  at 1.925 GiB; subsequent Python/document-only exact-source rebuilds reused
  the release objects and completed in about two seconds.
- Candidate Cargo overlay used Symbolica 2.2.0 and SymJIT 2.21.1.

Current prepared model:

- `.agent-work/final-jit/builtin-jit-o2.pyamplicol-model`
- Built from the exact candidate, built-in SM, portable JIT O2.
- 61 kernels, 5.053 s total preparation, 0.248 GiB peak RSS.

Fresh small native artifacts:

- `.agent-work/final-native/artifacts/eager-cpp-fix`
- `.agent-work/final-native/artifacts/compiled-cpp`
- `.agent-work/final-native/artifacts/eager-asm-fix`
- `.agent-work/final-native/artifacts/compiled-asm`

CPP and ASM both passed eager-vs-compiled totals, f64 resolved components, and
precision-32 resolved components for `d d~ > Z+2g` at `rtol=1e-12`,
`atol=1e-15`. The maximum absolute f64/exact differences were about
`5.42e-20`/`5.76e-20`. Fresh generation and post-build validation passed
under the watchdog.

Fresh representative JIT artifacts:

- `.agent-work/final-color/{eager,compiled}-{lc,nlc,full}`
- Process: `d d~ > Z+3g`.
- Eager uses the prepared portable O2 table; compiled uses fused process-local
  JIT O3.
- All six generations and their post-build validation passed; combined peak
  RSS was 0.224 GiB.
- f64 eager-vs-compiled totals and resolved values passed for batches
  1/127/129 in LC, NLC, and full color.
- Runtime helicity selection passed in all three accuracies.
- Runtime LC flow selection passed.
- LC f64 and precision-32 cross-lane resolved parity passed.

Focused test/build evidence accumulated for the cutover includes:

- Ruff green on changed Python/test paths.
- Focused Python suites covering producers, descriptors, arena execution,
  selectors, append, exact eager loading, and benchmark provenance.
- Fresh native CPP/ASM preparation/generation/post-build checks.
- Candidate Cargo checks for `f64-compiled` and `f64-symjit` near 1.01 GiB
  peak RSS.
- All-feature eager Rust test compilation near 2.01 GiB peak RSS.
- Native emitter/compiler tests: 7 passed.
- The final native arithmetic comparison above.

The raw tracked Cargo lock still carries the release-contract dependency
configuration; candidate overlay builds are the authoritative contributor
route for SymJIT 2.21.1. Do not interpret a raw, unmanaged Cargo invocation
against another dependency set as candidate evidence.

## Important unfinished or failed gates

Everything in this section remains part of the goal. None may be silently
reclassified as optional.

### 1. Compiled precision-32 NLC/full resolved-color defect

On the fresh `d d~ > Z+3g` artifacts:

- f64 eager and compiled agree in NLC/full at roughly `5e-23` maximum absolute
  resolved-component difference.
- Eager precision-32 resolved sums agree with eager f64 totals.
- Compiled precision-32 resolved sums are low:
  - NLC exact/f64 ratio `0.8069352723`.
  - Full exact/f64 ratio `0.7869288189`.
- LC precision-32 remains correct to about `1e-15`.
- All 48 nonzero helicities are affected in NLC/full, pointing to
  cross-coherent-group phase/contraction behavior rather than arena f64
  execution.

An attempted retained-helicity-domain fix passed 28 unit tests but did not
change these real ratios. It was fully reverted and must not be resurrected
without new evidence. The observed artifacts use singleton amplitude-route
domains, so global-flip alias routing was not the root cause.

Next work must compare the exact compiled amplitude groups before contraction
with the eager exact groups and audit
`src/pyamplicol/runtime/symbolica_exact.py`, especially repeated compact color
contraction, group ordering, conjugation, and phase/factor application. Add a
real NLC/full cross-lane regression, not only a synthetic reducer fixture.

### 2. Authoritative qq_Z6g timings are not complete

No new candidate qq_Z6g timing may be quoted as authoritative yet.

The topology run generated and numerically validated complete eager, compiled
O3, and recurrence artifacts, then entered the required 63-worker
seven-round interleaved schedule. It aborted after worker 46 because an
independent exact-color audit modified tracked source during the run. The
harness correctly failed closed rather than mixing source states. Do not
salvage those partial timings.

Generated topology artifacts are under:

- `.agent-work/final-z6g/builtin-topology/eager-jit-o3-topology-replay`
- `.agent-work/final-z6g/builtin-topology/compiled-jit-o3-topology-replay`
- `.agent-work/final-z6g/builtin-topology/recurrence-topology-replay`

The harness bug found during this attempt is fixed in `25f9770`: an explicit
prepared-model source does not turn a process-local compiled O3 evaluator into
O2. Eager and recurrence execute the immutable prepared O2 callables;
compiled still materializes fused O3 stages.

Rerun from a clean exact-source build:

```text
SYMBOLICA_LICENSE=dcec4a5e#6a95649c#7dca8216-8afe-57c8-975e-03eb5e68e4ee \
PYTHONPATH=/private/tmp/pyamplicol-eager-compiled-arena-current-build/site \
../pyamplicol_compiled_dag_optimization/.venv/bin/python \
tools/ci/memory_watchdog.py --limit-gib 30 -- \
../pyamplicol_compiled_dag_optimization/.venv/bin/python \
tools/developer/recurrence_z6g_benchmark.py \
--output-root .agent-work/final-z6g/builtin-topology \
--result-json .agent-work/final-z6g/builtin-topology/result.json \
--prepared-model .agent-work/final-jit/builtin-jit-o2.pyamplicol-model \
--jit-optimization-level 3 --lc-flow-layout topology-replay \
--mode eager --mode compiled --mode recurrence \
--batch-size 1 --batch-size 128 --batch-size 1024 \
--target-runtime 0.1 --minimum-samples 7 --subprocess-samples 7 \
--warmup-runs 2 --validation-samples 3 \
--color-flow flow:2,4,5,6,7,8,9,1 --force
```

Then run the complete all-flow-union artifact with the runtime selector:

`h:-1,+1,-1,+1,-1,+1,-1,+1,-1`.

Do not generation-fix flow or helicity for headline comparisons. Cross-check
each runtime-selected result against `evaluate_resolved()` from that same
complete artifact.

Frozen same-host references, which must be labelled as historical references
rather than new candidate measurements:

- Accepted recurrence built-in topology means: 37.690 us/point at batch 128,
  37.398 us/point at batch 1024.
- Accepted recurrence built-in union means: 335.793 us/point at batch 128,
  325.772 us/point at batch 1024.
- Accepted recurrence UFO topology means: 37.986 and 37.879 us/point.
- Accepted recurrence UFO union means: 365.471 and 365.043 us/point.
- Original AmpliCol selected-flow/helicity-sum: 39.401 us/point total,
  38.945 us/point amplitude.
- Original AmpliCol all-flow/single-helicity dynamic total:
  312.385 us/point; amplitude 306.148 us/point.
- Pre-arena compiled topology: 54.024 and 56.880 us/point.
- Pre-arena compiled union: 312.352 and 340.892 us/point.

The recurrence values above are old acceptance means/SD, not median/MAD; the
new harness produces seven-subprocess medians/raw MAD. Report the statistics
honestly and do not directly relabel means as medians.

### 3. Required performance matrix remains incomplete

The new arena implementation still needs authoritative wall-time captures for:

- `u u~ > Z+6g`, built-in and UFO-SM, both LC selector workloads.
- Medium LC/NLC/full cases from the approved plan.
- Color-heavy `g g > t t~+3g` and `g g > t t~+4g` in NLC/full.
- Eager and compiled separately, including generation time, payload, load
  time, RSS, warmed allocations, and packet/gather/scatter/remap counters.
- Same-host AArch64 results and an x86-64 validation runner.

The acceptance assertions (10% gain somewhere in each lane, no required cell
over 3%/three MAD regression, compiled qq_Z6g within 15% of recurrence, zero
warmed allocations/traffic, and generation/payload resource limits) have not
been proven as a complete matrix.

### 4. Full API/release closure remains incomplete

The implementation preserves the public API surfaces in focused checks, but
the final broad gates still need:

- Full focused/full Python and Rust suites from one exact source revision.
- Ruff and formatting checks from that same revision.
- C, C++, Fortran, Rust, and Python standalone API smokes for both eager and
  compiled current artifacts.
- Malformed artifact, panic containment, parameter, alias, structural-zero,
  global selector, and per-point selector coverage through the final build.
- Wheel integrity, candidate deployment/source gate, and clean-install smoke.
- Cross-architecture x86-64 evidence.

The repository's strict release policy may still intentionally fail closed
for unverified dependency locks and forbidden candidate sdists. Do not weaken
that policy; distinguish expected policy failure from candidate wheel/source
and deployment evidence.

### 5. Documentation and final integration

- Add a final evidence document with exact final SHA, complete statistics,
  resource measurements, and any intentionally deferred work.
- Fetch the latest main before final integration.
- If new main commits exist, incorporate them without dropping this feature
  series and rerun the material gates.
- Push coherent follow-up fixes to a feature branch before updating main.
- Mark the original migration goal complete only after the full requirement
  audit is genuinely satisfied. This handoff explicitly records that it is
  not yet satisfied.

## Immediate safe next steps

1. Confirm remote main/feature SHAs and the exact merge SHA recorded alongside
   this handoff commit.
2. Build a fresh candidate from that exact SHA using
   `.agent-work/build_current_candidate_site.py` under the watchdog.
3. Fix the compiled precision-32 NLC/full contraction defect with a real
   artifact regression.
4. Commit the fix so the benchmark worktree is clean.
5. Rerun topology and union qq_Z6g captures without concurrent edits.
6. Run the bounded remaining color/API/resource gates, write final evidence,
   and integrate the follow-up.

Do not spend cycles repeating already-green micro-tests unless a relevant
source area changes. Wall time per point on complete runtime-selector
workloads remains the ultimate performance arbiter.
