# Recurrence-generation optimization report

Status: implementation complete.  Guarded evidence through n=8, the
diagnostic n=9 topology point, and the paired fail-closed n=9 union scout are
recorded below.  On 2026-07-31 the user explicitly replaced the remaining
release-readiness campaign with a short integration finalization; no further
heavy generation or runtime campaign is required for this implementation
round.

This report implements the plan in
`RECURRENCE_GENERATION_OPTIMIZATION_PLAN.md`.  It covers only recurrence
generation.  Public Python APIs, CLI commands and options, native ABI,
recurrence plane schemas, parameter and selector semantics, runtime
evaluation, and runtime-bearing recurrence payloads remain unchanged.
The later user-authorized simplification of artifact trust and post-build
validation is called out separately below because it deliberately supersedes
the plan's original exact-whole-artifact constraint.

## Provenance and safety

The authenticated baseline is Git revision
`172e58fd33a3c65563866c50cfbb5e1ddcd7b302`, initially on a clean `main`
worktree.  Its native build-input digest is
`865408b317aa421fdc19f519ca8d8317d49cfbbe75a0f6135c1afe0b06c74bfe`;
the originally captured baseline extension SHA-256 is
`8137fb99a8682a5a3a5c558f8dc8f02197ceec03651d9a0d47732079c0e9245d`.
The explicit prepared-model SHA-256 is
`772ead40a64bc01f37725ad3173a6126765b3f2be70396acbac64bbfa100cf86`.
Complete baseline provenance is retained below
`.artifacts/recurrence-generation-opt/baseline/provenance/`.

Because that wheel authenticated the main checkout path, it correctly became
unusable once implementation work changed the main worktree.  A second clean
baseline wheel was built from an isolated worktree at the same revision.  Its
native build-input digest is
`8e2040eb2eba7345156fc91fc4a71a053f2ed8f3187d2e091673380ed7e73e8b`,
wheel SHA-256 is
`f5b330686bc9bd856d9c71b5e5ddb2209d4ae622d5fc8fda341f0c9d31855008`,
and extension SHA-256 is
`5881dd44bce0ca41dbff23035ebffbc3ff2ce6ffaf47a1a0264f9a9686a5aab8`.
Only checkout-local absolute dependency paths and their authenticated
configuration digest differ from the first baseline build.

The final frozen candidate implementation is revision
`4e2b1e02dddde2d55b7250cbd52a93001f09b2c2`, source tree
`a0d08154574d57f0cf91a380125bae18381ecb83`, built and installed from its own
clean detached worktree below `candidate-final/`.  Its authenticated native
build-input digest is
`67eee8bc0950f2e3921e0bb725c6a680afea61a6df7e978cae91ca6ff1e2af31`,
wheel SHA-256 is
`26f9e9ecb036094f02387cc169d4fa6189770e90ead41d488afb8ff0abcbb982`,
and installed extension SHA-256 is
`86ef0126995b68e77c01d3dc63283dabc2956359ce202765dfa971ad96edb301`.
All 500 hashed wheel `RECORD` entries verify.  Every installed entry is
byte-identical to the wheel except the unhashed, installer-regenerated
`RECORD`, and all audited import origins resolve below
`candidate-final/install/site-packages`.  The earlier `cf990b7` wheel remains
useful historical regression evidence but is superseded by this final
identity.

One discarded provenance-audit Cargo command is excluded from every accepted
result.  Its release wrapper dropped the requested offline environment,
downloaded dependencies, and matched zero tests.  The complete incident is
retained in
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-provenance/DISCARDED_CARGO_RUN.md`;
no accepted validation, parity, or performance result depends on that command.

All build, generation, profiling, and validation commands that could consume
material RAM are process-tree guarded with:

```bash
.venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- COMMAND
```

Temporary, cache, artifact, and benchmark roots are isolated below
`.artifacts/recurrence-generation-opt`.  No command was escalated.

## Diagnosis

Historic measurements did not support the hypothesis that native lowering
alone dominates end-to-end generation:

| Case | End-to-end | Rust construction | Direct lowering |
|---|---:|---:|---:|
| LC topology n=9 | 814 s | 61.4 s total native | 47.0 s |
| LC union n=8 | 1,263 s | 82.3 s semantic construction | 10.0 s |

Current pre-FFI Python preparation was independently material at about 49.7 s
for topology n=9 and 23.2 s for union n=9.  Repeated schedule normalization
and traversal accounted for about 14.48 s and 3.93 s respectively; topology
n=9 columnar preparation accounted for about 13.15 s.

An independent census of all 59 prepared evaluator states found
`Evaluator.load`, `get_instructions()`, and `repr(program)` negligible in
isolation:

| Operation | Total |
|---|---:|
| `Evaluator.load` | 0.007178499 s |
| `get_instructions()` | 0.002233076 s |
| `repr(program)` | 0.001650869 s |
| serialized state bytes | 49,817 B |
| instruction-representation bytes | 79,023 B |

The complete census is
`.artifacts/recurrence-generation-opt/profiles/evaluator-census.json`.
Translation, SIMD preparation/sealing, and storage serialization remain owned
by the coordinated SymJIT 2.22 migration.

## Retained exact-output implementation

### Python preparation

- Normalize and traverse each schedule once to derive both native and request
  digests.
- Construct a selected-sector projection once and derive exact selected-flow
  variants by immutable replacement.
- Reuse the complete immutable color plan for minimal coupling inference.
- Cache canonical direct-template catalog JSON once per generation model.
- Adopt already-owning exact C-contiguous columnar arrays without a second
  copy, while retaining unconditional boundary validation.
- Reuse immutable warm-up geometry, contracts, defaults, candidate indexes,
  and candidate spools.
- Remove an immediate compressed-payload pickle round trip.
- Record preparation, warm-up, certification, native, and serialization phase
  timings only in generation provenance extensions.

### Rust semantic construction

- Encode ordinary external support in exact compact masks through 128 sources,
  with a validated variable-width fallback beyond 128.
- Build support/stage buckets and enumerate only disjoint support pairs while
  preserving every original orientation, multiplicity, and ordinal.
- Predecode transition, witness/source, coupling-slot/order, exchange,
  operation, and static-result metadata.
- Build exact forward-feasibility and lane-specific backward-demand indexes
  and reject impossible candidates before merged-support, helicity, and key
  allocation.
- Index color targets by canonical fragment identity and sector bitmaps, with
  accepted-only exact memoization so rejected forests do not accumulate.
  A dense bitmap per fragment was rejected during adversarial review: at
  40,320 n=9 union sectors and approximately 369,122 distinct fragments, its
  raw words alone would require about 1.73 GiB and each cache miss would copy a
  5,040-byte bitmap.  The retained representation uses monotonically built
  exact sector postings, frozen sparse unless a dense bitmap is strictly
  smaller.  Multi-component acceptance intersects the lowest-cardinality
  posting, or all dense words directly, without cloning a posting.
- Precompute closure-sector color targets and contracted-color ownership once,
  index lane-local anchor/complement currents by exact support, and index
  prepared closure rows by input-state pair while retaining the original
  anchor, complement, closure-template traversal order.
- Retain transient support/key/hash indexes only during construction and release
  stage pair caches once schedule plans retain their current references.
- Delay cloning and reflection proof hashing until candidates have passed
  cheaper exact filters.
- Use Booth minimal rotation for exact linear cyclic canonicalization.

Canonical emission order, IDs, signs, duplicates, contribution order,
interaction endpoints, closure mapping, selector axes, and persisted runtime
bytes are validation requirements rather than implementation assumptions.

The first frozen post-unit wheel, revision `825af33`, exposed one important
reachability regression in the real prepared integration matrix: hoisting
transition output factors evaluated a zero, but unreachable, coupling
component while constructing the global catalog. The retained fix keeps the
coupling metadata hoisted but defers coupling authentication and component
selection until the transition passes the same quantum, coupling-order, and
structural-demand filters as the baseline. A focused Rust regression test
proves that catalog preparation accepts the unreachable row while consuming
that row still produces the exact baseline error. The rejected wheel and its
paired baseline/candidate evidence are retained under
`.artifacts/recurrence-generation-opt/candidate/rejected-825af33/` and
`.artifacts/recurrence-generation-opt/validation/baseline-integration-attribution/`.

## Rejected or deferred implementation

The implementation does not add or reintroduce `DirectApplication`,
`DirectTable`, scalar-plane lowering, plane or row bindings, broadcast or
scratch ownership, recurrence epilogues, or runtime row scheduling.  These are
owned by the stable
`pyamplicol-symjit-plane-application-v2` /
`pyamplicol-recurrence-plane-binding-v2` migration boundary.  The verbatim
plan records the application-v1 and binding-v1 names that were current when it
was approved; the migration subsequently advanced both schemas to v2 without
changing this optimization round's ownership boundary.

Retained native sessions, zero-relation finalization, batched
arbitrary-precision probes, shared structural DAGs with flow overlays,
persistent color arenas, streamed construction, deterministic parallel
enumeration, new structural proofs, and persisted row/liveness changes were
not implemented.  Their impact, RAM risk, implementation depth, ABI
consequences, and migration dependencies are documented in
`RECURRENCE_GENERATION_OPTIMIZATION_SCOUTING.md`.

## Correctness and exact-artifact validation

The accepted final results below bind to the frozen `4e2b1e0` source, wheel,
installed Python payload, and native extension identified above.  Discarded
collection attempts, the discarded zero-test Cargo command, and intermediate
candidate wheels are not counted.

### Test suites

| Suite | Result | Peak RSS / footprint | Notes |
|---|---|---:|---|
| Rust construction namespace | 55 passed after excluding 1 known projection panic | 4.323 GiB initial cold namespace | the isolated panic reproduces on the authenticated baseline |
| Rust color namespace | 8 passed | 0.015 GiB cached guard | exact color construction/canonicalization coverage |
| Rust Python numerical-evidence namespace | 27 passed, 1 ignored manual benchmark | 2.324 GiB | `numpy` feature enabled; zero failures |
| Final installed-wheel exact-session suites | 68 unique passed | 0.086 GiB maximum guard | recurrence warm-up 33, recurrence exact union 13, generic warm-up/session 22 |
| Focused installed-wheel Python units/tooling | 546 passed, 1 known baseline failure, 1 gated skip | 0.700 GiB | zero candidate-only failures |
| Prepared/selected-flow/three-line/process-set integration | 19 passed, 13 baseline-matched failures, 1 gated skip | 2.417 GiB | exact failing node set and normalized messages match baseline |
| Numerical-parity validator tooling | 17 passed | guarded; see retained logs | fail-closed schema, provenance, selector, fixture, component and watchdog checks |
| Supported numerical-parity cells | 4 passed, 2 baseline-unavailable | see parity table below | every supported total and resolved component is exactly equal |
| Full Python recurrence sweep | deferred | not run | explicitly moved to the later release-readiness pass by the user |
| Representative n=8 semantic census and artifact allowlist | 4 passed | two layouts, two exact gates per layout | recurrence-runtime bytes match, semantic differences are zero, and metadata-only differences are allowlisted |
| Diagnostic n=9 topology semantic census and artifact allowlist | 2 passed | one layout, two exact gates | recurrence-runtime bytes match, semantic differences are zero, and metadata-only differences are allowlisted |

The test
`topology_replay_color_projection_rejects_a_missing_internal_tuple` aborts on
the untouched starting revision with the same `Some` assertion result.  It is
a pre-existing baseline defect and will be reported separately from candidate
regressions.

The broad semantic unit group additionally exposed three stale baseline
failures.  Each was rerun by exact node ID against the untouched starting
revision and reproduced identically: a compiled-manifest fixture omits
`interaction_evaluation_count`, a synthetic three-line fixture uses an
unsupported closure-proof algorithm, and the NLC shared-DAG fixture expects
348 rather than the observed 420 interaction evaluations.  The paired JUnit
evidence is retained under
`.artifacts/recurrence-generation-opt/validation/baseline-regressions/`.

The three full-Rust failures are likewise pre-existing.  Each exact node ID
was rerun against the clean starting revision under an offline guarded Cargo
target and produced the same source line and values: the replay-selector
fixture supplies one Lorentz component where the runtime requires four, the
cache-target fixture observes 8 rather than 1, and the high-footprint fixture
observes 1,024 rather than 64.  Commands, output, provenance, and paired
diagnostics are retained in
`.artifacts/recurrence-generation-opt/validation/baseline-rust-regressions/`.

The rejected `825af33` wheel failed 21 of 32 focused prepared-execution tests,
including eight nodes that passed on baseline, and masked eleven other
baseline outcomes with the premature zero-coupling error. The corrected
`cf990b7` wheel passed all eight candidate-only nodes. Its intermediate result
is 19 passed and the exact same 13 failing node IDs as the untouched baseline;
normalized errors and assertion values also match, allowing only absolute
checkout paths in missing-prepared-pack messages. The corrected run took
333.065 seconds and peaked at 2.109 GiB guarded physical footprint.

On the final `4e2b1e0` candidate, the focused Python unit failure is
`test_execution_manifest_carries_additive_replay_contract`; it raises the
same normalized missing-`interaction_evaluation_count` `ValueError` as the
untouched baseline.  The 13 integration failures likewise have exactly the
same node IDs and normalized messages on baseline and candidate.  The skipped
tests are the explicitly gated full process-set generations.

The final Rust construction panic is
`topology_replay_color_projection_rejects_a_missing_internal_tuple`: both
candidate and authenticated baseline return `Some` where the test expects
`None`, then panic-abort.  The authoritative candidate run therefore consists
of the other 55 construction tests, all 8 color tests, and all 27
numerical-evidence tests passing, with no candidate-only Rust failure.

The final installed wheel also reproduces the independently retained
equal/opposite/zero exact-session oracle byte for byte.  Its canonical report
is 28,949 B with SHA-256
`993ca27bbe4bfd6fcb5c1535913c3bb0cd611a897b87dfd8d710eb3c4edcdd5f`;
its transported evidence is 15,792 B with SHA-256
`c1297d261ef48f9315b11bc2a8cb5272f9a116dbb08474dc81d5f04f682b7c1c`;
and its decision and certificate-set SHA-256 values are respectively
`2b8861cdf536348dcbf2e89fc326f505e6f9ae26ff3fa71b96d7be5f7abf55f0`
and
`f4220277a42258565b9d0740cc5659b81e030763f37600d8480d51ed68f9c36c`.

### Numerical parity

The guarded validator compares selected totals and resolved
per-helicity/per-color components at one authenticated point.  A cell passes
when every comparison satisfies absolute difference at most `1e-15` or
relative difference at most `1e-12`.  Topology replay selects the first color
and resolves every helicity; all-flow union selects the first
non-structural-zero helicity and resolves every color.

| Process | Layout | Result | Resolved components | Maximum absolute / relative difference | Comparison SHA-256 |
|---|---|---|---:|---:|---|
| `d d~ > Z g g g g` | topology replay | passed | 192 | `0 / 0` | `1c2f91449b069b098d710cf2a983a52f51e22197a377950a95cb333b418d5a42` |
| `d d~ > Z g g g g` | all-flow union | passed | 24 | `0 / 0` | `d8af99c50e89dee9e05f6e8f06dcd5857d8249f28e07bdc7a40189fcf8bea1e3` |
| `d d~ > t t~ g g g` | topology replay | unavailable on both variants | — | — | — |
| `d d~ > t t~ g g g` | all-flow union | passed | 48 | `0 / 0` | `352a702084176e3df26aed0d520f02d94794a2a99b1c89ca7b178dee34511d29` |
| `g g > g g g g` | topology replay | unavailable on both variants | — | — | — |
| `g g > g g g g` | all-flow union | passed | 120 | `0 / 0` | `998693d148a421689290bfd48727e63097977944eb053323e22fa622939fc4e8` |

The two unavailable topology cells fail before comparison with
`compact recurrence reduction coverage is incomplete` on both the baseline
and candidate; the baseline `t t~ + 3g` diagnostic is retained separately.
They are unsupported baseline modes, not candidate regressions and not
successful parity cells.  The four supported cells compare 384 resolved
components in total with exactly zero pointwise and componentwise difference.

### Artifact comparison

The representative n=8 topology-replay and all-flow-union exact gates and the
diagnostic n=9 topology-replay exact gate all pass.  At each point, the
recurrence-runtime `.pacbin`, schedule index, and ordinary runtime-bearing
payloads are byte-for-byte identical.  The semantic censuses match in all nine
domains with no changed domain and no truncated difference report.

| n | Layout | Artifact comparator | Recurrence `.pacbin` SHA-256 | Allowlisted metadata differences | Unknown differences | Semantic census differences | Census SHA-256 |
|---:|---|---|---|---:|---:|---:|---|
| 8 | topology replay | passed | `050f3799f6b9d2a5ec4f3ecc61e00ea95cea5e606359153d6952ea138c19a8b6` | 19 | 0 | 0 | `36538ffd708284fac386ada126cea4b985978bec9641673a634f2fbd03886095` |
| 8 | all-flow union | passed | `0b32935a185c2cb64d48ac38ad8bb7a4fa8b5a5706e5bd1b022d57f1f9d02b8d` | 17 | 0 | 0 | `79bb35ba5b0887a75f6771c66a08cdf404a02fb2897f4f70bab4fe5260fb7624` |
| 9 | topology replay, diagnostic | passed | `46917cb6893612540ee35bfb22b4b67f80d073b2c9562278143033972bb22496` | 19 | 0 | 0 | `99ab09f4725418d2fb03fc2bbf737848f04b737ec9e602ca72ab1637e1c4f8e9` |

The comparator's raw whole-artifact `exact_payload_bytes_match` field is
false only because execution, proof, and manifest documents retain explicitly
allowlisted timing, provenance, and derived metadata.  Runtime schedule-plan
bytes, runtime-bearing payloads, execution semantics, manifest semantics, and
policy/projection bodies all match; there are no unclassified differences.
The four n=8 canonical JSON results are retained under
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-remaining-gates/exact/`;
the n=9 topology comparator and census are retained under
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-n9-gates/exact/`.

## Generation performance and RAM

The smallest pre-change diagnostic canary, not a statistical comparison, was:

| n | Layout | Generation wall | Worker peak RSS | Native total | Lowering |
|---:|---|---:|---:|---:|---:|
| 2 | topology replay | 4.719176917 s | 401,375,232 B | 0.132704750 s | 0.031334209 s |
| 2 | all-flow union | 4.459412833 s | 409,272,320 B | 0.088116625 s | 0.002725583 s |

The authoritative n=2 through n=8 generation-only campaigns use unique cold
roots, alternating baseline/candidate order, three repetitions per variant
and point, ten post-build validation samples, and the shared content-hashed
driver.  The accepted n=9 topology diagnostic uses the same controls with two
repetitions per variant and the explicit diagnostic-incomplete flag.  All 88
completed samples passed their authenticated generation-only acceptance:

| Campaign | Scheduled | Passed | Failed | Censored |
|---|---:|---:|---:|---:|
| `authoritative-d106c21-n2-n4-r3` | 36 | 36 | 0 | 0 |
| `authoritative-d106c21-n5-n7-r3` | 36 | 36 | 0 | 0 |
| `authoritative-d106c21-n8-topology-r3` | 6 | 6 | 0 | 0 |
| `authoritative-d106c21-n8-union-r3` | 6 | 6 | 0 | 0 |
| `final-4e2b1e0-n9-topology-r2-diagnostic` | 4 | 4 | 0 | 0 |

Every accepted sample exited zero, did not time out, completed its watchdog
record without exceeding the 30 GiB guard, and has
`generation_only_acceptance.passes=true`.  The `authoritative-d106c21`
campaign labels record the later measurement-driver revision; the measured
source identities throughout are the authenticated baseline `172e58f` and
final candidate `4e2b1e0` above.

The earlier partial campaign `final-4e2b1e0-n9-topology-r2` is excluded from
all accepted timing statistics.  It was stopped after one successful baseline
generation because the invocation omitted
`--allow-diagnostic-incomplete-success`; generation, all ten post-build
validation samples, and generation-only acceptance had passed, but the
custom diagnostic process correctly caused the harness to exit nonzero
without that flag.  The provenance note is retained at
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-n9-gates/topology-r2/DISCARDED_MISSING_DIAGNOSTIC_FLAG.md`.

The following table reports the median inner generation wall.  Speedup is
baseline median divided by candidate median.  Variation is the sample
coefficient of variation over the completed repetitions: three through n=8
and two for the n=9 diagnostic.  RSS is the exact median generation-worker
`resource.getrusage` high-water lower bound in bytes, and the last column is
candidate RSS divided by baseline RSS.

| n | Layout | Baseline wall (s) | Candidate wall (s) | Speedup | CV baseline / candidate (%) | Baseline RSS (B) | Candidate RSS (B) | RSS ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | topology replay | 4.343057459 | 4.302687917 | 1.009382 | 1.059 / 0.489 | 423,575,552 | 408,977,408 | 0.965536 |
| 2 | all-flow union | 4.286637834 | 4.214903583 | 1.017019 | 0.293 / 1.222 | 427,655,168 | 433,373,184 | 1.013371 |
| 3 | topology replay | 4.574153750 | 4.493042792 | 1.018053 | 1.653 / 0.930 | 440,991,744 | 412,745,728 | 0.935949 |
| 3 | all-flow union | 4.346780042 | 4.297372250 | 1.011497 | 0.684 / 2.621 | 416,890,880 | 424,542,208 | 1.018353 |
| 4 | topology replay | 4.919377917 | 4.859933333 | 1.012232 | 0.407 / 0.108 | 426,147,840 | 408,485,888 | 0.958554 |
| 4 | all-flow union | 4.490413667 | 4.436693000 | 1.012108 | 0.463 / 0.236 | 414,253,056 | 421,543,936 | 1.017600 |
| 5 | topology replay | 5.851771333 | 5.702981500 | 1.026090 | 0.835 / 0.904 | 409,387,008 | 429,768,704 | 1.049786 |
| 5 | all-flow union | 5.242066125 | 5.134775416 | 1.020895 | 0.110 / 0.509 | 427,720,704 | 414,810,112 | 0.969815 |
| 6 | topology replay | 8.319462667 | 8.018135041 | 1.037581 | 0.729 / 0.351 | 407,633,920 | 429,457,408 | 1.053537 |
| 6 | all-flow union | 10.532521083 | 10.037093833 | 1.049360 | 1.371 / 0.116 | 411,238,400 | 425,345,024 | 1.034303 |
| 7 | topology replay | 16.053845292 | 15.085848042 | 1.064166 | 0.205 / 0.629 | 458,555,392 | 456,540,160 | 0.995605 |
| 7 | all-flow union | 57.890170042 | 54.043320833 | 1.071181 | 0.079 / 0.159 | 565,428,224 | 559,415,296 | 0.989366 |
| 8 | topology replay | 64.431134833 | 61.209558791 | 1.052632 | 0.456 / 0.181 | 750,764,032 | 754,155,520 | 1.004517 |
| 8 | all-flow union | 880.582939667 | 776.212633875 | 1.134461 | 0.391 / 0.832 | 1,412,300,800 | 1,456,046,080 | 1.03097448 |
| 9 | topology replay, diagnostic | 730.772257459 | 713.526591917 | 1.024170 | 0.116937 / 0.276678 | 7,322,886,144 | 7,887,659,008 | 1.077124354 |

The candidate median is faster at every completed point.  The small-n
differences remain comparable to run variation; one n=2 topology pair and one
n=3 union pair favored baseline.  At the stable larger points, paired median
speedups are 1.04046x/1.04936x for n=6 topology/union,
1.06195x/1.07239x for n=7, 1.05400x for n=8 topology, and the authoritative
n=8 union median speedup is 1.13446097x.
The accepted n=9 topology diagnostic walls are
730.168005792 and 731.376509125 s for baseline and
714.922540250 and 712.130643583 s for candidate.  Their medians improve
730.772257459 → 713.526591917 s, a 1.024169619x speedup and 2.359923%
reduction.  Both candidate samples complete in less than 12 minutes, and all
four samples complete in less than one hour.  The maximum variant wall-time
CV is 0.276678%, so no third repetition was scheduled.

Generation-worker RSS has no monotone regression.  The n=8 topology median is
754,155,520 B versus 750,764,032 B, or +0.4517%, and the n=8 union median is
1,456,046,080 B versus 1,412,300,800 B, a 1.03097448 ratio.  At n=9 topology,
the candidate median is 7,887,659,008 B versus 7,322,886,144 B, a
1.077124354 ratio; the independent watchdog-guard median ratio is
1.06998552.  These RSS values are worker self high-water lower bounds, not
aggregate process-tree samples, and every guard remained below 30 GiB.

Median construction/native phases, in seconds, are:

| n | Layout | Recurrence construction, B → C (speedup) | Semantic construction, B → C (speedup) | Native total, B → C (speedup) | Direct lowering, B → C |
|---:|---|---:|---:|---:|---:|
| 2 | topology replay | 1.111859084 → 1.043146125 (1.065871x) | 0.001346625 → 0.001752000 (0.768622x) | 0.127461958 → 0.126526333 (1.007395x) | 0.030603667 → 0.031428834 |
| 2 | all-flow union | 1.013568209 → 0.966857125 (1.048312x) | 0.000819750 → 0.001343500 (0.610160x) | 0.082499958 → 0.082838875 (0.995909x) | 0.002754834 → 0.002711208 |
| 3 | topology replay | 1.272971792 → 1.194546417 (1.065653x) | 0.002597250 → 0.002761417 (0.940550x) | 0.189463334 → 0.182687917 (1.037087x) | 0.062333167 → 0.062052959 |
| 3 | all-flow union | 1.065984584 → 1.006555541 (1.059042x) | 0.001054500 → 0.001538291 (0.685501x) | 0.098946542 → 0.093354709 (1.059899x) | 0.005431000 → 0.005264083 |
| 4 | topology replay | 1.646791750 → 1.566952083 (1.050952x) | 0.007114584 → 0.006145708 (1.157651x) | 0.337789625 → 0.318298416 (1.061236x) | 0.122457583 → 0.122330125 |
| 4 | all-flow union | 1.196315375 → 1.116872625 (1.071130x) | 0.002270292 → 0.002494333 (0.910180x) | 0.136544833 → 0.125354250 (1.089272x) | 0.015955125 → 0.016092208 |
| 5 | topology replay | 2.581762125 → 2.421450916 (1.066205x) | 0.023423000 → 0.017550208 (1.334628x) | 0.663139167 → 0.609189083 (1.088560x) | 0.243870291 → 0.249232750 |
| 5 | all-flow union | 1.857049292 → 1.722134375 (1.078342x) | 0.009728000 → 0.008153125 (1.193162x) | 0.313325750 → 0.284825042 (1.100064x) | 0.062319500 → 0.063514291 |
| 6 | topology replay | 4.953063541 → 4.594467416 (1.078050x) | 0.086879459 → 0.055304208 (1.570938x) | 1.374788625 → 1.242945625 (1.106073x) | 0.490121916 → 0.491847125 |
| 6 | all-flow union | 6.109609417 → 5.614799542 (1.088126x) | 0.086400917 → 0.058396417 (1.479559x) | 1.335870250 → 1.148989542 (1.162648x) | 0.308721041 → 0.314712333 |
| 7 | topology replay | 10.848446125 → 9.873823375 (1.098708x) | 0.299670208 → 0.169080292 (1.772354x) | 3.068671250 → 2.663059500 (1.152310x) | 1.116643208 → 1.078669375 |
| 7 | all-flow union | 38.570899125 → 34.661646959 (1.112783x) | 1.758055708 → 1.041920833 (1.687322x) | 9.069760417 → 7.397401833 (1.226074x) | 1.972434791 → 1.965610000 |
| 8 | topology replay | 29.983955667 → 26.700454500 (1.122975x) | 1.180313625 → 0.492112083 (2.398465x) | 8.580502500 → 7.216681209 (1.188982x) | 3.942910000 → 3.768811333 |
| 8 | all-flow union | 445.989065250 → 350.124570750 (1.273801x) | 75.755311959 → 38.657107458 (1.959674x) | 134.012103375 → 89.854000042 (1.491443x) | 19.965319084 → 19.637320125 |
| 9 | topology replay, diagnostic | 164.561121501 → 146.936152958 (1.119950x) | 5.119580459 → 1.403858229 (3.646793x) | 55.639694980 → 47.959819021 (1.160131x) | 41.824271230 → 38.942432604 |

At n=9 topology, recurrence construction falls by 10.7103%, semantic
construction improves 3.64679x, direct lowering improves 1.0740x, and native
total improves 1.16013x.  The much larger fixed generation remainder dilutes
those phase improvements to the 2.359923% end-to-end reduction above, below
the plan's 20% end-to-end target.

At n=8 union, recurrence construction falls by 21.4948%, semantic
construction improves 1.95967x, native total improves 1.49144x, and direct
lowering improves 1.01670x.  End-to-end generation falls
880.582939667 → 776.212633875 s, an 11.8524% reduction.  This is a material
improvement at the scaling point, although it misses the plan's 20%
end-to-end target.

At n=8 topology, model loading is unchanged
(1.079874167 → 1.078859667 s) and recurrence construction accounts for
3.283501167 s of the 3.221576042 s generation-wall reduction.  The computed
top-level phase sum improves 31.066171626 → 27.783297791 s, while the
remaining generation work is effectively fixed at
33.312787960 → 33.446871041 s.  That fixed remainder dilutes the 2.398x
semantic-construction and 1.189x native improvements to 1.053x end to end.

Candidate artifacts add deterministic, generation-only
`recurrence_schedule_profiles`.  External-baseline artifacts predate this
schema and expose only the coarse candidate-pair count, so the profile pass
named `baseline` must not be confused with the external A/B baseline.  Within
every candidate artifact, its internal `baseline` and `final` passes have
identical operation counters and serialized bytes.

The high-n final-pass counters that best explain scaling are:

| n | Layout | Theoretical / visited parent pairs (visited %) | Forward-transition probes | Closure candidates | Indexed hash lookups | Current-key clones | Container bytes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | topology replay | 8,994 / 2,904 (32.288%) | 3,605,700 | 96 | 7,738 | 390 | 164,488 |
| 5 | all-flow union | 3,504 / 710 (20.263%) | 1,445,850 | 576 | 1,044 | 225 | 99,016 |
| 6 | topology replay | 30,537 / 8,202 (26.859%) | 11,264,540 | 192 | 30,704 | 854 | 446,792 |
| 6 | all-flow union | 40,901 / 4,369 (10.682%) | 5,039,650 | 14,400 | 6,704 | 1,171 | 419,016 |
| 7 | topology replay | 94,976 / 21,648 (22.793%) | 34,678,980 | 384 | 102,691 | 1,798 | 1,944,392 |
| 7 | all-flow union | 574,716 / 31,111 (5.413%) | 16,807,560 | 518,400 | 50,482 | 7,093 | 2,427,080 |
| 8 | topology replay | 277,711 / 54,426 (19.598%) | 105,798,140 | 768 | 304,523 | 3,702 | 17,760,072 |

For topology, `candidate-processing` follows visited pairs and indexed hashes,
while `structural-feasibility` follows the roughly 3x-per-step growth in
forward-transition probes.  For union, `closure-processing` is the n=7
semantic cliff: closure candidates grow 576 → 14,400 → 518,400 and that phase
grows 0.000846 → 0.020424 → 0.749744 s.  Direct lowering follows serialized
container growth in both layouts.

The remaining n=8 topology profile counters are also exact:
54,426 of 277,711 theoretical parent pairs are visited (a 5.1025x reduction);
768 of 90,624 theoretical closure candidates survive; support-bucket cache
hits/misses are 1,084/168; color-acceptance cache hits/misses are
27,053/138,109; transition-index hits/misses are 32,600/21,826; current-key
hits are 23,380 of 27,082 lookups; accepted-parent-key clones are zero;
current-key clones are 3,702; indexed and color-fragment hash lookups are
304,523 and 138,109.  Candidate n=8 recurrence time is itself dominated by
nested exact numerical work: numerical warm-up 12.365448750 s, final native
generation 7.604290917 s, internal baseline native generation 2.985778875 s,
and the two plan loads 1.152987333 s and 1.130936459 s.

Both n=8 layouts and the diagnostic n=9 topology point have accepted
performance and exact-artifact/semantic-census evidence.  The paired n=9
union scout reached the 1 GiB raw-evidence boundary in both variants and
therefore intentionally produced no runtime artifact: baseline failed closed
after 5,914.048 s and candidate after 3,243.342 s.  At the same deterministic
boundary this is a 45.16% wall-time reduction, or 1.824x speedup, with
watchdog peak physical footprints of 3.227 and 3.013 GiB respectively.  This
is diagnostic construction evidence, not a successful n=9 union generation.
The planned additional multi-hour campaigns and broad Python/runtime sweep
were explicitly deferred by the user's short-finalization decision.

## Runtime no-regression gate

Representative n=8 and diagnostic n=9 topology recurrence-runtime byte
identity is recorded above.  The broader n=6/n=7 statistical release gate was
not run after the user explicitly moved it to the later full
release-readiness pass.  No generation-only timing result is presented as
runtime acceptance.

## User-directed validation and artifact simplification

The final integration also changes the default post-build validation sample
count from ten to two.  This validation evaluates the normal binary64 runtime
total and resolved selectors; it is separate from numerical-relation
discovery, whose 96-digit four candidate plus four independent verification
probes are unchanged.  Authoritative benchmark/reference tools that
deliberately request ten samples continue to do so explicitly.

Normal Python and Rust artifact loading now treats local artifacts as trusted:
it parses the schema, validates references and confined paths, and checks
runtime ABI/target compatibility, but does not recompute the top-level
manifest identity or eagerly hash every payload and directory entry.  Python
callers that need a complete integrity audit can still call
`validate_payloads(manifest)` explicitly.  The `artifact_id` is a compact
content label over runtime-bearing payload declarations, so validation
momenta, requested/effective configuration snapshots, timing, and provenance
do not perturb it.  This deliberately drops backward identity-policy
compatibility at the user's request.  The current identity contract is an
explicit required manifest extension: absence, an older interpretation, or an
older SymJIT plane ABI fails closed with regeneration guidance.  No
generated-artifact compatibility shim is retained.

## Original AmpliCol comparison

The pinned revision `79c96cecf2a722e50c3d2030b6894d755f96518a` built without
patching, network access, or escalation.  It remained clean and was benchmarked
read-only under the same 30 GiB guard.  Exact `d d~ > Z + (n-1)*g` setup grew
from 18.437698 s at n=8 to 594.051455 s at n=9, a 32.22x jump.  The n=9 probe
contained 308,644 currents, 1,324,649 vertices, and 40,320 color orders; its
setup peak was 8.770 GiB RSS and 24.146 GiB Darwin physical footprint.

Transferable and model-specific findings, source locations, commands, input
digests, limitations, and the full n=2 through n=9 table are in
`RECURRENCE_GENERATION_OPTIMIZATION_LEGACY_COMPARISON.md`.

## Reproducible commands

The exact accepted final-candidate command contracts and logs are retained in
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-provenance/RESULT.md`,
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-python/RESULT.md`,
and
`.artifacts/recurrence-generation-opt/validation/final-4e2b1e0-rust/RESULT.md`.
The commands below are the earlier candidate build/staging recipe retained
for historical reproducibility; they do not identify the final
`candidate-final` wheel by themselves.

### Historical candidate wheel

```bash
mkdir -p \
  "$PWD/.artifacts/recurrence-generation-opt/candidate/build-tmp" \
  "$PWD/.artifacts/recurrence-generation-opt/candidate/wheels"

TMPDIR="$PWD/.artifacts/recurrence-generation-opt/candidate/build-tmp" \
PYAMPLICOL_BUILD_MODE=candidate \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  .venv/bin/python -m build --wheel --no-isolation \
  --outdir "$PWD/.artifacts/recurrence-generation-opt/candidate/wheels"
```

### Source-runtime staging and isolated candidate install

```bash
TMPDIR="$PWD/.artifacts/recurrence-generation-opt/candidate/build-tmp" \
PYAMPLICOL_BUILD_MODE=candidate \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  .venv/bin/python tools/developer/prepare_source_runtime.py \
  --candidate \
  --wheel-directory \
  "$PWD/.artifacts/recurrence-generation-opt/candidate/wheels"
```

The additional release-readiness command appendices are intentionally
omitted.  The user will run one consolidated release-readiness validation pass
after this work and the SymJIT 2.22 arena migration are integrated.
