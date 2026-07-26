# Compiled-DAG O3 non-union DirectTable microkernels

## Objective and immutable inputs

This feature targets the compiled JIT O3 non-union execution lane. It must
improve the complete-artifact, runtime-selected `u u~ > Z+6g` workload by at
least 10% at both batch 128 and batch 1024 without turning the compiled DAG
into recurrence execution.

The implementation starts from pyAmpliCol
`f4606fa9be52355b4a66efcfa2b7072d489205eb`. Its dependency contract is:

- SymJIT repository:
  `https://github.com/ValentinHirschi/symjit_changes_for_pyamplicol.git`;
- branch: `pyamplicol-generic-direct-apis`;
- revision: `89efdb806e7fcd9ac68a9d38f3f2880adf1987d2`;
- archive SHA-256:
  `070ff7fc04d5cdc5ab769d7a47b3da04cbc2b97d87136d303180c95b9eb380cd`;
- source-tree SHA-256:
  `e42d648d995c61881e560aefc50f80a995e86fb24a67ed9b0f0b5a80d6773fcf`;
- configured candidate-tree SHA-256:
  `820675246517cd49198495936327768da7a7a1d25f8bf20749c21aad1c2f56da`;
- DirectApplication storage ABI: `symjit-direct-application-storage-v1`;
- DirectTable binding ABI: `symjit-direct-table-binding-v1`;
- DirectTable descriptor ABI: `symjit-direct-table-descriptor-v1`.

The contributor build applies no local SymJIT patches. This feature may use
the pinned DirectApplication and DirectTable APIs but must not add a SymJIT
patch. If those APIs cannot represent the required kernel without changing
the fork, the feature stops rather than expanding the dependency delta.

The historical same-host complete-artifact selected-flow reference is
`54.024 +/- 0.226 us/point` at batch 128 and
`56.880 +/- 0.361 us/point` at batch 1024. Its result SHA-256 is
`6ca4cfa0038d7e2d8dab6b21e0106fadd2572234fef1fbd760eb2476fad242ab`.
These values predate the exact fork-pinned source revision and are diagnostic
only. Before candidate performance is judged, a fresh immutable baseline
artifact and runtime identity must be generated from the source revision
above and recorded in this document.

### Fresh `f4606fa` baseline

The first exact-source baseline capture was generated on 2026-07-26 from the
clean source revision above. Its authenticated identities are:

- candidate native build-input SHA-256:
  `e84ab5ea52d1f523c5338e0adb7178fdb4a8cb8dbd41343fe701809ea2a84873`;
- native extension SHA-256:
  `f71d40debcb612f34da2e038a4b251a66ef010a27026f90946dee59f37d2ae7b`;
- artifact ID:
  `bce2e798113ebe0ff2677c96b0abcfffc65bd58bec065d135747cbc7862a1d8b`;
- artifact semantic-identity SHA-256:
  `403cec4e225634be1660d61a821d4ca0052181ae193ad2841b6ae37ffb836caf`;
- artifact tree SHA-256:
  `38ab1a323e22393e263b5ce01937d3e0614e8e0021fbc2931445f977dc503f0a`.

Generation took `10.979621 s`, reported a `399,654,912` byte process high-water
mark, and produced a `23,608,791` byte artifact, including a `7,801,720` byte
evaluator-state payload. The executable materialized schedule contains seven
current stages, 55 non-source current destinations, and 203 active
interactions. Its destination dimensions are 26 two-component, 15
four-component, and 14 six-component currents. The retained
1,425-current/8,338-interaction helicity recurrence structure is proof-only
metadata and is forbidden as the profitability denominator.

The first seven-sample five-second timing pass measured:

| Batch | Median us/point | MAD us/point |
|---:|---:|---:|
| 1 | 96.121 | 0.874 |
| 127 | 42.774 | 0.112 |
| 128 | 42.993 | 1.245 |
| 129 | 44.719 | 1.723 |
| 1023 | 42.575 | 0.811 |
| 1024 | 42.376 | 0.911 |
| 1025 | 42.809 | 0.309 |

An unrelated native link overlapped early samples in this pass. The identities,
artifact metrics, numerical validation, selector contract, and active-schedule
census are authoritative; these timing values are provisional diagnostic
anchors. Landing acceptance uses a new quiet, alternating baseline/candidate
capture and does not reuse these medians.

### Mandatory pre-code census outcome

The exact materialized `f4606fa` `u u~ > Z+6g` schedule was censused on
2026-07-26 before integrating the candidate. An initial audit incorrectly
treated one fully fused current as the source-motif unit. A second independent
audit corrected that interpretation: the approved design tableizes canonical
interaction contributions while retaining all-or-nothing ownership of each
selected current. The first contribution overwrites, later contributions
accumulate in original order, and the ordinary propagator/finalizer runs last.
No selected destination is split between table and residual execution.

The user subsequently authorized the profitability denominator to follow the
actual repeated source unit: each active occurrence of a prepared-kernel
identity that repeats in the 203-interaction materialized schedule. This
replaces the empty `evaluation_group_id` identity denominator. Those IDs
identify concrete execution rows and all have fanout one; they do not identify
shared generated source.

The frozen global prepared-kernel occurrence distribution is:

| Prepared kernel | Active occurrences |
|---:|---:|
| 4 | 35 |
| 7 | 49 |
| 24 | 49 |
| 36 | 40 |
| 41 | 30 |
| **Total** | **203** |

All 26 two-component destinations are wholly eligible. Their 98 contributions
collapse to two canonical vector--Weyl vertex kernels, used 49 times per
chirality:

| Prepared kernel | Chirality | Calls | Complex inputs | Complex outputs | Source bytes |
|---:|---:|---:|---:|---:|---:|
| 7 | +1 | 49 | 6 | 2 | 1,634 |
| 24 | -1 | 49 | 6 | 2 | 1,632 |

The contribution-count distribution is four destinations at each width one
through six and two destinations at width seven. It yields 26 overwrite and
72 ordered-accumulate attachments, followed by four propagator/identity
finalizer classes. The corrected slice is therefore structurally compact and
fits every hard arity, output, kernel-count, and source-size bound.

The initial implementation also includes the smallest structural extension:
five singleton four-component gluon currents, current IDs
`15,17,19,21,23`. They all use prepared three-vector kernel 4 and vector
propagator 37. Eligibility is structural: each current is homogeneous, owns
exactly one contribution, has 16 complex inputs and four complete complex
outputs, and has no residual contribution. Kernel 4 has canonical signature
`120acbff47e08ce698a108c3c8b0758555d5292f24c6b05c4504a977714ebd8c`
and 16,556 expression bytes. Propagator 37 has canonical signature
`b04f53f0c0046718e3090d041bea2f7d31dd363f02d1f7fc614ce620a7e7cacd`.

This authorized slice covers `103/203 = 50.739%` of the global prepared-kernel
occurrences and therefore passes the 50% gate. It uses three vertex identities,
at most seven known vertex/finalizer identities, at most 16 inputs, and no more
than two outputs for vector--Weyl kernels or four outputs for the singleton
three-vector family. Its prepared vertex source compression is
`19,822/242,814 = 8.164%`, and its estimated additional semantic rows are
1,440 bytes. Four-output kernels outside this exact structural family remain
ineligible.

The captured census JSON is
`COMPILED_DAG_O3_NON_UNION_MICROKERNEL_CENSUS.json`, with SHA-256
`aff953353500d6bff3a15e9dc6f44449ca2ce227439ffcae985b4db4bf5f0b88`.
It was produced from the clean generation worktree by constructing the
existing unit-test `u u~ > Z g g g g g g` evaluator process and calling
the materialized-stage and prepared-kernel catalog builders directly; no
pyAmpliCol native rebuild participated in the structural census.

| Census quantity | Required | Observed |
|---|---:|---:|
| Global repeated prepared-kernel occurrences | denominator | 203 |
| Wholly owned vector--Weyl destinations | exactly 26 | 26 |
| Structurally homogeneous singleton three-vector destinations | exactly 5 | 5 |
| Eligible occurrences | at least 50% | 103 / 203 (50.739%) |
| Canonical vertex kernel identities | at most 8 | 3 |
| Maximum complex inputs | at most 16 | 16 |
| Vector--Weyl outputs | exactly 2 | 2 |
| Singleton three-vector outputs | exactly 4 | 4 |
| Prepared vertex source ratio | at most 25% | 19,822 / 242,814 (8.164%) |
| Projected semantic rows | at most 4 MiB | below the cap |

A standalone Rust probe compiled four patchless SymJIT O3 source classes
(propagated/unpropagated crossed with chirality) using only DirectTable
binding-v1 and descriptor-v1. Their scalar plus SIMD DirectTable code was
11,584 bytes, versus 92,536 bytes for 26 isolated complete-current O3
applications, a projected 87.482% reduction. Descriptor payload was 1,104
bytes and invocation plus attachment rows were estimated at 10,528 bytes.
The emitted table kernels contained no branch-and-link calls, gathers, or
scatters. For this upper-bound probe the linear massless finalizer was
distributed into each contribution source; that is code-size evidence only,
not authorization to change the approved post-accumulation finalizer order.

The probe covers the dominant 98-call vector--Weyl core. No exact
three-vector machine-code number is inferred from source or descriptor bytes.
The complete candidate must still provide exact selected loaded machine-code
evidence before landing.

The pinned SymJIT API creates an observability constraint.
`DirectTableCallable` exposes exact scalar and SIMD code shapes, but
`DirectApplication` privately lowers its source to a distinct `Application`
and neither it nor the resulting `DirectApplet` exposes the lowered
`MachineCode::size`. The public source-application machine-code size is not
the executed lowered DirectApplication body. The independent auditor therefore
accepts only an exact
`executed-selected-machine-code-scalar-plus-simd-v1` metric for the 25% landing
gate. An unavailable exact metric fails that gate. DirectTable-only bytes and
portable source-Application bytes remain diagnostic and are never treated as
machine code.

Implementation proceeds with the authorized 103-occurrence slice. No further
authorization pause is required while a suitable in-scope improvement remains.
This does not relax landing: the candidate is deleted or withheld unless it
passes every numerical, performance, resource, exact code-size, cutover, and
regression gate below.

## Design

The compiled schedule remains the owner of topology, helicity, selector,
dependency, and arena policy. Repetition within complete eligible currents is
represented as DirectTable microkernel islands:

1. Generate one small O3 DirectApplication source for each exact canonical
   motif signature.
2. Bind current, momentum, parameter, and factor planes through DirectTable
   invocation rows.
3. Bind complete complex current destinations through ordered attachment rows
   using complex scale and explicit overwrite or accumulate semantics.
4. Run the ordinary compiled finalizer or propagator only after all
   contributions to that current.
5. Keep all ineligible currents as compressed O3 DirectApplication residual
   leaves.

One destination may not be split between table and residual execution.
Contributions to a destination retain their original evaluation-group order.
Independent destinations may be grouped only when the dependency certificate
proves that the grouping is order-independent.

The initial slice contains complete two-component vector--Weyl current
families plus structurally homogeneous singleton four-component three-vector
currents. Eligibility is structural, never process-name based. A motif key
includes source digest and ABI, canonical input order, input permutation,
result particle/chirality/width, mutable-parameter and coupling provenance,
selector-domain signature, finalizer identity, and optimization level.

The slice is bounded to eight kernel identities, at most 16 complex inputs,
64 KiB source payload per kernel, and 4 MiB semantic row data for the artifact.
Vector--Weyl kernels must have exactly two complex outputs. Four complex
outputs are allowed only for the authorized three-vector family, and every
destination in that family must have exactly one invocation, one overwrite
attachment, and no accumulation or residual contribution.

The pre-code census denominator is every active materialized occurrence of a
prepared-kernel identity that repeats globally. Eligible table invocations
must match exactly the `49 + 49 + 5` authorized occurrence distribution and
cover at least 50% of all 203 occurrences. Projected generated text must be at
most 25% of the text it replaces.

Implementation and measurement continue without another authorization pause
while there is a suitable improvement inside these bounds. Landing still
requires the final candidate to improve both primary batches by at least 10%.
No whole-schedule superkernel, generic MIR outlining, additional fusion, or
recurrence-style execution is in scope.

## Artifact cutover and diagnostics

Every newly generated compiled artifact uses compiled-stage-plan v2. It
contains residual DirectApplication leaves and zero or more DirectTable
islands, with source/descriptor digests and ABIs, canonical motif identities,
selector partitions, invocation and attachment rows, factor catalogs, plane
bindings, and dependency/order certificates.

The loader rejects compiled-stage-plan v1 and incompatible direct ABIs with an
actionable regenerate-artifact error. There is no v1 reader, converter,
dual-run production path, hidden fallback, or backward-compatibility toggle.
A v2 stage with no profitable islands is valid and executes its residual
leaves; that is the ordinary current representation, not legacy loading.
Malformed declared islands fail closed instead of silently becoming
residuals.

Inspection records expose island, kernel, invocation, and attachment counts;
table and residual machine-code bytes; semantic-row bytes; arena bytes; and
warmed allocation counts. Public Python APIs, CLI modes, selectors, and
configuration remain unchanged. Eager retains its existing O2 DirectTable
plan and is a non-regression lane, not a source of the claimed gain.

## Acceptance contract

The primary artifact is complete topology-replay `u u~ > Z+6g`, with physical
flow `flow:2,4,5,6,7,8,9,1` selected at runtime and a helicity sum. Baseline
and candidate use the same source, model, points, selector, runtime, and host.
Measurements cover batches 1, 128, and 1024 plus odd tails 127/129 and
1023/1025. At least seven alternating subprocess samples run for five seconds
each.

Landing requires:

- at least 10% lower median wall time at both batch 128 and batch 1024, with
  the paired change larger than three MAD;
- batch-1 regression no greater than 5%;
- exact executed selected scalar-plus-SIMD loaded machine code at least 25%
  smaller; unavailable exact evidence fails closed, and neither source bytes
  nor DirectTable-only bytes may substitute;
- no warmed execution allocations;
- generation time, artifact bytes, load time, and peak RSS no more than 10%
  worse;
- pointwise totals and resolved contributions within `rtol=1e-12` and
  `atol=1e-15`, with `evaluate()` agreeing with
  `evaluate_resolved().total()`.

Regression gates cover the `Z+6g` LC union workload; NLC and full-colour
`g g > t t~ +3g`; residual-only `d d~ > Z+3g`; built-in and UFO-SM models;
one existing fixture with four independent quark lines in compiled LC, NLC,
and full colour; mutable parameters; structural zeros; global and per-point
selectors; and malformed plans. Non-target execution and eager may regress by
no more than 2% or three MAD.

Native AArch64 validation runs under the 30 GiB watchdog. x86-64 compile and
test evidence comes from the existing CI path. No command requests elevated
permissions: blocked operations are replaced with workspace-local/offline
routes or reported as external blockers.

## Work ownership

The parent lane owns exact-source synchronization, integration, immutable
baseline and candidate builds, benchmarks, and the final main-token handoff.
Independent subagents own:

- generation, motif classification, and census;
- Rust DirectTable loading, validation, and stage-plan v2;
- adversarial correctness, cross-platform review, and performance audit.

Each lane uses isolated ownership and sends commits to the parent for review.
Only fully gated milestones may be merged. The implementation goal is complete
only after the accepted candidate is pushed to `main` and its landing SHA,
build/runtime identity, numerical evidence, and performance record are handed
off.
