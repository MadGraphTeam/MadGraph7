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

The initial slice is limited to complete two-component vector-Weyl current
families. Eligibility is structural, never process-name based. A motif key
includes source digest and ABI, canonical input order, input permutation,
result particle/chirality/width, mutable-parameter and coupling provenance,
selector-domain signature, finalizer identity, and optimization level.

The first slice is bounded to eight kernel identities, at most 16 complex
inputs and two complex outputs per kernel, 64 KiB source payload per kernel,
and 4 MiB semantic row data for the artifact. It proceeds only if a pre-code
census finds that eligible islands cover at least 50% of active repeated
evaluation groups and projected generated text is at most 25% of the text
they replace.

If both primary batches improve by at least 10%, expansion stops. If both
improve by at least 5% but either is below 10%, the same representation may be
extended to eligible four-component three-vector currents. If either batch is
below 5% after the first slice, or either remains below 10% after that single
extension, the candidate is removed and not landed. No whole-schedule
superkernel, generic MIR outlining, additional fusion, or recurrence-style
execution is in scope.

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
- selected loaded machine code at least 25% smaller;
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
