# Recurrence-generation redesign scouting

These designs could reduce generation time beyond construction-private
optimizations, but they are deliberately not implemented in this round.  Each
changes lifetime, evidence, persistence, parallelism or the migration-owned
plane/runtime boundary and therefore requires separate review.

## Retained native generation session

Retain authenticated decoded process/template catalogs, semantic construction
and optionally the baseline direct plan across the baseline, certification and
final lowering calls.

- Expected impact: high for the default certified-reuse path, which currently
  repeats native extraction, semantic construction and baseline lowering.
- RAM: medium to high retained memory proportional to the semantic program and
  direct plan; needs an explicit release point and peak-RSS measurements.
- Runtime risk: none if retained state is generation-only and immutable.
- Depth: medium; requires a native handle/session lifetime, fail-closed digest
  authentication and cleanup across Python exceptions.
- ABI/migration: a new private Python/native lifetime contract.  Report-only by
  user decision for this round.

## Certified zero-relation finalization

When authenticated evidence certifies no reusable relations, reuse the
already-built baseline plan or finalize it without constructing an equivalent
second plan.

- Expected impact: high on the measured gluon ladders, where certification
  found zero relations.
- RAM: low if the baseline plan is moved; medium if both plans coexist.
- Runtime risk: none if emitted bytes and evidence attachment are exactly
  equivalent.
- Depth: medium; requires an authenticated proof that the optimized relation
  transform is the identity and a serialization path that preserves all
  metadata.
- ABI/migration: touches native evidence/finalization ownership and must be
  reviewed with the migration task.

## Batched native arbitrary-precision probes

Move candidate and verification probe evaluation into one authenticated native
batch, retaining the same point domains, precision, tolerances, ordering and
evidence digests.

- Expected impact: high if Python/Symbolica calls and Decimal object creation
  dominate warm-up.
- RAM: medium; bounded point/current tiles are required.
- Runtime risk: none in principle, but evidence parity is critical.
- Depth: high; needs an independently validated arbitrary-precision executor
  and byte-identical evidence encoding.
- ABI/migration: new generation-only native API and potentially new Symbolica
  ownership.

## Shared structural DAG with flow overlays

Construct support/state/helicity structure once and attach exact color-flow
overlays rather than rebuilding full topology-replay lanes.

- Expected impact: potentially very high for topology replay as physical flow
  count grows.
- RAM: potentially lower if overlays are sparse, but a poorly chosen shared
  representation could retain a much larger global graph.
- Runtime risk: must lower to the same specialized runtime rows and must not
  introduce runtime indirection.
- Depth: high; changes current identity, proof ownership, liveness and
  canonical emission.
- ABI/migration: may alter semantic-program and artifact structure; review
  required before implementation.

## Persistent or chunked color arena

Represent color forests as interned fragments/ropes with structural sharing,
or construct color sectors in bounded chunks and discard rejected fragments.

- Expected impact: medium to high for all-flow union.
- RAM: potentially much lower; persistent-node overhead and hash-table
  retention require measurement.
- Runtime risk: none if flattened canonical colors are produced before
  lowering.
- Depth: medium to high, depending on whether the representation is transient
  or persisted.
- ABI/migration: transient arenas can be ABI-neutral; persisted fragment IDs
  require artifact review.

## Lazy or streamed semantic construction

Stream accepted contributions into stage-local aggregation and release
construction metadata immediately after closure/liveness proves it dead.

- Expected impact: medium generation improvement and potentially large RSS
  reduction.
- RAM: lower, with bounded stage or lane residency.
- Runtime risk: none if canonical final rows are reconstructed identically.
- Depth: high because reflection reconciliation, exact cancellation, closure
  proofs and deterministic IDs currently depend on complete stage state.
- ABI/migration: ABI-neutral only if the same canonical program is emitted.

## Deterministic parallel enumeration

Partition support pairs or color sectors, build independent ordered fragments,
and merge them by the existing canonical `(stage, parent IDs, transition
ordinal, witness ordinal)` order.

- Expected impact: medium to high on multi-core hosts after serial indexing
  bottlenecks are removed.
- RAM: higher because worker-local aggregation and merge buffers coexist.
- Runtime risk: none if only generation is parallel.
- Depth: high; exact factors, first-seen IDs, reflection certificates and
  progress/failure ordering must remain deterministic.
- ABI/migration: no ABI change is necessary, but deterministic artifact-byte
  proof is mandatory.

## Structural relation proofs and exact prefilters

Use model-certified linearity, symmetry and coupling/color identities to prove
candidate current relations before numerical probing, with numerical evidence
retained as an independent verification layer where required.

- Expected impact: medium to high by shrinking certification candidate sets.
- RAM: low to medium for structural signatures.
- Runtime risk: none if proofs only remove impossible relation candidates.
- Depth: high; every proof rule needs a versioned authenticated certificate and
  adversarial validation.
- ABI/migration: new evidence/proof semantics and therefore review required.

## Persisted homogeneous rows and liveness scratch

Persist rows grouped by executor/shape and allocate scratch from proven
last-use lifetimes, inspired by legacy AmpliCol's data-oriented generated
loops.

- Expected impact: may improve lowering, load time and runtime locality, but it
  is not guaranteed to improve generation alone.
- RAM: lower runtime scratch is possible.
- Runtime risk: direct and substantial; no runtime slowdown is acceptable.
- Depth: very high.
- ABI/migration: owned by the SymJIT 2.22 plane-application/binding migration.
  It must not reintroduce `DirectApplication`, `DirectTable`, scalar-plane
  lowering or recurrence-generated epilogues.

## Review order

The recommended review order is retained native session plus certified
zero-relation finalization, then shared structural overlays and a persistent
color arena.  Deterministic parallelism should be evaluated only after serial
candidate enumeration and color acceptance are no longer dominant.  Runtime
row/scratch changes remain last because they have the highest compatibility
and no-regression burden.
