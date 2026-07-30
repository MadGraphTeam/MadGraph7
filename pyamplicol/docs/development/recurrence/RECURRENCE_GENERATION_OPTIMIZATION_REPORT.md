# Recurrence-generation optimization report

Status: implementation and guarded validation in progress.

This report implements the plan in
`RECURRENCE_GENERATION_OPTIMIZATION_PLAN.md`.  It covers only recurrence
generation.  Public Python APIs, CLI commands and options, native ABI,
recurrence plane schemas, parameter and selector semantics, runtime
evaluation, and runtime-bearing recurrence payloads remain unchanged.

## Provenance and safety

The authenticated baseline is Git revision
`172e58fd33a3c65563866c50cfbb5e1ddcd7b302`, initially on a clean `main`
worktree.  Its native build-input digest is
`865408b317aa421fdc19f519ca8d8317d49cfbbe75a0f6135c1afe0b06c74bfe`;
the isolated baseline extension SHA-256 is
`8137fb99a8682a5a3a5c558f8dc8f02197ceec03651d9a0d47732079c0e9245d`.
The explicit prepared-model SHA-256 is
`772ead40a64bc01f37725ad3173a6126765b3f2be70396acbac64bbfa100cf86`.
Complete baseline provenance is retained below
`.artifacts/recurrence-generation-opt/baseline/provenance/`.

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

## Rejected or deferred implementation

The implementation does not add or reintroduce `DirectApplication`,
`DirectTable`, scalar-plane lowering, plane or row bindings, broadcast or
scratch ownership, recurrence epilogues, or runtime row scheduling.  These are
owned by the stable
`pyamplicol-symjit-plane-application-v1` /
`pyamplicol-recurrence-plane-binding-v2` migration boundary.  The verbatim
plan records the binding-v1 name that was current when it was approved; the
migration subsequently advanced that schema to v2 without changing this
optimization round's ownership boundary.

Retained native sessions, zero-relation finalization, batched
arbitrary-precision probes, shared structural DAGs with flow overlays,
persistent color arenas, streamed construction, deterministic parallel
enumeration, new structural proofs, and persisted row/liveness changes were
not implemented.  Their impact, RAM risk, implementation depth, ABI
consequences, and migration dependencies are documented in
`RECURRENCE_GENERATION_OPTIMIZATION_SCOUTING.md`.

## Correctness and exact-artifact validation

Guarded final results will be recorded here after the frozen candidate wheel is
built.

### Test suites

| Suite | Result | Peak RSS / footprint | Notes |
|---|---|---:|---|
| Rust focused construction tests | 12 passed | TODO | exact pair order, wide support, demand, color, metadata |
| Python unit/integration recurrence suites | TODO | TODO | |
| Semantic census | TODO | TODO | |
| Numerical parity | TODO | TODO | |

The test
`topology_replay_color_projection_rejects_a_missing_internal_tuple` aborts on
the untouched starting revision with the same `Some` assertion result.  It is
a pre-existing baseline defect and will be reported separately from candidate
regressions.

### Artifact comparison

TODO: record exact payload/kernel byte identities and the explicit
timing/provenance-only allowlist.

## Generation performance and RAM

The smallest pre-change diagnostic canary, not a statistical comparison, was:

| n | Layout | Generation wall | Worker peak RSS | Native total | Lowering |
|---:|---|---:|---:|---:|---:|
| 2 | topology replay | 4.719176917 s | 401,375,232 B | 0.132704750 s | 0.031334209 s |
| 2 | all-flow union | 4.459412833 s | 409,272,320 B | 0.088116625 s | 0.002725583 s |

TODO: insert cold-root, alternating-order n=2 through n=9 A/B tables, medians,
variation, phase counters, speedups, and peak RSS.

## Runtime no-regression gate

TODO: record byte-identity proof and load/evaluate canaries, or the complete
n=6/n=7, both-layout, batch 1/128/1024 statistical gate where byte identity is
not sufficient.

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

### Candidate wheel

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

TODO: append the exact isolated install, validation, A/B campaign, semantic
census, artifact comparison, numerical, and runtime commands after execution.
