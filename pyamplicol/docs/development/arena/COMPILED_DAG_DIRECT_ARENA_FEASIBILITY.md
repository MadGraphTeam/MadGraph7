# Compiled-DAG Direct-Arena Feasibility

## Scope

This is a read-only architecture and performance assessment of whether ideas
from the recurrence Direct-Arena design could reduce non-evaluator overhead in
the existing compiled-DAG Rusticol lane without changing its semantics or
giving up process-specific evaluator fusion.

No implementation, generation, cache mutation, commit, or push was performed
as part of the investigation.

## Verdict

Direct-Arena principles can materially reduce compiled-DAG overhead, but the
appropriate changes are narrower than the recurrence runtime:

- Existing fused stage evaluators should remain intact.
- Redundant gathering, output scattering, native input construction, and
  result copying can be reduced without changing the evaluator ABI.
- Eliminating all input packing and output assignment requires a new
  arena/strided-plane ABI in SymJIT and the C++/ASM backends.
- Replacing fused stage evaluators with per-current or per-interaction
  prepared kernels would turn compiled mode into eager/recurrence execution
  and is not appropriate.

Likely wall-time gains are workload dependent:

- regular high-multiplicity LC: approximately 5-15%;
- data-movement-heavy LC: approximately 20-40%;
- full color: generally below 10-15% from arena plumbing alone because color
  contraction and reduction dominate the largest wall/evaluator gaps.

## Existing Compiled Execution Path

The materialized compiled lane currently performs:

1. Source, momentum, and parameter filling into a reusable global row-major
   current-state arena:
   `rust/crates/rusticol-core/src/engine/evaluation.rs:373`.
2. A stage-level gather from global current state into a dense stage input:
   `rust/crates/rusticol-core/src/evaluator/stage.rs:80`.
3. For chunked stages, another mapping from that stage input into each
   chunk's pruned dense input:
   `rust/crates/rusticol-core/src/evaluator/backend.rs:90`.
4. A fused evaluator call into contiguous output scratch:
   `rust/crates/rusticol-core/src/evaluator/stage.rs:114`.
5. A copy/scatter from output scratch back to global current state:
   `rust/crates/rusticol-core/src/evaluator/stage.rs:128`.
6. Amplitude-root input packing, evaluation, optional output remapping, and
   Rust reduction:
   `rust/crates/rusticol-core/src/evaluator/amplitude.rs:159`.
7. A clone of the reusable final-value buffer for the returning API:
   `rust/crates/rusticol-core/src/engine/evaluation.rs:490`.

The native API also reconstructs a nested `Vec<Vec<[f64; 4]>>` from flat
momenta for each call:
`rust/crates/rusticol-core/src/engine/native_runtime.rs:1387`.
The C API then copies the returned vector into caller storage:
`rust/crates/rusticol-capi/src/lib.rs:890`.

These are real hot-path costs in repeated Monte Carlo calls. The native
benchmark path includes them:
`rust/crates/rusticol-core/src/engine/native_runtime.rs:848`.

## Timing Evidence

For compiled JIT O3 `u u~ > Z+6g`, all flows and one selected helicity, batch
64:

| Component | Time per point |
|---|---:|
| Wall | 1.704 ms |
| Evaluator calls | 0.639 ms |
| Stage input packing | 0.641 ms |
| Output assignment | 0.127 ms |
| Amplitude packing/evaluation | 0.007 ms |
| Reduction | 0.043 ms |
| Other Rusticol core | 0.253 ms |

The preserved artifact has nearly equal total stage-union and leaf-chunk input
widths: approximately 1,396 versus 1,388 complex values per point. This
confirms that chunked stages pack almost the same data twice.

Other established profiles bound the benefit:

- `d d~ > t t~+3g` full color: about 2.0-2.1 ms wall, 1.43 ms evaluator,
  0.13 ms stage pack, 0.078 ms output assignment, and 0.256 ms reduction.
- Pure-gluon n=5 NLC/full: evaluator time remains near 6 ms while wall grows
  substantially; output assignment and especially sparse color reduction are
  material.
- In full-color `g g > 5g`, wall grows far more than evaluator time relative
  to NLC. The contraction loop visits coherent groups and sparse entries per
  point:
  `rust/crates/rusticol-core/src/evaluator/amplitude.rs:299`.

## 1. Low-Risk Opportunities

These preserve evaluator fusion and require no new SymJIT/backend ABI.

### One-pass leaf input gathering

Precompose the stage-global component mapping with each leaf chunk mapping at
load time, then gather directly from global state into the leaf evaluator
buffer. The selected-chunk implementation already demonstrates composed
mapping:
`rust/crates/rusticol-core/src/evaluator/backend.rs:184`.

For the measured Z+6g artifact, this could remove roughly 0.30-0.33 ms/point.

### Persistent evaluator output slabs

Keep each stage/chunk output buffer as authoritative current storage rather
than immediately scattering it into a duplicate global state. Resolve later
input component IDs to `(producer slab, column)` at load time.

This preserves contiguous evaluator output and the current evaluator ABI. It
can remove most measured output-assignment time, at the cost of a segmented
source mapping for later gathers. Selector-pruned chunks require explicit
live-output metadata.

### Flat-input and in-place-output native API

Add an internal borrowed flat-momentum view and
`evaluate_f64_into(momenta, output)`. Keep existing allocating APIs as
wrappers. This can remove:

- nested momentum-vector construction;
- crossing copies where an indexed view is sufficient;
- the final values clone;
- the C API's second result copy;
- unnecessary selected-helicity resolved-output temporaries.

### Smaller execution cleanup

- Separate profiled and unprofiled hot loops so ordinary evaluation does not
  build timing vectors and take per-phase timestamps:
  `rust/crates/rusticol-core/src/engine/evaluation.rs:425`.
- Precompute amplitude canonical-output mappings and selected-helicity
  contraction metadata.
- Avoid amplitude output remapping when generation/load can expose canonical
  output order directly.

For Z+6g, the combined low-risk changes plausibly reduce wall time from
1.704 ms/point to approximately 1.05-1.20 ms/point. This estimate must be
validated because persistent slabs can trade output copies for less-local
subsequent gathers.

## 2. Changes Requiring A New Backend ABI

SymJIT currently accepts contiguous row-major parameter and output matrices:
`rust/crates/rusticol-core/src/evaluator/symjit.rs:113` and
`rust/crates/rusticol-core/src/evaluator/symjit.rs:211`.
The native C++/ASM evaluator has an equivalent dense contract:
`rust/crates/rusticol-core/src/evaluator/compiled.rs:87`.

The following therefore require a new ABI:

- evaluator reads directly from arbitrary current/source/momentum planes;
- gather descriptors, pointer tables, or strided inputs;
- evaluator writes directly into strided global-state destinations;
- multiple arena segments presented as one logical stage input;
- generated amplitude output fused directly into contraction or selected
  totals.

A plane-aware fused-stage ABI could remove virtually all stage input packing
and output assignment while preserving process-specific fusion. For Z+6g,
roughly 0.8-1.0 ms/point appears plausible, bounded by the 0.639 ms evaluator
cost plus reduction and residual runtime work.

This should be treated as a separate SymJIT/backend project after the low-risk
floor is measured.

## 3. Changes That Are Not Appropriate

The compiled lane should not:

- split fused stage evaluators into per-current or per-interaction calls;
- execute prepared transition kernels through Rust contribution tables;
- replace fused stages with dynamic current finalization schedules;
- reconstruct a compact recurrence schedule from the compiled DAG.

Those changes surrender cross-current CSE and stage-level Symbolica fusion and
would effectively create another eager/recurrence lane.

## Reduction Is A Separate Problem

Direct-Arena stage plumbing does not remove the arithmetic required by
NLC/full color contraction. For large pure-gluon artifacts, reduction over
coherent groups and contraction entries is a dominant owner.

Potential follow-up work belongs in a dedicated reduction project:

- structure-of-arrays contraction entries;
- grouping by left/right index;
- folding symmetry factors into weights;
- SIMD or threaded contraction kernels;
- direct evaluator output into coherent-group slots.

## Recommended Follow-Up Experiments

1. A/B direct-global-to-leaf gathering on the preserved Z+6g artifact at
   batches 64, 128, and 1024.
2. Count bytes and effective bandwidth separately for stage-union gather,
   leaf gather, output assignment, amplitude remapping, native input
   construction, and final result copying.
3. Prototype authoritative output slabs for one existing LC artifact and
   measure whether eliminated assignment outweighs segmented gather cost.
4. Add an internal flat-input/direct-output benchmark and compare it with the
   current full native API using identical samples.
5. Split `Other Rusticol core` into batch construction, state preparation,
   selector assembly, resolved-output materialization, and final copying.
6. Benchmark NLC/full reduction alone from a saved amplitude buffer.
7. Consider one plane-aware fused SymJIT stage only if packing plus assignment
   remains above 10% of wall after the low-risk work.

## Recommendation

Implement the one-pass leaf gather and flat-input/direct-output API first.
Prototype persistent output slabs second. These changes are compatible with
the current compiled evaluator fusion and provide a trustworthy measurement
of the remaining ABI-level opportunity. Do not import the recurrence
Direct-Arena interpreter or its per-current execution model into compiled
mode.
