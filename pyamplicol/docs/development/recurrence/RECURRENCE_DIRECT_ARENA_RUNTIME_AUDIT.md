# Recurrence Direct-Arena Runtime Audit

## Scope

This audit explains the remaining runtime gap for built-in-SM topology-replay
`u u~ > Z + 6g` after exact recurrence-count parity and Direct-Arena SIMD were
established. It is a performance diagnosis, not a change to recurrence
semantics.

## Structural Result

The recurrence artifact contains 1,425 currents, 8,338 contributions, and 384
closures. These counts match original AmpliCol after filtering. AmpliCol's
additional numerical filter removes only about 0.2%, so missing recurrence
reuse cannot explain the observed factor-of-two runtime gap.

The prepared applications are transformed by `prepare_simd()`. On AArch64,
each generated call evaluates two points. SIMD therefore applies within each
canonical current kernel, but it neither fuses recurrence rows nor vectorizes
across rows.

## Measured Execution Shape

The qq_Z6g schedule contains 8,338 contribution rows and 858 generated
finalization rows. At batch 1024 and SIMD width two, this produces about 4.71
million generated-function calls per tile, or 4,598 calls per physical point.

Compiled mode also makes many SIMD-block calls while traversing 384 nonzero
helicities. Its applications, however, fuse stage chunks, fold factors into
expressions, and consume contiguous packed inputs. Recurrence invokes small
canonical kernels through indirect arena-plane descriptors and a generic
destination operation.

The runtime currently:

- applies a generic complex factor even though every measured recurrence
  factor is real and about 94% of generated output components use exactly
  `+1` or `-1`;
- rebuilds about 227,000 plane descriptors and 20,000 scalar descriptors per
  tile;
- operates on an approximately 36 MiB current arena at batch 1024, compared
  with about 37 KiB for one point;
- executes 548 unit identity-finalization rows that rewrite 1,752 complex
  components;
- clears destination state before contributions that could initialize it
  directly.

Batch sizes 128 and 1024 are both SIMD-aligned. Their gap is therefore not a
scalar-tail artifact.

## Ranked Bounded Optimizations

1. Add model-generic destination variants for `+1`, `-1`, real, and complex
   factors.
2. Eliminate unit identity finalizers by aliasing or retaining their input
   slots.
3. Mark the first contribution to each current as initialization and later
   contributions as additions, then remove corresponding destination clears.
4. Execute aligned cache-sized point subtiles, initially testing 32, 64, and
   128 points while retaining fixed preallocated workspace.
5. Prebind immutable plane and scalar descriptors once at load.
6. Remove the redundant initial whole-arena clear while preserving necessary
   slot-reuse clears.

Each change requires a matched same-host compiled comparison and component
agreement. These are runtime-representation optimizations only; they must not
alter the proven builder, current identity, color state, helicity ancestry, or
closure semantics.

## SymJIT Observability Gap

`prepare_simd()` currently returns no status and discards internal compilation
failures. Useful future upstream additions would be:

- `try_prepare_simd() -> Result<lane_width, error>`;
- SIMD-block, scalar-tail, and failed-fallback counters;
- optionally, a callable for one aligned SIMD block to permit block-major
  scheduling.

These observability additions are useful but are not prerequisites for the
bounded optimizations above.
