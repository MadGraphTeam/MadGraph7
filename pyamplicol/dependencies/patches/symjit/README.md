# SymJIT patch series

This development-only series targets released SymJIT 2.21.1 at revision
`48197f32536c894b51ef25b2cf05ddd05c22675f`.

## Patches

1. `0001-aarch64-compression-and-direct-arena.patch` combines two compatible
   extensions against the same immutable upstream revision:
   - deterministic relative compressed-funclet calls for scalar, SIMD, and
     fast-complex AArch64 code;
   - the portable Direct-Arena application transform used by Rusticol
     recurrence execution.

   Direct-Arena kernels read and update aligned split-complex arena planes
   directly, so the recurrence runtime does not construct packed evaluator
   inputs or scatter outputs. Multi-component in-place finalizers snapshot
   their fixed inputs in generated stack storage before aliased writes,
   preserving full-current semantics without caller-side scratch buffers. The
   extension supports scalar and vector AMD64 and AArch64 generators.
   SymJIT 2.21.1 already provides the AArch64 scalar return-status fix; this
   patch does not replace or modify that upstream fix.

2. `0002-eager-and-compiled-direct-arena.patch` must be applied after `0001`.
   It adds:
   - the factor-free fused-stage writer
     `symjit-direct-application-storage-v3` used by compiled execution;
   - the fixed-width descriptor
     `symjit-direct-table-descriptor-v1` and binding contract
     `symjit-direct-table-binding-v2` used by table-aware eager execution;
   - checked metadata and alias validation, scalar/SIMD odd-tail handling, and
     allocation-free warmed calls.

   The current writer remains v3. A narrowly scoped, bounded, read-only
   `symjit-direct-application-storage-v1` loader is retained solely for
   accepted recurrence artifacts: it validates the already-lowered exact-factor
   O2 application and cannot rewrite it as v3. V2 remains unsupported.

3. `0003-x86-direct-table.patch` must be applied after `0002`. It adds native
   x86-64 execution for `DirectTableApplication`: scalar head/tail kernels and
   an AVX4 middle kernel execute the table row/point loops inline, load split
   arena planes directly, and scale/fan out multiple destinations without a
   dense-row wrapper. The patch also makes the table code-generation envelope
   architecture-neutral while retaining the accepted AArch64 implementation
   and strict prepared-O3 parity tests.

The ordinary `symjit-application-storage-v3` ABI and non-Direct-Arena
evaluation paths are unchanged. The local contributor build still applies its
existing manifest rewrite separately; that mechanical `cdylib` to `rlib`
change is intentionally excluded from this patch series. The lock records both
the post-patch/pre-rewrite tree and the final configured tree, which makes
rerunning the installer idempotent even though the ordered patches touch some
of the same files.

This patch series is a candidate for upstreaming. It is not included in release
wheels or sdists. A release build must use a published SymJIT implementation of
the same Direct-Arena contracts.
