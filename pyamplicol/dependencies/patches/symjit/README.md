# SymJIT patch series

This development-only series targets released SymJIT 2.21.1 at revision
`48197f32536c894b51ef25b2cf05ddd05c22675f`.

## Patches

1. `0001-direct-arena-applications.patch` adds the portable Direct-Arena
   application transform used by Rusticol recurrence execution. Generated
   kernels read and update aligned split-complex arena planes directly, so the
   recurrence runtime does not construct packed evaluator inputs or scatter
   outputs. Multi-component in-place finalizers snapshot their fixed inputs in
   generated stack storage before aliased writes, preserving full-current
   semantics without caller-side scratch buffers. The patch supports scalar
   and vector AMD64 and AArch64 generators.
   SymJIT 2.21.1 already provides the AArch64 scalar return-status fix; this
   patch does not replace or modify that upstream fix.

The ordinary SymJIT application ABI and non-Direct-Arena evaluation paths are
unchanged. The local contributor build still applies its existing manifest
rewrite separately; that mechanical `cdylib` to `rlib` change is intentionally
excluded from this patch.

This patch is a candidate for upstreaming. It is not included in release
wheels or sdists. A release build must use a published SymJIT implementation
of the same Direct-Arena contract.
