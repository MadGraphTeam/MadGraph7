# SymJIT contributor and upstream patch series

The patches in `upstream/` target SymJIT 2.21.1 at revision
`48197f32536c894b51ef25b2cf05ddd05c22675f`. They contain only generic
SymJIT capabilities; pyAmpliCol recurrence roles and artifact policy remain in
Rusticol.

## Upstream submission order

1. `0001-Build-SymJIT-as-an-rlib-for-Rust-integration-tests.patch`
   makes the crate available to Rust integration tests while retaining its
   `cdylib` output.
2. `0002-Support-compressed-funclets-on-AArch64.patch` adds deterministic
   relative funclet calls for scalar, SIMD, and fast-complex AArch64 code.
3. `0003-Add-generic-direct-plane-applications.patch` adds a direct
   split-plane application API with explicit overwrite/accumulate,
   live/snapshot-input, and identity/complex-scalar policies.
4. `0004-Add-generic-table-driven-direct-applications.patch` adds a generic
   table-driven direct application for repeated row/point execution and
   multi-destination fan-out on AArch64 and x86-64.

Patches 2–4 are the contributor-build patch closure. Patch 1 is excluded from
that closure because the installer already performs its equivalent mechanical
`cdylib`-to-`rlib` manifest rewrite; including both would make the configured
tree dependent on how the source was materialized.

The direct application input format is for trusted bytecode. It is not a
sandbox or hostile-input parser. Shape, range, and alias checks remain because
they are part of the safe calling contract and prevent accidental undefined
behavior.

The ordinary `symjit-application-storage-v3` contract and non-direct
evaluation paths are unchanged. The new, previously unreleased generic
contracts start at `symjit-direct-application-storage-v1`,
`symjit-direct-table-descriptor-v1`, and
`symjit-direct-table-binding-v1`; no pyAmpliCol-specific compatibility loader
is carried upstream.

See `docs/development/arena/SYMJIT_UPSTREAM_SEPARATION.md` for the ownership
boundary, validation evidence, and publication migration.
