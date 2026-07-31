# Archived SymJIT ARM64 scalar-status reproducer

This directory preserves the source of the historical
DirectApplication-based reproducer described in
[`SYMJIT_ARM64_DIRECT_ARENA_SCALAR_STATUS_MRE.md`](../../development/arena/SYMJIT_ARM64_DIRECT_ARENA_SCALAR_STATUS_MRE.md).

The defect was fixed upstream before SymJIT 2.22.0. pyAmpliCol no longer
depends on the private DirectApplication API, so the standalone Cargo manifest
and its stale SymJIT 2.20.2 lock were deliberately retired. `main.rs.txt` is
evidence, not active developer tooling, and is not expected to compile against
the standard P-kernel interface.

Current scalar/SIMD P-kernel status, alias, duplicate-plane, and tail behavior
is covered by the SymJIT raw-plane patch tests and Rusticol's focused plane
adapter tests.
