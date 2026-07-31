# SymJIT upstream separation

## Active dependency

pyAmpliCol uses
[`siravan/symjit-crate`](https://github.com/siravan/symjit-crate) 2.22.0 at
the immutable revision
`77789ff0f78232b1ea4608aceb397058df50b06d`. Contributor and release
dependency policy both authenticate that repository and revision. Contributor
installation downloads the checksummed archive into
`dependencies/checkouts/symjit`, verifies the pristine source tree, applies
the ordered generic patch set, verifies the resulting tree, and path-patches
both pyAmpliCol and Symbolica to that single checkout.

There is one generic local patch:

```text
dependencies/patches/symjit/upstream/
  0001-Expose-a-stable-raw-P-kernel-plane-descriptor.patch
```

Its revision, SHA-256, and patched-tree identity are authoritative in
`dependencies/contributor-lock.toml`. The patch adds a `#[repr(C)]` raw plane
descriptor and scalar/SIMD P-kernel accessors. It does not change generated
kernel bodies, add pyAmpliCol concepts, or encode amplitude schedules,
operations, factors, or fanout. The patch exists because Rust mutable-slice
descriptors cannot soundly express duplicate inputs or intentional
input/output aliases. It is intended to be submitted upstream as a generally
useful SymJIT interface.

This boundary supersedes the former private-fork DirectApplication and
DirectTable design. The complete migration and acceptance gates are recorded
in
[`SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md`](SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md).

## Standard P-kernel model

pyAmpliCol translates Symbolica's structured evaluator instructions into a
normal `symjit::Compiler` application with `Config::set_direct_arena(true)`.
It pins the currently reserved direct-arena operation field to canonical zero
and enables identity-output scaling; Rusticol continues to own overwrite/add
policy. The pinned upstream implements
`set_direct_arena_identity_output(false)` by multiplying outputs with the
values passed through `params`; pyAmpliCol deliberately leaves that coefficient
path dormant and retains its Rust-owned factor epilogues. Arena mode selects
SymJIT's standard Sympy-style P-kernel:

```text
f(null, plane_descriptors, point_index, params)
```

The application retains the ordinary
`symjit-application-storage-v3` portable-storage contract. pyAmpliCol records
a separate `pyamplicol-symjit-plane-application-v2` binding contract for
split-complex plane order, input and output counts, deterministic compiler
settings, target requirements, and the structured-instruction source digest.
Rusticol keeps the sealed `Applet` alive and invokes its scalar or SIMD
P-kernel directly through the raw plane descriptors.

Scalar and SIMD indices are actual row (point) indices. Rusticol handles
unaligned heads and tails with the scalar P-kernel and uses the available
two-lane AArch64 or four-lane x86 kernel only for complete blocks. A
scalar-only kernel remains a supported fallback.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| Structured instruction translation, P-kernel compilation, portable application storage, and scalar/SIMD machine code | SymJIT |
| Stable raw plane-descriptor type and raw P-kernel accessors | Generic SymJIT patch |
| Deterministic compiler configuration and plane-binding metadata | pyAmpliCol |
| Arena allocation, persistent broadcast and scratch planes, and descriptor lifetimes | Rusticol |
| Schedule order, recurrence roles, eager row orchestration, fanout, and hazard rejection | Rusticol |
| Overwrite/add policy, complex factors, and before-write snapshots | Rusticol |
| Scalar/SIMD head and tail dispatch, warmed-allocation accounting, and traffic accounting | Rusticol |
| Artifact capability selection, compatibility rejection, and prepared-model policy | pyAmpliCol/Rusticol |

SymJIT therefore owns a standard P-kernel prologue, body, and epilogue.
pyAmpliCol does not ask SymJIT to understand currents, closures, recurrence
roles, DirectTable rows, attachments, or model-specific scalar catalogs.

## Output semantics

Direct output binding is deliberately narrower than general recurrence or
eager semantics. Rusticol binds a P-kernel output plane directly to an arena
destination only when all of the following are proven:

1. the operation is overwrite;
2. the output factor is exactly the identity;
3. the destination is disjoint from every other output;
4. no input needs a before-write snapshot; and
5. the invocation has no unresolved alias or fanout hazard.

The P-kernel epilogue then performs the identity overwrite directly into the
destination plane. Compiled fused stages use this form for their authenticated
disjoint outputs. Recurrence rows and eager attachments may also use it for a
single proven-safe identity overwrite.

Every accumulation, non-identity factor, before-write alias, or multi-target
fanout instead evaluates the P-kernel once into persistent split-complex
scratch planes. Allocation-free Rusticol loops then apply the exact complex
factor and ordered overwrite/add policy. Point-independent literals,
couplings, and model parameters use persistent broadcast planes; constants
are filled once and parameter planes refresh only when their values change.
Structural zeros reuse the shared zero plane.

This preserves the normal SymJIT kernel body while keeping pyAmpliCol's
scheduling semantics explicit and testable.

## Validation and trust boundary

Serialized SymJIT applications are trusted compiler inputs, not a sandbox.
Before binding them, pyAmpliCol and Rusticol authenticate:

- application and plane-binding ABI identities;
- split-complex descriptor order and exact plane counts;
- structured-instruction source digest;
- optimization, SIMD, complex, compression, threading, fast-math, and arena
  settings;
- target word size, endianness, and runtime requirements;
- descriptor ranges, plane lengths, ownership, and allowed aliases; and
- the execution lane's overwrite, factor, attachment, and hazard invariants.

Old private-fork compiled, recurrence-direct, eager-table, and prepared-pack
artifacts are rejected deterministically with a regeneration instruction.
The same is true of `pyamplicol-symjit-plane-application-v1` applications:
they encode the superseded SIMD-block index contract rather than the pinned
actual-row contract. There is no compatibility loader which rewrites either
executable format.

The dependency gate and `cargo tree` must show one SymJIT 2.22.0 source from
`siravan/symjit-crate`. Historical performance captures may retain their
original dependency identities as provenance, but no active build,
configuration, generated asset, or runtime adapter may depend on the former
private fork.
