# Eager/Compiled Direct-Arena ABI Prototype

> Historical milestone record. The dependency, patches, and v1 binding
> identities below describe the superseded prototype, not the current build.
> See
> [`SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md`](SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md)
> for the active SymJIT 2.22.0 P-kernel design.

This milestone pins the risk-first SymJIT substrate needed by the independent
eager and compiled execution lanes. It is not yet an end-to-end runtime
performance claim.

## Exact inputs

- SymJIT release: `2.21.1` at
  `48197f32536c894b51ef25b2cf05ddd05c22675f`.
- Existing recurrence/AArch64 patch SHA-256:
  `6d456e69fc160ec5361188f60f994d10fb2dd3360eb47a91c4979a1bde69626e`.
- Eager/compiled patch source commit:
  `e6b261f3ae027c4381ac053dcd5c47c318d2f8ea`.
- Eager/compiled patch SHA-256:
  `3c5e07f6e47e42bb7c51a53ed16fdca402ecae05fceea1bcfcff8b4145bbb299`.
- Ordered patch-closure SHA-256:
  `089b01f39110c9da15f87d1d13af3befdc2ab6d999a437cf81bba2d7043934a7`.
- Post-patch/pre-configuration tree SHA-256:
  `c1aed5919ae63ef705299d220f7d3a872409ca65052c21432ecbb6f13757dd02`.
- Final configured tree SHA-256:
  `9fa001ea3add37341461009ae4ab25093a94f95db9440a2a6a7605373c5f50f7`.

## Frozen internal contracts

- Ordinary source application: `symjit-application-storage-v3`.
- Compiled fused-stage plane kernel:
  `pyamplicol-compiled-plane-kernel-v1`.
- Eager table descriptor: `pyamplicol-eager-plane-table-descriptor-v1`.
- Eager table binding: `pyamplicol-eager-plane-table-binding-v1`.

Compiled applications retain O3 compressed funclets, use identity overwrite
with no output factor, reject input/output and output/output aliasing, support
multiple outputs and odd tails, and allocate no heap storage after warmup.

Eager table applications execute rows outermost and points innermost, preserve
attachment and output order, write every fanout destination directly, reject
same-row/global-buffer alias hazards, and report zero packet, gather, and
scatter materializations. The portable descriptor is fixed-width,
little-endian, bounded, and source-authenticated.

Accepted recurrence storage-v1 payloads retain a narrowly bounded, read-only
loader. They are authenticated as exact-factor O2 applications and cannot be
rewritten as v3. V2 remains unsupported; v3 remains the only writer.

## Rusticol risk adapters

The compiled adapter loads one real compressed O3 DirectApplication v3,
authenticates factor-free identity overwrite and all logical plane/scalar
bindings, pins a fixed descriptor bundle to persistent arena storage, and
invokes the unchecked entry point only after validation. Tests cover point
ranges `1`, `2`, `3`, `127`, `(start=1,count=127)`, `128`, and `129`,
untouched sentinels, malformed payloads, input/output and output/output alias
rejection, literal/model-parameter scalars, panic containment, and zero warmed
allocations.

The eager adapter loads a prepared source application plus descriptor-v1,
owns immutable binding-v2 invocation and attachment rows, and binds stable
plane, scalar, and factor catalogs once. Tests cover two ordered rows, three
fanout attachments, a producer-to-consumer cross-row alias, rejection of a
same-row alias, a seven-point active tail over physical pitch eight, bitwise
parity with a reusable-scratch gather/call/scatter oracle, checked and
validated-unchecked calls, malformed inputs, and zero warmed allocations.

These adapters remain deliberately scheduler-independent. Neither timing
result below is an end-to-end runtime claim:

| prototype | direct median | predecessor median | direct/predecessor |
|---|---:|---:|---:|
| compiled compressed O3, 129 points, 9 interleaved × 10,000 calls | 207 ns/call | 552 ns/call | 0.3750 |
| eager table, 2 rows/3 attachments, 7 points, 9 interleaved × 20,000 calls | 58.496 ns/call | 62.260 ns/call | 0.939535 |

The compiled predecessor is a preallocated pack/call/scatter path. The eager
predecessor is an explicit reusable-scratch gather, ordinary SymJIT batch
call, and ordered scatter. No unit test asserts timing.

## Validation

- Rebuilt the ordered patch chain locally and reproduced the final configured
  tree digest exactly.
- SymJIT debug all-target suite: 49 passed.
- SymJIT release all-target suite: 49 passed.
- Dependency/provenance tests: 96 passed, one expected installed-environment
  skip.
- Offline candidate dependency gate: passed.
- Rusticol recurrence-focused tests against the patched candidate dependency:
  204 passed.
- Generic and recurrence warmed-allocation integration tests: 8 passed,
  including genuine topology-replay and all-flow-union fixtures.
- Compiled Rusticol adapter: 4 release tests passed; independent raw-storage
  safety audit returned GO; peak RSS 4.114 GiB.
- Eager Rusticol adapter: 5 release tests passed and one manual benchmark was
  ignored by default; peak RSS 4.041 GiB.
- Authoritative combined candidate reruns passed the compiled 4/4 and eager
  5/5 focused release tests, with peaks 4.033 GiB and 3.935 GiB respectively.

All mutable sources, caches, targets, patch-chain copies, and test temporary
directories used for this milestone lived inside the dedicated feature
worktree.

## Remaining prototype gates

- Wire the validated adapters into the real eager and compiled schedulers,
  artifact codecs, and lane-specific plans.
- Demonstrate an end-to-end wall-time win before mass migration.
- Implement equivalent x86-64 table callable generation; descriptor
  validation is portable, but table code generation is currently AArch64-only.
- Load and evaluate a retained accepted recurrence artifact end to end with the
  combined candidate build.
