# SymJIT upstream separation

## Objective

Replace pyAmpliCol's combined 9,231-line SymJIT delta with a generic,
reviewable SymJIT series and a small pyAmpliCol-owned adapter. The eventual
release path is patch-less: pyAmpliCol depends on a published SymJIT release
that contains the generic APIs and keeps all amplitude-specific policy in
Rusticol.

The separation is based on SymJIT 2.21.1 commit
`48197f32536c894b51ef25b2cf05ddd05c22675f`.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| AArch64 relative calls and deterministic funclet ordering | SymJIT |
| Direct split-plane JIT code generation | SymJIT |
| Overwrite versus accumulate destination semantics | Generic SymJIT API |
| Live versus before-write input snapshots | Generic SymJIT API |
| Identity versus complex-scalar output scaling | Generic SymJIT API |
| Table-driven row/point loops and destination fan-out | SymJIT |
| pyAmpliCol recurrence-role interpretation | Rusticol |
| Artifact capability selection and prepared-model policy | pyAmpliCol/Rusticol |

Rusticol maps recurrence roles to generic operations as follows:

| pyAmpliCol role | Destination | Input view |
| --- | --- | --- |
| initialize | overwrite | live |
| add contribution | accumulate | live |
| finalize in place | overwrite | before-write snapshot |
| closure add | accumulate | before-write snapshot |

All recurrence forms use complex-scalar scaling. Compiled identity writers use
overwrite, live input, and identity scaling.

## Proposed upstream commits

1. Build SymJIT as an `rlib` for Rust integration tests: 1 insertion and
   1 deletion.
2. Support compressed funclets on AArch64: 214 insertions and 12 deletions.
3. Add generic direct plane applications: 2,758 insertions and 111 deletions.
4. Add generic table-driven direct applications: 5,602 insertions and
   2 deletions.

The table layer is intentionally last and may be proposed separately if the
maintainer prefers a smaller initial review. The contributor installer applies
commits 2–4. It already performs the build-manifest rewrite from commit 1.

## Trusted input and validation

Stored direct applications are trusted bytecode. The loader is not a sandbox
and does not manually vet binaries for malicious content. It still validates
format identity, dimensions, ranges, and pointer-alias contracts: those checks
protect the API against accidental misuse and undefined behavior, not against
an adversary.

Because the old direct formats were never published, the generic upstream API
starts at:

- `symjit-direct-application-storage-v1`;
- `symjit-direct-table-descriptor-v1`;
- `symjit-direct-table-binding-v1`.

There is no legacy pyAmpliCol recurrence loader in the upstream proposal.
Prepared direct applications must be regenerated once when adopting the new
format.

## Validation

The clean upstream series was replayed from the exact base commit. On
AArch64, SymJIT passed 45 unit tests and 4 integration tests. The same tree
passed `cargo check --target x86_64-apple-darwin --all-targets`.

Rusticol, configured against the separated generic tree, passed 561 tests with
5 ignored. This covers the recurrence role mapping, compiled direct
applications, and eager table adapter.

## Publication path

The repository remains release-pinned to published SymJIT 2.18.9. As of
2026-07-26, crates.io exposes SymJIT 2.21.0, so neither the 2.21.1 Git state nor
this generic series is available to a crates.io-only build.

A publishable Rust crate cannot rely on a Git-only dependency in its packaged
dependency graph. The final sequence is therefore:

1. upstream and release the generic changes in SymJIT;
2. update pyAmpliCol's exact crates.io SymJIT pin and release lock;
3. regenerate prepared direct assets under the new v1 contracts;
4. run cross-platform wheels and strict release gates;
5. publish through the existing trusted-publisher workflow.

No publication action is part of this branch.
