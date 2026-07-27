# SymJIT upstream separation

## Objective

Replace pyAmpliCol's combined 9,231-line SymJIT delta with a generic,
reviewable SymJIT series and a small pyAmpliCol-owned adapter. The active build
path is patch-less: pyAmpliCol consumes one immutable fork revision containing
the generic APIs and keeps all amplitude-specific policy in Rusticol.

The series is based on SymJIT 2.21.1 commit
`48197f32536c894b51ef25b2cf05ddd05c22675f` and ends at fork revision
`60a9d66fbfb2181d36a5747c389714eccc187244`.

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
5. Permit scratch registers for internal complex stack spills while retaining
   the general-register direct-destination guard.
6. Normalize scratch-register direct outputs through a checked, reusable
   four-slot stack frame without clobbering live registers.
7. Make the pyAmpliCol integration branch `rlib`-only so contributor installs
   use the immutable fork without rewriting its manifest.

The table layer and each follow-up are independently reviewable. All seven
commits are on
`ValentinHirschi/symjit_changes_for_pyamplicol:pyamplicol-generic-direct-apis`
and are proposed in `siravan/symjit#12`. The contributor installer applies no
patches or source rewrites.

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

The clean upstream series was replayed from the exact base commit. On AArch64,
the final fork passed Cargo metadata, library check, all 23 focused direct
tests, and all 49 library tests. The earlier four-commit portability series
also passed `cargo check --target x86_64-apple-darwin --all-targets`.

Rusticol, configured against the separated generic tree, passed 561 tests with
5 ignored. This covers the recurrence role mapping, compiled direct
applications, and eager table adapter.

## Publication path

The release workspace redirects crates.io SymJIT to the exact fork revision
through `[patch.crates-io]`. This makes Symbolica 2.2.0 and Rusticol share the
same SymJIT instance and lets the cross-platform CI jobs compile pyAmpliCol
wheels now. The dependency gate verifies the repository, revision, lockfile
source, and absence of local path dependencies.

This is valid for pyAmpliCol because the publication unit is a precompiled
Python wheel; pyAmpliCol is not publishing its internal Rust crates to
crates.io. When the upstream PR is released, the remaining migration is to
remove the workspace override and select the exact crates.io release.

No publication action is part of this branch.
