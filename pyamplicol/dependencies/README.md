# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

## Release Mode

Release-equivalent builds use exact versions published on PyPI and crates.io,
except for SymJIT, which is redirected from crates.io to one immutable commit
of `siravan/symjit-crate`. The exact Git repository, revision, and
build-relevant versions are recorded in `release-lock.toml`.
The release lock also authenticates the upstream archive, its pristine tree,
the ordered generic raw-plane patch, the configured tree, and the
path-resolution form of `Cargo.lock`. The isolated backend materializes that
exact tree inside its build overlay, reusing a contributor checkout only after
its complete tree hash matches. It generates the sole local Cargo override
itself after verification; ambient Cargo configuration is never accepted. The
release sdist carries the authenticated patch and repeats the same
materialization when a wheel is rebuilt from the clean sdist.

`tools/release/check_dependencies.py` is a hard release gate. It verifies the
workspace-level SymJIT source pin, archive and tree identities, patch bytes,
release/contributor agreement, and both the immutable-Git and isolated
path-resolution forms of the fully resolved Cargo lock before release
artifacts are built. It is local and deterministic; the real package and Cargo
builds prove that the pinned archive remains downloadable.

The package-owned prepared models under
`src/pyamplicol/assets/prepared_models` remain candidate inputs in a source
checkout. Release prepared-model pairs are held separately under the
source-only `release_assets/prepared_models` store and are regenerated only
through the manual `release-prepared-models.yml` workflow. Its temporary
bootstrap wheel is explicitly non-publishable and omits both prepared-model
stores and the portable self-test fixture; the resulting architecture pairs
derive their dependency identity from
`release-lock.toml` and canonical `Cargo.lock`, never from contributor state.
A release overlay projects the complete release pair set over the canonical
package paths, deletes the auxiliary store, stages the authenticated SymJIT
patch, and validates the result. The retained sdist therefore contains only
canonical release payloads and the generic release-owned dependency patch;
contributor and bootstrap builds continue to use or omit the candidate
payloads exactly as before. Prepared-pack metadata binds the exact model
compiler, model source, prepared-pack compiler, canonical native build-input
closure, and configured SymJIT tree/patch closure. Packaging and runtime
validation reject drift in those identities and actually load the prepared
bundle. Files outside those explicit build-relevant closures, such as
documentation, do not invalidate a pack.

## Candidate Development Mode

`just dev-install` uses immutable Symbolica/GammaLoop source revisions and the
checksummed SymJIT source archive listed in `contributor-lock.toml`.
That archive is generated from the same upstream Git revision used by release
mode. Candidate and release modes apply the same narrowly scoped, generic SymJIT patch that
exposes a stable raw plane-descriptor callable for P-kernels. The patch does
not change generated kernel bodies or contain pyAmpliCol scheduling policy.
The callable is explicitly unsafe: callers must validate descriptor lifetime,
length, alignment, alias synchronization, and mutability before invocation.
Its accessor returns no callable for ordinary non-arena kernels, preventing a
B-kernel from being recast accidentally as a plane-oriented P-kernel.
The installer authenticates the ordered patch contract, verifies
forward/reverse applicability, and verifies both the pristine and complete
post-patch source-tree identities before building. Candidate mode exists for
development and physics validation only. Ordinary `just dev-install` builds a
complete candidate wheel from the tracked prepared-model packs and portable
self-test fixture. If those generated assets must be replaced after a native
ABI change, an explicitly requested prepared-model bootstrap produces a
non-publishable recovery wheel that omits both asset families; this exceptional
mode is never enabled by the installer itself.
If the managed SymJIT directory belongs to a superseded revision—even when the
crate version is unchanged—the installer moves that exact directory into the
workspace-local `.trash` store and materializes the authenticated revision
automatically. Archive extraction preserves only the authenticated executable
permission bits used by the source-tree digest; it does not import broader
archive permissions.
It installs the verified published `ufo-model-loader==0.1.7` wheel directly
from the hash-locked runtime closure. Artifacts produced in this mode record
the candidate revisions and resulting source-tree identity and are not
eligible for PyPI publication.

The contributor build uses the checksummed upstream archive for SymJIT 2.22.0
at revision `77789ff0f78232b1ea4608aceb397058df50b06d` on
`siravan/symjit-crate`. The installer verifies the archive SHA-256, its safe
member prefix, the complete pristine source-tree digest, the package version,
and the `rlib` crate type before using it. The verified candidate tree differs
from the pristine tree only by the authenticated generic raw-plane descriptor
patch recorded in the contributor lock. Rusticol builds its plane-oriented
arena adapter from SymJIT's standard P-kernel interface and owns all
pyAmpliCol-specific scheduling, factor, overwrite/accumulate, fanout, and
artifact-binding policies. The pinned upstream P2 contract interprets scalar
and SIMD indices as actual row numbers and can optionally scale outputs by
`params`; pyAmpliCol uses row indices and keeps identity output enabled.

The build uses Symbolica and symbolica-community at the immutable
planned-release revisions recorded in the lock. GammaLoop is pinned to the
merged main revision that provides Spenso's
cached symbolic-parallelism policy. Spynso3 initializes that policy in `Auto`
mode, checking the license once and keeping symbolic tensor reductions serial
for restricted users or parallel for licensed users.

Symbolica 2.2.0 is pinned from PyPI and crates.io. SymJIT is pinned through the
workspace's `[patch.crates-io]` table so Symbolica and Rusticol resolve the
same upstream Git revision rather than compiling two copies. This is suitable for
pyAmpliCol's precompiled Python-wheel publication workflow because no Rust
crate is published to crates.io. The Git pin intentionally targets upstream
2.22.0 before that release reaches crates.io; it can be replaced by an exact
registry version once the identical source is published.

The original Fortran AmpliCol checkout is optional, developer-only, and used
only as an independent validation and benchmarking reference. The pinned
`amplicol_with_patches` branch removes the unnecessary LHAPDF
link from its direct color probe, exposes complete recursion-kind diagnostics,
and reports each LC contraction-row partition. This
resolves the physical color component only for a genuinely single-flow case;
multi-flow fixtures use the rows only to verify the complete per-helicity
aggregate. None modifies
amplitude physics. `just legacy-physics` builds that probe and checks the
tracked low-multiplicity LC/NLC/full fixture, including every physical
helicity and every independently resolvable color component, against the
pinned Fortran implementation.

The pinned upstream revision contains no `LICENSE` or `COPYING` file. That fact
is recorded in contributor-only provenance, not the release dependency lock.
The checkout and its developer-only branch are never redistributed in a wheel
or sdist.
