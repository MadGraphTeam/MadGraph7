# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

## Release Mode

Release-equivalent builds use exact versions published on PyPI and crates.io,
except for SymJIT, which Cargo resolves from one immutable commit of the
official `siravan/symjit-crate` repository. `release-lock.toml`
records its version, repository, and full revision; `Cargo.lock` records the
normal Git resolution. There is no downloaded source archive, local patch
application, source-tree fingerprint, or release-only Cargo path projection.
The ordinary locked Cargo build is the dependency check. The generic
plane-descriptor change is already upstream in `siravan/symjit-crate#1`.

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
package paths, deletes the auxiliary store, and validates the result. The
retained sdist therefore contains only canonical release payloads;
contributor and bootstrap builds continue to use or omit the candidate
payloads exactly as before. Prepared-pack compatibility remains bound to its
model/compiler identities, project-owned storage and plane ABIs, target, and
payload hashes—not to a redundant dependency checkout fingerprint.

## Candidate Development Mode

`just dev-install` uses immutable Symbolica/GammaLoop source revisions and
clones the exact SymJIT upstream revision from `release-lock.toml` into
`dependencies/checkouts/symjit`. It checks the detached Git revision and the
crate name/version/`rlib` manifest, then path-patches both pyAmpliCol and
Symbolica to that checkout. The official revision carries the generic raw
plane-descriptor API; pyAmpliCol contains no local SymJIT patch machinery.
The change does not alter generated kernel bodies or contain pyAmpliCol
scheduling policy.
The callable is explicitly unsafe: callers must validate descriptor lifetime,
length, alignment, alias synchronization, and mutability before invocation.
Its accessor returns no callable for ordinary non-arena kernels, preventing a
B-kernel from being recast accidentally as a plane-oriented P-kernel.
Candidate mode exists for development and physics validation only. Ordinary
`just dev-install` builds a
complete candidate wheel from the tracked prepared-model packs and portable
self-test fixture. If those generated assets must be replaced after a native
ABI change, an explicitly requested prepared-model bootstrap produces a
non-publishable recovery wheel that omits both asset families; this exceptional
mode is never enabled by the installer itself.
If a managed checkout belongs to a superseded revision, `--update` moves it to
the pinned revision; `--reset` archives managed state in the workspace-local
`.trash` store and recreates it.
It installs the verified published `ufo-model-loader==0.1.7` wheel directly
from the hash-locked runtime closure. Artifacts produced in this mode record
the candidate revisions and are not eligible for PyPI publication.

The contributor and release builds use SymJIT 2.22.0 at immutable upstream
revision `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`. Rusticol builds its plane-oriented
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
same immutable Git revision rather than compiling two copies. This is suitable for
pyAmpliCol's precompiled Python-wheel publication workflow because no Rust
crate is published to crates.io.

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
