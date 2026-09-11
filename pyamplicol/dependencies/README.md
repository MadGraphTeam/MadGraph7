# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

This experimental branch uses the explicit local Cargo overrides
`TMP_FIXED_SYMJIT` and `TMP_FIXED_SPENSO`. They are not committed or distributed.
Prepare these checkouts before running `just dev-install`; the installer uses
their paths directly instead of cloning replacement SymJIT/GammaLoop sources.
It also installs the separate `FUTURE_ufo_model_loader` checkout at version
0.1.8 when present. The loader's own upstream `main` contains these changes;
the checkout is not part of this repository, and 0.1.8 is not yet on PyPI. The separate
upstream compatibility changes are documented in
`SPENSO_LATEST_SYMBOLICA_COMPATIBILITY_FIXES`. This configuration is not ready
for publication until the fixes are available upstream.

The future Python release requirements are Symbolica 3.0.0 and
ufo-model-loader 0.1.8. Their release-wheel entries remain empty until publication.
The pinned Symbolica development source still declares 2.2.0; the existing
candidate build projects that actual version into its wheel without relabelling
the CAS. Local checkout installations therefore use `--no-deps` until Symbolica
publishes 3.0. The loader itself retains its genuine `symbolica>=3.0` requirement.

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

`just dev-install` uses the Symbolica source revision in `contributor-lock.toml`
and the explicit root Cargo path overrides. Without a local override it clones
the pinned SymJIT or GammaLoop revision into `dependencies/checkouts`.
Community Cargo manifests resolve the same source paths as pyAmpliCol; the
installer does not rewrite Symbolica, Spenso, or SymJIT source files or their
manifests. The SymJIT checkout must expose the matching `rlib` library.
The official base revision carries the generic raw
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
The dedicated `FUTURE_ufo_model_loader` checkout replaces the older published
loader in the contributor installation, including the current-CAS compatibility
adaptations. Artifacts produced in this mode record
the candidate revisions and are not eligible for PyPI publication.

The temporary SymJIT checkout starts from 2.25.0 at upstream revision
`f1c193d301897149de6609f706297b0c97a4f018`, with the separately recorded local
compiler fixes. Rusticol builds its plane-oriented
arena adapter from SymJIT's standard P-kernel interface and owns all
pyAmpliCol-specific scheduling, factor, overwrite/accumulate, fanout, and
artifact-binding policies. The pinned upstream P2 contract interprets scalar
and SIMD indices as actual row numbers and can optionally scale outputs by
`params`; pyAmpliCol uses row indices and keeps identity output enabled.

The build uses Symbolica development revision
`0084bc7c1418940fdec652059cd704e00d07e9d1` with `wide >= 1.7` and the pinned
symbolica-community source. No Symbolica source patch is needed. The local
GammaLoop checkout starts from `simplify-spenso-api` revision
`5aadd389efabb7b039af74edad02a90d486a1c07` and contains the documented API
adaptations for that CAS. The same local tree holds the integration adaptation;
the independent UFO loader is in `FUTURE_ufo_model_loader`. Spynso3 initializes
its cached symbolic-parallelism policy in `Auto`
mode, checking the license once and keeping symbolic tensor reductions serial
for restricted users or parallel for licensed users.

The workspace's ordinary `[patch.crates-io]` entries keep the selected CAS
revision and local SymJIT implementation consistent. Contributor builds use
the matching managed CAS paths through the generated Cargo configuration.

The original Fortran AmpliCol checkout is optional, developer-only, and used
only as an independent validation and benchmarking reference. Enable it with
`just dev-install --with-legacy-amplicol`. The pinned
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

The Reference FFT checkout is likewise developer-only and is used only by the
FFT profiling and hardware-sensitive acceptance tools. Enable it with
`just dev-install --with-reference-fft`, which clones the exact
public `AllGluonsMultipletFFT` revision recorded in `contributor-lock.toml`
into `dependencies/checkouts/reference-fft`. That repository keeps the
upstream contraction implementation and adds the calibrated
batch/helicity-sum benchmark interface plus a high-multiplicity roundoff guard.
It is never redistributed in a wheel or sdist. Ordinary `just dev-install`
runs omit both external references. Enable both for the complete FFT profiling
toolchain with
`just dev-install --with-legacy-amplicol --with-reference-fft`.
Enabling either external profiling reference also installs the project-owned
`fft-profiling` Python extra into `.venv`; no editable project reinstall is
required or supported. When updating a source tree that already has the former
Reference FFT checkout, add `--update`; the installer migrates its managed
`origin` to the public repository before fetching the pinned revision.
