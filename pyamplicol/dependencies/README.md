# Dependency Modes

pyAmpliCol separates publishable release dependencies from pinned upstream
sources used for candidate development. Both modes target Symbolica **3.0.0**
and SymJIT **2.26.0**; candidate builds do not relabel dependency versions or
modify upstream source files.

## Release Mode

Release-equivalent builds use exact published versions. Canonical Cargo
resolution uses the published Symbolica 3.0.0 crate; its only
`[patch.crates-io]` entry is SymJIT, pinned to one immutable commit of the
official `siravan/symjit-crate` repository. `release-lock.toml` records that
version and revision, and `Cargo.lock` records normal Cargo resolution. The
ordinary locked Cargo build is the dependency check; no local dependency
patches or release-only source projections are needed.

Python Symbolica 3.0.0 and ufo-model-loader 0.1.8 remain publication blockers:
their release-wheel entries stay empty until the required releases are
available. Successful candidate builds do not establish release availability
or make candidate artifacts publishable.

Prepared models in `src/pyamplicol/assets/prepared_models` are candidate inputs.
Release pairs live separately in the source-only `release_assets/prepared_models`
store and are regenerated through the manual `release-prepared-models.yml`
workflow using `release-lock.toml` and canonical `Cargo.lock`. Its temporary
bootstrap wheel is non-publishable and omits prepared models and the portable
self-test fixture. The release overlay installs the release pairs at canonical
package paths and removes the auxiliary store, so the retained sdist contains
only release payloads. Prepared-pack compatibility uses model/compiler
identities, storage and plane ABIs, target, and payload hashes—not a separate
dependency checkout fingerprint.

## Candidate Development Mode

`just dev-install` uses unmodified managed checkouts under
`dependencies/checkouts`, pinned by `contributor-lock.toml` and the SymJIT
release entry:

| Dependency | Upstream source | Revision |
| --- | --- | --- |
| Symbolica 3.0.0, including Numerica and Graphica | `symbolica-dev/symbolica`, `main` | `7b31114c7a77571aabacf96a311f27ef0f44d007` |
| SymJIT 2.26.0 | `siravan/symjit-crate` | `530304a07d1be6d5abc80291aa5e92cf28ac5546` |
| GammaLoop, including Spenso/Idenso/Vakint | `alphal00p/gammaloop`, `main` | `312cf6aaf4cd1414f6742ad87ce39922c497a0b6` |
| symbolica-integrate 2.0.1 | `symbolica-dev/symbolica-integrate`, `main` | `716354a07f2660b38390c95702aa617d1f9472c7` |
| ufo-model-loader 0.1.8 | `alphal00p/ufo_model_loader`, `main` | `70ddee6b416f8c8b340e0d087646d77095c5d24b` |

The pinned Symbolica source declares 3.0.0, matching the loader's genuine
`symbolica>=3.0` requirement. The loader lives at
`dependencies/checkouts/ufo-model-loader`; no separate future-loader checkout
is needed. The symbolica-community revision is recorded in
`contributor-lock.toml`.

The generated candidate Cargo overlay selects these upstream sources for both
pyAmpliCol and the community extension without changing canonical release
resolution. The installer configures the community build manifest with the
managed source paths and Symbolica's matching allocator; upstream Rust source
files remain unchanged. Rusticol enables Symbolica's
`native_code_generation` feature. SymJIT supplies the standard P-kernel
interface; Rusticol owns scheduling, factors, accumulation, fanout, and artifact
binding. Its plane adapter uses actual row indices and identity output, with
descriptor lifetime, alignment, aliasing, and mutability checked by the caller.

Candidate mode is for development and physics validation, not PyPI publication.
Ordinary `just dev-install` builds a complete candidate wheel using the tracked
prepared-model packs and portable self-test fixture. If a native ABI change
requires replacing those assets, an explicitly requested bootstrap produces a
non-publishable recovery wheel that omits both; the installer never selects this
mode automatically. `--update` moves superseded managed checkouts to their
pinned revisions; `--reset` archives managed state in the workspace-local
`.trash` store before recreating it.

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
