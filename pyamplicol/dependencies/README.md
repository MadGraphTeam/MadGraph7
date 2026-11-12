# Dependency Modes

The `loop` development branch integrates Rust Symbolica 3.0.2, Numerica and
Graphica from symbolica-dev/symbolica's `community` branch `36cbac3a` (main
`9de26e4e` plus the author's merged commits) with one consolidated upstream
pull request on top: batched evaluation (`BatchExpressionEvaluator`),
`DualNumber` component views, saved evaluators that hold only the state
they reference, correct complex-modulus derivatives in Dual domains,
deterministic signs of integer polynomial factors, polynomial exponents
bounded by the i32 conversion range, a journaled conditional cache in the
direct evaluator translator (no cache copy per `if` arm; the generated
instructions are unchanged) and native Python `ComplexFloat` operands taken
at the more precise component's precision
([symbolica-dev/symbolica#163](https://github.com/symbolica-dev/symbolica/pull/163),
base `community`; its nine commits also apply cleanly on `main`). The pinned
revision is the PR head
[`dfe6bd97`](https://github.com/ValentinHirschi/symbolica/commit/dfe6bd97b974b1028dc63071ea165136cd2b0ef5)
(branch `loop_me_pyamplicol` of ValentinHirschi/symbolica).
Upstream `main` alone does not build the workspace: NinjaRS and
`rusticol-core` / `loop-reduction` use the batched evaluator, NinjaRS the
`DualNumber` trait, and NinjaRS's rank n+2 Dual derivatives need the
complex-modulus fix, and NinjaRS's determinism test the deterministic factor
signs; the Python Community modules (FeynKit, HepKit) need the `community`
branch's `symbolica::utils::resources`. The pin moves to upstream once the PR
is merged. NinjaRS (main `5fe7f392`, the merge of `71eddb15`: the box mu-expansion's conditioning bound, on main
`98e223da` = main `b3a29c85`, PR #14, its 0BSD relicence and the re-pins of its
manifests on this revision and oneloopmaster `bd703d09`) pins the same
revision, as does oneloopmaster (main `bd703d09`) in its own standalone `[patch]`,
which only applies when it is the root workspace. There
is no alphal00p Symbolica fork in the graph (`cargo tree -i symbolica` lists one
source, and `cargo tree -i oneloop` one oneloop). SymJIT stays 2.28.3. These
pins are recorded by Cargo. This is a source integration, with Rust 1.96 or
newer required; the git `[patch]` still fails the published-crate gate, so the
branch is not publishable until the PR is in a Symbolica release.

TIRust ([huashengshao/TIRust](https://github.com/huashengshao/TIRust), crates
`iregi-rust` and `iregi-masters`, pinned to a commit of its main after its one
history rewrite) is one more git source, linked by the cargo feature `tir`, a
default feature of every crate that links it (`--no-default-features` builds
without it). Its crates require `symbolica = "3.0"` and the oneloop revision
and features pinned here, so the root `[patch]` and oneloop pin serve them
(one source each, unchanged feature sets). Like the NinjaRS and oneloopmaster
pins it is a git source for the publication gate.

The published Python/Community distribution is pinned to **3.0.3**, which the
author builds from the same `community` branch (without the PR); its
`Evaluator.save()` envelope is the one the native reads. 3.0.2's raw-bincode
saves are refused: Symbolica 3.0.2 changed the saved-evaluator and expression
export formats, so the serialization ABI is `symbolica-bincode2-v2` and every
stored evaluator state was regenerated.
Without the PR, 3.0.3 saves the producer's whole Symbolica state: after a
loop_sm compile about 20-30 MB per evaluator, depending on every symbol the
Python process ever created. Stored as such, a u u~ > d d~ g loop artifact
took 4.7 GB instead of 5.5 MB and 29 s instead of 0.3 s to load. And any
save, the PR's included, numbers the symbols it references by the producer's
registry, so its bytes depended on the order in which the process created
namespaces and symbols (a cold or a warm model cache gave different
`parameters.bin`). Every generation writer therefore stores
`symbolica_helpers.saved_evaluator`, the native's canonical form
(`rusticol-core` `canonical_saved_evaluator`): only the symbols the evaluator
references (a whole state is cut with `State::export_symbols`, the form the
PR's save writes), their records ordered by name and numbered 0, 1, ...; the
payload is kept byte for byte (it names its symbols by definition, so the
numbering only maps the state), and the result must decode to the same
evaluator. The stored bytes then depend on the evaluator alone, not on the
producer nor on the process history (loop artifacts generated with PyPI 3.0.3
and with the Community build, the EW self-test included, are byte-identical
but for `artifact.json`'s creation time and stage timings). A section whose symbol data embeds an expression keeps its
numbering (the expression carries it; no pyAmpliCol evaluator references such
a symbol). Generation imports each whole state once into the native: a symbol
whose definition differs from the native's (the Community modules carry
gammaloop `9102f885`, the native's FeynKit `d1904bf2`; `gammalooprs::Q`
differs in one Spenso tag) is imported under a fresh name
(`<name>_pyamplicol_import`) instead of failing (`load_evaluator_state_bytes`, the same rule on load). Symbolica
warns about each imported symbol that lacks the user-defined functions the
native gave it; the extension routes Symbolica's diagnostics to standard error
(`rusticol_core::diagnostics`, installed at import), so they never mix with a
program's standard output (`pyamplicol self-test --json`).

The development Python kernel is a release build of the Community recipe
(symbolica-community `39b556c7`, the 3.0.3 recipe, unmodified) with only its
`[patch]` moved to `dfe6bd97`: `builds/community-release-python-dfe6bd97`
(wheel `builds/community-release-wheels-dfe6bd97/`, sha256 `872434c3…`;
recipe in `BUILD-INFO.json`). Native and Python therefore share one Symbolica
revision.

pyAmpliCol defaults to published runtime dependencies and release-mode native
builds. Historical pinned-source candidate machinery is retained, but its
upstream pins need a coherent update before use with the current tensor APIs.
Release mode uses Python Symbolica **3.0.3** (Rust kernel 3.0.2) and SymJIT
**2.28.3** (`release-lock.toml`). Historical candidate mode retains Symbolica
**3.0.0** and SymJIT **2.26.0** (`contributor-lock.toml`), so the two locks pin
different versions; version numbers alone do not establish Python API
compatibility. Candidate builds do not
relabel dependency versions or modify upstream source files.

## Release Mode

`just dev-install` uses pip to resolve the Python runtime dependencies declared
in the root `pyproject.toml`, then replaces published Symbolica and
ufo-model-loader without resolving dependencies again. This replaces older
source candidates reporting the same version. The upcoming release requires
ufo-model-loader 1.0.0, which is not yet published. Until publication, explicitly
supply the locally built loader wheel for contributor validation:

```console
just dev-install --loader-wheel /path/to/ufo_model_loader-1.0.0-py3-none-any.whl
PYTHON=.venv/bin/python just dev-test
```

The installer never chooses a local loader wheel automatically; `--loader-wheel`
is an explicit dependency-development override. Native build and staging use
the existing release-mode workflow. Building a new release wheel requires a
clean Git checkpoint and complete release assets; published-dependency mode
does not relax publication guards. For dirty development checkouts,
`--wheel-directory PATH` reuses an already-built compatible release wheel, or
`--dependencies-only` skips the project build and staging while leaving the
existing native runtime untouched. `--no-build` also skips runtime dependency
installation.
Developer tools and explicitly requested optional references may still be
set up with `--no-build`. No editable installation is used. `just dev-test`
defaults to release mode and includes a fresh build, so it also requires a clean
checkpoint. Dirty edits can use focused tests against the staged native runtime.
The developer dashboard retains its pinned Ratatui 0.4.2 build and FFI checkout
in both lanes; it is not a core runtime dependency.

Release-equivalent builds use exact versions. `release-lock.toml` records Rust
Symbolica 3.0.2 and SymJIT 2.28.3, and `Cargo.lock` records their resolution:
SymJIT from the registry with its checksum; Symbolica, Numerica and Graphica
from `dfe6bd97` (the PR head above) through the root `Cargo.toml`'s
`[patch.crates-io]`, the source patch that still fails the published-crate gate
(see above). The ordinary locked Cargo build is the dependency check; no
release-only source projections are needed.

Python Symbolica 3.0.3 is published on PyPI; its wheels are recorded in
`python-runtime-lock.toml`. The required ufo-model-loader 1.0.0
entry has no published wheel recorded in `python-runtime-lock.toml` yet. After
the loader is uploaded, record its official PyPI wheel metadata before running
the final release gates. A local wheel is not a published runtime-lock artifact.
Successful candidate builds do not establish release availability or make
candidate artifacts publishable.

Prepared models in `src/pyamplicol/assets/prepared_models` are the source-runtime
inputs. Release pairs use the release-locked dependencies and also
live in the source-only `release_assets/prepared_models` store and are regenerated
through the manual `release-prepared-models.yml`
workflow using `release-lock.toml` and canonical `Cargo.lock`. Its temporary
bootstrap wheel is non-publishable and omits prepared models and the portable
self-test fixture. The release overlay installs the release pairs at canonical
package paths and removes the auxiliary store, so the retained sdist contains
only release payloads. Prepared-pack compatibility uses model/compiler
identities, storage and plane ABIs, target, and payload hashes—not a separate
dependency checkout fingerprint.

## Historical Candidate Mode (Opt-in)

The retained `--candidate` dependency-development machinery selects managed
checkouts under `dependencies/checkouts`, pinned by `contributor-lock.toml`.
These historical pins are incompatible with the
current tensor implementation and must be updated together before use; this
is not a working current-source setup route. The retained pin table is:

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

Candidate mode is not for PyPI publication. Its retained test selector,
`PYAMPLICOL_BUILD_MODE=candidate`, does not make the historical pins compatible.
The retained build machinery uses the tracked prepared-model packs and
portable self-test fixture; an explicitly requested bootstrap omits both and
is non-publishable. Neither bootstrap nor `--update` repairs incompatible
upstream pins: `--update` only moves managed checkouts to the recorded revisions.
`--reset` archives managed state in the workspace-local `.trash` store before
recreating it.

## Optional Profiling References

These opt-ins are independent of the historical candidate machinery.

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
