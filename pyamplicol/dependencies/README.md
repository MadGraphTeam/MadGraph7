# Dependency Modes

The `loop` development branch integrates Rust Symbolica 3.0.2, Numerica 3.1.0
and Graphica 3.0.1 from symbolica-dev/symbolica's `community` branch at
[`0e2ad04c`](https://github.com/symbolica-dev/symbolica/commit/0e2ad04c4f281e28bd29d5ad8b1bebf2764379b7)
(main `c9406581` plus the author's community-only commits). No fork is linked:
the author took the parts of our former upstream pull request
[symbolica-dev/symbolica#163](https://github.com/symbolica-dev/symbolica/pull/163)
that pyAmpliCol needs (`DualNumber` component views, complex-modulus Dual
derivatives, normalized polynomial factors: signs, a single constant first,
sorted factors) and his own state-omitting evaluator saves (`c9406581`);
pyAmpliCol, NinjaRS, oneloopmaster and TIRust evaluate row by row instead of
the PR's batched evaluator. Three commits of the PR are not upstream and are
not needed by pyAmpliCol: the polynomial exponent bound (only TIRust's own
oversized-exponent test reaches it), the journaled conditional cache of the
direct evaluator translator (TIRust's rescue preparation; it changes no
generated instruction) and the precision of native Python `ComplexFloat`
operands (Python only; pyAmpliCol does no such arithmetic). The Python
Community modules (FeynKit, HepKit) need the `community` branch's
`symbolica::utils::resources`, so `main` alone is not the target; the pin
moves to a crates.io release once one contains this revision. Numerica 3.1
moved `conj` into its numeric trait, which GammaLoop's Spenso implements:
FeynKit and Spenso come from the fork ValentinHirschi/gammaloop
`pyamplicol-community` `ea3faaae` (alphal00p/gammaloop `d1904bf2` plus that
one change), the revision NinjaRS pins. NinjaRS (main
`be6d3e04`, the merge of its `symbolica-community` branch: main `b51b7465`'s
crate sources, the per-row evaluation without the
batch evaluator, pinned on this Symbolica, the gammaloop fork and
oneloopmaster `2e7afabb`) pins the same revision, as does oneloopmaster
(main `2e7afabb`: `3b6f229f`'s sources) in its
own standalone `[patch]`, which only applies when it is the root workspace.
`cargo tree -i symbolica` lists one source, as do `numerica`, `graphica`,
`oneloop` and `spenso`. SymJIT stays 2.28.3. These pins are recorded by Cargo.
This is a source integration, with Rust 1.96 or newer required; the git
`[patch]` still fails the published-crate gate, so the branch is not
publishable until this revision is in a Symbolica release.

TIRust ([huashengshao/TIRust](https://github.com/huashengshao/TIRust), crates
`iregi-rust` and `iregi-masters`, pinned to a commit of its main after its one
history rewrite, currently main `a5d4afbf`: PR #35 and the move to Symbolica
community `0e2ad04c` with per-row evaluation instead of the batch evaluator) is one more git source, linked by the cargo feature `tir`, a
default feature of every crate that links it (`--no-default-features` builds
without it). Its crates require `symbolica = "3.0"` and the oneloop revision
and features pinned here, so the root `[patch]` and oneloop pin serve them
(one source each, unchanged feature sets). Like the NinjaRS and oneloopmaster
pins it is a git source for the publication gate.

The published Python/Community distribution is pinned to **3.0.3**, which the
author builds from the `community` branch. Its `Evaluator.save()` writes
envelope version 1 (the native-code flag, always a Symbolica state, then the
bincode payload); the Community build of this revision writes version 2
(`c9406581`: a flag whether a state follows, and a state only for an
evaluator with symbolic data, i.e. non-built-in functions or tags). The native
reads both. 3.0.2's raw-bincode saves are refused: Symbolica 3.0.2 changed the
saved-evaluator and expression export formats, so the serialization ABI is
`symbolica-bincode2-v2`.
Both producers save the process's whole Symbolica state when they save one:
after a loop_sm compile about 20-30 MB per evaluator, depending on every
symbol the Python process ever created. Stored as such, a u u~ > d d~ g loop
artifact took 4.7 GB instead of 5.5 MB and 29 s instead of 0.3 s to load. And
a producer numbers the symbols it references by its registry, so its bytes
depended on the order in which the process created namespaces and symbols (a
cold or a warm model cache gave different `parameters.bin`). Every generation
writer therefore stores `symbolica_helpers.saved_evaluator`, the native's
canonical form (`rusticol-core` `canonical_saved_evaluator`), always envelope
version 1, which every Python Symbolica since 3.0.3 loads (PyPI 3.0.3 does
not read version 2): an evaluator without symbolic data (Symbolica's
`uses_symbolic_state`) keeps an empty partial state; otherwise only the
symbols the evaluator references (cut from a whole state by
`export_referenced_symbols`), their records ordered by name and numbered 0,
1, ...; the payload is kept byte for byte (it names its symbols by definition,
so the numbering only maps the state), and the result must decode to the same
evaluator. The stored bytes then depend on the evaluator alone, not on the
producer nor on the process history. A section whose symbol data embeds an expression keeps its
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
(symbolica-community `39b556c7`, the 3.0.3 recipe) with its `[patch]` moved to
`0e2ad04c` and its gammaloop `feynkit` crates (`9102f885`) on the same Spenso
`conj` change (tree-identical to ValentinHirschi/gammaloop `tirust-community`
`6ef9ab3a`): `builds/community-release-python-0e2ad04c` (wheel
`builds/community-release-wheels-0e2ad04c/`, sha256 `0754ed01…`; recipe in
`BUILD-INFO.json`). Native and Python therefore share one Symbolica revision.

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
from `0e2ad04c` (the `community` revision above) through the root `Cargo.toml`'s
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
