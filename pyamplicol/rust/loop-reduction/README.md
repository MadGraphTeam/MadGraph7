# NinjaRS consumer boundary

This standalone Rust crate exercises the actual reducer and scalar master
provider before they are integrated into the application's shared core. It
does not generate physical amplitudes or implement a second reduction method.

`PolynomialReducer<T>` binds lane-major polynomial parameters and carries
`Complex<T>` through NinjaRS, Symbolica and the master provider. `T` can be
`f64`, `DoubleFloat`, or arbitrary-precision `Float`. Its fifth polynomial
variable is explicit μ², with weighted rank two. `FastReducer<N>` accepts
NinjaRS's tensor numerator or its generic polynomial adapter in binary64.
All numerical stages use the upstream batch interfaces; this does not claim
hardware SIMD for every stage or arithmetic mode.

Both adapters replace results on each call, retain caller lane IDs, and return
no numerical value for failed lanes. A structural error returns `Err` for the
whole call. Parameter rows must follow the supplied lane order, including
compacted retry batches. The adapters do not perform precision rescue or
convert inputs through binary64.

`Laurent` names the finite, single-pole and double-pole entries. Its epsilon
convolution combines the separately reduced numerator sectors N0 + εN1 + ε²N2
through finite order. All values retain Ninja/OneLOop normalization:
μ_R^(2ε)/(i π^(2−ε) rΓ). Model prefactors and the public interference
normalization have not been applied. In particular, the reducer's rational
diagnostic is already included in the finite coefficient.

## Dependency setup and checks

Dependencies use ordinary pinned Git revisions and Cargo's lockfile. No ignored
reference directory or `upstream` symlink is needed. The standalone workspace
aligns Symbolica with the application's shared core and uses:

- NinjaRS `cf28d8e0018e0555256d3fba51e457d4941dcb0e` (main: its scoped Symbolica
  library licence, a key of its own held only around NinjaRS operations, on
  5fe7f392, the merge of 71eddb15, the box mu-expansion's conditioning bound: typed refusal of near-planar lanes in every carrier, the
  binary64 engines deciding a flagged lane a posteriori, on
  main `98e223da` = main `b3a29c85`, PRs #7-#12 and #14, the 0BSD relicence,
  and the re-pins on the Symbolica and oneloopmaster revisions below; UV/IR
  pole selection through
  the master providers, the fast engine's UV residues `fast::tensor_uv`,
  threshold-free light-like spinors).
- Symbolica, Numerica and Graphica `dfe6bd97b974b1028dc63071ea165136cd2b0ef5`:
  symbolica-dev/symbolica `community` 36cbac3a plus the consolidated upstream
  PR symbolica-dev/symbolica#163 (batched evaluation, `DualNumber` views,
  state-minimal evaluator saves, complex-modulus Dual derivatives,
  deterministic integer factor signs, polynomial exponents bounded by the
  i32 conversion range, a journaled conditional cache in the direct evaluator
  translator, native Python ComplexFloat operands at the more precise
  component's precision), branch `loop_me_pyamplicol` of
  `https://github.com/ValentinHirschi/symbolica` (see `dependencies/README.md`).
- TIRust `bc94b496e94d7b7686fc28adbfe9b42a68200d02` (its main: the merge of
  PR #31, triangles get the loop-rotation test on the binary64
  `TensorWorkspace` facade; the `DirectWorkspace` and `TensorStore` this
  crate calls are unchanged), behind the cargo feature `tir` (below).
- FeynKit `d1904bf2bd5bf4f26969a73b415095e941a55610`, the owner's coherent
  dependency revision recorded by NinjaRS.
- oneloopmaster `bd703d09a78f7849a6b86a07096e6f222fc01193` (its main: per-evaluator
  `PoleSelection`, TIRust's oneloop patches incl. #17/#18, the 0BSD relicence,
  #19's order-independent massless triangle (trif0) with a hand-written
  binary64/double-double kernel, and #20's exact bubble reduction of a C0 at a
  vanishing Källén function on the arbitrary-precision routes), with one
  feature set in every root build:
  `generated-evaluators` and `rust-codegen` (the latter only adds modules).

From this crate's directory, using the shared workspace environment:

```sh
export CARGO_TARGET_DIR="$PYLOOP_WORKSPACE/builds/loop-reduction"
export RUST_MIN_STACK=134217728
cargo test --locked --test anchors -- --test-threads=1
```

Supply a valid `SYMBOLICA_LICENSE` in the process environment when needed;
never persist it in this crate. Cargo's lockfile covers Git and registry
dependencies. Upstream sources remain read-only to this consumer. Generic
reduction accepts more than six propagators in every carrier. Source construction
delegates weighted-rank admission to the selected reducer, which supports n+2;
the profiling adapter forwards its rank-n+2 interfaces. This does not certify
every higher-rank physical model. Explicit-μ² polynomial containers do not
invoke the owner's optional graph-level FDH numerator transformation.

The owner revision adds native B1/B11/B111 and B1111 master families, along with
real-mass box relabelling and continuation repairs. Its report still records
conditioning and continuation limitations: ill-conditioned high-rank binary64
reductions, small-invariant rank-four bubbles, near-Gram boxes and a wide-complex-
mass box discrepancy requiring an independent reference. See the pinned owner's
`docs/rank-n2.md` and `docs/benchmark/sections.typ`. `success` therefore means
upstream computation completed, not that an accuracy target was established.
Cross-precision agreement and independent physical controls remain necessary;
removing a coordinator multiplicity limit is not universal master certification.

The AP80 zero-width box boundary is resolved on these pins: through Symbolica
e679e8c1 numerically equal sqrt-square and decimal-reparsed masses yielded
opposite finite imaginary signs, both with `success` status; Symbolica 9cdc2949
(and upstream since 3.0.2, `Float::unit_sign`) makes both take the reparsed branch (a `Float` product with an exact ±1 no
longer rounds the one extra bit of a square root away). No numerical snapping or
local master replacement is used.

The Rust anchors (`cargo test --locked --test anchors`) and the v2 reducer
tests (`runtime_v2_reduce`) are the consumer's checks on these pins. The former
Python `PolynomialPlan` binding, its rank-n+2 and any-n anchors
(`python/integration_*.py`) and the saved six-fermion plan checks belonged to
the removed legacy one-loop engine (MQ-LEGACY-REMOVE, 2026-10-29); the frozen
legacy family reductions are compared with `V2Reducer` by
`tests/unit/test_loop_v2_reduction.py`.

The analytic tests exercise A0 with real and complex squared masses, massless
B0, the divergent massless C0, μ² C0, and a four-dimensional q² numerator
whose nonzero rational term requires auxiliary μ² expansions,
all three arithmetic carriers, parameter rebinding, compact and empty batches,
retained input bits beyond binary64, replacement semantics, failed lanes,
and epsilon-sector ordering. They establish only these anchors, not the full
scalar-integral domain or process-level correctness.

`cargo test --locked --test diphoton_fixture` connects the independent Community
dimensional-algebra fixture in `tests/fixtures/diphoton_r2.json` to the
reducer. It preserves rational inputs and verifies the evanescent remainder in
all three arithmetic modes, including changing batch sizes and epsilon-sector
assembly. This internal fixture does not enable loop-induced production.

## TIRust (cargo feature `tir`)

TIRust (`https://github.com/huashengshao/TIRust`, crates `iregi-rust` and
`iregi-masters`), the tensor-integral reduction backend, is an optional
dependency pinned to a commit of its main and enabled by the cargo feature
`tir`, a default feature of this crate and of every crate that links it
(`rusticol-loop-python`, `rusticol-loop-sdk`, `rusticol-python`,
`rusticol-capi`): developer builds, tests, staged natives, CI, wheels and
releases have it. TIRust uses this workspace's Symbolica and oneloop sources
and feature sets, so the feature changes no NinjaRS value.
`--no-default-features` builds without it (`build_loop_native.py build
--no-tir`; the CI `tir` job checks that configuration and compares
`tests/tir_feature.rs`'s NinjaRS lanes bit for bit across the two).
`rusticol_loop_reduction::tir::available()` (Python:
`pyamplicol._rusticol._loop.tirust_available()`) reports whether a build links
it.

## Native Python bindings

The `python` member builds `_loop_reduction`, a developer-only PyO3 extension
whose classes are also registered in the ordinary pyAmpliCol extension
(`register`). It links the real shared core and reuses its production
`Evaluator.save()` decoder. With the legacy one-loop engine removed
(MQ-LEGACY-REMOVE, 2026-10-29) it holds the engine v2 bindings and the shared
numerical transport:

- `NativeBatch` / `ReductionBatch`: immutable typed lane-major complex values
  with explicit row IDs (`from_decimal`, `to_decimal`, `take_rows`,
  `take_columns`, `concat_columns`, `nonfinite_rows`) in `double` (53 bits),
  `double-double` (106-bit DoubleFloat) or `arbitrary` (MPFR Float; only it
  takes `precision` in decimal digits). Decimal strings are needed only at
  external ingress/egress. Failed lanes carry no value.
- `CoefficientEvaluator`: a saved exact evaluator mapped to the chosen carrier
  through Symbolica's instruction-batched evaluator (GIL released).
- Engine v2 (`docs/development/loop_v2/DESIGN.md` §11-§12): `V2Reducer`,
  `V2Executor`, `V2Program`, `V2PlanModel`, `V2Dag`, `V2Generation`,
  `SeedBuilder`, `V2SeedFourier`, `NativeCellEngine`/`NativeCellTable`, the
  colour orbits (`NativeAdjointOrbits`, `NativeDipoleOrbits`) and the plan
  readers.

```sh
cargo build --locked -p rusticol-loop-python
mkdir -p "$PYLOOP_WORKSPACE/builds/loop-python-module"
cp "$CARGO_TARGET_DIR/debug/lib_loop_reduction.so" \
  "$PYLOOP_WORKSPACE/builds/loop-python-module/_loop_reduction.so.new"
mv "$PYLOOP_WORKSPACE/builds/loop-python-module/_loop_reduction.so.new" \
  "$PYLOOP_WORKSPACE/builds/loop-python-module/_loop_reduction.so"
PYTHONPATH="$PYLOOP_WORKSPACE/builds/loop-python-module" \
  python -m pytest python -q
```

`runtime::V2Reducer` (Python `V2Reducer`) is the loop engine v2 adapter
(`docs/development/loop_v2/DESIGN.md` §11.1–11.2). Each family group passes
dense four-dimensional tensor coefficients in NinjaRS's layout
(`v2_tensor_monomials(rank)`) with shifts, masses, scales and the exact
invariant matrix. Lanes of all groups are bucketed by (propagators, true
rank) and reduced by upstream `TensorNumerator`s, with masters behind a
canonicalising cache (`runtime::MasterCache`), or by the generic engine from
one exponent-array template per bucket in DoubleFloat/arbitrary precision.
`check=True` re-reduces every lane in a dihedrally relabelled frame (default:
loop direction reversed) with the same coefficients and cached masters and
returns both results; `poles="none"` returns the finite column only and
`poles="split"` adds each lane's UV residue (`"all"` is the default; the
binding refuses any other value, a bool included). Families
must be in loop-cycle order.
The main workspace excludes this experimental workspace and has no NinjaRS path
dependency, so its normal builds do not require the `upstream` symlink.

## Remaining gates

- Higher-multiplicity physical controls and complex-master accuracy still need
  process-level validation despite the generic coordinator's new any-n support.
- The selected NinjaRS constructor owns weighted-rank admission (the current
  revision supports its n+2 domain). Saved layouts retain every term and may not
  truncate to fit a backend; this does not certify arbitrary higher-rank
  physical models.
- The binary64 native master provider fixes its on-shell snapping threshold
  at 1e-9. The generic constructor requires an explicit threshold policy;
  `None` disables snapping. These policies can differ for small invariants.
- Broad B0/C0/D0 analytic continuation, threshold and complex-mass validation
  remain upstream readiness gates. Passing A0 and μ² C0 is insufficient.
  The AP80 real-mass box branch defect of Symbolica e679e8c1 is resolved by
  9cdc2949 and upstream 3.0.2 (see above). This consumer does not normalize precision metadata
  or substitute a scalar integral.
- Native batching is established by the called interfaces, but end-to-end
  throughput, SIMD coverage and the MadLoop/Recola/OpenLoops performance
  targets require physical process benchmarks after integration.
- Explicit μ² reduction is only a numerical prerequisite for direct R2.
  Correct regularized numerator generation, epsilon sectors, UV terms and
  scheme alignment are separate requirements.
- Dependency licensing and packaging, including oneloopmaster's unpublished
  development status, must be resolved before distributing this stack.

## Native carrier transport

`NativeBatch.convert_arithmetic(arithmetic, precision=...)` converts existing
values directly through Symbolica's numerical carriers and preserves lane IDs.
It does not recover lost input information: a higher-precision rescue must rerun
the original exact inputs. Promotion alone is never accuracy evidence. A value
outside the requested carrier's range becomes nonfinite and must be rejected by
the caller's ordinary finite-lane guard.

`NativeBatch.merge_rows(original_ids, batches)` combines disjoint successful
groups in the original ID order, retaining only present rows. Missing lanes
remain absent, with no zero or stale-value placeholder. Groups must already use
the same requested output carrier and column layout.

These primitives add no second precision pass to ordinary evaluations. The
stability rescue is the Python API's (`loops/v2/rescue.py`); the native
agreement check and the SDK verification policy of the removed legacy runtime
are gone (MF-CLEAN).

### Removed: the plan v3 runtime

The legacy engine's plan v3 runtime (the shared-core `LoopExecutor` with the
`PhysicalBackend`, schedules, dual Born weights and state batches, the
`VirtualRuntime` artifact handle and the Python `NativeLoopProgram`,
`PortableLoopPlan`, `NativeVirtualRuntime`, `CoefficientPlan` and
`PolynomialPlan` bindings) was removed with the legacy one-loop engine
(MQ-LEGACY-REMOVE, 2026-10-29). Plan v4 artifacts run through
`v2_runtime` / `V2Executor`; a plan v3 artifact is refused with a typed
compatibility error.
