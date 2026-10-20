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

- NinjaRS `0d0d7e7a8033431b60a8580731084742e0655cda`.
- Symbolica `e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`, the owner's integration
  revision published on `https://github.com/alphal00p/symbolica`.
- FeynKit `1e8063a763d3aa515749c33616ee59aa827705be`, the owner's coherent
  dependency revision recorded by NinjaRS.
- oneloopmaster `09b2c7497743db875ed446da95123175dd00b1c8`.

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

The existing `tools/loop_reference/reproduce_cms_zero_box.py` still reproduces
the AP80 zero-width box boundary on this pin: numerically equal sqrt-square and
decimal-reparsed masses yield opposite finite imaginary signs, both with
`success` status. The coherent packaging update does not resolve that separate
CMS acceptance gate. No numerical snapping or local master replacement is used.

Focused consumer adoption checks pass on these pins: the seven existing analytic
and adapter anchors, profiling on/off for a screened rank-n+2 A0-cancellation
control, and the seven-line rank-eight C0 control below in generic double,
DoubleFloat, AP80 and binary64 fast-tensor evaluation. The existing saved complete
six-fermion QCD plan also passes all three carriers at three exact points/scales:
all Born/bare/UV/R2/full components agree with the independent binary64 references,
all 94 emitted family/order statuses succeed, and analytic poles, empty/permuted
batches and exact alpha_s updates pass. The DoubleFloat/AP80 update checks retain
their 3e-25/1e-60 cancellation-aware tolerances. These controls establish this
consumer's validated domain, not general high-rank or complex-mass accuracy.

The independent rank-four massless bubble control
`python/integration_rank_n2.py` checks the new bubble integration boundary in
generic double/DoubleFloat/AP80 and both fast adapters. With metric (+---),
`p=(0,1,0,0)` and null `n=(1,1,0,0)`, its numerator is `(q.n)^4`. Feynman
parameters give `I/B0=1/5+29*epsilon/300+O(epsilon²)`, fixing the finite rational
term as well as every pole. Exact scales, native batch identities, empty/tail
calls and authoritative rejection of rank n+3 are checked. Passing this algebraic
control removes the obsolete source n+1 ceiling without asserting general-model
physics certification.

The analytic tests exercise A0 with real and complex squared masses, massless
B0, the divergent massless C0, μ² C0, and a four-dimensional q² numerator
whose nonzero rational term requires auxiliary μ² expansions,
all three arithmetic carriers, parameter rebinding, compact and empty batches,
retained input bits beyond binary64, replacement semantics, failed lanes,
and epsilon-sector ordering. They establish only these anchors, not the full
scalar-integral domain or process-level correctness.

The separately invoked `python/integration_any_n.py` anchor has seven distinct
propagators and an explicit-μ² weighted-rank-eight numerator. Four denominator
factors cancel exactly to the analytic divergent massless C0; the numerical
reduction still uses the full seven-propagator input and actual scalar masters.
It checks changing scales, coefficients, lane IDs and empty/compact batches;
it does not certify non-cancelling box contributions or their analytic branches:

```sh
PYTHONPATH="$PYLOOP_WORKSPACE/builds/loop-python-module" \
  python -m pytest python/integration_any_n.py -q
```

This is deliberately outside default test collection. Its dense polynomial has
501 monomials, compared with 60 in the owner's sparse rank-eight `any_n.rs`
fixture. In an earlier e4c16a7 run, the generic binary64 test, including plan
construction, first-use setup and all batch checks, took 620 seconds; the dense
fast-tensor check took 0.11 seconds. The DoubleFloat and 80-digit generic tests
both passed in a combined 1,313 seconds, with absolute Laurent-entry tolerances
of 1e-22 and 1e-60 respectively. The owner's
`generic_scaling.rs` distinguishes reduction-kernel setup from the polynomial
adapter's steep first-use Laurent-expansion preparation. We have not sampled
the active native stack to assign the complete elapsed time between these
stages. The generic adapter exposes no public prepared-expansion export or
cross-carrier rebind: `exact_evaluator` and `exact_expansion` are crate-private,
and its expansion cache is private. Retained physical instructions therefore
still rebuild this dependency-owned setup in each fresh generic carrier. The
consumer does not duplicate the owner's polynomial or truncated-jet algebra.

`cargo test --locked --test diphoton_fixture` connects the independent Community
dimensional-algebra fixture in `tools/loop_reference/diphoton_r2.json` to the
reducer. It preserves rational inputs and verifies the evanescent remainder in
all three arithmetic modes, including changing batch sizes and epsilon-sector
assembly. This internal fixture does not enable loop-induced production.

## Experimental native Python consumer

The `python` member builds `_loop_reduction`, a developer-only PyO3 extension.
It links the real shared core and reuses its production `Evaluator.save()`
decoder. `CoefficientEvaluator` maps exact coefficients to the chosen carrier
and calls Symbolica's instruction-batched evaluator. `PolynomialPlan` reduces
three explicit numerator epsilon sectors and combines their Laurent values
inside Rust. Both release the Python GIL during batch evaluation.

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

The explicit `arithmetic` selector accepts `double` (53 bits), `double-double`
(true 106-bit DoubleFloat), or `arbitrary` (MPFR-backed Float). Only arbitrary
arithmetic accepts the required `precision` in decimal digits; it also accepts
16 or 32 digits without silently selecting either fixed-precision carrier. Decimal
input pairs avoid binary64 conversion; explicitly sized decimal output retains enough
digits to round-trip each carrier. Buffers are flattened lane-major complex
pairs, with four pairs per momentum and one original ID per lane. The three
returned entries are **finite, single pole, double pole** in raw integral
normalization. Failed lanes carry `None` instead of numerical values.

`massless_spinors_decimal` reuses the shared core's generic tree massless Dirac
wavefunctions. Its rows contain positive on-shell physical momenta, helicities
±1, and kinds `u`, `v`, `u_bar`, or `v_bar`; rows can mix kinds and helicities.
Chiral adjoints map the existing core states to these names. Execution is one
native bulk call, with no claim of hardware SIMD for these scalar routines.

`wavefunctions_batch(momenta, helicities, kinds)` calls the pinned FeynKit
precision-generic wavefunction API for real four-momenta. It supports `u`,
`u_bar`, `v`, `v_bar`, `epsilon`, `epsilon_bar`, and `scalar`. It preserves the
supplied momentum; FeynKit infers mass from its energy and spatial components.
The caller must separately validate consistency with a declared model mass.
`on_shell_wavefunctions_batch(spatial_momenta, masses, helicities, kinds)` is
an explicit construction interface using FeynKit's existing on-shell operation;
it returns both the resulting four-momenta and states. Masses are real and
nonnegative. Both calls preserve row IDs and all three arithmetic carriers.
Scalar rows have one component and vector/spinor rows have four; a batch must
have a common width. Optional `components` validates this width, including empty
scalar batches (the empty default is four). Fermion rest states are supported;
longitudinal vectors at rest remain an explicit dependency-domain error. No
unstable external particle or complex external mass prescription is implied.
Tests verify the calibrated massless convention, massive spin sums using
dependency gamma matrices, vector transversality/completeness, and mass bits
beyond binary64. Numerical execution is a native block of dependency routines.

`wavefunctions_partial_batch` uses the same dependency operation and returns
`(successful_native_rows, statuses)`, with a status for every original row ID.
Numerical state failures, including input or intermediate carrier overflow, have
no placeholder values. Invalid shapes, kinds and helicity metadata still raise.
FeynKit's checked wavefunction constructor rejects nonfinite components; this
adapter retains that existing guarantee. The ordinary raising interface is
unchanged. Process-level compaction must remove every helicity/leg row belonging
to a failed point before constructing currents. Recovery must reparse original
inputs in the rescue arithmetic; promoting an overflowed buffer cannot recover
them. This primitive does not establish numerical accuracy or enable rescue.

`NativeBatch` keeps immutable typed values with explicit row IDs. It supports
column selection/concatenation, row compaction/broadcasting, and row regrouping;
new IDs must be explicit when broadcasting or regrouping. Concatenation and
reduction reject mismatched IDs, carriers, or working precision. Decimal strings
are needed only at external ingress/egress, not between numerical stages.
`nonfinite_rows()` checks both components directly in the native carrier and
returns original row IDs, allowing physical assembly to reject nonfinite Born,
counterterm or contraction values without decimal conversion.

`CoefficientPlan` consumes saved local evaluators and integer input slot maps.
Its arena is `[global parameters | node 0 outputs | node 1 outputs | ...]`;
forward references and malformed maps fail at construction. Node evaluators
execute all lanes through Symbolica's native batch interface with reusable
arena/gather/output storage. Selected final slots become a typed `NativeBatch`
that feeds `PolynomialPlan.evaluate_batch` directly. The reducer also accepts
an optional lane-major n×n exact S-matrix of `(V_i−V_j)²`, which must retain
prescribed on-shell invariants for arbitrary-precision master classification.
The caller must derive these invariants exactly before rounding, using existing
model/kinematic algebra and exact input evaluation; subtracting already-rounded
shift vectors and passing that result does not establish exact invariants.
`ReductionBatch.statuses` covers every input row; `.values` is available when
all lanes succeed, while `.successful_values()` retains only successful IDs.

The Python producer can retain a frozen `PreparedCoefficientDAG`: its separate
`CoefficientInstructions` contains dependency bytecode and integer slot maps,
while exact symbols, polynomial labels and domain metadata retain the existing
sharing identity. `NativeCoefficientDAG.rebind(...)` uses this representation to
create a fresh native workspace in another arithmetic without revisiting the
producer DAG, optimizing expressions or saving instructions again. Numeric
buffers and evaluator workspaces are independent between views. This is an
internal in-memory boundary, not an artifact format or Python graph persistence.

Fast-engine timing is optional: `PolynomialPlan.set_profiling(True)` enables
native wall clocks, and `take_profile()` returns and resets accumulated totals
over the active epsilon sectors. Both `fast` and `fast-tensor` are supported;
generic arithmetic explicitly rejects this instrumentation. Profiling is off
by default, and that execution branch calls no clocks or timing wrappers.
Toggling the setting also resets previous totals. Delegating wrappers preserve
the original numerator, scalar-master results and errors, including failed
calls; no cache or reduction algorithm is added.

The raw seconds are `engine_seconds`, `masters_seconds`,
`numerator_callback_seconds`, and `numerator_preparation_seconds`. The first
includes the next two; preparation (parameter binding or dense coefficient-head
evaluation/scatter) occurs before the engine. `reduction_exclusive_seconds`
subtracts both masters and numerator callbacks from engine time, whereas
`engine_excluding_masters_seconds` still includes callbacks. Master time covers
the native oneloopmaster provider and its adapter, not arbitrary-precision or
external-library providers. Counts report engine calls, master calls/rows and
numerator calls/rows; rows count actual callback batch rows, not unique physical
points. A pure loop-numerator timing must add physical coefficient-DAG time to
callback and preparation time. Born currents, UV/R2 effective rules,
postprocessing and Python/native transport are separate work. Instrumentation
overhead remains in these wall-clock samples, so uninstrumented complete-call
measurements must be reported separately from this diagnostic breakdown.
`CoefficientEvaluator.evaluate_batch` and `massless_spinors_batch` share these
buffers, allowing couplings, wavefunctions and final contractions to stay native.
`CoefficientPlan(..., engine="jit")` optionally compiles the same saved exact
node evaluators using Symbolica's existing complex SymJIT backend and native
matrix evaluation. Internal threading is disabled and dependency SIMD dispatch
is enabled where supported by the host and kernel. It requires double arithmetic; DoubleFloat and arbitrary
precision retain the default `engine="interpreted"`. JIT construction failures
identify the node and propagate an explicit error, without silently changing
the engine. The original evaluator bytes and slot maps remain the kernel
identity; this adds no persistent cache or artifact format. Empty batches are
handled before the dependency matrix API. Constant-only nodes and changing
batch tails are checked against the interpreter.
For four or more propagators the generic consumer invokes NinjaRS's existing
`SymbolicaSamplingScales` once per batch and supplies its auxiliary scales to
each active epsilon sector. These are distinct from the renormalization scale
and the exact S-matrix. Failed scale-selection lanes retain their original IDs
and expose no values. The analytic `I4[mu^4] = -1/6` regression covers complex
masses, changing batch sizes and coefficient bits beyond binary64 in all three
arithmetic modes.

`ExactPolynomialEvaluator` reuses the shared saved-evaluator decoder but accepts
only straight-line rational polynomial instructions. Decimal inputs are parsed
exactly into Numerica integers/rationals, evaluated in Symbolica's existing
rational-ring evaluator, then converted once to the requested carrier. This
preserves both prescribed zeros and small nonzero invariants that would disappear
if momenta were rounded first. Exact preprocessing runs one native row block
using the dependency's scalar ring evaluator; it is not instruction-vectorized.
The caller remains responsible for exact source expressions and symbolic physical
assumptions before saving the evaluator. Mathematical constants, transcendental
functions, negative/noninteger powers, calls and control flow are rejected.

This is a native batch conformance boundary, not a public virtual-amplitude
API or a completed fast evaluation pipeline. Production graph/model scheduling,
physical normalization and precision rescue remain integration work. Both engines
skip symbolically zero epsilon sectors during evaluation; an all-zero plan
still executes one sector to retain the upstream input validation.
`PolynomialPlan(..., engine="fast")` selects the existing NinjaRS binary64 fast
coordinator and native master provider, using its generic polynomial callback
adapter (including explicit μ²). It accepts only `arithmetic="double"`, real
shifts/scales/invariant matrices, and the same complex squared masses supported
by the upstream fast input contract. Its fixed 1e-9 master threshold is distinct
from the default generic route's exact-zero policy; the fast route is opt-in.
Throughput tuning and physical performance gates remain open.
`engine="fast-tensor"` is an additive binary64 alternative using the same fast
coordinator/master provider. The owner's `PolynomialCoefficientPlan` evaluates
coefficient heads in a native batch, `TensorCoefficientPlan` scatters every
occupied and zero slot, and `MuTensorNumerator` evaluates the weighted μ² blocks
and all cut expansions. Work buffers are retained across calls. No polynomial
or tensor ordering is reimplemented here. Three explicit epsilon sectors still
use the existing Laurent convolution; this engine does not supply a scheme
conversion or R2 rules. Unsupported rank/layout is an error, with no truncation
or implicit fallback. The callback engine remains the explicit baseline.
The dependency coefficient-head evaluator rejects nonfinite input/output for
the whole batch; this engine has no new partial coefficient-evaluation API and
is not certified for the optional rescue wrapper. Upstream reduction failures
still retain original lane IDs and hide their numerical values.
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
returns both results; `poles=False` returns the finite column only. Families
must be in loop-cycle order.
The main workspace excludes this experimental workspace and has no NinjaRS path
dependency, so its normal builds do not require the `upstream` symlink.

## Remaining gates

- Higher-multiplicity physical controls and complex-master accuracy still need
  process-level validation despite the generic coordinator's new any-n support.
- The selected NinjaRS constructor owns weighted-rank admission (the current
  revision supports its n+2 domain). Saved layouts retain every term and may not
  truncate to fit a backend. The rank-four bubble is an independent normalized
  control; it does not certify arbitrary higher-rank physical models.
- The binary64 native master provider fixes its on-shell snapping threshold
  at 1e-9. The generic constructor requires an explicit threshold policy;
  `None` disables snapping. These policies can differ for small invariants.
- Broad B0/C0/D0 analytic continuation, threshold and complex-mass validation
  remain upstream readiness gates. Passing A0 and μ² C0 is insufficient.
  The AP80 real-mass box reproducer in
  `tools/loop_reference/reproduce_cms_zero_box.py` still selects opposite
  imaginary branches for numerically equal sqrt-square and reparsed masses
  at b5b93be; e4c16a7 does not modify the scalar master provider. This consumer
  does not normalize precision metadata or substitute
  a scalar integral to hide that unresolved owner issue.
- Native batching is established by the called interfaces, but end-to-end
  throughput, SIMD coverage and the MadLoop/Recola/OpenLoops performance
  targets require physical process benchmarks after integration.
- Explicit μ² reduction is only a numerical prerequisite for direct R2.
  Correct regularized numerator generation, epsilon sectors, UV terms and
  scheme alignment are separate requirements.
- Dependency licensing and packaging, including oneloopmaster's unpublished
  development status, must be resolved before distributing this stack.

## Native verification transport

`NativeBatch.convert_arithmetic(arithmetic, precision=...)` converts existing
values directly through Symbolica's numerical carriers and preserves lane IDs.
It does not recover lost input information: a higher-precision rescue must rerun
the original exact inputs. Promotion alone is never accuracy evidence. A value
outside the requested carrier's range becomes nonfinite and must be rejected by
the caller's ordinary finite-lane guard.

`agreement_rows(other, relative_tolerance, absolute_tolerance)` compares every
real and imaginary component with
`|a-b| <= absolute_tolerance + relative_tolerance*max(|a|,|b|)`. The comparison
uses the dependency's exact rational images of the represented numbers and
decimal tolerances. This preserves sparse DoubleFloat low words and avoids
binary64 complex-norm overflow or underflow. Nonfinite values always
disagree. The method requires identical row IDs and column counts and returns
only per-lane Boolean evidence; it does not claim independent reducer accuracy.

`NativeBatch.merge_rows(original_ids, batches)` combines disjoint successful
groups in the original ID order, retaining only present rows. Missing lanes
remain absent, with no zero or stale-value placeholder. Groups must already use
the same requested output carrier and column layout. A higher-precision result
must be compared again after its final output conversion if the configured
accuracy bound is to apply to the returned numbers.

These primitives do not enable an automatic verification or rescue policy, nor
add a second precision pass to ordinary evaluations. Process-level verification and rescue policy remain separate integration work.

The internal physical programs now expose `evaluate_ordered_partial_native` for
built-in full-colour, four-dimensional model-rule controls. This opt-in boundary
returns only points whose Born currents, family reductions, UV/R2/field currents,
all ordered components and totals are finite and whose reductions succeeded.
`requested_point_ids` records the input order; `batch.point_ids` names only the
survivors. Structured `failures` retain each original point, stage and available
order/family status. Custom point IDs support compact retry batches. Any failed
order or family evicts the whole point, including already integrated families;
missing values are never replaced by zeros. The ordinary evaluation methods
still raise on a numerical failure.

This partial boundary includes native external-state preparation. Its strict
`NativeBatch.from_exact_decimal` ingress reuses the existing exact rational parser
and carrier conversion: malformed or nonfinite original literals raise before
conversion, while finite originals overflowing a carrier remain distinguishable
numerical failures. Original scale literals are checked before point eviction.
Native state statuses compact whole physical points before coefficient DAGs;
original decimal inputs remain available for wider-precision retries. Global
model formula errors still raise for the batch. The bounded real-mass source
charged-lepton neutral-current model-rule control also supports partial
evaluation; CMS, source external fields, formal colour and spin/directed source
views remain excluded. The bounded direct-R2 four-gluon control is described
below. The partial method itself neither establishes
accuracy nor chooses a rescue precision. Genuine singular Born/CT and later-family
`DegenerateGram` controls validate failure isolation in all three carriers;
finite original state overflow is isolated in binary64/DoubleFloat and finite
state/current preparation is recovered from those originals in arbitrary precision.

`NativeModelStabilityProgram` is an internal optional view over one such complete
model program. Its `evaluate_native(..., verify=False)` performs one native
attempt and retains Ninja failure diagnostics with accuracy explicitly not
established. `verify=True` compiles higher arithmetic views lazily from the same
cold graph/model/CT definitions and reevaluates the copied original decimal
momenta, scales and global model tuple. Verification compares every real and
imaginary part of every Born order, all three poles in all six virtual component
sets, and their totals. A requested-precision result agreeing with the first
higher evaluation is retained. A failed or disagreeing result requires two
successful higher evaluations to agree before rescue; only pending points
continue. Previous successful results remain in their original carriers during
comparison, including sparse DoubleFloat low words.

A rescued value is converted directly to the requested output carrier and must
still pass the configured bound against the higher result. Unattainable output
tolerances therefore remain failures. Diagnostics distinguish evaluation and
output arithmetic, retain original point IDs and reduction failures, and call
this evidence `empirical-cross-precision-agreement`. It is not an independent
reducer or physics accuracy certificate. Unrecovered points raise by default;
`allow_partial=True` explicitly exposes sparse successful buffers and failure
metadata. Verification is never enabled by an ordinary evaluation call.

A bounded near-forward uudd control exercises an actual rescue: binary64 and
DoubleFloat report a degenerate Gram cut, while 70- and 100-digit evaluations
agree and satisfy the requested binary64 output bound. This tests the recovery
mechanism, not general near-singular numerical reliability. A cheap production
stability policy, correlated/formal coverage and benchmark-matched verification
cost remain separate work.


Source-owned partial and verification calls capture one checked global snapshot
before dropping any point. Its original overrides and exact source defaults are
re-evaluated in each higher arithmetic using the identical cold scalar plan;
rounded model values are never widened into higher inputs. Higher views retain
that snapshot even if the live source state changes during evaluation. They
publish no update to either owner. Ordinary and CT scale views remove nonfinite
scale/active-diagnostic lanes before consuming their values, and gather native
pointwise model rows, physical states and already reduced families consistently.
Original point IDs survive every gather. Invalid original global parameters,
failed atomic global updates and global formula errors remain whole-call errors;
this boundary does not promise rescue of a global binding that cannot be formed
in the requested carrier. Existing CMS declaration metadata and declared complex
mass parameters exclude that unvalidated domain independently of numerical mass
values.


The source rescue control uses the complete real-mass Gmu charged-lepton process
with custom IDs, one ordinary scale and one exact finite `mu_r2=1e400`.
Double/DoubleFloat reject the latter lane; 70- and 100-digit evaluations agree
across every component and satisfy the final double output bound. A mid-call
source Gf update after capture changes the next call but cannot change any
attempt of the current call. A separate actual `G_UVt` source CT exercises finite
`mu_r2=1e-320`: its logarithm argument overflows in double/DoubleFloat, so the
source scale view removes that lane before consumption; arbitrary precision
retains both lanes. These are bounded empirical controls, not general EW
stability certification.


Optional partial/verification evaluation also admits the certified full-colour
QCD `gg -> gg` direct-R2 control. Each family evaluates N4 and Delta from the
same coefficient/kinematic lanes, records both native reduction statuses, and
then removes whole failed points from both outputs and all prior families.
Final N4 and rational sums share the same gather. Higher arithmetic views rebuild
complete local dimensional numerators and epsilon sectors from original decimal
inputs; the selected `direct` strategy stays fixed, with source R2 excluded and
UV/field terms retained. No model/direct switching occurs during rescue. Native
complete and stability results record the executed R2 strategy, and verification
rejects an attempt whose strategy differs from the captured choice. The dense
`fast-tensor` adapter, source direct R2, CMS and formal/correlated views remain
outside this partial boundary.

Ninja's repeated-propagator validation remains a whole-batch unsupported-domain
error. Such a topology is not silently converted into a numerical status or a
higher-precision success. In particular, the former exact-forward uudd sparse
fixture now encounters this guard with the current producer/reducer snapshot;
it does not establish lane-local failure handling for repeated propagators.
Distinct collinear box shifts exercise actual degenerate-Gram/unstable statuses
in all three carriers without duplicate denominators.

### Complete shared-core execution (internal)

The validated loop PACBIN descriptor can now bind
`rusticol_core::loop_execution::LoopExecutor` with
`runtime::physical::PhysicalBackend`. The core owns the complete ordinary batch
schedule; this crate supplies the existing numerical primitives. It preserves
all Born/order columns, finite/single/double poles, UV, R2, local UV and external
field components, and the executed R2 strategy. No PyO3 or Python callback is
needed by the standalone executor. The experimental Python `NativeLoopProgram`
wraps that same core executor; public artifact and SDK dispatch remain separate.

The current native admission is full-colour legacy/source model-R2 and the
certified legacy four-gluon direct-R2 descriptor. Source owners publish checked
immutable snapshots; each evaluation shares one revision across ordinary
coefficients, pointwise masses, local CTs and fields. Named exact updates and
fresh-carrier rebinds use original override strings. Correlations, formal colour
and partial/rescue execution remain explicit capability boundaries.
Family projectors use the interpreter; coefficient JIT is separately selectable
for double. Numerical success is not an independent accuracy certificate.

The explicit generated-fixture test `runtime_physical` runs a complete uudd
artifact in all three carriers and verifies the existing reducer-construction
boundary. Set `PYAMPLICOL_NATIVE_UUDD_PACBIN` to a current generated uudd container
and run that test with `--ignored`. The coordinator stores the sizable reducer
workspaces behind `Box`, so construction and execution pass the ordinary 2 MiB
Rust test-worker stack; no enlarged-stack setting is needed. Returned native
blocks remain immutable after later calls and arithmetic rebinding.

### Verified virtual artifacts

`virtual_runtime::VirtualRuntime::open` loads the ordinary process artifact through
the shared `VerifiedArtifact` loop lane, then runs the same complete coordinator.
`_loop_reduction.NativeVirtualRuntime` is its thin Python ownership adapter. This
additive handle preserves Born/order and all three Laurent columns; it never
converts a virtual evaluation into a scalar tree result. Exact complex parameter
updates are atomic, arithmetic rebinds retain the original exact overrides, and
`clear` drops process workspaces while retaining the global parameter owner and
its revision. Callers supply positive real `mu_r2` values explicitly.

This initial delivery boundary admits source-owned, full-colour, inclusive
model-rule R2 artifacts with explicit scheme declarations. Scheme declarations
are metadata, not a general model certification. Resolved results, correlations,
mapping transport and partial/rescue dispatch remain unsupported here. The native
consumer remains a development build until the NinjaRS owner supplies its
self-contained packaging revision; this adapter does not establish normal wheel
or standalone SDK installation readiness.
