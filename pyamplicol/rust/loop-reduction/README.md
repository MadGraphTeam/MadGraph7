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

The ignored `upstream` symlink points to a NinjaRS checkout with its own
dependency references populated. This temporary standalone layout keeps the
application's ordinary checkout buildable while upstream packaging is pending.
Its Symbolica version is aligned with the application's shared core.
The implementation workspace uses:

- NinjaRS `5f35b071bda8b10df0bd40db20f31c3bd75bcecc`.
- Symbolica `e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`, the owner's integration
  revision published on `https://github.com/ValentinHirschi/symbolica`.
- FeynKit `474478a7a80bc9f9f21bead0e1cde6acbcffbe93`, the owner's patched
  dependency revision recorded by NinjaRS (not asserted merged upstream).
- oneloopmaster `264a6ced74e3c7a5de000b93dddf1330826cd3e9`.

From this crate's directory, using the shared workspace environment:

```sh
ln -s "$PYLOOP_WORKSPACE/references/ninjars-consumer" upstream
export CARGO_TARGET_DIR="$PYLOOP_WORKSPACE/builds/loop-reduction"
export RUST_MIN_STACK=134217728
cargo test --locked --test anchors -- --test-threads=1
```

Supply a valid `SYMBOLICA_LICENSE` in the process environment when needed;
never persist it in this crate. Cargo's lockfile covers registry dependencies.
The upstream sources are read-only to this consumer. NinjaRS revision 5f35b07
still limits the generic integrated coordinator to six propagators and rank
n+1. Its separate binary64 fast engine supports higher multiplicity; that does
not extend double-double or arbitrary-precision coverage. The native Python
consumer retains a common six-propagator domain across its arithmetic modes.
The latest owner update changes scalar-master branches and limiting cases,
with no reduction callback API change. Registry dependency versions are
unchanged, so the existing Cargo lockfile remains valid.

The owner's final report is narrower than a universal accuracy guarantee.
At revision 5f35b07, `docs/benchmark/sections.typ:99-123` records two
ill-conditioned synthetic lanes whose native stability checks report success:
`syn-gghhh-ggHHH-n5-r5` has a finite-part error of about 3.4e-4, while
`syn-n6-r6` loses about 24% after cancellation by roughly 1e12. In the latter,
most excess error comes from a nearly coincident-root triangle master
(`sections.typ:113-120`); high-precision reduction supplied the reference.
Consequently this adapter's `success` means that upstream computation
completed, not that an accuracy target was established. Independent numerator
reconstruction checks, cross-precision comparisons and a validated rescue policy
remain required for process-level accuracy certification. The generic
six-propagator bound is enforced at `crates/ninjars/src/amplitude.rs:108-113`;
our direct Rust boundary tests retain it for all three scalar carriers.

The analytic tests exercise A0 with real and complex squared masses, massless
B0, the divergent massless C0, μ² C0, and a four-dimensional q² numerator
whose nonzero rational term requires auxiliary μ² expansions,
all three arithmetic carriers, parameter rebinding, compact and empty batches,
retained input bits beyond binary64, replacement semantics, failed lanes,
and epsilon-sector ordering. They establish only these anchors, not the full
scalar-integral domain or process-level correctness.

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
The main workspace excludes this experimental workspace and has no NinjaRS path
dependency, so its normal builds do not require the `upstream` symlink.

## Remaining gates

- The generic upstream coordinator accepts at most six propagators; selected
  seven-propagator high-multiplicity cases need owner support and validation.
- Rank is limited to n+1 by NinjaRS. No numerator term may be truncated to fit.
- The binary64 native master provider fixes its on-shell snapping threshold
  at 1e-9. The generic constructor requires an explicit threshold policy;
  `None` disables snapping. These policies can differ for small invariants.
- Broad B0/C0/D0 analytic continuation, threshold and complex-mass validation
  remain upstream readiness gates. Passing A0 and μ² C0 is insufficient.
- Native batching is established by the called interfaces, but end-to-end
  throughput, SIMD coverage and the MadLoop/Recola/OpenLoops performance
  targets require physical process benchmarks after integration.
- Explicit μ² reduction is only a numerical prerequisite for direct R2.
  Correct regularized numerator generation, epsilon sectors, UV terms and
  scheme alignment are separate requirements.
- Dependency licensing and packaging, including oneloopmaster's unpublished
  development status, must be resolved before distributing this stack.
