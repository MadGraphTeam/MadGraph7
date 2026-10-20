# Isolated Float256 native-kernel study — execution held

This is a preparation-only research plan. Compilation, engine evaluation and
timing each require root's subsequent authorization and coordination. No
production source, ABI, environment switch, master provider, rescue threshold,
working precision or tensor recurrence changes belong to this study. The
existing lower-precision paths are out of scope.

## Three fixed arms in one test binary

Prepare one private copy of the corrected transport bridge, bound to every
source hash in the sealed `ninjars-mp-transport-audit-v1/proposed/` snapshot and
to production-ready.patch SHA256
`3863c862562f37d352c6c9463be4d1e77f24193653377ff7cbb765871d265b97`.
Verify it matches the corresponding corrected production source before copying;
record any later unrelated source change as a different baseline. Do not include
the separate tensor-allocation experiment.

Enable only the additional pinned ninjars-symbolica `native-kernels` Cargo
feature (which implies rust-codegen). Add a private, explicit per-MpState policy
constructor with three values:

1. `Interpreted`: unchanged `SymbolicaAmplitudeKernels::new` selection.
2. `NativeDenominators`: call `with_native_denominators()` on that factory.
3. `FullNative`: call `with_native_kernels()` on that factory.

The ordinary public constructors and all public ABI entry points must continue
to choose `Interpreted`. The selector is private experimental test code, not a
production environment flag. Keep separate context/state caches for each arm.
The policy is immutable during a context's lifetime; it must be applied when
each new (n,rank) amplitude entry is created. Record the actual factory backend
report for every shape. Keep the default Whole native batch policy and leave
rank-specialized corrections, pruning and any other opt-ins disabled.

One release test binary may run the inherited 18 ordinary tests once on the
unchanged default policy, followed by explicitly named ignored study tests.
These tests instantiate all three private policies directly. If extending the
ordinary suite to each policy needs a global/thread-local override, do not add
one: parameterize the relevant private fixture instead. Shape/error/reset and
late-entry-creation controls must explicitly exercise each native policy.

Use repository-local build/target/TMPDIR settings after sourcing the pinned
development environment. Preserve Cargo.lock and use --locked; record the
feature-resolution and compiler identities. All compilation products stay in
the experiment's ignored work directory, never the references or evidence tree.
Use a single serial numerical process under the shared Symbolica lock directory.
No build or numerical replay may overlap another agent's quiet timing window.

## Exact source scope and known limitation

Pinned NinjaRS is `0d0d7e7a8033431b60a8580731084742e0655cda`, oneloopmaster
`09b2c7497743db875ed446da95123175dd00b1c8`, Symbolica/numerica
`e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`. Constructors default to interpreted
execution; merely enabling the feature changes no selected backend.

NativeDenominators replaces T2/T3 denominator expansion bodies, including the
shared tadpole T3 body. FullNative additionally selects ordinary bubble source
projection, triangle-parent corrections of quotient degrees 0/1/2 and n+1
degree3, and triangle/bubble parent-axis tadpole projections. The rank n+2
triangle-bubble correction remains interpreted. Cuts/geometry, other reduction,
integration, accumulation, numerator callbacks and scalar masters remain the
same objects. Export support for LaurentDivision or TadpoleGeometryProbe is not
an installed factory backend and is not used here.

NativeFloat supports arbitrary-precision Float. Constants are prepared directly
at the explicit 256-bit prototype precision; input component values and tracked
precision are not padded. The type's default128 is not used by these explicit
constructors. Native workspace copies use set_from. Float arithmetic still
allocates as required; f64 zero-allocation tests are not evidence of Float256
allocation removal or a speedup.

The denominator/source projection bodies use emitted add/mul graphs. FullNative
corrections also use native complex inverses: the native finite non-axis inverse
scales first, whereas interpreted Complex::inv normally uses direct norm-squared.
Thus full-native is an unchanged symbolic reduction plan with potentially
different numerical primitive ordering, tracked precision and signed zeros. It
must not be called an exact-operation dispatch substitution. No master or
on-shell threshold change is permitted to make it agree.

## Untimed gates and retained records

First run the inherited corrected-transport 18-test suite serially, preserving
all failures. Then consume exactly the retained 1,392 successful MP request
records in their original order, with all eight expected ASCII fields per
request. Bind the compressed/uncompressed input identities from the sealed
transport/capture provenance. Never rewrite historical expected fields.

For every request and arm retain input bytes, sequence/shape, policy/backend
report, status/error, all eight output fields, and the raw eight output
components as exact represented dyadics with tracked precision and zero sign.
The latter are observed from the amplitude result after evaluation, without
changing arithmetic or padding precision. Record the full finite/CC/rational
split separately where available. Retain errors and incomplete outputs before
asserting; do not let a single failure suppress later fixed requests. Verify
input immutability and complete 1,392-record/11,136-field coverage per arm.

Interpreted must reproduce the corrected transport baseline exactly in status
and ASCII fields (the sealed transport replay currently also equals the old
historical fields). Recheck selected reused-context records after changed
coefficients, mass, scale and shape against fresh same-policy contexts. Explicit
negative-zero, complex-mass, invalid-input/no-stale-output and late shape-entry
fixtures remain mandatory. The public malformed-buffer tests stay attached to
the unchanged default ABI; these are not replaced by private evaluator tests.

Native arms must preserve successful/finite status on every historical success.
Compare all three complete complex Laurent coefficients and separately report
the engine-basis rational diagnostic. Exact equality, zero-sign differences,
tracked precision changes and value differences are independent classifications.
No sign or precision mismatch is silently normalized away. Exact equality is a
recorded result, not assumed for the full-native reciprocal path.

Predeclared value screen: for each request let N be the maximum absolute real
or imaginary component among the six baseline Laurent components. Require
each native Laurent component difference <= 1e-31*N; if N is zero, require exact
zero. This is the existing binary128-output convergence scale, not a substitute
for the final generated-process physics gate. Rational fields are also compared
and diagnosed independently. Any status/nonfinite, sign, tracked-precision or
nonzero value difference triggers an explicit discrepancy record and held
interpretation pending the following convergence analysis; no failed request is
dropped and no threshold is weakened.

Repeat every discrepant request, plus the existing retained complex-mass
rank-five convergence fixture and analytic tadpole/massless/spacelike bubble
pole controls, at both 256 and 384 working bits for all three policies. Inputs
must remain the same recovered113-bit binary128 values, simply widened; never
replace them with newly rounded higher-precision kinematics. Use the same
1e-31 common-norm convergence bound for full Laurent fields and retain all
384-bit cross-arm differences. Exact expected tadpole/bubble poles provide an
independent algebraic check; finite=CC+rational identity remains checked within
each policy. A field passing a tolerance does not explain an unexplored sign or
precision change. Escalate nonconvergence or altered branch/failure behavior to
root; do not infer correctness solely because two native variants agree.

Upstream coverage is supporting evidence only: Float256 tests compare native
denominators/bubbles/tadpole projections to the interpreter at65 decimal digits,
test mixed53/128/512-bit input precisions with matching output tracked precision,
batch shape transactionality and exact regeneration of emitted source. These
are not proof of this complete bridge's bitwise equivalence or MadLoop physics.

## Timing remains unplanned and held

No kernel or full-process timing is authorized by this plan. After every
untimed gate is complete, report scope, differences and cost expectations first.
Root may then choose a separately preregistered fixed-sample timing study with
actual recorded native dispatch coverage and independent original-Ninja gates.
Compilation or warmup cost must not be mislabeled as warmed reduction cost.
