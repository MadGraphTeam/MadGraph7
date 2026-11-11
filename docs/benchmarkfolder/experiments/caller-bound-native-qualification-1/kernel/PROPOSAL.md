# General runtime certificate: private source proposal

This proposal contains no point, momentum-bit, loop-group, or fixture allowlist.
It is not a production patch, complete executable adapter, or accuracy claim.
`certificate.f90` is an uncompiled integer-only certificate kernel. The pure
Python specification/tests exercise its mathematical criterion, not the
Fortran implementation or a reducer. Caller-bound construction and exporter
integration below remain required before a private build. No threshold, model,
rescue policy, or installed library changes are proposed here.

The predecessor's exact-input-bit lookup was only a diagnostic causal
intervention. Its separate clarification is index9c890f63; the original
42a14babb7353e7cd1393e238521d3b339a2aff8748ed37f7f5d08c9261609bd
packet remains unchanged. Such a lookup must never become a deliverable fix.

## Evidence and immediate scope

Closed replay index3a8e13486d4cd26425efefaceb7d6f2099be0a867a80e22fdb00fc3ff1e678d8
records10 fresh processes using installed Fortran MTIRUST and the original
eb3d bridge: ordinary CMS point1 DP/QP succeed; point20/21 stock DP/QP refuse;
replacing only two symmetric supplied invariant entries with the captured raw
spacelike transfer makes those four calls succeed. This is evidence of that
input projection causing these refusals. No coefficient buffers were inspected;
success does not establish coefficient accuracy or a complete amplitude.

Current CMS routing comes from `derived/cms/routing.json`, hashca9ee436…,
authenticated by derived index25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184.
Its26 routes include modes1/2/4/5 and actual closure-residual chords.
`derive_masks.py` derives all26 structural masks in `cms-structural-masks.json`
from those routes, with no numerical point input; it asserts the original and
reversed corner/prefix identities. A private
prototype may use these source-derived structural tables for arbitrary
admissible events of that export. It may not single out group3 or points20/21.
Unknown routes remain on the legacy path. The generator must ultimately emit
the same information from actual HELAS calls for any supported export.

## Proposed per-call interface and routing

Extend a private shared-helper entrypoint, leaving its compatibility entrypoint
unannotated and unchanged:

```
BUILD_KINEMATIC_MATRIX_WITH_PROVENANCE(
  N, P_LOOP, M2L, PAIR_CLASS, Q_ERROR_EXP, HAVE_BOUNDS, ENVIRONMENT_VALID,
  S_MAT, DIRECT_INV, USE_DIRECT)
```

Supply a corresponding MP entrypoint. `PAIR_CLASS(N,N)` is structural;
`Q_ERROR_EXP(0:3,N,N)` bounds the error in each actual computed difference.
`HAVE_BOUNDS` is per ordered pair. `DIRECT_INV` is in the current real DP/QP
kind; `USE_DIRECT` is false by default. These are automatic local arrays passed
explicitly from the valid amplitude path. The public tensor diagnostic defaults
to unknown/no bounds unless its caller explicitly supplies the complete
contract. Never read an old amplitude ID from COMMON to identify a public call.

Production generator insertion points remain those in42a14: collect signed
wavefunction/corner routes at the actual `get_loop_CT_calls` emission (writer
1943) and store them during `write_loopmatrix` before `write_CT_interface`
(exporter2353 before2370). Use the emitted grouped/un-grouped ID, actual HELAS
state signs, ALOHA `%P` expression, mother order and pairing. The existing
unsigned support sets alone do not prove a route or an error bound.

Store both signed integer leaf coefficients and unsigned leaf multiplicities
of the unreduced momentum-expression DAG. Cancellation in signed coefficients
must not remove construction error: a source expression P-P still has two
occurrences. Store a conservative total add/subtract operation count for each
chord expression, and its minimum arithmetic precision. Unknown/custom
momentum producers, unsupported projections, or unresolved source expressions
yield no bound. Tree reuse must retain occurrence multiplicity in a chord.

For modes2/5 with N>2, C'_j=-C_(N+1-j), M'_1=M_1 and
M'_j=M_(N-j+2); otherwise leave order unchanged. Transform source expressions
and error metadata with the same rule; do not mutate reusable base tables.
Chords are G1=sum(C_j), G_(j+1)=sum(C_1…C_j). G1 is the actual computed
closure accumulator, not a literal zero. Preserve its expression/error and
actual bits. Translation or boosts must not be hidden by substituting LSCALE.

Classify A=G_i-G_j modulo the external conservation vector c:

* A=k*c: protected structural zero;
* A=k*c ± unit(e), known generated external shell: protected shell;
* unsupported/ambiguous route or off-shell singleton: unknown;
* supported route with neither identity: composite.

Use the generated mass/off-shell/KEEP_OFFSHELL_MASS contract for shell identity,
not merely a wavefunction decay `onshell` flag. Complement singletons are
protected. Composite alone is insufficient: a composite can be collinear or
lightlike. For the current26-route prototype the generated source tables can
establish these structural categories, but they do not themselves provide the
operation counts or all inherited error bounds; those must be source-audited.

## General caller error bound

Define the reference external vectors as the same actual DP/MP vectors used
to construct the numerator for this call, after the existing repair. This does
not claim the finite input equals an unknown continuum phase-space point.
For each external component carry E_e, its certified input/copy/conversion error
relative to that declared reference. A same-kind sign/copy is exact. A narrowing
conversion requires its rounding bound; unknown mixed-kind paths disable the
certificate. An optional half-ULP input enclosure may be added conservatively,
but is not evidence of original physical accuracy. Do not infer113-bit input
accuracy from a widened DP word. Nonlinear phase-space repair is not a hidden
linear DAG: either start the reference at its actual output, as proposed, or
provide an independently justified repair error envelope.

For a chord expression with unsigned multiplicities m_e and total operation
count L, let S=sum(m_e*abs(P_e)), U=sum(m_e*E_e), u=2^-p, and
eta=2^(emin-p), half the smallest subnormal. Require L*u <=1/2 and the supported
IEEE rounding/underflow contract. Standard forward error for add/subtract and
exact sign copies gives the sufficient bound

```
E_chord <= 2*L*u*S + 2*U + 2*L*eta.
```

Use the lowest precision on any path; for mixed exponents use the largest
underflow unit among participating formats. L may conservatively count the
whole expanded chord DAG, not just its longest path. This includes W momentum
construction, corner sums and the actual prefix/closure sequence. A simple
per-call maximum over validated routes is acceptable initially; coefficients
and operation counts still come from source, not calibration against results.

For the final subtraction qhat=fl(q_i-q_j), use

```
e_mu = E_i,mu + E_j,mu + u*(abs(q_i,mu)+abs(q_j,mu)) + eta.
```

These bounds use actual external/W/chord components; they grow with boosts.
sqrt(s), MU_R, LSCALE, and a universal beam-energy cap are not bounds.
Retain unreduced coefficients and construction paths, even when classification
uses conservation to identify a protected shell. This prevents closure
roundoff from being silently declared zero.

`source_chord_bound_exp` / `difference_bound_exp` in the pure reference specify
power-of-two overestimates of these formulas. They accept already validated
counts/input contracts; they do not discover them. Production caller code must
derive those counts and inspect finite input bits. That integration is open.

## Sufficient spacelike certificate and proof

Let qhat be the four actual rounded differences and r the raw real DIFFSQ
computed by the original helper, before any mass or zero projection. Assume
each reference difference differs from qhat_mu by at most e_mu. Then

```
| t_reference - sum_minkowski(qhat_mu**2) |
    <= sum_mu(2*abs(qhat_mu)*e_mu + e_mu**2).
```

The original real dot evaluation has four products and three sequential
subtractions. A term passes through at most four roundings, hence
gamma4 <=8*u for these formats. Its absolute underflow contributions, including
their subsequent rounding amplification, are bounded by16*eta (seven local
errors, each amplified by less than2). Thus use

```
B = sum_mu(2*abs(qhat_mu)*e_mu + e_mu**2)
    + 8*u*sum_mu(qhat_mu**2) + 16*eta.
```

If r is finite, negative, and abs(r)>B, the reference invariant is strictly
spacelike within the declared input/construction envelope. This is sufficient,
not necessary; failures retain legacy behavior and are not errors. There is
no tuned fractional cutoff and OSThres does not enter this certificate.

To avoid introducing new FP exceptions, `certificate.f90` evaluates only a
coarser power-of-two bound using integer IEEE fields. For nonzero x, take
abs(x)<2^(floor_log2(abs(x))+1). Sum n powers by max_exponent+ceil_log2(n), and
multiply them by adding exponents. The kernel obtains B<=2^b, and accepts only
floor_log2(abs(r))>b. It uses integer comparisons/arithmetic/bit operations;
there are no FP adds, products, divisions, conversions, rounding changes or
flag clearing. NaN/Inf including signaling encodings, signed zero, unknown
formats and malformed bounds fail without FP comparison.

The kernel accepts zero-extended64-bit words or full128-bit words through a
GNU INTEGER kind with at least128 bits. Binary64 callers must mask away sign
extension when widening their INTEGER*8 TRANSFER result. Caller-side Q and raw
DIFFSQ must be those already computed by the unchanged helper arithmetic.
Saving the existing differences in locals must preserve expression order and
complex-real semantics; source/compiler/native checks remain necessary. Do not
claim source-only Python tests validate Fortran lowering or ABI transport.

Required environment: qualified binary64/binary128, round-to-nearest,
gradual underflow, no FTZ/DAZ, no unsafe reassociation/fast-math, and no unproved
momentum producer. Read-only mode checks may establish this; never switch or
restore modes/flags to force eligibility. If the active state is not known,
ENVIRONMENT_VALID=false. Arbitrarily large finite boosts are handled by integer
exponents without extra FP overflow; nonfinite arithmetic is ineligible. The
certificate does not add numerical operations to the physical value path.

## Shared projection and direct invariant output

Retain raw DIFFSQ and the old S_MAT construction/branch order:
initial S=raw-M_i-M_j; mass_i snap; mass_j snap; massless-zero branch.
Track `MASS_SNAP_SEEN` even if later branches would overwrite its result.
Only a composite, supported, no-mass-snap pair with a successful runtime
certificate may skip the innermost massless-zero assignment. If either mass
snap fired, preserve legacy behavior regardless of the certificate. All
protected/unknown paths retain every old operation, threshold and projection.

For this newly certified pair return DIRECT_INV=REAL(raw DIFFSQ), USE_DIRECT=T.
Its native invariant consumer must use DIRECT_INV, not reconstruct
REAL(S_MAT+M_i+M_j): subtracting a large complex mass then adding it back can
erase or perturb the small transfer, even if the zeroing branch was skipped.
Compute legacy S as before for consumers requiring that representation.

For every other row USE_DIRECT=F and the consumer retains its old reconstruction.
Do not silently replace protected/mass-snapped values with raw geometry.
A future all-row direct representation could track projections as raw/M_i/M_j/0,
but changing those fallback rounding results is outside this first patch.
Maintain the current shared helper semantics and upper-triangle/symmetry policy;
do not give TIRust a separate physical invariant definition. Update each actual
invariant consumer in the selected private experiment (Ninja/NinjaRS/TIRust)
through the same explicit interface. CutTools and COLLIER have distinct native
interfaces; no claim is made that they consume this helper or identical raw
tensor bases. No COLLIER cache change is involved.

## Narrow test plan before numerical promotion

Twelve completed pure tests cover a resolved transverse direction, protected and
unknown fallback, negative near-lightlike cancellation rejection, larger bounds
under boosts, unsigned multiplicity, inherited error, subtraction error,
unsupported count/error bounds, exact dyadic sufficiency, and mass round-trip
loss, conservation-complement protection and off-shell-singleton fallback.
They do not use the saved physical points as acceptance allowlists.

Next source integration must derive counts from the bound CMS26 routes, emit
all supported groups/modes, pass fresh per-call masks/bounds, and demonstrate
unknown fallback on unsupported paths. Compile/mock transport tests must verify
DP/QP integer word decoding, subnormals, NaNs/sNaNs, boundary exponents, rounding
and FTZ fallback, no additional IEEE flags, and ordering/closure masks.

Only after review should the existing ordinary point1 and CMS20/21 be used for
a bounded private A/B: all applicable route decisions, raw/projected/direct
invariants, actual native status/error, coefficients only after success, and
finite/pole comparison eligibility retained. Include a true photon shell,
massive snap and unsupported route control. These are tests of a general rule,
not eligibility keys. W03/incomplete sums or unchecked host120 cannot become
accuracy evidence. Independent coefficient truth and amplitude checks remain
necessary; avoiding the Gram refusal alone is not acceptance. Profiling follows
correctness, with precise final timing deferred to the user's local machine.

## EW mass-square check: separate and narrower than a model mismatch

The closed point1/2 trace prints exactly the same DP and MP top mass:
173.300000000000011368683772161602974, DP hex0x1.5a9999999999ap+7.
The captured loop M2L equals the DP multiplication m*m. Squaring the exact
widened DP m needs103 significant binary bits, so it fits binary128 exactly.
Exact rational subtraction reproduces
-8.844835974741728406193461996918241986576081359316958697e-13 GeV².
This is DP-squaring roundoff versus the square of the same widened DP mass,
not evidence of two different model mass parameters. `loop_exporters.py:1545`
forms DP M2L; `TIRUST_interface.inc:178` widens it, while the QP repair squares
the promoted mass. No new MP mass arguments or mass policy change are proposed.
