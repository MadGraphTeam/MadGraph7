# Generated tensor capacity: bounded source proposal

**Status: source inventory and proposed protocol only. No Fortran compilation,
reducer evaluation, new amplitude, or timing measurement was performed.**
Production, existing exports, installed libraries, frozen campaign inputs and
reference repositories are unchanged. The proposed worker must be reviewed and
authorized separately before execution.

## What the existing generated processes actually contain

The current main EW n4 export has 54 six-propagator calls, all rank four. Its
`loop_max_coefs.inc` capacity is 70. The QCD n4 export has 40 six-propagator
calls, all rank five, and capacity 126. Neither contains a physical rank-six
group. [inventory.json](inventory.json) records every call, physical source line,
mass expression, wavefunction order, file digest and first saved point's exact
binary64 coordinates. These are **source call sites**, not an executed-group
coverage claim.

Use these preselected groups and the first saved ordinary point (index zero,
`ordinary-0000`) from each frozen process; do not search for a point that passes:

| Export | Group | Generated call anchor | Propagators / rank | Corner wavefunctions | Mass arguments |
|---|---:|---|---|---|---|
| EW `epem_mupmumaa` | 269 | `loop_CT_calls_1.f:884` | 6 / 4 | 1,2,4,6,5,3 | all zero |
| EW `epem_mupmumaa` | 270 | `loop_CT_calls_1.f:887` | 6 / 4 | 1,2,4,6,5,3 | second argument `MDL_MZ`, others zero |
| QCD `uux_zzgg` | 399 | `loop_CT_calls_1.f:1220` | 6 / 5 | 1,2,4,6,5,3 | all zero |

Paths are the candidate exports under
`.benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate/`.
The actual `CT_interface.f` rotates unsquared mass arguments to squared
`M2L=(M6**2,M1**2,...,M5**2)` and forms corners from `-W(WE)%P`.
Thus group 270 places the Z squared mass in slot three, not slot two.
The external HELAS calls determine the momentum signs; fermion-flow guesses
are not an acceptable replacement. No tensor-call input dump is currently
retained for these groups: the inventory explicitly says
`native_group_inputs_captured=false`.

## Proposed private materialization

Use a new append-only attempt beneath this namespace and an ignored worktree
directory `.tirust-build/generated-rank6-poles-1`. Do not rebuild or relink the
campaign in place. Use separate executables for the two models to avoid their
unprefixed model COMMON blocks and routine collisions. Resolve the installed
Fortran module/archive and DSO from
`.tirust-build/qp-transport-install/mg/HEPTools/tirust`, checking their retained
identities; the DSO must be
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
No Rust compilation or instrumented derivative library is proposed.

1. Retain actual source/object/library identities and expanded compiler/link
   recipes before materialization. Read the card through
   `SETPARA('nominal/param_card.dat')` with the actual `ident_card.dat` beside
   it; a bare filename is invalid for this generated reader. Retain `param.log`.
2. For the three capture calls, initialize the actual external HELAS objects
   with the generated six external calls and their declared arguments. Reuse
   the exact saved point. Set the documented CT/filter COMMON inputs explicitly,
   select only backend 9, CT1, full evaluation, cache disabled. Call the actual
   generated `ML5_0_LOOP_6` for the three group IDs above. Give its contraction a
   single unit scalar numerator coefficient and zero all other coefficients.
   This is a tensor-input capture using actual generated routing, **not** the
   physical process numerator or an amplitude validation.
3. A forwarding GNU link wrapper on `ml5_0_tirustloop_` copies entry PL/M2L
   integer bits and exit coefficients/status to a fixed buffer, forwarding
   exactly once. No replacement evaluator, fabricated status, file I/O or
   floating-point operation runs inside this wrapper. Flush records after the
   real call. Record all IEEE flags/rounding without clearing or repairing them.
   The wrapper captures the actual CT routing, avoiding a hand-coded duplicate.
   Its buffer and full record count are bounded and overflow is a hard failure.
4. Replay those captured inputs through the unchanged generated public
   `ML5_0_TIRUST_TENSOR_DIAGNOSTIC` and MP equivalent at their actual ranks.
   The original-capacity full DP output must agree exactly with captured full
   moments whenever the native call succeeds; preserve every refusal.
5. Render a **new standalone tensor-capacity adapter** with distinct `CAP6_`
   prefix, `maxrank=6`, `LOOPMAXCOEFS=210`, one diagnostic group, and the actual
   production TIRust/COLLIER templates. Generate the permutation and exponent
   tables through `FortranPolynomialRoutines(6)` using the production exporter
   methods, not handwritten lists. Generate its polynomial constant module
   with `write_polynomial_constant_module()`. Materialize the few required
   kinematic/order helpers from actual generated routine bodies with only
   symbol-prefix changes; retain an exact transformation manifest. Compile
   against a private include directory: do not enlarge original COMMON arrays
   or replace `loop_max_coefs.inc` in the existing export.
6. Rank six is then requested on the same captured geometry. This qualifies
   generated adapter **capacity and tensor moments of a synthetic rank-six
   numerator**. It never establishes that the original process had rank-six
   amplitudes. Every moment through rank six is tested, not a single contraction.
7. Generate the independent COLLIER adapter the same way, including its own
   production Cartesian map. Its private initialization must use
   `OVERALLMAXRANK=6`: unchanged process `INITCOLLIER` would initialize only
   rank four/five (`MadLoopCommons.f:71`, `global_specs.inc:4`). Use the actual
   initialization routine with a private bound rank-six header. Explicitly
   enable both pole controls, disable its numerical caches and record its error
   indicators. Do not accept a default that disables COLLIER poles.

Before any compilation, freeze the exact materialized sources, macros,
prefix/header changes, object reuse list and link recipes for peer review.
The above is a materialization specification, not a claim that a runnable
worker has already been implemented or tested.

## Fixed call matrix and independent checks

Seven fixture IDs are fixed in advance:

| ID | Geometry and requested rank | Classification |
|---:|---|---|
| 101 | Captured EW269, rank4 | actual generated rank and kinematics |
| 102 | Captured QCD399, rank5 | actual generated rank and kinematics |
| 103 | Captured EW269, rank6 | synthetic rank extension |
| 104 | Captured QCD399, rank6 | synthetic rank extension |
| 105 | Captured EW270, rank6, slot3 changed to `MZ**2-i*MZ*WZ` | synthetic complex-mass tensor stress; not a CMS amplitude |
| 106 | Lightlike scaleless bubble, rank6: shifts `(0,0,0,0)`, `(2,0,0,2)`, zero masses | analytical separate UV/IR control |
| 107 | Fixture103 with MU_R doubled, hence mu squared multiplied by four | scale/key control |

Keep all physical card bits fixed. Fixture105's complex squared mass is an
explicit DP quantity, promoted exactly for the generated QP diagnostic's
existing DP-mass API; do not claim a new QP mass-squaring policy. QP coordinates
are exact promotions of captured DP coordinates, without PS repair or invented
extra precision. Retain incoming/outgoing input bits separately.

Before every call, copy PL and M2L afresh from immutable captured fixture
storage, including calls differing only in precision, selector, cache setting
or direction. `SWITCH_ORDER` has in/out arguments and can reverse/negate corners
and permute masses. Never feed a previous call's exit arrays into the next call;
that could silently apply a second reversal. Retain both entry and exit words,
and compare reversed calls against independently refreshed reversed references.

For each fixture request DP CT1 and genuine QP CT4; UV+IR, UV-only and IR-only;
full and poles-only. This is 84 separate public tensor calls. Do not invoke an
ordinary amplitude entrypoint with finite evaluation disabled. Reinitialize
or clear appropriate caches outside these uncached calls, preserving errors
before any statistics query. Never turn a refused reduction into a zero result.

Compare all 70/126/210 Cartesian coefficients, all Laurent slots, and both
real/imaginary components. Partial finite slots must be NaN+0i in the generated
API; full finite slots and evaluated poles must be finite. Check UV+IR
reconstruction, zero UV double poles, finite invariance under pole selection,
and scale-independent double residues. The simple pole's scale shift must
match the separately evaluated full result and COLLIER, so IR2-to-IR1 mixing
cannot pass merely because both partial slots were zeroed.

COLLIER evaluates each fixture in all three pole modes (21 adapter calls,
at most 63 actual `TNTEN_CLL` calls). It is an independent numerical reducer,
not a wrapper around TIRust. Require equality of effective kinematics before
claiming independent agreement: retain the actual on-shell-adjusted invariants
passed to both libraries. The adapters have different invariant builders.
Any discrepancy is reported as an input-policy mismatch, never silently
snapped away. Instrument private input observation only; do not change either
formula. COLLIER remains double precision even when compared to TIRust QP.

For fixture106, the Feynman-parameter integral gives each MadLoop-ordered
rank-r moment's UV residue `p0**a*p1**b*p2**c*p3**d/(r+1)`, with the opposite
IR1 residue, zero IR2 and zero combined/full result. Here `p=(2,0,0,2)` and
`r=a+b+c+d`; a spatial x/y factor makes the residue zero. This independently
tests all 210 monomials and cancellation rather than just full/partial agreement.
The previous raw-C ABI test uses the same identity in IREGI order; this proposal
tests its generated Fortran permutation and separate mode mapping.

Record both dimensional and dimensionless residuals. For N propagators and a
degree-r monomial, multiply coefficients by `Q**(2*N-4-r)`, where Q=1000 GeV
for the physical geometries and Q=2 GeV for fixture106. This prevents tiny
dimensionful hexagon coefficients from passing against zero just because a
comparison uses a unit floor. On each dimensionless real/imaginary component,
full-versus-partial bounds are `1e-10` DP and `1e-24` QP for
`abs(a-b)/max(1,abs(a),abs(b))`. Independent COLLIER comparison uses the agreed
`3e-6` criterion and separately reports the `1e-8` target. Retain ordinary
relative errors and near-zero handling explicitly. There must be nonzero IR
and separate scaleless UV/IR witnesses; an all-zero comparison is insufficient.
Failure/refusal at any required precision remains an open qualification item.

For fixture104 only, add six cached requests per precision: full both twice,
partial both twice, partial UV, partial IR. Compare with corresponding uncached
references. Require actual DP reuse counters and QP hit counters for successful
repeats, and distinct full/partial/selector output parity. Add one reversed-order
partial check (CT2/5) and one independent reversed-order uncached reference per
precision, then one zero-budget partial request per precision. Verify checks
do not consume primary cached tensors, zero budget leaves no entries/bytes,
and final clearing empties numerical caches. This adds 18 calls, for 105 total
TIRust calls including the three full captures. Context creation, registrations,
clears and every native status are retained. Failure cache invalidation and
subprocess isolation already have separate current evidence; no new failure
injection or lifecycle campaign is included here.

## What establishes finite work omission

The existing native evidence is substantive and remains the finite-skip
instrumentation layer; a second instrumented Rust build is unnecessary for
this bounded complement:

* [tensor-unit-tests.log](../../qualification/poles-only/tensor-unit-tests.log)
  lines260–261 executes the constructor assertion that both full scalar
  evaluator arrays are empty, and the arithmetic test that fills finite
  scratch with NaNs while checking pole multiplication, division and sums.
  The exact tests are in the pinned patch at lines620–626 and194–205.
* [scalar-regression.log](../../qualification/poles-only/scalar-regression.log)
  line12 checks that generated scalar roots contain only Laurent tags -1/-2;
  four tests cover 288 point/precision comparisons. The generation manifest
  and patch hashes are retained by the existing index.
* [tensor-integration-tests.log](../../qualification/poles-only/tensor-integration-tests.log)
  has all N1..6 at cumulative rank6, genuine DD partial rescue, small-Gram,
  normalization and cache tests. Its initial two configuration refusals remain
  in the historical evidence. Core MP tests use 113/192 bits; this is not a
  historical claim about generated 256-bit bridge calls.
* [rust-tests.log](../../qualification/tirust-bridge/rust-tests.log) lines382–389
  binds historical bridge checks for 256/384-bit convergence, rank6 scaleless
  packing, failed-DP cache clearing and full/partial QP cache separation.
* Current `vendor/tirust/src/lib.rs:93,169,220–272` routes finite-disabled calls
  to `DirectPoleWorkspace`; its compile-time `POLES=true` execution omits the
  finite scalar constructors, gather slot and recurrence arithmetic. The
  current package records the same pinned production patch hash `80ef17b3…`
  and `MP_BITS=256`. [The installed-package qualification](../../qualification/tirust-install/qp-transport-cache/README.md)
  identifies eb3d and its exact build provenance. The new source inventory
  binds these files rather than treating a historical log as a current DSO run.

New forwarding C-ABI observation should record finite flags, N/rank, precision,
selector, check/cache flags, statuses and complete input/output records for
the proposed eb3d run. This verifies that the generated adapter requests the
partial path. **It is not a dynamic finite-operation count.** NaN masking is
also not evidence of omitted work. The combined claim is current generated
dispatch + same-patch core omission instrumentation + independent Laurent
agreement; retain those three evidence scopes separately. If direct operation
counts in the installed optimized DSO become a separate requirement, they need
a separately reviewed observer or instrumented build and are outside this run.

## Resource and evidence contract

After separate authorization: one serialized run under the existing numerical
lock, a single CPU assigned by root (no overlap with timing), at most two model
workers, 4 GiB address-space limit per worker, 900 seconds per worker, no
automatic retry, no adaptive point search. Compile only the bounded private
driver/observer/capacity helpers; reuse source-bound existing numerical objects
where compatible. Cap compilation at 240 seconds and two compiler processes.
Before running, require the final materialized schedule to contain exactly
105 TIRust calls and 21 COLLIER adapter calls, with a hard stop above 120 TIRust
calls. If link closure would require a fresh full model build, stop and report
the changed resource requirement before doing it.

Retain append-only records with a versioned header, call IDs, fixture IDs and
origin class, source group/rank, effective requested rank/count, exact input
bits, all coefficient words, native and host statuses, error text on failure,
IEEE entry/exit state, cache snapshots, initialization/finalization markers,
exit status and timeout metadata. The analyzer rejects missing/extra/duplicate
records, omitted coefficient rows, silent rank skips, nonfinite evaluated
results and undocumented fallbacks. A final success marker is necessary but
not sufficient. Output JSON distinguishes `served`, `refused`, `mismatched`,
`input_policy_mismatch`, and `not_executed`; none may be silently discarded.

This proposal can close current generated Fortran rank-six tensor-capacity,
physical N6R4/R5 partial mapping, independent coefficient and selected cache
coverage. It cannot close an actual generated rank-six physical-amplitude
claim, full SM/CMS stress qualification, the full 100-point campaign, or performance
acceptance. No speed claim is attached to these bounded calls.

Recheck only the source inventory from the repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/inspect.py --check
```

The checker reads bytes and JSON only. It does not import MadGraph, invoke a
compiler, load a numerical library, execute a generated subprocess, or edit
any source/evidence file.
