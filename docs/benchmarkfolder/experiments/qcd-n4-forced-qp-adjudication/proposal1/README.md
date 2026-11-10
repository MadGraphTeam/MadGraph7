# Three-point forced-QP adjudication proposal

This is a source-only investigation plan for the closed QCD n4 campaign's
finite-term `1e-8` target misses. It makes no new numerical or performance
claim. All original comparisons pass the agreed `3e-6` gate. The selected
points are 61 (largest baseline discrepancy), 89 (largest discrepancy
between the four non-CutTools reducers), and 93 (borderline TIRust/COLLIER
comparison with accepted QP/DP results). `selected-evidence.json` retains
their original answer orders, coefficients, statuses, and accuracy estimates.

`points.dat` contains exact original decimal lines for bootstrap point 1,
then points 61, 89, and 93. The selection also records their binary64 words.
No coordinate, width, mass, coupling, on-shell threshold, pole mode, or
kinematic-repair policy changes. The original cards are retained verbatim.
The prior rank-six, CMS and adaptive Fortran packages remain unchanged.

## Proposed finite experiment

Use the existing closed candidate export for `u u~ > z z g g [virt=QCD]`.
Link its original generated numerical objects and archives to a private
driver and an additive observer. Do not regenerate the process or compile
modified numerical sources. The current object bytes are frozen in
`dependencies.json`; unlike the source and archive hashes, these object
hashes were not recorded by the original campaign, so their binding is
explicitly prospective.

Create fresh private workers for explicit backends 1, 8, and 9. Each worker
starts from its original campaign backend card. Keep the original
`CTModeInit=1`, `MLStabThres=1e-6`, `ImprovePSPoint=2`, full finite output,
both poles, cache policy and QP integrand settings. Use the existing worker's
bootstrap and steady-status predicate on point 1, with at most eight
bootstrap calls. Retain every call; failure to reach steady state stops
that worker. This short initialization is a diagnostic protocol, not a
replay of the original 100-point warm-up history.

After bootstrap, set only `CTModeRun=4` through the existing parameter
COMMON, and perform exactly one call at each of 61, 89, and 93 in that
order. Restore the saved parameter after the calls. Use the original
`ML5_0_SLOOPMATRIX_THRES` entrypoint and the original `UPDATE_AS_PARAM` and
scale construction. This explicitly requests QP without changing the
scientific input. Forced QP bypasses the ordinary independent stability
sequence: status `120` means unchecked QP, never checked success. The
record must keep negative/missing accuracy estimates, refusals, exceptional
results, and all IEEE entry/exit flags. A returned zero is not evidence of
a valid amplitude.

Build plain and observed executables from identical numerical objects and
the same diagnostic driver. The observed link adds only proven external
wrapper edges. Run the six backend/observation combinations in fresh
directories and require bitwise plain/observed parity for input/output,
status, model state, and IEEE state. Failure preserves both attempts and
blocks attribution to the observer. This doubles the small numerical
matrix to establish that the observation did not change the result.

Proposed resource bounds, requiring later explicit admission: two private
source compiles and two links under one 120-second/4-GiB cap; six serial
workers, each at most 900 seconds/4 GiB, CPU 12 nonexclusive, using the
existing reviewed owned-process supervisor and the shared numerical lock.
Loader checks and actual mapped-library identities must bind the original
released eb3d TIRust DSO and the closed NinjaRS/CutTools dependencies. No
new supervisor or automatic retry is proposed. Preserve `param.log`,
filters, raw records, console output and failures. These are provisional
resource observations, not final timing measurements.

## Actual Fortran values to observe

Python reconstruction of canonical momenta is not the input authority.
Use pass-through wrappers around existing cross-object calls, with integer
`TRANSFER` copies into bounded memory and formatting only after all calls.
Never clear/restore numerical IEEE flags around an evaluated operation or
query a native error-clearing getter before saving the original error.
Each wrapper forwards exactly once. Link relocation and runtime traversal
must establish the actual edges before the resulting evidence is accepted.

1. `ML5_0_SMATRIXHEL_SPLITORDERS`: retain the actual user momenta, helicity
   and flavor selection, DP model masses/couplings, and both returned Born
   orders. This is the Born used by the public amplitude.
2. `ML5_0_MP_IMPROVE_PS_POINT_PRECISION`: retain incoming promoted momenta,
   `KEEP_OFFSHELL_MASS`, MP model masses before/after repair, and the actual
   returned canonical C momenta. Record the raw binary128 words.
3. `ML5_0_MP_COMPUTE_LOOP_COEFS`: retain its actual PS argument, selected
   backend, CT mode, QP integrand flag and returned contributions. Require
   those PS words to equal the preceding actual canonical C capture.
4. `MP_UPDATE_AS_PARAM`: retain MP masses and couplings before and after
   its original call. This distinguishes repair-time masses from the
   masses actually used when constructing the QP numerator.
5. `ML5_0_MP_HELAS_CALLS_AMPB_1`: retain each actual helicity input and
   the 30 complex binary128 Born amplitudes from `/ML5_0_MP_AMPS/` after
   the original call. Compare them across the three reducers. This records
   the Born amplitudes entering QP interference, independently of the
   public DP Born result.

Capture initial, entry and exit IEEE flags, halting modes and rounding
state at these boundaries and the public call. Integer capture is allowed
to observe inherited sticky flags; inherited state must be distinguished
from newly raised flags. No fresh-state repair is performed. If an edge
is locally resolved and cannot be wrapped, do not silently substitute an
approximate Python computation or alter the generated routine; revise the
observer proposal explicitly.

The generated source establishes why these observations matter: public
Born is computed with `P_USER`; `SET_MP_PS(P_USER)` repairs promoted
momenta using MP model masses; QP numerator construction subsequently
calls `MP_UPDATE_AS_PARAM`. The original native reducer interfaces also
receive complex squared masses of DP kind and promote them. A QP loop
therefore must not be described as having arbitrary 113-bit mass inputs.

## Adjudication beyond reducer agreement

First compare actual C momenta, repair/numerator MP masses, couplings,
helicity selections and QP Born-amplitude words. If any differ, classify
the input difference before comparing finite terms. Keep original and
forced-QP results separate. A newly converged QP CutTools result can
identify a DP accuracy problem, but three similar answers do not by
themselves establish finite-term truth.

Independently recompute the color-summed Born from the captured QP Born
amplitudes using the exact generated color rational coefficients and
high-precision arithmetic. Use 80 and 160 decimal digits with identical
captured binary inputs and require convergence. Check the independent
universal QCD double pole with actual native coupling and the explicitly
identified Born/normalization convention. These checks establish input,
normalization and pole consistency; they do not establish the finite term.

If the finite ambiguity remains, the next bounded investigation must first
localize the contributing loop groups and preserve their actual numerator
coefficients, shifts, squared masses, invariants and scale. It must then
replay the *same frozen mathematical integral and numerator* in an
independent higher-precision reduction/master-integral calculation with
explicit residual/error and precision-convergence evidence. An independent
CutTools QP reconstruction test can constrain numerator reconstruction;
master-integral accuracy and cancellation still require their own evidence.
Which integral and independent representation to use is deliberately
deferred until the actual discrepant groups are known, instead of
preselecting an unrelated easy scalar fixture.

Neither adaptive 256-to-512 success nor `last_output_bits=512` is truth or
accepted-amplitude attribution. The historical all-512 DSO differs in
source and is excluded from this first matrix. A subsequent precision
ladder must isolate precision from source, kinematics, mass construction,
and cache changes; it cannot inherit qualification from that old DSO.
Unresolved finite target misses remain open in the report.

## Current state

The exact selected inputs, closed source/archive bindings, object snapshot,
and this plan are frozen. No diagnostic driver, observer, build or runtime
controller has yet been admitted or executed. The next source step is the
small driver/observer implementation and a strict parser using the above
finite matrix; the existing lifecycle/controller machinery is reused.
