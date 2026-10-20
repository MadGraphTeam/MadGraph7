# Binary128 numerator with binary64 reduction: feasibility review

This is a source-only study of committed numerical source
`f4bffc61ed67e5d083b80ff9901768d4d5779989`. No mixed-precision implementation,
amplitude evaluation, threshold change, or benchmark-policy change was made.
The concurrently authorized relocated-installation controls are a separate task.
`sources.json` binds the inspected files and their line anchors.

**Feasible as an explicitly identified intermediate attempt, but not as a flag
change to the current compensated-DP retry.** Computing the numerator in
binary128 and rounding its finished coefficients once can reduce coefficient
construction error. It cannot recover small coefficients already lost when
rounding to binary64, nor fix an ill-conditioned binary64 reduction, geometric
construction, or master integral. The first useful experiment therefore separates
these errors before adding another automatic retry.

## Existing seams and constraints

| Area | Evidence and consequence |
|---|---|
| Numerator precision | `mp_compute_loop_coefs.inc:209` chooses genuine QP construction through `UseQPIntegrandForNinjaRS`. Its false branch computes DP and upcasts to QP, which is the opposite of this proposal. The true branch refreshes MP parameters, computes MP Born/UV/R2 and color contractions, and forms `MP_LCOEFS`. Final coefficient merging occurs at line 687. A downcast must use the finished, merged coefficients for every physical squared order and loop group. |
| Reduction dispatch | `helas_loop_amplitude.inc:90` derives `DOING_QP` directly from `CTMODE>=4`. Its NinjaRS branches at lines 162–170 select the DP or MP wrapper accordingly. Setting mode 4 to construct the numerator also selects genuine MP reduction; keeping mode 1 currently skips MP construction. A private stage must separate numerator precision from reduction precision without changing public CT-mode meaning. |
| Existing retry | `loop_matrix_standalone.inc:1833–1868` permits one compensated-DP attempt after a finite native failure, excluding FPE/nonfinite failures. It resets the direction/rotation series and records native/compensated accuracies. The wrapper at `NINJARS_interface.inc:108` changes only the DP tensor evaluator. It does not reconstruct the numerator. Repurposing this flag would silently change its existing statistics and interpretation. |
| Full stability test | `loop_matrix_standalone.inc:1679–1750` stores complete Laurent/split-order results, repeats reverse direction and configured rotations, restores the original point, then calls `COMPUTE_ACCURACY`. A mixed attempt needs its own complete result series with the unchanged threshold, normalization, direction/rotation policy, and hard-failure handling. It must not compare one mixed result against old DP results as though those formed one precision-consistent series. |
| Geometry | `SET_MP_PS` at lines 2330–2352 improves the phase-space point in MP. MP construction also copies MP wavefunctions back to the DP containers (`mp_compute_loop_coefs.inc:501–509`). Simply calling it and continuing in DP can therefore change the denominator momenta. To isolate numerator error, explicitly promote the original DP external point without improvement and retain the exact original DP reduction momenta, masses, S matrix, scale, and thresholding. A later physical mixed scheme may choose another consistent geometry policy, but that is a separate change to test. |
| Loop-induced/amplitude reduction | `mp_compute_loop_coefs.inc:512–524` performs reductions inside the helicity computation when `AmplitudeReduction` is enabled. It cannot be treated as a single post-summed coefficient array. `SKIP_LOOPNUM_COEFS_CONSTRUCTION` can reuse a single helicity at label 300. An initial diagnostic should explicitly support only the QCD3 interference process; later support needs a per-helicity splice and coverage of loop-induced and mixed-order controls. |

For the interference path, the technical splice is after MP construction and
merging, before the normal DP loop calls: round all active MP coefficients with
an explicit kind conversion, select the native or compensated DP wrapper, and
run the normal acceptance machinery. Preserve the original DP geometry and
coefficient arrays separately during the diagnostic. Also separate two variants:
**numerator-only replacement with the original Born/counterterm contribution**,
and **QP numerator plus rounded QP Born/counterterm contribution**. The latter
is closer to a consistent QP-integrand attempt, but its improvement cannot be
attributed solely to the loop numerator. Check conversion finiteness before
calling the reducer; overflow/nonfinite conversion is a failed attempt.

## State, reuse, and fallback

The numerical state is larger than the tensor coefficient array:

* `MP_DONE`, `MP_DONE_ONCE`, `MP_PS_SET`, `DPW_COPIED`, `BUFFR_BIS`, both coefficient
  commons, `LOOPRES`, and the DP/MP split-order completion flags must refer to
  the same point, rotation, helicity selection, parameters, and numerator stage.
  `MP_DONE_ONCE` is accounting, not an adequate coefficient-cache key.
* Label 300 reuses coefficients for the reverse direction; label 200 clears
  caches and reconstructs a rotated point (`loop_matrix_standalone.inc:960`).
  A mixed stage must rebuild QP coefficients for each configured rotation and
  force construction when changing from DP to QP numerator, even with zero
  rotations. It must not enter label 300 with stale DP coefficients just because
  the current compensated retry can do so. Reset or restore the per-stage flags
  and all result slots on every exit, including failure and backend change.
* If a mixed attempt fails, genuine MP rescue may reuse only QP coefficients
  whose kinematics, model parameters, helicity/split-order selection and merging
  identity exactly match that rescue. The proposed diagnostic intentionally
  uses unimproved DP kinematics, so the ordinary improved-MP fallback must
  reconstruct them. Never overwrite the genuine MP coefficient storage with
  downcast/upcast values.
* NinjaRS DP and compensated contexts overwrite coefficients on every call
  (`vendor/ninjars/src/lib.rs:217,296`), so a new ABI is not intrinsically needed
  to supply rounded coefficients. Their topology reuse still requires retained
  versus fresh-context tests with changed coefficients, masses and scale. The
  ordinary TIR cache stores integral tensors independently of numerator
  coefficients (`TIR_interface.inc:249–273`); changing only coefficients does
  not invalidate those tensors. Geometry/CT-mode changes still require the
  existing cache-clearing rules. NinjaRS uses its direct wrapper, not this TIR
  contraction cache.

The safe prospective ordering is existing native DP, existing compensated DP,
then **at most one declared mixed stage**, then the unchanged genuine MP rescue.
An initial diagnostic can evaluate both native and compensated reduction of
the rounded QP coefficients to choose the stage; that does not authorize adding
both to the production chain. Keep the current backend fallback order and hard
failure policy. Explicit forced modes 4/5 must always retain genuine MP
reduction and must never enter this intermediate DP path.

## Honest precision and usage accounting

The genuine-rescue requirement is preserved only if the actual 256-bit NinjaRS
engine remains available and is used whenever the intermediate attempt fails,
including forced-QP controls. The MP wrapper still supplies binary128 inputs
through the 40-digit interface to the independent 256-bit reducer and masters;
the proposed intermediate attempt would supply binary64 inputs to the DP
engine. It is **mixed precision**, not a 256-bit rescue.

Existing H/T/U semantics could naturally yield `228` for stable DP reduction
after QP numerator work: H=2, T=2 because `MP_DONE_ONCE`, U=8 for NinjaRS DP.
That is an observation, not a new status allocation or implementation proposal.
It would remain counted as NinjaRS DP by `BinothLHA.f:500`, while overall QP-work
accounting increases through T=2. It must not be reported as `218` (DP-only work)
or `322` (genuine NinjaRS QP reduction). A separate explicit mixed-attempt count,
success/failure/hard-skip count, per-stage accuracy, and final-path indicator are
needed to avoid conflating it with the existing compensated retry. Preserve
all old status integers and distinguish actual QP-reducer use from any QP
numerator work in reports.

The frozen benchmark validator currently permits T=2 with H=2/U=8, so `228`
would not automatically be rejected. The proof layer classifies precision by H,
and would correctly not count that status as genuine-QP coverage, but neither
layer currently proves the existence or policy of this additional stage.
Therefore existing validation accepting that number is insufficient: a future
policy would need explicit stage evidence and fresh source/runtime/protocol
binding. Do not silently introduce this into the frozen collection tools.

## Bounded diagnostic that would answer the question

This plan is not launched or authorized by this review.

1. In an isolated copy of the exact current QCD3 generated process, use the
   unchanged stored points 44 and 80 plus declared stable controls 1 and 67.
   Retain every loop group/order and every existing direction/rotation; no
   selection of only favorable diagrams. Keep initialization bounded and
   record the whole attempted stream, including timeouts and hard failures.
2. Capture the complete DP numerator/reduction inputs and the freshly constructed
   binary128 numerator for the same promoted DP external momenta. Preserve DP
   geometry bytes. Retain rank/order/group identity, coefficient bits, masses,
   scale, S matrix, Born and counterterm pieces, and every reducer result.
3. For each stored input, compare the following matrix. A genuine-256 run on the
   **same promoted stored binary64 geometry and coefficients** distinguishes
   reduction error from numerator error; the existing full-MP run with improved
   kinematics is an additional physical reference, not the isolation oracle.

   | Coefficients | Native DP | Compensated DP | Genuine 256 |
   |---|---|---|---|
   | Original DP | current baseline | current retry | reduction-error reference |
   | Rounded QP | proposed diagnostic | proposed diagnostic | rounding-loss reference |
   | Full binary128 QP | not applicable | not applicable | coefficient-construction reference |

4. Reassemble all finite and pole coefficients and all physical orders, first
   with identical original counterterms and then with QP counterterms separately.
   Apply the unchanged MadLoop acceptance series and the task's existing
   coefficient comparison tolerances to the independent physical references.
   Report cancellation-sensitive absolute/point-scaled errors as well as
   relative errors. Direction agreement alone is not proof of accuracy.
5. Record construction, conversion and reduction times separately, but make no
   performance claim from this four-point diagnostic. If neither rescued point
   passes the unchanged gate with rounded coefficients, stop this hypothesis.
   If it works, an independently reviewed copied-process implementation must
   pass the complete 100-point stream, loop-induced/mixed-order/complex-mass
   controls, forced-QP controls and state-rebinding checks before any quiet
   full-process timing or policy proposal.

The cost benefit is uncertain. This stage still pays for QP Fortran numerator
construction and adds another DP reduction series. It can save the generic MP
tensor/reduction/master/geometry work only on points it successfully avoids
rescuing. If it fails, that work remains and the new attempt is overhead.
Both current points continuing to require rescue is evidence for investigating
the error source, not evidence that this specific numerator remedy will work.
