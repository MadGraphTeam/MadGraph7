This is a source-bound implementation proposal for a single effective point in
QP rescue. Production source is unchanged. The hook plan is **not apply-ready
or compile-ready**: the generated Born aggregate and checked repair entry must
be implemented together before any native test. The executable Python model
tests lifecycle requirements, not Fortran, IEEE behavior, or amplitudes.

The narrow supported branch is scalar optimized SM output with a Born,
ordinary on-shell external legs, and direct interference reduction. Preserve
the existing loop-induced branch. Color-flow/amplitude-reduction output,
CTMode0 internal CutTools rescue, off-shell/spacelike legs, and vector output
keep their explicit legacy path until separately wired; no new qualification
claim covers those paths. This is not a new gate on the existing benchmark
processes. No reducer, threshold, precision guard or model-mass policy changes.

The needed change extends beyond the nonfinite-Born trigger in the private QCD
prototype. The host publishes Born and `REF` on `P_USER` before the QP point
exists. A later QP loop rescue may consume repaired C even when that earlier
Born was finite. For every **adopted** C-based rescue, publish Born and set REF
from the same coordinate/parameter prescription as the interfering Born.
Finite original Born is therefore not a reason to skip this update.

| Integrand policy for the actual selected QP backend | Published Born and REF | Interference Born and coordinate view |
|---|---|---|
| `UseQPIntegrand... = .TRUE.` | QP Born at canonical C with the existing MP model; checked conversion only for the public DP result | Existing genuine MP HELAS calls at the current rotation of C, with the same MP model |
| `.FALSE.` | Actual DP Born at `REAL(C,8)` with the DP model | Existing DP HELAS calls at the rounded view, followed by promotion; do not label this full-C numerator arithmetic |

The current false branch copies DP wavefunction momenta into MP wavefunctions,
so reducer shifts also inherit that rounded coordinate view. The canonical C
remains available for reproducible rotations. QP tensor arithmetic is a
separate precision property. The paired MP propagator-mass proposal is a
separate denominator policy: it does not make DP and MP model parameter values
identical. This design neither changes that policy nor silently promotes DP
model parameters in the true branch.

Use one call-local state owner. `C_READY` means only that the existing repair
has explicitly succeeded and the required conversions are valid. It is not
`MP_DONE`, `MP_DONE_ONCE`, or an accuracy certificate. Preserve canonical C in
`MP_PS`, the current rotation in `MP_P`, and one rounded DP copy. A policy key
records true/false integrand mode and the current model/selection context; it
does not survive the host call. No new process-global scalar or mass cache is
needed. Within a call, helicity/flavour/polarization selections and the model
initialization contract must remain fixed.

The proposed transition order is:

1. Reset the current-call certificate before any possible early return. Execute
   the original DP Born path once. Ordinary successful DP evaluation uses its
   original Born, REF, coefficients, flags, filters and status; no MP model
   update, repair or additional Born evaluation occurs.
2. A nonfinite DP Born triggers the existing first available QP chain before
   REF-dependent work; a finite Born triggers the same handoff when the host
   first enters its ordinary QP rescue. Direct positive QP mode also reaches
   this handoff, but remains unchecked. Do not infer readiness from CTMode0's
   opaque legacy internal rescue.
3. Initialize through the existing model contract and call `MP_UPDATE_AS_PARAM`
   before genuine MP work. Prepare C once from the original saved input using
   a checked sibling of the existing repair entry. Adopt it after ordinary MP
   resets, or retain the prepared handoff through those resets. Do not repair
   the rounded C copy again and do not call the lazy setter a second time.
4. Resolve the integrand policy for **every** selected QP library, outside the
   one-time `.NOT.MP_PS_SET` guard and before any `MP_DONE`/coefficient reuse.
   On first adoption or a policy change, generate the
   complete Born split-order result at the matching view, validate arithmetic
   and DP conversion, then atomically replace `ANS(0,0:NSQSO_BORN)` and REF.
   Preserve the existing REF formula: sum entries `1:NSQSO_BORN`, not merely
   the separately chosen-order sum at index0. Do not pass the strided row
   `ANS(0,:)` to a contiguous legacy array dummy; return a Born buffer and copy.
5. Invalidate coefficients and integral stores on first adoption or policy
   change. A backend switch with the same view can reuse the Born, but a
   true/false policy change must refresh it and clear `MP_DONE`, any coefficient
   reuse flag, and incompatible cached results. Only the actual existing
   numerator routine may subsequently mark numerator completion. Preserve
   separate primary/check stores and the existing cache boundary for rotations.
   Start a fresh stability cohort on adoption/policy change: reset the current
   QP result slots, `EVAL_DONE` and `STAB_INDEX` before collecting that cohort.
   Keep historical DP/earlier-policy results only as separately labeled
   diagnostics; they must not enter this cohort's accuracy calculation.
   In the direct-interference path, a backend change with zero QP rotations
   currently jumps directly to label300, past label205 and the numerator call.
   Check the selected policy at that switch too; a changed policy must route
   through the staging/recomputation entry instead of that reuse shortcut.
   Same-policy direction tests may retain their existing shortcut.
6. Derive every rotation from canonical C. The published Born/reference stays
   associated with that canonical physical view; the interference uses the
   corresponding rotated view. The ordinary independent direction/rotation
   checks remain unchanged and are not a correctness oracle.
7. On preparation/Born/conversion failure, leave no current-call certificate,
   perform cleanup appropriate to the completed initialization stage, and
   return rejected status with accuracy1 and
   invalid numerical output. Do not reuse the original finite Born beside a
   failed C-based loop as an accepted result. A Born-only QP calculation must
   not manufacture `MP_DONE_ONCE` or a QP-loop return-code digit. A nonfinite-Born
   rescue still requires an actually checked successful loop before it can be
   accepted; fixed-QP diagnostics never create that certificate.

The nonfinite-Born branch precedes host `NTRY` advancement, filter publication,
the user-threshold backup, and bypass-filter snapshots. It cannot blindly jump
to `9999`: some cleanup inputs and the current backend identity are not yet
initialized. Record validity for each modified setting and restore only valid
snapshots. Early rejection must not advance host initialization counters or
publish filters; later rejection must restore the state actually modified.
Some rotation backups already exist at the original Born call, so use actual
stage validity rather than assuming that all cleanup snapshots are absent.

`SET_MP_PS` currently returns no success flag, and the repair routine can retain
finite original input after failure. The checked sibling must expose the
existing primary/alternate outcome and both error codes, preserving the same
arithmetic, `ImprovePSPoint` choices and `KEEP_OFFSHELL_MASS` semantics for the
supported path. A finite-array test alone is expressly insufficient. The
legacy entry remains unchanged. A shared rendering of the original body can
produce the legacy and checked entries; source tests should prove the checked
version adds only outcome reporting and adoption control. No new projection,
threshold or reinterpretation of the input point is proposed.

Generate the rare-path Born evaluator from the same deep-copied Born
`HelasMatrixElement` and color basis used by `write_bornmatrix`. Emit precision
specific HELAS/model names from the writer rather than rewriting a particular
generated process or promoting every DP parameter. Retain the generated
helicity list, flavour masks/index handling, polarization and beam weights,
color contractions, symmetry/initial averages, Born split-order map and chosen
orders. A stateless aggregate evaluates the requested physical helicity set in
the existing order and adds no `GOODHEL`, `NTRY`, `HELRESET` or loop-filter
learning updates. The original DP call retains its original learning behavior.
Its additional rare-path work must not overwrite loop waveform/AMP COMMON
storage. The evaluator and the existing interfering Born must be source-tested
to resolve the same model symbols and graph ordering. The private QCD helper's
hard-coded dimensions and promoted-DP couplings are not a general template.

The original DP Born itself owns saved `NTRY` and `GOODHEL` arrays. Both Born
templates currently update those before returning, and a nonfinite result can
satisfy the existing nonzero test. A host-only failure cleanup cannot restore
these private arrays. The generated supported Born routine therefore needs a
transaction for its learning state: a successful finite original evaluation
commits exactly the existing updates, while a rejected nonfinite attempt
restores entry `NTRY`, `GOODHEL`, `HELRESET` and selection state. This is a
source-generation prerequisite, not an implemented rollback or a reason to
invoke a broad helicity-filter reset. Its ordinary-path overhead is unmeasured.

IEEE handling is a separate acceptance requirement. Never disable caller traps
or change the rounding mode. Snapshot the status before the original DP Born
if a nonfinite-Born rescue is enabled. The ordinary successful path leaves all
flags untouched. For that specific fallback, discard only the failed DP Born
attempt's new flags, as in the retained wrapper, then merge rescue flags into
the caller's entry status. For a finite-Born-to-QP transition, snapshot at that
transition and retain accumulated earlier flags; do not erase failed loop
history. Check new invalid/divide-by-zero/overflow locally while traps retain
their original behavior. Include C rounding, Born output conversion and final
copy in the protected interval. Restore status/rounding after those conversions
and merge newly raised flags. Preparation success must not clear the backend's
separate `FPE_IN_*_REDUCTION` result. The Python model does not test any of these
IEEE operations.

No earlier DP cohort is a same-input truth test for adopted C. Preserve it for
diagnostics, but compare each QP cohort only within its declared input/policy
context. Normal model reinitialization must invalidate existing dependent
caches; this proposal adds no cross-call state that can survive it. Re-entering
after failure or an ordinary–stress–ordinary sequence starts with no stale C,
Born or completion certificate. Legitimate zero Born values are not replaced
by an arbitrary positive reference floor.

The retained topology-c evidence establishes why coordinates matter: the QCD
double-pole check changes from a `3.53039e-4` mismatch to `1.32950e-15` when the
published Born follows C, while loop coefficients are unchanged. That is one
coefficient and one parameter prescription, not finite/single truth or a proof
of this general design. The separate default-stability-policy audit remains
open: an internally checked return need not meet the external accuracy goal.

Next implementation work is limited to the checked repair sibling, generated
stateless Born evaluator/failed-attempt learning transaction, and the host
hooks described in `hook-plan.json`.
Before any native run, generate real/CMS SM source fixtures with helicity,
flavour, split-order and model-symbol checks; inspect unsupported-branch
fallbacks and each failure/cleanup exit. Native qualification must then cover
ordinary bit identity, finite-Born QP rescue, nonfinite Born, actual repair
failure, conversion failure, enabled traps/all entry flags/rounding modes,
true/false policy changes, rotations, model updates and subprocess isolation.
The ordinary-path snapshot/guard overhead is unmeasured and remains subject
to the existing performance criterion. Broader off-shell/vector/LI behavior is
not being redesigned.

Run `python3 -B verify.py` from this directory to verify the inventory, or use
the repository-relative path from the MadGraph root. `--sources` additionally
reconstructs the recorded source bytes from Git blobs or retained dirty-source
snapshots and checks their identities, the evidence links and hook anchors.
`--current-sources` also requires the current working source to match. The retained
unit-test result comes from pure Python lifecycle tests; no MadGraph import,
Fortran compiler, DSO load, integral or amplitude evaluation occurred.
