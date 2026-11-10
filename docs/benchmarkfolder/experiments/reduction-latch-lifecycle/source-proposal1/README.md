# Independent-call reduction failure latches

This private proposal adds exactly two executable assignments immediately before
`MP_DONE=.FALSE.` in the optimized `loop_matrix_standalone.inc`. It does not edit
production or change the existing contiguous precision-stream initialization
sequence. The original template, generated QCD n3 source and candidate copies
are retained beside `proposal.patch`.

The software DP/QP failure flags are initialized by DATA once, and otherwise only
cleared when automatic stability rescue changes backend or precision. Baseline
9228 has the same behavior. Forced modes therefore retain a previous independent
call's failure; automatic terminal failures can also leave state behind. The
host owns the independent-call boundary. A reducer failure must remain latched
across its groups, helicities, split orders, direction tests and rotations,
then clear on the next host call. Existing backend-transition resets remain.
Clearing caches, resetting flags per group or clearing them at return would
remove evidence required by stability checks or post-call diagnostics.

`stub_check.py` exercises source-bound compiled stub control flow only, with
and without the existing NinjaRS public wrapper. It reuses the verbatim entry,
accuracy-mask and rescue-reset fragments and models failing/successful reducers.
These tests establish latch behavior; they do not execute Standard Model
amplitudes, Rust libraries or master integrals, or establish numerical accuracy.

`sm-replay-plan.json` contains the two bounded compile/link commands and four
proposed representative parity runs using the existing closed QCD n3 build2
objects, runtime providers, driver and three physical points. Only the generated
matrix object changes. Real numerical replay remains unexecuted pending root
readiness. Status 120 and accuracy −1 remain unchecked; this plan cannot establish
100-point campaign or performance acceptance. COLLIER cache behavior is untouched.

Separate inherited behavior remains out of scope: forced QP starts with
DOING_QP_EVALS false, so generic final NaN detection can attribute that NaN to the
DP flag. Backend-specific QP failures already set their QP flag correctly.

The completed synthetic qualification passed eight pure checks and four compiled
builds with 32 fresh processes / 76 public stub calls. It reproduces the original
sticky-state defect and preserves within-call failures and automatic rescue.
The first compiler attempt failed in test-only fixed-form rendering before any
stub execution; its receipt and source remain retained separately. Totals across
both attempts are five compiler commands and 32 synthetic processes. These
results do not include a real SM amplitude or reducer execution.
