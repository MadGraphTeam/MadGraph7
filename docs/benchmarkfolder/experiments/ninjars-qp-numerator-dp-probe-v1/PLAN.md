# Held capture and replay diagnostic

Scope: copied generated Fortran only; no production, retry-controller, status,
threshold, contribution, or frozen benchmark-tool change. No promotion or
acceptance timing. Prepare/build uses CPU2. Capture and replay evaluations are
held until root explicitly assigns the Symbolica/MP slot.

The source is the clean f4b QCD3 export in
`.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1/outputs/candidate/qcd-3`.
Before copying, require its complete generated-source manifest and executable
to match retained generation evidence. Its executable must also equal the
validated fresh dev-QCD3 executable (all100 original backend8 rows are retained
there). Rehash complete process libraries, modules/native-link metadata, recipes,
physical/MadLoop cards and points. Copy and dereference the output into this
experiment's private work directory; never build in the source output.

Instrumentation changes only `check_sa.f` and `CT_interface.f`. The existing
driver sets an integer COMMON point tag during its existing untimed100-point
output pass, and zero during initialization/warmup/timed traversal. Wrapper
entry logs a BEGIN record; immediately after the unchanged NinjaRS call, log
the exact inputs and raw outputs using TRANSFER to INTEGER*8 and hexadecimal
integer formatting. No floating decimal formatting, IEEE procedure import,
conversion, or precision change is added to amplitude routines. A missing
payload after BEGIN is an explicit early-return failure, never an omitted
sample. Source edits, exact before/after files and diff are retained.

Capture only points44 and80. Every captured record is tagged by global sequence,
point, raw CTMODE, explicit stage (1 nativeDP,2 compensatedDP,3 genuineMP), loop
group ID, squared split order, rank and denominator count. Retain all coefficients
after the existing parity convention, full supplied S/momenta/mass/scale inputs,
raw Laurent result, rational diagnostic and native status. Binary64 has one
64-bit word/component; binary128 has two native-order words with a 1.0 layout
header. Never infer correspondence from rank or sequence alone.

This fixed case has116 loop groups, one physical order and zero rotations in
both precisions. A validated stage must have the complete ordered tuple list
`(point,direction,loop_id,split_order,n,rank)` with no duplicate semantic keys;
direction is1/2 for DP CTMODE1/2 and QP CTMODE4/5. Match native, compensated and
QP records only by that full tuple and require their ordered keys to agree.
Unexpected modes, duplicate/missing calls, early returns, unsupported layouts,
nonfinite payloads or partial files stop interpretation. The capture has a
2500-record ceiling and the amplitude subprocess a600-second timeout; failures
and partial raw files are retained. There is no adaptive exclusion or rerun.

Before interpreting any record, require the instrumented clone's complete100
point rows (Born, finite, both poles, physical/total orders), accuracies and
statuses to equal the bound original exactly; require98DP218+2QP322 and rescue
points44/80. Rehash source/runtime/card/points before and after the run. This
tests observed state preservation; it does not assert that integer I/O is
universally invisible to every compiler/runtime. All timing fields are ignored.

Replay matrix for each matched input (all loop groups, both directions):

| Coefficients | Geometry | Reducers |
|---|---|---|
| capturedDP | capturedDP, bit-identical | nativeDP, compensatedDP, genuine256 |
| rounded capturedQP | capturedDP, bit-identical | nativeDP, compensatedDP, genuine256 |
| full capturedQP | exact promotion of capturedDP | genuine256 |
| full capturedQP | capturedQP | genuine256 (raw-output reproduction control) |

The last two rows distinguish coefficient rounding from reduction geometry.
Promote stored DP values exactly; use the installed ABI3 generic Fortran module
for genuine256 with its unchanged113-bit-precision-derived on-shell threshold
(`1e15 * 2^-112`) and 40-digit binary128 transport. DP uses the unchanged native
options/masters. The supplied matrices are already thresholded by MadLoop;
never rebuild them or adjust thresholds in the replay. Preserve full input words
and all three Laurent outputs/statuses for every cell. Rational results are
diagnostic only because generic and native residue bases use different splits.

Important limit: capturedQP coefficients were constructed at the normal
improved MP point, with MP parameters/Born interference. The mixed geometry
rows measure this particular combination, not pure coefficient arithmetic at
identical external kinematics. A separate promoted-DP-point numerator rebuild
would be needed to isolate that cause further; it is not part of this capture.
Loop-induced/per-helicity support, physical acceptance of reassembled mixed
answers and any new retry policy are outside this diagnostic.

The replay is separately held and bounded: at most464 matched records ×8
cells,900 seconds total, retain failures without selecting favorable loops.
Start it only after capture invariance/matching passes and root reviews the
record inventory and any kinematics differences. No new mixed return code is
assigned and forcedCTMode4 remains the genuine path throughout.
