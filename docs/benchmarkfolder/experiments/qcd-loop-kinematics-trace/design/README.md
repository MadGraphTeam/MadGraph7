# Source-only design for the QCD soft/collinear loop trace

This package proposes an observational trace. No generated Fortran, production source, native library, card, phase-space point, or numerical gate has been changed, and no new native evaluation has been run. The separately frozen Born prototype remains unpromoted: backends 8 and 9 both reject the stress amplitude with status 420, and their double poles differ from the rescued-Born prediction by about 34.59%. A finite Born does not establish a correct loop amplitude.

## Question and existing evidence

Both QP adapters call `ML5_0_MP_BUILD_KINEMATIC_MATRIX` in the historical generated `CT_interface.f`. Its ordered operations are:

1. Compute `DIFFSQ=(q_i-q_j)^2` in binary128 and initialize `S_ij=DIFFSQ-m_i^2-m_j^2`.
2. If the relative distance to nonzero `m_i^2` is below `OSThres`, assign `S_ij=-m_j^2`.
3. Independently apply the analogous `m_j^2` test, potentially overwriting that assignment.
4. If `Eref=((abs(q_i0)+abs(q_j0))/2)^2` is nonzero and `abs(DIFFSQ/Eref)<OSThres`, assign `S_ij=-(m_i^2+m_j^2)`, potentially overwriting either massive test.

The last test can erase a nonzero composite invariant. The retained original-input Born audit proves `(p_1-p_5)^2=-5.00088252041288508e-14` for the exact binary64 inputs. Its smallness alone does not identify an external mass shell. The frozen card uses `OSThres=1e-8`. The actual loop QP inputs must nevertheless be captured: the historical host calls `MP_IMPROVE_PS_POINT_PRECISION`, so the exact original-input Born invariant must not be silently substituted for the QP loop's repaired invariant.

The EW investigation establishes a related but distinct caution. Its repaired external QP momentum uses a QP model mass, while internal mass squares are computed in DP and then promoted. Snapping to the latter can create a small supplied-distance/Cartesian-shift inconsistency even for a legitimate single-leg shell. Its rank-one diagnostic explains a tensor discrepancy through that inconsistency; it does not establish a QCD cause or qualify a general correction. See `../../ew-threshold-adjudication/tensor-replay/gram-consistency/` and the corresponding geometry supplement.

The private Born rescue deliberately uses an exact promotion of the original DP inputs, whereas the existing QP loop numerator is constructed after `SET_MP_PS` repairs those inputs. Thus the rescued Born and QP loop may describe measurably different effective kinematics near the collinear boundary. Tiny coordinate changes are insufficient evidence of equivalence. The trace must quantify this before attributing the pole discrepancy to thresholding.

## Capture boundaries

The future isolated instrumented export should retain the original ordinary/stress/ordinary driver, original cards, both backend selections, historical native libraries, and the reviewed private Born prototype. Bind all source, object, executable, library, input and card hashes before and after. Work products belong under ignored `.tirust-build`, not this evidence tree. Compilation and execution still require the parent-coordinated numerical slot; this design authorizes neither.

| Boundary | Required record | Reason |
|---|---|---|
| Driver and `SET_MP_PS` | Point sequence, input DP words, promoted pre-repair QP words, repaired `MP_PS`, actual rotated `MP_P`, model DP/MP masses, scale and card values | Separate the physical input from existing loop repair and rotation. |
| Generated `LOOP_*` dispatcher, before backend call | `LOOPNUM`, `SQUAREDSOINDEX`, rank, `NLOOPLINE`, library, CT mode, `WE`, `PAIRING`, DP `PL`, QP `MP_PL`, unsquared mass arguments and DP `M2L` | Give every denominator edge an explicit source routing and preserve the DP-square convention. |
| Adapter before/after `MP_SWITCH_ORDER` | Ordered edge momenta and squared masses, explicit permutation/sign map | Reversal is not inferred from adjacency or floating values. |
| Adapter before the matrix builder | Actual Cartesian shifts including the first residual shift | The first shift is the sum of all edges, not an imposed exact zero. Preserve the original accumulation order. |
| Matrix builder | Raw `DIFFSQ`, original `S_ij`, each taken branch in execution order, `Eref`, final complex `S_ij` for both ordered pairs | Identify the exact operation that changed an invariant and later overwrites. Diagonals have their own tag. |
| Adapter before backend call | Actual reconstructed real distances `Re(S_ij+m_i^2+m_j^2)`, Cartesian shifts, complex masses, actual scale, numerator coefficients | These are the actual backend inputs; recomputing them later can hide cancellation. |
| Adapter after backend call and contraction | Integer status, existing stable flag, TIRust error text when failed, raw output words, mapped moments or NinjaRS result, final DP conversion, host failure flag | Distinguish a backend refusal, nonfinite numerator, nonfinite tensor, contraction failure, and conversion overflow. |
| Host | Existing reduction-failure flags, direction accuracy, current-call QP certificate, returned H/T/U, Born and Laurent coefficients | Reproduce the retained rejection and avoid treating a zero direction estimator as success. |

NinjaRS and TIRust differ slightly in scale construction (`REAL(MU_R**2,16)` versus `REAL(MU_R,16)**2`) and normalization precision. Preserve and record the actual values rather than harmonizing them during this observation. TIRust separates tensor output from contraction; NinjaRS takes the numerator and returns the contraction. A matched unit-numerator tensor replay would be a later, separately bounded diagnostic, not part of the first trace.

## Compare the effective kinematics and Born normalization

Keep four distinct records: A, the original binary64 physical input; B, its exact binary128 promotion used by the rescued Born; C, the post-`SET_MP_PS` repaired `MP_PS`; and D, the actual `MP_P` after any rotation used by the loop numerator. Never overwrite one with another or call C the same numerical input as B. The original input file remains the sole user-supplied event.

For each external leg report its exact rational mass-shell residual using both the promoted DP mass square and the QP model mass square, together with the exact total four-momentum residual. For every source-proven composite channel, report its invariant separately at B, C and D, its signed absolute change, and `2*abs(x_C-x_B)/(abs(x_C)+abs(x_B))` when the denominator is nonzero. Handle exact zeros explicitly. Include all pair channels used by traced groups and, especially, `p1-p5`. Report coordinate changes and energy-scaled changes alongside these channel-relative changes; small coordinates alone must not decide equivalence. Also report B and C corner-prefix evaluation with the actual binary128 operation order when it is available, separately from exact-fraction invariants.

The first observational pass records the actual Born, normalization and QP numerator words without evaluating extra amplitudes in the host. A subsequently authorized, separate Born-only replay should evaluate the same reviewed QP Born aggregate at B and C (and D if rotation differences matter), with identical model parameters, helicities, polarization, split-order weights and couplings. Keep the raw QP Born values as well as their DP conversions. Do not call this extra aggregate inside the observed host because it could mutate shared wavefunctions, filters or coupling state.

Compute the analytic double-pole prediction independently from each Born value using the corresponding unchanged coupling convention. Report the rejected loop pole relative to both predictions, explicitly recording which kinematics built the loop numerator. Agreement with the repaired-input prediction alone does not qualify the original-input amplitude; disagreement with the original-input prediction alone does not establish a threshold defect. A future repair needs a coherent input convention for Born aggregation, interference numerator and pole normalization, with separate evidence that any physical input projection is authorized and numerically appropriate. No such projection or change is part of this design.

## Classify channels from generated routing

Build each tree-wavefunction momentum as an integer coefficient vector over the five physical external momenta, using the generated external-wavefunction calls and the actual DHELAS `%P` assignments. Propagate those vectors through the generated `WE`/`PAIRING` grouping, edge negation, direction reversal and prefix accumulation. Store the source lines proving every edge. Do not infer a single leg from adjacent propagator indices or from numerical closeness to a mass.

Classify each difference vector into:

- **Direct single leg:** exactly plus or minus one external unit vector. Record that external leg's known mass and both DP/MP mass-square conventions.
- **Single leg only modulo conservation:** equal to a single leg after adding a multiple of the signed total-conservation vector. Preserve the unreduced vector and its residual; finite precision does not make the equivalence exact.
- **Composite physical channel:** a sum/difference of multiple external legs that is not a single-leg class. Report its exact rational invariant and any soft/collinear scaling without assigning an on-shell value.
- **Zero or conservation residual:** identify algebraic zero separately from a formal total-conservation vector and from an actually nonzero residual shift.
- **Unresolved routing:** fail the classification audit; do not guess from values.

For every class retain both the literal vector and a canonical label modulo the conservation relation. Reconstruct the momentum from the captured external binary128 words and compare with the captured shifted-momentum difference. The difference quantifies accumulation order and phase-space repair; it must not be silently zeroed. The post-repair external vectors, not the original binary64 vectors, are authoritative for this comparison with actual loop inputs.

## Make the observer numerically passive

Insert records inside already-taken clamp branches; do not reevaluate their floating conditions or replace their formulas. Copy existing floating variables through `TRANSFER` into integer words. Keep raw complex components and signed zeros, including nonfinite output bit patterns. Decode binary64/binary128 words into exact Python fractions offline. Decimal text is an additional presentation, never the input identity.

Use fixed, bounded storage and integer-only output after the sampled host call. Preserve the current IEEE status around any logging helper, with no change to rounding or enabled traps. Do not temporarily suppress a caller trap. Avoid additional floating arithmetic, `IEEE_IS_FINITE` scans, or floating formatting in the instrumented hot path. The trace may observe the existing flag state but must not clear a legitimate reducer flag. A logger error, overflow, or dropped record invalidates the trace instead of changing the host result.

Record every measured call, including rejected calls and early returns. Disable records during unchanged initialization. Reserve at least 4096 group-call records and explicit overflow counters; the process has 116 loop groups, but filters, repeated split orders and stability retries mean an assumed count of `116*2` is insufficient. Index records by point sequence, backend, CT mode, group, split order and invocation number. A final index checks nesting/completeness rather than assuming a fixed traversal.

## Bounded qualification and interpretation

After source review and scheduling approval, compile the private observer with the original flags on CPU 11. Replay backend 9 and then backend 8 under the shared numerical lock, with a separate 180-second process bound and raw cleanup/provenance logs. No precision change, threshold change, input repair, cache-policy change, or backend substitution is included.

Require exact ordinary and stress output bits, statuses, accuracies and caller IEEE flags to match the frozen uninstrumented prototype for each backend. The expected stress result is still rejection 420; observing it is not a new acceptance. Require zero dropped records, correct direction routing, complete source bindings and unchanged original artifacts. If instrumentation changes behavior, retain the failed attempt and stop causal interpretation.

Offline analysis then lists every changed nonzero invariant, its topology class, pre/post values, clamp sequence, supplied-distance versus Cartesian difference, and all failed groups. Compare the same channel across forward/reverse order and both backends. A snapped composite channel is evidence of an input transformation; it becomes evidence for the amplitude failure only after a separate controlled replay demonstrates the relevant effect. A supplied/Cartesian mismatch alone likewise does not prove the source of the 34.59% pole discrepancy.

The effective-kinematics comparison and Born normalization analysis are a prerequisite for claiming any threshold-policy correction is sufficient. Until then, classify the causes as unresolved even if a composite clamp is observed.

Any later correction must protect genuine external shells and DP/MP mass conventions explicitly. Globally lowering `OSThres`, globally recomputing every distance, or relaxing a Gram gate is not proposed. Neither the current design nor a successful observational trace qualifies a production change.
