# Private QCD topology and Born-coordinate comparison

The six authorized replays completed without an integrity failure. They do **not** qualify a production fix or a complete physical amplitude. The source-stage README and index remain unchanged; this file records the later execution and offline analysis.

Each of three private arms was compiled once using the reviewed historical ordered compiler/link arguments and a recorded environment allowlist. Each arm then ran backend9 and backend8 once, on CPU11, one thread, with a180-second process cap and the shared numerical lock. There were no retries, Rust builds, production edits, card changes, or threshold changes. All calls loaded the historical TIRust DSO `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.

The stream was ordinary/stress/ordinary. All six executions preserve the ordinary numerical rows, raw floating-point words, return codes, and IEEE state exactly against the frozen original-b control. The repeated ordinary point also matches. Trace acknowledgements and footers have zero faults and I/O failures. All protected source, native-object, input, card, resource, and runtime-library hashes match before and after. No performance claim follows from the diagnostic wall times.

## What changed in the stress calculation

The topology arms apply the old on-shell clamp branches only to statically certified single-external-leg or complementary channels. Composite and unknown channels retain their computed value. This is a uniform graph-derived rule, not a list of the19 problematic pairs. The C arms use the actual existing `SET_MP_PS` repaired coordinates for the rescued Born and the QP loop. They change effective coordinates relative to exact promotion of the original binary64 input; they are not unchanged-Cartesian-input results.

| Arm | NinjaRS served /232 | NinjaRS host code | TIRust served /232 | TIRust host code | NinjaRS double-pole relative error |
|---|---:|---:|---:|---:|---:|
| original-b, retained control |212|420|170|420|0.3459232136|
| topology-b |232|322|181|420|0.0003530392044|
| original-c |212|420|170|420|0.3461540469|
| topology-c |232|322|181|420|1.32950e-15|

The independent check in the final column is the universal QCD double pole, using each arm's actual host Born and captured coupling: `Born * -(26/3) * G²/(8π²)`. The two incoming quarks and two outgoing gluons contribute `2CF+2CA=26/3`; the Z is colourless. This tests one Laurent coefficient only. No independent finite or single-pole truth is available here.

In particular, topology-b's checked host code322 does not prevent its double pole from missing the3e-6 external physics criterion. Aligning Born with C is necessary for this normalization check. The measured Born changes from `4.30746489426097521e19` to `4.30898559824016835e19`; the three loop coefficients are bit-identical between the B/C variants at fixed topology policy. The combined topology-c NinjaRS result passes the double-pole check, while its finite and single coefficients remain unqualified. TIRust still returns420; its incomplete aggregate is not an accepted amplitude.

`analysis1/handoff.json` establishes that every stress replay has one preparation triplet, exact original-DP promotion before repair, and one bit-identical C point after repair. All232 QP calls per replay consume that same C in both momentum COMMON blocks. Eight of20 coordinates change. C is bit-identical across all old and new arms. Four topology-only input comparisons cover232 calls each and find no changes in numerators, shifts, masses, scales, order, contexts, or pre-clamp matrix arithmetic. The Born helper's use of C is established by the frozen source handoff, not a separately instrumented Born-argument event.

## Matrix and refusal evidence

The offline matrix analysis covers464 calls and1,908 unordered pair rows per arm. All observed numerators are finite. Each direction has116 groups,477 pairs and19 small composite targets. The topology policy preserves all19 nonzero computed, supplied and Cartesian target channels in each direction. It removes only the composite clamps: all remaining taken branches are the original massless-energy branch3 on certified external/complement channels. No massive-mass branch1/2 is taken. The64 eligible massive-Z pairs per direction are not clamped.

For TIRust, original-c has62 refusals:22 supplied Gram determinants are exactly zero even though their Cartesian counterparts are not; the other40 fail a conditioning guard. The topology arms have51 refusals, all reported as ill-conditioned and all with nonzero exact supplied and Cartesian determinants. The largest relative difference between these determinants is5.75333e-16. Each refused group's pair-level differences are decomposed exactly into routing rounding, invariant arithmetic, mass subtraction, actual clamp change, and reconstruction rounding in `analysis1/adjudication.json`. This separates representation arithmetic from the removed composite clamping; it does not prove a tensor error or certify a precision requirement.

`analysis1/guard-source-audit.json` binds every exact native error to the historical installed source. The full-QP path uses MPFR `Float`,113-bit parsed inputs,256-bit arithmetic, direct native scalar masters, and fixed threshold `2^-136 = 1.14794370197e-41`. It has no small-Gram expansion, double-double partial rescue, or precision ladder enabled. Denominator-relation and lightlike-bubble plan variants remain active.

The guard compares `condition^subset.coefficient_rank` with the threshold, but the error message prints the **unpowered** condition. The condition is the smaller of the Hadamard-relative determinant and the determinant-root divided by the recursion scale. Source/rank propagation and the recorded inequality determine exponent3 for11 calls and exponent4 for20 calls;20 triangle calls permit either3 or4 because runtime relation profiles were not logged. For example, CT4 group61 prints approximately1.998e-19, but its box exponent4 gives approximately1.593e-75, below the fixed gate. These are heuristic refusals, not proof that512 bits are needed or sufficient. No guard was changed.

## Mass and model consistency

The remaining51 TIRust refusals comprise39 massless calls, six bottom boxes, and six top boxes. Generated denominators square DP masses before QP promotion. The numerator uses the existing MP model aliases. All observed MP masses equal exact DP promotion, consistent with the model's REAL*8 input buffer.

Bottom mass squared differs by approximately+1.74083e-15 between DP-square-then-promotion and the exact square of the observed MP mass. This affects28 calls per topology arm: six refusals and22 successes. Top mass173 has exact square29929. Thus45 of51 refusals lack this internal-mass-square difference. The external Z repair uses the existing MP mass; its square differs from the promoted DP square, but no massive-Z clamp is taken. `analysis1/mass-source-audit.json` records source ancestry, exact words and derived rational squares; derived squares are not presented as captured runtime variables.

## Rechecking the retained evidence

Run from the MadGraph repository root:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/verify.py
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/verify.py --live
```

The first command verifies the portable inventory, frozen controller, build/run records, and analysis bindings. `--live` additionally checks the recorded external native/source/runtime paths. Neither command compiles or runs a reducer. The frozen analysis scripts retain their exact inputs and one-shot output protection; they must not overwrite the captured attempt. Independent review scope is recorded separately when available. This private result remains outside production and the acceptance campaign.
