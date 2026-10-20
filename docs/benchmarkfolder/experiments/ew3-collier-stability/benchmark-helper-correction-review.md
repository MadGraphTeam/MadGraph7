# Independent benchmark handling review

The generated multiple-precision model helper defect requires distinct
correctness and performance reference identities. A corrected reference must
not replace the original archived revision or be labelled as its unmodified
result.

1. Preserve revision `27147b3300f556a75661a886e92bdbe00558ee25`, its archived
   source, generated outputs, executable hashes and measured results. Keep its
   warmed original-Ninja runtime as the requested `1.05 ×` performance
   denominator. A helper-corrected reference's timing is a separate diagnostic.
2. Retain every observed original-result discrepancy, including the point,
   split order, coefficient, accuracy estimate and return code. Explain the
   demonstrated helper defect; do not discard QP points, accept their original
   numerical values as an oracle, or change stability thresholds to avoid them.
3. Construct separately named `original+mp-helper-fix` outputs. Record the exact
   original generation identity, helper-only patch, before/after source hashes,
   compiler flags, model provenance, unchanged parameter/point hashes and new
   executable hashes. The original archived source remains untouched.
4. Establish the helper correction independently: positive/negative real and
   imaginary axes, values distinguishable beyond double precision, general
   complex inputs and both sheet-shift directions should agree with direct
   binary128 analytic expressions. The trajectory constructors need independent
   mass/width and branch tests. Changing the model's `EP` parameter is not the
   same operation and is not justified by this helper defect.
5. Require the four corrected candidate backends—CutTools, Ninja, Collier and
   NinjaRS—to meet the original physical coefficient/order tolerance at every
   one of the same 100 fixed points. Compare each optimized legacy backend to
   its helper-corrected original implementation so the generator optimization
   and helper correction are tested separately.
6. Preserve the original all-target comparison as a failed diagnostic when its
   uncorrected QP values differ. Introduce an explicit, versioned correctness
   policy selecting the helper-corrected original references, rather than
   changing the meaning of an existing “passed” field or relabelling rows.
   The report should show original-to-corrected shifts and both identities.
7. Bind the corrected correctness evidence to the exact archives, generated
   source manifests, executable hashes, input hashes and protocol identity
   used in the final candidate measurements. Refresh representative parity
   and affected controls after the final generator/library changes.

The helpers are generic exporter code, so the impact audit must include
`loop_sm` QCD rescues as well as the EW model. Prior measurements remain valid
historical observations of their retained binaries; they do not automatically
certify newly generated corrected binaries.

This is a review recommendation, not a change to the frozen measurement
protocol, a numerical pass, a weakened tolerance, or authorization to mark the
overall integration goal complete. The final report must state this correction
and the associated policy revision explicitly.
