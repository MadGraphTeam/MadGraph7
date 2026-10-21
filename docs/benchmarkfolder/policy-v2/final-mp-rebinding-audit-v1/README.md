# Independent explicit-QP rebinding audit

The audit reads the two completed ten-point direct-QP streams in
`../final-mp-rebinding-v1`, their already passed DP parents and actual private
source/library/executable inputs. It invokes no compiler, numerical worker or
test suite. `audit.py` exited zero on CPU 3; `receipt.json` retains the checks,
raw-file bindings, all comparison counts and maximum discrepancies.

Both QP clones differ from their DP source manifests only at
`SubProcesses/P0_gg_hh/check_sa.f`. Their exact sole source edit is the approved
public driver switch `FORCE_STABILITY_CHECK(.TRUE.)` to `.FALSE.`. The corrected
DP/MP updater is byte-identical to the ordinary helper with SHA256
`93e41b821058e89c04d01f72070406ede09a55cb37021b5995cb27635897ff9c`.
The ten fixed momenta, top-mass/scale schedule, physical cards, independent DP
and MP assignments before `COUP()` and context-finalization choice are retained.
The only reduction-card difference is `CTModeRun=-1` becoming `4`; backend 8,
the true QP-integrand flag and all thresholds remain fixed. Actual source,
full reducer signature `8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff`
and executable bytes match their recorded bindings.

Independent reparsing finds exactly twenty physical rows per stream: total
and split order 1 at each of ten points, with all four Born/Laurent coefficients.
Every QP point returns `120` with accuracy `-1`, the direct-mode unavailable
stability estimate. The DP parents return `218`. These observations establish
explicit QP execution and numerical agreement; they do not claim automatic
stability acceptance, rescue or performance acceptance.

The unchanged physical comparison policy passes all **240 scalar comparisons**:
80 retained-QP versus DP, 80 finalized-QP versus DP, and 80 retained-QP versus
finalized-QP. The QP lifecycle variants have exactly equal coefficient values.
The largest QP/DP discrepancy is `0.00032883178657856993` of the allowed
tolerance, at point 6's total single pole. Every comparison uses all four
coefficients and the common point's largest Born/Laurent magnitude; no poles
or split-order rows are omitted. The original three comparison summaries
match the independent recomputation exactly.

This is a fixed ten-point state-rebinding diagnostic. It does not replace the
core correctness, stability or runtime campaign. `manifest.json` and
`retained-files.txt` seal the independent audit without compiled products.
