# Correct a backend-specific annotation in report postprocessing

Completed actual ggH, ggHH and top-threshold collections exposed a reader bug:
`shared_numerical_settings` contains the derived annotation
`selected_backend_qp_integrand=true` for CutTools, Ninja and NinjaRS. Collier has
no matching QP-integrand option, so its metadata omits that annotation. The
supplement and stress readers compared those dictionaries literally and rejected
valid cross-backend inputs after successful collection.

The existing core policy already handles this distinction correctly. The new
shared analysis helper follows that same rule: require the exact Boolean `true`
for non-Collier QP integrands, enforce the same common threshold/pole settings,
then remove only that derived annotation for cross-backend equality. Every other
physical/numerical key stays in the comparison and raw dictionaries stay intact.
The stress reader reuses this helper for ordinary/control/reference comparisons.

Only the two analysis readers and their focused MG7 tests change. No worker,
collection function, protocol, core policy, configuration, numerical arithmetic,
threshold, point, sample count or final acceptance gate changes. Collection
snapshots retain the actual code hashes used at execution. This analysis fix is
kept separate from the physics collection revision and is committed only after
the corresponding raw collection milestone, as directed by root.

`source-before/` preserves the two original readers. `change-record.json` records
all four before/after source hashes. The focused software run retains all eighty-
one report/supplement/stress/precision tests, including five new regression cases
for the actual Collier/non-Collier metadata shape, strict QP integrands, unchanged
raw inputs and physical-setting mismatches. Its logs are in `software-attempt-1/`.

`analyze_actual.py` runs no physics and invents no core dataset identity. It uses
the existing checked source bindings and retained raw measurements from all four
actual candidate/helper collections, applies the corrected settings rule, and
recomputes all twenty-one physical backend pairings at 100 ggH points, 100 ggHH
points and twelve threshold points. All 35,616 component comparisons pass and
match the earlier disclosed numerical diagnostic exactly. Maximum tolerance
fractions are respectively 0.009391, 0.013055 and 0.020135. Accuracy estimates,
observed discrepancies and final-status histograms remain separate in the new
`actual-data-analysis.json`.

The first rejected analysis, original diagnostic and raw publication seal remain
unchanged and hash-bound. The new result explicitly keeps
`final_report_validation_passed=false` and `timing_acceptance_evaluated=false`:
complete final report assembly still needs the actual core dataset identity and
the unchanged final timing campaign. No numerical rerun is needed to repair
this postprocessing error.
