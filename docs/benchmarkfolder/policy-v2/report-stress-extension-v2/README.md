# Reviewed optional-reference failure visibility

The resumed forty-test report run in `report-stress-extension-v1/attempt-1`
passed. Independent source review then found a presentation defect: the stress
table displayed an optional helper cohort's agreement status even when a later
helper collection had failed. A passed agreement could therefore conceal the
separate retained failure.

The final template always displays the optional reference's primary status,
its independent agreement status when available, and the number of retained
failures. The explicit synthetic PDF fixture now contains a failed optional
helper collection alongside a passed agreement. All forty report tests pass
again, including that layout and byte reproducibility across ambient creation
epochs. `attempt-2/` retains this rerun and exact unchanged before/after hashes.

The earlier v1 package and its seal remain unchanged; its recorded old source
identities are not relabelled as this correction. This package verifies the
final report source/test/template and style hashes, all forty-test log records,
and the retained first attempt's result/logs and manifest. Independently owned
readers have their own final combined review.

No numerical process, scalar integral or acceptance benchmark ran in these
software checks. Existing core and ggH/ggHH gates stay unchanged; the two new
stress cases carry diagnostic times and no new performance threshold. The
top-level historical `benchmark.typ` and `benchmark.pdf` remain untouched.
