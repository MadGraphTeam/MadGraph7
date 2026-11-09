# Independent complex-mass evidence review

Reviewed `tools/tirust_complex_stress.py`, its fixture/parser tests, and retained
complex-mass experiment summaries. This review ran no physical timing.

The summary gives `failure.json` priority over a passing summary. Nonfinite or
rejected rows are retained as failures and excluded from accepted comparisons.
Fixed-mode QP results with status 120 are explicitly unchecked diagnostics;
they do not establish independent MadLoop stability certification. Common-row
comparisons identify the original point indices and leave omitted points
unqualified.

The original subset comparison joined rows by index without checking common
physical inputs. Following this audit, the runner verifies each retained
stream against the frozen parent point hash, subset indices and point payload,
own point-file hash, and saved physical parameter-card hash. Joined diagnostics
require the same parent stream and physical card and retain those digests.
I independently validated these bindings for all eleven raw streams available
at review time.

The aggregate comparison flag is now `diagnostic_numerical_agreement`.
`primary_policy_agreement` is unavailable until all five reducers complete
the full 21-point stream with automatic precision and the primary on-shell
threshold. Neither alternate-`OSThres` QP diagnostics nor passing ordinary or
moderate subsets can satisfy that gate. The reviewed summary records ordinary
five-backend coverage and passing diagnostic comparisons, while full coverage
is false and primary-policy agreement is null. This is consistent with the
retained extreme-forward failures.

The separate complex-mass experiment explicitly keeps its original
`MLStabThres=1e-3` policy. Its numerical results are not performance-acceptance
evidence for the main campaign's later frozen `1e-6` policy.

Verification: all five tests in
`tests.unit_tests.various.test_tirust_complex_stress` passed, including physical
point conservation, the small-Gram ladder, and failed/nonfinite row handling.
