# Closed-input stability plotter

This immutable source packet renders only an explicitly named JSON data file.
It does not launch native evaluation, discover a latest attempt, validate an
amplitude, or grant numerical acceptance. It supports separately labelled raw
forced-DP consistency, accepted post-rescue composite estimates, and accepted
finite-coefficient reducer disagreement. These quantities are never labelled
independent true errors.

All five intended backend lanes share physical point IDs. Duplicate answer
orders are not extra points. A missing value stays in the intended denominator;
the reported tail bounds are [known exceedances / N, (known exceedances + missing) / N].
Threshold comparison is strict >. Ninja ratios are unavailable when the
Ninja tail has no exceedances or either population is incomplete. Valid zero
ratios appear on a linear lower axis. Exact-zero metrics are counted and
omitted from logarithmic y drawing, without changing the raw values.

For N=100 the empirical resolution is 0.01; no population confidence claim is
made. Accepted-output estimates include whatever rescue was performed and do
not supply missing pre-rescue DP values. The finite comparison metric is
abs(a-b)/abs(a+b), with explicit zero-denominator and overflow treatment.
The three-probe internal metric is range / abs(mean), kept distinct.

Tests: `python3 -B -m unittest discover -s THIS_DIRECTORY -p test_plot_metrics.py`.
Rendering: `python3 -B THIS_DIRECTORY/plot_metrics.py INPUT.json --output-prefix FRESH_PREFIX`.
Output is vector PDF, PNG, and JSON. Existing artifacts are never overwritten.
