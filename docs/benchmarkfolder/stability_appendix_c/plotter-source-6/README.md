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

This successor preserves the ordinary legacy arithmetic exactly: range divided
by abs(sum(probes)/3). Only overflow and subnormal-risk inputs use rescaling.
Vertical markers belong to the plotted metric: the raw-DP paper diagnostic
uses 1e-3; coefficient disagreement uses 3e-6 and 1e-8. Post-rescue host
estimates show a separate paper reference and an actual host threshold only
when explicitly bound in input provenance. Marker meanings appear in the
legend. Exact-zero tail fractions are also omitted from log-y drawing; this
is distinct from exact-zero point metrics, which are counted separately.
Backend IDs must be integers, excluding booleans and floating point aliases.

Presentation-only successor: wider individual panels, explicitly wrapped
shorter titles, and a dedicated footer row avoid clipping and annotation
overlap. PDF creation/modification timestamps are omitted for reproducible
files. Scientific metric, validation, threshold, tail and ratio functions are
unchanged from source2. The retained patch documents the complete source diff.

Final presentation correction: upper tail and lower ratio panels share the
same declared threshold domain. A supplied explicit reference label appears
in the disagreement subtitle. These changes do not alter any point value,
metric, threshold, count, ratio or uncertainty calculation.

Empirical-staircase correction: the plotted grid now contains every observed
finite positive metric in [1e-16,1], every exact labelled threshold, and the
background logarithmic grid. The strict > definition therefore drops exactly
at tied metric values, rather than at a later plotting-grid bin. The QCD n3
3e-6 target has zero exceedances for every lane; no positive tail is drawn
through it. Earlier source4 retains the same point data and formulas but its
coarse display grid could extend a 1% plateau past the true final jump.

This source6 presentation successor adds exact-zero and unavailable/N counts to
every backend legend label, including a backend with all points unavailable.
The footer identifies both counts. The source5 statistical functions and all
plotting calculations are unchanged. Two mocked plotting tests inspect the
actual emitted labels without importing matplotlib or writing plot artifacts.
The raw caption specifies host forced-DP/card-selected backends; it does not
observe reducer-internal precision. Dotted curves are explicitly upper bounds
from unavailable outcomes, not measured failure fractions. An entirely
unavailable lane has unknown bounds [0,1] over all 100 intended points.
