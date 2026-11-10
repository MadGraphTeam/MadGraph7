# Raw-DP stability figures

Host forced-DP, card-selected backends; reducer-internal precision and actual backend are unobserved. Each process has 100 cut-qualified physical points (1% empirical resolution). Dotted upper bounds account for unavailable outcomes; they are not measured failure fractions. Incomplete populations and zero Ninja exceedances mask ratios. Internal consistency is not a true-error bound; no performance claim is made.

Each figure uses one case from the complete copied point-metrics.json, retaining its provenance and the SHA-256 of the complete four-case dataset. All five intended reducers remain present, with zero and unavailable counts.

Verification requires explicit data and frozen plotter roots. It checks figure bytes and recomputes statistical JSON without importing matplotlib or reading raw journals.
The returned figure-index SHA must be checked against its independently admitted pin; self-consistency verification alone does not establish admission.
`python3 -B figures.py verify FIGURE_ROOT --data-root DATA_ROOT --plotter-root PLOTTER_ROOT`
