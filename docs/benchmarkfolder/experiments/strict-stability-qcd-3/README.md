# Strict common-stability QCD three-body calibration audit

The retained original CutTools, Ninja and Collier calibrations use the same
100-point `u u~ > z g g [virt=QCD]` stream, physics parameter card,
`MLStabThres=1e-9`, and `COLLIERRequiredAccuracy=1e-10`. All three pairwise
comparisons pass the unchanged benchmark correctness criterion across 200
point/order rows and all four output coefficients: Born, finite, single pole
and double pole. The summed order and sole physical split order are both
checked. The largest difference divided by the common point scale is
`1.00268e-9`; the largest fraction of the permitted coefficient tolerance is
`0.126731`.

The numerical and timed return-code histograms agree:

| Original backend | Stable DP points | Own QP rescue points |
| --- | ---: | ---: |
| CutTools | 48 (`211`) | 52 (`329`) |
| Ninja | 98 (`216`) | 2 (`328`) |
| Collier | 100 (`217`) | 0 |

These are calibration results, not repeated performance evidence. The extra
rescues remain part of full evaluation runtime. No phase-space point,
coefficient tolerance, or acceptance threshold was changed for this audit.

For diagnosis only, the previously retained candidate NinjaRS calibration at
the **different default settings** (`MLStabThres=1e-3`,
`COLLIERRequiredAccuracy=1e-8`) also agrees with every strict original backend.
Its largest point-scaled difference is `3.85647e-9`, and its largest fraction of
the coefficient tolerance is `0.675486`. Its 100 points returned DP code `218`.
This comparison does not establish common-setting correctness or performance
for NinjaRS; the stricter candidate measurement is still required.

`agreement.json` records every pair, the worst comparison, failure lists,
rescue counts, configuration values, and SHA-256 hashes of the copied raw
records. Its criterion is
`abs(a-b) <= relative_tolerance*max(abs(a),abs(b)) + scale_tolerance*point_scale`,
where `point_scale` is the largest absolute coefficient at that point across
all selected records and orders. Strict comparisons use only the three strict
original records to define that scale; the separately labelled diagnostic
comparisons also include the older NinjaRS record.
