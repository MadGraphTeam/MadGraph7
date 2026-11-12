#import "style.typ": *
#let data = json("analytic-repair-data.json")
#show: body => report-style(body, date: "12 November 2026", revision: "unapplied Fortran candidate · control 0fcd2cd873d6")
#set heading(numbering: none)

= Analytic Fortran phase-space repair

#note(fill: warm)[*One-process shared-host study; patch remains unapplied.*
  Original #raw("0fcd2cd873d6") control versus a same-flag Fortran candidate.
  Hua-Sheng’s precise local performance acceptance remains pending.]

The 2→1 process $u overline(u) -> Z$ with QCD corrections uses #raw("loop_sm") and explicitly selected TIRust backend 9. The ordinary sample contains 100 saved longitudinal boosts and retains independent MadLoop stability checks. Its measured tensor topology has three propagators and rank two. Only the Fortran repair unit is replaced; all compiler flags and other numerical objects remain unchanged.

#table-style[
  #table(columns: (2.5fr, 1.2fr, 1.2fr, 1.25fr), align: (left, right, right, right),
    inset: 6pt, stroke: table-stroke, fill: table-fill,
    [Complete warmed amplitude], [Median µs/point], [Unscaled MAD µs/point], [Accepted 219 calls],
    [Original control], [#fixed(data.arms.original.median_microseconds_per_point, 4)],
      [#fixed(data.arms.original.unscaled_MAD_microseconds_per_point, 4)], [89,600],
    [Unapplied Fortran candidate], [#fixed(data.arms.candidate.median_microseconds_per_point, 4)],
      [#fixed(data.arms.candidate.unscaled_MAD_microseconds_per_point, 4)], [89,600],
  )
]
#text(8.3pt)[Each arm has seven batches of 128 full 100-point traversals (12,800 calls per batch), on nonexclusive CPU 12. MAD is the median absolute deviation of batch time per point; it is unscaled, without conversion to a standard error.]

== Seven matched-history pairs

#table-style[
  #table(columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: right,
    inset: 4.6pt, stroke: table-stroke,
    fill: table-fill,
    [Pair], [1], [2], [3], [4], [5], [6], [7],
    [Original s], ..data.pairs.map(p => [#fixed(p.original_seconds, 4)]),
    [Candidate s], ..data.pairs.map(p => [#fixed(p.candidate_seconds, 4)]),
    [C/O], ..data.pairs.map(p => table.cell(fill: ratio-fill(p.candidate_over_original))[#text(fill: if p.candidate_over_original < 1 { fast-green } else if p.candidate_over_original > 1 { slow-red } else { navy })[#fixed(p.candidate_over_original, 4)]]),
  )
]
#text(8.3pt)[Order is original→candidate for odd pairs and candidate→original for even pairs. Every actual batch duration is retained in the machine-readable data.]

#note[
  *Median paired C/O: #fixed(data.paired_median_candidate_over_original, 4).* 
  Ratio of the two arm medians: #fixed(data.ratio_of_arm_medians_candidate_over_original, 4), a different statistic.
  All 14 batches span #fixed(data.minimum_batch_seconds, 6)–#fixed(data.maximum_batch_seconds, 6) s.
  All 179,200 timed statuses are accepted DP code 219.
]

== Numerical scope and limits

Two standalone mains complete 544 outer DP/QP repair-wrapper invocations across 136 fixtures per precision and arm. DP outputs match bit for bit; 94 QP fixture results differ in 233 component words while satisfying the frozen physical-residual limits. Refused-input outputs, diagnostics and observed IEEE states preserve the original behavior. These tests do not instrument actual fast-branch coverage.

Fresh learning and final 100-point amplitude runs retain exact coefficient, accuracy and public-status words on all 100 physical points. Each arm returns 200 rows (overall plus the sole split order), containing 800 coefficient and 200 accuracy fields. The 14 timing batches retain only their final 100-point numerical outputs; their status histograms cover every timed call. The physical model’s MP mass remains its existing binary64-to-binary128 lift.

This is a descriptive comparison against the current control, separate from the starting-revision default-chain and eight-process acceptance campaign. Broader process/model qualification and an independent finite-answer oracle remain open. Recorded IEEE states establish observational parity, without a trap-freedom claim. The source patch remains unapplied; these results do not promote a backend or establish formal local runtime acceptance.

#text(7.8pt, fill: ink2)[Reproduction and full-precision values: #raw("README.md"), #raw("derive_timing.py") and #raw("analytic-repair-data.json"). Evidence index #raw("f8bc6ce06201…"); timing review #raw("9e10f47d4599…"). Typography and timing palette follow the preserved pyAmpliCol source, revision #raw("7266c1600f6e…").]
