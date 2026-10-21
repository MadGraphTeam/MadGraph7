// NRS-first authority; shared typography/palette remain in the unchanged style.
#import "style.typ": *
#let data = json("report-data.json")
#show: body => report-style(body, date: data.generated_utc)
#let status(s) = text(weight: "semibold", fill: (
  passed: rgb("#287240"), failed: rgb("#b44339"), unresolved: rgb("#966016"),
  pending: grey, "accepted-observation": navy,
).at(s, default: grey), s)
#let decision(passed, states) = if passed { "passed" } else if "failed" in states {
  "failed"
} else if "unresolved" in states { "unresolved" } else { "pending" }
#let numerical-state = decision(data.nrs_correctness_passed, data.cases.map(c => c.agreement.status))
#let performance-state = decision(data.nrs_performance_passed,
  data.cases.map(c => (c.performance.status, c.timing_agreement.status)).flatten())
#let full-state = decision(data.nrs_full_qualification_passed,
  (data.nrs_core_status, data.runtime_provenance.status) + data.qualification_obligations.values().map(q => q.status))
#let publication-state = decision(data.final_publication_passed,
  (full-state,) + data.publication_obligations.values().map(p => p.status))
#let sci(x) = if x == none { [—] } else if x == 0 { [0] } else {
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  [#fmt(x / calc.pow(10, e), digits: 3) × 10#super(str(e))]
}
#let label(key) = (
  "qcd-1": [$u overline(u) -> Z$], "ew-1": [$u overline(d) -> W^+$],
  "qcd-2": [$g g -> t overline(t)$], "ew-2": [$e^+ e^- -> mu^+ mu^-$],
  "qcd-3": [$u overline(u) -> Z g g$], "ew-3": [$e^+ e^- -> t overline(t) gamma$],
  "qcd-4": [$u overline(u) -> Z Z g g$], "ew-4": [$e^+ e^- -> mu^+ mu^- gamma gamma$],
  "supplement-gg-h": [$g g -> H$], "supplement-gg-hh": [$g g -> H H$],
).at(key, default: key)
#let maximum(c) = if c.agreement.checks.len() == 0 { none } else {
  c.agreement.checks.map(r => r.max_tolerance_fraction).fold(0, calc.max)
}
#let timing-card(c, b) = {
  let t = c.timings.candidate.at(b)
  block(width: 100%, fill: ratio-fill(t.ratio), inset: 6pt, radius: 2pt)[
    #text(size: 7pt, fill: grey)[µs / point]
    #linebreak()
    #text(font: "DejaVu Sans", size: 10.8pt, weight: "bold", fill: navy)[#value-fmt(t.median_us)]
    #linebreak()
    #text(size: 7.1pt)[#if t.ratio == none { [Timing pending] } else { [×#fmt(t.ratio, digits: 2) vs original Ninja] }]
    #linebreak()
    #text(size: 6.9pt, fill: grey)[#t.repetitions retained timing repetitions]
  ]
}
#let grid(rows) = table(columns: (.95fr, 1fr, 1fr, 1fr, 1fr), align: top,
  inset: (x: 4pt, y: 7pt),
  stroke: (x, y) => (top: if y > 1 { .4pt + luma(225) } else { none }),
  fill: (x, y) => if y == 0 { navy } else if calc.rem(y, 2) == 0 { rgb("#f5f7f9") } else { none },
  table.header(header[Process], header[CutTools], header[Ninja], header[Collier], header[NinjaRS]),
  ..rows.map(c => (
    [#text(size: 9pt, weight: "semibold")[#label(c.id)]\ #text(size: 7pt, fill: grey)[#c.id]],
    ..("1", "6", "7", "8").map(b => timing-card(c, b)),
  )).flatten(),
)

#align(center)[
  #text(29pt, fill: navy, weight: "bold")[NinjaRS in MadLoop]
  #v(2pt)
  #text(12pt, fill: grey)[NRS-first validation · full obligations · external findings]
  #v(3pt)
  #text(8pt, fill: grey)[#data.generated_utc]
]
#v(8pt)
#note[
  *Independent results stay distinct.* NinjaRS core #status(data.nrs_core_status).
  Numerical correctness #status(numerical-state).
  Paired performance #status(performance-state).
  Full qualification #status(full-state).
  External-backend findings #status(data.external_backend_comparison_status).
  Final publication #status(publication-state).
  A passing core does not certify installation/export, higher precision, loop-induced
  validation, focused controls, regressions, profiles or publication. External
  failures remain visible and do not automatically veto NinjaRS core validation.
]
#v(7pt)
#section-label[QCD CORRECTIONS · LOOP_SM]
= Complete evaluation · timing observations
#grid(data.cases.filter(c => c.id.starts-with("qcd")))
#v(7pt)
#legend
#v(7pt)
#small[
  Cards contain only actual accepted timing series. Empty cells remain pending;
  numerical calibration observations never populate timing cards. The denominator
  is literal original MadLoop with Ninja. Timing includes numerator construction,
  reduction, scalar masters, assembly, normal stability checks and actual rescue.
  Profiling is separate and excluded from performance acceptance.
]
#v(8pt)
= Numerical-reference scope
Correctness compares every retained normal NinjaRS observation with both accepted
corrected-original CutTools and Ninja calibrations. All #data.config.points_per_case
fixed points, Born/Laurent columns and split-order slots remain in the comparison.
The common point scale contains those three accepted corrected streams. A rejected
external return is recorded separately and cannot become an accepted reference.

#pagebreak()
#section-label[ELECTROWEAK CORRECTIONS · LOOP_QCD_QED_SM_GMU]
= Complete evaluation · electroweak controls
#grid(data.cases.filter(c => c.id.starts-with("ew")))
#v(7pt)
#legend
#v(8pt)
= NinjaRS / original Ninja · unchanged performance gate
#tab((1.1fr, .5fr, .7fr, 1.2fr, .75fr, .85fr),
  table.header(th[Process], th[Pairs], th[Median ratio], th[95% interval], th[Ratio gate], th[Same outcome]),
  ..data.cases.map(c => (
    label(c.id), str(c.timings.candidate.at("8").repetitions),
    fmt(c.performance.at("runtime_ratio", default: none), digits: 3),
    if c.performance.at("ratio_confidence_interval_95", default: none) == none { [—] } else {
      let ci = c.performance.ratio_confidence_interval_95
      [#fmt(ci.at(0), digits: 3) – #fmt(ci.at(1), digits: 3)]
    }, status(c.performance.status), status(c.timing_agreement.status),
  )).flatten(),
)
#v(7pt)
#note[
  *Threshold ×#data.config.max_runtime_ratio.* At least #data.config.repetitions
  genuine paired repetitions are required. The unchanged paired-bootstrap rule
  passes only when the upper 95% bound is at or below the threshold, fails when
  the lower bound exceeds it, and otherwise remains unresolved. No external
  calibration, isolated kernel or corrected reference replaces original Ninja.
  Every matched original-Ninja timed coefficient must also agree with candidate
  NinjaRS and both corrected references. A ratio pass with a failed same-outcome
  gate is not a qualified performance pass; NRS numerical correctness stays separate.
]

#pagebreak()
#section-label[NUMERICAL AGREEMENT · CALIBRATION IS NOT TIMING]
= Complete point streams and actual observations
#tab((1.1fr, .8fr, .85fr, .9fr, 1.1fr),
  table.header(th[Process], th[NRS commands], th[Reference timing reps], th[Max. / tolerance], th[Numerical decision]),
  ..data.cases.map(c => (
    label(c.id), str(c.nrs_numerical_commands),
    c.numerical_reference_usage.values().map(r => str(r.timing_repetitions)).join(" / "),
    fmt(maximum(c), digits: 5), status(c.agreement.status),
  )).flatten(),
)
#v(7pt)
#small[
  Each accepted reference is one #(data.config.points_per_case)-point calibration
  command with zero timing repetitions. Correctness and timing commands reuse the
  fixed point stream; calibration elapsed time is not performance evidence.
]
#v(8pt)
= Actual timed DP / QP usage
#let use(c, source, b) = {
  let t = c.timings.at(source).at(b)
  if t.timed_calls == none { [Pending] } else { [#str(t.dp) DP / #str(t.qp) QP] }
}
#tab((1.2fr, 1fr, 1fr),
  table.header(th[Process], th[Original Ninja], th[NinjaRS]),
  ..data.cases.map(c => (label(c.id), use(c, "baseline", "6"), use(c, "candidate", "8"))).flatten(),
)
#v(7pt)
#small[
  Counts cover successful final statuses from every timed traversal, excluding
  internal attempts. Missing counts remain pending; calibration usage and all raw
  status histograms are retained in report-data.json.
]
#v(8pt)
= Estimates and observed discrepancies
#let stable(c, field) = if c.stability.status != "measured" { [—] } else {
  let t = c.stability.targets.at("candidate_corrected/8")
  if field == "estimate" { sci(t.estimated_accuracy.all_orders.maximum) }
  else { sci(t.observed_discrepancy.max_point_scaled_error) }
}
#tab((1.2fr, 1fr, 1fr, .8fr),
  table.header(th[Process], th[Max. internal estimate], th[Max. observed difference / $S$], th[Selected command]),
  ..data.cases.map(c => (
    label(c.id), stable(c, "estimate"), stable(c, "observed"),
    if c.stability.status != "measured" { [—] } else { str(c.stability.selected_observation.repetition) },
  )).flatten(),
)
#v(7pt)
#small[
  Stability uses the selected first accepted NRS command and both references;
  correctness checks every observation and timed counts aggregate all repetitions.
  Internal estimates and observed differences remain distinct. Pairwise differences
  do not bound the true amplitude error analytically.
]

#pagebreak()
#section-label[EXTERNAL TOOLS · ALL FINDINGS REMAIN VISIBLE]
= Four-tool comparison states
#tab((1.1fr, 1fr, 1fr, 1fr, .9fr),
  table.header(th[Process], th[CutTools], th[Ninja], th[Collier], th[NinjaRS core]),
  ..data.cases.map(c => (
    label(c.id), ..("1", "6", "7").map(b => c.external_backend_cells.at(b)),
    status(c.agreement.status),
  )).flatten(),
)
#v(7pt)
#small[
  “Numerical-reference-only” identifies the corrected-original calibration scope;
  it is not a measured candidate timing or an all-tool qualification. Pending and
  failed observations are kept distinct. Known external return 410 must never be
  relabelled a normal DP/QP success, even if its finite coefficients match another
  stream. Bound finding details, all raw evidence paths and actual upstream links
  remain in the JSON; no issue URL or successful external result is invented.
]
#v(7pt)
#if data.external_diagnostics.findings.len() == 0 {
  note[*External accounting pending.* No bound external finding manifest was
    supplied. This is not a statement that external tools had no failures.]
} else {
  tab((1fr, .5fr, .7fr, .7fr, 1.7fr), left-columns: (0, 4),
    table.header(th[Case], th[Backend], th[Status], th[Points], th[Bound evidence]),
    ..data.external_diagnostics.findings.map(f => (
      str(f.at("case", default: "retained finding")), str(f.at("backend", default: "—")),
      status(f.status), f.at("points", default: ()).map(str).join(", "),
      [#str(f.evidence.len()) raw evidence bindings\ #str(f.at("summary", default: "Detailed finding retained without suppressing failed rows"))],
    )).flatten(),
  )
}
#v(8pt)
= Raw-only completion and original controller outcome
#let provenance = data.correctness_collection_provenance
#if provenance.at("kind", default: "pending") == "derived-raw-correctness-copy-view" {
  note[
    *Original controller failed finalization; derived raw view is separate.*
    The original status remains #status(provenance.original_status.status).
    Byte-exact copied observations are independently validated in a new view;
    the report does not claim original-controller success. No new amplitude
    evaluations, retries or fresh-process parity are inferred from postprocessing.
    The receipt and #provenance.copy_map.len() copy bindings remain in the JSON.
  ]
} else {
  small[The source binding is retained. This report does not infer an original
    controller-success or new fresh-process claim merely from raw numerical rows.]
}

#pagebreak()
#section-label[HIGHER PRECISION · LOOP-INDUCED · FULL OBLIGATIONS]
= Qualification beyond the eight-case core
#tab((1fr, .7fr, 2.5fr), left-columns: (0, 2),
  table.header(th[Obligation], th[Status], th[Scope]),
  ..data.qualification_obligations.pairs().map(p => (
    p.at(0), status(p.at(1).status),
    str(p.at(1).at("reason", default: p.at(1).at("scope", default: "Existing separate authority; inspect bound raw evidence"))),
  )).flatten(),
)
#v(7pt)
#note[
  Genuine precision/rescue qualification combines actual installed public-ABI
  and 256/384-bit controls with normal accepted DP/QP representative-runtime
  coverage. Core rows cannot certify those obligations. Direct mode-4 returns
  120 / accuracy −1 remain diagnostics; they cannot replace normal stability.
  Recomputing numerator and improved MP kinematics is not an isolated scalar
  finite-part oracle. No unsupported obligation is promoted from a pass label.
]
#v(8pt)
= Loop-induced amplitudes
#let loop = data.qualification_obligations.loop_induced.at("evidence", default: none)
#if loop == none or loop.cases.len() == 0 {
  note[*Loop-induced evidence pending.* The required ggH/ggHH correctness,
    performance, threshold and precision obligations are separate. Earlier
    focused controls do not populate final timing cards or close these gates.]
} else {
  grid(loop.cases)
  v(7pt)
  tab((1.1fr, .9fr, 1.4fr, .9fr),
    table.header(th[Process], th[Median ratio], th[95% interval], th[Timing gate]),
    ..loop.cases.map(c => (
      label(c.id), fmt(c.performance.at("runtime_ratio", default: none), digits: 3),
      if c.performance.at("ratio_confidence_interval_95", default: none) == none { [—] } else {
        [#fmt(c.performance.ratio_confidence_interval_95.at(0), digits: 3) – #fmt(c.performance.ratio_confidence_interval_95.at(1), digits: 3)]
      }, status(c.performance.status),
    )).flatten(),
  )
}
#v(8pt)
= Final publication is a separate result
#tab((1.2fr, .7fr, 2.3fr), left-columns: (0, 2),
  table.header(th[Publication obligation], th[Status], th[Evidence still required]),
  ..data.publication_obligations.pairs().map(p => (
    p.at(0), status(p.at(1).status), str(p.at(1).at("reason", default: "Actual checked profile count: " + str(p.at(1).at("count", default: 0)))),
  )).flatten(),
)
#v(7pt)
#small[
  Nonempty source-bound final profiles, honest four-tool accounting, reproducible
  rendering/visual review and pushed milestones are mandatory for publication.
  Neither profiling nor a complete core self-certifies physics or final publication.
]

#pagebreak()
#section-label[RETAINED RESEARCH · DISTINCT SOURCES AND TIMING BOUNDARIES]
#include "research.typ"
#v(8pt)
= Provenance and reproduction
#small[
  Dataset #raw(data.dataset_identity_sha256). Full source/runtime, calibration,
  raw-copy, precision and profile input hashes are retained in report-data.json.
  The render manifest identifies this v3 template, unchanged shared style,
  reader implementations and exact reused historical research fragment.
  Fixed PDF creation epoch and explicit visible timestamp support reproduction
  into fresh destinations. A synthetic software render is not a scientific
  benchmark or publication pass; the historical main report is not overwritten.
]
#v(7pt)
#small[
  Style adapted with attribution from
  #link("https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ")[MadGraphTeam/pyamplicol, loop report].
  Typography, margins, navy headings, striped tables, pale panels and logarithmic
  green–white–red palette use the same retained style asset.
]
