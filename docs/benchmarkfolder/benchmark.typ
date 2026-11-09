#import "style.typ": *
#let d = json("summary.json")
#let env = json("implementation-start.json")
#show: body => report-style(body, date: d.at("date", default: "9 November 2026"),
  revision: if d.at("candidate_revision", default: none) == none { "not yet measured" } else { d.candidate_revision.slice(0,12) })
#let names = ("1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS", "9": "TIRust")
#let ids = (1, 6, 7, 8, 9)
#let main = d.cases.filter(c => c.role == "main")
#let controls = d.cases.filter(c => c.role != "main")
#let tool-name(x) = names.at(str(x), default: str(x))
#let time-us(x) = if x == none { [—] } else { [#number(x * 1000000)] }
#let cell-status(s) = text(fill: status-ink(s), weight: "bold", str(s))
#let get-timing(c, id) = c.at("timings", default: ()).find(t => str(t.backend) == str(id))
#let get-check(c, id) = c.correctness.find(t => str(t.backend) == str(id))
#let process-label(c) = {
  let labels = (
    "qcd_n1_uux_z": $u overline(u) -> Z$,
    "qcd_n2_gg_ttx": $g g -> t overline(t)$,
    "qcd_n3_uux_zgg": $u overline(u) -> Z g g$,
    "qcd_n4_uux_zzgg": $u overline(u) -> Z Z g g$,
    "ew_n1_udx_wp": $u overline(d) -> W^+$,
    "ew_n2_epem_mupmum": $e^+ e^- -> mu^+ mu^-$,
    "ew_n3_epem_ttxa": $e^+ e^- -> t overline(t) gamma$,
    "ew_n4_epem_mupmumaa": $e^+ e^- -> mu^+ mu^- gamma gamma$,
    "mixed_uux_ga": $u overline(u) -> g gamma$,
    "li_gg_h": $g g -> H$,
    "li_gg_hh": $g g -> H H$,
  )
  labels.at(c.id, default: raw(c.process))
}

#align(center)[
  #text(26pt, weight: "bold", fill: navy)[MadLoop with TIRust]
  #v(5pt)
  #text(12pt, fill: ink2)[One-loop reduction · numerical agreement · runtime]
  #v(4pt)
  #text(8.5pt, fill: luma(110))[CutTools · Ninja · COLLIER · NinjaRS · TIRust]
  #v(3pt)
  #text(8pt, fill: luma(110))[Starting MadLoop revision #raw(d.baseline_revision.slice(0,12)) · one pinned CPU · full helicity/color sums]
]
#v(7pt)
#note(fill: if d.gate.complete and d.gate.passed { pale } else { warm })[
  *Acceptance: #cell-status(if d.gate.complete and d.gate.passed { "pass" } else if d.gate.complete { "fail" } else { "incomplete" }).*
  Every main process requires five-backend agreement, an upper 95% paired runtime-ratio bound no greater than 1.05, and passing existing-backend regression checks. Unmeasured cells are shown as —. A failed or incomplete row is not a speedup claim.
]
#v(7pt)
#text(8pt)[Runtime colour: green = faster, white = parity, red = slower. All displayed ratios are candidate / reference; smaller is faster.]
#v(5pt)
#figure(
  table-style(table(columns: (1.35fr, .75fr, .9fr, .9fr, 1.1fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process · correction], [Physics], [Baseline µs/point], [TIRust chain µs/point], [Ratio · 95% interval]),
    ..main.map(c => {
      let p = c.performance
      let ratio = p.at("geometric_mean_ratio", default: none)
      (
        [#text(9pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()
         #text(7pt, fill: ink2)[#if c.id.starts-with("qcd") { [QCD] } else { [EW] } · 2→#c.nfinal · #if c.points == none { [unmeasured] } else { [#c.points points] }]],
        [#cell-status(c.status)],
        [#text(11pt, weight: "bold")[#time-us(p.at("baseline_median_seconds_per_point", default: none))]],
        [#text(11pt, weight: "bold")[#time-us(p.at("candidate_median_seconds_per_point", default: none))]],
        table.cell(fill: ratio-fill(ratio), [
          #text(11pt, weight: "bold")[#ratio-text(ratio)]#linebreak()
          #if p.at("lower_95", default: none) != none {
            [#fixed(p.lower_95,3)–#fixed(p.upper_95,3)#linebreak()]
          }
          #cell-status(p.status)
        ]),
      )
    }).flatten(),
  )),
  caption: [Main process matrix. Timings include the actual rescue work. Physics and runtime acceptance are separate.],
)
#v(5pt)
#text(8pt, fill: ink2)[This study measures the new integration against the untouched starting branch, which already includes NinjaRS and earlier Fortran optimizations. It does not attribute those earlier improvements to TIRust.]

#pagebreak()
= Five-backend numerical agreement

The same saved momenta, model parameters, coupling orders and scale are evaluated by all five tools. The regression criterion is #raw("3e-6"), with #raw("1e-8") as the ordinary-point target. Born, finite, simple-pole and double-pole coefficients are retained. Near-zero poles use the existing MadLoop comparison policy. Native double-precision failures and successful rescues are recorded separately in the numerical evidence.

#figure(table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
  inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
  table.header([Process], ..ids.map(id => [#tool-name(id)])),
  ..d.cases.map(c => (
    [#text(8.6pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()#text(7pt, fill: ink2)[#c.role · #if c.points == none { [unmeasured] } else { [#c.points points] }]],
    ..ids.map(id => {
      let q = get-check(c,id)
      if q == none { [pending] } else {
        [#cell-status(q.status)#linebreak()
         #if q.at("max_relative_error", default: none) != none {
           text(7pt)[#str(q.max_relative_error)]
         }]
      }
    }),
  )).flatten(),
)), caption: [Comparison status and largest retained relative discrepancy. A dash or pending status denotes absent evidence.]))

#note[
  Agreement is assessed using independent numerical backends and higher precision when a discrepancy occurs. Known difficult scalar-box regions are retained as stress cases. A backend's own success flag does not establish cross-backend agreement.
]

#pagebreak()
= Absolute timing by reduction backend

Values below are warmed full-amplitude times in µs per point. Initialization, compilation and model generation are excluded and recorded separately. All tools use matching pole, helicity/color and stability settings. Cache behaviour and actual rescue usage remain part of the measurement.

#figure(table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
  inset: (x: 4pt, y: 8pt), stroke: table-stroke, fill: table-fill,
  table.header([Process], ..ids.map(id => [#tool-name(id)])),
  ..d.cases.map(c => (
    [#text(8.6pt, font: "Libertinus Serif")[#process-label(c)]],
    ..ids.map(id => {
      let t = get-timing(c,id)
      if t == none { [—] } else {
        [#text(10pt, weight: "bold")[#time-us(t.at("median_seconds_per_point", default: none))]#linebreak()
         #if t.at("mad_seconds_per_point", default: none) != none {
           text(6.6pt, fill: ink2)[MAD #time-us(t.mad_seconds_per_point)]
         }]
      }
    }),
  )).flatten(),
)), caption: [Per-backend timing observations. Missing measurements are not represented as zero.]))

#if d.at("ratio_figure", default: none) != none {
  figure(image(d.ratio_figure, width: 100%),
    caption: [Paired candidate/baseline ratios and 95% intervals; the acceptance limit is 1.05.])
}

#pagebreak()
= Methods and qualification

== Physical samples and timing protocol

The main matrix contains QCD and electroweak corrections at every multiplicity from 2→1 through 2→4, using #raw("loop_sm") and #raw("loop_qcd_qed_sm_Gmu"). Each main case declares 100 deterministic points. On-shell 2→1 points vary their boost at the physical invariant; higher multiplicities use 1 TeV. Mixed QCD/QED and loop-induced cases supplement the main matrix. Threshold, soft/collinear, small-Gram and complex-mass cases are separate stress controls.

Performance acceptance uses at least seven alternating paired repetitions, with each timed batch calibrated to approximately one second. The starting revision and candidate share their compiler, library installations, model/cards, point order and physical settings. Timings use a single allowed CPU and no simultaneous compilation. The upper 95% paired ratio bound must be ≤1.05 separately for every main process. Averages cannot hide a failing row. A FULL−SKIP quantity, when reported in supporting data, is an *estimated reduction stage* rather than a pure tensor-kernel stopwatch.

== Interface, poles and caches

MadLoop selects TIRust explicitly as #raw("MLReductionLib=9"). A Fortran adapter transfers supplied invariants, complex squared masses and the squared renormalization scale to a separate Rust bridge. Tensor moments are permuted into MadLoop order and contracted in Fortran. The quadruple-precision path preserves binary128 inputs and uses MPFR working precision; double-double arithmetic is not labelled quadruple precision.

#raw("LoopPoleMode") selects bare-loop UV/IR residues while preserving the ordinary finite coefficient. The distinct poles-only diagnostic path omits finite scalar roots and finite tensor arithmetic; it preserves double-pole contributions to simple poles and scaleless UV/IR separation. Partial results cannot be used as complete physical amplitudes. Cache and precision stores are separated by evaluation mode, and stability rechecks use independent values.

#let quals = d.at("qualification", default: ())
#if quals.len() > 0 {
  table-style(table(columns: (2fr, .7fr, 2fr), inset: 5pt, stroke: table-stroke, fill: table-fill,
    table.header([Qualification], [Status], [Evidence]),
    ..quals.map(q => ([#q.name], [#cell-status(q.status)], [#q.evidence])).flatten()))
} else {
  note(fill: warm)[Detailed interface, cache, QP, poles-only, and relocation qualification records have not yet been incorporated in this data release. Their absence is not a passing result.]
}

== Reproduction and provenance

The accompanying #raw("cases.json") fixes the matrix and thresholds; #raw("summary.json") is derived from retained observations. The source plan, baseline manifest, dependency revisions, build environment, and raw measurement paths accompany this report. Rebuilding the PDF does not rerun numerical measurements.

#table-style(table(columns: (1fr, 3fr), inset: 5pt, stroke: table-stroke, fill: table-fill,
  table.header([Component], [Recorded environment]),
  [Baseline], [#raw(d.baseline_revision)],
  [Rust], [#env.rustc],
  [Fortran], [#env.gfortran],
  [Typst], [#env.typst],
  [TIRust], [#raw(env.dependency_revisions.TIRust)],
  [oneloopmaster], [#raw(env.dependency_revisions.oneloopmaster)],
))

#v(6pt)
#text(8pt, fill: ink2)[Presentation follows the #link("https://github.com/MadGraphTeam/pyamplicol/blob/7266c1600f6ec4e39be39986e6503a515c380c52/docs/performance_reports/loop/benchmarks.typ")[pyAmpliCol report at revision 7266c1600f6e]: its page geometry, fonts, navy headings, compact tables and logarithmic timing palette. All numerical measurements and conclusions in this report belong to the present MadLoop integration.]
