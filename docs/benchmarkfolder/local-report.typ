// Presentation uses the existing, attributed pyAmpliCol-derived style.
// This document consumes a collected dossier; it never evaluates amplitudes.
#import "style.typ": *
#let dossier-path = sys.inputs.at("dossier", default: "dossier.json")
#let d = json(dossier-path)
#show: body => report-style(body, date: d.date,
  revision: d.canonical.source_revision.slice(0, 12))
// The Nix Libertinus bundle also includes a converted Math TTF without MATH
// tables. Use the unambiguous OpenType-math face from the existing DejaVu set.
#show math.equation: set text(font: "DejaVu Math TeX Gyre")

#let names = ("1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS", "9": "TIRust")
#let ids = (1, 6, 7, 8, 9)
#let state(value) = text(fill: if value in ("invalid-evidence", "timing-ineligible") {
  status-ink("failed")
} else { status-ink(value) }, weight: "bold", str(value))
#let check(value) = if value == none { [—] } else { state(if value { "pass" } else { "fail" }) }
#let usec(value) = if value == none { [—] } else { fixed(value * 1000000, 2) }
#let sci(value) = if value == none { [—] } else if value == 0 { [0] } else {
  let exponent = calc.floor(calc.log(calc.abs(value), base: 10))
  [#fixed(value / calc.pow(10, exponent), 2) × 10#super(str(exponent))]
}
// Permit long evidence paths and full digests to wrap inside table cells.
#let code(value) = if value == none { [—] } else {
  let rendered = if type(value) == str { value } else { repr(value) }
  text(font: "DejaVu Sans Mono", size: 6.6pt,
    rendered.clusters().chunks(16).map(part => part.join()).join("\u{200b}"))
}
#let process(c) = {
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
  labels.at(c.id, default: code(c.at("process", default: c.id)))
}
#let comparison(label) = if label == "tirust-first" { [TIRust-first chain] } else if label.starts-with("backend-") {
  [#names.at(label.slice(8), default: label) standalone]
} else { code(label) }
#let interval(row) = if row.lower_95 == none or row.upper_95 == none { [—] } else {
  [#fixed(row.lower_95, 3)–#fixed(row.upper_95, 3)]
}
#let usage(value) = {
  if value == none or value.len() == 0 { return [—] }
  value.pairs().map(pair => {
    let bits = pair.at(0).split(":")
    [#if bits.len() == 2 {
      [#names.at(bits.first(), default: bits.first()) #bits.last()]
    } else { [#pair.at(0)] }: #pair.at(1)]
  }).join([; ])
}
#let usage-by-source(value) = {
  if value == none or value.len() == 0 { return [—] }
  value.pairs().map(pair => [#pair.at(0): #usage(pair.at(1))]).join(linebreak())
}
#let tier-counts(value) = {
  if value == none { return [unknown] }
  let counts = ("primary", "check").map(side => {
    let block = value.at(side, default: (:))
    ("106", "256", "512").map(bits => str(block.at(bits, default: 0))).join("/")
  })
  [P #counts.at(0); C #counts.at(1)]
}
#let arithmetic(value) = {
  if value == none or value.len() == 0 { return [unknown] }
  let labels = value.at("precision_classifications", default: ())
  let highest = value.at("highest_observed_internal_bits", default: none)
  [#if labels.len() == 0 { [unknown] } else { labels.join([; ]) }
   #if highest != none {
     [ · observed tier #highest bits]
     linebreak()
     text(6.2pt)[success106/256/512: #tier-counts(value.at("higher_precision_success_counts", default: none))]
     linebreak()
     text(6.2pt)[attempt106/256/512: #tier-counts(value.at("higher_precision_attempt_counts", default: none))]
   }
   #linebreak()#text(6.5pt)[route coverage: #value.at("route_interval_coverage", default: "unknown")]
   #linebreak()#text(6.5pt)[attempts: #value.at("attempt_count_scope", default: "unknown")]]
}
#let arithmetic-by-source(value) = {
  if value == none or value.len() == 0 { return [unknown] }
  value.pairs().map(pair => [#pair.at(0): #arithmetic(pair.at(1))]).join(linebreak())
}
#let property-table(entries) = table-style(table(columns: (1fr, 2.7fr),
  inset: 5pt, stroke: table-stroke, fill: table-fill,
  table.header([Identity / input], [Recorded value]),
  ..entries.map(pair => ([#pair.at(0)], [#code(pair.at(1))])).flatten()))

#align(center)[
  #text(26pt, weight: "bold", fill: navy)[MadLoop with TIRust]
  #v(5pt)
  #text(12pt, fill: ink2)[Local numerical validation and benchmarks]
  #v(4pt)
  #text(8.5pt, fill: luma(110))[CutTools · Ninja · COLLIER · NinjaRS · TIRust]
  #v(4pt)
  #text(8pt, fill: luma(110))[#d.date · #d.profile · #code(d.id)]
]
#v(8pt)
#note(fill: warm)[*Overall integration goal: unqualified.*
  This report describes a local deployment of the frozen campaign
  #code(d.canonical.campaign). A local numerical or timing pass does not close
  the implementation, stress, poles-only, precision, or delivery requirements.
  Historical results and failures remain attached to their original sources.]
#v(7pt)
#note(fill: if d.timing.main_gate.status == "passed" { pale } else { warm })[
  *Local main-process timing gate: #state(d.timing.main_gate.status).*
  Every main process requires the TIRust-first chain and existing-backend
  comparisons to pass their timed numerical checks and the upper 95% paired
  runtime-ratio bound ≤#d.policy.performance_upper_ratio.
  Controls and standalone TIRust speed measurements are descriptive.]
#v(8pt)

= Full-point numerical agreement

Each lane evaluates the same #d.policy.points_per_case saved physical points.
Born, finite, single-pole and double-pole coefficients are compared with the
baseline CutTools lane using the #sci(d.policy.relative_tolerance) regression
criterion. The #sci(d.policy.ordinary_target) ordinary-point target is recorded
separately. These untimed checks do not supply the correctness of timed batches.

#figure(table-style(table(columns: (1.35fr, .77fr, .77fr, .77fr, .77fr, .77fr),
  inset: (x: 4pt, y: 4pt), stroke: table-stroke, fill: table-fill,
  table.header([Process · points], ..ids.map(id => [#names.at(str(id))])),
  ..d.numerical.cases.map(c => (
    [#text(8.5pt, font: "Libertinus Serif")[#process(c)]#linebreak()
     #text(6.7pt, fill: ink2)[#c.role · #c.points points]],
    ..ids.map(id => {
      let lane = c.lanes.find(q => q.source == "candidate" and q.backend == id)
      if lane == none { return [#state("pending")] }
      [#if lane.status == "complete" { check(lane.regression_pass) } else { state(lane.status) }
       #linebreak()#text(6.5pt)[target: #check(lane.target_pass)]
       #linebreak()#text(6.5pt, fill: ink2)[#sci(lane.max_relative_error)]]
    }),
  )).flatten())), caption: [Untimed numerical matrix. Each cell shows the regression result,
    ordinary-point target and maximum retained relative discrepancy.])

#text(8pt, fill: ink2)[Full-point numerical status: #state(d.numerical.status).
  Missing measurements appear as —; failure and incomplete statuses are never
  converted into zero error or zero runtime.]

#pagebreak()
= Paired full-amplitude timing

The formal local protocol uses at least #d.policy.repetitions alternating,
single-core pairs, full traversals of all #d.policy.points_per_case saved
points, and a minimum observed batch duration of
#number(d.policy.minimum_batch_seconds) s. Initialization is reported separately.
Runtime includes the observed rescue work. Candidate / baseline ratios below
one are faster; the green–white–red palette shows speed only, not qualification.

#text(8pt, fill: ink2)[The baseline is the frozen starting implementation, which
  already includes earlier NinjaRS and Fortran improvements. This report does
  not attribute those improvements to TIRust. No FULL−SKIP estimate is presented
  as a directly measured reduction time.]
#v(5pt)

#for c in d.timing.cases {
  heading(level: 2)[#process(c) · #c.role]
  table-style(table(columns: (1.22fr, .83fr, .7fr, .7fr, 1fr, .65fr),
    inset: (x: 4pt, y: 6pt), stroke: table-stroke, fill: table-fill,
    table.header([Comparison · pairs], [Evidence status], [Baseline µs/point],
      [Candidate µs/point], [Ratio · 95% interval], [Timed physics]),
    ..c.comparisons.map(row => (
      [#comparison(row.label)#linebreak()#text(6.5pt, fill: ink2)[
        #if row.pairs == none { [— pairs] } else { [#row.pairs pairs] }
        · #if row.speed_required { [speed required] } else { [descriptive speed] }]],
      [#state(row.status)#linebreak()#text(6.3pt)[
        #if row.speed_required { [speed: #check(row.passed)] } else { [no speed gate] }]],
      [#usec(row.baseline_seconds_per_point)],
      [#usec(row.candidate_seconds_per_point)],
      table.cell(fill: ratio-fill(row.ratio), [#text(9pt, weight: "bold")[#ratio-text(row.ratio)]
        #linebreak()#text(6.5pt)[#interval(row)]]),
      [#check(row.numerical_pass)],
    )).flatten()))
  let errors = c.comparisons.filter(row => row.error != none)
  if errors.len() > 0 {
    v(4pt)
    note(fill: warm)[#for row in errors {
      [*#comparison(row.label):* #row.error#linebreak()]
    }]
  }
  v(5pt)
}

#pagebreak()
= Timed standalone cross-backend correctness

This independent check compares the candidate's standalone reducers with its
timed CutTools observations, for each retained pair. It does not reuse the
untimed full-point matrix. A passing baseline/candidate comparison in the timing
tables is a distinct check and cannot substitute for this matrix.

#table-style(table(columns: (1.35fr, .77fr, .77fr, .77fr, .77fr, .77fr),
  inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
  table.header([Process], ..ids.map(id => [#names.at(str(id))])),
  ..d.timing.cases.map(c => (
    [#text(8.5pt, font: "Libertinus Serif")[#process(c)]#linebreak()
     #text(6.7pt, fill: ink2)[#c.role]],
    ..ids.map(id => {
      let row = c.at("standalone_correctness", default: ()).find(q => q.backend == id)
      if row == none { return [#state("pending")] }
      [#if row.status == "complete" { check(row.passed) } else { state(row.status) }
       #linebreak()#text(6.5pt)[#row.samples pair checks]
       #linebreak()#text(6.5pt)[target: #check(row.ordinary_target_passed)]
       #linebreak()#text(6.5pt, fill: ink2)[#sci(row.max_relative_error)]]
    }),
  )).flatten()))

#v(7pt)
#note(fill: warm)[These rows establish only the numerical scope of the observed
  timed samples. Performance eligibility, the main-process runtime gate and
  ordinary full-point agreement retain their own requirements.]

#pagebreak()
= Initialization and observed precision

Initialization and warmup elapsed times below come from the untimed numerical
runs and are reported separately from warmed batches. They include the first
amplitude initialization and saved-stream warmup, excluding process startup
and parameter loading. Accepted backend/precision counts
describe final input/output transport. TIRust219 may include internal106-bit
DD or256/512-bit tensor rescues of DP inputs and a DP numerator. Genuine113-bit
QP transport323 is separate from recomputing a QP numerator, which also requires
the effective card. Optional route snapshots are taken outside timed batches;
Fortran coefficient-cache hits bypass these bridge counters. Missing/censored
route data and complete-amplitude native precision remain unknown. This dataset
does not provide separately timed native-DP-only or rescued cohorts.

#for c in d.numerical.cases {
  heading(level: 2)[#process(c)]
  table-style(table(columns: (1fr, .55fr, 2fr), inset: (x: 5pt, y: 5pt),
    stroke: table-stroke, fill: table-fill,
    table.header([Numerical lane], [Init. + warmup · s], [Accepted backend and precision · points]),
    ..c.lanes.map(lane => (
      [#lane.source #names.at(str(lane.backend), default: str(lane.backend))],
      [#number(lane.cold_seconds)],
      [#usage(lane.backend_precision_counts)],
    )).flatten()))
  v(3pt)
}

#heading(level: 2)[Usage in selected timing observations]
#table-style(table(columns: (1.05fr, 1fr, 1.7fr, 2.2fr), inset: 5pt,
  stroke: table-stroke, fill: table-fill,
  table.header([Process], [Comparison], [Accepted transport · evaluations], [Observed arithmetic]),
  ..d.timing.cases.map(c => c.comparisons.map(row => (
    [#process(c)], [#comparison(row.label)], [#usage-by-source(row.usage)],
    [#arithmetic-by-source(row.at("precision_usage", default: (:)))],
  )).flatten()).flatten()))

#pagebreak()
= Retained attempts and predecessor failures

The local registry retains every attempt. Selected observations are identified
below; an eligible unfavorable result cannot be hidden by a favorable retry.
Unselected, invalid and incomplete attempts remain visible with their original
diagnostics. Historical failures are separate evidence from earlier deployments.

#if d.timing.attempts.len() == 0 {
  note(fill: warm)[No local paired attempts have been recorded.]
} else {
  table-style(table(columns: (1fr, .9fr, .85fr, 1fr, 1.8fr), inset: 5pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Process · comparison], [Status], [Selected / eligible],
      [Ratio · 95% interval], [Evidence / diagnostic]),
    ..d.timing.attempts.map(row => (
      [#code(row.case)#linebreak()#comparison(row.label)],
      [#state(row.status)#linebreak()#check(row.passed)],
      [#if row.selected { [yes] } else { [no] } /
       #if row.acceptance_eligible == true { [yes] } else if row.acceptance_eligible == false { [no] } else { [—] }],
      table.cell(fill: ratio-fill(row.ratio), [#ratio-text(row.ratio)#linebreak()#interval(row)]),
      [#code(row.directory)#if row.error != none { linebreak(); text(6.7pt)[#row.error] }],
    )).flatten()))
}

#heading(level: 2)[Historical qualified failures]
#if d.historical_failures.len() == 0 {
  [No historical qualified failures are listed in the bound ancestry.]
} else {
  table-style(table(columns: (1fr, .9fr, 1fr, 2.2fr), inset: 5pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Process · comparison], [Predecessor], [Ratio · 95% interval], [Retained evidence]),
    ..d.historical_failures.map(row => (
      [#code(row.case)#linebreak()#comparison(row.label)],
      [#code(row.predecessor_id)#linebreak()#state("failed")],
      table.cell(fill: ratio-fill(row.ratio), [#ratio-text(row.ratio)#linebreak()#interval(row)]),
      [#code(row.directory)#linebreak()#code(row.evidence_sha256)],
    )).flatten()))
}

#pagebreak()
= Sources, local deployment and reproduction

The collected dossier and render manifest bind the exact inputs used for this
document. Full digests below identify the source, native artifacts and frozen
dependency resolution. Local deployment results do not replace the canonical
campaign or its historical failures.

#property-table((
  ("Local measurement", d.id),
  ("Canonical campaign", d.canonical.campaign),
  ("Canonical manifest SHA-256", d.canonical.manifest_sha256),
  ("Source revision", d.canonical.source_revision),
  ("Source files SHA-256", d.canonical.source_files_sha256),
  ("Numerical identity SHA-256", d.canonical.numerical_identity_sha256),
  ("Native source binding", d.deployment.source_binding),
  ("Selected logical CPU", d.deployment.cpu),
  ("Measurement manifest SHA-256", d.provenance.measurement_sha256),
  ("oneloopmaster revision", d.policy.oneloopmaster_revision),
))
#v(7pt)
#property-table(d.deployment.host.pairs())
#v(7pt)
#property-table(d.deployment.native_files.pairs())
#v(7pt)
#property-table((
  ("Cargo manifest SHA-256", d.deployment.recipe.manifest_sha256),
  ("Resolved Cargo.lock SHA-256", d.deployment.recipe.lock_sha256),
  ("TIRust upstream revision", d.deployment.recipe.at("tirust_revision", default: none)),
  ("TIRust patch SHA-256", d.deployment.recipe.at("patch_sha256", default: none)),
))

#heading(level: 2)[Complete resolved Git dependency pins]
#table-style(table(columns: (1fr, .5fr, 3fr), inset: 5pt,
  stroke: table-stroke, fill: table-fill,
  table.header([Package], [Version], [Source and complete revision]),
  ..d.deployment.recipe.git_dependencies.map(row => (
    [#row.name], [#row.version], [#code(row.source)],
  )).flatten()))

#v(8pt)
#note[The exact invocation, root directory, #raw("SOURCE_DATE_EPOCH") and font
  input digests are recorded in #raw("build.json"). A rendering command within
  the retained render directory is
  #raw("typst compile --root . --ignore-system-fonts --ignore-embedded-fonts --input dossier=dossier.json local-report.typ benchmark.pdf").
  Restore the recorded environment and font inputs for reproduction.
  The local launcher collects evidence and copies this template and
  #raw("style.typ") into the render directory. The renderer does not execute
  native amplitudes. Measurement commands, raw streams and complete evidence
  inventories remain in the local measurement registry.]
#v(7pt)
#text(8pt, fill: ink2)[Typography, margins, navy headings, tables, panels and the
  green–white–red timing palette follow the preserved pyAmpliCol benchmark style:
  #link("https://github.com/MadGraphTeam/pyamplicol/blob/7266c1600f6ec4e39be39986e6503a515c380c52/docs/performance_reports/loop/benchmarks.typ")[
    MadGraphTeam/pyamplicol, revision 7266c1600f6ec4e39be39986e6503a515c380c52].
  The existing NinjaRS report is preserved.]
