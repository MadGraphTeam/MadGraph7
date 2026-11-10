#import "style.typ": *

#let status(value) = text(fill: status-ink(value), weight: "bold", str(value))
#let names = ("1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS", "9": "TIRust")
#let ids = (1, 6, 7, 8, 9)
#let usec(value) = if value == none { [—] } else { [#fixed(value * 1000000, 2)] }

#let show-campaign-numerical(data, cases, process-label) = {
  note(fill: warm)[This table reparses the final complete steady warmup traversal:
    100 saved points, baseline CutTools and five candidate reducers. It is untimed
    numerical evidence. It supplies neither paired runtime acceptance nor native-DP
    cohort timings. Failure flags and internal retry counters are *unobserved*.]
  v(7pt)
  table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process · saved points], ..ids.map(id => [#names.at(str(id))])),
    ..data.numerical.cases.map(c => (
      [#text(8.5pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()
       #text(6.5pt, fill: ink2)[#c.role · 100 expected]],
      ..ids.map(id => {
        let lane = c.lanes.find(q => q.source == "candidate" and q.backend == id)
        [#status(if lane.status == "complete" { if lane.regression_pass { "pass" } else { "fail" } } else { lane.status })
         #if lane.status == "complete" {
           linebreak()
           text(6.4pt)[DP #lane.accepted_dp_count / QP #lane.accepted_qp_count]
           linebreak()
           text(6.4pt, fill: ink2)[1e-8: #if lane.ordinary_target_pass { [pass] } else { [miss] }]
         }]
      }),
    )).flatten()))
  v(5pt)
  text(8pt, fill: ink2)[Every lane is compared with the baseline CutTools rows on
    exactly matching point and split-order indices. The separate 1e-8 target is
    not substituted for the 3e-6 regression criterion. Failed, incomplete and
    corrupt observations retain their diagnostics and never become successful rows.]
}

#let show-campaign-details(data, cases, process-label) = {
  pagebreak()
  heading(level: 1)[Timing eligibility and control-process scope]
  [Formal main-process timing uses a 2 s calibration target and the frozen
  0.95 s minimum observed batch duration. Each batch traverses all 100 saved
  points. All valid attempts remain visible; a qualified poor result cannot be
  replaced by a favorable retry. These limits are distinct from the separate
  traced-cohort diagnostic duration rule.]
  v(7pt)
  let durations = data.paired.at("batch_durations", default: ())
  table-style(table(columns: (1.3fr, .8fr, .8fr, 1fr), inset: 6pt, stroke: table-stroke, fill: table-fill,
    table.header([Main process], [Min. baseline s], [Min. candidate s], [Retained paired attempts]),
    ..cases.filter(c => c.role == "main").map(c => {
      let observed = durations.filter(q => q.case == c.id)
      (
        [#text(8.4pt, font: "Libertinus Serif")[#process-label(c)]],
        [#if observed.len() == 0 { [—] } else { fixed(calc.min(..observed.map(q => q.minimum_seconds.baseline)), 3) }],
        [#if observed.len() == 0 { [—] } else { fixed(calc.min(..observed.map(q => q.minimum_seconds.candidate)), 3) }],
        [#observed.len() · #observed.filter(q => not q.minimum_batch_duration_met).len() short],
      )
    }).flatten()))
  v(5pt)
  text(8pt, fill: ink2)[Minima include all successfully reparsed retained timing
    attempts and comparison labels, including ineligible short attempts.
    Missing or invalid observations do not supply a duration. Selected runs and
    each attempt's exact minima, labels and eligibility remain in the dossier.]
  v(9pt)
  heading(level: 2)[Descriptive controls]
  [Mixed QCD/QED and loop-induced controls require numerical qualification.
  Their speed measurements are descriptive and do not carry the main-process
  ≤1.05 runtime gate. Generic raw-summary failure labels can include slow
  controls; the role-specific interpretation below preserves those labels
  without turning control speed into a main acceptance failure.]
  v(6pt)
  let control-details = data.paired.at("controls", default: ())
  table-style(table(columns: (1.3fr, 1fr, 1fr, 1fr), inset: 6pt, stroke: table-stroke, fill: table-fill,
    table.header([Control], [Paired numerical evidence], [Raw summary status], [Runtime gate]),
    ..cases.filter(c => c.role != "main").map(c => {
      let row = control-details.find(q => q.case == c.id)
      (
        [#text(8.4pt, font: "Libertinus Serif")[#process-label(c)]],
        [#status(if row == none { "pending" } else { row.numerical_status })],
        [#if row == none { [pending] } else { row.raw_summary_status }],
        [not applicable],
      )
    }).flatten()))
  v(5pt)
  text(8pt, fill: ink2)[The earlier untimed numerical table is a separate dataset.
    It can be complete before any paired control run exists.]

  pagebreak()
  heading(level: 1)[Native-DP and rescued cohort diagnostics]
  note(fill: warm)[#data.cohorts.scope]
  v(7pt)
  table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process], ..ids.map(id => [#names.at(str(id))])),
    ..cases.map(c => (
      [#text(8.4pt, font: "Libertinus Serif")[#process-label(c)]],
      ..ids.map(id => {
        let row = data.cohorts.rows.find(q => q.case == c.id and q.backend == id)
        if row.attempts.len() != 1 or row.attempts.first().status != "diagnostic" {
          [#status(row.status)#if row.attempts.len() > 0 { linebreak(); text(6.4pt)[#row.attempts.len() retained attempts] }]
        } else {
          let groups = row.attempts.first().cohorts
          [#status(row.status)#linebreak()
           #text(6.4pt)[DP #groups.dp.indices.len(): #usec(groups.dp.at("median_seconds_per_point", default: none))]#linebreak()
           #text(6.4pt)[Comp #groups.compensated_dp.indices.len(): #usec(groups.compensated_dp.at("median_seconds_per_point", default: none))]#linebreak()
           #text(6.4pt)[QP #groups.qp.indices.len(): #usec(groups.qp.at("median_seconds_per_point", default: none))]#linebreak()
           #text(6.4pt, fill: ink2)[Failed #groups.failed.indices.len()]]
        }
      }),
    )).flatten()))
  v(5pt)
  text(8pt, fill: ink2)[Cells give frozen point counts and diagnostic median µs per
    call. Empty cohorts have no timing. Short batches, migrations or failure flags
    cannot yield a valid median. The complete original stream is warmed before
    membership is frozen; QP timings include failed DP work and natural rescue.]

  pagebreak()
  heading(level: 1)[Registered implementation and delivery status]
  let criterion-labels = (installed_abi_and_fortran_checks: "Installed ABI and Fortran checks",
    subprocess_event_cache_lifecycle: "Subprocess/event cache lifecycle",
    statistics_roundtrip: "Current statistics round trip")
  table-style(table(columns: (2fr, 1fr), inset: 6pt, stroke: table-stroke, fill: table-fill,
    table.header([Required criterion], [Observed status]),
    ..data.criteria.pairs().map(p => ([#criterion-labels.at(p.at(0), default: p.at(0).replace("_", " "))], [#status(p.at(1))])).flatten()))
  v(8pt)
  let impl = data.implementation
  note[*Installed package.* DSO #raw(impl.library_sha256.slice(0,16)); Fortran archive
    #raw(impl.fortran_archive_sha256.slice(0,16)); actual installer source
    #raw(impl.installer_source_revision.slice(0,12)). This is distinct from the
    registered exporter anchor and each observed export HEAD.]
  v(7pt)
  [The public Fortran parity replay contains #impl.public_fortran_calls_per_installation
  calls per installation. The raw DP replay contains #impl.raw_dp_calls_per_library
  calls per library, with #impl.raw_dp_successes_per_library successes and
  #impl.raw_dp_expected_failures_per_library expected failures. Cache transport
  rejection is checked against the retained old-library negative control.
  These checks qualify the recorded interface and tested transport/cache
  rejection behavior; their elapsed times are not physical-amplitude benchmark
  samples.]
  v(6pt)
  note(fill: if data.subprocess_event_cache_lifecycle.status == "passed" { pale } else { warm })[*Subprocess/event cache lifecycle:
    #status(data.subprocess_event_cache_lifecycle.status).*
    #data.subprocess_event_cache_lifecycle.scope
    #if data.subprocess_event_cache_lifecycle.evidence != none {
      v(3pt)
      text(6.3pt, fill: ink2)[#raw(data.subprocess_event_cache_lifecycle.summary_evidence)]
    }]
  v(6pt)
  note(fill: if data.statistics.status == "passed" { pale } else { warm })[
    *Current statistics: #status(data.statistics.status).* #data.statistics.scope
    #if data.statistics.evidence != none {
      v(3pt)
      text(6.3pt, fill: ink2)[#raw(data.statistics.evidence)]
    }]
  v(6pt)
  [Higher-rank terminal Fortran qualification is bound to the completed source,
  IEEE/trap and fixed-input coefficient checks in
  #raw("experiments/dp-higher-rank-production/validation-summary.json"). Its
  compiled checks and CutTools parity do not substitute for the registered
  five-backend campaign or establish runtime admission frequency.]
  pagebreak()
  heading(level: 1)[Retained accuracy limitations and diagnostics]
  text(8pt, fill: ink2)[The report binds the completed portable inventories and
    their recorded scope; it does not rerun scientific adjudication. Historical
    library results and private prototypes do not qualify the current installed
    library. Ordinary numerical and paired runtime gates remain separate.
    The QCD n3 optional target and higher-precision workstream are diagnostic
    disclosures; neither adds a stress acceptance requirement.]
  v(7pt)
  table-style(table(columns: (1.1fr, .5fr, 3fr), inset: 5pt, stroke: table-stroke, fill: table-fill,
    table.header([Case], [Status], [Qualification limit]),
    ..data.stress.map(row => ([#row.title], [#status(row.status)],
      [#row.text#if row.evidence != none { v(3pt); text(6.3pt, fill: ink2)[#raw(row.evidence)] }])).flatten()))
  v(6pt)
  text(8pt, fill: ink2)[The historical c91 qualified runtime failure remains an
    ancestor result with its original source and interval. New source candidates
    do not erase it. Ordinary-point agreement and a passing eight-process runtime
    gate alone cannot declare the full integration goal complete.]
}
