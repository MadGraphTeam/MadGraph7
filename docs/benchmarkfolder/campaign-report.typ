#import "style.typ": *

#let status(value) = text(fill: status-ink(value), weight: "bold", str(value))
#let names = ("1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS", "9": "TIRust")
#let ids = (1, 6, 7, 8, 9)
#let usec(value) = if value == none { [—] } else { [#fixed(value * 1000000, 2)] }

#let comparison-status(value) = if value == none { "uncompared" } else if value { "pass" } else { "fail" }
#let target-status(value) = if value == none { "uncompared" } else if value { "pass" } else { "miss" }

#let show-campaign-numerical(data, cases, process-label) = {
  note(fill: warm)[This table reparses the final complete steady warmup traversal:
    100 saved points, baseline CutTools and five candidate reducers. It is untimed
    numerical evidence. It supplies neither paired runtime acceptance nor native-DP
    cohort timings. Internal retry counters and hardware FPE flags are *unobserved*.
    This warmup-only numerical traversal finishes before unit 6 is redirected,
    so its console diagnostics are retained. The numerical parser does not
    attribute or gate every console warning per call. Subsequent post-READY
    timed RUN batches suppress unit 6 output.]
  v(7pt)
  table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process · saved points], ..ids.map(id => [#names.at(str(id))])),
    ..data.numerical.cases.map(c => {
      let baseline = c.lanes.find(q => q.source == "baseline" and q.backend == 1)
      (
      [#text(8.5pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()
       #text(6.5pt, fill: ink2)[#c.role · 100 expected]#linebreak()
       #text(6.4pt, fill: ink2)[Baseline CT: #status(if baseline == none { "missing" } else { baseline.status })]],
      ..ids.map(id => {
        let lane = c.lanes.find(q => q.source == "candidate" and q.backend == id)
        [#status(if lane.status == "complete" { comparison-status(lane.regression_pass) } else { lane.status })
         #if lane.status == "complete" {
           linebreak()
           text(6.4pt)[DP #lane.accepted_dp_count / QP #lane.accepted_qp_count]
           linebreak()
           text(6.4pt, fill: ink2)[1e-8: #target-status(lane.ordinary_target_pass)]
         }]
      }),
      )
    }).flatten()))
  v(5pt)
  text(8pt, fill: ink2)[Every available comparison uses baseline CutTools rows on
    exactly matching point and split-order indices. A failed baseline leaves
    complete candidate lanes uncompared; their DP/QP counts remain observed. The separate 1e-8 target is
    not substituted for the 3e-6 regression criterion. Failed, incomplete and
    corrupt observations retain their diagnostics and never become successful rows.]
}

#let show-campaign-provisional(data, cases, process-label) = {
  if data.at("refresh", default: none) != none and data.refresh.at("provisional", default: none) != none {
    import "campaign-provisional.typ" as appendix
    appendix.show-provisional(data.refresh.provisional, cases, process-label)
  }
}

#let show-retrospective-stability(plots, process-label) = {
  for item in plots.cases {
    for plot in item.plots {
      pagebreak()
      heading(level: 1)[Retrospective SM stability · #process-label(item)]
      let internal = plot.metric == "accepted-internal-estimate"
      text(9pt, fill: ink2)[#if internal {
        [Accepted composite host diagnostic after rescue]
      } else { [Finite disagreement with current candidate CutTools] }]
      v(5pt)
      figure(image("/" + plot.figure_path, width: 100%, height: 170mm, fit: "contain"),
        caption: [#if internal {
          [Recorded post-rescue MadLoop composite estimates include finite and
          pole information. Estimator definitions differ, including COLLIER's;
          these curves do not rank true numerical accuracy. Exact-zero estimates
          are recorded zero diagnostics, not certified digits or infinite precision.
          #if item.id == "qcd_n4_uux_zzgg" {
            [TIRust has 86 zero estimates alongside 14 DP / 86 QP final acceptances;
            most of these zeros are associated with accepted QP diagnostics.]
          }
          Accepted DP/QP counts describe final outcomes, not all internal attempts.
          The 10#super("−3") line is a paper reference for a different diagnostic,
          not an independently byte-bound host-policy marker.]
        } else {
          [The finite metric is #raw("abs(a-b)/abs(a+b)") against current candidate
          CutTools. That reference is not independently known truth; its 100
          self-comparisons are exactly zero. The 3e-6 regression reference and
          stricter 1e-8 target remain distinct, and retained target misses are
          not discarded. Agreement with this reference does not rank true accuracy.]
        }
        Each curve uses the same 100 ordinary saved points, with 1% empirical
        tail resolution. These are post-rescue accepted-output diagnostics,
        not raw-DP probes or asserted paper-cut-qualified fixtures. Zero tails
        are omitted on the log axis; ratios over Ninja are masked for a zero
        denominator, without a floor or pseudocount. No rarer population-tail,
        independent-truth or performance acceptance follows.])
      v(4pt)
      text(6.2pt, fill: ink2)[Indexed PDF export: #raw(plot.export_pdf_path)]
    }
  }
}

#let show-campaign-details(data, cases, process-label) = {
  show-campaign-provisional(data, cases, process-label)
  pagebreak()
  heading(level: 1)[Timing eligibility and control-process scope]
  if data.at("refresh", default: none) != none {
    note(fill: warm)[*TODO — Hua-Sheng, local machine.*
      Rebuild the delivered candidate and baseline with identical pinned inputs;
      replay all 100 saved points; then run at least seven alternating single-core
      pairs with approximately one-second batches. Report initialization and
      actual backend/precision use separately. The upper 95% ratio bound ≤1.05
      must be checked for every main chain and existing-backend comparison.
      Shared-host work is best effort without reserving CPUs; it cannot pass this
      final performance requirement.]
    v(5pt)
    text(7.3pt, fill: ink2)[User amendment: #raw(data.refresh.amendment_path).
      #if data.refresh.local_handoff_path != none { [Runnable instructions:
        #link("https://github.com/MadGraphTeam/MadGraph7/blob/madloop_tirust/docs/benchmarkfolder/" + data.refresh.local_handoff_path)[Hua-Sheng’s local precise timing handoff].
        It covers fresh installation, local registration, full numerical replay,
        paired timing and a separate local PDF; the handoff is not a measurement.] }]
    if data.refresh.at("next_campaign", default: none) != none {
      v(5pt)
      block[
        #set par(justify: false)
        #text(7.3pt, fill: ink2)[The new local handoff targets #raw(data.refresh.next_campaign.id)
        at revision #raw(data.refresh.next_campaign.registered_exporter_revision.slice(0, 12)).
        #linebreak()Its source identity is #raw(data.refresh.next_campaign.source_sha256).
        #linebreak()
        Fresh qualification is pending for all #data.refresh.next_campaign.case_count cases;
        zero observations are not a numerical failure. This registration supplies
        none of the older candidate’s displayed measurements.]
      ]
    }
    if data.refresh.at("precision_stream_todo_path", default: none) != none {
      v(5pt)
      text(7.3pt, fill: ink2)[Additional local accepted-precision diagnostics:
        #link("https://github.com/MadGraphTeam/MadGraph7/blob/madloop_tirust/docs/benchmarkfolder/" + data.refresh.precision_stream_todo_path)[full-stream precision TODO].
        This separate diagnostic cannot satisfy the formal 1.05 runtime gate.]
    }
    v(7pt)
  }
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
  heading(level: 2)[Controls in the formal campaign]
  [Mixed QCD/QED and loop-induced controls require numerical qualification.
  Their speed measurements are descriptive and do not carry the main-process
  ≤1.05 runtime gate. Generic raw-summary failure labels can include slow
  controls; the role-specific interpretation below preserves those labels
  without turning control speed into a main acceptance failure.]
  v(6pt)
  let control-details = data.paired.at("controls", default: ())
  table-style(table(columns: (1.3fr, 1fr, 1fr, 1fr), inset: 6pt, stroke: table-stroke, fill: table-fill,
    table.header([Control], [Formal paired numerical evidence], [Raw summary status], [Runtime gate]),
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
  text(8pt, fill: ink2)[The untimed numerical and provisional shared-host tables are
    separate datasets. Their completed observations do not populate these
    formal-campaign fields.]

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
  let criterion-labels = (installed_abi_and_fortran_checks: "Installed ABI / Fortran: recorded test groups",
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
  if data.at("refresh", default: none) != none {
    v(5pt)
    if data.refresh.at("implementation_supplement", default: none) != none {
      note(fill: warm)[*Integration work remains open.* Selected standalone runtime,
        control and diagnostic proofs are recorded below. Adaptive-bridge production promotion
        remains pending a separately bound receipt.
        #if data.refresh.implementation_supplement.strict_installer_promotion == "applied; recorded tests passed" {
          [The installer metadata correction has a bound production test receipt.]
        } else { [Installer hardening also remains pending a production receipt.] }
        These scoped proofs do not close the full campaign, stress or
        performance gates.]
    } else {
      note(fill: warm)[*Integration work remains open.* The installer version-type
      correction is tested only in an isolated module and is not yet applied to
      production. Adaptive-precision and effective-point changes still need
      actual backend 8/9 physical qualification. Passed rows describe their
      recorded test scope; they do not mean all implementation audits are closed.]
    }
  }
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
    The historical split-dp-terminal-1 QCD n3 optional target and higher-precision workstream are diagnostic
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
  if data.at("refresh", default: none) != none {
    pagebreak()
    let supplemented = data.refresh.at("implementation_supplement", default: none) != none
    if supplemented {
      heading(level: 1)[Additional implementation proofs]
      [Each updated claim below is derived from an explicitly selected package
      whose index and retained files are hash-bound to this dossier. These
      separate experiments do not replace the source identity, precision usage
      or numerical observations of the ordinary campaign.]
      v(6pt)
    } else {
      heading(level: 1)[Experimental adaptive-precision bridge]
    }
    [The experimental bridge is an isolated development build, whose source and evidence
    are recorded in this public feature branch. It has not replaced the installed
    TIRust bridge used for the ordinary campaign. The branch is
    #raw(data.refresh.branch); the intended future PR target is
    #raw(data.refresh.future_pr_target).]
    v(8pt)
    for row in data.refresh.private_diagnostics {
      heading(level: 2)[#row.title]
      [#row.text]
      v(4pt)
      text(6.6pt, fill: ink2)[#raw(row.evidence)]
      v(8pt)
    }
    if supplemented {
      note(fill: warm)[Actual Fortran tensor transport, unchecked fixed-QP controls
        and a separate physical stream each have a defined scope. They do not
        establish adaptive full-amplitude accuracy, general stress qualification
        or final runtime acceptance. The older campaign remains unchanged.]
    } else {
      note(fill: warm)[Compilation, mock tests, unchecked fixed-QP calls and
        a separate all-512-bit experiment each have limited scope. None establishes
        adaptive-library physical accuracy or final runtime acceptance.]
    }
  }
  if data.at("refresh", default: none) != none and data.refresh.at("appendix_c_retrospective", default: none) != none {
    show-retrospective-stability(data.refresh.appendix_c_retrospective, process-label)
  }
}
