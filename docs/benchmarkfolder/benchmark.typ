#import "style.typ": *
#let summary-path = sys.inputs.at("summary", default: "summary.json")
#let manifest-path = sys.inputs.at("manifest", default: "")
#let context-path = sys.inputs.at("context", default: "")
#let d = json(summary-path)
#let campaign = if manifest-path == "" { none } else { json(manifest-path) }
#let report = if context-path == "" {
  (cases_path: "cases.json", predecessors: (), ratio_figure: d.at("ratio_figure", default: none))
} else { json(context-path) }
#let pilots = json("experiments/numerical/completed-pilots-index.json")
#let intermediate = if report.at("intermediate_path", default: none) == none { none } else { json(report.intermediate_path) }
#let current-dossier = if report.at("campaign_report_path", default: none) == none { none } else { json(report.campaign_report_path) }
#let refresh = if current-dossier == none { none } else { current-dossier.at("refresh", default: none) }
#let env = json("implementation-start.json")
#show: body => report-style(body, date: d.at("date", default: "9 November 2026"),
  revision: if campaign != none { "registered " + campaign.candidate.exporter_identity.revision.slice(0, 12) } else if d.at("candidate_revision", default: none) == none { "not yet measured" } else {
    d.candidate_revision.slice(0,12) + if intermediate != none { " (measured)" } else { "" }
  })
#let names = ("1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS", "9": "TIRust")
#let ids = (1, 6, 7, 8, 9)
#let main = d.cases.filter(c => c.role == "main")
#let controls = d.cases.filter(c => c.role != "main")
#let known-failures = d.gate.at("failed_cases", default: ()).len() > 0
#let tool-name(x) = names.at(str(x), default: str(x))
#let time-us(x) = if x == none { [—] } else { [#number(x * 1000000)] }
#let cell-status(s) = text(fill: status-ink(s), weight: "bold", str(s))
#let scientific(x) = if x == 0 { [0] } else {
  let exponent = calc.floor(calc.log(calc.abs(x), base: 10))
  [#fixed(x / calc.pow(10, exponent), 2) × 10#super(str(exponent))]
}
#let get-timing(c, id) = c.at("timings", default: ()).find(t => str(t.backend) == str(id))
#let get-check(c, id) = c.correctness.find(t => str(t.backend) == str(id))
#let checks-status(items) = if items.len() == 0 or items.any(q => q.passed == none) {
  "pending"
} else if items.all(q => q.passed) { "pass" } else { "fail" }
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
  #text(12pt, fill: ink2)[One-loop reduction · #if campaign == none { [intermediate qualification report] } else { [registered campaign report] }]
  #v(4pt)
  #text(8.5pt, fill: luma(110))[CutTools · Ninja · COLLIER · NinjaRS · TIRust]
  #v(3pt)
  #text(8pt, fill: luma(110))[Starting MadLoop revision #raw(d.baseline_revision.slice(0,12)) · full helicity/color sums]
]
#v(7pt)
#if campaign != none {
  text(8pt, fill: ink2)[Selected campaign: #raw(campaign.id). Selected numerical identity:
    #raw(campaign.candidate.numerical_identity_sha256.slice(0,16)).]
  v(4pt)
  text(8pt, fill: ink2)[Registered exporter anchor: #raw(campaign.candidate.exporter_identity.revision.slice(0,12)).
    Observed formal paired-timing export revisions: #if report.at("observed_candidate_revisions", default: ()).len() == 0 { [not yet measured] } else { raw(report.observed_candidate_revisions.map(r => r.slice(0,12)).join(", ")) }.
    Documentary revisions can differ while the registered exporter bytes remain identical.]
  v(4pt)
  note(fill: warm)[*Complete delivery qualification:
    #cell-status(if current-dossier != none and current-dossier.delivery_complete { "pass" } else { "incomplete" }).*
    #if refresh != none {
      if refresh.at("historical_report", default: none) != none {
        [#refresh.closed_cases of #refresh.expected_cases physical cases have closed: #refresh.closed_lanes lane slots × 100 saved points.
        #refresh.complete_cases complete cases; #refresh.regression_passed_cases pass 3e-6.
        Current numerical status: #cell-status(refresh.numerical_status).
        The 1e-8 target is missed by #refresh.current_target_miss_cases.len() complete cases; all lane outcomes remain visible.]
      } else { [#refresh.closed_cases of #refresh.expected_cases physical cases are closed: #refresh.closed_lanes lanes × 100 saved points pass 3e-6. #if refresh.closed_cases < refresh.expected_cases { [Unclosed cases remain pending in this frozen snapshot.] } else { [The ordinary five-backend numerical matrix is complete.] }] }
    }
    The eleven-process numerical matrix,
    controls, existing-backend regressions, precision cohorts, implementation checks and
    unresolved stress cases retain separate statuses.]
  v(4pt)
}
#if refresh != none {
note(fill: warm)[*Precise performance acceptance: deferred to Hua-Sheng.*
  Final timings will run on his local machine. Shared-host timings are provisional,
  without CPU reservation, and cannot establish the ≤1.05 upper confidence bound.
  #if refresh.at("provisional", default: none) == none {
    [This snapshot contains no new paired timing samples.]
  } else {
    [The explicitly bound shared-host observations below are descriptive;
    unavailable comparisons remain visible.]
  }]
if refresh.at("historical_report", default: none) != none {
  v(5pt)
  note[*Current-source numerical checkpoint.* This report selects #raw(campaign.id).
    Earlier shared-host timings remain in the #link("https://github.com/MadGraphTeam/MadGraph7/blob/madloop_tirust/docs/benchmarkfolder/" + refresh.historical_report.pdf_path)[archived 19-page split-dp-terminal-1 report],
    bound to commit #raw(refresh.historical_report.report_commit.slice(0, 12)).
    Those timings retain their historical source identity. Current performance gates remain deferred.]
}
if refresh.at("next_campaign", default: none) != none {
  v(5pt)
  note[*Current source registration: #raw(refresh.next_campaign.id).*
    Revision #raw(refresh.next_campaign.registered_exporter_revision.slice(0, 12))
    includes the promoted Fortran change. Fresh numerical qualification for its
    #refresh.next_campaign.case_count processes and local precise timing remain pending. The results
    in this PDF belong to the older #raw(campaign.id) candidate.]
}
} else { note(fill: if d.gate.complete and d.gate.passed { pale } else { warm })[
  *Main campaign acceptance: #cell-status(if d.gate.complete and d.gate.passed { "pass" } else if d.gate.complete or known-failures { "fail" } else { "incomplete" }).* #if known-failures and not d.gate.complete { [The remaining campaign is incomplete.] }
  Every main process requires five-backend agreement, an upper 95% paired runtime-ratio bound no greater than 1.05, and passing existing-backend regression checks. Unmeasured cells are shown as —. A failed or incomplete row is not a speedup claim.
] }
#if intermediate != none {
  v(5pt)
  text(8pt, fill: ink2)[The main matrix retains the historical measured candidate.
  Later implementation checks and failed diagnostics appear in the intermediate
  supplement; the optimized final campaign is pending.]
}
#v(7pt)
#text(8pt)[Runtime colour: green = faster, white = parity, red = slower. All displayed ratios are candidate / reference; smaller is faster.]
#v(5pt)
#figure(
  table-style(table(columns: (1.35fr, .7fr, .85fr, .85fr, 1.05fr, .8fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process · correction], [Physics], [Baseline µs/point], [TIRust chain µs/point], [Ratio · 95% interval], [Existing-backend regressions]),
    ..main.map(c => {
      let p = c.performance
      let ratio = p.at("geometric_mean_ratio", default: none)
      let regressions = c.at("existing_backend_regressions", default: ())
      let bounds = regressions.filter(q => q.statistics != none).map(q => q.statistics.upper_95).filter(x => x != none)
      (
        [#text(9pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()
         #text(7pt, fill: ink2)[#if c.id.starts-with("qcd") { [QCD] } else { [EW] } · 2→#c.nfinal · #if c.points == none { [unmeasured] } else { [#c.points points] }]],
        [#cell-status(if refresh == none { checks-status(c.correctness) } else {
          let q = current-dossier.numerical.cases.find(q => q.id == c.id)
          if q.regression_pass == true { "pass" } else if q.regression_pass == false { "fail" } else { q.status }
        })],
        [#text(11pt, weight: "bold")[#time-us(p.at("baseline_median_seconds_per_point", default: none))]],
        [#text(11pt, weight: "bold")[#time-us(p.at("candidate_median_seconds_per_point", default: none))]],
        table.cell(fill: ratio-fill(ratio), [
          #text(11pt, weight: "bold")[#ratio-text(ratio)]#linebreak()
          #if p.at("lower_95", default: none) != none {
            [#fixed(p.lower_95,3)–#fixed(p.upper_95,3)#linebreak()]
          }
          #cell-status(if refresh != none { "deferred" } else if p.passed == none { p.status } else if p.passed { "pass" } else { "fail" })
        ]),
        [#cell-status(if refresh != none { "deferred" } else { checks-status(regressions) })#linebreak()
          #if bounds.len() > 0 { [max. upper#linebreak()#ratio-text(calc.max(..bounds))] }],
      )
    }).flatten(),
  )),
  caption: [Main process matrix. Timings include the actual rescue work. Physics and runtime acceptance are separate.],
)
#v(5pt)
#text(8pt, fill: ink2)[This study measures the new integration against the untouched starting branch, which already includes NinjaRS and earlier Fortran optimizations. It does not attribute those earlier improvements to TIRust.]

#if report.predecessors.len() > 0 {
  pagebreak()
  heading(level: 1)[Retained predecessor failures]
  [The following qualified failures belong to earlier candidate implementations.
  They remain authoritative for those numerical sources and fixed policies.
  Results for the selected campaign appear in the main matrix; a new campaign
  does not replace or erase a previous candidate's failure.]
  v(7pt)
  figure(table-style(table(columns: (1.1fr, 1.3fr, 1.1fr, 1.5fr),
    inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
    table.header([Process · comparison], [Previous candidate source],
      [Retained ratio · 95% interval], [Retained failure evidence]),
    ..report.predecessors.map(p => (
      [#process-label(p)#linebreak()#text(7pt)[#raw(p.label)]],
      [#raw(p.predecessor_id)#linebreak()
       #text(7pt)[revision #raw(p.revision.slice(0,12))#linebreak()
         source SHA #raw(p.source_sha256.slice(0,16))]],
      [#if p.geometric_mean_ratio != none { [#fixed(p.geometric_mean_ratio,5)#linebreak()] }
       #if p.lower_95 != none and p.upper_95 != none {
         [#fixed(p.lower_95,5)–#fixed(p.upper_95,5)#linebreak()]
       }
       #cell-status("fail")],
      [#text(6.7pt)[#raw(p.directory)]],
    )).flatten(),
  )), caption: [All retained qualified ancestor failures, including discovered and unselected attempts. Source identities and every read evidence file are bound in the report build manifest.])
}

#pagebreak()
= Five-backend numerical agreement

The same saved momenta, model parameters, coupling orders and scale are evaluated by all five tools. The regression criterion is #raw("3e-6"), with #raw("1e-8") as the ordinary-point target. Born, finite, simple-pole and double-pole coefficients are retained. Near-zero poles use the existing MadLoop comparison policy. Final accepted DP/QP outcomes and failed evaluations are retained; the histograms do not enumerate internal rescue attempts.

#if current-dossier != none {
  import "campaign-report.typ": show-campaign-numerical
  show-campaign-numerical(current-dossier, d.cases, process-label)
} else {
figure(table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
  inset: (x: 4pt, y: 7pt), stroke: table-stroke, fill: table-fill,
  table.header([Process], ..ids.map(id => [#tool-name(id)])),
  ..d.cases.map(c => (
    [#text(8.6pt, font: "Libertinus Serif")[#process-label(c)]#linebreak()#text(7pt, fill: ink2)[#c.role · #if c.points == none { [unmeasured] } else { [#c.points points] }]],
    ..ids.map(id => {
      let q = get-check(c,id)
      if q == none { [pending] } else {
        [#cell-status(if q.passed == none { q.status } else if q.passed { "pass" } else { "fail" })#linebreak()
         #if q.at("max_relative_error", default: none) != none {
           text(7pt)[#str(q.max_relative_error)]
         }]
      }
    }),
  )).flatten(),
)), caption: [Comparison status and largest retained relative discrepancy. A dash or pending status denotes absent evidence.])
}

#note[
  Agreement is assessed using independent numerical backends and higher precision when a discrepancy occurs. Known difficult scalar-box regions are retained as stress cases. A backend's own success flag does not establish cross-backend agreement.
]

#pagebreak()
= Completed physical pilot checks

These eight separately retained checks each evaluate 100 saved physical points with all five reducers. They use the common strict stability policy and pass the #raw("3e-6") regression criterion. Their timings are ineligible for performance acceptance, and they do not fill the pending timing-campaign rows above. An independent audit reparsed 4,800 evaluations and checked the retained inputs, executable provenance and raw statuses.

#if campaign != none {
  note(fill: warm)[These historical pilots retain their original candidate sources in the pilot index. They do not establish numerical acceptance for the selected campaign #raw(campaign.id).]
}

#figure(table-style(table(columns: (1.5fr, .7fr, .8fr, 1fr, .9fr),
  inset: (x: 4pt, y: 8pt), stroke: table-stroke, fill: table-fill,
  table.header([Process · 100 points], [Five-backend regression], [Ordinary target], [Largest symmetric residual], [TIRust accepted DP / QP]),
  ..pilots.cases.map(c => {
    let t = c.backends.find(b => b.backend == 9)
    (
      [#text(9pt, font: "Libertinus Serif")[#process-label(c)]],
      [#cell-status(if c.all_five_regression_pass { "pass" } else { "fail" })],
      [#cell-status(if c.all_five_ordinary_target_pass { "pass" } else { "miss" })],
      [#scientific(c.max_symmetric_non_negligible_residual)],
      [#t.dp_count / #t.qp_count],
    )
  }).flatten(),
)), caption: [Historical physical pilots, independent of the runtime gate. Residuals exclude poles accepted by the documented small-pole exception. All four existing reducers finally accepted 100 DP results in each row.])

#note[
  *Historical split-dp-terminal-1 QCD 2→3 adjudication.* The original comparison against CutTools DP misses the #raw("1e-8") target at 28 points. CutTools, Ninja and TIRust QP agree within #raw("9.89e-11") on four diagnostic points. Across all 28 target misses, the saved TIRust and COLLIER results meet the target against CutTools QP; CutTools DP misses all 28, while Ninja and NinjaRS miss only point 39. Every comparison still passes the regression criterion. These fixed-QP diagnostics are explicitly unchecked and do not establish automatic rescue stability.
]

The index #raw("experiments/numerical/completed-pilots-index.json") binds 491 retained files and preserves the original comparisons. The detailed QP adjudication is in #raw("experiments/qcd-n3-adjudication/"). Final accepted precision does not enumerate every internal reduction attempt. The remaining three large-process pilots are outside this frozen eight-case index.

#pagebreak()
= Absolute timing by reduction backend

Values below are warmed full-amplitude times in µs per point. Initialization, compilation and model generation are excluded and recorded separately. All tools use matching pole, helicity/color and stability settings. Cache behaviour and actual rescue work remain part of the measurement. Recorded backend/precision histograms identify the final accepted reducer for every timed amplitude; they do not count every internal rescue attempt. Control-process speeds are descriptive; the ≤1.05 runtime gate applies to the eight main processes.

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
)), caption: [Per-backend timing observations. Missing measurements are not represented as zero.])

#if report.ratio_figure != none {
  figure(image(report.ratio_figure, width: 100%),
    caption: [Paired candidate/baseline ratios and 95% intervals; the acceptance limit is 1.05.])
}

#pagebreak()
= Initialization and accepted precision

Each cell gives the first-amplitude initialization plus complete-stream warmup time in seconds, followed by the fraction of timed amplitudes finally accepted in QP. Process startup and parameter-card loading are outside the cold interval. A DP-only label refers to accepted amplitudes; it does not assert that no unsuccessful internal attempt occurred. The warmed timings on the previous page include all rescue work.

#figure(table-style(table(columns: (1.4fr, .8fr, .8fr, .8fr, .8fr, .8fr),
  inset: (x: 4pt, y: 8pt), stroke: table-stroke, fill: table-fill,
  table.header([Process], ..ids.map(id => [#tool-name(id)])),
  ..d.cases.map(c => (
    [#text(8.6pt, font: "Libertinus Serif")[#process-label(c)]],
    ..ids.map(id => {
      let t = get-timing(c,id)
      if t == none { [—] } else {
        let cold = t.at("initialization_and_warmup_seconds", default: none)
        let usage = t.at("actual_backend_precision_counts", default: (:))
        let total = usage.values().fold(0, (a,b) => a+b)
        let qp = usage.pairs().filter(p => p.at(0).ends-with(":QP")).fold(0, (a,p) => a+p.at(1))
        [#if cold == none { [—] } else { [#number(cold) s] }#linebreak()
         #if total > 0 {
           text(7pt, fill: ink2)[#if qp == 0 { [DP only] } else { [QP #fixed(100*qp/total,1)%] }]
         }]
      }
    }),
  )).flatten(),
)), caption: [Cold intervals and final accepted precision. Exact backend/precision counts and point-level results accompany the raw observations.])

#include "poles-timing.typ"

#pagebreak()
= Methods and qualification

== Physical samples and timing protocol

The main matrix contains QCD and electroweak corrections at every multiplicity from 2→1 through 2→4, using #raw("loop_sm") and #raw("loop_qcd_qed_sm_Gmu"). Each main case declares 100 deterministic points. On-shell 2→1 points vary their boost at the physical invariant; higher multiplicities use 1 TeV. Mixed QCD/QED and loop-induced cases supplement the main matrix. Threshold, soft/collinear, small-Gram and complex-mass cases are separate stress controls.

Performance acceptance uses at least seven alternating paired repetitions. The minimum batch target is one second; calibration aims for two seconds to accommodate shared-host speed variation. Actual durations are retained, and every batch must reach 95% of the minimum. The starting revision and candidate share their compiler, library installations, model/cards, point order and physical settings. Timing begins after the campaign's builds finish and uses one allowed CPU. The upper 95% paired ratio bound must be ≤1.05 separately for every main process. Averages cannot hide a failing row. A FULL−SKIP quantity, when reported in supporting data, is an *estimated reduction stage* rather than a pure tensor-kernel stopwatch.

All timing attempts are retained. Only an objectively short series with passing numerical checks can be retried within the same candidate campaign. A qualified runtime failure remains authoritative; a later favorable repetition cannot erase it. Source optimizations require a new candidate identity and campaign.

#let chains = d.at("reduction_chains", default: (:))
#if chains.at("baseline", default: none) != none and chains.at("candidate", default: none) != none {
  [For the eight main processes, the materialized baseline chain is #raw(chains.baseline.map(str).join("|")); the TIRust-first chain is #raw(chains.candidate.map(str).join("|")). Both run the same independent MadLoop stability policy.]
}

The loop-induced controls retain their generated default chain #raw("7|6|1"); their prescribed TIRust-first chain is #raw("9|7|6|1"). These are configured chains, not measurements of which reducer ultimately supplied a result.

The recorded cold interval includes first-amplitude initialization and warming the complete saved point stream; it excludes process startup and parameter-card loading. Numerical objects and linked archives are bound to the retained compiler recipes and hashes. The benchmark runs on a shared host; host-load observations accompany timing pairs, and this study does not claim exclusive use of the machine.

The common benchmark card sets #raw("MLStabThres=1e-6") for both revisions and all reducers, with independent stability checks enabled. This policy was frozen before acceptance measurements after a native stress audit exposed correlated errors below the original threshold. Production retains its #raw("1e-3") default. The tighter estimator does not replace the independent numerical truth gate or guarantee detection of every error.

== Interface, poles and caches

MadLoop selects TIRust explicitly as #raw("MLReductionLib=9"). A Fortran adapter transfers supplied invariants, complex squared masses and the squared renormalization scale to a separate Rust bridge. Tensor moments are permuted into MadLoop order and contracted in Fortran. The quadruple-precision path preserves binary128 inputs and uses MPFR working precision; double-double arithmetic is not labelled quadruple precision.

#raw("LoopPoleMode") selects bare-loop UV/IR residues while preserving the ordinary finite coefficient. The distinct poles-only diagnostic path omits finite scalar roots and finite tensor arithmetic; it preserves double-pole contributions to simple poles and scaleless UV/IR separation. Partial results cannot be used as complete physical amplitudes. Cache and precision stores are separated by evaluation mode, and stability rechecks use independent values.

#let quals = if intermediate == none { d.at("qualification", default: ()) } else { intermediate.qualification }
#if quals.len() > 0 {
  table-style(table(columns: (1.3fr, .6fr, 3fr), inset: 5pt, stroke: table-stroke, fill: table-fill,
    table.header([Qualification], [Status], [Evidence]),
    ..quals.map(q => ([#q.name], [#cell-status(q.status)], [#q.evidence])).flatten()))
} else {
  note(fill: warm)[Detailed interface, cache, QP, poles-only, and relocation qualification records have not yet been incorporated in this data release. Their absence is not a passing result.]
}

== Earlier measured Fortran change

The selected optimization computes ordered momentum prefixes once, retaining the original zero seed and addition order. For six propagators it reduces prefix additions from 60 to 20. At each of #raw("-O0") and #raw("-O3"), 481,536 comparisons passed bitwise output and IEEE-exception checks across DP, QP and four rounding modes. Isolated six-propagator kernel median paired ratios were 0.443/0.347 (DP/QP, #raw("-O0")) and 0.703/0.356 (#raw("-O3")). These are shared-host microbenchmarks, not physical-amplitude speedups. Full-wrapper measurements were smaller and less consistent. Raw observations and source are retained in #raw("experiments/fortran-prefix/").

== Retained numerical limitations

The bridge's 66 DP/QP stress evaluations include 14 explicit DP refusals and one served DP tensor coefficient above the diagnostic #raw("3e-6") criterion. All 33 QP evaluations pass. For the failing massive rank-five box, native MadLoop reversal checks at the default #raw("MLStabThres=1e-3") can accept correlated DP errors. At #raw("1e-6"), the actual production rescue block with real tensor callbacks switches to QP and resolves the discrepancy below #raw("1.2e-15"). The fixture, failing components, native check values and QP adjudication remain under #raw("qualification/tirust-bridge/stability-audit/"). This isolated control-flow test omits physical helicity/filter initialization and does not replace matrix-element comparisons.

Pole-only tests prove that finite evaluators and finite tensor arithmetic are skipped. This does not imply every scalar topology is faster: illustrative warm scalar measurements include slower simple massless triangles and boxes, alongside faster massive and soft examples. A completely collapsed massless scalar box returns an explicit error. No unsupported request is counted as accelerated execution.

== Reproduction and provenance

The accompanying #raw(report.cases_path) fixes the matrix and thresholds; #raw(summary-path) is derived from retained observations. #if campaign != none { [The selected manifest is #raw(manifest-path). Its ancestor summary snapshots and qualified failures retain the previous candidates' identities separately.] } The source plan, baseline manifest, dependency revisions, build environment, and raw measurement paths accompany this report. Rebuilding the PDF does not rerun numerical measurements.

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

#if intermediate != none {
  import "intermediate.typ": show-intermediate
  show-intermediate(intermediate)
}

#if current-dossier != none {
  import "campaign-report.typ": show-campaign-details
  show-campaign-details(current-dossier, d.cases, process-label)
}
