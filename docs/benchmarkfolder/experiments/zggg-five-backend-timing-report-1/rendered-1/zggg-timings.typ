#import "style.typ": *
#let d = json("zggg-timing-data.json")
#assert(d.schema == "madloop-zggg-five-backend-timing-report-data-v1")
#assert(d.cases.len() == 1)
#let c = d.cases.first()
#assert(c.case_id == "qcd_n4_uux_zggg")
#assert(c.model == "loop_sm" and c.process == "u u~ > z g g g [virt=QCD]")
#show: body => report-style(body, date:d.date, revision:d.source_label)
#set heading(numbering:none)
#let ids = (1,6,7,8,9)
#let names = ("1":"CutTools", "6":"Ninja", "7":"COLLIER", "8":"NinjaRS", "9":"TIRust")
#let sig(x, digits:4) = {
  if x == none { return [—] }
  if x == 0 { return [0] }
  let e = calc.floor(calc.log(calc.abs(x),base:10))
  if e < -2 or e > 3 { [#fixed(x/calc.pow(10,e),digits - 1) × 10#super(str(e))] }
  else { fixed(x,calc.max(0,digits - 1 - e)) }
}
#let lane(id) = c.summary.backends.at(str(id),default:none)
#let check(id) = c.numerical_validation.backends.at(str(id),default:(:))
#let good(id) = lane(id) != none and check(id).at("regression_pass",default:false) == true
#let ratio(id) = if good(id) and good(8) { lane(id).median_ratio_to_8 } else { none }
#let usage(id) = {
  let l = lane(id)
  if l == none or l.aggregate_backend_precision_counts.len() == 0 { return [—] }
  l.aggregate_backend_precision_counts.pairs().sorted(key:p=>p.first()).map(p=>[#p.first()=#p.last()]).join([; ])
}
#let runtime(id) = {
  let l = lane(id)
  if l == none or l.at("timing_available",default:false) == false or l.median_seconds_per_point == none or l.mad_seconds_per_point == none { return [unavailable] }
  [#sig(l.median_seconds_per_point * 1000) ± #sig(l.mad_seconds_per_point * 1000)]
}
#let gate(v, target:false) = {
  if v == true { text(fill:status-ink("pass"))[pass] }
  else if v == false { text(fill:status-ink("fail"))[#if target {[miss]} else {[fail]}] }
  else { [unqualified] }
}

= QCD $u overline(u) -> Z g g g$: five reducers

#note(fill:warm)[*Descriptive timings on a shared host.*
This $2 -> 4$ study requests all five reducers; TIRust times out before READY.
There is no CPU reservation or formal performance-acceptance claim. The preceding 44 pages are preserved.]

#text(9pt)[#d.machine_note]

CLI generation: `loop_sm; u u~ > z g g g [virt=QCD]`.
Each traversal uses all *#c.summary.points_per_traversal saved points* at $sqrt(s) = #c.cohort.sqrt_s_GeV$ GeV.
#d.pilot_note
#d.cycle_note

#figure(table-style(table(columns:(.85fr,.64fr,1.24fr,.95fr,1.38fr,1.04fr),
  inset:(x:3pt,y:5pt),stroke:table-stroke,fill:table-fill,
  table.header([Reducer], [READY s], [Warmed ms/point\ median ± MAD], [Ratio / 8],
    [Accepted timed\ backend:precision calls], [3e-6 / 1e-8]),
  ..ids.map(id=>{
    let r=ratio(id)
    let q=check(id)
    ([#names.at(str(id))],if id == 9 { [600 s\ timeout] } else { sig(c.initialization_seconds.at(str(id),default:none)) },runtime(id),
      table.cell(fill:ratio-fill(r),if r == none {[—]} else {
        let ci=lane(id).at("paired_ratio_ci95_to_8",default:none)
        [×#sig(r)#if ci != none { [#linebreak()#text(6.8pt)[[#sig(ci.lower), #sig(ci.upper)]]] }]
      }),usage(id),
      if id == 9 { [unqualified] } else { [#gate(q.at("regression_pass",default:none)) / #gate(q.at("ordinary_target_pass",default:none),target:true)] })
  }).flatten()
)),caption:[Warmed full-amplitude wall time: median ± unscaled MAD. Ratios use matched cycles relative to NinjaRS8; bracketed 95% intervals are descriptive.])

#d.clock_note
#d.at("interval_note",default:"")
Actual timed batches span #sig(d.actual_batch_seconds.minimum)–#sig(d.actual_batch_seconds.maximum) s.
READY includes the first amplitude and warmup, excluding startup and SETPARA. #d.initialization_note

== Numerical results and precision

#d.accuracy_note
#d.failure_note

#d.precision_note
Accepted counts do not expose internal retries or exception flags.
#d.policy_note

#note[Hua-Sheng’s local TODO remains: source-matched baseline/rescue-chain comparisons and upper-95% paired runtime-ratio bound ≤1.05 for every main process.]

#text(8pt)[TIRust source: `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`;
DSO SHA-256 (runtime scope described above): `85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0`.
#d.evidence_label
CSV retains full precision. Style: pyAmpliCol `7266c1600f6e…`; complete revision and attribution are in style-provenance.json.]
