#import "style.typ": *
#let d=json("timing-report-data.json")
#show: body=>report-style(body,date:d.at("date",default:"closed timing snapshot"),
  revision:d.at("source_label",default:"source identities pending"))
#set heading(numbering:none)
#let ids=(1,6,7,8,9)
#let four=d.cases.find(c=>c.case_id=="qcd_n4_uux_epem_mupmum")
#let names=("1":"CutTools","6":"Ninja","7":"COLLIER","8":"NinjaRS","9":"TIRust")
#let sig(x,digits:3) = {
  if x==none { return [—] }
  if x==0 { return [0] }
  let e=calc.floor(calc.log(calc.abs(x),base:10))
  if e < -2 or e > 3 { [#fixed(x/calc.pow(10,e),digits - 1) × 10#super(str(e))] }
  else { fixed(x,calc.max(0,digits - 1 - e)) }
}
#let lane(c,id) = if c.at("summary",default:none)==none { none } else { c.summary.backends.at(str(id),default:none) }
#let check(c,id) = {
  let n=c.at("numerical_validation",default:none)
  if n==none { (:) } else { n.at("backends",default:(:)).at(str(id),default:(:)) }
}
#let eligible(c,id) = c.status=="closed" and lane(c,id)!=none and check(c,id).at("regression_pass",default:false)==true
#let points(c) = if c.at("summary",default:none)==none { [pending] } else { [#c.summary.points_per_traversal] }
#let usage(c,id) = {
  let item=lane(c,id)
  if item==none { return [—] }
  let counts=item.aggregate_backend_precision_counts
  let dp=0
  let qp=0
  for (key,value) in counts {
    if key.ends-with(":DP") {dp+=value}
    if key.ends-with(":QP") {qp+=value}
  }
  [#dp / #qp]
}
#let runtime(c,id) = {
  let item=lane(c,id)
  if item==none { return [—] }
  [#sig(item.median_seconds_per_point*1000) ± #sig(item.mad_seconds_per_point*1000)]
}
#let ratio(c,id) = if eligible(c,id) and eligible(c,8) { lane(c,id).median_ratio_to_8 } else { none }
#let initialization(c,id) = {
  let values=c.at("initialization_seconds",default:none)
  if values==none { [—] } else { sig(values.at(str(id),default:none)) }
}
#let target-count = {
  let n=0
  for c in d.cases { for id in ids { if check(c,id).at("ordinary_target_pass",default:false)==true { n+=1 } } }
  n
}

= Five-backend Standard Model timings

#note(fill:warm)[*Descriptive shared-host measurements.*
These full-amplitude timings compare explicit CutTools, Ninja, COLLIER, NinjaRS
and TIRust on the selected saved cohorts. There is no CPU reservation and no
confidence-interval or final local-performance acceptance claim. The earlier
39 pages retain their separately bound scientific and timing history.]

#text(9pt)[#d.machine_note]

Elapsed time is the unchanged Fortran `SYSTEM_CLOCK` wall clock, divided by
the actual `points × loops`. Each backend has ten position-balanced cycles
(five Latin positions followed by reversal), with loops fixed after two
retained calibrations. The nominal batch target is two seconds; actual batch
durations and counts remain in the closed evidence. The table reports the
median and *unscaled* median absolute deviation, not an error bar on the mean.
Ratios are medians of ten within-cycle backend / NinjaRS8 ratios.
Actual timed batches range from
#sig(calc.min(..d.actual_timed_batch_durations.map(x=>x.minimum_seconds))) to
#sig(calc.max(..d.actual_timed_batch_durations.map(x=>x.maximum_seconds))) seconds;
integer full-cohort traversal counts can exceed the nominal target.

#block(breakable:true)[#figure(table-style(table(columns:(1.20fr,.72fr,.68fr,1.25fr,.65fr,.90fr,.90fr),
  inset:(x:3pt,y:5pt),stroke:table-stroke,fill:table-fill,
  table.header([Process · cohort], [Reducer], [READY s], [Wall ms/point · median ± MAD], [Ratio / 8], [Accepted DP / QP], [Status]),
  ..d.cases.map(c=>ids.map(id=>{
    let r=ratio(c,id)
    let q=check(c,id)
    let state=if c.status!="closed" {c.status} else if q.at("regression_pass",default:none)==false {"numerical fail"}
      else if q.at("regression_pass",default:none)==none {"check pending"}
      else if q.at("ordinary_target_pass",default:none)==false {"3e-6 pass; target miss"} else {"3e-6 pass"}
    let ink=if not eligible(c,id) {state} else if q.at("ordinary_target_pass",default:none)==false {"pending"} else {"pass"}
    (if id==1 {[#c.label #linebreak()N=#points(c)]} else {[]},
     [#names.at(str(id))],initialization(c,id),runtime(c,id),
     table.cell(fill:ratio-fill(r),if r==none {[—]} else {[×#sig(r)]}),
     usage(c,id),text(fill:status-ink(ink),state))
  }).flatten()).flatten()
)),caption:[Warmed full-amplitude wall time. Ratios below one mean that backend is faster than NinjaRS8. All actual saved-cohort sizes and accepted output-call counts are retained.])]

#text(9pt)[The separate ordinary 1e-8 target passes in #target-count / 20 lanes.]

READY time includes the first amplitude and initialization/learning, excluding
SETPARA and process startup; it is separate from warmed timing. Accepted DP/QP
counts cover timed output calls and do not recover unobserved internal retries
or floating-point exception flags. Numerical checks and timing status are
separate; failed or unavailable comparisons remain explicit. Coefficient
checks cover the retained final 100-point traversal of learning, each
calibration and each timed batch; accepted-status counts cover all timed
repetitions.

The $u overline(u) -> Z$ cohort varies physical on-shell Z boosts. The other
three selected cohorts use 1 TeV. These labels follow actual saved-point
physics and generation records, not the global configuration energy field.

#pagebreak()
= Timing comparison and scope

#if sys.inputs.at("include-plots",default:"false")=="true" {
  figure(image("timing-comparison.png",width:100%),caption:[Absolute warmed wall timing for each case and within-cycle ratios relative to NinjaRS8. Whiskers show unscaled MAD, not confidence intervals. Green means a smaller backend/NinjaRS8 ratio, red a larger ratio. Unavailable cells are not zero.])
} else { note[Plot source is prepared. No timing plot is included until the closed data are selected and rendered.] }

The fresh four-lepton QCD $2 -> 4$ generation has maximum loop
N=#four.loop_topology.max_denominators and rank=#four.loop_topology.overall_max_rank,
across #four.loop_topology.loop_call_groups loop-call groups, as reparsed from
its compiled source text. Four final particles do not imply six
propagators in each loop. The separately validated ZZgg process retains its
full numerical coverage and precise local timing TODO. These shared-host comparisons do not
replace the full registered numerical matrix or Hua-Sheng's precise local
per-process timing campaign. The agreed upper-95% runtime-ratio bound ≤1.05,
source-matched baseline/rescue-chain comparison and existing-backend regressions
remain separate acceptance items.

#if d.at("checked_cms20",default:none)!=none {
  let c=d.checked_cms20
  if c.at("closed",default:false) and c.at("authenticated",default:false) {
    note[*Separate private checked-CMS20 result.* #c.scope
      #link(c.evidence_url)[Closed evidence]. This does not qualify the production timing inputs or generic execution.]
  }
}

#text(8pt)[Closed timing evidence and reproduction commands: #d.evidence_label.
Clock, source/runtime/card/cohort pins, actual durations, calibration and all
accepted-status/precision histograms are retained with that packet.]
