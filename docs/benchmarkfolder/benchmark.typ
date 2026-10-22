// Final measured report; regenerate offline using final-report-v1/reproduce.py.
#import "final-report-v1/style.typ": *
#show: body => two-tool-style(body)
#let d=json("final-report-v1/report-data.json")
#let correctness=d.correctness
#let labels=(
 "qcd-1": $u overline(u) arrow Z$, "ew-1": $u overline(d) arrow W^+$,
 "qcd-2": $g g arrow t overline(t)$, "ew-2": $e^+ e^- arrow mu^+ mu^-$,
 "qcd-3": $u overline(u) arrow Z g g$, "ew-3": $e^+ e^- arrow t overline(t) gamma$,
 "qcd-4": $u overline(u) arrow Z Z g g$,
 "ew-4": $e^+ e^- arrow mu^+ mu^- gamma gamma$,
 "supplement-gg-h": $g g arrow H$, "supplement-gg-hh": $g g arrow H H$,
 "supplement-gg-hhh": $g g arrow H H H$,
 "supplement-gg-hhhh": $g g arrow H H H H$,
 "control-gg-h": $g g arrow H$, "control-gg-hh": $g g arrow H H$,
 "control-gg-hhh": $g g arrow H H H$,
 "control-gg-hhhh": $g g arrow H H H H$,
 "profile-gg-hhh": $g g arrow H H H$,
 "profile-gg-hhhh": $g g arrow H H H H$,
)
#let label(id)=labels.at(id,default:[#id])
#let us(x)=fmt(x*1000000)
#let panel(body)=block(fill:pale,inset:8pt,radius:4pt,width:100%,body)
#let named-status(s)=(
 "complete_comparable":"Matched sessions complete",
 "complete_incomparable":"Complete; stage scopes differ",
 "failed_attempt":"Attempt failed",
 "pending":"Not yet observed",
 "comparable":"Matched positive timings",
 "nonpositive_samples":"Signed or zero estimate; no ratio",
 "incomparable_stage_scope":"Stage scopes differ; no ratio",
).at(s,default:s)
#let ci(x)=[#us(x.at(0))–#us(x.at(1))]

#align(center)[
 #text(size:25pt,weight:"bold",fill:navy)[MadLoop: Ninja and NinjaRS]
 #v(4pt)#text(size:12pt)[Standard Model · reduction, total time and numerical stability]
 #v(4pt)#text(size:9pt,fill:luma(100))[#d.date · actual measured components]
]
#v(10pt)
#panel[
 Three measurements answer distinct questions. The stock MG7 CLI reports an estimated reduction stage, a numerator-stage reference, and total time per phase-space point for the current exports. A separate six-process test compares warmed, full-helicity, rescue-enabled MadLoop against original revision #raw(d.light.literal_original_revision.slice(0,12)). Native-DP transformation tests compare Ninja and NinjaRS on identical random proposal streams with rescue disabled. Their samples, timing boundaries and error estimates are kept separate.
]

= Methods and sample scopes

The main CLI table uses full helicity/color sums. Its *estimated reduction* is FULL−SKIP, including master integrals, callbacks and Laurent assembly. The *numerator stage (SKIP)* also contains non-reduction driver, Born and kinematic work; it is not an instruction-level tensor stopwatch. Adaptive traversal counts and raw floating timing metadata are retained for every stage. Three independent sessions per tool give empirical median-bootstrap intervals, with no small-sample coverage guarantee. All 72 sessions were retained. Ten processes have matching scopes across all six sessions. For the two on-shell SM cases, the selected timing helicity and filter changed after session 0; their per-tool medians remain shown, but the unchanged conservative reader suppresses aggregate ratios as NA. These direct-DP timings supply no timed rescue histogram. Optional polarized rows are shown only for positive saved helicity weight; a zero-weight selected configuration supplies no physical reduction-speed comparison.

The random-DP panels use 100 declared points for each process and tool. Born processes use full sums; $g g arrow H$ and $g g arrow H H$ use the preregistered all-minus helicity. On-shell $2 arrow 1$ samples vary boosts at one physical invariant. Other cases use 1 TeV and a declared massive-RAMBO proposal with fixed cuts, rather than uniform massive phase-space integration. The empirical tail resolution is 1%; a zero count has an exact one-sided 95% binomial reference upper bound of 2.95%. A missing Ninja tail produces a ratio gap, without adding an epsilon.

Typography, margins, navy headings, pale panels and logarithmic green–white–red timing cells follow the #link("https://github.com/alphal00p/ninjars/blob/ca1009e89ecb03d3ff477f57b170241f7a9f8669/docs/benchmark/benchmark.typ")[NinjaRS report]. Transformation comparisons follow #link("https://arxiv.org/pdf/1604.01363")[Hirschi–Peraro]; the present finite sample is stated explicitly rather than identified with the paper's larger surveys.

// The accepted stock-CLI series will be inserted here through the separately
// validated CLI consumer. Its entire section is absent before that series
// exists: no missing values are printed as measurements.
#if d.stock_cli!=none {
 pagebreak()
 heading(level:1)[MG7 CLI: estimated reduction and total time]
 // The final data view supplies only accepted raw-series observations and
 // display-only optional-polarized eligibility. It does not alter authority
 // statistics. Full-sum tables are the primary comparison.
 for group in d.stock_cli.pages {
  if group.index>0 {pagebreak()}
  heading(level:2)[#group.title]
  text(size:10pt)[Large values: *NinjaRS*. The comparison line gives Ninja; all times are in µs per point. Each interval uses the three actual sessions. Gray NA cells retain per-tool values without a cross-session ratio.]
  v(6pt)
  table(columns:(1.05fr,1.25fr,1.25fr,1.25fr),
   inset:(x:5pt,y:9pt),stroke:.3pt+luma(225),
   fill:(x,y)=>if y==0 {navy} else if x==0 and calc.rem(y,2)==0 {pale} else {none},
   table.header(text(fill:white)[Process / actual calls],text(fill:white)[Estimated reduction],
    text(fill:white)[Numerator stage (SKIP)],text(fill:white)[Total]),
   ..group.rows.map(c=>{
    let cells=(
     [#set par(justify:false)
      #label(c.id)#linebreak()#text(size:10pt)[3 sessions per tool]#linebreak()#text(size:9.5pt)[#c.stage_counts]],
    )
    for q in ("reduction_estimate","construction","total") {
     let v=c.quantities.at(q); let r=v.ninjars_over_ninja
     cells.push(table.cell(fill:if r.ratio==none {pale} else {ratio-fill(r.ratio)},[
      #text(size:18pt,weight:"bold",fill:navy)[#us(v.backend.at("8").median_seconds_per_point) µs]
      #linebreak()#text(size:10.5pt)[95% CI #ci(v.backend.at("8").confidence_interval95)]
      #linebreak()#text(size:10.5pt)[MAD #us(v.backend.at("8").median_absolute_deviation) µs]
      #linebreak()#text(size:10.5pt)[Ninja #us(v.backend.at("6").median_seconds_per_point) µs]
      #linebreak()#text(size:10.5pt,weight:"semibold")[#if r.ratio==none {[NA · scopes differ]} else {[×#fixed(r.ratio,3)]}]
      #linebreak()#text(size:10pt)[#if r.ratio!=none {[95% CI #fixed(r.confidence_interval95.at(0),3)–#fixed(r.confidence_interval95.at(1),3)]}]
     ]))
    }
    cells
   }).flatten(),
  )
  v(6pt)
  text(size:9.5pt)[N lists the actual unpolarized traversals in sessions 0/1/2. FULL and SKIP use the same N; polarized stages use 2N. Exact per-session phase counts, selected helicities, filter hashes and raw timings remain in JSON.]
 }

}

#pagebreak()
= MG7 profile: process-indexed timings
#image("final-report-v1/timing.svg",width:100%)
#v(5pt)
#text(size:9pt)[Circles denote the estimated reduction stage and squares the total. Open markers are Ninja; filled markers are NinjaRS. Bars give empirical independent-session 95% intervals. No line connects unrelated process families. The ratio panel preserves the large uncertainty of some three-session SM estimates; QCD1/EW1 remain NA because the selected helicity/filter changed between sessions. The numerator stage and exact phase counts are given in the preceding tables.]
#v(7pt)
#table(columns:(.4fr,1.5fr,.4fr,1.5fr),inset:3pt,stroke:.3pt+luma(225),
 ..(("qcd-1","ew-1"),("qcd-2","ew-2"),("qcd-3","ew-3"),("qcd-4","ew-4"),("control-gg-h","control-gg-hh"),("profile-gg-hhh","profile-gg-hhhh")).enumerate().map(((i,ids))=>(str(i*2+1),[#label(ids.at(0))],str(i*2+2),[#label(ids.at(1))])).flatten(),
)

#pagebreak()
= Original MadLoop comparison: six warmed processes

Each row uses the same ten fixed prefix points, seven adjacent alternating pairs, full helicity/color sums, poles and normal rescue. The denominator is original MadLoop with Ninja at #raw(d.light.literal_original_revision.slice(0,12)); the numerator is the installed NinjaRS cohort in the provenance records. A 1.05 ratio is shown as a reference for this sample. The former heavy eight-process runtime requirement was withdrawn when the measurement scope was narrowed.

#v(7pt)
#table(columns:(1.15fr,1fr,1fr,1.5fr),inset:(x:5pt,y:6pt),stroke:.3pt+luma(225),
 fill:(x,y)=>if y==0 {navy} else if calc.rem(y,2)==0 {pale} else {white},
 table.header(text(fill:white)[Process],text(fill:white)[Original Ninja#linebreak()µs / point],
  text(fill:white)[NinjaRS#linebreak()µs / point],text(fill:white)[NinjaRS / original Ninja]),
 ..d.light.rows.map(c=>{
  let t=c.timing; let r=t.runtime_ratio; let ci=t.ratio_confidence_interval_95
  ([#label(c.id)#linebreak()#text(size:8.5pt)[#c.points points · #c.pairs pairs]],
   [#text(size:14pt,weight:"bold",fill:navy)[#us(t.measurements.baseline.median)]#linebreak()#text(size:8.5pt)[MAD #us(t.measurements.baseline.median_absolute_deviation)]],
   [#text(size:14pt,weight:"bold",fill:navy)[#us(t.measurements.candidate.median)]#linebreak()#text(size:8.5pt)[MAD #us(t.measurements.candidate.median_absolute_deviation)]],
   table.cell(fill:ratio-fill(r),[#text(size:16pt,weight:"bold",fill:navy)[×#fixed(r,3)]#linebreak()#text(size:9pt)[95% CI #fixed(ci.at(0),3)–#fixed(ci.at(1),3)]]))
 }).flatten(),
)

#v(8pt)
The 0.5 s setting is the calibration target used once to select the loop count. Actual warmed commands lasted #fixed(d.light.actual_command_seconds_range.at(0),3)–#fixed(d.light.actual_command_seconds_range.at(1),3) s. Saved status vectors describe the final traversal of each command; they do not form a complete histogram over every adaptive-loop evaluation. All selected final traversals returned normal DP status for both tools.

#let groups=(
 ("On-shell 2→1", "on-shell"), ("Two final-state particles", "two-particle"),
 ("Three final-state particles", "three-particle"), ("Loop-induced amplitudes", "loop-induced"),
)
#for group in groups {
 pagebreak()
 heading(level:1)[Native-DP stability: #group.at(0)]
 image("/"+d.stability.assets.at("assets/combined-"+group.at(1)+".svg"),width:100%)
 v(4pt)
 text(size:9pt)[The combined spread uses the same original, reversed-direction and Lorentz-transformed point. Hollow triangles mark zero-event upper references, not observed failures. All observations in this selected sample were finite. Full direction-only and Lorentz-only vector panels and exact tail counts accompany the report.]
}

#pagebreak()
= Numerical observations and practical limits

#table(columns:(1.25fr,.8fr,1.15fr,.9fr),inset:(x:4pt,y:5pt),stroke:.3pt+luma(225),
 fill:(x,y)=>if y==0 {navy} else if calc.rem(y,2)==0 {pale} else {white},
 table.header(text(fill:white)[Process],text(fill:white)[Finite / N#linebreak()Ninja · NRS],
  text(fill:white)[Largest spread#linebreak()Ninja · NRS],text(fill:white)[Saved invalid flags#linebreak()Ninja · NRS]),
 ..("qcd-1","ew-1","qcd-2","ew-2","qcd-3","ew-3","supplement-gg-h","supplement-gg-hh").map(id=>{
  let c=d.stability.counts.at(id); let a=c.tools.at("6"); let b=c.tools.at("8")
  let ha=d.stability.hardware.at(id).tools.at("6").hardware_exception_observations
  let hb=d.stability.hardware.at(id).tools.at("8").hardware_exception_observations
  ([#label(id)], [#a.classifications.combined.finite / #c.N#linebreak()#b.classifications.combined.finite / #c.N],
   [#a.maximum_spread_display#linebreak()#b.maximum_spread_display],
   [#ha.ieee_invalid_seen · #hb.ieee_invalid_seen#linebreak()#text(size:8pt)[300 calls / tool]])
 }).flatten(),
)

#v(8pt)
Saved hardware IEEE-invalid flags occur on 300 transformed observations for NinjaRS in $g g arrow t overline(t)$, $e^+ e^- arrow mu^+ mu^-$, $u overline(u) arrow Z g g$, $e^+ e^- arrow t overline(t) gamma$ and $g g arrow H H$. Plain-prefix checks reproduce the flags. Saved divide-by-zero/overflow, algorithmic FPE/rejections, nonfinite results and MP/compensated use are zero. Hardware flags are retained separately from numerical rejection by the frozen observer.

Transformation self-consistency and reference agreement do not provide a scalar finite-part analytic oracle. Analytic pole controls, higher-precision convergence, installation/export checks and software regressions retain their own concrete evidence and scope.

#pagebreak()
= Numerical agreement and normal rescue

The core comparison includes both $2 arrow 4$ processes. Born, finite, single-pole, double-pole and split-order coefficients passed the retained common-point tolerance. Every case has one fixed 100-point stream, calibration and two repeated observations; the coefficient rows and status histograms repeat exactly. The counts below describe one traversal of each unique stream, rather than treating repeated traversals as additional independent points.

#v(8pt)
#table(columns:(1.4fr,.65fr,.8fr,.8fr,.85fr),inset:(x:5pt,y:6pt),stroke:.3pt+luma(225),
 fill:(x,y)=>if y==0 {navy} else if calc.rem(y,2)==0 {pale} else {white},
 table.header(text(fill:white)[SM process],text(fill:white)[Points],text(fill:white)[NinjaRS DP],text(fill:white)[NinjaRS QP rescue],text(fill:white)[Agreement]),
 ..("qcd-1","ew-1","qcd-2","ew-2","qcd-3","ew-3","qcd-4","ew-4").map(id=>{
  let c=correctness.core.at(id)
  ([#label(id)],[#c.points_per_unique_stream],[#c.normal_dp],[#c.automatic_qp],[#c.agreement])
 }).flatten(),
)
#v(7pt)
Across these 800 unique points NinjaRS returned DP code 218 for 768 points and genuine automatic QP-rescue code 322 for 32. This normal-controller sample is separate from the rescue-disabled DP figures and from the CLI cache; its rescue frequencies are not inserted into timing cells that did not save full timed-call status histories.

#v(9pt)
#panel[
 The additional $g g arrow H H H$ and $g g arrow H H H H$ controls passed matched Ninja6/NinjaRS8 full-sum, full-pole, normal-rescue comparisons at two fixed physical points per process, with zero Born terms. Their four native calls succeeded. A versioned, read-only finalizer corrected an analysis-only Python import error after collection; it repeated no native call and preserved the original parent exit 1. This is N=2 correctness evidence, distinct from the requested CLI timing sessions and from random stability.
]

// Final assembly appends the concrete installation/precision/focused/software
// proof summaries and historical research attribution once their final current
// bound records are selected. No old optimization result is relabelled as a
// timing measurement of the installed final runtime.

#pagebreak()
= Integration, precision and software validation

NinjaRS is selected with #raw("MLReductionLib=8") through process-prefixed Fortran wrappers and a versioned Rust C ABI. Installation uses #raw("install ninjars"); after updating the checkout, #raw("install ninjars --update") installs its pinned sources and lockfile transactionally. The measured numerical source is #raw("856e76e66961"), with NinjaRS #raw("aa6a6307") and oneloopmaster #raw("8ec256fc"); the final CLI observer is #raw("464b441fe04d"). The later NinjaRS main #raw("ca1009e") contains report-only changes: all 582 non-report source entries match the measured pin. The final watch is dated 22 October 2026.

#v(7pt)
#table(columns:(1fr,3.3fr),inset:(x:6pt,y:6pt),stroke:.3pt+luma(225),
 fill:(x,y)=>if calc.rem(y,2)==0 {pale} else {white},
 [Installation / export], [Actual CLI update and compiled installed smoke; relocated standalone DP/QP/normal-controller returns 110/120/218. FKS/NLO passed 20/20 pole checks at its documented $10^(-5)$ smoke tolerance, plus FKS-export soft/collinear checks with zero failures. This is separate from the stricter coefficient comparisons; no shower validation is claimed.],
 [Focused physics], [45 persistent/fresh comparison pairs, 5,400 Born/Laurent/split components, cover complex masses, mixed orders and near-singular automatic rescue. Nine additional fixed two-point controls cover explicit QP direction and reused/finalized state changes; the separate normal-HH anchor supplies the normal-controller comparison.],
 [Genuine precision], [The numerator and improved kinematics are recomputed in Fortran binary128. Forty-digit input records recover each original 113-bit value before exact widening; reduction and scalar masters use 256-bit arithmetic. Actual installed-ABI controls contain 64 calls. All 1,392 retained reduction requests converge at 256/384 bits, including complete Laurent, CC and same-basis R1 components; the largest scaled Laurent difference is $2.73 times 10^(-62)$.],
 [Software], [The integrated selected-source campaign ran 293 tests: 292 passed and one unchanged conditional skip requiring a generated QCD2 tree. The final observer/resource-path successor passed 95 affected software checks. These overlapping software scopes are not summed as independent tests, and do not replace the physical campaigns.],
)

#v(8pt)
The standard Rust suite passed 24 checks with 7 ignored. When explicitly enabled, two ignored diagnostics retain the historical bit-identity mismatch and the unresolved exceptional direct-provider case.

The 256-bit calculation does not make the Fortran numerator 256-bit accurate. Returning through 40-digit output records can round to an adjacent binary128 value: 16 of 22,272 examined fields showed this, with largest Laurent-scaled difference $4.20 times 10^(-36)$. Historical replay is deliberately not labelled bitwise identical: 468 binary128 records changed, and the original bit-identity test exit 101 remains retained. The maximum scaled complete-Laurent difference from historical outputs is $1.48 times 10^(-34)$; a much larger basis-dependent R1 diagnostic change is not a change of that physical Laurent coefficient.

Direct scalar-provider exceptional-input controls still retain two unsafe successes upstream. The tested full-bridge planar calls were accurate in DP or safely rejected in MP; that bridge behavior does not resolve the direct-provider finding. The retained source-bound receipts distinguish this limitation from accepted MadLoop amplitudes and from the finite-sample stability plots.

= Measured Fortran work and retained research

The accepted allocation change measured a 3.15% whole-process improvement in the historical QCD3 study (ratio 0.9685, 95% interval 0.9569–0.9936). That study used generated source #raw("f4bffc61") and archive pair #raw("832fc0d1 → 366ae91d"), distinct from the final installed #raw("b0d69deb") archive. Its source-body attribution and exact records accompany the report. Earlier 9.24% DP callback-kernel and 3.19% Ninja6 Fortran-QP prototype results have different boundaries and are not additive. A separate 1,392-request Float256 native/interpreted screen had intervals crossing parity and was not promoted. The actual final timings above, including the slower EW2 ancillary ratio, determine the reported performance rather than these historical experiments.

#pagebreak()
= First-session output, compilation and initialization

The following captured CLI costs provide preparation context for the per-point timings. Every cell gives *Ninja6 above NinjaRS8*, in seconds, for session 0 of that tool and process. These are direct metadata values, rounded to three significant digits for display; exact values and source hashes are in #raw("final-report-v1/preparation-costs.csv"). They are not three-session statistics.

#v(8pt)
#table(columns:(1.35fr,1fr,1fr,1fr,1fr,.9fr),inset:(x:5pt,y:6pt),stroke:.3pt+luma(225),
 fill:(x,y)=>if y==0 {navy} else if calc.rem(y,2)==0 {pale} else {white},
 table.header(text(fill:white)[Process],text(fill:white)[Output / export],text(fill:white)[HELAS / model compile],text(fill:white)[Initial process compile],text(fill:white)[Initialization],text(fill:white)[Boot]),
 ..d.preparation.rows.map(c=>{
  let values=([#label(c.id)],)
  for key in d.preparation.fields {
   values.push([#c.tools.at("6").at(key).display#linebreak()#c.tools.at("8").at(key).display])
  }
  values
 }).flatten(),
)
#v(9pt)
#text(size:9.5pt)[The exact captured keys are #raw("Process_output"), #raw("HELAS_MODEL_compilation"), #raw("Process_compilation"), #raw("Initialization") and #raw("Booting_time"), respectively. Output/export is not labelled as standalone symbolic generation. The costs have different boundaries and are not asserted to form an additive total. Additional timed-stage recompilations and complete phase metadata remain in the raw records; boot subtraction and adaptive traversal counts retain their original CLI meanings.]

#pagebreak()
= Provenance and reproduction

The final profile study contains 72 accepted command records: twelve processes, three sessions and tools 6/8. The first session creates a 20-point CLI DP cache, with zero QP entries; later sessions reuse it. This cache is not a matched-point random-stream authority or a complete timed-call status history. Three earlier Ninja-only pilot attempts failed before a complete profile result; their distinct source/specification, error and cleanup records are retained separately and their partial timing fragments are excluded.

The full-sum CLI comparison includes both SM $2 arrow 4$ processes and loop-induced H, HH, HHH and HHHH. The separate normal-rescue six-process comparison uses original Ninja revision #raw("27147b3300f5") and final NinjaRS runtime #raw("a7577a9a3329") / archive #raw("b0d69deb8c16"). It passes the agreed 1.05 upper-interval reference in that amended ten-point, seven-pair scope. No canceled heavy eight-process timing requirement is reinstated, and no uniform SM speedup is inferred from the mixed, uncertain CLI estimates.

#v(7pt)
#panel[
 *Offline PDF reproduction:* from the repository root, run #raw("python -B docs/benchmarkfolder/final-report-v1/reproduce.py --output reproduced.pdf"). It verifies the document's finite data/vector/font/style closure and compiles with Typst 0.15.1, bundled Libertinus fonts, a fixed #raw("SOURCE_DATE_EPOCH") and creation timestamp. No native archive, compiler, network or amplitude evaluation is needed.
]

#v(8pt)
The source and raw custody verification is a separate operation requiring the exact retained installation and the concrete authority readers. Regenerating from pinned sources with a different compiler can produce new executable and measurement identities. The reproduction README supplies both workflows and the exact CLI cards, validated materialization, scalar coefficient records, timing metadata and plotting source. The original partial report is preserved under #raw("historical/20261021-partial").

#v(8pt)
Styling is attributed to #link("https://github.com/alphal00p/ninjars/blob/ca1009e89ecb03d3ff477f57b170241f7a9f8669/docs/benchmark/benchmark.typ")[NinjaRS ca1009e] and the earlier #link("https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ")[pyamplicol report]. Tail and timing comparisons are inspired by #link("https://arxiv.org/pdf/1604.01363")[Hirschi–Peraro, arXiv:1604.01363]. The external optional plotting archive was inspected for methodology; no external plotting code is included. Libertinus 7.051 is distributed unmodified under the bundled SIL Open Font License.
