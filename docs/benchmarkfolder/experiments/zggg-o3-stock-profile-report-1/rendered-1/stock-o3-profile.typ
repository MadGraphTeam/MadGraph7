#import "style.typ": *
#let d=json("stock-o3-data.json")
#assert(d.schema=="madloop-stock-o3-zggg-report-data-v1" and d.closed)
#assert(d.source_mg_revision=="cda704e28d634e893fabc3ac358d9a27789f93aa")
#assert(d.all_object_o3_proven)
#show: body=>report-style(body,date:d.date,revision:"cda704e2 · stock O3")
#set heading(numbering:none)
#set text(size:9.5pt)
#set par(leading:.5em)
#let value(x,digits:6)=if x==none {[—]} else {fixed(x,digits)}

= Stock O3 profile: QCD $u overline(u) -> Z g g g$

#note[*A separate stock-profile measurement.* Actual `loop_sm` exports use
O3 for all generated MODEL/DHELAS/process/driver objects, normal learned filters and DP-interface
timing without the independent stability test. Earlier checked, fixed-point O0
results keep their own input/filter policy; ratios across these pages are not
compiler speedups.]

== Five explicit reduction backends

#figure(table-style(table(columns:(.95fr,.9fr,.9fr,1.2fr,.9fr,.9fr,.9fr),
 inset:(x:3pt,y:6pt),stroke:table-stroke,fill:table-fill,
 table.header([Reducer], [FULL ms], [SKIP ms], [FULL−SKIP\ estimate ms], [FULL points], [FULL wall s], [FULL ratio\ / NinjaRS 8]),
 ..d.rows.map(r=>([#r.name],value(r.full_ms),value(r.skip_ms),
 value(r.full_minus_skip_estimate_ms),[#(if r.full_points==none {"—"} else {str(r.full_points)})],
 value(r.full_wall_seconds,digits:3),table.cell(fill:ratio-fill(r.ratio_to_8),
 if r.ratio_to_8==none {[—]} else {[×#fixed(r.ratio_to_8,3)]}))).flatten()
)),caption:[Actual stock `check profile` per-point estimates. Each row is a separate run with its own batch size; FULL−SKIP estimates reduction cost, not a directly measured kernel time. Unavailable/failed results remain explicit.])

#d.result_summary

== Measured configuration and clock

`CTModeRun=1`, `LoopPoleMode=-1`, `HelicityFilterLevel=2`; normal filter learning
is retained. COLLIER’s UV and IR controls are both true. These effective cards,
selected helicities/filter state, model and all-generated-object O3 compilation records
are retained with the raw logs. Installed reducer release builds are unchanged.
Timed CTMode1 omits MadLoop direction/rotation checks and whole-amplitude QP
rescue. #d.configuration_scope

#d.clock_scope

#figure(table-style(table(columns:(1fr,1fr,1.4fr,2.1fr),
 inset:(x:4pt,y:4pt),stroke:table-stroke,fill:table-fill,
 table.header([Reducer], [Initialization s], [Status], [Observed precision scope]),
 ..d.rows.map(r=>([#r.name],value(r.initialization_seconds,digits:5),[#r.status],[#r.precision_scope])).flatten()
)),caption:[Initialization is separate from warmed stage estimates. A null initialization denotes unavailable measurement. Accepted DP transport does not establish native-DP arithmetic for TIRust.])

== Accuracy, scope and remaining acceptance

#d.numerical_scope

TIRust’s DP-input path may use local 106-bit DD or 256/512-bit tensor arithmetic
with DP inputs/numerators. Those tiers differ from genuine 113-bit Fortran-QP
input transport. Stock timings alone do not assign per-point tensor routes;
missing counts must not be read as zero. #d.precision_evidence_scope

#note(fill:warm)[These are unpaired shared-host stock runs, with no paired
confidence interval or main-process acceptance claim. All 66 fresh local numerical
lanes and eight main runtime gates remain pending. The upper 95% paired
candidate/baseline runtime bound ≤1.05 for every main process is not established.
The integration goal remains incomplete.]

#text(8pt)[#set par(justify:false)
Source #raw(d.source_mg_revision); TIRust DSO #raw(d.tirust_dso_sha256).
Exact FULL/SKIP/boot counts, clocks, filters, compiler recipes, failures and
initialization records accompany `stock-o3-timings.csv` and the pure reproduction
commands. The preceding 47 pages are preserved transitively, without replaying
or relabeling their physics. Style: pyAmpliCol 7266c1600f6e, preserved typography
and green–white–red timing palette.]
