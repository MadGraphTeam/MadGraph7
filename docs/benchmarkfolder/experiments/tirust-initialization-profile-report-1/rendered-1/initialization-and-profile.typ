#import "style.typ": *
#let d=json("initialization-profile-data.json")
#assert(d.closed and d.production_artifact_qualified and not d.preview_only)
#show: body=>report-style(body,date:d.date,revision:"77eb57b8 / d427")
#set heading(numbering:none)
#set text(size:9.2pt)
#set par(leading:.5em)
#let a=d.candidate
#let p=d.pilot
#let s=d.stock_current
#let c=d.controls
#let sig(x,digits:5)={
  if x==none{return [—]}
  if x==0{return [0]}
  let e=calc.floor(calc.log(calc.abs(x),base:10))
  if e < -2 or e > 4 {[#fixed(x/calc.pow(10,e),digits - 1) × 10#super(str(e))]}
  else {fixed(x,calc.max(0,digits - 1 - e))}
}

= Final source: five reducers and stock CLI

#note(fill:warm)[*Initialization improves; warmed performance remains slower.*
At the diagnosed Zggg point TIRust/NinjaRS has matched-cycle median ratio
*×#fixed(p.rows.last().median_paired_ratio_to_8,3)*. The separate stock CLI gives
×#fixed(s.full_ratio_tirust_over_ninjars,3). Neither closes local runtime acceptance.]

== QCD $u overline(u) -> Z g g g$: one diagnosed physical point

Actual `loop_sm` CLI export; final source `77eb57b8`, library `d427`.
This follow-up selects *point 1* of the original frozen 100-point set, after its
initialization diagnosis. It is separate from the original ten-point timeout.
Seven balanced cycles give 35 warmed full-amplitude batches, each one evaluation;
actual batches span #fixed(p.batch_min_seconds,3)–#fixed(p.batch_max_seconds,3) s.

#figure(table-style(table(columns:(1fr,1.4fr,1.05fr,.85fr,1.2fr),
 inset:(x:4pt,y:5pt),stroke:table-stroke,fill:table-fill,
 table.header([Reducer], [Median ± MAD\ ms / point], [Paired ratio\ / NinjaRS 8], [READY s], [3e−6 / 1e−8]),
 ..p.rows.map(r=>([#r.name],[#fixed(r.median_ms,3) ± #fixed(r.mad_ms,3)],
 table.cell(fill:ratio-fill(r.median_paired_ratio_to_8),[×#fixed(r.median_paired_ratio_to_8,3)]),
 [#fixed(r.ready_seconds,3)],[PASS / #text(fill:if r.ordinary_target_pass {rgb("#0f7356")} else {rgb("#b2182b")})[ #(if r.ordinary_target_pass {"PASS"} else {"MISS"})]])).flatten()
)),caption:[Unscaled MAD of seven times; ratios are medians of matched within-cycle ratios, not ratios of medians. READY separately measures bootstrap plus one same-point warmup. No confidence interval is claimed.])

All 45 retained comparisons pass `3e-6`. The separate `1e-8` target misses the
finite coefficient for 6/7/8/9 versus CutTools by ≈3.534 × 10#super[−8]; CutTools
is its own reference. Old genuine-QP and the other reducers corroborate the
cluster, without assigning automatic fault or absolute truth. No point is dropped.

Every reducer has seven accepted DP-transport dispatches. TIRust has *zero genuine
Fortran-QP* dispatches but observed local tensor tiers. Timed primary routes are
7448 direct, 392 method-choice, 238 local106 and 42 local256; independent checks
are 7518/350/182/70. Attempts106/256 are primary 252/42, check 182/70. Local512,
refusal and unreported counts are zero. These counters exclude Fortran
coefficient-cache hits. Local tiers lift 53-bit inputs with DP numerators;
*219 is not a native-DP-only label*. Getter queries and writes are outside timing.

GNU Fortran 15.2.0; original model/math objects retained, usual generated sources
without an optimization flag and polynomial source at O3 (exact recipes retained).
Single CPU 12 affinity on a shared, unreserved Linux host; sampled one-minute
load #fixed(p.host_load1_min,1)–#fixed(p.host_load1_max,1). This is a selected-point
pilot, not a process average, full 100-point campaign or baseline-chain comparison.

== Corrected stock `check profile`: QCD $u overline(u) -> Z$

#figure(table-style(table(columns:(.9fr,.9fr,.9fr,1.3fr,.8fr,.9fr),
 inset:(x:3pt,y:4pt),stroke:table-stroke,fill:table-fill,
 table.header([Reducer], [FULL ms], [SKIP ms], [FULL−SKIP\ estimate ms], [Points], [FULL wall s]),
 ..s.rows.map(r=>([#r.name],[#fixed(r.full_ms,6)],[#fixed(r.skip_ms,6)],
 [#fixed(r.full_minus_skip_estimate_ms,6)],[#r.full_points],[#fixed(r.full_batch_wall_seconds,3)])).flatten()
)),caption:[One stock CLI run per backend; unequal short batches, unpaired and descriptive. FULL−SKIP estimates reduction cost; it is not a measured kernel time.])

All configured helicities, full finite+UV+IR, pole mode 0 and CTModeRun 1.
FULL now uses monotonic child-completion timing independent of RSS polling;
SKIP/boot retain Python wall clocks. Stock values subtract boot time and divide
by actual stage counts. Only the retained all-helicity end result establishes
nonzero Born/finite; per-timed-point statuses and tensor routes are unavailable.
Missing route counts are *not zero*. The configured polarized result is zero.
Separate ten-point within-backend stability probes have 0 QP/unstable/exceptional;
maximum spreads are #sig(s.rows.first().maximum_within_backend_spread) /
#sig(s.rows.last().maximum_within_backend_spread), not independent error.
NinjaRS fresh initialization is #fixed(s.rows.first().initialization_seconds,5) s;
TIRust reused its export, so initialization is *unavailable (null)*.

#pagebreak()
= Initialization repair and retained precision limits

#note[*Final artifact and prototype timings have distinct identities.*
Final d427 passes 16 native tests, genuine 113→256 QP/poles/route smoke and
67 root integration tests. Route-aware tools add 50 tests and a 52-call tensor
toy parity check, not another SM/performance qualification. Backend 9 stays explicit.]

== Quiet initialization: original versus adaptive prototype

The same original O0 Fortran executable, frozen point, cards, stability policy
and CPU 12 affinity are used in one ordered old→prototype pair. No shim, preload
or counter query; other owned captures were held. Shared CPUs were not reserved.

#figure(table-style(table(columns:(1.4fr,1fr,.9fr,1.9fr),
 inset:(x:4pt,y:5pt),stroke:table-stroke,fill:table-fill,
 table.header([Library], [READY s], [Ratio / old], [Outer result]),
 [Original signed S /85ee],[#fixed(a.clean.old_ready_seconds,6)],[×1],[323 · genuine Fortran QP],
 [Adaptive prototype /470],[#fixed(a.clean.candidate_ready_seconds,6)],
 table.cell(fill:ratio-fill(a.clean.candidate_over_old_ratio),[×#fixed(a.clean.candidate_over_old_ratio,5)]),
 [219 · DP inputs, local tensor tiers]
)),caption:[Existing Fortran SYSTEM_CLOCK READY: bootstrap plus one same-point warmup, excluding launch and SETPARA. One descriptive pair, not warmed milliseconds/point or a confidence bound.])

The observed initialization ratio is *#fixed(a.clean.old_over_candidate_ratio,3)×*
old/prototype. Final d427 has its own 14.119 s READY above; the prototype timing
is not relabeled. The original ten-point attempt’s 600 s timeout remains on page 45.

== Separate functional observer and tensor reference

The prototype observer overlapped background validation; its elapsed time is
excluded. Of 4640 full-DP bridge requests (2320 primary / 2320 check), final routes
are 4276 direct, 212 method-choice, 120 local106/DD, 32 local256; no 512/refusal/
unreported or genuine-QP entry. Attempts106/256 are primary 72/12, check 52/20;
four 106 attempts continue to 256. Context-lifetime counters survive cache clears
and omit pre-validation errors and Fortran coefficient-cache hits.

All comparisons with old true QP and reducers 1/6/7/8 pass `3e-6`; old QP and 6/7/8
also pass `1e-8`. Maximum candidate/old-QP residual is
#sig(a.functional.comparisons.at("genuine113bit_input_QP_baseline").maximum_raw_relative_error).
The CutTools finite target miss 3.534 × 10#super[−8] was already 3.532 × 10#super[−8]
against old QP. Genuine Fortran QP retains 113-bit input /256-bit work; local tiers
use 53-bit inputs and a DP numerator. Numerical-body identity and final native
qualification are recorded separately from prototype observations.

Eight exact-bit tensor fixtures compare against a *same-engine Float512 reference*
on 53-bit inputs. All 176 degree×Laurent vectors pass `1e-8` (maximum
#sig(float(a.reference.maximum_slot_residual))); actual C-ABI fixture routes are
7 local106 + 1 local256. *50 individual component-relative misses remain* near UV
zeros: absolute residue≤#sig(float(a.reference.component_miss_max_absolute)),
reference norm≤#sig(float(a.reference.component_miss_max_reference_norm)).
All 3136 complex differences are retained; no norm floor. Vector agreement is
neither universal componentwise agreement nor an independent physics oracle.

== Instrumented attribution; O2 remains unpromoted

#figure(image("phase-bars.svg",width:100%),caption:[Original/O2 instrumented controls. Remainder is outside DP/QP+setup, not pure Fortran or clean kernel cost; it includes untraced/boundary work, scheduling and observer overhead.])

Original READY #fixed(c.baseline.ready_seconds,3) s contains setup
#fixed(c.baseline.setup_seconds,3), DP-entry #fixed(c.baseline.dp_seconds,3),
QP-entry #fixed(c.baseline.qp_seconds,3), remainder
#fixed(c.baseline.outside_dp_qp_setup_seconds,3) s; first DP→QP gap
#fixed(c.baseline.dp_to_qp_gaps_seconds.first(),3) s. Its 152 DP refusals lead to
whole-QP retries; the separate functional observer accepts 152 local tensor
rescues. This attribution is not a phase decomposition of the quiet pair.
Full-O2 READY is #fixed(c.o2.ready_seconds,3) s (×#fixed(c.o2_over_original_ready_ratio,3));
literal outputs and metadata/status fingerprints match, without full-input proof.
No convincing single-control benefit; O2 is unpromoted.

#note(fill:warm)[*Acceptance remains open.* All 66 fresh source-matched numerical
lanes and eight main runtime gates await Hua-Sheng’s local run. TIRust remains
slower in both measured warmed comparisons above. The upper 95% paired candidate/
baseline bound≤1.05 for every main process is not established. Stress failures,
strict-target misses, old coarse CLI controls and original timeout remain retained.]

#text(7.5pt)[#set par(justify:false)
Current source 77eb57b82d48 / final DSO d427667a0f2f; prototype 607e7f58842b /470d46df9c8f.
Instrumented MG cd014592; old stock cafb35d7 /85ee remains separately archived.
Signed TIRust 10ce34ed8ad5; exact hashes, nine CSVs, raw points/checks/routes and
pure reproduction accompany this appendix. Original 16 MB source archive is reused by hash.
Typography/palette: pyAmpliCol 7266c1600f6e, preserved style.]
