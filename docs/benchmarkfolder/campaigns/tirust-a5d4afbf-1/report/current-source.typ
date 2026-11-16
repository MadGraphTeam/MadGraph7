#import "/style.typ": *
#let d = json("report-data.json")
#let v = d.display
#let rev = d.identity.exporter_revision.slice(0, 10)
#let pin = d.identity.tirust_revision.slice(0, 8)
#show: body => report-style(body, date: "2026-11-16", revision: rev + " · TIRust " + pin)
#set heading(numbering: none)
// These pages follow the 48 preceding report pages in the merged PDF.
#let base-pages = 48
#set page(numbering: (..n) => none, footer: context align(center, text(9pt,
  [#(counter(page).get().first() + base-pages) / #(counter(page).final().first() + base-pages)])))
#set text(size: 9.5pt)
#set par(leading: .5em)
#let grid-table(columns, header, rows, caption) = figure(table-style(table(columns: columns,
  inset: (x: 3pt, y: 4pt), stroke: table-stroke, fill: table-fill,
  table.header(..header), ..rows.flatten().map(x => [#x]))), caption: caption)

= Current-source requalification: TIRust pin #raw(pin)

#note(fill: warm)[#set par(justify: false)
*Final pin on TIRust `main`.* These pages qualify MadGraph
#raw(rev) with TIRust #raw(d.identity.tirust_revision) on TIRust `main`, after PR #35
was merged as `a2cfa4b` (Symbolica community `0e2ad04c`, oneloopmaster `2e7afabb`). The
source includes the merged `madloop_ninjars` (return-code format 2: TIRust returns 213
in double and 323 in quadruple precision) and the bridge's QP check-pass cache. They
replace the rehearsal pages on `9a781998` and supersede the earlier pages only for this
source and runtime (DSO #raw(d.identity.native_files.at("lib/libmadloop_tirust.so").slice(0, 12))).
Shared-host results; precise paired timing acceptance remains Hua-Sheng's local TODO.
The stress page was corrected on 2026-11-16: TIRust accepts one inaccurate value there.]

The fresh `install tirust` build of this pin passes `smoke --require-poles`, the pinned
upstream C-ABI fixtures (27 DP and 54 QP lanes) and links only `libgcc_s`, `libm`
and `libc`. Exports were regenerated from the campaign's frozen cards, models and
`-O3` flags; the baseline is the starting revision 9228fc58.

== Ordinary matrix: 11 processes × 6 lanes × 100 points

#grid-table((1.5fr, .55fr, 1.15fr, .8fr, .75fr, .75fr),
  ([Process], [3e-6], [1e-8 misses\ CT/Ninja/COLLIER/NinjaRS/TIRust], [TIRust max\ finite residual],
   [TIRust\ DP/QP], [previous\ DP/QP]),
  v.numerical_rows,
  [Born, finite and both poles against starting-revision CutTools (`LoopMEComparator`).
   #v.numerical_totals.passed of #v.numerical_totals.lanes lanes pass `3e-6`;
   #v.numerical_totals.target meet the separate `1e-8` target. Misses count scalar
   comparisons (split-order rows duplicate). "Previous" is source 0fcd2cd873 / runtime
   854b79dc; the earlier EW 2→4 TIRust lane (40 DP / 60 QP) was recorded separately.])

TIRust now accepts double precision where the previous runtime needed a genuine QP
rescue: its internal double-double and MPFR tensor tiers recover refused lanes
before MadLoop rebuilds a QP numerator. In QCD $u overline(u) -> Z g g$, Ninja,
COLLIER, NinjaRS and TIRust miss the `1e-8` target on the same 28 points with the
same residual, which points at the CutTools reference; the comparison itself assigns
no independent truth. Candidate CutTools now differs from starting-revision CutTools
by up to 6.5e-10 (rehearsal: 0); the candidate source now includes the merged
`madloop_ninjars` and MadGraph `main`.

== Double-double gate (headroom 26)

#grid-table((1.5fr, .7fr, .6fr, .8fr, .9fr, .9fr, .9fr),
  ([Process], [DP calls], [DD calls], [new-window\ calls], [max error\ new window], [max error\ all DD], [verdict]),
  v.at("dd_rows", default: ()),
  [Every bridge call of the TIRust lane was captured (`tools/tirust_capture`) and replayed:
   the production build reproduces all recorded bits; references with headroom 40
   (the previous 2^−66 gate) and 105 (no lane in DD) evaluate the same lanes at 256
   bits. Error = max |difference| / max |value| of the same tensor degree. Flag 1e-12,
   fail 1e-9. #v.at("dd_totals", default: (dd: "–")).dd DD calls in total. DP calls include
   MadLoop's helicity-filter learning, which evaluates each helicity separately on the first
   three points (with `LoopPoleMode = 0` since `madloop_ninjars`); the bridge's exact-input
   cache answers all but the first evaluation of each tensor there.])

#pagebreak()
== Per-point arithmetic provenance

#grid-table((1.5fr, .7fr, .5fr, .6fr, .55fr, .5fr, .75fr, .75fr, .75fr),
  ([Process], [binary64\ only], [DD\ tier], [MPFR\ tier], [genuine\ QP], [failed], [status =\ campaign], [cache off\ identical], [1 MiB budget\ identical]),
  v.at("provenance_rows", default: ()),
  [Each of the 100 ordinary points per process evaluated once by a private clone with
   bridge route counters around every amplitude call (`tools/tirust_provenance.py`).
   MadLoop clears the TIRust and Fortran coefficient caches per point, so the deltas
   cover every reduction a point reused. Cache-off and a 1 MiB budget (insertion
   refusals) reproduce every coefficient, accuracy and status bit for bit.])

== Physical stress points

#grid-table((1.25fr, .8fr, 1fr, 1.25fr, 1.3fr, .6fr),
  ([Process], [point], [reference], [TIRust alone], [chain 9|6|7|1], [TIRust QP]),
  v.at("stress_rows", default: ()),
  [Threshold, soft/collinear, small-Gram and complex-mass points (`tools/tirust_stress.py`).
   Reference: forced-QP CutTools and Ninja when they agree, otherwise at least two agreeing
   stability-checked values of independent reducers; "unverified" marks a point with only one.
   Every value must also have the case's IR double pole, whose ratio to the Born is kinematics
   independent and fixed by the ordinary points. TIRust may refuse. Blocking outcomes are an
   accepted value that is WRONG (disagrees with the reference at `3e-6`, or has the wrong double
   pole) or INACCURATE (its actual error in MadLoop's accuracy measure exceeds
   `MLStabThres = 1e-6` while its claimed accuracy does not): #v.at("stress_blocking", default: "–")
   lane outcomes. "Process STOP": CutTools ends the program (CT692) on the ZZgg soft-collinear
   point; on the most forward CMS small-Gram point it never returns. The driver records the
   point and resumes with the next one. "TIRust QP" compares unchecked forced QP
   (`CTModeRun=4`) with the reference. At the $t overline(t) gamma$ threshold the checked
   reducers disagree (Ninja's single pole is 112× too large). TIRust's accepted QP value there
   agrees with NinjaRS QP to 1.1e-9 and with CutTools QP to 3.7e-6.])

#note(fill: warm)[#set par(justify: false)
*Corrected 2026-11-16.* The first version of this page reported no blocking outcome. It
compared values only at the `3e-6` campaign tolerance and let CutTools serve as its own
reference.

*TIRust, open before merge.* At the complex-mass forward point $theta = 10^(-3)$, TIRust
returns 213 with a claimed accuracy of 3.1e-15, but it is 2.0e-6 from the QP reference
(4.1e-6 in MadLoop's measure). CutTools DP (2.6e-8, claimed 9.5e-8) and COLLIER (6.9e-10)
are accurate there. TIRust's method-choice expansions stop near 1e-9 per tensor, and
cancellations amplify this. The error is the same in both loop directions, so MadLoop's
direction check cannot see it. A forward-angle scan by the TIRust session at default
settings, outside this campaign's points, finds up to 2.8e-5 at $theta = 5 dot 10^(-4)$;
with method choice disabled the same points agree with QP within 1e-9. In the ordinary
matrix every TIRust value is within 2.1e-7 of COLLIER or NinjaRS in MadLoop's measure,
below `MLStabThres` at every point.

*Outside TIRust.* At CMS $theta = 10^(-6)$ and the EW $mu^+ mu^-$ small-Gram point,
MadLoop's `OSThres` projection sets a small invariant exactly to zero, which changes the
integral. TIRust refuses (correctly). CutTools QP (329), in the baseline as well, returns
exactly half of the universal double pole, and the 9|6|7|1 chain returns that value. With
`OSThres = 1e-14`, TIRust and Ninja QP agree to 10 digits at $theta = 10^(-6)$. CutTools DP
also accepts a finite part of $2.4 dot 10^18$ (true $4.0 dot 10^(-5)$) at the loop-induced
$H H$ small-Gram point, and COLLIER, Ninja and NinjaRS accept a wrong single pole at the
$t overline(t)$ threshold. TIRust is right at both points.]

#pagebreak()
== Stability (Appendix C of arXiv:1604.01363)

#figure(image("../stability/stability-raw.png", width: 100%),
  caption: [Raw double precision, no automatic rescue: MadLoop's internal estimate of
  the total answer on 100 frozen cut-qualified points per process. Shaded bands are
  unknown tails from unavailable points; a TIRust point is unavailable when its
  reduction refuses. TIRust's DP includes its internal DD/MPFR tensor tiers.])

#pagebreak()
#figure(image("../stability/stability-rescued.png", width: 100%),
  caption: [After MadLoop's automatic rescue at the campaign threshold `MLStabThres = 1e-6`.])

#grid-table((.6fr, 1.4fr, .8fr, .7fr, .75fr, .75fr, .75fr),
  ([Variant], [Process], [Reducer], [available], [> 1e-3], [> 1e-6], [> 1e-9]),
  v.at("stability_rows", default: ()),
  [Fractions of the 100 points whose estimate exceeds the threshold; brackets give
   [known, known + unavailable] bounds. At 100 points the resolution is 0.01; larger
   samples remain a TODO. The estimate is MadLoop's self-consistency test, not an
   independent error.])

#if "profile_rows" in v [
== Provisional stock O3 profile (not acceptance)

#grid-table((1fr, .9fr, 1.6fr, .8fr, .9fr),
  ([Reducer], [FULL ms/point\ (median)], [three repetitions], [initialization s], [ratio / NinjaRS]),
  v.at("profile_rows", default: ()),
  [Stock `check profile` of QCD $u overline(u) -> Z g g g$ on the existing O3 export (learned filters,
   48/96 helicities, both poles, timed `CTModeRun=1` without stability test), 50 points, one CPU,
   three repetitions in alternating reducer order, TIRust DSO #raw(v.at("profile_dso", default: "–")).
   TIRust/NinjaRS #v.at("profile_ratio", default: "–") (the 48-page report measured 1.558 with
   the previous runtime). Unpaired shared-host data: no confidence bound, no acceptance claim.])
] else [
== Provisional stock O3 profile
Not rerun on this pin: the shared host was overloaded (load average above 10 000 on 384
cores) when the campaign finished. The rehearsal profile on DSO `6c11cfd9` measured
TIRust/NinjaRS 1.177 on QCD $u overline(u) -> Z g g g$; precise timing is Hua-Sheng's
local TODO.
]

#text(8pt)[#set par(justify: false)
Campaign `tirust-a5d4afbf-1` (manifest #raw(d.manifest_sha256.slice(0, 12))), exporter
#raw(d.identity.exporter_files_sha256.slice(0, 12)). Commands, summaries and the
local timing handoff are in `docs/benchmarkfolder/campaigns/tirust-a5d4afbf-1/`.
The preceding 48 pages are unchanged.]
