#import "style.typ": *
#let d = json("data.json")
#show: body => report-style(body, date: "11 November 2026", revision: "private CMS diagnostic")
// Render separately after source review. Append to the frozen 35-page PDF;
// never rerender or replace those pages to incorporate this supplement.
#set heading(numbering: none)
#let sci(s, digits: 5) = {
  let value = float(s)
  if value == 0 { return [0] }
  let exponent = calc.floor(calc.log(calc.abs(value), base: 10))
  [#fixed(value / calc.pow(10, exponent), digits) × 10#super(str(exponent))]
}
#let pole-table(pole) = figure(
  table-style(table(columns: (.60fr,1.7fr,1.25fr,1.25fr,1fr),
    inset: (x:4pt,y:6pt), stroke:table-stroke, fill:table-fill,
    table.header([Point], [Eligible fixed-QP worker], [Predicted pole], [Observed pole], [Relative residual]),
    ..d.pole_comparisons.filter(r=>r.pole==pole).sorted(key:r=>r.point*10 + if r.worker=="cms-p20-mode4-lib1-original" {0} else {1}).map(r=>(
      [#r.point#if r.phase==0 { [†] }], [#r.label], sci(r.expected), sci(r.observed), sci(r.relative_residual,digits:3)
    )).flatten())),
  caption: if pole == -2 { [Universal QED double pole.] } else { [Universal QED single IR pole of the intended renormalized amplitude.] })

= CMS stress diagnostic: closed outcomes and limits

#note(fill:warm)[*Private prototype; no production promotion.*
The source-derived Fortran invariant certificate has conditional QP pole support
at the forward CMS fixture. Its complete stress comparison remains failed and
incomplete; the finite term remains unresolved. These observations do not
qualify normal MadLoop stability/rescue or precise performance.]

The fixture is $e^+ e^- -> mu^+ mu^-$ with QED virtual corrections and the existing
complex-mass Gmu model. The prototype changes isolated generated Fortran; the
original TIRust shared library is unchanged. Fixed mode 4 returns 120, an
unchecked return. The eligible workers have both post-call software failure
flags clear; this supplies no independent host accuracy guarantee.

#figure(table-style(table(columns:(1.1fr,.85fr,2.7fr),inset:(x:5pt,y:6pt),
  stroke:table-stroke,fill:table-fill,
  table.header([Closed stage], [Outcome], [Scope retained]),
  [First build attempt], [Failed], [Controller `KeyError('size')`; zero native commands, loader queries or amplitudes.],
  [Successor ordinary], [8 / 8 eligible], [Point 1, original/candidate × CutTools/TIRust × DP/QP. Ordinary target met; private fixed-mode evidence only.],
  [Full stress], [Incomplete; exit 1], [Eight point-20 workers complete. Ninth worker, original CutTools DP at point 21, times out after 300 s, is terminated by signal 15 and reaped. Seven workers never launch.],
  [Point-20 subset], [5 / 8 eligible], [Three failed TIRust workers' buffers excluded. Candidate TIRust QP is eligible but its comparison with CutTools QP fails; no complete stress result exists.],
  [Legacy reference probe], [0 / 2 eligible], [NinjaRS and Ninja complete their protocols; both fail their scientific eligibility gates. No coefficient comparison, including bootstrap, is interpreted.]
)),caption:[Execution completeness, eligibility and numerical agreement are separate outcomes.])

Original/candidate CutTools controls agree exactly. Original TIRust DP/QP and
candidate TIRust DP retain software failures and W03 warnings at point 20.
Candidate TIRust QP has clear post-call flags but differs from CutTools, with
maximum recorded `abs(a-b)/max(abs(a),abs(b))` of 1.237729 across the compared
coefficients. The later universal-pole audits below
support its poles under their reconstruction assumptions; they do not turn that
failed full-coefficient comparison or the incomplete stage into a pass.

#figure(table-style(table(columns:(.75fr,.85fr,2.9fr),inset:(x:5pt,y:6pt),
  stroke:table-stroke,fill:table-fill,
  table.header([Legacy QP probe], [Eligible], [Why no coefficient arithmetic is allowed]),
  [NinjaRS (8)], [No], [Five point-20 box groups return status 1: 3, 4, 8, 12, 24. The software QP failure flag is set.],
  [Ninja (6)], [No], [Adapter statuses are zero but final finite and both pole words are nonfinite. The host software DP failure flag is set despite zero observed DP adapter calls.]
)),caption:[Both probes use the legacy kinematic helper; matched native input geometry with the prototype is unestablished.])

Their 104 completed adapter calls establish the QP interface route
(storage width 128, precision digits 113), not every reducer's internal working
precision. No scalar cutoff was changed and no particular scalar-call cause is proven.

#pagebreak()
= Independent universal-pole comparisons

For four massless unit-charge external fermions, the double-pole prediction is
$A_(-2) = -4 B alpha/(2 pi)$.
The source-derived single-IR-pole prediction is
$A_(-1) = B alpha/(2 pi) [-6 - 2 sum_(i<j) Q_i Q_j ln(abs(2 k_i dot k_j)/mu^2)]$,
with all-outgoing charges $(-1,+1,+1,-1)$ and $mu^2=10^6$.
The retained FKS convention cancels the auxiliary QES scale algebraically;
these checks introduce no scale-varied amplitude evaluations.

#pole-table(-2)
#pole-table(-1)
#text(8pt)[† Point 1 rows are the selected point-20 workers' bootstrap, not additional independent physical samples.
Two exactly duplicated answer orders are collapsed once. Current candidate
CutTools controls match the displayed original CutTools rows exactly.]

#note[At point 20, private candidate TIRust QP agrees with the reconstructed
double and single IR poles to about $2.68 times 10^(-16)$ and
$5.23 times 10^(-16)$ respectively. Original CutTools QP differs by about
50.0% and 53.0%. This is conditional coefficient consistency; the finite
coefficient and full-amplitude accuracy remain unresolved.]

// A future reviewed render may include the plot below. This proposal contains
// rendering source and exact data but deliberately no rendered artifact.
#if sys.inputs.at("include-pole-plot",default:"false")=="true" {
  figure(image("pole-comparison.png",width:100%),caption:[Observed/predicted poles and relative residuals for the eight eligible observations. Failed legacy probes are excluded, not assigned zero.])
}

== Assumptions that limit the conclusion

The coupling is reconstructed as
$alpha = sqrt(2) G_F abs(mu_W^2 (1-mu_W^2/mu_Z^2))/pi$,
$mu_V^2=M_V^2-i M_V Gamma_V$, with the existing `GFRedefinitionChoice=1` and
the retained model's decimal pi literal. Parsing card values through binary64
is an explicit assumption: retained packets do not
contain the card-reader/MP-widening implementation or observed runtime coupling
bits. The original alpha and pole calculation uses 70-digit Decimal working
arithmetic; independent pole-factor and scale-cancellation reviews use 90
digits. Neither measures native coupling precision or emulates every Fortran
rounding step.

The single-pole calculation uses earlier closed group-3 QP trace leaves for
points 1/20 and exact rational dot products of those words. Their original
point-20 stress run remains failed. Matching cards, point streams and an unchanged
momentum-source path justify the documented reuse; the current prototype made no
new external-word capture. Restored QP leaf components must not be equated with
the un-restored binary64 input stream. The prediction applies to the intended
UV-renormalized full `ANS`, with counterterms and UV+IR selection. Inclusion of
counterterms is source-linked; UV cancellation and every model counterterm were
not separately tested. This is not a bare tensor `IR1` test.

== Reproduction and remaining acceptance

The pure generator authenticates the four closed public indexes and every
indexed payload, joins the selected eligible pole words to the closed parent,
and preserves failed reference buffers without decoding or arithmetic.
It does not re-execute the packets' broader dependency verifiers, rehash today's
external binaries, or reparse all historical generated sources.

#text(8pt)[Public packet seals: double pole `1565ac55fe6a…`; single pole `99603d41d662…`;
private amplitude `c04a73c86f6a…`; legacy references `248bd61e987a…`.
Complete seals and selected leaf hashes are in `data.json`.]

#note(fill:warm)[*Local precise timing TODO — Hua-Sheng.*
Use identical local builds, points, models and cards; run at least seven
alternating paired batches of about one second. Require the upper 95% runtime
ratio bound to be ≤1.05 for every main process, with existing-backend regressions
and native-DP/rescued cohorts reported separately. Current stage elapsed times
are accountability records, not measurements satisfying that requirement.]
