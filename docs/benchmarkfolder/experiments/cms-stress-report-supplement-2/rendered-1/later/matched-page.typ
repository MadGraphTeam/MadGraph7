#import "style.typ": *
#let d = json("matched-data.json")
#show: body=>report-style(body,date:"11 November 2026",revision:"later matched-input QP diagnostic")
#set heading(numbering:none)
#let coefficients=("Born":"Born","finite":"Finite","single_pole":"Single pole","double_pole":"Double pole")
#let sci(s,digits:5) = {
  let value=float(s)
  if value==0 { return [0] }
  let exponent=calc.floor(calc.log(calc.abs(value),base:10))
  [#fixed(value/calc.pow(10,exponent),digits) × 10#super(str(exponent))]
}

= Later matched-input QP comparison

#note[*Later evidence; the preceding three pages retain their earlier chronology.*
Two eligible private fixed-QP workers now agree at the same CMS point 20 in
Born, finite, single- and double-pole outputs. The maximum retained
`abs(a-b)/max(abs(a),abs(b))` is $2.014 times 10^(-16)$.
This is matched-input agreement, not an analytic finite oracle or an
algorithmic winner. The private input-policy change is not promoted.]

The closed comparison uses backend 8 (NinjaRS) and backend 9 (TIRust), one
point-1 bootstrap and one point-20 measurement each: four outer calls and
104 native adapter calls. All native statuses are zero, both whole-worker
eligibility gates and bootstrap controls pass, and post-call software DP/QP
failure flags are clear. Fixed return 120 remains an unchecked evaluation.

== What was matched and what changed

For all 52 paired native requests, the ordered shifts, full scalar-invariant
matrix, complex squared masses, scale and canonical numerator words match
exactly. Explicit routing/producer metadata and entry/use environment tokens
also match. The ordered group-3 entries $(2,4)$ and $(4,2)$ actually use the
certified direct spacelike invariant. QP interface storage is 128 bits with
113 precision digits; this does not assert every reducer's internal precision.

*Backend 8 here uses a separate private input-policy derivative.* It shares
the candidate's unchanged certified invariant helper, retains a provably
negative raw invariant where a qualified composite pair would otherwise be
overwritten by the legacy zero branch, aligns the authoritative-upper symmetry
policy, and evaluates the scale as `REAL(MU_R,KIND=KI_QNRS)**2`. Original legacy
8 retains its original helper and `REAL(MU_R**2,KIND=KI_QNRS)` route. No library,
scalar cutoff, physical card or production backend is changed by this test.
The earlier failed original-8/Ninja-6 reference probe remains failed.

#figure(table-style(table(columns:(1.05fr,1.25fr,1.25fr,.60fr,1.05fr),
  inset:(x:4pt,y:6pt),stroke:table-stroke,fill:table-fill,
  table.header([Coefficient], [Private NinjaRS 8], [Private TIRust 9], [ULP gap], [Relative residual]),
  ..d.table.map(r=>([#coefficients.at(r.coefficient)],
     sci(r.backend8),sci(r.backend9),[#r.binary64_word_ULP_gap],sci(r.relative_error,digits:3))).flatten()
)),caption:[Point-20 retained binary64 outer-output words from fixed-QP evaluations. Both answer orders agree; exact duplicate orders are counted once.])

Born and double-pole words are bit-identical. Finite and single-pole outputs
differ by one ULP of the retained binary64 word. All coefficients and both
orders satisfy the existing regression criterion and the $10^(-8)$ target.
Native success alone was not used to admit coefficient arithmetic.

== Connection to the earlier pole audits and remaining scope

The current backend-9 output words match the prior eligible private candidate
exactly: sixteen Born/Laurent words across bootstrap/measurement and two orders,
plus four ancillary diagnostic words. Source/runtime-provider bindings are
retained. Historical native input words were not captured, so this output
parity does not retroactively measure equality of the earlier native inputs.
The preceding conditional pole audits concern those same observed outputs;
their coupling-reconstruction, restored-leaf and UV-cancellation limitations
remain in force. No failed reference coefficient is selected or decoded.

#note(fill:warm)[*Still unqualified.* Two-reducer agreement does not establish
independent full-amplitude truth, a general source/exporter fix, normal checked
MadLoop stability/rescue, generic process/topology coverage or performance.
The earlier incomplete stress stage is not reclassified. Precise local timing
remains Hua-Sheng's TODO with the agreed per-process upper-95% ratio bound ≤1.05.]

#text(8pt)[Closed run `52faa470d8be…`; comparison `af4b58a379ac…`; root review `9ca6601ab14a…`;
prior/current word parity `5f9f4cc6c8d2…`. Full pins and the public evidence join
are retained in the companion data/checker.
#link("https://github.com/MadGraphTeam/MadGraph7/blob/2b4277c39e20e1f7011efad515e7af3a6c99f64a/docs/benchmarkfolder/experiments/matched-native-input-qualification-1/README.md")[Closed public evidence].]
