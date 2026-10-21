// Policy-v2 renderer: python tools/ninjars_report_v2.py DATASET --compile
#import "style.typ": *
#let data = json("report-data.json")
#let config = data.config
#show: body => report-style(body, date: data.generated_utc)

#let label(key) = (
  "qcd-1": [$u overline(u) -> Z$],
  "ew-1": [$u overline(d) -> W^+$],
  "qcd-2": [$g g -> t overline(t)$],
  "ew-2": [$e^+ e^- -> mu^+ mu^-$],
  "qcd-3": [$u overline(u) -> Z g g$],
  "ew-3": [$e^+ e^- -> t overline(t) gamma$],
  "qcd-4": [$u overline(u) -> Z Z g g$],
  "ew-4": [$e^+ e^- -> mu^+ mu^- gamma gamma$],
  "supplement-gg-h": [$g g -> H$],
  "supplement-gg-hh": [$g g -> H H$],
  "stress-qcd-4-gg-ttgg": [$g g -> t overline(t) g g$],
  "stress-ew-4-uu-4l": [$u overline(u) -> e^+ e^- mu^+ mu^-$],
).at(key, default: key)
#let status(s) = text(weight: "semibold", fill: (
  passed: rgb("#287240"), failed: rgb("#b44339"), unresolved: rgb("#966016"),
  pending: grey, partial: grey, measured: navy,
).at(s, default: grey), s)
#let short-hash(s) = if s == none { [Pending] } else { str(s).slice(0, calc.min(12, str(s).len())) }
#let recorded-revision(s) = if s == none { [Final installed provenance pending] } else { text(size: 7.4pt, s) }
#let scientific(x) = if x == none { [—] } else if x == 0 { [0] } else {
  let exponent = calc.floor(calc.log(calc.abs(x), base: 10))
  if exponent < -323 { str(x) } else {
    [#fmt(x / calc.pow(10, exponent), digits: 2) × 10#super(str(exponent))]
  }
}
#let prepare(c, source, stage) = {
  let value = c.preparation.at(source).at(stage)
  if value == none { none } else { value.seconds }
}
#let build-time(c, source) = {
  let value = c.preparation.at(source).compilation
  if value == none { fmt(none) } else {
    let kind = value.at("build_kind", default: "unknown")
    let marker = ("clean-subprocess": "C", incremental: "I").at(kind, default: "?")
    [#fmt(value.seconds, digits: 2) #text(size: 6.5pt, fill: grey)[#marker]]
  }
}
#let timing-card(c, backend) = {
  let t = c.timings.candidate.at(backend)
  let ref = c.timings.baseline.at("6")
  table.cell(inset: (x: 3pt, y: 8pt), align: top, {
    set par(justify: false, leading: 0.25em, spacing: 0pt)
    set block(spacing: 0pt)
    block(width: 100%, fill: ratio-fill(t.ratio), inset: (x: 5pt, y: 6pt), radius: 2pt)[
      #text(size: 7pt, fill: luma(70))[µs / point]
      #h(1fr)
      #text(font: "DejaVu Sans", size: 11.2pt, weight: "bold", value-fmt(t.median_us))
      #linebreak()
      #if t.ratio != none {
        text(size: 7.1pt)[*×#fmt(t.ratio, digits: 2)* vs original Ninja]
        linebreak()
        text(size: 7.1pt, fill: grey)[reference #value-fmt(ref.median_us) µs]
      } else {
        text(size: 7.1pt, fill: grey)[Timing pending]
        linebreak()
        text(size: 7.1pt, fill: grey)[#t.repetitions / #config.repetitions repetitions]
      }
    ]
    v(6pt)
    text(size: 7.2pt, fill: navy, weight: "bold")[ML init + warm #fmt(t.cold_seconds, digits: 3) s]
    linebreak()
    text(size: 6.9pt, fill: grey)[#t.repetitions retained repetitions]
    linebreak()
    if t.median_us != none {
      text(size: 6.9pt, fill: grey)[MAD #value-fmt(t.mad_us) µs / point]
    } else {
      text(size: 6.9pt, fill: grey)[No final timing inferred]
    }
  })
}
#let timing-grid(rows) = {
  set par(justify: false, leading: 0.25em)
  table(columns: (0.94fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr),
    align: top, inset: (x: 5pt, y: 8pt),
    stroke: (x, y) => (top: if y > 1 { 0.6pt + luma(205) } else { none },
      left: if x > 0 and y > 0 { 0.45pt + luma(224) } else { none }),
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header(header[Process], header[CutTools], header[Ninja], header[Collier], header[NinjaRS]),
    ..rows.map(c => (
      table.cell(align: top, {
        v(5pt)
        text(size: 10pt, weight: "semibold", label(c.id))
        linebreak()
        v(4pt)
        text(size: 7.3pt, fill: grey)[2 → #c.nfinal · #c.at("points", default: config.points_per_case) points]
        linebreak()
        text(size: 7.3pt, fill: grey)[#c.id]
      }),
      ..("1", "6", "7", "8").map(b => timing-card(c, b)),
    )).flatten())
}

#align(center)[
  #v(3pt)
  #text(29pt, weight: "bold", fill: navy)[NinjaRS in MadLoop]
  #v(-2pt)
  #text(12.5pt, fill: luma(85))[Fortran integration · scalar masters · measured benchmarks]
  #v(3pt)
  #text(8.5pt, fill: luma(110))[#data.generated_utc · #data.host_cpu]
]
#v(8pt)
#note[
  *Four reduction libraries, identical inputs.* Each coloured card gives the median
  complete evaluation time in *µs per phase-space point*, with its multiplier against
  *original MadLoop + Ninja*. The new NinjaRS interface uses oneloopmaster scalar
  integrals. All candidate cards use the same MadLoop source and input stream.
]
#v(7pt)
#section-label[QCD CORRECTIONS · LOOP_SM]
#timing-grid(data.cases.filter(c => c.id.starts-with("qcd")))
#v(7pt)
#legend
#v(5pt)
#small[
  Snapshot: #data.summary.measured_candidate_cells / #(data.summary.cases * 4) timing cards complete;
  #data.summary.correctness_passed / #data.summary.cases correctness cases passed;
  #data.summary.performance_passed / #data.summary.cases performance gates passed.
  Missing or incomplete observations remain pending. Measurements and validation are
  reported separately; an empty card is never treated as a speedup.
  Policy v2 validates the corrected model arithmetic. All 21 pairings among the
  seven corrected targets are required. Original-source discrepancies remain visible
  and the runtime denominator remains untouched original MadLoop with Ninja.
  The eight core gates and the loop-induced/stability supplement are separate:
  complete campaign #status(data.campaign_status).
]

#pagebreak()
#section-label[ELECTROWEAK CORRECTIONS · LOOP_QCD_QED_SM_GMU]
= Complete evaluation · electroweak controls
#small[Real masses and zero widths; all generated virtual correction orders are retained.
Full model names and MG5aMC process commands are listed with the reproducibility record.]
#v(7pt)
#timing-grid(data.cases.filter(c => c.id.starts-with("ew")))
#v(7pt)
#legend
#v(7pt)
#note[
  *Timing boundary.* Timed calls include the MadLoop numerator, loop reduction,
  scalar integrals, assembly and normal stability checks or rescues, together with
  per-point scale/coupling updates and return-code counting. Output is written after
  the timed region. “ML init + warm” begins at the first MadLoop evaluation and
  includes helicity initialization and one complete additional point-stream pass.
  It excludes parameter-card/model initialization, input reading and allocation.
]
#v(7pt)
#small[
  At least #config.repetitions complete repetitions are required before a card receives a
  timing. Calibration samples are excluded. Each repetition traverses the same
  #config.points_per_case distinct input points enough times to target a one-second
  timed interval. Native backend caches remain part of normal execution.
]

#pagebreak()
#section-label[LOOP-INDUCED AMPLITUDES · LOOP_SM]
= Complete evaluation · ggH and ggHH
#if data.supplement.cases.len() == 0 {
  note[*Supplement pending.* No validated final loop-induced timing dataset has
    been supplied. Earlier ten-point controls remain regression evidence; their
    times do not populate these final benchmark cards.]
} else {
  timing-grid(data.supplement.cases)
  v(7pt)
  legend
  v(7pt)
  tab((1.2fr, 0.7fr, 1fr, 1fr, 1.15fr),
    table.header(th[Process], th[Checks], th[Max. error / bound], th[Agreement], th[Timing gate]),
    ..data.supplement.cases.map(c => (
      label(c.id), value-fmt(c.agreement.comparisons),
      fmt(c.agreement.max_tolerance_fraction, digits: 4),
      status(c.agreement.status), status(c.performance.status),
    )).flatten(),
  )
}
#v(7pt)
#small[
  These squared one-loop amplitudes have no Born interference. All four Laurent/Born
  slots and every generated squared coupling order remain in the comparison;
  the Born slot is verified to vanish. The ggH stream contains boosted on-shell
  configurations with constrained invariants, rather than 100 independent
  invariant configurations. The ggHH stream probes a distinct box topology.
  The seven timing targets and three corrected-original references use the same
  physical input and full stability settings as the core campaign. Literal original
  Ninja remains the runtime denominator; neither a corrected reference nor an
  isolated tensor kernel can replace it.
]
#v(8pt)
= Supplemental campaign status
#note[
  *#status(data.supplement.status).* Completion requires both loop-induced
  numerical/timing gates, the twelve-point top-threshold stability grid and the
  final runtime's independent precision controls. Missing or failed evidence
  remains outstanding, even when the eight core cases pass.
]

#pagebreak()
#section-label[ADDITIONAL TWO-TO-FOUR · CORRECTNESS AND STATE REUSE]
= Top-pair plus jets and four-lepton stress tests
#note[
  *Stress campaign: #status(data.stress.status). No performance acceptance gate for
  these two new cases.* Twenty fixed ordinary
  points, four fixed rotations and eight scale-sequence calls per process exercise
  all four candidate backends. Two complete commands are retained for every stream.
  Their diagnostic times do not replace or relax the original eight core gates or
  the previously approved ggH/ggHH performance gates.
]
#v(7pt)
#if data.stress.cases.len() == 0 {
  small[Final stress evidence is pending. Missing, failed or incomplete mandatory
    candidate/rotation/scale comparisons prevent a complete campaign declaration.]
} else {
  let checked(gate) = [#status(gate.status)\ #scientific(gate.at("max_tolerance_fraction", default: none))]
  tab((1.5fr, 0.85fr, 1fr, 1fr, 0.95fr),
    table.header(th[Process], th[Ordinary], th[Rotation / invariance], th[Scale / restoration], th[Optional references]),
    ..data.stress.cases.map(c => (
      label(c.id), checked(c.ordinary.agreement),
      [#checked(c.rotation.agreement) / #checked(c.rotation.invariance)],
      [#checked(c.scale.agreement) / #checked(c.scale.restoration)],
      [#status(c.optional_reference.status)\
        #if c.optional_reference.at("agreement", default: none) != none {
          [agreement #checked(c.optional_reference.agreement)]
        }\ #c.optional_reference.at("failures", default: ()).len() retained failures],
    )).flatten(),
  )
  v(7pt)
  small[Each number is the worst component discrepancy divided by the unchanged
    acceptance bound. Optional corrected-original references, availability and
    failures remain distinct from the mandatory four-candidate comparisons.]
  v(8pt)
  let diagnostic(c, b) = {
    let t = c.ordinary.diagnostic_timings.at(b, default: none)
    if t == none { return [Pending] }
    [#value-fmt(t.median_us) µs\ init + warm #fmt(t.cold_seconds, digits: 3) s\
      #value-fmt(t.dp) DP / #value-fmt(t.qp) QP]
  }
  tab((1.4fr, 1fr, 1fr, 1fr, 1fr),
    table.header(th[Ordinary stream], th[CutTools], th[Ninja], th[Collier], th[NinjaRS]),
    ..data.stress.cases.map(c => (label(c.id), ..("1", "6", "7", "8").map(b => diagnostic(c, b)))).flatten(),
  )
  v(7pt)
  let observed(c, b, estimate: false) = {
    if c.ordinary.stability.status != "measured" { return [—] }
    let s = c.ordinary.stability.targets.at("candidate_corrected/" + b)
    scientific(if estimate { s.estimated_accuracy.all_orders.maximum }
               else { s.observed_discrepancy.max_point_scaled_error })
  }
  tab((1.25fr, 0.65fr, 1fr, 1fr, 1fr, 1fr),
    table.header(th[Process], th[Quantity], th[CutTools], th[Ninja], th[Collier], th[NinjaRS]),
    ..data.stress.cases.map(c => (
      label(c.id), [Observed], ..("1", "6", "7", "8").map(b => observed(c, b)),
      [], [Estimate], ..("1", "6", "7", "8").map(b => observed(c, b, estimate: true)),
    )).flatten(),
  )
}
#v(7pt)
#small[
  Diagnostic times summarize two ordinary-stream commands, including full numerator,
  reduction, scalar integrals and actual rescue. Init + warm is recorded separately;
  DP/QP counts sum the timed commands. No five-percent decision or confidence interval
  is inferred from these diagnostic times. “Observed” is the corrected-cohort
  point-scaled discrepancy; “Estimate” is the raw MadLoop estimate, never a measured
  precision. Per-point/order estimates, all H/T/U histograms, control-stream outcomes
  and failure records remain in the raw-bound stress snapshot.
  Rotations compare physical coefficients at matching points. The scale sequence
  $1,2,0.5,1$ checks matched-scale agreement and restoration of the first result;
  changed-scale amplitudes are allowed to change. Physical cards, pole settings,
  thresholds and numerical precision remain common across backends.
]

#pagebreak()
#section-label[VALIDATION · SAME INPUTS, EVERY COEFFICIENT]
= Numerical agreement
#tab((1.4fr, 0.65fr, 0.65fr, 0.9fr, 1.1fr, 0.85fr),
  table.header(th[Process], th[Points], th[Orders], th[Checks], th[Max. error / bound], th[Status]),
  ..data.cases.map(c => (
    label(c.id), fmt(c.agreement.points, digits: 0), fmt(c.agreement.orders, digits: 0),
    value-fmt(c.agreement.comparisons), fmt(c.agreement.max_tolerance_fraction, digits: 4),
    status(c.agreement.status),
  )).flatten(),
)
#v(7pt)
#small[
  Every point and squared coupling order contributes Born, finite, single-pole and
  double-pole values. All 21 pairings among four corrected candidate backends and
  three original-code backends with the verified model correction are tested.
  “Orders” includes the total order slot. The reported maximum is the ratio below,
  where $S$ is the largest
  absolute coefficient at that point across all orders in the corrected cohort. Agreement
  requires this ratio ≤ 1. Exactly zero pairs contribute zero.
]
#align(center)[#text(10.5pt)[$ abs(a-b) / (10^(-7) max(abs(a),abs(b)) + 10^(-10) S) $]]
#v(7pt)
= Performance acceptance · NinjaRS / original Ninja
#tab((1.4fr, 0.8fr, 0.9fr, 1.4fr, 1.05fr),
  table.header(th[Process], th[Repetitions], th[Median ratio], th[95% interval], th[Decision]),
  ..(data.cases + data.supplement.cases).map(c => (
    label(c.id), fmt(c.timings.candidate.at("8").repetitions, digits: 0),
    fmt(c.performance.runtime_ratio, digits: 3),
    if c.performance.ratio_confidence_interval_95 == none { [—] } else {
      [#fmt(c.performance.ratio_confidence_interval_95.at(0), digits: 3) – #fmt(c.performance.ratio_confidence_interval_95.at(1), digits: 3)]
    }, status(c.performance.status),
  )).flatten(),
)
#v(8pt)
#note[
  *Acceptance threshold: ×#config.max_runtime_ratio.* The harness resamples paired run indices
  10,000 times and forms a percentile 95% interval for the ratio of medians. A case
  passes when the upper bound is ≤ #config.max_runtime_ratio; it fails when the lower
  bound is > #config.max_runtime_ratio. An interval crossing the threshold is
  *unresolved*, not a pass. Each pair uses the same CPU, points and parameter card;
  measurement order alternates between repetitions.
]
#v(7pt)
#note[
  *Original-source numerical discrepancies are retained.* Corrected-original references
  change only the verified model helper and internal-parameter postimages, rebuild
  their model archive, and retain original MadLoop reduction code and libraries.
  Their timings never replace original Ninja in the performance denominator.
]

#pagebreak()
#section-label[MODEL PRECISION · ORIGINAL DISCREPANCIES RETAINED]
= Original-source numerical comparisons
#tab((1.1fr, 0.7fr, 0.8fr, 1.1fr),
  table.header(th[Process], th[Pairings], th[Mismatches], th[Original comparisons]),
  ..data.cases.map(c => (
    label(c.id), fmt(c.original_diagnostics.pairs, digits: 0),
    if c.original_diagnostics.status == "pending" { [—] } else {
      value-fmt(c.original_diagnostics.failures.len())
    }, status(c.original_diagnostics.status),
  )).flatten(),
)
#v(7pt)
#small[
  Seven pairings compare original CutTools, Ninja and Collier to their corrected
  references and candidate counterparts, plus original Ninja to candidate NinjaRS.
  Mismatches count point/order/coefficient records for each pairing; all records
  remain in the dataset and snapshot. Every original numerical discrepancy remains
  recorded; runtime and status failures remain disqualifying.
  These diagnostics are distinct from the mandatory 21 corrected-cohort pairings.
]
#v(8pt)
#note[
  *Why separate source roles are necessary.* The original MP model generator contains
  default-kind casts and a mistyped continuation constant. Compiled controls and
  independent higher-precision checks establish the exact corrections. A corrected
  result is assessed against independent corrected-original reduction paths;
  matching an erroneous original value is not an acceptance requirement.
]
#v(8pt)
= Verified correction and source separation
Fresh CLI generation proves each complete model-source delta. Corrected-original
references are cloned from archived original exports and receive only the exact
verified model postimages. All other generated source, reduction dependencies and
DHELAS libraries remain original. Their model objects, archive and subprocess are
rebuilt explicitly. Candidate exports additionally contain the NinjaRS interface
and validated Fortran optimization, with separate source identities.

A refreshed representative proof covers all three roles, fresh and reused workers,
all four DP backends, genuine QP rescue, loop-induced amplitudes and mixed coupling
orders. Missing references, invalid statuses, changed cards, missing points or stale
executables prevent publication. Runtime acceptance still requires the original
Ninja denominator, alternating paired order, at least seven repetitions and a
resolved 95% interval within the same 1.05 threshold.

#pagebreak()
#section-label[PREPARATION · SEPARATE WALL CLOCKS]
= Generation, compilation and initialization
#tab((1.2fr, 1fr, 1fr, 1fr, 1fr, 1.05fr),
  table.header(th[Process], th[Original gen.], th[Original build], th[Candidate gen.], th[Candidate build], th[NinjaRS ML init + warm]),
  ..data.cases.map(c => (
    label(c.id), fmt(prepare(c, "baseline", "generation"), digits: 2),
    build-time(c, "baseline"),
    fmt(prepare(c, "candidate", "generation"), digits: 2),
    build-time(c, "candidate"),
    fmt(c.timings.candidate.at("8").cold_seconds, digits: 3),
  )).flatten(),
)
#v(7pt)
#small[
  All preparation values are seconds. Generation is the recorded CLI model import,
  process generation and standalone export invocation; it can include library work
  performed by export. Build is the retained `make -j4 check` invocation:
  *C* begins with no subprocess object files; *I* is incremental; *?* was not classified.
  A clean subprocess build does not include rebuilding external libraries.
  ML initialization plus warming is the median of the completed run preparations
  beginning at their first MadLoop call; parameter-card/model setup is excluded.
  Generation is pending when its retained identity does not match the measured build.
  Generation and compilation have no performance acceptance gate.
]
#v(8pt)
= Measurement method
The process list is fixed in `cases.json`. For two-to-many scattering, the harness
constructs deterministic RAMBO-derived on-shell momenta at #config.energy_gev GeV,
including the generated external masses. For 2 → 1, a physical on-shell resonance
is produced with distinct three-dimensional boosts. Momentum conservation and masses
are checked before measurements; each case uses a different deterministic seed.

The same retained text point stream and parameter-card SHA-256 hashes are required
across reducers. All decay widths are set to zero. Each point sets the renormalization
scale to its partonic centre-of-mass energy. Model parameters and couplings otherwise
come from the generated card.

Each executable run selects one reduction library. The driver forces stability
checks, allows normal rescue within that backend, and records every timed return
code. The harness rejects initialization codes, unchecked estimates, failures and
unexpected backend identities. It rebuilds helicity filters before each invocation.
The timing baseline is the untouched original source revision recorded on the final
page. Each corrected-original reference is separately identified and compared against
all four candidate backends. The common Collier scalar accuracy is $10^(-12)$ and
the MadLoop stability threshold remains $10^(-9)$.

The process is pinned to one CPU with `taskset`; OpenMP and OpenBLAS thread counts
are one. Generation may use four build jobs. Timing uses Fortran `SYSTEM_CLOCK`,
and includes the driver’s scale updates and histogram increment. Calibration, text
input, numerical output and initialization are outside the warm interval. No
separate scalar-master or numerator-stage timing is inferred from total runtime.
#v(6pt)
#note[
  *Limits of interpretation.* Confidence intervals quantify the retained run-to-run
  samples on this host; they do not establish performance on other CPUs or workloads.
  A shared host can introduce noise. Finite random streams are a regression test,
  not a proof over all phase space. Pending correctness, rescue or timing results
  remain outstanding requirements.
]

#pagebreak()
#section-label[STABILITY · ACTUAL BACKEND USE]
= Timed double- and higher-precision evaluations
#let count-cell(c, b) = {
  let t = c.timings.candidate.at(b)
  if t.timed_calls == none { text(fill: grey)[Pending] } else {
    [#value-fmt(t.dp) DP\ #value-fmt(t.qp) QP]
  }
}
#tab((1.2fr, 1fr, 1fr, 1fr, 1fr),
  table.header(th[Process], th[CutTools], th[Ninja], th[Collier], th[NinjaRS]),
  ..(data.cases + data.supplement.cases).map(c => (label(c.id), ..("1", "6", "7", "8").map(b => count-cell(c, b)))).flatten(),
)
#v(7pt)
#small[
  Counts sum all retained complete timed repetitions and include
  repeated traversals of the point stream. They classify final successful evaluation
  status, rather than counting individual internal loop-reduction attempts. Full
  histograms are retained in `report-data.json` and the original measurement JSON.
  A pending entry means no accepted run is present.
]
#v(7pt)
= Backward-compatible status interpretation
#tab((1.3fr, 1fr, 1.8fr), left-columns: (0, 1, 2),
  table.header(th[Backend], th[Precision], th[Hundreds / units digits]),
  [CutTools], [DP / QP], [$H=2, U=1$ / $H=3, U=9$],
  [Ninja], [DP / QP], [$H=2, U=6$ / $H=3, U=8$],
  [Collier], [DP], [$H=2, U=7$],
  [NinjaRS], [DP / QP], [$H=2, U=8$ / $H=3, U=2$],
)
#v(7pt)
#small[
  The tens digit must identify a completed stability check (1 or 2). Old return
  integers are unchanged: the hundreds digit distinguishes Ninja QP from NinjaRS DP
  and PJFry++ DP from NinjaRS QP. Dedicated NinjaRS counters preserve legacy
  statistics interpretation.
]
#v(7pt)
= Integration audit
The Fortran wrapper preserves MadLoop’s momentum ordering, odd-rank sign reversal,
thresholded invariants, Laurent ordering and normalization. Compiled recording-backend
tests exercise both amplitude and interference paths, REAL16 input beyond binary64
precision, failures and nonfinite output, card parsing, old return codes, and cleanup
with or without the installed module.

The bridge audit corrected the generic higher-precision auxiliary sampling scale
to match Ninja’s invariant/mass scale. Scalar tests cover spacelike massless and
massive bubble orderings and reuse of coefficients, masses and scale. Higher-precision
transport carries actual binary128 input into 256-bit arithmetic. Context cleanup
joins the retained worker. See `audit.md` for scope, evidence and remaining limits.

#pagebreak()
#section-label[NUMERICAL STABILITY · OBSERVED AND ESTIMATED]
= Independent discrepancies and MadLoop estimates
#let stable-cell(c, b, kind) = {
  if c.stability.status != "measured" { return [—] }
  let s = c.stability.targets.at("candidate_corrected/" + b)
  if kind == "estimate" { scientific(s.estimated_accuracy.all_orders.maximum) }
  else { scientific(s.observed_discrepancy.max_point_scaled_error) }
}
#tab((1.05fr, 0.65fr, 1fr, 1fr, 1fr, 1fr),
  table.header(th[Process], th[Quantity], th[CutTools], th[Ninja], th[Collier], th[NinjaRS]),
  ..(data.cases + data.supplement.cases).map(c => (
    label(c.id), [Observed], ..("1", "6", "7", "8").map(b => stable-cell(c, b, "observed")),
    [], [Estimate], ..("1", "6", "7", "8").map(b => stable-cell(c, b, "estimate")),
  )).flatten(),
)
#v(7pt)
#note[
  *Different quantities.* “Observed” is the largest absolute coefficient difference
  from any other corrected-cohort target, divided by the common point's largest
  Born/Laurent magnitude across all orders. It is an independent cross-backend
  discrepancy, not a known error against an exact amplitude. “Estimate” is the
  largest MadLoop stability estimate over the retained point/order slots. It is
  reported without treating it as a measured error or a proof of significant digits.
  An exactly zero estimate is shown as zero. The acceptance test uses the separate
  component-dependent tolerance defined earlier.
]
#v(7pt)
#small[
  The snapshot retains minimum, median, nearest-rank 95th percentile, maximum and
  zero count for estimates over all orders and for the total-order slot separately.
  Actual rescue counts refer to successful final statuses once per physical point;
  timed histograms count every traversal. Neither count measures internal reduction
  attempts. Worst discrepant point/order/coefficient records remain available even
  for passing cases. Finite sampled agreement does not bound the instability rate
  elsewhere in phase space, and agreement between backends is not an analytic oracle.
]
#v(8pt)
= Top-pair threshold in loop-induced ggHH
#if data.supplement.threshold_grid == none {
  small[The final twelve-point threshold grid is pending. No stability rate is inferred
    from earlier random-point or focused-control measurements.]
} else {
  let grid = data.supplement.threshold_grid
  note[
    *Grid agreement: #status(grid.agreement.status).* Twelve fixed configurations use
    $sqrt(s) = 2 m_t (1 + delta)$, $delta = plus.minus(10^(-2), 10^(-4), 10^(-6))$,
    $cos(theta) = 0, 0.8$, azimuth zero and the unchanged zero-width physical card.
    Every recorded point/order/Laurent coefficient is compared across the seven
    corrected targets. All failures and raw estimates remain in the supplement.
  ]
  if grid.at("point_summaries", default: ()).len() > 0 {
    v(6pt)
    tab((0.45fr, 0.9fr, 0.65fr, 1fr, 1.15fr),
      table.header(th[Point], th[δ], th[cos θ], th[Max. error / bound], th[NinjaRS estimate]),
      ..grid.point_summaries.map(p => (
        str(p.point), str(p.delta), str(p.cos_theta),
        scientific(p.max_tolerance_fraction),
        scientific(p.estimated_accuracy.at("candidate_corrected/8")),
      )).flatten(),
    )
  }
}

#pagebreak()
#section-label[HIGHER PRECISION · CONTROLS AND OPTIMIZATION]
= Genuine higher-precision rescue
MadLoop recomputes numerator coefficients and kinematics in its existing binary128
path. Forty-significant-digit ASCII records transport each original 113-bit value
through the C ABI before exact widening to 256-bit arithmetic. Signed zeros and
values distinguishable only beyond binary64 are preserved. The generic reduction
engine uses oneloopmaster-backed scalar masters at the same working precision.
MadLoop retains retry ordering, rotations, acceptance and final result conversion.

The dimensionless MadLoop on-shell threshold remains distinct from the scalar
provider's absolute threshold. Higher precision does not relax stability checks,
omit poles or alter the physical card. Stored-input 256/384-bit convergence and
analytic pole controls have a separate scope from full-amplitude agreement.
#v(7pt)
#if data.supplement.precision_controls == none {
  note[*Final installed-runtime precision controls pending.* Earlier transport,
    reset, scalar and convergence fixtures remain source-specific research evidence
    and do not certify a newly installed archive.]
} else {
  let precision = data.supplement.precision_controls
  note[*Final precision controls: #status(precision.status).* Working precision
    #precision.working_bits bits; independent convergence checks at
    #precision.convergence_bits bits. Full Laurent coefficients are the physical
    cross-basis oracle; rational R1 contributions are retained separately.]
  if precision.at("checks", default: ()).len() > 0 {
    v(7pt)
    tab((2.7fr, 0.7fr),
      table.header(th[Control], th[Decision]),
      ..precision.checks.map(c => (c.name, status(c.status))).flatten(),
    )
  }
}
#v(8pt)
= Measured optimizations and retained negative results
The retained Fortran QP contraction-sharing prototype resolves a 3.19% full QCD3
runtime reduction with Ninja (backend 6) against its pre-factor candidate generated at revision
`b44d20e725f1ec68589df20284e4ce9116e9e2ab`: ratio 0.968095, 95% interval [0.958491, 0.982138]. Coefficients, stability
estimates and rescue counts are unchanged. This earlier backend and revision
scope is distinct from the final original-Ninja denominator and five-percent gate.

The accepted allocation-only tensor change resolves a *3.15% full MadLoop QCD3
runtime reduction*: ratio 0.968518, paired 95% interval [0.956878, 0.993640]
over 21 pairs. The EW2 control uses seven pairs: ratio 0.997640, interval
[0.994716, 0.999339]. Numerator construction, reduction, scalar integrals,
stability checks and actual rescue are timed together. All 44,800 coefficient
comparisons and raw accuracy/status rows match exactly. The retained measured
release tensor body is the source of the accepted production change; this study
uses generated Fortran from revision
`f4bffc61ed67e5d083b80ff9901768d4d5779989` and compares two NinjaRS tensor
implementations. Its original/allocation archives, `832fc0d11988` / `366ae91da891`,
are historical timing identities, distinct from the final installation `800a941faf3c`. This
study does not replace the final eight-process original-Ninja acceptance campaign.

A separate DP factorization study finds no full-process gain: ratio 1.013546,
95% interval [0.996598, 1.030949]. That DP implementation is not promoted.
The earlier allocation-only higher-precision tensor experiment finds a 9.24%
*callback-only* gain: ratio 0.907561, interval [0.898288, 0.912750], with exact
value, precision and sign matches. Its timing boundary differs from full MadLoop.

The native Float256 request screen is *rejected*: neither native arm demonstrates
a gain over interpreted execution in seven fixed triplets of a 1,392-request
stream. Denominator-only/interpreted gives 1.021698 [0.973592, 1.045666];
full-native/interpreted gives 1.007070 [0.981060, 1.040110]. Both intervals cross
one, establishing neither improvement nor slowdown. Parsing, state reset,
reduction, masters and formatting are included; Fortran numerator/kinematics,
the stability controller and public-manager IPC are excluded. These are
request-stream measurements; “full-native” names an arithmetic
policy, not a full MadLoop timing. Interpreted execution is retained and no
native dispatch or Cargo feature is promoted.
#v(7pt)
#small[
  Exact retained research bindings (SHA-256):\
  Allocation analysis: `ninjars-mp-tensor-allocation-process-v2/timings/attempt-1/analysis.json`\
  `79e661e62a4ce253181bdff22fa1dd530269ccbe570b61e2e2b9c9712e3f1a4d`\
  Historical source/archive bindings: `ninjars-mp-tensor-allocation-process-v2/frozen-bindings.json`\
  `64766a546d751705993e4b074f830d9ff32260c89e4ecfd6f6e4aa9114e2871e`\
  Measured release tensor body: `ninjars-mp-tensor-allocation-transport-v1/production-tensor.rs`\
  `240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70`\
  Native selection analysis: `ninjars-native-float256-v1/selection-analysis.json`\
  `0581c1fe7ac1371197610c894a360a8ccd82798a3970a167b9d6fc3002d5044c`\
  Paths are relative to `docs/benchmarkfolder/experiments/`. Earlier scopes remain
  in `symmetric-factorization*`, `symmetric-factorization-dp/process-timing21-v2`
  and `ninjars-mp-tensor-allocation-performance-v1`. Percentages from different
  timing boundaries or source/runtime revisions are not added together.
]

#pagebreak()
#section-label[MG7 REGRESSIONS · REVIEWED TEST COVERAGE]
= Generator, Fortran and export regressions
#if data.regressions.status == "pending" {
  note[*Final regression evidence pending.* The report has not been supplied with
    a checked MG7 regression manifest for the measured candidate generator.]
} else {
  tab((2fr, 0.6fr, 0.6fr, 0.6fr, 0.6fr),
    table.header(th[Suite], th[Run], th[Failures], th[Errors], th[Skipped]),
    ..data.regressions.suites.map(s => (
      s.name, str(s.tests_run), str(s.failures), str(s.errors), str(s.skipped),
    )).flatten(),
  )
  v(7pt)
  note[
    *#data.regressions.tests_run cases run; #data.regressions.tests_passed passed;
    #data.regressions.total_skipped explicitly skipped.* Raw unittest footers,
    skip reasons, reviewed IO fixtures and source hashes are independently checked.
    Every tested production file in the generator scope must match the measured
    candidate source exactly.
  ]
  for s in data.regressions.suites {
    if s.skip_details.len() > 0 {
      v(6pt)
      small[#s.name: #s.skip_details.map(item => item.at(0) + " — " + item.at(1)).join("; ")]
    }
  }
  v(7pt)
  small[#data.regressions.scope]
}
#v(8pt)
#small[
  These suites cover their recorded generator, Fortran and exporter paths.
  They do not execute the Rust reduction/scalar engine or replace the final
  physical process campaign. Installed-archive numerical, precision, lifecycle
  and convergence controls are reported separately. Historical incomplete or
  failed fixture runs remain in the evidence; correcting a stale expected IO
  fixture does not turn the original failed attempt into a passed run.
]

#pagebreak()
#section-label[PROFILING · SAMPLED CPU WORK]
= Where evaluation time is spent
#note[
  *Diagnostic samples, separate from timing acceptance.* Profiles run a copy of an
  exact measured worker after initialization and warming. Sampling includes profiler
  overhead and output serialization between complete stream traversals. Percentages
  are CPU-clock sample shares, not exclusive stage timers. They are never used to
  decide the five-percent performance gate.
]
#v(7pt)
#let profile-labels = (
  fortran_dp_numerator: [Fortran numerator · DP],
  fortran_qp_numerator: [Fortran numerator · QP],
  dp_tensor: [Native DP tensor expansion],
  compensated_dp_tensor: [Compensated DP tensor expansion],
  mp_tensor: [Higher-precision tensor expansion],
  mp_scalar_masters: [Higher-precision scalar masters],
  mp_geometry: [Higher-precision cut geometry],
  mp_reduction: [Higher-precision reduction],
  bridge_numeric_transport: [Numeric transport],
  other: [Other / unclassified],
)
#let source-labels = (
  original: "original",
  candidate_corrected: "candidate",
  original_helper_corrected: "corrected original",
)
#let backend-labels = ("1": "CutTools", "6": "Ninja", "7": "Collier", "8": "NinjaRS")
#if data.profiles.len() == 0 {
  small[No diagnostic profile has been supplied for this dataset. Earlier research
    profiles remain separate evidence and are not presented as current measurements.]
} else {
  for p in data.profiles {
    block(breakable: false)[
      #v(7pt)
      #text(fill: navy, weight: "semibold")[
        #label(p.case) · #source-labels.at(p.role) #backend-labels.at(str(p.backend))
      ]
      #linebreak()
      #small[
        #p.points points · #value-fmt(p.evaluations) evaluations · #p.blocks blocks ·
        #value-fmt(p.samples) samples · CPU #p.cpu
      ]
      #v(4pt)
      #tab((2fr, 0.8fr, 0.8fr),
        table.header(th[Observed work], th[Samples], th[Share]),
        ..p.categories.pairs().map(((name, item)) => (
          profile-labels.at(name), value-fmt(item.samples),
          [#fmt(item.percent, digits: 1)%],
        )).flatten(),
      )
    ]
  }
}
#v(8pt)
#small[
  Each sample uses the first recognized stack owner or ancestor; unmatched work
  stays “other”. In particular, the scalar-master category covers higher-precision
  NinjaRS masters only. Every recorded block must reproduce all point/order
  coefficients, accuracy estimates and statuses of its bound measurement exactly.
  The report reader rechecks source/runtime identities, cards, retained raw outputs,
  raw recording hashes and stack-category counts. Full paths, hashes and usage
  counts are in `report-data.json`.
]

#pagebreak()
#section-label[REPRODUCIBILITY · PINNED SOURCES AND RETAINED INPUTS]
= Source and toolchain record
#let candidates = data.runtime_provenance.source_roles.at("candidate_corrected", default: ())
#tab((1.05fr, 2.6fr), left-columns: (0, 1),
  table.header(th[Component], th[Recorded revision / version]),
  [Original MadLoop], text(size: 7.4pt, data.provenance.baseline_revision),
  [Candidate source], text(size: 7.4pt, if candidates.len() == 0 { "Pending" } else {
    candidates.map(c => c.at("revision", default: "Revision not recorded")).join(", ")
  }),
  [NinjaRS], recorded-revision(data.runtime_provenance.dependencies.ninjars),
  [oneloopmaster], recorded-revision(data.runtime_provenance.dependencies.oneloopmaster),
  [Symbolica], recorded-revision(data.runtime_provenance.dependencies.symbolica),
  [Measured runtime ABI], [#if data.runtime_provenance.abi_version == none { [Pending] } else { str(data.runtime_provenance.abi_version) } · provenance #status(data.runtime_provenance.status)],
  [NinjaRS archive SHA-256], text(size: 7.4pt, if data.runtime_provenance.archive_sha256.len() == 0 { "Pending" } else { data.runtime_provenance.archive_sha256.join(", ") }),
  [Compilers], [Rust #data.provenance.toolchain.rustc · gfortran #data.provenance.toolchain.gfortran],
  [Report tools], [Python #data.provenance.toolchain.python · Typst #data.provenance.toolchain.typst],
  [Reference libraries], [Ninja #data.provenance.reference_libraries.Ninja.version · OneLOop #data.provenance.reference_libraries.OneLOop.version · Collier #data.provenance.reference_libraries.Collier.version],
)
#v(7pt)
#tab((0.6fr, 2.1fr, 1fr), left-columns: (0, 1, 2),
  table.header(th[Case], th[MG5aMC generate command], th[Point SHA-256 prefix]),
  ..data.cases.map(c => (
    c.id, text(font: "DejaVu Sans Mono", size: 7.2pt, c.process),
    text(size: 7.5pt, short-hash(c.point_hash)),
  )).flatten(),
)
#v(7pt)
#small[
  QCD cases use `loop_sm`; electroweak cases use `loop_qcd_qed_sm_Gmu`.
  Setup source/archive and model hashes are retained in `provenance.json`.
  Final NinjaRS dependency revisions above come from retained installed Cargo files
  whose successful installation record matches the measured library hash. Historical
  setup pins and ABI metadata never substitute for a new runtime's provenance.
  Full point/card hashes, per-run executable hashes when recorded, raw timings,
  Laurent coefficients, statuses and run logs remain with the benchmark inputs.
  `report-data.json` includes a SHA-256 manifest of the inputs used to build it.
]
#v(7pt)
= Reproduce this validated report
```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python tools/ninjars_report_v2.py DATASET --compile \
  --timestamp 'SNAPSHOT_UTC' --output NEW_REPORT_DIRECTORY
```
#small[
  Replace `DATASET` with the retained dataset path in `report-data.json`.
  Replace `SNAPSHOT_UTC` with its `generated_utc` value and choose a new output
  directory. Reusing this date preserves both the visible date and PDF bytes.
  Collection and correction specifications, CLI cards, all source roles, exact
  postimages, fresh/reused worker proof and raw measurements are bound by the
  dataset manifest. See `policy-v2/README.md` for the complete collection sequence.
  The renderer independently recomputes all numerical comparisons and confidence
  intervals. It rejects changed inputs and stale conclusions, and writes into a
  new directory so earlier reports remain intact.
  When diagnostic profiles are included, also pass `--profile-spec` with the
  retained profile manifest listed in the snapshot's input hashes.
  When `generation_inputs` is set in the snapshot, pass that retained directory
  with `--generation-inputs` to reproduce its generation measurements.
  When `supplement_specification` is set, pass the same retained manifest with
  `--supplement`; its loop-induced, threshold and precision evidence must match
  the core candidate generator and numerical runtime identities exactly.
  When `runtime_provenance_specification` is set, also pass that manifest with
  `--runtime-provenance`. The report rejects altered installed Cargo revisions,
  configuration or archive identity. These final dependency pins do not come
  from the live checkout or the earlier setup metadata.
  When `validation_specification` is set, pass it with `--validation-spec`.
  Its retained verifier rechecks raw suite logs, skip details, reviewed fixtures
  and tested production hashes against the measured candidate generator.
  Pass `--stress` with `stress_specification` for the additional fixed two-to-four
  controls. Its reader checks the same core source identity and complete measured
  runtime descriptor. The diagnostic timings carry no additional performance gate.
]
#v(5pt)
#small[
  Dataset #short-hash(data.dataset_identity_sha256) ·
  policy #short-hash(data.policy_identity_sha256) ·
  correction #short-hash(data.correction_identity_sha256) ·
  representative proof #short-hash(data.representative_proof_identity_sha256).
]
#v(7pt)
#note[
  *Style attribution.* Typography, margins, navy headings, pale notes, striped tables,
  provider cards and the logarithmic green–white–red ratio palette follow
  #link(data.provenance.report_style_source)[MadGraphTeam/pyamplicol’s one-loop benchmark report].
  Its physics results and performance claims are not reused here.
]
