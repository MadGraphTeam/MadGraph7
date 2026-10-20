// Regenerate the measured snapshot and PDF: python tools/ninjars_report.py --compile
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
).at(key)
#let status(s) = text(weight: "semibold", fill: (
  passed: rgb("#287240"), failed: rgb("#b44339"), unresolved: rgb("#966016"),
  pending: grey, partial: grey, measured: navy,
).at(s, default: grey), s)
#let short-hash(s) = if s == none { [Pending] } else { str(s).slice(0, 12) }
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
        text(size: 7.3pt, fill: grey)[2 → #c.nfinal · #config.points_per_case points]
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
  double-pole values. The six pairings of CutTools, Ninja, Collier and NinjaRS are tested.
  “Orders” includes the total order slot. The reported maximum is the ratio below,
  where $S$ is the largest
  absolute coefficient at that point across all orders and reducers. Agreement
  requires this ratio ≤ 1. Exactly zero pairs contribute zero.
]
#align(center)[#text(10.5pt)[$ abs(a-b) / (10^(-7) max(abs(a),abs(b)) + 10^(-10) S) $]]
#v(7pt)
= Performance acceptance · NinjaRS / original Ninja
#tab((1.4fr, 0.8fr, 0.9fr, 1.4fr, 1.05fr),
  table.header(th[Process], th[Repetitions], th[Median ratio], th[95% interval], th[Decision]),
  ..data.cases.map(c => (
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
#if data.provenance.at("outstanding_findings", default: ()).len() > 0 {
  v(7pt)
  note[
    *Outstanding diagnostics.* These are retained investigations, separate from
    accepted timing series.
    #for finding in data.provenance.outstanding_findings [
      #parbreak()
      #finding
    ]
  ]
}

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
The baseline is the untouched original source revision recorded on the final page.

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
  ..data.cases.map(c => (label(c.id), ..("1", "6", "7", "8").map(b => count-cell(c, b)))).flatten(),
)
#v(7pt)
#small[
  Counts sum all retained timed repetitions, including partial schedules, and include
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
#section-label[REPRODUCIBILITY · PINNED SOURCES AND RETAINED INPUTS]
= Source and toolchain record
#tab((1.05fr, 2.6fr), left-columns: (0, 1),
  table.header(th[Component], th[Recorded revision / version]),
  [Original MadLoop], text(size: 7.4pt, data.provenance.baseline_revision),
  [NinjaRS], text(size: 7.4pt, data.provenance.ninjars_revision),
  [oneloopmaster], text(size: 7.4pt, data.provenance.oneloopmaster_revision),
  [Symbolica], text(size: 7.4pt, data.provenance.symbolica_revision),
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
  Full dependency revisions, source archive and model hashes are in `provenance.json`.
  Full point/card hashes, per-run executable hashes when recorded, raw timings,
  Laurent coefficients, statuses and run logs remain with the benchmark inputs.
  `report-data.json` includes a SHA-256 manifest of the inputs used to build it.
]
#v(7pt)
= Rebuild the measurements and report
```sh
source .ninjars-build/dev-env.sh
python tools/ninjars_benchmark.py generate --source both
python tools/ninjars_benchmark.py build --source both
python tools/ninjars_benchmark_workers.py measure --source both \
  --cases all --cpu 207 --repetitions 7 --clean-build --run-id core
python tools/ninjars_benchmark_workers.py promote --runs \
  docs/benchmarkfolder/experiments/persistent-workers/core
python tools/ninjars_report.py --compile
```
#small[
  Run from the repository root after installing the pinned libraries and preparing
  the untouched baseline checkout. Choose an allowed quiet CPU; 207 is this run's
  choice. Refresh representative protocol validation as described in
  `persistent-workers.md`. Per-case selection uses `--cases qcd-1`.
  Failed and unresolved performance comparisons exit unsuccessfully and retain
  their JSON evidence. The report generator rejects stale summaries and mismatched
  input streams instead of silently combining them.
]
#v(7pt)
#note[
  *Style attribution.* Typography, margins, navy headings, pale notes, striped tables,
  provider cards and the logarithmic green–white–red ratio palette follow
  #link(data.provenance.report_style_source)[MadGraphTeam/pyamplicol’s one-loop benchmark report].
  Its physics results and performance claims are not reused here.
]
