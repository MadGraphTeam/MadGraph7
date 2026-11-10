// Separate historical diagnostic: all definitions remain local to this block.
#{
  import "style.typ": *
  let data = json("experiments/poles-timing/historical-release-paired/analysis-summary.json")
  let tensor-name(row) = (
    "massive-tadpole-r6": "Massive tadpole",
    "massive-bubble-r4": "Massive bubble",
    "infrared-triangle-r3": "Infrared triangle",
    "massive-box-r4": "Massive box",
    "massive-5point-r2": "Massive pentagon",
    "massive-6point-r2": "Massive hexagon",
  ).at(row.name)
  pagebreak(weak: true)
  heading(level: 1)[Full versus poles-only tensors]

  [#data.measured_settings of #data.settings.len() predeclared settings completed
   seven alternating full/poles-only timing pairs; two DP settings were refused
   in both modes. All returned requested pole coefficients agree. These warmed
   tensor diagnostics use the historical release, with numerical caches disabled.
   They do not qualify full-amplitude performance.]
  v(6pt)

  figure(
    table-style(table(
      columns: (1.35fr, .68fr, .40fr, .65fr, .65fr, 1.12fr),
      inset: (x: 4pt, y: 5pt), stroke: table-stroke, fill: table-fill,
      table.header([Tensor · N / rank], [Route], [Prec.],
                   [Full µs], [Poles µs], [Poles / full · 95% interval]),
      ..data.settings.map(row => {
        let measured = row.status == "measured"
        let stats = row.at("wall_poles_over_full", default: none)
        let ratio = if stats == none { none } else { stats.geometric_mean }
        let route = if row.scope == "generated_adapter" {
          [Generated#linebreak()adapter]
        } else {
          [Public#linebreak()Fortran]
        }
        (
          [#text(8.1pt, font: "Libertinus Serif")[#tensor-name(row)]#linebreak()
           #text(7pt, fill: ink2)[N=#row.n · rank #row.rank]],
          route,
          [#row.precision],
          [#number(row.at("median_full_wall_microseconds", default: none))],
          [#number(row.at("median_poles_wall_microseconds", default: none))],
          table.cell(fill: if measured { ratio-fill(ratio) } else { warm }, [
            #if measured {
              [#text(10pt, weight: "bold")[#ratio-text(ratio)]#linebreak()
               #fixed(stats.confidence_interval_95.at(0), 3)–#fixed(stats.confidence_interval_95.at(1), 3)]
            } else { text(fill: status-ink("refused"))[Both modes#linebreak()refused] }
          ]),
        )
      }).flatten(),
    )),
    caption: [Median wall µs per call and geometric mean paired ratios. Green
      favors poles-only; red favors full evaluation. Intervals use seven paired
      log ratios and Student-t with six degrees of freedom.],
  )
  v(4pt)

  note[
    *Scope and observed limits.* The rank-six tadpole calls the public Fortran
    bridge; the other tensors use the existing generated QCD 2→4 adapter,
    whose rank capacity is five. No generated rank limit was changed. The DP
    tadpole is about 4% slower in poles-only mode; QP is near parity. The massive
    N=5 and N=6 fixtures have zero poles, so their large QP gains do not describe
    infrared-singular pentagons or hexagons. DP refusals carry no timing claim.
  ]
  v(5pt)
  text(8.2pt)[
    Numerical caches are off; structural plans and scratch remain warm. Each
    measured batch lasted 0.825–1.925 s on pinned CPU 9. Driver tracing and
    coefficient output are outside the timer; input restoration, adapter mapping
    and bridge serialization remain inside. Shared-host compilation was active;
    host-load snapshots and CPU times are retained. These timings are explicitly
    ineligible for the full-amplitude acceptance gate.
  ]
  v(4pt)
  text(8.2pt)[
    Requested poles match within the declared DP/QP bounds; the largest scaled
    residual is #raw("3.47e-18"). Omitted finite slots are NaN in the generated
    adapter and zero placeholders in the public bridge. The historical DSO is
    pinned by SHA256 #raw("49af12f0b882…cd71ca12") and subsequent builds are outside
    this measurement. Raw batches, exact identities, and the independently
    audited 79-file inventory are retained under #raw("experiments/poles-timing/").
  ]
}
