#import "style.typ": *

#let pair-values(values, scale: 1, digits: 3) = if values == none { [—] } else {
  [#fixed(values.baseline * scale, digits) / #fixed(values.candidate * scale, digits)]
}
#let usage(values) = if values == none { [unavailable] } else {
  values.pairs().map(p => p.at(0) + "=" + str(p.at(1))).join(", ")
}

#let show-provisional(data, cases, process-label) = {
  pagebreak()
  heading(level: 1)[Shared-host observations · descriptive only]
  note(fill: warm)[These closed observations use *three alternating pairs*, with
    all *100 saved points* in every traversal. They provide no confidence interval
    and no ≤1.05 acceptance verdict. Final precise timing remains Hua-Sheng’s local
    TODO. CPUs were not reserved. Unavailable and timed-out work supplies no ratio.]
  v(7pt)
  block[
    #set par(justify: false)
    #text(7.5pt, fill: ink2)[Frozen candidate exporter: #raw(data.candidate_exporter_revision).#linebreak()
      Source identity: #raw(data.candidate_source_sha256).#linebreak()
      Numerical identity: #raw(data.candidate_numerical_identity_sha256).#linebreak()
      Baseline: #raw(data.baseline_revision).#linebreak()
      These measurements apply only to this older frozen candidate. They do not
      measure later symmetry-v2 or other experimental source changes.]
  ]
  v(7pt)
  let label(row) = {
    let c = cases.find(c => c.id == row.case)
    if c == none { row.case } else { process-label(c) }
  }
  let ratio(row) = if row.ratio == none { [—] } else { [#fixed(row.ratio, 4)] }
  let state(row) = if row.status == "complete" { [observed · numerical pass] } else {
    row.status.replace("_", " ")
  }
  table-style(table(columns: (1.5fr, 1.25fr, .65fr, 1fr, 1fr), inset: 5pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Chain / process], [Closed status], [C/B ratio], [µs/point B / C], [Init. s B / C]),
    ..data.rows.filter(row => row.kind == "chain").map(row => (
      [#text(8pt, font: "Libertinus Serif")[#label(row)]],
      [#text(6.8pt)[#state(row)]], table.cell(fill: ratio-fill(row.ratio), [#ratio(row)]),
      [#pair-values(row.median_seconds_per_point, scale: 1000000, digits: 2)],
      [#pair-values(row.initialization_seconds)],
    )).flatten()))
  v(5pt)
  text(7.5pt, fill: ink2)[C/B is the geometric mean of the three paired ratios;
    µs/point is each side’s median. Initialization includes the original warmup
    and is reported separately. The candidate chain prepends TIRust to the
    baseline’s materialized default chain. No performance gate changes.]
  v(6pt)
  text(7pt, fill: ink2)[Controller status: #data.closed_status.
    Observed elapsed: #if data.elapsed_seconds == none { [unavailable] } else { [#fixed(data.elapsed_seconds, 3) s] };
    work budget: #data.budget_seconds s.
    Publication scope: #data.publication_scope.
    #if data.publication_scope != none and data.publication_scope.starts-with("administrative-after-deadline") {
      [Administrative closure after budget expiry is retained; its publication is not a timing sample.]
    }]

  pagebreak()
  heading(level: 1)[Standalone observations and unavailable comparisons]
  note(fill: warm)[Each row retains its original schedule status. Backend 9 is
    compared with baseline CutTools; it is not compared with a previous TIRust
    implementation. The other standalone rows compare the same reducer across
    the frozen baseline and candidate. All performance acceptance remains deferred.]
  v(7pt)
  table-style(table(columns: (1.35fr, .3fr, 1.2fr, .6fr, 1fr, .9fr), inset: 4pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Process], [ID], [Closed status], [C/B ratio], [µs/point B / C], [Init. s B / C]),
    ..data.rows.filter(row => row.kind == "standalone").map(row => (
      [#text(7.1pt, font: "Libertinus Serif")[#label(row)]], [#row.backend],
      [#text(6.3pt)[#state(row)]], table.cell(fill: ratio-fill(row.ratio), [#ratio(row)]),
      [#pair-values(row.median_seconds_per_point, scale: 1000000, digits: 2)],
      [#pair-values(row.initialization_seconds)],
    )).flatten()))

  pagebreak()
  heading(level: 1)[Observed batches and accepted precision use]
  text(7.5pt, fill: ink2)[All three batch durations and their ratio range are
    shown for completed numerical passes only. Usage counts include every timed
    traversal and describe final accepted amplitudes. Internal reducer attempts,
    retries and IEEE flags are unobserved. Unavailable rows above have no usage
    measurement. These are full-amplitude timings.]
  v(7pt)
  table-style(table(columns: (1.3fr, 1fr, 1fr, 1.4fr), inset: 4pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Comparison], [Batch s B / C], [Ratio range], [Accepted backend:precision counts]),
    ..data.rows.filter(row => row.status == "complete").map(row => (
      [#text(6.8pt)[#label(row)#linebreak()#row.kind#if row.backend != none { [ · #row.backend] }]],
      [#text(6.3pt)[#row.batch_seconds.baseline.map(x => fixed(x, 3)).join(", ")#linebreak()
        #row.batch_seconds.candidate.map(x => fixed(x, 3)).join(", ")]],
      [#text(6.8pt)[#fixed(row.ratio_range.at(0), 4)–#fixed(row.ratio_range.at(1), 4)]],
      [#text(6.2pt)[B: #usage(row.accepted_usage.baseline)#linebreak()C: #usage(row.accepted_usage.candidate)]],
    )).flatten()))
  v(6pt)
  text(6.5pt, fill: ink2)[Sealed evidence: #raw(data.index_path).
    Index SHA-256: #raw(data.index_sha256). Every displayed observation is checked
    against its retained raw stream by the frozen portable replay.]
}
