// Canonical destination: docs/benchmarkfolder/current-source-supplement.typ.
#import "style.typ": *

#let source-check(value, target: false) = if value == none { [—] } else {
  let label = if value { "pass" } else if target { "miss" } else { "fail" }
  text(fill: status-ink(label), weight: "bold", label)
}

#let source-sci(x) = if x == 0 { [0] } else {
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  [#fixed(x / calc.pow(10, e), 3) × 10#super(str(e))]
}

#let source-link(path) = "https://github.com/MadGraphTeam/MadGraph7/blob/madloop_tirust/docs/benchmarkfolder/" + path

#let show-current-source-raw(s) = {
  let r = s.raw_stability
  pagebreak()
  heading(level: 1)[Raw stability pilot · #raw(s.current.revision.slice(0, 12))]
  text(8pt, fill: ink2)[Current-source QCD n3 · 100 cut-qualified physical points ·
    card-selected reducers 1/6/7/8/9]
  v(4pt)
  text(7.5pt, table-style(table(columns: (.55fr, .8fr, .8fr, .8fr, .85fr, 1fr),
    inset: 3pt, stroke: table-stroke, fill: table-fill,
    table.header([Tool], [Execution], [Available / 100], [Unavailable], [Rejected probes], [Tail at 1e-3]),
    ..r.summary.rows.map(row => (
      [#row.backend], [#row.state], [#row.available_points], [#row.unavailable_points],
      [#row.primary_outcome_counts.at("rejected", default: 0)],
      [#fixed(row.tail_at_1e3.lower * 100, 0)% – #fixed(row.tail_at_1e3.upper * 100, 0)%],
    )).flatten(),
  )))
  v(4pt)
  image(r.figure_path, width: 100%, height: 125mm, fit: "contain")
  v(3pt)
  text(8pt)[#r.figure_caption]
  v(4pt)
  text(8pt)[The same three-probe finite/Born relative-spread diagnostic covers
    #r.primary_calls primary and #r.bootstrap_calls separate bootstrap calls.
    TIRust’s five unavailable points (25, 34, 45, 70, 84) retain 15 rejected probes
    and 27 failed active-group observations. Five subsequent healthy calls clear
    incoming software latches under the source-qualified entry reset; this does
    not qualify the failed calls. Exact legacy annotations remain retained.
    Hardware IEEE state was not observed. The 1e-3 TIRust fraction is unknown
    within 0–5%, and its Ninja ratio is masked.]
  v(3pt)
  text(7.5pt, fill: ink2)[#link(source-link(r.data_index_path))[Raw data and all outcomes] ·
    #link(source-link(r.figure_index_path))[Figure and statistical bounds].
    This selected study adds no ordinary-case, true-error or performance acceptance.]
}

#let show-current-source(s) = {
  set page(header: context {
    set text(8pt, fill: luma(110))
    [MadLoop with TIRust #h(1fr) Current source #raw(s.current.revision.slice(0, 12))]
  })
  pagebreak(weak: true)
  heading(level: 1)[Current-source progress · #raw(s.current.revision.slice(0, 12))]
  text(8pt, fill: ink2)[Campaign #raw(s.current.campaign)]
  v(6pt)
  note(fill: warm)[The preceding report and plots retain campaign
    #raw(s.historical.campaign), source #raw(s.historical.source_revision.slice(0, 12)),
    and published PDF commit #raw(s.historical.pdf_commit.slice(0, 12)).
    This supplement records the separately registered source
    #raw(s.current.revision.slice(0, 12)). Its evidence does not change the
    historical measurements or their source identities.]
  v(5pt)
  [The generated Fortran now resets the DP/QP software reduction-failure latches
    at each independent amplitude entry and checks context generation before
    clearing TIRust events. The source-specific records below qualify their
    stated numerical scope.]
  v(7pt)
  table-style(table(columns: (1fr, 3.2fr), inset: 4pt,
    stroke: table-stroke, fill: table-fill,
    table.header([Current source binding], [Retained identity]),
    [Exporter inventory], [#raw(s.current.source_sha256)],
    [Numerical identity], [#raw(s.current.numerical_identity_sha256)],
    [Native DSO], [#raw(s.current.native_dso_sha256)],
  ))
  v(8pt)
  heading(level: 2)[Frozen ordinary numerical coverage]
  [#s.numerical.closed_cases of #s.numerical.total_cases cases have closed
    records: #s.numerical.closed_lanes of #s.numerical.total_lanes lanes,
    with #s.numerical.points_per_lane saved points per lane.
    #s.numerical.qualified_cases cases pass the 3e-6 regression criterion.
    The remaining #(s.numerical.total_cases - s.numerical.closed_cases) cases
    and #(s.numerical.total_lanes - s.numerical.closed_lanes) lanes are pending
    in this frozen selection.]
  v(6pt)
  figure(table-style(table(columns: (1.5fr, .65fr, .45fr, .6fr, .6fr, .85fr),
    inset: (x: 4pt, y: 5pt), stroke: table-stroke, fill: table-fill,
    table.header([Process], [Status], [Lanes], [3e-6], [1e-8], [Accepted DP / QP]),
    ..s.numerical.cases.map(row => (
      [#text(7.1pt)[#raw(row.process)]#linebreak()
       #text(6.4pt, fill: ink2)[#row.role]],
      [#text(fill: status-ink(if row.status == "qualified" { "pass" } else { row.status }), weight: "bold", row.status)],
      [#row.closed_lanes / 6],
      [#source-check(row.regression_pass)],
      [#source-check(row.ordinary_target_pass, target: true)],
      [#if row.closed_lanes > 0 { [#row.accepted_dp / #row.accepted_qp] } else { [—] }],
    )).flatten(),
  )), caption: [Source #raw(s.current.revision.slice(0, 12)): retained ordinary
    numerical records. Regression and ordinary-target outcomes are shown separately.])
  v(5pt)
  text(8pt, fill: ink2)[Pending denotes absence of selected closed evidence;
    it makes no claim about active, later or unselected work. Accepted DP/QP totals
    cover baseline CutTools and candidate reducers 1/6/7/8/9. They describe final
    accepted outcomes and do not enumerate internal retries. Software/hardware
    exception flags were not recorded. These records supply
    no paired runtime acceptance.]
  if s.numerical.target_misses.len() > 0 {
    v(6pt)
    table-style(table(columns: (1.5fr, .4fr, .55fr, 1.1fr, .6fr, 1.1fr), inset: 4pt,
      stroke: table-stroke, fill: table-fill,
      table.header([Target-miss case], [Tool], [Scalar misses], [B / F / 1 / 2], [Saved points], [Max. finite residual]),
      ..s.numerical.target_misses.map(row => (
        [#raw(row.case)], [#row.backend], [#row.scalar_misses],
        [#row.columns.at("0") / #row.columns.at("1") / #row.columns.at("2") / #row.columns.at("3")],
        [#row.distinct_points],
        [#source-sci(row.maximum_finite_symmetric_residual)],
      )).flatten(),
    ))
    v(3pt)
    text(8pt, fill: ink2)[B / F / 1 / 2 gives the Born, finite, single-pole and
      double-pole miss counts. Their sum equals the scalar count.
      The ordinary-target counts retain both order rows at each
      saved point; they are not counts of distinct physical failures. Residuals
      compare against the source-matched baseline CutTools result and supply no
      independent truth estimate.
      #if s.numerical.target_misses.any(row => row.case == "qcd_n3_uux_zgg") {
        [Born and both pole columns pass 1e-8 in the retained QCD n3 miss case;
        its misses are finite only.]
      }]
  }
  for note-text in s.numerical.build_scope_notes {
    v(5pt)
    text(8pt, fill: ink2, note-text)
  }

  v(8pt)
  heading(level: 1)[Cache-guard scope · #raw(s.current.revision.slice(0, 12))]
  note(fill: warm)[Scope: #s.cache.scope. The following paired observations
    compare the guarded candidate with the original guard-off code. This
    comparison excludes the entry-latch reset. Wrapper rows measure lifecycle
    operations alone; full-amplitude rows cover one shared-host QCD n1 process.
    These measurements do not measure the combined source changes or establish
    general performance, campaign acceptance or precise local performance.]
  v(8pt)
  figure(table-style(table(columns: (2.5fr, .8fr, 1.4fr),
    inset: (x: 5pt, y: 8pt), stroke: table-stroke, fill: table-fill,
    table.header([Guard-only scope], [Guarded / original], [95% interval]),
    ..s.cache.rows.map(row => (
      [#row.label],
      table.cell(fill: ratio-fill(row.ratio), [#fixed(row.ratio, 6)]),
      [#fixed(row.lower_95, 6) – #fixed(row.upper_95, 6)],
    )).flatten(),
  )), caption: [All retained guard-only scopes, with #s.cache.pairs pairs per
    scope. Ratios above 1 indicate overhead in that scope. The 95% paired
    intervals describe uncertainty.])
  v(6pt)
  [Retained numerical-byte parity: #source-check(s.cache.numerical_byte_parity).
    Active-wrapper overhead is approximately
    #fixed((s.cache.rows.at(1).ratio - 1) * 100, 2)%.
    Cache-operation timing and byte parity do not establish independent physical
    accuracy, complete campaign coverage or precise local performance acceptance.]
  v(10pt)
  heading(level: 2)[Fresh local qualification · #raw(s.current.revision.slice(0, 12))]
  [All #s.local.total_lanes fresh local numerical lanes remain
    #text(fill: status-ink(s.local.status), weight: "bold", s.local.status).
    The closed records above and the guard-only measurements are separate from
    this local prerequisite.]
  v(6pt)
  [Follow #link(source-link(s.local.handoff_path))[Hua-Sheng’s current-source local timing handoff]
    for a fresh native installation and receipt, local preparation, full saved-point
    numerical replay, paired timing and a separate local report.]
  v(5pt)
  [The #link(source-link(s.local.precision_todo_path))[full-stream precision TODO]
    retains the remaining precision and reporting work.]
  v(9pt)
  note(fill: warm)[Overall delivery:
    #text(fill: status-ink(if s.delivery_complete { "pass" } else { "incomplete" }),
      weight: "bold", if s.delivery_complete { "pass" } else { "incomplete" }).
    Precise local performance acceptance remains deferred to Hua-Sheng.]
  if s.at("raw_stability", default: none) != none {
    show-current-source-raw(s)
  }
}
