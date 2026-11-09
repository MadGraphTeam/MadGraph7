#import "style.typ": *

#let show-intermediate(data) = {
  for (index, section) in data.sections.enumerate() {
    pagebreak()
    heading(level: 1)[#section.title]
    note(fill: warm)[#section.note]
    v(7pt)
    table-style(table(columns: (1.1fr, .6fr, 3.1fr), inset: (x: 5pt, y: 8pt),
      stroke: table-stroke, fill: table-fill,
      table.header([Check / experiment], [Status], [Observed result and limits]),
      ..section.claims.map(q => (
        [#text(weight: "bold")[#q.title]#linebreak()#text(6.7pt, fill: ink2)[#q.scope]],
        [#text(fill: status-ink(q.status), weight: "bold")[#q.status]],
        [#q.text#v(4pt)#text(6.2pt, fill: ink2)[#raw(q.evidence.first())]],
      )).flatten()))
    if index == 0 {
      v(9pt)
      heading(level: 2)[Diagnostic improvement evidence]
      [Seven-pair shared-host comparisons motivated the implemented changes.
      These ratios compare the variants named below; they do not qualify the
      final candidate, and they do not replace the original failed n1 gate.]
      v(5pt)
      table-style(table(columns: (2.4fr, .65fr, 1.2fr), inset: 5pt,
        stroke: table-stroke, fill: table-fill,
        table.header([Private full-amplitude comparison], [Ratio], [95% paired interval]),
        ..data.diagnostic_timings.map(row => (
          [#row.title],
          table.cell(fill: ratio-fill(row.statistics.geometric_mean_ratio),
            [#fixed(row.statistics.geometric_mean_ratio, 5)]),
          [#fixed(row.statistics.lower_95, 5)–#fixed(row.statistics.upper_95, 5)],
        )).flatten()))
      v(5pt)
      text(8pt, fill: ink2)[The split-workspace experiment still exceeded the
      baseline limit. Different diagnostic DSOs and compiler/host contexts are
      retained with each experiment. The confidence intervals do not remove
      the shared-host limitation.]
    }
  }
  v(8pt)
  note(fill: warm)[*Final campaign: pending.* The installed DSO is
    #raw(data.installer_identity.library_sha256.slice(0, 16)); its installer source
    is #raw(data.installer_identity.source_revision.slice(0, 12)). A new sealed
    campaign must bind the final exporter, native library, compiler settings,
    model/cards, points and policy. Quiet paired timing, all five reducers,
    existing-backend regressions and rescued-cohort measurements remain required.
    The assigned integration goal is not complete.]
}
