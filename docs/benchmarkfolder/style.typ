// Visual style adapted from MadGraphTeam/pyamplicol, loop branch:
// docs/performance_reports/loop/benchmarks.typ. Report claims and data are ours.
#let navy = rgb("#203c60")
#let blue = rgb("#356b92")
#let pale = rgb("#edf3f8")
#let grey = luma(95)
#let rule = luma(215)
#let palette = gradient.linear(rgb("#5aae61"), rgb("#d9f0d3"), white, rgb("#fddbc7"), rgb("#d6604d"))
#let ratio-fill(r) = if r == none { pale } else {
  palette.sample(calc.clamp(0.5 + calc.log(r, base: 2) / 8, 0, 1) * 100%)
}
#let report-style(body, date: "") = {
  set document(title: "NinjaRS in MadLoop — integration and benchmarks", author: "MadLoop–NinjaRS development")
  set page(paper: "a4", margin: (x: 1.65cm, top: 1.65cm, bottom: 1.7cm),
    numbering: "1 / 1", header: context {
      if counter(page).get().first() > 1 {
        text(8pt, fill: grey)[MadLoop · NinjaRS integration #h(1fr) #date]
        v(3pt)
        line(length: 100%, stroke: 0.4pt + rule)
      }
    })
  set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.61em)
  set heading(numbering: none)
  show heading.where(level: 1): it => {
    v(2pt)
    block(text(17pt, weight: "bold", fill: navy, it.body))
    v(3pt)
  }
  show heading.where(level: 2): it => {
    v(3pt)
    block(text(11.5pt, weight: "bold", fill: navy, it.body))
  }
  show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
  show link: set text(fill: blue)
  show figure.where(kind: table): set figure.caption(position: top)
  body
}
#let note(body) = block(width: 100%, fill: pale, inset: 10pt, radius: 3pt, text(9.4pt, body))
#let small(body) = text(size: 8.7pt, fill: grey, body)
#let section-label(s) = text(8.4pt, fill: blue, weight: "semibold", tracking: 0.7pt, s)
#let fmt(x, digits: 1) = {
  if x == none { return [—] }
  let parts = str(calc.round(x, digits: digits)).split(".")
  if digits == 0 { return parts.first() }
  let decimal = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.first() + "." + decimal + "0" * calc.max(0, digits - decimal.len())
}
#let value-fmt(x) = {
  if x == none { return [—] }
  let digits = if x >= 1000 { 0 } else if x >= 100 { 1 } else { 2 }
  if x >= 10000 {
    let rest = str(int(calc.round(x)))
    let groups = ()
    while rest.len() > 3 {
      groups.insert(0, rest.slice(rest.len() - 3))
      rest = rest.slice(0, rest.len() - 3)
    }
    groups.insert(0, rest)
    return groups.join(sym.space.thin)
  }
  fmt(x, digits: digits)
}
#let th(body) = text(fill: white, size: 8.1pt, weight: "semibold", body)
#let header(body) = text(font: "DejaVu Sans", size: 8pt, fill: white, weight: "bold", body)
#let tab(columns, left-columns: (0,), ..cells) = {
  set par(justify: false, leading: 0.35em)
  set text(size: 8.7pt)
  table(columns: columns, inset: (x: 5pt, y: 6pt),
    stroke: (x, y) => (top: if y > 1 { 0.4pt + luma(225) } else { none }),
    fill: (x, y) => if y == 0 { navy } else if calc.rem(y, 2) == 0 { rgb("#f5f7f9") } else { none },
    align: (x, y) => if x in left-columns { left + horizon } else { right + horizon },
    ..cells.pos())
}
#let legend = align(center)[
  #box(width: 9cm, height: 0.27cm, fill: palette, radius: 2pt)
  #v(-5pt)
  #box(width: 9cm, text(size: 7.3pt)[×1/16 #h(1fr) ×1/4 #h(1fr) ×1 #h(1fr) ×4 #h(1fr) ×16])
  #v(-3pt)
  #text(size: 8pt, fill: luma(75))[Logarithmic scale · lower is faster · white is parity · red is slower]
]
