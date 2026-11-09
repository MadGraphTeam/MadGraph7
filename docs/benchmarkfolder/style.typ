// Presentation matched to pyAmpliCol benchmarks.typ, revision
// 7266c1600f6ec4e39be39986e6503a515c380c52 (style provenance in README).
#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let warm = rgb("#fdf3e7")
#let ink2 = luma(90)
#let hair = 0.4pt + luma(222)
#let sans = "DejaVu Sans"
#let fast-green = rgb("#1a9e77")
#let slow-red = rgb("#d73027")
#let palette = gradient.linear(fast-green, rgb("#c4e6d9"), white, rgb("#fbd0c3"), slow-red)
#let ratio-fill(r) = if r == none or r <= 0 { none } else {
  palette.sample(calc.clamp(0.5 + calc.log(r, base: 2) / 8, 0, 1) * 100%)
}
#let stage-colours = (rgb("#2a78d6"), rgb("#eb6834"), rgb("#1baf7a"), rgb("#eda100"), rgb("#e87ba4"), rgb("#6250d6"), luma(170))
#let report-style(body, date: "", revision: "") = {
  set document(title: "MadLoop with TIRust — numerical validation and benchmarks",
    author: "MadLoop–TIRust development", date: none)
  set page(paper: "a4", margin: (x: 1.6cm, top: 1.7cm, bottom: 1.6cm), numbering: "1 / 1",
    header: context if counter(page).get().first() > 1 {
      set text(8pt, fill: luma(110))
      [MadLoop with TIRust #h(1fr) #date · revision #revision]
    })
  set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    v(0.45em); block(text(14pt, fill: navy, it)); v(0.1em)
  }
  show heading.where(level: 2): it => block(above: 0.9em, below: 0.55em,
    text(11pt, fill: navy, it))
  show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
  show link: set text(fill: rgb("#1f5fa8"))
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set block(breakable: true)
  show figure.caption: set text(9pt)
  show figure.caption: set par(justify: true)
  show figure.caption: set align(left)
  set figure(gap: 0.55em)
  body
}
#let table-style(body) = {
  set align(left)
  set text(7.6pt, font: sans)
  set par(justify: false, leading: 0.4em)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
}
#let table-stroke(x, y) = (top: if y <= 1 { none } else { hair })
#let table-fill(x, y) = if y == 0 { navy } else { none }
#let note(body, fill: pale) = block(width: 100%, fill: fill, breakable: true,
  inset: 9pt, radius: 3pt, text(9pt, body))
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s.split(".").first() }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * calc.max(0, d - frac.len())
}
#let number(x) = if x == none { [—] } else {
  let a = calc.abs(x)
  fixed(x, if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else { 3 })
}
#let ratio-text(r) = if r == none { [—] } else { [×#fixed(r, 3)] }
#let status-ink(s) = if s in ("pass", "passed") { rgb("#0f7356") } else if s in ("fail", "failed") { rgb("#b2182b") } else { rgb("#8a4b08") }
