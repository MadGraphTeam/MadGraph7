// Style adapted from NinjaRS ca1009e; all data/claims below are ours.
#let data=json("report-data.json")
#let two-tool-style(body)={
set document(title: "MadLoop: Ninja and NinjaRS", author: "MadLoop–NinjaRS development")
set page(
  paper: "a4",
  margin: (x: 1.6cm, y: 1.8cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(110))
      MadLoop Ninja / NinjaRS #h(1fr) #data.date
    ]
  },
)
set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
show math.equation: set text(font: "Libertinus Math")
set par(justify: true, leading: 0.62em)
set heading(numbering: "1.1")
show heading.where(level: 1): it => {
  v(0.6em)
  block(text(14pt, fill: rgb("#1f3b63"), it))
  v(0.2em)
}
show heading.where(level: 2): it => block(text(11.5pt, fill: rgb("#1f3b63"), it))
show raw: set text(font: "Libertinus Mono", size: 8.5pt)
show link: set text(fill: rgb("#1f5fa8"))
show figure.where(kind: table): set block(breakable: true)
show figure.where(kind: table): set figure.caption(position: top)


body
}
// ---------------------------------------------------------------- helpers --

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let is-num(x) = type(x) == int or type(x) == float

// Fixed-decimal formatting (keeps trailing zeros: 0.30, 1.05).
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}

// Significant-figure formatting suited to timings.
#let fmt(x) = {
  if not is-num(x) { return [—] }
  let a = calc.abs(x)
  let d = if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else { 3 }
  let s = fixed(x, d)
  if a >= 10000 {
    // thin-space thousands separator
    let int-part = str(int(calc.round(x)))
    let groups = ()
    let rest = int-part
    while rest.len() > 3 {
      groups.insert(0, rest.slice(rest.len() - 3))
      rest = rest.slice(0, rest.len() - 3)
    }
    groups.insert(0, rest)
    s = groups.join(sym.space.thin)
  }
  s
}

#let ratio(a, b) = if is-num(a) and is-num(b) and b != 0 { a / b } else { none }

// Diverging palette: green (NinjaRS faster) — white (parity) — red (slower).
#let palette = gradient.linear(
  rgb("#5aae61"), rgb("#d9f0d3"), rgb("#ffffff"), rgb("#fddbc7"), rgb("#d6604d"),
)
#let ratio-fill(r) = {
  if r == none { return none }
  // log2 scale: ×1/4 → 0 %, ×1 → 50 %, ×4 → 100 %
  let t = calc.clamp(0.5 + calc.log(r, base: 2) / 4, 0, 1)
  palette.sample(t * 100%)
}
#let ratio-cell(r) = table.cell(
  fill: ratio-fill(r),
  if r == none { [—] } else { text(weight: "semibold")[×#fmt(r)] },
)
