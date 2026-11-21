// Presentation of the pyAmpliCol one-loop report
// (pyamplicol/docs/performance_reports/loop/benchmarks.typ, revision
// 7266c1600f6ec4e39be39986e6503a515c380c52 of MadGraphTeam/pyamplicol, as in
// /shared/external/pyLoopAmpliCol): page, fonts, navy headings, number
// formatters, the RdBu ratio palette, table styling, notes and flags.

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let warm = rgb("#fdf3e7")
#let ink2 = luma(90)
#let hair = 0.4pt + luma(222)
#let sans = "DejaVu Sans"

#let report-style(body, title: "", short: "", date: "", revision: "") = {
  set document(title: title, author: "MadLoop–onia development", date: none)
  set page(paper: "a4", margin: (x: 1.6cm, top: 1.7cm, bottom: 1.6cm), numbering: "1 / 1",
    header: context if counter(page).get().first() > 1 {
      set text(8pt, fill: luma(110))
      [#short #h(1fr) #date · revision #revision]
    })
  set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => { v(0.45em); block(text(14pt, fill: navy, it)); v(0.1em) }
  show heading.where(level: 2): it => block(above: 0.9em, below: 0.55em, text(11pt, fill: navy, it))
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

// ---------------------------------------------------------------- numbers --
#let is-num(x) = type(x) == int or type(x) == float
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s.split(".").first() }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * calc.max(0, d - frac.len())
}
#let group(n) = {
  let rest = str(n)
  let out = ()
  while rest.len() > 3 { out.insert(0, rest.slice(rest.len() - 3)); rest = rest.slice(0, rest.len() - 3) }
  out.insert(0, rest)
  out.join(sym.space.thin)
}
// Three significant figures, thin-space thousands.
#let sig(x) = {
  if not is-num(x) { return [—] }
  let a = calc.abs(x)
  if a == 0 { return "0" }
  if a >= 1000 {
    let e = calc.floor(calc.log(a, base: 10)) - 2
    return group(int(calc.round(x / calc.pow(10.0, e)) * calc.pow(10, e)))
  }
  let d = if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else if a >= 0.1 { 3 } else if a >= 0.01 { 4 } else { 5 }
  fixed(x, d)
}
// Scientific notation, d decimals in the mantissa, Unicode superscripts.
#let sup-digits = ("0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "−": "⁻")
#let sci(x, d: 1) = {
  if not is-num(x) { return [—] }
  if x == 0 { return [0] }
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  let mant = x / calc.pow(10.0, e)
  if calc.abs(calc.round(mant, digits: d)) >= 10 { mant = mant / 10; e += 1 }
  let m = fixed(mant, d)
  if m.starts-with("-") { m = "−" + m.slice(1) }
  [#m×10#str(e).clusters().map(c => sup-digits.at(c)).join()]
}
// A value with n decimals and a true minus sign.
#let dec(x, n) = if not is-num(x) [—] else {
  let s = fixed(x, n)
  if s.starts-with("-") { "−" + s.slice(1) } else { s }
}
// Milliseconds with the precision of the measurement.
#let ms2(x) = if x >= 1000 { group(int(calc.round(x))) } else if x >= 100 { fixed(x, 0) } else if x >= 10 { fixed(x, 1) } else { fixed(x, 2) }
#let mult(r) = [×#fixed(r, 2)]

// --------------------------------------------------------- colour scales --
// Diverging ColorBrewer RdBu (colour-blind safe): blue faster, red slower.
#let palette = gradient.linear(rgb("#4393c3"), rgb("#d1e5f0"), white, rgb("#fddbc7"), rgb("#d6604d"))
#let ratio-fill(r) = if r == none or r <= 0 { none } else { palette.sample(calc.clamp(0.5 + calc.log(r, base: 2) / 8, 0, 1) * 100%) }
#let legend(caption: [time ÷ baseline, log scale, clamped]) = align(center, {
  set text(7.2pt, font: sans)
  box(width: 10cm, {
    box(width: 10cm, height: 0.28cm, fill: palette, radius: 2pt)
    v(-0.45em)
    [×1/16 #h(1fr) ×1/4 #h(1fr) ×1 #h(1fr) ×4 #h(1fr) ×16]
    v(-0.55em)
    text(fill: ink2)[faster #h(1fr) #caption #h(1fr) slower]
  })
})
// Accuracy cells: green-white for agreement, red beyond the target.
#let good = rgb("#d9efe1")
#let bad = rgb("#f8d4cc")
#let acc-fill(x, target) = if not is-num(x) { none } else if x <= target { good } else { bad }

// ----------------------------------------------------------------- tables --
#let tstyle(body) = {
  set text(7.6pt, font: sans)
  set par(justify: false, leading: 0.4em)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
}
#let tstroke(x, y) = (top: if y <= 1 { none } else { hair })
#let thead-fill(x, y) = if y == 0 { navy } else { none }
#let dim(body) = text(fill: ink2, body)
#let tiny(body) = text(6.6pt, fill: ink2, body)
#let box-note(fill: pale, stroke: none, breakable: true, body) = block(width: 100%, fill: fill, stroke: stroke, breakable: breakable, inset: 9pt, radius: 3pt, text(9pt, body))
#let flag(body) = box(fill: warm, inset: (x: 3pt, y: 1.5pt), radius: 2pt, text(6.6pt, weight: "bold", fill: rgb("#8a4b08"), body))
#let okflag(body) = box(fill: good, inset: (x: 3pt, y: 1.5pt), radius: 2pt, text(6.6pt, weight: "bold", fill: rgb("#1b5e3b"), body))
