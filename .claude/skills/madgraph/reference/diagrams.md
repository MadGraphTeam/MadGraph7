# Feynman diagrams

Use this when the user wants to see, count or check the Feynman diagrams of
a process, or to know which diagrams contribute (s- versus t-channel,
resonances, coupling orders).

## Without an output directory (fast)

```
# diag.mg7
set crash_on_error True --no_save
import model sm
generate p p > t t~ j
display diagrams diag_eps --no_open --merge
display diagrams_text diag_txt --no_open --merge
```

This takes seconds and creates no process directory.
- `display diagrams <dir> --no_open` writes one EPS file per subprocess.
  `--merge` also writes `<dir>/all_diagrams.pdf`, which needs ghostscript
  (`gs`).
- `display diagrams_text <dir> --no_open --merge` writes the same diagrams as
  text, one file per subprocess plus `all_diagrams_text.txt`. **Read this
  one yourself** to count or classify diagrams.
- Always give a directory and `--no_open`. Without `--no_open`, MadGraph tries
  to open a viewer, and `--no_open` without a directory is an error.

## Only the leading coupling order, unless you ask

`generate` keeps only the diagrams of the **lowest** coupling-order
combination: that is what the header `WEIGHTED<=3` means. For u u~ > t t~ g
it gives the 5 pure-QCD diagrams (QCD=3) and silently drops the 8 with a
photon or Z (QCD=1, QED=2). When the user asks for "all diagrams" or "the
SM diagrams", give the orders explicitly and report both counts:

```
generate u u~ > t t~ g QED=2      # 13 diagrams: 5 QCD + 8 photon/Z
```

The model restriction also matters. In the default `sm`, the light quarks
are massless, so there are no Higgs diagrams for them.

## Reading the text format

```
Process: u u~ > t t~ g WEIGHTED<=3 @1
5 diagrams:
1  ((1(-2),2(2)>1(21),id:58),(3(6),4(-6)>3(21),id:60),(1(21),3(21),5(21),id:3)) (QCD=3,QED=0,WEIGHTED=3)
4  ((1(-2),5(21)>1(-2),id:58),(3(6),4(-6)>3(21),id:60),(1(-2),2(2),3(21),id:58)) (QCD=3,QED=0,WEIGHTED=3)
```

- Each `(a(pdg),b(pdg)>c(pdg),id:N)` is a vertex: legs a and b merge into the
  internal line c, with its PDG code. The internal line takes the lowest leg
  number of what it contains. The last group closes the diagram.
- Legs are numbered as in the process (1 and 2 are incoming). The coupling
  orders of the diagram follow at the end.
- **Incoming particles appear with the opposite PDG sign** (everything is
  written as outgoing): the incoming u is `1(-2)`.
- **s- or t-channel**: follow which external legs each internal line
  collects.
  - A line that collects **exactly one** incoming leg (plus final-state legs)
    is t-channel.
  - A line that collects both incoming legs, or only final-state legs, is
    s-channel.
- Diagram 1: u u~ → g* (legs 1+2, s-channel), t t~ → g* (legs 3+4,
  s-channel), and a three-gluon vertex with the emitted gluon 5.
- Diagram 4: the incoming u emits the gluon (`1(-2),5(21)>1(-2)`: legs 1+5,
  a t-channel u), and t t~ → g* (legs 3+4, an s-channel gluon carrying
  m(t t~)).

## Flavour grouping: `Q` and PDG 81

By default the light quarks are grouped into one flavour class. Diagrams show
`Q`/`Qx` (in text: `_quark`, PDG ±81) instead of u, d, s, c, and there is one
subprocess per class. This happens **even when you type the flavour
explicitly** (`generate u u~ > t t~ g`). The file is still named
`diagrams_1_uux_ttxg.txt`, but its header reads `_quark _anti_quark > ...`. For per-flavour diagrams (`u u~ > t t~ g`, ...), add
this before `import model`/`generate`:

```
set apply_flavor_grouping False --no_save
```

Then p p > t t~ j has 13 subprocesses instead of 4. Keep `--no_save`: the
setting would otherwise change all later runs of the user.

## After an `output`

Diagram files are written anyway, per subprocess `SubProcesses/P*/`:

| Output | Files |
|---|---|
| mg7 (`output`) | `diagrams.svg` (all diagrams of the subprocess), `diagrams.json` |
| madevent (`output madevent`) | `matrixN.ps`, `diagrams.html` |
| aMC@NLO (`[QCD]`) | `born.ps` (Born), `matrix_N.ps` (real emissions), `diagrams.html` |

## Looking at them, and showing them

To look at a drawing yourself, convert it to PNG and read the image:

```bash
rsvg-convert -w 900 SubProcesses/P1_gg_ttx/diagrams.svg -o gg_ttx.png
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r150 -sOutputFile=page%d.png diag_eps/all_diagrams.pdf
```

Use about 150 dpi when you need to read particle labels (at 80 dpi `t~`
looks like `t-`). If `rsvg-convert` or `gs` is missing, use the text format and hand the
SVG/PDF files to the user. Give the user the PDF or SVG paths; the
`--merge` PDF is the most convenient single file.

## Counting and selecting diagrams

- The number of diagrams per subprocess is in the text header (`16
  diagrams:`).
- To remove diagrams, use the process syntax and then redraw to check:
  - `/ h` removes the diagrams with an internal h;
  - `$$ z` removes those with an s-channel Z;
  - `QED=0` or `QCD=2` restrict the coupling orders.

  If `$$` or `/` removes **every** diagram of the leading order, MadGraph
  silently moves to the next coupling order instead of failing. For
  example, `u u~ > t t~ g $$ g` returns the 8 photon/Z diagrams. Check the
  `QCD=`/`QED=` of what is left.

  `$ z` (single dollar) is different: it **keeps** the diagrams and only
  removes the on-shell region of the s-channel Z (within `bwcutoff` widths),
  so it changes the cross section but not the drawing. Removing diagrams
  breaks gauge invariance in general; warn the user if they ask for it.
