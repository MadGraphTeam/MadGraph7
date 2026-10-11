# Plots of distributions

Use this when the user wants a distribution (m_tt, pT, rapidity, ...), with
or without uncertainty bands, or a comparison (LO vs NLO, two setups, a
K-factor per bin).

## Choosing the route

| The user wants | Route |
|---|---|
| LO distributions with scale and PDF bands, quickly | the mg7 `[histograms]` of the run card: `Events/run_XX/plots/*.pdf` |
| NLO fixed-order distributions | the aMC@NLO fixed-order analysis: `Events/run_XX/MADatNLO.HwU` / `.pdf` |
| an overlay with a ratio panel (LO vs NLO, K-factor, two runs) | `madgraph/various/histograms.py` on HwU files, or your own plot from them (section 3) |
| any other observable or selection, or a custom style | your own Python on the events (`events.npy` or LHE) |
| hadron-level (showered) events | Rivet or MadAnalysis5, if installed. Not covered here: ask the user |

**Which LO.** For an LO/NLO comparison or a K-factor, take the LO curve
from the aMC@NLO `lo` run of the same directory. It uses the NLO PDF and
α_s. The mg7 LO uses the LO PDF set and gives a different number: t t̄ at
13.6 TeV is 476 pb against 422 pb. Use one LO consistently across the plots
of one request, and state which.

## 1. mg7 runs: the `[histograms]` section

Every mg7 run fills the histograms listed in `[histograms]` of
`Cards/run_card.toml` from its final events. It draws them as
`Events/run_XX/plots/<name>.pdf`, with the scale band, the PDF band and a
ratio panel, and stores the numbers in `info.json`. With
`set write_hwu True` it also writes `MADatLO.HwU`. That setting persists in
the card, like every `set`.

**Defining histograms.** `set histograms.<name>...` is rejected (`invalid
set command`); only `set histograms off` and `set histograms default` exist.
Edit the section in the card **between `output` and `launch`**. Each
histogram is three lines:

```
[histograms]
t_1-t_2-pair_mass.min = 300.0
t_1-t_2-pair_mass.max = 1500.0
t_1-t_2-pair_mass.bin_count = 24
```

To replace the generated defaults:

```python
import re
p = "PROC/Cards/run_card.toml"
s = open(p).read()
new = """[histograms]
t_1-t_2-pair_mass.min = 300.0
t_1-t_2-pair_mass.max = 1500.0
t_1-t_2-pair_mass.bin_count = 24
t_1-y.min = -3.0
t_1-y.max = 3.0
t_1-y.bin_count = 30
"""
s = re.sub(r"\[histograms\]\n(?:(?!\n\[).)*", new, s, flags=re.S)
open(p, "w").write(s)
```

**Names** follow the cut syntax of `reference/mg7_run_card.md`. The groups
are those of `[multiparticles]` in the generated card (`jet`, `lepton`, `t`,
`w`, ...).
- `<group>_<n>-<obs>` is the n-th hardest of a group, by pT: `t_1-pt`,
  `jet_2-eta`.
- `<a>_1-<b>_1-<obs>` is a two-particle quantity: `t_1-t_2-pair_mass`,
  `t_1-t_2-delta_phi`, `jet_1-jet_2-delta_r`.
- `<a>_1-<b>_1-sum-<obs>` sums the momenta first: `t_1-t_2-sum-pt` is
  pT(tt).
- `<group>-<obs>-sum` sums the observable over the group: `jet-pt-sum` is H_T.
- `sqrt_s` (all outgoing particles) and `weight` (the event weight in units
  of σ) are event-level.
- Observables: `e px py pz pt p_mag phi theta y y_abs eta eta_abs mass
  pair_mass sfos_pair_mass delta_eta delta_phi delta_r sqrt_s`.

**Particle versus antiparticle.** The generated groups merge them (`t =
[6, -6]`, ordered by pT). To separate them, add groups to `[multiparticles]`
(`top = [6]`, `antitop = [-6]`) and book `top_1-y`, `antitop_1-y`. A
difference of two observables (such as Δ|y| = |y_t| − |y_t̄|) cannot be
booked: use section 4.

**Reading the output.**
- The **plots** show **dσ/dx** (the cross section per bin divided by the
  bin width). Under- and overflow are dropped, so the integral is σ inside
  the plotted range. The log or linear axis is chosen automatically.
- **`MADatLO.HwU` holds σ per bin**, like every HwU file (aMC@NLO's too):
  its values sum to σ inside the range. MadGraph7 versions before PR #259
  wrote dσ/dx there. To check a file, compare `sum(central value)` with the
  run's σ.
- The histograms come from the unweighted events, so the statistical error
  per bin scales with the number of events in it. 20k events gave about 1%
  near the m_tt peak and 20-30% in the 1.4 TeV tail. Ask for enough events
  for the tail the user cares about.
- Without matplotlib the run says so and only writes the numbers.

## 2. NLO fixed order: the FO analysis

The analysis is chosen by `FO_ANALYSE` in `Cards/FO_analyse_card.dat`. The
default (`analysis_HwU_template.o`) fills almost nothing. Shipped analyses,
in `<dir>/FixedOrderAnalysis/`: `analysis_HwU_pp_V`, `_pp_lplm`, `_pp_lvl`,
`_pp_h`, `_pp_hjj`, `_pp_ttx`, `_pp_tj`, `_pp_wpz`, `_general`, ...

```bash
perl -pi -e 's/^FO_ANALYSE = .*/FO_ANALYSE = analysis_HwU_pp_ttx.o/' PROC/Cards/FO_analyse_card.dat
```

- Change only that line. Rewriting the whole card drops `FO_EXTRAPATHS` and
  the run dies with `KeyError: 'fo_extrapaths'`.
- **The analysis must match the process.** Each one assumes a final state:
  `analysis_HwU_pp_V` stops with "error #1 in analysis_fill" when run on
  p p > z j, and every job fails. Read its header comment first.
- **Their binning is fixed in the Fortran.** For example, `analysis_HwU_pp_ttx`
  books `tt inv m` with 60 bins from 300 to 1000 GeV, so about 2.5% of σ
  (above 1 TeV) is never shown. List the ranges with
  `grep -n HwU_book <dir>/FixedOrderAnalysis/<analysis>.f`. For another range
  or another observable, copy the analysis (or `analysis_HwU_template.f`)
  into `FixedOrderAnalysis/` under a new name, change its
  `HwU_book(id, title, nbins, xmin, xmax)` lines and the filling in
  `analysis_fill`, and set `FO_ANALYSE = <new name>.o`.
- The shipped analyses book each histogram twice, without and with their
  own cuts (titles ending in `cuts`).
- **`MADatNLO.HwU` holds σ per bin, NOT dσ/dx** (as every HwU file does).
  Its values sum to σ inside the range. Divide by the bin width for a dσ/dx
  plot.
- **Precision**: a bin needs far more than the total. At `req_acc_FO 0.002`
  one t t̄ run gave 0.1% on σ but 10-20% per 35 GeV bin above 900 GeV, with
  spikes and even a negative bin. Run several seeds and combine them
  (section 3), merge bins, or both. For t t̄ on 18 cores (after the first
  compilation), each fixed-order run takes 30 s to 1.5 min.
- Output: `Events/run_XX/MADatNLO.HwU`, plus `MADatNLO.pdf` and the scripts
  that drew it.

## 3. Overlays, ratios and seed combination from HwU files

**Quick look with `histograms.py`:**

```bash
python3 <MG7>/madgraph/various/histograms.py \
    PROC/Events/run_05_LO/MADatNLO.HwU PROC/Events/run_04/MADatNLO.HwU \
    --out=plots/mtt --matplotlib --no_open \
    --assign_types=LO,NLO --titles="tt inv m" --rebin=4
cd plots && python3 mtt.py          # writes plots/mtt.pdf
```

- The scale bands are kept and drawn. MadGraph7 versions before PR #259
  silently dropped aMC@NLO's scale columns (`dyn=-1 muR= ... muF= ...`).
  There, add `--keep_all_weights`, and check that the output header still
  lists them.
- `--out` takes a path **without extension**; with one it fails.
- **The ratio panel is relative to the first file.** Put the reference
  first: LO then NLO gives a "K-factor" panel.
- `--titles` keeps the histograms whose title **contains** one of the
  comma-separated strings: "tt inv m" also matches "tt inv m cuts" (a second
  page).
- Histograms are matched **by title and binning**: compare files of the
  same analysis (the `lo` and `nlo` runs of one directory, or two mg7 runs
  with the same `[histograms]`). An mg7 HwU and an aMC@NLO HwU have the
  same normalisation, but their titles and binnings differ (`t_1-t_2-pair_mass`
  against `tt inv m`), so they do not pair. Read both yourself (below).
- The result is a quick look, not a figure to hand over. The x label is
  "bin", there is an extra "rel. unc." panel, and empty bins below a
  threshold draw a ratio of 0. For a deliverable, read the HwU files and
  draw your own (below).
- `--average` is an unweighted mean of identical histograms. For seeds of
  different precision, combine them yourself, weighting by 1/error².

**Reading HwU files yourself** (works for both formats):

```python
import numpy as np

def read_hwu(path):
    """{title: {"lo", "hi", "central value", "dy", <weight column>: array}}"""
    text = open(path).read()
    columns = [c.strip() for c in text.split("##&", 1)[1].split("\n", 1)[0].split("&")]
    hists = {}
    for block in text.split("<histogram>")[1:]:
        head, body = block.split("\n", 1)
        title = head.split('"')[1].split("|")[0].strip()
        rows = np.array([l.split() for l in body.split("<\\histogram>")[0].strip().splitlines()], float)
        h = {"lo": rows[:, 0], "hi": rows[:, 1]}
        h.update({name: rows[:, i] for i, name in enumerate(columns) if 1 < i < rows.shape[1]})
        hists[title] = h
    return hists

h = read_hwu("PROC/Events/run_04/MADatNLO.HwU")["tt inv m"]
width = h["hi"] - h["lo"]
dsdx = h["central value"] / width                    # aMC@NLO: sigma per bin -> dsigma/dx
scale = np.array([h[k] for k in h if "muR" in k])    # 9 columns (central included) for aMC@NLO
band_lo, band_hi = scale.min(0) / width, scale.max(0) / width
```

- The scale columns contain `muR`: 9 of them, central included, in both
  aMC@NLO and (since PR #259) mg7 files.
- Both are σ per bin: divide by the width for dσ/dx.
- **Seeds**: weight each run's bins by 1/dy²
  (`mean = Σ v/dy² / Σ 1/dy²`, `error = (Σ 1/dy²)^-1/2`), or simply weight
  whole runs by 1/(total MC error)². Average the scale columns the same way.
- **Ratio panel**: divide the NLO central value and band by the LO central
  value. Skip bins where the LO value is 0.

## 4. Your own observable, from the events

**mg7 `events.npy`** (default `output_format = "lhe_npy"`) is a numpy
structured array with one row per event:
- `weight`, whose **mean** is σ in pb;
- `rwgt_<k>` for the scale/PDF variations;
- per particle `part<i>_pdg_id`, `_status_code` (-1 incoming, 1 final,
  2 intermediate), `_px`, `_py`, `_pz`, `_energy`, `_mass`, ...

**Which `rwgt_<k>` is which**: `events.weights.json` in the run directory
has `scale.weight_ids` (the 8 non-central muR × muF points) and
`pdf[0].weight_ids` (the PDF members, the first being member 0, the
nominal). The `rwgt_<k>` are full weights: the mean of `rwgt_<k>` is that
variation's σ. **A scale envelope must include the central `weight`** as
well as the 8 variations.

**Select particles by PDG code and status, never by position**: the order
can differ between subprocesses. With `output_format = "compact_npy"`,
convert first with `<PROC>/bin/npy_to_lhe <run_name>` and use the LHE route.

```python
import json, numpy as np
run = "PROC/Events/run_02"
ev = np.load(run + "/events.npy")
N = len(ev)

def momentum(ev, pdg, status=1):
    """(E, px, py, pz) of the first particle with this PDG code and status."""
    out = [np.zeros(len(ev)) for _ in range(4)]
    found = np.zeros(len(ev), bool)
    i = 1
    while "part%d_pdg_id" % i in ev.dtype.names:
        sel = (ev["part%d_pdg_id" % i] == pdg) & (ev["part%d_status_code" % i] == status) & ~found
        for arr, key in zip(out, ("energy", "px", "py", "pz")):
            arr[sel] = ev["part%d_%s" % (i, key)][sel]
        found |= sel
        i += 1
    assert found.all(), "PDG %d (status %d) missing in %d events" % (pdg, status, (~found).sum())
    return out

def rapidity(E, px, py, pz):
    return 0.5 * np.log((E + pz) / (E - pz))

E, px, py, pz = (a + b for a, b in zip(momentum(ev, 6), momentum(ev, -6)))
mtt = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0))
edges = np.linspace(300, 1500, 25)
width = np.diff(edges)

def dsdx(weights):
    return np.histogram(mtt, edges, weights=weights / N)[0] / width      # pb / GeV

central = dsdx(ev["weight"])
error = np.sqrt(np.histogram(mtt, edges, weights=(ev["weight"] / N)**2)[0]) / width
ids = json.load(open(run + "/events.weights.json"))["scale"]["weight_ids"]
variations = np.array([central] + [dsdx(ev["rwgt_%d" % k]) for k in ids])
band_lo, band_hi = variations.min(0), variations.max(0)
```

The central curve reproduced MadGraph's own `t_1-t_2-pair_mass` histogram
of the same run to 3e-8 (the print precision of the HwU file).

**LHE files** (madevent, aMC@NLO events, MadSpin output) are read with
MadGraph's own parser:

```python
import sys, numpy as np
sys.path.insert(0, "<MG7>")
from madgraph.various import lhe_parser
mll, w = [], []
for event in lhe_parser.EventFile("PROC/Events/run_01/events.lhe.gz"):
    lep = [p for p in event if p.status == 1 and abs(p.pid) == 11]
    mll.append((lhe_parser.FourMomentum(lep[0]) + lhe_parser.FourMomentum(lep[1])).mass)
    w.append(event.wgt)
mll, w = np.array(mll), np.array(w)
```

- With `event_norm = average` (the default; check the banner), the **mean**
  weight is σ: histogram with `weights = w / len(w)`, as above. With
  `event_norm = sum`, the weights already add up to σ.
- **NLO events have negative weights. Keep the signs**, never `abs()`.
  The error per bin is still sqrt(Σ w²).

**Drawing**: draw a step histogram of dσ/dx with error bars. Use a log y
axis for steep spectra (pT, masses) and a linear one for angles and
rapidities. Label the axis, e.g. `dσ/dm_tt [pb/GeV]`. Render the PDF to PNG
and look at it before handing it over:

```bash
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r100 -sOutputFile=check.png plot.pdf
```

## Reporting a plot

Say what is on it:
- the normalisation: dσ/dx, so the integral is σ in the plotted range;
- the order (LO/NLO), and for an LO curve which LO (see "Which LO");
- what the bands are (9-point scale envelope, PDF replicas);
- the number of events, or the seeds and precision, behind the statistical
  errors;
- the plotted range, and the fraction of σ outside it if it is
  significant;
- for a ratio, which curve is the reference.
