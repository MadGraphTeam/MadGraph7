# Handover to the MG7 paper, lepton-collider section

Figure and draft text live next to this file:

- `beamstrahlung_fccee.pdf` — the validation figure
- `runsetup.tex` — run setup and panel description, written for JHEP
- `mg7plot.py`, `dump.cc` — how to regenerate both

The figure follows `MG7_paper/plots/lepton/plots.py`: usetex serif 14,
`7*0.75` wide, `lw=1.2`, frameless line-legends via `update_legend`, and the
pgf backend because plain `usetex` drops the math minus in PDF output.

## 1. Replace the paragraph saying showering is not possible

It no longer is. What the code now does:

- MadGraph samples the splitting `x = x_BS * x_ISR` event by event. The
  dressed-lepton density is a Mellin convolution of a CIRCE-style
  beamstrahlung spectrum with the ISR structure function, so the split is
  exact rather than modelled.
- It records the result in the event file as a new per-event LHEF element,
  `<beamstrahlung> x_BS1 x_BS2 </beamstrahlung>`.
- We ship a dedicated Pythia8 main that reads those fractions and degrades
  each beam to `x_BS*p` before the event is processed, so the space-like
  shower backward-evolves `x_ISR` alone. Without it the shower attributes
  the whole of `1-x` to ISR off a full-energy beam and invents radiation
  that beamstrahlung never produced.

Two Pythia facts, if a footnote is wanted: `Beams:allowVariableEnergy` is
silently inert with Les Houches input (only read for `frameType<4`), so the
degradation goes through `Beams:allowMomentumSpread` with a custom
`BeamShape`; and Pythia never boosts an incoming Les Houches event inward,
so the main also boosts each event into its post-beamstrahlung CM frame,
without which Pythia rejects most events as having `x>1`.

## 2. Include the figure

Three panels, `e-e+ -> mu-mu+`, FCC-ee 240 and 365 GeV plus ILC 500 GeV as a
linear-collider reference. Colour is the machine, line style the beam
treatment. Top: forward-photon energy the shower builds, beams nominal vs
degraded. Middle: the beamstrahlung spectra. Bottom:
`sqrt(s')/sqrt(s) = sqrt(x_BS1 x_BS2)`.

Citation keys in `runsetup.tex` are placeholders (`Frixione:2021zdp`,
`Bierlich:2022pfr`) — check against `biblio.bib`.

## 3. State that beamstrahlung is irrelevant for FCC-ee, with a reference

Our numbers: 5.9% (240 GeV) and 6.0% (365 GeV) of beams radiate at all,
mean relative energy loss 1.6e-5 per beam, fewer than 0.2% of events below
`sqrt(s')/sqrt(s) = 0.999`. At ILC 500 the same quantities are 23.1%,
7.5e-3 and 28%.

A reference is still needed. I sanity-checked against an average loss of
about 1.42 MeV per crossing quoted at the WW threshold, same order of
magnitude, but that is a different energy and should not be cited as-is —
the FCC-ee CDR or the overview 2203.08310 would be the place to look.

**Careful:** the average energy *loss* and the beamstrahlung-induced
centre-of-mass energy *spread* (usually quoted as ~0.1% for FCC-ee) are
different quantities differing by a large factor, and are easy to conflate.

## Two caveats that must not be dropped

- **Do not write that the shipped ILC spectra disagree with the ILC
  design.** I believed that at one point and was wrong twice: the
  often-quoted 4.5% is the 1 TeV row (500 GeV is 1.9%), and `delta_BS` is a
  fractional RMS whereas I had computed a mean. `ilc500ll` implies mean
  0.89% and RMS 3.14% per beam, so the design value sits between them, and
  the implied 500/250 ratio (1.85) matches the design one (~2). The ILC
  fits are sound.
- **CLIC 3 TeV is not validated.** `clic3000ll` implies mean 2.69% / RMS
  11.5% against a design figure of ~30%, a gap that does not close under
  any reading of `delta_BS`. It is also the only one of the six parameter
  files whose coefficients are single-precision literals and whose
  exponential factor is a function of `x/z` rather than `1-x/z`. Do not
  quote CLIC numbers from these grids without flagging it.

## Provenance of the spectra

Fits to GuineaPig runs of up to 1e8 events: four terms, a delta for the
undeflected beam plus regular terms carrying `(1-y)^kappa` with
`kappa = -2/3`, with deliberately different functional forms for circular
and linear machines.
