# Differential jet rates from a showered event record

Notes for `madgraph/various/djr_from_hepmc.py`, the tool that reconstructs the
DJRs used to validate MLM merging.

`madgraph/various/djr_from_hepmc.py` reconstructs the differential jet rates
from a HepMC file and plots one line per jet-multiplicity sample together with
their sum. This is the plot that says whether the merging works.

It reruns the kT jet algorithm on the final state with the parameters the
jet-matching veto uses (`JetMatching:coneRadius`, `JetMatching:etaJetMax`,
visible final-state particles, `slowJetPower = 1`), so the rates come out on
the same scale the veto acts on. Each sample then has to switch off where the
next switches on: `d01` is filled below qcut by the 0-jet sample, whose hardest
jet can only come from the shower, and above qcut by the 1-jet sample, whose
hardest jet is a matrix-element parton; `d12` separates 1-jet from 2-jet, and
so on. Summed, the distribution has to be smooth across qcut.

Doing the clustering here rather than reading the `djrs.dat` that the
MG5aMC-PY8 interface writes matters twice over: it does not depend on the
merging implementation under test, and it works the same for madevent and for
madspace/mg7, which has no such interface.

    djr_from_hepmc.py --sample 0j:wj0.hepmc:1234.5 \
                      --sample 1j:wj1.hepmc:567.8 \
                      --sample 2j:wj2.hepmc:89.0 \
                      --qcut 30 --output-prefix djr

Each `--sample` is `LABEL:PATH[:CROSS_SECTION_IN_PB]`; the cross section is read
from the file when it is not given. It has to be right for the sum to mean
anything, since that is what puts the samples on a common normalisation.

Besides the plots the tool prints how big a step the summed distribution has at
qcut, by extrapolating a fit from each side and taking the ratio:

      d01 : ratio  0.952 +- 0.097  (0.5 sigma)  smooth
      d12 : ratio  1.148 +- 0.139  (1.1 sigma)  smooth

A normalisation mismatch between samples shows up here before it is visible by
eye: injecting a 40% over-count into the 1-jet sample of a synthetic set moves
`d01` to `1.337 +- 0.135`, flagged at 2.5 sigma, while the plot only shows a
slight kink.

Both the HepMC2 and HepMC3 ascii formats are read, gzipped or not. Cost is
about 19 ms per event for a 500-particle hadron-level record, so a 10k-event
sample takes a few minutes; `--max-events` cuts that down while iterating.
