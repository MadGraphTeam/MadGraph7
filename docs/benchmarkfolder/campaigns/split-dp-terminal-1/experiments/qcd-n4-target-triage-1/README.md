# Closed QCD n4 target-miss triage

This offline analysis reparses all six completed 100-point streams for
`u u~ > z z g g [virt=QCD]`, bound to the immutable `qcd-n4-1` checkpoint.
Every original comparison and all ten pairs of candidate backends pass the
agreed `3e-6` criterion. The `1e-8` target misses remain visible and unchanged.
They affect finite coefficients only: 45 physical points in the union,
with both answer orders reported for each coefficient.

| Candidate backend | Points missing the target against baseline CutTools | Coefficient comparisons |
| --- | ---: | ---: |
| CutTools | 0 | 0 |
| Ninja | 44 | 88 |
| COLLIER | 40 | 80 |
| NinjaRS | 40 | 80 |
| TIRust | 39 | 78 |

The largest symmetric baseline discrepancy is `4.150911600344546e-7`,
COLLIER's finite result at point 61. The largest discrepancy among the four
other backends is `2.1667771641157215e-7`, Ninja versus COLLIER at point 89.
COLLIER and TIRust meet the target at every point except point 93, where their
finite symmetric residual is `1.1008650601963355e-8`. At that point CutTools,
Ninja, NinjaRS and TIRust all finished in QP, while COLLIER finished in DP.
These observations classify the disagreements; they do not assign truth
by voting or replace an independent finite-term adjudication.

`analysis.json` contains the exact point sets and component residuals,
all five accepted status/precision/accuracy records at each affected point,
all ten reducer-pair summaries and groups defined solely by which backends
miss the original target. Final accepted DP/QP counts are CT 88/12,
Ninja 97/3, COLLIER 100/0, NinjaRS 96/4 and TIRust 14/86. Internal attempts,
retry counters and FPE flags were not observed.

An additional source-bound check uses the universal massless QCD double
pole `-Born*(2*CF+2*CA)*G^2/(8*pi^2)`. The two Z bosons are colorless;
the external quarks and gluons contribute `2*CF+2*CA=26/3`. The copied,
compilation-bound Born and loop sources both divide by 144. The generated
broken-symmetry factor is 1: its initial 4 is divided by 2 for each identical
Z and gluon pair. The card gives `alpha_s=0.118`; the copied model source
defines `G=2*sqrt(alpha_s*pi)` and does not run it when the worker changes
`MU_R` to 1000 GeV. TIR interference supplies `2Re` and `1/(16*pi^2)`.

The maximum `abs(observed-expected)/abs(expected)` for this double-pole
check over all 100 points is `1.91413e-13` (CT), `8.91452e-11` (Ninja),
`3.79859e-9` (COLLIER), `2.05163e-12` (NinjaRS) and `4.88171e-13` (TIRust).
The coupling is reconstructed from exact source/card versions, not traced
from native execution. Born uses the original user momenta before loop
kinematic repair. This check does not establish finite or single-pole truth.

No new reducer evaluation, compile or timing is performed. The active EW
n4 case and mutable supervisor status are excluded. Reproduce the derived
JSON from the repository root without an ignored live build or network:

```sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/qcd-n4-target-triage-1/analyze.py
```

The replay verifies every consumed raw/parser/input file through the exact
closed checkpoint, verifies copied normalization sources against retained
compilation/generation provenance, and regenerates the analysis with the
retained parser and comparator. The original `3e-6` gate and `1e-8` target
are unchanged.
