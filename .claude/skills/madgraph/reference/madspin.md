# MadSpin: decaying heavy particles

Use this when the user wants the decays of tops, W, Z, H or BSM particles
added to a sample, with spin correlations, and possibly off-shell effects.
The production is generated with the particles stable. MadSpin then decays
them event by event, so one production sample can serve several decay
channels.

**What "off-shell" means here.** In the off-shell modes the resonances get
Breit-Wigner line shapes, up to `BW_cut` widths from the pole, and the decay
matrix elements are evaluated off shell. Non-resonant diagrams (e.g.
single-top-like contributions to b b̄ ℓν jj) are **not** included. Those
need a full generation of the final state (`p p > b b~ l+ vl j j`), which
is much more expensive and, for jets, also picks up QCD backgrounds. Say
which one you did.

## Two ways to run it

### During the launch (mg7, madevent or aMC@NLO outputs)

For simple decays, type them in the launch block:

```
launch PROC_tt
madspin=ON
set nevents 10000
set spinmode onshell
decay t > w+ b, w+ > e+ ve
decay t~ > w- b~, w- > j j
done
```

- `decay ...` lines go in the launch block **without `set`**. `set decay
  ...` is rejected, and MadSpin then runs with its default card.
- MadSpin options can be set by their bare name (`set spinmode onshell`).
  `set madspin_card <option> ...` is refused ("Command set not allowed for
  modifying the madspin_card"), and a name that also exists elsewhere
  (`seed`) goes to the other card.
- A `decay X` line replaces the default line for X. The **other default
  lines stay** (`decay w+ > light light`, `decay z > light light`, ... where
  `light` is quarks **and** leptons), and apply to any such particle of the
  production.

**For anything more (a seed, `@` tags, `define`s, several options), write
`Cards/madspin_card.dat` yourself between `output` and `launch`.** After
`output` only `madspin_card_default.dat` exists. A `madspin_card.dat` you
write is kept as is, and switches MadSpin on by itself (`madspin=ON` is
then optional). Example (semi-leptonic t t̄, reproducible):

```
set seed 1361
set decay_generator madevent
set spinmode madspin
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm
define vl~ = ve~ vm~
decay t > w+ b, w+ > l+ vl @1
decay t~ > w- b~, w- > j j @1
decay t > w+ b, w+ > j j @2
decay t~ > w- b~, w- > l- vl~ @2
launch
```

Then:
- The decayed sample is a separate run: `Events/run_XX_decayed_1/
  unweighted_events.lhe.gz`. The undecayed production stays in
  `Events/run_XX/`; an mg7 run has two identical copies of it there
  (`events.lhe.gz` and `unweighted_events.lhe.gz`).
- `scripts/run_summary.py PROC run_XX_decayed_1` prints σ×BR, the number of
  events, the spin mode, the seed, the decay lines and the decay generator.
- An mg7 run switches to LHE output automatically when MadSpin is on.

### On an existing LHE file (any generator, or an earlier run)

```
# madspin_card.txt
import decay_tt/events.lhe.gz
set spinmode madspin
set seed 11
set decay_generator madevent
decay t > w+ b, w+ > e+ ve
decay t~ > w- b~, w- > mu- vm~
launch
```

```bash
python3 <MG7>/MadSpin/madspin madspin_card.txt > madspin.log 2>&1 < /dev/null
```

- Use MadGraph7's Python (the one `bin/madgraph` runs with).
- The output is `<input name>_decayed.lhe.gz` next to the input
  (`decay_tt/events_decayed.lhe.gz`).
- **Put a copy of the input in its own directory.** MadSpin writes about 20
  working directories next to it (`madspin_me`, `ms_wstatus_*`,
  `decay_*`), and in `spinmode none` it leaves the input decompressed
  (`events.lhe`).

## Reproducibility: `seed` alone is not enough

With the default `decay_generator mg7`, the decays are **different on every
run, even with `set seed`**: the seed does not reach the generation of the
decay events. Measured: two runs, same card, same input, different events.

Add `set decay_generator madevent` (Fortran decay generator). The output is
then byte-identical across reruns, and costs about 20 s more (57 s against
38 s for 5000 events).

The log line `Will use seed 123176626` (not the number in your card) is
expected: in the density modes, MadSpin derives its working seed from
yours, deterministically.

## Spin modes (`set spinmode ...`)

| Mode | Spin correlations | Off-shell effects | `@` tags |
|---|---|---|---|
| `madspin` (default; `full` is an alias) | yes | yes: line shapes and off-shell decay matrix elements | rectangular groups (see below) |
| `PA` | yes | Breit-Wigner masses (pole approximation) | rectangular groups |
| `onshell` | yes | no: exactly on-shell resonances | rectangular groups |
| `none` | **no** | no | **ignored** (warns; produces every combination) |
| `madspin_v1` | yes | yes | fully supported |
| `onshell_v1` | yes | no | legacy, for debugging only |

`madspin`, `PA` and `onshell` are the **density modes**. `madspin_v1` is
the legacy implementation (no density matrix). It is needed for
polarisation braces on the decay side, and it writes the BR table into the
banner.

Measured on the same 10k t t̄ events (13 TeV), decayed to e⁺ν b μ⁻ν̄ b̄,
14-23 s per mode on 18 cores:

| Mode | σ×BR | m(bℓν) spread | Lepton Δφ asymmetry (±0.010) | C_hel = −9⟨cos θ₊ cos θ₋⟩ (±0.03) |
|---|---|---|---|---|
| `none` | 4.441 pb | 0 | +0.267 | −0.05 (no correlation) |
| `onshell` | 4.474 pb | 0 | +0.172 | +0.31 |
| `PA` | 4.275 pb | 3.1 GeV | +0.167 | |
| `madspin` | 4.275 pb | 3.2 GeV | +0.185 | +0.27 |
| `madspin_v1` | 4.293 pb | 3.2 GeV | +0.185 | |

- The off-shell modes are about 4% lower because σ×BR includes the
  **Breit-Wigner truncation** at `BW_cut`. The log says so ("Breit-Wigner
  truncation at BW_cut = 15 keeps 0.95785 ... the reported sigma is scaled
  by it"), and prints the BR itself ("decay group rates ... (BR ...)").
  Quote σ×BR from the mode you used, and never mix normalisations of
  different modes.
- Spin correlations do **not** show up in a single lepton's angle:
  ⟨cos θ_ℓ⟩ is 0 in every mode, because QCD t t̄ tops are unpolarised. They
  show up in **correlations between the two decays**: ⟨cos θ₊ cos θ₋⟩, the
  lepton Δφ, or the angle between the lepton and the down-type quark of the
  other top.

Choosing:
- `madspin` (the default) unless there is a reason not to.
- `onshell` when the analysis should not see the resonance line shape, or
  to compare with an on-shell calculation.
- `none` as the uncorrelated reference, or for decays whose spin
  information does not matter (scalars).

**An uncorrelated reference with tagged channels**: `none` ignores the
tags. Decay the same production twice, untagged: once with the lepton from
the top (`decay t > w+ b, w+ > l+ vl` with `decay t~ > w- b~, w- > j j`),
once from the antitop. Then combine the two files.

## Decay angles in the top rest frame

MadSpin output keeps the decay chain: the top and W are status-2 lines,
with mother links. `lhe_parser` turns those links into particles, and
`FourMomentum.boost_to_restframe(p)` returns a momentum in the rest frame
of `p`.

```python
import sys, numpy as np
sys.path.insert(0, "<MG7>")
from madgraph.various import lhe_parser
FM = lhe_parser.FourMomentum

def ancestor(p, pdg):
    """the first particle up the mother links with this PDG code"""
    while p is not None and not isinstance(p, (int, float)):
        if p.pid == pdg:
            return p
        p = p.mother1
    return None

def cosang(a, b):
    return (a.px*b.px + a.py*b.py + a.pz*b.pz) / (a.norm * b.norm)

cc = []
for ev in lhe_parser.EventFile("PROC/Events/run_01_decayed_1/unweighted_events.lhe.gz"):
    lp = [p for p in ev if p.pid == -11 and p.status == 1][0]          # e+ from t
    lm = [p for p in ev if p.pid == 13 and p.status == 1][0]           # mu- from t~
    t, tb = ancestor(lp, 6), ancestor(lm, -6)
    tt = FM(t) + FM(tb)
    t_tt, tb_tt = FM(t).boost_to_restframe(tt), FM(tb).boost_to_restframe(tt)
    c1 = cosang(FM(lp).boost_to_restframe(tt).boost_to_restframe(t_tt), t_tt)    # helicity basis
    c2 = cosang(FM(lm).boost_to_restframe(tt).boost_to_restframe(tb_tt), tb_tt)
    cc.append(c1 * c2)
print("C_hel = %.3f" % (-9 * np.mean(cc)))
```

Weight each event with its weight (`ev.wgt`) if the sample is weighted;
MadSpin output from an unweighted production is (almost) unweighted.

## Useful options (`set <option> <value>`)

| Option | Default | Use |
|---|---|---|
| `seed` | from the clock | set it, **together with `decay_generator madevent`**, for reproducible decays |
| `decay_generator` | `mg7` | `madevent` makes the decays reproducible (see above) |
| `BW_cut` | -1 = from the run card (15) | how far off shell (in widths) the off-shell modes go; it also scales σ×BR |
| `nb_core` | the global `nb_core` | cores for the unweighting |
| `ms_dir` | none | directory where MadSpin keeps its decay setup, so later runs with the same decays skip that step |
| `decay_output` | `auto` (unweighted) | `weighted`: one decay per event, the convolution goes into the event weight, about 4x fewer matrix-element calls, but the **output is weighted** |
| `Nevents_for_max_weight`, `max_weight_ps_point` | from the sample | precision of the maximum-weight estimate (more = fewer overweights, slower) |

More options exist (`unweighting`, `keep_weight_for_polarization_*`,
`pure_interference`); their descriptions are in
`MadSpin/interface_madspin.py` (search for `add_param`).

## Decay syntax

- Decay chains as in `generate`: `decay t > w+ b, w+ > l+ vl`. Use
  multiparticles (`define` lines before the decays). The sm `l+`, `vl` and
  `j` definitions apply.
- **Correlated channels** use tags, as in the card example above: lines
  with the same tag are used together, and lines without a tag are common
  to all groups.
  - The density modes support this rectangular shape (one line per
    particle and per tag), and log "N decay groups (@1, @2)". Anything else
    warns and falls back to every combination; `none` always does.
    `madspin_v1` supports them fully.
  - **Write tagged decays in the card file, not in the launch block.**
    Each `decay t ...` line typed in a launch block replaces the earlier
    `decay t` lines, tags or not (from the code; not tested).
- Polarised decays (braces in the decay chain) need `spinmode madspin_v1`
  or `none`.

## Normalisation and weights

- The `<init>` cross section of the decayed file is **σ × BR**, with the
  branching ratios computed by MadSpin from the param-card widths. In the
  off-shell density modes it also includes the Breit-Wigner truncation (see
  above). The effective BR is σ×BR (`<init>`) divided by the production σ;
  the log has the pure BR.
- Events keep the production weights. An unweighted production gives
  (almost) constant weights; mg7 samples carry a handful of overweight
  events, and MadSpin propagates them.

## NLO samples

MadSpin decays aMC@NLO events, inside the aMC@NLO launch (`madspin=ON`) or
standalone on its `events.lhe.gz`. The decays are computed at LO, while the
production keeps its NLO accuracy **and its negative weights**.

Measured on 2000 NLO t t̄ events (no shower) decayed to e⁺ν b μ⁻ν̄ b̄:
- 22.8% negative weights before and after the decay;
- σ×BR 7.96 pb (`onshell`) and 7.63 pb (`madspin`), from σ_NLO = 674.9 pb;
- 24-32 s per mode.

MadSpin works on LHE events, so the decay comes before the shower: shower
the decayed file (with the shower the MC@NLO events were matched to).

## Log lines

Harmless, don't report them:
- `CRITICAL: MadSpin unweight efficiency`, `CRITICAL: MadSpin phase timings`
  (statistics printed at CRITICAL level);
- `CRITICAL: kwargs[merge_same_topologies] = True` (a leftover debug print);
- `WARNING: Previous runs exists but they will not be present in the html
  output.`;
- after MadSpin on an **mg7** output, `ERROR: I/O error (2): No such file or
  directory` (MadGraph tries to save `HTML/results.pkl`, which mg7 outputs
  do not have). The decayed sample is complete.

Worth reading:
- `Running MadSpin in spinmode <mode>`: check it is the one you asked for.
- The Breit-Wigner truncation and BR lines.
- The decay-group line, when you used tags.
- The unweighting efficiency (trials per event).

## Reporting

Give:
- the spin mode, the decay channels, and what "off-shell" includes;
- σ × BR, the production σ, and the BR with the truncation factor;
- the seed and the decay generator (reproducible or not);
- which file holds the decayed events.
