# Reweighting

Use this when the user wants the same events with other parameters:
couplings, widths, EFT coefficients, a mass inside its width, another model
or process with the same external particles. Each event gets an extra
weight, |M_new|² / |M_old|² × its weight, so no new generation is needed.

## The reweight card

Write `Cards/reweight_card.dat` before launching. `set` in the launch block
cannot edit it. So the workflow is three steps: a script with `generate` +
`output`, then the card file, then a script with the `launch` block.
**Always write the whole card yourself**: the default card of an aMC@NLO
output starts with `change mode NLO`, which does not work on this version
(see NLO samples).

```
change helicity True              # the default; see Limitations
launch --rwgt_name=wz26           # one launch block = one new weight per event
set decay 23 2.6                  # set <block> <id> <value>, as in the param card
launch --rwgt_name=mt1725
set mass 6 172.5
```

- Parameters use param-card blocks (`set decay 23 2.6`, `set yukawa 6
  170`, `set <block> <id> <value>`), or a line with the path of a full
  param card.
- **Derived parameters move too.** In `sm`, M_W and sin²θ_W are computed
  from `sminputs` (1/α_EW, G_F) and M_Z. `set sminputs 1 127.9` therefore
  also moves M_W (80.419 → 79.825 GeV), and the reweighting does not warn
  you. Look at the model's `parameters.py` (or launch once with the new
  value: the direct mg7 launch prints "the mass of particle 24 (w+) is
  changed"). See the validity rules below.
- `--rwgt_name=` names the weight; otherwise the names are `rwgt_1`,
  `rwgt_2`, ...
- Before the first `launch` line, you can add:
  - `change model <name>`, then give each block a full param card path for
    the new model;
  - `change process <definition>` (`--add` for several), with **exactly
    the same external particles**;
  - `change helicity False` (see Limitations);
  - `change mode LO|NLO|LO+NLO|NLO_tree`, which only matters for NLO samples.
- α_s and the scales are taken from each event, not from the card.

## Running it

**At generation**: add `reweight=ON` to the launch block (mg7, madevent or
aMC@NLO). Always run the script with `< /dev/null`: the reweighting builds
its own matrix elements, and that internal step asks a question the script
cannot answer. With stdin open it waits forever.

**On an existing run** (madevent and aMC@NLO outputs; an mg7 output has no
`reweight` command and only reweights at generation):

```
set crash_on_error True --no_save
set nb_core 1 --no_save
launch PROC -i
reweight run_01 -f
```

- The weights are added to that run's event file.
- With several cores the file is split, and the parts are reweighted by
  child processes. If a child fails, the parent **waits forever**.
  `set nb_core 1 --no_save` avoids the split, at the cost of speed. Use it
  whenever the reweighting might fail (NLO samples, new models).
- Reweighting acts on the LHE file, before any shower. A sample generated
  with the shower off can be reweighted and showered afterwards.

## Reading the result

- Each event gets `<wgt id='NAME'> value </wgt>` in its `<rwgt>` block, and
  the header gets a `<weight id='NAME'>` line describing the change.
- `scripts/run_summary.py PROC run_XX` lists every reweighting weight it
  finds. For each, it gives the new σ, the **ratio to the original σ**,
  that ratio times the run's quoted σ, the spread of the per-event ratios,
  and the effective number of events.
- **Quote ratio × the run's quoted σ.** For NLO samples the mean of the
  original weights in the file differs slightly from the run's σ (113.30
  against 114.2 pb for W+W−).
- MadGraph files use `event_norm = average` (the mean weight is σ); an mg7
  LHE file has no `event_norm` line, but is always average.
- **mg7 runs leave two event files**: `events.lhe.gz` has **no**
  reweighting weights; `unweighted_events.lhe.gz` has them. Use the second.
- **The log's "new cross-section" is wrong for mg7 samples**: it is
  multiplied by the number of events (6.84e6 pb instead of 684 pb for 10k
  events). Compute it from the weights. For aMC@NLO samples the log line
  was right.

Validated: LO Drell-Yan (mg7, 10k events), Γ_Z 2.441 → 2.6 GeV:
reweighted 684.01 ± 0.30 pb, direct generation at 2.6 GeV 683.0 ± 1.2 pb.

## Limitations

**When reweighting is valid.** The new |M|² must be well described by the
events of the old one:
- **Never change the mass of an external particle** (a final-state top in
  t t̄), **including through a derived parameter** (M_W from `sminputs` in
  `sm`): the momenta stay on the old mass shell. Measured on p p > w+ w-
  with 1/α_EW → 127.9: reweighting gave −2.03% (LO) and −2.40% (NLO, LO
  mode), direct generation −0.74% and −0.56%.
- **When in doubt, generate one point directly** (cheap at LO with mg7) and
  compare it with the reweighted σ before trusting a scan.
- Changing a resonance mass by much more than its width, opening a new
  resonance or channel, or switching on a coupling whose diagrams the
  original sample did not populate gives large or undefined weights. An
  event with an original |M|² of 0 stops the run ("Invalid matrix element
  for original computation (weight=0)").
- When the ratios vary a lot, the effective statistics drop. Look at the
  spread of new/old; if a few events dominate, regenerate instead.

**Helicity.** By default (`change helicity True`), when the events carry
helicities (MadGraph LO samples do), the ratio is taken **for the event's
helicity configuration**: |M_new(h)|² / |M_old(h)|². Otherwise it uses
helicity-summed |M|², which is what `change helicity False` forces.
- The cross section is the same either way. Measured on Drell-Yan: 0.93854
  vs 0.93853 ± 0.00058.
- Individual events differ (up to 3.7% there). Keep the default when the
  helicities in the file are used downstream (τ decays, polarisation
  observables), so the weights stay consistent with them.
- MadGraph turns it off by itself, with a warning in the log ("due to
  ordering ambiguity, we flip off helicity per helicity reweighting"), when
  identical particles or decay chains make the matching ambiguous.
  Events with helicity 9 (unknown) always use the sum.

**Colour.** The ratio is always **colour-summed**, and the event's colour
flow (the one the parton shower uses) is **not changed**. If the new model
or process changes the colour structure (a new coloured resonance, a
different dominant colour flow), the colour flows in the file still follow
the old process. Showered results are then inconsistent: regenerate.

## NLO samples (aMC@NLO events)

- **NLO-accurate reweighting needs extra information in the events**, set
  by `store_rwgt_info = True` in `Cards/run_card.dat` **before
  generating**. It is switched on automatically when `reweight=ON` is part
  of the generation launch. Without it, only the LO mode is possible.
- Modes (`change mode ...`):
  - `LO`: approximate. The whole NLO event weight is multiplied by a ratio
    of tree-level matrix elements.
  - `NLO`: exact. Each part of the NLO event weight is reweighted
    separately, including the virtual corrections; MadGraph builds a loop
    matrix element for this (`rw_mevirt`).
  - `LO+NLO`: both, as `NAME_lo` and `NAME_nlo`.
  - `NLO_tree`: reweights the tree-level parts only.
- **Current status on this version: only `LO` works.** `NLO`, `LO+NLO` and
  `NLO_tree` fail while loading `rw_mevirt` (`ImportError ... symbol not
  found ... _ml5_3_get_density_`).
  - With several cores (at generation, or after the fact without
    `nb_core 1`), a child dies and the **run hangs**. The progress line
    (`Idle: 0, Running: 3, Completed: 1`) then stops changing, and the
    children sit at 0% CPU. A healthy run shows the same line for a few
    seconds, so wait a couple of minutes before concluding.
  - With `nb_core 1` the error is logged at DEBUG level, and the run exits
    0 without the weight.
  - Grep the log for `ImportError`, and tell the user that only LO-mode
    reweighting is available.
  - For an NLO-accurate answer instead, run **fixed-order NLO directly** at
    both parameter points (a few seeds each, `reference/nlo.md`). It takes
    minutes for 2 → 2.
  - Cleanup after a killed run:
    - delete `<dir>/RunWeb`;
    - delete the split files `Events/run_XX/events.lhe.gz_*.lhe` and the
      `*_debug.log` next to them;
    - the untouched original is `events_orig.lhe.gz`.
- The LO mode is accurate when the K-factor does not depend on the change.
  Drell-Yan at NLO, Γ_Z → 2.6 GeV: LO-mode reweighting 1770.1 pb, direct
  NLO 1763 ± 2.7 pb (0.2% apart).
- Fixed-order NLO runs have no events, so they cannot be reweighted:
  rerun them.
- aMC@NLO reweighting leaves empty files in the work directory
  (`events.lhe.rwgt`, `reweight_xsec_events.output`,
  `scale_pdf_dependence.dat`).

## Density-matrix reweighting

`reweight=density` (or `reweight --mode=density`) uses a separate card
(`Cards/density_card_default.dat`) for polarisation and decay studies. It
is not covered here; read that card's comments first.

## Reporting

Report the new σ, computed from the weights, next to the original one. Also
give:
- what was changed;
- the helicity mode;
- for NLO samples, the mode (and that only LO mode works on this version);
- whether the change is safely inside the validity conditions above.

Both cross sections come from the same events, so their MC errors are
strongly correlated, and their ratio is far more precise than either.
