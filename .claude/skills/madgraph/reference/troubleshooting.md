# MadGraph7 troubleshooting

## Reading the log

Logs are long, and an LHAPDF download prints its progress bar into them, so
never print a whole log. Grep it:

```bash
grep -nE "Traceback|Error|ERROR|CRITICAL|invalid set command|Ambiguous key|WARNING" run.log \
  | grep -vE "caffeinate|loading of madgraph too slow"
grep -nE "^set |madgraph7: (set|modify)|INFO: modify" run.log        # which settings were applied
grep -E "integral:" run.log | tail -3                                 # integration progress
```

These lines are harmless; don't report them:
`WARNING: launching caffeinate ...`, `('WARNING: loading of madgraph too
slow!!!', ...)`, `stty: stdin isn't a terminal`, `Trying to add argument
mode in DensityCard ... Did you mean ['matrix_normalisation']`, `Please set
the 'fastjet'/'eMELA' variable ...` (only needed for NLO matching tools),
`... lhapdf-config does not seem to correspond to a valid lhapdf-config
executable` followed by a `set lhapdf to ...` line (the first guess failed,
the next one was found), `value '[]' for entry 'bypass_check' is not
valid` (madevent launches; unrelated to the "not valid for current
question" launch trap),
`CRITICAL: MadSpin unweight efficiency` / `MadSpin phase timings` (MadSpin
prints its statistics at CRITICAL level), `Using default gzip`. The NLO
warnings (α_s update, Z width set to zero, missing run-card arguments,
multiparticles at NLO) are explained in `reference/nlo.md`.

Logs and `display` output show flavour-grouped names such as `_quark`,
`_anti_quark`, `_lepton` or `_neutrino` (PDG codes 81-83): these are the
grouped light flavours (see `reference/diagrams.md`), not an error.

## Failures that do not look like failures

| Symptom | Meaning | Fix |
|---|---|---|
| exit code 0, but a `Traceback` in the log | the script ran without `set crash_on_error True --no_save` | put it on line 1. Still grep the log: an error that MadGraph handles itself prints `ERROR:` and continues |
| `WARNING: invalid set command X` | X was **ignored**, the run used the old value | use the mg7 name (`reference/mg7_run_card.md`) |
| `WARNING: Ambiguous key 'pdf'` | ignored, the key exists in several sections | `set beam.pdf ...` |
| `KeyError: '<name>'` at launch | a cut names a group that is not in `[multiparticles]` | use an existing group, or add it to the card. The bad cut stays in `Cards/run_card.toml`: remove that line |
| settings from an earlier launch reappear | every `set` persists in `<dir>/Cards/` | set everything explicitly, or start from `run_card_default.toml` |
| MadSpin sample has the wrong decays | `set decay ...` was rejected | write `decay ...` without `set` |
| no `.lhe.gz` in an mg7 run | the default output is `events.npy` | `set output_format lhe`, or convert with `<dir>/bin/npy_to_lhe` |

## The run does not finish

- **Divergent integrand.** Signature: `RSD` in the hundreds or more,
  `unw. eff.: 0.00000`, and the sample count running into billions, while the
  event count barely moves. A cut that keeps massless particles (photons,
  jets, massless leptons) separated and away from zero pT is missing, often
  after `set no_parton_cut`. Stop the run and add the cuts. Don't wait it out.
- **Very low efficiency in a valid setup** (healthy RSD, slow events): the
  default `generation.max_overweight_truncation` (1e-3) is strict. 1e-2 is
  about 2x faster, at the price of a percent-level truncation of the weight
  tail. Say so to the user if you change it.
- **High multiplicity**: generation and compilation scale steeply with the
  number of legs (`p p > t t~ j j j` takes far longer than `p p > t t~`). Start
  from the lower multiplicity, and estimate the time from it.

## Environment and installation

- **First launch builds madspace** (C++, a few minutes). If it fails, run
  `install madspace` at the MadGraph prompt, or see
  `docs/source/madspace/installation.rst`. A process directory created with
  another madspace build refuses to run ("source hash mismatch"): re-run
  `output` for it.
- **PDF sets** are downloaded on first use into the LHAPDF data directory,
  so the first run needs network access. mg7 takes LHAPDF set names and
  aMC@NLO takes lhaids. `lhapdf-config` comes from the `lhapdf` option of
  `input/mg7_configuration.txt` or of the user's
  `~/.mg7/mg7_configuration.txt` (`~/.mg5` is never read). To see what a
  process directory will use: `grep -n lhapdf <dir>/Cards/*configuration.txt`.
  The log confirms it with `Using LHAPDF v...`. If LHAPDF is missing, ask
  the user before running `set lhapdf /path/to/lhapdf-config`. That one
  should be saved, so without `--no_save`.
- **Pythia8/Delphes/MadAnalysis5** are installed by MadGraph itself:
  `install pythia8` (asks for nothing; takes several minutes). Ask the user
  before installing anything.
- **Stray files in the working directory**: a crash leaves `debug` /
  `MG5_debug`, and runs can leave `nsqso_born.inc` and an empty
  `additional_command`. The Fortran matrix-element module writes
  `param.log` into the current directory when it is initialised. Run scripts from a
  dedicated work directory, not from the repository root.
- The `caffeinate` warning on macOS is harmless.

## mg7 limitations: when to use `output madevent`

The mg7 output is LO only. Use the madevent (Fortran) path when the user
needs:
- **NLO**: `generate p p > t t~ [QCD]` followed by a plain `output` produces an
  aMC@NLO directory (see `reference/nlo.md`).
- **MLM matching / merging** (see `reference/mlm.md`), other features of the
  classic run card that have no mg7 equivalent, or a comparison against an
  MG5 sample.

In a madevent directory the run card is `Cards/run_card.dat`. All classic
names work there (`set ptj 20`, `set ebeam1 6500`, `set lhaid 331900`,
`set pdlabel lhapdf`, ...). The results are in
`Events/run_XX/run_XX_tag_1_banner.txt` and `unweighted_events.lhe.gz`, and
`scripts/run_summary.py` reads them as well.

## Reporting a bug

If the failure is a crash inside MadGraph (a `Traceback` from MadGraph code,
not a user error), report the command, the last 30 lines of the log, and the
`debug` / `MG5_debug` file. Suggest that the user report it to
https://github.com/MadGraphTeam/MadGraph7/issues, and don't patch MadGraph
unasked.
