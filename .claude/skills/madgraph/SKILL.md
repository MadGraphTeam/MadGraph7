---
name: madgraph
description: Run MadGraph7 for the user. Generate a process, compute an LO cross
  section, produce parton-level events (with cuts, scale/PDF uncertainties,
  histograms, showering), and report the results. Use when the user asks for a
  cross section, events, distributions or a MadGraph/MG7/MG5 run, or asks why a
  MadGraph run failed.
---

# Running MadGraph7

MadGraph7 is driven by **command scripts** passed to `./bin/madgraph` at the
root of this repository. Never type commands into the interactive prompt, and
never pipe them through stdin: write a script file and run it.

## The workflow

1. **Write the script** in a work directory (not inside the repository, unless
   the user asks), e.g. `run_tt.mg7`:

   ```
   set crash_on_error True --no_save
   import model sm
   generate p p > t t~
   output PROC_tt
   launch PROC_tt
   set nevents 10000
   set iseed 42
   set cuts.t-pt.min 50
   done
   ```

   - **Line 1 is mandatory.** Without `crash_on_error`, MadGraph exits 0 even
     after a crash and simply returns to the prompt. `--no_save` keeps the
     setting out of `input/mg7_configuration.txt`. Never drop `--no_save`:
     the saved option would change every later session of the user.
   - `output <dir>` writes an **mg7** process directory (vectorised C++/GPU
     matrix elements plus the madspace phase-space library). Use
     `output madevent <dir>` only when the user wants the Fortran MG5 workflow,
     or for a feature mg7 lacks (see `reference/troubleshooting.md`).
   - Everything between `launch` and `done` answers the launch question: the
     `set` lines edit the cards, and `done` starts the run. Settings are written
     into `<dir>/Cards/` and **persist** for the next launch of that directory.
   - Several `launch` blocks may follow one `output`, e.g. a scan.

2. **Run it**, merging stdout and stderr into a log, and keep the exit code:

   ```bash
   /path/to/MadGraph7/bin/madgraph run_tt.mg7 > run_tt.log 2>&1; echo "exit=$?"
   ```

   Runs can take from seconds to hours. For anything beyond a quick check, run
   it in the background and poll the log with `grep "integral:" run.log | tail -1`;
   don't block on it. If the RSD in those lines runs into the hundreds while
   `unw. eff.` stays at 0, the integrand is divergent (a missing cut). Stop the
   run instead of waiting. The first launch in a fresh installation also
   builds the madspace library (a few minutes, needs a C++ compiler and
   Python >= 3.12).

3. **Check for failure before reading any result**: a non-zero exit code, or
   `Traceback`, `Error`, `invalid set command`, `Ambiguous key` or `WARNING`
   lines in the log. Grep for them; never print a whole log, because PDF
   downloads fill it with progress bars. `reference/troubleshooting.md` has
   the grep commands and what each symptom means.
   A `WARNING: invalid set command ...` is NOT fatal: the run continues
   **without** that setting. Treat it as a failed request and fix the script.
   On a crash, MadGraph writes a `debug` or `MG5_debug` file in the work
   directory. Quote the error to the user and don't guess.

4. **Read the results** with the bundled script (standard library only, so any
   `python3` works). It handles mg7 runs, madevent runs and MadSpin-decayed
   runs:

   ```bash
   python3 .claude/skills/madgraph/scripts/run_summary.py PROC_tt          # latest run
   python3 .claude/skills/madgraph/scripts/run_summary.py PROC_tt run_02   # a given run
   ```

   It prints the cross section with its MC error, the event count, the seed,
   the collider, PDF, scale choice and **all active cuts**, the scale and PDF
   bands with their definitions, and where the events and plots are. The
   settings come from the run's own copy of the card, so they are what that
   run really used.

## Reporting to the user

- Give the cross section **with its Monte Carlo error and unit** (pb), plus
  the scale and PDF uncertainty, with how they are defined, when they were
  computed. Say which PDF set, collider energy, scale choice and cuts were
  used. Include the **default** cuts that apply to the final state (e.g. the
  lepton cuts for a W -> l nu process), because the user did not ask for
  them. A bare number without these is not a result.
- Mention every setting that was **not** applied (invalid `set`) and every
  default you chose for the user (e.g. collider energy, PDF set).
- Point to the event file and plots (paths relative to the work directory).
- These are **LO** numbers unless the process was generated with `[QCD]`. Say
  so if the user may expect NLO accuracy.

## Choosing settings

- The mg7 run card is `<dir>/Cards/run_card.toml`, a TOML file with sections
  (`[run]`, `[beam]`, `[generation]`, `[cuts]`, `[histograms]`, ...). Read
  `reference/mg7_run_card.md` before setting anything beyond `nevents`/`iseed`.
- Cuts are on **groups of particles**, named in the `[multiparticles]` section
  of the generated card: `jet`, `bottom`, `lepton`, `missing`, `photon`, plus
  one group per other final-state particle, named without the charge sign
  (`t` for t/t~, `w` for W+/W-). Check the card before writing a cut. Use
  `set cuts.<group>-<observable>.<min|max> <value>`, always with the `cuts.`
  prefix. A group that does not exist crashes the launch with `KeyError`.
- Use `section.key` for any key that is not unique (`set beam.pdf ...`, not
  `set pdf ...`).
- Familiar madevent names (`set ptj 20`, `set mmll 50`, `set ebeam1 6500`)
  are mostly **not** understood by mg7: they are ignored with a warning. The
  mg7 equivalents are in `reference/mg7_run_card.md`.
- Always set a seed (`set iseed N`) so that the run can be reproduced, and
  report it.
- **Cross section only**: `set nevents 0` runs just the integration (survey).
  It gives roughly a 2% MC error and **no** scale/PDF bands, which is fine
  for a quick estimate. For a number with bands, or better than about 1%,
  generate events: 10k events gave about 0.4% for p p > w+ j. Tell the user
  which one you did.
- When the user didn't specify the collider, energy or PDF, the defaults are
  13 TeV pp and the NNPDF4.0 LO set. Use them, but state them.

## Before generating a new process

- If unsure of the particle names, run a script with `import model <model>`
  and `display particles` (or `display multiparticles`) first. In `sm`:
  `p`/`j` contain the gluon and the u, d, s, c quarks and their antiquarks (no b),
  `l+`/`l-` = e and mu, `vl`/`vl~` = all three neutrinos, the top is `t`/`t~`.
- Use `$ X` / `/ X` to exclude a propagating particle, `,` for decay chains
  (`p p > t t~, t > b w+, t~ > b~ w-`), and coupling-order constraints such
  as `QED=2`, as in MG5. For spin-correlated decays of a large sample,
  MadSpin (`madspin=ON`, see `reference/mg7_run_card.md`) is usually the
  better choice.
- Large multiplicities (e.g. `p p > t t~ j j j`) can take a long time to
  generate and to run. Warn the user and start small.

## Reference files

- `reference/mg7_run_card.md`: every `set` you are likely to need, cut
  syntax, madevent-to-mg7 names, and the Pythia8 shower and MadSpin switches.
- `reference/troubleshooting.md`: log grepping, harmless lines, failures that
  look like successes, runs that don't converge, installation (madspace,
  LHAPDF, Pythia8), and when to fall back to `output madevent`.
