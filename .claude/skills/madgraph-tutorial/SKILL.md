---
name: madgraph-tutorial
description: Teach the user MadGraph7 through its built-in interactive
  tutorials (the `tutorial` command) instead of improvising lessons. Use when
  the user wants to learn MadGraph or one of its features, asks for a
  tutorial, says they are new to MadGraph, or asks "how do I learn / get
  started with" generating processes, models, BSM, decays, NLO, loops,
  standalone matrix elements or checks. Not for getting a result: when the
  user wants a number or a sample, use the madgraph skill.
---

# Teaching MadGraph7 with its built-in tutorials

MadGraph7 ships interactive tutorials, written and (for most of them)
validated by its developers. They react to what the user types, check it,
and explain each step on the user's own machine. When a user wants to
**learn**, send them there rather than writing your own course. Your role
is to choose the tutorials, get them started, and stay alongside as a
tutor.

## 1. Choose the tutorials

| Tutorial | What it teaches | Status |
|---|---|---|
| `lo` (alias `MadGraph5`) | from a cold start to an LO event sample with madspace: generate, diagrams, output, launch | validated |
| `nlo` (alias `aMCatNLO`) | generate, output and run a process at NLO (aMC@NLO) | validated, but **legacy MG5 text**: the commands work, some details are dated (see below) |
| `syntax` | process syntax: coupling orders, interference (`^2`), decay chains, s-channels, gauge traps, polarisation and frames | validated |
| `model` | import, inspect, restrict and customise a physics model (including your own UFO) | validated |
| `bsm` | an EFT model (SMEFTatNLO): its coupling order, conventions, a restriction of your own, auto widths, broad resonances | validated |
| `decays` | decay chains, MadSpin, widths, and which to use when (at LO, on the mg7 output) | validated |
| `madloop` | generate and check standalone loop matrix elements | validated |
| `mg7` | madspace phase space, VEGAS, training MadNIS | AI-generated, not yet validated |
| `madevent` | `output madevent`: the classic directory, cards and tools | AI-generated, not yet validated |
| `standalone` | \|M\|² as a callable, linked into Python or C++ | AI-generated, not yet validated |
| `checks` | the `check` command family and what each check tells you | AI-generated, not yet validated |
| `exercises` | questions with the answers checked, and the mistake named | AI-generated, not yet validated |

**Say it when you recommend a tutorial that is not yet validated**, and
treat its statements with the same care as anything else you have not
checked. The list on the user's installation is what counts: `tutorial
list` (see section 5) prints the menu with these groups, and plugins can
add tutorials of their own.

**Learning paths.** `lo` always comes first. Its last step only says
`tutorial list`, so propose the order yourself:

| The user's goal | Path |
|---|---|
| general first contact | `lo` → `syntax` |
| BSM / EFT / their own model | `lo` → `syntax` → `model` → `bsm` (→ `checks` before trusting a model) |
| decays and spin correlations (tops, Ws) | `lo` → `syntax` (polarisation) → `decays` |
| NLO precision | `lo` → `syntax` → `nlo` (→ `madloop` for the loops themselves) |
| embedding matrix elements in their own code | `lo` → `standalone` |
| speed / phase space / MadNIS | `lo` → `mg7` |

For a first answer to someone describing a research project, keep to this:
- the path;
- what to start with now;
- the gaps the tutorials do not cover (below).

Don't lecture on the physics up front. It comes up step by step.

**Gaps no tutorial covers** (answer them from the `madgraph` skill's
`reference/*.md` files):
- MadSpin on NLO samples (`decays` is LO; `nlo` does not mention MadSpin):
  see `reference/madspin.md`, "NLO samples".
- **A BSM model needs an NLO version of its UFO for `[QCD]`.** A tree-level
  model is refused ("The model ... cannot handle loop processes"); `bsm`
  uses `SMEFTatNLO-NLO`, which has one. Tell BSM-at-NLO users early.
- Reweighting, MLM merging, plots: `reference/reweight.md`, `mlm.md`,
  `plots.md`.

**The `nlo` tutorial's dated details**:
- it mentions the StdHEP format and ".hep" files (the shower now writes
  HepMC);
- it says couplings other than QCD "cannot be perturbed yet" (EW
  corrections exist);
- its welcome suggests `p p > e+ ve [QCD]`, while the step expects
  `p p > t t~ [QCD]`.

Warn the user, and check against `reference/nlo.md` of the `madgraph`
skill.

**Status of the tool.** The README calls MadGraph7 an alpha release, "for
testing and feedback, not for production", with MG5_aMC@NLO as the stable
release. The commands carry over. When a user plans results for a thesis
or a paper, mention it once, and suggest they agree with their supervisor
on the version.

## 2. Prerequisites

- **Run from the MadGraph7 directory**: the one containing `bin/madgraph`
  (here: this repository's root).
- **Python ≥ 3.12**, the one MadGraph7 runs with: `python3 --version`. If
  the user keeps MadGraph7 in a virtual environment or direnv, start it from
  there.
- **madspace** (the integrator) is built at the first `launch` (a few
  minutes, needs a C++ compiler). The `lo` tutorial has a step for it, and
  says when it is already installed.
- `bsm` downloads the SMEFTatNLO model (network access).
- `nlo` launches with the Pythia8 shower: check the `pythia8_path` line of
  `input/mg7_configuration.txt` or `~/.mg7/mg7_configuration.txt`. If it is
  missing, the user can answer `shower=OFF` at the launch, or install
  Pythia8 (`install pythia8`, several minutes: ask first).
- On macOS MadGraph7 starts `caffeinate` (to keep the machine awake);
  `./bin/madgraph -s` prevents that. The warning about it is harmless.

## 3. Get it started: the user types, not you

The tutorials are interactive and are meant to be typed by the user. A
step's `hint` and `solution` **print** the command and never run it. Keep
that rule: don't run the steps for the user, and don't paste whole
solutions unless they ask.

Tell the user, from the MadGraph7 directory:

```bash
./bin/madgraph
```

and then, at the `MG7>` prompt:

```
tutorial lo
```

(or `tutorial` alone for the menu).
- If you have a tool to open a terminal for the user (e.g. a Terminal-panel
  tool in the desktop app), **offer** to start `./bin/madgraph` there; do it
  once they agree. They then type `tutorial <name>` themselves.
- An interactive session cannot be run through a scripted
  `bin/madgraph file` call: it reads the user's keyboard.

Commands the user can type while a tutorial runs:

| Command | Effect |
|---|---|
| `hint` | a hint for the current step |
| `solution` / `next` | print the command the step expects (not run) |
| `repeat` | print the current step again |
| `back` | go back one step |
| `skip`, `skip N` | move on, or jump to step N |
| `tutorial status` | the list of steps and the current one (`status` alone is not a command) |
| `tutorial stop` | leave the tutorial |
| `tutorial help` | the list above |

## 4. Tutor alongside

- **Read the step the user is on before answering.** The tutorials are
  Python modules: `madgraph/interface/tutorials/<name>.py` (`nlo` and
  `madloop` partly take their text from `madgraph/interface/
  tutorial_text_*.py`). Each `Step(...)` has a `title=`, the text it
  prints, and its `hint` and `solution`. Explain in the same terms as the
  tutorial, and don't contradict it without saying why.
- When the user refers to what they see ("this error", "what does this
  mean"), ask them to paste it, or read their terminal if you have a tool
  for that. The start-up lines `Running MG7 in debug mode`, `loading of
  madgraph too slow`, and `Please set the 'lhapdf'/'fastjet'/'eMELA'
  variable` (followed by a `set ... to` line that found it) are harmless.
- Answer side questions briefly, then bring them back to the step. When the
  question belongs to a later tutorial, say which one. When no tutorial
  covers it, answer from the `madgraph` skill's references.
- In `exercises`, let the user answer: the tutorial checks the answer and
  names the mistake. Give an answer only when asked.
- If a tutorial step is wrong or outdated (more likely in the
  not-yet-validated ones and in `nlo`), say so, show what actually works,
  and suggest reporting it at https://github.com/MadGraphTeam/MadGraph7/issues.

## 5. Looking at a tutorial yourself

To see what a tutorial teaches without the user, or to check that it
starts on this installation, run its first lines in a script, **in a
scratch directory** (not the repository). It stops at the first step that
waits for a user command:

```
# tut.mg7
tutorial list
tutorial lo
tutorial status
tutorial stop
```

```bash
<MG7>/bin/madgraph -s tut.mg7 > tut.log 2>&1 < /dev/null
```

The log has colour codes and the harmless start-up lines listed above. To
read the whole lesson instead, read the module: the steps are listed in
order, each with its `title=`.

## 6. When the user wants a result, not a lesson

If the user stops learning and asks for a cross section, a sample or a plot
("now do it for p p > t t~ j at 13.6 TeV"), use the **madgraph** skill:
Claude runs the scripts and reports the result. Offer to explain what the
script did in terms of the tutorial they just followed.
