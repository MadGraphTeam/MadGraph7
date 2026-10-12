# NLO with aMC@NLO

Use this when the user asks for an NLO cross section, a K-factor, NLO
distributions or NLO+PS events. This is the MG5_aMC@NLO machinery, unchanged
in MadGraph7. Its cards and outputs differ from the mg7 LO workflow, so don't
apply `mg7_run_card.md` here.

## Two scripts: generate, then launch

Splitting the work lets you check the generated cards before spending CPU on
the run. A launch-only script works on any existing directory.

```
# gen.mg7
set crash_on_error True --no_save
import model sm-no_b_mass
generate p p > z j [QCD]
output PROC_zj_nlo
```

```
# launch.mg7 (several launch blocks may follow each other, e.g. NLO then LO)
set crash_on_error True --no_save
launch PROC_zj_nlo
order=NLO
fixed_order=ON
set iseed 11
set lhc 13.6
set jetalgo -1
set jetradius 0.4
set ptj 30
set etaj 2.5
set reweight_pdf True
done
launch PROC_zj_nlo
order=LO
fixed_order=ON
set iseed 12
done
```

- `[QCD]` asks for NLO QCD corrections. A plain `output` then writes an
  **aMC@NLO** directory: `Cards/run_card.dat` in the classic format, not
  `run_card.toml`.
- **The first NLO output of an installation compiles CutTools, IREGI and
  StdHEP** (several minutes, once). Measured afterwards on 18 cores: about
  1 min to generate p p > z j, and 30 s-1.5 min per fixed-order run of
  p p > t t~ or p p > z j (at `req_acc_FO` 0.01 down to 0.001; a tighter
  target costs less than you would expect). The precision of
  distributions is another matter, see `reference/plots.md`. Larger
  multiplicities cost much more: start small and extrapolate.
- `p p > z j` gives a **stable on-shell Z** (its width is set to zero, so
  there is no branching ratio). If the user wants leptons, generate
  `p p > e+ e- j [QCD]` instead, and say which one you did.

## Flavour scheme: decide it, then state it

| Model | Proton and jets | Default PDF | α_s(M_Z) |
|---|---|---|---|
| `sm` (default) | g u d s c, **no b** (4-flavour, massive b) | 334700 = NNPDF40_nlo_as_01180_nf_4 | 0.1126 |
| `sm-no_b_mass` | g u d s c **b** (5-flavour) | 331700 = NNPDF40_nlo_as_01180 | 0.118 |

MadGraph picks the PDF to match the model unless `lhaid` is set by hand.
Rule when the user did not say:
- V+jets, H+jets, dijets and other light-parton processes: **5F**
  (`sm-no_b_mass`), which is the standard choice.
- b quarks in the final state (Z b b~, t t~ b b~, b b~): **4F** (`sm`), which
  keeps the b mass.
- Top pairs: either. The difference is small, so keep the default and state it.
- W+W− and other processes where a 5F b-initiated real emission contains a
  top resonance (g b → W W b): **4F**, which keeps that top contribution
  out of the NLO correction.

The `WARNING: Use of multiparticles is non-trivial for NLO process generation
...` printed at generation is about exactly this choice. Once you have
followed the rule above, it needs no action.

## Modes: always state the mode

The first lines after `launch` select the mode. Without them the run
**reuses the mode of the previous launch** of that directory.

| Mode | Switch lines | Shortcut | Log says | Result |
|---|---|---|---|---|
| NLO cross section and histograms | `order=NLO` + `fixed_order=ON` | `nlo` | `will run in mode: NLO`, then `Doing fixed order NLO` | `Events/run_XX/` |
| LO on the same setup (K-factor) | `order=LO` + `fixed_order=ON` | `lo` | `will run in mode: LO`, then `Doing fixed order LO` | `Events/run_XX_LO/` |
| NLO events + Pythia8 shower (MC@NLO) | `order=NLO` + `fixed_order=OFF` + `shower=PYTHIA8` | `aMC@NLO` | `will run in mode: aMC@NLO`, then `Generating events` | LHE + HepMC |
| NLO events, no shower | `order=NLO` + `fixed_order=OFF` + `shower=OFF` | `noshower` | `will run in mode: noshower` | LHE only |

Prefer the switch lines. `aMC@NLO` used to be rejected in scripts (fixed in
`extended_cmd.py`), and older installations still reject it. The switch table
echoed in the log is printed **before** your switch lines are read, so it may
show `fixed_order = OFF` even when you set it ON. Trust the `will run in
mode` and `Doing fixed order` lines.

## The trap specific to the launch block

If a line is not understood in the switch question (a wrong mode word, a
typo), MadGraph prints `This answer is not valid for current question` and
runs **with the previous settings**. All the lines that follow then execute
**after** the run, at the main prompt, where `set nevents ...` crashes. So a
wrong line costs a full run with the wrong settings. Check every launch:

```bash
grep -nE "not valid for current question|will run in mode|Doing fixed order|Generating events|INFO: modify" run.log
```

Every `set` must have its `INFO: modify parameter ...` line.

**Progress** (the mg7 `integral:` lines do not exist here):

```bash
grep -E "Idle:|Total cross section|Doing fixed order|Generating events" run.log | tail -3
```

## Settings (`Cards/run_card.dat`, classic names)

All classic names work at NLO: `set ptj 20`, `set ebeam1 6800`, `set lhaid ...`.

| Goal | Command | Default / remark |
|---|---|---|
| collider energy | `set lhc 13.6` (TeV) or `set ebeam1 6800` + `set ebeam2 6800` | 13 TeV |
| seed | `set iseed 11` | the default 0 is **not random**: it is a fixed seed, despite the card's comment ("0=assigned automatically"). The card is reset to 0 after each run, and the seed used is in the run's banner |
| fixed-order precision | `set req_acc_FO 0.002` | 0.01. The quoted MC error under-covers, see "Precision" below |
| events | `set nevents 10000` | event runs only |
| PDF | `set lhaid <id>` | chosen by the model (see the flavour-scheme table above); set it by hand only for another PDF family |
| PDF band | `set reweight_pdf True` | **off by default** |
| scale band | `set reweight_scale True` | on: 9-point envelope, muR and muF x 0.5/1/2 independently |
| scale choice | `set dynamical_scale_choice 3` (or 1, 2, 4) | -1 = default = H_T/2, half the sum of the final-state transverse masses |
| model parameters | `set param_card <block> <id> <value>` (e.g. `set param_card sminputs 1 127.9`), `set mt 172.5` | derived parameters (M_W in `sm`) follow |
| fixed scale | `set fixed_ren_scale True`, `set fixed_fac_scale True`, `set muR_ref_fixed 172.5`, `set muF_ref_fixed 172.5` | |
| jets | `set jetalgo -1`, `set jetradius 0.4`, `set ptj 30`, `set etaj 2.5` | **kT with R = 0.7**, ptj 10, no eta cut. Use anti-kT with R = 0.4 unless the user says otherwise |
| leptons | `set ptl ...`, `set etal ...`, `set mll_sf 60`, `set drll ...` | mll_sf 30, the others off |
| photons | `set ptgmin ...`, `set etagamma ...`, Frixione isolation `R0gamma`, `xn`, `epsgamma` | ptgmin 20 |

**Jet cuts at NLO are not LO generation cuts.** `ptj`/`etaj` act only on
processes that have jets at Born level. They then require the Born number of
jets (clustered with `jetalgo`/`jetradius`) to pass, so the result is
inclusive (Z + ≥1 jet for p p > z j). They do nothing for p p > t t~, and they
cannot be used to cut on the extra emission. Any other cut must be IR-safe.
Never put a cut on a single parton, on the real emission, or one that vetoes
soft or collinear radiation.

## Results

Summarise with the bundled script, using the absolute path to the
installation:

```bash
python3 <skill>/scripts/run_summary.py PROC_zj_nlo run_01
```

It prints the cross section to quote, the bands, the seed, the scale choice,
and **only the cuts that can act on this final state**.

- **Fixed order**: `Events/run_XX/summary.txt`.
  - Quote its `Total cross section`. The scale and PDF blocks print a slightly
    different central value "computed from histogram information"; apply
    their percentages to the total.
  - `res_1.txt` has the per-channel breakdown.
  - The default analysis fills only a couple of plots (`MADatNLO.HwU` /
    `.pdf`). Physics distributions need a user analysis in
    `FixedOrderAnalysis/`, selected in `Cards/FO_analyse_card.dat`.
- **Events**: `summary.txt` has only the total. The script computes the scale
  and PDF bands from the event weights.
  - Events have **negative weights**: 6-7% for Drell-Yan, more for processes
    with jets. This is normal. Report the fraction, and tell the user to keep
    the signs when filling histograms.
  - Events without a shower (`noshower`) are **not physical** on their own:
    the MC@NLO counterterms assume the shower that follows. Say so if you
    deliver them.
  - Showered events are in `Events/run_XX/events_PYTHIA8_0.hepmc.gz`. The
    default shower card has hadronisation on and the **underlying event (MPI)
    off** (`ue_enabled = F` in `Cards/shower_card.dat`). Mention it.

## Precision: seeds, combination, K-factor

A single fixed-order run's MC error is typically 20-40% too small. For
p p > z j, three NLO seeds gave 8054 ± 47, 8055 ± 51 and 8175 ± 50 pb.
Whenever the number matters (a K-factor, a comparison, anything below about
1%), run 3-5 seeds (one launch block per seed) and combine them:

```bash
python3 <skill>/scripts/run_summary.py --combine PROC_zj_nlo run_01 run_03 run_04
```

This gives the weighted mean, with its error scaled by sqrt(χ²/ndf) when the
runs scatter more than they claim (PDG prescription).

**K-factor**: run `lo` in the same directory, which uses the same NLO PDF and
α_s at both orders; that is one valid convention, and you should say so.
Combine the seeds of each order separately, then compute
K = σ_NLO / σ_LO, with δK/K = sqrt((δσ_NLO/σ_NLO)² + (δσ_LO/σ_LO)²) (MC only).

Examples, at 13.6 TeV with the defaults:
- p p > t t~: 691.5 / 475.8 pb, so K = 1.45. The scale band shrinks from
  +28/-20% to +9/-10%.
- p p > z j (4F, anti-kT 0.4, pT > 30, |η| < 2.5): K = 1.367 ± 0.007.

## Log lines

These are normal at NLO, and some are worth reporting:
- `WARNING: update the strong coupling value (alpha_s) to the value from the
  pdf selected: 0.1126...`: α_s is taken from the PDF set. Report it, since
  it is not 0.118 in 4F.
- `INFO: For gauge cancellation, the width of 'Z' has been set to zero`:
  expected for an external Z, W or top.
- `run_card missed argument X. Takes default: ...`: harmless.
- `WARNING: You have chosen not to run a parton shower. NLO events without
  showering are NOT physical`: true. Pass it on to the user if they asked
  for events.
- `WARNING: No HepMC3 installation was found ... uses HepMC2 instead`:
  harmless; the HepMC file is version 2.

## Reporting an NLO result

- The NLO cross section with its MC error (combined over seeds if you ran
  several), the scale band (9-point envelope) and the PDF band (with the set
  and how many members).
- The flavour scheme with its PDF and α_s, the scale choice, the jet
  algorithm and radius, and the cuts that act.
- Whether heavy particles are stable (no branching ratio).
- That this is fixed order, or NLO+PS with negative weights.
