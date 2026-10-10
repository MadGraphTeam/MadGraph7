# MLM merging (LO multi-jet samples)

Use this when the user wants an LO sample of a process with a variable
number of extra jets merged with the parton shower: V+jets, t t~ + jets,
H + jets. In this version **only the madevent output does MLM**: the mg7
output has no matching. So always use `output madevent`.

## The scripts

```
# gen.mg7
set crash_on_error True --no_save
import model sm-no_b_mass
generate p p > mu+ vm @0
add process p p > mu+ vm j @1
add process p p > mu+ vm j j @2
output madevent PROC_wjj_mlm
```

```
# launch.mg7
set crash_on_error True --no_save
launch PROC_wjj_mlm
shower=Pythia8
set nevents 7500
set iseed 31
set lhc 13.6
set ickkw 1
set xqcut 20
set JetMatching:qCut 30
done
```

- One `add process` per jet multiplicity, all with the **same** core process
  and the same `j` definition. The highest multiplicity is treated
  inclusively by the shower, the others exclusively. This is automatic.
- `ickkw 1` turns MLM on. `xqcut` is the minimum kT separation between
  partons at generation. An output with several multiplicities already gets
  `ickkw = 1` and `xqcut = 30` by default; set both explicitly anyway.
- `shower=Pythia8` is required: the MLM veto happens in the shower. Without
  it there is no merged sample and no merged cross section.
- **`nevents` counts parton-level events before the veto**, and the veto
  keeps about 65-70% of them (V+jets, xqcut 20, qCut 30). To deliver N
  merged events, ask for about N / 0.65. The summary script prints how many
  survived.
- The run card is the classic madevent one (`Cards/run_card.dat`): `set lhc
  13.6`, `set ebeam1 6800`, `set mmll 60`, `set ptl 10` all work.
- **Set `JetMatching:qCut` explicitly in every launch.** It is written into
  `Cards/pythia8_card.dat` and persists, so after a qCut scan the next
  launch would otherwise silently use the last scanned value.
- Measured on 18 cores: W/Z + 0, 1, 2 jets takes about 2 minutes per launch
  (5000-7500 events, integration and shower), and about 40 s per re-shower.
  Each extra jet multiplicity costs much more.
- **Disk**: about 75 kB per showered event in HepMC (370 MB for 5000 events,
  per shower).

## Which process

- `p p > mu+ vm` (+ jets) is the full final state: it includes off-shell W
  effects, and the lepton cuts apply to the real muon.
- `p p > w+, w+ > mu+ vm` (decay chain) forces an on-shell W (within
  `bwcutoff` widths).
- `p p > w+` with the decay left to the shower or to MadSpin keeps the W
  stable at matrix-element level.

Prefer the full final state unless the user asks otherwise, and state which
one you used.

## Settings that matter

| Setting | Where | Default | Remark |
|---|---|---|---|
| `xqcut` | `set xqcut 20` | 30 for a multi-jet output | V+jets at the LHC: 10-30 GeV; t t~ + jets: about 30 GeV |
| `JetMatching:qCut` | `set JetMatching:qCut 30` (Pythia8 card) | automatic: 1.5 x xqcut | keep it at or above 1.5 x xqcut (MadGraph warns below that) |
| `JetMatching:nJetMax` | automatic | highest jet multiplicity generated | |
| `maxjetflavor` | automatic from `p`/`j` | 4 with `sm`, 5 with `sm-no_b_mass` | must match the flavours in `j` |
| `auto_ptj_mjj` | run card | True | sets the generation cuts `ptj`/`mjj` from xqcut |
| `dynamical_scale_choice` | run card | -1 = CKKW back-clustering | the scale MLM needs; leave it |

- **Don't add jet cuts at generation** (`ptj`, `etaj`, `drjj`): the merging
  needs the soft and collinear region, and `auto_ptj_mjj` already sets what
  is needed. Jet cuts belong in the analysis of the showered events.
- Lepton and photon cuts (`mmll`, `ptl`, ...) are fine. Default lepton cuts
  (pT > 10 GeV, |η| < 2.5) apply, so report them.
- **Flavour scheme**: the rule of `reference/nlo.md` holds at LO too. V+jets
  is normally 5F (`sm-no_b_mass`, `maxjetflavor 5`). At LO the PDF is
  NNPDF40_lo_as_01180 (331900) in **both** schemes: NNPDF4.0 has no
  4-flavour LO set.
- **Progress**: madevent has no `integral:` lines. Poll with
  `grep -E "Idle:|Cross-section :|Matched cross" run.log | tail -3`.

## Results

```bash
python3 <MG7>/.claude/skills/madgraph/scripts/run_summary.py PROC_wjj_mlm
```

It prints:
- the parton-level cross section and its bands, both marked "before the
  MLM veto";
- the settings, and the cuts that act;
- one `matched` line per shower, with qCut, error, fraction kept and the
  number of merged events in its HepMC file.

What to take from it:
- **Quote the matched (merged) cross section**, not the parton-level one.
  For W+ → μ+ν + 0, 1, 2 jets: 8993 pb at parton level, 6074 pb matched.
- **Identify showers by their qCut, never by their tag number.** The tag
  counter (`run_tag`) carries over between runs. A new run's first shower
  can be `tag_3`, while `tag_1` of the same run is a later re-shower.
- The merged events are `Events/run_XX/tag_N_pythia8_events.hepmc.gz`
  (showered and hadronised). `unweighted_events.lhe.gz` is the unmerged
  parton-level input, and is not the merged sample.
- `tag_N_merged_xsecs.txt` only contains "Cross-sections could not be read
  ...": ignore it. The matched value is in the banner and in the log line
  `Matched cross-section`.
- **No DJR plots** (differential jet rates) are produced in this version:
  they came from the old MG5aMC_PY8_interface, which has been removed. Say so
  if the user asks for them. They can be computed from the HepMC file with a
  kT clustering (FastJet) if needed.
- **Scale and PDF bands** come from `parton_systematics.log`. They are
  computed on the parton-level events before the veto. Quote them as
  relative uncertainties of the merged result, and say so. They don't
  include the merging-scale uncertainty, which the qCut scan estimates.
- **Seeds**: `run_summary.py --combine PROC run_01 run_02` combines MLM runs
  per qCut, using the matched cross sections.

## The merging-scale uncertainty: qCut scan

Without DJRs, check how stable the matched cross section is under the
merging scale. Re-shower the same parton-level events with other `qCut`
values; each shower becomes a new tag of the same run:

```
# reshower.mg7
set crash_on_error True --no_save
launch PROC_wjj_mlm -i
shower run_01
pythia8
set JetMatching:qCut 45
done
```

Answer the `pythia8` line explicitly: the shower first asks which program to
run, and without it the `set` line is held over to the next question (it
still works, but the log then shows a "not valid for current question"
message).

How to quote the scan:
- Scan **upwards** from the nominal 1.5 x xqcut (e.g. 30, 45, 60 for xqcut
  20). Values below 1.5 x xqcut are not recommended.
- Quote the change relative to the nominal value over the range. For W +
  0, 1, 2 jets: +2.4% at 45 and +4.0% at 60, so "+4% over qCut 30-60", or
  ±2% around the middle.
- The points reuse the same parton events, so their MC errors are strongly
  correlated. A steady trend smaller than the per-point errors is still
  real.
- For a fuller estimate, also vary `xqcut` (with qCut = 1.5 x xqcut). That
  needs a new launch, because it changes the parton-level generation, but
  not a new output.
- A large jump between neighbouring points means the setup needs another
  look.
- Afterwards, the last scanned qCut is still in `Cards/pythia8_card.dat`. Set
  it explicitly in the next launch.

## Reporting an MLM result

- The matched cross section with its MC error (combined over seeds if you ran
  several), the multiplicities merged (e.g. W + 0, 1, 2 jets), xqcut and
  qCut, and the qCut spread.
- The parton-level scale and PDF bands as relative uncertainties (before the
  veto), the flavour scheme and the PDF, and the cuts that act, defaults
  included.
- That it is LO merged with Pythia8, with the Pythia8 defaults: hadronisation
  and the underlying event (MPI) on. The NLO shower card instead has MPI off.
- The HepMC file with its number of events, and that jet observables must be
  computed from it, not from the LHE file.
