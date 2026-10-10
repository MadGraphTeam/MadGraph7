# The mg7 run card (`Cards/run_card.toml`)

An `output` (mg7) directory is configured by `Cards/run_card.toml`. Every
parameter is commented in the card itself. Read it (`grep -n` for a key) when
you need one that is not listed here. Each `set` line after `launch` edits
this file and the edit persists.

## How to write a `set`

- **Prefer the `section.key` form**: `set beam.pdf NNPDF31_lo_as_0118`.
  A bare key works only when it is unique: `set pdf ...` is rejected with
  `Ambiguous key 'pdf'`, because `[beam]` and `[systematics]` both have one,
  and the run then goes on with the old value.
- **Cuts always take the `cuts.` prefix**: `set cuts.jet-pt.min 30`. Without
  it, a cut that is not yet in the card is rejected (`invalid set command`).
- Values may use units and expressions: `set cuts.jet-pt.min 0.03 TeV`.
- Check the log for every `set`: an accepted one echoes a `madgraph7: set ...`,
  `madgraph7: modify ...` or `INFO: modify parameter ...` line. A rejected one
  prints `WARNING: invalid set command ...` and is **ignored**.

## Common settings

| Goal | Command |
|---|---|
| number of unweighted events | `set nevents 10000` |
| reproducible seed | `set iseed 42` (0 or -1 = random; the seed used is in `info.json`) |
| pp collider energy | `set lhc 13.6` (TeV), or `set beam.e_cm 13600` (GeV, sqrt(s)) |
| e+e- collider | `set lep 250` (GeV; also switches off the PDFs) |
| PDF set | `set beam.pdf <LHAPDF set name>`: a name, **not** an lhaid; it is downloaded automatically if missing |
| fixed scales | `set fixed_scale 91.188` (renormalisation and factorisation) |
| dynamical scale | `set dynamical_scale_choice <choice>`, where choice is `transverse_energy`, `transverse_mass`, `half_transverse_mass` (default), `partonic_energy`, or a shortcut like `HT/4` |
| scale factor | `set beam.scale_factor 0.5` |
| scale/PDF variations | on by default (`[systematics]`); `set systematics.enable False` to skip them |
| LHE output | `set output_format lhe` (default output is `events.npy` + `header.lhe`) |
| no plots / histograms | `set histograms off` |
| GPU | `set run.device ["cuda"]` (or `hip`) |
| MadNIS (ML phase space) | `set madnis.enable True` |
| unweighting efficiency | `set generation.max_overweight_truncation 0.01` (default 1e-3; larger is faster, but truncates the weight tail) |
| decay-width cutoff | `set phasespace.bw_cutoff 15` |

An existing npy run can be converted afterwards with
`<dir>/bin/npy_to_lhe` (run it with `--help` first).

## Cuts

A cut is `<groups>-<observable>.<min|max>`:

- **Groups** come from `[multiparticles]` in the generated card. They are
  always `jet`, `bottom`, `lepton`, `missing` (neutrinos) and `photon`, plus one
  group per other final-state particle type. A particle and its antiparticle
  share one group, named **without** the charge sign: `t` for t/t~, `w` for
  W+/W-, `z`, `h`. **Read `[multiparticles]` in the generated card before
  writing any cut**. A group that is not defined there makes the launch crash
  with `KeyError: '<name>'`. To cut on a new group, add it to
  `[multiparticles]` by editing the card (e.g. `heavy = [6, -6, 25]`) before
  launching.
- **Observables** (use these spellings; the comment in the card says `dR`
  and `eta`, but the keys are `delta_r` and `eta_abs`): `pt`, `eta_abs`,
  `eta`, `delta_r`, `mass`, `pair_mass`,
  `sfos_pair_mass` (same-flavour opposite-sign lepton pairs, like madevent's
  `mmll`), `sqrt_s` (all outgoing particles, no group).
  - Two groups give a mixed cut: `jet-lepton-delta_r.min`.
  - `-sum-` adds the momenta first: `lepton-lepton-sum-mass` is the mass of
    a lepton pair, and `lepton-sum-mass` is the mass of all leptons together.

Defaults: jet pT > 20, |eta| < 5, dR(jj) > 0.4; lepton pT > 10, |eta| < 2.5,
dR(ll) > 0.4, dR(jl) > 0.4. `set no_parton_cut` removes them all, including
the separation cuts that keep **massless** radiation finite. Re-add an
explicit cut for every photon/jet/massless lepton, or the run never
converges (see troubleshooting).

## madevent names and their mg7 equivalents

The mg7 launch understands `nevents`, `iseed`, `gridpack`, `fixed_ren_scale`,
`scalefact`, `scale`, `dsqrt_q2fact1/2`, `bwcutoff`, `cut_decays`, `use_syst`.
Most other madevent names are ignored with `invalid set command`:

| madevent | mg7 |
|---|---|
| `ebeam1/ebeam2 E` | `set beam.e_cm 2E` or `set lhc ...` |
| `lpp1/lpp2` | `set lhc ...` / `set lep ...` (`beam.leptonic`) |
| `pdlabel`, `lhaid` | `set beam.pdf <set name>` |
| `ptj` / `ptjmax` | `cuts.jet-pt.min` / `.max` |
| `etaj` | `cuts.jet-eta_abs.max` |
| `drjj` | `cuts.jet-delta_r.min` |
| `ptb`, `etab`, `drbb` | `cuts.bottom-pt.min`, `cuts.bottom-eta_abs.max`, `cuts.bottom-delta_r.min` |
| `ptl`, `etal`, `drll` | `cuts.lepton-pt.min`, `cuts.lepton-eta_abs.max`, `cuts.lepton-delta_r.min` |
| `pta`, `etaa`, `draa` | `cuts.photon-pt.min`, `cuts.photon-eta_abs.max`, `cuts.photon-delta_r.min` |
| `drjl`, `dral`, `draj` | `cuts.jet-lepton-delta_r.min`, `cuts.photon-lepton-delta_r.min`, `cuts.photon-jet-delta_r.min` |
| `misset` | `cuts.missing-pt.min` |
| `mmjj`, `mmbb`, `mmaa` | `cuts.jet-pair_mass.min`, `cuts.bottom-pair_mass.min`, `cuts.photon-pair_mass.min` |
| `mmll` / `mmllmax` | `cuts.lepton-sfos_pair_mass.min` / `.max` |
| `use_syst` | `systematics.enable` |
| `nb_core` | `run.cpu_thread_pool_size` |

## Other cards and switches (same launch block)

- Post-processing switches, written as `name=value` lines (no `set`):
  `shower=Pythia8`, `madspin=ON`. Turn a step off with `shower=OFF`.
  Switches persist in the process directory too.
- Pythia8 card: `set Main:numberOfEvents 100` (any `Key:sub value`).
- param card (masses, widths, couplings): `set mt 172.5`,
  `set width wt Auto`, `set param_card <block> <id> <value>`.
- MadSpin decays: `decay t > w+ b, w+ > e+ ve` (**no `set`**; with `set` it is
  rejected and MadSpin runs with the default card, which decays t, W and Z
  inclusively into its `light` multiparticle: quarks and all leptons). Check
  the `decay` lines of `Cards/madspin_card.dat` before trusting the sample;
  the default lines stay in the card, and yours are added below them.
