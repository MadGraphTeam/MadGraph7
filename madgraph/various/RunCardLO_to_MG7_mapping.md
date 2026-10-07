# RunCardLO → RunCardMG7 conversion — feasibility map

Goal: an auto-converter that takes a legacy LO `run_card.dat` (`banner.RunCardLO`,
~220 parameters) and produces the equivalent `run_card.toml` (`banner.RunCardMG7`),
so the *same* physics setup can be run/compared in both, and so we can offer a
"retro-compatible" launch mode.

This file is the **analysis done before implementation**: it classifies every LO
parameter as portable / convertible / partial / not-representable, and lists what
the MG7 card would need before a given item can be carried over faithfully.

Legend for the mapping:
- **[=]** direct: same meaning, at most a rename → safe to port.
- **[~]** convertible: needs a value transform (documented) → port with a helper.
- **[!]** partial: LO is finer-grained than MG7 can express → port the common
  case, warn on the rest.
- **[x]** not representable: MG7 has no equivalent yet → cannot port; needs an MG7
  feature first (or must warn/ignore).
- **[mg7-only]** exists only in MG7 (no LO source) → keep MG7 default.

---

## 1. Beams / collider

| LO parameter(s) | MG7 target | class | notes |
|---|---|---|---|
| `ebeam1`, `ebeam2` | `beam.e_cm` | [~] | `e_cm = ebeam1 + ebeam2` for a head-on collider. Asymmetric beams keep e_cm but lose the boost. |
| `lpp1`, `lpp2` | `beam.leptonic` | [!] | `leptonic = |lpp| in {3,4}` (e/µ). MG7 only stores hadronic-vs-leptonic; it cannot express antiproton vs proton (`lpp=±1`), elastic photon (`lpp=2`), EVA (`lpp=±3/±4` variants), plugin (`lpp=9`) or the no-PDF fixed-energy beam (`lpp=0`). |
| `polbeam1`, `polbeam2` | — | [x] | beam polarization: no MG7 field. |
| `nb_proton1/2`, `nb_neutron1/2`, `mass_ion1/2` | — | [x] | heavy-ion beams: no MG7 field. |

## 2. PDF

| LO parameter(s) | MG7 target | class | notes |
|---|---|---|---|
| `pdlabel='lhapdf'` + `lhaid` | `beam.pdf` | [~] | map `lhaid` → LHAPDF set *name* (via `pdfsets.index`). |
| `pdlabel` built-in (`nn23lo1`, `cteq6l1`, …) | `beam.pdf` | [~] | map the ~10 built-in labels → their LHAPDF set name. |
| `pdlabel1`, `pdlabel2` (per-beam / `mixed`) | `beam.pdf` | [!] | MG7 has a single `pdf`; different PDFs per beam cannot be expressed. |
| `pdlabel='none'` / `lpp=0` | — | [!] | MG7 always uses a PDF grid; "no PDF" beams aren't expressible. |

## 3. Scales

| LO parameter(s) | MG7 target | class | notes |
|---|---|---|---|
| `fixed_ren_scale` | `beam.fixed_ren_scale` | [=] | |
| `scale` | `beam.ren_scale` | [=] | rename |
| `dsqrt_q2fact1`, `dsqrt_q2fact2` | `beam.fact_scale1`, `beam.fact_scale2` | [=] | rename |
| `fixed_fac_scale` (and `_scale1/_scale2`) | `beam.fixed_fact_scale` | [!] | MG7 has one flag; LO can fix beam 1 and beam 2 independently. |
| `dynamical_scale_choice` (int) | `beam.dynamical_scale_choice` (str) | [~] | clean int→string table: `1`(ΣEt)→`transverse_energy`, `2`(HT=Σ transverse mass)→`transverse_mass`, `3`(HT/2)→`half_transverse_mass`, `4`(partonic CM energy)→`partonic_energy`. `-1`(CKKW back-clustering, LO default) and `10` have no MG7 equivalent → fall back to MG7 default `half_transverse_mass` [!]; `0`(user hook) → [x]. |
| `scalefact` | `beam.scale_factor` | [=] | rename; multiplies the dynamical scale only, as in `setscales.f`. |
| `mue_over_ref`, `mue_ref_fixed`, `fixed_extra_scale` | — | [x] | EW-scale extras: no MG7 field. |

## 4. Generation / run

| LO parameter(s) | MG7 target | class | notes |
|---|---|---|---|
| `nevents` | `generation.events` | [=] | rename |
| `gridpack` | `gridpack.save_gridpack` | [=] | |
| `run_tag` | `run.run_name` | [!] | close but not identical semantics (tag vs run name). |
| `bwcutoff` | `phasespace.bw_cutoff` | [=] | rename |
| `cut_decays` | `phasespace.cut_decays` | [=] | decay-chain products uncut when false (default in both) |
| `SDE_strategy` (int 1/2) | `phasespace.sde_strategy` (str) | [~] | `1`(single-diagram enhanced)→`diagrams`, `2`(product of denominators)→`denominators`. |
| `maxjetflavor` | `multiparticles.jet` | [~] | rebuild the jet pdg list from maxjetflavor (± up to N + gluon). |
| `use_syst` / `systematics_*` | `generation.systematics` (bool) | [!] | MG7 only has on/off; the systematics program/arguments/pdf/scale sets are lost. |
| `iseed` | — | [x] | no seed field in the MG7 card (seed handled elsewhere). |
| `nhel`, `limhel`, `hel_*` | — | [x] | helicity-sampling controls: no MG7 field (MG7 has `dummy_matrix_element` only). |

## 5. Cuts

MG7 expresses cuts as `<group>[-<group>]-<observable>.{min,max}` over the groups
`jet, bottom, lepton, missing, photon` and observables `pt, eta_abs, delta_r,
mass, pair_mass, sfos_pair_mass, sqrt_s`. Mapping of the common LO cuts:

| LO parameter(s) | MG7 target | class |
|---|---|---|
| `ptj`/`ptjmax`, `ptb`/`ptbmax`, `pta`/`ptamax`, `ptl`/`ptlmax` | `jet-pt`, `bottom-pt`, `photon-pt`, `lepton-pt` `.min/.max` | [=] |
| `misset`/`missetmax` | `missing-pt.min/.max` | [=] |
| `etaj`, `etab`, `etaa`, `etal` | `<grp>-eta_abs.max` | [~] (LO η-max → eta_abs.max) |
| `drjj`, `drbb`, `drll`, `draa`, `drbj`, `draj`, `drjl`, `drab`, `drbl`, `dral` (+ `*max`) | `<grp>[-<grp>]-delta_r.min/.max` | [=] |
| `mmjj`, `mmbb`, `mmaa` (+ `*max`) | `<grp>-pair_mass.min/.max` | [=] |
| `mmll` (+ `mmllmax`) | `lepton-sfos_pair_mass.min/.max` | [=] (same-flavour opposite-sign pairs, as in LO) |
| `dsqrt_shat`/`dsqrt_shatmax` | `sqrt_s.min/.max` | [=] |

Further LO cuts map onto the same syntax, because the groups (`[multiparticles]`), ordered
selections (`jet_1`), the `-sum-` prefix and the `-sum` suffix compose freely:

| LO parameter(s) | MG7 target | class |
|---|---|---|
| `etajmin`, `etabmin`, `etaamin`, `etalmin` | `<grp>-eta_abs.min` | [=] |
| `ej/eb/ea/el` (+ max) | `<grp>-e.min/.max` | [=] |
| `ptj1min..ptj4max`, `ptl1min..ptl4max` | `jet_N-pt`, `lepton_N-pt` `.min/.max` | [=] |
| `xptj`, `xptl` | `jet_1-pt.min`, `lepton_1-pt.min` | [=] |
| `htjmin/max` | `jet-pt-sum.min/.max` | [=] |
| `ihtmin/max` | `<all partons group>-pt-sum` | [~] needs a group with all quarks and the gluon |
| `ht2min..ht4max` | `jet_1-jet_2-pt-sum`, `jet_1-jet_2-jet_3-pt-sum`, ... | [=] |
| `ptllmin/max` | `<lepton+neutrino group>-sum-pt` | [~] needs a group |
| `mmnl`, `mmnlmax` | `<lepton+neutrino group>-sum-mass` | [~] needs a group |
| `ptheavy` | `<heavy group>_1-pt.min` | [~] needs a group |
| `pt/e/eta_{min,max}_pdg` | `<group of that pdg>-pt/e/eta_abs` | [~] needs a group |
| `mxx_min_pdg` | `<group>-pair_mass.min`, or `-sfos_pair_mass` with `mxx_only_part_antipart` | [~] |

Cuts that are **not representable** in the current MG7 cut engine ([x]):
- `cutuse = 1` (accept if any ordered jet pt cut passes; `0` is the default behavior).
- WBF cuts `xetamin`, `deltaeta` (opposite-hemisphere condition on the two hardest jets).
- photon isolation `ptgmin`, `r0gamma`, `xn`, `epsgamma`, `isoem`.

## 6. Matching / merging  — all [x]

`ickkw`, `xqcut`, `ktdurham`, `dparameter`, `ptlund`, `highestmult`, `ktscheme`,
`alpsfact`, `chcluster`, `pdfwgt`, `asrwgtflavor`, `clusinfo`, `auto_ptj_mjj`,
`pdgs_for_merging_cut`: MLM/CKKW(-L) merging is not implemented in MG7.

## 7. Bias — all [x]

`bias_module`, `bias_parameters`: no biasing in MG7.

## 8. Engine / technical (mostly not 1:1)

- LO run engine: `vector_size`, `nb_warp`, `vecsize_memmax`, `hard_survey`,
  `job_strategy`, `survey_splitting`, `survey_nchannel_per_job`,
  `refine_evt_by_job`, `second_refine_treshold`, `tmin_for_channel`,
  `disable_multichannel`, `hel_recycling/filtering/splitamp/zeroamp`,
  `gridrun`, `gseed`, `issgridfile`, `d`, `xmtcentral`, `mc_grouped_subproc`,
  `fixed_couplings`, `global_flag`, `aloha_flag`, `small_width_treatment`.
  → MG7 has its **own** engine knobs (`[vegas]`, `[generation]`, `run.device`,
  thread pools, `phasespace.mode/t_channel/flat_mode/…`). A few have loose
  analogues (`disable_multichannel`↔`phasespace.mode`, survey settings↔
  `generation.survey_*`/`[vegas]`), but most are **[x] engine-specific** and
  should just keep MG7 defaults, not be ported.
- Misc LO: `time_of_flight`, `allow_overshoot_events`, `bypass_check`,
  `python_seed`, `lhe_version`, `boost_event`/`me_frame`/`frame_id`,
  `event_norm`, `keep_log`, `custom_fcts`, `ievo_eva/evaorder/eva_xcut` → [x].

## 9. MG7-only (no LO source, keep default) — [mg7-only]

`run.device`, `run.cpu_mode`, `run.simd_vector_size`, `run.{cpu,gpu,combine}_thread_pool_size`,
`run.output_format`, `run.verbosity`, `run.dummy_matrix_element`,
`gridpack.{include_source,include_madspace,include_madspace_source}`,
`generation.{cpu,gpu}_batch_size`,
`generation.freeze_max_weight_after`, `generation.max_overweight_truncation`,
`generation.cut_efficiency_threshold`, `generation.max_cut_repetitions`,
all of `[vegas]`, `phasespace.{mode,t_channel,flat_mode,invariant_power,
simplified_channel_count}`, all of `[madnis]`.

---

## Summary / recommendation

**Safe to port now (the converter should cover these):**
- Beams: `ebeam1/2`→`e_cm`, `lpp`→`leptonic` (common case).
- PDF: `lhaid`/`pdlabel`→`pdf` (with a label/id → name table).
- Scales: `fixed_ren_scale`, `scale`→`ren_scale`, `dsqrt_q2fact1/2`→`fact_scale1/2`,
  `dynamical_scale_choice` (int→str table), `fixed_fac_scale`, `scalefact`→`scale_factor`.
- Generation: `nevents`→`events`, `gridpack`, `bwcutoff`, `sde_strategy`,
  `maxjetflavor`→jet multiparticle, `use_syst`→`systematics`.
- Cuts: pt/eta(max)/deltaR/mass/sqrt_s for jet/bottom/lepton/photon/missing.

**Port with a warning (partial):** per-beam factorization flags, mixed/per-beam
PDF, η-min cuts, run_tag, systematics detail.

**Cannot port — emit a clear "not supported in mg7" warning and skip:** beam
polarization, heavy ion, matching/merging (`ickkw`/`xqcut`/`ktdurham`/…), bias,
photon isolation, `cutuse = 1`, the WBF cuts, helicity controls,
scale-variation extras, LO-engine technical knobs.

**Suggested implementation shape (later):**
- `RunCardMG7.from_LO(run_card_lo)` classmethod (or `RunCardLO.to_mg7()`),
  driven by an explicit mapping table (`{lo_name: (mg7_key, transform)}`) plus
  cut and pdf/scale helpers.
- Collect every LO parameter that is **user_set** but falls in the [x]/[!]
  buckets and report them together as "not (fully) transferable to run_card.toml",
  so the retro-compatible mode is transparent about what it dropped.
