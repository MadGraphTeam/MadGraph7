<!-- SPDX-License-Identifier: 0BSD -->

# One-loop QCD+QED on loop_qcd_qed_sm, coming from MadLoop

pyAmpliCol ships H.-S. Shao's two NLO UFO models with QCD and QED
counterterms (arXiv:1804.10017) as [UFOmodeldb](https://github.com/mg5amcnlo/UFOmodeldb)
ships them (commit 6ca4b120), with all their restriction cards, and with the
model patch of [UFOmodeldb issue 1](https://github.com/mg5amcnlo/UFOmodeldb/issues/1)
(V. Hirschi, [gist](https://gist.github.com/ValentinHirschi/81a26a3c9a03ca38cd2f6d6cf6a2b75f),
also posted on mg5amcnlo#475):

- `loop_qcd_qed_sm` (alpha(M_Z) input scheme), as downloaded and patched;
- `loop_qcd_qed_sm_Gmu` (G_mu input scheme), patched, as the mechanical
  Python-3 conversion of its Python-2 download (package-relative imports and
  `.iteritems()` -> `.items()` only; no declaration changes). The patch's
  four `eCoup_*_UV_EW.value[k] += '...'` statements are folded into those
  counterterms' constructors (same strings): the model loader certifies no
  value written after its constructor. Source checksums, the patch and the
  conversion are in `src/pyamplicol/assets/models/PROVENANCE.toml`.

The patch (1) splits the G0 t t~ R2 couplings `R2_G0tt_{d,s,b}` by loop
content, `[W+, G+, q]` and `[G+, q]`, keeping their sum (MadLoop, which inserts
an R2 coupling only for a loop of exactly its label, now inserts them: its
finite part of, e.g., `t t~ > z h` moves by up to 8 % and of `t t~ > t t~` by
up to 2.5 %; pyAmpliCol, which inserts every R2 vertex once, is unchanged); (2) makes the CKM
matrix of the `ckm` restriction exactly unitary (standard parametrisation from
the same four Wolfenstein inputs; it was truncated at O(lambda^3)); (3)
restores the bottom-mass term of the G_mu charge counterterm (with a massive b,
e.g. `u d~ > t b~` moves by 9.5e-5 x Born and now equals Recola); (4) adds the
missing `SMINPUTS 1` line to the G_mu `restrict_with_b_mass.dat`. UFOmodeldb's
commit 60ee58bb took item (1) for the b quark; the rest is local to the
shipped copies until UFOmodeldb takes it.

**Retained limitation.** The electroweak counterterms of both models omit the
tau mass. With a massive tau (the `ckm` restriction has MTA = 1.777) the single
pole of, e.g., `u d~ > e+ ve` keeps a residual
MTA^2 / (2 sw^2 (s - MW^2)) x (alpha/2pi) x Born (about 3e-5 x (alpha/2pi) x Born
at sqrt(s) = 500 GeV); MadLoop has the same. Fix `MTA` and `ymtau` to zero to
remove it.

UFOmodeldb's models (60ee58bb and the unpatched 6ca4b120) and the copies MG5_aMC's online model database
downloads (`import model loop_qcd_qed_sm`; they differ from UFOmodeldb's in
one coupling, `R2_ZZHH`, UFOmodeldb's fix of the R2 vertices Z Z H H and
Z Z G0 G0) take the same shipped `[loop.model]` profile, so pointing
`model.source` at such a copy works too and reproduces MadLoop on it. An
unpatched G_mu copy carries the defects the patch fixes (with a massive b, or
on the `ckm` restriction); `z z > h h`-type processes differ between the
download and UFOmodeldb's model.

A request with electroweak (`QED`) or joint (`QCD`, `QED`) corrections runs
on them with no `[loop.model]` section and no compatibility setting:

```toml
action = "generate"

[model]
source = "loop_qcd_qed_sm_Gmu"  # the shipped model; a path of that name wins

[process]
entries = [{ expression = "u u~ > e- e+", name = "dy" }]
loop_coupling_orders = ["QCD", "QED"]  # MG5_aMC: [virt=QCD QED]
coupling_order_policy = "minimal"

[loop.preparation]
fixed_parameters = { WT = "0", WZ = "0", WW = "0", WH = "0" }

[generation]
output = "artifacts/dy"
```

- **Scheme declaration.** pyAmpliCol ships the declaration of both models
  (QCD and QED corrections, complete UV and R2 counterterms,
  't Hooft-Veltman, on-shell electroweak renormalisation in the model's
  input scheme, real on-shell masses) for the exact sources: it applies to a
  model whose UFO declarations have the recorded digest, which the shipped
  copy, the reference download and every restriction share. A modified copy
  (a massive-muon variant, edited parameters) has other declarations and
  needs its own `[loop.model]`. An explicit `[loop.model]` always wins.
- **Source repair.** The models' legacy object library drops the Goldstone
  flag of `G-` (the antiparticle of `G+`). Model preparation restores it by
  structure and records it (`pyamplicol_source_repairs`); the former
  `compatibility = "gmu_particle_anti"` is not needed and is still accepted.
- **Gauge.** Born, loops and counterterms of an electroweak request take the
  't Hooft-Feynman gauge of the models' counterterms, with explicit
  Goldstones and ghosts.
- **Colour.** Full colour by default, as MadLoop. Leading and next-to-leading
  colour are refused for these models (`colour-declaration-coverage`: no
  shipped N_c declaration covers their scheme).
- **Tree level.** The same models serve tree requests (Born and real
  emission from one UFO).
- **Python 2.** A Python-2 UFO (the raw G_mu download) is refused with
  `ufo-python2`, naming MG5_aMC's conversion and the shipped model.

`pyamplicol self-test` evaluates an electroweak virtual (`e- e+ > e- e+`
[virt=QED] on `loop_qcd_qed_sm`) from a shipped prepared artifact and
compares it with MadLoop.

## MadLoop process syntax

| MadLoop / MG5_aMC | pyAmpliCol |
| --- | --- |
| `import model loop_qcd_qed_sm_Gmu-a0` | `[model] source = "loop_qcd_qed_sm_Gmu"`, `restriction = "a0"` |
| `[virt=QCD QED]`, `[virt=QED]` | `process.loop_coupling_orders = ["QCD", "QED"]`, `["QED"]` |
| `QED=2` (Born amplitude order) | `process.max_coupling_orders = { QED = 2 }` |
| `QCD^2==4`, `QED^2<=4` (squared orders) | every squared-order sector is a separate partial result: read it out (`result.for_order(...)`) |
| `/ x` | `generation.particle_veto = ["x"]`; a vetoed vector boson takes its ghosts. A veto that removes a loop species of a requested correction is refused on these models (`veto-unlabelled-counterterms`: their electroweak counterterms are not split by loop content), so photon-only QED is not expressible |
| `set complex_mass_scheme True` | not available yet (the complex-mass scheme is in progress); resonant widths are refused, see the recipes |
| `!a!` (a tagged photon) | `[loop] external_photons` (`"final"` by default, `"all"`, `"none"` or 1-based legs): a tagged photon couples with alpha(0) with the conversion term, which needs `restriction = "a0"` on `loop_qcd_qed_sm_Gmu`; an untagged one keeps the loops' coupling (MadLoop standalone's raw values: `"none"`). See [configuration](configuration.md#external-photons-alpha0-per-leg) |
| `p p > e+ e-` with a multiparticle label | the same expression: one artifact entry per subprocess with a Born; members without one are skipped, see below |

MG5_aMC syntax inside a process expression is refused with `madloop-syntax`.
MadLoop's default order selection for processes with Born amplitudes of
several orders drops the highest-alpha squared order unless the squared
orders are given; pyAmpliCol returns every squared-order sector of the
request's limits, so compare sector by sector.

## Recipes

| Need | Recipe |
| --- | --- |
| a real on-shell virtual (widths zero) | `[loop.preparation] fixed_parameters = { WT = "0", WZ = "0", WW = "0", WH = "0" }`; a width that propagates in a tree current of an electroweak request is refused (the message names the parameters). Use `fixed_parameters`, not the `no_widths` restriction of `loop_qcd_qed_sm_Gmu`, which also changes MT and MH |
| the parameters of a MadLoop run | the default values are the model's `restrict_default.dat` (on `loop_qcd_qed_sm_Gmu`: MT = ymt = 173.3 GeV); fix every parameter the run's `param_card.dat` sets differently, e.g. `fixed_parameters = { ..., MT = "173", ymt = "173" }`. A different top mass leaves the Born and the poles of `u u~ > z z` unchanged and moves only the electroweak finite part (through the counterterms), by about 1e-4 of the Born for 0.3 GeV |
| external photons (alpha(0)) | tagged photons: `loop_qcd_qed_sm_Gmu` with `restriction = "a0"` (it sets `aEW0M1`); `loop_qcd_qed_sm` has no alpha(0) input, so tagged photons with QED corrections are refused there (`external-photon-scheme`), as in MG5_aMC; untagged photons (`external_photons = "none"`) run on both |
| massive b | `restriction = "with_b_mass"` (`loop_qcd_qed_sm`: `with_b_mass_no_widths` sets the widths too) |
| non-diagonal CKM | `restriction = "ckm"` |
| resonant W, Z, H, t (complex-mass scheme) | not available yet |

## Multiparticle requests

A request with a multiparticle label generates its concrete processes that
have a Born amplitude within the coupling-order limits. A member without one
(`g g > e+ e-` in `p p > e+ e-`) is loop-induced: it is skipped, with a
typed record (key `loop-induced`) in the dry run
(`estimated_coverage.skipped_members`) and in the artifact (manifest
extension `loop_skipped_members`). The stored members' sum is MadLoop's
Born x virtual; the loop-squared contribution of a skipped member, which
MadLoop's multiparticle output holds, is not computed. A request none of whose
members has a Born is refused (`loop-induced`). `p` and `j` are the model's
massless quarks and gluon; add the photon with `process.multiparticles`.

## Refusals

Every refusal is typed: the message starts with a listed key, names the rule
and the remedy, and the exception carries `key`. `pyamplicol generate
--dry-run --json` reports what the model and the Born need and lists every
key with its remedy:

- `estimated_coverage.loop_configuration_refusals`: run-card settings and
  model declarations (decided before any tree);
- `loop_admission_refusals`: structures the generation refuses;
- `loop_internal_refusals`: engine conditions met during generation;
- `native_export_refusals` and `process_request_refusals`.

| Key | Request |
| --- | --- |
| `loop-model-declaration` | a model with no `[loop.model]` that is not a shipped profile (a modified copy) |
| `loop-model-corrections` | corrections the declaration does not cover (`QED` on loop_sm) |
| `loop-model-completion` | a declaration without complete UV or model R2 counterterms |
| `undeclared-corrections` | corrections the source does not expand perturbatively |
| `scale-parameter` | no identifiable renormalisation-scale parameter (e.g. fixed) |
| `veto-external-particle` | a `particle_veto` naming an external particle |
| `tree-only-model` | a compiled model without its NLO UFO source |
| `external-photon-scheme` | tagged on-shell photons with QED corrections on a model without the alpha(0) convention |
| `external-photon-legs` | an `external_photons` leg list naming a leg that is no on-shell photon |
| `feynman-gauge` | a model that cannot be taken in 't Hooft-Feynman gauge |
| `veto-unlabelled-counterterms` | a veto of a loop species of a requested correction on these models |
| `anomalous-veto` | a veto that breaks an anomaly-free multiplet (`generation.allow_anomalous_veto` overrides) |
| `colour-declaration-coverage` | LC or NLC without a covering colour declaration |
| `loop-induced` | no Born amplitude within the limits |
| `coloured-width` | a real on-shell request with a coloured particle of a width in a tree current |
| `electroweak-width` | a real on-shell electroweak request with any particle of a width in a tree current |
| `mass-scheme` | a `mass_scheme` that is neither `real-on-shell` nor `complex-mass` |
| `complex-mass-scheme`, `cms-internal-mass`, `cms-yukawa-input`, `cms-external-width` | a complex-mass request outside the transform's assumptions (see [Configuration](configuration.md)) |
| `complex-fixed-parameter` | a complex value in `[loop.preparation] fixed_parameters` |
| `ufo-python2` | a Python-2 UFO |
| `ufo-value-writes` | a UFO that writes declared values after their constructors (e.g. the UFOmodeldb issue-1 patch of `loop_qcd_qed_sm_Gmu` as published; the shipped copy has them folded) |
| `process-aliases`, `process-group-analysis` | permutation aliases or numerical process-group analysis of a one-loop request |
| `direct-r2`, `fft-basis-trace`, `colour-selectors`, `relation-discovery`, `evaluator-backend`, `evaluator-tuning`, `covariant-multiplets` | settings the one-loop engine does not take (each message names the setting) |
| `fermion-sign`, `reading-sign`, `unlowered-counterterm`, `counterterm-coupling`, `loop-kernel-catalogue`, `counterterm-completeness` | internal: an engine condition met during generation |

A process whose point inputs the native export does not cover is refused
with the export's key (`NativeExportRefused`, e.g. `plane-mismatch`). See
[UFO Model Coverage](ufo-model-coverage.md) for every one-loop refusal and
[One-loop QCD on loop_sm](loop-sm-and-madloop.md) for the QCD model.
