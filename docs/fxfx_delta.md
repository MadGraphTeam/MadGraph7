# FxFx merging with MC@NLO-Delta matching

This note describes the `FxFx-Delta` branch. It combines FxFx merging
(`ickkw=3`) with MC@NLO-Delta matching. It also adds an optional
backward-ISR correction to the Delta no-emission probability. The shower
interface itself is described in `docs/mcatnlo_shower_interface.md`.

## Enabling it

Set these values in the NLO run card:

| Setting | Value |
|---|---|
| `ickkw` | `3` |
| `mcatnlo_delta` | `True` |
| `parton_shower` | `PYTHIA8` |

The usual FxFx requirements still apply: one multiplicity sample per
jet multiplicity (`@0`, `@1`, ...), `ptj` in the run card, and `Qcut` and
`njmax` in the shower card. `AddInfoLHE` must stay on, because FxFx events
need the `#aMCatNLO` information line.

## Event weights

The native-history no-emission probability `Delta_b` is the physical
Delta probability of the selected Born history.

- Each MC counterterm and each G term is multiplied by `Delta_b` once.
- The real emission `R_b` is split into `Delta_b R_b` in H events and
  `(1 - Delta_b) R_b` in S events. Both pieces keep the same real-state
  FxFx Sudakov factor and the same coupling and PDF scales, so their sum
  is the ordinary FxFx real integrand.
- The existing FxFx factors are unchanged. These are the Born-state and
  real-state Sudakov factors on the MC and G terms, the expanded FxFx
  Born subtraction and the H repartition between outer FKS sectors.

The weights are assembled in `compute_native_NLOPS_weights` and its
helpers in `fks_singular.f`.

## Shower scales

### S events

FxFx fixes the S-event starting scale. It is the smallest Born
clustering scale times `shower_scale_factor`, with no damping draw. Delta
evaluates its no-emission probability with Pythia's directed dipole
limits of that scale. The hard damping interval of the subtraction stays
`[Q, Q+3 GeV]`. A native history takes its Born scale from the
clustering cache, also when its Born provider belongs to another
subprocess (`set_FxFx_shower_scale_nbody` in `scale_module.f90`).

For the first emission these dipole limits reproduce the phase-space
ceilings already present in ordinary FxFx and in its MC subtraction. The
analytic FxFx Sudakov factor and its expansion keep their endpoints.
Since `Delta = 1 + O(alpha_s)`, the ordinary first-order compensation is
preserved. This argument relies on the common FxFx boundary. Independent
S-event starting scales would need a matching change in the analytic
Sudakov factor. The argument does not establish the logarithmic accuracy
of the combination at higher orders.

### H events

An H event with reconstructed Delta stopping scales gets the unweighted
geometric mean of its positive directed starting scales as `SCALUP`:

```
SCALUP = exp(sum(log(Q_ij)) / N)
```

Each active directed matrix entry counts once. Diagonal, absent and
nonpositive entries are excluded. There is no Born hard-scale floor. The
mean is taken after the selected outer colour flow is restored and
before the inner histories are evaluated (`set_FxFx_delta_H_scale` in
`scale_module.f90`, called from `driver_mintMC.f`).

When no Delta reconstruction or no positive entry exists, the ordinary
FxFx fallback applies. That fallback is the larger of the real-emission
dipole bound and the Born hard scale, or only the former without a Born
flow. Ordinary FxFx always uses this fallback. The scalar scale and the
colours follow the same outer sector and fold. H contributions with
different scalar scales stay distinct during unweighting (`emsca_H_hard`
in `fks_singular.f`).

The geometric mean is a scalar approximation. Pythia picks the largest
candidate evolution scale among competing channels and then lowers the
ceiling for all of them, so changing one channel's upper limit changes
the chance that another channel wins. A single scalar cannot reproduce
separate directed limits. It raises some channels' limits and lowers
others. This affects higher-order shower terms. The previous hard
scalar, however, allowed radiation far above the reconstructed stopping
scales.

### Alternative: the smallest scale with Sudakov factors

Set `#FxFxDeltaHMinScale` to `.true.` in `Cards/FKS_params.dat` to use
the smallest positive directed starting scale `Q_min` as `SCALUP`
instead. It is off by default and only affects FxFx-Delta H events with
a Delta reconstruction. The fallback, the S events and standalone
MC@NLO-Delta are unchanged.

With separate directed starts, each H-event colour line `(a,b)` would
radiate between its own start `t_ab` and `Q_min`. The scalar start
removes that radiation, so every H contribution of the event is
multiplied by the probability that none of it happens:

```
P_H = prod_a P_a(t_ab -> Q_min)
```

Each leg `a` of the H event contributes its colour lines with Delta's
line formula (eqs. 3.31 and 3.34) and Sudakov tables. The H-event
flavours, Bjorken x and dipole masses are used. ISR lines use the
beam-pair mass and FSR lines the mass of the line, capped as in Delta.
Incoming legs include their PDF ratio, and `DeltaBackwardISR` also
applies to them. A line that starts at `Q_min` does not contribute.

The Sudakov tables are Monte Carlo estimates of the no-emission
probability, which cannot decrease with the scale. For final-state lines
of heavy systems (large dipole masses) they are noisy at low scales,
with statistical zeros and dips. Delta's S events divide by the table at
their starting scale, which lies above that region; here both `t_ab` and
`Q_min` can lie inside it, and the noise would make a ratio exceed one or
divide by zero. The H-event lines therefore read the tables through their
monotone envelope, the largest table value at or below the scale
(`pysudakov_envelope`). For a monotone table this is the table value.
A line that the table does not resolve at all, zero at and below `t_ab`,
gets zero, as an unresolved stopping scale does in Delta. The S-event
Delta is unchanged.

`P_H` is computed once per physical H event, for the outer history and
colour flow that also own `SCALUP` (`delta_H_min_probability` in
`montecarlocounter.f`). It multiplies all H records of that event,
including the inner-history H sum of `MCExplicitKLSum`
(`scale_H_records` in `driver_mintMC.f`). The real emission in S events,
`(1 - Delta_b) R_b`, is not multiplied, so the H events lose a
second-order amount of cross section. Unlike the geometric mean, this
prescription never lets the shower radiate above any directed limit.
Changing the switch changes the event weights, so a new integration is
needed.

Validation with `p p > w+ z [QCD] @0` plus `p p > w+ z j [QCD] @1` (13 TeV,
FxFx, Pythia8, 10^6 events each) compared against the geometric mean
with all other cards equal:

| | Geometric mean | Smallest scale |
| --- | ---: | ---: |
| Merged rate (pb) | 30.938 ± 0.054 | 31.106 ± 0.053 |
| @0 / @1 (pb) | 20.930 / 10.008 | 20.859 / 10.248 |
| Signed integral (pb) | 49.47 ± 0.11 | 49.24 ± 0.07 |
| Negative fraction, LHE / merged | 17.6% / 9.0% | 16.9% / 8.1% |
| @1 H events | 85,359 | 66,365 |
| @1 H `SCALUP` p01 / median (GeV) | 0.88 / 14.4 | 2.25 / 15.4 |

The merged pT(WZ) integrals agree within about 1%. The sub-GeV tail of
H-event starting scales is removed by the Sudakov factor, and the number
of @1 H events drops by about 22%.

## Event format and shower interface

FxFx-Delta events use the ordinary FxFx format:

- `SCALUP` holds the scalar scale described above.
- The `<scales>` record carries the `pt_clust_i` matching tags and no
  directed `scalup_i_j` tags.
- Delta still enters the event weights, and its internal directed scales
  are used to compute the no-emission probability.

`write_events_lhe` (`write_event.f`) and `write_lhef_event`
(`handling_lhe_events.f`) write the directed `scalup_i_j` record only for
standalone MC@NLO-Delta. Both Fortran readers parse it with
`read_lhef_delta_scales`, which also handles multi-digit particle
indices. In `madgraph/various/lhe_parser.py`, `parse_matching_scale`
removes the `<scales>` tag from `Event.tag` and keeps its other
attributes verbatim, for example `mur`, `muf` and Delta's `scalup_i_j`.
They are written back together with the matching scales.

In `mcatnlo_shower.pythia8_settings`, Delta's dipole starting scales
(`Beams:setDipoleShowerStartingScalesFromLHEF`) are requested only
without FxFx. FxFx-Delta therefore uses Pythia's own
`JetMatchingMadgraph` through `CombineMatchingInput`, with
`Beams:setProductionScalesFromLHEF = on`, exactly as ordinary FxFx does.
It also runs with Pythia versions that lack the dipole setting. Pythia
attaches the matching scales to the LHE particles before reordering them,
so pre-decayed resonance daughters cannot shift a production parton's
matching label. No custom matching hook is needed. Standalone
MC@NLO-Delta keeps its directed shower scales.

## Optional backward-ISR correction

Set `#DeltaBackwardISR` to `.true.` in `Cards/FKS_params.dat` to enable
this correction. It is off by default. It applies to standalone
MC@NLO-Delta and to FxFx-Delta. It does not change the H-event scale
prescription or the endpoints of the FxFx analytic Sudakov factor.

For each supported incoming quark or gluon, the forward Sudakov-table
ratio and PDF ratio are replaced by Pythia's backward no-emission
probability. It comes from the LO identity `B = F * R * exp(C)`:

- `F` is the analytic, PDF-free forward probability.
- `R` is the grouped PDF ratio between the starting and stopping scales.
- `C` restores the PDF evolution for `Z(t) < z < 1`, outside the
  massless initial-initial shower boundary
  `Z = 1 - r/2 (sqrt(1 + 4/r) - 1)` with `r = t / m_dip^2`.

Every live incoming colour connection contributes, including both lines
of an incoming gluon. Pythia's ISR recoils against the other incoming
parton, so initial-final connections use the same scales and dipole mass
as initial-initial ones. The table is kept for final-state legs. It is
also kept for an incoming leg with a non-proton beam, a flavour group
mixing quarks and gluons, a stopping scale at or below the table's
1.01 GeV continuation, or a vanishing PDF. The code is
`delta_backward_leg` and its helpers in `montecarlocounter.f`.

The running coupling always comes from `pysudakov_isr_parameters` in the
active standalone `sudakov.f` provider. There are no coupling controls in
the FKS card, and the shower-card and hard-process couplings do not
override it. The shipped PDFmode-1 provider
(`vendor/SudGen/sudakov_pdfmode1.f`) declares its audited effective
profile: `alpha_s(MZ) = 0.1365`, one-loop running, CMW off, at most five
flavours, and thresholds of 1.5 and 4.8 GeV. This coupling is used only
for the no-emission probability. It does not change the hard matrix
elements, the hard PDFs, the FxFx analytic factors or the Pythia shower.

The exporter (`madgraph/iolibs/export_fks.py`) keeps the provider
metadata. Old or custom providers without this API get an explicit
unknown-status implementation, so they still work while the option is
off. Enabling the option with unknown metadata stops the run. The
prototype supports PDFmode 1, one- or two-loop running, CMW off,
five-flavour running and the thresholds above. Incompatible metadata
also stops the run rather than silently substituting another coupling.

The switch changes the paired H/S weights before unweighting. A new
integration, grid setup and event generation are therefore needed.
Reshowering existing LHE files does not apply it. The finite quadrature,
the grouped flavour treatment and the LO PDF-evolution identity are
approximations, particularly with NLO PDFs.

## Validation

All numbers below were obtained before the branch was merged into
MadGraph7, with the earlier Pythia8 steering script and Pythia 8.317.
They have not been repeated with the restructured shower interface.

### Single-event shower response

One saved `u g -> Z u` H event was showered 5,000 times per prescription
with Pythia 8.318. MPI, QED radiation, hadronization and the FxFx veto
were off. Its four directed H-event starting scales were 5.368 GeV and
three times 2.968 GeV, giving a geometric mean of 3.442 GeV. The
first-emission scale is Pythia's evolution scale.

| H-event prescription | First emissions above 6 GeV | Mean first-emission scale (GeV) |
|---|---:|---:|
| Separate directed starts | 0 / 5,000 | 2.667 |
| Previous scalar, 91.188 GeV | 4,701 / 5,000 | 32.20 |
| Geometric mean, 3.442 GeV | 0 / 5,000 | 2.598 |

### WZ + 0,1 jet

The setup was `p p > w+ z [QCD] @0` plus `p p > w+ z j [QCD] @1` at
13 TeV. It used `ptj = 10` GeV, `Qcut = 70` GeV, one merged jet, stable
W and Z bosons, and one million events per sample. MPI and hadronization
were off.

Geometric-mean H scales against the previous hard scalar, with the same
integration grids:

| Sample | Hard scalar (pb) | Geometric mean (pb) | Change |
|---|---:|---:|---:|
| @0 | 20.886 +/- 0.043 | 20.925 +/- 0.043 | +0.18% |
| @1 | 10.073 +/- 0.033 | 10.121 +/- 0.033 | +0.48% |
| Sum | 30.960 +/- 0.054 | 31.046 +/- 0.054 | +0.28% |

The median H-event `SCALUP` fell from 155 to 87 GeV in @0 and from 26 to
14 GeV in @1. The merged `pT(WZ)` rate between 20 and 70 GeV rose by
0.8%.

`DeltaBackwardISR` off against on, each freshly integrated:

| Sample | Off (pb) | On (pb) | Change |
|---|---:|---:|---:|
| @0 | 20.930 +/- 0.043 | 20.881 +/- 0.043 | -0.23% |
| @1 | 10.008 +/- 0.033 | 10.098 +/- 0.033 | +0.90% |
| Sum | 30.938 +/- 0.054 | 30.980 +/- 0.054 | +0.13% |

The negative-weight fractions were essentially unchanged: 17.6% before
showering and 9.0% after merging. The merged `pT(WZ)` rates in the 0-20,
20-70 and 70-150 GeV ranges changed by less than 0.1%. Between 400 and
1000 GeV the central values of `pT(WZ)`, the leading-jet `pT` and
`sqrt(d01)` shifted by 5-8%, with limited statistics and no significance
assigned.

The quoted errors are each run's conditional shower sampling errors. No
cross-run covariance was estimated. These comparisons measure one
merging-scale setup. They do not establish exact agreement with the
competing shower, merging-scale independence or logarithmic accuracy.

The full reports, plots, bin tables, cards and analysis scripts are kept
in the git history under `docs/audits/fxfx_delta_wz_h_scales`,
`docs/audits/fxfx_delta_wz_backward_isr` and `fxfx_diagnostics` at
commit `8dbb60493`.

## Tests

The FxFx-Delta and backward-ISR tests run in the `unittest_native_born`
CI job:

```
./tests/test_manager.py test_fxfx_delta_weights test_fxfx_delta_event_scales \
    test_fxfx_delta_lhe test_mcatnlo_delta_scales test_delta_backward_isr \
    test_sudakov_table test_mc_event_colours TestPythiaDeltaSteering \
    test_lhe_parser -t0
```

- `test_fxfx_delta_weights` covers 144 combinations of Delta
  probabilities, merging factors, cut outcomes and H/S evaluation modes,
  plus reuse of the native H cache.
- `test_fxfx_delta_event_scales` checks the physical S dipole limits, the
  H-event geometric mean and minimum, sentinel exclusion, sector and fold
  ownership, and the ordinary-mode controls.
- `test_mcatnlo_delta_scales` also checks the minimal-scale H probability
  against closed forms for final- and initial-state lines and across a
  table hole under a starting scale, and `test_fxfx_delta_weights` checks
  that only the event's H records are multiplied.
- `test_fxfx_delta_lhe` round-trips ordinary FxFx, FxFx-Delta and
  standalone Delta events through both Fortran readers and the Python
  parser.
- `test_delta_backward_isr`, `test_mcatnlo_delta_scales` and
  `test_sudakov_table` cover the running coupling, the backward identity,
  the colour-line factors, the fallbacks and the provider metadata.
- `TestPythiaDeltaSteering` checks the Pythia8 settings for MC@NLO,
  MC@NLO-Delta, FxFx and FxFx-Delta.

`test_fxfx_delta_hook` needs a real Pythia installation and is not part
of CI, because skipped tests fail the CI jobs. It checks stable Z bosons
and pre-decayed `Z -> e+ e-` records, with the daughters before or after
the production gluon:

```
PYTHIA8=/path/to/pythia8 ./tests/test_manager.py test_fxfx_delta_hook -t0
```

These tests validate the implementation, not merged cross sections or
logarithmic accuracy.
