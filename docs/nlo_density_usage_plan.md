# Using the NLO spin-density matrix: density weights, analyses, MadSpin — plan

Status (2026-11-01): V0, V1 and V2 done (section 5). V3 and V4 are not
ready: each needs its own plan before it starts. Builds on `docs/nlo_density_matrix_plan.md`
(branch `claude/density-matrix-mode-plan-f2e101`). There, every NLO building
block has a spin-density matrix rho for the selected massive legs, checked
element by element in the soft and collinear limits (ME and MC
counterterms) and in the poles.

**Goals.**
1. **MadSpin with NLO spin correlations.** For each aMC@NLO event, compute
   the event's NLO density matrix from the event's weight information, and
   decay with it.
2. **A density analysis.** It keeps track of the density matrix, so that a
   HwU histogram gives the average density matrix of the sample in each
   bin. At fixed order first.

**Agreed order of work.**
1. The density pieces of every contribution, and the fixed-order density
   analysis built on them.
2. The method to get the density matrix of an event by combining the
   density weights, following the NLO reweighting method.
3. **Stop before MadSpin**, but with every interface MadSpin will need in
   place. MadSpin will use direct signed NLO acceptance (see U-D6).

## 1. Starting point

### 1.1 How NLO reweighting rebuilds a weight

**In the event.** An aMC@NLO event stores, per contribution, in `<mgrwgt>`
(`fill_rwgt_lines`, `fks_singular.f:3947`):
- `pwgt(1:3)`: the scale-independent piece and the log(mu_R^2/Q^2) and
  log(mu_F^2/Q^2) coefficients;
- the PDGs, `bjx`, `scales2`, `g_strong`, `QCDpower`, the momentum
  configuration, `itype`, `nFKS`;
- `ref_wgt`: the contribution's full weight;
- the bias.

**Rebuilding.**
- The python side parses this into `event.nloweight`
  (`lhe_parser.py:2938`, `OneNLOWeight` at `:5781`).
- NLO reweighting rebuilds any weight as
  `sum_c xlum_c(muF) · (pwgt0 + pwgt1 ln(muR²/Q²) + pwgt2 ln(muF²/Q²)) · g^p`
  (`combine_wgt_local`, `reweight_interface.py:1785`). For a new model it
  first multiplies `pwgt` by a matrix-element ratio.
- The Fortran `reweight_xsec_events` does the same for scale and PDF
  variations. It links no matrix element.

### 1.2 The projector (this branch)

**What it covers.** `DENS_W`/`DENS_W_ON` projects every matrix-element piece
the subtraction uses:
- the Born, the colour-linked Borns and the spin-correlated Born;
- the real emission;
- the MC counterterms (`get_mbar`);
- the MadLoop virtual.

Because the subtraction is linear in them, a contribution computed with W
set is exactly that contribution's density element. That is what the
per-element limit and pole tests checked.

**Limits:**
- standalone output only;
- refused for the extra counterterms;
- needs the full helicity sum;
- the real emission has no amplitude cache.

### 1.3 Why the density pieces must come from the generation

The rebuild of a weight needs no matrix element only because the pieces
`pwgt` are stored. A density element of a contribution is a different
linear combination of matrix-element objects:
- soft counterterms: colour-linked Borns;
- collinear and MC counterterms: B and the spin-correlated B~;
- the virtual: MadLoop plus I-operator links.

A single matrix-element ratio, as used for parameter reweighting, is not
exact for it: B~ and the links do not scale like B, element by element.
**So the density pieces are computed once, exactly, where the matrix
elements are evaluated, and stored next to `pwgt`.** After that,
everything downstream is a recombination, as in NLO reweighting:
- the central density;
- scale and PDF variations of the density;
- MadSpin's per-event density.

## 2. Design decisions

### U-D1 — Density pieces per contribution

For a density output run with density weights on, every contribution c
carries, besides `pwgt(1:3)`:
- **`pwgt_dens(1:3,K)`** for K = 1..N_COMB^2, the elements W_K of the
  hermitian basis:
  - `rho_II` for E_II;
  - `Re_rho_IJ` for E_IJ + E_JI;
  - `Im_rho_IJ` for i(E_IJ - E_JI).

  They are the same three pieces, projected on W_K.
- **`ref_dens(K)`**: the central density weight of the contribution, the
  analogue of `ref_wgt`, so that the central density needs no PDF library.

Identity: sum_I `pwgt_dens(:,rho_II)` = `pwgt` per contribution, and the
same for `ref_dens`. It is checked.

### U-D2 — Computing them: the projector loop

In each `compute_*` (Born, n-body-noborn including the virtual, real, soft,
collinear and soft-collinear counterterms, MC counterterms, and the
Sudakov contributions if present):
1. Evaluate as now, with W off.
2. Repeat the matrix-element part once per K with W = W_K. `add_wgt` in
   "density mode" adds into `pwgt_dens(:,K,icontr)` of the contribution
   created by the unprojected call. The calls come in the same order;
   check that itype, nFKS and amppos match.

`fks_singular.f`'s subtraction algebra is unchanged.

**Cost control.**
- The amplitudes are computed once per helicity and reused, so each extra K
  costs colour contractions plus subtraction bookkeeping:
  - Born: `saveamp`;
  - virtual: the JAMPs stored in MadLoop;
  - real emission: a **new JAMP cache**.
- Measured per process. If too slow:
  - a user list of elements (e.g. the diagonal only);
  - or, later, density-valued matrix elements.

### U-D3 — Storage in the event: `<mgdens>`

A new block, written next to `<mgrwgt>` and only in density runs.

**Format.**
- **Header line:** N_COMB^2, the legs (user numbering) and
  `me_frame`/`frame_id`.
- **Then one line per contribution**, in exactly the order of the
  `<mgrwgt>` contribution lines, holding `pwgt_dens(1:3,K)` and
  `ref_dens(K)` for every K.

**Why a separate block.** `<mgrwgt>`'s fixed format is read by the Fortran
`reweight_xsec_events` and by `lhe_parser`, and stays untouched. Readers
that do not know `<mgdens>` ignore it.

**Run header.** The banner and `<initrwgt>` carry the basis map: element K,
its kind (diag/Re/Im) and its combination pair (I,J). Each combination I
lists the helicity of each density leg, in the NLO side's order. The MadSpin
order differs (spin 1/2 is `[1,-1]` there), so the map is explicit.

**Unweighting.** When an event is unweighted, `ref_dens` is rescaled by
`XWGTUP/wgtref`, exactly like the `ref_wgt` the reweighting uses.

### U-D4 — The combination method

One method, with two implementations that must agree.

**Python** (`madgraph/various/lhe_parser.py`, next to `parse_nlo_weight`):
- `Event.parse_nlo_density()` reads `<mgdens>` and attaches `pwgt_dens` and
  `ref_dens` to each `OneNLOWeight`.
- `Event.get_nlo_density(pdf=None, mur_fact=1, muf_fact=1, pdf_member=None)`
  returns the event's density matrix as a full complex N_COMB x N_COMB
  numpy array, together with the basis map:
  - **central** (`pdf=None`, factors 1): rho = sum_c `ref_dens(c)`, with no
    PDF or alpha_s needed, so it works even without a python LHAPDF;
  - **variations:** the `combine_wgt_local` formula applied to each K, with
    `pwgt_dens` in place of `pwgt`.

  It is normalised so that Tr rho = the event weight. **Accessing the
  density means combining the density weights, exactly as NLO reweighting
  does.**

**Fortran** (`reweight_xsec_events`):
- read `<mgdens>`;
- build the density for the central scale and for each scale and PDF
  variation;
- optionally write them as `<rwgt>` columns `dens_*`, e.g. for the Pythia8
  HwU analyses, which receive `<rwgt>` columns automatically (at most 100
  slots).

### U-D5 — Fixed order: density columns

**At fixed order there are no events.** The same `pwgt_dens` are combined
in the integrand, by `include_PDF_and_alphas` and `reweight_scale`/`_pdf`,
into extra columns of `wgts`, after the scale and PDF ones, labelled
`dens <kind> <I,J>`. `fill_plots` merges them unchanged, since they are
linear, and `HwU_fill` histograms them.

**Per-bin errors.** HwU keeps the error of column 1 only, so:
- `HwU.f` gets, for the density columns, per-point sums of w_K^2 and
  w_K·w_1 (correlated points summed first, as for `histy2`);
- python computes the delta-method error of rho_K = sum w_K / sum w.

**Post-processing** (`madgraph/various/histograms.py`). A new `--density`
command turns the `dens` columns into rho per bin, and into derived
quantities:
- polarisation fractions;
- f_0, f_L, f_R for one vector boson;
- the spin-correlation matrix C_ij and the polarisations B_i (helicity
  basis) for two spin-1/2 particles.

### U-D6 — MadSpin (to do later; interface prepared now)

**Decision: direct signed NLO acceptance.**

**The density.** For each event, MadSpin calls `Event.get_nlo_density()`
(U-D4), at the run's central scale. That is the computation from the
weight information.

**Acceptance.**
- The acceptance weight is `Tr(rho_hat x D)/prod Tr D_i`, with
  rho_hat = rho/Tr rho and D the decay densities.
- Accept with its absolute value against the maximum weight; the sign is
  carried by the event weight.
- A non-positive rho_hat (H events, or S events with large virtual or
  counterterm parts) can give a ratio that is negative or above the LO
  bound. This needs:
  - a maximum-weight scan done on NLO densities;
  - a record of the fraction above the bound and of the negative ones;
  - `_check_production_density` relaxed (Tr rho may be negative: work with
    rho_hat).

**Modes.** `onshell` and `PA` only: the density is the one at the event's
on-shell kinematics. `madspin`/`full` recompute the production on reshuffled
momenta.

**Interfaces delivered now, so that MadSpin needs no further production-side
work:**
1. `Event.get_nlo_density()` and the basis map (U-D4).
2. A conversion helper `to_madspin_basis(rho, basis_map, decayed_legs)` that
   returns MadSpin's packed upper-triangle layout. It re-orders legs (by
   PDG and position; the real emission's order for H events) and the
   spin-1/2 helicity order. It refuses if a decayed particle is not a
   density leg.
3. The frame in the banner (`me_frame`), read with
   `frame_and_beampol_from_run_card`.

Unit tests cover 1 and 2 on stored events. Nothing is wired into
`interface_madspin.py` yet.

## 3. Milestones

| | content | depends on |
|---|---|---|
| V0 | density code in aMC@NLO production outputs | — |
| V1 | `pwgt_dens` per contribution (projector loop, caches) | V0 |
| V2 | fixed-order density columns and the HwU density analysis | V1 |
| V3 | `<mgdens>` storage and the combination method (python + Fortran). **NOT READY: needs its own plan before starting** | V1, V2 (its FO results are V3's reference) |
| V4 | MadSpin interfaces (U-D6 items 1-3), no MadSpin wiring. **NOT READY: needs its own plan before starting** | V3 |
| (later) | MadSpin direct signed NLO acceptance | V4 |

### V0 — production plumbing

- **Output.** `output DIR --density=i,j [--me_frame=...]` for the
  production aMC@NLO output, not only `standalone_fortran --fks`:
  - the same leg checks;
  - `density_info.inc`;
  - the `density` template context;
  - MadLoop density code;
  - `HelicityFilterLevel=1`.
- **run_card.** A switch `density_weights` (default False). False must give
  numbers identical to a plain output.
- **Refused at launch:**
  - extra counterterms;
  - MC helicity sampling of the Born or virtual (the full sum is forced);
  - an IR-unsafe `me_frame`.
- **Gate.** A production output without `--density` is byte-identical.

### V1 — density pieces per contribution

- `pwgt_dens(3,K,icontr)` in `weight_lines.f`.
- `add_wgt` density mode, and the projector loop in every `compute_*`.
- A JAMP cache for the real emission's projected path.
- **Gates:**
  - per contribution, sum of the diagonal elements = the unprojected
    weight, to rounding;
  - cost per phase-space point measured for N_COMB^2 = 9, 16 and 81, with
    `w+ j`, `t t~` and `w+ z`.

### V2 — fixed-order density analysis

- Density columns in `wgts`, `initplot` labels, `outfun`.
- `HwU.f` error sums.
- An analysis template `analysis_HwU_density.f`: p_T and rapidity of the
  density legs, pair mass and Delta-phi, jet p_T.
- `histograms.py --density`.
- **Gates:**
  - rho per bin is hermitian and positive within errors, with trace 1;
  - in `w+ j`, the W's f_0, f_L, f_R per bin agree with polarised NLO runs
    (`p p > w+{0} j [QCD]`, ...; `me_frame=[3]`) within errors;
  - (checked again in V3) the event-based density, histogrammed at LHE
    level, agrees with this FO one for observables insensitive to the
    shower and the S/H split.

### V3 — storage and the combination method

**NOT READY.** Write a dedicated plan, and have it agreed, before starting.
What follows is only the outline.


- Write `<mgdens>` (in `fill_rwgt_lines`/`write_lhef_event`), with the
  header map and the unweighting rescale.
- Python: `parse_nlo_density`, `get_nlo_density`.
- Fortran: the `reweight_xsec_events` density columns.
- **Gates:**
  - per event, Tr `get_nlo_density()` = `XWGTUP` (central);
  - with the PDF set, the python and Fortran variations agree;
  - per event, `rho_II/Tr` of an unpolarised `p p > w+ j` sample, averaged,
    agrees with the polarised cross-section fractions within statistics;
  - the existing NLO reweighting still works on density-run events;
  - the LHE-level histogram of the event densities agrees with the V2
    fixed-order result, for observables insensitive to the S/H split.

### V4 — MadSpin interfaces

**NOT READY.** Write a dedicated plan, and have it agreed, before starting.
What follows is only the outline.


- `to_madspin_basis` and the leg and helicity-order mapping.
- Banner frame read-back.
- Unit tests on stored density events: a full rho round trip; the packed
  layout identical to `get_density` for a Born event with the LO density
  injected.
- **Stop here.** No change to `interface_madspin.py`.

## 4. Risks and open questions

1. **Cost of the projector loop** (N_COMB^2 = 81 for `w+ z`). Measure in
   V1 before going on.
2. **Size of `<mgdens>`.** It is N_COMB^2 x 4 numbers per contribution line:
   81 x 4 x (about 5-20 contributions) per event for `w+ z`. Option: a
   per-run element subset.
3. **H events combine frames.** The real emission is in the real frame,
   the MC counterterms in the Born frame. They agree only in the limits,
   which is IR safe; at finite separation it is a convention at relative
   order alpha_s. Document it with the results.
4. **MadSpin with signed acceptance** (later). The maximum-weight bound and
   the efficiency on non-positive rho_hat are unknown until measured.
5. **IOTest goldens** change once V0 touches the production templates. Ask
   before regenerating; the 67 `matrix_N.f` goldens of the first plan are
   still pending.

## 5. Implementation log

### V0 — production plumbing (done)

- `output DIR --density=i,j [--me_frame=...]` for the production aMC@NLO
  output: the per-P-directory leg checks, `density_info.inc`, the `density`
  template context, the MadLoop density code, `HelicityFilterLevel=1`, and
  `me_frame` in the run_card (the density legs by default). Moved from the
  standalone class to `ProcessOptimizedExporterFortranFKS`; refused for the
  non-optimized loop output.
- run_card `density_weights` (hidden, default False) -> `run.inc`.
- A plain output (no `--density`) gives P directories identical to the
  parent commit: their makefile compiles the shared
  `SubProcesses/density_weights_dummy.f` (no element) instead of a
  `density_weights.f` (a stub in every P directory had broken the
  `test_ppzz_ewsudakov` IOTest).
- A density output without `density_weights` reproduces itself bit for bit
  (grid setup 5.223e4 and final 5.303e4 pb, `p p > w+ j [QCD]`), but not the
  plain output (5.192e4 +- 510): the matrix elements are evaluated in the
  frame of the density legs and the virtual from its colour flows, and see
  V1 for the real emission.

### V1 — density pieces per contribution (done)

**Mechanism.**
- `weight_lines.f`: `wgt_dens(3,K,icontr)`, `dens_lum(icontr)`, the rerun
  flags, `density_lines_ensure`.
- `driver_mintFO.f`: each `compute_*` is evaluated unprojected, then once
  per element K with W = W_K (`density_begin`, `density_rerun_start(K)`,
  `density_rerun_end`). The draw deciding whether the virtual is computed
  is reused (`dens_saved_do_virt`), `virtual_over_born` is restored, and the
  reruns do not feed the virtual-trick accumulators (`*_wgt_mint`).
- `add_wgt` in rerun mode adds into `wgt_dens(:,K,i)` of the contribution
  with the same itype, amppos, nFKS and orders tag (the next unused one);
  an unmatched one is counted and reported.
- `include_density_weights` appends one column per element after the scale
  and PDF ones, built exactly as `wgts(1,:)`; `initplot` labels them
  `dens rho_I_I`, `dens Re_rho_I_J`, `dens Im_rho_I_J`.
- `density_weights.f` (per P directory): the element map and
  `density_set_element(K)`; `density_virt.f` passes W to MadLoop
  (`DENSITY_SET_W`, which takes precedence over `density_w.dat`). It is
  linked next to `BinothLHA.o`; `BinothLHADummy.f` has the no-op version
  for the executables without MadLoop.
- With `density_weights`, the virtual sums over helicities (`mc_hel=0`):
  the projected virtual needs every helicity.
- `separate_flavour_configs` is refused.

**Caches** (each element after the first costs colour contractions):
1. MadLoop: a projected call at the momenta, `MU_R` and `G` of the last
   full evaluation only re-projects its stored colour flows. The reruns
   then also leave MadLoop's initialisation counters alone.
2. Real emission (`SMATRIX_N_W`): the JAMPs of every helicity are kept for
   the same momenta and `G`; the helicities vanishing in the helicity sum
   (`GOODHEL`) are skipped; the helicity index tables are built once per
   FKS configuration.
3. Born: the cached `SBORN` result also records the projector it was
   computed with; the helicity tables (`BORN_DENS_TABLES`, `BORN_DENS_FLIP`)
   are built once.

**Bugs found on the way.**
- **Identical-helicity shortcut of the real emission (`T_IDENT`).** On the
  first point, two helicity configurations with equal |M|^2 (to 1e-12) are
  tied and only one is computed afterwards. In the frame of the density
  legs this is not a symmetry: for `u d~ > w+ g g` in the W rest frame,
  (u-, d~+, W+1, g-, g-) and (u-, d~+, W+1, g+, g+) agree to 1e-15 at the
  first point and differ elsewhere. The sum of the projected reals then
  missed the unprojected one by 1e-5 to 1e-2 per contribution, and the
  central real of a density output was itself slightly biased. The
  shortcut is off in density outputs. The plain output (partonic frame)
  ties nothing on its first point. **Not checked: polarised NLO outputs
  with `me_frame`, which use the same template.**
- **Pole check of the reruns.** BinothLHA's pole check (every point with
  the helicity sum) ran on the projected virtuals. Elements that vanish at
  Born level fail the relative comparison, and `t t~` and `w+ z` stopped
  with "TOO MANY FAILURES". The check is now skipped in the reruns: the
  unprojected virtual is checked at the same point, and each element's
  poles by `check_poles` of the standalone output.

**Runtime check.** `include_density_weights` compares, per contribution,
the sum of the diagonal elements with the central weight (1e-7, or 1e-4
for the virtual) and warns on the first ten failures.

**Gates.**
- Sum of the diagonal elements = central weight, per contribution: all
  tree-level contributions to rounding. The virtual agrees to MadLoop's
  accuracy (1e-8 to 3e-7 on 4 of ~30k points), because its density uses the
  colour flows of the reference evaluation of the stability test. No
  runtime warning in the `w+ j`, `t t~` and `w+ z` runs. In the HwU totals
  it holds to the print precision (<= 2.6e-7).
- The central result does not depend on `density_weights`: identical for
  `t t~` and `w+ z` (fixed numbers of points), and `w+ j` (adaptive) gives
  5.169e4 vs 5.171e4 pb, rounding-level differences amplified by the grid
  adaptation.
- Standalone per-element tests after the template changes, unchanged:
  `w+ j` ME 54/54, MC 60/60, poles 60/60; `w+ z` 162/162, 164/164, 164/164;
  `t t~` 48/48, 51/51, 51/51.

**Cost.** Time with `density_weights` over the same output without it
(virtual helicities summed in both: `NHelForMCoverHels=-1`):

| process | elements | total | Born | counterterms | integrated CT | reals | virtual |
|---|---|---|---|---|---|---|---|
| `w+ j` (no caches) | 9 | 7.7 | 30 | 29 | 45 | 42 | 9.7 |
| `w+ j` | 9 | 2.1 | 14 | 12 | 11 | 4.5 | 1.0 |
| `t t~` | 16 | 1.3 | 22 | 19 | 13 | 3.9 | 1.0 |
| `w+ z` | 81 | 3.8 | 147 | 114 | 102 | 16 | 1.25 |

The default run samples the virtual's helicities: for `w+ j` it is 1.6x
faster than the summed one. What remains grows with the number of elements:
the Born-level pieces (Born, counterterms, integrated counterterms) are
re-traced for each W. If `w+ z` is too slow: density-valued Born-level
pieces, or a user subset of elements (U-D2).

### V2 — fixed-order density analysis (done)

- `HwU.f`: for the weights labelled `dens ...`, per bin the uncertainty of
  w_K and of w_K + w_central, written as `<label> @dy` and `<label> @dyc`
  (iterations combined with the coefficients of the central value; nothing
  changes without density columns).
- `histograms.py`: `is_error_label`; the `@dy`/`@dyc` columns are combined
  in quadrature by every sum (operations, lazy views, rebinning, the
  streaming `--sum`). `--density[=<density_basis.json>]` outputs, per
  histogram, the elements of the normalised density matrix of each bin
  (delta-method uncertainty, with the covariance with the central value
  from `@dyc`), `f_L`, `f_0`, `f_R` for one spin-1 leg, `B1_i`, `B2_i`,
  `C_ij` for two spin-1/2 legs (uncertainty: an upper bound, the
  correlations between elements are not kept), and `<out>_density.json`
  with the matrices and their smallest eigenvalue.
- Output: `SubProcesses/density_basis.json` (legs, PDG codes, frame,
  helicities of each combination, element order) and `density_pdg.inc` per
  P directory; `analysis_HwU_density.f` (total rate, p_T and y of the
  density legs, pair mass, p_T and Delta-phi, leading-jet p_T) is the
  default FO analysis of a density output.

**Gates** (`p p > w+ j [QCD]`, `--density=3`, `me_frame=[3]`, req_acc
0.005; polarised runs `w+{L}`, `w+{0}`, `w+{R}` with `me_frame=3` and the
same analysis):
- Totals: sigma_L + sigma_0 + sigma_R = 5.259e4 +- 142 pb, unpolarised
  5.254e4 +- 210 pb.
- f_L, f_0, f_R of the density run vs sigma_lambda / sum of the polarised
  runs, per bin (bins with a polarised uncertainty below 20%):

| histogram | f_L chi2/n | f_0 chi2/n | f_R chi2/n | bins |
|---|---|---|---|---|
| total rate | 0.44 | 1.03 | 0.87 | 1 |
| p_T(W) | 0.93 | 1.76 | 1.59 | 21 |
| y(W) | 1.01 | 0.75 | 1.04 | 20 |
| p_T(j1) | 1.02 | 1.01 | 1.21 | 24 |

  Total: f_L 0.4463 +- 0.0020 vs 0.4448 +- 0.0013, f_0 0.1082 +- 0.0004
  vs 0.1076 +- 0.0004, f_R 0.4455 +- 0.0020 vs 0.4477 +- 0.0013. The
  p_T(W) pulls scatter in sign with no trend (f_0 rises from 0.05 to 0.4
  in both); the per-bin HwU uncertainties are only estimates. The
  polarised `w+{R}` real ties no helicities on its first point (`T_IDENT`,
  V1), so it is not biased here.
- rho per bin: hermitian and of trace 1 by construction; its smallest
  eigenvalue is positive in every bin of the four histograms (>= 0.03).
- `t t~` (16 elements): the diagonal is CP symmetric, rho(-+) = rho(+-) =
  0.1667 and rho(--) = rho(++) = 0.3333 at NLO (helicities in the t t~
  rest frame).
- Unit tests (`test_histograms.py`): the quadrature sums of the `@dy`
  columns (operations, streaming `--sum`, command line), and `--density`
  on a known rho (elements, delta-method uncertainty, f_L/f_0/f_R, the
  command line and its json) and on the two-fermion triplet state
  (B = 0, C = diag(1, 1, -1)).

### t t~ spin-density matrix in (theta, beta / M_tt), LO and NLO

**Frame: `density_nrk` (run_card, hidden, default False).** The density
matrix in the helicity frame {n,r,k} of the first density leg: the frame of
the matrix elements rotated so that this leg moves along +z with the z axis
in the (y,z) plane, exactly as the reweight module's density mode
(`refchoicep`/`rotationp`, components below 1e-10 of the energy zeroed).
A rotation multiplies the amplitude of each helicity of a density leg by a
phase, so rho_nrk = D rho D^dagger: before each matrix-element call
(`boost_to_frame.f` wrappers), `density_frame_rephase` (density_weights.f)
turns the projector into W' = D^dagger W D, D from the HELAS wave functions
of the density legs at the momentum and at the rotated momentum (a check
stops if they are not proportional). Without it, the coherences that pick
an azimuthal phase average out (rho(++,--) of t t~: 0.002 instead of 0.079).
`density_info.inc` gains the wave-function type of each density leg and the
base projector.

**Setup.** `p p > t t~ [QCD]`, `--density=3,4`, 13 TeV, NNPDF4.0 (334700),
mu_R = mu_F = m_t = 173 GeV fixed, `density_nrk`; analysis
`analysis_HwU_density.f`: theta (top angle to +z in the t t~ rest frame,
reached by a boost from the lab) x beta (10 x 18 cells), and theta x M_tt
(24 mass bins 346-1000 GeV x 20). LO method: `generate p p > t t~`,
madevent, 500k events, `reweight=density` with `helicity_direction [6]`,
`particle_in_density_matrix [6, -6]`, `boost_choice [6, -6]`, same PDF and
scales.

**LO: fixed order vs the reweight module** (159 cells with >= 100 events):
- sigma: 465.6 +- 0.3 pb (FO) and 465.45 +- 0.10 pb (madevent).
- Conventions: the reweight library's matrix (`square_matrix`) is the
  complex conjugate of rho(i,j) = M_i conj(M_j) of the density weights; the
  comparison uses the latter. The top width must be zero in the reweight
  (aMC@NLO sets it to zero): with Gamma_t = 1.49 GeV in the t-channel
  propagator, the real parts of the four single-flip elements (C_kn, B_n:
  zero at tree level) are ~1e-4 instead of 0.
- 15 parameters per cell, each against its own uncertainty: chi2/ndf =
  1140.6/1431 (p ~ 1), pulls of mean 0.01 and width 0.89, none above 3
  sigma.
- With the covariance of the event sample (4 independent directions per
  cell: rho(++) = rho(--), rho(+-) = rho(-+) and the single-flip imaginary
  parts are equal up to sign event by event): chi2/ndf = 311/636.
- Concurrence (lambda1 - lambda2 - lambda3 - lambda4 before max(0,.), with
  the event covariance propagated): chi2/ndf = 130.8/159, p = 0.95.
- Standalone limit tests with `density_nrk`: MC 51/51; ME and poles pass
  for every element, but 15 of them (the real parts of the four single-flip
  elements and Im rho(--,++), zero at Born level in the {n,r,k} frame) only
  with W + 1. The launcher decides which elements vanish from check_fks,
  which does not apply the rotation, so it tested them with W alone (0/0).
  **To do:** make that decision in the rotated frame.

**Fig. 5(a) of arXiv:2510.17730.** Its map is reproduced cell for cell by the
*average of the per-event concurrences* of the reweight events, not by the
concurrence of the bin-averaged rho (which Sec. 2.1 of the paper requires):
M_tt < 350: 0.73 vs 0.51; 350-355: 0.65 vs 0.42; 355-365: 0.55 vs 0.34;
365-380: 0.42 vs 0.22; 400-450: 0.13 vs 0.00 GeV. Near threshold the q q~
events (23% there, C ~ 0: spins along the beam, a separable mixture) dilute
the g g singlet (C = 0.87). Both fixed-order maps are the concurrence of the
bin-averaged rho; a weighted NLO computation has no per-event state.

**NLO** (same setup, 3 x 600k points): sigma = 691.6 +- 0.6 pb (K = 1.49
at mu = m_t). Same map topology; the concurrence of the high-M, theta ->
pi/2 lobe drops by about 0.03. In the narrow mass bins next to threshold
(346-362 GeV) NLO - LO alternates in sign from row to row by up to 0.15:
migration of real emission and counterterms between narrow bins at a
kinematic edge; coarser bins are needed for a statement there.
