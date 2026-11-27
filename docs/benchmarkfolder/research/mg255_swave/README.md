# S-wave onium + jet virtuals against MG5 2.5.5

These scripts compare MadLoop's one-loop virtuals for S-wave quarkonium plus a jet with the virtuals of the MG5 2.5.5 onia outputs (`SubProcesses/P0_*/V0_*`) and, for J/ψ, of the standalone package `Jpsi_double_cs_NEW`. They are all in `/shared/external/madloop_onia/references/NLO_Onia_Nov2026`.

| Reference output | Channels | m_Q | n_lq | Reference finite part |
|---|---|---|---|---|
| `2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan` | gg → η_c g, gu → η_c u, ug → η_c u, uū → η_c g | 1.55 | 3 | analytic, OneLOop QP boxes (`calcoption=3`) |
| `Jpsi_double_cs_NEW` | gg → J/ψ g | 1.55 | 3 | analytic (`calcoption=3`) and grid (`calcoption=1`); the virtual split into transverse and longitudinal J/ψ, summed here |
| `2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan` | gg → η_b g, gu → η_b u, ug → η_b u, uū → η_b g | 4.75 | 4 | analytic (`calcoption=3`) and grid (`calcoption=1`, the default) |
| `2.5.5_bOnia3S11/pp2Yj_Grid_FKS_MultiChan` | gg → Υ g | 4.75 | 4 | analytic (`calcoption=3`) and grid (`calcoption=1`, the default) |

## Conventions

The 2.5.5 code passes the V0 result to MadFKS unchanged (`BinothLHA.f`: no DR-to-CDR conversion, "already in the CDR scheme format" as MadLoop's). Its poles are checked against MadFKS's `getpoles`. MadLoop's conventions therefore apply directly. The settings are:
- **Reference.** `renormalisation_scheme=1`: the heavy quark is decoupled (zero-momentum subtraction) and α_s has n_lq flavours, as in MadLoop.
- **MadLoop.** The heavy quarks the reference decouples are excluded from the loops (`ml_side.CASES`), the exact m → ∞ limit:
  - Charmonium uses `loop_sm_onia-c_mass` with `/ t t~ b b~`, so that only n_lq = 3 light flavours and the charm remain, as in the reference.
  - Bottomonium uses `loop_sm_onia-no_tau_mass` (massless charm, n_lq = 4), with m_b = 4.75 GeV and `/ t t~`.
  - Until 2026-11-27 they were decoupled with masses of 10⁴ GeV instead (`ml_side.CASES_1E4`, still used by `pol_jpsi.py`). Their R2 counterterms carry M² and leave rounding at soft jets (below), up to 2.3e-6 in the stable points of the earlier results.
  - The reduction library is TIRust, with MadLoop's stability checks (`CTModeRun=-1`).
- **Normalisation.** Both results are given in units of Born·α_s/2π. The reference Born is its double pole divided by −Σ C_i (9 for gg → onium g, 17/3 for the quark channels).
- **Scale.** μ_R = 11 GeV, so the log(μ²/m_Q²) terms of the scheme are tested.

## Files

| File | Role |
|---|---|
| `refdrv.F`, `build_ref.sh` | builds a reference V0 with a driver that reads G, μ_R, m_Q, the LDME factor, the scheme, `calcoption` and the points (OneLOop and mpfun90 from `vendor/CutTools`, the module renamed `avh_olo`). For a V0 split into two polarisations (`TWO_POL`) it prints their sum and each; for a standalone package it sets the grid file its `main.f` would |
| `ml_side.py`, `gen.py` | generate and evaluate the MadLoop outputs (`tools/onia_loop_check.py`) |
| `compare.py <case>` | the reference's `PS*.input` points and three random points at each of three energies |
| `summarize.py` | the largest differences per case |
| `data/cmp_<case>.json` | the results per point: MadLoop, and the reference for each `calcoption` (for J/ψ also per polarisation, `T` and `L`, each in units of its own Born) |
| `pol_jpsi.py`, `data/pol_gg_jpsig.json` | g g > Jpsi g per J/ψ polarisation (`ONIA_SELECT_JZ`) against the transverse and longitudinal virtuals of `Jpsi_double_cs_NEW` |
| `data/cmp_gg_jpsig_255grid.json` | the earlier J/ψ comparison with the grid of `2.5.5_Onia/pp2psij_Grid_FKS_MultiChan` (9 random points, 1.3e-4) |

The paths are those of the scratch workspace in which the scripts ran.

## Results

μ_R = 11 GeV. The reference is evaluated directly (`calcoption=3`). Each case uses the reference's `PS*.input` points (for J/ψ those of g g > etac g, same m_c) and three random points at each of three energies. "Stable" means the points MadLoop reports stable (return codes 1xx to 3xx). The columns give the largest relative difference of the finite part and the largest absolute differences of the 1/ε and 1/ε² coefficients, all in units of Born·α_s/2π:

| Case | Stable points (√s, GeV) | Finite | 1/ε | 1/ε² | Points reported unstable |
|---|---|---|---|---|---|
| g g > etac g | 12 (5–203) | 1.2e-10 | 3.1e-11 | 3e-11 | 4 (√s ≤ 3.15) |
| g u > etac u | 13 (3.15–203) | 3.7e-9 | 2.2e-8 | 2e-9 | 3 (√s ≤ 3.12) |
| u g > etac u | 13 (3.15–203) | 2.5e-9 | 2.3e-8 | 7e-9 | 3 (√s ≤ 3.12) |
| u u~ > etac g | 12 (5–203) | 2.5e-11 | 2.1e-9 | 5e-10 | 4 (√s ≤ 3.15) |
| g g > Jpsi g | 12 (5–203) | 2.3e-11 | 1.2e-10 | 6e-12 | 4 (√s ≤ 3.15) |
| g g > etab g | 9 (11–200) | 6.5e-10 | 3.2e-10 | 8e-11 | 8 (√s ≤ 11.2) |
| g u > etab u | 16 (9.506–200) | 1.6e-12 | 2.2e-11 | 6e-12 | 0 |
| u g > etab u | 16 (9.506–200) | 2.6e-11 | 6.4e-10 | 1e-10 | 0 |
| u u~ > etab g | 11 (11–200) | 9.3e-11 | 6.8e-10 | 7e-10 | 5 (√s ≤ 9.55) |
| g g > Upsilon g | 9 (11–200) | 8.2e-10 | 4.2e-9 | 9e-11 | 7 (√s ≤ 11) |

With b, t excluded (2026-11-27, pinned TIRust). With them at 10⁴ GeV the finite parts were within 2.3e-6 (g g > etac g), 1.0e-6, 3.4e-7, 3.3e-7, 1.3e-8, 6.5e-8, 6.5e-7, 3.8e-7, 2.8e-7 and 2.8e-9, in the order of the table (`docs/benchmarkfolder/data/raw/v16all/pinned1e4_cmp_<case>.json`); the polarisation items below keep that setup.

- **Borns.** The Borns agree up to the LDMEs of the model: 0.3867 against 1.16/3 for η_c (ratio 1.000086), 3.093 against 9.28/3 for η_b (ratio 0.999892), and 1.16 for J/ψ and Υ (ratio 1).
- **J/ψ.** `Jpsi_double_cs_NEW` evaluates the virtual directly, and MadLoop agrees with it to 1.3e-8. Its grid differs from that evaluation by up to 6.8e-4, and the earlier comparison with the grid of `2.5.5_Onia` gave 1.3e-4. The other grids (`calcoption=1`) agree with their direct evaluation to 1e-8, except for Υ, where they differ by up to 1.1e-4.
- **J/ψ polarisation.** `Jpsi_double_cs_NEW` also gives the transverse and longitudinal J/ψ in the helicity frame of the partonic centre-of-mass frame. `pol_jpsi.py` evaluates MadLoop with `ONIA_SELECT_JZ` (J_z = ±1 and 0) at the same points (`data/pol_gg_jpsig.json`). The polarised Borns agree to 1e-12.
  - **Pinned TIRust, at the 12 stable points.** T agrees within 2.3e-8 and L within 9.2e-7. L's double-precision value carries MadLoop's rounding of ~1e-7 relative to its suppressed Born.
  - **TIRust 08edee3 in forced quadruple precision, at all 16 points.** T, L and their sum agree within 3e-12, and the 1/ε coefficients within 5e-11.
  - **TIRust 08edee3 in double precision.** Every point is reported stable, but at the soft points the result is wrong: up to 440% for L at √s = 3.1015. The cause is MadLoop's double-precision R2 counterterms of the 10⁴ GeV b and t (next item), not TIRust.
- **Points reported unstable.** All of them have a soft or low-p_T jet: p_T below 0.05 GeV for charmonium, and up to 0.86 GeV in g g > etab/Upsilon g, which are stable from 1.05 GeV. All are in the gg and qq̄ channels; the qg channels are stable down to p_T = 6 MeV.
  - At these points TIRust refuses some integrals in both precisions: the Coulomb triangle, and the Coulomb boxes and pentagons with the jet attached.
  - MadLoop returns code 4x0 with accuracy 1, and the 1/ε coefficient misses −2C_F or −C_F/2 of the Born.
  - MadLoop's own pole prediction (`ONIA_IR_POLES`) agrees with the reference at these points.
- **With TIRust 08edee3** (not pinned) these points are served, and at the softest ones MadLoop's double precision is silently wrong. For example, g g > Upsilon g at √s = 9.51 is 5.6e-3 off, and g g > Jpsi g at 3.1015 is 22% off, with accuracy estimates of 1e-8 to 1e-12. Quadruple precision gives the reference.
  - **Not TIRust.** With every TIRust array replaced by its quadruple-precision value (call dumps and substitution, scratch), the double-precision result is unchanged. The loop part agrees between the two precisions to 4e-8.
  - **The cause.** The R2 counterterms of the gluon self-energy with the b and t quarks (`R2_GGB`, `R2_GGT`) carry M² ≈ 1e8 for the 10⁴ GeV masses that decouple them here. They multiply current contractions that vanish exactly (1e-17 in quadruple precision, ±17 in double precision).
  - **Scaling.** The error scales with M². For g g > Jpsi g at 3.1015 it is 22% with m_b = m_t = 1e4, 4.9e-3 with 1e3, and 1.7e-4 with the physical 4.75 and 173.
  - **Unseen by the stability tests.** MadLoop's stability tests vary only the loop part, not the counterterm amplitudes.
  - **Fixed.** Bound-state outputs now add `epsilon sqrt(sum t^2) / |c0|` to the double-precision accuracy, over the terms t of the counterterm interference, and rescue a point beyond `MLStabThres` in quadruple precision. With 08edee3 and the 1e4 GeV b and t, all 16 J/psi points are right for every polarisation; the four soft points are rescued (code 323). `data/pol_gg_jpsig.json` holds this run under the keys `*_-1_08edee3_ctacc`.
