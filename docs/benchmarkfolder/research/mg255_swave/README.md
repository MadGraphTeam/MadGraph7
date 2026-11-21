# S-wave onium + jet virtuals against MG5 2.5.5

These scripts compare MadLoop's one-loop virtuals for S-wave quarkonium plus a jet with the virtuals of the MG5 2.5.5 onia outputs (`SubProcesses/P0_*/V0_*`). Those outputs are in `/shared/external/madloop_onia/references/NLO_Onia_Nov2026`.

| Reference output | Channels | m_Q | n_lq | Reference finite part |
|---|---|---|---|---|
| `2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan` | gg → η_c g, gu → η_c u, ug → η_c u, uū → η_c g | 1.55 | 3 | analytic, OneLOop QP boxes (`calcoption=3`) |
| `2.5.5_Onia/pp2psij_Grid_FKS_MultiChan` | gg → J/ψ g | 1.55 | 3 | grid of virtual/Born |
| `2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan` | gg → η_b g, gu → η_b u, ug → η_b u, uū → η_b g | 4.75 | 4 | analytic (`calcoption=3`) and grid (`calcoption=1`, the default) |
| `2.5.5_bOnia3S11/pp2Yj_Grid_FKS_MultiChan` | gg → Υ g | 4.75 | 4 | analytic (`calcoption=3`) and grid (`calcoption=1`, the default) |

## Conventions

The 2.5.5 code passes the V0 result to MadFKS unchanged (`BinothLHA.f`: no DR-to-CDR conversion, "already in the CDR scheme format" as MadLoop's). Its poles are checked against MadFKS's `getpoles`. MadLoop's conventions therefore apply directly. The settings are:
- **Reference.** `renormalisation_scheme=1`: the heavy quark is decoupled (zero-momentum subtraction) and α_s has n_lq flavours, as in MadLoop.
- **MadLoop.**
  - Charmonium uses `loop_sm_onia-c_mass`, with m_b = m_t = 10⁴ GeV so that only n_lq = 3 light flavours and the charm remain, as in the reference.
  - Bottomonium uses `loop_sm_onia-no_tau_mass` (massless charm, n_lq = 4), with m_b = 4.75 and m_t = 10⁴ GeV.
  - The reduction library is TIRust, with MadLoop's stability checks (`CTModeRun=-1`).
- **Normalisation.** Both results are given in units of Born·α_s/2π. The reference Born is its double pole divided by −Σ C_i (9 for gg → onium g, 17/3 for the quark channels).
- **Scale.** μ_R = 11 GeV, so the log(μ²/m_Q²) terms of the scheme are tested.

## Files

| File | Role |
|---|---|
| `refdrv.F`, `build_ref.sh` | builds a reference V0 with a driver that reads G, μ_R, m_Q, the LDME factor, the scheme, `calcoption` and the points (OneLOop and mpfun90 from `vendor/CutTools`, the module renamed `avh_olo`) |
| `ml_side.py`, `gen.py` | generate and evaluate the MadLoop outputs (`tools/onia_loop_check.py`) |
| `compare.py <case>` | the reference's `PS*.input` points and three random points at each of three energies |
| `summarize.py` | the largest differences per case |
| `data/cmp_<case>.json` | the results per point: MadLoop, and the reference for each `calcoption` |

The paths are those of the scratch workspace in which the scripts ran.

## Results

μ_R = 11 GeV. The reference is evaluated directly (`calcoption=3`), except for J/ψ, which has only its grid. Each case uses the reference's `PS*.input` points and three random points at each of three energies. "Stable" means the points MadLoop reports stable (return codes 1xx to 3xx). The columns give the largest relative difference of the finite part and the largest absolute differences of the 1/ε and 1/ε² coefficients, all in units of Born·α_s/2π:

| Case | Stable points (√s, GeV) | Finite | 1/ε | 1/ε² | Points reported unstable |
|---|---|---|---|---|---|
| g g > etac g | 12 (5–203) | 2.3e-6 | 8.7e-8 | 3e-11 | 4 (√s ≤ 3.15) |
| g u > etac u | 13 (3.15–203) | 1.0e-6 | 1.9e-6 | 2e-9 | 3 (√s ≤ 3.12) |
| u g > etac u | 13 (3.15–203) | 3.4e-7 | 3.5e-7 | 7e-9 | 3 (√s ≤ 3.12) |
| u u~ > etac g | 12 (5–203) | 3.3e-7 | 2.8e-9 | 5e-10 | 4 (√s ≤ 3.15) |
| g g > Jpsi g | 9 (5–40) | 1.3e-4 (reference grid) | 5e-11 | 6e-12 | 0 |
| g g > etab g | 9 (11–200) | 6.5e-8 | 1.1e-8 | 8e-11 | 8 (√s ≤ 11.2) |
| g u > etab u | 16 (9.506–200) | 6.5e-7 | 2.4e-7 | 6e-12 | 0 |
| u g > etab u | 16 (9.506–200) | 3.8e-7 | 3.4e-6 | 1e-10 | 0 |
| u u~ > etab g | 11 (11–200) | 2.8e-7 | 6.8e-10 | 7e-10 | 5 (√s ≤ 9.55) |
| g g > Upsilon g | 9 (11–200) | 2.8e-9 | 4.2e-9 | 9e-11 | 7 (√s ≤ 11) |

- **Borns.** The Borns agree up to the LDMEs of the model: 0.3867 against 1.16/3 for η_c (ratio 1.000086), 3.093 against 9.28/3 for η_b (ratio 0.999892), and 1.16 for J/ψ and Υ (ratio 1).
- **J/ψ.** The reference's finite part is interpolated from a grid of virtual/Born, and its accuracy limits the comparison. For the other cases the reference evaluates analytically. Its grids (`calcoption=1`) agree with that evaluation to 1e-8, except for Υ, where they differ by up to 1.1e-4.
- **Points reported unstable.** All of them have a soft or low-p_T jet: p_T below 0.05 GeV for charmonium, and up to 0.86 GeV in g g > etab/Upsilon g, which are stable from 1.05 GeV. All are in the gg and qq̄ channels; the qg channels are stable down to p_T = 6 MeV.
  - At these points TIRust refuses some integrals in both precisions: the Coulomb triangle, and the Coulomb boxes and pentagons with the jet attached.
  - MadLoop returns code 4x0 with accuracy 1, and the 1/ε coefficient misses −2C_F or −C_F/2 of the Born.
  - MadLoop's own pole prediction (`ONIA_IR_POLES`) agrees with the reference at these points.
