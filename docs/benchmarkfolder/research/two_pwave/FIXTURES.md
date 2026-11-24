# Two P-wave bound states: TIRust fixtures (Phase 1)

One-loop QCD virtuals with two P-wave states, first target e⁺e⁻ → χ_c0 + h_c (`e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD]`, model `loop_sm_onia-c_mass`, m_c = 1.55 GeV). The plan is in `references/PLAN.md` (two-P-wave plan, phases 0–5).

## State (Phase 1)

- The output generates, builds and runs (commit 417f251782).
- The Born of the loop output equals the tree `standalone_fortran` output (5.9411540544807900·10⁻¹⁰ at the test point).
- The virtual is reported unstable (return code 410): TIRust refuses part of the raised-power integrals (below).
- The quadruple-precision path stops with a message (Phase 3).

## Raised-power integrals

The second-order dual term (∂²/∂ε₁∂ε₂) needs these integrals:
- one line doubled (rank r+1, and the rank-r D₁₂ term);
- one line tripled (rank r+2);
- two lines doubled (rank r+2).

`fixtures.py` dumps every such call with the inputs MadLoop passes: powers, masses², invariants, shifts and μ², as hex doubles. It records whether TIRust's capability check accepts the call, its status, and the binary64 array when the call is served. It covers both passes (CTModeRun 1, and 2 with the reversed loop).

```
fixtures.py OUTDIR OUT.json.gz 10.58:2 20:2
```

TIRust 5025e12 + gate + relation tiers, √s = 10.58 and 20 GeV, two points each, both passes: 225 raised-power calls per point and pass, of which 53 are refused.

| Family | Lines, rank | Served | Refused |
|---|---|---|---|
| one line doubled | N = 2–5, rank 2–4 | 1104 | 56 with status 1 (N = 4 rank 3: 16; N = 5 rank 4: 40) |
| one line tripled | N = 2–4, rank 4 | 272 | 0 |
| two lines doubled | N = 3, 4, rank 4 | 0 | 368, by the capability check (`max_dotted_propagators = 1`) |

Every one of the 8 (point, pass) records has the same pattern (46 capability refusals and 7 status refusals each).

## Files (sent to TIRust on 2026-11-23)

- `data/fixtures_ee_chic0_hc.json.gz` (1.9 MB): e⁺e⁻ → χ_c0 h_c, every raised-power call, as above.
- `data/fixtures_gg_chic0_chic0_refused.json.gz` (23 kB): `g g > chic0(1|3P01) chic0(1|3P01) [virt=QCD]` at √s = 20 GeV, one point, both passes. Only the refused calls are kept, with `served_counts` per (n, rank, powers) for the others. That is 277 distinct inputs, from `fixtures.py OUTDIR OUT.json.gz 20:1` on TIRust d17085a with the bridge v2 (its double-precision raised powers are those of 5025e12).

| g g → χ_c0 χ_c0, per pass | Served | Refused |
|---|---|---|
| one line doubled | 1608 | 32 with status 1 |
| one line tripled | 519 | 1 with status 1 |
| two lines doubled (N = 3–5, rank 3–5) | 0 | 634, by the capability check |

## Acceptance of TIRust 59aee45 (two dotted propagators)

TIRust 59aee45 (branch hs/pwave-two-dotted, on d17085a) with the bridge patch v3 (`madloop-tirust-gate-relation-tiers-low-words-extended-inputs-v3.patch`, c44e3351). Its poles-only patch is regenerated for the new jets. The scratch build is `pwave_fd/build.sh` with `REV=59aee45`, the vendor part of v3 as `BRIDGE_PATCH` and `FOLLOWUP=none`. `acceptance.py OUTDIR OUT.json E:N ...` runs both passes and the stability test (CTModeRun 1, 2 and −1, `MLStabThres` 1d99, since two-P-wave outputs have no quadruple precision yet). It prints the computed poles against `ONIA_IR_POLES`.

**A MadLoop bug first.** The first run gave a 1/ε coefficient per Born of −1.46 at 10.58 GeV, against the predicted 0.71. It also depended on the order of the two states (`e+ e- > hc chic0` gave +1.96), although the UV part of the poles was exact (counterterms + UV poles of the loops = −4 C_F, the IR part of Z₂). The finite-difference check of `pwave_fd` for two states (`fd2_patch.py`, `pw2_fd_driver.f`) located the problem in the boxes. The polynomial routines were written for rank R+1, while `LOOPMAXCOEFS` was sized for R+2. `NCOEF_R(RANK+2)` read past the array, and every rank-3 loop group dropped its tripled and two-doubled terms. The exporter now uses R + npwave for both. That adds 51 rank-5 raised calls per point and pass, 276 in all.

**e⁺e⁻ → χ_c0 h_c, after the fix** (7, 10.58, 20 and 30 GeV, two points each, seed 1):

| | Result |
|---|---|
| refused raised calls | 0 of 276 per point and pass, return code 110 in each pass, 213 in the stability test |
| fixtures (10.58 and 20 GeV, matched by inputs) | all 1376 calls served before bit-identical; all 424 refused before now served; 408 new rank-5 calls (N = 3, 4, 5) served |
| 1/ε against the prediction | 9e-13 to 4e-10 in the stability test, 1.1e-9 at worst in one pass (30 GeV) |
| 1/ε² | 0, as predicted |
| the other order of the states | the same poles, and finite parts to 1e-9 |
| MadLoop's accuracy estimate | ≤ 4e-8 |

The 1/ε coefficient per Born is 0.72155466287 (7 GeV), 0.71167194524 (10.58), 0.68419551460 (20) and 0.67519653333 (30). The finite part per Born·α_s/2π, at the driver's default scale (`onia_loop_check.py`, μ = −1), is 21.960261 (7 GeV), 38.381345 (10.58), 52.97172 (20) and 66.40398 (30), the same at both points to 1e-8.

**Other checks with this build:**
- e⁺e⁻ → J/ψ χ_c0 and e⁺e⁻ → h_c η_c (one P wave and one S wave) give 1/ε = 0 to 1e-10, as predicted. These processes are IR finite.
- The 16 P-wave checks of `pwave_fd/pw_check.py` are bit-identical to 5025e12 with the gate.
- The 80-bit switch at the 24 points of `pwave_stability` (u ū and g g → χ_c0 g, all four modes) is bit-identical to d17085a with the bridge v2.
- One-P-wave outputs are file-identical before and after the rank fix.

**g g → χ_c0 χ_c0.** On 59aee45, the driver hangs in a unit-power hexagon (N = 6, rank 4), in TIRust's relation search (`DirectExecutor::affine_relation`). It is a TIRust regression, fixed in 8305f8b, which refuses that lane as d17085a did.

The run on d5442e6 (8305f8b plus a test-only commit) with the bridge v4 (845c66de) is built like v3. The rank fix brings g g's two-state families to rank 6 (R = 4), so it has lanes that no earlier fixture had. At √s = 20 GeV, fixture point 0, per pass:

| Calls | Served | Refused (status 1) |
|---|---|---|
| raised powers | 2900 (two doubled up to rank 6: 74, tripled at rank 6: 42) | 10 two doubled at rank 6: boxes (1,1,2,2) gggg, 2; pentagons (1,1,1,2,2) ggQgg, 8 |
| unit powers | 1671 | 87 at rank 4: boxes QQQQ, 3; pentagons QQgQQ, 4; hexagons, 80 |

The unit-power refusals were then taken to be lanes that TIRust serves in quadruple precision, but the replay in the next section shows that it refuses them in both precisions. The point is reported unstable (410, 413 in the stability test). The 1/ε² is −2.44 per Born, where −2 C_A = −6 is predicted, so the poles cannot be tested until every group is served. The 97 refused inputs are in `data/refused_gg_chic0chic0_d5442e6.txt` (sha256 221cabcc…), sent to TIRust with the results and with the request of 2026-11-24 to serve the 87 unit-power calls.

## Quadruple precision (Phase 3a)

Two-P-wave outputs now have a quadruple-precision path:
- **Dual numbers.** `aloha/template_files/mp_dual2_variables.f` has both directions and the mixed derivative, and is written as the output's `mp_dual_variables.f`. One-P-wave outputs keep the one-direction module, and with it their extended-precision code.
- **Routines.** The `MP_` copies of the two-state routines of `onia_loop.f` (weights, counterterm passes, the four-component coefficient construction, the raised-power families).
- **Templates.** The extra arrays in the quadruple-precision call templates.

`qp_check.py` runs CTModeRun 1, 4 and 5 with MLStabThres 1d99, and −1 with 1d-30, which sends every point to the rescue. Results are in `data/qp_ee_chic0_hc_d5442e6.json`.

**e⁺e⁻ → χ_c0 h_c** (TIRust d5442e6 with the bridge v4, seed 1):

| √s (GeV), point | QP 1/ε against the prediction | DP 1/ε against the prediction | finite part, QP against DP | rescue |
|---|---|---|---|---|
| 7, 0 | 1.1e-15 | 3.2e-12 | 2.4e-12 | 323 |
| 10.58, 0 | 1.1e-14 | 2.8e-12 | 3.4e-13 | 323 |
| 10.58, 1 | 1.7e-14 | 3.0e-12 | 1.1e-9 | 323 |
| 30, 0 | 1.2e-14 | 1.7e-10 | 1.9e-11 | 323 |

- **Loop directions.** CTModeRun 4 and 5 (the loop reversed) agree exactly in double precision, so MadLoop's estimate is 0.
- **Finite part.** In QP both 10.58 GeV points give 38.381344559631 per Born·α_s/2π, so the 1.1e-9 at point 1 is the double-precision error.
- **Cost.** A QP point takes about 180 s, against 3 s in DP. Stack samples (gdb, 30 samples) put 27 in the binary128 coefficient construction (`MP_UPDATE_WL_*`, `__multf3`, `__addtf3`) and 1 to 2 in TIRust.
- **One-P-wave outputs.** They are unchanged: `g g > chic0`, `g g > chic0 g` (with its extended-precision code) and `e+ e- > Jpsi chic0`, generated from HEAD and from the change, have identical sources, apart from the declaration order of ALOHA routines, which varies from run to run.

**g g → χ_c0 χ_c0 in quadruple precision.** TIRust 196bdcc with the bridge v5 (c8cb1ea6) serves the 10 raised rank-6 calls of the d5442e6 run. At √s = 20 GeV, point 0:
- **Double precision.** Both passes still return 410. The 1/ε² is −1.6424 per Born, where −2 C_A = −6 is predicted.
- **Quadruple precision.** Both passes (CTModeRun 4 and 5) return 420, with the same 1/ε² to 12 digits. Each takes 72 minutes.

The same loop groups are missing in both precisions. `replay_calls.f90` replays the 97 calls refused in DP through both entries of 196bdcc (`data/qp_replay_gg_chic0_chic0_196bdcc.txt`):

| Calls | QP and DP | TIRust's message (QP, 256 and 512 bits) |
|---|---|---|
| 10 raised, rank 6, two doubled (boxes, pentagons) | served | |
| 3 unit-power boxes QQQQ, rank 4 | refused | singular Gram matrix; linearly dependent propagators without a massless member |
| 4 unit-power pentagons QQgQQ, rank 4 | refused | the same |
| 80 unit-power hexagons, rank 4 | refused | ill-conditioned or singular Gram matrix, or six-point coordinate system (conditioning 0) |

The unit-power lanes have all-massive heavy-quark propagators that are linearly dependent at the two thresholds: the constituents carry P₁/2 and P₂/2. TIRust's quadruple-precision path refuses them like its double-precision one, so they are not a matter of MadLoop's precision. A refused QP hexagon costs about 75 s, so most of the 72 minutes goes to those attempts.
