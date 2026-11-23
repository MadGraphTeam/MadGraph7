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

**g g → χ_c0 χ_c0.** On 59aee45, the driver hangs in a unit-power hexagon (N = 6, rank 4), in TIRust's relation search (`DirectExecutor::affine_relation`). It is a TIRust regression, fixed in 8305f8b (v4 patches), which refuses that lane as d17085a did.
