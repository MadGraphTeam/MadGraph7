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

The unit-power lanes have all-massive heavy-quark propagators that are linearly dependent at the two thresholds: the constituents carry P₁/2 and P₂/2. TIRust's quadruple-precision path refuses them like its double-precision one, so they are not a matter of MadLoop's precision. A refused QP hexagon cost about 75 s in the replay. Those attempts are not where the 72 minutes go, though: on eb363d6, where refusals are fast, the passes take as long (next section).

**g g → χ_c0 χ_c0 on TIRust eb363d6** (bridge v6, local on 196bdcc, not pushed). eb363d6 raises a heavy line of a linearly dependent all-massive set, and reduces a hexagon whose points span three dimensions by its relations. The 97 records of `data/refused_gg_chic0chic0_d5442e6.txt` are all served in DP (`data/qp_replay_gg_chic0_chic0_eb363d6.txt`). In QP, with the binary64 records widened, 63 are served and 34 hexagons are refused ("ill-conditioned Gram matrix"): widened, those inputs are off the thresholds by binary64's rounding, and TIRust serves the exact kinematics in binary128. The replay takes 6 s, so refusals are fast now. e⁺e⁻ → χ_c0 h_c (DP at the four points, QP at 10.58 GeV) is bit-identical to d5442e6, and so are the 16 one-P-wave `pw_check.py` cases against 59aee45.

At √s = 20 GeV, point 0, in DP (`data/qp_gg_chic0_chic0_eb363d6.json`):

| | CTModeRun 1 | CTModeRun 2 | predicted |
|---|---|---|---|
| 1/ε² per Born | −5.99999999997 | −6.0000000001 | −6 |
| 1/ε per Born | −9850 | −510349 | 23.232 |
| return code | 410 | 410 | |

The two finite parts differ by a factor of 13. With the unit-power calls served, MadLoop reaches the raised calls of those loop groups, and TIRust refuses 26 of the 3480 per pass (13 in a normal run, where a refusal ends its group). All 26 are two-doubled at rank 6 with both doubled lines heavy:
- CTModeRun 1: 6 boxes QQQQ (1,1,2,2), 8 pentagons QQgQQ (1,1,1,2,2), 12 hexagons QQgQQg (1,1,1,2,2,1);
- CTModeRun 2: the same on the reversed lines.

TIRust refuses them in both precisions with "lightlike bubble [i, j] with massive lines, both differentiated (two jet variables)". Here [i, j] is the doubled pair: two heavy lines whose offsets differ by a lightlike momentum. The inputs are in `data/refused_gg_chic0chic0_eb363d6.txt` (both passes) and the replay in `data/qp_replay_gg_chic0_chic0_refused_eb363d6.txt`. These are the only refusals left at this point. They do not account for the whole 1/ε: the next section has them served, and 1/ε is still wrong.

In quadruple precision (CTModeRun 4 and 5, the same file), both passes return 420 with 1/ε² = −5.999999999999992. CTModeRun 4 gives 1/ε = −9850.2576 and agrees with CTModeRun 1 to 1e-11 (finite part to 1e-12). CTModeRun 5 gives −510567.4, 4e-4 from CTModeRun 2. Each pass takes 73.5 minutes (4411 and 4419 s), against 72 on 196bdcc and 82 to 88 s in DP. A copy of the output with timers around the two QP TIRust entries splits CTModeRun 4 (4402 s in that run). The first QP TIRust call comes at 4239 s, since MadLoop builds all binary128 loop coefficients before reducing. The QP TIRust calls then take 161 s: 7.7 s for about 1660 unit-power calls and 153 s for about 3340 raised-power calls. So 96% of the pass is MadLoop's binary128 numerator.

**g g → χ_c0 χ_c0 on TIRust 632ed10** (bridge v7 f453cd7a, local on eb363d6, not pushed). 632ed10 evaluates the equal-mass lightlike bubble with both heavy lines as jet variables in closed form. Its acceptance:
- **Replays.** The 52 records of `data/refused_gg_chic0chic0_eb363d6.txt` are all served, in QP (widened) and DP. The 97 d5442e6 records give the same statuses and messages as on eb363d6.
- **Other checks.** e⁺e⁻ → χ_c0 h_c (DP at the four points, QP at 10.58 GeV) and the 16 `pw_check.py` cases are bit-identical to eb363d6.
- **g g.** At 20 GeV, point 0, `fixtures.py` finds no refusal: 3480 raised-power calls per pass, code 110.

The virtual is still wrong (`data/qp_gg_chic0_chic0_632ed10.json`):

| | CTModeRun 1 | CTModeRun 2 | predicted |
|---|---|---|---|
| 1/ε² per Born | −5.99999999997 | −6.0000000001 | −6 |
| 1/ε per Born | +39355 | −517385 | 23.232 |

The 1/ε² is bit-identical to eb363d6; the finite parts differ by a factor of 130. To split the 1/ε by loop size, a patch of the DP wrappers returns zero integrals for loops with a number of lines N outside `ONIA_NRANGE`, and each range is taken against the counterterm-only run (no loops). The table gives 1/ε per Born with TIRust's cache off (`TIRustGlobalCache` 0):

| N | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| CTModeRun 1 | +6.249 | +0.403 | +134.159 | +214.03 | +46218.5 |
| CTModeRun 2 | +6.249 | +0.403 | +134.159 | −63.30 | −517439.5 |

Up to N = 4 the two loop directions agree to 1e-12. With the cache off the ranges add up to the full result. With the cache on (the default), CTModeRun 1 changes to +519.6 at N = 5 and +38757.8 at N = 6, and the ranges no longer add up. CTModeRun 2 does not depend on the cache.

`fixtures.py`, run once with the cache and once without, dumps every raised-power call with its inputs and binary64 array. The dumps (26 MB each) are not kept in the repository. Three problems show in them, all in TIRust:
- **The cache changes results.** 11 calls of CTModeRun 1 get different arrays with the cache on, by 0.22 to 34 relative to their largest entry. All are pentagons and hexagons at rank 5 and 6 with a doubled heavy line.
- **Results depend on earlier calls**, also with `cache_enabled=.false.`. `replay_arrays.f90` replays records through the DP entry and prints their arrays. In `data/pairs_gg_chic0_chic0_632ed10.txt`, record 13 (a QQgQgg hexagon, rank 6, powers (1,1,1,2,1,2)) gives an array 0.77 away from its result alone when record 9 (a ggQgQQ hexagon with the same powers) runs before it. In MadLoop, six of the eight calls sampled there return arrays O(1) away from the same inputs replayed alone.
- **Results depend on the line order.** Records 5 and 7 are the same six lines (shifts, masses and powers) in the order of each pass: a QQQgQg hexagon, rank 5, powers (1,1,1,1,1,2). Replayed alone, they differ by 1.5, with either check-pass flag.

Over the whole point, 97 calls of CTModeRun 1 differ by more than 1e-6 from the CTModeRun 2 call with the same set of lines, and 79 by more than 0.1. All have five or six lines, which matches the split by N. The scalar coefficient (rank 0, all four slots) agrees between the passes and between repeats for all 719 distinct integrals; only the higher tensor coefficients differ. The lists are in `data/state_gg_chic0_chic0_632ed10.json`. Reported to TIRust on 2026-11-24. On 08cca00 all three problems are gone (next paragraph).

**g g → χ_c0 χ_c0 on TIRust 08cca00** (bridge v8 28f89b1f, poles-only patch cd6a21c1, on 632ed10; pushed as hs/pwave-two-dotted in huashengshao/TIRust#58, stacked on #56). 08cca00 fixes two defects of TIRust's direct engine that the dumps above showed:
- **A missing term.** A partial fraction that holds a line raised by another set dropped that line's term at nonzero Taylor order. A hexagon takes the first set it holds, so the result followed the line order.
- **Stale orders.** A lightlike bubble that holds one of several jet variables kept the other variables' orders from the previous call.

At 20 GeV, point 0, in DP (`data/qp_gg_chic0_chic0_08cca00.json`):

| | CTModeRun 1 | CTModeRun 2 | predicted |
|---|---|---|---|
| 1/ε² per Born | −5.99999999997 | −6.0000000001 | −6 |
| 1/ε per Born | 23.23247964008 | 23.23247953233 | 23.23247950875 |
| relative error of 1/ε | 5.7e-9 | 1.0e-9 | |
| finite part per Born·α_s/2π | 259.45055952 | 259.45055414 | |

Both passes return code 110 with no refusal. The finite parts agree to 2.1e-8. By loop size (1/ε per Born against the counterterm-only run, which gives −23.3653283640):

| N | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| CTModeRun 1 | +6.248957622 | +0.402553115 | +134.158987771 | −63.303770804 | −30.908919700 |
| CTModeRun 2 | +6.248957622 | +0.402553115 | +134.158987770 | −63.303770806 | −30.908919805 |

The numbers are the same with TIRust's cache on and off, and the ranges add up to the full result. On 632ed10, CTModeRun 2 already had the right N = 5 sum, and both passes were wrong at N = 6.

The dumps of `fixtures.py` (with and without the cache) give:
- **Cache.** The two dumps are identical for all 6960 raised-power calls.
- **Passes.** Each CTModeRun 1 call was compared with the CTModeRun 2 call with the same set of lines. They agree to at most 1.2e-8, and none is above 1e-6 (97 on 632ed10).
- **Changes from 632ed10.** Against its dump without the cache, 100 arrays changed: 56 in CTModeRun 1 and 44 in CTModeRun 2, by 8.6e-5 to 1.4e4 relative to their largest entry. All of them have five or six lines, with two raised lines or a doubled hexagon. The scalars agree, as before.

`replay_arrays.f90` on `data/pairs_gg_chic0_chic0_632ed10.txt`:
- All 32 records give the same array in sequence as alone.
- Record 13 after record 9 equals record 13 alone.
- The two passes of each pair agree to 3.2e-9 or better (records 5 and 7: 4.2e-14).

The 97 and 52 replays are byte for byte those of 632ed10. e⁺e⁻ → χ_c0 h_c (DP at the four points, QP at 10.58 GeV) is bit-identical to 632ed10, and the 16 `pw_check.py` cases are bit-identical too. The lists are in `data/state_gg_chic0_chic0_08cca00.json`. Reported to TIRust on 2026-11-24.

**g g → χ_c0 χ_c0 on 08cca00: more points, quadruple precision and the stability test.** `qp_check.py` ran at 8, 12, 20, 50 and 200 GeV, with its new option `--thres`, at the points of seed 1. These points have the same angles at every energy: cos θ of a χ_c0 is −0.371, −0.986, 0.420 and −0.604 for points 0 to 3. Point 1 is 9.7° from the beam. The runs:
- CTModeRun 1 and 2 with MLStabThres 1d99;
- CTModeRun −1 with MadLoop's default threshold, 1.0d-3.

The data are in `data/more_gg_chic0_chic0_08cca00.json`.

| √s (GeV), point | Born | 1/ε error, CTModeRun 1 / 2 | finite parts, 1 against 2 | code 1 / 2 / −1 |
|---|---|---|---|---|
| 8, 0 | 1.3e-8 | 5.6e-12 / 1.4e-11 | 1.9e-13 | 110 / 110 / 213 |
| 8, 1 | 1.2e-6 | 4.4e-12 / 1.5e-11 | 2.7e-9 | 110 / 110 / 213 |
| 12, 0 | 3.2e-9 | 1.4e-11 / 1.8e-11 | 2.5e-8 | 110 / 110 / 213 |
| 12, 1 | 1.5e-5 | 5.6e-11 / 6.2e-11 | 1.9e-11 | 110 / 110 / 213 |
| 20, 0 | 2.7e-10 | 5.7e-9 / 1.0e-9 | 2.1e-8 | 110 / 110 / 213 |
| 20, 1 | 8.0e-5 | 0.78 / 0.78 (49 / 50 calls fail) | | 410 / 410 / 420 |
| 20, 2 | 4.2e-10 | 1.1e-9 / 4.2e-11 | 1.1e-8 | 110 / 110 / 213 |
| 20, 3 | 2.4e-9 | 9.7e-10 / 0.29 (0 / 12 calls fail) | 7.8e-4 | 110 / 410 / 423 |
| 50, 0 | 5.2e-12 | 8.4e-6 / 1.4e-5 | 7.3e-6 | 110 / 110 / 213 |
| 50, 1 | 3.3e-5 | 25 / 25 (49 / 50 calls fail) | | 410 / 410 / 420 |
| 200, 0 | 2.0e-14 | 13 / 31 (no call fails) | 0.52 | 110 / 110 / 323 |
| 200, 1 | 9.6e-8 | 0.62 / 0.87 (49 / 50 calls fail) | | 410 / 410 / 420 |

The failing calls were counted with `fixtures.py` at the same points.

**Served points.** Wherever no call fails, the poles are as predicted, except at 200 GeV, point 0 (below):
- 1/ε² = −6 to 1e-10 (8.5e-8 at 50 GeV);
- 1/ε to 6e-11 at 8 and 12 GeV, to 6e-9 at 20 GeV and to 1.4e-5 at 50 GeV.

The stability test calls these points stable (213), with accuracy estimates from 6.5e-11 to 2.2e-5 (the largest at 50 GeV).

**Point 1 from 20 GeV on.** TIRust accepts some raised-power calls and then returns status 1. These are 49 calls in CTModeRun 1 and 50 in CTModeRun 2, all with five or six lines and rank 5 or 6. The counts and the power patterns are the same at 20, 50 and 200 GeV. At 8 and 12 GeV the same angle is served.
- **In MadLoop.** PW_REFUSED reports the point unstable (410). The rescue fails as well, since quadruple precision refuses the same calls (420).
- **Replayed alone.** `replay_calls.f90` replays the 171 distinct inputs in `data/refused_gg_chic0chic0_08cca00.txt` (the CTModeRun 2 records carry the check-pass flag). All 171 fail in both precisions. TIRust's messages:
  - 147: "dotted massless line k in a collinear pair of pentagon […]: its (D + 2)-dimensional term is not reduced";
  - 24: "ill-conditioned Gram matrix for […]": conditioning 0 at 256 bits in 20 cases, and 7e-159 or 1e-34 at 512 bits in 4.

**Point 3 at 20 GeV.** Twelve calls of the reversed pass fail with the first message. These are six-line integrals with line 2 or 3 doubled: rank 5, or rank 6 with line 4 doubled as well. The stability test sends the point to quadruple precision. There, 1/ε is right to 6.7e-15, and the finite part equals CTModeRun 1 to 2.2e-9. The return code is still 423 (unstable). Replayed in quadruple precision with the check-pass flag, the 12 calls fail too, so the reversed QP pass (CTModeRun 5) cannot confirm the result.

**Point 0 at 200 GeV.** No call fails, but double precision is wrong:
- 1/ε is −399.5 and −1002 against 32.97, and 1/ε² is −6.005 and −6.032;
- both passes return 110.

The raised-power tensors of the two passes agree to 5e-8 (none above 1e-6; `tensors.py` pairing as above), as they do at 20 and 50 GeV. So the difference between the passes does not come from those integrals. It comes from the unit-power integrals or from MadLoop's double-precision numerator. At this angle the Born falls from 2.7e-10 at 20 GeV to 5.2e-12 at 50 GeV and 2.0e-14 at 200 GeV, and the 1/ε error grows with it: 6e-9, 1e-5, then of order 1.

The stability test sees the two passes disagree. Its quadruple-precision rescue gives:
- 1/ε to 1.8e-11 and 1/ε² to 2e-12;
- a finite part of 527.029 per Born·α_s/2π;
- code 323.

**Quadruple precision at 20 GeV, point 0.** CTModeRun 4 and 5 agree to all printed digits:
- 1/ε = 23.23247950875515, 1.4e-14 from the prediction, and 1/ε² = −6;
- a finite part of 259.4505590323. This is the CTModeRun 1 value to 1.9e-9 and the CTModeRun 2 value to 2.1e-8.

The passes took 4452 s and 4476 s. The forced rescue (MLStabThres 1d-30) gives the same numbers, with code 323, in 4873 s. A rescue at the other points took 4730 to 4940 s.

**Summary.** At the default threshold, MadLoop's return code flags every wrong result on 08cca00. The points with failing calls are 410 or 420, and the stability test catches the 200 GeV loss and rescues it. What TIRust still lacks, at forward angles from 20 GeV:
- the (D + 2)-dimensional term of a dotted massless line in a collinear pair of a pentagon;
- the degenerate Gram cases.

Reported to TIRust on 2026-11-24, with the 171 inputs.

## g g → ³P₂[8] ³P₁[8] g at 20 GeV (TIRust 4f1b06c + bridge v9)

`g g > Jpsi(1|3P28) Jpsi(1|3P18) g [virt=QCD]`, √s = 20 GeV, seed-1 point, CTModeRun 1. The output has 786 Born amplitudes, 50862 loops and 480 heptagon loop calls, with MAXRANK 7 (base 5 + 2). The point stops after about 80 min with "No dp loop library can deal with this integral": MadLoop's `DETECT_TIRUST` admits N ≤ 6 and rank ≤ 6, so the first heptagon ends the run. This section covers the loops before that stop.

**Capture.** The capture shim (`tools/tirust_capture`, kind 6 for the raised-power entry, `TIRUST_TRACE_FAILED_ONLY=1`) recorded 27868 DP calls before the stop. The bridge refused 1232 of them, all inside its domain: five or six lines, rank 4 or 5. There are 516 distinct inputs, in `data/refused_ggg_3p28_3p18_20gev_4f1b06c.txt.gz` (sha256 775966130e4905da…). The records use the format above, and `zcat … | ./replay` replays them.

| Lines | Rank | Powers | Calls |
|---|---|---|---|
| 5 | 4 | one doubled | 306 |
| 5 | 5 | one doubled | 432 |
| 5 | 5 | two doubled | 14 |
| 6 | 4 | unit | 64 |
| 6 | 5 | one doubled | 416 |

**Replay.** `replay_calls.f90` on 4f1b06c + v9 (`data/replay_ggg_3p28_3p18_20gev_4f1b06c.txt.gz`). DP refuses all 516; QP serves 14 (unit-power hexagons, rank 4) and refuses 502:
- 448: "pentagon Gram matrix for […] decided only by the supplied invariants' mismatch … with the shifts" (512 bits; the mismatch is of order 1e-16, the binary64 rounding of the inputs);
- 36: "dotted massless line k in a collinear pair of pentagon […]: its (D + 2)-dimensional term is not reduced";
- 18 unit-power hexagons, rank 4: "ill-conditioned Gram matrix" (conditioning 0 in 12) or "singular Gram matrix".

So, apart from the heptagons, the point would still be unstable (410) on 4f1b06c + v9. Sent to TIRust on 2026-11-25; the rerun waits for its green light.
