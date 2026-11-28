# P-wave stability at high energy: u ū → χ_c0 g and g g → χ_c0 g

## Question

Why do so many P-wave points lose accuracy in double precision at √s = 1 TeV (setup A of the stability profiles)? 75% of the u ū → χ_c0 g points and 40% of the g g → χ_c0 g points have a MadLoop estimate above 10⁻³. And how can this be improved?

## Setup

- Points: those of setup A of `research/stability/stability.py`: the same generator and seed, the first 24 points. Setup B (20 GeV) is used for comparison.
- TIRust revisions: u ū → χ_c0 g with TIRust 48958d1, g g → χ_c0 g with 370b632. Both are scratch builds made with `research/pwave_fd/build.sh`.
- Reference: forced quadruple precision (`CTModeRun 4`). The same DP momenta are used throughout.

## Tools

`instrument.py` patches the `TIRUST_interface.f` of an output. Every TIRust call is numbered in the order of the run, and environment variables control the following:

| Variable | Effect |
|---|---|
| `ONIA_DBG_DUMP` | Dump the returned arrays. |
| `ONIA_DBG_SUBST` | Replace chosen DP arrays by given values, here the QP arrays rounded to double. |
| `ONIA_DBG_INP` | Dump the inputs of the calls with powers. |
| `ONIA_DBG_NOSHIFT` | Switch off TIRust's invariant shift in the check pass. |
| `ONIA_DBG_MIXED=N` | Evaluate DP calls with powers and at least N lines through TIRust's QP path. |

The scripts:

| Script | What it does |
|---|---|
| `estim.py` | Per point: the DP primary pass (`CTModeRun 1`), the check pass (`CTModeRun 2`), the primary pass at the rotated momenta, and QP. |
| `run.py` | DP and QP with call dumps, then DP with QP arrays substituted: all calls, plain only, powers only, or powers with five lines only (`IDX=` selects points). |
| `percall.py` | Per-call DP vs QP deviation. |
| `onebyone.py` | One powers call substituted at a time. |
| `noshift.py` | The check pass without the invariant shift. |
| `rev_subst.py` | The check pass with the pentagons taken from the QP check pass (`CTModeRun 5`). |
| `mixed.py` | The `ONIA_DBG_MIXED` prototype. |
| `qptime.py` | Call timing from the dump timestamps. |
| `cost.py` | Two TIRust builds on the same points, alternating batches: time per point and QP rescues. |
| `tail.py` | The points of a stability profile above a given estimate, in DP with several builds and in QP. |
| `hybrid.py`, `hybrid_run.py` | QP MadLoop with TIRust's DP integrals (`ONIA_DBG_DPTIR`). |
| `ep_retarget.py` | Phase 1 prototype of the 80-bit mode: retargets the QP path of an output copy to kind 10, with TIRust's DP entry. |
| `lowwords.py`, `lowwords_run.py`, `lowwords_cost.py` | The low-word tests: TIRust's raised-power `RAW + RAW_LO` summed in extended precision and the extended-input entry (`--ext`), in QP MadLoop (hybrid) and in the 80-bit copy; the route split; the paired cost of the 80-bit modes. |
| `qsubst.py`, `qsubst_run.py` | DP MadLoop with the doubled-line integrals at full QP precision, contracted in QP (`ONIA_DBG_QDUMP`, `ONIA_DBG_QSUBST`, `ONIA_DBG_QMODE`). |

The results are in `data/`.

## Findings

1. **True errors at 1 TeV** (24 points; `data/estim_*_1000.json`):

   | Process | Points above 10⁻³ | Largest error | Median error |
   |---|---|---|---|
   | u ū → χ_c0 g | 4/24 | 3.4·10⁻² | 1.4·10⁻⁴ |
   | g g → χ_c0 g | 1/24 | 3.9·10⁻² | 1.3·10⁻⁵ |
   | e⁺e⁻ → cc̄[³P₀⁽⁸⁾] g | 0/24 | 7.7·10⁻⁸ | 1.4·10⁻⁸ |

   At 20 GeV the largest u ū error is 7.6·10⁻⁸.

   The colour octet (TIRust 48958d1, `data/estim_ee_3p08g_1000.json`) has no pentagons, only boxes. MadLoop's estimate for it is about 10⁻⁶ (median 1.3·10⁻⁶ over the 10⁴ points of the stability profile). That estimate comes from the reversed-loop pass, which is off by 1.5·10⁻⁷ to 2.5·10⁻⁶. The primary pass is 30 to 100 times more accurate.

2. **The cause: TIRust's DP evaluation of the doubled-propagator pentagons.**
   - Replacing the arrays of the calls with powers and five lines by their QP values brings the error down:
     - u ū, 7 points: from 5.5·10⁻⁴…3.4·10⁻² to 6·10⁻⁷…2.4·10⁻⁵ (five points with only the five-line calls replaced, two with every call with powers);
     - g g, 3 points: from 1.2·10⁻⁴…3.9·10⁻² to 6·10⁻⁷…3.4·10⁻⁶.
   - These are 10 calls per point in u ū and 40 in g g. Replacing the plain integrals changes nothing (`data/subst_*.json`).
   - At u ū point 0, two calls carry 98% of the error, about half each (`data/onebyone_uu_chic0g_1000_p0.json`). Both are pentagons of rank 4 with:
     - line powers (1,1,1,1,2) and masses² (0,0,0,0,m_c²);
     - the doubled line being the heavy quark between the two constituents.
   - The two constituent legs are both −P/2, so the Gram determinant vanishes. The pentagon reduces through the relation D₄ − 2D₅ + D₁ = 4m_c², while the invariants are of order s = 10⁶ GeV².
   - Their DP arrays deviate from QP by only 5·10⁻¹² of their largest entry (`percall.py`). The P-wave sum amplifies this by about 10⁸.
   - Even with exact pentagons, rounding the arrays to double leaves about 10⁻⁶. That is the floor of the DP contraction at 1 TeV.

3. **MadLoop's estimate is pessimistic, for the same reason.**
   - The reversed-loop pass evaluates the same pentagons much less accurately than the primary pass. For u ū, the reversed pass is off by up to 2.3, against 3.4·10⁻² for the primary. With TIRust's invariant shift switched off it is still off by up to 0.31 (`data/ct2_noshift_*.json`).
   - With QP pentagons the reversed pass is accurate too: 5·10⁻⁷…1.5·10⁻⁶ at the four worst points (`rev_subst.py`).
   - MadLoop's accuracy sums the spreads of the finite part and both poles and divides by the mean of the three, roughly three times the spread of the finite part.
   - Result: the campaign flags 75% of the u ū and 40% of the g g points (7/24 of this g g sample), against 4/24 and 1/24 of the samples truly above 10⁻³.
   - The counterterm rounding estimate is negligible here (about 7·10⁻¹⁵).

4. **A MadLoop-side prototype, and why it is not enough** (`mixed.py`, `data/mixed_*.txt`). The prototype sends the calls with powers and at least five lines through TIRust's QP path, from the DP inputs.
   - u ū at 1 TeV: the largest error is 5·10⁻⁵ in both passes. The estimate is ≤ 2·10⁻⁴ and never below the true error, and no point is above 10⁻³. The time is 0.52 s per point with the stability test. Today it is 0.13 s, plus a 4.2 s QP rescue for most points.
   - u ū at 20 GeV: the largest error is 3.6·10⁻¹¹, at 0.51 s per point, four times today's cost for points that are already fine.
   - It does not generalise:
     - Extended to four lines, every u ū point is refused (`data/mixed4_*`).
     - For g g → χ_c0 g, refused lanes leave the result off by up to 5.5·10⁻² (`data/mixed_gg_*`).
   - TIRust's QP path cannot recognise threshold relations that hold only to binary64 rounding in the inputs, for example Coulomb relations in the g g pentagons. MadLoop then reports those points unstable (estimate 1), so the result is safe but not better.

## Remedy

The robust fix is in TIRust's binary64 entry point. For these lanes it would:
- detect the relations at the precision of the binary64 inputs, as today;
- evaluate in double-double or binary128 arithmetic, like the binary64 gate does for ill-conditioned lanes.

The lanes are the pentagons with zero Gram determinant and a doubled line, at large s/m². The substitution tests give the expected effect. The DP errors would drop to about 10⁻⁶ in both passes, so the estimate would become honest and no point at 1 TeV would need the rescue. Only the doubled pentagons would cost extra time.

## Candidate fix: TIRust 14f2696

The TIRust session's answer to the reproducer below. The runs in this section use 14f2696, with `oneloop` 07fc754. 8b436f9 and 5025e12, in PR #56, give the same results (see "On 8b436f9 and 5025e12 (PR #56)" below):
- **TIRust 14f2696**, one commit on 370b632. It adds an embedder option, `relation_tiers`. Binary64 lanes that the first pass serves through an affine relation (a constant identity or a Coulomb partial fraction) are evaluated again from the double-double tier up. The inputs are still declared binary64, so the same relations are found.
- **The bridge patch** `madloop-tirust-relation-tiers.patch`, one line: `relation_tiers: key.2 != 0`. Only calls with raised powers, i.e. the P waves, use the option.

Build: `research/pwave_fd/build.sh` with `REV=14f2696`, `FOLLOWUP=0001-followup-pwave-collinear.patch` and `BRIDGE_PATCH=madloop-tirust-relation-tiers.patch`. The diagnosis below uses the same 24 points of setup A and the same QP reference, which is bit-identical between the builds.

| | 48958d1 / 370b632 | 14f2696 |
|---|---|---|
| u ū → χ_c0 g, 1 TeV: largest error, primary / reversed pass | 3.4·10⁻² / 2.3 | 2.7·10⁻⁵ / 4.3·10⁻⁵ |
| u ū → χ_c0 g, 1 TeV: median error, primary pass | 1.4·10⁻⁴ | 3.7·10⁻⁶ |
| u ū → χ_c0 g, 20 GeV: largest error, primary / reversed pass | 7.6·10⁻⁸ / 4.2·10⁻⁸ | 3.3·10⁻¹¹ / 1.2·10⁻¹¹ |
| g g → χ_c0 g, 1 TeV: largest error, primary / reversed pass | 3.9·10⁻² / 9.3·10⁻³ | 4.2·10⁻⁶ / 5.2·10⁻⁶ |
| MadLoop's estimate ÷ true error, 1 TeV (u ū; g g) | 2.5 to 1800; 1.5 to 350 | 1.8 to 240; 2.5 to 520 |

(`data/estim_*_14f2696.json`.) The errors left are those the QP-substituted pentagons gave in finding 2.

Cost (`cost.py`, alternating batches of the two builds on the same points; `data/cost_*_14f2696.json`), mean time per point:

| Process | Setting | Before | 14f2696 |
|---|---|---|---|
| u ū → χ_c0 g | production, 1 TeV (rescued 29/40 → 0/40) | 3.5 s | 0.10 s |
| u ū → χ_c0 g | production, 20 GeV | 70 ms | 107 ms |
| u ū → χ_c0 g | profile settings, 1 TeV / 20 GeV | 162 / 156 ms | 207 / 211 ms |
| g g → χ_c0 g | production, 1 TeV (rescued 6/16 → 0/16) | 18 s | 1.2 s |
| g g → χ_c0 g | production, 20 GeV | 0.92 s | 1.08 s |
| e⁺e⁻ → cc̄[³P₀⁽⁸⁾] g | production, 1 TeV / 20 GeV | 25 / 25 ms | 31 / 30 ms |

"Production" means MadLoop's defaults (`MLStabThres` 10⁻³, no rotations, QP rescue). "Profile settings" means those of the stability profiles. The stability profiles of every P wave with this candidate are in `data/profiles_14f2696/`.

The P-wave checks of `research/pwave_fd` (`pw_check.py`, all 16 cases, the same points) give the same results with this candidate (`data/pw_check_14f2696.json`). The finite parts agree within 10⁻¹⁰ and the poles match the prediction to 5·10⁻¹¹, every point stable.

### On 8b436f9 and 5025e12 (PR #56)

The relation tiers are now in PR #56. 8b436f9, its head at the time of these runs, is 14f2696 plus merges of TIRust's main, and the merges re-pin `oneloop` from 07fc754 to cdbc912 (a small-momentum C0 series).
- **Build.** The bridge's pole kernels (`vendor/tirust/pole-kernels`) still pin `oneloop` 07fc754. With cdbc912 in TIRust, two versions of the crate meet, and `iregi-rust` does not compile. `build.sh` therefore takes `ONELOOP_REV=cdbc912ab7b8327a23fe5fc08ddb0c58b9d55a21`, which moves the pole kernels of the scratch copy to the same revision; `vendor/tirust` itself is unchanged.
- **Same results.** The 24-point diagnosis (`data/estim_*_8b436f9.json`: u ū at 1 TeV and 20 GeV, g g at 1 TeV, all three passes and QP) is bit-identical to 14f2696. 22 of the 24 stability profiles have bit-identical estimates. In the two g g profiles at 20 GeV, the new `oneloop` changes estimates of about 10⁻¹² by factors of 0.6 to 1.6, with the same maximum and the same points above 10⁻⁹. The P-wave checks (`data/pw_check_8b436f9.json`) are bit-identical in 15 of the 16 cases, and g g → χ_c0 g differs by 6·10⁻¹⁶.
- **Same cost.** In production settings (`data/cost_8b436f9_vs_14f2696_*.json`): u ū → χ_c0 g takes 96–97 ms per point with either build, at both energies, and g g → χ_c0 g at 20 GeV 1.06 s against 1.10 s.

The head of PR #56 is now 5025e12, one commit on 8b436f9. In it, a relation-tier lane that every precision tier refuses comes back unstable, where 8b436f9 kept its binary64 value. It is built the same way. Against 8b436f9, every result is bit-identical: all 24 stability profiles (estimates and return codes), the 24-point diagnosis (u ū at 1 TeV and 20 GeV, g g at 1 TeV) and the 16 P-wave checks (`data/pw_check_5025e12.json`). So no point of these samples meets a lane that every tier refuses.

These gate-less 5025e12 profiles are kept in `data/profiles_5025e12_nogate/`, and the 14f2696 ones in `data/profiles_14f2696/`. The report's dotted curves are 5025e12 with the gate as well (next section).

## S waves on 5025e12

The S-wave profiles were also run on 5025e12, with the same build. The relation-tier patch acts on raised powers only, so for the S waves this is the plain 5025e12 bridge, with TIRust's default binary64 gate of 10⁻¹².
- 5025e12 serves every point the pinned TIRust refuses, so no S-wave point is above 10⁻³, and every point is stable.
- The J/ψ and Υ processes keep a tail above 10⁻⁶: 1.8% and 1.7% of the points at 1 TeV, the largest at 10⁻⁴. 08edee3 with the bridge gate patch 0006 (gate 10⁻¹⁰) has no point there.
- That tail is a true error. `tail.py` took the first 16 points of the 5025e12 profile of g g → J/ψ g at 1 TeV with an estimate of at least 10⁻⁶ (`data/tail_jpsig_A_5025e12.json`). Against QP:
  - 5025e12 is off by 1.2·10⁻⁸ to 3.2·10⁻⁶, and MadLoop's estimate is at least twice that;
  - the pinned build is off by the same at 10 of them, by 1.1·10⁻⁵ to 1.7·10⁻⁵ at five, and by 0.32 at one where it refuses an integral (that point is then reported unstable);
  - 08edee3 with the gate patch is off by at most 2.3·10⁻¹⁰.
- So it is the gate, not the TIRust revision, that removes the tail. Patch 0006 does not apply on top of the relation-tier patch, because both change the same `EmbedderOptions` block.
- The pin 6ba4b53 (since 2026-11-27, MadGraph's bridge without the gate) keeps the tail: 1.90% (J/ψ) and 1.68% (Υ) of the points at 1 TeV are at or above 10⁻⁶ (`data/raw/stability/*6ba_*.json` of the report). `tail.py` on the first 16 tail points of its own J/ψ profile (`data/tail_jpsig_A_6ba4b53.json`): off by 1.2·10⁻⁸ to 3.2·10⁻⁶ against QP, MadLoop's estimate at least twice that. At the 15 of them shared with the 5025e12 check, its DP values equal those of 5025e12 to 1.2·10⁻¹⁴, and the QP values of both agree to 6·10⁻¹⁵.

## Everything on 5025e12 with the gate and the relation tiers

The TIRust session merged the two bridge patches into `madloop-tirust-gate-and-relation-tiers.patch` (in `/shared/external/iregi/pwave-patches/`, notes in GATE-AND-RELATION-TIERS.md; the gate is MadGraph7#528). It applies to onia_loop's `vendor/tirust`. Build: `build.sh` with `REV=5025e12`, `BRIDGE_PATCH=madloop-tirust-gate-and-relation-tiers.patch` and `ONELOOP_REV=cdbc912…`. All 36 profiles were run again on it; they are the report's dotted curves (`data/raw/stability/*56_*.json`).
- **S waves.** No point is above 10⁻⁶ in any of the 12 profiles, and every point is stable. The J/ψ and Υ tail is gone, with the largest estimate 8·10⁻⁷. At the 16 tail points above (`data/tail_jpsig_A_5025e12_gate.json`), this build is off by 2·10⁻¹² to 2.3·10⁻¹⁰ against QP, exactly as 08edee3 with the gate. The 08edee3 profiles are therefore kept only in `data/profiles_08edee3_gate/`.
- **P waves.** The octet profiles are bit-identical to those without the gate. The singlets are the same up to rounding: the largest g g estimate at 20 GeV drops from 9·10⁻⁷ to 3·10⁻⁸. The 24-point diagnosis gives the same errors (`data/estim_*_5025e12_gate.json`: u ū largest 2.7·10⁻⁵, g g 4.2·10⁻⁶), and the 16 P-wave checks are bit-identical (`data/pw_check_5025e12_gate.json`).
- **Cost of the gate.** In production settings, alternating gate and no-gate builds on the same points (`data/cost_gate_vs_nogate_*.json`): g g → J/ψ g at 1 TeV takes 88 vs 89 ms per point, and u ū → χ_c0 g 93.5 vs 93.3 ms (1 TeV) and 92.6 vs 93.8 ms (20 GeV). None of this is measurable here.

## What is left at 1 TeV: the double-precision floor

With 5025e12 and the gate, the singlets at 1 TeV still have estimates of about 10⁻⁵ (u ū → χ_c0 g) and 10⁻⁶ (g g → χ_c0 g); the true errors at the 24 points have a median of 3.7·10⁻⁶ and 1.3·10⁻⁶. Two complementary tests on the 5025e12 + gate outputs locate them:

| u ū → χ_c0 g, 24 points, 1 TeV, against QP | largest | median |
|---|---|---|
| DP, as in production | 2.7·10⁻⁵ | 3.7·10⁻⁶ |
| QP MadLoop, DP integrals with raised powers (`hybrid.py`, 1) | 1.9·10⁻⁵ | 4.6·10⁻⁶ |
| QP MadLoop, all integrals DP (`hybrid.py`, 2) | 1.3·10⁻⁵ | 3.8·10⁻⁶ |
| DP MadLoop, doubled-line integrals in QP, contracted in QP (`qsubst.py`, 1) | 3.3·10⁻⁵ | 4.0·10⁻⁶ |
| the same with their numerator formed in QP (`qsubst.py`, 2) | 1.3·10⁻⁵ | 2.3·10⁻⁶ |
| control: the QP arrays rounded to DP, QP sum (`qsubst.py`, 3) | 3.3·10⁻⁵ | 3.9·10⁻⁶ |

(`data/hybrid_*.json`, `data/qsubst_uu_chic0g_1000.json`.) For g g → χ_c0 g, the hybrid test gives 1.3·10⁻⁶ largest and 4.9·10⁻⁷ median, against 4.2·10⁻⁶ and 1.3·10⁻⁶ in production. The `qsubst.py` test is not valid for g g: even its control is off by order 1, so the DP and QP call sequences do not line up there.

What this shows:
- Rounding the integrals to double alone leaves the production error: MadLoop in quadruple precision does not help while the integrals are DP.
- Rounding MadLoop's loop coefficients to double alone leaves it too: exact doubled-line integrals and an exact contraction do not help while the coefficients are DP.
- Both enter the doubled-line terms, and their rounding (10⁻¹⁶) is amplified by about 10¹⁰ in the P-wave amplitude at √s/M ≈ 300. That is the double-precision floor.

Lowering it needs both the integrals and the loop coefficients of the P-wave loops beyond double precision:
- a double-double output from TIRust, which already computes these lanes in double-double;
- the coefficient construction of these loops (MadLoop's polynomial updaters, its largest cost) in double-double or quadruple precision.

The first would be cheap. The second costs a multiple of the double-precision time of these processes, so no further improvement is possible without slowing them down. The quadruple-precision rescue reaches the same end, at about 40 times the time for the points it rescues. At the default threshold `MLStabThres` = 10⁻³ no point needs it, and MadLoop's estimate is at least 1.8 times the true error.

## An 80-bit extended-precision mode: Phase 1 prototype

The user asked for an opt-in switch, off by default, that runs the P-wave loop coefficients beyond double precision. They chose x87 80-bit extended, `REAL(10)` with a 64-bit mantissa, over double-double (plan: generated like the QP path, kind `SELECTED_REAL_KIND(18)`, with a QP fallback on ARM). Before any generator work, `ep_retarget.py` measures it on copies of the 5025e12 + gate outputs:
- It retargets the whole QP path (the phase-space improvement, Born, loop coefficients, contractions and projections; Source/DHELAS, Source/MODEL, SubProcesses) to kind 10.
- It replaces TIRust's binary128 calls by its DP entry.
- CTModeRun 4 of the copy is then "MadLoop in 80-bit, TIRust in DP".

| 24 points, 1 TeV | 80-bit, DP integrals (hi only) | DP production | time per point: 80-bit / DP / QP |
|---|---|---|---|
| u ū → χ_c0 g | largest 1.3·10⁻⁵, median 3.8·10⁻⁶ | 2.7·10⁻⁵, 3.7·10⁻⁶ | 0.51 / 0.071 / 4.24 s |
| g g → χ_c0 g | 1.3·10⁻⁶, 4.9·10⁻⁷ | 4.2·10⁻⁶, 1.3·10⁻⁶ | 4.84 / 0.90 / 42.7 s |

(`data/ep80_*_hi.json`.) The 80-bit chain costs 5 to 7 times DP and is 8 to 9 times faster than QP. With DP integrals its error is that of the hybrid test, as expected: the integrals' rounding still limits. The gate (plan: well below 10⁻⁶, at an acceptable cost) waits for the TIRust session's double-double output of the raised-power integrals (`RAW_LO`). The test then:
- 80-bit MadLoop with `RAW + RAW_LO`, at MadLoop's improved point;
- if that is not enough, the input low words (`S_LO`, `P_LO`) as well.

### With TIRust's low words (RAW_LO)

The TIRust session built `EmbedderOptions::low_words`: TIRust a8937ea (local branch `hs/pwave-low-words`, #56's head 5025e12 plus one commit), with the bridge patch `madloop-tirust-gate-relation-tiers-low-words.patch`. Its new raised-power entry `tirust_tensor_evaluate_powers_dd` returns:
- `RAW`, bit for bit the binary64 result;
- `RAW_LO`, the low words of the precision tier that served the call;
- `ROUTE`: 0 for binary64 (with `RAW_LO` = 0), 2 for double-double, 3 and 4 for 256 and 512 bits.

Scratch build `$SP/pw59` (`pwave_fd/build.sh` with `FOLLOWUP=0001-followup-pwave-collinear.patch`, `ONELOOP_REV=cdbc912`; the vendor-only part of the patch). `lowwords.py` adds an entry that sums `RAW + RAW_LO` in quadruple precision (kind 10 after `ep_retarget.py`). The data are in `data/lowwords_*_a8937ea.json`.

| 24 points, 1 TeV: largest / median | u ū → χ_c0 g | g g → χ_c0 g | time per point (u ū / g g) |
|---|---|---|---|
| DP production | 2.7·10⁻⁵ / 3.7·10⁻⁶ | 4.2·10⁻⁶ / 1.3·10⁻⁶ | 0.071 / 0.90 s |
| 80-bit, `RAW` | 1.3·10⁻⁵ / 3.8·10⁻⁶ | 1.3·10⁻⁶ / 4.9·10⁻⁷ | 0.52 / 5.2 s |
| 80-bit, `RAW + RAW_LO` | 5.5·10⁻⁶ / 4.3·10⁻⁷ | 4.8·10⁻⁷ / 3.7·10⁻⁸ | 0.53 / 5.1 s |
| QP MadLoop, `RAW + RAW_LO` | 1.1·10⁻⁵ / 1.3·10⁻⁶ | 7.0·10⁻⁷ / 5.5·10⁻⁸ | 3.9 / 39 s |
| … only the tier-served calls (route ≥ 2) | 1.1·10⁻⁵ / 1.3·10⁻⁶ | 7.0·10⁻⁷ / 5.5·10⁻⁸ | |
| … only the binary64-served calls (route 0) | 1.6·10⁻⁹ / 1.2·10⁻¹⁰ | 2.4·10⁻⁹ / 4.0·10⁻¹¹ | |

- **Points at or above 10⁻⁶:**
  - u ū: 22/24 in DP, 17/24 with 80-bit `RAW`, 9/24 with 80-bit `RAW + RAW_LO`;
  - g g: 14/24 in DP, 1/24 with 80-bit `RAW`, none with 80-bit `RAW + RAW_LO`.
- **Cost:** the low words add nothing measurable (0.525 vs 0.517 s/pt).
- **Routes:** at point 0, 20 of the 45 raised-power calls of u ū (92 of 183 for g g) are served by binary64 (route 0). The rest take the double-double tier (route 2).
- **The split** (`ONIA_DBG_LW=3` and `4`) puts all of what is left in the tier-served calls. The binary64-served calls contribute at most 2.4·10⁻⁹, so sending them to the tiers would not help. TIRust's `RAW + RAW_LO` is good to 10⁻²² of 512 bits at the same binary64 inputs, so the error left is that of the **inputs' rounding**: the loop momenta rounded to binary64, and the shifts and invariants formed from them in binary64 arithmetic. Binary128 cannot measure it directly: it refuses the relation lanes of binary64-rounded inputs (`ONIA_DBG_LW=2`, code 420).
- **Consequence:**
  - g g → χ_c0 g passes the gate (< 10⁻⁶ everywhere) at 5.7× DP;
  - u ū → χ_c0 g needs the input low words as well: shifts and invariants in extended precision from MadLoop, consumed by TIRust's double-double tier. With 80-bit inputs, 11 more bits should take its 10⁻⁵ to about 10⁻⁸.

### With extended inputs (the 80-bit gate)

TIRust's next candidate takes the integral inputs as hi + lo words: TIRust d17085a (local branch `hs/pwave-low-words`, a8937ea plus one commit), with the bridge patch `madloop-tirust-gate-relation-tiers-low-words-extended-inputs.patch` and the scratch build `$SP/pw61`. Its entry `tirust_tensor_evaluate_powers_ext` adds:
- `shifts_lo` and `invariants_lo`;
- `input_bits`, the precision that hi + lo carries.

A call with nonzero low words always goes to the precision tiers. `lowwords.py --ext` adds `ML5_0_TIRUST_TENSOR_POWERS_EXT`, a copy of the binary128 raised-power routine that forms the shifts and invariants in its own precision (kind 10 in the 80-bit copy) and passes them split. The data are in `data/extinputs_*_d17085a.json`.

| 24 points, 1 TeV: largest / median | u ū → χ_c0 g | g g → χ_c0 g |
|---|---|---|
| DP production | 2.7·10⁻⁵ / 3.7·10⁻⁶ | 4.2·10⁻⁶ / 1.3·10⁻⁶ |
| QP MadLoop, extended raised-power inputs (106 bits) | 0 / 0 (identical to QP) | 1.6·10⁻¹² / 0 |
| … with binary64 unit-power integrals | 6.4·10⁻⁶ / 1.3·10⁻⁶ | 3.5·10⁻⁷ / 6.3·10⁻⁸ |
| … with the unit powers on extended inputs too | 5.5·10⁻¹¹ / 1.3·10⁻¹¹ | 6.3·10⁻¹² / 1.9·10⁻¹² |
| 80-bit, `RAW + RAW_LO` | 5.5·10⁻⁶ / 4.3·10⁻⁷ | 4.8·10⁻⁷ / 3.7·10⁻⁸ |
| 80-bit, extended raised-power inputs (64 bits) | 6.4·10⁻⁶ / 1.3·10⁻⁶ | 3.5·10⁻⁷ / 6.3·10⁻⁸ |
| **80-bit, every TIRust call on extended inputs** | **2.2·10⁻⁸ / 4.1·10⁻⁹** | **2.2·10⁻⁹ / 6.0·10⁻¹⁰** |

- **Extended inputs for the raised powers alone change nothing in the 80-bit copy.** The QP hybrid shows why: with the unit-power integrals in binary64 from rounded inputs, as the 80-bit copy takes them, it gives the same numbers to the digit. The binary64 rounding of the unit-power inputs is the next floor, and MadLoop's 80-bit arithmetic stays below it.
- **TIRust already serves unit powers through the extended entry** (powers all 1): every call goes to the double-double tier. The result is rounded to binary64, since unit-power workspaces keep no low words, and that is enough: 5.5·10⁻¹¹ in the QP hybrid. In the 80-bit copy, `ONIA_EP_LOWWORDS=3` sends every call there.
- **The gate is passed.** With every call on extended inputs (64 bits), no point is at 10⁻⁶ or above; u ū has 3 of 24 at 10⁻⁸ or above, g g none.
- **No refusals:** over the 24 points every call took route 2 (u ū: 1080 raised-power, 1896 unit-power calls; g g: 4392 and 7176); every return code is 120.
- **Cost** (`lowwords_cost.py`, four alternating pairs of 24-point batches, median time per point): every call on extended inputs is +3% against `RAW + RAW_LO` for u ū (0.543 vs 0.522 s) and +0% for g g (5.44 vs 5.45 s). Raised powers alone: +1% and −0.4%. The 80-bit mode as a whole stays at 7.6× (u ū) and 6.0× (g g) DP.

## The generated switch (Phase 2)

MadLoop now generates the 80-bit mode. It is off by default and switched on with `#PWaveExtendedPrecision .TRUE.` in `Cards/MadLoopParams.dat`:
- **EP code:** `madgraph/loop/onia_ep.py` derives it from the quadruple-precision (MP) code when the output is written. It renames `MP_` to `EP_` and changes the kind to `EPK = SELECTED_REAL_KIND(18)`: 10 on x86, 16 where there is no x87 format.
- **Templates:** they hold the dispatch of the double-precision passes (CTMODE 1, 2) and the 80-bit TIRust routines. Every call, unit powers included, goes to `tirust_tensor_evaluate_powers_ext` with 64-bit inputs.
- **When the code is written:** P-wave outputs with one P-wave state, whose TIRust installation has `raised_power_extended_inputs` in its metadata. Other P-wave outputs warn when the switch is on and ignore it.

Outputs without the EP code are unchanged, apart from the new card parameter and the warning in P-wave outputs. `ep_switch.py` measures the switch on generated outputs (TIRust d17085a with the bridge patch, `$SP/pw61`, with the metadata flag added). The data are in `data/ep_switch_*_chic0g_1000_d17085a.json`.

| 24 points, 1 TeV: largest / median (s per point) | u ū → χ_c0 g | g g → χ_c0 g |
|---|---|---|
| off, `CTModeRun 1` | 2.7·10⁻⁵ / 3.7·10⁻⁶ (0.078) | 4.2·10⁻⁶ / 1.3·10⁻⁶ (0.97) |
| off, `CTModeRun -1` (the default stability test) | 2.9·10⁻⁵ / 5.3·10⁻⁶ (0.102) | 4.1·10⁻⁶ / 1.1·10⁻⁶ (1.03) |
| **on, `CTModeRun 1`** | **2.2·10⁻⁸ / 4.1·10⁻⁹** (0.495) | **2.2·10⁻⁹ / 6.0·10⁻¹⁰** (5.28) |
| on, `CTModeRun 2` (loop direction reversed) | 3.2·10⁻⁶ / 2.0·10⁻⁷ (0.545) | 3.4·10⁻⁷ / 8.4·10⁻⁸ (5.62) |
| on, `CTModeRun -1` | 1.6·10⁻⁶ / 9.8·10⁻⁸ (0.539) | 1.7·10⁻⁷ / 4.1·10⁻⁸ (5.45) |

- **One 80-bit pass reproduces the prototype to the digit**, at 6.3× (u ū) and 5.4× (g g) the double-precision time.
- **The reversed pass loses about two digits to TIRust's check pass.** For `CTModeRun 2` (and 5) MadLoop asks TIRust for its check pass. That pass moves every invariant by the relative factor 1 ± 2⁻⁵² (`CHECK_INVARIANT_SHIFT = f64::EPSILON` of the bridge, one binary64 unit), so the stability test sees the sensitivity to the rounding of binary64 inputs. The extended entry keeps that shift with 64-bit inputs. The reversed pass then has the error of a binary64-rounded input, the level of the 80-bit test with binary64 inputs above.
- **The stability test returns the mean of the two passes**, so with the switch on it gains 18× (u ū) and 24× (g g) in the worst point and 54× and 27× in the median, instead of three digits, at 5.3× the double-precision time for both. It stays conservative: its estimate is below the true error at 1 of 24 u ū points (1.3·10⁻⁹ against 1.2·10⁻⁹).
- **Without the check pass** in the reversed 80-bit pass (a diagnostic, not the generated code), that pass is as accurate as the forward one: u ū 2.4·10⁻⁸ / 2.7·10⁻⁹, and the stability test returns 2.1·10⁻⁸ / 2.2·10⁻⁹. But the two passes then agree too well. The estimate falls below the true error at 7 of 24 points, 4 of them with an estimate of 0. The check pass is needed; its shift should follow the input precision, 2⁻⁶³ for 64-bit inputs.
- **No refusals:** every return code is 110 (`CTModeRun 1, 2`) or 213 (`-1`).

### With the check-pass shift at the input precision (bridge v2)

TIRust then changed its bridge only, keeping TIRust at d17085a. For a check pass with nonzero low words, the extended entry now shifts hi + lo by 1 ± 2^-(input_bits-1), 2⁻⁶³ at 64 bits. The binary64 check pass (2⁻⁵²) and the other entries are unchanged.
- **Patches:** `madloop-tirust-ext-check-shift.patch` on top of 84e94158, or all together `madloop-tirust-gate-relation-tiers-low-words-extended-inputs-v2.patch`, both in `/shared/external/iregi/pwave-patches`.
- **Build:** scratch `$SP/pw62`. Its bridge source is that of `$SP/pw61` plus the incremental patch, file for file.
- **Data:** the outputs were regenerated without any MadLoop change; the results are in `data/ep_switch_*_chic0g_1000_d17085a_v2.json`.

| 24 points, 1 TeV: largest / median (s per point) | u ū → χ_c0 g | g g → χ_c0 g |
|---|---|---|
| off, `CTModeRun -1` | 2.9·10⁻⁵ / 5.3·10⁻⁶ (0.109) | 4.1·10⁻⁶ / 1.1·10⁻⁶ (1.01) |
| on, `CTModeRun 1` | 2.2·10⁻⁸ / 4.1·10⁻⁹ (0.499) | 2.2·10⁻⁹ / 6.0·10⁻¹⁰ (5.10) |
| on, `CTModeRun 2` | 2.6·10⁻⁸ / 2.1·10⁻⁹ (0.528) | 2.5·10⁻⁹ / 8.5·10⁻¹⁰ (5.34) |
| **on, `CTModeRun -1`** | **2.0·10⁻⁸ / 2.5·10⁻⁹** (0.536) | **2.3·10⁻⁹ / 6.2·10⁻¹⁰** (5.32) |

- **The reversed pass is now as accurate as the forward one,** and the stability test keeps the full gain: 1450× (u ū) and 1800× (g g) in the worst point, about three digits, at 4.9× and 5.2× its double-precision time. Before the change it gained 18× and 24×.
- **Unchanged otherwise:** the double-precision passes and the forward 80-bit pass give the same numbers as before. The cost does not change (0.536 against 0.539 s, 5.32 against 5.45 s), and there are no refusals (codes 110 and 213).
- **MadLoop's accuracy estimate behaves as in double precision.** It is an estimate, not a bound, and it is never 0:

| ratio of MadLoop's estimate to the true error | points below 1 | lowest | median |
|---|---|---|---|
| u ū, double precision | 3/24 | 0.32 | 3.3 |
| u ū, 80-bit, shift 2⁻⁵² | 1/24 | 0.96 | 5.5 |
| u ū, 80-bit, no check pass | 7/24 | 0 | 2.9 |
| u ū, 80-bit, shift 2⁻⁶³ | 5/24 | 0.23 | 1.9 |
| g g, double precision | 1/24 | 0.10 | 3.0 |
| g g, 80-bit, shift 2⁻⁵² | 1/24 | 0.80 | 9.2 |
| g g, 80-bit, shift 2⁻⁶³ | 5/24 | 0.32 | 2.0 |

  With the 2⁻⁵² shift the estimate was larger only because it measured the error that the shift itself added.

## Reproducer for TIRust

`data/repro_doubled_pentagons.json.gz` is written by `repro.py` (`python3 repro.py OUTDIR OUT.json.gz 0 7`, on an output patched by `instrument.py`). It holds the ten DP calls with powers and five lines of u ū → χ_c0 g at points 0 and 7 of setup A, in both passes. For each call it records:
- the inputs MadLoop passes: powers, masses², invariants, shifts and μ² (= 10⁶ GeV²);
- the DP array;
- the QP array of the same call, from `CTModeRun 4` or 5 and rounded to double.

All numbers are hex doubles.

There are five families of pentagons, each twice per pass, all of rank 4, with m_c² = 2.4025 GeV². In the primary pass:

| Line powers | Masses² |
|---|---|
| (1,1,1,1,2) | (0,0,0,0,m²) |
| (1,1,1,2,1) | (0,0,0,m²,m²) |
| (1,1,1,1,2) | (0,0,0,m²,m²) |
| (1,1,2,1,1) | (0,0,m²,0,0) |
| (1,1,1,2,1) | (0,0,0,m²,0) |

The finite parts of the amplitude, relative error against QP:

| Point | Primary pass | Reversed pass |
|---|---|---|
| 0 | 9.3·10⁻⁵ | 1.5·10⁻⁵ |
| 7 | 3.4·10⁻² | 0.45 |

The largest array deviation, relative to the largest finite entry of the call, is 2·10⁻¹⁰.

In the reversed pass TIRust also shifts the invariants itself (`CHECK_PASS`), and its DP array includes that shift, while the recorded inputs are the unshifted ones.
