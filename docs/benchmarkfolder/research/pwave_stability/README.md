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

The TIRust session's answer to the reproducer below. The runs in this section use 14f2696, with `oneloop` 07fc754. 8b436f9, in PR #56, gives the same results (see "On 8b436f9 (PR #56)" below):
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

### On 8b436f9 (PR #56)

The relation tiers are now in PR #56. 8b436f9, its head at the time of these runs, is 14f2696 plus merges of TIRust's main, and the merges re-pin `oneloop` from 07fc754 to cdbc912 (a small-momentum C0 series).
- **Build.** The bridge's pole kernels (`vendor/tirust/pole-kernels`) still pin `oneloop` 07fc754. With cdbc912 in TIRust, two versions of the crate meet, and `iregi-rust` does not compile. `build.sh` therefore takes `ONELOOP_REV=cdbc912ab7b8327a23fe5fc08ddb0c58b9d55a21`, which moves the pole kernels of the scratch copy to the same revision; `vendor/tirust` itself is unchanged.
- **Same results.** The 24-point diagnosis (`data/estim_*_8b436f9.json`: u ū at 1 TeV and 20 GeV, g g at 1 TeV, all three passes and QP) is bit-identical to 14f2696. 22 of the 24 stability profiles have bit-identical estimates. In the two g g profiles at 20 GeV, the new `oneloop` changes estimates of about 10⁻¹² by factors of 0.6 to 1.6, with the same maximum and the same points above 10⁻⁹. The P-wave checks (`data/pw_check_8b436f9.json`) are bit-identical in 15 of the 16 cases, and g g → χ_c0 g differs by 6·10⁻¹⁶.
- **Same cost.** In production settings (`data/cost_8b436f9_vs_14f2696_*.json`): u ū → χ_c0 g takes 96–97 ms per point with either build, at both energies, and g g → χ_c0 g at 20 GeV 1.06 s against 1.10 s.

The head of PR #56 is now 5025e12, one commit on 8b436f9 and not run here. In it, a relation-tier lane that every precision tier refuses comes back unstable instead of with its binary64 value. According to the TIRust session, no lane of the reproducer takes that path. A point of the profiles that did take it would be reported unstable (or rescued in QP) instead of keeping the binary64 value.

The report's dotted curves are the 8b436f9 profiles (`data/raw/stability/*56_*.json`). The 14f2696 profiles are kept in `data/profiles_14f2696/`.

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
