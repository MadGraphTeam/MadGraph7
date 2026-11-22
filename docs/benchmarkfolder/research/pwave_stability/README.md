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
