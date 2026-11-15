# g g > z z z z [noborn=QCD]: TIRust in MadLoop

Check of a report from pyAmpliCol, where TIRust "failed closed" (status Unstable) on 4 of 128 evaluations of this process. pyAmpliCol main (`c458db8dc4`, TIRust `a5d4afb`) calls TIRust's `DirectWorkspace` with its precision tiers off on purpose and leaves refused rows to its own rescue ladder (binary64 TIRust, NinjaRS, then double-double and MPFR), so a binary64 refusal is expected there. MadLoop's bridge enables TIRust's `EmbedderOptions::precision_tiers` (double-double 106 bits, then 256 and 512 bits).

Source `777e081b0c`; TIRust pin 9a781998 with the genuine-QP 512-bit retry; libmadloop_tirust.so sha256 a01c763975e6f5c4.... Per-point values, accuracies and TIRust route deltas: [`ggzzzz-tirust.json`](ggzzzz-tirust.json). Timings are shared-host and provisional.

## Setup

```bash
source tools/tirust_env.sh
export FFLAGS='-w -fPIC -O3' LDFLAGS='-lstdc++ -Wl,--no-relax'
# MG: set loop_optimized_output True; set output_dependencies external; set ninja/collier/ninjars/tirust <libs>
#     import model loop_sm; generate g g > z z z z [noborn=QCD]; output standalone_fortran <dir>
# Points (tools/tirust_benchmark_points.py, masses [MZ=91.188]*4):
#   warm-up: generate_points(count=3, energy=1000, seed=20261114)
#   1-40: generate_points(count=40, energy=1000, seed=20261115); 41-60: count=20, energy=500, seed=20261116
#   61-62: generate_stress_points(energy=1000, seed=20261117)   # threshold, small Gram
# Per lane: check_sa.f = tirust_stress.render(tools/tirust_stress_driver.f, routes=TIRust in chain),
#   MadLoopParams: MLReductionLib=<lane>, CTModeRun=-1; make check
taskset -c <cpu> ./check warm.dat points.dat checked
```

## Lanes

| Lane | MLReductionLib | Points | Return codes | s / ordinary point (approx.) |
|---|---|---|---|---|
| TIRust | 9 | 62/62 | 213×61, 323×1 | 31 |
| TIRust-first chain | 9\|6\|7\|1 | 62/62 | 213×61, 323×1 | 32 |
| Ninja | 6 | 62/62 | 216×60, 420×2 | 141 |
| COLLIER | 7 | 62/62 | 217×60, 410×2 | 30 |
| NinjaRS | 8 | 60/62 | 212×60 | 118 |
| CutTools | 1 | 0/62 | – | – |

Return codes are MadLoop format 2 (100H+10T+U): TIRust 213 is stable in double precision (U=3 is shared with IREGI), 323 a quadruple-precision rescue, 4xx exceptional.

## Ordinary points (1–60)

- TIRust returns 213 at 60 of 60 points. The TIRust-first chain gives identical codes and values, so it never falls back.
- Route totals over the 60 points: primary 13898 direct + 13482 method choice + 7 double-double; check 36668 + 33860 + 32. Refused 0, errors 0, 256/512-bit tiers 0.
- The double-double tier is needed at 2 of 60 points (11, 44); every attempt there succeeds. A standalone run of those two points reproduces the same values and route counts.
- Elsewhere TIRust agrees with the median of Ninja, COLLIER and NinjaRS to 3.9e-07 or better.

| Point | TIRust code, accuracy | TIRust routes (DD primary/check) | TIRust vs COLLIER | Ninja vs TIRust | NinjaRS vs TIRust |
|---|---|---|---|---|---|
| 11 `ordinary-0010` √s=1000 | 213, 3.4e-08 | 3/12 | 8.6e-08 | 2.0e-05 (own 7.2e-06) | 2.2e-05 (own 2.8e-04) |
| 44 `ordinary500-0003` √s=500 | 213, 8.5e-10 | 4/20 | 1.9e-08 | 6.5e-06 (own 2.3e-05) | 6.1e-06 (own 2.3e-05) |

## Stress points

| Point | TIRust | TIRust-first chain | Ninja | COLLIER | NinjaRS | CutTools |
|---|---|---|---|---|---|---|
| 61 `stress-threshold` √s=364.752 | 323: 1.969e-13 (acc 1e-08) | 323: 1.969e-13 (acc 1e-08) | 420: 2.518e-01 (acc 1e-03) | 410: 4.281e-11 (acc 1e+00) | stopped | stopped |
| 62 `stress-small-gram` √s=1000 | 213: 2.900e-15 (acc 1e-04) | 213: 2.900e-15 (acc 1e-04) | 420: 3.206e-15 (acc 2e-01) | 410: 1.350e-11 (acc 6e-02) | stopped | stopped |

Wall time of the stress points (from row timestamps): b6-auto-from38 point 62: 4.36 h; b9-auto point 61: 13.18 h; b9-auto point 62: 32 s; chain9671-auto point 61: 13.98 h; chain9671-auto point 62: 31 s.

- Point 61 (threshold): TIRust rescues in quadruple precision (323, accuracy 1.5e-08), with the 256- and 512-bit tiers on the double-precision input (primary/check: 23/96 at 256 bits, 96/288 at 512 bits). The value is bit-identical to an independent run of the TIRust session on pin `a5d4afb` with the bridge's check-pass QP cache (`78a03130ec`). No other reducer gives a value here: Ninja and COLLIER flag the point exceptional, and the NinjaRS lane was stopped in it. By the TIRust session's profile the 13 h are almost all in the quadruple-precision check pass, which this build did not cache; `78a03130ec` caches it.
- Point 62 (small Gram): TIRust accepts in double precision (213, accuracy 1.1e-04, below `MLStabThres`=1e-3), all through the direct engine and method choice with no tier attempt. Method-choice lanes (the small-Gram expansion or the alternative PV) are accepted without a per-lane estimate and left to MadLoop's direction check. A forced quadruple-precision TIRust run of this point (`CTModeRun=4`, unchecked; code 120, 0.27 h) gives 2.89960042336768975e-15. Its path is independent of the double-precision route: the bridge's QP entry runs the plain direct engine at 256 bits (gate 2^-136, 512-bit retry), without method choice or the expansion. The accepted double-precision value differs by +1.7e-05, inside its estimate. Ninja (420) differs by +1.1e-01 and COLLIER (410) by a factor 4.7e+03; both flag the point exceptional.
- So TIRust's accepted value at point 62 is right to its stated accuracy. Point 61 has no independent check: no other reducer returns a value there, so it rests on TIRust's own quadruple-precision estimate.

## Notes

- CutTools did not finish the first of its three warm-up points in 6 h and was stopped.
- The Ninja lane was killed at point 37 by an operator error and resumed from point 38 on the same filter (lanes `b6-auto` and `b6-auto-from38`, merged by point index).
- The lane wrappers were stopped at 16:06 UTC, just before their 6 h timeout, so the drivers could finish the stress points; their outputs were collected afterwards.
- The NinjaRS lane was stopped on 2026-11-15 at 13:35 UTC, 25 h into stress point 61 without a value, to free the single unlicensed Symbolica thread for the final requalification; it did not run point 62.
- Ordinary-point times include two double-precision evaluations per point (forced stability check) and are not comparable to `check profile` numbers. MadLoop evaluates a loop-induced process helicity by helicity: for tensor-integral reducers (TIRust, COLLIER) `TIR_interface` caches the tensor-integral coefficients per loop group and reuses them across helicities, while the integrand reducers (Ninja, NinjaRS, CutTools) reduce every helicity again. That is the likely reason for the factor of about 4 here, against 7% for g g > g g g g (not measured separately).
