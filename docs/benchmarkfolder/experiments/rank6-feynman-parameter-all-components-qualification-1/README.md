# Independent check of all 210 N6/rank-six finite components

Direct Feynman-parameter integration corroborates **all 210 TIRust Cartesian
finite components** of the retained synthetic fixture 108. At order 18, every
real/imaginary part passes both the existing `3e-6` criterion and the `1e-8`
target: **420/420**, with maximum symmetric residual **9.625e-20**.

The uncached COLLIER result passes **298/420** parts at `3e-6` and **92/420** at
`1e-8`. Its 122/328 respective disagreements reproduce the historical pattern.
The maximum residual is **1.328504e-3**, at index 161, imaginary part. This
independent representation supports a COLLIER accuracy limitation on this
fixture, consistent with its already retained accuracy flag−1. It does not
assign a specific internal algorithm bug.

| Gaussian order | TIRust passes `3e-6` / `1e-8` | Maximum TIRust residual | COLLIER passes `3e-6` / `1e-8` |
|---:|---:|---:|---:|
| 8 | 420 / 420 | `1.711e-12` | 298 / 92 |
| 16 | 420 / 420 | `9.122e-20` | 298 / 92 |
| 18 | 420 / 420 | `9.625e-20` | 298 / 92 |

The maximum change from order 16 to 18 is **7.126e-20** symmetric residual.
The seven overlapping scalar-implementation controls agree at both shared
orders 8 and 18, within `6.172e-17` and `4.765e-17`, respectively. The original
four normalization, odd-rank and metric controls are among those comparisons.

## Analytic error and arithmetic are reported separately

The defining parameter formulas, Gaussian rules and exact analytic error bounds
are inherited from the immutable
[scalar proposal](../rank6-feynman-parameter-proposal-1/README.md). The new
[vector source](../rank6-feynman-parameter-all-components-proposal-1/README.md)
integrates 296 unique primitive moments, then reconstructs the 210 components
using 660 exact dyadic coefficients. No reduction library or scalar-master
kernel is called in this independent calculation.

At order 18, allowing the conservative analytic quadrature error alone preserves
every `3e-6` classification:

| Comparator | Conditional agreement | Conditional disagreement | Inconclusive |
|---|---:|---:|---:|
| TIRust QP | 420 | 0 | 0 |
| COLLIER | 298 | 122 | 0 |

At the `1e-8` target, the corresponding analytic-error-only counts are 357/0/63
for TIRust and 90/326/4 for COLLIER. Point-estimate passes at `1e-8` must not be
confused with a rigorous error enclosure at that threshold.

These intervals exclude finite-precision node, weight and arithmetic error.
The actual vector calculation uses NumPy 2.4.2 `longdouble` with a **64-bit
significand**, not 113-bit Fortran QP. Analytic synthetic controls cover all 296
primitives and 210 components, including nonconstant numerators with exact
Dirichlet moments, permutations, reflections and partial chunks. The existing
seven-component proof additionally retains an 80-digit Decimal arithmetic
cross-check. Convergence and same-order independent implementation comparisons
support the practical numerical cross-check; there is no formal whole-program
floating-point certificate.

## Complete retained comparison

`analysis.json.gz` contains all 420 part comparisons against both backends at
every completed order, every absolute difference, unchanged symmetric residual,
analytic-tail-only enclosure, rank summary and successive-order difference.
`summary.json` preserves both criteria and all three analytic-interval outcomes.

No finite native part is exactly zero in this fixture. The parser nevertheless
reports exact numerical zero flags and absolute differences. Same-rank maxima
from the frozen native reference arrays provide an additional absolute-scale
diagnostic. This never adds a floor, clips a value, changes an acceptance count
or asserts a mathematical zero.

The exact native finite words, effective inputs, previous DP Gram refusals and
IEEE flags remain unchanged in the
[historical audit](../generated-rank6-poles-proposal/attempt1/common-input-proposal/finite-audit-1/README.md).
The parser joins every coefficient index and power tuple and decodes the
binary128 reference words exactly. An independent Fraction-based audit
recomputes all 420 comparisons and all 210 analytic bounds. Both native answers
remain present, including every COLLIER disagreement.

COLLIER is the unchanged 1.2.9 library with global caching disabled for this
fixture. No cache code is changed. The separately documented cached N7/rank-nine
issue is not an established cause here.

## Execution and reproduction

One admitted worker evaluated orders 8/16/18 in **29.110 seconds**, on nonreserved
CPU 10, under a 4 GiB address-space limit and 180-second aggregate cap. It exited 0
and was reaped; no retry occurred. Source fingerprints before and after match.
The worker made **zero native reducer calls**. This wall time describes an
independent diagnostic, not MadLoop performance acceptance.

Exact original JSON, controller, empty stdout and execution receipt are retained
in `data/`. Deterministic gzip preserves the original result bytes, SHA
`3d1c56b4e3fbecc4aac439714cbbe1dd6d273b6a4a5d03e85984f33477168cee`.
No quadrature is performed during replay or analysis. Completed raw orders also
remain accessible if a later comparison or timed worker fails; unavailable
orders are explicitly reported rather than filled with zeros.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-all-components-qualification-1/replay.py
```

The frozen source proposal gives the separately gated bounded quadrature
command. A reproduction must use a new output file and the same external
resource controls; replay does not launch that command.

This result covers the same undotted synthetic N=6/rank-six finite tensor only.
It does not certify physical complex-mass amplitudes, arbitrary kinematics,
stability rescues, poles or runtime acceptance. The original evidence packages
remain immutable.
