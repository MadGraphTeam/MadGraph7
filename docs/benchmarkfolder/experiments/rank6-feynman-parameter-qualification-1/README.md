# Direct parameter integration of the disputed rank-six components

An independent Feynman-parameter calculation corroborates **TIRust on all
seven sampled finite tensor components** of the retained synthetic N=6,
rank-six fixture108. At the final order, its largest real/imaginary symmetric
residual against TIRust QP is **4.762e-17**. The four scalar, odd-rank and metric
controls pass the `1e-8` target before interpreting the three disputed tensors.

The same calculation disagrees with the uncached COLLIER result at the existing
`3e-6` criterion on **five of the six real/imaginary parts** of indices55/111/161.
The largest residual is **1.328504e-3**. This supplies independent numerical
evidence for a comparator accuracy limitation on these sampled components;
COLLIER already reported a requested-accuracy miss, flag−1. It does not identify
a specific internal algorithm bug or establish a result for every tensor
component.

| Component | Powers | Maximum observed residual vs TIRust QP | Maximum vs COLLIER |
|---|---|---:|---:|
| 55 | `(1,1,1,1)` | `4.762e-17` | `1.254131e-3` |
| 111 | `(1,1,1,2)` | `2.061e-17` | `1.216841e-4` |
| 161 | `(3,1,1,1)` | `4.354e-17` | `1.328504e-3` |

The original thirty-row experiment, exact effective-input words, all 122
historical discrepant parts, DP Gram refusals, IEEE observations and both
native answers remain unchanged in the
[common-input finite audit](../generated-rank6-poles-proposal/attempt1/common-input-proposal/finite-audit-1/README.md).
This experiment ran no native reducer and used no amplitude or model-card
initialization. It directly integrates the independently derived smooth
parameter representation, with the supplied exact shifts and squared masses.
The invariant matrix is mutually consistent with those four-dimensional shifts.
The finite normalized measure and MadLoop component convention include the
odd-rank sign and Minkowski trace terms; no amplitude factor `1/(16*pi^2)` is
introduced.

The defining formulas, exact positive denominator bounds and Gaussian error
derivation are frozen in the
[source proposal](../rank6-feynman-parameter-proposal-1/README.md), index
`03ffcd9f8b88855b59d78387cde255460a36d35cca251fce093741a9c0065db3`.
The separately audited
[offline parser](../rank6-feynman-parameter-analysis-proposal-1/README.md)
checks the component mapping, exact binary128 decoding and analytic-bound join.

Binary64 quadrature used orders4/6/8/10 and the predeclared order18 endpoint.
An 80-digit Decimal calculation at order8 agrees with binary64 at the same order
within **6.173e-17** symmetric residual across all fourteen parts. Every order,
Gaussian node/weight record and raw Decimal string is retained. At order18,
the rigorous analytic Gaussian-error bound alone enlarges the TIRust residual
to at most **3.558e-7**, still below `3e-6`; all five COLLIER misses persist after
allowing that same analytic error.

That last enclosure excludes finite-precision node, weight and arithmetic
error. Those effects are checked empirically by the Decimal comparison and
convergence; there is no formal whole-program floating-point certificate.
Together, the checks support the practical sampled-component cross-check at
`3e-6`. They do not qualify all210 tensor components, physical complex-mass
amplitudes, stability/rescue policies or runtime acceptance.

| Closed calculation | Wall time | Native reducer calls |
|---|---:|---:|
| Binary64 orders4/6/8/10 | `7.431 s` | 0 |
| Decimal80 order8 | `6.333 s` | 0 |
| Binary64 order18 | `120.443 s` | 0 |

These are diagnostic wall times, not MadLoop/reducer performance benchmarks.
The three workers used nonreserved CPU10 and each exited0 within its
180-second cap. Original root execution receipts and terminal confirmations are
retained alongside deterministic gzip copies of the exact JSON results.
No additional quadrature was run during analysis or packaging.

COLLIER is the unchanged **1.2.9** library with its cache disabled for this
fixture. The separately documented cached N=7/rank-nine issue is not an
established cause here; no cache code was changed.

Portable verification from the MadGraph checkout performs no numerical
integration, build or reducer execution:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-qualification-1/replay.py
```

The verifier checks every owned byte and source binding, decompresses the
original records, joins each execution receipt to its actual result/log hashes,
replays all comparisons and requires byte-equivalent JSON content for the
stored analysis and numerical summary. The source proposal retains the
commands needed to reproduce the original bounded quadrature separately.
