# Independent parameter-integral discriminator: source proposal

This packet proposes a direct, independent Feynman-parameter calculation for the retained **synthetic N=6, rank-six fixture 108**. It contains no new integral evaluation, reducer execution, build, or performance result. The existing disagreement and all original evidence remain unchanged. The aim is to distinguish the finite Cartesian components at MadLoop indices **55, 111 and 161**, after first checking four lower-rank controls. A future result must keep its mathematical quadrature error separate from its finite-precision arithmetic error.

The exact input is [`../generated-rank6-poles-proposal/attempt1/common-input-proposal/fixtures.json`](../generated-rank6-poles-proposal/attempt1/common-input-proposal/fixtures.json). The retained comparator values, finite disagreement and qualification limits are in [`finite-audit-1`](../generated-rank6-poles-proposal/attempt1/common-input-proposal/finite-audit-1/README.md). Neither a reducer success status nor a precision-convergence observation establishes the defining integral by itself.

## Defining integral and conventions

Let the six denominators be `D_i=(q+p_i)^2-m_i^2`, with Minkowski metric `diag(1,-1,-1,-1)`. The supplied shifts and complex squared masses are used exactly; each mass has a strictly negative imaginary part. On the five-dimensional simplex `x_i>=0`, `sum(x_i)=1`, define

```
P_mu(x) = sum_i x_i p_i,mu
F(x)    = sum_i x_i m_i^2 - sum_(i<j) x_i x_j s_ij,
s_ij    = (p_i-p_j)^2.
```

The loop shift is `l=q+P`, so the original numerator is `(l-P)^a`. Combining denominators contributes `Gamma(6)`; it cancels the denominator in the normalized loop-moment formula. For `2k` loop indices the remaining coefficient is

```
(-1)^(6-k) Gamma(4-k) / 2^k * symmetrized metric products * F^(k-4).
```

The MadLoop `(-1)^rank` convention cancels the shift's `(-1)^(rank-2k)`. Odd loop moments vanish. Tensor weights are actual components, without a multinomial multiplicity. These are the finite OneLOop-normalized moments; the amplitude-level `1/(16*pi^2)` factor is absent. The normalization and sign conversion follow the pinned TIRust `convention.rs` and `types.rs` contracts. All these N=6 moments through rank six are UV convergent, and the positive massive parameter polynomial excludes an IR singularity; the finite defining integral therefore needs no pole subtraction or scale-dependent finite remnant.

Write `J_a,p = integral_simplex P^a F^(-p) dx`, using simplex volume `1/120`. The selected moments are:

| MadLoop index | Powers `(a0,a1,a2,a3)` | Direct expression |
|---|---|---|
| 0 | `(0,0,0,0)` | `6 J_(0,0,0,0),4` |
| 1 | `(1,0,0,0)` | `6 J_(1,0,0,0),4` |
| 5 | `(2,0,0,0)` | `6 J_(2,0,0,0),4 - J_(0,0,0,0),3` |
| 7 | `(0,2,0,0)` | `6 J_(0,2,0,0),4 + J_(0,0,0,0),3` |
| 55 | `(1,1,1,1)` | `6 J_(1,1,1,1),4` |
| 111 | `(1,1,1,2)` | `6 J_(1,1,1,2),4 + J_(1,1,1,0),3` |
| 161 | `(3,1,1,1)` | `6 J_(3,1,1,1),4 - 3 J_(1,1,1,1),3` |

The index/powers join is checked against the frozen finite-audit monomial list, rather than assuming TIRust's own packed order. The opposite signs in the two squared-component controls test the Minkowski metric. The odd-rank control tests the MadLoop sign. Low-rank agreement must precede any interpretation of the three disputed components.

## Exact parameter-domain bounds

Homogenize the mass term with `sum(x)=1`. The real polynomial is a degree-two simplex Bernstein combination with diagonal coefficients `Re(m_i^2)` and off-diagonal coefficients `(Re(m_i^2)+Re(m_j^2)-s_ij)/2`. The associated weights `x_i^2` and `2 x_i x_j` are nonnegative and sum to one. Inspection of all 21 exact coefficients gives

```
256 <= Re F <= 324.5,
-0.75 <= Im F <= -0.125.
```

In particular, `Re F` has a strict lower bound of **256**. Set `c=290.25=1161/4`. Then

```
|(F-c)/c| <= sqrt(18778)/1161
           = 0.11803001100659... < q=118031/1000000.
```

The final rational comparison has the positive integer certificate
`118031^2 * 1161^2 - 18778 * 1000000^2 = 314689388081`.
Using `0.11803` would be invalid: it lies below the exact majorant. The component bounds are `|P_mu| <= (3,4,5,5)`.

For inverse power `p`, truncate the convergent binomial series for `F^(-p)` after degree `K` in `z=(F-c)/c`. Its dimensionless tail is bounded by

```
B_p(K) = binomial(p+K,K+1) q^(K+1)
         / (1-q*(p+K+1)/(K+2)).
```

This follows because the ratio of consecutive absolute terms, `q*(p+k)/(k+1)`, decreases with `k` for `p>=1`. The first omitted term has `k=K+1`. Every quantity in this bound can be evaluated as an exact rational; [`derivation.json`](derivation.json) records the complete fractions and decimal displays.

## Gaussian rule and analytic quadrature error

Use stick variables `0<=t_j<=1` for `0<=j<5`:

```
x_j = t_j product_(h<j)(1-t_h),
x_5 = product_(h<5)(1-t_h),
dx  = product_(j=0)^4 (1-t_j)^(4-j) dt.
```

The product rule has `n` Gauss-Jacobi nodes in each dimension, with `alpha=4-j`, `beta=0` on `[-1,1]`. Convert `u` to `t=(1+u)/2` and divide each weight by `2^(alpha+1)`. This accounts for the complete Jacobian; a second simplex-volume or Jacobian multiplier must not be applied.

Each `P_mu` has degree at most one in each stick variable; `F` has degree at most two. Therefore the numerator times a degree-`K` Taylor polynomial has degree at most `rank+2K` in each variable. For `rank<=6` and `K=n-4`, this is at most `2n-2`, within Gaussian exactness through `2n-1`. Exact Gaussian nodes and weights thus integrate the entire truncated polynomial exactly.

For one term `A P^a/F^p`, define `M_a=product_mu(Pmax_mu^a_mu)`. The exact integral and the positive Gaussian rule each contribute a Taylor-remainder bound. Consequently the direct rational-integrand rule obeys

```
|I-Q_n| <= 2*(1/120)*|A|*M_a*c^(-p)*B_p(n-4).
```

Sum this bound for multi-term tensor expressions. The factor **two** is required because the actual evaluator uses `F^(-p)`, rather than the Taylor polynomial. The following are absolute complex-error bounds for exact nodes, weights and arithmetic, rounded upward for display:

| Nodes per dimension | Total nodes | Taylor degree | Index 55 | Index 111 | Index 161 |
|---:|---:|---:|---:|---:|---:|
| 8 | 32,768 | 4 | `6.590e-12` | `5.630e-11` | `4.096e-10` |
| 12 | 248,832 | 8 | `4.884e-15` | `3.607e-14` | `2.188e-13` |
| 16 | 1,048,576 | 12 | `2.385e-18` | `1.621e-17` | `8.571e-17` |
| 18 | 1,889,568 | 14 | `4.823e-20` | `3.183e-19` | `1.591e-18` |
| 20 | 3,200,000 | 16 | `9.359e-22` | `6.028e-21` | `2.865e-20` |

Orders 4, 6, 8 and 10 can expose convention errors and give a bounded convergence preview, but their conservative bounds cannot resolve the disputed components. Orders 18 and 20 are proposed discrimination endpoints. Comparing successive numerical orders alone does not replace these bounds or an arithmetic error certificate.

## Finite precision remains a separate qualification

The source prototype isolates exact rational Jacobi roots and constructs high-precision weights before evaluating a floating-point weighted sum. The exact-arithmetic bounds above do **not** certify finite-precision nodes, weights, parameter construction, complex divisions or accumulation. Those contributions require independent review and an explicit bound. A proposed global-envelope multiplier such as `2^-36` is only a candidate until its operation-count, node enclosure, cancellation and summation proof has been checked against the exact implementation. It must not be presented as a measured or certified uncertainty before that review.

A future admitted run must retain the selected order, all raw values, timings, node/weight metadata, arithmetic-bound status and every failure. It must preserve both TIRust and COLLIER values, make the low-rank convention checks explicit, and compare the defining integral against each backend independently. This proposal does not change COLLIER's cache code or ascribe the current disagreement to the known unrelated cache issue. It does not qualify physical amplitudes, complex-mass stress coverage, stability rescues, poles-only arithmetic or performance.

Source-only proposal inspection and algebra tests:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-proposal-1/quadrature.py proposal
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/rank6-feynman-parameter-proposal-1 -p 'test_*.py'
```

The future physics integration remains gated. A first bounded convergence preview uses seven selected channels at orders 4/6/8/10. Its proposed command requires the explicit admission flag and a new output file:

```sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-proposal-1/quadrature.py run \
  --orders 4 6 8 10 --output /absolute/path/to/a/new/float-result.json --admit-physics
```

An optional small-order arithmetic cross-check uses 80-digit Decimal nodes, weights, parameter construction and accumulation:

```sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-proposal-1/quadrature.py run \
  --engine decimal --orders 8 --output /absolute/path/to/a/new/decimal-result.json --admit-physics
```

The Decimal path checks arithmetic against the binary64 path at the same mathematical order; it does not turn an observed order difference into a certified quadrature error. Orders 18/20 remain optional analytic endpoints, rather than mandatory work before the first discriminator. Each command has its own explicit 180-second cap; this proposal does not authorize either execution.

No such physics integration was executed for this source-only packet.
