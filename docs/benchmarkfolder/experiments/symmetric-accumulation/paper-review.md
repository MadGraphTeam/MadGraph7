# Symmetric tensor accumulation: paper-to-code research

This is an implementation research note, not a measured speedup claim. The
user requested a deeper investigation of symmetric accumulation while the
NinjaRS integration and its current optimization ports continue.

Reference: V. Hirschi and T. Peraro, *Tensor integrand reduction via Laurent
expansion*, [arXiv:1604.01363](https://arxiv.org/pdf/1604.01363), especially
sections 4.2, 5 and 6.

The paper builds symmetric tensor products recursively, sharing lower-rank
intermediates between Laurent terms. Section 6 describes the implementation in
Ninja. This avoids explicit permutation enumeration and redundant construction
of projectors. It should be distinguished from generating the process-dependent
matrix-valued numerator coefficients in MadLoop.

## Existing implementation

- `madgraph/various/q_polynomial.py` already represents independent monomials,
  with `binomial(r+4,4)` coefficients through rank `r`. Its coefficient product
  map is generated ahead of execution. The runtime does not enumerate tensor
  permutations or compute factorials.
- `write_compact_wl_updater` multiplies matrix-valued polynomials. For each
  nonzero vertex coefficient it scatters products through `COMB_COEF_POS` into
  the loop wavefunction output. Each fixed output receives contributions in
  vertex-coefficient then contracted-component order.
- `write_expanded_wl_updater` emits explicit coefficient expressions, but the
  exporter uses it only for selected ranks. Larger updates use the compact
  scatter kernel.
- Pinned NinjaRS `crates/ninjars/src/fast/tensor.rs` already implements a
  transposed Horner sweep with truncated jets for the reduction-side expansions.
  Replacing that blindly with materialized tensor powers is not an established
  optimization.

These observations follow inspection of the current sources and are not claims
that their implementation is optimal.

## Fortran experiment

For a vertex linear in loop momentum, write its polynomial as
`B(q) = B_0 + sum_mu B_mu q_mu`. For each exponent tuple `alpha`, the updated
coefficient is

```text
C[j, alpha, i] =
    sum_k sum_{beta in {0,e0,e1,e2,e3}, beta <= alpha}
        A[k, alpha-beta, i] * B[j, beta, k].
```

Use generation-time predecessor lists in the existing monomial ordering.
Accumulate one output coefficient in a local value and write it once. Initially
keep the same `beta,k` order and complex multiplication operand order as the
corresponding existing routine. Missing predecessors are absent contributions,
not multiplications by a manufactured zero.

Compare at least these variants against the current updater:

1. Direct predecessor indexing, with current arithmetic order and zero rules.
2. Selective straight-line expansion for the hot rank-2/3/4-by-rank-1 kernels.
3. Proven vertex sparsity specialization, if the ALOHA expression graph provides
   a stable structural pattern. Numerical zeros alone do not prove a pattern.

The first two variants principally target indexing, repeated output access and
compiler scheduling; the mathematical product count need not decrease. The
third may reduce arithmetic, but needs a separate proof of its active entries.
Shared subexpressions across vertex components are another candidate only where
the generated expressions establish their equality.

## Validation and retention

Keep experiments in copied generated processes and the existing Fortran code
path. Retain the current public fallback for unsupported layouts or arithmetic
conditions. Compare DP and QP, all required output components, input immutability,
exceptional operands and floating-point behavior. A reordered sum requires
independent numerical/stability validation and must be labelled as such.

Measure kernel cost and full warmed process evaluations separately. Include
QCD3 and a process with no QP rescue, then broader ranks/processes if promising.
Only a reproducible benefit with complete numerical validation warrants another
production change. The previously measured terminal-QP gain remains distinct:
it avoids unconsumed off-diagonal components and does not establish a faster
general symmetric accumulation algorithm.
