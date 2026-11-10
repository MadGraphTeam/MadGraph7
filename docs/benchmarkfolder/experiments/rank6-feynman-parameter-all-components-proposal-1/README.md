# Source proposal: independent integration of all 210 tensor components

This private diagnostic extends the independently derived parameter-integral
check to all 210 Cartesian components through rank six of the same synthetic
N=6 fixture 108. **The actual fixture has not been integrated by this new
implementation.** No native reducer, model initialization, production Fortran
change or reference-repository change is part of the proposal.

The immutable scalar implementation and Gaussian rule remain pinned to
`03ffcd9f8b88855b59d78387cde255460a36d35cca251fce093741a9c0065db3`.
The earlier seven-component
[qualification](../rank6-feynman-parameter-qualification-1/README.md), index
`c2fe806d2996e7ee758e2a7894318d04388b640126b9f636fd23be99650b90d7`,
remains unchanged. This proposal preserves every existing answer and failure.

## Reusable primitive moments

Each Cartesian tensor is a sparse combination of primitive integrals
`integral_simplex P^a/F^p`. There are 296 unique primitives:

| Inverse denominator power | Maximum numerator rank | Primitive count |
|---:|---:|---:|
| 4 | 6 | 210 |
| 3 | 4 | 70 |
| 2 | 2 | 15 |
| 1 | 0 | 1 |

The historical MadLoop coefficient order is joined to its frozen monomial
list. It gives rank prefixes of lengths 210/70/15/1. A recurrence builds each
numerator monomial once per point chunk from a previously built lower-rank
monomial. All four denominator groups reuse that table. The resulting primitive
vectors require **660 exact dyadic coefficients** to reconstruct all 210
components, with denominators at most 8. The existing component convention,
odd-rank sign and Minkowski trace factors apply once. For example,

```
q0^6: 6 P0^6/F^4 - 15 P0^4/F^3 + (45/4) P0^2/F^2 - (15/8)/F.
```

Only Gaussian point arithmetic is vectorized. No TIRust, oneloopmaster,
COLLIER or other reducer is used to form these independent values.

## Bounded arrays and precision

The first Gaussian coordinate is processed one slice at a time. Each inner
four-dimensional slice is further split into at most 4096 points by default,
with an admitted upper limit 8192. The Cartesian table has shape `[210,chunk]`;
at chunk 4096 it occupies 13,762,560 bytes on this host, and at 8192 it occupies
27,525,120 bytes. Full-grid monomial arrays are never materialized.

The implementation uses `NumPy.longdouble` throughout coordinates, polynomial
construction, primitive reduction and reconstruction. On the pinned host this
has a **64-bit significand**, despite occupying 16 storage bytes. It is extended
precision and **is not 113-bit Fortran QP**. The Gaussian nodes and weights are
parsed directly from their preserved 80-digit decimal strings. No intermediate
passes through `math.fsum` or binary64 transport. Contiguous NumPy reductions
sum each point chunk, and Neumaier vector compensation combines the chunks.
Every BLAS/OpenMP thread variable is set to 1 before importing NumPy.

The analytic Gaussian error bound is inherited separately for each of the 210
components. It remains a bound on exact-node/weight/arithmetic quadrature;
finite-precision arithmetic is assessed separately. Neither longdouble nor
agreement with an existing reducer alone is a formal global error certificate.

## Source-only validation and cost

The synthetic tests exercise all 296 primitives and all 210 components using
constant nonzero shifts and equal complex masses; zero-shift identities test
odd zeros and every even-rank metric trace. A nonconstant null-shift fixture
keeps `F` constant and admits exact Dirichlet moments:

```
p_i = lambda_i*(1,1,0,0), lambda_i=0..5,
integral_simplex L^r = r!/(r+5)! * h_r(lambda).
```

This independently checks Gaussian normalization, parameter indexing,
nonconstant numerators and partial point chunks. Spatial permutations and
momentum reflection test component mapping and rank signs. Chunk sizes 1,37
and 4096 are exercised. A mocked controller test verifies that completed raw
primitives/tensors survive a later comparison-loader failure.

The saved seven-component transport test injects previously recorded values
into the comparison function; **it is not a new integration agreement result**.
After root admission, actual orders 8 and 18 will compare their seven overlapping
components with those saved scalar implementation values. Order 16 has no
preexisting scalar result and explicitly reports that absence.

The independently retained toy cost study uses synthetic arrays only. It
supports a bounded attempt, not a physics or performance result. The intended
orders are 8/16/18 under one 180-second aggregate limit, nonreserved CPU 10 and
an external 4 GiB address-space cap. The worker checks a cooperative deadline;
the root supervisor must also apply its hard timeout and resource limits.
There is no CPU reservation or native reducer call.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-all-components-proposal-1/all_components.py proposal
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/rank6-feynman-parameter-all-components-proposal-1 \
  -p 'test_*.py' -v
```

Only after source review and admission, the externally supervised worker would
be invoked as follows:

```sh
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-all-components-proposal-1/all_components.py run \
  --orders 8 16 18 --chunk-size 4096 --seconds 180 \
  --output /absolute/path/to/a/new/all-components-result.json --admit-physics
```

Every completed order retains its 296 primitive values, 210 tensor values,
Gaussian rules, analytic error bounds and selected-component comparisons. Raw
results are written before the comparison stage. Failure and timeout records
retain all earlier completed orders. No fallback, result replacement or
relaxation of the existing numerical criterion is introduced.
