# Reuse contracted rows before distributing momentum powers

The most promising small prototype keeps the original constant-vertex
accumulation and factors only the four linear momentum coefficients. It
reuses a **sum of products**, so it changes rounding. It is distinct from the
previous exact-product cache and is not a production change or a measured
speedup.

The paper's symmetric recurrences reuse lower-rank tensor intermediates
between projector terms; they do not establish that a process-dependent
Lorentz matrix is low rank. MadLoop already stores independent monomials.
The useful analogous step here is to reuse a contraction before distributing
it to different momentum powers. This additional identity must be proved
from the actual vertex source. [Hirschi and Peraro, sections 4.2–6,
especially equation (4.9)](https://arxiv.org/pdf/1604.01363).

## Exact source result

`analyze.py` parses the current generated Fortran assignments and expands
them as sparse polynomials over exact Gaussian rationals. It checks source
hashes against `symmetric-accumulation/final-inventory.json`, bound to
numerical source `b44d20e725f1ec68589df20284e4ce9116e9e2ab`. There is no
floating-point substitution, amplitude evaluation, compiler, or Symbolica
call in this analysis.

Write `B(q)=B0+sum_mu q_mu B_(mu+1)` and define `H=B1`. Every tested fermion
map satisfies

```text
B_m(j,k) = phase(m,j) * H(row(m,j),k),   m=1..4,
phase ∈ {1,-1,i,-i}.
```

The test proves all 64 coefficient identities for each vertex. All six
inventoried QCD3 FFV maps pass, as do all ten QCD4 FFV/FFS maps, including
the chiral `FFV2_3`/`FFV2_5` and scalar `FFS1` cases. The vector `VVV1L2P0_1`
does not have this signed-permutation property and is excluded.

For `MP_FFV1L2_1`, applying each matrix to a vector `U` gives:

```text
B1: ( U1,  U2,  U3,  U4)
B2: ( U2,  U1, -U4, -U3)
B3: (iU2,-iU1,-iU4, iU3)
B4: ( U1, -U2, -U3,  U4).
```

The retained map is `analysis.json` →
`cases[case].vertex_proofs[vertex].q_rows[m-1][j-1]`, containing a one-based
`source_row` and integer complex `phase=[real,imag]`.
The final parser reads numeric source spelling directly into `Fraction`;
`parser-check.json` tests decimals beyond binary64 resolution and rejects
unsupported unary syntax. The initial proof is retained in `source-proof-v1/`;
all maps and operation counts are unchanged by that parser correction.

For every coefficient `alpha` of the incoming polynomial and cut component
`I`, compute

```text
U(r,alpha,I) = sum_k H(r,k) * A(k,alpha,I).
```

Then scatter `phase(m,j)*U(row(m,j),alpha,I)` to
`OUT(j,alpha+e_(m-1),I)`. The existing monomial product map supplies the
destination; no symmetry or multinomial factor is introduced. Keep all
`B0*A` products and additions in their original order. For a terminal
diagonal, `I=J`; only two rows of `U` are needed for each `J`, and contractions
must not be shared between different `I`.

## Arithmetic potential and actual zeros

For dominant `FFV1L1_2`/`FFV1L2_1` maps, the original derivative block has
32 nonzero complex products per incoming coefficient and cut component.
The full contraction needs eight, followed by signs/component swaps and
additions. Keeping 16 constant-block products gives **48→24**. A terminal
diagonal needs four derivative products instead of eight, giving **12→8**
per diagonal. This includes the new `U` products; it does not count them as
free. Additions, guards, loads, stores and sparse tests still cost time.

| Source inventory | Selected compact products after literal vertex zeros | Proposed products | Static reduction |
| --- | ---: | ---: | ---: |
| QCD3 | 255,220 | 156,740 | 38.59% |
| QCD4 | 3,226,340 | 1,899,780 | 41.12% |

These totals leave unmatched vector/ghost helpers unchanged and include
QCD4 rank-four terminal calls. They are **before runtime A/B zero skips**,
not a runtime estimate.

The existing full-stream QCD3 trace observes 5,503,000 selected products
after both zero skips, of which 5,255,320 belong to the proved maps.
It does not separate constant and derivative products. If the actual
rounded-B relation guard passes, let `C` be surviving constant products and
`D` surviving H products. Full helpers have `M=C+4D`; terminal helpers
have `M=C+2D`. Thus their savings are `3D` and `D` even with arbitrary
zero A entries. Using only `C`'s structural ceiling gives aggregate
conditional bounds of **916,002.5–3,201,792.5** saved products, or
16.65%–58.18% of the recorded selected products. The fractional endpoints
are conservative rational bounds, not executed counts.

The prototype must separately count `B0`/derivative products, surviving
U terms, exact runtime row-relation failures and all-zero U contractions.
It must preserve the A-zero skips and avoid building all four U rows in
terminal helpers. These counters resolve the uncertainty left by the
current trace; no assumption of dense numerator arrays is justified.

## Minimal isolated implementation and risks

Start with the two dominant FFV1 orientations, ranks 2–4, using the existing
updater signature in copied generated outputs. The constant block stays
unchanged; no ALOHA fusion, propagator-mass transport or reconstruction of
`B0` is needed. Generate direct U expressions and phase/scatter maps, with
local scratch scoped to one updater invocation. Do not retain state between
points.

The symbolic identity is over exact arithmetic. Independently generated
binary128 B entries can differ after rounding, so verify the actual row
relations before using them. Use integer component swapping/sign-bit
operations for the phase comparison after finite/layout checks, rather
than new complex multiplications. A conservative bit comparison includes
signed zero; a failed relation selects the original routine.

Use the original routine for unsupported shapes/layouts/rounding,
nonfinite, subnormal or exponent-unsafe inputs and enabled traps. Require
IEEE_INEXACT already sticky: reassociation can change whether inexact is
raised, even when serious exceptions cannot occur. The ±1000 binary128
operand bound leaves ample exponent headroom for the short contractions,
but the generated operation graph and intermediate sums still need an
explicit bound. Do not clear/replay exceptions or silently disable traps.

The principal numerical risk is cancellation in U or between U and the
constant contribution. Summing two/four products before adding them to
OUT is not the former accumulation order; signed zeros and inexact outcomes
can differ. A local absolute-error bound depends on the sum of product
magnitudes, not the small cancelled U. No universal bit identity or
correct-rounding claim follows from the symbolic proof.

Validate direct outputs against an independent wider-precision contraction
over dense, sparse, phase-related, changed-scale and adversarial cancelling
inputs. Include all rank-four maps, input immutability, guard failures,
signed zeros, nonfinite/subnormal/extreme inputs and actual standalone/NLO
flags. Then compare full corrected QCD3/QCD4 streams, every finite/pole/order,
unchanged directions/rotations/stability thresholds and successful rescue
behavior. Full-process agreement is a necessary finite-sample physics gate,
not a proof of algebra or IEEE invariance. Only then assess uninstrumented
process timings and the original-Ninja acceptance gates.

The larger `(slash(p+q)+m)G` factorization is deferred: it would also change
constant-block arithmetic and cancellation. If vector helpers dominate a
remaining profile, their metric/outer-product structure is a separate
source-proof candidate; it is not covered by the FFV row maps above.

Reproduce the source analysis in a new output namespace or remove only an
unsealed local analysis result; the script refuses to overwrite retained
`analysis.json`:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization/analyze.py
```

`analyze.py` and `analysis.json` are frozen for the independent prototype
owner. They read generated sources and the earlier operation-count record;
they do not alter either source tree or any earlier experiment.
