---
title: "Correlation Conventions"
parent: "Python API"
nav_order: 5
---
<!-- SPDX-License-Identifier: 0BSD -->
# Conventions for spin- and colour-correlated Born quantities

This page connects the notation of the
[MadNkLO colour-correlator note](https://github.com/mg5amcnlo/pyamplicol/blob/main/IMPLEMENTATION_DOCS/REFERENCES/ColorCorrelators_MadNkLO.pdf)
to pyAmpliCol's [correlated API](correlators.md). In particular, it makes the
bra/ket convention explicit: pyAmpliCol contracts the **adjoint of the bra
connection with the ket connection**, rather than adopting the note's
oriented, non-conjugated product unchanged.

The implementation currently gives full-colour SU(3) tree-level quantities.
The order of a colour connection (NLO, NNLO, N3LO) is independent of the
colour approximation (LC, NLC, full); correlated LC/NLC approximations are
not yet implemented.

## Amplitudes, basis tensors, and correlated matrices

| Symbol | Meaning |
| --- | --- |
| `N_c = 3` | Number of colours, not the size of the colour basis |
| `c, d = 1, ..., N_basis` | Labels of the retained colour-basis tensors |
| `h` | External helicity configuration, or spectator helicities after spin replacement |
| `C_c` | A colour tensor, including its external colour indices |
| `A[h,c]` | The coherent, complex partial amplitude in the generated convention |
| `S_B`, `S_K` | Ordered colour connections applied on the bra and ket side |
| `N` | The ordinary overall normalization, including any factored common coupling, averaging, and identical-particle factors |

With momenta and model parameters left implicit, the definitions are:

```text
|M_h>       = sum_c A[h,c] |C_c>
K[B,K][c,d] = < S_B C_c | S_K C_d >
R[B,K]     = N sum_(h,c,d) conj(A[h,c]) K[B,K][c,d] A[h,d]
```

The matrix includes the process's colour normalization factors. Its entries
are evaluated exactly as complex rational numbers at generation. At runtime,
`R[B,K]` is returned as a `CorrelatedValue` with decimal real and imaginary
parts. The ID `"born"` uses empty connections, giving the usual colour metric
`<C_c|C_d>`.

Only spectator helicities are summed incoherently. Colour components are
combined **before squaring**, retaining their interference. Separate squared
flow/helicity components are therefore not sufficient to reconstruct a
general colour or spin correlation: their relative phases have been lost.

For two different connections, neither an individual colour matrix nor its
answer is assumed real or symmetric. Interchanging the bra and ket gives
the Hermitian-conjugate matrix and the complex-conjugate answer:

```text
K[K,B] = K[B,K]†
R[K,B] = conj(R[B,K])
```

For identical connections, the overlap is a squared norm. This distinction
also explains why an ordered connection should not be interpreted as a
positive squared matrix element in every case.

## Colour charges, indices, and crossing

We use Hermitian physical generators with
`tr(t^a t^b) = T_R delta(a,b)`, `T_R = 1/2`, and
`[t^a,t^b] = i f^(abc) t^c`. In **output, input** index order the charges are:

| All-outgoing representation | Charge matrix for emitted gluon colour `a` |
| --- | --- |
| Quark, `3` | `t^a[out,in]` |
| Antiquark, `3bar` | `-t^a[in,out]` |
| Gluon, `8` | `-i f^(a,out,in)` |

These conventions give `C_F = (N_c^2-1)/(2 N_c) = 4/3` and `C_A = N_c = 3`.
Internally, some colour-ordered tensors use generators `tau = sqrt(2) t`;
the correlated matrices include that conversion. Do not supply an extra
factor of two or `sqrt(2)` at runtime.

All colour charges use the all-outgoing convention: an incoming quark is an
outgoing antiquark for the colour algebra, and vice versa. Leg **labels do not
change**: they remain one-based in the generated public process order,
including colourless particles. The caller supplies normal physical incoming
momenta and must not add another colour-crossing sign.

At NLO,

```text
B_ij = N sum_h <M_h | T_i . T_j | M_h>
     = N sum_(h,a) <T_i^a M_h | T_j^a M_h>
```

is requested as `ColorCorrelator.dipole("Tij", i, j)`. This is the quantity
multiplying soft eikonal factors in a factorized soft limit. pyAmpliCol does
not include the eikonal factor, its soft-limit sign, or the associated
`4 pi alpha_s` factor in `B_ij`. Diagonal entries are retained: their eikonal
coefficients vanish for massless legs but need not vanish for massive legs.

## Ordered connections and auxiliary labels

A connection is a chronological tuple of operations. If its entries are
`(E1, E2, ..., Em)`, its action on a ket is `S = Em ... E2 E1`:
the **first tuple entry acts first**. The bra is adjointed afterwards.
Operations on the same colour line generally do not commute.

`EmitGluon(i, x)` keeps emitter `i` and adds an adjoint leg `x`.
`SplitGluon(x, q, qbar)` removes an adjoint parent `x` and replaces it with
a fundamental leg `q` and an antifundamental leg `qbar`, through
`t^a[q,qbar]`. New labels must be fresh, including with respect to retired
parents. Born labels are positive and one-based; we recommend reserving fresh
negative labels for auxiliary partons, although the operation classes do not
enforce that sign convention. Auxiliary partons have **no additional Born
phase-space momenta**. A later operation may act on an auxiliary leg
only after that leg has been introduced.

The note's four connection classes have the following direct API counterparts:

| Class in the note | Chronological pyAmpliCol operations |
| --- | --- |
| A: emissions from different Born lines | `(EmitGluon(i, -1), EmitGluon(j, -2))`, `i != j` |
| B: successive emissions from one Born line | `(EmitGluon(i, -1), EmitGluon(i, -2))` |
| C: an emitted gluon emits another gluon | `(EmitGluon(i, -1), EmitGluon(-1, -2))` |
| D: an emitted gluon splits into a quark pair | `(EmitGluon(i, -1), SplitGluon(-1, -2, -3))` |

The algebra also permits splitting an active Born gluon. The note's
soft-connection class D instead starts with an already-emitted gluon which
can itself become soft; algebraic support for a Born-leg splitting does not
imply a soft singularity of that hard leg.

Thus the note's emission triple `(i, x, i)` becomes `EmitGluon(i, x)`.
For a quark-pair splitting, the API explicitly names the two daughter
representations and uses fresh daughter labels; it does not reuse the
parent label as one daughter. When translating a whole pair of connections,
relabel their final partons consistently on both sides.

The note canonically sorts some commuting operations and assigns integer
indices to its connection list. pyAmpliCol instead takes the ordered lists
as supplied and uses user-chosen **string IDs for the requested pair**.
It does not sort operations, enumerate a minimal connection basis, or insert
multiplicities for equivalent lists. In particular, do not apply the note's
descending-label sort to a chain whose next emitter has not yet been created.

Both sides must have the same order, up to three operations per side, and
finish with identical labelled colour representations. A gluon-only final
colour space cannot be contracted with a quark-pair colour space. Invalid
pairs are rejected rather than returned as physical zeroes. One-, two-, and
three-step connections supply NLO, NNLO, and N3LO colour structures; this is
not a claim to supply the kinematic ingredients of an N3LO calculation.

## Translating the note's non-conjugated convention

The note's sections 2.1-2.2 keep orientations in the connection labels and do
not conjugate the second connection. Our definition is instead
`<S_B M|S_K M>`. Although the NLO dipoles agree once index directions are
matched, beyond NLO one must translate the adjoint, the order of charges,
fundamental/antifundamental orientations, and final-parton labels together.
Copying two legacy connection IDs is not a defined conversion.

The quark-pair example is particularly useful. Let

```python
from pyamplicol import ColorCorrelator, EmitGluon, SplitGluon

pair = (EmitGluon(1, -1), SplitGluon(-1, -2, -3))
request = ColorCorrelator("pair-self", bra=pair, ket=pair)
```

The API's self-overlap is nonzero in general. Conjugation pairs a fundamental
index with its antifundamental partner and gives

```text
sum_(q,qbar) conj(t^a[q,qbar]) t^b[q,qbar] = T_R delta(a,b)
R[pair,pair] = T_R B_11
```

For a quark emitter this is `(1/2)(4/3) B = (2/3) B`; for a gluon emitter
it is `(1/2)(3) B = (3/2) B`, where `B` is the ordinary Born result.
This is not the zero of the note's oriented `(D.1,D.1)` pairing: that pairing
does not represent the same adjointed contraction. With the generator
normalization above, the closed pair trace is `T_R delta(a,b)`, not
`C_F delta(a,b)`.

`SplitGluon` represents **one labelled quark flavour**. Neither a flavour sum
`n_f`, additional powers of the strong coupling, unresolved phase space,
kinematic splitting functions, nor unresolved-particle symmetry factors are
included. These belong to the application using the correlated Born.

## Products and anticommutators with a shared colour line

The note's double-soft example contains an anticommutator
`{A,B} = AB + BA`, with `A = T_2 . T_3` and `B = T_1 . T_2`.
The shared leg 2 makes the order important. In the API's adjointed convention,
one valid construction is:

```python
bra = (EmitGluon(3, -1), EmitGluon(1, -2))
ab = ColorCorrelator(
    "AB", bra=bra,
    ket=(EmitGluon(2, -2), EmitGluon(2, -1)),
)
ba = ColorCorrelator(
    "BA", bra=bra,
    ket=(EmitGluon(2, -1), EmitGluon(2, -2)),
)
# Include ab and ba in CorrelatorConfig.color_correlations before generation.
```

For `AB`, the ket acts as `T_2^a T_2^b`; the bra adjoint supplies
`T_1^b T_3^a`. Charges on different Born legs commute, giving
`T_2^a T_3^a T_1^b T_2^b = AB`, with `a,b` summed. Swapping the
ket operations gives `BA`, without commuting the charges on leg 2.

After generating and loading a runtime containing these IDs:

```python
r_ab = runtime.evaluate_correlated(points, color_correlation="AB")
r_ba = runtime.evaluate_correlated(points, color_correlation="BA")
anticommutator = tuple(
    (x.real + y.real, x.imag + y.imag)
    for x, y in zip(r_ab, r_ba, strict=True)
)
```

There is no factor of `1/2` in this definition of the anticommutator.
When combining high-precision `Decimal` results, use a decimal context with
at least the requested precision. More general connection products should
likewise be built at the tensor level, not by multiplying two Born overlap
matrices: a nonorthogonal colour basis has a nontrivial metric, and a Born
basis need not remain closed under an insertion.

## Colour-coherence checks

For a colour-conserving Born amplitude, the standard checks are

```text
sum_j B_ij = 0                       (sum includes j = i)
B_ii = C_i B                         (C_i = C_F or C_A)
sum_(j != i) B_ij = -C_i B
```

The sum runs over all coloured Born legs, with incoming charges already
crossed. A nontrivial connected tensor also obeys colour conservation, but
now the sum includes **all its current coloured legs**, including emitted
gluons or daughter quarks, but excluding a parent removed by a splitting.
This gives iterative checks at NNLO and N3LO.
One should not expect an unweighted sum over every distinct higher-order
connection to vanish: those entries describe different emission histories,
and sometimes different final colour spaces.

The note also groups histories in which only Born legs emit gluons, its
"abelian-like" subset. If commuting histories are collapsed to one canonical
representative, the missing ordering multiplicity must be restored. With
`m` emissions and `n_i` emissions on Born leg `i`, preserving the order on
each individual line, this multiplicity is `m! / product_i(n_i!)`.
At NNLO it is 2 for two distinct emitters and 1 for two emissions from the
same emitter; at N3LO the corresponding patterns give 6, 3, and 1.
These weights describe such a canonicalized sum; the API does not multiply
individual requested correlations by them. They apply only after identifying
genuinely equivalent labelled contractions: charges on a shared line must
not be commuted, and any relabelling of an auxiliary index must be made
consistently across the whole bra/ket contraction.

For example, interchanging `EmitGluon(i, -1)` with `EmitGluon(j, -2)`
commutes when `i != j` and their labels are kept attached to the same emitters.
Changing the assignment to `EmitGluon(j, -1)` and `EmitGluon(i, -2)` is a
different labelled map. Preserve these assignments, or define a consistent
symmetrization on both sides, before using a weighted higher-order coherence
sum from the note.

## Spin contractions and Ward identities

External spin states and vector components are four-dimensional. The API
does not supply a full `d = 4 - 2 epsilon` spin tensor or the
epsilon-dependent polarization averages of conventional dimensional
regularization.

For a declared external vector leg, the setter replaces its source by the
literal contravariant vector `(v0, vx, vy, vz)`, using metric `(+,-,-,-)`.
There is no normalization or transverse projection. In schematic notation,
if the amplitude is `v^mu M_mu`, the same vector on the ket and its complex
conjugate on the bra form the rank-one spin contraction. Distinct vectors on
several declared legs act jointly, not as a product of separate squared
matrix elements.

No extra average over the supplied vectors is introduced. Original incoming
spin/colour averages are retained, and unreplaced helicities are summed.
The usual two-polarization completeness sum recovers the unpolarized
massless-vector result. Scaling a single vector by `z` scales the answer by
`abs(z)**2`.

For a massless gauge boson, replacing its polarization by its momentum tests
the Ward identity with the other gauge legs in physical transverse states;
several momentum replacements are also allowed if their joint class was
declared. The vanishing is not guaranteed if another gauge leg is instead
given an arbitrary non-transverse source. This is a literal nonzero momentum
source, not its
projection onto physical helicities. A massive vector instead satisfies the
appropriate Goldstone relation; it is not subject to the same zero test.
The setter currently exposes rank-one contractions, not independent bra/ket
vectors or a general spin-density tensor.

See [Born Correlations](correlators.md) for runnable generation/evaluation
examples, vector batching, reset behaviour, and the exact execution scope.
