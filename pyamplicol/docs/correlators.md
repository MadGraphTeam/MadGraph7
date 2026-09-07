---
title: "Born Correlations"
parent: "Python API"
nav_order: 4
---
<!-- SPDX-License-Identifier: 0BSD -->
# Tree-level spin and colour correlations

Correlations are an explicit generation-time option. Declare the colour
operators and allowed joint spin replacements once, then select an operator
by ID and supply numerical spin vectors at runtime. This provides
four-dimensional tree-level correlated Born quantities, including ordered
colour connections through three unresolved emissions (N3LO colour
structures). It is **not** a complete N3LO subtraction implementation: no
loop amplitudes, kinematic splitting kernels, integrated counterterms, or
extra-dimensional spin components are supplied.

## Generate and evaluate

This small example uses the built-in Standard Model and a new output directory:

```python
from pyamplicol import (
    ColorCorrelator, CorrelatorConfig, Generator, ModelSource, Runtime,
)

declarations = CorrelatorConfig(
    color_correlations=(ColorCorrelator.dipole("T13", 1, 3),),
    spin_correlations=((3,), (3, 4)),
)
result = Generator().generate(
    "g g > g g",
    "artifacts/gg_correlated",
    model=ModelSource.built_in_sm(),
    correlators=declarations,
)
runtime = Runtime.load(result.output)
point = (
    (500, 0, 0, 500),
    (500, 0, 0, -500),
    (500, 300, 0, 400),
    (500, -300, 0, -400),
)

# "born" is always included: the ordinary full-colour overlap.
born = runtime.evaluate_correlated((point,), precision=40)[0]
colour = runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]

# Replace leg 3's polarization by a literal transverse four-vector.
runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0)})
spin_colour = runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
print(born.real, colour.real, spin_colour.real, spin_colour.imag)
runtime.set_spin_correlation_vectors(None)
```

The result is a tuple of `CorrelatedValue` objects, one per phase-space point.
Each has `real` and `imag` fields of type `Decimal`; `precision` specifies
decimal digits and defaults to 16. `complex(value)` is convenient but converts
to ordinary double precision. An ordered colour correlation can be complex:
do not discard its imaginary part before forming the intended physical
combination. All calls use the runtime's current model parameters and
normalization settings.

## Spin vectors and joint classes

Leg labels are one-based, in the generated public process order, and components
are contravariant `(v0, vx, vy, vz)` with metric `(+,-,-,-)`. Values may be real
or complex. A four-vector broadcasts to every point; a sequence of four-vectors
provides one vector per point:

```python
runtime.set_spin_correlation_vectors({3: ((0, 0, 1, 0), (0, 0, 2j, 0))})
batch = runtime.evaluate_correlated((point, point), precision=40)

runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0), 4: (0, 0, 1, 0)})
joint = runtime.evaluate_correlated((point,), precision=40)[0]
runtime.set_spin_correlation_vectors({})  # Reset; None is equivalent.
```

The nonempty set of supplied legs must match a declared class **exactly**.
Here `{3}` and `{3, 4}` are allowed, but `{4}` is not. Every declared leg must
be a spin-1 source with four components. Batch lengths must match the number
of evaluated points. State belongs to each `Runtime` instance; setters copy
their inputs, and a failed setter leaves the previous vectors intact.

Vectors replace the external source literally, including its temporal
component. They are not normalized, projected onto helicity states, or
conjugated on input; the existing source crossing phase is applied once.
Use physical incoming/outgoing momenta in their normal public ordering, with
no manual crossing of the supplied vectors. The same vector enters the
amplitude and is conjugated through the bra, giving a rank-one spin
contraction. Joint replacements contract the same amplitude with several
vectors; they are not products of separate one-leg results. Independent
bra/ket spin vectors and full spin-density tensors are not exposed.

Each replaced leg counts once; spectator helicities are summed incoherently.
The original incoming-spin and colour averages and identical-particle factors
are retained, without an extra average over supplied vectors. Vector magnitude
is preserved: scaling one vector by `z` scales the correlation by `abs(z)**2`.
For a physical massless polarization choose `p.v = 0`; no such condition is
enforced by the setter. Setting, for example, `{3: point[2]}` supplies a literal
nonzero momentum vector for a Ward-identity check. Its vanishing is a
massless gauge-boson statement, not a generic test for massive vectors.

## Ordered colour connections

`ColorCorrelator.dipole("T13", 1, 3)` prepares
`<M | T_1 . T_3 | M>`. More general requests supply ordered `bra` and `ket`
operations, with zero to three steps on each side:

```python
from pyamplicol import EmitGluon, SplitGluon

cascade = (
    EmitGluon(1, -1),       # Emitter, new gluon label.
    SplitGluon(-1, -2, -3), # Parent gluon, new quark, new antiquark.
    EmitGluon(-2, -4),      # A later step may act on an emitted leg.
)
request = ColorCorrelator("cascade", bra=cascade, ket=cascade)
# Include request in color_correlations before generating the artifact.
```

Positive labels identify Born legs; fresh negative labels identify auxiliary
particles. Both sides must have the same number of steps and finish with the
same labelled colour representations. These are colour operations only:
**do not add unresolved momenta** to the Born phase-space input. Prepare only
the IDs you need; IDs must be unique and `"born"` is reserved.

The convention is explicitly **bra-adjoint times ket**. For connected basis
tensors, `K[c,d] = <C_bra,c | C_ket,d>` and the runtime computes
`normalization * sum_h,c,d conjugate(A[h,c]) * K[c,d] * A[h,d]`.
The full directed matrix is used, without assuming Hermiticity. Physical
SU(3) generators satisfy `tr(t^a t^b) = delta(a,b)/2`; colour roles use the
all-outgoing convention, with incoming fundamental representations crossed
by generation. No additional crossing sign should be supplied by the caller.
`SplitGluon` represents one labelled quark flavour with physical `T_R = 1/2`;
no flavour sum (`n_f`), extra `g_s` factors, unresolved phase-space factors, or
symmetry multiplicities are included automatically.
Build higher connections from ordered operations, not by multiplying Born
overlap matrices: those matrices are not action matrices in an orthonormal
closed basis.

## JSON and command line

The equivalent declaration file `correlators.json` is:

```json
{
  "color_correlations": [
    {
      "id": "T13",
      "bra": [{"kind": "emit-gluon", "emitter": 1, "emitted": -1}],
      "ket": [{"kind": "emit-gluon", "emitter": 3, "emitted": -1}]
    }
  ],
  "spin_correlations": [[3], [3, 4]]
}
```

```console
pyamplicol generate 'g g > g g' artifacts/gg_correlated_cli \
  --model built-in-sm --correlators correlators.json
```

`CorrelatorConfig.to_json_dict()` and `from_json_dict()` use this same schema.
A splitting step uses `{"kind": "split-gluon", "parent": -1,
"quark": -2, "antiquark": -3}`. The CLI option prepares the artifact;
numerical vector setting and correlated evaluation currently use Python.

## Execution scope and references

Correlated generation selects complete full-colour, direct, generic compiled
amplitudes and disables helicity-specific zero/parity reductions and current
reuse that would invalidate arbitrary source vectors. Correlated evaluation
uses a separate Python exact executor over the retained Symbolica evaluator
states. RustiCol remains the ordinary compiled executor; the native
C/C++/Fortran/Rust interfaces are not correlation-aware yet. This reference
path prioritizes correctness, not native warmed-loop throughput.

FFT, recurrence, on-the-fly execution, replay-based reductions, nonidentity
process aliases/permutations, partial helicity/colour generation, and append
mode are not supported by this correlation path. Existing artifacts without
the explicit declaration cannot be upgraded merely by setting vectors.
Ordinary generation and `Runtime.evaluate(...)` are unchanged; setting spin
vectors affects only `evaluate_correlated(...)`.

The [original MadNkLO note](https://github.com/mg5amcnlo/pyamplicol/blob/main/IMPLEMENTATION_DOCS/REFERENCES/ColorCorrelators_MadNkLO.pdf)
and [MadNkLO source at revision 646a3db](https://github.com/madnklo/madnklo/tree/646a3db9c8efd7b4cb00e9d89b9197cd5394c01b)
provide background. The note's oriented, non-conjugated connections are not
the runtime's literal bra-adjoint/ket convention: translate adjoints,
orientations, signs, and phases before comparing individual ordered entries.
