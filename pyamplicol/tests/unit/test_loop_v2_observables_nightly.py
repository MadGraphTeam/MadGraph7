# SPDX-License-Identifier: 0BSD
"""Long v2 observable controls (WP2.5; nightly).

* O1: the helicity-resolved outputs of u u~ > g g g at 200 bits against the
  frozen legacy record (``test_loop_v2_observables.py`` runs the others);
* the helicity x Born-basis entries of u u~ > g g g in binary64 sum to the
  inclusive result, and the native resolved columns equal the Python ones;
* O14 with resolved-colour seed columns: BDK's partial amplitudes of
  qbar q g g g reproduce v2's helicity-resolved bare virtual at 200 bits.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from .loop_v2_bdk import BDK_PREFACTOR, bdk_bracket, bdk_structures, laurent_times
from .test_loop_v2_observables import (
    CARRIERS,
    FIXTURES,
    PARAMETERS,
    assert_helicity_resolved_legacy,
    assert_resolved_entries_sum,
    observable_builder,
)

pytestmark = pytest.mark.loop_nightly


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    return observable_builder(loop_sm_compiled_model, loop_native)


def test_five_leg_helicity_resolved_outputs_equal_legacy_at_200_bits(generate):
    """O1 of u u~ > g g g's helicity-resolved outputs at 200 bits (about 8
    minutes in the Python reference executor)."""

    assert_helicity_resolved_legacy(generate, "uux_ggg", "arbitrary-200")


def test_five_leg_resolved_entries_sum_to_the_inclusive_result(generate):
    """Helicity x Born-basis entries of u u~ > g g g in binary64 (about 5.5
    minutes, mostly the Python executor's one-hot helicity x colour
    columns)."""

    assert_resolved_entries_sum(generate, "uux_ggg", "double")


def _rank(vectors: list[list[Fraction]]) -> int:
    """The rank of rational vectors (exact elimination)."""

    rows: list[tuple[int, list[Fraction]]] = []
    for vector in vectors:
        vector = list(vector)
        for lead, row in rows:
            if vector[lead]:
                factor = vector[lead] / row[lead]
                vector = [a - factor * b for a, b in zip(vector, row, strict=True)]
        lead = next((k for k, value in enumerate(vector) if value), None)
        if lead is not None:
            rows.append((lead, vector))
    return len(rows)


def _inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """The exact inverse of a nonsingular rational matrix (Gauss-Jordan)."""

    size = len(matrix)
    rows = [
        [*row, *(Fraction(int(i == j)) for j in range(size))]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        lead = rows[column][column]
        rows[column] = [value / lead for value in rows[column]]
        for r in range(size):
            if r != column and rows[r][column]:
                factor = rows[r][column]
                rows[r] = [
                    a - factor * b for a, b in zip(rows[r], rows[column], strict=True)
                ]
    return [row[size:] for row in rows]


def _generalised_inverse(gram: list[list[Fraction]]) -> list[list[Fraction]]:
    """An exact generalised inverse M (G M G = G) of a symmetric positive
    semidefinite rational matrix: the inverse of its principal submatrix on a
    maximal independent column set, zero elsewhere."""

    size = len(gram)
    columns = [[gram[row][column] for row in range(size)] for column in range(size)]
    chosen: list[int] = []
    for column in range(size):
        if _rank([columns[c] for c in (*chosen, column)]) > len(chosen):
            chosen.append(column)
    inverse = _inverse([[gram[a][b] for b in chosen] for a in chosen])
    out = [[Fraction(0)] * size for _ in range(size)]
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            out[a][b] = inverse[i][j]
    return out


def test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14(generate):
    """O14 with resolved-colour seed columns (design §7.6, §14), at 200 bits:
    the projections of v2's bare loop amplitude of qbar q g g g onto BDK's
    one-loop colour structures (``ObservableSlice.project_loop``) determine
    partial amplitudes A_{5;1}, A_{5;3}, A_{5;4} through an exact generalised
    inverse of the N_c = 3 Gram matrix (which has null directions; the
    projections lie in its range to the carrier's rounding). With the tree
    partial amplitudes of v2's Born, BDK's colour sum (N^2-1)/N sum_s
    A_tree(s)^* bracket(s) (hep-ph/9409393 eq. NLOCrossSection) reproduces
    v2's helicity-resolved bare virtual. Gram, tree and BDK matrices are
    exact; every product runs at the carrier's working digits."""

    from decimal import Decimal, localcontext

    from pyamplicol.color.connections import (
        contract_connected_tensor_nc_terms,
        evaluate_color_nc_terms,
    )
    from pyamplicol.loops.v2.assemble import _single

    observables = generate(
        "u~ u > g g g", born={"QCD": 3, "QED": 0}, loop={"QCD": 5, "QED": 0}
    )
    slice_ = observables.slice
    tree, loop = bdk_structures()
    assert next(iter(tree.values())).legs == slice_.basis.legs
    fixture = json.loads((FIXTURES / "legacy/uux_ggg.json").read_text())
    point = list(fixture["points"][0])
    point[0], point[1] = point[1], point[0]
    scale = fixture["scales_squared"][0]
    precision = CARRIERS["arbitrary-200"]

    def at_three(cell) -> Fraction:
        value = evaluate_color_nc_terms(cell)
        assert not value.imag
        return value.real

    trees, loops = list(tree), list(loop)
    gram_tree = [
        [at_three(contract_connected_tensor_nc_terms(tree[a], tree[b])) for b in trees]
        for a in trees
    ]
    gram_loop = [
        [at_three(contract_connected_tensor_nc_terms(loop[a], loop[b])) for b in loops]
        for a in loops
    ]
    bdk = [
        [
            sum(
                (
                    v * Fraction(3) ** p
                    for p, v in laurent_times(
                        bdk_bracket(s).get(key, {}), BDK_PREFACTOR
                    ).items()
                ),
                Fraction(0),
            )
            for key in loops
        ]
        for s in trees
    ]
    tree_inverse = _inverse(gram_tree)
    loop_inverse = _generalised_inverse(gram_loop)
    assert _rank(gram_loop) < len(loops)
    evaluation = observables.point(point, scale, PARAMETERS, precision)
    with localcontext(prec=precision.working_digits):

        def number(value):
            if isinstance(value, Fraction):
                return Decimal(value.numerator) / value.denominator
            return value

        def apply(matrix, vector):
            """A rational matrix times a vector of (real, imaginary) Decimals."""

            return [
                (
                    sum(
                        (number(m) * x for m, (x, _y) in zip(row, vector, strict=True)),
                        Decimal(0),
                    ),
                    sum(
                        (number(m) * y for m, (_x, y) in zip(row, vector, strict=True)),
                        Decimal(0),
                    ),
                )
                for row in matrix
            ]

        def size(vector):
            return max(abs(x) + abs(y) for x, y in vector)

        def tree_amplitudes(row):
            """Tree partial amplitudes: Born = sum_s a_s tree_s."""

            projected = []
            for s in trees:
                re = im = Decimal(0)
                for vector in evaluation.born_vectors[row].values():
                    for (tensor, power), value in vector.items():
                        overlap = number(
                            at_three(
                                contract_connected_tensor_nc_terms(
                                    tree[s], _single(tensor, slice_.basis.legs)
                                )
                            )
                            * Fraction(3) ** power
                        )
                        re += overlap * value.real
                        im += overlap * value.imag
                projected.append((re, im))
            return apply(tree_inverse, projected)

        amplitudes = {row: tree_amplitudes(row) for row in observables.rows}
        largest = max(size(a) for a in amplitudes.values())
        rows = [row for row, a in amplitudes.items() if size(a) > largest / 1000][:2]
    assert len(rows) == 2
    projections = observables.project_loop(
        point,
        scale,
        [loop[key] for key in loops],
        rows=rows,
        parameters=PARAMETERS,
        precision=precision,
    )
    resolved = observables.evaluate_resolved(
        point, scale, resolution="helicity", parameters=PARAMETERS, precision=precision
    )
    (order,) = {label[2] for label in projections}
    with localcontext(prec=precision.working_digits):
        for row in rows:
            (bare,) = resolved.entries[row, None].components["bare"].values()
            expected = []
            for k in range(3):
                target = [
                    (
                        Decimal(projections[row, j, order].coefficients[k].real),
                        Decimal(projections[row, j, order].coefficients[k].imag),
                    )
                    for j in range(len(loops))
                ]
                partial = apply(loop_inverse, target)
                residual = [
                    (a - c, b - d)
                    for (a, b), (c, d) in zip(
                        apply(gram_loop, partial), target, strict=True
                    )
                ]
                assert size(residual) <= Decimal("1e-45") * size(target)
                contracted = apply(bdk, partial)
                interference = sum(
                    (
                        a * c + b * d
                        for (a, b), (c, d) in zip(
                            amplitudes[row], contracted, strict=True
                        )
                    ),
                    Decimal(0),
                )
                expected.append(2 * interference / slice_.normalisation.average)
            scale_ = max(abs(x) for x in expected)
            for actual, value in zip(bare.coefficients, expected, strict=True):
                assert abs(Decimal(actual) - value) <= Decimal("1e-40") * scale_, row
