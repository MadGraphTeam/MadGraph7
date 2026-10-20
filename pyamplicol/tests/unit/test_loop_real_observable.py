# SPDX-License-Identifier: 0BSD
"""Real physical closure must not discard a directed operator's complex part."""

from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout
from pyamplicol.loops.physical_process import NativeBareProcessProgram

pytestmark = pytest.mark.loop_fast


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "1e-14"),
        ("double-double", None, "1e-29"),
        ("arbitrary", 70, "1e-66"),
    ],
)
def test_saved_born_real_closure_and_complex_directed_metric(
    loop_native, arithmetic, precision, tolerance
):
    native = loop_native
    from symbolica import E

    options = {"arithmetic": arithmetic, "precision": precision}
    program = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    program.backend, program.options = native, options
    program._evaluator_states = {}
    program.helicities, program.born_colors = ((),), (None, None)
    order = (("QED", 2),)
    program.born_term_orders = (order, order)
    program.order_layout = PhysicalOrderLayout(
        (BornSquaredSector((("QED", 4),), ((order, order),)),), (), ()
    )
    weight = E("1/12")
    program.born_metric = ((weight, weight), (weight, weight))

    # The actual source DY metric is this exact matrix. Dependency algebra
    # proves its Hermitian sum equals |B_gamma+B_Z|^2/12 before evaluation.
    ((left, right),) = program._born_symbols()
    expanded = sum(a.conj() * b * weight for a in (left, right) for b in (left, right))
    absolute_square = (left + right).conj() * (left + right) * weight
    assert (expanded - absolute_square).expand() == E("0")
    program._compile_born_projection()
    state = program._evaluator_states["born_projector"]
    restored = native.CoefficientEvaluator(state, **options)
    # A directed metric has only the left*right cross term and is complex.
    directed = program._make_born_projector((((E("0"), E("1")), (E("0"), E("0"))),))
    pairs = [
        ("0.123456789012345678901234567890123456789", "0.071234567890123456789"),
        ("-0.017654321098765432109876543210987654321", "0.025432109876543210987"),
    ]
    values = native.NativeBatch.from_decimal(pairs, 1, 2, row_ids=[91], **options)
    ordinary = restored.evaluate_batch(values)
    complex_result = directed.evaluate_batch(values)
    assert ordinary.row_ids == complex_result.row_ids == [91]
    ((real, imaginary),) = ordinary.to_decimal()
    ((direct_real, direct_imaginary),) = complex_result.to_decimal()
    assert Decimal(imaginary) == 0
    with localcontext() as context:
        context.prec = 100
        a, b, c, d = (Decimal(v) for pair in pairs for v in pair)
        expected = ((a + c) ** 2 + (b + d) ** 2) / 12
        tolerance = Decimal(tolerance)
        assert abs(Decimal(real) - expected) < tolerance * abs(expected)
        assert abs(Decimal(direct_real) - (a * c + b * d)) < tolerance
        assert abs(Decimal(direct_imaginary) - (a * d - b * c)) < tolerance
        assert Decimal(direct_imaginary) != 0
    empty = native.NativeBatch.from_decimal([], 0, 2, **options)
    assert restored.evaluate_batch(empty).rows == 0
