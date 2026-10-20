# SPDX-License-Identifier: 0BSD
"""Native ordered charges against existing independent tree colour contractions."""

from fractions import Fraction

import pytest

from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    ColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    OpenColorString,
    SplitGluon,
    apply_color_connection,
    contract_connected_tensor_nc_terms,
)
from pyamplicol.loops.color_bilinears import SUNColorSpace
from pyamplicol.loops.color_connections import (
    apply_native_color_connection,
    directed_color_bilinears,
)

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def algebra():
    s = pytest.importorskip("symbolica")
    from symbolica.community import tensor as t

    nc = s.S("loop_color_connection_test::Nc", is_real=True)
    space = SUNColorSpace(nc)
    generator = t.TensorExpression.color_t(8, 3)
    fundamental, adjoint = t.Representation.cof(3), t.Representation.coad(8)
    identity = t.TensorExpression.g(fundamental, fundamental.dual())(1, 2)
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 8), ColorLeg(4, 8))
    native = (
        generator(3, 1, "k") * generator(4, "k", 2),
        generator(4, 1, "k") * generator(3, "k", 2),
        identity * t.TensorExpression.g(adjoint, adjoint)(3, 4),
    )
    # Tree colour strings use tau=sqrt(2)T; two generators give a factor1/2.
    literal = (
        (ColorTensor((OpenColorString(1, (3, 4), 2),)), Fraction(1, 2)),
        (ColorTensor((OpenColorString(1, (4, 3), 2),)), Fraction(1, 2)),
        (ColorTensor((OpenColorString(1, (), 2),), ((3, 4),)), Fraction(1)),
    )
    return s, t, nc, space, legs, native, literal


def _literal_overlap(algebra, left, alpha, right, beta):
    s, _t, nc, *_ = algebra
    terms = contract_connected_tensor_nc_terms(
        apply_color_connection(
            left[0], alpha, coefficient=ExactColorCoefficient(left[1])
        ),
        apply_color_connection(
            right[0], beta, coefficient=ExactColorCoefficient(right[1])
        ),
    )
    return sum(
        (
            (s.E(str(coefficient.real)) + s.E("1i") * s.E(str(coefficient.imag)))
            * nc**power
            for power, coefficient in terms.items()
        ),
        s.E("0"),
    )


@pytest.mark.parametrize(
    "operations",
    [
        (),
        (EmitGluon(1, -1),),
        (EmitGluon(2, -1),),
        (EmitGluon(3, -1),),
        (EmitGluon(4, -1),),
        (EmitGluon(3, -1), EmitGluon(-1, -2)),
        (EmitGluon(1, -1), SplitGluon(-1, -2, -3)),
        (SplitGluon(3, -1, -2),),
    ],
)
def test_charges_and_splittings_match_existing_exact_tree_oracle(algebra, operations):
    _s, _t, _nc, space, legs, native, literal = algebra
    connection = ColorConnection(legs, operations)
    images = tuple(apply_native_color_connection(value, connection) for value in native)
    for i in range(3):
        for j in range(3):
            actual = space.overlap(images[i], images[j])
            expected = _literal_overlap(
                algebra, literal[i], connection, literal[j], connection
            )
            assert (actual - expected).together() == 0


def test_charge_conservation_and_physical_casimirs(algebra):
    _s, t, nc, space, *_ = algebra
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 8))
    value = t.TensorExpression.color_t(8, 3)(3, 1, 2)
    images = [
        apply_native_color_connection(value, ColorConnection(legs, (EmitGluon(i, -1),)))
        for i in (1, 2, 3)
    ]
    assert space.overlap(sum(images[1:], images[0]), sum(images[1:], images[0])) == 0
    norm = space.overlap(value, value)
    for image, casimir in zip(
        images, ((nc**2 - 1) / (2 * nc), (nc**2 - 1) / (2 * nc), nc), strict=True
    ):
        assert (space.overlap(image, image) - casimir * norm).together() == 0
    fundamental = t.Representation.cof(3)
    identity = t.TensorExpression.g(fundamental, fundamental.dual())(1, 2)
    q, qbar = [
        apply_native_color_connection(
            identity, ColorConnection(legs[:2], (EmitGluon(i, -1),))
        )
        for i in (1, 2)
    ]
    assert space.overlap(q, qbar) == -(nc**2 - 1) / 2


def test_rectangular_directed_cross_terms_and_quartic_invariant_substitution(algebra):
    s, _t, nc, space, legs, native, literal = algebra
    alpha = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(1, -2)))
    beta = ColorConnection(legs, (EmitGluon(2, -1), EmitGluon(3, -2)))
    kernels = directed_color_bilinears(space, native[:2], native, alpha, beta)
    swapped = directed_color_bilinears(space, native[:2], native, beta, alpha)
    for i in range(2):
        for j in range(3):
            expected = _literal_overlap(algebra, literal[i], alpha, literal[j], beta)
            assert (kernels.born_loop[i][j] - expected).together() == 0
            assert (
                kernels.loop_born[j][i] - swapped.born_loop[i][j].conj()
            ).together() == 0
    b, loop = (s.E("1+2i"), s.E("3-1i")), (s.E("2+1i"), s.E("-1+3i"), s.E("4-2i"))
    first = sum(
        b[i].conj() * kernels.born_loop[i][j] * loop[j]
        for i in range(2)
        for j in range(3)
    )
    second = sum(
        loop[i].conj() * kernels.loop_born[i][j] * b[j]
        for i in range(3)
        for j in range(2)
    )
    assert (first + second).replace(nc, 3).together() == s.E("-232/3+72i")
    assert (first + first.conj()).replace(nc, 3).together() == s.E("146/3")
    # This history pair retains a mixed quartic Gram invariant until the
    # dependency receives the recognized relation dA=Nc**2-1.
    alpha = ColorConnection(legs, (EmitGluon(2, -1), EmitGluon(1, -2)))
    beta = ColorConnection(legs, (EmitGluon(3, -2), EmitGluon(1, -1)))
    matrix = directed_color_bilinears(space, native[:2], native, alpha, beta)
    for i in range(2):
        for j in range(3):
            expected = _literal_overlap(algebra, literal[i], alpha, literal[j], beta)
            assert (matrix.born_loop[i][j] - expected).together() == 0


def test_ports_zero_and_internal_index_scoping(algebra):
    s, t, _nc, space, legs, native, _literal = algebra
    connection = ColorConnection(legs, (EmitGluon(1, -1),))
    generator = t.TensorExpression.color_t(8, 3)
    collision = s.S("loop_color_connection::contracted_0")
    colliding = generator(3, 1, collision) * generator(4, collision, 2)
    expected = apply_native_color_connection(native[0], connection)
    actual = apply_native_color_connection(colliding, connection)
    assert space.overlap(actual, actual) == space.overlap(expected, expected)
    zero = apply_native_color_connection(t.TensorExpression(s.E("0")), connection)
    assert zero.to_expression() == 0 and set(zero.axes) == set(expected.axes)
    with pytest.raises(ValueError, match="typed initial"):
        apply_native_color_connection(
            native[0].rename_indices({1: 2, 2: 1}), connection
        )
    with pytest.raises(TypeError, match="TensorExpression"):
        apply_native_color_connection(s.E("1"), connection)
    with pytest.raises(ValueError, match="output colour legs"):
        directed_color_bilinears(
            space, native[:2], native, connection, ColorConnection(legs)
        )
