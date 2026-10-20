# SPDX-License-Identifier: 0BSD
"""Independent exact tensors test complete-family grading, not event values."""

from dataclasses import replace

import pytest

from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    EmitGluon,
    SplitGluon,
)
from pyamplicol.loops.directed_color_projection import (
    DirectedColorColumns,
    project_directed_color_columns,
)

from .test_loop_color_connections import _literal_overlap
from .test_loop_color_connections import algebra as _algebra

pytestmark = pytest.mark.loop_fast

BORN = (("QCD", 1),)
LOOP = (("QCD", 3),)
BB = (((BORN, BORN),),)
BL = (((BORN, LOOP),),)


@pytest.fixture(scope="module")
def algebra():
    return _algebra.__wrapped__()


def _project(nc, accuracy, histories, born, loop, *, extra=()):
    from symbolica import E

    return project_directed_color_columns(
        nc,
        accuracy,
        histories,
        born,
        (BORN,) * len(born),
        BB,
        BL,
        (
            DirectedColorColumns(
                tuple(loop), (LOOP,) * len(loop), (E("1"),) * len(loop)
            ),
            *extra,
        ),
    )


@pytest.fixture(scope="module")
def qqg(algebra):
    _s, t, nc, _space, *_ = algebra
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 8))
    tensor = t.TensorExpression.color_t(8, 3)(3, 1, 2)
    histories = tuple(ColorConnection(legs, (EmitGluon(i, -1),)) for i in (1, 2, 3))
    return nc, legs, tensor, histories


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_suppressed_dipole_is_not_promoted_and_grouping_is_invariant(qqg, accuracy):
    nc, _legs, tensor, histories = qqg
    pair = (histories[0], histories[1])
    alone = _project(nc, accuracy, (pair,), (tensor,), (tensor,))
    grouped = _project(
        nc,
        accuracy,
        tuple((histories[0], right) for right in histories),
        (tensor,),
        (tensor,),
    )
    assert (
        alone.families[0].virtual_degrees == grouped.families[0].virtual_degrees == (3,)
    )
    assert alone.columns[0][0] == grouped.columns[0][2]
    assert sum(matrix[0][0] for matrix in grouped.columns[0][::2]).together() == 0
    expected = {"full": (nc**2 - 1) / (4 * nc), "lc": 0, "nlc": nc / 4}[accuracy]
    assert (alone.columns[0][0][0][0] - expected).together() == 0


def test_orthogonal_born_loop_direction_is_activated(algebra):
    s, t, nc, _space, *_ = algebra
    fundamental = t.Representation.cof(3)
    identity = t.TensorExpression.g(fundamental, fundamental.dual())
    generator = t.TensorExpression.color_t(8, 3)
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 3), ColorLeg(4, -3))
    born = identity(1, 2) * identity(3, 4)
    loop = generator("internal", 1, 2) * generator("internal", 3, 4)
    histories = tuple(ColorConnection(legs, (EmitGluon(i, -1),)) for i in range(1, 5))
    pairs = tuple((a, b) for a in histories for b in histories)
    result = _project(nc, "lc", pairs, (born,), (loop,))
    assert result.families[0].virtual_degrees == (2,)
    assert result.columns[0][2 * 2][0][0] == nc**2 / 4
    assert all(result.columns[0][2 * (i * 4 + i)][0][0] == 0 for i in range(4))
    assert all(
        sum(
            (result.columns[0][2 * (i * 4 + j)][0][0] for j in range(4)), s.E("0")
        ).together()
        == 0
        for i in range(4)
    )


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_ct_and_field_grades_share_anchor_and_keep_odd_suppression(qqg, accuracy):
    nc, _legs, tensor, histories = qqg
    high = DirectedColorColumns((tensor,), (LOOP,), (nc**2,))
    odd = replace(high, factors=(nc,))
    inactive = replace(high, factors=(nc * 0,))
    result = _project(
        nc,
        accuracy,
        ((histories[0], histories[0]),),
        (tensor,),
        (tensor,),
        extra=(high, odd, inactive),
    )
    assert result.families[0].born_degrees == (3,)
    assert result.families[0].virtual_degrees == (5,)
    assert result.columns[3][0][0][0] == 0
    if accuracy == "lc":
        assert result.columns[0][0][0][0] == result.columns[2][0][0][0] == 0
        assert result.columns[1][0][0][0] == nc**5 / 4
    if accuracy == "nlc":
        assert result.columns[0][0][0][0] == nc**3 / 4
        assert result.columns[2][0][0][0] == nc**4 / 4


def test_zero_sector_identity_and_original_gluon_splitting(qqg, algebra):
    nc, legs, tensor, _histories = qqg
    s, t, *_ = algebra
    identity = ColorConnection(legs)
    split = ColorConnection(legs, (SplitGluon(3, 17, 29),))
    zero = t.TensorExpression(s.E("0"))
    result = _project(
        nc, "nlc", ((identity, identity), (split, split)), (tensor,), (zero,)
    )
    assert [x.order for x in result.families] == [0, 1]
    assert all(x.virtual_degrees == (None,) for x in result.families)
    assert all(matrix[0][0] == 0 for matrix in result.columns[0])
    assert result.families[1].born_degrees == (2,)
    assert (result.born[1][0][0] - (nc**2 - 1) / 4).together() == 0


def test_nonhermitian_rectangular_nlc_against_complete_independent_tree_oracle(algebra):
    s, _t, nc, _space, legs, native, literal = algebra
    alpha = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(1, -2)))
    beta = ColorConnection(legs, (EmitGluon(2, -1), EmitGluon(3, -2)))
    result = _project(nc, "nlc", ((alpha, beta), (beta, alpha)), native[:2], native)
    family = result.families[0]
    # The independent tree tensor oracle determines the common degree over the
    # complete family, including entries absent from these two requested pairs.
    exact_rows = [
        _literal_overlap(algebra, left, a, right, b)
        for a in family.histories
        for b in family.histories
        for left in literal[:2]
        for right in literal
    ]
    from pyamplicol.loops.color_projection import prepare_color_projection

    reference = prepare_color_projection(nc, [[value] for value in exact_rows])
    assert family.virtual_degrees == (reference.leading_degree,)
    for direction, (a, b) in enumerate(((alpha, beta), (beta, alpha))):
        for i in range(2):
            for j in range(3):
                expression = _literal_overlap(algebra, literal[i], a, literal[j], b)
                row = prepare_color_projection(nc, [[expression]])
                expected = replace(
                    row, leading_degree=reference.leading_degree
                ).project("nlc")[0]
                assert (result.columns[0][direction][i][j] - expected).together() == 0
    assert result.columns[0][0] == result.columns[0][3]
    assert result.columns[0][1] == result.columns[0][2]
    born, loop = (s.E("1+2i"), s.E("3-1i")), (s.E("2+1i"), s.E("-1+3i"), s.E("4-2i"))

    def cross(matrix):
        return sum(
            (
                born[i].conj() * matrix[i][j] * loop[j]
                for i in range(2)
                for j in range(3)
            ),
            s.E("0"),
        )

    first, swapped = map(cross, result.columns[0][:2])
    actual = (first + swapped.conj()).replace(nc, 3).together()
    wrong = (first + first.conj()).replace(nc, 3).together()
    assert actual != wrong and actual != actual.conj()
    assert (cross(result.columns[0][2]) + cross(result.columns[0][3]).conj()).replace(
        nc, 3
    ).together() == actual.conj()
