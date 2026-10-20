# SPDX-License-Identifier: 0BSD
"""Independent small complete-family oracle for exact same-process recolouring.

These mathematical inputs do not certify a process/source owner. The physical
factory's ownership gate is exercised separately with retained actual programs.
"""

from dataclasses import replace

import pytest

from pyamplicol.color.connections import ColorConnection, ColorLeg, EmitGluon
from pyamplicol.loops.directed_color_projection import (
    DirectedColorColumns,
    DirectedColorInputs,
    project_directed_color_columns,
    reproject_directed_color_columns,
)

from .test_loop_color_connections import algebra as _algebra

pytestmark = pytest.mark.loop_physics

BORN = (("QCD", 1),)
LOOP = (("QCD", 3),)
OTHER_BORN = (("QED", 1),)
OTHER_LOOP = (("QED", 3),)


@pytest.fixture(scope="module")
def algebra():
    return _algebra.__wrapped__()


def _direct(nc, accuracy, histories, inputs):
    # This builds the complete family from the supplied actual dependency
    # tensors. It does not consume the reference's selected rows or degrees.
    return project_directed_color_columns(
        nc,
        accuracy,
        histories,
        inputs.born_tensors,
        inputs.born_orders,
        inputs.born_pairs,
        inputs.virtual_pairs,
        inputs.columns,
    )


def _equal(actual, expected):
    assert actual.nc == expected.nc
    assert actual.accuracy == expected.accuracy
    assert actual.families == expected.families
    assert actual.history_families == expected.history_families
    assert actual.born == expected.born
    assert actual.columns == expected.columns


@pytest.fixture(scope="module")
def graded(algebra):
    s, t, nc, *_ = algebra
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 8))
    tensor = t.TensorExpression.color_t(8, 3)(3, 1, 2)
    q, qbar = (ColorConnection(legs, (EmitGluon(leg, -1),)) for leg in (1, 2))
    histories = ((q, q), (q, qbar), (qbar, q))
    bare = DirectedColorColumns((tensor, tensor), (LOOP, LOOP), (s.E("1"),) * 2)
    uv = DirectedColorColumns((tensor,), (LOOP,), (nc**2,))
    r2 = replace(uv, factors=(nc,))
    inactive = replace(uv, orders=(OTHER_LOOP,), factors=(s.E("0"),))
    reference = DirectedColorInputs(
        (tensor, tensor),
        (BORN, OTHER_BORN),
        (((BORN, BORN),), ((OTHER_BORN, OTHER_BORN),)),
        (((BORN, LOOP),), ((OTHER_BORN, OTHER_LOOP),)),
        (bare, uv, r2, inactive),
        ("bare", "UV", "R2", "field"),
    )
    # Reorder domains and Born columns, remove duplicate bare occurrences, and
    # add repeated UV/Born columns. Complete support stays equal; multiplicities
    # deliberately differ. Every target current/output column must survive.
    target = replace(
        reference,
        born_tensors=(tensor, tensor, tensor),
        born_orders=(OTHER_BORN, BORN, BORN),
        columns=(
            inactive,
            replace(bare, tensors=(tensor,), orders=(LOOP,), factors=(s.E("1"),)),
            replace(
                uv, tensors=(tensor,) * 3, orders=(LOOP,) * 3, factors=(nc**2,) * 3
            ),
            r2,
            uv,
        ),
        domains=("field", "bare", "UV", "R2", "UV"),
    )
    full = _direct(nc, "full", histories, reference)
    return nc, tensor, histories, reference, target, full


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_reprojection_matches_separately_built_complete_graded_family(graded, accuracy):
    nc, _tensor, histories, reference, target, full = graded
    actual = reproject_directed_color_columns(full, accuracy, reference, target)
    expected = _direct(nc, accuracy, histories, target)
    _equal(actual, expected)
    assert actual.families[0].born_degrees == (3, 3)
    assert actual.families[0].virtual_degrees == (5, None)
    assert [len(block[0][0]) for block in actual.columns] == [1, 1, 3, 1, 1]
    assert all(len(matrix) == 3 for block in actual.columns for matrix in block)
    assert all(
        value == 0 for matrix in actual.columns[0] for row in matrix for value in row
    )
    # The q/qbar dipole cannot choose a new, lower anchor from its selected row.
    assert full.columns[0][2][0][0] != 0
    if accuracy == "lc":
        assert actual.columns[1][2][1][0] == 0
        assert actual.columns[3][0][1][0] == 0
        assert actual.columns[2][0][1][0] == nc**5 / 4
    if accuracy == "nlc":
        assert actual.columns[1][0][1][0] == nc**3 / 4
        assert actual.columns[3][0][1][0] == nc**4 / 4
    assert actual.bound(18) == expected.bound(18)


@pytest.fixture(scope="module")
def rectangular(algebra):
    s, _t, nc, _space, legs, native, _literal = algebra
    alpha = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(1, -2)))
    beta = ColorConnection(legs, (EmitGluon(2, -1), EmitGluon(3, -2)))
    histories = ((alpha, beta), (beta, alpha))
    reference = DirectedColorInputs(
        native[:2],
        (BORN, BORN),
        (((BORN, BORN),),),
        (((BORN, LOOP),),),
        (DirectedColorColumns(native, (LOOP,) * 3, (s.E("1"),) * 3),),
        ("bare",),
    )
    target = replace(
        reference,
        born_tensors=(native[1], native[0], native[1]),
        born_orders=(BORN,) * 3,
        columns=(
            DirectedColorColumns(
                (native[2], native[0], native[1], native[2]),
                (LOOP,) * 4,
                (s.E("1"),) * 4,
            ),
        ),
    )
    full = _direct(nc, "full", histories, reference)
    return histories, reference, target, full


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_rectangular_both_directions_survive_column_gather(
    algebra, rectangular, accuracy
):
    s, _t, nc, *_ = algebra
    histories, reference, target, full = rectangular
    actual = reproject_directed_color_columns(full, accuracy, reference, target)
    _equal(actual, _direct(nc, accuracy, histories, target))
    assert actual.columns[0][0] == actual.columns[0][3]
    assert actual.columns[0][1] == actual.columns[0][2]
    # The full rectangular test must distinguish the two integrated terms;
    # complex test currents expose a mistaken 2Re(first) gather shortcut.
    if accuracy == "full":
        born = tuple(map(s.E, ("1+2i", "3-i", "2-3i")))
        loop = tuple(map(s.E, ("2+i", "-1+3i", "4-2i", "-3+i")))

        def cross(matrix):
            return sum(
                (
                    born[i].conj() * value * loop[j]
                    for i, row in enumerate(matrix)
                    for j, value in enumerate(row)
                ),
                s.E("0"),
            )

        first, second = map(cross, actual.columns[0][:2])
        correct = (first + second.conj()).replace(nc, 3).together()
        wrong = (first + first.conj()).replace(nc, 3).together()
        assert correct != correct.conj() and correct != wrong
        swapped = cross(actual.columns[0][2]) + cross(actual.columns[0][3]).conj()
        assert swapped.replace(nc, 3).together() == correct.conj()


def _column(inputs, index, **changes):
    columns = list(inputs.columns)
    columns[index] = replace(columns[index], **changes)
    return replace(inputs, columns=tuple(columns))


@pytest.mark.parametrize(
    "change",
    (
        "formal_factor",
        "formal_tensor",
        "inactive_tensor",
        "inactive_mask",
        "compensating_factor",
        "domain",
        "absolute_order",
        "missing_support",
        "born_tensor",
        "dual_port",
        "sector_catalogue",
    ),
)
def test_rejects_different_exact_support_despite_fixed_nc_or_zero_agreement(
    graded, change
):
    nc, tensor, _histories, reference, target, full = graded
    if change == "formal_factor":
        changed = _column(target, 4, factors=(nc**2 + nc - 3,))
    elif change == "formal_tensor":
        changed = _column(target, 4, tensors=((1 + nc - 3) * tensor,))
    elif change == "inactive_tensor":
        changed = _column(target, 0, tensors=(2 * tensor,))
    elif change == "inactive_mask":
        changed = _column(target, 0, factors=(nc - 3,))
    elif change == "compensating_factor":
        changed = _column(target, 4, tensors=(nc * tensor,), factors=(nc,))
    elif change == "domain":
        changed = replace(target, domains=("field", "R2", "UV", "R2", "UV"))
    elif change == "absolute_order":
        changed = _column(target, 4, orders=(OTHER_LOOP,))
    elif change == "missing_support":
        changed = replace(
            target,
            columns=target.columns[:3] + target.columns[4:],
            domains=target.domains[:3] + target.domains[4:],
        )
    elif change == "born_tensor":
        changed = replace(target, born_tensors=(2 * tensor, *target.born_tensors[1:]))
    elif change == "dual_port":
        # The dependency tensor has the same representation dimensions but
        # reverses the actual fundamental/antifundamental external ports.
        from symbolica import community

        changed = _column(
            target,
            4,
            tensors=(community.tensor.TensorExpression.color_t(8, 3)(3, 2, 1),),
        )
    else:
        changed = replace(target, virtual_pairs=target.virtual_pairs[::-1])
    with pytest.raises(ValueError, match=r"support|sector"):
        reproject_directed_color_columns(full, "nlc", reference, changed)


@pytest.mark.parametrize(
    "change",
    (
        "projected_reference",
        "born_shape",
        "direction_shape",
        "column_shape",
        "degree_axis",
        "missing_family",
        "understated_degree",
    ),
)
def test_rejects_incomplete_or_nonfull_reference(graded, change):
    _nc, _tensor, _histories, reference, target, full = graded
    if change == "projected_reference":
        changed = replace(full, accuracy="lc")
    elif change == "born_shape":
        changed = replace(full, born=(full.born[0][:-1], *full.born[1:]))
    elif change == "direction_shape":
        changed = replace(full, columns=(full.columns[0][:-1], *full.columns[1:]))
    elif change == "column_shape":
        block = (
            (full.columns[0][0][0][:-1], *full.columns[0][0][1:]),
            *full.columns[0][1:],
        )
        changed = replace(full, columns=(block, *full.columns[1:]))
    elif change == "degree_axis":
        changed = replace(
            full, families=(replace(full.families[0], virtual_degrees=(5,)),)
        )
    elif change == "missing_family":
        changed = replace(
            full, history_families=(len(full.families), *full.history_families[1:])
        )
    else:
        changed = replace(
            full, families=(replace(full.families[0], virtual_degrees=(0, None)),)
        )
    with pytest.raises(ValueError):
        reproject_directed_color_columns(changed, "nlc", reference, target)
