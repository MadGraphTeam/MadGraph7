# SPDX-License-Identifier: 0BSD
"""Independent typed tensor actions and exact rectangular coverage controls."""

from itertools import permutations
from types import SimpleNamespace

import pytest

from pyamplicol.color.symmetric_group import _relative_permutation
from pyamplicol.loops.color_bilinears import PhysicalColorSpace, SUNColorSpace
from pyamplicol.loops.color_orbits import (
    _canonical_with_ports,
    prepare_rectangular_color_certificate,
)

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def algebra():
    s = pytest.importorskip("symbolica")
    from symbolica.community import tensor as t

    g = t.TensorExpression.g(t.Representation.coad(8), t.Representation.coad(8))
    x = g("a", "c") * g("b", "d")
    y = g("a", "d") * g("b", "c")
    z = g("a", "b") * g("c", "d")
    return (
        s,
        t,
        (x, y, z),
        tuple(
            next(
                axis.index
                for axis in x.axes
                if axis.index.get_name().rsplit("::", 1)[-1] == name
            )
            for name in ("c", "d")
        ),
    )


def _expanded_metric(plan, bank):
    """Expand without summing amplitudes; every alias pair must occur once."""
    s = pytest.importorskip("symbolica")
    compressed = {}
    elements = tuple(permutations(range(plan.degree)))
    ranks = {p: i for i, p in enumerate(elements)}
    for kernel in plan.kernels:
        if kernel.bank != bank:
            continue
        left = plan.born.orbits[kernel.born_orbit]
        right = plan.loop.orbits[kernel.loop_orbit]
        for g, i in zip(elements, left.aliases, strict=True):
            for h, j in zip(elements, right.aliases, strict=True):
                assert (i, j) not in compressed
                compressed[i, j] = kernel.values[ranks[_relative_permutation(g, h)]]
    for entry in plan.residual_entries:
        if entry.bank == bank:
            pair = entry.born_alias, entry.loop_alias
            assert pair not in compressed
            compressed[pair] = entry.value
    assert len(compressed) == len(plan.born.aliases) * len(plan.loop.aliases)
    result = [
        [s.E("0") for _ in range(plan.loop.column_count)]
        for _ in range(plan.born.column_count)
    ]
    for (i, j), value in compressed.items():
        for column_i in plan.born.aliases[i].columns:
            for column_j in plan.loop.aliases[j].columns:
                result[column_i][column_j] = value
    return tuple(map(tuple, result))


def test_complex_rectangular_actual_tensors_aliases_and_both_residual_sides(algebra):
    s, _, (x, y, z), labels = algebra
    born = (x, y, 2 * s.E("1i") * x, 2 * s.E("1i") * y, z, x)
    loop = (
        3 * x,
        3 * y,
        s.E("1+1i") * x,
        s.E("1+1i") * y,
        7 * x,
        7 * y,
        z,
        2 * z,
        3 * x,
    )
    metric = PhysicalColorSpace().rectangular(born, loop)
    qcd, qed = (("QCD", 2),), (("QED", 2),)
    plan = prepare_rectangular_color_certificate(
        born,
        loop,
        born_orders=(qcd, qcd, qed, qed, qcd, qcd),
        loop_orders=((("QCD", 4),),) * len(loop),
        labels=labels,
        metrics={"physical": metric},
    )
    assert len(plan.born.orbits) == 2
    assert len(plan.loop.orbits) == 3
    assert plan.born.residual_aliases == (4,)
    assert plan.loop.residual_aliases == (6, 7)
    assert plan.born.aliases[0].columns == (0, 5)
    assert plan.loop.aliases[0].columns == (0, 8)
    assert plan.born.aliases[2].order == qed
    assert len(plan.kernels) == 6
    assert len(plan.residual_entries) == 16  # 5*8 - (2*2)*(3*2).
    assert _expanded_metric(plan, "physical") == metric
    # Independent explicit representation components fix a genuinely complex
    # overlap; certificate construction does not evaluate components itself.
    left, right = born[2].to_tensor().to_dense(), loop[2].to_tensor().to_dense()
    right = right.permute_axes([right.axes.index(axis) for axis in left.axes])
    dense = sum(left[i].conj() * right[i] for i in range(len(left)))
    assert metric[2][2] == dense == s.E("128-128i")


def test_formal_order_banks_and_non_equivariant_pair_remain_exact(algebra):
    s, _, (x, y, z), labels = algebra
    nc = s.S("rectangular_orbit_test::Nc", is_real=True)
    born, loop = (x, y, x, y, z), (x, y, 3 * x, 3 * y, z)
    qcd, qed = (("QCD", 2),), (("QED", 2),)
    born_orders = (qcd, qcd, qed, qed, qcd)
    loop_orders = (qcd, qcd, qed, qed, qcd)
    formal = SUNColorSpace(nc).rectangular(born, loop)
    # Same tensor at a different absolute order remains an independent class.
    selected = tuple(
        tuple(
            value if born_orders[i] == loop_orders[j] else s.E("0")
            for j, value in enumerate(row)
        )
        for i, row in enumerate(formal)
    )
    broken = [list(row) for row in selected]
    broken[0][0] += nc - 3  # Cannot certify by testing only Nc=3.
    broken = tuple(map(tuple, broken))
    plan = prepare_rectangular_color_certificate(
        born,
        loop,
        born_orders=born_orders,
        loop_orders=loop_orders,
        labels=labels,
        metrics={"same-order": selected, "marked": broken},
        nc=nc,
    )
    assert len(plan.born.aliases) == 5
    assert plan.born.aliases[0].order != plan.born.aliases[2].order
    assert len([k for k in plan.kernels if k.bank == "same-order"]) == 4
    assert len([k for k in plan.kernels if k.bank == "marked"]) == 3
    assert len([e for e in plan.residual_entries if e.bank == "marked"]) == 13
    assert _expanded_metric(plan, "same-order") == selected
    assert _expanded_metric(plan, "marked") == broken
    assert any(nc in e.value.get_all_symbols() for e in plan.residual_entries)


def test_partial_stabilized_empty_identity_and_large_action(algebra):
    s, t, (x, y, z), labels = algebra
    order = (("QCD", 2),)
    # z is stabilized; the x orbit is incomplete. Both remain exact residuals.
    plan = prepare_rectangular_color_certificate(
        (x, z),
        (y, z),
        born_orders=(order, order),
        loop_orders=(order, order),
        labels=labels,
        metrics={"partial": PhysicalColorSpace().rectangular((x, z), (y, z))},
    )
    assert not plan.born.orbits and not plan.loop.orbits
    assert len(plan.residual_entries) == 4
    identity = prepare_rectangular_color_certificate(
        (x,),
        (s.E("1i") * x,),
        born_orders=(order,),
        loop_orders=(order,),
        labels=(),
        metrics={"identity": ((s.E("64i"),),)},
    )
    assert identity.degree == 0 and len(identity.kernels) == 1
    assert _expanded_metric(identity, "identity") == ((s.E("64i"),),)
    empty = prepare_rectangular_color_certificate(
        (),
        (),
        born_orders=(),
        loop_orders=(),
        labels=(),
        metrics={"empty": ()},
    )
    assert not empty.kernels and not empty.residual_entries
    # An actual twelve-port adjoint tensor admits the action domain. One class
    # cannot make a 12! regular orbit, so no factorial-size list is allocated.
    names = tuple(s.S(f"large_orbit_action::a{i}") for i in range(12))
    g = t.TensorExpression.g(t.Representation.coad(8), t.Representation.coad(8))
    tensor = g(names[0], names[1])
    for i in range(2, 12, 2):
        tensor *= g(names[i], names[i + 1])
    large = prepare_rectangular_color_certificate(
        (tensor,),
        (tensor,),
        born_orders=(order,),
        loop_orders=(order,),
        labels=names,
        metrics={"large": ((s.E(str(8**6)),),)},
    )
    assert large.degree == 12 and not large.born.orbits
    assert len(large.residual_entries) == 1


def test_typed_port_direction_and_dependency_canonicalization_guard(algebra):
    s, t, (x, y, _), labels = algebra
    order = (("QCD", 2),)
    fundamental = t.Representation.cof(3)
    delta = t.TensorExpression.g(fundamental, fundamental.dual())
    direct = delta("i", "j") * x
    malformed = delta("j", "i") * x
    # Fixed fundamental ports must survive even when only adjoints are moved.
    fake = SimpleNamespace(axes=direct.axes, canonize=lambda: malformed)
    canonical, reason = _canonical_with_ports(fake)
    assert canonical is None and "typed external ports" in reason
    with pytest.raises(ValueError, match="port types or directions"):
        prepare_rectangular_color_certificate(
            (direct,),
            (malformed,),
            born_orders=(order,),
            loop_orders=(order,),
            labels=labels,
            metrics={"bad": ((s.E("1"),),)},
        )
    with pytest.raises(ValueError, match="adjoint ports"):
        prepare_rectangular_color_certificate(
            (direct,),
            (direct,),
            born_orders=(order,),
            loop_orders=(order,),
            labels=tuple(
                axis.index
                for axis in direct.axes
                if axis.representation.name != t.Representation.coad(8).name
            ),
            metrics={"bad": ((s.E("1"),),)},
        )
    # A scalar lookalike or matrix coincidence cannot bypass actual tensor types.
    with pytest.raises(TypeError, match="dependency tensors"):
        prepare_rectangular_color_certificate(
            (s.E("1"),),
            (s.E("1"),),
            born_orders=(order,),
            loop_orders=(order,),
            labels=(),
            metrics={"bad": ((s.E("1"),),)},
        )
    with pytest.raises(ValueError, match="exact tensor/order aliases"):
        prepare_rectangular_color_certificate(
            (x, x),
            (y,),
            born_orders=(order, order),
            loop_orders=(order,),
            labels=labels,
            metrics={"bad": ((s.E("1"),), (s.E("2"),))},
        )


def test_nonabelian_tensor_action_preserves_existing_formal_projection(algebra):
    s, t, _, _ = algebra
    from pyamplicol.loops.color_projection import prepare_color_projection

    nc = s.S("nonabelian_orbit_test::Nc", is_real=True)
    labels = tuple(s.S(f"nonabelian_action::{name}") for name in ("x", "y", "z"))
    anchors = tuple(s.S(f"nonabelian_action::{name}") for name in ("a", "b", "c"))
    g = t.TensorExpression.g(t.Representation.coad(8), t.Representation.coad(8))
    seed = (
        g(anchors[0], labels[0]) * g(anchors[1], labels[1]) * g(anchors[2], labels[2])
    )
    born = tuple(
        seed.rename_indices(dict(zip(labels, image, strict=True)))
        for image in permutations(labels)
    )
    # An unequal superposition prevents a central class-function overlap from
    # hiding a left/right group-action error in this nonabelian control.
    loop_seed = s.E("2+3i") * born[0] + s.E("1-2i") * born[1] + 3 * born[2]
    loop = tuple(
        loop_seed.rename_indices(dict(zip(labels, image, strict=True)))
        for image in permutations(labels)
    )
    matrix = SUNColorSpace(nc).rectangular(born, loop)
    projection = prepare_color_projection(
        nc, [(value,) for row in matrix for value in row]
    )
    assert projection.leading_degree == 6
    metrics = {}
    for accuracy in ("full", "lc", "nlc"):
        values = projection.project(accuracy)
        metrics[accuracy] = tuple(tuple(values[i * 6 : (i + 1) * 6]) for i in range(6))
    order = (("QCD", 4),)
    plan = prepare_rectangular_color_certificate(
        born,
        loop,
        born_orders=(order,) * 6,
        loop_orders=(order,) * 6,
        labels=labels,
        metrics=metrics,
        nc=nc,
    )
    assert plan.degree == 3
    assert len(plan.born.orbits) == len(plan.loop.orbits) == 1
    assert len(plan.kernels) == 3 and not plan.residual_entries
    for kernel in plan.kernels:
        # Lexicographic S3 elements 1 and 2 are different transpositions.
        # The kernel must distinguish them even after LC/NLC projection.
        assert (kernel.values[1] - kernel.values[2]).together() != 0
    for name, metric in metrics.items():
        assert _expanded_metric(plan, name) == metric
    assert metrics["lc"] != metrics["nlc"] != metrics["full"]


def test_action_rejects_mixed_adjoint_dimensions(algebra):
    s, t, _, _ = algebra
    a, b, x, y = s.S(*(f"typed_action::{name}" for name in "abxy"))
    adjoint3, adjoint8 = t.Representation.coad(3), t.Representation.coad(8)
    tensor = t.TensorExpression.g(adjoint3, adjoint3)(a, x) * t.TensorExpression.g(
        adjoint8, adjoint8
    )(b, y)
    with pytest.raises(ValueError, match="representations or directions"):
        prepare_rectangular_color_certificate(
            (tensor,),
            (tensor,),
            born_orders=((),),
            loop_orders=((),),
            labels=(x, y),
            metrics={"mixed": ((s.E("24"),),)},
        )
