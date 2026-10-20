# SPDX-License-Identifier: 0BSD
"""Canonical Born anchors are coherent, signed and distinct from tree flows."""

import json
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pyamplicol.loops.color_components import (
    BornColorComponents,
    prepare_born_color_components,
)
from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout
from pyamplicol.loops.physical_process import NativeBareProcessProgram

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def tensors():
    pytest.importorskip("symbolica")
    from symbolica.community import tensor as t

    fundamental = t.Representation.cof(3)
    delta = t.TensorExpression.g(fundamental, fundamental.dual())
    direct = delta("leg0", "leg1") * delta("leg2", "leg3")
    crossed = delta("leg0", "leg3") * delta("leg2", "leg1")
    return direct, crossed


def test_native_dummy_canonicalization_and_stable_component_order(tensors):
    from symbolica.community import tensor as t

    direct, crossed = tensors
    generator = t.TensorExpression.color_t(8, 3)
    first = generator("x", "leg0", "leg1") * generator("x", "leg2", "leg3")
    renamed = generator("y", "leg2", "leg3") * generator("y", "leg0", "leg1")
    original = (direct, first, crossed, renamed, direct)
    plan = prepare_born_color_components(original)
    assert len(plan.ids) == 3
    assert set(plan.members) == {(0, 4), (1, 3), (2,)}
    permutation = (2, 4, 3, 0, 1)
    reordered = prepare_born_color_components(tuple(original[i] for i in permutation))
    assert plan.ids == reordered.ids
    assert tuple(set(group) for group in plan.members) == tuple(
        {permutation[index] for index in group} for group in reordered.members
    )
    # A physical external-leg permutation has the same free-port set but a
    # different tensor. It must not be quotiented like an internal dummy rename.
    permuted = direct.rename_indices({"leg1": "leg3", "leg3": "leg1"})
    assert permuted.to_expression() == crossed.to_expression()
    assert len(prepare_born_color_components((direct, permuted)).ids) == 2
    # No claim that this overcomplete set solves Fierz/linear basis identities.
    assert plan.as_dict()["kind"] == "canonical-born-tensor-partition"


def test_partition_and_port_errors(tensors):
    direct, crossed = tensors
    for ids, members in (
        (["a", "a"], [[0], [1]]),
        (["a"], [[0, 0]]),
        (["a"], [[1]]),
        (["a"], [[]]),
    ):
        with pytest.raises(ValueError, match="Born colour components"):
            BornColorComponents(ids, members)
    plan = prepare_born_color_components((direct, crossed))
    with pytest.raises(ValueError, match="every Born column"):
        plan.mask_bra(((1,),))
    with pytest.raises(ValueError, match="rectangular"):
        plan.mask_bra(((1,), (1, 2)))
    with pytest.raises(ValueError, match="external ports"):
        prepare_born_color_components(
            (direct, crossed.rename_indices({"leg0": "other"}))
        )


def test_component_ids_ignore_symbol_registration_order():
    # The symbol registry is process-global, so reversing construction in this
    # pytest process cannot test registration independence. Adjoint metrics keep
    # this ordering control separate from the open-fundamental duality control.
    script = """
import json, sys
from symbolica import S
from symbolica.community import tensor as t
from pyamplicol.loops.color_components import prepare_born_color_components
names = ['anchor_order_a', 'anchor_order_z']
for name in names[::int(sys.argv[1])]:
    S(name)
a, z = map(S, names)
r = t.Representation.coad(8)
g = t.TensorExpression.g(r, r)
metric = g('leg0', 'leg1') * g('leg2', 'leg3')
tensors = (metric * (a + 2*z), metric * (2*a + z), metric * (a + 2*z))
print(json.dumps(prepare_born_color_components(tensors).as_dict()))
"""
    results = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script, direction],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        for direction in ("1", "-1")
    ]
    assert results[0] == results[1]
    assert {tuple(group) for group in results[0]["members"]} == {(0, 2), (1,)}


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 70)],
)
def test_signed_coherent_born_components_reconstruct_hermitian_observable(
    loop_native, tensors, arithmetic, precision
):
    native = loop_native
    from symbolica import E

    direct, crossed = tensors
    # Explicit fixture partition keeps this numerical boundary independent of
    # the dependency canonicalization control above.
    components = BornColorComponents(("direct", "crossed"), ((0, 2), (1,)))
    program = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    program.backend = native
    program.options = {"arithmetic": arithmetic, "precision": precision}
    program._evaluator_states = {}
    program.helicities, program.born_colors = ((),), (direct, crossed, direct)
    order = (("QCD", 2),)
    program.born_term_orders = (order,) * 3
    program.order_layout = PhysicalOrderLayout(
        (BornSquaredSector((("QCD", 4),), ((order, order),)),), (), ()
    )
    # Independent fundamental-flow contraction: each closed index line gives 3.
    # Direct/direct has two lines (9); direct/crossed has one (3).
    metric = tuple(
        tuple(E(str(value)) for value in row)
        for row in ((9, 3, 9), (3, 9, 3), (9, 3, 9))
    )
    projected = program._make_born_projector(
        components.mask_bra(metric), real_result=True
    )
    inclusive = program._make_born_projector((metric,), real_result=True)
    inputs = native.NativeBatch.from_decimal(
        [("2", "0"), ("-4", "0"), ("-1", "0")], 1, 3, row_ids=[71], **program.options
    )
    result = projected.evaluate_batch(inputs)
    values = [
        Decimal(real)
        for real, imaginary in result.to_decimal()
        if Decimal(imaginary) == 0
    ]
    assert sorted(values) == [Decimal(-3), Decimal(132)]
    assert (
        sum(values)
        == Decimal(inclusive.evaluate_batch(inputs).to_decimal()[0][0])
        == 129
    )
    assert result.row_ids == [71]


def test_counterterm_anchors_keep_uv_rational_fields_and_poles(loop_native, tensors):
    native = loop_native
    from symbolica import E

    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram

    direct, crossed = tensors
    components = BornColorComponents(("direct", "crossed"), ((0, 2), (1,)))
    born_order, loop_order = (("QCD", 2),), (("QCD", 4),)
    bare = SimpleNamespace(
        born_colors=(direct, crossed, direct),
        born_term_orders=(born_order,) * 3,
        helicities=((),),
        backend=native,
        options={"arithmetic": "double"},
    )
    program = NativeModelVirtualProgram.__new__(NativeModelVirtualProgram)
    program.bare, program._evaluator_states = bare, {}
    program.sector_pairs = ({(born_order, loop_order)},)

    def current(kind, epsilon, weights):
        return SimpleNamespace(
            term_count=1,
            insertion=SimpleNamespace(
                rule=SimpleNamespace(kind=kind), graph=SimpleNamespace(order=loop_order)
            ),
            epsilon_power=epsilon,
            color_weights=tuple((E(str(value)),) for value in weights),
        )

    program.currents = (current("UV", -1, (9, 3, 9)), current("R2", 0, (3, 9, 3)))
    program.field_currents = (
        SimpleNamespace(
            insertion=SimpleNamespace(order=loop_order),
            epsilon_power=-2,
            born_indices=(0, 2),
            color_weights=tuple(
                tuple(E(str(value)) for value in row)
                for row in ((9, 9), (3, 3), (9, 9))
            ),
        ),
    )
    inclusive = program._make_projector()
    resolved = program._make_projector(color_components=components)
    # Born (2,-4,-1), one UV amplitude=2, R2 amplitude=3, field factor=5.
    inputs = native.NativeBatch.from_decimal(
        [(str(value), "0") for value in (2, -4, -1, 2, 3, 5)], 1, 6, row_ids=[91]
    )
    whole = [
        Decimal(real)
        for real, imaginary in inclusive.evaluate_batch(inputs).to_decimal()
    ]
    rows = resolved.evaluate_batch(inputs).to_decimal()
    grouped = [
        [Decimal(rows[9 * component + column][0]) for column in range(9)]
        for component in range(2)
    ]
    assert whole == [0, -12, 0, -198, 0, 0, 0, 0, -30]
    assert [sum(row[column] for row in grouped) for column in range(9)] == whole
    # Exact pole positions and contributions distinguish a transpose/flatten bug.
    by_members = dict(zip(components.members, grouped, strict=True))
    assert by_members[(0, 2)] == [0, 36, 0, 18, 0, 0, 0, 0, 90]
    assert by_members[(1,)] == [0, -48, 0, -216, 0, 0, 0, 0, -120]


def test_weighted_numerator_anchors_stay_complex_before_integration(loop_native):
    native = loop_native
    from symbolica import E

    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector

    components = BornColorComponents(
        ("direct", "crossed", "zero"), ((0, 2), (1,), (3,))
    )
    program = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    program.backend, program.options = native, {"arithmetic": "double"}
    program.projector_engine = "interpreted"
    program._evaluator_states = {}
    program.helicities, program.born_colors = ((),), (None,) * 4
    born_order, loop_order = (("QCD", 2),), (("QCD", 4),)
    program.born_term_orders = (born_order,) * 4
    program.order_layout = PhysicalOrderLayout(
        (),
        (VirtualOrderSector((("QCD", 6),), (BornLoopPair(born_order, loop_order),)),),
        ("QCD",),
    )
    program.loop_graphs = (SimpleNamespace(order=loop_order),)
    program.loop_term_offsets = (0,)
    power = (0, 0, 0, 0, 0, 0)
    program.loop_reduction_powers = (((power,),),)
    metric = tuple((E(str(value)),) for value in (9, 3, 9, 0))
    projector = program._compile_family_projection(
        (0,), (power,), (0,), metrics=components.mask_bra(metric)
    )
    batch = native.NativeBatch.from_decimal(
        [("2", "0"), ("-4", "0"), ("-1", "0"), ("7", "0"), ("0", "1")],
        1,
        5,
        row_ids=[91],
    )
    result = projector.evaluate_batch(batch)
    assert [(Decimal(a), Decimal(b)) for a, b in result.to_decimal()] == [
        (0, 9),
        (0, -12),
        (0, 0),
    ]
    # The zero ordinary overlap remains an explicit anchor; an inserted colour
    # operator may make that direction nonzero in a future observable plan.
    assert result.columns == 3


def test_failed_dependency_canonicalization_has_actionable_capability_error(
    monkeypatch,
):
    import symbolica.community.tensor as tensor

    class UnsupportedTensor:
        axes = ()

        def canonize(self):
            raise ValueError("typed external duality changed")

    monkeypatch.setattr(tensor, "TensorExpression", UnsupportedTensor)
    with pytest.raises(
        NotImplementedError, match="compatible dependency implementation"
    ) as failure:
        prepare_born_color_components((UnsupportedTensor(),))
    assert isinstance(failure.value.__cause__, ValueError)
    assert "typed external duality" in str(failure.value.__cause__)
