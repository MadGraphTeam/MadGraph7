# SPDX-License-Identifier: 0BSD
"""Physical multi-order controls with independent Born and infrared poles."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.orders import BornLoopPair, compile_order_plan
from pyamplicol.loops.physical_process import NativeBareProcessProgram


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires a licensed Symbolica runtime")
    return pytest.importorskip("_loop_reduction")


@pytest.fixture(scope="module")
def setup(native):
    from symbolica import E, S

    from pyamplicol.loops.control_models import qcd_qed_control_model

    model = qcd_qed_control_model()
    process = build_loop_process(
        model.process(["u", "u~"], ["d", "d~"]),
        selected_corrections=["QCD", "QED"],
        perturbative_orders={"QCD": 1, "QED": 1},
    )
    alpha, inverse_alpha, top = S(
        "mixed_order::alpha_s", "mixed_order::aEWM1", "mixed_order::mt"
    )
    payload = json.loads(model.to_json())
    expressions = {f"UFO::{name}": E("0") for name in ("MB", "MC", "Me", "MM", "MTA")}
    expressions["UFO::MT"] = top
    for record in payload["couplings"]:
        coupling = model.coupling(record["name"])
        expressions[coupling.symbol.get_name()] = coupling.expression.replace(
            S("UFO::G"), (4 * E("pi") * alpha) ** E("1/2")
        ).replace(S("UFO::ee"), (4 * E("pi") / inverse_alpha) ** E("1/2"))
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(path.read_text())["points"], dtype=str)
    return process, expressions, (alpha, inverse_alpha, top), points


def _program(setup, native, *, process=None, **options):
    from symbolica import E

    catalogue, expressions, inputs, _ = setup
    return NativeBareProcessProgram(
        catalogue if process is None else process,
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=inputs,
        backend=native,
        **options,
    )


def _order(qcd, qed):
    return tuple((name, power) for name, power in (("QCD", qcd), ("QED", qed)) if power)


def _targets(point):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(x) for x in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((x * x for x in vector[1:]), E("0"))

    s = square([a + b for a, b in zip(p[0], p[1], strict=True)])
    t = square([a - b for a, b in zip(p[0], p[2], strict=True)])
    u = square([a - b for a, b in zip(p[0], p[3], strict=True)])
    alpha, qed = E("59/500"), E("1000/132507")
    strong_born = E("4/9") * (4 * E("pi") * alpha) ** 2 * (t * t + u * u) / s**2
    photon_born = E("8/81") * (4 * E("pi") * qed) ** 2 * (t * t + u * u) / s**2
    qcd_pole, qed_pole = (
        -E("16/3") * alpha / (2 * E("pi")),
        -E("10/9") * qed / (2 * E("pi")),
    )
    expressions = [
        strong_born,
        photon_born,
        qcd_pole * strong_born,
        qed_pole * strong_born,
        qcd_pole * photon_born,
        qed_pole * photon_born,
    ]
    values = Expression.evaluator_multiple(
        expressions, [], n_cores=1, jit_compile=False
    ).evaluate_complex_with_prec([], 110)
    return [value.to_decimal_tuple()[0] for value in values]


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("3e-11")),
        ("double-double", None, "generic", Decimal("3e-25")),
        ("arbitrary", 60, "generic", Decimal("3e-49")),
    ],
)
def test_order_axes_born_and_universal_double_poles(
    setup, native, arithmetic, precision, engine, tolerance
):
    program = _program(
        setup,
        native,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    assert len(program.loop_graphs) == 35
    assert (
        sum(
            not graph.eligibility.admits(frozenset({"QCD"}))
            and not graph.eligibility.admits(frozenset({"QED"}))
            for graph in program.loop_graphs
        )
        == 4
    )
    points = setup[3]
    result = program.evaluate_ordered_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "132.507", "173"]
    )
    assert result.born.shape == (3, 3, 2)
    assert result.virtual.shape == (3, 4, 3, 2)
    assert result.epsilon_powers == (0, -1, -2)
    assert result.orders.correction_names == ("QCD", "QED")
    assert set(result.born_orders) == {_order(4, 0), _order(2, 2), _order(0, 4)}
    assert set(result.virtual_orders) == {
        _order(6, 0),
        _order(4, 2),
        _order(2, 4),
        _order(0, 6),
    }
    assert all(Decimal(x) == 0 for x in result.born_order(_order(2, 2)).flat)
    with localcontext() as context:
        context.prec = 120
        for i, point in enumerate(points):
            targets = _targets(point)
            actual = [
                result.born_order(_order(4, 0))[i, 0],
                result.born_order(_order(0, 4))[i, 0],
            ]
            actual.extend(
                result.virtual_order(order)[i, 2, 0]
                for order in (_order(6, 0), _order(4, 2), _order(2, 4), _order(0, 6))
            )
            for value, target in zip(actual, targets, strict=True):
                assert abs(Decimal(value) - target) < abs(target) * tolerance
    assert all(status == "success" for row in result.statuses for _, status in row)
    assert len(result.channels) == len(result.statuses[0])
    assert all(point == i for i, row in enumerate(result.statuses) for point, _ in row)
    assert not result.born.flags.writeable and not result.virtual.flags.writeable
    rebound = program.evaluate_ordered_decimal(
        points[1:2], ["1000000"], model_values=["0.472", "14.723", "173"]
    )
    with localcontext() as context:
        context.prec = 120
        for orders, original, changed in (
            (result.born_orders, result.born, rebound.born),
            (result.virtual_orders, result.virtual, rebound.virtual),
        ):
            for i, order in enumerate(orders):
                powers = dict(order)
                factor = 2 ** powers.get("QCD", 0) * 3 ** powers.get("QED", 0)
                # A cancelled component can acquire working-precision roundoff
                # when the rescaled evaluator reassociates sums. Use the sector
                # norm for those zeros; the structural Born zero is exact above.
                norm = max(
                    abs(Decimal(value) * factor) for value in original[1, i].flat
                )
                for old, new in zip(
                    original[1, i].flat, changed[0, i].flat, strict=True
                ):
                    expected = Decimal(old) * factor
                    if norm == 0:
                        assert Decimal(new) == 0
                    else:
                        assert abs(Decimal(new) - expected) < norm * tolerance
        for values, summed in (
            (result.born, result.sum_orders().born),
            (result.virtual, result.sum_orders().virtual),
        ):
            for row, total in zip(values, summed, strict=True):
                for column in range(total.size):
                    expected = sum(Decimal(order.reshape(-1)[column]) for order in row)
                    actual = Decimal(total.reshape(-1)[column])
                    assert (
                        abs(actual - expected)
                        < max(abs(expected), Decimal("1e-40")) * tolerance
                    )
    with pytest.raises(ValueError, match="absent"):
        result.virtual_order(_order(8, 0))
    with pytest.raises(NotImplementedError, match="ordered bare"):
        program.evaluate_decimal(
            points[:1], ["1000000"], model_values=["0.118", "132.507", "173"]
        )
    empty = program.evaluate_ordered_decimal(
        points[:0], [], model_values=["0.118", "132.507", "173"]
    )
    assert empty.born.shape == (0, 3, 2) and empty.virtual.shape == (0, 4, 3, 2)
    assert empty.statuses == ()
    assert empty.sum_orders().virtual.shape == (0, 3, 2)
    reference_path = os.environ.get("PYAMPLICOL_MIXED_UUDD_BARE_REFERENCE")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        assert reference["native_ct_amplitudes"] == 0
        for index, order in reference["born_orders"].items():
            np.testing.assert_allclose(
                result.born_order(_order(order["QCD"], order["QED"]))[:, 0].astype(
                    float
                ),
                [row["born"][index] for row in reference["rows"]],
                rtol=3e-12,
                atol=1e-15,
            )
        for index, order in reference["loop_orders"].items():
            np.testing.assert_allclose(
                result.virtual_order(_order(order["QCD"], order["QED"]))[
                    :, :, 0
                ].astype(float),
                [
                    [
                        row["bare"][index][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ]
                    for row in reference["rows"]
                ],
                rtol=3e-11,
                atol=1e-15,
            )
        np.testing.assert_allclose(
            result.sum_orders().virtual[:, :, 0].astype(float),
            [row["total"][1:] for row in reference["rows"]],
            rtol=3e-11,
            atol=1e-15,
        )


def test_explicit_pair_masks_include_both_mixed_interferences(setup, native):
    catalogue, _, _, points = setup
    options = dict(reduction_engine="fast", coefficient_engine="jit")
    full = _program(setup, native, **options).evaluate_ordered_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "132.507", "173"]
    )
    contributions = []
    reference_path = os.environ.get("PYAMPLICOL_MIXED_UUDD_BARE_REFERENCE")
    reference = (
        None if reference_path is None else json.loads(Path(reference_path).read_text())
    )
    for pair in (
        BornLoopPair(_order(2, 0), _order(2, 2)),
        BornLoopPair(_order(0, 2), _order(4, 0)),
    ):
        plan = compile_order_plan(
            selected_corrections=catalogue.orders.correction_names,
            perturbative_orders={"QCD": 1, "QED": 1},
            born_orders=catalogue.orders.born_orders,
            contributions=catalogue.orders.contributions,
            allowed_pairs=[pair],
        )
        restricted = _program(
            setup, native, process=replace(catalogue, orders=plan), **options
        )
        value = (
            restricted.evaluate_ordered_decimal(
                points, ["1000000"] * 3, model_values=["0.118", "132.507", "173"]
            )
            .virtual_order(_order(4, 2))
            .astype(float)
        )
        contributions.append(value)
        if reference is not None:
            source = "B20" if pair.born == _order(2, 0) else "B02"
            index = next(
                index
                for index, order in reference["loop_orders"].items()
                if _order(order["QCD"], order["QED"]) == pair.squared
            )
            np.testing.assert_allclose(
                value[:, :, 0],
                [
                    [
                        row["by_born_order"][source][index][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ]
                    for row in reference["rows"]
                ],
                rtol=3e-11,
                atol=1e-15,
            )
    assert all(np.max(np.abs(value)) > 1e-8 for value in contributions)
    np.testing.assert_allclose(
        sum(contributions),
        full.virtual_order(_order(4, 2)).astype(float),
        rtol=3e-11,
        atol=1e-15,
    )


def test_unextended_completions_reject_multiple_orders_upfront(setup, native):
    from symbolica import S

    from pyamplicol.loops.physical_color import project_physical_color
    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram

    program = _program(setup, native, reduction_engine="fast")
    with pytest.raises(NotImplementedError, match="formal or direct model"):
        NativeModelVirtualProgram(
            program, (), source_bindings=(), formal_nc=S("order_test::Nc")
        )
    with pytest.raises(NotImplementedError, match="formal physical color"):
        project_physical_color(program, S("order_test::Nc"), [], "full")
    with pytest.raises(NotImplementedError, match="multi-order dimensional"):
        _program(setup, native, dimensional_numerators=True)
    with pytest.raises(ArithmeticError, match="point, order, family, status"):
        program.evaluate_ordered_decimal(
            setup[3][:1], ["-1"], model_values=["0.118", "132.507", "173"]
        )


@pytest.mark.parametrize("correction,count", [("QCD", 16), ("QED", 15)])
def test_reselected_plan_preserves_graph_eligibility_not_only_absolute_order(
    setup, native, correction, count
):
    catalogue = setup[0]
    contributions = tuple(
        item
        for item in catalogue.orders.contributions
        if item.eligibility.admits(frozenset({correction}))
    )
    loop_orders = {item.order for item in contributions}
    pairs = tuple(
        pair
        for sector in catalogue.orders.sectors
        for pair in sector.pairs
        if pair.loop in loop_orders
    )
    plan = compile_order_plan(
        selected_corrections=[correction],
        perturbative_orders={"QCD": 1, "QED": 1},
        born_orders=catalogue.orders.born_orders,
        contributions=catalogue.orders.contributions,
        allowed_pairs=pairs,
    )
    program = _program(
        setup, native, process=replace(catalogue, orders=plan), reduction_engine="fast"
    )
    assert len(program.loop_graphs) == count
    assert {graph.id for graph in program.loop_graphs} == {
        item.id for item in contributions
    }
    assert sum(graph.order == _order(2, 2) for graph in program.loop_graphs) == 2
