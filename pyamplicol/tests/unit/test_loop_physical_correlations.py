# SPDX-License-Identifier: 0BSD
"""Directed physical bare controls with complex absorptive cross terms."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.color.connections import ColorConnection, EmitGluon
from pyamplicol.loops.kernels import NativeCoefficientDAG
from pyamplicol.loops.physical_correlations import (
    NativeDirectedBareProgram,
    directed_color_legs,
)
from pyamplicol.loops.physical_process import distinct_flavour_qcd_program

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def native(loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency generation required")
    return loop_native


@pytest.fixture(scope="module")
def points():
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


@pytest.fixture(scope="module")
def model_source(loop_sm_ufo, native):
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    path = str(loop_sm_ufo)
    return build_loop_model(load_raw_loop_model(path))


@pytest.mark.parametrize("massive", [False, True])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("3e-11")),
        ("double-double", None, "generic", Decimal("3e-24")),
        ("arbitrary", 60, "generic", Decimal("3e-47")),
    ],
)
def test_complete_model_components_share_currents_and_keep_complex_histories(
    model_source, native, monkeypatch, massive, arithmetic, precision, engine, tolerance
):
    from pyamplicol.loops.physical_counterterms import (
        distinct_flavour_model_virtual_program,
        top_pair_model_virtual_program,
    )
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram

    factory = (
        top_pair_model_virtual_program
        if massive
        else distinct_flavour_model_virtual_program
    )
    name = "top_pair_qcd.json" if massive else "uudd_qcd.json"
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference" / name
    momenta = np.asarray(json.loads(path.read_text())["points"], dtype=str)[:2]
    model = factory(
        model_source,
        backend=native,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    baseline = model.evaluate_ordered_decimal(
        momenta, ["1000000", "900000"], model_values=["0.118", "173"]
    )
    directed = NativeDirectedModelProgram(
        model, _histories(model.bare, conservation=True)
    )
    calls = Counter()
    evaluate = NativeCoefficientDAG.evaluate_batch

    def counted(executor, arguments):
        calls[id(executor)] += 1
        return evaluate(executor, arguments)

    monkeypatch.setattr(NativeCoefficientDAG, "evaluate_batch", counted)
    result = directed.evaluate_decimal(
        momenta, ["1000000", "900000"], model_values=["0.118", "173"]
    )
    expected = Counter(id(current.executor) for current in model.currents)
    for executors, sharing in (
        (model.bare.born_executors, model.bare.born_sharing),
        (model.bare.loop_executors, model.bare.loop_sharing),
    ):
        expected.update(id(executors[index]) for index in sharing.representatives)
    assert calls == expected  # CT work also occurs once, independent of directions.
    with localcontext() as context:
        context.prec = 110
        for name in (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
        ):
            values = _decimals(getattr(result, name))
            _close(values[:, 0], _decimals(getattr(baseline, name)), tolerance)
            _close(values[:, 1], values[:, 0] * Decimal(4) / 3, tolerance)
            _close(
                sum(values[:, i] for i in (1, 4, 5, 6)),
                np.zeros_like(values[:, 0]),
                tolerance,
            )
            _close(values[:, 3], _conjugate(values[:, 2]), tolerance)
            assert not getattr(result, name).flags.writeable
        _close(
            _decimals(result.virtual),
            _decimals(result.first) + _decimals(result.second),
            tolerance,
        )
        _close(
            _decimals(result.first[:, 2]),
            _conjugate(_decimals(result.second[:, 3])),
            tolerance,
        )
        _close(
            _decimals(result.virtual),
            _decimals(result.bare) + _decimals(result.uv) + _decimals(result.r2),
            tolerance,
        )
        assert abs(Decimal(result.virtual[0, 2, 0, 0, 1])) > Decimal("1e-4")
        if massive:
            assert abs(Decimal(result.uv_external_fields[0, 0, 0, 0, 0])) > Decimal(
                "1e-3"
            )
    for point, row in enumerate(result.statuses):
        for (lane, status), _channel in zip(row, result.channels, strict=True):
            assert lane == point
            assert status == "success"
    empty = directed.evaluate_decimal(momenta[:0], [], model_values=["0.118", "173"])
    assert empty.virtual.shape == (0, 7, 1, 3, 2) and empty.statuses == ()
    assert empty.uv_external_fields.shape == empty.virtual.shape


def test_complete_directed_rejects_singular_counterterms_and_scope_mutation(
    model_source, points
):
    from dataclasses import replace

    from symbolica import E

    from pyamplicol.loops.model_expressions import LaurentExpression
    from pyamplicol.loops.physical_counterterms import (
        NativeModelVirtualProgram,
        distinct_flavour_model_virtual_program,
    )
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram

    base = distinct_flavour_model_virtual_program(model_source, reduction_engine="fast")
    original = base.insertions[0]
    singular = replace(
        original,
        rule=replace(
            original.rule,
            couplings=tuple(
                (
                    name,
                    replace(
                        entry,
                        coupling=replace(
                            entry.coupling,
                            value=LaurentExpression(
                                entry.coupling.value.expression
                                / (base.bare.scale_squared_symbol - E("4000000"))
                            ),
                        ),
                    ),
                )
                for name, entry in original.rule.couplings
            ),
        ),
    )
    # This deliberately incomplete catalogue tests failure reporting only.
    model = NativeModelVirtualProgram(
        base.bare, [singular], source_bindings=base.source_bindings
    )
    directed = NativeDirectedModelProgram(model, _histories(base.bare)[:2])
    with pytest.raises(ArithmeticError, match=r"counterterm scalar.*\[\(1, 1\)\]"):
        directed.evaluate_native(
            points[:2], ["1000000", "4000000"], model_values=["0.118", "173"]
        )
    with pytest.raises(ValueError, match="bound to a model program"):
        base.bare.set_color_metrics(base.bare.born_metric, base.bare.loop_metric)
    model.r2_strategy = "direct"
    with pytest.raises(NotImplementedError, match="model R2"):
        directed.evaluate_native(points[:0], [], model_values=["0.118", "173"])
    model.r2_strategy = "model"
    model.currents.pop()
    with pytest.raises(ValueError, match="catalogue changed"):
        directed.evaluate_native(points[:0], [], model_values=["0.118", "173"])


def _histories(bare, *, conservation=False):
    legs = directed_color_legs(bare)
    identity = ColorConnection(legs)
    charged = [ColorConnection(legs, (EmitGluon(leg.label, -1),)) for leg in legs]
    alpha = ColorConnection(legs, (EmitGluon(0, -1), EmitGluon(0, -2)))
    beta = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(2, -2)))
    result = [
        (identity, identity),
        (charged[0], charged[0]),
        (alpha, beta),
        (beta, alpha),
    ]
    if conservation:
        result.extend((charged[0], other) for other in charged[1:])
    return tuple(result)


def _decimals(values):
    return np.asarray(
        [Decimal(str(value)) for value in values.flat], dtype=object
    ).reshape(values.shape)


def _close(left, right, tolerance):
    left, right = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    assert left.shape == right.shape
    assert all(
        abs(a - b) < tolerance * max(Decimal(1), abs(b))
        for a, b in zip(left.flat, right.flat, strict=True)
    )


def _conjugate(values):
    result = values.copy()
    result[..., 1] *= -1
    return result


def _close_component(actual, expected, tolerance, *, scale=None):
    actual, expected = (
        np.asarray(actual, dtype=object),
        np.asarray(expected, dtype=object),
    )
    assert actual.shape == expected.shape
    norm = max(
        Decimal("1e-100"),
        *(abs(value) for value in (expected if scale is None else scale).flat),
    )
    assert all(
        abs(a - b) <= tolerance * norm
        for a, b in zip(actual.flat, expected.flat, strict=True)
    )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("3e-11")),
        ("double-double", None, "generic", Decimal("3e-24")),
        ("arbitrary", 60, "generic", Decimal("3e-47")),
    ],
)
def test_shared_currents_identity_casimir_conservation_and_complex_directions(
    native,
    points,
    monkeypatch,
    arithmetic,
    precision,
    engine,
    tolerance,
):
    bare = distinct_flavour_qcd_program(
        backend=native,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    baseline = bare.evaluate_ordered_decimal(
        points[:2],
        ["1000000", "900000"],
        model_values=["0.118", "173"],
    )
    directed = NativeDirectedBareProgram(bare, _histories(bare, conservation=True))
    # Ports are the actual native amplitude interfaces, including crossing.
    assert [(leg.label, leg.representation) for leg in directed.color_legs] == [
        (0, 3),
        (1, -3),
        (2, -3),
        (3, 3),
    ]
    calls, kinematics = Counter(), Counter()
    evaluate = NativeCoefficientDAG.evaluate_batch
    family_bindings = bare.family_kinematics_native

    def counted(executor, arguments):
        calls[id(executor)] += 1
        return evaluate(executor, arguments)

    def counted_bindings(currents, family):
        kinematics[family.index] += 1
        return family_bindings(currents, family)

    monkeypatch.setattr(NativeCoefficientDAG, "evaluate_batch", counted)
    monkeypatch.setattr(bare, "family_kinematics_native", counted_bindings)
    result = directed.evaluate_decimal(
        points[:2],
        ["1000000", "900000"],
        model_values=["0.118", "173"],
    )
    expected = Counter(
        id(executors[index])
        for executors, sharing in (
            (bare.born_executors, bare.born_sharing),
            (bare.loop_executors, bare.loop_sharing),
        )
        for index in sharing.representatives
    )
    assert calls == expected
    assert kinematics == Counter({family.index: 1 for family in bare.families})
    with localcontext() as context:
        context.prec = 110
        born, virtual, first, second = map(
            _decimals,
            (
                result.born,
                result.virtual,
                result.first,
                result.second,
            ),
        )
        _close(born[:, 0], _decimals(baseline.born), tolerance)
        _close(virtual[:, 0], _decimals(baseline.virtual), tolerance)
        _close(born[:, 1], born[:, 0] * Decimal(4) / 3, tolerance)
        _close(virtual[:, 1], virtual[:, 0] * Decimal(4) / 3, tolerance)
        _close(
            sum(born[:, i] for i in (1, 4, 5, 6)), np.zeros_like(born[:, 0]), tolerance
        )
        _close(
            sum(virtual[:, i] for i in (1, 4, 5, 6)),
            np.zeros_like(virtual[:, 0]),
            tolerance,
        )
        _close(born[:, 3], _conjugate(born[:, 2]), tolerance)
        _close(virtual[:, 3], _conjugate(virtual[:, 2]), tolerance)
        _close(first[:, 2], _conjugate(second[:, 3]), tolerance)
        _close(virtual, first + second, tolerance)
        # An actual non-Hermitian colour history on an absorptive loop process.
        assert abs(virtual[0, 2, 0, 0, 1]) > Decimal("0.01")
        assert abs(first[0, 0, 0, 0, 1]) > Decimal("0.09")
        assert abs(virtual[0, 2, 0, 0, 0] - 2 * first[0, 2, 0, 0, 0]) > Decimal("0.01")
    assert not result.virtual.flags.writeable
    for point, row in enumerate(result.statuses):
        assert len(row) == len(result.channels)
        for (lane, status), _channel in zip(row, result.channels, strict=True):
            assert lane == point
            assert status == "success"
    rebound = directed.evaluate_decimal(
        points[1:2],
        ["900000"],
        model_values=["0.118", "173"],
    )
    with localcontext() as context:
        context.prec = 110
        _close(_decimals(rebound.virtual), _decimals(result.virtual[1:2]), tolerance)
    empty = directed.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.born.shape == (0, 7, 1, 2)
    assert empty.virtual.shape == (0, 7, 1, 3, 2) and empty.statuses == ()


def _mixed_program(native, **options):
    from symbolica import E, S

    from pyamplicol.loops.control_models import qcd_qed_control_model
    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    model = qcd_qed_control_model()
    process = build_loop_process(
        model.process(["u", "u~"], ["d", "d~"]),
        selected_corrections=["QCD", "QED"],
        perturbative_orders={"QCD": 1, "QED": 1},
    )
    alpha, inverse, top = S(
        "directed_test::alpha", "directed_test::inverse", "directed_test::top"
    )
    expressions = {f"UFO::{name}": E("0") for name in ("MB", "MC", "Me", "MM", "MTA")}
    expressions["UFO::MT"] = top
    for coupling in model.couplings:
        value = coupling.expression
        if value.contains(S("UFO::ee")):
            # Exact complex scalar-binding diagnostic, not an SM parameter claim.
            value *= E("1+1i")
        expressions[coupling.symbol.get_name()] = value.replace(
            S("UFO::G"), (4 * E("pi") * alpha) ** E("1/2")
        ).replace(S("UFO::ee"), (4 * E("pi") / inverse) ** E("1/2"))
    return NativeBareProcessProgram(
        process,
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=(alpha, inverse, top),
        backend=native,
        **options,
    )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", Decimal("3e-11")),
        ("double-double", None, "generic", Decimal("3e-24")),
        ("arbitrary", 60, "generic", Decimal("3e-47")),
    ],
)
def test_multiple_squared_orders_complex_born_and_lane_identity(
    native,
    points,
    arithmetic,
    precision,
    engine,
    tolerance,
):
    bare = _mixed_program(
        native, arithmetic=arithmetic, precision=precision, reduction_engine=engine
    )
    baseline = bare.evaluate_ordered_decimal(
        points[:2], ["1000000", "900000"], model_values=["0.118", "132.507", "173"]
    )
    directed = NativeDirectedBareProgram(bare, _histories(bare))
    result = directed.evaluate_decimal(
        points[:2], ["1000000", "900000"], model_values=["0.118", "132.507", "173"]
    )
    assert result.born.shape == (2, 4, 3, 2)
    assert result.virtual.shape == (2, 4, 4, 3, 2)
    with localcontext() as context:
        context.prec = 110
        born, virtual = _decimals(result.born), _decimals(result.virtual)
        _close(born[:, 0], _decimals(baseline.born), tolerance)
        _close(virtual[:, 0], _decimals(baseline.virtual), tolerance)
        _close(born[:, 3], _conjugate(born[:, 2]), tolerance)
        _close(virtual[:, 3], _conjugate(virtual[:, 2]), tolerance)
        mixed = _decimals(result.born_order((("QCD", 2), ("QED", 2))))
        assert abs(mixed[0, 2, 1]) > Decimal("1e-4")
    for point, row in enumerate(result.statuses):
        for (lane, status), _channel in zip(row, result.channels, strict=True):
            assert lane == point
            assert status == "success"
    with pytest.raises(ValueError, match="absent"):
        result.virtual_order((("QCD", 99),))


def test_tail_batches_and_scope_mutation_guards(native, points):
    bare = distinct_flavour_qcd_program(backend=native, reduction_engine="fast")
    directed = NativeDirectedBareProgram(bare, _histories(bare)[:1])
    batch = np.tile(points, (6, 1, 1))[:17]
    result = directed.evaluate_decimal(
        batch, ["1000000"] * 17, model_values=["0.118", "173"]
    )
    short = directed.evaluate_decimal(
        batch[:3], ["1000000"] * 3, model_values=["0.118", "173"]
    )
    _close(_decimals(result.virtual[:3]), _decimals(short.virtual), Decimal("3e-11"))
    assert result.statuses[-1][0][0] == 16
    with pytest.raises(ValueError, match="directed program"):
        bare.set_color_metrics(bare.born_metric, bare.loop_metric)
    bare._model_color_metrics_bound = True
    with pytest.raises(ValueError, match="model-bound"):
        directed.evaluate_native(points[:0], [], model_values=["0.118", "173"])
    with pytest.raises(ValueError, match="model-bound"):
        NativeDirectedBareProgram(bare, _histories(bare))
    bare._model_color_metrics_bound = False
    bare.loop_metric = tuple(tuple(2 * x for x in row) for row in bare.loop_metric)
    with pytest.raises(ValueError, match="changed"):
        directed.evaluate_native(points[:0], [], model_values=["0.118", "173"])
    fresh = distinct_flavour_qcd_program(backend=native, reduction_engine="fast")
    fresh.set_color_metrics(fresh.born_metric, fresh.loop_metric)
    with pytest.raises(NotImplementedError, match="unmodified"):
        NativeDirectedBareProgram(fresh, _histories(fresh))
