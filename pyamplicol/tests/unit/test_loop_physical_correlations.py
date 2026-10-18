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


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency generation required")
    return pytest.importorskip("_loop_reduction")


@pytest.fixture(scope="module")
def points():
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


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
        for (lane, status), (direction, sector, _family) in zip(
            row, result.channels, strict=True
        ):
            assert lane == point * directed.directions + direction + sector
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
    from symbolica.community import hepkit as hep

    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    model = hep.Model.qcd_qed()
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
        for (lane, status), (direction, sector, _family) in zip(
            row, result.channels, strict=True
        ):
            assert lane == point * directed.directions * 4 + direction * 4 + sector
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
    assert result.statuses[-1][0][0] == 32
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
