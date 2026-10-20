# SPDX-License-Identifier: 0BSD
"""Actual projected identity sources preserve all complete native components."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_native_directed import _blocks, _values
from .test_loop_physical_correlations import _close
from .test_loop_portable import _rewrite

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def projected_spin(saved_loop_fixture):
    root = str(saved_loop_fixture("projected-spin-v43ff063"))
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return Path(root), _rusticol._loop


def _point():
    return [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "300", "0", "400"],
        ["500", "-300", "0", "-400"],
    ]


def _basis():
    return (
        [["0", "0"], ["0.8", "0"], ["0", "0"], ["-0.6", "0"]],
        [["0", "0"], ["0", "0"], ["1", "0"], ["0", "0"]],
    )


def _scaled(vector, real, imaginary):
    return [
        [
            str(real * Decimal(a) - imaginary * Decimal(b)),
            str(real * Decimal(b) + imaginary * Decimal(a)),
        ]
        for a, b in vector
    ]


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-9", 20),
        ("double-double", None, "generic", "8e-23", 70),
        ("arbitrary", 70, "generic", "8e-55", 170),
    ],
)
def test_projected_complete_components_sources_and_exact_carriers(
    projected_spin, accuracy, arithmetic, precision, engine, tolerance, bit
):
    root, native = projected_spin
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    runtime = native.NativeLoopProgram(
        str(root / f"gg-{accuracy}-spin.pacbin"), **options
    )
    ordinary_path = (
        root.parent / "dual-born-weights/gg-lc-weighted.pacbin"
        if accuracy == "lc"
        else root / "gg-nlc-ordinary.pacbin"
    )
    ordinary = native.NativeLoopProgram(str(ordinary_path), **options)
    data = json.loads(runtime.metadata_json())
    assert data["color_projection"]["accuracy"] == accuracy
    assert data.get("born_color_components") is None
    assert data.get("helicity_projection") is None
    assert (
        data["directed_projection"]["convention"]
        == "sun-graded-identity-all-outgoing-v1"
    )
    assert [x["legs"] for x in data["spin_projections"]] == [[3], [3, 4]]
    p, (a, b) = _point(), _basis()
    with localcontext() as context:
        context.prec = 120
        tiny = Decimal(2) ** -bit
        changes = {"aS": [str(Decimal("0.118") * (1 + tiny)), "0"], "MT": ["173", "0"]}
        runtime.source_state.update(changes)
        ordinary.source_state.update(changes)
        snapshot = runtime.source_state.snapshot()
        baseline = runtime.evaluate_directed([p], ["1000000"], point_ids=[91])
        expected = ordinary.evaluate([p], ["1000000"], point_ids=[91])
        for left, right in zip(
            (baseline.born, *baseline.component_batches),
            (expected.born, *expected.component_batches),
            strict=True,
        ):
            assert all(ok for _, ok in left.agreement_rows(right, tolerance, tolerance))
        ids = [2**64 - 1, 7, 42, 0, 2**63]
        vectors = [a, b, _scaled(a, 2, -3), [["0", "0"]] * 4, _scaled(a, 1 + tiny, 0)]
        singleton = runtime.evaluate_directed(
            [p] * 5, ["1000000"] * 5, point_ids=ids, spin_vectors={3: vectors}
        )
        joint = runtime.evaluate_directed(
            [p] * 4,
            ["1000000"] * 4,
            spin_vectors={3: [a, a, b, b], 4: [a, b, a, b]},
        )
        assert singleton.point_ids == ids and singleton.spin_legs == [3]
        assert joint.spin_legs == [3, 4]
        for one, two, full in zip(
            _blocks(singleton), _blocks(joint), _blocks(baseline), strict=True
        ):
            values = _values(one, (5, one.columns))
            target = _values(full, (1, full.columns))[0]
            _close(values[0] + values[1], target, Decimal(tolerance))
            _close(values[2], 13 * values[0], Decimal(tolerance))
            _close(values[3], 0 * values[0], Decimal(tolerance))
            _close(
                _values(two, (4, two.columns)).sum(axis=0), target, Decimal(tolerance)
            )
        born = _values(singleton.born, (5, singleton.born.columns))
        actual_shift = born[4, 0, 0] - born[0, 0, 0]
        expected_shift = born[0, 0, 0] * (2 * tiny + tiny * tiny)
        assert abs(actual_shift - expected_shift) < abs(expected_shift) * Decimal(
            "2e-4"
        )
        assert runtime.source_state.snapshot().overrides == snapshot.overrides
        empty = runtime.evaluate_directed(
            [], [], point_ids=[], spin_vectors={3: [], 4: []}
        )
        assert empty.point_ids == [] and empty.spin_legs == [3, 4]
        assert all(block.rows == 0 for block in _blocks(empty))
        again = runtime.evaluate_directed(
            [p], ["1000000"], point_ids=[42], spin_vectors={3: [a]}
        )
        for left, right in zip(_blocks(again), _blocks(singleton), strict=True):
            _close(
                _values(left, (1, left.columns))[0],
                _values(right, (5, right.columns))[0],
                Decimal(tolerance),
            )


def test_projected_native_descriptor_rejects_unproved_operator_and_partition(
    projected_spin, tmp_path
):
    root, native = projected_spin
    source = root / "gg-lc-spin.pacbin"
    for index, mutate in enumerate(
        (
            lambda d: d["directed_projection"].update(
                convention="su3-literal-tau-physical-charge-all-outgoing-v1"
            ),
            lambda d: d.update(color_projection=None),
            lambda d: d["directed_projection"]["histories"][0].update(
                bra=[{"emitter": 0, "gluon": -1}]
            ),
            lambda d: d["spin_projections"][0]["directed"].update(
                convention="su3-literal-tau-physical-charge-all-outgoing-v1"
            ),
            lambda d: d.update(
                born_color_components={
                    "kind": "canonical-born-tensor-partition",
                    "ids": ["b0"],
                    "members": [[0]],
                }
            ),
        )
    ):
        path = tmp_path / f"bad-{index}.pacbin"
        _rewrite(source, path, mutate)
        with pytest.raises(ValueError):
            native.PortableLoopPlan(str(path))


def test_public_projected_identity_shape_and_subset_rescue(projected_spin):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
    from pyamplicol.api.errors import EvaluationError

    root, _ = projected_spin
    runtime = Runtime.load(root / "lc-artifact")
    assert runtime.physics.color_ids == ("lc",)
    assert not runtime.physics.color_flows
    assert not runtime.physics.born_anchored_color_components
    assert runtime.physics.reduction.kind == "contracted-color"
    info = runtime.inspect()
    assert info["resolved_color_components"] == []
    assert info["resolved_color_semantics"] == "one contracted colour component"
    assert info["spin_correlations"] == ((3,), (3, 4))
    a, b = _basis()
    request = {"joint": CorrelatedRequest("born", {3: [a, b, a], 4: b})}
    points, scales = [_point()] * 3, ["1000000", "1e-400", "1000000"]
    with pytest.raises(EvaluationError) as failure:
        runtime.evaluate_correlated_many(points, request, mu_r2=scales)
    assert {item.point_id for item in failure.value.failures} == {1}
    rescued = runtime.evaluate_correlated_many(
        points,
        request,
        mu_r2=scales,
        stability=VirtualStabilityPolicy(
            (
                VirtualArithmeticSpec("arbitrary", 60),
                VirtualArithmeticSpec("arbitrary", 80),
            ),
            "1e-8",
            "1e-9",
        ),
    )["joint"]
    assert [row.status for row in rescued.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert rescued.values[0] == rescued.values[2]
    runtime.set_spin_correlation_vectors({3: a, 4: b})
    before = runtime.evaluate_correlated(points[:1], mu_r2=scales[:1])
    runtime.clear()
    assert (
        runtime.evaluate_correlated(points[:1], mu_r2=scales[:1]).values
        == before.values
    )
    assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)
