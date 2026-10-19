# SPDX-License-Identifier: 0BSD
"""Saved complete-family descriptors preserve their actual producer boundaries."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_native_directed import _blocks, _values
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _decimals
from .test_loop_portable import _rewrite


@pytest.fixture(scope="module")
def ordered_runtime():
    root = os.environ.get("PYAMPLICOL_ORDERED_COLOR_FIXTURES")
    if not root:
        pytest.skip("actual retained ordered-family artifacts required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    return Path(root), _rusticol._loop


def _projections(data):
    return (
        data["directed_projection"],
        *(item["directed"] for item in data["spin_projections"]),
    )


def _change(data, operation):
    for projection in _projections(data):
        operation(projection)


def _operations(projection, operations):
    projection["histories"][1].update(bra=operations, ket=operations)
    bank = projection["color_family_projection"]
    bank["families"][bank["history_families"][1]]["order"] = len(operations)


def test_native_complete_family_descriptor_and_positive_fresh_labels(
    ordered_runtime, tmp_path
):
    root, native = ordered_runtime
    source = root / "w-lc-ordered.pacbin"
    plan = native.PortableLoopPlan(str(source))
    data = json.loads(plan.metadata_json())
    bank = data["directed_projection"]["color_family_projection"]
    assert bank["kind"] == "complete-typed-history-families-v1"
    assert [item["order"] for item in bank["families"]] == [0, 1, 1]
    assert [item["virtual_degrees"] for item in bank["families"]] == [[3], [4], [3]]
    assert all(p["color_family_projection"] == bank for p in _projections(data))
    assert data["color_projection"]["virtual_degrees"] == [3]
    assert data.get("born_color_components") is None

    def rename(data):
        for projection in _projections(data):
            for history in projection["histories"]:
                for step in (*history["bra"], *history["ket"]):
                    for name in step.keys() - {"kind"}:
                        if step[name] < 0:
                            step[name] += 2**64

    changed = tmp_path / "positive-fresh.pacbin"
    _rewrite(source, changed, rename)
    native.PortableLoopPlan(str(changed))


@pytest.mark.parametrize(
    "operation,message",
    (
        (lambda p: p.update(color_family_projection=None), "convention"),
        (
            lambda p: p.update(convention="sun-graded-identity-all-outgoing-v1"),
            "convention",
        ),
        (
            lambda p: p["color_family_projection"].update(accuracy="nlc"),
            "family descriptor",
        ),
        (
            lambda p: p["color_family_projection"]["families"][1].update(
                virtual_degrees=[]
            ),
            "degree axes",
        ),
        (
            lambda p: p["color_family_projection"]["families"][0].update(
                virtual_degrees=[99]
            ),
            "identity family degrees",
        ),
        (
            lambda p: p["color_family_projection"]["history_families"].__setitem__(
                1, 99
            ),
            "family descriptor",
        ),
        (
            lambda p: p["color_family_projection"]["families"].append(
                p["color_family_projection"]["families"][0]
            ),
            "unique and referenced",
        ),
        (
            lambda p: p["color_family_projection"]["families"][1].update(
                output_legs=[[0, 3]]
            ),
            "output roles",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "emit-gluon", "emitter": 0, "emitted": 1.5}]
            ),
            "labels must be integers",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "emit-gluon", "emitter": 0, "emitted": True}]
            ),
            "invalid directed colour operation",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "split-gluon", "parent": 0, "quark": -1, "antiquark": -2}]
            ),
            "active gluon",
        ),
        (
            lambda p: _operations(
                p, [{"kind": "split-gluon", "parent": 3, "quark": 3, "antiquark": -2}]
            ),
            "remain fresh",
        ),
        (
            lambda p: _operations(
                p,
                [
                    {"kind": "split-gluon", "parent": 3, "quark": -1, "antiquark": -2},
                    {"kind": "emit-gluon", "emitter": -1, "emitted": -2},
                ],
            ),
            "remain fresh",
        ),
        (
            lambda p: _operations(
                p,
                [
                    {"kind": "emit-gluon", "emitter": 3, "emitted": -1},
                    {"kind": "split-gluon", "parent": -1, "quark": -1, "antiquark": -2},
                ],
            ),
            "remain fresh",
        ),
        (
            lambda p: p["histories"][1].update(
                ket=[{"kind": "emit-gluon", "emitter": 0, "emitted": -2}]
            ),
            "output roles",
        ),
    ),
)
def test_native_rejects_unproved_family_or_typed_operation(
    ordered_runtime, tmp_path, operation, message
):
    root, native = ordered_runtime
    destination = tmp_path / "bad.pacbin"
    _rewrite(root / "w-lc-ordered.pacbin", destination, lambda d: _change(d, operation))
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(destination))


def test_native_rejects_matching_changed_spin_metadata(ordered_runtime, tmp_path):
    root, native = ordered_runtime

    def change(data):
        for item in data["spin_projections"]:
            item["directed"]["color_family_projection"]["families"][1][
                "virtual_degrees"
            ] = [99]

    destination = tmp_path / "foreign-spin-family.pacbin"
    _rewrite(root / "w-lc-ordered.pacbin", destination, change)
    with pytest.raises(ValueError, match="history columns differ"):
        native.PortableLoopPlan(str(destination))


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    (
        ("double", None, "1e-8"),
        ("double-double", None, "1e-24"),
        ("arbitrary", 70, "1e-60"),
    ),
)
def test_saved_complete_components_match_actual_family_producer(
    ordered_runtime, accuracy, arithmetic, precision, tolerance
):
    root, native = ordered_runtime
    data = json.loads((root / "producer.json").read_text())
    records = [
        r
        for r in data["records"]
        if r["accuracy"] == accuracy and r["arithmetic"] == arithmetic
    ]
    assert len(records) == 4
    runtime = native.NativeLoopProgram(
        str(root / f"w-{accuracy}-ordered.pacbin"),
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )
    runtime.source_state.update(records[0]["parameters"])
    snapshot = runtime.source_state.snapshot()
    with localcontext() as context:
        context.prec = 120
        for record in records:
            sources = {int(leg): values for leg, values in record["sources"].items()}
            result = runtime.evaluate_directed(
                [data["point"]],
                ["1000000"],
                point_ids=[2**64 - 1],
                spin_vectors=sources,
            )
            assert result.point_ids == [2**64 - 1]
            assert result.spin_legs == record["spin_legs"]
            names = (
                "born",
                "virtual",
                "bare",
                "uv",
                "r2",
                "uv_local",
                "uv_external_fields",
                "first",
                "second",
            )
            for name, block in zip(names, _blocks(result)[:-1], strict=True):
                expected = _decimals(np.asarray(record["blocks"][name], dtype=object))
                actual = _values(block, expected.shape[:-1])
                _close_component(actual, expected, Decimal(tolerance))
        assert runtime.source_state.snapshot().overrides == snapshot.overrides
        empty = runtime.evaluate_directed(
            [], [], point_ids=[], spin_vectors={3: [], 4: []}
        )
        assert empty.point_ids == [] and empty.spin_legs == [3, 4]
        assert all(block.rows == 0 for block in _blocks(empty))
        again = runtime.evaluate_directed([data["point"]], ["1000000"], point_ids=[42])
        ordinary = next(r for r in records if not r["spin_legs"])
        expected = _decimals(np.asarray(ordinary["blocks"]["virtual"], dtype=object))
        _close_component(
            _values(again.component_batches[0], expected.shape[:-1]),
            expected,
            Decimal(tolerance),
        )


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_public_ordered_families_grouped_sources_and_subset_rescue(
    ordered_runtime, accuracy
):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
    from pyamplicol.api.errors import EvaluationError

    root, _ = ordered_runtime
    point = json.loads((root / "producer.json").read_text())["point"]
    runtime = Runtime.load(root / f"{accuracy}-artifact")
    assert runtime.physics.reduction.kind == "contracted-color"
    assert not runtime.physics.born_anchored_color_components
    assert runtime.inspect()["spin_correlations"] == ((3,), (3, 4), (4,))
    history_ids = {item.id for item in runtime.available_color_correlations()}
    assert {"born", *(f"history-{i}" for i in range(11))} <= history_ids
    w = [("2", "3"), ("5", "-7"), ("11", "13"), ("-17", "19")]
    g = [("0", "0"), ("0.8", "0"), ("0", "1"), ("-0.6", "0")]
    request = {
        "identity": CorrelatedRequest("born", {3: w, 4: g}),
        "qg": CorrelatedRequest("history-3", {3: w, 4: g}),
        "gq": CorrelatedRequest("history-7", {3: w, 4: g}),
        "split": CorrelatedRequest("history-10", {3: w, 4: g}),
    }
    points, scales = [point] * 3, ["1000000", "1e-400", "1000000"]
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
    )
    assert set(rescued) == set(request)
    for result in rescued.values():
        assert result.shape == (3, 1, 3)
        assert result.epsilon_powers == (0, -1, -2)
        assert [row.status for row in result.diagnostics] == [
            "success",
            "rescued",
            "success",
        ]
        assert result.values[0] == result.values[2]
    runtime.set_spin_correlation_vectors({3: w, 4: g})
    before = runtime.evaluate_correlated(
        points[:1], color_correlation="history-3", mu_r2=scales[:1]
    )
    runtime.clear()
    assert (
        runtime.evaluate_correlated(
            points[:1], color_correlation="history-3", mu_r2=scales[:1]
        ).values
        == before.values
    )
    assert runtime.evaluate_correlated(
        [], color_correlation="history-10", mu_r2=[]
    ).shape == (0, 1, 3)
