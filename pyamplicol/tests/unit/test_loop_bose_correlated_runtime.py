# SPDX-License-Identifier: 0BSD
"""Actual saved Bose history/source transport against an independent member.

These controls use normally published artifacts, never generated graphs. The
separate pure three-cycle tests establish exact coordinate action; Bose symmetry
alone cannot detect consistently omitting both the momentum and source action.
"""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_native_directed import _blocks, _values
from .test_loop_physical_correlations import _close
from .test_loop_portable import _rewrite
from .test_loop_projected_flavor_runtime import _close_values


@pytest.fixture(scope="module")
def saved_bose_sources():
    location = os.environ.get("PYAMPLICOL_BOSE_CORRELATED_FIXTURE")
    if not location:
        pytest.skip("requires the actual admitted Bose source/history artifacts")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = json.loads(path.read_text(), parse_float=str, parse_int=str)["points"]
    return Path(location), _rusticol._loop, points


def _vectors():
    return (
        [("0.1", "0.2"), ("0.8", "-0.3"), ("0.4", "1"), ("-0.6", "0.5")],
        [("-0.3", "0.7"), ("-0.2", "0.8"), ("0.6", "0.1"), ("0.5", "-0.4")],
    )


def _source_rows(key, size):
    a, b = _vectors()
    return {leg: [a if leg == 3 else b for _ in range(size)] for leg in key}


def _compare_selected(actual, reference, ids, key, tolerance):
    assert actual.point_ids == actual.requested_point_ids == ids
    assert reference.point_ids == reference.requested_point_ids == ids
    assert actual.history_ids == ["born", "casimir"]
    assert actual.spin_legs == list(key)
    assert actual.observable_kind == "directed-complex"
    assert not actual.failures
    columns = [reference.history_ids.index(name) for name in actual.history_ids]
    for a, b in zip(_blocks(actual), _blocks(reference), strict=True):
        width, remainder = divmod(b.columns, len(reference.history_ids))
        assert remainder == 0
        assert a.columns == len(columns) * width
        assert a.rows == b.rows == len(ids)
        _close(
            _values(a, (len(ids), len(columns), width)),
            _values(b, (len(ids), len(reference.history_ids), width))[:, columns],
            Decimal(tolerance),
        )
    assert len(actual.statuses) == len(ids)
    for index, (row, original) in enumerate(
        zip(actual.statuses, reference.statuses, strict=True)
    ):
        assert len(row) * len(reference.history_ids) == len(original) * len(columns)
        assert all(row_id == index and status == "success" for row_id, status in row)
    _close(
        _values(
            actual.component_batches[0], (len(ids), actual.component_batches[0].columns)
        ),
        _values(actual.directed_first, (len(ids), actual.directed_first.columns))
        + _values(actual.directed_second, (len(ids), actual.directed_second.columns)),
        Decimal(tolerance),
    )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-10", 20),
        ("double-double", None, "generic", "3e-23", 70),
        ("arbitrary", 70, "generic", "8e-54", 170),
    ],
)
def test_saved_bose_selected_components_sources_and_exact_parameters(
    saved_bose_sources, arithmetic, precision, engine, tolerance, bit
):
    root, native, points = saved_bose_sources
    with localcontext() as context:
        context.prec = 120
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        parameters = {"aS": (alpha, "0"), "MT": ("173", "0"), "MU_R": ("1000", "0")}
        selected = native.NativeVirtualRuntime(
            str(root / "artifact"),
            process="ssgg-swap",
            parameters=parameters,
            arithmetic=arithmetic,
            precision=precision,
        )
        independent = native.NativeLoopProgram(
            str(root / "independent.pacbin"),
            parameters=parameters,
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine=engine,
        )
        metadata = json.loads(selected.metadata_json())
        assert metadata["physics"]["correlations"]
        assert not metadata["physics"]["resolved"]
        assert metadata["representative_physics"]["correlations"]
        catalog = metadata["selected_correlations"]
        assert [h["id"] for h in catalog["histories"]] == ["born", "casimir"]
        assert catalog["spin_classes"] == [[3], [3, 4], [4]]
        ordinary = metadata["plan"]["directed_projection"]["histories"]
        assert catalog["histories"] == [
            next(h for h in ordinary if h["id"] == name) for name in ("born", "casimir")
        ]
        assert {h["id"] for h in ordinary} == {"born", "casimir", "ordered", "swapped"}
        for key in ((), (3,), (4,), (3, 4)):
            sizes = (0, 1, 3) if arithmetic == "double" else (1,)
            for size in sizes:
                batch = [points[1], points[0], points[1]][:size]
                ids = [2**64 - 1, 7, 42][:size]
                scales = ["900000", "1000000", "900000"][:size]
                sources = _source_rows(key, size)
                actual = selected.evaluate_directed(
                    batch, scales, point_ids=ids, spin_vectors=sources
                )
                expected = independent.evaluate_directed(
                    batch, scales, point_ids=ids, spin_vectors=sources
                )
                _compare_selected(actual, expected, ids, key, tolerance)
                if size == 3:
                    assert (
                        actual.component_batches[0].take_rows([0]).to_decimal()
                        == actual.component_batches[0].take_rows([2]).to_decimal()
                    )
        with pytest.raises(ValueError, match="resolved"):
            selected.evaluate_resolved(points[:1], ["1000000"])


def test_single_member_source_class_needs_only_inverse_representative_class(
    saved_bose_sources,
):
    root, native, points = saved_bose_sources
    selected = native.NativeVirtualRuntime(
        str(root / "single-source-artifact"), process="ssgg-swap"
    )
    metadata = json.loads(selected.metadata_json())
    assert metadata["selected_correlations"]["spin_classes"] == [[4]]
    assert [item["legs"] for item in metadata["plan"]["spin_projections"]] == [[3]]
    source = _source_rows((4,), 1)
    result = selected.evaluate_directed(
        points[:1], ["1000000"], point_ids=[2**64 - 1], spin_vectors=source
    )
    independent = native.NativeLoopProgram(str(root / "independent.pacbin"))
    reference = independent.evaluate_directed(
        points[:1], ["1000000"], point_ids=[2**64 - 1], spin_vectors=source
    )
    _compare_selected(result, reference, [2**64 - 1], (4,), "8e-10")
    with pytest.raises(ValueError, match=r"class|source"):
        selected.evaluate_directed(
            points[:1], ["1000000"], spin_vectors=_source_rows((3,), 1)
        )


def test_public_selected_scope_retains_representative_catalog_and_literal_sources(
    saved_bose_sources,
):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api.errors import CompatibilityError

    root, _, points = saved_bose_sources
    selected = Runtime.load(root / "artifact", process="ssgg-swap")
    representative = Runtime.load(root / "artifact", process="ddgg")
    independent = Runtime.load(root / "independent-artifact", process="ssgg")
    assert {h.id for h in selected.available_color_correlations()} == {
        "born",
        "casimir",
    }
    assert {h.id for h in representative.available_color_correlations()} == {
        "born",
        "casimir",
        "ordered",
        "swapped",
    }
    assert selected.inspect()["spin_correlations"] == ((3,), (3, 4), (4,))
    a, b = _vectors()
    requests = {
        "ordinary": CorrelatedRequest("born"),
        "single": CorrelatedRequest("casimir", {4: b}),
        "joint": CorrelatedRequest("casimir", {3: a, 4: b}),
    }
    actual = selected.evaluate_correlated_many(points[:1], requests, mu_r2="1000000")
    expected = independent.evaluate_correlated_many(
        points[:1], requests, mu_r2="1000000"
    )
    for name in requests:
        assert actual[name].epsilon_powers == (0, -1, -2)
        assert actual[name].born_orders == expected[name].born_orders
        assert actual[name].sectors == expected[name].sectors
        _close_values(actual[name].values, expected[name].values)
    for name in ("ordered", "swapped"):
        with pytest.raises(CompatibilityError, match=r"histories|history|correlation"):
            selected.evaluate_correlated(
                points[:1], color_correlation=name, mu_r2="1000000"
            )
    selected.set_spin_correlation_vectors({4: b})
    before = selected.evaluate_correlated(
        points[:1], color_correlation="casimir", mu_r2="1000000"
    )
    selected.clear()
    assert (
        selected.evaluate_correlated(
            points[:1], color_correlation="casimir", mu_r2="1000000"
        ).values
        == before.values
    )
    assert selected.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)


def test_public_bose_subset_rescue_preserves_sources_and_attempt_labels(
    saved_bose_sources,
):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy

    root, _, points = saved_bose_sources
    runtime = Runtime.load(root / "artifact", process="ssgg-swap")
    a, b = _vectors()
    with localcontext() as context:
        context.prec = 100
        result = runtime.evaluate_correlated_many(
            [points[0]] * 3,
            {"joint": CorrelatedRequest("casimir", {3: a, 4: b})},
            mu_r2=["1000000", "1e-400", "1000000"],
            stability=VirtualStabilityPolicy(
                (
                    VirtualArithmeticSpec("arbitrary", 60),
                    VirtualArithmeticSpec("arbitrary", 80),
                ),
                "1e-8",
                "1e-9",
            ),
        )["joint"]
    assert [row.status for row in result.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert result.values[0] == result.values[2]
    attempts = result.diagnostics[1].verification.attempts
    assert {f.point_id for attempt in attempts for f in attempt.failures} == {1}
    assert [attempt.arithmetic.arithmetic for attempt in attempts][-2:] == [
        "arbitrary",
        "arbitrary",
    ]
    assert all(
        f.history_index is None or 0 <= f.history_index < 2
        for attempt in attempts
        for f in attempt.failures
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_ids", []),
        ("history_ids", ["born", "born"]),
        ("history_ids", ["missing"]),
        ("history_ids", ["ordered"]),
        ("spin_classes", [[0]]),
        ("spin_classes", [[True]]),
        ("spin_classes", [[4, 3]]),
        ("spin_classes", [[3], [3]]),
        ("spin_classes", [[1]]),
    ],
)
def test_saved_scope_rejects_missing_or_unproved_rows(
    saved_bose_sources, tmp_path, field, value
):
    root, native, _ = saved_bose_sources
    destination = tmp_path / "bad-scope.pacbin"

    def change(plan):
        plan["admitted_flavor_members"][0]["bose_correlations"][field] = value

    _rewrite(root / "admitted.pacbin", destination, change)
    with pytest.raises(ValueError):
        native.PortableLoopPlan(str(destination))
