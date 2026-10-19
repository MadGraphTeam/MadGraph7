# SPDX-License-Identifier: 0BSD
"""Public partial transport must preserve physical positions and every axis."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, Runtime
from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.api.virtual_results import VirtualFailureDiagnostic

from .test_public_virtual_colour import _transport_backend
from .test_public_virtual_spin import _runtime
from .test_runtime_virtual_policy import _policy


def _failure(point, *, directed=False):
    return VirtualFailureDiagnostic(
        point,
        1,
        2,
        7,
        "reduction",
        "partial transport control",
        color_component_index=None if directed else 2,
        history_index=1 if directed else None,
        direction=0 if directed else None,
    )


def _compact(batch, rows):
    width, flat = batch.columns, batch.to_decimal()
    retained = [
        value for row in rows for value in flat[row * width : (row + 1) * width]
    ]
    return SimpleNamespace(
        rows=len(rows),
        columns=width,
        row_ids=list(range(len(rows))),
        to_decimal=lambda: retained,
    )


def _partial(handle, *, directed=False, rejected=None):
    name = "evaluate_directed" if directed else "evaluate_partial"
    complete = getattr(handle, name)

    def evaluate(points, scales, **options):
        result = complete(points, scales, **options)
        bad = (
            rejected(options)
            if rejected is not None
            else {
                point
                for point, scale in enumerate(scales)
                if Decimal(scale) < Decimal("1e-300")
            }
        )
        successful = [point for point in range(len(points)) if point not in bad]
        rows_per_point = result.born.rows // len(points) if points else 1
        rows = [
            point * rows_per_point + row
            for point in successful
            for row in range(rows_per_point)
        ]
        result.born = _compact(result.born, rows)
        result.component_batches = [
            _compact(batch, rows) for batch in result.component_batches
        ]
        result.point_ids = successful
        result.failures = [_failure(point, directed=directed) for point in sorted(bad)]
        result.statuses = [[] for _ in rows]
        return result

    setattr(handle, name, evaluate)
    return evaluate


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("scales", [(), ("1e-400",), ("1", "1e-400", "3", "1e-400")])
def test_partial_points_keep_original_positions_and_all_order_axes(
    monkeypatch, resolved, scales
):
    backend, native = _transport_backend(monkeypatch)
    runtime = Runtime(backend)
    operation = runtime.evaluate_resolved if resolved else runtime.evaluate
    points = [[[1, 0, 0, 1]]] * len(scales)
    baseline = operation(points, mu_r2=1)
    _partial(native)
    if "1e-400" in scales:
        with pytest.raises(EvaluationError) as caught:
            operation(points, mu_r2=scales)
        expected_failures = caught.value.failures
    else:
        expected_failures = ()
    result = operation(points, mu_r2=scales, on_failure="return")
    assert result.shape == baseline.shape and result.born_shape == baseline.born_shape
    assert result.sectors == baseline.sectors and result.epsilon_powers == (0, -1, -2)
    assert result.r2 == baseline.r2 and result.color_accuracy == "lc"
    assert result.valid == tuple(scale != "1e-400" for scale in scales)
    assert (
        tuple(f for lane in result.diagnostics for f in lane.failures)
        == expected_failures
    )
    for point, valid in enumerate(result.valid):
        if valid:
            assert result.values[point] == baseline.values[point]
            assert result.born_values[point] == baseline.born_values[point]
            assert result.diagnostics[point].status == "success"
        else:
            assert result.values[point] is result.born_values[point] is None
            assert result.diagnostics[point].failures == (_failure(point),)
    total = result.total() if resolved else result
    assert total.valid == result.valid
    assert total.diagnostics == result.diagnostics
    assert total.for_order(total.sectors[2].order).diagnostics == result.diagnostics
    assert tuple(value is None for value in total.sum_orders()) == tuple(
        not valid for valid in result.valid
    )
    for coefficients in (total.finite, total.single_pole, total.double_pole):
        assert tuple(value is None for value in coefficients) == tuple(
            not valid for valid in result.valid
        )
        assert all(
            isinstance(value, Decimal)
            for row in coefficients
            if row is not None
            for value in row
        )


def test_grouped_sources_keep_independent_masks_and_complex_precision(monkeypatch):
    runtime, _, _ = _runtime(monkeypatch)
    native = runtime._backend._native
    exact = "1.00000000000000000000000000000000000000001"
    runtime.set_spin_correlation_vectors({1: [0, (exact, "-2"), 0, 0]})
    original_sources = deepcopy(runtime._backend._spin_vectors)
    requests = {
        "source": CorrelatedRequest("x"),
        "physical": CorrelatedRequest("born", {}),
        "same-source": CorrelatedRequest("born"),
    }
    points = [[[1, 0, 0, 1]]] * 3
    baseline = runtime.evaluate_correlated_many(points, requests, mu_r2=1)
    _partial(
        native,
        directed=True,
        rejected=lambda options: {1} if options["spin_vectors"] else {0, 2},
    )
    result = runtime.evaluate_correlated_many(
        points, requests, mu_r2=1, on_failure="return"
    )
    assert list(result) == list(requests)
    for label, value in result.items():
        expected_mask = (
            (False, True, False) if label == "physical" else (True, False, True)
        )
        assert value.valid == expected_mask
        assert value.correlated and value.sectors == baseline[label].sectors
        for point, valid in enumerate(expected_mask):
            if valid:
                assert value.values[point] == baseline[label].values[point]
                assert isinstance(value.values[point][0][0], CorrelatedValue)
            else:
                assert value.values[point] is None
                assert value.diagnostics[point].failures == (
                    _failure(point, directed=True),
                )
    assert result["source"].diagnostics == result["same-source"].diagnostics
    # Per-request failure policy does not mutate the retained source setter.
    assert runtime._backend._spin_vectors == original_sources


def test_unestablished_accuracy_never_leaks_the_last_attempt(monkeypatch):
    backend, native = _transport_backend(monkeypatch, "full")
    partial = _partial(native)
    policy = _policy()

    def verified(points, scales, *args, **kwargs):
        kwargs.pop("verify_successful")
        result = partial(points, scales, **kwargs)
        diagnostics = [
            dict(
                point_id=point,
                status="accuracy-not-established" if point == 1 else "success",
                accuracy_established=point != 1,
                agreeing_bits=None,
                attempts=[],
                detail="agreement control",
            )
            for point in range(len(points))
        ]
        return SimpleNamespace(result=result, diagnostics_json=json.dumps(diagnostics))

    native.evaluate_verified = verified
    points = [[[1, 0, 0, 1]]] * 2
    result = backend.evaluate(
        points, mu_r2=[1, "1e-400"], stability=policy, on_failure="return"
    )
    assert result.valid == (True, False)
    assert result.values[1] is None
    assert result.diagnostics[1].status == "accuracy-not-established"
    assert not result.diagnostics[1].accuracy_established
    assert result.diagnostics[1].verification.detail == "agreement control"
    with pytest.raises(EvaluationError) as caught:
        backend.evaluate(points, mu_r2=[1, "1e-400"], stability=policy)
    assert caught.value.verification[1] == result.diagnostics[1].verification
    assert caught.value.failures == result.diagnostics[1].failures


@pytest.mark.parametrize(
    "corruption", ["requested", "duplicate", "foreign", "missing", "status", "shape"]
)
def test_partial_return_rejects_inconsistent_native_metadata(monkeypatch, corruption):
    backend, native = _transport_backend(monkeypatch)
    partial = _partial(native)

    def corrupt(*args, **kwargs):
        result = partial(*args, **kwargs)
        if corruption == "requested":
            result.requested_point_ids = [0, 9]
        elif corruption == "duplicate":
            result.point_ids = [0, 0]
        elif corruption == "foreign":
            result.failures = [replace(result.failures[0], point_id=9)]
        elif corruption == "missing":
            result.failures = []
        elif corruption == "status":
            result.statuses.append([])
        else:
            result.born.row_ids = [1]
        return result

    native.evaluate_partial = corrupt
    with pytest.raises(EvaluationError):
        backend.evaluate([[[1, 0, 0, 1]]] * 2, mu_r2=[1, "1e-400"], on_failure="return")


def test_partial_return_keeps_input_and_capability_errors_hard(monkeypatch):
    backend, _ = _transport_backend(monkeypatch)
    runtime = Runtime(backend)
    for options in ({"mu_r2": -1}, {"mu_r2": 1, "on_failure": "ignore"}):
        with pytest.raises(ValueError):
            runtime.evaluate([], **options)
    with pytest.raises(CompatibilityError):
        runtime.evaluate([], mu_r2=1, r2_strategy="direct", on_failure="return")
    with pytest.raises(ValueError):
        runtime.evaluate([[["NaN", 0, 0, 1]]], mu_r2=1, on_failure="return")
    # Additive virtual-only return selection does not alter old tree signatures.
    runtime._backend = SimpleNamespace(contribution_kind="tree")
    with pytest.raises(CompatibilityError, match="partial-value"):
        runtime._evaluation_options(None, None, None, on_failure="return")
    assert runtime._evaluation_options(None, None, None) == {}


def test_failure_diagnostics_are_frozen_and_accuracy_status_is_invalid():
    from pyamplicol.api.virtual_results import VirtualLaneDiagnostic

    values = [_failure(1)]
    diagnostic = VirtualLaneDiagnostic(
        "accuracy-not-established", "double", 53, False, failures=values
    )
    values.clear()
    assert diagnostic.failures == (_failure(1),) and not diagnostic.valid
    assert asdict(diagnostic)["failures"][0]["point_id"] == 1
    with pytest.raises(ValueError, match="unrecovered"):
        replace(diagnostic, status="success")
