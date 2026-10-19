# SPDX-License-Identifier: 0BSD
"""Public directed columns stay complex and retain every order in one call."""

import io
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, Runtime
from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.cli.main import write_result

from .test_public_virtual_colour import _transport_backend


def _directed(monkeypatch):
    backend, handle = _transport_backend(monkeypatch, "full")
    backend._metadata["physics"].update(resolved=False, correlations=True)
    backend._metadata["plan"]["directed_projection"] = {
        "histories": [{"id": name, "bra": [], "ket": []} for name in ("born", "x")]
    }
    calls = []

    def batch(rows, width):
        return SimpleNamespace(
            rows=rows,
            columns=width,
            row_ids=list(range(rows)),
            to_decimal=lambda: [
                (str(r * 100 + c + 1), str(-(r * 100 + c + 2)))
                for r in range(rows)
                for c in range(width)
            ],
        )

    def evaluate(points, scales, *, partial, r2_strategy):
        calls.append((points, scales, partial, r2_strategy))
        return SimpleNamespace(
            point_ids=list(range(len(points))),
            r2_strategy="model",
            failures=handle.failures,
            statuses=[[] for _ in points],
            born=batch(len(points), 4),
            component_batches=[batch(len(points), 18)] * 6,
            observable_kind="directed-complex",
            history_ids=["born", "x"],
        )

    handle.evaluate_directed = evaluate
    return Runtime(backend), handle, calls


def test_grouped_directed_transport_preserves_unequal_orders_and_complex_axes(
    monkeypatch,
):
    runtime, _, calls = _directed(monkeypatch)
    assert [item.id for item in runtime.available_color_correlations()] == ["born", "x"]
    requests = {"second": CorrelatedRequest("x"), "first": CorrelatedRequest("born")}
    result = runtime.evaluate_correlated_many([[[1, 0, 0, 1]]] * 3, requests, mu_r2=1)
    assert len(calls) == 1
    assert list(result) == ["second", "first"]
    for label, h in (("second", 1), ("first", 0)):
        value = result[label]
        assert value.correlated and value.color_accuracy == "full"
        assert value.shape == (3, 3, 3) and value.born_shape == (3, 2)
        assert value.diagnostics[0].arithmetic == "arbitrary"
        for p in range(3):
            for b in range(2):
                c = h * 2 + b
                assert value.born_values[p][b] == CorrelatedValue(
                    Decimal(p * 100 + c + 1), Decimal(-(p * 100 + c + 2))
                )
            for v in range(3):
                for e in range(3):
                    c = h * 9 + 3 * v + e
                    assert value.values[p][v][e] == CorrelatedValue(
                        Decimal(p * 100 + c + 1), Decimal(-(p * 100 + c + 2))
                    )
        for format in ("json", "human"):
            stream = io.StringIO()
            write_result(value, format=format, stream=stream)
            if format == "json":
                assert json.loads(stream.getvalue())["correlated"] is True
            else:
                assert "Directed colour correlations" in stream.getvalue()
    empty = runtime.evaluate_correlated([], mu_r2=[])
    assert empty.shape == (0, 3, 3)
    assert empty.correlated


def test_directed_unsupported_requests_fail_before_native_work(monkeypatch):
    runtime, _, calls = _directed(monkeypatch)
    for kwargs in (
        {"color_correlation": "missing"},
        {"helicities": ["h0"]},
    ):
        with pytest.raises(CompatibilityError):
            runtime.evaluate_correlated([], mu_r2=[], **kwargs)
    with pytest.raises(CompatibilityError, match="spin sources"):
        runtime.set_spin_correlation_vectors({1: (1, 0, 0, 1)})
    with pytest.raises(CompatibilityError, match="spin sources"):
        runtime.evaluate_correlated_many(
            [], {"spin": CorrelatedRequest("born", {1: (1, 0, 0, 1)})}, mu_r2=[]
        )
    assert runtime.evaluate_correlated_many([], {}, mu_r2=[]) == {}
    assert calls == []
    runtime.set_spin_correlation_vectors(None)
    runtime.set_spin_correlation_vectors({})


def test_directed_failures_keep_history_and_direction(monkeypatch):
    runtime, native, _ = _directed(monkeypatch)
    native.failures = [
        SimpleNamespace(
            point_id=1,
            helicity=None,
            order_index=2,
            family_index=4,
            stage="reduction",
            detail="injected directed transport failure",
            external_leg=None,
            color_component_index=None,
            history_index=1,
            direction=0,
        )
    ]
    with pytest.raises(EvaluationError) as caught:
        runtime.evaluate_correlated([[[1, 0, 0, 1]]] * 3, mu_r2=1)
    failure = caught.value.failures[0]
    assert (failure.point_id, failure.history_index, failure.direction) == (1, 1, 0)
    with pytest.raises(ValueError, match="direction"):
        replace(failure, direction=2)


def test_reserved_born_alias_requires_one_semantic_identity(monkeypatch):
    runtime, native, calls = _directed(monkeypatch)
    histories = runtime._backend._metadata["plan"]["directed_projection"]["histories"]
    histories[0]["id"] = "identity"
    step = {"kind": "emit-gluon", "emitter": 1, "emitted": -1}
    histories[1]["bra"] = histories[1]["ket"] = [step]
    evaluate = native.evaluate_directed

    def actual_names(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result.history_ids = ["identity", "x"]
        return result

    native.evaluate_directed = actual_names
    assert [x.id for x in runtime.available_color_correlations()] == [
        "identity",
        "x",
        "born",
    ]
    value = runtime.evaluate_correlated([[[1, 0, 0, 1]]], mu_r2=1)
    assert value.born_values[0][0] == CorrelatedValue(Decimal(1), Decimal(-2))
    assert len(calls) == 1
    histories[1]["bra"] = histories[1]["ket"] = []
    with pytest.raises(CompatibilityError, match="undeclared"):
        runtime.evaluate_correlated([], mu_r2=[])
    histories[1]["id"] = "born"
    histories[1]["bra"] = histories[1]["ket"] = [step]
    with pytest.raises(CompatibilityError, match="empty bra and ket"):
        runtime.available_color_correlations()


def test_colour_label_transport_preserves_auxiliaries_and_order():
    from pyamplicol import ColorCorrelator, EmitGluon, SplitGluon
    from pyamplicol.loops.correlation_labels import (
        native_color_history,
        public_color_history,
    )

    native = ColorCorrelator(
        "ordered",
        (EmitGluon(0, -1), SplitGluon(-1, 4, -2)),
        (EmitGluon(2, -1), SplitGluon(-1, 4, -2)),
    )
    public = public_color_history(native)
    assert public.bra == (EmitGluon(1, -1), SplitGluon(-1, 5, -2))
    assert public.ket[0] == EmitGluon(3, -1)
    assert native_color_history(public) == native
    with pytest.raises(ValueError, match="public colour labels"):
        native_color_history(native)
