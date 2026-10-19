# SPDX-License-Identifier: 0BSD
"""Public source snapshots and grouping retain exact per-point complex inputs."""

from decimal import Decimal

import pytest

from pyamplicol import CorrelatedRequest
from pyamplicol.api.errors import EvaluationError

from .test_public_virtual_directed import _directed


def _runtime(monkeypatch):
    runtime, native, calls = _directed(monkeypatch)
    runtime._backend._metadata["plan"]["spin_projections"] = [{"legs": [1]}]
    observed = []
    evaluate = native.evaluate_directed

    def capture(*args, **kwargs):
        observed.append(kwargs["spin_vectors"])
        return evaluate(*args, **kwargs)

    native.evaluate_directed = capture
    native.clear = lambda: None
    return runtime, calls, observed


def test_grouped_spin_snapshots_exact_broadcast_deduplication_and_clear(monkeypatch):
    runtime, calls, observed = _runtime(monkeypatch)
    tiny = Decimal("1.00000000000000000000000000000000000000000000000000001")
    vector = [(tiny, Decimal("-2.125")), 2, 3, 4]
    runtime.set_spin_correlation_vectors({1: vector})
    vector[0] = 0
    points = [[[1, 0, 0, 1]]] * 2
    exact = [(tiny, Decimal("-2.125")), 2, 3, 4]
    result = runtime.evaluate_correlated_many(
        points,
        {
            "default": CorrelatedRequest("x"),
            "same": CorrelatedRequest("born", {1: [exact, exact]}),
            "physical": CorrelatedRequest("born", {}),
        },
        mu_r2=[1, 2],
    )
    assert list(result) == ["default", "same", "physical"]
    assert len(calls) == 2
    assert observed[0][1][0][0] == [str(tiny), "-2.125"]
    assert observed[0][1][1] == observed[0][1][0]
    assert observed[1] == {}
    assert runtime.inspect()["spin_correlations"] == ((1,),)
    runtime.clear()
    runtime.evaluate_correlated(points, mu_r2=1)
    assert observed[-1] == observed[0]
    runtime.set_spin_correlation_vectors(None)
    runtime.evaluate_correlated(points, mu_r2=1)
    assert observed[-1] == {}


def test_spin_requests_validate_atomically_and_empty_keeps_class(monkeypatch):
    runtime, calls, observed = _runtime(monkeypatch)
    runtime.set_spin_correlation_vectors({1: [0, 1, 0, 0]})
    with pytest.raises(EvaluationError, match=r"four-vector|batch"):
        runtime.set_spin_correlation_vectors({1: []})
    with pytest.raises(EvaluationError, match="not declared"):
        runtime.evaluate_correlated_many(
            [[[1, 0, 0, 1]]],
            {
                "valid": CorrelatedRequest(),
                "bad": CorrelatedRequest("born", {2: [0, 1, 0, 0]}),
            },
            mu_r2=1,
        )
    assert calls == []
    result = runtime.evaluate_correlated([], mu_r2=[])
    assert result.shape == (0, 3, 3)
    assert observed == [{1: []}]
    runtime.evaluate_correlated([[[1, 0, 0, 1]]], mu_r2=1)
    assert observed[-1][1] == [[["0", "0"], ["1", "0"], ["0", "0"], ["0", "0"]]]
