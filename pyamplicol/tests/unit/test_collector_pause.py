# SPDX-License-Identifier: 0BSD
"""The cyclic collector paused during Runtime evaluation calls
(``pyamplicol._collector``; docs/user/python-api.md, "Garbage collector")."""

from __future__ import annotations

import gc
import inspect

import pytest

from pyamplicol import _collector
from pyamplicol.api import Runtime

EVALUATIONS = (
    "evaluate",
    "evaluate_with_prec",
    "evaluate_resolved",
    "evaluate_resolved_with_prec",
    "evaluate_orders",
    "evaluate_graded",
    "evaluate_correlated",
    "evaluate_correlated_graded",
    "evaluate_correlated_many",
)


@pytest.fixture(autouse=True)
def _restore_collector():
    enabled = gc.isenabled()
    yield
    if enabled:
        gc.enable()
    else:
        gc.disable()


def test_the_collector_is_off_during_a_call_and_restored_after(monkeypatch):
    monkeypatch.delenv(_collector.ENVIRONMENT, raising=False)
    seen = []

    @_collector.paused
    def inner():
        seen.append(("inner", gc.isenabled()))
        return 1

    @_collector.paused
    def outer():
        seen.append(("outer", gc.isenabled()))
        value = inner()
        seen.append(("after inner", gc.isenabled()))
        return value + 1

    gc.enable()
    assert outer() == 2
    assert seen == [("outer", False), ("inner", False), ("after inner", False)]
    assert gc.isenabled()
    # A collector the caller disabled stays disabled.
    gc.disable()
    assert outer() == 2
    assert not gc.isenabled()


def test_the_state_is_restored_when_the_call_raises(monkeypatch):
    monkeypatch.delenv(_collector.ENVIRONMENT, raising=False)

    @_collector.paused
    def failing():
        raise ValueError("refused")

    gc.enable()
    with pytest.raises(ValueError, match="refused"):
        failing()
    assert gc.isenabled()


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " OFF "])
def test_the_environment_opts_out(monkeypatch, value):
    monkeypatch.setenv(_collector.ENVIRONMENT, value)

    @_collector.paused
    def call():
        return gc.isenabled()

    gc.enable()
    assert call() is True


def test_every_runtime_evaluation_pauses_the_collector_and_keeps_its_signature():
    for name in EVALUATIONS:
        method = getattr(Runtime, name)
        assert inspect.unwrap(method) is not method, name
        assert inspect.signature(method) == inspect.signature(inspect.unwrap(method))
