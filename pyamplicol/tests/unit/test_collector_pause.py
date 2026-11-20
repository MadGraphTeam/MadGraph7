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


def test_an_interrupt_inside_the_entry_leaves_the_collector_on(monkeypatch):
    """G3: an exception (here a KeyboardInterrupt raised by gc.disable, i.e.
    after the call's token is registered) anywhere in the entry is undone by
    the call's own ``finally``."""

    monkeypatch.delenv(_collector.ENVIRONMENT, raising=False)
    calls = []

    @_collector.paused
    def call():
        calls.append(1)

    real = gc.disable

    def interrupted():
        raise KeyboardInterrupt

    gc.enable()
    monkeypatch.setattr(_collector.gc, "disable", interrupted)
    with pytest.raises(KeyboardInterrupt):
        call()
    monkeypatch.setattr(_collector.gc, "disable", real)
    assert calls == [] and gc.isenabled() and not _collector._ACTIVE
    # The pause works normally afterwards.
    seen = []

    @_collector.paused
    def again():
        seen.append(gc.isenabled())

    again()
    assert seen == [False] and gc.isenabled() and not _collector._ACTIVE


def test_overlapping_threads_catch_up_and_one_thread_never_does(monkeypatch):
    """G1: calls of two threads overlapping for longer than CATCH_UP seconds
    run a young collection when one returns; a thread's own nested calls
    never do; the last call out restores the collector."""

    import threading

    monkeypatch.delenv(_collector.ENVIRONMENT, raising=False)
    clock = [100.0]
    monkeypatch.setattr(_collector, "_now", lambda: clock[0])
    collected = []
    monkeypatch.setattr(
        _collector.gc, "collect", lambda generation=2: collected.append(generation)
    )
    inside, release = threading.Event(), threading.Event()

    @_collector.paused
    def long_call():
        inside.set()
        release.wait(10)

    @_collector.paused
    def nested():
        clock[0] += 5.0
        return short()

    @_collector.paused
    def short():
        return gc.isenabled()

    gc.enable()
    # One thread, nested calls, long past CATCH_UP: no catch-up.
    assert nested() is False and collected == [] and gc.isenabled()
    worker = threading.Thread(target=long_call)
    worker.start()
    assert inside.wait(10)
    clock[0] += _collector.CATCH_UP + 0.5
    assert short() is False  # the collector is off while the other call runs
    assert collected == [1] and not gc.isenabled()
    assert short() is False and collected == [1]  # not again within CATCH_UP
    release.set()
    worker.join(10)
    assert gc.isenabled() and not _collector._ACTIVE
