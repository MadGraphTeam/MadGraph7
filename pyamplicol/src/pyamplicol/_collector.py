# SPDX-License-Identifier: 0BSD
"""The cyclic garbage collector paused during an evaluation call.

A ``Runtime`` evaluation allocates many short-lived containers (the
per-point evaluations, lanes and result records) that are all freed by
reference counting when the call returns: a call creates no reference
cycles. While the call runs, though, the collector's young-generation
passes see them alive and promote them to the oldest generation, so that
every few calls a full pass walks every tracked object of the process,
the loaded runtime's records included (CPython 3.11: ~120 k objects for a
loaded loop-induced artifact, 60-120 ms per pass, ~10 % of a repeated
binary64 call on g g > z z; docs/user/python-api.md, "Garbage collector").
``paused`` keeps the collector off for the duration of the call, as
``timeit`` does, and restores its state afterwards: values are unchanged,
no object outlives the call because of it, and garbage cycles that other
threads create meanwhile are collected after it.

Set ``PYAMPLICOL_GC_PAUSE=0`` in the environment to leave the collector
alone (read at each call).
"""

from __future__ import annotations

import functools
import gc
import os
import threading
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

ENVIRONMENT = "PYAMPLICOL_GC_PAUSE"

_LOCK = threading.Lock()
# Evaluation calls in progress (all threads) and whether the collector was
# enabled when the first of them began.
_DEPTH = 0
_RESUME = False


def enabled() -> bool:
    """Whether evaluation calls pause the collector (``PYAMPLICOL_GC_PAUSE``:
    on unless ``0``, ``off``, ``false`` or ``no``)."""

    value = os.environ.get(ENVIRONMENT, "").strip().lower()
    return value not in {"0", "off", "false", "no"}


def paused(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """``function`` with the collector off while it runs (outermost call of
    nested or concurrent ones; the state found is restored when the last
    one returns)."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        global _DEPTH, _RESUME
        if not enabled():
            return function(*args, **kwargs)
        with _LOCK:
            if _DEPTH == 0:
                _RESUME = gc.isenabled()
                gc.disable()
            _DEPTH += 1
        try:
            return function(*args, **kwargs)
        finally:
            with _LOCK:
                _DEPTH -= 1
                if _DEPTH == 0 and _RESUME:
                    gc.enable()

    return wrapper


__all__ = ["ENVIRONMENT", "enabled", "paused"]
