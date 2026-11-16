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

State: the calls in progress (all threads) are a set of per-call tokens; the
first one in turns the collector off if it was on, the last one out turns it
back on only if this module turned it off. Entering and leaving are each one
protected region that the call's ``finally`` undoes by its token, so an
exception (``KeyboardInterrupt`` included) at any point of the entry leaves
nothing behind. While calls overlap continuously (several threads), every
``CATCH_UP`` seconds a returning call while another thread's call is still
running runs one young collection (generation 1; every ``FULL_EVERY``-th
one a full collection), so that overlapping calls never hold the collector
off indefinitely; one thread's calls (nested ones included) never pay it.
A ``gc.disable()`` made by another thread while calls run is not seen (the
collector is off either way): the last call out turns it back on.
"""

from __future__ import annotations

import functools
import gc
import os
import threading
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

ENVIRONMENT = "PYAMPLICOL_GC_PAUSE"

_LOCK = threading.Lock()
# The calls in progress (one token each, all threads), whether this module
# turned the collector off when the first of them began, and the catch-up
# clock of a continuous pause.
_ACTIVE: dict[object, int] = {}
_OURS = False
_CAUGHT_UP = 0.0
_CATCH_UPS = 0
# Seconds of continuous pause (overlapping calls) between two catch-up
# collections, and the share of them that are full collections.
CATCH_UP = 1.0
FULL_EVERY = 10
_now = time.monotonic


def enabled() -> bool:
    """Whether evaluation calls pause the collector (``PYAMPLICOL_GC_PAUSE``:
    on unless ``0``, ``off``, ``false`` or ``no``)."""

    value = os.environ.get(ENVIRONMENT, "").strip().lower()
    return value not in {"0", "off", "false", "no"}


def _enter(token: object) -> None:
    """Register a call; the first one turns the collector off. The token is
    added before the collector is touched, and the state that ``_leave``
    restores is recorded before that: an exception anywhere here leaves a
    state that ``_leave(token)`` undoes (or nothing to undo)."""

    global _OURS, _CAUGHT_UP
    with _LOCK:
        first = not _ACTIVE
        if first:
            _OURS = gc.isenabled()
            _CAUGHT_UP = _now()
        _ACTIVE[token] = threading.get_ident()
        if first and _OURS:
            gc.disable()


def _leave(token: object) -> int | None:
    """Unregister a call (nothing if ``_enter`` did not register it); the
    last one turns the collector back on if this module turned it off. The
    collector is restored before the token is dropped, so an exception after
    the restore can leave the pause unused, never the collector off; one
    delivered before it (a ``KeyboardInterrupt`` at the very end of a call,
    as in any interrupted ``finally``) can leave it off. Returns the
    generation of a catch-up collection due (calls overlapping for
    ``CATCH_UP`` seconds), else None."""

    global _CAUGHT_UP, _CATCH_UPS
    with _LOCK:
        if token not in _ACTIVE:
            return None
        if len(_ACTIVE) == 1:
            if _OURS:
                gc.enable()
            del _ACTIVE[token]
            return None
        thread = _ACTIVE.pop(token)
        # Only calls of other threads hold the pause on (a thread's nested
        # calls end with its outermost one).
        if (
            not _OURS
            or _now() - _CAUGHT_UP < CATCH_UP
            or all(other == thread for other in _ACTIVE.values())
        ):
            return None
        _CAUGHT_UP = _now()
        _CATCH_UPS += 1
        return 2 if _CATCH_UPS % FULL_EVERY == 0 else 1


def paused(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """``function`` with the collector off while it runs (module docstring:
    nested and concurrent calls, the restored state, the catch-up)."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not enabled():
            return function(*args, **kwargs)
        token = object()
        try:
            _enter(token)
            return function(*args, **kwargs)
        finally:
            generation = _leave(token)
            if generation is not None:
                # Outside the lock: a finaliser may evaluate again.
                gc.collect(generation)

    return wrapper


__all__ = ["ENVIRONMENT", "enabled", "paused"]
