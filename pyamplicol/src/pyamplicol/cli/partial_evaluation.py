# SPDX-License-Identifier: 0BSD
"""``pyamplicol evaluate`` over its points in successive runtime calls, so
that the CLI can show the points done and, on Ctrl-C, print the tables of
the points evaluated so far (m-lit ux-cli C9).

Each point is evaluated independently of the batch it is in (the native
lanes are independent; the stability run's N=1 vs N=4 identity relies on the
same property), so the concatenated result is the one-call result: the
per-point fields are concatenated in point order and every ``point_id`` of
a lane's diagnostics, which the runtime numbers from 0 within a call, is
shifted by the chunk's first point.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from pyamplicol.reporting.progress import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)

@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    """How ``evaluate`` sizes its successive calls on a live display.

    These are display tuning values, not physics: every point is evaluated
    independently of the call it is in, so they change only how often the
    bar moves, how soon a Ctrl-C is honoured and the small extra time of the
    calls' fixed cost, never a value. Pass another policy to
    :func:`evaluate_in_chunks` to override them.
    """

    #: The shortest wall time a later call aims at: the refresh granularity
    #: of the progress display and the usual wait between a Ctrl-C and the
    #: stop.
    seconds: float = 2.0
    #: The largest share of the evaluation time spent on the fixed cost of
    #: the extra calls. A runtime call has a fixed cost (about 0.1 s on
    #: u u~ > t t~ g g) on top of its cost per point, so a call aims at no
    #: less than ``fixed / overhead`` seconds.
    overhead: float = 0.03
    #: The most points of the second call, the first one measured warm.
    probe_points: int = 8
    #: The most a call grows over the largest call measured so far, in
    #: points; it also never aims beyond one and a half target times by the
    #: measured cost of that call, so one mis-measured call cannot make the
    #: next one swallow the evaluation.
    growth: float = 4.0


DEFAULT_CHUNK_POLICY = ChunkPolicy()


def _cost_model(warm: Sequence[tuple[int, float]]) -> tuple[float, float]:
    """``(fixed, per_point)`` seconds of one call from the warm calls
    ``(points, seconds)`` (every call but the first, which carries the
    first-call cost). The least-squares line when it rises; otherwise no
    fixed cost and the best measured average per point, an upper bound of
    the cost per point."""

    average = min(seconds / points for points, seconds in warm)
    if len({points for points, _ in warm}) >= 2:
        n = len(warm)
        mean_x = sum(points for points, _ in warm) / n
        mean_y = sum(seconds for _, seconds in warm) / n
        sxx = sum((points - mean_x) ** 2 for points, _ in warm)
        sxy = sum((points - mean_x) * (seconds - mean_y) for points, seconds in warm)
        slope = sxy / sxx
        if slope > 0.0:
            cheapest = min(seconds for _, seconds in warm)
            fixed = min(max(mean_y - slope * mean_x, 0.0), cheapest)
            return fixed, slope
    return 0.0, average


def _next_size(
    calls: Sequence[tuple[int, float]],
    remaining: int,
    policy: ChunkPolicy = DEFAULT_CHUNK_POLICY,
) -> int:
    """The points of the next call: one point first, a few more measured
    warm, then calls of about ``max(seconds, fixed / overhead)`` each,
    growing at most by ``growth`` and by the measured cost of the largest
    call; the rest goes in one call once it fits in one and a half such
    calls within that bound."""

    if remaining <= 0:
        return 0
    if not calls:
        return 1
    if len(calls) == 1:
        seconds = calls[0][1]
        wanted = int(policy.seconds / seconds) if seconds > 0 else policy.probe_points
        return min(remaining, max(2, min(policy.probe_points, wanted)))
    warm = calls[1:]
    fixed, per_point = _cost_model(warm)
    target = max(policy.seconds, fixed / policy.overhead)
    largest_points, largest_seconds = max(warm)
    factor = policy.growth
    if largest_seconds > 0:
        factor = min(factor, 1.5 * target / largest_seconds)
    cap = max(1, int(largest_points * max(1.0, factor)))
    if remaining <= cap and fixed + per_point * remaining <= 1.5 * target:
        return remaining
    wanted = max(1, int((target - fixed) / per_point))
    return min(remaining, cap, wanted)


# The per-point fields of a VirtualEvaluation (concatenated) and the fields
# common to every point (equal across the chunks). Every field is listed in
# one of the two, so a field added later fails loudly instead of merging
# silently.
_VIRTUAL_PER_POINT = (
    "born_values",
    "values",
    "diagnostics",
    "stability",
    "pole_parts",
    "rescue",
)
_VIRTUAL_COMMON = (
    "born_orders",
    "sectors",
    "r2",
    "correlated",
    "color_accuracy",
    "poles",
    "contribution",
)


@dataclass(frozen=True, slots=True)
class PartialEvaluation:
    """An evaluate result over the first ``measured`` of ``requested``
    points, interrupted by Ctrl-C."""

    result: Any
    requested: int
    measured: int
    interrupted: bool = True

    def notice(self) -> str:
        return (
            f"interrupted: {self.measured} of {self.requested} requested points "
            "evaluated; the tables hold those points"
        )


def _shift_point_ids(value: Any, offset: int) -> Any:
    """``value`` with every dataclass field ``point_id`` shifted by
    ``offset`` (lane diagnostics number their points within one call)."""

    if offset == 0:
        return value
    if isinstance(value, tuple):
        shifted = tuple(_shift_point_ids(item, offset) for item in value)
        return (
            value
            if all(a is b for a, b in zip(shifted, value, strict=True))
            else shifted
        )
    if is_dataclass(value) and not isinstance(value, type):
        changes: dict[str, Any] = {}
        for item in fields(value):
            if not item.init:
                continue
            current = getattr(value, item.name)
            if item.name == "point_id" and isinstance(current, int):
                changes[item.name] = current + offset
                continue
            shifted = _shift_point_ids(current, offset)
            if shifted is not current:
                changes[item.name] = shifted
        return replace(value, **changes) if changes else value
    return value


def concatenate(parts: Sequence[Any], offsets: Sequence[int]) -> Any:
    """One result from the results of successive calls over consecutive
    point ranges starting at ``offsets``."""

    from pyamplicol.api.virtual_results import VirtualEvaluation

    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if isinstance(first, tuple):
        return tuple(value for part in parts for value in part)
    if type(first) is not VirtualEvaluation:
        raise TypeError(f"cannot concatenate {type(first).__name__} results")
    names = {item.name for item in fields(first)}
    if names != set(_VIRTUAL_PER_POINT) | set(_VIRTUAL_COMMON):
        raise TypeError("VirtualEvaluation fields changed: update concatenate")
    for part in parts[1:]:
        for name in _VIRTUAL_COMMON:
            if getattr(part, name) != getattr(first, name):
                raise ValueError(f"successive evaluations differ in {name}")
    changes: dict[str, Any] = {}
    for name in _VIRTUAL_PER_POINT:
        values = [getattr(part, name) for part in parts]
        if all(value is None for value in values):
            changes[name] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"successive evaluations differ in {name}")
        changes[name] = tuple(
            _shift_point_ids(item, offset)
            for value, offset in zip(values, offsets, strict=True)
            for item in value
        )
    return replace(first, **changes)


def evaluate_in_chunks(
    progress: ProgressSink,
    operation: Callable[..., Any],
    momenta: Sequence[Any],
    per_point: dict[str, Sequence[Any]],
    options: dict[str, Any],
    policy: ChunkPolicy = DEFAULT_CHUNK_POLICY,
) -> Any:
    """``operation(momenta, **per_point, **options)`` over all points, in
    successive calls sized by ``policy`` and reported to ``progress``; on Ctrl-C the
    :class:`PartialEvaluation` of the calls completed (a Ctrl-C before the
    first one completes propagates). A call that raises is repeated as one
    call over all points, so the error is the one-call error."""

    total = len(momenta)
    task = "evaluate-points"
    started = time.perf_counter()
    progress.emit(
        ProgressStart(
            task,
            f"Evaluating {total} point(s)",
            total=total,
            unit="points",
            details={"step": f"evaluating {total} point(s)"},
        )
    )
    parts: list[Any] = []
    offsets: list[int] = []
    calls: list[tuple[int, float]] = []
    done = 0
    success = False
    try:
        try:
            while done < total or not parts:
                stop = min(done + _next_size(calls, total - done, policy), total)
                began = time.perf_counter()
                try:
                    part = operation(
                        momenta[done:stop],
                        **{name: v[done:stop] for name, v in per_point.items()},
                        **options,
                    )
                except Exception:
                    if done == 0 and stop == total:
                        raise
                    # The one-call failure, raised by one call: the runtime
                    # names a failing point by its index within the call.
                    result = operation(momenta, **per_point, **options)
                    success = True
                    return result
                calls.append((stop - done, time.perf_counter() - began))
                parts.append(part)
                offsets.append(done)
                done = stop
                progress.emit(
                    ProgressUpdate(
                        task,
                        done,
                        total,
                        details={"step": f"{done} of {total} point(s)"},
                    )
                )
        except KeyboardInterrupt:
            if not parts:
                raise
            return PartialEvaluation(concatenate(parts, offsets), total, done)
        success = True
        return concatenate(parts, offsets)
    finally:
        progress.emit(
            ProgressEnd(
                task,
                success=success,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
