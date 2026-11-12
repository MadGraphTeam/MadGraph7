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

#: The wall time one runtime call aims at: the refresh granularity of the
#: progress display and the longest wait between a Ctrl-C and the stop. The
#: first call evaluates one point; later calls take as many points as the
#: measured cost per point fits into this time.
CHUNK_SECONDS = 0.5

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
) -> Any:
    """``operation(momenta, **per_point, **options)`` over all points, in
    successive calls reported to ``progress``; on Ctrl-C the
    :class:`PartialEvaluation` of the calls completed (a Ctrl-C before the
    first one completes propagates)."""

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
    done = 0
    success = False
    try:
        try:
            size = 1
            while done < total or not parts:
                stop = min(done + size, total)
                part = operation(
                    momenta[done:stop],
                    **{name: values[done:stop] for name, values in per_point.items()},
                    **options,
                )
                parts.append(part)
                offsets.append(done)
                done = stop
                elapsed = time.perf_counter() - started
                progress.emit(
                    ProgressUpdate(
                        task,
                        done,
                        total,
                        details={"step": f"{done} of {total} point(s)"},
                    )
                )
                size = max(1, int(CHUNK_SECONDS * done / max(elapsed, 1e-9)))
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
