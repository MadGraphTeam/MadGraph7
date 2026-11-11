# SPDX-License-Identifier: 0BSD
"""Warnings returned alongside a result (m-lit ux-besteffort).

One structure, :class:`ResultWarning`, carries every warning an operation
returns as DATA instead of printing, logging or raising it (USER 2026-11-11:
a Monte-Carlo run must never have its logs swamped, the caller decides):

- ``source="numerical"``: points of a virtual evaluation that no configured
  alternative served with an accepted value: ``kind="best-effort"`` (the
  point carries its best available value, the unchecked reduction's, with
  its uncertainty in the lane's ``reduction.best_effort`` record) and
  ``kind="failed"`` (no value at all). One entry per kind, with the count of
  points, the first point's reason and every affected point index.
- ``source="symbolica"``: a Symbolica log event of a known, benign class
  captured during the operation (``pyamplicol._internal.symbolica_log``):
  ``kind`` is Symbolica's log target and level (e.g.
  ``"WARN symbolica::coefficient"``), ``message`` its first message.

``where`` names the operation (and, for captured events, the stage) the
warning arose in. A result holds them in its ``warnings`` tuple, in
first-seen order; ``--json`` writes them as ``warnings``.

Nothing is emitted by default. :meth:`ResultWarning.emit` (and a result's
``emit_warnings()``) turns the numerical ones into ONE Python warning of the
dedicated, filterable category :class:`NumericalAccuracyWarning` per call,
for a caller that wants that.
"""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal


class NumericalAccuracyWarning(UserWarning):
    """Points of a call were returned without an accepted value: with their
    best available value and its uncertainty (status ``"best-effort"``) or
    as failed lanes, the reasons in the result's records. Never raised by
    pyAmpliCol on its own: only a caller's ``emit_warnings()`` emits it."""


WARNING_SOURCES = ("numerical", "symbolica", "ufo-model-loader")


@dataclass(frozen=True, slots=True)
class ResultWarning:
    """One warning class returned with a result: its ``source``
    (``"numerical"`` or ``"symbolica"``), ``kind`` (``"best-effort"`` /
    ``"failed"``, or the Symbolica log target), the first ``message``, the
    ``count`` of occurrences (points, or log events), ``where`` it arose
    and, for numerical warnings, the affected ``points`` (caller point
    indices)."""

    source: Literal["numerical", "symbolica", "ufo-model-loader"]
    kind: str
    message: str
    count: int
    where: str
    points: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in WARNING_SOURCES:
            raise ValueError(f"unknown warning source {self.source!r}")
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise TypeError("a warning count is an integer")
        if self.count < 1:
            raise ValueError("a warning is counted at least once")
        object.__setattr__(self, "points", tuple(int(p) for p in self.points))

    def summary(self) -> str:
        """One line: count, kind and the first message."""

        return f"{self.count} x {self.kind} ({self.where}): {self.message}"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "kind": self.kind,
            "message": self.message,
            "count": self.count,
            "where": self.where,
        }
        if self.points:
            out["points"] = list(self.points)
        return out

    @classmethod
    def from_json(cls, data: Any) -> ResultWarning:
        return cls(
            str(data["source"]),  # type: ignore[arg-type]
            str(data["kind"]),
            str(data["message"]),
            int(data["count"]),
            str(data["where"]),
            tuple(int(p) for p in data.get("points", ())),
        )


def merge_warnings(*groups: Iterable[ResultWarning]) -> tuple[ResultWarning, ...]:
    """Every warning of ``groups`` with one entry per (source, kind, where):
    counts added, points joined, the first message kept."""

    merged: dict[tuple[str, str, str], ResultWarning] = {}
    for group in groups:
        for item in group:
            key = (item.source, item.kind, item.where)
            found = merged.get(key)
            if found is None:
                merged[key] = item
                continue
            merged[key] = ResultWarning(
                found.source,
                found.kind,
                found.message,
                found.count + item.count,
                found.where,
                found.points + item.points,
            )
    return tuple(merged.values())


def numerical_warnings(
    diagnostics: Sequence[Any], where: str
) -> tuple[ResultWarning, ...]:
    """The numerical warnings of a virtual result's lane ``diagnostics``:
    one entry for the best-effort lanes and one for the failed lanes (the
    first lane's reason, every affected lane)."""

    found: dict[str, list[tuple[int, str]]] = {}
    for index, lane in enumerate(diagnostics):
        status = getattr(lane, "status", None)
        if status == "best-effort":
            best = lane.best_effort
            found.setdefault("best-effort", []).append(
                (index, best.refusal if best is not None else lane.detail)
            )
        elif status == "failed":
            failures = getattr(lane, "failures", ())
            reason = failures[0].detail if failures else lane.detail
            found.setdefault("failed", []).append((index, reason))
    out = []
    for kind in ("best-effort", "failed"):
        items = found.get(kind)
        if items:
            out.append(
                ResultWarning(
                    "numerical",
                    kind,
                    items[0][1],
                    len(items),
                    where,
                    tuple(index for index, _ in items),
                )
            )
    return tuple(out)


def warning_message(warnings: Sequence[ResultWarning]) -> str | None:
    """The one-line text of the numerical warnings of a call (None when it
    has none)."""

    numerical = [item for item in warnings if item.source == "numerical"]
    if not numerical:
        return None
    counts = ", ".join(f"{item.count} {item.kind}" for item in numerical)
    first = numerical[0]
    where = f", point {first.points[0]}" if first.points else ""
    return (
        f"{counts} point(s) without an accepted value (see result.warnings and "
        f"the lane diagnostics); first ({first.kind}{where}): {first.message}"
    )


def emit(warnings: Sequence[ResultWarning], stacklevel: int = 3) -> bool:
    """Emit the numerical ``warnings`` of one call as ONE
    :class:`NumericalAccuracyWarning` (opt-in; filterable with the standard
    :mod:`warnings` machinery); returns whether one was emitted."""

    message = warning_message(warnings)
    if message is None:
        return False
    _warnings.warn(message, NumericalAccuracyWarning, stacklevel=stacklevel)
    return True


def summary_line(warnings: Sequence[ResultWarning]) -> str | None:
    """The one dim summary line the CLI's tables print under a table (None
    without warnings)."""

    if not warnings:
        return None
    total = sum(item.count for item in warnings)
    parts = ", ".join(f"{item.count} {item.kind}" for item in warnings)
    return f"{total} captured warning(s) ({parts}); see --json / result.warnings"


class LaneWarnings:
    """The per-point status and the result-level warning metadata of a
    virtual result with lane ``diagnostics`` (mixed into
    ``VirtualEvaluation`` and ``ResolvedVirtualEvaluation``): cheap,
    computed on access, nothing emitted."""

    __slots__ = ()
    diagnostics: tuple[Any, ...]

    @property
    def statuses(self) -> tuple[str, ...]:
        """Each point's lane status: ``"success"``, ``"rescued"``,
        ``"best-effort"`` (a value without an accepted accuracy; its
        refusal and uncertainty in ``diagnostics[i].best_effort``),
        ``"failed"`` (no value), ``"unsupported"`` or
        ``"accuracy-not-established"``."""

        return tuple(lane.status for lane in self.diagnostics)

    @property
    def warnings(self) -> tuple[ResultWarning, ...]:
        """The numerical warnings of this result (``ResultWarning``: one
        entry for the best-effort points, one for the failed ones)."""

        return numerical_warnings(self.diagnostics, "evaluate")

    @property
    def has_warnings(self) -> bool:
        """Whether any point was returned without an accepted value."""

        return any(
            lane.status in ("best-effort", "failed") for lane in self.diagnostics
        )

    def emit_warnings(self) -> bool:
        """Opt-in: emit this result's numerical warnings as ONE
        :class:`NumericalAccuracyWarning` (see :func:`emit`)."""

        return emit(self.warnings, stacklevel=3)


__all__ = [
    "LaneWarnings",
    "NumericalAccuracyWarning",
    "ResultWarning",
    "emit",
    "merge_warnings",
    "numerical_warnings",
    "summary_line",
    "warning_message",
]
