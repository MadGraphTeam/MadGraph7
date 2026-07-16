# SPDX-License-Identifier: 0BSD
"""Typed progress phases for the generation service."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)


@dataclass(slots=True)
class PhaseHandle:
    task_id: str
    sink: ProgressSink | None
    total: int | None
    completed: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update(
        self,
        completed: int,
        *,
        message: str | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            self._update_locked(completed, message=message, total=total)

    def advance(self, *, message: str | None = None) -> None:
        with self._lock:
            self._update_locked(self.completed + 1, message=message, total=None)

    def _update_locked(
        self,
        completed: int,
        *,
        message: str | None,
        total: int | None,
    ) -> None:
        active_total = self.total if total is None else total
        bounded = completed
        if active_total is not None:
            bounded = min(completed, active_total)
        self.completed = max(self.completed, bounded)
        if self.sink is not None:
            self.sink.emit(
                ProgressUpdate(
                    self.task_id,
                    completed=self.completed,
                    total=active_total,
                    message=message,
                )
            )


@dataclass(slots=True)
class GenerationPhaseReporter:
    sink: ProgressSink | None
    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(
        self,
        name: str,
        description: str,
        *,
        total: int | None = None,
    ) -> Iterator[PhaseHandle]:
        task_id = f"generation:{name}"
        if self.sink is not None:
            self.sink.emit(ProgressStart(task_id, description, total=total))
        handle = PhaseHandle(task_id, self.sink, total)
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            self.timings[name] = elapsed
            if self.sink is not None:
                self.sink.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
            raise
        elapsed = time.perf_counter() - started
        self.timings[name] = elapsed
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            self.sink.emit(ProgressEnd(task_id, message=f"{elapsed:.6f}s"))


__all__ = ["GenerationPhaseReporter", "PhaseHandle"]
