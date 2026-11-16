# SPDX-License-Identifier: 0BSD
"""Typed progress phases for the generation service."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock

from pyamplicol.reporting import (
    NullProgressSink,
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)
from pyamplicol.reporting.progress import ProgressDetailValue


@dataclass(slots=True)
class PhaseTimes:
    """The disjoint top-level phases of one generation call in order
    (``collect_phase_times``): a phase that starts while another is open
    (a nested generation) is part of the open one and not recorded."""

    seconds: dict[str, float] = field(default_factory=dict)
    depth: int = 0

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed


_PHASE_TIMES: ContextVar[PhaseTimes | None] = ContextVar(
    "pyamplicol_generation_phase_times", default=None
)


@contextmanager
def collect_phase_times() -> Iterator[PhaseTimes]:
    """Collect the phases that ``GenerationPhaseReporter`` times in this
    context (the caller's thread) until the block ends."""

    times = PhaseTimes()
    token = _PHASE_TIMES.set(times)
    try:
        yield times
    finally:
        _PHASE_TIMES.reset(token)


def record_phase(name: str, seconds: float) -> None:
    """Record a step of a generation call timed outside any phase (the
    call's set-up) as a phase of its own, when phases are collected."""

    times = _PHASE_TIMES.get()
    if times is not None and times.depth == 0:
        times.add(name, seconds)


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
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> None:
        with self._lock:
            self._update_locked(
                completed,
                message=message,
                total=total,
                details=details,
            )

    def advance(
        self,
        *,
        message: str | None = None,
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> None:
        with self._lock:
            self._update_locked(
                self.completed + 1,
                message=message,
                total=None,
                details=details,
            )

    def _update_locked(
        self,
        completed: int,
        *,
        message: str | None,
        total: int | None,
        details: Mapping[str, ProgressDetailValue] | None,
    ) -> None:
        active_total = self.total if total is None else total
        if total is not None:
            self.total = total
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
                    details={} if details is None else details,
                )
            )

    @contextmanager
    def child(
        self,
        key: str,
        description: str,
        *,
        total: int | None = None,
        unit: str = "items",
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> Iterator[PhaseHandle]:
        task_id = f"{self.task_id}:{key}"
        if self.sink is not None:
            self.sink.emit(
                ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=self.task_id,
                    unit=unit,
                    details={} if details is None else details,
                )
            )
        handle = PhaseHandle(task_id, self.sink, total)
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            if self.sink is not None:
                self.sink.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=elapsed,
                    )
                )
            raise
        elapsed = time.perf_counter() - started
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            self.sink.emit(ProgressEnd(task_id, elapsed_seconds=elapsed))


@dataclass(slots=True)
class GenerationPhaseReporter:
    sink: ProgressSink | None
    timings: dict[str, float] = field(default_factory=dict)
    root: PhaseHandle | None = None
    _root_started_at: float | None = None

    def __post_init__(self) -> None:
        # Progress-off must not install callbacks into generation machinery.
        if isinstance(self.sink, NullProgressSink):
            self.sink = None

    def start_generation(
        self,
        *,
        total: int,
        details: Mapping[str, ProgressDetailValue],
    ) -> None:
        task_id = "generation"
        self._root_started_at = time.perf_counter()
        if self.sink is not None:
            self.sink.emit(
                ProgressStart(
                    task_id,
                    "Generating processes",
                    total=total,
                    unit="phases",
                    details=details,
                )
            )
        self.root = PhaseHandle(task_id, self.sink, total)

    def finish_generation(
        self,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        if self.root is None or self._root_started_at is None:
            return
        elapsed = time.perf_counter() - self._root_started_at
        if (
            success
            and self.root.total is not None
            and self.root.completed < self.root.total
        ):
            self.root.update(self.root.total)
        if self.sink is not None:
            self.sink.emit(
                ProgressEnd(
                    self.root.task_id,
                    success=success,
                    message=message,
                    elapsed_seconds=elapsed,
                )
            )
        self._root_started_at = None

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
            self.sink.emit(
                ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=None if self.root is None else self.root.task_id,
                )
            )
        handle = PhaseHandle(task_id, self.sink, total)
        times = _PHASE_TIMES.get()
        top = times is not None and times.depth == 0
        if times is not None:
            times.depth += 1
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            self._close_phase(times, top, name, elapsed)
            self.timings[name] = elapsed
            if self.sink is not None:
                self.sink.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=elapsed,
                    )
                )
            raise
        elapsed = time.perf_counter() - started
        self._close_phase(times, top, name, elapsed)
        self.timings[name] = elapsed
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            self.sink.emit(ProgressEnd(task_id, elapsed_seconds=elapsed))
        if self.root is not None:
            self.root.advance(
                message=description,
                details={"step": description},
            )

    @staticmethod
    def _close_phase(
        times: PhaseTimes | None, top: bool, name: str, elapsed: float
    ) -> None:
        if times is None:
            return
        times.depth -= 1
        if top:
            times.add(name, elapsed)

    def record(self, name: str, seconds: float) -> None:
        """Record a step timed outside ``phase`` (between two phases) as a
        phase of its own."""

        self.timings[name] = seconds
        record_phase(name, seconds)


__all__ = [
    "GenerationPhaseReporter",
    "PhaseHandle",
    "PhaseTimes",
    "collect_phase_times",
    "record_phase",
]
