# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import contextlib
import importlib
import io
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from types import MappingProxyType
from typing import Any, Literal, Protocol, TextIO, TypeAlias, runtime_checkable

from .eta import Estimate, combined, count_estimate, remaining_estimate
from .logging import get_logger
from .resources import ResourceUsage, ResourceUsageMonitor

ProgressDetailValue: TypeAlias = str | int | float | bool | None
ProgressDetails: TypeAlias = Mapping[str, ProgressDetailValue]


def terminal_capable(stream: TextIO) -> bool:
    """Whether ``stream`` is a terminal that takes cursor movement: a tty
    whose ``TERM`` is not "dumb" (a dumb or unknown-capability terminal
    gets the line display)."""

    try:
        tty = bool(getattr(stream, "isatty", lambda: False)())
    except (OSError, ValueError):
        return False
    return tty and os.environ.get("TERM", "") != "dumb"


def color_wanted(stream: TextIO) -> bool:
    """``--color auto``: colour on a capable terminal, unless ``NO_COLOR``
    is set (to anything but the empty string, https://no-color.org)."""

    return not os.environ.get("NO_COLOR") and terminal_capable(stream)


# A display failure is reported once per process; the command goes on.
_DISPLAY_WARNED = Event()


def _display_failed(stream: TextIO | None, error: BaseException, fallback: str) -> None:
    """Report a failure of the progress display (one line, once), which
    continues as ``fallback`` ("plain progress lines", "no progress")."""

    if _DISPLAY_WARNED.is_set():
        return
    _DISPLAY_WARNED.set()
    text = (
        f"warning: the progress display failed ({type(error).__name__}: {error}); "
        f"continuing with {fallback}\n"
    )
    with contextlib.suppress(Exception):
        target = stream if stream is not None else sys.stderr
        target.write(text)
        target.flush()


def _terminal_paint(text: str, color_name: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    try:
        colorama: Any = importlib.import_module("colorama")
    except ImportError:
        return text
    return f"{getattr(colorama.Fore, color_name)}{text}{colorama.Style.RESET_ALL}"


def _task_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("progress task_id must be a non-empty string")
    return value


def _total(value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError("progress total must be a non-negative integer or null")
    return value


def _details(value: ProgressDetails | None) -> ProgressDetails:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("progress details must be a mapping")
    result: dict[str, ProgressDetailValue] = {}
    for key, detail in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("progress detail names must be non-empty strings")
        if not isinstance(detail, (str, int, float, bool, type(None))):
            raise ValueError(
                f"progress detail {key!r} has unsupported type {type(detail).__name__}"
            )
        if isinstance(detail, float) and not math.isfinite(detail):
            raise ValueError(f"progress detail {key!r} must be finite")
        result[key] = detail
    return MappingProxyType(result)


def _elapsed(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("progress elapsed_seconds must be a number or null")
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("progress elapsed_seconds must be finite and non-negative")
    return elapsed


@dataclass(frozen=True, slots=True)
class ProgressStart:
    task_id: str
    description: str
    total: int | None = None
    parent_task_id: str | None = None
    unit: str = "items"
    details: ProgressDetails = field(default_factory=dict)

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("progress description must be a non-empty string")
        _total(self.total)
        if self.parent_task_id is not None:
            _task_id(self.parent_task_id)
            if self.parent_task_id == self.task_id:
                raise ValueError("progress task may not be its own parent")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("progress unit must be a non-empty string")
        object.__setattr__(self, "details", _details(self.details))


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    task_id: str
    completed: int
    total: int | None = None
    message: str | None = None
    details: ProgressDetails = field(default_factory=dict)

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if (
            isinstance(self.completed, bool)
            or not isinstance(self.completed, int)
            or self.completed < 0
        ):
            raise ValueError("progress completed must be a non-negative integer")
        total = _total(self.total)
        if total is not None and self.completed > total:
            raise ValueError("progress completed may not exceed total")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("progress message must be a string or null")
        object.__setattr__(self, "details", _details(self.details))


@dataclass(frozen=True, slots=True)
class ProgressEnd:
    task_id: str
    success: bool = True
    message: str | None = None
    elapsed_seconds: float | None = None
    details: ProgressDetails = field(default_factory=dict)

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if not isinstance(self.success, bool):
            raise ValueError("progress success must be a boolean")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("progress message must be a string or null")
        object.__setattr__(self, "elapsed_seconds", _elapsed(self.elapsed_seconds))
        object.__setattr__(self, "details", _details(self.details))


ProgressEvent: TypeAlias = ProgressStart | ProgressUpdate | ProgressEnd


@runtime_checkable
class ProgressSink(Protocol):
    def emit(self, event: ProgressEvent) -> None:
        """Consume one typed progress event."""


@dataclass(frozen=True, slots=True)
class CallbackProgressSink:
    callback: Callable[[ProgressEvent], None]

    def emit(self, event: ProgressEvent) -> None:
        self.callback(event)


@dataclass(frozen=True, slots=True)
class NullProgressSink:
    def emit(self, event: ProgressEvent) -> None:
        del event


@dataclass(slots=True)
class LoggingProgressSink:
    logger: logging.Logger = field(default_factory=lambda: get_logger("progress"))
    level: int = logging.INFO
    minimum_interval: float = 1.0
    color: bool = False
    _last_emit: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def emit(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        with self._lock:
            if isinstance(event, ProgressUpdate):
                last = self._last_emit.get(event.task_id, float("-inf"))
                complete = event.total is not None and event.completed == event.total
                if not complete and now - last < self.minimum_interval:
                    return
            self.logger.log(self.level, _format_event(event, color=self.color))
            self._last_emit[event.task_id] = now
            if isinstance(event, ProgressEnd):
                self._last_emit.pop(event.task_id, None)


@dataclass(frozen=True, slots=True)
class StreamProgressSink:
    stream: TextIO

    def emit(self, event: ProgressEvent) -> None:
        self.stream.write(f"{_format_event(event)}\n")
        self.stream.flush()


@dataclass(slots=True)
class TtyProgressSink:
    """Render a thread-safe adaptive three-line progress dashboard."""

    stream: TextIO
    color: bool | None = None
    _tasks: dict[str, _TaskState] = field(default_factory=dict, init=False, repr=False)
    _dashboard: _Dashboard | _SidecarDashboard | None = field(
        default=None, init=False, repr=False
    )
    _line_sink: _LineProgressSink | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _output_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _message_stream: _TtyMessageStream = field(init=False, repr=False)
    # 0: the live display; 1: plain lines after a display failure; 2: none
    # after a failure of those too (the command never fails for its display).
    _degraded: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._message_stream = _TtyMessageStream(self)

    def emit(self, event: ProgressEvent) -> None:
        if self._degraded >= 2:
            return
        try:
            broken = self._dashboard is not None and self._dashboard.broken
            if broken is not None and broken is not False:
                raise broken
            self._emit(event)
        except Exception as error:  # the display never fails the work
            self._degrade(error, event)

    def _degrade(self, error: BaseException, event: ProgressEvent | None) -> None:
        """A failure of the display: the live display gives way to plain
        lines, plain lines to none; one warning line."""

        with self._lock:
            dashboard, self._dashboard = self._dashboard, None
            level = self._degraded
            self._degraded = level + 1 if level < 2 else 2
        if dashboard is not None:
            with contextlib.suppress(Exception):
                dashboard.close()
        if level == 0:
            _display_failed(self.stream, error, "plain progress lines")
            if event is not None:
                with contextlib.suppress(Exception):
                    self._line().emit(event)
        else:
            _display_failed(self.stream, error, "no progress display")
            self._degraded = 2

    def _line(self) -> _LineProgressSink:
        if self._line_sink is None:
            self._line_sink = _LineProgressSink(
                self.stream,
                color=False,
                minimum_interval=2.0,
                heartbeat=True,
            )
        return self._line_sink

    def _emit(self, event: ProgressEvent) -> None:
        dashboard: _Dashboard | _SidecarDashboard | None = None
        start_dashboard = False
        stop_dashboard = False
        kept: str | None = None
        with self._lock:
            if not self._interactive:
                self._line().emit(event)
                return
            now = time.monotonic()
            if isinstance(event, ProgressStart):
                if event.parent_task_id is None and not event.details.get(
                    "placeholder"
                ):
                    # The command's own first task replaces the command's
                    # placeholder ("pyamplicol generate ... starting").
                    for task_id in [
                        key
                        for key, task in self._tasks.items()
                        if task.details.get("placeholder")
                    ]:
                        del self._tasks[task_id]
                existing = self._tasks.get(event.task_id)
                if existing is not None and existing.finished_at is None:
                    # A task started twice: the later start wins (the
                    # display does not police its callers).
                    existing.finished_at = now
                self._tasks[event.task_id] = _TaskState(
                    task_id=event.task_id,
                    description=event.description,
                    total=event.total,
                    parent_task_id=event.parent_task_id,
                    unit=event.unit,
                    details=dict(event.details),
                    started_at=now,
                    updated_at=now,
                )
                if self._dashboard is None:
                    self._dashboard = _new_dashboard(
                        stream=self.stream,
                        color=self._color_enabled,
                        snapshot=self._snapshot,
                        render_lock=self._output_lock,
                    )
                    start_dashboard = True
            else:
                task = self._tasks.get(event.task_id)
                if task is None:
                    # Keep custom sinks forgiving when a backend emits a late pulse.
                    return
                if isinstance(event, ProgressUpdate):
                    moved = event.completed != task.completed
                    task.record(event.completed, now)
                    task.completed = max(task.completed, event.completed)
                    task.total = event.total if event.total is not None else task.total
                    if event.message is not None or moved:
                        # An update of details only (a rescaled prior)
                        # keeps what the task said.
                        task.message = event.message
                    task.details.update(event.details)
                    task.updated_at = now
                else:
                    task.success = event.success
                    task.message = event.message
                    task.details.update(event.details)
                    task.finished_at = now
                    task.updated_at = now
                    if event.elapsed_seconds is not None:
                        task.elapsed_seconds = event.elapsed_seconds
                    if event.details.get("keep") and self._dashboard is not None:
                        kept = _INDENT * _depth(self._tasks, task) + _kept_line(
                            task, color=self._color_enabled
                        )
            dashboard = self._dashboard
            if (
                isinstance(event, ProgressEnd)
                and dashboard is not None
                and all(task.finished_at is not None for task in self._tasks.values())
            ):
                self._dashboard = None
                stop_dashboard = True

        # Rendering snapshots task state and therefore acquires ``self._lock``.
        # Keep it outside that lock so the heartbeat's render lock and event
        # delivery can never form a lock-order cycle.
        if dashboard is not None and kept is not None:
            # A finished phase's one line stays in the scroll-back.
            dashboard.write_message(kept)
        if dashboard is not None:
            if stop_dashboard:
                dashboard.close()
            elif start_dashboard:
                dashboard.start()
            else:
                dashboard.render_now(force=isinstance(event, ProgressStart))

    def close(self) -> None:
        with self._lock:
            dashboard = self._dashboard
            self._dashboard = None
            line_sink = self._line_sink
        if dashboard is not None:
            try:
                dashboard.close()
            except Exception as error:  # never fail the command
                _display_failed(self.stream, error, "no progress display")
        if line_sink is not None:
            with contextlib.suppress(Exception):
                line_sink.close()

    @property
    def message_stream(self) -> TextIO:
        """A logging stream that preserves dashboard line ownership."""

        return self._message_stream

    def _write_message(self, text: str) -> int:
        with self._lock:
            dashboard = self._dashboard
        if dashboard is not None:
            try:
                dashboard.write_message(text)
                return len(text)
            except Exception as error:  # the message still goes out
                self._degrade(error, None)
        with self._output_lock:
            self.stream.write(text)
        return len(text)

    def _flush_messages(self) -> None:
        with self._output_lock:
            self.stream.flush()

    @property
    def _interactive(self) -> bool:
        return self._degraded == 0 and terminal_capable(self.stream)

    @property
    def live(self) -> bool:
        """Whether this sink draws a live display on a terminal."""

        return self._interactive

    @property
    def _color_enabled(self) -> bool:
        if self.color is not None:
            return self.color
        return color_wanted(self.stream)

    def _snapshot(self) -> tuple[_TaskState, ...]:
        with self._lock:
            return tuple(task.copy() for task in self._tasks.values())


_MARKS_KEPT = 64
# The silence after which the detail line says "no progress for", and the
# busy cores below which the process tree counts as idle.
_QUIET_SECONDS = 5.0
_IDLE_CORES = 0.05
_MARKS_HEAD = 4


@dataclass(slots=True)
class _TaskState:
    task_id: str
    description: str
    total: int | None
    parent_task_id: str | None
    unit: str
    details: dict[str, ProgressDetailValue]
    started_at: float
    updated_at: float
    completed: int = 0
    message: str | None = None
    success: bool | None = None
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    # Completion marks [time, completed] of the counted items (the ETA
    # estimator's input, ``eta.count_estimate``): the first ones and the
    # recent ones.
    marks: list[list[float]] = field(default_factory=list)

    def copy(self) -> _TaskState:
        return _TaskState(
            task_id=self.task_id,
            description=self.description,
            total=self.total,
            parent_task_id=self.parent_task_id,
            unit=self.unit,
            details=dict(self.details),
            started_at=self.started_at,
            updated_at=self.updated_at,
            completed=self.completed,
            message=self.message,
            success=self.success,
            finished_at=self.finished_at,
            elapsed_seconds=self.elapsed_seconds,
            marks=[list(mark) for mark in self.marks],
        )

    def record(self, completed: int, now: float) -> None:
        """Keep a completion mark when the count moves (at most
        ``_MARKS_KEPT``: the first marks and the most recent ones)."""

        if completed > self.completed or (not self.marks and completed > 0):
            self.marks.append([now, completed])
            if len(self.marks) > _MARKS_KEPT:
                del self.marks[_MARKS_HEAD]


# The longest silence of a line log while work runs, by the run's age: a
# status line after 2 s in the first 15 s (a start is never silent), 5 s in
# the first minute, then every 10 s, 20 s from 5 minutes, 30 s from 20
# minutes (``_LineProgressSink`` heartbeat).
_LOG_CADENCE = (
    (15.0, 2.0),
    (60.0, 5.0),
    (300.0, 10.0),
    (1200.0, 20.0),
    (math.inf, 30.0),
)
# A step's start line is written once it has run this long (a step that
# ends sooner leaves no line, unless it fails).
_LOG_START_DELAY = 1.0


def _track_event(
    states: dict[str, _TaskState], event: ProgressEvent, now: float
) -> _TaskState | None:
    """Keep a task's state from its events (a line sink's heartbeat and
    an interruption's whereabouts)."""

    if isinstance(event, ProgressStart):
        states[event.task_id] = _TaskState(
            task_id=event.task_id,
            description=event.description,
            total=event.total,
            parent_task_id=event.parent_task_id,
            unit=event.unit,
            details=dict(event.details),
            started_at=now,
            updated_at=now,
        )
        return states[event.task_id]
    state = states.get(event.task_id)
    if state is None:
        return None
    if isinstance(event, ProgressUpdate):
        state.record(event.completed, now)
        state.completed = max(state.completed, event.completed)
        state.total = event.total if event.total is not None else state.total
        state.details.update(event.details)
        state.updated_at = now
    else:
        state.success = event.success
        state.finished_at = now
        state.updated_at = now
    return state


@dataclass(slots=True)
class _LineProgressSink:
    """One plain line per progress event (``--progress log`` and a
    non-terminal ``tty``): starts and ends of the top tasks always, of a
    sub-task once it has run ``_LOG_START_DELAY`` (its end then too),
    updates at most every ``minimum_interval`` seconds per task (and at
    completion), with the estimated time left. With ``heartbeat``, a
    status line of the running task (count, "no progress for", ETA,
    elapsed, memory, busy cores) whenever the log has been silent for the
    ``_LOG_CADENCE`` interval."""

    stream: TextIO
    color: bool = False
    descriptions: dict[str, str] = field(default_factory=dict)
    minimum_interval: float = 0.0
    _last: dict[str, float] = field(default_factory=dict)

    # The cadence of a task's heartbeat updates (``details["heartbeat"]``:
    # the same counts, its clocks moved on) in a line log: alive, not noisy.
    heartbeat_interval: float = 10.0
    heartbeat: bool = False
    _states: dict[str, _TaskState] = field(default_factory=dict)
    _announced: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock)
    _last_line: float = field(default_factory=time.monotonic)
    _thread: Thread | None = None
    _stop: Event = field(default_factory=Event)
    _monitor: ResourceUsageMonitor | None = None
    # The process tree the status lines measure (a renderer process: its
    # parent, without itself).
    monitor_root: int | None = None
    # Whether the stream is still this log's to write to (a renderer
    # process: its parent is alive).
    alive: Callable[[], bool] | None = None
    # A failure of the log display switches it off (the command goes on).
    _disabled: bool = False

    def emit(self, event: ProgressEvent) -> None:
        if self._disabled:
            return
        try:
            with self._lock:
                self._emit_locked(event)
        except Exception as error:  # the display never fails the work
            self._disable(error)

    def _disable(self, error: BaseException) -> None:
        self._disabled = True
        _display_failed(self.stream, error, "no progress display")
        with contextlib.suppress(Exception):
            self._stop_heartbeat()

    def _emit_locked(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        before = self._states.get(event.task_id)
        unchanged = (
            isinstance(event, ProgressUpdate)
            and before is not None
            and event.completed <= before.completed
            and event.message is None
            and not any(
                key in event.details
                for key in ("readings", "progress_kind", "step", "heartbeat")
            )
        )
        state = self._track(event, now)
        if unchanged:
            # Details only (a root's priors): nothing a reader would see move.
            return
        if isinstance(event, ProgressUpdate):
            complete = event.total is not None and event.completed == event.total
            last = self._last.get(event.task_id, float("-inf"))
            interval = (
                self.heartbeat_interval
                if event.details.get("heartbeat")
                else self.minimum_interval
            )
            if not complete and now - last < interval:
                return
            self._last[event.task_id] = now
            if (
                state is not None
                and "eta_seconds" not in event.details
                and not complete
            ):
                estimate = _task_estimate(state, now)
                if estimate is not None and estimate.seconds is not None:
                    event = ProgressUpdate(
                        event.task_id,
                        event.completed,
                        event.total,
                        event.message,
                        {
                            **event.details,
                            "eta_seconds": estimate.seconds,
                            "eta_estimated": estimate.estimated,
                        },
                    )
        if isinstance(event, ProgressStart):
            self.descriptions[event.task_id] = event.description
            if self.heartbeat and event.parent_task_id is not None:
                # Announced once it has run a while (``_announce``).
                return
            self._announced.add(event.task_id)
        elif event.task_id not in self._announced:
            if isinstance(event, ProgressUpdate):
                self._announce(event.task_id, now)
            elif event.success and not event.details.get("keep"):
                self._forget(event.task_id)
                return
        description = self.descriptions.get(event.task_id)
        self._write(
            self._indent(event.task_id) + _format_event(event, description=description),
            event,
        )
        if isinstance(event, ProgressEnd):
            self._forget(event.task_id)

    def _indent(self, task_id: str) -> str:
        state = self._states.get(task_id)
        return "" if state is None else _INDENT * _depth(self._states, state)

    def _write(self, rendered: str, event: ProgressEvent | None = None) -> None:
        color_name = (
            "RED"
            if isinstance(event, ProgressEnd) and not event.success
            else "LIGHTBLACK_EX"
            if event is None
            else "CYAN"
        )
        if self.alive is not None and not self.alive():
            self._disabled = True
            return
        painted = _terminal_paint(rendered, color_name, enabled=self.color)
        self.stream.write(f"{painted}\n")
        self.stream.flush()
        self._last_line = time.monotonic()

    def _forget(self, task_id: str) -> None:
        self.descriptions.pop(task_id, None)
        self._last.pop(task_id, None)
        self._announced.discard(task_id)
        state = self._states.pop(task_id, None)
        if state is not None and not any(
            task.finished_at is None for task in self._states.values()
        ):
            self._states.clear()
            self._stop_heartbeat()

    def _track(self, event: ProgressEvent, now: float) -> _TaskState | None:
        if not self.heartbeat:
            return None
        state = _track_event(self._states, event, now)
        if isinstance(event, ProgressStart):
            self._start_heartbeat()
        return state

    def _announce(self, task_id: str, now: float) -> None:
        state = self._states.get(task_id)
        if state is None or task_id in self._announced:
            return
        parent = state.parent_task_id
        if parent is not None and parent not in self._announced:
            self._announce(parent, now)
        self._announced.add(task_id)
        self._write(
            self._indent(task_id)
            + _format_event(
                ProgressStart(
                    task_id,
                    state.description,
                    total=state.total,
                    parent_task_id=state.parent_task_id,
                )
            )
        )

    # --- heartbeat --------------------------------------------------------
    def _start_heartbeat(self) -> None:
        if self._thread is not None:
            return
        self._stop = Event()
        if self._monitor is None:
            self._monitor = ResourceUsageMonitor(
                root_pid=self.monitor_root,
                interval_seconds=1.0,
                exclude=frozenset({os.getpid()})
                if self.monitor_root is not None
                else frozenset(),
            )
            self._monitor.start()
        self._thread = Thread(
            target=self._beat, name="pyamplicol-progress-log-heartbeat", daemon=True
        )
        self._thread.start()

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None

    def close(self) -> None:
        with self._lock:
            self._states.clear()
        self._stop_heartbeat()

    def _beat(self) -> None:
        while not self._stop.wait(0.5):
            if self._disabled:
                return
            try:
                self._beat_once()
            except Exception as error:  # the log lines go on
                self._disabled = True
                _display_failed(self.stream, error, "no progress display")
                return

    def _beat_once(self) -> None:
        with self._lock:
            now = time.monotonic()
            for state in list(self._states.values()):
                if (
                    state.finished_at is None
                    and state.parent_task_id is not None
                    and state.task_id not in self._announced
                    and now - state.started_at >= _LOG_START_DELAY
                ):
                    self._announce(state.task_id, now)
            active = [t for t in self._states.values() if t.finished_at is None]
            if not active:
                return
            roots = [t for t in active if t.parent_task_id is None] or active
            age = now - min(t.started_at for t in roots)
            cadence = next(limit for bound, limit in _LOG_CADENCE if age < bound)
            if now - self._last_line < cadence:
                return
            self._write(self._status_line(active, now))

    def _status_line(self, active: list[_TaskState], now: float) -> str:
        """The running task's status: the deepest running task (by start),
        its count, the silence since the last event, its ETA, the run's
        elapsed time, memory and busy cores."""

        parents = {task.parent_task_id for task in active}
        leaves = [task for task in active if task.task_id not in parents] or active
        task = max(leaves, key=lambda item: item.started_at)
        roots = [t for t in active if t.parent_task_id is None] or active
        parts: list[str] = []
        if task.total is not None:
            parts.append(f"{min(task.completed, task.total)}/{task.total}")
        quiet = now - max(t.updated_at for t in self._states.values())
        parts.append(
            "running"
            if quiet < _QUIET_SECONDS
            else f"no progress for {_duration_text(quiet)}"
        )
        estimate = None if "eta_seconds" in task.details else _task_estimate(task, now)
        counted = _counted_down(task, "eta_seconds", now)
        if counted is not None:
            estimated = bool(task.details.get("eta_estimated"))
            parts.append(f"ETA {_eta_text(max(counted, 0.0), estimated=estimated)}")
        elif estimate is not None:
            if estimate.seconds is None:
                parts.append(f"overdue {_duration_text(estimate.overdue)}")
            elif estimate.at_least:
                parts.append(f"ETA >{_eta_text(estimate.seconds, estimated=False)}")
            else:
                text = _eta_text(estimate.seconds, estimated=estimate.estimated)
                parts.append(f"ETA {text}")
        parts.append(
            f"elapsed {_duration_text(now - min(t.started_at for t in roots))}"
        )
        usage = self._monitor.usage if self._monitor is not None else None
        if usage is not None and usage.current_rss_bytes is not None:
            parts.append(f"RSS {_rss_text(usage)}")
        if usage is not None and usage.busy_cores is not None:
            parts.append(
                "idle"
                if usage.busy_cores < _IDLE_CORES
                else f"cores {usage.busy_cores:.1f}"
            )
        return self._indent(task.task_id) + f"{task.description}: " + "; ".join(parts)


class _TtyMessageStream(io.TextIOBase, TextIO):
    def __init__(self, sink: TtyProgressSink) -> None:
        self._sink = sink

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._sink._interactive

    def fileno(self) -> int:
        return self._sink.stream.fileno()

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("TTY progress messages must be strings")
        return self._sink._write_message(text)

    def flush(self) -> None:
        self._sink._flush_messages()


class _Dashboard:
    def __init__(
        self,
        *,
        stream: TextIO,
        color: bool,
        snapshot: Callable[[], tuple[_TaskState, ...]],
        render_lock: RLock,
        resource_monitor: ResourceUsageMonitor | None = None,
        alive: Callable[[], bool] | None = None,
    ) -> None:
        self._stream = stream
        self._color = color
        self._snapshot = snapshot
        self._resource_monitor = resource_monitor or ResourceUsageMonitor(
            interval_seconds=1.0
        )
        self._stop = Event()
        self._thread: Thread | None = None
        self._render_lock = render_lock
        self._painted = False
        self._closed = False
        self._last_render = float("-inf")
        # Whether the terminal is still the drawing process's to draw on (a
        # renderer process: its parent is alive); once not, nothing more is
        # written, not even a clear.
        self._alive = alive or (lambda: True)
        # The failure of a heartbeat redraw, for the sink to degrade on.
        self.broken: BaseException | None = None

    def start(self) -> None:
        self._resource_monitor.start()
        self.render_now()
        self._thread = Thread(
            target=self._run,
            name="pyamplicol-progress-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._resource_monitor.close()
        with self._render_lock:
            self._closed = True
            if self._alive():
                self._clear_locked()
            self._stream.flush()

    def abandon(self) -> None:
        """Stop drawing and leave the screen as it is (the terminal is no
        longer this display's: nothing is erased)."""

        self._stop.set()
        with self._render_lock:
            self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._resource_monitor.close()

    def render_now(self, *, force: bool = False) -> None:
        # Event-driven redraws at most every 50 ms (a task reporting each of
        # thousands of items); the heartbeat thread draws the rest. A task
        # that starts is drawn at once (``force``), however short it is.
        now = time.monotonic()
        if not force and now - self._last_render < 0.05:
            return
        with self._render_lock:
            if not self._closed:
                self._render_locked()

    def write_message(self, text: str) -> None:
        with self._render_lock:
            if self._closed:
                self._stream.write(text)
                return
            self._clear_locked()
            self._stream.write(text)
            if text and not text.endswith(("\n", "\r")):
                self._stream.write("\n")
            self._render_locked()

    def _render_locked(self) -> None:
        # Keeping the final column empty avoids delayed-wrap terminals turning
        # one logical dashboard row into two physical rows.
        if not self._alive():
            self._closed = True
            return
        self._last_render = time.monotonic()
        width = max(_terminal_width(self._stream) - 1, 1)
        lines = _dashboard_lines(
            self._snapshot(),
            now=time.monotonic(),
            usage=self._resource_monitor.usage,
            width=width,
            color=self._color,
        )
        if self._painted:
            self._stream.write(f"\x1b[{len(lines)}F")
        for line in lines:
            self._stream.write(f"\r\x1b[2K{line}\n")
        self._painted = True
        self._stream.flush()

    def _clear_locked(self) -> None:
        if not self._painted:
            return
        self._stream.write("\x1b[3F\r\x1b[J")
        self._painted = False

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.render_now()
            except Exception as error:  # reported to the sink
                self.broken = error
                return


# The renderer process: the dashboard is drawn by a small child process fed
# with the task states over a pipe, so it keeps moving (clocks, resources,
# "no progress for") while the main process is inside native code that
# holds the GIL (a Symbolica expression load of 25 s, for example) and the
# terminal is restored even when the main process dies.
_RENDERER_ENV = "PYAMPLICOL_PROGRESS_RENDERER"
_TASK_FIELDS = tuple(name for name in _TaskState.__dataclass_fields__)
_RENDERER_BOOTSTRAP = (
    "import sys, types\n"
    "for name, path in (('pyamplicol', sys.argv[1]), "
    "('pyamplicol.reporting', sys.argv[2])):\n"
    "    module = types.ModuleType(name)\n"
    "    module.__path__ = [path]\n"
    "    sys.modules[name] = module\n"
    "from pyamplicol.reporting.progress import _renderer_main\n"
    "_renderer_main(sys.argv[3:])\n"
)


def _new_dashboard(
    *,
    stream: TextIO,
    color: bool,
    snapshot: Callable[[], tuple[_TaskState, ...]],
    render_lock: RLock,
) -> _Dashboard | _SidecarDashboard:
    """The renderer-process dashboard on a real terminal (unless
    PYAMPLICOL_PROGRESS_RENDERER=inline), else the in-process one."""

    if os.environ.get(_RENDERER_ENV, "process") != "inline":
        sidecar = _SidecarDashboard.spawn(
            stream=stream, color=color, snapshot=snapshot, render_lock=render_lock
        )
        if sidecar is not None:
            return sidecar
    return _Dashboard(
        stream=stream, color=color, snapshot=snapshot, render_lock=render_lock
    )


class _RendererLink:
    """The pipe to a renderer process (``_renderer_main``): JSON lines in;
    at the end an acknowledgement once its output is written, then the
    process is reaped and its usage handed to the resource measurements,
    which leave it out."""

    # Bytes the renderer may fall behind by before it counts as stuck (a
    # stopped or hung renderer never blocks the work: the pipe is written
    # without blocking).
    _BACKLOG = 4 * 1024 * 1024

    def __init__(self, process: Any, ack_fd: int) -> None:
        self.process = process
        self.ack_fd = ack_fd
        self._fd = process.stdin.fileno()
        os.set_blocking(self._fd, False)
        self._backlog = bytearray()

    @classmethod
    def spawn(
        cls, stream: TextIO, *, color: bool, mode: str, tty_only: bool
    ) -> _RendererLink | None:
        """A renderer writing to ``stream``'s file descriptor (a terminal
        only with ``tty_only``), or None where none can start."""

        import subprocess

        try:
            fd = stream.fileno()
            if (tty_only and not os.isatty(fd)) or not sys.executable:
                return None
            os.fstat(fd)
            here = os.path.dirname(os.path.abspath(__file__))
            ack_read, ack_write = os.pipe()
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        _RENDERER_BOOTSTRAP,
                        os.path.dirname(here),
                        here,
                        str(os.getpid()),
                        "1" if color else "0",
                        getattr(stream, "encoding", None) or "utf-8",
                        str(ack_write),
                        mode,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=fd,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(ack_write,),
                    close_fds=True,
                )
            finally:
                os.close(ack_write)
            from .resources import register_monitor_process

            register_monitor_process(process.pid)
            return cls(process, ack_read)
        except Exception:  # no renderer: the in-process display
            return None

    def send(self, message: Mapping[str, object], *, droppable: bool = False) -> bool:
        """Queue one JSON line and write what the pipe takes now; a
        ``droppable`` message (a state superseded by the next) is skipped
        while earlier ones wait. False: the renderer is gone or stuck."""

        import json

        if droppable and self._backlog:
            return True
        try:
            line = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        except (TypeError, ValueError):
            return False
        self._backlog += line
        return self._drain() and len(self._backlog) <= self._BACKLOG

    def _drain(self) -> bool:
        while self._backlog:
            try:
                written = os.write(self._fd, self._backlog)
            except BlockingIOError:
                return True
            except OSError:
                return False
            del self._backlog[:written]
        return True

    def _drain_within(self, seconds: float) -> None:
        import select

        deadline = time.monotonic() + seconds
        while self._backlog and time.monotonic() < deadline:
            try:
                select.select([], [self._fd], [], max(deadline - time.monotonic(), 0))
            except (OSError, ValueError):
                return
            if not self._drain():
                return

    def finish(self) -> None:
        """Wait (bounded) for the renderer to write its last lines and exit."""

        import select

        self._drain_within(2.0)
        with contextlib.suppress(OSError, ValueError):
            self.process.stdin.close()
        try:
            ready, _, _ = select.select([self.ack_fd], [], [], 2.0)
            if ready:
                os.read(self.ack_fd, 1)
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                os.close(self.ack_fd)
        self.reap(timeout=1.0)

    def abandon(self) -> None:
        with contextlib.suppress(OSError):
            self.process.kill()
        self.reap(timeout=1.0)
        with contextlib.suppress(OSError):
            os.close(self.ack_fd)

    def reap(self, *, timeout: float) -> None:
        """Wait for the renderer (killed after ``timeout``) and report its
        usage to the resource measurements."""

        from .resources import monitor_process_ended

        pid = self.process.pid
        deadline = time.monotonic() + timeout
        killed = False
        while True:
            try:
                done, status, usage = os.wait4(pid, os.WNOHANG)
            except ChildProcessError:
                return
            if done:
                self.process.returncode = os.waitstatus_to_exitcode(status)
                peak = int(usage.ru_maxrss)
                monitor_process_ended(
                    pid,
                    peak if sys.platform == "darwin" else peak * 1024,
                    usage.ru_utime + usage.ru_stime,
                )
                return
            if time.monotonic() >= deadline and not killed:
                try:
                    self.process.kill()
                except OSError:
                    return
                killed = True
                deadline = time.monotonic() + timeout
            elif time.monotonic() >= deadline:
                return
            time.sleep(0.01)


class _SidecarDashboard:
    """The ``_Dashboard`` interface, drawn by a renderer process
    (``_renderer_main``) that owns the dashboard lines of the terminal:
    task states and messages go to it as JSON lines; at close it clears its
    lines and acknowledges before the caller writes again. Any failure of
    the renderer falls back to the in-process dashboard."""

    _MIN_INTERVAL = 0.05

    def __init__(
        self,
        *,
        stream: TextIO,
        color: bool,
        snapshot: Callable[[], tuple[_TaskState, ...]],
        render_lock: RLock,
        link: _RendererLink,
    ) -> None:
        self._stream = stream
        self._color = color
        self._snapshot = snapshot
        self._render_lock = render_lock
        self._link = link
        self._process = link.process
        self._fallback: _Dashboard | None = None
        self._last_sent = float("-inf")
        self._pending = False
        self._stop = Event()
        self._thread: Thread | None = None
        self._closed = False

    @classmethod
    def spawn(
        cls,
        *,
        stream: TextIO,
        color: bool,
        snapshot: Callable[[], tuple[_TaskState, ...]],
        render_lock: RLock,
    ) -> _SidecarDashboard | None:
        link = _RendererLink.spawn(stream, color=color, mode="dashboard", tty_only=True)
        if link is None:
            return None
        return cls(
            stream=stream,
            color=color,
            snapshot=snapshot,
            render_lock=render_lock,
            link=link,
        )

    # --- the _Dashboard interface -------------------------------------------
    def start(self) -> None:
        self._send_state()
        self._thread = Thread(
            target=self._run, name="pyamplicol-progress-renderer-feed", daemon=True
        )
        self._thread.start()

    def render_now(self, *, force: bool = False) -> None:
        if self._fallback is not None:
            self._fallback.render_now(force=force)
            return
        if force or time.monotonic() - self._last_sent >= self._MIN_INTERVAL:
            self._send_state()
        else:
            self._pending = True

    def write_message(self, text: str) -> None:
        if self._fallback is not None:
            self._fallback.write_message(text)
            return
        with self._render_lock:
            if self._closed:
                self._stream.write(text)
                return
            self._send({"op": "state", "tasks": self._states()})
            self._send({"op": "msg", "text": text})

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fallback is not None:
            self._fallback.close()
            return
        with self._render_lock:
            if self._closed:
                return
            self._send({"op": "state", "tasks": self._states()})
            self._send({"op": "close"})
            self._closed = True
            self._link.finish()

    # --- internals ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.wait(self._MIN_INTERVAL):
            if self._pending and self._fallback is None:
                self._send_state()

    def _states(self) -> list[dict[str, object]]:
        return [
            {name: getattr(task, name) for name in _TASK_FIELDS}
            for task in self._snapshot()
        ]

    def _send_state(self) -> None:
        with self._render_lock:
            if self._closed or self._fallback is not None:
                return
            self._pending = False
            self._last_sent = time.monotonic()
            self._send({"op": "state", "tasks": self._states()}, droppable=True)

    def _send(self, message: dict[str, object], *, droppable: bool = False) -> None:
        if self._fallback is None and not self._link.send(message, droppable=droppable):
            self._take_over()

    @property
    def broken(self) -> BaseException | None:
        """A failure of the in-process display this one fell back to."""

        return None if self._fallback is None else self._fallback.broken

    def _take_over(self) -> None:
        """The renderer is gone: draw in this process from now on."""

        if self._fallback is not None:
            return
        self._link.abandon()
        self._fallback = _Dashboard(
            stream=self._stream,
            color=self._color,
            snapshot=self._snapshot,
            render_lock=self._render_lock,
        )
        self._fallback.start()


def _event_record(event: ProgressEvent) -> dict[str, object]:
    """A progress event as a renderer process's JSON input."""

    if isinstance(event, ProgressStart):
        return {
            "op": "start",
            "task_id": event.task_id,
            "description": event.description,
            "total": event.total,
            "parent_task_id": event.parent_task_id,
            "unit": event.unit,
            "details": dict(event.details),
        }
    if isinstance(event, ProgressUpdate):
        return {
            "op": "update",
            "task_id": event.task_id,
            "completed": event.completed,
            "total": event.total,
            "message": event.message,
            "details": dict(event.details),
        }
    return {
        "op": "end",
        "task_id": event.task_id,
        "success": event.success,
        "message": event.message,
        "elapsed_seconds": event.elapsed_seconds,
        "details": dict(event.details),
    }


def _event_of(record: Mapping[str, Any]) -> ProgressEvent:
    fields = {key: value for key, value in record.items() if key != "op"}
    if record["op"] == "start":
        return ProgressStart(**fields)
    if record["op"] == "update":
        return ProgressUpdate(**fields)
    return ProgressEnd(**fields)


@dataclass(slots=True)
class _SidecarLineSink:
    """``--progress log`` written by a renderer process (``_renderer_main``
    in log mode, a ``_LineProgressSink`` with its heartbeat), so the log
    keeps its status lines while this process is inside native code that
    holds the GIL. The task states are kept here too (an interruption's
    whereabouts). Any failure falls back to an in-process line sink."""

    stream: TextIO
    color: bool = False
    minimum_interval: float = 2.0
    _link: _RendererLink | None = None
    _fallback: _LineProgressSink | None = None
    _states: dict[str, _TaskState] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    @classmethod
    def spawn(
        cls, stream: TextIO, *, color: bool, minimum_interval: float
    ) -> _SidecarLineSink | None:
        link = _RendererLink.spawn(stream, color=color, mode="log", tty_only=False)
        if link is None:
            return None
        return cls(stream, color=color, minimum_interval=minimum_interval, _link=link)

    def emit(self, event: ProgressEvent) -> None:
        try:
            with self._lock:
                _track_event(self._states, event, time.monotonic())
                if isinstance(event, ProgressEnd) and not any(
                    task.finished_at is None for task in self._states.values()
                ):
                    self._states.clear()
                if self._fallback is not None:
                    self._fallback.emit(event)
                    return
                if self._link is None or not self._link.send(_event_record(event)):
                    self._take_over(event)
        except Exception as error:  # the display never fails the work
            self._link = None
            self._fallback = _LineProgressSink(self.stream, _disabled=True)
            _display_failed(self.stream, error, "no progress display")

    def close(self) -> None:
        with self._lock:
            if self._fallback is not None:
                with contextlib.suppress(Exception):
                    self._fallback.close()
                return
            link, self._link = self._link, None
            if link is not None:
                try:
                    link.send({"op": "close"})
                    link.finish()
                except Exception as error:  # never fail the command
                    _display_failed(self.stream, error, "no progress display")

    def _take_over(self, event: ProgressEvent) -> None:
        if self._link is not None:
            self._link.abandon()
            self._link = None
        self._fallback = _LineProgressSink(
            self.stream,
            color=self.color,
            minimum_interval=self.minimum_interval,
            heartbeat=True,
        )
        self._fallback.emit(event)


# The kernel's "exiting" flag of a task (PF_EXITING in /proc/<pid>/stat's
# flags word): set the moment a process starts to exit (SIGKILL, SIGTERM,
# exit()), before its memory is torn down (up to hundreds of milliseconds)
# and before its pipes close or its children are re-parented.
_PF_EXITING = 0x4


def _exiting(pid: int) -> bool:
    """Whether process ``pid`` has begun to exit (Linux; elsewhere: not
    known, False)."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            text = handle.read()
        fields = text[text.rfind(b")") + 2 :].split()
        return bool(int(fields[6]) & _PF_EXITING) or fields[0] in (b"Z", b"X")
    except (OSError, ValueError, IndexError):
        return False


def _renderer_main(argv: list[str]) -> None:
    """The renderer process (``_SidecarDashboard``): argv = parent pid,
    colour flag, stream encoding, acknowledgement fd, mode. Ctrl-C is the
    parent's to handle. Only the parent's "close" (it waits for the
    acknowledgement before writing again) clears the dashboard; once the
    parent is gone (end of input without "close", or re-parented) the
    terminal may be the shell's again: nothing more is written, not even a
    clear, so the renderer only ever erases lines it drew while they were
    the last lines of the terminal."""

    import json
    import signal
    import sys

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent, color, encoding, ack_fd = (
        int(argv[0]),
        argv[1] == "1",
        argv[2],
        int(argv[3]),
    )
    stream = io.TextIOWrapper(
        io.FileIO(1, "w", closefd=False),
        encoding=encoding,
        errors="replace",
        write_through=True,
    )

    def alive() -> bool:
        return os.getppid() == parent and not _exiting(parent)

    if len(argv) > 4 and argv[4] == "log":
        _log_renderer(parent, color, stream, ack_fd, alive)
        return
    holder: list[tuple[_TaskState, ...]] = [()]
    dashboard = _Dashboard(
        stream=stream,
        color=color,
        snapshot=lambda: holder[0],
        render_lock=RLock(),
        resource_monitor=ResourceUsageMonitor(
            root_pid=parent, interval_seconds=1.0, exclude=frozenset({os.getpid()})
        ),
        alive=alive,
    )
    started = False
    closed = False
    try:
        for line in sys.stdin.buffer:
            message = json.loads(line)
            op = message.get("op")
            if op == "state":
                known = {task.task_id for task in holder[0]}
                holder[0] = tuple(_TaskState(**task) for task in message["tasks"])
                if not started:
                    dashboard.start()
                    started = True
                else:
                    dashboard.render_now(
                        force=any(task.task_id not in known for task in holder[0])
                    )
            elif op == "msg":
                dashboard.write_message(str(message["text"]))
            elif op == "close":
                closed = True
                break
    finally:
        if closed and alive():
            dashboard.close()
            with contextlib.suppress(Exception):
                stream.flush()
        else:
            dashboard.abandon()
        with contextlib.suppress(OSError):
            os.write(ack_fd, b"x")


def _log_renderer(
    parent: int,
    color: bool,
    stream: TextIO,
    ack_fd: int,
    alive: Callable[[], bool],
) -> None:
    """The renderer process in log mode: the events through a line sink
    with its heartbeat, measuring the parent's process tree; nothing is
    written once the parent is gone."""

    import json

    sink = _LineProgressSink(
        stream,
        color=color,
        minimum_interval=2.0,
        heartbeat=True,
        monitor_root=parent,
        alive=alive,
    )
    try:
        for line in sys.stdin.buffer:
            record = json.loads(line)
            if record.get("op") == "close":
                break
            sink.emit(_event_of(record))
    finally:
        sink.close()
        with contextlib.suppress(Exception):
            stream.flush()
        with contextlib.suppress(OSError):
            os.write(ack_fd, b"x")


# A step's lines are indented by its depth below its phase (a phase's own
# lines are not): the scroll-back reads as an outline.
_INDENT = "  "


def _depth(states: Mapping[str, _TaskState], task: _TaskState) -> int:
    """How many tasks above ``task`` are below its root (a phase: 0)."""

    depth = -1
    parent = task.parent_task_id
    while parent is not None and depth < 32:
        depth += 1
        state = states.get(parent)
        parent = None if state is None else state.parent_task_id
    return max(depth, 0)


def _kept_line(task: _TaskState, *, color: bool) -> str:
    """The scroll-back line of a finished phase that asks for one
    (``details["keep"]``): its description, its summary message and what
    it cost, "failed" in red."""

    elapsed = task.elapsed_seconds or max(
        (task.finished_at or task.updated_at) - task.started_at, 0.0
    )
    cost = f"{elapsed:.1f}s" if elapsed < 10.0 else _duration_text(elapsed)
    if task.success is False and task.message == "interrupted":
        # Ctrl-C: where the run stood, not a failure.
        return _paint_fragments(
            [(task.description, "CYAN"), (f"  interrupted after {cost}", "YELLOW")],
            width=10_000,
            enabled=color,
        )
    if task.success is False:
        # Where the run stopped; the error itself is reported once, below.
        return _paint_fragments(
            [(task.description, "CYAN"), (f"  failed after {cost}", "RED")],
            width=10_000,
            enabled=color,
        )
    fragments = [
        (task.description, "CYAN"),
        (f"  {task.message}" if task.message else "", None),
        (f"  done in {cost}", "GREEN"),
    ]
    return _paint_fragments(fragments, width=10_000, enabled=color)


def _terminal_width(stream: TextIO) -> int:
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        columns = 0
    if columns <= 0:
        # Unknown (a terminal reporting no size counts as unknown too).
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns
    return max(1, columns)


def _dashboard_lines(
    tasks: tuple[_TaskState, ...],
    *,
    now: float,
    usage: ResourceUsage,
    width: int,
    color: bool,
) -> tuple[str, str, str]:
    if not tasks:
        return ("", "", "")
    active = tuple(task for task in tasks if task.finished_at is None)
    roots = tuple(task for task in active if task.parent_task_id is None)
    if not roots:
        roots = tuple(task for task in tasks if task.parent_task_id is None)
    root = max(roots or tasks, key=lambda task: task.updated_at)
    children = tuple(task for task in active if task.parent_task_id == root.task_id)
    phase = max(children, key=lambda task: task.updated_at) if children else root
    parent_ids = {task.parent_task_id for task in active if task.parent_task_id}
    leaves = tuple(task for task in active if task.task_id not in parent_ids)
    if leaves:
        ordered = tuple(sorted(leaves, key=lambda task: task.task_id))
        rotation_anchor = min(task.started_at for task in ordered)
        detail = ordered[int(max(now - rotation_anchor, 0.0) / 2.0) % len(ordered)]
    else:
        detail = max(tasks, key=lambda task: task.updated_at)
    root = _with_total_estimate(root, phase, now)
    phase = _breadcrumb(phase, active)
    last_event = max(task.updated_at for task in tasks)
    return (
        _overall_line(root, now=now, usage=usage, width=width, color=color),
        _phase_line(
            phase,
            now=now,
            active_children=sum(
                task.parent_task_id == phase.task_id for task in active
            ),
            width=width,
            color=color,
        ),
        _detail_line(
            detail,
            now=now,
            width=width,
            color=color,
            usage=usage,
            last_event=last_event,
        ),
    )


def _breadcrumb(phase: _TaskState, active: tuple[_TaskState, ...]) -> _TaskState:
    """The phase line's task: a phase of one item (or uncounted) that runs
    steps shows its deepest counted running step (or its deepest step),
    named "phase > step": its count, bar and ETA are the step's."""

    if phase.total is not None and phase.total > 1:
        return phase
    chain = [phase]
    while True:
        children = [task for task in active if task.parent_task_id == chain[-1].task_id]
        if not children:
            break
        chain.append(max(children, key=lambda task: task.started_at))
    if len(chain) == 1:
        return phase
    counted = [task for task in chain[1:] if task.total is not None and task.total > 1]
    shown = counted[-1] if counted else chain[-1]
    crumb = shown.copy()
    crumb.description = f"{phase.description} > {shown.description}"
    return crumb


def _with_total_estimate(root: _TaskState, phase: _TaskState, now: float) -> _TaskState:
    """The root with the whole command's estimate as its ``eta_seconds``
    when it reports none itself but declares the priors of its phases still
    to come (``later_prior_seconds``; ``later_unknown`` when a phase to
    come has no prior: a lower bound, shown as an estimate)."""

    if (
        root.finished_at is not None
        or "eta_seconds" in root.details
        or "later_prior_seconds" not in root.details
    ):
        return root
    later = _detail_float(root.details, "later_prior_seconds") or 0.0
    active = _task_estimate(phase, now) if phase is not root else None
    total = combined(
        active, later, later_known=not bool(root.details.get("later_unknown"))
    )
    if total is None or total.seconds is None:
        return root
    derived = root.copy()
    derived.details["eta_seconds"] = total.seconds
    derived.details["eta_estimated"] = total.estimated
    derived.updated_at = now
    return derived


def _overall_line(
    root: _TaskState,
    *,
    now: float,
    usage: ResourceUsage,
    width: int,
    color: bool,
) -> str:
    elapsed = root.elapsed_seconds or max(
        (root.finished_at or now) - root.started_at,
        0.0,
    )
    execution_mode = root.details.get("execution_mode")
    backend = root.details.get("backend")
    optimization = root.details.get("optimization")
    identity = " | ".join(
        str(value) for value in (execution_mode, backend, optimization) if value
    )
    title = root.description + (f" [{identity}]" if identity else "")
    if width < 80:
        title = root.description
        rss = _rss_text(usage, compact=True)
        fragments = [
            (title, "CYAN"),
            (f"  {_duration_text(elapsed)}", None),
            (f"  RSS {rss}", "YELLOW" if usage.peak_rss_bytes is not None else None),
        ]
        return _paint_fragments(fragments, width=width, enabled=color)
    rss = _rss_text(usage, compact=width < 120)
    child_count = (
        "N/A" if usage.process_count is None else str(max(usage.process_count - 1, 0))
    )
    fragments = [
        (title, "CYAN"),
        (f"  elapsed {_duration_text(elapsed)}", None),
        *_total_eta_fragments(root, now=now),
        (f"  RSS {rss}", "YELLOW" if usage.peak_rss_bytes is not None else None),
        (
            "" if usage.busy_cores is None else f"  cores {usage.busy_cores:.1f}",
            None,
        ),
        (f"  children {child_count}", None),
    ]
    return _paint_fragments(fragments, width=width, enabled=color)


def _counted_down(task: _TaskState, key: str, now: float) -> float | None:
    """A task's ``key`` duration detail counted down since its last update
    (a value reported at an update keeps running between updates)."""

    value = _detail_float(task.details, key)
    if value is None:
        return None
    if task.finished_at is None:
        value -= max(now - task.updated_at, 0.0)
    return value


def _eta_text(seconds: float, *, estimated: bool) -> str:
    """An ETA for work that remains: never 0 ("<1s" at the end), "~" when
    it is a prior rather than a measured rate."""

    text = "<1s" if seconds < 1.0 else _duration_text(seconds)
    return ("~" if estimated else "") + text


def _total_eta_fragments(
    task: _TaskState, *, now: float
) -> list[tuple[str, str | None]]:
    """The whole command's ETA and, when it has one, its budget's remainder
    (a root task's ``eta_seconds``, ``eta_estimated``, ``budget_seconds``
    details), in the warning colour when the ETA passes the budget."""

    if task.finished_at is not None:
        return []
    eta = _counted_down(task, "eta_seconds", now)
    budget = _counted_down(task, "budget_seconds", now)
    fragments: list[tuple[str, str | None]] = []
    if eta is not None:
        over = budget is not None and eta > budget + max(0.25 * budget, 5.0)
        fragments.append(
            (
                "  ETA "
                + _eta_text(
                    max(eta, 0.0), estimated=bool(task.details.get("eta_estimated"))
                ),
                "YELLOW" if over else "GREEN",
            )
        )
        if eta >= 60.0:
            # When it should be done, on the wall clock (none past what the
            # platform's clock can represent).
            try:
                done_at = time.strftime("%H:%M", time.localtime(time.time() + eta))
            except (OverflowError, OSError, ValueError):
                done_at = None
            if done_at is not None:
                fragments.append((f" (done ~{done_at})", "LIGHTBLACK_EX"))
    if budget is not None:
        fragments.append(
            (
                f"  budget {_duration_text(max(budget, 0.0))} left"
                if budget > 0.0
                else "  budget spent",
                None,
            )
        )
    return fragments


def _task_estimate(task: _TaskState, now: float) -> Estimate | None:
    """A task's remaining time from its completion marks and its
    ``prior_seconds`` detail (its expected duration), or None."""

    prior = _detail_float(task.details, "prior_seconds")
    if task.unit == "phases":
        # Phases are not alike: their count says nothing of the time left
        # (a whole command's estimate comes from its phases' priors).
        return remaining_estimate(
            prior_seconds=prior, started_at=task.started_at, now=now
        )
    if task.total is not None and task.total > 0:
        marks = [(float(t), int(c)) for t, c in task.marks]
        if not marks and task.completed > 0:
            marks = [(task.updated_at, task.completed)]
        return count_estimate(
            total=task.total,
            completed=task.completed,
            started_at=task.started_at,
            marks=marks,
            now=now,
            prior_seconds=prior,
        )
    return remaining_estimate(prior_seconds=prior, started_at=task.started_at, now=now)


def _estimate_fragments(estimate: Estimate | None) -> list[tuple[str, str | None]]:
    """ "ETA 1m20s", "ETA ~1m20s" (a prior), or "overdue 12s" in the warning
    colour once the expected end has passed with work left."""

    if estimate is None:
        return []
    if estimate.seconds is None:
        return [
            (
                ("  ~" if estimate.estimated else "  ")
                + f"overdue {_duration_text(estimate.overdue)}",
                "YELLOW",
            )
        ]
    if estimate.at_least:
        # A lower bound: never a point estimate, no "~".
        return [(f"  ETA >{_eta_text(estimate.seconds, estimated=False)}", None)]
    return [
        ("  ETA " + _eta_text(estimate.seconds, estimated=estimate.estimated), None)
    ]


def _phase_line(
    phase: _TaskState,
    *,
    now: float,
    active_children: int,
    width: int,
    color: bool,
) -> str:
    elapsed = phase.elapsed_seconds or max(
        (phase.finished_at or now) - phase.started_at,
        0.0,
    )
    total = phase.total
    completed = min(phase.completed, total) if total is not None else phase.completed
    # An uncounted task that never counted shows no count (its clock moves).
    count = (
        ("" if completed == 0 else str(completed))
        if total is None
        else f"{completed}/{total}"
    )
    if total and total > 0:
        fraction = completed / total
        bar_width = 20 if width >= 110 else 10
        filled = min(bar_width, max(0, round(fraction * bar_width)))
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        percent = f" {fraction:>5.1%}"
        # A task that knows its costs reports its own estimate (counted down
        # between its updates); a task reporting none (the key present) has
        # none; otherwise the estimator (``eta.count_estimate``) on its
        # completion marks and its prior.
        eta = _counted_down(phase, "eta_seconds", now)
        if eta is not None and completed < total and phase.finished_at is None:
            eta = max(eta, 0.0)
    else:
        bar = "" if not count else "[...]"
        percent = ""
        eta = None
    estimate = (
        _task_estimate(phase, now)
        if "eta_seconds" not in phase.details and phase.finished_at is None
        else None
    )
    active_text = f"  active {active_children}" if active_children else ""
    fragments = [
        (phase.description, "CYAN"),
        (f"  {count} " if count else "", None),
        (bar, "GREEN"),
        (percent, None),
        (
            ""
            if eta is None or completed >= (total or 0)
            else "  ETA "
            + (
                _eta_text(eta, estimated=bool(phase.details.get("eta_estimated")))
                if phase.finished_at is None
                else _duration_text(eta)
            ),
            None,
        ),
        *_estimate_fragments(estimate),
        (f"  phase {_duration_text(elapsed)}", None),
        (active_text, None),
    ]
    return _paint_fragments(fragments, width=width, enabled=color)


def _detail_line(
    task: _TaskState,
    *,
    now: float,
    width: int,
    color: bool,
    usage: ResourceUsage | None = None,
    last_event: float | None = None,
) -> str:
    details = _display_details(task)
    process = details.get("process") or details.get("process_id")
    step = details.get("step") or task.message or task.description
    fragments: list[tuple[str, str | None]] = []
    if process:
        fragments.append((str(process), "CYAN"))
        fragments.append(("  ", None))
    step_color = (
        "RED"
        if task.success is False
        else "GREEN"
        if task.success is True
        else "MAGENTA"
    )
    fragments.append((str(step), step_color))
    fragments += _reading_fragments(details.get("readings"))
    backend = details.get("backend")
    if backend:
        fragments.append((f"  {backend}", "MAGENTA"))
    stage_index = _detail_int(details, "stage_index")
    stage_total = _detail_int(details, "stage_total")
    subset_size = _detail_int(details, "subset_size")
    if stage_index is not None:
        stage = str(stage_index) + (f"/{stage_total}" if stage_total else "")
        fragments.append(("  stage ", "MAGENTA"))
        fragments.append((stage, "GREEN"))
        if subset_size is not None:
            fragments.append(("  subset ", "MAGENTA"))
            fragments.append((str(subset_size), "GREEN"))
    for label, current_key, total_key in (
        ("masks", "mask_index", "mask_total"),
        ("chunks", "chunk_index", "chunk_total"),
        ("kernels", "kernel_index", "kernel_total"),
        (
            "parent pairs",
            "candidate_parent_pair_count",
            "candidate_parent_pair_total",
        ),
    ):
        current = _detail_int(details, current_key)
        total = _detail_int(details, total_key)
        if current is not None:
            fragments.append(
                (f"  {label} {current}" + (f"/{total}" if total else ""), None)
            )
    for label, key in (
        ("currents", "current_count"),
        ("colour states", "dynamic_color_state_count"),
        ("colour-pruned", "color_target_prune_count"),
        ("sources", "source_count"),
        ("colour sectors", "color_sector_count"),
        ("interactions", "interaction_count"),
        ("contributions", "contribution_count"),
        ("amplitudes", "amplitude_count"),
        ("invocations", "invocation_count"),
        ("attachments", "attachment_count"),
        ("finalizations", "finalization_count"),
        ("closures", "closure_count"),
        ("kernels", "kernel_count"),
        ("files", "file_count"),
        ("inputs", "input_count"),
        ("outputs", "output_count"),
    ):
        value = _detail_int(details, key)
        if value is not None:
            fragments.append((f"  {label} {value:,}", None))
    sample_index = _detail_int(details, "sample_index")
    sample_total = _detail_int(details, "sample_total")
    if sample_index is not None:
        sample = str(sample_index) + (f"/{sample_total}" if sample_total else "")
        fragments.append((f"  samples {sample}", None))
    waiting = details.get("waiting_seconds")
    stalled = _counted_up(task, "stalled_seconds", now)
    if isinstance(waiting, (float, int)) and not isinstance(waiting, bool):
        fragments.append((f"  waiting {_duration_text(float(waiting))}", "YELLOW"))
    elif stalled is not None:
        fragments.append((f"  no progress for {_duration_text(stalled)}", "YELLOW"))
    elif (
        task.finished_at is None
        and "running_seconds" not in details
        and now - (last_event or task.updated_at) >= _QUIET_SECONDS
    ):
        # The work's liveness: the time since any task last reported, and
        # whether the process tree is computing (busy cores) or idle.
        quiet = now - (last_event or task.updated_at)
        busy = None if usage is None else usage.busy_cores
        idle = busy is not None and busy < _IDLE_CORES
        fragments.append(
            (f"  no progress for {_duration_text(quiet)}", "RED" if idle else "YELLOW")
        )
        if busy is not None:
            fragments.append(
                ("  idle", "RED") if idle else (f"  busy {busy:.1f} cores", None)
            )
    native_current_rss = _detail_int(details, "native_current_rss_bytes")
    native_peak_rss = _detail_int(details, "native_peak_rss_bytes")
    if native_current_rss is not None or native_peak_rss is not None:
        fragments.append(
            (
                "  self RSS " + _native_rss_text(native_current_rss, native_peak_rss),
                "YELLOW",
            )
        )
    return _paint_fragments(fragments, width=width, enabled=color)


def _counted_up(task: _TaskState, key: str, now: float) -> float | None:
    """A task's ``key`` duration detail kept running since its last update."""

    value = _detail_float(task.details, key)
    if value is None:
        return None
    if task.finished_at is None:
        value += max(now - task.updated_at, 0.0)
    return value


# A reading of a ``readings`` detail: label, value and note fields, readings
# separated by RECORD; an empty value shows "N/A" (muted), the note muted.
READING_FIELD = "\x1f"
READING_RECORD = "\x1e"


def readings_detail(readings: Sequence[tuple[str, str | None, str | None]]) -> str:
    """A ``readings`` detail from (label, value or None, note or None)
    triples: the current best estimates a task shows on its detail line
    (``_reading_fragments``), e.g. each backend's time per point."""

    return READING_RECORD.join(
        READING_FIELD.join((label, value or "", note or ""))
        for label, value, note in readings
    )


def _readings(text: object) -> list[tuple[str, str, str]]:
    if not isinstance(text, str) or not text:
        return []
    out = []
    for record in text.split(READING_RECORD):
        label, value, note = [*record.split(READING_FIELD), "", "", ""][:3]
        out.append((label, value, note))
    return out


def _reading_fragments(text: object) -> list[tuple[str, str | None]]:
    """The detail line's readings: "  TIRust 2.31 ms (dd x37)  NinjaRS N/A",
    the label in the step's colour, a missing value "N/A" and the note
    muted."""

    fragments: list[tuple[str, str | None]] = []
    for label, value, note in _readings(text):
        fragments.append((f"  {label} ", "CYAN"))
        fragments.append((value or "N/A", None if value else "LIGHTBLACK_EX"))
        if note:
            fragments.append((f" {note}", "LIGHTBLACK_EX"))
    return fragments


def _readings_plain(text: object) -> str:
    """The readings in a log line: "TIRust 2.31 ms (dd x37), NinjaRS N/A"."""

    return ", ".join(
        f"{label} {value or 'N/A'}" + (f" {note}" if note else "")
        for label, value, note in _readings(text)
    )


_CHUNK_DETAIL = re.compile(
    r"\bchunk\s+(?P<index>\d+)/(?P<total>\d+)"
    r"(?:\s+p=(?P<inputs>\d+)/(?:\d+))?\b",
    re.IGNORECASE,
)
_EVALUATOR_DETAIL = re.compile(r"\bevaluator\s+\d+/(?P<outputs>\d+)\b", re.IGNORECASE)
_WAITING_DETAIL = re.compile(
    r"\bwaiting\s+(?P<seconds>\d+(?:\.\d+)?)s\b",
    re.IGNORECASE,
)


def _display_details(task: _TaskState) -> dict[str, ProgressDetailValue]:
    """Enrich legacy evaluator callbacks without changing their stable ABI."""

    details = dict(task.details)
    message = task.message or ""
    chunk = _CHUNK_DETAIL.search(message)
    if chunk is not None:
        details.setdefault("chunk_index", int(chunk.group("index")))
        details.setdefault("chunk_total", int(chunk.group("total")))
        if chunk.group("inputs") is not None:
            details.setdefault("input_count", int(chunk.group("inputs")))
        details.setdefault("step", "chunk compilation")
    evaluator = _EVALUATOR_DETAIL.search(message)
    if evaluator is not None:
        details.setdefault("output_count", int(evaluator.group("outputs")))
        details.setdefault("step", "evaluator construction")
    waiting = _WAITING_DETAIL.search(message)
    if waiting is not None:
        details.setdefault("waiting_seconds", float(waiting.group("seconds")))

    stage = str(details.get("stage", "")).strip().lower()
    backend = str(details.get("backend", "")).strip().lower()
    if "step" not in details:
        if stage.startswith("jit "):
            details["step"] = f"JIT {stage.removeprefix('jit ')}"
        elif stage.startswith("c++"):
            details["step"] = (
                "ASM compilation" if backend == "asm" else "C++ compilation"
            )
    return details


def _detail_int(details: ProgressDetails, key: str) -> int | None:
    value = details.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _detail_float(details: ProgressDetails, key: str) -> float | None:
    value = details.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _paint_fragments(
    fragments: list[tuple[str, str | None]],
    *,
    width: int,
    enabled: bool,
) -> str:
    remaining = max(width, 1)
    fragments = _shorten_lead(fragments, remaining)
    rendered: list[str] = []
    for text, color_name in fragments:
        if not text or remaining <= 0:
            continue
        clipped = text
        if len(clipped) > remaining:
            suffix = "..." if remaining >= 3 else ""
            clipped = clipped[: max(remaining - 3, 0)] + suffix
        remaining -= len(clipped)
        rendered.append(
            _terminal_paint(clipped, color_name, enabled=enabled)
            if color_name is not None
            else clipped
        )
    return "".join(rendered)


# A line's leading description keeps at least this many characters when it
# gives way to the clocks and counts after it.
_LEAD_MINIMUM = 24


def _shorten_lead(
    fragments: list[tuple[str, str | None]], width: int
) -> list[tuple[str, str | None]]:
    """A line too wide for ``width``: its first fragment (a description, a
    path) shortened in the middle first, so the clocks, counts and
    resources after it stay visible."""

    if len(fragments) < 2 or not fragments[0][0]:
        return fragments
    excess = sum(len(text) for text, _ in fragments) - width
    lead, color_name = fragments[0]
    if excess <= 0 or len(lead) <= _LEAD_MINIMUM:
        return fragments
    keep = max(len(lead) - excess, _LEAD_MINIMUM) - 3
    head = (keep + 1) // 2
    tail = keep - head
    shortened = lead[:head] + "..." + (lead[-tail:] if tail > 0 else "")
    return [(shortened, color_name), *fragments[1:]]


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    rounded = max(int(seconds), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _rss_text(usage: ResourceUsage, *, compact: bool = False) -> str:
    if usage.current_rss_bytes is None or usage.peak_rss_bytes is None:
        return "N/A"
    gib = 1024**3
    if compact:
        return f"{usage.current_rss_bytes / gib:.2f}/{usage.peak_rss_bytes / gib:.2f}G"
    return (
        f"{usage.current_rss_bytes / gib:.2f}/"
        f"{usage.peak_rss_bytes / gib:.2f} GiB current/peak"
    )


def _native_rss_text(current: int | None, peak: int | None) -> str:
    gib = 1024**3
    current_text = "N/A" if current is None else f"{current / gib:.2f}"
    peak_text = "N/A" if peak is None else f"{peak / gib:.2f}"
    return f"{current_text}/{peak_text} GiB current/peak"


def _format_event(
    event: ProgressEvent,
    *,
    description: str | None = None,
    color: bool = False,
) -> str:
    if isinstance(event, ProgressStart):
        result = _terminal_paint(event.description, "CYAN", enabled=color)
        if event.total is not None:
            result += (
                " ("
                + _terminal_paint(f"0/{event.total}", "MAGENTA", enabled=color)
                + ")"
            )
        return result
    label = description or event.task_id
    if isinstance(event, ProgressUpdate):
        count = (
            f"{event.completed}/{event.total}"
            if event.total is not None
            else str(event.completed)
        )
        message = _benchmark_statistics_message(event.details, color=color)
        if message is None:
            message = event.message
        if message is None and isinstance(event.details.get("step"), str):
            message = str(event.details["step"])
        readings = _readings_plain(event.details.get("readings"))
        if readings:
            message = (message + "; " if message else "") + readings
        eta = _detail_float(event.details, "eta_seconds")
        if eta is not None:
            message = (
                (message + "; " if message else "")
                + "ETA "
                + _eta_text(
                    max(eta, 0.0), estimated=bool(event.details.get("eta_estimated"))
                )
            )
        budget = _detail_float(event.details, "budget_seconds")
        if budget is not None:
            message = (message + "; " if message else "") + (
                f"budget {_duration_text(max(budget, 0.0))} left"
                if budget > 0.0
                else "budget spent"
            )
        stalled = _detail_float(event.details, "stalled_seconds")
        if stalled is not None:
            message = (message + "; " if message else "") + (
                f"no progress for {_duration_text(stalled)}"
            )
        suffix = f": {message}" if message else ""
        context = _log_detail_text(event.details)
        if event.total is None and event.completed == 0:
            # A task without counts (a whole command's ETA line).
            return _terminal_paint(label, "CYAN", enabled=color) + suffix + context
        return (
            _terminal_paint(label, "CYAN", enabled=color)
            + " "
            + _terminal_paint(count, "MAGENTA", enabled=color)
            + suffix
            + context
        )
    state = "done" if event.success else "failed"
    state_color = "GREEN" if event.success else "RED"
    suffix = (
        ": " + _terminal_paint(event.message, state_color, enabled=color)
        if event.message
        else ""
    )
    elapsed = (
        " in "
        + _terminal_paint(
            _duration_text(event.elapsed_seconds), "MAGENTA", enabled=color
        )
        if event.elapsed_seconds is not None
        else ""
    )
    return (
        _terminal_paint(label, "CYAN", enabled=color)
        + " "
        + _terminal_paint(state, state_color, enabled=color)
        + elapsed
        + suffix
    )


def _benchmark_statistics_message(
    details: ProgressDetails,
    *,
    color: bool,
) -> str | None:
    """Render typed benchmark statistics without putting ANSI in backend events."""

    if details.get("progress_kind") != "benchmark-statistics":
        return None
    elapsed = _detail_float(details, "elapsed_seconds")
    target = _detail_float(details, "target_seconds")
    mean = _detail_float(details, "wall_seconds_per_point")
    repetitions = _detail_int(details, "repetitions")
    batch_size = _detail_int(details, "batch_size")
    sample_count = _detail_int(details, "sample_count")
    if (
        elapsed is None
        or target is None
        or mean is None
        or repetitions is None
        or batch_size is None
        or sample_count is None
        or min(elapsed, target, mean, repetitions, batch_size, sample_count) < 0
    ):
        return None
    if mean < 1.0e-6:
        scale, unit = 1.0e9, "ns"
    elif mean < 1.0e-3:
        scale, unit = 1.0e6, "us"
    elif mean < 1.0:
        scale, unit = 1.0e3, "ms"
    else:
        scale, unit = 1.0, "s"

    result = _terminal_paint(f"{elapsed:.3g}/{target:.3g}s", "MAGENTA", enabled=color)
    result += "; " + _terminal_paint("wall", "CYAN", enabled=color) + " "
    result += _terminal_paint(
        f"{mean * scale:.5g} {unit}/point", "GREEN", enabled=color
    )
    if sample_count < 2:
        result += " " + _terminal_paint("SE pending", "YELLOW", enabled=color)
    else:
        error = _detail_float(details, "wall_standard_error_seconds_per_point")
        relative_error = _detail_float(details, "wall_relative_standard_error")
        if error is None or relative_error is None or min(error, relative_error) < 0:
            return None
        relative_color = (
            "GREEN"
            if relative_error <= 0.01
            else "YELLOW"
            if relative_error <= 0.05
            else "RED"
        )
        result += " +/- " + _terminal_paint(
            f"{error * scale:.2g} {unit}", "YELLOW", enabled=color
        )
        result += " (" + _terminal_paint(
            "relative standard error", "CYAN", enabled=color
        )
        result += (
            " "
            + _terminal_paint(f"{relative_error:.2%}", relative_color, enabled=color)
            + ")"
        )
    result += "; " + _terminal_paint(f"{repetitions} calls", "CYAN", enabled=color)
    result += " x " + _terminal_paint(f"{batch_size} points", "CYAN", enabled=color)
    return result


def _log_detail_text(details: ProgressDetails) -> str:
    fragments: list[str] = []
    process = details.get("process") or details.get("process_id")
    if process:
        fragments.append(f"process={process}")
    stage_index = _detail_int(details, "stage_index")
    stage_total = _detail_int(details, "stage_total")
    if stage_index is not None:
        stage = str(stage_index) + (f"/{stage_total}" if stage_total else "")
        fragments.append(f"stage={stage}")
    for name in (
        "subset_size",
        "candidate_parent_pair_count",
        "candidate_parent_pair_total",
        "current_count",
        "dynamic_color_state_count",
        "color_target_prune_count",
        "contribution_count",
    ):
        value = _detail_int(details, name)
        if value is not None:
            fragments.append(f"{name}={value}")
    native_current_rss = _detail_int(details, "native_current_rss_bytes")
    native_peak_rss = _detail_int(details, "native_peak_rss_bytes")
    if native_current_rss is not None or native_peak_rss is not None:
        fragments.append(
            "self_rss=" + _native_rss_text(native_current_rss, native_peak_rss)
        )
    return "" if not fragments else " [" + " ".join(fragments) + "]"


def progress_whereabouts(sink: ProgressSink | None) -> str | None:
    """Where a run stands, for an interruption message: "after 1m02s, in
    Generating the loop processes > Exporting the loop families (1 of 3
    phases done)"; None when the sink keeps no task state."""

    states: dict[str, _TaskState] | None = None
    if isinstance(sink, TtyProgressSink):
        states = sink._tasks if sink._line_sink is None else sink._line_sink._states
    elif isinstance(sink, (_LineProgressSink, _SidecarLineSink)):
        states = sink._states
    if not states:
        return None
    tasks = tuple(states.values())
    active = tuple(task for task in tasks if task.finished_at is None)
    roots = [task for task in active if task.parent_task_id is None]
    if not roots:
        return None
    root = max(roots, key=lambda task: task.started_at)
    chain = [root]
    while True:
        children = [task for task in active if task.parent_task_id == chain[-1].task_id]
        if not children:
            break
        chain.append(max(children, key=lambda task: task.started_at))
    text = f"after {_duration_text(time.monotonic() - root.started_at)}"
    if len(chain) > 1:
        text += ", in " + " > ".join(task.description for task in chain[1:])
    if root.unit == "phases" and root.total:
        done = sum(
            1
            for task in tasks
            if task.parent_task_id == root.task_id
            and task.finished_at is not None
            and task.success is not False
        )
        text += f" ({done} of {root.total} phases done)"
    return text


def close_progress_sink(sink: ProgressSink | None) -> None:
    """Close an interactive sink without requiring custom sinks to implement it."""

    close = getattr(sink, "close", None)
    if callable(close):
        close()


def progress_sink(
    mode: Literal["auto", "tty", "log", "off"],
    *,
    stream: TextIO,
    logger: logging.Logger | None = None,
    color: bool | None = None,
) -> ProgressSink:
    """Create a sink for the configured CLI progress mode."""

    selected = mode
    if selected == "auto":
        # A dumb terminal (TERM=dumb) gets the line display, as a file does.
        selected = "tty" if terminal_capable(stream) else "log"
    if selected == "off":
        return NullProgressSink()
    if selected == "log":
        # Plain lines on the stream, never in the logger format (the CLI
        # shows no logger records at its default level).
        del logger
        if os.environ.get(_RENDERER_ENV, "process") != "inline":
            sidecar = _SidecarLineSink.spawn(
                stream, color=bool(color), minimum_interval=2.0
            )
            if sidecar is not None:
                return sidecar
        return _LineProgressSink(
            stream, color=bool(color), minimum_interval=2.0, heartbeat=True
        )
    if selected == "tty":
        return TtyProgressSink(stream, color=color)
    raise ValueError(f"unknown progress mode {mode!r}")


__all__ = [
    "CallbackProgressSink",
    "LoggingProgressSink",
    "NullProgressSink",
    "ProgressEnd",
    "ProgressEvent",
    "ProgressSink",
    "ProgressStart",
    "ProgressUpdate",
    "StreamProgressSink",
    "TtyProgressSink",
    "close_progress_sink",
    "color_wanted",
    "progress_sink",
    "progress_whereabouts",
    "terminal_capable",
]
