# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, Protocol, TextIO, TypeAlias, cast, runtime_checkable

from .logging import get_logger


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


@dataclass(frozen=True, slots=True)
class ProgressStart:
    task_id: str
    description: str
    total: int | None = None

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("progress description must be a non-empty string")
        _total(self.total)


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    task_id: str
    completed: int
    total: int | None = None
    message: str | None = None

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


@dataclass(frozen=True, slots=True)
class ProgressEnd:
    task_id: str
    success: bool = True
    message: str | None = None

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if not isinstance(self.success, bool):
            raise ValueError("progress success must be a boolean")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("progress message must be a string or null")


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
            self.logger.log(self.level, _format_event(event))
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
    """Render thread-safe progressbar2 bars for typed generation phases."""

    stream: TextIO
    color: bool | None = None
    _bars: dict[str, _ProgressBar] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            if isinstance(event, ProgressStart):
                if event.task_id in self._bars:
                    raise ValueError(f"progress task already started: {event.task_id}")
                self._bars[event.task_id] = self._start(event)
                return
            bar = self._bars.get(event.task_id)
            if bar is None:
                self.stream.write(f"{_format_event(event)}\n")
                self.stream.flush()
                return
            if isinstance(event, ProgressUpdate):
                bar.update(
                    event.completed,
                    force=True,
                    status=event.message or "",
                )
                return
            bar.finish(dirty=not event.success)
            if event.message:
                self.stream.write(f"{event.task_id}: {event.message}\n")
                self.stream.flush()
            self._bars.pop(event.task_id, None)

    def _start(self, event: ProgressStart) -> _ProgressBar:
        color = (
            bool(getattr(self.stream, "isatty", lambda: False)())
            if self.color is None
            else self.color
        )
        description = _terminal_paint(event.description, "CYAN", enabled=color)
        try:
            progressbar: Any = importlib.import_module("progressbar")
        except ImportError:
            self.stream.write(
                f"{_terminal_paint(_format_event(event), 'CYAN', enabled=color)}\n"
            )
            self.stream.flush()
            return _LineProgress(self.stream, event.task_id)
        if event.total is None:
            widgets: list[Any] = [
                f"{description}: ",
                progressbar.AnimatedMarker(),
                " ",
                progressbar.Counter(),
                " ",
                progressbar.Timer(),
                " ",
                progressbar.DynamicMessage("status"),
            ]
            maximum = progressbar.UnknownLength
        else:
            widgets = [
                f"{description}: ",
                progressbar.Percentage(),
                " ",
                progressbar.Bar(
                    marker=_terminal_paint("#", "GREEN", enabled=color),
                ),
                " ",
                progressbar.Counter(),
                f"/{event.total} ",
                progressbar.ETA(),
                " ",
                progressbar.DynamicMessage("status"),
            ]
            maximum = event.total
        bar = progressbar.ProgressBar(
            max_value=maximum,
            widgets=widgets,
            fd=self.stream,
            enable_colors=color,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        return cast(_ProgressBar, bar.start())


class _ProgressBar(Protocol):
    def update(
        self,
        completed: int,
        *,
        force: bool = False,
        status: str = "",
    ) -> None: ...

    def finish(self, *, dirty: bool = False) -> None: ...


@dataclass(slots=True)
class _LineProgress:
    stream: TextIO
    task_id: str

    def update(
        self,
        completed: int,
        *,
        force: bool = False,
        status: str = "",
    ) -> None:
        del force
        message = f": {status}" if status else ""
        self.stream.write(f"{self.task_id} {completed}{message}\n")
        self.stream.flush()

    def finish(self, *, dirty: bool = False) -> None:
        state = "failed" if dirty else "done"
        self.stream.write(f"{self.task_id} {state}\n")
        self.stream.flush()


def _format_event(event: ProgressEvent) -> str:
    if isinstance(event, ProgressStart):
        total = f" (0/{event.total})" if event.total is not None else ""
        return f"{event.description}{total}"
    if isinstance(event, ProgressUpdate):
        count = (
            f"{event.completed}/{event.total}"
            if event.total is not None
            else str(event.completed)
        )
        suffix = f": {event.message}" if event.message else ""
        return f"{event.task_id} {count}{suffix}"
    state = "done" if event.success else "failed"
    suffix = f": {event.message}" if event.message else ""
    return f"{event.task_id} {state}{suffix}"


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
        selected = "tty" if bool(getattr(stream, "isatty", lambda: False)()) else "log"
    if selected == "off":
        return NullProgressSink()
    if selected == "log":
        return LoggingProgressSink(logger or get_logger("progress"))
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
    "progress_sink",
]
