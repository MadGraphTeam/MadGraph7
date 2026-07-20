# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from types import MappingProxyType
from typing import Any, Literal, Protocol, TextIO, TypeAlias, runtime_checkable

from .logging import get_logger
from .resources import ResourceUsage, ResourceUsageMonitor

ProgressDetailValue: TypeAlias = str | int | float | bool | None
ProgressDetails: TypeAlias = Mapping[str, ProgressDetailValue]


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
                f"progress detail {key!r} has unsupported type "
                f"{type(detail).__name__}"
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
    """Render a thread-safe adaptive three-line progress dashboard."""

    stream: TextIO
    color: bool | None = None
    _tasks: dict[str, _TaskState] = field(default_factory=dict, init=False, repr=False)
    _dashboard: _Dashboard | None = field(default=None, init=False, repr=False)
    _line_sink: _LineProgressSink | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def emit(self, event: ProgressEvent) -> None:
        dashboard: _Dashboard | None = None
        start_dashboard = False
        with self._lock:
            if not self._interactive:
                if self._line_sink is None:
                    self._line_sink = _LineProgressSink(
                        self.stream,
                        color=False,
                    )
                self._line_sink.emit(event)
                return
            now = time.monotonic()
            if isinstance(event, ProgressStart):
                existing = self._tasks.get(event.task_id)
                if existing is not None and existing.finished_at is None:
                    raise ValueError(f"progress task already started: {event.task_id}")
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
                    self._dashboard = _Dashboard(
                        stream=self.stream,
                        color=self._color_enabled,
                        snapshot=self._snapshot,
                    )
                    start_dashboard = True
            else:
                task = self._tasks.get(event.task_id)
                if task is None:
                    # Keep custom sinks forgiving when a backend emits a late pulse.
                    return
                if isinstance(event, ProgressUpdate):
                    task.completed = max(task.completed, event.completed)
                    task.total = event.total if event.total is not None else task.total
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
            dashboard = self._dashboard

        # Rendering snapshots task state and therefore acquires ``self._lock``.
        # Keep it outside that lock so the heartbeat's render lock and event
        # delivery can never form a lock-order cycle.
        if dashboard is not None:
            if start_dashboard:
                dashboard.start()
            else:
                dashboard.render_now()

    def close(self) -> None:
        with self._lock:
            dashboard = self._dashboard
            self._dashboard = None
        if dashboard is not None:
            dashboard.close()

    @property
    def _interactive(self) -> bool:
        if not bool(getattr(self.stream, "isatty", lambda: False)()):
            return False
        try:
            importlib.import_module("progressbar")
        except ImportError:
            return False
        return True

    @property
    def _color_enabled(self) -> bool:
        if self.color is not None:
            return self.color
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def _snapshot(self) -> tuple[_TaskState, ...]:
        with self._lock:
            return tuple(task.copy() for task in self._tasks.values())


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
        )


@dataclass(slots=True)
class _LineProgressSink:
    stream: TextIO
    color: bool = False
    descriptions: dict[str, str] = field(default_factory=dict)

    def emit(self, event: ProgressEvent) -> None:
        if isinstance(event, ProgressStart):
            self.descriptions[event.task_id] = event.description
        description = self.descriptions.get(event.task_id)
        rendered = _format_event(event, description=description)
        color_name = (
            "RED"
            if isinstance(event, ProgressEnd) and not event.success
            else "CYAN"
        )
        painted = _terminal_paint(rendered, color_name, enabled=self.color)
        self.stream.write(f"{painted}\n")
        self.stream.flush()
        if isinstance(event, ProgressEnd):
            self.descriptions.pop(event.task_id, None)


class _Dashboard:
    def __init__(
        self,
        *,
        stream: TextIO,
        color: bool,
        snapshot: Callable[[], tuple[_TaskState, ...]],
    ) -> None:
        progressbar: Any = importlib.import_module("progressbar")

        class _RefreshableMultiBar(progressbar.MultiBar):
            # progressbar2 4.5 omits ``yield from`` for already-started bars.
            def _render_bar(self, bar: Any, now: float, expired: float | None):
                def update(force: bool = True, write: bool = True):
                    self._label_bar(bar)
                    bar.update(force=force)
                    if write:
                        yield bar.fd.line

                if bar.finished():
                    yield from self._render_finished_bar(bar, now, expired, update)
                elif bar.started():
                    yield from update()
                elif self.initial_format is None:
                    bar.start()
                    yield from update()
                else:  # pragma: no cover - initial_format is deliberately None
                    yield self.initial_format.format(label=bar.label)

        self._stream = stream
        self._color = color
        self._snapshot = snapshot
        self._resource_monitor = ResourceUsageMonitor(interval_seconds=1.0)
        self._stop = Event()
        self._thread: Thread | None = None
        self._render_lock = RLock()
        self._widgets = [
            progressbar.FormatCustomText("%(line)s", {"line": ""}) for _ in range(3)
        ]
        self._multi = _RefreshableMultiBar(
            fd=stream,
            prepend_label=False,
            append_label=False,
            initial_format=None,
            show_initial=True,
            show_finished=False,
            remove_finished=False,
            update_interval=0.25,
        )
        for index, widget in enumerate(self._widgets):
            bar = progressbar.ProgressBar(
                max_value=progressbar.UnknownLength,
                widgets=[widget],
                fd=stream,
                enable_colors=color,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._multi[f"line-{index}"] = bar

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
        self.render_now()
        self._stream.write("\n")
        self._stream.flush()

    def render_now(self) -> None:
        with self._render_lock:
            lines = _dashboard_lines(
                self._snapshot(),
                now=time.monotonic(),
                usage=self._resource_monitor.usage,
                width=_terminal_width(self._stream),
                color=self._color,
            )
            for widget, line in zip(self._widgets, lines, strict=True):
                widget.update_mapping(line=line)
            self._multi.render(force=True)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self.render_now()


def _terminal_width(stream: TextIO) -> int:
    try:
        return max(48, os.get_terminal_size(stream.fileno()).columns)
    except (AttributeError, OSError, ValueError):
        return max(48, shutil.get_terminal_size(fallback=(120, 24)).columns)


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
    children = tuple(
        task for task in active if task.parent_task_id == root.task_id
    )
    phase = max(children, key=lambda task: task.updated_at) if children else root
    parent_ids = {task.parent_task_id for task in active if task.parent_task_id}
    leaves = tuple(task for task in active if task.task_id not in parent_ids)
    if leaves:
        ordered = tuple(sorted(leaves, key=lambda task: task.task_id))
        rotation_anchor = min(task.started_at for task in ordered)
        detail = ordered[int(max(now - rotation_anchor, 0.0) / 2.0) % len(ordered)]
    else:
        detail = max(tasks, key=lambda task: task.updated_at)
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
        _detail_line(detail, now=now, width=width, color=color),
    )


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
        "N/A"
        if usage.process_count is None
        else str(max(usage.process_count - 1, 0))
    )
    fragments = [
        (title, "CYAN"),
        (f"  elapsed {_duration_text(elapsed)}", None),
        (f"  RSS {rss}", "YELLOW" if usage.peak_rss_bytes is not None else None),
        (f"  children {child_count}", None),
    ]
    return _paint_fragments(fragments, width=width, enabled=color)


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
    count = str(completed) if total is None else f"{completed}/{total}"
    if total and total > 0:
        fraction = completed / total
        bar_width = 20 if width >= 110 else 10
        filled = min(bar_width, max(0, round(fraction * bar_width)))
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        percent = f" {fraction:>5.1%}"
        eta = (
            elapsed * (total - completed) / completed if completed > 0 else None
        )
    else:
        bar = "[...]"
        percent = ""
        eta = None
    active_text = f"  active {active_children}" if active_children else ""
    fragments = [
        (phase.description, "CYAN"),
        (f"  {count} ", None),
        (bar, "GREEN"),
        (percent, None),
        (f"  phase {_duration_text(elapsed)}", None),
        (f"  ETA {_duration_text(eta)}" if eta is not None else "", None),
        (active_text, None),
    ]
    return _paint_fragments(fragments, width=width, enabled=color)


def _detail_line(
    task: _TaskState,
    *,
    now: float,
    width: int,
    color: bool,
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
    backend = details.get("backend")
    if backend:
        fragments.append((f"  {backend}", "MAGENTA"))
    stage_index = _detail_int(details, "stage_index")
    stage_total = _detail_int(details, "stage_total")
    subset_size = _detail_int(details, "subset_size")
    if stage_index is not None:
        stage = str(stage_index) + (f"/{stage_total}" if stage_total else "")
        if subset_size is not None:
            stage += f" subset {subset_size}"
        fragments.append((f"  stage {stage}", "MAGENTA"))
    for label, current_key, total_key in (
        ("masks", "mask_index", "mask_total"),
        ("chunks", "chunk_index", "chunk_total"),
        ("kernels", "kernel_index", "kernel_total"),
    ):
        current = _detail_int(details, current_key)
        total = _detail_int(details, total_key)
        if current is not None:
            fragments.append(
                (f"  {label} {current}" + (f"/{total}" if total else ""), None)
            )
    for label, key in (
        ("currents", "current_count"),
        ("sources", "source_count"),
        ("colour sectors", "color_sector_count"),
        ("interactions", "interaction_count"),
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
    if isinstance(waiting, (float, int)) and not isinstance(waiting, bool):
        fragments.append((f"  waiting {_duration_text(float(waiting))}", "YELLOW"))
    elif task.finished_at is None and now - task.updated_at >= 5.0:
        fragments.append(
            (f"  active {_duration_text(now - task.updated_at)}", "YELLOW")
        )
    return _paint_fragments(fragments, width=width, enabled=color)


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


def _paint_fragments(
    fragments: list[tuple[str, str | None]],
    *,
    width: int,
    enabled: bool,
) -> str:
    remaining = max(width, 1)
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
        return (
            f"{usage.current_rss_bytes / gib:.2f}/"
            f"{usage.peak_rss_bytes / gib:.2f}G"
        )
    return (
        f"{usage.current_rss_bytes / gib:.2f}/"
        f"{usage.peak_rss_bytes / gib:.2f} GiB current/peak"
    )


def _format_event(
    event: ProgressEvent,
    *,
    description: str | None = None,
) -> str:
    if isinstance(event, ProgressStart):
        total = f" (0/{event.total})" if event.total is not None else ""
        return f"{event.description}{total}"
    label = description or event.task_id
    if isinstance(event, ProgressUpdate):
        count = (
            f"{event.completed}/{event.total}"
            if event.total is not None
            else str(event.completed)
        )
        suffix = f": {event.message}" if event.message else ""
        return f"{label} {count}{suffix}"
    state = "done" if event.success else "failed"
    suffix = f": {event.message}" if event.message else ""
    elapsed = (
        f" in {_duration_text(event.elapsed_seconds)}"
        if event.elapsed_seconds is not None
        else ""
    )
    return f"{label} {state}{elapsed}{suffix}"


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
    "close_progress_sink",
    "progress_sink",
]
