# SPDX-License-Identifier: 0BSD
"""Typed progress phases for the generation service."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, current_thread, main_thread

from pyamplicol.reporting import (
    NullProgressSink,
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)
from pyamplicol.reporting.progress import ProgressDetailValue, ProgressEvent


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


# The handle of the phase or step running in this context: ``step`` opens
# its sub-steps under it without the generation code passing it down.
_CURRENT: ContextVar[PhaseHandle | None] = ContextVar(
    "pyamplicol_generation_progress_handle", default=None
)

# A finished step shorter than this leaves no line in the scroll-back (its
# time still counts in its phase and in the history).
_KEEP_SECONDS = 1.0

# Set on Ctrl-C (``cancel_generation``): a worker thread of the generation
# stops at its next progress report instead of computing on, which would
# starve the main thread's cleanup of the GIL.
_CANCELLED = Event()


class GenerationCancelled(Exception):
    """Raised in a generation worker thread that reports progress after
    Ctrl-C (``cancel_generation``)."""


def cancel_generation() -> None:
    """Ctrl-C: stop the generation's worker threads at their next report."""

    _CANCELLED.set()


class _AbandonedPoolFilter:
    """Drop the "Exception in worker" records of pyAmpliCol's process-phase
    pools stopped after Ctrl-C (``interrupt_threads``)."""

    def filter(self, record: object) -> bool:
        thread = str(getattr(record, "threadName", ""))
        message = getattr(record, "msg", "")
        return not (
            thread.startswith("pyamplicol-") and message == "Exception in worker"
        )


def interrupt_threads(threads: tuple[object, ...]) -> None:
    """Ctrl-C: raise ``GenerationCancelled`` in the generation's worker
    ``threads`` at their next bytecode (CPython; a thread inside a native
    call stops when it returns). Their ``with`` and ``finally`` blocks run;
    their results are abandoned."""

    import ctypes
    import logging
    import sys

    if sys.implementation.name != "cpython":
        return
    # A worker stopped this way is reported by the pool as "Exception in
    # worker" with a traceback, past the display: those records of the
    # abandoned pyAmpliCol pools are dropped (nothing else of the logger).
    pool_logger = logging.getLogger("concurrent.futures")
    if not any(isinstance(item, _AbandonedPoolFilter) for item in pool_logger.filters):
        pool_logger.addFilter(_AbandonedPoolFilter())
    for thread in threads:
        ident = getattr(thread, "ident", None)
        if ident is None or thread is current_thread():
            continue
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(ident), ctypes.py_object(GenerationCancelled)
        )


def _stop_if_cancelled() -> None:
    if _CANCELLED.is_set() and current_thread() is not main_thread():
        raise GenerationCancelled("interrupted")


def _emit(sink: ProgressSink | None, event: Callable[[], ProgressEvent]) -> None:
    """Build and emit one progress event; a failure of either (a value the
    event refuses, a sink that raises) is the display's, never the
    generation's: it is reported once and the work goes on."""

    if sink is None:
        return
    try:
        sink.emit(event())
    except Exception as error:  # the display never fails the work
        from pyamplicol.reporting.progress import _display_failed

        _display_failed(None, error, "the other progress events")


@dataclass(slots=True)
class PhaseHandle:
    task_id: str
    sink: ProgressSink | None
    total: int | None
    completed: int = 0
    # What the phase or step produced ("152 evaluators, 32.6 MiB"), shown
    # on its scroll-back line when it ends.
    summary: str | None = None
    history: GenerationHistory | None = field(default=None, repr=False)
    path: str = ""
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update(
        self,
        completed: int,
        *,
        message: str | None = None,
        total: int | None = None,
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> None:
        _stop_if_cancelled()
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
        _stop_if_cancelled()
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
            _emit(
                self.sink,
                lambda: ProgressUpdate(
                    self.task_id,
                    completed=self.completed,
                    total=active_total,
                    message=message,
                    details={} if details is None else details,
                ),
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
        keep: bool = False,
    ) -> Iterator[PhaseHandle]:
        """A sub-task of this phase; ``keep``: a step whose end leaves a
        line in the scroll-back (when it lasted ``_KEEP_SECONDS``) and whose
        time is kept in the generation history."""

        _stop_if_cancelled()
        task_id = f"{self.task_id}:{key}"
        path = f"{self.path}/{key}" if self.path else key
        prior = self.history.prior(path) if keep and self.history else None
        if self.sink is not None:
            start_details = dict(details or {})
            if prior is not None:
                start_details["prior_seconds"] = prior
            _emit(
                self.sink,
                lambda: ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=self.task_id,
                    unit=unit,
                    details=start_details,
                ),
            )
        handle = PhaseHandle(task_id, self.sink, total, history=self.history, path=path)
        if keep and self.history is not None:
            self.history.opened(task_id, path, handle)
        token = _CURRENT.set(handle) if keep else None
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            if self.history is not None:
                self.history.closed(task_id)
            failure = _failure_message(exc)
            if self.sink is not None:
                _emit(
                    self.sink,
                    lambda: ProgressEnd(
                        task_id,
                        success=False,
                        message=failure,
                        elapsed_seconds=elapsed,
                        details={"keep": keep},
                    ),
                )
            raise
        finally:
            if token is not None:
                _CURRENT.reset(token)
        elapsed = time.perf_counter() - started
        if keep and self.history is not None:
            self.history.closed(task_id)
            self.history.observe(path, elapsed)
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            _emit(
                self.sink,
                lambda: ProgressEnd(
                    task_id,
                    message=handle.summary,
                    elapsed_seconds=elapsed,
                    details={"keep": keep and elapsed >= _KEEP_SECONDS},
                ),
            )


@contextmanager
def step(
    key: str,
    description: str,
    *,
    total: int | None = None,
    details: Mapping[str, ProgressDetailValue] | None = None,
) -> Iterator[PhaseHandle]:
    """A counted or single sub-step of the phase or step running in this
    context (``GenerationPhaseReporter.phase``): its own line on the
    display, its cost on a scroll-back line when it ends, its time in the
    generation history (the prior of the next generation of the same
    request). Without a running phase (a library call without progress)
    it costs nothing."""

    parent = _CURRENT.get()
    if parent is None or parent.sink is None:
        yield PhaseHandle(key, None, total)
        return
    with parent.child(
        key, description, total=total, details=details, keep=True
    ) as handle:
        yield handle


def adopt_task(task_id: str, sink: ProgressSink | None) -> object:
    """Make the running progress task ``task_id`` (started by the caller on
    ``sink``) the parent of the ``step``s run in this context, e.g. a check
    test that regenerates a process; returns the token for ``release``."""

    handle = None if sink is None else PhaseHandle(task_id, sink, None, path=task_id)
    return _CURRENT.set(handle)


def release_task(token: object) -> None:
    """Undo ``adopt_task`` (its token)."""

    _CURRENT.reset(token)  # type: ignore[arg-type]


def _failure_message(exc: BaseException) -> str | None:
    """A failed phase's or step's end message: "interrupted" for Ctrl-C;
    none for an error (the command reports the error once, its lines say
    only where it stopped)."""

    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    return None


def history_signature(kind: str, run_config: object | None) -> str | None:
    """The ``GenerationHistory`` signature of a generation request: its
    kind and every setting that changes what is generated (model, processes,
    colour, loop, evaluator, the generation settings but the output and
    its mode)."""

    if run_config is None:
        return None
    try:
        return _signature(kind, run_config)
    except Exception:  # no history rather than a failure
        return None


def _signature(kind: str, run_config: object) -> str:
    import dataclasses

    generation = getattr(run_config, "generation", None)
    generation_items = (
        tuple(
            (item.name, getattr(generation, item.name))
            for item in dataclasses.fields(generation)
            if item.name not in {"output", "mode"}
        )
        if dataclasses.is_dataclass(generation)
        else generation
    )
    return GenerationHistory.signature_of(
        kind,
        *(
            getattr(run_config, name, None)
            for name in ("model", "process", "color", "loop", "evaluator")
        ),
        generation_items,
    )


# Phases and steps shorter than this say little about a run's pace (their
# ratio is noise) and do not enter its factor; a change of the factor by
# more than this ratio is pushed to the open tasks' priors.
_RATIO_MINIMUM = 0.05
_RATIO_PUSH = 1.15


class GenerationHistory:
    """The phase and step times of the previous generation of the same
    request (same model, processes, colour, loop, evaluator and generation
    settings but the output; ``signature``), kept in the pyAmpliCol cache
    (``generation-history.json``), and the times of this one, stored when
    it completes. Missing, unreadable or unwritable: no priors."""

    _LIMIT = 200
    _FILE = "generation-history.json"

    def __init__(self, signature: str | None, *, root: Path | None = None) -> None:
        self.signature = signature
        self._root = root
        self.previous: dict[str, float] = {}
        self.order: list[str] = []
        self.current: dict[str, float] = {}
        # This run against the history: the summed actual and prior seconds
        # of the phases and steps finished so far (``factor``).
        self._seen = [0.0, 0.0]
        self._pushed = 1.0
        # The open phases and steps that carry a prior (task id -> path,
        # handle), whose priors follow the factor (``on_change``).
        self._open: dict[str, tuple[str, PhaseHandle]] = {}
        self.on_change: Callable[[], None] | None = None
        try:
            self._read(signature)
        except Exception:  # a history that cannot be read: no priors
            self.previous, self.order = {}, []

    def _read(self, signature: str | None) -> None:
        if signature is not None:
            record = self._load().get(signature)
            if isinstance(record, dict):
                steps = record.get("steps")
                if isinstance(steps, dict):
                    # A hand-edited or foreign file: only finite, non-negative
                    # seconds are priors (JSON admits NaN and Infinity).
                    self.previous = {
                        str(key): float(value)
                        for key, value in steps.items()
                        if isinstance(value, (int, float))
                        and math.isfinite(value)
                        and value >= 0
                    }
                order = record.get("phases")
                if isinstance(order, list):
                    self.order = [str(name) for name in order]

    @staticmethod
    def signature_of(*parts: object) -> str:
        return hashlib.sha256(
            "\x1f".join(repr(part) for part in parts).encode()
        ).hexdigest()

    def _raw(self, path: str) -> float | None:
        value = self.previous.get(path)
        return value if isinstance(value, float) and math.isfinite(value) else None

    def factor(self) -> float:
        """How this run compares with the history: the actual over the
        prior seconds of what it finished so far (1 before anything): the
        history is a prior, the measured run corrects it at once (a history
        100 times off is off by the run's own factor, not 100 times)."""

        prior, actual = self._seen
        if prior <= 0.0 or actual <= 0.0:
            return 1.0
        return actual / prior

    def prior(self, path: str) -> float | None:
        raw = self._raw(path)
        if raw is None:
            return None
        scaled = raw * self.factor()
        return scaled if math.isfinite(scaled) else None

    def later(self, started: set[str]) -> tuple[float, bool] | None:
        """The previous run's time of the phases not started yet (None
        without a previous run)."""

        if not self.order:
            return None
        try:
            seconds = self.factor() * sum(
                self.previous.get(name, 0.0)
                for name in self.order
                if name not in started
            )
        except Exception:
            return None
        return (seconds, True) if math.isfinite(seconds) else None

    def opened(self, task_id: str, path: str, handle: PhaseHandle) -> None:
        if self._raw(path) is not None:
            self._open[task_id] = (path, handle)

    def closed(self, task_id: str) -> None:
        self._open.pop(task_id, None)

    def observe(self, path: str, seconds: float) -> None:
        with contextlib.suppress(Exception):
            self.current[path] = self.current.get(path, 0.0) + seconds
            raw = self._raw(path)
            if raw is None or raw <= 0.0 or seconds < _RATIO_MINIMUM:
                return
            self._seen[0] += raw
            self._seen[1] += seconds
            factor = self.factor()
            if abs(math.log(factor / self._pushed)) > math.log(_RATIO_PUSH):
                self._pushed = factor
                if self.on_change is not None:
                    self.on_change()

    def store(self, phases: list[str]) -> None:
        if self.signature is None:
            return
        try:
            path = self._path()
            data = self._load()
            data.pop(self.signature, None)
            data[self.signature] = {
                "phases": phases,
                "steps": {key: round(value, 3) for key, value in self.current.items()},
                "stored": time.time(),
            }
            while len(data) > self._LIMIT:
                data.pop(next(iter(data)))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}")
            temporary.write_text(json.dumps(data), encoding="utf-8")
            os.replace(temporary, path)
        except Exception:  # an unwritable history: none stored
            return

    def _path(self) -> Path:
        if self._root is not None:
            return self._root / self._FILE
        configured = os.environ.get("PYAMPLICOL_CACHE_DIR")
        if configured:
            root = Path(configured).expanduser()
        else:
            from platformdirs import user_cache_path

            root = user_cache_path("pyamplicol")
        return root / self._FILE

    def _load(self) -> dict[str, object]:
        try:
            data = json.loads(self._path().read_text(encoding="utf-8"))
        except Exception:  # unreadable: no history
            return {}
        return data if isinstance(data, dict) else {}


@dataclass(slots=True)
class GenerationPhaseReporter:
    sink: ProgressSink | None
    timings: dict[str, float] = field(default_factory=dict)
    root: PhaseHandle | None = None
    _root_started_at: float | None = None
    history: GenerationHistory | None = None
    _phases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Progress-off must not install callbacks into generation machinery.
        if isinstance(self.sink, NullProgressSink):
            self.sink = None

    def start_generation(
        self,
        *,
        total: int,
        details: Mapping[str, ProgressDetailValue],
        signature: str | None = None,
        description: str = "Generating processes",
    ) -> None:
        """Open the generation; ``signature`` (``GenerationHistory.
        signature_of`` of the request's settings) names its history: the
        previous generation's phase and step times are this one's priors."""

        task_id = "generation"
        _CANCELLED.clear()
        self._root_started_at = time.perf_counter()
        if self.sink is not None and signature is not None:
            self.history = GenerationHistory(signature)
            self.history.on_change = self._refresh_priors
        root_details = dict(details)
        later = self.history.later(set()) if self.history else None
        if later is not None:
            root_details["later_prior_seconds"] = later[0]
        if self.sink is not None:
            _emit(
                self.sink,
                lambda: ProgressStart(
                    task_id,
                    description,
                    total=total,
                    unit="phases",
                    details=root_details,
                ),
            )
        self.root = PhaseHandle(task_id, self.sink, total, history=self.history)

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
        if success and self.history is not None:
            self.history.store(self._phases)
        if self.sink is not None:
            _emit(
                self.sink,
                lambda: ProgressEnd(
                    self.root.task_id,
                    success=success,
                    message=message,
                    elapsed_seconds=elapsed,
                ),
            )
        self._root_started_at = None

    def _refresh_priors(self) -> None:
        """The run's factor moved (``GenerationHistory.factor``): the open
        phases and steps get their rescaled priors, the root the rescaled
        time of the phases to come."""

        history = self.history
        if history is None or self.sink is None:
            return
        for task_id, (path, handle) in list(history._open.items()):
            prior = history.prior(path)
            if prior is None:
                continue
            _emit(
                self.sink,
                lambda task_id=task_id, handle=handle, prior=prior: ProgressUpdate(
                    task_id,
                    completed=handle.completed,
                    total=handle.total,
                    details={"prior_seconds": prior},
                ),
            )
        later = history.later(set(self._phases))
        if later is not None and self.root is not None:
            root = self.root
            _emit(
                self.sink,
                lambda: ProgressUpdate(
                    root.task_id,
                    completed=root.completed,
                    total=root.total,
                    details={"later_prior_seconds": later[0]},
                ),
            )

    @contextmanager
    def phase(
        self,
        name: str,
        description: str,
        *,
        total: int | None = None,
    ) -> Iterator[PhaseHandle]:
        task_id = f"generation:{name}"
        history = self.history
        if name not in self._phases:
            self._phases.append(name)
        prior = history.prior(name) if history is not None else None
        if self.sink is not None:
            if history is not None and self.root is not None:
                later = history.later(set(self._phases))
                if later is not None:
                    # The root's estimate: this phase's plus the phases to come.
                    _emit(
                        self.sink,
                        lambda: ProgressUpdate(
                            self.root.task_id,
                            completed=self.root.completed,
                            total=self.root.total,
                            details={"later_prior_seconds": later[0]},
                        ),
                    )
            _emit(
                self.sink,
                lambda: ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=None if self.root is None else self.root.task_id,
                    details={} if prior is None else {"prior_seconds": prior},
                ),
            )
        handle = PhaseHandle(task_id, self.sink, total, history=history, path=name)
        if history is not None:
            history.opened(task_id, name, handle)
        token = _CURRENT.set(handle)
        times = _PHASE_TIMES.get()
        top = times is not None and times.depth == 0
        if times is not None:
            times.depth += 1
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            _CURRENT.reset(token)
            if history is not None:
                history.closed(task_id)
            self._close_phase(times, top, name, elapsed)
            self.timings[name] = elapsed
            failure = _failure_message(exc)
            if self.sink is not None:
                _emit(
                    self.sink,
                    lambda: ProgressEnd(
                        task_id,
                        success=False,
                        message=failure,
                        elapsed_seconds=elapsed,
                        details={"keep": True},
                    ),
                )
            raise
        _CURRENT.reset(token)
        elapsed = time.perf_counter() - started
        self._close_phase(times, top, name, elapsed)
        self.timings[name] = elapsed
        if history is not None:
            history.closed(task_id)
            history.observe(name, elapsed)
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            _emit(
                self.sink,
                lambda: ProgressEnd(
                    task_id,
                    message=handle.summary,
                    elapsed_seconds=elapsed,
                    details={"keep": True},
                ),
            )
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
    "GenerationCancelled",
    "GenerationHistory",
    "GenerationPhaseReporter",
    "PhaseHandle",
    "PhaseTimes",
    "adopt_task",
    "cancel_generation",
    "collect_phase_times",
    "history_signature",
    "interrupt_threads",
    "record_phase",
    "release_task",
    "step",
]
