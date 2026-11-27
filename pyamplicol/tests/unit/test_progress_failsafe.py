# SPDX-License-Identifier: 0BSD
"""The progress display never fails the command it draws for: a failure in
rendering, estimating, the history, resource sampling or the renderer
process degrades the display (one warning line at most) and the work goes on.
Also: NO_COLOR / TERM=dumb, and the interrupt of a pool detected exactly."""

from __future__ import annotations

import io
import os
import signal
import sys
import time

import pytest

from pyamplicol.generation import progress as generation_progress
from pyamplicol.generation.progress import GenerationHistory, GenerationPhaseReporter
from pyamplicol.reporting import progress
from pyamplicol.reporting.progress import (
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
    TtyProgressSink,
)
from pyamplicol.reporting.resources import ResourceUsageMonitor

POSIX = pytest.mark.skipif(
    sys.platform == "win32", reason="the live display needs a POSIX terminal"
)


@pytest.fixture(autouse=True)
def _fresh_warning(monkeypatch):
    monkeypatch.setattr(progress, "_DISPLAY_WARNED", progress.Event())


@pytest.fixture
def terminal(monkeypatch):
    import fcntl
    import pty
    import struct
    import termios

    monkeypatch.setenv("TERM", "xterm-256color")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    stream = os.fdopen(slave, "w", buffering=1, encoding="utf-8")
    try:
        yield master, stream
    finally:
        stream.close()
        os.close(master)


def _read_all(fd: int) -> str:
    import select

    chunks = []
    while True:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            break
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks).decode(errors="replace")


def _read_until_quiet(fd: int, seconds: float) -> str:
    """Read ``fd`` until it has been quiet for 3 s (at most ``seconds``)."""

    import select

    chunks, deadline, quiet = [], time.monotonic() + seconds, time.monotonic()
    while time.monotonic() < deadline and time.monotonic() - quiet < 3.0:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            chunks.append(data)
            quiet = time.monotonic()
    return b"".join(chunks).decode(errors="replace")


def _generate(sink, *, signature: str | None = None) -> str:
    """A fake generation through the reporter: a counted phase and steps."""

    reporter = GenerationPhaseReporter(sink)
    reporter.start_generation(total=2, details={}, signature=signature)
    with reporter.phase("compile", "Compiling", total=4) as phase:
        for index in range(4):
            time.sleep(0.02)
            phase.update(index + 1, message=f"item {index + 1}")
    with (
        reporter.phase("export", "Exporting", total=1),
        generation_progress.step("families", "Families", total=3) as families,
    ):
        for index in range(3):
            families.update(index + 1)
    reporter.finish_generation()
    progress.close_progress_sink(sink)
    return "done"


def _warnings(text: str) -> int:
    return text.count("warning: the progress display failed")


def _boom(*_args, **_kwargs):
    raise RuntimeError("injected display failure")


@POSIX
@pytest.mark.parametrize("renderer", ["inline", "process"])
def test_a_rendering_failure_degrades_to_plain_lines(
    terminal, monkeypatch, renderer
) -> None:
    master, stream = terminal
    monkeypatch.setenv("PYAMPLICOL_PROGRESS_RENDERER", renderer)
    # In the renderer process the drawing code is not patched: its own
    # failure is the renderer's (next test); here the parent's state
    # snapshot fails.
    monkeypatch.setattr(progress, "_dashboard_lines", _boom)
    monkeypatch.setattr(TtyProgressSink, "_snapshot", _boom)
    sink = TtyProgressSink(stream, color=False)
    assert _generate(sink) == "done"
    text = _read_all(master)
    assert _warnings(text) == 1
    assert "Exporting" in text


@POSIX
def test_an_estimator_failure_degrades_to_plain_lines(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.setenv("PYAMPLICOL_PROGRESS_RENDERER", "inline")
    monkeypatch.setattr(progress, "_task_estimate", _boom)
    sink = TtyProgressSink(stream, color=False)
    assert _generate(sink) == "done"
    assert _warnings(_read_all(master)) == 1


def test_a_line_log_failure_switches_the_log_off(monkeypatch) -> None:
    monkeypatch.setenv("PYAMPLICOL_PROGRESS_RENDERER", "inline")
    monkeypatch.setattr(progress, "_format_event", _boom)
    stream = io.StringIO()
    sink = progress.progress_sink("log", stream=stream)
    assert _generate(sink) == "done"
    assert _warnings(stream.getvalue()) == 1


def test_a_failing_sink_and_a_refused_value_never_fail_the_generation(
    monkeypatch,
) -> None:
    class Failing:
        def emit(self, event):
            raise ValueError("a sink that raises")

    assert _generate(Failing()) == "done"
    # A prior the event refuses (non-finite detail) is the display's problem.
    monkeypatch.setattr(GenerationHistory, "prior", lambda self, path: float("nan"))
    monkeypatch.setattr(
        GenerationHistory, "later", lambda self, started: (float("inf"), True)
    )
    assert _generate(progress.progress_sink("log", stream=io.StringIO()), signature="x")


def test_a_history_that_cannot_be_read_or_written(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(GenerationHistory, "_load", _boom)
    assert _generate(io_sink(), signature="unreadable") == "done"
    monkeypatch.undo()
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(GenerationHistory, "_path", _boom)
    assert _generate(io_sink(), signature="unwritable") == "done"
    assert generation_progress.history_signature("k", object()) is not None
    assert generation_progress.history_signature("k", _Unrepresentable()) is None


class _Unrepresentable:
    @property
    def model(self):
        raise RuntimeError("a config that cannot be read")


def io_sink():
    return progress.progress_sink("log", stream=io.StringIO())


def test_resource_sampling_failures_keep_the_last_usage() -> None:
    monitor = ResourceUsageMonitor(root_pid=os.getpid(), snapshotter=_boom)
    monitor.start()
    monitor._sample_once()
    monitor.close()
    assert monitor.usage.current_rss_bytes is None
    good = ResourceUsageMonitor()
    good._sample_once()
    good._busy_cores = _boom  # type: ignore[method-assign]
    good._sample_once()
    assert good.usage.current_rss_bytes is not None


@POSIX
def test_a_renderer_that_dies_or_hangs_never_holds_the_work(
    terminal, monkeypatch
) -> None:
    import threading

    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    # The terminal reads what is written (as a real one does).
    chunks: list[str] = []
    reader = threading.Thread(
        target=lambda: chunks.append(_read_until_quiet(master, 30.0)), daemon=True
    )
    reader.start()
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=5000))
    dashboard = sink._dashboard
    assert isinstance(dashboard, progress._SidecarDashboard)
    pid = dashboard._process.pid
    # A stopped renderer: the pipe fills, nothing may block.
    os.kill(pid, signal.SIGSTOP)
    started = time.monotonic()
    try:
        for index in range(5000):
            sink.emit(ProgressUpdate("root", index + 1, 5000, message="x" * 2000))
            if index % 50 == 0:
                sink._message_stream.write(f"a kept line {index}\n")
    finally:
        os.kill(pid, signal.SIGCONT)
    assert time.monotonic() - started < 20.0
    # Resumed, it catches up: no kept line is lost (states are dropped).
    sink.emit(ProgressEnd("root"))
    sink.close()
    reader.join(timeout=40.0)
    assert "a kept line 4950" in "".join(chunks)


def test_a_renderer_falling_too_far_behind_is_replaced(monkeypatch) -> None:
    class Stuck:
        pid = 0
        returncode = None

        class stdin:  # the Popen attribute's shape
            @staticmethod
            def fileno() -> int:
                return write_end

            @staticmethod
            def close() -> None:
                os.close(write_end)

        def kill(self) -> None:
            pass

    read_end, write_end = os.pipe()
    ack_read, ack_write = os.pipe()
    link = progress._RendererLink(Stuck(), ack_read)
    monkeypatch.setattr(progress._RendererLink, "_BACKLOG", 1024 * 1024)
    monkeypatch.setattr(progress._RendererLink, "reap", lambda self, timeout: None)
    message = {"op": "msg", "text": "x" * 4096}
    sent = [link.send(message) for _ in range(400)]
    assert sent[0] and not sent[-1]  # stuck: the caller takes over
    for fd in (read_end, ack_write):
        os.close(fd)
    link.abandon()


@POSIX
def test_a_killed_renderer_is_replaced(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=3))
    dashboard = sink._dashboard
    os.kill(dashboard._process.pid, signal.SIGKILL)
    time.sleep(0.3)
    for index in range(3):
        sink.emit(ProgressUpdate("root", index + 1, 3))
        time.sleep(0.06)
    sink.emit(ProgressEnd("root"))
    sink.close()
    assert dashboard._fallback is not None
    _read_all(master)


@POSIX
def test_term_dumb_gets_the_line_display_and_no_colour(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.setenv("TERM", "dumb")
    assert not progress.terminal_capable(stream)
    assert not progress.color_wanted(stream)
    sink = progress.progress_sink("auto", stream=stream)
    assert not isinstance(sink, TtyProgressSink)
    explicit = TtyProgressSink(stream)
    explicit.emit(ProgressStart("root", "Generating processes", total=1))
    explicit.emit(ProgressEnd("root", elapsed_seconds=1.0))
    explicit.close()
    progress.close_progress_sink(sink)
    text = _read_all(master)
    assert "\x1b[" not in text
    assert "Generating processes done in 1s" in text


@POSIX
def test_no_color_turns_auto_colour_off(terminal, monkeypatch) -> None:
    from pyamplicol.cli.main import _color_enabled

    _master, stream = terminal
    assert _color_enabled("auto", stream)
    monkeypatch.setenv("NO_COLOR", "1")
    assert not _color_enabled("auto", stream)
    assert _color_enabled("always", stream)
    monkeypatch.setenv("NO_COLOR", "")
    assert _color_enabled("auto", stream)


def test_a_pool_run_inside_a_callers_interrupt_handler_is_not_interrupted(
    monkeypatch,
) -> None:
    """M7: only an interrupt of this call stops the pool's workers."""

    from concurrent.futures import ThreadPoolExecutor

    from pyamplicol.generation import service

    calls: list[str] = []
    monkeypatch.setattr(
        generation_progress, "cancel_generation", lambda: calls.append("cancel")
    )
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(time.sleep, 0.01).result()
        service._shutdown_process_executor(executor, interrupted=False)
    assert calls == []
    executor = ThreadPoolExecutor(max_workers=1)
    service._shutdown_process_executor(executor, interrupted=True)
    assert calls == ["cancel"]
    generation_progress._CANCELLED.clear()
