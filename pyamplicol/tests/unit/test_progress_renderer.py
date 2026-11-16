# SPDX-License-Identifier: 0BSD
"""The tty dashboard's renderer process: it keeps the display alive while
the main process holds the GIL, clears its lines at close and when the main
process goes away."""

from __future__ import annotations

import os
import pty
import re
import sys
import time

import pytest

from pyamplicol.reporting.progress import (
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
    TtyProgressSink,
    _SidecarDashboard,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the renderer process needs a POSIX terminal"
)

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


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


@pytest.fixture
def terminal():
    import fcntl
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    stream = os.fdopen(slave, "w", buffering=1, encoding="utf-8")
    try:
        yield master, stream
    finally:
        stream.close()
        os.close(master)


def _hold_gil(seconds: float) -> None:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(100.0)
    try:
        started = time.perf_counter()
        value = 0
        while time.perf_counter() - started < seconds:
            value += 1
    finally:
        sys.setswitchinterval(previous)


def test_renderer_moves_while_the_main_thread_holds_the_gil(
    terminal, monkeypatch
) -> None:
    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=2))
    sink.emit(ProgressStart("root:a", "Exporting", total=4, parent_task_id="root"))
    sink.emit(ProgressUpdate("root:a", completed=1, total=4))
    assert isinstance(sink._dashboard, _SidecarDashboard)
    time.sleep(0.5)
    _read_all(master)
    # Nothing is sent while the GIL is held: only the renderer can draw.
    _hold_gil(6.5)
    during = ANSI.sub("", _read_all(master))
    assert "no progress for" in during
    assert re.search(r"no progress for [5-9]s", during)
    sink.emit(ProgressEnd("root:a"))
    sink.emit(ProgressEnd("root"))
    sink.close()


def test_renderer_clears_its_lines_and_exits_at_close(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    dashboard = sink._dashboard
    assert isinstance(dashboard, _SidecarDashboard)
    time.sleep(0.6)
    sink._message_stream.write("a kept line\n")
    sink.emit(ProgressEnd("root"))
    assert dashboard._process.poll() is not None
    stream.write("after\n")
    text = _read_all(master)
    assert "a kept line" in text
    # The last dashboard paint is cleared before the caller writes again.
    assert text.rindex("\x1b[J") < text.rindex("after")


def test_renderer_cleans_up_when_the_main_process_goes_away(
    terminal, monkeypatch
) -> None:
    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    dashboard = sink._dashboard
    assert isinstance(dashboard, _SidecarDashboard)
    time.sleep(0.5)
    # A crashed main process: its end of the pipe closes without "close".
    dashboard._process.stdin.close()
    dashboard._process.wait(timeout=5)
    text = _read_all(master)
    assert text.rstrip().endswith("\x1b[J") or "\x1b[J" in text[-40:]
    dashboard._stop.set()


def test_inline_renderer_on_request(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.setenv("PYAMPLICOL_PROGRESS_RENDERER", "inline")
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    assert not isinstance(sink._dashboard, _SidecarDashboard)
    sink.emit(ProgressEnd("root"))
    sink.close()
    _read_all(master)
