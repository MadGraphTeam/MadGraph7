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
def terminal(monkeypatch):
    # A capable terminal (TERM=dumb gets the line display).
    monkeypatch.setenv("TERM", "xterm-256color")
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


def test_renderer_writes_nothing_once_its_input_ends_without_close(
    terminal, monkeypatch
) -> None:
    master, stream = terminal
    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    dashboard = sink._dashboard
    assert isinstance(dashboard, _SidecarDashboard)
    time.sleep(0.6)
    _read_all(master)
    # A crashed main process: its end of the pipe closes without "close".
    dashboard._process.stdin.close()
    dashboard._process.wait(timeout=5)
    # The terminal may be the shell's again: no clear (a heartbeat frame
    # drawn just before the pipe closed may still be read here).
    assert "\x1b[J" not in _read_all(master)
    dashboard._stop.set()


_KILLED_PARENT = """
import os, sys, time
from pyamplicol.reporting.progress import ProgressStart, TtyProgressSink
stream = os.fdopen(int(sys.argv[1]), "w", buffering=1, encoding="utf-8")
sink = TtyProgressSink(stream, color=False)
sink.emit(ProgressStart("root", "Generating processes", total=1))
print(sink._dashboard._process.pid, flush=True)
time.sleep(60)
"""


@pytest.mark.parametrize("signal_name", ["SIGKILL", "SIGTERM"])
def test_renderer_leaves_the_terminal_alone_after_its_parent_is_killed(
    terminal, signal_name
) -> None:
    """M1: after SIGTERM / SIGKILL of the main process the renderer never
    clears and draws nothing after the line the shell prints next: that
    line survives."""

    import signal
    import subprocess

    master, stream = terminal
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    env.pop("PYAMPLICOL_PROGRESS_RENDERER", None)
    parent = subprocess.Popen(
        [sys.executable, "-c", _KILLED_PARENT, str(stream.fileno())],
        pass_fds=(stream.fileno(),),
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    renderer = int(parent.stdout.readline())
    time.sleep(1.0)
    _read_all(master)
    parent.send_signal(getattr(signal, signal_name))
    parent.wait(timeout=10)
    # What the shell prints after its child is gone.
    stream.write("SHELL-PROMPT\n")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(renderer, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(renderer, signal.SIGKILL)
        pytest.fail("the renderer outlived its parent by 10 s")
    after = _read_all(master)
    assert "SHELL-PROMPT" in after
    assert "\x1b[" not in after.split("SHELL-PROMPT", 1)[1]
    # Never a clear; a redraw in place of its own lines may still land
    # while the parent is exiting (before the shell's line, harmless).
    assert "\x1b[J" not in after


def test_inline_renderer_on_request(terminal, monkeypatch) -> None:
    master, stream = terminal
    monkeypatch.setenv("PYAMPLICOL_PROGRESS_RENDERER", "inline")
    sink = TtyProgressSink(stream, color=False)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    assert not isinstance(sink._dashboard, _SidecarDashboard)
    sink.emit(ProgressEnd("root"))
    sink.close()
    _read_all(master)


def test_log_renderer_writes_status_lines_while_the_gil_is_held(
    tmp_path, monkeypatch
) -> None:
    from pyamplicol.reporting import progress

    monkeypatch.delenv("PYAMPLICOL_PROGRESS_RENDERER", raising=False)
    path = tmp_path / "log.txt"
    with open(path, "w", encoding="utf-8") as stream:
        sink = progress.progress_sink("log", stream=stream)
        assert isinstance(sink, progress._SidecarLineSink)
        sink.emit(ProgressStart("root", "Generating processes", total=4))
        sink.emit(ProgressUpdate("root", completed=1, total=4))
        _hold_gil(7.0)
        sink.emit(ProgressEnd("root"))
        progress.close_progress_sink(sink)
    lines = path.read_text(encoding="utf-8").splitlines()
    status = [line for line in lines if line.startswith("Generating processes: ")]
    assert status, lines
    assert any("no progress for" in line for line in status)
    assert lines[-1].startswith("Generating processes done")


_EXITING_PARENT = """
import os, sys, time
from pyamplicol.reporting.progress import ProgressStart, TtyProgressSink
stream = os.fdopen(int(sys.argv[1]), "w", buffering=1, encoding="utf-8")
# A large process exits slowly (its memory is torn down for ~0.1 s).
ballast = bytearray(1024**3)
ballast[::4096] = b"x" * len(ballast[::4096])
sink = TtyProgressSink(stream, color=False)
sink.emit(ProgressStart("root", "Generating processes", total=10))
print("ready", flush=True)
time.sleep(60)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PF_EXITING is Linux")
@pytest.mark.parametrize("signal_name", ["SIGKILL", "SIGTERM"])
def test_renderer_draws_nothing_once_its_parent_begins_to_exit(
    terminal, signal_name
) -> None:
    """While a killed parent is still exiting (re-parenting and the pipe's
    end come only after its memory is freed) the renderer no longer draws:
    it reads the parent's exiting flag."""

    import select
    import signal
    import subprocess
    import threading

    master, stream = terminal
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    env.pop("PYAMPLICOL_PROGRESS_RENDERER", None)
    for run in range(3):
        parent = subprocess.Popen(
            [sys.executable, "-c", _EXITING_PARENT, str(stream.fileno())],
            pass_fds=(stream.fileno(),),
            stdout=subprocess.PIPE,
            env=env,
            text=True,
        )
        assert parent.stdout.readline().strip() == "ready"
        chunks: list[tuple[float, bytes]] = []
        stop = threading.Event()

        def read(stop: threading.Event = stop, chunks: list = chunks) -> None:
            while not stop.is_set():
                ready, _, _ = select.select([master], [], [], 0.05)
                if ready:
                    try:
                        chunks.append((time.monotonic(), os.read(master, 65536)))
                    except OSError:
                        return

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        time.sleep(0.5 + 0.08 * run)
        killed = time.monotonic()
        parent.send_signal(getattr(signal, signal_name))
        parent.wait(timeout=30)
        time.sleep(1.0)
        stop.set()
        reader.join()
        late = [data for when, data in chunks if when > killed + 0.002]
        assert not any(b"\x1b[" in data for data in late), (run, late)
