# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
from dataclasses import FrozenInstanceError

import pytest

from pyamplicol.reporting import (
    CallbackProgressSink,
    LoggingProgressSink,
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
    StreamProgressSink,
    TtyProgressSink,
    progress_sink,
)


def test_progress_events_are_frozen_and_validated() -> None:
    event = ProgressStart("build", "Building", total=3)
    with pytest.raises(FrozenInstanceError):
        event.total = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="exceed"):
        ProgressUpdate("build", completed=4, total=3)


def test_callback_and_stream_sinks_receive_typed_events() -> None:
    received: list[object] = []
    CallbackProgressSink(received.append).emit(ProgressEnd("build"))
    assert received == [ProgressEnd("build")]

    stream = io.StringIO()
    StreamProgressSink(stream).emit(ProgressUpdate("build", 2, 3, "currents"))
    assert stream.getvalue() == "build 2/3: currents\n"


def test_auto_progress_uses_logging_for_non_tty_streams() -> None:
    assert isinstance(progress_sink("auto", stream=io.StringIO()), LoggingProgressSink)
    assert isinstance(progress_sink("tty", stream=io.StringIO()), TtyProgressSink)


def test_logging_progress_rate_limits_updates_but_always_emits_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = LoggingProgressSink(minimum_interval=60.0)
    with caplog.at_level("INFO", logger="pyamplicol.progress"):
        sink.emit(ProgressStart("build", "Building", 3))
        sink.emit(ProgressUpdate("build", 1, 3))
        sink.emit(ProgressUpdate("build", 2, 3))
        sink.emit(ProgressUpdate("build", 3, 3))
        sink.emit(ProgressEnd("build"))
    messages = [record.getMessage() for record in caplog.records]
    assert "build 2/3" not in messages
    assert "build 3/3" in messages
    assert messages[-1] == "build done"


def test_tty_progress_accepts_concurrent_phase_shape() -> None:
    stream = io.StringIO()
    sink = TtyProgressSink(stream)
    sink.emit(ProgressStart("dag", "Building DAG", 2))
    sink.emit(ProgressUpdate("dag", 1, 2, "currents"))
    sink.emit(ProgressUpdate("dag", 2, 2))
    sink.emit(ProgressEnd("dag"))
    assert "Building DAG" in stream.getvalue()


def test_tty_progress_is_colored_when_requested() -> None:
    stream = io.StringIO()
    sink = TtyProgressSink(stream, color=True)
    sink.emit(ProgressStart("profile", "Profiling runtime", 1))
    sink.emit(ProgressUpdate("profile", 1, 1, "measured"))
    sink.emit(ProgressEnd("profile"))

    assert "Profiling runtime" in stream.getvalue()
    assert "measured" in stream.getvalue()
    assert "\x1b[" in stream.getvalue()
