# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools.performance_report.phase_state import (
    WorkerPhaseChannel,
    WorkerPhaseReporter,
    WorkerPhaseStateError,
    read_worker_phase_state,
)


class NanosecondClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now


def test_reporter_publishes_authenticated_atomic_phase_boundaries(
    tmp_path: Path,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    clock = NanosecondClock()
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=42,
        clock_ns=clock,
        track_post_generation_stages=True,
    )

    pre = read_worker_phase_state(channel, expected_pid=42)
    assert (pre.sequence, pre.phase) == (0, "pre-generation")
    assert pre.generation_elapsed_seconds(now_seconds=50.0) is None

    phase = reporter.generation()
    clock.now = 2_000_000_000
    phase.__enter__()
    active = read_worker_phase_state(channel, expected_pid=42)
    assert (active.sequence, active.phase) == (1, "generation")
    assert active.generation_elapsed_seconds(now_seconds=3.25) == 1.25

    clock.now = 3_500_000_000
    phase.__exit__(None, None, None)
    post = read_worker_phase_state(channel, expected_pid=42)
    assert (post.sequence, post.phase) == (2, "post-generation")
    assert post.generation_elapsed_seconds(now_seconds=500.0) == 1.5
    assert not list(tmp_path.glob("*.tmp"))

    clock.now = 4_000_000_000
    reporter.profiling_started()
    profiling = read_worker_phase_state(channel, expected_pid=42)
    assert (profiling.sequence, profiling.phase) == (3, "profiling")
    assert profiling.generation_elapsed_seconds(now_seconds=500.0) == 1.5
    assert profiling.stage_elapsed_seconds("profiling", now_seconds=4.25) == 0.25
    assert profiling.stage_elapsed_seconds("validation", now_seconds=4.25) is None

    clock.now = 5_000_000_000
    reporter.validation_started()
    validation = read_worker_phase_state(channel, expected_pid=42)
    assert (validation.sequence, validation.phase) == (4, "validation")
    assert validation.stage_elapsed_seconds("profiling", now_seconds=500.0) == 1.0
    assert validation.stage_elapsed_seconds("validation", now_seconds=5.5) == 0.5

    clock.now = 6_000_000_000
    reporter.complete()
    complete = read_worker_phase_state(channel, expected_pid=42)
    assert (complete.sequence, complete.phase) == (5, "complete")
    assert complete.stage_elapsed_seconds("profiling", now_seconds=500.0) == 1.0
    assert complete.stage_elapsed_seconds("validation", now_seconds=500.0) == 1.0

    with (
        pytest.raises(WorkerPhaseStateError, match="exactly one"),
        reporter.generation(),
    ):
        pass


def test_phase_state_rejects_tampering_wrong_pid_and_malformed_fields(
    tmp_path: Path,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    WorkerPhaseReporter(channel, worker_pid=42, clock_ns=lambda: 1)

    with pytest.raises(WorkerPhaseStateError, match="PID"):
        read_worker_phase_state(channel, expected_pid=41)

    payload = json.loads(channel.path.read_text(encoding="ascii"))
    payload["phase"] = "generation"
    channel.path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(WorkerPhaseStateError, match="authentication"):
        read_worker_phase_state(channel, expected_pid=42)

    channel.path.write_text('{"abi":"incomplete"}\n', encoding="ascii")
    with pytest.raises(WorkerPhaseStateError, match="fields"):
        read_worker_phase_state(channel, expected_pid=42)


def test_generation_gate_is_released_before_profiling_and_validation(
    tmp_path: Path,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    events: list[str] = []

    @contextmanager
    def gate():
        events.append("gate-enter")
        try:
            yield
        finally:
            state = read_worker_phase_state(channel, expected_pid=42)
            events.append(f"gate-exit-{state.phase}")

    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=42,
        clock_ns=lambda: 1,
        track_post_generation_stages=True,
        generation_gate=gate,
    )
    with reporter.generation():
        events.append("generation")
    reporter.profiling_started()
    events.append("profiling")
    reporter.validation_started()
    reporter.complete()

    assert events == [
        "gate-enter",
        "generation",
        "gate-exit-post-generation",
        "profiling",
    ]
