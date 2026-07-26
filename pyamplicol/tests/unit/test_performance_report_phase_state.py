# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
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
