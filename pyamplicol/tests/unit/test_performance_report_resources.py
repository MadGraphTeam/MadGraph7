# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.performance_report.resources import (
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ProcessRecord,
    ProcessTreeSampler,
    ResourceMonitor,
    ResourceProbeError,
    _linux_proc_snapshot,
    _parse_proc_stat,
    _parse_proc_status_rss,
    _parse_ps_output,
    process_snapshot,
    supervise_worker,
)


def test_linux_proc_parsers_and_snapshot_include_cpu_time(tmp_path: Path) -> None:
    stat = "42 (worker (phase 2)) S 7 42 42 0 -1 0 0 0 0 0 200 50"
    status = "Name:\tworker\nVmPeak:\t9000 kB\nVmRSS:\t1536 kB\n"

    assert _parse_proc_stat(stat, clock_ticks_per_second=100) == (7, 2.5)
    assert _parse_proc_status_rss(status) == 1536 * 1024

    process_dir = tmp_path / "42"
    process_dir.mkdir()
    (process_dir / "stat").write_text(stat, encoding="ascii")
    (process_dir / "status").write_text(status, encoding="ascii")

    assert _linux_proc_snapshot(
        tmp_path,
        clock_ticks_per_second=100,
    ) == {
        42: ProcessRecord(
            pid=42,
            ppid=7,
            rss_bytes=1536 * 1024,
            cpu_seconds=2.5,
        )
    }


def test_ps_parser_and_platform_fallback_are_injectable(tmp_path: Path) -> None:
    output = " 12 7 1536\n 13 12 512\n"
    assert _parse_ps_output(output) == {
        12: ProcessRecord(12, 7, 1536 * 1024),
        13: ProcessRecord(13, 12, 512 * 1024),
    }

    @dataclass
    class Completed:
        returncode: int = 0
        stdout: str = output
        stderr: str = ""

    commands: list[tuple[str, ...]] = []

    def run_ps(command: Sequence[str]) -> Completed:
        commands.append(tuple(command))
        return Completed()

    records = process_snapshot(
        "Linux",
        proc_root=tmp_path / "missing",
        ps_runner=run_ps,
    )
    assert records[12].rss_bytes == 1536 * 1024
    assert commands == [("ps", "-axo", "pid=,ppid=,rss=")]


def test_process_tree_aggregates_descendants_and_tracks_reparenting() -> None:
    sampler = ProcessTreeSampler(10)
    first = sampler.sample(
        {
            10: ProcessRecord(10, 1, 100, 1.0),
            11: ProcessRecord(11, 10, 200, 2.0),
            12: ProcessRecord(12, 11, 300, 3.0),
            20: ProcessRecord(20, 1, 900, 9.0),
        }
    )
    assert first.rss_bytes == 600
    assert first.child_count == 2
    assert first.cpu_seconds == 6.0
    assert first.member_pids == (10, 11, 12)

    second = sampler.sample(
        {
            10: ProcessRecord(10, 1, 110, 1.5),
            12: ProcessRecord(12, 1, 310, 3.5),
            20: ProcessRecord(20, 1, 900, 9.5),
        }
    )
    assert second.rss_bytes == 420
    assert second.child_count == 1
    assert second.member_pids == (10, 12)


def test_monitor_tracks_current_peak_cpu_wall_and_probe_failure() -> None:
    now = 100.0
    snapshots = iter(
        (
            {
                10: ProcessRecord(10, 1, 100, 1.0),
                11: ProcessRecord(11, 10, 50, 0.5),
            },
            {10: ProcessRecord(10, 1, 90, 2.0)},
        )
    )
    monitor = ResourceMonitor(
        10,
        snapshotter=lambda: next(snapshots),
        clock=lambda: now,
    )

    now = 101.0
    first = monitor.sample_once()
    assert first.available
    assert first.current_rss_bytes == 150
    assert first.peak_rss_bytes == 150
    assert first.child_count == 1
    assert first.cpu_seconds == 1.5
    assert first.wall_seconds == 1.0

    now = 103.0
    second = monitor.sample_once()
    assert second.current_rss_bytes == 90
    assert second.peak_rss_bytes == 150
    assert second.child_count == 0
    assert second.cpu_seconds == 2.0
    assert second.wall_seconds == 3.0

    failed = ResourceMonitor(
        10,
        snapshotter=lambda: (_ for _ in ()).throw(ResourceProbeError("offline")),
        clock=lambda: 4.0,
    ).sample_once()
    assert not failed.available
    assert failed.current_rss_bytes is None
    assert failed.error == "offline"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class FakeProcess:
    def __init__(self, pid: int = 100) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode


def test_supervisor_enforces_timeout_on_worker_process_group() -> None:
    clock = FakeClock()
    process = FakeProcess()
    popen_calls: list[tuple[tuple[str, ...], bool]] = []
    signals: list[tuple[int, tuple[int, ...], int]] = []

    def popen(command: Sequence[str], *, start_new_session: bool) -> FakeProcess:
        popen_calls.append((tuple(command), start_new_session))
        return process

    def signal_tree(pgid: int, members: object, selected_signal: int) -> None:
        member_tuple = tuple(sorted(members))  # type: ignore[arg-type]
        signals.append((pgid, member_tuple, selected_signal))
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker", "--cell", "one"),
        timeout_seconds=2.0,
        interval_seconds=1.0,
        snapshotter=lambda: {
            100: ProcessRecord(100, 1, 100),
            101: ProcessRecord(101, 100, 50),
        },
        popen_factory=popen,
        clock=clock,
        sleeper=clock.sleep,
        signaler=signal_tree,
    )

    assert popen_calls == [(("worker", "--cell", "one"), True)]
    assert result.reason == "timeout"
    assert result.returncode == -signal.SIGTERM
    assert result.usage.wall_seconds == 2.0
    assert signals == [(100, (100, 101), signal.SIGTERM)]


def test_supervisor_enforces_memory_limit_and_fails_open_on_probe_error() -> None:
    process = FakeProcess()

    def terminate(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    limited = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        snapshotter=lambda: {
            100: ProcessRecord(100, 1, 80),
            101: ProcessRecord(101, 100, 30),
        },
        popen_factory=lambda *_args, **_kwargs: process,
        signaler=terminate,
    )
    assert limited.reason == "memory_limit"
    assert limited.usage.current_rss_bytes == 110
    assert limited.usage.peak_rss_bytes == 110

    completed = FakeProcess()
    completed.returncode = 7
    unavailable = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        snapshotter=lambda: (_ for _ in ()).throw(OSError("unavailable")),
        popen_factory=lambda *_args, **_kwargs: completed,
    )
    assert unavailable.reason == "completed"
    assert unavailable.returncode == 7
    assert not unavailable.usage.available
    assert unavailable.usage.error == "unavailable"


def test_defaults_and_invalid_limits() -> None:
    assert DEFAULT_SAMPLE_INTERVAL_SECONDS == 1.0
    with pytest.raises(ValueError, match="timeout_seconds"):
        supervise_worker(("worker",), timeout_seconds=0)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        supervise_worker(("worker",), max_rss_bytes=0)
