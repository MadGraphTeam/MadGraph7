# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ctypes
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.performance_report.phase_state import (
    WorkerPhaseChannel,
    WorkerPhaseReporter,
)
from tools.performance_report.resources import (
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    PROCESS_TREE_MEMORY_METRIC_ABI,
    ProcessRecord,
    ProcessTreeSampler,
    ResourceMonitor,
    ResourceProbeError,
    _darwin_process_identity,
    _DarwinProcBsdInfo,
    _linux_proc_snapshot,
    _parse_linux_process_identity,
    _parse_proc_stat,
    _parse_proc_status_rss,
    _parse_ps_output,
    _ProcessIdentity,
    _signal_process_tree,
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


def test_linux_process_identity_uses_one_stat_snapshot() -> None:
    stat = (
        "42 (worker (phase 2)) S 1 77 77 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 12345"
    )

    assert _parse_linux_process_identity(stat) == _ProcessIdentity(
        birth_token=12345,
        process_group=77,
    )

    with pytest.raises(ValueError, match="process identity"):
        _parse_linux_process_identity(stat.replace(" 77 77 ", " 0 77 ", 1))


def test_darwin_process_identity_uses_bsd_birth_and_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcPidInfo:
        argtypes: object = None
        restype: object = None

        def __init__(self) -> None:
            self.short = False
            self.mismatched_pid = False

        def __call__(
            self,
            pid: int,
            flavor: int,
            arg: int,
            buffer: object,
            size: int,
        ) -> int:
            assert flavor == 3
            assert arg == 0
            assert size == ctypes.sizeof(_DarwinProcBsdInfo)
            info = ctypes.cast(
                buffer,
                ctypes.POINTER(_DarwinProcBsdInfo),
            ).contents
            info.pbi_pid = pid + int(self.mismatched_pid)
            info.pbi_pgid = 81
            info.pbi_start_tvsec = 123
            info.pbi_start_tvusec = 456
            return size - 1 if self.short else size

    fake_proc_pidinfo = FakeProcPidInfo()

    class FakeLibproc:
        proc_pidinfo = fake_proc_pidinfo

    monkeypatch.setattr(
        "tools.performance_report.resources._darwin_libproc",
        lambda: FakeLibproc(),
    )

    assert _darwin_process_identity(42) == _ProcessIdentity(
        birth_token=(123, 456),
        process_group=81,
    )
    fake_proc_pidinfo.short = True
    assert _darwin_process_identity(42) is None
    fake_proc_pidinfo.short = False
    fake_proc_pidinfo.mismatched_pid = True
    assert _darwin_process_identity(42) is None


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
    assert first.physical_footprint_bytes is None
    assert first.guard_bytes == 600

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
    assert second.guard_bytes == 420


def test_process_tree_sums_physical_footprint_with_race_fallback() -> None:
    sampler = ProcessTreeSampler(10)
    records = {
        10: ProcessRecord(10, 1, 100, 1.0),
        11: ProcessRecord(11, 10, 200, 2.0),
        12: ProcessRecord(12, 11, 300, 3.0),
    }

    sample = sampler.sample(
        records,
        physical_footprint_probe=lambda _pids: {10: 80, 11: 400},
    )

    assert sample.rss_bytes == 600
    assert sample.physical_footprint_bytes == 780
    assert sample.guard_bytes == 780


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
    assert failed.memory_probe_reason == "process-tree-memory-probe-unavailable"


def test_monitor_tracks_physical_footprint_and_conservative_guard() -> None:
    snapshots = iter(
        (
            {10: ProcessRecord(10, 1, 100)},
            {10: ProcessRecord(10, 1, 140)},
        )
    )
    footprints = iter(({10: 180}, {10: 120}))
    monitor = ResourceMonitor(
        10,
        snapshotter=lambda: next(snapshots),
        physical_footprint_probe=lambda _pids: next(footprints),
        clock=lambda: 0.0,
    )

    first = monitor.sample_once()
    second = monitor.sample_once()

    assert first.current_physical_footprint_bytes == 180
    assert first.current_guard_bytes == 180
    assert second.current_rss_bytes == 140
    assert second.current_physical_footprint_bytes == 120
    assert second.current_guard_bytes == 140
    assert second.peak_rss_bytes == 140
    assert second.peak_physical_footprint_bytes == 180
    assert second.peak_guard_bytes == 180
    assert second.memory_metric_abi == PROCESS_TREE_MEMORY_METRIC_ABI


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

    def popen(
        command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
    ) -> FakeProcess:
        assert env
        popen_calls.append((tuple(command), start_new_session))
        return process

    def signal_tree(pgid: int | None, members: object, selected_signal: int) -> None:
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
    assert result.reason == "worker_timeout"
    assert result.returncode == -signal.SIGTERM
    assert result.usage.wall_seconds == 2.0
    assert signals == [(100, (), signal.SIGTERM)]


def test_process_tree_signaler_broadcasts_once_and_signals_only_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group_signals: list[tuple[int, int]] = []
    individual_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "tools.performance_report.resources.os.killpg",
        lambda pgid, selected_signal: group_signals.append((pgid, selected_signal)),
    )
    monkeypatch.setattr(
        "tools.performance_report.resources.os.kill",
        lambda pid, selected_signal: individual_signals.append(
            (pid, selected_signal)
        ),
    )

    _signal_process_tree(100, (201, 202), signal.SIGTERM)

    assert group_signals == [(100, signal.SIGTERM)]
    assert individual_signals == [
        (201, signal.SIGTERM),
        (202, signal.SIGTERM),
    ]


def test_stubborn_memory_capped_tree_keeps_primary_outcome_until_reaped() -> None:
    clock = FakeClock()
    process = FakeProcess()
    child_alive = True
    kill_seen = False
    post_kill_polls = 0
    signals: list[tuple[int, tuple[int, ...], int]] = []
    observed_phases: list[str | None] = []

    snapshot_calls = 0

    def snapshot() -> dict[int, ProcessRecord]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            return {
                100: ProcessRecord(100, 1, 60),
                101: ProcessRecord(101, 100, 30),
            }
        return (
            {}
            if process.returncode is not None
            else {100: ProcessRecord(100, 1, 110)}
        )

    def identity(pid: int) -> _ProcessIdentity | None:
        if pid == process.pid and process.returncode is None:
            return _ProcessIdentity(birth_token=1, process_group=process.pid)
        if pid == 101 and child_alive:
            return _ProcessIdentity(birth_token=2, process_group=process.pid)
        return None

    def signal_tree(pgid: int | None, members: object, selected_signal: int) -> None:
        nonlocal kill_seen
        signals.append(
            (pgid, tuple(sorted(members)), selected_signal)  # type: ignore[arg-type]
        )
        if selected_signal == signal.SIGKILL:
            kill_seen = True

    def sleep(duration: float) -> None:
        nonlocal child_alive, post_kill_polls
        clock.now += duration
        if not kill_seen:
            return
        post_kill_polls += 1
        if post_kill_polls == 12:
            process.returncode = -signal.SIGKILL
        elif post_kill_polls == 14:
            child_alive = False

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.1,
        termination_grace_seconds=0.5,
        snapshotter=snapshot,
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
        process_identity_probe=identity,
        observation_callback=lambda observation: observed_phases.append(
            observation.phase
        ),
    )

    assert result.reason == "memory_limit"
    assert result.returncode == -signal.SIGKILL
    assert result.signal_name == "SIGKILL"
    assert result.teardown_escalated
    assert result.teardown_seconds > 0.5
    assert signals == [
        (100, (), signal.SIGTERM),
        (100, (), signal.SIGKILL),
    ]
    assert "terminating" in observed_phases
    assert observed_phases.count("waiting-for-reap") >= 12
    assert not child_alive
    assert snapshot_calls > 2


def test_stubborn_wall_timeout_keeps_primary_outcome_until_root_is_reaped() -> None:
    clock = FakeClock()
    process = FakeProcess()
    kill_seen = False
    post_kill_polls = 0
    signals: list[tuple[int | None, tuple[int, ...], int]] = []

    def identity(pid: int) -> _ProcessIdentity | None:
        return (
            _ProcessIdentity(birth_token=1, process_group=process.pid)
            if pid == process.pid and process.returncode is None
            else None
        )

    def signal_tree(
        pgid: int | None,
        members: object,
        selected_signal: int,
    ) -> None:
        nonlocal kill_seen
        signals.append(
            (pgid, tuple(sorted(members)), selected_signal)  # type: ignore[arg-type]
        )
        if selected_signal == signal.SIGKILL:
            kill_seen = True

    def sleep(duration: float) -> None:
        nonlocal post_kill_polls
        clock.now += duration
        if not kill_seen:
            return
        post_kill_polls += 1
        if post_kill_polls == 3:
            process.returncode = -signal.SIGKILL

    result = supervise_worker(
        ("worker",),
        timeout_seconds=0.1,
        interval_seconds=0.1,
        termination_grace_seconds=0.2,
        snapshotter=lambda: (
            {}
            if process.returncode is not None
            else {100: ProcessRecord(100, 1, 10)}
        ),
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
        process_identity_probe=identity,
    )

    assert result.reason == "worker_timeout"
    assert result.returncode == -signal.SIGKILL
    assert result.teardown_escalated
    assert signals == [
        (100, (), signal.SIGTERM),
        (100, (), signal.SIGKILL),
    ]


def test_descendant_group_change_keeps_birth_identity_until_gone() -> None:
    clock = FakeClock()
    process = FakeProcess()
    child_alive = True
    child_process_group = process.pid
    kill_seen = False
    post_kill_polls = 0
    signals: list[tuple[int | None, tuple[int, ...], int]] = []

    def identity(pid: int) -> _ProcessIdentity | None:
        if pid == process.pid and process.returncode is None:
            return _ProcessIdentity(birth_token=1, process_group=process.pid)
        if pid == 101 and child_alive:
            return _ProcessIdentity(
                birth_token=2,
                process_group=child_process_group,
            )
        return None

    def signal_tree(
        pgid: int | None,
        members: object,
        selected_signal: int,
    ) -> None:
        nonlocal child_process_group, kill_seen
        selected_members = tuple(sorted(members))  # type: ignore[arg-type]
        signals.append((pgid, selected_members, selected_signal))
        if selected_signal == signal.SIGTERM:
            child_process_group = 101
        else:
            assert selected_members == (101,)
            kill_seen = True
            process.returncode = -signal.SIGKILL

    def sleep(duration: float) -> None:
        nonlocal child_alive, post_kill_polls
        clock.now += duration
        if not kill_seen:
            return
        post_kill_polls += 1
        if post_kill_polls == 3:
            child_alive = False

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.1,
        termination_grace_seconds=0.2,
        snapshotter=lambda: {
            **(
                {100: ProcessRecord(100, 1, 70)}
                if process.returncode is None
                else {}
            ),
            **({101: ProcessRecord(101, 100, 40)} if child_alive else {}),
        },
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
        process_identity_probe=identity,
    )

    assert result.reason == "memory_limit"
    assert result.returncode == -signal.SIGKILL
    assert result.teardown_escalated
    assert not child_alive
    assert post_kill_polls == 3
    assert signals == [
        (100, (), signal.SIGTERM),
        (100, (101,), signal.SIGKILL),
    ]


def test_natural_exit_does_not_signal_a_reused_descendant_pid() -> None:
    clock = FakeClock()
    process = FakeProcess()
    child_identity: _ProcessIdentity | None = _ProcessIdentity(
        birth_token=2,
        process_group=process.pid,
    )
    signals: list[tuple[int | None, tuple[int, ...], int]] = []

    def identity(pid: int) -> _ProcessIdentity | None:
        if pid == process.pid and process.returncode is None:
            return _ProcessIdentity(birth_token=1, process_group=process.pid)
        if pid == 101:
            return child_identity
        return None

    def sleep(duration: float) -> None:
        nonlocal child_identity
        clock.now += duration
        process.returncode = 0
        child_identity = _ProcessIdentity(
            birth_token=3,
            process_group=process.pid,
        )

    result = supervise_worker(
        ("worker",),
        interval_seconds=0.1,
        snapshotter=lambda: {
            **(
                {100: ProcessRecord(100, 1, 10)}
                if process.returncode is None
                else {}
            ),
            101: ProcessRecord(
                101,
                100 if process.returncode is None else 1,
                10,
            ),
        },
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=lambda pgid, members, selected_signal: signals.append(
            (pgid, tuple(sorted(members)), selected_signal)
        ),
        process_identity_probe=identity,
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert signals == []


def test_completed_root_cannot_leave_a_known_descendant_running() -> None:
    clock = FakeClock()
    process = FakeProcess()
    child_alive = True
    signals: list[tuple[int, tuple[int, ...], int]] = []

    def snapshot() -> dict[int, ProcessRecord]:
        records = (
            {} if process.returncode is not None else {100: ProcessRecord(100, 1, 100)}
        )
        if child_alive:
            records[101] = ProcessRecord(
                101,
                100 if process.returncode is None else 1,
                50,
            )
        return records

    def sleep(duration: float) -> None:
        clock.now += duration
        process.returncode = 0

    def signal_tree(pgid: int | None, members: object, selected_signal: int) -> None:
        nonlocal child_alive
        signals.append(
            (pgid, tuple(sorted(members)), selected_signal)  # type: ignore[arg-type]
        )
        if selected_signal == signal.SIGKILL:
            child_alive = False

    result = supervise_worker(
        ("worker",),
        interval_seconds=0.1,
        termination_grace_seconds=0.1,
        snapshotter=snapshot,
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
        process_identity_probe=lambda pid: (
            _ProcessIdentity(birth_token=pid, process_group=process.pid)
            if (
                (pid == process.pid and process.returncode is None)
                or (pid == 101 and child_alive)
            )
            else None
        ),
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert not child_alive
    assert signals == [
        (100, (), signal.SIGTERM),
        (100, (), signal.SIGKILL),
    ]
    assert result.teardown_escalated
    assert result.teardown_seconds >= 0.1


def test_completed_root_reports_graceful_descendant_teardown() -> None:
    clock = FakeClock()
    process = FakeProcess()
    child_alive = True
    signals: list[tuple[int | None, tuple[int, ...], int]] = []

    def sleep(duration: float) -> None:
        clock.now += duration
        process.returncode = 0

    def signal_tree(
        pgid: int | None,
        members: object,
        selected_signal: int,
    ) -> None:
        nonlocal child_alive
        signals.append(
            (pgid, tuple(sorted(members)), selected_signal)  # type: ignore[arg-type]
        )
        assert selected_signal == signal.SIGTERM
        child_alive = False

    result = supervise_worker(
        ("worker",),
        interval_seconds=0.1,
        termination_grace_seconds=0.2,
        snapshotter=lambda: {
            **(
                {100: ProcessRecord(100, 1, 100)}
                if process.returncode is None
                else {}
            ),
            **({101: ProcessRecord(101, 100, 50)} if child_alive else {}),
        },
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
        process_identity_probe=lambda pid: (
            _ProcessIdentity(birth_token=pid, process_group=process.pid)
            if (
                (pid == process.pid and process.returncode is None)
                or (pid == 101 and child_alive)
            )
            else None
        ),
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert signals == [(100, (), signal.SIGTERM)]
    assert not result.teardown_escalated
    assert result.teardown_seconds > 0.0


def test_supervisor_propagates_symbolica_license_to_worker_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    license_value = "ordinary-test-license"
    monkeypatch.setenv("SYMBOLICA_LICENSE", license_value)
    process = FakeProcess()
    process.returncode = 0
    observed: dict[str, object] = {}

    def popen(
        command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
    ) -> FakeProcess:
        observed["command"] = tuple(command)
        observed["start_new_session"] = start_new_session
        observed["license"] = env.get("SYMBOLICA_LICENSE")
        return process

    result = supervise_worker(
        ("worker", "--cell", "one"),
        snapshotter=lambda: {},
        popen_factory=popen,
    )

    assert observed == {
        "command": ("worker", "--cell", "one"),
        "start_new_session": True,
        "license": license_value,
    }
    assert result.returncode == 0


def test_supervisor_merges_explicit_worker_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYMBOLICA_LICENSE", "ordinary-test-license")
    process = FakeProcess()
    process.returncode = 0
    observed: dict[str, str | None] = {}

    def popen(
        _command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
    ) -> FakeProcess:
        assert start_new_session
        observed["license"] = env.get("SYMBOLICA_LICENSE")
        observed["gate"] = env.get("PYAMPLICOL_NATIVE_COMPILER_GATE_DIR")
        observed["slots"] = env.get("PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT")
        return process

    result = supervise_worker(
        ("worker",),
        environment_overrides={
            "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR": "/tmp/compiler-gate",
            "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT": "4",
        },
        snapshotter=lambda: {},
        popen_factory=popen,
    )

    assert observed == {
        "license": "ordinary-test-license",
        "gate": "/tmp/compiler-gate",
        "slots": "4",
    }
    assert result.returncode == 0


def test_split_worker_scrubs_import_environment_and_sets_measured_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_environment = {
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYAMPLICOL_EXACT_IMPORT_PATHS",
        "PYAMPLICOL_EXACT_PYTHON_REEXEC",
        "PYAMPLICOL_CACHE_DIR",
        "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP",
        "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR",
        "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT",
        "VIRTUAL_ENV",
        "_OLD_VIRTUAL_PATH",
        "__PYVENV_LAUNCHER__",
    }
    for name in import_environment:
        monkeypatch.setenv(name, f"attacker-{name}")
    process = FakeProcess()
    process.returncode = 0
    observed: dict[str, object] = {}

    def popen(
        command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
        cwd: str,
    ) -> FakeProcess:
        observed["command"] = tuple(command)
        observed["start_new_session"] = start_new_session
        observed["cwd"] = cwd
        observed["present"] = import_environment.intersection(env)
        return process

    result = supervise_worker(
        ("worker",),
        scrub_import_environment=True,
        environment_overrides={
            "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR": "/authenticated/gate",
            "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT": "4",
        },
        working_directory=tmp_path,
        snapshotter=lambda: {},
        popen_factory=popen,
    )

    assert result.returncode == 0
    assert observed == {
        "command": ("worker",),
        "start_new_session": True,
        "cwd": str(tmp_path.resolve()),
        "present": {
            "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR",
            "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT",
        },
    }
    with pytest.raises(ValueError, match="environment override"):
        supervise_worker(
            ("worker",),
            scrub_import_environment=True,
            environment_overrides={"PYTHONPATH": "/attacker"},
        )


def test_ordinary_worker_environment_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/ordinary/controller/path")
    process = FakeProcess()
    process.returncode = 0
    observed: dict[str, object] = {}

    def popen(
        _command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
    ) -> FakeProcess:
        assert start_new_session
        observed["pythonpath"] = env.get("PYTHONPATH")
        return process

    result = supervise_worker(
        ("worker",),
        snapshotter=lambda: {},
        popen_factory=popen,
    )

    assert result.returncode == 0
    assert observed["pythonpath"] == "/ordinary/controller/path"


def test_supervisor_enforces_memory_limit_and_preserves_exit_on_probe_error() -> None:
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
    assert limited.memory_limit_reason == "process-tree-rss-limit"

    completed = FakeProcess()
    completed.returncode = 7
    unavailable = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        snapshotter=lambda: (_ for _ in ()).throw(OSError("unavailable")),
        popen_factory=lambda *_args, **_kwargs: completed,
    )
    assert unavailable.reason == "worker_exit"
    assert unavailable.returncode == 7
    assert not unavailable.usage.available
    assert unavailable.usage.error == "unavailable"
    assert (
        unavailable.usage.memory_probe_reason == "process-tree-memory-probe-unavailable"
    )


def test_supervisor_enforces_physical_footprint_over_rss() -> None:
    process = FakeProcess()

    def terminate(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    limited = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 80)},
        physical_footprint_probe=lambda _pids: {100: 150},
        popen_factory=lambda *_args, **_kwargs: process,
        signaler=terminate,
    )

    assert limited.reason == "memory_limit"
    assert limited.memory_limit_bytes == 100
    assert limited.memory_limit_reason == "darwin-process-tree-physical-footprint-limit"
    assert limited.usage.peak_rss_bytes == 80
    assert limited.usage.peak_physical_footprint_bytes == 150
    assert limited.usage.peak_guard_bytes == 150


def test_supervisor_fails_closed_when_worker_exits_during_probe_retry() -> None:
    process = FakeProcess()
    footprint_calls = 0

    def footprint(_pids: object) -> dict[int, int]:
        nonlocal footprint_calls
        footprint_calls += 1
        raise ResourceProbeError("synthetic footprint probe failure")

    def exit_during_retry(_duration: float) -> None:
        process.returncode = 0

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.01,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 80)},
        physical_footprint_probe=footprint,
        popen_factory=lambda *_args, **_kwargs: process,
        sleeper=exit_during_retry,
    )

    assert result.reason == "memory_probe_error"
    assert result.returncode == 0
    assert footprint_calls == 1
    assert not result.usage.available
    assert (
        result.usage.memory_probe_reason
        == "darwin-process-tree-physical-footprint-probe-unavailable"
    )


def test_supervisor_terminates_after_third_required_probe_failure() -> None:
    process = FakeProcess()
    footprint_calls = 0
    signals: list[int] = []

    def footprint(_pids: object) -> dict[int, int]:
        nonlocal footprint_calls
        footprint_calls += 1
        raise ResourceProbeError("synthetic footprint probe failure")

    def terminate(_pgid: int, _members: object, selected_signal: int) -> None:
        signals.append(selected_signal)
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.01,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 80)},
        physical_footprint_probe=footprint,
        popen_factory=lambda *_args, **_kwargs: process,
        sleeper=lambda _duration: None,
        signaler=terminate,
    )

    assert result.reason == "memory_probe_error"
    assert result.returncode == -signal.SIGTERM
    assert footprint_calls == 3
    assert signals == [signal.SIGTERM]
    assert (
        result.usage.memory_probe_reason
        == "darwin-process-tree-physical-footprint-probe-unavailable"
    )


def test_supervisor_clears_recovered_probe_failure_while_worker_is_alive() -> None:
    process = FakeProcess()
    footprint_calls = 0
    sleep_calls = 0

    def footprint(_pids: object) -> dict[int, int]:
        nonlocal footprint_calls
        footprint_calls += 1
        if footprint_calls == 1:
            raise ResourceProbeError("synthetic footprint probe failure")
        return {100: 90}

    def complete_after_recovery(_duration: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            process.returncode = 0

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.01,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 80)},
        physical_footprint_probe=footprint,
        popen_factory=lambda *_args, **_kwargs: process,
        sleeper=complete_after_recovery,
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert footprint_calls >= 2
    assert result.usage.available
    assert result.usage.memory_probe_reason is None
    assert result.usage.peak_guard_bytes == 90


def test_supervisor_enforces_cap_on_first_recovered_probe_sample() -> None:
    process = FakeProcess()
    footprint_calls = 0

    def footprint(_pids: object) -> dict[int, int]:
        nonlocal footprint_calls
        footprint_calls += 1
        if footprint_calls == 1:
            raise ResourceProbeError("synthetic footprint probe failure")
        return {100: 150}

    def terminate(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        max_rss_bytes=100,
        interval_seconds=0.01,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 80)},
        physical_footprint_probe=footprint,
        popen_factory=lambda *_args, **_kwargs: process,
        sleeper=lambda _duration: None,
        signaler=terminate,
    )

    assert result.reason == "memory_limit"
    assert result.memory_limit_reason == (
        "darwin-process-tree-physical-footprint-limit"
    )
    assert footprint_calls == 2
    assert result.usage.memory_probe_reason is None
    assert result.usage.peak_guard_bytes == 150


def test_defaults_and_invalid_limits() -> None:
    assert DEFAULT_SAMPLE_INTERVAL_SECONDS == 1.0
    with pytest.raises(ValueError, match="timeout_seconds"):
        supervise_worker(("worker",), timeout_seconds=0)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        supervise_worker(("worker",), max_rss_bytes=0)
    with pytest.raises(ValueError, match="generation_timeout_seconds"):
        supervise_worker(("worker",), generation_timeout_seconds=0)
    with pytest.raises(ValueError, match="authenticated phase_channel"):
        supervise_worker(("worker",), generation_timeout_seconds=1)
    with pytest.raises(ValueError, match="environment override"):
        supervise_worker(
            ("worker",),
            environment_overrides={"INVALID=NAME": "value"},
        )
    with pytest.raises(ValueError, match="stderr_limit_bytes"):
        supervise_worker(("worker",), stderr_limit_bytes=0)


@pytest.mark.parametrize(
    ("script", "expected_returncode", "expected_signal"),
    (
        ("raise SystemExit(23)", 23, None),
        (
            "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
            -signal.SIGSEGV,
            "SIGSEGV",
        ),
        (
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
            -signal.SIGKILL,
            "SIGKILL",
        ),
    ),
)
def test_supervisor_preserves_worker_exit_and_decodes_signals(
    script: str,
    expected_returncode: int,
    expected_signal: str | None,
) -> None:
    result = supervise_worker(
        (sys.executable, "-c", script),
        interval_seconds=0.01,
        capture_stderr=True,
    )

    assert result.reason == "worker_exit"
    assert result.returncode == expected_returncode
    assert result.signal_name == expected_signal
    assert result.signal_number == (
        None if expected_returncode >= 0 else -expected_returncode
    )
    assert result.pid is not None
    assert result.pid in result.member_pids
    assert result.started_at_utc is not None
    assert result.finished_at_utc is not None
    assert result.supervisor_stderr is None


def test_supervisor_retains_bounded_native_abort_stderr() -> None:
    marker = "native evaluator abort: synthetic fault"
    script = (
        "import os; "
        f"os.write(2, ({marker!r} + '\\n').encode()); "
        "os.abort()"
    )
    result = supervise_worker(
        (sys.executable, "-c", script),
        interval_seconds=0.01,
        capture_stderr=True,
        stderr_limit_bytes=32,
    )

    assert result.reason == "worker_exit"
    assert result.returncode == -signal.SIGABRT
    assert result.signal_name == "SIGABRT"
    assert result.supervisor_stderr is not None
    assert result.supervisor_stderr.endswith("synthetic fault\n")
    assert result.supervisor_stderr_truncated
    assert result.supervisor_stderr_limit_bytes == 32


def test_worker_exit_remains_primary_when_phase_state_also_fails(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    process.returncode = -signal.SIGSEGV
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    channel.path.write_text("{malformed", encoding="ascii")

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        phase_channel=channel,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
    )

    assert result.reason == "worker_exit"
    assert result.returncode == -signal.SIGSEGV
    assert result.signal_name == "SIGSEGV"
    assert result.phase_state_error is not None
    assert "valid JSON" in result.phase_state_error
    assert result.generation_phase is not None
    assert result.generation_phase.supervisor_reason == "worker_exit"


def test_generation_limit_excludes_preparation_and_post_generation(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
    )
    generation = reporter.generation()
    sleep_count = 0
    signals: list[int] = []

    def sleep(_duration: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            clock.now = 50.0
            generation.__enter__()
        elif sleep_count == 2:
            clock.now = 51.0
            generation.__exit__(None, None, None)
        else:
            clock.now = 101.0
            process.returncode = 0

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        signals.append(selected_signal)
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert signals == []
    assert result.generation_phase is not None
    assert result.generation_phase.authenticated
    assert result.generation_phase.supervisor_reason == "completed"
    assert result.generation_phase.final_phase == "post-generation"
    assert result.generation_phase.generation_elapsed_seconds == 1.0


def test_manual_generation_guard_includes_worker_preparation(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
    )

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        generation_guard_includes_preparation=True,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=clock.sleep,
        signaler=signal_tree,
    )

    assert result.reason == "generation_timeout"
    assert result.usage.wall_seconds == 2.0
    assert result.generation_phase is not None
    assert result.generation_phase.final_phase == "pre-generation"
    assert result.generation_phase.generation_elapsed_seconds == 2.0


@pytest.mark.parametrize(
    ("stage", "expected_reason"),
    (
        ("profiling", "profiling_timeout"),
        ("validation", "validation_timeout"),
    ),
)
def test_supervisor_enforces_authenticated_post_generation_stage_budget(
    tmp_path: Path,
    stage: str,
    expected_reason: str,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / f"{stage}.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
        track_post_generation_stages=True,
    )
    with reporter.generation():
        pass
    reporter.profiling_started()
    if stage == "validation":
        reporter.validation_started()

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        timeout_seconds=30.0,
        generation_timeout_seconds=10.0,
        profiling_timeout_seconds=2.0,
        validation_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=clock.sleep,
        signaler=signal_tree,
    )

    assert result.reason == expected_reason
    assert result.returncode == -signal.SIGTERM
    assert result.usage.wall_seconds == 2.0


def test_supervisor_allows_authenticated_stage_budget_without_generation_limit(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "profiling-only.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
        track_post_generation_stages=True,
    )
    with reporter.generation():
        pass
    reporter.profiling_started()

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        profiling_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=clock.sleep,
        signaler=signal_tree,
    )

    assert result.reason == "profiling_timeout"
    assert result.generation_phase is None


@pytest.mark.parametrize(
    ("boundaries", "expected_reason"),
    (
        ((0.0, 3.0, 3.5, 4.0), "generation_timeout"),
        ((0.0, 1.0, 4.0, 4.5), "profiling_timeout"),
        ((0.0, 1.0, 1.5, 4.0), "validation_timeout"),
        # Deterministic timeline precedence chooses the earlier profiling
        # overrun when both closed stages crossed their limits between polls.
        ((0.0, 1.0, 4.0, 7.0), "profiling_timeout"),
    ),
)
def test_supervisor_enforces_closed_stage_boundaries_between_samples(
    tmp_path: Path,
    boundaries: tuple[float, float, float, float],
    expected_reason: str,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / f"{expected_reason}.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
        track_post_generation_stages=True,
    )
    transitioned = False

    def cross_all_boundaries(_duration: float) -> None:
        nonlocal transitioned
        if transitioned:
            clock.now += 1.0
            return
        transitioned = True
        generation_start, generation_end, validation_start, complete = boundaries
        clock.now = generation_start
        generation = reporter.generation()
        generation.__enter__()
        clock.now = generation_end
        generation.__exit__(None, None, None)
        reporter.profiling_started()
        clock.now = validation_start
        reporter.validation_started()
        clock.now = complete
        reporter.complete()

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        timeout_seconds=30.0,
        generation_timeout_seconds=2.0,
        profiling_timeout_seconds=2.0,
        validation_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=cross_all_boundaries,
        signaler=signal_tree,
    )

    assert result.reason == expected_reason
    assert result.returncode == -signal.SIGTERM


@pytest.mark.parametrize(
    ("phase", "expected_phase"),
    (
        ("preparation", "pre-generation"),
        ("generation", "generation"),
        ("profiling", "post-generation"),
    ),
)
def test_cancellation_terminates_the_process_tree_in_every_worker_phase(
    tmp_path: Path,
    phase: str,
    expected_phase: str,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / f"{phase}.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
    )
    generation = reporter.generation()
    if phase == "generation":
        generation.__enter__()
    elif phase == "profiling":
        with generation:
            pass
    cancellation = False
    signals: list[tuple[int, tuple[int, ...], int]] = []

    def sleep(duration: float) -> None:
        nonlocal cancellation
        clock.now += duration
        cancellation = True

    def signal_tree(pgid: int | None, members: object, selected_signal: int) -> None:
        signals.append(
            (pgid, tuple(sorted(members)), selected_signal)  # type: ignore[arg-type]
        )
        process.returncode = -selected_signal

    try:
        result = supervise_worker(
            ("worker",),
            generation_timeout_seconds=60.0,
            phase_channel=channel,
            interval_seconds=1.0,
            snapshotter=lambda: {
                100: ProcessRecord(100, 1, 100),
                101: ProcessRecord(101, 100, 50),
            },
            popen_factory=lambda *_args, **_kwargs: process,
            clock=clock,
            sleeper=sleep,
            signaler=signal_tree,
            cancellation_requested=lambda: cancellation,
        )
    finally:
        if phase == "generation":
            generation.__exit__(None, None, None)

    assert result.reason == "cancelled"
    assert result.returncode == -signal.SIGTERM
    assert signals == [(100, (), signal.SIGTERM)]
    assert result.generation_phase is not None
    assert result.generation_phase.authenticated
    assert result.generation_phase.final_phase == expected_phase


def test_supervisor_accepts_completed_zero_work_generation_phase(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    clock.now = 10.0
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
    )
    with reporter.generation():
        pass
    process.returncode = 0

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert result.reason == "completed"
    assert result.returncode == 0
    assert result.generation_phase is not None
    assert result.generation_phase.authenticated
    assert result.generation_phase.final_sequence == 2
    assert result.generation_phase.final_phase == "post-generation"
    assert result.generation_phase.generation_elapsed_seconds == 0.0


def test_generation_limit_terminates_only_active_generation(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=process.pid,
        clock_ns=lambda: int(clock.now * 1_000_000_000),
    )
    generation = reporter.generation()
    sleep_count = 0

    def sleep(_duration: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            clock.now = 50.0
            generation.__enter__()
        else:
            clock.now += 1.0

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        phase_channel=channel,
        interval_seconds=1.0,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=sleep,
        signaler=signal_tree,
    )

    assert result.reason == "generation_timeout"
    assert result.returncode == -signal.SIGTERM
    assert result.usage.wall_seconds == 52.0
    assert result.generation_phase is not None
    assert result.generation_phase.authenticated
    assert result.generation_phase.supervisor_reason == "generation_timeout"
    assert result.generation_phase.final_phase == "generation"
    assert result.generation_phase.generation_elapsed_seconds == 2.0


def test_generation_phase_monitor_fails_closed_on_malformed_state(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    process = FakeProcess()
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    channel.path.write_text("{malformed", encoding="ascii")

    def signal_tree(_pgid: int, _members: object, selected_signal: int) -> None:
        process.returncode = -selected_signal

    result = supervise_worker(
        ("worker",),
        generation_timeout_seconds=2.0,
        phase_channel=channel,
        snapshotter=lambda: {100: ProcessRecord(100, 1, 100)},
        popen_factory=lambda *_args, **_kwargs: process,
        clock=clock,
        sleeper=clock.sleep,
        signaler=signal_tree,
    )

    assert result.reason == "phase_state_error"
    assert result.returncode == -signal.SIGTERM
    assert result.generation_phase is not None
    assert not result.generation_phase.authenticated
    assert result.generation_phase.supervisor_reason == "phase_state_error"
    assert "valid JSON" in str(result.generation_phase.error)
