# SPDX-License-Identifier: 0BSD
"""Portable process-tree resource monitoring for report workers."""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0


class ResourceProbeError(RuntimeError):
    """Raised when a host process snapshot cannot be collected."""


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """Resource data for one process."""

    pid: int
    ppid: int
    rss_bytes: int
    cpu_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProcessTreeSample:
    """Aggregate resource data for a root process and its descendants."""

    rss_bytes: int
    child_count: int
    cpu_seconds: float | None
    member_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Latest and peak resource data exposed to report scheduling."""

    available: bool
    current_rss_bytes: int | None
    peak_rss_bytes: int | None
    child_count: int | None
    cpu_seconds: float | None
    wall_seconds: float
    error: str | None = None


TerminationReason = Literal["completed", "timeout", "memory_limit"]


@dataclass(frozen=True, slots=True)
class SupervisedResult:
    """Result of one worker process supervised under optional limits."""

    returncode: int
    reason: TerminationReason
    usage: ResourceUsage


class CompletedProcessLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


class WorkerProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


Snapshotter = Callable[[], Mapping[int, ProcessRecord]]
PsRunner = Callable[[Sequence[str]], CompletedProcessLike]
PopenFactory = Callable[..., WorkerProcess]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
TreeSignaler = Callable[[int, Collection[int], int], None]


def _parse_proc_stat(text: str, *, clock_ticks_per_second: int) -> tuple[int, float]:
    """Return ``(ppid, cpu_seconds)`` from one Linux ``/proc/PID/stat`` row."""

    if clock_ticks_per_second <= 0:
        raise ValueError("clock_ticks_per_second must be positive")
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process-name terminator")
    fields = text[closing_parenthesis + 1 :].split()
    # fields starts at the kernel's field 3 (state).
    if len(fields) < 13:
        raise ValueError("incomplete /proc stat record")
    ppid = int(fields[1])
    cpu_ticks = int(fields[11]) + int(fields[12])
    if min(ppid, cpu_ticks) < 0:
        raise ValueError("negative /proc process value")
    return ppid, cpu_ticks / clock_ticks_per_second


def _parse_proc_status_rss(text: str) -> int:
    """Return RSS bytes from one Linux ``/proc/PID/status`` payload."""

    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2].lower() != "kb":
            raise ValueError(f"invalid VmRSS record: {line!r}")
        rss_kib = int(fields[1])
        if rss_kib < 0:
            raise ValueError("negative VmRSS value")
        return rss_kib * 1024
    return 0


def _parse_ps_output(text: str) -> dict[int, ProcessRecord]:
    """Parse ``ps -axo pid=,ppid=,rss=`` output."""

    records: dict[int, ProcessRecord] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid ps row {line_number}: {line!r}")
        pid, ppid, rss_kib = map(int, fields)
        if min(pid, ppid, rss_kib) < 0:
            raise ValueError(f"negative ps value on row {line_number}")
        records[pid] = ProcessRecord(
            pid=pid,
            ppid=ppid,
            rss_bytes=rss_kib * 1024,
        )
    return records


def _linux_proc_snapshot(
    proc_root: Path = Path("/proc"),
    *,
    clock_ticks_per_second: int | None = None,
) -> dict[int, ProcessRecord]:
    """Collect a Linux process snapshot directly from procfs."""

    if clock_ticks_per_second is None:
        clock_ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise ResourceProbeError(f"cannot enumerate {proc_root}: {error}") from error

    records: dict[int, ProcessRecord] = {}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            status = (entry / "status").read_text(
                encoding="utf-8",
                errors="replace",
            )
            ppid, cpu_seconds = _parse_proc_stat(
                stat,
                clock_ticks_per_second=clock_ticks_per_second,
            )
            pid = int(entry.name)
            records[pid] = ProcessRecord(
                pid=pid,
                ppid=ppid,
                rss_bytes=_parse_proc_status_rss(status),
                cpu_seconds=cpu_seconds,
            )
        except (OSError, ValueError):
            # A process can disappear or become inaccessible during the scan.
            continue
    if not records:
        raise ResourceProbeError(f"{proc_root} yielded no readable processes")
    return records


def _default_ps_runner(command: Sequence[str]) -> CompletedProcessLike:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        return subprocess.run(
            tuple(command),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResourceProbeError(f"cannot execute ps: {error}") from error


def _ps_snapshot(*, runner: PsRunner = _default_ps_runner) -> dict[int, ProcessRecord]:
    """Collect a portable process snapshot through ``ps``."""

    completed = runner(("ps", "-axo", "pid=,ppid=,rss="))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ResourceProbeError(f"ps process probe failed: {detail}")
    try:
        records = _parse_ps_output(completed.stdout)
    except ValueError as error:
        raise ResourceProbeError(str(error)) from error
    if not records:
        raise ResourceProbeError("ps process probe yielded no records")
    return records


def process_snapshot(
    system: str | None = None,
    *,
    proc_root: Path = Path("/proc"),
    ps_runner: PsRunner = _default_ps_runner,
) -> dict[int, ProcessRecord]:
    """Collect one process snapshot on Linux or macOS."""

    host = system or platform.system()
    if host == "Linux":
        try:
            return _linux_proc_snapshot(proc_root)
        except ResourceProbeError:
            return _ps_snapshot(runner=ps_runner)
    if host == "Darwin":
        return _ps_snapshot(runner=ps_runner)
    raise ResourceProbeError(f"unsupported host for resource monitoring: {host!r}")


class ProcessTreeSampler:
    """Aggregate a root and recursively discovered descendants."""

    def __init__(self, root_pid: int) -> None:
        if root_pid <= 0:
            raise ValueError("root_pid must be positive")
        self.root_pid = root_pid
        self._known_pids = {root_pid}

    def sample(self, records: Mapping[int, ProcessRecord]) -> ProcessTreeSample:
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)

        selected = {pid for pid in self._known_pids if pid in records}
        if self.root_pid in records:
            selected.add(self.root_pid)
        pending = list(selected)
        while pending:
            parent = pending.pop()
            for child in children.get(parent, ()):
                if child not in selected:
                    selected.add(child)
                    pending.append(child)

        self._known_pids = selected
        members = tuple(sorted(selected))
        cpu_values = [
            records[pid].cpu_seconds
            for pid in members
            if records[pid].cpu_seconds is not None
        ]
        return ProcessTreeSample(
            rss_bytes=sum(records[pid].rss_bytes for pid in members),
            child_count=max(len(members) - (self.root_pid in selected), 0),
            cpu_seconds=sum(cpu_values) if cpu_values else None,
            member_pids=members,
        )


class ResourceMonitor:
    """Sample current and peak resource usage, optionally in a daemon thread."""

    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        snapshotter: Snapshotter = process_snapshot,
        clock: Clock = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._sampler = ProcessTreeSampler(root_pid)
        self._interval_seconds = float(interval_seconds)
        self._snapshotter = snapshotter
        self._clock = clock
        self._started_at = clock()
        self._usage = ResourceUsage(
            available=False,
            current_rss_bytes=None,
            peak_rss_bytes=None,
            child_count=None,
            cpu_seconds=None,
            wall_seconds=0.0,
        )
        self._last_members: tuple[int, ...] = ()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            return self._usage

    @property
    def member_pids(self) -> tuple[int, ...]:
        with self._lock:
            return self._last_members

    def sample_once(self) -> ResourceUsage:
        """Take one sample and fail open when the host probe is unavailable."""

        wall_seconds = max(self._clock() - self._started_at, 0.0)
        try:
            sample = self._sampler.sample(self._snapshotter())
        except (
            OSError,
            ResourceProbeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            with self._lock:
                self._usage = ResourceUsage(
                    available=False,
                    current_rss_bytes=None,
                    peak_rss_bytes=self._usage.peak_rss_bytes,
                    child_count=None,
                    cpu_seconds=self._usage.cpu_seconds,
                    wall_seconds=wall_seconds,
                    error=str(error),
                )
                return self._usage

        with self._lock:
            peak_rss = max(self._usage.peak_rss_bytes or 0, sample.rss_bytes)
            self._last_members = sample.member_pids
            self._usage = ResourceUsage(
                available=True,
                current_rss_bytes=sample.rss_bytes,
                peak_rss_bytes=peak_rss,
                child_count=sample.child_count,
                cpu_seconds=sample.cpu_seconds,
                wall_seconds=wall_seconds,
            )
            return self._usage

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="pyamplicol-report-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0 * self._interval_seconds, 1.0))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample_once()

    def __enter__(self) -> ResourceMonitor:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _default_popen(command: Sequence[str], **kwargs: object) -> WorkerProcess:
    return subprocess.Popen(tuple(command), **kwargs)


def _signal_process_tree(
    root_process_group: int,
    member_pids: Collection[int],
    selected_signal: int,
) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(root_process_group, selected_signal)
    for pid in member_pids:
        if pid == root_process_group:
            continue
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, selected_signal)


def _terminate_worker(
    process: WorkerProcess,
    *,
    member_pids: Collection[int],
    grace_seconds: float,
    signaler: TreeSignaler,
) -> int:
    signaler(process.pid, member_pids, signal.SIGTERM)
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signaler(process.pid, member_pids, signal.SIGKILL)
        return process.wait(timeout=max(grace_seconds, 1.0))


def supervise_worker(
    command: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    max_rss_bytes: int | None = None,
    interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    snapshotter: Snapshotter = process_snapshot,
    popen_factory: PopenFactory = _default_popen,
    clock: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
    signaler: TreeSignaler = _signal_process_tree,
) -> SupervisedResult:
    """Run a worker in a new session and enforce optional wall/RSS limits."""

    if not command:
        raise ValueError("command must not be empty")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when specified")
    if max_rss_bytes is not None and max_rss_bytes <= 0:
        raise ValueError("max_rss_bytes must be positive when specified")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if termination_grace_seconds < 0:
        raise ValueError("termination_grace_seconds must be non-negative")

    process = popen_factory(tuple(command), start_new_session=True)
    monitor = ResourceMonitor(
        process.pid,
        interval_seconds=interval_seconds,
        snapshotter=snapshotter,
        clock=clock,
    )
    reason: TerminationReason = "completed"
    returncode: int | None = None
    try:
        while True:
            usage = monitor.sample_once()
            if (
                max_rss_bytes is not None
                and usage.current_rss_bytes is not None
                and usage.current_rss_bytes > max_rss_bytes
            ):
                reason = "memory_limit"
                returncode = _terminate_worker(
                    process,
                    member_pids=monitor.member_pids,
                    grace_seconds=termination_grace_seconds,
                    signaler=signaler,
                )
                break
            if timeout_seconds is not None and usage.wall_seconds >= timeout_seconds:
                reason = "timeout"
                returncode = _terminate_worker(
                    process,
                    member_pids=monitor.member_pids,
                    grace_seconds=termination_grace_seconds,
                    signaler=signaler,
                )
                break

            returncode = process.poll()
            if returncode is not None:
                break
            sleeper(interval_seconds)
    finally:
        if process.poll() is None:
            _terminate_worker(
                process,
                member_pids=monitor.member_pids,
                grace_seconds=termination_grace_seconds,
                signaler=signaler,
            )

    if returncode is None:
        returncode = process.wait()
    return SupervisedResult(
        returncode=returncode,
        reason=reason,
        usage=monitor.usage,
    )


__all__ = [
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "DEFAULT_TERMINATION_GRACE_SECONDS",
    "ProcessRecord",
    "ProcessTreeSample",
    "ProcessTreeSampler",
    "ResourceMonitor",
    "ResourceProbeError",
    "ResourceUsage",
    "SupervisedResult",
    "process_snapshot",
    "supervise_worker",
]
