# SPDX-License-Identifier: 0BSD
"""Portable process-tree resource monitoring for report workers."""

from __future__ import annotations

import math
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

from tools.ci.memory_watchdog import (
    DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON,
    DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON,
    MEMORY_PROBE_REASON,
    RSS_LIMIT_REASON,
    DarwinPhysicalFootprintProbe,
)
from tools.ci.memory_watchdog import (
    ProbeError as WatchdogProbeError,
)

from .phase_state import (
    WORKER_PHASE_STATE_ABI,
    WorkerPhaseChannel,
    WorkerPhaseState,
    WorkerPhaseStateError,
    read_worker_phase_state,
)

DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0
DEFAULT_PHASE_STATE_STARTUP_GRACE_SECONDS = 30.0
GENERATION_PHASE_EVIDENCE_ABI = "pyamplicol-report-generation-phase-evidence-v1"
PROCESS_TREE_MEMORY_METRIC_ABI = "pyamplicol-process-tree-memory-metric-v1"
MAX_CONSECUTIVE_MEMORY_PROBE_FAILURES = 3
_WORKER_IMPORT_ENVIRONMENT = frozenset(
    {
        "VIRTUAL_ENV",
        "_OLD_VIRTUAL_PATH",
        "__PYVENV_LAUNCHER__",
    }
)
_EXPLICIT_PYAMPLICOL_WORKER_CONTROLS = frozenset(
    {
        "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR",
        "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT",
    }
)


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
    physical_footprint_bytes: int | None = None
    guard_bytes: int | None = None


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
    current_physical_footprint_bytes: int | None = None
    peak_physical_footprint_bytes: int | None = None
    current_guard_bytes: int | None = None
    peak_guard_bytes: int | None = None
    memory_metric_abi: str | None = None
    memory_probe_reason: str | None = None


TerminationReason = Literal[
    "completed",
    "cancelled",
    "timeout",
    "generation_timeout",
    "memory_limit",
    "memory_probe_error",
    "phase_state_error",
]


@dataclass(frozen=True, slots=True)
class WorkerObservation:
    """One non-authoritative live sample for dashboards and logs."""

    pid: int
    usage: ResourceUsage
    phase: str | None
    phase_sequence: int | None
    member_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenerationPhaseEvidence:
    """Supervisor-authenticated evidence for the worker generation interval."""

    configured_timeout_seconds: float
    supervisor_reason: TerminationReason
    authenticated: bool
    run_id: str
    worker_pid: int
    final_sequence: int | None
    final_phase: str | None
    generation_started_monotonic_ns: int | None
    generation_finished_monotonic_ns: int | None
    generation_elapsed_seconds: float | None
    final_state_sha256: str | None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "abi": GENERATION_PHASE_EVIDENCE_ABI,
            "phase_state_abi": WORKER_PHASE_STATE_ABI,
            "configured_timeout_seconds": self.configured_timeout_seconds,
            "supervisor_reason": self.supervisor_reason,
            "authenticated": self.authenticated,
            "run_id": self.run_id,
            "worker_pid": self.worker_pid,
            "final_sequence": self.final_sequence,
            "final_phase": self.final_phase,
            "generation_started_monotonic_ns": (self.generation_started_monotonic_ns),
            "generation_finished_monotonic_ns": (self.generation_finished_monotonic_ns),
            "generation_elapsed_seconds": self.generation_elapsed_seconds,
            "final_state_sha256": self.final_state_sha256,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class SupervisedResult:
    """Result of one worker process supervised under optional limits."""

    returncode: int
    reason: TerminationReason
    usage: ResourceUsage
    generation_phase: GenerationPhaseEvidence | None = None
    memory_limit_bytes: int | None = None
    memory_limit_reason: str | None = None


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
PhysicalFootprintProbe = Callable[[Collection[int]], Mapping[int, int]]
PsRunner = Callable[[Sequence[str]], CompletedProcessLike]
PopenFactory = Callable[..., WorkerProcess]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
TreeSignaler = Callable[[int, Collection[int], int], None]
ObservationCallback = Callable[[WorkerObservation], None]
CancellationCheck = Callable[[], bool]


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


def _parse_ps_cpu_time(value: str) -> float:
    """Parse BSD ``ps time=`` as cumulative CPU seconds."""

    raw = value.strip()
    if not raw:
        raise ValueError("empty ps CPU time")
    days = 0
    if "-" in raw:
        raw_days, raw = raw.split("-", 1)
        days = int(raw_days)
    fields = raw.split(":")
    if len(fields) == 2:
        hours = 0
        minutes = int(fields[0])
        seconds = float(fields[1])
    elif len(fields) == 3:
        hours = int(fields[0])
        minutes = int(fields[1])
        seconds = float(fields[2])
    else:
        raise ValueError(f"invalid ps CPU time: {value!r}")
    if (
        min(days, hours, minutes) < 0
        or not math.isfinite(seconds)
        or seconds < 0.0
        or seconds >= 60.0
        or (len(fields) == 3 and minutes >= 60)
    ):
        raise ValueError(f"invalid ps CPU time: {value!r}")
    return days * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds


def _parse_darwin_ps_output(text: str) -> dict[int, ProcessRecord]:
    """Parse Darwin process rows including truthful cumulative CPU time."""

    records: dict[int, ProcessRecord] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid Darwin ps row {line_number}: {line!r}")
        pid, ppid, rss_kib = map(int, fields[:3])
        if min(pid, ppid, rss_kib) < 0:
            raise ValueError(f"negative Darwin ps value on row {line_number}")
        records[pid] = ProcessRecord(
            pid=pid,
            ppid=ppid,
            rss_bytes=rss_kib * 1024,
            cpu_seconds=_parse_ps_cpu_time(fields[3]),
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


def _darwin_ps_snapshot(
    *,
    runner: PsRunner = _default_ps_runner,
) -> dict[int, ProcessRecord]:
    completed = runner(("ps", "-axo", "pid=,ppid=,rss=,time="))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ResourceProbeError(f"Darwin ps process probe failed: {detail}")
    try:
        records = _parse_darwin_ps_output(completed.stdout)
    except ValueError as error:
        raise ResourceProbeError(str(error)) from error
    if not records:
        raise ResourceProbeError("Darwin ps process probe yielded no records")
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
        return _darwin_ps_snapshot(runner=ps_runner)
    raise ResourceProbeError(f"unsupported host for resource monitoring: {host!r}")


class ProcessTreeSampler:
    """Aggregate a root and recursively discovered descendants."""

    def __init__(self, root_pid: int) -> None:
        if root_pid <= 0:
            raise ValueError("root_pid must be positive")
        self.root_pid = root_pid
        self._known_pids = {root_pid}

    def sample(
        self,
        records: Mapping[int, ProcessRecord],
        physical_footprint_probe: PhysicalFootprintProbe | None = None,
    ) -> ProcessTreeSample:
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
        rss_bytes = sum(records[pid].rss_bytes for pid in members)
        physical_footprint_bytes: int | None = None
        if physical_footprint_probe is not None:
            try:
                footprints = physical_footprint_probe(members)
            except WatchdogProbeError as error:
                raise ResourceProbeError(str(error)) from error
            physical_footprint_bytes = sum(
                footprints.get(pid, records[pid].rss_bytes) for pid in members
            )
        return ProcessTreeSample(
            rss_bytes=rss_bytes,
            child_count=max(len(members) - (self.root_pid in selected), 0),
            cpu_seconds=(
                sum(cpu_values) if members and len(cpu_values) == len(members) else None
            ),
            member_pids=members,
            physical_footprint_bytes=physical_footprint_bytes,
            guard_bytes=max(
                rss_bytes,
                physical_footprint_bytes
                if physical_footprint_bytes is not None
                else rss_bytes,
            ),
        )


class ResourceMonitor:
    """Sample current and peak resource usage, optionally in a daemon thread."""

    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        snapshotter: Snapshotter = process_snapshot,
        physical_footprint_probe: PhysicalFootprintProbe | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._sampler = ProcessTreeSampler(root_pid)
        self._interval_seconds = float(interval_seconds)
        self._snapshotter = snapshotter
        self._physical_footprint_probe = physical_footprint_probe
        self._clock = clock
        self._started_at = clock()
        self._usage = ResourceUsage(
            available=False,
            current_rss_bytes=None,
            peak_rss_bytes=None,
            child_count=None,
            cpu_seconds=None,
            wall_seconds=0.0,
            current_guard_bytes=None,
            peak_guard_bytes=None,
            memory_metric_abi=PROCESS_TREE_MEMORY_METRIC_ABI,
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
        """Take one sample and record unavailable host metrics."""

        wall_seconds = max(self._clock() - self._started_at, 0.0)
        try:
            records = self._snapshotter()
        except (
            OSError,
            ResourceProbeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            return self._record_probe_failure(
                wall_seconds,
                error,
                MEMORY_PROBE_REASON,
            )
        try:
            sample = self._sampler.sample(
                records,
                physical_footprint_probe=self._physical_footprint_probe,
            )
        except (
            OSError,
            ResourceProbeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            return self._record_probe_failure(
                wall_seconds,
                error,
                (
                    DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON
                    if self._physical_footprint_probe is not None
                    else MEMORY_PROBE_REASON
                ),
            )

        with self._lock:
            peak_rss = max(self._usage.peak_rss_bytes or 0, sample.rss_bytes)
            peak_physical_footprint = self._usage.peak_physical_footprint_bytes
            if sample.physical_footprint_bytes is not None:
                peak_physical_footprint = max(
                    peak_physical_footprint or 0,
                    sample.physical_footprint_bytes,
                )
            assert sample.guard_bytes is not None
            peak_guard = max(
                self._usage.peak_guard_bytes or 0,
                sample.guard_bytes,
            )
            self._last_members = sample.member_pids
            self._usage = ResourceUsage(
                available=True,
                current_rss_bytes=sample.rss_bytes,
                peak_rss_bytes=peak_rss,
                child_count=sample.child_count,
                cpu_seconds=sample.cpu_seconds,
                wall_seconds=wall_seconds,
                current_physical_footprint_bytes=(sample.physical_footprint_bytes),
                peak_physical_footprint_bytes=peak_physical_footprint,
                current_guard_bytes=sample.guard_bytes,
                peak_guard_bytes=peak_guard,
                memory_metric_abi=PROCESS_TREE_MEMORY_METRIC_ABI,
            )
            return self._usage

    def _record_probe_failure(
        self,
        wall_seconds: float,
        error: BaseException,
        reason: str,
    ) -> ResourceUsage:
        with self._lock:
            self._usage = ResourceUsage(
                available=False,
                current_rss_bytes=None,
                peak_rss_bytes=self._usage.peak_rss_bytes,
                child_count=None,
                cpu_seconds=self._usage.cpu_seconds,
                wall_seconds=wall_seconds,
                error=str(error),
                current_physical_footprint_bytes=None,
                peak_physical_footprint_bytes=(
                    self._usage.peak_physical_footprint_bytes
                ),
                current_guard_bytes=None,
                peak_guard_bytes=self._usage.peak_guard_bytes,
                memory_metric_abi=PROCESS_TREE_MEMORY_METRIC_ABI,
                memory_probe_reason=reason,
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


def _worker_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    scrub_import_environment: bool = False,
) -> dict[str, str]:
    """Copy the controller environment explicitly for supervised workers."""

    environment = os.environ.copy()
    if scrub_import_environment:
        for name in tuple(environment):
            if (
                name in _WORKER_IMPORT_ENVIRONMENT
                or name.startswith("PYTHON")
                or name.startswith("PYAMPLICOL")
            ):
                environment.pop(name, None)
    symbolica_license = os.environ.get("SYMBOLICA_LICENSE")
    if symbolica_license is not None:
        environment["SYMBOLICA_LICENSE"] = symbolica_license
    for name, value in (overrides or {}).items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
            or (
                scrub_import_environment
                and (
                    name in _WORKER_IMPORT_ENVIRONMENT
                    or name.startswith("PYTHON")
                    or (
                        name.startswith("PYAMPLICOL")
                        and name not in _EXPLICIT_PYAMPLICOL_WORKER_CONTROLS
                    )
                )
            )
        ):
            raise ValueError("worker environment override is invalid")
        environment[name] = value
    return environment


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
    generation_timeout_seconds: float | None = None,
    phase_channel: WorkerPhaseChannel | None = None,
    max_rss_bytes: int | None = None,
    interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    phase_state_startup_grace_seconds: float = (
        DEFAULT_PHASE_STATE_STARTUP_GRACE_SECONDS
    ),
    environment_overrides: Mapping[str, str] | None = None,
    scrub_import_environment: bool = False,
    working_directory: Path | None = None,
    snapshotter: Snapshotter = process_snapshot,
    physical_footprint_probe: PhysicalFootprintProbe | None = None,
    popen_factory: PopenFactory = _default_popen,
    clock: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
    signaler: TreeSignaler = _signal_process_tree,
    observation_callback: ObservationCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> SupervisedResult:
    """Run a worker and enforce wall, generation-only, and memory limits.

    ``generation_timeout_seconds`` requires an authenticated phase channel.
    Missing, malformed, replayed, or incomplete phase evidence terminates the
    worker rather than silently turning the generation limit into a wall limit.
    """

    if not command:
        raise ValueError("command must not be empty")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when specified")
    if generation_timeout_seconds is not None and (
        generation_timeout_seconds <= 0 or not math.isfinite(generation_timeout_seconds)
    ):
        raise ValueError("generation_timeout_seconds must be positive when specified")
    if (generation_timeout_seconds is None) != (phase_channel is None):
        raise ValueError(
            "generation_timeout_seconds and phase_channel must be specified together"
        )
    if max_rss_bytes is not None and max_rss_bytes <= 0:
        raise ValueError("max_rss_bytes must be positive when specified")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if termination_grace_seconds < 0:
        raise ValueError("termination_grace_seconds must be non-negative")
    if phase_state_startup_grace_seconds < 0:
        raise ValueError("phase_state_startup_grace_seconds must be non-negative")
    if working_directory is not None:
        working_directory = working_directory.expanduser().resolve(strict=True)
        if not working_directory.is_dir():
            raise ValueError("working_directory must be a directory")

    if (
        max_rss_bytes is not None
        and physical_footprint_probe is None
        and platform.system() == "Darwin"
        and snapshotter is process_snapshot
    ):
        try:
            darwin_probe = DarwinPhysicalFootprintProbe()
        except WatchdogProbeError as error:
            raise ResourceProbeError(
                f"Darwin physical-footprint probe is unavailable: {error}"
            ) from error

        def physical_footprint_probe(
            pids: Collection[int],
        ) -> Mapping[int, int]:
            return darwin_probe(pids)

    phase_monitor_started = clock()
    popen_arguments: dict[str, object] = {
        "start_new_session": True,
        "env": _worker_environment(
            environment_overrides,
            scrub_import_environment=scrub_import_environment,
        ),
    }
    if working_directory is not None:
        popen_arguments["cwd"] = os.fspath(working_directory)
    process = popen_factory(tuple(command), **popen_arguments)
    monitor = ResourceMonitor(
        process.pid,
        interval_seconds=interval_seconds,
        snapshotter=snapshotter,
        physical_footprint_probe=physical_footprint_probe,
        clock=clock,
    )
    reason: TerminationReason = "completed"
    returncode: int | None = None
    phase_state: WorkerPhaseState | None = None
    phase_error: str | None = None
    memory_limit_reason: str | None = None
    consecutive_memory_probe_failures = 0
    pending_memory_probe_reason: str | None = None

    def terminate(selected_reason: TerminationReason) -> None:
        nonlocal reason, returncode
        reason = selected_reason
        returncode = _terminate_worker(
            process,
            member_pids=monitor.member_pids,
            grace_seconds=termination_grace_seconds,
            signaler=signaler,
        )

    def observe_phase(now_seconds: float) -> None:
        nonlocal phase_state, phase_error
        assert phase_channel is not None
        try:
            observed = read_worker_phase_state(
                phase_channel,
                expected_pid=process.pid,
            )
        except FileNotFoundError:
            if (
                phase_state is not None
                or now_seconds - phase_monitor_started
                >= phase_state_startup_grace_seconds
            ):
                phase_error = "worker phase-state file is missing"
            return
        except WorkerPhaseStateError as error:
            phase_error = str(error)
            return

        earliest_seconds = phase_monitor_started - interval_seconds
        latest_seconds = now_seconds + interval_seconds
        observed_timestamps = (
            observed.transition_monotonic_ns,
            observed.generation_started_monotonic_ns,
            observed.generation_finished_monotonic_ns,
        )
        if any(
            timestamp is not None
            and not (earliest_seconds <= timestamp / 1_000_000_000 <= latest_seconds)
            for timestamp in observed_timestamps
        ):
            phase_error = (
                "worker phase-state transition timestamp is outside the "
                "supervised process lifetime"
            )
            return
        if phase_state is not None:
            if observed.sequence < phase_state.sequence:
                phase_error = "worker phase-state sequence moved backwards"
                return
            if (
                observed.sequence == phase_state.sequence
                and observed.sha256 != phase_state.sha256
            ):
                phase_error = (
                    "worker phase-state changed without advancing its sequence"
                )
                return
            prior_start = phase_state.generation_started_monotonic_ns
            current_start = observed.generation_started_monotonic_ns
            if prior_start is not None and current_start != prior_start:
                phase_error = "worker generation start timestamp changed"
                return
        phase_state = observed

    try:
        while True:
            if cancellation_requested is not None and cancellation_requested():
                terminate("cancelled")
                break
            if pending_memory_probe_reason is not None:
                returncode = process.poll()
                if returncode is not None:
                    reason = "memory_probe_error"
                    break

            usage = monitor.sample_once()
            now_seconds = clock()
            if observation_callback is not None:
                # Monitoring is deliberately informational. A broken
                # dashboard must never invalidate a measurement.
                with suppress(Exception):
                    observation_callback(
                        WorkerObservation(
                            pid=process.pid,
                            usage=usage,
                            phase=(None if phase_state is None else phase_state.phase),
                            phase_sequence=(
                                None if phase_state is None else phase_state.sequence
                            ),
                            member_pids=monitor.member_pids,
                        )
                    )
            if max_rss_bytes is not None and usage.memory_probe_reason is not None:
                consecutive_memory_probe_failures += 1
                pending_memory_probe_reason = usage.memory_probe_reason
                returncode = process.poll()
                if returncode is not None:
                    reason = "memory_probe_error"
                    break
                if (
                    consecutive_memory_probe_failures
                    >= MAX_CONSECUTIVE_MEMORY_PROBE_FAILURES
                ):
                    terminate("memory_probe_error")
                    break
                sleeper(interval_seconds)
                continue
            if pending_memory_probe_reason is not None:
                returncode = process.poll()
                if returncode is not None:
                    reason = "memory_probe_error"
                    break
                pending_memory_probe_reason = None
                consecutive_memory_probe_failures = 0
            if (
                max_rss_bytes is not None
                and usage.current_guard_bytes is not None
                and usage.current_guard_bytes > max_rss_bytes
            ):
                physical = usage.current_physical_footprint_bytes
                rss = usage.current_rss_bytes
                memory_limit_reason = (
                    DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON
                    if physical is not None and rss is not None and physical >= rss
                    else RSS_LIMIT_REASON
                )
                terminate("memory_limit")
                break
            if phase_channel is not None:
                observe_phase(now_seconds)
                if phase_error is not None:
                    terminate("phase_state_error")
                    break
                if phase_state is not None:
                    generation_elapsed = phase_state.generation_elapsed_seconds(
                        now_seconds=now_seconds
                    )
                    assert generation_timeout_seconds is not None
                    if (
                        generation_elapsed is not None
                        and generation_elapsed >= generation_timeout_seconds
                    ):
                        terminate("generation_timeout")
                        break
            if timeout_seconds is not None and usage.wall_seconds >= timeout_seconds:
                terminate("timeout")
                break

            returncode = process.poll()
            if returncode is not None:
                if phase_channel is not None:
                    if phase_state is None:
                        reason = "phase_state_error"
                        phase_error = (
                            "worker exited without authenticated phase-state evidence"
                        )
                    elif phase_state.phase == "generation":
                        reason = "phase_state_error"
                        phase_error = (
                            "worker exited before closing its generation interval"
                        )
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
    generation_phase = None
    if phase_channel is not None:
        assert generation_timeout_seconds is not None
        generation_phase = GenerationPhaseEvidence(
            configured_timeout_seconds=generation_timeout_seconds,
            supervisor_reason=reason,
            authenticated=phase_state is not None and phase_error is None,
            run_id=phase_channel.run_id,
            worker_pid=process.pid,
            final_sequence=None if phase_state is None else phase_state.sequence,
            final_phase=None if phase_state is None else phase_state.phase,
            generation_started_monotonic_ns=(
                None
                if phase_state is None
                else phase_state.generation_started_monotonic_ns
            ),
            generation_finished_monotonic_ns=(
                None
                if phase_state is None
                else phase_state.generation_finished_monotonic_ns
            ),
            generation_elapsed_seconds=(
                None
                if phase_state is None
                else phase_state.generation_elapsed_seconds(
                    now_seconds=clock(),
                )
            ),
            final_state_sha256=(None if phase_state is None else phase_state.sha256),
            error=phase_error,
        )
    return SupervisedResult(
        returncode=returncode,
        reason=reason,
        usage=monitor.usage,
        generation_phase=generation_phase,
        memory_limit_bytes=max_rss_bytes,
        memory_limit_reason=memory_limit_reason,
    )


__all__ = [
    "DEFAULT_PHASE_STATE_STARTUP_GRACE_SECONDS",
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "DEFAULT_TERMINATION_GRACE_SECONDS",
    "GENERATION_PHASE_EVIDENCE_ABI",
    "MAX_CONSECUTIVE_MEMORY_PROBE_FAILURES",
    "PROCESS_TREE_MEMORY_METRIC_ABI",
    "GenerationPhaseEvidence",
    "ProcessRecord",
    "ProcessTreeSample",
    "ProcessTreeSampler",
    "ResourceMonitor",
    "ResourceProbeError",
    "ResourceUsage",
    "SupervisedResult",
    "WorkerObservation",
    "process_snapshot",
    "supervise_worker",
]
