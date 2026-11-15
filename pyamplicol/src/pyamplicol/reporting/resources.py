# SPDX-License-Identifier: 0BSD
"""Portable, read-only process-tree resource sampling for live progress."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ResourceProbeError(RuntimeError):
    """Raised when resident-memory information cannot be collected."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    pid: int
    ppid: int
    rss_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    current_rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    process_count: int | None = None


ProcessSnapshot = Callable[[], dict[int, ProcessInfo]]


def _parse_proc_stat_ppid(text: str) -> int:
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process-name terminator")
    fields = text[closing_parenthesis + 1 :].split()
    if len(fields) < 2:
        raise ValueError("incomplete /proc stat record")
    return int(fields[1])


def _parse_proc_status_rss(text: str) -> int:
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2].lower() != "kb":
            raise ValueError(f"invalid VmRSS record: {line!r}")
        return int(fields[1]) * 1024
    return 0


def _parse_ps_output(text: str) -> dict[int, ProcessInfo]:
    records: dict[int, ProcessInfo] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid ps row {line_number}: {line!r}")
        pid, ppid, rss_kib = map(int, fields)
        if min(pid, ppid, rss_kib) < 0:
            raise ValueError(f"negative ps value on row {line_number}")
        records[pid] = ProcessInfo(pid, ppid, rss_kib * 1024)
    return records


def _linux_proc_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    records: dict[int, ProcessInfo] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise ResourceProbeError(f"cannot enumerate {proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            status = (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            )
            pid = int(entry.name)
            records[pid] = ProcessInfo(
                pid=pid,
                ppid=_parse_proc_stat_ppid(stat),
                rss_bytes=_parse_proc_status_rss(status),
            )
        except (OSError, ValueError):
            # Processes may disappear between enumeration and reading.
            continue
    if not records:
        raise ResourceProbeError(f"{proc_root} yielded no readable processes")
    return records


def _ps_snapshot() -> dict[int, ProcessInfo]:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        probe = subprocess.Popen(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        stdout, stderr = probe.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        probe.kill()
        probe.communicate()
        raise ResourceProbeError("ps process probe timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceProbeError(f"cannot execute ps: {exc}") from exc
    if probe.returncode != 0:
        detail = stderr.strip() or f"exit code {probe.returncode}"
        raise ResourceProbeError(f"ps process probe failed: {detail}")
    try:
        records = _parse_ps_output(stdout)
    except ValueError as exc:
        raise ResourceProbeError(str(exc)) from exc
    # The sampling command is itself a short-lived descendant of pyAmpliCol.
    records.pop(probe.pid, None)
    if not records:
        raise ResourceProbeError("ps process probe yielded no records")
    return records


def process_snapshot(system: str | None = None) -> dict[int, ProcessInfo]:
    host = system or platform.system()
    if host == "Linux":
        try:
            return _linux_proc_snapshot()
        except ResourceProbeError:
            return _ps_snapshot()
    if host == "Darwin":
        return _ps_snapshot()
    raise ResourceProbeError(f"unsupported host for RSS monitoring: {host!r}")


class ProcessTreeSampler:
    """Sample one root and its descendants, retaining live re-parented children."""

    def __init__(self, root_pid: int) -> None:
        if root_pid < 1:
            raise ValueError("resource sampler root PID must be positive")
        self.root_pid = root_pid
        self._known_pids = {root_pid}

    def sample(self, records: dict[int, ProcessInfo]) -> tuple[int, int]:
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
        return sum(records[pid].rss_bytes for pid in selected), len(selected)


class ResourceUsageMonitor:
    """Sample aggregate RSS in a daemon thread without affecting generation."""

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 1.0,
        snapshotter: ProcessSnapshot = process_snapshot,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        self._sampler = ProcessTreeSampler(root_pid or os.getpid())
        self._interval_seconds = float(interval_seconds)
        self._snapshotter = snapshotter
        self._usage = ResourceUsage()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            return self._usage

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="pyamplicol-resource-monitor",
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
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            rss_bytes, process_count = self._sampler.sample(self._snapshotter())
        except (OSError, ResourceProbeError, subprocess.SubprocessError, ValueError):
            return
        with self._lock:
            peak = max(self._usage.peak_rss_bytes or 0, rss_bytes)
            self._usage = ResourceUsage(
                current_rss_bytes=rss_bytes,
                peak_rss_bytes=peak,
                process_count=process_count,
            )


@dataclass(frozen=True, slots=True)
class PeakMemory:
    """The peak resident set size of this process over a profiled region
    (``peak_memory``): ``peak_rss_bytes`` the highest resident set size,
    ``start_rss_bytes`` the value at the start of the region,
    ``increase_rss_bytes`` the peak less the start (the memory the region
    added at its peak), in bytes. ``scope`` says what the peak covers:
    "profiled region" (the region's own peak: the kernel's high-water mark
    reset at its start, or a lifetime peak the region raised) or "process
    lifetime" (the lifetime peak, which the region did not raise: the
    region's peak is at most that, and the increase is that of the lifetime
    peak); ``method`` names the source. None fields: not measurable on this
    platform."""

    peak_rss_bytes: int | None
    start_rss_bytes: int | None
    increase_rss_bytes: int | None
    scope: str
    method: str


_PROC_STATUS = Path("/proc/self/status")
_PROC_CLEAR_REFS = Path("/proc/self/clear_refs")


def _proc_status_kib(field: str) -> int | None:
    try:
        text = _PROC_STATUS.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(field + ":"):
            fields = line.split()
            if len(fields) == 3 and fields[2].lower() == "kb":
                return int(fields[1]) * 1024
    return None


def _rusage_peak_bytes() -> int | None:
    """getrusage's ru_maxrss of this process in bytes (kibibytes on Linux
    and the BSDs, bytes on macOS), None without the resource module."""

    try:
        import resource
    except ImportError:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def _current_rss_bytes() -> int | None:
    """The current resident set size: /proc/self/status VmRSS (Linux), else
    ``ps`` of this process (macOS and the BSDs)."""

    rss = _proc_status_kib("VmRSS")
    if rss is not None:
        return rss
    try:
        record = _ps_snapshot().get(os.getpid())
    except ResourceProbeError:
        return None
    return None if record is None else record.rss_bytes


def _lifetime_peak_bytes() -> tuple[int | None, str]:
    """The process lifetime's peak resident set size and its source:
    /proc/self/status VmHWM (Linux), else getrusage's ru_maxrss."""

    hwm = _proc_status_kib("VmHWM")
    if hwm is not None:
        return hwm, "VmHWM"
    return _rusage_peak_bytes(), "ru_maxrss"


class peak_memory:
    """Measure the peak resident set size of this process over a ``with``
    block (``PeakMemory`` in ``result`` once it exits).

    Linux: the kernel's high-water mark VmHWM of /proc/self/status, reset
    to the current resident set size at the start of the block by writing
    "5" to /proc/self/clear_refs (the peak resident set size only: page
    reference bits are untouched), so the peak is the block's, sampled by
    no timer. The kernel records the mark from its per-CPU batched
    resident counters when memory is released, so a peak released before
    the end of the block can read short by up to the counters' batch on
    each CPU the process ran on; the resident size at the end is read
    exactly (VmHWM is at least VmRSS in one read). Where the reset is
    refused (an old kernel, a restricted /proc), and elsewhere (getrusage's
    ru_maxrss, bytes on macOS, with ``ps`` for the start value), only the
    lifetime peak is known: it is the block's peak when the block raised
    it, else an upper bound of the block's peak ("process lifetime"). The
    measurement adds two small file reads (or one ``ps`` call) outside the
    block's work."""

    def __init__(self) -> None:
        self.result: PeakMemory | None = None
        self._start: int | None = None
        self._start_method = "VmRSS"
        self._region = False
        self._lifetime: int | None = None

    def __enter__(self) -> peak_memory:
        self._start = _proc_status_kib("VmRSS")
        self._start_method = "VmRSS"
        if self._start is None:
            self._start = _current_rss_bytes()
            self._start_method = "ps"
        self._region = False
        self._lifetime, _ = _lifetime_peak_bytes()
        if _proc_status_kib("VmHWM") is not None:
            try:
                _PROC_CLEAR_REFS.write_text("5")
                self._region = True
            except OSError:
                self._region = False
        if self._start is None:
            # No current resident set size: the lifetime peak before the
            # block (its increase is what the block can be credited with).
            self._start = self._lifetime
            self._start_method = "ru_maxrss"
        return self

    def __exit__(self, *exc: object) -> None:
        peak, source = _lifetime_peak_bytes()
        if self._region:
            method = "VmHWM after clear_refs"
        else:
            method = (
                source
                if source == "VmHWM"
                else f"{source}, start from {self._start_method}"
            )
        # The lifetime peak is the block's when the block raised it.
        region = self._region or (
            peak is not None and self._lifetime is not None and peak > self._lifetime
        )
        start = self._start
        if peak is not None and start is not None:
            peak = max(peak, start)
        self.result = PeakMemory(
            peak_rss_bytes=peak,
            start_rss_bytes=start,
            increase_rss_bytes=(
                None if peak is None or start is None else max(peak - start, 0)
            ),
            scope="profiled region" if region else "process lifetime",
            method=method if peak is not None else "unavailable",
        )


def memory_text(size: float) -> str:
    """A memory size in bytes with three significant digits."""

    for unit, scale in (("GiB", 1024.0**3), ("MiB", 1024.0**2), ("KiB", 1024.0)):
        if abs(size) >= scale:
            return f"{size / scale:.3g} {unit}"
    return f"{size:.3g} B"


def peak_memory_text(peak: PeakMemory | None) -> str | None:
    """The peak's report text: "2.21 GiB", "<= 2.21 GiB" when only the
    lifetime peak bounds it (scope "process lifetime"); None when nothing
    was measured."""

    if peak is None or peak.peak_rss_bytes is None:
        return None
    bound = "<= " if peak.scope != "profiled region" else ""
    return bound + memory_text(peak.peak_rss_bytes)


__all__ = [
    "PeakMemory",
    "ProcessInfo",
    "ProcessTreeSampler",
    "ResourceProbeError",
    "ResourceUsage",
    "ResourceUsageMonitor",
    "memory_text",
    "peak_memory",
    "peak_memory_text",
    "process_snapshot",
]
