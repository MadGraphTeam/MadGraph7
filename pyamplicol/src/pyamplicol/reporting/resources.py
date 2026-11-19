# SPDX-License-Identifier: 0BSD
"""Portable, read-only process-tree resource sampling for live progress."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
import weakref
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
    peak_process_count: int | None = None
    # The cores the process tree kept busy over the last sampling interval
    # (its CPU seconds over the interval's wall seconds; None until two
    # samples, or where the platform reports no CPU time).
    busy_cores: float | None = None


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
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
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


# The progress display's own processes (the dashboard's renderer): left out
# of every process-tree sample, and the resource usage of those reaped, which
# ``process_tree_peak_memory`` takes out of getrusage(RUSAGE_CHILDREN): the
# display does not make a generation one that "started processes".
_MONITOR_LOCK = threading.Lock()
_MONITOR_PIDS: set[int] = set()
_MONITOR_ENDED = [0, 0.0]  # largest peak (bytes), CPU seconds


def register_monitor_process(pid: int) -> None:
    """Leave the display process ``pid`` out of the resource measurements."""

    with _MONITOR_LOCK:
        _MONITOR_PIDS.add(pid)


def monitor_process_ended(pid: int, peak_rss_bytes: int, cpu_seconds: float) -> None:
    """The display process ``pid`` was reaped with this usage (wait4)."""

    with _MONITOR_LOCK:
        _MONITOR_PIDS.discard(pid)
        _MONITOR_ENDED[0] = max(int(_MONITOR_ENDED[0]), int(peak_rss_bytes))
        _MONITOR_ENDED[1] = float(_MONITOR_ENDED[1]) + float(cpu_seconds)


def _monitor_pids() -> frozenset[int]:
    with _MONITOR_LOCK:
        return frozenset(_MONITOR_PIDS)


def _monitor_ended() -> tuple[int, float]:
    with _MONITOR_LOCK:
        return int(_MONITOR_ENDED[0]), float(_MONITOR_ENDED[1])


class ProcessTreeSampler:
    """Sample one root and its descendants, retaining live re-parented children."""

    def __init__(self, root_pid: int, *, exclude: frozenset[int] = frozenset()) -> None:
        if root_pid < 1:
            raise ValueError("resource sampler root PID must be positive")
        self.root_pid = root_pid
        # Processes left out with their descendants (a progress renderer
        # process measures the tree it belongs to without itself).
        self.exclude = frozenset(exclude)
        self._known_pids = {root_pid}
        # Per process: whether it is multiprocessing's resource tracker (a
        # helper the first spawned process starts, not a worker: "children
        # N" counts the workers), read once from its command line.
        self._helpers: dict[int, bool] = {}

    def _helper(self, pid: int) -> bool:
        known = self._helpers.get(pid)
        if known is None:
            try:
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            except OSError:
                command = b""
            known = b"multiprocessing.resource_tracker" in command
            self._helpers[pid] = known
        return known

    def sample(self, records: dict[int, ProcessInfo]) -> tuple[int, int]:
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)

        excluded = self.exclude | _monitor_pids()
        selected = {
            pid for pid in self._known_pids if pid in records and pid not in excluded
        }
        if self.root_pid in records:
            selected.add(self.root_pid)
        pending = list(selected)
        while pending:
            parent = pending.pop()
            for child in children.get(parent, ()):
                if (
                    child not in selected
                    and child not in excluded
                    and not self._helper(child)
                ):
                    selected.add(child)
                    pending.append(child)

        self._known_pids = selected
        self._helpers = {
            pid: self._helpers[pid] for pid in records if pid in self._helpers
        }
        return sum(records[pid].rss_bytes for pid in selected), len(selected)

    def known_pids(self) -> frozenset[int]:
        """The processes of the tree at the last sample."""

        return frozenset(self._known_pids)


def _tree_cpu_seconds(pids: frozenset[int]) -> float | None:
    """The CPU seconds (user + system, children waited for included) of
    ``pids`` from ``/proc/<pid>/stat``; None where it does not exist. A
    process that ended between samples drops out (its CPU then counts as
    its parent's once waited for)."""

    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        ticks = os.sysconf("SC_CLK_TCK")
    except (AttributeError, OSError, ValueError):
        return None
    total = 0
    for pid in pids:
        try:
            text = (proc / str(pid) / "stat").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        fields = text[text.rfind(")") + 1 :].split()
        try:
            total += sum(int(fields[index]) for index in (11, 12, 13, 14))
        except (IndexError, ValueError):
            continue
    return total / ticks


class ResourceUsageMonitor:
    """Sample aggregate RSS in a daemon thread without affecting generation."""

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 1.0,
        snapshotter: ProcessSnapshot | None = None,
        exclude: frozenset[int] = frozenset(),
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        self._sampler = ProcessTreeSampler(root_pid or os.getpid(), exclude=exclude)
        self._interval_seconds = float(interval_seconds)
        # Default: the monitored tree only (a scan of every process costs
        # ~0.1 ms per process, 0.6 s of Python per sample on a host running
        # 7000 processes: a thread holding the GIL next to the work).
        self._snapshotter = snapshotter or self._tree_snapshot
        self._usage = ResourceUsage()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            return self._usage

    def _tree_snapshot(self) -> dict[int, ProcessInfo]:
        """The root's process tree from the kernel's children lists (Linux),
        with the processes of the last sample that were re-parented away
        from it; every process otherwise (``process_snapshot``)."""

        if platform.system() == "Linux":
            try:
                records = _linux_descendant_snapshot(self._sampler.root_pid)
            except ResourceProbeError:
                return process_snapshot()
            page = os.sysconf("SC_PAGE_SIZE")
            for pid in self._sampler.known_pids() - records.keys():
                base = Path("/proc") / str(pid)
                try:
                    resident = int((base / "statm").read_text().split()[1]) * page
                    ppid = _parse_proc_stat_ppid((base / "stat").read_text())
                except (OSError, ValueError, IndexError):
                    continue
                records[pid] = ProcessInfo(pid, ppid, resident)
            return records
        return process_snapshot()

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
        # A failed sample keeps the last usage: the monitor never fails the
        # work it watches, nor stops on one bad read.
        try:
            rss_bytes, process_count = self._sampler.sample(self._snapshotter())
        except Exception:
            return
        try:
            busy = self._busy_cores()
        except Exception:
            busy = None
        with self._lock:
            peak = max(self._usage.peak_rss_bytes or 0, rss_bytes)
            self._usage = ResourceUsage(
                current_rss_bytes=rss_bytes,
                peak_rss_bytes=peak,
                process_count=process_count,
                peak_process_count=max(
                    self._usage.peak_process_count or 0, process_count
                ),
                busy_cores=busy,
            )

    _cpu_mark: tuple[float, float] | None = None

    def _busy_cores(self) -> float | None:
        """The process tree's busy cores since the previous sample, from the
        CPU seconds of its processes (Linux ``/proc/<pid>/stat``; None
        elsewhere or before two samples)."""

        seconds = _tree_cpu_seconds(self._sampler.known_pids())
        if seconds is None:
            return None
        now = time.monotonic()
        mark, self._cpu_mark = self._cpu_mark, (now, seconds)
        if mark is None or now <= mark[0]:
            return None
        return max(seconds - mark[1], 0.0) / (now - mark[0])


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


# Whether pyAmpliCol owns this process (its command line, or a worker process
# it spawned): only then may peak_memory reset the kernel's high-water mark,
# which is process-wide (it lowers the process's VmHWM and ru_maxrss, and so
# what a parent's wait4, `time -v` or the host program's own accounting
# read). Called as a library, pyAmpliCol never changes it.
_OWNS_PROCESS = False


def own_process_accounting() -> None:
    """Declare that pyAmpliCol owns this process (the command line, a
    worker process it spawned): ``peak_memory`` may then reset the kernel's
    high-water mark at the start of a region. Never call it from a program
    that uses pyAmpliCol as a library: the reset lowers the program's own
    peak RSS (VmHWM, ru_maxrss) for every later reader."""

    global _OWNS_PROCESS
    _OWNS_PROCESS = True


def owns_process_accounting() -> bool:
    """Whether ``own_process_accounting`` was declared in this process."""

    return _OWNS_PROCESS


# The peak_memory regions open now: a region opened inside another one does
# not reset the high-water mark (that would lose the outer region's peak).
# Weak references, so that a region left without __exit__ does not count.
_ACTIVE_REGIONS: weakref.WeakSet[peak_memory] = weakref.WeakSet()
_ACTIVE_REGIONS_LOCK = threading.Lock()


class peak_memory:
    """Measure the peak resident set size of this process over a ``with``
    block (``PeakMemory`` in ``result`` once it exits).

    Linux: the kernel's high-water mark VmHWM of /proc/self/status, reset
    to the current resident set size at the start of the block by writing
    "5" to /proc/self/clear_refs (the peak resident set size only: page
    reference bits are untouched), so the peak is the block's, sampled by
    no timer. The reset is process-wide, so it is done only where
    pyAmpliCol owns the process (``own_process_accounting``: its command
    line and its worker processes); called as a library, the block is
    measured as with a refused reset and the caller's peak is untouched.
    The kernel records the mark from its per-CPU batched resident counters
    when memory is released, so a peak released before the end of the
    block can read short by up to the counters' batch on
    each CPU the process ran on; the resident size at the end is read
    exactly (VmHWM is at least VmRSS in one read). A block opened inside
    another open block does not reset the mark (that would lose the outer
    block's peak) and is measured like a refused reset. Where the reset is
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
        with _ACTIVE_REGIONS_LOCK:
            nested = len(_ACTIVE_REGIONS) > 0
            _ACTIVE_REGIONS.add(self)
        if not nested and _OWNS_PROCESS and _proc_status_kib("VmHWM") is not None:
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
        with _ACTIVE_REGIONS_LOCK:
            _ACTIVE_REGIONS.discard(self)
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


def _linux_descendant_snapshot(
    root_pid: int, proc_root: Path = Path("/proc")
) -> dict[int, ProcessInfo]:
    """``root_pid`` and its live descendants, read from the ``children``
    lists of each thread (/proc/<pid>/task/<tid>/children) and the resident
    pages of /proc/<pid>/statm: cheaper than a scan of every process.
    Raises ResourceProbeError when the kernel has no ``children`` lists."""

    page = os.sysconf("SC_PAGE_SIZE")
    records: dict[int, ProcessInfo] = {}
    pending = [(root_pid, 0)]
    while pending:
        pid, ppid = pending.pop()
        if pid in records:
            continue
        base = proc_root / str(pid)
        try:
            resident = int((base / "statm").read_text().split()[1]) * page
            tasks = os.listdir(base / "task")
        except (OSError, ValueError, IndexError) as exc:
            if pid == root_pid:
                raise ResourceProbeError(f"cannot read {base}: {exc}") from exc
            continue  # a descendant that has just exited
        records[pid] = ProcessInfo(pid, ppid, resident)
        for task in tasks:
            try:
                text = (base / "task" / task / "children").read_text()
            except FileNotFoundError as exc:
                if pid == root_pid and task == str(root_pid):
                    raise ResourceProbeError(
                        "the kernel provides no /proc children lists"
                    ) from exc
                continue
            except OSError:
                continue
            pending.extend((int(child), pid) for child in text.split())
    return records


def _children_usage() -> tuple[int, float] | None:
    """getrusage(RUSAGE_CHILDREN): the largest peak resident set size of
    the terminated, waited-for descendants in bytes, and their CPU seconds;
    None without the resource module."""

    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak = int(usage.ru_maxrss)
    return (
        peak if platform.system() == "Darwin" else peak * 1024,
        float(usage.ru_utime + usage.ru_stime),
    )


@dataclass(frozen=True, slots=True)
class ProcessTreePeakMemory:
    """The peak resident memory of this process and the processes it
    started over a region (``process_tree_peak_memory``).

    ``peak_rss_bytes`` is the value to report and ``bound`` what it is:
    "exact" (no descendant process ran: the process's own peak, its threads
    included), "upper bound" (no descendant ran, only the process lifetime's
    peak is known), "lower bound" (descendants ran: the largest of the
    process's peak, the summed resident size of the tree sampled every
    ``interval`` and the largest descendant's own peak; descendants
    running at the same time may have needed more), "approximate"
    (descendants ran and only the lifetime peak of the process is known)
    or "unknown" (nothing measurable; ``peak_rss_bytes`` None).
    ``process`` is the process's own ``PeakMemory``; ``child_processes``
    whether a descendant ran (None: not observable);
    ``child_peak_rss_bytes`` the largest peak of a descendant that ended
    in the region (getrusage RUSAGE_CHILDREN), when it raised that
    lifetime maximum and exceeds the process's own peak (the kernel credits
    a started process with the resident size of the image it is forked or
    vforked from, so a smaller value says nothing about the descendant);
    ``sampled_peak_rss_bytes`` the largest sampled sum;
    ``method`` the sources."""

    peak_rss_bytes: int | None
    bound: str
    process: PeakMemory | None
    child_processes: bool | None
    child_peak_rss_bytes: int | None
    sampled_peak_rss_bytes: int | None
    method: str


class process_tree_peak_memory:
    """Measure the peak resident memory of this process and its descendant
    processes over a ``with`` block (``ProcessTreePeakMemory`` in
    ``result``); never raises.

    The process itself: ``peak_memory`` (threads share its memory, so
    thread pools, native ones included, are counted). Descendants
    (compilers, worker processes): a daemon thread sums the resident sizes
    of the process tree every ``interval_seconds`` (Linux: the /proc
    ``children`` lists of this process, else a /proc scan; macOS and the
    BSDs: ``ps``), and getrusage(RUSAGE_CHILDREN) gives the largest peak of
    a descendant that ended (Linux only: elsewhere the ``ps`` samples are
    descendants too).

    Only processes started inside the block count: the descendants alive
    when it starts (a pool or helper of the caller, multiprocessing's
    resource tracker) are left out of the samples with their own
    descendants, and when one of them ends inside the block,
    RUSAGE_CHILDREN (which cannot tell it from the block's) is not used."""

    def __init__(self, *, interval_seconds: float = 1.0) -> None:
        self.result: ProcessTreePeakMemory | None = None
        self._interval = float(interval_seconds)
        self._process: peak_memory | None = None
        self._monitor: ResourceUsageMonitor | None = None
        self._children: tuple[int, float] | None = None
        self._monitors: tuple[int, float] = (0, 0.0)
        self._sampler = "none"
        # The descendants alive at the start: not the block's.
        self._existing: frozenset[int] | None = None

    def _snapshot(self) -> dict[int, ProcessInfo]:
        pid = os.getpid()
        if platform.system() == "Linux":
            try:
                records = _linux_descendant_snapshot(pid)
                self._sampler = "/proc children"
                return records
            except ResourceProbeError:
                pass
        self._sampler = "/proc scan" if platform.system() == "Linux" else "ps"
        return process_snapshot()

    def __enter__(self) -> process_tree_peak_memory:
        try:
            self._children = _children_usage()
            self._monitors = _monitor_ended()
            self._existing = self._descendants()
            self._process = peak_memory().__enter__()
            self._monitor = ResourceUsageMonitor(
                interval_seconds=self._interval,
                snapshotter=self._snapshot,
                exclude=self._existing or frozenset(),
            )
            self._monitor.start()
        except Exception:  # the measurement never fails the block
            self._monitor = None
        return self

    def _descendants(self) -> frozenset[int] | None:
        """This process's descendants alive now; None when not readable."""

        try:
            records = self._snapshot()
        except (OSError, ResourceProbeError, subprocess.SubprocessError, ValueError):
            return None
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)
        found: set[int] = set()
        pending = [os.getpid()]
        while pending:
            for child in children.get(pending.pop(), ()):
                if child not in found:
                    found.add(child)
                    pending.append(child)
        return frozenset(found)

    def __exit__(self, *exc: object) -> None:
        try:
            self.result = self._measure()
        except Exception:
            self.result = ProcessTreePeakMemory(
                None, "unknown", None, None, None, None, "unavailable"
            )

    def _measure(self) -> ProcessTreePeakMemory:
        usage = ResourceUsage()
        if self._monitor is not None:
            self._monitor.close()
            usage = self._monitor.usage
        process: PeakMemory | None = None
        if self._process is not None:
            self._process.__exit__(None, None, None)
            process = self._process.result
        linux = platform.system() == "Linux" and self._sampler != "ps"
        before, after = self._children, _children_usage()
        # A process alive at the start that ended inside the block was
        # reaped into RUSAGE_CHILDREN like the block's own: not attributable.
        alive = self._descendants() if self._existing else None
        attributable = self._existing is not None and (
            not self._existing or (alive is not None and self._existing <= alive)
        )
        if not attributable:
            before = after = None
        # The display's processes reaped over the region are not the
        # region's: their CPU time is taken out, and a peak counts only above
        # theirs once one of them raised the descendants' largest peak.
        monitors = _monitor_ended()
        monitor_cpu = monitors[1] - self._monitors[1]
        if before is not None and monitors[0] > self._monitors[0]:
            before = (max(before[0], monitors[0]), before[1])
        ended = (
            linux
            and before is not None
            and after is not None
            and (after[0] > before[0] or after[1] - monitor_cpu > before[1] + 0.005)
        )
        sampled = (usage.peak_process_count or 0) > 1
        if usage.peak_process_count is None and not linux:
            children: bool | None = None
        else:
            children = ended or sampled
        own = None if process is None else process.peak_rss_bytes
        # The kernel credits a started process with the resident size of
        # the process that starts it (the image it is forked or vforked
        # from), so only a peak above this process's own is the started
        # process's.
        child_peak = (
            after[0]
            if children
            and before is not None
            and after is not None
            and after[0] > before[0]
            and (own is None or after[0] > own)
            and (linux or sampled)
            else None
        )
        region = process is not None and process.scope == "profiled region"
        sources = [] if process is None else [process.method]
        if children is None:
            sources.append("descendants not observable")
        else:
            sources.append(
                f"descendants every {self._interval:g} s from {self._sampler}"
                + (
                    " and RUSAGE_CHILDREN"
                    if linux and attributable
                    else " (RUSAGE_CHILDREN not attributable)"
                    if linux
                    else ""
                )
            )
        if not children:
            peak = own
            bound = "exact" if region else "upper bound"
        else:
            values = [
                value
                for value in (own, usage.peak_rss_bytes, child_peak)
                if value is not None
            ]
            peak = max(values) if values else None
            bound = "lower bound" if region else "approximate"
        return ProcessTreePeakMemory(
            peak_rss_bytes=peak,
            bound=bound if peak is not None else "unknown",
            process=process,
            child_processes=children,
            child_peak_rss_bytes=child_peak,
            sampled_peak_rss_bytes=usage.peak_rss_bytes,
            method="; ".join(sources) if peak is not None else "unavailable",
        )


_BOUND_PREFIX = {
    "exact": "",
    "upper bound": "<= ",
    "lower bound": ">= ",
    "approximate": "~ ",
}


def bounded_memory_text(size: int | None, bound: str | None) -> str | None:
    """A peak with its bound's prefix ("<= 2.21 GiB" for an upper bound,
    ">= " a lower bound, "~ " an approximation); None without a size."""

    if size is None:
        return None
    return _BOUND_PREFIX.get(str(bound), "~ ") + memory_text(size)


def memory_text(size: float) -> str:
    """A memory size in bytes with three significant digits."""

    for unit, scale in (("GiB", 1024.0**3), ("MiB", 1024.0**2), ("KiB", 1024.0)):
        if abs(size) >= scale:
            value = size / scale
            # 1000-1023 of a unit: four digits, not "1.01e+03".
            return (
                f"{value:.0f} {unit}" if abs(value) >= 999.5 else f"{value:.3g} {unit}"
            )
    return f"{size:.0f} B" if abs(size) >= 999.5 else f"{size:.3g} B"


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
    "ProcessTreePeakMemory",
    "ProcessTreeSampler",
    "ResourceProbeError",
    "ResourceUsage",
    "ResourceUsageMonitor",
    "bounded_memory_text",
    "memory_text",
    "own_process_accounting",
    "owns_process_accounting",
    "peak_memory",
    "peak_memory_text",
    "process_snapshot",
    "process_tree_peak_memory",
]
