#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run a command under an aggregate process-tree RSS limit.

The implementation deliberately uses only the Python standard library. Linux
RSS data comes from ``/proc``; macOS uses the platform ``ps`` command. The
child starts in a new process session so the watchdog can terminate both its
process group and descendants that create another process group.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import os
import platform
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

GIB = 1024**3
MIB = 1024**2
DEFAULT_LIMIT_GIB = 30.0
DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_TERMINATE_GRACE = 5.0
MEMORY_LIMIT_EXIT_CODE = 137
WATCHDOG_ERROR_EXIT_CODE = 125


class ProbeError(RuntimeError):
    """Raised when resident-memory information cannot be collected."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """One process record from a platform RSS probe."""

    pid: int
    ppid: int
    pgid: int
    rss_bytes: int


@dataclass(frozen=True, slots=True)
class MemorySample:
    """Aggregate memory and members belonging to the guarded command."""

    rss_bytes: int
    members: tuple[ProcessInfo, ...]


Snapshotter = Callable[[], dict[int, ProcessInfo]]


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _nonnegative_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def _parse_proc_stat(text: str) -> tuple[int, int]:
    """Return ``(ppid, pgid)`` from Linux ``/proc/PID/stat`` text."""

    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process-name terminator")
    fields = text[closing_parenthesis + 1 :].split()
    # fields starts at the kernel's field 3 (state).
    if len(fields) < 3:
        raise ValueError("incomplete /proc stat record")
    return int(fields[1]), int(fields[2])


def _parse_proc_status_rss(text: str) -> int:
    """Return RSS bytes from Linux ``/proc/PID/status`` text."""

    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2].lower() != "kb":
            raise ValueError(f"invalid VmRSS record: {line!r}")
        return int(fields[1]) * 1024
    return 0


def _parse_ps_output(text: str) -> dict[int, ProcessInfo]:
    """Parse ``ps -axo pid=,ppid=,pgid=,rss=`` output."""

    records: dict[int, ProcessInfo] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid ps row {line_number}: {line!r}")
        pid, ppid, pgid, rss_kib = map(int, fields)
        if min(pid, ppid, pgid, rss_kib) < 0:
            raise ValueError(f"negative ps value on row {line_number}")
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_kib * 1024)
    return records


def _linux_proc_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    """Collect a Linux process snapshot directly from procfs."""

    records: dict[int, ProcessInfo] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise ProbeError(f"cannot enumerate {proc_root}: {error}") from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            status_text = (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            )
            ppid, pgid = _parse_proc_stat(stat_text)
            rss_bytes = _parse_proc_status_rss(status_text)
        except (OSError, ValueError):
            # Processes can disappear or become inaccessible between directory
            # enumeration and reading their records.
            continue
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_bytes)
    if not records:
        raise ProbeError(f"{proc_root} yielded no readable process records")
    return records


def _ps_snapshot() -> dict[int, ProcessInfo]:
    """Collect a portable BSD/POSIX ``ps`` process snapshot."""

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss="],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProbeError(f"cannot execute ps: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ProbeError(f"ps process probe failed: {detail}")
    try:
        records = _parse_ps_output(completed.stdout)
    except ValueError as error:
        raise ProbeError(str(error)) from error
    if not records:
        raise ProbeError("ps process probe yielded no records")
    return records


def _parse_darwin_bsdinfo(text: bytes) -> tuple[int, int, int]:
    """Return ``(pid, ppid, pgid)`` from Darwin ``proc_bsdinfo`` bytes."""

    # These stable fields precede variable-size names in libproc.h.  pbi_pgid
    # follows pbi_nfiles after the two fixed 16/32-byte name buffers.
    if len(text) < 104:
        raise ValueError("incomplete Darwin proc_bsdinfo record")
    pid = struct.unpack_from("=I", text, 12)[0]
    ppid = struct.unpack_from("=I", text, 16)[0]
    pgid = struct.unpack_from("=I", text, 100)[0]
    return pid, ppid, pgid


def _parse_darwin_taskinfo_rss(text: bytes) -> int:
    """Return resident bytes from Darwin ``proc_taskinfo`` bytes."""

    if len(text) < 16:
        raise ValueError("incomplete Darwin proc_taskinfo record")
    return struct.unpack_from("=Q", text, 8)[0]


def _darwin_libproc_snapshot() -> dict[int, ProcessInfo]:
    """Collect a Darwin snapshot through libproc when ``ps`` is unavailable."""

    library_name = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    try:
        library = ctypes.CDLL(library_name)
    except OSError as error:
        raise ProbeError(f"cannot load Darwin libproc: {error}") from error

    library.proc_listallpids.argtypes = (ctypes.c_void_p, ctypes.c_int)
    library.proc_listallpids.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    library.proc_pidinfo.restype = ctypes.c_int

    estimated_count = library.proc_listallpids(None, 0)
    if estimated_count <= 0:
        raise ProbeError("Darwin libproc process enumeration failed")
    capacity = estimated_count + 1024
    pid_buffer = (ctypes.c_int * capacity)()
    listed_count = library.proc_listallpids(pid_buffer, ctypes.sizeof(pid_buffer))
    if listed_count <= 0:
        raise ProbeError("Darwin libproc process enumeration yielded no records")

    # PROC_PIDTBSDINFO and PROC_PIDTASKINFO are stable public libproc flavours.
    proc_pidtbsdinfo = 3
    proc_pidtaskinfo = 4
    bsd_buffer = ctypes.create_string_buffer(256)
    task_buffer = ctypes.create_string_buffer(256)
    records: dict[int, ProcessInfo] = {}
    for raw_pid in pid_buffer[: min(listed_count, capacity)]:
        pid = int(raw_pid)
        if pid <= 0:
            continue
        bsd_size = library.proc_pidinfo(
            pid,
            proc_pidtbsdinfo,
            0,
            bsd_buffer,
            len(bsd_buffer),
        )
        task_size = library.proc_pidinfo(
            pid,
            proc_pidtaskinfo,
            0,
            task_buffer,
            len(task_buffer),
        )
        if bsd_size < 104 or task_size < 16:
            continue
        try:
            record_pid, ppid, pgid = _parse_darwin_bsdinfo(
                bsd_buffer.raw[:bsd_size]
            )
            rss_bytes = _parse_darwin_taskinfo_rss(task_buffer.raw[:task_size])
        except ValueError:
            continue
        if record_pid != pid:
            continue
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_bytes)
    if not records:
        raise ProbeError("Darwin libproc process probe yielded no records")
    return records


def process_snapshot(system: str | None = None) -> dict[int, ProcessInfo]:
    """Collect one process snapshot on supported macOS and Linux hosts."""

    host = system or platform.system()
    if host == "Linux":
        try:
            return _linux_proc_snapshot()
        except ProbeError:
            return _ps_snapshot()
    if host == "Darwin":
        try:
            return _ps_snapshot()
        except ProbeError:
            return _darwin_libproc_snapshot()
    raise ProbeError(f"unsupported host for RSS monitoring: {host or '<unknown>'}")


class ProcessTreeSampler:
    """Track one process tree while retaining descendants that re-parent."""

    def __init__(self, root_pid: int, root_pgid: int) -> None:
        self.root_pid = root_pid
        self.root_pgid = root_pgid
        self._known_pids = {root_pid}

    def sample(self, records: dict[int, ProcessInfo]) -> MemorySample:
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)

        selected = {pid for pid in self._known_pids if pid in records} | {
            record.pid for record in records.values() if record.pgid == self.root_pgid
        }
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
        members = tuple(records[pid] for pid in sorted(selected))
        return MemorySample(
            rss_bytes=sum(record.rss_bytes for record in members),
            members=members,
        )


def _format_bytes(value: int) -> str:
    return f"{value / GIB:.3f} GiB"


def _normalized_exit_code(returncode: int) -> int:
    if returncode >= 0:
        return returncode
    return 128 + min(-returncode, 127)


def _signal_members(
    members: Iterable[ProcessInfo], root_pgid: int, selected_signal: int
) -> None:
    try:
        os.killpg(root_pgid, selected_signal)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    for member in members:
        if member.pgid == root_pgid:
            continue
        try:
            os.kill(member.pid, selected_signal)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _terminate_tree(
    process: subprocess.Popen[bytes],
    sampler: ProcessTreeSampler,
    snapshotter: Snapshotter,
    *,
    grace_period: float,
    poll_interval: float,
) -> None:
    try:
        sample = sampler.sample(snapshotter())
    except ProbeError:
        sample = MemorySample(0, ())
    _signal_members(sample.members, sampler.root_pgid, signal.SIGTERM)

    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        try:
            sample = sampler.sample(snapshotter())
        except ProbeError:
            sample = MemorySample(0, ())
        if process.poll() is not None and not sample.members:
            return
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0.01)))

    try:
        sample = sampler.sample(snapshotter())
    except ProbeError:
        sample = MemorySample(0, ())
    _signal_members(sample.members, sampler.root_pgid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(grace_period, 1.0))


def run_guarded(
    command: Sequence[str],
    *,
    limit_bytes: int,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
    snapshotter: Snapshotter = process_snapshot,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run ``command`` and return its shell-compatible exit status."""

    if not command:
        raise ValueError("command must not be empty")
    if limit_bytes <= 0:
        raise ValueError("limit_bytes must be positive")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if terminate_grace < 0:
        raise ValueError("terminate_grace must be non-negative")

    try:
        process = subprocess.Popen(tuple(command), start_new_session=True)
    except FileNotFoundError:
        print(f"memory-watchdog: command not found: {command[0]}", file=stderr)
        return 127
    except OSError as error:
        print(f"memory-watchdog: cannot start command: {error}", file=stderr)
        return 126

    sampler = ProcessTreeSampler(process.pid, process.pid)
    print(
        "memory-watchdog: guarding"
        f" pid={process.pid} limit={_format_bytes(limit_bytes)}"
        f" command={shlex.join(command)}",
        file=stderr,
        flush=True,
    )

    received_signal: int | None = None
    previous_handlers: dict[int, signal.Handlers] = {}

    def remember_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    if threading.current_thread() is threading.main_thread():
        for selected_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, remember_signal)

    peak_rss = 0
    peak_processes = 0
    consecutive_probe_failures = 0
    try:
        while True:
            if received_signal is not None:
                _terminate_tree(
                    process,
                    sampler,
                    snapshotter,
                    grace_period=terminate_grace,
                    poll_interval=poll_interval,
                )
                return 128 + received_signal

            try:
                sample = sampler.sample(snapshotter())
            except ProbeError as error:
                consecutive_probe_failures += 1
                print(
                    "memory-watchdog: RSS probe failed"
                    f" ({consecutive_probe_failures}/3): {error}",
                    file=stderr,
                    flush=True,
                )
                if consecutive_probe_failures >= 3:
                    _terminate_tree(
                        process,
                        sampler,
                        snapshotter,
                        grace_period=terminate_grace,
                        poll_interval=poll_interval,
                    )
                    return WATCHDOG_ERROR_EXIT_CODE
            else:
                consecutive_probe_failures = 0
                peak_rss = max(peak_rss, sample.rss_bytes)
                peak_processes = max(peak_processes, len(sample.members))
                if sample.rss_bytes > limit_bytes:
                    print(
                        "memory-watchdog: RSS limit exceeded:"
                        f" {_format_bytes(sample.rss_bytes)} >"
                        f" {_format_bytes(limit_bytes)} across"
                        f" {len(sample.members)} processes; terminating tree",
                        file=stderr,
                        flush=True,
                    )
                    _terminate_tree(
                        process,
                        sampler,
                        snapshotter,
                        grace_period=terminate_grace,
                        poll_interval=poll_interval,
                    )
                    return MEMORY_LIMIT_EXIT_CODE

            returncode = process.poll()
            if returncode is not None:
                normalized = _normalized_exit_code(returncode)
                print(
                    "memory-watchdog: command finished"
                    f" exit={normalized} peak_rss={_format_bytes(peak_rss)}"
                    f" peak_processes={peak_processes}",
                    file=stderr,
                    flush=True,
                )
                return normalized
            time.sleep(poll_interval)
    finally:
        for selected_signal, previous in previous_handlers.items():
            signal.signal(selected_signal, previous)
        if process.poll() is None:
            _terminate_tree(
                process,
                sampler,
                snapshotter,
                grace_period=terminate_grace,
                poll_interval=poll_interval,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run a command and terminate its process tree when aggregate RSS "
            "exceeds a limit"
        )
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--limit-gib",
        type=_positive_finite,
        help=f"RSS limit in GiB (default: {DEFAULT_LIMIT_GIB:g})",
    )
    limits.add_argument(
        "--limit-mib",
        type=_positive_finite,
        help="RSS limit in MiB; useful for focused watchdog tests",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_finite,
        default=DEFAULT_POLL_INTERVAL,
        help=f"seconds between probes (default: {DEFAULT_POLL_INTERVAL:g})",
    )
    parser.add_argument(
        "--terminate-grace",
        type=_nonnegative_finite,
        default=DEFAULT_TERMINATE_GRACE,
        help=(
            "seconds between TERM and KILL when stopping a tree "
            f"(default: {DEFAULT_TERMINATE_GRACE:g})"
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run, conventionally preceded by --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")

    if arguments.limit_mib is not None:
        limit_bytes = int(arguments.limit_mib * MIB)
    else:
        limit_gib = arguments.limit_gib or DEFAULT_LIMIT_GIB
        limit_bytes = int(limit_gib * GIB)
    return run_guarded(
        command,
        limit_bytes=limit_bytes,
        poll_interval=arguments.poll_interval,
        terminate_grace=arguments.terminate_grace,
    )


if __name__ == "__main__":
    raise SystemExit(main())
