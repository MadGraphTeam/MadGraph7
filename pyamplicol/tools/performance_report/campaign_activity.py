# SPDX-License-Identifier: 0BSD
"""Bounded campaign-idle census that excludes report-only publication work."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_MEASUREMENT_ACTIONS = frozenset({"populate", "_worker", "_prepare"})
_INSTALL_ENTRYPOINTS = frozenset(
    {
        "install_dependencies.py",
        "prepare_source_runtime.py",
    }
)
_NATIVE_BUILD_EXECUTABLES = frozenset({"cargo", "maturin", "rustc"})
_LATEX_EXECUTABLES = frozenset(
    {"latexmk", "lualatex", "pdflatex", "xelatex"}
)


class CampaignActivityError(RuntimeError):
    """Raised when a measurement or mutation owner remains active."""


@dataclass(frozen=True, slots=True)
class OpenFile:
    """One machine-readable ``lsof`` file-descriptor record."""

    pid: int
    command: str
    descriptor: str
    file_type: str
    path: Path


@dataclass(frozen=True, slots=True)
class CampaignActivity:
    """Blocking processes and coordination file descriptors."""

    processes: tuple[str, ...]
    coordination_files: tuple[OpenFile, ...]

    @property
    def idle(self) -> bool:
        return not self.processes and not self.coordination_files


def _arguments(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return tuple(command.split())


def _argument_basename(argument: str) -> str:
    return Path(argument).name.lower()


def is_snapshot_publisher_process(command: str) -> bool:
    """Return whether ``command`` is private snapshot render/compile work."""

    arguments = _arguments(command)
    lowered = tuple(argument.lower() for argument in arguments)
    if "publish-snapshot" in lowered:
        return True
    basenames = {_argument_basename(argument) for argument in arguments}
    return bool(
        basenames & _LATEX_EXECUTABLES
        and any(Path(argument).name == "pyAmpliCol.tex" for argument in arguments)
    )


def _is_campaign_process(
    command: str,
    *,
    entrypoints: frozenset[str],
) -> bool:
    if is_snapshot_publisher_process(command):
        return False
    arguments = _arguments(command)
    if not arguments:
        return False
    basenames = {_argument_basename(argument) for argument in arguments}
    if basenames & _INSTALL_ENTRYPOINTS:
        return True
    if _argument_basename(arguments[0]) in _NATIVE_BUILD_EXECUTABLES:
        return True
    if not any(argument in _MEASUREMENT_ACTIONS for argument in arguments):
        return False
    if not entrypoints:
        return True
    resolved_arguments = {
        os.path.realpath(argument)
        for argument in arguments
        if argument.endswith(".py")
    }
    return bool(resolved_arguments & entrypoints)


def blocking_process_lines(
    process_output: str,
    *,
    entrypoints: Iterable[Path] = (),
) -> tuple[str, ...]:
    """Filter ``ps -axo pid=,comm=,args=`` output to campaign blockers."""

    exact_entrypoints = frozenset(
        os.path.realpath(os.fspath(path)) for path in entrypoints
    )
    blocked: list[str] = []
    for raw_line in process_output.splitlines():
        line = raw_line.strip()
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        if _is_campaign_process(fields[2], entrypoints=exact_entrypoints):
            blocked.append(line)
    return tuple(blocked)


def parse_lsof_field_output(output: str) -> tuple[OpenFile, ...]:
    """Parse ``lsof -Fpcfnt`` output without depending on display columns."""

    records: list[OpenFile] = []
    pid: int | None = None
    command = ""
    descriptor = ""
    file_type = ""
    path: Path | None = None

    def flush_file() -> None:
        nonlocal descriptor, file_type, path
        if pid is not None and descriptor and path is not None:
            records.append(
                OpenFile(
                    pid=pid,
                    command=command,
                    descriptor=descriptor,
                    file_type=file_type,
                    path=path,
                )
            )
        descriptor = ""
        file_type = ""
        path = None

    for line in output.splitlines():
        if not line:
            continue
        field, value = line[0], line[1:]
        if field == "p":
            flush_file()
            try:
                pid = int(value)
            except ValueError:
                pid = None
            command = ""
        elif field == "c":
            command = value
        elif field == "f":
            flush_file()
            descriptor = value
            file_type = ""
        elif field == "t":
            file_type = value
        elif field == "n":
            path = Path(value)
    flush_file()
    return tuple(records)


def is_publisher_private_coordination_file(
    record: OpenFile,
    *,
    coordination_root: Path,
) -> bool:
    """Recognize only regular publisher-private coordination state."""

    publication_root = (
        coordination_root.expanduser().resolve(strict=False) / "publication"
    )
    path = record.path.expanduser().resolve(strict=False)
    return (
        record.file_type == "REG"
        and path != publication_root
        and publication_root in path.parents
    )


def blocking_coordination_files(
    lsof_output: str,
    *,
    coordination_root: Path,
) -> tuple[OpenFile, ...]:
    """Exclude publisher-private files but retain every lock/write owner."""

    return tuple(
        record
        for record in parse_lsof_field_output(lsof_output)
        if not is_publisher_private_coordination_file(
            record,
            coordination_root=coordination_root,
        )
    )


def campaign_activity(
    *,
    coordination_root: Path,
    entrypoints: Sequence[Path] = (),
) -> CampaignActivity:
    """Take one read-only process/coordination census."""

    processes = subprocess.run(
        ("ps", "-axo", "pid=,comm=,args="),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lsof = subprocess.run(
        (
            "lsof",
            "-Fpcfnt",
            "+D",
            os.fspath(coordination_root),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return CampaignActivity(
        processes=blocking_process_lines(
            processes,
            entrypoints=entrypoints,
        ),
        coordination_files=blocking_coordination_files(
            lsof.stdout,
            coordination_root=coordination_root,
        ),
    )


def require_campaign_idle(
    *,
    coordination_root: Path,
    entrypoints: Sequence[Path] = (),
) -> None:
    """Reject real campaign activity without treating the publisher as a worker."""

    activity = campaign_activity(
        coordination_root=coordination_root,
        entrypoints=entrypoints,
    )
    if activity.processes:
        raise CampaignActivityError(
            "measurement or install process remains active: "
            f"{list(activity.processes[:10])}"
        )
    if activity.coordination_files:
        held = [
            (
                f"{record.command}[{record.pid}] "
                f"{record.descriptor} {record.file_type} {record.path}"
            )
            for record in activity.coordination_files[:10]
        ]
        raise CampaignActivityError(
            f"campaign coordination files remain held: {held}"
        )


__all__ = [
    "CampaignActivity",
    "CampaignActivityError",
    "OpenFile",
    "blocking_coordination_files",
    "blocking_process_lines",
    "campaign_activity",
    "is_publisher_private_coordination_file",
    "is_snapshot_publisher_process",
    "parse_lsof_field_output",
    "require_campaign_idle",
]
