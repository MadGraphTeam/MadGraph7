# SPDX-License-Identifier: 0BSD
"""Explain why a checkout cannot record a publishable release revision.

The build backend and ``dependencies/install_dependencies.py`` (which runs on a
bare interpreter before the build starts) share this stdlib-only module, so
both report the same message.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

ALLOW_DIRTY_SETTING = "pyamplicol.allow-dirty"
# The first offending paths are listed; `git status` shows the rest.
_LISTED_PATHS = 10


def _git_environment() -> dict[str, str]:
    return {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }


def _git(
    root: Path, environment: Mapping[str, str], *arguments: str
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=dict(environment),
        check=False,
        capture_output=True,
    )


def _first_line(output: bytes) -> str:
    lines = os.fsdecode(output).strip().splitlines()
    return lines[0] if lines else "no output"


def _path_count(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _status_report(status: bytes) -> str:
    records = [record for record in status.split(b"\0") if record]
    untracked = sum(record.startswith(b"??") for record in records)
    changed = len(records) - untracked
    kinds = [
        text
        for count, text in (
            (changed, _path_count(changed, "uncommitted change")),
            (untracked, _path_count(untracked, "untracked file")),
        )
        if count
    ]
    lines = [f"the checkout has {' and '.join(kinds)}:"]
    lines.extend(
        f"    {os.fsdecode(record[:2])} {os.fsdecode(record[3:])}"
        for record in records[:_LISTED_PATHS]
    )
    if len(records) > _LISTED_PATHS:
        lines.append(
            f"    ... and {len(records) - _LISTED_PATHS} more (see `git status`)"
        )
    return "\n".join(lines)


def dirty_checkout_reason(
    root: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, bool] | None:
    """Return why ``root`` has no clean release revision, or ``None``.

    This repeats the release checks of the backend's ``_clean_source_revision``
    (the Git top level is ``root``, ``git status --porcelain
    --untracked-files=all`` is empty, ``HEAD`` resolves). The flag is true when
    committing or stashing would fix the checkout.
    """

    if not (root / ".git").exists():
        return None
    environment = _git_environment() if environment is None else environment
    try:
        top_level = _git(root, environment, "rev-parse", "--show-toplevel")
        if top_level.returncode != 0:
            return (
                "Git cannot inspect the checkout: " + _first_line(top_level.stderr),
                False,
            )
        top = Path(os.fsdecode(top_level.stdout.strip()))
        if top.resolve() != root.resolve():
            return (
                f"the Git top level is {top}, not the project root "
                f"{root.resolve()} (pyAmpliCol is nested in another repository)",
                False,
            )
        status = _git(
            root,
            environment,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        )
        if status.returncode != 0:
            return "`git status` failed: " + _first_line(status.stderr), False
        if status.stdout:
            return _status_report(status.stdout), True
        head = _git(root, environment, "rev-parse", "HEAD")
        revision = os.fsdecode(head.stdout).strip()
        if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            return "the checkout has no commit to record (`git rev-parse HEAD`)", True
    except OSError as error:
        return f"Git could not be run: {error}", False
    return None


def dirty_checkout_message(reason: tuple[str, bool] | None) -> str:
    """Format the user-facing refusal for a checkout without a clean revision."""

    text, fixable = reason or ("its Git state could not be confirmed as clean", True)
    remedy = (
        "commit or stash the changes (`git stash --include-untracked`) and rerun"
        if fixable
        else "build from a standalone clone of pyAmpliCol"
    )
    return (
        "publishable release artifacts require a clean Git source revision, "
        f"but {text}\n"
        "Either:\n"
        f"  - {remedy};\n"
        "  - or build a non-publishable development wheel from this checkout: "
        "`just dev-install --allow-dirty`\n"
        "    (`python -m build --wheel --config-setting "
        f"{ALLOW_DIRTY_SETTING}=true`);\n"
        "  - or install an existing wheel: "
        "`just dev-install --wheel-directory PATH`;\n"
        "  - or install only the dependencies: "
        "`just dev-install --dependencies-only`.\n"
        "See docs/user/installation.md."
    )
