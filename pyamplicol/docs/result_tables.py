#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compatibility entry point for the performance-report service."""

from __future__ import annotations

import sys
from pathlib import Path


def _repository_root(entrypoint: Path) -> Path:
    for candidate in entrypoint.parents:
        if (
            (candidate / "tools/performance_report").is_dir()
            and (candidate / "src/pyamplicol").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "result_tables.py must run from a pyAmpliCol source checkout"
    )


def _embedded_profile(entrypoint: Path, repo_root: Path) -> str | None:
    profile_parent = repo_root / "docs/performance_reports"
    try:
        relative = entrypoint.parent.relative_to(profile_parent)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) == 1 else None


ENTRYPOINT = Path(__file__).resolve()
REPOSITORY_ROOT = _repository_root(ENTRYPOINT)
EMBEDDED_PROFILE = _embedded_profile(ENTRYPOINT, REPOSITORY_ROOT)
ARGUMENTS = list(sys.argv[1:])
if (
    EMBEDDED_PROFILE is not None
    and "--report-profile" not in ARGUMENTS
    and "--docs-dir" not in ARGUMENTS
):
    ARGUMENTS[:0] = ("--report-profile", EMBEDDED_PROFILE)
NATIVE_PACKAGE_DIRS = tuple(
    package_dir
    for entry in sys.path
    if (package_dir := Path(entry) / "pyamplicol").is_dir()
    and any(package_dir.glob("_rusticol*"))
)
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

if any(command in ARGUMENTS for command in {"_prepare", "_worker"}):
    import pyamplicol

    for package_dir in NATIVE_PACKAGE_DIRS:
        if str(package_dir) not in pyamplicol.__path__:
            pyamplicol.__path__.append(str(package_dir))

from tools.performance_report.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(ARGUMENTS))
