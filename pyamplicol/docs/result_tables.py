#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compatibility entry point for the performance-report service."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE_DIRS = tuple(
    package_dir
    for entry in sys.path
    if (package_dir := Path(entry) / "pyamplicol").is_dir()
    and any(package_dir.glob("_rusticol*"))
)
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import pyamplicol  # noqa: E402

for package_dir in NATIVE_PACKAGE_DIRS:
    if str(package_dir) not in pyamplicol.__path__:
        pyamplicol.__path__.append(str(package_dir))

from tools.performance_report.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
