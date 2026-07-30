#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Manually steer the fresh MacBook M3 performance-report campaign."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repository_root(entrypoint: Path) -> Path:
    for candidate in entrypoint.parents:
        if (candidate / "tools/performance_report").is_dir() and (
            candidate / "src/pyamplicol"
        ).is_dir():
            return candidate
    raise RuntimeError("campaign entrypoint is not inside a pyAmpliCol checkout")


def _reexecute_with_repository_python(repo_root: Path) -> None:
    expected = repo_root / ".venv/bin/python"
    if not expected.is_file():
        raise RuntimeError(
            f"repository Python is unavailable at {expected}; run `just dev-install`"
        )
    try:
        current_environment = Path(sys.prefix).resolve(strict=True)
        expected_environment = (repo_root / ".venv").resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve repository Python: {error}") from error
    if current_environment == expected_environment:
        return
    environment = dict(os.environ)
    # macOS framework Python leaves this launcher hint behind when a shebang
    # process execs a virtual-environment interpreter.  Keeping it can make
    # the new interpreter ignore the repository venv's site-packages.
    environment.pop("__PYVENV_LAUNCHER__", None)
    environment.pop("PYTHONHOME", None)
    os.execve(
        os.fspath(expected),
        (os.fspath(expected), os.fspath(Path(__file__).resolve()), *sys.argv[1:]),
        environment,
    )


def main() -> int:
    entrypoint = Path(__file__).resolve()
    repo_root = _repository_root(entrypoint)
    _reexecute_with_repository_python(repo_root)
    sys.path.insert(0, os.fspath(repo_root))
    sys.path.insert(0, os.fspath(repo_root / "src"))
    from tools.performance_report.manual_campaign import main as campaign_main

    return campaign_main(sys.argv[1:], repo_root=repo_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted; workers were asked to stop cleanly.\n")
        raise SystemExit(130) from None
