#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Manually steer the fresh MacBook M3 performance-report campaign."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


def _repository_root(entrypoint: Path) -> Path:
    for candidate in entrypoint.parents:
        if (candidate / "tools/performance_report").is_dir() and (
            candidate / "src/pyamplicol"
        ).is_dir():
            return candidate
    raise RuntimeError("campaign entrypoint is not inside a pyAmpliCol checkout")


def _embedded_profile(entrypoint: Path, repo_root: Path) -> str:
    """Derive the campaign identity from its directory under report profiles."""

    profile_parent = repo_root / "docs/performance_reports"
    try:
        relative = entrypoint.parent.relative_to(profile_parent)
    except ValueError as error:
        raise RuntimeError(
            "campaign entrypoint must be directly inside "
            "docs/performance_reports/<campaign>"
        ) from error
    if len(relative.parts) != 1:
        raise RuntimeError(
            "campaign entrypoint must be directly inside "
            "docs/performance_reports/<campaign>"
        )
    return relative.parts[0]


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
    profile = entrypoint.parent.name
    try:
        repo_root = _repository_root(entrypoint)
        source_profile = _embedded_profile(entrypoint, repo_root)
    except RuntimeError:
        installed = import_module(
            "pyamplicol._performance_report.manual_campaign"
        )
        return installed.main(
            sys.argv[1:],
            repo_root=entrypoint.parent,
            profile=profile,
            docs_dir=entrypoint.parent,
            installed=True,
        )

    _reexecute_with_repository_python(repo_root)
    sys.path.insert(0, os.fspath(repo_root))
    sys.path.insert(0, os.fspath(repo_root / "src"))
    from tools.performance_report.manual_campaign import main as campaign_main

    return campaign_main(sys.argv[1:], repo_root=repo_root, profile=source_profile)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted; workers were asked to stop cleanly.\n")
        raise SystemExit(130) from None
