#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run Cargo against the clean release or candidate source overlay."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("candidate", "release"), required=True)
    parser.add_argument("cargo_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.cargo_arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    if not arguments:
        parser.error("a Cargo subcommand is required after '--'")

    with backend._overlay(args.mode) as (overlay, target):
        updates = {
            "CARGO_HOME": str(target.parent / "cargo-home"),
            "CARGO_TARGET_DIR": str(target),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if sys.platform == "darwin":
            updates["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
        environment = backend._clean_environment(updates)
        completed = subprocess.run(
            ["cargo", *arguments],
            cwd=overlay,
            env=environment,
            check=False,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
