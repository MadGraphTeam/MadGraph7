#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build and verify a wheel from one clean, externally unpacked sdist."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from _artifacts import audit_sdist, audit_wheel, compare_wheels
from _common import (
    CANDIDATE_ARTIFACTS,
    DIST,
    ReleaseError,
    build_mode,
    check_dependency_gate,
    clean_environment,
    exactly_one,
    external_temporary_directory,
    run,
    safe_extract_sdist,
)


def build_wheel_from_sdist(
    sdist: Path,
    source_wheel: Path,
    output_directory: Path,
    *,
    mode: str,
    python: Path,
    replace: bool = False,
) -> Path:
    """Build from an untouched sdist and retain it only after parity succeeds."""

    sdist = sdist.resolve()
    source_wheel = source_wheel.resolve()
    output_directory = output_directory.resolve()
    audit_sdist(sdist, mode=mode)
    audit_wheel(source_wheel, mode=mode)
    with external_temporary_directory("pyamplicol-from-sdist-") as temporary:
        source = safe_extract_sdist(sdist, temporary / "unpacked")
        wheel_directory = temporary / "wheel"
        wheel_directory.mkdir()
        run(
            [python, "-m", "build", "--wheel", "--outdir", wheel_directory],
            cwd=source,
            env=clean_environment(mode=mode),
        )
        rebuilt = exactly_one(
            list(wheel_directory.glob("pyamplicol-*.whl")),
            "wheel rebuilt from sdist",
        )
        audit_wheel(rebuilt, mode=mode)
        if rebuilt.name != source_wheel.name:
            raise ReleaseError(
                "source and sdist wheels have different filenames: "
                f"{source_wheel.name} != {rebuilt.name}"
            )
        compare_wheels(source_wheel, rebuilt)

        output_directory.mkdir(parents=True, exist_ok=True)
        target = output_directory / rebuilt.name
        if target.exists() and not replace:
            raise ReleaseError(f"refusing to overwrite existing wheel: {target}")
        staged = output_directory / f".{rebuilt.name}.from-sdist"
        if staged.exists():
            raise ReleaseError(f"stale staged wheel must be removed first: {staged}")
        shutil.copy2(rebuilt, staged)
        os.replace(staged, target)
    print(f"Verified source/sdist wheel parity: {target}")
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--source-wheel", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination for the retained sdist-built wheel; defaults to artifact dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    check_dependency_gate(mode, online=mode == "release")
    artifact_directory = CANDIDATE_ARTIFACTS if mode == "candidate" else DIST
    sdist = args.sdist or exactly_one(
        list(artifact_directory.glob("pyamplicol-*.tar.gz")), "pyamplicol sdist"
    )
    source_wheel = args.source_wheel or exactly_one(
        list(artifact_directory.glob("pyamplicol-*.whl")),
        "direct-source pyamplicol wheel",
    )
    output_directory = args.output_dir or artifact_directory
    build_wheel_from_sdist(
        sdist,
        source_wheel,
        output_directory,
        mode=mode,
        python=args.python,
        replace=args.output_dir is None,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
