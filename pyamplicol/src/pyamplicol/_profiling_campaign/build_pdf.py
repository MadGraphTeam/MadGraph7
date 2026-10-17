# SPDX-License-Identifier: 0BSD
"""Compile and verify an exported pyAmpliCol report using only its TeX folder."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from shutil import which


class StandaloneBuildError(RuntimeError):
    """The standalone publication could not be compiled or verified."""


_UNRESOLVED_PATTERNS = (
    re.compile(r"LaTeX Warning: Citation .* undefined"),
    re.compile(r"LaTeX Warning: Reference .* undefined"),
    re.compile(r"LaTeX Warning: There were undefined references"),
    re.compile(r"Package rerunfilecheck Warning:"),
)


def validate_latex_log(log: str) -> None:
    """Reject layout overflow and unresolved cross-reference diagnostics."""

    if "Overfull \\hbox" in log or "Overfull \\vbox" in log:
        raise StandaloneBuildError("LaTeX output contains an overfull box")
    for pattern in _UNRESOLVED_PATTERNS:
        if pattern.search(log):
            raise StandaloneBuildError(
                f"LaTeX output is unresolved: {pattern.pattern}"
            )


def compile_report(
    report_dir: Path,
    *,
    engine: str = "pdflatex",
    passes: int = 3,
) -> Path:
    """Compile ``pyAmpliCol.tex`` and reject unresolved or overflowing output."""

    root = report_dir.expanduser().resolve(strict=True)
    source = root / "pyAmpliCol.tex"
    if not source.is_file():
        raise StandaloneBuildError(f"missing report source: {source}")
    if passes < 2:
        raise StandaloneBuildError("at least two LaTeX passes are required")
    executable = which(engine)
    if executable is None:
        raise StandaloneBuildError(f"LaTeX engine is not available: {engine}")

    command = (
        executable,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        source.name,
    )
    for _ in range(passes):
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stdout + completed.stderr)[-4000:]
            raise StandaloneBuildError(
                f"LaTeX compilation failed with exit {completed.returncode}:\n"
                f"{detail}"
            )

    log_path = root / "pyAmpliCol.log"
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise StandaloneBuildError(f"cannot read LaTeX log: {error}") from error
    validate_latex_log(log)

    output = root / "pyAmpliCol.pdf"
    if not output.is_file() or output.stat().st_size == 0:
        raise StandaloneBuildError(f"LaTeX did not produce {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and verify this standalone pyAmpliCol report",
    )
    parser.add_argument("--engine", default="pdflatex")
    parser.add_argument("--passes", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = compile_report(
        Path(__file__).resolve().parent,
        engine=arguments.engine,
        passes=arguments.passes,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
