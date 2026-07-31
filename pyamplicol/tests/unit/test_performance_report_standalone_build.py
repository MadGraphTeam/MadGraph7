# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from tools.performance_report.standalone_build import (
    StandaloneBuildError,
    compile_report,
    validate_latex_log,
)


def _fake_engine(tmp_path: Path, log: str) -> Path:
    engine = tmp_path / "fake-pdflatex"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('pyAmpliCol.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        f"Path('pyAmpliCol.log').write_text({log!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    return engine


def test_standalone_builder_compiles_relocated_report(tmp_path: Path) -> None:
    report = tmp_path / "relocated report"
    report.mkdir()
    (report / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}ok\\end{document}\n",
        encoding="ascii",
    )

    output = compile_report(
        report,
        engine=str(_fake_engine(tmp_path, "clean publication log\n")),
        passes=2,
    )

    assert output == report / "pyAmpliCol.pdf"
    assert output.is_file()


def test_standalone_builder_rejects_overfull_output(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    (report / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}bad\\end{document}\n",
        encoding="ascii",
    )

    with pytest.raises(StandaloneBuildError, match="overfull"):
        compile_report(
            report,
            engine=str(
                _fake_engine(
                    tmp_path,
                    "Overfull \\hbox (1.0pt too wide)\n",
                )
            ),
            passes=2,
        )


def test_latex_log_can_explicitly_allow_overfull_boxes() -> None:
    validate_latex_log(
        "Overfull \\hbox (1.0pt too wide)\n"
        "Overfull \\vbox (2.0pt too high)\n",
        allow_overfull_boxes=True,
    )


def test_latex_log_still_rejects_unresolved_references_when_overfull_allowed() -> None:
    with pytest.raises(StandaloneBuildError, match="unresolved"):
        validate_latex_log(
            "Overfull \\hbox (1.0pt too wide)\n"
            "LaTeX Warning: There were undefined references.\n",
            allow_overfull_boxes=True,
        )
