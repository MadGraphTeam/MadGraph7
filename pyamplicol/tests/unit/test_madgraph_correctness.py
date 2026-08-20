# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from tools.developer.madgraph_correctness import (
    CommandResult,
    MadGraphAdapterError,
    madgraph_command_card,
    reject_failed_generation,
)


def _result(tmp_path: Path, *, stderr: str = "") -> CommandResult:
    return CommandResult(
        args=("mg5_aMC",),
        cwd=tmp_path,
        elapsed_seconds=0.0,
        returncode=0,
        stdout="",
        stderr=stderr,
    )


def test_correctness_command_card_binds_exact_ufo_and_heft_order() -> None:
    assert madgraph_command_card(
        "g g > H g g",
        model_import="/tmp/prepared-heft-ufo",
        coupling_orders={"HIG": 1},
    ) == (
        "import model /tmp/prepared-heft-ufo\n"
        "generate g g > H g g HIG=1\n"
        "output standalone standalone -f\n"
        "launch -f\n"
    )


def test_correctness_generation_guard_rejects_import_fallback(tmp_path: Path) -> None:
    cards = tmp_path / "standalone/Cards"
    cards.mkdir(parents=True)
    process_card = cards / "proc_card_mg5.dat"
    expected = "/tmp/prepared-heft-ufo"
    process_card.write_text(f"import model {expected}\n", encoding="utf-8")

    reject_failed_generation(
        _result(tmp_path),
        tmp_path / "standalone",
        expected_model_import=expected,
    )

    process_card.write_text("import model sm\n", encoding="utf-8")
    with pytest.raises(MadGraphAdapterError, match="bound exclusively"):
        reject_failed_generation(
            _result(tmp_path),
            tmp_path / "standalone",
            expected_model_import=expected,
        )

    process_card.write_text(f"import model {expected}\n", encoding="utf-8")
    with pytest.raises(MadGraphAdapterError, match="possible model fallback"):
        reject_failed_generation(
            _result(tmp_path, stderr="UFOImportError: rejected"),
            tmp_path / "standalone",
            expected_model_import=expected,
        )
