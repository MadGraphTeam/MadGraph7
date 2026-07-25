# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.worker import _atomic_json, write_cell_result


def test_atomic_worker_result_is_canonical_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "attempt" / "result.json"
    _atomic_json(path, {"status": "ok", "value": 1})

    assert json.loads(path.read_text(encoding="ascii")) == {
        "status": "ok",
        "value": 1,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_every_catalog_cell_has_unique_worker_identity() -> None:
    cells = REPORT_CATALOG.measurement_cells()
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_worker_failure_is_structured_and_traceback_stays_in_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setattr("tools.performance_report.worker.measure_cell", fail)
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "worker.log"
    result = write_cell_result(
        "cell",
        result_path,
        log_path=log_path,
    )

    assert result["status"] == "error"
    assert result["failure"]["message"] == "deliberate worker failure"
    assert json.loads(result_path.read_text(encoding="ascii"))["status"] == "error"
    assert "Traceback" in log_path.read_text(encoding="utf-8")
