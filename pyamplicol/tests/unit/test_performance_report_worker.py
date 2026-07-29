# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.phase_state import (
    WorkerPhaseChannel,
    WorkerPhaseReporter,
    read_worker_phase_state,
)
from tools.performance_report.worker import (
    _atomic_json,
    measure_cell,
    write_cell_result,
)


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


def test_worker_rejects_catalog_static_na_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.cell(
        "reference-amplicol-full-n6-dd-4q-lines-contracted"
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.require_eligible_report_source",
        lambda _root: pytest.fail(
            "source authentication must not run for catalog static N/A"
        ),
    )

    with pytest.raises(ValueError, match="catalog static N/A cell"):
        measure_cell(
            cell.cell_id,
            repo_root=tmp_path,
            attempt_root=tmp_path / "attempt",
            target_runtime_seconds=1.0,
            batch_size=1,
            worker_cores=1,
        )


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


def test_worker_constructs_and_threads_parent_phase_reporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    observed: list[str] = []

    def measure(
        _cell_id: str,
        *,
        phase_reporter: WorkerPhaseReporter | None,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert phase_reporter is not None
        state = read_worker_phase_state(
            channel,
            expected_pid=phase_reporter.worker_pid,
        )
        observed.append(state.phase)
        return {"status": "ok", "provenance": {}}

    monkeypatch.setattr("tools.performance_report.worker.measure_cell", measure)
    result = write_cell_result(
        "cell",
        tmp_path / "result.json",
        phase_state_path=channel.path,
        phase_state_run_id=channel.run_id,
        phase_state_authentication_key=channel.authentication_key,
    )

    assert result["status"] == "ok"
    assert observed == ["pre-generation"]
