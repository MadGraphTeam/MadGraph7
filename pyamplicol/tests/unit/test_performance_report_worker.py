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
    _JsonlProgressSink,
    _source_identity,
    measure_cell,
    write_cell_result,
)
from tools.performance_report.worker_harness import worker_harness_identity


def test_atomic_worker_result_is_canonical_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "attempt" / "result.json"
    _atomic_json(path, {"status": "ok", "value": 1})

    assert json.loads(path.read_text(encoding="ascii")) == {
        "status": "ok",
        "value": 1,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_manual_source_identity_is_git_free_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.performance_report.worker.require_eligible_report_source",
        lambda _root: pytest.fail("manual source identity must not invoke Git"),
    )

    identity = _source_identity(tmp_path, "A" * 40, "b" * 40)

    assert identity.revision == "a" * 40
    assert identity.tree == "b" * 40
    assert identity.dirty_paths == ()
    with pytest.raises(ValueError, match="specified together"):
        _source_identity(tmp_path, "a" * 40, None)
    with pytest.raises(ValueError, match="revision must be"):
        _source_identity(tmp_path, "not-a-revision", "b" * 40)


def test_jsonl_progress_sink_captures_compact_typed_events(tmp_path: Path) -> None:
    from pyamplicol.reporting import ProgressEnd, ProgressStart, ProgressUpdate

    path = tmp_path / "progress.jsonl"
    sink = _JsonlProgressSink(path)
    sink.emit(ProgressStart("profile", "Profiling", 2, unit="samples"))
    sink.emit(ProgressUpdate("profile", 1, 2, "sampled"))
    sink.emit(ProgressEnd("profile", elapsed_seconds=0.25))

    events = [
        json.loads(line) for line in path.read_text(encoding="ascii").splitlines()
    ]
    assert [event["event"] for event in events] == ["start", "update", "end"]
    assert {event["task_id"] for event in events} == {"profile"}
    assert events[0]["total"] == 2
    assert events[1]["completed"] == 1
    assert events[2]["elapsed_seconds"] == 0.25


def test_every_catalog_cell_has_unique_worker_identity() -> None:
    cells = REPORT_CATALOG.measurement_cells()
    assert len({cell.cell_id for cell in cells}) == len(cells)


@pytest.mark.parametrize(
    ("cell_id", "reason"),
    (
        (
            "reference-amplicol-full-n6-dd-4q-lines-contracted",
            "original-amplicol-open-quark-line-limit",
        ),
        (
            "z-builtin-sm-n7-dd-z-jets-asm-o3-selected-flow",
            "user cap: native C\\+\\+/ASM generation is not attempted above n=6",
        ),
    ),
)
def test_worker_rejects_catalog_static_na_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
    reason: str,
) -> None:
    cell = REPORT_CATALOG.cell(cell_id)
    monkeypatch.setattr(
        "tools.performance_report.worker.require_eligible_report_source",
        lambda _root: pytest.fail(
            "source authentication must not run for catalog static N/A"
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"catalog static N/A cell.*{reason}",
    ):
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


def test_worker_success_carries_authenticated_split_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = worker_harness_identity(
        study_contract_sha256="1" * 64,
        policy_wrapper_revision="2" * 40,
        policy_wrapper_tree="3" * 40,
        policy_entrypoint_sha256="4" * 64,
        legacy_adapter_sha256="5" * 64,
        measured_source_revision="6" * 40,
        measured_source_tree="7" * 40,
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.measure_cell",
        lambda *_args, **_kwargs: {"status": "ok", "provenance": {}},
    )

    result = write_cell_result(
        "cell",
        tmp_path / "result.json",
        worker_harness=harness,
    )

    assert result["provenance"]["worker_harness"] == harness
    assert (
        json.loads((tmp_path / "result.json").read_text(encoding="ascii"))[
            "provenance"
        ]["worker_harness"]
        == harness
    )


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


def test_legacy_worker_threads_generation_phase_reporter_to_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.cell("reference-amplicol-lc-n1-dd-z-jets-selected-flow")
    reporter = object()
    observed: list[object] = []

    class SourceIdentity:
        def provenance(self) -> dict[str, object]:
            return {}

    class Adapter:
        def measure(self, *_args: object, **kwargs: object) -> dict[str, object]:
            observed.append(kwargs.get("phase_reporter"))
            return {
                "status": "ok",
                "validation": {"status": "ok"},
                "provenance": {},
            }

    identity = SourceIdentity()
    monkeypatch.setattr(
        "tools.performance_report.worker.require_eligible_report_source",
        lambda _root: identity,
    )
    monkeypatch.setattr(
        "tools.performance_report.legacy.LegacyMeasurementAdapter",
        Adapter,
    )

    result = measure_cell(
        cell.cell_id,
        repo_root=tmp_path,
        attempt_root=tmp_path / "attempt",
        target_runtime_seconds=1.0,
        batch_size=1,
        worker_cores=1,
        phase_reporter=reporter,  # type: ignore[arg-type]
        legacy_repository=tmp_path / "legacy",
    )

    assert result["status"] == "ok"
    assert observed == [reporter]
