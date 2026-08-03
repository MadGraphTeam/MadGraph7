# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

import tools.performance_report.worker as worker_module
from tools.performance_report.artifacts import DiskFullError
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import load_measurement
from tools.performance_report.phase_state import (
    WorkerPhaseChannel,
    WorkerPhaseReporter,
    read_worker_phase_state,
)
from tools.performance_report.worker import (
    _atomic_json,
    _JsonlProgressSink,
    _portable_current_paths,
    _selector_provider_measurement,
    _source_identity,
    _worker_legacy_workspace,
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


def test_worker_materializes_portable_peer_from_canonical_current_path(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "manual"
    state = campaign / "campaign_artifacts"
    (state / "coordination").mkdir(parents=True)
    worker_attempt = (
        state
        / "cells/worker/attempts/11111111-1111-4111-8111-111111111111"
    )
    peer_attempt = state / "cells/peer/attempts/22222222-2222-4222-8222-222222222222"
    worker_attempt.mkdir(parents=True)
    peer_artifact = peer_attempt / "artifact"
    peer_artifact.mkdir(parents=True)
    result_path = peer_attempt / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "artifact": {
                    "path": "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/cells/peer/attempts/"
                    "22222222-2222-4222-8222-222222222222/artifact"
                },
            }
        ),
        encoding="utf-8",
    )

    paths = _portable_current_paths(
        repo_root=tmp_path / "source",
        attempt_root=worker_attempt,
    )

    assert paths is not None
    loaded = load_measurement(result_path, publication_paths=paths)
    assert loaded["artifact"]["path"] == str(peer_artifact)


def test_atomic_worker_result_normalizes_enospc_with_target_and_free_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "attempt" / "worker-result.json"

    def no_space(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(worker_module.os, "replace", no_space)
    with pytest.raises(
        DiskFullError,
        match=(
            r"disk full while writing .*worker-result\.json; "
            r"[0-9]+ bytes available"
        ),
    ):
        worker_module._atomic_json(path, {"status": "error"})

    assert not path.exists()
    assert not tuple(path.parent.glob(".worker-result.json.*.tmp"))


def test_worker_source_identity_uses_controller_values_without_git(
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


def test_no_legacy_baseline_lc_all_flow_gets_selected_flow_selector_peer() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n7-dd-4q-lines-all-flow"
    )
    peer_id = "matrix-recurrence-builtin-sm-lc-n7-dd-4q-lines-selected-flow"
    provider = {
        "status": "ok",
        "selector_contract": {
            "selected_color_flow_ids": ["flow:1"],
            "selected_color_words": [[1]],
            "all_flow_helicity_ids": ["h:-1"],
            "all_flow_source_helicities": [[1, -1]],
            "point_digest": "a" * 64,
        },
    }

    assert REPORT_CATALOG.validation_baseline_cell(cell) is None
    assert (
        _selector_provider_measurement(
            cell.cell_id,
            {peer_id: provider},
            catalog=REPORT_CATALOG,
        )
        is provider
    )


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


def test_legacy_workspace_copy_lives_only_inside_supervised_worker_scope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "original-amplicol"
    source.mkdir()
    (source / "source.f03").write_text("program fixture\n", encoding="ascii")
    destination = tmp_path / "worker" / "legacy-workspace"

    with _worker_legacy_workspace(
        repository=None,
        source_repository=source,
        workspace=destination,
        copy_source=True,
    ) as prepared:
        assert prepared == destination
        assert (destination / "source.f03").read_text(encoding="ascii") == (
            "program fixture\n"
        )

    assert not destination.exists()


def test_worker_threads_effective_stage_budgets_and_tracks_extended_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    observed: dict[str, object] = {}

    def measure(
        _cell_id: str,
        *,
        phase_reporter: WorkerPhaseReporter | None,
        **kwargs: object,
    ) -> dict[str, object]:
        assert phase_reporter is not None
        observed.update(kwargs)
        with phase_reporter.generation():
            pass
        phase_reporter.profiling_started()
        observed["phase"] = read_worker_phase_state(
            channel,
            expected_pid=phase_reporter.worker_pid,
        ).phase
        return {"status": "ok", "provenance": {}}

    monkeypatch.setattr("tools.performance_report.worker.measure_cell", measure)
    result = write_cell_result(
        "cell",
        tmp_path / "result.json",
        phase_state_path=channel.path,
        phase_state_run_id=channel.run_id,
        phase_state_authentication_key=channel.authentication_key,
        worker_wall_limit_seconds=12.0,
        profiling_time_limit_seconds=5.0,
        validation_time_limit_seconds=3.0,
    )

    assert result["status"] == "ok"
    assert observed == {
        "worker_wall_limit_seconds": 12.0,
        "profiling_time_limit_seconds": 5.0,
        "validation_time_limit_seconds": 3.0,
        "phase": "profiling",
    }


@pytest.mark.parametrize(
    "field",
    (
        "worker_wall_limit_seconds",
        "profiling_time_limit_seconds",
        "validation_time_limit_seconds",
    ),
)
def test_worker_rejects_invalid_effective_stage_budget(
    tmp_path: Path,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        measure_cell(
            "does-not-need-to-exist",
            repo_root=tmp_path,
            attempt_root=tmp_path / "attempt",
            target_runtime_seconds=1.0,
            batch_size=1,
            worker_cores=1,
            **{field: float("inf")},
        )


def test_legacy_worker_threads_generation_phase_reporter_to_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.cell("reference-amplicol-lc-n1-dd-z-jets-selected-flow")
    events: list[str] = []

    class Reporter:
        def complete(self) -> None:
            events.append("complete")

    reporter = Reporter()
    observed: list[tuple[object, object]] = []

    class SourceIdentity:
        def provenance(self) -> dict[str, object]:
            return {}

    class Adapter:
        def measure(self, *_args: object, **kwargs: object) -> dict[str, object]:
            events.append("adapter")
            observed.append(
                (
                    kwargs.get("phase_reporter"),
                    kwargs.get("selector_provider"),
                )
            )
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
    monkeypatch.setattr(
        "tools.performance_report.worker.attach_direct_agreements",
        lambda *_args, **_kwargs: events.append("agreements"),
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
    assert observed == [(reporter, None)]
    assert events == ["adapter", "agreements", "complete"]


def test_legacy_all_flow_worker_passes_selected_flow_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.cell("reference-amplicol-lc-n1-dd-z-jets-all-flow")
    provider = {"status": "ok", "selector_contract": {"fixture": True}}
    observed: list[object] = []

    class SourceIdentity:
        def provenance(self) -> dict[str, object]:
            return {}

    class Adapter:
        def measure(self, *_args: object, **kwargs: object) -> dict[str, object]:
            observed.append(kwargs.get("selector_provider"))
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
        "tools.performance_report.worker._selector_provider_measurement",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "tools.performance_report.legacy.LegacyMeasurementAdapter",
        Adapter,
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.attach_direct_agreements",
        lambda *_args, **_kwargs: None,
    )

    result = measure_cell(
        cell.cell_id,
        repo_root=tmp_path,
        attempt_root=tmp_path / "attempt",
        target_runtime_seconds=1.0,
        batch_size=1,
        worker_cores=1,
        legacy_repository=tmp_path / "legacy",
    )

    assert result["status"] == "ok"
    assert observed == [provider]


def test_pyamplicol_worker_passes_all_validation_peers_to_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted"
    )
    baseline = {"status": "ok", "matrix_element": 1.0}
    peer = {
        "status": "ok",
        "matrix_element": 1.0,
        "validation": {
            "legacy_imode2_diagnostic": {
                "authoritative_value": 1.0,
                "imode2_value": 1.0,
            }
        },
    }
    baseline_path = tmp_path / "baseline.json"
    peer_path = tmp_path / "peer.json"
    baseline_path.write_text(json.dumps(baseline), encoding="ascii")
    peer_path.write_text(json.dumps(peer), encoding="ascii")
    observed: list[object] = []

    class SourceIdentity:
        def provenance(self) -> dict[str, object]:
            return {}

    def measure(*_args: object, **kwargs: object) -> dict[str, object]:
        observed.append(kwargs.get("validation_peers"))
        return {
            "status": "validation_failed",
            "validation": {"status": "validation_failed"},
            "provenance": {},
        }

    identity = SourceIdentity()
    monkeypatch.setattr(
        "tools.performance_report.worker.require_eligible_report_source",
        lambda _root: identity,
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.measure_pyamplicol_cell",
        measure,
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.attach_direct_agreements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.worker.attach_validation_failure_precision_diagnostic",
        lambda *_args, **_kwargs: None,
    )

    result = measure_cell(
        cell.cell_id,
        repo_root=tmp_path,
        attempt_root=tmp_path / "attempt",
        target_runtime_seconds=1.0,
        batch_size=1,
        worker_cores=1,
        baseline_json=baseline_path,
        peer_json=(("reference-amplicol-full-n4-dd-tt-jets-contracted", peer_path),),
    )

    assert result["status"] == "validation_failed"
    assert observed == [
        {"reference-amplicol-full-n4-dd-tt-jets-contracted": peer}
    ]
