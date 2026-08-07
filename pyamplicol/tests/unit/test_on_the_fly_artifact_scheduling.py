# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.performance_report.artifacts import ArtifactStore, CurrentRecord
from tools.performance_report.cache import empty_measurement
from tools.performance_report.campaign_policy import PolicyMeasurementState
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.resources import ResourceUsage, SupervisedResult
from tools.performance_report.scheduler import (
    CampaignScheduler,
    CampaignSettings,
    PlannedCell,
    _artifact_consumer_cell_ids,
    plan_campaign,
)
from tools.performance_report.service import ReportPaths, ReportService

_REVISION = "current-revision"


def _otf_pair() -> tuple[CellSpec, CellSpec]:
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_on_the_fly_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
    )
    selected = next(
        cell for cell in cells if cell.workload is Workload.SELECTED_FLOW
    )
    all_flow = next(cell for cell in cells if cell.workload is Workload.ALL_FLOW)
    return selected, all_flow


def _measurement(cell: CellSpec) -> dict[str, object]:
    measurement = empty_measurement()
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {},
            "selector_contract": None,
            "validation": {"status": ResultStatus.OK.value},
            "resources": {},
            "provenance": {
                "report_source_revision": _REVISION,
                "test_cell_id": cell.cell_id,
            },
        }
    )
    return measurement


def _publish_current(
    store: ArtifactStore,
    cell: CellSpec,
) -> CurrentRecord:
    return store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
        _measurement(cell)
    )


def _service(tmp_path: Path) -> ReportService:
    repo_root = tmp_path / "repo"
    (repo_root / "src/pyamplicol/_profiling_campaign").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=repo_root, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-tests@example.invalid"),
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Tests"),
        cwd=repo_root,
        check=True,
    )
    (repo_root / "README.md").write_text("# report fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "README.md"), cwd=repo_root, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "Initialize fixture"),
        cwd=repo_root,
        check=True,
    )
    return ReportService(
        ReportPaths.from_repo(
            repo_root,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "locks",
        )
    )


def test_otf_artifact_reuse_is_directional_and_preserves_owner_consumers() -> None:
    selected, all_flow = _otf_pair()

    assert REPORT_CATALOG.equivalent_cells(selected) == ()
    assert REPORT_CATALOG.equivalent_cells(all_flow) == (selected,)
    assert _artifact_consumer_cell_ids(
        selected,
        catalog=REPORT_CATALOG,
    ) == tuple(sorted((selected.cell_id, all_flow.cell_id)))


def test_regenerate_all_flow_forces_its_fresh_selected_owner_into_the_plan(
    tmp_path: Path,
) -> None:
    selected, all_flow = _otf_pair()
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    current = _publish_current(store, selected)

    def resolve(
        _cell: CellSpec,
    ) -> tuple[CurrentRecord, PolicyMeasurementState]:
        return current, PolicyMeasurementState.SUCCESS

    planned = plan_campaign(
        (all_flow,),
        store=store,
        settings=CampaignSettings(artifact_policy=ArtifactPolicy.REGENERATE),
        current_resolver=resolve,
    )
    otf = tuple(
        item
        for item in planned
        if item.cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    )

    assert tuple(item.cell for item in otf) == (selected, all_flow)
    assert otf[0].dependency is True
    assert selected.cell_id in otf[1].prerequisite_cell_ids
    assert otf[0].rank < otf[1].rank


@pytest.mark.parametrize("force_refresh", [False, True])
def test_regenerate_and_force_refresh_generate_selected_once_then_reuse_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_refresh: bool,
) -> None:
    selected, all_flow = _otf_pair()
    service = _service(tmp_path)
    old_selected = _publish_current(service.store, selected) if force_refresh else None
    old_all_flow = _publish_current(service.store, all_flow) if force_refresh else None
    commands: list[list[str]] = []

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        captured = list(command)
        commands.append(captured)
        cell = REPORT_CATALOG.cell(captured[captured.index("--cell-id") + 1])
        result_path = Path(captured[captured.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(_measurement(cell), sort_keys=True) + "\n",
            encoding="ascii",
        )
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler._symbolica_generation_lock_path",
        lambda *_args, **_kwargs: tmp_path / "generation.lock",
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=ArtifactPolicy.REGENERATE,
            rerun=force_refresh,
            target_runtime_seconds=1.0,
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    scheduler.source_revision = _REVISION

    selected_outcome = scheduler._run_cell(PlannedCell(selected, False, None, 0))
    selected_current = service.store.load_current(selected.cell_id)
    assert selected_current is not None
    all_flow_outcome = scheduler._run_cell(PlannedCell(all_flow, False, None, 1))

    assert selected_outcome.status == all_flow_outcome.status == "ok"
    assert len(commands) == 2
    selected_command, all_flow_command = commands
    assert "--reused-measurement-json" not in selected_command
    assert "--generation-lock-path" in selected_command
    assert "--generation-lock-path" not in all_flow_command
    assert all_flow_command[
        all_flow_command.index("--reused-measurement-json") + 1
    ] == str(selected_current.result_path)
    if old_selected is not None:
        assert selected_current.attempt_id != old_selected.attempt_id
        assert str(old_selected.result_path) not in all_flow_command
    if old_all_flow is not None:
        assert str(old_all_flow.result_path) not in selected_command
        assert str(old_all_flow.result_path) not in all_flow_command


def test_directional_owner_contract_is_scoped_to_otf_lc() -> None:
    selected, _all_flow = _otf_pair()
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == selected.process_key
        and cell.n_final == selected.n_final
        and cell.workload is Workload.SELECTED_FLOW
    )

    assert REPORT_CATALOG.equivalent_cells(recurrence)
