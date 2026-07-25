# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.performance_report.artifacts import ArtifactStore, CurrentRecord
from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.resources import ResourceUsage, SupervisedResult
from tools.performance_report.scheduler import (
    CampaignScheduler,
    CampaignSettings,
    CellOutcome,
    CellSelection,
    PlannedCell,
    _fresh_equivalent_current,
    plan_campaign,
    select_cells,
)
from tools.performance_report.service import ReportPaths, ReportService


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )


def _ok_measurement(*, revision: str | None = None) -> dict[str, object]:
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
            "validation": {"status": ResultStatus.OK.value},
            "resources": {},
            "provenance": (
                {} if revision is None else {"report_source_revision": revision}
            ),
        }
    )
    return measurement


def test_selection_combines_all_supported_filter_axes() -> None:
    selected = select_cells(
        CellSelection(
            datasets=frozenset({"matrix_eager_builtin_sm_lc"}),
            modes=frozenset({ExecutionMode.EAGER}),
            models=frozenset({ModelKey.BUILTIN_SM}),
            accuracies=frozenset({Accuracy.LC}),
            process_keys=frozenset({"dd_z_jets"}),
            multiplicities=frozenset({3}),
            workloads=frozenset({Workload.ALL_FLOW}),
        )
    )

    assert len(selected) == 1
    assert selected[0].cell_id.endswith("all-flow")


def test_dependency_plan_orders_amplicol_recurrence_then_candidate(
    tmp_path: Path,
) -> None:
    candidate = select_cells(
        CellSelection(
            datasets=frozenset({"matrix_compiled_builtin_sm_lc"}),
            process_keys=frozenset({"dd_z_jets"}),
            multiplicities=frozenset({1}),
            workloads=frozenset({Workload.SELECTED_FLOW}),
        )
    )
    planned = plan_campaign(
        candidate,
        store=_store(tmp_path),
        settings=CampaignSettings(),
    )

    assert len(planned) == 3
    assert [item.rank for item in planned] == [0, 1, 2]
    assert [item.cell.measurement.execution_mode for item in planned] == [
        ExecutionMode.AMPLICOL,
        ExecutionMode.RECURRENCE,
        ExecutionMode.COMPILED,
    ]
    assert [item.dependency for item in planned] == [True, True, False]


def test_missing_only_skips_completed_candidate_and_dependencies(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    store.new_attempt(candidate.cell_id, ArtifactPolicy.REGENERATE).publish(
        _ok_measurement()
    )

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
    )

    assert planned == ()


def test_missing_only_rejects_stale_report_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    store.new_attempt(candidate.cell_id, ArtifactPolicy.REGENERATE).publish(
        _ok_measurement(revision="old")
    )

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
        expected_revision="new",
    )

    assert [item.rank for item in planned] == [0, 1, 2]


def _matrix_cell(
    dataset_id: str,
    *,
    workload: Workload = Workload.SELECTED_FLOW,
) -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == dataset_id
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
    )


def _service(tmp_path: Path) -> ReportService:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-tests@example.invalid"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Tests"),
        cwd=root,
        check=True,
    )
    (root / "README.md").write_text("# report fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "README.md"), cwd=root, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "Initialize fixture"),
        cwd=root,
        check=True,
    )
    return ReportService(
        ReportPaths.from_repo(
            root,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "locks",
        )
    )


def _publish_current(
    store: ArtifactStore,
    cell_id: str,
    *,
    revision: str,
) -> CurrentRecord:
    return store.new_attempt(cell_id, ArtifactPolicy.REGENERATE).publish(
        _ok_measurement(revision=revision)
    )


def test_equivalent_reuse_requires_fresh_revision_and_never_uses_amplicol(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    equivalent = REPORT_CATALOG.equivalent_cells(candidate)[0]
    stale = _publish_current(store, equivalent.cell_id, revision="old")

    assert (
        _fresh_equivalent_current(
            store,
            candidate,
            catalog=REPORT_CATALOG,
            expected_revision="new",
        )
        is None
    )

    fresh = _publish_current(store, equivalent.cell_id, revision="new")
    selected = _fresh_equivalent_current(
        store,
        candidate,
        catalog=REPORT_CATALOG,
        expected_revision="new",
    )
    assert selected is not None
    assert selected.attempt_id == fresh.attempt_id
    assert selected.attempt_id != stale.attempt_id

    reference = next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    assert (
        _fresh_equivalent_current(
            store,
            reference,
            catalog=REPORT_CATALOG,
            expected_revision="new",
        )
        is None
    )


def _run_with_captured_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: ArtifactPolicy,
    rerun: bool = False,
    target_current: bool = False,
) -> tuple[list[str], CellOutcome, CurrentRecord, CurrentRecord]:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    equivalent = REPORT_CATALOG.equivalent_cells(target)[0]
    baseline = REPORT_CATALOG.baseline_cell(target)
    assert baseline is not None
    revision = "current-revision"
    baseline_record = _publish_current(
        service.store,
        baseline.cell_id,
        revision=revision,
    )
    equivalent_record = _publish_current(
        service.store,
        equivalent.cell_id,
        revision=revision,
    )
    if target_current:
        _publish_current(
            service.store,
            target.cell_id,
            revision=revision,
        )
    command_seen: list[str] = []

    def fake_supervise(
        command: Sequence[str],
        **_kwargs: object,
    ) -> SupervisedResult:
        command_seen.extend(command)
        result_path = Path(command_seen[command_seen.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(_ok_measurement(revision=revision)) + "\n",
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
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=policy,
            rerun=rerun,
            target_runtime_seconds=1.0,
        ),
    )
    scheduler.source_revision = revision
    outcome = scheduler._run_cell(
        PlannedCell(
            target,
            dependency=False,
            baseline_cell_id=baseline.cell_id,
            rank=1,
        )
    )
    return command_seen, outcome, equivalent_record, baseline_record


@pytest.mark.parametrize("policy", (ArtifactPolicy.REUSE, ArtifactPolicy.RETIME))
def test_reuse_and_retime_use_equivalent_artifact_but_target_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: ArtifactPolicy,
) -> None:
    command, outcome, equivalent, baseline = _run_with_captured_worker(
        tmp_path,
        monkeypatch,
        policy=policy,
    )

    assert outcome.status == "ok"
    assert command[command.index("--reused-measurement-json") + 1] == str(
        equivalent.result_path
    )
    assert command[command.index("--baseline-json") + 1] == str(baseline.result_path)
    assert command[command.index("--cell-id") + 1] == (
        "matrix-recurrence-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )


@pytest.mark.parametrize(
    ("policy", "rerun", "target_current"),
    (
        (ArtifactPolicy.REGENERATE, False, False),
        (ArtifactPolicy.RETIME, True, True),
        (ArtifactPolicy.REUSE, True, True),
    ),
)
def test_explicit_regenerate_or_rerun_forces_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: ArtifactPolicy,
    rerun: bool,
    target_current: bool,
) -> None:
    command, outcome, _equivalent, _baseline = _run_with_captured_worker(
        tmp_path,
        monkeypatch,
        policy=policy,
        rerun=rerun,
        target_current=target_current,
    )

    assert outcome.status == "ok"
    assert "--reused-measurement-json" not in command
