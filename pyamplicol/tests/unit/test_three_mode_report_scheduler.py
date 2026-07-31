# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
)
from tools.performance_report.artifacts import ArtifactStore, CurrentRecord
from tools.performance_report.cache import empty_measurement
from tools.performance_report.campaign_policy import (
    MACBOOK_M3_MEMORY_LIMIT_BYTES,
    MACBOOK_M3_Z_TABLE_F_POLICY,
    X86_EPYC_MEMORY_LIMIT_BYTES,
    X86_EPYC_NATIVE_COMPILER_SLOTS,
    X86_EPYC_POLICY,
    X86_EPYC_PROFILE,
    X86_EPYC_WORKERS,
    PolicyMeasurementState,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import failure_measurement
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.phase_state import WorkerPhaseChannel
from tools.performance_report.resources import ResourceUsage, SupervisedResult
from tools.performance_report.scheduler import (
    CampaignScheduler,
    CampaignSettings,
    CellOutcome,
    CellSelection,
    PlannedCell,
    _fresh_equivalent_current,
    _worker_environment_overrides,
    plan_campaign,
    select_cells,
)
from tools.performance_report.service import ReportPaths, ReportService


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )


def test_epyc_worker_environment_injects_shared_native_compiler_gate(
    tmp_path: Path,
) -> None:
    settings = CampaignSettings(
        workers=X86_EPYC_WORKERS,
        cell_cores=1,
        target_runtime_seconds=5.0,
        max_rss_bytes=X86_EPYC_MEMORY_LIMIT_BYTES,
        allow_symbolica_parallel=True,
        campaign_policy=X86_EPYC_POLICY,
        report_profile=X86_EPYC_PROFILE,
    )

    assert _worker_environment_overrides(settings, tmp_path / "coordination") == {
        "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR": str(
            (tmp_path / "coordination" / "native-compiler-slots").resolve()
        ),
        "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT": str(X86_EPYC_NATIVE_COMPILER_SLOTS),
    }
    assert (
        _worker_environment_overrides(
            CampaignSettings(),
            tmp_path / "strict-coordination",
        )
        == {}
    )


def _ok_measurement(
    cell: CellSpec | None = None,
    *,
    revision: str | None = None,
) -> dict[str, object]:
    measurement = empty_measurement()
    selector = (
        None
        if cell is None or cell.measurement.accuracy is not Accuracy.LC
        else {
            "selected_color_flow_ids": ["flow:2,1"],
            "selected_color_words": [[2, 1]],
            "all_flow_helicity_ids": ["h:-1,+1,-1"],
            "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
            "point_digest": "a" * 64,
        }
    )
    validation: dict[str, object] = {
        "status": ResultStatus.OK.value,
        DIRECT_AGREEMENT_FIELD: [],
    }
    if selector is not None:
        assert cell is not None
        validation[LC_COMMON_COMPONENT_FIELD] = {
            "abi": LC_COMMON_COMPONENT_ABI,
            "cell_id": cell.cell_id,
            "value": 1.0,
            "point_digest": selector["point_digest"],
            "helicity_ids": selector["all_flow_helicity_ids"],
            "color_flow_ids": selector["selected_color_flow_ids"],
        }
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
            "selector_contract": selector,
            "validation": validation,
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


def test_plan_excludes_held_cell_without_suppressing_independent_work(
    tmp_path: Path,
) -> None:
    requested = select_cells(
        CellSelection(
            datasets=frozenset({"matrix_compiled_builtin_sm_lc"}),
            process_keys=frozenset({"dd_z_jets", "ud_w_jets"}),
            multiplicities=frozenset({1}),
            workloads=frozenset({Workload.SELECTED_FLOW}),
        )
    )
    held = next(cell for cell in requested if cell.process_key == "dd_z_jets")

    planned = plan_campaign(
        requested,
        store=_store(tmp_path),
        settings=CampaignSettings(),
        excluded_cell_ids=frozenset({held.cell_id}),
    )

    planned_ids = {item.cell.cell_id for item in planned}
    assert held.cell_id not in planned_ids
    assert any(item.cell.process_key == "ud_w_jets" for item in planned)


def test_plan_excludes_unresolved_descendants_of_held_dependency(
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    initial = plan_campaign(
        (candidate,),
        store=_store(tmp_path / "initial"),
        settings=CampaignSettings(),
    )
    held_baseline = initial[0].cell.cell_id

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path / "excluded"),
        settings=CampaignSettings(),
        excluded_cell_ids=frozenset({held_baseline}),
    )

    assert planned == ()


def test_missing_only_rechecks_completed_candidate_with_missing_dependencies(
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
        _ok_measurement(candidate)
    )

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
    )

    assert candidate.cell_id in {item.cell.cell_id for item in planned}
    target = next(item for item in planned if item.cell == candidate)
    assert target.force_recompare is True
    assert all(
        peer_id in {item.cell.cell_id for item in planned}
        for peer_id in target.comparison_peer_ids
    )


def test_missing_only_skips_only_after_all_dependencies_are_fresh(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    initial = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(),
    )
    for item in initial:
        _publish_current(store, item.cell, revision="same")

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
        expected_revision="same",
    )

    assert planned == ()


def test_four_line_candidates_never_plan_unsupported_legacy_dependencies(
    tmp_path: Path,
) -> None:
    requested = tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.process_key == "dd_4q_lines"
        and cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
    )

    planned = plan_campaign(
        requested,
        store=_store(tmp_path),
        settings=CampaignSettings(),
    )

    assert len(requested) == 32
    assert len(planned) == len(requested)
    assert all(
        item.cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        for item in planned
    )
    recurrence = tuple(
        item
        for item in planned
        if item.cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    cross_mode = tuple(
        item
        for item in planned
        if item.cell.measurement.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    )
    assert recurrence
    assert all(item.baseline_cell_id is None for item in recurrence)
    assert cross_mode
    assert all(item.baseline_cell_id is not None for item in cross_mode)
    assert all(
        REPORT_CATALOG.cell(item.baseline_cell_id).measurement.execution_mode
        is ExecutionMode.RECURRENCE
        for item in cross_mode
        if item.baseline_cell_id is not None
    )
    unavailable_legacy = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    )
    legacy_plan = plan_campaign(
        unavailable_legacy,
        store=_store(tmp_path / "legacy"),
        settings=CampaignSettings(),
    )
    assert len(unavailable_legacy) == 32
    assert legacy_plan == ()


def test_z_native_cap_is_resolved_before_any_attempt_is_planned(
    tmp_path: Path,
) -> None:
    capped = tuple(
        cell
        for cell in REPORT_CATALOG.z_cells()
        if cell.n_final == 7 and cell.variant in {"asm_o3", "cpp_o3"}
    )

    assert len(capped) == 8
    assert {REPORT_CATALOG.static_na_reason(cell) for cell in capped} == {
        "native-backend-generation-cap-n6-v1"
    }
    store_root = tmp_path / "native-cap"
    planned = plan_campaign(
        capped,
        store=_store(store_root),
        settings=CampaignSettings(),
    )

    assert planned == ()
    assert not tuple(store_root.glob("cells/*/attempts/*"))


def test_terminal_censor_rescan_cannot_reinsert_static_na_cell(
    tmp_path: Path,
) -> None:
    lower = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "reference_amplicol_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    higher = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == lower.dataset_id
        and cell.process_key == lower.process_key
        and cell.n_final == 2
        and cell.workload is lower.workload
    )

    class StaticHigherCatalog:
        def __getattr__(self, name: str) -> object:
            return getattr(REPORT_CATALOG, name)

        def static_na_reason(self, cell: CellSpec) -> str | None:
            if cell == higher:
                return "synthetic-static-na"
            return REPORT_CATALOG.static_na_reason(cell)

    settings = CampaignSettings(
        workers=X86_EPYC_WORKERS,
        cell_cores=1,
        target_runtime_seconds=5.0,
        max_rss_bytes=X86_EPYC_MEMORY_LIMIT_BYTES,
        allow_symbolica_parallel=True,
        campaign_policy=X86_EPYC_POLICY,
        report_profile=X86_EPYC_PROFILE,
    )
    planned = plan_campaign(
        (lower, higher),
        store=_store(tmp_path),
        settings=settings,
        catalog=StaticHigherCatalog(),  # type: ignore[arg-type]
    )

    assert tuple(item.cell for item in planned) == (lower,)


def test_campaign_run_rejects_forged_static_na_plan_before_worker_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("reference-amplicol-full-n6-dd-4q-lines-contracted")
    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    monkeypatch.setattr(
        scheduler,
        "_ensure_prepared_model",
        lambda _planned: pytest.fail("static N/A plan reached prepared-model setup"),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_cell",
        lambda _planned: pytest.fail("static N/A plan reached a worker"),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"reference-amplicol-full-n6-dd-4q-lines-contracted.*"
            "original-amplicol-open-quark-line-limit"
        ),
    ):
        scheduler.run(
            (
                PlannedCell(
                    cell,
                    dependency=False,
                    baseline_cell_id=None,
                    rank=0,
                ),
            )
        )

    assert service.store.load_current(cell.cell_id, missing_ok=True) is None
    assert tuple(service.paths.artifact_root.glob("cells/*/attempts/*")) == ()


def test_campaign_cancelled_while_queued_starts_no_preflight_or_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell(
        "z-builtin-sm-n3-dd-z-jets-recurrence-jit-o2-selected-flow"
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(cancellation_requested=lambda: True),
    )
    monkeypatch.setattr(
        scheduler,
        "_ensure_prepared_model",
        lambda _planned: pytest.fail("cancelled campaign reached model preflight"),
    )
    result = scheduler.run(
        (
            PlannedCell(
                cell,
                dependency=False,
                baseline_cell_id=None,
                rank=0,
            ),
        )
    )

    assert result.outcomes == ()
    assert tuple(service.paths.artifact_root.glob("cells/*/attempts/*")) == ()


@pytest.mark.parametrize(
    ("reason", "returncode", "succeeds"),
    (
        ("completed", 0, True),
        ("completed", 1, False),
        ("cancelled", -15, False),
    ),
)
def test_prepared_model_preflight_always_emits_transient_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    returncode: int,
    succeeds: bool,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    planned = PlannedCell(
        cell,
        dependency=False,
        baseline_cell_id=None,
        rank=1,
    )
    prepared_model = tmp_path / "prepared-model.pyamplicol-model"
    prepared_model.write_text("fixture\n", encoding="ascii")
    events: list[dict[str, object]] = []

    def observe(payload: Mapping[str, object]) -> None:
        events.append(dict(payload))

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        if succeeds:
            result_path = Path(command[command.index("--result-json") + 1])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"path": str(prepared_model)}) + "\n",
                encoding="ascii",
            )
        return SupervisedResult(
            returncode,
            reason,  # type: ignore[arg-type]
            ResourceUsage(True, 0, 0, 0, 0.0, 0.1),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(progress_observer=observe),
    )

    if succeeds:
        scheduler._ensure_prepared_model((planned,))
    else:
        with pytest.raises(RuntimeError, match="prepared-model preflight failed"):
            scheduler._ensure_prepared_model((planned,))

    attempt_id = "prepared-model-builtin_sm"
    assert [event["event"] for event in events[:2]] == ["started", "worker"]
    assert events[1]["attempt_id"] == attempt_id
    assert events[-1] == {
        "event": "preflight-finished",
        "cell_id": cell.cell_id,
        "attempt_id": attempt_id,
    }


def test_contracted_n6_multi_quark_plans_separate_legacy_capability(
    tmp_path: Path,
) -> None:
    three_line = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-full-n6-dd-3q-lines-contracted"
    )
    four_line = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-full-n6-dd-4q-lines-contracted"
    )
    four_line_compiled = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n6-dd-4q-lines-contracted"
    )

    three_line_plan = plan_campaign(
        (three_line,),
        store=_store(tmp_path / "three-line"),
        settings=CampaignSettings(),
    )
    assert {item.cell.cell_id for item in three_line_plan} == {
        "reference-amplicol-full-n6-dd-3q-lines-contracted",
        three_line.cell_id,
    }
    assert (
        next(
            item for item in three_line_plan if item.cell == three_line
        ).baseline_cell_id
        == "reference-amplicol-full-n6-dd-3q-lines-contracted"
    )

    four_line_plan = plan_campaign(
        (four_line,),
        store=_store(tmp_path / "four-line"),
        settings=CampaignSettings(),
    )
    assert tuple(item.cell for item in four_line_plan) == (four_line,)
    assert four_line_plan[0].baseline_cell_id is None

    compiled_plan = plan_campaign(
        (four_line_compiled,),
        store=_store(tmp_path / "compiled"),
        settings=CampaignSettings(),
    )
    assert four_line in {item.cell for item in compiled_plan}
    assert (
        next(
            item for item in compiled_plan if item.cell == four_line_compiled
        ).baseline_cell_id
        == four_line.cell_id
    )


def test_missing_only_schedules_stale_direct_peer_and_recomparison(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    initial = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(),
    )
    target = next(item for item in initial if item.cell == candidate)
    baseline_id = target.baseline_cell_id
    stale_peer_id = next(
        peer_id for peer_id in target.comparison_peer_ids if peer_id != baseline_id
    )
    for item in initial:
        _publish_current(
            store,
            item.cell,
            revision=("old" if item.cell.cell_id == stale_peer_id else "new"),
        )

    repaired = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
        expected_revision="new",
    )
    repaired_by_id = {item.cell.cell_id: item for item in repaired}

    assert stale_peer_id in repaired_by_id
    assert candidate.cell_id in repaired_by_id
    assert repaired_by_id[candidate.cell_id].force_recompare is True
    assert repaired_by_id[stale_peer_id].rank < repaired_by_id[candidate.cell_id].rank


def test_z_cell_explicitly_reuses_valid_cross_source_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    candidate = REPORT_CATALOG.cell(
        "z-builtin-sm-n8-dd-z-jets-recurrence-jit-o2-all-flow"
    )
    initial = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(),
    )
    dependency_ids = {item.cell.cell_id for item in initial if item.cell != candidate}
    for item in initial:
        if item.cell == candidate:
            continue
        result = _ok_measurement(item.cell, revision="a" * 40)
        result["provenance"]["report_source_tree"] = "b" * 40
        store.new_attempt(
            item.cell.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(result)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_policy_measurement",
        lambda *_args, **_kwargs: PolicyMeasurementState.SUCCESS,
    )

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(
            workers=1,
            cell_cores=1,
            target_runtime_seconds=5.0,
            max_rss_bytes=MACBOOK_M3_MEMORY_LIMIT_BYTES,
            campaign_policy=MACBOOK_M3_Z_TABLE_F_POLICY,
            report_profile="macbook_M3",
            study_contract_sha256="c" * 64,
            reuse_cross_source_comparison_dependencies=True,
        ),
        expected_revision="d" * 40,
        expected_tree="e" * 40,
    )

    assert dependency_ids
    assert tuple(item.cell for item in planned) == (candidate,)


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
        _ok_measurement(candidate, revision="old")
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
    (root / "docs/arxiv").mkdir(parents=True)
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
    cell: CellSpec,
    *,
    revision: str,
) -> CurrentRecord:
    return store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
        _ok_measurement(cell, revision=revision)
    )


def test_equivalent_reuse_requires_fresh_revision_and_never_uses_amplicol(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    equivalent = REPORT_CATALOG.equivalent_cells(candidate)[0]
    stale = _publish_current(store, equivalent, revision="old")

    assert (
        _fresh_equivalent_current(
            store,
            candidate,
            catalog=REPORT_CATALOG,
            expected_revision="new",
        )
        is None
    )

    fresh = _publish_current(store, equivalent, revision="new")
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
        baseline,
        revision=revision,
    )
    equivalent_record = _publish_current(
        service.store,
        equivalent,
        revision=revision,
    )
    if target_current:
        _publish_current(
            service.store,
            target,
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
            json.dumps(_ok_measurement(target, revision=revision)) + "\n",
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
    assert command[command.index("--docs-dir") + 1].endswith("/repo/docs/arxiv")
    assert command[command.index("--artifact-root") + 1].endswith("/artifacts")
    assert command[command.index("--coordination-root") + 1].endswith("/locks")


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


def test_scheduler_never_publishes_first_failed_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    baseline = REPORT_CATALOG.baseline_cell(target)
    assert baseline is not None
    revision = "current-revision"
    _publish_current(service.store, baseline, revision=revision)

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(
                failure_measurement(
                    ResultStatus.ERROR,
                    "deterministic worker failure",
                )
            )
            + "\n",
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
        settings=CampaignSettings(artifact_policy=ArtifactPolicy.REGENERATE),
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

    assert outcome.status == ResultStatus.ERROR.value
    assert service.store.load_current(target.cell_id, missing_ok=True) is None
    attempt_roots = tuple(
        (service.paths.artifact_root / "cells").glob(f"*/attempts/{outcome.detail}")
    )
    assert len(attempt_roots) == 1
    manifest = json.loads(
        (attempt_roots[0] / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["status"] == "failed"
    assert manifest["result_path"] is None
    assert not (attempt_roots[0] / "result.json").exists()
    worker_result = json.loads(
        (attempt_roots[0] / "worker-result.json").read_text(encoding="ascii")
    )
    assert worker_result["status"] == ResultStatus.ERROR.value


def test_scheduler_never_publishes_first_skip(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    scheduler = CampaignScheduler(service, settings=CampaignSettings())

    outcome = scheduler._publish_skip(
        cell,
        "required dependency is unavailable",
        current=None,
    )

    assert outcome.status == ResultStatus.SKIP.value
    assert service.store.load_current(cell.cell_id, missing_ok=True) is None
    attempt_roots = tuple(
        (service.paths.artifact_root / "cells").glob(f"*/attempts/{outcome.detail}")
    )
    assert len(attempt_roots) == 1
    manifest = json.loads(
        (attempt_roots[0] / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["status"] == "failed"
    assert manifest["result_path"] is None


def test_scheduler_plumbs_authenticated_generation_only_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    baseline = REPORT_CATALOG.baseline_cell(target)
    assert baseline is not None
    revision = "current-revision"
    _publish_current(service.store, baseline, revision=revision)
    captured: dict[str, object] = {}

    def fake_supervise(
        command: Sequence[str],
        **arguments: object,
    ) -> SupervisedResult:
        captured["command"] = tuple(command)
        captured.update(arguments)
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(_ok_measurement(target, revision=revision)) + "\n",
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
        settings=CampaignSettings(generation_time_limit_seconds=7200.0),
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

    assert outcome.status == "ok"
    assert captured["generation_timeout_seconds"] == 7200.0
    channel = captured["phase_channel"]
    assert isinstance(channel, WorkerPhaseChannel)
    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[command.index("--phase-state-path") + 1] == str(channel.path)
    assert command[command.index("--phase-state-run-id") + 1] == channel.run_id
    assert (
        command[command.index("--phase-state-authentication-key") + 1]
        == channel.authentication_key
    )


def test_campaign_run_never_publishes_or_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    planned = (
        PlannedCell(
            target,
            dependency=False,
            baseline_cell_id=None,
            rank=0,
        ),
    )
    monkeypatch.setattr(scheduler, "_ensure_prepared_model", lambda _items: None)
    monkeypatch.setattr(
        scheduler,
        "_run_cell",
        lambda item: CellOutcome(item.cell.cell_id, "ok", "complete"),
    )
    monkeypatch.setattr(
        service,
        "publish",
        lambda **_kwargs: pytest.fail("measurement waited on report publication"),
    )

    result = scheduler.run(planned)

    assert result.failed == ()
    assert result.outcomes[0].cell_id == target.cell_id


def test_successful_current_resume_never_duplicates_post_populate_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    baseline = REPORT_CATALOG.baseline_cell(target)
    assert baseline is not None
    revision = "measured-revision"
    _publish_current(service.store, baseline, revision=revision)
    published = _publish_current(service.store, target, revision=revision)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        lambda *_args, **_kwargs: pytest.fail(
            "authenticated post-populate current must not be rerun"
        ),
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(missing_only=True),
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

    current = service.store.load_current(target.cell_id)
    assert outcome.status == "skipped-current"
    assert outcome.detail == "already complete"
    assert current.attempt_id == published.attempt_id


@pytest.mark.parametrize("limit", (0.0, float("inf"), float("nan")))
def test_campaign_rejects_invalid_generation_only_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="generation_time_limit_seconds"):
        CampaignSettings(generation_time_limit_seconds=limit)
