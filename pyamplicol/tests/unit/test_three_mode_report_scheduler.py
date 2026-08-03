# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    independent_numerical_authorities,
)
from tools.performance_report.artifacts import (
    ArtifactAttempt,
    ArtifactStore,
    CurrentRecord,
    DiskFullError,
)
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
    policy_status_label,
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
from tools.performance_report.resources import (
    GenerationPhaseEvidence,
    ResourceUsage,
    SupervisedResult,
    WorkerObservation,
)
from tools.performance_report.scheduler import (
    CampaignResult,
    CampaignScheduler,
    CampaignSettings,
    CellOutcome,
    CellSelection,
    PlannedCell,
    _CoordinationDeferred,
    _fresh_equivalent_current,
    _partition_dependency_records,
    _PreparationFailed,
    _symbolica_generation_lock_path,
    _worker_environment_overrides,
    plan_campaign,
    select_cells,
)
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity


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

    ufo_compiled = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.model is ModelKey.UFO_SM
        and cell.measurement.execution_mode is ExecutionMode.COMPILED
    )
    strict_root = tmp_path / "strict-coordination"
    assert (
        _symbolica_generation_lock_path(
            CampaignSettings(),
            strict_root,
            ufo_compiled,
        )
        == (strict_root / "symbolica-ufo-compiled-generation.lock").resolve()
    )
    assert (
        _symbolica_generation_lock_path(
            CampaignSettings(allow_symbolica_parallel=True),
            strict_root,
            ufo_compiled,
        )
        is None
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


def test_compiled_plan_can_opt_out_of_automatic_numerical_authorities(
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

    assert len(planned) == 1
    target = planned[0]
    assert target.cell.measurement.execution_mode is ExecutionMode.COMPILED
    assert not target.dependency
    assert target.baseline_cell_id is None
    assert target.optional_baseline_cell_id == (
        "matrix-recurrence-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )
    assert target.numerical_authority_cell_ids == (
        "matrix-recurrence-builtin-sm-lc-n1-dd-z-jets-selected-flow",
        "reference-amplicol-lc-n1-dd-z-jets-selected-flow",
    )
    assert target.prerequisite_cell_ids == ()


@pytest.mark.parametrize(
    "dataset_id",
    ("matrix_compiled_builtin_sm_lc", "matrix_eager_builtin_sm_lc"),
)
def test_automatic_authority_closure_orders_amplicol_recurrence_then_candidate(
    dataset_id: str,
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell(dataset_id)
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path),
        settings=CampaignSettings(
            add_optional_dependencies=True,
            original_amplicol_available=True,
        ),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert set(by_id) == {candidate.cell_id, recurrence.cell_id, amplicol.cell_id}
    assert not by_id[candidate.cell_id].dependency
    assert by_id[recurrence.cell_id].dependency
    assert by_id[amplicol.cell_id].dependency
    assert by_id[amplicol.cell_id].prerequisite_cell_ids == ()
    assert by_id[recurrence.cell_id].prerequisite_cell_ids == (amplicol.cell_id,)
    assert recurrence.cell_id in by_id[candidate.cell_id].prerequisite_cell_ids
    assert by_id[amplicol.cell_id].rank < by_id[recurrence.cell_id].rank
    assert by_id[recurrence.cell_id].rank < by_id[candidate.cell_id].rank


def test_automatic_authority_closure_without_legacy_adds_recurrence_only(
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path),
        settings=CampaignSettings(add_optional_dependencies=True),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert set(by_id) == {candidate.cell_id, recurrence.cell_id}
    assert by_id[recurrence.cell_id].dependency
    assert by_id[candidate.cell_id].prerequisite_cell_ids == (recurrence.cell_id,)


def test_automatic_authority_closure_reuses_active_source_authorities(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    _publish_current(store, recurrence, revision="same")
    _publish_current(store, amplicol, revision="same")

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(
            add_optional_dependencies=True,
            original_amplicol_available=True,
        ),
        expected_revision="same",
    )

    assert tuple(item.cell for item in planned) == (candidate,)


def test_automatic_authority_closure_replans_historical_authorities(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    _publish_current(store, recurrence, revision="old")
    _publish_current(store, amplicol, revision="old")

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(
            add_optional_dependencies=True,
            original_amplicol_available=True,
        ),
        expected_revision="new",
    )

    assert {item.cell.cell_id for item in planned} == {
        candidate.cell_id,
        recurrence.cell_id,
        amplicol.cell_id,
    }


def test_automatic_authority_closure_reuses_terminal_active_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    terminal = _publish_current(store, recurrence, revision="same")

    def current(
        cell: CellSpec,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == recurrence:
            return terminal, PolicyMeasurementState.GENERATION_LIMIT
        return None

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(add_optional_dependencies=True),
        current_resolver=current,
    )

    assert tuple(item.cell for item in planned) == (candidate,)
    assert planned[0].prerequisite_cell_ids == ()
    assert recurrence.cell_id in planned[0].numerical_authority_cell_ids


def test_no_dependencies_added_still_closes_hard_lc_provider_dependency(
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path),
        settings=CampaignSettings(add_optional_dependencies=False),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert candidate.cell_id in by_id
    hard_ids = set(by_id[candidate.cell_id].prerequisite_cell_ids)
    assert hard_ids
    assert all(by_id[cell_id].dependency for cell_id in hard_ids)
    assert all(
        by_id[cell_id].cell.measurement.execution_mode is ExecutionMode.COMPILED
        for cell_id in hard_ids
    )


def test_automatic_authority_addition_recursively_closes_its_dependencies(
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    recurrence_provider = REPORT_CATALOG.cell(
        recurrence.cell_id.replace("all-flow", "selected-flow")
    )

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path),
        settings=CampaignSettings(add_optional_dependencies=True),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert recurrence.cell_id in by_id[candidate.cell_id].prerequisite_cell_ids
    assert recurrence_provider.cell_id in by_id
    assert by_id[recurrence_provider.cell_id].dependency
    assert (
        recurrence_provider.cell_id
        in by_id[recurrence.cell_id].prerequisite_cell_ids
    )
    assert by_id[recurrence_provider.cell_id].rank < by_id[recurrence.cell_id].rank


def test_catalog_wide_automatic_authority_closure_is_exact_and_acyclic(
    tmp_path: Path,
) -> None:
    direct = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
        and REPORT_CATALOG.static_na_reason(cell) is None
    )
    planned = plan_campaign(
        direct,
        store=_store(tmp_path),
        settings=CampaignSettings(
            add_optional_dependencies=True,
            original_amplicol_available=True,
        ),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert len(by_id) == len(planned)
    for candidate in direct:
        item = by_id[candidate.cell_id]
        for authority in independent_numerical_authorities(candidate):
            assert authority.cell_id in by_id
            assert by_id[authority.cell_id].rank < item.rank
            assert (
                authority.process_key,
                authority.n_final,
                authority.measurement.accuracy,
                authority.workload,
            ) == (
                candidate.process_key,
                candidate.n_final,
                candidate.measurement.accuracy,
                candidate.workload,
            )
    for item in planned:
        assert all(
            by_id[prerequisite].rank < item.rank
            for prerequisite in item.prerequisite_cell_ids
        )


def test_explicit_optional_amplicol_orders_recurrence_without_becoming_hard(
    tmp_path: Path,
) -> None:
    recurrence = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None

    planned = plan_campaign(
        (recurrence, amplicol),
        store=_store(tmp_path),
        settings=CampaignSettings(),
    )
    by_id = {item.cell.cell_id: item for item in planned}
    target = by_id[recurrence.cell_id]

    assert set(by_id) == {recurrence.cell_id, amplicol.cell_id}
    assert target.baseline_cell_id is None
    assert target.optional_baseline_cell_id == amplicol.cell_id
    assert target.prerequisite_cell_ids == (amplicol.cell_id,)
    assert by_id[amplicol.cell_id].rank < target.rank


def test_selected_authority_chain_is_encoded_as_ordering_prerequisites(
    tmp_path: Path,
) -> None:
    compiled = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(compiled)
    assert recurrence is not None
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    eager = _matrix_cell("matrix_eager_builtin_sm_lc")

    planned = plan_campaign(
        (compiled, eager, recurrence, amplicol),
        store=_store(tmp_path),
        settings=CampaignSettings(),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert by_id[amplicol.cell_id].prerequisite_cell_ids == ()
    assert by_id[recurrence.cell_id].prerequisite_cell_ids == (amplicol.cell_id,)
    for candidate in (compiled, eager):
        assert by_id[candidate.cell_id].prerequisite_cell_ids == (recurrence.cell_id,)


def test_optional_amplicol_current_is_forwarded_only_when_successful(
    tmp_path: Path,
) -> None:
    recurrence = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    record = _publish_current(_store(tmp_path), amplicol, revision="same")

    baseline, peers, blockers = _partition_dependency_records(
        baseline_cell_id=None,
        comparison_peer_ids=(),
        optional_baseline_cell_id=amplicol.cell_id,
        currents={
            amplicol.cell_id: (record, PolicyMeasurementState.GENERATION_LIMIT)
        },
    )
    assert baseline is None
    assert peers == {}
    assert blockers == ()

    baseline, peers, blockers = _partition_dependency_records(
        baseline_cell_id=None,
        comparison_peer_ids=(),
        optional_baseline_cell_id=amplicol.cell_id,
        currents={amplicol.cell_id: (record, PolicyMeasurementState.SUCCESS)},
    )
    assert baseline is record
    assert peers == {}
    assert blockers == ()


@pytest.mark.parametrize(
    ("state", "expects_baseline"),
    (
        (PolicyMeasurementState.SUCCESS, True),
        (PolicyMeasurementState.GENERATION_LIMIT, False),
    ),
)
def test_scheduler_launches_recurrence_with_only_successful_optional_amplicol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: PolicyMeasurementState,
    expects_baseline: bool,
) -> None:
    service = _service(tmp_path)
    recurrence = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    optional_record = _publish_current(
        service.store,
        amplicol,
        revision="current-revision",
    )
    captured: list[str] = []

    def current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        assert comparison_dependency is (cell == amplicol)
        return (optional_record, state) if cell == amplicol else None

    def supervise(command: Sequence[str], **_kwargs: object) -> SupervisedResult:
        captured.extend(command)
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(_ok_measurement(recurrence, revision="current-revision"))
            + "\n",
            encoding="ascii",
        )
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    scheduler.source_revision = "current-revision"
    monkeypatch.setattr(scheduler, "_current", current)
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        supervise,
    )

    outcome = scheduler._run_cell(
        PlannedCell(
            recurrence,
            dependency=False,
            baseline_cell_id=None,
            optional_baseline_cell_id=amplicol.cell_id,
            rank=1,
        )
    )

    assert outcome.status == ResultStatus.OK.value
    assert ("--baseline-json" in captured) is expects_baseline


@pytest.mark.parametrize(
    ("variant", "workload"),
    (
        ("jit_o1", Workload.SELECTED_FLOW),
        ("jit_o1", Workload.ALL_FLOW),
        ("eager_jit_o2", Workload.SELECTED_FLOW),
        ("eager_jit_o2", Workload.ALL_FLOW),
    ),
)
def test_z_compiled_and_eager_plan_recurrence_without_amplicol(
    tmp_path: Path,
    variant: str,
    workload: Workload,
) -> None:
    candidate = _z_cell(variant=variant, workload=workload)
    baseline = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert baseline is not None

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path),
        settings=CampaignSettings(),
    )
    by_id = {item.cell.cell_id: item for item in planned}
    target = by_id[candidate.cell_id]
    recurrence_id = target.numerical_authority_cell_ids[0]

    assert target.baseline_cell_id is None
    assert target.optional_baseline_cell_id == baseline.cell_id
    assert target.numerical_authority_cell_ids == (
        recurrence_id,
        baseline.cell_id,
    )
    assert recurrence_id in target.optional_comparison_peer_ids
    assert recurrence_id not in by_id
    assert recurrence_id not in target.prerequisite_cell_ids
    assert all(
        item.cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        for item in planned
    )


@pytest.mark.parametrize(
    ("variant", "workload"),
    (
        ("jit_o1", Workload.SELECTED_FLOW),
        ("jit_o1", Workload.ALL_FLOW),
        ("eager_jit_o2", Workload.SELECTED_FLOW),
        ("eager_jit_o2", Workload.ALL_FLOW),
    ),
)
@pytest.mark.parametrize(
    ("recurrence_state", "amplicol_state", "authority_index"),
    (
        (
            PolicyMeasurementState.SUCCESS,
            PolicyMeasurementState.SUCCESS,
            0,
        ),
        (
            PolicyMeasurementState.GENERATION_LIMIT,
            PolicyMeasurementState.SUCCESS,
            1,
        ),
        (None, PolicyMeasurementState.SUCCESS, 1),
    ),
)
def test_z_worker_prefers_recurrence_then_falls_back_to_amplicol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    workload: Workload,
    recurrence_state: PolicyMeasurementState | None,
    amplicol_state: PolicyMeasurementState,
    authority_index: int,
) -> None:
    service = _service(tmp_path)
    candidate = _z_cell(variant=variant, workload=workload)
    baseline = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert baseline is not None
    planned = plan_campaign(
        (candidate,),
        store=service.store,
        settings=CampaignSettings(),
    )
    target = next(item for item in planned if item.cell == candidate)
    recurrence = REPORT_CATALOG.cell(target.numerical_authority_cell_ids[0])
    dependency_records = {
        peer_id: _publish_current(
            service.store,
            REPORT_CATALOG.cell(peer_id),
            revision="current-revision",
        )
        for peer_id in target.comparison_peer_ids
    }
    recurrence_record = _publish_current(
        service.store,
        recurrence,
        revision="current-revision",
    )
    amplicol_record = _publish_current(
        service.store,
        baseline,
        revision="current-revision",
    )
    captured: list[str] = []

    def current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == candidate:
            assert comparison_dependency is False
            return None
        assert comparison_dependency is True
        if cell == baseline:
            return amplicol_record, amplicol_state
        if cell == recurrence:
            return (
                None
                if recurrence_state is None
                else (recurrence_record, recurrence_state)
            )
        record = dependency_records.get(cell.cell_id)
        return (
            None
            if record is None
            else (record, PolicyMeasurementState.SUCCESS)
        )

    def supervise(command: Sequence[str], **_kwargs: object) -> SupervisedResult:
        captured.extend(command)
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(_ok_measurement(candidate, revision="current-revision"))
            + "\n",
            encoding="ascii",
        )
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    scheduler.source_revision = "current-revision"
    monkeypatch.setattr(scheduler, "_current", current)
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        supervise,
    )

    outcome = scheduler._run_cell(target)

    expected_baseline = (recurrence_record, amplicol_record)[authority_index]
    assert outcome.status == ResultStatus.OK.value
    assert captured[captured.index("--baseline-json") + 1] == str(
        expected_baseline.result_path
    )
    assert tuple(
        captured[index + 1]
        for index, value in enumerate(captured)
        if value == "--expected-authority-cell-id"
    ) == target.numerical_authority_cell_ids
    assert captured[captured.index("--selected-authority-cell-id") + 1] == (
        target.numerical_authority_cell_ids[authority_index]
    )


@pytest.mark.parametrize(
    "recurrence_state",
    (None, PolicyMeasurementState.GENERATION_LIMIT),
)
def test_z_worker_runs_when_every_numerical_authority_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recurrence_state: PolicyMeasurementState | None,
) -> None:
    service = _service(tmp_path)
    candidate = _z_cell(
        variant="jit_o1",
        workload=Workload.SELECTED_FLOW,
    )
    baseline = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert baseline is not None
    planned = plan_campaign(
        (candidate,),
        store=service.store,
        settings=CampaignSettings(),
    )
    target = next(item for item in planned if item.cell == candidate)
    recurrence = REPORT_CATALOG.cell(target.numerical_authority_cell_ids[0])
    amplicol_record = _publish_current(
        service.store,
        baseline,
        revision="current-revision",
    )
    terminal_recurrence = _ok_measurement(
        recurrence,
        revision="current-revision",
    )
    terminal_recurrence["status"] = ResultStatus.TIMEOUT.value
    provenance = terminal_recurrence["provenance"]
    assert isinstance(provenance, dict)
    provenance["policy_censor_sha256"] = "f" * 64
    recurrence_record = service.store.new_attempt(
        recurrence.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(terminal_recurrence)

    def current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == candidate:
            return None
        assert comparison_dependency is True
        if cell == baseline:
            return amplicol_record, PolicyMeasurementState.GENERATION_LIMIT
        if cell == recurrence:
            return (
                None
                if recurrence_state is None
                else (recurrence_record, recurrence_state)
            )
        pytest.fail(f"unexpected dependency query for {cell.cell_id}")

    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    scheduler.source_revision = "current-revision"
    monkeypatch.setattr(scheduler, "_current", current)
    launched: list[Sequence[str]] = []

    def supervise(command: Sequence[str], **_kwargs: object) -> SupervisedResult:
        launched.append(command)
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(
                failure_measurement(
                    ResultStatus.ERROR,
                    "synthetic worker reached without authority",
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

    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        supervise,
    )

    outcome = scheduler._run_cell(target)

    assert outcome.status == ResultStatus.ERROR.value
    assert len(launched) == 1
    command = launched[0]
    assert "--baseline-json" not in command
    assert tuple(
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--expected-authority-cell-id"
    ) == target.numerical_authority_cell_ids
    assert "--selected-authority-cell-id" not in command


def test_postworker_late_recurrence_supersedes_prelaunch_amplicol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    candidate = _z_cell(variant="jit_o1", workload=Workload.SELECTED_FLOW)
    target = plan_campaign(
        (candidate,),
        store=service.store,
        settings=CampaignSettings(),
    )[0]
    recurrence = REPORT_CATALOG.cell(target.numerical_authority_cell_ids[0])
    amplicol = REPORT_CATALOG.cell(target.numerical_authority_cell_ids[1])
    recurrence_record = _publish_current(
        service.store,
        recurrence,
        revision="current-revision",
    )
    amplicol_record = _publish_current(
        service.store,
        amplicol,
        revision="current-revision",
    )
    worker_finished = False

    def current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == candidate:
            return None
        assert comparison_dependency is True
        if cell == recurrence:
            return (
                (recurrence_record, PolicyMeasurementState.SUCCESS)
                if worker_finished
                else None
            )
        if cell == amplicol:
            return amplicol_record, PolicyMeasurementState.SUCCESS
        pytest.fail(f"unexpected dependency query for {cell.cell_id}")

    captured: list[str] = []

    def supervise(command: Sequence[str], **_kwargs: object) -> SupervisedResult:
        nonlocal worker_finished
        captured.extend(command)
        result_path = Path(command[command.index("--result-json") + 1])
        measurement = _ok_measurement(candidate, revision="current-revision")
        validation = measurement["validation"]
        assert isinstance(validation, dict)
        validation["independent_authority"] = {
            "abi": "pyamplicol-report-independent-authority-v1",
            "expected_cell_ids": list(target.numerical_authority_cell_ids),
            "selected_cell_id": amplicol.cell_id,
            "status": "verified",
            "reason": "independent-authority-agreement",
            "same_artifact_diagnostics_are_authority": False,
        }
        validation["pointwise"] = {"status": ResultStatus.OK.value}
        result_path.write_text(json.dumps(measurement) + "\n", encoding="ascii")
        worker_finished = True
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    reconciled: list[str] = []

    def reconcile(
        _cell: CellSpec,
        measurement: Mapping[str, object],
        _authority: Mapping[str, object],
        *,
        authority_cell_id: str,
    ) -> dict[str, object]:
        reconciled.append(authority_cell_id)
        return dict(measurement)

    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    scheduler.source_revision = "current-revision"
    monkeypatch.setattr(scheduler, "_current", current)
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        supervise,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.reconcile_independent_authority",
        reconcile,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.attach_direct_agreements",
        lambda *_args, **_kwargs: None,
    )

    outcome = scheduler._run_cell(target)

    assert outcome.status == ResultStatus.OK.value
    assert captured[captured.index("--selected-authority-cell-id") + 1] == (
        amplicol.cell_id
    )
    assert reconciled == [recurrence.cell_id]


def test_postworker_tampered_unverified_is_rejected_before_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    candidate = _z_cell(variant="jit_o1", workload=Workload.SELECTED_FLOW)
    target = plan_campaign(
        (candidate,),
        store=service.store,
        settings=CampaignSettings(),
    )[0]
    recurrence = REPORT_CATALOG.cell(target.numerical_authority_cell_ids[0])
    recurrence_record = _publish_current(
        service.store,
        recurrence,
        revision="current-revision",
    )
    worker_finished = False

    def current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == candidate:
            return None
        assert comparison_dependency is True
        if cell == recurrence and worker_finished:
            return recurrence_record, PolicyMeasurementState.SUCCESS
        return None

    def supervise(command: Sequence[str], **_kwargs: object) -> SupervisedResult:
        nonlocal worker_finished
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps(
                failure_measurement(
                    ResultStatus.UNVERIFIED,
                    "tampered diagnostic fixture",
                )
            )
            + "\n",
            encoding="ascii",
        )
        worker_finished = True
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    scheduler.source_revision = "current-revision"
    monkeypatch.setattr(scheduler, "_current", current)
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        supervise,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.reconcile_independent_authority",
        lambda *_args, **_kwargs: pytest.fail(
            "tampered diagnostic must be rejected before reconciliation"
        ),
    )

    with pytest.raises(ValueError, match="measured result requires"):
        scheduler._run_cell(target)


def test_selected_authorities_finish_before_compiled_and_eager_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = CampaignScheduler(
        _service(tmp_path),
        settings=CampaignSettings(workers=2),
    )
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    amplicol = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert amplicol is not None
    eager = _matrix_cell("matrix_eager_builtin_sm_lc")
    start_order: list[ExecutionMode] = []
    order_guard = threading.Lock()

    def run_in_lane(item: PlannedCell) -> CellOutcome:
        with order_guard:
            start_order.append(item.cell.measurement.execution_mode)
        if item.cell == amplicol:
            return CellOutcome(amplicol.cell_id, "ok", "authority complete")
        if item.cell == recurrence:
            assert start_order == [ExecutionMode.AMPLICOL, ExecutionMode.RECURRENCE]
            return CellOutcome(recurrence.cell_id, "ok", "authority complete")
        if item.cell == candidate:
            assert ExecutionMode.RECURRENCE in start_order
            return CellOutcome(candidate.cell_id, "ok", "verified")
        assert item.cell == eager
        assert ExecutionMode.RECURRENCE in start_order
        return CellOutcome(eager.cell_id, "ok", "verified")

    monkeypatch.setattr(scheduler, "_run_cell_in_lane", run_in_lane)
    result = scheduler.run(
        tuple(
            reversed(
                (
                    PlannedCell(amplicol, False, None, 0),
                    PlannedCell(
                        recurrence,
                        False,
                        None,
                        1,
                        prerequisite_cell_ids=(amplicol.cell_id,),
                    ),
                    PlannedCell(
                        candidate,
                        False,
                        None,
                        2,
                        numerical_authority_cell_ids=(
                            recurrence.cell_id,
                            amplicol.cell_id,
                        ),
                        prerequisite_cell_ids=(recurrence.cell_id,),
                    ),
                    PlannedCell(
                        eager,
                        False,
                        None,
                        3,
                        numerical_authority_cell_ids=(
                            recurrence.cell_id,
                            amplicol.cell_id,
                        ),
                        prerequisite_cell_ids=(recurrence.cell_id,),
                    ),
                )
            )
        )
    )

    assert start_order[:2] == [ExecutionMode.AMPLICOL, ExecutionMode.RECURRENCE]
    assert set(start_order[2:]) == {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    assert {outcome.cell_id for outcome in result.outcomes} == {
        recurrence.cell_id,
        amplicol.cell_id,
        candidate.cell_id,
        eager.cell_id,
    }


@pytest.mark.parametrize(
    "authority_status",
    ("error", "generation_limit", "memory_limit", "worker_timeout"),
)
def test_terminal_optional_authority_releases_baseline_free_candidate(
    authority_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = CampaignScheduler(
        _service(tmp_path),
        settings=CampaignSettings(workers=2),
    )
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    recurrence = REPORT_CATALOG.validation_baseline_cell(candidate)
    assert recurrence is not None
    start_order: list[str] = []

    def run_in_lane(item: PlannedCell) -> CellOutcome:
        start_order.append(item.cell.cell_id)
        if item.cell == recurrence:
            return CellOutcome(recurrence.cell_id, authority_status, "terminal")
        assert item.cell == candidate
        return CellOutcome(candidate.cell_id, "unverified", "baseline-free")

    monkeypatch.setattr(scheduler, "_run_cell_in_lane", run_in_lane)
    result = scheduler.run(
        (
            PlannedCell(recurrence, False, None, 0),
            PlannedCell(
                candidate,
                False,
                None,
                1,
                numerical_authority_cell_ids=(recurrence.cell_id,),
                prerequisite_cell_ids=(recurrence.cell_id,),
            ),
        )
    )

    assert start_order == [recurrence.cell_id, candidate.cell_id]
    outcomes = {outcome.cell_id: outcome.status for outcome in result.outcomes}
    assert outcomes == {
        recurrence.cell_id: authority_status,
        candidate.cell_id: "unverified",
    }


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


def test_plan_does_not_exclude_candidate_for_held_optional_authority(
    tmp_path: Path,
) -> None:
    candidate = _matrix_cell("matrix_compiled_builtin_sm_lc")
    initial = plan_campaign(
        (candidate,),
        store=_store(tmp_path / "initial"),
        settings=CampaignSettings(),
    )
    assert len(initial) == 1
    held_authority = initial[0].numerical_authority_cell_ids[0]

    planned = plan_campaign(
        (candidate,),
        store=_store(tmp_path / "excluded"),
        settings=CampaignSettings(),
        excluded_cell_ids=frozenset({held_authority}),
    )

    assert tuple(item.cell for item in planned) == (candidate,)


def test_missing_only_reuses_valid_candidate_without_optional_authorities(
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

    assert planned == ()


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

    assert len(requested) == 40
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
    assert all(item.baseline_cell_id is None for item in cross_mode)
    assert all(
        item.numerical_authority_cell_ids
        and REPORT_CATALOG.cell(
            item.numerical_authority_cell_ids[0]
        ).measurement.execution_mode
        is ExecutionMode.RECURRENCE
        for item in cross_mode
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
    assert len(unavailable_legacy) == 34
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
        **arguments: object,
    ) -> SupervisedResult:
        observation_callback = arguments["observation_callback"]
        assert callable(observation_callback)
        observation_callback(
            WorkerObservation(
                pid=100,
                usage=ResourceUsage(True, 0, 0, 0, 0.0, 0.1),
                phase="waiting-for-reap",
                phase_sequence=None,
                member_pids=(100,),
            )
        )
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
    assert any(
        event.get("event") == "resource"
        and event.get("phase") == "waiting-for-reap"
        for event in events
    )
    assert events[-1] == {
        "event": "preflight-finished",
        "cell_id": cell.cell_id,
        "attempt_id": attempt_id,
    }


@pytest.mark.parametrize(
    "three_line_key",
    ("dd_3q_lines", "dd_3q_identical_lines"),
)
def test_contracted_n6_multi_quark_plans_separate_legacy_capability(
    tmp_path: Path,
    three_line_key: str,
) -> None:
    three_line = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-full-n6-"
        f"{three_line_key.replace('_', '-')}-contracted"
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
    assert tuple(item.cell for item in three_line_plan) == (three_line,)
    three_line_item = three_line_plan[0]
    assert three_line_item.baseline_cell_id is None
    assert three_line_item.optional_baseline_cell_id == (
        "reference-amplicol-full-n6-"
        f"{three_line_key.replace('_', '-')}-contracted"
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
    assert tuple(item.cell for item in compiled_plan) == (four_line_compiled,)
    compiled_item = compiled_plan[0]
    assert compiled_item.baseline_cell_id is None
    assert compiled_item.numerical_authority_cell_ids == (four_line.cell_id,)


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


@pytest.mark.parametrize(
    "execution_mode",
    (ExecutionMode.COMPILED, ExecutionMode.EAGER),
)
def test_missing_only_reconciles_amplicol_current_when_recurrence_later_succeeds(
    tmp_path: Path,
    execution_mode: ExecutionMode,
) -> None:
    store = _store(tmp_path)
    candidate = REPORT_CATALOG.cell(
        f"matrix-{execution_mode.value}-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )
    authorities = independent_numerical_authorities(candidate)
    recurrence, amplicol = authorities
    candidate_measurement = _ok_measurement(candidate, revision="current")
    validation = candidate_measurement["validation"]
    assert isinstance(validation, dict)
    validation[INDEPENDENT_AUTHORITY_FIELD] = {
        "abi": "pyamplicol-report-independent-authority-v1",
        "expected_cell_ids": [authority.cell_id for authority in authorities],
        "selected_cell_id": amplicol.cell_id,
        "status": "verified",
        "reason": "independent-authority-agreement",
        "same_artifact_diagnostics_are_authority": False,
    }
    candidate_current = store.new_attempt(
        candidate.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(candidate_measurement)
    recurrence_current = _publish_current(store, recurrence, revision="current")

    def current(
        cell: CellSpec,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell == candidate:
            return candidate_current, PolicyMeasurementState.SUCCESS
        if cell == recurrence:
            return recurrence_current, PolicyMeasurementState.SUCCESS
        return None

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(missing_only=True),
        current_resolver=current,
    )

    assert tuple(item.cell for item in planned) == (candidate,)
    assert planned[0].force_recompare is True


@pytest.mark.parametrize("dataset_id", ("scalar_contact", "scalar_gravity"))
def test_self_certified_scalar_compiled_current_is_stale(
    tmp_path: Path,
    dataset_id: str,
) -> None:
    cell = next(
        candidate
        for candidate in REPORT_CATALOG.measurement_cells()
        if candidate.dataset_id == dataset_id
    )
    store = _store(tmp_path)
    _publish_current(store, cell, revision="active")

    planned = plan_campaign(
        (cell,),
        store=store,
        settings=CampaignSettings(),
        expected_revision="active",
    )

    assert tuple(item.cell for item in planned) == (cell,)
    assert planned[0].numerical_authority_cell_ids == ()


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


def test_z_table_f_preflight_does_not_require_optional_amplicol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    candidate = REPORT_CATALOG.cell(
        "z-builtin-sm-n8-dd-z-jets-recurrence-jit-o2-all-flow"
    )
    settings = CampaignSettings(
        workers=1,
        cell_cores=1,
        target_runtime_seconds=5.0,
        max_rss_bytes=MACBOOK_M3_MEMORY_LIMIT_BYTES,
        campaign_policy=MACBOOK_M3_Z_TABLE_F_POLICY,
        report_profile="macbook_M3",
        study_contract_sha256="c" * 64,
    )
    planned = plan_campaign(
        (candidate,),
        store=service.store,
        settings=settings,
    )
    candidate_plan = next(item for item in planned if item.cell == candidate)
    optional_ids = {
        dependency_id
        for dependency_id in (
            candidate_plan.optional_baseline_cell_id,
            *candidate_plan.optional_comparison_peer_ids,
        )
        if dependency_id is not None
    }
    required_ids = {
        dependency_id
        for dependency_id in (
            candidate_plan.baseline_cell_id,
            *candidate_plan.comparison_peer_ids,
        )
        if dependency_id is not None
    }
    queried: list[str] = []
    scheduler = object.__new__(CampaignScheduler)
    scheduler.settings = settings
    scheduler.catalog = REPORT_CATALOG

    def current(cell: CellSpec, **_kwargs: object) -> object:
        queried.append(cell.cell_id)
        return object()

    monkeypatch.setattr(scheduler, "_current", current)

    scheduler._validate_z_table_f_plan((candidate_plan,))

    assert optional_ids
    assert set(queried) == required_ids
    assert set(queried).isdisjoint(optional_ids)


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

    assert tuple(item.cell for item in planned) == (candidate,)


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


def _z_cell(*, variant: str, workload: Workload) -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "z_builtin_sm"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.variant == variant
        and cell.workload is workload
    )


def _service(tmp_path: Path) -> ReportService:
    root = tmp_path / "repo"
    (root / "src/pyamplicol/_profiling_campaign").mkdir(parents=True)
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
    monkeypatch.setattr(
        "tools.performance_report.scheduler._symbolica_generation_lock_path",
        lambda *_arguments: tmp_path / "generation.lock",
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=policy,
            rerun=rerun,
            target_runtime_seconds=1.0,
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
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
    assert command[command.index("--docs-dir") + 1].endswith(
        "/repo/src/pyamplicol/_profiling_campaign"
    )
    assert command[command.index("--artifact-root") + 1].endswith("/artifacts")
    assert command[command.index("--coordination-root") + 1].endswith("/locks")
    assert "--generation-lock-path" not in command


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
    assert command[command.index("--generation-lock-path") + 1].endswith(
        "/generation.lock"
    )


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
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
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


def test_scheduler_result_rewrite_preserves_actionable_disk_full_error(
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
            json.dumps(_ok_measurement(target, revision=revision)) + "\n",
            encoding="ascii",
        )
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 1, 1, 0, 0.1, 0.1),
        )

    original_write_json = ArtifactAttempt.write_json

    def fail_result_rewrite(
        attempt: ArtifactAttempt,
        relative_path: str,
        payload: Mapping[str, object],
    ) -> Path:
        if relative_path == "worker-result.json":
            raise DiskFullError(
                f"disk full while writing {attempt.root / relative_path}; "
                "0 bytes available"
            )
        return original_write_json(attempt, relative_path, payload)

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    monkeypatch.setattr(ArtifactAttempt, "write_json", fail_result_rewrite)
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(artifact_policy=ArtifactPolicy.REGENERATE),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    scheduler.source_revision = revision

    with pytest.raises(
        DiskFullError,
        match=r"disk full while writing .*worker-result.json; 0 bytes available",
    ):
        scheduler._run_cell(
            PlannedCell(
                target,
                dependency=False,
                baseline_cell_id=baseline.cell_id,
                rank=1,
            )
        )

    assert service.store.load_current(target.cell_id, missing_ok=True) is None


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


def test_dependency_block_retains_exact_prerequisite_identity(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    prerequisite = "reference-amplicol-lc-n1-dd-z-jets-selected-flow"

    outcome = scheduler._publish_blocked_dependency(
        cell,
        prerequisite,
        current=None,
    )

    assert outcome.status == "blocked_dependency"
    assert outcome.prerequisite_cell_ids == (prerequisite,)
    assert outcome.detail == (
        f"blocked by dependency: required prerequisite {prerequisite!r} is unavailable"
    )
    assert service.store.load_current(cell.cell_id, missing_ok=True) is None
    attempts = tuple((service.store._cell_root(cell.cell_id) / "attempts").iterdir())
    assert len(attempts) == 1
    result = json.loads(
        (attempts[0] / "worker-result.json").read_text(encoding="utf-8")
    )
    assert result["blocked_dependency"] == {
        "prerequisite_cell_ids": [prerequisite]
    }


def test_dag_terminal_direct_peer_blocks_dependent_before_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    prerequisite = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    dependent = _matrix_cell("matrix_compiled_builtin_sm_lc")
    observed: list[Mapping[str, object]] = []
    launched: list[str] = []
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            workers=1,
            max_rss_bytes=30_000_000_000,
            manual_terminal_censors=True,
            source_identity_override=ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
            progress_observer=lambda payload: observed.append(dict(payload)),
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
    blocked_after_terminal: list[str] = []
    publish_blocked = scheduler._publish_blocked_dependency

    def assert_prerequisite_is_terminal_before_blocking(
        cell: CellSpec,
        prerequisite_cell_ids: str | Sequence[str],
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        prerequisite_current = service.store.load_current(prerequisite.cell_id)
        assert prerequisite_current is not None
        assert prerequisite_current.result["status"] == ResultStatus.MEMORY_LIMIT.value
        blocked_after_terminal.append(cell.cell_id)
        return publish_blocked(
            cell,
            prerequisite_cell_ids,
            current=current,
        )

    monkeypatch.setattr(
        scheduler,
        "_publish_blocked_dependency",
        assert_prerequisite_is_terminal_before_blocking,
    )

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        cell_id = str(command[command.index("--cell-id") + 1])
        launched.append(cell_id)
        if cell_id != prerequisite.cell_id:
            pytest.fail("dependency-blocked worker was launched")
        return SupervisedResult(
            -9,
            "memory_limit",
            ResourceUsage(
                True,
                30_000_000_001,
                30_000_000_001,
                0,
                0.1,
                0.2,
            ),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    planned = (
        PlannedCell(prerequisite, dependency=True, baseline_cell_id=None, rank=0),
        PlannedCell(
            dependent,
            dependency=False,
            baseline_cell_id=None,
            rank=1,
            comparison_peer_ids=(prerequisite.cell_id,),
            prerequisite_cell_ids=(prerequisite.cell_id,),
        ),
    )

    result = scheduler.run(planned)

    assert launched == [prerequisite.cell_id]
    outcomes = {outcome.cell_id: outcome for outcome in result.outcomes}
    assert outcomes[prerequisite.cell_id].status == "memory_limit"
    blocked = outcomes[dependent.cell_id]
    assert blocked.status == "blocked_dependency"
    assert blocked.prerequisite_cell_ids == (prerequisite.cell_id,)
    assert blocked_after_terminal == [dependent.cell_id]
    finished = [
        payload
        for payload in observed
        if payload.get("event") == "finished"
        and payload.get("cell_id") == dependent.cell_id
    ]
    assert len(finished) == 1
    assert finished[0]["prerequisite_cell_ids"] == (prerequisite.cell_id,)


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
        settings=CampaignSettings(
            timeout_seconds=90.0,
            generation_time_limit_seconds=7200.0,
            profiling_time_limit_seconds=60.0,
            validation_time_limit_seconds=30.0,
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)
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
    assert captured["timeout_seconds"] == 90.0
    assert captured["generation_timeout_seconds"] == 7200.0
    assert captured["profiling_timeout_seconds"] == 60.0
    assert captured["validation_timeout_seconds"] == 30.0
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
    assert command[command.index("--worker-wall-limit") + 1] == "90.0"
    assert command[command.index("--profiling-time-limit") + 1] == "60.0"
    assert command[command.index("--validation-time-limit") + 1] == "30.0"


def test_scheduler_persists_abrupt_worker_exit_diagnostics_with_empty_worker_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    captured: dict[str, object] = {}

    def crash(
        command: Sequence[str],
        **arguments: object,
    ) -> SupervisedResult:
        captured.update(arguments)
        Path(command[command.index("--log-path") + 1]).touch()
        phase_error = "worker exited before closing generation"
        return SupervisedResult(
            -11,
            "worker_exit",
            ResourceUsage(True, 0, 1234, 0, 0.2, 0.3),
            generation_phase=GenerationPhaseEvidence(
                configured_timeout_seconds=3600.0,
                supervisor_reason="worker_exit",
                authenticated=False,
                run_id="phase-run-7",
                worker_pid=4321,
                final_sequence=1,
                final_phase="generation",
                generation_started_monotonic_ns=10,
                generation_finished_monotonic_ns=None,
                generation_elapsed_seconds=0.3,
                final_state_sha256="c" * 64,
                error=phase_error,
            ),
            pid=4321,
            member_pids=(4321, 4322),
            signal_number=11,
            signal_name="SIGSEGV",
            supervisor_stderr="native evaluator abort\n",
            supervisor_stderr_limit_bytes=65536,
            phase_state_error=phase_error,
            started_at_utc="2026-08-02T00:00:00+00:00",
            finished_at_utc="2026-08-02T00:00:01+00:00",
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        crash,
    )
    source_revision = "a" * 40
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            source_identity_override=ReportSourceIdentity(
                source_revision,
                "b" * 40,
                (),
            ),
            campaign_invocation_id="manual-invocation-7",
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)

    outcome = scheduler._run_cell(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=0)
    )

    assert outcome.status == "error"
    assert captured["capture_stderr"] is True
    attempt_roots = tuple(
        (service.paths.artifact_root / "cells").glob(f"*/attempts/{outcome.detail}")
    )
    assert len(attempt_roots) == 1
    attempt_root = attempt_roots[0]
    assert (attempt_root / "worker.log").read_bytes() == b""
    result = json.loads(
        (attempt_root / "worker-result.json").read_text(encoding="ascii")
    )
    assert result["status"] == "error"
    assert result["failure"] == {
        "kind": "WorkerProcessExitError",
        "message": "worker terminated by SIGSEGV (signal 11, return code -11)",
    }
    supervisor = result["resources"]["supervisor"]
    assert supervisor == {
        "abi": "pyamplicol-report-worker-supervisor-v1",
        "campaign_invocation_id": "manual-invocation-7",
        "finished_at_utc": "2026-08-02T00:00:01+00:00",
        "member_pids": [4321, 4322],
        "phase": "generation",
        "phase_state_error": "worker exited before closing generation",
        "pid": 4321,
        "reason": "worker_exit",
        "returncode": -11,
        "signal_name": "SIGSEGV",
        "signal_number": 11,
        "source_revision": source_revision,
        "started_at_utc": "2026-08-02T00:00:00+00:00",
        "stderr": "native evaluator abort\n",
        "stderr_limit_bytes": 65536,
        "stderr_truncated": False,
        "teardown_escalated": False,
        "teardown_seconds": 0.0,
    }
    assert result["provenance"]["report_source_revision"] == source_revision


def test_scheduler_persists_and_forwards_exact_phase_state_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    phase_error = (
        "worker phase-state transition timestamp is outside the supervised "
        "process lifetime"
    )

    def invalid_phase(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        Path(command[command.index("--log-path") + 1]).touch()
        return SupervisedResult(
            -15,
            "phase_state_error",
            ResourceUsage(True, 0, 1234, 0, 0.2, 0.3),
            pid=4321,
            member_pids=(4321,),
            phase_state_error=phase_error,
    )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        invalid_phase,
    )
    events: list[dict[str, object]] = []
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            source_identity_override=ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
            progress_observer=lambda payload: events.append(dict(payload)),
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)

    outcome = scheduler._run_cell(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=0)
    )

    assert outcome.status == ResultStatus.ERROR.value
    assert outcome.terminal_detail == phase_error
    attempt_roots = tuple(
        (service.paths.artifact_root / "cells").glob(
            f"*/attempts/{outcome.detail}"
        )
    )
    assert len(attempt_roots) == 1
    result = json.loads(
        (attempt_roots[0] / "worker-result.json").read_text(encoding="ascii")
    )
    assert result["failure"]["message"] == phase_error
    assert result["resources"]["supervisor"]["reason"] == "phase_state_error"
    assert result["resources"]["supervisor"]["phase_state_error"] == phase_error
    finished = [event for event in events if event["event"] == "finished"]
    assert len(finished) == 1
    assert finished[0]["terminal_detail"] == phase_error


def test_prepared_model_worker_exit_persists_structured_failed_attempt(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    source_revision = "a" * 40
    phase_error = "worker exited before closing preparation"
    supervised = SupervisedResult(
        -6,
        "worker_exit",
        ResourceUsage(True, 0, 2048, 1, 0.4, 0.5),
        pid=7654,
        member_pids=(7654, 7655),
        signal_number=6,
        signal_name="SIGABRT",
        supervisor_stderr="native model compiler abort\n",
        supervisor_stderr_limit_bytes=65536,
        phase_state_error=phase_error,
        started_at_utc="2026-08-02T01:00:00+00:00",
        finished_at_utc="2026-08-02T01:00:01+00:00",
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            source_identity_override=ReportSourceIdentity(
                source_revision,
                "b" * 40,
                (),
            ),
            campaign_invocation_id="manual-preflight-9",
        ),
    )

    outcome = scheduler._preparation_failure_outcome(
        cell,
        _PreparationFailed(
            ModelKey.BUILTIN_SM,
            "worker_exit",
            "prepared-model worker aborted",
            supervised,
        ),
    )

    assert outcome.status == "error"
    attempt_roots = tuple(
        (service.paths.artifact_root / "cells").glob(f"*/attempts/{outcome.detail}")
    )
    assert len(attempt_roots) == 1
    attempt_root = attempt_roots[0]
    result = json.loads(
        (attempt_root / "worker-result.json").read_text(encoding="ascii")
    )
    assert result["status"] == "error"
    assert result["failure"] == {
        "kind": "WorkerProcessExitError",
        "message": "worker terminated by SIGABRT (signal 6, return code -6)",
    }
    assert result["resources"]["supervisor"] == {
        "abi": "pyamplicol-report-worker-supervisor-v1",
        "campaign_invocation_id": "manual-preflight-9",
        "finished_at_utc": "2026-08-02T01:00:01+00:00",
        "member_pids": [7654, 7655],
        "phase": "preparation",
        "phase_state_error": phase_error,
        "pid": 7654,
        "reason": "worker_exit",
        "returncode": -6,
        "signal_name": "SIGABRT",
        "signal_number": 6,
        "source_revision": source_revision,
        "started_at_utc": "2026-08-02T01:00:00+00:00",
        "stderr": "native model compiler abort\n",
        "stderr_limit_bytes": 65536,
        "stderr_truncated": False,
        "teardown_escalated": False,
        "teardown_seconds": 0.0,
    }
    assert result["provenance"]["report_source_revision"] == source_revision
    manifest = json.loads((attempt_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert [record["path"] for record in manifest["artifacts"]] == [
        "worker-result.json"
    ]


@pytest.mark.parametrize(
    ("reason", "state", "kind", "label"),
    (
        ("generation_timeout", "generation_limit", "generation_limit", ">7s"),
        ("worker_timeout", "worker_timeout", "worker_timeout", "worker >11s"),
        (
            "profiling_timeout",
            "profiling_timeout",
            "profiling_timeout",
            "profile >5s",
        ),
        (
            "validation_timeout",
            "validation_timeout",
            "validation_timeout",
            "validation >3s",
        ),
    ),
)
def test_manual_stage_timeout_is_published_as_distinct_addressed_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    state: str,
    kind: str,
    label: str,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        lambda _command, **_arguments: SupervisedResult(
            -15,
            reason,
            ResourceUsage(True, 0, 1, 0, 0.1, 11.0),
        ),
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            timeout_seconds=11.0,
            generation_time_limit_seconds=7.0,
            profiling_time_limit_seconds=5.0,
            validation_time_limit_seconds=3.0,
            manual_terminal_censors=True,
            source_identity_override=ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)

    outcome = scheduler._run_cell(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=0)
    )

    assert outcome.status == state
    assert not CampaignResult((PlannedCell(cell, False, None, 0),), (outcome,)).failed
    current = service.store.load_current(cell.cell_id)
    assert current is not None
    assert current.result["status"] == ResultStatus.TIMEOUT.value
    assert current.result["resources"]["terminal_reason"] == reason
    assert current.result["provenance"]["policy_censor"]["kind"] == kind
    assert policy_status_label(current.result) == label
    resolved = scheduler._current(cell)
    assert resolved is not None
    assert resolved[1].value == state
    for timing in (
        "generation_seconds",
        "wall_seconds_per_point",
        "execution_seconds_per_point",
        "sample_count",
    ):
        assert current.result[timing] is None


def test_manual_clean_child_profiling_timeout_keeps_typed_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")

    class ProfilingTimeLimitError(RuntimeError):
        pass

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
        result_path = Path(command[command.index("--result-json") + 1])
        result = failure_measurement(
            ResultStatus.TIMEOUT,
            ProfilingTimeLimitError("profiling budget exhausted"),
            resources={"terminal_reason": "profiling_timeout"},
        )
        result_path.write_text(json.dumps(result) + "\n", encoding="ascii")
        return SupervisedResult(
            0,
            "completed",
            ResourceUsage(True, 0, 1, 0, 0.1, 5.0),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            timeout_seconds=11.0,
            generation_time_limit_seconds=7.0,
            profiling_time_limit_seconds=5.0,
            validation_time_limit_seconds=3.0,
            manual_terminal_censors=True,
            source_identity_override=ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
        ),
    )
    monkeypatch.setattr(scheduler, "_prepare_model_for", lambda _planned: None)

    outcome = scheduler._run_cell(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=0)
    )

    assert outcome.status == PolicyMeasurementState.PROFILING_TIMEOUT.value
    current = service.store.load_current(cell.cell_id)
    assert current is not None
    assert current.result["failure"]["kind"] == "ProfilingTimeLimitError"
    assert current.result["resources"]["terminal_reason"] == "profiling_timeout"


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


def test_shared_model_preparation_does_not_hold_independent_ready_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(workers=2),
    )
    recurrence = PlannedCell(
        _matrix_cell("matrix_recurrence_builtin_sm_lc"),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    scalar = PlannedCell(
        REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted"),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    preparation_started = threading.Event()
    independent_started = threading.Event()

    def prepare(_items: Sequence[PlannedCell]) -> None:
        preparation_started.set()
        assert independent_started.wait(timeout=1.0)
        scheduler._prepared_model_paths[ModelKey.BUILTIN_SM] = (
            tmp_path / "prepared-model.bin"
        )

    def run_cell(item: PlannedCell) -> CellOutcome:
        if item.cell.cell_id == recurrence.cell.cell_id:
            scheduler._prepare_model_for(item)
        else:
            assert preparation_started.wait(timeout=1.0)
            independent_started.set()
        return CellOutcome(item.cell.cell_id, "ok", "complete")

    monkeypatch.setattr(scheduler, "_ensure_prepared_model", prepare)
    monkeypatch.setattr(scheduler, "_run_cell", run_cell)

    result = scheduler.run((recurrence, scalar))

    assert {outcome.cell_id for outcome in result.outcomes} == {
        recurrence.cell.cell_id,
        scalar.cell.cell_id,
    }


def test_failed_shared_model_preparation_is_memoized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = CampaignScheduler(_service(tmp_path), settings=CampaignSettings())
    planned = PlannedCell(
        _matrix_cell("matrix_recurrence_builtin_sm_lc"),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    calls = 0
    failure = _PreparationFailed(
        ModelKey.BUILTIN_SM,
        "worker_timeout",
        "prepared model timed out",
    )

    def fail(_items: Sequence[PlannedCell]) -> None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(scheduler, "_ensure_prepared_model", fail)

    with pytest.raises(_PreparationFailed):
        scheduler._prepare_model_for(planned)
    with pytest.raises(_PreparationFailed):
        scheduler._prepare_model_for(planned)

    assert calls == 1


def test_preparation_failure_is_handled_while_cell_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    scheduler = CampaignScheduler(service, settings=CampaignSettings())
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    planned = PlannedCell(
        cell,
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    held: set[str] = set()

    class Lock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            held.add(self.name)

        def __exit__(self, *_arguments: object) -> None:
            held.remove(self.name)

    monkeypatch.setattr(
        service.store,
        "named_lock",
        lambda name, **_kwargs: Lock(name),
    )

    def fail(_planned: PlannedCell) -> None:
        raise _PreparationFailed(
            ModelKey.BUILTIN_SM,
            "error",
            "preparation failed",
        )

    monkeypatch.setattr(scheduler, "_prepare_model_for", fail)

    def handle(
        _cell: CellSpec,
        _failure: _PreparationFailed,
    ) -> CellOutcome:
        assert f"campaign-cell-{cell.cell_id}" in held
        return CellOutcome(cell.cell_id, "preparation_error", "handled")

    monkeypatch.setattr(scheduler, "_preparation_failure_outcome", handle)

    outcome = scheduler._run_cell(planned)

    assert outcome.status == "preparation_error"
    assert held == set()


@pytest.mark.parametrize(
    ("worker_limit", "generation_limit", "expected_reason"),
    (
        (11.0, 7.0, "generation_timeout"),
        (5.0, 7.0, "worker_timeout"),
    ),
)
def test_preparation_timeout_keeps_the_effective_limit_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_limit: float,
    generation_limit: float,
    expected_reason: str,
) -> None:
    scheduler = CampaignScheduler(
        _service(tmp_path),
        settings=CampaignSettings(
            timeout_seconds=worker_limit,
            generation_time_limit_seconds=generation_limit,
        ),
    )
    planned = PlannedCell(
        _matrix_cell("matrix_recurrence_builtin_sm_lc"),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    captured: dict[str, object] = {}

    def timeout(_command: Sequence[str], **arguments: object) -> SupervisedResult:
        captured.update(arguments)
        return SupervisedResult(
            -15,
            "worker_timeout",
            ResourceUsage(True, 0, 1, 0, 0.1, min(worker_limit, generation_limit)),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        timeout,
    )

    with pytest.raises(_PreparationFailed) as raised:
        scheduler._ensure_prepared_model((planned,))

    assert raised.value.reason == expected_reason
    assert captured["timeout_seconds"] == min(worker_limit, generation_limit)


def test_manual_preparation_wall_timeout_is_generation_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    supervised = SupervisedResult(
        -15,
        "worker_timeout",
        ResourceUsage(True, 0, 1, 0, 0.1, 7.0),
    )
    failure = _PreparationFailed(
        ModelKey.BUILTIN_SM,
        "generation_timeout",
        "prepared model timed out",
        supervised,
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            timeout_seconds=11.0,
            generation_time_limit_seconds=7.0,
            profiling_time_limit_seconds=5.0,
            validation_time_limit_seconds=3.0,
            manual_terminal_censors=True,
            source_identity_override=ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
        ),
    )

    def fail(_planned: PlannedCell) -> None:
        raise failure

    monkeypatch.setattr(scheduler, "_prepare_model_for", fail)

    outcome = scheduler._run_cell(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=0)
    )

    assert outcome.status == PolicyMeasurementState.GENERATION_LIMIT.value
    current = service.store.load_current(cell.cell_id)
    assert current is not None
    assert current.result["resources"]["terminal_reason"] == "generation_timeout"


def test_ready_queue_admits_later_rank_while_an_earlier_cell_is_still_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cells = tuple(
        sorted(
            (
                REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted"),
                REPORT_CATALOG.cell("scalar-gravity-n2-scalar-gravity-contracted"),
                REPORT_CATALOG.cell("scalar-contact-n3-scalar-contact-contracted"),
            ),
            key=lambda cell: cell.cell_id,
        )
    )
    slow, quick, later = cells
    release_slow = threading.Event()
    later_started = threading.Event()
    failures: list[BaseException] = []
    results: list[object] = []

    scheduler = CampaignScheduler(service, settings=CampaignSettings(workers=2))
    monkeypatch.setattr(scheduler, "_ensure_prepared_model", lambda _items: None)

    def run_cell(item: PlannedCell) -> CellOutcome:
        if item.cell == slow:
            assert release_slow.wait(2.0)
        elif item.cell == later:
            later_started.set()
        return CellOutcome(item.cell.cell_id, "ok", "complete")

    monkeypatch.setattr(scheduler, "_run_cell", run_cell)
    planned = (
        PlannedCell(slow, False, None, 0),
        PlannedCell(quick, False, None, 0),
        PlannedCell(later, False, None, 4),
    )

    def run() -> None:
        try:
            results.append(scheduler.run(planned))
        except BaseException as error:  # pragma: no cover - asserted below.
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert later_started.wait(1.0), "later ready work remained behind a rank wave"
    release_slow.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert failures == []
    assert len(results) == 1


def test_ready_queue_releases_dependents_without_waiting_for_unrelated_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    prerequisite = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    unrelated = REPORT_CATALOG.cell("scalar-gravity-n2-scalar-gravity-contracted")
    dependent = REPORT_CATALOG.cell("scalar-contact-n3-scalar-contact-contracted")
    release_unrelated = threading.Event()
    dependent_started = threading.Event()
    failures: list[BaseException] = []

    scheduler = CampaignScheduler(service, settings=CampaignSettings(workers=2))
    monkeypatch.setattr(scheduler, "_ensure_prepared_model", lambda _items: None)

    def run_cell(item: PlannedCell) -> CellOutcome:
        if item.cell == unrelated:
            assert release_unrelated.wait(2.0)
        elif item.cell == dependent:
            dependent_started.set()
        return CellOutcome(item.cell.cell_id, "ok", "complete")

    monkeypatch.setattr(scheduler, "_run_cell", run_cell)
    planned = (
        PlannedCell(prerequisite, False, None, 0),
        PlannedCell(unrelated, False, None, 0),
        PlannedCell(
            dependent,
            False,
            None,
            3,
            prerequisite_cell_ids=(prerequisite.cell_id,),
        ),
    )

    def run() -> None:
        try:
            scheduler.run(planned)
        except BaseException as error:  # pragma: no cover - asserted below.
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert dependent_started.wait(1.0)
    release_unrelated.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert failures == []


def test_busy_cell_lock_is_deferred_without_consuming_the_only_worker_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cells = tuple(
        sorted(
            (
                REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted"),
                REPORT_CATALOG.cell("scalar-gravity-n2-scalar-gravity-contracted"),
            ),
            key=lambda cell: cell.cell_id,
        )
    )
    locked, available = cells
    available_finished = threading.Event()
    observed: list[str] = []
    progress_events: list[Mapping[str, object]] = []
    failures: list[BaseException] = []

    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            workers=1,
            progress_observer=lambda payload: progress_events.append(dict(payload)),
        ),
    )
    monkeypatch.setattr(scheduler, "_ensure_prepared_model", lambda _items: None)

    def run_in_lane(item: PlannedCell) -> CellOutcome:
        observed.append(item.cell.cell_id)
        if item.cell == available:
            available_finished.set()
        return CellOutcome(item.cell.cell_id, "ok", "complete")

    monkeypatch.setattr(scheduler, "_run_cell_in_lane", run_in_lane)
    planned = (
        PlannedCell(locked, False, None, 0),
        PlannedCell(available, False, None, 0),
    )

    def run() -> None:
        try:
            scheduler.run(planned)
        except BaseException as error:  # pragma: no cover - asserted below.
            failures.append(error)

    with service.store.named_lock(f"campaign-cell-{locked.cell_id}"):
        thread = threading.Thread(target=run)
        thread.start()
        assert available_finished.wait(1.0)
    thread.join(2.0)

    assert not thread.is_alive()
    assert failures == []
    assert observed == [available.cell_id, locked.cell_id]
    assert not any(
        payload.get("event") == "finished"
        and payload.get("cell_id") == locked.cell_id
        and payload.get("status") == "error"
        for payload in progress_events
    )


def test_busy_equivalent_owner_lock_does_not_block_independent_ready_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    target = _matrix_cell("matrix_recurrence_builtin_sm_lc")
    equivalent = REPORT_CATALOG.equivalent_cells(target)[0]
    baseline = REPORT_CATALOG.baseline_cell(target)
    assert baseline is not None
    independent = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-nlc-n1-dd-z-jets-contracted"
    )
    revision = "current-revision"
    owner = _publish_current(service.store, equivalent, revision=revision)
    _publish_current(service.store, baseline, revision=revision)
    _publish_current(service.store, independent, revision=revision)
    independent_finished = threading.Event()
    failures: list[BaseException] = []
    results: list[CampaignResult] = []

    def observe(payload: Mapping[str, object]) -> None:
        if (
            payload.get("event") == "finished"
            and payload.get("cell_id") == independent.cell_id
        ):
            independent_finished.set()

    def fake_supervise(
        command: Sequence[str],
        **_arguments: object,
    ) -> SupervisedResult:
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
        settings=CampaignSettings(
            workers=1,
            artifact_policy=ArtifactPolicy.REUSE,
            progress_observer=observe,
        ),
    )
    scheduler.source_revision = revision
    planned = (
        PlannedCell(target, False, baseline.cell_id, 0),
        PlannedCell(independent, False, None, 1),
    )

    def run() -> None:
        try:
            results.append(scheduler.run(planned))
        except BaseException as error:  # pragma: no cover - asserted below.
            failures.append(error)

    with service.store.named_lock(f"campaign-artifact-use-{owner.attempt_id}"):
        thread = threading.Thread(target=run)
        thread.start()
        assert independent_finished.wait(1.0)
    thread.join(2.0)

    assert not thread.is_alive()
    assert failures == []
    assert len(results) == 1
    assert {outcome.cell_id for outcome in results[0].outcomes} == {
        target.cell_id,
        independent.cell_id,
    }


def test_ufo_compiled_cells_overlap_outside_the_worker_side_compiler_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(workers=2),
    )
    ufo_compiled = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.model is ModelKey.UFO_SM
        and cell.measurement.execution_mode is ExecutionMode.COMPILED
    )[:2]
    assert len(ufo_compiled) == 2
    generating = threading.Event()
    profiling = threading.Event()

    def run_in_lane(item: PlannedCell) -> CellOutcome:
        if item.cell == ufo_compiled[0]:
            generating.set()
            assert profiling.wait(1.0)
        else:
            assert generating.wait(1.0)
            profiling.set()
        return CellOutcome(item.cell.cell_id, "ok", "complete")

    monkeypatch.setattr(scheduler, "_run_cell_in_lane", run_in_lane)
    result = scheduler.run(
        tuple(PlannedCell(cell, False, None, 0) for cell in ufo_compiled)
    )

    assert generating.is_set() and profiling.is_set()
    assert not result.failed

    # A reusable cell does not touch the worker-side compiler gate at all.
    monkeypatch.setattr(
        scheduler,
        "_run_cell_in_lane",
        lambda item: CellOutcome(item.cell.cell_id, "reused", "current"),
    )
    with service.store.named_lock("campaign-symbolica-compiled-capacity-1"):
        outcome = scheduler._run_cell(PlannedCell(ufo_compiled[0], False, None, 0))

    assert outcome.status == "reused"
    assert "campaign-symbolica-compiled-capacity-1" not in (
        scheduler._coordination_lock_names(ufo_compiled[0])
    )


def test_plan_campaign_builds_a_real_resource_predecessor_chain(
    tmp_path: Path,
) -> None:
    lane = tuple(
        sorted(
            (
                cell
                for cell in REPORT_CATALOG.measurement_cells()
                if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
                and cell.measurement.accuracy is Accuracy.LC
                and cell.workload is Workload.SELECTED_FLOW
                and cell.process_key == "dd_z_jets"
                and cell.n_final in {1, 2, 3}
            ),
            key=lambda cell: cell.n_final,
        )
    )
    assert tuple(cell.n_final for cell in lane) == (1, 2, 3)
    settings = CampaignSettings(
        campaign_policy=X86_EPYC_POLICY,
        report_profile=X86_EPYC_PROFILE,
        workers=X86_EPYC_WORKERS,
        max_rss_bytes=X86_EPYC_MEMORY_LIMIT_BYTES,
        allow_symbolica_parallel=True,
    )
    planned = plan_campaign(
        lane,
        store=_store(tmp_path),
        settings=settings,
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert by_id[lane[0].cell_id].resource_predecessor_ids == ()
    assert by_id[lane[1].cell_id].resource_predecessor_ids == (lane[0].cell_id,)
    assert by_id[lane[2].cell_id].resource_predecessor_ids == (lane[1].cell_id,)


def test_same_model_preparation_lock_contention_is_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = CampaignScheduler(_service(tmp_path), settings=CampaignSettings())
    planned = PlannedCell(
        _matrix_cell("matrix_recurrence_builtin_sm_lc"),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    started = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def prepare(_items: Sequence[PlannedCell]) -> None:
        started.set()
        assert release.wait(2.0)

    monkeypatch.setattr(scheduler, "_ensure_prepared_model", prepare)

    def first() -> None:
        try:
            scheduler._prepare_model_for(planned)
        except BaseException as error:  # pragma: no cover - asserted below.
            failures.append(error)

    thread = threading.Thread(target=first)
    thread.start()
    assert started.wait(1.0)
    with pytest.raises(_CoordinationDeferred):
        scheduler._prepare_model_for(planned)
    release.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert failures == []


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
