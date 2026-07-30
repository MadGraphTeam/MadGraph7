# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.agreements import (
    BUILTIN_UFO_RECURRENCE,
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    LC_CROSS_LAYOUT_COMPONENT,
    LC_LEGACY_PYAMPLICOL_COMPONENT,
    Z_RECURRENCE_CROSS_MODE,
    agreement_edges,
    attach_direct_agreements,
    evaluate_lc_common_component,
    incoming_agreement_edges,
)
from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.cache import (
    build_reset_cache,
    empty_measurement,
    validate_cache,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.final_audit import (
    FinalAuditError,
    _audit_replayed_direct_agreements,
    _direct_replay_category,
    _ReplayObservation,
)
from tools.performance_report.models import (
    ArtifactPolicy,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.runner import (
    SelectorContract,
    pointwise_validation,
)
from tools.performance_report.scheduler import CampaignSettings, plan_campaign


def _selector() -> dict[str, object]:
    return {
        "selected_color_flow_ids": ["flow:2,1"],
        "selected_color_words": [[2, 1]],
        "all_flow_helicity_ids": ["h:-1,+1,-1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
        "point_digest": "a" * 64,
    }


def _measurement(cell_id: str, value: float, component: float) -> dict[str, object]:
    selector = _selector()
    return {
        "status": ResultStatus.OK.value,
        "matrix_element": value,
        "selector_contract": selector,
        "validation": {
            "status": ResultStatus.OK.value,
            DIRECT_AGREEMENT_FIELD: [],
            LC_COMMON_COMPONENT_FIELD: {
                "abi": LC_COMMON_COMPONENT_ABI,
                "cell_id": cell_id,
                "value": component,
                "point_digest": selector["point_digest"],
                "helicity_ids": selector["all_flow_helicity_ids"],
                "color_flow_ids": selector["selected_color_flow_ids"],
            },
        },
        "failure": None,
    }


def _cell(
    dataset_id: str,
    *,
    workload: Workload,
    variant: str | None = None,
):
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == dataset_id
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
        and cell.variant == variant
    )


def test_lc_common_component_uses_signed_zero_runtime_alias() -> None:
    cell = _cell(
        "matrix_recurrence_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    contract = SelectorContract(
        selected_color_flow_ids=("flow:2,1",),
        selected_color_words=((2, 1),),
        all_flow_helicity_ids=("h:-1,+1,-1,+1,-1,0",),
        all_flow_source_helicities=(
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 0),
        ),
        point_digest="a" * 64,
    )

    class Runtime:
        def evaluate_resolved(self, _points: object, **selectors: object) -> object:
            assert selectors["helicities"] == ("h:-1,+1,-1,+1,-1,+0",)
            return SimpleNamespace(
                helicity_ids=("h:-1,+1,-1,+1,-1,+0",),
                color_ids=("flow:2,1",),
                values=(((1.0 + 0.0j,),),),
            )

    component = evaluate_lc_common_component(
        Runtime(),
        object(),
        cell=cell,
        contract=contract,
    )

    assert component["value"] == 1.0
    assert component["helicity_ids"] == ["h:-1,+1,-1,+1,-1,0"]


def test_canonical_n4_direct_agreement_graph_has_exact_locked_counts() -> None:
    counts = Counter(
        edge.kind for edge in agreement_edges(maximum_n_final=4)
    )

    assert counts == {
        BUILTIN_UFO_RECURRENCE: 136,
        Z_RECURRENCE_CROSS_MODE: 80,
        LC_CROSS_LAYOUT_COMPONENT: 208,
        LC_LEGACY_PYAMPLICOL_COMPONENT: 176,
    }
    assert Counter(
        _direct_replay_category(edge)
        for edge in agreement_edges(maximum_n_final=4)
    ) == {
        "fully-replayed-pyamplicol": 392,
        "replayed-pyamplicol-vs-authenticated-legacy": 176,
        "authenticated-stored-legacy-layout": 32,
    }


def test_full_direct_agreement_graph_excludes_unavailable_four_line_legacy() -> None:
    edges = agreement_edges()
    counts = Counter(edge.kind for edge in edges)
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_lc"
        and cell.process_key == "dd_4q_lines"
        and cell.n_final == 6
        and cell.workload is Workload.ALL_FLOW
    )
    legacy = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "reference_amplicol_lc"
        and cell.process_key == "dd_4q_lines"
        and cell.n_final == 6
        and cell.workload is Workload.ALL_FLOW
    )

    assert counts == {
        BUILTIN_UFO_RECURRENCE: 306,
        Z_RECURRENCE_CROSS_MODE: 180,
        LC_CROSS_LAYOUT_COMPONENT: 590,
        LC_LEGACY_PYAMPLICOL_COMPONENT: 484,
    }
    assert {
        edge.kind for edge in incoming_agreement_edges(candidate)
    } == {LC_CROSS_LAYOUT_COMPONENT}
    assert incoming_agreement_edges(legacy) == ()

    measurable_ids = {
        cell.cell_id
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is None
    }
    measurable_edges = tuple(
        edge
        for edge in edges
        if edge.candidate.cell_id in measurable_ids
        and edge.baseline.cell_id in measurable_ids
    )
    assert Counter(edge.kind for edge in measurable_edges) == {
        BUILTIN_UFO_RECURRENCE: 306,
        Z_RECURRENCE_CROSS_MODE: 156,
        LC_CROSS_LAYOUT_COMPONENT: 578,
        LC_LEGACY_PYAMPLICOL_COMPONENT: 472,
    }
    assert Counter(
        _direct_replay_category(edge) for edge in measurable_edges
    ) == {
        "fully-replayed-pyamplicol": 946,
        "replayed-pyamplicol-vs-authenticated-legacy": 472,
        "authenticated-stored-legacy-layout": 94,
    }


def test_three_line_lc_still_requires_legacy_and_layout_agreements() -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_lc"
        and cell.process_key == "dd_3q_lines"
        and cell.n_final == 4
        and cell.workload is Workload.ALL_FLOW
    )

    assert {
        edge.kind for edge in incoming_agreement_edges(candidate)
    } == {
        LC_CROSS_LAYOUT_COMPONENT,
        LC_LEGACY_PYAMPLICOL_COMPONENT,
    }


def test_ufo_recurrence_can_pass_independent_oracle_and_fail_direct_edge() -> None:
    candidate = _cell(
        "matrix_recurrence_ufo_sm_lc",
        workload=Workload.SELECTED_FLOW,
    )
    edge = incoming_agreement_edges(candidate)[0]
    independent = 1.0
    builtin = _measurement(edge.baseline.cell_id, 1.0, 0.25)
    ufo = _measurement(candidate.cell_id, 1.0 + 5.0e-9, 0.25)

    assert pointwise_validation(
        float(builtin["matrix_element"]),
        independent,
        relative_tolerance=1.0e-8,
    )["status"] == ResultStatus.OK.value
    assert pointwise_validation(
        float(ufo["matrix_element"]),
        independent,
        relative_tolerance=1.0e-8,
    )["status"] == ResultStatus.OK.value

    attach_direct_agreements(
        candidate,
        ufo,
        {edge.baseline.cell_id: builtin},
    )

    assert ufo["status"] == ResultStatus.VALIDATION_FAILED.value
    record = ufo["validation"][DIRECT_AGREEMENT_FIELD][0]
    assert record["edge_kind"] == BUILTIN_UFO_RECURRENCE
    assert record["relative_tolerance"] == 1.0e-12
    assert record["absolute_tolerance"] == 1.0e-15
    assert record["status"] == ResultStatus.VALIDATION_FAILED.value


def test_z_timing_baseline_stays_amplicol_but_numerics_consume_recurrence() -> None:
    candidate = _cell(
        "z_builtin_sm",
        workload=Workload.SELECTED_FLOW,
        variant="jit_o1",
    )
    timing_baseline = REPORT_CATALOG.baseline_cell(candidate)
    assert timing_baseline is not None
    assert timing_baseline.measurement.execution_mode is ExecutionMode.AMPLICOL
    edge = incoming_agreement_edges(candidate)[0]
    assert edge.kind == Z_RECURRENCE_CROSS_MODE
    assert edge.baseline.variant == "recurrence_jit_o2"

    amplicol_value = 1.0
    recurrence = _measurement(edge.baseline.cell_id, 1.0, 0.25)
    compiled = _measurement(candidate.cell_id, 1.0 + 5.0e-9, 0.25)
    assert pointwise_validation(
        float(compiled["matrix_element"]),
        amplicol_value,
        relative_tolerance=1.0e-8,
    )["status"] == ResultStatus.OK.value

    attach_direct_agreements(
        candidate,
        compiled,
        {edge.baseline.cell_id: recurrence},
    )

    assert compiled["status"] == ResultStatus.VALIDATION_FAILED.value
    record = compiled["validation"][DIRECT_AGREEMENT_FIELD][0]
    assert record["edge_kind"] == Z_RECURRENCE_CROSS_MODE
    assert record["baseline_cell_id"] == edge.baseline.cell_id


def test_cross_layout_component_failure_is_not_hidden_by_equal_totals() -> None:
    candidate = _cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    edges = incoming_agreement_edges(candidate)
    edge = next(item for item in edges if item.kind == LC_CROSS_LAYOUT_COMPONENT)
    legacy_edge = next(
        item for item in edges if item.kind == LC_LEGACY_PYAMPLICOL_COMPONENT
    )
    topology = _measurement(edge.baseline.cell_id, 1.0, 0.25)
    union = _measurement(candidate.cell_id, 1.0, 0.25 + 5.0e-9)
    legacy = _measurement(
        legacy_edge.baseline.cell_id,
        1.0,
        0.25 + 5.0e-9,
    )

    attach_direct_agreements(
        candidate,
        union,
        {
            edge.baseline.cell_id: topology,
            legacy_edge.baseline.cell_id: legacy,
        },
    )

    assert topology["matrix_element"] == union["matrix_element"]
    assert union["status"] == ResultStatus.VALIDATION_FAILED.value
    assert (
        next(
            record
            for record in union["validation"][DIRECT_AGREEMENT_FIELD]
            if record["edge_kind"] == LC_CROSS_LAYOUT_COMPONENT
        )["value_kind"]
        == LC_COMMON_COMPONENT_FIELD
    )


def test_final_replay_checks_cross_layout_component_not_only_totals() -> None:
    candidate = _cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    edge = incoming_agreement_edges(candidate)[0]
    topology = _measurement(edge.baseline.cell_id, 1.0, 0.25)
    union = _measurement(candidate.cell_id, 1.0, 0.25)
    replayed = {
        edge.baseline.cell_id: _ReplayObservation(1.0, 0.0, 0.0, 0.25),
        candidate.cell_id: _ReplayObservation(
            1.0,
            0.0,
            0.0,
            0.25 + 5.0e-9,
        ),
    }

    with pytest.raises(FinalAuditError, match="direct-edge replay failed"):
        _audit_replayed_direct_agreements(
            (edge,),
            {
                edge.baseline.cell_id: topology,
                candidate.cell_id: union,
            },
            replayed,
        )


def test_legacy_pyamplicol_component_uses_independent_tolerance() -> None:
    candidate = _cell(
        "matrix_compiled_builtin_sm_lc",
        workload=Workload.ALL_FLOW,
    )
    edges = incoming_agreement_edges(candidate)
    layout = next(
        edge for edge in edges if edge.kind == LC_CROSS_LAYOUT_COMPONENT
    )
    legacy = next(
        edge for edge in edges if edge.kind == LC_LEGACY_PYAMPLICOL_COMPONENT
    )
    observed = 1.0 + 5.0e-9
    measurement = _measurement(candidate.cell_id, 1.0, observed)

    attach_direct_agreements(
        candidate,
        measurement,
        {
            layout.baseline.cell_id: _measurement(
                layout.baseline.cell_id,
                1.0,
                observed,
            ),
            legacy.baseline.cell_id: _measurement(
                legacy.baseline.cell_id,
                1.0,
                1.0,
            ),
        },
    )

    record = next(
        item
        for item in measurement["validation"][DIRECT_AGREEMENT_FIELD]
        if item["edge_kind"] == LC_LEGACY_PYAMPLICOL_COMPONENT
    )
    assert measurement["status"] == ResultStatus.OK.value
    assert record["relative_tolerance"] == 1.0e-8
    assert record["absolute_tolerance"] == 1.0e-15


def test_replay_categories_are_explicit_and_missing_endpoints_fail_closed() -> None:
    edges = agreement_edges(maximum_n_final=1)
    pac_pac = next(
        edge
        for edge in edges
        if edge.baseline.measurement.execution_mode is not ExecutionMode.AMPLICOL
        and edge.candidate.measurement.execution_mode is not ExecutionMode.AMPLICOL
    )
    pac_legacy = next(
        edge
        for edge in edges
        if edge.kind == LC_LEGACY_PYAMPLICOL_COMPONENT
    )
    legacy_layout = next(
        edge
        for edge in edges
        if edge.kind == LC_CROSS_LAYOUT_COMPONENT
        and edge.candidate.measurement.execution_mode is ExecutionMode.AMPLICOL
    )
    selected = (pac_pac, pac_legacy, legacy_layout)
    cells = {
        endpoint.cell_id: endpoint
        for edge in selected
        for endpoint in (edge.baseline, edge.candidate)
    }
    measurements = {
        cell_id: _measurement(cell_id, 1.0, 1.0)
        for cell_id in cells
    }
    replayed = {
        cell_id: _ReplayObservation(1.0, 0.0, 0.0, 1.0)
        for cell_id, cell in cells.items()
        if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
    }

    counts = _audit_replayed_direct_agreements(
        selected,
        measurements,
        replayed,
    )

    assert sorted(counts.values()) == [1, 1, 1]
    missing = dict(replayed)
    missing.pop(pac_pac.baseline.cell_id)
    with pytest.raises(FinalAuditError, match="missing its required pyAmpliCol replay"):
        _audit_replayed_direct_agreements(
            selected,
            measurements,
            missing,
        )


def _cache_measurement(cell_id: str) -> dict[str, object]:
    compact = _measurement(cell_id, 1.0, 1.0)
    result = empty_measurement()
    result.update(
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
            "selector_contract": compact["selector_contract"],
            "validation": compact["validation"],
            "resources": {},
            "provenance": {},
        }
    )
    return result


def test_cache_admission_requires_agreement_and_lc_component_evidence() -> None:
    cell = _cell(
        "reference_amplicol_lc",
        workload=Workload.SELECTED_FLOW,
    )
    measurement = _cache_measurement(cell.cell_id)
    cache = build_reset_cache(cell.dataset_id, (cell,))
    cache["entries"][0]["measurement"] = measurement
    validate_cache(cache, expected_cells=(cell,))

    validation = measurement["validation"]
    assert isinstance(validation, dict)
    direct = validation.pop(DIRECT_AGREEMENT_FIELD)
    with pytest.raises(ValueError, match=DIRECT_AGREEMENT_FIELD):
        validate_cache(cache, expected_cells=(cell,))
    validation[DIRECT_AGREEMENT_FIELD] = direct
    validation.pop(LC_COMMON_COMPONENT_FIELD)
    with pytest.raises(ValueError, match=LC_COMMON_COMPONENT_FIELD):
        validate_cache(cache, expected_cells=(cell,))


def test_campaign_plan_schedules_every_direct_peer_before_z_union(
    tmp_path: Path,
) -> None:
    candidate = _cell(
        "z_external_sm",
        workload=Workload.ALL_FLOW,
        variant="jit_o1",
    )
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )

    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=CampaignSettings(artifact_policy=ArtifactPolicy.REGENERATE),
    )
    by_id = {item.cell.cell_id: item for item in planned}
    target = by_id[candidate.cell_id]

    assert set(target.comparison_peer_ids) == {
        edge.baseline.cell_id for edge in incoming_agreement_edges(candidate)
    }
    assert all(
        peer_id in by_id and by_id[peer_id].rank < target.rank
        for peer_id in target.comparison_peer_ids
    )
    assert any(
        item.cell.variant == "recurrence_jit_o2"
        and item.cell.dataset_id == "z_external_sm"
        for item in planned
    )


def test_canonical_n4_plan_keeps_bounded_acyclic_dependency_depth(
    tmp_path: Path,
) -> None:
    selected = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.n_final <= 4
    )
    planned = plan_campaign(
        selected,
        store=ArtifactStore(
            artifact_root=tmp_path / "artifacts",
            lock_root=tmp_path / "locks",
        ),
        settings=CampaignSettings(artifact_policy=ArtifactPolicy.REGENERATE),
    )
    by_id = {item.cell.cell_id: item for item in planned}

    assert len(planned) == len(by_id) == 742
    assert max(item.rank for item in planned) == 4
    for item in planned:
        dependency_ids = set(item.comparison_peer_ids)
        if item.baseline_cell_id is not None:
            dependency_ids.add(item.baseline_cell_id)
        assert all(
            dependency_id in by_id
            and by_id[dependency_id].rank < item.rank
            for dependency_id in dependency_ids
        )
