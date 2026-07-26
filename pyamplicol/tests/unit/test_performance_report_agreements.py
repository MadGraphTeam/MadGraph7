# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tools.performance_report.agreements import (
    BUILTIN_UFO_RECURRENCE,
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    LC_CROSS_LAYOUT_COMPONENT,
    Z_RECURRENCE_CROSS_MODE,
    agreement_edges,
    attach_direct_agreements,
    incoming_agreement_edges,
)
from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.final_audit import (
    FinalAuditError,
    _audit_replayed_direct_agreements,
    _ReplayObservation,
)
from tools.performance_report.models import (
    ArtifactPolicy,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.runner import pointwise_validation
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


def test_canonical_n4_direct_agreement_graph_has_exact_locked_counts() -> None:
    counts = Counter(
        edge.kind for edge in agreement_edges(maximum_n_final=4)
    )

    assert counts == {
        BUILTIN_UFO_RECURRENCE: 136,
        Z_RECURRENCE_CROSS_MODE: 80,
        LC_CROSS_LAYOUT_COMPONENT: 208,
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
    edge = incoming_agreement_edges(candidate)[0]
    assert edge.kind == LC_CROSS_LAYOUT_COMPONENT
    topology = _measurement(edge.baseline.cell_id, 1.0, 0.25)
    union = _measurement(candidate.cell_id, 1.0, 0.25 + 5.0e-9)

    attach_direct_agreements(
        candidate,
        union,
        {edge.baseline.cell_id: topology},
    )

    assert topology["matrix_element"] == union["matrix_element"]
    assert union["status"] == ResultStatus.VALIDATION_FAILED.value
    assert (
        union["validation"][DIRECT_AGREEMENT_FIELD][0]["value_kind"]
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
