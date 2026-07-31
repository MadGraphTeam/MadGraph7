# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer.catalog_structural_parity_audit import (
    CatalogParityError,
    WorkCounts,
    _candidate_counts,
    _legacy_counts,
    _parity_exit_code,
    audit_source_manifest,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode, Workload


def _module(path: Path, currents: int, interactions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_calls = "\n".join(
        f"call ext_scalar(p,val_c(1,{current}))"
        for current in range(1, currents)
    )
    interaction_ids = ",".join(str(value) for value in range(1, interactions + 1))
    path.write_text(
        "subroutine evaluate(p,val_c,int_c)\n"
        f"complex(kind=8),dimension(1:6,{currents}) :: val_c\n"
        f"complex(kind=8),dimension(1:6,{interactions}) :: int_c\n"
        "end subroutine evaluate\n"
        "subroutine compute_external_currents(p,val_c)\n"
        f"{source_calls}\n"
        "end subroutine compute_external_currents\n"
        "subroutine vertex_type_fixture(p,val_c,int_c)\n"
        f"integer,parameter,dimension({interactions}) :: int1=[{interaction_ids}]\n"
        "end subroutine vertex_type_fixture\n"
        "subroutine combine_currents_fixture(p,val_c,int_c)\n"
        f"integer,parameter,dimension(0:{interactions},1) :: int1="
        f"reshape([{currents},{interaction_ids}],"
        f"shape=[{interactions + 1},1])\n"
        "end subroutine combine_currents_fixture\n"
    )


def test_selected_flow_uses_exact_process_row_module(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _module(
        artifact / "selected-flow-generated-library/Library/amp1_1_lib.f03",
        31,
        37,
    )
    _module(
        artifact / "selected-flow-generated-library/Library/amp2_1_lib.f03",
        41,
        47,
    )
    counts = _legacy_counts(
        {
            "artifact": {
                "path": str(artifact),
                "process_row": "group:2:integral:1",
            }
        },
        workload=Workload.SELECTED_FLOW,
    )
    assert counts.active is not None
    assert counts.active.current_count == 41
    assert counts.active.evaluation_count == 47
    assert counts.active.attachment_count == 47
    assert counts.static.current_count == 31 + 41
    assert counts.static.evaluation_count == 37 + 47
    assert counts.static.attachment_count == 37 + 47
    assert counts.selected_module == "amp2_1_lib.f03"
    assert counts.selected_module_object_mapping is not None
    assert counts.selected_module_object_mapping.current_ids_complete is True
    assert counts.selected_module_object_mapping.interaction_ids_complete is True


def test_contracted_without_call_histogram_keeps_active_work_unresolved(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    for index in range(1, 7):
        _module(
            artifact / f"contracted-generated-library/Library/amp{index}_1_lib.f03",
            406,
            2440,
        )
    probe = (
        artifact
        / "contracted-generated-library"
        / "amplicol_color_library_probe.output"
    )
    probe.write_text(
        "Total number of currents, vertices and amplitudes after filter "
        f"{720 * 406} {720 * 2440} 720\n"
    )
    counts = _legacy_counts(
        {
            "artifact": {
                "path": str(artifact),
                "process_row": "group:1:integral:1",
            }
        },
        workload=Workload.CONTRACTED,
    )
    assert counts.static.current_count == 6 * 406
    assert counts.static.evaluation_count == 6 * 2440
    assert counts.static.attachment_count == 6 * 2440
    assert counts.active is None
    assert counts.limitation is not None


def test_contracted_uses_exact_call_histogram_for_nonuniform_modules(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _module(
        artifact / "contracted-generated-library/Library/amp1_1_lib.f03",
        10,
        20,
    )
    _module(
        artifact / "contracted-generated-library/Library/amp2_1_lib.f03",
        30,
        40,
    )
    probe = (
        artifact
        / "contracted-generated-library"
        / "amplicol_color_library_probe.output"
    )
    probe.write_text(
        "Total number of currents, vertices and amplitudes after filter 123 456 7\n"
        "AMPICOL_COLOR_PROBE_LIBRARY_CALLS 1 1 2\n"
        "AMPICOL_COLOR_PROBE_LIBRARY_CALLS 2 1 5\n"
    )
    counts = _legacy_counts(
        {
            "artifact": {
                "path": str(artifact),
                "process_row": "group:2:integral:1",
            }
        },
        workload=Workload.CONTRACTED,
    )
    assert counts.active == WorkCounts(
        2 * 10 + 5 * 30,
        2 * 20 + 5 * 40,
        2 * 20 + 5 * 40,
    )
    assert counts.static == WorkCounts(40, 60, 60)
    assert counts.limitation is None


def test_recurrence_selector_certificate_separates_active_from_persisted(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    process = artifact / "processes/p/execution.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-runtime-recurrence-execution",
                "plan": {
                    "inspection_summary": {
                        "schedule": {
                            "current_count": 100,
                            "contribution_count": 200,
                        },
                        "construction": {
                            "peak_current_count": 120,
                            "peak_contribution_count": 240,
                            "peak_contribution_count_semantics": (
                                "resident-pending-contributions-v1"
                            ),
                        },
                        "selector_work_certificate": {
                            "representatives": [
                                {
                                    "current_count": 30,
                                    "contribution_count": 40,
                                },
                                {
                                    "current_count": 35,
                                    "contribution_count": 45,
                                },
                            ]
                        },
                    }
                },
            }
        )
    )
    counts = _candidate_counts(
        {
            "artifact": {"path": str(artifact)},
            "provenance": {"report_measured_source_revision": "abc"},
        },
        workload=Workload.SELECTED_FLOW,
    )
    assert counts.active.current_count == 35
    assert counts.active.evaluation_count == 45
    assert counts.active.attachment_count == 45
    assert counts.final_materialized.current_count == 100
    assert counts.peak_materialized.evaluation_count == 240
    assert counts.peak_materialized.attachment_count == 240
    assert counts.selector_certificate_available is True


def test_recurrence_peak_requires_resident_semantics_marker(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    process = artifact / "processes/p/execution.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-runtime-recurrence-execution",
                "plan": {
                    "inspection_summary": {
                        "schedule": {
                            "current_count": 100,
                            "contribution_count": 200,
                        },
                        "construction": {
                            "peak_current_count": 120,
                            "peak_contribution_count": 240,
                        },
                    }
                },
            }
        )
    )

    with pytest.raises(
        CatalogParityError,
        match="authenticated resident peak-contribution semantics",
    ):
        _candidate_counts(
            {
                "artifact": {"path": str(artifact)},
                "provenance": {"report_measured_source_revision": "abc"},
            },
            workload=Workload.CONTRACTED,
        )


def test_compiled_all_flow_uses_primary_fixed_helicity_dag(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    process = artifact / "processes/p/execution.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-runtime-execution",
                "dag_summary": {
                    "current_count": 31,
                    "interaction_count": 47,
                    "interaction_evaluation_count": 29,
                },
                "helicity_selector_executions": [
                    {
                        "execution": {
                            "dag_summary": {
                                "current_count": 19,
                                "interaction_count": 23,
                                "interaction_evaluation_count": 17,
                            }
                        }
                    }
                ],
            }
        )
    )
    counts = _candidate_counts(
        {"artifact": {"path": str(artifact)}},
        workload=Workload.ALL_FLOW,
    )
    assert counts.active.current_count == 19
    assert counts.active.evaluation_count == 17
    assert counts.active.attachment_count == 23
    assert counts.final_materialized == WorkCounts(50, 46, 70)
    assert counts.active_evidence_kind == ("compiled-fixed-helicity-worst-class")
    assert counts.selector_certificate_available is True


def test_compiled_selected_flow_uses_worst_color_sector(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    process = artifact / "processes/p/execution.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-runtime-execution",
                "dag_summary": {
                    "current_count": 11,
                    "interaction_count": 13,
                    "interaction_evaluation_count": 7,
                },
                "helicity_sum_execution": {
                    "dag_summary": {
                        "current_count": 101,
                        "interaction_count": 211,
                        "interaction_evaluation_count": 151,
                    },
                    "color_selector_executions": [
                        {
                            "execution": {
                                "dag_summary": {
                                    "current_count": 31,
                                    "interaction_count": 41,
                                    "interaction_evaluation_count": 29,
                                }
                            }
                        },
                        {
                            "execution": {
                                "dag_summary": {
                                    "current_count": 37,
                                    "interaction_count": 43,
                                    "interaction_evaluation_count": 31,
                                }
                            }
                        },
                    ],
                },
            }
        )
    )
    counts = _candidate_counts(
        {"artifact": {"path": str(artifact)}},
        workload=Workload.SELECTED_FLOW,
    )
    assert counts.active == WorkCounts(37, 31, 43)
    assert counts.final_materialized == WorkCounts(180, 218, 308)
    assert counts.active_evidence_kind == ("compiled-selected-color-worst-sector")
    assert counts.selector_certificate_available is True


def test_eager_selector_census_separates_active_from_materialized(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    process = artifact / "processes/p/execution.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-runtime-eager-execution",
                "plan": {
                    "inspection_summary": {
                        "current_count": 101,
                        "invocation_count": 151,
                        "attachment_count": 211,
                        "selector_work": {
                            "abi": "pyamplicol-eager-selector-work-v1",
                            "selected_flow_current_count": 37,
                            "selected_flow_evaluation_count": 31,
                            "selected_flow_attachment_count": 43,
                            "all_flow_current_count": 41,
                            "all_flow_evaluation_count": 35,
                            "all_flow_attachment_count": 47,
                            "contracted_current_count": 101,
                            "contracted_evaluation_count": 151,
                            "contracted_attachment_count": 211,
                        },
                    }
                },
            }
        )
    )
    counts = _candidate_counts(
        {"artifact": {"path": str(artifact)}},
        workload=Workload.SELECTED_FLOW,
    )
    assert counts.active == WorkCounts(37, 31, 43)
    assert counts.final_materialized == WorkCounts(101, 151, 211)
    assert counts.active_evidence_kind == "eager-selected-flow-selector-census"
    assert counts.selector_certificate_available is True


def test_complete_parity_gate_fails_closed() -> None:
    passing = {"summary": {"fully_certified_catalog_parity": True}}
    failing = {"summary": {"fully_certified_catalog_parity": False}}
    assert _parity_exit_code(passing, required=False) == 0
    assert _parity_exit_code(failing, required=False) == 0
    assert _parity_exit_code(passing, required=True) == 0
    assert _parity_exit_code(failing, required=True) == 2
    assert _parity_exit_code({}, required=True) == 2


def test_source_preflight_requires_exact_all_catalog_coverage() -> None:
    revision = "a" * 40
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode
        in {ExecutionMode.RECURRENCE, ExecutionMode.COMPILED, ExecutionMode.EAGER}
    )
    assert len(cells) == 1184
    counts = {
        "current_count": 100,
        "evaluation_count": 200,
        "attachment_count": 300,
    }
    rows = []
    for cell in cells:
        candidate = {
            "active": counts,
            "final_materialized": counts,
            "peak_materialized": counts,
            "active_closure_count": 4,
            "final_closure_count": 4,
            "peak_closure_count": 4,
            "dynamic_color_projection_applied": False,
            "dynamic_color_projection": None,
        }
        row = {
            "cell_id": cell.cell_id,
            "source_revision": revision,
            "proof_strength": "exact-source-construction",
            "candidate": candidate,
        }
        if REPORT_CATALOG.legacy_reference_available(cell):
            row |= {
                "status": "ok",
                "legacy": {
                    "active": counts,
                    "static": counts,
                },
            }
        else:
            row |= {
                "status": "legacy-scope-unavailable",
                "scope_reason": "original AmpliCol supports at most three quark lines",
                "legacy": None,
            }
        rows.append(row)
    payload = audit_source_manifest(
        {
            "schema": "pyamplicol-catalog-source-structural-evidence-v1",
            "source_revision": revision,
            "cells": rows,
        }
    )

    assert payload["summary"]["catalog_candidate_cell_count"] == 1184
    assert payload["summary"]["complete_catalog_coverage"] is True
    assert payload["summary"]["fully_certified_catalog_parity"] is True
