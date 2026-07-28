from __future__ import annotations

import json
from pathlib import Path

from tools.developer.catalog_structural_parity_audit import (
    _candidate_counts,
    _legacy_counts,
    _parity_exit_code,
)
from tools.performance_report.models import Workload


def _module(path: Path, currents: int, interactions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"complex(kind=8),dimension(1:6,{currents}) :: val_c\n"
        f"complex(kind=8),dimension(1:6,{interactions}) :: int_c\n"
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
    assert counts.active.interaction_count == 47
    assert counts.static.current_count == 31 + 41
    assert counts.static.interaction_count == 37 + 47
    assert counts.selected_module == "amp2_1_lib.f03"


def test_contracted_separates_static_templates_from_dynamic_replay(
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
        "Total number of currents, vertices and amplitudes after filter 1597 4260 720\n"
    )
    counts = _legacy_counts(
        {"artifact": {"path": str(artifact)}},
        workload=Workload.CONTRACTED,
    )
    assert counts.static.current_count == 6 * 406
    assert counts.static.interaction_count == 6 * 2440
    assert counts.active is not None
    assert counts.active.current_count == 720 * 406
    assert counts.active.interaction_count == 720 * 2440


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
    assert counts.active.interaction_count == 45
    assert counts.final_materialized.current_count == 100
    assert counts.peak_materialized.interaction_count == 240
    assert counts.selector_certificate_available is True


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
                },
            }
        )
    )
    counts = _candidate_counts(
        {"artifact": {"path": str(artifact)}},
        workload=Workload.ALL_FLOW,
    )
    assert counts.active.current_count == 31
    assert counts.active.interaction_count == 47
    assert counts.active_evidence_kind == ("compiled-primary-fixed-helicity-all-flow")


def test_complete_parity_gate_fails_closed() -> None:
    passing = {"summary": {"fully_certified_catalog_parity": True}}
    failing = {"summary": {"fully_certified_catalog_parity": False}}
    assert _parity_exit_code(passing, required=False) == 0
    assert _parity_exit_code(failing, required=False) == 0
    assert _parity_exit_code(passing, required=True) == 0
    assert _parity_exit_code(failing, required=True) == 2
    assert _parity_exit_code({}, required=True) == 2
