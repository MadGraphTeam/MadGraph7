from __future__ import annotations

from copy import deepcopy

from tools.developer.catalog_restart_parity_gate import SCHEMA, validate_manifest
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode

REVISION = "1" * 40
DIGEST = "2" * 64


def _counts(value: int = 10) -> dict[str, int]:
    return {
        "source_current_count": 2,
        "produced_current_count": value - 2,
        "kernel_evaluation_count": value,
        "attachment_count": value,
        "amplitude_destination_count": 2,
    }


def _candidate() -> dict[str, object]:
    return {
        "source_revision": REVISION,
        "active": _counts(),
        "final_materialized": _counts(),
        "peak_materialized": _counts(),
        "semantic_proof": {
            "status": "proven",
            "strength": "exact-symbolic",
            "source_revision": REVISION,
            "current_member_map_sha256": DIGEST,
            "interaction_row_map_sha256": DIGEST,
            "closure_map_sha256": DIGEST,
            "source_contract_sha256": DIGEST,
        },
        "numerical_validation": {
            "status": "ok",
            "precision_decimal_digits": 50,
            "comparison_sha256": DIGEST,
        },
        "persisted_lane_inventory": {
            "status": "complete",
            "inventory_sha256": DIGEST,
            "objects": [
                {
                    "object_id": "plan-0",
                    "content_sha256": DIGEST,
                    "counts": _counts(),
                }
            ],
            "roles": [{"role": "primary", "object_id": "plan-0"}],
        },
    }


def _legacy(cell: object) -> dict[str, object]:
    if not REPORT_CATALOG.legacy_reference_available(cell):
        return {
            "scope": "unavailable",
            "reason": "original-amplicol-open-quark-line-limit",
        }
    return {
        "scope": "available",
        "active": _counts(),
        "static": _counts(),
        "object_mapping": {
            "status": "exact",
            "current_object_map_sha256": DIGEST,
            "kernel_term_map_sha256": DIGEST,
            "combine_route_map_sha256": DIGEST,
        },
        "row_multiplicity": {
            "status": "exact",
            "histogram_sha256": DIGEST,
            "call_count": 1,
        },
    }


def _manifest() -> dict[str, object]:
    cells = []
    for cell in REPORT_CATALOG.matrix_cells():
        if cell.measurement.execution_mode not in {
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        }:
            continue
        comparable = REPORT_CATALOG.legacy_reference_available(cell)
        cells.append(
            {
                "cell_id": cell.cell_id,
                "process_key": cell.process_key,
                "n_final": cell.n_final,
                "mode": cell.measurement.execution_mode.value,
                "model": cell.measurement.model.value,
                "accuracy": cell.measurement.accuracy.value,
                "workload": cell.workload.value,
                "classification": (
                    "certified-parity"
                    if comparable
                    else "legacy-scope-unavailable"
                ),
                "candidate": _candidate(),
                "legacy": _legacy(cell),
            }
        )
    return {"schema": SCHEMA, "source_revision": REVISION, "cells": cells}


def test_complete_exact_manifest_passes_every_catalog_row() -> None:
    result = validate_manifest(_manifest(), expected_revision=REVISION)
    assert result["summary"]["expected_cell_count"] == 1136
    assert result["summary"]["failed_row_count"] == 0
    assert result["summary"]["classification_counts"] == {
        "certified-parity": 1112,
        "legacy-scope-unavailable": 24,
    }
    assert result["summary"]["restart_ready"] is True
    assert len(result["rows"]) == 1136


def test_summary_boolean_cannot_replace_per_row_evidence() -> None:
    result = validate_manifest(
        {
            "schema": SCHEMA,
            "source_revision": REVISION,
            "fully_certified_catalog_parity": True,
            "cells": [],
        },
        expected_revision=REVISION,
    )
    assert result["summary"]["restart_ready"] is False
    assert result["summary"]["failure_reason_counts"] == {"missing-record": 1136}


def test_missing_and_unexpected_rows_are_listed_with_reasons() -> None:
    manifest = _manifest()
    cells = manifest["cells"]
    assert isinstance(cells, list)
    cells.pop()
    cells.append(
        {
            "cell_id": "not-in-the-report-catalog",
            "classification": "certified-parity",
        }
    )
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "missing-record": 1,
        "unexpected-record": 1,
    }


def test_numerical_only_generation_relation_fails_closed() -> None:
    manifest = _manifest()
    cells = manifest["cells"]
    assert isinstance(cells, list)
    candidate = cells[0]["candidate"]
    candidate["numerical_relation_discovery"] = {
        "used_for_generation": True,
        "reconstructed_exact": False,
    }
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "numerical-only-relation-used-for-generation": 1
    }


def test_counts_only_certificate_and_stale_source_are_rejected() -> None:
    manifest = _manifest()
    cells = manifest["cells"]
    assert isinstance(cells, list)
    candidate = cells[0]["candidate"]
    candidate["source_revision"] = "3" * 40
    candidate["semantic_proof"] = {"status": "proven"}
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "candidate-source-revision-mismatch": 1,
        "semantic-proof-incomplete": 1,
        "semantic-proof-not-exact": 1,
        "semantic-proof-source-revision-mismatch": 1,
    }


def test_static_inventory_and_ratio_are_recomputed_not_trusted() -> None:
    manifest = _manifest()
    cells = manifest["cells"]
    assert isinstance(cells, list)
    record = cells[0]
    record["producer_ratio"] = 0.01
    candidate = record["candidate"]
    candidate["final_materialized"] = _counts(20)
    result = validate_manifest(manifest, expected_revision=REVISION)
    reasons = result["rows"][0]["reason_codes"]
    assert "persisted-object-counts-do-not-reconcile" in reasons
    assert "final_materialized-logical_current-exceeds-1.05" in reasons
    assert "final_materialized-kernel_evaluation-exceeds-1.05" in reasons
    assert "final_materialized-attachment-exceeds-1.05" in reasons


def test_legacy_scope_unavailable_still_requires_candidate_self_proof() -> None:
    manifest = _manifest()
    cells = manifest["cells"]
    assert isinstance(cells, list)
    target = next(cell for cell in cells if cell["process_key"] == "dd_4q_lines")
    target["candidate"] = deepcopy(target["candidate"])
    del target["candidate"]["numerical_validation"]
    result = validate_manifest(manifest, expected_revision=REVISION)
    row = next(row for row in result["rows"] if row["cell_id"] == target["cell_id"])
    assert row["reason_codes"] == ["numerical-validation-incomplete"]
