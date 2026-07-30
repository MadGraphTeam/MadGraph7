# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer.catalog_final_source_structural_preflight import (
    FinalSourceProducerError,
    _counts,
    _legacy_record,
)
from tools.developer.emit_candidate_structural_preflight import _numerical_proof
from tools.performance_report.catalog import REPORT_CATALOG


def test_structural_counts_do_not_accept_summary_placeholders() -> None:
    with pytest.raises(FinalSourceProducerError, match="non-negative integer"):
        _counts(
            {
                "source_current_count": 1,
                "produced_current_count": 2,
                "kernel_evaluation_count": None,
                "attachment_count": 4,
                "amplitude_destination_count": 1,
            },
            "candidate.active",
        )


def test_opaque_precision50_digest_is_not_accepted_as_truth(tmp_path: Path) -> None:
    revision = "a" * 40
    path = tmp_path / "precision50.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "source_revision": revision,
                "precision_decimal_digits": 50,
                "comparison_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(
        FinalSourceProducerError,
        match="recomputable independent precision>=50 witness",
    ):
        _numerical_proof(path, revision)


def test_legacy_proof_is_bound_to_exact_reference_cell(tmp_path: Path) -> None:
    reference = next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if REPORT_CATALOG.legacy_reference_available(cell)
    )
    (tmp_path / "legacy-structural-proof.json").write_text(
        json.dumps(
            {
                "schema": "pyamplicol-legacy-final-structural-proof-v1",
                "cell_id": f"{reference.cell_id}-wrong",
                "accuracy": reference.measurement.accuracy.value,
                "workload": reference.workload.value,
                "source_revision": "a" * 40,
            }
        )
    )
    result = {
        "status": "ok",
        "artifact": {"path": str(tmp_path)},
        "provenance": {"revision": "a" * 40},
    }
    with pytest.raises(FinalSourceProducerError, match="wrong identity"):
        _legacy_record(reference, result)
