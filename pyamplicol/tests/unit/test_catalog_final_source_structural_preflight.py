from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer.catalog_final_source_structural_preflight import (
    FinalSourceProducerError,
    _counts,
)
from tools.developer.emit_candidate_structural_preflight import _numerical_proof


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
