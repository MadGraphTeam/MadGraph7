from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.developer.catalog_final_source_structural_preflight import (
    FinalSourceProducerError,
    _authenticate_inventory,
    _counts,
)


def _work(value: int = 3) -> dict[str, int]:
    return {
        "source_current_count": 1,
        "produced_current_count": value - 1,
        "kernel_evaluation_count": value,
        "attachment_count": value + 1,
        "amplitude_destination_count": 1,
    }


def test_inventory_authenticates_real_payload_bytes(tmp_path: Path) -> None:
    payload = tmp_path / "schedule.bin"
    payload.write_bytes(b"exact structural payload")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    inventory = _authenticate_inventory(
        tmp_path,
        {
            "status": "complete",
            "objects": [
                {
                    "object_id": "schedule",
                    "path": "schedule.bin",
                    "content_sha256": digest,
                    "counts": _work(),
                }
            ],
            "roles": [{"role": "primary", "object_id": "schedule"}],
        },
        label="candidate",
    )

    assert inventory["status"] == "complete"
    assert inventory["objects"][0]["content_sha256"] == digest
    assert len(inventory["inventory_sha256"]) == 64


def test_inventory_rejects_changed_payload(tmp_path: Path) -> None:
    payload = tmp_path / "schedule.bin"
    payload.write_bytes(b"changed")
    with pytest.raises(FinalSourceProducerError, match="absent or changed"):
        _authenticate_inventory(
            tmp_path,
            {
                "status": "complete",
                "objects": [
                    {
                        "object_id": "schedule",
                        "path": "schedule.bin",
                        "content_sha256": "0" * 64,
                        "counts": _work(),
                    }
                ],
                "roles": [{"role": "primary", "object_id": "schedule"}],
            },
            label="candidate",
        )


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
