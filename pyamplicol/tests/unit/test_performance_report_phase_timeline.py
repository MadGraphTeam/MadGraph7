# SPDX-License-Identifier: 0BSD
"""Phase-timeline evidence must remain direct, typed, and reusable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.performance_report.phase_timeline import (
    PHASE_TIMELINE_SCHEMA,
    build_phase_timeline,
)


def _rows(timeline: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = timeline["entries"]
    assert isinstance(entries, list)
    return {str(entry["key"]): entry for entry in entries if isinstance(entry, dict)}


def test_pyamplicol_timeline_keeps_only_direct_phase_clocks(tmp_path: Path) -> None:
    progress = tmp_path / "worker-progress.jsonl"
    events = (
        {
            "event": "start",
            "task_id": "generation:validation",
            "parent_task_id": "generation",
            "description": "Validating generated artifact",
        },
        {
            "event": "end",
            "task_id": "generation:validation",
            "success": True,
            "elapsed_seconds": 0.125,
        },
    )
    progress.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    result = {
        "status": "ok",
        "generation_seconds": 4.5,
        "validation": {"status": "ok"},
        "resources": {
            "wall_seconds": 9.0,
            "cpu_seconds": 7.0,
            "peak_rss_bytes": 123,
            "peak_guard_bytes": 456,
        },
        "provenance": {
            "model_preparation_seconds": 0.25,
            "model_preparation_reused": False,
            "runtime_profile": {
                "warmup_elapsed_seconds": 0.5,
                "calibration_outer_elapsed_seconds": 0.75,
                "measurement_phase_elapsed_seconds": 2.0,
                "achieved_runtime_seconds": 1.8,
                "profile_attribution_elapsed_seconds": 1.0,
                "profile_total_elapsed_seconds": 4.25,
                "completed_sample_count": 12,
                "calibration_block_count": 3,
            },
            "manual_campaign": {
                "cell_identity": {"execution_mode": "recurrence"},
            },
        },
    }

    timeline = build_phase_timeline(result, progress_path=progress)
    rows = _rows(timeline)

    assert timeline["schema"] == PHASE_TIMELINE_SCHEMA
    assert rows["generation"]["seconds"] == pytest.approx(4.5)
    assert rows["generation-step:validation"]["seconds"] == pytest.approx(0.125)
    assert rows["generation-step:validation"]["included_in"] == "generation"
    assert rows["runtime-warmup"]["seconds"] == pytest.approx(0.5)
    assert rows["runtime-calibration"]["detail"] == "3 blocks"
    assert rows["runtime-measurement"]["seconds"] == pytest.approx(1.8)
    assert rows["runtime-measurement"]["detail"] == "12 samples"
    assert rows["runtime-attribution"]["seconds"] == pytest.approx(1.0)
    assert rows["runtime-profile-total"]["seconds"] == pytest.approx(4.25)
    assert rows["attempt-publication"]["seconds"] is None
    assert "not separately timed" in str(rows["attempt-publication"]["detail"])
    assert timeline["total_worker_wall_seconds"] == pytest.approx(9.0)
    assert timeline["peak_guard_bytes"] == 456


def test_legacy_timeline_marks_calibration_not_applicable() -> None:
    result = {
        "status": "ok",
        "generation_seconds": 2.0,
        "validation": None,
        "resources": {},
        "provenance": {
            "runtime_profile": {
                "warmup": {"elapsed_seconds": 0.2},
                "measurement": {
                    "achieved_runtime_seconds": 5.1,
                    "chunk_count": 8,
                    "chunks": [
                        {"elapsed_seconds": 2.6},
                        {"elapsed_seconds": 2.7},
                    ],
                },
            },
            "manual_campaign": {
                "cell_identity": {"execution_mode": "amplicol"},
            },
        },
    }

    rows = _rows(build_phase_timeline(result))

    assert rows["runtime-warmup"]["seconds"] == pytest.approx(0.2)
    assert rows["runtime-calibration"]["status"] == "not_applicable"
    assert rows["runtime-calibration"]["seconds"] is None
    assert rows["runtime-measurement"]["seconds"] == pytest.approx(5.3)
    assert rows["runtime-measurement"]["detail"] == "8 chunks"


def test_missing_phase_clocks_remain_unavailable() -> None:
    result = {
        "status": "ok",
        "generation_seconds": 1.0,
        "resources": {"wall_seconds": 10.0},
        "provenance": {
            "runtime_profile": {"achieved_runtime_seconds": 3.0},
            "manual_campaign": {
                "cell_identity": {"execution_mode": "compiled"},
            },
        },
    }

    rows = _rows(build_phase_timeline(result))

    assert rows["runtime-warmup"]["seconds"] is None
    assert rows["runtime-calibration"]["seconds"] is None
    assert rows["runtime-attribution"]["seconds"] is None
    assert rows["runtime-profile-total"]["seconds"] is None
    assert rows["runtime-measurement"]["seconds"] == pytest.approx(3.0)
