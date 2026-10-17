# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.boundary import (
    CELL_BOUNDARY_SCHEMA,
    CellBoundaryError,
    authenticate_current_delta,
    load_cell_boundary,
    snapshot_cell_boundary,
)
from tools.performance_report.cli import _parser
from tools.performance_report.models import ArtifactPolicy


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "coordination",
    )


def _publish_worker_result(
    store: ArtifactStore,
    cell_id: str,
) -> str:
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    attempt.path("worker-result.json").write_text(
        '{"status":"ok"}\n',
        encoding="ascii",
    )
    return attempt.publish(
        {"status": "ok"},
        artifact_paths=("worker-result.json",),
    ).attempt_id


def test_boundary_accepts_exactly_one_authenticated_current_delta(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    before = snapshot_cell_boundary(store, "cell")
    attempt_id = _publish_worker_result(store, "cell")

    accepted = authenticate_current_delta(
        store,
        cell_id="cell",
        expected_attempt_id=attempt_id,
        before=before,
    )

    assert accepted["schema"] == CELL_BOUNDARY_SCHEMA
    assert accepted["attempt_ids"] == [attempt_id]
    current = accepted["current"]
    assert isinstance(current, dict)
    assert current["attempt_id"] == attempt_id
    assert current["status"] == "ok"
    assert len(str(current["manifest_sha256"])) == 64
    assert len(str(current["result_sha256"])) == 64
    assert len(str(current["worker_result_sha256"])) == 64


def test_boundary_rejects_an_extra_attempt_even_with_valid_current(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    before = snapshot_cell_boundary(store, "cell")
    failed = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    failed.mark_failed("failed")
    accepted_id = _publish_worker_result(store, "cell")

    with pytest.raises(CellBoundaryError, match="exactly"):
        authenticate_current_delta(
            store,
            cell_id="cell",
            expected_attempt_id=accepted_id,
            before=before,
        )


def test_boundary_runs_result_validation_before_acceptance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    before = snapshot_cell_boundary(store, "cell")
    attempt_id = _publish_worker_result(store, "cell")

    def reject_result(result: Mapping[str, Any]) -> None:
        assert result == {"status": "ok"}
        raise ValueError("measurement schema differs")

    with pytest.raises(ValueError, match="measurement schema"):
        authenticate_current_delta(
            store,
            cell_id="cell",
            expected_attempt_id=attempt_id,
            before=before,
            validate_result=reject_result,
        )


def test_boundary_rejects_current_without_authenticated_worker_result(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    before = snapshot_cell_boundary(store, "cell")
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    attempt.publish({"status": "ok"})

    with pytest.raises(
        CellBoundaryError,
        match=r"worker-result\.json",
    ):
        authenticate_current_delta(
            store,
            cell_id="cell",
            expected_attempt_id=attempt.attempt_id,
            before=before,
        )


def test_boundary_loads_legacy_fast_wrapper_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "pre-populate-current.json"
    path.write_text(
        json.dumps(
            {
                "attempt_ids": [],
                "cell_root": "/legacy/cell",
                "current": None,
            }
        ),
        encoding="ascii",
    )

    assert load_cell_boundary(path)["attempt_ids"] == []


def test_boundary_commands_have_explicit_cell_inputs() -> None:
    snapshot = _parser().parse_args(
        ("snapshot-cell-boundary", "--cell-id", "cell")
    )
    assert snapshot.cell_id == "cell"

    accepted = _parser().parse_args(
        (
            "accept-cell-boundary",
            "--cell-id",
            "cell",
            "--expected-attempt-id",
            "attempt",
            "--before-snapshot",
            "before.json",
        )
    )
    assert accepted.cell_id == "cell"
    assert accepted.expected_attempt_id == "attempt"
    assert accepted.before_snapshot == Path("before.json")
