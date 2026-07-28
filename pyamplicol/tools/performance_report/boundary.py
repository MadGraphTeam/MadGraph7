# SPDX-License-Identifier: 0BSD
"""Authoritative per-cell boundaries for asynchronous report publication."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, ArtifactStoreError, CurrentRecord

CELL_BOUNDARY_SCHEMA = "pyamplicol-performance-cell-boundary-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CellBoundaryError(ArtifactStoreError):
    """A fast-controller cell boundary was not authoritative."""


def _artifact_digest(record: CurrentRecord, relative_path: str) -> str:
    matches = [
        artifact.sha256
        for artifact in record.artifacts
        if artifact.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise CellBoundaryError(
            f"current attempt does not authenticate exactly one {relative_path}"
        )
    return matches[0]


def _current_identity(record: CurrentRecord) -> dict[str, object]:
    return {
        "attempt_id": record.attempt_id,
        "manifest_path": str(record.manifest_path),
        "manifest_sha256": record.manifest_sha256,
        "result_path": str(record.result_path),
        "result_sha256": next(
            artifact.sha256
            for artifact in record.artifacts
            if artifact.path == record.result_path
        ),
        "status": record.result.get("status"),
        "worker_result_sha256": _artifact_digest(
            record,
            "worker-result.json",
        ),
    }


def snapshot_cell_boundary(
    store: ArtifactStore,
    cell_id: str,
) -> dict[str, object]:
    """Snapshot one locked attempt inventory and authenticated current."""

    with store.cell_lock(cell_id):
        return _snapshot_cell_boundary_unlocked(store, cell_id)


def _snapshot_cell_boundary_unlocked(
    store: ArtifactStore,
    cell_id: str,
) -> dict[str, object]:
    attempt_ids = store.cell_attempt_ids(cell_id)
    current = store.load_current(cell_id, missing_ok=True)
    return {
        "schema": CELL_BOUNDARY_SCHEMA,
        "cell_id": cell_id,
        "attempt_ids": list(attempt_ids),
        "current": None if current is None else _current_identity(current),
    }


def load_cell_boundary(path: Path) -> Mapping[str, Any]:
    """Load a new snapshot or the legacy fast-wrapper snapshot shape."""

    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CellBoundaryError(
            f"cannot read pre-populate cell boundary {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CellBoundaryError("pre-populate cell boundary must be an object")
    return payload


def authenticate_current_delta(
    store: ArtifactStore,
    *,
    cell_id: str,
    expected_attempt_id: str,
    before: Mapping[str, Any],
    validate_result: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Require exactly one new immutable attempt and its successful current."""

    before_schema = before.get("schema")
    if before_schema not in {None, CELL_BOUNDARY_SCHEMA}:
        raise CellBoundaryError("pre-populate cell boundary schema differs")
    before_cell = before.get("cell_id")
    if before_cell not in {None, cell_id}:
        raise CellBoundaryError("pre-populate cell boundary cell_id differs")
    before_attempts = before.get("attempt_ids")
    if (
        not isinstance(before_attempts, list)
        or not all(isinstance(value, str) for value in before_attempts)
        or len(set(before_attempts)) != len(before_attempts)
    ):
        raise CellBoundaryError(
            "pre-populate cell boundary has an invalid attempt inventory"
        )
    if before.get("current") is not None:
        raise CellBoundaryError(
            "accepted missing cell had a current before populate"
        )

    with store.cell_lock(cell_id):
        after = _snapshot_cell_boundary_unlocked(store, cell_id)
        after_attempts = after["attempt_ids"]
        assert isinstance(after_attempts, list)
        added = sorted(set(after_attempts) - set(before_attempts))
        removed = sorted(set(before_attempts) - set(after_attempts))
        if (
            added != [expected_attempt_id]
            or removed
            or len(after_attempts) != len(before_attempts) + 1
        ):
            raise CellBoundaryError(
                "populate did not add exactly its reported immutable attempt"
            )
        current = after["current"]
        if not isinstance(current, dict):
            raise CellBoundaryError("accepted cell has no authenticated current")
        if (
            current.get("attempt_id") != expected_attempt_id
            or current.get("status") != "ok"
        ):
            raise CellBoundaryError(
                "accepted current identity or result status differs"
            )
        for field in (
            "manifest_sha256",
            "result_sha256",
            "worker_result_sha256",
        ):
            value = current.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise CellBoundaryError(
                    f"accepted current lacks authenticated {field}"
                )
        record = store.load_current(cell_id)
        assert record is not None
        if validate_result is not None:
            validate_result(record.result)
        return after


__all__ = [
    "CELL_BOUNDARY_SCHEMA",
    "CellBoundaryError",
    "authenticate_current_delta",
    "load_cell_boundary",
    "snapshot_cell_boundary",
]
