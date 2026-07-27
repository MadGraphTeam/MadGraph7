# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from tools.performance_report.artifacts import (
    ATTEMPT_SCHEMA,
    CURRENT_SCHEMA,
    ArtifactAction,
    ArtifactStore,
    ArtifactStoreError,
    LockTimeoutError,
    ManifestValidationError,
)
from tools.performance_report.models import ArtifactPolicy


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "coordination",
    )


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _rewrite_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _rewrite_pointer_digest(pointer_path: Path, manifest_path: Path) -> None:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _rewrite_json(pointer_path, pointer)


def _try_named_lock(
    artifact_root: str,
    lock_root: str,
    queue: multiprocessing.Queue[str],
) -> None:
    store = ArtifactStore(
        artifact_root=Path(artifact_root),
        lock_root=Path(lock_root),
    )
    try:
        with store.named_lock("prepared-ufo", timeout=0.1, poll_interval=0.01):
            queue.put("acquired")
    except LockTimeoutError:
        queue.put("timeout")


def test_successful_attempt_is_uuid_qualified_and_published_atomically(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("matrix/row:1", ArtifactPolicy.REGENERATE)
    artifact = attempt.path("payload/process.bin")
    artifact.write_bytes(b"compiled process")

    current = attempt.publish(
        {"status": "ok", "wall_seconds": 1.25},
        artifact_paths=("payload/process.bin",),
    )

    assert attempt.root.name == attempt.attempt_id
    assert attempt.root.parent.name == "attempts"
    assert current.cell_id == "matrix/row:1"
    assert current.result == {"status": "ok", "wall_seconds": 1.25}
    assert {entry.relative_path for entry in current.artifacts} == {
        "payload/process.bin",
        "result.json",
    }
    pointer = json.loads(
        (attempt.root.parent.parent / "current.json").read_text(encoding="utf-8")
    )
    assert pointer["schema"] == CURRENT_SCHEMA
    assert pointer["manifest_path"] == (
        f"attempts/{attempt.attempt_id}/manifest.json"
    )
    assert not list(attempt.root.rglob("*.tmp-*"))


def test_failed_and_interrupted_attempts_cannot_replace_valid_current(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    original = first.publish({"sequence": 1})

    failed = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    failed.mark_failed("backend failed")
    interrupted = store.new_attempt("cell", ArtifactPolicy.RETIME, based_on=original)
    interrupted.mark_interrupted("worker was terminated")

    current = store.load_current("cell")
    assert current is not None
    assert current.attempt_id == original.attempt_id
    assert current.result == {"sequence": 1}
    assert json.loads(
        (failed.root / "manifest.json").read_text(encoding="utf-8")
    )["status"] == "failed"
    assert json.loads(
        (interrupted.root / "manifest.json").read_text(encoding="utf-8")
    )["status"] == "interrupted"


def test_attempt_context_records_interruption_without_masking_exception(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        store.new_attempt("cell", ArtifactPolicy.REGENERATE) as attempt,
    ):
        raise RuntimeError("deliberate")

    manifest = json.loads(
        (attempt.root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "interrupted"
    assert manifest["error"] == "RuntimeError: deliberate"
    assert store.load_current("cell", missing_ok=True) is None


def test_artifact_policy_decisions_are_explicit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.decide("cell", ArtifactPolicy.REUSE).action is ArtifactAction.GENERATE
    assert (
        store.decide("cell", ArtifactPolicy.RETIME).action
        is ArtifactAction.GENERATE
    )
    current = store.new_attempt("cell", ArtifactPolicy.REGENERATE).publish(
        {"status": "ok"}
    )

    reuse = store.decide("cell", ArtifactPolicy.REUSE)
    retime = store.decide("cell", ArtifactPolicy.RETIME)
    regenerate = store.decide("cell", ArtifactPolicy.REGENERATE)
    assert reuse.action is ArtifactAction.REUSE_CURRENT
    assert reuse.current == current
    assert not reuse.requires_timing
    assert retime.action is ArtifactAction.RETIME_CURRENT
    assert retime.current == current
    assert retime.requires_timing and not retime.requires_generation
    assert regenerate.action is ArtifactAction.GENERATE
    assert regenerate.requires_generation


def test_recovery_enumerates_only_validated_current_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    expected = {
        store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
        .publish({"cell": cell_id})
        .attempt_id
        for cell_id in ("cell-a", "cell-b")
    }
    store.new_attempt("failed-cell", ArtifactPolicy.REGENERATE).mark_failed(
        "not publishable"
    )

    recovered = store.recover_current_records()
    assert {record.attempt_id for record in recovered} == expected
    assert {record.cell_id for record in recovered} == {"cell-a", "cell-b"}


def test_recovery_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    current = store.new_attempt("cell", ArtifactPolicy.REGENERATE).publish({"ok": 1})
    current.manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="manifest digest mismatch"):
        store.recover_current_records()


def test_recovery_rejects_pointer_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    attempt.publish({"ok": 1})
    pointer_path = attempt.root.parent.parent / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_path"] = "../manifest.json"
    _rewrite_json(pointer_path, pointer)

    with pytest.raises(
        ManifestValidationError,
        match="canonical attempt manifest",
    ):
        store.recover_current_records()


def test_recovery_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    attempt.publish({"ok": 1})
    manifest_path = attempt.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../outside.json"
    _rewrite_json(manifest_path, manifest)
    pointer_path = attempt.root.parent.parent / "current.json"
    _rewrite_pointer_digest(pointer_path, manifest_path)

    with pytest.raises(ManifestValidationError, match="unsafe path"):
        store.recover_current_records()


@pytest.mark.parametrize(
    ("target", "schema"),
    (
        ("pointer", "unknown-current-v9"),
        ("manifest", "unknown-attempt-v9"),
    ),
)
def test_recovery_rejects_schema_mismatch(
    tmp_path: Path,
    target: str,
    schema: str,
) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    attempt.publish({"ok": 1})
    pointer_path = attempt.root.parent.parent / "current.json"
    manifest_path = attempt.root / "manifest.json"
    path = pointer_path if target == "pointer" else manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = schema
    _rewrite_json(path, payload)
    if target == "manifest":
        _rewrite_pointer_digest(pointer_path, manifest_path)

    with pytest.raises(ManifestValidationError, match=r"unsupported .* schema"):
        store.recover_current_records()


def test_recovery_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    payload_path = attempt.path("payload.bin")
    payload_path.write_bytes(b"original")
    attempt.publish({"ok": 1}, artifact_paths=("payload.bin",))
    payload_path.write_bytes(b"modified")

    with pytest.raises(
        ManifestValidationError,
        match=r"artifact (size|digest) mismatch",
    ):
        store.recover_current_records()


def test_attempt_paths_cannot_escape_explicit_artifact_root(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REGENERATE)
    with pytest.raises(ManifestValidationError, match="unsafe path"):
        attempt.path("../outside.json")
    assert not (tmp_path / "outside.json").exists()


def test_based_on_record_must_belong_to_same_cell(tmp_path: Path) -> None:
    store = _store(tmp_path)
    current = store.new_attempt("first", ArtifactPolicy.REGENERATE).publish({"ok": 1})
    with pytest.raises(ArtifactStoreError, match="different cell"):
        store.new_attempt("second", ArtifactPolicy.RETIME, based_on=current)


def test_named_filesystem_lock_times_out_across_processes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue[str] = context.Queue()
    with store.named_lock("prepared-ufo"):
        process = context.Process(
            target=_try_named_lock,
            args=(
                str(store.artifact_root),
                str(store.lock_root),
                queue,
            ),
        )
        process.start()
        process.join(timeout=3)
    assert process.exitcode == 0
    try:
        outcome = queue.get(timeout=1)
    except Empty as error:
        raise AssertionError("child did not report its lock outcome") from error
    assert outcome == "timeout"


def test_named_filesystem_lock_is_reentrant_in_same_thread(tmp_path: Path) -> None:
    store = _store(tmp_path)
    competing = ArtifactStore(
        artifact_root=store.artifact_root,
        lock_root=store.lock_root,
    )

    with (
        store.named_lock("measurement-lineage"),
        competing.named_lock("measurement-lineage", timeout=0.0),
        store.named_lock("measurement-lineage", timeout=0.0),
    ):
        pass

    with store.named_lock("measurement-lineage", timeout=0.0):
        pass


def test_cell_and_named_locks_use_separate_explicit_roots(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store.cell_lock("cell"), store.named_lock("report-writer"):
        assert store.artifact_root == (tmp_path / "artifacts").resolve()
        assert store.lock_root == (tmp_path / "coordination").resolve()
    assert list((store.lock_root / "cells").glob("*.lock"))
    assert list((store.lock_root / "named").glob("*.lock"))


def test_manifest_shape_is_stable_and_records_policy_lineage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.new_attempt("cell", ArtifactPolicy.REGENERATE).publish({"run": 1})
    second_attempt = store.new_attempt(
        "cell",
        ArtifactPolicy.RETIME,
        based_on=first,
    )
    second_attempt.publish({"run": 2})
    manifest = json.loads(
        (second_attempt.root / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema"] == ATTEMPT_SCHEMA
    assert manifest["artifact_policy"] == "retime"
    assert manifest["based_on"] == {
        "attempt_id": first.attempt_id,
        "manifest_sha256": first.manifest_sha256,
    }
