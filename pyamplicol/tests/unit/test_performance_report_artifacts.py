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
    PORTABLE_CURRENT_SCHEMA,
    ArtifactAction,
    ArtifactStore,
    ArtifactStoreError,
    LockTimeoutError,
    ManifestValidationError,
)
from tools.performance_report.models import ArtifactPolicy
from tools.performance_report.service import ReportPaths


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "coordination",
    )


def _portable_store(campaign: Path, *, repo: Path) -> ArtifactStore:
    artifact_root = campaign / "campaign_artifacts"
    paths = ReportPaths(
        repo_root=repo,
        docs_dir=campaign,
        results_dir=campaign / "results",
        artifact_root=artifact_root,
        coordination_root=artifact_root / "coordination",
    )
    return ArtifactStore(
        artifact_root=paths.artifact_root,
        lock_root=paths.coordination_root,
        current_publication_paths=paths,
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


def test_portable_current_bytes_survive_campaign_move_and_materialize(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    campaign_a = tmp_path / "parent-a" / "manual-a"
    store_a = _portable_store(campaign_a, repo=repo)
    attempt = store_a.new_attempt("cell", ArtifactPolicy.REGENERATE)
    artifact = attempt.path("artifact")
    artifact.mkdir()
    (artifact / "payload.bin").write_bytes(b"payload")
    identity = {"native_extension": {"path": "/authenticated/runtime.so"}}
    current_a = attempt.publish(
        {
            "status": "ok",
            "artifact": {"path": str(artifact)},
            "provenance": {
                "worker_log": str(attempt.root / "worker.log"),
                "runtime_identity": identity,
                "runtime_identity_sha256": hashlib.sha256(
                    json.dumps(
                        identity,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest(),
            },
        },
        artifact_paths=("artifact/payload.bin",),
    )
    stored = json.loads((attempt.root / "result.json").read_text())
    pointer = json.loads(
        (attempt.root.parent.parent / "current.json").read_text(encoding="utf-8")
    )
    assert pointer["schema"] == PORTABLE_CURRENT_SCHEMA
    assert stored["artifact"]["path"].startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    assert stored["provenance"]["worker_log"].startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    assert stored["provenance"]["runtime_identity"] == identity

    campaign_b = tmp_path / "parent-b" / "renamed-campaign"
    campaign_b.parent.mkdir()
    campaign_a.rename(campaign_b)
    store_b = _portable_store(campaign_b, repo=repo)

    current_b = store_b.load_current("cell")
    lightweight = store_b.lightweight_current_payload("cell")
    assert current_b is not None
    assert lightweight is not None
    expected_artifact = current_b.result_path.parent / "artifact"
    assert current_b.result["artifact"]["path"] == str(expected_artifact)
    assert lightweight[1]["artifact"]["path"] == str(expected_artifact)
    assert current_b.manifest_sha256 == current_a.manifest_sha256


def test_portable_store_rejects_old_absolute_current_without_migration(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    campaign = tmp_path / "manual"
    artifact_root = campaign / "campaign_artifacts"
    ordinary = ArtifactStore(
        artifact_root=artifact_root,
        lock_root=artifact_root / "coordination",
    )
    attempt = ordinary.new_attempt("cell", ArtifactPolicy.REGENERATE)
    artifact = attempt.path("artifact")
    artifact.mkdir()
    attempt.publish({"artifact": {"path": str(artifact)}})

    portable = _portable_store(campaign, repo=repo)
    with pytest.raises(ManifestValidationError, match="current pointer schema"):
        portable.load_current("cell")
    with pytest.raises(ManifestValidationError, match="unsupported shape"):
        portable.lightweight_current_payload("cell")


def test_portable_store_rejects_artifactless_legacy_terminal_current(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    campaign = tmp_path / "manual"
    artifact_root = campaign / "campaign_artifacts"
    ordinary = ArtifactStore(
        artifact_root=artifact_root,
        lock_root=artifact_root / "coordination",
    )
    attempt = ordinary.new_attempt("cell", ArtifactPolicy.REGENERATE)
    attempt.publish({"status": "memory_limit", "artifact": None})

    portable = _portable_store(campaign, repo=repo)
    with pytest.raises(ManifestValidationError, match="current pointer schema"):
        portable.load_current("cell")
    with pytest.raises(ManifestValidationError, match="unsupported shape"):
        portable.lightweight_current_payload("cell")


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


def test_unpublished_attempt_can_be_discarded_without_touching_current(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    current = store.new_attempt("cell", ArtifactPolicy.REGENERATE).publish(
        {"status": "ok"}
    )
    partial = store.new_attempt(
        "cell",
        ArtifactPolicy.REGENERATE,
        based_on=current,
    )
    partial.path("worker.log").write_text("partial\n", encoding="ascii")
    partial_root = partial.root

    partial.discard()

    assert not partial_root.exists()
    assert store.load_current("cell").attempt_id == current.attempt_id
    with pytest.raises(ArtifactStoreError, match="already sealed"):
        partial.discard()


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
    context = multiprocessing.get_context("spawn")
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


def test_seal_existing_worker_result_publishes_same_authenticated_attempt(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REUSE)
    attempt.write_json(
        "worker-result.json",
        {"status": "ok", "validation": {"status": "ok"}},
    )
    attempt.path("worker.log").write_text("completed\n", encoding="utf-8")
    worker_result = attempt.root / "worker-result.json"
    digest = hashlib.sha256(worker_result.read_bytes()).hexdigest()

    current = store.seal_existing_worker_result(
        "cell",
        attempt.attempt_id,
        worker_result_sha256=digest,
        artifact_policy=ArtifactPolicy.REUSE,
        validate_result=lambda result, root: (
            None
            if result["status"] == "ok" and (root / "worker.log").is_file()
            else (_ for _ in ()).throw(ValueError("incomplete evidence"))
        ),
    )

    assert current.attempt_id == attempt.attempt_id
    assert current.artifact_policy is ArtifactPolicy.REUSE
    assert current.result == {
        "status": "ok",
        "validation": {"status": "ok"},
    }
    assert {artifact.relative_path for artifact in current.artifacts} == {
        "result.json",
        "worker-result.json",
        "worker.log",
    }
    assert json.loads((attempt.root / "result.json").read_text()) == current.result
    assert store.load_current("cell") == current


@pytest.mark.parametrize("mutation", ("digest", "wrong-cell", "incomplete"))
def test_seal_existing_worker_result_rejects_untrusted_or_incomplete_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = _store(tmp_path)
    attempt = store.new_attempt("cell", ArtifactPolicy.REUSE)
    attempt.write_json("worker-result.json", {"status": "ok"})
    worker_result = attempt.root / "worker-result.json"
    digest = hashlib.sha256(worker_result.read_bytes()).hexdigest()

    with pytest.raises((ArtifactStoreError, ValueError)):
        store.seal_existing_worker_result(
            "other-cell" if mutation == "wrong-cell" else "cell",
            attempt.attempt_id,
            worker_result_sha256=(
                "0" * 64 if mutation == "digest" else digest
            ),
            artifact_policy=ArtifactPolicy.REUSE,
            validate_result=(
                (lambda _result, _root: (_ for _ in ()).throw(
                    ValueError("incomplete evidence")
                ))
                if mutation == "incomplete"
                else lambda _result, _root: None
            ),
        )

    assert not (attempt.root / "result.json").exists()
    assert not (attempt.root / "manifest.json").exists()
    assert store.load_current("cell", missing_ok=True) is None
