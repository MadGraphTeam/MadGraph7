# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.performance_report.artifacts import (
    ArtifactPolicy,
    ArtifactStore,
    ArtifactStoreError,
)
from tools.performance_report.campaign_policy import MACBOOK_M3_POLICY
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement_lineage import (
    CLASS_C_HZZ_IMPACT,
    MEASUREMENT_LINEAGE_FILENAME,
    MeasurementLineageError,
    _resolve_recurrence_artifact_owner,
    class_c_pending_path,
    finalize_class_c_bridge,
    hzz_agreement_closure,
    hzz_impacted_cells,
    load_measurement_lineage,
    prepare_class_c_bridge,
)
from tools.performance_report.models import ExecutionMode, ModelKey
from tools.performance_report.scheduler import CampaignScheduler, CampaignSettings
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.workspace import (
    ReportWorkspaceError,
    _authenticated_environment_payload,
    _environment_tex,
    initialize_profile,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: str, value: str | bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _runtime(package_tree: str, *, fingerprint: str = "candidate-fixture") -> dict:
    return {
        "package_version": "0.1.0",
        "native_build_inputs_sha256": "a" * 64,
        "native_extension": {"sha256": "b" * 64},
        "python_package_tree": {"sha256": package_tree},
        "candidate_build_identity": {
            "candidate_fingerprint": fingerprint,
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": ["neon"],
        },
    }


def _repository(tmp_path: Path) -> tuple[Path, Path, ArtifactStore, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "lineage@example.invalid")
    _git(repo, "config", "user.name", "Lineage Test")
    profile = repo / "docs/performance_reports/macbook_M3"
    _write(
        repo,
        "docs/performance_reports/macbook_M3/report-workspace.json",
        json.dumps({"campaign_policy": {"name": "fixture"}}) + "\n",
    )
    _write(repo, "src/pyamplicol/models/builtin/lowering.py", "VALUE = 1\n")
    feature_paths = (
        "tests/unit/test_model_builtin.py",
        "tests/unit/test_packaged_prepared_model.py",
        "tests/unit/test_recurrence_catalog_builder.py",
    )
    for path in feature_paths:
        _write(repo, path, "VALUE = 1\n")
    for root in (
        "src/pyamplicol/assets/prepared_models",
        "release_assets/prepared_models",
    ):
        for architecture in ("aarch64", "x86_64"):
            stem = f"built-in-sm-jit-o2-{architecture}"
            _write(repo, f"{root}/{stem}.pyamplicol-model", b"old bundle\n")
            _write(repo, f"{root}/{stem}.metadata.json", '{"old":true}\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "ancestor")
    ancestor = _git(repo, "rev-parse", "HEAD")
    environment = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=ancestor,
        active_runtime=_runtime("c" * 64),
    )
    _write(
        repo,
        "docs/performance_reports/macbook_M3/report_environment.json",
        json.dumps(environment, sort_keys=True) + "\n",
    )
    store = ArtifactStore(
        artifact_root=repo / ".artifacts/performance-report/macbook_M3",
        lock_root=repo / ".artifacts/performance-report-coordination/macbook_M3",
    )
    retained = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
    )
    ancestor_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    store.new_attempt(retained.cell_id, ArtifactPolicy.REGENERATE).publish(
        {
            "status": "ok",
            "provenance": {
                "report_source_revision": ancestor,
                "report_source_tree": ancestor_tree,
                "report_measured_source_revision": ancestor,
                "report_measured_source_tree": ancestor_tree,
            },
        }
    )
    _write(repo, "src/pyamplicol/models/builtin/lowering.py", "VALUE = 2\n")
    for path in feature_paths:
        _write(repo, path, "VALUE = 2\n")
    for root in (
        "src/pyamplicol/assets/prepared_models",
        "release_assets/prepared_models",
    ):
        for architecture in ("aarch64", "x86_64"):
            stem = f"built-in-sm-jit-o2-{architecture}"
            _write(repo, f"{root}/{stem}.pyamplicol-model", b"new bundle\n")
            _write(repo, f"{root}/{stem}.metadata.json", '{"new":true}\n')
    _git(repo, "add", "src/pyamplicol/models/builtin/lowering.py")
    _git(repo, "add", "src/pyamplicol/assets/prepared_models")
    _git(repo, "add", "release_assets/prepared_models")
    for path in feature_paths:
        _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", "descendant")
    return repo, profile, store, ancestor, _git(repo, "rev-parse", "HEAD")


def _prepare(
    repo: Path,
    profile: Path,
    store: ArtifactStore,
    ancestor: str,
    descendant: str,
) -> Path:
    prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_HZZ_IMPACT,
    )
    return class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )


def _finalize(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    profile: Path,
    store: ArtifactStore,
    descendant: str,
    pending: Path,
) -> None:
    environment = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=descendant,
        active_runtime=_runtime("d" * 64),
    )

    def fake_refresh(*_args: object, **_kwargs: object) -> dict[str, str]:
        (profile / "report_environment.json").write_text(
            json.dumps(environment, sort_keys=True) + "\n",
            encoding="ascii",
        )
        (profile / "report_environment.tex").write_text(
            _environment_tex(environment),
            encoding="utf-8",
        )
        return environment

    monkeypatch.setattr(
        "tools.performance_report.workspace.refresh_profile_environment",
        fake_refresh,
    )
    finalize_class_c_bridge(
        repo,
        profile,
        store,
        pending_path=pending,
        expected_active_source_revision=descendant,
        runtime_auditor=lambda _revision, _root: _runtime("d" * 64),
    )


def test_required_descendant_closure_is_exact_even_without_currents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    pending = _prepare(repo, profile, store, ancestor, descendant)
    _finalize(monkeypatch, repo, profile, store, descendant, pending)
    lineage = load_measurement_lineage(
        repo,
        profile,
        expected_active_revision=descendant,
        expected_active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
    )

    assert lineage is not None
    expected = {
        cell.cell_id for cell in (*hzz_impacted_cells(), *hzz_agreement_closure())
    }
    current_ids = {record.cell_id for record in store.recover_current_records()}
    assert len(expected) == 40
    assert lineage.required_descendant_cell_ids == expected
    assert not (expected & current_ids)


def test_attempt_artifact_byte_tamper_blocks_finalization(tmp_path: Path) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    pending = _prepare(repo, profile, store, ancestor, descendant)
    result_path = store.recover_current_records()[0].result_path
    result_path.write_bytes(result_path.read_bytes() + b" ")

    with pytest.raises(
        ArtifactStoreError,
        match=r"artifact (size|digest) mismatch",
    ):
        finalize_class_c_bridge(
            repo,
            profile,
            store,
            pending_path=pending,
            expected_active_source_revision=descendant,
            runtime_auditor=lambda _revision, _root: _runtime("d" * 64),
        )
    assert not (profile / MEASUREMENT_LINEAGE_FILENAME).exists()


def test_new_orphan_attempt_blocks_finalization(tmp_path: Path) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    pending = _prepare(repo, profile, store, ancestor, descendant)
    no_current = hzz_impacted_cells()[0]
    store.new_attempt(
        no_current.cell_id,
        ArtifactPolicy.REGENERATE,
    ).mark_failed("new orphan attempt")

    with pytest.raises(
        MeasurementLineageError,
        match="current or attempt state changed",
    ):
        finalize_class_c_bridge(
            repo,
            profile,
            store,
            pending_path=pending,
            expected_active_source_revision=descendant,
            runtime_auditor=lambda _revision, _root: _runtime("d" * 64),
        )
    assert not (profile / MEASUREMENT_LINEAGE_FILENAME).exists()


def test_failed_refresh_rolls_back_environment_and_lineage_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    pending = _prepare(repo, profile, store, ancestor, descendant)
    json_path = profile / "report_environment.json"
    tex_path = profile / "report_environment.tex"
    lineage_path = profile / MEASUREMENT_LINEAGE_FILENAME
    tex_path.write_bytes(b"exact ancestor tex bytes\n")
    lineage_path.write_bytes(b"exact prior lineage bytes\n")
    before = {
        path: path.read_bytes() for path in (json_path, tex_path, lineage_path)
    }
    expected = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=descendant,
        active_runtime=_runtime("d" * 64),
    )

    def corrupting_refresh(*_args: object, **_kwargs: object) -> dict[str, str]:
        json_path.write_bytes(b"corrupt environment\n")
        tex_path.write_bytes(b"corrupt tex\n")
        lineage_path.write_bytes(b"corrupt lineage\n")
        returned = dict(expected)
        returned["candidate_fingerprint"] = "mutated-after-authentication"
        return returned

    monkeypatch.setattr(
        "tools.performance_report.workspace.refresh_profile_environment",
        corrupting_refresh,
    )
    with pytest.raises(
        MeasurementLineageError,
        match="written descendant environment differs",
    ):
        finalize_class_c_bridge(
            repo,
            profile,
            store,
            pending_path=pending,
            expected_active_source_revision=descendant,
            runtime_auditor=lambda _revision, _root: _runtime("d" * 64),
        )

    assert {path: path.read_bytes() for path in before} == before


def test_environment_projects_only_candidate_fingerprint() -> None:
    first = _runtime("d" * 64)
    second = _runtime("d" * 64)
    first["candidate_build_identity"]["host_checkout"] = "/private/host/one"
    first["candidate_build_identity"]["nested"] = {"digest": "1" * 64}
    second["candidate_build_identity"]["host_checkout"] = "/other/host/two"
    second["candidate_build_identity"]["nested"] = {"digest": "2" * 64}

    left = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision="e" * 40,
        active_runtime=first,
    )
    right = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision="e" * 40,
        active_runtime=second,
    )

    assert left == right
    assert left["candidate_fingerprint"] == "candidate-fixture"
    assert "/private/host" not in json.dumps(left)


def test_non_reset_profile_clone_rejects_source_lineage(tmp_path: Path) -> None:
    repo, profile, _store, _ancestor, _descendant = _repository(tmp_path)
    (profile / MEASUREMENT_LINEAGE_FILENAME).write_text("{}\n", encoding="ascii")

    with pytest.raises(
        ReportWorkspaceError,
        match=r"mixed Class-C source lineage.*reset_measurements=True",
    ):
        initialize_profile(
            repo,
            "x86_EPYC",
            source_profile="macbook_M3",
            reset_measurements=False,
        )
    assert not (repo / "docs/performance_reports/x86_EPYC").exists()


def test_scheduler_rejects_mixed_source_without_lineage_before_attempt(
    tmp_path: Path,
) -> None:
    repo, _profile, store, _ancestor, _descendant = _repository(tmp_path)
    before = tuple(store.artifact_root.rglob("manifest.json"))
    policy = MACBOOK_M3_POLICY
    settings = CampaignSettings(
        workers=policy.workers or 1,
        cell_cores=policy.cell_cores or 1,
        target_runtime_seconds=policy.target_runtime_seconds,
        max_rss_bytes=policy.memory_limit_bytes,
        campaign_policy=policy,
        report_profile="macbook_M3",
    )
    service = ReportService(ReportPaths.from_repo(repo, profile="macbook_M3"))

    with pytest.raises(
        MeasurementLineageError,
        match="mixed/ancestor measurements but no authenticated measurement lineage",
    ):
        CampaignScheduler(service, settings=settings)

    assert tuple(store.artifact_root.rglob("manifest.json")) == before


def _runtime_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _owner_provenance(
    *,
    observation_path: str,
    revision: str = "a" * 40,
    tree: str = "b" * 40,
) -> dict[str, object]:
    observations = [{"path": observation_path}]
    policy = {
        "kind": "fixture-loaded-origin-policy",
        "authenticated": True,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": _runtime_digest(observations),
    }
    identity = {
        "kind": "fixture-runtime-identity",
        "artifact_id": "shared-artifact",
        "loaded_module_origin_policy": policy,
    }
    stable_identity = dict(identity)
    stable_policy = dict(policy)
    for field in (
        "observed_module_count",
        "observations",
        "observations_sha256",
    ):
        stable_policy.pop(field)
    stable_identity["loaded_module_origin_policy"] = stable_policy
    stable_digest = _runtime_digest(stable_identity)
    return {
        "report_source_revision": revision,
        "report_source_tree": tree,
        "report_measured_source_revision": revision,
        "report_measured_source_tree": tree,
        "runtime_identity": identity,
        "runtime_identity_sha256": _runtime_digest(identity),
        "runtime_identity_stable_sha256": stable_digest,
        "runtime_identity_postflight_stable_sha256": stable_digest,
        "runtime_identity_postflight_match": True,
    }


def _owner_cells() -> tuple[object, object]:
    consumer = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id.startswith("z_")
        and cell.n_final == 1
        and cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    owners = tuple(
        cell
        for cell in REPORT_CATALOG.equivalent_cells(consumer)
        if cell.dataset_id.startswith("matrix_")
    )
    assert len(owners) == 1
    return consumer, owners[0]


def _publish_owned_artifact(
    store: ArtifactStore,
    cell_id: str,
    *,
    observation_path: str,
    payload: bytes = b"authenticated artifact payload\n",
) -> tuple[object, Path]:
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    artifact_file = attempt.path("artifact/payload.bin")
    artifact_file.write_bytes(payload)
    artifact_root = artifact_file.parent
    record = attempt.publish(
        {
            "status": "ok",
            "artifact": {
                "path": str(artifact_root),
                "process_id": "fixture-process",
            },
            "provenance": _owner_provenance(
                observation_path=observation_path,
            ),
        },
        artifact_paths=("artifact/payload.bin",),
    )
    return record, artifact_root.resolve(strict=True)


def _publish_artifact_consumer(
    store: ArtifactStore,
    cell_id: str,
    artifact_root: Path,
) -> object:
    return store.new_attempt(
        cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(
        {
            "status": "ok",
            "artifact": {
                "path": str(artifact_root),
                "process_id": "fixture-process",
            },
            "provenance": _owner_provenance(
                observation_path="/consumer/runtime.py",
            ),
        }
    )


def test_recurrence_reachability_accepts_exact_matrix_peer_artifact_owner(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "profile/artifacts",
        lock_root=tmp_path / "profile/locks",
    )
    consumer, owner_cell = _owner_cells()
    owner, artifact_root = _publish_owned_artifact(
        store,
        owner_cell.cell_id,
        observation_path="/owner/runtime.py",
    )
    consumer_record = _publish_artifact_consumer(
        store,
        consumer.cell_id,
        artifact_root,
    )

    evidence = _resolve_recurrence_artifact_owner(
        store,
        REPORT_CATALOG,
        consumer_record,
        consumer,
        artifact_root,
        "fixture-process",
    )

    assert evidence["relation"] == "equivalent-matrix-peer"
    assert evidence["consumer_cell_id"] == consumer.cell_id
    assert evidence["owner_cell_id"] == owner_cell.cell_id
    assert evidence["owner_attempt_id"] == owner.attempt_id
    assert evidence["owner_current_locator"].endswith("/current.json")
    assert evidence["consumer_runtime_identity_sha256"] != (
        evidence["owner_runtime_identity_sha256"]
    )
    assert evidence["consumer_runtime_identity_stable_sha256"] == (
        evidence["owner_runtime_identity_stable_sha256"]
    )


def test_recurrence_reachability_rejects_foreign_artifact_owner(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "profile/artifacts",
        lock_root=tmp_path / "profile/locks",
    )
    consumer, owner_cell = _owner_cells()
    _publish_owned_artifact(
        store,
        owner_cell.cell_id,
        observation_path="/owner/runtime.py",
    )
    foreign_cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id.startswith("matrix_")
        and cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
        and cell.process_key != consumer.process_key
    )
    _foreign, foreign_root = _publish_owned_artifact(
        store,
        foreign_cell.cell_id,
        observation_path="/foreign/runtime.py",
    )
    consumer_record = _publish_artifact_consumer(
        store,
        consumer.cell_id,
        foreign_root,
    )

    with pytest.raises(
        MeasurementLineageError,
        match="not owned by its matrix peer",
    ):
        _resolve_recurrence_artifact_owner(
            store,
            REPORT_CATALOG,
            consumer_record,
            consumer,
            foreign_root,
            "fixture-process",
        )


def test_recurrence_reachability_rejects_owner_artifact_hash_drift(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "profile/artifacts",
        lock_root=tmp_path / "profile/locks",
    )
    consumer, owner_cell = _owner_cells()
    _owner, artifact_root = _publish_owned_artifact(
        store,
        owner_cell.cell_id,
        observation_path="/owner/runtime.py",
    )
    consumer_record = _publish_artifact_consumer(
        store,
        consumer.cell_id,
        artifact_root,
    )
    (artifact_root / "payload.bin").write_bytes(b"altered after publication\n")

    with pytest.raises(
        MeasurementLineageError,
        match="owner current is invalid",
    ):
        _resolve_recurrence_artifact_owner(
            store,
            REPORT_CATALOG,
            consumer_record,
            consumer,
            artifact_root,
            "fixture-process",
        )
