# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement_lineage import (
    CLASS_C_HZZ_IMPACT,
    MEASUREMENT_LINEAGE_FILENAME,
    MeasurementLineageError,
    audit_measurement_lineage,
    class_c_pending_path,
    finalize_class_c_bridge,
    hzz_agreement_closure,
    hzz_impacted_cells,
    load_measurement_lineage,
    prepare_class_c_bridge,
)
from tools.performance_report.models import ArtifactPolicy, ExecutionMode, ModelKey
from tools.performance_report.workspace import (
    _authenticated_environment_payload,
    _environment_tex,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime(
    package_tree: str,
    *,
    native_build_inputs: str = "a" * 64,
    native_extension: str = "b" * 64,
) -> dict[str, object]:
    return {
        "package_version": "0.1.0",
        "native_build_inputs_sha256": native_build_inputs,
        "native_extension": {"sha256": native_extension},
        "python_package_tree": {"sha256": package_tree},
        "candidate_build_identity": {
            "candidate_fingerprint": "candidate-fixture"
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": ["neon"],
        },
    }


def _write(repo: Path, relative: str, value: str | bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _repository(tmp_path: Path) -> tuple[Path, Path, ArtifactStore, str]:
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
    for path in (
        "tests/unit/test_model_builtin.py",
        "tests/unit/test_packaged_prepared_model.py",
        "tests/unit/test_recurrence_catalog_builder.py",
    ):
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
        lock_root=repo
        / ".artifacts/performance-report-coordination/macbook_M3",
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
    for path in (
        "tests/unit/test_model_builtin.py",
        "tests/unit/test_packaged_prepared_model.py",
        "tests/unit/test_recurrence_catalog_builder.py",
    ):
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
    _git(repo, "add", "tests/unit/test_model_builtin.py")
    _git(repo, "add", "tests/unit/test_packaged_prepared_model.py")
    _git(repo, "add", "tests/unit/test_recurrence_catalog_builder.py")
    _git(repo, "commit", "-q", "-m", "descendant")
    return repo, profile, store, ancestor


def test_hzz_impact_and_direct_agreement_closure_are_exact() -> None:
    impacted = hzz_impacted_cells()
    closure = hzz_agreement_closure()

    assert len(impacted) == 20
    assert len(closure) == 20
    assert all(
        cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
        and cell.process_key == "dd_zzz_jets"
        and cell.n_final >= 3
        for cell in impacted
    )
    assert all(
        cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.UFO_SM
        and cell.process_key == "dd_zzz_jets"
        and cell.n_final >= 3
        for cell in closure
    )
    assert {cell.cell_id for cell in impacted}.isdisjoint(
        cell.cell_id for cell in closure
    )


def test_prepare_finalize_and_audit_class_c_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor = _repository(tmp_path)
    descendant = _git(repo, "rev-parse", "HEAD")
    prepared = prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_HZZ_IMPACT,
    )
    pending = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )

    assert prepared["state"] == "pending"
    assert len(prepared["retained_currents"]) == 1
    assert len(prepared["impacted_cells"]) == 20
    assert len(prepared["agreement_closure_cells"]) == 20
    assert len(prepared["no_attempt_cells"]) == len(
        REPORT_CATALOG.measurement_cells()
    ) - 1

    relinked_runtime = _runtime(
        "d" * 64,
        native_extension="e" * 64,
    )
    new_environment = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=descendant,
        active_runtime=relinked_runtime,
    )

    def fake_refresh(
        _repo: Path,
        _profile: str,
        *,
        expected_source_revision: str,
        runtime_auditor: object,
        _skip_workspace_validation: bool,
    ) -> dict[str, str]:
        assert expected_source_revision == descendant
        assert _skip_workspace_validation is True
        del runtime_auditor
        with store.named_lock("measurement-lineage", timeout=0.0):
            pass
        (profile / "report_environment.json").write_text(
            json.dumps(new_environment, sort_keys=True) + "\n",
            encoding="ascii",
        )
        (profile / "report_environment.tex").write_text(
            _environment_tex(new_environment),
            encoding="utf-8",
        )
        return new_environment

    monkeypatch.setattr(
        "tools.performance_report.workspace.refresh_profile_environment",
        fake_refresh,
    )
    finalized = finalize_class_c_bridge(
        repo,
        profile,
        store,
        pending_path=pending,
        expected_active_source_revision=descendant,
        runtime_auditor=lambda _revision, _root: relinked_runtime,
    )

    assert finalized["state"] == "finalized"
    assert (
        finalized["ancestor_environment"]["native_extension_sha256"]
        != finalized["descendant_environment"]["native_extension_sha256"]
    )
    assert "native_extension_sha256" not in finalized["runtime_invariant_fields"]
    assert (profile / MEASUREMENT_LINEAGE_FILENAME).is_file()
    lineage = load_measurement_lineage(
        repo,
        profile,
        expected_active_revision=descendant,
        expected_active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
    )
    assert lineage is not None
    retained = store.recover_current_records()[0]
    assert lineage.source_for_current(
        retained,
        active_revision=descendant,
        active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
    ) == (ancestor, _git(repo, "rev-parse", f"{ancestor}^{{tree}}"))
    assert audit_measurement_lineage(
        repo,
        profile,
        store,
        expected_active_source_revision=descendant,
    )["runtime_invariants_match"] is True


def test_class_c_bridge_rejects_changed_native_build_inputs(
    tmp_path: Path,
) -> None:
    repo, profile, store, ancestor = _repository(tmp_path)
    descendant = _git(repo, "rev-parse", "HEAD")
    prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_HZZ_IMPACT,
    )
    pending = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )

    with pytest.raises(
        MeasurementLineageError,
        match="dependency/native/host runtime identity",
    ):
        finalize_class_c_bridge(
            repo,
            profile,
            store,
            pending_path=pending,
            expected_active_source_revision=descendant,
            runtime_auditor=lambda _revision, _root: _runtime(
                "d" * 64,
                native_build_inputs="f" * 64,
            ),
        )


def test_class_c_bridge_rejects_disallowed_source_change(tmp_path: Path) -> None:
    repo, profile, store, ancestor = _repository(tmp_path)
    _write(repo, "dependencies/lock.toml", "changed = true\n")
    _git(repo, "add", "dependencies/lock.toml")
    _git(repo, "commit", "-q", "-m", "disallowed")

    with pytest.raises(MeasurementLineageError, match="disallowed path"):
        prepare_class_c_bridge(
            repo,
            profile,
            store,
            ancestor_revision=ancestor,
            descendant_revision=_git(repo, "rev-parse", "HEAD"),
            impact=CLASS_C_HZZ_IMPACT,
        )


def test_lineage_envelope_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor = _repository(tmp_path)
    descendant = _git(repo, "rev-parse", "HEAD")
    prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_HZZ_IMPACT,
    )
    pending = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )
    new_environment = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=descendant,
        active_runtime=_runtime("d" * 64),
    )

    def fake_refresh(*_args: object, **_kwargs: object) -> dict[str, str]:
        (profile / "report_environment.json").write_text(
            json.dumps(new_environment, sort_keys=True) + "\n",
            encoding="ascii",
        )
        (profile / "report_environment.tex").write_text(
            _environment_tex(new_environment),
            encoding="utf-8",
        )
        return new_environment

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
    lineage_path = profile / MEASUREMENT_LINEAGE_FILENAME
    raw = json.loads(lineage_path.read_text(encoding="ascii"))
    raw["payload"]["retained_currents"][0]["manifest_sha256"] = "0" * 64
    lineage_path.write_text(json.dumps(raw) + "\n", encoding="ascii")

    with pytest.raises(MeasurementLineageError, match="canonical digest"):
        load_measurement_lineage(
            repo,
            profile,
            expected_active_revision=descendant,
            expected_active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
        )
