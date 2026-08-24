# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import tools.performance_report.measurement_lineage as measurement_lineage
from tools.performance_report.artifacts import (
    ArtifactPolicy,
    ArtifactStore,
    ArtifactStoreError,
)
from tools.performance_report.cache import digest_json, empty_measurement
from tools.performance_report.campaign_policy import MACBOOK_M3_POLICY
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement_lineage import (
    _RECURRENCE_SUMMARY_CAP_ALLOWED_PATHS,
    CLASS_C_HZZ_IMPACT,
    CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
    MEASUREMENT_LINEAGE_FILENAME,
    MeasurementLineageError,
    _agreement_graph_digest_matches,
    _is_authorized_native_inputs_transition,
    _require_environment_transition,
    _resolve_recurrence_artifact_owner,
    class_c_pending_path,
    finalize_class_c_bridge,
    hzz_agreement_closure,
    hzz_impacted_cells,
    load_measurement_lineage,
    prepare_class_c_bridge,
    recurrence_summary_cap_agreement_closure,
    recurrence_summary_cap_impacted_cells,
    signed_zero_helicity_agreement_closure,
    signed_zero_helicity_impacted_cells,
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


def _lineage_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(f"{encoded}\n".encode("ascii")).hexdigest()


def _runtime(
    package_tree: str,
    *,
    source_revision: str = "e" * 40,
    fingerprint: str = "candidate-fixture",
    native_build_inputs: str = "a" * 64,
) -> dict:
    return {
        "package_version": "0.1.0",
        "native_build_inputs_sha256": native_build_inputs,
        "native_extension": {"sha256": "b" * 64},
        "python_package_tree": {"sha256": package_tree},
        "candidate_build_identity": {
            "candidate_fingerprint": fingerprint,
            "source_revision": source_revision,
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": ["neon"],
        },
    }


def _validation_failed_result(
    ancestor: str,
    ancestor_tree: str,
    artifact_root: Path,
    *,
    failure_kind: str = "MeasurementValidationError",
) -> dict[str, object]:
    observations = [{"module": "fixture", "origin": "authenticated"}]
    origin_policy = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": digest_json(observations),
    }
    identity = {"loaded_module_origin_policy": origin_policy}
    identity_sha256 = digest_json(identity)
    stable_policy = {
        key: value
        for key, value in origin_policy.items()
        if key
        not in {
            "observed_module_count",
            "observations",
            "observations_sha256",
        }
    }
    stable_identity_sha256 = digest_json({"loaded_module_origin_policy": stable_policy})
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "validation_failed",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0,
            "execution_seconds_per_point": 1.0,
            "matrix_element": 4.0,
            "sample_count": 1,
            "standard_error_seconds_per_point": 0.1,
            "relative_standard_error": 0.1,
            "artifact": {
                "path": str(artifact_root),
                "process_id": "fixture-process",
                "policy": "generated",
            },
            "validation": {
                "status": "validation_failed",
                "resolved_sum": {
                    "status": "ok",
                    "maximum_absolute_difference": 0.0,
                    "maximum_relative_difference": 0.0,
                    "relative_tolerance": 1.0e-12,
                    "absolute_tolerance": 1.0e-15,
                },
                "pointwise": {
                    "status": "validation_failed",
                    "candidate": 4.0,
                    "baseline": 1.0,
                    "absolute_difference": 3.0,
                    "relative_difference": 3.0,
                    "relative_tolerance": 1.0e-12,
                    "absolute_tolerance": 1.0e-15,
                },
                "direct_agreements": [],
            },
            "provenance": {
                "report_source_revision": ancestor,
                "report_source_tree": ancestor_tree,
                "report_measured_source_revision": ancestor,
                "report_measured_source_tree": ancestor_tree,
                "runtime_identity": identity,
                "runtime_identity_sha256": identity_sha256,
                "runtime_identity_stable_sha256": stable_identity_sha256,
                "runtime_identity_postflight_stable_sha256": (stable_identity_sha256),
                "runtime_identity_postflight_loaded_module_origin_policy": (
                    origin_policy
                ),
                "runtime_identity_postflight_match": True,
            },
            "failure": {
                "kind": failure_kind,
                "message": "candidate or same-artifact numerical validation failed",
            },
        }
    )
    return measurement


_SUMMARY_CAP_FAILURE_BYTES = {
    "matrix-recurrence-builtin-sm-lc-n7-gg-gluons-selected-flow": 4_270_140,
    "matrix-recurrence-builtin-sm-lc-n8-dd-tt-jets-selected-flow": 1_083_926,
    "matrix-recurrence-builtin-sm-lc-n9-dd-z-jets-selected-flow": 4_449_888,
    "matrix-recurrence-builtin-sm-lc-n9-ud-w-jets-selected-flow": 4_449_912,
}
_SUMMARY_CAP_DIFF_PATHS = tuple(
    sorted(_RECURRENCE_SUMMARY_CAP_ALLOWED_PATHS)
)


def _summary_cap_failure(summary_bytes: int) -> dict[str, object]:
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "error",
            "failure": {
                "kind": "GenerationError",
                "message": (
                    "Rust recurrence execution summary must be smaller than "
                    f"1 MiB; received {summary_bytes} bytes"
                ),
            },
        }
    )
    return measurement


def _summary_cap_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, ArtifactStore, str, str]:
    repo, profile, store, predecessor_ancestor, ancestor = _repository(tmp_path)
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_PREDECESSOR_REVISION",
        predecessor_ancestor,
    )
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION",
        ancestor,
    )
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_PROFILE",
        profile.name,
    )
    predecessor_pending = _prepare(
        repo,
        profile,
        store,
        predecessor_ancestor,
        ancestor,
    )
    _finalize(
        monkeypatch,
        repo,
        profile,
        store,
        ancestor,
        predecessor_pending,
    )
    ancestor_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    retained = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and store.load_current(cell.cell_id, missing_ok=True) is None
    )
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
    closure_current_id = (
        "matrix-recurrence-builtin-sm-lc-n7-gg-gluons-all-flow"
    )
    store.new_attempt(
        closure_current_id,
        ArtifactPolicy.REGENERATE,
    ).publish(
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
    for cell_id, summary_bytes in sorted(_SUMMARY_CAP_FAILURE_BYTES.items()):
        store.new_attempt(cell_id, ArtifactPolicy.REGENERATE).publish(
            _summary_cap_failure(summary_bytes)
        )
    excluded = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.cell_id not in _SUMMARY_CAP_FAILURE_BYTES
        and cell.cell_id != closure_current_id
    )
    store.new_attempt(excluded.cell_id, ArtifactPolicy.REGENERATE).publish(
        _summary_cap_failure(99)
    )
    for path in _SUMMARY_CAP_DIFF_PATHS:
        _write(repo, path, "DESCENDANT = 2\n")
    _git(repo, "add", *_SUMMARY_CAP_DIFF_PATHS)
    _git(repo, "commit", "-q", "-m", "descendant")
    descendant = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_DESCENDANT_REVISION",
        descendant,
    )
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_ANCESTOR_NATIVE_INPUTS_SHA256",
        "a" * 64,
    )
    monkeypatch.setattr(
        "tools.performance_report.measurement_lineage."
        "_RECURRENCE_SUMMARY_CAP_DESCENDANT_NATIVE_INPUTS_SHA256",
        "d" * 64,
    )
    return repo, profile, store, ancestor, descendant


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
        active_runtime=_runtime("c" * 64, source_revision=ancestor),
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
    *,
    package_tree: str = "d" * 64,
    native_build_inputs: str = "a" * 64,
) -> None:
    environment = _authenticated_environment_payload(
        profile.name,
        expected_source_revision=descendant,
        active_runtime=_runtime(
            package_tree,
            source_revision=descendant,
            native_build_inputs=native_build_inputs,
        ),
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
        runtime_auditor=lambda _revision, _root: _runtime(
            package_tree,
            source_revision=_revision,
            native_build_inputs=native_build_inputs,
        ),
    )


def test_recurrence_summary_cap_native_transition_is_exactly_pinned() -> None:
    ancestor_revision = "be11d8304fdc04893dc0e23e9619be848126e3bc"
    descendant_revision = "2594d8b520b802f71d60bd646f73ebaa5547927a"
    ancestor_digest = (
        "23b9637d5d3fba0947d78cf688df18799b0c9ee5b3bcbfa6a2963a1f1a21f870"
    )
    descendant_digest = (
        "96e1ff79a007aaf67a0900dd6d67327ee00f6bd2cca002589b879aa3a734de08"
    )
    ancestor_environment = _authenticated_environment_payload(
        "x86_EPYC",
        expected_source_revision=ancestor_revision,
        active_runtime=_runtime(
            "c" * 64,
            source_revision=ancestor_revision,
            native_build_inputs=ancestor_digest,
        ),
    )
    descendant_environment = _authenticated_environment_payload(
        "x86_EPYC",
        expected_source_revision=descendant_revision,
        active_runtime=_runtime(
            "d" * 64,
            source_revision=descendant_revision,
            native_build_inputs=descendant_digest,
        ),
    )

    _require_environment_transition(
        impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        ancestor_revision=ancestor_revision,
        descendant_revision=descendant_revision,
        ancestor_environment=ancestor_environment,
        descendant_environment=descendant_environment,
    )
    assert _is_authorized_native_inputs_transition(
        impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        ancestor_revision=ancestor_revision,
        descendant_revision=descendant_revision,
        ancestor_digest=ancestor_digest,
        descendant_digest=descendant_digest,
    )

    for (
        changed_ancestor,
        changed_descendant,
        changed_ancestor_environment,
        changed_descendant_environment,
    ) in (
        (
            "0" * 40,
            descendant_revision,
            ancestor_environment,
            descendant_environment,
        ),
        (
            ancestor_revision,
            "1" * 40,
            ancestor_environment,
            descendant_environment,
        ),
        (
            ancestor_revision,
            descendant_revision,
            {
                **ancestor_environment,
                "native_build_inputs_sha256": "2" * 64,
            },
            descendant_environment,
        ),
        (
            ancestor_revision,
            descendant_revision,
            ancestor_environment,
            {
                **descendant_environment,
                "native_build_inputs_sha256": ancestor_digest,
            },
        ),
        (
            ancestor_revision,
            descendant_revision,
            {
                **ancestor_environment,
                "native_build_inputs_sha256": descendant_digest,
            },
            {
                **descendant_environment,
                "native_build_inputs_sha256": ancestor_digest,
            },
        ),
        (
            ancestor_revision,
            descendant_revision,
            ancestor_environment,
            {
                **descendant_environment,
                "candidate_fingerprint": "unreviewed-native-drift",
            },
        ),
    ):
        with pytest.raises(
            MeasurementLineageError,
            match="exact recurrence-summary-cap transition",
        ):
            _require_environment_transition(
                impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
                ancestor_revision=changed_ancestor,
                descendant_revision=changed_descendant,
                ancestor_environment=changed_ancestor_environment,
                descendant_environment=changed_descendant_environment,
            )

    with pytest.raises(
        MeasurementLineageError,
        match="dependency/native/host runtime identity",
    ):
        _require_environment_transition(
            impact=CLASS_C_HZZ_IMPACT,
            ancestor_revision=ancestor_revision,
            descendant_revision=descendant_revision,
            ancestor_environment=ancestor_environment,
            descendant_environment=descendant_environment,
        )


def test_controller_agreement_transition_is_exactly_lineage_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_digest = "7" * 64
    new_digest = "4" * 64
    monkeypatch.setattr(
        measurement_lineage,
        "_PRE_ARBITRARY_QUARK_LINES_AGREEMENT_GRAPH_SHA256",
        old_digest,
    )
    monkeypatch.setattr(
        measurement_lineage,
        "_ARBITRARY_QUARK_LINES_AGREEMENT_GRAPH_SHA256",
        new_digest,
    )
    exact = {
        "profile": "x86_EPYC",
        "impact": CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        "ancestor_revision": (
            measurement_lineage._RECURRENCE_SUMMARY_CAP_ANCESTOR_REVISION
        ),
        "descendant_revision": (
            measurement_lineage._RECURRENCE_SUMMARY_CAP_DESCENDANT_REVISION
        ),
        "agreement_graph_sha256": old_digest,
    }

    assert _agreement_graph_digest_matches(exact, old_digest)
    assert _agreement_graph_digest_matches(exact, new_digest)
    for changed in (
        {**exact, "profile": "macbook_M3"},
        {**exact, "impact": CLASS_C_HZZ_IMPACT},
        {**exact, "ancestor_revision": "0" * 40},
        {**exact, "descendant_revision": "1" * 40},
        {**exact, "agreement_graph_sha256": "2" * 64},
    ):
        assert not _agreement_graph_digest_matches(changed, new_digest)
    assert not _agreement_graph_digest_matches(exact, "3" * 64)


def test_recurrence_summary_cap_bridge_has_exact_failure_and_closure_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _summary_cap_repository(
        tmp_path,
        monkeypatch,
    )
    prepared = prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
    )

    impacted_ids = {
        cell.cell_id for cell in recurrence_summary_cap_impacted_cells()
    }
    closure_ids = {
        cell.cell_id for cell in recurrence_summary_cap_agreement_closure()
    }
    signed_zero_ids = {
        cell.cell_id for cell in signed_zero_helicity_impacted_cells()
    }
    signed_zero_closure_ids = {
        cell.cell_id for cell in signed_zero_helicity_agreement_closure()
    }
    certificate = prepared["reachability_certificate"]
    assert isinstance(certificate, dict)
    assert impacted_ids == set(_SUMMARY_CAP_FAILURE_BYTES)
    assert len(closure_ids) == 12
    assert not impacted_ids & closure_ids
    assert len(signed_zero_ids) == 24
    assert len(signed_zero_closure_ids) == 42
    assert not (impacted_ids | closure_ids) & (
        signed_zero_ids | signed_zero_closure_ids
    )
    assert {
        str(cell["cell_id"]) for cell in prepared["impacted_cells"]
    } == impacted_ids | signed_zero_ids
    assert {
        str(cell["cell_id"])
        for cell in prepared["agreement_closure_cells"]
    } == closure_ids | signed_zero_closure_ids
    assert certificate["target_summary_bytes"] == _SUMMARY_CAP_FAILURE_BYTES
    assert certificate["inspected_current_count"] == 8
    assert certificate["successful_current_count"] == 3
    assert certificate["excluded_non_success_current_count"] == 1
    assert len(certificate["predecessor"]["retained_currents"]) == 1
    assert {
        str(record["cell_id"]) for record in certificate["records"]
    } == impacted_ids
    assert {
        str(pin["cell_id"]) for pin in prepared["invalidated_currents"]
    } == impacted_ids
    assert len(prepared["retained_currents"]) == 2
    assert [
        pin["cell_id"] for pin in prepared["recompare_currents"]
    ] == ["matrix-recurrence-builtin-sm-lc-n7-gg-gluons-all-flow"]

    pending = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )
    historical_graph = measurement_lineage._agreement_digest(REPORT_CATALOG)
    current_graph = "4" * 64
    monkeypatch.setattr(
        measurement_lineage,
        "_PRE_ARBITRARY_QUARK_LINES_AGREEMENT_GRAPH_SHA256",
        historical_graph,
    )
    monkeypatch.setattr(
        measurement_lineage,
        "_ARBITRARY_QUARK_LINES_AGREEMENT_GRAPH_SHA256",
        current_graph,
    )
    monkeypatch.setattr(
        measurement_lineage,
        "_agreement_digest",
        lambda _catalog: current_graph,
    )
    _finalize(
        monkeypatch,
        repo,
        profile,
        store,
        descendant,
        pending,
        package_tree="e" * 64,
        native_build_inputs="d" * 64,
    )
    lineage = load_measurement_lineage(
        repo,
        profile,
        expected_active_revision=descendant,
        expected_active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
    )
    assert lineage is not None
    assert lineage.required_descendant_cell_ids == (
        impacted_ids
        | closure_ids
        | signed_zero_ids
        | signed_zero_closure_ids
    )


def test_recurrence_summary_cap_bridge_rejects_wrong_failure_byte_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _summary_cap_repository(
        tmp_path,
        monkeypatch,
    )
    cell_id, summary_bytes = next(iter(_SUMMARY_CAP_FAILURE_BYTES.items()))
    store.new_attempt(cell_id, ArtifactPolicy.REGENERATE).publish(
        _summary_cap_failure(summary_bytes + 1)
    )

    with pytest.raises(
        MeasurementLineageError,
        match="not the authenticated recurrence summary-cap GenerationError",
    ):
        prepare_class_c_bridge(
            repo,
            profile,
            store,
            ancestor_revision=ancestor,
            descendant_revision=descendant,
            impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        )


def test_recurrence_summary_cap_bridge_rejects_excluded_current_pointer_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _summary_cap_repository(
        tmp_path,
        monkeypatch,
    )
    prepared = prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
    )
    certificate = prepared["reachability_certificate"]
    assert isinstance(certificate, dict)
    excluded = certificate["excluded_non_success_currents"]
    assert isinstance(excluded, list)
    assert len(excluded) == 1
    cell_id = str(excluded[0]["cell_id"])
    store.new_attempt(cell_id, ArtifactPolicy.REGENERATE).publish(
        _summary_cap_failure(100)
    )
    pending = class_c_pending_path(
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
    )
    with pytest.raises(
        MeasurementLineageError,
        match="certificate changed",
    ):
        finalize_class_c_bridge(
            repo,
            profile,
            store,
            pending_path=pending,
            expected_active_source_revision=descendant,
            runtime_auditor=lambda _revision, _root: _runtime(
                "e" * 64,
                source_revision=_revision,
            ),
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


def test_prepare_partitions_exact_epyc_validation_failures_for_replacement(
    tmp_path: Path,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    ancestor_tree = _git(repo, "rev-parse", f"{ancestor}^{{tree}}")
    failed_ids = {
        "matrix-recurrence-builtin-sm-full-n3-dd-zzz-jets-contracted",
        "matrix-recurrence-builtin-sm-lc-n3-dd-zzz-jets-selected-flow",
    }
    for cell_id in sorted(failed_ids):
        attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
        attempt.write_json("artifact/execution.json", {"fixture": True})
        attempt.publish(
            _validation_failed_result(
                ancestor,
                ancestor_tree,
                attempt.root / "artifact",
            ),
            artifact_paths=("artifact/execution.json",),
        )

    prepared = prepare_class_c_bridge(
        repo,
        profile,
        store,
        ancestor_revision=ancestor,
        descendant_revision=descendant,
        impact=CLASS_C_HZZ_IMPACT,
    )

    reachability = prepared["reachability_certificate"]
    assert isinstance(reachability, dict)
    assert set(reachability["invalidated_validation_failed_current_ids"]) == failed_ids
    assert failed_ids == {
        str(pin["cell_id"]) for pin in prepared["invalidated_currents"]
    }
    assert not failed_ids & {
        str(pin["cell_id"]) for pin in prepared["retained_currents"]
    }


def test_prepare_rejects_validation_failure_outside_hzz_closure(
    tmp_path: Path,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    ancestor_tree = _git(repo, "rev-parse", f"{ancestor}^{{tree}}")
    outside = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
        and cell.process_key != "dd_zzz_jets"
    )
    attempt = store.new_attempt(outside.cell_id, ArtifactPolicy.REGENERATE)
    attempt.write_json("artifact/execution.json", {"fixture": True})
    attempt.publish(
        _validation_failed_result(
            ancestor,
            ancestor_tree,
            attempt.root / "artifact",
        ),
        artifact_paths=("artifact/execution.json",),
    )

    with pytest.raises(
        MeasurementLineageError,
        match="outside the exact HZZ replacement closure",
    ):
        prepare_class_c_bridge(
            repo,
            profile,
            store,
            ancestor_revision=ancestor,
            descendant_revision=descendant,
            impact=CLASS_C_HZZ_IMPACT,
        )


def test_prepare_rejects_wrong_failure_class_inside_hzz_closure(
    tmp_path: Path,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    ancestor_tree = _git(repo, "rev-parse", f"{ancestor}^{{tree}}")
    cell_id = "matrix-recurrence-builtin-sm-full-n3-dd-zzz-jets-contracted"
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    attempt.write_json("artifact/execution.json", {"fixture": True})
    attempt.publish(
        _validation_failed_result(
            ancestor,
            ancestor_tree,
            attempt.root / "artifact",
            failure_kind="UnexpectedFailure",
        ),
        artifact_paths=("artifact/execution.json",),
    )

    with pytest.raises(
        MeasurementLineageError,
        match="not an authenticated numerical-validation failure",
    ):
        prepare_class_c_bridge(
            repo,
            profile,
            store,
            ancestor_revision=ancestor,
            descendant_revision=descendant,
            impact=CLASS_C_HZZ_IMPACT,
        )


def test_prepare_rejects_in_closure_failure_runtime_digest_tamper(
    tmp_path: Path,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    ancestor_tree = _git(repo, "rev-parse", f"{ancestor}^{{tree}}")
    cell_id = "matrix-recurrence-builtin-sm-full-n3-dd-zzz-jets-contracted"
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    attempt.write_json("artifact/execution.json", {"fixture": True})
    result = _validation_failed_result(
        ancestor,
        ancestor_tree,
        attempt.root / "artifact",
    )
    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    provenance["runtime_identity_sha256"] = "0" * 64
    attempt.publish(result, artifact_paths=("artifact/execution.json",))

    with pytest.raises(
        MeasurementLineageError,
        match="runtime identity is invalid",
    ):
        prepare_class_c_bridge(
            repo,
            profile,
            store,
            ancestor_revision=ancestor,
            descendant_revision=descendant,
            impact=CLASS_C_HZZ_IMPACT,
        )


def test_lineage_rejects_failed_pin_moved_into_retained_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, profile, store, ancestor, descendant = _repository(tmp_path)
    ancestor_tree = _git(repo, "rev-parse", f"{ancestor}^{{tree}}")
    cell_id = "matrix-recurrence-builtin-sm-full-n3-dd-zzz-jets-contracted"
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    attempt.write_json("artifact/execution.json", {"fixture": True})
    attempt.publish(
        _validation_failed_result(
            ancestor,
            ancestor_tree,
            attempt.root / "artifact",
        ),
        artifact_paths=("artifact/execution.json",),
    )
    pending = _prepare(repo, profile, store, ancestor, descendant)
    _finalize(monkeypatch, repo, profile, store, descendant, pending)

    lineage_path = profile / MEASUREMENT_LINEAGE_FILENAME
    raw = json.loads(lineage_path.read_text(encoding="ascii"))
    payload = raw["payload"]
    failed_pin = next(
        pin for pin in payload["invalidated_currents"] if pin["cell_id"] == cell_id
    )
    payload["invalidated_currents"].remove(failed_pin)
    payload["retained_currents"].append(failed_pin)
    payload["retained_currents"].sort(key=lambda pin: pin["cell_id"])
    snapshot = {
        field: payload[field]
        for field in (
            "retained_currents",
            "invalidated_currents",
            "recompare_currents",
            "attempt_inventory",
            "no_attempt_cells",
        )
    }
    payload["current_snapshot_sha256"] = _lineage_digest(snapshot)
    raw["payload_sha256"] = _lineage_digest(payload)
    lineage_path.write_text(
        json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        MeasurementLineageError,
        match="current groups do not match",
    ):
        load_measurement_lineage(
            repo,
            profile,
            expected_active_revision=descendant,
            expected_active_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
        )


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
            runtime_auditor=lambda _revision, _root: _runtime(
                "d" * 64,
                source_revision=_revision,
            ),
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
            runtime_auditor=lambda _revision, _root: _runtime(
                "d" * 64,
                source_revision=_revision,
            ),
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
    before = {path: path.read_bytes() for path in (json_path, tex_path, lineage_path)}
    expected = _authenticated_environment_payload(
        "macbook_M3",
        expected_source_revision=descendant,
        active_runtime=_runtime("d" * 64, source_revision=descendant),
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
            runtime_auditor=lambda _revision, _root: _runtime(
                "d" * 64,
                source_revision=_revision,
            ),
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
    assert (
        evidence["consumer_runtime_identity_sha256"]
        != (evidence["owner_runtime_identity_sha256"])
    )
    assert (
        evidence["consumer_runtime_identity_stable_sha256"]
        == (evidence["owner_runtime_identity_stable_sha256"])
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
