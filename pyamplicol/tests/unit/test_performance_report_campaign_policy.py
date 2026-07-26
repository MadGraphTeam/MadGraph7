# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.cache import build_reset_caches
from tools.performance_report.campaign_policy import (
    STRICT_POLICY,
    X86_EPYC_GENERATION_LIMIT_SECONDS,
    X86_EPYC_MEMORY_LIMIT_BYTES,
    X86_EPYC_POLICY,
    CampaignPolicyError,
    PolicyCensorKind,
    PolicyMeasurementState,
    dependency_reference,
    generation_limit_exempt,
    policy_censor_measurement,
    policy_status_label,
    validate_policy_measurement,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.cli import _gb_bytes, _parser
from tools.performance_report.final_audit import audit_final_report
from tools.performance_report.models import ArtifactPolicy, Workload
from tools.performance_report.publication import portable_publication_value
from tools.performance_report.render import _status
from tools.performance_report.resources import (
    GenerationPhaseEvidence,
    ResourceUsage,
    SupervisedResult,
)
from tools.performance_report.scheduler import (
    CampaignScheduler,
    CampaignSettings,
    PlannedCell,
    plan_campaign,
)
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity
from tools.performance_report.workspace import (
    WORKSPACE_MANIFEST,
    WORKSPACE_SCHEMA,
    ReportWorkspaceError,
    load_profile_campaign_policy,
)

_REVISION = "1" * 40
_TREE = "2" * 40
_IDENTITY = ReportSourceIdentity(_REVISION, _TREE, ())


def _resources(peak: int) -> dict[str, object]:
    return {
        "available": True,
        "current_rss_bytes": peak,
        "peak_rss_bytes": peak,
        "child_count": 1,
        "cpu_seconds": 1.0,
        "wall_seconds": 2.0,
        "probe_error": None,
    }


def _x86_settings(**changes: object) -> CampaignSettings:
    values: dict[str, object] = {
        "workers": 10,
        "cell_cores": 1,
        "target_runtime_seconds": 5.0,
        "max_rss_bytes": X86_EPYC_MEMORY_LIMIT_BYTES,
        "allow_symbolica_parallel": True,
        "campaign_policy": X86_EPYC_POLICY,
        "report_profile": "x86_EPYC",
    }
    values.update(changes)
    return CampaignSettings(**values)  # type: ignore[arg-type]


def _memory_censor(cell, *, peak: int) -> dict[str, object]:
    return policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        cell,
        kind=PolicyCensorKind.MEMORY_LIMIT,
        source_identity=_IDENTITY,
        resources=_resources(peak),
        observed_rss_bytes=peak,
    )


def test_x86_policy_has_the_exact_canonical_n4_split() -> None:
    cells = tuple(
        cell for cell in REPORT_CATALOG.measurement_cells() if cell.n_final <= 4
    )
    exempt = tuple(cell for cell in cells if generation_limit_exempt(cell))

    assert len(cells) == 742
    assert len(exempt) == 264
    assert len(cells) - len(exempt) == 478
    assert all(
        cell.measurement.execution_mode.value == "amplicol"
        or (
            cell.measurement.execution_mode.value in {"compiled", "recurrence"}
            and cell.measurement.accuracy.value == "lc"
            and cell.workload is Workload.SELECTED_FLOW
        )
        for cell in exempt
    )


def test_x86_settings_are_exact_and_use_decimal_100_gb() -> None:
    settings = _x86_settings()

    assert settings.max_rss_bytes == 100_000_000_000
    assert settings.timeout_seconds is None
    with pytest.raises(CampaignPolicyError, match="max_rss_bytes"):
        _x86_settings(max_rss_bytes=100 * 1024**3)
    with pytest.raises(CampaignPolicyError, match="workers"):
        _x86_settings(workers=9)
    assert CampaignSettings().campaign_policy is STRICT_POLICY
    assert _gb_bytes(100.0) == X86_EPYC_MEMORY_LIMIT_BYTES
    parsed = _parser().parse_args(
        (
            "--report-profile",
            "x86_EPYC",
            "populate",
            "--max-ram-gb",
            "100",
        )
    )
    assert parsed.max_ram_gb == 100.0


def test_policy_censors_are_canonical_and_tamper_evident() -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.n_final <= 4 and not generation_limit_exempt(cell)
    )
    phase_evidence = {
        "abi": "pyamplicol-report-generation-phase-evidence-v1",
        "phase_state_abi": "pyamplicol-report-worker-phase-state-v1",
        "configured_timeout_seconds": X86_EPYC_GENERATION_LIMIT_SECONDS,
        "supervisor_reason": "generation_timeout",
        "authenticated": True,
        "run_id": "run-1",
        "worker_pid": 123,
        "final_sequence": 1,
        "final_phase": "generation",
        "generation_started_monotonic_ns": 1,
        "generation_finished_monotonic_ns": None,
        "generation_elapsed_seconds": X86_EPYC_GENERATION_LIMIT_SECONDS,
        "final_state_sha256": "3" * 64,
        "error": None,
    }
    resources = _resources(1234)
    resources["generation_phase"] = phase_evidence
    result = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        cell,
        kind=PolicyCensorKind.GENERATION_LIMIT,
        source_identity=_IDENTITY,
        resources=resources,
        observed_generation_seconds=X86_EPYC_GENERATION_LIMIT_SECONDS,
        phase_evidence=phase_evidence,
    )

    assert policy_status_label(result) == ">2h"
    assert (
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            cell,
            result,
            expected_source_revision=_REVISION,
            expected_source_tree=_TREE,
        )
        is PolicyMeasurementState.GENERATION_LIMIT
    )
    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    record = provenance["policy_censor"]
    assert isinstance(record, dict)
    record["observed_generation_seconds"] = 7201.0
    with pytest.raises(CampaignPolicyError, match="sha256"):
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            cell,
            result,
            expected_source_revision=_REVISION,
            expected_source_tree=_TREE,
        )


def test_missing_only_reuses_exact_dependency_censor_and_rebinds_changes(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    baseline = REPORT_CATALOG.baseline_cell(candidate)
    assert baseline is not None
    first = _memory_censor(
        baseline,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )
    store.new_attempt(
        baseline.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(first)
    dependency = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        candidate,
        kind=PolicyCensorKind.DEPENDENCY,
        source_identity=_IDENTITY,
        resources=None,
        dependencies=(dependency_reference(baseline.cell_id, first),),
    )
    store.new_attempt(
        candidate.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(dependency)
    settings = _x86_settings(missing_only=True)

    assert plan_campaign(
        (candidate,),
        store=store,
        settings=settings,
        expected_revision=_REVISION,
        expected_tree=_TREE,
    ) == ()

    second = _memory_censor(
        baseline,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 2,
    )
    store.new_attempt(
        baseline.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(second)
    planned = plan_campaign(
        (candidate,),
        store=store,
        settings=settings,
        expected_revision=_REVISION,
        expected_tree=_TREE,
    )

    assert tuple(item.cell.cell_id for item in planned) == (candidate.cell_id,)
    assert planned[0].force_recompare is True


def test_rendered_policy_markers_are_explicit() -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.n_final <= 4
    )
    result = _memory_censor(
        cell,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )

    assert policy_status_label(result) == ">100GB"
    assert ">100GB" in _status(result)


def test_workspace_policy_is_bound_to_the_exact_measured_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    profile = repo / "docs" / "performance_reports" / "x86_EPYC"
    profile.mkdir(parents=True)
    manifest = {
        "schema": WORKSPACE_SCHEMA,
        "profile": "x86_EPYC",
        "campaign_policy": X86_EPYC_POLICY.as_manifest(),
    }
    (profile / WORKSPACE_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="ascii",
    )
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=repo,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "profile"), cwd=repo, check=True)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        load_profile_campaign_policy(
            repo,
            "x86_EPYC",
            expected_source_revision=revision,
        )
        is X86_EPYC_POLICY
    )
    manifest["campaign_policy"] = STRICT_POLICY.as_manifest()
    (profile / WORKSPACE_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="ascii",
    )
    with pytest.raises(ReportWorkspaceError):
        load_profile_campaign_policy(
            repo,
            "x86_EPYC",
            expected_source_revision=revision,
        )


def test_legacy_workers_receive_distinct_pinned_shared_object_clones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy-source"
    source.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=source,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=source,
        check=True,
    )
    (source / "makefile").write_text("all:\n\t@true\n", encoding="ascii")
    subprocess.run(("git", "add", "."), cwd=source, check=True)
    subprocess.run(("git", "commit", "-qm", "legacy"), cwd=source, check=True)
    legacy_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class FakeLegacyApi:
        default_repository = source

        def expected_revision(self) -> str:
            return legacy_revision

        def validate_checkout(self, repository: Path) -> None:
            observed = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert observed == legacy_revision
            status = subprocess.run(
                ("git", "status", "--porcelain=v1", "--untracked-files=no"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            assert status == ""

    monkeypatch.setattr(
        "tools.performance_report.legacy.MaintainedLegacyApi",
        FakeLegacyApi,
    )
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=repo, check=True)
    service = ReportService(
        ReportPaths.from_repo(
            repo,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "locks",
        )
    )
    scheduler = CampaignScheduler(service, settings=CampaignSettings())

    first = scheduler._prepare_legacy_workspace("first")
    second = scheduler._prepare_legacy_workspace("second")

    assert first != second
    assert (first / ".git").is_dir()
    assert (second / ".git").is_dir()
    (first / "makefile").write_text("changed\n", encoding="ascii")
    assert (second / "makefile").read_text(encoding="ascii") == "all:\n\t@true\n"


def test_scheduler_publishes_authenticated_generation_censor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=repo, check=True)
    service = ReportService(
        ReportPaths.from_repo(
            repo,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "locks",
        )
    )
    cell = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )

    def fake_supervise(command, **arguments):
        channel = arguments["phase_channel"]
        assert arguments["generation_timeout_seconds"] == 7200.0
        assert channel is not None
        assert "--phase-state-path" in command
        return SupervisedResult(
            returncode=-15,
            reason="generation_timeout",
            usage=ResourceUsage(True, 1, 1, 0, 1.0, 7200.0),
            generation_phase=GenerationPhaseEvidence(
                configured_timeout_seconds=7200.0,
                supervisor_reason="generation_timeout",
                authenticated=True,
                run_id=channel.run_id,
                worker_pid=123,
                final_sequence=1,
                final_phase="generation",
                generation_started_monotonic_ns=1,
                generation_finished_monotonic_ns=None,
                generation_elapsed_seconds=7200.0,
                final_state_sha256="4" * 64,
                error=None,
            ),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    scheduler = CampaignScheduler(service, settings=_x86_settings())

    outcome = scheduler._run_cell(
        PlannedCell(
            cell=cell,
            dependency=False,
            baseline_cell_id=None,
            rank=0,
        )
    )

    assert outcome.status == PolicyMeasurementState.GENERATION_LIMIT.value
    current = service.store.load_current(cell.cell_id)
    assert current is not None
    assert policy_status_label(current.result) == ">2h"
    assert (
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            cell,
            current.result,
            expected_source_revision=scheduler.source_revision,
            expected_source_tree=scheduler.source_tree,
        )
        is PolicyMeasurementState.GENERATION_LIMIT
    )


def test_final_audit_counts_policy_terminal_cells_without_claiming_numerics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.n_final == 1
    )

    class OneCellCatalog:
        matrix_datasets = ()
        process_families = ()
        scalar_datasets = ()
        z_variants = ()
        models: ClassVar[dict[object, object]] = {}

        def measurement_cells(self):
            return (cell,)

        def baseline_cell(self, _cell):
            return None

    catalog = OneCellCatalog()
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "artifacts",
        coordination_root=tmp_path / "locks",
    )
    service = ReportService(paths, catalog=catalog)  # type: ignore[arg-type]
    terminal = _memory_censor(
        cell,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )
    service.store.new_attempt(
        cell.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(terminal)
    caches = build_reset_caches(catalog)  # type: ignore[arg-type]
    for payload in caches.values():
        entry = payload["entries"][0]
        entry["measurement"] = portable_publication_value(terminal, paths)
    paths.results_dir.mkdir(parents=True)
    for name, payload in caches.items():
        (paths.results_dir / name).write_text(
            json.dumps(payload),
            encoding="ascii",
        )
    monkeypatch.setattr(
        service,
        "audit",
        lambda: {"cache_render_match": True},
    )

    result = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=1,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        source_auditor=lambda *_args: None,
        pdf_auditor=lambda _service: {"status": "ok"},
        campaign_policy=X86_EPYC_POLICY,
    )

    assert result["policy_state_counts"] == {"memory_limit": 1}
    assert result["policy_complete_cell_count"] == 1
    assert result["numerically_evidenced_cell_count"] == 0
    visible = result["visible_completeness"]
    assert isinstance(visible, dict)
    assert visible["status"] == "ok"
