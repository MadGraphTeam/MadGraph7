# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.cache import build_reset_caches, empty_measurement
from tools.performance_report.campaign_policy import (
    MACBOOK_M3_MEMORY_LIMIT_BYTES,
    MACBOOK_M3_POLICY,
    STRICT_POLICY,
    X86_EPYC_GENERATION_LIMIT_SECONDS,
    X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES,
    X86_EPYC_LEGACY_WORKERS,
    X86_EPYC_MEMORY_LIMIT_BYTES,
    X86_EPYC_NATIVE_COMPILER_SLOTS,
    X86_EPYC_POLICY,
    X86_EPYC_WORKERS,
    CampaignPolicyError,
    PolicyCensorKind,
    PolicyMeasurementState,
    _validate_generation_phase,
    dependency_reference,
    generation_limit_exempt,
    policy_censor_measurement,
    policy_from_manifest,
    policy_status_label,
    resource_frontier_reference,
    resource_lane_identity,
    validate_policy_measurement,
)
from tools.performance_report.catalog import (
    REPORT_CATALOG,
    STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT,
)
from tools.performance_report.cli import _gb_bytes, _parser
from tools.performance_report.final_audit import FinalAuditError, audit_final_report
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
        "workers": X86_EPYC_WORKERS,
        "cell_cores": 1,
        "target_runtime_seconds": 5.0,
        "max_rss_bytes": X86_EPYC_MEMORY_LIMIT_BYTES,
        "allow_symbolica_parallel": True,
        "campaign_policy": X86_EPYC_POLICY,
        "report_profile": "x86_EPYC",
    }
    values.update(changes)
    return CampaignSettings(**values)  # type: ignore[arg-type]


def _mac_settings(**changes: object) -> CampaignSettings:
    values: dict[str, object] = {
        "workers": 1,
        "cell_cores": 1,
        "target_runtime_seconds": 5.0,
        "max_rss_bytes": MACBOOK_M3_MEMORY_LIMIT_BYTES,
        "campaign_policy": MACBOOK_M3_POLICY,
        "report_profile": "macbook_M3",
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
    full = REPORT_CATALOG.measurement_cells()
    assert len(full) == 1666
    assert sum(generation_limit_exempt(cell) for cell in full) == 669


def test_x86_settings_are_exact_and_use_decimal_80_gb() -> None:
    settings = _x86_settings()

    assert settings.workers == 25
    assert settings.cell_cores == 1
    assert settings.max_rss_bytes == 80_000_000_000
    assert settings.timeout_seconds is None
    with pytest.raises(CampaignPolicyError, match="max_rss_bytes"):
        _x86_settings(max_rss_bytes=80 * 1024**3)
    with pytest.raises(CampaignPolicyError, match="workers"):
        _x86_settings(workers=9)
    with pytest.raises(CampaignPolicyError, match="workers"):
        _x86_settings(workers=X86_EPYC_LEGACY_WORKERS)
    assert CampaignSettings().campaign_policy is STRICT_POLICY
    assert _gb_bytes(80.0) == X86_EPYC_MEMORY_LIMIT_BYTES
    assert X86_EPYC_NATIVE_COMPILER_SLOTS == 4
    parsed = _parser().parse_args(
        (
            "--report-profile",
            "x86_EPYC",
            "populate",
            "--max-ram-gb",
            "80",
        )
    )
    assert parsed.max_ram_gb == 80.0


def test_x86_policy_accepts_only_the_pinned_workers10_manifest_for_continuity() -> None:
    legacy = X86_EPYC_POLICY.as_manifest()
    legacy["workers"] = X86_EPYC_LEGACY_WORKERS
    legacy["memory_limit_bytes"] = X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES

    assert policy_from_manifest(legacy, profile="x86_EPYC") is X86_EPYC_POLICY
    assert X86_EPYC_POLICY.as_manifest()["workers"] == X86_EPYC_WORKERS

    unsupported = dict(legacy)
    unsupported["workers"] = 24
    with pytest.raises(CampaignPolicyError, match="canonical definition"):
        policy_from_manifest(unsupported, profile="x86_EPYC")

    unsupported = dict(legacy)
    unsupported["memory_limit_bytes"] = X86_EPYC_MEMORY_LIMIT_BYTES
    with pytest.raises(CampaignPolicyError, match="canonical definition"):
        policy_from_manifest(unsupported, profile="x86_EPYC")


def test_macbook_policy_is_exact_memory_only_decimal_30_gb() -> None:
    settings = _mac_settings()

    assert settings.workers == 1
    assert settings.cell_cores == 1
    assert settings.target_runtime_seconds == 5.0
    assert settings.max_rss_bytes == 30_000_000_000
    assert settings.generation_time_limit_seconds is None
    assert settings.timeout_seconds is None
    assert settings.allow_symbolica_parallel is False
    with pytest.raises(CampaignPolicyError, match="max_rss_bytes"):
        _mac_settings(max_rss_bytes=30 * 1024**3)
    with pytest.raises(CampaignPolicyError, match="generation_time_limit"):
        _mac_settings(generation_time_limit_seconds=7200.0)

    cell = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )
    peak = MACBOOK_M3_MEMORY_LIMIT_BYTES + 1
    result = policy_censor_measurement(
        MACBOOK_M3_POLICY,
        "macbook_M3",
        cell,
        kind=PolicyCensorKind.MEMORY_LIMIT,
        source_identity=_IDENTITY,
        resources=_resources(peak),
        observed_rss_bytes=peak,
    )
    assert policy_status_label(result) == ">30GB"
    assert (
        validate_policy_measurement(
            MACBOOK_M3_POLICY,
            "macbook_M3",
            cell,
            result,
            expected_source_revision=_REVISION,
            expected_source_tree=_TREE,
        )
        is PolicyMeasurementState.MEMORY_LIMIT
    )


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


def test_completed_reused_artifact_accepts_zero_work_generation_phase() -> None:
    phase_evidence = {
        "abi": "pyamplicol-report-generation-phase-evidence-v1",
        "phase_state_abi": "pyamplicol-report-worker-phase-state-v1",
        "configured_timeout_seconds": X86_EPYC_GENERATION_LIMIT_SECONDS,
        "supervisor_reason": "completed",
        "authenticated": True,
        "run_id": "run-1",
        "worker_pid": 123,
        "final_sequence": 2,
        "final_phase": "post-generation",
        "generation_started_monotonic_ns": 10,
        "generation_finished_monotonic_ns": 10,
        "generation_elapsed_seconds": 0.0,
        "final_state_sha256": "3" * 64,
        "error": None,
    }

    assert (
        _validate_generation_phase(
            phase_evidence,
            X86_EPYC_POLICY,
            expected_reason="completed",
        )
        is phase_evidence
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

    assert policy_status_label(result) == ">80GB"
    assert ">80GB" in _status(result)


def test_resource_frontier_is_lane_bound_and_tamper_evident() -> None:
    source = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )
    target = REPORT_CATALOG.cell(
        "scalar-contact-n3-scalar-contact-contracted"
    )
    root = _memory_censor(
        source,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )
    frontier = resource_frontier_reference(target, source, root)
    result = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        target,
        kind=PolicyCensorKind.RESOURCE_FRONTIER,
        source_identity=_IDENTITY,
        resources=None,
        frontier=frontier,
    )

    assert policy_status_label(result) == "dependency >80GB"
    assert (
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            target,
            result,
            expected_source_revision=_REVISION,
            expected_source_tree=_TREE,
        )
        is PolicyMeasurementState.RESOURCE_FRONTIER
    )
    assert frontier["lane"] == resource_lane_identity(target)
    assert frontier["root"] == {
        "cell_id": source.cell_id,
        "n_final": 2,
        "kind": PolicyCensorKind.MEMORY_LIMIT.value,
        "status": "memory_limit",
        "censor_sha256": root["provenance"]["policy_censor_sha256"],
    }

    other_lane = REPORT_CATALOG.cell(
        "scalar-gravity-n3-scalar-gravity-contracted"
    )
    with pytest.raises(CampaignPolicyError, match="same lane"):
        resource_frontier_reference(other_lane, source, root)


def test_full_catalog_resource_lanes_are_unique_and_monotone(
    tmp_path: Path,
) -> None:
    cells = REPORT_CATALOG.measurement_cells()
    identities = tuple(
        (tuple(resource_lane_identity(cell).items()), cell.n_final)
        for cell in cells
    )
    assert len(cells) == 1666
    assert len(set(identities)) == len(identities)
    assert len({lane for lane, _n_final in identities}) == 306

    selected = tuple(
        REPORT_CATALOG.cell(cell_id)
        for cell_id in (
            "scalar-contact-n2-scalar-contact-contracted",
            "scalar-contact-n3-scalar-contact-contracted",
            "scalar-gravity-n2-scalar-gravity-contracted",
            "scalar-gravity-n3-scalar-gravity-contracted",
        )
    )
    planned = plan_campaign(
        selected,
        store=ArtifactStore(
            artifact_root=tmp_path / "artifacts",
            lock_root=tmp_path / "locks",
        ),
        settings=_x86_settings(),
        expected_revision=_REVISION,
        expected_tree=_TREE,
    )
    ranks = {item.cell.cell_id: item.rank for item in planned}
    assert (
        ranks["scalar-contact-n3-scalar-contact-contracted"]
        > ranks["scalar-contact-n2-scalar-contact-contracted"]
    )
    assert (
        ranks["scalar-gravity-n3-scalar-gravity-contracted"]
        > ranks["scalar-gravity-n2-scalar-gravity-contracted"]
    )
    assert (
        ranks["scalar-contact-n2-scalar-contact-contracted"]
        == ranks["scalar-gravity-n2-scalar-gravity-contracted"]
    )


def test_missing_only_reuses_and_rebinds_resource_frontier(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    source = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )
    target = REPORT_CATALOG.cell(
        "scalar-contact-n3-scalar-contact-contracted"
    )
    first = _memory_censor(
        source,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )
    store.new_attempt(
        source.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(first)
    frontier = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        target,
        kind=PolicyCensorKind.RESOURCE_FRONTIER,
        source_identity=_IDENTITY,
        resources=None,
        frontier=resource_frontier_reference(target, source, first),
    )
    store.new_attempt(
        target.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(frontier)
    settings = _x86_settings(missing_only=True)

    assert plan_campaign(
        (target,),
        store=store,
        settings=settings,
        expected_revision=_REVISION,
        expected_tree=_TREE,
    ) == ()

    second = _memory_censor(
        source,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 2,
    )
    store.new_attempt(
        source.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(second)
    planned = plan_campaign(
        (target,),
        store=store,
        settings=settings,
        expected_revision=_REVISION,
        expected_tree=_TREE,
    )
    assert tuple(item.cell.cell_id for item in planned) == (target.cell_id,)
    assert planned[0].force_recompare is True


def test_workspace_policy_is_bound_to_the_exact_measured_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    profile = repo / "docs" / "performance_reports" / "x86_EPYC"
    profile.mkdir(parents=True)
    legacy_policy = X86_EPYC_POLICY.as_manifest()
    legacy_policy["workers"] = X86_EPYC_LEGACY_WORKERS
    legacy_policy["memory_limit_bytes"] = X86_EPYC_LEGACY_MEMORY_LIMIT_BYTES
    manifest = {
        "schema": WORKSPACE_SCHEMA,
        "profile": "x86_EPYC",
        "campaign_policy": legacy_policy,
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
    (repo / "docs/arxiv").mkdir(parents=True)
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
    (repo / "docs/arxiv").mkdir(parents=True)
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


def test_scheduler_uses_mac_memory_limit_without_generation_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "docs/arxiv").mkdir(parents=True)
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

    def fake_supervise(_command, **arguments):
        assert arguments["generation_timeout_seconds"] is None
        assert arguments["phase_channel"] is None
        assert arguments["max_rss_bytes"] == MACBOOK_M3_MEMORY_LIMIT_BYTES
        peak = MACBOOK_M3_MEMORY_LIMIT_BYTES + 1
        return SupervisedResult(
            returncode=-15,
            reason="memory_limit",
            usage=ResourceUsage(True, peak, peak, 0, 1.0, 2.0),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        fake_supervise,
    )
    scheduler = CampaignScheduler(service, settings=_mac_settings())
    outcome = scheduler._run_cell(
        PlannedCell(
            cell=cell,
            dependency=False,
            baseline_cell_id=None,
            rank=0,
        )
    )

    assert outcome.status == PolicyMeasurementState.MEMORY_LIMIT.value
    current = service.store.load_current(cell.cell_id)
    assert current is not None
    assert policy_status_label(current.result) == ">30GB"


def test_scheduler_never_launches_above_authenticated_resource_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "docs/arxiv").mkdir(parents=True)
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
    scheduler = CampaignScheduler(service, settings=_x86_settings())
    source = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )
    target = REPORT_CATALOG.cell(
        "scalar-contact-n3-scalar-contact-contracted"
    )
    peak = X86_EPYC_MEMORY_LIMIT_BYTES + 1
    root = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        source,
        kind=PolicyCensorKind.MEMORY_LIMIT,
        source_identity=scheduler.source_identity,
        resources=_resources(peak),
        observed_rss_bytes=peak,
    )
    service.store.new_attempt(
        source.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(root)

    def forbidden_supervisor(*_args, **_kwargs):
        raise AssertionError("frontier target must not launch a worker")

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        forbidden_supervisor,
    )
    outcome = scheduler._run_cell(
        PlannedCell(
            cell=target,
            dependency=False,
            baseline_cell_id=None,
            rank=1,
        )
    )

    assert outcome.status == PolicyMeasurementState.RESOURCE_FRONTIER.value
    current = service.store.load_current(target.cell_id)
    assert current is not None
    assert policy_status_label(current.result) == "dependency >80GB"
    assert (
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            target,
            current.result,
            expected_source_revision=scheduler.source_revision,
            expected_source_tree=scheduler.source_tree,
        )
        is PolicyMeasurementState.RESOURCE_FRONTIER
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
    static_cell = REPORT_CATALOG.cell(
        "reference-amplicol-full-n6-dd-4q-lines-contracted"
    )

    class OneCellCatalog:
        matrix_datasets = ()
        process_families = ()
        scalar_datasets = ()
        z_variants = ()
        models: ClassVar[dict[object, object]] = {}

        def measurement_cells(self):
            return (cell, static_cell)

        def baseline_cell(self, _cell):
            return None

        def validation_baseline_cell(self, _cell):
            return None

        def static_na_reason(self, candidate):
            if candidate == static_cell:
                return STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT
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
        for entry in payload["entries"]:
            if entry["cell_id"] == cell.cell_id:
                entry["measurement"] = portable_publication_value(
                    terminal,
                    paths,
                )
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
        max_n_final=6,
        expected_cell_count=2,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        source_auditor=lambda *_args: None,
        pdf_auditor=lambda _service: {"status": "ok"},
        campaign_policy=X86_EPYC_POLICY,
    )

    assert result["policy_state_counts"] == {"memory_limit": 1}
    assert result["policy_complete_cell_count"] == 1
    assert result["numerically_evidenced_cell_count"] == 0
    assert result["declared_cell_count"] == 2
    assert result["measurable_cell_count"] == 1
    assert result["catalog_static_na_cell_count"] == 1
    assert result["catalog_static_na_reason_counts"] == {
        STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT: 1,
    }
    assert result["authenticated_current_count"] == 1
    visible = result["visible_completeness"]
    assert isinstance(visible, dict)
    assert visible["status"] == "ok"
    assert visible["declared_measurement_cell_count"] == 2
    assert visible["required_measurement_count"] == 1
    assert visible["rendered_required_measurement_count"] == 1
    assert visible["catalog_static_na_cell_count"] == 1
    assert visible["rendered_catalog_static_na_cell_count"] == 1

    service.store.new_attempt(
        static_cell.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(empty_measurement())
    with pytest.raises(
        FinalAuditError,
        match="catalog static N/A cell has a published current",
    ):
        audit_final_report(
            tmp_path,
            expected_source_revision=_REVISION,
            max_n_final=6,
            expected_cell_count=2,
            catalog=catalog,  # type: ignore[arg-type]
            service=service,
            source_auditor=lambda *_args: None,
            verify_pdf=False,
            campaign_policy=X86_EPYC_POLICY,
        )


def test_final_audit_requires_exact_monotone_resource_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = REPORT_CATALOG.cell(
        "scalar-contact-n2-scalar-contact-contracted"
    )
    target = REPORT_CATALOG.cell(
        "scalar-contact-n3-scalar-contact-contracted"
    )

    class TwoCellCatalog:
        matrix_datasets = ()
        process_families = ()
        scalar_datasets = ()
        z_variants = ()
        models: ClassVar[dict[object, object]] = {}

        def measurement_cells(self):
            return (source, target)

        def baseline_cell(self, _cell):
            return None

        def validation_baseline_cell(self, _cell):
            return None

        def static_na_reason(self, _cell):
            return None

    catalog = TwoCellCatalog()
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "artifacts",
        coordination_root=tmp_path / "locks",
    )
    service = ReportService(paths, catalog=catalog)  # type: ignore[arg-type]
    root = _memory_censor(
        source,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 1,
    )
    frontier = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        target,
        kind=PolicyCensorKind.RESOURCE_FRONTIER,
        source_identity=_IDENTITY,
        resources=None,
        frontier=resource_frontier_reference(target, source, root),
    )

    def publish(measurements: tuple[dict[str, object], ...]) -> None:
        for cell, measurement in zip(
            (source, target),
            measurements,
            strict=True,
        ):
            service.store.new_attempt(
                cell.cell_id,
                ArtifactPolicy.REGENERATE,
            ).publish(measurement)
        caches = build_reset_caches(catalog)  # type: ignore[arg-type]
        for payload in caches.values():
            entries = payload["entries"]
            assert isinstance(entries, list)
            for entry in entries:
                cell_id = str(entry["cell_id"])
                measurement = measurements[
                    0 if cell_id == source.cell_id else 1
                ]
                entry["measurement"] = portable_publication_value(
                    measurement,
                    paths,
                )
        paths.results_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in caches.items():
            (paths.results_dir / name).write_text(
                json.dumps(payload),
                encoding="ascii",
            )

    publish((root, frontier))
    monkeypatch.setattr(
        service,
        "audit",
        lambda: {"cache_render_match": True},
    )
    result = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=3,
        expected_cell_count=2,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        source_auditor=lambda *_args: None,
        pdf_auditor=lambda _service: {"status": "ok"},
        campaign_policy=X86_EPYC_POLICY,
    )
    assert result["hard_resource_censor_count"] == 1
    assert result["resource_frontier_cell_count"] == 1

    second_root = _memory_censor(
        target,
        peak=X86_EPYC_MEMORY_LIMIT_BYTES + 2,
    )
    publish((root, second_root))
    with pytest.raises(
        FinalAuditError,
        match="higher multiplicity is not bound to resource frontier",
    ):
        audit_final_report(
            tmp_path,
            expected_source_revision=_REVISION,
            max_n_final=3,
            expected_cell_count=2,
            catalog=catalog,  # type: ignore[arg-type]
            service=service,
            source_auditor=lambda *_args: None,
            pdf_auditor=lambda _service: {"status": "ok"},
            campaign_policy=X86_EPYC_POLICY,
        )
