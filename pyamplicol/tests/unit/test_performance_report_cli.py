# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report import final_audit
from tools.performance_report.campaign_policy import (
    MACBOOK_M3_POLICY,
    MACBOOK_M3_Z_TABLE_F_POLICY_NAME,
    X86_EPYC_POLICY,
    CampaignPolicyError,
    PolicyMeasurementState,
    validate_policy_measurement,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.cli import (
    _compile_pdf,
    _is_pinned_epyc_orphan_without_rss,
    _launch_async_publication,
    _parser,
    main,
)
from tools.performance_report.service import ReportPaths, ReportService


def _initialize_git_repo(repo: Path) -> None:
    (repo / "src/pyamplicol/_profiling_campaign/results").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-tests@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Tests"),
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("# report fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "README.md"), cwd=repo, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "Initialize fixture"),
        cwd=repo,
        check=True,
    )


def _fake_latexmk(tmp_path: Path, log: str) -> Path:
    executable = tmp_path / "fake-latexmk"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "assert not Path('campaign_artifacts').exists()\n"
        "Path('pyAmpliCol.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        f"Path('pyAmpliCol.log').write_text({log!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_table_filler_defaults_to_five_seconds_per_cell() -> None:
    populate = _parser().parse_args(("populate",))
    assert populate.target_runtime == 5.0
    assert populate.generation_time_limit_seconds is None
    assert populate.fast_lineage is False
    assert populate.exclude_cell_id == []

    publisher = _parser().parse_args(("publish-snapshot",))
    assert publisher.watch is False
    assert publisher.interval_seconds == 600.0
    assert publisher.pdf_timeout_seconds == 900.0
    assert publisher.expected_page_count == 73

    worker = _parser().parse_args(
        (
            "_worker",
            "--cell-id",
            "cell",
            "--attempt-root",
            "attempt",
            "--result-json",
            "result.json",
            "--memory-limit-bytes",
            "15000000000",
            "--madgraph",
            "/tmp/madgraph",
        )
    )
    assert worker.target_runtime == 5.0
    assert worker.memory_limit_bytes == 15_000_000_000
    assert worker.madgraph == Path("/tmp/madgraph")

    limited = _parser().parse_args(
        ("populate", "--generation-time-limit-seconds", "7200")
    )
    assert limited.generation_time_limit_seconds == 7200.0

    decimal_limits = _parser().parse_args(
        (
            "populate",
            "--max-ram-gb",
            "30",
            "--campaign-max-ram-gb",
            "30",
        )
    )
    assert decimal_limits.max_ram_gb == 30.0
    assert decimal_limits.campaign_max_ram_gb == 30.0

    study = _parser().parse_args(
        (
            "populate",
            "--study-policy",
            MACBOOK_M3_Z_TABLE_F_POLICY_NAME,
            "--max-ram-gb",
            "30",
        )
    )
    assert study.study_policy == MACBOOK_M3_Z_TABLE_F_POLICY_NAME
    assert study.max_ram_gb == 30.0


def test_worker_cli_threads_madgraph_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = REPORT_CATALOG.measurement_cells()[0]
    installation = tmp_path / "madgraph"
    captured: dict[str, object] = {}

    def write(
        cell_id: str,
        result_path: Path,
        **arguments: object,
    ) -> dict[str, object]:
        captured["cell_id"] = cell_id
        captured["result_path"] = result_path
        captured.update(arguments)
        return {"status": "ok"}

    monkeypatch.setattr("tools.performance_report.cli.write_cell_result", write)
    assert (
        main(
            (
                "--repo-root",
                str(tmp_path),
                "_worker",
                "--cell-id",
                cell.cell_id,
                "--attempt-root",
                str(tmp_path / "attempt"),
                "--result-json",
                str(tmp_path / "result.json"),
                "--madgraph",
                str(installation),
            )
        )
        == 0
    )

    assert captured["cell_id"] == cell.cell_id
    assert captured["madgraph_installation"] == installation


def test_async_publication_can_run_the_authenticated_wrapper_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = tmp_path / "measured"
    measured.mkdir()
    wrapper_entrypoint = (
        tmp_path / "wrapper/src/pyamplicol/_profiling_campaign/result_tables.py"
    )
    wrapper_entrypoint.parent.mkdir(parents=True)
    wrapper_entrypoint.write_text("# authenticated wrapper\n", encoding="ascii")
    service = ReportService(
        ReportPaths.from_repo(
            measured,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    captured: dict[str, object] = {}

    def popen(command, **arguments):
        captured["command"] = tuple(command)
        captured.update(arguments)
        return SimpleNamespace()

    monkeypatch.setattr("tools.performance_report.cli.subprocess.Popen", popen)

    log_path = _launch_async_publication(
        service,
        entrypoint=wrapper_entrypoint,
    )

    command = captured["command"]
    assert isinstance(command, tuple)
    assert Path(command[4]) == wrapper_entrypoint.resolve()
    assert command[command.index("--repo-root") + 1] == str(
        measured.resolve()
    )
    assert command[-1] == "publish-snapshot"
    assert log_path == (
        service.paths.artifact_root
        / "publication-logs/refresh-pdf-end.log"
    )


def test_pinned_epyc_orphan_is_the_only_unavailable_rss_exception() -> None:
    cell = REPORT_CATALOG.cell(
        "reference-amplicol-lc-n8-gg-gluons-selected-flow"
    )
    measurement = {
        "status": "ok",
        "generation_seconds": 1.0,
        "resources": {
            "monitor": "external-cell-supervisor",
            "peak_rss_gib": None,
        },
        "provenance": {
            "report_source_identity_schema": "pyamplicol-report-source-v1",
            "report_source_revision": "1" * 40,
            "report_source_tree": "2" * 40,
            "report_measured_source_revision": "1" * 40,
            "report_measured_source_tree": "2" * 40,
            "report_source_clean": True,
        },
    }
    identity = {
        "profile": "x86_EPYC",
        "cell_id": cell.cell_id,
        "attempt_id": "83e5c9c7-dbf6-4d61-b724-f4580df2cfa3",
        "worker_result_sha256": (
            "5f3a42f9e3d034efedd8b670e7acbf2b54a427449106dbabc29050f3d93afbe6"
        ),
        "result": measurement,
    }

    assert _is_pinned_epyc_orphan_without_rss(**identity)
    with pytest.raises(CampaignPolicyError, match="resource monitoring"):
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            cell,
            measurement,
            expected_source_revision="1" * 40,
            expected_source_tree="2" * 40,
        )
    assert (
        validate_policy_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            cell,
            measurement,
            expected_source_revision="1" * 40,
            expected_source_tree="2" * 40,
            allow_pinned_orphan_unavailable_resources=True,
        )
        is PolicyMeasurementState.SUCCESS
    )

    for field, value in (
        ("profile", "macbook_M3"),
        ("cell_id", "reference-amplicol-lc-n7-gg-gluons-selected-flow"),
        ("attempt_id", "00000000-0000-4000-8000-000000000000"),
        ("worker_result_sha256", "0" * 64),
    ):
        changed = dict(identity)
        changed[field] = value
        assert not _is_pinned_epyc_orphan_without_rss(**changed)

    resources = measurement["resources"]
    assert isinstance(resources, dict)
    for tampered in (
        {"monitor": "external-cell-supervisor"},
        {"monitor": "external-cell-supervisor", "peak_rss_gib": 0.0},
        {"monitor": "other", "peak_rss_gib": None},
        {
            "monitor": "external-cell-supervisor",
            "peak_rss_gib": None,
            "peak_rss_bytes": 0,
        },
    ):
        changed_measurement = dict(measurement)
        changed_measurement["resources"] = tampered
        changed = dict(identity)
        changed["result"] = changed_measurement
        assert not _is_pinned_epyc_orphan_without_rss(**changed)
        with pytest.raises(CampaignPolicyError, match="resource monitoring"):
            validate_policy_measurement(
                X86_EPYC_POLICY,
                "x86_EPYC",
                cell,
                changed_measurement,
                expected_source_revision="1" * 40,
                expected_source_tree="2" * 40,
                allow_pinned_orphan_unavailable_resources=True,
            )


def test_report_profile_is_a_global_architecture_scope() -> None:
    parsed = _parser().parse_args(("--report-profile", "macbook_M3", "populate"))
    assert parsed.report_profile == "macbook_M3"


def test_final_audit_is_routed_through_the_isolated_result_tables_entrypoint() -> None:
    arguments = _parser().parse_args(
        (
            "--report-profile",
            "macbook_M3",
            "final-audit",
            "--expected-source-revision",
            "a" * 40,
            "--publication-revision",
            "b" * 40,
            "--structural-only",
        )
    )

    assert arguments.command == "final-audit"
    assert arguments.report_profile == "macbook_M3"
    assert arguments.expected_source_revision == "a" * 40
    assert arguments.publication_revision == "b" * 40
    assert arguments.max_n_final == 9
    assert arguments.expected_cell_count == 2162
    assert arguments.structural_only is True


def test_final_audit_receives_the_bound_architecture_profile_service(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    profile = "macbook_M3"
    expected_paths = ReportPaths.from_repo(repo, profile=profile)
    expected_service = ReportService(expected_paths)
    observed: dict[str, object] = {}
    environment_checks: list[tuple[Path, str, str]] = []

    def construct_service(paths: ReportPaths) -> ReportService:
        assert paths == expected_paths
        return expected_service

    def fake_audit(
        repo_root: Path,
        **arguments: object,
    ) -> dict[str, object]:
        observed["repo_root"] = repo_root
        observed.update(arguments)
        return {"final_gate_complete": True}

    monkeypatch.setattr(
        "tools.performance_report.cli.ReportService",
        construct_service,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.require_active_profile_environment",
        lambda root, active_profile, *, expected_source_revision: (
            environment_checks.append((root, active_profile, expected_source_revision))
            or {}
        ),
    )
    monkeypatch.setattr(final_audit, "audit_final_report", fake_audit)

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "--report-profile",
                profile,
                "final-audit",
                "--expected-source-revision",
                "a" * 40,
                "--publication-revision",
                "b" * 40,
                "--structural-only",
            )
        )
        == 0
    )

    assert observed["repo_root"] == repo.resolve()
    assert observed["service"] is expected_service
    assert expected_service.paths.docs_dir == (
        repo.resolve() / "docs/performance_reports/macbook_M3"
    )
    assert expected_service.paths.artifact_root == (
        repo.resolve() / ".artifacts/performance-report/macbook_M3"
    )
    assert expected_service.paths.coordination_root == (
        repo.resolve() / ".artifacts/performance-report-coordination/macbook_M3"
    )
    assert observed["expected_source_revision"] == "a" * 40
    assert observed["expected_publication_revision"] == "b" * 40
    assert observed["max_n_final"] == 9
    assert observed["expected_cell_count"] == 2162
    assert observed["replay"] is False
    assert environment_checks == [(repo.resolve(), profile, "a" * 40)]
    assert json.loads(capsys.readouterr().out)["final_gate_complete"] is True


def test_refresh_environment_is_routed_to_the_active_profile(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    observed: dict[str, object] = {}

    def fake_refresh(
        root: Path,
        profile: str,
        *,
        expected_source_revision: str,
    ) -> dict[str, str]:
        observed.update(
            {
                "root": root,
                "profile": profile,
                "expected_source_revision": expected_source_revision,
            }
        )
        return {
            "status": "authenticated",
            "source_revision": expected_source_revision,
        }

    monkeypatch.setattr(
        "tools.performance_report.cli.refresh_profile_environment",
        fake_refresh,
    )

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "--report-profile",
                "macbook_M3",
                "refresh-profile-environment",
                "--expected-source-revision",
                "a" * 40,
            )
        )
        == 0
    )

    assert observed == {
        "root": repo.resolve(),
        "profile": "macbook_M3",
        "expected_source_revision": "a" * 40,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "authenticated",
        "source_revision": "a" * 40,
    }


def test_final_audit_requires_an_architecture_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called = False

    def fake_audit(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"final_gate_complete": True}

    monkeypatch.setattr(final_audit, "audit_final_report", fake_audit)

    with pytest.raises(SystemExit, match="2"):
        main(
            (
                "--repo-root",
                str(tmp_path / "repo"),
                "final-audit",
                "--expected-source-revision",
                "a" * 40,
            )
        )

    assert called is False


def test_refresh_pdf_rejects_successful_latex_with_overfull_box(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    service = ReportService(ReportPaths.from_repo(repo))
    service.paths.docs_dir.mkdir(parents=True, exist_ok=True)
    (service.paths.docs_dir / "campaign_artifacts").mkdir()
    (service.paths.docs_dir / "campaign_artifacts/sentinel").write_text(
        "private\n",
        encoding="ascii",
    )
    (service.paths.docs_dir / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}bad\\end{document}\n",
        encoding="ascii",
    )
    executable = _fake_latexmk(
        tmp_path,
        "Overfull \\hbox (1.0pt too wide)\n",
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.shutil.which",
        lambda _name: str(executable),
    )

    with pytest.raises(RuntimeError, match="overfull"):
        _compile_pdf(service)


def test_reset_and_validate_cli_use_new_service(tmp_path: Path, capsys) -> None:
    repo = tmp_path / "repo"
    _initialize_git_repo(repo)

    assert main(("--repo-root", str(repo), "reset")) == 0
    reset_output = capsys.readouterr().out
    assert (
        "src/pyamplicol/_profiling_campaign/"
        "result_matrix_recurrence_builtin_sm_lc_table.tex"
        in reset_output
    )

    assert main(("--repo-root", str(repo), "validate")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["table_count"] == 27
    assert payload["cache_count"] > 12

    assert main(("--repo-root", str(repo), "audit")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_render_match"]


def test_populate_dry_run_supports_exact_filters_and_dependencies(
    tmp_path: Path,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    _initialize_git_repo(repo)
    main(("--repo-root", str(repo), "reset"))
    capsys.readouterr()
    subprocess.run(
        ("git", "add", "src/pyamplicol/_profiling_campaign"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "commit", "-q", "-m", "Track reset profile"),
        cwd=repo,
        check=True,
    )

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "populate",
                "--dataset",
                "matrix_compiled_builtin_sm_lc",
                "--process-key",
                "dd_z_jets",
                "--n-final",
                "1",
                "--workload",
                "selected-flow",
                "--dry-run",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested"] == 1
    assert payload["scheduled"] == 1
    assert [cell["rank"] for cell in payload["cells"]] == [2]


def test_epyc_workers25_dry_run_never_creates_an_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    _initialize_git_repo(repo)
    monkeypatch.setattr(
        "tools.performance_report.cli.load_profile_campaign_policy",
        lambda *_args, **_kwargs: X86_EPYC_POLICY,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.load_and_audit_measurement_lineage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.artifacts.ArtifactStore.new_attempt",
        lambda *_args, **_kwargs: pytest.fail("dry-run created an attempt"),
    )
    command = (
        "--repo-root",
        str(repo),
        "--report-profile",
        "x86_EPYC",
        "populate",
        "--dataset",
        "matrix_compiled_builtin_sm_lc",
        "--process-key",
        "dd_z_jets",
        "--n-final",
        "1",
        "--workload",
        "selected-flow",
        "--workers",
        "25",
        "--cell-cores",
        "1",
        "--target-runtime",
        "5",
        "--max-ram-gb",
        "80",
        "--allow-symbolica-parallel",
        "--dry-run",
    )

    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["scheduled"] == 1

    stale = list(command)
    stale[stale.index("25")] = "10"
    with pytest.raises(SystemExit):
        main(tuple(stale))
    assert "workers=10, expected 25" in capsys.readouterr().err

    stale = list(command)
    stale[stale.index("80")] = "100"
    with pytest.raises(SystemExit):
        main(tuple(stale))
    assert (
        "max_rss_bytes=100000000000, expected 80000000000"
        in capsys.readouterr().err
    )


def test_profile_population_requires_the_active_authenticated_environment(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    _initialize_git_repo(repo)
    expected_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checked: list[tuple[Path, str, str]] = []
    scheduler_settings: list[object] = []

    def require_environment(
        root: Path,
        profile: str,
        *,
        expected_source_revision: str,
    ) -> dict[str, str]:
        checked.append((root, profile, expected_source_revision))
        return {}

    class FakeScheduler:
        def __init__(self, _service, *, settings) -> None:
            self.settings = settings
            scheduler_settings.append(settings)

        def run(self, planned):
            return SimpleNamespace(
                planned=planned,
                outcomes=(),
                failed=(),
            )

    monkeypatch.setattr(
        "tools.performance_report.cli.require_active_profile_environment",
        require_environment,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.load_profile_campaign_policy",
        lambda *_args, **_kwargs: MACBOOK_M3_POLICY,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.CampaignScheduler",
        FakeScheduler,
    )

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "--report-profile",
                "macbook_M3",
                "populate",
                "--dataset",
                "matrix_compiled_builtin_sm_lc",
                "--process-key",
                "dd_z_jets",
                "--n-final",
                "1",
                "--workload",
                "selected-flow",
                "--max-ram-gb",
                "30",
            )
        )
        == 0
    )

    assert checked == [(repo.resolve(), "macbook_M3", expected_revision)]
    assert len(scheduler_settings) == 1
    assert scheduler_settings[0].max_rss_bytes == 30_000_000_000
    assert scheduler_settings[0].campaign_max_rss_bytes is None
    assert json.loads(capsys.readouterr().out)["planned"] == 1


def test_profile_fast_lineage_skips_historical_replay(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    _initialize_git_repo(repo)
    expected_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_tree = subprocess.run(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lineage = SimpleNamespace()
    loaded: list[tuple[Path, Path, str, str]] = []
    bound: list[object] = []

    def load_fast(
        root: Path,
        docs_dir: Path,
        *,
        expected_active_revision: str,
        expected_active_tree: str,
    ):
        loaded.append(
            (
                root,
                docs_dir,
                expected_active_revision,
                expected_active_tree,
            )
        )
        return lineage

    class FakeScheduler:
        def __init__(self, service, *, settings) -> None:
            assert settings.report_profile == "macbook_M3"
            bound.append(service._authenticated_measurement_lineage)

        def run(self, planned):
            return SimpleNamespace(planned=planned, outcomes=(), failed=())

    monkeypatch.setattr(
        "tools.performance_report.cli.load_measurement_lineage",
        load_fast,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.load_and_audit_measurement_lineage",
        lambda *_args, **_kwargs: pytest.fail("historical replay was called"),
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.require_active_profile_environment",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.load_profile_campaign_policy",
        lambda *_args, **_kwargs: MACBOOK_M3_POLICY,
    )
    monkeypatch.setattr(
        "tools.performance_report.cli.CampaignScheduler",
        FakeScheduler,
    )

    assert (
        main(
            (
                "--repo-root",
                str(repo),
                "--report-profile",
                "macbook_M3",
                "populate",
                "--fast-lineage",
                "--dataset",
                "matrix_compiled_builtin_sm_lc",
                "--process-key",
                "dd_z_jets",
                "--n-final",
                "1",
                "--workload",
                "selected-flow",
                "--max-ram-gb",
                "30",
            )
        )
        == 0
    )

    assert loaded == [
        (
            repo.resolve(),
            repo / "docs/performance_reports/macbook_M3",
            expected_revision,
            expected_tree,
        )
    ]
    assert bound == [lineage]
    assert json.loads(capsys.readouterr().out)["planned"] == 1
