# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report import final_audit
from tools.performance_report.campaign_policy import STRICT_POLICY
from tools.performance_report.cli import _compile_pdf, _parser, main
from tools.performance_report.service import ReportPaths, ReportService


def _initialize_git_repo(repo: Path) -> None:
    (repo / "docs/results").mkdir(parents=True)
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

    worker = _parser().parse_args(
        (
            "_worker",
            "--cell-id",
            "cell",
            "--attempt-root",
            "attempt",
            "--result-json",
            "result.json",
        )
    )
    assert worker.target_runtime == 5.0

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
    assert arguments.expected_cell_count == 1646
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
    assert observed["expected_cell_count"] == 1646
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
    assert "docs/result_matrix_recurrence_builtin_sm_lc_table.tex" in reset_output

    assert main(("--repo-root", str(repo), "validate")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["table_count"] == 20
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
    assert payload["scheduled"] == 3
    assert [cell["rank"] for cell in payload["cells"]] == [0, 1, 2]


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
        lambda *_args, **_kwargs: STRICT_POLICY,
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
                "--campaign-max-ram-gb",
                "30",
            )
        )
        == 0
    )

    assert checked == [(repo.resolve(), "macbook_M3", expected_revision)]
    assert len(scheduler_settings) == 1
    assert scheduler_settings[0].max_rss_bytes == 30_000_000_000
    assert scheduler_settings[0].campaign_max_rss_bytes == 30_000_000_000
    assert json.loads(capsys.readouterr().out)["planned"] == 3
