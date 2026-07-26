# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.performance_report import workspace as workspace_module
from tools.performance_report.artifacts import LockTimeoutError
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.service import (
    ReportPaths,
    ReportService,
    validate_profile_name,
)
from tools.performance_report.source_identity import (
    inspect_report_source,
    require_report_only_publication,
)
from tools.performance_report.workspace import (
    ENVIRONMENT_JSON,
    ENVIRONMENT_SCHEMA,
    ENVIRONMENT_TEX,
    STANDALONE_BUILDER,
    WORKSPACE_MANIFEST,
    ReportWorkspaceError,
    export_profile,
    initialize_profile,
    refresh_profile_environment,
    require_active_profile_environment,
    require_authenticated_profile_environment,
)


def _seed_template(repo: Path) -> None:
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}report\\end{document}\n",
        encoding="ascii",
    )
    (docs / "result_tables.py").write_text(
        "#!/usr/bin/env python3\n",
        encoding="ascii",
    )
    ReportService(ReportPaths.from_repo(repo)).publish(
        reset=True,
        merge_artifacts=False,
    )
    (docs / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")


def test_profile_names_and_paths_are_safe_and_machine_isolated(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    mac = ReportPaths.from_repo(repo, profile="macbook_M3")
    cluster = ReportPaths.from_repo(repo, profile="cluster_EPYC")

    assert mac.docs_dir == repo / "docs/performance_reports/macbook_M3"
    assert mac.artifact_root == repo / ".artifacts/performance-report/macbook_M3"
    assert mac.coordination_root == (
        repo / ".artifacts/performance-report-coordination/macbook_M3"
    )
    assert mac.results_dir == mac.docs_dir / "results"
    assert mac.artifact_root != cluster.artifact_root
    assert mac.coordination_root != cluster.coordination_root
    assert validate_profile_name("cluster_EPYC") == "cluster_EPYC"

    for invalid in ("", "../escape", "a..b", "/absolute", "white space"):
        with pytest.raises(ValueError, match="report profile"):
            ReportPaths.from_repo(repo, profile=invalid)


def test_initialize_profile_copies_publication_data_but_not_local_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    template_state = repo / ".artifacts/performance-report/cells/attempt"
    template_state.mkdir(parents=True)
    (template_state / "worker.log").write_text("private\n", encoding="ascii")
    coordination = repo / "docs/results/.coordination"
    coordination.mkdir(exist_ok=True)
    (coordination / "writer.lock").write_text("", encoding="ascii")

    profile = initialize_profile(repo, "macbook_M3")

    assert (profile / "pyAmpliCol.tex").is_file()
    assert not (profile / "pyAmpliCol.pdf").exists()
    assert (profile / "result_tables.py").is_file()
    assert (profile / STANDALONE_BUILDER).is_file()
    environment_tex = (profile / ENVIRONMENT_TEX).read_text()
    assert r"\renewcommand{\ReportProfileName}{macbook\_M3}" in environment_tex
    assert r"\renewcommand{\ReportPlatformSummary}" in environment_tex
    assert r"\renewcommand{\ReportToolchainSummary}" in environment_tex
    assert r"\renewcommand{\ReportEditionStatement}" in environment_tex
    assert "pending authenticated post-checkpoint build" in environment_tex
    assert "source checkout" not in environment_tex
    environment = json.loads((profile / ENVIRONMENT_JSON).read_text())
    assert environment["schema"] == ENVIRONMENT_SCHEMA
    assert environment["status"] == "pending_exact_runtime"
    assert environment["source_revision"] == "pending"
    assert "source checkout" not in environment["pyamplicol"]
    assert (profile / "results/report-cache.schema.json").is_file()
    assert not (profile / ".artifacts").exists()
    assert not (profile / "results/.coordination").exists()
    manifest = json.loads((profile / WORKSPACE_MANIFEST).read_text())
    assert manifest["profile"] == "macbook_M3"
    assert manifest["measurement_state"] == "copied"
    assert manifest["initialized_environment"]["profile"] == "macbook_M3"
    assert (
        manifest["initialized_environment"]["status"]
        == "pending_exact_runtime"
    )
    assert manifest["environment_json"] == ENVIRONMENT_JSON
    assert manifest["initialized_source_identity"] == {
        "schema": "pyamplicol-report-source-v1",
        "revision": "unknown",
        "tree": "unknown",
        "clean": False,
        "dirty_paths": [],
    }
    assert manifest["artifact_root"] == (
        ".artifacts/performance-report/macbook_M3"
    )
    assert ReportService(
        ReportPaths.from_repo(repo, profile="macbook_M3")
    ).audit()["cache_render_match"]

    with pytest.raises(ReportWorkspaceError, match="already exists"):
        initialize_profile(repo, "macbook_M3")


def test_profile_readme_requires_four_audited_five_second_campaigns(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)

    profile = initialize_profile(repo, "macbook_M3")
    readme = (profile / "README.md").read_text(encoding="utf-8")

    for multiplicity in range(1, 5):
        assert f"--n-final {multiplicity} --missing-only --artifact-policy reuse" in (
            readme
        )
    assert readme.count("--target-runtime 5 --refresh-pdf end") == 4
    assert readme.count("result_tables.py audit") == 5
    assert "1..4" in readme
    assert "--artifact-policy regenerate" not in readme
    assert "visually review" in readme
    assert 'MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"' in readme
    assert 'PUBLICATION_REVISION="$(git rev-parse HEAD)"' in readme
    assert (
        "python3 docs/performance_reports/macbook_M3/result_tables.py "
        "final-audit \\" in readme
    )
    assert (
        '--expected-source-revision "$MEASURED_SOURCE_REVISION" \\\n'
        '  --publication-revision "$PUBLICATION_REVISION" &&\n'
        "git push origin HEAD"
    ) in readme
    assert "python3 docs/result_tables.py final-audit" not in readme
    assert readme.count("git push origin HEAD") == 2
    checkpoint_push = readme.index("git push origin HEAD")
    publication_push = readme.index("git push origin HEAD", checkpoint_push + 1)
    assert readme.index("MEASURED_SOURCE_REVISION=") < readme.index(
        'test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"'
    )
    assert readme.index(
        'test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"'
    ) < checkpoint_push
    assert checkpoint_push < readme.index("refresh-profile-environment")
    assert readme.index("PUBLICATION_REVISION=") < readme.index(" final-audit \\")
    assert publication_push > readme.index(" final-audit \\")
    for path in (
        "report_environment.json",
        "report_environment.tex",
        "results/*.json",
        "result_*_table.tex",
        "result_validation_summary.tex",
        "pyAmpliCol.pdf",
    ):
        assert f"docs/performance_reports/macbook_M3/{path}" in readme


def test_root_report_readme_requires_separate_mac_and_cluster_campaigns() -> None:
    readme = (
        Path(__file__).resolve().parents[2] / "docs/README.md"
    ).read_text(encoding="utf-8")

    for profile in ("macbook_M3", "cluster_EPYC"):
        for multiplicity in range(1, 5):
            command = (
                f"{profile}/result_tables.py populate \\\n"
                f"  --n-final {multiplicity} --missing-only "
                "--artifact-policy reuse"
            )
            assert command in readme
    assert readme.count("--target-runtime 5 --refresh-pdf end") == 8
    assert "--n-final 1..4" not in readme
    assert "--artifact-policy regenerate" not in readme
    assert "visually review the refreshed PDF after every" in readme
    assert "python3 docs/result_tables.py final-audit" not in readme
    assert readme.count('MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"') == 2
    assert readme.count('PUBLICATION_REVISION="$(git rev-parse HEAD)"') == 2
    assert readme.count(
        '--expected-source-revision "$MEASURED_SOURCE_REVISION" \\'
    ) == 2
    assert readme.count(
        '--publication-revision "$PUBLICATION_REVISION" &&\n'
        "git push origin HEAD"
    ) == 2
    assert readme.count(
        'test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION" &&\n'
        "git push origin HEAD"
    ) == 2
    assert readme.count("git push origin HEAD") == 4

    mac_start = readme.index("Create the first Mac workspace")
    cluster_start = readme.index("Create an independent cluster workspace")
    for profile, lifecycle in (
        ("macbook_M3", readme[mac_start:cluster_start]),
        ("cluster_EPYC", readme[cluster_start:]),
    ):
        final_command = (
            f"python3 docs/performance_reports/{profile}/result_tables.py "
            "final-audit \\"
        )
        assert final_command in lifecycle
        assert lifecycle.count("git push origin HEAD") == 2
        checkpoint_push = lifecycle.index("git push origin HEAD")
        publication_push = lifecycle.index(
            "git push origin HEAD",
            checkpoint_push + 1,
        )
        assert lifecycle.index("MEASURED_SOURCE_REVISION=") < lifecycle.index(
            'test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"'
        )
        assert lifecycle.index(
            'test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"'
        ) < checkpoint_push
        assert checkpoint_push < lifecycle.index("refresh-profile-environment")
        assert lifecycle.index("PUBLICATION_REVISION=") > lifecycle.index(
            f'git commit -m "Publish {profile} performance report"'
        )
        assert lifecycle.index("PUBLICATION_REVISION=") < lifecycle.index(
            final_command
        )
        assert publication_push > lifecycle.index(final_command)
        for path in (
            "report_environment.json",
            "report_environment.tex",
            "results/*.json",
            "result_*_table.tex",
            "result_validation_summary.tex",
            "pyAmpliCol.pdf",
        ):
            assert f"docs/performance_reports/{profile}/{path}" in lifecycle


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-q", "-m", message), cwd=repo, check=True)
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_environment_refresh_authenticates_runtime_without_dirtying_source(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
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
    profile = initialize_profile(repo, "macbook_M3")
    measured = _commit_all(repo, "Initialize measured-source scaffold")
    observed: list[tuple[str, Path]] = []

    def runtime_auditor(revision: str, checkout: Path) -> dict[str, object]:
        observed.append((revision, checkout))
        return {
            "package_version": "0.1.0",
            "native_build_inputs_sha256": "a" * 64,
            "native_extension": {"sha256": "b" * 64},
            "python_package_tree": {"sha256": "c" * 64},
            "candidate_build_identity": {
                "candidate_fingerprint": "candidate-aarch64"
            },
            "native_target": {
                "triple": "aarch64-apple-darwin",
                "cpu_features": ["neon", "fp-armv8"],
            },
        }

    environment = refresh_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
        runtime_auditor=runtime_auditor,
    )

    assert observed == [(measured, repo.resolve())]
    assert environment["status"] == "authenticated"
    assert environment["source_revision"] == measured
    assert environment["pyamplicol"] == "0.1.0"
    assert environment["native_target"] == "aarch64-apple-darwin"
    assert environment["native_cpu_features"] == "neon, fp-armv8"
    assert environment["native_extension_sha256"] == "b" * 64
    assert environment["python_package_tree_sha256"] == "c" * 64
    assert environment["candidate_fingerprint"] == "candidate-aarch64"
    assert "source checkout" not in json.dumps(environment)
    assert require_authenticated_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
    ) == environment
    assert require_active_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
        runtime_auditor=runtime_auditor,
    ) == environment
    source = inspect_report_source(repo)
    assert source.eligible
    assert source.revision == measured

    publication = _commit_all(repo, "Publish authenticated environment")
    lineage = require_report_only_publication(
        repo,
        measured_revision=measured,
        profile="macbook_M3",
        publication_revision=publication,
    )
    assert lineage.eligible
    assert lineage.changed_paths == (
        "docs/performance_reports/macbook_M3/report_environment.json",
        "docs/performance_reports/macbook_M3/report_environment.tex",
    )
    assert (profile / ENVIRONMENT_JSON).is_file()


def test_pending_or_wrong_source_environment_fails_closed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
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
    initialize_profile(repo, "macbook_M3")
    measured = _commit_all(repo, "Initialize measured-source scaffold")

    with pytest.raises(
        ReportWorkspaceError,
        match="not authenticated for measurement source",
    ):
        require_authenticated_profile_environment(
            repo,
            "macbook_M3",
            expected_source_revision=measured,
        )
    with pytest.raises(
        ReportWorkspaceError,
        match="clean evaluator source checkout",
    ):
        refresh_profile_environment(
            repo,
            "macbook_M3",
            expected_source_revision="b" * 40,
            runtime_auditor=lambda _revision, _root: {},
        )


def test_active_runtime_must_still_match_recorded_environment(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
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
    initialize_profile(repo, "macbook_M3")
    measured = _commit_all(repo, "Initialize measured-source scaffold")

    def runtime(version: str) -> dict[str, object]:
        return {
            "package_version": version,
            "native_build_inputs_sha256": "a" * 64,
            "native_extension": {"sha256": "b" * 64},
            "python_package_tree": {"sha256": "c" * 64},
            "candidate_build_identity": {
                "candidate_fingerprint": "candidate-aarch64"
            },
            "native_target": {
                "triple": "aarch64-apple-darwin",
                "cpu_features": ["neon"],
            },
        }

    refresh_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
        runtime_auditor=lambda _revision, _root: runtime("0.1.0"),
    )

    with pytest.raises(
        ReportWorkspaceError,
        match="active measurement runtime differs",
    ):
        require_active_profile_environment(
            repo,
            "macbook_M3",
            expected_source_revision=measured,
            runtime_auditor=lambda _revision, _root: runtime("0.2.0"),
        )


def test_new_cluster_profile_resets_measurements_and_drops_source_pdf(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    initialize_profile(repo, "macbook_M3")

    cluster = initialize_profile(
        repo,
        "cluster_EPYC",
        source_profile="macbook_M3",
        reset_measurements=True,
    )
    result = ReportService(
        ReportPaths.from_repo(repo, profile="cluster_EPYC")
    ).validate()

    assert result["statuses"] == {
        "not_available": len(REPORT_CATALOG.measurement_cells())
    }
    assert not (cluster / "pyAmpliCol.pdf").exists()
    manifest = json.loads((cluster / WORKSPACE_MANIFEST).read_text())
    assert manifest["initialized_from"].endswith("/macbook_M3")
    assert manifest["measurement_state"] == "reset"


def test_export_profile_contains_only_fresh_publication_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    profile = initialize_profile(repo, "macbook_M3")
    stale_pdf = b"%PDF-1.4\n% stale source PDF\n%%EOF\n"
    fresh_pdf = b"%PDF-1.4\n% fresh exported PDF\n%%EOF\n"
    (profile / "pyAmpliCol.pdf").write_bytes(stale_pdf)
    local_state = (
        repo
        / ".artifacts/performance-report/macbook_M3/cells/private/worker.log"
    )
    local_state.parent.mkdir(parents=True)
    local_state.write_text("private\n", encoding="ascii")
    compiled: list[Path] = []

    def fake_compile(report_dir: Path) -> Path:
        compiled.append(report_dir)
        output = report_dir / "pyAmpliCol.pdf"
        output.write_bytes(fresh_pdf)
        (report_dir / "pyAmpliCol.log").write_text(
            "clean\n",
            encoding="ascii",
        )
        return output

    monkeypatch.setattr(
        "tools.performance_report.workspace.compile_report",
        fake_compile,
    )

    exported = export_profile(
        repo,
        "macbook_M3",
        tmp_path / "exports/macbook_M3",
    )

    assert (exported / "pyAmpliCol.tex").is_file()
    assert (exported / "pyAmpliCol.pdf").is_file()
    assert (exported / "pyAmpliCol.pdf").read_bytes() == fresh_pdf
    assert (profile / "pyAmpliCol.pdf").read_bytes() == stale_pdf
    assert len(compiled) == 1
    assert (exported / STANDALONE_BUILDER).is_file()
    assert (exported / WORKSPACE_MANIFEST).is_file()
    assert (exported / ENVIRONMENT_JSON).is_file()
    assert (exported / ENVIRONMENT_TEX).is_file()
    assert not (exported / ".artifacts").exists()
    assert not tuple(exported.rglob("*.lock"))
    assert not tuple(exported.rglob("*.log"))


def test_export_profile_can_omit_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    initialize_profile(repo, "macbook_M3")

    def unexpected_compile(_report_dir: Path) -> Path:
        raise AssertionError("PDF compilation must be disabled")

    monkeypatch.setattr(
        "tools.performance_report.workspace.compile_report",
        unexpected_compile,
    )
    exported = export_profile(
        repo,
        "macbook_M3",
        tmp_path / "exports/macbook_M3",
        include_pdf=False,
    )

    assert not (exported / "pyAmpliCol.pdf").exists()


def test_export_holds_source_writer_lock_while_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    initialize_profile(repo, "macbook_M3")
    competing = ReportService(
        ReportPaths.from_repo(repo, profile="macbook_M3")
    )
    original_copy = workspace_module._copy_publication_members
    observed_lock = False

    def checking_copy(source: Path, destination: Path) -> None:
        nonlocal observed_lock
        with (
            pytest.raises(LockTimeoutError),
            competing.store.named_lock("report-writer", timeout=0.0),
        ):
            pass
        observed_lock = True
        original_copy(source, destination)

    monkeypatch.setattr(
        workspace_module,
        "_copy_publication_members",
        checking_copy,
    )
    export_profile(
        repo,
        "macbook_M3",
        tmp_path / "exports/macbook_M3",
        include_pdf=False,
    )

    assert observed_lock


def test_reset_profile_rejects_preexisting_local_artifact_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)
    stale = repo / ".artifacts/performance-report/macbook_M3/cells/stale"
    stale.mkdir(parents=True)
    (stale / "current.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(
        ReportWorkspaceError,
        match="artifact root already contains local state",
    ):
        initialize_profile(
            repo,
            "macbook_M3",
            reset_measurements=True,
        )
