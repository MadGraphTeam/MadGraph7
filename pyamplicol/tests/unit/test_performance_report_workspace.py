# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import multiprocessing
import subprocess
from pathlib import Path

import pytest

from tools.performance_report import workspace as workspace_module
from tools.performance_report.artifacts import LockTimeoutError
from tools.performance_report.campaign_policy import (
    MACBOOK_M3_POLICY,
    X86_EPYC_POLICY,
)
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
    TABLE_FILLING_RUNBOOK,
    WORKSPACE_MANIFEST,
    ReportWorkspaceError,
    export_profile,
    initialize_profile,
    record_authenticated_profile_environment,
    refresh_profile_environment,
    require_active_profile_environment,
    require_authenticated_profile_environment,
)


def _seed_template(repo: Path) -> None:
    docs = repo / "src/pyamplicol/_profiling_campaign"
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


def _try_report_writer_lock(
    repo: str,
    profile: str,
    outcomes: object,
) -> None:
    service = ReportService(
        ReportPaths.from_repo(Path(repo), profile=profile),
    )
    try:
        with service.store.named_lock("report-writer", timeout=0.0):
            outcomes.put("acquired")
    except LockTimeoutError:
        outcomes.put("timeout")


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
    template_state = (
        repo / ".artifacts/performance-report/canonical/cells/attempt"
    )
    template_state.mkdir(parents=True)
    (template_state / "worker.log").write_text("private\n", encoding="ascii")
    coordination = (
        repo / ".artifacts/performance-report-coordination/canonical"
    )
    coordination.mkdir(parents=True, exist_ok=True)
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
    assert "pending an authenticated post-checkpoint build" in environment_tex
    assert "source checkout" not in environment_tex
    environment = json.loads((profile / ENVIRONMENT_JSON).read_text())
    assert environment["schema"] == ENVIRONMENT_SCHEMA
    assert environment["status"] == "pending_exact_runtime"
    assert environment["source_revision"] == "pending"
    assert environment["platform"] == "pending measurement-host authentication"
    assert environment["machine"] == "pending measurement-host authentication"
    assert "source checkout" not in environment["pyamplicol"]
    assert (profile / "results/report-cache.schema.json").is_file()
    assert not (profile / ".artifacts").exists()
    assert not (profile / "results/.coordination").exists()
    manifest = json.loads((profile / WORKSPACE_MANIFEST).read_text())
    assert manifest["profile"] == "macbook_M3"
    assert manifest["campaign_policy"] == MACBOOK_M3_POLICY.as_manifest()
    assert (profile / TABLE_FILLING_RUNBOOK).is_file()
    assert manifest["measurement_state"] == "copied"
    assert manifest["initialized_environment"]["profile"] == "macbook_M3"
    assert (
        manifest["initialized_environment"]["status"]
        == "pending_exact_runtime"
    )
    assert manifest["environment_json"] == ENVIRONMENT_JSON
    assert "result_tables.py" in manifest["tracked_content"]
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
    runbook = (profile / TABLE_FILLING_RUNBOOK).read_text(encoding="utf-8")
    assert (
        "30 GB authenticated memory guard defined as max(process-tree RSS, "
        "Darwin physical footprint)"
    ) in runbook
    assert "Legacy v2 RSS-only censor evidence remains readable" in runbook
    assert "30 GB hard RSS" not in runbook
    assert ReportService(
        ReportPaths.from_repo(repo, profile="macbook_M3")
    ).audit()["cache_render_match"]

    with pytest.raises(ReportWorkspaceError, match="already exists"):
        initialize_profile(repo, "macbook_M3")


def test_initialize_x86_epyc_profile_binds_parallel_resource_policy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)

    profile = initialize_profile(repo, "x86_EPYC", reset_measurements=True)
    manifest = json.loads((profile / WORKSPACE_MANIFEST).read_text())
    readme = (profile / "README.md").read_text(encoding="utf-8")
    runbook = (profile / "TABLE_FILLING.md").read_text(encoding="utf-8")

    assert manifest["campaign_policy"] == X86_EPYC_POLICY.as_manifest()
    assert manifest["campaign_policy"]["workers"] == 25
    assert manifest["campaign_policy"]["memory_limit_bytes"] == 80_000_000_000
    assert "--workers 25 --cell-cores 1" in runbook
    assert "--max-ram-gb 80" in runbook
    assert "25 independent workers" in runbook
    assert "80 GB authenticated process-tree memory guard per worker" in runbook
    assert "Legacy v2 RSS-only censor evidence remains readable" in runbook
    assert "80 GB hard RSS" not in runbook
    assert "`TABLE_FILLING.md` is the sole authoritative campaign procedure" in (
        readme
    )
    assert "populate" not in readme
    assert "export-profile x86_EPYC" in readme


def test_profile_readme_requires_full_audited_five_second_campaign(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_template(repo)

    profile = initialize_profile(repo, "macbook_M3")
    readme = (profile / "README.md").read_text(encoding="utf-8")

    assert "`TABLE_FILLING.md` is the sole authoritative campaign procedure" in (
        readme
    )
    assert "populate" not in readme
    assert "render --compile" in readme
    assert "result_tables.py audit" in readme
    assert "export-profile macbook_M3" in readme


def test_documentation_index_links_the_four_published_report_pdfs() -> None:
    docs = Path(__file__).resolve().parents[2] / "docs"
    root_readme = (docs / "README.md").read_text(encoding="utf-8")
    report_readme = (docs / "performance_reports/README.md").read_text(
        encoding="utf-8"
    )

    assert "performance_reports/README.md" in root_readme
    assert "macbook_M3_pyAmpliCol.pdf" in report_readme
    assert "EPYC_pyAmpliCol.pdf" in report_readme
    assert "summary_plots_final.pdf" in report_readme
    assert "summary_plots_final_helicity_sum.pdf" in report_readme
    assert "performance_reports/results/" not in report_readme
    assert "profiling-campaign copy" in report_readme
    assert "refresh-pdf" in report_readme


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


def _test_environment(
    *,
    platform_summary: str = "Darwin 24.0.0; arm64; Apple M3 Pro",
    machine: str = "arm64",
    processor: str = "Apple M3 Pro",
) -> dict[str, str]:
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "profile": "macbook_M3",
        "platform": platform_summary,
        "machine": machine,
        "processor": processor,
        "python": "3.12.6",
        "python_implementation": "CPython",
        "status": "authenticated",
        "source_revision": "a" * 40,
        "pyamplicol": "0.1.0",
        "numpy": "2.1.0",
        "native_target": "aarch64-apple-darwin",
        "native_cpu_features": "neon",
        "native_build_inputs_sha256": "b" * 64,
        "native_extension_sha256": "c" * 64,
        "python_package_tree_sha256": "d" * 64,
        "candidate_fingerprint": "candidate-aarch64",
    }


def _test_runtime(source_revision: str = "a" * 40) -> dict[str, object]:
    return {
        "package_version": "0.1.0",
        "native_build_inputs_sha256": "a" * 64,
        "native_extension": {"sha256": "b" * 64},
        "python_package_tree": {"sha256": "c" * 64},
        "candidate_build_identity": {
            "candidate_fingerprint": "candidate-aarch64",
            "source_revision": source_revision,
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": ["neon", "fp-armv8"],
        },
    }


def test_processor_description_uses_sysctl_and_falls_back_when_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    with monkeypatch.context() as context:
        context.setattr(workspace_module.platform, "system", lambda: "Darwin")
        context.setattr(workspace_module.platform, "processor", lambda: "arm")
        context.setattr(
            workspace_module.subprocess,
            "run",
            lambda *_args, **_kwargs: Completed(0, "Apple M3 Pro\n"),
        )
        assert workspace_module._processor_description() == "Apple M3 Pro"

    attempted: list[tuple[str, ...]] = []

    def denied(command, **_kwargs):
        attempted.append(command)
        return Completed(1, "")

    with monkeypatch.context() as context:
        context.setattr(workspace_module.platform, "system", lambda: "Darwin")
        context.setattr(workspace_module.platform, "processor", lambda: "arm")
        context.setattr(workspace_module.subprocess, "run", denied)
        assert workspace_module._processor_description() == "arm"

    assert attempted == [
        ("sysctl", "-n", "machdep.cpu.brand_string"),
        ("sysctl", "-n", "hw.model"),
    ]


def test_active_runtime_accepts_only_processor_display_label_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    hosts = iter(
        (
            {
                "schema": ENVIRONMENT_SCHEMA,
                "profile": "macbook_M3",
                "platform": "Darwin 24.0.0; arm64; Apple M3 Pro",
                "machine": "arm64",
                "processor": "Apple M3 Pro",
                "python": "3.12.6",
                "python_implementation": "CPython",
            },
            {
                "schema": ENVIRONMENT_SCHEMA,
                "profile": "macbook_M3",
                "platform": "Darwin 24.0.0; arm64; arm",
                "machine": "arm64",
                "processor": "arm",
                "python": "3.12.6",
                "python_implementation": "CPython",
            },
        )
    )
    monkeypatch.setattr(
        workspace_module,
        "_host_environment_payload",
        lambda _profile: next(hosts),
    )

    recorded = refresh_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
        runtime_auditor=lambda _revision, _root: _test_runtime(measured),
    )

    assert recorded["processor"] == "Apple M3 Pro"
    assert require_active_profile_environment(
        repo,
        "macbook_M3",
        expected_source_revision=measured,
        runtime_auditor=lambda _revision, _root: _test_runtime(measured),
    ) == recorded


@pytest.mark.parametrize(
    "overrides",
    (
        {"platform": "Darwin 25.0.0; arm64; arm"},
        {
            "platform": "Linux 6.8.0; arm64; arm",
        },
        {
            "platform": "Darwin 24.0.0; x86_64; arm",
            "machine": "x86_64",
        },
        {"python": "3.13.1"},
        {"source_revision": "e" * 40},
        {"candidate_fingerprint": "candidate-other"},
        {"native_build_inputs_sha256": "e" * 64},
        {"native_extension_sha256": "e" * 64},
        {"python_package_tree_sha256": "e" * 64},
        {"native_target": "x86_64-apple-darwin"},
    ),
)
def test_stable_environment_identity_retains_strict_runtime_fields(
    overrides: dict[str, str],
) -> None:
    recorded = _test_environment()
    active = {
        **recorded,
        "platform": "Darwin 24.0.0; arm64; arm",
        "processor": "arm",
        **overrides,
    }

    assert workspace_module._stable_environment_identity(
        active
    ) != workspace_module._stable_environment_identity(recorded)


def test_stable_environment_identity_rejects_noncanonical_processor_suffix() -> None:
    environment = _test_environment(
        platform_summary="Darwin 24.0.0; arm64; Apple M3 Pro",
        processor="arm",
    )

    with pytest.raises(
        ReportWorkspaceError,
        match="does not match its processor label",
    ):
        workspace_module._stable_environment_identity(environment)


def test_stable_environment_identity_retains_linux_processor_model() -> None:
    recorded = _test_environment(
        platform_summary="Linux 6.8.0; x86_64; AMD EPYC 7551",
        machine="x86_64",
        processor="AMD EPYC 7551",
    )
    active = _test_environment(
        platform_summary="Linux 6.8.0; x86_64; AMD EPYC 7763",
        machine="x86_64",
        processor="AMD EPYC 7763",
    )

    assert workspace_module._stable_environment_identity(
        active
    ) != workspace_module._stable_environment_identity(recorded)


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
                "candidate_fingerprint": "candidate-aarch64",
                "source_revision": measured,
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


def test_explicit_environment_recorder_supports_portable_release_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "copied-campaign"
    docs.mkdir()
    monkeypatch.setattr(
        workspace_module,
        "_host_environment_payload",
        lambda profile: {
            "schema": ENVIRONMENT_SCHEMA,
            "profile": profile,
            "platform": "Linux 6.8.0; x86_64; AMD EPYC",
            "machine": "x86_64",
            "processor": "AMD EPYC",
            "python": "3.12.6",
            "python_implementation": "CPython",
        },
    )

    class Numpy:
        __version__ = "2.1.0"

    monkeypatch.setattr(
        workspace_module.importlib,
        "import_module",
        lambda name: Numpy() if name == "numpy" else None,
    )
    runtime = _test_runtime()
    runtime["candidate_build_identity"] = {
        "publishable": True,
        "source_revision": "a" * 40,
    }
    runtime["candidate_build_identity_sha256"] = "d" * 64

    recorded = record_authenticated_profile_environment(
        docs,
        "copied-campaign",
        expected_source_revision="a" * 40,
        active_runtime=runtime,
    )

    assert recorded["status"] == "authenticated"
    assert recorded["profile"] == "copied-campaign"
    assert recorded["source_revision"] == "a" * 40
    assert recorded["candidate_fingerprint"] == f"release:{'d' * 64}"
    assert json.loads((docs / ENVIRONMENT_JSON).read_text()) == recorded
    tex = (docs / ENVIRONMENT_TEX).read_text(encoding="utf-8")
    assert r"\renewcommand{\ReportProfileName}{copied-campaign}" in tex
    assert "pending" not in tex

    invalid = dict(runtime)
    invalid["candidate_build_identity_sha256"] = "short"
    with pytest.raises(ReportWorkspaceError, match="is not SHA-256"):
        record_authenticated_profile_environment(
            docs,
            "copied-campaign",
            expected_source_revision="a" * 40,
            active_runtime=invalid,
        )
    assert json.loads((docs / ENVIRONMENT_JSON).read_text()) == recorded

    wrong_source = dict(runtime)
    wrong_source["candidate_build_identity"] = {
        "publishable": True,
        "source_revision": "b" * 40,
    }
    with pytest.raises(ReportWorkspaceError, match="expected source revision"):
        record_authenticated_profile_environment(
            docs,
            "copied-campaign",
            expected_source_revision="a" * 40,
            active_runtime=wrong_source,
        )
    assert json.loads((docs / ENVIRONMENT_JSON).read_text()) == recorded

    missing_source = dict(runtime)
    missing_source["candidate_build_identity"] = {"publishable": True}
    with pytest.raises(ReportWorkspaceError, match="expected source revision"):
        record_authenticated_profile_environment(
            docs,
            "copied-campaign",
            expected_source_revision="a" * 40,
            active_runtime=missing_source,
        )
    assert json.loads((docs / ENVIRONMENT_JSON).read_text()) == recorded


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
                "candidate_fingerprint": "candidate-aarch64",
                "source_revision": measured,
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
    assert (exported / TABLE_FILLING_RUNBOOK).is_file()
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
    original_copy = workspace_module._copy_publication_members
    observed_lock = False

    def checking_copy(source: Path, destination: Path) -> None:
        nonlocal observed_lock

        context = multiprocessing.get_context("fork")
        outcomes = context.Queue()
        process = context.Process(
            target=_try_report_writer_lock,
            args=(
                str(repo),
                "macbook_M3",
                outcomes,
            ),
        )
        process.start()
        process.join(timeout=3)
        assert process.exitcode == 0
        assert outcomes.get(timeout=1) == "timeout"
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
