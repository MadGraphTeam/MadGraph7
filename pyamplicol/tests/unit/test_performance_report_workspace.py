# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
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
from tools.performance_report.workspace import (
    ENVIRONMENT_TEX,
    STANDALONE_BUILDER,
    WORKSPACE_MANIFEST,
    ReportWorkspaceError,
    export_profile,
    initialize_profile,
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
    assert (profile / "results/report-cache.schema.json").is_file()
    assert not (profile / ".artifacts").exists()
    assert not (profile / "results/.coordination").exists()
    manifest = json.loads((profile / WORKSPACE_MANIFEST).read_text())
    assert manifest["profile"] == "macbook_M3"
    assert manifest["measurement_state"] == "copied"
    assert manifest["environment"]["profile"] == "macbook_M3"
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
