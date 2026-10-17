# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.performance_report.final_audit import (
    FinalAuditError,
    _report_publication_lineage,
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "report@example.invalid")
    _git(repo, "config", "user.name", "Report Test")
    _git(repo, "config", "core.fileMode", "true")
    (repo / "src").mkdir()
    (repo / "src/evaluator.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "docs/arxiv/results").mkdir(parents=True)
    (repo / "docs/arxiv/results/cache.json").write_text(
        "{}\n",
        encoding="ascii",
    )
    (repo / "docs/arxiv/pyAmpliCol.tex").write_text(
        "\\documentclass{article}\n",
        encoding="ascii",
    )
    return repo, _commit(repo, "measurement source")


def test_same_commit_publication_lineage_is_authenticated(tmp_path: Path) -> None:
    repo, measurement_revision = _repository(tmp_path)

    result = _report_publication_lineage(repo, measurement_revision)

    assert result["measurement_source_revision"] == measurement_revision
    assert result["publication_revision"] == measurement_revision
    assert result["relationship"] == "same-commit"
    assert result["changed_paths"] == []
    assert result["executable_source_unchanged"] is True


def test_report_only_descendant_accepts_canonical_and_profile_outputs(
    tmp_path: Path,
) -> None:
    repo, measurement_revision = _repository(tmp_path)
    canonical = repo / "src/pyamplicol/_profiling_campaign"
    (canonical / "results").mkdir(parents=True)
    (canonical / "results/cache.json").write_text(
        '{"measured":true}\n',
        encoding="ascii",
    )
    profile = repo / "docs/performance_reports/macbook_M3"
    (profile / "results").mkdir(parents=True)
    publications = {
        "README.md": "# M3 report\n",
        "architecture-profile.json": "{}\n",
        "report-workspace.json": "{}\n",
        "result_example_table.tex": "% generated\n",
        "section_results.tex": "% prose\n",
    }
    for name, content in publications.items():
        (profile / name).write_text(content, encoding="ascii")
    (profile / "results/cache.json").write_text("{}\n", encoding="ascii")
    (profile / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    publication_revision = _commit(repo, "publish report")

    result = _report_publication_lineage(repo, measurement_revision)

    assert result["publication_revision"] == publication_revision
    assert result["relationship"] == "report-only-descendant"
    changed = result["changed_paths"]
    assert isinstance(changed, list)
    assert {entry["path"] for entry in changed} == {
        "src/pyamplicol/_profiling_campaign/results/cache.json",
        "docs/performance_reports/macbook_M3/README.md",
        "docs/performance_reports/macbook_M3/architecture-profile.json",
        "docs/performance_reports/macbook_M3/pyAmpliCol.pdf",
        "docs/performance_reports/macbook_M3/report-workspace.json",
        "docs/performance_reports/macbook_M3/result_example_table.tex",
        "docs/performance_reports/macbook_M3/results/cache.json",
        "docs/performance_reports/macbook_M3/section_results.tex",
    }
    assert result["changed_path_count"] == len(changed)


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("src/evaluator.py", "VALUE = 2\n"),
        (
            "docs/performance_reports/macbook_M3/result_tables.py",
            "raise SystemExit(1)\n",
        ),
        (
            "docs/performance_reports/macbook_M3/results/worker.log",
            "private\n",
        ),
        ("docs/development/runtime.md", "runtime changed\n"),
    ],
)
def test_publication_descendant_rejects_every_nonreport_path(
    tmp_path: Path,
    relative_path: str,
    content: str,
) -> None:
    repo, measurement_revision = _repository(tmp_path)
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _commit(repo, "forbidden publication")

    with pytest.raises(FinalAuditError, match="outside the explicit"):
        _report_publication_lineage(repo, measurement_revision)


def test_publication_descendant_rejects_executable_allowed_member(
    tmp_path: Path,
) -> None:
    repo, measurement_revision = _repository(tmp_path)
    readme = repo / "src/pyamplicol/_profiling_campaign/README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# report\n", encoding="utf-8")
    readme.chmod(readme.stat().st_mode | 0o111)
    _commit(repo, "executable report")

    with pytest.raises(FinalAuditError, match="executable, a symlink"):
        _report_publication_lineage(repo, measurement_revision)


def test_publication_descendant_rejects_symlink_allowed_member(
    tmp_path: Path,
) -> None:
    repo, measurement_revision = _repository(tmp_path)
    readme = repo / "src/pyamplicol/_profiling_campaign/README.md"
    readme.parent.mkdir(parents=True)
    os.symlink("../src/evaluator.py", readme)
    _commit(repo, "symlink report")

    with pytest.raises(FinalAuditError, match="executable, a symlink"):
        _report_publication_lineage(repo, measurement_revision)


def test_publication_lineage_rejects_dirty_report_output(tmp_path: Path) -> None:
    repo, measurement_revision = _repository(tmp_path)
    (repo / "docs/arxiv/results/cache.json").write_text(
        '{"uncommitted":true}\n',
        encoding="ascii",
    )

    with pytest.raises(FinalAuditError, match="completely clean"):
        _report_publication_lineage(repo, measurement_revision)


def test_publication_lineage_rejects_non_descendant_commit(tmp_path: Path) -> None:
    repo, measurement_revision = _repository(tmp_path)
    empty_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    unrelated = _git(repo, "commit-tree", empty_tree, "-m", "unrelated publication")
    _git(repo, "checkout", "--detach", unrelated)

    with pytest.raises(FinalAuditError, match="not a descendant"):
        _report_publication_lineage(repo, measurement_revision)
