# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.performance_report.source_identity import (
    PUBLICATION_SOURCE_IDENTITY_SCHEMA,
    SOURCE_IDENTITY_SCHEMA,
    ReportSourceIdentityError,
    inspect_report_publication,
    inspect_report_source,
    require_eligible_report_source,
    require_report_only_publication,
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


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "report@example.invalid")
    _git(repo, "config", "user.name", "Report Test")
    (repo / "src").mkdir()
    (repo / "src/evaluator.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "docs/results").mkdir(parents=True)
    (repo / "docs/results/cache.json").write_text("{}\n", encoding="ascii")
    (repo / "docs/result_example_table.tex").write_text(
        "% generated\n",
        encoding="ascii",
    )
    profile = repo / "docs/performance_reports/macbook_M3"
    (profile / "results").mkdir(parents=True)
    (profile / "pyAmpliCol.tex").write_text("% report\n", encoding="ascii")
    (profile / "result_tables.py").write_text("# entry point\n", encoding="ascii")
    (profile / "results/cache.json").write_text("{}\n", encoding="ascii")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_source_identity_records_commit_tree_and_clean_state(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)

    identity = require_eligible_report_source(repo)

    assert identity.revision == _git(repo, "rev-parse", "HEAD")
    assert identity.tree == _git(repo, "rev-parse", "HEAD^{tree}")
    assert identity.dirty_paths == ()
    assert identity.provenance() == {
        "report_source_identity_schema": SOURCE_IDENTITY_SCHEMA,
        "report_source_revision": identity.revision,
        "report_source_tree": identity.tree,
        "report_measured_source_revision": identity.revision,
        "report_measured_source_tree": identity.tree,
        "report_source_clean": True,
    }


def test_generated_report_outputs_do_not_dirty_measurement_source(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / "docs/results/cache.json").write_text('{"measured":true}\n')
    (repo / "docs/result_example_table.tex").write_text("% refreshed\n")
    profile = repo / "docs/performance_reports/macbook_M3"
    (profile / "results/cache.json").write_text('{"measured":true}\n')
    (profile / "result_example_table.tex").write_text("% generated\n")
    (profile / "result_validation_summary.tex").write_text("% summary\n")
    (profile / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n")
    (profile / "pyAmpliCol.aux").write_text("\\relax\n")
    artifacts = repo / ".artifacts/performance-report/macbook_M3"
    artifacts.mkdir(parents=True)
    (artifacts / "attempt.bin").write_bytes(b"attempt")

    identity = require_eligible_report_source(repo)

    assert identity.eligible
    assert identity.dirty_paths == ()


@pytest.mark.parametrize(
    "path",
    (
        "docs/performance_reports/macbook_M3/result_tables.py",
        "docs/performance_reports/macbook_M3/pyAmpliCol.tex",
        "docs/performance_reports/macbook_M3/section_scope.tex",
        "docs/performance_reports/macbook_M3/README.md",
        "docs/performance_reports/macbook_M3/report-workspace.json",
        "docs/performance_reports/macbook_M3/arbitrary.txt",
        "docs/performance_reports/macbook_M3/results/nested/raw.json",
    ),
)
def test_profile_scaffold_or_arbitrary_changes_dirty_measurement_source(
    tmp_path: Path,
    path: str,
) -> None:
    repo = _repository(tmp_path)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n", encoding="utf-8")

    identity = inspect_report_source(repo)

    assert path in identity.dirty_paths


def test_tracked_or_untracked_source_changes_are_ineligible(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / "src/evaluator.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools/untracked.py").write_text("VALUE = 3\n", encoding="utf-8")

    identity = inspect_report_source(repo)

    assert not identity.eligible
    assert identity.dirty_paths == (
        "src/evaluator.py",
        "tools/untracked.py",
    )
    with pytest.raises(
        ReportSourceIdentityError,
        match=r"dirty source paths: src/evaluator\.py, tools/untracked\.py",
    ):
        require_eligible_report_source(repo)


def test_generated_profile_environment_is_a_report_output(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    profile = repo / "docs/performance_reports/macbook_M3"
    (profile / "report_environment.json").write_text(
        '{"status":"authenticated"}\n',
        encoding="ascii",
    )
    (profile / "report_environment.tex").write_text(
        "% authenticated environment\n",
        encoding="ascii",
    )

    identity = require_eligible_report_source(repo)

    assert identity.eligible
    assert identity.dirty_paths == ()


def _commit_path(repo: Path, path: str, content: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", f"update {path}")
    return _git(repo, "rev-parse", "HEAD")


def test_publication_descendant_authenticates_both_shas_and_report_paths(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    profile = "macbook_M3"
    _commit_path(
        repo,
        f"docs/performance_reports/{profile}/result_example_table.tex",
        "% measured table\n",
    )
    publication = _commit_path(
        repo,
        f"docs/performance_reports/{profile}/results/raw.json",
        '{"status":"ok"}\n',
    )

    identity = require_report_only_publication(
        repo,
        measured_revision=measured,
        profile=profile,
    )

    assert identity.measured_revision == measured
    assert identity.publication_revision == publication
    assert identity.measured_tree == _git(repo, "rev-parse", f"{measured}^{{tree}}")
    assert identity.publication_tree == _git(
        repo,
        "rev-parse",
        f"{publication}^{{tree}}",
    )
    assert identity.eligible
    assert identity.provenance() == {
        "report_publication_source_identity_schema": (
            PUBLICATION_SOURCE_IDENTITY_SCHEMA
        ),
        "report_profile": profile,
        "report_measured_source_revision": measured,
        "report_measured_source_tree": identity.measured_tree,
        "report_publication_revision": publication,
        "report_publication_tree": identity.publication_tree,
        "report_publication_report_only": True,
        "report_publication_changed_paths": [
            f"docs/performance_reports/{profile}/result_example_table.tex",
            f"docs/performance_reports/{profile}/results/raw.json",
        ],
    }


def test_publication_descendant_allows_authenticated_environment_outputs(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    profile = "macbook_M3"
    _commit_path(
        repo,
        f"docs/performance_reports/{profile}/report_environment.json",
        '{"status":"authenticated"}\n',
    )
    publication = _commit_path(
        repo,
        f"docs/performance_reports/{profile}/report_environment.tex",
        "% authenticated environment\n",
    )

    identity = require_report_only_publication(
        repo,
        measured_revision=measured,
        profile=profile,
        publication_revision=publication,
    )

    assert identity.eligible
    assert identity.changed_paths == (
        f"docs/performance_reports/{profile}/report_environment.json",
        f"docs/performance_reports/{profile}/report_environment.tex",
    )


@pytest.mark.parametrize(
    "path",
    (
        "src/evaluator.py",
        "rust/runtime.rs",
        "dependencies/symbolica/source.rs",
        "build_backend/backend.py",
        "tools/performance_report/runner.py",
    ),
)
def test_publication_rejects_evaluator_or_tool_changes(
    tmp_path: Path,
    path: str,
) -> None:
    repo = _repository(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    _commit_path(repo, path, "CHANGED = True\n")

    identity = inspect_report_publication(
        repo,
        measured_revision=measured,
        profile="macbook_M3",
    )

    assert identity.disallowed_paths == (path,)
    with pytest.raises(
        ReportSourceIdentityError,
        match="disallowed paths",
    ):
        require_report_only_publication(
            repo,
            measured_revision=measured,
            profile="macbook_M3",
        )


def test_publication_rejects_scaffold_wrong_profile_and_nested_payloads(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    paths = (
        "docs/performance_reports/cluster_EPYC/results/raw.json",
        "docs/performance_reports/macbook_M3/result_tables.py",
        "docs/performance_reports/macbook_M3/results/nested/raw.json",
        "docs/performance_reports/macbook_M3/section_scope.tex",
    )
    for path in paths:
        _commit_path(repo, path, "payload\n")

    identity = inspect_report_publication(
        repo,
        measured_revision=measured,
        profile="macbook_M3",
    )

    assert identity.disallowed_paths == paths


def test_publication_rejects_rename_executable_and_symlink_modes(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    profile = repo / "docs/performance_reports/macbook_M3"
    measured = _git(repo, "rev-parse", "HEAD")

    old_table = profile / "result_old_table.tex"
    old_table.write_text("% old\n", encoding="ascii")
    _git(repo, "add", old_table.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "add old table before measurement")
    measured = _git(repo, "rev-parse", "HEAD")

    new_table = profile / "result_new_table.tex"
    old_table.rename(new_table)
    executable = profile / "result_executable_table.tex"
    executable.write_text("% executable\n", encoding="ascii")
    executable.chmod(0o755)
    symlink = profile / "result_symlink_table.tex"
    symlink.symlink_to("result_new_table.tex")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "unsafe publication members")

    identity = inspect_report_publication(
        repo,
        measured_revision=measured,
        profile="macbook_M3",
    )

    assert identity.disallowed_paths == (
        "docs/performance_reports/macbook_M3/result_executable_table.tex",
        "docs/performance_reports/macbook_M3/result_new_table.tex",
        "docs/performance_reports/macbook_M3/result_old_table.tex",
        "docs/performance_reports/macbook_M3/result_symlink_table.tex",
    )


def test_publication_must_descend_from_measured_commit(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--orphan", "unrelated")
    _git(repo, "rm", "-rf", ".")
    publication = _commit_path(repo, "README.md", "unrelated\n")

    with pytest.raises(
        ReportSourceIdentityError,
        match="not a descendant",
    ):
        inspect_report_publication(
            repo,
            measured_revision=measured,
            publication_revision=publication,
            profile="macbook_M3",
        )
