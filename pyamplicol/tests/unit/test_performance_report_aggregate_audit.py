# SPDX-License-Identifier: 0BSD
"""Tests for architecture-report aggregation authentication."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.performance_report.aggregate_audit import (
    AggregateAuditError,
    audit_aggregate_report,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _seed(repo: Path) -> str:
    _git(repo, "init")
    _git(repo, "config", "user.email", "report@example.invalid")
    _git(repo, "config", "user.name", "Report Test")
    for profile in ("macbook_M3", "x86_EPYC"):
        root = repo / "docs/performance_reports" / profile
        _write(root / "README.md", "fixed prose\n")
        _write(root / "results/data.json", '{"value":0}\n')
        _write(root / "pyAmpliCol.pdf", "reset\n")
    _write(repo / "src/runtime.py", "fixed = True\n")
    return _commit(repo, "base")


def _profile_commit(repo: Path, profile: str, branch: str, value: str) -> str:
    _git(repo, "switch", "-c", branch)
    root = repo / "docs/performance_reports" / profile
    _write(root / "results/data.json", f'{{"value":{value}}}\n')
    _write(root / "result_validation_summary.tex", f"summary {value}\n")
    _write(root / "pyAmpliCol.pdf", f"pdf {value}\n")
    return _commit(repo, profile)


def test_accepts_exact_subtrees_from_two_audited_profiles(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")
    _git(repo, "switch", "--detach", base)
    x86 = _profile_commit(repo, "x86_EPYC", "x86", "2")
    _git(repo, "switch", "-c", "aggregate", mac)
    _git(repo, "merge", "--no-ff", "--no-edit", x86)
    aggregate = _git(repo, "rev-parse", "HEAD")

    result = audit_aggregate_report(
        repo,
        base_revision=base,
        revision=aggregate,
        audited_profiles={"macbook_M3": mac, "x86_EPYC": x86},
    )

    assert result["status"] == "ok"
    assert result["changed_profiles"] == ["macbook_M3", "x86_EPYC"]


def test_accepts_audited_profile_after_main_advances_independently(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    measured_source = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")
    _git(repo, "switch", "--detach", measured_source)
    _git(repo, "switch", "-c", "advanced-main")
    _write(repo / "src/runtime.py", "fixed = 'later-main-change'\n")
    landing_base = _commit(repo, "advance main independently")
    _git(repo, "merge", "--no-ff", "--no-edit", mac)
    aggregate = _git(repo, "rev-parse", "HEAD")

    result = audit_aggregate_report(
        repo,
        base_revision=landing_base,
        revision=aggregate,
        audited_profiles={"macbook_M3": mac},
    )

    assert result["status"] == "ok"
    assert result["base_revision"] == landing_base
    assert result["changed_profiles"] == ["macbook_M3"]


@pytest.mark.parametrize(
    ("relative", "mutation"),
    (
        ("docs/performance_reports/macbook_M3/README.md", "prose\n"),
        ("src/runtime.py", "fixed = False\n"),
    ),
)
def test_rejects_non_output_changes(
    tmp_path: Path,
    relative: str,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")
    _write(repo / relative, mutation)
    revision = _commit(repo, "forbidden")

    with pytest.raises(
        AggregateAuditError,
        match="non-publication output",
    ):
        audit_aggregate_report(
            repo,
            base_revision=base,
            revision=revision,
            audited_profiles={"macbook_M3": mac},
        )


def test_rejects_changed_profile_without_audited_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")

    with pytest.raises(AggregateAuditError, match="without audited revisions"):
        audit_aggregate_report(
            repo,
            base_revision=base,
            revision=mac,
            audited_profiles={"x86_EPYC": base},
        )


def test_rejects_profile_subtree_that_differs_from_attestation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")
    _write(
        repo / "docs/performance_reports/macbook_M3/results/data.json",
        '{"value":3}\n',
    )
    aggregate = _commit(repo, "tampered")

    with pytest.raises(AggregateAuditError, match="subtree differs"):
        audit_aggregate_report(
            repo,
            base_revision=base,
            revision=aggregate,
            audited_profiles={"macbook_M3": mac},
        )


def test_rejects_executable_or_deleted_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _seed(repo)
    mac = _profile_commit(repo, "macbook_M3", "mac", "1")
    pdf = repo / "docs/performance_reports/macbook_M3/pyAmpliCol.pdf"
    pdf.chmod(0o755)
    executable = _commit(repo, "executable")

    with pytest.raises(AggregateAuditError, match="non-executable"):
        audit_aggregate_report(
            repo,
            base_revision=base,
            revision=executable,
            audited_profiles={"macbook_M3": executable},
        )

    _git(repo, "reset", "--hard", mac)
    pdf.unlink()
    deleted = _commit(repo, "deleted")
    with pytest.raises(AggregateAuditError, match="added or modified"):
        audit_aggregate_report(
            repo,
            base_revision=base,
            revision=deleted,
            audited_profiles={"macbook_M3": deleted},
        )
