# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from pathlib import Path

from tools.performance_report.publisher import (
    publish_once,
    validate_published_snapshot,
)
from tools.performance_report.service import ReportPaths, ReportService


def _clean_report_service(tmp_path: Path) -> ReportService:
    root = tmp_path / "repo"
    (root / "docs/arxiv").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(
        ("git", "config", "user.email", "publisher-tests@example.invalid"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Publisher Tests"),
        cwd=root,
        check=True,
    )
    (root / "README.md").write_text("# publisher fixture\n", encoding="ascii")
    service = ReportService(ReportPaths.from_repo(root))
    service.publish(reset=True, merge_artifacts=False)
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "Initialize report fixture"),
        cwd=root,
        check=True,
    )
    return service


def test_publish_once_installs_one_consistent_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = _clean_report_service(tmp_path)

    def fake_compile(
        docs_dir: Path,
        *,
        expected_page_count: int,
        timeout_seconds: float,
    ) -> int:
        assert expected_page_count == 59
        assert timeout_seconds == 900.0
        (docs_dir / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        return 59

    monkeypatch.setattr(
        "tools.performance_report.publisher._compile_pdf",
        fake_compile,
    )

    published = publish_once(service)
    validated = validate_published_snapshot(service)

    assert published.current_count == 0
    assert published.page_count == 59
    assert validated["status"] == "ok"
    assert validated["snapshot_sha256"] == published.snapshot_sha256
    assert (service.paths.docs_dir / "pyAmpliCol.pdf").is_file()
