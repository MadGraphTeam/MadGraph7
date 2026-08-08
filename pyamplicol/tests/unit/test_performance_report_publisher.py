# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from tools.performance_report.publisher import (
    PublicationResult,
    ReportPublisherError,
    _compile_pdf,
    _copy_report_source,
    publish_once,
    run_publisher,
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


def test_publisher_source_copy_excludes_only_top_level_campaign_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyAmpliCol.tex").write_text("report\n", encoding="ascii")
    (source / "campaign_artifacts").mkdir()
    (source / "campaign_artifacts/sentinel").write_text("private\n", encoding="ascii")
    nested = source / "appendix/campaign_artifacts"
    nested.mkdir(parents=True)
    (nested / "published.txt").write_text("keep\n", encoding="ascii")
    destination = tmp_path / "staging"

    _copy_report_source(source, destination)

    assert not (destination / "campaign_artifacts").exists()
    assert (destination / "appendix/campaign_artifacts/published.txt").is_file()


def test_pdf_compiler_can_allow_overfull_boxes_for_interactive_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}ok\\end{document}\n",
        encoding="ascii",
    )
    latexmk = tmp_path / "latexmk"
    latexmk.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('pyAmpliCol.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        "Path('pyAmpliCol.log').write_text("
        "'Overfull \\\\hbox (1.0pt too wide)\\n"
        "Output written on pyAmpliCol.pdf (60 pages, 123 bytes).\\n', "
        "encoding='utf-8')\n",
        encoding="utf-8",
    )
    latexmk.chmod(0o755)
    monkeypatch.setattr(
        "tools.performance_report.publisher.shutil.which",
        lambda _name: str(latexmk),
    )

    assert (
        _compile_pdf(
            docs,
            expected_page_count=60,
            timeout_seconds=10.0,
            allow_overfull_boxes=True,
        )
        == 60
    )

    with pytest.raises(ReportPublisherError, match="overfull"):
        _compile_pdf(
            docs,
            expected_page_count=60,
            timeout_seconds=10.0,
        )


def test_pdf_compiler_streams_only_when_requested(
    tmp_path: Path,
    monkeypatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pyAmpliCol.tex").write_text(
        "\\documentclass{article}\\begin{document}ok\\end{document}\n",
        encoding="ascii",
    )
    latexmk = tmp_path / "latexmk"
    latexmk.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "print('live latex stdout', flush=True)\n"
        "print('live latex stderr', file=sys.stderr, flush=True)\n"
        "Path('pyAmpliCol.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        "Path('pyAmpliCol.log').write_text("
        "'Output written on pyAmpliCol.pdf (3 pages, 123 bytes).\\n', "
        "encoding='utf-8')\n",
        encoding="utf-8",
    )
    latexmk.chmod(0o755)
    monkeypatch.setattr(
        "tools.performance_report.publisher.shutil.which",
        lambda _name: str(latexmk),
    )

    assert (
        _compile_pdf(
            docs,
            expected_page_count=3,
            timeout_seconds=10.0,
        )
        == 3
    )
    captured = capfd.readouterr()
    assert "live latex stdout" not in captured.out
    assert "live latex stderr" not in captured.err

    assert (
        _compile_pdf(
            docs,
            expected_page_count=3,
            timeout_seconds=10.0,
            stream_output=True,
        )
        == 3
    )
    captured = capfd.readouterr()
    assert "live latex stdout" in captured.out
    assert "live latex stderr" in captured.err


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
        assert expected_page_count == 73
        assert timeout_seconds == 900.0
        (docs_dir / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        return 73

    monkeypatch.setattr(
        "tools.performance_report.publisher._compile_pdf",
        fake_compile,
    )

    published = publish_once(service)
    validated = validate_published_snapshot(service)

    assert published.current_count == 0
    assert published.page_count == 73
    assert validated["status"] == "ok"
    assert validated["snapshot_sha256"] == published.snapshot_sha256
    assert (service.paths.docs_dir / "pyAmpliCol.pdf").is_file()


def test_active_publisher_never_holds_the_controller_named_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = _clean_report_service(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    failure: list[BaseException] = []

    def fake_publish_once(*_args: object, **_kwargs: object) -> PublicationResult:
        entered.set()
        assert release.wait(timeout=5.0)
        return PublicationResult(
            current_count=0,
            page_count=59,
            snapshot_sha256="a" * 64,
            captured_at_utc="2026-01-01T00:00:00Z",
            published_at_utc="2026-01-01T00:00:01Z",
        )

    def publisher() -> None:
        try:
            run_publisher(service, watch=False)
        except BaseException as error:
            failure.append(error)

    monkeypatch.setattr(
        "tools.performance_report.publisher.publish_once",
        fake_publish_once,
    )
    thread = threading.Thread(target=publisher)
    thread.start()
    assert entered.wait(timeout=5.0)
    assert (
        service.paths.coordination_root / "publication" / "daemon.json"
    ).is_file()

    with (
        service.store.named_lock("report-publisher-daemon", timeout=0.0),
        service.store.named_lock("campaign-controller-boundary", timeout=0.0),
    ):
        pass
    with pytest.raises(ReportPublisherError, match="already active"):
        run_publisher(service, watch=False)

    release.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert failure == []


def test_publisher_backs_off_behind_controller_report_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = _clean_report_service(tmp_path)
    compiled = threading.Event()
    completed: list[PublicationResult] = []
    failure: list[BaseException] = []

    def fake_compile(
        docs_dir: Path,
        *,
        expected_page_count: int,
        timeout_seconds: float,
    ) -> int:
        assert expected_page_count == 73
        assert timeout_seconds == 900.0
        (docs_dir / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        compiled.set()
        return 73

    def publisher() -> None:
        try:
            completed.append(publish_once(service))
        except BaseException as error:
            failure.append(error)

    monkeypatch.setattr(
        "tools.performance_report.publisher._compile_pdf",
        fake_compile,
    )
    with service.store.named_lock("report-writer"):
        thread = threading.Thread(target=publisher)
        thread.start()
        assert compiled.wait(timeout=30.0)
        thread.join(timeout=0.15)
        assert thread.is_alive()

    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert failure == []
    assert len(completed) == 1
