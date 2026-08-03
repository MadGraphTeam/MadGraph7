# SPDX-License-Identifier: 0BSD
"""Nonblocking snapshot, render, and PDF publication for live campaigns."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import CurrentRecord, LockTimeoutError
from .measurement_lineage import load_measurement_lineage
from .service import ReportPaths, ReportService
from .source_identity import require_eligible_report_source
from .standalone_build import StandaloneBuildError, validate_latex_log

PUBLICATION_SNAPSHOT_SCHEMA = "pyamplicol-report-publication-snapshot-v1"
DEFAULT_PUBLICATION_INTERVAL_SECONDS = 600.0
DEFAULT_PDF_TIMEOUT_SECONDS = 900.0
DEFAULT_EXPECTED_PAGE_COUNT = 59
_INSTALL_LOCK_INITIAL_BACKOFF_SECONDS = 0.05
_INSTALL_LOCK_MAXIMUM_BACKOFF_SECONDS = 1.0

_PAGE_COUNT_RE = re.compile(
    r"Output written on .+? \((?P<pages>[1-9][0-9]*) pages?,"
)


class ReportPublisherError(RuntimeError):
    """Raised when a report snapshot cannot be built or published."""


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """One internally consistent cache/table/PDF publication."""

    current_count: int
    page_count: int
    snapshot_sha256: str
    captured_at_utc: str
    published_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PUBLICATION_SNAPSHOT_SCHEMA,
            "current_count": self.current_count,
            "page_count": self.page_count,
            "snapshot_sha256": self.snapshot_sha256,
            "captured_at_utc": self.captured_at_utc,
            "published_at_utc": self.published_at_utc,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _digest_bytes(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_manifest_path(service: ReportService) -> Path:
    return service.paths.coordination_root / "publication" / "current.json"


def _publisher_state_path(service: ReportService) -> Path:
    return service.paths.coordination_root / "publication" / "daemon.json"


@contextmanager
def _publisher_state_lease(
    service: ReportService,
    *,
    interval_seconds: float,
) -> Iterator[None]:
    """Claim the publisher state without a long-lived campaign named lock."""

    state_path = _publisher_state_path(service)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    guard_path = state_path.parent / "daemon.guard"
    guard = guard_path.open("a+b")
    lease_id = uuid.uuid4().hex
    payload = {
        "schema": PUBLICATION_SNAPSHOT_SCHEMA,
        "pid": os.getpid(),
        "lease_id": lease_id,
        "started_at_utc": _utc_now(),
        "interval_seconds": interval_seconds,
    }
    try:
        try:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise ReportPublisherError(
                "another report publisher is already active"
            ) from None
        _atomic_json(state_path, payload)
        try:
            yield
        finally:
            try:
                current = json.loads(state_path.read_text(encoding="ascii"))
            except (json.JSONDecodeError, OSError):
                current = None
            if (
                isinstance(current, Mapping)
                and current.get("lease_id") == lease_id
            ):
                state_path.unlink(missing_ok=True)
                _fsync_directory(state_path.parent)
    finally:
        try:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
        finally:
            guard.close()


@contextmanager
def _publisher_writer_lock(service: ReportService) -> Iterator[None]:
    """Yield only after the publisher can take the install lock without waiting."""

    backoff = _INSTALL_LOCK_INITIAL_BACKOFF_SECONDS
    while True:
        stack = ExitStack()
        try:
            stack.enter_context(
                service.store.named_lock("report-writer", timeout=0.0)
            )
        except LockTimeoutError:
            stack.close()
            time.sleep(backoff)
            backoff = min(
                _INSTALL_LOCK_MAXIMUM_BACKOFF_SECONDS,
                backoff * 2.0,
            )
            continue
        with stack:
            yield
        return


def _current_snapshot(
    records: tuple[CurrentRecord, ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "cell_id": record.cell_id,
            "attempt_id": record.attempt_id,
            "manifest_sha256": record.manifest_sha256,
        }
        for record in sorted(records, key=lambda item: item.cell_id)
    )


def _capture_current_records(
    service: ReportService,
) -> tuple[tuple[CurrentRecord, ...], tuple[dict[str, str], ...]]:
    """Read a stable pointer set without taking any campaign writer lock."""

    for _attempt in range(3):
        records = service.store.recover_current_records()
        snapshot = _current_snapshot(records)
        confirmed = service.store.recover_current_records()
        if _current_snapshot(confirmed) == snapshot:
            return records, snapshot
    raise ReportPublisherError(
        "current pointers changed during three consecutive snapshot reads"
    )


def _report_source_copy_ignore(
    source: Path,
    *patterns: str,
) -> Callable[[str, list[str]], set[str]]:
    """Ignore build debris and only the report root's private campaign state."""

    pattern_ignore = shutil.ignore_patterns(*patterns)
    source_path = os.path.abspath(source)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(pattern_ignore(directory, names))
        if os.path.abspath(directory) == source_path and "campaign_artifacts" in names:
            ignored.add("campaign_artifacts")
        return ignored

    return ignore


def _copy_report_source(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=_report_source_copy_ignore(
            source,
            "*.aux",
            "*.fdb_latexmk",
            "*.fls",
            "*.log",
            "*.out",
            "*.toc",
            "pyAmpliCol.pdf",
            ".coordination",
        ),
    )


def _compile_pdf(
    docs_dir: Path,
    *,
    expected_page_count: int | None,
    timeout_seconds: float,
    allow_overfull_boxes: bool = False,
    stream_output: bool = False,
) -> int:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise ReportPublisherError("latexmk is required for report publication")
    if expected_page_count is not None and expected_page_count < 1:
        raise ReportPublisherError("expected PDF page count must be positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ReportPublisherError("PDF timeout must be finite and positive")
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
    try:
        completed = subprocess.run(
            (
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "pyAmpliCol.tex",
            ),
            cwd=docs_dir,
            check=False,
            capture_output=not stream_output,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ReportPublisherError(
            f"latexmk exceeded the {timeout_seconds:g}s publication timeout"
        ) from error
    if completed.returncode != 0:
        if stream_output:
            raise ReportPublisherError(
                f"latexmk failed with exit {completed.returncode}; "
                "see the compilation output above"
            )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        tail = "\n".join((*stdout.splitlines(), *stderr.splitlines())[-80:])
        raise ReportPublisherError(
            f"latexmk failed with exit {completed.returncode}:\n{tail}"
        )
    log_path = docs_dir / "pyAmpliCol.log"
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        validate_latex_log(
            log,
            allow_overfull_boxes=allow_overfull_boxes,
        )
    except OSError as error:
        raise ReportPublisherError(f"cannot read LaTeX log: {error}") from error
    except StandaloneBuildError as error:
        raise ReportPublisherError(str(error)) from error
    matches = tuple(_PAGE_COUNT_RE.finditer(log))
    if not matches:
        raise ReportPublisherError("LaTeX log does not report a PDF page count")
    page_count = int(matches[-1].group("pages"))
    if (
        expected_page_count is not None
        and page_count != expected_page_count
    ):
        raise ReportPublisherError(
            "compiled PDF page count differs from the expected stable layout: "
            f"{page_count} != {expected_page_count}"
        )
    pdf = docs_dir / "pyAmpliCol.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise ReportPublisherError("latexmk did not produce a non-empty PDF")
    return page_count


def _staged_publications(
    staging_docs: Path,
    service: ReportService,
    table_names: tuple[str, ...],
) -> tuple[tuple[Path, Path, str], ...]:
    cache_names = tuple(
        sorted(path.name for path in (staging_docs / "results").glob("*.json"))
    )
    return (
        *(
            (
                staging_docs / "results" / name,
                service.paths.results_dir / name,
                f"results/{name}",
            )
            for name in cache_names
        ),
        *(
            (
                staging_docs / name,
                service.paths.docs_dir / name,
                name,
            )
            for name in table_names
        ),
        (
            staging_docs / "pyAmpliCol.pdf",
            service.paths.docs_dir / "pyAmpliCol.pdf",
            "pyAmpliCol.pdf",
        ),
    )


def _install_snapshot(
    service: ReportService,
    publications: tuple[tuple[Path, Path, str], ...],
    manifest: Mapping[str, object],
) -> None:
    staging_root = publications[0][0].parents[2]
    backup_root = staging_root / "previous"
    backup_root.mkdir()
    replaced: list[tuple[Path, Path | None]] = []
    with _publisher_writer_lock(service):
        try:
            for source, destination, relative in publications:
                if not source.is_file():
                    raise ReportPublisherError(
                        f"publication snapshot member is missing: {source}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup: Path | None = None
                if destination.exists():
                    backup = backup_root / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, backup)
                temporary = destination.with_name(
                    f".{destination.name}.publisher-{uuid.uuid4().hex}"
                )
                try:
                    shutil.copy2(source, temporary)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                    _fsync_directory(destination.parent)
                finally:
                    temporary.unlink(missing_ok=True)
                replaced.append((destination, backup))
            _atomic_json(_snapshot_manifest_path(service), manifest)
        except BaseException:
            for destination, backup in reversed(replaced):
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    temporary = destination.with_name(
                        f".{destination.name}.rollback-{uuid.uuid4().hex}"
                    )
                    try:
                        shutil.copy2(backup, temporary)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
            raise


def validate_published_snapshot(service: ReportService) -> dict[str, object]:
    """Validate only the committed schema and snapshot file identities."""

    service.validate_payloads(service.load_caches())
    path = _snapshot_manifest_path(service)
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportPublisherError(
            f"cannot read publication snapshot manifest {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema") != (
        PUBLICATION_SNAPSHOT_SCHEMA
    ):
        raise ReportPublisherError("publication snapshot manifest is malformed")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ReportPublisherError("publication snapshot file inventory is missing")
    observed: list[dict[str, object]] = []
    for index, item in enumerate(files):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size"}
            or not isinstance(item["path"], str)
            or Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
        ):
            raise ReportPublisherError(
                f"publication snapshot file record {index} is malformed"
            )
        source = service.paths.docs_dir / item["path"]
        if (
            not source.is_file()
            or source.stat().st_size != item["size"]
            or _sha256(source) != item["sha256"]
        ):
            raise ReportPublisherError(
                f"published report differs from snapshot member {item['path']!r}"
            )
        observed.append(dict(item))
    expected_digest = payload.get("snapshot_sha256")
    unsigned = dict(payload)
    unsigned.pop("snapshot_sha256", None)
    if expected_digest != _digest_bytes(unsigned):
        raise ReportPublisherError("publication snapshot digest does not recompute")
    return {
        "schema": PUBLICATION_SNAPSHOT_SCHEMA,
        "snapshot_sha256": expected_digest,
        "current_count": payload.get("current_count"),
        "page_count": payload.get("page_count"),
        "file_count": len(observed),
        "status": "ok",
    }


def publish_once(
    service: ReportService,
    *,
    expected_page_count: int = DEFAULT_EXPECTED_PAGE_COUNT,
    pdf_timeout_seconds: float = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> PublicationResult:
    """Render and compile an immutable current-pointer snapshot off-thread."""

    source = require_eligible_report_source(service.paths.repo_root)
    try:
        profile_relative = service.paths.docs_dir.relative_to(
            service.paths.repo_root / "docs/performance_reports"
        )
    except ValueError:
        profile_relative = None
    lineage = (
        load_measurement_lineage(
            service.paths.repo_root,
            service.paths.docs_dir,
            expected_active_revision=source.revision,
            expected_active_tree=source.tree,
        )
        if profile_relative is not None and len(profile_relative.parts) == 1
        else None
    )
    service.bind_measurement_lineage(lineage)
    captured_at = _utc_now()
    records, current_snapshot = _capture_current_records(service)
    build_root = service.paths.artifact_root / "publication-builds"
    build_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="report-publication-", dir=build_root)
    )
    try:
        staging_docs = staging / "docs"
        _copy_report_source(service.paths.docs_dir, staging_docs)
        staging_service = ReportService(
            ReportPaths.from_repo(
                service.paths.repo_root,
                docs_dir=staging_docs,
                artifact_root=service.paths.artifact_root,
                coordination_root=service.paths.coordination_root,
            ),
            catalog=service.catalog,
        )
        staging_service.bind_measurement_lineage(lineage)
        caches = staging_service.load_caches()
        staging_service.merge_current(caches, records)
        tables = staging_service._render_tables(caches)
        staging_service._snapshot_files(caches, tables)

        checked_caches = staging_service.load_caches()
        staging_service.validate_payloads(checked_caches)
        rerendered = staging_service._render_tables(checked_caches)
        if rerendered != tables:
            raise ReportPublisherError(
                "staged cache snapshot does not reproduce its staged tables"
            )
        page_count = _compile_pdf(
            staging_docs,
            expected_page_count=expected_page_count,
            timeout_seconds=pdf_timeout_seconds,
        )
        publications = _staged_publications(
            staging_docs,
            service,
            tuple(sorted(tables)),
        )
        file_inventory = [
            {
                "path": relative,
                "size": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
            for source_path, _destination, relative in publications
        ]
        published_at = _utc_now()
        manifest: dict[str, object] = {
            "schema": PUBLICATION_SNAPSHOT_SCHEMA,
            "source_revision": source.revision,
            "source_tree": source.tree,
            "captured_at_utc": captured_at,
            "published_at_utc": published_at,
            "current_count": len(current_snapshot),
            "current_snapshot": list(current_snapshot),
            "current_snapshot_sha256": _digest_bytes(current_snapshot),
            "page_count": page_count,
            "files": file_inventory,
        }
        manifest["snapshot_sha256"] = _digest_bytes(manifest)
        _install_snapshot(service, publications, manifest)
        validated = validate_published_snapshot(service)
        if validated["snapshot_sha256"] != manifest["snapshot_sha256"]:
            raise ReportPublisherError(
                "installed publication snapshot identity changed"
            )
        return PublicationResult(
            current_count=len(current_snapshot),
            page_count=page_count,
            snapshot_sha256=str(manifest["snapshot_sha256"]),
            captured_at_utc=captured_at,
            published_at_utc=published_at,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_publisher(
    service: ReportService,
    *,
    watch: bool,
    interval_seconds: float = DEFAULT_PUBLICATION_INTERVAL_SECONDS,
    expected_page_count: int = DEFAULT_EXPECTED_PAGE_COUNT,
    pdf_timeout_seconds: float = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> PublicationResult:
    """Publish immediately and optionally repeat at a start-to-start cadence."""

    if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ReportPublisherError(
            "publication interval must be finite and positive"
        )
    with _publisher_state_lease(
        service,
        interval_seconds=interval_seconds,
    ):
        latest: PublicationResult | None = None
        while True:
            cycle_started = time.monotonic()
            try:
                latest = publish_once(
                    service,
                    expected_page_count=expected_page_count,
                    pdf_timeout_seconds=pdf_timeout_seconds,
                )
                print(
                    json.dumps(
                        latest.as_dict(),
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "schema": PUBLICATION_SNAPSHOT_SCHEMA,
                            "status": "error",
                            "kind": type(error).__name__,
                            "message": str(error),
                            "recorded_at_utc": _utc_now(),
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not watch:
                    raise
            if not watch:
                assert latest is not None
                return latest
            remaining = interval_seconds - (
                time.monotonic() - cycle_started
            )
            if remaining > 0.0:
                time.sleep(remaining)


__all__ = [
    "DEFAULT_EXPECTED_PAGE_COUNT",
    "DEFAULT_PDF_TIMEOUT_SECONDS",
    "DEFAULT_PUBLICATION_INTERVAL_SECONDS",
    "PUBLICATION_SNAPSHOT_SCHEMA",
    "PublicationResult",
    "ReportPublisherError",
    "publish_once",
    "run_publisher",
    "validate_published_snapshot",
]
