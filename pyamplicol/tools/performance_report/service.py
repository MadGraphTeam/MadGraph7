# SPDX-License-Identifier: 0BSD
"""Transactional cache merge, validation, and report-table publication."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactStore, CurrentRecord
from .cache import (
    build_reset_caches,
    schema_document,
    validate_cache,
    validate_measurement,
)
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import source_revision
from .publication import portable_publication_value
from .render import render_all_tables, summarize_visible_completeness
from .report_policy import publication_measurement_policy_issues


class ReportServiceError(RuntimeError):
    """Raised when cache or table publication cannot be completed."""


_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def validate_profile_name(value: str) -> str:
    """Return a filesystem-safe, human-readable report profile identifier."""

    if _PROFILE_RE.fullmatch(value) is None or ".." in value:
        raise ValueError(
            "report profile must contain 1-64 letters, digits, dots, "
            "underscores, or hyphens; it cannot contain '..'"
        )
    return value


@dataclass(frozen=True, slots=True)
class ReportPaths:
    repo_root: Path
    docs_dir: Path
    results_dir: Path
    artifact_root: Path
    coordination_root: Path

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        profile: str | None = None,
        docs_dir: Path | None = None,
        artifact_root: Path | None = None,
        coordination_root: Path | None = None,
    ) -> ReportPaths:
        root = repo_root.expanduser().resolve(strict=False)
        if profile is not None:
            profile = validate_profile_name(profile)
        if profile is not None and docs_dir is not None:
            raise ValueError("profile and docs_dir are mutually exclusive")
        docs = (
            root / "docs"
            if profile is None
            else root / "docs" / "performance_reports" / profile
        )
        if docs_dir is not None:
            docs = docs_dir.expanduser().resolve(strict=False)
        default_artifacts = root / ".artifacts/performance-report"
        if profile is not None:
            default_artifacts /= profile
        artifacts = (
            default_artifacts
            if artifact_root is None
            else artifact_root.expanduser().resolve(strict=False)
        )
        coordination = (
            (
                docs / "results/.coordination"
                if profile is None
                else root
                / ".artifacts/performance-report-coordination"
                / profile
            )
            if coordination_root is None
            else coordination_root.expanduser().resolve(strict=False)
        )
        return cls(root, docs, docs / "results", artifacts, coordination)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


class ReportService:
    def __init__(
        self,
        paths: ReportPaths,
        *,
        catalog: ReportCatalog = REPORT_CATALOG,
    ) -> None:
        self.paths = paths
        self.catalog = catalog
        self.store = ArtifactStore(
            artifact_root=paths.artifact_root,
            lock_root=paths.coordination_root,
        )

    def reset_payloads(self) -> dict[str, dict[str, object]]:
        return build_reset_caches(self.catalog)

    def load_caches(self) -> dict[str, dict[str, object]]:
        expected = self.reset_payloads()
        caches: dict[str, dict[str, object]] = {}
        for name in expected:
            path = self.paths.results_dir / name
            try:
                payload = json.loads(path.read_text(encoding="ascii"))
            except FileNotFoundError:
                payload = expected[name]
            except (OSError, json.JSONDecodeError) as error:
                raise ReportServiceError(
                    f"cannot load report cache {path}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ReportServiceError(f"report cache {path} must be an object")
            caches[name] = payload
        self.validate_payloads(caches)
        return caches

    def validate_payloads(
        self,
        caches: Mapping[str, Mapping[str, object]],
    ) -> None:
        cells_by_dataset: dict[str, list[object]] = {}
        for cell in self.catalog.measurement_cells():
            cells_by_dataset.setdefault(cell.dataset_id, []).append(cell)
        expected_names = {f"{dataset_id}.json" for dataset_id in cells_by_dataset}
        if set(caches) != expected_names:
            raise ReportServiceError(
                "report cache set differs from catalog; "
                f"missing={sorted(expected_names - set(caches))}, "
                f"extra={sorted(set(caches) - expected_names)}"
            )
        for name, payload in caches.items():
            dataset_id = name.removesuffix(".json")
            validate_cache(
                payload,
                expected_cells=cells_by_dataset[dataset_id],  # type: ignore[arg-type]
            )

    def merge_current(
        self,
        caches: dict[str, dict[str, object]],
        records: tuple[CurrentRecord, ...] | None = None,
    ) -> int:
        records = self.store.recover_current_records() if records is None else records
        expected_revision = source_revision(
            self.paths.repo_root,
            require_clean=True,
        )
        by_cell = {
            record.cell_id: record
            for record in records
            if isinstance(record.result.get("provenance"), Mapping)
            and record.result["provenance"].get("report_source_revision")
            == expected_revision
        }
        merged = 0
        for payload in caches.values():
            entries = payload["entries"]
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, dict)
                record = by_cell.get(str(entry["cell_id"]))
                if record is None:
                    continue
                measurement = record.result
                validate_measurement(measurement)
                entry["measurement"] = dict(measurement)
                merged += 1
        self.validate_payloads(caches)
        return merged

    def _snapshot_files(
        self,
        caches: Mapping[str, Mapping[str, object]],
        tables: Mapping[str, str],
    ) -> tuple[Path, ...]:
        portable_caches = {
            name: portable_publication_value(payload, self.paths)
            for name, payload in caches.items()
        }
        self.validate_payloads(portable_caches)  # type: ignore[arg-type]
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".report-snapshot-",
                dir=self.paths.docs_dir,
            )
        )
        written: list[Path] = []
        replaced: list[tuple[Path, Path | None]] = []
        try:
            staged_results = staging / "results"
            staged_results.mkdir()
            backup_root = staging / "previous"
            backup_root.mkdir()
            schema_path = staged_results / "report-cache.schema.json"
            schema_path.write_bytes(_canonical_bytes(schema_document()))
            for name, payload in portable_caches.items():
                (staged_results / name).write_bytes(_canonical_bytes(payload))
            for name, content in tables.items():
                (staging / name).write_text(content, encoding="ascii")

            publications = [
                (
                    schema_path,
                    self.paths.results_dir / "report-cache.schema.json",
                    Path("results/report-cache.schema.json"),
                ),
                *(
                    (
                        staged_results / name,
                        self.paths.results_dir / name,
                        Path("results") / name,
                    )
                    for name in sorted(caches)
                ),
                *(
                    (
                        staging / name,
                        self.paths.docs_dir / name,
                        Path(name),
                    )
                    for name in sorted(tables)
                ),
            ]
            try:
                for source, destination, relative in publications:
                    backup: Path | None = None
                    if destination.exists():
                        backup = backup_root / relative
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, backup)
                    source.replace(destination)
                    replaced.append((destination, backup))
                    written.append(destination)
            except BaseException:
                for destination, backup in reversed(replaced):
                    if backup is None:
                        destination.unlink(missing_ok=True)
                    else:
                        backup.replace(destination)
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return tuple(written)

    def publish(
        self,
        *,
        reset: bool = False,
        merge_artifacts: bool = True,
    ) -> tuple[Path, ...]:
        with self.store.named_lock("report-writer"):
            caches = self.reset_payloads() if reset else self.load_caches()
            if merge_artifacts and not reset:
                self.merge_current(caches)
            tables = render_all_tables(caches, catalog=self.catalog)
            return self._snapshot_files(caches, tables)

    def validate(self) -> dict[str, object]:
        caches = self.load_caches()
        tables = render_all_tables(caches, catalog=self.catalog)
        statuses: Counter[str] = Counter()
        for payload in caches.values():
            for entry in payload["entries"]:  # type: ignore[index]
                statuses[str(entry["measurement"]["status"])] += 1
        return {
            "cache_count": len(caches),
            "table_count": len(tables),
            "statuses": dict(sorted(statuses.items())),
        }

    def validate_publication_policy(
        self,
        caches: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Reject successful measurements that are not publication-grade."""

        issues: list[str] = []
        for payload in caches.values():
            entries = payload.get("entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                measurement = entry.get("measurement")
                if not isinstance(measurement, Mapping):
                    continue
                for issue in publication_measurement_policy_issues(measurement):
                    issues.append(
                        f"{entry.get('cell_id', '<unknown>')}: "
                        f"{issue.field}: {issue.detail}"
                    )
        if issues:
            displayed = "; ".join(issues[:12])
            if len(issues) > 12:
                displayed += f"; ... ({len(issues)} issues total)"
            raise ReportServiceError(
                "checked-in measurements violate publication policy: "
                + displayed
            )

    def audit(self) -> dict[str, object]:
        """Validate cache coverage and exact checked-in render correspondence."""

        result = self.validate()
        caches = self.load_caches()
        self.validate_publication_policy(caches)
        rendered = render_all_tables(caches, catalog=self.catalog)
        visible_completeness = summarize_visible_completeness(
            caches,
            catalog=self.catalog,
        )
        expected_cache_files = {
            *caches,
            "report-cache.schema.json",
        }
        actual_cache_files = {
            path.name for path in self.paths.results_dir.glob("*.json")
        }
        if actual_cache_files != expected_cache_files:
            raise ReportServiceError(
                "checked-in report cache files differ from the catalog; "
                f"missing={sorted(expected_cache_files - actual_cache_files)}, "
                f"extra={sorted(actual_cache_files - expected_cache_files)}"
            )
        mismatched_tables = [
            name
            for name, content in rendered.items()
            if not (self.paths.docs_dir / name).is_file()
            or (self.paths.docs_dir / name).read_text(encoding="ascii") != content
        ]
        if mismatched_tables:
            raise ReportServiceError(
                "checked-in report tables differ from canonical rendering: "
                + ", ".join(sorted(mismatched_tables))
            )
        return {
            **result,
            "cache_render_match": True,
            "visible_completeness": visible_completeness.as_dict(),
        }


__all__ = [
    "ReportPaths",
    "ReportService",
    "ReportServiceError",
    "validate_profile_name",
]
