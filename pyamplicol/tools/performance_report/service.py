"""Transactional cache merge, validation, and report-table publication."""

from __future__ import annotations

import json
import os
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
from .render import render_all_tables


class ReportServiceError(RuntimeError):
    """Raised when cache or table publication cannot be completed."""


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
        artifact_root: Path | None = None,
        coordination_root: Path | None = None,
    ) -> ReportPaths:
        root = repo_root.expanduser().resolve(strict=False)
        docs = root / "docs"
        artifacts = (
            root / ".artifacts/performance-report"
            if artifact_root is None
            else artifact_root.expanduser().resolve(strict=False)
        )
        coordination = (
            docs / "results/.coordination"
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
                raise ReportServiceError(f"cannot load report cache {path}: {error}")
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
        expected_names = {
            f"{dataset_id}.json" for dataset_id in cells_by_dataset
        }
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
        by_cell = {record.cell_id: record for record in records}
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
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".report-snapshot-",
                dir=self.paths.docs_dir,
            )
        )
        written: list[Path] = []
        try:
            staged_results = staging / "results"
            staged_results.mkdir()
            schema_path = staged_results / "report-cache.schema.json"
            schema_path.write_bytes(_canonical_bytes(schema_document()))
            for name, payload in caches.items():
                (staged_results / name).write_bytes(_canonical_bytes(payload))
            for name, content in tables.items():
                (staging / name).write_text(content, encoding="ascii")

            destination_schema = (
                self.paths.results_dir / "report-cache.schema.json"
            )
            schema_path.replace(destination_schema)
            written.append(destination_schema)
            for name in sorted(caches):
                destination = self.paths.results_dir / name
                (staged_results / name).replace(destination)
                written.append(destination)
            for name in sorted(tables):
                destination = self.paths.docs_dir / name
                (staging / name).replace(destination)
                written.append(destination)
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


__all__ = [
    "ReportPaths",
    "ReportService",
    "ReportServiceError",
]
