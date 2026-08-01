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
    reset_entry,
    schema_document,
    validate_cache,
    validate_measurement,
)
from .campaign_reset import OriginalAmplicolSeed, load_seed_if_present
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement_lineage import (
    MeasurementLineage,
    load_and_audit_measurement_lineage,
)
from .models import CellSpec
from .publication import portable_publication_value
from .render import render_all_tables, summarize_visible_completeness
from .report_policy import publication_measurement_policy_issues
from .source_identity import require_eligible_report_source


class ReportServiceError(RuntimeError):
    """Raised when cache or table publication cannot be completed."""


_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
CANONICAL_REPORT_ROOT = Path("src/pyamplicol/_profiling_campaign")
PROFILE_REPORT_ROOT = Path("docs/performance_reports")
CANONICAL_REPORT_ENTRYPOINT = CANONICAL_REPORT_ROOT / "result_tables.py"


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
        docs = root / (
            CANONICAL_REPORT_ROOT
            if profile is None
            else PROFILE_REPORT_ROOT / profile
        )
        if docs_dir is not None:
            docs = docs_dir.expanduser().resolve(strict=False)
        identity = "canonical" if profile is None else profile
        default_artifacts = root / ".artifacts/performance-report" / identity
        artifacts = (
            default_artifacts
            if artifact_root is None
            else artifact_root.expanduser().resolve(strict=False)
        )
        coordination = (
            (
                root
                / ".artifacts/performance-report-coordination"
                / identity
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
        self._authenticated_measurement_lineage: MeasurementLineage | None = None
        self._measurement_lineage_bound = False
        self._authenticated_original_amplicol_seed: OriginalAmplicolSeed | None = None
        self._original_amplicol_seed_bound = False

    def bind_measurement_lineage(
        self,
        lineage: MeasurementLineage | None,
    ) -> None:
        """Reuse one already fully audited bridge through a campaign transaction."""

        self._authenticated_measurement_lineage = lineage
        self._measurement_lineage_bound = True

    def _profile_name(self) -> str | None:
        try:
            relative = self.paths.docs_dir.relative_to(
                self.paths.repo_root / PROFILE_REPORT_ROOT
            )
        except ValueError:
            return None
        if (
            len(relative.parts) != 1
            or _PROFILE_RE.fullmatch(relative.parts[0]) is None
            or ".." in relative.parts[0]
        ):
            return None
        return relative.parts[0]

    def bind_original_amplicol_seed(
        self,
        seed: OriginalAmplicolSeed | None,
    ) -> None:
        """Reuse one already authenticated campaign-seed manifest."""

        self._authenticated_original_amplicol_seed = seed
        self._original_amplicol_seed_bound = True

    def _original_amplicol_seed(self) -> OriginalAmplicolSeed | None:
        if self._original_amplicol_seed_bound:
            return self._authenticated_original_amplicol_seed
        profile = self._profile_name()
        seed = (
            None
            if profile is None
            else load_seed_if_present(
                profile=profile,
                store=self.store,
                catalog=self.catalog,
            )
        )
        self.bind_original_amplicol_seed(seed)
        return seed

    def _measurement_lineage(self) -> MeasurementLineage | None:
        if self._measurement_lineage_bound:
            return self._authenticated_measurement_lineage
        if self._profile_name() is None:
            return None
        if not (
            self.paths.docs_dir / "measurement_lineage.json"
        ).exists() and (
            not self.store.recover_current_records()
            or self._original_amplicol_seed() is not None
        ):
            self.bind_measurement_lineage(None)
            return None
        source = require_eligible_report_source(self.paths.repo_root)
        lineage = load_and_audit_measurement_lineage(
            self.paths.repo_root,
            self.paths.docs_dir,
            self.store,
            expected_active_source_revision=source.revision,
            catalog=self.catalog,
        )
        self.bind_measurement_lineage(lineage)
        return lineage

    def _render_tables(
        self,
        caches: Mapping[str, Mapping[str, object]],
    ) -> dict[str, str]:
        lineage = self._measurement_lineage()
        source_lineage = (
            None
            if lineage is None
            else (lineage.ancestor_revision, lineage.descendant_revision)
        )
        if source_lineage is None:
            return render_all_tables(caches, catalog=self.catalog)
        return render_all_tables(
            caches,
            catalog=self.catalog,
            authenticated_source_lineage=source_lineage,
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
        cells_by_id: dict[str, CellSpec] = {}
        for cell in self.catalog.measurement_cells():
            cells_by_dataset.setdefault(cell.dataset_id, []).append(cell)
            cells_by_id[cell.cell_id] = cell
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
            entries = payload["entries"]
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, Mapping)
                cell = cells_by_id[str(entry["cell_id"])]
                if (
                    self.catalog.static_na_reason(cell) is not None
                    and entry.get("measurement")
                    != reset_entry(cell)["measurement"]
                ):
                    raise ReportServiceError(
                        f"{cell.cell_id}: catalog static N/A cache entry "
                        "differs from the canonical reset measurement"
                    )

    def merge_current(
        self,
        caches: dict[str, dict[str, object]],
        records: tuple[CurrentRecord, ...] | None = None,
    ) -> int:
        records = self.store.recover_current_records() if records is None else records
        source = require_eligible_report_source(self.paths.repo_root)
        expected_revision = source.revision
        lineage = self._measurement_lineage()
        seed = self._original_amplicol_seed()
        by_cell = {
            record.cell_id: record
            for record in records
            if (
                (
                    lineage.source_for_current(
                        record,
                        active_revision=source.revision,
                        active_tree=source.tree,
                    )
                    is not None
                    if lineage is not None
                    else (
                        isinstance(record.result.get("provenance"), Mapping)
                        and record.result["provenance"].get(
                            "report_source_revision"
                        )
                        == expected_revision
                        and record.result["provenance"].get("report_source_tree")
                        == source.tree
                    )
                )
                or (
                    seed is not None
                    and seed.source_for_current(
                        record,
                        active_revision=source.revision,
                        active_tree=source.tree,
                    )
                    is not None
                )
            )
        }
        cells_by_id = {
            cell.cell_id: cell for cell in self.catalog.measurement_cells()
        }
        merged = 0
        for payload in caches.values():
            entries = payload["entries"]
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, dict)
                cell_id = str(entry["cell_id"])
                cell = cells_by_id[cell_id]
                if self.catalog.static_na_reason(cell) is not None:
                    entry["measurement"] = reset_entry(cell)["measurement"]
                    continue
                if (
                    lineage is not None
                    and cell_id in lineage.required_descendant_cell_ids
                ):
                    entry["measurement"] = reset_entry(cell)["measurement"]
                record = by_cell.get(cell_id)
                if record is None:
                    continue
                measurement = record.result
                validate_measurement(
                    measurement,
                    expected_cell=cell,
                )
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
            tables = self._render_tables(caches)
            return self._snapshot_files(caches, tables)

    def validate(self) -> dict[str, object]:
        caches = self.load_caches()
        tables = self._render_tables(caches)
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
        rendered = self._render_tables(caches)
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
