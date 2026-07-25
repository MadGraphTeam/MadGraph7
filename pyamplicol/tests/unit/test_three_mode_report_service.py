# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import source_revision
from tools.performance_report.models import ArtifactPolicy
from tools.performance_report.service import ReportPaths, ReportService


def _service(tmp_path: Path) -> ReportService:
    repo = tmp_path / "repo"
    (repo / "docs/results").mkdir(parents=True)
    return ReportService(ReportPaths.from_repo(repo))


def test_reset_publishes_only_canonical_na_caches_and_sixteen_tables(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    paths = service.publish(reset=True, merge_artifacts=False)
    result = service.validate()

    assert result["table_count"] == 16
    assert result["statuses"] == {
        "not_available": len(REPORT_CATALOG.measurement_cells())
    }
    assert len([path for path in paths if path.suffix == ".tex"]) == 16
    assert (service.paths.results_dir / "report-cache.schema.json").is_file()


def test_merge_joins_immutable_current_record_by_cell_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.publish(reset=True, merge_artifacts=False)
    cell = service.catalog.measurement_cells()[0]
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 2.0,
            "execution_seconds_per_point": None,
            "matrix_element": 3.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {},
            "selector_contract": None,
            "validation": {"status": "ok"},
            "resources": {},
                "provenance": {
                    "report_source_revision": source_revision(
                        service.paths.repo_root
                    )
                },
        }
    )
    service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
        measurement
    )

    service.publish()
    cache_path = service.paths.results_dir / f"{cell.dataset_id}.json"
    payload = json.loads(cache_path.read_text(encoding="ascii"))
    entry = next(item for item in payload["entries"] if item["cell_id"] == cell.cell_id)
    assert entry["measurement"]["matrix_element"] == 3.0


def test_failed_snapshot_publication_restores_previous_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    service.publish(reset=True, merge_artifacts=False)
    table = service.paths.docs_dir / "result_scalar_contact_table.tex"
    cache = service.paths.results_dir / "scalar_contact.json"
    old_table = table.read_bytes()
    old_cache = cache.read_bytes()

    original_replace = Path.replace
    calls = 0

    def fail_after_one_replace(source: Path, destination: Path) -> Path:
        nonlocal calls
        if ".report-snapshot-" in str(source):
            calls += 1
            if calls == 3:
                raise OSError("injected publication interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_after_one_replace)
    with pytest.raises(OSError, match="injected"):
        service.publish(reset=True, merge_artifacts=False)

    assert table.read_bytes() == old_table
    assert cache.read_bytes() == old_cache
