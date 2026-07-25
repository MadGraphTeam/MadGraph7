# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ArtifactPolicy
from tools.performance_report.service import ReportPaths, ReportService


def _service(tmp_path: Path) -> ReportService:
    repo = tmp_path / "repo"
    (repo / "docs/results").mkdir(parents=True)
    return ReportService(ReportPaths.from_repo(repo))


def test_reset_publishes_only_canonical_na_caches_and_fourteen_tables(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    paths = service.publish(reset=True, merge_artifacts=False)
    result = service.validate()

    assert result["table_count"] == 14
    assert result["statuses"] == {
        "not_available": len(REPORT_CATALOG.measurement_cells())
    }
    assert len([path for path in paths if path.suffix == ".tex"]) == 14
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
            "provenance": {},
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
