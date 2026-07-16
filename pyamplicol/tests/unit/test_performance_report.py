# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS))

import result_tables as report  # noqa: E402

from pyamplicol.api import BenchmarkResult, BenchmarkStatistics  # noqa: E402
from pyamplicol.config import BenchmarkConfig  # noqa: E402


def _measurement_is_na(measurement: object) -> bool:
    assert isinstance(measurement, dict)
    numeric_fields = (
        "generation_seconds",
        "sample_count",
        "wall_seconds_per_point",
        "evaluator_seconds_per_point",
        "standard_deviation_seconds_per_point",
        "standard_error_seconds_per_point",
        "relative_standard_error",
        "matrix_element",
    )
    return (
        measurement["status"] == report.NA_STATUS
        and all(measurement[field] is None for field in numeric_fields)
        and measurement["requested_config"] is None
        and measurement["effective_config"] is None
        and measurement["environment"] == {}
    )


def test_checked_in_caches_match_schema_and_are_reset_to_na() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    assert len(caches) == 10
    assert (
        json.loads(
            (DOCS / "results" / "report-cache.schema.json").read_text(encoding="utf-8")
        )
        == report.schema_document()
    )

    for payload in caches.values():
        report.validate_cache(payload)
        assert payload["spdx_license_identifier"] == "0BSD"
        assert payload["updated_at"] is None
        for entry in payload["entries"]:
            assert entry["status"] == report.NA_STATUS
            if payload["kind"] == report.CacheKind.PROCESS_MATRIX:
                assert _measurement_is_na(entry["reference"])
                assert _measurement_is_na(entry["pyamplicol"])
                assert entry["relative_difference"] is None
            else:
                assert _measurement_is_na(entry["measurement"])
                if payload["kind"] == report.CacheKind.MODEL_LADDER:
                    assert entry["high_precision_matrix_element"] is None
                    assert entry["relative_difference"] is None


def test_process_matrices_preserve_families_and_multiplicity_grids() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    expected_keys = {family.key for family in report.PROCESS_FAMILIES}
    expected_lc_maxima = {
        family.key: family.maximum_lc_n for family in report.PROCESS_FAMILIES
    }

    for spec in report.MATRIX_SPECS:
        payload = caches[spec.cache_name]
        assert len(payload["process_families"]) == 14
        assert {
            family["key"] for family in payload["process_families"]
        } == expected_keys
        assert payload["multiplicities"] == list(spec.multiplicities)
        assert len(payload["entries"]) == 14 * len(spec.multiplicities)
        cells = {
            (entry["process_key"], entry["n_final"]) for entry in payload["entries"]
        }
        assert cells == {
            (key, n_final) for key in expected_keys for n_final in spec.multiplicities
        }
        for family in payload["process_families"]:
            expected_maximum = (
                expected_lc_maxima[family["key"]] if spec.color_accuracy == "lc" else 5
            )
            assert family["maximum_n"] == expected_maximum


def test_ladders_preserve_expected_multiplicities_and_variants() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    expected = {
        "z_builtin_sm": list(range(1, 10)),
        "z_external_sm": list(range(1, 7)),
        "scalar_contact": list(range(2, 9)),
        "scalar_gravity": list(range(2, 5)),
    }
    for spec in report.LADDER_SPECS:
        payload = caches[spec.cache_name]
        assert payload["multiplicities"] == expected[spec.dataset_id]
        if spec.kind == report.CacheKind.PERFORMANCE_LADDER:
            assert [variant["key"] for variant in payload["variants"]] == [
                variant.key for variant in report.Z_VARIANTS
            ]
            assert len(payload["entries"]) == len(spec.multiplicities) * len(
                report.Z_VARIANTS
            )
        else:
            assert len(payload["entries"]) == len(spec.multiplicities)


def test_generated_tables_are_complete_and_input_by_main_tex() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    expected_tables = report.render_tables(caches)
    tex = (DOCS / "pyAmpliCol.tex").read_text(encoding="utf-8")
    inputs = tuple(re.findall(r"\\input\{([^}]+_table\.tex)\}", tex))
    assert inputs == report.TABLE_INPUTS

    for name, expected_text in expected_tables.items():
        path = DOCS / name
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == expected_text
        assert expected_text.startswith("% SPDX-License-Identifier: 0BSD\n")
        assert expected_text.count(r"\ReportNA") > 0


def test_report_excludes_internal_campaign_narrative() -> None:
    report_text = (DOCS / "pyAmpliCol.tex").read_text(encoding="utf-8").lower()
    forbidden = (
        "memory-watchdog",
        "watchdog",
        "historical bug",
        "archive narrative",
        "developer diary",
        "python vs rust remains friendly",
    )
    assert not any(term in report_text for term in forbidden)


def test_benchmark_observation_uses_public_typed_result() -> None:
    requested = BenchmarkConfig(target_runtime=2.0, batch_size=32)
    effective = BenchmarkConfig(target_runtime=2.0, batch_size=16)
    result = BenchmarkResult(
        requested_config=requested,
        effective_config=effective,
        sample_count=7,
        wall_time_per_point=2.5e-6,
        evaluator_time_per_point=2.0e-6,
        uncertainty=BenchmarkStatistics(1.0e-7, 4.0e-8, 0.016),
        environment={"platform": "test"},
    )

    observation = report.BenchmarkObservation.from_result(result)
    cache_fields = observation.as_cache_fields()
    assert cache_fields["sample_count"] == 7
    assert cache_fields["requested_config"]["batch_size"] == 32
    assert cache_fields["effective_config"]["batch_size"] == 16
    assert cache_fields["environment"] == {"platform": "test"}


def test_multi_file_publish_rolls_back_on_replacement_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    existing = docs / "existing.txt"
    existing.write_text("old\n", encoding="utf-8")
    real_replace = report.os.replace
    failed = False

    def fail_once(source: object, destination: object) -> None:
        nonlocal failed
        source_path = Path(source)  # type: ignore[arg-type]
        destination_path = Path(destination)  # type: ignore[arg-type]
        if not failed and source_path.name == "new.txt":
            failed = True
            raise OSError("injected publish failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(report.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected publish failure"):
        report._publish_files(
            report.ReportPaths(docs),
            {
                Path("existing.txt"): "replacement\n",
                Path("new.txt"): "new\n",
            },
        )

    assert existing.read_text(encoding="utf-8") == "old\n"
    assert not (docs / "new.txt").exists()
