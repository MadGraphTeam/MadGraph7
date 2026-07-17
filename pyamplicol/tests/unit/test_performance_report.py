# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS))

import result_tables as report  # noqa: E402

from pyamplicol.api import BenchmarkResult, BenchmarkStatistics  # noqa: E402
from pyamplicol.cli import parse_cli  # noqa: E402
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
        for entry in payload["entries"]:
            if payload["kind"] == report.CacheKind.PROCESS_MATRIX:
                assert "legacy_amplicol" in entry
                assert "pyamplicol_jit_o3" in entry
                assert "generation_multiplier" in entry
                assert "runtime_multiplier" in entry
                assert "pointwise_validation" in entry
                assert "parameter_alignment" in entry
            else:
                assert "measurement" in entry
                if payload["kind"] == report.CacheKind.MODEL_LADDER:
                    assert "high_precision_matrix_element" in entry
                    assert "relative_difference" in entry


def test_reset_caches_are_canonical_na() -> None:
    caches = report.build_reset_caches()
    for payload in caches.values():
        assert payload["updated_at"] is None
        for entry in payload["entries"]:
            assert entry["status"] == report.NA_STATUS
            if payload["kind"] == report.CacheKind.PROCESS_MATRIX:
                assert _measurement_is_na(entry["legacy_amplicol"])
                assert _measurement_is_na(entry["pyamplicol_jit_o3"])
                assert _measurement_is_na(entry["reference"])
                assert _measurement_is_na(entry["pyamplicol"])
                assert entry["generation_multiplier"] is None
                assert entry["runtime_multiplier"] is None
                assert entry["relative_difference"] is None
            else:
                assert _measurement_is_na(entry["measurement"])


def test_process_matrices_preserve_families_and_multiplicity_grids() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    expected_keys = {family.key for family in report.PROCESS_FAMILIES}
    expected_lc_maxima = {
        family.key: family.maximum_lc_n for family in report.PROCESS_FAMILIES
    }

    for spec in report.MATRIX_SPECS:
        payload = caches[spec.cache_name]
        assert payload["benchmark_contract"]["config_overrides"] == dict(
            report.REPORT_CONFIG_OVERRIDES
        )
        assert (
            payload["benchmark_contract"]["config_overrides"][
                "evaluator.jit.optimization_level"
            ]
            == 3
        )
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
            assert payload["process_family"] == spec.process_family


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
        assert (
            r"\ReportNA" in expected_text
            or r"\matrixna" in expected_text
            or r"\texttt{N/A}" in expected_text
        )

    scalar_contact = expected_tables["result_scalar_contact_table.tex"]
    scalar_gravity = expected_tables["result_scalar_gravity_table.tex"]
    assert r"scalar\_0 scalar\_0 > X*scalar\_0" in scalar_contact
    assert r"scalar\_0 scalar\_0 > X*graviton" in scalar_gravity
    assert "evaluator [s/pt]" in scalar_contact
    assert "evaluator [s/pt]" in scalar_gravity
    assert "scalar\\_0 scalar\\_0 scalar\\_0" not in scalar_contact
    assert "graviton graviton" not in scalar_gravity


def test_matrix_renderer_shows_legacy_jit_o3_multipliers_only() -> None:
    spec = report.MATRIX_SPECS[0]
    payload = report.build_matrix_cache(spec)
    entry = payload["entries"][0]
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 4.0,
            "sample_count": 1,
            "wall_seconds_per_point": 8.0,
            "evaluator_seconds_per_point": None,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": 10.0,
            "requested_config": {},
            "effective_config": {},
            "metadata": {
                "old_matrix_format": {
                    "generation_s": 4.0,
                    "all_flow_generation_s": 6.0,
                    "runtime_us_per_point": 8.0,
                    "all_flow_runtime_us_per_point": 12.0,
                }
            },
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 2.0,
            "sample_count": 1,
            "wall_seconds_per_point": 1.0,
            "evaluator_seconds_per_point": None,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": 10.0,
            "requested_config": {},
            "effective_config": {},
            "metadata": {
                "old_matrix_format": {
                    "selected_generation_s": 2.0,
                    "all_flow_generation_s": 3.0,
                    "wall_us_per_point": 1.0,
                    "runtime_us_per_point": 0.5,
                    "all_flow_wall_us_per_point": 2.0,
                    "all_flow_runtime_us_per_point": 1.0,
                }
            },
        }
    )
    entry["legacy_amplicol"] = legacy
    entry["pyamplicol_jit_o3"] = pyamplicol
    entry["pointwise_validation"] = {
        **report._empty_validation(),
        "status": report.ResultStatus.OK.value,
        "reference_matrix_element": 10.0,
        "pyamplicol_matrix_element": 10.0,
        "absolute_difference": 0.0,
        "relative_difference": 0.0,
    }
    report._refresh_matrix_derived_fields(entry)

    table = report.render_matrix_table(spec, payload)

    assert r"\matrixcell" in table
    assert r"\textbf{gen O3 one-flow, hel. sum}" in table
    assert r"\textbf{gen O3 all-flows, one-hel}" in table
    assert r"\textbf{run O3 one-flow, hel. sum}" in table
    assert r"\textbf{run O3 all-flows, one-hel}" in table
    assert r"\matrixratio{ReportGreen}{0.5}" in table
    assert r"\matrixratiopair{ReportGreen}{x0.125}{ReportGreen}{0.0625}" in table
    assert report._matrix_py_over_ref_ratio(1.5) == r"\matrixratio{ReportOrange}{1.5}"
    assert report._matrix_py_over_ref_ratio(2.0) == r"\matrixratio{ReportRed}{2}"
    assert "C++" not in table
    assert "ASM" not in table


def test_lc_matrix_fallback_runtime_seconds_are_converted_consistently() -> None:
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "wall_seconds_per_point": 0.005,
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "wall_seconds_per_point": 0.010,
            "evaluator_seconds_per_point": 0.0075,
        }
    )

    reference = report._matrix_reference_pair(
        legacy,
        "runtime_us_per_point",
        "all_flow_runtime_us_per_point",
        report._matrix_plain_number,
        selected_fallback_key="wall_seconds_per_point",
        selected_fallback_scale=1.0e6,
    )
    ratio = report._matrix_lc_runtime_ratio(
        {**legacy, "metadata": {"old_matrix_format": {"runtime_us_per_point": 5000}}},
        pyamplicol,
        legacy_key="runtime_us_per_point",
        py_wall_key="wall_us_per_point",
        py_eval_key="runtime_us_per_point",
        py_wall_fallback_key="wall_seconds_per_point",
        py_eval_fallback_key="evaluator_seconds_per_point",
    )

    assert reference == r"\texttt{5e3}"
    assert ratio == r"\matrixratiopair{ReportRed}{x2}{ReportOrange}{1.5}"


def test_pointwise_validation_covers_all_flow_fixed_helicity() -> None:
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 10.0,
            "metadata": {
                "old_matrix_format": {
                    "all_flow_status": report.ResultStatus.OK.value,
                    "all_flow_reference_value": 20.0,
                }
            },
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 10.0,
            "metadata": {
                "old_matrix_format": {
                    "all_flow_status": report.ResultStatus.OK.value,
                    "all_flow_matrix_element": 21.0,
                }
            },
        }
    )

    validation = report._pointwise_validation(
        legacy,
        pyamplicol,
        require_all_flow=True,
    )

    assert validation["status"] == report.ResultStatus.VALIDATION_FAILED.value
    assert validation["all_flow_status"] == report.ResultStatus.VALIDATION_FAILED.value
    assert validation["all_flow_reference_matrix_element"] == 20.0
    assert validation["all_flow_pyamplicol_matrix_element"] == 21.0
    assert validation["all_flow_relative_difference"] == pytest.approx(0.05)


def test_z_ladder_revalidates_variants_when_reference_is_available() -> None:
    payload = report.build_ladder_cache(report.LADDER_SPECS[0])
    n_final = report.LADDER_SPECS[0].multiplicities[0]
    by_variant = {
        entry["variant"]: entry
        for entry in payload["entries"]
        if entry["n_final"] == n_final
    }
    reference = report._empty_measurement()
    reference.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 10.0,
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "all_flow_status": report.ResultStatus.OK.value,
                    "all_flow_reference_value": 20.0,
                }
            },
        }
    )
    observed = report._empty_measurement()
    observed.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 11.0,
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "all_flow_status": report.ResultStatus.OK.value,
                    "all_flow_matrix_element": 20.0,
                }
            },
        }
    )
    by_variant["reference"].update(status="ok", measurement=reference)
    by_variant["jit_o3"].update(status="ok", measurement=observed)

    report._refresh_performance_ladder_validation(payload, n_final=n_final)

    entry = next(
        item
        for item in payload["entries"]
        if item["n_final"] == n_final and item["variant"] == "jit_o3"
    )
    assert entry["status"] == report.ResultStatus.VALIDATION_FAILED.value
    validation = entry["measurement"]["metadata"]["pointwise_validation"]
    assert validation["status"] == report.ResultStatus.VALIDATION_FAILED.value
    assert validation["all_flow_status"] == report.ResultStatus.OK.value


def test_z_tables_use_old_selected_and_all_flow_layout() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    builtin = report.render_performance_ladder(
        report.LADDER_SPECS[0],
        caches["z_builtin_sm.json"],
    )
    ufo = report.render_performance_ladder(
        report.LADDER_SPECS[1],
        caches["z_external_sm.json"],
        built_in_payload=caches["z_builtin_sm.json"],
    )

    assert r"\begin{longtable}{@{}r L{0.92in} L{0.48in} L{0.86in}" in builtin
    assert r"\textbf{selected flow, helicity sum}" in builtin
    assert r"\textbf{all flows, fixed helicity}" in builtin
    assert r"\PAC\ JIT \(\mathrm{O}1\)" in builtin
    assert r"\PAC\ C++ \(\mathrm{O}3\)" in builtin
    assert "Staged/Rusticol performance summary" not in builtin

    assert "UFO-SM Dedicated" in ufo
    assert r"\textbf{vs blt-in}" in ufo
    assert "External-SM" not in ufo


def test_z_reproduction_commands_use_the_current_cli() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    rendered = report.render_performance_ladder(
        report.LADDER_SPECS[0],
        caches["z_builtin_sm.json"],
    )
    listing = re.search(
        r"\\begin\{lstlisting\}.*?\n(.*?)\\end\{lstlisting\}",
        rendered,
        re.S,
    )
    assert listing is not None
    commands = [line for line in listing.group(1).splitlines() if line.strip()]
    assert len(commands) == 2
    assert "generate-process" not in rendered
    assert "time-process" not in rendered

    for command in commands:
        tokens = shlex.split(command)
        assert tokens[:3] == [".venv/bin/python", "-m", "pyamplicol"]
        parse_cli(tokens[3:])


def test_missing_only_treats_generic_z_rows_as_stale() -> None:
    payload = report.build_ladder_cache(report.LADDER_SPECS[0])
    entry = payload["entries"][0]
    measurement = report._empty_measurement()
    measurement.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "sample_count": 1,
            "wall_seconds_per_point": 2.0e-6,
            "evaluator_seconds_per_point": 1.0e-6,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": 1.0,
            "requested_config": {},
            "effective_config": {},
        }
    )
    entry["status"] = report.ResultStatus.OK.value
    entry["measurement"] = measurement

    cells = report._select_cells(
        datasets={"z_builtin_sm"},
        variants={entry["variant"]},
        n_values={entry["n_final"]},
        missing_only=True,
        caches={report.LADDER_SPECS[0].cache_name: payload},
    )

    assert len(cells) == 1


def test_failure_status_labels_render_in_cells() -> None:
    measurement = report._failure_measurement(
        report.ResultStatus.MEMORY_LIMIT,
        "too large",
        limit_gib=800,
    )
    assert report._measurement_cell(measurement) == r"\ReportStatus{RAM>800G}"
    assert (
        report._measurement_cell(
            report._failure_measurement(report.ResultStatus.TIMEOUT, "slow")
        )
        == r"\ReportStatus{t/o}"
    )
    assert (
        report._measurement_cell(
            report._failure_measurement(
                report.ResultStatus.VALIDATION_FAILED,
                "mismatch",
            )
        )
        == r"\ReportStatus{VALIDATION FAILED}"
    )


def test_campaign_schedule_is_fast_first_and_duplicates_sm_tables() -> None:
    cells = report._select_cells(limit=20)
    assert cells
    assert cells[0].n_final == 1
    matrix_datasets = {spec.dataset_id for spec in report.MATRIX_SPECS}
    z_datasets = {
        spec.dataset_id
        for spec in report.LADDER_SPECS
        if spec.kind == report.CacheKind.PERFORMANCE_LADDER
    }
    scheduled_datasets = {cell.dataset_id for cell in report._campaign_cells()}
    assert matrix_datasets <= scheduled_datasets
    assert {"z_builtin_sm", "z_external_sm"} <= z_datasets <= scheduled_datasets
    assert any(cell.dataset_id == "matrix_builtin_sm_lc" for cell in cells)
    assert any(
        cell.dataset_id == "matrix_external_sm_lc" for cell in report._campaign_cells()
    )


def test_campaign_workers_serialize_symbolica_by_default() -> None:
    requested, scheduled_cap, effective, reason = report._campaign_worker_selection(
        50,
        459,
        allow_symbolica_parallel=False,
    )
    assert requested == 50
    assert scheduled_cap == 50
    assert effective == 1
    assert reason is not None
    assert "Symbolica" in reason

    requested, scheduled_cap, effective, reason = report._campaign_worker_selection(
        50,
        459,
        allow_symbolica_parallel=True,
    )
    assert requested == 50
    assert scheduled_cap == 50
    assert effective == 50
    assert reason is None


def test_campaign_worker_timeout_accounts_for_every_generation_workload() -> None:
    cells = report._campaign_cells()
    matrix_lc = next(
        cell for cell in cells if cell.dataset_id == "matrix_builtin_sm_lc"
    )
    matrix_nlc = next(
        cell for cell in cells if cell.dataset_id == "matrix_builtin_sm_nlc"
    )
    z_reference = next(
        cell
        for cell in cells
        if cell.dataset_id == "z_builtin_sm" and cell.variant == "reference"
    )
    z_jit = next(
        cell
        for cell in cells
        if cell.dataset_id == "z_builtin_sm" and cell.variant == "jit_o3"
    )

    assert report._campaign_worker_timeout_seconds(matrix_lc, 3600) == 11_700
    assert report._campaign_worker_timeout_seconds(matrix_nlc, 3600) == 8_100
    assert report._campaign_worker_timeout_seconds(z_reference, 3600) == 4_500
    assert report._campaign_worker_timeout_seconds(z_jit, 3600) == 8_100
    assert report._campaign_worker_timeout_seconds(matrix_lc, 0) is None


def test_memory_limit_exit_requires_the_watchdog_marker(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("process died with status 137\n", encoding="utf-8")
    assert not report._worker_log_reports_memory_limit(log)
    log.write_text(
        "memory-watchdog: RSS limit exceeded: 31 GiB\n",
        encoding="utf-8",
    )
    assert report._worker_log_reports_memory_limit(log)


def test_legacy_measurement_freshness_tracks_the_contributor_lock() -> None:
    from tools.developer import legacy_amplicol

    current = legacy_amplicol.expected_revision()
    assert report._legacy_measurement_revision_current(
        {"environment": {"revision": current}}
    )
    assert not report._legacy_measurement_revision_current(
        {"environment": {"revision": "0" * 40}}
    )


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


def test_cell_merges_reload_latest_caches_under_writer_lock(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    paths = report.ReportPaths(docs)
    first_service = report.ReportService(paths)
    second_service = report.ReportService(paths)
    first_service._refresh(report.build_reset_caches(), compile_pdf=False)
    cells = tuple(
        cell
        for cell in report._campaign_cells()
        if cell.kind == "matrix"
        and cell.cache_name == report.MATRIX_SPECS[0].cache_name
    )[:2]
    assert len(cells) == 2

    for service, cell in zip((first_service, second_service), cells, strict=True):
        entry = report._failure_entry_for_cell(
            cell,
            status=report.ResultStatus.ERROR,
            message=f"failure for {cell.cell_id}",
            artifact_root=tmp_path / "artifacts",
            limit_gib=800,
            timeout_seconds=3600,
        )
        service._merge_and_refresh(cell, entry, compile_pdf=False)

    caches = report.load_caches(paths)
    entries = caches[cells[0].cache_name]["entries"]
    statuses = {
        (entry["process_key"], entry["n_final"]): entry["status"]
        for entry in entries
        if isinstance(entry, dict)
    }
    for cell in cells:
        assert statuses[(cell.process_key, cell.n_final)] == report.ResultStatus.ERROR
