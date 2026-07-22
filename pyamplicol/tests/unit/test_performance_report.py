# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import re
import signal
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

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


def test_generation_slice_import_guard_restores_repo_root() -> None:
    root = os.fspath(ROOT)
    resolved_root = os.fspath(ROOT.resolve())
    original = list(sys.path)
    try:
        sys.path[:] = [
            entry for entry in sys.path if entry not in {"", root, resolved_root}
        ]
        GenerationSlice, generate_slice = report._generation_slice_tools()
        assert root in sys.path or resolved_root in sys.path
        assert GenerationSlice.__name__ == "GenerationSlice"
        assert callable(generate_slice)
    finally:
        sys.path[:] = original


def test_checked_in_caches_match_schema_and_are_reset_to_na() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    assert len(caches) == 13
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
            elif payload["kind"] == report.CacheKind.EAGER_PROCESS_MATRIX:
                assert "eager_jit_o3" in entry
                assert "pointwise_validation" in entry
                assert "selector_contract" in entry
                assert "legacy_amplicol" not in entry
                assert "pyamplicol_jit_o3" not in entry
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
            elif payload["kind"] == report.CacheKind.EAGER_PROCESS_MATRIX:
                assert _measurement_is_na(entry["eager_jit_o3"])
                assert entry["pointwise_validation"] == report._empty_validation()
                assert (
                    entry["selector_contract"]
                    == report._empty_eager_selector_contract()
                )
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
        "z_external_sm": list(range(1, 10)),
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

    rendered_tables = "\n".join(expected_tables.values())
    assert (
        r"\ReportNA" in rendered_tables
        or r"\matrixna" in rendered_tables
        or r"\texttt{N/A}" in rendered_tables
    )
    for name, expected_text in expected_tables.items():
        path = DOCS / name
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == expected_text
        assert expected_text.startswith("% SPDX-License-Identifier: 0BSD\n")

    scalar_contact = expected_tables["result_scalar_contact_table.tex"]
    scalar_gravity = expected_tables["result_scalar_gravity_table.tex"]
    assert r"scalar\_0 scalar\_0 > n*scalar\_0" in scalar_contact
    assert r"scalar\_0 scalar\_0 > n*graviton" in scalar_gravity
    assert r"\textbf{\texttt{n=2}}" in scalar_contact
    assert "status &" not in scalar_contact
    assert "status &" not in scalar_gravity
    assert "generation [s]" in scalar_contact
    assert "generation [s]" in scalar_gravity
    assert r"evaluator [$\mu$s/pt]" in scalar_contact
    assert r"evaluator [$\mu$s/pt]" in scalar_gravity
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


def test_lc_matrix_renderer_shows_py_timings_when_reference_is_unavailable() -> None:
    spec = report.MATRIX_SPECS[0]
    payload = report.build_matrix_cache(spec)
    entry = next(
        candidate
        for candidate in payload["entries"]
        if candidate["process_key"] == "dd_4q_lines" and candidate["n_final"] == 6
    )
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.UNSUPPORTED.value,
            "failure_message": (
                f"{report.ORIGINAL_AMPLICOL_OPEN_LINE_LIMIT_REASON}; "
                "d d~ > u u~ s s~ c c~ has 4 open quark lines"
            ),
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.UNSUPPORTED.value,
                    "all_flow_status": report.ResultStatus.UNSUPPORTED.value,
                    "reference_unavailable_reason": (
                        report.ORIGINAL_AMPLICOL_OPEN_LINE_LIMIT_REASON
                    ),
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
            "wall_seconds_per_point": 1.0e-6,
            "evaluator_seconds_per_point": 0.5e-6,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": 10.0,
            "requested_config": {},
            "effective_config": {},
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "selected_generation_s": 2.0,
                    "all_flow_generation_s": 3.0,
                    "wall_us_per_point": 1.0,
                    "runtime_us_per_point": 0.5,
                    "all_flow_wall_us_per_point": 2.0,
                    "all_flow_runtime_us_per_point": 1.0,
                    "all_flow_status": report.ResultStatus.OK.value,
                }
            },
        }
    )
    entry["legacy_amplicol"] = legacy
    entry["pyamplicol_jit_o3"] = pyamplicol
    report._refresh_matrix_derived_fields(entry)

    assert report._matrix_reference_unavailable_by_design(entry)

    table = report.render_matrix_table(spec, payload)

    assert "UNSUPPORTED" not in table
    assert r"\texttt{2 s}" in table
    assert r"\texttt{3 s}" in table
    assert r"\texttt{1 us}" in table
    assert r"\texttt{0.5 us}" in table
    assert r"\texttt{2 us}" in table


def test_lc_matrix_renderer_localizes_union_out_of_reach() -> None:
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 10.0,
            "wall_seconds_per_point": 3.0e-6,
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "generation_s": 10.0,
                    "runtime_us_per_point": 3.0,
                    "all_flow_status": report.ResultStatus.OK.value,
                    "all_flow_generation_s": 10.0,
                    "all_flow_runtime_us_per_point": 4.0,
                }
            },
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 2.0,
            "wall_seconds_per_point": 1.0e-6,
            "evaluator_seconds_per_point": 0.5e-6,
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "selected_generation_s": 2.0,
                    "wall_us_per_point": 1.0,
                    "runtime_us_per_point": 0.5,
                    "all_flow_status": report.ResultStatus.OUT_OF_REACH.value,
                    "all_flow_error": "union lane is too large",
                },
                "selected_flow_measurement": {
                    **report._empty_measurement(),
                    "status": report.ResultStatus.OK.value,
                },
                "all_flow_measurement": report._failure_measurement(
                    report.ResultStatus.OUT_OF_REACH,
                    "union lane is too large",
                    metadata={
                        "lane_status_policy": report.LC_LANE_STATUS_POLICY,
                        "runtime_selector_role": "all-flows-fixed-helicity",
                        "lc_flow_layout": report.LC_ALL_FLOW_UNION_LAYOUT,
                    },
                ),
            },
        }
    )
    entry = {
        "applicable": True,
        "status": report.ResultStatus.OK.value,
        "legacy_amplicol": legacy,
        "pyamplicol_jit_o3": pyamplicol,
    }

    cell = report._matrix_cell(entry, color_accuracy="lc")

    assert r"\matrixratio{ReportGreen}{0.2}" in cell
    assert r"\texttt{out-of-reach}" in cell


def test_lc_matrix_reference_runtime_pair_uses_tight_spacing() -> None:
    macros = "\n".join(report._matrix_table_macros())

    assert r"\matrixslot{0.86in}{#4}" in macros
    assert r"\makebox[0.27in][l]{#1}" in macros
    assert r"\hspace{0.006in}\matrixpunct{/}\hspace{0.012in}" in macros


def _synthetic_source_provenance(
    compiled_model_schema_version: int,
    model_compiler_version: int,
) -> dict[str, object]:
    return {
        "head": report._git_rev_parse("HEAD"),
        "report_version": report.REPORT_VERSION,
        "cache_schema_version": report.CACHE_SCHEMA_VERSION,
        "compiled_model_schema_version": compiled_model_schema_version,
        "model_compiler_version": model_compiler_version,
    }


def _write_current_effective_config(artifact_dir: Path) -> None:
    config_dir = artifact_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_chunk_size = report.REPORT_CONFIG_OVERRIDES["evaluator.output_chunk_size"]
    (config_dir / "effective.toml").write_text(
        f"[evaluator]\noutput_chunk_size = {output_chunk_size}\n",
        encoding="utf-8",
    )


def _write_current_report_artifact(
    artifact_dir: Path,
    *,
    layout: str | None = None,
    producer_version: str = "current",
    process_artifact_schema: int = report.PROCESS_ARTIFACT_SCHEMA_VERSION,
    compiled_model_schema: int = 9,
) -> None:
    _write_current_effective_config(artifact_dir)
    if layout is not None:
        config_path = artifact_dir / "config" / "effective.toml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + f'\n[color]\nlc_flow_layout = "{layout}"\n',
            encoding="utf-8",
        )
    (artifact_dir / "artifact.json").write_text(
        json.dumps(
            {
                "producer": {
                    "version": producer_version,
                    "versions": {
                        "process_artifact": process_artifact_schema,
                        "compiled_model": compiled_model_schema,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "compiled-model.json").write_text(
        json.dumps(
            {"schema_version": compiled_model_schema, "model_compiler_version": 13}
        ),
        encoding="utf-8",
    )


def test_report_source_provenance_records_git_and_compiler_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report._report_source_provenance.cache_clear()
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 12))
    try:
        provenance = report._report_source_provenance()
    finally:
        report._report_source_provenance.cache_clear()

    assert provenance["head"] == report._git_rev_parse("HEAD")
    assert provenance["report_version"] == report.REPORT_VERSION
    assert provenance["cache_schema_version"] == report.CACHE_SCHEMA_VERSION
    assert provenance["compiled_model_schema_version"] == 9
    assert provenance["model_compiler_version"] == 12


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


def test_pointwise_validation_keeps_selected_ok_when_union_lane_out_of_reach() -> None:
    legacy = report._empty_measurement()
    legacy.update(
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
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 10.0,
            "metadata": {
                "old_matrix_format": {
                    "status": report.ResultStatus.OK.value,
                    "all_flow_status": report.ResultStatus.OUT_OF_REACH.value,
                    "all_flow_error": "union lane is too large",
                }
            },
        }
    )

    validation = report._pointwise_validation(
        legacy,
        pyamplicol,
        require_all_flow=True,
    )
    eager_validation = report._eager_pointwise_validation(
        legacy,
        pyamplicol,
        require_all_flow=True,
    )

    assert validation["status"] == report.ResultStatus.OK.value
    assert validation["all_flow_status"] == report.ResultStatus.OUT_OF_REACH.value
    assert eager_validation["status"] == report.ResultStatus.OK.value
    assert (
        eager_validation["all_flow_status"]
        == report.ResultStatus.OUT_OF_REACH.value
    )


def test_selected_flow_reference_order_prefers_source_mapped_order() -> None:
    cell = report.CampaignCell(
        kind="matrix",
        cache_name="matrix_builtin_sm_lc.json",
        dataset_id="matrix_builtin_sm_lc",
        n_final=2,
        process="d d~ > z g",
        process_key="dd_z_jets",
    )
    old_fields = {
        "reference_color_order": [2, 4, 1, 3],
        "reference_color_order_process_file": [2, 3, 1, 4],
    }

    assert report._selected_flow_reference_color_order(old_fields, cell) == [
        2,
        4,
        1,
        3,
    ]
    assert report._selected_flow_reference_color_order(
        {"reference_color_order_process_file": [2, 3, 1, 4]},
        cell,
    ) == [2, 3, 1, 4]


def test_generated_library_probe_parser_requires_matching_row() -> None:
    output = "AMPICOL_PROBE_VALUE 1 3 2 1.60358797632899820000000E+000\n"

    assert report._parse_legacy_generated_library_probe_value(
        output,
        expected_group=3,
        expected_integral=2,
    ) == pytest.approx(1.6035879763289982)
    with pytest.raises(RuntimeError, match="row mismatch"):
        report._parse_legacy_generated_library_probe_value(
            output,
            expected_group=1,
            expected_integral=2,
        )


def test_legacy_lc_color_probe_scope_matches_fortran_probe_limit() -> None:
    assert report._legacy_lc_color_probe_supported((1, -1, 23, 21, 21, 21))
    assert not report._legacy_lc_color_probe_supported((1, -1, 2, -2, 3, -3))
    assert report._legacy_direct_color_probe_supported((1, -1, 2, -2, 3, -3))
    assert not report._legacy_direct_color_probe_supported((1, -1, 2, -2, 3, -3, 4, -4))
    assert report._legacy_probe_scope_limited(
        "d d~ > u u~ s s~: 3 quark lines exceed the legacy scope"
    )


def test_selected_flow_library_probe_record_uses_indexed_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.developer import legacy_amplicol

    entry = SimpleNamespace(group=4, integral=2)
    calls: list[dict[str, object]] = []

    def fake_probe(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        assert os.fspath(tmp_path) in os.environ["LD_LIBRARY_PATH"].split(":")
        return SimpleNamespace(
            value=1.25,
            group=4,
            integral=2,
            process_pdgs=(1, -1, 2, -2, 3, -3),
            color_order=(2, 1, 3, 4, 5, 6),
            amplitudes=11,
            color_factor=3,
            identical_factor=1,
            singlet_vertices=0,
            normalization=0.25,
            value_decimal=Decimal("1.25"),
            normalization_decimal=Decimal("0.25"),
        )

    monkeypatch.setattr(
        legacy_amplicol,
        "run_selected_flow_library_probe",
        fake_probe,
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    record, value, payload = report._legacy_run_selected_flow_library_probe_record(
        tmp_path,
        entry=entry,
        source_pdgs=(1, -1, 2, -2, 3, -3),
        momenta=(),
        points=1,
        output_path=tmp_path / "selected-flow-library-probe.json",
    )

    assert value == pytest.approx(1.25)
    assert record["returncode"] == 0
    assert record["cwd"] == os.fspath(tmp_path.resolve(strict=False))
    assert record["env"]["LD_LIBRARY_PATH"].split(":")[0] == os.fspath(
        tmp_path.resolve(strict=False)
    )
    assert record["executable"] == os.fspath(
        (tmp_path / "amplicol_library_benchmark").resolve(strict=False)
    )
    assert calls[0]["kwargs"]["entry"] is entry
    assert calls[0]["kwargs"]["source_pdgs"] == (1, -1, 2, -2, 3, -3)
    assert os.environ["LD_LIBRARY_PATH"] == "/existing"
    assert payload["process_pdgs"] == [1, -1, 2, -2, 3, -3]
    assert (
        json.loads(Path(record["output_path"]).read_text(encoding="utf-8"))[
            "value_decimal"
        ]
        == "1.25"
    )


def test_snapshot_legacy_generated_library_preserves_executable_and_libs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    destination = tmp_path / "artifact"
    repository.mkdir()
    destination.mkdir()
    (destination / "stale").write_text("old\n", encoding="utf-8")
    (repository / "libamp1.so").write_bytes(b"library")
    (repository / "Library").mkdir()
    (repository / "Library" / "amplitudes.bin").write_bytes(b"amplitudes")
    (repository / "amplicol_library_benchmark").write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )
    (repository / "amplicol_generate").write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )
    process_file = repository / "processes.txt"
    process_file.write_text("d d~ > z\n", encoding="utf-8")

    record = report._snapshot_legacy_generated_library(
        repository,
        destination,
        required_executables=("amplicol_library_benchmark",),
        process_file=process_file,
    )

    assert not (destination / "stale").exists()
    assert (destination / "libamp1.so").read_bytes() == b"library"
    assert (destination / "Library" / "amplitudes.bin").read_bytes() == b"amplitudes"
    assert (destination / "amplicol_library_benchmark").is_file()
    assert (destination / "amplicol_generate").is_file()
    assert (destination / "processes.txt").read_text(encoding="utf-8") == "d d~ > z\n"
    copied = {Path(item["path"]).name: item for item in record["files"]}
    assert copied["libamp1.so"]["sha256"] == report._file_sha256(
        destination / "libamp1.so"
    )
    assert record["artifact_path"] == os.fspath(destination)


def test_lc_partition_matrix_element_uses_source_mapped_reference_order() -> None:
    probe = SimpleNamespace(
        lc_row_partitions=(
            SimpleNamespace(row=1, value=1.6035879763289982, permutation=(2, 3, 1)),
            SimpleNamespace(row=2, value=0.084286088058938, permutation=(2, 1, 3)),
        )
    )

    value, metadata = report._legacy_lc_partition_matrix_element(
        probe,
        reference_color_order=[2, 4, 1, 3],
        source_to_row_permutation=(0, 1, 3, 2),
    )

    assert value == pytest.approx(1.6035879763289982)
    assert metadata["reference_color_order_coloured"] == [2, 4, 1]
    assert metadata["reference_lc_partition_row"] == 1


def test_lc_selected_flow_matrix_element_sums_fixed_helicity_partitions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, ...]] = []
    entry = SimpleNamespace(process_pdgs=(21, 21, 25, 25))

    def fake_probe(*_args: object, helicities: tuple[int, ...], **_kwargs: object):
        calls.append(tuple(helicities))
        value = 1.0 if helicities[0] == helicities[1] else 2.0
        probe = SimpleNamespace(
            lc_row_partitions=(SimpleNamespace(row=1, value=value, permutation=(1, 2)),)
        )
        return {"args": ["probe", *map(str, helicities)]}, [], probe

    monkeypatch.setattr(report, "_legacy_run_color_probe_timed", fake_probe)

    value, commands, _rows, metadata = report._legacy_lc_selected_flow_matrix_element(
        tmp_path,
        process_file=tmp_path / "processes.txt",
        entry=entry,
        source_pdgs=(21, 21, 25, 25),
        momenta=(),
        reference_color_order=[1, 2, 3, 4],
    )

    assert value == pytest.approx(6.0)
    assert calls == [(-1, -1, 0, 0), (-1, 1, 0, 0), (1, -1, 0, 0), (1, 1, 0, 0)]
    assert len(commands) == 4
    assert metadata["selected_flow_helicity_count"] == 4


def test_legacy_adaptive_profile_points_are_target_runtime_bounded() -> None:
    assert (
        report._legacy_adaptive_profile_points(
            0.05,
            target_runtime=20.0,
        )
        == 40_000
    )
    assert (
        report._legacy_adaptive_profile_points(
            1.0e-9,
            target_runtime=20.0,
        )
        == report.DEFAULT_LEGACY_PROFILE_MAX_POINTS
    )
    assert (
        report._legacy_adaptive_profile_points(
            1000.0,
            target_runtime=20.0,
        )
        == report.DEFAULT_LEGACY_PROFILE_MIN_POINTS
    )


def test_legacy_profiled_command_warms_up_and_selects_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_command_record(
        args: list[str],
        *,
        cwd: Path,
        env: object | None = None,
    ) -> tuple[dict[str, object], str]:
        del cwd, env
        count = int(args[1])
        calls.append(count)
        seconds = 0.05 if count == report.DEFAULT_LEGACY_PROFILE_WARMUP_POINTS else 20.0
        return (
            {"args": args, "cwd": str(tmp_path), "elapsed_s": seconds, "returncode": 0},
            f"Timing summary\namplitude evaluation {seconds}\n",
        )

    monkeypatch.setattr(report, "_legacy_command_record", fake_command_record)

    record, _output, _rows, points, profile = report._legacy_run_command_profiled(
        lambda count: ["bench", str(count)],
        cwd=tmp_path,
        env=None,
        target_runtime=20.0,
        probe="bench",
        timing_labels=("amplitude evaluation",),
    )

    assert calls == [report.DEFAULT_LEGACY_PROFILE_WARMUP_POINTS, 40_000]
    assert points == 40_000
    assert record["profile_points"] == 40_000
    assert profile["legacy_profile_policy"] == report.LEGACY_PROFILE_POLICY
    assert profile["measurement_points"] == 40_000


def test_legacy_color_probe_profile_passes_color_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []
    probe = SimpleNamespace(value=2.0)

    def fake_probe(
        *_args: object,
        color_accuracy: str,
        points: int,
        **_kwargs: object,
    ) -> tuple[dict[str, object], list[dict[str, object]], object]:
        calls.append((color_accuracy, points))
        return (
            {"args": ["probe"], "elapsed_s": 0.1, "returncode": 0},
            [{"label": "total", "seconds": 0.1}],
            probe,
        )

    monkeypatch.setattr(report, "_legacy_run_color_probe_timed", fake_probe)

    _record, _rows, returned, points, profile = report._legacy_run_color_probe_profiled(
        tmp_path,
        process_file=tmp_path / "processes.txt",
        entry=SimpleNamespace(group=1, integral=1, process_pdgs=(1, -1)),
        source_pdgs=(1, -1),
        momenta=(),
        color_accuracy="full",
        helicities=None,
        target_runtime=0.2,
    )

    assert returned is probe
    assert calls == [
        ("full", report.DEFAULT_LEGACY_PROFILE_WARMUP_POINTS),
        ("full", points),
    ]
    assert profile["probe"] == "amplicol_color_probe"


def test_terminate_worker_process_sends_process_group_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | float | None, int | None]] = []

    class FakeProcess:
        pid = 42

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            calls.append(("wait", timeout, None))
            return -signal.SIGTERM

    def fake_killpg(pid: int, sig: int) -> None:
        calls.append(("killpg", None, sig))
        assert pid == 42

    monkeypatch.setattr(report.os, "killpg", fake_killpg)

    report._terminate_worker_process(FakeProcess(), grace_seconds=1.5)  # type: ignore[arg-type]

    assert calls == [
        ("killpg", None, signal.SIGTERM),
        ("wait", 1.5, None),
    ]


def test_missing_only_retries_matrix_error_rows() -> None:
    payload = report.build_matrix_cache(report.MATRIX_SPECS[0])
    entry = payload["entries"][0]
    entry["status"] = report.ResultStatus.ERROR.value
    entry["legacy_amplicol"] = {
        **report._empty_measurement(),
        "status": report.ResultStatus.ERROR.value,
    }
    cell = report.CampaignCell(
        kind="matrix",
        cache_name=report.MATRIX_SPECS[0].cache_name,
        dataset_id=report.MATRIX_SPECS[0].dataset_id,
        n_final=int(entry["n_final"]),
        process=str(entry["process"]),
        process_key=str(entry["process_key"]),
    )

    assert report._campaign_cell_needs_measurement(
        cell,
        {report.MATRIX_SPECS[0].cache_name: payload},
    )


def test_missing_only_retries_z_unsupported_rows() -> None:
    spec = report.LADDER_SPECS[0]
    payload = report.build_ladder_cache(spec)
    entry = next(
        item
        for item in payload["entries"]
        if item["n_final"] == 1 and item["variant"] == "asm_o3"
    )
    entry["measurement"] = {
        **report._empty_measurement(),
        "status": report.ResultStatus.UNSUPPORTED.value,
    }
    cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=spec.process(int(entry["n_final"])),
        variant=str(entry["variant"]),
    )

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})


def test_missing_only_skips_out_of_reach_z_rows() -> None:
    spec = report.LADDER_SPECS[0]
    payload = report.build_ladder_cache(spec)
    entry = next(
        item
        for item in payload["entries"]
        if item["n_final"] == 1 and item["variant"] == "eager_jit_o3"
    )
    entry["measurement"] = {
        **report._empty_measurement(),
        "status": report.ResultStatus.OUT_OF_REACH.value,
    }
    cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=spec.process(int(entry["n_final"])),
        variant=str(entry["variant"]),
    )

    assert not report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})


def test_known_eager_z_out_of_reach_is_migrated_to_union_lane_only() -> None:
    spec = next(
        item for item in report.LADDER_SPECS if item.dataset_id == "z_builtin_sm"
    )
    payload = report.build_ladder_cache(spec)
    entry = next(
        item
        for item in payload["entries"]
        if item["n_final"] == 8 and item["variant"] == "eager_jit_o3"
    )
    entry["status"] = report.ResultStatus.OUT_OF_REACH.value
    entry["measurement"] = report._failure_measurement(
        report.ResultStatus.OUT_OF_REACH,
        "campaign policy stop",
        metadata={
            "campaign_classification": {
                "policy": "high_multiplicity_out_of_reach_v1",
                "status": report.ResultStatus.OUT_OF_REACH.value,
            },
            "cell": {
                "dataset_id": "z_builtin_sm",
                "n_final": 8,
                "variant": "eager_jit_o3",
            },
        },
    )

    normalized = report.normalize_cache_payload(payload)
    migrated = next(
        item
        for item in normalized["entries"]
        if item["n_final"] == 8 and item["variant"] == "eager_jit_o3"
    )
    measurement = migrated["measurement"]
    assert migrated["status"] == report.NA_STATUS
    assert measurement["status"] == report.NA_STATUS
    old = report._measurement_old_matrix_fields(measurement)
    assert old["status"] == report.NA_STATUS
    assert old["all_flow_status"] == report.ResultStatus.OUT_OF_REACH.value
    row = report._z_old_row_from_measurement(
        measurement,
        variant_key="eager_jit_o3",
    )
    assert row["status"] == "missing"
    assert row["all_flow_status"] == "out_of_reach"
    cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=8,
        process=spec.process(8),
        variant="eager_jit_o3",
    )
    assert report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: normalized},
    )


def test_ambiguous_whole_row_out_of_reach_remains_terminal() -> None:
    spec = next(
        item for item in report.LADDER_SPECS if item.dataset_id == "z_builtin_sm"
    )
    payload = report.build_ladder_cache(spec)
    entry = next(
        item
        for item in payload["entries"]
        if item["n_final"] == 8 and item["variant"] == "eager_jit_o3"
    )
    entry["status"] = report.ResultStatus.OUT_OF_REACH.value
    entry["measurement"] = report._failure_measurement(
        report.ResultStatus.OUT_OF_REACH,
        "generic whole-cell policy stop",
    )

    normalized = report.normalize_cache_payload(payload)
    terminal = next(
        item
        for item in normalized["entries"]
        if item["n_final"] == 8 and item["variant"] == "eager_jit_o3"
    )
    assert terminal["status"] == report.ResultStatus.OUT_OF_REACH.value
    assert not report._lc_measurement_has_explicit_lanes(terminal["measurement"])
    cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=8,
        process=spec.process(8),
        variant="eager_jit_o3",
    )
    assert not report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: normalized},
    )


def test_missing_only_retries_model_ladder_failures() -> None:
    spec = next(
        item
        for item in report.LADDER_SPECS
        if item.kind == report.CacheKind.MODEL_LADDER
    )
    payload = report.build_ladder_cache(spec)
    entry = payload["entries"][0]
    entry["measurement"] = {
        **report._empty_measurement(),
        "status": report.ResultStatus.TIMEOUT.value,
    }
    cell = report.CampaignCell(
        kind="model_ladder",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=spec.process(int(entry["n_final"])),
    )

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})


def test_missing_only_accepts_builtin_schema8_artifacts_after_schema9_bump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = next(
        spec
        for spec in report.MATRIX_SPECS
        if spec.dataset_id == "matrix_builtin_sm_nlc"
    )
    payload = report.build_matrix_cache(spec)
    entry = payload["entries"][0]
    artifact_dir = tmp_path / "artifact"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    (artifact_dir / "artifact.json").write_text(
        json.dumps({"producer": {"version": "current"}}),
        encoding="utf-8",
    )
    _write_current_effective_config(artifact_dir)
    compiled_model = model_dir / "compiled-model.json"
    compiled_model.write_text(
        json.dumps({"schema_version": 7, "model_compiler_version": 6}),
        encoding="utf-8",
    )
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "requested_config": report._legacy_profile_requested_config(20.0),
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "artifact_path": str(artifact_dir),
            "environment": {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            },
            "metadata": {
                "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
                "generation_timer_excludes_model_compile": True,
                "source_provenance": _synthetic_source_provenance(9, 9),
            },
        }
    )
    entry.update(
        {
            "status": report.ResultStatus.OK.value,
            "legacy_amplicol": legacy,
            "pyamplicol_jit_o3": pyamplicol,
        }
    )
    cell = report.CampaignCell(
        kind="matrix",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=str(entry["process"]),
        process_key=str(entry["process_key"]),
    )
    monkeypatch.setattr(report, "_legacy_measurement_revision_current", lambda _: True)
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 9))
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})

    compiled_model.write_text(
        json.dumps({"schema_version": 8, "model_compiler_version": 8}),
        encoding="utf-8",
    )

    assert not report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: payload},
    )

    compiled_model.write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 10}),
        encoding="utf-8",
    )

    assert not report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: payload},
    )


def test_missing_only_retries_external_schema8_artifacts_after_schema9_bump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = next(
        spec
        for spec in report.MATRIX_SPECS
        if spec.dataset_id == "matrix_external_sm_nlc"
    )
    payload = report.build_matrix_cache(spec)
    entry = payload["entries"][0]
    artifact_dir = tmp_path / "artifact"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    (artifact_dir / "artifact.json").write_text(
        json.dumps({"producer": {"version": "current"}}),
        encoding="utf-8",
    )
    _write_current_effective_config(artifact_dir)
    compiled_model = model_dir / "compiled-model.json"
    compiled_model.write_text(
        json.dumps({"schema_version": 8, "model_compiler_version": 8}),
        encoding="utf-8",
    )
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "requested_config": report._legacy_profile_requested_config(20.0),
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "artifact_path": str(artifact_dir),
            "environment": {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            },
            "metadata": {
                "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
                "generation_timer_excludes_model_compile": True,
                "source_provenance": _synthetic_source_provenance(9, 9),
            },
        }
    )
    entry.update(
        {
            "status": report.ResultStatus.OK.value,
            "legacy_amplicol": legacy,
            "pyamplicol_jit_o3": pyamplicol,
        }
    )
    cell = report.CampaignCell(
        kind="matrix",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=str(entry["process"]),
        process_key=str(entry["process_key"]),
    )
    monkeypatch.setattr(report, "_legacy_measurement_revision_current", lambda _: True)
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 9))
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})

    compiled_model.write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 8}),
        encoding="utf-8",
    )

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})

    compiled_model.write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 9}),
        encoding="utf-8",
    )

    assert not report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: payload},
    )


def test_missing_only_retries_old_python_wall_timing_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = next(
        spec
        for spec in report.MATRIX_SPECS
        if spec.dataset_id == "matrix_builtin_sm_nlc"
    )
    payload = report.build_matrix_cache(spec)
    entry = payload["entries"][0]
    artifact_dir = tmp_path / "artifact"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "compiled-model.json").write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 10}),
        encoding="utf-8",
    )
    legacy = report._empty_measurement()
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "requested_config": report._legacy_profile_requested_config(20.0),
        }
    )
    pyamplicol = report._empty_measurement()
    pyamplicol.update(
        {
            "status": report.ResultStatus.OK.value,
            "artifact_path": str(artifact_dir),
            "environment": {
                "wall_time_source": "runtime_evaluate_wall_time",
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            },
        }
    )
    entry.update(
        {
            "status": report.ResultStatus.OK.value,
            "legacy_amplicol": legacy,
            "pyamplicol_jit_o3": pyamplicol,
        }
    )
    cell = report.CampaignCell(
        kind="matrix",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=str(entry["process"]),
        process_key=str(entry["process_key"]),
    )
    monkeypatch.setattr(report, "_legacy_measurement_revision_current", lambda _: True)
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 10))

    assert report._campaign_cell_needs_measurement(cell, {spec.cache_name: payload})


def test_retiming_reuses_only_current_pyamplicol_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 10))
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")
    artifact_dir = tmp_path / "artifact"
    previous = {
        "generation_seconds": 12.5,
        "artifact_path": str(artifact_dir),
        "metadata": {
            "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
            "generation_timer_excludes_model_compile": True,
            "source_provenance": _synthetic_source_provenance(9, 10),
        },
    }
    builtin_cell = report.CampaignCell(
        kind="matrix",
        cache_name="matrix_builtin_sm_nlc",
        dataset_id="matrix_builtin_sm_nlc",
        n_final=1,
        process="d d~ > z",
        process_key="dd_z_jets",
    )

    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
        )
        is None
    )

    _write_current_report_artifact(
        artifact_dir,
        producer_version="0.1.0.dev0+candidate.previous",
        compiled_model_schema=9,
    )

    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
        )
        == 12.5
    )

    _write_current_report_artifact(
        artifact_dir,
        producer_version="current",
        process_artifact_schema=2,
        compiled_model_schema=9,
    )

    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
        )
        is None
    )

    _write_current_report_artifact(
        artifact_dir,
        producer_version="current",
        compiled_model_schema=9,
    )

    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
        )
        == 12.5
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
            expected_lc_flow_layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
        )
        == 12.5
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            previous,
            expected_lc_flow_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        )
        is None
    )
    wrong_artifact = {
        **previous,
        "artifact_path": str(tmp_path / "other-artifact"),
    }
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            wrong_artifact,
        )
        is None
    )
    union_previous = {
        **previous,
        "metadata": {
            **previous["metadata"],
            "lc_flow_layout": report.LC_ALL_FLOW_UNION_LAYOUT,
        },
    }
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            union_previous,
            expected_lc_flow_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        )
        == 12.5
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            {"generation_seconds": 0.0},
        )
        is None
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            builtin_cell,
            artifact_dir,
            {"generation_seconds": 12.5},
        )
        is None
    )

    external_cell = report.CampaignCell(
        kind="matrix",
        cache_name="matrix_external_sm_nlc",
        dataset_id="matrix_external_sm_nlc",
        n_final=1,
        process="d d~ > z",
        process_key="dd_z_jets",
    )
    (artifact_dir / "model" / "compiled-model.json").write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 9}),
        encoding="utf-8",
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            external_cell,
            artifact_dir,
            previous,
        )
        is None
    )

    (artifact_dir / "model" / "compiled-model.json").write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 10}),
        encoding="utf-8",
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            external_cell,
            artifact_dir,
            previous,
        )
        == 12.5
    )


def test_runtime_only_revision_hops_allow_generation_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "head": "ff6690f892f210e401f0639aa33059f5c009574f",
        "report_version": report.REPORT_VERSION,
        "cache_schema_version": report.CACHE_SCHEMA_VERSION,
        "compiled_model_schema_version": 9,
        "model_compiler_version": 13,
    }
    monkeypatch.setattr(report, "_report_source_provenance", lambda: current)

    for previous_head in (
        "e307d218c169e246e6ce8f8e1392799c36108785",
        "0144af352a216ce8511b76b5271a5fce90d15e08",
    ):
        previous = dict(current)
        previous["head"] = previous_head
        assert report._source_provenance_generation_reusable(previous)

    stale_contract = dict(current)
    stale_contract["head"] = "e307d218c169e246e6ce8f8e1392799c36108785"
    stale_contract["model_compiler_version"] = 12
    assert not report._source_provenance_generation_reusable(stale_contract)

    eager_fix_current = dict(current)
    eager_fix_current["head"] = "e3342771aa6f56853fcd98035982f6056e68211f"
    monkeypatch.setattr(report, "_report_source_provenance", lambda: eager_fix_current)
    previous = dict(eager_fix_current)
    previous["head"] = "a0fd4a458c281b1838df10c6547395edc6e65618"
    assert report._source_provenance_generation_reusable(previous)
    stale_eager_measurement = {
        "metadata": {
            "source_provenance": previous,
        },
    }
    eager_n3_cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name="z_builtin_sm.json",
        dataset_id="z_builtin_sm",
        n_final=3,
        process="d d~ > z g g",
        variant="eager_jit_o3",
    )
    eager_n2_cell = report.CampaignCell(
        kind="performance_ladder",
        cache_name="z_builtin_sm.json",
        dataset_id="z_builtin_sm",
        n_final=2,
        process="d d~ > z g",
        variant="eager_jit_o3",
    )
    assert report._measurement_requires_runtime_refresh(
        eager_n3_cell,
        stale_eager_measurement,
        execution_mode="eager",
    )
    assert not report._measurement_requires_runtime_refresh(
        eager_n3_cell,
        stale_eager_measurement,
        execution_mode="compiled",
    )
    assert not report._measurement_requires_runtime_refresh(
        eager_n2_cell,
        stale_eager_measurement,
        execution_mode="eager",
    )


def test_lc_replay_runtime_fix_reuses_pre_fix_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "head": "post-replay-fix-head",
        "report_version": report.REPORT_VERSION,
        "cache_schema_version": report.CACHE_SCHEMA_VERSION,
        "compiled_model_schema_version": 9,
        "model_compiler_version": 13,
    }
    monkeypatch.setattr(report, "_report_source_provenance", lambda: current)
    monkeypatch.setattr(
        report,
        "_git_is_ancestor",
        lambda ancestor, descendant: (
            ancestor == report.LC_HELICITY_REPLAY_RUNTIME_FIX_REVISION
            and descendant == "post-replay-fix-head"
        ),
    )
    previous = dict(current)
    previous["head"] = "55bfedc80df4695dc7aa55bc5d40669d248d2f14"

    assert report._source_provenance_generation_reusable(previous)


def test_lc_union_source_transition_allows_only_named_prefeature_bases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "head": "post-union-head",
        "report_version": report.REPORT_VERSION,
        "cache_schema_version": report.CACHE_SCHEMA_VERSION,
        "compiled_model_schema_version": 9,
        "model_compiler_version": 13,
    }
    monkeypatch.setattr(report, "_report_source_provenance", lambda: current)
    monkeypatch.setattr(
        report,
        "_git_is_ancestor",
        lambda ancestor, descendant: (
            ancestor == report.LC_ALL_FLOW_UNION_IMPLEMENTATION_REVISION
            and descendant == "post-union-head"
        ),
    )

    for previous_head in report.LC_ALL_FLOW_UNION_REUSE_BASE_REVISIONS:
        previous = {**current, "head": previous_head}
        assert report._source_provenance_generation_reusable(previous)

    unrelated = {**current, "head": "unrelated-old-head"}
    assert not report._source_provenance_generation_reusable(unrelated)


def test_union_transition_reuses_topology_generation_but_not_preunion_union(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pre_union_revision = "c7e45b090747097965e62b919386d6ee598f94a7"
    current_revision = "post-union-head"
    current_provenance = {
        **_synthetic_source_provenance(9, 13),
        "head": current_revision,
    }
    monkeypatch.setattr(
        report,
        "_report_source_provenance",
        lambda: current_provenance,
    )
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 13))
    monkeypatch.setattr(
        report,
        "_git_is_ancestor",
        lambda ancestor, descendant: (
            ancestor == report.LC_ALL_FLOW_UNION_IMPLEMENTATION_REVISION
            and descendant == current_revision
        ),
    )
    cell = report.CampaignCell(
        kind="matrix",
        cache_name="matrix_builtin_sm_lc",
        dataset_id="matrix_builtin_sm_lc",
        n_final=3,
        process="d d~ > z g g",
        process_key="dd_z_jets",
    )

    def prior_measurement(
        artifact_dir: Path,
        *,
        layout: str,
    ) -> dict[str, object]:
        _write_current_report_artifact(artifact_dir, layout=layout)
        return {
            "generation_seconds": 4.0,
            "artifact_path": str(artifact_dir),
            "metadata": {
                "lc_flow_layout": layout,
                "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
                "generation_timer_excludes_model_compile": True,
                "source_provenance": {
                    **current_provenance,
                    "head": pre_union_revision,
                },
            },
        }

    topology_dir = tmp_path / "complete-lc"
    topology = prior_measurement(
        topology_dir,
        layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
    )
    pre_union_dir = tmp_path / "all-flow-union"
    pre_union = prior_measurement(
        pre_union_dir,
        layout=report.LC_ALL_FLOW_UNION_LAYOUT,
    )
    before = json.dumps(
        {"topology": topology, "pre_union": pre_union},
        sort_keys=True,
    )

    assert (
        report._reusable_pyamplicol_generation_seconds(
            cell,
            topology_dir,
            topology,
            expected_lc_flow_layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
        )
        == 4.0
    )
    assert (
        report._reusable_pyamplicol_generation_seconds(
            cell,
            pre_union_dir,
            pre_union,
            expected_lc_flow_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        )
        is None
    )
    assert (
        json.dumps(
            {"topology": topology, "pre_union": pre_union},
            sort_keys=True,
        )
        == before
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_union_transition_preserves_non_lc_measurements_without_mutating_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accuracy: str,
) -> None:
    spec = next(
        item
        for item in report.MATRIX_SPECS
        if item.dataset_id == f"matrix_builtin_sm_{accuracy}"
    )
    payload = report.build_matrix_cache(spec)
    entry = payload["entries"][0]
    artifact_dir = tmp_path / accuracy / "artifact"
    _write_current_report_artifact(artifact_dir)
    provenance = _synthetic_source_provenance(9, 13)
    provenance["head"] = "c7e45b090747097965e62b919386d6ee598f94a7"
    measurement = report._empty_measurement()
    measurement.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "artifact_path": str(artifact_dir),
            "environment": {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "evaluator_time_source": ("runtime_profile_core_evaluator_call_time"),
            },
            "metadata": {
                "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
                "generation_timer_excludes_model_compile": True,
                "source_provenance": provenance,
            },
        }
    )
    legacy = report._empty_measurement()
    legacy["status"] = report.ResultStatus.OK.value
    entry.update(
        {
            "status": report.ResultStatus.OK.value,
            "legacy_amplicol": legacy,
            "pyamplicol_jit_o3": measurement,
        }
    )
    cell = report.CampaignCell(
        kind="matrix",
        cache_name=spec.cache_name,
        dataset_id=spec.dataset_id,
        n_final=int(entry["n_final"]),
        process=str(entry["process"]),
        process_key=str(entry["process_key"]),
    )
    before = json.dumps(payload, sort_keys=True)
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 13))
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")
    monkeypatch.setattr(report, "_legacy_measurement_revision_current", lambda _: True)
    monkeypatch.setattr(report, "_legacy_measurement_profile_current", lambda _: True)
    monkeypatch.setattr(report, "_git_is_ancestor", lambda *_args: True)

    assert not report._campaign_cell_needs_measurement(
        cell,
        {spec.cache_name: payload},
    )
    assert json.dumps(payload, sort_keys=True) == before


def test_legacy_revision_check_accepts_non_numerical_followup_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "38937fc4a0a66ae14c55e77ba455de8c6170547b"
    measurement = {
        "environment": {
            "revision": revision,
        },
    }
    from tools.developer import legacy_amplicol

    monkeypatch.setattr(
        legacy_amplicol,
        "expected_revision",
        lambda: "754064d751224ec96c182d5f5d21fd6a11ad28f6",
    )

    assert report._legacy_measurement_revision_current(measurement)


def test_legacy_lc_contract_rejects_probe_setup_generation_source() -> None:
    measurement = report._empty_measurement()
    measurement.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 4.0,
            "requested_config": report._legacy_profile_requested_config(20.0),
            "metadata": {
                "old_matrix_format": {
                    "generation_s": 4.0,
                    "all_flow_generation_s": 0.001,
                    "all_flow_generation_source": (
                        "amplicol_color_probe_imode2_explicit_setup"
                    ),
                    "all_flow_status": report.ResultStatus.OK.value,
                }
            },
        }
    )

    assert not report._legacy_lc_measurement_contract_current(measurement)

    fields = measurement["metadata"]["old_matrix_format"]  # type: ignore[index]
    fields["all_flow_generation_s"] = 4.0  # type: ignore[index]
    fields["all_flow_generation_source"] = (  # type: ignore[index]
        report.LEGACY_LC_ALL_FLOW_GENERATION_SOURCE
    )

    assert report._legacy_lc_measurement_contract_current(measurement)


def test_legacy_lc_contract_rejects_fixed_point_profile_config() -> None:
    measurement = report._empty_measurement()
    measurement.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 4.0,
            "requested_config": {
                "method": "legacy_amplicol_generated_library",
                "library_benchmark_points": 100_000,
                "color_probe_points": 100_000,
            },
            "metadata": {
                "old_matrix_format": {
                    "generation_s": 4.0,
                    "all_flow_generation_s": 4.0,
                    "all_flow_generation_source": (
                        report.LEGACY_LC_ALL_FLOW_GENERATION_SOURCE
                    ),
                    "all_flow_status": report.ResultStatus.OK.value,
                }
            },
        }
    )

    assert not report._legacy_measurement_profile_current(measurement)
    assert not report._legacy_lc_measurement_contract_current(measurement)


def test_compiled_lc_uses_two_complete_layout_artifacts_and_runtime_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol.api as api
    import pyamplicol.config.resolver as resolver

    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    spec = report.LADDER_SPECS[0]
    legacy = report._empty_measurement()
    fixed_helicity = report._source_helicity_choice_payload(
        cell.process,
        {"1": -1, "2": 1, "3": -1, "4": 1, "5": -1},
        selection_source="test",
        validation_note="test",
    )
    legacy.update(
        {
            "status": report.ResultStatus.OK.value,
            "matrix_element": 1.0,
            "metadata": {
                "old_matrix_format": {
                    "reference_color_order": [2, 4, 5, 1, 3],
                    "all_flow_source_helicities": fixed_helicity["source_helicities"],
                }
            },
        }
    )
    resolution = SimpleNamespace(
        requested=SimpleNamespace(),
        effective=SimpleNamespace(benchmark=BenchmarkConfig(target_runtime=0.1)),
    )
    generated: list[tuple[str, Path, object, str]] = []
    profile_calls: list[dict[str, object]] = []

    class FakeGenerator:
        def __init__(self, config: object) -> None:
            self.config = config

        def generate(
            self,
            process: str,
            artifact_dir: Path,
            *,
            model: object,
            mode: str,
        ) -> None:
            generated.append((process, artifact_dir, model, mode))
            artifact_dir.mkdir(parents=True)

    class FakeRuntime:
        physics = SimpleNamespace()

    class FakeRuntimeLoader:
        @staticmethod
        def load(_artifact_dir: Path, *, process: str) -> FakeRuntime:
            assert process == cell.process
            return FakeRuntime()

    def fake_profile(
        _runtime: object,
        *,
        benchmark_config: object,
        points: object,
        helicity_ids: Sequence[str] = (),
        color_flow_ids: Sequence[str] = (),
    ) -> dict[str, object]:
        profile_calls.append(
            {
                "benchmark_config": benchmark_config,
                "points": points,
                "helicity_ids": tuple(helicity_ids),
                "color_flow_ids": tuple(color_flow_ids),
            }
        )
        measurement = report._empty_measurement()
        measurement.update(
            {
                "status": report.ResultStatus.OK.value,
                "sample_count": 2,
                "wall_seconds_per_point": 2.0e-6,
                "evaluator_seconds_per_point": 1.0e-6,
                "standard_deviation_seconds_per_point": 0.0,
                "standard_error_seconds_per_point": 0.0,
                "relative_standard_error": 0.0,
                "matrix_element": 1.0,
                "requested_config": {},
                "effective_config": {},
                "environment": {
                    "wall_time_source": "runtime_core_repeated_wall_time",
                    "evaluator_time_source": (
                        "runtime_profile_core_evaluator_call_time"
                    ),
                },
                "metadata": {
                    "helicity_ids": list(helicity_ids),
                    "color_flow_ids": list(color_flow_ids),
                },
            }
        )
        return measurement

    monkeypatch.setattr(api, "Generator", FakeGenerator)
    monkeypatch.setattr(api, "Runtime", FakeRuntimeLoader)
    monkeypatch.setattr(
        resolver,
        "resolve_config",
        lambda *_args, **_kwargs: resolution,
    )
    monkeypatch.setattr(resolver, "config_to_dict", lambda _value: {})
    monkeypatch.setattr(
        report,
        "_precompile_model_for_generation",
        lambda *_args, **_kwargs: (
            "compiled-model",
            {
                "model_precompile_policy": (
                    report.PYAMPLICOL_GENERATION_PROFILE_POLICY
                ),
                "model_precompile_seconds": 0.0,
                "model_precompile_cache_dir": None,
                "model_precompile_used_cache": True,
                "model_precompile_source_kind": "built-in-sm",
                "generation_timer_excludes_model_compile": True,
            },
        ),
    )
    monkeypatch.setattr(
        report,
        "_single_artifact_process_id",
        lambda _artifact_dir, *, fallback: fallback,
    )
    monkeypatch.setattr(report, "_profile_eager_runtime", fake_profile)
    monkeypatch.setattr(
        report,
        "_lc_cross_artifact_validation",
        lambda *_args, **_kwargs: {"status": report.ResultStatus.OK.value},
    )
    monkeypatch.setattr(
        report,
        "_lc_runtime_selector_contract",
        lambda **_kwargs: {
            **report._empty_eager_selector_contract(),
            "status": report.ResultStatus.OK.value,
            "reference_digest": report._eager_reference_digest(cell, legacy),
            "selected_reference_color_order": [2, 4, 5, 1, 3],
            "selected_color_flow_ids": ["flow:2,4,5,1"],
            "all_flow_source_helicities": fixed_helicity["source_helicities"],
            "all_flow_helicity_ids": ["h:-1,+1,-1,+1,-1"],
        },
    )
    monkeypatch.setattr(
        report,
        "_generation_slice_tools",
        lambda: pytest.fail("compiled LC report path must not use GenerationSlice"),
    )
    monkeypatch.setattr(
        report,
        "_report_source_provenance",
        lambda: {
            "head": "test",
            "report_version": report.REPORT_VERSION,
            "cache_schema_version": report.CACHE_SCHEMA_VERSION,
            "compiled_model_schema_version": 9,
            "model_compiler_version": 13,
        },
    )
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 13))

    measurement, returned_points = report._measure_pyamplicol_lc_two_workloads(
        cell=cell,
        spec=spec,
        variant_overrides={
            "evaluator.backend": "jit",
            "evaluator.jit.optimization_level": 3,
        },
        legacy=legacy,
        artifact_root=tmp_path,
        generation_timeout_seconds=60.0,
        target_runtime=0.1,
        cell_cores=1,
        points=("point",),
        fixed_helicity=fixed_helicity,
    )

    assert returned_points == ("point",)
    assert len(generated) == 2
    assert generated[0][1].parts[-1] == "complete-lc"
    assert generated[1][1].parts[-1] == "all-flow-union"
    assert generated[0][3] == "replace"
    assert generated[1][3] == "replace"
    assert profile_calls[0]["color_flow_ids"] == ("flow:2,4,5,1",)
    assert profile_calls[0]["helicity_ids"] == ()
    assert profile_calls[1]["color_flow_ids"] == ()
    assert profile_calls[1]["helicity_ids"] == ("h:-1,+1,-1,+1,-1",)
    metadata = measurement["metadata"]
    assert metadata["generation_slice"] is None
    assert metadata["runtime_selector_policy"] == "complete_lc_runtime_selectors_v2"
    old = metadata["old_matrix_format"]
    assert old["selected_output_dir"] != old["all_flow_output_dir"]
    assert old["selected_output_dir"].endswith("complete-lc")
    assert old["all_flow_output_dir"].endswith("all-flow-union")
    assert old["selected_color_flow_ids"] == ["flow:2,4,5,1"]
    assert old["all_flow_helicity_ids"] == ["h:-1,+1,-1,+1,-1"]
    assert (
        metadata["selected_flow_measurement"]["metadata"]["lc_flow_layout"]
        == report.LC_TOPOLOGY_REPLAY_LAYOUT
    )
    assert (
        metadata["all_flow_measurement"]["metadata"]["lc_flow_layout"]
        == report.LC_ALL_FLOW_UNION_LAYOUT
    )
    point_digest = report._measurement_point_digest(("point",))
    assert (
        metadata["selected_flow_measurement"]["metadata"]["measurement_point_digest"]
        == point_digest
    )
    assert (
        metadata["all_flow_measurement"]["metadata"]["measurement_point_digest"]
        == point_digest
    )
    snapshot = (
        tmp_path
        / "cells"
        / cell.cell_id
        / "inputs"
        / "pyamplicol-complete-lc-inputs.json"
    )
    snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert snapshot_payload["generation_slice"] is None
    assert snapshot_payload["measurement_point_digest"] == point_digest
    assert (
        snapshot_payload["measurement_point_source"] == "caller-supplied-report-point"
    )
    all_flow_snapshot = (
        tmp_path
        / "cells"
        / cell.cell_id
        / "inputs"
        / "pyamplicol-all-flow-union-inputs.json"
    )
    all_flow_snapshot_payload = json.loads(
        all_flow_snapshot.read_text(encoding="utf-8")
    )
    assert (
        all_flow_snapshot_payload["lc_flow_layout"] == report.LC_ALL_FLOW_UNION_LAYOUT
    )
    assert all_flow_snapshot_payload["measurement_point_digest"] == point_digest


def test_compiled_lc_refreshes_only_stale_all_flow_union(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    spec = report.LADDER_SPECS[0]
    fixed_helicity = report._source_helicity_choice_payload(
        cell.process,
        {"1": -1, "2": 1, "3": -1, "4": 1, "5": -1},
        selection_source="test",
        validation_note="test",
    )
    legacy = {"status": report.ResultStatus.OK.value}
    contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": report._eager_reference_digest(cell, legacy),
        "selected_reference_color_order": [2, 4, 5, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,5,1"],
        "all_flow_source_helicities": fixed_helicity["source_helicities"],
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1,-1"],
    }
    selected = {
        **report._empty_measurement(),
        "status": report.ResultStatus.OK.value,
        "generation_seconds": 2.0,
        "wall_seconds_per_point": 2.0e-6,
        "evaluator_seconds_per_point": 1.0e-6,
        "artifact_path": "/preserved/complete-lc",
        "metadata": {
            "lc_flow_layout": report.LC_TOPOLOGY_REPLAY_LAYOUT,
            "selector_contract": contract,
        },
    }
    stale_all_flow = {
        **selected,
        "generation_seconds": 3.0,
        "artifact_path": "/preserved/old-complete-lc",
    }
    previous = {
        **selected,
        "metadata": {
            "selected_flow_measurement": selected,
            "all_flow_measurement": stale_all_flow,
            "selector_contract": contract,
        },
    }
    lane_calls: list[dict[str, object]] = []

    def nested_current(
        _cell: report.CampaignCell,
        measurement: object,
        *,
        expected_layout: str,
        execution_mode: str,
    ) -> bool:
        assert execution_mode == "compiled"
        return (
            measurement is selected
            and expected_layout == report.LC_TOPOLOGY_REPLAY_LAYOUT
        )

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        lane_calls.append(dict(kwargs))
        result = {
            **report._empty_measurement(),
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 7.0,
            "wall_seconds_per_point": 4.0e-6,
            "evaluator_seconds_per_point": 3.0e-6,
            "matrix_element": 1.0,
            "artifact_path": "/fresh/all-flow-union",
            "metadata": {
                "lc_flow_layout": report.LC_ALL_FLOW_UNION_LAYOUT,
                "selector_contract": contract,
            },
        }
        return result, ("point",), contract

    monkeypatch.setattr(report, "_lc_nested_measurement_current", nested_current)
    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)
    monkeypatch.setattr(
        report,
        "_load_lc_runtime_for_cross_validation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "_lc_cross_artifact_validation",
        lambda *_args, **_kwargs: {"status": report.ResultStatus.OK.value},
    )

    measurement, returned_points = report._measure_pyamplicol_lc_two_workloads(
        cell=cell,
        spec=spec,
        variant_overrides={"evaluator.backend": "jit"},
        legacy=legacy,
        artifact_root=tmp_path,
        generation_timeout_seconds=60.0,
        target_runtime=0.1,
        cell_cores=1,
        points=("point",),
        fixed_helicity=fixed_helicity,
        previous_measurement=previous,
    )

    assert returned_points == ("point",)
    assert len(lane_calls) == 1
    assert lane_calls[0]["layout"] == report.LC_ALL_FLOW_UNION_LAYOUT
    assert lane_calls[0]["artifact_label"] == "all-flow-union"
    metadata = measurement["metadata"]
    assert metadata["selected_flow_measurement"] == selected
    assert metadata["old_matrix_format"]["selected_generation_s"] == 2.0
    assert metadata["old_matrix_format"]["all_flow_generation_s"] == 7.0


def test_compiled_lc_measures_missing_selected_lane_and_skips_union_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    spec = report.LADDER_SPECS[0]
    fixed_helicity = report._source_helicity_choice_payload(
        cell.process,
        {"1": -1, "2": 1, "3": -1, "4": 1, "5": -1},
        selection_source="test",
        validation_note="test",
    )
    legacy = {"status": report.ResultStatus.OK.value}
    contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": report._eager_reference_digest(cell, legacy),
        "selected_reference_color_order": [2, 4, 5, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,5,1"],
        "all_flow_source_helicities": fixed_helicity["source_helicities"],
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1,-1"],
    }
    selected_missing = report._empty_measurement()
    selected_missing["metadata"] = {
        "lane_status_policy": report.LC_LANE_STATUS_POLICY,
        "runtime_selector_role": "selected-flow-helicity-sum",
        "lc_flow_layout": report.LC_TOPOLOGY_REPLAY_LAYOUT,
    }
    union_terminal = report._failure_measurement(
        report.ResultStatus.OUT_OF_REACH,
        "union lane is too large",
        metadata={
            "lane_status_policy": report.LC_LANE_STATUS_POLICY,
            "runtime_selector_role": "all-flows-fixed-helicity",
            "lc_flow_layout": report.LC_ALL_FLOW_UNION_LAYOUT,
        },
    )
    previous = {
        **selected_missing,
        "metadata": {
            "lane_status_policy": report.LC_LANE_STATUS_POLICY,
            "selected_flow_measurement": selected_missing,
            "all_flow_measurement": union_terminal,
        },
    }
    lane_calls: list[str] = []

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        layout = str(kwargs["layout"])
        lane_calls.append(layout)
        assert layout == report.LC_TOPOLOGY_REPLAY_LAYOUT
        return (
            {
                **report._empty_measurement(),
                "status": report.ResultStatus.OK.value,
                "generation_seconds": 2.0,
                "wall_seconds_per_point": 2.0e-6,
                "evaluator_seconds_per_point": 1.0e-6,
                "matrix_element": 1.0,
                "artifact_path": "/artifact/complete-lc",
                "metadata": {
                    "lc_flow_layout": report.LC_TOPOLOGY_REPLAY_LAYOUT,
                    "selector_contract": contract,
                },
            },
            ("point",),
            contract,
        )

    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)

    measurement, returned_points = report._measure_pyamplicol_lc_two_workloads(
        cell=cell,
        spec=spec,
        variant_overrides={"evaluator.backend": "jit"},
        legacy=legacy,
        artifact_root=tmp_path,
        generation_timeout_seconds=60.0,
        target_runtime=0.1,
        cell_cores=1,
        points=("point",),
        fixed_helicity=fixed_helicity,
        previous_measurement=previous,
    )

    assert lane_calls == [report.LC_TOPOLOGY_REPLAY_LAYOUT]
    assert returned_points == ("point",)
    assert measurement["status"] == report.ResultStatus.OK.value
    metadata = measurement["metadata"]
    assert (
        metadata["all_flow_measurement"]["status"]
        == report.ResultStatus.OUT_OF_REACH.value
    )
    assert (
        metadata["old_matrix_format"]["all_flow_status"]
        == report.ResultStatus.OUT_OF_REACH.value
    )


@pytest.mark.parametrize(
    "stale_layout",
    (report.LC_TOPOLOGY_REPLAY_LAYOUT, report.LC_ALL_FLOW_UNION_LAYOUT),
)
def test_compiled_lc_reuse_rejects_stale_reference_digest_for_each_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stale_layout: str,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    spec = report.LADDER_SPECS[0]
    legacy = {
        "status": report.ResultStatus.OK.value,
        "matrix_element": 3.0,
    }
    current_digest = report._eager_reference_digest(cell, legacy)
    fixed_helicity = report._source_helicity_choice_payload(
        cell.process,
        {"1": -1, "2": 1, "3": -1, "4": 1, "5": -1},
        selection_source="test",
        validation_note="test",
    )
    current_contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": current_digest,
        "selected_reference_color_order": [2, 4, 5, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,5,1"],
        "all_flow_source_helicities": fixed_helicity["source_helicities"],
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1,-1"],
    }
    stale_contract = {**current_contract, "reference_digest": "stale-reference"}

    def lane(layout: str, contract: Mapping[str, object]) -> dict[str, object]:
        artifact_label = (
            "complete-lc"
            if layout == report.LC_TOPOLOGY_REPLAY_LAYOUT
            else "all-flow-union"
        )
        return {
            **report._empty_measurement(),
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 2.0,
            "wall_seconds_per_point": 2.0e-6,
            "evaluator_seconds_per_point": 1.0e-6,
            "matrix_element": 1.0,
            "artifact_path": f"/artifact/{artifact_label}",
            "metadata": {
                "lc_flow_layout": layout,
                "selector_contract": dict(contract),
            },
        }

    selected_contract = (
        stale_contract
        if stale_layout == report.LC_TOPOLOGY_REPLAY_LAYOUT
        else current_contract
    )
    all_flow_contract = (
        stale_contract
        if stale_layout == report.LC_ALL_FLOW_UNION_LAYOUT
        else current_contract
    )
    previous_selected = lane(report.LC_TOPOLOGY_REPLAY_LAYOUT, selected_contract)
    previous_all_flow = lane(report.LC_ALL_FLOW_UNION_LAYOUT, all_flow_contract)
    previous = {
        **previous_selected,
        "metadata": {
            "selected_flow_measurement": previous_selected,
            "all_flow_measurement": previous_all_flow,
            "selector_contract": dict(selected_contract),
        },
    }
    calls: list[str] = []

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        layout = str(kwargs["layout"])
        calls.append(layout)
        return lane(layout, current_contract), ("point",), current_contract

    monkeypatch.setattr(
        report,
        "_lc_nested_measurement_current",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)
    monkeypatch.setattr(
        report,
        "_load_lc_runtime_for_cross_validation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "_lc_cross_artifact_validation",
        lambda *_args, **_kwargs: {"status": report.ResultStatus.OK.value},
    )

    measurement, _points = report._measure_pyamplicol_lc_two_workloads(
        cell=cell,
        spec=spec,
        variant_overrides={"evaluator.backend": "jit"},
        legacy=legacy,
        artifact_root=tmp_path,
        generation_timeout_seconds=60.0,
        target_runtime=0.1,
        cell_cores=1,
        points=("point",),
        fixed_helicity=fixed_helicity,
        previous_measurement=previous,
    )

    assert calls == [stale_layout]
    metadata = measurement["metadata"]
    selected = metadata["selected_flow_measurement"]
    all_flow = metadata["all_flow_measurement"]
    assert report._cached_lc_selector_contract(selected) == current_contract
    assert report._cached_lc_selector_contract(all_flow) == current_contract


def test_lc_combined_freshness_requires_matching_current_selector_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    reference = {"status": report.ResultStatus.OK.value, "matrix_element": 3.0}
    digest = report._eager_reference_digest(cell, reference)
    contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": digest,
    }

    def lane(selector_contract: Mapping[str, object]) -> dict[str, object]:
        return {"metadata": {"selector_contract": dict(selector_contract)}}

    monkeypatch.setattr(
        report,
        "_lc_nested_measurement_current",
        lambda *_args, **_kwargs: True,
    )
    selected = lane(contract)
    mismatched = lane({**contract, "all_flow_helicity_ids": ["h:other"]})
    combined = {
        "metadata": {
            "selected_flow_measurement": selected,
            "all_flow_measurement": mismatched,
            "selector_contract": dict(contract),
        }
    }

    assert not report._lc_combined_measurement_current(
        cell,
        combined,
        execution_mode="compiled",
        reference_measurement=reference,
    )

    combined["metadata"]["all_flow_measurement"] = lane(contract)
    assert report._lc_combined_measurement_current(
        cell,
        combined,
        execution_mode="compiled",
        reference_measurement=reference,
    )

    combined["metadata"]["selector_contract"] = {
        **contract,
        "all_flow_helicity_ids": ["h:other"],
    }
    assert not report._lc_combined_measurement_current(
        cell,
        combined,
        execution_mode="compiled",
        reference_measurement=reference,
    )

    combined["metadata"]["selector_contract"] = dict(contract)
    changed_reference = {**reference, "matrix_element": 4.0}
    assert not report._lc_combined_measurement_current(
        cell,
        combined,
        execution_mode="compiled",
        reference_measurement=changed_reference,
    )


def test_lc_nested_freshness_invalidates_only_old_all_flow_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.cell_id == "z-builtin-sm-n3-jit-o3"
    )
    pre_union_revision = "c7e45b090747097965e62b919386d6ee598f94a7"
    current_revision = report.LC_ALL_FLOW_UNION_IMPLEMENTATION_REVISION
    monkeypatch.setattr(report, "_current_pyamplicol_version", lambda: "current")
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 13))
    monkeypatch.setattr(
        report,
        "_git_is_ancestor",
        lambda ancestor, descendant: (
            ancestor == report.LC_ALL_FLOW_UNION_IMPLEMENTATION_REVISION
            and descendant == current_revision
        ),
    )

    def measurement(
        artifact_dir: Path,
        *,
        layout: str,
        source_revision: str,
    ) -> dict[str, object]:
        _write_current_report_artifact(artifact_dir, layout=layout)
        return {
            **report._empty_measurement(),
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "artifact_path": str(artifact_dir),
            "effective_config": {
                "color": {"lc_flow_layout": layout},
                "evaluator": {"execution_mode": "compiled"},
            },
            "environment": {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "evaluator_time_source": ("runtime_profile_core_evaluator_call_time"),
            },
            "metadata": {
                "lc_flow_layout": layout,
                "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
                "generation_timer_excludes_model_compile": True,
                "source_provenance": {
                    **_synthetic_source_provenance(9, 13),
                    "head": source_revision,
                },
            },
        }

    topology = measurement(
        tmp_path / "complete-lc",
        layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
        source_revision=pre_union_revision,
    )
    pre_union = measurement(
        tmp_path / "pre-union" / "all-flow-union",
        layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        source_revision=pre_union_revision,
    )
    union = measurement(
        tmp_path / "current" / "all-flow-union",
        layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        source_revision=current_revision,
    )
    sliced = {
        **topology,
        "metadata": {
            **topology["metadata"],
            "generation_slice": {"selected_color_sector_ids": [0]},
        },
    }
    before = json.dumps(
        {"topology": topology, "pre_union": pre_union, "union": union},
        sort_keys=True,
    )

    assert report._lc_nested_measurement_current(
        cell,
        topology,
        expected_layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
        execution_mode="compiled",
    )
    assert not report._lc_nested_measurement_current(
        cell,
        topology,
        expected_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        execution_mode="compiled",
    )
    assert not report._lc_nested_measurement_current(
        cell,
        pre_union,
        expected_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        execution_mode="compiled",
    )
    assert report._lc_nested_measurement_current(
        cell,
        union,
        expected_layout=report.LC_ALL_FLOW_UNION_LAYOUT,
        execution_mode="compiled",
    )
    assert not report._lc_nested_measurement_current(
        cell,
        sliced,
        expected_layout=report.LC_TOPOLOGY_REPLAY_LAYOUT,
        execution_mode="compiled",
    )
    assert (
        json.dumps(
            {"topology": topology, "pre_union": pre_union, "union": union},
            sort_keys=True,
        )
        == before
    )


def test_eager_previous_cache_lookup_matches_process_and_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item for item in report._campaign_cells() if item.kind == "eager_matrix"
    )
    wrong = {
        "process_key": "wrong-process",
        "n_final": cell.n_final,
        "eager_jit_o3": {"marker": "wrong"},
    }
    expected = {
        "process_key": cell.process_key,
        "n_final": cell.n_final,
        "eager_jit_o3": {"marker": "expected"},
    }
    monkeypatch.setattr(
        report,
        "load_caches",
        lambda _paths: {cell.cache_name: {"entries": [wrong, expected]}},
    )

    assert report._previous_cache_entry_for_cell(cell) == expected


def test_eager_lc_uses_topology_replay_and_all_flow_union_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.kind == "eager_matrix" and item.dataset_id.endswith("_lc")
    )
    spec = report._spec_by_dataset()[cell.dataset_id]
    assert isinstance(spec, report.EagerMatrixSpec)
    contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "selected_reference_color_order": [2, 4, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1, "4": 1},
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1"],
    }
    calls: list[dict[str, object]] = []

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        calls.append(dict(kwargs))
        layout = str(kwargs["layout"])
        generation = 2.0 if layout == report.LC_TOPOLOGY_REPLAY_LAYOUT else 7.0
        measurement = {
            **report._empty_measurement(),
            "status": report.ResultStatus.OK.value,
            "generation_seconds": generation,
            "wall_seconds_per_point": 2.0e-6,
            "evaluator_seconds_per_point": 1.0e-6,
            "matrix_element": 1.0,
            "artifact_path": f"/artifact/{kwargs['artifact_label']}",
            "metadata": {"lc_flow_layout": layout, "selector_contract": contract},
        }
        return measurement, ("point",), contract

    monkeypatch.setattr(
        report,
        "_prepared_model_source_for_eager",
        lambda *_args: (tmp_path / "prepared.pyamplicol-model", {"kind": "test"}),
    )
    monkeypatch.setattr(
        report,
        "_lc_nested_measurement_current",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)
    monkeypatch.setattr(
        report,
        "_load_lc_runtime_for_cross_validation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "_lc_cross_artifact_validation",
        lambda *_args, **_kwargs: {"status": report.ResultStatus.OK.value},
    )
    reference_measurement = {"status": report.ResultStatus.OK.value}

    measurement, returned_points, returned_contract = (
        report._measure_pyamplicol_eager_lc_two_workloads(
            cell=cell,
            spec=spec,
            reference_measurement=reference_measurement,
            artifact_root=tmp_path,
            generation_timeout_seconds=60.0,
            target_runtime=0.1,
            cell_cores=1,
            points=("point",),
            previous_measurement=None,
        )
    )

    assert returned_points == ("point",)
    assert returned_contract == {
        **contract,
        "reference_digest": report._eager_reference_digest(
            cell,
            reference_measurement,
        ),
    }
    assert [call["artifact_label"] for call in calls] == [
        "eager-complete",
        "eager-all-flow-union",
    ]
    assert [call["layout"] for call in calls] == [
        report.LC_TOPOLOGY_REPLAY_LAYOUT,
        report.LC_ALL_FLOW_UNION_LAYOUT,
    ]
    old = measurement["metadata"]["old_matrix_format"]
    assert old["selected_generation_s"] == 2.0
    assert old["all_flow_generation_s"] == 7.0


def test_eager_lc_partial_reuse_rejects_stale_compiled_reference_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.kind == "eager_matrix" and item.dataset_id.endswith("_lc")
    )
    spec = report._spec_by_dataset()[cell.dataset_id]
    assert isinstance(spec, report.EagerMatrixSpec)
    current_contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": "current-reference",
        "selected_reference_color_order": [2, 4, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1, "4": 1},
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1"],
    }
    stale_contract = {**current_contract, "reference_digest": "stale-reference"}

    def lane(layout: str, contract: dict[str, object]) -> dict[str, object]:
        return {
            **report._empty_measurement(),
            "status": report.ResultStatus.OK.value,
            "generation_seconds": 2.0,
            "wall_seconds_per_point": 2.0e-6,
            "evaluator_seconds_per_point": 1.0e-6,
            "matrix_element": 1.0,
            "artifact_path": f"/artifact/{layout}",
            "metadata": {"lc_flow_layout": layout, "selector_contract": contract},
        }

    previous_selected = lane(report.LC_TOPOLOGY_REPLAY_LAYOUT, stale_contract)
    previous_all_flow = lane(report.LC_ALL_FLOW_UNION_LAYOUT, current_contract)
    previous = {
        **previous_selected,
        "metadata": {
            "selected_flow_measurement": previous_selected,
            "all_flow_measurement": previous_all_flow,
            "selector_contract": stale_contract,
        },
    }
    calls: list[str] = []

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        layout = str(kwargs["layout"])
        calls.append(layout)
        return lane(layout, current_contract), ("point",), current_contract

    monkeypatch.setattr(
        report, "_eager_reference_digest", lambda *_args: "current-reference"
    )
    monkeypatch.setattr(
        report,
        "_prepared_model_source_for_eager",
        lambda *_args: (tmp_path / "prepared.pyamplicol-model", {"kind": "test"}),
    )
    monkeypatch.setattr(
        report,
        "_lc_nested_measurement_current",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)
    monkeypatch.setattr(
        report,
        "_load_lc_runtime_for_cross_validation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "_lc_cross_artifact_validation",
        lambda *_args, **_kwargs: {"status": report.ResultStatus.OK.value},
    )

    measurement, _points, contract = report._measure_pyamplicol_eager_lc_two_workloads(
        cell=cell,
        spec=spec,
        reference_measurement={"status": report.ResultStatus.OK.value},
        artifact_root=tmp_path,
        generation_timeout_seconds=60.0,
        target_runtime=0.1,
        cell_cores=1,
        points=("point",),
        previous_measurement=previous,
    )

    assert calls == [report.LC_TOPOLOGY_REPLAY_LAYOUT]
    assert contract["reference_digest"] == "current-reference"
    metadata = measurement["metadata"]
    assert metadata["selector_contract"]["reference_digest"] == "current-reference"
    assert (
        metadata["selected_flow_measurement"]["metadata"]["selector_contract"][
            "reference_digest"
        ]
        == "current-reference"
    )
    assert metadata["all_flow_measurement"] == previous_all_flow


def test_eager_lc_measures_missing_selected_lane_and_skips_union_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = next(
        item
        for item in report._campaign_cells()
        if item.kind == "eager_matrix" and item.dataset_id.endswith("_lc")
    )
    spec = report._spec_by_dataset()[cell.dataset_id]
    assert isinstance(spec, report.EagerMatrixSpec)
    contract = {
        **report._empty_eager_selector_contract(),
        "status": report.ResultStatus.OK.value,
        "reference_digest": "current-reference",
        "selected_reference_color_order": [2, 4, 1, 3],
        "selected_color_flow_ids": ["flow:2,4,1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1, "4": 1},
        "all_flow_helicity_ids": ["h:-1,+1,-1,+1"],
    }
    selected_missing = report._empty_measurement()
    selected_missing["metadata"] = {
        "lane_status_policy": report.LC_LANE_STATUS_POLICY,
        "runtime_selector_role": "selected-flow-helicity-sum",
        "lc_flow_layout": report.LC_TOPOLOGY_REPLAY_LAYOUT,
    }
    union_terminal = report._failure_measurement(
        report.ResultStatus.OUT_OF_REACH,
        "union lane is too large",
        metadata={
            "lane_status_policy": report.LC_LANE_STATUS_POLICY,
            "runtime_selector_role": "all-flows-fixed-helicity",
            "lc_flow_layout": report.LC_ALL_FLOW_UNION_LAYOUT,
        },
    )
    previous = {
        **selected_missing,
        "metadata": {
            "lane_status_policy": report.LC_LANE_STATUS_POLICY,
            "selected_flow_measurement": selected_missing,
            "all_flow_measurement": union_terminal,
        },
    }
    calls: list[str] = []

    def measure_lane(
        **kwargs: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        layout = str(kwargs["layout"])
        calls.append(layout)
        assert layout == report.LC_TOPOLOGY_REPLAY_LAYOUT
        return (
            {
                **report._empty_measurement(),
                "status": report.ResultStatus.OK.value,
                "generation_seconds": 2.0,
                "wall_seconds_per_point": 2.0e-6,
                "evaluator_seconds_per_point": 1.0e-6,
                "matrix_element": 1.0,
                "artifact_path": "/artifact/eager-complete",
                "metadata": {
                    "lc_flow_layout": report.LC_TOPOLOGY_REPLAY_LAYOUT,
                    "selector_contract": contract,
                },
            },
            ("point",),
            contract,
        )

    monkeypatch.setattr(
        report, "_eager_reference_digest", lambda *_args: "current-reference"
    )
    monkeypatch.setattr(
        report,
        "_prepared_model_source_for_eager",
        lambda *_args: (tmp_path / "prepared.pyamplicol-model", {"kind": "test"}),
    )
    monkeypatch.setattr(report, "_measure_pyamplicol_lc_lane", measure_lane)

    measurement, _points, returned_contract = (
        report._measure_pyamplicol_eager_lc_two_workloads(
            cell=cell,
            spec=spec,
            reference_measurement={"status": report.ResultStatus.OK.value},
            artifact_root=tmp_path,
            generation_timeout_seconds=60.0,
            target_runtime=0.1,
            cell_cores=1,
            points=("point",),
            previous_measurement=previous,
        )
    )

    assert calls == [report.LC_TOPOLOGY_REPLAY_LAYOUT]
    assert returned_contract["reference_digest"] == "current-reference"
    assert measurement["status"] == report.ResultStatus.OK.value
    assert (
        measurement["metadata"]["old_matrix_format"]["all_flow_status"]
        == report.ResultStatus.OUT_OF_REACH.value
    )


def test_lc_cross_artifact_validation_matches_components_by_id() -> None:
    selected = SimpleNamespace(
        helicity_ids=("h:minus", "h:plus"),
        color_ids=("flow:a", "flow:b"),
        values=[[[1.0, 2.0], [3.0, 4.0]]],
    )
    reordered = SimpleNamespace(
        helicity_ids=("h:plus", "h:minus"),
        color_ids=("flow:b", "flow:a"),
        values=[[[4.0, 3.0], [2.0, 1.0]]],
    )
    selected_runtime = SimpleNamespace(
        evaluate_resolved=lambda *_args, **_kwargs: selected
    )
    union_runtime = SimpleNamespace(
        evaluate_resolved=lambda *_args, **_kwargs: reordered
    )

    validation = report._lc_cross_artifact_validation(
        selected_runtime,
        union_runtime,
        ("point",),
        {
            "selected_color_flow_ids": ["flow:a"],
            "all_flow_helicity_ids": ["h:minus"],
        },
    )

    assert validation["status"] == report.ResultStatus.OK.value
    assert validation["maximum_absolute_difference"] == 0.0
    assert validation["measurement_point_digest"] == report._measurement_point_digest(
        ("point",)
    )


def test_lc_cross_artifact_validation_rejects_missing_and_extra_ids() -> None:
    selected = SimpleNamespace(
        helicity_ids=("h:minus",),
        color_ids=("flow:a", "flow:b"),
        values=[[[1.0, 2.0]]],
    )
    mismatched = SimpleNamespace(
        helicity_ids=("h:minus",),
        color_ids=("flow:a", "flow:c"),
        values=[[[1.0, 2.0]]],
    )
    selected_runtime = SimpleNamespace(
        evaluate_resolved=lambda *_args, **_kwargs: selected
    )
    union_runtime = SimpleNamespace(
        evaluate_resolved=lambda *_args, **_kwargs: mismatched
    )

    validation = report._lc_cross_artifact_validation(
        selected_runtime,
        union_runtime,
        ("point",),
        {
            "selected_color_flow_ids": ["flow:a"],
            "all_flow_helicity_ids": ["h:minus"],
        },
    )

    assert validation["status"] == report.ResultStatus.ERROR.value
    assert validation["measurement_point_digest"] == report._measurement_point_digest(
        ("point",)
    )
    assert "missing=[('h:minus', 'flow:b')]" in validation["message"]
    assert "extra=[('h:minus', 'flow:c')]" in validation["message"]


def test_measurement_point_digest_tracks_exact_canonical_point() -> None:
    point = (((500.0, 0.0, 0.0, 500.0), (500.0, 0.0, 0.0, -500.0)),)

    assert report._measurement_point_digest(point) == report._measurement_point_digest(
        json.loads(json.dumps(point))
    )
    assert report._measurement_point_digest(point) != report._measurement_point_digest(
        (((500.0, 0.0, 0.0, 499.0), (500.0, 0.0, 0.0, -500.0)),)
    )


def test_reusable_legacy_lc_measurement_preserves_current_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement = {"status": report.ResultStatus.OK.value, "marker": "legacy"}
    monkeypatch.setattr(report, "_legacy_measurement_revision_current", lambda _v: True)
    monkeypatch.setattr(report, "_legacy_measurement_profile_current", lambda _v: True)
    monkeypatch.setattr(
        report,
        "_legacy_lc_measurement_contract_current",
        lambda _v: True,
    )

    assert report._reusable_legacy_lc_measurement(measurement) == measurement


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

    assert r"\begin{longtable}{@{}r L{0.92in} L{0.48in} L{0.86in} R{0.66in}" in builtin
    assert r"@{\hspace{0.025in}}p{0.22in}@{\hspace{0.025in}}" in builtin
    assert r"R{0.62in} R{0.52in} R{0.62in}" in ufo
    assert r"\textbf{selected flow, helicity sum}" in builtin
    assert r"\textbf{all flows, fixed helicity}" in builtin
    assert r"\PAC\ JIT \(\mathrm{O}1\)" in builtin
    assert r"\PAC\ C++ \(\mathrm{O}3\)" in builtin
    assert "Staged/Rusticol performance summary" not in builtin

    assert "UFO-SM Dedicated" in ufo
    assert r"\textbf{vs blt-in}" in ufo
    assert "External-SM" not in ufo


def test_z_tables_are_table_only_fragments() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    rendered = report.render_performance_ladder(
        report.LADDER_SPECS[0],
        caches["z_builtin_sm.json"],
    )
    assert r"\begin{landscape}" not in rendered
    assert r"\end{landscape}" not in rendered
    assert r"\begin{lstlisting}" not in rendered
    assert "Reproduce the" not in rendered
    assert r"\PAC\ model source" not in rendered


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
    out_of_reach = report._failure_measurement(
        report.ResultStatus.OUT_OF_REACH,
        "held by campaign policy",
    )
    assert report._measurement_cell(out_of_reach) == r"\ReportStatus{out-of-reach}"
    assert (
        report._matrix_failure_label(out_of_reach)
        == r"\textcolor{ReportMuted}{\texttt{out-of-reach}}"
    )
    assert report._z_old_status(out_of_reach["status"]) == "out_of_reach"
    assert (
        report._z_old_status_cell("out_of_reach")
        == r"\textcolor{ReportMuted}{\texttt{out-of-reach}}"
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


def test_campaign_selects_exact_cell_id_and_process_expression() -> None:
    target = next(
        cell
        for cell in report._campaign_cells()
        if cell.dataset_id == "matrix_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 4
    )

    assert report._select_cells(cell_ids={target.cell_id}) == (target,)

    selected = report._select_cells(
        datasets={target.dataset_id},
        processes={"  d   d~   >   z   g   g   g  "},
        n_values={target.n_final},
    )

    assert selected == (target,)


def test_campaign_rejects_unknown_exact_filters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as cell_error:
        report.main(["populate", "--dry-run", "--cell-id", "missing-cell"])
    assert cell_error.value.code == 2
    assert "unknown --cell-id" in capsys.readouterr().err

    with pytest.raises(SystemExit) as process_error:
        report.main(["populate", "--dry-run", "--process", "x x > y"])
    assert process_error.value.code == 2
    assert "unknown --process" in capsys.readouterr().err


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
    z_jit_o1 = next(
        cell
        for cell in cells
        if cell.dataset_id == "z_builtin_sm" and cell.variant == "jit_o1"
    )

    assert (
        report._campaign_worker_timeout_seconds(
            matrix_lc,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=3600,
        )
        == 177_300
    )
    assert (
        report._campaign_worker_timeout_seconds(
            matrix_nlc,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=3600,
        )
        == 90_900
    )
    assert (
        report._campaign_worker_timeout_seconds(
            z_reference,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=3600,
        )
        == 4_500
    )
    assert (
        report._campaign_worker_timeout_seconds(
            z_jit,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=3600,
        )
        == 173_700
    )
    assert (
        report._campaign_worker_timeout_seconds(
            z_jit_o1,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=3600,
        )
        == 173_700
    )
    assert (
        report._campaign_worker_timeout_seconds(
            matrix_lc,
            generation_timeout_seconds=3600,
            jit_o3_generation_timeout_seconds=86_400,
            reference_timeout_seconds=0,
        )
        is None
    )


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
