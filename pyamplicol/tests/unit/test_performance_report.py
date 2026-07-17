# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import re
import signal
import sys
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
            entry
            for entry in sys.path
            if entry not in {"", root, resolved_root}
        ]
        GenerationSlice, generate_slice = report._generation_slice_tools()
        assert root in sys.path or resolved_root in sys.path
        assert GenerationSlice.__name__ == "GenerationSlice"
        assert callable(generate_slice)
    finally:
        sys.path[:] = original


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


def test_lc_matrix_reference_runtime_pair_uses_tight_spacing() -> None:
    macros = "\n".join(report._matrix_table_macros())

    assert r"\matrixslot{0.86in}{#4}" in macros
    assert r"\makebox[0.27in][l]{#1}" in macros
    assert (
        r"\hspace{0.006in}\matrixpunct{/}\hspace{0.012in}"
        in macros
    )


def test_report_source_provenance_records_git_and_compiler_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report._report_source_provenance.cache_clear()
    monkeypatch.setattr(report, "_current_compiled_model_contract", lambda: (9, 11))
    try:
        provenance = report._report_source_provenance()
    finally:
        report._report_source_provenance.cache_clear()

    assert provenance["head"] == report._git_rev_parse("HEAD")
    assert provenance["report_version"] == report.REPORT_VERSION
    assert provenance["cache_schema_version"] == report.CACHE_SCHEMA_VERSION
    assert provenance["compiled_model_schema_version"] == 9
    assert provenance["model_compiler_version"] == 11


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
    output = (
        "AMPICOL_PROBE_VALUE 1 3 2 1.60358797632899820000000E+000\n"
    )

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
    assert report._legacy_probe_scope_limited(
        "STOP 1\n more than two quarks           3\n"
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
    assert json.loads(Path(record["output_path"]).read_text(encoding="utf-8"))[
        "value_decimal"
    ] == "1.25"


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
            lc_row_partitions=(
                SimpleNamespace(row=1, value=value, permutation=(1, 2)),
            )
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
    assert report._legacy_adaptive_profile_points(
        0.05,
        target_runtime=20.0,
    ) == 40_000
    assert report._legacy_adaptive_profile_points(
        1.0e-9,
        target_runtime=20.0,
    ) == report.DEFAULT_LEGACY_PROFILE_MAX_POINTS
    assert report._legacy_adaptive_profile_points(
        1000.0,
        target_runtime=20.0,
    ) == report.DEFAULT_LEGACY_PROFILE_MIN_POINTS


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
    previous = {
        "generation_seconds": 12.5,
        "metadata": {
            "model_precompile_policy": report.PYAMPLICOL_GENERATION_PROFILE_POLICY,
            "generation_timer_excludes_model_compile": True,
        },
    }
    artifact_dir = tmp_path / "artifact"
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

    (artifact_dir / "model").mkdir(parents=True)
    (artifact_dir / "artifact.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model" / "compiled-model.json").write_text(
        json.dumps({"schema_version": 9, "model_compiler_version": 9}),
        encoding="utf-8",
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
