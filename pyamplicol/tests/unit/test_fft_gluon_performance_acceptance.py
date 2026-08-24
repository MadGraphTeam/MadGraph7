# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from tools.developer import fft_gluon_performance_acceptance as acceptance


def _probe(
    *,
    lane: str,
    warm: float,
    rss_kib: int = 1024,
    point_values: tuple[float, ...] = (1.0,) * 10,
    load: float = 0.2,
    first_warm: float = 0.3,
) -> acceptance.CandidateProbeMetrics:
    samples = tuple(warm * (1.0 + 0.001 * index) for index in range(10))
    warm_cells = tuple((sample,) for sample in samples)
    return acceptance.CandidateProbeMetrics(
        process="gg_N4",
        execution_mode=lane,
        helicity_coverage_count=1 if lane == "recurrence" else 16,
        selected_helicity_id="h:-1,-1,+1,+1",
        point_count=10,
        point_values=point_values,
        load_seconds=load,
        first_warm_seconds=first_warm,
        warm_up_api_seconds=0.0 if lane == "recurrence" else 0.25,
        calibration_calls=(100,),
        calibration_seconds=(0.26,),
        warm_cell_seconds=warm_cells,
        warm_samples_seconds=samples,
        warm_median_seconds=acceptance.statistics.median(samples),
        minimum_absolute_value=1.0,
        max_rss_kib=rss_kib,
    )


def _candidate(
    lane: str,
    total_gluons: int,
    warm: float,
    *,
    generation: float = 1.0,
    rss_kib: int = 1024,
    numerical_passes: bool = True,
    load: float = 0.2,
    first_warm: float = 0.3,
) -> acceptance.CandidateMetrics:
    scale_factor = acceptance.candidate_reference_scale_factor(total_gluons)
    point_values = [1.0 / scale_factor] * 10
    if not numerical_passes:
        point_values[4] *= 1.0 + 1.0e-8
    numerical_parity = acceptance.compare_candidate_to_reference(
        total_gluons=total_gluons,
        candidate_values=point_values,
        reference_values=(1.0,) * 10,
    )
    return acceptance.CandidateMetrics(
        lane=lane,
        total_gluons=total_gluons,
        generator_seed=acceptance.generator_seed(total_gluons),
        generation_seconds=generation,
        load_seconds=load,
        first_warm_seconds=first_warm,
        max_rss_kib=rss_kib,
        probe=_probe(
            lane=lane,
            warm=warm,
            rss_kib=rss_kib,
            point_values=tuple(point_values),
            load=load,
            first_warm=first_warm,
        ),
        numerical_parity=numerical_parity,
    )


def _reference(
    total_gluons: int,
    *,
    warm: float = 1.0,
    rss_kib: int = 1024,
    build: float = 1.0,
    setup: float | None = None,
) -> acceptance.ReferenceMetrics:
    return acceptance.ReferenceMetrics(
        total_gluons=total_gluons,
        generator_seed=acceptance.generator_seed(total_gluons),
        backend=acceptance.REFERENCE_BACKEND,
        clean_build_scope=acceptance.REFERENCE_CLEAN_BUILD_SCOPE,
        clean_build_command_count=11,
        clean_build_seconds=build,
        setup_to_driver_seconds=build if setup is None else setup,
        initialization_seconds=0.2,
        first_pass_seconds=0.3,
        warm_samples_seconds=(warm,) * 10,
        warm_median_seconds=warm,
        max_rss_kib=rss_kib,
        selected_helicity=(-1, -1, 1, 1),
        selected_path="mhv",
        event_paths=tuple(f"point-{index}.event" for index in range(10)),
        matrix_elements=(1.0,) * 10,
    )


def test_reference_formal_cold_and_scaling_setup_metrics_remain_distinct() -> None:
    reference = _reference(4, build=1.0, setup=2.0)

    assert reference.cold_to_ready_seconds == pytest.approx(1.5)
    assert reference.setup_to_ready_seconds == pytest.approx(2.5)
    payload = acceptance._plain_reference(reference)
    assert payload["cold_to_ready_seconds"] == pytest.approx(1.5)
    assert payload["setup_to_ready_seconds"] == pytest.approx(2.5)
    assert payload["clean_build_scope"] == "ampligluon-trace-backend-only"


def test_reference_clean_build_boundary_excludes_support_tools(tmp_path: Path) -> None:
    build_dir = tmp_path / "reference" / "N4" / "build"
    backend_dir = build_dir / acceptance.REFERENCE_BUILD_PROGRAM
    backend_compile = (
        "gfortran",
        f"-J{backend_dir / 'modules'}",
        "-c",
        "trace_colour_matrix.f90",
        "-o",
        str(backend_dir / "05_trace_colour_matrix.o"),
    )
    rambo_compile = (
        "gfortran",
        "-c",
        "find_zero.f90",
        "-o",
        str(build_dir / "generate_ampligluon_events" / "00_find_zero.o"),
    )
    proxy_link = (
        "gfortran",
        "benchmark_helicity_proxy.f90",
        "-o",
        str(build_dir / "benchmark_helicity_proxy" / "benchmark_helicity_proxy"),
    )

    assert acceptance._is_reference_backend_build_command(
        backend_compile, build_dir=build_dir, build_phase_active=True
    )
    assert not acceptance._is_reference_backend_build_command(
        rambo_compile, build_dir=build_dir, build_phase_active=True
    )
    assert not acceptance._is_reference_backend_build_command(
        proxy_link, build_dir=build_dir, build_phase_active=True
    )
    assert not acceptance._is_reference_backend_build_command(
        (str(backend_dir / acceptance.REFERENCE_BUILD_PROGRAM),),
        build_dir=build_dir,
        build_phase_active=False,
    )


def test_reference_cold_timeout_remains_live_for_the_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acceptance.time, "perf_counter", lambda: 105.0)

    assert acceptance._bounded_reference_cold_timeout(30.0, 10.0, 100.0) == 5.0
    with pytest.raises(
        acceptance.ReferenceColdLimitError, match="aggregate cold deadline"
    ):
        acceptance._bounded_reference_cold_timeout(30.0, 5.0, 100.0)
    source = Path(acceptance.__file__).read_text(encoding="utf-8")
    assert "cold_setup_active" not in source


def _probe_output(
    *,
    calibration: float = 0.26,
    lane: str = "recurrence",
) -> str:
    rows = [
        "FFT_CANDIDATE_PROBE_V4",
        "PROCESS gg_N4",
        f"EXECUTION_MODE {lane}",
        "TIMER_SOURCE process-cpu-time",
        "HELICITY_COVERAGE_COUNT 1",
        "SELECTED_HELICITY_ID h:-1,-1,+1,+1",
        "POINT_COUNT 10",
        "LOAD_SECONDS 2.0e-1",
        "FIRST_WARM_SECONDS 3.0e-1",
        f"WARM_UP_API_SECONDS {0.0 if lane == 'recurrence' else 0.25}",
        *(f"POINT_VALUE {point} {1.0 / 512.0}" for point in range(1, 11)),
        f"CALIBRATION_CELL 1 101 {calibration}",
        *(f"WARM_CELL_SECONDS {batch} 1 {batch * 1.0e-6}" for batch in range(1, 11)),
        "MIN_ABSOLUTE_VALUE 1.0e-20",
        "MAX_RSS_KIB 4096",
    ]
    return "\n".join(rows) + "\n"


def _first_ready_output(*, lane: str = "recurrence", rss_kib: int = 4096) -> str:
    return (
        "\n".join(
            (
                "FFT_CANDIDATE_FIRST_READY_V1",
                "PROCESS gg_N4",
                f"EXECUTION_MODE {lane}",
                "TIMER_SOURCE process-cpu-time",
                "HELICITY_COVERAGE_COUNT 1",
                "SELECTED_HELICITY_ID h:-1,-1,+1,+1",
                "POINT_COUNT 10",
                "LOAD_SECONDS 0.2",
                "FIRST_WARM_SECONDS 0.3",
                f"WARM_UP_API_SECONDS {0.0 if lane == 'recurrence' else 0.25}",
                "MIN_ABSOLUTE_VALUE 1e-20",
                f"MAX_RSS_KIB {rss_kib}",
            )
        )
        + "\n"
    )


def _without_warm_cell(output: str, batch: int, point: int) -> str:
    prefix = f"WARM_CELL_SECONDS {batch} {point} "
    return (
        "\n".join(line for line in output.splitlines() if not line.startswith(prefix))
        + "\n"
    )


def _without_point_value(output: str, point: int) -> str:
    prefix = f"POINT_VALUE {point} "
    return (
        "\n".join(line for line in output.splitlines() if not line.startswith(prefix))
        + "\n"
    )


def _duplicate_warm_cell(
    output: str,
    source: tuple[int, int],
    destination: tuple[int, int],
) -> str:
    prefix = f"WARM_CELL_SECONDS {source[0]} {source[1]} "
    return (
        "\n".join(
            (
                f"WARM_CELL_SECONDS {destination[0]} {destination[1]} {line.split()[3]}"
                if line.startswith(prefix)
                else line
            )
            for line in output.splitlines()
        )
        + "\n"
    )


def test_candidate_probe_parser_requires_canonical_ten_sample_evidence() -> None:
    parsed = acceptance.parse_candidate_probe_output(_probe_output())

    assert parsed.execution_mode == "recurrence"
    assert parsed.point_count == 10
    assert parsed.point_values == pytest.approx((1.0 / 512.0,) * 10)
    assert parsed.warm_up_api_seconds == 0.0
    assert parsed.calibration_calls == (101,)
    assert parsed.calibration_seconds == pytest.approx((0.26,))
    assert len(parsed.warm_cell_seconds) == 10
    assert all(len(row) == 1 for row in parsed.warm_cell_seconds)
    assert parsed.warm_samples_seconds == pytest.approx(
        tuple(batch * 1.0e-6 for batch in range(1, 11))
    )
    assert parsed.warm_median_seconds == pytest.approx(5.5e-6)
    assert parsed.max_rss_kib == 4096


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (_probe_output(calibration=0.249), "0.25 seconds"),
        (_without_point_value(_probe_output(), 10), "all 10 point values"),
        (_without_warm_cell(_probe_output(), 10, 1), "point-1 cell"),
        (
            _probe_output().replace(
                "CALIBRATION_CELL 1 101 0.26",
                "CALIBRATION_CELL 1 101 0.26\nCALIBRATION_CELL 2 102 0.26",
            ),
            "calibrate only representative point 1",
        ),
        (
            _duplicate_warm_cell(_probe_output(), (10, 1), (9, 1)),
            "duplicated",
        ),
    ),
)
def test_candidate_probe_parser_rejects_incomplete_policy_evidence(
    payload: str,
    message: str,
) -> None:
    with pytest.raises(acceptance.AcceptanceError, match=message):
        acceptance.parse_candidate_probe_output(payload)


def test_candidate_probe_parser_enforces_lane_specific_warm_up_contract() -> None:
    otf = acceptance.parse_candidate_probe_output(_probe_output(lane="on-the-fly"))
    assert otf.warm_up_api_seconds == pytest.approx(0.25)

    bad_recurrence = _probe_output().replace(
        "WARM_UP_API_SECONDS 0.0", "WARM_UP_API_SECONDS 0.25"
    )
    with pytest.raises(acceptance.AcceptanceError, match="OTF-only"):
        acceptance.parse_candidate_probe_output(bad_recurrence)


@pytest.mark.parametrize("lane", acceptance.LANES)
def test_candidate_identity_requires_lane_coverage_and_the_paired_helicity(
    lane: str,
) -> None:
    metrics = _probe(lane=lane, warm=1.0)
    expected = (-1, -1, 1, 1)

    acceptance._validate_candidate_probe_identity(
        metrics,
        lane=lane,
        total_gluons=4,
        expected_helicities=expected,
    )
    invalid_coverage = 2 if lane == "recurrence" else 1
    with pytest.raises(acceptance.AcceptanceError, match=r"coverage|specialization"):
        acceptance._validate_candidate_probe_identity(
            replace(metrics, helicity_coverage_count=invalid_coverage),
            lane=lane,
            total_gluons=4,
            expected_helicities=expected,
        )
    with pytest.raises(acceptance.AcceptanceError, match="known-nonzero helicity"):
        acceptance._validate_candidate_probe_identity(
            replace(metrics, selected_helicity_id="h:-1,+1,-1,+1"),
            lane=lane,
            total_gluons=4,
            expected_helicities=expected,
        )


def test_observed_default_bg_n4_protocol_times_one_cell_but_retains_ten_mes() -> None:
    rows = [
        f"BACKEND {acceptance.REFERENCE_BACKEND}",
        "DIMENSION 6",
        "INITIALIZATION_SECONDS 1.0e-3",
        "FIRST_HELICITY_SWEEP_SECONDS 2.0e-3",
        "EVALUATIONS_PER_SWEEP 1",
        *(f"MATRIX_ELEMENT {point} 1 {point * 1.0e-3}" for point in range(1, 11)),
        *(
            f"EVALUATION_CELL_SECONDS {batch} 1 1 {batch * 1.0e-6}"
            for batch in range(1, 11)
        ),
    ]
    reference = acceptance._load_reference_module()
    run = reference.parse_driver_output(
        "\n".join(rows) + "\n",
        acceptance.REFERENCE_BACKEND,
        False,
    )

    samples = acceptance._reference_representative_warm_samples(run.cell_timings)

    assert set(run.matrix_elements) == {(point, 1) for point in range(1, 11)}
    assert samples == pytest.approx(tuple(batch * 1.0e-6 for batch in range(1, 11)))
    assert acceptance.statistics.median(samples) == pytest.approx(5.5e-6)

    with pytest.raises(acceptance.AcceptanceError, match="one point-1 cell"):
        acceptance._reference_representative_warm_samples(
            run.cell_timings | {(1, 2, 1): 1.0e-6}
        )


@pytest.mark.parametrize("total_gluons", (4, 9, 11))
def test_numerical_parity_aligns_unit_coupling_and_me_conventions(
    total_gluons: int,
) -> None:
    reference_values = tuple(float(point * point + 1) for point in range(1, 11))
    scale_factor = 256 * acceptance.math.factorial(total_gluons - 2)
    candidate_values = tuple(value / scale_factor for value in reference_values)

    evidence = acceptance.compare_candidate_to_reference(
        total_gluons=total_gluons,
        candidate_values=candidate_values,
        reference_values=reference_values,
    )

    assert evidence.normalization_alpha_s_me_check == pytest.approx(
        1.0 / (4.0 * acceptance.math.pi)
    )
    assert evidence.candidate_scale_factor == scale_factor
    assert evidence.relative_tolerance == 1.0e-10
    assert evidence.maximum_relative_error <= 1.0e-15
    assert evidence.passes is True


@pytest.mark.parametrize("mismatch_multiplier", (1.0 + 2.0e-10, 0.0, -1.0))
def test_numerical_parity_rejects_a_strict_scale_relative_mismatch(
    mismatch_multiplier: float,
) -> None:
    reference_values = (1.0,) * 10
    scale_factor = acceptance.candidate_reference_scale_factor(4)
    candidate_values = [1.0 / scale_factor] * 10
    candidate_values[6] *= mismatch_multiplier

    evidence = acceptance.compare_candidate_to_reference(
        total_gluons=4,
        candidate_values=candidate_values,
        reference_values=reference_values,
    )

    assert evidence.maximum_relative_error_point == 7
    assert evidence.maximum_relative_error > 1.0e-10
    assert evidence.passes is False


def test_candidate_first_ready_parser_uses_the_same_lane_contract() -> None:
    parsed = acceptance.parse_candidate_first_ready_output(_first_ready_output())
    assert parsed.execution_mode == "recurrence"
    assert parsed.warm_up_api_seconds == 0.0
    zero_candidate = acceptance.parse_candidate_first_ready_output(
        _first_ready_output().replace(
            "MIN_ABSOLUTE_VALUE 1e-20", "MIN_ABSOLUTE_VALUE 0"
        )
    )
    assert zero_candidate.minimum_absolute_value == 0.0


def test_darwin_reference_command_uses_getrusage_for_gnu_and_direct_wrappers() -> None:
    wrapped = (
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "600s",
        "/usr/bin/time",
        "--format=BENCHMARK_MAX_RSS_KIB %M",
        "/usr/bin/prlimit",
        "--as=32212254720",
        "--",
        "/workspace/reference-driver",
        "default-bg",
        "fft",
    )

    translated, measured = acceptance._translate_darwin_reference_command(
        wrapped, python="/venv/python"
    )

    assert measured is True
    assert translated == (
        "/venv/python",
        str(acceptance.SCALING_STUDY_DRIVER),
        "_time-rss",
        "/workspace/reference-driver",
        "default-bg",
        "fft",
    )
    direct, measured = acceptance._translate_darwin_reference_command(
        ("/usr/bin/time", "-l", "/workspace/reference-driver"),
        python="/venv/python",
    )
    assert measured is True
    assert direct == (
        "/venv/python",
        str(acceptance.SCALING_STUDY_DRIVER),
        "_time-rss",
        "/workspace/reference-driver",
    )
    compiler = ("gfortran", "--version")
    assert acceptance._translate_darwin_reference_command(compiler) == (
        compiler,
        False,
    )


def test_darwin_getrusage_rss_is_adapted_to_the_reference_protocol() -> None:
    getrusage = subprocess.CompletedProcess(
        args=("_time-rss", "reference-driver"),
        returncode=0,
        stdout="BACKEND AmpliGluonTraceDefaultBG\n",
        stderr="FFT_MAX_RSS_KIB 125\n",
    )
    normalized = acceptance._synthesize_reference_rss_marker(getrusage)
    assert normalized.stderr == ("FFT_MAX_RSS_KIB 125\nBENCHMARK_MAX_RSS_KIB 125\n")
    assert (
        acceptance._parse_translated_reference_child_rss_kib(normalized.stderr) == 125
    )
    with pytest.raises(acceptance.AcceptanceError, match="matching exact child"):
        acceptance._parse_translated_reference_child_rss_kib(
            "FFT_MAX_RSS_KIB 125\nBENCHMARK_MAX_RSS_KIB 999\n"
        )
    compiler_watchdog_rss_kib = 200
    wrapper_watchdog_rss_kib = 4096
    formal_evaluator_rss_kib = acceptance._formal_reference_evaluator_rss_kib(
        125,
        compiler_watchdog_rss_kib,
        125,
    )
    assert formal_evaluator_rss_kib == 125
    assert formal_evaluator_rss_kib != compiler_watchdog_rss_kib
    assert formal_evaluator_rss_kib != wrapper_watchdog_rss_kib


def test_global_lane_is_selected_once_from_both_valid_lane_workloads() -> None:
    references = {
        total: _reference(total) for total in acceptance.MANDATORY_MULTIPLICITIES
    }
    candidates = {
        "on-the-fly": {
            total: _candidate("on-the-fly", total, 0.1)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
        "recurrence": {
            total: _candidate("recurrence", total, 1.0)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
    }

    winner = acceptance.select_global_lane(references, candidates)
    eligibility = acceptance.lane_eligibility(candidates)

    assert winner == "on-the-fly"
    assert eligibility["on-the-fly"]["eligible"] is True
    assert eligibility["on-the-fly"]["generation_specialized"] is False
    assert eligibility["on-the-fly"]["complete_helicity_coverage"] is True
    assert eligibility["recurrence"]["eligible"] is True
    assert eligibility["recurrence"]["generation_specialized"] is True
    assert eligibility["recurrence"]["complete_helicity_coverage"] is False
    assert all(
        candidates["on-the-fly"][total].probe.warm_median_seconds
        < candidates["recurrence"][total].probe.warm_median_seconds
        for total in acceptance.MANDATORY_MULTIPLICITIES
    )


def test_otf_without_complete_selector_coverage_is_ineligible() -> None:
    candidates = {
        "on-the-fly": {
            total: _candidate("on-the-fly", total, 0.1)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
        "recurrence": {
            total: _candidate("recurrence", total, 1.0)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
    }

    for total, candidate in tuple(candidates["on-the-fly"].items()):
        candidates["on-the-fly"][total] = replace(
            candidate,
            probe=replace(candidate.probe, helicity_coverage_count=1),
        )

    assert acceptance.lane_eligibility(candidates)["on-the-fly"]["eligible"] is False
    with pytest.raises(acceptance.AcceptanceError, match="workload contract"):
        acceptance.evaluate_gates("on-the-fly", {}, candidates)


def test_slower_passing_lane_wins_when_faster_lane_fails_rss() -> None:
    references = {
        total: _reference(total, rss_kib=1000)
        for total in acceptance.MANDATORY_MULTIPLICITIES
    }
    candidates = {
        "recurrence": {
            total: _candidate("recurrence", total, 0.1, rss_kib=2001)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
        "on-the-fly": {
            total: _candidate("on-the-fly", total, 1.0, rss_kib=1000)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
    }

    assert acceptance.select_global_lane(references, candidates) == "on-the-fly"


def test_incremental_viability_keeps_otf_when_recurrence_fails() -> None:
    references = {4: _reference(4)}
    candidates = {
        "recurrence": {4: _candidate("recurrence", 4, 2.0)},
        "on-the-fly": {4: _candidate("on-the-fly", 4, 0.1)},
    }

    viability = acceptance._observed_mandatory_gate_viability(
        references, candidates, multiplicities=(4,)
    )

    assert viability["viable_lanes"] == ["on-the-fly"]
    assert viability["lanes"]["recurrence"]["viable"] is False
    assert viability["lanes"]["on-the-fly"]["eligible"] is True
    assert viability["lanes"]["on-the-fly"]["viable"] is True


def test_gate_ratios_use_only_the_global_winner_and_include_thresholds() -> None:
    references = {
        total: _reference(
            total,
            warm=1.0,
            rss_kib=1000,
            build=1.0,
            setup=100.0,
        )
        for total in acceptance.MANDATORY_MULTIPLICITIES
    }
    candidates = {
        "on-the-fly": {
            total: _candidate("on-the-fly", total, 100.0, rss_kib=100_000)
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
        "recurrence": {
            total: _candidate(
                "recurrence",
                total,
                1.25 / 1.0045,
                generation=7.0,
                rss_kib=2000,
            )
            for total in acceptance.MANDATORY_MULTIPLICITIES
        },
    }

    gates = acceptance.evaluate_gates("recurrence", references, candidates)

    assert {gate.lane for gate in gates} == {"recurrence"}
    assert len(gates) == 6
    assert all(gate.warm_ratio == pytest.approx(1.25) for gate in gates)
    assert all(gate.rss_ratio == pytest.approx(2.0) for gate in gates)
    assert all(gate.cold_ratio == pytest.approx(5.0) for gate in gates)
    assert all(gate.passes for gate in gates)


def test_numerical_mismatch_fails_an_otherwise_passing_gate() -> None:
    references = {
        total: _reference(total) for total in acceptance.MANDATORY_MULTIPLICITIES
    }
    candidates = {
        lane: {
            total: _candidate(
                lane,
                total,
                1.0 / 1.0045,
                numerical_passes=(lane != "recurrence" or total != 7),
            )
            for total in acceptance.MANDATORY_MULTIPLICITIES
        }
        for lane in acceptance.LANES
    }

    gates = acceptance.evaluate_gates("recurrence", references, candidates)

    failed = next(gate for gate in gates if gate.total_gluons == 7)
    assert failed.warm_passes is True
    assert failed.rss_passes is True
    assert failed.cold_passes is True
    assert failed.numerical_passes is False
    assert failed.passes is False
    assert acceptance._gate_report(gates)["passes"] is False


def _gate(total_gluons: int, *, passes: bool) -> acceptance.GateResult:
    return acceptance.GateResult(
        total_gluons=total_gluons,
        lane="recurrence",
        warm_ratio=1.0 if passes else 1.26,
        rss_ratio=1.0,
        cold_ratio=1.0,
        warm_passes=passes,
        rss_passes=True,
        cold_passes=True,
        numerical_passes=True,
    )


@pytest.mark.parametrize(
    ("optional_gates", "expected"),
    (
        ((_gate(10, passes=False),), False),
        ((_gate(10, passes=True), _gate(11, passes=True)), True),
        ((), True),
    ),
    ids=("measured-fail", "measured-pass", "skipped"),
)
def test_gate_report_includes_every_measured_optional_row(
    monkeypatch: pytest.MonkeyPatch,
    optional_gates: tuple[acceptance.GateResult, ...],
    expected: bool,
) -> None:
    mandatory = tuple(
        _gate(total_gluons, passes=True)
        for total_gluons in acceptance.MANDATORY_MULTIPLICITIES
    )

    report = acceptance._gate_report((*mandatory, *optional_gates))

    assert report["passes"] is expected
    serialized = report["gates"]
    assert isinstance(serialized, list)
    assert [gate["total_gluons"] for gate in serialized] == [
        *acceptance.MANDATORY_MULTIPLICITIES,
        *(gate.total_gluons for gate in optional_gates),
    ]
    assert all(gate["passes"] for gate in serialized[:6])
    assert [gate["passes"] for gate in serialized[6:]] == [
        gate.passes for gate in optional_gates
    ]
    monkeypatch.setattr(acceptance, "_campaign", lambda _arguments: report)
    assert acceptance.main(("--run-id", "pytest-optional-gate-report")) == (
        0 if expected else 1
    )


def test_optional_numerical_mismatch_is_fatal_and_persisted_not_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance_root = tmp_path / "performance"
    monkeypatch.setattr(acceptance, "PERFORMANCE_ROOT", performance_root)
    source_identity = acceptance.ReportSourceIdentity("a" * 40, "b" * 40, ())
    monkeypatch.setattr(
        acceptance,
        "_require_campaign_source_identity",
        lambda _expected=None: source_identity,
    )
    reference_source_identity = acceptance.ReferenceSourceIdentity(
        acceptance.REFERENCE_REVISION, "c" * 64, 4
    )
    monkeypatch.setattr(
        acceptance,
        "_require_reference_source_identity",
        lambda _expected=None: reference_source_identity,
    )
    monkeypatch.setattr(acceptance, "_workspace_environment", lambda _python: {})
    monkeypatch.setattr(
        acceptance,
        "_enforce_one_core",
        lambda _platform: {"requested_cpu_cores": 1},
    )
    monkeypatch.setattr(
        acceptance,
        "_build_probe",
        lambda **_kwargs: (
            tmp_path / "probe",
            {"target": "test-target", "package_version": "test-version"},
        ),
    )
    monkeypatch.setattr(acceptance, "_load_reference_module", object)

    def fake_reference(**kwargs: object) -> acceptance._ReferenceRun:
        total_gluons = int(kwargs["total_gluons"])
        return acceptance._ReferenceRun(
            metrics=_reference(total_gluons),
            event_paths=tuple(
                tmp_path / f"N{total_gluons}-point-{point}.event" for point in range(10)
            ),
        )

    def fake_candidate(**kwargs: object) -> acceptance.CandidateMetrics:
        total_gluons = int(kwargs["total_gluons"])
        return _candidate(
            str(kwargs["lane"]),
            total_gluons,
            0.5 if kwargs["lane"] == "recurrence" else 1.0,
            numerical_passes=total_gluons != 10,
        )

    monkeypatch.setattr(acceptance, "_run_reference", fake_reference)
    monkeypatch.setattr(acceptance, "_run_candidate", fake_candidate)
    arguments = acceptance._parser().parse_args(
        ("--include-optional", "--run-id", "optional-numerical-mismatch")
    )

    with pytest.raises(acceptance.NumericalParityError, match="N=10"):
        acceptance._campaign(arguments)

    report = json.loads(
        (
            performance_root / "runs" / "optional-numerical-mismatch" / "report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "failed-numerical-parity"
    assert report["optional"]["10"]["status"] == "failed-numerical-parity"
    optional_lanes = [
        lane for lane, records in report["candidates"].items() if "10" in records
    ]
    assert optional_lanes == ["recurrence"]
    evidence = report["candidates"][optional_lanes[0]]["10"]
    assert len(evidence["probe"]["point_values"]) == 10
    assert evidence["numerical_parity"]["passes"] is False
    assert evidence["numerical_parity"]["maximum_relative_error"] > 1.0e-10
    assert len(report["reference"]["10"]["matrix_elements"]) == 10


def test_optional_policy_enforces_independent_stage_and_memory_caps() -> None:
    accepted = _candidate(
        "recurrence",
        10,
        1.0,
        generation=899.75,
        first_warm=0.5,
    )
    slow_generation = _candidate("recurrence", 10, 1.0, generation=900.0)
    slow_first_warm = _candidate("recurrence", 10, 1.0, first_warm=900.0)
    large = _candidate(
        "recurrence",
        10,
        1.0,
        rss_kib=(30 * 1024**2) + 1,
    )

    assert accepted.cold_to_ready_seconds > 900.0
    assert acceptance.optional_candidate_is_feasible(accepted)
    assert not acceptance.optional_candidate_is_feasible(slow_generation)
    assert not acceptance.optional_candidate_is_feasible(slow_first_warm)
    assert not acceptance.optional_candidate_is_feasible(large)


def test_candidate_generation_commands_lock_paired_helicity_semantics(
    tmp_path: Path,
) -> None:
    on_the_fly = acceptance.candidate_generation_command(
        python="python-test",
        lane="on-the-fly",
        total_gluons=4,
        helicities=(1, 1, 1, 1),
        artifact=tmp_path / "artifact",
    )
    recurrence = acceptance.candidate_generation_command(
        python="python-test",
        lane="recurrence",
        total_gluons=4,
        helicities=(1, 1, 1, 1),
        artifact=tmp_path / "recurrence-artifact",
    )

    for command, lane in ((on_the_fly, "on-the-fly"), (recurrence, "recurrence")):
        assert command[:4] == ("python-test", "-m", "pyamplicol", "generate")
        assert command[command.index("--color-accuracy") + 1] == "full"
        assert command[command.index("--color-contraction") + 1] == (
            "symmetric-group-fft"
        )
        assert command[command.index("--execution-mode") + 1] == lane
        assert "evaluator.batch_size=1" in command
        assert "evaluator.optimization.cores=1" in command
        assert "model.cache=false" in command
    selection = "process.selected_source_helicities={1=1,2=1,3=1,4=1}"
    assert selection not in on_the_fly
    assert selection in recurrence


def test_dry_run_is_nonwriting_and_covers_locked_seeds(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "pytest-dry-run-fft-policy"
    run_root = acceptance.PERFORMANCE_ROOT / "runs" / run_id
    assert not run_root.exists()

    monkeypatch.setattr(acceptance.sys, "platform", "darwin")
    result = acceptance.main(("--dry-run", "--include-optional", "--run-id", run_id))

    assert result == 0
    assert not run_root.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["host_platform"] == "darwin"
    assert payload["reference"]["process_measurement"] == (
        "darwin-getrusage-child-plus-watchdog"
    )
    assert payload["reference"]["rss_marker"] == "FFT_MAX_RSS_KIB"
    assert [item["total_gluons"] for item in payload["multiplicities"]] == list(
        range(4, 12)
    )
    assert [item["generator_seed"] for item in payload["multiplicities"]] == list(
        range(1733, 1741)
    )
    assert all(
        item["candidate_lanes"] == ["recurrence", "on-the-fly"]
        for item in payload["multiplicities"][:6]
    )
    assert all(
        item["candidate_lanes"] == ["global-winner"]
        for item in payload["multiplicities"][6:]
    )
    assert payload["measurement"] | {"cpu_policy": None} == {
        "batch_size": 1,
        "calibration_scope": "representative-point-1-only",
        "calibration_seconds_minimum": 0.25,
        "candidate_timer_source": "process-cpu-time",
        "cpu_cores": 1,
        "cpu_policy": None,
        "fresh_process_same_os_rss": True,
        "mandatory_lane_order": ["recurrence", "on-the-fly"],
        "memory_limit_gib": 30.0,
        "model_cache": "disabled-per-lane-and-multiplicity",
        "optional_cold_and_warm_processes_separate": True,
        "timed_event_cells_per_batch": 1,
        "warm_batch_reduction": "point-1-cell-direct",
        "warm_excludes_first_call": True,
        "warm_samples": 10,
    }
    cpu_policy = payload["measurement"]["cpu_policy"]
    assert cpu_policy["method"] == "single-thread-environment"
    assert cpu_policy["requested_cpu_cores"] == 1
    assert cpu_policy["affinity_available"] is False
    assert cpu_policy["affinity_enforced"] is False
    assert cpu_policy["cpu"] is None
    assert set(cpu_policy["thread_environment"]) == set(
        acceptance.SINGLE_THREAD_ENVIRONMENT
    )
    assert set(cpu_policy["thread_environment"].values()) == {"1"}
    assert "on-the-fly" in payload["global_lane_policy"]
    assert (
        "eligible under their respective helicity workloads"
        in payload["global_lane_policy"]
    )
    assert "passing every mandatory gate" in payload["global_lane_policy"]
    assert payload["helicity_policy"] == {
        "on-the-fly": "complete-runtime-selector-fixed-helicity-query",
        "recurrence": "generation-specialized-known-nonzero",
    }
    assert payload["thresholds"] | {
        "warm_ratio_maximum": None,
        "rss_ratio_maximum": None,
        "cold_to_ready_ratio_maximum": None,
    } == {
        "warm_ratio_maximum": None,
        "rss_ratio_maximum": None,
        "cold_to_ready_ratio_maximum": None,
        "optional_generation_seconds_maximum_exclusive": 900.0,
        "optional_first_warm_seconds_maximum_exclusive": 900.0,
        "optional_continuation_deadlines_are_independent": True,
    }
    assert payload["reference"]["formal_cold_to_ready_metric"] == (
        "clean AmpliGluonTrace backend build plus initialization plus first "
        "complete pass"
    )
    assert payload["reference"]["clean_build_scope"] == (
        "ampligluon-trace-backend-only"
    )
    assert payload["reference"]["clean_build_program"] == ("benchmark_ampligluon_trace")
    assert (
        "RAMBO and helicity-proxy builds excluded"
        in payload["reference"]["clean_build_timing"]
    )
    assert payload["reference"]["scaling_setup_to_ready_metric"] == (
        "all setup through driver plus initialization plus first complete pass"
    )
    assert payload["schema_version"] == 8
    assert payload["numerical_parity"] == {
        "reference_values": "ordered-10-AmpliGluonTrace-matrix-elements",
        "candidate_values": (
            "recurrence-first-pass-all-10;"
            "otf-first-pass-points-2-10-plus-point-1-calibration"
        ),
        "candidate_runtime_parameter": "normalization.alpha_s_me_check",
        "candidate_runtime_parameter_value": pytest.approx(
            1.0 / (4.0 * acceptance.math.pi)
        ),
        "candidate_to_reference_scale": "256*factorial(N-2)",
        "relative_error_scale": "max(abs(candidate),abs(reference))",
        "relative_tolerance": 1.0e-10,
        "failure_is_fatal": True,
        "extra_evaluations": 0,
    }


def test_dry_run_rejects_unsafe_run_id_without_writing() -> None:
    assert acceptance.main(("--dry-run", "--run-id", "../outside")) == 1


def test_probe_commands_separate_optional_cold_from_warmed_campaign(
    tmp_path: Path,
) -> None:
    common = {
        "probe": tmp_path / "probe",
        "artifact": tmp_path / "artifact",
        "total_gluons": 4,
        "target_seconds": 0.25,
        "event_paths": tuple(tmp_path / f"point-{index}.event" for index in range(10)),
    }
    first_ready = acceptance._candidate_probe_command(**common, first_ready_only=True)
    warmed = acceptance._candidate_probe_command(**common, first_ready_only=False)

    assert "--first-ready-only" in first_ready
    assert "--first-ready-only" not in warmed
    assert first_ready[-10:] == warmed[-10:]


def test_optional_candidate_gives_generation_and_first_ready_independent_deadlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], float]] = []
    outputs = iter(("", _first_ready_output(), _probe_output()))

    def fake_run_watched(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> acceptance.WatchedCompletedProcess:
        calls.append((command, float(kwargs["timeout_seconds"])))
        return acceptance.WatchedCompletedProcess(
            command,
            0,
            next(outputs),
            "",
            elapsed_seconds=1.0,
            log_write_seconds=0.1,
            timed_out=False,
            timeout_cleanup="not-required",
        )

    clock = iter((100.0, 105.0, 105.1))
    monkeypatch.setattr(acceptance, "_run_watched", fake_run_watched)
    monkeypatch.setattr(acceptance.time, "perf_counter", lambda: next(clock))
    reference = acceptance._ReferenceRun(
        metrics=_reference(4),
        event_paths=tuple(tmp_path / f"point-{index}.event" for index in range(10)),
    )

    candidate = acceptance._run_candidate(
        lane="recurrence",
        total_gluons=4,
        reference=reference,
        probe=tmp_path / "probe",
        run_root=tmp_path / "run",
        python=sys.executable,
        environment={},
        target_seconds=0.25,
        timeout_seconds=3600.0,
        continuation_stage_limit_seconds=900.0,
    )

    assert candidate.generation_seconds == pytest.approx(4.9)
    assert candidate.cold_to_ready_seconds == pytest.approx(5.4)
    assert candidate.max_rss_kib == 4096
    assert calls[0][1] == pytest.approx(900.0)
    assert "--first-ready-only" in calls[1][0]
    assert calls[1][1] == pytest.approx(3600.0)
    assert "--first-ready-only" not in calls[2][0]
    assert calls[2][1] == pytest.approx(3600.0)


def test_native_probe_uses_public_allocation_free_call_and_lane_guard() -> None:
    source = acceptance.PROBE_SOURCE.read_text(encoding="utf-8")

    assert "#include <rusticol.h>" in source
    assert "#include <rusticol.hpp>" not in source
    assert "rusticol_runtime_evaluate_f64" in source
    assert "rusticol_runtime_evaluate_selected_f64" in source
    assert 'execution_mode == "recurrence" && runtime_helicity_count == 1' in source
    assert "arguments.color_id.empty()" in source
    assert "if (generation_specialized_total)" in source
    assert "&selected_helicity_index" in source
    assert 'execution_mode == "compiled" ? nullptr' not in source
    assert 'if (execution_mode == "on-the-fly")' in source
    assert source.count("rusticol_runtime_warm_up_f64") == 1
    assert "first_evaluated_point" not in source
    assert "for (std::size_t point = 0; point < events.size(); ++point)" in source
    assert "CLOCK_PROCESS_CPUTIME_ID" in source
    assert "rusticol_runtime_set_model_parameter" in source
    assert '"normalization.alpha_s_me_check"' in source
    assert "kUnitCouplingAlphaS" in source
    assert source.count("evaluate_one(") == 4
    assert source.count("evaluate_repeated_batch(") == 2
    assert 'token == "--batch-size"' in source
    assert '"BENCHMARK_BATCH_SIZE "' in source
    assert "events[point % events.size()]" in source
    assert "authenticate_batch_results(" in source
    assert "expected[index % expected.size()]" in source
    assert '"BATCH_INPUT_PATTERN cyclic-10-events' in source
    assert '"BATCH_AUTHENTICATED_POINT_COUNT "' in source
    assert '"BATCH_MAX_RELATIVE_ERROR "' in source
    assert "CLOCK_MONOTONIC" in source
    assert '"START_TO_FIRST_WARM_WALL_SECONDS "' in source
    assert "FFT_CANDIDATE_PROBE_V4" in source
    assert "point_values[point] = evaluate_one(" in source
    assert "point_values[kRepresentativePoint] = last_value" in source
    assert "std::array<double, kWarmSampleCount> warm_cells" in source
    assert "calibration_calls[point]" not in source
    assert "warm_cells[sample][point]" not in source
    assert "POINT_VALUE" in source
    assert "CALIBRATION_CELL" in source
    assert "WARM_CELL_SECONDS" in source


class _GracefulTimeoutPopen:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.communicate_timeouts: list[float | None] = []
        self.terminated = False
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_timeouts.append(timeout)
        if len(self.communicate_timeouts) == 1:
            raise subprocess.TimeoutExpired(("watchdog",), timeout)
        self.returncode = -15
        return "partial stdout", "watchdog cleaned child tree"

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _FallbackTimeoutPopen(_GracefulTimeoutPopen):
    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_timeouts.append(timeout)
        if len(self.communicate_timeouts) <= 2:
            raise subprocess.TimeoutExpired(("watchdog",), timeout)
        self.returncode = -9
        return "partial stdout", "watchdog required fallback kill"


def test_watched_timeout_signals_watchdog_and_waits_for_tree_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _GracefulTimeoutPopen()
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *args, **kwargs: fake)
    log_path = tmp_path / "timeout.json"

    with pytest.raises(acceptance.AcceptanceError, match="sigterm-complete"):
        acceptance._run_watched(
            ("payload",),
            python=sys.executable,
            environment={},
            timeout_seconds=0.01,
            log_path=log_path,
        )

    assert fake.terminated is True
    assert fake.killed is False
    assert fake.communicate_timeouts[1] > 5.0
    evidence = json.loads(log_path.read_text(encoding="utf-8"))
    assert evidence["timed_out"] is True
    assert evidence["timeout_cleanup"] == "sigterm-complete"
    assert evidence["elapsed_excludes_log_write"] is True


def test_watched_timeout_records_fallback_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FallbackTimeoutPopen()
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *args, **kwargs: fake)
    log_path = tmp_path / "timeout-fallback.json"

    with pytest.raises(acceptance.AcceptanceError, match="sigkill-fallback"):
        acceptance._run_watched(
            ("payload",),
            python=sys.executable,
            environment={},
            timeout_seconds=0.01,
            log_path=log_path,
        )

    assert fake.terminated is True
    assert fake.killed is True
    evidence = json.loads(log_path.read_text(encoding="utf-8"))
    assert evidence["timeout_cleanup"] == "sigkill-fallback"


def test_watchdog_report_preserves_exact_peak_guard_for_rss_aggregation(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "watchdog.json"
    report_path.write_text(
        json.dumps(
            {
                "complete": True,
                "passes": True,
                "enforcement": {
                    "peak_rss_bytes": 125_953,
                    "peak_guard_bytes": 126_977,
                },
            }
        ),
        encoding="utf-8",
    )

    assert acceptance._read_watchdog_usage_kib(report_path) == (124, 125)
    command = acceptance._watchdog_command(
        ("payload",),
        python=sys.executable,
        memory_limit_gib=20.0,
        report_json=report_path,
    )
    assert command[-4:] == ("--report-json", str(report_path), "--", "payload")


def test_candidate_probe_compiles_and_links_against_public_sdk() -> None:
    compiler = acceptance.shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("no C++ compiler is available for the public-SDK smoke test")
    run_root = (
        acceptance.PERFORMANCE_ROOT
        / "test-native-smoke"
        / f"pytest-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        acceptance._create_workspace_directories(run_root)
        sdk_root = acceptance.ROOT / "src" / "pyamplicol" / "_sdk"
        metadata = json.loads((sdk_root / "metadata.json").read_text(encoding="utf-8"))
        link = json.loads((sdk_root / "link.json").read_text(encoding="utf-8"))
        probe = run_root / "candidate-probe" / "fft-gluon-candidate-probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        acceptance._run_watched(
            (
                *compiler,
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                f"-I{sdk_root / 'include'}",
                str(acceptance.PROBE_SOURCE),
                "-o",
                str(probe),
                str(sdk_root / str(metadata["archive"])),
                *(f"-l{name}" for name in link.get("system_libraries", [])),
                *(
                    token
                    for name in link.get("frameworks", [])
                    for token in ("-framework", str(name))
                ),
                *acceptance._candidate_probe_executable_link_flags(sys.platform),
            ),
            python=sys.executable,
            environment=acceptance._workspace_environment(sys.executable),
            timeout_seconds=600.0,
            log_path=run_root / "logs" / "candidate-probe-build.json",
        )
        assert probe.is_file()
        assert metadata["abi_version"] == 1
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    (
        (
            "darwin",
            ("-Wl,-dead_strip", "-Wl,-no_exported_symbols"),
        ),
        ("linux", ()),
        ("win32", ()),
    ),
)
def test_candidate_probe_executable_link_flags_are_darwin_scoped(
    platform_name: str,
    expected: tuple[str, ...],
) -> None:
    assert acceptance._candidate_probe_executable_link_flags(platform_name) == expected


def _mock_performance_campaign_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recurrence_warm: float,
    otf_warm: float = 0.1,
) -> tuple[Path, list[tuple[str, int]]]:
    performance_root = tmp_path / "performance"
    source_identity = acceptance.ReportSourceIdentity("a" * 40, "b" * 40, ())
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(acceptance, "PERFORMANCE_ROOT", performance_root)
    monkeypatch.setattr(
        acceptance,
        "_require_campaign_source_identity",
        lambda _expected=None: source_identity,
    )
    reference_source_identity = acceptance.ReferenceSourceIdentity(
        acceptance.REFERENCE_REVISION, "c" * 64, 4
    )
    monkeypatch.setattr(
        acceptance,
        "_require_reference_source_identity",
        lambda _expected=None: reference_source_identity,
    )
    monkeypatch.setattr(acceptance, "_workspace_environment", lambda _python: {})
    monkeypatch.setattr(
        acceptance,
        "_enforce_one_core",
        lambda _platform: {"requested_cpu_cores": 1},
    )
    monkeypatch.setattr(
        acceptance,
        "_build_probe",
        lambda **_kwargs: (
            tmp_path / "probe",
            {"target": "test-target", "package_version": "test-version"},
        ),
    )
    monkeypatch.setattr(acceptance, "_load_reference_module", object)

    def fake_reference(**kwargs: object) -> acceptance._ReferenceRun:
        total_gluons = int(kwargs["total_gluons"])
        calls.append(("reference", total_gluons))
        return acceptance._ReferenceRun(
            metrics=_reference(total_gluons),
            event_paths=tuple(
                tmp_path / f"N{total_gluons}-point-{point}.event" for point in range(10)
            ),
        )

    def fake_candidate(**kwargs: object) -> acceptance.CandidateMetrics:
        lane = str(kwargs["lane"])
        total_gluons = int(kwargs["total_gluons"])
        calls.append((lane, total_gluons))
        return _candidate(
            lane,
            total_gluons,
            recurrence_warm if lane == "recurrence" else otf_warm,
        )

    monkeypatch.setattr(acceptance, "_run_reference", fake_reference)
    monkeypatch.setattr(acceptance, "_run_candidate", fake_candidate)
    return performance_root, calls


def test_campaign_stops_after_first_dispositive_mandatory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance_root, calls = _mock_performance_campaign_dependencies(
        tmp_path,
        monkeypatch,
        recurrence_warm=2.0,
        otf_warm=2.0,
    )
    arguments = acceptance._parser().parse_args(("--run-id", "early-mandatory-failure"))

    result = acceptance._campaign(arguments)

    assert calls == [("reference", 4), ("recurrence", 4), ("on-the-fly", 4)]
    assert result["status"] == "failed-performance-gates"
    assert result["terminal"] is True
    assert result["passes"] is False
    assert result["dry_run"] is False
    assert result["policy"]["dry_run"] is False
    assert (
        result["source_identity"]
        == acceptance.ReportSourceIdentity("a" * 40, "b" * 40, ()).provenance()
    )
    assert (
        result["reference_source_identity"]
        == acceptance.ReferenceSourceIdentity(
            acceptance.REFERENCE_REVISION, "c" * 64, 4
        ).provenance()
    )
    assert result["failure"]["kind"] == "no-global-lane-remains"
    viability = result["observed_gate_viability"]
    assert viability["completed_multiplicities"] == [4]
    assert viability["viable_lanes"] == []
    assert viability["lanes"]["recurrence"]["eligible"] is True
    assert viability["lanes"]["recurrence"]["viable"] is False
    recurrence_gate = viability["lanes"]["recurrence"]["gates"][0]
    assert recurrence_gate["warm_passes"] is False
    assert recurrence_gate["rss_passes"] is True
    assert recurrence_gate["cold_passes"] is True
    report = json.loads(
        (
            performance_root / "runs" / "early-mandatory-failure" / "report.json"
        ).read_text(encoding="utf-8")
    )
    assert report == json.loads(json.dumps(result))


def test_campaign_keeps_running_when_only_otf_remains_viable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _performance_root, calls = _mock_performance_campaign_dependencies(
        tmp_path,
        monkeypatch,
        recurrence_warm=2.0,
        otf_warm=0.1,
    )
    arguments = acceptance._parser().parse_args(("--run-id", "otf-survives"))

    result = acceptance._campaign(arguments)

    expected_calls = [
        call
        for total in acceptance.MANDATORY_MULTIPLICITIES
        for call in (
            ("reference", total),
            ("recurrence", total),
            ("on-the-fly", total),
        )
    ]
    assert calls == expected_calls
    assert result["status"] == "complete"
    assert result["passes"] is True
    assert result["global_winning_lane"] == "on-the-fly"
    assert {gate["lane"] for gate in result["gates"]} == {"on-the-fly"}


def test_campaign_uses_one_stable_winner_for_every_mandatory_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _performance_root, calls = _mock_performance_campaign_dependencies(
        tmp_path,
        monkeypatch,
        recurrence_warm=0.5,
        otf_warm=1.0,
    )
    arguments = acceptance._parser().parse_args(("--run-id", "stable-winner"))

    result = acceptance._campaign(arguments)

    assert len(calls) == 3 * len(acceptance.MANDATORY_MULTIPLICITIES)
    assert result["status"] == "complete"
    assert result["passes"] is True
    assert result["global_winning_lane"] == "recurrence"
    assert [gate["total_gluons"] for gate in result["gates"]] == list(
        acceptance.MANDATORY_MULTIPLICITIES
    )
    assert {gate["lane"] for gate in result["gates"]} == {"recurrence"}


def test_campaign_source_preflight_rejects_dirty_tracked_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_source(_root: Path) -> acceptance.ReportSourceIdentity:
        raise acceptance.ReportSourceIdentityError(
            "dirty source paths: tools/developer/fft_gluon_performance_acceptance.py"
        )

    monkeypatch.setattr(acceptance, "require_eligible_report_source", reject_source)
    monkeypatch.setattr(acceptance, "PERFORMANCE_ROOT", tmp_path / "performance")
    monkeypatch.setattr(
        acceptance,
        "_create_workspace_directories",
        lambda _root: pytest.fail("dirty source must fail before workspace creation"),
    )
    arguments = acceptance._parser().parse_args(("--run-id", "dirty-source"))

    with pytest.raises(acceptance.AcceptanceError, match="dirty source paths"):
        acceptance._campaign(arguments)


def test_campaign_source_postflight_rejects_head_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = acceptance.ReportSourceIdentity("a" * 40, "b" * 40, ())
    changed = acceptance.ReportSourceIdentity("c" * 40, "d" * 40, ())
    monkeypatch.setattr(
        acceptance,
        "require_eligible_report_source",
        lambda _root: changed,
    )

    with pytest.raises(acceptance.AcceptanceError, match="identity changed"):
        acceptance._require_campaign_source_identity(started)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_reference_source_identity_allows_dirty_inputs_and_detects_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "reference"
    repository.mkdir()
    subprocess.run(
        ("git", "init", "--quiet", str(repository)),
        check=True,
        capture_output=True,
        text=True,
    )
    (repository / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "tracked.txt")
    _git(
        repository,
        "-c",
        "user.name=FFT acceptance",
        "-c",
        "user.email=fft-acceptance@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    revision = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(acceptance, "REFERENCE_ROOT", repository)
    monkeypatch.setattr(acceptance, "REFERENCE_REVISION", revision)

    # Intentional tracked and nonignored-untracked patches are authenticated,
    # not rejected for making the reference checkout dirty.
    (repository / "tracked.txt").write_text("intentional patch\n", encoding="utf-8")
    (repository / "extra.txt").write_text("campaign input\n", encoding="utf-8")
    started = acceptance._require_reference_source_identity()
    assert started.revision == revision
    assert started.file_count == 3
    assert acceptance._require_reference_source_identity(started) == started

    # Ignored build products are deliberately outside the source identity.
    (repository / "ignored.tmp").write_text("build output\n", encoding="utf-8")
    assert acceptance._require_reference_source_identity(started) == started

    (repository / "extra.txt").write_text("changed campaign input\n", encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="identity changed"):
        acceptance._require_reference_source_identity(started)


def test_reference_source_identity_requires_the_locked_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert acceptance.REFERENCE_REVISION == "9c3cb4fb4658200884553bab796e85bd5e7fe7a9"
    repository = tmp_path / "reference"
    repository.mkdir()
    subprocess.run(
        ("git", "init", "--quiet", str(repository)),
        check=True,
        capture_output=True,
        text=True,
    )
    (repository / "input.txt").write_text("input\n", encoding="utf-8")
    _git(repository, "add", "input.txt")
    _git(
        repository,
        "-c",
        "user.name=FFT acceptance",
        "-c",
        "user.email=fft-acceptance@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    monkeypatch.setattr(acceptance, "REFERENCE_ROOT", repository)

    with pytest.raises(acceptance.AcceptanceError, match="wrong revision"):
        acceptance._require_reference_source_identity()
