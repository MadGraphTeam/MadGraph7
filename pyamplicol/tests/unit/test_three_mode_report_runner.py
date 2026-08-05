# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.generation.artifact_writer import _evaluator as _serialize_evaluator
from tools.performance_report.arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    EMPTY_ARENA_PHASE_VECTOR_FIELDS,
    ZERO_ARENA_COUNTER_FIELDS,
    ZERO_ARENA_PHASE_TIME_FIELDS,
    ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS,
    ArenaProfileEvidenceError,
    build_arena_profile_evidence,
    digest_arena_profile_value,
    validate_arena_profile_evidence,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    Workload,
)
from tools.performance_report.runner import (
    LOADED_RUNTIME_PROFILE_COMMAND_PATH,
    PAIRED_ARENA_PROFILE_COMMAND_PATH,
    PRECOMPILED_GENERATION_COMMAND_PATH,
    PUBLIC_CLI_COMMAND_PATH,
    ProfilingTimeLimitError,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    _artifact_numerical_relation_metadata,
    _authenticated_direct_codegen_identity,
    _authenticated_effective_config,
    _authenticated_recurrence_source_identity,
    _authenticated_symjit_plane_leaves,
    _benchmark_measurement,
    _calibrate_arena_repetitions,
    _real_nonnegative,
    _regular_file_identity,
    _run_arena_benchmark,
    _run_report_benchmark,
    _selector_kwargs,
    config_values,
    derive_selector_contract,
    generate_artifact,
    point_digest,
    pointwise_validation,
    profiling_chunk_guard,
    resolved_sum_validation,
    runtime_identity_payload,
    validate_artifact_contract,
    validate_runtime_contract,
    validate_selector_contract,
)


def test_profiling_chunk_guard_rejects_only_chunks_larger_than_remaining() -> None:
    guard = profiling_chunk_guard(10.0, clock=lambda: 9.0)

    assert guard is not None
    guard(None, "one-point probe")
    guard(1.0, "exactly fitting chunk")
    with pytest.raises(ProfilingTimeLimitError, match=r"estimated=1\.00001s"):
        guard(1.00001, "oversized chunk")


def test_profiling_chunk_guard_rejects_launch_after_deadline() -> None:
    guard = profiling_chunk_guard(10.0, clock=lambda: 10.0)

    assert guard is not None
    with pytest.raises(ProfilingTimeLimitError, match="remaining=0s"):
        guard(None, "one-point probe")


def test_artifact_numerical_relation_metadata_preserves_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.artifacts as artifacts
    from pyamplicol.generation.recurrence_numerical_current_warmup import (
        RecurrenceNumericalEvidenceGeometry,
    )

    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=4,
        component_count=4,
        candidate_probe_count=2,
        verification_probe_count=2,
        runtime_parameter_count=2,
    )
    relation_report = {
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "certified_relation_count": 0,
        "applied_relation_count": 0,
        "relation_correctness": {
            "abi": "pyamplicol-numerical-current-relation-correctness-v1",
            "state": "no-applied-relations",
            "applied_relation_count": 0,
        },
        "fallback": {
            "reason": "evidence-envelope-fallback",
            "memory_envelope_bytes": 1,
            "spooled_producer_resident_bytes": (
                geometry.spooled_producer_resident_upper_bound()
            ),
            "geometry": geometry.to_json_dict(),
        },
    }
    manifest = SimpleNamespace(
        extensions={
            "generation": {
                "concrete_processes": [
                    {
                        "id": "process",
                        "filters": {"relation_discovery": relation_report},
                    }
                ]
            }
        }
    )
    loads: list[tuple[Path, bool]] = []

    def load_manifest(path: Path, *, verify_payloads: bool) -> object:
        loads.append((path, verify_payloads))
        return manifest

    monkeypatch.setattr(artifacts, "load_manifest", load_manifest)

    correctness, fallback = _artifact_numerical_relation_metadata(
        tmp_path / "artifact",
        "process",
    )

    assert loads == [(tmp_path / "artifact", False)]
    assert correctness["state"] == "no-applied-relations"
    assert fallback == {
        "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "reason": "evidence-envelope-fallback",
        "geometry": geometry.to_json_dict(),
        "certified_relation_count": 0,
        "applied_relation_count": 0,
    }


def _raw_arena_profile(
    *,
    execution_mode: str = "compiled",
    points: int = 128,
    wall_time: float = 1.1e-6 * 128,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "execution_mode": execution_mode,
        "profile_boundary": ARENA_PROFILE_BOUNDARY,
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
        "evaluator_timing_available": False,
        "points": points,
        "wall_time_s": wall_time,
        "orchestration_time_s": wall_time,
        **{field: 0 for field in ZERO_ARENA_COUNTER_FIELDS},
        **{field: 0.0 for field in ZERO_ARENA_PHASE_TIME_FIELDS},
        **{field: [] for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS},
        **{field: 0 for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS},
        "state_component_count": 11,
        "source_component_count": 7,
    }
    if execution_mode == "compiled":
        profile.update(
            {
                "compiled_direct_arena_engine_count": 2,
                "compiled_direct_arena_call_count": 3,
                "evaluator_backend_call_count": 3,
            }
        )
    return profile


def _benchmark_fixture(
    *,
    arena_authenticated: bool = True,
    execution_mode: str = "compiled",
) -> SimpleNamespace:
    evidence = build_arena_profile_evidence(
        (_raw_arena_profile(execution_mode=execution_mode),) * 5,
        execution_mode=execution_mode,
        repetitions_per_profile=1,
        batch_size=128,
    )
    return SimpleNamespace(
        uncertainty=SimpleNamespace(
            standard_error=1.0e-9,
            relative_standard_error=0.01,
        ),
        wall_time_per_point=1.0e-6,
        evaluator_time_per_point=None,
        sample_count=5,
        effective_config=SimpleNamespace(target_runtime=5.0),
        timing_breakdown=SimpleNamespace(
            wall_time=SimpleNamespace(
                mean_seconds_per_point=evidence[
                    "warmed_boundary_wall_seconds_per_point"
                ]
            ),
            evaluator_call_time=None,
        ),
        arena_profile_evidence=evidence,
        environment={
            "evaluator_time_raw_seconds_per_point": None,
            "evaluator_time_status": "unavailable",
            "evaluator_time_ratio_eligible": False,
            "evaluator_time_sample_pass": "runtime._profile_arena_repeated",
            "timing_breakdown_sample_pass": "runtime._profile_arena_repeated",
            "profile_protocol": "arena",
            "report_command_path": PAIRED_ARENA_PROFILE_COMMAND_PATH,
            "report_public_cli_path": None,
            "profile_attribution_boundary": (
                "warmed-direct-arena-borrowed-input-preallocated-output-v1"
            ),
            "profile_attribution_borrowed_flat_input": arena_authenticated,
            "profile_attribution_preallocated_output": True,
            "profile_attribution_phase_timing_scope": ("coarse-arena-boundary-only-v1"),
            "profile_attribution_evaluator_timing_available": False,
            "profile_attribution_paired_with_headline": True,
            "profile_attribution_identical_batch": True,
            "profile_attribution_identical_repetitions": True,
            "execution_mode": execution_mode,
            "evaluator_sample_count": 5,
            "native_profile_points_per_sample": 128,
            "native_profile_repetitions_per_sample": 1,
            "native_profile_batch_size": 128,
            "timing_sample_contract": (
                "paired_unprofiled_headline_profiled_attribution_v1"
            ),
            "elapsed_seconds": 5.0,
            "completed_sample_count": 5,
            "planned_sample_count": 5,
            "repetitions_per_sample": 128,
            "measured_point_count": 640,
            "interrupted": False,
        },
    )


def test_benchmark_measurement_records_authenticated_arena_unavailable_timing() -> None:
    measurement = _benchmark_measurement(
        _benchmark_fixture(),
        matrix_element=2.0,
    )

    assert measurement["execution_seconds_per_point"] is None
    assert measurement["evaluator_total_timing"] is None
    assert measurement["execution_timing"] == {
        "abi": "pyamplicol-report-arena-execution-timing-v2",
        "status": "unavailable",
        "ratio_eligible": False,
        "raw_seconds_per_point": None,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
        "profile_protocol": "arena",
        "profile_sample_pass": "runtime._profile_arena_repeated",
        "profile_boundary": (
            "warmed-direct-arena-borrowed-input-preallocated-output-v1"
        ),
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": "coarse-arena-boundary-only-v1",
        "evaluator_timing_available": False,
        "paired_with_headline": True,
        "identical_batch": True,
        "identical_repetitions": True,
        "execution_mode": "compiled",
        "warmed_boundary_wall_seconds_per_point": 1.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(
            measurement["arena_profile_evidence"]
        ),
    }
    assert measurement["benchmark_evidence"] == {
        "report_command_path": PAIRED_ARENA_PROFILE_COMMAND_PATH,
        "report_public_cli_path": None,
        "measurement_phase_elapsed_seconds": 5.0,
        "target_runtime_seconds": 5.0,
        "achieved_runtime_seconds": 5.0,
        "target_runtime_achieved": True,
        "completed_sample_count": 5,
        "planned_sample_count": 5,
        "repetitions_per_sample": 128,
        "measured_point_count": 640,
        "interrupted": False,
    }


def test_benchmark_measurement_rejects_unauthenticated_unavailable_timing() -> None:
    with pytest.raises(RunnerError, match="warmed Arena profile boundary"):
        _benchmark_measurement(
            _benchmark_fixture(arena_authenticated=False),
            matrix_element=2.0,
        )


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_benchmark_measurement_authenticates_accumulated_evaluator_total(
    execution_mode: str,
) -> None:
    benchmark = _benchmark_fixture(execution_mode=execution_mode)
    benchmark.wall_time_per_point = 1.0e-6
    benchmark.evaluator_total_time_per_point = 1.0e-6
    benchmark.environment.update(
        {
            "evaluator_total_time_raw_seconds_per_point": 1.0e-6,
            "evaluator_total_time_status": "measured",
            "evaluator_total_time_ratio_eligible": False,
            "evaluator_total_time_source": (
                "runtime._benchmark_f64_wall_time.accumulated"
            ),
            "evaluator_total_time_sample_contract": (
                "accumulated-repeated-warmed-evaluator-total-v1"
            ),
            "evaluator_total_accumulated_seconds": 6.4e-4,
        }
    )

    measurement = _benchmark_measurement(benchmark, matrix_element=2.0)

    assert measurement["execution_seconds_per_point"] is None
    assert measurement["evaluator_total_timing"] == {
        "abi": "pyamplicol-report-evaluator-total-timing-v1",
        "status": "measured",
        "ratio_eligible": False,
        "raw_seconds_per_point": 1.0e-6,
        "source": "runtime._benchmark_f64_wall_time.accumulated",
        "execution_mode": execution_mode,
        "sample_contract": ("accumulated-repeated-warmed-evaluator-total-v1"),
        "sample_count": 5,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "points_per_sample": 128,
        "measured_point_count": 640,
        "accumulated_seconds": 6.4e-4,
    }


def test_benchmark_measurement_rejects_inconsistent_accumulated_evaluator_total() -> (
    None
):
    benchmark = _benchmark_fixture()
    benchmark.evaluator_total_time_per_point = 1.0e-6
    benchmark.environment.update(
        {
            "evaluator_total_time_raw_seconds_per_point": 1.0e-6,
            "evaluator_total_time_status": "measured",
            "evaluator_total_time_ratio_eligible": False,
            "evaluator_total_time_source": (
                "runtime._benchmark_f64_wall_time.accumulated"
            ),
            "evaluator_total_time_sample_contract": (
                "accumulated-repeated-warmed-evaluator-total-v1"
            ),
            "evaluator_total_accumulated_seconds": 1.0,
        }
    )

    with pytest.raises(RunnerError, match="accumulated evaluator-total"):
        _benchmark_measurement(benchmark, matrix_element=2.0)


def test_benchmark_measurement_rejects_synthetic_zero_execution() -> None:
    benchmark = _benchmark_fixture()
    benchmark.evaluator_time_per_point = 0.0
    benchmark.environment.update(
        {
            "evaluator_time_status": "measured",
            "evaluator_time_raw_seconds_per_point": 0.0,
            "evaluator_time_ratio_eligible": False,
            "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            "compiled_direct_arena_active": False,
        }
    )

    with pytest.raises(RunnerError, match="measured execution timing"):
        _benchmark_measurement(benchmark, matrix_element=2.0)


def test_benchmark_measurement_rejects_uncertainty_for_unexposed_execution() -> None:
    benchmark = _benchmark_fixture()
    benchmark.evaluator_uncertainty = SimpleNamespace(standard_error=0.0)

    with pytest.raises(RunnerError, match="warmed Arena profile boundary"):
        _benchmark_measurement(benchmark, matrix_element=2.0)


def test_benchmark_measurement_retains_supported_recurrence_execution_timing() -> None:
    benchmark = _benchmark_fixture()
    benchmark.evaluator_time_per_point = 8.0e-7
    benchmark.evaluator_total_time_per_point = 1.2e-6
    benchmark.environment.update(
        {
            "evaluator_time_status": "measured",
            "evaluator_time_raw_seconds_per_point": 8.0e-7,
            "evaluator_time_ratio_eligible": True,
            "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            "compiled_direct_arena_active": False,
            "execution_mode": "recurrence",
            "timing_sample_contract": "paired-native-repeated-profile-v1",
            "evaluator_total_time_raw_seconds_per_point": 1.2e-6,
            "evaluator_total_time_status": "measured",
            "evaluator_total_time_ratio_eligible": False,
            "evaluator_total_time_source": (
                "runtime._benchmark_f64_wall_time.accumulated"
            ),
            "evaluator_total_time_sample_contract": (
                "accumulated-repeated-warmed-evaluator-total-v1"
            ),
            "evaluator_total_accumulated_seconds": 7.68e-4,
        }
    )

    measurement = _benchmark_measurement(benchmark, matrix_element=2.0)

    assert measurement["execution_seconds_per_point"] == 8.0e-7
    assert measurement["arena_profile_evidence"] is None
    assert measurement["evaluator_total_timing"] == {
        "abi": "pyamplicol-report-evaluator-total-timing-v1",
        "status": "measured",
        "ratio_eligible": False,
        "raw_seconds_per_point": 1.2e-6,
        "source": "runtime._benchmark_f64_wall_time.accumulated",
        "execution_mode": "recurrence",
        "sample_contract": ("accumulated-repeated-warmed-evaluator-total-v1"),
        "sample_count": 5,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "points_per_sample": 128,
        "measured_point_count": 640,
        "accumulated_seconds": 7.68e-4,
    }
    assert measurement["execution_timing"] == {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "measured",
        "ratio_eligible": True,
        "raw_seconds_per_point": 8.0e-7,
        "source": "runtime_profile_core_evaluator_call_time",
        "compiled_direct_arena_active": False,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "sample_contract": "paired-native-repeated-profile-v1",
    }


@pytest.mark.parametrize(
    "execution_mode",
    (ExecutionMode.EAGER, ExecutionMode.COMPILED),
)
def test_report_arena_benchmark_uses_private_profiler_without_public_fallback(
    execution_mode: ExecutionMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.wall_calls: list[tuple[object, int, dict[str, object]]] = []
            self.arena_calls: list[tuple[object, int, dict[str, object]]] = []
            self.public_profile_calls = 0

        def _benchmark_f64_wall_time(
            self,
            batch: object,
            repetitions: int,
            **kwargs: object,
        ) -> float:
            self.wall_calls.append((batch, repetitions, dict(kwargs)))
            return 1.0e-3

        def _profile_arena_repeated(
            self,
            batch: object,
            repetitions: int,
            **kwargs: object,
        ) -> dict[str, object]:
            self.arena_calls.append((batch, repetitions, dict(kwargs)))
            assert isinstance(batch, tuple)
            return _raw_arena_profile(
                execution_mode=execution_mode.value,
                points=len(batch) * repetitions,
                wall_time=2.0e-3,
            )

        def profile_repeated(self, *_args: object, **_kwargs: object) -> object:
            self.public_profile_calls += 1
            raise AssertionError("public profile_repeated must not be used")

    backend = Runtime()
    runtime = SimpleNamespace(_backend=backend)
    clock_values = [0.0, 0.0, 0.2e-3, 0.2e-3, 0.4e-3, 0.4e-3]
    clock = 0.4e-3
    for _ in range(5):
        clock_values.extend((clock, clock + 1.2e-3, clock + 1.2e-3, clock + 3.2e-3))
        clock += 3.2e-3
    clock_values.extend((clock, clock + 0.1e-3))
    clock_iterator = iter(clock_values)
    monkeypatch.setattr(
        "tools.performance_report.runner.time.perf_counter",
        lambda: next(clock_iterator),
    )
    result = _run_report_benchmark(
        runtime,
        (((1.0, 0.0, 0.0, 1.0),),),
        execution_mode=execution_mode,
        benchmark_config=SimpleNamespace(
            target_runtime=5.0e-3,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=5,
            precision=16,
        ),
        selectors={"helicities": None, "color_flows": ("flow:1",)},
    )

    assert result.sample_count == 5
    assert result.evaluator_time_per_point is None
    assert result.wall_time_per_point == pytest.approx(6.0e-4)
    assert result.evaluator_total_time_per_point == pytest.approx(5.0e-4)
    assert result.wall_time_per_point != result.evaluator_total_time_per_point
    assert result.environment["elapsed_seconds"] == pytest.approx(6.0e-3)
    assert result.environment["evaluator_total_accumulated_seconds"] == pytest.approx(
        5.0e-3
    )
    assert result.environment["measured_point_count"] == 10
    assert result.environment["warmup_elapsed_seconds"] == pytest.approx(0.2e-3)
    assert result.environment["calibration_elapsed_seconds"] == pytest.approx(0.2e-3)
    assert result.environment["calibration_outer_elapsed_seconds"] == pytest.approx(
        0.2e-3
    )
    assert result.environment["measurement_phase_elapsed_seconds"] == pytest.approx(
        16.0e-3
    )
    assert result.environment["profile_total_elapsed_seconds"] == pytest.approx(16.5e-3)
    measurement = _benchmark_measurement(result, matrix_element=2.0)
    total_timing = measurement["evaluator_total_timing"]
    assert isinstance(total_timing, dict)
    assert total_timing["execution_mode"] == execution_mode.value
    assert total_timing["raw_seconds_per_point"] == pytest.approx(5.0e-4)
    assert total_timing["accumulated_seconds"] == pytest.approx(5.0e-3)
    assert total_timing["measured_point_count"] == 10
    assert result.arena_profile_evidence["profile_count"] == 5
    assert backend.public_profile_calls == 0
    assert len(backend.arena_calls) == 6
    for wall_call, arena_call in zip(
        backend.wall_calls[-5:],
        backend.arena_calls[-5:],
        strict=True,
    ):
        wall_batch, wall_repetitions, wall_kwargs = wall_call
        arena_batch, arena_repetitions, arena_kwargs = arena_call
        assert arena_batch == wall_batch
        assert arena_repetitions == wall_repetitions
        assert arena_kwargs == {**wall_kwargs, "include_values": False}


def test_report_arena_benchmark_recovers_from_busy_floor_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        value = 0.0

        def advance(self, duration: float) -> None:
            self.value += duration

    class Runtime:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock
            self.wall_repetitions: list[int] = []
            self.profile_repetitions: list[int] = []

        def _benchmark_f64_wall_time(
            self,
            batch: object,
            repetitions: int,
            **_kwargs: object,
        ) -> float:
            assert isinstance(batch, tuple)
            self.wall_repetitions.append(repetitions)
            duration = (
                3.753739778
                if len(self.wall_repetitions) == 1
                else 0.0181 * repetitions
            )
            self.clock.advance(duration)
            return duration

        def _profile_arena_repeated(
            self,
            batch: object,
            repetitions: int,
            **_kwargs: object,
        ) -> dict[str, object]:
            assert isinstance(batch, tuple)
            self.profile_repetitions.append(repetitions)
            duration = 1.0e-3
            self.clock.advance(duration)
            return _raw_arena_profile(
                execution_mode="compiled",
                points=len(batch) * repetitions,
                wall_time=duration,
            )

    clock = Clock()
    runtime = Runtime(clock)
    guarded_estimates: list[float | None] = []
    monkeypatch.setattr(
        "tools.performance_report.runner.time.perf_counter",
        lambda: clock.value,
    )

    result = _run_arena_benchmark(
        runtime,
        (((1.0, 0.0, 0.0, 1.0),),),
        execution_mode="compiled",
        benchmark_config=SimpleNamespace(
            target_runtime=5.0,
            batch_size=2,
            warmup_runs=0,
            minimum_samples=5,
            precision=16,
        ),
        selectors={"helicities": None, "color_flows": None},
        chunk_guard=lambda estimate, _label: guarded_estimates.append(estimate),
    )

    assert result.sample_count == 5
    assert result.environment["elapsed_seconds"] == pytest.approx(5.068)
    assert result.environment["repetitions_per_sample"] == 56
    assert result.environment["calibration"]["blocks"] == [
        {"repetitions": 1, "duration_seconds": 3.753739778},
        {"repetitions": 1, "duration_seconds": 0.0181},
        {"repetitions": 56, "duration_seconds": pytest.approx(1.0136)},
    ]
    assert runtime.wall_repetitions == [1, 1, 56, 56, 56, 56, 56, 56]
    assert runtime.profile_repetitions == [56] * 5
    assert guarded_estimates[:3] == [
        None,
        pytest.approx(3.753739778),
        pytest.approx(1.0136),
    ]


def test_report_arena_calibration_confirms_genuinely_slow_floor_once() -> None:
    calls: list[int] = []

    def timer(
        _batch: object,
        repetitions: int,
        **_arguments: object,
    ) -> float:
        calls.append(repetitions)
        return 2.0

    repetitions, blocks = _calibrate_arena_repetitions(
        timer,
        (((1.0, 0.0, 0.0, 1.0),),),
        target_runtime=5.0,
        sample_count=5,
        selector_arguments={},
    )

    assert repetitions == 1
    assert calls == [1, 1]
    assert blocks == [
        {"repetitions": 1, "duration_seconds": 2.0},
        {"repetitions": 1, "duration_seconds": 2.0},
    ]


def test_report_arena_benchmark_requires_private_profiler() -> None:
    runtime = SimpleNamespace(
        _benchmark_f64_wall_time=lambda *_args, **_kwargs: 1.0,
        profile_repeated=lambda *_args, **_kwargs: {},
    )

    with pytest.raises(RunnerError, match=r"_profile_arena_repeated"):
        _run_arena_benchmark(
            runtime,
            (((1.0, 0.0, 0.0, 1.0),),),
            execution_mode="compiled",
            benchmark_config=SimpleNamespace(
                target_runtime=5.0,
                batch_size=2,
                warmup_runs=1,
                minimum_samples=5,
                precision=16,
            ),
            selectors={"helicities": None, "color_flows": None},
        )


def test_report_arena_benchmark_reports_typed_profile_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.reporting import (
        CallbackProgressSink,
        ProgressEnd,
        ProgressStart,
        ProgressUpdate,
    )

    class Runtime:
        def _benchmark_f64_wall_time(
            self,
            batch: object,
            repetitions: int,
            **_kwargs: object,
        ) -> float:
            del batch, repetitions
            return 1.0e-3

        def _profile_arena_repeated(
            self,
            batch: object,
            repetitions: int,
            **_kwargs: object,
        ) -> dict[str, object]:
            assert isinstance(batch, tuple)
            return _raw_arena_profile(
                points=len(batch) * repetitions,
                wall_time=2.0e-3,
            )

    clock = -1.0e-3

    def tick() -> float:
        nonlocal clock
        clock += 1.0e-3
        return clock

    monkeypatch.setattr(
        "tools.performance_report.runner.time.perf_counter",
        tick,
    )
    events: list[ProgressStart | ProgressUpdate | ProgressEnd] = []
    _run_arena_benchmark(
        Runtime(),
        (((1.0, 0.0, 0.0, 1.0),),),
        execution_mode="compiled",
        benchmark_config=SimpleNamespace(
            target_runtime=4.0e-3,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=5,
            precision=16,
        ),
        selectors={"helicities": None, "color_flows": None},
        progress=CallbackProgressSink(events.append),
    )

    starts = {event.task_id for event in events if isinstance(event, ProgressStart)}
    updates = {event.task_id for event in events if isinstance(event, ProgressUpdate)}
    assert starts == {
        "report-profile",
        "report-profile:warmup",
        "report-profile:calibration",
        "report-profile:samples",
    }
    assert updates == {
        "report-profile:warmup",
        "report-profile:calibration",
        "report-profile:samples",
    }
    assert isinstance(events[-1], ProgressEnd)
    assert events[-1].task_id == "report-profile"
    assert events[-1].success is True


def test_report_benchmark_keeps_recurrence_on_supported_public_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.cli

    calls: list[tuple[object, object]] = []
    parsed_argv: list[tuple[str, ...]] = []
    real_parse_cli = pyamplicol.cli.parse_cli

    def parse_cli(argv: tuple[str, ...]):
        parsed_argv.append(tuple(argv))
        return real_parse_cli(argv)

    @dataclass(frozen=True)
    class Result:
        environment: dict[str, object]

    expected = Result(environment={"existing": "evidence"})

    class BenchmarkRunner:
        def __init__(
            self,
            config: object,
            *,
            progress: object | None = None,
        ) -> None:
            self.config = config
            assert progress is not None

        def run(self, runtime: object, *, points: object) -> object:
            calls.append((runtime, points))
            return expected

    monkeypatch.setattr("pyamplicol.api.BenchmarkRunner", BenchmarkRunner)
    monkeypatch.setattr(pyamplicol.cli, "parse_cli", parse_cli)
    runtime = SimpleNamespace(_backend=SimpleNamespace())
    points = (((1.0, 0.0, 0.0, 1.0),),)
    from pyamplicol.config import BenchmarkConfig

    result = _run_report_benchmark(
        runtime,
        points,
        execution_mode=ExecutionMode.RECURRENCE,
        benchmark_config=BenchmarkConfig(),
        selectors={"helicities": None, "color_flows": None},
    )

    assert result is not expected
    assert result.environment["profile_total_elapsed_seconds"] >= 0.0
    observed_environment = dict(result.environment)
    observed_environment.pop("profile_total_elapsed_seconds")
    assert observed_environment == {
        "existing": "evidence",
        "report_command_path": LOADED_RUNTIME_PROFILE_COMMAND_PATH,
        "report_public_cli_path": PUBLIC_CLI_COMMAND_PATH,
    }
    assert calls == [(runtime, points)]
    assert len(parsed_argv) == 1
    assert parsed_argv[0][0] == "profile"
    assert "--target-runtime" in parsed_argv[0]


@pytest.mark.parametrize(
    "field",
    (
        "native_input_container_allocation_count",
        "native_input_pack_bytes",
        "stage_evaluator_output_gather_component_count",
        "selector_scatter_value_count",
        "amplitude_output_remap_component_count",
    ),
)
def test_arena_profile_evidence_rejects_warmed_boundary_traffic(
    field: str,
) -> None:
    profile = _raw_arena_profile()
    profile[field] = 1

    with pytest.raises(ArenaProfileEvidenceError, match=field):
        build_arena_profile_evidence(
            (profile,),
            execution_mode="compiled",
            repetitions_per_profile=1,
            batch_size=128,
        )


@pytest.mark.parametrize(
    "field",
    (
        "compiled_direct_arena_engine_count",
        "compiled_direct_arena_call_count",
        "evaluator_backend_call_count",
    ),
)
def test_arena_profile_evidence_keeps_compiled_activity_fail_closed(
    field: str,
) -> None:
    profile = _raw_arena_profile()
    profile[field] = 0

    with pytest.raises(ArenaProfileEvidenceError, match="activity counters"):
        build_arena_profile_evidence(
            (profile,),
            execution_mode="compiled",
            repetitions_per_profile=1,
            batch_size=128,
        )


def test_arena_profile_evidence_recomputes_all_native_counters() -> None:
    evidence = build_arena_profile_evidence(
        (_raw_arena_profile(),) * 5,
        execution_mode="compiled",
        repetitions_per_profile=1,
        batch_size=128,
    )

    assert evidence["counter_totals"]["points"] == 640
    assert evidence["counter_totals"]["state_component_count"] == 55
    assert evidence["counter_totals"]["source_component_count"] == 35

    evidence["counter_totals"]["state_component_count"] = 54
    with pytest.raises(
        ArenaProfileEvidenceError,
        match="independently recomputed",
    ):
        validate_arena_profile_evidence(
            evidence,
            execution_mode="compiled",
            sample_count=5,
            native_profile_points_per_sample=128,
        )


def test_arena_profile_evidence_rejects_recurrence_protocol_claim() -> None:
    with pytest.raises(ArenaProfileEvidenceError, match="unsupported"):
        build_arena_profile_evidence(
            (_raw_arena_profile(execution_mode="recurrence"),),
            execution_mode="recurrence",
            repetitions_per_profile=1,
            batch_size=128,
        )


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _binding_digest(binding: dict[str, object]) -> str:
    if binding["kind"] != "rusticol-intrinsic":
        payload = dict(binding)
        payload.pop("payload_digest", None)
        return _digest_json(payload)
    fields = (
        "abi",
        "contribution_parent_permutation",
        "kind",
        "runtime_template",
    )
    return _digest_json({field: binding.get(field) for field in fields})


def _refresh_recurrence_catalog(
    execution: dict[str, object],
    pack: dict[str, object],
    *,
    refresh_binding_digests: bool = True,
) -> None:
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        binding = template["payload_binding"]
        assert isinstance(binding, dict)
        if refresh_binding_digests:
            binding["payload_digest"] = _binding_digest(binding)
        semantic = dict(template)
        semantic.pop("semantic_digest", None)
        template["semantic_digest"] = _digest_json(semantic)
    semantic_catalog = dict(catalog)
    semantic_catalog.pop("catalog_digest", None)
    catalog["catalog_digest"] = _digest_json(semantic_catalog)
    execution["direct_template_catalog_digest"] = catalog["catalog_digest"]
    plan = execution["plan"]
    assert isinstance(plan, dict)
    plan["direct_template_catalog_digest"] = catalog["catalog_digest"]


def _canonical_symjit_evaluator(
    application_path: str,
    plane_path: str,
    *,
    optimization_level: int,
    input_len: int = 1,
    output_len: int = 1,
    source_digest: str = "1" * 64,
) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": "symjit.application.complex-f64.v1",
        "input_len": input_len,
        "output_len": output_len,
        "application_path": application_path,
        "application_abi": "symjit-application-storage-v3",
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": f"{application_path}.state",
        "evaluator_state_runtime_capability": (
            "symbolica.legacy-jit-container.complex-f64.v1"
        ),
        "plane_application": {
            "application_path": plane_path,
            "application_abi": "pyamplicol-symjit-plane-application-v2",
            "storage_abi": "symjit-application-storage-v3",
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": input_len,
            "output_complex_count": output_len,
            "input_plane_count": 2 * input_len,
            "output_plane_count": 2 * output_len,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": optimization_level,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "compression": False,
            "threading": False,
            "direct_arena": True,
            "source_digest": source_digest,
            "target": {"word_bits": 64, "endianness": "little"},
        },
    }


@pytest.mark.parametrize("optimization_level", (1, 3))
@pytest.mark.parametrize("chunked", (False, True))
def test_canonical_symjit_serializer_round_trips_into_report_authentication(
    optimization_level: int,
    chunked: bool,
) -> None:
    raw = _canonical_symjit_evaluator(
        "evaluators/left.symjit",
        "evaluators/left.plane.symjit",
        optimization_level=optimization_level,
    )
    raw["backend"] = "jit"
    if chunked:
        right = _canonical_symjit_evaluator(
            "evaluators/right.symjit",
            "evaluators/right.plane.symjit",
            optimization_level=optimization_level,
        )
        right["backend"] = "jit"
        serialized = _serialize_evaluator(
            {
                "kind": "chunked-symbolica-evaluator",
                "input_len": 2,
                "chunk_input_indices": [[0], [1]],
                "required_runtime_capabilities": ["symjit.application.complex-f64.v1"],
                "chunks": [raw, right],
            }
        )
    else:
        serialized = _serialize_evaluator(raw)

    leaves = _authenticated_symjit_plane_leaves(
        serialized,
        source_optimization_level=optimization_level,
        context="serialized evaluator",
    )

    assert len(leaves) == (2 if chunked else 1)
    for leaf in serialized["chunks"] if chunked else [serialized]:
        assert "backend" not in leaf
        assert leaf["compiler_type"] == "native"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "native"),
        ("backend", None),
        ("compiler_type", "interpreter"),
    ),
)
def test_symjit_report_authentication_rejects_contradictory_identity(
    field: str,
    value: object,
) -> None:
    evaluator = _canonical_symjit_evaluator(
        "evaluators/stage.symjit",
        "evaluators/stage.plane.symjit",
        optimization_level=3,
    )
    evaluator[field] = value

    with pytest.raises(RunnerError, match="leaf contract drifted"):
        _authenticated_symjit_plane_leaves(
            evaluator,
            source_optimization_level=3,
            context="corrupt evaluator",
        )


def test_symjit_report_authentication_accepts_fallback_free_portable_leaf() -> None:
    evaluator = _canonical_symjit_evaluator(
        "evaluators/stage.symjit",
        "evaluators/stage.plane.symjit",
        optimization_level=3,
    )
    evaluator["evaluator_state_path"] = None
    evaluator["evaluator_state_runtime_capability"] = None

    assert _authenticated_symjit_plane_leaves(
        evaluator,
        source_optimization_level=3,
        context="portable evaluator",
    ) == [("evaluators/stage.plane.symjit", "1" * 64)]


@pytest.mark.parametrize(
    ("state_path", "capability"),
    (
        (None, "symbolica.legacy-jit-container.complex-f64.v1"),
        ("evaluators/stage.state", None),
    ),
)
def test_symjit_report_authentication_rejects_partial_fallback_pair(
    state_path: str | None,
    capability: str | None,
) -> None:
    evaluator = _canonical_symjit_evaluator(
        "evaluators/stage.symjit",
        "evaluators/stage.plane.symjit",
        optimization_level=3,
    )
    evaluator["evaluator_state_path"] = state_path
    evaluator["evaluator_state_runtime_capability"] = capability

    with pytest.raises(RunnerError, match="fallback contract drifted"):
        _authenticated_symjit_plane_leaves(
            evaluator,
            source_optimization_level=3,
            context="partial fallback evaluator",
        )


@pytest.mark.parametrize(
    "target",
    (
        {"word_bits": 64.0, "endianness": "little"},
        {
            "word_bits": 64,
            "endianness": "little",
            "cpu_features": ["z", "a"],
        },
        {"word_bits": 64, "endianness": "little", "unknown": True},
    ),
)
def test_symjit_report_authentication_rejects_noncanonical_target(
    target: dict[str, object],
) -> None:
    evaluator = _canonical_symjit_evaluator(
        "evaluators/stage.symjit",
        "evaluators/stage.plane.symjit",
        optimization_level=3,
    )
    plane = evaluator["plane_application"]
    assert isinstance(plane, dict)
    plane["target"] = target

    with pytest.raises(RunnerError, match="plane application contract drifted"):
        _authenticated_symjit_plane_leaves(
            evaluator,
            source_optimization_level=3,
            context="noncanonical target evaluator",
        )


def _recurrence_source_fixture(
    source_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    ordinary_source_path = "kernels/000000/application-0.symjit"
    source_path = "kernels/000000/application-0.plane.symjit"
    prepared_binding: dict[str, object] = {
        "abi": "pyamplicol-recurrence-plane-binding-v2",
        "contribution_parent_permutation": [0, 1],
        "destination_operation": "finalize-in-place",
        "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
        "exact_factor_scalar_slots": [0, 1],
        "input_plane_count": 1,
        "input_plane_projections": [{"kind": "destination-current"}],
        "intrinsic_contract_digest": None,
        "kind": "prepared-direct-call",
        "output_alias_inputs": [0],
        "parameter_bindings": [{"index": 0, "kind": "plane"}],
        "payload_digest": "",
        "payload_paths": [source_path],
        "prepared_kernel_id": 0,
        "prepared_template_semantic_digest": "d" * 64,
        "role": "finalization",
        "runtime_template": None,
        "scalar_input_count": 2,
        "scalar_projections": [
            {"imaginary": False, "kind": "exact-factor"},
            {"imaginary": True, "kind": "exact-factor"},
        ],
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
        "source_application_path": source_path,
        "source_application_sha256": source_sha256,
        "state_plane_indices": [],
    }
    intrinsic_binding: dict[str, object] = {
        "abi": "pyamplicol-recurrence-plane-binding-v2",
        "contribution_parent_permutation": [0, 1],
        "destination_operation": None,
        "direct_application_abi": None,
        "exact_factor_scalar_slots": [],
        "input_plane_count": 0,
        "input_plane_projections": [],
        "intrinsic_contract_digest": None,
        "kind": "rusticol-intrinsic",
        "output_alias_inputs": [],
        "parameter_bindings": [],
        "payload_digest": "",
        "payload_paths": [],
        "prepared_kernel_id": None,
        "prepared_template_semantic_digest": None,
        "role": None,
        "runtime_template": "source-current",
        "scalar_input_count": 0,
        "scalar_projections": [],
        "source_application_abi": None,
        "source_application_path": None,
        "source_application_sha256": None,
        "state_plane_indices": [],
    }
    templates: list[dict[str, object]] = [
        {
            "abi": "pyamplicol-recurrence-direct-template-v1",
            "alignment_bytes": 64,
            "backend": "jit",
            "coupling_slot_count": 0,
            "destination_aliasing": True,
            "destination_component_count": 1,
            "destination_operation": "finalize-in-place",
            "direct_executor_id": 0,
            "evaluator_binding_id": 0,
            "evaluator_resolver_key": "prepared-kernel-0",
            "exact_expression_digest": "e" * 64,
            "momentum_operand_count": 0,
            "optimization_level": 2,
            "parameter_slot_count": 0,
            "parent_arity": 1,
            "parent_component_counts": [1],
            "payload_binding": prepared_binding,
            "portable": True,
            "role": "finalization",
            "semantic_digest": "",
            "semantic_template_ids": ["prepared-finalization"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "template-0",
        },
        {
            "abi": "pyamplicol-recurrence-direct-template-v1",
            "alignment_bytes": 64,
            "backend": "jit",
            "coupling_slot_count": 0,
            "destination_aliasing": False,
            "destination_component_count": 1,
            "destination_operation": "initialize",
            "direct_executor_id": 1,
            "evaluator_binding_id": 1,
            "evaluator_resolver_key": "source-current",
            "exact_expression_digest": "f" * 64,
            "momentum_operand_count": 0,
            "optimization_level": 2,
            "parameter_slot_count": 0,
            "parent_arity": 0,
            "parent_component_counts": [],
            "payload_binding": intrinsic_binding,
            "portable": True,
            "role": "source",
            "semantic_digest": "",
            "semantic_template_ids": ["source-current"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "template-1",
        },
    ]
    prepared_digest = "a" * 64
    catalog: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-template-v1",
        "backend": "jit",
        "backend_abi": "rusticol.recurrence-direct-backend.v1",
        "canonicalization_abi": "pyamplicol-canonical-json-v1",
        "catalog_digest": "",
        "compiled_model_digest": "b" * 64,
        "optimization_level": 2,
        "optimization_settings_digest": "c" * 64,
        "portable": True,
        "prepared_kernel_contract_digest": "4" * 64,
        "prepared_kernel_pack_digest": prepared_digest,
        "prepared_kernel_payload_digest": "5" * 64,
        "recurrence_template_catalog_digest": "6" * 64,
        "target_triple": "symjit-storage-v3-portable",
        "templates": templates,
    }
    execution: dict[str, object] = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "plan": {
            "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
            "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
            "prepared_kernel_pack_digest": prepared_digest,
            "direct_template_catalog_digest": "",
        },
    }
    pack: dict[str, object] = {
        "backend": "jit",
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "recurrence_direct_template": catalog,
        "kernel_variants": [],
        "kernels": [
            {
                "kernel_id": 0,
                "f64_evaluator_manifest": _canonical_symjit_evaluator(
                    ordinary_source_path,
                    source_path,
                    optimization_level=2,
                ),
            }
        ],
    }
    _refresh_recurrence_catalog(execution, pack)
    return execution, pack


def _write_json_payload(
    root: Path,
    relative: str,
    value: object,
    *,
    process_id: str | None,
) -> SimpleNamespace:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return SimpleNamespace(
        path=relative,
        role="evaluator-manifest",
        media_type="application/json",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        executable=False,
        process_id=process_id,
    )


def _runner_recurrence_artifact(
    root: Path,
) -> tuple[
    SimpleNamespace,
    dict[str, object],
    dict[str, object],
    SimpleNamespace,
    SimpleNamespace,
]:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_relative = f"processes/{process_id}/execution.json"
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.plane.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": f"{process_id}/execution.json",
            }
        ],
    }
    index_payload = _write_json_payload(
        root,
        index_relative,
        index,
        process_id=None,
    )
    execution_payload = _write_json_payload(
        root,
        execution_relative,
        execution,
        process_id=process_id,
    )
    pack_payload = _write_json_payload(
        root,
        pack_relative,
        pack,
        process_id=None,
    )
    source_path = root / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_data)
    source_payload = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        size_bytes=len(source_data),
        sha256=source_sha256,
        executable=False,
        process_id=None,
    )
    manifest = SimpleNamespace(
        root=root,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(
            index_payload,
            execution_payload,
            pack_payload,
            source_payload,
        ),
        extensions={},
    )
    return manifest, execution, pack, execution_payload, pack_payload


def _rewrite_json_payload(
    root: Path,
    record: SimpleNamespace,
    value: object,
) -> None:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (root / record.path).write_bytes(data)
    record.size_bytes = len(data)
    record.sha256 = hashlib.sha256(data).hexdigest()


def test_checked_runtime_identity_rejects_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW is unavailable")
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises(RunnerError, match="checked fd"):
        _regular_file_identity(link)


def test_effective_config_is_read_from_its_authenticated_artifact_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative = "config/effective.toml"
    data = b'[evaluator]\nbackend = "jit"\nexecution_mode = "compiled"\n'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    manifest = SimpleNamespace(
        root=tmp_path,
        configuration={"effective_path": relative},
        payloads=(
            SimpleNamespace(
                path=relative,
                role="configuration-effective",
                media_type="application/toml",
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                executable=False,
                process_id=None,
            ),
        ),
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _path, *, verify_payloads: manifest,
    )

    assert _authenticated_effective_config(tmp_path) == {
        "evaluator": {
            "backend": "jit",
            "execution_mode": "compiled",
        }
    }

    path.write_bytes(data.replace(b"compiled", b"eager   "))
    with pytest.raises(RunnerError, match="artifact manifest"):
        _authenticated_effective_config(tmp_path)


@dataclass(frozen=True)
class Flow:
    id: str
    word: tuple[int, ...]


@dataclass(frozen=True)
class Helicity:
    id: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class Particle:
    label: int
    pdg_id: int = 1


class Resolved:
    def __init__(self, values: object, totals: tuple[complex, ...]) -> None:
        self.values = values
        self._totals = totals

    def total(self) -> tuple[complex, ...]:
        return self._totals


class FakeRuntime:
    def __init__(self) -> None:
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            external_particles=(Particle(1), Particle(2), Particle(3)),
            helicities=(
                Helicity("h:-1,-1,-1", (-1, -1, -1)),
                Helicity("h:-1,+1,-1", (-1, 1, -1)),
            ),
            color_flows=(
                Flow("flow:2,1,3", (2, 1, 3)),
                Flow("flow:1,2,3", (1, 2, 3)),
            ),
        )
        self.optimized = (3.0 + 0.0j,)
        self.resolved_total = (3.0 + 0.0j,)

    def evaluate(self, _points: object, **_selectors: object) -> tuple[complex, ...]:
        return self.optimized

    def evaluate_resolved(
        self,
        _points: object,
        **_selectors: object,
    ) -> Resolved:
        return Resolved(
            (
                (
                    (0.0 + 0.0j,),
                    (3.0 + 0.0j,),
                ),
            ),
            self.resolved_total,
        )


def _cell(
    mode: ExecutionMode,
    accuracy: Accuracy,
    workload: Workload,
):
    return next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is mode
        and cell.measurement.accuracy is accuracy
        and cell.workload is workload
    )


def test_generation_phase_watchdog_covers_preparation_and_generator_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.api
    import pyamplicol.cli
    import tools.performance_report.runner as report_runner

    events: list[str] = []
    parsed_argv: list[tuple[str, ...]] = []
    real_parse_cli = pyamplicol.cli.parse_cli

    def parse_cli(argv: tuple[str, ...]):
        parsed_argv.append(tuple(argv))
        return real_parse_cli(argv)

    class FakeSource:
        def compile(self, **_kwargs: object) -> object:
            events.append("model-preparation")
            return object()

    class FakeModelSource:
        @staticmethod
        def built_in_sm() -> FakeSource:
            return FakeSource()

        @staticmethod
        def from_path(_path: Path) -> FakeSource:
            return FakeSource()

    class FakeGenerator:
        def __init__(
            self,
            _resolution: object,
            *,
            progress: object | None = None,
        ) -> None:
            assert progress is not None

        def generate(self, *_args: object, **_kwargs: object) -> None:
            events.append("Generator.generate")

    class SpyReporter:
        @contextmanager
        def generation(self):
            events.append("phase-enter")
            try:
                yield
            finally:
                events.append("phase-exit")

    monkeypatch.setattr(pyamplicol.api, "Generator", FakeGenerator)
    monkeypatch.setattr(pyamplicol.api, "ModelSource", FakeModelSource)
    monkeypatch.setattr(pyamplicol.cli, "parse_cli", parse_cli)
    monkeypatch.setattr(
        report_runner,
        "_authenticated_effective_config",
        lambda _path: events.append("post-generation-validation") or {},
    )
    monkeypatch.setattr(
        report_runner,
        "_single_process_id",
        lambda _path, fallback: fallback,
    )
    monkeypatch.setattr(
        report_runner,
        "_artifact_numerical_relation_metadata",
        lambda _path, _process_id: (
            {
                "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                "state": "no-applied-relations",
                "applied_relation_count": 0,
            },
            None,
        ),
    )

    generated = generate_artifact(
        _cell(ExecutionMode.COMPILED, Accuracy.LC, Workload.SELECTED_FLOW),
        tmp_path / "artifact",
        settings=RunnerSettings(),
        repo_root=tmp_path,
        phase_reporter=SpyReporter(),  # type: ignore[arg-type]
    )

    assert events == [
        "phase-enter",
        "model-preparation",
        "Generator.generate",
        "phase-exit",
        "post-generation-validation",
    ]
    assert len(parsed_argv) == 1
    assert parsed_argv[0][0] == "generate"
    assert generated.generation_command_path == PRECOMPILED_GENERATION_COMMAND_PATH


@pytest.mark.parametrize(
    "mode",
    (ExecutionMode.EAGER, ExecutionMode.RECURRENCE),
)
def test_prepared_generation_uses_public_cli_parser_and_default_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: ExecutionMode,
) -> None:
    import pyamplicol.api
    import pyamplicol.cli
    import pyamplicol.cli.handlers
    import tools.performance_report.runner as report_runner

    parsed_argv: list[tuple[str, ...]] = []
    handler_configs: list[object] = []
    generated_models: list[object] = []
    real_parse_cli = pyamplicol.cli.parse_cli
    real_generate = pyamplicol.cli.handlers.DefaultCliServices.generate

    def parse_cli(argv: tuple[str, ...]):
        parsed_argv.append(tuple(argv))
        return real_parse_cli(argv)

    def public_generate(
        self: object,
        config: object,
        progress: object,
    ) -> object:
        handler_configs.append(config)
        return real_generate(self, config, progress)  # type: ignore[arg-type]

    class FakeSource:
        pass

    class FakeModelSource:
        @staticmethod
        def built_in_sm() -> FakeSource:
            return FakeSource()

        @staticmethod
        def from_config(_config: object) -> FakeSource:
            return FakeSource()

    class FakeGenerator:
        def __init__(
            self,
            *,
            config: object,
            progress: object | None = None,
        ) -> None:
            assert progress is not None
            self.config = config

        def generate(
            self,
            _processes: object,
            _output: Path,
            *,
            model: object,
            mode: str,
        ) -> None:
            assert mode == "replace"
            generated_models.append(model)

    monkeypatch.setattr(pyamplicol.api, "Generator", FakeGenerator)
    monkeypatch.setattr(pyamplicol.api, "ModelSource", FakeModelSource)
    monkeypatch.setattr(pyamplicol.cli, "parse_cli", parse_cli)
    monkeypatch.setattr(
        pyamplicol.cli.handlers.DefaultCliServices,
        "generate",
        public_generate,
    )
    monkeypatch.setattr(
        report_runner,
        "_authenticated_effective_config",
        lambda _path: {},
    )
    monkeypatch.setattr(
        report_runner,
        "_single_process_id",
        lambda _path, fallback: fallback,
    )
    monkeypatch.setattr(
        report_runner,
        "_artifact_numerical_relation_metadata",
        lambda _path, _process_id: (
            {
                "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                "state": "no-applied-relations",
                "applied_relation_count": 0,
            },
            None,
        ),
    )
    cell = REPORT_CATALOG.cell(
        f"matrix-{mode.value}-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )

    generated = generate_artifact(
        cell,
        tmp_path / "artifact",
        settings=RunnerSettings(),
        repo_root=tmp_path,
    )

    assert len(parsed_argv) == 1
    assert parsed_argv[0][:2] == ("generate", cell.process)
    assert len(handler_configs) == 1
    assert len(generated_models) == 1
    assert generated.generation_command_path == PUBLIC_CLI_COMMAND_PATH


@pytest.mark.parametrize(
    ("mode", "workload", "expected_layout", "expected_level"),
    (
        (
            ExecutionMode.RECURRENCE,
            Workload.SELECTED_FLOW,
            "topology-replay",
            2,
        ),
        (
            ExecutionMode.RECURRENCE,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
        (
            ExecutionMode.COMPILED,
            Workload.SELECTED_FLOW,
            "topology-replay",
            3,
        ),
        (
            ExecutionMode.EAGER,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
    ),
)
def test_config_steers_complete_coverage_and_layout_only(
    mode: ExecutionMode,
    workload: Workload,
    expected_layout: str,
    expected_level: int,
) -> None:
    cell = _cell(mode, Accuracy.LC, workload)
    values = config_values(
        cell,
        RunnerSettings(worker_cores=1),
        repo_root=Path("/repo"),
    )

    assert values["color"]["lc_flow_layout"] == expected_layout  # type: ignore[index]
    assert values["evaluator"]["execution_mode"] == mode.value  # type: ignore[index]
    assert (
        values["evaluator"]["jit"]["optimization_level"]  # type: ignore[index]
        == expected_level
    )
    serialized = repr(values)
    assert "selected_color_sector_ids" not in serialized
    assert "selected_source_helicities" not in serialized
    assert "reference_color_order" not in serialized


def test_config_threads_manual_profile_sampling_settings() -> None:
    values = config_values(
        _cell(
            ExecutionMode.RECURRENCE,
            Accuracy.LC,
            Workload.SELECTED_FLOW,
        ),
        RunnerSettings(warmup_runs=7, minimum_samples=11),
        repo_root=Path("/repo"),
    )

    assert values["benchmark"]["warmup_runs"] == 7  # type: ignore[index]
    assert values["benchmark"]["minimum_samples"] == 11  # type: ignore[index]
    assert values["generation"]["validation"] == {  # type: ignore[index]
        "enabled": True,
        "samples": 1,
        "seed": 12345,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-300,
        "post_build_validation": False,
    }


@pytest.mark.parametrize(
    ("mode", "workspace_key"),
    (
        (ExecutionMode.RECURRENCE, "recurrence"),
        (ExecutionMode.EAGER, "eager"),
    ),
)
def test_config_derives_selected_engine_workspace_from_campaign_memory_cap(
    mode: ExecutionMode,
    workspace_key: str,
) -> None:
    values = config_values(
        _cell(mode, Accuracy.LC, Workload.SELECTED_FLOW),
        RunnerSettings(memory_limit_bytes=15_000_000_000),
        repo_root=Path("/repo"),
    )
    evaluator = values["evaluator"]
    assert isinstance(evaluator, dict)
    assert evaluator[workspace_key] == {"workspace_mib": 14_305}
    assert ({"eager", "recurrence"} - {workspace_key}).isdisjoint(evaluator)

    compiled = config_values(
        _cell(ExecutionMode.COMPILED, Accuracy.LC, Workload.SELECTED_FLOW),
        RunnerSettings(memory_limit_bytes=15_000_000_000),
        repo_root=Path("/repo"),
    )["evaluator"]
    assert isinstance(compiled, dict)
    assert "eager" not in compiled
    assert "recurrence" not in compiled


def test_runner_rejects_sub_mib_workspace_budget() -> None:
    with pytest.raises(ValueError, match="at least one MiB"):
        RunnerSettings(memory_limit_bytes=1024 * 1024 - 1)


def test_nlc_and_full_use_contracted_topology_replay_configuration() -> None:
    for accuracy in (Accuracy.NLC, Accuracy.FULL):
        cell = _cell(ExecutionMode.RECURRENCE, accuracy, Workload.CONTRACTED)
        values = config_values(
            cell,
            RunnerSettings(),
            repo_root=Path("/repo"),
        )
        assert values["color"] == {
            "accuracy": accuracy.value,
            "lc_flow_layout": "topology-replay",
        }


def test_selector_contract_uses_canonical_minimum_flow_and_fixed_helicity() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)

    contract = derive_selector_contract(runtime, points)

    assert contract.selected_color_flow_ids == ("flow:1,2,3",)
    assert contract.selected_color_words == ((1, 2, 3),)
    assert contract.all_flow_helicity_ids == ("h:-1,+1,-1",)
    assert contract.all_flow_source_helicities == ((1, -1), (2, 1), (3, -1))
    assert contract.point_digest == point_digest(points)
    assert SelectorContract.from_mapping(contract.as_dict()) == contract
    validate_selector_contract(runtime, contract, points)


def test_selector_contract_is_independent_of_runtime_flow_enumeration() -> None:
    points = (((1.0, 0.0, 0.0, 1.0),),)
    forward = FakeRuntime()
    reverse = FakeRuntime()
    reverse.physics.color_flows = tuple(reversed(forward.physics.color_flows))

    assert derive_selector_contract(forward, points) == derive_selector_contract(
        reverse,
        points,
    )


def test_selector_contract_rejects_zero_canonical_minimum_flow() -> None:
    class CanonicalMinimumZeroRuntime(FakeRuntime):
        def evaluate_resolved(
            self,
            _points: object,
            **selectors: object,
        ) -> Resolved:
            assert selectors.get("helicities") == ("h:-1,+1,-1",)
            selected_flows = selectors.get("color_flows")
            if selected_flows == ("flow:1,2,3",):
                return Resolved(
                    (
                        (
                            (0.0 + 0.0j,),
                            (0.0 + 0.0j,),
                        ),
                    ),
                    self.resolved_total,
                )
            pytest.fail("selector must not fall back to a noncanonical flow")

    with pytest.raises(
        RunnerError,
        match="canonical fixed-helicity selector is zero",
    ):
        derive_selector_contract(
            CanonicalMinimumZeroRuntime(),
            (((1.0, 0.0, 0.0, 1.0),),),
        )


def test_selector_contract_ignores_larger_noncanonical_flow_component() -> None:
    class CanonicalMinimumRuntime(FakeRuntime):
        def evaluate_resolved(
            self,
            _points: object,
            **selectors: object,
        ) -> Resolved:
            selected_flows = selectors.get("color_flows")
            if selected_flows == ("flow:1,2,3",):
                return Resolved(
                    (
                        (
                            (0.0 + 0.0j,),
                            (1.0e-18 + 0.0j,),
                        ),
                    ),
                    self.resolved_total,
                )
            pytest.fail("selector must not rank noncanonical flow components")

    contract = derive_selector_contract(
        CanonicalMinimumRuntime(),
        (((1.0, 0.0, 0.0, 1.0),),),
    )

    assert contract.selected_color_flow_ids == ("flow:1,2,3",)
    assert contract.all_flow_helicity_ids == ("h:-1,+1,-1",)


def test_selector_contract_rejects_duplicate_physical_flow_words() -> None:
    runtime = FakeRuntime()
    runtime.physics.color_flows = (
        Flow("flow:1,2,3", (1, 2, 3)),
        Flow("flow:1,2,3", (1, 2, 3)),
    )

    with pytest.raises(RunnerError, match="color-flow words must be unique"):
        derive_selector_contract(
            runtime,
            (((1.0, 0.0, 0.0, 1.0),),),
        )


def test_selector_contract_rejects_noncanonical_physical_axis_ids() -> None:
    runtime = FakeRuntime()
    runtime.physics.color_flows = (
        Flow("opaque", (1, 2, 3)),
        Flow("flow:2,1,3", (2, 1, 3)),
    )
    with pytest.raises(RunnerError, match="flow ID does not encode"):
        derive_selector_contract(
            runtime,
            (((1.0, 0.0, 0.0, 1.0),),),
        )

    runtime = FakeRuntime()
    runtime.physics.helicities = (
        Helicity("opaque", (-1, 1, -1)),
        Helicity("h:-1,-1,-1", (-1, -1, -1)),
    )
    with pytest.raises(RunnerError, match="helicity ID does not encode"):
        derive_selector_contract(
            runtime,
            (((1.0, 0.0, 0.0, 1.0),),),
        )


def test_selector_contract_rejects_nonfinite_canonical_component() -> None:
    class NonfiniteRuntime(FakeRuntime):
        def evaluate_resolved(
            self,
            _points: object,
            **selectors: object,
        ) -> Resolved:
            assert selectors == {
                "color_flows": ("flow:1,2,3",),
                "helicities": ("h:-1,+1,-1",),
            }
            return Resolved(
                ((((complex(float("nan"), 0.0),),),)),
                self.resolved_total,
            )

    with pytest.raises(RunnerError, match="non-finite component"):
        derive_selector_contract(
            NonfiniteRuntime(),
            (((1.0, 0.0, 0.0, 1.0),),),
        )


def test_selector_contract_rejects_all_zero_canonical_components() -> None:
    class ZeroRuntime(FakeRuntime):
        def evaluate_resolved(
            self,
            _points: object,
            **_selectors: object,
        ) -> Resolved:
            return Resolved(
                (
                    (
                        (0.0 + 0.0j,),
                        (0.0 + 0.0j,),
                    ),
                ),
                self.resolved_total,
            )

    with pytest.raises(
        RunnerError,
        match="canonical fixed-helicity selector is zero",
    ):
        derive_selector_contract(
            ZeroRuntime(),
            (((1.0, 0.0, 0.0, 1.0),),),
        )


def test_selector_contract_rejects_changed_point_or_axis() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)

    with pytest.raises(RunnerError, match="measurement point differ"):
        validate_selector_contract(
            runtime,
            contract,
            (((2.0, 0.0, 0.0, 2.0),),),
        )

    runtime.physics.color_flows = (Flow("different", (2, 1, 3)),)
    with pytest.raises(RunnerError, match="selected physical flow"):
        validate_selector_contract(runtime, contract, points)


def test_selector_contract_maps_only_exact_signed_zero_runtime_alias() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    runtime.physics.external_particles = tuple(
        Particle(label, 23 if label == 6 else 1)
        for label in range(1, 7)
    )
    runtime.physics.helicities = (
        Helicity(
            "h:-1,+1,-1,+1,-1,+0",
            (-1, 1, -1, 1, -1, 0),
        ),
    )
    contract = SelectorContract(
        selected_color_flow_ids=("flow:2,1,3",),
        selected_color_words=((2, 1, 3),),
        all_flow_helicity_ids=("h:-1,+1,-1,+1,-1,0",),
        all_flow_source_helicities=(
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 0),
        ),
        point_digest=point_digest(points),
    )

    assert contract.runtime_all_flow_helicity_ids == ("h:-1,+1,-1,+1,-1,+0",)
    validate_selector_contract(runtime, contract, points)
    assert _selector_kwargs(
        _cell(ExecutionMode.RECURRENCE, Accuracy.LC, Workload.ALL_FLOW),
        contract,
    ) == {
        "helicities": ("h:-1,+1,-1,+1,-1,+0",),
        "color_flows": None,
    }

    changed = SelectorContract(
        selected_color_flow_ids=contract.selected_color_flow_ids,
        selected_color_words=contract.selected_color_words,
        all_flow_helicity_ids=("h:+1,+1,-1,+1,-1,0",),
        all_flow_source_helicities=contract.all_flow_source_helicities,
        point_digest=contract.point_digest,
    )
    assert changed.runtime_all_flow_helicity_ids == changed.all_flow_helicity_ids
    with pytest.raises(RunnerError, match="selected physical helicity"):
        validate_selector_contract(runtime, changed, points)


def test_runtime_contract_requires_both_lc_selector_axes() -> None:
    runtime = FakeRuntime()
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    validate_runtime_contract(cell, runtime)

    runtime.physics.selector_capabilities = ("helicity",)
    with pytest.raises(RunnerError, match="color_flow"):
        validate_runtime_contract(cell, runtime)


def test_artifact_contract_rejects_generation_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    process = SimpleNamespace(
        execution_mode="recurrence",
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_flow_layout="topology-replay",
    )
    inspection = SimpleNamespace(processes=(process,))
    inspection.runtime_capabilities = (
        "rusticol.recurrence-direct-arena.complex-f64.v1",
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact",
        lambda _path: inspection,
    )
    validate_artifact_contract(cell, Path("/artifact"))

    process.generation_specialized_axes = ("color_flow",)
    with pytest.raises(RunnerError, match="complete runtime coverage"):
        validate_artifact_contract(cell, Path("/artifact"))


@pytest.mark.parametrize(
    ("cell", "capability", "expected_evaluator_abi", "optimization_identity"),
    (
        (
            _cell(
                ExecutionMode.RECURRENCE,
                Accuracy.LC,
                Workload.SELECTED_FLOW,
            ),
            "rusticol.recurrence-direct-arena.complex-f64.v1",
            "pyamplicol-recurrence-runtime-layout-v2",
            {"source_jit_optimization_level": 2},
        ),
        (
            _cell(
                ExecutionMode.EAGER,
                Accuracy.LC,
                Workload.SELECTED_FLOW,
            ),
            "eager-direct-arena-v1",
            "pyamplicol-eager-plane-table-binding-v2",
            {"source_jit_optimization_level": 2},
        ),
        (
            next(
                cell
                for cell in REPORT_CATALOG.z_cells()
                if cell.variant == "jit_o1" and cell.workload is Workload.SELECTED_FLOW
            ),
            "compiled-plane-arena-v1",
            "pyamplicol-compiled-plane-kernel-v2",
            {
                "source_jit_optimization_level": 1,
                "direct_codegen_optimization_level": 1,
            },
        ),
        (
            next(
                cell
                for cell in REPORT_CATALOG.z_cells()
                if cell.variant == "cpp_o3" and cell.workload is Workload.SELECTED_FLOW
            ),
            "compiled-plane-arena-v1",
            "pyamplicol-native-compiled-direct-application-v1",
            {},
        ),
    ),
)
def test_runtime_identity_binds_native_artifact_and_arena(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cell: CellSpec,
    capability: str,
    expected_evaluator_abi: str,
    optimization_identity: dict[str, int],
) -> None:
    import pyamplicol

    artifact_id = "a" * 64
    manifest = SimpleNamespace(
        artifact_id=artifact_id,
        processes=(
            {
                "id": "process-1",
                "required_runtime_capabilities": [capability],
            },
        ),
        runtime={"engine_version": "0.1.0.test"},
    )
    candidate_source_revision = "d" * 40
    measurement_source_revision = "e" * 40
    package_root = tmp_path / "pyamplicol"
    package_root.mkdir()
    package_init = package_root / "__init__.py"
    package_init.write_bytes(b"")
    (package_root / "api.py").write_bytes(b"VALUE=1\n")
    (package_root / "ignored.pyo").write_bytes(b"ignored")
    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    native_path = package_root / "_rusticol.so"
    native_path.write_bytes(b"native")
    native = SimpleNamespace(
        __file__=str(native_path),
        package_version=lambda: pyamplicol.__version__,
        native_build_inputs_sha256=lambda: "b" * 64,
        target_info=lambda: SimpleNamespace(
            triple="aarch64-apple-darwin",
            cpu_features=("neon",),
        ),
    )
    package_records = [
        {
            "root_index": 0,
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (package_init, native_path, package_root / "api.py")
    ]
    bytecode_policy = {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }
    package_tree_identity = {
        "kind": "pyamplicol-python-package-tree-v2",
        "root": str(package_root.resolve()),
        "roots": [str(package_root.resolve())],
        "file_count": 3,
        "total_bytes": sum(record["size"] for record in package_records),
        "sha256": hashlib.sha256(
            json.dumps(
                package_records,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "member_set_stable": True,
        "namespace_bound_to_root_fd": True,
        "bytecode_policy": bytecode_policy,
    }
    native_identity = {
        "path": str(native_path.resolve()),
        "size": native_path.stat().st_size,
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    loaded_origin_policy = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": 1,
        "observations": [
            {
                "module": "pyamplicol",
                "kind": "package-member",
                "root_index": 0,
                "path": "__init__.py",
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
    }
    loaded_origin_policy["observations_sha256"] = _digest_json(
        loaded_origin_policy["observations"]
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _path, *, verify_payloads: manifest,
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.importlib.import_module",
        lambda name: native if name == "pyamplicol._rusticol" else None,
    )
    monkeypatch.setattr(
        "pyamplicol._internal.versions._active_build_info",
        lambda: {
            "schema_version": 1,
            "version": pyamplicol.__version__,
            "candidate_fingerprint": "candidate",
            "source_revision": candidate_source_revision,
            "source_checkout": "/repo",
            "native_build_inputs_sha256": "b" * 64,
            "publishable": False,
        },
    )
    monkeypatch.setattr(pyamplicol, "__file__", str(package_init))
    monkeypatch.setattr(pyamplicol, "__path__", [str(package_root)])
    monkeypatch.setattr(
        "tools.performance_report.runner.established_preimport_runtime_identity",
        lambda: {
            "kind": "pyamplicol-preimport-runtime-identity-v1",
            "python_package_tree": package_tree_identity,
            "native_extension": native_identity,
        },
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.python_package_tree_identity",
        lambda roots: (
            package_tree_identity
            if tuple(roots) == (package_root,)
            else pytest.fail(f"unexpected package roots: {roots}")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.loaded_pyamplicol_origin_policy",
        lambda roots, **kwargs: (
            loaded_origin_policy
            if (
                tuple(roots) == (package_root,)
                and kwargs["native_extension"] == native_path
                and kwargs["expected_package_identity"] == package_tree_identity
                and kwargs["expected_native_identity"] == native_identity
            )
            else pytest.fail("unexpected loaded-origin evidence request")
        ),
    )

    def direct_identity(
        _manifest: object,
        *,
        process_id: str,
        source_optimization_level: int,
    ) -> dict[str, object]:
        assert process_id == "process-1"
        return {
            "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
            "optimization_level": source_optimization_level,
            "source_optimization_level": source_optimization_level,
            "leaf_count": 2,
            "execution_manifest_path": "execution.json",
            "execution_manifest_sha256": "e" * 64,
        }

    monkeypatch.setattr(
        "tools.performance_report.runner._authenticated_direct_codegen_identity",
        direct_identity,
    )

    recurrence_identity = {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 5,
        "prepared_direct_template_count": 3,
        "source_evaluator_leaf_count": 4,
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
        "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
        "prepared_kernel_pack_digest": "7" * 64,
        "direct_template_catalog_digest": "8" * 64,
        "execution_manifest_path": "processes/process-1/execution.json",
        "execution_manifest_sha256": "f" * 64,
        "kernel_pack_path": "model/eager-kernel-pack.json",
        "kernel_pack_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        "tools.performance_report.runner._authenticated_recurrence_source_identity",
        lambda _manifest, *, process_id, source_optimization_level: (
            recurrence_identity
            if process_id == "process-1" and source_optimization_level == 2
            else pytest.fail("unexpected recurrence identity request")
        ),
    )

    identity = runtime_identity_payload(
        cell,
        SimpleNamespace(
            artifact_id=artifact_id,
            execution_mode=cell.measurement.execution_mode.value,
        ),
        Path("/artifact"),
        "process-1",
        expected_source_revision=measurement_source_revision,
    )

    assert identity["artifact_id"] == artifact_id
    assert identity["loaded_artifact_id"] == artifact_id
    assert identity["loaded_execution_mode"] == cell.measurement.execution_mode.value
    assert identity["required_arena_capability"] == capability
    assert identity["expected_evaluator_abi"] == expected_evaluator_abi
    expected_source_abi = (
        "pyamplicol-native-compiled-direct-application-v1"
        if (
            cell.measurement.execution_mode is ExecutionMode.COMPILED
            and cell.measurement.backend != "jit"
        )
        else {
            "jit": "pyamplicol-symjit-plane-application-v2",
            "cpp": "symbolica.compiled-cpp.complex-f64.v1",
            "asm": "symbolica.compiled-asm.complex-f64.v1",
        }[cell.measurement.backend]
    )
    assert identity["expected_source_evaluator_abi"] == expected_source_abi
    assert (
        identity["expected_source_evaluator_runtime_capability"]
        == {
            "jit": "symjit.application.complex-f64.v1",
            "cpp": "symbolica.compiled-cpp.complex-f64.v1",
            "asm": "symbolica.compiled-asm.complex-f64.v1",
        }[cell.measurement.backend]
    )
    assert identity["native_build_inputs_sha256"] == "b" * 64
    assert identity["source_revision"] == measurement_source_revision
    assert (
        identity["candidate_build_identity"]["source_revision"]
        == candidate_source_revision
    )
    package_tree = identity["python_package_tree"]
    assert package_tree == package_tree_identity
    assert identity["loaded_module_origin_policy"] == loaded_origin_policy
    for field in (
        "source_jit_optimization_level",
        "direct_codegen_optimization_level",
    ):
        if field in optimization_identity:
            assert identity[field] == optimization_identity[field]
        else:
            assert field not in identity
    if cell.measurement.execution_mode is ExecutionMode.COMPILED and (
        cell.measurement.backend == "jit"
    ):
        assert identity["direct_codegen_identity"] == {
            "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
            "optimization_level": cell.measurement.jit_optimization_level,
            "source_optimization_level": cell.measurement.jit_optimization_level,
            "leaf_count": 2,
            "execution_manifest_path": "execution.json",
            "execution_manifest_sha256": "e" * 64,
        }
    else:
        assert "direct_codegen_identity" not in identity
    if cell.measurement.execution_mode is ExecutionMode.RECURRENCE:
        assert identity["source_jit_identity"] == recurrence_identity
    else:
        assert "source_jit_identity" not in identity

    with pytest.raises(RunnerError, match="loaded artifact identity"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(
                artifact_id="c" * 64,
                execution_mode=cell.measurement.execution_mode.value,
            ),
            Path("/artifact"),
            "process-1",
            expected_source_revision=measurement_source_revision,
        )

    mismatched_mode = (
        "compiled"
        if cell.measurement.execution_mode is not ExecutionMode.COMPILED
        else "eager"
    )
    with pytest.raises(RunnerError, match="loaded execution mode"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(
                artifact_id=artifact_id,
                execution_mode=mismatched_mode,
            ),
            Path("/artifact"),
            "process-1",
            expected_source_revision=measurement_source_revision,
        )

    with pytest.raises(RunnerError, match=r"does not expose.*execution mode"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(artifact_id=artifact_id),
            Path("/artifact"),
            "process-1",
            expected_source_revision=measurement_source_revision,
        )


def test_direct_codegen_identity_follows_authenticated_process_index(
    tmp_path: Path,
) -> None:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_entry_relative = f"{process_id}/execution.json"
    execution_relative = f"processes/{execution_entry_relative}"
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "amplitude_stage": {
                    "evaluator": _canonical_symjit_evaluator(
                        "evaluators/amplitude.symjit",
                        "evaluators/amplitude.plane.symjit",
                        optimization_level=1,
                    ),
                    "compiled_plane_arena": {
                        "leaves": [
                            {
                                "application_path": (
                                    "evaluators/amplitude.plane.symjit"
                                ),
                                "source_application_abi": (
                                    "pyamplicol-symjit-plane-application-v2"
                                ),
                                "optimization_level": 1,
                                "direct_codegen_optimization_level": 1,
                            }
                        ]
                    },
                }
            }
        },
    }
    evaluator_index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": execution_entry_relative,
                "required_runtime_capabilities": ["compiled-plane-arena-v1"],
            }
        ],
    }

    def payload(
        relative: str,
        value: dict[str, object],
        *,
        payload_process_id: str | None,
    ) -> SimpleNamespace:
        data = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return SimpleNamespace(
            path=relative,
            role="evaluator-manifest",
            media_type="application/json",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            executable=False,
            process_id=payload_process_id,
        )

    index_payload = payload(
        index_relative,
        evaluator_index,
        payload_process_id=None,
    )
    execution_payload = payload(
        execution_relative,
        execution,
        payload_process_id=process_id,
    )
    manifest = SimpleNamespace(
        root=tmp_path,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(index_payload, execution_payload),
    )

    identity = _authenticated_direct_codegen_identity(
        manifest,
        process_id=process_id,
        source_optimization_level=1,
    )

    assert identity == {
        "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
        "optimization_level": 1,
        "source_optimization_level": 1,
        "leaf_count": 1,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_payload.sha256,
    }


def test_recurrence_source_identity_follows_authenticated_direct_template_pack(
    tmp_path: Path,
) -> None:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_relative = f"processes/{process_id}/execution.json"
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.plane.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    evaluator_index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": f"{process_id}/execution.json",
            }
        ],
    }

    def payload(
        relative: str,
        value: dict[str, object],
        *,
        payload_process_id: str | None,
    ) -> SimpleNamespace:
        data = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return SimpleNamespace(
            path=relative,
            role="evaluator-manifest",
            media_type="application/json",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            executable=False,
            process_id=payload_process_id,
        )

    index_payload = payload(
        index_relative,
        evaluator_index,
        payload_process_id=None,
    )
    execution_payload = payload(
        execution_relative,
        execution,
        payload_process_id=process_id,
    )
    pack_payload = payload(
        pack_relative,
        pack,
        payload_process_id=None,
    )
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_data)
    source_payload = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        size_bytes=len(source_data),
        sha256=source_sha256,
        executable=False,
        process_id=None,
    )
    manifest = SimpleNamespace(
        root=tmp_path,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(index_payload, execution_payload, pack_payload, source_payload),
        extensions={},
    )

    identity = _authenticated_recurrence_source_identity(
        manifest,
        process_id=process_id,
        source_optimization_level=2,
    )

    assert identity == {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 2,
        "prepared_direct_template_count": 1,
        "source_evaluator_leaf_count": 1,
        "source_application_abi": "pyamplicol-symjit-plane-application-v2",
        "direct_application_abi": "pyamplicol-symjit-plane-application-v2",
        "prepared_kernel_pack_digest": "a" * 64,
        "direct_template_catalog_digest": execution["direct_template_catalog_digest"],
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_payload.sha256,
        "kernel_pack_path": pack_relative,
        "kernel_pack_sha256": pack_payload.sha256,
    }

    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    template = templates[1]
    assert isinstance(template, dict)
    template["optimization_level"] = 1
    _refresh_recurrence_catalog(execution, pack)
    replacement = json.dumps(
        pack,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / pack_relative).write_bytes(replacement)
    pack_payload.size_bytes = len(replacement)
    pack_payload.sha256 = hashlib.sha256(replacement).hexdigest()
    execution_replacement = json.dumps(
        execution,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / execution_relative).write_bytes(execution_replacement)
    execution_payload.size_bytes = len(execution_replacement)
    execution_payload.sha256 = hashlib.sha256(execution_replacement).hexdigest()
    with pytest.raises(RunnerError, match="template 1 contract"):
        _authenticated_recurrence_source_identity(
            manifest,
            process_id=process_id,
            source_optimization_level=2,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("execution_template_abi", "direct_template_abi"),
        ("plan_template_abi", "direct_template_abi"),
        ("execution_backend_abi", "direct_backend_abi"),
        ("plan_backend_abi", "direct_backend_abi"),
        ("plan_prepared_digest", "source digests"),
        ("plan_catalog_digest", "source digests"),
        ("pack_prepared_digest", "kernel-pack digests"),
        ("execution_catalog_link", "kernel-pack digests"),
        ("catalog_backend_abi", "optimization identity"),
        ("catalog_canonicalization_abi", "optimization identity"),
        ("catalog_digest", "catalog digest"),
        ("template_portable", "template 0 contract"),
        ("template_abi", "template 0 contract"),
        ("binding_abi", "payload-binding contract"),
        ("source_application_abi", "source application"),
        ("direct_application_abi", "source application"),
        ("payload_digest", "payload-binding contract"),
        ("intrinsic_source_application_abi", "intrinsic source contract"),
        ("intrinsic_direct_application_abi", "intrinsic source contract"),
        ("source_leaf_runtime_capability", "source evaluator leaf contract"),
        ("source_leaf_application_abi", "source evaluator leaf contract"),
        ("source_leaf_optimization", "source evaluator leaf contract"),
        ("source_leaf_path", "payload is missing"),
        ("source_payload_digest_link", "not bound to its prepared kernel"),
    ),
)
def test_recurrence_source_identity_rejects_every_broken_link(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    (
        manifest,
        execution,
        pack,
        execution_payload,
        pack_payload,
    ) = _runner_recurrence_artifact(tmp_path)
    plan = execution["plan"]
    assert isinstance(plan, dict)
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    prepared = templates[0]
    intrinsic = templates[1]
    assert isinstance(prepared, dict)
    assert isinstance(intrinsic, dict)
    prepared_binding = prepared["payload_binding"]
    intrinsic_binding = intrinsic["payload_binding"]
    assert isinstance(prepared_binding, dict)
    assert isinstance(intrinsic_binding, dict)
    kernels = pack["kernels"]
    assert isinstance(kernels, list)
    kernel = kernels[0]
    assert isinstance(kernel, dict)
    source_leaf = kernel["f64_evaluator_manifest"]
    assert isinstance(source_leaf, dict)
    refresh_catalog: bool | None = None

    if corruption == "execution_template_abi":
        execution["direct_template_abi"] = "wrong"
    elif corruption == "plan_template_abi":
        plan["direct_template_abi"] = "wrong"
    elif corruption == "execution_backend_abi":
        execution["direct_backend_abi"] = "wrong"
    elif corruption == "plan_backend_abi":
        plan["direct_backend_abi"] = "wrong"
    elif corruption == "plan_prepared_digest":
        plan["prepared_kernel_pack_digest"] = "0" * 64
    elif corruption == "plan_catalog_digest":
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "pack_prepared_digest":
        catalog["prepared_kernel_pack_digest"] = "0" * 64
        refresh_catalog = True
    elif corruption == "execution_catalog_link":
        execution["direct_template_catalog_digest"] = "0" * 64
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "catalog_backend_abi":
        catalog["backend_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_canonicalization_abi":
        catalog["canonicalization_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_digest":
        catalog["compiled_model_digest"] = "0" * 64
    elif corruption == "template_portable":
        prepared["portable"] = False
        refresh_catalog = True
    elif corruption == "template_abi":
        prepared["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "binding_abi":
        prepared_binding["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "source_application_abi":
        prepared_binding["source_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "direct_application_abi":
        prepared_binding["direct_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "payload_digest":
        prepared_binding["payload_digest"] = "0" * 64
        refresh_catalog = False
    elif corruption == "intrinsic_source_application_abi":
        intrinsic_binding["source_application_abi"] = "symjit-application-storage-v3"
        refresh_catalog = True
    elif corruption == "intrinsic_direct_application_abi":
        intrinsic_binding["direct_application_abi"] = (
            "pyamplicol-symjit-plane-application-v0"
        )
        refresh_catalog = True
    elif corruption == "source_leaf_runtime_capability":
        source_leaf["runtime_capability"] = "wrong"
    elif corruption == "source_leaf_application_abi":
        source_leaf["application_abi"] = "wrong"
    elif corruption == "source_leaf_optimization":
        source_leaf["optimization_level"] = 1
    elif corruption == "source_leaf_path":
        plane = source_leaf["plane_application"]
        assert isinstance(plane, dict)
        plane["application_path"] = "kernels/000000/missing.plane.symjit"
    elif corruption == "source_payload_digest_link":
        prepared_binding["source_application_sha256"] = "0" * 64
        refresh_catalog = True
    else:
        pytest.fail(f"unknown corruption {corruption}")

    if refresh_catalog is not None:
        _refresh_recurrence_catalog(
            execution,
            pack,
            refresh_binding_digests=refresh_catalog,
        )
    _rewrite_json_payload(tmp_path, execution_payload, execution)
    _rewrite_json_payload(tmp_path, pack_payload, pack)
    with pytest.raises(RunnerError, match=message):
        _authenticated_recurrence_source_identity(
            manifest,
            process_id="process-1",
            source_optimization_level=2,
        )


def test_resolved_sum_validation_and_pointwise_tolerances() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )

    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "ok"
    )
    runtime.resolved_total = (2.0 + 0.0j,)
    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "validation_failed"
    )

    assert pointwise_validation(1.0 + 1.0e-13, 1.0)["status"] == "ok"
    assert pointwise_validation(2.0, 1.0)["status"] == "validation_failed"


def test_matrix_element_conversion_rejects_sign_and_complex_drift() -> None:
    assert _real_nonnegative(2.0 + 0.0j) == 2.0
    assert _real_nonnegative(-1.0e-16 + 0.0j) == 0.0
    with pytest.raises(RunnerError, match="materially negative"):
        _real_nonnegative(-1.0e-3 + 0.0j)
    with pytest.raises(RunnerError, match="imaginary part"):
        _real_nonnegative(1.0 + 1.0e-3j)
