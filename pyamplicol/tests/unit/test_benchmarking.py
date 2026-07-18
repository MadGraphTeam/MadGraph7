# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

import pyamplicol.benchmarking as benchmark_module
from pyamplicol.api import (
    ColorFlow,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
    ReductionGroup,
    ResolvedEvaluation,
)
from pyamplicol.api.errors import EvaluationError
from pyamplicol.benchmarking import BenchmarkBackend
from pyamplicol.config import BenchmarkConfig
from pyamplicol.reporting import (
    CallbackProgressSink,
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
)


class _Runtime:
    calls = 0
    last_options: dict[str, object] | None = None

    @property
    def physics(self) -> ProcessPhysics:
        return ProcessPhysics(
            "test",
            "d d~ > z",
            "lc",
            "all",
            "all",
            "flow",
            0,
            (),
            (HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0),),
            (ColorFlow("c0", 0, (1,), True, "c0", 1.0),),
            (),
            PhysicsReduction(
                "lc-diagonal",
                (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),),
            ),
            (),
            (),
        )

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        self.calls += 1
        self.last_options = selectors
        return tuple(1.0 + 0.0j for _ in momenta)  # type: ignore[union-attr]

    def evaluate_resolved(
        self, momenta: object, **selectors: object
    ) -> ResolvedEvaluation:
        del momenta, selectors
        return ResolvedEvaluation((((1.0 + 0.0j,),),), ("h0",), ("c0",))

    def set_model_parameters(self, mapping: object) -> None:
        del mapping

    def mute_warnings(self) -> None:
        pass

    def unmute_warnings(self) -> None:
        pass


class _RuntimeWithValidation(_Runtime):
    def validation_momenta(self) -> object:
        return (((1.0, 0.0, 0.0, 1.0),),)


class _RuntimeWithProfile(_Runtime):
    profile_calls = 0

    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        self.profile_calls += 1
        assert kwargs["helicities"] == ("h0",)
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        return {
            "points": len(momenta),  # type: ignore[arg-type]
            "wall_time_s": 12.0e-6,
            "source_fill_time_s": 1.0e-6,
            "momentum_setup_time_s": 0.5e-6,
            "stage_input_pack_time_s": 3.0e-6,
            "stage_evaluator_call_time_s": 2.0e-6,
            "output_assign_time_s": 1.0e-6,
            "amplitude_input_pack_time_s": 1.0e-6,
            "amplitude_evaluator_call_time_s": 6.0e-6,
            "reduction_time_s": 0.5e-6,
            "stage_input_pack_by_stage_time_s": [1.0e-6, 2.0e-6],
            "stage_evaluator_call_by_stage_time_s": [0.5e-6, 1.5e-6],
            "stage_output_assign_by_stage_time_s": [0.25e-6, 0.75e-6],
        }


class _RuntimeWithUnavailableProfile(_Runtime):
    supports_profiling = False

    def profile(self, _momenta: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("unavailable native profiler must not be called")


class _RuntimeWithPerStageProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["stage_evaluator_call_time_s"] = -1.0
        profile["stage_evaluator_call_by_stage_time_s"] = [0.5e-6, 1.5e-6]
        return profile


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def perf_counter(self) -> float:
        return self.value


class _TimedRuntime(_Runtime):
    def __init__(self, clock: _Clock, duration: float = 1.0e-3) -> None:
        self.clock = clock
        self.duration = duration

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        self.calls += 1
        self.last_options = selectors
        self.clock.value += self.duration
        return tuple(1.0 + 0.0j for _ in momenta)  # type: ignore[union-attr]


class _TimedRuntimeWithProfile(_TimedRuntime):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(clock)
        self.profile_calls = 0

    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        self.profile_calls += 1
        return {
            "points": len(momenta),  # type: ignore[arg-type]
            "stage_evaluator_call_time_s": 2.0e-6,
            "amplitude_evaluator_call_time_s": 6.0e-6,
        }


class _TimedRuntimeWithNativeWall(_TimedRuntimeWithProfile):
    native_wall_calls = 0

    def _benchmark_f64_wall_time(
        self,
        momenta: object,
        repetitions: int,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> float:
        assert len(momenta) == 2  # type: ignore[arg-type]
        assert helicities is None
        assert color_flows is None
        assert precision == 16
        self.native_wall_calls += 1
        return repetitions * 0.5e-3


def test_benchmark_measures_minimum_samples_and_requested_batch() -> None:
    runtime = _Runtime()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=2,
        minimum_samples=5,
    )
    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )
    assert result.sample_count == 5
    assert runtime.calls == 7
    assert result.wall_time_per_point >= 0.0
    assert result.evaluator_time_per_point == result.wall_time_per_point
    assert result.environment["batch_size"] == 4
    assert result.environment["evaluator_time_source"] == "runtime_evaluate_wall_time"
    assert result.repetitions_per_sample == 1
    assert result.evaluation_count == 5
    assert result.evaluated_point_count == 20


def test_benchmark_calibrates_blocks_and_repetitions_toward_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntime(clock)
    events: list[object] = []
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, CallbackProgressSink(events.append)).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 4
    assert result.repetitions_per_sample == 25
    assert result.evaluation_count == 100
    assert result.evaluated_point_count == 200
    assert result.environment["elapsed_seconds"] == pytest.approx(0.1)
    assert result.environment["calibration_evaluation_count"] == 25
    assert runtime.calls == 126
    assert result.uncertainty.standard_error == pytest.approx(0.0)
    assert events[0] == ProgressStart(
        "runtime-profile-calibration", "Calibrating runtime profile"
    )
    assert (
        ProgressEnd(
            "runtime-profile-calibration",
            message="4 blocks x 25 repetitions",
        )
        in events
    )
    assert ProgressStart("runtime-benchmark", "Profiling runtime", 4) in events


def test_native_profile_calls_are_bounded_independently_of_wall_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithProfile(clock)
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.repetitions_per_sample == 25
    assert runtime.profile_calls == config.warmup_runs + result.sample_count
    assert result.environment["native_profile_calls_per_block"] == 1


def test_native_profile_calls_are_capped_for_long_target_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    config = BenchmarkConfig(
        target_runtime=20.0,
        batch_size=2,
        warmup_runs=2,
        minimum_samples=5,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 80
    assert result.environment["native_profile_sample_count"] == 5
    assert result.environment["native_profile_sample_limit"] == 5
    assert result.environment["native_profile_calls_per_block"] == pytest.approx(
        5 / 80
    )
    assert runtime.profile_calls == config.warmup_runs + 5


def test_benchmark_uses_repeated_native_rusticol_wall_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.wall_time_per_point == pytest.approx(0.25e-3)
    assert result.environment["wall_time_source"] == ("runtime_core_repeated_wall_time")
    assert result.environment["elapsed_seconds"] == pytest.approx(0.1)
    assert runtime.native_wall_calls >= result.sample_count


def test_keyboard_interrupt_returns_statistics_for_complete_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    events: list[object] = []

    def interrupt_after_two_samples(event: object) -> None:
        events.append(event)
        if (
            isinstance(event, ProgressUpdate)
            and event.task_id == "runtime-benchmark"
            and event.completed == 2
        ):
            raise KeyboardInterrupt

    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=0.1,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=4,
        ),
        CallbackProgressSink(interrupt_after_two_samples),
    ).run(runtime, points=(((1.0, 0.0, 0.0, 1.0),),))

    assert result.interrupted is True
    assert result.sample_count == 2
    assert result.environment["interrupted"] is True
    assert result.environment["completed_sample_count"] == 2
    assert result.environment["completion_fraction"] == pytest.approx(0.5)
    assert result.uncertainty.standard_error >= 0.0
    assert events[-1] == ProgressEnd(
        "runtime-benchmark",
        success=False,
        message=("interrupted after 2/4 complete blocks; reporting partial statistics"),
    )


def test_benchmark_reduces_planned_blocks_for_slow_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntime(clock, duration=2.0)
    config = BenchmarkConfig(
        target_runtime=10.0,
        batch_size=1,
        warmup_runs=1,
        minimum_samples=5,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 5
    assert result.repetitions_per_sample == 1
    assert result.environment["elapsed_seconds"] == pytest.approx(10.0)
    assert runtime.calls == 6


def test_benchmark_uses_native_profile_for_evaluator_time() -> None:
    runtime = _RuntimeWithProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )
    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )
    assert result.sample_count == 3
    assert runtime.calls == 4
    assert runtime.profile_calls == 4
    assert result.wall_time_per_point == pytest.approx(3.0e-6)
    assert result.evaluator_time_per_point == pytest.approx(2.0e-6)
    assert result.environment["wall_time_source"] == "runtime_profile_core_wall_time"
    assert (
        result.environment["evaluator_time_source"]
        == "runtime_profile_core_evaluator_call_time"
    )
    assert result.environment["evaluator_sample_count"] == 3
    assert result.environment["precision"] == 16
    assert result.evaluator_uncertainty is not None
    assert result.process_id == "test"
    assert result.process_expression == "d d~ > z"
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.sample_count == 3
    assert result.timing_breakdown.source_fill_time is not None
    assert result.timing_breakdown.source_fill_time.mean_seconds_per_point == (
        pytest.approx(0.25e-6)
    )
    assert len(result.timing_breakdown.stages) == 2


def test_benchmark_falls_back_to_valid_per_stage_native_timings() -> None:
    runtime = _RuntimeWithPerStageProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.evaluator_time_per_point == pytest.approx(2.0e-6)


def test_non_f64_benchmark_forwards_precision_without_native_profile() -> None:
    runtime = _RuntimeWithProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        precision=80,
        warmup_runs=1,
        minimum_samples=2,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.profile_calls == 0
    assert runtime.last_options is not None
    assert runtime.last_options["precision"] == 80
    assert result.environment["precision"] == 80
    assert result.environment["native_profile_unavailable_reason"] == (
        "non_f64_precision"
    )
    assert result.evaluator_time_per_point == result.wall_time_per_point


def test_benchmark_falls_back_when_native_profile_is_unavailable() -> None:
    runtime = _RuntimeWithUnavailableProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=2,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.calls == 3
    assert result.environment["wall_time_source"] == "runtime_evaluate_wall_time"
    assert result.evaluator_time_per_point == result.wall_time_per_point


def test_benchmark_uses_runtime_validation_point_when_points_are_omitted() -> None:
    runtime = _RuntimeWithValidation()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=2,
    )
    result = BenchmarkBackend(config, None).run(runtime)
    assert result.sample_count == 2
    assert runtime.calls == 3


def test_benchmark_without_points_or_artifact_fixture_is_explicit() -> None:
    with pytest.raises(EvaluationError, match="deterministic validation point"):
        BenchmarkBackend(BenchmarkConfig(), None).run(_Runtime())
