# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

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

    def profile_repeated(
        self, _momenta: object, _repetitions: int, **_kwargs: object
    ) -> dict[str, object]:
        raise AssertionError("unavailable repeated native profiler must not be called")


class _RuntimeWithPerStageProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["stage_evaluator_call_time_s"] = -1.0
        profile["stage_evaluator_call_by_stage_time_s"] = [0.5e-6, 1.5e-6]
        return profile


class _RuntimeWithEagerProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["wall_time_s"] = 100.0e-6
        profile["stage_input_pack_by_stage_time_s"] = []
        profile["stage_evaluator_call_by_stage_time_s"] = []
        profile["stage_output_assign_by_stage_time_s"] = []
        profile["stage_evaluator_call_time_s"] = 80.0e-6
        profile["amplitude_evaluator_call_time_s"] = 0.0
        return profile


class _RuntimeWithDetailedEagerProfile(_RuntimeWithEagerProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile.update(
            {
                "execution_mode": "eager",
                "eager_initialize_time_s": 2.0e-6,
                "eager_gather_time_s": 4.0e-6,
                "eager_kernel_call_time_s": 24.0e-6,
                "eager_invocation_scatter_time_s": 12.0e-6,
                "eager_finalization_time_s": 20.0e-6,
                "eager_scatter_finalization_time_s": 32.0e-6,
                "eager_closure_time_s": 4.0e-6,
                "eager_copy_out_time_s": 6.0e-6,
            }
        )
        return profile


class _RuntimeWithAmplitudeOnlyCompiledProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["stage_input_pack_by_stage_time_s"] = []
        profile["stage_evaluator_call_by_stage_time_s"] = []
        profile["stage_output_assign_by_stage_time_s"] = []
        profile["stage_evaluator_call_time_s"] = 0.0
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


class _TimedRuntimeWithRepeatedProfile(_TimedRuntimeWithNativeWall):
    repeated_profile_calls = 0

    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["helicities"] is None
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        self.repeated_profile_calls += 1
        points = len(momenta) * repetitions  # type: ignore[arg-type]
        return {
            "points": points,
            "wall_time_s": points * 4.0e-6,
            "stage_evaluator_call_time_s": points * 2.0e-6,
            "amplitude_evaluator_call_time_s": points * 0.5e-6,
            "reduction_time_s": points * 0.5e-6,
        }


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


def _write_effective_color_config(
    artifact: Path,
    *,
    accuracy: str = "lc",
    layout: str | None = "topology-replay",
) -> None:
    config = artifact / "config"
    config.mkdir(parents=True)
    layout_line = "" if layout is None else f'lc_flow_layout = "{layout}"\n'
    (config / "effective.toml").write_text(
        f'[color]\naccuracy = "{accuracy}"\n{layout_line}',
        encoding="utf-8",
    )


def test_benchmark_recommends_union_for_topology_replay_all_flow_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout=None)
    runtime = _Runtime()
    backend = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=1.0e-12,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=1,
            helicity_ids=("h0",),
        ),
        None,
    )
    monkeypatch.setattr(backend, "_runtime", lambda _target: runtime)

    with pytest.warns(UserWarning, match="--lc-flow-layout all-flow-union"):
        result = backend.run(
            artifact,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert result.environment["lc_flow_layout"] == "topology-replay"
    recommendation = result.environment["lc_flow_layout_recommendation"]
    assert isinstance(recommendation, str)
    assert "--lc-flow-layout all-flow-union" in recommendation
    assert "\x1b" not in recommendation


@pytest.mark.parametrize(
    ("layout", "color_accuracy", "helicity_ids", "color_flow_ids"),
    (
        ("all-flow-union", "lc", ("h0",), ()),
        ("topology-replay", "lc", ("h0",), ("c0",)),
        ("topology-replay", "lc", (), ()),
        ("topology-replay", "nlc", ("h0",), ()),
        ("topology-replay", "full", ("h0",), ()),
    ),
)
def test_lc_flow_layout_recommendation_exclusions(
    layout: str,
    color_accuracy: str,
    helicity_ids: tuple[str, ...],
    color_flow_ids: tuple[str, ...],
) -> None:
    assert (
        benchmark_module._lc_flow_layout_recommendation(
            color_accuracy=color_accuracy,
            lc_flow_layout=layout,
            selected_helicity_ids=helicity_ids,
            selected_color_ids=color_flow_ids,
        )
        is None
    )


def test_runtime_backend_has_no_artifact_layout_recommendation() -> None:
    runtime = _Runtime()
    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=1.0e-12,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=1,
            helicity_ids=("h0",),
        ),
        None,
    ).run(runtime, points=(((1.0, 0.0, 0.0, 1.0),),))

    assert "lc_flow_layout" not in result.environment
    assert "lc_flow_layout_recommendation" not in result.environment


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
    assert result.environment["native_profile_calls_per_block"] == pytest.approx(5 / 80)
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.environment["native_profile_points_per_sample"] == 2
    assert result.environment["timing_sample_contract"] == (
        "separate_native_profile_diagnostic_v1"
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
    assert result.environment["timing_sample_contract"] == (
        "separate_native_profile_diagnostic_v1"
    )
    assert result.environment["wall_time_sample_pass"] == (
        "runtime._benchmark_f64_wall_time"
    )
    assert result.environment["evaluator_time_sample_pass"] == "runtime.profile"
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.repetitions_per_sample > 1


def test_repeated_native_profile_supplies_headline_and_breakdown_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithRepeatedProfile(clock)
    runtime.repeated_profile_calls = 0
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

    assert runtime.repeated_profile_calls == result.sample_count
    assert result.wall_time_per_point == pytest.approx(4.0e-6)
    assert result.evaluator_time_per_point == pytest.approx(2.5e-6)
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.wall_time is not None
    assert result.timing_breakdown.wall_time.mean_seconds_per_point == pytest.approx(
        result.wall_time_per_point
    )
    assert result.timing_breakdown.wall_time.uncertainty == result.uncertainty
    assert result.timing_breakdown.sample_count == result.sample_count
    assert result.environment["native_profile_sample_count"] == result.sample_count
    assert result.environment["native_profile_repetitions_per_sample"] == (
        result.repetitions_per_sample
    )
    assert result.environment["native_profile_points_per_sample"] == (
        result.repetitions_per_sample * config.batch_size
    )
    assert result.environment["wall_time_sample_pass"] == "runtime.profile_repeated"
    assert result.environment["evaluator_time_sample_pass"] == (
        "runtime.profile_repeated"
    )
    assert result.environment["timing_sample_contract"] == (
        "shared_native_repeated_profile_v1"
    )


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
    assert result.environment["wall_time_sample_pass"] == "runtime.profile"
    assert (
        result.environment["evaluator_time_source"]
        == "runtime_profile_core_evaluator_call_time"
    )
    assert result.environment["evaluator_sample_count"] == 3
    assert result.environment["timing_sample_contract"] == (
        "shared_native_single_profile_v1"
    )
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.environment["precision"] == 16
    assert result.environment["execution_mode"] == "compiled"
    assert result.environment["color_workload"] == ("all 1 generated physical LC flows")
    assert result.environment["helicity_workload"] == "selected 1/1 helicities: h0"
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


def test_native_profile_reports_unaccounted_rusticol_core_time() -> None:
    profile = {
        "points": 2,
        "wall_time_s": 20.0e-6,
        "source_fill_time_s": 1.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_input_pack_time_s": 2.0e-6,
        "stage_evaluator_call_time_s": 4.0e-6,
        "output_assign_time_s": 2.0e-6,
        "amplitude_input_pack_time_s": 1.0e-6,
        "amplitude_evaluator_call_time_s": 3.0e-6,
        "reduction_time_s": 2.0e-6,
    }

    sample = benchmark_module._native_profile_sample(profile, fallback_points=2)

    assert sample.other_core_time == pytest.approx(2.0e-6)


def test_native_eager_profile_rejects_overlapping_top_level_phases() -> None:
    profile = {
        "execution_mode": "eager",
        "points": 4,
        "wall_time_s": 10.0e-6,
        "source_fill_time_s": 8.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_evaluator_call_time_s": 4.0e-6,
        "amplitude_evaluator_call_time_s": 0.0,
        "stage_input_pack_by_stage_time_s": [],
        "stage_evaluator_call_by_stage_time_s": [],
        "stage_output_assign_by_stage_time_s": [],
    }

    with pytest.raises(EvaluationError, match="exclusive top-level phases"):
        benchmark_module._native_profile_sample(profile, fallback_points=4)


def test_native_eager_profile_rejects_children_exceeding_inclusive_execution() -> None:
    profile = {
        "execution_mode": "eager",
        "points": 1,
        "wall_time_s": 20.0e-6,
        "source_fill_time_s": 1.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_evaluator_call_time_s": 10.0e-6,
        "amplitude_evaluator_call_time_s": 0.0,
        "eager_kernel_call_time_s": 11.0e-6,
        "stage_input_pack_by_stage_time_s": [],
        "stage_evaluator_call_by_stage_time_s": [],
        "stage_output_assign_by_stage_time_s": [],
    }

    with pytest.raises(EvaluationError, match="exclusive execution phases"):
        benchmark_module._native_profile_sample(profile, fallback_points=1)


def test_native_eager_shared_samples_partition_wall_time() -> None:
    samples = []
    for source_fill in (1.0e-6, 1.5e-6, 2.0e-6):
        sample = benchmark_module._native_profile_sample(
            {
                "execution_mode": "eager",
                "points": 1,
                "wall_time_s": 12.0e-6,
                "source_fill_time_s": source_fill,
                "momentum_setup_time_s": 1.0e-6,
                "stage_evaluator_call_time_s": 8.0e-6,
                "amplitude_evaluator_call_time_s": 0.0,
                "stage_input_pack_by_stage_time_s": [],
                "stage_evaluator_call_by_stage_time_s": [],
                "stage_output_assign_by_stage_time_s": [],
            },
            fallback_points=1,
        )
        assert sample.other_core_time is not None
        assert sample.source_fill_time is not None
        assert sample.momentum_setup_time is not None
        assert (
            sample.source_fill_time
            + sample.momentum_setup_time
            + sample.stage_evaluator_call_time
            + sample.other_core_time
        ) == pytest.approx(sample.wall_time)
        samples.append(sample)

    breakdown = benchmark_module._timing_breakdown(samples)
    assert breakdown.source_fill_time is not None
    assert breakdown.source_fill_time.uncertainty.standard_deviation > 0.0


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


def test_eager_profile_uses_nonzero_aggregate_when_stage_vectors_are_empty() -> None:
    runtime = _RuntimeWithEagerProfile()
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

    assert result.evaluator_time_per_point == pytest.approx(20.0e-6)
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.execution_mode == "eager"
    assert result.timing_breakdown.stage_evaluator_call_time is None
    aggregate = result.timing_breakdown.eager_execution_time
    assert aggregate is not None
    assert aggregate.mean_seconds_per_point == pytest.approx(20.0e-6)
    assert result.timing_breakdown.stage_input_pack_time is None
    assert result.timing_breakdown.output_assign_time is None
    assert result.timing_breakdown.amplitude_input_pack_time is None
    assert result.timing_breakdown.amplitude_evaluator_call_time is None
    assert result.timing_breakdown.reduction_time is None
    assert result.timing_breakdown.stages == ()


def test_eager_profile_preserves_detailed_native_phase_timings() -> None:
    runtime = _RuntimeWithDetailedEagerProfile()
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

    breakdown = result.timing_breakdown
    assert breakdown is not None
    assert breakdown.execution_mode == "eager"
    assert breakdown.eager_initialize_time is not None
    assert breakdown.eager_initialize_time.mean_seconds_per_point == pytest.approx(
        0.5e-6
    )
    assert breakdown.eager_gather_time is not None
    assert breakdown.eager_gather_time.mean_seconds_per_point == pytest.approx(1.0e-6)
    assert breakdown.eager_kernel_call_time is not None
    assert breakdown.eager_kernel_call_time.mean_seconds_per_point == pytest.approx(
        6.0e-6
    )
    assert breakdown.eager_invocation_scatter_time is not None
    invocation_scatter = breakdown.eager_invocation_scatter_time
    assert invocation_scatter.mean_seconds_per_point == pytest.approx(3.0e-6)
    assert breakdown.eager_finalization_time is not None
    assert breakdown.eager_finalization_time.mean_seconds_per_point == pytest.approx(
        5.0e-6
    )
    assert breakdown.eager_scatter_finalization_time is not None
    assert breakdown.eager_closure_time is not None
    assert breakdown.eager_copy_out_time is not None
    assert breakdown.eager_copy_out_time.mean_seconds_per_point == pytest.approx(1.5e-6)


def test_empty_stage_vectors_do_not_relabel_compiled_amplitude_only_profile() -> None:
    runtime = _RuntimeWithAmplitudeOnlyCompiledProfile()
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

    assert result.timing_breakdown is not None
    assert result.timing_breakdown.execution_mode == "compiled"
    assert result.evaluator_time_per_point == pytest.approx(1.5e-6)


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
