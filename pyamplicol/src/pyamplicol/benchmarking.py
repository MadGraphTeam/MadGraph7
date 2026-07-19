# SPDX-License-Identifier: 0BSD
"""Typed calibrated runtime profiling for Rusticol runtime backends."""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.protocols import Momenta, RuntimeBackend
from pyamplicol.api.results import (
    BenchmarkComponentTiming,
    BenchmarkResult,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
)
from pyamplicol.config import BenchmarkConfig, RunConfig
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)

_MAX_SAMPLE_RUNTIME_SECONDS = 0.25
_MAX_CALIBRATION_BLOCKS = 2
_MAX_REPETITIONS_PER_SAMPLE = 1_000_000_000
_CALIBRATION_LOWER_RATIO = 0.8
_CALIBRATION_UPPER_RATIO = 1.25
_MIN_CLOCK_INTERVAL_SECONDS = 1.0e-12
_MAX_NATIVE_PROFILE_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class _Calibration:
    sample_count: int
    repetitions_per_sample: int
    target_sample_seconds: float
    probe_seconds: float
    block_count: int
    evaluation_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _NativeProfileSample:
    execution_mode: str | None
    wall_time: float | None
    source_fill_time: float | None
    momentum_setup_time: float | None
    stage_input_pack_time: float | None
    stage_evaluator_call_time: float
    output_assign_time: float | None
    amplitude_input_pack_time: float | None
    amplitude_evaluator_call_time: float | None
    reduction_time: float | None
    stage_input_pack_times: tuple[float, ...] | None
    stage_evaluator_call_times: tuple[float, ...] | None
    stage_output_assign_times: tuple[float, ...] | None
    eager_gather_time: float | None
    eager_kernel_call_time: float | None
    eager_scatter_finalization_time: float | None
    eager_closure_time: float | None


class BenchmarkBackend:
    """Measure the optimized summed runtime path without changing its semantics."""

    def __init__(
        self,
        config: BenchmarkConfig | RunConfig | None,
        progress: ProgressSink | None,
    ) -> None:
        self._run_config = config if isinstance(config, RunConfig) else None
        self._config = (
            config.benchmark
            if isinstance(config, RunConfig)
            else config or BenchmarkConfig()
        )
        self._progress = progress

    def run(
        self,
        target: RuntimeBackend | os.PathLike[str] | str,
        *,
        points: Momenta | None = None,
    ) -> BenchmarkResult:
        target_path = (
            None
            if isinstance(target, RuntimeBackend)
            else Path(os.fspath(target)).expanduser().resolve(strict=False)
        )
        runtime = self._runtime(target)
        if points is None:
            loader = getattr(runtime, "validation_momenta", None)
            points = loader() if callable(loader) else None
        if points is None or len(points) == 0:
            raise EvaluationError(
                "benchmarking requires at least one phase-space point and the "
                "selected runtime has no deterministic validation point"
            )
        batch = _benchmark_batch(points, self._config.batch_size)
        helicities = self._config.helicity_ids or None
        color_flows = self._config.color_flow_ids or None
        profiler = _native_profiler(runtime) if self._config.precision == 16 else None
        native_wall_timer = (
            _native_wall_timer(runtime) if self._config.precision == 16 else None
        )
        def evaluate_once() -> object:
            return runtime.evaluate(
                batch,
                helicities=helicities,
                color_flows=color_flows,
                precision=self._config.precision,
            )

        def measure_repetitions(repetitions: int) -> float:
            if native_wall_timer is not None:
                return native_wall_timer(
                    batch,
                    repetitions,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=self._config.precision,
                )
            return _timed_repetitions(evaluate_once, repetitions)

        task_id = "runtime-benchmark"
        calibration_task_id = "runtime-profile-calibration"
        active_task_id = calibration_task_id
        warmup_elapsed = 0.0
        calibration: _Calibration | None = None
        samples: list[float] = []
        evaluator_samples: list[float] | None = [] if profiler else None
        native_profile_samples: list[_NativeProfileSample] | None = (
            [] if profiler else None
        )
        elapsed = 0.0
        evaluator_elapsed = 0.0
        interrupted = False
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    calibration_task_id,
                    "Calibrating runtime profile",
                    total=None,
                )
            )
        try:
            last_warmup_seconds: float | None = None
            for warmup_index in range(self._config.warmup_runs):
                warmup_started = time.perf_counter()
                last_warmup_seconds = _timed_repetitions(evaluate_once, 1)
                if profiler is not None:
                    profiler(
                        batch,
                        helicities=helicities,
                        color_flows=color_flows,
                        precision=self._config.precision,
                        include_values=False,
                    )
                warmup_elapsed += time.perf_counter() - warmup_started
                if self._progress is not None:
                    self._progress.emit(
                        ProgressUpdate(
                            calibration_task_id,
                            completed=warmup_index + 1,
                            total=None,
                            message="warmup",
                        )
                    )

            calibration = _calibrate_repetitions(
                evaluate_once,
                self._config,
                initial_seconds=(
                    None if native_wall_timer is not None else last_warmup_seconds
                ),
                timer=measure_repetitions,
            )
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        calibration_task_id,
                        message=(
                            f"{calibration.sample_count} blocks x "
                            f"{calibration.repetitions_per_sample} repetitions"
                        ),
                    )
                )
                self._progress.emit(
                    ProgressStart(
                        task_id,
                        "Profiling runtime",
                        total=calibration.sample_count,
                    )
                )
            active_task_id = task_id
            repetitions = calibration.repetitions_per_sample
            native_profile_sample_limit = min(
                calibration.sample_count,
                max(
                    1,
                    min(
                        self._config.minimum_samples,
                        _MAX_NATIVE_PROFILE_SAMPLES,
                    ),
                ),
            )
            try:
                for sample_index in range(calibration.sample_count):
                    duration = measure_repetitions(repetitions)
                    native_sample: _NativeProfileSample | None = None
                    profile_duration = 0.0
                    if (
                        profiler is not None
                        and evaluator_samples is not None
                        and native_profile_samples is not None
                        and sample_index < native_profile_sample_limit
                    ):
                        profile_started = time.perf_counter()
                        profile = profiler(
                            batch,
                            helicities=helicities,
                            color_flows=color_flows,
                            precision=self._config.precision,
                            include_values=False,
                        )
                        profile_duration = time.perf_counter() - profile_started
                        native_sample = _native_profile_sample(profile, len(batch))

                    samples.append(duration / (repetitions * len(batch)))
                    elapsed += duration
                    if native_sample is not None:
                        assert evaluator_samples is not None
                        assert native_profile_samples is not None
                        native_profile_samples.append(native_sample)
                        evaluator_samples.append(
                            native_sample.stage_evaluator_call_time
                            + (native_sample.amplitude_evaluator_call_time or 0.0)
                        )
                        evaluator_elapsed += profile_duration
                    if self._progress is not None:
                        self._progress.emit(
                            ProgressUpdate(
                                task_id,
                                completed=len(samples),
                                total=calibration.sample_count,
                                message=_sample_progress_message(
                                    samples,
                                    elapsed_seconds=elapsed,
                                    target_seconds=self._config.target_runtime,
                                    repetitions=repetitions,
                                    batch_size=len(batch),
                                ),
                            )
                        )
            except KeyboardInterrupt:
                if not samples:
                    raise
                interrupted = True
                if self._progress is not None:
                    self._progress.emit(
                        ProgressEnd(
                            task_id,
                            success=False,
                            message=(
                                f"interrupted after {len(samples)}/"
                                f"{calibration.sample_count} complete blocks; "
                                "reporting partial statistics"
                            ),
                        )
                    )
        except KeyboardInterrupt:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        active_task_id,
                        success=False,
                        message="interrupted before a complete timing block",
                    )
                )
            raise
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(active_task_id, success=False, message=str(exc))
                )
            if isinstance(exc, EvaluationError):
                raise
            raise EvaluationError(f"runtime benchmark failed: {exc}") from exc
        if self._progress is not None and not interrupted:
            self._progress.emit(ProgressEnd(task_id))

        assert calibration is not None

        wall_samples = samples
        wall_time_source = (
            "runtime_core_repeated_wall_time"
            if native_wall_timer is not None
            else "runtime_evaluate_wall_time"
        )
        if native_wall_timer is None and native_profile_samples is not None:
            native_wall_samples = [
                sample.wall_time
                for sample in native_profile_samples
                if sample.wall_time is not None
            ]
            if len(native_wall_samples) == len(native_profile_samples):
                wall_samples = native_wall_samples
                wall_time_source = "runtime_profile_core_wall_time"
        mean, deviation, error, relative_error = _sample_statistics(wall_samples)
        uncertainty = BenchmarkStatistics(deviation, error, relative_error)
        if evaluator_samples is None:
            evaluator_time_per_point = mean
            evaluator_uncertainty = uncertainty
            timing_breakdown = None
            evaluator_environment: dict[str, object] = {
                "wall_time_source": wall_time_source,
                "evaluator_time_source": "runtime_evaluate_wall_time",
                "native_profile_unavailable_reason": (
                    "non_f64_precision" if self._config.precision != 16 else None
                ),
            }
        else:
            (
                evaluator_time_per_point,
                evaluator_deviation,
                evaluator_error,
                evaluator_relative,
            ) = _sample_statistics(evaluator_samples)
            evaluator_uncertainty = BenchmarkStatistics(
                evaluator_deviation,
                evaluator_error,
                evaluator_relative,
            )
            assert native_profile_samples is not None
            timing_breakdown = _timing_breakdown(native_profile_samples)
            evaluator_environment = {
                "wall_time_source": wall_time_source,
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
                "evaluator_sample_count": len(evaluator_samples),
                "evaluator_elapsed_seconds": evaluator_elapsed,
                "native_profile_sample_count": len(evaluator_samples),
                "native_profile_sample_limit": native_profile_sample_limit,
                "native_profile_calls_per_block": (
                    len(evaluator_samples) / len(samples) if samples else 0.0
                ),
                "native_profile_warmup_call_count": self._config.warmup_runs,
                "native_profile_total_call_count": (
                    self._config.warmup_runs + len(evaluator_samples)
                ),
                "evaluator_standard_deviation_seconds_per_point": (evaluator_deviation),
                "evaluator_standard_error_seconds_per_point": evaluator_error,
                "evaluator_relative_standard_error": evaluator_relative,
            }
        physics = runtime.physics
        measured_evaluations = len(samples) * calibration.repetitions_per_sample
        return BenchmarkResult(
            requested_config=self._config,
            effective_config=self._config,
            sample_count=len(samples),
            wall_time_per_point=mean,
            evaluator_time_per_point=evaluator_time_per_point,
            uncertainty=uncertainty,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "target": None if target_path is None else str(target_path),
                "batch_size": len(batch),
                "precision": self._config.precision,
                "elapsed_seconds": elapsed,
                "interrupted": interrupted,
                "completed_sample_count": len(samples),
                "completion_fraction": len(samples) / calibration.sample_count,
                "warmup_elapsed_seconds": warmup_elapsed,
                "planned_sample_count": calibration.sample_count,
                "repetitions_per_sample": calibration.repetitions_per_sample,
                "measured_evaluation_count": measured_evaluations,
                "measured_point_count": measured_evaluations * len(batch),
                "target_sample_seconds": calibration.target_sample_seconds,
                "calibration_probe_seconds": calibration.probe_seconds,
                "calibration_block_count": calibration.block_count,
                "calibration_evaluation_count": calibration.evaluation_count,
                "calibration_elapsed_seconds": calibration.elapsed_seconds,
                **evaluator_environment,
            },
            interrupted=interrupted,
            repetitions_per_sample=calibration.repetitions_per_sample,
            evaluator_uncertainty=evaluator_uncertainty,
            process_id=physics.process_id,
            process_expression=physics.process,
            timing_breakdown=timing_breakdown,
        )

    def _runtime(
        self, target: RuntimeBackend | os.PathLike[str] | str
    ) -> RuntimeBackend:
        if isinstance(target, RuntimeBackend):
            return target
        from pyamplicol.runtime import load_runtime_backend

        run = self._run_config
        process = None if run is None else run.evaluation.process
        return load_runtime_backend(
            Path(os.fspath(target)).expanduser().resolve(strict=False),
            process=process,
            model_parameters=None,
            mute_warnings=False,
        )


def _benchmark_batch(points: Momenta, batch_size: int) -> Momenta:
    source = tuple(points)
    return tuple(source[index % len(source)] for index in range(batch_size))


def _sample_progress_message(
    samples: list[float],
    *,
    elapsed_seconds: float,
    target_seconds: float,
    repetitions: int,
    batch_size: int,
) -> str:
    mean, _deviation, error, relative_error = _sample_statistics(samples)
    if mean < 1.0e-6:
        scale, unit = 1.0e9, "ns"
    elif mean < 1.0e-3:
        scale, unit = 1.0e6, "us"
    elif mean < 1.0:
        scale, unit = 1.0e3, "ms"
    else:
        scale, unit = 1.0, "s"
    uncertainty = (
        "SE pending"
        if len(samples) < 2
        else f"+/- {error * scale:.2g} {unit} ({relative_error:.2%} rel. SE)"
    )
    return (
        f"{elapsed_seconds:.3g}/{target_seconds:.3g}s; "
        f"wall {mean * scale:.5g} {unit}/point {uncertainty}; "
        f"{repetitions} calls x {batch_size} points"
    )


def _planned_sample_count(
    config: BenchmarkConfig,
    *,
    probe_seconds: float,
) -> int:
    runtime_samples = math.ceil(config.target_runtime / _MAX_SAMPLE_RUNTIME_SECONDS)
    desired = max(config.minimum_samples, runtime_samples)
    maximum_for_runtime = max(
        math.floor(
            config.target_runtime / max(probe_seconds, _MIN_CLOCK_INTERVAL_SECONDS)
        ),
        1,
    )
    return max(config.minimum_samples, min(desired, maximum_for_runtime))


def _timed_repetitions(callback: Callable[[], object], repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        callback()
    return time.perf_counter() - started


def _estimated_repetitions(
    current: int,
    observed_seconds: float,
    target_seconds: float,
) -> int:
    observed = max(observed_seconds, _MIN_CLOCK_INTERVAL_SECONDS)
    estimate = math.ceil(current * target_seconds / observed)
    return min(max(estimate, 1), _MAX_REPETITIONS_PER_SAMPLE)


def _calibrate_repetitions(
    callback: Callable[[], object],
    config: BenchmarkConfig,
    *,
    initial_seconds: float | None,
    timer: Callable[[int], float] | None = None,
) -> _Calibration:
    measure = timer or (lambda repetitions: _timed_repetitions(callback, repetitions))
    block_count = 0
    evaluation_count = 0
    calibration_elapsed = 0.0
    if initial_seconds is None:
        initial_seconds = measure(1)
        block_count = 1
        evaluation_count = 1
        calibration_elapsed = initial_seconds

    probe_seconds = initial_seconds
    sample_count = _planned_sample_count(config, probe_seconds=probe_seconds)
    target_sample_seconds = config.target_runtime / sample_count
    observed_seconds = initial_seconds
    repetitions = 1
    for _ in range(_MAX_CALIBRATION_BLOCKS):
        candidate = _estimated_repetitions(
            repetitions,
            observed_seconds,
            target_sample_seconds,
        )
        if candidate == repetitions:
            break
        observed_seconds = measure(candidate)
        block_count += 1
        evaluation_count += candidate
        calibration_elapsed += observed_seconds
        repetitions = candidate
        ratio = observed_seconds / target_sample_seconds
        if _CALIBRATION_LOWER_RATIO <= ratio <= _CALIBRATION_UPPER_RATIO:
            break
    else:
        repetitions = _estimated_repetitions(
            repetitions,
            observed_seconds,
            target_sample_seconds,
        )

    return _Calibration(
        sample_count=sample_count,
        repetitions_per_sample=repetitions,
        target_sample_seconds=target_sample_seconds,
        probe_seconds=probe_seconds,
        block_count=block_count,
        evaluation_count=evaluation_count,
        elapsed_seconds=calibration_elapsed,
    )


def _native_profiler(
    runtime: RuntimeBackend,
) -> Callable[..., Mapping[str, object]] | None:
    if getattr(runtime, "supports_profiling", None) is False:
        return None
    for name in ("profile", "evaluate_profile"):
        profiler = getattr(runtime, name, None)
        if callable(profiler):
            return cast(Callable[..., Mapping[str, object]], profiler)
    return None


def _native_wall_timer(runtime: RuntimeBackend) -> Callable[..., float] | None:
    timer = getattr(runtime, "_benchmark_f64_wall_time", None)
    return timer if callable(timer) else None


def _profile_float(profile: Mapping[str, object], key: str) -> float:
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvaluationError(f"native runtime profile field {key!r} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise EvaluationError(f"native runtime profile field {key!r} is invalid")
    return value


def _profile_float_or_none(profile: Mapping[str, object], key: str) -> float | None:
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _profile_float_sequence(
    profile: Mapping[str, object], key: str
) -> tuple[float, ...]:
    values = profile.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EvaluationError(
            f"native runtime profile field {key!r} is not a numeric sequence"
        )
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise EvaluationError(
                f"native runtime profile field {key!r} is not a numeric sequence"
            )
        entry = float(value)
        if not math.isfinite(entry) or entry < 0.0:
            raise EvaluationError(f"native runtime profile field {key!r} is invalid")
        result.append(entry)
    return tuple(result)


def _profile_float_sequence_or_none(
    profile: Mapping[str, object], key: str
) -> tuple[float, ...] | None:
    if key not in profile:
        return None
    values = _profile_float_sequence(profile, key)
    # Eager execution has no compiled-stage timing vector. Rusticol still
    # supplies the aggregate evaluator timer, so an empty vector means
    # "unavailable", not a measured zero.
    return values or None


def _profile_execution_mode(
    profile: Mapping[str, object],
    *,
    stage_vectors_present_but_empty: bool,
) -> str | None:
    value = profile.get("execution_mode")
    if value is not None:
        if value not in {"compiled", "eager"}:
            raise EvaluationError(
                "native runtime profile execution_mode must be compiled or eager"
            )
        return str(value)
    stage_aggregate = profile.get("stage_evaluator_call_time_s")
    amplitude_aggregate = profile.get("amplitude_evaluator_call_time_s")
    if (
        stage_vectors_present_but_empty
        and isinstance(stage_aggregate, (float, int))
        and not isinstance(stage_aggregate, bool)
        and float(stage_aggregate) > 0.0
        and isinstance(amplitude_aggregate, (float, int))
        and not isinstance(amplitude_aggregate, bool)
        and float(amplitude_aggregate) == 0.0
    ):
        return "eager"
    return None


def _profile_point_count(profile: Mapping[str, object], fallback: int) -> int:
    value: Any = profile.get("points", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError("native runtime profile point count is invalid")
    return int(value)


def _native_profile_sample(
    profile: Mapping[str, object], fallback_points: int
) -> _NativeProfileSample:
    points = _profile_point_count(profile, fallback_points)
    per_point = 1.0 / points
    stage_vector_keys = (
        "stage_input_pack_by_stage_time_s",
        "stage_evaluator_call_by_stage_time_s",
        "stage_output_assign_by_stage_time_s",
    )
    stage_vectors_present_but_empty = all(
        key in profile
        and isinstance(profile[key], Sequence)
        and not isinstance(profile[key], (str, bytes))
        and len(profile[key]) == 0  # type: ignore[arg-type]
        for key in stage_vector_keys
    )
    stage_input_pack = _profile_float_sequence_or_none(
        profile, "stage_input_pack_by_stage_time_s"
    )
    stage_evaluator_call = _profile_float_sequence_or_none(
        profile, "stage_evaluator_call_by_stage_time_s"
    )
    stage_output_assign = _profile_float_sequence_or_none(
        profile, "stage_output_assign_by_stage_time_s"
    )

    stage_input_pack_total = (
        sum(stage_input_pack)
        if stage_input_pack is not None
        else _profile_float_or_none(profile, "stage_input_pack_time_s")
    )
    stage_evaluator_call_total = (
        sum(stage_evaluator_call)
        if stage_evaluator_call is not None
        else _profile_float_or_none(profile, "stage_evaluator_call_time_s")
    )
    if stage_evaluator_call_total is None:
        raise EvaluationError("native runtime stage evaluator timing is unavailable")
    output_assign_total = (
        sum(stage_output_assign)
        if stage_output_assign is not None
        else _profile_float_or_none(profile, "output_assign_time_s")
    )
    execution_mode = _profile_execution_mode(
        profile,
        stage_vectors_present_but_empty=stage_vectors_present_but_empty,
    )
    amplitude_evaluator_call: float | None = None
    if execution_mode != "eager":
        amplitude_evaluator_call = _profile_float_or_none(
            profile, "amplitude_evaluator_call_time_s"
        )
        if amplitude_evaluator_call is None:
            amplitude_evaluator_call = _profile_float(
                profile, "amplitude_evaluator_time_s"
            )

    def normalized(key: str) -> float | None:
        value = _profile_float_or_none(profile, key)
        return None if value is None else value * per_point

    def normalized_sequence(
        values: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        if values is None:
            return None
        return tuple(value * per_point for value in values)

    return _NativeProfileSample(
        execution_mode=execution_mode,
        wall_time=normalized("wall_time_s"),
        source_fill_time=normalized("source_fill_time_s"),
        momentum_setup_time=normalized("momentum_setup_time_s"),
        stage_input_pack_time=(
            None
            if execution_mode == "eager" or stage_input_pack_total is None
            else stage_input_pack_total * per_point
        ),
        stage_evaluator_call_time=stage_evaluator_call_total * per_point,
        output_assign_time=(
            None
            if execution_mode == "eager" or output_assign_total is None
            else output_assign_total * per_point
        ),
        amplitude_input_pack_time=(
            None
            if execution_mode == "eager"
            else normalized("amplitude_input_pack_time_s")
        ),
        amplitude_evaluator_call_time=(
            None
            if amplitude_evaluator_call is None
            else amplitude_evaluator_call * per_point
        ),
        reduction_time=(
            normalized("eager_reduction_time_s")
            if execution_mode == "eager"
            else normalized("reduction_time_s")
        ),
        stage_input_pack_times=normalized_sequence(stage_input_pack),
        stage_evaluator_call_times=normalized_sequence(stage_evaluator_call),
        stage_output_assign_times=normalized_sequence(stage_output_assign),
        eager_gather_time=normalized("eager_gather_time_s"),
        eager_kernel_call_time=normalized("eager_kernel_call_time_s"),
        eager_scatter_finalization_time=normalized(
            "eager_scatter_finalization_time_s"
        ),
        eager_closure_time=normalized("eager_closure_time_s"),
    )


def _component_timing(
    values: Sequence[float | None],
) -> BenchmarkComponentTiming | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    mean, deviation, error, relative_error = _sample_statistics(available)
    return BenchmarkComponentTiming(
        mean_seconds_per_point=mean,
        uncertainty=BenchmarkStatistics(deviation, error, relative_error),
        sample_count=len(available),
    )


def _stage_component_timing(
    samples: Sequence[_NativeProfileSample],
    attribute: str,
    stage_index: int,
) -> BenchmarkComponentTiming | None:
    values: list[float | None] = []
    for sample in samples:
        stage_values = getattr(sample, attribute)
        if not isinstance(stage_values, tuple) or stage_index >= len(stage_values):
            values.append(None)
        else:
            values.append(stage_values[stage_index])
    return _component_timing(values)


def _timing_breakdown(
    samples: Sequence[_NativeProfileSample],
) -> BenchmarkTimingBreakdown:
    if not samples:
        raise EvaluationError("native runtime profile returned no timing samples")
    stage_count = max(
        (
            len(values)
            for sample in samples
            for values in (
                sample.stage_input_pack_times,
                sample.stage_evaluator_call_times,
                sample.stage_output_assign_times,
            )
            if values is not None
        ),
        default=0,
    )
    stages: list[BenchmarkStageTiming] = []
    for stage_index in range(stage_count):
        input_pack = _stage_component_timing(
            samples, "stage_input_pack_times", stage_index
        )
        evaluator_call = _stage_component_timing(
            samples, "stage_evaluator_call_times", stage_index
        )
        output_assign = _stage_component_timing(
            samples, "stage_output_assign_times", stage_index
        )
        if any(
            value is not None for value in (input_pack, evaluator_call, output_assign)
        ):
            stages.append(
                BenchmarkStageTiming(
                    stage_index=stage_index + 1,
                    input_pack_time=input_pack,
                    evaluator_call_time=evaluator_call,
                    output_assign_time=output_assign,
                )
            )

    execution_modes = {
        sample.execution_mode
        for sample in samples
        if sample.execution_mode is not None
    }
    if len(execution_modes) > 1:
        raise EvaluationError("native runtime profile changed execution mode")
    execution_mode = cast(
        Literal["compiled", "eager"],
        next(iter(execution_modes), "compiled"),
    )
    evaluator_call_time = _component_timing(
        [sample.stage_evaluator_call_time for sample in samples]
    )
    return BenchmarkTimingBreakdown(
        sample_count=len(samples),
        execution_mode=execution_mode,
        wall_time=_component_timing([sample.wall_time for sample in samples]),
        source_fill_time=_component_timing(
            [sample.source_fill_time for sample in samples]
        ),
        momentum_setup_time=_component_timing(
            [sample.momentum_setup_time for sample in samples]
        ),
        stage_input_pack_time=_component_timing(
            [sample.stage_input_pack_time for sample in samples]
        ),
        stage_evaluator_call_time=(
            None if execution_mode == "eager" else evaluator_call_time
        ),
        output_assign_time=_component_timing(
            [sample.output_assign_time for sample in samples]
        ),
        amplitude_input_pack_time=_component_timing(
            [sample.amplitude_input_pack_time for sample in samples]
        ),
        amplitude_evaluator_call_time=_component_timing(
            [sample.amplitude_evaluator_call_time for sample in samples]
        ),
        reduction_time=_component_timing([sample.reduction_time for sample in samples]),
        eager_execution_time=(
            evaluator_call_time if execution_mode == "eager" else None
        ),
        eager_gather_time=_component_timing(
            [sample.eager_gather_time for sample in samples]
        ),
        eager_kernel_call_time=_component_timing(
            [sample.eager_kernel_call_time for sample in samples]
        ),
        eager_scatter_finalization_time=_component_timing(
            [sample.eager_scatter_finalization_time for sample in samples]
        ),
        eager_closure_time=_component_timing(
            [sample.eager_closure_time for sample in samples]
        ),
        stages=tuple(stages),
    )


def _sample_statistics(samples: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    error = deviation / math.sqrt(len(samples))
    relative_error = error / mean if mean > 0.0 else 0.0
    return mean, deviation, error, relative_error


def create_benchmark_backend(
    config: BenchmarkConfig | RunConfig | None,
    progress: ProgressSink | None,
) -> BenchmarkBackend:
    return BenchmarkBackend(config, progress)


__all__ = ["BenchmarkBackend", "create_benchmark_backend"]
