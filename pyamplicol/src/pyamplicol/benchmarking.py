# SPDX-License-Identifier: 0BSD
"""Typed calibrated runtime profiling for Rusticol runtime backends."""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
import tomllib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.protocols import Momenta, RuntimeBackend
from pyamplicol.api.results import (
    BenchmarkComponentTiming,
    BenchmarkProfileCounters,
    BenchmarkResult,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    ProcessPhysics,
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
_LC_TOPOLOGY_REPLAY_LAYOUT = "topology-replay"
_LC_ALL_FLOW_UNION_LAYOUT = "all-flow-union"
_LC_ALL_FLOW_PROFILE_RECOMMENDATION = (
    "this LC topology-replay artifact is profiling all color flows with "
    "runtime-selected helicities; regenerate with "
    "--lc-flow-layout all-flow-union for the optimized "
    "all-flows/single-helicity workload"
)


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
    native_input_pack_time: float | None
    native_input_crossing_time: float | None
    orchestration_time: float | None
    state_prepare_time: float | None
    state_clear_time: float | None
    source_fill_time: float | None
    momentum_input_setup_time: float | None
    momentum_setup_time: float | None
    model_parameter_setup_time: float | None
    stage_input_pack_time: float | None
    stage_leaf_input_pack_time: float | None
    stage_evaluator_call_time: float
    stage_backend_call_time: float | None
    stage_evaluator_output_gather_time: float | None
    output_assign_time: float | None
    amplitude_input_pack_time: float | None
    amplitude_evaluator_call_time: float | None
    amplitude_leaf_input_pack_time: float | None
    amplitude_backend_call_time: float | None
    amplitude_evaluator_output_gather_time: float | None
    amplitude_output_remap_time: float | None
    reduction_time: float | None
    total_materialization_time: float | None
    final_output_copy_time: float | None
    selector_planner_time: float | None
    selector_gather_time: float | None
    selector_scatter_time: float | None
    other_core_time: float | None
    stage_input_pack_times: tuple[float, ...] | None
    stage_leaf_input_pack_times: tuple[float, ...] | None
    stage_evaluator_call_times: tuple[float, ...] | None
    stage_backend_call_times: tuple[float, ...] | None
    stage_evaluator_output_gather_times: tuple[float, ...] | None
    stage_output_assign_times: tuple[float, ...] | None
    eager_initialize_time: float | None
    eager_gather_time: float | None
    eager_kernel_call_time: float | None
    eager_invocation_scatter_time: float | None
    eager_finalization_time: float | None
    eager_scatter_finalization_time: float | None
    eager_closure_time: float | None
    eager_copy_out_time: float | None
    recurrence_momentum_fill_time: float | None
    recurrence_union_source_fill_time: float | None
    recurrence_schedule_time: float | None
    recurrence_source_kernel_time: float | None
    recurrence_contribution_kernel_time: float | None
    recurrence_finalization_time: float | None
    recurrence_closure_time: float | None
    recurrence_replay_output_mapping_time: float | None
    counters: _NativeProfileCounterSample | None


@dataclass(frozen=True, slots=True)
class _NativeProfileCounterSample:
    native_input_components_per_point: float | None
    native_input_pack_bytes_per_point: float | None
    native_input_crossing_bytes_per_point: float | None
    state_components_per_point: float | None
    state_clear_components_per_point: float | None
    source_components_per_point: float | None
    momentum_components_per_point: float | None
    model_parameter_components_per_point: float | None
    stage_input_copy_components_per_point: float | None
    stage_leaf_input_copy_components_per_point: float | None
    stage_evaluator_output_gather_components_per_point: float | None
    stage_output_assign_components_per_point: float | None
    amplitude_input_copy_components_per_point: float | None
    amplitude_leaf_input_copy_components_per_point: float | None
    amplitude_evaluator_output_gather_components_per_point: float | None
    amplitude_output_remap_components_per_point: float | None
    reduction_input_components_per_point: float | None
    selector_gather_points_per_point: float | None
    selector_gather_bytes_per_point: float | None
    selector_scatter_values_per_point: float | None
    resolved_materialized_components_per_point: float | None
    total_materialized_values_per_point: float | None
    final_output_copy_values_per_point: float | None
    native_input_container_allocations_per_call: float | None
    evaluator_backend_calls_per_call: float | None
    observed_scratch_reallocations_per_call: float | None
    native_output_allocations_per_call: float | None


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
        physics = runtime.physics
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
        color_flows = (
            _resolve_color_flow_ordinals(physics, self._config.color_flow_ids) or None
        )
        lc_flow_layout = _artifact_lc_flow_layout(target_path)
        lc_flow_layout_recommendation = _lc_flow_layout_recommendation(
            color_accuracy=physics.color_accuracy,
            lc_flow_layout=lc_flow_layout,
            selected_helicity_ids=tuple(helicities or ()),
            selected_color_ids=tuple(color_flows or ()),
        )
        if lc_flow_layout_recommendation is not None:
            warnings.warn(
                lc_flow_layout_recommendation,
                UserWarning,
                stacklevel=2,
            )
        profiler = _native_profiler(runtime) if self._config.precision == 16 else None
        repeated_profiler = (
            _native_repeated_profiler(runtime) if self._config.precision == 16 else None
        )
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
        evaluator_samples: list[float] | None = (
            [] if profiler is not None or repeated_profiler is not None else None
        )
        native_profile_samples: list[_NativeProfileSample] | None = (
            [] if profiler is not None or repeated_profiler is not None else None
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
                last_warmup_seconds = measure_repetitions(1)
                if repeated_profiler is not None:
                    repeated_profiler(
                        batch,
                        1,
                        helicities=helicities,
                        color_flows=color_flows,
                        precision=self._config.precision,
                        include_values=False,
                    )
                elif profiler is not None:
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
                initial_seconds=last_warmup_seconds,
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
            native_profile_sample_limit = (
                calibration.sample_count
                if repeated_profiler is not None
                else min(
                    calibration.sample_count,
                    max(
                        1,
                        min(
                            self._config.minimum_samples,
                            _MAX_NATIVE_PROFILE_SAMPLES,
                        ),
                    ),
                )
            )
            try:
                for sample_index in range(calibration.sample_count):
                    native_sample: _NativeProfileSample | None = None
                    profile_duration = 0.0
                    duration = measure_repetitions(repetitions)
                    sample_seconds_per_point = duration / (repetitions * len(batch))
                    if repeated_profiler is not None:
                        profile_started = time.perf_counter()
                        profile = repeated_profiler(
                            batch,
                            repetitions,
                            helicities=helicities,
                            color_flows=color_flows,
                            precision=self._config.precision,
                            include_values=False,
                        )
                        profile_duration = time.perf_counter() - profile_started
                        native_sample = _native_profile_sample(
                            profile,
                            len(batch) * repetitions,
                            repetitions=repetitions,
                        )
                        if native_sample.wall_time is None:
                            raise EvaluationError(
                                "repeated native profile did not report core wall time"
                            )
                    elif (
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

                    samples.append(sample_seconds_per_point)
                    elapsed += duration
                    if native_sample is not None:
                        assert evaluator_samples is not None
                        assert native_profile_samples is not None
                        native_profile_samples.append(native_sample)
                        if native_sample.execution_mode == "recurrence":
                            if native_sample.recurrence_schedule_time is None:
                                raise EvaluationError(
                                    "native recurrence schedule timing is unavailable"
                                )
                            evaluator_samples.append(
                                native_sample.recurrence_schedule_time
                            )
                        else:
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
        wall_sample_pass = (
            "runtime._benchmark_f64_wall_time"
            if native_wall_timer is not None
            else "runtime.evaluate"
        )
        timing_sample_contract = (
            "paired_unprofiled_headline_profiled_attribution_v1"
            if repeated_profiler is not None
            else "separate_native_profile_diagnostic_v1"
        )
        mean, deviation, error, relative_error = _sample_statistics(wall_samples)
        uncertainty = BenchmarkStatistics(deviation, error, relative_error)
        if evaluator_samples is None:
            evaluator_time_per_point = mean
            evaluator_uncertainty = uncertainty
            timing_breakdown = None
            evaluator_environment: dict[str, object] = {
                "wall_time_source": wall_time_source,
                "wall_time_sample_pass": wall_sample_pass,
                "evaluator_time_source": "runtime_evaluate_wall_time",
                "timing_sample_contract": "headline_only_no_breakdown_v1",
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
            evaluator_time_source = (
                "runtime_profile_core_recurrence_schedule_time"
                if timing_breakdown.execution_mode == "recurrence"
                else "runtime_profile_core_evaluator_call_time"
            )
            native_profile_repetitions = (
                calibration.repetitions_per_sample
                if repeated_profiler is not None
                else 1
            )
            native_profile_warmup_calls = (
                self._config.warmup_runs
                if profiler is not None or repeated_profiler is not None
                else 0
            )
            paired_profile_attribution = repeated_profiler is not None
            evaluator_environment = {
                "wall_time_source": wall_time_source,
                "wall_time_sample_pass": wall_sample_pass,
                "evaluator_time_source": evaluator_time_source,
                "evaluator_time_sample_pass": (
                    "runtime.profile_repeated"
                    if repeated_profiler is not None
                    else "runtime.profile"
                ),
                "timing_breakdown_sample_pass": (
                    "runtime.profile_repeated"
                    if repeated_profiler is not None
                    else "runtime.profile"
                ),
                "evaluator_sample_count": len(evaluator_samples),
                "evaluator_elapsed_seconds": evaluator_elapsed,
                "native_profile_sample_count": len(evaluator_samples),
                "native_profile_sample_limit": native_profile_sample_limit,
                "native_profile_repetitions_per_sample": (native_profile_repetitions),
                "native_profile_points_per_sample": (
                    native_profile_repetitions * len(batch)
                ),
                "native_profile_calls_per_block": (
                    len(evaluator_samples) / len(samples) if samples else 0.0
                ),
                "native_profile_warmup_call_count": native_profile_warmup_calls,
                "native_profile_total_call_count": (
                    native_profile_warmup_calls + len(evaluator_samples)
                ),
                "profile_attribution_paired_with_headline": (
                    paired_profile_attribution
                ),
                "profile_attribution_identical_batch": paired_profile_attribution,
                "profile_attribution_identical_repetitions": (
                    paired_profile_attribution
                ),
                "profile_attribution_evaluation_count": (
                    len(evaluator_samples) * native_profile_repetitions
                ),
                "profile_attribution_point_count": (
                    len(evaluator_samples) * native_profile_repetitions * len(batch)
                ),
                "timing_sample_contract": timing_sample_contract,
                "evaluator_standard_deviation_seconds_per_point": (evaluator_deviation),
                "evaluator_standard_error_seconds_per_point": evaluator_error,
                "evaluator_relative_standard_error": evaluator_relative,
            }
        execution_mode = (
            timing_breakdown.execution_mode
            if timing_breakdown is not None
            else str(getattr(runtime, "execution_mode", "unavailable"))
        )
        selected_helicity_ids = tuple(helicities or ())
        selected_color_ids = tuple(color_flows or ())
        measured_evaluations = len(samples) * calibration.repetitions_per_sample
        layout_environment: dict[str, object] = {}
        if lc_flow_layout is not None:
            layout_environment = {
                "lc_flow_layout": lc_flow_layout,
                "lc_flow_layout_recommendation": lc_flow_layout_recommendation,
            }
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
                "execution_mode": execution_mode,
                "color_accuracy": physics.color_accuracy,
                "color_workload": _color_workload_text(
                    physics,
                    selected_color_ids,
                ),
                "helicity_workload": _helicity_workload_text(
                    physics,
                    selected_helicity_ids,
                ),
                "selected_color_ids": selected_color_ids,
                "selected_helicity_ids": selected_helicity_ids,
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
                **layout_environment,
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
        path = Path(os.fspath(target)).expanduser().resolve(strict=False)
        task_id = "process-output-load"
        started = time.perf_counter()
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    f"Loading process output {path}",
                    details={"step": "loading process output", "path": str(path)},
                )
            )
        try:
            runtime = load_runtime_backend(
                path,
                process=process,
                model_parameters=None,
                mute_warnings=False,
            )
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=str(exc),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
            raise
        if self._progress is not None:
            self._progress.emit(
                ProgressEnd(
                    task_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        return runtime


def _artifact_lc_flow_layout(target_path: Path | None) -> str | None:
    if target_path is None:
        return None
    effective_path = target_path / "config" / "effective.toml"
    try:
        payload = tomllib.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    color = payload.get("color")
    if not isinstance(color, Mapping) or color.get("accuracy") != "lc":
        return None
    layout = color.get("lc_flow_layout", _LC_TOPOLOGY_REPLAY_LAYOUT)
    if layout not in {_LC_TOPOLOGY_REPLAY_LAYOUT, _LC_ALL_FLOW_UNION_LAYOUT}:
        return None
    return str(layout)


def _lc_flow_layout_recommendation(
    *,
    color_accuracy: str,
    lc_flow_layout: str | None,
    selected_helicity_ids: Sequence[str],
    selected_color_ids: Sequence[str],
) -> str | None:
    if (
        color_accuracy == "lc"
        and lc_flow_layout == _LC_TOPOLOGY_REPLAY_LAYOUT
        and selected_helicity_ids
        and not selected_color_ids
    ):
        return _LC_ALL_FLOW_PROFILE_RECOMMENDATION
    return None


def _benchmark_batch(points: Momenta, batch_size: int) -> Momenta:
    source = tuple(points)
    return tuple(source[index % len(source)] for index in range(batch_size))


def _resolve_color_flow_ordinals(
    physics: ProcessPhysics,
    requested: Sequence[str],
) -> tuple[str, ...]:
    available = physics.color_ids
    resolved: list[str] = []
    for value in requested:
        if value in available:
            resolved.append(value)
            continue
        try:
            ordinal = int(value, 10)
        except ValueError:
            resolved.append(value)
            continue
        if str(ordinal) != value.strip() or ordinal < 1 or ordinal > len(available):
            maximum = len(available)
            raise EvaluationError(
                f"color-flow ordinal {value!r} is out of range; choose 1..{maximum} "
                "or a stable color component ID"
            )
        resolved.append(available[ordinal - 1])
    return tuple(resolved)


def _color_workload_text(
    physics: ProcessPhysics,
    selected: Sequence[str],
) -> str:
    if physics.color_accuracy != "lc":
        return f"contracted {physics.color_accuracy.upper()} color total"
    count = len(physics.color_ids)
    if not selected:
        return f"all {count} generated physical LC flows"
    return f"selected {len(selected)}/{count} physical LC flows: {', '.join(selected)}"


def _helicity_workload_text(
    physics: ProcessPhysics,
    selected: Sequence[str],
) -> str:
    count = len(physics.helicity_ids)
    if not selected:
        structural = physics.structural_zero_helicity_count
        suffix = f"; {structural} structural zeros" if structural else ""
        return f"all {count} generated helicity configurations{suffix}"
    return f"selected {len(selected)}/{count} helicities: {', '.join(selected)}"


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
        else (
            f"+/- {error * scale:.2g} {unit} "
            f"(relative standard error {relative_error:.2%})"
        )
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


def _native_repeated_profiler(
    runtime: RuntimeBackend,
) -> Callable[..., Mapping[str, object]] | None:
    if getattr(runtime, "supports_profiling", None) is False:
        return None
    profiler = getattr(runtime, "profile_repeated", None)
    return (
        cast(Callable[..., Mapping[str, object]], profiler)
        if callable(profiler)
        else None
    )


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


def _profile_count_or_none(
    profile: Mapping[str, object],
    key: str,
    *,
    denominator: int,
) -> float | None:
    if key not in profile:
        return None
    value = profile[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(
            f"native runtime profile counter {key!r} is not a non-negative integer"
        )
    return value / denominator


def _native_profile_counters(
    profile: Mapping[str, object],
    *,
    points: int,
    repetitions: int,
) -> _NativeProfileCounterSample | None:
    def per_point(key: str) -> float | None:
        return _profile_count_or_none(profile, key, denominator=points)

    def per_call(key: str) -> float | None:
        return _profile_count_or_none(profile, key, denominator=repetitions)

    counters = _NativeProfileCounterSample(
        native_input_components_per_point=per_point("native_input_component_count"),
        native_input_pack_bytes_per_point=per_point("native_input_pack_bytes"),
        native_input_crossing_bytes_per_point=per_point("native_input_crossing_bytes"),
        state_components_per_point=per_point("state_component_count"),
        state_clear_components_per_point=per_point("state_clear_component_count"),
        source_components_per_point=per_point("source_component_count"),
        momentum_components_per_point=per_point("momentum_component_count"),
        model_parameter_components_per_point=per_point(
            "model_parameter_component_count"
        ),
        stage_input_copy_components_per_point=per_point(
            "stage_input_copy_component_count"
        ),
        stage_leaf_input_copy_components_per_point=per_point(
            "stage_leaf_input_copy_component_count"
        ),
        stage_evaluator_output_gather_components_per_point=per_point(
            "stage_evaluator_output_gather_component_count"
        ),
        stage_output_assign_components_per_point=per_point(
            "stage_output_assign_component_count"
        ),
        amplitude_input_copy_components_per_point=per_point(
            "amplitude_input_copy_component_count"
        ),
        amplitude_leaf_input_copy_components_per_point=per_point(
            "amplitude_leaf_input_copy_component_count"
        ),
        amplitude_evaluator_output_gather_components_per_point=per_point(
            "amplitude_evaluator_output_gather_component_count"
        ),
        amplitude_output_remap_components_per_point=per_point(
            "amplitude_output_remap_component_count"
        ),
        reduction_input_components_per_point=per_point(
            "reduction_input_component_count"
        ),
        selector_gather_points_per_point=per_point("selector_gather_point_count"),
        selector_gather_bytes_per_point=per_point("selector_gather_bytes"),
        selector_scatter_values_per_point=per_point("selector_scatter_value_count"),
        resolved_materialized_components_per_point=per_point(
            "resolved_materialized_component_count"
        ),
        total_materialized_values_per_point=per_point("total_materialized_value_count"),
        final_output_copy_values_per_point=per_point("final_output_copy_value_count"),
        native_input_container_allocations_per_call=per_call(
            "native_input_container_allocation_count"
        ),
        evaluator_backend_calls_per_call=per_call("evaluator_backend_call_count"),
        observed_scratch_reallocations_per_call=per_call(
            "observed_scratch_reallocation_count"
        ),
        native_output_allocations_per_call=per_call("native_output_allocation_count"),
    )
    if not any(getattr(counters, field.name) is not None for field in fields(counters)):
        return None
    return counters


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
        if value not in {"compiled", "eager", "recurrence"}:
            raise EvaluationError(
                "native runtime profile execution_mode must be compiled, eager, "
                "or recurrence"
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
    profile: Mapping[str, object],
    fallback_points: int,
    *,
    repetitions: int = 1,
) -> _NativeProfileSample:
    points = _profile_point_count(profile, fallback_points)
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise EvaluationError("native runtime profile repetition count is invalid")
    if points % repetitions != 0:
        raise EvaluationError(
            "native runtime profile point count is not divisible by repetitions"
        )
    per_point = 1.0 / points
    counters = _native_profile_counters(
        profile,
        points=points,
        repetitions=repetitions,
    )
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
    stage_leaf_input_pack = _profile_float_sequence_or_none(
        profile, "stage_leaf_input_pack_by_stage_time_s"
    )
    stage_evaluator_call = _profile_float_sequence_or_none(
        profile, "stage_evaluator_call_by_stage_time_s"
    )
    stage_backend_call = _profile_float_sequence_or_none(
        profile, "stage_backend_call_by_stage_time_s"
    )
    stage_evaluator_output_gather = _profile_float_sequence_or_none(
        profile, "stage_evaluator_output_gather_by_stage_time_s"
    )
    stage_output_assign = _profile_float_sequence_or_none(
        profile, "stage_output_assign_by_stage_time_s"
    )
    execution_mode = _profile_execution_mode(
        profile,
        stage_vectors_present_but_empty=stage_vectors_present_but_empty,
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
    if stage_evaluator_call_total is None and execution_mode != "recurrence":
        raise EvaluationError("native runtime stage evaluator timing is unavailable")
    if stage_evaluator_call_total is None:
        stage_evaluator_call_total = 0.0
    output_assign_total = (
        sum(stage_output_assign)
        if stage_output_assign is not None
        else _profile_float_or_none(profile, "output_assign_time_s")
    )
    stage_leaf_input_pack_total = (
        sum(stage_leaf_input_pack)
        if stage_leaf_input_pack is not None
        else _profile_float_or_none(profile, "stage_leaf_input_pack_time_s")
    )
    stage_backend_call_total = (
        sum(stage_backend_call)
        if stage_backend_call is not None
        else _profile_float_or_none(profile, "stage_backend_call_time_s")
    )
    stage_evaluator_output_gather_total = (
        sum(stage_evaluator_output_gather)
        if stage_evaluator_output_gather is not None
        else _profile_float_or_none(profile, "stage_evaluator_output_gather_time_s")
    )
    amplitude_evaluator_call: float | None = None
    if execution_mode not in {"eager", "recurrence"}:
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

    wall_time = normalized("wall_time_s")
    native_input_pack_time = normalized("native_input_pack_time_s")
    native_input_crossing_time = normalized("native_input_crossing_time_s")
    orchestration_time = normalized("orchestration_time_s")
    state_prepare_time = normalized("state_prepare_time_s")
    state_clear_time = normalized("state_clear_time_s")
    source_fill_time = normalized("source_fill_time_s")
    momentum_setup_time = normalized("momentum_setup_time_s")
    momentum_input_setup_time = normalized("momentum_input_setup_time_s")
    model_parameter_setup_time = normalized("model_parameter_setup_time_s")
    if momentum_input_setup_time is None:
        momentum_input_setup_time = momentum_setup_time
        if (
            momentum_input_setup_time is not None
            and model_parameter_setup_time is not None
        ):
            momentum_input_setup_time = max(
                momentum_input_setup_time - model_parameter_setup_time,
                0.0,
            )
    compiled_profile = execution_mode not in {"eager", "recurrence"}
    stage_input_pack_time = (
        None
        if not compiled_profile or stage_input_pack_total is None
        else stage_input_pack_total * per_point
    )
    stage_leaf_input_pack_time = (
        None
        if not compiled_profile or stage_leaf_input_pack_total is None
        else stage_leaf_input_pack_total * per_point
    )
    stage_evaluator_call_time = stage_evaluator_call_total * per_point
    stage_backend_call_time = (
        None
        if not compiled_profile or stage_backend_call_total is None
        else stage_backend_call_total * per_point
    )
    stage_evaluator_output_gather_time = (
        None
        if not compiled_profile or stage_evaluator_output_gather_total is None
        else stage_evaluator_output_gather_total * per_point
    )
    output_assign_time = (
        None
        if not compiled_profile or output_assign_total is None
        else output_assign_total * per_point
    )
    amplitude_input_pack_time = (
        normalized("amplitude_input_pack_time_s") if compiled_profile else None
    )
    amplitude_evaluator_call_time = (
        None
        if amplitude_evaluator_call is None
        else amplitude_evaluator_call * per_point
    )
    amplitude_leaf_input_pack_time = (
        normalized("amplitude_leaf_input_pack_time_s") if compiled_profile else None
    )
    amplitude_backend_call_time = (
        normalized("amplitude_backend_call_time_s") if compiled_profile else None
    )
    amplitude_evaluator_output_gather_time = (
        normalized("amplitude_evaluator_output_gather_time_s")
        if compiled_profile
        else None
    )
    amplitude_output_remap_time = (
        normalized("amplitude_output_remap_time_s") if compiled_profile else None
    )
    reduction_time = (
        normalized("eager_reduction_time_s")
        if execution_mode == "eager"
        else normalized("reduction_time_s")
    )
    total_materialization_time = normalized("total_materialization_time_s")
    final_output_copy_time = normalized("final_output_copy_time_s")
    selector_planner_time = normalized("selector_planner_time_s")
    selector_gather_time = normalized("selector_gather_time_s")
    selector_scatter_time = normalized("selector_scatter_time_s")
    eager_execution_time = (
        stage_evaluator_call_time if execution_mode == "eager" else None
    )
    eager_initialize_time = normalized("eager_initialize_time_s")
    eager_gather_time = normalized("eager_gather_time_s")
    eager_kernel_call_time = normalized("eager_kernel_call_time_s")
    eager_invocation_scatter_time = normalized("eager_invocation_scatter_time_s")
    eager_finalization_time = normalized("eager_finalization_time_s")
    eager_scatter_finalization_time = normalized("eager_scatter_finalization_time_s")
    eager_closure_time = normalized("eager_closure_time_s")
    eager_copy_out_time = normalized("eager_copy_out_time_s")
    recurrence_momentum_fill_time = normalized("recurrence_momentum_fill_time_s")
    recurrence_union_source_fill_time = normalized(
        "recurrence_union_source_fill_time_s"
    )
    recurrence_schedule_time = normalized("recurrence_schedule_time_s")
    recurrence_source_kernel_time = normalized("recurrence_source_kernel_time_s")
    recurrence_contribution_kernel_time = normalized(
        "recurrence_contribution_kernel_time_s"
    )
    recurrence_finalization_time = normalized("recurrence_finalization_time_s")
    recurrence_closure_time = normalized("recurrence_closure_time_s")
    recurrence_replay_output_mapping_time = normalized(
        "recurrence_replay_output_mapping_time_s"
    )
    if execution_mode == "recurrence" and recurrence_schedule_time is None:
        raise EvaluationError("native recurrence schedule timing is unavailable")

    common_accounted = (
        native_input_pack_time,
        native_input_crossing_time,
        orchestration_time,
        state_prepare_time,
        state_clear_time,
        source_fill_time,
        momentum_input_setup_time,
        model_parameter_setup_time,
    )
    if execution_mode == "eager":
        mode_accounted = (eager_execution_time,)
    elif execution_mode == "recurrence":
        mode_accounted = (
            recurrence_momentum_fill_time,
            recurrence_union_source_fill_time,
            recurrence_schedule_time,
            recurrence_replay_output_mapping_time,
            reduction_time,
        )
    else:
        mode_accounted = (
            stage_input_pack_time,
            stage_evaluator_call_time,
            output_assign_time,
            amplitude_input_pack_time,
            amplitude_evaluator_call_time,
            reduction_time,
        )
    accounted = (
        *common_accounted,
        *mode_accounted,
        total_materialization_time,
        final_output_copy_time,
        selector_planner_time,
        selector_gather_time,
        selector_scatter_time,
    )
    accounted_total = sum(value or 0.0 for value in accounted)
    accounting_tolerance = (
        1.0e-12 if wall_time is None else max(1.0e-12, wall_time * 1.0e-12)
    )
    if wall_time is not None and accounted_total > wall_time + accounting_tolerance:
        raise EvaluationError(
            "native profile exclusive top-level phases account for "
            f"{accounted_total:.9e}s/point, exceeding wall time "
            f"{wall_time:.9e}s/point"
        )
    if execution_mode == "eager" and eager_execution_time is not None:
        scatter_phases = (
            (eager_invocation_scatter_time, eager_finalization_time)
            if eager_invocation_scatter_time is not None
            or eager_finalization_time is not None
            else (eager_scatter_finalization_time,)
        )
        exclusive_eager_phases = (
            eager_initialize_time,
            eager_gather_time,
            eager_kernel_call_time,
            *scatter_phases,
            eager_closure_time,
            reduction_time,
            eager_copy_out_time,
        )
        exclusive_eager_total = sum(value or 0.0 for value in exclusive_eager_phases)
        if exclusive_eager_total > eager_execution_time + accounting_tolerance:
            raise EvaluationError(
                "native eager profile exclusive execution phases account for "
                f"{exclusive_eager_total:.9e}s/point, exceeding the inclusive "
                f"eager execution time {eager_execution_time:.9e}s/point"
            )
    if execution_mode == "recurrence" and recurrence_schedule_time is not None:
        recurrence_attribution = (
            recurrence_source_kernel_time,
            recurrence_contribution_kernel_time,
            recurrence_finalization_time,
            recurrence_closure_time,
        )
        recurrence_attributed_total = sum(
            value or 0.0 for value in recurrence_attribution
        )
        if recurrence_attributed_total > (
            recurrence_schedule_time + accounting_tolerance
        ):
            raise EvaluationError(
                "native recurrence profile schedule sub-attribution accounts for "
                f"{recurrence_attributed_total:.9e}s/point, exceeding the "
                "inclusive recurrence schedule time "
                f"{recurrence_schedule_time:.9e}s/point"
            )
    other_core_time = (
        None if wall_time is None else max(wall_time - accounted_total, 0.0)
    )

    return _NativeProfileSample(
        execution_mode=execution_mode,
        wall_time=wall_time,
        native_input_pack_time=native_input_pack_time,
        native_input_crossing_time=native_input_crossing_time,
        orchestration_time=orchestration_time,
        state_prepare_time=state_prepare_time,
        state_clear_time=state_clear_time,
        source_fill_time=source_fill_time,
        momentum_input_setup_time=momentum_input_setup_time,
        momentum_setup_time=momentum_setup_time,
        model_parameter_setup_time=model_parameter_setup_time,
        stage_input_pack_time=stage_input_pack_time,
        stage_leaf_input_pack_time=stage_leaf_input_pack_time,
        stage_evaluator_call_time=stage_evaluator_call_time,
        stage_backend_call_time=stage_backend_call_time,
        stage_evaluator_output_gather_time=stage_evaluator_output_gather_time,
        output_assign_time=output_assign_time,
        amplitude_input_pack_time=amplitude_input_pack_time,
        amplitude_evaluator_call_time=amplitude_evaluator_call_time,
        amplitude_leaf_input_pack_time=amplitude_leaf_input_pack_time,
        amplitude_backend_call_time=amplitude_backend_call_time,
        amplitude_evaluator_output_gather_time=(amplitude_evaluator_output_gather_time),
        amplitude_output_remap_time=amplitude_output_remap_time,
        reduction_time=reduction_time,
        total_materialization_time=total_materialization_time,
        final_output_copy_time=final_output_copy_time,
        selector_planner_time=selector_planner_time,
        selector_gather_time=selector_gather_time,
        selector_scatter_time=selector_scatter_time,
        other_core_time=other_core_time,
        stage_input_pack_times=(
            normalized_sequence(stage_input_pack) if compiled_profile else None
        ),
        stage_leaf_input_pack_times=(
            normalized_sequence(stage_leaf_input_pack) if compiled_profile else None
        ),
        stage_evaluator_call_times=(
            normalized_sequence(stage_evaluator_call) if compiled_profile else None
        ),
        stage_backend_call_times=(
            normalized_sequence(stage_backend_call) if compiled_profile else None
        ),
        stage_evaluator_output_gather_times=normalized_sequence(
            stage_evaluator_output_gather if compiled_profile else None
        ),
        stage_output_assign_times=(
            normalized_sequence(stage_output_assign) if compiled_profile else None
        ),
        eager_initialize_time=eager_initialize_time,
        eager_gather_time=eager_gather_time,
        eager_kernel_call_time=eager_kernel_call_time,
        eager_invocation_scatter_time=eager_invocation_scatter_time,
        eager_finalization_time=eager_finalization_time,
        eager_scatter_finalization_time=eager_scatter_finalization_time,
        eager_closure_time=eager_closure_time,
        eager_copy_out_time=eager_copy_out_time,
        recurrence_momentum_fill_time=recurrence_momentum_fill_time,
        recurrence_union_source_fill_time=recurrence_union_source_fill_time,
        recurrence_schedule_time=recurrence_schedule_time,
        recurrence_source_kernel_time=recurrence_source_kernel_time,
        recurrence_contribution_kernel_time=recurrence_contribution_kernel_time,
        recurrence_finalization_time=recurrence_finalization_time,
        recurrence_closure_time=recurrence_closure_time,
        recurrence_replay_output_mapping_time=(
            recurrence_replay_output_mapping_time
        ),
        counters=counters,
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


def _profile_counter_summary(
    samples: Sequence[_NativeProfileSample],
) -> BenchmarkProfileCounters | None:
    counter_samples = [
        sample.counters for sample in samples if sample.counters is not None
    ]
    if not counter_samples:
        return None
    if len(counter_samples) != len(samples):
        raise EvaluationError(
            "native runtime profile counter availability changed between samples"
        )

    def mean(attribute: str) -> float | None:
        values = [
            cast(float | None, getattr(sample, attribute)) for sample in counter_samples
        ]
        available = [value for value in values if value is not None]
        if available and len(available) != len(values):
            raise EvaluationError(
                f"native runtime profile counter {attribute!r} changed availability"
            )
        return statistics.fmean(available) if available else None

    return BenchmarkProfileCounters(
        sample_count=len(counter_samples),
        native_input_components_per_point=mean("native_input_components_per_point"),
        native_input_pack_bytes_per_point=mean("native_input_pack_bytes_per_point"),
        native_input_crossing_bytes_per_point=mean(
            "native_input_crossing_bytes_per_point"
        ),
        state_components_per_point=mean("state_components_per_point"),
        state_clear_components_per_point=mean("state_clear_components_per_point"),
        source_components_per_point=mean("source_components_per_point"),
        momentum_components_per_point=mean("momentum_components_per_point"),
        model_parameter_components_per_point=mean(
            "model_parameter_components_per_point"
        ),
        stage_input_copy_components_per_point=mean(
            "stage_input_copy_components_per_point"
        ),
        stage_leaf_input_copy_components_per_point=mean(
            "stage_leaf_input_copy_components_per_point"
        ),
        stage_evaluator_output_gather_components_per_point=mean(
            "stage_evaluator_output_gather_components_per_point"
        ),
        stage_output_assign_components_per_point=mean(
            "stage_output_assign_components_per_point"
        ),
        amplitude_input_copy_components_per_point=mean(
            "amplitude_input_copy_components_per_point"
        ),
        amplitude_leaf_input_copy_components_per_point=mean(
            "amplitude_leaf_input_copy_components_per_point"
        ),
        amplitude_evaluator_output_gather_components_per_point=mean(
            "amplitude_evaluator_output_gather_components_per_point"
        ),
        amplitude_output_remap_components_per_point=mean(
            "amplitude_output_remap_components_per_point"
        ),
        reduction_input_components_per_point=mean(
            "reduction_input_components_per_point"
        ),
        selector_gather_points_per_point=mean("selector_gather_points_per_point"),
        selector_gather_bytes_per_point=mean("selector_gather_bytes_per_point"),
        selector_scatter_values_per_point=mean("selector_scatter_values_per_point"),
        resolved_materialized_components_per_point=mean(
            "resolved_materialized_components_per_point"
        ),
        total_materialized_values_per_point=mean("total_materialized_values_per_point"),
        final_output_copy_values_per_point=mean("final_output_copy_values_per_point"),
        native_input_container_allocations_per_call=mean(
            "native_input_container_allocations_per_call"
        ),
        evaluator_backend_calls_per_call=mean("evaluator_backend_calls_per_call"),
        observed_scratch_reallocations_per_call=mean(
            "observed_scratch_reallocations_per_call"
        ),
        native_output_allocations_per_call=mean("native_output_allocations_per_call"),
    )


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
                sample.stage_leaf_input_pack_times,
                sample.stage_evaluator_call_times,
                sample.stage_backend_call_times,
                sample.stage_evaluator_output_gather_times,
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
        leaf_input_pack = _stage_component_timing(
            samples, "stage_leaf_input_pack_times", stage_index
        )
        backend_call = _stage_component_timing(
            samples, "stage_backend_call_times", stage_index
        )
        evaluator_output_gather = _stage_component_timing(
            samples, "stage_evaluator_output_gather_times", stage_index
        )
        output_assign = _stage_component_timing(
            samples, "stage_output_assign_times", stage_index
        )
        if any(
            value is not None
            for value in (
                input_pack,
                evaluator_call,
                output_assign,
                leaf_input_pack,
                backend_call,
                evaluator_output_gather,
            )
        ):
            stages.append(
                BenchmarkStageTiming(
                    stage_index=stage_index + 1,
                    input_pack_time=input_pack,
                    evaluator_call_time=evaluator_call,
                    output_assign_time=output_assign,
                    leaf_input_pack_time=leaf_input_pack,
                    backend_call_time=backend_call,
                    evaluator_output_gather_time=evaluator_output_gather,
                )
            )

    execution_modes = {
        sample.execution_mode for sample in samples if sample.execution_mode is not None
    }
    if len(execution_modes) > 1:
        raise EvaluationError("native runtime profile changed execution mode")
    execution_mode = cast(
        Literal["compiled", "eager", "recurrence"],
        next(iter(execution_modes), "compiled"),
    )
    evaluator_call_time = _component_timing(
        [sample.stage_evaluator_call_time for sample in samples]
    )
    return BenchmarkTimingBreakdown(
        sample_count=len(samples),
        execution_mode=execution_mode,
        wall_time=_component_timing([sample.wall_time for sample in samples]),
        native_input_pack_time=_component_timing(
            [sample.native_input_pack_time for sample in samples]
        ),
        native_input_crossing_time=_component_timing(
            [sample.native_input_crossing_time for sample in samples]
        ),
        orchestration_time=_component_timing(
            [sample.orchestration_time for sample in samples]
        ),
        state_prepare_time=_component_timing(
            [sample.state_prepare_time for sample in samples]
        ),
        state_clear_time=_component_timing(
            [sample.state_clear_time for sample in samples]
        ),
        source_fill_time=_component_timing(
            [sample.source_fill_time for sample in samples]
        ),
        momentum_setup_time=_component_timing(
            [sample.momentum_setup_time for sample in samples]
        ),
        momentum_input_setup_time=_component_timing(
            [sample.momentum_input_setup_time for sample in samples]
        ),
        model_parameter_setup_time=_component_timing(
            [sample.model_parameter_setup_time for sample in samples]
        ),
        stage_input_pack_time=_component_timing(
            [sample.stage_input_pack_time for sample in samples]
        ),
        stage_leaf_input_pack_time=_component_timing(
            [sample.stage_leaf_input_pack_time for sample in samples]
        ),
        stage_evaluator_call_time=(
            None
            if execution_mode in {"eager", "recurrence"}
            else evaluator_call_time
        ),
        stage_backend_call_time=_component_timing(
            [sample.stage_backend_call_time for sample in samples]
        ),
        stage_evaluator_output_gather_time=_component_timing(
            [sample.stage_evaluator_output_gather_time for sample in samples]
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
        amplitude_leaf_input_pack_time=_component_timing(
            [sample.amplitude_leaf_input_pack_time for sample in samples]
        ),
        amplitude_backend_call_time=_component_timing(
            [sample.amplitude_backend_call_time for sample in samples]
        ),
        amplitude_evaluator_output_gather_time=_component_timing(
            [sample.amplitude_evaluator_output_gather_time for sample in samples]
        ),
        amplitude_output_remap_time=_component_timing(
            [sample.amplitude_output_remap_time for sample in samples]
        ),
        reduction_time=_component_timing([sample.reduction_time for sample in samples]),
        total_materialization_time=_component_timing(
            [sample.total_materialization_time for sample in samples]
        ),
        final_output_copy_time=_component_timing(
            [sample.final_output_copy_time for sample in samples]
        ),
        selector_planner_time=_component_timing(
            [sample.selector_planner_time for sample in samples]
        ),
        selector_gather_time=_component_timing(
            [sample.selector_gather_time for sample in samples]
        ),
        selector_scatter_time=_component_timing(
            [sample.selector_scatter_time for sample in samples]
        ),
        other_core_time=_component_timing(
            [sample.other_core_time for sample in samples]
        ),
        eager_execution_time=(
            evaluator_call_time if execution_mode == "eager" else None
        ),
        eager_initialize_time=_component_timing(
            [sample.eager_initialize_time for sample in samples]
        ),
        eager_gather_time=_component_timing(
            [sample.eager_gather_time for sample in samples]
        ),
        eager_kernel_call_time=_component_timing(
            [sample.eager_kernel_call_time for sample in samples]
        ),
        eager_invocation_scatter_time=_component_timing(
            [sample.eager_invocation_scatter_time for sample in samples]
        ),
        eager_finalization_time=_component_timing(
            [sample.eager_finalization_time for sample in samples]
        ),
        eager_scatter_finalization_time=_component_timing(
            [sample.eager_scatter_finalization_time for sample in samples]
        ),
        eager_closure_time=_component_timing(
            [sample.eager_closure_time for sample in samples]
        ),
        eager_copy_out_time=_component_timing(
            [sample.eager_copy_out_time for sample in samples]
        ),
        recurrence_momentum_fill_time=_component_timing(
            [sample.recurrence_momentum_fill_time for sample in samples]
        ),
        recurrence_union_source_fill_time=_component_timing(
            [sample.recurrence_union_source_fill_time for sample in samples]
        ),
        recurrence_schedule_time=_component_timing(
            [sample.recurrence_schedule_time for sample in samples]
        ),
        recurrence_source_kernel_time=_component_timing(
            [sample.recurrence_source_kernel_time for sample in samples]
        ),
        recurrence_contribution_kernel_time=_component_timing(
            [sample.recurrence_contribution_kernel_time for sample in samples]
        ),
        recurrence_finalization_time=_component_timing(
            [sample.recurrence_finalization_time for sample in samples]
        ),
        recurrence_closure_time=_component_timing(
            [sample.recurrence_closure_time for sample in samples]
        ),
        recurrence_replay_output_mapping_time=_component_timing(
            [sample.recurrence_replay_output_mapping_time for sample in samples]
        ),
        stages=tuple(stages),
        counters=_profile_counter_summary(samples),
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
