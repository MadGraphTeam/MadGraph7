# SPDX-License-Identifier: 0BSD
"""Typed wall-clock benchmarking for Rusticol runtime backends."""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.protocols import Momenta, RuntimeBackend
from pyamplicol.api.results import BenchmarkResult, BenchmarkStatistics
from pyamplicol.config import BenchmarkConfig, RunConfig
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)


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
        profiler = _native_profiler(runtime)
        task_id = "runtime-benchmark"
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    "Benchmarking runtime",
                    total=None,
                )
            )
        evaluator_samples: list[float] | None = None
        evaluator_environment: dict[str, object]
        try:
            samples: list[float] = []
            elapsed = 0.0
            if profiler is None:
                for _ in range(self._config.warmup_runs):
                    runtime.evaluate(
                        batch,
                        helicities=helicities,
                        color_flows=color_flows,
                    )
                while (
                    len(samples) < self._config.minimum_samples
                    or elapsed < self._config.target_runtime
                ):
                    started = time.perf_counter()
                    runtime.evaluate(
                        batch,
                        helicities=helicities,
                        color_flows=color_flows,
                    )
                    duration = time.perf_counter() - started
                    elapsed += duration
                    samples.append(duration / len(batch))
                    if self._progress is not None:
                        self._progress.emit(
                            ProgressUpdate(
                                task_id,
                                completed=len(samples),
                                total=None,
                                message=f"{elapsed:.3g}s sampled",
                            )
                        )
                evaluator_environment = {
                    "wall_time_source": "runtime_evaluate_wall_time",
                    "evaluator_time_source": "runtime_evaluate_wall_time",
                }
            else:
                for _ in range(self._config.warmup_runs):
                    profiler(
                        batch,
                        helicities=helicities,
                        color_flows=color_flows,
                        precision=16,
                        include_values=False,
                    )
                evaluator_samples = []
                while (
                    len(samples) < self._config.minimum_samples
                    or elapsed < self._config.target_runtime
                ):
                    started = time.perf_counter()
                    profile = profiler(
                        batch,
                        helicities=helicities,
                        color_flows=color_flows,
                        precision=16,
                        include_values=False,
                    )
                    elapsed += time.perf_counter() - started
                    points_in_profile = _profile_point_count(profile, len(batch))
                    samples.append(
                        _profile_float(profile, "wall_time_s") / points_in_profile
                    )
                    evaluator_samples.append(
                        _profile_core_evaluator_seconds(profile) / points_in_profile
                    )
                    if self._progress is not None:
                        self._progress.emit(
                            ProgressUpdate(
                                task_id,
                                completed=len(samples),
                                total=None,
                                message=f"{elapsed:.3g}s profiled",
                            )
                        )
                (
                    evaluator_mean,
                    evaluator_deviation,
                    evaluator_error,
                    evaluator_relative,
                ) = _sample_statistics(evaluator_samples)
                evaluator_environment = {
                    "wall_time_source": "runtime_profile_wall_time",
                    "evaluator_time_source": "runtime_profile_core_evaluator_time",
                    "evaluator_sample_count": len(evaluator_samples),
                    "evaluator_elapsed_seconds": elapsed,
                    "evaluator_standard_deviation_seconds_per_point": (
                        evaluator_deviation
                    ),
                    "evaluator_standard_error_seconds_per_point": evaluator_error,
                    "evaluator_relative_standard_error": evaluator_relative,
                }
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(task_id, success=False, message=str(exc))
                )
            if isinstance(exc, EvaluationError):
                raise
            raise EvaluationError(f"runtime benchmark failed: {exc}") from exc
        if self._progress is not None:
            self._progress.emit(ProgressEnd(task_id))

        mean, deviation, error, relative_error = _sample_statistics(samples)
        evaluator_time_per_point = (
            mean if evaluator_samples is None else evaluator_mean
        )
        return BenchmarkResult(
            requested_config=self._config,
            effective_config=self._config,
            sample_count=len(samples),
            wall_time_per_point=mean,
            evaluator_time_per_point=evaluator_time_per_point,
            uncertainty=BenchmarkStatistics(deviation, error, relative_error),
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "batch_size": len(batch),
                "elapsed_seconds": elapsed,
                **evaluator_environment,
            },
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


def _native_profiler(
    runtime: RuntimeBackend,
) -> Callable[..., Mapping[str, object]] | None:
    for name in ("profile", "evaluate_profile"):
        profiler = getattr(runtime, name, None)
        if callable(profiler):
            return profiler
    return None


def _profile_float(profile: Mapping[str, object], key: str) -> float:
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvaluationError(f"native runtime profile field {key!r} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise EvaluationError(f"native runtime profile field {key!r} is invalid")
    return value


def _profile_point_count(profile: Mapping[str, object], fallback: int) -> int:
    value: Any = profile.get("points", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError("native runtime profile point count is invalid")
    return value


def _profile_core_evaluator_seconds(profile: Mapping[str, object]) -> float:
    return _profile_float(profile, "stage_evaluator_time_s") + _profile_float(
        profile,
        "amplitude_evaluator_time_s",
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
