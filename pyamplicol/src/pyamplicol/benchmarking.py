# SPDX-License-Identifier: 0BSD
"""Typed wall-clock benchmarking for Rusticol runtime backends."""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from pathlib import Path

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
        task_id = "runtime-benchmark"
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    "Benchmarking runtime",
                    total=None,
                )
            )
        try:
            for _ in range(self._config.warmup_runs):
                runtime.evaluate(
                    batch,
                    helicities=helicities,
                    color_flows=color_flows,
                )
            samples: list[float] = []
            elapsed = 0.0
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

        mean = statistics.fmean(samples)
        deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
        error = deviation / math.sqrt(len(samples))
        relative_error = error / mean if mean > 0.0 else 0.0
        return BenchmarkResult(
            requested_config=self._config,
            effective_config=self._config,
            sample_count=len(samples),
            wall_time_per_point=mean,
            evaluator_time_per_point=None,
            uncertainty=BenchmarkStatistics(deviation, error, relative_error),
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "batch_size": len(batch),
                "elapsed_seconds": elapsed,
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


def create_benchmark_backend(
    config: BenchmarkConfig | RunConfig | None,
    progress: ProgressSink | None,
) -> BenchmarkBackend:
    return BenchmarkBackend(config, progress)


__all__ = ["BenchmarkBackend", "create_benchmark_backend"]
