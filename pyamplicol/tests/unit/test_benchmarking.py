# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

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


class _Runtime:
    calls = 0

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
        del selectors
        self.calls += 1
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
            "stage_evaluator_time_s": 2.0e-6,
            "amplitude_evaluator_time_s": 6.0e-6,
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
    assert runtime.calls == 0
    assert runtime.profile_calls == 4
    assert result.wall_time_per_point == pytest.approx(3.0e-6)
    assert result.evaluator_time_per_point == pytest.approx(2.0e-6)
    assert result.environment["wall_time_source"] == "runtime_profile_wall_time"
    assert (
        result.environment["evaluator_time_source"]
        == "runtime_profile_core_evaluator_time"
    )
    assert result.environment["evaluator_sample_count"] == 3


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
