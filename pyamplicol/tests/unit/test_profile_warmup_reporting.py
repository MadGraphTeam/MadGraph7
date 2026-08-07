# SPDX-License-Identifier: 0BSD
"""Focused public-profile warm-up timing and presentation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol import benchmarking
from pyamplicol.config import BenchmarkConfig
from pyamplicol.reporting.summary import render_summary
from tools.performance_report.runner import (
    CONVENTIONAL_WARMUP_FIELDS,
    CONVENTIONAL_WARMUP_TIMING_SCOPE,
    OTF_COLD_WARMUP_FIELDS,
    OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
    OTF_COLD_WARMUP_TIMING_SCOPE,
    WARMUP_TIMER_SOURCE,
    RunnerError,
    _benchmark_measurement,
)


class _OnTheFlyRuntime:
    execution_mode = "on-the-fly"

    def __init__(self) -> None:
        self.evaluate_calls: list[tuple[int, object, object, int]] = []

    @property
    def physics(self) -> object:
        return SimpleNamespace()

    def _on_the_fly_benchmark_context(
        self, requested: tuple[str, ...]
    ) -> dict[str, object]:
        assert requested == ()
        return {
            "process_id": "otf_test",
            "process_expression": "d d~ > z",
            "color_accuracy": "lc",
            "helicity_count": 2,
            "color_count": 1,
            "selected_color_ids": [],
        }

    def evaluate(
        self,
        momenta: object,
        *,
        helicities: object = None,
        color_flows: object = None,
        precision: int = 16,
    ) -> tuple[complex, ...]:
        batch = tuple(momenta)  # type: ignore[arg-type]
        self.evaluate_calls.append((len(batch), helicities, color_flows, precision))
        return tuple(1.0 + 0.0j for _ in batch)

    def evaluate_resolved(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("resolved evaluation is not part of profiling")

    def set_model_parameters(self, _mapping: object) -> None:
        return None

    def clear(self) -> None:
        return None

    def mute_warnings(self) -> None:
        return None

    def unmute_warnings(self) -> None:
        return None


def test_otf_initial_preparation_and_conventional_warmups_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(
        (
            0.000,
            0.000,
            0.010,
            0.012,
            0.020,
            0.021,
            0.025,
            0.026,
            0.030,
            0.031,
            0.034,
            0.035,
            0.040,
            0.041,
            0.050,
            0.051,
            0.053,
            0.054,
        )
    )
    monkeypatch.setattr(benchmarking.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        benchmarking,
        "_calibrate_repetitions",
        lambda *_args, **_kwargs: benchmarking._Calibration(
            sample_count=1,
            repetitions_per_sample=1,
            target_sample_seconds=0.001,
            probe_seconds=0.004,
            block_count=0,
            evaluation_count=0,
            elapsed_seconds=0.0,
        ),
    )
    config = BenchmarkConfig(
        target_runtime=0.001,
        batch_size=3,
        precision=8,
        warmup_runs=2,
        minimum_samples=1,
    )
    runtime = _OnTheFlyRuntime()
    result = benchmarking.BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    environment = result.environment
    assert environment["cold_warmup_elapsed_seconds"] == pytest.approx(0.012)
    assert environment["cold_warmup_run_count"] == 1
    assert environment["cold_warmup_batch_size"] == 3
    assert environment["cold_warmup_point_count"] == 3
    assert environment["cold_warmup_runtime_freshness"] == (
        "not-authenticated-by-benchmark"
    )
    assert environment["cold_warmup_ratio_eligible"] is False
    assert environment["cold_warmup_acceptance_eligible"] is False
    assert environment["cold_warmup_timer_source"] == WARMUP_TIMER_SOURCE
    assert environment["cold_warmup_timing_scope"] == OTF_COLD_WARMUP_TIMING_SCOPE
    assert "Runtime/artifact load are excluded" in str(
        environment["cold_warmup_timing_scope"]
    )

    assert environment["warmup_elapsed_seconds"] == pytest.approx(0.011)
    assert environment["warmup_configured_run_count"] == 2
    assert environment["warmup_batch_size"] == 3
    assert environment["warmup_point_count"] == 6
    assert environment["warmup_run_outer_wall_seconds"] == pytest.approx((0.006, 0.005))
    assert environment["first_warmup_run_outer_wall_seconds"] == pytest.approx(0.006)
    assert environment["warmup_timer_source"] == WARMUP_TIMER_SOURCE
    assert environment["warmup_timing_scope"] == CONVENTIONAL_WARMUP_TIMING_SCOPE
    assert len(runtime.evaluate_calls) == 4
    assert result.wall_time_per_point == pytest.approx(0.004 / 3.0)
    assert result.evaluator_total_time_per_point == pytest.approx(0.002 / 3.0)

    rendered = render_summary(result, color=True)
    assert rendered is not None
    assert "OTF initial preparation" in rendered
    assert "12 ms" in rendered
    assert "runtime freshness not authenticated here" in rendered
    assert "warm-up wall (total)" in rendered
    assert "11 ms" in rendered
    assert "first warm-up run" in rendered
    assert "6 ms" in rendered
    assert "\x1b[" in rendered


def _complete_otf_profile_environment() -> dict[str, object]:
    return {
        "execution_mode": "on-the-fly",
        "batch_size": 128,
        "evaluator_time_raw_seconds_per_point": 8.0e-7,
        "evaluator_time_status": "measured",
        "evaluator_time_ratio_eligible": True,
        "evaluator_time_source": "runtime_profile_core_recurrence_schedule_time",
        "native_profile_points_per_sample": 128,
        "timing_sample_contract": "paired-native-repeated-profile-v1",
        "elapsed_seconds": 1.0,
        "cold_warmup_elapsed_seconds": 4.767252833,
        "cold_warmup_run_count": 1,
        "cold_warmup_batch_size": 128,
        "cold_warmup_point_count": 128,
        "cold_warmup_timer_source": WARMUP_TIMER_SOURCE,
        "cold_warmup_timing_scope": OTF_COLD_WARMUP_TIMING_SCOPE,
        "cold_warmup_runtime_freshness": OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
        "cold_warmup_ratio_eligible": False,
        "cold_warmup_acceptance_eligible": False,
        "warmup_elapsed_seconds": 0.021,
        "warmup_configured_run_count": 2,
        "warmup_batch_size": 128,
        "warmup_point_count": 256,
        "warmup_run_outer_wall_seconds": (0.010, 0.011),
        "first_warmup_run_outer_wall_seconds": 0.010,
        "warmup_timer_source": WARMUP_TIMER_SOURCE,
        "warmup_timing_scope": CONVENTIONAL_WARMUP_TIMING_SCOPE,
    }


def _complete_otf_benchmark() -> SimpleNamespace:
    return SimpleNamespace(
        uncertainty=SimpleNamespace(
            standard_error=1.0e-8,
            relative_standard_error=0.01,
        ),
        environment=_complete_otf_profile_environment(),
        evaluator_time_per_point=8.0e-7,
        wall_time_per_point=1.0e-6,
        sample_count=5,
        effective_config=SimpleNamespace(
            target_runtime=1.0,
            batch_size=128,
            warmup_runs=2,
        ),
    )


def test_campaign_preserves_complete_cold_and_conventional_warmup_evidence() -> None:
    measurement = _benchmark_measurement(
        _complete_otf_benchmark(),
        matrix_element=2.0,
    )

    evidence = measurement["benchmark_evidence"]
    expected = OTF_COLD_WARMUP_FIELDS | CONVENTIONAL_WARMUP_FIELDS
    assert expected <= evidence.keys()
    source = _complete_otf_profile_environment()
    assert {field: evidence[field] for field in expected} == {
        field: source[field] for field in expected
    }


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_private_arena_historical_single_warmup_elapsed_remains_valid(
    execution_mode: str,
) -> None:
    benchmark = _complete_otf_benchmark()
    benchmark.environment["execution_mode"] = execution_mode
    for field in OTF_COLD_WARMUP_FIELDS | CONVENTIONAL_WARMUP_FIELDS:
        benchmark.environment.pop(field, None)
    benchmark.environment["warmup_elapsed_seconds"] = 0.2

    measurement = _benchmark_measurement(benchmark, matrix_element=2.0)

    evidence = measurement["benchmark_evidence"]
    assert evidence["warmup_elapsed_seconds"] == pytest.approx(0.2)
    assert (
        not (
            (OTF_COLD_WARMUP_FIELDS | CONVENTIONAL_WARMUP_FIELDS)
            - {"warmup_elapsed_seconds"}
        )
        & evidence.keys()
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "cold_warmup_acceptance_eligible",
            True,
            "ineligible for ratios and acceptance",
        ),
        ("cold_warmup_point_count", 127, "full benchmark batch"),
        ("warmup_point_count", 255, "configured runs times batch size"),
        ("warmup_run_outer_wall_seconds", (0.010,), "configured run count"),
        ("warmup_elapsed_seconds", 2.0, "does not equal its per-run"),
        ("cold_warmup_timing_scope", "load included", "scope must exclude"),
    ),
)
def test_campaign_rejects_semantically_invalid_otf_warmup_evidence(
    field: str,
    replacement: object,
    message: str,
) -> None:
    benchmark = _complete_otf_benchmark()
    benchmark.environment[field] = replacement

    with pytest.raises(RunnerError, match=message):
        _benchmark_measurement(benchmark, matrix_element=2.0)


@pytest.mark.parametrize(
    "field",
    ("cold_warmup_run_count", "warmup_timer_source"),
)
def test_campaign_rejects_incomplete_otf_warmup_evidence(field: str) -> None:
    benchmark = _complete_otf_benchmark()
    del benchmark.environment[field]

    with pytest.raises(RunnerError, match=r"incomplete .* warm-up evidence"):
        _benchmark_measurement(benchmark, matrix_element=2.0)
