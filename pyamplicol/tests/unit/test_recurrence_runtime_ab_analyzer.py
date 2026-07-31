# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from tools.developer import recurrence_generation_ab_ladder as ladder
from tools.developer import recurrence_runtime_ab_analyzer as analyzer
from tools.developer import recurrence_z6g_benchmark as harness


def _runtime_peak(value: int) -> dict[str, object]:
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": value,
        "maximum_child_peak_bytes": 0,
        "observed_lower_bound_bytes": value,
        "semantics": "profile worker process high-water",
    }


def _outer_sample(
    *,
    pair_index: int,
    variant: str,
    order_in_pair: int,
    timing_ratios: Sequence[float],
    include_runtime_rss: bool,
    rss_ratio: float,
) -> dict[str, object]:
    workers = []
    for round_index, ratio in enumerate(timing_ratios):
        baseline_timing = 1.0e-6 * (1.0 + round_index / 100.0)
        timing = baseline_timing if variant == "baseline" else baseline_timing * ratio
        baseline_rss = 1_000_000_000 + round_index * 1_000_000
        rss = baseline_rss if variant == "baseline" else round(baseline_rss * rss_ratio)
        worker: dict[str, object] = {
            "schedule_index": round_index,
            "round": round_index,
            "wall_seconds_per_point": timing,
            "internal_sample_count": 7,
            "repetitions_per_sample": 3,
            "evaluation_count": 21,
            "evaluated_point_count": 21,
            "interrupted": False,
            "worker_wall_seconds": 5.1,
        }
        if include_runtime_rss:
            worker["peak_rss_after_profile"] = _runtime_peak(rss)
        workers.append(worker)
    return {
        "sample_id": f"pair-{pair_index}-{variant}",
        "pair_index": pair_index,
        "order_in_pair": order_in_pair,
        "variant": variant,
        "multiplicity": 6,
        "layout": "topology-replay",
        "runtime_enabled": True,
        "status": "passed",
        # These records must never be accepted as runtime-only memory evidence.
        "watchdog": {"peak_rss": {"bytes_rounded_from_watchdog": 123}},
        "telemetry": {
            "generation_peak_rss": {"observed_lower_bound_bytes": 456},
            "runtime_profile": {
                "measurements": [
                    {
                        "batch_size": 1,
                        "sample_count": len(workers),
                        "wall_seconds_per_point": 1.0e-6,
                        "wall_seconds_per_point_median": 1.0e-6,
                        "wall_seconds_per_point_mad": 0.0,
                        "interrupted": False,
                        "subprocess_samples": workers,
                    }
                ]
            },
        },
    }


def _campaign(
    *,
    pair_count: int = 1,
    ratio_for_observation: Callable[[int], float] = lambda _index: 0.98,
    include_runtime_rss: bool = True,
    rss_ratio: float = 1.0,
) -> dict[str, object]:
    samples = []
    observation_index = 0
    for pair_index in range(pair_count):
        ratios = []
        for _round_index in range(7):
            ratios.append(ratio_for_observation(observation_index))
            observation_index += 1
        order = (
            ("baseline", "candidate")
            if pair_index % 2 == 0
            else ("candidate", "baseline")
        )
        for order_in_pair, variant in enumerate(order):
            samples.append(
                _outer_sample(
                    pair_index=pair_index,
                    variant=variant,
                    order_in_pair=order_in_pair,
                    timing_ratios=ratios,
                    include_runtime_rss=include_runtime_rss,
                    rss_ratio=rss_ratio,
                )
            )
    return {
        "kind": analyzer.CAMPAIGN_KIND,
        "schema_version": analyzer.CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": "runtime-test",
        "configuration": {
            "multiplicities": [6],
            "layouts": ["topology-replay"],
            "runtime_multiplicities": [6],
            "batch_sizes": [1],
            "warmup_runs": 2,
            "target_runtime_seconds": 5.0,
            "minimum_samples": 7,
            "subprocess_samples": 7,
            "ordering_policy": "alternating-baseline-candidate-pairs-v1",
            "cold_cache_policy": "unique-roots-per-outer-sample-v1",
        },
        "samples": samples,
    }


def _analyze(campaign: dict[str, Any]) -> dict[str, object]:
    return analyzer.analyze_campaign(
        campaign,
        multiplicities=(6,),
        layouts=("topology-replay",),
        batch_sizes=(1,),
    )


def test_paired_log_interval_and_seven_sample_pass_are_deterministic() -> None:
    interval = analyzer.paired_log_ratio_interval([1.0] * 7, [0.98] * 7)
    assert interval["sample_count"] == 7
    assert interval["candidate_over_baseline_geometric_mean"] == pytest.approx(0.98)
    assert interval["candidate_over_baseline_upper_confidence_bound"] == pytest.approx(
        0.98
    )

    result = _analyze(_campaign())
    assert result["status"] == "passed"
    assert result["passes"] is True
    cell = result["cells"][0]
    assert cell["paired_sample_count"] == 7
    assert cell["timing"]["gate"]["status"] == "passed"
    assert cell["runtime_rss"]["gate"]["status"] == "passed"


def test_inconclusive_seven_samples_requests_expansion_to_twenty_one() -> None:
    result = _analyze(
        _campaign(ratio_for_observation=lambda index: 0.99 if index % 2 == 0 else 1.01)
    )
    cell = result["cells"][0]
    assert result["status"] == "needs-more-samples"
    assert cell["timing"]["gate"]["status"] == "needs-more-samples"
    assert cell["timing"]["gate"]["additional_paired_samples_allowed"] == 14


def test_inconclusive_twenty_one_samples_rejects_runtime_change() -> None:
    result = _analyze(
        _campaign(
            pair_count=3,
            ratio_for_observation=lambda index: 0.99 if index % 2 == 0 else 1.01,
        )
    )
    cell = result["cells"][0]
    assert result["status"] == "rejected"
    assert cell["paired_sample_count"] == 21
    assert cell["timing"]["gate"]["status"] == "rejected-inconclusive"
    assert cell["timing"]["gate"]["additional_paired_samples_allowed"] == 0


def test_statistically_supported_slowdown_rejects_immediately() -> None:
    result = _analyze(_campaign(ratio_for_observation=lambda _index: 1.02))
    cell = result["cells"][0]
    assert result["status"] == "rejected"
    assert cell["timing"]["gate"]["status"] == "rejected-supported-slowdown"


def test_runtime_rss_gate_ignores_generation_and_outer_watchdog_memory() -> None:
    result = _analyze(_campaign(include_runtime_rss=False))
    cell = result["cells"][0]
    assert result["status"] == "incomplete-runtime-rss"
    assert cell["runtime_rss"]["interval"] is None
    assert cell["runtime_rss"]["gate"]["status"] == "missing-runtime-only-rss"


def test_supported_material_runtime_rss_increase_rejects() -> None:
    result = _analyze(_campaign(rss_ratio=1.20))
    cell = result["cells"][0]
    assert result["status"] == "rejected"
    assert (
        cell["runtime_rss"]["gate"]["status"]
        == "rejected-supported-material-rss-increase"
    )


def test_outer_pair_order_must_alternate() -> None:
    campaign = _campaign(pair_count=3)
    samples = campaign["samples"]
    assert isinstance(samples, list)
    pair_one = [
        sample
        for sample in samples
        if isinstance(sample, dict) and sample["pair_index"] == 1
    ]
    for sample in pair_one:
        sample["order_in_pair"] = 0 if sample["variant"] == "baseline" else 1
    with pytest.raises(analyzer.RuntimeABError, match="did not alternate"):
        _analyze(campaign)


def test_runtime_rss_survives_harness_aggregation_and_campaign_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = harness._build_profile_schedule(
        ("recurrence",),
        (1,),
        subprocess_samples=1,
    )
    semantic_identity: dict[str, object] = {"test": "identity"}
    semantic_sha256 = harness._canonical_sha256(semantic_identity)
    peak = _runtime_peak(1_234_567_890)
    worker: dict[str, object] = {
        "mode": "recurrence",
        "schedule_index": 0,
        "schedule_round": 0,
        "process_id": "process",
        "process_expression": "d d~ > Z g g g g g",
        "selector_contract": {},
        "validation": {},
        "pre_timing_verification": {
            "artifact_semantic_identity": semantic_identity,
            "artifact_semantic_identity_sha256": semantic_sha256,
        },
        "post_timing_loaded_runtime_artifact": {},
        "worker_command": {},
        "worker_invocation": {},
        "worker_process_record": {},
        "worker_result_record": {},
        "peak_rss_after_cold_load": _runtime_peak(1_000_000_000),
        "peak_rss_after_profile": peak,
        "timing_configuration": {},
        "profiles": [
            {
                "batch_size": 1,
                "sample_count": 7,
                "repetitions_per_sample": 1,
                "evaluation_count": 7,
                "evaluated_point_count": 7,
                "wall_seconds_per_point": 1.0e-6,
                "inner_native_wall_blocks": {},
                "timing_sources": {},
                "environment": {},
                "interrupted": False,
            }
        ],
    }
    monkeypatch.setattr(
        harness,
        "_retained_profile_worker_result_record",
        lambda *_args, **_kwargs: {
            "addressed_payload_sha256": "a" * 64,
        },
    )
    profiles = harness._aggregate_profile_workers(schedule, [worker])
    aggregated = profiles["recurrence"]["profiles"][0]["subprocess_samples"][0]
    assert aggregated["peak_rss_after_profile"] == peak

    compact = ladder._runtime_profile_summary(
        {
            "process_id": "process",
            "process_expression": "d d~ > Z g g g g g",
            "profiles": profiles["recurrence"]["profiles"],
        }
    )
    assert compact is not None
    compact_worker = compact["measurements"][0]["subprocess_samples"][0]
    assert compact_worker["peak_rss_after_profile"] == peak
