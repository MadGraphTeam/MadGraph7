#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Analyze the recurrence runtime no-regression A/B gate.

The input is a ``recurrence_generation_ab_ladder.py`` campaign.  Timing
observations are paired by outer A/B pair and inner subprocess round, so host
drift is not silently treated as independent noise.  The candidate/baseline
ratio is analyzed in log space with a one-sided 95% Student-t bound.

This analyzer deliberately accepts only runtime-worker RSS evidence.  The
generation worker and outer watchdog both cover artifact generation and cannot
establish runtime-only memory use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_PARENT = ROOT / ".artifacts" / "recurrence-generation-opt"

CAMPAIGN_KIND = "pyamplicol-recurrence-generation-ab-ladder"
CAMPAIGN_SCHEMA_VERSION = 1
RESULT_KIND = "pyamplicol-recurrence-runtime-ab-analysis"
RESULT_SCHEMA_VERSION = 1
LAYOUTS = ("topology-replay", "all-flow-union")
DEFAULT_MULTIPLICITIES = (6, 7)
DEFAULT_BATCH_SIZES = (1, 128, 1024)
MINIMUM_PAIRED_SAMPLES = 7
MAXIMUM_PAIRED_SAMPLES = 21
CONFIDENCE_LEVEL = 0.95
NO_SLOWDOWN_RATIO = 1.0
DEFAULT_MATERIAL_RSS_RATIO = 1.10

# One-sided 95% Student-t quantiles.  The gate admits 7..21 paired
# observations, so only df=6..20 are reachable.  Values are fixed here to
# avoid introducing a scientific-library dependency into the developer tool.
_ONE_SIDED_T_95 = {
    6: 1.9431802805153022,
    7: 1.894578605061305,
    8: 1.8595480375228424,
    9: 1.8331129326536337,
    10: 1.8124611228107335,
    11: 1.7958848187036691,
    12: 1.782287555649159,
    13: 1.7709333959867988,
    14: 1.7613101357748562,
    15: 1.7530503556925547,
    16: 1.74588367627624,
    17: 1.7396067260750672,
    18: 1.734063606617536,
    19: 1.729132811521367,
    20: 1.7247182429207863,
}


class RuntimeABError(RuntimeError):
    """Raised when runtime A/B evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class WorkerObservation:
    """One baseline/candidate worker pair from the same schedule round."""

    pair_index: int
    round_index: int
    baseline_order_in_pair: int
    candidate_order_in_pair: int
    baseline_seconds_per_point: float
    candidate_seconds_per_point: float
    baseline_runtime_peak_rss_bytes: int | None
    candidate_runtime_peak_rss_bytes: int | None


def _mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeABError(f"{description} must be a JSON object")
    return value


def _list(value: object, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeABError(f"{description} must be a JSON array")
    return value


def _positive_int(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeABError(f"{description} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeABError(f"{description} must be a nonnegative integer")
    return value


def _positive_float(value: object, *, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeABError(f"{description} must be a positive finite number")
    return float(value)


def _runtime_peak_rss_bytes(sample: Mapping[str, Any]) -> int | None:
    """Return only the profile worker's post-runtime high-water evidence."""

    normalized = sample.get("runtime_peak_rss_bytes")
    peak = sample.get("peak_rss_after_profile")
    observed: int | None = None
    if normalized is not None:
        observed = _positive_int(
            normalized,
            description="runtime_peak_rss_bytes",
        )
    if peak is not None:
        record = _mapping(
            peak,
            description="profile worker peak_rss_after_profile",
        )
        source = record.get("source")
        if source != "resource.getrusage":
            raise RuntimeABError("profile worker RSS must come from resource.getrusage")
        record_observed = _positive_int(
            record.get("observed_lower_bound_bytes"),
            description=("profile worker peak_rss_after_profile observed lower bound"),
        )
        if observed is not None and observed != record_observed:
            raise RuntimeABError("normalized and structured runtime RSS disagree")
        observed = record_observed
    return observed


def paired_log_ratio_interval(
    baseline: Sequence[float | int],
    candidate: Sequence[float | int],
) -> dict[str, float | int | str]:
    """Return a deterministic paired one-sided 95% log-ratio interval."""

    if len(baseline) != len(candidate):
        raise RuntimeABError("baseline and candidate sample counts differ")
    sample_count = len(baseline)
    if not MINIMUM_PAIRED_SAMPLES <= sample_count <= MAXIMUM_PAIRED_SAMPLES:
        raise RuntimeABError(
            "paired sample count must remain between 7 and 21 inclusive"
        )
    log_ratios: list[float] = []
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for index, (baseline_value, candidate_value) in enumerate(
        zip(baseline, candidate, strict=True)
    ):
        baseline_float = _positive_float(
            baseline_value,
            description=f"baseline observation {index}",
        )
        candidate_float = _positive_float(
            candidate_value,
            description=f"candidate observation {index}",
        )
        baseline_values.append(baseline_float)
        candidate_values.append(candidate_float)
        log_ratios.append(math.log(candidate_float / baseline_float))

    mean_log_ratio = statistics.fmean(log_ratios)
    standard_deviation = statistics.stdev(log_ratios)
    standard_error = standard_deviation / math.sqrt(sample_count)
    critical = _ONE_SIDED_T_95[sample_count - 1]
    half_width = critical * standard_error
    return {
        "statistics_contract": ("paired-log-ratio-student-t-one-sided-95-v1"),
        "confidence_level": CONFIDENCE_LEVEL,
        "sample_count": sample_count,
        "degrees_of_freedom": sample_count - 1,
        "student_t_critical": critical,
        "candidate_over_baseline_geometric_mean": math.exp(mean_log_ratio),
        "candidate_over_baseline_lower_confidence_bound": math.exp(
            mean_log_ratio - half_width
        ),
        "candidate_over_baseline_upper_confidence_bound": math.exp(
            mean_log_ratio + half_width
        ),
        "baseline_median": statistics.median(baseline_values),
        "candidate_median": statistics.median(candidate_values),
        "candidate_over_baseline_median_ratio": (
            statistics.median(candidate_values) / statistics.median(baseline_values)
        ),
        "log_ratio_sample_standard_deviation": standard_deviation,
        "log_ratio_standard_error": standard_error,
    }


def _gate_decision(
    interval: Mapping[str, object],
    *,
    allowed_ratio: float,
    failure_label: str,
) -> dict[str, object]:
    sample_count = _positive_int(
        interval.get("sample_count"),
        description="confidence interval sample count",
    )
    upper = _positive_float(
        interval.get("candidate_over_baseline_upper_confidence_bound"),
        description="candidate/baseline upper confidence bound",
    )
    lower = _positive_float(
        interval.get("candidate_over_baseline_lower_confidence_bound"),
        description="candidate/baseline lower confidence bound",
    )
    if upper <= allowed_ratio:
        status = "passed"
    elif lower > allowed_ratio:
        status = failure_label
    elif sample_count < MAXIMUM_PAIRED_SAMPLES:
        status = "needs-more-samples"
    else:
        status = "rejected-inconclusive"
    return {
        "status": status,
        "passes": status == "passed",
        "allowed_candidate_over_baseline_ratio": allowed_ratio,
        "additional_paired_samples_allowed": max(
            0,
            MAXIMUM_PAIRED_SAMPLES - sample_count,
        ),
    }


def _measurement(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
) -> Mapping[str, Any]:
    telemetry = _mapping(sample.get("telemetry"), description="sample telemetry")
    profile = _mapping(
        telemetry.get("runtime_profile"),
        description="sample recurrence runtime profile",
    )
    measurements = _list(
        profile.get("measurements"),
        description="runtime profile measurements",
    )
    matches = [
        _mapping(item, description="runtime profile measurement")
        for item in measurements
        if isinstance(item, Mapping) and item.get("batch_size") == batch_size
    ]
    if len(matches) != 1:
        raise RuntimeABError(
            f"runtime profile must contain exactly one batch {batch_size} measurement"
        )
    measurement = matches[0]
    subprocess_samples = _list(
        measurement.get("subprocess_samples"),
        description=f"batch {batch_size} subprocess samples",
    )
    sample_count = _positive_int(
        measurement.get("sample_count"),
        description=f"batch {batch_size} sample count",
    )
    if sample_count != len(subprocess_samples):
        raise RuntimeABError(
            f"batch {batch_size} sample count does not match its worker list"
        )
    if measurement.get("interrupted") is not False:
        raise RuntimeABError(f"batch {batch_size} measurement was interrupted")
    return measurement


def _worker_rounds(
    measurement: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[int, Mapping[str, Any]]:
    raw_samples = _list(
        measurement.get("subprocess_samples"),
        description=f"batch {batch_size} subprocess samples",
    )
    rounds: dict[int, Mapping[str, Any]] = {}
    schedule_indices: set[int] = set()
    for raw in raw_samples:
        sample = _mapping(raw, description="runtime worker sample")
        round_index = _nonnegative_int(
            sample.get("round"),
            description="runtime worker round",
        )
        schedule_index = _nonnegative_int(
            sample.get("schedule_index"),
            description="runtime worker schedule index",
        )
        if round_index in rounds:
            raise RuntimeABError(
                f"batch {batch_size} duplicates worker round {round_index}"
            )
        if schedule_index in schedule_indices:
            raise RuntimeABError(
                f"batch {batch_size} reuses worker schedule index {schedule_index}"
            )
        _positive_float(
            sample.get("wall_seconds_per_point"),
            description="runtime worker wall seconds per point",
        )
        if sample.get("interrupted") is not False:
            raise RuntimeABError("runtime worker sample was interrupted")
        rounds[round_index] = sample
        schedule_indices.add(schedule_index)
    if set(rounds) != set(range(len(raw_samples))):
        raise RuntimeABError(
            f"batch {batch_size} worker rounds are not contiguous from zero"
        )
    return rounds


def _sample_variant(sample: Mapping[str, Any]) -> str:
    variant = sample.get("variant")
    if variant not in {"baseline", "candidate"}:
        raise RuntimeABError("campaign sample has an invalid A/B variant")
    return str(variant)


def _cell_observations(
    samples: Sequence[Mapping[str, Any]],
    *,
    multiplicity: int,
    layout: str,
    batch_size: int,
) -> list[WorkerObservation]:
    pair_members: dict[int, dict[str, Mapping[str, Any]]] = {}
    for sample in samples:
        if sample.get("multiplicity") != multiplicity or sample.get("layout") != layout:
            continue
        if sample.get("runtime_enabled") is not True:
            raise RuntimeABError(
                f"n={multiplicity}/{layout} sample did not enable runtime profiling"
            )
        if sample.get("status") != "passed":
            raise RuntimeABError(
                f"n={multiplicity}/{layout} has a non-passing runtime sample"
            )
        pair_index = _nonnegative_int(
            sample.get("pair_index"),
            description="outer A/B pair index",
        )
        variant = _sample_variant(sample)
        members = pair_members.setdefault(pair_index, {})
        if variant in members:
            raise RuntimeABError(
                f"outer pair {pair_index} duplicates the {variant} sample"
            )
        members[variant] = sample

    observations: list[WorkerObservation] = []
    previous_first_variant: str | None = None
    for pair_index, members in sorted(pair_members.items()):
        if set(members) != {"baseline", "candidate"}:
            raise RuntimeABError(
                f"outer pair {pair_index} is missing a baseline or candidate"
            )
        baseline_sample = members["baseline"]
        candidate_sample = members["candidate"]
        baseline_order = _nonnegative_int(
            baseline_sample.get("order_in_pair"),
            description="baseline order in outer pair",
        )
        candidate_order = _nonnegative_int(
            candidate_sample.get("order_in_pair"),
            description="candidate order in outer pair",
        )
        if {baseline_order, candidate_order} != {0, 1}:
            raise RuntimeABError(
                f"outer pair {pair_index} does not contain positions zero and one"
            )
        first_variant = "baseline" if baseline_order == 0 else "candidate"
        if previous_first_variant == first_variant:
            raise RuntimeABError(
                "outer baseline/candidate order did not alternate between pairs"
            )
        previous_first_variant = first_variant

        baseline_rounds = _worker_rounds(
            _measurement(baseline_sample, batch_size=batch_size),
            batch_size=batch_size,
        )
        candidate_rounds = _worker_rounds(
            _measurement(candidate_sample, batch_size=batch_size),
            batch_size=batch_size,
        )
        if set(baseline_rounds) != set(candidate_rounds):
            raise RuntimeABError(f"outer pair {pair_index} has unmatched worker rounds")
        for round_index in sorted(baseline_rounds):
            baseline_worker = baseline_rounds[round_index]
            candidate_worker = candidate_rounds[round_index]
            observations.append(
                WorkerObservation(
                    pair_index=pair_index,
                    round_index=round_index,
                    baseline_order_in_pair=baseline_order,
                    candidate_order_in_pair=candidate_order,
                    baseline_seconds_per_point=_positive_float(
                        baseline_worker.get("wall_seconds_per_point"),
                        description="baseline runtime wall seconds per point",
                    ),
                    candidate_seconds_per_point=_positive_float(
                        candidate_worker.get("wall_seconds_per_point"),
                        description="candidate runtime wall seconds per point",
                    ),
                    baseline_runtime_peak_rss_bytes=_runtime_peak_rss_bytes(
                        baseline_worker
                    ),
                    candidate_runtime_peak_rss_bytes=_runtime_peak_rss_bytes(
                        candidate_worker
                    ),
                )
            )
    return observations


def _cell_result(
    observations: Sequence[WorkerObservation],
    *,
    multiplicity: int,
    layout: str,
    batch_size: int,
    material_rss_ratio: float,
) -> dict[str, object]:
    if not observations:
        return {
            "multiplicity": multiplicity,
            "layout": layout,
            "batch_size": batch_size,
            "status": "missing",
            "passes": False,
            "paired_sample_count": 0,
        }
    if len(observations) > MAXIMUM_PAIRED_SAMPLES:
        raise RuntimeABError(
            f"n={multiplicity}/{layout}/batch={batch_size} exceeds the 21-pair cap"
        )
    baseline_timing = [
        observation.baseline_seconds_per_point for observation in observations
    ]
    candidate_timing = [
        observation.candidate_seconds_per_point for observation in observations
    ]
    timing_interval = paired_log_ratio_interval(
        baseline_timing,
        candidate_timing,
    )
    timing_gate = _gate_decision(
        timing_interval,
        allowed_ratio=NO_SLOWDOWN_RATIO,
        failure_label="rejected-supported-slowdown",
    )

    rss_complete = all(
        observation.baseline_runtime_peak_rss_bytes is not None
        and observation.candidate_runtime_peak_rss_bytes is not None
        for observation in observations
    )
    if rss_complete:
        baseline_rss = [
            int(observation.baseline_runtime_peak_rss_bytes)
            for observation in observations
        ]
        candidate_rss = [
            int(observation.candidate_runtime_peak_rss_bytes)
            for observation in observations
        ]
        rss_interval: dict[str, object] | None = paired_log_ratio_interval(
            baseline_rss,
            candidate_rss,
        )
        rss_gate: dict[str, object] = _gate_decision(
            rss_interval,
            allowed_ratio=material_rss_ratio,
            failure_label="rejected-supported-material-rss-increase",
        )
    else:
        rss_interval = None
        rss_gate = {
            "status": "missing-runtime-only-rss",
            "passes": False,
            "allowed_candidate_over_baseline_ratio": material_rss_ratio,
            "missing_paired_observation_count": sum(
                observation.baseline_runtime_peak_rss_bytes is None
                or observation.candidate_runtime_peak_rss_bytes is None
                for observation in observations
            ),
        }

    statuses = {str(timing_gate["status"]), str(rss_gate["status"])}
    if any(status.startswith("rejected") for status in statuses):
        status = "rejected"
    elif "missing-runtime-only-rss" in statuses:
        status = "incomplete-runtime-rss"
    elif "needs-more-samples" in statuses:
        status = "needs-more-samples"
    else:
        status = "passed"
    return {
        "multiplicity": multiplicity,
        "layout": layout,
        "batch_size": batch_size,
        "status": status,
        "passes": status == "passed",
        "paired_sample_count": len(observations),
        "pairing_contract": "outer-pair-index-and-inner-round-v1",
        "observations": [
            {
                "pair_index": observation.pair_index,
                "round": observation.round_index,
                "baseline_order_in_pair": observation.baseline_order_in_pair,
                "candidate_order_in_pair": observation.candidate_order_in_pair,
                "baseline_seconds_per_point": (observation.baseline_seconds_per_point),
                "candidate_seconds_per_point": (
                    observation.candidate_seconds_per_point
                ),
                "baseline_runtime_peak_rss_bytes": (
                    observation.baseline_runtime_peak_rss_bytes
                ),
                "candidate_runtime_peak_rss_bytes": (
                    observation.candidate_runtime_peak_rss_bytes
                ),
            }
            for observation in observations
        ],
        "timing": {
            "interval": timing_interval,
            "gate": timing_gate,
        },
        "runtime_rss": {
            "evidence_contract": ("profile-worker-post-runtime-resource-high-water-v1"),
            "interval": rss_interval,
            "gate": rss_gate,
        },
    }


def analyze_campaign(
    campaign: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] = DEFAULT_MULTIPLICITIES,
    layouts: Sequence[str] = LAYOUTS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    material_rss_ratio: float = DEFAULT_MATERIAL_RSS_RATIO,
) -> dict[str, object]:
    """Analyze every requested runtime cell and fail closed on missing evidence."""

    if (
        campaign.get("kind") != CAMPAIGN_KIND
        or campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION
    ):
        raise RuntimeABError("input is not a supported recurrence A/B campaign")
    if not math.isfinite(material_rss_ratio) or material_rss_ratio < 1.0:
        raise RuntimeABError("material RSS ratio must be finite and at least one")

    requested_multiplicities = tuple(dict.fromkeys(multiplicities))
    requested_layouts = tuple(dict.fromkeys(layouts))
    requested_batches = tuple(dict.fromkeys(batch_sizes))
    if not requested_multiplicities or not requested_layouts or not requested_batches:
        raise RuntimeABError("the requested runtime matrix must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (*requested_multiplicities, *requested_batches)
    ):
        raise RuntimeABError("multiplicities and batch sizes must be positive")
    if any(layout not in LAYOUTS for layout in requested_layouts):
        raise RuntimeABError("the runtime matrix contains an unsupported LC layout")

    configuration = _mapping(
        campaign.get("configuration"),
        description="campaign configuration",
    )
    configuration_errors: list[str] = []
    if configuration.get("warmup_runs") != 2:
        configuration_errors.append("warmup_runs must equal 2")
    if configuration.get("target_runtime_seconds") != 5.0:
        configuration_errors.append("target_runtime_seconds must equal 5")
    minimum_samples = configuration.get("minimum_samples")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < 7
    ):
        configuration_errors.append("minimum_samples must be at least 7")
    subprocess_samples = configuration.get("subprocess_samples")
    if (
        isinstance(subprocess_samples, bool)
        or not isinstance(subprocess_samples, int)
        or not 7 <= subprocess_samples <= 21
    ):
        configuration_errors.append("subprocess_samples must be between 7 and 21")
    if (
        configuration.get("ordering_policy")
        != "alternating-baseline-candidate-pairs-v1"
    ):
        configuration_errors.append("outer A/B ordering policy is not alternating")
    if configuration.get("cold_cache_policy") != "unique-roots-per-outer-sample-v1":
        configuration_errors.append("outer samples do not use isolated cache roots")
    configured_batches = configuration.get("batch_sizes")
    if not isinstance(configured_batches, list) or not set(requested_batches).issubset(
        configured_batches
    ):
        configuration_errors.append("campaign does not configure every batch size")
    configured_runtime_n = configuration.get("runtime_multiplicities")
    if not isinstance(configured_runtime_n, list) or not set(
        requested_multiplicities
    ).issubset(configured_runtime_n):
        configuration_errors.append(
            "campaign does not runtime-profile every multiplicity"
        )
    configured_layouts = configuration.get("layouts")
    if not isinstance(configured_layouts, list) or not set(requested_layouts).issubset(
        configured_layouts
    ):
        configuration_errors.append("campaign does not include every LC layout")
    if configuration_errors:
        raise RuntimeABError(
            "campaign configuration violates the runtime gate: "
            + "; ".join(configuration_errors)
        )

    raw_samples = _list(campaign.get("samples"), description="campaign samples")
    samples = [
        _mapping(sample, description="campaign sample") for sample in raw_samples
    ]
    cell_results = []
    for multiplicity in requested_multiplicities:
        for layout in requested_layouts:
            for batch_size in requested_batches:
                observations = _cell_observations(
                    samples,
                    multiplicity=multiplicity,
                    layout=layout,
                    batch_size=batch_size,
                )
                cell_results.append(
                    _cell_result(
                        observations,
                        multiplicity=multiplicity,
                        layout=layout,
                        batch_size=batch_size,
                        material_rss_ratio=material_rss_ratio,
                    )
                )

    statuses = {str(cell["status"]) for cell in cell_results}
    if "rejected" in statuses:
        status = "rejected"
    elif "missing" in statuses:
        status = "incomplete-matrix"
    elif "incomplete-runtime-rss" in statuses:
        status = "incomplete-runtime-rss"
    elif "needs-more-samples" in statuses:
        status = "needs-more-samples"
    else:
        status = "passed"
    return {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "passes": status == "passed",
        "source_campaign_id": campaign.get("campaign_id"),
        "gate_contract": {
            "multiplicities": list(requested_multiplicities),
            "layouts": list(requested_layouts),
            "batch_sizes": list(requested_batches),
            "minimum_paired_samples": MINIMUM_PAIRED_SAMPLES,
            "maximum_paired_samples": MAXIMUM_PAIRED_SAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            "timing_allowed_candidate_over_baseline_ratio": NO_SLOWDOWN_RATIO,
            "material_rss_candidate_over_baseline_ratio": material_rss_ratio,
            "warmup_runs": 2,
            "target_runtime_seconds": 5.0,
        },
        "cell_count": len(cell_results),
        "passing_cell_count": sum(cell.get("passes") is True for cell in cell_results),
        "cells": cell_results,
    }


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeABError(f"cannot read campaign JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeABError("campaign JSON must contain one object")
    return value


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _material_rss_ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and at least 1.0")
    return parsed


def _write_json(path: Path, value: object, *, replace: bool) -> None:
    allowed = ALLOWED_OUTPUT_PARENT.resolve()
    destination = path.expanduser().resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as error:
        raise RuntimeABError(
            f"output must remain under {ALLOWED_OUTPUT_PARENT}"
        ) from error
    if destination.exists() and not replace:
        raise RuntimeABError(
            f"output already exists; pass --replace to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeABError(f"cannot write analysis JSON: {destination}") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign", type=Path, required=True)
    result.add_argument("--output-json", type=Path)
    result.add_argument("--replace", action="store_true")
    result.add_argument(
        "--n",
        dest="multiplicities",
        action="append",
        type=_positive_cli_int,
    )
    result.add_argument("--layout", action="append", choices=LAYOUTS)
    result.add_argument(
        "--batch-size",
        action="append",
        type=_positive_cli_int,
    )
    result.add_argument(
        "--material-rss-ratio",
        type=_material_rss_ratio,
        default=DEFAULT_MATERIAL_RSS_RATIO,
        help="largest candidate/baseline runtime RSS ratio accepted (default: 1.10)",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
        campaign = _json_object(arguments.campaign)
        analysis = analyze_campaign(
            campaign,
            multiplicities=(
                DEFAULT_MULTIPLICITIES
                if arguments.multiplicities is None
                else arguments.multiplicities
            ),
            layouts=LAYOUTS if arguments.layout is None else arguments.layout,
            batch_sizes=(
                DEFAULT_BATCH_SIZES
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            material_rss_ratio=arguments.material_rss_ratio,
        )
        if arguments.output_json is not None:
            _write_json(arguments.output_json, analysis, replace=arguments.replace)
    except RuntimeABError as error:
        print(f"recurrence-runtime-ab-analyzer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(analysis, allow_nan=False, sort_keys=True))
    return 0 if analysis["passes"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
