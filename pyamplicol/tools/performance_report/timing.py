# SPDX-License-Identifier: 0BSD
"""Authenticated native execution-timing metadata shared by report views."""

from __future__ import annotations

import math
from collections.abc import Mapping

MEASURED_EXECUTION_TIMING_ABI = "pyamplicol-report-execution-timing-v1"
ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI = (
    "pyamplicol-report-arena-execution-timing-v2"
)
EXECUTION_TIMING_KEY = "execution_timing"
UNAVAILABLE_STATUS = "unavailable"
ARENA_PROFILE_PROTOCOL = "arena"
ARENA_PROFILE_SAMPLE_PASS = "runtime._profile_arena_repeated"
ARENA_PROFILE_BOUNDARY = (
    "warmed-direct-arena-borrowed-input-preallocated-output-v1"
)
ARENA_PHASE_TIMING_SCOPE = "coarse-arena-boundary-only-v1"
PAIRED_TIMING_SAMPLE_CONTRACT = (
    "paired_unprofiled_headline_profiled_attribution_v1"
)
ARENA_UNAVAILABLE_EXECUTION_TIMING_FIELDS = frozenset(
    {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "sample_count",
        "native_profile_points_per_sample",
        "sample_contract",
        "profile_protocol",
        "profile_sample_pass",
        "profile_boundary",
        "borrowed_flat_input",
        "preallocated_output",
        "phase_timing_scope",
        "evaluator_timing_available",
        "paired_with_headline",
        "identical_batch",
        "identical_repetitions",
        "execution_mode",
        "warmed_boundary_wall_seconds_per_point",
    }
)


def unavailable_execution_timing_record(
    measurement: Mapping[str, object],
    field: str,
) -> Mapping[str, object] | None:
    """Return an authenticated Arena record for an unexposed timing field."""

    if field != "execution_seconds_per_point":
        return None
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    record = provenance.get(EXECUTION_TIMING_KEY)
    if not isinstance(record, Mapping):
        return None
    raw = record.get("raw_seconds_per_point")
    sample_count = record.get("sample_count")
    native_points = record.get("native_profile_points_per_sample")
    warmed_wall = record.get("warmed_boundary_wall_seconds_per_point")
    execution_mode = record.get("execution_mode")
    if (
        set(record) != ARENA_UNAVAILABLE_EXECUTION_TIMING_FIELDS
        or record.get("abi") != ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI
        or record.get("status") != UNAVAILABLE_STATUS
        or record.get("ratio_eligible") is not False
        or raw is not None
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or isinstance(native_points, bool)
        or not isinstance(native_points, int)
        or native_points < 1
        or record.get("sample_contract") != PAIRED_TIMING_SAMPLE_CONTRACT
        or record.get("profile_protocol") != ARENA_PROFILE_PROTOCOL
        or record.get("profile_sample_pass") != ARENA_PROFILE_SAMPLE_PASS
        or record.get("profile_boundary") != ARENA_PROFILE_BOUNDARY
        or record.get("borrowed_flat_input") is not True
        or record.get("preallocated_output") is not True
        or record.get("phase_timing_scope") != ARENA_PHASE_TIMING_SCOPE
        or record.get("evaluator_timing_available") is not False
        or record.get("paired_with_headline") is not True
        or record.get("identical_batch") is not True
        or record.get("identical_repetitions") is not True
        or execution_mode not in {"compiled", "eager", "recurrence"}
        or isinstance(warmed_wall, bool)
        or not isinstance(warmed_wall, (int, float))
        or not math.isfinite(float(warmed_wall))
        or float(warmed_wall) <= 0.0
        or measurement.get(field) is not None
    ):
        return None
    return record


__all__ = [
    "ARENA_PHASE_TIMING_SCOPE",
    "ARENA_PROFILE_BOUNDARY",
    "ARENA_PROFILE_PROTOCOL",
    "ARENA_PROFILE_SAMPLE_PASS",
    "ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI",
    "ARENA_UNAVAILABLE_EXECUTION_TIMING_FIELDS",
    "EXECUTION_TIMING_KEY",
    "MEASURED_EXECUTION_TIMING_ABI",
    "PAIRED_TIMING_SAMPLE_CONTRACT",
    "UNAVAILABLE_STATUS",
    "unavailable_execution_timing_record",
]
