# SPDX-License-Identifier: 0BSD
"""Authenticated native execution-timing metadata shared by report views."""

from __future__ import annotations

import math
from collections.abc import Mapping

EXECUTION_TIMING_ABI = "pyamplicol-report-execution-timing-v1"
EXECUTION_TIMING_KEY = "execution_timing"
BELOW_RESOLUTION_STATUS = "below_timer_resolution"
COMPILED_ARENA_EXECUTION_SOURCE = (
    "runtime_profile_core_compiled_direct_arena_orchestration_time"
)


def below_resolution_record(
    measurement: Mapping[str, object],
    field: str,
) -> Mapping[str, object] | None:
    """Return an authenticated compiled-Arena below-resolution record."""

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
    sample_contract = record.get("sample_contract")
    if (
        record.get("abi") != EXECUTION_TIMING_ABI
        or record.get("status") != BELOW_RESOLUTION_STATUS
        or record.get("ratio_eligible") is not False
        or isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) != 0.0
        or record.get("source") != COMPILED_ARENA_EXECUTION_SOURCE
        or record.get("compiled_direct_arena_active") is not True
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or isinstance(native_points, bool)
        or not isinstance(native_points, int)
        or native_points < 1
        or not isinstance(sample_contract, str)
        or not sample_contract
        or measurement.get(field) is not None
    ):
        return None
    return record


__all__ = [
    "BELOW_RESOLUTION_STATUS",
    "COMPILED_ARENA_EXECUTION_SOURCE",
    "EXECUTION_TIMING_ABI",
    "EXECUTION_TIMING_KEY",
    "below_resolution_record",
]
