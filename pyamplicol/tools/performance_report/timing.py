# SPDX-License-Identifier: 0BSD
"""Authenticated native execution-timing metadata shared by report views."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    ARENA_PROFILE_PROTOCOL,
    ARENA_PROFILE_SAMPLE_PASS,
    PAIRED_TIMING_SAMPLE_CONTRACT,
    ArenaProfileEvidenceError,
    digest_arena_profile_value,
    validate_arena_profile_evidence,
)

MEASURED_EXECUTION_TIMING_ABI = "pyamplicol-report-execution-timing-v1"
ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI = (
    "pyamplicol-report-arena-execution-timing-v2"
)
EXECUTION_TIMING_KEY = "execution_timing"
EVALUATOR_TOTAL_TIMING_ABI = "pyamplicol-report-evaluator-total-timing-v1"
EVALUATOR_TOTAL_TIMING_KEY = "evaluator_total_timing"
EVALUATOR_TOTAL_TIMING_SOURCE = (
    "runtime._benchmark_f64_wall_time.accumulated"
)
EVALUATOR_TOTAL_TIMING_SOURCES = frozenset(
    {
        EVALUATOR_TOTAL_TIMING_SOURCE,
        "runtime.evaluate.accumulated",
    }
)
EVALUATOR_TOTAL_SAMPLE_CONTRACT = (
    "accumulated-repeated-warmed-evaluator-total-v1"
)
RECURRENCE_EXECUTION_TIMING_SOURCE = (
    "runtime_profile_core_recurrence_schedule_time"
)
UNAVAILABLE_STATUS = "unavailable"
ARENA_UNAVAILABLE_EXECUTION_TIMING_FIELDS = frozenset(
    {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "sample_count",
        "native_profile_points_per_sample",
        "repetitions_per_sample",
        "batch_size",
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
        "arena_profile_evidence_sha256",
    }
)
EVALUATOR_TOTAL_TIMING_FIELDS = frozenset(
    {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "source",
        "execution_mode",
        "sample_contract",
        "sample_count",
        "repetitions_per_sample",
        "batch_size",
        "points_per_sample",
        "measured_point_count",
        "accumulated_seconds",
    }
)
RUNTIME_PROFILE_TOTAL_FIELDS = frozenset(
    {
        "achieved_runtime_seconds",
        "completed_sample_count",
        "interrupted",
        "measured_point_count",
        "planned_sample_count",
        "repetitions_per_sample",
        "target_runtime_achieved",
        "target_runtime_seconds",
    }
)
_MEASURED_EXECUTION_TIMING_FIELDS = frozenset(
    {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "source",
        "compiled_direct_arena_active",
        "sample_count",
        "native_profile_points_per_sample",
        "sample_contract",
    }
)


def _runtime_profile_evaluator_total(
    measurement: Mapping[str, object],
) -> float | None:
    """Recover the accumulated warmed total from canonical profile evidence."""

    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    profile = provenance.get("runtime_profile")
    if not isinstance(profile, Mapping) or set(profile) != RUNTIME_PROFILE_TOTAL_FIELDS:
        return None
    execution = provenance.get(EXECUTION_TIMING_KEY)
    raw_execution = (
        execution.get("raw_seconds_per_point")
        if isinstance(execution, Mapping)
        else None
    )
    achieved = profile.get("achieved_runtime_seconds")
    measured_points = profile.get("measured_point_count")
    completed = profile.get("completed_sample_count")
    planned = profile.get("planned_sample_count")
    repetitions = profile.get("repetitions_per_sample")
    target = profile.get("target_runtime_seconds")
    if (
        measurement.get("status") != "ok"
        or not isinstance(execution, Mapping)
        or set(execution) != _MEASURED_EXECUTION_TIMING_FIELDS
        or execution.get("abi") != MEASURED_EXECUTION_TIMING_ABI
        or execution.get("status") != "measured"
        or execution.get("ratio_eligible") is not True
        or execution.get("source") != RECURRENCE_EXECUTION_TIMING_SOURCE
        or execution.get("compiled_direct_arena_active") is not False
        or execution.get("sample_contract") != PAIRED_TIMING_SAMPLE_CONTRACT
        or isinstance(raw_execution, bool)
        or not isinstance(raw_execution, (int, float))
        or not math.isfinite(float(raw_execution))
        or float(raw_execution) <= 0.0
        or measurement.get("execution_seconds_per_point") != raw_execution
        or isinstance(achieved, bool)
        or not isinstance(achieved, (int, float))
        or not math.isfinite(float(achieved))
        or float(achieved) <= 0.0
        or isinstance(measured_points, bool)
        or not isinstance(measured_points, int)
        or measured_points < 1
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 1
        or isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned != completed
        or measurement.get("sample_count") != completed
        or execution.get("sample_count") != completed
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or measured_points % (completed * repetitions) != 0
        or execution.get("native_profile_points_per_sample")
        != measured_points // completed
        or profile.get("interrupted") is not False
        or profile.get("target_runtime_achieved") is not True
        or isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(float(target))
        or float(target) <= 0.0
        or float(achieved) < float(target)
    ):
        return None
    result = float(achieved) / measured_points
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def evaluator_total_timing_record(
    measurement: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return an authenticated accumulated warmed evaluator-total record."""

    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    record = provenance.get(EVALUATOR_TOTAL_TIMING_KEY)
    if not isinstance(record, Mapping):
        return None
    raw = record.get("raw_seconds_per_point")
    accumulated = record.get("accumulated_seconds")
    sample_count = record.get("sample_count")
    repetitions = record.get("repetitions_per_sample")
    batch_size = record.get("batch_size")
    points_per_sample = record.get("points_per_sample")
    measured_points = record.get("measured_point_count")
    if (
        measurement.get("status") != "ok"
        or set(record) != EVALUATOR_TOTAL_TIMING_FIELDS
        or record.get("abi") != EVALUATOR_TOTAL_TIMING_ABI
        or record.get("status") != "measured"
        or record.get("ratio_eligible") is not False
        or record.get("source") not in EVALUATOR_TOTAL_TIMING_SOURCES
        or record.get("execution_mode") not in {
            "compiled",
            "eager",
            "recurrence",
        }
        or record.get("sample_contract") != EVALUATOR_TOTAL_SAMPLE_CONTRACT
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or measurement.get("sample_count") != sample_count
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(points_per_sample, bool)
        or not isinstance(points_per_sample, int)
        or points_per_sample != repetitions * batch_size
        or isinstance(measured_points, bool)
        or not isinstance(measured_points, int)
        or measured_points != sample_count * points_per_sample
        or isinstance(accumulated, bool)
        or not isinstance(accumulated, (int, float))
        or not math.isfinite(float(accumulated))
        or float(accumulated) <= 0.0
        or isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) <= 0.0
        or not math.isclose(
            float(raw),
            float(accumulated) / measured_points,
            rel_tol=1.0e-15,
            abs_tol=0.0,
        )
    ):
        return None
    return record


def evaluator_total_seconds_per_point(
    measurement: Mapping[str, object],
) -> float | None:
    """Return the authenticated accumulated evaluator total for every mode.

    New measurements carry the dedicated ABI record. Historical recurrence
    measurements can be recovered from the same immutable accumulated runtime
    and measured-point counters that the worker already authenticated.
    """

    record = evaluator_total_timing_record(measurement)
    if record is not None:
        return float(record["raw_seconds_per_point"])
    return _runtime_profile_evaluator_total(measurement)


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
    repetitions = record.get("repetitions_per_sample")
    batch_size = record.get("batch_size")
    warmed_wall = record.get("warmed_boundary_wall_seconds_per_point")
    execution_mode = record.get("execution_mode")
    evidence = provenance.get("arena_profile_evidence")
    evidence_digest = record.get("arena_profile_evidence_sha256")
    evidence_valid = False
    if (
        isinstance(execution_mode, str)
        and isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and isinstance(native_points, int)
        and not isinstance(native_points, bool)
        and isinstance(evidence_digest, str)
    ):
        try:
            validate_arena_profile_evidence(
                evidence,
                execution_mode=execution_mode,
                sample_count=sample_count,
                native_profile_points_per_sample=native_points,
            )
        except ArenaProfileEvidenceError:
            pass
        else:
            evidence_valid = (
                digest_arena_profile_value(evidence) == evidence_digest
                and isinstance(evidence, Mapping)
                and evidence.get("repetitions_per_profile") == repetitions
                and evidence.get("batch_size") == batch_size
                and evidence.get("warmed_boundary_wall_seconds_per_point")
                == warmed_wall
            )
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
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or repetitions * batch_size != native_points
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
        or execution_mode not in {"compiled", "eager"}
        or isinstance(warmed_wall, bool)
        or not isinstance(warmed_wall, (int, float))
        or not math.isfinite(float(warmed_wall))
        or float(warmed_wall) <= 0.0
        or not evidence_valid
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
    "EVALUATOR_TOTAL_SAMPLE_CONTRACT",
    "EVALUATOR_TOTAL_TIMING_ABI",
    "EVALUATOR_TOTAL_TIMING_FIELDS",
    "EVALUATOR_TOTAL_TIMING_KEY",
    "EVALUATOR_TOTAL_TIMING_SOURCE",
    "EVALUATOR_TOTAL_TIMING_SOURCES",
    "EXECUTION_TIMING_KEY",
    "MEASURED_EXECUTION_TIMING_ABI",
    "PAIRED_TIMING_SAMPLE_CONTRACT",
    "RECURRENCE_EXECUTION_TIMING_SOURCE",
    "RUNTIME_PROFILE_TOTAL_FIELDS",
    "UNAVAILABLE_STATUS",
    "evaluator_total_seconds_per_point",
    "evaluator_total_timing_record",
    "unavailable_execution_timing_record",
]
