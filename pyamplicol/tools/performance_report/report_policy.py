# SPDX-License-Identifier: 0BSD
"""Strict publication policies not duplicated by the exhaustive final auditor."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .models import ResultStatus
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .source_identity import SOURCE_IDENTITY_SCHEMA
from .timing import below_resolution_record

Measurement = Mapping[str, object]
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_MINIMUM_TARGET_FRACTION = 0.95
_MINIMUM_TIMED_SAMPLES = 5


@dataclass(frozen=True, slots=True)
class ReportPolicyIssue:
    field: str
    detail: str


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _runtime_evidence(measurement: Measurement) -> Mapping[str, object] | None:
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    runtime = provenance.get("runtime_profile")
    if not isinstance(runtime, Mapping):
        return None
    legacy = runtime.get("measurement")
    return legacy if isinstance(legacy, Mapping) else runtime


def timing_policy_issues(
    measurement: Measurement,
    *,
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS,
) -> tuple[ReportPolicyIssue, ...]:
    """Return timing defects for one otherwise successful report measurement."""

    if measurement.get("status") != ResultStatus.OK.value:
        return ()
    issues: list[ReportPolicyIssue] = []
    for field in ("generation_seconds", "wall_seconds_per_point"):
        value = _finite_number(measurement.get(field))
        if value is None or value <= 0.0:
            issues.append(
                ReportPolicyIssue(field, "must be finite and strictly positive")
            )

    execution = _finite_number(measurement.get("execution_seconds_per_point"))
    if execution is None:
        if below_resolution_record(
            measurement,
            "execution_seconds_per_point",
        ) is None:
            issues.append(
                ReportPolicyIssue(
                    "execution_seconds_per_point",
                    (
                        "must be measured or carry authenticated compiled-Arena "
                        "below-resolution evidence"
                    ),
                )
            )
    elif execution < 0.0:
        issues.append(
            ReportPolicyIssue(
                "execution_seconds_per_point",
                "must not be negative",
            )
        )
    elif execution == 0.0:
        issues.append(
            ReportPolicyIssue(
                "execution_seconds_per_point",
                "raw zero must be represented as unavailable with authenticated "
                "below-resolution evidence",
            )
        )

    sample_count = measurement.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < _MINIMUM_TIMED_SAMPLES
    ):
        issues.append(
            ReportPolicyIssue(
                "sample_count",
                f"at least {_MINIMUM_TIMED_SAMPLES} timed samples are required",
            )
        )
    for field in (
        "standard_error_seconds_per_point",
        "relative_standard_error",
    ):
        value = _finite_number(measurement.get(field))
        if value is None or value <= 0.0:
            issues.append(
                ReportPolicyIssue(
                    field,
                    "must be finite, measured, and strictly positive",
                )
            )

    evidence = _runtime_evidence(measurement)
    if evidence is None:
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile",
                "measured-duration evidence is required",
            )
        )
        return tuple(issues)
    requested = _finite_number(evidence.get("target_runtime_seconds"))
    achieved = _finite_number(evidence.get("achieved_runtime_seconds"))
    if requested is None or not math.isclose(
        requested,
        target_runtime_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.target_runtime_seconds",
                f"must equal {target_runtime_seconds:g} seconds",
            )
        )
    if (
        achieved is None
        or achieved < _MINIMUM_TARGET_FRACTION * target_runtime_seconds
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.achieved_runtime_seconds",
                (
                    "must record at least "
                    f"{_MINIMUM_TARGET_FRACTION * target_runtime_seconds:g} "
                    "seconds of timed evaluation"
                ),
            )
        )
    if evidence.get("target_runtime_achieved") is not True:
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.target_runtime_achieved",
                "must be true",
            )
        )
    if evidence.get("interrupted") is True:
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.interrupted",
                "partial timing runs are not publishable",
            )
        )
    completed = evidence.get("completed_sample_count")
    planned = evidence.get("planned_sample_count")
    if (completed is not None or planned is not None) and (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(planned, bool)
            or not isinstance(planned, int)
            or completed < _MINIMUM_TIMED_SAMPLES
            or completed != planned
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.completed_sample_count",
                "all planned timing blocks must complete",
            )
        )
    chunk_count = evidence.get("chunk_count")
    if chunk_count is not None and (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < _MINIMUM_TIMED_SAMPLES
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.chunk_count",
                f"at least {_MINIMUM_TIMED_SAMPLES} timed chunks are required",
            )
        )
    if completed is None and chunk_count is None:
        issues.append(
            ReportPolicyIssue(
                "provenance.runtime_profile.sample_count",
                "timed sample or chunk completion evidence is required",
            )
        )
    return tuple(issues)


def source_policy_issues(
    measurement: Measurement,
) -> tuple[ReportPolicyIssue, ...]:
    """Require the exact clean measurement commit and tree in provenance."""

    if measurement.get("status") != ResultStatus.OK.value:
        return ()
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return (
            ReportPolicyIssue("provenance", "source identity is required"),
        )
    issues: list[ReportPolicyIssue] = []
    if provenance.get("report_source_identity_schema") != SOURCE_IDENTITY_SCHEMA:
        issues.append(
            ReportPolicyIssue(
                "provenance.report_source_identity_schema",
                f"must equal {SOURCE_IDENTITY_SCHEMA}",
            )
        )
    for field in (
        "report_source_revision",
        "report_source_tree",
        "report_measured_source_revision",
        "report_measured_source_tree",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or _FULL_SHA_RE.fullmatch(value) is None:
            issues.append(
                ReportPolicyIssue(
                    f"provenance.{field}",
                    "must be a full lowercase Git object ID",
                )
            )
    if provenance.get("report_source_revision") != provenance.get(
        "report_measured_source_revision"
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.report_measured_source_revision",
                "must match the measured source revision",
            )
        )
    if provenance.get("report_source_tree") != provenance.get(
        "report_measured_source_tree"
    ):
        issues.append(
            ReportPolicyIssue(
                "provenance.report_measured_source_tree",
                "must match the measured source tree",
            )
        )
    if provenance.get("report_source_clean") is not True:
        issues.append(
            ReportPolicyIssue(
                "provenance.report_source_clean",
                "measurement source must be clean",
            )
        )
    return tuple(issues)


def publication_measurement_policy_issues(
    measurement: Measurement,
    *,
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS,
) -> tuple[ReportPolicyIssue, ...]:
    return (
        *timing_policy_issues(
            measurement,
            target_runtime_seconds=target_runtime_seconds,
        ),
        *source_policy_issues(measurement),
    )


__all__ = [
    "ReportPolicyIssue",
    "publication_measurement_policy_issues",
    "source_policy_issues",
    "timing_policy_issues",
]
