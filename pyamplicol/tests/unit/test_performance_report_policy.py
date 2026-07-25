# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from tools.performance_report.cache import empty_measurement
from tools.performance_report.report_policy import (
    publication_measurement_policy_issues,
    timing_policy_issues,
)
from tools.performance_report.runner import _benchmark_measurement
from tools.performance_report.source_identity import SOURCE_IDENTITY_SCHEMA


def _measurement() -> dict[str, object]:
    revision = "a" * 40
    tree = "b" * 40
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 2.0e-6,
            "execution_seconds_per_point": 1.0e-6,
            "matrix_element": 3.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-9,
            "relative_standard_error": 1.0e-3,
            "artifact": {},
            "selector_contract": None,
            "validation": {"status": "ok"},
            "resources": {},
            "provenance": {
                "runtime_profile": {
                    "target_runtime_seconds": 5.0,
                    "achieved_runtime_seconds": 5.01,
                    "target_runtime_achieved": True,
                    "completed_sample_count": 5,
                    "planned_sample_count": 5,
                    "interrupted": False,
                },
                "report_source_identity_schema": SOURCE_IDENTITY_SCHEMA,
                "report_source_revision": revision,
                "report_source_tree": tree,
                "report_measured_source_revision": revision,
                "report_measured_source_tree": tree,
                "report_source_clean": True,
            },
            "failure": None,
        }
    )
    return measurement


def test_complete_five_second_timing_and_source_identity_pass() -> None:
    assert publication_measurement_policy_issues(_measurement()) == ()


def test_native_benchmark_duration_is_retained_as_publication_evidence() -> None:
    benchmark = SimpleNamespace(
        uncertainty=SimpleNamespace(
            standard_error=1.0e-9,
            relative_standard_error=1.0e-3,
        ),
        effective_config=SimpleNamespace(target_runtime=5.0),
        wall_time_per_point=2.0e-6,
        evaluator_time_per_point=1.0e-6,
        sample_count=5,
        environment={
            "elapsed_seconds": 5.02,
            "completed_sample_count": 5,
            "planned_sample_count": 5,
            "repetitions_per_sample": 100,
            "measured_point_count": 64000,
            "interrupted": False,
        },
    )

    result = _benchmark_measurement(benchmark, matrix_element=3.0)

    assert result["benchmark_evidence"] == {
        "target_runtime_seconds": 5.0,
        "achieved_runtime_seconds": 5.02,
        "target_runtime_achieved": True,
        "completed_sample_count": 5,
        "planned_sample_count": 5,
        "repetitions_per_sample": 100,
        "measured_point_count": 64000,
        "interrupted": False,
    }


def test_requested_and_achieved_runtime_are_both_enforced() -> None:
    measurement = _measurement()
    runtime = measurement["provenance"]["runtime_profile"]
    runtime["target_runtime_seconds"] = 1.0
    runtime["achieved_runtime_seconds"] = 0.2
    runtime["target_runtime_achieved"] = False

    issues = timing_policy_issues(measurement)
    fields = {issue.field for issue in issues}

    assert "provenance.runtime_profile.target_runtime_seconds" in fields
    assert "provenance.runtime_profile.achieved_runtime_seconds" in fields
    assert "provenance.runtime_profile.target_runtime_achieved" in fields


def test_five_timed_samples_are_required() -> None:
    measurement = _measurement()
    measurement["sample_count"] = 4
    runtime = measurement["provenance"]["runtime_profile"]
    runtime["completed_sample_count"] = 4
    runtime["planned_sample_count"] = 4

    fields = {issue.field for issue in timing_policy_issues(measurement)}

    assert "sample_count" in fields
    assert "provenance.runtime_profile.completed_sample_count" in fields


def test_zero_execution_requires_explicit_resolution_bound() -> None:
    measurement = _measurement()
    measurement["execution_seconds_per_point"] = None

    assert any(
        issue.field == "execution_seconds_per_point"
        for issue in timing_policy_issues(measurement)
    )

    provenance = measurement["provenance"]
    provenance["execution_timing"] = {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "below_timer_resolution",
        "ratio_eligible": False,
        "raw_seconds_per_point": 0.0,
        "source": (
            "runtime_profile_core_compiled_direct_arena_orchestration_time"
        ),
        "compiled_direct_arena_active": True,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "sample_contract": (
            "paired_unprofiled_headline_profiled_attribution_v1"
        ),
    }
    assert not any(
        issue.field == "execution_seconds_per_point"
        for issue in timing_policy_issues(measurement)
    )


def test_hardcoded_zero_uncertainty_and_partial_blocks_are_rejected() -> None:
    measurement = _measurement()
    measurement["standard_error_seconds_per_point"] = 0.0
    measurement["relative_standard_error"] = 0.0
    runtime = measurement["provenance"]["runtime_profile"]
    runtime["completed_sample_count"] = 4
    runtime["interrupted"] = True

    fields = {issue.field for issue in timing_policy_issues(measurement)}

    assert "standard_error_seconds_per_point" in fields
    assert "relative_standard_error" in fields
    assert "provenance.runtime_profile.completed_sample_count" in fields
    assert "provenance.runtime_profile.interrupted" in fields


def test_missing_or_mismatched_source_tree_is_rejected() -> None:
    measurement = deepcopy(_measurement())
    provenance = measurement["provenance"]
    provenance.pop("report_measured_source_tree")
    provenance["report_source_clean"] = False

    fields = {
        issue.field
        for issue in publication_measurement_policy_issues(measurement)
    }

    assert "provenance.report_measured_source_tree" in fields
    assert "provenance.report_source_clean" in fields
