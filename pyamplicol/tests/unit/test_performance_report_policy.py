# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tools.performance_report.arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    EMPTY_ARENA_PHASE_VECTOR_FIELDS,
    ZERO_ARENA_COUNTER_FIELDS,
    ZERO_ARENA_PHASE_TIME_FIELDS,
    ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS,
    build_arena_profile_evidence,
    digest_arena_profile_value,
)
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
            "evaluator_time_raw_seconds_per_point": 1.0e-6,
            "evaluator_time_status": "measured",
            "evaluator_time_ratio_eligible": True,
            "evaluator_time_source": (
                "runtime_profile_core_evaluator_call_time"
            ),
            "compiled_direct_arena_active": False,
            "evaluator_sample_count": 5,
            "native_profile_points_per_sample": 128,
            "timing_sample_contract": (
                "paired_unprofiled_headline_profiled_attribution_v1"
            ),
        },
    )

    result = _benchmark_measurement(benchmark, matrix_element=3.0)

    assert result["benchmark_evidence"] == {
        "measurement_phase_elapsed_seconds": 5.02,
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


def test_unavailable_execution_requires_authenticated_arena_boundary() -> None:
    measurement = _measurement()
    measurement["execution_seconds_per_point"] = None

    assert any(
        issue.field == "execution_seconds_per_point"
        for issue in timing_policy_issues(measurement)
    )

    provenance = measurement["provenance"]
    raw_profile = {
        "execution_mode": "compiled",
        "profile_boundary": ARENA_PROFILE_BOUNDARY,
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
        "evaluator_timing_available": False,
        "points": 128,
        "wall_time_s": 128 * 2.1e-6,
        "orchestration_time_s": 128 * 2.1e-6,
        **{field: 0 for field in ZERO_ARENA_COUNTER_FIELDS},
        **{field: 0.0 for field in ZERO_ARENA_PHASE_TIME_FIELDS},
        **{field: [] for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS},
        **{
            field: 0
            for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS
        },
        "compiled_direct_arena_engine_count": 1,
        "compiled_direct_arena_call_count": 128,
        "evaluator_backend_call_count": 128,
    }
    arena_evidence = build_arena_profile_evidence(
        [raw_profile] * 5,
        execution_mode="compiled",
        repetitions_per_profile=1,
        batch_size=128,
    )
    provenance["arena_profile_evidence"] = arena_evidence
    provenance["execution_timing"] = {
        "abi": "pyamplicol-report-arena-execution-timing-v2",
        "status": "unavailable",
        "ratio_eligible": False,
        "raw_seconds_per_point": None,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "sample_contract": (
            "paired_unprofiled_headline_profiled_attribution_v1"
        ),
        "profile_protocol": "arena",
        "profile_sample_pass": "runtime._profile_arena_repeated",
        "profile_boundary": (
            "warmed-direct-arena-borrowed-input-preallocated-output-v1"
        ),
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": "coarse-arena-boundary-only-v1",
        "evaluator_timing_available": False,
        "paired_with_headline": True,
        "identical_batch": True,
        "identical_repetitions": True,
        "execution_mode": "compiled",
        "warmed_boundary_wall_seconds_per_point": 2.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(
            arena_evidence
        ),
    }
    assert not any(
        issue.field == "execution_seconds_per_point"
        for issue in timing_policy_issues(measurement)
    )


def test_zero_execution_is_not_treated_as_an_unavailable_attribution() -> None:
    measurement = _measurement()
    measurement["execution_seconds_per_point"] = 0.0

    assert any(
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


# The one-loop benchmark report keeps only the essentials (user 2026-11-06
# ~13:25: "keep this document to the essential with mostly just the relevant
# tables"; docs/performance_reports/loop/README.md): the two summary tables
# first, then only what is needed to read them, newest measurements only.
LOOP_REPORT = (
    Path(__file__).resolve().parents[2] / "docs" / "performance_reports" / "loop"
)
# Its items in order: Table 1 (Born-interfered) and Table 2 (loop-induced)
# open it, then Figure 1 (Table 1 against the gate), Table 3 (the gate and
# stage numbers behind Table 1's chips and bars) and the reading guide.
LOOP_REPORT_ITEMS = (
    "tab:summary",
    "tab:summary-li",
    "fig:gate",
    "tab:gate",
    "reading",
)
# What the report no longer prints: milestone names and narratives, and the
# history fields of the campaign records (earlier runs, previous milestones).
LOOP_REPORT_HISTORY = re.compile(
    r"milestone|earlier_runs|previous_|retime|history", re.IGNORECASE
)


def _loop_report_code() -> str:
    """benchmarks.typ without its comment lines."""
    lines = (LOOP_REPORT / "benchmarks.typ").read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("//"))


def test_loop_report_renders_only_the_two_summary_files() -> None:
    code = _loop_report_code()

    assert sorted(re.findall(r'json\("([^"]+)"\)', code)) == [
        "data/summary-li.json",
        "data/summary.json",
    ]
    assert not re.search(r"#(?:include|import)\b|\b(?:read|csv|yaml|toml|image)\(", code)


def test_loop_report_holds_the_summary_tables_first_and_only_their_readers() -> None:
    code = _loop_report_code()

    labels = re.findall(r"[)\]]\s*<([a-z][a-z0-9:-]*)>", code)
    assert tuple(labels) == LOOP_REPORT_ITEMS
    # No numbered section: one unnumbered heading, the reading guide.
    assert not re.search(r"(?m)^\s*=+\s", code)
    assert re.findall(r"#heading\(([^)]*)\)\[([^\]]*)\]", code) == [
        ("numbering: none", "Reading the tables")
    ]
    assert not LOOP_REPORT_HISTORY.search(code)


def test_loop_report_tables_state_their_provenance_in_one_line() -> None:
    code = _loop_report_code()

    # Table 1's first row names the configuration (engine revision, native
    # build and digest, date) and the host; Table 2's its records' or the
    # engine's, the providers' host and the date.
    table1 = code[code.index("#let header-prov") : code.index("<tab:summary>")]
    assert "native-of(x)" in code and '"native_digest"' in code
    assert "sm.host" in table1 and "header-prov" in table1.split("table.header(")[1]
    table2 = code[code.index("#let li-head") : code.index("<tab:summary-li>")]
    for field in ("li-meta.date", "li-hosts", "native-of(li-engine)", "native-of(x)"):
        assert field in table2
    assert "li-head" in table2.split("table.header(")[1]
    # Table 2's provenance names pyAmpliCol's host, its generation cell the
    # generation-time probes (the full production generate cost).
    assert "li-our-hosts" in table2 and "timing.host" in code
    generation = code[code.index("#let li-generation") : code.index("#let li-timing")]
    assert "m.probes_seconds" in generation and "m.probe_share" in generation


def test_loop_report_head_line_labels_table_1_and_names_the_report() -> None:
    code = _loop_report_code()

    # The running header and the page-1 subtitle name the report's date and
    # newest measured engine (summary.json meta.report) and label Table 1's
    # configuration as Table 1's, so no reader takes the whole report to be
    # at Table 1's engine.
    assert "#let rep = sm.report" in code
    header = code[code.index("header: context") : code.index("#set text(")]
    page1 = code[code.index("#align(center)[") : code.index("#let m-rows")]
    for line in (header, page1):
        assert "rep.date" in line and "rep.head" in line
        assert "Table 1: engine #raw(sm.engine_on_loop)" in line
    assert "t2-engine" in page1


def test_loop_report_readme_states_the_essentials_rule() -> None:
    readme = (LOOP_REPORT / "README.md").read_text(encoding="utf-8")

    for label in LOOP_REPORT_ITEMS:
        assert f"`<{label}>`" in readme
    assert "The essentials rule" in readme

