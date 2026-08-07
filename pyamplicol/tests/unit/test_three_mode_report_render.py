# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import tools.performance_report.render as report_render
from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_ABI,
    DIRECT_AGREEMENT_V2_ABI,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    STRICT_ABSOLUTE_TOLERANCE,
    incoming_agreement_edges,
)
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
from tools.performance_report.cache import build_reset_caches, empty_measurement
from tools.performance_report.campaign_policy import (
    X86_EPYC_GENERATION_LIMIT_SECONDS,
    X86_EPYC_MEMORY_LIMIT_BYTES,
    X86_EPYC_POLICY,
    PolicyCensorKind,
    dependency_reference,
    policy_censor_measurement,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.render import (
    BaselineCandidateAdapter,
    _ratio_value,
    render_all_best_mode_tables,
    render_all_matrix_tables,
    render_all_tables,
    render_all_z_ladders,
    render_best_mode_table,
    render_matrix_table,
    render_scalar_ladder,
    render_z_ladder,
    summarize_visible_completeness,
)
from tools.performance_report.runner import (
    OTF_RECURRENCE_AUTHORITY_VALIDATION_FIELD,
    pointwise_validation,
)
from tools.performance_report.source_identity import ReportSourceIdentity
from tools.performance_report.validation_summary import (
    render_validation_summary,
    summarize_validation,
)


def test_unavailable_execution_timing_is_never_ratioed() -> None:
    baseline = {
        "status": ResultStatus.OK.value,
        "execution_seconds_per_point": 1.0e-6,
    }
    candidate = {
        "status": ResultStatus.OK.value,
        "execution_seconds_per_point": 1.0e-9,
        "provenance": {
            "execution_timing": {
                "status": "unavailable",
                "ratio_eligible": False,
            }
        },
    }

    assert _ratio_value(candidate, baseline, "execution_seconds_per_point") is None


@pytest.fixture
def reset_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    monkeypatch.setattr(
        report_render,
        "validate_on_the_fly_recurrence_authority_validation_record",
        lambda *_args, **_kwargs: None,
    )
    return build_reset_caches()


def test_packaged_blank_tables_match_canonical_reset_renderer(
    reset_caches: dict[str, dict[str, object]],
) -> None:
    packaged_root = (
        Path(__file__).resolve().parents[2] / "src/pyamplicol/_profiling_campaign"
    )
    rendered = render_all_tables(reset_caches, catalog=REPORT_CATALOG)
    packaged = {
        path.name: path.read_text(encoding="utf-8")
        for path in packaged_root.glob("result_*.tex")
    }

    assert packaged == rendered


def _cache_by_dataset(
    caches: dict[str, dict[str, object]],
    dataset_id: str,
) -> dict[str, object]:
    return next(
        payload for payload in caches.values() if payload["dataset_id"] == dataset_id
    )


def _set_ok(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
    generation: float,
    wall: float,
    execution: float | None,
    variant: str | None = None,
) -> None:
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == n_final
        and item["workload"] == workload.value
        and item["variant"] == variant
    )
    cell = REPORT_CATALOG.cell(str(entry["cell_id"]))
    entry["measurement"] = _ok_measurement(
        cell,
        generation=generation,
        wall=wall,
        execution=execution,
    )


_TEST_SELECTOR_CONTRACT = {
    "all_flow_helicity_ids": ["h:test"],
    "all_flow_source_helicities": {"1": 1},
    "point_digest": "0" * 64,
    "selected_color_flow_ids": ["flow:test"],
    "selected_color_words": [[1]],
}


def _ok_measurement(
    cell,
    *,
    generation: float,
    wall: float,
    execution: float | None,
) -> dict[str, object]:
    selector = (
        copy.deepcopy(_TEST_SELECTOR_CONTRACT)
        if cell.measurement.accuracy is Accuracy.LC
        else None
    )
    direct_agreements = [
        {
            "abi": DIRECT_AGREEMENT_ABI,
            "edge_kind": edge.kind,
            "value_kind": edge.value_kind,
            "baseline_cell_id": edge.baseline.cell_id,
            "candidate_cell_id": cell.cell_id,
            "status": ResultStatus.OK.value,
            "candidate": 1.0,
            "baseline": 1.0,
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
            "relative_tolerance": edge.relative_tolerance,
            "absolute_tolerance": STRICT_ABSOLUTE_TOLERANCE,
        }
        for edge in incoming_agreement_edges(cell)
    ]
    validation: dict[str, object] = {
        "status": ResultStatus.OK.value,
        "pointwise": {
            "status": ResultStatus.OK.value,
            "candidate": 1.0,
            "baseline": 1.0,
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
            "relative_tolerance": 1.0e-8,
            "absolute_tolerance": 1.0e-15,
        },
        "high_precision": {
            "status": ResultStatus.OK.value,
            "candidate": 1.0,
            "baseline": 1.0,
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
            "relative_tolerance": 1.0e-12,
            "absolute_tolerance": 1.0e-15,
        },
        "direct_agreements": direct_agreements,
    }
    if selector is not None:
        validation[LC_COMMON_COMPONENT_FIELD] = {
            "abi": LC_COMMON_COMPONENT_ABI,
            "cell_id": cell.cell_id,
            "value": 1.0,
            "point_digest": selector["point_digest"],
            "helicity_ids": selector["all_flow_helicity_ids"],
            "color_flow_ids": selector["selected_color_flow_ids"],
        }
    execution_mode = cell.measurement.execution_mode
    artifact_id = (
        "a" * 64
        if execution_mode is ExecutionMode.ON_THE_FLY
        else "b" * 64
        if execution_mode is ExecutionMode.RECURRENCE
        else "c" * 64
    )
    if execution_mode is ExecutionMode.ON_THE_FLY:
        validation[OTF_RECURRENCE_AUTHORITY_VALIDATION_FIELD] = {
            "authority": {"artifact_id": "b" * 64},
        }
    return {
        "status": ResultStatus.OK.value,
        "generation_seconds": generation,
        "wall_seconds_per_point": wall,
        "execution_seconds_per_point": execution,
        "matrix_element": 1.0,
        "sample_count": 5,
        "standard_error_seconds_per_point": 1.0e-9,
        "relative_standard_error": 0.01,
        "artifact": {"digest": "artifact"},
        "selector_contract": selector,
        "validation": validation,
        "resources": {"peak_rss_gib": 1.0},
        "provenance": {
            "source": "test",
            "runtime_identity": {"artifact_id": artifact_id},
            **(
                {"runtime_profile": {"cold_warmup_elapsed_seconds": 0.25}}
                if execution_mode is ExecutionMode.ON_THE_FLY
                else {}
            ),
        },
        "failure": None,
    }


def _set_status(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
    status: ResultStatus,
) -> None:
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == n_final
        and item["workload"] == workload.value
        and item["variant"] is None
    )
    measurement = empty_measurement()
    measurement["status"] = status.value
    entry["measurement"] = measurement


def _set_unverified(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
    generation: float,
    wall: float,
    evaluator_total: float,
    variant: str | None = None,
) -> None:
    entry = _entry(
        cache,
        process_key=process_key,
        n_final=n_final,
        workload=workload,
        variant=variant,
    )
    cell = REPORT_CATALOG.cell(str(entry["cell_id"]))
    measurement = _ok_measurement(
        cell,
        generation=generation,
        wall=wall,
        execution=None,
    )
    _mark_evaluator_total(
        measurement,
        execution_mode=cell.measurement.execution_mode.value,
        total=evaluator_total,
    )
    measurement["status"] = ResultStatus.UNVERIFIED.value
    measurement["failure"] = {
        "kind": "IndependentAuthorityUnavailable",
        "message": "no successful independent authority",
    }
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    validation["status"] = ResultStatus.UNVERIFIED.value
    entry["measurement"] = measurement


def _set_presentation_outcome(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
    outcome: str,
    label: str | None = None,
    variant: str | None = None,
) -> None:
    labels = {
        "error": (ResultStatus.ERROR, "error"),
        "validation_failed": (
            ResultStatus.VALIDATION_FAILED,
            "validation failed",
        ),
        "blocked_dependency": (
            ResultStatus.FAILED,
            "blocked dependency",
        ),
    }
    status, default_label = labels.get(
        outcome,
        (ResultStatus.FAILED, outcome.replace("_", " ")),
    )
    entry = _entry(
        cache,
        process_key=process_key,
        n_final=n_final,
        workload=workload,
        variant=variant,
    )
    measurement = empty_measurement()
    measurement["status"] = status.value
    measurement["failure"] = {
        "kind": f"ManualCampaignOutcome:{outcome}",
        "message": default_label if label is None else label,
    }
    entry["measurement"] = measurement


def _set_best_mode_nlc_reference(caches: dict[str, dict[str, object]]) -> None:
    _set_ok(
        _cache_by_dataset(caches, "reference_amplicol_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )


def _mark_arena_unavailable(
    measurement: dict[str, object],
    *,
    execution_mode: str = "compiled",
) -> None:
    measurement["execution_seconds_per_point"] = None
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    raw_profile = {
        "execution_mode": execution_mode,
        "profile_boundary": ARENA_PROFILE_BOUNDARY,
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
        "evaluator_timing_available": False,
        "points": 128,
        "wall_time_s": 128 * 1.1e-6,
        "orchestration_time_s": 128 * 1.1e-6,
        **{field: 0 for field in ZERO_ARENA_COUNTER_FIELDS},
        **{field: 0.0 for field in ZERO_ARENA_PHASE_TIME_FIELDS},
        **{field: [] for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS},
        **{field: 0 for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS},
    }
    if execution_mode == "compiled":
        raw_profile.update(
            {
                "compiled_direct_arena_engine_count": 1,
                "compiled_direct_arena_call_count": 128,
                "evaluator_backend_call_count": 128,
            }
        )
    arena_evidence = build_arena_profile_evidence(
        [raw_profile] * 5,
        execution_mode=execution_mode,
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
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
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
        "execution_mode": execution_mode,
        "warmed_boundary_wall_seconds_per_point": 1.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(arena_evidence),
    }


def _mark_evaluator_total(
    measurement: dict[str, object],
    *,
    execution_mode: str = "compiled",
    total: float | None = None,
) -> None:
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    wall = measurement["wall_seconds_per_point"]
    assert isinstance(wall, float)
    raw_total = wall if total is None else total
    provenance["evaluator_total_timing"] = {
        "abi": "pyamplicol-report-evaluator-total-timing-v1",
        "status": "measured",
        "ratio_eligible": False,
        "raw_seconds_per_point": raw_total,
        "source": "runtime._benchmark_f64_wall_time.accumulated",
        "execution_mode": execution_mode,
        "sample_contract": ("accumulated-repeated-warmed-evaluator-total-v1"),
        "sample_count": 5,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "points_per_sample": 128,
        "measured_point_count": 640,
        "accumulated_seconds": raw_total * 640,
    }


def _mark_recurrence_core(measurement: dict[str, object]) -> None:
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    raw_core = measurement["execution_seconds_per_point"]
    assert isinstance(raw_core, float)
    provenance["execution_timing"] = {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "measured",
        "ratio_eligible": True,
        "raw_seconds_per_point": raw_core,
        "source": "runtime_profile_core_recurrence_schedule_time",
        "compiled_direct_arena_active": False,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
    }


def _fill_visible_scope(
    caches: dict[str, dict[str, object]],
    *,
    max_n_final: int = 4,
) -> None:
    for payload in caches.values():
        entries = payload["entries"]
        assert isinstance(entries, list)
        for entry in entries:
            if entry["n_final"] > max_n_final:
                continue
            cell = REPORT_CATALOG.cell(str(entry["cell_id"]))
            if REPORT_CATALOG.static_na_reason(cell) is not None:
                continue
            entry["measurement"] = _ok_measurement(
                cell,
                generation=1.0,
                wall=2.0e-6,
                execution=1.0e-6,
            )


def _fill_visible_n4_scope(caches: dict[str, dict[str, object]]) -> None:
    _fill_visible_scope(caches, max_n_final=4)


_POLICY_IDENTITY = ReportSourceIdentity("1" * 40, "2" * 40, ())


def _entry(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
    variant: str | None = None,
) -> dict[str, object]:
    entries = cache["entries"]
    assert isinstance(entries, list)
    return next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == n_final
        and item["workload"] == workload.value
        and item["variant"] == variant
    )


def _policy_resources(peak: int) -> dict[str, object]:
    return {
        "available": True,
        "current_rss_bytes": peak,
        "peak_rss_bytes": peak,
        "child_count": 1,
        "cpu_seconds": 1.0,
        "wall_seconds": 2.0,
        "probe_error": None,
    }


def _generation_censor(cell) -> dict[str, object]:
    phase = {
        "abi": "pyamplicol-report-generation-phase-evidence-v1",
        "phase_state_abi": "pyamplicol-report-worker-phase-state-v1",
        "configured_timeout_seconds": X86_EPYC_GENERATION_LIMIT_SECONDS,
        "supervisor_reason": "generation_timeout",
        "authenticated": True,
        "run_id": "render-policy-test",
        "worker_pid": 123,
        "final_sequence": 1,
        "final_phase": "generation",
        "generation_started_monotonic_ns": 1,
        "generation_finished_monotonic_ns": None,
        "generation_elapsed_seconds": X86_EPYC_GENERATION_LIMIT_SECONDS,
        "final_state_sha256": "3" * 64,
        "error": None,
    }
    resources = _policy_resources(1_000_000)
    resources["generation_phase"] = phase
    return policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        cell,
        kind=PolicyCensorKind.GENERATION_LIMIT,
        source_identity=_POLICY_IDENTITY,
        resources=resources,
        observed_generation_seconds=X86_EPYC_GENERATION_LIMIT_SECONDS,
        phase_evidence=phase,
    )


def _memory_censor(cell) -> dict[str, object]:
    peak = X86_EPYC_MEMORY_LIMIT_BYTES + 1
    return policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        cell,
        kind=PolicyCensorKind.MEMORY_LIMIT,
        source_identity=_POLICY_IDENTITY,
        resources=_policy_resources(peak),
        observed_rss_bytes=peak,
    )


def test_all_fifteen_matrices_render_in_catalog_order(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)
    expected = [dataset.table_name for dataset in REPORT_CATALOG.matrix_datasets]

    assert len(rendered) == 15
    assert list(rendered) == expected
    lc_tex = next(
        rendered[dataset.table_name]
        for dataset in REPORT_CATALOG.matrix_datasets
        if dataset.candidate.accuracy is Accuracy.LC
    )
    contracted_tex = next(
        rendered[dataset.table_name]
        for dataset in REPORT_CATALOG.matrix_datasets
        if dataset.candidate.accuracy is Accuracy.NLC
    )
    assert r"\textbf{ID} & \textbf{base process} & \textbf{metric}" in lc_tex
    assert r"\multicolumn{8}{c}{\textbf{n=1}}" in lc_tex
    assert r"\multicolumn{8}{c}{\textbf{n=2}}" in lc_tex
    assert r"\multicolumn{3}{c}{\textbf{n=1}}" in contracted_tex
    assert r"\multicolumn{3}{c}{\textbf{n=2}}" in contracted_tex
    assert r"\matrixfitwidth{%" in lc_tex
    assert r"\rowcolor{refblue}" in lc_tex
    assert lc_tex.count(r"\multicolumn{8}{c}") >= 4
    tabular_spec = next(
        line for line in lc_tex.splitlines() if line.startswith(r"\begin{tabular}")
    )
    assert r"@{\hspace" not in tabular_spec
    assert ":" not in tabular_spec
    assert r"\dashlinedash" not in lc_tex
    assert r"\dashlinegap" not in lc_tex
    assert r"\texttt{abs }" not in lc_tex
    assert r"\textcolor{ReportMuted}{\scriptsize gen. [s]}" in lc_tex
    assert (
        r"\textcolor{ReportMuted}{\scriptsize run "
        r"[\(\mu\mathrm{s}/\mathrm{pt}\)]}"
    ) in lc_tex
    assert r"\matrixcolumnheading" not in lc_tex

    obsolete_clock_macros = (
        r"\matrixwallclock{",
        r"\matrixwallabsolute{",
        r"\matrixtotalevaluator{",
        r"\matrixrecurrencecore{",
        r"\matrixruntimepair{",
        r"\matrixruntimetriple{",
    )
    best_mode_rendered = render_all_best_mode_tables(reset_caches)
    for tex in (*rendered.values(), *best_mode_rendered.values()):
        table_body = "\n".join(
            line for line in tex.splitlines() if not line.startswith(r"\providecommand")
        )
        assert all(macro not in table_body for macro in obsolete_clock_macros)


def test_identical_quark_line_family_renders_in_every_matrix_block(
    reset_caches,
) -> None:
    rendered = render_all_matrix_tables(reset_caches)
    best = render_all_best_mode_tables(reset_caches)
    row = (
        r"\texttt{15} & "
        r"$d\bar d\to u\bar u\,u\bar u+(n-4)g$"
    )

    for name, expected_blocks in (
        ("result_matrix_recurrence_builtin_sm_lc_table.tex", 3),
        ("result_matrix_recurrence_builtin_sm_nlc_table.tex", 2),
        ("result_matrix_recurrence_builtin_sm_full_table.tex", 2),
    ):
        assert rendered[name].count(row) == expected_blocks
    assert best["result_matrix_best_builtin_sm_lc_table.tex"].count(row) == 3
    assert best["result_matrix_best_builtin_sm_nlc_table.tex"].count(row) == 2
    assert best["result_matrix_best_builtin_sm_full_table.tex"].count(row) == 2


def test_matrix_summary_backgrounds_restart_for_each_multiplicity_block(
    reset_caches,
) -> None:
    rendered = render_all_matrix_tables(reset_caches)

    def summary_patterns(tex: str, span: int) -> list[list[bool]]:
        rows = [line for line in tex.splitlines() if r"\textbf{summary:" in line]
        return [
            [
                cell.startswith(rf"\multicolumn{{{span}}}{{c}}{{\cellcolor{{refblue}}")
                for cell in row.split(" & ")[1:]
            ]
            for row in rows
        ]

    lc_tex = next(
        rendered[dataset.table_name]
        for dataset in REPORT_CATALOG.matrix_datasets
        if dataset.candidate.accuracy is Accuracy.LC
    )
    contracted_tex = next(
        rendered[dataset.table_name]
        for dataset in REPORT_CATALOG.matrix_datasets
        if dataset.candidate.accuracy is Accuracy.NLC
    )

    def assert_alternating(tex: str, span: int) -> None:
        rows = [line for line in tex.splitlines() if r"\textbf{summary:" in line]
        patterns = summary_patterns(tex, span)
        for pattern in patterns:
            assert pattern == [index % 2 == 0 for index in range(len(pattern))]
        assert all(
            not row.startswith(r"\multicolumn{3}{@{}l}{\cellcolor{refblue}")
            for row in tex.splitlines()
            if r"\textbf{summary:" in row
        )
        assert all(row.endswith(r" \\[0.08em]") for row in rows[::2])
        assert all(row.endswith(" \\\\") for row in rows[1::2])
        assert r"\addlinespace[0.08em]" not in tex

        summary_headers = [
            line
            for line in tex.splitlines()
            if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\textbf{n=" in line
        ]
        assert len(summary_headers) == len(rows) // 2
        for header in summary_headers:
            groups = header.count(r"\textbf{n=")
            assert groups in (2, 3)
            assert [
                cell.startswith(rf"\multicolumn{{{span}}}{{c}}{{\cellcolor{{refblue}}")
                for cell in header.split(" & ")[1:]
            ] == [index % 2 == 0 for index in range(groups)]

    assert_alternating(lc_tex, 8)
    assert_alternating(contracted_tex, 3)

    for accuracy, span in ((Accuracy.LC, 8), (Accuracy.NLC, 3)):
        best_tex = render_best_mode_table(accuracy, reset_caches)
        assert_alternating(best_tex, span)

    fixed_block_two_header = next(
        line
        for line in lc_tex.splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\textbf{n=4}" in line
    )
    best_block_two_header = next(
        line
        for line in render_best_mode_table(Accuracy.LC, reset_caches).splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\textbf{n=4}" in line
    )
    expected_block_two_header = (
        r"\multicolumn{3}{@{}l}{} & "
        r"\multicolumn{8}{c}{\cellcolor{refblue}\textbf{n=4}} & "
        r"\multicolumn{8}{c}{\textbf{n=5}} & "
        r"\multicolumn{8}{c}{\cellcolor{refblue}\textbf{n=6}} \\"
    )
    assert fixed_block_two_header == expected_block_two_header
    assert best_block_two_header == expected_block_two_header

    lc_runtime_summary_label = (
        r"\matrixsummaryworkloads"
        r"{\textbf{summary: run single-flow, hel. sum}}"
        r"{\textbf{summary: run all-flows, single hel.}}"
    )
    best_lc_tex = render_best_mode_table(Accuracy.LC, reset_caches)
    assert lc_runtime_summary_label in lc_tex
    assert lc_runtime_summary_label in best_lc_tex
    assert r"\textbf{summary: wall}" not in lc_tex
    assert r"\textbf{summary: wall}" not in best_lc_tex
    assert r"\textbf{summary: wall}" in contracted_tex

    for tex in (lc_tex, contracted_tex, best_lc_tex):
        table_body = "\n".join(
            line for line in tex.splitlines() if not line.startswith(r"\providecommand")
        )
        assert r"\matrixtotalevaluator{" not in table_body
        assert r"\matrixrecurrencecore{" not in table_body


def test_best_mode_summary_selects_wall_winner_per_lc_workload(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidates = {
        mode: _cache_by_dataset(
            caches,
            f"matrix_{mode.value}_builtin_sm_lc",
        )
        for mode in (
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        )
    }
    for workload in (Workload.SELECTED_FLOW, Workload.ALL_FLOW):
        _set_ok(
            reference,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=10.0,
            wall=10.0e-6,
            execution=None,
        )
    timings = {
        ExecutionMode.RECURRENCE: (3.0e-6, 4.0e-6),
        ExecutionMode.COMPILED: (1.0e-6, 2.0e-6),
        ExecutionMode.EAGER: (2.0e-6, 1.0e-6),
    }
    generations = {
        ExecutionMode.RECURRENCE: 6.0,
        ExecutionMode.COMPILED: 4.0,
        ExecutionMode.EAGER: 5.0,
    }
    for mode, (selected_wall, all_flow_wall) in timings.items():
        for workload, wall in (
            (Workload.SELECTED_FLOW, selected_wall),
            (Workload.ALL_FLOW, all_flow_wall),
        ):
            _set_ok(
                candidates[mode],
                process_key="dd_z_jets",
                n_final=1,
                workload=workload,
                generation=generations[mode],
                wall=wall,
                execution=wall,
            )

    adapter = BaselineCandidateAdapter(caches)
    view = adapter.best_mode_cell(
        Accuracy.LC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].mode is ExecutionMode.COMPILED
    assert view.workloads[1].mode is ExecutionMode.EAGER

    tex = render_best_mode_table(Accuracy.LC, caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )
    assert (
        r"\textcolor{ReportMuted}{\scriptsize gen. [s]}"
        r" & \texttt{10.0} & \matrixpunct{|} & \texttt{10.0} & "
        r"\bestmodecodeprefix{c} & "
        r"\bestmoderatio{ReportGreen}{0.400} & "
        r"\matrixpunct{|} & "
        r"\bestmodeabsoluteprefix{\texttt{5.00}} & "
        r"\bestmodecode{e}"
    ) in row
    assert runtime_row.count(r" &  & \bestmodewallratio{ReportGreen}{0.100}") == 2
    assert r"\bestmodeopenprefix" not in runtime_row
    assert r"\bestmodecode" not in runtime_row
    assert (
        r"\providecommand{\bestmodecompactprefix}[1]{"
        r"\matrixpunct{(}\textcolor{ReportMuted}{\texttt{[x#1]}}"
        r"\hspace{0.04in}}"
    ) in tex
    assert (
        r"\providecommand{\bestmodemix}[1]{"
        r"\textcolor{ReportBlue}{\texttt{[#1]}}}"
    ) in tex
    assert (
        r"\providecommand{\matrixsummaryfont}{"
        r"\fontsize{6.2pt}{7.4pt}\selectfont}"
    ) in tex
    assert r"\setlength{\fboxsep}{0.9pt}" in tex
    assert r"\setlength{\fboxrule}{0.35pt}" in tex
    assert r"\usefont{T1}{lmtt}{b}{n}x#2" in tex
    assert r">{\matrixentryfontlc}r@{}>{\matrixentryfontlc}l" in tex
    assert (
        r"\textbf{metric} & \multicolumn{8}"
        r"{c}{\textbf{n=1}}"
    ) in tex
    assert r"\matrixcolumnheading" not in tex
    assert r"\shortstack{\textbf{n=" not in tex
    assert r"\bestmodecodeprefix{c}" in row
    assert r"\bestmodecode{e}" in row
    assert r"\matrixtotalevaluator{" not in row
    assert r"\matrixrecurrencecore{" not in row
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    summary_header = next(
        line
        for line in tex.splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\bestmodemix{" in line
    )
    assert (
        r"\textbf{n=1}\hspace{0.08in}\bestmodemix{r:0|c:1|e:0|o:0}"
        in summary_header
    )
    assert summary_header.count(r"\bestmodemix{") == 1
    assert r"r:0|c:0|e:1" not in summary_header
    assert r"\bestmodemix{" not in generation_summary
    assert r"\bestmodesummarystats" not in tex
    assert generation_summary.count(r"\matrixsummaryratio{ReportGreen}{0.400}") == 4
    assert (
        generation_summary.count(r"\matrixsummaryratiohighlight{ReportGreen}{0.400}")
        == 1
    )
    assert r"\texttt{10.0}" not in generation_summary
    assert r"\texttt{5.00}" not in generation_summary


@pytest.mark.parametrize(
    ("accuracy", "expected_mode_counts"),
    (
        (
            Accuracy.LC,
            (
                "r:2|c:0|e:0|o:0",
                "r:8|c:0|e:0|o:0",
                "r:9|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
                "r:14|c:0|e:0|o:0",
            ),
        ),
        (
            Accuracy.NLC,
            (
                "r:2|c:0|e:0",
                "r:8|c:0|e:0",
                "r:9|c:0|e:0",
                "r:14|c:0|e:0",
                "r:14|c:0|e:0",
                "r:2|c:0|e:0",
            ),
        ),
        (
            Accuracy.FULL,
            (
                "r:2|c:0|e:0",
                "r:8|c:0|e:0",
                "r:9|c:0|e:0",
                "r:14|c:0|e:0",
                "r:14|c:0|e:0",
                "r:2|c:0|e:0",
            ),
        ),
    ),
)
def test_best_mode_summary_headers_hold_one_exact_mode_mix_per_populated_group(
    reset_caches: dict[str, dict[str, object]],
    accuracy: Accuracy,
    expected_mode_counts: tuple[str, ...],
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_scope(caches, max_n_final=9)

    populated_tex = render_best_mode_table(accuracy, caches)
    populated_headers = tuple(
        line
        for line in populated_tex.splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\textbf{n=" in line
    )
    populated_body = "\n".join(
        line
        for line in populated_tex.splitlines()
        if not line.startswith(r"\providecommand")
    )
    assert (
        tuple(
            token
            for header in populated_headers
            for token in re.findall(r"\\bestmodemix\{([^}]*)\}", header)
        )
        == expected_mode_counts
    )
    assert all(
        header.count(r"\bestmodemix{") == header.count(r"\textbf{n=")
        for header in populated_headers
    )
    assert populated_body.count(r"\bestmodemix{") == len(expected_mode_counts)
    assert all(
        r"\bestmodemix{" not in line
        for line in populated_tex.splitlines()
        if r"\textbf{summary:" in line
    )
    assert r"\bestmodesummarystats" not in populated_tex

    blank_tex = render_best_mode_table(accuracy, reset_caches)
    blank_headers = tuple(
        line
        for line in blank_tex.splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{} & ") and r"\textbf{n=" in line
    )
    blank_body = "\n".join(
        line
        for line in blank_tex.splitlines()
        if not line.startswith(r"\providecommand")
    )
    assert sum(header.count(r"\textbf{n=") for header in blank_headers) == len(
        expected_mode_counts
    )
    assert r"\bestmodemix{" not in blank_body
    assert r"\bestmodesummarystats" not in blank_tex


def test_ratio_summary_reports_five_exact_statistics_without_absolute_times() -> None:
    summary = report_render._ratio_statistics_tex(((1.0, 1.0), (3.0, 9.0)))

    assert summary == (
        r"\matrixsummarystats{"
        r"\matrixsummaryratio{ReportOrange}{1.00}}{"
        r"\matrixsummaryratio{ReportRed}{3.00}}{"
        r"\matrixsummaryratio{ReportRed}{2.00}}{"
        r"\matrixsummaryratio{ReportRed}{2.00}}{"
        r"\matrixsummaryratiohighlight{ReportRed}{2.50}}"
    )
    assert r"\texttt{" not in summary


def test_summary_statistics_share_fixed_anchors_and_compact_notes(
    reset_caches,
) -> None:
    fixed_tex = render_matrix_table(
        next(
            dataset
            for dataset in REPORT_CATALOG.matrix_datasets
            if dataset.candidate.accuracy is Accuracy.LC
        ),
        reset_caches,
    )
    best_tex = render_best_mode_table(Accuracy.LC, reset_caches)
    summary_tables = tuple(
        tex
        for tex in render_all_tables(reset_caches).values()
        if r"\providecommand{\matrixsummarystats}" in tex
    )
    separator = r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}"
    summary_column_layout = (
        r"\begin{tabular}[t]{@{}l"
        + separator
        + "l"
        + separator
        + "l"
        + separator
        + "l"
        + separator
        + r"l@{}}"
    )
    fixed_slot_layout = (
        r"\makebox[3.6em][l]{#1}&"
        r"\makebox[4.6em][l]{#2}&"
        r"\makebox[3.6em][l]{#3}&"
        r"\makebox[3.6em][l]{#4}&"
        r"\makebox[4.2em][l]{#5}"
    )

    assert len(summary_tables) == 22
    for tex in summary_tables:
        assert summary_column_layout in tex
        assert fixed_slot_layout in tex
        assert r"\makebox[3.6em][r]" not in tex
        assert r"\makebox[4.6em][r]" not in tex
        assert r"\makebox[4.2em][r]" not in tex

    for tex in (fixed_tex, best_tex):
        assert r"\ReportTableNote{{\scriptsize " in tex
        assert "Every displayed number uses exactly three significant digits" in tex
        assert "weighted-average order" in tex
        assert "weighted average" in tex
        assert "framed bold entry" in tex

    assert "Fixed-engine tables intentionally omit mode letters" in fixed_tex
    assert "selected independently in each cell and workload" in best_tex
    assert "Summary mode counts use r|c|e" in best_tex


def test_lc_summaries_omit_union_generation_and_keep_both_wall_workloads(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_lc",
    )
    candidate = _cache_by_dataset(
        caches,
        "matrix_compiled_builtin_sm_lc",
    )
    selected_pairs = ((1.0, 1.0), (3.0, 9.0))
    union_pairs = ((2.0, 20.0), (4.0, 80.0))
    for process_key, selected, union in zip(
        ("dd_z_jets", "ud_w_jets"),
        selected_pairs,
        union_pairs,
        strict=True,
    ):
        for workload, (baseline_value, candidate_value) in (
            (Workload.SELECTED_FLOW, selected),
            (Workload.ALL_FLOW, union),
        ):
            _set_ok(
                baseline,
                process_key=process_key,
                n_final=1,
                workload=workload,
                generation=baseline_value,
                wall=baseline_value,
                execution=baseline_value,
            )
            _set_ok(
                candidate,
                process_key=process_key,
                n_final=1,
                workload=workload,
                generation=candidate_value,
                wall=candidate_value,
                execution=candidate_value,
            )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_lc"),
        caches,
    )
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line
        for line in tex.splitlines()
        if r"\textbf{summary: run single-flow, hel. sum}" in line
    )
    selected_stats = report_render._ratio_statistics_tex(selected_pairs)
    union_stats = report_render._ratio_statistics_tex(union_pairs)

    assert selected_stats in generation_summary
    assert union_stats not in generation_summary
    assert r"\matrixsummaryworkloads" not in generation_summary
    assert rf"\matrixsummaryworkloads{{{selected_stats}}}{{{union_stats}}}" in (
        wall_summary
    )
    assert (
        r"\matrixsummaryworkloads"
        r"{\textbf{summary: run single-flow, hel. sum}}"
        r"{\textbf{summary: run all-flows, single hel.}}"
    ) in wall_summary
    assert r"\textbf{summary: wall}" not in tex
    assert r"\matrixsummarypair" not in generation_summary
    assert r"\matrixsummarypair" not in wall_summary


def test_best_mode_summary_tie_breaks_in_documented_mode_order(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    for workload in (Workload.SELECTED_FLOW, Workload.ALL_FLOW):
        _set_ok(
            _cache_by_dataset(caches, "reference_amplicol_lc"),
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=10.0,
            wall=10.0e-6,
            execution=None,
        )
        for mode in (
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        ):
            _set_ok(
                _cache_by_dataset(
                    caches,
                    f"matrix_{mode.value}_builtin_sm_lc",
                ),
                process_key="dd_z_jets",
                n_final=1,
                workload=workload,
                generation=5.0,
                wall=1.0e-6,
                execution=1.0e-6,
            )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.LC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert {item.mode for item in view.workloads} == {ExecutionMode.RECURRENCE}
    rendered = render_all_best_mode_tables(caches)
    assert list(rendered) == [
        "result_matrix_best_builtin_sm_lc_table.tex",
        "result_matrix_best_builtin_sm_nlc_table.tex",
        "result_matrix_best_builtin_sm_full_table.tex",
    ]


@pytest.mark.parametrize(
    ("outcome", "label"),
    (
        ("error", "error"),
        ("validation_failed", "validation failed"),
        ("blocked_dependency", "blocked dependency"),
        ("dependency_backend_error", "dependency backend error"),
    ),
)
def test_fixed_matrix_renders_generic_presentation_outcomes(
    reset_caches,
    outcome: str,
    label: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_presentation_outcome(
        _cache_by_dataset(caches, "matrix_compiled_builtin_sm_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome=outcome,
    )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity=f"presentation:{outcome}",
            label=label,
            color="ReportRed",
        )
    )
    marker = rf"\matrixstatus{{ReportRed}}{{{display}}}"

    assert tex.count(marker) == 2
    assert "ManualCampaignOutcome" not in tex
    if outcome == "blocked_dependency":
        assert display == "blocked dep."
        assert label not in tex
    if outcome == "dependency_backend_error":
        assert display == "depe back erro"
        assert label not in tex
    if "_" in outcome:
        assert outcome not in tex


def test_fixed_matrix_keeps_policy_like_presentation_outcomes_orange(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "matrix_compiled_builtin_sm_nlc")
    _set_presentation_outcome(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="generation_limit",
        label=">1h",
    )
    _set_presentation_outcome(
        candidate,
        process_key="ud_w_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="error",
    )
    _set_presentation_outcome(
        candidate,
        process_key="dd_z_jets",
        n_final=2,
        workload=Workload.CONTRACTED,
        outcome="resource_frontier",
    )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )

    assert tex.count(r"\matrixstatus{ReportOrange}{>1h}") == 2
    assert tex.count(r"\matrixstatus{ReportRed}{error}") == 2
    assert tex.count(r"\matrixstatus{ReportOrange}{resource frontier}") == 2
    assert r"\matrixstatus{ReportRed}{>1h}" not in tex
    assert r"\matrixstatus{ReportOrange}{error}" not in tex


@pytest.mark.parametrize(
    "label",
    (
        "W" * 64,
        "W" * 12,
        "W" * 11 + " " + "W" * 11,
        "w" * 12 + " " + "w" * 11,
        "Q" * 12 + " " + "Q" * 11,
        "O" * 12 + " " + "O" * 11,
    ),
)
def test_fixed_matrix_compacts_one_worst_case_wide_future_label(
    reset_caches,
    label: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "matrix_compiled_builtin_sm_nlc")
    _set_presentation_outcome(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="future_wide_status",
        label=label,
    )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="presentation:future_wide_status",
            label=label,
            color="ReportRed",
        )
    )

    assert display == "futu wide stat"
    assert tex.count(rf"\matrixstatus{{ReportRed}}{{{display}}}") == 2
    assert label not in tex


def test_known_error_slug_does_not_exempt_a_wide_mismatched_label() -> None:
    label = "W" * 64
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="presentation:error",
            label=label,
            color="ReportRed",
        )
    )

    assert display == "error"


def test_policy_slug_does_not_exempt_an_unrealistically_wide_cap_label() -> None:
    label = ">" + "9" * 60 + "GB"
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="presentation:memory_limit",
            label=label,
            color="ReportOrange",
        )
    )

    assert display == "memo limi"


def test_policy_slug_compacts_a_too_wide_otherwise_valid_cap_label() -> None:
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="presentation:dependency",
            label="dependency >9999.9999s",
            color="ReportOrange",
        )
    )

    assert display == "dependency"


def test_authenticated_policy_state_keeps_semantics_for_an_overwide_cap() -> None:
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="policy:dependency",
            label="dependency >9999.9999s",
            color="ReportOrange",
        )
    )

    assert display == "dependency"


@pytest.mark.parametrize(
    ("outcome", "label"),
    (
        ("error", "error"),
        ("validation_failed", "validation failed"),
        ("blocked_dependency", "blocked dependency"),
        ("dependency_backend_error", "dependency backend error"),
    ),
)
def test_best_mode_renders_failure_only_generic_presentation_outcomes(
    reset_caches,
    outcome: str,
    label: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_best_mode_nlc_reference(caches)
    for mode in ExecutionMode.RECURRENCE, ExecutionMode.COMPILED, ExecutionMode.EAGER:
        _set_presentation_outcome(
            _cache_by_dataset(
                caches,
                f"matrix_{mode.value}_builtin_sm_nlc",
            ),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
            outcome=outcome,
        )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity=f"presentation:{outcome}",
            label=label,
            color="ReportRed",
        )
    )
    marker = rf"\matrixstatus{{ReportRed}}{{{display}}}"

    assert tex.count(marker) == 4
    table_body = "\n".join(
        line for line in tex.splitlines() if not line.startswith(r"\providecommand")
    )
    assert r"\bestmodecode{" not in table_body


def test_best_mode_mixed_generic_outcomes_have_deterministic_labels(
    reset_caches,
) -> None:
    assignments = (
        ("error", "validation_failed", "blocked_dependency"),
        ("error", "blocked_dependency", "validation_failed"),
    )
    rendered_tables: list[str] = []
    for outcomes in assignments:
        caches = copy.deepcopy(reset_caches)
        _set_best_mode_nlc_reference(caches)
        for mode, outcome in zip(
            (
                ExecutionMode.RECURRENCE,
                ExecutionMode.COMPILED,
                ExecutionMode.EAGER,
            ),
            outcomes,
            strict=True,
        ):
            _set_presentation_outcome(
                _cache_by_dataset(
                    caches,
                    f"matrix_{mode.value}_builtin_sm_nlc",
                ),
                process_key="dd_z_jets",
                n_final=1,
                workload=Workload.CONTRACTED,
                outcome=outcome,
            )

        tex = render_best_mode_table(Accuracy.NLC, caches)
        view = BaselineCandidateAdapter(caches).best_mode_cell(
            Accuracy.NLC,
            REPORT_CATALOG.process_families[0],
            1,
        )
        joined = view.workloads[0]
        marker = r"\matrixstatus{ReportRed}{error}"

        assert tex.count(marker) == 4
        assert joined.mode is ExecutionMode.RECURRENCE
        assert joined.terminal_label is None
        assert r"\bestmodecodeprefix{r}" in tex
        assert r"\matrixstatus{ReportRed}{validation failed}" not in tex
        assert r"\matrixstatus{ReportRed}{blocked dep.}" not in tex
        rendered_tables.append(tex)

    assert rendered_tables[0] == rendered_tables[1]


def test_best_mode_compacts_three_maximum_length_future_outcomes(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_best_mode_nlc_reference(caches)
    outcomes = (
        ("future_alpha", "alpha_" + "a" * 58),
        ("future_bravo", "bravo&" + "b" * 58),
        ("future_charlie", "charl%" + "c" * 58),
    )
    assert all(len(label) == 64 for _outcome, label in outcomes)
    for mode, (outcome, label) in zip(
        (
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        ),
        outcomes,
        strict=True,
    ):
        _set_presentation_outcome(
            _cache_by_dataset(
                caches,
                f"matrix_{mode.value}_builtin_sm_nlc",
            ),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
            outcome=outcome,
            label=label,
        )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.NLC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    joined = view.workloads[0]
    assert joined.mode is ExecutionMode.RECURRENCE
    assert joined.terminal_label is None

    tex = render_best_mode_table(Accuracy.NLC, caches)
    marker = r"\matrixstatus{ReportRed}{futu alph}"
    assert tex.count(marker) == 4
    assert r"\bestmodecodeprefix{r}" in tex
    assert all(label not in tex for _outcome, label in outcomes)


def test_best_mode_many_unique_outcomes_have_one_hard_bounded_identity() -> None:
    outcomes = tuple(
        report_render._TerminalOutcome(
            identity=f"presentation:future_{index:02d}",
            label=f"future outcome {index:02d}",
            color="ReportRed",
        )
        for index in range(12)
    )

    summary = report_render._canonical_best_mode_terminal_label(outcomes)
    reversed_summary = report_render._canonical_best_mode_terminal_label(
        tuple(reversed(outcomes))
    )

    assert summary is not None
    assert reversed_summary is not None
    assert summary.outcomes == reversed_summary.outcomes
    assert summary.label == reversed_summary.label
    assert len(summary.label) <= 48
    assert re.fullmatch(r"12 outcomes \[[0-9a-f]{6}\]", summary.label)
    assert {item.identity for item in summary.outcomes} == {
        item.identity for item in outcomes
    }
    assert re.fullmatch(
        r"N out\.\[[0-9a-f]{6}\]",
        report_render._compact_terminal_summary_display(summary),
    )


@pytest.mark.parametrize("count", (12, 1796))
def test_best_mode_outcome_count_display_has_fixed_width(count: int) -> None:
    outcome = report_render._TerminalOutcome(
        identity="presentation:error",
        label="error",
        color="ReportRed",
    )
    summary = report_render._TerminalSummary(
        outcomes=(outcome, outcome),
        label=f"{count} outcomes [abcdef]",
        color="ReportRed",
    )

    assert report_render._compact_terminal_summary_display(summary) == (
        "N out.[abcdef]"
    )


def test_best_mode_success_wins_over_generic_presentation_outcomes(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_best_mode_nlc_reference(caches)
    _set_presentation_outcome(
        _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="error",
    )
    _set_ok(
        _cache_by_dataset(caches, "matrix_compiled_builtin_sm_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=1.0e-6,
        execution=1.0e-6,
    )
    _set_presentation_outcome(
        _cache_by_dataset(caches, "matrix_eager_builtin_sm_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="blocked_dependency",
    )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.NLC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].mode is ExecutionMode.COMPILED

    tex = render_best_mode_table(Accuracy.NLC, caches)
    generation_row = next(
        line for line in tex.splitlines() if line.startswith(r"\texttt{1}")
    )
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )

    assert r"\bestmodecode{c}" in generation_row
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
    assert r"\bestmodeabsoluteprefix{\texttt{1.00}}" in runtime_row
    assert r"\bestmoderatio{" not in generation_row
    assert r"\bestmodewallratio{" not in runtime_row
    assert r"\bestmodemix{r:0|c:1|e:0}" not in tex
    assert r"\matrixstatus{ReportRed}{error}" not in tex
    assert r"\matrixstatus{ReportRed}{blocked dependency}" not in tex


def test_best_mode_uses_an_exact_equivalent_z_success_over_a_matrix_cap(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_ok(
        _cache_by_dataset(caches, "reference_amplicol_lc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )
    for mode in (ExecutionMode.RECURRENCE, ExecutionMode.COMPILED):
        entry = _entry(
            _cache_by_dataset(caches, f"matrix_{mode.value}_builtin_sm_lc"),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.SELECTED_FLOW,
        )
        entry["measurement"] = _memory_censor(
            REPORT_CATALOG.cell(str(entry["cell_id"]))
        )
    _set_ok(
        _cache_by_dataset(caches, "z_builtin_sm"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        variant="recurrence_jit_o2",
        generation=7.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    equivalent_terminal = _entry(
        _cache_by_dataset(caches, "z_builtin_sm"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.ALL_FLOW,
        variant="recurrence_jit_o2",
    )
    equivalent_terminal["measurement"] = _generation_censor(
        REPORT_CATALOG.cell(str(equivalent_terminal["cell_id"]))
    )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.LC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    selected = view.workloads[0]
    assert selected.mode is ExecutionMode.RECURRENCE
    assert selected.candidate["status"] == ResultStatus.OK.value
    assert selected.candidate["wall_seconds_per_point"] == 2.0e-6
    assert selected.comparison_linked
    all_flow = view.workloads[1]
    assert all_flow.mode is ExecutionMode.RECURRENCE
    assert all_flow.candidate["status"] == ResultStatus.TIMEOUT.value

    tex = render_best_mode_table(Accuracy.LC, caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    assert row.count(r"\bestmodecodeprefix{r}") == 2
    assert r"\matrixstatus{ReportOrange}{>2h}" in row
    assert r"\matrixstatus{ReportOrange}{>80GB}" not in tex


def test_best_mode_prefers_its_owned_success_before_equivalent_success(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_ok(
        _cache_by_dataset(caches, "reference_amplicol_lc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )
    _set_ok(
        _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_lc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=5.0,
        wall=5.0e-6,
        execution=4.0e-6,
    )
    _set_ok(
        _cache_by_dataset(caches, "z_builtin_sm"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        variant="recurrence_jit_o2",
        generation=1.0,
        wall=0.5e-6,
        execution=0.4e-6,
    )
    _set_ok(
        _cache_by_dataset(caches, "matrix_compiled_builtin_sm_lc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=2.0,
        wall=2.0e-6,
        execution=1.5e-6,
    )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.LC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    selected = view.workloads[0]
    assert selected.mode is ExecutionMode.COMPILED
    assert selected.candidate["wall_seconds_per_point"] == 2.0e-6


def test_best_mode_mixed_policy_censors_fall_back_to_recurrence(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_ok(
        _cache_by_dataset(caches, "reference_amplicol_nlc"),
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )
    candidate_entries = {
        mode: _entry(
            _cache_by_dataset(
                caches,
                f"matrix_{mode.value}_builtin_sm_nlc",
            ),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
        )
        for mode in (
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        )
    }
    recurrence_cell = REPORT_CATALOG.cell(
        str(candidate_entries[ExecutionMode.RECURRENCE]["cell_id"])
    )
    recurrence_censor = _generation_censor(recurrence_cell)
    candidate_entries[ExecutionMode.RECURRENCE]["measurement"] = recurrence_censor
    compiled_cell = REPORT_CATALOG.cell(
        str(candidate_entries[ExecutionMode.COMPILED]["cell_id"])
    )
    candidate_entries[ExecutionMode.COMPILED]["measurement"] = _memory_censor(
        compiled_cell
    )
    eager_cell = REPORT_CATALOG.cell(
        str(candidate_entries[ExecutionMode.EAGER]["cell_id"])
    )
    candidate_entries[ExecutionMode.EAGER]["measurement"] = policy_censor_measurement(
        X86_EPYC_POLICY,
        "x86_EPYC",
        eager_cell,
        kind=PolicyCensorKind.DEPENDENCY,
        source_identity=_POLICY_IDENTITY,
        resources=None,
        dependencies=(
            dependency_reference(
                recurrence_cell.cell_id,
                recurrence_censor,
            ),
        ),
    )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )

    marker = r"\matrixstatus{ReportOrange}{>2h}"
    assert marker in row
    assert marker in runtime_row
    assert (
        r">{\matrixentryfont}l"
        r">{\matrixentryfont}r@{}"
        r">{\matrixentryfont}l"
    ) in tex
    assert r"\bestmodecodeprefix{r}" in row
    assert r"\matrixstatus{ReportOrange}{>80GB}" not in row
    assert r"\matrixstatus{ReportOrange}{dependency}" not in row
    assert r"\matrixna{ReportMuted}" not in row


def test_best_mode_mixed_terminal_summaries_are_visibly_complete(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    recurrence = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    compiled = _cache_by_dataset(
        caches,
        "matrix_compiled_builtin_sm_nlc",
    )
    eager = _cache_by_dataset(caches, "matrix_eager_builtin_sm_nlc")
    recurrence_entries = recurrence["entries"]
    assert isinstance(recurrence_entries, list)
    for recurrence_entry in recurrence_entries:
        if recurrence_entry["n_final"] != 1:
            continue
        process_key = str(recurrence_entry["process_key"])
        recurrence_cell = REPORT_CATALOG.cell(str(recurrence_entry["cell_id"]))
        recurrence_censor = _generation_censor(recurrence_cell)
        recurrence_entry["measurement"] = recurrence_censor
        compiled_entry = _entry(
            compiled,
            process_key=process_key,
            n_final=1,
            workload=Workload.CONTRACTED,
        )
        compiled_entry["measurement"] = _memory_censor(
            REPORT_CATALOG.cell(str(compiled_entry["cell_id"]))
        )
        eager_entry = _entry(
            eager,
            process_key=process_key,
            n_final=1,
            workload=Workload.CONTRACTED,
        )
        eager_entry["measurement"] = policy_censor_measurement(
            X86_EPYC_POLICY,
            "x86_EPYC",
            REPORT_CATALOG.cell(str(eager_entry["cell_id"])),
            kind=PolicyCensorKind.DEPENDENCY,
            source_identity=_POLICY_IDENTITY,
            resources=None,
            dependencies=(
                dependency_reference(
                    recurrence_cell.cell_id,
                    recurrence_censor,
                ),
            ),
        )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: wall}" in line
    )
    marker = r"\matrixstatus{ReportOrange}{>2h}"
    assert generation_summary.count(marker) == 1
    assert wall_summary.count(marker) == 1

    completeness = summarize_visible_completeness(
        caches,
        max_n_final=4,
        policy=X86_EPYC_POLICY,
    )
    assert completeness.complete
    assert not completeness.applicable_na_display_slots


def test_best_mode_terminal_baselines_stay_in_rows_but_not_summaries(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_nlc")
    entries = reference["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        if entry["n_final"] == 1:
            entry["measurement"] = _memory_censor(
                REPORT_CATALOG.cell(str(entry["cell_id"]))
            )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    lines = tex.splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    generation_row = lines[row_index]
    runtime_row = lines[row_index + 2]
    marker = r"\matrixstatus{ReportOrange}{>80GB}"
    assert generation_row.count(marker) == 1
    assert runtime_row.count(marker) == 1
    assert r"\bestmodeabsoluteprefix{\texttt{1.00}}" in generation_row
    assert r"\bestmodecode{r}" in generation_row
    assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in runtime_row
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: wall}" in line
    )
    assert marker not in generation_summary
    assert marker not in wall_summary
    assert r"\matrixna{ReportMuted}" not in generation_row
    assert r"\matrixna{ReportMuted}" not in runtime_row
    assert r"\matrixna{ReportMuted}" in generation_summary
    assert r"\matrixna{ReportMuted}" in wall_summary


@pytest.mark.parametrize(
    ("accuracy", "workloads"),
    (
        (Accuracy.LC, (Workload.SELECTED_FLOW, Workload.ALL_FLOW)),
        (Accuracy.NLC, (Workload.CONTRACTED,)),
        (Accuracy.FULL, (Workload.CONTRACTED,)),
    ),
)
def test_recurrence_renders_absolute_when_amplicol_baseline_is_terminal(
    reset_caches,
    accuracy: Accuracy,
    workloads: tuple[Workload, ...],
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, f"reference_amplicol_{accuracy.value}")
    candidate = _cache_by_dataset(
        caches,
        f"matrix_recurrence_builtin_sm_{accuracy.value}",
    )
    for index, workload in enumerate(workloads):
        baseline_entry = _entry(
            baseline,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
        )
        baseline_entry["measurement"] = _memory_censor(
            REPORT_CATALOG.cell(str(baseline_entry["cell_id"]))
        )
        _set_ok(
            candidate,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=4.0 + 6.0 * index,
            wall=(2.0 + index) * 1.0e-6,
            execution=(1.0 + 0.5 * index) * 1.0e-6,
        )

    dataset = next(
        item
        for item in REPORT_CATALOG.matrix_datasets
        if item.dataset_id == f"matrix_recurrence_builtin_sm_{accuracy.value}"
    )
    marker = r"\matrixstatus{ReportOrange}{>80GB}"
    for tex, best_mode in (
        (render_matrix_table(dataset, caches), False),
        (render_best_mode_table(accuracy, caches), True),
    ):
        lines = tex.splitlines()
        row_index = next(
            index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
        )
        generation_row = lines[row_index]
        runtime_row = lines[row_index + 2]
        assert generation_row.count(marker) == len(workloads)
        assert runtime_row.count(marker) == len(workloads)
        assert generation_row.count(r"\bestmodeabsoluteprefix{") == len(workloads)
        assert runtime_row.count(r"\bestmodeabsoluteprefix{") == len(workloads)
        assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
        assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in runtime_row
        assert r"\bestmoderatio{" not in generation_row
        assert r"\bestmodeprimaryratio{" not in runtime_row
        if best_mode:
            assert generation_row.count(r"\bestmodecode{r}") == len(workloads)

    best_tex = render_best_mode_table(accuracy, caches)
    generation_summary = next(
        line
        for line in best_tex.splitlines()
        if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line
        for line in best_tex.splitlines()
        if "summary: run" in line or r"\textbf{summary: wall}" in line
    )
    assert marker not in generation_summary
    assert marker not in wall_summary
    assert r"\matrixsummarystats{" not in generation_summary
    assert r"\matrixsummarystats{" not in wall_summary
    assert r"\matrixna{ReportMuted}" in generation_summary
    assert r"\matrixna{ReportMuted}" in wall_summary


def test_recurrence_renders_absolute_when_amplicol_was_not_run(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    dataset = next(
        item
        for item in REPORT_CATALOG.matrix_datasets
        if item.dataset_id == "matrix_recurrence_builtin_sm_nlc"
    )

    for tex in (
        render_matrix_table(dataset, caches),
        render_best_mode_table(Accuracy.NLC, caches),
    ):
        lines = tex.splitlines()
        row_index = next(
            index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
        )
        generation_row = lines[row_index]
        runtime_row = lines[row_index + 2]
        assert r"\matrixstatus{ReportMuted}{N/A}" in generation_row
        assert r"\matrixstatus{ReportMuted}{N/A}" in runtime_row
        assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
        assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in runtime_row
        assert r"\bestmoderatio{" not in generation_row
        assert r"\bestmodeprimaryratio{" not in runtime_row


def test_later_amplicol_current_does_not_retroactively_link_recurrence(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    reference = _cache_by_dataset(caches, "reference_amplicol_nlc")
    _set_ok(
        recurrence,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    recurrence_measurement = _entry(
        recurrence,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
    )["measurement"]
    assert isinstance(recurrence_measurement, dict)
    validation = recurrence_measurement["validation"]
    assert isinstance(validation, dict)
    validation.pop("pointwise")
    _set_ok(
        reference,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )

    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_nlc")
    view = BaselineCandidateAdapter(caches).matrix_cell(
        dataset,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert not view.workloads[0].comparison_linked
    tex = render_matrix_table(dataset, caches)
    generation_row = next(
        line for line in tex.splitlines() if line.startswith(r"\texttt{1}")
    )
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
    assert r"\bestmoderatio{" not in generation_row
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    assert r"\matrixsummarystats{" not in generation_summary
    assert r"\matrixna{ReportMuted}" in generation_summary
    assert "unlinked baselines keep their status" in tex


@pytest.mark.parametrize("mode", (ExecutionMode.COMPILED, ExecutionMode.EAGER))
@pytest.mark.parametrize("broken_chain", (False, True))
def test_best_dag_ratio_requires_exact_transitive_amplicol_link(
    reset_caches,
    mode: ExecutionMode,
    broken_chain: bool,
) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_nlc")
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    candidate = _cache_by_dataset(
        caches,
        f"matrix_{mode.value}_builtin_sm_nlc",
    )
    for cache, generation, wall in (
        (reference, 10.0, 10.0e-6),
        (recurrence, 6.0, 3.0e-6),
        (candidate, 2.0, 1.0e-6),
    ):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
            generation=generation,
            wall=wall,
            execution=wall,
        )
    if broken_chain:
        recurrence_measurement = _entry(
            recurrence,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
        )["measurement"]
        assert isinstance(recurrence_measurement, dict)
        validation = recurrence_measurement["validation"]
        assert isinstance(validation, dict)
        pointwise = validation["pointwise"]
        assert isinstance(pointwise, dict)
        pointwise["baseline"] = 2.0

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.NLC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].mode is mode
    assert view.workloads[0].comparison_linked is not broken_chain
    generation_row = next(
        line
        for line in render_best_mode_table(Accuracy.NLC, caches).splitlines()
        if line.startswith(r"\texttt{1}")
    )
    if broken_chain:
        assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in generation_row
        assert r"\bestmoderatio{" not in generation_row
    else:
        assert r"\bestmoderatio{ReportGreen}{0.200}" in generation_row
        assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" not in generation_row


@pytest.mark.parametrize("broken_selector", (False, True))
def test_direct_pointwise_link_requires_exact_selector(
    reset_caches,
    broken_selector: bool,
) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_lc")
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_lc")
    for cache, generation in ((reference, 10.0), (recurrence, 4.0)):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.SELECTED_FLOW,
            generation=generation,
            wall=generation * 1.0e-6,
            execution=1.0e-6,
        )
    if broken_selector:
        measurement = _entry(
            recurrence,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.SELECTED_FLOW,
        )["measurement"]
        assert isinstance(measurement, dict)
        selector = measurement["selector_contract"]
        assert isinstance(selector, dict)
        selector["point_digest"] = "1" * 64

    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc")
    view = BaselineCandidateAdapter(caches).matrix_cell(
        dataset,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].comparison_linked is not broken_selector
    generation_row = next(
        line
        for line in render_matrix_table(dataset, caches).splitlines()
        if line.startswith(r"\texttt{1}")
    )
    if broken_selector:
        assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
        assert r"\bestmoderatio{" not in generation_row
    else:
        assert r"\bestmoderatio{ReportGreen}{0.400}" in generation_row


@pytest.mark.parametrize("broken_component", (False, True))
def test_direct_agreement_link_requires_exact_component_identity(
    reset_caches,
    broken_component: bool,
) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_lc")
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_lc")
    for cache, generation in ((reference, 10.0), (recurrence, 4.0)):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.ALL_FLOW,
            generation=generation,
            wall=generation * 1.0e-6,
            execution=1.0e-6,
        )
    if broken_component:
        measurement = _entry(
            recurrence,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.ALL_FLOW,
        )["measurement"]
        assert isinstance(measurement, dict)
        validation = measurement["validation"]
        assert isinstance(validation, dict)
        component = validation[LC_COMMON_COMPONENT_FIELD]
        assert isinstance(component, dict)
        component["color_flow_ids"] = ["flow:other"]

    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc")
    view = BaselineCandidateAdapter(caches).matrix_cell(
        dataset,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[1].comparison_linked is not broken_component
    lines = render_matrix_table(dataset, caches).splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    runtime_row = lines[row_index + 2]
    if broken_component:
        assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in runtime_row
        assert r"\bestmodeprimaryratio{" not in runtime_row
    else:
        assert r"\bestmodeprimaryratio{ReportGreen}{0.400}" in runtime_row


def test_direct_agreement_v2_remains_comparison_linked(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_lc")
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_lc")
    for cache, generation in ((reference, 10.0), (recurrence, 4.0)):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.ALL_FLOW,
            generation=generation,
            wall=generation * 1.0e-6,
            execution=1.0e-6,
        )
    measurement = _entry(
        recurrence,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.ALL_FLOW,
    )["measurement"]
    assert isinstance(measurement, dict)
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    records = validation["direct_agreements"]
    assert isinstance(records, list) and records
    for record in records:
        assert isinstance(record, dict)
        comparison = pointwise_validation(
            float(record["candidate"]),
            float(record["baseline"]),
            relative_tolerance=float(record["relative_tolerance"]),
            comparison_binding={"point_digest": "0" * 64},
        )
        comparison.pop("abi")
        identity = {
            field: record[field]
            for field in (
                "edge_kind",
                "value_kind",
                "baseline_cell_id",
                "candidate_cell_id",
            )
        }
        record.clear()
        record.update({"abi": DIRECT_AGREEMENT_V2_ABI, **identity, **comparison})

    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc")
    view = BaselineCandidateAdapter(caches).matrix_cell(
        dataset,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[1].comparison_linked


@pytest.mark.parametrize("mode", (ExecutionMode.COMPILED, ExecutionMode.EAGER))
def test_fixed_compiled_and_eager_preserve_terminal_recurrence_baseline(
    reset_caches,
    mode: ExecutionMode,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    candidate = _cache_by_dataset(
        caches,
        f"matrix_{mode.value}_builtin_sm_nlc",
    )
    baseline_entry = _entry(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
    )
    baseline_entry["measurement"] = _memory_censor(
        REPORT_CATALOG.cell(str(baseline_entry["cell_id"]))
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    dataset = next(
        item
        for item in REPORT_CATALOG.matrix_datasets
        if item.dataset_id == f"matrix_{mode.value}_builtin_sm_nlc"
    )

    lines = render_matrix_table(dataset, caches).splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    marker = r"\matrixstatus{ReportOrange}{>80GB}"
    assert lines[row_index].count(marker) == 1
    assert lines[row_index + 2].count(marker) == 1
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in lines[row_index]
    assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in lines[row_index + 2]


@pytest.mark.parametrize("mode", (ExecutionMode.COMPILED, ExecutionMode.EAGER))
def test_fixed_compiled_and_eager_show_unverified_absolute_diagnostic_clocks(
    reset_caches,
    mode: ExecutionMode,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    candidate = _cache_by_dataset(
        caches,
        f"matrix_{mode.value}_builtin_sm_nlc",
    )
    baseline_entry = _entry(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
    )
    baseline_entry["measurement"] = _memory_censor(
        REPORT_CATALOG.cell(str(baseline_entry["cell_id"]))
    )
    _set_unverified(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        evaluator_total=1.0e-6,
    )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset(f"matrix_{mode.value}_builtin_sm_nlc"),
        caches,
    )
    lines = tex.splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    generation_row = lines[row_index]
    runtime_row = lines[row_index + 2]
    authority_marker = r"\matrixstatus{ReportOrange}{>80GB}"
    unverified_marker = r"\matrixstatus{ReportOrange}{unverified}"

    assert generation_row.count(authority_marker) == 1
    assert runtime_row.count(authority_marker) == 1
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in generation_row
    assert unverified_marker in generation_row
    assert (
        r"\bestmodeunverifiedclockprefix{\texttt{1.00}}{\texttt{2.00}}" in runtime_row
    )
    assert unverified_marker in runtime_row
    assert "Unverified compiled or eager diagnostics show absolute generation" in tex
    assert "never enter ratios or summaries" in tex


def test_best_mode_never_selects_or_summarizes_unverified_diagnostics(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _set_best_mode_nlc_reference(caches)
    for mode in ExecutionMode.COMPILED, ExecutionMode.EAGER:
        _set_unverified(
            _cache_by_dataset(
                caches,
                f"matrix_{mode.value}_builtin_sm_nlc",
            ),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
            generation=4.0,
            wall=2.0e-6,
            evaluator_total=1.0e-6,
        )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        Accuracy.NLC,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].mode is None
    assert view.workloads[0].terminal_label is None

    tex = render_best_mode_table(Accuracy.NLC, caches)
    lines = tex.splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    marker = r"\matrixstatus{ReportOrange}{unverified}"
    assert marker not in lines[row_index]
    assert marker not in lines[row_index + 2]
    assert r"\bestmodecodeprefix{r}" not in lines[row_index]
    assert r"\matrixstatus{ReportMuted}{N/A}" in lines[row_index]
    assert r"\matrixstatus{ReportMuted}{N/A}" in lines[row_index + 2]
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" not in lines[row_index]

    generation_summary = next(
        line for line in lines if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(line for line in lines if r"\textbf{summary: wall}" in line)
    assert marker not in generation_summary
    assert marker not in wall_summary
    assert r"\matrixsummarystats{" not in generation_summary
    assert r"\matrixsummarystats{" not in wall_summary
    assert "never eligible for best-mode selection or summaries" in tex


def test_presentation_only_unverified_is_fixed_only_and_not_a_best_mode(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "matrix_compiled_builtin_sm_nlc")
    _set_presentation_outcome(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        outcome="unverified",
    )

    fixed = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )
    best = render_best_mode_table(Accuracy.NLC, caches)
    marker = r"\matrixstatus{ReportOrange}{unverified}"
    red_marker = r"\matrixstatus{ReportRed}{unverified}"

    assert marker in fixed
    assert marker not in best
    assert red_marker not in fixed
    assert red_marker not in best


@pytest.mark.parametrize(
    ("mode", "code"),
    ((ExecutionMode.COMPILED, "c"), (ExecutionMode.EAGER, "e")),
)
def test_best_compiled_and_eager_render_absolute_without_amplicol_baseline(
    reset_caches,
    mode: ExecutionMode,
    code: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(
        caches,
        f"matrix_{mode.value}_builtin_sm_nlc",
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )

    lines = render_best_mode_table(Accuracy.NLC, caches).splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(r"\texttt{1}")
    )
    assert r"\matrixstatus{ReportMuted}{N/A}" in lines[row_index]
    assert r"\matrixstatus{ReportMuted}{N/A}" in lines[row_index + 2]
    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in lines[row_index]
    assert rf"\bestmodecode{{{code}}}" in lines[row_index]
    assert r"\bestmodeabsoluteprefix{\texttt{2.00}}" in lines[row_index + 2]
    assert r"\bestmoderatio{" not in lines[row_index]
    assert r"\bestmodeprimaryratio{" not in lines[row_index + 2]


@pytest.mark.parametrize(
    ("variant", "setup", "execution_mode"),
    (
        ("recurrence_jit_o2", "recurrence JIT O2", "recurrence"),
        ("jit_o3", "compiled JIT O3", "compiled"),
        ("eager_jit_o2", "eager-DAG JIT O2", "eager"),
    ),
)
def test_z_pyamplicol_renders_all_clocks_absolute_without_amplicol_baseline(
    reset_caches,
    variant: str,
    setup: str,
    execution_mode: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    for index, workload in enumerate((Workload.SELECTED_FLOW, Workload.ALL_FLOW)):
        baseline_entry = _entry(
            baseline,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
        )
        baseline_entry["measurement"] = _memory_censor(
            REPORT_CATALOG.cell(str(baseline_entry["cell_id"]))
        )
        _set_ok(
            candidate,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=4.0 + 6.0 * index,
            wall=(2.0 + index) * 1.0e-6,
            execution=(1.0 + 0.5 * index) * 1.0e-6,
            variant=variant,
        )
        candidate_entry = _entry(
            candidate,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            variant=variant,
        )
        measurement = candidate_entry["measurement"]
        assert isinstance(measurement, dict)
        _mark_evaluator_total(
            measurement,
            execution_mode=execution_mode,
            total=(1.7 + index) * 1.0e-6,
        )

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    reference_row = next(
        line for line in tex.splitlines() if line.startswith(r"1 & \AC{} reference")
    )
    candidate_row = next(
        line for line in tex.splitlines() if line.startswith(f"1 & {setup}")
    )
    assert reference_row.count(r"\matrixstatus{ReportOrange}{>80GB}") == 4
    assert candidate_row.count(r"\matrixncabsolute{") == 6
    assert r"\matrixratio{" not in candidate_row
    assert r"\matrixnaratio{" not in candidate_row
    assert "successful pyAmpliCol values are absolute" in tex


@pytest.mark.parametrize(
    ("variant", "setup", "execution_mode"),
    (
        ("jit_o3", "compiled JIT O3", ExecutionMode.COMPILED),
        ("eager_jit_o2", "eager-DAG JIT O2", ExecutionMode.EAGER),
    ),
)
@pytest.mark.parametrize("broken_chain", (False, True))
def test_z_dag_ratio_accepts_only_exact_recurrence_to_amplicol_chain(
    reset_caches,
    variant: str,
    setup: str,
    execution_mode: ExecutionMode,
    broken_chain: bool,
) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_lc")
    z_cache = _cache_by_dataset(caches, "z_builtin_sm")
    _set_ok(
        reference,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=10.0,
        wall=10.0e-6,
        execution=None,
    )
    _set_ok(
        z_cache,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=6.0,
        wall=3.0e-6,
        execution=2.0e-6,
        variant="recurrence_jit_o2",
    )
    _set_ok(
        z_cache,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=2.0,
        wall=1.0e-6,
        execution=0.8e-6,
        variant=variant,
    )
    candidate = _entry(
        z_cache,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        variant=variant,
    )["measurement"]
    recurrence = _entry(
        z_cache,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        variant="recurrence_jit_o2",
    )["measurement"]
    assert isinstance(candidate, dict)
    assert isinstance(recurrence, dict)
    candidate_validation = candidate["validation"]
    recurrence_validation = recurrence["validation"]
    assert isinstance(candidate_validation, dict)
    assert isinstance(recurrence_validation, dict)
    assert isinstance(candidate_validation.get("pointwise"), dict)
    if broken_chain:
        recurrence_pointwise = recurrence_validation["pointwise"]
        assert isinstance(recurrence_pointwise, dict)
        recurrence_pointwise["baseline"] = 2.0

    z_variant = next(item for item in REPORT_CATALOG.z_variants if item.key == variant)
    joined = BaselineCandidateAdapter(caches).z_workload(
        model=ModelKey.BUILTIN_SM,
        n_final=1,
        variant=z_variant,
        workload=Workload.SELECTED_FLOW,
    )
    assert z_variant.execution_mode is execution_mode
    assert joined.comparison_linked is not broken_chain
    candidate_row = next(
        line
        for line in render_z_ladder(ModelKey.BUILTIN_SM, caches).splitlines()
        if line.startswith(f"1 & {setup}")
    )
    if broken_chain:
        assert candidate_row.startswith(
            f"1 & {setup} & \\matrixncabsolute{{\\texttt{{2}}}}"
        )
        assert r"\matrixratio{" not in candidate_row
    else:
        assert r"\matrixratio{ReportGreen}{0.2}" in candidate_row


def test_best_mode_missing_candidates_remain_incomplete_under_strict_policy(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    for mode in (
        ExecutionMode.RECURRENCE,
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    ):
        entry = _entry(
            _cache_by_dataset(
                caches,
                f"matrix_{mode.value}_builtin_sm_nlc",
            ),
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
        )
        entry["measurement"] = empty_measurement()

    completeness = summarize_visible_completeness(
        caches,
        max_n_final=4,
    )

    assert not completeness.complete
    assert any(
        "result_matrix_best_builtin_sm_nlc_table.tex/n1/dd_z_jets/contracted" in slot
        for slot in completeness.applicable_na_display_slots
    )


def test_matrix_baseline_labels_follow_dataset_contract(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)

    for dataset in REPORT_CATALOG.matrix_datasets:
        tex = rendered[dataset.table_name]
        if dataset.candidate.execution_mode in {
            ExecutionMode.RECURRENCE,
            ExecutionMode.ON_THE_FLY,
        }:
            assert "Baseline: original AmpliCol" in tex
        else:
            assert "Baseline: recurrence JIT O2" in tex
        assert dataset.title in tex


def test_matrix_tables_are_fixed_nonsplittable_blocks(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)

    for tex in rendered.values():
        assert "longtable" not in tex
        assert r"\begin{minipage}{\linewidth}" in tex
        assert r"\end{minipage}" in tex
        assert tex.count(r"\begin{minipage}{\linewidth}") == tex.count(
            r"\end{minipage}"
        )
        assert tex.count(r"\clearpage") == tex.count(r"\begin{minipage}{\linewidth}")
        assert r"\begin{tabular}" in tex
        assert r"\ifdim\wd0>\linewidth" in tex
        assert r"\resizebox{\linewidth}{!}{\usebox0}" in tex
        assert tex.count(r"\matrixfitwidth{%") == tex.count(
            r"\begin{minipage}{\linewidth}"
        )
        assert r"\makebox[\linewidth][c]{%" not in tex
        assert r"\fontsize{6.5pt}{7.5pt}\selectfont" in tex
        assert r"\begingroup\matrixsummaryfont" in tex
        assert r"\hspace{0.014in}" in tex
        assert tex.index(r"\clearpage") < tex.index(r"\subsection{")
        assert tex.index(r"\subsection{") < tex.index(
            r"\noindent\begin{minipage}{\linewidth}"
        )


def test_inapplicable_not_run_and_reset_cells_use_distinct_markers(
    reset_caches,
) -> None:
    lc = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc"),
        reset_caches,
    )
    assert r"\matrixnotapplicable{ReportMuted}" in lc
    assert r"\matrixstatus{ReportMuted}{not run}" not in lc
    assert r"\matrixstatus{ReportMuted}{N/A}" in lc

    for accuracy in (Accuracy.NLC, Accuracy.FULL):
        fixed = render_matrix_table(
            REPORT_CATALOG.dataset(f"matrix_recurrence_builtin_sm_{accuracy.value}"),
            reset_caches,
        )
        best = render_best_mode_table(accuracy, reset_caches)
        for tex in (fixed, best):
            assert tex.count(r"\matrixstatus{ReportMuted}{not run}") == 12
            assert tex.count(r"\matrixnotapplicable{ReportMuted}") == 28
        assert "Not run marks an otherwise defined process" in fixed

        n6_family_one = next(
            line
            for line in fixed.splitlines()
            if line.startswith(r"\texttt{1}")
            and r"\matrixstatus{ReportMuted}{not run}" in line
        )
        assert r"\matrixnotapplicable{ReportMuted}" not in n6_family_one
        family_fourteen = next(
            line
            for line in fixed.splitlines()
            if line.startswith(r"\texttt{14}")
            and line.count(r"\matrixnotapplicable{ReportMuted}") == 2
        )
        assert r"\matrixstatus{ReportMuted}{not run}" not in family_fourteen

    assert "Fixed-engine tables intentionally omit mode letters" in lc


def test_z_reference_execution_is_explicitly_not_exposed(reset_caches) -> None:
    tex = render_z_ladder(ModelKey.BUILTIN_SM, reset_caches)
    reference_rows = [line for line in tex.splitlines() if r"\AC{} reference" in line]

    assert len(reference_rows) == 9
    for row in reference_rows:
        assert row.count(r"\matrixnotexposed{ReportMuted}") == 2
        assert r"\matrixna{ReportMuted}" not in row
    assert "not a missing measurement" in tex


@pytest.mark.parametrize("model", (ModelKey.BUILTIN_SM, ModelKey.UFO_SM))
def test_z_native_generation_cap_renders_static_na_in_both_models(
    reset_caches,
    model: ModelKey,
) -> None:
    tex = render_z_ladder(model, reset_caches)
    capped_rows = [
        line
        for line in tex.splitlines()
        if line.startswith(("7 & ", "8 & ", "9 & "))
        and ("compiled ASM O3" in line or "compiled C++ O3" in line)
    ]

    assert len(capped_rows) == 6
    assert all(row.count(r"\matrixstaticna{ReportMuted}") == 6 for row in capped_rows)
    assert "user cap: native C++/ASM generation is not attempted above n=6" in tex
    assert "native-backend-generation-cap-n6-v1" in tex


def test_z_evaluator_total_is_mode_independent_and_not_execution_attribution(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    for workload in (Workload.SELECTED_FLOW, Workload.ALL_FLOW):
        _set_ok(
            baseline,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=1.0,
            wall=1.0e-6,
            execution=9.0e-6,
        )
    modes = (
        ("jit_o1", "compiled", "compiled JIT O1", 31.0, 41.0),
        ("eager_jit_o2", "eager", "eager-DAG JIT O2", 32.0, 42.0),
        (
            "recurrence_jit_o2",
            "recurrence",
            "recurrence JIT O2",
            33.0,
            43.0,
        ),
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    for variant, execution_mode, _label, selected_total, all_total in modes:
        for workload, total in (
            (Workload.SELECTED_FLOW, selected_total),
            (Workload.ALL_FLOW, all_total),
        ):
            _set_ok(
                candidate,
                process_key="dd_z_jets",
                n_final=1,
                workload=workload,
                generation=2.0,
                wall=(10.0 + total) * 1.0e-6,
                execution=(70.0 + total) * 1.0e-6,
                variant=variant,
            )
            measurement = next(
                entry["measurement"]
                for entry in entries
                if entry["process_key"] == "dd_z_jets"
                and entry["n_final"] == 1
                and entry["workload"] == workload.value
                and entry["variant"] == variant
            )
            assert isinstance(measurement, dict)
            _mark_evaluator_total(
                measurement,
                execution_mode=execution_mode,
                total=total * 1.0e-6,
            )
            if execution_mode == "recurrence":
                _mark_recurrence_core(measurement)

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    for _variant, _mode, label, selected_total, all_total in modes:
        row = next(
            line
            for line in tex.splitlines()
            if line.startswith("1 & ") and label in line
        )
        assert (
            rf"\texttt{{{selected_total:.0f}}}\,"
            rf"\matrixratio{{ReportRed}}{{{selected_total:.0f}}}"
        ) in row
        assert (
            rf"\texttt{{{all_total:.0f}}}\,"
            rf"\matrixratio{{ReportRed}}{{{all_total:.0f}}}"
        ) in row
        assert r"\matrixzrecurrenceclocks" not in row
        assert r"\matrixtotalevaluator" not in row
        assert r"\matrixrecurrencecore" not in row
    assert tex.count(r"\textbf{eval [\(\mu\mathrm{s}/\mathrm{pt}\)]}") == 6
    assert r"\textbf{eval total T}" not in tex
    assert r"\textbf{rec. core C}" not in tex
    assert "original-AmpliCol wall measurement as denominator" in tex


def test_z_unverified_retains_absolute_clocks_without_ratios(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    _set_unverified(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=4.0,
        wall=2.0e-6,
        evaluator_total=1.0e-6,
        variant="jit_o3",
    )

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & compiled JIT O3")
    )
    marker = r"\matrixstatus{ReportOrange}{unverified}"

    for value in ("4.00", "2.00", "1.00"):
        assert (
            rf"\matrixruntimepair{{\texttt{{{value}}}}}"
            rf"{{{marker}}}"
        ) in row
    assert row.count(marker) == 3
    assert r"\matrixratio{" not in row
    assert "Unverified diagnostics retain absolute clocks" in tex


def test_z_wall_and_evaluator_total_keep_distinct_six_digit_values(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=1.0,
        wall=100.0e-6,
        execution=90.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=2.0,
        wall=218.105e-6,
        execution=183.456e-6,
        variant="recurrence_jit_o2",
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.SELECTED_FLOW.value
        and entry["variant"] == "recurrence_jit_o2"
    )
    assert isinstance(measurement, dict)
    _mark_evaluator_total(
        measurement,
        execution_mode="recurrence",
        total=217.812e-6,
    )
    _mark_recurrence_core(measurement)

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & ") and "recurrence JIT O2" in line
    )

    assert r"\texttt{218.105}" in row
    assert r"\texttt{217.812}\,\matrixratio{ReportRed}{2.18}" in row
    assert r"\texttt{183.456}" not in row
    assert r"\matrixrecurrencecore" not in row


def test_z_historical_recurrence_without_dedicated_total_is_not_exposed(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=2.0,
        wall=9.0e-6,
        execution=7.0e-6,
        variant="recurrence_jit_o2",
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.SELECTED_FLOW.value
        and entry["variant"] == "recurrence_jit_o2"
    )
    assert isinstance(measurement, dict)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance["runtime_profile"] = {
        "achieved_runtime_seconds": 0.003,
        "completed_sample_count": 5,
        "interrupted": False,
        "measured_point_count": 1000,
        "planned_sample_count": 5,
        "repetitions_per_sample": 1,
        "target_runtime_achieved": True,
        "target_runtime_seconds": 0.002,
    }
    provenance["execution_timing"] = {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "measured",
        "ratio_eligible": True,
        "raw_seconds_per_point": 7.0e-6,
        "source": "runtime_profile_core_recurrence_schedule_time",
        "compiled_direct_arena_active": False,
        "sample_count": 5,
        "native_profile_points_per_sample": 200,
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
    }

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & ") and "recurrence JIT O2" in line
    )

    assert r"\matrixnotexposed{ReportMuted}" in row
    assert r"\texttt{7}" not in row
    assert r"\matrixzrecurrenceclocks" not in row

    provenance["execution_timing"]["source"] = (
        "runtime_profile_core_evaluator_call_time"
    )
    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & ") and "recurrence JIT O2" in line
    )
    assert r"\matrixnotexposed{ReportMuted}" in row
    assert r"\matrixzrecurrenceclocks" not in row


def test_visible_completeness_accounts_for_every_n4_slot(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)

    summary = summarize_visible_completeness(caches, max_n_final=4)
    evidence = summary.as_dict()

    assert summary.complete
    assert evidence["required_measurement_count"] == 828
    assert evidence["rendered_required_measurement_count"] == 828
    assert evidence["structurally_not_applicable_display_slot_count"] == 405
    assert evidence["not_exposed_display_slot_count"] == 16
    assert evidence["applicable_na_display_slot_count"] == 0
    assert evidence["missing_rendered_cell_count"] == 0


def test_visible_completeness_authenticates_catalog_static_na_slots(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_scope(caches, max_n_final=9)

    summary = summarize_visible_completeness(caches, max_n_final=9)
    evidence = summary.as_dict()

    assert summary.complete
    assert evidence["declared_measurement_cell_count"] == 1962
    assert evidence["required_measurement_count"] == 1828
    assert evidence["catalog_static_na_cell_count"] == 134
    assert evidence["rendered_catalog_static_na_cell_count"] == 134
    assert evidence["applicable_na_display_slot_count"] == 0
    assert evidence["missing_rendered_cell_count"] == 0


def test_visible_completeness_rejects_na_in_applicable_slot(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    cache = _cache_by_dataset(caches, "z_builtin_sm")
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["n_final"] == 1
        and item["variant"] == "jit_o3"
        and item["workload"] == Workload.SELECTED_FLOW.value
    )
    measurement = entry["measurement"]
    assert isinstance(measurement, dict)
    measurement["generation_seconds"] = None

    summary = summarize_visible_completeness(caches, max_n_final=4)

    assert not summary.complete
    assert any(entry["cell_id"] in slot for slot in summary.applicable_na_display_slots)


def test_na_detector_recognizes_ratio_macro() -> None:
    assert report_render._renders_na(r"\matrixnaratio{ReportMuted}")


def test_visible_completeness_rejects_missing_candidate_generation_ratio(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    cache = _cache_by_dataset(caches, "matrix_compiled_builtin_sm_full")
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["process_key"] == "dd_z_jets"
        and item["n_final"] == 1
        and item["workload"] == Workload.CONTRACTED.value
    )
    measurement = entry["measurement"]
    assert isinstance(measurement, dict)
    measurement["generation_seconds"] = None

    summary = summarize_visible_completeness(caches, max_n_final=4)

    assert not summary.complete
    assert any(
        entry["cell_id"] in slot and "/candidate" in slot
        for slot in summary.applicable_na_display_slots
    )


def test_visible_completeness_rejects_non_ok_required_status(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)
    cache = _cache_by_dataset(caches, "z_builtin_sm")
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["n_final"] == 1
        and item["variant"] == "jit_o3"
        and item["workload"] == Workload.SELECTED_FLOW.value
    )
    measurement = entry["measurement"]
    assert isinstance(measurement, dict)
    measurement["status"] = ResultStatus.TIMEOUT.value

    summary = summarize_visible_completeness(caches, max_n_final=4)

    assert not summary.complete
    assert not any(
        entry["cell_id"] in slot for slot in summary.applicable_na_display_slots
    )
    assert any(
        entry["cell_id"] in error
        and repr(ResultStatus.TIMEOUT.value) in error
        and repr(ResultStatus.OK.value) in error
        for error in summary.contract_errors
    )


def test_adapter_joins_recurrence_baseline_without_copying_timing(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    candidate = _cache_by_dataset(
        caches,
        "matrix_compiled_builtin_sm_nlc",
    )
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.5e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=1.0e-6,
        execution=0.5e-6,
    )

    dataset = REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc")
    family = REPORT_CATALOG.process_families[0]
    joined = BaselineCandidateAdapter(caches).matrix_cell(dataset, family, 1)
    tex = render_matrix_table(dataset, caches)

    assert joined.workloads[0].baseline is not joined.workloads[0].candidate
    assert joined.workloads[0].baseline["generation_seconds"] == 4.0
    assert r"\texttt{4.00}" in tex
    assert r"\bestmoderatio{ReportGreen}{0.500}" in tex


def test_recurrence_matrix_cell_uses_compact_wall_and_core_ratios(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_nlc")
    candidate = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=100.0e-6,
        execution=90.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=218.105e-6,
        execution=183.456e-6,
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    assert isinstance(measurement, dict)
    _mark_evaluator_total(
        measurement,
        execution_mode="recurrence",
        total=217.812e-6,
    )
    _mark_recurrence_core(measurement)

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_nlc"),
        caches,
    )

    assert r"\bestmodecompactprefix{2.04}" in tex
    assert r"\bestmodeprimaryratio{ReportRed}{2.18}" in tex
    assert r"\texttt{217.812}" not in tex
    assert r"\texttt{183.456}" not in tex
    assert r"\matrixruntimetriple{" not in tex
    assert "An evaluator total is never divided by a recurrence core" in tex


def test_unavailable_execution_keeps_only_compact_wall_ratio(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    candidate = _cache_by_dataset(
        caches,
        "matrix_compiled_builtin_sm_nlc",
    )
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=1.0e-6,
        execution=None,
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    _mark_arena_unavailable(measurement)

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )

    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )
    assert r"\bestmodeopenprefix" not in runtime_row
    assert r" &  & \bestmodewallratio{ReportGreen}{0.500}" in runtime_row
    assert r"\matrixruntimepair{" not in tex
    assert r"\matrixtotalevaluator{" not in tex
    assert r"{x0}" not in tex


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_dag_matrix_never_divides_evaluator_total_by_recurrence_core(
    reset_caches,
    execution_mode: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    dataset_id = (
        "matrix_compiled_builtin_sm_nlc"
        if execution_mode == "compiled"
        else "matrix_eager_builtin_sm_nlc"
    )
    candidate = _cache_by_dataset(caches, dataset_id)
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=1.0e-6,
        execution=None,
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    _mark_arena_unavailable(measurement, execution_mode=execution_mode)
    _mark_evaluator_total(
        measurement,
        execution_mode=execution_mode,
        total=0.9e-6,
    )

    tex = render_matrix_table(REPORT_CATALOG.dataset(dataset_id), caches)

    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )
    assert r"\bestmodeopenprefix" not in runtime_row
    assert r" &  & \bestmodewallratio{ReportGreen}{0.500}" in runtime_row
    assert r"\bestmodecompactprefix{0.900}" not in runtime_row
    assert r"\texttt{0.900}" not in tex
    assert r"\matrixtotalevaluator{" not in tex
    assert r"\matrixruntimetriple{" not in tex


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_dag_matrix_prefers_authenticated_evaluator_total_pair(
    reset_caches,
    execution_mode: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    dataset_id = (
        "matrix_compiled_builtin_sm_nlc"
        if execution_mode == "compiled"
        else "matrix_eager_builtin_sm_nlc"
    )
    candidate = _cache_by_dataset(caches, dataset_id)
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=1.0e-6,
        execution=None,
    )
    baseline_measurement = next(
        entry["measurement"]
        for entry in baseline["entries"]
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    candidate_measurement = next(
        entry["measurement"]
        for entry in candidate["entries"]
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    assert isinstance(baseline_measurement, dict)
    assert isinstance(candidate_measurement, dict)
    _mark_evaluator_total(
        baseline_measurement,
        execution_mode="recurrence",
        total=1.8e-6,
    )
    _mark_arena_unavailable(
        candidate_measurement,
        execution_mode=execution_mode,
    )
    _mark_evaluator_total(
        candidate_measurement,
        execution_mode=execution_mode,
        total=0.9e-6,
    )

    tex = render_matrix_table(REPORT_CATALOG.dataset(dataset_id), caches)
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )

    assert r"\bestmodecompactprefix{0.500}" in runtime_row
    assert r"\bestmodeprimaryratio{ReportGreen}{0.500}" in runtime_row
    assert r"\bestmodecompactprefix{0.900}" not in runtime_row
    assert "evaluator-total fallback is diagnostic only" in tex


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
@pytest.mark.parametrize(
    ("legacy_execution", "expected_secondary"),
    ((1.5e-6, "0.600"), (None, "0.450")),
)
def test_best_mode_uses_candidate_total_against_legacy_direct_or_wall(
    reset_caches,
    execution_mode: str,
    legacy_execution: float | None,
    expected_secondary: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    reference = _cache_by_dataset(caches, "reference_amplicol_nlc")
    recurrence = _cache_by_dataset(caches, "matrix_recurrence_builtin_sm_nlc")
    candidate = _cache_by_dataset(
        caches,
        f"matrix_{execution_mode}_builtin_sm_nlc",
    )
    _set_ok(
        reference,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        execution=legacy_execution,
    )
    _set_ok(
        recurrence,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=3.0,
        wall=1.5e-6,
        execution=1.2e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.CONTRACTED,
        generation=2.0,
        wall=1.0e-6,
        execution=None,
    )
    candidate_measurement = next(
        entry["measurement"]
        for entry in candidate["entries"]
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    assert isinstance(candidate_measurement, dict)
    _mark_arena_unavailable(
        candidate_measurement,
        execution_mode=execution_mode,
    )
    _mark_evaluator_total(
        candidate_measurement,
        execution_mode=execution_mode,
        total=0.9e-6,
    )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
    )

    assert rf"\bestmodecompactprefix{{{expected_secondary}}}" in runtime_row
    assert r"\bestmodeprimaryratio{ReportGreen}{0.500}" in runtime_row
    assert "evaluator-total fallback is diagnostic only" in tex
    assert "uses the AmpliCol direct execution clock" in tex


@pytest.mark.parametrize(
    "surface",
    ("z_builtin_sm", "scalar_contact", "scalar_gravity"),
)
def test_ladders_render_generic_presentation_outcomes_through_shared_status_path(
    reset_caches,
    surface: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    cache = _cache_by_dataset(caches, surface)
    if surface == "z_builtin_sm":
        _set_presentation_outcome(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.SELECTED_FLOW,
            outcome="dependency_backend_error",
            variant="jit_o3",
        )
        tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    else:
        _set_presentation_outcome(
            cache,
            process_key=surface,
            n_final=2,
            workload=Workload.CONTRACTED,
            outcome="dependency_backend_error",
        )
        dataset = next(
            item
            for item in REPORT_CATALOG.scalar_datasets
            if item.dataset_id == surface
        )
        tex = render_scalar_ladder(dataset, caches)

    display = report_render._compact_terminal_display(
        report_render._TerminalOutcome(
            identity="presentation:dependency_backend_error",
            label="dependency backend error",
            color="ReportRed",
        )
    )
    marker = rf"\matrixstatus{{ReportRed}}{{{display}}}"
    assert tex.count(marker) >= 3
    assert "dependency backend error" not in tex
    assert r"\matrixstatus{ReportOrange}{dependency backend error}" not in tex
    assert "dependency_backend_error" not in tex


def test_z_ladder_prints_not_exposed_instead_of_zero(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    _set_ok(
        baseline,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=4.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )
    _set_ok(
        candidate,
        process_key="dd_z_jets",
        n_final=1,
        workload=Workload.SELECTED_FLOW,
        generation=2.0,
        wall=1.0e-6,
        execution=None,
        variant="jit_o3",
    )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["variant"] == "jit_o3"
        and entry["workload"] == Workload.SELECTED_FLOW.value
    )
    _mark_arena_unavailable(measurement)

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)

    assert r"\matrixnotexposed{ReportMuted}" in tex
    assert r"{x0}" not in tex


def test_unavailable_execution_is_not_treated_as_a_summary_ratio(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_nlc",
    )
    candidate = _cache_by_dataset(
        caches,
        "matrix_compiled_builtin_sm_nlc",
    )
    for cache, execution in ((baseline, 4.0e-7), (candidate, None)):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.CONTRACTED,
            generation=1.0,
            wall=1.0e-6,
            execution=execution,
        )
    entries = candidate["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["process_key"] == "dd_z_jets"
        and entry["n_final"] == 1
        and entry["workload"] == Workload.CONTRACTED.value
    )
    _mark_arena_unavailable(measurement)
    dataset = REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc")
    view = BaselineCandidateAdapter(caches).matrix_cell(
        dataset,
        REPORT_CATALOG.process_families[0],
        1,
    )

    summary = report_render._summary_pair(
        (view,),
        Workload.CONTRACTED,
        "execution_seconds_per_point",
    )

    assert summary == r"\matrixnotexposed{ReportMuted}"


def test_adapter_joins_original_amplicol_for_primary_recurrence(
    reset_caches,
) -> None:
    dataset = REPORT_CATALOG.dataset("matrix_recurrence_ufo_sm_full")
    family = REPORT_CATALOG.process_families[0]
    adapter = BaselineCandidateAdapter(reset_caches)
    joined = adapter.matrix_cell(dataset, family, 1)
    reference = adapter.index.get(
        "reference_amplicol_full",
        family.key,
        1,
        Workload.CONTRACTED,
    )

    assert joined.workloads[0].baseline is reference


def test_lc_cells_join_two_layout_specific_baselines(reset_caches) -> None:
    dataset = REPORT_CATALOG.dataset("matrix_eager_builtin_sm_lc")
    family = REPORT_CATALOG.process_families[0]
    joined = BaselineCandidateAdapter(reset_caches).matrix_cell(
        dataset,
        family,
        1,
    )

    assert tuple(item.workload for item in joined.workloads) == (
        Workload.SELECTED_FLOW,
        Workload.ALL_FLOW,
    )


def test_amplicol_all_flow_setup_generation_ratio_is_not_comparable(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_lc",
    )
    for workload, baseline_generation, candidate_generation in (
        (Workload.SELECTED_FLOW, 2.0, 4.0),
        (Workload.ALL_FLOW, 1.0e-4, 10.0),
    ):
        _set_ok(
            baseline,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=baseline_generation,
            wall=2.0e-6,
            execution=1.0e-6,
        )
        _set_ok(
            candidate,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=candidate_generation,
            wall=1.0e-6,
            execution=0.5e-6,
        )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc"),
        caches,
    )

    assert tex.count(r"\bestmodeabsoluteprefix{\texttt{10.0}}") == 1
    generation_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: generation}" in line
    )
    assert r"\texttt{10}" not in generation_summary
    assert r"\matrixratio{ReportRed}{1e+05}" not in tex
    assert "setup boundary differs from the reference" in tex
    assert "n.c." not in tex


def test_four_line_recurrence_renders_absolute_values_without_legacy_oracle(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_lc",
    )
    for workload, generation in (
        (Workload.SELECTED_FLOW, 4.0),
        (Workload.ALL_FLOW, 10.0),
    ):
        _set_ok(
            candidate,
            process_key="dd_4q_lines",
            n_final=6,
            workload=workload,
            generation=generation,
            wall=2.0e-6,
            execution=1.0e-6,
        )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc"),
        caches,
    )

    assert r"\bestmodeabsoluteprefix{\texttt{4.00}}" in tex
    assert r"\bestmodeabsoluteprefix{\texttt{10.0}}" in tex
    assert tex.count(r"\texttt{2.00}") >= 2
    assert r"\matrixruntimetriple{" not in tex
    assert tex.count(r"\matrixstaticna{ReportMuted}") >= 2
    assert "static N/A beyond three open quark lines" in tex
    assert "n.c." not in tex


def test_four_line_contracted_n6_renders_without_legacy_dependency(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    candidate = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_full",
    )
    _set_ok(
        candidate,
        process_key="dd_4q_lines",
        n_final=6,
        workload=Workload.CONTRACTED,
        generation=12.0,
        wall=2.0e-6,
        execution=1.0e-6,
    )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_full"),
        caches,
    )

    assert r"\textbf{n=6}" in tex
    fixed_generation_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r"\texttt{14}")
        and r"\bestmodeabsoluteprefix{\texttt{12.0}}" in line
    )
    fixed_runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
        and r"\bestmodeabsoluteprefix{\texttt{2.00}}" in line
    )
    assert (
        r"\matrixstaticna{ReportMuted} & "
        r"\bestmodeabsoluteprefix{\texttt{12.0}} & "
    ) in fixed_generation_row
    assert (
        r"\matrixstaticna{ReportMuted} & "
        r"\bestmodeabsoluteprefix{\texttt{2.00}} & "
    ) in fixed_runtime_row
    assert r"\matrixstaticna{ReportMuted}" in tex
    assert "static N/A beyond three open quark lines" in tex
    assert "n.c." not in tex

    best_tex = render_best_mode_table(Accuracy.FULL, caches)
    best_generation_row = next(
        line
        for line in best_tex.splitlines()
        if line.startswith(r"\texttt{14}")
        and r"\bestmodeabsoluteprefix{\texttt{12.0}}" in line
    )
    best_runtime_row = next(
        line
        for line in best_tex.splitlines()
        if line.startswith(r" &  & \textcolor{ReportMuted}{\scriptsize run")
        and r"\bestmodeabsoluteprefix{\texttt{2.00}}" in line
    )
    assert (
        r"\matrixstaticna{ReportMuted} & "
        r"\bestmodeabsoluteprefix{\texttt{12.0}} & \bestmodecode{r}"
    ) in best_generation_row
    assert (
        r"\matrixstaticna{ReportMuted} & "
        r"\bestmodeabsoluteprefix{\texttt{2.00}} & "
    ) in best_runtime_row
    assert r"\bestmodecode" not in best_runtime_row


def test_z_all_flow_setup_generation_ratio_is_not_comparable(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(caches, "reference_amplicol_lc")
    candidate = _cache_by_dataset(caches, "z_builtin_sm")
    for workload, baseline_generation, candidate_generation in (
        (Workload.SELECTED_FLOW, 2.0, 4.0),
        (Workload.ALL_FLOW, 1.0e-4, 10.0),
    ):
        _set_ok(
            baseline,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=baseline_generation,
            wall=2.0e-6,
            execution=1.0e-6,
        )
        _set_ok(
            candidate,
            process_key="dd_z_jets",
            n_final=1,
            workload=workload,
            generation=candidate_generation,
            wall=1.0e-6,
            execution=0.5e-6,
            variant="jit_o3",
        )

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)

    assert r"\matrixncabsolute{\texttt{10}}" in tex
    assert r"\matrixratio{ReportRed}{1e+05}" not in tex
    assert "setup boundary differs from the reference" in tex
    assert "n.c." not in tex


def test_cross_mode_all_flow_generation_ratio_remains_layout_matched(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    baseline = _cache_by_dataset(
        caches,
        "matrix_recurrence_builtin_sm_lc",
    )
    candidate = _cache_by_dataset(caches, "matrix_compiled_builtin_sm_lc")
    for cache, generation in ((baseline, 4.0), (candidate, 2.0)):
        _set_ok(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.ALL_FLOW,
            generation=generation,
            wall=2.0e-6,
            execution=1.0e-6,
        )

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_lc"),
        caches,
    )

    assert r"\bestmoderatio{ReportGreen}{0.500}" in tex
    assert r"\matrixncabsolute{\texttt" not in tex


def test_two_z_ladders_use_single_line_mode_labels(reset_caches) -> None:
    rendered = render_all_z_ladders(reset_caches)

    assert list(rendered) == [
        "result_z_builtin_sm_table.tex",
        "result_z_external_sm_table.tex",
    ]
    for tex in rendered.values():
        assert "longtable" not in tex
        assert "eager-DAG JIT O2" in tex
        assert "recurrence JIT O2" in tex
        assert "eager-DAG JIT O2\\\\" not in tex
        assert "recurrence JIT O2\\\\" not in tex
        assert r"\begin{minipage}{\linewidth}" in tex
        assert tex.index(r"\clearpage") < tex.index(r"\subsection{")
        assert tex.index(r"\subsection{") < tex.index(
            r"\noindent\begin{minipage}{\linewidth}"
        )


def test_dense_scalar_ladder_fits_narrower_columns(reset_caches) -> None:
    dataset = next(
        item
        for item in REPORT_CATALOG.scalar_datasets
        if item.dataset_id == "scalar_contact"
    )

    tex = render_scalar_ladder(dataset, reset_caches)

    assert r"@{}L{1.00in}" in tex
    assert tex.count(r"L{0.72in}") == len(dataset.multiplicities)
    assert r"\setlength{\tabcolsep}{2.2pt}" in tex
    assert (
        r"{\fontsize{6.4pt}{7.4pt}\selectfont"
        r"\mbox{evaluator total [\(\mu\mathrm{s}/\mathrm{pt}\)]}} & "
        in tex
    )
    assert (
        "\nevaluator total " r"[\(\mu\mathrm{s}/\mathrm{pt}\)] & "
        not in tex
    )


@pytest.mark.parametrize("dataset_id", ("scalar_contact", "scalar_gravity"))
def test_scalar_timing_marks_unavailable_arena_attribution_not_exposed(
    reset_caches,
    dataset_id: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    dataset = next(
        item for item in REPORT_CATALOG.scalar_datasets if item.dataset_id == dataset_id
    )
    cache = _cache_by_dataset(caches, dataset_id)
    _set_ok(
        cache,
        process_key=dataset_id,
        n_final=2,
        workload=Workload.CONTRACTED,
        generation=1.0,
        wall=1.0e-6,
        execution=None,
    )
    entries = cache["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"] for entry in entries if entry["n_final"] == 2
    )
    _mark_arena_unavailable(measurement)

    tex = render_scalar_ladder(dataset, caches)

    assert (
        r"{\fontsize{6.4pt}{7.4pt}\selectfont"
        r"\mbox{evaluator total [\(\mu\mathrm{s}/\mathrm{pt}\)]}}"
        in tex
    )
    assert r"\matrixnotexposed{ReportMuted}" in tex
    assert "successful wall measurement" in tex


def test_scalar_timing_uses_dedicated_evaluator_total_not_wall(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    dataset = next(
        item
        for item in REPORT_CATALOG.scalar_datasets
        if item.dataset_id == "scalar_contact"
    )
    cache = _cache_by_dataset(caches, "scalar_contact")
    _set_ok(
        cache,
        process_key="scalar_contact",
        n_final=2,
        workload=Workload.CONTRACTED,
        generation=1.0,
        wall=218.105e-6,
        execution=None,
    )
    entries = cache["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"] for entry in entries if entry["n_final"] == 2
    )
    assert isinstance(measurement, dict)
    _mark_arena_unavailable(measurement)
    _mark_evaluator_total(
        measurement,
        execution_mode="compiled",
        total=217.812e-6,
    )

    tex = render_scalar_ladder(dataset, caches)
    wall_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r"wall [\(\mu\mathrm{s}/\mathrm{pt}\)]")
    )
    total_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(
            r"{\fontsize{6.4pt}{7.4pt}\selectfont"
            r"\mbox{evaluator total [\(\mu\mathrm{s}/\mathrm{pt}\)]}}"
        )
    )

    assert r"\texttt{218.105}" in wall_row
    assert r"\texttt{217.812}" in total_row
    assert r"\texttt{218.105}" not in total_row
    assert r"execution [\(\mu\mathrm{s}/\mathrm{pt}\)]" not in tex
    assert "never copied from or derived from wall time" in tex


@pytest.mark.parametrize("dataset_id", ("scalar_contact", "scalar_gravity"))
def test_scalar_unverified_retains_absolute_clocks_with_marker(
    reset_caches,
    dataset_id: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    dataset = next(
        item for item in REPORT_CATALOG.scalar_datasets if item.dataset_id == dataset_id
    )
    cache = _cache_by_dataset(caches, dataset_id)
    _set_unverified(
        cache,
        process_key=dataset_id,
        n_final=2,
        workload=Workload.CONTRACTED,
        generation=4.0,
        wall=2.0e-6,
        evaluator_total=1.0e-6,
    )

    tex = render_scalar_ladder(dataset, caches)
    marker = r"\matrixstatus{ReportOrange}{unverified}"
    expected = {
        "generation [s]": "4.00",
        r"wall [\(\mu\mathrm{s}/\mathrm{pt}\)]": "2.00",
        (
            r"{\fontsize{6.4pt}{7.4pt}\selectfont"
            r"\mbox{evaluator total [\(\mu\mathrm{s}/\mathrm{pt}\)]}}"
        ): "1.00",
    }
    for prefix, value in expected.items():
        row = next(line for line in tex.splitlines() if line.startswith(prefix))
        assert (
            rf"\matrixruntimepair{{\texttt{{{value}}}}}"
            rf"{{{marker}}}"
        ) in row
    assert r"\matrixratio{" not in tex
    assert "Unverified diagnostics retain absolute generation" in tex


@pytest.mark.parametrize("surface", ("z_builtin_sm", "scalar_contact"))
def test_ladder_presentation_only_unverified_is_orange(
    reset_caches,
    surface: str,
) -> None:
    caches = copy.deepcopy(reset_caches)
    cache = _cache_by_dataset(caches, surface)
    if surface == "z_builtin_sm":
        _set_presentation_outcome(
            cache,
            process_key="dd_z_jets",
            n_final=1,
            workload=Workload.SELECTED_FLOW,
            outcome="unverified",
            variant="jit_o3",
        )
        tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    else:
        _set_presentation_outcome(
            cache,
            process_key=surface,
            n_final=2,
            workload=Workload.CONTRACTED,
            outcome="unverified",
        )
        dataset = next(
            item
            for item in REPORT_CATALOG.scalar_datasets
            if item.dataset_id == surface
        )
        tex = render_scalar_ladder(dataset, caches)

    assert r"\matrixstatus{ReportOrange}{unverified}" in tex
    assert r"\matrixstatus{ReportRed}{unverified}" not in tex


def test_all_outputs_include_matrices_z_and_scalar_ladders(
    reset_caches,
) -> None:
    rendered = render_all_tables(reset_caches)

    assert len(rendered) == 23
    assert "result_validation_summary.tex" in rendered
    assert set(render_all_best_mode_tables(reset_caches)) < set(rendered)
    assert set(render_all_matrix_tables(reset_caches)) < set(rendered)
    assert set(render_all_z_ladders(reset_caches)) < set(rendered)
    assert "result_scalar_contact_table.tex" in rendered
    assert "result_scalar_gravity_table.tex" in rendered


def test_validation_summary_counts_complete_scope_and_comparison_kinds(
    reset_caches,
) -> None:
    caches = copy.deepcopy(reset_caches)
    revision = "0123456789abcdef0123456789abcdef01234567"
    records = (
        (
            "reference_amplicol_lc",
            "dd_z_jets",
            1,
            Workload.SELECTED_FLOW,
            {
                "status": "ok",
                "method": "independent-original-amplicol-oracle",
            },
        ),
        (
            "matrix_recurrence_builtin_sm_lc",
            "dd_z_jets",
            1,
            Workload.SELECTED_FLOW,
            {
                "status": "ok",
                "pointwise": {
                    "status": "ok",
                    "relative_difference": 1.0e-9,
                },
                "resolved_sum": {
                    "status": "ok",
                    "maximum_relative_difference": 2.0e-13,
                },
            },
        ),
        (
            "matrix_compiled_builtin_sm_lc",
            "dd_z_jets",
            1,
            Workload.SELECTED_FLOW,
            {
                "status": "ok",
                "pointwise": {
                    "status": "ok",
                    "relative_difference": 3.0e-13,
                },
                "resolved_sum": {
                    "status": "ok",
                    "maximum_relative_difference": 4.0e-14,
                },
            },
        ),
        (
            "scalar_contact",
            "scalar_contact",
            2,
            Workload.CONTRACTED,
            {
                "status": "ok",
                "high_precision": {
                    "status": "ok",
                    "relative_difference": 5.0e-14,
                },
                "resolved_sum": {
                    "status": "ok",
                    "maximum_relative_difference": 6.0e-14,
                },
            },
        ),
    )
    for dataset_id, process_key, n_final, workload, validation in records:
        cache = _cache_by_dataset(caches, dataset_id)
        _set_ok(
            cache,
            process_key=process_key,
            n_final=n_final,
            workload=workload,
            generation=1.0,
            wall=1.0e-6,
            execution=5.0e-7,
        )
        entries = cache["entries"]
        assert isinstance(entries, list)
        measurement = next(
            entry["measurement"]
            for entry in entries
            if entry["process_key"] == process_key
            and entry["n_final"] == n_final
            and entry["workload"] == workload.value
        )
        measurement["validation"] = validation
        measurement["provenance"] = {
            "report_measured_source_revision": revision,
        }

    summary = summarize_validation(caches)
    tex = render_validation_summary(caches)

    assert r"\begin{tabular}{@{}l r r r l@{}}" in tex
    assert summary.expected_by_n == (
        (1, 68),
        (2, 202),
        (3, 224),
        (4, 334),
        (5, 305),
        (6, 201),
        (7, 165),
        (8, 165),
        (9, 164),
    )
    assert summary.declared_total == 1962
    assert summary.static_na_by_n == (
        (1, 4),
        (2, 16),
        (3, 18),
        (4, 28),
        (5, 28),
        (6, 10),
        (7, 10),
        (8, 10),
        (9, 10),
    )
    assert summary.static_na_total == 134
    assert summary.expected_total == 1828
    assert summary.passed_total == 4
    assert summary.status_counts == (
        ("not_available", 1824),
        ("ok", 4),
        ("static-na", 134),
    )
    assert summary.oracle_count == 1
    assert summary.independent_count == 1
    assert summary.independent_maximum_relative_difference == 1.0e-9
    assert summary.cross_mode_count == 1
    assert summary.cross_mode_maximum_relative_difference == 3.0e-13
    assert summary.resolved_count == 3
    assert summary.resolved_maximum_relative_difference == 2.0e-13
    assert summary.high_precision_count == 1
    assert summary.high_precision_maximum_relative_difference == 5.0e-14
    assert summary.uniform_source_revision == revision
    assert "1962 & 134 & 4" in tex
    assert "1962 declared cells" in tex
    assert "1828 measurable cells" in tex
    assert "134 catalog-authenticated static N/A" in tex
    assert "539 matrix process/multiplicity positions" in tex
    assert "36 reference execution fields" in tex
    assert rf"\nolinkurl{{{revision}}}" in tex
