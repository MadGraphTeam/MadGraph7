# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy

import pytest

import tools.performance_report.render as report_render
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
def reset_caches() -> dict[str, dict[str, object]]:
    return build_reset_caches()


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
    entry["measurement"] = {
        "status": ResultStatus.OK.value,
        "generation_seconds": generation,
        "wall_seconds_per_point": wall,
        "execution_seconds_per_point": execution,
        "matrix_element": 1.0,
        "sample_count": 5,
        "standard_error_seconds_per_point": 1.0e-9,
        "relative_standard_error": 0.01,
        "artifact": {"digest": "artifact"},
        "selector_contract": {"digest": "selector"},
        "validation": {"status": ResultStatus.OK.value},
        "resources": {"peak_rss_gib": 1.0},
        "provenance": {"source": "test"},
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
        **{
            field: 0
            for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS
        },
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
        "execution_mode": execution_mode,
        "warmed_boundary_wall_seconds_per_point": 1.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(
            arena_evidence
        ),
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
        "sample_contract": (
            "accumulated-repeated-warmed-evaluator-total-v1"
        ),
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
        "sample_contract": (
            "paired_unprofiled_headline_profiled_attribution_v1"
        ),
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
            entry["measurement"] = {
                "status": ResultStatus.OK.value,
                "generation_seconds": 1.0,
                "wall_seconds_per_point": 2.0e-6,
                "execution_seconds_per_point": 1.0e-6,
                "matrix_element": 1.0,
                "sample_count": 5,
                "standard_error_seconds_per_point": 1.0e-9,
                "relative_standard_error": 0.01,
                "artifact": {"digest": "artifact"},
                "selector_contract": {"digest": "selector"},
                "validation": {
                    "status": ResultStatus.OK.value,
                    "high_precision": {
                        "status": ResultStatus.OK.value,
                        "relative_difference": 0.0,
                    },
                },
                "resources": {"peak_rss_gib": 1.0},
                "provenance": {"source": "test"},
                "failure": None,
            }


def _fill_visible_n4_scope(caches: dict[str, dict[str, object]]) -> None:
    _fill_visible_scope(caches, max_n_final=4)


_POLICY_IDENTITY = ReportSourceIdentity("1" * 40, "2" * 40, ())


def _entry(
    cache: dict[str, object],
    *,
    process_key: str,
    n_final: int,
    workload: Workload,
) -> dict[str, object]:
    entries = cache["entries"]
    assert isinstance(entries, list)
    return next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == n_final
        and item["workload"] == workload.value
        and item["variant"] is None
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


def test_all_twelve_matrices_render_in_catalog_order(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)
    expected = [dataset.table_name for dataset in REPORT_CATALOG.matrix_datasets]

    assert len(rendered) == 12
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
    assert r"\multicolumn{6}{@{\hspace{0.06in}}c}{\textbf{n=1}}" in lc_tex
    assert r"\multicolumn{3}{@{\hspace{0.06in}}c}{\textbf{n=1}}" in contracted_tex
    assert r"\textcolor{ReportMuted}{\scriptsize generation [s]}" in lc_tex
    assert (
        r"\textcolor{ReportMuted}{\scriptsize runtime "
        r"[\(\mu\mathrm{s}/\mathrm{pt}\)]}"
    ) in lc_tex
    assert r"\matrixcolumnheading" not in lc_tex


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
        if line.startswith(
            r" &  & \textcolor{ReportMuted}{\scriptsize runtime"
        )
    )
    assert (
        r"\textcolor{ReportMuted}{\scriptsize generation [s]}"
        r" & \texttt{10.0} & \texttt{10.0} &  & "
        r"\bestmoderatio{ReportGreen}{0.400} & "
        r"\bestmodencprefix{\texttt{5.00}} & \bestmodenclabel"
    ) in row
    assert (
        runtime_row.count(
            r"\bestmodeopenprefix & "
            r"\bestmodeprimaryratio{ReportGreen}{0.100}"
        )
        == 2
    )
    assert (
        r">{\matrixentryfontlc}r@{\hspace{0.08em}}"
        r">{\matrixentryfontlc}l"
    ) in tex
    assert (
        r"\textbf{metric} & \multicolumn{6}"
        r"{@{\hspace{0.06in}}c}{\textbf{n=1}}"
    ) in tex
    assert r"\matrixcolumnheading" not in tex
    assert r"\shortstack{\textbf{n=" not in tex
    assert r"\bestmodecode{" not in row
    assert r"\matrixtotalevaluator{" not in row
    assert r"\matrixrecurrencecore{" not in row
    generation_summary = next(
        line
        for line in tex.splitlines()
        if r"\textbf{summary: generation}" in line
    )
    assert (
        r"\providecommand{\bestmodesummarypair}[3]{"
        r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}"
        r"#1&#2\\[-0.16em]\multicolumn{2}{@{}l@{}}{#3}\end{tabular}}"
    ) in tex
    assert r"\bestmodesummarypair{" in generation_summary
    assert r"}{\bestmodemix{A:0|B:1|C:0}}" in generation_summary
    assert r"}{\bestmodemix{A:0|B:0|C:1}}" in generation_summary
    assert r"\matrixratio{ReportGreen}{0.400}\bestmodemix" not in generation_summary
    assert r"\matrixratio{ReportGreen}{0.500}\bestmodemix" not in generation_summary


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
    assert {item.mode for item in view.workloads} == {
        ExecutionMode.RECURRENCE
    }
    rendered = render_all_best_mode_tables(caches)
    assert list(rendered) == [
        "result_matrix_best_builtin_sm_lc_table.tex",
        "result_matrix_best_builtin_sm_nlc_table.tex",
        "result_matrix_best_builtin_sm_full_table.tex",
    ]


def test_best_mode_renders_mixed_policy_censors_without_a_winner_code(
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
    candidate_entries[ExecutionMode.EAGER]["measurement"] = (
        policy_censor_measurement(
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
    )

    tex = render_best_mode_table(Accuracy.NLC, caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    runtime_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(
            r" &  & \textcolor{ReportMuted}{\scriptsize runtime"
        )
    )

    marker = r"\matrixstatus{ReportOrange}{>2h | >80GB | dependency}"
    assert row.count(marker) == 1
    assert runtime_row.count(marker) == 1
    assert (
        r">{\matrixentryfont}l@{\hspace{0.050in}}"
        r">{\matrixentryfont}r@{\hspace{0.08em}}"
        r">{\matrixentryfont}l"
    ) in tex
    assert r"\bestmodecode{" not in row
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
        recurrence_cell = REPORT_CATALOG.cell(
            str(recurrence_entry["cell_id"])
        )
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
        line
        for line in tex.splitlines()
        if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: wall}" in line
    )
    marker = r"\matrixstatus{ReportOrange}{>2h | >80GB | dependency}"
    assert generation_summary.count(marker) == 1
    assert wall_summary.count(marker) == 1

    completeness = summarize_visible_completeness(
        caches,
        max_n_final=4,
        policy=X86_EPYC_POLICY,
    )
    assert completeness.complete
    assert not completeness.applicable_na_display_slots


def test_best_mode_terminal_baselines_are_visibly_complete(
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
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    marker = r"\matrixstatus{ReportOrange}{>80GB}"
    assert row.count(marker) >= 2
    generation_summary = next(
        line
        for line in tex.splitlines()
        if r"\textbf{summary: generation}" in line
    )
    wall_summary = next(
        line for line in tex.splitlines() if r"\textbf{summary: wall}" in line
    )
    assert marker in generation_summary
    assert marker in wall_summary
    assert r"\matrixna{ReportMuted}" not in row
    assert r"\matrixna{ReportMuted}" not in generation_summary
    assert r"\matrixna{ReportMuted}" not in wall_summary


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
        "result_matrix_best_builtin_sm_nlc_table.tex/n1/"
        "dd_z_jets/contracted"
        in slot
        for slot in completeness.applicable_na_display_slots
    )


def test_matrix_baseline_labels_follow_dataset_contract(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)

    for dataset in REPORT_CATALOG.matrix_datasets:
        tex = rendered[dataset.table_name]
        if dataset.candidate.execution_mode is ExecutionMode.RECURRENCE:
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
        assert r"\fontsize{6.5pt}{7.5pt}\selectfont" in tex
        assert r"\begingroup\matrixentryfontlc" in tex
        assert r"\hspace{0.03in}" in tex
        assert tex.index(r"\clearpage") < tex.index(r"\subsection{")
        assert tex.index(r"\subsection{") < tex.index(
            r"\noindent\begin{minipage}{\linewidth}"
        )


def test_inapplicable_and_reset_cells_use_distinct_markers(reset_caches) -> None:
    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc")
    tex = render_matrix_table(dataset, reset_caches)

    assert r"\matrixnotapplicable{ReportMuted}" in tex
    assert r"\matrixstatus{ReportMuted}{N/A}" in tex
    assert "neither label denotes an unfilled measurement" in tex


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
    assert all(
        row.count(r"\matrixstaticna{ReportMuted}") == 6
        for row in capped_rows
    )
    assert (
        "user cap: native C++/ASM generation is not attempted above n=6"
        in tex
    )
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
    for _variant, mode, label, selected_total, all_total in modes:
        row = next(
            line
            for line in tex.splitlines()
            if line.startswith("1 & ") and label in line
        )
        assert (
            rf"\matrixtotalevaluator{{\texttt{{{selected_total:.0f}}}}}"
            in row
        )
        assert (
            rf"\matrixtotalevaluator{{\texttt{{{all_total:.0f}}}}}"
            in row
        )
        assert r"\matrixtotalevaluator{\texttt{1" not in row
        assert r"\matrixtotalevaluator{\texttt{10" not in row
        if mode == "recurrence":
            assert (
                r"\matrixzrecurrenceclocks{"
                rf"\matrixtotalevaluator{{\texttt{{{selected_total:.0f}}}}}"
                r"}{\matrixrecurrencecore{"
                rf"\texttt{{{70.0 + selected_total:.0f}}}}}"
            ) in row
            assert (
                r"\matrixzrecurrenceclocks{"
                rf"\matrixtotalevaluator{{\texttt{{{all_total:.0f}}}}}"
                r"}{\matrixrecurrencecore{"
                rf"\texttt{{{70.0 + all_total:.0f}}}}}"
            ) in row
    assert r"\textbf{eval total T}" in tex
    assert r"\textbf{rec. core C}" in tex
    assert r"\textbf{[\(\mu\mathrm{s}/\mathrm{pt}\)]}" in tex
    assert "Neither T nor C is derived from wall time or from the other" in tex


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
    assert r"\matrixtotalevaluator{\texttt{217.812}}" in row
    assert r"\matrixrecurrencecore{\texttt{183.456}}" in row


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
        "sample_contract": (
            "paired_unprofiled_headline_profiled_attribution_v1"
        ),
    }

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & ") and "recurrence JIT O2" in line
    )

    assert (
        r"\matrixzrecurrenceclocks{"
        r"\matrixtotalevaluator{\matrixnotexposed{ReportMuted}}}"
        r"{\matrixrecurrencecore{\texttt{7}}}"
    ) in row
    assert r"\matrixtotalevaluator{\texttt{3}}" not in row

    provenance["execution_timing"]["source"] = (
        "runtime_profile_core_evaluator_call_time"
    )
    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith("1 & ") and "recurrence JIT O2" in line
    )
    assert (
        r"\matrixzrecurrenceclocks{"
        r"\matrixtotalevaluator{\matrixnotexposed{ReportMuted}}}"
        r"{\matrixrecurrencecore{\matrixnotexposed{ReportMuted}}}"
    ) in row
    assert r"\matrixtotalevaluator{\texttt{3}}" not in row


def test_visible_completeness_accounts_for_every_n4_slot(reset_caches) -> None:
    caches = copy.deepcopy(reset_caches)
    _fill_visible_n4_scope(caches)

    summary = summarize_visible_completeness(caches, max_n_final=4)
    evidence = summary.as_dict()

    assert summary.complete
    assert evidence["required_measurement_count"] == 742
    assert evidence["rendered_required_measurement_count"] == 742
    assert evidence["structurally_not_applicable_display_slot_count"] == 288
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
    assert evidence["declared_measurement_cell_count"] == 1666
    assert evidence["required_measurement_count"] == 1634
    assert evidence["catalog_static_na_cell_count"] == 32
    assert evidence["rendered_catalog_static_na_cell_count"] == 32
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
    assert r"\texttt{4}" in tex
    assert r"\matrixratio{ReportGreen}{0.5}" in tex


def test_recurrence_matrix_cell_keeps_wall_total_and_core_distinct(
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

    assert (
        r"\matrixruntimetriple{\matrixwallclock{ReportRed}{x2.18}}"
        r"{\matrixtotalevaluator{\texttt{217.812}}}"
        r"{\matrixrecurrencecore{\texttt{183.456}}}"
    ) in tex
    assert "Neither T nor C is derived from wall time or from the other" in tex
    assert "C is never relabeled as evaluator total" in tex


def test_unavailable_execution_is_not_exposed_and_has_no_zero_ratio(
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

    assert (
        r"\matrixruntimepair{\matrixwallclock{ReportGreen}{x0.5}}"
        r"{\matrixtotalevaluator{\matrixnotexposed{ReportMuted}}}"
    ) in tex
    assert "Not exposed means that a successful wall-time measurement" in tex
    assert r"{x0}" not in tex


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_future_dag_evaluator_total_is_absolute_and_never_ratioed(
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

    assert (
        r"\matrixruntimepair{\matrixwallclock{ReportGreen}{x0.5}}"
        r"{\matrixtotalevaluator{\texttt{0.9}}}"
    ) in tex
    assert "absolute evaluator-total value marked T" in tex
    assert (
        r"\matrixruntimetriple{\matrixwallclock{ReportGreen}{x0.5}}"
        not in tex
    )


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

    assert tex.count(r"\matrixncabsolute{\texttt{10}}") >= 2
    assert r"\matrixratio{ReportRed}{1e+05}" not in tex
    assert "n.c. (not comparable)" in tex


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

    assert r"\matrixncabsolute{\texttt{4}}" in tex
    assert r"\matrixncabsolute{\texttt{10}}" in tex
    assert tex.count(
        r"\matrixruntimetriple{"
        r"\matrixncabsolute{\matrixwallabsolute{\texttt{2}}}"
    ) >= 2
    assert tex.count(r"\matrixstaticna{ReportMuted}") >= 2
    assert "n.c. (not comparable)" in tex


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
    assert r"\matrixncabsolute{\texttt{12}}" in tex
    assert r"\matrixstaticna{ReportMuted}" in tex
    assert "n.c. means not comparable" in tex


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

    assert r"\matrixratio{ReportGreen}{0.5}" in tex
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


def test_scalar_timing_marks_unavailable_arena_attribution_not_exposed(
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
        wall=1.0e-6,
        execution=None,
    )
    entries = cache["entries"]
    assert isinstance(entries, list)
    measurement = next(
        entry["measurement"]
        for entry in entries
        if entry["n_final"] == 2
    )
    _mark_arena_unavailable(measurement)

    tex = render_scalar_ladder(dataset, caches)

    assert r"evaluator total [$\mu$s/pt]" in tex
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
        entry["measurement"]
        for entry in entries
        if entry["n_final"] == 2
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
        line for line in tex.splitlines() if line.startswith(r"wall [$\mu$s/pt]")
    )
    total_row = next(
        line
        for line in tex.splitlines()
        if line.startswith(r"evaluator total [$\mu$s/pt]")
    )

    assert r"\texttt{218.105}" in wall_row
    assert r"\texttt{217.812}" in total_row
    assert r"\texttt{218.105}" not in total_row
    assert r"execution [$\mu$s/pt]" not in tex
    assert "never copied from or derived from wall time" in tex


def test_all_outputs_include_matrices_z_and_scalar_ladders(
    reset_caches,
) -> None:
    rendered = render_all_tables(reset_caches)

    assert len(rendered) == 20
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
        (1, 64),
        (2, 186),
        (3, 206),
        (4, 286),
        (5, 285),
        (6, 181),
        (7, 155),
        (8, 155),
        (9, 116),
    )
    assert summary.declared_total == 1666
    assert summary.static_na_by_n == (
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 4),
        (7, 10),
        (8, 10),
        (9, 8),
    )
    assert summary.static_na_total == 32
    assert summary.expected_total == 1634
    assert summary.passed_total == 4
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
    assert "1666 & 32 & 4" in tex
    assert "1666 declared cells" in tex
    assert "1634 measurable cells" in tex
    assert "32 catalog-authenticated static N/A" in tex
    assert "412 matrix process/multiplicity positions" in tex
    assert "36 reference execution fields" in tex
    assert rf"\nolinkurl{{{revision}}}" in tex
