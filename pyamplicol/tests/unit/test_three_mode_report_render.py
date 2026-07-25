# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy

import pytest

import tools.performance_report.render as report_render
from tools.performance_report.cache import build_reset_caches
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.render import (
    BaselineCandidateAdapter,
    render_all_matrix_tables,
    render_all_tables,
    render_all_z_ladders,
    render_matrix_table,
    render_scalar_ladder,
    render_z_ladder,
)
from tools.performance_report.validation_summary import (
    render_validation_summary,
    summarize_validation,
)


@pytest.fixture
def reset_caches() -> dict[str, dict[str, object]]:
    return build_reset_caches()


def _cache_by_dataset(
    caches: dict[str, dict[str, object]],
    dataset_id: str,
) -> dict[str, object]:
    return next(
        payload
        for payload in caches.values()
        if payload["dataset_id"] == dataset_id
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


def _mark_below_resolution(measurement: dict[str, object]) -> None:
    measurement["execution_seconds_per_point"] = None
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
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


def test_all_twelve_matrices_render_in_catalog_order(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)
    expected = [
        dataset.table_name for dataset in REPORT_CATALOG.matrix_datasets
    ]

    assert len(rendered) == 12
    assert list(rendered) == expected


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
        assert tex.count(r"\clearpage") == tex.count(
            r"\begin{minipage}{\linewidth}"
        )
        assert r"\begin{tabular}" in tex
        assert tex.index(r"\clearpage") < tex.index(r"\subsection{")
        assert tex.index(r"\subsection{") < tex.index(
            r"\noindent\begin{minipage}{\linewidth}"
        )


def test_inapplicable_and_reset_cells_render_explicit_na(reset_caches) -> None:
    dataset = REPORT_CATALOG.dataset("matrix_recurrence_builtin_sm_lc")
    tex = render_matrix_table(dataset, reset_caches)

    assert r"\matrixna{ReportMuted}" in tex
    assert r"\matrixstatus{ReportMuted}{N/A}" in tex


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


def test_below_resolution_execution_is_explicit_and_has_no_zero_ratio(
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
    _mark_below_resolution(measurement)

    tex = render_matrix_table(
        REPORT_CATALOG.dataset("matrix_compiled_builtin_sm_nlc"),
        caches,
    )

    assert (
        r"\matrixratiopair{ReportGreen}{x0.5}"
        r"{ReportMuted}{below res.}"
    ) in tex
    assert r"{x0}" not in tex


def test_z_ladder_prints_below_resolution_status_instead_of_zero(
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
    _mark_below_resolution(measurement)

    tex = render_z_ladder(ModelKey.BUILTIN_SM, caches)

    assert r"\matrixstatus{ReportMuted}{below res.}" in tex
    assert r"{x0}" not in tex


def test_below_resolution_measurements_do_not_enter_summary_ratios(
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
    _mark_below_resolution(measurement)
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

    assert summary == r"\matrixna{ReportMuted}"


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


def test_scalar_timing_uses_explicit_below_resolution_bound(
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
    _mark_below_resolution(measurement)

    tex = render_scalar_ladder(dataset, caches)

    assert r"\matrixstatus{ReportMuted}{below res.}" in tex


def test_all_outputs_include_matrices_z_and_scalar_ladders(
    reset_caches,
) -> None:
    rendered = render_all_tables(reset_caches)

    assert len(rendered) == 17
    assert "result_validation_summary.tex" in rendered
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

    assert summary.expected_by_n == (
        (1, 64),
        (2, 186),
        (3, 206),
        (4, 286),
    )
    assert summary.expected_total == 742
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
    assert "742 & 4" in tex
    assert rf"\nolinkurl{{{revision}}}" in tex
