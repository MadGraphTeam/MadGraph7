# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy

import pytest

from tools.performance_report.cache import build_reset_caches
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.render import (
    BaselineCandidateAdapter,
    _ratio_value,
    render_all_matrix_tables,
    render_all_tables,
    render_all_z_ladders,
    render_matrix_table,
)


def test_below_resolution_execution_timing_is_never_ratioed() -> None:
    baseline = {
        "status": ResultStatus.OK.value,
        "execution_seconds_per_point": 1.0e-6,
    }
    candidate = {
        "status": ResultStatus.OK.value,
        "execution_seconds_per_point": 1.0e-9,
        "provenance": {
            "execution_timing": {
                "status": "below_timer_resolution",
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
    execution: float,
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


def test_all_twelve_matrices_render_in_catalog_order(reset_caches) -> None:
    rendered = render_all_matrix_tables(reset_caches)
    expected = [dataset.table_name for dataset in REPORT_CATALOG.matrix_datasets]

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
        assert tex.count(r"\clearpage") == tex.count(r"\begin{minipage}{\linewidth}")
        assert r"\begin{tabular}" in tex


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


def test_all_outputs_include_matrices_z_and_scalar_ladders(
    reset_caches,
) -> None:
    rendered = render_all_tables(reset_caches)

    assert len(rendered) == 16
    assert set(render_all_matrix_tables(reset_caches)) < set(rendered)
    assert set(render_all_z_ladders(reset_caches)) < set(rendered)
    assert "result_scalar_contact_table.tex" in rendered
    assert "result_scalar_gravity_table.tex" in rendered
