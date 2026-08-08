# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest

from tools.performance_report import render as report_render
from tools.performance_report.cache import build_reset_caches
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.render import (
    BaselineCandidateAdapter,
    JoinedWorkload,
    _fixed_mode_generation_comparison_layout,
    _fixed_mode_runtime_comparison_layout,
    render_best_mode_table,
    render_matrix_table,
)
from tools.performance_report.runner import (
    OTF_RECURRENCE_AUTHORITY_VALIDATION_FIELD,
)
from tools.performance_report.validation_summary import render_validation_summary

_POINT_DIGEST = "1" * 64
_OTF_ARTIFACT_ID = "a" * 64
_RECURRENCE_ARTIFACT_ID = "b" * 64
_AMPLICOL_ARTIFACT_ID = "c" * 64
_SELECTOR_CONTRACT = {
    "selected_color_flow_ids": ["flow:2,1"],
    "selected_color_words": [[2, 1]],
    "all_flow_helicity_ids": ["h:-1,+1,-1"],
    "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
    "point_digest": _POINT_DIGEST,
}


def _cache(caches: dict[str, dict[str, object]], dataset_id: str) -> dict[str, object]:
    return next(
        payload for payload in caches.values() if payload["dataset_id"] == dataset_id
    )


def _set_measurement(
    cache: dict[str, object],
    *,
    process_key: str,
    workload: Workload,
    generation: float,
    wall: float,
    cold_warmup: float | None = None,
    selector_contract: Mapping[str, object] | None = None,
    artifact_id: str = _AMPLICOL_ARTIFACT_ID,
    validation: Mapping[str, object] | None = None,
) -> None:
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == 1
        and item["workload"] == workload.value
        and item["variant"] is None
    )
    runtime_profile = {
        "warmup_elapsed_seconds": 50.0,
    }
    if cold_warmup is not None:
        runtime_profile["cold_warmup_elapsed_seconds"] = cold_warmup
    entry["measurement"] = {
        "status": "ok",
        "generation_seconds": generation,
        "wall_seconds_per_point": wall,
        "execution_seconds_per_point": wall,
        "matrix_element": 1.0,
        "sample_count": 5,
        "standard_error_seconds_per_point": 1.0e-9,
        "relative_standard_error": 0.01,
        "artifact": {"digest": "test"},
        "selector_contract": copy.deepcopy(selector_contract),
        "validation": dict(validation or {"status": "ok"}),
        "resources": {"peak_rss_gib": 1.0},
        "provenance": {
            "runtime_load_seconds": 100.0,
            "runtime_profile": runtime_profile,
            "runtime_identity": {"artifact_id": artifact_id},
        },
        "failure": None,
    }


def _measurement(
    cache: dict[str, object],
    *,
    process_key: str,
    workload: Workload,
) -> dict[str, object]:
    entries = cache["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if item["process_key"] == process_key
        and item["n_final"] == 1
        and item["workload"] == workload.value
        and item["variant"] is None
    )
    measurement = entry["measurement"]
    assert isinstance(measurement, dict)
    return measurement


def _pointwise_validation() -> dict[str, object]:
    return {
        "status": "ok",
        "pointwise": {
            "status": "ok",
            "candidate": 1.0,
            "baseline": 1.0,
        },
    }


@pytest.fixture
def otf_render_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    caches = build_reset_caches()
    baseline = _cache(caches, "reference_amplicol_lc")
    candidate = _cache(caches, "matrix_on_the_fly_builtin_sm_lc")
    recurrence = _cache(caches, "matrix_recurrence_builtin_sm_lc")
    for workload, baseline_generation, generation, cold_warmup in (
        (Workload.SELECTED_FLOW, 10.0, 4.0, 1.0),
        (Workload.ALL_FLOW, 20.0, 6.0, 2.0),
    ):
        _set_measurement(
            baseline,
            process_key="dd_z_jets",
            workload=workload,
            generation=baseline_generation,
            wall=10.0e-6,
            selector_contract=_SELECTOR_CONTRACT,
        )
        _set_measurement(
            recurrence,
            process_key="dd_z_jets",
            workload=workload,
            generation=8.0,
            wall=2.0e-6,
            selector_contract=_SELECTOR_CONTRACT,
            artifact_id=_RECURRENCE_ARTIFACT_ID,
        )
        _set_measurement(
            candidate,
            process_key="dd_z_jets",
            workload=workload,
            generation=generation,
            wall=1.0e-6,
            cold_warmup=cold_warmup,
            selector_contract=_SELECTOR_CONTRACT,
            artifact_id=_OTF_ARTIFACT_ID,
            validation={
                "status": "ok",
                OTF_RECURRENCE_AUTHORITY_VALIDATION_FIELD: {
                    "authority": {"artifact_id": _RECURRENCE_ARTIFACT_ID},
                },
            },
        )
    monkeypatch.setattr(
        report_render,
        "validate_on_the_fly_recurrence_authority_validation_record",
        lambda *_args, **_kwargs: None,
    )
    return caches


def test_otf_fixed_lc_table_pairs_generation_with_only_first_cold_warmup(
    otf_render_caches: dict[str, dict[str, object]],
) -> None:
    dataset = REPORT_CATALOG.dataset("matrix_on_the_fly_builtin_sm_lc")
    tex = render_matrix_table(dataset, otf_render_caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))

    assert (
        r"\bestmodecompactprefix{0.400} & "
        r"\bestmodeprimaryratio{ReportGreen}{0.500}"
    ) in row
    assert (
        r"\bestmodecompactabsoluteprefix{6.00} & "
        r"\bestmodeabsoluteprimary{8.00}"
    ) in row
    assert (
        r"\providecommand{\bestmodecompactabsolute}[2]{"
        r"\textcolor{ReportMuted}{\texttt{[#1]}}"
        r"\hspace{0.04in}\textcolor{black}{\texttt{#2}}}"
    ) in tex
    assert r"\renewcommand{\arraystretch}{1.00}" in tex
    assert "W excludes artifact loading and conventional benchmark warm-ups" in tex
    assert "104" not in row
    assert "56.0" not in row


def test_otf_best_mode_winner_keeps_pair_semantics_and_o_marker(
    otf_render_caches: dict[str, dict[str, object]],
) -> None:
    tex = render_best_mode_table(Accuracy.LC, otf_render_caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))

    assert (
        r"\bestmodecodeprefix{o}\bestmodecompactprefix{0.400} & "
        r"\bestmodeprimaryratio{ReportGreen}{0.500}"
    ) in row
    assert (
        r"\bestmodecompactabsoluteprefix{6.00} & "
        r"\bestmodeabsoluteprimary{8.00}"
        r"\hspace{0.025in}\bestmodecode{o}"
    ) in row
    assert "r|c|e|o" in tex
    assert "follows the enclosing cold-start contract" in tex


@pytest.mark.parametrize("accuracy", (Accuracy.NLC, Accuracy.FULL))
def test_otf_contracted_best_mode_winner_is_in_row_mix_and_legend(
    accuracy: Accuracy,
) -> None:
    caches = build_reset_caches()
    workload = Workload.CONTRACTED
    baseline = _cache(caches, f"reference_amplicol_{accuracy.value}")
    recurrence = _cache(
        caches,
        f"matrix_recurrence_builtin_sm_{accuracy.value}",
    )
    candidate = _cache(
        caches,
        f"matrix_on_the_fly_builtin_sm_{accuracy.value}",
    )
    _set_measurement(
        baseline,
        process_key="dd_z_jets",
        workload=workload,
        generation=10.0,
        wall=10.0e-6,
    )
    _set_measurement(
        recurrence,
        process_key="dd_z_jets",
        workload=workload,
        generation=8.0,
        wall=2.0e-6,
        artifact_id=_RECURRENCE_ARTIFACT_ID,
        validation=_pointwise_validation(),
    )
    _set_measurement(
        candidate,
        process_key="dd_z_jets",
        workload=workload,
        generation=4.0,
        wall=1.0e-6,
        cold_warmup=1.0,
        artifact_id=_OTF_ARTIFACT_ID,
        validation=_pointwise_validation(),
    )

    view = BaselineCandidateAdapter(caches).best_mode_cell(
        accuracy,
        REPORT_CATALOG.process_families[0],
        1,
    )
    assert view.workloads[0].mode is ExecutionMode.ON_THE_FLY

    tex = render_best_mode_table(accuracy, caches)
    row = next(line for line in tex.splitlines() if line.startswith(r"\texttt{1}"))
    assert (
        r"\bestmodecodeprefix{o}\bestmodecompactprefix{0.400} & "
        r"\bestmodeprimaryratio{ReportGreen}{0.500}"
    ) in row
    assert r"\bestmodemix{r:0|c:0|e:0|o:1}" in tex
    assert "Summary mode counts use r|c|e|o" in tex
    assert "contracted cold-start contract" in tex


def test_otf_amplicol_links_are_metric_and_workload_specific(
    otf_render_caches: dict[str, dict[str, object]],
) -> None:
    baseline = _cache(otf_render_caches, "reference_amplicol_lc")
    for workload in (Workload.SELECTED_FLOW, Workload.ALL_FLOW):
        measurement = _measurement(
            baseline,
            process_key="dd_z_jets",
            workload=workload,
        )
        selector = measurement["selector_contract"]
        assert isinstance(selector, dict)
        selector["selected_color_flow_ids"] = ["flow:2,4,3,1"]
        selector["selected_color_words"] = [[2, 4, 3, 1]]

    dataset = REPORT_CATALOG.dataset("matrix_on_the_fly_builtin_sm_lc")
    family = next(
        item for item in REPORT_CATALOG.process_families if item.key == "dd_z_jets"
    )
    view = BaselineCandidateAdapter(otf_render_caches).matrix_cell(
        dataset,
        family,
        1,
    )
    selected, all_flow = view.workloads

    assert selected.generation_comparison_linked
    assert not selected.comparison_linked
    assert all_flow.generation_comparison_linked
    assert all_flow.comparison_linked

    selected_generation = _fixed_mode_generation_comparison_layout(
        selected,
        candidate_mode=ExecutionMode.ON_THE_FLY,
        baseline_mode=ExecutionMode.AMPLICOL,
    )
    selected_runtime = _fixed_mode_runtime_comparison_layout(
        selected,
        candidate_mode=ExecutionMode.ON_THE_FLY,
        baseline_mode=ExecutionMode.AMPLICOL,
    )
    all_flow_runtime = _fixed_mode_runtime_comparison_layout(
        all_flow,
        candidate_mode=ExecutionMode.ON_THE_FLY,
        baseline_mode=ExecutionMode.AMPLICOL,
    )
    assert "bestmodecompactratio{0.400}" in selected_generation.inline
    assert selected_runtime.inline == r"\texttt{1.00}"
    assert "bestmodecompactratio{0.100}" in all_flow_runtime.inline

    fixed_tex = render_matrix_table(dataset, otf_render_caches)
    generation_summary = next(
        line
        for line in fixed_tex.splitlines()
        if line.startswith(r"\multicolumn{3}{@{}l}{\textbf{summary: generation}}")
    )
    assert r"\matrixsummaryratio{ReportGreen}{0.400}" in generation_summary
    assert "0.500" not in generation_summary

    best_tex = render_best_mode_table(Accuracy.LC, otf_render_caches)
    best_lines = best_tex.splitlines()
    generation_index = next(
        index
        for index, line in enumerate(best_lines)
        if line.startswith(r"\texttt{1}")
    )
    generation_row = best_lines[generation_index]
    runtime_row = best_lines[generation_index + 2]
    assert r"\bestmodecodeprefix{o}\bestmodecompactprefix{0.400}" in generation_row
    assert r"\bestmodecode{o}" in generation_row
    assert r"\bestmodeabsoluteprefix{\texttt{1.00}}" in runtime_row
    assert r"\bestmodecompactprefix{0.100}" in runtime_row

    best_view = BaselineCandidateAdapter(otf_render_caches).best_mode_cell(
        Accuracy.LC,
        family,
        1,
    )
    best_selected, best_all_flow = best_view.workloads
    assert best_selected.mode is ExecutionMode.ON_THE_FLY
    assert best_selected.generation_comparison_linked
    assert not best_selected.comparison_linked
    assert best_all_flow.comparison_linked

    best_generation_summary = next(
        line
        for line in best_lines
        if line.startswith(r"\multicolumn{3}{@{}l}{\textbf{summary: generation}}")
    )
    assert r"\matrixsummaryratio{ReportGreen}{0.400}" in best_generation_summary
    assert "0.500" not in best_generation_summary


def test_validation_summary_names_otf_recurrence_cross_mode() -> None:
    tex = render_validation_summary(build_reset_caches())

    assert "compiled/eager/OTF versus recurrence" in tex
    assert "cross-mode (including OTF/recurrence)" in tex


def test_otf_pair_is_provenance_strict_and_never_used_against_recurrence() -> None:
    baseline = {
        "status": "ok",
        "generation_seconds": 10.0,
    }
    candidate = {
        "status": "ok",
        "generation_seconds": 4.0,
        "provenance": {
            "runtime_load_seconds": 100.0,
            "runtime_profile": {"cold_warmup_elapsed_seconds": 1.0},
        },
    }
    joined = JoinedWorkload(
        Workload.SELECTED_FLOW,
        baseline,
        candidate,
        comparison_linked=True,
    )

    recurrence_layout = _fixed_mode_generation_comparison_layout(
        joined,
        candidate_mode=ExecutionMode.ON_THE_FLY,
        baseline_mode=ExecutionMode.RECURRENCE,
    )
    assert recurrence_layout.inline == r"\bestmoderatio{ReportGreen}{0.400}"

    missing_cold = copy.deepcopy(candidate)
    missing_cold["provenance"]["runtime_profile"].pop(  # type: ignore[index]
        "cold_warmup_elapsed_seconds"
    )
    missing_layout = _fixed_mode_generation_comparison_layout(
        JoinedWorkload(
            Workload.SELECTED_FLOW,
            baseline,
            missing_cold,
            comparison_linked=True,
        ),
        candidate_mode=ExecutionMode.ON_THE_FLY,
        baseline_mode=ExecutionMode.AMPLICOL,
    )
    assert missing_layout.inline == r"\matrixna{ReportMuted}"


@pytest.mark.parametrize("accuracy", [Accuracy.NLC, Accuracy.FULL])
def test_static_otf_color_tables_keep_the_regular_contracted_layout(
    accuracy: Accuracy,
) -> None:
    caches = build_reset_caches()
    dataset = REPORT_CATALOG.dataset(f"matrix_on_the_fly_builtin_sm_{accuracy.value}")
    tex = render_matrix_table(dataset, caches)

    assert r"\multicolumn{3}{c}{\textbf{n=1}}" in tex
    assert r"\matrixstaticna{ReportMuted}" in tex
