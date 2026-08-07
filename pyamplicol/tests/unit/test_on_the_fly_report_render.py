# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy

import pytest

from tools.performance_report.cache import build_reset_caches
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.render import (
    BaselineCandidateAdapter,
    JoinedWorkload,
    _fixed_mode_generation_comparison_layout,
    render_best_mode_table,
    render_matrix_table,
)


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
        "selector_contract": None,
        "validation": {"status": "ok"},
        "resources": {"peak_rss_gib": 1.0},
        "provenance": {
            "runtime_load_seconds": 100.0,
            "runtime_profile": runtime_profile,
        },
        "failure": None,
    }


@pytest.fixture
def otf_render_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    caches = build_reset_caches()
    baseline = _cache(caches, "reference_amplicol_lc")
    candidate = _cache(caches, "matrix_on_the_fly_builtin_sm_lc")
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
        )
        _set_measurement(
            candidate,
            process_key="dd_z_jets",
            workload=workload,
            generation=generation,
            wall=1.0e-6,
            cold_warmup=cold_warmup,
        )
    monkeypatch.setattr(
        BaselineCandidateAdapter,
        "_comparison_linked",
        lambda *_args, **_kwargs: True,
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
    assert r"\matrixna{ReportMuted}" in tex
