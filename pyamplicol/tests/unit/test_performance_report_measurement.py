# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import (
    _baseline_matrix_element,
    _baseline_selector_contract,
    failure_measurement,
)
from tools.performance_report.models import ResultStatus
from tools.performance_report.runner import RunnerError, SelectorContract


def _contract() -> SelectorContract:
    return SelectorContract(
        selected_color_flow_ids=("flow:1,2,3",),
        selected_color_words=((1, 2, 3),),
        all_flow_helicity_ids=("h:-1,+1,-1",),
        all_flow_source_helicities=((1, -1), (2, 1), (3, -1)),
        point_digest="a" * 64,
    )


def test_baseline_contract_and_matrix_element_are_strict() -> None:
    baseline = {
        "status": "ok",
        "matrix_element": 2.0,
        "selector_contract": _contract().as_dict(),
    }
    assert _baseline_selector_contract(baseline) == _contract()
    assert _baseline_matrix_element(baseline) == 2.0

    with pytest.raises(RunnerError, match="not a valid completed"):
        _baseline_matrix_element({"status": "error"})
    with pytest.raises(RunnerError, match="no matrix element"):
        _baseline_matrix_element({"status": "ok", "matrix_element": None})


def test_failure_measurement_preserves_compact_cache_shape() -> None:
    measurement = failure_measurement(
        ResultStatus.MEMORY_LIMIT,
        RuntimeError("over limit"),
        resources={"peak_rss_bytes": 42},
    )

    assert measurement["status"] == "memory_limit"
    assert measurement["generation_seconds"] is None
    assert measurement["resources"] == {"peak_rss_bytes": 42}
    assert measurement["failure"] == {
        "kind": "RuntimeError",
        "message": "over limit",
    }


def test_catalog_contains_no_amplicol_candidate_matrix_cell() -> None:
    assert all(
        cell.measurement.execution_mode.value != "amplicol"
        for cell in REPORT_CATALOG.matrix_cells()
    )
