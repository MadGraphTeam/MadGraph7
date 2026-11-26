# SPDX-License-Identifier: 0BSD
"""A relative uncertainty or stability estimate above 1 is shown as ">1"
(USER 2026-11-18, a display rule): one helper (``loops.v2.stability.
relative_text``) for every human-readable place; the records, the JSON and
the samples keep the number."""

from __future__ import annotations

import pytest

from pyamplicol.api.virtual_results import (
    VirtualBestEffort,
    VirtualRescueAttempt,
    VirtualRescueRecord,
    best_effort_detail,
)
from pyamplicol.loops.v2.rescue import _detail
from pyamplicol.loops.v2.stability import relative_text


@pytest.mark.loop_fast
def test_relative_text_shows_above_one_as_greater_than_one():
    assert relative_text(3.4e18) == ">1"
    assert relative_text(float("inf")) == ">1"
    assert relative_text(1.0000001) == ">1"
    # 1 itself and below keep the number, in the caller's format.
    assert relative_text(1.0) == "1.00e+00"
    assert relative_text(2.5e-7) == "2.50e-07"
    assert relative_text(0.25, ".3g") == "0.25"
    assert relative_text(0.0, ".3e") == "0.000e+00"


@pytest.mark.loop_fast
def test_lane_details_show_greater_than_one_and_records_keep_the_number():
    record = VirtualBestEffort(
        "NinjaRS refused the point",
        49.0,
        "recommended-checks",
        backend_deviation=7.5,
        gram_conditioning=3.0e3,
    )
    text = best_effort_detail(record)
    assert "relative uncertainty >1 (recommended checks)" in text
    assert "backend deviation >1" in text
    # A conditioning measure is not an uncertainty: its number stays.
    assert "Gram conditioning 3.00e+03" in text
    assert record.uncertainty == 49.0 and record.backend_deviation == 7.5
    small = VirtualBestEffort("refused", 2.5e-7, "recommended-checks")
    assert "relative uncertainty 2.50e-07" in best_effort_detail(small)

    attempt = VirtualRescueAttempt("arbitrary", 190, 14.0)
    rescued = VirtualRescueRecord(0, "arbitrary", 190, 1e-5, False, (attempt,))
    assert _detail(rescued, "relative") == (
        "rescue level 0 (arbitrary): estimated relative accuracy >1, "
        "target 1.0e-05 not met at the final level"
    )
    assert rescued.attempts[0].estimate == 14.0
