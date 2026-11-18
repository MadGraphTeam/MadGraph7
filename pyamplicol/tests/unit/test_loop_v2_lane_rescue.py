# SPDX-License-Identifier: 0BSD
"""The lane rescue's records (USER 2026-11-18): a reduction lane the backend
refuses in the evaluation's carrier is reduced again in a higher carrier
inside the native executor (``rusticol-core loop_execution::v2``, its own
Rust tests); these tests cover what Python makes of the executor's
``rescued`` output: the per-point record (``VirtualLaneRescue``), its sum
over a point's evaluations, the lane diagnostic, and ``profile``'s cause
tables."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_a_lane_rescue_record_names_its_carriers_and_counts():
    from pyamplicol.api.virtual_results import VirtualLaneRescue

    record = VirtualLaneRescue((("double-double", 3), ("arbitrary", 1)))
    assert (record.count, record.carrier) == (4, "arbitrary")
    assert VirtualLaneRescue((("double-double", 2),)).carrier == "double-double"
    for bad in ((), (("double", 1),), (("double-double", 0),)):
        with pytest.raises(ValueError):
            VirtualLaneRescue(bad)


def test_the_native_rescued_lanes_are_counted_per_carrier_over_evaluations():
    from pyamplicol.loops.v2.plan_export import merged_rescue, point_rescued

    output = {
        "rescued": [
            [((3, 0, 0), "double-double"), ((3, 0, 1), "double-double")],
            [],
        ]
    }
    extra = {"rescued": [[((1, 0, 0), "arbitrary")], []]}
    assert point_rescued(output, extra, 0) == (("arbitrary", 1), ("double-double", 2))
    assert point_rescued(output, None, 1) == ()
    # An output without the key (an older executor): nothing rescued.
    assert point_rescued({}, None, 0) == ()
    reference = SimpleNamespace(lane_rescue=(("double-double", 2),))
    rotated = SimpleNamespace(lane_rescue=(("arbitrary", 1), ("double-double", 1)))
    plain = SimpleNamespace()
    assert merged_rescue([reference, rotated, plain]) == (
        ("arbitrary", 1),
        ("double-double", 3),
    )
    assert merged_rescue([plain]) == ()


def test_a_rescued_point_is_a_success_whose_diagnostic_carries_the_record():
    from pyamplicol.api.virtual_results import VirtualLaneDiagnostic, VirtualLaneRescue
    from pyamplicol.benchmarking import lane_rescue_reason

    lane = VirtualLaneDiagnostic(
        "success",
        "double",
        53,
        False,
        lane_rescue=VirtualLaneRescue((("double-double", 1),)),
    )
    assert lane.valid and lane.lane_rescue.count == 1
    assert lane_rescue_reason(lane) == "reduction lanes rescued in double-double"
    assert lane_rescue_reason(VirtualLaneDiagnostic("success", "double", 53, False)) is None


def test_profile_counts_lane_rescued_points_in_the_stability_statistics():
    from pyamplicol.loop_profile import _backend_statistics, failure_causes

    plain = [((1.0, None, None),), ((2.0, None, None),)]
    reference = [((1.0, None, None),), ((2.0, None, None),)]
    estimate = SimpleNamespace(relative=1e-12, absolute=None, finite=1e-12)
    row = _backend_statistics(
        "ninjars",
        plain,
        [estimate, estimate],
        reference,
        1e-5,
        0.1,
        rescues=[None, "reduction lanes rescued in double-double"],
    )
    assert row.lane_rescued == 1
    assert [point.lane_rescue for point in row.per_point] == [
        None,
        "reduction lanes rescued in double-double",
    ]
    # Rescued points are measured like any other (no failure, no flag).
    assert (row.points, row.failed_binary64, row.flagged) == (2, 0, 0)
    causes = failure_causes(
        [("ninjars", "stability", "lane rescue", row.per_point[1].lane_rescue)],
        {("ninjars", "stability"): 2},
    )
    assert [(c.kind, c.count, c.total, c.reason) for c in causes] == [
        ("lane rescue", 1, 2, "reduction lanes rescued in double-double")
    ]


def test_the_lane_rescue_stage_is_part_of_the_reduction_group():
    from pyamplicol.runtime.virtual import PROFILE_REDUCTION, profile_stages

    stages = profile_stages({"reduction": 1.0, "lane rescue": 0.5}, 2.0, "virtual")
    assert stages[PROFILE_REDUCTION] == pytest.approx(1.5)
