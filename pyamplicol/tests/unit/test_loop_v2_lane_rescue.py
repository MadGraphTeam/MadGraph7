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


def test_the_rescue_text_counts_and_names_the_added_check_only_when_added():
    """Review m-a: "1 lane" / "N lanes" (and moments), and the estimate
    clause only when the rescaled check was added to the call's checks."""

    from pyamplicol.api.virtual_results import (
        VirtualLaneRescue,
        VirtualStabilityChecks,
        rescue_text,
    )
    from pyamplicol.loops.v2.reduction import rescued_checks

    one = VirtualLaneRescue((("double-double", 1),), (("arbitrary", 1),))
    assert rescue_text(one, False) == (
        "lane rescue: 1 lane in double-double, 1 term-scale moment in arbitrary"
    )
    two = VirtualLaneRescue((("double-double", 2),))
    assert rescue_text(two, True) == (
        "lane rescue: 2 lanes in double-double (its estimate includes the "
        "rescaled reduction-direction check)"
    )
    # Added for one rotation; not for the reversed or the rescaled check.
    assert rescued_checks(VirtualStabilityChecks(1)) is not None
    assert rescued_checks(VirtualStabilityChecks(1, True)) is None
    assert rescued_checks(VirtualStabilityChecks(1, True, rescaled=True)) is None


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
    plain = VirtualLaneDiagnostic("success", "double", 53, False)
    assert lane_rescue_reason(plain) is None


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


# ------------------------------------------------------------ native paths

from tests.unit.test_loop_v2_best_effort import aag  # noqa: E402,F401  (fixture)


@pytest.mark.loop_physics
def test_the_c_abi_serves_refused_lanes_and_its_setter_turns_the_rescue_off(
    aag,  # noqa: F811
    loop_native,
):
    """g g > a a g on explicit TIRust (LI-33's stream points, which it
    refuses at the default acceptance): the Python API serves the refused
    points by the lane rescue (records), and without it (``lane_rescue=
    False``) they fail; the C ABI does the same by default (status
    success-class, values bitwise the Python API's) and with
    ``rusticol_virtual_set_lane_rescue(handle, 0)`` they fail again; any
    other value of the setter is an invalid argument."""

    import ctypes
    from fractions import Fraction

    from pyamplicol import Runtime
    from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
    from tests.unit.test_loop_v2_sdk_induced import (
        FAILED_STATUS,
        STATUS_INVALID_ARGUMENT,
        SUCCESS_STATUS,
        _evaluate,
        _open,
        _python_flat,
        _same,
    )

    artifact, exact_points, exact_scales = aag
    points = [[[float(c) for c in leg] for leg in point] for point in exact_points]
    scales = [float(Fraction(s)) for s in exact_scales]
    runtime = Runtime.load(artifact, process="p")
    call = dict(
        mu_r2=exact_scales, poles="none", reduction="tirust", on_failure="return"
    )
    rescued = runtime.evaluate(exact_points, **call)
    plain = runtime.evaluate(exact_points, lane_rescue=False, **call)
    served = [p for p, lane in enumerate(rescued.diagnostics) if lane.lane_rescue]
    assert served, "the fixture points include rescued ones"
    assert all(rescued.diagnostics[p].valid for p in served)
    assert all(plain.statuses[p] == "failed" for p in served)
    sdk = _sdk()
    sdk.rusticol_virtual_set_lane_rescue.argtypes = [ctypes.c_void_p, ctypes.c_int]
    sdk.rusticol_virtual_set_on_failure.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle = _open(sdk, artifact)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, b"none") == 0, _error(sdk)
        assert sdk.rusticol_virtual_set_reduction(handle, b"tirust") == 0
        assert sdk.rusticol_virtual_set_on_failure(handle, b"return") == 0
        found = _evaluate(sdk, handle, points, scales)
        assert all(found["statuses"][p] == SUCCESS_STATUS for p in served), found
        assert _same(found["virtual"], _python_flat(rescued))
        assert (
            sdk.rusticol_virtual_set_lane_rescue(handle, 2) == STATUS_INVALID_ARGUMENT
        )
        assert sdk.rusticol_virtual_set_lane_rescue(handle, 0) == 0, _error(sdk)
        off = _evaluate(sdk, handle, points, scales)
        assert all(off["statuses"][p] == FAILED_STATUS for p in served), off
        assert sdk.rusticol_virtual_set_lane_rescue(handle, 1) == 0
        again = _evaluate(sdk, handle, points, scales)
        assert again["statuses"] == found["statuses"]
    finally:
        sdk.rusticol_virtual_free(handle)


@pytest.mark.loop_physics
def test_a_rescued_point_estimate_includes_the_reduction_direction_check(
    aag,  # noqa: F811
):
    """With checks that lack the reduction-direction check (one rotation),
    a point with rescued lanes is evaluated again with the rescaled one
    added (``reduction.rescued_checks``), on the call's backend (TIRust
    here), its value unchanged and its estimate the one of a call with
    those checks; the other points keep the call's checks; a call without
    checks runs nothing more."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    artifact, exact_points, exact_scales = aag
    runtime = Runtime.load(artifact, process="p")
    call = dict(
        mu_r2=exact_scales, poles="none", reduction="tirust", on_failure="return"
    )
    checked = runtime.evaluate(
        exact_points, stability_checks=VirtualStabilityChecks(1), **call
    )
    plain = runtime.evaluate(exact_points, **call)
    direct = runtime.evaluate(
        exact_points,
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
        **call,
    )
    served = [p for p, lane in enumerate(checked.diagnostics) if lane.lane_rescue]
    assert served
    for p, estimate in enumerate(checked.stability):
        names = {name for name, _budget in estimate.checks}
        if p in served:
            assert "reduction-rescaled" in names, names
            assert checked.diagnostics[p].lane_rescue is not None
            # Review B2: the detail names the rescue and the added check.
            assert "lane rescue:" in checked.diagnostics[p].detail
            assert "reduction-direction check" in checked.diagnostics[p].detail
            # Re-reduced on TIRust (not the default backend): the estimate
            # of the direct call with the same checks, bitwise.
            assert estimate == direct.stability[p]
        else:
            assert "reduction-rescaled" not in names, names
    assert checked.values == plain.values
    # Review m-a: with the call's own (reversed) reduction check nothing is
    # added, and the detail does not claim it.
    reversed_ = runtime.evaluate(
        exact_points, stability_checks=VirtualStabilityChecks(1, True), **call
    )
    for p, lane in enumerate(reversed_.diagnostics):
        if lane.lane_rescue is None:
            continue
        names = {name for name, _budget in reversed_.stability[p].checks}
        assert "reduction-rescaled" not in names, names
        assert "lane rescue:" in lane.detail
        assert "its estimate includes" not in lane.detail, lane.detail


def _estimates(node):
    """Every stability estimate in a result's (possibly nested) estimates."""

    if node is None:
        return []
    if hasattr(node, "checks") and hasattr(node, "relative"):
        return [node]
    if isinstance(node, (tuple, list)):
        return [e for item in node for e in _estimates(item)]
    if hasattr(node, "values") and callable(node.values):
        return [e for item in node.values() for e in _estimates(item)]
    return []


@pytest.mark.loop_physics
@pytest.mark.parametrize("route", ["helicities", "resolved", "correlated"])
def test_the_mapping_routes_record_the_rescue_and_extend_the_estimate(
    aag,  # noqa: F811
    route,
):
    """Review B1: on the selected (``helicities=``), resolved and correlated
    routes a point with rescued lanes carries its record
    (``VirtualLaneDiagnostic.lane_rescue``, merged over its reference and
    rotated evaluations) and, with checks that lack the reduction check,
    the estimate of every output with the rescaled reduction check added,
    bitwise a direct call's with the recommended checks; the other points
    keep the call's checks."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    artifact, exact_points, exact_scales = aag
    runtime = Runtime.load(artifact, process="p")
    common = dict(
        mu_r2=exact_scales, poles="none", reduction="tirust", on_failure="return"
    )
    calls = {
        "helicities": lambda **kw: runtime.evaluate(
            exact_points, helicities=["h0"], **common, **kw
        ),
        "resolved": lambda **kw: runtime.evaluate_resolved(
            exact_points, **common, **kw
        ),
        "correlated": lambda **kw: runtime.evaluate_correlated(
            exact_points, **common, **kw
        ),
    }
    call = calls[route]
    checked = call(stability_checks=VirtualStabilityChecks(1))
    direct = call(stability_checks=VirtualStabilityChecks(1, True, rescaled=True))
    served = [p for p, lane in enumerate(checked.diagnostics) if lane.lane_rescue]
    assert served, "the fixture points include rescued ones"
    for p, lane in enumerate(checked.diagnostics):
        if not lane.valid:
            continue
        found = _estimates(checked.stability[p])
        assert found, (route, p)
        names = [{name for name, _ in e.checks} for e in found]
        if p in served:
            assert all("reduction-rescaled" in n for n in names), (route, p, names)
            assert found == _estimates(direct.stability[p]), (route, p)
            assert "lane rescue:" in lane.detail
        else:
            assert all("reduction-rescaled" not in n for n in names), (route, p)


@pytest.fixture(scope="module")
def uux_ttxgg(loop_native, tmp_path_factory):
    """u u~ > t t~ g g (Born-interfered; explicit TIRust refuses lanes on
    about a quarter of the flat points) from its shipped card, and sixteen
    flat points (seed 1), as ``test_loop_v2_best_effort.heavy``."""

    import shutil

    from pyamplicol.cli.flat_points import draw_flat_points
    from pyamplicol.cli.main import main
    from tests.unit.test_examples import EXAMPLES

    if "tirust" not in __import__("json").loads(loop_native.reduction_backends()):
        pytest.skip("the staged native extension is built without TIRust")
    root = tmp_path_factory.mktemp("uux_ttxgg")
    for item in EXAMPLES.iterdir():
        if item.name.endswith(".toml"):
            shutil.copy(item, root / item.name)
    shutil.copytree(EXAMPLES / "data", root / "data")
    artifact = root / "artifact"
    card = root / "loop_uux_ttxgg_generate.toml"
    assert main(
        [str(card), "--progress", "off", "--set", f"generation.output={artifact}"]
    ) in (0, None)
    flat = draw_flat_points(artifact, process=None, points=16, seed=1, sqrt_s=None)
    return artifact, flat.as_inputs(exact=True), flat.mu_r2()


@pytest.mark.loop_physics
@pytest.mark.parametrize("route", ["helicities", "resolved", "correlated"])
def test_the_born_interfered_routes_record_the_rescue(uux_ttxgg, route):
    """Review m-c: the Born-interfered mapping routes (``StoredColumns``)
    carry a rescued point's record and its extended estimate, bitwise a
    direct call's with the recommended checks (the loop-induced routes:
    ``test_the_mapping_routes_record_the_rescue_and_extend_the_estimate``)."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    artifact, points, scales = uux_ttxgg
    runtime = Runtime.load(artifact)
    common = dict(mu_r2=scales, poles="none", reduction="tirust", on_failure="return")
    calls = {
        "helicities": lambda **kw: runtime.evaluate(
            points, helicities=[f"h{i}" for i in range(8)], **common, **kw
        ),
        "resolved": lambda **kw: runtime.evaluate_resolved(points, **common, **kw),
        "correlated": lambda **kw: runtime.evaluate_correlated(points, **common, **kw),
    }
    call = calls[route]
    checked = call(stability_checks=VirtualStabilityChecks(1))
    direct = call(stability_checks=VirtualStabilityChecks(1, True, rescaled=True))
    plain = call(stability_checks=VirtualStabilityChecks(1), lane_rescue=False)
    served = [p for p, lane in enumerate(checked.diagnostics) if lane.lane_rescue]
    # Exactly the points that fail without the rescue carry its record (the
    # selected rows may refuse no lane: the resolved and correlated routes,
    # over every row, do).
    assert served == [p for p, lane in enumerate(plain.diagnostics) if not lane.valid]
    if route != "helicities":
        assert served, "the fixture points include rescued ones"
    for p in served:
        found = _estimates(checked.stability[p])
        assert found and found == _estimates(direct.stability[p]), (route, p)
        assert all(
            "reduction-rescaled" in {name for name, _ in e.checks} for e in found
        )
        assert "lane rescue:" in checked.diagnostics[p].detail
