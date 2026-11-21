# SPDX-License-Identifier: 0BSD
"""The default stability checks per backend (USER 2026-11-19: the rescaled
reduction-direction check "for TIRust as part of the default stability
stack ..., so only for TIRust"): inside ``reduction.default_checks`` a point
TIRust serves runs the call's checks with the rescaled reduction check
added, a point NinjaRS serves the call's checks unchanged; outside it (every
check set a caller gives) nothing changes."""

from __future__ import annotations

import pytest


def test_the_default_set_adds_the_reduction_check_on_tirust_only():
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import backend_checks, default_checks

    one = VirtualStabilityChecks(1)
    recommended = VirtualStabilityChecks(1, True, rescaled=True)
    reversed_ = VirtualStabilityChecks(1, True)
    # Explicit checks: never changed, on any backend.
    for backend in ("tirust", "ninjars", None):
        assert backend_checks(one, backend) is one
    with default_checks():
        assert backend_checks(one, "tirust") == recommended
        assert backend_checks(one, "ninjars") is one
        # A set with a reduction check, or none at all, gains nothing.
        assert backend_checks(reversed_, "tirust") is reversed_
        assert backend_checks(recommended, "tirust") is recommended
        assert backend_checks(None, "tirust") is None
        assert backend_checks(VirtualStabilityChecks(), "tirust") == (
            VirtualStabilityChecks()
        )
    assert backend_checks(one, "tirust") is one


def test_every_default_rescue_level_already_runs_the_reduction_check():
    """The default stacks (``VirtualRescuePolicy()`` and the loop-induced
    one) run one rotation and the rescaled reduction check at every level,
    on both backends: the default set changes nothing there."""

    from pyamplicol.api.virtual_results import VirtualRescuePolicy
    from pyamplicol.loops.v2.rescue import loop_induced_stack

    for policy in (VirtualRescuePolicy(), loop_induced_stack()):
        for level in policy.levels:
            checks = level.checks
            assert len(checks.rotations) == 1
            assert checks.reduction_direction and checks.rescaled, level


from tests.unit.test_loop_v2_best_effort import aag  # noqa: E402,F401  (fixture)


@pytest.mark.loop_physics
def test_default_checks_follow_the_backend_that_serves_each_point(
    aag,  # noqa: F811
):
    """g g > a a g (loop-induced: ``"auto"`` is TIRust with the per-point
    NinjaRS fallback), lane rescue off so that TIRust's refusals go to the
    fallback: inside ``default_checks`` the one-rotation call's TIRust
    points carry the estimate of the recommended checks (bitwise a direct
    call's), its NinjaRS fallback points the one-rotation estimate (bitwise
    an explicit NinjaRS call's guard-free estimate set: rotation only); an
    explicit ``reduction="ninjars"`` call is bitwise the plain one; values
    never move."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import default_checks

    artifact, points, scales = aag
    runtime = Runtime.load(artifact, process="p")
    one = VirtualStabilityChecks(1)
    recommended = VirtualStabilityChecks(1, True, rescaled=True)
    call = dict(mu_r2=scales, poles="none", on_failure="return", lane_rescue=False)

    def names(estimate):
        return {name for name, _budget in estimate.checks}

    # Explicit TIRust: every point TIRust's, so the recommended estimate.
    plain = runtime.evaluate(points, stability_checks=one, reduction="tirust", **call)
    direct = runtime.evaluate(
        points, stability_checks=recommended, reduction="tirust", **call
    )
    with default_checks():
        found = runtime.evaluate(
            points, stability_checks=one, reduction="tirust", **call
        )
    assert found.values == plain.values == direct.values
    assert found.statuses == direct.statuses
    for p, estimate in enumerate(found.stability):
        if estimate is None:
            continue
        assert estimate == direct.stability[p]
        assert "reduction-rescaled" in names(estimate)
    # Explicit NinjaRS: unchanged, bitwise.
    plain = runtime.evaluate(points, stability_checks=one, reduction="ninjars", **call)
    with default_checks():
        found = runtime.evaluate(
            points, stability_checks=one, reduction="ninjars", **call
        )
    assert found.values == plain.values
    assert found.stability == plain.stability
    # "auto": per point, the serving backend's set.
    plain = runtime.evaluate(points, stability_checks=one, **call)
    with default_checks():
        found = runtime.evaluate(points, stability_checks=one, **call)
    assert found.values == plain.values
    fallback = [
        p
        for p, lane in enumerate(found.diagnostics)
        if lane.reduction is not None and lane.reduction.effective == "ninjars"
    ]
    served = [
        p
        for p, estimate in enumerate(found.stability)
        if estimate is not None and p not in fallback
    ]
    assert fallback, "the fixture points include TIRust refusals"
    assert served, "and points TIRust serves"
    for p in fallback:
        if found.stability[p] is not None:
            assert found.stability[p] == plain.stability[p]
            assert "reduction-rescaled" not in names(found.stability[p])
    for p in served:
        assert "reduction-rescaled" in names(found.stability[p])
        assert found.stability[p] == direct.stability[p]
