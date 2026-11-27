# SPDX-License-Identifier: 0BSD
"""Near-forward accuracy through the rotation-based stability checks and the
rescue (m-lit lit-ninja-fwd; USER 2026-11-10 ~18:54 and ~19:20, DECISIONS
2026-11-10 18:54 and 19:20).

At nearly degenerate 2 -> 2 (sub-)box kinematics (near-forward, near-backward,
near-threshold) the NinjaRS binary64 fast engine and oneloop's binary64 D0 can
return values far off with no estimate when no stability check is requested
(the 1.0.2 behaviour, documented). Accuracy there comes from the rotation-based
checks the user requests (``RECOMMENDED_CHECKS``) and the rescue to
double-double / arbitrary precision, whose rotations act on the PROMOTED and
IMPROVED point:

* g g > z z at theta 1e-3 / 1e-4 (``tests/fixtures/loop/near_forward_zz.json``,
  the CM-frame texts CR and the rotated + boosted texts QR): binary64 "auto"
  with ``RECOMMENDED_CHECKS`` returns each summed value within 1e-5 of the
  consistent truth or flags it (estimate above 1e-5); the default rescue
  returns each within 1e-5 or fails it typed. QR theta 1e-4 is the case MadLoop's
  chain alone left 5.3 off at double-double, met (ORIG refuses the frame, PSMC
  leaves its conservation residual, which NinjaRS amplifies ~1/Gram and no
  rotation sees): the default chain now finishes with the projection
  (``kinematics_improvement::FINISH``) and the level is exact.
* ``pyamplicol check``: every evaluation of a rung above binary64, with or
  without its ``RECOMMENDED_CHECKS``, takes the point improved in that
  carrier (scripted, ``loop_fast``).

The rescue level and the resolved route in double-double are tested on
g g > g g in ``test_kinematics_improvement.py``.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parents[1] / "fixtures" / "loop" / "near_forward_zz.json"
LIMIT = 1e-5


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _points(sets=("CR", "QR"), thetas=("1e-3", "1e-4")):
    found = _fixture()
    labels, points, truths = [], [], []
    for name in sets:
        for theta in thetas:
            entry = found["sets"][name][theta]
            labels.append(f"{name} {theta}")
            points.append(entry["point"])
            truths.append(entry["truth"])
    return labels, points, [found["mu_r2"]] * len(points), truths


def _recommended():
    """``RECOMMENDED_CHECKS`` through the public API: one rotation and the
    rescaled reduction-direction check."""

    from pyamplicol.api import VirtualStabilityChecks

    return VirtualStabilityChecks(1, True, rescaled=True)


def _error(values, truth: str) -> float:
    found = sum(Decimal(str(sector.finite)) for sector in values)
    exact = Decimal(truth)
    return float(abs(found - exact) / abs(exact))


@pytest.fixture(scope="module")
def zz(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > z z (LI-21) through the public Generator (process ``p``), in a
    fresh interpreter (``test_loop_v2_fallback_guard._generated``)."""

    from .test_loop_v2_fallback_guard import _generated

    return _generated(tmp_path_factory.mktemp("zz"), "li_gg_zz", "g g > z z")


@pytest.mark.loop_physics
def test_recommended_checks_flag_the_near_forward_binary64_values(zz):
    """Binary64 "auto" with ``RECOMMENDED_CHECKS`` (DECISIONS 2026-11-10
    18:54 (a)): every returned summed value is within 1e-5 of the consistent
    truth or its estimate flags it (above 1e-5). Measured
    (``tmp/lit-ninja-fwd/rc/out/zz-{CR,QR}.json``): the values are 2.6e6 /
    2.0e18 (CR) and 3.2e6 / 2.4e18 (QR) off at theta 1e-3 / 1e-4, flagged at
    0.47 / 7.6e-4 and 0.46 / 5.6e-3: the estimates understate the errors by
    up to 2.6e21, but they flag them."""

    from pyamplicol import Runtime

    labels, points, scales, truths = _points()
    runtime = Runtime.load(zz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        poles="none",
        on_failure="return",
        stability_checks=_recommended(),
    )
    for p, label in enumerate(labels):
        if found.values[p] is None:
            assert found.diagnostics[p].status == "failed", label
            continue
        error = _error(found.values[p], truths[p])
        estimate = found.stability[p].relative
        assert error <= LIMIT or not estimate <= LIMIT, (label, error, estimate)


@pytest.mark.loop_physics
def test_the_default_rescue_serves_the_near_forward_points(zz):
    """The default rescue (``VirtualRescuePolicy()``: TIRust binary64 ->
    NinjaRS binary64 -> double-double -> 57 digits, each with a rotation and
    the rescaled check) returns every point within 1e-5 of the consistent
    truth or fails it typed (DECISIONS 2026-11-10 18:54 (a)). At QR theta
    1e-4 (beams off the z axis) the double-double level's point is produced
    by the default chain's finish (ORIG refuses, code 200; PSMC's point
    misses exactness; the projection makes it exact; USER ~19:20): the value
    MadLoop's chain alone left 5.3 off, met at estimate 1.6e-7, is within
    1e-5 (DECISIONS 2026-11-10 19:20 (F1))."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescuePolicy

    labels, points, scales, truths = _points()
    runtime = Runtime.load(zz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        poles="none",
        on_failure="return",
        stability=VirtualRescuePolicy(),
    )
    for p, label in enumerate(labels):
        record = found.rescue[p]
        if found.values[p] is None:
            assert found.diagnostics[p].status == "failed", label
            continue
        assert _error(found.values[p], truths[p]) <= LIMIT, (label, record)
        if label.startswith("QR"):
            # Every level above binary64 improves the original texts in its
            # carrier with the default chain, finished by the projection.
            for attempt in record.attempts:
                if attempt.arithmetic == "double":
                    assert attempt.kinematics is None, (label, attempt)
                    continue
                kinematics = attempt.kinematics
                assert kinematics.attempts == (
                    ("z-shift", 200),
                    ("rescaling", 0),
                    ("projection", 0),
                ), (label, kinematics)
                assert kinematics.produced == "projection" and kinematics.exact
                assert kinematics.warning is None


@pytest.mark.loop_physics
def test_the_default_chain_is_frame_independent_at_double_double(zz):
    """``"auto"`` double-double with ``RECOMMENDED_CHECKS`` on the QR texts
    (beams off the z axis), the summed route: under the default chain each
    value is within 1e-5 of the consistent truth (the CM-frame texts CR,
    which ORIG makes exact, likewise), never a value far off with an
    estimate within 1e-5 (before the finish: 8.1e-8 / 5.3 off at theta 1e-3
    / 1e-4, the latter at estimate 1.6e-7)."""

    from pyamplicol import Runtime

    labels, points, scales, truths = _points()
    runtime = Runtime.load(zz, process="p")
    found = runtime.evaluate(
        points,
        mu_r2=scales,
        arithmetic="double-double",
        poles="none",
        on_failure="return",
        stability_checks=_recommended(),
    )
    for p, label in enumerate(labels):
        kinematics = found.diagnostics[p].kinematics
        assert kinematics.exact, (label, kinematics)
        expected = "projection" if label.startswith("QR") else "z-shift"
        assert kinematics.produced == expected, (label, kinematics)
        if found.values[p] is None:
            assert found.diagnostics[p].status == "failed", label
            continue
        error = _error(found.values[p], truths[p])
        estimate = found.stability[p].relative
        assert error <= LIMIT, (label, error, estimate)


# ------------------------------------------------- pyamplicol check's rungs


@pytest.mark.loop_fast
def test_the_check_rungs_evaluate_the_improved_point(monkeypatch):
    """``pyamplicol check`` (DECISIONS 2026-11-10 18:54 (b)): the ladder's
    double-double evaluation, the rung's own double-double estimate
    (``RECOMMENDED_CHECKS``: the rotation) and its arbitrary-precision re-run
    each take the point improved in that carrier (``checks._improved``, the
    public chain), never the binary64 point; a binary64 rung takes the point
    as given (the improvement is off in binary64 by default). Scripted: a row
    double-double cannot decide (1/eps off by 1e-11, estimate 1e-11) is
    re-run at 34 digits, where it is decided."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks, induced, plan_export, ward
    from pyamplicol.loops.v2.assemble import Precision

    sector = (("QED", 4),)
    calls = []

    def laurent(*values):
        return NS(coefficients=list(values))

    def improved(_executor, point, _scale, carrier, settings):
        assert settings == "settings"
        if carrier.arithmetic == "double":
            return point, None
        return ("improved", carrier.arithmetic, carrier.digits, point), "projection"

    def evaluate(_executor, point, _scale, carrier, checks_):
        calls.append((carrier.arithmetic, carrier.digits, checks_ is not None, point))
        if carrier.arithmetic == "double":
            single, error = 2.0, 1e-3
        elif carrier.arithmetic == "double-double":
            single, error = 2.0 * (1 + 1e-11), (1e-11 if checks_ is not None else None)
        else:
            single, error = 2.0 * (1 + 1e-14), 1e-15
        stability = None
        if error is not None:
            stability = NS(
                relative=error,
                absolute={sector: laurent(0.0, error, 1e-40)},
                scale={sector: laurent(1.0, 1.0, 1.0)},
            )
        return NS(
            stability=stability,
            components={"virtual": {sector: laurent(0.0, single, -3.0)}},
            ir_poles={sector: (2.0, -3.0)},
        )

    monkeypatch.setattr(plan_export, "NativeExecutor", lambda *a, **k: None)
    monkeypatch.setattr(checks, "_evaluate", evaluate)
    monkeypatch.setattr(checks, "_improved", improved)
    monkeypatch.setattr(ward, "gauge_legs", lambda _target: ())
    monkeypatch.setattr(induced, "goldstone_applicability", lambda _t: "n/a")
    target = NS(unpredicted_sectors=lambda: {})
    carriers = (Precision.double(), Precision.double_double())
    rung = checks.Rung(lambda index, carrier: (("lifted", carrier.digits), 1))
    rows = checks.born_interfered_rows(
        target, [("given", 1)], carriers, rung, kinematics="settings"
    )
    assert [(a, d, c) for a, d, c, _p in calls] == [
        ("double", None, True),
        ("double-double", None, False),
        ("double-double", None, True),
        ("arbitrary", 34, True),
    ]
    for arithmetic, digits, _checked, point in calls:
        if arithmetic == "double":
            assert point == "given"
        elif arithmetic == "double-double":
            assert point == ("improved", "double-double", None, "given"), point
        else:
            assert point == ("improved", "arbitrary", digits, ("lifted", digits))
    row = {r.subject: r for r in rows if r.check == "poles"}["QED^4 1/eps"]
    assert (row.verdict, row.carrier, row.kinematics) == (
        checks.PASS,
        "arbitrary",
        "projection",
    ), row
