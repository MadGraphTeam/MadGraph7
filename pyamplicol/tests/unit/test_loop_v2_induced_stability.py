# SPDX-License-Identifier: 0BSD
"""Loop-induced stability estimates on NinjaRS (M-LIT LIT-STAB-N, m-lit PLAN
§2.a.8 with DECISIONS 2026-11-06 ~09:00 (1)).

The estimate of a loop-induced |A|^2 is E = SAFETY x (max over rotations +
reduction re-test) + FLOOR_UNITS x u x S, with budgets quadratic in the
colour-metric norm (``induced.quadratic_budget``, the rotation budget after
removing each helicity row's little-group phase) and S the |A|^2-level
product of the per-row pre-reduction term floors (``induced.scale``). On
the two calibration rows, g g > h h (large cancellations) and g g > z
(Landau-Yang: every family integrates to zero by itself), on the stored
runtime's route (native point inputs) against Float(256):

* single-point estimates bracket the true error within the user's x10
  margin, per sector and Laurent coefficient, in binary64 and
  double-double (``ESTIMATE_MARGIN``, DESIGN §10.4);
* g g > z is served as "zero within its estimate" (RULED Q13: no
  Landau-Yang declaration): |A|^2 <= 10 E in binary64 and double-double;
  its E / S is the floor's 16 u, so the opt-in rescue metric "scale" (E <=
  target x S, PLAN §2.a.8, ``VirtualRescueLevel.metric``) accepts it at the
  first level, while its relative estimate exceeds every target and the
  default rescue keeps it at the last level with the target not met (both
  ladders end to end on the public runtime);
* the double-double estimate shrinks with the carrier's unit roundoff (the
  rotation budget compares the rotated amplitudes after a little-group
  phase normalised in the carrier, ``induced._modulus``);
* the stored route's rotation check samples on rows without counterterm
  kets (g g > z, g g > h a: ``induced.rotation_budget``), so the Furry
  zero g g > h a is within its estimate there too;
* the committed light calibration (``tools/loop_v2_stability_calibration.
  py``, ``tests/fixtures/loop/v2/stability_calibration_li.json``) records
  no point beyond the x10 margin in any check configuration;
* the calibration's exact loop-induced points are on shell and momentum
  conserving.
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from .test_loop_v2_induced import _fixture, _generate, _Rows

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = (
    ROOT / "tests" / "fixtures" / "loop" / "v2" / "stability_calibration_li.json"
)
# USER DECISION (2026-10-23; DESIGN §10.4): a single-point test of a
# stability estimate E holds the true error to at most 10 E, for the whole
# estimate and per Laurent coefficient.
ESTIMATE_MARGIN = 10
CALIBRATION_ROWS = ("li_gg_hh", "li_gg_z")
TRUTH_DIGITS = 57


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li-stab"))


def _evaluate(li, name, precision, index, checks):
    return li.native_value(
        name, precision, inputs="native", index=index, stability=checks
    )


def _errors(result, truth):
    """|value - truth| per sector and Laurent coefficient (Decimal)."""

    exact = truth.components["virtual"]
    return {
        sector: [
            abs(Decimal(str(value)) - Decimal(str(reference)))
            for value, reference in zip(
                laurent.coefficients, exact[sector].coefficients, strict=True
            )
        ]
        for sector, laurent in result.components["virtual"].items()
    }


# ------------------------------------------------------------- single points


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", ["double", "double_double"])
@pytest.mark.parametrize("name", CALIBRATION_ROWS)
def test_single_point_estimates_bracket_the_error(li, name, carrier):
    """On every fixture point of the calibration rows the true error of the
    recommended estimate (rescaled re-test + one generic rotation) is at
    most ten times E, per sector and Laurent coefficient (the poles of a
    loop-induced |A|^2 vanish, so only the absolute comparison holds)."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    precision = getattr(Precision, carrier)()
    for index in range(len(_fixture(name)["points"])):
        result = _evaluate(li, name, precision, index, RECOMMENDED_CHECKS)
        truth = _evaluate(li, name, Precision.arbitrary(TRUTH_DIGITS), index, None)
        for sector, errors in _errors(result, truth).items():
            estimate = result.stability.absolute[sector].coefficients
            for k, error in enumerate(errors):
                assert float(error) <= ESTIMATE_MARGIN * estimate[k], (
                    index,
                    sector,
                    k,
                    float(error),
                    estimate[k],
                )


@pytest.mark.loop_physics
def test_double_double_estimate_shrinks_with_the_unit_roundoff(li):
    """g g > h h: the double-double estimate of every Laurent coefficient is
    at most ten times the binary64 one scaled by u_DD / u_64 (PLAN §2.a.8:
    in double-double the bound shrinks by ~1e-16). A little-group phase
    normalised in binary64 left every rotated double-double value off by
    ~1e-16 relative: E of the finite part was 1e10 times too large."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    double, extended = Precision.double(), Precision.double_double()
    ratio = extended.unit_roundoff / double.unit_roundoff
    for index in range(len(_fixture("li_gg_hh")["points"])):
        b64 = _evaluate(li, "li_gg_hh", double, index, RECOMMENDED_CHECKS)
        dd = _evaluate(li, "li_gg_hh", extended, index, RECOMMENDED_CHECKS)
        for sector, budget in dd.stability.absolute.items():
            reference = b64.stability.absolute[sector].coefficients
            for k, value in enumerate(budget.coefficients):
                assert value <= ESTIMATE_MARGIN * ratio * reference[k], (index, k)


# ------------------------------------------------------------- integrated zero


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", ["double", "double_double"])
def test_g_g_to_z_is_zero_within_its_estimate(li, carrier):
    """g g > z (Landau-Yang) is evaluated numerically, never declared zero
    (RULED Q13): on every fixture point |A|^2 lies within ten times E per
    Laurent coefficient. E rests on the term floor (E / S is the floor's
    FLOOR_UNITS x u): E / S meets the default rescue target, so the opt-in
    rescue metric "scale" (E <= target x S) accepts it, whereas the
    relative estimate (E against the value, itself rounding) exceeds it by
    far."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import (
        FLOOR_UNITS,
        RECOMMENDED_CHECKS,
        RESCUE_TARGET,
    )

    precision = getattr(Precision, carrier)()
    floor = FLOOR_UNITS * precision.unit_roundoff
    for index in range(len(_fixture("li_gg_z")["points"])):
        result = _evaluate(li, "li_gg_z", precision, index, RECOMMENDED_CHECKS)
        estimate = result.stability
        for sector, laurent in result.components["virtual"].items():
            absolute = estimate.absolute[sector].coefficients
            scale = estimate.scale[sector].coefficients
            for k, value in enumerate(laurent.coefficients):
                assert abs(float(value)) <= ESTIMATE_MARGIN * absolute[k], (index, k)
                assert floor <= absolute[k] / scale[k] <= ESTIMATE_MARGIN * floor
                assert absolute[k] / scale[k] <= RESCUE_TARGET
        assert estimate.relative > 1 / RESCUE_TARGET


@pytest.mark.loop_physics
def test_the_rescue_of_g_g_to_z_terminates_under_either_metric(li, tmp_path):
    """Under the default (relative) rescue metric an integrated zero can
    never meet its target: g g > z climbs the loop-induced default stack
    (TIRust binary64 -> NinjaRS binary64 -> double-double -> arbitrary(57),
    m-lit F3 (b): ``VirtualRescuePolicy()`` on a loop-induced "auto" call)
    and is kept at the last level with the target marked as not met (PLAN
    §2.a.8: the rescue terminates, at the cost of every level). The opt-in
    "scale" metric (E <= target x S, ``VirtualRescueLevel.metric``) on the
    same ladder accepts it at the first level, its E / S
    the floor's FLOOR_UNITS x u, its value the binary64 one and within its
    absolute estimate (Q13: zero within the estimate)."""

    from dataclasses import replace

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescuePolicy
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.rescue import loop_induced_stack
    from pyamplicol.loops.v2.stability import FLOOR_UNITS

    output = _generate(tmp_path, (("gg_z", "g g > z"),), _fixture("li_gg_z"))
    point, scale, parameters = li.point("li_gg_z")
    runtime = Runtime.load(output, process="gg_z", model_parameters=parameters)
    # The stack the default policy runs on this loop-induced "auto" call
    # (TIRust is built in; without it the default ladder itself).
    from pyamplicol.loops.v2.reduction import native_backends

    policy = (
        loop_induced_stack()
        if "tirust" in native_backends(runtime._backend._native)
        else VirtualRescuePolicy()
    )
    result = runtime.evaluate(
        [point], mu_r2=[str(scale)], stability=VirtualRescuePolicy()
    )
    (record,) = result.rescue
    assert record.level == len(policy.levels) - 1 and not record.met
    assert [attempt.arithmetic for attempt in record.attempts] == [
        level.arithmetic for level in policy.levels
    ]
    assert all(
        attempt.estimate > policy.levels[0].target for attempt in record.attempts
    )
    assert result.diagnostics[0].status == "rescued"
    assert not result.diagnostics[0].accuracy_established
    assert result.contribution == "loop-induced"

    ladder = VirtualRescuePolicy(
        tuple(replace(level, metric="scale") for level in policy.levels)
    )
    scaled = runtime.evaluate([point], mu_r2=[str(scale)], stability=ladder)
    (record,) = scaled.rescue
    assert record.level == 0 and record.met and len(record.attempts) == 1
    (estimate,) = scaled.stability
    floor = FLOOR_UNITS * Precision.double().unit_roundoff
    assert floor <= estimate.scale <= ESTIMATE_MARGIN * floor
    assert record.attempts[0].estimate == estimate.scale <= ladder.levels[0].target
    (value,) = scaled.values[0]
    (error,) = estimate.absolute
    assert abs(value.finite) <= ESTIMATE_MARGIN * error.finite
    unchecked = runtime.evaluate([point], mu_r2=[str(scale)])
    assert scaled.values == unchecked.values
    assert scaled.contribution == "loop-induced"


# ------------------------------------------------------------ rotation check


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", ["double", "double_double"])
def test_the_native_rotation_check_samples_without_counterterms(
    li, carrier, monkeypatch
):
    """g g > z has no counterterm kets, so the executor forms none: on the
    stored runtime's route its rotation check still compares the rotated
    amplitudes after the row phase (``induced.rotation_budget``), never the
    raw lanes a native amplitude-mode point does not hold (an empty, hence
    zero, budget that left the check silently measuring nothing)."""

    from pyamplicol.loops.v2 import induced, stability
    from pyamplicol.loops.v2.assemble import Precision, _NativePoint
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    # The Python assembly (the oracle; the binary64 estimate's native
    # implementation equals it bitwise, test_loop_v2_sdk_stability.py).
    monkeypatch.setattr(stability, "NATIVE_BINARY64", False)
    seen = []
    original = _NativePoint.rotation_budget

    def spy(self, rotated):
        budget = original(self, rotated)
        seen.append(
            (self.native_counterterms, budget, induced.rotation_budget(self, rotated))
        )
        return budget

    monkeypatch.setattr(_NativePoint, "rotation_budget", spy)
    precision = getattr(Precision, carrier)()
    result = _evaluate(li, "li_gg_z", precision, 0, RECOMMENDED_CHECKS)
    assert seen and not any(counterterms for counterterms, _, _ in seen)
    for _, budget, reference in seen:
        assert budget == reference
        assert any(x > 0 for b in budget.values() for x in b.coefficients)
    assert result.stability.checks["rotation-0"] > 0


@pytest.mark.loop_physics
def test_g_g_to_h_a_is_zero_within_its_estimate_on_the_stored_route(li, tmp_path):
    """g g > h a (Furry; no counterterms) on the stored runtime's route: its
    term floor is at rounding, so the rotation budget carries the estimate.
    Through the executor with native inputs and through the public runtime
    |A|^2 <= 10 E per Laurent coefficient (no correct digit claimed for a
    value that is rounding), in binary64 and double-double, the rotation
    check sampling."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    for precision in (Precision.double(), Precision.double_double()):
        result = _evaluate(li, "li_gg_ha", precision, 0, RECOMMENDED_CHECKS)
        assert result.stability.checks["rotation-0"] > 0
        for sector, laurent in result.components["virtual"].items():
            absolute = result.stability.absolute[sector].coefficients
            for k, value in enumerate(laurent.coefficients):
                assert abs(float(value)) <= ESTIMATE_MARGIN * absolute[k], (k, value)
    fixture = _fixture("li_gg_ha")
    output = _generate(tmp_path, (("gg_ha", "g g > h a"),), fixture)
    point, scale, parameters = li.point("li_gg_ha")
    runtime = Runtime.load(output, process="gg_ha", model_parameters=parameters)
    for arithmetic in ("double", "double-double"):
        policy = VirtualRescuePolicy((VirtualRescueLevel(arithmetic),))
        result = runtime.evaluate([point], mu_r2=[str(scale)], stability=policy)
        (estimate,) = result.stability
        assert dict(estimate.checks)["rotation-0"] > 0, (arithmetic, estimate)
        assert ESTIMATE_MARGIN * estimate.finite >= 1, (arithmetic, estimate)
        for value, error in zip(result.values[0], estimate.absolute, strict=True):
            assert abs(value.finite) <= ESTIMATE_MARGIN * error.finite, arithmetic


# ---------------------------------------------------------------- calibration


@pytest.mark.loop_fast
def test_loop_induced_calibration_has_no_miss_beyond_the_margin():
    """The recorded light calibration (``tools/loop_v2_stability_calibration.
    py report``, 2000 binary64 and 400 double-double points per row against
    Float(256)): no point fails, no check configuration has a point whose
    absolute error exceeds ten times its estimate (per sector and Laurent
    coefficient), and the recommended configuration (the rescaled re-test
    with a rotation) misses none at all; g g > z stays within its estimate
    with E / S at the floor. Every requested check sampled: no check's
    budget is identically zero over a row's points (a check that measures
    nothing, as the stored route's rotation check of a row without
    counterterm kets once did, passes every margin unnoticed)."""

    from pyamplicol.loops.v2.stability import RESCUE_TARGET

    data = json.loads(CALIBRATION.read_text())
    for name in CALIBRATION_ROWS:
        for carrier, points in (("double", 2000), ("double-double", 400)):
            record = data["processes"][name][carrier]
            assert record["points"] >= points and record["failed"] == 0, name
            assert record["failed_truth"] == 0
            for check, sampled in record["checks"].items():
                assert sampled["check_over_error"]["max"] > 0, (name, carrier, check)
            for configuration, rates in record["configurations"].items():
                absolute = rates["absolute"]
                assert absolute["beyond_margin"] == 0, (name, carrier, configuration)
                if (
                    "reduction-rescaled" in configuration
                    and "rotation" in configuration
                ):
                    assert absolute["false_negatives"] == 0, (name, configuration)
                if name == "li_gg_z":
                    assert absolute["value_over_estimate"]["max"] <= 1
                    assert absolute["estimate_over_scale"]["max"] <= RESCUE_TARGET


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", CALIBRATION_ROWS)
def test_calibration_points_are_exact(name):
    """``loop_induced_point``: exact rationals on the fixture's mass shells,
    momentum conserving, deterministic per index."""

    from tools.loop_v2_stability_calibration import LOOP_INDUCED, loop_induced_point

    data = LOOP_INDUCED[name].data()
    masses = [Fraction(m) for m in data["point_construction"]["masses"]]
    for index in range(20):
        point = loop_induced_point(name, index)
        assert point == loop_induced_point(name, index)
        for momentum, mass in zip(point, masses, strict=True):
            energy, *space = momentum
            assert energy > 0
            assert energy * energy - sum(x * x for x in space) == mass * mass
        for mu in range(4):
            assert point[0][mu] + point[1][mu] == sum(p[mu] for p in point[2:])
