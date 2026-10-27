# SPDX-License-Identifier: 0BSD
"""Absolute thresholds of the binary64 fast engine (genericity audit 2,
G2-M3; package onshell-threshold).

On the pinned NinjaRS 41a4c00 the binary64 fast engine compares its inputs
with absolute constants in the model's units: ``OneLoopNativeMasters`` sets
squared masses and invariants below 1e-9 to zero and an invariant within
1e-9 of a squared mass onto it, and the engine skips massless bubbles and
builds light-like cut bases below its ``infrared_eps`` (1e-9). A keV-scale
mass in GeV (m^2 = 1e-12) thus became massless in binary64 alone, and the
result depended on the units: e+ e- > t t~ [QCD] with MT = 1e-6 was 107 % off
MPFR in binary64 at the point below, with a reduction-direction estimate of
1e-14 (slice native at 11542fd5). The reducer now hands every lane whose
inputs reach one of these thresholds to the generic binary64 engine
(``v2_reduce`` module docs) and keeps the fast engine's bits for every other
lane; the executor's report counts the routed lanes.

The tests evaluate loop_sm processes with a light top (MT = 1e-6, m^2 = 1e-12)
in the production path (native executor, native inputs):

* e+ e- > t t~ [QCD] (the light top in every loop line and both legs) and
  u u~ > g g [QCD] (a closed light-top loop): binary64 equals MPFR within
  1e-12, or within its own tenfold estimate (DESIGN §10.4), with lanes
  routed to the exact engine; DoubleFloat equals MPFR within 1e-24; at
  loop_sm's top mass no lane is routed (the fast engine's bits);
* the dimensionless virtual is unit independent: GeV and MeV (every momentum
  and dimensionful parameter times 1e3, its square and the scale times 1e6)
  agree to binary64 accuracy, the light top routed in GeV (m^2 = 1e-12) and
  on the fast engine in MeV (1e-6);
* where oneloopmaster's binary64 masters cannot evaluate the light lanes
  (g g > g g: a box of four light lines, D0 not finite), the point fails with
  the master's reason instead of taking the massless integral, and
  DoubleFloat evaluates it (to its own conditioning there, 2.1e-18).
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import FailedPoints, NativeExecutor
from pyamplicol.loops.v2.stability import Budget, StabilityChecks, deviation

ROUTED = "binary64 lanes on the exact engine"
FIXED = {"MB": "0", "WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
LIGHT_MT = "0.000001"
BINARY64 = 1e-12
DOUBLE_DOUBLE = 1e-24
ESTIMATE_CAP = 1e-9
# USER DECISION (2026-10-23; DESIGN §10.4): a single-point test of a
# stability estimate E holds the true error to at most 10 E.
ESTIMATE_MARGIN = 10
SCALE = "250000"
MEV = 1000

# Incoming legs along the beam, outgoing at E = 500 GeV with |p| = 500
# (massless gluons; the light top's |p| rounds to E, and the executor binds
# each leg's invariant to its exact squared mass).
POINT = [
    ["500.0", "0.0", "0.0", "500.0"],
    ["500.0", "0.0", "0.0", "-500.0"],
    ["500.0", "300.0", "240.0", "320.0"],
    ["500.0", "-300.0", "-240.0", "-320.0"],
]
# The same energy with MT = 173: |p| = sqrt(500^2 - 173^2), rounded.
HEAVY_POINT = [
    ["500.0", "0.0", "0.0", "500.0"],
    ["500.0", "0.0", "0.0", "-500.0"],
    ["500.0", "281.47035367867784", "225.17628294294227", "300.23504392392306"],
    ["500.0", "-281.47035367867784", "-225.17628294294227", "-300.23504392392306"],
]

# name: process, Born orders, loop orders, the model parameters that carry a
# unit besides MT (value, mass dimension) for the unit test.
PROCESSES = {
    "epem_ttx": (
        "e+ e- > t t~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        {"MZ": ("91.188", 1), "Gf": ("1.16639e-05", -2)},
    ),
    "uux_gg": ("u u~ > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, {}),
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, {}),
}


@pytest.fixture(scope="module")
def executors(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, NativeExecutor] = {}

    def build(name: str) -> NativeExecutor:
        if name not in built:
            process, born, loop, _units = PROCESSES[name]
            slice_ = LoopSlice.generate(
                compiled,
                SliceRequest(process, born, loop, fixed_parameters=FIXED),
                loop_native,
            )
            built[name] = NativeExecutor(slice_, inputs="native")
        return built[name]

    return build


def _parameters(name: str, mt: str, factor: int = 1) -> dict[str, str]:
    """aS, MT and the process's other dimensionful inputs in units of
    GeV / ``factor``."""

    parameters = {"aS": "0.118", "MT": repr(float(mt) * factor)}
    for key, (value, dimension) in PROCESSES[name][3].items():
        parameters[key] = repr(float(value) * float(factor) ** dimension)
    return parameters


def _evaluate(executor, point, scale, parameters, precision=None, stability=None):
    (result,) = executor.evaluate(
        [point],
        [scale],
        parameters=parameters,
        precision=precision,
        stability=stability,
    )
    return result


def _binary64(laurents) -> dict:
    """Laurent coefficients rounded to binary64."""

    return {
        sector: Budget(tuple(float(x) for x in laurent.coefficients))
        for sector, laurent in laurents.items()
    }


def _virtual(result) -> dict:
    return result.components["virtual"]


def _routed(executor) -> int:
    return executor.last_report["counts"][ROUTED]


def _within_estimate(reference, double) -> None:
    estimate = double.stability.relative
    assert estimate <= ESTIMATE_CAP, estimate
    error = deviation(_binary64(_virtual(reference)), _virtual(double))
    assert error <= max(BINARY64, ESTIMATE_MARGIN * estimate), (error, estimate)


@pytest.mark.loop_fast
def test_a_light_top_takes_the_exact_engine_and_agrees_with_mpfr(executors):
    """e+ e- > t t~ with MT = 1e-6: binary64 (lanes routed to the exact
    engine) equals MPFR to binary64's accuracy (the slice native's fast
    engine was 107 % off); at MT = 173 no lane is routed."""

    executor = executors("epem_ttx")
    light = _parameters("epem_ttx", LIGHT_MT)
    checks = StabilityChecks(reduction_direction=True)
    double = _evaluate(executor, POINT, SCALE, light, stability=checks)
    assert _routed(executor) > 0
    reference = _evaluate(executor, POINT, SCALE, light, Precision.arbitrary(32))
    _within_estimate(reference, double)
    _evaluate(
        executor, HEAVY_POINT, SCALE, _parameters("epem_ttx", "173"), stability=checks
    )
    assert _routed(executor) == 0


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["epem_ttx", "uux_gg"])
def test_a_light_top_agrees_in_every_carrier(executors, name):
    """The light top in loop lines and legs (e+ e- > t t~) and in a closed
    loop (u u~ > g g): binary64 with the rescaled check (masters recomputed)
    within its estimate of MPFR, DoubleFloat within 1e-24; MT = 173 routes
    nothing."""

    executor = executors(name)
    light = _parameters(name, LIGHT_MT)
    reference = _evaluate(executor, POINT, SCALE, light, Precision.arbitrary(32))
    double_double = _evaluate(executor, POINT, SCALE, light, Precision.double_double())
    assert deviation(_virtual(reference), _virtual(double_double)) < DOUBLE_DOUBLE
    checks = StabilityChecks(reduction_direction=True, rescaled=True)
    double = _evaluate(executor, POINT, SCALE, light, stability=checks)
    assert _routed(executor) > 0
    _within_estimate(reference, double)
    heavy = HEAVY_POINT if name == "epem_ttx" else POINT
    _evaluate(executor, heavy, SCALE, _parameters(name, "173"), stability=checks)
    assert _routed(executor) == 0


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "mt", "routed_in_gev"),
    [
        ("epem_ttx", LIGHT_MT, True),
        ("uux_gg", LIGHT_MT, True),
        ("uux_gg", "173", False),
    ],
    ids=["epem_ttx-light", "uux_gg-light", "uux_gg-loop_sm"],
)
def test_the_virtual_is_unit_independent(executors, name, mt, routed_in_gev):
    """GeV against MeV: the same dimensionless virtual to binary64 accuracy
    and the same outcome of the point; the light top routes in GeV only
    (m^2 = 1e-12 against 1e-6 in MeV)."""

    executor = executors(name)
    point = HEAVY_POINT if name == "epem_ttx" and mt == "173" else POINT
    checks = StabilityChecks(reduction_direction=True)
    gev = _evaluate(executor, point, SCALE, _parameters(name, mt), stability=checks)
    assert (_routed(executor) > 0) is routed_in_gev
    mev = _evaluate(
        executor,
        [[repr(float(x) * MEV) for x in leg] for leg in point],
        repr(float(SCALE) * MEV**2),
        _parameters(name, mt, MEV),
        stability=checks,
    )
    assert _routed(executor) == 0
    estimate = max(gev.stability.relative, mev.stability.relative)
    assert estimate <= ESTIMATE_CAP, estimate
    error = deviation(_virtual(gev), _virtual(mev))
    assert error <= max(BINARY64, ESTIMATE_MARGIN * estimate), (error, estimate)


@pytest.mark.loop_physics
def test_light_lanes_beyond_the_binary64_masters_fail_closed(executors):
    """g g > g g with MT = 1e-6: a box of four light lines has no finite
    binary64 D0 in oneloopmaster, so the point fails naming it (the snap gave
    the massless box instead). DoubleFloat evaluates it; the same box's
    conditioning (m^2/s ~ 1e-18) limits it to 2.1e-18 of MPFR here, so it is
    held to the binary64 rule only."""

    executor = executors("gg_gg")
    light = _parameters("gg_gg", LIGHT_MT)
    with pytest.raises(FailedPoints, match="D0, failure: NonFiniteResult"):
        _evaluate(executor, POINT, SCALE, light)
    reference = _evaluate(executor, POINT, SCALE, light, Precision.arbitrary(32))
    double_double = _evaluate(executor, POINT, SCALE, light, Precision.double_double())
    assert deviation(_virtual(reference), _virtual(double_double)) < BINARY64
