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
  loop_sm's top mass no lane is routed (the fast engine's bits). Only
  e+ e- > t t~ detects the snap end to end (the slice native was 107 % off
  there): the closed light-top loop of u u~ > g g is smooth as m -> 0, so
  the massless integrals the snap computed gave the same virtual to
  binary64 accuracy; it checks the routed path's accuracy on boxes and
  bubbles of light lines (the Rust tests detect the snap integral by
  integral);
* the dimensionless virtual is unit independent: GeV and MeV (every momentum
  and dimensionful parameter times 1e3, its square and the scale times 1e6)
  agree to binary64 accuracy, the light top routed in GeV (m^2 = 1e-12) and
  on the fast engine in MeV (1e-6);
* a power-of-ten scan of the units (g g > t t~ in the default tier,
  u u~ > d d~ g nightly; nothing routed): small units and the GeV-like range
  keep the virtual and its estimate, while large factors with the beams along
  z do not, which strict xfails pin: NinjaRS's light-like spinors take a beam
  along -z, formed with a rounding residue above their absolute 1e-9 window,
  in a cancelling form (silently 6-22 % off with checks off; the rescaled
  check's false alarm on an exact value); the exact 90-degree rotation that
  takes the beams off the z axis holds at the same factors;
* where oneloopmaster's binary64 masters cannot evaluate the light lanes
  (g g > g g: a box of four light lines, D0 not finite), the point fails with
  the master's reason instead of taking the massless integral, and
  DoubleFloat evaluates it (to its own conditioning there, 2.1e-18);
* routed lanes use Symbolica in the native code, under pyAmpliCol's package
  key: unlicensed, with the restricted permit held elsewhere, a routed
  binary64 evaluation completes with the key and aborts without it, through
  the executor and through the reducer on the caller's own thread (the
  reducer's explicit guard);
* the artifact records the routing: a public generation with MT = 1e-6 fixed
  names MT with its square and the threshold in the generation record of
  e+ e- > t t~ (``binary64_exact_engine``) and writes no entry for
  e+ e- > u u~, which has no top line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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

# loop_sm's other dimensionful inputs (value, mass dimension).
UNITS = {
    "MZ": ("91.188", 1),
    "Gf": ("1.16639e-05", -2),
    "MH": ("125", 1),
    "ymt": ("173", 1),
}

# name: process, Born orders, loop orders, the model parameters that carry a
# unit besides MT (value, mass dimension) for the unit tests.
PROCESSES = {
    "epem_ttx": (
        "e+ e- > t t~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        {"MZ": ("91.188", 1), "Gf": ("1.16639e-05", -2)},
    ),
    "uux_gg": ("u u~ > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, {}),
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, {}),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, UNITS),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, UNITS),
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


def _parameters(name: str, mt: str, factor: float = 1) -> dict[str, str]:
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
@pytest.mark.parametrize(
    "name",
    # u u~ > g g does not detect the snap (docstring): nightly.
    ["epem_ttx", pytest.param("uux_gg", marks=pytest.mark.loop_nightly)],
)
def test_a_light_top_agrees_in_every_carrier(executors, name):
    """The light top in loop lines and legs (e+ e- > t t~, where the snap
    was 107 % off) and in a closed loop (u u~ > g g, smooth as m -> 0, where
    the snap was not wrong: a check of the routed path's accuracy): binary64
    with the rescaled check (masters recomputed) within its estimate of MPFR,
    DoubleFloat within 1e-24; MT = 173 routes nothing."""

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
        pytest.param("epem_ttx", LIGHT_MT, True, id="epem_ttx-light"),
        pytest.param("uux_gg", LIGHT_MT, True, id="uux_gg-light"),
        # The heavy top routes nothing: a unit check of the fast engine,
        # which the power-of-ten scan below makes in the default tier.
        pytest.param(
            "uux_gg", "173", False, id="uux_gg-loop_sm", marks=pytest.mark.loop_nightly
        ),
    ],
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


# Units of the model far from GeV (genericity audit 2, G2-M3 item 3; the
# evidence audit of 2026-10-28): every momentum and dimensionful parameter
# times a power of ten f (the scale times f^2, G_F times f^-2). The virtual's
# Laurent coefficients over the Born are dimensionless, so they must not move
# beyond binary64 accuracy. g g > t t~: beams along z at E = 400, the top at
# cos(theta) = 0.21, phi = 0.7 (MT = 173, mu_R^2 = 160000); u u~ > d d~ g:
# the provider fixture's point 1 (mu_R = 700).
_TOP = math.sqrt(400.0**2 - 173.0**2)
_SIN = math.sqrt(1 - 0.21**2)
_K = (_TOP * _SIN * math.cos(0.7), _TOP * _SIN * math.sin(0.7), _TOP * 0.21)
UNIT_POINTS = {
    "gg_ttx": (
        [
            [400.0, 0.0, 0.0, 400.0],
            [400.0, 0.0, 0.0, -400.0],
            [400.0, *_K],
            [400.0, *(-k for k in _K)],
        ],
        160000.0,
    ),
    "uux_ddxg": (
        [
            [float(x) for x in leg]
            for leg in json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "fixtures/loop/required/uux_ddxg.json"
                ).read_text()
            )["points"][1]
        ],
        700.0**2,
    ),
}
# Owner of the large-unit defect (DESIGN §11.2 'Other thresholds'): the
# strict xfails below turn into failures when the fix is pinned.
NINJA_SPINORS = (
    "pinned NinjaRS 41a4c00, fast/geometry.rs Spinor::new: a beam along -z "
    "formed as a difference of loop shifts carries a rounding residue in "
    "p0 + p3, below infrared_eps (1e-9, absolute) in GeV-like units and above "
    "it at large energies in the model's units, where the cancelling "
    "sqrt(p0 + p3) form divides rounding noise by its root (O(10 %) silent "
    "errors); routed to the light-like spinor fix, "
    "workspace/reports/engine-v2/upstream/ninjars-lightlike-spinor/NOTE.md"
)


def _x90(point):
    """The exact rotation by 90 degrees about x: (E, x, y, z) -> (E, x, -z, y)
    (the beams along -y and y)."""

    return [[e, x, -z, y] for e, x, y, z in point]


def _over_born(result) -> list[complex]:
    (born,) = result.born.values()
    (virtual,) = _virtual(result).values()
    return [complex(x) / complex(born) for x in virtual.coefficients]


def _unit_error(reference, result) -> float:
    """Largest deviation of the virtual's Laurent coefficients over the Born,
    relative to the largest reference coefficient."""

    want = _over_born(reference)
    got = _over_born(result)
    return max(abs(a - b) for a, b in zip(got, want, strict=True)) / max(map(abs, want))


@pytest.fixture(scope="module")
def unit_references(executors):
    """Per (process, frame): DoubleFloat and the rescaled check's estimate in
    GeV, the reference of every factor."""

    found = {}

    def reference(name: str, frame: str):
        if (name, frame) not in found:
            executor = executors(name)
            point, scale = UNIT_POINTS[name]
            point = _x90(point) if frame == "x90" else point
            point = [[repr(x) for x in leg] for leg in point]
            parameters = _parameters(name, "173")
            double_double = _evaluate(
                executor, point, repr(scale), parameters, Precision.double_double()
            )
            checks = StabilityChecks(reduction_direction=True, rescaled=True)
            gev = _evaluate(executor, point, repr(scale), parameters, stability=checks)
            found[name, frame] = (double_double, gev.stability.relative)
        return found[name, frame]

    return reference


def _units(name, factor, frame="z", *, defect=None, nightly=False):
    marks = []
    if defect is not None:
        marks.append(
            pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=f"{defect}: {NINJA_SPINORS}"
            )
        )
    if nightly:
        marks.append(pytest.mark.loop_nightly)
    return pytest.param(
        name, factor, frame, marks=marks, id=f"{name}-{factor:g}-{frame}"
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "factor", "frame"),
    [
        _units("gg_ttx", 1e-6),
        _units("gg_ttx", 1e5),
        _units("gg_ttx", 7e5, defect="22 % off, checks off"),
        _units("gg_ttx", 1e6, defect="value exact; rescaled check 0.43"),
        _units("gg_ttx", 7e5, "x90"),
        _units("gg_ttx", 3e7, defect="9.7 % off, checks off", nightly=True),
        _units("gg_ttx", 3e7, "x90", nightly=True),
        _units("uux_ddxg", 1e-6, nightly=True),
        _units("uux_ddxg", 1e4, defect="6.8 % off, checks off", nightly=True),
        _units("uux_ddxg", 1e6, defect="5.7 % off, checks off", nightly=True),
        _units("uux_ddxg", 1e6, "x90", nightly=True),
    ],
)
def test_power_of_ten_units_keep_the_virtual_and_its_estimate(
    executors, unit_references, name, factor, frame
):
    """Binary64 (the fast engine, nothing routed) in units of 1/f GeV against
    DoubleFloat in GeV: with checks off, the virtual over the Born within
    tenfold the GeV evaluation's own estimate (or 1e-12); with the rescaled
    reduction check, an estimate below 1e-9. Small units and the GeV-like
    range hold. With the beams along z, large factors do not (strict xfails,
    NINJA_SPINORS: silently wrong values, a false alarm of the rescaled check
    on an exact value, whose frame is 0.7 times the inputs); the exact
    rotation of the same points by 90 degrees about x, which takes the beams
    off the z axis, holds at the same factors: the defect is the beam axis'
    spinor, not the units of the masses or invariants."""

    executor = executors(name)
    double_double, gev_estimate = unit_references(name, frame)
    point, scale = UNIT_POINTS[name]
    point = _x90(point) if frame == "x90" else point
    point = [[repr(x * factor) for x in leg] for leg in point]
    parameters = _parameters(name, "173", factor)
    scaled = repr(scale * factor**2)
    value = _evaluate(executor, point, scaled, parameters)
    assert _routed(executor) == 0
    error = _unit_error(double_double, value)
    assert error <= max(BINARY64, ESTIMATE_MARGIN * gev_estimate), (error, gev_estimate)
    checks = StabilityChecks(reduction_direction=True, rescaled=True)
    estimate = _evaluate(executor, point, scaled, parameters, stability=checks)
    assert estimate.stability.relative <= ESTIMATE_CAP, estimate.stability.relative


# DoubleFloat on the box of four light lines (m^2/s ~ 1e-18): measured
# 2.05e-18 from MPFR(32), which is itself within 2.9e-35 of MPFR(64) there.
LIGHT_BOX_DOUBLE_DOUBLE = 1e-17


@pytest.mark.loop_physics
def test_light_lanes_beyond_the_binary64_masters_fail_closed(executors):
    """g g > g g with MT = 1e-6: a box of four light lines has no finite
    binary64 D0 in oneloopmaster, so the point fails naming it (the snap gave
    the massless box instead). DoubleFloat evaluates it, limited by the same
    box's conditioning (m^2/s ~ 1e-18): within 1e-17 of MPFR (measured
    2.05e-18; the 32-digit reference is within 2.9e-35 of 64 digits)."""

    executor = executors("gg_gg")
    light = _parameters("gg_gg", LIGHT_MT)
    with pytest.raises(FailedPoints, match="D0, failure: NonFiniteResult"):
        _evaluate(executor, POINT, SCALE, light)
    reference = _evaluate(executor, POINT, SCALE, light, Precision.arbitrary(32))
    double_double = _evaluate(executor, POINT, SCALE, light, Precision.double_double())
    error = deviation(_virtual(reference), _virtual(double_double))
    assert error < LIGHT_BOX_DOUBLE_DOUBLE, error


def _light_top_card(model: Path, output: Path, entries) -> str:
    """A public generation card of loop_sm with the light top fixed."""

    processes = ", ".join(
        f"{{expression={json.dumps(expression)}, name={json.dumps(name)}}}"
        for name, expression in entries
    )
    fixed = ", ".join(
        f"{name} = {json.dumps(value)}"
        for name, value in {**FIXED, "MT": LIGHT_MT}.items()
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{processes}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


@pytest.fixture(scope="module")
def light_artifact(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """A public v2 artifact of e+ e- > t t~ and e+ e- > u u~ with MT = 1e-6
    fixed."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    entries = (("epem_ttx", "e+ e- > t t~"), ("epem_uux", "e+ e- > u u~"))
    root = tmp_path_factory.mktemp("light-top")
    output = root / "artifact"
    card = root / "card.toml"
    card.write_text(_light_top_card(loop_sm_compiled_model, output, entries))
    processes = ProcessSet(
        tuple(
            ProcessRequest.parse(expression, name=name) for name, expression in entries
        )
    )
    Generator(resolve_config(card)).generate(processes, output)
    return output


@pytest.mark.loop_physics
def test_the_artifact_records_the_light_masses(light_artifact, loop_native):
    """The generation record names the masses whose binary64 lanes take the
    exact engine at the generation parameters: MT = 1e-6 fixed gives
    e+ e- > t t~ the entry ``binary64_exact_engine`` with MT's square and the
    fast engine's threshold; e+ e- > u u~ (no top line, massless legs) has
    none, so its record is unchanged."""

    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(light_artifact, verify_payloads=True)
    filters = {
        record["id"]: record["filters"]
        for record in manifest.extensions["generation"]["concrete_processes"]
    }
    threshold = loop_native.V2_FAST_ENGINE_THRESHOLD
    assert threshold == 1e-9
    assert filters["epem_ttx"]["binary64_exact_engine"] == {
        "threshold": repr(threshold),
        "squared_masses": {"MT": "1e-12"},
    }
    assert "binary64_exact_engine" not in filters["epem_uux"]


# The portability tool: unlicensed evaluation processes and the user's
# restricted Symbolica permit (``tests/unit/test_loop_artifact_portability.py``).
_TOOL = Path(__file__).resolve().parents[2] / "tools" / "loop_artifact_portability.py"
RESTRICTED = "restricted Symbolica"


def _portability():
    import importlib.util

    spec = importlib.util.spec_from_file_location("loop_artifact_portability", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.loop_physics
def test_routed_binary64_lanes_run_under_the_package_key(light_artifact, tmp_path):
    """The one binary64 path that touches Symbolica (DESIGN §12.1 (e)): a
    routed lane is reduced by the generic engine, which builds and evaluates
    Symbolica expressions under pyAmpliCol's native library key. While
    another process holds the user's restricted permit (a private lock
    directory), an unlicensed binary64 evaluation of e+ e- > t t~ with
    MT = 1e-6 completes with the key and aborts with it disabled (the
    control); e+ e- > u u~ of the same artifact (nothing routed) completes
    with the key disabled, as every unrouted binary64 evaluation does."""

    import os
    import subprocess
    import sys

    portability = _portability()
    points = tmp_path / "points.json"
    points.write_text(json.dumps(portability.points(4, 1)))
    lock = tmp_path / "locks"
    lock.mkdir(mode=0o700)
    environment = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    no_key = {**environment, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}
    runs = (
        ("epem_ttx", environment),
        ("epem_ttx", no_key),
        ("epem_uux", no_key),
    )
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        running = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(_TOOL),
                    "evaluate",
                    str(light_artifact),
                    "--process",
                    process,
                    "--points",
                    str(points),
                    "--arithmetic",
                    "double",
                    "--out",
                    str(tmp_path / f"{k}-{process}.json"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=permit.environment(env),
                cwd=tmp_path,
                preexec_fn=portability._no_core,
            )
            for k, (process, env) in enumerate(runs)
        ]
        done = []
        for run in running:
            out, err = run.communicate(timeout=1800)
            done.append((run.returncode, out, err))
    assert permit.record["verified"], permit.record
    (keyed, out, err), (refused, r_out, r_err), (unrouted, u_out, u_err) = done
    assert keyed == 0, out[-2000:] + err[-2000:]
    assert RESTRICTED not in out + err
    assert refused != 0
    assert RESTRICTED in r_out + r_err
    assert unrouted == 0, u_out[-2000:] + u_err[-2000:]


# One light bubble, B0(p^2 = 1e4; m^2, m^2) at rank 0, reduced by the v2
# family reducer on the calling thread (as the Python reference path of
# ``assemble.py`` does): ``light`` routes it (m^2 = 1e-12), ``heavy`` does
# not (m^2 = 29929). Prints the statuses, values and routed-lane count.
_REDUCER_SOURCE = """
import json, sys
from pyamplicol.loops.native import load_loop_native

native = load_loop_native()
mass = {"light": "1e-12", "heavy": "29929"}[sys.argv[1]]


def batch(values):
    return native.NativeBatch.from_decimal(
        [[x, "0"] for x in values], 1, len(values), "double", None, [5]
    )


group = (
    2,
    batch(["0", "0", "0", "0", "100", "0", "0", "0"]),
    batch([mass, mass]),
    batch(["250000"]),
    batch(["0", "10000", "10000", "0"]),
    batch(["1.5"]),
)
reducer = native.V2Reducer("double", None, sys.argv[2])
(direct,), _check = reducer.reduce([group])
print(json.dumps({
    "statuses": direct.statuses,
    "values": direct.successful_values().to_decimal(),
    "routed": reducer.take_exact_lanes(),
}))
"""


@pytest.mark.loop_physics
def test_the_reference_reducer_takes_the_key_on_its_own_thread(tmp_path):
    """The explicit guard of a routed reduction (``FastEngine::reduce``,
    ``pyamplicol::unlock``): the Python reference path reduces on the caller's
    thread, which holds no executor worker's key. While another process holds
    the restricted permit (a private lock directory), an unlicensed process
    reduces a light bubble on the binary64 fast reducer: the lane is routed
    (``V2Reducer.take_exact_lanes``) and completes with the package key,
    bitwise equal to the generic binary64 reducer here (licensed); it aborts
    with the key disabled (the control), while the heavy bubble, not routed,
    completes with the key disabled. Without the guard the first run aborts
    as the second does."""

    import os
    import subprocess
    import sys

    portability = _portability()
    lock = tmp_path / "locks"
    lock.mkdir(mode=0o700)
    environment = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    no_key = {**environment, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}
    runs = (("light", environment), ("light", no_key), ("heavy", no_key))
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        done = [
            subprocess.run(
                [sys.executable, "-c", _REDUCER_SOURCE, mass, "fast"],
                capture_output=True,
                text=True,
                env=permit.environment(env),
                cwd=tmp_path,
                preexec_fn=portability._no_core,
                timeout=600,
            )
            for mass, env in runs
        ]
    assert permit.record["verified"], permit.record
    keyed, refused, unrouted = done
    assert keyed.returncode == 0, keyed.stdout[-2000:] + keyed.stderr[-2000:]
    assert RESTRICTED not in keyed.stdout + keyed.stderr
    assert refused.returncode != 0
    assert RESTRICTED in refused.stdout + refused.stderr
    assert unrouted.returncode == 0, unrouted.stdout[-2000:] + unrouted.stderr[-2000:]
    light = json.loads(keyed.stdout.splitlines()[-1])
    heavy = json.loads(unrouted.stdout.splitlines()[-1])
    assert light["statuses"] == heavy["statuses"] == [[5, "success"]]
    assert (light["routed"], heavy["routed"]) == (1, 0)
    generic = subprocess.run(
        [sys.executable, "-c", _REDUCER_SOURCE, "light", "generic"],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    reference = json.loads(generic.stdout.splitlines()[-1])
    assert reference["routed"] == 0
    assert light["values"] == reference["values"]
