# SPDX-License-Identifier: 0BSD
"""Pole selection of the v2 virtual: ``poles = "all" | "none" | "uv" | "ir"``
(design §12.4; ``loops.v2.poles``).

Checks, one phase-space point per process (HANDOFF §0.3):

* the finite part of every component is identical, bit for bit, in all four
  modes (native binary64 and DoubleFloat), and ``"all"`` is the default;
* the UV and IR parts of every component and of the virtual add up to the
  combined poles; the double pole is infrared; ``"uv"`` leaves the
  renormalised virtual without poles and ``"ir"`` keeps those of ``"all"``;
* the IR poles of the renormalised virtual equal Catani's operator, massless
  (g g > g g, u u~ > g g g) and massive (g g > t t~, u u~ > t t~), and the
  provider poles of g g > t t~;
* the bare-loop UV poles cancel the UV_local counterterm poles up to the
  textbook Feynman-gauge field renormalisation of the external legs,
  independently of the reduction: -(bare_UV + uv_local) = sum_legs Z_i^UV
  Born with Z_g^UV = (alpha_s/4pi)(5/3 C_A - 4/3 T_F n_f) and Z_q^UV =
  -(alpha_s/4pi) C_F (massless and massive quarks);
* the Python reference path (``LoopSlice.evaluate``) and the native executor
  agree in split mode, and the stability estimates exist in every mode.

The lane-level residue is checked against NinjaRS's generic engine with
oneloopmaster's UV- and IR-selected masters in
``rust/loop-reduction/tests/runtime_v2_reduce.rs``.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    LoopSlice,
    Precision,
    SliceRequest,
)
from pyamplicol.loops.v2.fields import FIELD_COMPONENT
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.poles import (
    MODES,
    pole_mode,
    reducer_poles,
    select_lane,
    slot_switches,
    split_sector,
)
from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
# The providers' loop_sm-no_b_mass scheme; a stable top (a top width enters
# the loop propagators as a complex mass, whose CMS counterterms are WP3.2).
FIXED = {"MB": "0", "WT": "0"}
ALPHA_S = 0.118
NC = 3
CA, CF, TF = 3.0, 4.0 / 3.0, 0.5
# Quark flavours in the loops of loop_sm (u d s c b t).
LOOP_FLAVOURS = 6
POLE_TOLERANCE = 1e-10
SPLIT_TOLERANCE = 1e-12


def _legacy_point(name: str):
    data = json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())
    return data["points"][0], data["scales_squared"][0]


def _required_point(name: str, index: int = 0):
    data = json.loads((FIXTURES / "required" / f"{name}.json").read_text())
    mu = Decimal(str(data["mu_r"][index]))
    return data["points"][index], format(mu * mu, "f"), data["rows"][index]


# name: process, Born orders, point, external legs (gluons, quarks)
PROCESSES = {
    "gg_gg": (
        "g g > g g",
        {"QCD": 2, "QED": 0},
        lambda: _legacy_point("gg_gg"),
        (4, 0),
    ),
    "uux_ggg": (
        "u u~ > g g g",
        {"QCD": 3, "QED": 0},
        lambda: _legacy_point("uux_ggg"),
        (3, 2),
    ),
    "gg_ttx": (
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        lambda: _required_point("gg_ttx")[:2],
        (2, 2),
    ),
    "uux_ttx": (
        "u u~ > t t~",
        {"QCD": 2, "QED": 0},
        lambda: _legacy_point("tt"),
        (0, 4),
    ),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, tuple[LoopSlice, NativeExecutor]] = {}

    def build(name: str) -> tuple[LoopSlice, NativeExecutor]:
        if name not in built:
            process, born, _point, _legs = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            slice_ = LoopSlice.generate(
                compiled,
                SliceRequest(process, born, loop, fixed_parameters=FIXED),
                loop_native,
            )
            built[name] = slice_, NativeExecutor(slice_)
        return built[name]

    return build


# Massive external legs run on the Python reference path: the native
# executor's family numerators of g g > t t~ and u u~ > t t~ differ from it
# (``test_native_executor_massive_external_legs``).
PYTHON_PATH = ("gg_ttx", "uux_ttx")


@pytest.fixture(scope="module")
def modes(slices):
    """One point per process and carrier in every pole mode: the native
    executor (binary64, DoubleFloat), or for ``PYTHON_PATH`` the slice's
    reference path (binary64, 32 digits)."""

    cache: dict = {}

    def evaluate(name: str, carrier: str = "double"):
        key = (name, carrier)
        if key not in cache:
            slice_, executor = slices(name)
            point, scale = PROCESSES[name][2]()
            python = name in PYTHON_PATH
            if carrier == "double":
                precision = Precision.double()
            else:
                precision = (
                    Precision.arbitrary(32) if python else Precision.double_double()
                )
            out = {}
            for mode in (True, *MODES):
                if python:
                    out[mode] = slice_.evaluate(
                        point,
                        scale,
                        parameters=PARAMETERS,
                        precision=precision,
                        poles=mode,
                    )
                else:
                    (out[mode],) = executor.evaluate(
                        [point],
                        [scale],
                        parameters=PARAMETERS,
                        precision=precision,
                        poles=mode,
                    )
            cache[key] = out
        return cache[key]

    return evaluate


def _scale(laurent) -> float:
    return max(abs(complex(x)) for x in laurent.coefficients)


# ----------------------------------------------------------------- modes


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", ("double", "extended"))
@pytest.mark.parametrize("name", ("gg_gg", "gg_ttx"))
def test_finite_parts_are_identical_in_every_mode(modes, name, carrier):
    results = modes(name, carrier)
    default, combined = results[True], results["all"]
    assert default.poles == combined.poles == "all"
    assert default.pole_origins is None
    # "all" is the default, bit for bit.
    assert default.totals == combined.totals
    for mode in MODES:
        result = results[mode]
        assert result.poles == mode
        assert result.born == combined.born
        for component in COMPONENTS:
            for sector, laurent in combined.components[component].items():
                assert result.components[component][sector].finite == laurent.finite
        if mode == "none":
            assert all(
                laurent.single_pole is None
                for sectors in result.components.values()
                for laurent in sectors.values()
            )
            assert result.ir_poles is None


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg", "gg_ttx"))
def test_uv_and_ir_parts_add_up_to_the_combined_poles(modes, name):
    results = modes(name)
    combined, uv, ir = results["all"], results["uv"], results["ir"]
    assert uv.ir_poles is None and ir.ir_poles == combined.ir_poles
    for origin, result in (("uv", uv), ("ir", ir)):
        # The selected components are the origin's parts.
        assert result.pole_origins[origin] == result.components
    for component in COMPONENTS:
        for sector, whole in combined.components[component].items():
            u, i = uv.components[component][sector], ir.components[component][sector]
            scale = max(_scale(whole), _scale(combined.virtual(sector)))
            assert abs(u.single_pole + i.single_pole - whole.single_pole) <= (
                SPLIT_TOLERANCE * scale
            ), component
            assert u.double_pole == 0, component
            assert abs(i.double_pole - whole.double_pole) <= SPLIT_TOLERANCE * scale
    for sector, whole in combined.components["virtual"].items():
        scale = _scale(whole)
        # The renormalised virtual is UV finite; its IR poles are all of them.
        assert abs(uv.virtual(sector).single_pole) <= SPLIT_TOLERANCE * scale
        assert abs(ir.virtual(sector).single_pole - whole.single_pole) <= (
            SPLIT_TOLERANCE * scale
        )
        assert ir.virtual(sector).double_pole == whole.double_pole
        # UV_local and R2 poles are ultraviolet.
        for component in ("uv_local", "r2"):
            assert ir.components[component][sector].single_pole == 0
    # The bare loop has both kinds of poles (scaleless bubbles included).
    for sector in combined.components["bare"]:
        assert abs(uv.components["bare"][sector].single_pole) > 0
        assert abs(ir.components["bare"][sector].single_pole) > 0


# ----------------------------------------------------------------- physics


def _catani(result, sector):
    single, double = result.ir_poles[sector]
    return float(single), float(double)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg", "gg_ttx", "uux_ttx"))
def test_ir_poles_of_the_renormalised_virtual_equal_catani(modes, name):
    result = modes(name)["ir"]
    for sector, virtual in result.components["virtual"].items():
        single, double = _catani(result, sector)
        scale = max(abs(single), abs(double))
        assert abs(float(virtual.single_pole.real) - single) <= POLE_TOLERANCE * scale
        assert abs(float(virtual.double_pole.real) - double) <= POLE_TOLERANCE * scale


@pytest.mark.loop_physics
def test_massive_catani_equals_the_gg_ttx_provider_poles(modes):
    """g g > t t~ at the providers' first point (MadLoop, Recola, OpenLoops
    agree to 6e-15): the massive Catani prediction and the virtual's poles."""

    _point, _scale_squared, row = _required_point("gg_ttx")
    result = modes("gg_ttx")["all"]
    ((sector, virtual),) = result.components["virtual"].items()
    single, double = _catani(result, sector)
    born = float(result.born[next(iter(result.born))])
    assert abs(born - row["born"]) <= 1e-12 * row["born"]
    scale = abs(row["double_pole"])
    for got in ((single, double), (virtual.single_pole.real, virtual.double_pole.real)):
        assert abs(float(got[0]) - row["single_pole"]) <= POLE_TOLERANCE * scale
        assert abs(float(got[1]) - row["double_pole"]) <= POLE_TOLERANCE * scale


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg", "gg_ttx"))
def test_bare_uv_poles_cancel_the_counterterms_up_to_field_renormalisation(modes, name):
    """Independent of the reduction's residue: the textbook Feynman-gauge
    Z^UV of every external leg times the Born (the field factor multiplies
    the Born once per leg, delta-Z / 2 in the amplitude)."""

    gluons, quarks = PROCESSES[name][3]
    a = ALPHA_S / (4 * math.pi)
    z_gluon = a * (5 / 3 * CA - 4 / 3 * TF * LOOP_FLAVOURS)
    z_quark = -a * CF
    result = modes(name)["uv"]
    ((sector, bare),) = result.components["bare"].items()
    local = result.components["uv_local"][sector]
    fields = result.components[FIELD_COMPONENT][sector]
    born = float(result.born[next(iter(result.born))])
    expected = (gluons * z_gluon + quarks * z_quark) * born
    got = -(float(bare.single_pole.real) + float(local.single_pole.real))
    assert abs(got - expected) <= POLE_TOLERANCE * abs(float(bare.single_pole.real))
    assert abs(float(fields.single_pole.real) - expected) <= POLE_TOLERANCE * abs(
        float(bare.single_pole.real)
    )


@pytest.mark.loop_physics
def test_python_path_and_native_executor_agree_in_split_mode(slices, modes):
    slice_, _executor = slices("gg_gg")
    point, scale = PROCESSES["gg_gg"][2]()
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
    for mode in ("uv", "ir"):
        python = slice_.evaluate(
            point, scale, parameters=PARAMETERS, poles=mode, stability=checks
        )
        native = modes("gg_gg")[mode]
        assert python.poles == mode
        for component in COMPONENTS:
            for sector, laurent in native.components[component].items():
                other = python.components[component][sector]
                bound = 1e-12 * _scale(native.virtual(sector)) + 1e-12 * _scale(laurent)
                for x, y in zip(laurent.coefficients, other.coefficients, strict=True):
                    assert abs(complex(x) - complex(y)) <= bound, (mode, component)
        # Every check configuration estimates the selected coefficients.
        assert python.stability is not None
        assert len(python.stability.coefficients) == 3


@pytest.mark.loop_physics
def test_native_estimates_exist_in_every_mode(slices):
    _slice, executor = slices("gg_gg")
    point, scale = PROCESSES["gg_gg"][2]()
    for mode in MODES:
        for rescaled in (False, True):
            (result,) = executor.evaluate(
                [point],
                [scale],
                parameters=PARAMETERS,
                poles=mode,
                stability=StabilityChecks((), True, rescaled),
            )
            assert len(result.stability.coefficients) == (1 if mode == "none" else 3)
            assert result.stability.finite < 1e-10


# ----------------------------------------------------------------- unit


@pytest.mark.loop_fast
def test_pole_mode_options():
    assert pole_mode(True) == "all" and pole_mode(False) == "none"
    assert [pole_mode(m) for m in MODES] == list(MODES)
    with pytest.raises(ValueError):
        pole_mode("both")
    assert [reducer_poles(m) for m in MODES] == [True, False, "split", "split"]
    on = slot_switches("none")
    assert on("uv_local", 0) and not on("uv_local", -1) and not on("r2", -2)
    for mode in ("all", "uv", "ir"):
        assert all(slot_switches(mode)("uv_local", p) for p in (0, -1, -2))


@pytest.mark.loop_fast
def test_lane_and_sector_splits_are_exact_for_exact_numbers():
    from pyamplicol.loops.v2.assemble import Laurent

    q = Fraction
    assert select_lane((q(1), q(5), q(-2), q(3)), "uv", 0) == (1, 3, 0)
    assert select_lane((q(1), q(5), q(-2), q(3)), "ir", 0) == (1, 2, -2)
    values = {
        "bare": Laurent(q(7), q(5), q(-2)),
        "r2": Laurent(q(1), q(0), q(0)),
        "uv_local": Laurent(q(0), q(-4), q(0)),
        FIELD_COMPONENT: Laurent(q(1, 2), q(3), q(0)),
    }
    parts = split_sector(values, Laurent(q(7), q(3), q(0)), FIELD_COMPONENT, Laurent)
    assert parts["uv"][FIELD_COMPONENT] == Laurent(q(1, 2), q(1), q(0))
    assert parts["ir"][FIELD_COMPONENT] == Laurent(q(1, 2), q(2), q(0))
    assert parts["ir"]["uv_local"] == Laurent(q(0), q(0), q(0))
    for name, whole in values.items():
        u, i = parts["uv"][name], parts["ir"][name]
        assert u.finite == i.finite == whole.finite
        assert u.single_pole + i.single_pole == whole.single_pole
        assert u.double_pole + i.double_pole == whole.double_pole
    total = {o: sum((p.single_pole for p in parts[o].values()), q(0)) for o in parts}
    assert total["uv"] == 0
    assert total["ir"] == sum(v.single_pole for v in values.values())


@pytest.mark.loop_physics
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="pre-existing (slice 2a56a0e7, native bda5e256): the native executor's "
    "family integrals of g g > t t~ differ from the reference path's at massive "
    "external legs (the reference path equals MadLoop); not a pole-selection defect",
)
def test_native_executor_massive_external_legs(slices):
    slice_, executor = slices("gg_ttx")
    point, scale = PROCESSES["gg_ttx"][2]()
    python = slice_.evaluate(point, scale, parameters=PARAMETERS)
    (native,) = executor.evaluate([point], [scale], parameters=PARAMETERS)
    for sector, laurent in python.components["virtual"].items():
        other = native.components["virtual"][sector]
        for x, y in zip(laurent.coefficients, other.coefficients, strict=True):
            assert abs(complex(x) - complex(y)) <= 1e-12 * _scale(laurent)
