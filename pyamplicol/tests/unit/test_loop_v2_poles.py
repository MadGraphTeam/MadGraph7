# SPDX-License-Identifier: 0BSD
"""Pole selection of the v2 virtual: ``poles = "all" | "none" | "uv" | "ir"``
(design §12.4; ``loops.v2.poles``).

Checks, one phase-space point per process (HANDOFF §0.3):

* the finite part of every component is identical, bit for bit, in all four
  modes (native binary64, DoubleFloat and, for g g > g g, MPFR at 32
  digits), and ``"all"`` is the default;
* the UV and IR parts of every component and of the virtual add up to the
  combined poles; the double pole is infrared; ``"uv"`` returns the bare
  loop's UV poles, which the counterterms' UV poles cancel, and ``"ir"`` the
  renormalised virtual's IR poles, those of ``"all"`` (user decision
  2026-10-23);
* the IR poles of the renormalised virtual equal Catani's operator, massless
  (g g > g g, u u~ > g g g, and g g > b b~ with a symbolic mass bound to 0 at
  rounded momenta) and massive (g g > t t~, u u~ > t t~), and the provider
  poles of g g > t t~;
* the ``"uv"`` values (the bare-loop UV poles) cancel the UV_local
  counterterm poles up to the textbook Feynman-gauge field renormalisation of
  the external legs, independently of the reduction: -(bare_UV + uv_local)
  = sum_legs Z_i^UV Born with Z_g^UV = (alpha_s/4pi)(5/3 C_A - 4/3 T_F n_f)
  and Z_q^UV = -(alpha_s/4pi) C_F (massless and massive quarks), with and
  without delta g_s (Drell-Yan), with four quarks and a massive pair. This
  is the falsifiable form of bare-UV + counterterm-UV = 0: the field
  factors' UV part in the result is fixed by UV finiteness (design §12.4);
* the Python reference path (``LoopSlice.evaluate``) and the native executor
  agree in split mode (also at massive external legs), and the stability
  estimates exist in every mode and carrier; in binary64 a rotation check
  brackets the "uv" and "ir" errors against DoubleFloat (the reduction check
  alone is blind to numerator errors of the "uv" UV residue).

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

from pyamplicol.config import PoleSelection
from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    LoopSlice,
    Precision,
    SliceRequest,
)
from pyamplicol.loops.v2.fields import FIELD_COMPONENT
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.poles import (
    pole_mode,
    reducer_poles,
    select_lane,
    select_sector,
    split_sector,
)
from pyamplicol.loops.v2.stability import (
    REDUCTION,
    ROTATION,
    Rotation,
    StabilityChecks,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
# Drell-Yan, the photon-only control (the Z excluded from the request).
DRELL_YAN_PARAMETERS = {"aS": "0.118", "aEWM1": "132.507"}
MODES = tuple(PoleSelection)
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


# name: process, Born orders, point, external legs (gluons, quarks)[,
# ignored particle ids, parameters]
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
    "uux_uux": (
        "u u~ > u u~",
        {"QCD": 2, "QED": 0},
        lambda: _legacy_point("uux_uux"),
        (0, 4),
    ),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        lambda: _required_point("uux_ddxg")[:2],
        (1, 4),
    ),
    "dy": (
        "u u~ > e- e+",
        {"QCD": 0, "QED": 2},
        lambda: _legacy_point("dy"),
        (0, 2),
        (23,),
        DRELL_YAN_PARAMETERS,
    ),
}


def _parameters(name: str) -> dict[str, str]:
    entry = PROCESSES[name]
    return entry[5] if len(entry) > 5 else PARAMETERS


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, tuple[LoopSlice, NativeExecutor]] = {}

    def build(name: str) -> tuple[LoopSlice, NativeExecutor]:
        if name not in built:
            process, born, _point, _legs, *rest = PROCESSES[name]
            ignored = rest[0] if rest else ()
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            slice_ = LoopSlice.generate(
                compiled,
                SliceRequest(
                    process,
                    born,
                    loop,
                    fixed_parameters=FIXED,
                    particle_veto=ignored,
                ),
                loop_native,
            )
            built[name] = slice_, NativeExecutor(slice_)
        return built[name]

    return build


@pytest.fixture(scope="module")
def modes(slices):
    """One point per process and carrier in every pole mode through the
    native executor (binary64, DoubleFloat, MPFR at 32 digits); ``None`` is
    the default (no ``poles`` argument)."""

    cache: dict = {}

    def evaluate(name: str, carrier: str = "double"):
        key = (name, carrier)
        if key not in cache:
            _slice, executor = slices(name)
            point, scale = PROCESSES[name][2]()
            precision = {
                "double": Precision.double,
                "extended": Precision.double_double,
                "arbitrary": lambda: Precision.arbitrary(32),
            }[carrier]()
            out = {}
            for mode in (None, *MODES):
                options = {} if mode is None else {"poles": mode}
                (out[mode],) = executor.evaluate(
                    [point],
                    [scale],
                    parameters=_parameters(name),
                    precision=precision,
                    **options,
                )
            cache[key] = out
        return cache[key]

    return evaluate


def _scale(laurent) -> float:
    return max(abs(complex(x)) for x in laurent.coefficients)


def _assert_selection(result) -> None:
    """The selected components of a "uv" or "ir" result
    (``poles.select_sector``): the origin's parts, except that the virtual's
    poles in "uv" are the bare loop's UV poles, with the renormalised
    virtual's finite part."""

    parts = result.pole_origins[result.poles]
    for component in COMPONENTS:
        if component != "virtual" or result.poles == "ir":
            assert result.components[component] == parts[component], component
    if result.poles == "uv":
        for sector, virtual in result.components["virtual"].items():
            bare = parts["bare"][sector]
            assert virtual.finite == parts["virtual"][sector].finite
            assert virtual.single_pole == bare.single_pole
            assert virtual.double_pole == bare.double_pole == 0


# ----------------------------------------------------------------- modes


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "carrier"),
    (
        ("gg_gg", "double"),
        ("gg_gg", "extended"),
        ("gg_gg", "arbitrary"),
        ("gg_ttx", "double"),
        ("gg_ttx", "extended"),
    ),
)
def test_finite_parts_are_identical_in_every_mode(modes, name, carrier):
    results = modes(name, carrier)
    default, combined = results[None], results["all"]
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
    assert uv.pole_origins == ir.pole_origins
    for result in (uv, ir):
        _assert_selection(result)
    origins = uv.pole_origins
    for component in COMPONENTS:
        for sector, whole in combined.components[component].items():
            u, i = origins["uv"][component][sector], origins["ir"][component][sector]
            scale = max(_scale(whole), _scale(combined.virtual(sector)))
            assert abs(u.single_pole + i.single_pole - whole.single_pole) <= (
                SPLIT_TOLERANCE * scale
            ), component
            assert u.double_pole == 0, component
            assert abs(i.double_pole - whole.double_pole) <= SPLIT_TOLERANCE * scale
    for sector, whole in combined.components["virtual"].items():
        scale = _scale(whole)
        # "ir": the renormalised virtual's IR poles, all of its poles.
        assert abs(ir.virtual(sector).single_pole - whole.single_pole) <= (
            SPLIT_TOLERANCE * scale
        )
        assert ir.virtual(sector).double_pole == whole.double_pole
        # "uv": the bare loop's UV poles; the counterterms' UV poles cancel
        # them (the renormalised virtual's UV part).
        bare_uv = uv.virtual(sector).single_pole
        counterterms = (
            uv.components["uv"][sector].single_pole
            + uv.components["r2"][sector].single_pole
        )
        assert abs(bare_uv) > 1e-3 * scale
        assert abs(bare_uv + counterterms) <= SPLIT_TOLERANCE * scale
        assert abs(origins["uv"]["virtual"][sector].single_pole) <= (
            SPLIT_TOLERANCE * scale
        )
        # UV_local and R2 poles are ultraviolet.
        for component in ("uv_local", "r2"):
            assert ir.components[component][sector].single_pole == 0
    # The bare loop has both kinds of poles (scaleless bubbles included).
    for sector in combined.components["bare"]:
        assert abs(uv.components["bare"][sector].single_pole) > 0
        assert abs(ir.components["bare"][sector].single_pole) > 0


@pytest.mark.loop_physics
@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_leading_colour_selections_split_the_projected_poles(
    loop_sm_compiled_model, loop_native, accuracy
):
    """LC and NLC (committed QCD continuation): the scaleless companion is
    pruned like the slice, the parts add up and the finite part is shared."""

    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "g g > g g",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters=FIXED,
            accuracy=accuracy,
            colour_declaration=declaration,
        ),
        loop_native,
    )
    assert slice_.scaleless is not None
    executor = NativeExecutor(slice_)
    point, scale = PROCESSES["gg_gg"][2]()
    results = {
        mode: executor.evaluate([point], [scale], parameters=PARAMETERS, poles=mode)[0]
        for mode in ("all", "uv", "ir")
    }
    combined = results["all"].components["virtual"]
    for mode in ("uv", "ir"):
        _assert_selection(results[mode])
    for sector, whole in combined.items():
        uv = results["uv"].virtual(sector)
        ir = results["ir"].virtual(sector)
        assert uv.finite == ir.finite == whole.finite
        renormalised = results["uv"].pole_origins["uv"]["virtual"][sector]
        assert abs(renormalised.single_pole) <= SPLIT_TOLERANCE * _scale(whole)
        assert abs(ir.single_pole - whole.single_pole) <= SPLIT_TOLERANCE * _scale(
            whole
        )
        bare = {m: results[m].components["bare"][sector] for m in results}
        assert abs(
            bare["uv"].single_pole + bare["ir"].single_pole - bare["all"].single_pole
        ) <= SPLIT_TOLERANCE * _scale(whole)


# ----------------------------------------------------------- observables


ORDERED = (((1, -1), (1, -2)), ((2, -1), (3, -2)))


def _observables(model, native, accuracy: str):
    """g g > g g with a dipole and a directed (not symmetric) colour history
    (``ObservableSlice.generate``)."""

    from pyamplicol.color.connections import EmitGluon
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.models.loading import load_compiled_model

    declaration = None
    if accuracy != "full":
        from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
        from tools.loop_reference import legacy_capture

        declaration = LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        )
    bra, ket = (tuple(EmitGluon(*step) for step in side) for side in ORDERED)
    histories = (ColorCorrelator("ordered", bra, ket),)
    if accuracy == "full":
        histories = (ColorCorrelator.dipole("T12", 1, 2), *histories)
    return ObservableSlice.generate(
        load_compiled_model(model),
        SliceRequest(
            "g g > g g",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters=FIXED,
            accuracy=accuracy,
            colour_declaration=declaration,
        ),
        native,
        correlators=CorrelatorConfig(color_correlations=histories),
    )


def _assert_split_outputs(results, scales):
    """Per output: the finite part is that of "all" bit for bit, the UV and
    IR parts of every component add up to the poles of "all", "uv" selects
    the output's bare-loop UV poles (the renormalised virtual has no UV
    pole) and "ir" the renormalised IR poles, those of "all" (``scales`` per
    output and sector)."""

    for key, whole in results["all"].items():
        uv, ir = results["uv"][key], results["ir"][key]
        assert uv.pole_origins == ir.pole_origins
        for result in (uv, ir):
            _assert_selection(result)
        for component in COMPONENTS:
            for sector, laurent in whole.components[component].items():
                bound = SPLIT_TOLERANCE * scales[key, sector]
                u, i = (
                    uv.pole_origins["uv"][component][sector],
                    ir.pole_origins["ir"][component][sector],
                )
                assert u.finite == i.finite == laurent.finite, (key, component)
                assert uv.components[component][sector].finite == laurent.finite
                for k in (1, 2):
                    total = u.coefficients[k] + i.coefficients[k]
                    assert abs(total - laurent.coefficients[k]) <= bound
        for sector, laurent in whole.components["virtual"].items():
            bound = SPLIT_TOLERANCE * scales[key, sector]
            renormalised = uv.pole_origins["uv"]["virtual"][sector]
            assert abs(renormalised.single_pole) <= bound
            for k in (1, 2):
                difference = (
                    ir.virtual(sector).coefficients[k] - laurent.coefficients[k]
                )
                assert abs(difference) <= bound


@pytest.mark.loop_physics
def test_correlated_and_resolved_outputs_split_their_poles(
    loop_sm_compiled_model, loop_native
):
    """Observable columns in "uv" and "ir" (full colour): the inclusive
    column, a dipole and a directed history, natively and on the Python
    path (agreeing to 1e-11), and the helicity-resolved outputs, whose sum
    is the inclusive column's split (both include the certified-scaleless
    companion's column plan, ``ColumnPlan.scaleless``)."""

    from pyamplicol.loops.v2.observables import INCLUSIVE, NativeObservables

    observables = _observables(loop_sm_compiled_model, loop_native, "full")
    point, scale = PROCESSES["gg_gg"][2]()
    histories = (INCLUSIVE, "T12", "ordered")
    native = NativeObservables.correlated(observables, histories, lanes=4)
    results = {
        mode: native.evaluate([point], [scale], parameters=PARAMETERS, poles=mode)[0]
        for mode in ("all", "uv", "ir")
    }
    scales = {
        (key, sector): _scale(result.virtual(sector))
        for key, result in results["all"].items()
        for sector in result.components["virtual"]
    }
    _assert_split_outputs(results, scales)
    python = observables.evaluate(
        point, scale, correlations=histories, parameters=PARAMETERS, poles="uv"
    )
    for key, result in results["uv"].items():
        for component in COMPONENTS:
            for sector, laurent in result.components[component].items():
                other = python[key].components[component][sector]
                for x, y in zip(laurent.coefficients, other.coefficients, strict=True):
                    assert abs(complex(x) - complex(y)) <= 1e-11 * scales[key, sector]
    resolved = NativeObservables.resolved(observables, "helicity", lanes=4)
    (entries,) = resolved.evaluate([point], [scale], parameters=PARAMETERS, poles="ir")
    inclusive = results["ir"][INCLUSIVE]
    for origin in ("uv", "ir"):
        for component in COMPONENTS:
            for sector, laurent in inclusive.pole_origins[origin][component].items():
                total = sum(
                    (
                        e.pole_origins[origin][component][sector]
                        for e in entries.values()
                    ),
                    start=type(laurent)(*[0.0] * 3),
                )
                for x, y in zip(total.coefficients, laurent.coefficients, strict=True):
                    assert abs(x - y) <= 1e-12 * scales[INCLUSIVE, sector]


@pytest.mark.loop_physics
def test_leading_colour_directed_history_splits_its_poles(
    loop_sm_compiled_model, loop_native
):
    """At LC a directed history needs the slice unpruned at generation; its
    certified-scaleless companion is rebuilt unpruned and pruned for the
    history (``ObservableSlice.scaleless``)."""

    from pyamplicol.loops.v2.observables import INCLUSIVE, NativeObservables

    observables = _observables(loop_sm_compiled_model, loop_native, "lc")
    companion = observables.scaleless()
    assert companion is not None and companion.slice is not observables.slice.scaleless
    point, scale = PROCESSES["gg_gg"][2]()
    native = NativeObservables.correlated(observables, (INCLUSIVE, "ordered"), lanes=4)
    results = {
        mode: native.evaluate([point], [scale], parameters=PARAMETERS, poles=mode)[0]
        for mode in ("all", "uv", "ir")
    }
    scales = {
        (key, sector): _scale(result.virtual(sector))
        for key, result in results["all"].items()
        for sector in result.components["virtual"]
    }
    _assert_split_outputs(results, scales)


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
@pytest.mark.parametrize(
    "name", ("gg_gg", "uux_ggg", "gg_ttx", "uux_ttx", "uux_uux", "dy")
)
def test_bare_uv_poles_cancel_the_counterterms_up_to_field_renormalisation(modes, name):
    """The "uv" values are the bare loop's UV poles. Independently of the
    reduction's residue, the UV_local counterterms and the textbook
    Feynman-gauge Z^UV of every external leg times the Born (the field factor
    multiplies the Born once per leg, delta-Z / 2 in the amplitude) cancel
    them: with delta g_s (gluons, four quarks), without it (Drell-Yan, the
    photon vertex), and with a massive pair (delta m_t, the top's field
    factor mixing UV and IR)."""

    gluons, quarks = PROCESSES[name][3]
    a = ALPHA_S / (4 * math.pi)
    z_gluon = a * (5 / 3 * CA - 4 / 3 * TF * LOOP_FLAVOURS)
    z_quark = -a * CF
    result = modes(name)["uv"]
    ((sector, virtual),) = result.components["virtual"].items()
    bare = result.pole_origins["uv"]["bare"][sector]
    assert (virtual.single_pole, virtual.double_pole) == (bare.single_pole, 0)
    local = result.components["uv_local"][sector]
    fields = result.components[FIELD_COMPONENT][sector]
    born = float(result.born[next(iter(result.born))])
    expected = (gluons * z_gluon + quarks * z_quark) * born
    bare_uv = float(virtual.single_pole.real)
    got = -(bare_uv + float(local.single_pole.real))
    assert abs(got - expected) <= POLE_TOLERANCE * abs(bare_uv)
    assert abs(float(fields.single_pole.real) - expected) <= POLE_TOLERANCE * abs(
        bare_uv
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


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", ("extended", "arbitrary"))
def test_split_estimates_in_the_extended_carriers(slices, carrier):
    """ "uv" and "ir" with a rotation and the rescaled reduction check in
    DoubleFloat and MPFR (32 digits): the estimates exist and bound the
    finite part and the selected single pole far below binary64."""

    _slice, executor = slices("gg_gg")
    point, scale = PROCESSES["gg_gg"][2]()
    precision = (
        Precision.double_double() if carrier == "extended" else Precision.arbitrary(32)
    )
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True, True)
    for mode in ("uv", "ir"):
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=PARAMETERS,
            precision=precision,
            poles=mode,
            stability=checks,
        )
        estimate = result.stability
        assert len(estimate.coefficients) == 3
        assert estimate.finite < 1e-20 and estimate.coefficients[1] < 1e-20, mode
    # The rotated point directly: the selected poles equal binary64's (the
    # MPFR residue once kept 2-3 bits of exact low-precision inputs there).
    rotated = [
        [str(x) for x in momentum]
        for momentum in checks.rotations[0].apply(
            tuple(tuple(Fraction(x) for x in momentum) for momentum in point)
        )
    ]
    for mode in ("uv", "ir"):
        (wide,) = executor.evaluate(
            [rotated], [scale], parameters=PARAMETERS, precision=precision, poles=mode
        )
        (narrow,) = executor.evaluate(
            [rotated], [scale], parameters=PARAMETERS, poles=mode
        )
        for sector, laurent in narrow.components["virtual"].items():
            other = wide.components["virtual"][sector]
            for k in (1, 2):
                difference = complex(other.coefficients[k]) - complex(
                    laurent.coefficients[k]
                )
                assert abs(difference) <= 1e-12 * _scale(laurent), (mode, k)


@pytest.mark.loop_physics
def test_a_rotation_brackets_the_split_poles_in_binary64(slices):
    """In "uv" the reduction-direction check is blind to numerator errors:
    the selected single pole is the bare loop's UV residue, an exact
    functional of each lane's coefficients that does not depend on the
    reduction frame, so re-reducing a lane in another frame sees only the
    residue's own rounding (design §12.4 "Carriers and estimates"). At this
    point the reduction check alone underestimates the binary64 error of the
    "uv" single pole by 7.7x (3.9x rescaled); a rotation re-evaluates the
    numerators and brackets it. So, at one point of u u~ > d d~ g, binary64
    with one rotation and the reduction check: the rotation's estimate alone
    and the combined estimate bound the true error of every coefficient of
    every sector, measured against DoubleFloat, in "uv" and "ir" (with no
    margin: the true error is at most E)."""

    _slice, executor = slices("uux_ddxg")
    point, scale = PROCESSES["uux_ddxg"][2]()
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
    for mode in ("uv", "ir"):
        (truth,) = executor.evaluate(
            [point],
            [scale],
            parameters=PARAMETERS,
            precision=Precision.double_double(),
            poles=mode,
        )
        (result,) = executor.evaluate(
            [point], [scale], parameters=PARAMETERS, poles=mode, stability=checks
        )
        combined = result.stability
        assert set(combined.checks) == {ROTATION.format(0), REDUCTION}
        rotation = combined.subset((ROTATION.format(0),))
        for sector, laurent in result.components["virtual"].items():
            exact = truth.components["virtual"][sector]
            for k in range(3):
                error = abs(
                    complex(laurent.coefficients[k]) - complex(exact.coefficients[k])
                )
                for name, estimate in (("rotation", rotation), ("combined", combined)):
                    bound = abs(float(estimate.absolute[sector].coefficients[k]))
                    assert error <= bound, (mode, name, sector, k, error, bound)
            if mode == "uv":
                assert laurent.double_pole == exact.double_pole == 0


# ----------------------------------------------------------------- unit


@pytest.mark.loop_fast
def test_pole_mode_options():
    """One spelling: a ``PoleSelection`` or its value; no bool."""

    assert [pole_mode(m.value) for m in MODES] == list(MODES)
    assert all(pole_mode(m) is m for m in MODES)
    for bad in ("both", True, False, None):
        with pytest.raises(ValueError, match="poles must be one of"):
            pole_mode(bad)
    assert [reducer_poles(m) for m in MODES] == ["all", "none", "split", "split"]


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
    # The sums: uv = uv_local + fields; virtual = bare + uv + r2, UV finite.
    assert parts["uv"]["virtual"] == Laurent(q(17, 2), q(0), q(0))
    assert parts["ir"]["virtual"] == Laurent(q(17, 2), q(4), q(-2))
    assert parts["uv"]["uv"] == Laurent(q(1, 2), q(-3), q(0))
    # "ir" selects the IR parts; "uv" the UV parts, with the bare loop's UV
    # poles as the virtual's (the unrenormalised amplitude).
    assert select_sector(parts, PoleSelection.IR, Laurent) == parts["ir"]
    uv = select_sector(parts, PoleSelection.UV, Laurent)
    assert uv["virtual"] == Laurent(q(17, 2), q(3), q(0))
    assert {k: v for k, v in uv.items() if k != "virtual"} == {
        k: v for k, v in parts["uv"].items() if k != "virtual"
    }


@pytest.mark.loop_fast
def test_regulated_counterterms_split_without_a_remainder():
    """A model with pole regulators (MS-POLES): the counterterm components'
    UV parts are given (the counterterm program at the IR regulator 0), so no
    component takes a remainder, bare + counterterm UV poles are a sum, not
    zero by construction, and the IR parts are the rest."""

    from pyamplicol.loops.v2.assemble import Laurent

    q = Fraction
    values = {
        "bare": Laurent(q(7), q(5), q(-2)),
        "r2": Laurent(q(1), q(0), q(0)),
        "uv_local": Laurent(q(0), q(-4), q(0)),
        FIELD_COMPONENT: Laurent(q(1, 2), q(0), q(0)),
    }
    # e.g. a massless delta-Z (epsUV - epsIR): UV part -1, IR part +1.
    given = {
        "r2": Laurent(q(9), q(0), q(0)),
        "uv_local": Laurent(q(9), q(-2), q(0)),
        FIELD_COMPONENT: Laurent(q(9), q(-1), q(0)),
    }
    bare_uv = Laurent(q(7), q(3), q(0))
    parts = split_sector(values, bare_uv, FIELD_COMPONENT, Laurent, given)
    for name, whole in values.items():
        u, i = parts["uv"][name], parts["ir"][name]
        # Every part keeps the combined finite part (not the given ones).
        assert u.finite == i.finite == whole.finite
        assert u.single_pole + i.single_pole == whole.single_pole
        assert u.double_pole + i.double_pole == whole.double_pole
    assert parts["uv"]["uv_local"] == Laurent(q(0), q(-2), q(0))
    assert parts["ir"]["uv_local"] == Laurent(q(0), q(-2), q(0))
    assert parts["uv"][FIELD_COMPONENT] == Laurent(q(1, 2), q(-1), q(0))
    assert parts["ir"][FIELD_COMPONENT] == Laurent(q(1, 2), q(1), q(0))
    # bare_UV + CT_UV = 3 - 2 - 1 + 0 = 0 here, from the parts themselves.
    assert parts["uv"]["virtual"] == Laurent(q(17, 2), q(0), q(0))
    # A UV part that does not cancel the bare one stays visible.
    leaky = dict(given, uv_local=Laurent(q(9), q(-1), q(0)))
    leaked = split_sector(values, bare_uv, FIELD_COMPONENT, Laurent, leaky)
    assert leaked["uv"]["virtual"].single_pole == 1
    assert leaked["ir"]["virtual"].single_pole == values["bare"].single_pole - 3 + (
        -4 + 1
    ) + (0 + 1)
    # Without regulators the remainder definition is unchanged.
    assert split_sector(values, bare_uv, FIELD_COMPONENT, Laurent)["uv"][
        "virtual"
    ] == Laurent(q(17, 2), q(0), q(0))


@pytest.mark.loop_fast
def test_pole_regulators_are_found_by_code_and_need_both_adjustable():
    """``poles.pole_regulators``: the real external parameters at UFO/SLHA
    (TECHNICAL, 1) and (TECHNICAL, 2), whatever their names; none when one
    is missing, duplicated, complex or fixed; the regulated point sets the IR
    one to 0 and keeps every other value."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.poles import (
        PoleRegulators,
        pole_regulators,
        regulated_parameters,
    )

    def record(name, block, code, nature="external", kind="real"):
        return SimpleNamespace(
            name=name,
            nature=nature,
            parameter_type=kind,
            lhablock=block,
            lhacode=(code,),
        )

    model = [
        record("MU_R", "LOOP", 666),
        record("xUV", "technical", 1),
        record("xIR", "TECHNICAL", 2),
        record("LogSwitch", "TECHNICAL", 3),
        record("ee", None, 0, nature="internal"),
    ]
    found = pole_regulators(model)
    assert found == PoleRegulators("xUV", "xIR")
    assert PoleRegulators.from_record(found.record()) == found
    assert PoleRegulators.from_record(None) is None
    assert pole_regulators(model, fixed={"xIR": "1"}) is None
    assert pole_regulators(model[:2]) is None
    assert pole_regulators([*model, record("yIR", "TECHNICAL", 2)]) is None
    assert (
        pole_regulators([model[1], record("xIR", "TECHNICAL", 2, kind="complex")])
        is None
    )
    assert pole_regulators([record("MU_R", "LOOP", 1)]) is None
    values = {"xUV": "1.5", "xIR": "1", "MU_R": "91"}
    assert regulated_parameters(values, found) == {
        "xUV": "1.5",
        "xIR": "0",
        "MU_R": "91",
    }
    assert values["xIR"] == "1"


@pytest.mark.loop_fast
def test_regulated_points_need_the_regulator_bound():
    """``plan_export._regulated``: the native lanes' points at the IR
    regulator 0 (shallow copies, the originals unchanged) fail closed when
    the tree program does not bind the regulator (it would change nothing,
    so the counterterms' UV part would silently be all of them)."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.plan_export import _regulated
    from pyamplicol.loops.v2.poles import PoleRegulators

    trees = SimpleNamespace(program={"parameters": {"names": ["xUV", "xIR", "MU_R"]}})
    point = SimpleNamespace(parameters={"xIR": "1", "MU_R": "91"}, scale=2)
    (twin,) = _regulated([point], PoleRegulators("xUV", "xIR"), trees)
    assert twin.parameters == {"xIR": "0", "MU_R": "91"} and twin.scale == 2
    assert point.parameters["xIR"] == "1"
    with pytest.raises(ValueError, match="not a parameter of the tree program"):
        _regulated([point], PoleRegulators("xUV", "yIR"), trees)


@pytest.mark.loop_fast
def test_pole_regulators_are_read_only_by_pole_coefficients():
    """``poles.model_pole_regulators``, the structural guard: the parameters
    at the regulator codes count only when nothing but the pole coefficients
    of Laurent values reads them (a model using those codes otherwise has
    none, so no lane sets them to 0); ``pole_regulators`` with the raw
    model requires the same pair."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.poles import (
        PoleRegulators,
        model_pole_regulators,
        pole_regulators,
    )

    def raw(parameter="xUV", coupling="g", finite="g"):
        values = {
            ("parameters", 0): {"kind": "expression", "expression": "1.0"},
            ("parameters", 1): {"kind": "expression", "expression": "1.0"},
            ("parameters", 2): {"kind": "expression", "expression": f"2*{parameter}"},
            ("counterterm_parameters", 0): {
                "kind": "laurent",
                "terms": ((-1, "g*(xUV - xIR)"), (0, finite)),
            },
            ("couplings", 0): {"kind": "expression", "expression": coupling},
        }

        def external(index, name, code):
            fields = {"nature": "external", "type": "real", "lhablock": "TECHNICAL"}
            return {"id": index, "name": name, "fields": {**fields, "lhacode": [code]}}

        collections = {
            "parameters": [
                external(0, "xUV", 1),
                external(1, "xIR", 2),
                {"id": 2, "name": "g", "fields": {"nature": "internal"}},
            ],
            "counterterm_parameters": [{"id": 0, "name": "dZ", "fields": {}}],
            "couplings": [{"id": 0, "name": "GC", "fields": {}}],
        }
        return SimpleNamespace(
            collections=collections,
            exact_value=lambda collection, index: values[collection, index],
        )

    found = PoleRegulators("xUV", "xIR")
    assert model_pole_regulators(raw(parameter="1")) == found
    assert model_pole_regulators(None) is None
    # A tree parameter, a coupling or a finite part reads one: none.
    assert model_pole_regulators(raw()) is None
    assert model_pole_regulators(raw(parameter="1", coupling="xIR*g")) is None
    assert model_pole_regulators(raw(parameter="1", finite="xIR")) is None
    # Attribute access and longer names are not reads.
    assert model_pole_regulators(raw(parameter="1", coupling="a.xIR + xIRb")) == found
    compiled = [
        SimpleNamespace(
            name=name,
            nature="external",
            parameter_type="real",
            lhablock="TECHNICAL",
            lhacode=(code,),
        )
        for name, code in (("xUV", 1), ("xIR", 2))
    ]
    assert pole_regulators(compiled, (), raw(parameter="1")) == found
    assert pole_regulators(compiled, (), raw()) is None


@pytest.mark.loop_fast
def test_catani_takes_the_masses_from_the_model():
    """The legs' masses are the model's runtime values, not the p^2 of the
    given momenta: with rounded massless momenta (p^2 = -1e-17 relative) the
    prediction is the massless one. No prediction (None) at a vanishing
    invariant or a massive pair at threshold; tiny masses do not cancel in
    1 - v."""

    from pyamplicol.loops.v2.poles import catani_poles

    q = Fraction
    exact = {
        1: (q(1), q(0), q(0), q(1)),
        2: (q(1), q(0), q(0), q(-1)),
        3: (q(-1), q(0), q(3, 5), q(4, 5)),
        4: (q(-1), q(0), q(-3, 5), q(-4, 5)),
    }
    rounded = dict(exact)
    rounded[3] = (q(-1), q(0), q(3, 5), q(4, 5) + q(1, 10**17))
    massless = dict.fromkeys(exact, q(0))

    def poles(momenta, masses):
        return catani_poles(
            (1, 2, 3, 4),
            {1: 8, 2: 8, 3: 8, 4: 8},
            momenta,
            masses,
            lambda j, k: -1.5,
            casimir=lambda r: q(3),
            gamma=lambda j: q(11, 2) - q(1, 3) * 5,
            mu2=1.0,
            log=math.log,
            real=float,
            sqrt=math.sqrt,
        )

    expected = poles(exact, massless)
    got = poles(rounded, massless)
    assert all(abs(a - b) <= 1e-12 * abs(b) for a, b in zip(got, expected, strict=True))
    tiny = {**massless, 3: q(1, 10**30), 4: q(1, 10**30)}
    assert all(math.isfinite(x) for x in poles(rounded, tiny))
    collinear = {**exact, 4: (q(-1), q(0), q(0), q(-1))}
    assert poles(collinear, massless) is None
    # (p_3.p_4)^2 = 4 <= m_3^2 m_4^2 = 100.
    assert poles(exact, {**massless, 3: q(10), 4: q(10)}) is None


@pytest.mark.loop_physics
def test_symbolic_mass_bound_to_zero_at_rounded_momenta(
    loop_sm_compiled_model, loop_native
):
    """g g > b b~ with MB left symbolic and bound to 0 at runtime, at rounded
    massless momenta (p_b^2 = O(1e-12)): every mode evaluates (Catani took
    p^2 for the mass and raised), and Catani's operator with the runtime
    masses equals the virtual's poles. Natively in binary64 and on the
    Python path."""

    from pyamplicol.models.loading import load_compiled_model

    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "g g > b b~",
            {"QCD": 2, "QED": 0},
            {"QCD": 4, "QED": 0},
            fixed_parameters={"WT": "0"},
            # The CT recursion's lowered couplings refuse a symbolic mass
            # bound to 0 (ct_recursion: they assume it non-zero); the site
            # closure binds it at runtime.
            counterterms="sites",
        ),
        loop_native,
    )
    theta, phi = 2.3, 0.3
    x = 500 * math.sin(theta) * math.cos(phi)
    y = 500 * math.sin(theta) * math.sin(phi)
    z = 500 * math.cos(theta)
    point = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", repr(x), repr(y), repr(z)],
        ["500", repr(-x), repr(-y), repr(-z)],
    ]
    p3 = [Fraction(v) for v in point[2]]
    assert p3[0] ** 2 - p3[1] ** 2 - p3[2] ** 2 - p3[3] ** 2 != 0
    parameters = {**PARAMETERS, "MB": "0"}
    executor = NativeExecutor(slice_)
    results = {
        mode: executor.evaluate(
            [point], ["1000000"], parameters=parameters, poles=mode
        )[0]
        for mode in MODES
    }
    python = slice_.evaluate(point, "1000000", parameters=parameters, poles="all")
    for result in (results["all"], results["ir"], python):
        ((sector, virtual),) = result.components["virtual"].items()
        single, double = _catani(result, sector)
        scale = max(abs(single), abs(double))
        assert abs(float(virtual.single_pole.real) - single) <= POLE_TOLERANCE * scale
        assert abs(float(virtual.double_pole.real) - double) <= POLE_TOLERANCE * scale
    for mode in MODES:
        for component in COMPONENTS:
            for sector, laurent in results["all"].components[component].items():
                other = results[mode].components[component][sector]
                assert other.finite == laurent.finite, (mode, component)


@pytest.mark.loop_physics
def test_native_executor_at_massive_external_legs(slices):
    """g g > t t~: the native executor equals the Python reference path (which
    equals MadLoop), combined and split. Before the slice's mass-flipped
    kernel copies (``reference_executor.mass_flips``, WP2.6) the native
    family integrals of top lanes differed (double pole -0.0666 against
    -0.1133)."""

    slice_, executor = slices("gg_ttx")
    point, scale = PROCESSES["gg_ttx"][2]()
    for mode in ("all", "ir"):
        python = slice_.evaluate(point, scale, parameters=PARAMETERS, poles=mode)
        (native,) = executor.evaluate(
            [point], [scale], parameters=PARAMETERS, poles=mode
        )
        for component in COMPONENTS:
            for sector, laurent in python.components[component].items():
                other = native.components[component][sector]
                bound = 1e-12 * _scale(python.virtual(sector)) + 1e-12 * _scale(laurent)
                for x, y in zip(laurent.coefficients, other.coefficients, strict=True):
                    assert abs(complex(x) - complex(y)) <= bound, (mode, component)
