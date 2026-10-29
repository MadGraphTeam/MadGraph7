# SPDX-License-Identifier: 0BSD
"""Complex pole masses in the native lane (M-SM MS-CMS-N).

A complex-mass scheme gives every particle with a width the pole mass
``CMASS_M = sqrt(M^2 - i M Gamma)`` (MG5_aMC's ``Model.
change_mass_to_complex_scheme``), an internal parameter of complex type that
its propagators read with width zero. The native trees take its components
like any derived parameter's; the loop propagators take it as a
``["complex", o]`` plan mass of the parameter program (``native_trees.
_parameter_program``), bound to the square of the carrier value's exact
complex rational (``loop_v2::trees::TreeProgram::bind``,
``BoundParameters::masses_imaginary``), and the reduction lanes run it
complex. External states stay stable (real masses).

The fixture is loop_sm with the top's complex pole mass
(``loop_v2_ufo_variants.complex_mass_ufo``): ``b b~ > w+ w-`` [QCD] with the
top in the Born's t channel and in the loops, the counterterms on the real
MT ("plain") or continued to CMASS_MT as MG5_aMC does ("continued", the
scheme of loop_sm with a top width only), and ``g g > g g`` [QCD] with top
loops on the continued model for the Ward identity:

* the record holds the complex plan mass, unchecked for positivity, and
  loop_sm's own record none;
* at zero width (CMASS_MT = MT) native inputs equal Python inputs and the
  bound masses equal the Python lane's, exactly;
* at the card's width the bound squared mass is M^2 - i M Gamma to the
  carrier's rounding in binary64, double-double and MPFR, and the carriers
  agree;
* the Python-input batch reads ``[re, im]`` squared masses as the texts;
* the gluon Ward identity holds with the complex top in the loops
  (double-double and binary64; gauge invariance of the complex-mass loops,
  not a test of their mass value);
* CM10 (``u d~ > w+ b b~ QCD=0`` [QCD] on loop_sm under the complex-mass
  scheme with the top width, MS-CMS's ``model_cms.complex_mass_model``)
  equals MadLoop CMS with native inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import plan_export, ward
from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

from .loop_v2_ufo_variants import complex_mass_ufo

pytestmark = pytest.mark.loop_physics

PROCESS = ("b b~ > w+ w-", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2})
GLUONS = ("g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0})
FIXED = {"WZ": "0", "WW": "0", "WH": "0", "MB": "0"}
SCALE = "62500"
ZERO_WIDTH = {"WT": "0"}
CARRIERS = {
    "double": (Precision.double(), 1e-15),
    "double-double": (Precision.double_double(), 1e-30),
    "arbitrary": (Precision.arbitrary(32), 1e-30),
}


@pytest.fixture(scope="module")
def models(loop_sm_ufo, tmp_path_factory):
    """``model(kind)``: loop_sm (``"loop_sm"``) or its complex top pole mass
    fixture (``"plain"``, ``"continued"``), compiled as
    ``loop_sm_compiled_model`` is."""

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("complex-top")
    built: dict = {}

    def model(kind: str):
        if kind not in built:
            ufo = loop_sm_ufo
            if kind != "loop_sm":
                ufo = complex_mass_ufo(
                    loop_sm_ufo,
                    root / f"loop_sm_{kind}",
                    "t",
                    "MT",
                    "WT",
                    references=kind == "continued",
                )
            compiled = compile_model_source(ufo, cache_dir=root / f"cache-{kind}")
            built[kind] = replace(
                compiled,
                raw_loop_model=prepare_loop_sm_scalar_declarations(
                    compiled.raw_loop_model
                ).derived,
                _serialized_path=None,
                _prepared_bundle=None,
            )
        return built[kind]

    return model


@pytest.fixture(scope="module")
def slices(models, loop_native):
    built: dict = {}

    def slice_(kind: str, process=PROCESS, generate=LoopSlice.generate):
        key = (kind, process[0], generate)
        if key not in built:
            text, born, loop = process
            built[key] = generate(
                models(kind),
                SliceRequest(text, born, loop, fixed_parameters=FIXED),
                loop_native,
            )
        return built[key]

    return slice_


def _point(slice_) -> list[list[str]]:
    """b b~ > w+ w- at sqrt(s) = 500 GeV, the W's on the model's shell."""

    from pyamplicol.loops.v2.assemble import _model_masses

    mw = float(_model_masses(slice_.compiled, {}, 40)("MW"))
    energy = 250.0
    size = math.sqrt(energy * energy - mw * mw)
    legs = [
        (energy, 0.0, 0.0, energy),
        (energy, 0.0, 0.0, -energy),
        (energy, 0.8 * size, 0.0, 0.6 * size),
        (energy, -0.8 * size, 0.0, -0.6 * size),
    ]
    return [[repr(x) for x in leg] for leg in legs]


def _top_mass(executor) -> int:
    """The plan mass id of the top's loop propagators."""

    (index,) = [
        k
        for k, mass in enumerate(executor.exported.masses)
        if not mass.massless and mass.mass.endswith("MT")
    ]
    return index


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_a_complex_pole_mass_is_a_complex_plan_mass(slices):
    """The top's loop propagators read CMASS_MT as a complex plan mass
    (``["complex", o]``, o not checked positive); loop_sm's record has none
    (non-CMS records are unchanged: every plan mass an input index)."""

    executor = NativeExecutor(slices("plain"), inputs="native")
    record = executor.exported.trees.program["parameters"]
    complex_masses = [m for m in record["plan_masses"] if isinstance(m, list)]
    assert complex_masses and all(kind == "complex" for kind, _o in complex_masses)
    assert not {o for _kind, o in complex_masses} & set(record["positive"])
    assert executor.exported.masses[_top_mass(executor)].mass == "CMASS_MT"
    plain = NativeExecutor(slices("loop_sm"), inputs="native")
    record = plain.exported.trees.program["parameters"]
    assert not any(isinstance(m, list) for m in record["plan_masses"])


def test_zero_width_native_inputs_equal_python_inputs(slices):
    """At zero width CMASS_MT = MT: the native lane (complex plan mass)
    binds the Python lane's squared masses exactly and its virtual equals
    the Python-input lane's in binary64 (both through the native executor),
    and loop_sm's (MT real) to its rounding."""

    slice_ = slices("plain")
    point = _point(slice_)
    native = NativeExecutor(slice_, inputs="native")
    python = NativeExecutor(slice_, inputs="python")
    masses, _legs = native.bound_masses([point], [SCALE], ZERO_WIDTH)
    mass = Fraction(str(slice_.compiled.parameter_defaults["MT"][0]))
    assert masses[_top_mass(native)] == mass**2
    (mine,) = native.evaluate([point], [SCALE], parameters=ZERO_WIDTH)
    (theirs,) = python.evaluate([point], [SCALE], parameters=ZERO_WIDTH)
    reference = NativeExecutor(slices("loop_sm"), inputs="native")
    (real,) = reference.evaluate([point], [SCALE], parameters=ZERO_WIDTH)
    for k in range(4):
        scale = abs(float(theirs.totals[0])) if k == 0 else abs(float(theirs.totals[1]))
        assert abs(float(mine.totals[k]) - float(theirs.totals[k])) <= 1e-12 * scale
        assert abs(float(real.totals[k]) - float(theirs.totals[k])) <= 1e-12 * scale


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_the_pole_mass_square_is_bound_complex(slices, carrier):
    """At the card's width the top's bound squared mass is the square of the
    carrier's CMASS_MT, M^2 - i M Gamma to the carrier's rounding (the
    imaginary part negative: the propagator's causal side), while the legs'
    stay real."""

    slice_ = slices("plain")
    native = NativeExecutor(slice_, inputs="native")
    precision, rounding = CARRIERS[carrier]
    defaults = slice_.compiled.parameter_defaults
    mass, width = (Fraction(str(defaults[name][0])) for name in ("MT", "WT"))
    assert width > 0
    masses, legs = native.bound_masses([_point(slice_)], [SCALE], None, precision)
    square = masses[_top_mass(native)]
    assert isinstance(square, tuple)
    assert abs(square[0] - mass**2) <= Fraction(rounding) * mass**2
    assert abs(square[1] + mass * width) <= Fraction(rounding) * mass**2
    assert square[1] < 0
    assert all(isinstance(leg, Fraction) for leg in legs)


def test_complex_pole_masses_agree_across_carriers(slices):
    """The continued fixture at the card's width: the native binary64 and
    double-double virtuals agree (binary64 within 1e-9 of the virtual), and
    the width moves the virtual off its zero-width value."""

    slice_ = slices("continued")
    native = NativeExecutor(slice_, inputs="native")
    point = _point(slice_)
    (double,) = native.evaluate([point], [SCALE])
    (dd,) = native.evaluate([point], [SCALE], precision=Precision.double_double())
    (zero,) = native.evaluate([point], [SCALE], parameters=ZERO_WIDTH)
    for k in range(4):
        assert _relative(float(double.totals[k]), float(dd.totals[k])) <= 1e-9, k
    assert _relative(float(zero.totals[1]), float(dd.totals[1])) > 1e-6


def test_python_input_batches_read_complex_squared_masses(slices, monkeypatch):
    """The executor's Python-input batches take a squared mass as a text or
    an ``[re, im]`` pair (``plan_export._mass_text``): pairs with a zero
    imaginary part give the texts' result bitwise, a nonzero one on the
    massive lines moves the virtual, not the Born."""

    slice_ = slices("plain")
    point = _point(slice_)
    (texts,) = NativeExecutor(slice_, inputs="python").evaluate(
        [point], [SCALE], parameters=ZERO_WIDTH
    )
    text = plan_export._mass_text

    def pair(imaginary):
        def mass_text(value):
            # A massive line's squared mass moved off the real axis.
            return [text(value), imaginary if value else "0"]

        return mass_text

    monkeypatch.setattr(plan_export, "_mass_text", pair("0"))
    (pairs,) = NativeExecutor(slice_, inputs="python").evaluate(
        [point], [SCALE], parameters=ZERO_WIDTH
    )
    assert [float(x) for x in pairs.totals] == [float(x) for x in texts.totals]
    monkeypatch.setattr(plan_export, "_mass_text", pair("-260"))
    (moved,) = NativeExecutor(slice_, inputs="python").evaluate(
        [point], [SCALE], parameters=ZERO_WIDTH
    )
    assert float(moved.totals[1]) != float(texts.totals[1])
    assert float(moved.totals[0]) == float(texts.totals[0])


def test_gluon_ward_identity_with_a_complex_top_pole_mass(slices):
    """g g > g g [QCD] with complex top loops on the continued fixture at the
    card's width: every gluon leg's Ward identity holds in double-double
    (``ward.ward_check``) and binary64 (its O1 rule). The bare loops vanish
    on their own (as at real masses), so this checks the complex-mass loops'
    gauge invariance, not their mass value."""

    slice_ = slices("continued", GLUONS, ward.ward_slice)
    executor = NativeExecutor(slice_, inputs="native", forward=False)
    point = [
        ["250.0", "0.0", "0.0", "250.0"],
        ["250.0", "0.0", "0.0", "-250.0"],
        ["250.0", "150.0", "0.0", "200.0"],
        ["250.0", "-150.0", "0.0", "-200.0"],
    ]
    for carrier in ("double", "double-double"):
        check = ward.ward_check(executor, point, SCALE, carrier)
        assert check["passed"] is True, (carrier, check["rows"])


# ------------------------------------------------------------ MadLoop CMS rows

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/loop/required/msm"


def _madloop_rows(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _laurent_errors(result, row, coupling: float) -> list[float]:
    """Born and the virtual's (finite, 1/eps, 1/eps^2) against a MadLoop row
    in the campaign metric: relative, a Laurent coefficient against
    max(|its value|, |Born| coupling / 2 pi)."""

    mine = [float(x) for x in result.totals[:4]]
    theirs = [row["born"], row["finite"], row["single_pole"], row["double_pole"]]
    flux = abs(theirs[0]) * coupling / (2 * math.pi)
    return [
        abs(a - b) / max(abs(b), flux if k else 0.0)
        for k, (a, b) in enumerate(zip(mine, theirs, strict=True))
    ]


def test_cms_top_width_equals_madloop(loop_sm_ufo, loop_native, tmp_path):
    """CM10: u d~ > w+ b b~ QCD=0 [QCD] on loop_sm (no_b_mass) under the
    complex-mass scheme with the top width (``model_cms.complex_mass_model``,
    the port of MG5_aMC's transform; the resonant top in the Born, complex
    top loops and counterterms) with native inputs equals MadLoop CMS in
    quadruple precision (``required/msm/udx_wpbbx_cms.json``) at its four
    points: double-double to 1e-12 (measured 7e-14), binary64 to 1e-8
    (measured 3.4e-10)."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    reference = _madloop_rows("udx_wpbbx_cms")
    compiled = compile_model_source(
        loop_sm_ufo, restriction=reference["restriction"], cache_dir=tmp_path
    )
    compiled = replace(
        compiled,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            compiled.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    fixed = reference["fixed_parameters"]
    model, scheme = complex_mass_model(
        compiled,
        fixed_parameters={k: v for k, v in fixed.items() if k.startswith("W")},
        cache_dir=tmp_path,
    )
    assert dict(scheme.masses) == {"MT": "CMASS_MT"}
    fixed = {k: v for k, v in fixed.items() if k in model.parameter_defaults}
    slice_ = LoopSlice.generate(
        model,
        SliceRequest(
            "u d~ > w+ b b~",
            reference["born_orders"],
            reference["loop_orders"],
            fixed_parameters=fixed,
        ),
        loop_native,
    )
    executor = NativeExecutor(slice_, inputs="native")
    record = executor.exported.trees.program["parameters"]
    assert any(isinstance(mass, list) for mass in record["plan_masses"])
    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in fixed
    }
    scales = [str(Fraction(mu) ** 2) for mu in reference["mu_r"]]
    for precision, tolerance in (
        (Precision.double_double(), 1e-12),
        (Precision.double(), 1e-8),
    ):
        results = executor.evaluate(
            reference["points"], scales, parameters=parameters, precision=precision
        )
        for k, (result, row) in enumerate(zip(results, reference["rows"], strict=True)):
            errors = _laurent_errors(result, row, float(parameters["aS"]))
            assert max(errors) <= tolerance, (precision.arithmetic, k, errors)
