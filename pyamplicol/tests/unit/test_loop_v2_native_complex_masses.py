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
* at the card's width native inputs equal the Python inputs (continued
  fixture);
* CM10 (``u d~ > w+ b b~ QCD=0`` [QCD] on loop_sm under the complex-mass
  scheme with the top width, MS-CMS's ``model_cms.complex_mass_model``)
  equals MadLoop CMS with native inputs, and through the public route
  (``mass_scheme = "complex-mass"``) its artifact stores native inputs,
  relocates bitwise and equals MadLoop CMS;
* nightly: CM08 (``u d~ > e+ ve b b~`` [QCD], every width) and CM09 (``u d~ >
  e+ ve a`` [QCD QED], a tagged photon, MG5_aMC's formula with the real part
  of the complex conversion) on the G_mu UFO equal MadLoop CMS with native
  inputs, and CM07 (the [U4] acceptance process, Gamma_t = 0: the top on its
  line mass, ``model_cms`` step 4b) does too.
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


# Complex-mass outputs of a parameter program on the inputs (M, W, MZ, WZ):
# the pole mass, its square, a complex-mass W coupling, a logarithm of a
# complex square and the real part of a complex-mass alpha (u: the
# imaginary unit).
POLE_OUTPUTS = (
    "(x^2 - u*x*y)^(1/2)",
    "x^2 - u*x*y",
    "(1 - (x^2 - u*x*y)/(z^2 - u*z*w))^(-1/2)",
    "log(x^2 - u*x*y)",
    "abs(x^2 - u*x*y)*(1 - abs((x^2 - u*x*y)/(z^2 - u*z*w)))",
)


@pytest.mark.loop_fast
def test_plane_check_takes_complex_mass_outputs(loop_native):
    """The SymJIT plane check (``compile_v2_plane``) on complex-mass outputs
    (``POLE_OUTPUTS``) at the physical point (MW = 80.419, WW = 2.09291,
    MZ = 91.188, WZ = 2.49877): accepted, the plane application equals the
    exact evaluator to 1e-14 and the complex outputs keep their imaginary
    parts (the causal side, Im(M^2) = -M W < 0)."""

    import numpy as np
    from symbolica import E, Evaluator, Expression, S

    from pyamplicol import _rusticol

    names = ("x", "y", "z", "w")
    state = bytes(
        Expression.evaluator_multiple(
            [E(text).replace(S("u"), Expression.I) for text in POLE_OUTPUTS],
            [S(name) for name in names],
            jit_compile=False,
        ).save()
    )
    point = [80.419, 2.09291, 91.188, 2.49877]
    plane, inputs, outputs = loop_native.compile_v2_plane(state, 2, False, [point])
    assert (inputs, outputs) == (4, len(POLE_OUTPUTS))
    rows = np.ascontiguousarray([[complex(x) for x in point]], dtype=np.complex128)
    expected = np.asarray(Evaluator.load(state).evaluate_complex(rows.tolist()))
    flat = _rusticol._evaluate_symjit_plane_application_v2(
        bytes(plane), inputs, outputs, rows.view(np.float64).ravel().tolist(), 1
    )
    got = np.asarray(flat, dtype=np.float64).view(np.complex128).reshape(1, -1)
    assert np.allclose(got, expected, rtol=1e-14, atol=0), (got, expected)
    assert got[0, 1].imag == pytest.approx(-point[0] * point[1], rel=1e-14)
    assert got[0, 0].imag < 0 and got[0, 4].imag == 0


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


@pytest.mark.parametrize(
    ("carrier", "tolerance"), [("double", 1e-12), ("double-double", 1e-24)]
)
def test_native_inputs_equal_python_inputs_at_the_width(slices, carrier, tolerance):
    """The continued fixture at the card's top width: the native lane (the
    parameter program binds CMASS_MT, its square and every complex coupling)
    and the Python-input lane (``_mass_values`` with the pole masses: an
    ``[re, im]`` squared mass, complex kernel constants) on the same plan give
    Born and every component within ``tolerance`` of the sector's virtual
    (measured: binary64 1.8e-14, the parameter program's binary64 rounding;
    double-double bitwise)."""

    slice_ = slices("continued")
    point = _point(slice_)
    native = NativeExecutor(slice_, inputs="native", forward=False)
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    precision = CARRIERS[carrier][0]
    (mine,) = native.evaluate([point], [SCALE], precision=precision)
    (theirs,) = python.evaluate([point], [SCALE], precision=precision)
    for order, value in theirs.born.items():
        assert _relative(float(mine.born[order]), float(value)) <= tolerance
    assert mine.components.keys() == theirs.components.keys()
    for component, sectors in theirs.components.items():
        for sector, laurent in sectors.items():
            size = max(
                abs(complex(x))
                for x in theirs.components["virtual"][sector].coefficients
            )
            for x, y in zip(
                mine.components[component][sector].coefficients,
                laurent.coefficients,
                strict=True,
            ):
                assert abs(complex(x) - complex(y)) <= tolerance * size, (
                    component,
                    sector,
                )


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


_CMS_CARD = """
action = "generate"
[model]
source = "loop_sm"
restriction = "{restriction}"
[process]
entries = [{{expression = "{expression}", name = "{name}"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "explicit"
[process.max_coupling_orders]
{orders}
[loop]
r2_strategy = "model"
[loop.model.scheme]
mass_scheme = "complex-mass"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def test_cms_artifact_stores_native_inputs_relocates_and_equals_madloop(tmp_path):
    """The public route under the complex-mass scheme (``[loop.model.scheme]
    mass_scheme = "complex-mass"`` on the shipped loop_sm, CM10's request:
    u d~ > w+ b b~ QCD=0 [QCD] with the top width): the stored process
    declares native point inputs (``point_inputs`` "native-v2"), its tree
    program binds the top's loop propagators as a complex plan mass, the
    artifact-portability check passes (no host strings, storage-v4 plane
    applications only, bitwise-identical binary64 results at a relocated copy
    evaluated without a licence), and the loaded artifact equals MadLoop CMS
    (``required/msm/udx_wpbbx_cms.json``) at its first point: binary64 to
    1e-8, double-double to 1e-12."""

    import zlib

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS
    from tools import loop_artifact_portability as portability

    reference = _madloop_rows("udx_wpbbx_cms")
    expression, name = "u d~ > w+ b b~", "udx_wpbbx"
    output = tmp_path / "artifact"
    card = tmp_path / "card.toml"
    card.write_text(
        _CMS_CARD.format(
            restriction=reference["restriction"],
            expression=expression,
            name=name,
            orders="\n".join(
                f"{order} = {power}"
                for order, power in sorted(reference["born_orders"].items())
            ),
            fixed=", ".join(
                f'{k} = "{v}"' for k, v in reference["fixed_parameters"].items()
            ),
            output=json.dumps(str(output)),
        )
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(expression, name=name),)), output
    )
    declared = [
        json.loads(path.read_text())["point_inputs"]
        for path in output.rglob("execution.json")
    ]
    assert declared == [LOOP_V2_NATIVE_POINT_INPUTS], declared
    (pacbin,) = output.rglob("*.pacbin")
    members = {path: data for path, _kind, data in portability._members(pacbin)}
    program = json.loads(
        zlib.decompress(members["point-inputs/tree-program.json.zlib"])
    )
    plan_masses = program["parameters"]["plan_masses"]
    assert [m[0] for m in plan_masses if isinstance(m, list)] == ["complex"]
    verdict = portability.check(
        output, name, tmp_path / "check", count=2, licence=False
    )
    scan = verdict["scan"]
    assert scan["clean"], scan["findings"]
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]
    runtime = Runtime.load(output, process=name)
    # The reference's exact decimal point (binary64 momenta would move the
    # resonant top's Born by ~1e-14 and the finite part by ~1e-12).
    point = reference["points"][0]
    scale = str(Fraction(reference["mu_r"][0]) ** 2)
    row = reference["rows"][0]
    for arithmetic, tolerance in (("double", 1e-8), ("double-double", 1e-12)):
        result = runtime.evaluate([point], mu_r2=[scale], arithmetic=arithmetic)
        ((born,),) = result.born_values
        ((laurent,),) = result.values
        theirs = [row["born"], row["finite"], row["single_pole"], row["double_pole"]]
        flux = abs(row["born"]) * float(reference["parameters"]["aS"]) / (2 * math.pi)
        mine = [float(born), *(float(x) for x in laurent)]
        errors = [
            abs(a - b) / max(abs(b), flux if k else 0.0)
            for k, (a, b) in enumerate(zip(mine, theirs, strict=True))
        ]
        assert max(errors) <= tolerance, (arithmetic, errors)


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


def _gmu_cms_slice(
    loop_gmu_ufo, loop_native, tmp_path, reference, process, corrections, loops=None
):
    """A G_mu UFO request under the complex-mass scheme at the reference's
    widths (``model_cms.complex_mass_model``, Feynman gauge) and its native
    executor: the reference's Born orders, one loop order per correction (or
    ``loops``)."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_gmu_ufo,
        restriction=reference["restriction"],
        cache_dir=tmp_path,
        require_supported=False,
        gauge="feynman",
    )
    fixed = reference["fixed_parameters"]
    model, _scheme = complex_mass_model(
        compiled,
        fixed_parameters={k: v for k, v in fixed.items() if k.startswith("W")},
        cache_dir=tmp_path,
    )
    fixed = {k: v for k, v in fixed.items() if k in model.parameter_defaults}
    born = {"QCD": 0, "QED": 0, **reference["born_orders"]}
    if loops is None:
        loops = tuple({**born, order: born[order] + 2} for order in sorted(corrections))
    slice_ = LoopSlice.generate(
        model,
        SliceRequest(
            process,
            reference["born_orders"],
            loops[0] if len(loops) == 1 else loops,
            corrections=tuple(sorted(corrections)),
            fixed_parameters=fixed,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in fixed
    }
    return NativeExecutor(slice_, inputs="native"), parameters


def _sector_errors(result, row, born_scale: float, want=None) -> list[float]:
    """Per loop sector of a MadLoop row (or ``want``: its converted values),
    (finite, 1/eps, 1/eps^2) in the campaign metric against max(|value|,
    |Born| alpha / 2 pi) with the correction's coupling (alpha_s for a QCD
    sector, alpha_Gmu ~ 1/132 for a QED one)."""

    # Sector keys without their zero powers (the engine's orders omit them).
    virtual = {
        tuple(
            sorted((name, power) for name, power in dict(sector).items() if power)
        ): laurent
        for sector, laurent in result.components["virtual"].items()
    }
    errors = []
    for k, sector in enumerate(row["sectors"]["loop"]):
        laurent = virtual[
            tuple((name, sector[name]) for name in ("QCD", "QED") if sector[name])
        ]
        values = (
            want[k]
            if want is not None
            else [sector["finite"], sector["single_pole"], sector["double_pole"]]
        )
        qcd = sector["QCD"] > sector["born_sector"]["QCD"]
        flux = born_scale * (0.118 if qcd else 1 / 132.0) / (2 * math.pi)
        errors += [
            abs(float(a) - b) / max(abs(b), flux)
            for a, b in zip(laurent.coefficients, values, strict=True)
        ]
    return errors


@pytest.mark.loop_nightly
def test_cms_widths_in_a_two_to_four_process_equal_madloop(
    loop_gmu_ufo, loop_native, tmp_path
):
    """CM08: u d~ > e+ ve b b~ QCD=0 [QCD] on the G_mu UFO under the
    complex-mass scheme with the card's top, W, Z and H widths (resonant W
    and top in the Born, complex masses in the loops and counterterms), native
    inputs: = MadLoop CMS in quadruple precision (``required/msm/
    udx_epvebbx_cms.json``) at its four points, double-double to 1e-10
    (measured 1.3e-14), binary64 to 1e-8 (1.2e-13). Generation about 10 min,
    25 GB."""

    reference = _madloop_rows("udx_epvebbx_cms")
    executor, parameters = _gmu_cms_slice(
        loop_gmu_ufo, loop_native, tmp_path, reference, "u d~ > e+ ve b b~", ("QCD",)
    )
    record = executor.exported.trees.program["parameters"]
    assert any(isinstance(mass, list) for mass in record["plan_masses"])
    scales = [str(Fraction(mu) ** 2) for mu in reference["mu_r"]]
    for precision, tolerance in (
        (Precision.double_double(), 1e-10),
        (Precision.double(), 1e-8),
    ):
        results = executor.evaluate(
            reference["points"], scales, parameters=parameters, precision=precision
        )
        for k, (result, row) in enumerate(zip(results, reference["rows"], strict=True)):
            born = sum(float(v) for v in result.born.values())
            assert abs(born - row["born"]) <= tolerance * abs(row["born"])
            errors = _sector_errors(result, row, abs(row["born"]))
            assert max(errors) <= tolerance, (precision.arithmetic, k, errors)


def _tagged_photon_conversion(loop_gmu_ufo, reference, points):
    """Per point, MG5_aMC's tagged-photon factors of a G_mu CMS reference:
    (r, conversion pole, conversion finite) with r = (alpha(0)/alpha)^n over
    the reference's n tagged photons and the conversion the real part of
    the complex-mass scheme's eCoup_DGmua0_UV_EW at the point's MU_R
    (MG5 assigns it to a double-precision function,
    ``rescale_alpha_tagged.inc``). The values are the transformed model's
    (``model_cms.complex_mass_scheme``, which equals MG5_aMC's own CMS model
    to 1e-12, ``test_loop_model_cms``); the complex finite part equals the
    MadLoop export's own value per MU_R (``export_parameters``, double
    precision) to 1e-12."""

    from pyamplicol.loops.model_cms import complex_mass_scheme
    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions
    from pyamplicol.loops.scalar_bindings import (
        NativeScalarBindings,
        prepare_scalar_bindings,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    fixed = reference["fixed_parameters"]
    scheme = complex_mass_scheme(
        load_raw_loop_model(str(loop_gmu_ufo), restriction=reference["restriction"]),
        fixed_parameters={k: v for k, v in fixed.items() if k.startswith("W")},
    )
    raw = prepare_loop_restrictions(scheme.derived, simplify=False).derived
    names = {record["name"] for record in raw.collections["parameters"]}
    plan = prepare_scalar_bindings(
        raw,
        (("eCoup_DGmua0_UV_EW", -1), ("eCoup_DGmua0_UV_EW", 0), ("aEW", 0)),
        fixed_parameters={k: v for k, v in fixed.items() if k in names},
    )
    photons = len(reference["tagged_photon_legs"])
    alpha0 = 1 / float(reference["parameters"]["aEW0M1"])
    exported = reference["export_parameters"]
    assert exported["mu_r"] == reference["mu_r"]
    out = []
    for index, mu in enumerate(reference["mu_r"][: len(points)]):
        parameters = {
            name: value
            for name, value in reference["parameters"].items()
            if name in plan.external_names
        }
        parameters["MU_R"] = mu
        pole, finite, alpha = (
            complex(*map(float, value))
            for value in NativeScalarBindings(
                plan, arithmetic="double", parameters=parameters
            ).values.to_decimal()
        )
        theirs = complex(*exported["values"]["MDL_ECOUP_DGMUA0_UV_EW_FIN_"][index])
        assert abs(finite - theirs) <= 1e-12 * abs(theirs), (mu, finite, theirs)
        out.append(((alpha0 / alpha.real) ** photons, pole.real, finite.real))
    return out


@pytest.mark.loop_nightly
def test_cms_tagged_photon_equals_madloop_with_the_mg5_conversion(
    loop_gmu_ufo, loop_native, tmp_path
):
    """CM09: u d~ > e+ ve a [QCD QED] on the G_mu UFO (restriction a0) under
    the complex-mass scheme with the card's widths (resonant W in the Born,
    complex masses in the loops and counterterms), its photon tagged
    (alpha(0), the photon sources scaled natively; the conversion a field
    insertion whose interference takes the real part of the complex
    eCoup_DGmua0_UV_EW), native inputs: Born and both sectors equal MadLoop
    CMS in quadruple precision (``required/msm/udx_epvea_cms.json``) times
    MG5_aMC's formula over the tagged photon (r = (alpha(0)/alpha)^n; the EW
    sector's finite part and single pole minus 2 n Re(DGmua0) Born), at its
    four points: double-double to 1e-10, binary64 to 1e-8."""

    reference = _madloop_rows("udx_epvea_cms")
    executor, parameters = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        reference,
        "u d~ > e+ ve a",
        ("QCD", "QED"),
    )
    record = executor.exported.trees.program["parameters"]
    assert any(isinstance(mass, list) for mass in record["plan_masses"])
    # The record's leg labels count from 1, the fixture's ``pdgs`` index from 0.
    assert record["photons"][1] == [leg + 1 for leg in reference["tagged_photon_legs"]]
    factors = _tagged_photon_conversion(loop_gmu_ufo, reference, reference["points"])
    scales = [str(Fraction(mu) ** 2) for mu in reference["mu_r"]]
    for precision, tolerance in (
        (Precision.double_double(), 1e-10),
        (Precision.double(), 1e-8),
    ):
        results = executor.evaluate(
            reference["points"], scales, parameters=parameters, precision=precision
        )
        for k, (result, row) in enumerate(zip(results, reference["rows"], strict=True)):
            r, pole, finite = factors[k]
            n = len(reference["tagged_photon_legs"])
            born = row["born"]
            mine = sum(float(v) for v in result.born.values())
            assert abs(mine - r * born) <= tolerance * abs(r * born)
            want = []
            for sector in row["sectors"]["loop"]:
                electroweak = sector["QCD"] == sector["born_sector"]["QCD"]
                shift = 2 * n * born if electroweak else 0.0
                want.append(
                    [
                        r * (sector["finite"] - shift * finite),
                        r * (sector["single_pole"] - shift * pole),
                        r * sector["double_pole"],
                    ]
                )
            errors = _sector_errors(result, row, abs(r * born), want)
            assert max(errors) <= tolerance, (precision.arithmetic, k, errors)


def test_a_zero_width_yukawa_mass_propagates_as_its_line_mass(loop_gmu_ufo):
    """At Gamma_t = 0 (CM07) the transform ties ymt := MT, as MG5_aMC does,
    and the top propagates with its line mass LMASS_MT := MT (real;
    ``model_cms`` step 4b), so no coupling reads the mass parameter whose
    sign the tree reading of a top loop line flips (``conventions``,
    ``LeafWeights._require_mass_free_couplings``); with the card's width the
    top has its pole mass (the tie in MT and WT) and no line mass."""

    from pyamplicol.loops.model_cms import complex_mass_scheme
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(str(loop_gmu_ufo), restriction="default")
    scheme = complex_mass_scheme(raw, fixed_parameters={"WT": "0"})
    assert dict(scheme.lines) == {"MT": "LMASS_MT"} and "MT" not in scheme.masses
    assert scheme.yukawas["ymt"] == "MT"
    derived = scheme.derived
    line = derived.record("parameters", "LMASS_MT")
    assert (line["fields"]["nature"], line["fields"]["type"]) == ("internal", "real")
    assert derived.exact_value("parameters", line["id"])["expression"] == "MT"
    assert derived.record("particles", "t")["fields"]["mass"]["name"] == "LMASS_MT"
    assert derived.metadata["complex_mass_scheme"]["lines"] == {"MT": "LMASS_MT"}
    ymt = derived.record("parameters", "ymt")
    assert derived.exact_value("parameters", ymt["id"])["expression"] == "MT"
    default = complex_mass_scheme(raw)
    assert not default.lines and default.masses["MT"] == "CMASS_MT"
    assert default.derived.record("particles", "t")["fields"]["mass"]["name"] == (
        "CMASS_MT"
    )


@pytest.mark.loop_nightly
def test_cms_acceptance_process_equals_madloop(loop_gmu_ufo, loop_native, tmp_path):
    """CM07 ([U4], the complex-mass acceptance): g g > t t~ e- e+ g
    QCD=3 QED=2 [virt=QCD QED] QCD^2==6 QED^2==6 on the G_mu UFO under the
    complex-mass scheme with Gamma_Z, Gamma_W > 0 and Gamma_t = 0 (complex Z,
    W and H masses in the trees, loops and counterterms; the top on its line
    mass LMASS_MT := MT, the Yukawa tie ymt := MT), native inputs: Born and
    the EW sector equal MadLoop CMS in quadruple precision
    (``required/msm/gg_ttxemepg_cms.json``) at its three points,
    double-double to 1e-10, binary64 to 1e-6. About 10 min, 7 GB."""

    reference = _madloop_rows("gg_ttxemepg_cms")
    executor, parameters = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        reference,
        "g g > t t~ e- e+ g",
        ("QCD", "QED"),
        loops=(dict(reference["loop_orders"]),),
    )
    record = executor.exported.trees.program["parameters"]
    assert any(isinstance(mass, list) for mass in record["plan_masses"])
    scales = [str(Fraction(mu) ** 2) for mu in reference["mu_r"]]
    for precision, tolerance in (
        (Precision.double_double(), 1e-10),
        (Precision.double(), 1e-6),
    ):
        results = executor.evaluate(
            reference["points"], scales, parameters=parameters, precision=precision
        )
        for k, (result, row) in enumerate(zip(results, reference["rows"], strict=True)):
            born = sum(float(v) for v in result.born.values())
            assert abs(born - row["born"]) <= tolerance * abs(row["born"])
            errors = _sector_errors(result, row, abs(row["born"]))
            assert max(errors) <= tolerance, (precision.arithmetic, k, errors)
