# SPDX-License-Identifier: 0BSD
"""The complex-mass scheme of v2 virtuals with Python inputs (M-SM MS-CMS).

``[loop.model.scheme] mass_scheme = "complex-mass"`` (alone: MadLoop's ``set
complex_mass_scheme True`` over the model's own declaration) generates on
MG5_aMC's complex-mass model of the source (``loops.model_cms``): every
particle of a nonzero width at the card propagates with its complex pole
mass ``CMASS_M = sqrt(M^2 - i M Gamma)``, which the Python assembly takes in
its loop propagators, kernel constants and reduction lanes
(``assemble._model_masses(pole_masses=True)``). The policy is typed and the
dry run shows it:

* ``mass-scheme`` (a value other than real-on-shell / complex-mass),
  ``complex-fixed-parameter``, and from the transform ``cms-internal-mass``
  (loop_sm's W: its mass is internal), ``cms-yukawa-input`` and
  ``cms-external-width`` (EW-25: stable external states);
* real on-shell requests keep ``coloured-width`` and ``electroweak-width``,
  whose remedies name the complex-mass scheme.

A mass that a Yukawa tie reads without a pole mass (a zero width at the
card: every external top, EW-25) propagates with its real line mass
``LMASS_M := M`` (``model_cms`` step 4b), so no coupling reads the loop
line's mass: on loop_sm at zero widths the scheme equals real on-shell with
ymt = MT (default physics tier), and CM04 (g g > t t~, Gamma_t = 0) equals
MadLoop CMS.

The CM rows of m-sm PLAN §4.2 with Python inputs (``LoopSlice.evaluate`` in
binary64 and arbitrary precision; CM01 also in double-double, the native
executor with Python inputs) equal MadLoop CMS: CM10 (loop_sm with the top
width) in the default physics tier, CM01, CM02, CM04, CM05 and CM09 (the
G_mu UFO; CM09 times MG5_aMC's tagged-photon formula) nightly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.model_cms import (
    NATIVE_POLE_MASSES,
    REFUSAL_KEYS,
    ComplexMassSchemeUnsupported,
    require_mass_scheme,
    require_real_fixed_parameters,
)
from pyamplicol.loops.v2.refusals import LoopRequestUnsupported, listed_refusal_keys

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / "tests/fixtures/loop"


@pytest.mark.loop_fast
def test_mass_scheme_values_and_fixed_parameters_are_typed():
    """Only the two schemes are served; a misspelt complex-mass request is
    refused, never computed real on-shell. Fixed parameters are real; every
    key of the policy is a listed loop-admission refusal."""

    assert set(REFUSAL_KEYS) <= set(listed_refusal_keys())
    for value in ("real-on-shell", "complex-mass"):
        assert require_mass_scheme(value) == value
    for value in ("complex mass", "CMS", "real masses"):
        with pytest.raises(ComplexMassSchemeUnsupported) as refused:
            require_mass_scheme(value)
        assert refused.value.key == "mass-scheme"
        assert isinstance(refused.value, LoopRequestUnsupported | ValueError)
    assert require_real_fixed_parameters({"WT": ("0", "0"), "MB": ("4.7", "0")}) == {
        "WT": "0",
        "MB": "4.7",
    }
    with pytest.raises(ComplexMassSchemeUnsupported, match="'MT' complex") as refused:
        require_real_fixed_parameters({"MT": ("173", "-0.7"), "WT": ("0", "0")})
    assert refused.value.key == "complex-fixed-parameter"
    with pytest.raises(ValueError, match="unknown complex-mass refusal"):
        ComplexMassSchemeUnsupported("coloured-width", "not a policy key")


# ----------------------------------------------------- dry run (public route)


def _card(output: Path, expression: str, fixed: dict, scheme: str | None) -> str:
    parameters = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    declared = (
        "" if scheme is None else f'[loop.model.scheme]\nmass_scheme = "{scheme}"'
    )
    return f"""
action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[loop]
r2_strategy = "model"
{declared}
[loop.preparation]
fixed_parameters = {{{parameters}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


STABLE = {"WZ": "0", "WW": "0", "WH": "0"}


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,expression,fixed,scheme",
    [
        (None, "u d~ > w+ b b~", STABLE, "complex-mass"),
        ("cms-external-width", "g g > t t~", STABLE, "complex-mass"),
        ("cms-internal-mass", "u d~ > w+ b b~", {"WZ": "0", "WH": "0"}, "complex-mass"),
        (
            "cms-yukawa-input",
            "u d~ > w+ b b~",
            {**STABLE, "ymt": "172"},
            "complex-mass",
        ),
        (
            "complex-fixed-parameter",
            "u d~ > w+ b b~",
            {**STABLE, "MT": ["173", "1"]},
            "complex-mass",
        ),
        ("mass-scheme", "u d~ > w+ b b~", STABLE, "complex mass"),
        ("coloured-width", "g g > t t~", STABLE, None),
    ],
)
def test_mass_scheme_refusals_fire_in_the_dry_run(
    tmp_path, key, expression, fixed, scheme
):
    """The mass-scheme policy on the public route, in the dry run, on the
    shipped loop_sm (no_b_mass, the top width 1.4915 at the card): a
    ``mass_scheme``-only declaration selects the complex-mass scheme over the
    shipped profile; each refusal is its listed key; the real on-shell
    coloured-width refusal names the complex-mass scheme as a remedy."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.models.loading import compile_model_source

    compile_model_source("loop_sm", restriction="no_b_mass")
    card = tmp_path / "card.toml"
    card.write_text(_card(tmp_path / "out", expression, fixed, scheme))
    plan = Generator(resolve_config(card)).plan(
        ProcessSet((ProcessRequest.parse(expression, name="p"),))
    )
    if key is None:
        coverage = plan.estimated_coverage
        assert coverage["model_declaration"]["scheme"]["mass_scheme"] == "complex-mass"
        if not NATIVE_POLE_MASSES:
            # Stored artifacts' native inputs take no pole mass yet (MS-CMS-N).
            (blocker,) = plan.unsupported_features
            assert blocker.startswith("complex-mass-scheme: "), blocker
            assert "MS-CMS-N" in blocker
            return
        assert plan.unsupported_features == ()
        (entry,) = coverage["processes"]
        assert entry["engine"] == "v2"
        return
    (blocker,) = plan.unsupported_features
    assert blocker.startswith(f"{key}: "), blocker
    if key == "cms-external-width":
        assert "PDG 6 (CMASS_MT)" in blocker and 'WT = "0"' in blocker
    if key == "coloured-width":
        assert 'mass_scheme = "complex-mass"' in blocker


_EW_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{{expression = "e- e+ > m- m+", name = "p"}}]
loop_coupling_orders = ["QED"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.model]
loop_coupling_orders = ["QED"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO on-shell electroweak"
mass_scheme = {scheme}
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_qcd_qed_sm lhv=1 supplied QED rules"
[loop.preparation]
fixed_parameters = {fixed}
compatibility = "gmu_particle_anti"
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,scheme,widths",
    [
        ("electroweak-width", "real-on-shell", ("WT", "WW", "WH")),
        ("complex-mass-scheme", "complex-mass", ()),
        (None, "real-on-shell", ("WT", "WW", "WH", "WZ")),
    ],
    ids=["width", "pole-mass", "stable"],
)
def test_electroweak_width_policy_in_the_dry_run(
    loop_qcd_qed_sm_ufo, tmp_path, key, scheme, widths
):
    """The electroweak width policy on the public route, in the dry run of
    e- e+ > mu- mu+ [QED] on loop_qcd_qed_sm (its s-channel Z): real on
    shell with the Z width at the card is refused at the Born DAG
    (``loop_service._v2_request``: ``electroweak-width``, naming the
    complex-mass scheme); the complex-mass scheme with every width at the
    card is ``complex-mass-scheme`` until stored artifacts take native pole
    masses (MS-CMS-N); every width fixed to zero passes."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.models.loading import compile_model_source

    # The dry run reads compiled models from the cache only.
    compile_model_source(str(loop_qcd_qed_sm_ufo), require_supported=False)
    fixed = "{" + ", ".join(f'{name} = "0"' for name in widths) + "}"
    card = tmp_path / "card.toml"
    card.write_text(
        _EW_CARD.format(
            model=json.dumps(str(loop_qcd_qed_sm_ufo)),
            scheme=json.dumps(scheme),
            fixed=fixed,
            output=json.dumps(str(tmp_path / "out")),
        )
    )
    plan = Generator(resolve_config(card)).plan(
        ProcessSet((ProcessRequest.parse("e- e+ > m- m+", name="p"),))
    )
    if key == "complex-mass-scheme" and NATIVE_POLE_MASSES:
        key = None  # stored artifacts take native pole masses (MS-CMS-N)
    if key is None:
        assert plan.unsupported_features == ()
        return
    (blocker,) = plan.unsupported_features
    assert blocker.startswith(f"{key}: "), blocker
    if key == "electroweak-width":
        assert "complex-mass" in blocker and "WZ" in blocker, blocker
    else:
        assert "MS-CMS-N" in blocker, blocker


# ----------------------------------------------------- CM rows, Python inputs


def _reference(name: str) -> dict:
    return json.loads((FIXTURES / "required/msm" / f"{name}.json").read_text())


def _laurent_errors(values, born, theirs, coupling: float) -> list[float]:
    """(finite, 1/eps, 1/eps^2) against a MadLoop row in the campaign metric:
    relative, against max(|value|, |Born| coupling / 2 pi)."""

    flux = abs(born) * coupling / (2 * math.pi)
    return [
        abs(float(a) - b) / max(abs(b), flux)
        for a, b in zip(values, theirs, strict=True)
    ]


@pytest.fixture(scope="module")
def loop_sm_no_b_mass(loop_sm_ufo, tmp_path_factory):
    """loop_sm at its no_b_mass restriction with the scalar declarations
    of the M-QCD references, and its model cache."""

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("loop-sm")
    compiled = compile_model_source(
        loop_sm_ufo, restriction="no_b_mass", cache_dir=root
    )
    compiled = replace(
        compiled,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            compiled.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return compiled, root


@pytest.fixture(scope="module")
def cm10(loop_sm_no_b_mass, loop_native):
    """CM10's slice: u d~ > w+ b b~ QCD=0 [QCD] on loop_sm (no_b_mass) under
    the complex-mass scheme with the top width (WW = WZ = WH = 0), the top
    resonant in the Born and complex in the loops and counterterms."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    compiled, root = loop_sm_no_b_mass
    reference = _reference("udx_wpbbx_cms")
    assert reference["restriction"] == "no_b_mass"
    fixed = reference["fixed_parameters"]
    model, scheme = complex_mass_model(compiled, fixed_parameters=fixed, cache_dir=root)
    fixed = {k: v for k, v in fixed.items() if k in model.parameter_defaults}
    request = SliceRequest(
        "u d~ > w+ b b~",
        reference["born_orders"],
        reference["loop_orders"],
        fixed_parameters=fixed,
    )
    return reference, model, scheme, LoopSlice.generate(model, request, loop_native)


@pytest.mark.loop_physics
def test_cm10_top_width_with_python_inputs_equals_madloop_cms(cm10):
    """CM10 with Python inputs = MadLoop CMS in quadruple precision
    (``required/msm/udx_wpbbx_cms.json``) at its four points: arbitrary
    precision (32 digits) to 1e-11 (measured 6.8e-14), binary64 to 1e-8
    (3.4e-10). The slice's loop propagators take the complex top: its
    squared mass M^2 - i M Gamma; the conventions record the scheme; the
    recommended stability estimate bounds the binary64 error."""

    from pyamplicol.loops.v2.assemble import Precision, _model_masses, _squared
    from pyamplicol.loops.v2.ew_policy import scheme_conventions

    reference, model, scheme, slice_ = cm10
    assert dict(scheme.masses) == {"MT": "CMASS_MT"} and "MT" not in scheme.lines
    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in slice_.request.fixed_parameters
    }
    top = _squared(_model_masses(model, parameters, 40, pole_masses=True)("CMASS_MT"))
    mt, wt = Fraction(parameters["MT"]), Fraction(reference["fixed_parameters"]["WT"])
    assert abs(top.real - mt * mt) < Fraction(1, 10**30)
    assert abs(top.imag + mt * wt) < Fraction(1, 10**30)
    record = scheme_conventions(slice_)
    assert record["mass_scheme"] == "complex-mass"
    assert record["complex_mass"]["pole_masses"] == {"MT": "CMASS_MT"}
    assert record["complex_mass"]["reglog_convention"] == "ufo-standard"
    alpha_s = float(parameters["aS"])
    for precision, tolerance in (
        (Precision.arbitrary(32), 1e-11),
        (Precision.double(), 1e-8),
    ):
        for k, (point, mu, row) in enumerate(
            zip(reference["points"], reference["mu_r"], reference["rows"], strict=True)
        ):
            result = slice_.evaluate(
                point,
                str(Fraction(mu) ** 2),
                parameters=parameters,
                precision=precision,
            )
            born, *laurent = (float(x) for x in result.totals[:4])
            assert abs(born - row["born"]) <= tolerance * abs(row["born"])
            errors = _laurent_errors(
                laurent,
                row["born"],
                [row["finite"], row["single_pole"], row["double_pole"]],
                alpha_s,
            )
            assert all(e <= tolerance for e in errors), (
                precision.arithmetic,
                k,
                errors,
            )
    # The recommended stability checks (a rotation and the rescaled reduction,
    # which scales the complex m^2 with the other dimensionful inputs) bound
    # the binary64 error (campaign rule 3: error <= 10 E).
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    row = reference["rows"][0]
    result = slice_.evaluate(
        reference["points"][0],
        str(Fraction(reference["mu_r"][0]) ** 2),
        parameters=parameters,
        precision=Precision.double(),
        stability=RECOMMENDED_CHECKS,
    )
    estimate = result.stability.relative
    error = abs(float(result.totals[1].real) - row["finite"]) / abs(row["finite"])
    assert set(result.stability.checks) == {"rotation-0", "reduction-rescaled"}
    assert 0 < estimate < 1e-6 and error <= 10 * estimate, (error, estimate)


def _supplied_point(masses) -> list[list[str]]:
    """The first regular point of ``tools/loop_reference/supplied_points``
    (massive beams allowed), as decimal strings."""

    from tools.loop_reference import supplied_points

    with localcontext() as context:
        context.prec = supplied_points.CONSTRUCTION_DIGITS
        points, scales, _ = supplied_points.generate(
            [Fraction(m) for m in masses], count=1
        )
        point = supplied_points.quantise(points[0], scales[0])
    return [[supplied_points.text(x) for x in leg] for leg in point]


@pytest.mark.loop_physics
def test_a_zero_width_yukawa_mass_propagates_as_its_line_mass(
    loop_sm_no_b_mass, loop_native
):
    """At zero widths the transform ties ymt := MT (MG5_aMC) and the top
    propagates with its line mass LMASS_MT := MT (``model_cms`` step 4b), so
    the top Yukawa vertices on QCD loop lines read no mass of the line they
    emit, whose sign the tree reading of a fermion loop line flips
    (``conventions.LeafWeights._require_mass_free_couplings``, which refused
    this request before). t t~ > h h [QCD] on loop_sm then equals the real
    on-shell virtual at the same card (ymt = MT = 173), binary64 and
    arbitrary precision, and at a run-time MT = ymt = 172. About 20 s."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest

    compiled, root = loop_sm_no_b_mass
    fixed = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
    model, scheme = complex_mass_model(compiled, fixed_parameters=fixed, cache_dir=root)
    assert not scheme.masses and scheme.yukawas["ymt"] == "MT"
    assert dict(scheme.lines) == {"MT": "LMASS_MT", "MTA": "LMASS_MTA"}
    derived = scheme.derived
    line = derived.record("parameters", "LMASS_MT")
    assert (line["fields"]["nature"], line["fields"]["type"]) == ("internal", "real")
    assert derived.exact_value("parameters", line["id"])["expression"] == "MT"
    assert derived.record("particles", "t")["fields"]["mass"]["name"] == "LMASS_MT"
    assert derived.metadata["complex_mass_scheme"]["lines"] == dict(scheme.lines)
    assert {p["mass"] for p in model.model["particles"] if p["pdg_code"] == 6} == {
        "LMASS_MT"
    }

    born, loops = {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}
    cms = LoopSlice.generate(
        model,
        SliceRequest(
            "t t~ > h h",
            born,
            loops,
            fixed_parameters={
                k: v for k, v in fixed.items() if k in model.parameter_defaults
            },
        ),
        loop_native,
    )
    real = LoopSlice.generate(
        compiled,
        SliceRequest(
            "t t~ > h h",
            born,
            loops,
            fixed_parameters={
                k: v for k, v in fixed.items() if k in compiled.parameter_defaults
            },
        ),
        loop_native,
    )
    # At the card, and at a run-time top mass: the line mass is derived, so
    # the top's sources take its value at the point's parameters too
    # (``symbolica_exact._particle_mass``; 2e-4 off with the card's value).
    for mt, precision in (
        ("173", Precision.double()),
        ("173", Precision.arbitrary(32)),
        ("172", Precision.double()),
    ):
        point = _supplied_point((mt, mt, 125, 125))
        ours, theirs = (
            [
                complex(x)
                for x in s.evaluate(
                    point, "250000", parameters=parameters, precision=precision
                ).totals[:4]
            ]
            for s, parameters in (
                (cms, {"MT": mt}),
                (real, {"MT": mt, "ymt": mt}),
            )
        )
        scale = max(abs(x) for x in theirs)
        assert scale > 0 and theirs[1] != 0
        assert (
            max(abs(a - b) for a, b in zip(ours, theirs, strict=True)) <= 1e-13 * scale
        ), (mt, precision.arithmetic, ours, theirs)


@pytest.mark.loop_physics
def test_public_complex_mass_artifact_at_zero_widths_equals_real_on_shell(tmp_path):
    """The public route at zero widths (no pole mass, so no native guard):
    ``Generator.generate`` stores t t~ > h h [QCD] on loop_sm under the
    complex-mass scheme (ymt := MT, the top on its line mass) with native
    point inputs, and ``Runtime`` evaluates it as the real on-shell artifact
    at a run-time MT = ymt = 172. About 30 s."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.models.loading import compile_model_source

    compile_model_source("loop_sm", restriction="no_b_mass")
    expression = "t t~ > h h"
    fixed = {**STABLE, "WT": "0"}
    point = _supplied_point((172, 172, 125, 125))
    values = {}
    for scheme, parameters in (
        ("complex-mass", {"MT": "172"}),
        (None, {"MT": "172", "ymt": "172"}),
    ):
        output = tmp_path / str(scheme)
        card = tmp_path / f"{scheme}.toml"
        card.write_text(_card(output, expression, fixed, scheme))
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
        )
        runtime = Runtime.load(output, process="p", model_parameters=parameters)
        result = runtime.evaluate([point], mu_r2=["250000"])
        values[scheme] = [float(result.born_values[0][0])] + [
            float(x) for x in result.values[0][0]
        ]
    ours, theirs = values["complex-mass"], values[None]
    scale = max(abs(x) for x in theirs)
    assert theirs[1] != 0
    assert max(abs(a - b) for a, b in zip(ours, theirs, strict=True)) <= 1e-13 * scale


# ------------------------------------------- the G_mu UFO, every width (nightly)

GMU_PARAMETERS = {
    "Gf": "0.0000116639",
    "aS": "0.118",
    "MT": "173.3",
    "MZ": "91.188",
    "MW": "80.419",
    "MH": "125",
}


def _gmu_cms_slice(
    loop_gmu_ufo,
    loop_native,
    root,
    process,
    born,
    corrections,
    *,
    restriction="default",
    fixed=None,
    reglog_convention="ufo-standard",
):
    """A G_mu UFO request (Feynman gauge) under the complex-mass scheme at the
    card of ``fixed`` (``complex_mass_model``): the Born orders ``born``, one
    loop order per correction."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_gmu_ufo,
        restriction=restriction,
        cache_dir=root,
        require_supported=False,
        gauge="feynman",
    )
    fixed = dict(fixed or {})
    model, scheme = complex_mass_model(
        compiled,
        fixed_parameters=fixed,
        reglog_convention=reglog_convention,
        cache_dir=root,
    )
    fixed = {k: v for k, v in fixed.items() if k in model.parameter_defaults}
    born = {"QCD": 0, "QED": 0, **born}
    loops = tuple({**born, order: born[order] + 2} for order in sorted(corrections))
    request = SliceRequest(
        process,
        born,
        loops[0] if len(loops) == 1 else loops,
        corrections=tuple(sorted(corrections)),
        fixed_parameters=fixed,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    return model, scheme, LoopSlice.generate(model, request, loop_native)


CM01_CARRIERS = (("double", 1e-8), ("double-double", 1e-8), ("arbitrary", 1e-8))
# The widths of the CM01 blocks (``cms-source-gmu.json`` records none).
CM01_WIDTHS = {
    "finite": {"WZ": "2.49877", "WW": "2.09291", "WT": "1.37758", "WH": "0.00407468"},
    "zero": {"WZ": "0", "WW": "0", "WT": "0", "WH": "0"},
}
# The small-width control (MadLoop double, every width x 0.001, the
# UFO-standard reglog), m-sm PLAN §4.2 CM01.
CM01_SMALL_WIDTHS = "electroweak-mumu-cms-small-width-control.json"


def _cm01_blocks(reference: dict, convention: str) -> dict:
    """CM01's reference blocks: finite and zero widths from the view's
    fixture, the small widths from their control (UFO-standard reglog)."""

    blocks = {block: reference[block] for block in ("finite", "zero")}
    if convention == "ufo-standard":
        control = json.loads((FIXTURES / CM01_SMALL_WIDTHS).read_text())
        assert Fraction(control["mu_r2"]) == 1000000
        blocks["small"] = {
            "points": control["exact_points"],
            "widths": control["widths"],
            "born": [row["born"] for row in control["rows"]],
            "full": [row["full"] for row in control["rows"]],
        }
    return blocks


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "convention,fixture",
    [("ufo-standard", "cms-ufo-standard-gmu.json"), ("source", "cms-source-gmu.json")],
)
def test_cm01_mumu_with_python_inputs_equals_madloop_cms(
    loop_gmu_ufo, loop_native, tmp_path, convention, fixture
):
    """CM01: e- e+ > mu- mu+ [QED] on the G_mu UFO under the complex-mass
    scheme (every width at the card: Z, W, t, H pole masses), Python inputs,
    against MadLoop CMS (double-precision exports, ``fixture``): the Born and
    the virtual's three Laurent coefficients at the continuum and three
    Z-resonance points, at zero widths and at the widths x 0.001 (runtime
    widths, one generation), binary64, double-double (the native executor
    with Python inputs) and arbitrary precision (32 digits) to 1e-8;
    double-double = AP(32) to 1e-20 everywhere and binary64 = AP(32) to 1e-9
    at the small widths (the branch side of the helpers on their cuts;
    ``electroweak-mumu-cms-small-width-control``); AP(40) = AP(60) to 1e-30
    at a Z-resonance point. ``"source"``: the G_mu source's reglog override,
    a compatibility view ([U3]). Generation ~5 min, ~6 GB."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    reference = json.loads((FIXTURES / fixture).read_text())
    _model, scheme, slice_ = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        "e- e+ > m- m+",
        {"QED": 2},
        ("QED",),
        fixed={"MB": "0", "MTA": "0"},
        reglog_convention=convention,
    )
    assert set(scheme.masses) == {"MZ", "MW", "MT", "MH"}
    executor = NativeExecutor(slice_, inputs="python")
    precisions = {
        "double": Precision.double(),
        "double-double": Precision.double_double(),
        "arbitrary": Precision.arbitrary(32),
    }
    for block, expected in _cm01_blocks(reference, convention).items():
        parameters = {
            **GMU_PARAMETERS,
            **(expected.get("widths") or CM01_WIDTHS[block]),
        }
        values, exact = {}, {}
        for carrier, tolerance in CM01_CARRIERS:
            precision = precisions[carrier]
            points = expected["points"]
            results = (
                executor.evaluate(
                    points,
                    ["1000000"] * len(points),
                    parameters=parameters,
                    precision=precision,
                )
                if carrier == "double-double"
                else [
                    slice_.evaluate(
                        point, "1000000", parameters=parameters, precision=precision
                    )
                    for point in points
                ]
            )
            for k, result in enumerate(results):
                if carrier != "double":
                    exact[carrier, k] = [Decimal(x) for x in result.totals[:4]]
                born, *laurent = (complex(x) for x in result.totals[:4])
                values[carrier, k] = [born, *laurent]
                theirs = [float(x) for x in expected["full"][k]]
                born_reference = float(expected["born"][k])
                assert abs(born.real - born_reference) <= tolerance * born_reference
                errors = _laurent_errors(
                    [x.real for x in laurent], born_reference, theirs, 1 / 132.0
                )
                assert all(e <= tolerance for e in errors), (block, carrier, k, errors)
        for k in range(len(expected["points"])):
            # Double-double against arbitrary precision (§1.3).
            dd, ap = exact["double-double", k], exact["arbitrary", k]
            scale = max(abs(x) for x in ap)
            assert max(abs(a - b) for a, b in zip(dd, ap, strict=True)) <= (
                Decimal("1e-20") * scale
            ), (
                block,
                k,
                [float(abs(a - b) / scale) for a, b in zip(dd, ap, strict=True)],
            )
        if block == "small":
            for k in range(len(expected["points"])):
                double, arbitrary = values["double", k], values["arbitrary", k]
                scale = max(abs(x) for x in arbitrary)
                assert (
                    max(abs(a - b) for a, b in zip(double, arbitrary, strict=True))
                    <= 1e-9 * scale
                )
        if block == "finite" and convention == "ufo-standard":
            # Arbitrary precision against itself at more digits (§1.3).
            point = expected["points"][1]
            low, high = (
                [
                    Decimal(x)
                    for x in slice_.evaluate(
                        point,
                        "1000000",
                        parameters=parameters,
                        precision=Precision.arbitrary(digits),
                    ).totals[:4]
                ]
                for digits in (40, 60)
            )
            scale = max(abs(x) for x in high)
            assert max(abs(a - b) for a, b in zip(low, high, strict=True)) <= (
                Decimal("1e-30") * scale
            ), [float(abs(a - b) / scale) for a, b in zip(low, high, strict=True)]


def _sector_key(orders) -> tuple[tuple[str, int], ...]:
    """A squared-order sector as the result keys it: its nonzero QCD and
    QED powers."""

    return tuple(
        (name, int(orders.get(name, 0)))
        for name in ("QCD", "QED")
        if int(orders.get(name, 0))
    )


def _check_sectors(slice_, reference, model, carriers, points, executor=None):
    """Per point and per squared-order sector of a MadLoop CMS row: the Born,
    the set of the virtual's sectors, their sum and each sector's Laurent
    coefficients in the campaign metric (the correction's coupling: alpha_s
    for a QCD sector, alpha_Gmu ~ 1/132 for an EW one and the sum). With
    ``executor`` (a ``NativeExecutor``: binary64 only) its points instead of
    the slice's Python inputs."""

    from pyamplicol.loops.v2.assemble import Precision

    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in slice_.request.fixed_parameters
    }
    for carrier, tolerance in carriers:
        precision = (
            Precision.double() if carrier == "double" else Precision.arbitrary(32)
        )
        for k in points:
            row = reference["rows"][k]
            scale = str(Fraction(reference["mu_r"][k]) ** 2)
            if executor is not None:
                assert carrier == "double", carrier
                (result,) = executor.evaluate(
                    [reference["points"][k]], [scale], parameters=parameters
                )
            else:
                result = slice_.evaluate(
                    reference["points"][k],
                    scale,
                    parameters=parameters,
                    precision=precision,
                )
            born = sum(complex(v).real for v in result.born.values())
            assert abs(born - row["born"]) <= tolerance * abs(row["born"])
            virtual = {
                _sector_key(dict(sector)): laurent
                for sector, laurent in result.components["virtual"].items()
            }
            expected = {_sector_key(sector) for sector in row["sectors"]["loop"]}
            assert set(virtual) == expected, (carrier, k, sorted(virtual))
            total = row["sectors"]["total"]
            errors = _laurent_errors(
                [complex(x).real for x in result.totals[1:4]],
                row["born"],
                [total["finite"], total["single_pole"], total["double_pole"]],
                1 / 132.0,
            )
            # all(), not max(): max([0.0, nan]) is 0.0 (a NaN pole passed).
            assert all(e <= tolerance for e in errors), (carrier, k, "total", errors)
            for sector in row["sectors"]["loop"]:
                laurent = virtual[_sector_key(sector)]
                qcd = sector["QCD"] > sector["born_sector"]["QCD"]
                errors = _laurent_errors(
                    [complex(x).real for x in laurent.coefficients],
                    row["born"],
                    [sector["finite"], sector["single_pole"], sector["double_pole"]],
                    float(parameters["aS"]) if qcd else 1 / 132.0,
                )
                assert all(e <= tolerance for e in errors), (carrier, k, sector, errors)


# The CM rows whose native point inputs ``_check_cms_split`` checks here;
# CM05's native lane is checked against MadLoop CMS, split included, in
# ``test_loop_v2_master_census.py`` (its parameter program's plane is the
# level-0 one: SymJIT's optimised SIMD code gave NaN for one output, MS-MASTERS).
NATIVE_SPLIT_ROWS = ("uux_emep_cms", "gg_ttx_cms")


def _check_cms_split(slice_, reference, model, lanes=("python", "native")):
    """MS-POLES under the scheme: the transform keeps the G_mu UFO's pole
    regulators adjustable (``model_cms`` step 5), so "uv" and "ir" split the
    counterterms' poles by them, with no remainder. At the row's first
    point, with Python and native point inputs (``lanes``): per sector bare +
    counterterm UV poles vanish (1e-12 of the bare UV pole), the "ir"
    virtual's poles are those of "all" (to the same bound) and the finite
    part is the same in every selection."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    assert slice_.pole_regulators.record() == {"uv": "epsUV", "ir": "epsIR"}
    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in slice_.request.fixed_parameters
    }
    point = reference["points"][0]
    scale = str(Fraction(reference["mu_r"][0]) ** 2)
    # Only a native lane exports native inputs.
    native = (
        NativeExecutor(slice_, forward=False, inputs="native")
        if "native" in lanes
        else None
    )

    def run(lane, poles):
        if lane == "python":
            return slice_.evaluate(point, scale, parameters=parameters, poles=poles)
        (result,) = native.evaluate(
            [point], [scale], parameters=parameters, poles=poles
        )
        return result

    for lane in lanes:
        whole, uv, ir = (run(lane, poles) for poles in ("all", "uv", "ir"))
        for sector, laurent in whole.components["virtual"].items():
            total = [complex(x) for x in laurent.coefficients]
            parts = uv.pole_origins["uv"]
            bare_uv = complex(parts["bare"][sector].coefficients[1])
            residue = complex(parts["virtual"][sector].coefficients[1])
            assert abs(residue) <= 1e-12 * abs(bare_uv), (lane, sector, residue)
            assert parts["virtual"][sector].coefficients[2] == 0
            poles = [complex(x) for x in ir.components["virtual"][sector].coefficients]
            for k in (1, 2):
                bound = 1e-12 * max(abs(bare_uv), abs(total[k]))
                assert abs(poles[k] - total[k]) <= bound, (lane, sector, k)
            for result in (uv, ir):
                finite = result.components["virtual"][sector].coefficients[0]
                assert complex(finite) == total[0], (lane, sector)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "name,process",
    [
        ("uux_emep_cms", "u u~ > e- e+"),
        ("gg_ttx_cms", "g g > t t~"),
        ("uux_emepg_cms", "u u~ > e- e+ g"),
    ],
    ids=["CM02", "CM04", "CM05"],
)
def test_cm_joint_corrections_with_python_inputs_equal_madloop_cms(
    loop_gmu_ufo, loop_native, tmp_path, name, process
):
    """CM02 (u u~ > e- e+, at and off the Z pole), CM04 (g g > t t~ with
    Gamma_t = 0, EW-25: the top on its line mass LMASS_MT := MT, the Yukawa
    tie ymt := MT on its EW loop lines) and CM05 (u u~ > e- e+ g) [QCD QED]
    on the G_mu UFO under the complex-mass scheme with every other width
    (``required/msm/<name>.json``, MadLoop CMS in quadruple precision), Python
    inputs: every squared-order sector and their sum at the four points in
    binary64 and at the first point in arbitrary precision (32 digits), to
    1e-8; the counterterms' poles split by the pole regulators
    (``_check_cms_split``, MS-POLES). About 3 min generation each (CM02,
    CM05: 176 s), 6 GB."""

    reference = _reference(name)
    model, _scheme, slice_ = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        process,
        reference["born_orders"],
        ("QCD", "QED"),
        restriction=reference["restriction"],
        fixed=reference["fixed_parameters"],
    )
    _check_sectors(
        slice_, reference, model, (("double", 1e-8),), range(len(reference["points"]))
    )
    _check_sectors(slice_, reference, model, (("arbitrary", 1e-8),), [0])
    lanes = ("python", "native") if name in NATIVE_SPLIT_ROWS else ("python",)
    _check_cms_split(slice_, reference, model, lanes)


@pytest.mark.loop_nightly
def test_cm09_tagged_photon_with_python_inputs_equals_madloop_cms(
    loop_gmu_ufo, loop_native, tmp_path
):
    """CM09: u d~ > e+ ve a [QCD QED] on the G_mu UFO (a0) under the
    complex-mass scheme with the card's widths, its photon tagged (alpha(0);
    the conversion a field insertion taking the real part of the complex
    eCoup_DGmua0_UV_EW), Python inputs (``LoopSlice.evaluate``): Born and
    both sectors equal MadLoop CMS (``required/msm/udx_epvea_cms.json``,
    quadruple precision) times MG5_aMC's formula over the tagged photon (as
    the native-input test of MS-CMS-N, whose slice and conversion this
    reuses), binary64 at the four points and arbitrary precision (32 digits)
    at the first, to 1e-8. About 5 min."""

    from pyamplicol.loops.v2.assemble import Precision

    from . import test_loop_v2_native_complex_masses as native_cms

    reference = native_cms._madloop_rows("udx_epvea_cms")
    executor, parameters = native_cms._gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        reference,
        "u d~ > e+ ve a",
        ("QCD", "QED"),
    )
    factors = native_cms._tagged_photon_conversion(
        loop_gmu_ufo, reference, reference["points"]
    )
    n = len(reference["tagged_photon_legs"])
    for precision, points in (
        (Precision.double(), range(len(reference["points"]))),
        (Precision.arbitrary(32), [0]),
    ):
        for k in points:
            row = reference["rows"][k]
            result = executor.slice.evaluate(
                reference["points"][k],
                str(Fraction(reference["mu_r"][k]) ** 2),
                parameters=parameters,
                precision=precision,
            )
            r, pole, finite = factors[k]
            born = row["born"]
            mine = sum(complex(v).real for v in result.born.values())
            assert abs(mine - r * born) <= 1e-8 * abs(r * born)
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
            errors = native_cms._sector_errors(result, row, abs(r * born), want)
            assert all(e <= 1e-8 for e in errors), (precision.arithmetic, k, errors)
