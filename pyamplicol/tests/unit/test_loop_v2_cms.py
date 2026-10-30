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

The CM rows of m-sm PLAN §4.2 with Python inputs (``LoopSlice.evaluate`` in
binary64 and arbitrary precision; double-double runs in the native executor,
MS-CMS-N) equal MadLoop CMS: CM10 (loop_sm with the top width) in the default
physics tier, CM01, CM02 and CM05 (the G_mu UFO, every width) nightly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.model_cms import (
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
        assert plan.unsupported_features == ()
        coverage = plan.estimated_coverage
        assert coverage["model_declaration"]["scheme"]["mass_scheme"] == "complex-mass"
        (entry,) = coverage["processes"]
        assert entry["engine"] == "v2"
        return
    (blocker,) = plan.unsupported_features
    assert blocker.startswith(f"{key}: "), blocker
    if key == "cms-external-width":
        assert "PDG 6 (CMASS_MT)" in blocker and 'WT = "0"' in blocker
    if key == "coloured-width":
        assert 'mass_scheme = "complex-mass"' in blocker


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
def cm10(loop_sm_ufo, loop_native, tmp_path_factory):
    """CM10's slice: u d~ > w+ b b~ QCD=0 [QCD] on loop_sm (no_b_mass) under
    the complex-mass scheme with the top width (WW = WZ = WH = 0), the top
    resonant in the Born and complex in the loops and counterterms."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("cm10")
    reference = _reference("udx_wpbbx_cms")
    compiled = compile_model_source(
        loop_sm_ufo, restriction=reference["restriction"], cache_dir=root
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
    squared mass M^2 - i M Gamma; the conventions record the scheme."""

    from pyamplicol.loops.v2.assemble import Precision, _model_masses, _squared
    from pyamplicol.loops.v2.ew_policy import scheme_conventions

    reference, model, scheme, slice_ = cm10
    assert dict(scheme.masses) == {"MT": "CMASS_MT"}
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
            assert max(errors) <= tolerance, (precision.arithmetic, k, errors)
