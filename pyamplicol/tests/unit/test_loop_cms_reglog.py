# SPDX-License-Identifier: 0BSD
"""Prepared convention checks; no general CMS physics certificate."""

from __future__ import annotations

import cmath
from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.model_cms import complex_mass_scheme
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def declarations(loop_gmu_ufo):
    path = str(loop_gmu_ufo)
    raw = load_raw_loop_model(path, restriction="full")
    return {
        convention: complex_mass_scheme(raw, reglog_convention=convention)
        for convention in ("source", "ufo-standard")
    }


def test_explicit_reserved_convention_preserves_original_and_other_helpers(
    declarations,
):
    source, standard = (declarations[k] for k in ("source", "ufo-standard"))
    assert source.original is standard.original
    raw = source.original
    selected = raw.metadata["function_bindings"]["reglog"]
    assert source.derived.collections["functions"] == raw.collections["functions"]
    for index, record in enumerate(raw.collections["functions"]):
        if index != selected:
            assert standard.derived.collections["functions"][index] == record
    helper = standard.derived.collections["functions"][selected]["fields"]
    assert helper["expr"] == "(0.0 if z==0.0 else cmath.log(z))"
    assert helper["expression_source"]["operation"] == "select-ufo-standard-reglog"
    assert raw.collections["functions"][selected]["fields"]["expr"] != helper["expr"]
    metadata = standard.derived.metadata["complex_mass_scheme"]
    assert metadata["reglog_convention"] == "ufo-standard"
    assert source.derived.metadata["complex_mass_scheme"]["reglog_convention"] == (
        "source"
    )
    assert (
        RawLoopModel.from_dict(standard.derived.to_dict()).to_dict()
        == standard.derived.to_dict()
    )
    with pytest.raises(ValueError, match="reglog_convention"):
        complex_mass_scheme(raw, reglog_convention="automatic")


@pytest.fixture(scope="module")
def plans(declarations):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.scalar_bindings import prepare_scalar_bindings

    zeros = (
        "aEW0M1",
        "MD",
        "MU",
        "MS",
        "MC",
        "MB",
        "Me",
        "MM",
        "MTA",
        "lamWS",
        "AWS",
        "rhoWS",
        "etaWS",
    )
    ones = ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice")
    return {
        name: prepare_scalar_bindings(
            view.derived,
            [("eCoup_DGmua0_UV_EW", 0)],
            fixed_parameters={**dict.fromkeys(zeros, "0"), **dict.fromkeys(ones, "1")},
        )
        for name, view in declarations.items()
    }


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 80)],
)
def test_complex_charge_ct_anchor_width_limit_and_active_local_shifts(
    loop_native, plans, arithmetic, precision
):
    native = loop_native
    from symbolica import S

    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    options = {"arithmetic": arithmetic, "precision": precision}
    parameters = {
        "Gf": "0.0000116639",
        "aS": "0.118",
        "MU_R": "1000",
        "MT": "173.3",
        "MZ": "91.188",
        "MW": "80.419",
        "MH": "125",
        "WZ": "2.49877",
        "WW": "2.09291",
        "WT": "1.37758",
        "WH": "0.00407468",
    }
    states = {
        name: NativeScalarBindings(plan, parameters=parameters, **options)
        for name, plan in plans.items()
    }
    # Independent compiled Recola real-point CMS charge CT, dr/2. Original
    # MadLoop principal REGLOG agrees; its finite-width CT uses a different
    # (unexpanded) pole continuation and is therefore a separate anchor.
    real_anchor = complex(0.03251413826093476, 0.01510486664807422)
    finite_anchor = complex(0.0324314071994037667, 0.0157070264483830796)
    values = {}
    for factor in ("1", "0.000001", "0"):
        with localcontext() as context:
            context.prec = 100
            widths = {
                name: str(Decimal(parameters[name]) * Decimal(factor))
                for name in ("WZ", "WW", "WT", "WH")
            }
        values[factor] = {
            name: complex(*map(float, state.update(widths).values.to_decimal()[0]))
            for name, state in states.items()
        }
    assert abs(values["1"]["ufo-standard"] - finite_anchor) < 3e-14
    assert abs(values["0"]["ufo-standard"] - real_anchor) < 3e-14
    assert abs(values["0.000001"]["ufo-standard"] - real_anchor) < 1e-9
    delta = values["0"]["source"] - values["0"]["ufo-standard"]
    assert abs(delta - complex(0, -0.10293555751343791)) < 3e-14
    assert abs(values["0.000001"]["source"] - real_anchor) > 0.1

    # Actual four upper-left source CT arguments at zero widths. Their nonzero
    # imaginary parts are algebraic below-threshold roots, not a width crossing.
    # Exercise both saved helper owners natively, including a lower-left control.
    arguments = [
        ("-0.5", "1.833516781573637"),
        ("-0.5", "0.7264664251753891"),
        ("-0.06046135266407415", "0.9981705389531543"),
        ("-0.6428772371640453", "0.9340574566183896"),
        ("-0.5", "-0.7"),
    ]
    x = S("UFO::cms_reglog_probe")
    batch = native.NativeBatch.from_decimal(arguments, len(arguments), 1, **options)
    results = {}
    for name, plan in plans.items():
        library = plan.program.library
        expression = library.lower("reglog(cms_reglog_probe)")
        evaluator = native.CoefficientEvaluator(
            library.evaluator((expression,), (x,)).save(), **options
        )
        output = evaluator.evaluate_batch(batch)
        assert not output.nonfinite_rows()
        rows = output.to_decimal()
        # Diagnostics carry finite predicate values as well as invalid-path
        # NaNs; the checked native output, rather than zero, is the contract.
        results[name] = [
            complex(*map(float, rows[2 * row])) for row in range(len(arguments))
        ]
    for row, pair in enumerate(arguments):
        expected = cmath.log(complex(*map(float, pair)))
        assert abs(results["ufo-standard"][row] - expected) < 2e-15
        expected_shift = -2j * cmath.pi if row < 4 else 0
        assert (
            abs(results["source"][row] - results["ufo-standard"][row] - expected_shift)
            < 2e-15
        )
