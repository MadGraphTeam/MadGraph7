# SPDX-License-Identifier: 0BSD
"""Explicit bundle colour input binds exact shared source values and constraints."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.loops.scalar_bindings import (
    NativeScalarBindings,
    prepare_scalar_bindings,
)
from pyamplicol.loops.source_color_declaration import (
    prepare_declared_source_color,
    source_color_fixed_parameters,
)
from pyamplicol.models.loop_color_declaration import (
    LoopColorDeclaration,
    WholeRuleColorRecipe,
)
from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

from .test_loop_source_color import source

pytestmark = pytest.mark.loop_fast

__all__ = ["source"]


def declared(**changes):
    value = LoopColorDeclaration(
        scheme=LoopSchemeDeclaration(
            "HV", "4d", "none", "test", "real", "none", "synthetic"
        ),
        formal_symbol="Nc",
        parameter_identities={
            "Ncol": "Nc",
            "CA": "Nc",
            "CF": "(Nc**2-1)/(2*Nc)",
            "TF": "0.5",
        },
        kernel_keys=(),
        field_keys=(),
        scheme_parameter_values={"external_colour": "3", "Ncol": "3"},
    )
    return replace(value, **changes)


def prepared(source, fixed=None):
    from pyamplicol.loops.model import build_loop_model

    plan = prepare_scalar_bindings(source, (("a", 0),), fixed_parameters=fixed)
    structure = SimpleNamespace(
        raw=source,
        scalar_bindings=lambda outputs: prepare_scalar_bindings(
            source, outputs, program=plan.program, fixed_parameters=fixed
        ),
    )
    return SimpleNamespace(
        model=build_loop_model(source, scalar_program=plan.program),
        structure=structure,
        counterterms=SimpleNamespace(rules={}, unavailable_entries=()),
    )


def test_exact_scheme_ties_merge_consistently_and_reject_atomic_updates(source):
    data = declared()
    exact = "0.10000000000000000000000000000000000000001"
    fixed = source_color_fixed_parameters(
        source,
        data,
        fixed_parameters={"a": exact},
        parameters={"external_colour": "3.0"},
    )
    assert fixed == {"a": (exact, "0"), "external_colour": ("3", "0")}
    assert "Ncol" not in fixed  # Internal formulas are proved, not overwritten.
    model = prepared(source, fixed)
    same, continuation = prepare_declared_source_color(model, data, scheme=data.scheme)
    assert same is model and continuation.model is model.model
    owner = model.structure.scalar_bindings((("GCplain", 0),))
    runtime = NativeScalarBindings(owner)
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="fixed"):
        runtime.update({"z": ("7", "9"), "external_colour": "4"})
    assert runtime.snapshot() is before
    for field in ("fixed_parameters", "parameters"):
        with pytest.raises(ValueError, match="conflicts with an explicit value"):
            source_color_fixed_parameters(
                source, data, **{field: {"external_colour": "4"}}
            )
    with pytest.raises(ValueError, match="must be finite"):
        source_color_fixed_parameters(
            source, declared(scheme_parameter_values={"external_colour": "NaN"})
        )


def test_default_numeric_equality_cannot_supply_a_missing_fixed_tie(source):
    data = declared()
    with pytest.raises(ValueError, match="must remain fixed"):
        prepare_declared_source_color(prepared(source), data, scheme=data.scheme)
    fixed = source_color_fixed_parameters(source, data)
    bad = declared(scheme_parameter_values={"Ncol": "4"})
    with pytest.raises(ValueError, match="must remain fixed"):
        prepare_declared_source_color(prepared(source, fixed), bad, scheme=bad.scheme)


def test_no_scalar_identities_or_ties_need_no_invented_scalar_output(source):
    data = declared(parameter_identities={}, scheme_parameter_values={})
    model = prepared(source)
    same, continuation = prepare_declared_source_color(model, data, scheme=data.scheme)
    assert same is model
    assert not continuation.parameter_identities
    assert not continuation.scheme_parameter_values


def test_symbolically_constant_scheme_value_does_not_discard_its_diagnostic(source):
    data = declared(scheme_parameter_values={"zero": "0"})
    with pytest.raises(ValueError, match="unresolved active diagnostic"):
        prepare_declared_source_color(prepared(source), data, scheme=data.scheme)


def test_exact_identity_dependencies_and_original_active_diagnostics(source):
    from symbolica import S

    data = declared(
        parameter_identities={"adjustableCA": "Nc*a", "safeCA": "Nc*safeCA/3"}
    )
    model = prepared(source, source_color_fixed_parameters(source, data))
    _, continuation = prepare_declared_source_color(model, data, scheme=data.scheme)
    scalar = model.structure.scalar_bindings((("adjustableCA", 0), ("safeCA", 0)))
    colour_plan = continuation.scalars(
        scalar.outputs, program=scalar.program, fixed_parameters=scalar.fixed_parameters
    )
    formal = colour_plan.formal
    for actual, original in zip(formal.expressions, scalar.expressions, strict=True):
        assert (
            actual.value.replace(continuation.nc, 3).together()
            == original.value.together()
        )
        assert actual.diagnostic == original.diagnostic
    assert formal.expressions[1].diagnostic != 0
    assert S("UFO::a") not in formal.expressions[0].value.get_all_symbols(False)
    assert set(formal.expressions[0].value.get_all_symbols(False)) <= {
        continuation.nc,
        *scalar.input_symbols,
    }
    runtime = NativeScalarBindings(colour_plan.coefficients)
    before = runtime.snapshot()
    with pytest.raises(ArithmeticError, match="active diagnostics"):
        runtime.update({"a": "-1"})
    assert runtime.snapshot() is before


@pytest.mark.parametrize(
    "change,error",
    [
        ({"formal_symbol": "a"}, "collides"),
        ({"parameter_identities": {"CA": "Nc+hidden"}}, "unbound symbols"),
        ({"parameter_identities": {"CA": "Nc+1"}}, "differs from source"),
        ({"parameter_identities": {"CA": "Nc*safeCA/3"}}, "diagnostic-bearing"),
        ({"kernel_keys": (("vertices", 98, 0, 0, None),)}, "do not belong"),
        ({"field_keys": ((0, "QCD", 0),)}, "actual field rule"),
        (
            {
                "whole_rules": (
                    WholeRuleColorRecipe(
                        "counterterm_vertices", 98, 0, (("QCD", 4),), "a", "gauge"
                    ),
                )
            },
            "no actual source group",
        ),
    ],
)
def test_authoritative_source_binding_rejects_incomplete_data(source, change, error):
    data = declared(**change)
    model = prepared(source, source_color_fixed_parameters(source, data))
    with pytest.raises((ValueError, NotImplementedError), match=error):
        prepare_declared_source_color(model, data, scheme=data.scheme)


def test_effective_scheme_must_match_explicit_colour_scheme(source):
    data = declared()
    model = prepared(source, source_color_fixed_parameters(source, data))
    with pytest.raises(ValueError, match="effective loop scheme differ"):
        prepare_declared_source_color(
            model, data, scheme=replace(data.scheme, dimensional_regularization="FDH")
        )
