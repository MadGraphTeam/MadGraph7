# SPDX-License-Identifier: 0BSD
"""Explicit bundle colour input binds exact shared source values and constraints."""

import json
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


def source_declaration(prepared):
    """Explicit SU(N_c) source continuation of every kernel but the four-gluon R2
    (from the removed test_loop_source_color_physical.py)."""
    from symbolica import E, S

    from pyamplicol.loops.source_color import SourceColorContinuation
    from pyamplicol.loops.wavefunction import prepare_external_field_rules
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    nc = S("source_dy_control::Nc")
    # Explicit test model input: standard SUN tensors plus CA/CF/TF/Ncol.
    # Fixed-SU3 four-gluon R2 has deliberately NOT been admitted. This fixture
    # does not certify all entries of loop_sm or a public model-name domain.
    keys = frozenset(
        entry.key
        for entry in prepared.model.kernels
        if not (
            entry.kind == "R2"
            and len(entry.particle_ids) == 4
            and all(
                prepared.model.raw.collections["particles"][i]["fields"]["spin"] == 3
                and prepared.model.raw.collections["particles"][i]["fields"]["color"]
                == 8
                for i in entry.particle_ids
            )
        )
    )
    return SourceColorContinuation(
        prepared.model,
        nc,
        {"Ncol": nc, "CA": nc, "CF": (nc**2 - 1) / (2 * nc), "TF": E("1/2")},
        keys,
        frozenset(
            (rule.particle_id, rule.correction, rule.loop_group_index)
            for rule in prepare_external_field_rules(prepared.model)
        ),
        LoopSchemeDeclaration(
            "HV",
            "physical-four-dimensional",
            "source-4d",
            "source-QCD",
            "real",
            "source-local-rules",
            "explicit test source continuation",
        ),
    )


def test_real_source_whole_rule_recipes_rebind_after_json_roundtrip(loop_sm_ufo):
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
        with_loop_color_declaration,
    )
    from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

    path = str(loop_sm_ufo)
    raw = load_raw_loop_model(path)
    options = dict(
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    original = prepare_loop_request_model(raw, **options)
    base = source_declaration(original)
    recipes = []
    for rule in original.counterterms.rules.values():
        first = rule.entries[0]
        fields = [
            original.model.raw.collections["particles"][i]["fields"]
            for i in first.particle_ids
        ]
        if (
            rule.kind != "R2"
            or len(fields) != 4
            or any((field["spin"], field["color"]) != (3, 8) for field in fields)
        ):
            continue
        gauge = all(
            alternative == (first.particle_ids[0],)
            for alternative in first.loop_particles
        )
        recipes.append(
            WholeRuleColorRecipe(
                first.collection,
                first.vertex_id,
                first.loop_group_index,
                first.order,
                "G",
                "gauge" if gauge else "fundamental-dirac",
            )
        )
    assert len(recipes) == 2
    data = LoopColorDeclaration(
        scheme=base.scheme,
        formal_symbol="Nc",
        parameter_identities={
            "Ncol": "Nc",
            "CA": "Nc",
            "CF": "(Nc**2-1)/(2*Nc)",
            "TF": "1/2",
        },
        kernel_keys=tuple(base.kernel_keys),
        field_keys=tuple(base.field_keys),
        scheme_parameter_values={"lhv": "1"},
        whole_rules=tuple(recipes),
    )
    restored = RawLoopModel.from_dict(
        json.loads(json.dumps(with_loop_color_declaration(raw, data).to_dict()))
    )
    declared_data = resolve_loop_color_declaration(restored)
    actual = prepare_loop_request_model(restored, **options)
    continued, owned = prepare_declared_source_color(
        actual, declared_data, scheme=data.scheme
    )
    assert continued is not actual
    assert continued.model is actual.model
    assert (
        continued.counterterms.source_entries_complete
        == actual.counterterms.source_entries_complete
    )
    assert len(owned.whole_rule_continuations) == 2
    assert (
        sum(len(item.source_entries) for item in owned.whole_rule_continuations) == 57
    )
    for item in owned.whole_rule_continuations:
        assert item.key in {recipe.key for recipe in recipes}
        assert all(
            any(entry is candidate for candidate in actual.model.kernels)
            for entry in item.source_entries
        )
        assert not any(
            entry is candidate
            for entry in item.source_entries
            for candidate in original.model.kernels
        )
        assert item.loop_particles is item.source_entries[0].loop_particles
    for rule in continued.counterterms.rules.values():
        if rule.whole_rule_continuation is not None:
            assert rule.entries == rule.whole_rule_continuation.source_entries
            assert rule.active_coupling_symbols == rule.native_coupling_symbols
    # A syntactically valid recipe with the wrong source coupling fails the
    # complete independent Nc=3 tensor check; metadata presence is insufficient.
    wrong = replace(data, whole_rules=(replace(recipes[0], gauge_coupling="2*G"),))
    with pytest.raises(ValueError, match="whole source tensor differs"):
        prepare_declared_source_color(actual, wrong, scheme=data.scheme)
