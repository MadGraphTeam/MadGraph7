# SPDX-License-Identifier: 0BSD
"""Source-backed cold proof controls; no process generation or numeric fitting."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.loops.flavor_mappings import RetainedFlavorPlan, prove_flavor_relabeling

SWAP = {1: 3, 3: 1, -1: -3, -3: -1}


@pytest.fixture(scope="module")
def actual_declared_plan():
    symbolica = pytest.importorskip("symbolica")
    path = os.environ.get("PYAMPLICOL_PROJECTED_FLAVOR_MODEL")
    if not path or not symbolica.is_licensed():
        pytest.skip(
            "set PYAMPLICOL_PROJECTED_FLAVOR_MODEL and license for cold model proof"
        )
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.source_color_declaration import (
        prepare_declared_source_color,
        source_color_fixed_parameters,
    )
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    compiled = load_compiled_model(path)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(
            (
                Path(__file__).parents[2]
                / "examples/loop/source_colour/loop_sm_qcd_subset.json"
            ).read_text()
        )
    )
    original = prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters=source_color_fixed_parameters(
            compiled.raw_loop_model, declaration
        ),
    )
    prepared, continuation = prepare_declared_source_color(
        original, declaration, scheme=declaration.scheme
    )
    ordinary = tuple(rule.name for rule in original.structure.native.vertex_rules)
    return (
        RetainedFlavorPlan(
            original.structure, original.counterterms, ordinary, ("QCD",)
        ),
        RetainedFlavorPlan(
            prepared.structure, prepared.counterterms, ordinary, ("QCD",), continuation
        ),
    )


def test_actual_formal_plan_checks_complete_source_domain(actual_declared_plan):
    original, declared = actual_declared_plan
    for plan in (original, declared):
        proof = prove_flavor_relabeling(plan, SWAP)
        assert (
            proof.ordinary_rule_count,
            proof.counterterm_rule_count,
            proof.external_field_rule_count,
        ) == (9, 64, 6)
        assert not proof.external_transport_validated
    whole = [
        rule
        for rule in declared.counterterms.rules.values()
        if rule.whole_rule_continuation is not None
    ]
    assert [(len(rule.entries), len(rule.native_coefficients)) for rule in whole] == [
        (28, 12),
        (29, 24),
    ]
    with pytest.raises(ValueError, match=r"mass|metadata|UV/R2|field"):
        prove_flavor_relabeling(declared, {2: 4, 4: 2, -2: -4, -4: -2})
    with pytest.raises(ValueError, match="formal source continuation"):
        prove_flavor_relabeling(replace(declared, source_color=None), SWAP)


@pytest.mark.parametrize(
    "damage",
    [
        "coefficient",
        "missing",
        "duplicate",
        "source-multiplicity",
        "foreign-owner",
        "tensor",
        "color",
        "order",
        "slot",
    ],
)
def test_actual_whole_rule_replacement_cannot_hide_altered_owned_data(
    actual_declared_plan, damage
):
    from symbolica.community import hepkit as hep

    _, plan = actual_declared_plan
    rule = next(
        rule
        for rule in plan.counterterms.rules.values()
        if rule.whole_rule_continuation is not None
    )
    rules = dict(plan.counterterms.rules)
    native = plan.counterterms.native
    if damage == "coefficient":
        alias, value = rule.native_coefficients[0]
        rules[rule.name] = replace(
            rule,
            native_coefficients=(
                (alias, value + plan.source_color.nc - 3),
                *rule.native_coefficients[1:],
            ),
        )
    elif damage == "missing":
        rules[rule.name] = replace(
            rule, native_coefficients=rule.native_coefficients[:-1]
        )
    elif damage == "duplicate":
        rules[rule.name] = replace(
            rule,
            native_coefficients=(
                rule.native_coefficients[0],
                *rule.native_coefficients[:-1],
            ),
        )
    elif damage == "source-multiplicity":
        rules[rule.name] = replace(rule, entries=(*rule.entries, rule.entries[0]))
    elif damage == "foreign-owner":
        rules[rule.name] = replace(
            rule, whole_rule_continuation=replace(rule.whole_rule_continuation)
        )
    else:
        payload = json.loads(native.to_json())
        vertex = next(
            item for item in payload["vertex_rules"] if item["name"] == rule.name
        )
        if damage == "tensor":
            tensor = next(
                item
                for item in payload["lorentz_structures"]
                if item["name"] == vertex["lorentz_structures"][0]
            )
            tensor["structure"] = "2*(" + tensor["structure"] + ")"
        elif damage == "color":
            vertex["color_structures"][0] = "2*(" + vertex["color_structures"][0] + ")"
        elif damage == "slot":
            vertex["couplings"][0][0] = None
        else:
            coupling = next(
                item
                for item in payload["couplings"]
                if item["name"] == rule.native_coefficients[0][0]
            )
            coupling["orders"] = [
                [name, power + (name == "QCD")] for name, power in coupling["orders"]
            ]
        native = hep.Model.from_json(json.dumps(payload))
    changed = replace(
        plan, counterterms=replace(plan.counterterms, rules=rules, native=native)
    )
    with pytest.raises(ValueError, match=r"whole-rule|continuation"):
        prove_flavor_relabeling(changed, SWAP)
