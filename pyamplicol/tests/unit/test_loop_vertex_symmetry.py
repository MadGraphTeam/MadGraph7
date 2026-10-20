# SPDX-License-Identifier: 0BSD
"""Exact Bose-port proofs and the rooted-current false-negative regression."""

import json
import re

import pytest

from pyamplicol.loops.counterterms import _rooted_current_key
from pyamplicol.loops.vertex_symmetry import symmetric_boson_slots


@pytest.fixture
def hep():
    pytest.importorskip("symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _star(hep, model):
    return hep.FeynmanDiagram.from_dot(
        model,
        'digraph {ext[style=invis];v[int_id="V_36"];'
        'ext->v[particle="g"];ext->v[particle="g"];ext->v[particle="g"];}',
    )


def test_full_vertices_prove_bose_symmetry_and_preserve_external_root_identity(hep):
    model = hep.Model.qcd_qed()
    classes = symmetric_boson_slots(model, ["V_36", "V_37"])
    assert classes == {"V_36": (0, 0, 0), "V_37": (0, 0, 0, 0)}
    current = _star(hep, model)
    # Exchange a physical port and the artificial boundary, as in the actual
    # five-leg gluon-current attachment; physical external labels stay fixed.
    encoded = re.sub(
        r"target_slot=([02])",
        lambda match: f"target_slot={2 - int(match.group(1))}",
        current.to_dot(),
    )
    swapped = hep.FeynmanDiagram.from_dot(model, encoded)
    assert _rooted_current_key(current, {0, 1}) != _rooted_current_key(swapped, {0, 1})
    assert _rooted_current_key(current, {0, 1}, classes) == _rooted_current_key(
        swapped, {0, 1}, classes
    )
    assert _rooted_current_key(current, {0, 1}, classes) != _rooted_current_key(
        current, {0, 2}, classes
    )


@pytest.mark.parametrize("kind", ["color_only", "asymmetric_lorentz", "coupling"])
def test_proof_uses_complete_tensor_and_current_model_couplings(hep, kind):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    rule_name = "V_36"
    if kind == "coupling":
        rule_name = "V_37"
        rule = next(
            rule for rule in payload["vertex_rules"] if rule["name"] == rule_name
        )
        rule["couplings"][0][0] = "GC_10"
    else:
        lorentz = next(
            item for item in payload["lorentz_structures"] if item["name"] == "VVV1"
        )
        # Colour f alone has cyclic symmetry, which must not be mistaken for
        # permission to identify every slot under the full permutation group.
        lorentz["structure"] = (
            "1"
            if kind == "color_only"
            else "Metric(idx(1,1),idx(1,2))*P(idx(1,3),idx(1,1))"
        )
    changed = hep.Model.from_json(json.dumps(payload))
    assert symmetric_boson_slots(changed, [rule_name]) == {}
    # No stale rule-name-only certificate leaks between distinct native models.
    assert rule_name in symmetric_boson_slots(hep.Model.qcd_qed(), [rule_name])


def _aliased_whole_vertex(hep):
    """Separate native aliases retain the same complete four-vector rule."""
    payload = json.loads(hep.Model.qcd_qed().to_json())
    rule = next(rule for rule in payload["vertex_rules"] if rule["name"] == "V_37")
    couplings = {entry["name"]: entry for entry in payload["couplings"]}
    aliases = []
    for row_index, row in enumerate(rule["couplings"]):
        for column, name in enumerate(row):
            if name is None:
                continue
            alias = f"bose_whole_{row_index}_{column}"
            payload["couplings"].append({**couplings[name], "name": alias})
            row[column] = alias
            aliases.append(alias)
    return hep.Model.from_json(json.dumps(payload)), aliases


def test_exact_alias_expansion_proves_the_complete_vertex(hep):
    from symbolica import E

    model, aliases = _aliased_whole_vertex(hep)
    assert symmetric_boson_slots(model, ["V_37"]) == {}
    coefficient = E("bose_test::g^2*(bose_test::Nc^2-1)/bose_test::Nc")
    expansions = dict.fromkeys(aliases, coefficient)
    assert symmetric_boson_slots(model, ["V_37"], coupling_expressions=expansions) == {
        "V_37": (0, 0, 0, 0)
    }
    # No rule-name cache or model mutation can turn the unexpanded aliases equal.
    assert symmetric_boson_slots(model, ["V_37"]) == {}


@pytest.mark.parametrize("conserve_momentum", (False, True))
@pytest.mark.parametrize("power", (0, 2))
def test_complete_vertex_cannot_hide_formal_laurent_asymmetry(
    hep, power, conserve_momentum
):
    from symbolica import E

    from pyamplicol.loops.model_expressions import epsilon_symbol

    model, aliases = _aliased_whole_vertex(hep)
    nc, g = E("bose_test::Nc"), E("bose_test::g")
    expansions = dict.fromkeys(aliases, g**2)
    expansions[aliases[0]] += epsilon_symbol() ** power * g**2 * (nc - 3)
    # At physical colour both source values coincide; the exact formal rule does
    # not have Bose closure, including when only a positive epsilon power differs.
    assert all(value.replace(nc, 3) == g**2 for value in expansions.values())
    assert (
        symmetric_boson_slots(
            model,
            ["V_37"],
            coupling_expressions=expansions,
            conserve_momentum=conserve_momentum,
        )
        == {}
    )


def test_coupling_expansion_rejects_unknown_or_unresolved_native_aliases(hep):
    from symbolica import E

    model, aliases = _aliased_whole_vertex(hep)
    with pytest.raises(ValueError, match="unknown native coupling"):
        symmetric_boson_slots(model, ["V_37"], coupling_expressions={"absent": E("1")})
    with pytest.raises(ValueError, match="fully resolved"):
        symmetric_boson_slots(
            model,
            ["V_37"],
            coupling_expressions={aliases[0]: model.coupling(aliases[1]).symbol},
        )


def test_local_two_point_conservation_is_off_shell_and_opt_in(hep):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    original = next(rule for rule in payload["vertex_rules"] if rule["name"] == "V_36")
    payload["lorentz_structures"].append(
        {
            "name": "BoseTwoPoint",
            "spins": [3, 3],
            "structure": ("Metric(idx(1,1),idx(1,2))*P(dummy(1),idx(1,1))^2"),
        }
    )
    payload["vertex_rules"].append(
        {
            "name": "BoseTwoPoint",
            "particles": [original["particles"][0]] * 2,
            "color_structures": ["Identity(idx(1,1),idx(1,2))"],
            "lorentz_structures": ["BoseTwoPoint"],
            "couplings": [[original["couplings"][0][0]]],
        }
    )
    model = hep.Model.from_json(json.dumps(payload))
    assert symmetric_boson_slots(model, ["BoseTwoPoint"]) == {}
    assert symmetric_boson_slots(model, ["BoseTwoPoint"], conserve_momentum=True) == {
        "BoseTwoPoint": (0, 0)
    }
    particle = original["particles"][0]
    star = hep.FeynmanDiagram.from_dot(
        model,
        'digraph {ext[style=invis];v[int_id="BoseTwoPoint"];'
        + f'ext->v[particle="{particle}"];ext->v[particle="{particle}"];}}',
    ).apply_feynman_rules()
    # Routing retains a nonzero generic p² numerator: no on-shell zero was used.
    assert (
        star.loop_momentum_basis.route_expression(
            star.vertices[0].numerator_expression().to_expression()
        )
        != 0
    )
