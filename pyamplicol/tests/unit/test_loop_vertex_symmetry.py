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
