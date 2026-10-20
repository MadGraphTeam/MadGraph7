# SPDX-License-Identifier: 0BSD
"""Exact fixed-component proofs never substitute scalar or formal colour data."""

import json
from dataclasses import replace

import pytest

from pyamplicol.loops.vertex_symmetry import symmetric_boson_slots

from .test_loop_vertex_symmetry import _aliased_whole_vertex

# Supplied only by the retained graph-free declared-model owner when requested.
RETAINED_FORMAL_PLAN = None


@pytest.fixture
def hep():
    pytest.importorskip("symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def test_fixed_components_keep_default_symbolic_proof_and_physical_asymmetry(hep):
    model = hep.Model.qcd_qed()
    expected = {"V_36": (0, 0, 0)}
    assert symmetric_boson_slots(model, ["V_36"]) == expected
    assert symmetric_boson_slots(model, ["V_36"], fixed_representation=True) == expected
    payload = json.loads(model.to_json())
    lorentz = next(
        item for item in payload["lorentz_structures"] if item["name"] == "VVV1"
    )
    # The first two vector indices form a metric; the third contracts only its
    # own momentum. This complete off-shell tensor is not Bose symmetric.
    lorentz["structure"] = "Metric(idx(1,1),idx(1,2))*P(idx(1,3),idx(1,3))"
    changed = hep.Model.from_json(json.dumps(payload))
    assert symmetric_boson_slots(changed, ["V_36"]) == {}
    assert symmetric_boson_slots(changed, ["V_36"], fixed_representation=True) == {}


def test_fixed_components_preserve_epsilon_squared_scalar_default_asymmetry(hep):
    from symbolica import E

    from pyamplicol.loops.model_expressions import epsilon_symbol

    model, aliases = _aliased_whole_vertex(hep)
    symbol, coupling = E("bose_fixed_control::Nc"), E("bose_fixed_control::g")
    coefficients = dict.fromkeys(aliases, coupling**2)
    coefficients[aliases[0]] += epsilon_symbol() ** 2 * coupling**2 * (symbol - 3)
    assert all(item.replace(symbol, 3) == coupling**2 for item in coefficients.values())
    assert (
        symmetric_boson_slots(model, ["V_37"], coupling_expressions=coefficients) == {}
    )
    assert (
        symmetric_boson_slots(
            model,
            ["V_37"],
            coupling_expressions=coefficients,
            fixed_representation=True,
            conserve_momentum=True,
        )
        == {}
    )


def test_fixed_components_reject_different_typed_external_interfaces(hep):
    payload = json.loads(hep.Model.qcd_qed().to_json())
    lorentz = next(
        item for item in payload["lorentz_structures"] if item["name"] == "VVV1"
    )
    # Deliberately omit the first vector port so the FIRST tested generator
    # changes the interface. Omitting the third port would be rejected earlier
    # as an unproved antisymmetric tensor, without reaching this boundary.
    lorentz["structure"] = "Metric(idx(1,2),idx(1,3))"
    changed = hep.Model.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match=r"port|interface|indices|structure"):
        symmetric_boson_slots(changed, ["V_36"], fixed_representation=True)


def test_actual_formal_owner_never_enters_fixed_component_fallback(monkeypatch):
    if RETAINED_FORMAL_PLAN is None:
        pytest.skip("requires retained graph-free formal source owner")
    from pyamplicol.loops import vertex_symmetry
    from pyamplicol.loops.flavor_transport import _bose_local_rules

    plan = RETAINED_FORMAL_PLAN
    assert plan.source_color is not None
    observed = []
    original = vertex_symmetry.symmetric_boson_slots

    def require_symbolic(*args, **kwargs):
        observed.append(kwargs.get("fixed_representation", False))
        assert kwargs.get("fixed_representation", False) is False
        return original(*args, **kwargs)

    monkeypatch.setattr(vertex_symmetry, "symmetric_boson_slots", require_symbolic)
    binding = _bose_local_rules(plan)
    assert observed == [False]
    assert binding.program is plan.structure.scalars
    assert binding.fixed_parameters == plan.structure.fixed_parameters
    # Removing the actual continuation pointer cannot reinterpret whole-rule
    # aliases as fixed-colour native coefficients.
    with pytest.raises(ValueError, match=r"whole-rule.*formal"):
        _bose_local_rules(replace(plan, source_color=None))
