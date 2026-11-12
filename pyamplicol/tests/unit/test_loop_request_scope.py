# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from pyamplicol.loops.request_scope import prepare_loop_request_scope

pytestmark = pytest.mark.loop_fast


def _raw():
    def coupling(index, order):
        return dict(
            id=index,
            name=f"C{index}",
            fields={"order": {"kind": "mapping", "entries": order}},
        )

    def vertex(index, coupling_id, ct=False):
        return dict(
            id=index,
            fields={
                "couplings": {
                    "kind": "mapping",
                    "entries": [
                        (
                            {"kind": "tuple", "items": [0, 0, 3] if ct else [0, 0]},
                            {
                                "kind": "reference",
                                "collection": "couplings",
                                "id": coupling_id,
                                "name": f"C{coupling_id}",
                            },
                        )
                    ],
                }
            },
        )

    return SimpleNamespace(
        collections={
            "coupling_orders": [{"name": "QCD"}, {"name": "QED"}],
            "couplings": [
                coupling(0, [("QCD", 1)]),
                coupling(1, [("QED", 2)]),
                coupling(2, [("QCD", 3)]),
                coupling(3, [("QED", 0)]),
            ],
            "vertices": [vertex(0, 0), vertex(1, 1), vertex(2, 2), vertex(3, 3)],
            "counterterm_vertices": [vertex(0, 2, True), vertex(1, 1, True)],
        }
    )


def test_scope_uses_union_upper_bounds_and_preserves_exact_basis_groups():
    raw = _raw()
    scope = prepare_loop_request_scope(
        raw,
        born_order_bounds={"QCD": 1, "QED": 0},
        loop_order_bounds={"QCD": 3, "QED": 0},
    )
    assert scope.ordinary_ceilings == (("QCD", 3), ("QED", 0))
    assert ("vertices", 2, 0, 0, None) in scope.admitted_keys
    assert ("counterterm_vertices", 0, 0, 0, 3) in scope.admitted_keys
    assert len(scope.excluded) == 2
    assert all(
        (item.exceeding_order, item.power, item.ceiling) == ("QED", 2, 0)
        for item in scope.excluded
    )
    with pytest.raises(ValueError, match="different raw"):
        scope.require_raw(copy.deepcopy(raw))
    with pytest.raises(ValueError, match="bounds differ"):
        scope.require_bounds({"QCD": 1}, {"QCD": 3, "QED": 0})
    scope.require_bounds({"QCD": (1, 1), "QED": 0}, {"QCD": 3, "QED": 0})


@pytest.mark.parametrize("born", [None, {}, {"QED": (0, None)}])
def test_unbounded_born_coordinate_cannot_prune_ordinary_rules(born):
    scope = prepare_loop_request_scope(
        _raw(), born_order_bounds=born, loop_order_bounds={"QED": 0}
    )
    assert ("vertices", 1, 0, 0, None) in scope.admitted_keys
    assert ("counterterm_vertices", 1, 0, 0, 3) not in scope.admitted_keys
    assert not scope.ordinary_ceilings


@pytest.mark.parametrize("loop", [None, {}, {"QED": (0, None)}])
def test_unbounded_loop_coordinate_cannot_prune_any_rule(loop):
    scope = prepare_loop_request_scope(
        _raw(), born_order_bounds={"QED": 0}, loop_order_bounds=loop
    )
    assert len(scope.admitted_keys) == 6
    assert not scope.excluded


@pytest.mark.parametrize(
    "change,match",
    [
        (None, "missing explicit"),
        ([("QCD", -1)], "nonnegative"),
        ([("QCD", 1.0)], "nonnegative"),
        ([("UNDECLARED", 0)], "undeclared"),
        ([("QCD", 1), ("QCD", 2)], "unique"),
    ],
)
def test_untrusted_source_order_data_never_justifies_exclusion(change, match):
    raw = _raw()
    fields = raw.collections["couplings"][0]["fields"]
    if change is None:
        del fields["order"]
    else:
        fields["order"]["entries"] = change
    with pytest.raises(ValueError, match=match):
        prepare_loop_request_scope(
            raw, born_order_bounds={"QCD": 0}, loop_order_bounds={"QCD": 0}
        )


def test_local_ct_coverage_is_exact_and_does_not_claim_field_completion():
    raw = _raw()
    scope = prepare_loop_request_scope(
        raw, born_order_bounds={"QED": 0}, loop_order_bounds={"QED": 0}
    )
    expected = ("counterterm_vertices", 0, 0, 0, 3)
    prepared = SimpleNamespace(
        source=SimpleNamespace(raw=raw),
        unavailable_entries=(),
        rules={"r": SimpleNamespace(entries=(SimpleNamespace(key=expected),))},
    )
    scope.require_local_counterterms(prepared)
    prepared.rules = {}
    with pytest.raises(ValueError, match="counterterm bases"):
        scope.require_local_counterterms(prepared)
    prepared.rules = {
        "r": SimpleNamespace(entries=(SimpleNamespace(key=expected),) * 2)
    }
    with pytest.raises(ValueError, match="counterterm bases"):
        scope.require_local_counterterms(prepared)


def test_duplicate_source_basis_is_not_hidden_by_set_coverage():
    raw = _raw()
    entries = raw.collections["vertices"][0]["fields"]["couplings"]["entries"]
    entries.append(copy.deepcopy(entries[0]))
    with pytest.raises(ValueError, match="duplicate local basis"):
        prepare_loop_request_scope(raw)


def test_supplied_species_facts_must_name_actual_species_and_declared_orders():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native model control requires licensed Symbolica")
    from symbolica.community import hepkit

    from tools.loop_reference.feynkit_loop_graphs import _validated_species_orders

    model = hepkit.Model.qcd_qed()
    assert _validated_species_orders(model, {2: {"QCD", "QED"}}) == {
        2: frozenset({"QCD", "QED"})
    }
    for pdg in (-2, True, 9999999):
        with pytest.raises(ValueError, match="positive PDG"):
            _validated_species_orders(model, {pdg: {"QCD"}})
    with pytest.raises(ValueError, match="undeclared order"):
        _validated_species_orders(model, {2: {"UNDECLARED"}})
    with pytest.raises(TypeError, match="collection"):
        _validated_species_orders(model, {2: "QCD"})
