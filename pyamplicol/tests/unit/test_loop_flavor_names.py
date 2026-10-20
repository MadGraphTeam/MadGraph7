# SPDX-License-Identifier: 0BSD
"""Public spelling uses the retained particle authority, never guessed aliases."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.generation.process_grouping import GroupMember, LegMapping
from pyamplicol.loops.flavor_transport import AdmittedFlavorMember
from pyamplicol.processes.model import ModelParticleCatalog, resolve_model_particle_name

from .test_external_model_independence import _particle

pytestmark = pytest.mark.loop_fast


def test_existing_particle_catalog_name_resolution_is_preserved():
    upper = _particle("G", "G", 21, spin=3, color=8)
    catalog = ModelParticleCatalog("actual", (upper,))
    assert catalog.resolve("G") is upper
    assert catalog.resolve("g") is upper
    hidden = _particle("g", "g", 9001, spin=-1, color=8, ghost_number=1)
    catalog = ModelParticleCatalog("actual", (upper, hidden))
    assert catalog.resolve("g") is upper  # ordinary public external-state filter
    assert catalog.resolve("g", external=False) is hidden
    first = replace(upper, name="Gauge", antiname="Gauge")
    second = replace(upper, name="GAUGE", antiname="GAUGE", pdg_code=9002)
    catalog = ModelParticleCatalog("ambiguous", (first, second))
    assert catalog.resolve("Gauge") is first
    assert catalog.resolve("GAUGE") is second
    with pytest.raises(ValueError, match="case-ambiguous"):
        catalog.resolve("gauge")
    with pytest.raises(ValueError, match="not an external state"):
        catalog.resolve("missing")
    assert resolve_model_particle_name("g", ("G",), model_name="actual") == "G"


def _relation(records=None, native_names=None, fields=None, native_fields=None):
    records = records or {"s": 3, "s~": -3, "G": 21, "A": 22}
    native_names = native_names or {pdg: name for name, pdg in records.items()}
    fields = fields or {}
    native_fields = native_fields or {}
    structure = SimpleNamespace(
        raw=SimpleNamespace(
            name="actual",
            collections={
                "particles": tuple(
                    {
                        "name": name,
                        "fields": {"pdg_code": pdg, "spin": 3, **fields.get(pdg, {})},
                    }
                    for name, pdg in records.items()
                )
            },
        ),
        native=SimpleNamespace(
            to_json=lambda: json.dumps(
                {
                    "particles": [
                        {"pdg_code": pdg, "spin": 3, **native_fields.get(pdg, {})}
                        for pdg in native_names
                    ]
                }
            ),
            particle_by_pdg=lambda pdg: SimpleNamespace(
                name=native_names[pdg], pdg_code=pdg
            ),
        ),
    )
    return SimpleNamespace(
        symmetry=SimpleNamespace(plan=SimpleNamespace(structure=structure)),
        member_request=SimpleNamespace(
            incoming=("s", "s~"), outgoing=("G", "G"), external_pdgs=(3, -3, 21, 21)
        ),
        mapping=LegMapping((0, 1, 2, 3), False, "exact-loop-species"),
    )


def _member(expression):
    return GroupMember(
        "ssgg",
        expression,
        (3, -3, 21, 21),
        2,
        permutation=(0, 1, 2, 3),
        generated=False,
    )


def test_loop_admission_keeps_public_spelling_and_exact_ordered_identity():
    relation = _relation()
    for expression in ("s s~ > G G", "s s~ > g g", "s s~ > G g"):
        member = _member(expression)
        assert AdmittedFlavorMember(relation, member).member.expression == expression
    for expression in ("s s~ > A A", "s~ s > G G", "s > s~ G G", "s s~ G > G"):
        with pytest.raises(ValueError, match="differs from the admitted"):
            AdmittedFlavorMember(relation, _member(expression))
    with pytest.raises(ValueError, match="not an external state"):
        AdmittedFlavorMember(relation, _member("s s~ > unknown G"))
    with pytest.raises(ValueError, match="one concrete collision"):
        AdmittedFlavorMember(relation, _member("s s~ > G G, G > s s~"))
    for changes in (
        {"factor": 2},
        {"conjugated": True},
        {"generated": True},
        {"permutation": (1, 0, 2, 3)},
    ):
        with pytest.raises(ValueError, match="differs from the admitted"):
            AdmittedFlavorMember(relation, replace(_member("s s~ > g g"), **changes))


def test_loop_admission_rejects_case_ambiguity_and_exact_source_native_conflicts():
    relation = _relation({"s": 3, "s~": -3, "Gauge": 21, "GAUGE": 22})
    with pytest.raises(ValueError, match="case-ambiguous"):
        AdmittedFlavorMember(relation, _member("s s~ > gauge gauge"))
    relation = _relation(native_names={3: "s", -3: "s~", 21: "G", 22: "G"})
    with pytest.raises(ValueError, match="ambiguous source/native"):
        AdmittedFlavorMember(relation, _member("s s~ > G G"))


def test_loop_names_share_public_external_eligibility_without_ghost_shadow():
    records = {"s": 3, "s~": -3, "G": 21, "g": 9001}
    for hidden in (
        {"spin": -1},
        {"ghost_number": 1},
        {"goldstoneboson": True},
        {"propagating": False},
        {"auxiliary_kind": "auxiliary"},
    ):
        relation = _relation(records, native_fields={9001: hidden})
        member = _member("s s~ > g g")
        assert AdmittedFlavorMember(relation, member).member is member
    relation = _relation(fields={21: {"GhostNumber": 1}})
    with pytest.raises(ValueError, match="external eligibility differs"):
        AdmittedFlavorMember(relation, _member("s s~ > g g"))
