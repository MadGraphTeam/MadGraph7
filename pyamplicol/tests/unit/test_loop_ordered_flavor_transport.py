# SPDX-License-Identifier: 0BSD
"""Exact paired bank admission over retained independent projected processes."""

from dataclasses import replace

import pytest

RETAINED_ADMISSION = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_ADMISSION is None:
        pytest.skip("actual retained independent complete-family owners required")
    return RETAINED_ADMISSION


def _pair(retained, accuracy):
    from pyamplicol.generation.process_grouping import LegMapping

    sources = tuple(retained["sources"][name, accuracy] for name in ("ddgg", "ssgg"))
    plans = tuple(retained["prepared"][name, accuracy] for name in ("ddgg", "ssgg"))
    classes = tuple(
        tuple(
            item for joint, item in retained["classes"][name, accuracy].items() if joint
        )
        for name in ("ddgg", "ssgg")
    )
    legs = len(sources[0].source_bare.external_request.external_pdgs)
    return (
        sources,
        plans,
        classes,
        LegMapping(tuple(range(legs)), False, "identity-species"),
    )


def _admit(retained, accuracy, *, plans=None, classes=None):
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    sources, ordinary, spin, mapping = _pair(retained, accuracy)
    plans = ordinary if plans is None else plans
    classes = spin if classes is None else classes
    return IdentityFlavorTransport.admit(
        retained["proof"],
        *sources,
        mapping,
        representative_directed=plans[0],
        member_directed=plans[1],
        representative_spin=classes[0],
        member_spin=classes[1],
    )


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_actual_complete_family_and_sources_admit(retained, accuracy):
    sources, plans, classes, _ = _pair(retained, accuracy)
    relation = _admit(retained, accuracy)
    assert relation.representative_directed is plans[0]
    assert relation.member_directed is plans[1]
    assert relation.representative_plan is sources[0].prepared
    assert relation.member_plan is sources[1].prepared
    for source, ordinary, spin in zip(sources, plans, classes, strict=True):
        bank = ordinary.bare.color_bank
        bank.require_owner(source.virtual)
        assert [len(family.histories) for family in bank.projection.families] == [
            1,
            4,
            28,
        ]
        assert all(item.bare.color_bank is bank for item in spin)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("absent", "absent or foreign colour bank"),
        ("foreign-plan", "different retained bare owner"),
        ("foreign-formal", "different projection owner"),
        ("degree", "families or degrees"),
        ("born-degree", "families or degrees"),
        ("family-order", "families or degrees"),
        ("catalogue", "families or degrees"),
        ("assignment", "families or degrees"),
        ("nc", "formal colour parameter"),
        ("accuracy", "different retained bare owner"),
    ),
)
def test_actual_bank_mutations_cannot_borrow_the_species_proof(
    retained, mutation, match
):
    sources, plans, _, _ = _pair(retained, "nlc")
    left, right = plans
    original = left.bare.color_bank
    bank = original
    if mutation == "absent":
        bank = None
    elif mutation == "foreign-plan":
        bank = right.bare.color_bank
    elif mutation == "foreign-formal":
        bank = replace(bank, formal=right.bare.color_bank.formal)
    else:
        projection = bank.projection
        families = projection.families
        if mutation in ("degree", "born-degree"):
            family = families[-1]
            assert family.order == 2
            field = "virtual_degrees" if mutation == "degree" else "born_degrees"
            degrees = getattr(family, field)
            assert any(value is not None for value in degrees)
            changed = tuple(None if value is None else value + 1 for value in degrees)
            projection = replace(
                projection,
                families=(*families[:-1], replace(family, **{field: changed})),
            )
        elif mutation == "family-order":
            projection = replace(projection, families=(*families[:-1], families[0]))
        elif mutation == "catalogue":
            family = families[-1]
            projection = replace(
                projection,
                families=(
                    *families[:-1],
                    replace(family, histories=family.histories[:-1]),
                ),
            )
        elif mutation == "assignment":
            projection = replace(
                projection,
                history_families=tuple(reversed(projection.history_families)),
            )
        elif mutation == "nc":
            projection = replace(projection, nc=projection.nc + (projection.nc - 3))
        elif mutation == "accuracy":
            projection = replace(projection, accuracy="lc")
        bank = replace(bank, projection=projection)
    altered = replace(left, bare=replace(left.bare, color_bank=bank))
    with pytest.raises(ValueError, match=match):
        _admit(retained, "nlc", plans=(altered, right), classes=((), ()))
    # Rejected metadata never replaces either live bank or its actual owner.
    assert left.bare.color_bank is original
    original.require_owner(sources[0].virtual)


def test_source_class_must_share_its_actual_ordinary_bank(retained):
    _, plans, classes, _ = _pair(retained, "nlc")
    first = classes[0][0]
    changed = replace(
        first,
        bare=replace(first.bare, color_bank=replace(first.bare.color_bank)),
    )
    with pytest.raises(ValueError, match="different ordinary colour bank"):
        _admit(retained, "nlc", classes=((changed, *classes[0][1:]), classes[1]))
    assert first.bare.color_bank is plans[0].bare.color_bank


@pytest.mark.parametrize("field", ("phase_states", "helicities"))
def test_matching_source_substitutions_do_not_replace_owner_derivation(retained, field):
    _, _, classes, _ = _pair(retained, "nlc")
    changed = []
    for side in classes:
        first = side[0]
        layout = first.bare.spin_layout
        if field == "phase_states":
            values = (*layout.phase_states[:-1], layout.phase_states[-1] + b"\0")
            match = "model-owned crossing phase"
            bare_changes = {}
        else:
            assert len(layout.helicities) > 1
            values = tuple(reversed(layout.helicities))
            match = "owner spectator helicities"
            bare_changes = {"helicities": values}
        altered = replace(
            first,
            bare=replace(
                first.bare,
                spin_layout=replace(layout, **{field: values}),
                **bare_changes,
            ),
        )
        changed.append((altered, *side[1:]))
    with pytest.raises(ValueError, match=match):
        _admit(retained, "nlc", classes=tuple(changed))


def test_writer_requires_the_actually_paired_ordinary_plan(retained):
    from pyamplicol.generation.process_grouping import GroupMember
    from pyamplicol.loops.flavor_transport import AdmittedFlavorMember

    sources, plans, classes, mapping = _pair(retained, "nlc")
    relation = _admit(retained, "nlc")
    request = relation.member_request
    member = GroupMember(
        "ssgg",
        " ".join((*request.incoming, ">", *request.outgoing)),
        request.external_pdgs,
        len(request.incoming),
        generated=False,
        permutation=mapping.permutation,
    )
    admission = AdmittedFlavorMember(relation, member)
    saved = admission.for_program(
        sources[0].virtual, directed=plans[0], spin_directed=classes[0][::-1]
    )
    assert saved["external_pdgs"] == member.pdgs
    for ordinary in (None, replace(plans[0]), plans[1]):
        with pytest.raises(
            ValueError,
            match=r"saved ordinary directed plan|different retained model plan",
        ):
            admission.for_program(
                sources[0].virtual, directed=ordinary, spin_directed=classes[0]
            )
    for actual in ((), classes[0][:-1], classes[1], (*classes[0], classes[0][0])):
        with pytest.raises(ValueError, match="saved spin classes"):
            admission.for_program(
                sources[0].virtual, directed=plans[0], spin_directed=actual
            )
