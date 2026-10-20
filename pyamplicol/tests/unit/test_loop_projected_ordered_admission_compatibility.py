# SPDX-License-Identifier: 0BSD
"""Retained safe composition versus paired projected nonidentity admission.

These controls bind existing instructions and inspect writer admission. They
neither construct process graphs nor evaluate another physical point.
"""

from dataclasses import replace

import pytest

RETAINED_ADMISSION = None
RETAINED_FIXED_FULL = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_ADMISSION is None:
        pytest.skip("actual retained projected source owners required")
    return RETAINED_ADMISSION


@pytest.fixture(scope="module")
def fixed_full():
    if RETAINED_FIXED_FULL is None:
        pytest.skip("actual retained fixed-full source owners required")
    return RETAINED_FIXED_FULL


def _relation(proof, sources, plans=None):
    from pyamplicol.generation.process_grouping import LegMapping
    from pyamplicol.loops.flavor_transport import IdentityFlavorTransport

    legs = len(sources[0].source_bare.external_request.external_pdgs)
    paired = (
        {}
        if plans is None
        else {
            "representative_directed": plans[0],
            "member_directed": plans[1],
        }
    )
    return IdentityFlavorTransport.admit(
        proof,
        *sources,
        LegMapping(tuple(range(legs)), False, "exact-loop-species"),
        **paired,
    )


def _admission(relation):
    from pyamplicol.generation.process_grouping import GroupMember
    from pyamplicol.loops.flavor_transport import AdmittedFlavorMember

    request = relation.member_request
    return AdmittedFlavorMember(
        relation,
        GroupMember(
            "ssgg",
            " ".join((*request.incoming, ">", *request.outgoing)),
            request.external_pdgs,
            len(request.incoming),
            generated=False,
            permutation=relation.mapping.permutation,
        ),
    )


def _bind_and_export(relation, source, directed):
    from pyamplicol.loops.flavor_transport import NativeIdentityFlavorProgram

    original = source.source_bare.state.snapshot()
    directed.require_owner(source.virtual)
    for mapped in (
        relation.bind(source, directed=directed),
        NativeIdentityFlavorProgram(relation, source, directed),
    ):
        assert mapped.directed_plan is directed
        assert mapped.representative.prepared is source.prepared
        rebound = mapped.rebind(arithmetic="double")
        assert rebound.directed_plan is directed
        assert rebound.representative.prepared is source.prepared
    saved = _admission(relation).for_program(source.virtual, directed=directed)
    assert saved["external_pdgs"] == relation.member_request.external_pdgs
    assert source.source_bare.state.snapshot() is original


def _identity_pair(retained, accuracy):
    sources = tuple(
        retained["identity_sources"][name, accuracy] for name in ("ddgg", "ssgg")
    )
    plans = tuple(
        retained["identity_catalogues"][name, accuracy][0] for name in ("ddgg", "ssgg")
    )
    assert all(
        not connection.emissions
        for plan in plans
        for pair in plan.bare.histories
        for connection in pair
    )
    assert all(plan.bare.color_bank is None for plan in plans)
    return sources, plans


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_ordinary_only_projected_identity_still_binds_and_exports(retained, accuracy):
    sources, plans = _identity_pair(retained, accuracy)
    relation = _relation(retained["proof"], sources)
    assert relation.representative_directed is None
    assert sources[0].virtual.color_projection is not None
    _bind_and_export(relation, sources[0], plans[0])


def test_ordinary_only_fixed_full_ordered_still_binds_and_exports(fixed_full):
    sources, plans = fixed_full["programs"], fixed_full["ordinary"]
    assert all(source.virtual.color_projection is None for source in sources)
    assert all(plan.bare.color_bank is None for plan in plans)
    assert any(
        connection.emissions for pair in plans[0].bare.histories for connection in pair
    )
    relation = _relation(fixed_full["relation"].symmetry, sources)
    assert relation.representative_directed is None
    _bind_and_export(relation, sources[0], plans[0])


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_paired_identity_retains_exact_object_authority(retained, accuracy):
    sources, plans = _identity_pair(retained, accuracy)
    relation = _relation(retained["proof"], sources, plans)
    _bind_and_export(relation, sources[0], plans[0])
    # The copied record still belongs to the same physical model. It is not
    # the specific ordinary preparation that was paired during this admission.
    substituted = replace(plans[0])
    substituted.require_owner(sources[0].virtual)
    with pytest.raises(ValueError, match="paired admission"):
        relation.bind(sources[0], directed=substituted)
    with pytest.raises(ValueError, match="paired flavour admission"):
        _admission(relation).for_program(sources[0].virtual, directed=substituted)


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_ordinary_only_cannot_gain_projected_ordered_capability(retained, accuracy):
    from pyamplicol.loops.flavor_transport import NativeIdentityFlavorProgram

    sources = tuple(retained["sources"][name, accuracy] for name in ("ddgg", "ssgg"))
    directed = retained["prepared"]["ddgg", accuracy]
    relation = _relation(retained["proof"], sources)
    directed.require_owner(sources[0].virtual)
    assert directed.bare.color_bank is not None
    assert any(
        connection.emissions for pair in directed.bare.histories for connection in pair
    )
    with pytest.raises(ValueError, match="paired admission"):
        relation.bind(sources[0], directed=directed)
    with pytest.raises(ValueError, match="paired admission"):
        NativeIdentityFlavorProgram(relation, sources[0], directed)
    with pytest.raises(ValueError, match="paired flavour admission"):
        _admission(relation).for_program(sources[0].virtual, directed=directed)
    # An ordinary rebind is still usable and must enforce the same boundary.
    rebound = relation.bind(sources[0]).rebind(arithmetic="double")
    with pytest.raises(ValueError, match="paired admission"):
        NativeIdentityFlavorProgram(relation, rebound.representative, directed)


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_paired_projected_ordered_bind_and_rebind_keep_exact_plan(retained, accuracy):
    from pyamplicol.loops.flavor_transport import NativeIdentityFlavorProgram

    sources = tuple(retained["sources"][name, accuracy] for name in ("ddgg", "ssgg"))
    plans = tuple(retained["prepared"][name, accuracy] for name in ("ddgg", "ssgg"))
    relation = _relation(retained["proof"], sources, plans)
    _bind_and_export(relation, sources[0], plans[0])
    substituted = replace(plans[0])
    substituted.require_owner(sources[0].virtual)
    with pytest.raises(ValueError, match="paired admission"):
        NativeIdentityFlavorProgram(relation, sources[0], substituted)
    with pytest.raises(ValueError, match="paired flavour admission"):
        _admission(relation).for_program(sources[0].virtual, directed=substituted)
