# SPDX-License-Identifier: 0BSD
"""Complete ordered families include original adjoint splitting without a cap."""

import pytest

from pyamplicol.color.connection_catalogue import (
    build_color_correlator_catalogue,
    canonical_color_connections,
)
from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    EmitGluon,
    SplitGluon,
)


def _contains(catalogue, bra, ket):
    left, right = canonical_color_connections((bra, ket))
    return any(
        item.bra == left.emissions and item.ket == right.emissions for item in catalogue
    )


def test_default_soft_catalogue_and_explicit_full_order_one():
    legs = (ColorLeg(1, 8),)
    soft = build_color_correlator_catalogue(legs, through_order=3)
    assert [x.to_json_dict() for x in soft] == [
        x.to_json_dict()
        for x in build_color_correlator_catalogue(
            legs, through_order=3, include_physical_splittings=False
        )
    ]
    full = build_color_correlator_catalogue(
        legs, through_order=1, include_physical_splittings=True
    )
    assert len(full) == 3
    assert sum(isinstance(x.bra[0], EmitGluon) for x in full) == 1
    assert sum(isinstance(x.bra[0], SplitGluon) for x in full) == 2
    assert _contains(
        full,
        ColorConnection(legs, (SplitGluon(1, 17, 23),)),
        ColorConnection(legs, (SplitGluon(1, 17, 23),)),
    )


def test_physical_split_daughter_and_parent_emission_are_both_present():
    legs = (ColorLeg(1, 8),)
    a = ColorConnection(legs, (SplitGluon(1, 11, 12), EmitGluon(11, 13)))
    b = ColorConnection(legs, (EmitGluon(1, 13), SplitGluon(1, 11, 12)))
    catalogue = build_color_correlator_catalogue(
        legs, through_order=2, include_physical_splittings=True
    )
    assert _contains(catalogue, a, b) and _contains(catalogue, b, a)
    left, right = canonical_color_connections((a, b))
    assert left.emissions != right.emissions
    assert left.output_legs == right.output_legs
    assert not any(leg.label == 1 for leg in left.output_legs)
    assert left.emissions[0].parent_label == 1
    assert right.emissions[0].emitter_label == 1


def test_combined_physical_and_auxiliary_splits_keep_original_roles_fixed():
    legs = (ColorLeg(1, 8),)
    a = ColorConnection(
        legs,
        (EmitGluon(1, -90), SplitGluon(-90, -10, -20), SplitGluon(1, -30, -40)),
    )
    b = ColorConnection(
        legs,
        (EmitGluon(1, 90), SplitGluon(1, -30, -40), SplitGluon(90, -10, -20)),
    )
    left, right = canonical_color_connections((a, b))
    assert left == right  # Independent splits commute, original emission stays first.
    assert len(left.output_legs) == 4 and left.order == 3
    assert {leg.label for leg in left.output_legs} == {-1, -2, -3, -4}
    assert {x.parent_label for x in left.emissions if isinstance(x, SplitGluon)} == {
        1,
        -5,
    }
    catalogue = build_color_correlator_catalogue(
        legs, through_order=3, include_physical_splittings=True
    )
    assert _contains(catalogue, a, b)


def test_one_final_role_bijection_preserves_ordered_relative_pair():
    legs = (ColorLeg(1, 3), ColorLeg(2, -3))
    a = ColorConnection(legs, (EmitGluon(1, 17), EmitGluon(1, 29)))
    b = ColorConnection(legs, (EmitGluon(1, 29), EmitGluon(1, 17)))
    ca, cb = canonical_color_connections((a, b))
    assert ca != cb
    assert ca.emissions == (EmitGluon(1, -1), EmitGluon(1, -2))
    assert cb.emissions == (EmitGluon(1, -2), EmitGluon(1, -1))
    with pytest.raises(ValueError, match="matching order and typed legs"):
        canonical_color_connections((a, ColorConnection(legs)))
    with pytest.raises(TypeError, match="boolean"):
        build_color_correlator_catalogue(
            legs, through_order=1, include_physical_splittings=1
        )
