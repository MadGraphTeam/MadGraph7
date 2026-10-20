# SPDX-License-Identifier: 0BSD
"""Independent cold Bose/source controls over retained actual preparations.

The injected fixture never generates graphs. Pure three-cycle controls use
explicit expected labels; actual owner/phase controls require the separately
retained pair and make no numerical or independent-physics claim.
"""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    EmitGluon,
    SplitGluon,
)
from pyamplicol.color.correlator_matrices import ColorCorrelator

RETAINED_BOSE_SOURCES = None


def _permuted(history, permutation, *, physical_labels=None):
    from pyamplicol.loops.correlation_labels import permute_native_color_history

    return permute_native_color_history(
        history,
        permutation,
        physical_labels=(
            tuple(range(len(permutation)))
            if physical_labels is None
            else physical_labels
        ),
    )


def test_colourless_slot_number_is_an_auxiliary_label_not_a_physical_port():
    initial = tuple(ColorLeg(i, rep) for i, rep in enumerate((-3, 3, 8)))
    steps = (EmitGluon(0, 3), SplitGluon(3, 9, -2))
    history = ColorCorrelator("positive-auxiliary", steps, steps)
    actual = _permuted(history, (0, 1, 2, 4, 3), physical_labels=(0, 1, 2))
    assert actual == history
    assert ColorConnection(initial, actual.bra).output_legs == ColorConnection(
        initial, steps
    ).output_legs


def test_partial_physical_set_moves_retired_adjoint_but_preserves_positive_auxiliary():
    initial = tuple(ColorLeg(i, rep) for i, rep in ((0, -3), (1, 3), (2, 8), (5, 8)))
    steps = (SplitGluon(2, 3, -2), EmitGluon(3, 9))
    history = ColorCorrelator("retired-physical", steps, steps)
    actual = _permuted(
        history, (0, 1, 5, 4, 3, 2), physical_labels=(0, 1, 2, 5)
    )
    expected = (SplitGluon(5, 3, -2), EmitGluon(3, 9))
    assert actual.bra == actual.ket == expected
    output = ColorConnection(initial, actual.bra).output_legs
    assert 5 not in {leg.label for leg in output}
    assert {2, 3, -2, 9} <= {leg.label for leg in output}


def test_three_cycle_maps_only_original_physical_history_labels():
    permutation = (0, 1, 3, 4, 2)
    history = ColorCorrelator(
        "literal-definition",
        (EmitGluon(2, 9), EmitGluon(9, -2), EmitGluon(4, -3)),
        (EmitGluon(3, 9), EmitGluon(9, -2), EmitGluon(2, -3)),
    )
    actual = _permuted(history, permutation)
    assert actual == ColorCorrelator(
        "literal-definition",
        (EmitGluon(3, 9), EmitGluon(9, -2), EmitGluon(2, -3)),
        (EmitGluon(4, 9), EmitGluon(9, -2), EmitGluon(3, -3)),
    )
    assert actual != history
    assert actual != _permuted(history, (0, 1, 4, 2, 3))
    assert _permuted(actual, (0, 1, 4, 2, 3)) == history


def test_split_retired_physical_parent_moves_and_auxiliary_roles_do_not():
    initial = tuple(
        ColorLeg(i, representation) for i, representation in enumerate((-3, 3, 8, 8, 8))
    )
    history = ColorCorrelator(
        "split-then-daughter",
        (SplitGluon(2, 9, -2), EmitGluon(9, -3), SplitGluon(-3, 10, -4)),
        (EmitGluon(2, -3), SplitGluon(2, 9, -2), SplitGluon(-3, 10, -4)),
    )
    actual = _permuted(history, (0, 1, 3, 4, 2))
    assert actual.bra == (
        SplitGluon(3, 9, -2),
        EmitGluon(9, -3),
        SplitGluon(-3, 10, -4),
    )
    assert actual.ket == (
        EmitGluon(3, -3),
        SplitGluon(3, 9, -2),
        SplitGluon(-3, 10, -4),
    )
    left, right = (
        ColorConnection(initial, steps) for steps in (actual.bra, actual.ket)
    )
    assert left.output_legs == right.output_legs
    assert 3 not in {leg.label for leg in left.output_legs}
    assert -3 not in {leg.label for leg in left.output_legs}
    assert {9, -2, 10, -4} <= {leg.label for leg in left.output_legs}


@pytest.mark.parametrize("legs", [(3,), (4,), (3, 4)])
def test_literal_source_three_cycle_uses_inverse_keys_without_touching_rows(legs):
    from pyamplicol.loops.flavor_transport import NativeIdentityFlavorProgram

    # Exercise only the pure key-routing seam. Actual prepared owners are
    # exercised below; this recorder is not a synthetic physics certificate.
    permutation = (0, 1, 3, 4, 2)
    recorder = SimpleNamespace(
        transport=SimpleNamespace(
            permutes_external_legs=True,
            mapping=SimpleNamespace(permutation=permutation),
        ),
        directed_plan=SimpleNamespace(
            bare=SimpleNamespace(spin_layout=SimpleNamespace(legs=legs))
        ),
    )
    rows = {
        member: tuple(
            (Decimal(f"{member}.{part}000000000000000000000000000001"), Decimal("-2.5"))
            for part in range(4)
        )
        for member in (4, 5)
    }
    sources = {permutation[leg - 1] + 1: rows[permutation[leg - 1] + 1] for leg in legs}
    routed = NativeIdentityFlavorProgram._spin_sources(recorder, sources)
    assert tuple(sorted(routed)) == legs
    for representative in legs:
        assert routed[representative] is sources[permutation[representative - 1] + 1]
    assert sources == {key: rows[key] for key in sources}
    for bad in ({}, {True: rows[4]}, {3: rows[4]}):
        with pytest.raises(ValueError):
            NativeIdentityFlavorProgram._spin_sources(recorder, bad)


def test_ordinary_source_class_has_no_synthetic_vector_requirement():
    from pyamplicol.loops.flavor_transport import NativeIdentityFlavorProgram

    recorder = SimpleNamespace(
        transport=SimpleNamespace(
            permutes_external_legs=True,
            mapping=SimpleNamespace(permutation=(0, 1, 3, 4, 2)),
        ),
        directed_plan=SimpleNamespace(bare=SimpleNamespace(spin_layout=None)),
    )
    assert NativeIdentityFlavorProgram._spin_sources(recorder, None) == {}
    assert NativeIdentityFlavorProgram._spin_sources(recorder, {}) == {}


@pytest.fixture
def retained_sources():
    if RETAINED_BOSE_SOURCES is None:
        pytest.skip("requires the retained independently generated Bose source pair")
    return RETAINED_BOSE_SOURCES


def _options(control):
    return dict(
        representative_directed=control["ordinary"][0],
        member_directed=control["ordinary"][1],
        representative_histories=control["histories"][0],
        member_histories=control["histories"][1],
        representative_spin=control["classes"][0],
        member_spin=control["classes"][1],
        selected_history_ids=control["selected_ids"],
    )


def _admit(control, **changes):
    return control["relation"].with_bose_correlations(
        *control["programs"], **(_options(control) | changes)
    )


def test_actual_subset_requires_same_id_and_actual_operator_definition(
    retained_sources,
):
    control = retained_sources
    admitted = _admit(control)
    assert admitted is not control["relation"]
    assert admitted.symmetry is control["relation"].symmetry
    assert admitted.bose_scalar_bindings is control["relation"].bose_scalar_bindings
    for ids in (
        (),
        (*control["selected_ids"], control["selected_ids"][0]),
        ("absent",),
    ):
        with pytest.raises(ValueError):
            _admit(control, selected_history_ids=ids)
    first = control["histories"][1][0]
    changed = replace(first, bra=(EmitGluon(0, -99),), ket=(EmitGluon(1, -99),))
    with pytest.raises(ValueError):
        _admit(control, member_histories=(changed, *control["histories"][1][1:]))
    # An existing unclosed ordered request is not certified merely because its
    # ID occurs on both sides of the original prepared catalogue.
    for history_id in control["unavailable_ids"]:
        with pytest.raises((ValueError, NotImplementedError)):
            _admit(control, selected_history_ids=(history_id,))


def test_actual_source_owners_and_class_sets_remain_authoritative(retained_sources):
    control = retained_sources
    first = control["classes"][1][0]
    for values in (
        control["classes"][0],
        (*control["classes"][1], first),
    ):
        with pytest.raises(ValueError):
            _admit(control, member_spin=values)
    bad_layout = replace(
        first.bare.spin_layout, owner=control["programs"][0].virtual.bare.prepared
    )
    bad = replace(first, bare=replace(first.bare, spin_layout=bad_layout))
    with pytest.raises(ValueError):
        _admit(control, member_spin=(bad, *control["classes"][1][1:]))
    with pytest.raises(ValueError):
        control["relation"].with_bose_correlations(
            *control["programs"][::-1], **_options(control)
        )


@pytest.mark.parametrize("field", ["phase_states", "helicities", "contracts"])
def test_matching_substitutions_cannot_replace_source_authority(
    retained_sources, field
):
    control = retained_sources
    changed_classes = []
    for classes in control["classes"]:
        first = classes[0]
        layout = first.bare.spin_layout
        # Keep both copies equal and their owner pointers intact. A paired
        # comparison alone must not prove this incomplete spectator set.
        value = layout.helicities[:-1] if field == "helicities" else ()
        altered_layout = replace(layout, **{field: value})
        bare = replace(
            first.bare,
            spin_layout=altered_layout,
            **({"helicities": value} if field == "helicities" else {}),
        )
        changed_classes.append((replace(first, bare=bare), *classes[1:]))
    with pytest.raises(ValueError):
        _admit(
            control,
            representative_spin=changed_classes[0],
            member_spin=changed_classes[1],
        )

    # Direct immutable-record construction must meet the same source boundary.
    # Valid history routes alone cannot certify substituted source classes.
    admitted = _admit(control)
    with pytest.raises(ValueError):
        replace(
            admitted.bose_correlations,
            representative_spin=changed_classes[0],
            member_spin=changed_classes[1],
        )
    with pytest.raises(ValueError):
        promoted = replace(
            control["relation"],
            representative_directed=admitted.representative_directed,
            member_directed=admitted.member_directed,
            representative_spin=changed_classes[0],
            member_spin=changed_classes[1],
            bose_correlations=admitted.bose_correlations,
        )
        promoted.bind(control["programs"][0], directed=control["ordinary"][0])


def test_old_ordinary_admission_cannot_be_promoted_by_richer_representative(
    retained_sources,
):
    control = retained_sources
    with pytest.raises((ValueError, NotImplementedError)):
        control["relation"].bind(
            control["programs"][0], directed=control["ordinary"][0]
        )
    admitted = _admit(control)
    with pytest.raises(ValueError):
        admitted.bind(control["programs"][0], directed=control["ordinary"][1])
