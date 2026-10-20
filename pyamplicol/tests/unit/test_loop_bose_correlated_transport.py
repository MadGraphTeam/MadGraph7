# SPDX-License-Identifier: 0BSD
"""Independent cold Bose/source controls over retained actual preparations.

The injected fixture never generates graphs. Pure three-cycle controls use
explicit expected labels; actual owner/phase controls require the separately
retained pair and make no numerical or independent-physics claim.
"""

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

pytestmark = pytest.mark.loop_fast



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
















