# SPDX-License-Identifier: 0BSD
"""The phase-space channels that integrate one leading-colour order together:
where its colour singlets attach to the colour lines, which non-cyclic order
its lines take, and which s-channel resonances it maps out."""

from __future__ import annotations

import pytest

from pyamplicol.artifacts.umami import (
    _configurations,
    _line_blocks,
    _phasespace_partners,
)

# all-outgoing colour representations
Q, QBAR, G, S = 3, -3, 8, 1


def test_line_blocks_follow_the_word():
    roles = [QBAR, Q, G, Q, QBAR, G]
    assert _line_blocks([1, 2, 0, 3, 5, 4], roles) == [[1, 2, 0], [3, 5, 4]]
    # a gluon trace has no open lines
    assert _line_blocks([0, 1, 2], [G, G, G]) is None


def test_singlets_follow_the_antiquark_in_every_order():
    # q q~ > e+ e- g, the word q g q~ crossed: slots 1 (q~ in) ... 0 (q in)
    roles = [QBAR, Q, S, S, G]
    partners = _phasespace_partners([1, 4, 0], [2, 3], roles, [], "q q~ > e+ e- g")
    assert [order for order, _ in partners] == [[1, 4, 0, 2, 3], [1, 4, 0, 3, 2]]
    assert all(configuration == () for _, configuration in partners)


def test_singlets_are_distributed_over_the_lines():
    # two lines and one singlet: after the first or the second antiquark
    roles = [Q, QBAR, Q, QBAR, S]
    partners = _phasespace_partners([0, 1, 2, 3], [4], roles, [], "4q + W")
    assert [order for order, _ in partners] == [[0, 1, 2, 3, 4], [0, 1, 4, 2, 3]]


def test_three_lines_take_both_cyclic_orders():
    roles = [Q, QBAR] * 3
    partners = _phasespace_partners(list(range(6)), [], roles, [], "6q")
    assert [order for order, _ in partners] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 4, 5, 2, 3],
    ]


def test_a_resonance_keeps_its_decay_products_together():
    roles = [QBAR, Q, S, S, G]
    z = (23, frozenset({2, 3}), 1)
    partners = _phasespace_partners([1, 4, 0], [2, 3], roles, [z], "q q~ > e+ e- g")
    # the non-resonant placements and the Z, with its legs as one unit
    assert partners == [
        ([1, 4, 0, 2, 3], ()),
        ([1, 4, 0, 3, 2], ()),
        ([1, 4, 0, 2, 3], ((23, frozenset({2, 3})),)),
    ]


def test_a_colour_singlet_decaying_to_a_complete_line():
    roles = [Q, QBAR, Q, QBAR, G]
    z = (23, frozenset({2, 3}), 1)
    # the Z decays to the line 2 3 of this order, but not to legs of two lines
    assert len(_phasespace_partners([0, 1, 2, 3], [], roles[:4], [z], "4q")) == 2
    assert len(_phasespace_partners([0, 3, 2, 1], [], roles[:4], [z], "4q")) == 1


def test_overlapping_resonances_are_alternatives():
    # e+ e- e+ ve: a Z from either positron, with the W from the other
    z1, z2 = (23, frozenset({2, 4})), (23, frozenset({3, 4}))
    w1, w2 = (24, frozenset({2, 5})), (24, frozenset({3, 5}))
    configurations = _configurations([z1, z2, w1, w2])
    assert configurations[0] == ()
    assert sorted(map(set, configurations[1:]), key=str) == sorted(
        [{z1, w2}, {z2, w1}], key=str
    )


def test_nested_resonances_stay_contiguous():
    roles = [QBAR, Q, S, S, S, S]
    h = (25, frozenset({2, 3, 4, 5}), 1)
    z1, z2 = (23, frozenset({2, 4}), 1), (23, frozenset({3, 5}), 1)
    partners = _phasespace_partners([1, 0], [2, 3, 4, 5], roles, [h, z1, z2], "4l")
    resonant = [order for order, configuration in partners if len(configuration) == 3]
    assert resonant == [[1, 0, 2, 4, 3, 5]]


@pytest.mark.parametrize(
    "resonance, message",
    [
        ((6, frozenset({2, 3, 4}), 3), "coloured s-channel resonance 6"),
        ((25, frozenset({2, 4}), 1), "coloured and colourless"),
    ],
)
def test_unsupported_resonances_raise(resonance, message):
    roles = [QBAR, Q, Q, S, S]
    with pytest.raises(ValueError, match=message):
        _phasespace_partners([1, 2, 0], [3, 4], roles, [resonance], "process")
