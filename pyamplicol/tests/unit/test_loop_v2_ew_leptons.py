# SPDX-License-Identifier: 0BSD
"""Colour-singlet fermion lines through the loop and at the root (WP3.2 EWI-5).

The blocks carry the tree engine's Koszul signs of colour-singlet fermions
(``generation/fermion_ordering.py``): a block row is its current's
coefficient of the product of its singlet sources in source order. The loop
leaf weights and the root closure complete them with one rule,
``conventions.colour_singlet_sign``: the Koszul sign of the blocks' singlet
supports in the order they are read, times the merge sign of the open ends
each line joins outside the blocks, listed (antiparticle, particle) as the
tree engine's two-fermion merge takes them. Checks:

* the rule's defining identity, against an independent count of inversions:
  the product of the blocks' own Koszul signs and the rule is the sign of the
  whole diagram, every line listed (antiparticle, particle), in every block
  order; with every block one source, that sign is the tree engine's
  ``FermionOrderingSign.tree_sign``;
* processes with two colour-singlet lines, refused before EWI-5 (Born closure
  / tree runtime 0.53 and 0.34): the Born closed at leg 1 is the tree
  runtime's amplitude up to a real sign, with identical leptons too, and Born
  and virtual (QCD corrections) are invariant under a relabelling of the
  final state that changes the interleaving of the lines' supports (without
  the loop's Koszul sign the virtual moves by O(1)); the leaf weights' fast
  path equals the full rule there (``check_leaf_weights``, WP3.1 D1).
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.conventions import (
    ANTIPARTICLE,
    PARTICLE,
    FermionSignError,
    colour_singlet_sign,
)

pytestmark = pytest.mark.loop_physics

TOLERANCE = 1e-12


# ---------------------------------------------------------------- the rule


def _parity(sequence) -> int:
    """Sign of the permutation sorting ``sequence`` (an independent count)."""

    inversions = sum(a > b for i, a in enumerate(sequence) for b in sequence[i + 1 :])
    return -1 if inversions % 2 else 1


def _diagram(rng: random.Random):
    """A random diagram's singlet lines and blocks: each line is closed
    inside one block or joins the open ends of two blocks; a block holds
    closed lines and at most one open end. Returns (lines as (antiparticle,
    particle) labels, blocks as (closed lines, open end or None))."""

    count = rng.randint(1, 5)
    labels = list(range(1, 2 * count + 1))
    rng.shuffle(labels)
    lines = [tuple(labels[2 * i : 2 * i + 2]) for i in range(count)]
    blocks: list[tuple[list, int | None]] = []
    for line in lines:
        if rng.random() < 0.5:
            blocks.append(([line], None))
        else:
            blocks.extend(([], end) for end in line)
    rng.shuffle(blocks)
    merged: list[tuple[list, int | None]] = []
    for closed, end in blocks:
        if merged and rng.random() < 0.4 and (end is None or merged[-1][1] is None):
            previous, other = merged[-1]
            merged[-1] = (previous + closed, end if other is None else other)
        else:
            merged.append((list(closed), end))
    return lines, merged


def _block_sign(closed, end) -> int:
    """A block row's Koszul sign: its sources in order against its closed
    lines listed (antiparticle, particle), the open end last."""

    listing = [label for line in closed for label in line]
    if end is not None:
        listing.append(end)
    position = {label: rank for rank, label in enumerate(sorted(listing))}
    return _parity([position[label] for label in listing])


def _rule(lines, blocks) -> int:
    supports = [
        tuple(label for line in closed for label in line)
        + (() if end is None else (end,))
        for closed, end in blocks
    ]
    index = {end: i for i, (_closed, end) in enumerate(blocks) if end is not None}
    joined = [
        (index[anti], ANTIPARTICLE, index[part], PARTICLE)
        for anti, part in lines
        if anti in index
    ]
    return colour_singlet_sign(supports, joined)


@pytest.mark.loop_fast
def test_singlet_sign_completes_the_blocks_to_the_diagram_sign():
    """Blocks' Koszul signs times the rule = the diagram's sign with every
    line listed (antiparticle, particle), in every order of the blocks."""

    rng = random.Random(20261026)
    for _case in range(400):
        lines, blocks = _diagram(rng)
        diagram = _parity([label for line in lines for label in line])
        for order in itertools.islice(itertools.permutations(blocks), 24):
            product = 1
            for closed, end in order:
                product *= _block_sign(closed, end)
            assert product * _rule(lines, list(order)) == diagram, (lines, order)


class _Model:
    """The particle facts ``FermionOrderingSign`` reads: colour-singlet
    Dirac fermions, positive codes particles."""

    def is_fermion(self, particle_id: int) -> bool:
        return True

    def color_rep(self, particle_id: int) -> int:
        return 1

    def source_orientation(self, particle_id: int) -> str:
        return PARTICLE if particle_id > 0 else ANTIPARTICLE


@pytest.mark.loop_fast
def test_singlet_sign_of_single_sources_is_the_tree_engines_tree_sign():
    """With every block one source (a tree's external legs), the rule is
    the tree engine's sign of a tree with these lines."""

    from pyamplicol.generation.fermion_ordering import FermionOrderingSign

    rng = random.Random(7)
    for _case in range(200):
        count = rng.randint(1, 4)
        labels = list(range(1, 2 * count + 1))
        rng.shuffle(labels)
        lines = [tuple(labels[2 * i : 2 * i + 2]) for i in range(count)]
        codes = {anti: -11 for anti, _part in lines}
        codes.update({part: 11 for _anti, part in lines})
        legs = {
            label: SimpleNamespace(
                label=label,
                statistics="fermion",
                color_role="singlet",
                outgoing_pdg=codes[label],
            )
            for label in sorted(codes)
        }
        process = SimpleNamespace(legs=tuple(legs.values()))
        engine = FermionOrderingSign(process, _Model())
        pairs = [(legs[a], legs[p]) for a, p in lines]
        rng.shuffle(pairs)
        expected = engine.tree_sign([tuple(rng.sample(pair, 2)) for pair in pairs])
        order = sorted(codes, key=lambda _label: rng.random())
        index = {label: i for i, label in enumerate(order)}
        assert (
            colour_singlet_sign(
                [(label,) for label in order],
                [(index[a], ANTIPARTICLE, index[p], PARTICLE) for a, p in lines],
            )
            == expected
        )


@pytest.mark.loop_fast
def test_singlet_sign_refuses_lines_it_cannot_pair():
    """Two ends of one orientation (Majorana-like), or a line whose two ends
    are one block's, are refused."""

    with pytest.raises(FermionSignError):
        colour_singlet_sign([(1,), (2,)], [(0, PARTICLE, 1, PARTICLE)])
    with pytest.raises(FermionSignError):
        colour_singlet_sign([(1, 2)], [(0, ANTIPARTICLE, 0, PARTICLE)])
