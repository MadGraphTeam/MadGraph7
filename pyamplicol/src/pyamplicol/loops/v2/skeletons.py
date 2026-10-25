# SPDX-License-Identifier: 0BSD
"""Skeleton contents of a family (design §3.3-§3.4, §4.2 (2)).

A skeleton is a ring of position *contents*. A content is one colour state of
one position between two loop species; it sums every routing term (vertex
template and attached block currents) that produces that colour state, e.g. the
cubic and quartic parts of a planar gluon position (design §3.4). Terms of
different coupling orders are distinct contents, so every content and skeleton
carries one exact order vector (design §3.6) and its closure lands in one
squared-order sector. Skeletons are built per family from the family's routings
and are never listed globally.

Colour states come from a ``ColourModel`` (WP1.2 block colour states and the
WP1.4 seed classes). ``DressedColour`` keeps one content per routing term;
``PlanarWords`` is the interim colour-ordered model used for counts and kernel
shapes until those packages are integrated: every block contributes every
ordering of its letters and multi-current positions concatenate their words.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations, product
from typing import Any, Protocol, cast

from .families import (
    Closure,
    Family,
    FamilyKey,
    FamilySet,
    LoopVertexCatalogue,
    Orders,
    Position,
    add_orders,
)


class ColourModel(Protocol):
    """Colour states of position contents."""

    def states(self, position: Position) -> tuple[Hashable, ...]:
        """Colour states of the content(s) a routing term contributes to."""
        ...

    def reflected(self, state: Hashable) -> Hashable:
        """The same state read with the loop direction reversed."""
        ...


class DressedColour:
    """Colour-dressed contents: every routing term is its own content."""

    def states(self, position: Position) -> tuple[Hashable, ...]:
        return (position.blocks,)

    def reflected(self, state: Hashable) -> Hashable:
        return state


class PlanarWords:
    """Colour-ordered words over external-leg letters (interim colour model).

    ``letters`` are leg masks that occupy one place in a colour word (by
    default every leg is a letter). A block contributes every ordering of its
    letters; a position with several currents contributes every concatenation
    of their words. Reflection reverses a word.
    """

    def __init__(self, legs: int, letters: Sequence[int] | None = None):
        letters = (
            tuple(letters)
            if letters is not None
            else tuple(1 << i for i in range(legs))
        )
        covered = 0
        for letter in letters:
            if not letter or covered & letter:
                raise ValueError("letters must be disjoint non-empty leg masks")
            covered |= letter
        if covered != (1 << legs) - 1:
            raise ValueError("letters must cover every external leg")
        self.letters = letters
        self._words: dict[int, tuple[tuple[int, ...], ...]] = {}

    def _block_words(self, mask: int) -> tuple[tuple[int, ...], ...]:
        words = self._words.get(mask)
        if words is None:
            inside = [letter for letter in self.letters if letter & mask]
            if any(letter & ~mask for letter in inside):
                raise ValueError(f"block {mask:#b} splits a letter")
            words = self._words[mask] = tuple(permutations(inside))
        return words

    def states(self, position: Position) -> tuple[Hashable, ...]:
        out = set()
        for blocks in permutations(position.blocks):
            for parts in product(*(self._block_words(block.mask) for block in blocks)):
                out.add(tuple(letter for part in parts for letter in part))
        return tuple(sorted(out))

    def reflected(self, state: Hashable) -> Hashable:
        return tuple(reversed(cast(tuple[int, ...], state)))


#: A content's label: (colour state, loop in, loop out, mask, orders).
_Label = tuple[Hashable, str, str, int, Orders]


@dataclass(frozen=True, slots=True, order=True)
class Content:
    """One position of a skeleton: a colour state between two loop species.

    Every summed term has the coupling orders ``orders`` (vertex and blocks).
    """

    colour: Hashable
    loop_in: str
    loop_out: str
    mask: int
    orders: Orders
    terms: tuple[Position, ...]
    # The hash of the fields, computed once: contents key the generator's
    # maps and hashing their terms dominated the 7-leg seed tables.
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(
                self,
                "_hash",
                hash(
                    (
                        self.colour,
                        self.loop_in,
                        self.loop_out,
                        self.mask,
                        self.orders,
                        self.terms,
                    )
                ),
            )
        return self._hash  # type: ignore[return-value]

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return Content, (
            self.colour,
            self.loop_in,
            self.loop_out,
            self.mask,
            self.orders,
            self.terms,
        )

    @property
    def q_degree(self) -> int:
        """Max-plus q-degree of the summed terms."""
        return max(term.vertex.q_degree for term in self.terms)

    @property
    def label(self) -> tuple[Hashable, str, str, int, Orders]:
        return self.colour, self.loop_in, self.loop_out, self.mask, self.orders


@dataclass(frozen=True, slots=True)
class Skeleton:
    """A ring of contents read in the F frame (cut before ``contents[0]``)."""

    family: FamilyKey
    contents: tuple[Content, ...]
    weight: Fraction  # automorphism weight 1/|Stab|
    multiplicity: int = 1  # merged closed-loop flavours (design §4.3)
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)
    # ``orders``, computed once: the seed stage and the LC/NLC prune read it
    # per skeleton (millions at seven legs).
    _orders: Orders | None = field(default=None, init=False, repr=False, compare=False)

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(
                self,
                "_hash",
                hash((self.family, self.contents, self.weight, self.multiplicity)),
            )
        return self._hash  # type: ignore[return-value]

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return Skeleton, (self.family, self.contents, self.weight, self.multiplicity)

    @property
    def cut(self) -> str:
        return self.contents[-1].loop_out

    @property
    def rank(self) -> int:
        return sum(content.q_degree for content in self.contents)

    @property
    def species(self) -> tuple[str, ...]:
        """Species of the propagator following each position."""
        return tuple(content.loop_out for content in self.contents)

    @property
    def orders(self) -> Orders:
        """Coupling orders of the loop amplitude (its sector offset, §3.6)."""
        if self._orders is None:
            object.__setattr__(
                self, "_orders", add_orders(*(c.orders for c in self.contents))
            )
        return self._orders  # type: ignore[return-value]


def reflect_content(
    content: Content, colour: ColourModel, catalogue: LoopVertexCatalogue
) -> Content:
    species = catalogue.species
    return Content(
        colour.reflected(content.colour),
        species(content.loop_out).antiname,
        species(content.loop_in).antiname,
        content.mask,
        content.orders,
        tuple(
            sorted(
                Position(catalogue.reflected(term.vertex), term.blocks)
                for term in content.terms
            )
        ),
    )


def reflect_ring(
    contents: Sequence[Content], colour: ColourModel, catalogue: LoopVertexCatalogue
) -> tuple[Content, ...]:
    """Read a ring backwards from its first position."""
    order = (contents[0], *reversed(contents[1:]))
    return tuple(reflect_content(content, colour, catalogue) for content in order)


def family_skeletons(
    families: FamilySet,
    family: Family,
    colour: ColourModel,
    catalogue: LoopVertexCatalogue,
) -> tuple[Skeleton, ...]:
    """The skeletons of one family, in the F frame of its routings."""
    rings: dict[tuple[_Label, ...], list[set[Position]]] = {}
    combos: dict[tuple[_Label, ...], set[tuple[Position, ...]]] = {}
    symmetric: dict[tuple[_Label, ...], bool] = {}
    multiplicity: dict[tuple[_Label, ...], int] = {}
    for closure in family.closures:
        for labels, positions in _labelled_rings(families, closure, colour):
            slots = rings.setdefault(labels, [set() for _ in positions])
            for slot, position in zip(slots, positions, strict=True):
                slot.add(position)
            combos.setdefault(labels, set()).add(positions)
            symmetric[labels] = symmetric.get(labels, False) or closure.symmetric
            if (
                multiplicity.setdefault(labels, closure.multiplicity)
                != closure.multiplicity
            ):
                raise ValueError(f"inconsistent flavour multiplicity in {family.key}")
    out = {}
    for labels, slots in rings.items():
        count = 1
        for slot in slots:
            count *= len(slot)
        if count != len(combos[labels]):
            raise ValueError(
                f"routings of family {family.key} do not factorise per position"
            )
        contents = tuple(
            Content(*label, tuple(sorted(slot)))
            for label, slot in zip(labels, slots, strict=True)
        )
        weight = Fraction(1)
        if symmetric[labels]:
            # N <= 2 and the routing equals its reflection: the reversed word
            # reading is the same skeleton (identified, weight 1/2 if equal).
            mirror = reflect_ring(contents, colour, catalogue)
            if mirror == contents:
                weight = Fraction(1, 2)
            contents = min(contents, mirror)
        if contents in out:
            continue
        out[contents] = Skeleton(family.key, contents, weight, multiplicity[labels])
    return tuple(out[key] for key in sorted(out))


def _labelled_rings(
    families: FamilySet, closure: Closure, colour: ColourModel
) -> Iterator[tuple[tuple[_Label, ...], tuple[Position, ...]]]:
    _cut, positions = families.routing(closure)
    options = [colour.states(position) for position in positions]
    for states in product(*options):
        labels = tuple(
            (state, p.vertex.loop_in, p.vertex.loop_out, p.mask, p.orders)
            for state, p in zip(states, positions, strict=True)
        )
        yield labels, positions
