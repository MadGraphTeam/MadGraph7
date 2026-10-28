# SPDX-License-Identifier: 0BSD
"""Denominator families from the loop-prefix recursion (design §3.3, §4.2 (1)).

A loop prefix is an ordered sequence of positions starting at the cut. Each
position is one loop vertex with the block currents it attaches and the species
leaving it. Prefixes are extended one position at a time with leg sets disjoint
from the legs already used; a prefix that covers every leg and returns to the cut
species closes into a *routing*. Equal canonical keys of closed routings define
a denominator family. No Feynman diagram is enumerated: positions consume block
currents (sums of all trees on their leg sets), and the recursion only ever
visits prefixes that can still close (memoised closability).

Conventions (design §3.3):

* the cut sits before position 1, which contains the lowest external leg; for
  N >= 3 the direction has min(B_2) < min(B_N) (Recola's canonical cut and
  orientation); for N <= 2 the reading that sorts first is kept and
  reflection-invariant routings are flagged (their automorphism weight is 1/2);
* a family key is the lexicographic minimum of the cyclic sequence of
  (position leg mask, mass id of the following propagator) over all 2N
  rotations and reflections (Recola's tensor-integral identity);
* the selection rule drops external wavefunction corrections, VEV tadpoles and
  rings with exactly one coloured attachment; every rejected family is logged;
* massless families whose every invariant is structurally zero after on-shell
  bindings are dropped with a logged certificate.

The block and loop-vertex tables are consumed through the ``BlockTable`` and
``LoopVertexCatalogue`` protocols (interfaces of WP1.2 and WP1.1). Until those
packages are integrated, ``ModelBlockTable`` and ``ModelLoopCatalogue`` provide
field-level implementations from a native model description.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations, product
from typing import Any, Protocol

from pyamplicol.loops.orders import LoopEligibility

from ..._internal.physics.symbols import symbols

Orders = tuple[tuple[str, int], ...]

ZERO = "0"


def add_orders(*terms: Orders) -> Orders:
    totals: dict[str, int] = {}
    for orders in terms:
        for name, power in orders:
            totals[name] = totals.get(name, 0) + power
    return tuple(sorted((name, power) for name, power in totals.items() if power))


def orders_within(orders: Orders, cap: Orders) -> bool:
    bound = dict(cap)
    return all(power <= bound.get(name, 0) for name, power in orders)


def lowest_leg(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def onshell_leg(arc: int, full: int) -> int | None:
    """The external leg whose mass shell fixes the invariant p(arc)^2: the
    leg of a single-leg arc or of a single-leg complement (momentum
    conservation), else None. Such an invariant is the leg's squared mass by
    definition of the external state, never the p^2 of the given (possibly
    rounded) momenta (design §11.1, on-shell external invariants)."""

    for legs in (arc, full ^ arc):
        if legs and legs & (legs - 1) == 0:
            return lowest_leg(legs)
    return None


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True, order=True)
class MassId:
    """Exact model mass and width expressions of a propagator ("0" if zero)."""

    mass: str
    width: str = ZERO

    @property
    def massless(self) -> bool:
        return self.mass == ZERO and self.width == ZERO


@dataclass(frozen=True, slots=True)
class LoopSpecies:
    name: str
    antiname: str
    spin: int  # UFO 2s+1; -1 marks a ghost
    colour: int
    mass: MassId
    perturbative_orders: frozenset[str]


@dataclass(frozen=True, slots=True, order=True)
class LoopVertex:
    """A model vertex with two legs on the loop (a WP1.1 catalogue template).

    ``loop_in`` enters the vertex along the loop, ``loop_out`` leaves it and
    ``attached`` are the incoming block fields (sorted). ``q_degree`` is the
    vertex q-degree plus the q-degree of the outgoing propagator numerator.
    ``signature`` identifies the vertex's couplings, Lorentz and colour
    structures with the two loop legs abstracted: templates of different
    flavours with equal signatures are structurally identical.
    """

    rule: str
    loop_in: str
    attached: tuple[str, ...]
    loop_out: str
    orders: Orders
    q_degree: int
    signature: Hashable = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class ExternalLeg:
    index: int
    field: str  # incoming particle (outgoing legs are crossed)
    mass: MassId
    colour: int


@dataclass(frozen=True, slots=True, order=True)
class BlockCurrent:
    """A tree current over a leg set entering a loop vertex as ``field``."""

    mask: int
    field: str
    orders: Orders


@dataclass(frozen=True, slots=True, order=True)
class Position:
    """One loop vertex of a routing with the block currents it attaches."""

    vertex: LoopVertex
    blocks: tuple[BlockCurrent, ...]  # sorted by mask
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(self, "_hash", hash((self.vertex, self.blocks)))
        return self._hash  # type: ignore[return-value]

    def __reduce__(self):
        return Position, (self.vertex, self.blocks)

    @property
    def mask(self) -> int:
        mask = 0
        for block in self.blocks:
            mask |= block.mask
        return mask

    @property
    def orders(self) -> Orders:
        return add_orders(self.vertex.orders, *(block.orders for block in self.blocks))


class LoopVertexCatalogue(Protocol):
    """Loop species and vertex templates (WP1.1 interface)."""

    def species(self, name: str) -> LoopSpecies: ...

    def loop_species(self) -> tuple[str, ...]: ...

    def vertices(self, loop_in: str) -> tuple[LoopVertex, ...]: ...

    def reflected(self, vertex: LoopVertex) -> LoopVertex: ...


class BlockTable(Protocol):
    """Field-level block currents over leg sets (WP1.2 interface)."""

    @property
    def legs(self) -> tuple[ExternalLeg, ...]: ...

    def currents(self, mask: int) -> tuple[BlockCurrent, ...]: ...


@dataclass(frozen=True, slots=True)
class LoopOrderSelection:
    """Selected corrections and the admitted loop-amplitude coupling orders.

    Routings are admitted by ``LoopEligibility`` (MadLoop's joint selection,
    ``loops/orders.py``); its species condition also prunes the recursion.
    """

    corrections: frozenset[str]
    loop_orders: frozenset[Orders]
    cap: Orders = field(init=False)

    def __post_init__(self) -> None:
        corrections = frozenset(self.corrections)
        targets = frozenset(add_orders(order) for order in self.loop_orders)
        if not corrections or not targets:
            raise ValueError("a loop selection needs corrections and loop orders")
        cap: dict[str, int] = {}
        for order in targets:
            for name, power in order:
                cap[name] = max(cap.get(name, 0), power)
        object.__setattr__(self, "corrections", corrections)
        object.__setattr__(self, "loop_orders", targets)
        object.__setattr__(self, "cap", tuple(sorted(cap.items())))


# --------------------------------------------------------------------------- keys

FamilyKey = tuple[tuple[int, MassId], ...]


def reading(items: Sequence[Any], gaps: Sequence[Any], start: int, reflected: bool):
    """Positions and following gaps of a ring read from ``start``.

    Reading backwards, the gap after position ``i`` is the one before it.
    """
    count = len(items)
    if reflected:
        order = [(start - step) % count for step in range(count)]
        return (
            tuple(items[i] for i in order),
            tuple(gaps[(i - 1) % count] for i in order),
        )
    order = [(start + step) % count for step in range(count)]
    return tuple(items[i] for i in order), tuple(gaps[i] for i in order)


def ring_readings(count: int) -> Iterator[tuple[int, bool]]:
    for reflected in (False, True):
        for start in range(count):
            yield start, reflected


def canonical_family_key(masks: Sequence[int], masses: Sequence[MassId]) -> FamilyKey:
    return min(
        tuple(zip(*reading(masks, masses, start, reflected), strict=True))
        for start, reflected in ring_readings(len(masks))
    )


@dataclass(frozen=True, slots=True)
class FamilyFrame:
    """A reading of a family's canonical key: cut before ``start``.

    The loop momentum flows through the last propagator of the reading; the
    shift of propagator i is the leg mask of positions 1..i (design §3.3).
    """

    start: int
    reflected: bool

    def apply(self, key: FamilyKey) -> FamilyKey:
        masks, masses = zip(*key, strict=True)
        return tuple(
            zip(*reading(masks, masses, self.start, self.reflected), strict=True)
        )


def shifts(key: FamilyKey) -> tuple[int, ...]:
    """Leg masks V_i = p(B_1) + ... + p(B_i); V_N is the full leg mask."""
    total, out = 0, []
    for mask, _mass in key:
        total |= mask
        out.append(total)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Relabelling:
    """Exact affine re-test frame of a family (design §10.2, addendum C.2).

    ``source`` is the frame, a reading of the canonical key, in which the
    family polynomial is produced (the B-trie frame). ``frame`` reads the
    source reading from another cut and, if reflected, with the loop direction
    reversed. Propagator i of the new frame is propagator ``propagators[i]`` of
    the source frame, and q = sign * q' - V(shift_mask), with V the shifts of
    the source reading, maps the new loop momentum q' to the source one.
    """

    source: FamilyFrame
    frame: FamilyFrame
    propagators: tuple[int, ...]
    shift_mask: int
    sign: int

    def apply(self, key: FamilyKey) -> FamilyKey:
        """The re-test reading of the canonical key."""
        return self.frame.apply(self.source.apply(key))


KEY_FRAME = FamilyFrame(0, False)


def relabelling(
    key: FamilyKey, frame: FamilyFrame, source: FamilyFrame = KEY_FRAME
) -> Relabelling:
    """The affine map from the ``source`` reading of ``key`` to ``frame`` of it."""
    count = len(key)
    if frame.reflected:
        # Reading backwards from position ``start``, the propagator following
        # reading position i is the one before key position start - i.
        propagators = tuple((frame.start - step - 1) % count for step in range(count))
    else:
        propagators = tuple((frame.start + step) % count for step in range(count))
    # The new loop momentum flows through the old propagator propagators[-1]:
    # q' = sign * (q + V_anchor), with V_{N-1} = 0 by momentum conservation.
    anchor = propagators[-1]
    shift_mask = shifts(source.apply(key))[anchor] if anchor != count - 1 else 0
    return Relabelling(
        source, frame, propagators, shift_mask, -1 if frame.reflected else 1
    )


def retest_relabelling(key: FamilyKey, source: FamilyFrame) -> Relabelling:
    """Reversed loop direction with the cut moved by N // 2 (k_f != 0).

    Taken relative to the production frame ``source``, so the map always has
    sign -1 and never repeats the primary reduction, whatever frame the cost
    policy picked.
    """
    return relabelling(key, FamilyFrame(len(key) // 2, True), source)


@dataclass(frozen=True, slots=True)
class MasterKey:
    """A pinched sub-ring with at most four propagators (canonical, generation)."""

    key: FamilyKey
    scaleless: bool


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class Closure:
    """A closed routing: a leaf of the prefix graph in the F frame."""

    state: int
    symmetric: bool  # reading equals its reflection (N <= 2)
    multiplicity: int = 1  # merged closed-loop flavours (design §4.3)


@dataclass(frozen=True, slots=True)
class Family:
    key: FamilyKey
    rank: int
    closures: tuple[Closure, ...]

    @property
    def size(self) -> int:
        return len(self.key)

    @property
    def masses(self) -> tuple[MassId, ...]:
        return tuple(mass for _mask, mass in self.key)

    def invariant_masks(self) -> tuple[int, ...]:
        """Leg masks of the proper contiguous arcs (S-matrix invariants).

        An arc and its complement carry the same invariant; the representative
        without the lowest external leg is returned.
        """
        masks = [mask for mask, _mass in self.key]
        full = shifts(self.key)[-1]
        out = set()
        for first in range(len(masks)):
            arc = 0
            for step in range(len(masks) - 1):
                arc |= masks[(first + step) % len(masks)]
                out.add(arc if not arc & 1 else full ^ arc)
        return tuple(sorted(out))

    def master_keys(self, legs: Sequence[ExternalLeg]) -> tuple[MasterKey, ...]:
        """Canonical pinched sub-rings with 1..4 propagators of this family."""
        masks = [mask for mask, _mass in self.key]
        masses = self.masses
        count = len(masks)
        found = {}
        for lines in range(1, min(4, count) + 1):
            for kept in combinations(range(count), lines):
                merged, merged_masses = [], []
                for index, gap in enumerate(kept):
                    following = kept[(index + 1) % lines]
                    mask, step = 0, (gap + 1) % count
                    while True:
                        mask |= masks[step]
                        if step == following:
                            break
                        step = (step + 1) % count
                    merged.append(mask)
                    merged_masses.append(masses[following])
                key = canonical_family_key(merged, merged_masses)
                found[key] = MasterKey(key, certified_scaleless(key, legs))
        return tuple(found[key] for key in sorted(found))

    def onshell_bindings(self, legs: Sequence[ExternalLeg]) -> dict[int, MassId]:
        """Invariants fixed by external masses, by the one on-shell rule
        every carrier evaluates (``onshell_leg``): a single-leg arc or a
        single-leg complement is that leg's mass shell."""
        full = shifts(self.key)[-1]
        return {
            mask: legs[leg].mass
            for mask in self.invariant_masks()
            if (leg := onshell_leg(mask, full)) is not None
        }


@dataclass(frozen=True, slots=True)
class LoggedFamily:
    key: FamilyKey
    reason: str


class PrefixGraph:
    """Loop prefixes of every admitted routing, shared across families (F frame).

    State 0.. are interned by (parent, position). Roots have no position and
    record the cut species; every other state records the species leaving its
    last position.
    """

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.position: list[Position | None] = []
        self.species: list[str] = []
        self.depth: list[int] = []
        self._index: dict[tuple[int, Position | str], int] = {}

    def __len__(self) -> int:
        return len(self.parent)

    def _intern(self, parent: int, label: Position | str, species: str) -> int:
        key = (parent, label)
        state = self._index.get(key)
        if state is None:
            state = len(self.parent)
            self._index[key] = state
            self.parent.append(parent)
            self.position.append(label if isinstance(label, Position) else None)
            self.species.append(species)
            self.depth.append(0 if parent < 0 else self.depth[parent] + 1)
        return state

    def root(self, cut_species: str) -> int:
        return self._intern(-1, cut_species, cut_species)

    def extend(self, state: int, position: Position) -> int:
        return self._intern(state, position, position.vertex.loop_out)

    def path(self, state: int) -> tuple[str, tuple[Position, ...]]:
        """Cut species and positions of the prefix ending at ``state``."""
        positions = []
        while self.parent[state] >= 0:
            positions.append(self.position[state])
            state = self.parent[state]
        return self.species[state], tuple(reversed(positions))


@dataclass(frozen=True, slots=True)
class FamilySet:
    """Admitted families, their routings and the logs of dropped families.

    ``merged`` maps every member of a merged closed-loop flavour class to its
    representative (design §4.3); routings use representatives only and a
    closure's multiplicity is the class size.

    ``scaleless_families`` keeps the certified-scaleless families (logged in
    ``scaleless``) with their routings: their integrals vanish, but their UV
    and IR poles do not separately (B0(0;0,0) = 1/eps_UV - 1/eps_IR), so the
    ``uv`` and ``ir`` pole selections dress them (design §12.4). Their
    closures are interned after every admitted family's.
    """

    legs: tuple[ExternalLeg, ...]
    families: Mapping[FamilyKey, Family]
    prefixes: PrefixGraph
    scaleless: tuple[LoggedFamily, ...]
    rejected: tuple[LoggedFamily, ...]
    merged: Mapping[str, str]
    scaleless_families: Mapping[FamilyKey, Family] = field(default_factory=dict)

    def routing(self, closure: Closure) -> tuple[str, tuple[Position, ...]]:
        return self.prefixes.path(closure.state)

    def block_requests(self) -> frozenset[BlockCurrent]:
        """Block currents used by admitted routings (loop-aware reachability)."""
        found = set()
        for family in self.families.values():
            for closure in family.closures:
                for position in self.routing(closure)[1]:
                    found.update(position.blocks)
        return frozenset(found)


# --------------------------------------------------------------------------- rules


def certified_scaleless(key: FamilyKey, legs: Sequence[ExternalLeg]) -> bool:
    """Structural zero: massless lines and every invariant fixed to zero.

    Every proper contiguous arc of a ring must be a single external leg bound
    on shell to a structurally zero mass (its complement is equivalent by
    momentum conservation). Never numerical (design §3.3).
    """
    if any(not mass.massless for _mask, mass in key):
        return False
    family = Family(key, 0, ())
    bound = family.onshell_bindings(legs)
    return len(bound) == len(family.invariant_masks()) and all(
        mass.massless for mass in bound.values()
    )


def selection_rejection(
    positions: Sequence[Position], catalogue: LoopVertexCatalogue
) -> str | None:
    """Structural form of ``GraphSelectionConvention`` (``loops/graphs.py``)."""
    blocks = [block for position in positions for block in position.blocks]
    if (
        len(positions) <= 2
        and len(blocks) == 2
        and any(block.mask & (block.mask - 1) == 0 for block in blocks)
    ):
        return "external-wavefunction-correction"
    if len(positions) == 1 and len(blocks) <= 1:
        return "vev-tadpole"
    coloured = sum(abs(catalogue.species(block.field).colour) != 1 for block in blocks)
    if coloured == 1:
        return "single-coloured-attachment"
    return None


# --------------------------------------------------------------------------- recursion


class _Recursion:
    def __init__(self, blocks: BlockTable, catalogue: LoopVertexCatalogue, orders):
        self.blocks = blocks
        self.catalogue = catalogue
        self.orders = orders
        self.legs = tuple(blocks.legs)
        self.full = (1 << len(self.legs)) - 1
        # The species condition of ``LoopEligibility`` prunes the recursion.
        eligible = frozenset(
            name
            for name in catalogue.loop_species()
            if catalogue.species(name).perturbative_orders & orders.corrections
        )
        self.merged = _closed_loop_classes(eligible, catalogue, blocks)
        sizes = defaultdict(int)
        for representative in self.merged.values():
            sizes[representative] += 1
        self.multiplicity = dict(sizes)
        self.eligible = frozenset(
            name for name in eligible if self.merged.get(name, name) == name
        )
        self.attachments = cache(self._attachments)
        self.positions = cache(self._positions)
        self.closable = cache(self._closable)

    def _attachments(
        self, mask: int, fields: tuple[str, ...]
    ) -> tuple[tuple[BlockCurrent, ...], ...]:
        if len(fields) == 1:
            return tuple(
                (block,)
                for block in self.blocks.currents(mask)
                if block.field == fields[0]
            )
        out = []
        low = mask & -mask
        rest = mask ^ low
        sub = rest
        while True:
            part = sub | low
            if part != mask:
                for name in dict.fromkeys(fields):
                    others = list(fields)
                    others.remove(name)
                    tails = self.attachments(mask ^ part, tuple(others))
                    if not tails:
                        continue
                    for block in self.blocks.currents(part):
                        if block.field == name:
                            out.extend((block, *tail) for tail in tails)
            if not sub:
                break
            sub = (sub - 1) & rest
        return tuple(tuple(sorted(item)) for item in out)

    def _positions(self, mask: int, species: str) -> tuple[Position, ...]:
        size = bin(mask).count("1")
        out = []
        for vertex in self.catalogue.vertices(species):
            if vertex.loop_out not in self.eligible or len(vertex.attached) > size:
                continue
            for blocks in self.attachments(mask, vertex.attached):
                position = Position(vertex, blocks)
                if orders_within(position.orders, self.orders.cap):
                    out.append(position)
        return tuple(out)

    def _closable(self, remaining: int, species: str, cut: str, orders: Orders) -> bool:
        sub = remaining
        while sub:
            for position in self.positions(sub, species):
                total = add_orders(orders, position.orders)
                if not orders_within(total, self.orders.cap):
                    continue
                if sub == remaining:
                    if (
                        position.vertex.loop_out == cut
                        and total in self.orders.loop_orders
                    ):
                        return True
                elif self.closable(
                    remaining ^ sub, position.vertex.loop_out, cut, total
                ):
                    return True
            sub = (sub - 1) & remaining
        return False

    def run(self) -> FamilySet:
        graph = PrefixGraph()
        admitted: dict[
            FamilyKey, list[tuple[tuple[str, tuple[Position, ...]], bool]]
        ] = defaultdict(list)
        rejected: dict[FamilyKey, str] = {}
        first = self.full & -self.full
        for cut in sorted(self.eligible):
            stack = [((), 0, cut, ())]
            while stack:
                positions, used, species, orders = stack.pop()
                remaining = self.full ^ used
                sub = remaining
                while sub:
                    if positions or sub & first:
                        for position in self.positions(sub, species):
                            total = add_orders(orders, position.orders)
                            if not orders_within(total, self.orders.cap):
                                continue
                            out = position.vertex.loop_out
                            extended = (*positions, position)
                            if sub == remaining:
                                if out == cut and total in self.orders.loop_orders:
                                    self._close(cut, extended, admitted, rejected)
                            elif self.closable(remaining ^ sub, out, cut, total):
                                stack.append((extended, used | sub, out, total))
                    sub = (sub - 1) & remaining
        families = {}
        scaleless = []
        for key in sorted(admitted):
            rejected.pop(key, None)
            if certified_scaleless(key, self.legs):
                scaleless.append(LoggedFamily(key, "certified-scaleless"))
                continue
            families[key] = self._family(graph, key, admitted[key])
        # The scaleless families after every admitted one: the admitted
        # families' prefix states do not depend on them.
        scaleless_families = {
            entry.key: self._family(graph, entry.key, admitted[entry.key])
            for entry in scaleless
        }
        return FamilySet(
            self.legs,
            families,
            graph,
            tuple(scaleless),
            tuple(
                LoggedFamily(key, reason) for key, reason in sorted(rejected.items())
            ),
            self.merged,
            scaleless_families,
        )

    def _family(self, graph: PrefixGraph, key: FamilyKey, routings) -> Family:
        closures = []
        rank = 0
        for (cut, positions), symmetric in routings:
            state = graph.root(cut)
            for position in positions:
                state = graph.extend(state, position)
            closures.append(Closure(state, symmetric, self.multiplicity.get(cut, 1)))
            rank = max(rank, sum(p.vertex.q_degree for p in positions))
        if rank > len(key) + 2:
            raise ValueError(
                f"family {key} has structural rank {rank} > N + 2, outside "
                "the integrand-reduction domain"
            )
        return Family(key, rank, tuple(closures))

    def _close(self, cut, positions, admitted, rejected) -> None:
        count = len(positions)
        symmetric = False
        if count >= 3:
            if lowest_leg(positions[1].mask) > lowest_leg(positions[-1].mask):
                return
        else:
            mirror = self._reflected(positions)
            routing = (cut, positions)
            if mirror < routing:
                return
            symmetric = mirror == routing
        key = canonical_family_key(
            [position.mask for position in positions],
            [self.catalogue.species(p.vertex.loop_out).mass for p in positions],
        )
        eligibility = LoopEligibility(
            tuple(
                self.catalogue.species(p.vertex.loop_out).perturbative_orders
                for p in positions
            ),
            frozenset(name for p in positions for name, _power in p.vertex.orders),
        )
        if not eligibility.admits(self.orders.corrections):
            return
        reason = selection_rejection(positions, self.catalogue)
        if reason is not None:
            rejected.setdefault(key, reason)
            return
        admitted[key].append(((cut, positions), symmetric))

    def _reflected(self, positions: tuple[Position, ...]):
        """The N <= 2 routing read in the opposite direction from position 1."""
        catalogue = self.catalogue
        mirrored = tuple(
            Position(catalogue.reflected(position.vertex), position.blocks)
            for position in (positions[:1] + positions[1:][::-1])
        )
        return mirrored[-1].vertex.loop_out, mirrored


def _closed_loop_classes(
    eligible: frozenset[str], catalogue: LoopVertexCatalogue, blocks: BlockTable
) -> dict[str, str]:
    """Massless closed-quark multiplicity, generalised (design §4.3).

    Species of equal mass, spin and colour whose loop vertices are pairwise
    structurally identical (equal signatures) and never change the species
    along the loop are merged into one representative. Only vertices whose
    attached fields occur as block currents of this process are compared, so
    the classes depend on the process (an external flavour line or a
    flavour-sensitive attachment keeps its species apart). Returns
    {member: representative} for the members of merged classes.
    """
    full = (1 << len(blocks.legs)) - 1
    fields = {block.field for mask in range(1, full) for block in blocks.currents(mask)}
    conjugate = {name: catalogue.species(name).antiname for name in eligible}

    def usable(vertex: LoopVertex) -> bool:
        return all(name in fields for name in vertex.attached)

    entering: dict[str, set[str]] = defaultdict(set)
    for name in eligible:
        for vertex in catalogue.vertices(name):
            if usable(vertex):
                entering[vertex.loop_out].add(name)

    def signature(name: str):
        own = {name, conjugate[name]}
        if entering[name] - own:
            return None
        found = set()
        for vertex in catalogue.vertices(name):
            if not usable(vertex):
                continue
            if (
                vertex.loop_out not in own
                or own & set(vertex.attached)
                or vertex.signature is None
            ):
                return None
            found.add(
                (
                    vertex.signature,
                    vertex.attached,
                    vertex.loop_out == name,
                    vertex.orders,
                    vertex.q_degree,
                )
            )
        species = catalogue.species(name)
        return species.mass, species.spin, species.colour, frozenset(found)

    groups: dict[Hashable, list[str]] = defaultdict(list)
    for name in sorted(eligible):
        key = signature(name)
        if key is not None and key[3]:
            groups[key].append(name)
    out: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2 or members[0] in out:
            continue
        # The conjugate class takes the conjugate representative, so that
        # reading a merged routing backwards stays within representatives.
        out.update(dict.fromkeys(members, members[0]))
        out.update(
            dict.fromkeys((conjugate[name] for name in members), conjugate[members[0]])
        )
    return out


def generate_families(
    blocks: BlockTable, catalogue: LoopVertexCatalogue, orders: LoopOrderSelection
) -> FamilySet:
    """Run the loop-prefix recursion and return the admitted families."""
    return _Recursion(blocks, catalogue, orders).run()


# ------------------------------------------------------------------- model tables


def model_data(model: Any) -> Mapping[str, Any]:
    """The native model's JSON tables (particles, vertices, ...); a mapping
    passes through unchanged."""
    if isinstance(model, Mapping):
        return model
    return json.loads(model.to_json())


def mass_name(name: str | None) -> str:
    """A model mass or width parameter name, with every zero spelling as ZERO."""
    return ZERO if name in (None, "ZERO", "0") else name


@cache
def _momentum_degree(expression: str, slots: tuple[int, int] | None) -> int:
    """Degree in the loop momentum q of a UFO structure.

    With ``slots = (in, out)`` the loop momentum enters at leg ``in`` and leaves
    at leg ``out`` (all UFO momenta incoming): P_in -> P_in + q and
    P_out -> P_out - q, so q cancels from structures in P_in + P_out. With
    ``None`` every momentum is the line momentum (a propagator numerator).
    A vertex momentum in any other form than ``P(mu, leg)`` is rejected rather
    than read as degree 0 (stand-in for WP1.1 ``LoopSegmentBinding.degree``).
    """
    from symbolica import E

    t, q = symbols.qualified("pyamplicol_v2_t"), symbols.qualified("pyamplicol_v2_q")
    momentum, slash, lorentz, rest = (
        symbols.qualified("P"),
        symbols.qualified("PSlash"),
        symbols.qualified("x_"),
        symbols.qualified("x___"),
    )
    value = E(expression)
    if slots is None:
        value = value.replace(momentum(rest), t * momentum(rest))
        value = value.replace(slash(rest), t * slash(rest))
    else:
        index, label = symbols.qualified("idx"), symbols.qualified("y_")
        other = value.replace(
            momentum(lorentz, index(label, symbols.qualified("z_"))), t
        )
        if other.matches(momentum(rest)) or other.matches(slash(rest)):
            raise NotImplementedError(
                f"vertex momentum outside P(mu, leg) form in {expression!r}"
            )
        for slot, sign in zip(slots, (1, -1), strict=True):
            leg = index(label, slot)
            value = value.replace(
                momentum(lorentz, leg), momentum(lorentz, leg) + sign * t * q(lorentz)
            )
    zero = E("0")
    return max(
        (
            int(power.to_polynomial(vars=[t]).degree(t))
            for power, coefficient in value.expand().coefficient_list(t)
            if coefficient.expand() != zero
        ),
        default=0,
    )


class ModelLoopCatalogue:
    """Loop species and vertex templates of a native HepKit model.

    An interim field-level implementation of ``LoopVertexCatalogue`` until the
    WP1.1 kernel catalogue is integrated. q-degrees are read from the exact UFO
    Lorentz structures and propagator numerators with Symbolica.

    Perturbative species (MadLoop's ``is_perturbating``) are the dependency
    facts ``species_orders`` keyed by |PDG|, e.g. a prepared structure's
    ``perturbative_species_orders``; a counterterm-augmented or request-pruned
    model must pass them. Without them they are derived from this model's
    rules by the same rule as the diagram oracle (a species is perturbative in
    an order when it enters a vertex of that single order), which production
    modules may not import.

    ``veto`` (|PDG| codes, ``particle_veto``) removes those particles from the
    loops: they are no loop species, and no vertex with one of them is a loop
    vertex.

    ``vanishing`` (vertex rule name, coupling orders) are the vertices whose
    couplings at those orders vanish identically at the model's restriction
    (``ct_sites.vanishing_loop_vertices``, decided from the coupling values):
    they are no loop vertices at those orders (fail-closed-2; e.g. a W vertex
    of a CKM element a diagonal-CKM restriction sets to zero, which the
    restricted tree model does not hold).
    """

    def __init__(
        self,
        model: Any,
        *,
        species_orders: Mapping[int, frozenset[str]] | None = None,
        veto: Iterable[int] = (),
        vanishing: Iterable[tuple[str, Orders]] = (),
    ):
        data = model_data(model)
        vanishing = frozenset(vanishing)
        veto = frozenset(abs(int(pdg)) for pdg in veto)
        vetoed = {p["name"] for p in data["particles"] if abs(p["pdg_code"]) in veto}
        self._species: dict[str, LoopSpecies] = {}
        coupling_orders = {
            entry["name"]: tuple(
                sorted((name, power) for name, power in entry["orders"])
            )
            for entry in data["couplings"]
        }
        lorentz = {
            entry["name"]: entry["structure"] for entry in data["lorentz_structures"]
        }
        numerators = {
            entry["particle"]: entry["numerator"] for entry in data["propagators"]
        }
        rules = []
        for rule in data["vertex_rules"]:
            if vetoed.intersection(rule["particles"]):
                continue
            orders = {
                add_orders(coupling_orders[name])
                for row in rule["couplings"]
                for name in row
                if name is not None
            }
            for order in sorted(orders):
                if (rule["name"], order) not in vanishing:
                    rules.append((rule, order))
        if species_orders is None:
            pdg = {p["name"]: abs(p["pdg_code"]) for p in data["particles"]}
            derived: dict[int, set[str]] = defaultdict(set)
            for rule, order in rules:
                if len(order) == 1:
                    for name in rule["particles"]:
                        derived[pdg[name]].add(order[0][0])
            species_orders = derived
        self._conjugate = {}
        for particle in data["particles"]:
            name = particle["name"]
            self._conjugate[name] = particle["antiname"]
            if not particle.get("propagating", True) or name in vetoed:
                continue
            self._species[name] = LoopSpecies(
                name,
                particle["antiname"],
                particle["spin"],
                particle["color"],
                MassId(mass_name(particle["mass"]), mass_name(particle["width"])),
                frozenset(species_orders.get(abs(particle["pdg_code"]), ())),
            )
        self._numerators = numerators
        templates: dict[tuple, LoopVertex] = {}
        for rule, order in rules:
            particles = rule["particles"]
            structures = [lorentz[name] for name in rule["lorentz_structures"]]
            for first in range(len(particles)):
                for second in range(len(particles)):
                    if first == second:
                        continue
                    loop_in = particles[first]
                    loop_out = self._conjugate[particles[second]]
                    if loop_in not in self._species or loop_out not in self._species:
                        continue
                    attached = tuple(
                        sorted(
                            p
                            for i, p in enumerate(particles)
                            if i not in (first, second)
                        )
                    )
                    degree = max(
                        _momentum_degree(structure, (first + 1, second + 1))
                        for structure in structures
                    ) + self._numerator_degree(loop_out)
                    identity = (rule["name"], loop_in, attached, loop_out, order)
                    previous = templates.get(identity)
                    if previous is None or previous.q_degree < degree:
                        signature = (
                            tuple(rule["lorentz_structures"]),
                            tuple(rule["color_structures"]),
                            tuple(map(tuple, rule["couplings"])),
                            tuple(p for p in particles if p in attached),
                            first,
                            second,
                        )
                        templates[identity] = LoopVertex(
                            rule["name"],
                            loop_in,
                            attached,
                            loop_out,
                            order,
                            degree,
                            signature,
                        )
        self._vertices: dict[str, tuple[LoopVertex, ...]] = defaultdict(tuple)
        grouped = defaultdict(list)
        for vertex in templates.values():
            grouped[vertex.loop_in].append(vertex)
        self._vertices.update({name: tuple(sorted(v)) for name, v in grouped.items()})
        self._by_identity = {
            (v.rule, v.loop_in, v.attached, v.loop_out, v.orders): v
            for v in templates.values()
        }

    def _numerator_degree(self, name: str) -> int:
        numerator = self._numerators.get(name)
        if numerator is None:
            raise NotImplementedError(f"particle {name!r} has no propagator numerator")
        return _momentum_degree(numerator, None)

    def species(self, name: str) -> LoopSpecies:
        return self._species[name]

    def conjugate(self, name: str) -> str:
        return self._conjugate[name]

    def loop_species(self) -> tuple[str, ...]:
        return tuple(sorted(self._species))

    def vertices(self, loop_in: str) -> tuple[LoopVertex, ...]:
        return self._vertices.get(loop_in, ())

    def reflected(self, vertex: LoopVertex) -> LoopVertex:
        return self._by_identity[
            (
                vertex.rule,
                self._conjugate[vertex.loop_out],
                vertex.attached,
                self._conjugate[vertex.loop_in],
                vertex.orders,
            )
        ]


class ModelBlockTable:
    """Field-level tree reachability over leg sets of a native HepKit model.

    An interim implementation of ``BlockTable`` until the WP1.2 loop-mode tree
    program exports its blocks: a current (leg set, field, orders) exists iff a
    tree of model vertices joins its legs with orders within ``cap``. Outgoing
    legs are crossed to incoming antiparticles.
    """

    def __init__(
        self,
        model: Any,
        incoming: Sequence[str],
        outgoing: Sequence[str],
        *,
        cap: Orders,
        ignored_particle_ids: Iterable[int] = (),
    ):
        data = model_data(model)
        particles = {entry["name"]: entry for entry in data["particles"]}
        # As the tree compiler: no current of an ignored (all-outgoing)
        # particle id is produced.
        ignored = frozenset(int(pdg) for pdg in ignored_particle_ids)
        conjugate = {name: entry["antiname"] for name, entry in particles.items()}
        coupling_orders = {
            entry["name"]: tuple(
                sorted((name, power) for name, power in entry["orders"])
            )
            for entry in data["couplings"]
        }
        legs = []
        for index, name in enumerate((*incoming, *(conjugate[n] for n in outgoing))):
            entry = particles[name]
            legs.append(
                ExternalLeg(
                    index,
                    name,
                    MassId(mass_name(entry["mass"]), mass_name(entry["width"])),
                    entry["color"],
                )
            )
        self._legs = tuple(legs)
        self._cap = cap
        # (attached fields, result field, orders) of every tree vertex use.
        self._merges: dict[tuple[str, ...], set[tuple[str, Orders]]] = defaultdict(set)
        for rule in data["vertex_rules"]:
            names = rule["particles"]
            orders = {
                add_orders(coupling_orders[name])
                for row in rule["couplings"]
                for name in row
                if name is not None
            }
            for result in range(len(names)):
                if int(particles[names[result]]["pdg_code"]) in ignored:
                    continue
                attached = tuple(sorted(n for i, n in enumerate(names) if i != result))
                for order in orders:
                    self._merges[attached].add((conjugate[names[result]], order))
        self._arity = max(map(len, self._merges))
        self._table: dict[int, tuple[BlockCurrent, ...]] = {}

    @property
    def legs(self) -> tuple[ExternalLeg, ...]:
        return self._legs

    def currents(self, mask: int) -> tuple[BlockCurrent, ...]:
        found = self._table.get(mask)
        if found is None:
            found = self._table[mask] = tuple(sorted(self._build(mask)))
        return found

    def _build(self, mask: int) -> set[BlockCurrent]:
        full = (1 << len(self._legs)) - 1
        if mask == full or not mask:
            return set()
        if mask & (mask - 1) == 0:
            return {BlockCurrent(mask, self._legs[lowest_leg(mask)].field, ())}
        out = set()
        for parts in _set_partitions(mask, 2, self._arity):
            for options in product(*(self.currents(part) for part in parts)):
                fields = tuple(sorted(block.field for block in options))
                for result, order in self._merges.get(fields, ()):
                    total = add_orders(order, *(block.orders for block in options))
                    if orders_within(total, self._cap):
                        out.add(BlockCurrent(mask, result, total))
        return out


def _set_partitions(mask: int, minimum: int, maximum: int) -> Iterator[tuple[int, ...]]:
    """Unordered partitions of ``mask`` into ``minimum..maximum`` parts."""

    def split(rest: int, parts: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if not rest:
            if len(parts) >= minimum:
                yield parts
            return
        if len(parts) == maximum:
            return
        low = rest & -rest
        others = rest ^ low
        sub = others
        while True:
            yield from split(others ^ sub, (*parts, sub | low))
            if not sub:
                break
            sub = (sub - 1) & others

    yield from split(mask, ())
