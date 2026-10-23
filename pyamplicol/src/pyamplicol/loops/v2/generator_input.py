# SPDX-License-Identifier: 0BSD
"""Inputs and decoders of the native v2 generator stages (WP3.1).

The native stages (``rusticol-core`` ``loop_v2::generate``) replace stages of
the Python generator one by one and must reproduce them exactly; the Python
generator stays their oracle. This module hands the native code Θ(model)
tables plus the field-level block table, with every name replaced by its rank
in a sorted table: the native code compares ids, and id order is Python's
object order (strings by code point, tuples lexicographically). The decoders
rebuild the Python objects, so every later stage reads the same objects
whichever engine produced them.

``NativeGeneration`` runs ``families.generate_families`` and
``schedules.build_schedules`` natively.
"""

from __future__ import annotations

import gc
import itertools
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .families import (
    BlockCurrent,
    BlockTable,
    Closure,
    Family,
    FamilyFrame,
    FamilyKey,
    FamilySet,
    LoggedFamily,
    LoopOrderSelection,
    LoopVertex,
    LoopVertexCatalogue,
    MassId,
    MasterKey,
    Orders,
    Position,
    PrefixGraph,
    _closed_loop_classes,
    retest_relabelling,
)
from .schedules import (
    STRUCTURED,
    CostModel,
    FamilySchedule,
    LoopSchedules,
    Segment,
    SkeletonView,
    SuffixTrie,
    TrieNode,
)
from .skeletons import ColourModel, Content, Skeleton

#: Generator stages run natively (WP3.1); the Python stages stay their
#: oracles. Removing a name runs that stage (and every later one) in Python.
NATIVE_STAGES: frozenset[str] = frozenset({"families", "schedules"})


@dataclass(frozen=True, slots=True)
class _Tables:
    """Sorted name, order-name, mass, vertex and block-current tables."""

    fields: tuple[str, ...]
    field_id: Mapping[str, int]
    order_names: tuple[str, ...]
    order_id: Mapping[str, int]
    masses: tuple[MassId, ...]
    mass_id: Mapping[MassId, int]
    vertices: tuple[LoopVertex, ...]
    vertex_id: Mapping[LoopVertex, int]
    currents: tuple[BlockCurrent, ...]

    def orders(self, orders: Orders) -> list[list[int]]:
        return [[self.order_id[name], power] for name, power in orders]


def _tables(
    table: BlockTable, catalogue: LoopVertexCatalogue, selection: LoopOrderSelection
) -> _Tables:
    legs = table.legs
    full = (1 << len(legs)) - 1
    species = tuple(catalogue.loop_species())
    vertices: set[LoopVertex] = set()
    for name in species:
        for vertex in catalogue.vertices(name):
            vertices.add(vertex)
            vertices.add(catalogue.reflected(vertex))
    currents = tuple(
        current for mask in range(1, full + 1) for current in table.currents(mask)
    )
    fields = {leg.field for leg in legs} | set(species)
    fields.update(catalogue.species(name).antiname for name in species)
    for vertex in vertices:
        fields.update((vertex.loop_in, vertex.loop_out, *vertex.attached))
    fields.update(current.field for current in currents)
    names = set(selection.corrections) | {name for name, _ in selection.cap}
    for orders in selection.loop_orders:
        names.update(name for name, _ in orders)
    for vertex in vertices:
        names.update(name for name, _ in vertex.orders)
    for current in currents:
        names.update(name for name, _ in current.orders)
    for name in species:
        names.update(catalogue.species(name).perturbative_orders)
    masses = {leg.mass for leg in legs} | {
        catalogue.species(name).mass for name in species
    }
    fields_ = tuple(sorted(fields))
    names_ = tuple(sorted(names))
    masses_ = tuple(sorted(masses))
    vertices_ = tuple(sorted(vertices))
    if len(set(vertices_)) != len(vertices_) or any(
        not a < b for a, b in itertools.pairwise(vertices_)
    ):
        raise ValueError("loop-vertex templates must be totally ordered")
    if any(not a < b for a, b in itertools.pairwise(currents)):
        raise ValueError("block currents must be strictly increasing")
    return _Tables(
        fields_,
        {name: i for i, name in enumerate(fields_)},
        names_,
        {name: i for i, name in enumerate(names_)},
        masses_,
        {mass: i for i, mass in enumerate(masses_)},
        vertices_,
        {vertex: i for i, vertex in enumerate(vertices_)},
        currents,
    )


def family_input(
    table: BlockTable, catalogue: LoopVertexCatalogue, selection: LoopOrderSelection
) -> tuple[_Tables, dict[str, Any], Mapping[str, str]]:
    """The native family-recursion input, its tables and the merged classes."""

    tables = _tables(table, catalogue, selection)
    fid, oid = tables.field_id, tables.order_id
    eligible = frozenset(
        name
        for name in catalogue.loop_species()
        if catalogue.species(name).perturbative_orders & selection.corrections
    )
    merged = _closed_loop_classes(eligible, catalogue, table)
    sizes: dict[str, int] = defaultdict(int)
    for representative in merged.values():
        sizes[representative] += 1
    representatives = sorted(
        name for name in eligible if merged.get(name, name) == name
    )
    species: list[dict[str, Any] | None] = [None] * len(tables.fields)
    for name in catalogue.loop_species():
        entry = catalogue.species(name)
        species[fid[name]] = {
            "antiname": fid[entry.antiname],
            "colour": entry.colour,
            "mass": tables.mass_id[entry.mass],
            "perturbative": sorted(oid[o] for o in entry.perturbative_orders),
        }
    by_loop_in: list[list[int]] = [[] for _ in tables.fields]
    for name in catalogue.loop_species():
        by_loop_in[fid[name]] = [
            tables.vertex_id[vertex] for vertex in catalogue.vertices(name)
        ]
    multiplicity = [1] * len(tables.fields)
    for name, size in sizes.items():
        multiplicity[fid[name]] = size
    payload = {
        "legs": [
            {"field": fid[leg.field], "mass": tables.mass_id[leg.mass]}
            for leg in table.legs
        ],
        "species": species,
        "massless": [mass.massless for mass in tables.masses],
        "vertices": [
            {
                "loop_in": fid[vertex.loop_in],
                "attached": [fid[name] for name in vertex.attached],
                "loop_out": fid[vertex.loop_out],
                "orders": tables.orders(vertex.orders),
                "q_degree": vertex.q_degree,
                "reflected": tables.vertex_id[catalogue.reflected(vertex)],
            }
            for vertex in tables.vertices
        ],
        "vertices_by_loop_in": by_loop_in,
        "currents": [
            {
                "mask": current.mask,
                "field": fid[current.field],
                "orders": tables.orders(current.orders),
            }
            for current in tables.currents
        ],
        "eligible": [fid[name] for name in representatives],
        "multiplicity": multiplicity,
        "corrections": sorted(oid[name] for name in selection.corrections),
        "loop_orders": [
            tables.orders(orders) for orders in sorted(selection.loop_orders)
        ],
        "cap": tables.orders(selection.cap),
    }
    return tables, payload, merged


def _key(tables: _Tables, raw) -> tuple[tuple[int, MassId], ...]:
    masses = tables.masses
    return tuple([(mask, masses[mass]) for mask, mass in raw])


_ONE = Fraction(1)
_HALF = Fraction(1, 2)


class NativeGeneration:
    """The native generator stages of one process (WP3.1 I8/I9).

    The loop-prefix recursion runs at construction; ``families()`` decodes it
    into ``families.generate_families``'s ``FamilySet`` and ``schedules()``
    builds the skeletons, the view and the schedule-B tries of
    ``schedules.build_schedules``. ``native`` is the staged loop core
    (``loops.native.load_loop_native``). Every decoded object equals the
    Python stage's (``tests/unit/test_loop_v2_generate.py``).
    """

    def __init__(
        self,
        table: BlockTable,
        catalogue: LoopVertexCatalogue,
        selection: LoopOrderSelection,
        native: Any,
    ):
        tables, payload, merged = family_input(table, catalogue, selection)
        self.tables = tables
        self.catalogue = catalogue
        self.legs = tuple(table.legs)
        self._merged = merged
        self._native = native.V2Generation(_json(payload))
        self._families: FamilySet | None = None
        self._positions: list[Position] = []
        self._state_positions: list[int] = []

    def _position(self, vertex: int, blocks) -> Position:
        tables = self.tables
        return Position(
            tables.vertices[vertex], tuple(tables.currents[index] for index in blocks)
        )

    def families(self) -> FamilySet:
        """``generate_families(table, catalogue, selection)``."""

        if self._families is not None:
            return self._families
        with _no_cyclic_gc():
            return self._decode_families()

    def _decode_families(self) -> FamilySet:
        tables = self.tables
        (
            raw_positions,
            parents,
            labels,
            species,
            raw_families,
            scaleless,
            rejected,
            raw_scaleless,
        ) = self._native.families()
        positions = [self._position(vertex, blocks) for vertex, blocks in raw_positions]
        graph = PrefixGraph()
        fields = tables.fields
        for parent, label, name in zip(parents, labels, species, strict=True):
            state = len(graph.parent)
            graph.parent.append(parent)
            if parent < 0:
                graph.position.append(None)
                graph._index[(parent, fields[name])] = state
                graph.depth.append(0)
                self._state_positions.append(-1)
            else:
                position = positions[label]
                graph.position.append(position)
                graph._index[(parent, position)] = state
                graph.depth.append(graph.depth[parent] + 1)
                self._state_positions.append(label)
            graph.species.append(fields[name])
        def decoded(raws) -> dict[FamilyKey, Family]:
            out = {}
            for raw, rank, closures in raws:
                key = _key(tables, raw)
                if rank > len(key) + 2:
                    raise ValueError(
                        f"family {key} has structural rank {rank} > N + 2, "
                        "outside the integrand-reduction domain"
                    )
                out[key] = Family(
                    key,
                    rank,
                    tuple(
                        Closure(state, symmetric, multiplicity)
                        for state, symmetric, multiplicity in closures
                    ),
                )
            return out

        families = decoded(raw_families)
        # The certified-scaleless families with their routings, interned
        # after every admitted family's states (the poles companion).
        scaleless_families = decoded(raw_scaleless)
        self._positions = positions
        self._families = FamilySet(
            self.legs,
            families,
            graph,
            tuple(
                LoggedFamily(_key(tables, raw), "certified-scaleless")
                for raw in scaleless
            ),
            tuple(LoggedFamily(_key(tables, raw), reason) for raw, reason in rejected),
            self._merged,
            scaleless_families,
        )
        return self._families

    def schedules(
        self,
        colour: ColourModel,
        *,
        model: CostModel = STRUCTURED,
        frames: Mapping[FamilyKey, FamilyFrame] | None = None,
        keep: Callable[[Skeleton], bool] | None = None,
    ) -> LoopSchedules:
        """``build_schedules(families, catalogue, colour, ...)`` natively.

        Without ``frames`` the policy is ``cheapest(lambda mask: 2 **
        mask.bit_count(), model)``; with ``frames`` (a family's stored frame,
        e.g. ``_prune_colour``'s ``same_frame``) only that frame's trie is
        built. ``keep`` filters skeletons as ``SkeletonView(keep=...)``.
        """

        families = self.families()
        used = sorted({label for label in self._state_positions if label >= 0})
        per_position = {p: colour.states(self._positions[p]) for p in used}
        states: set = set()
        for found in per_position.values():
            states.update(found)
        pending = list(states)
        reflection = {}
        while pending:
            state = pending.pop()
            mirror = reflection[state] = colour.reflected(state)
            if mirror not in states:
                states.add(mirror)
                pending.append(mirror)
        ordered = sorted(states)
        if any(not a < b for a, b in itertools.pairwise(ordered)):
            raise ValueError("colour states must be totally ordered")
        rank = {state: i for i, state in enumerate(ordered)}
        state_rows: list[list[int] | None] = [None] * len(self._positions)
        for position, found in per_position.items():
            state_rows[position] = [rank[state] for state in found]
        keys = list(families.families)
        if frames is None:
            policy: dict[str, Any] = {
                "kind": "cheapest",
                "model": {
                    name: getattr(model, name)
                    for name in (
                        "fused",
                        "split",
                        "vertex",
                        "seed_fused",
                        "seed_split",
                        "closure",
                        "columns",
                    )
                },
            }
        else:
            policy = {
                "kind": "fixed",
                "frames": [
                    (
                        [frames[key].start, frames[key].reflected]
                        if key in frames
                        else None
                    )
                    for key in keys
                ],
            }
        payload: dict[str, Any] = {
            "states": state_rows,
            "reflected": [rank[reflection[state]] for state in ordered],
            "policy": policy,
        }
        if keep is not None:
            payload["keep"] = self._keep_flags(keep, ordered, payload, keys)
        raw_contents, labels, raw_orders, raw_schedules = self._native.schedules(
            _json(payload)
        )
        with _no_cyclic_gc():
            return self._decode(
                colour, keep, ordered, raw_contents, labels, raw_orders, raw_schedules
            )

    def _keep_flags(self, keep, ordered, payload, keys) -> list[list[bool]]:
        """``keep`` evaluated on every skeleton of every family."""

        raw_contents, raw_orders, raw_skeletons = self._native.skeletons(_json(payload))
        orders = self._orders(raw_orders)
        contents = [self._content(ordered, orders, raw) for raw in raw_contents]
        return [
            [keep(self._skeleton(key, contents, raw)) for raw in family]
            for key, family in zip(keys, raw_skeletons, strict=True)
        ]

    def _orders(self, raw_orders) -> list[Orders]:
        names = self.tables.order_names
        return [tuple((names[n], power) for n, power in o) for o in raw_orders]

    def _content(self, states, orders, raw) -> Content:
        state, loop_in, loop_out, mask, order, terms, _q_degree = raw
        fields = self.tables.fields
        position = self._position
        return Content(
            states[state],
            fields[loop_in],
            fields[loop_out],
            mask,
            orders[order],
            tuple(position(vertex, blocks) for vertex, blocks in terms),
        )

    @staticmethod
    def _skeleton(key, contents, raw) -> Skeleton:
        ids, half, multiplicity = raw
        return Skeleton(
            key,
            tuple(map(contents.__getitem__, ids)),
            _HALF if half else _ONE,
            multiplicity,
        )

    def _decode(
        self, colour, keep, ordered, raw_contents, labels, raw_orders, raw_schedules
    ) -> LoopSchedules:
        families = self.families()
        catalogue = self.catalogue
        orders = self._orders(raw_orders)
        contents = [self._content(ordered, orders, raw) for raw in raw_contents]
        view = SkeletonView(catalogue, colour, keep=keep)
        view._contents = [contents[c] for c in labels]
        view._labels = {content: label for label, content in enumerate(view._contents)}
        segments = [
            Segment(
                label,
                content.mask,
                catalogue.species(content.loop_out).mass,
                raw_contents[c][6],
            )
            for label, (c, content) in enumerate(
                zip(labels, view._contents, strict=True)
            )
        ]
        segment = segments.__getitem__
        keys = list(families.families)
        out = {}
        for (
            index,
            raw_skeletons,
            viewed,
            (start, reflected),
            leaves,
            nodes,
            rank,
            masters,
        ) in raw_schedules:
            key = keys[index]
            skeletons = [self._skeleton(key, contents, raw) for raw in raw_skeletons]
            chosen = [skeletons[i] for i in viewed]
            frame = FamilyFrame(start, reflected)
            trie = SuffixTrie(
                key,
                frame,
                tuple((chosen[i], tuple(map(segment, ring))) for i, ring in leaves),
                tuple(
                    TrieNode(
                        depth,
                        orders[order],
                        tuple(map(segment, suffix)),
                        rank_in,
                        tuple((segments[label], child) for label, child in edges),
                    )
                    for depth, order, suffix, rank_in, edges in nodes
                ),
                rank,
            )
            out[key] = FamilySchedule(
                families.families[key],
                trie,
                retest_relabelling(key, frame),
                tuple(
                    MasterKey(_key(self.tables, raw), scaleless)
                    for raw, scaleless in masters
                ),
            )
        return LoopSchedules(families, view, out)


@contextmanager
def _no_cyclic_gc():
    """Decoding builds millions of acyclic objects; the cyclic collector
    would traverse the growing heap again and again while they are made."""

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))
