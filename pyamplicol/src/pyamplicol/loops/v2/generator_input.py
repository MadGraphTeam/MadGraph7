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
``schedules.build_schedules`` natively, the latter also for the poles
companion's certified-scaleless families (``scaleless_companion``).

**Retained arena** (WP3.1 B1, package native-seeds). The native generation
keeps every schedule build (``loop_v2::generate::arena``) and this module
keeps the Python objects decoded from it (``NativeBuild``): the contents by
arena id and, for a build of every skeleton, every skeleton of every
scheduled family. The LC/NLC prune (``schedules(frames=..., keep=...)``)
rebuilds the tries natively from those skeletons instead of rebuilding and
decoding them again: ``keep`` is evaluated on the retained skeleton objects
(or comes as per-reading flags of the native prune liveness,
``ReadingKeep``), and the rebuilt plan reuses the retained objects. The seed
stage reads a build's readings in-crate (``NativeBuild.readings_of``), and
``NativeBuild.max_rank`` answers ``MonomialLayout``'s rank bound.

**Frame cost** (genericity audit m4). The cheapest-frame policy prices a
segment by the helicity rows of its legs: ``leg_rows(blocks)`` counts each
leg's helicities in the request's enumeration (1 for a scalar, 3 for a
massive vector, ...) and ``schedules(leg_rows=...)`` (native) and
``cheapest(frame_rows(...), model)`` (Python) use their product. Without
``leg_rows`` both price two rows per leg, exact whenever every leg has two
helicity states (fermions, massless vectors).
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import sys
import weakref
from array import array
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from operator import attrgetter
from typing import Any, cast

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


def _key(tables: _Tables, raw: Iterable[Any]) -> tuple[tuple[int, MassId], ...]:
    masses = tables.masses
    return tuple([(mask, masses[mass]) for mask, mass in raw])


_ONE = Fraction(1)
_HALF = Fraction(1, 2)

#: Keep each stage's native inputs as JSON (``NativeGeneration.input_json``,
#: ``native_seeds.NativeSeedStage.spec_text``, ``reduce_inputs`` and
#: ``reductions_text``) to replay the stage in another process (tests). Off
#: in production: the copies would live as long as the generation.
KEEP_REPLAY_INPUTS = False


def leg_rows(blocks: Any) -> tuple[int, ...]:
    """The helicity rows of each external leg (bit ``i`` of a mask is tree
    label ``i + 1``): the distinct helicities of the rows of the leg's source
    blocks (``TreeBlocks.blocks``), i.e. the request's own helicity
    enumeration. The frame policy prices a segment by their product
    (``frame_rows``, ``NativeGeneration.schedules(leg_rows=...)``)."""

    found: dict[int, set[int]] = defaultdict(set)
    for block in blocks.blocks:
        if block.is_source:
            for row in block.rows:
                for label, helicity in row.helicities:
                    found[label].add(helicity)
    count = len(blocks.process.legs)
    if set(found) != set(range(1, count + 1)):
        raise ValueError("every external leg needs a source block with rows")
    return tuple(len(found[label]) for label in range(1, count + 1))


def frame_rows(rows: Sequence[int]) -> Callable[[int], int]:
    """``schedules.cheapest``'s ``rows`` from per-leg helicity rows: the
    product over a mask's legs (the Python twin of the native policy with
    ``leg_rows``; ``2 ** mask.bit_count()`` when every leg has two)."""

    counts = tuple(int(count) for count in rows)

    def product(mask: int) -> int:
        out = 1
        while mask:
            low = mask & -mask
            out *= counts[low.bit_length() - 1]
            mask ^= low
        return out

    return product


class NativeBuild:
    """A schedule build retained by the native generation (its arena
    handle) with the Python objects decoded from it.

    ``contents`` are the decoded contents by arena content id. A build of
    every skeleton (``complete``) also keeps ``skeletons``: every skeleton
    of every scheduled family, in the native order, which its rebuilds
    reuse. ``plan`` is the decoded ``LoopSchedules``.
    """

    def __init__(
        self,
        generation: NativeGeneration,
        handle: int,
        colour: ColourModel,
        payload: Mapping[str, Any],
        ordered: list[Hashable],
        contents: list[Content],
        skeletons: list[list[Skeleton]],
        complete: bool,
    ):
        self.generation = generation
        self.handle = handle
        self.colour = colour
        self.payload = payload
        self.ordered = ordered
        self.contents = contents
        self.skeletons = skeletons
        self.complete = complete
        self.plan: LoopSchedules | None = None
        # The seed stage that read the build in-crate (a complete build;
        # ``native_seeds.prune_keep`` reads its native liveness).
        self.seed_stage: Any = None
        self.first_leaf: Skeleton | None = None
        self.leaves = 0
        self._readings: tuple[bytes, bytes, list[int]] | None = None
        _LIVE_BUILDS.add(self)

    @property
    def native(self) -> Any:
        """The native generation holding the build (``V2Generation``)."""

        return self.generation._native

    def readings(self) -> tuple[bytes, bytes, list[int]]:
        """The build's readings (``reference_executor.reading_skeletons``):
        little-endian u64 offsets and u32 arena content ids, and the
        distinct content ids, first seen."""

        if self._readings is None:
            self._readings = self.native.readings(self.handle)
        return self._readings

    def readings_of(self, skeletons: Sequence[Skeleton]) -> list[int] | None:
        """The distinct reading contents (arena ids, first seen) when
        ``skeletons`` are exactly this build's readings: the same content
        objects in the same places. None otherwise."""

        if 2 * self.leaves != len(skeletons) or not skeletons:
            return None
        if skeletons[0] is not self.first_leaf:
            return None
        offsets, ids, used = self.readings()
        index = {id(content): i for i, content in enumerate(self.contents)}
        contents = attrgetter("contents")
        try:
            given = array(
                "I",
                map(
                    index.__getitem__,
                    map(id, itertools.chain.from_iterable(map(contents, skeletons))),
                ),
            )
        except KeyError:
            return None
        spans = array("Q", itertools.accumulate(map(len, map(contents, skeletons))))
        spans.insert(0, 0)
        if sys.byteorder != "little":
            given.byteswap()
            spans.byteswap()
        if given.tobytes() != ids or spans.tobytes() != offsets:
            return None
        return used

    def max_rank(self) -> int:
        """``MonomialLayout``'s rank bound of the build's plan: the largest
        ``rank_in`` plus edge q-degree over every trie node."""

        return int(self.native.max_rank(self.handle))


#: Live retained builds, for the in-crate seed stage (``build_of``).
_LIVE_BUILDS: weakref.WeakSet[NativeBuild] = weakref.WeakSet()


def build_of(skeletons: Sequence[Skeleton]) -> tuple[NativeBuild, list[int]] | None:
    """The retained build whose readings are ``skeletons`` and its distinct
    reading contents (``NativeBuild.readings_of``), or None."""

    if not skeletons:
        return None
    for build in sorted(_LIVE_BUILDS, key=attrgetter("handle"), reverse=True):
        if build.first_leaf is skeletons[0]:
            used = build.readings_of(skeletons)
            if used is not None:
                return build, used
    return None


@dataclass(frozen=True, eq=False)
class ReadingKeep:
    """Prune flags per reading of a complete build (one byte each; the F
    reading of a skeleton decides it): the native prune liveness
    (``native_seeds.NativeSeedStage.keep``). ``NativeGeneration.schedules``
    takes it in place of a ``keep`` callable."""

    build: NativeBuild
    flags: bytes

    def __call__(self, skeleton: Skeleton) -> bool:
        kept = self._kept()
        found = kept.get(id(skeleton))
        if found is None or found[0] is not skeleton:
            raise KeyError("the skeleton is not a reading of the pruned build")
        return found[1]

    def _kept(self) -> dict[int, tuple[Skeleton, bool]]:
        cache = self.__dict__.get("_cache")
        if cache is None:
            cache = {}
            plan = self.build.plan
            if plan is None:
                raise RuntimeError("the build has no decoded plan")
            r = 0
            for schedule in plan.schedules.values():
                for skeleton, _ring in schedule.trie.leaves:
                    cache[id(skeleton)] = (skeleton, bool(self.flags[r]))
                    r += 2
            object.__setattr__(self, "_cache", cache)
        return cache


class NativeGeneration:
    """The native generator stages of one process (WP3.1 I8/I9).

    The loop-prefix recursion runs at construction; ``families()`` decodes it
    into ``families.generate_families``'s ``FamilySet`` and ``schedules()``
    builds the skeletons, the view and the schedule-B tries of
    ``schedules.build_schedules``. ``native`` is the staged loop core
    (``loops.native.load_loop_native``). Every decoded object equals the
    Python stage's (``tests/unit/test_loop_v2_generate.py``).

    ``scaleless_companion`` returns the same generation scheduling a subset
    of the certified-scaleless families instead (the poles companion).

    Every build is retained (``NativeBuild``, module docstring); ``base`` is
    the last build of every skeleton, which a pruned rebuild reuses, and
    ``build_of_plan`` finds the build of a decoded plan.
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
        text = _json(payload)
        #: The native family input (JSON), to rebuild the generation in
        #: another process; kept only with ``KEEP_REPLAY_INPUTS``.
        self.input_json: str | None = text if KEEP_REPLAY_INPUTS else None
        self._native = native.V2Generation(text)
        self._families: FamilySet | None = None
        self._positions: list[Position] = []
        self._state_positions: list[int] = []
        # Indices into the native certified-scaleless families scheduled
        # instead of the admitted ones (``scaleless_companion``).
        self._scaleless: list[int] | None = None
        # The last build of every skeleton of this family list (a rebuild's
        # base) and every build by id of its decoded plan.
        self.base: NativeBuild | None = None
        self._builds: dict[int, NativeBuild] = {}
        # Python decodes of native output (RUST-MIGRATION rule 9): complete
        # and filtered builds, rebuilds from a base, and skeleton lists
        # decoded to evaluate a keep callback without a base.
        self.decodes: dict[str, int] = defaultdict(int)

    def build_of_plan(self, plan: LoopSchedules) -> NativeBuild | None:
        """The retained build ``plan`` was decoded from, or None."""

        build = self._builds.get(id(plan))
        return build if build is not None and build.plan is plan else None

    def _position(self, vertex: int, blocks: Iterable[int]) -> Position:
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

    def scaleless_companion(self, families: FamilySet) -> NativeGeneration:
        """This generation scheduling ``families.families``, certified-
        scaleless families of this one (``FamilySet.scaleless_families``), in
        their order: the poles companion's ``build_schedules(families, ...)``
        (``LoopSlice._scaleless_companion``). The prefix graph is shared."""

        index = {key: i for i, key in enumerate(self.families().scaleless_families)}
        missing = [key for key in families.families if key not in index]
        if missing:
            raise ValueError(
                f"families {missing} are not certified-scaleless families of "
                "this generation"
            )
        twin = copy.copy(self)
        twin._families = families
        twin._scaleless = [index[key] for key in families.families]
        twin.base = None
        twin._builds = {}
        twin.decodes = defaultdict(int)
        return twin

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

        def decoded(raws: Iterable[Any]) -> dict[FamilyKey, Family]:
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
        leg_rows: Sequence[int] | None = None,
    ) -> LoopSchedules:
        """``build_schedules(families, catalogue, colour, ...)`` natively.

        Without ``frames`` the policy is ``cheapest(frame_rows(leg_rows),
        model)``, the helicity rows of each leg (``leg_rows(blocks)``,
        required) pricing a frame. With ``frames`` (a family's stored frame,
        e.g. ``_prune_colour``'s ``same_frame``) only that frame's trie is
        built. ``keep`` filters skeletons as ``SkeletonView(keep=...)``; it
        may also be a ``ReadingKeep`` of the base build. With a base build
        of the same colour model the filtered schedules are rebuilt from its
        retained skeletons (``keep`` evaluated on them).
        """

        base = self.base
        if keep is not None and base is not None and base.colour is colour:
            return self._rebuild(base, colour, model, frames, keep, leg_rows)
        if isinstance(keep, ReadingKeep):
            raise ValueError("reading flags need their complete build as the base")
        payload, ordered = self._colour_payload(colour)
        keys = list(self.families().families)
        payload = {**payload, "policy": self._policy(model, frames, keys, leg_rows)}
        if keep is not None:
            payload["keep"] = self._keep_flags(keep, payload, ordered, keys)
        raw = self._native.schedules(_json(payload))
        with _no_cyclic_gc():
            return self._decode(colour, keep, payload, ordered, raw, None)

    def _colour_payload(
        self, colour: ColourModel
    ) -> tuple[dict[str, Any], list[Hashable]]:
        """The colour states of the used positions as ranks, their
        reflections (and the scheduled certified-scaleless families), and
        the ordered states the ranks index."""

        families = self.families()
        if self._scaleless is None:
            used = sorted({label for label in self._state_positions if label >= 0})
        else:
            used = sorted(self._path_positions(families))
        per_position = {p: colour.states(self._positions[p]) for p in used}
        states: set[Hashable] = set()
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
        # Colour states are totally ordered (checked); typed as Any to sort.
        ordered: list[Any] = sorted(cast(set[Any], states))
        if any(not a < b for a, b in itertools.pairwise(ordered)):
            raise ValueError("colour states must be totally ordered")
        rank = {state: i for i, state in enumerate(ordered)}
        state_rows: list[list[int] | None] = [None] * len(self._positions)
        for position, found in per_position.items():
            state_rows[position] = [rank[state] for state in found]
        payload: dict[str, Any] = {
            "states": state_rows,
            "reflected": [rank[reflection[state]] for state in ordered],
        }
        if self._scaleless is not None:
            payload["scaleless"] = self._scaleless
        return payload, ordered

    def _policy(
        self,
        model: CostModel,
        frames: Mapping[FamilyKey, FamilyFrame] | None,
        keys: Sequence[FamilyKey],
        leg_rows: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if frames is None:
            # The cheapest frame prices a segment by its legs' helicity rows:
            # the request's counts are required (no count per leg is assumed;
            # genericity audit 2, m8).
            if leg_rows is None:
                raise ValueError(
                    "the cheapest frame policy needs the helicity rows of every "
                    "leg (leg_rows=generator_input.leg_rows(blocks))"
                )
            rows = [int(count) for count in leg_rows]
            if len(rows) != len(self.legs) or min(rows, default=1) < 1:
                raise ValueError("leg_rows needs a positive count per leg")
            return {
                "kind": "cheapest",
                "leg_rows": rows,
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
        return {
            "kind": "fixed",
            "frames": [
                ([frames[key].start, frames[key].reflected] if key in frames else None)
                for key in keys
            ],
        }

    def _rebuild(
        self,
        base: NativeBuild,
        colour: ColourModel,
        model: CostModel,
        frames: Mapping[FamilyKey, FamilyFrame] | None,
        keep: Callable[[Skeleton], bool],
        leg_rows: Sequence[int] | None,
    ) -> LoopSchedules:
        """The filtered schedules rebuilt from ``base``'s skeletons."""

        keys = list(self.families().families)
        payload = {
            **base.payload,
            "policy": self._policy(model, frames, keys, leg_rows),
        }
        if isinstance(keep, ReadingKeep):
            if keep.build is not base:
                raise ValueError("reading flags of another build than the base")
            raw = self._native.rebuild(
                base.handle, _json(payload), keep.flags, readings=True
            )
        else:
            flags = bytes(
                map(bool, map(keep, itertools.chain.from_iterable(base.skeletons)))
            )
            raw = self._native.rebuild(base.handle, _json(payload), flags)
        with _no_cyclic_gc():
            return self._decode(colour, keep, payload, base.ordered, raw, base)

    def _path_positions(self, families: FamilySet) -> set[int]:
        """The positions on the prefix paths of ``families``' closures."""

        parent = families.prefixes.parent
        seen: set[int] = set()
        found: set[int] = set()
        for family in families.families.values():
            for closure in family.closures:
                state = closure.state
                while state >= 0 and state not in seen:
                    seen.add(state)
                    if (label := self._state_positions[state]) >= 0:
                        found.add(label)
                    state = parent[state]
        return found

    def _keep_flags(
        self,
        keep: Callable[[Skeleton], bool],
        payload: Mapping[str, Any],
        ordered: list[Hashable],
        keys: Sequence[FamilyKey],
    ) -> list[list[bool]]:
        """``keep`` evaluated on every skeleton of every family (without a
        base build to reuse)."""

        self.decodes["skeletons"] += 1
        raw_contents, raw_orders, raw_skeletons = self._native.skeletons(_json(payload))
        orders = self._orders(raw_orders)
        contents = [self._content(ordered, orders, raw) for raw in raw_contents]
        return [
            [keep(self._skeleton(key, contents, raw)) for raw in family]
            for key, family in zip(keys, raw_skeletons, strict=True)
        ]

    def _orders(self, raw_orders: Iterable[Iterable[tuple[int, int]]]) -> list[Orders]:
        names = self.tables.order_names
        return [tuple((names[n], power) for n, power in o) for o in raw_orders]

    def _content(
        self, states: Sequence[Hashable], orders: Sequence[Orders], raw: Sequence[Any]
    ) -> Content:
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
    def _skeleton(
        key: FamilyKey, contents: Sequence[Content], raw: Sequence[Any]
    ) -> Skeleton:
        ids, half, multiplicity = raw
        return Skeleton(
            key,
            tuple(map(contents.__getitem__, ids)),
            _HALF if half else _ONE,
            multiplicity,
        )

    def _decode(
        self,
        colour: ColourModel,
        keep: Callable[[Skeleton], bool] | None,
        payload: Mapping[str, Any],
        ordered: list[Hashable],
        raw: tuple[Any, ...],
        base: NativeBuild | None,
    ) -> LoopSchedules:
        """The ``LoopSchedules`` of a native build (``base``: the build a
        rebuild reused, whose contents and skeletons it shares)."""

        handle, first, raw_contents, labels, raw_orders, raw_schedules = raw
        self.decodes["build" if base is None else "rebuild"] += 1
        families = self.families()
        catalogue = self.catalogue
        orders = self._orders(raw_orders)
        if base is None:
            if first != 0:
                raise ValueError("a first build starts at content 0")
            contents = []
        else:
            if first != len(base.contents):
                raise ValueError("a rebuild continues its base's contents")
            contents = list(base.contents)
        contents.extend(self._content(ordered, orders, raw) for raw in raw_contents)
        view = SkeletonView(catalogue, colour, keep=keep)
        view._contents = [contents[c] for c in labels]
        view._labels = {content: label for label, content in enumerate(view._contents)}
        segments = [
            Segment(
                label,
                content.mask,
                catalogue.species(content.loop_out).mass,
                content.q_degree,
            )
            for label, content in enumerate(view._contents)
        ]
        segment = segments.__getitem__
        keys = list(families.families)
        retained: list[list[Skeleton]] = [] if base is None else base.skeletons
        if base is None:
            by_family: dict[int, list[Skeleton]] = {}
        out = {}
        first_leaf = None
        leaves_count = 0
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
            if base is None:
                skeletons = [
                    self._skeleton(key, contents, raw) for raw in raw_skeletons
                ]
                by_family[index] = skeletons
            else:
                skeletons = retained[index]
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
            if first_leaf is None and trie.leaves:
                first_leaf = trie.leaves[0][0]
            leaves_count += len(trie.leaves)
            out[key] = FamilySchedule(
                families.families[key],
                trie,
                retest_relabelling(key, frame),
                tuple(
                    MasterKey(_key(self.tables, raw), scaleless)
                    for raw, scaleless in masters
                ),
            )
        plan = LoopSchedules(families, view, out)
        complete = base is None and "keep" not in payload
        if base is None:
            # Families without a viewed skeleton have none (a complete build
            # views every family with a skeleton).
            retained = [by_family.get(i, []) for i in range(len(keys))]
        send = {k: v for k, v in payload.items() if k not in ("policy", "keep")}
        build = NativeBuild(
            self, handle, colour, send, ordered, contents, retained, complete
        )
        build.plan = plan
        build.first_leaf = first_leaf
        build.leaves = leaves_count
        self._builds[id(plan)] = build
        if complete:
            self.base = build
        return plan


@contextmanager
def _no_cyclic_gc() -> Iterator[None]:
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
