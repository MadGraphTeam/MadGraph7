# SPDX-License-Identifier: 0BSD
"""Schedules over the skeletons of one generator (design §2.2, §3.5, §5.4).

The three schedules are key projections of the same skeleton set:

* **B** (default): per family, a suffix trie in a per-family cost-optimal
  frame. A node holds the undressed positions k+1..N; its edges are the
  distinct contents of position k, each from the node of positions k..N.
  Leaves (depth 0) are skeletons, a depth-N node is a family accumulator.
  Nodes also carry the loop orders of their skeletons, so a family has one
  accumulator (one reduction) per loop order.
* **F** (baseline, oracle): prefixes in the F frame of the family recursion,
  shared across families; closures scatter into families.
* **M** (merged): dressed positions keep only their leg sets and species, the
  undressed remainder is one colour word, so prefixes merge across families
  and seed orderings. It needs colour-ordered contents (words).

Contents carry exact coupling orders, and F and M states carry the orders of
their dressed prefix, so no state mixes squared-order sectors (design §3.5).

Frames, buckets (N, R), master keys and re-test relabellings complete the
per-family reduction data; ``trie_shapes`` exports the B node shapes for the
kernel gate G0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb

from .families import (
    Family,
    FamilyFrame,
    FamilyKey,
    FamilySet,
    LoopVertexCatalogue,
    MassId,
    MasterKey,
    Orders,
    Relabelling,
    add_orders,
    retest_relabelling,
    ring_readings,
)
from .skeletons import (
    ColourModel,
    Content,
    Skeleton,
    family_skeletons,
    reflect_ring,
)

# --------------------------------------------------------------------------- views


@dataclass(frozen=True, slots=True, order=True)
class Segment:
    """A content as seen by a schedule.

    ``mass`` is the mass id of the propagator following the position.
    """

    label: Hashable
    mask: int
    mass: MassId
    q_degree: int


Ring = tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class ViewedSkeleton:
    """A skeleton with its 2N readings: forward from each start, then reversed."""

    skeleton: Skeleton
    readings: tuple[Ring, ...]

    @property
    def canonical(self) -> Ring:
        return min(self.readings)


class SkeletonView:
    """Projection of skeleton contents onto schedule segments.

    By default a segment is keyed by the full content (colour state, both
    species, orders and every summed term), interned as an integer label;
    ``content(label)`` recovers it. ``classes(skeleton)`` may return a species
    projection for that skeleton (e.g. to compare with a cost model that merges
    flavours or orientations; orders are kept); ``keep`` filters skeletons.
    """

    def __init__(
        self,
        catalogue: LoopVertexCatalogue,
        colour: ColourModel,
        *,
        classes: Callable[[Skeleton], Callable[[str], Hashable]] | None = None,
        keep: Callable[[Skeleton], bool] | None = None,
    ):
        self.catalogue = catalogue
        self.colour = colour
        self.classes = classes
        self.keep = keep or (lambda skeleton: True)
        self._labels: dict[Content, int] = {}
        self._contents: list[Content] = []

    def content(self, label: int) -> Content:
        return self._contents[label]

    def _segment(self, classes, content: Content) -> Segment:
        if classes is None:
            label = self._labels.get(content)
            if label is None:
                label = self._labels[content] = len(self._contents)
                self._contents.append(content)
        else:
            label = (
                content.colour,
                classes(content.loop_in),
                classes(content.loop_out),
                content.orders,
            )
        mass = self.catalogue.species(content.loop_out).mass
        return Segment(label, content.mask, mass, content.q_degree)

    def view(self, skeleton: Skeleton) -> ViewedSkeleton:
        classes = self.classes(skeleton) if self.classes is not None else None
        contents = skeleton.contents
        count = len(contents)
        forward = [self._segment(classes, c) for c in contents]
        # mirror[i] is position -i read backwards.
        mirror = [
            self._segment(classes, c)
            for c in reflect_ring(contents, self.colour, self.catalogue)
        ]
        readings = []
        for start, reflected in ring_readings(count):
            if reflected:
                readings.append(
                    tuple(mirror[(step - start) % count] for step in range(count))
                )
            else:
                readings.append(
                    tuple(forward[(start + step) % count] for step in range(count))
                )
        return ViewedSkeleton(skeleton, tuple(readings))

    def select(self, skeletons: Iterable[Skeleton]) -> list[ViewedSkeleton]:
        """Kept skeletons, one per distinct ring of the view."""
        seen: dict[Ring, ViewedSkeleton] = {}
        for skeleton in skeletons:
            if self.keep(skeleton):
                viewed = self.view(skeleton)
                seen.setdefault(viewed.canonical, viewed)
        return list(seen.values())


# --------------------------------------------------------------------------- cost


@dataclass(frozen=True, slots=True)
class CostModel:
    """Dressing work per B node in complex MACs (design §5.4).

    rows_out x columns x C(r+4,4) x min(fused * X, split * X + vertex) for an
    input of rank r >= 1; the first step from a scalar seed costs rows_out x
    columns x min(seed_fused * X, seed_split * X + vertex). X is the summed
    block rows of the node's edges. The family closure costs closure x C(R+4,4).
    """

    fused: int = 200
    split: int = 64
    vertex: int = 300
    seed_fused: int = 80
    seed_split: int = 4
    closure: int = 4
    columns: int = 1

    def node(self, rows_out: int, rank_in: int, x_rows: int) -> int:
        if rank_in == 0:
            work = min(self.seed_fused * x_rows, self.seed_split * x_rows + self.vertex)
        else:
            work = comb(rank_in + 4, 4) * min(
                self.fused * x_rows, self.split * x_rows + self.vertex
            )
        return rows_out * self.columns * work

    def close(self, rank: int) -> int:
        return self.closure * comb(rank + 4, 4)


STRUCTURED = CostModel()
DENSE = CostModel(fused=320, vertex=1300)


# --------------------------------------------------------------------------- B tries


@dataclass(frozen=True, slots=True)
class TrieNode:
    depth: int  # positions 1..depth are dressed
    orders: Orders  # loop orders of the skeletons below
    suffix: Ring  # undressed positions depth+1..N
    rank_in: int  # rank entering the segment of position ``depth``
    edges: tuple[tuple[Segment, int], ...]  # (content of position depth, child)


@dataclass(frozen=True, slots=True)
class SuffixTrie:
    """Schedule-B trie of one family in one frame.

    ``nodes`` holds depths 1..N (the depth-N nodes are the family
    accumulators, one per loop order); a child index refers to ``nodes`` for
    depth > 1 and to ``leaves`` (the oriented skeletons) for depth 1.
    """

    family: FamilyKey
    frame: FamilyFrame
    leaves: tuple[tuple[Skeleton, Ring], ...]
    nodes: tuple[TrieNode, ...]
    rank: int

    @property
    def accumulators(self) -> tuple[TrieNode, ...]:
        return tuple(node for node in self.nodes if node.depth == len(self.family))

    def cost(self, rows: Callable[[int], int], model: CostModel = STRUCTURED) -> int:
        total = model.close(self.rank) * len(self.accumulators)
        for node in self.nodes:
            total += model.node(
                _rows(self.suffix_masks(node), rows), node.rank_in, _x_rows(node, rows)
            )
        return total

    @staticmethod
    def suffix_masks(node: TrieNode) -> tuple[int, ...]:
        return tuple(segment.mask for segment in node.suffix)


def _rows(masks: Iterable[int], rows: Callable[[int], int]) -> int:
    out = 1
    for mask in masks:
        out *= rows(mask)
    return out


def _x_rows(node: TrieNode, rows: Callable[[int], int]) -> int:
    return sum(rows(segment.mask) for segment, _child in node.edges)


def build_trie(
    family: Family, frame: FamilyFrame, oriented: Sequence[tuple[Skeleton, Ring]]
) -> SuffixTrie:
    count = family.size
    nodes: list[TrieNode] = []
    # (orders, suffix at depth k-1) -> index (leaf or node)
    previous: dict[tuple[Orders, Ring], int] = {}
    for leaf, (skeleton, ring) in enumerate(oriented):
        previous.setdefault((skeleton.orders, ring), leaf)
    for depth in range(1, count + 1):
        edges: dict[tuple[Orders, Ring], dict[Segment, int]] = defaultdict(dict)
        rank_in: dict[tuple[Orders, Ring], int] = defaultdict(int)
        for skeleton, ring in oriented:
            key = skeleton.orders, ring[depth:]
            edges[key][ring[depth - 1]] = previous[skeleton.orders, ring[depth - 1 :]]
            rank_in[key] = max(
                rank_in[key], sum(segment.q_degree for segment in ring[: depth - 1])
            )
        current = {}
        for key in sorted(edges):
            current[key] = len(nodes)
            nodes.append(
                TrieNode(depth, *key, rank_in[key], tuple(sorted(edges[key].items())))
            )
        previous = current
    rank = max(
        sum(segment.q_degree for segment in ring) for _skeleton, ring in oriented
    )
    return SuffixTrie(family.key, frame, tuple(oriented), tuple(nodes), rank)


def frames(family: Family) -> list[FamilyFrame]:
    """Readings of the family key with distinct (mask, mass) sequences."""
    seen = {}
    for start, reflected in ring_readings(family.size):
        frame = FamilyFrame(start, reflected)
        seen.setdefault(frame.apply(family.key), frame)
    return list(seen.values())


ReadingChoice = Callable[[Sequence[Ring]], Ring]


def shared_suffix(readings: Sequence[Ring]) -> Ring:
    """Prefer the reading whose undressed remainder sorts first.

    Two readings of a skeleton fit a frame only for N <= 2 rings whose mass
    sequence is reflection invariant; a canonical remainder lets the
    skeleton share its trie nodes with the most others.
    """
    return min(readings, key=lambda ring: (ring[1:], ring))


def orient(
    family: Family,
    skeletons: Sequence[ViewedSkeleton],
    choose: ReadingChoice = shared_suffix,
) -> list[tuple[FamilyFrame, list[tuple[Skeleton, Ring]]]]:
    """Every skeleton read in every frame of its family."""
    keyed = {frame.apply(family.key): frame for frame in frames(family)}
    fitting: dict[FamilyKey, list[list[Ring]]] = {
        target: [[] for _ in skeletons] for target in keyed
    }
    for index, viewed in enumerate(skeletons):
        for ring in viewed.readings:
            target = tuple((segment.mask, segment.mass) for segment in ring)
            fitting[target][index].append(ring)
    out = []
    for target, frame in keyed.items():
        readings = fitting[target]
        if not all(readings):
            raise ValueError(f"skeleton does not belong to family {family.key}")
        out.append(
            (
                frame,
                [
                    (viewed.skeleton, choose(rings))
                    for viewed, rings in zip(skeletons, readings, strict=True)
                ],
            )
        )
    return out


FramePolicy = Callable[[Family, Sequence[tuple[FamilyFrame, SuffixTrie]]], SuffixTrie]


def cheapest(rows: Callable[[int], int], model: CostModel = STRUCTURED) -> FramePolicy:
    """Per-family cost-optimal frame (design §4.2: argmin over 2N cuts)."""

    def choose(family, candidates):
        return min(candidates, key=lambda item: item[1].cost(rows, model))[1]

    return choose


def largest_first(
    size: Callable[[int], int] = int.bit_count,
    first: Callable[[int], int] = lambda mask: (mask & -mask).bit_length(),
) -> FramePolicy:
    """Cut before the largest position, then the smallest ``first`` leg, then
    towards the larger neighbour (a cost-free heuristic frame)."""

    def score(family: Family, frame: FamilyFrame):
        masks = [mask for mask, _mass in frame.apply(family.key)]
        sizes = [size(mask) for mask in masks]
        return sizes[0], -first(masks[0]), sizes[1 % len(sizes)], sizes[-1]

    def choose(family, candidates):
        best = max(score(family, frame) for frame, _trie in candidates)
        return next(trie for frame, trie in candidates if score(family, frame) == best)

    return choose


def candidate_tries(
    family: Family,
    skeletons: Sequence[ViewedSkeleton],
    choose: ReadingChoice = shared_suffix,
) -> list[tuple[FamilyFrame, SuffixTrie]]:
    """The B trie of a family in each of its frames."""
    return [
        (frame, build_trie(family, frame, oriented))
        for frame, oriented in orient(family, skeletons, choose)
    ]


def family_trie(
    family: Family,
    skeletons: Sequence[ViewedSkeleton],
    policy: FramePolicy,
    choose: ReadingChoice = shared_suffix,
) -> SuffixTrie:
    return policy(family, candidate_tries(family, skeletons, choose))


# --------------------------------------------------------------------------- F and M


@dataclass(slots=True)
class StateGraph:
    """Prefix states of schedule F or M.

    ``entries`` are the states seeded at the cut and ``starts`` the entry of
    each skeleton; an edge (parent, child, content) dresses one position.
    Closures scatter a state into family accumulators, one entry per skeleton
    (with its weight).
    """

    states: dict[Hashable, int] = field(default_factory=dict)
    edges: set[tuple[int, int, Hashable]] = field(default_factory=set)
    entries: set[int] = field(default_factory=set)
    starts: dict[Skeleton, int] = field(default_factory=dict)
    closures: dict[int, list[Skeleton]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __len__(self) -> int:
        return len(self.states)

    def state(self, key: Hashable) -> int:
        return self.states.setdefault(key, len(self.states))


def forward_graph(skeletons: Iterable[Skeleton]) -> StateGraph:
    """Schedule F: a state is the sequence of contents dressed from the cut.

    Contents carry the cut species and their orders; helicities of attached
    legs are explicit, so prefixes are shared across families but not across
    orderings.
    """
    graph = StateGraph()
    for skeleton in skeletons:
        prefix: tuple = ()
        state = graph.state(prefix)
        graph.entries.add(state)
        graph.starts[skeleton] = state
        for content in skeleton.contents:
            prefix = (*prefix, content.label)
            child = graph.state(prefix)
            graph.edges.add((state, child, content.label))
            state = child
        graph.closures[state].append(skeleton)
    return graph


def merged_graph(skeletons: Iterable[Skeleton]) -> StateGraph:
    """Schedule M: dressed positions keep (leg set, species); the rest is a word.

    The seed of a chain is the full colour word read from the cut. States
    merge across families and seed orderings whenever the dressed leg sets,
    species, prefix orders and the undressed word agree (design §3.5), so a
    state can have several parents.

    Rings of N <= 2 positions merge only among themselves, and each closes
    in its own state. Their F reading is the one whose contents sort first,
    not a leg-set rule: a prefix of a longer ring followed by the remainder
    of a bubble can be the mirror reading of another bubble (counted twice),
    and two bubbles of one family can be read in opposite directions, so
    their closures must not share a frame.
    """
    graph = StateGraph()
    for skeleton in skeletons:
        words = [content.colour for content in skeleton.contents]
        if not all(isinstance(word, tuple) for word in words):
            raise TypeError("schedule M needs colour-ordered (word) contents")
        cut = skeleton.cut
        short = len(words) <= 2
        state = graph.state((cut, short, (), sum(words, ()), ()))
        graph.entries.add(state)
        graph.starts[skeleton] = state
        dressed: tuple = ()
        orders = ()
        for index, content in enumerate(skeleton.contents):
            dressed = (*dressed, (content.mask, content.loop_out))
            orders = add_orders(orders, content.orders)
            remainder = sum(words[index + 1 :], ())
            child = graph.state((cut, short, dressed, remainder, orders))
            graph.edges.add((state, child, content.label))
            state = child
        graph.closures[state].append(skeleton)
    return graph


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True, slots=True)
class FamilySchedule:
    """Reduction-side data of one family (design §3.3 per-family data)."""

    family: Family
    trie: SuffixTrie
    relabelling: Relabelling
    masters: tuple[MasterKey, ...]

    @property
    def bucket(self) -> tuple[int, int]:
        return self.family.size, self.family.rank


@dataclass(frozen=True, slots=True)
class LoopSchedules:
    families: FamilySet
    view: SkeletonView
    schedules: Mapping[FamilyKey, FamilySchedule]

    def buckets(self) -> dict[tuple[int, int], tuple[FamilyKey, ...]]:
        """(N, R) buckets: the families reduced together in one call.

        ``lanes()`` lists the reduction lanes of each bucket.
        """
        out: dict[tuple[int, int], list[FamilyKey]] = defaultdict(list)
        for key, schedule in self.schedules.items():
            out[schedule.bucket].append(key)
        return {bucket: tuple(sorted(keys)) for bucket, keys in sorted(out.items())}

    def lanes(self) -> dict[tuple[int, int], tuple[tuple[FamilyKey, Orders], ...]]:
        """Reduction lanes per (N, R) bucket: one per family and loop order,
        i.e. per accumulator of the family's trie."""
        out: dict[tuple[int, int], list[tuple[FamilyKey, Orders]]] = defaultdict(list)
        for key, schedule in self.schedules.items():
            out[schedule.bucket].extend(
                (key, node.orders) for node in schedule.trie.accumulators
            )
        return {bucket: tuple(sorted(lanes)) for bucket, lanes in sorted(out.items())}

    def master_requests(self) -> dict[MasterKey, int]:
        """Distinct masters and how many families request each per event."""
        out: dict[MasterKey, int] = defaultdict(int)
        for schedule in self.schedules.values():
            for master in schedule.masters:
                out[master] += 1
        return dict(out)


def build_schedules(
    families: FamilySet,
    catalogue: LoopVertexCatalogue,
    colour: ColourModel,
    *,
    policy: FramePolicy,
    view: SkeletonView | None = None,
    choose: ReadingChoice = shared_suffix,
) -> LoopSchedules:
    """Skeletons, B tries in the chosen frames and reduction data per family."""
    view = view or SkeletonView(catalogue, colour)
    out = {}
    for key, family in families.families.items():
        viewed = view.select(family_skeletons(families, family, colour, catalogue))
        if not viewed:
            continue
        trie = family_trie(family, viewed, policy, choose)
        out[key] = FamilySchedule(
            family,
            trie,
            retest_relabelling(key, trie.frame),
            family.master_keys(families.legs),
        )
    return LoopSchedules(families, view, out)


def trie_shapes(
    schedules: LoopSchedules, rows: Callable[[int], int], model: CostModel = STRUCTURED
) -> list[dict[str, object]]:
    """B node shapes for the kernel gate G0 (design §5.4, WP0.3).

    One record per node: loop orders, input rows (undressed legs after the
    step), summed block rows X of its edges, edge count, input rank and
    coefficient count, output rank, loop-line species and modelled work.
    A projected view (``SkeletonView.classes``) has no interned contents, so
    its records carry ``None`` for the species and vertex-template fields.
    """
    projected = schedules.view.classes is not None
    out = []
    for key, schedule in schedules.schedules.items():
        trie = schedule.trie
        for node in trie.nodes:
            rows_out = _rows(trie.suffix_masks(node), rows)
            x_rows = _x_rows(node, rows)
            segments = [segment for segment, _child in node.edges]
            species = {"loop_in": None, "loop_out": None, "vertex_templates": None}
            if not projected:
                contents = [schedules.view.content(s.label) for s in segments]
                species = {
                    "loop_in": sorted({c.loop_in for c in contents}),
                    "loop_out": sorted({c.loop_out for c in contents}),
                    "vertex_templates": len(
                        {term.vertex for c in contents for term in c.terms}
                    ),
                }
            out.append(
                {
                    "family": [[mask, mass.mass, mass.width] for mask, mass in key],
                    "depth": node.depth,
                    "orders": [list(order) for order in node.orders],
                    "positions": len(key),
                    "rows_out": rows_out,
                    "x_rows": x_rows,
                    "edges": len(node.edges),
                    "rank_in": node.rank_in,
                    "coefficients_in": comb(node.rank_in + 4, 4),
                    "rank_out": node.rank_in + max(s.q_degree for s in segments),
                    **species,
                    "work": model.node(rows_out, node.rank_in, x_rows),
                }
            )
    return out
