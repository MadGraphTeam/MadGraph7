# SPDX-License-Identifier: 0BSD
"""Schedule M on generated slices: the planar state graph (design §2.2, §3.5, WP2.8).

Schedule M injects seeds at the cut, one per colour ordering, and shares
dressed prefixes across families. That needs seeds that depend on the
colour ordering only, which certified primitive classes provide (design
§3.4, §7.2): a certified skeleton's seed is

    kappa_s = sum over position terms t of (prod_i c_{t_i}) sign(w_t) P(w_t),

one term per choice of a decomposition term at every position
(``colour.CompressedColour``), with P(w) the canonical primitive of the
choice's ring word w and c_t the exact coefficient of the vertex
decomposition term. This module runs every choice as one colour-ordered
*path*:

* the **entry** of a path is its seed read from the cut, the primitive
  class c of P(w) (``SeedTable`` index, proportional classes merged), with
  the entry weight w_s sign(w) x_c (leaf weight, primitive sign, class
  ratio);
* each **step** dresses one position content and carries the coefficient
  c_t of its term as an exact edge weight (moved towards the entries where
  the graph allows, ``_fold_weights``);
* the path's **tokens** are the letters of its terms (adjoint letters,
  directed fundamental letters, exact quark-ring positions) plus a token for
  the legs of a position outside its letters (colour singlets).

A path reads its skeleton's ring from the cut and in the orientation of
the family's B trie (the cost-optimal reading of design §4.2), so its
closure lands in the trie frame and M dresses the same positions in the
same order as B; the skeleton's own reading is the alternative (closures
then map into the trie frame per family frame map).

States are keyed by (chunk, cut species, short ring, dressed (leg set,
species) positions, remaining tokens, prefix orders, total orders), so two
paths share a state when their dressed positions agree and the rest of their
cut word is the same: prefixes are shared across compositions and families
(design §3.5). Entries add the seed and weight to that key.

The **chunk** of a path is its cut species and its first ``chunk_depth``
dressed positions (``chunk_depth`` 0: one chunk). States of different
chunks never merge, so the executor runs chunks independently and releases
each chunk's states with it (design §5.3). A state's key holds its dressed
positions, so a chunk key of depth 1 loses no merge at all; deeper keys
repeat the shallower (cheap, low-rank) states in every chunk below them.
Every skeleton of a family starts at the family's first position, so the
chunks of a family are consecutive (chunks are ordered by their leading leg
sets): a family is complete, and its accumulator can stream into the
reduction, as soon as its chunks are done.
Within a chunk states are ordered by their dressed positions, a depth-first
topological order (a parent's positions are a prefix of its child's) that
keeps few open-loop objects live.

Correctness is certified, not assumed: every skeleton's choices must sum to
its B seed exactly (``class_of``), and the graph must have exactly one
cut-to-closure path per choice (a merged state continues every prefix with
every completion, so a spurious prefix-completion pair would be counted).
Closing states must land in one family accumulator and one frame map.
Families with a skeleton outside the certified classes (generic seeds) run
schedule B ("generic fallback to B per class").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

from pyamplicol.color.connections import ExactColorCoefficient

from . import reference_executor as rx
from .colour import CompressedColour, _ClassKeys, _ratio, canonical_primitive
from .families import FamilyKey, Orders, add_orders
from .skeletons import Content, Skeleton

_ONE = ExactColorCoefficient(1)


class MergedUnavailable(ValueError):
    """Schedule M does not apply (the families keep schedule B)."""


@dataclass(frozen=True, slots=True)
class PlanarStep:
    """One position of a path: a content, its decomposition term, the term's
    tokens and exact coefficient."""

    content: Content
    term: int
    tokens: tuple[Hashable, ...]
    coefficient: ExactColorCoefficient

    @property
    def label(self) -> tuple[Hashable, int]:
        return self.content.label, self.term


@dataclass(frozen=True, slots=True)
class PlanarPath:
    """One colour-ordered term of a certified skeleton in one ring reading:
    ``relabel`` maps the reading's loop momentum into the family's trie
    frame (None: the reading is the trie's own)."""

    skeleton: Skeleton
    steps: tuple[PlanarStep, ...]
    seed: int
    weight: ExactColorCoefficient
    relabel: Any = None

    @property
    def cut(self) -> str:
        """The species of the line that closes into the cut."""
        return self.steps[-1].content.loop_out

    @property
    def word(self) -> tuple[Hashable, ...]:
        return tuple(token for step in self.steps for token in step.tokens)


class PlanarExpansion:
    """The colour-ordered paths of a slice's certified skeletons."""

    def __init__(self, slice_: Any):
        self.slice = slice_
        self.table = slice_.table
        self.compressed = CompressedColour(slice_.colour)
        self._keys = _ClassKeys(self.table.basis.legs, merge=True)
        self._classes: dict[Hashable, tuple[int, ExactColorCoefficient]] = {}
        for c, seed in enumerate(self.table.classes):
            key, scale = self._keys(seed)
            self._classes.setdefault(key, (c, scale))

    def _class(self, primitive) -> tuple[int, ExactColorCoefficient] | None:
        """(class, x) with primitive = x kappa_class, if the table has it."""

        key, scale = self._keys(primitive)
        found = self._classes.get(key)
        if found is None:
            return None
        c, class_scale = found
        return c, _ratio(scale, class_scale)

    def paths(
        self, skeleton: Skeleton, contents: Sequence[Content] | None = None
    ) -> tuple[PlanarPath, ...]:
        """Every term choice of a certified skeleton, read as ``contents``
        (one of its ring readings; default its own), as one path.

        Raises ``MergedUnavailable`` for a skeleton outside the certified
        classes or whose choices do not sum to its B seed exactly.
        """

        contents = skeleton.contents if contents is None else tuple(contents)
        reading = rx._reading(skeleton, contents)
        compressed = self.compressed
        kind = compressed.kind(reading)
        if kind is None or kind not in self.table.certificates:
            raise MergedUnavailable("generic seed (no certified primitive class)")
        combination = self.slice.class_of.get(reading)
        if combination is None or not self.table.certified(combination):
            raise MergedUnavailable("the B seed is not a certified combination")
        # Per-propagator factors of the leaf weight ride on the steps, so
        # compositions of one cut word share their entry weight.
        weights = self.slice.weights
        weight = weights(skeleton, contents)
        signs = 1
        options = []
        for content in contents:
            sign = weights.propagator_sign(content)
            signs *= sign
            _position, _single, terms = compressed._content(content)
            options.append(
                [
                    PlanarStep(
                        content,
                        t,
                        _tokens(content, descriptor),
                        compressed._coefficient((ids,)).scaled(sign),
                    )
                    for t, (ids, descriptor) in enumerate(terms)
                ]
            )
        weight = ExactColorCoefficient(weight * signs)
        out = []
        total: dict[int, ExactColorCoefficient] = defaultdict(ExactColorCoefficient)
        for steps in product(*options):
            descriptors = tuple(
                compressed._content(step.content)[2][step.term][1] for step in steps
            )
            if kind == "quark-ring":
                primitive, sign = canonical_primitive(kind, descriptors)
            else:
                word = compressed._ring_word(kind, descriptors)
                if word is None:
                    raise MergedUnavailable("a term choice closes no single trace")
                primitive, sign = canonical_primitive(kind, word)
            found = self._class(primitive)
            if found is None:
                raise MergedUnavailable("a primitive outside the seed table")
            c, x = found
            x = x.scaled(sign)
            coefficient = _ONE
            for step in steps:
                coefficient = coefficient * step.coefficient
            total[c] = total[c] + x * coefficient.scaled(signs)
            out.append(PlanarPath(skeleton, tuple(steps), c, weight * x))
        if {c: v for c, v in total.items() if v} != dict(combination):
            raise MergedUnavailable("the planar terms do not sum to the B seed")
        return tuple(out)


def _tokens(content: Content, descriptor: tuple) -> tuple[Hashable, ...]:
    """Colour tokens of one position term (module docstring)."""

    kind = descriptor[0]
    if kind == "exact":
        return (("exact", content.mask, descriptor[1]),)
    letters = descriptor[1]
    if kind == "adjoint":
        tokens: tuple[Hashable, ...] = tuple(letters)
    elif descriptor[2]:
        # A forward string is (tau^w)_{out, in}: along the loop, later
        # positions multiply from the left, so the cut word reads each
        # position's letters reversed.
        tokens = tuple(("f", letter) for letter in reversed(letters))
    else:
        tokens = tuple(("b", letter) for letter in letters)
    covered = 0
    for letter in letters:
        covered |= 1 << (letter - 1)
    if covered & ~content.mask:
        raise MergedUnavailable("a position letter outside its leg set")
    rest = content.mask & ~covered
    return (*tokens, ("rest", rest)) if rest else tokens


# --------------------------------------------------------------------- graph


@dataclass
class MergedState:
    """One dressed state: edges from entries or earlier states (source, the
    step it dresses, exact weight), and the closure of a full ring."""

    key: Hashable
    depth: int
    shift: int
    out_mask: int
    edges: list[tuple[int, PlanarStep, ExactColorCoefficient]] = field(
        default_factory=list
    )
    # (family, loop orders, frame map into the trie frame) of a full ring.
    closure: tuple[FamilyKey, Orders, Any] | None = None


@dataclass
class MergedGraph:
    """Schedule M of a slice's certified families.

    ``entries`` are (seed key, entry weight, first content) per entry; state
    sources are ``-1 - entry`` for entries and the state index otherwise.
    States are ordered chunk by chunk, depth-first within a chunk.
    """

    families: frozenset[FamilyKey]
    paths: int
    entries: list[tuple[int, Orders, ExactColorCoefficient, Content]]
    states: list[MergedState]
    chunks: list[int]
    fallback: dict[FamilyKey, str]

    def counts(self) -> dict[str, int]:
        return {
            "families": len(self.families),
            "paths": self.paths,
            "entries": len(self.entries),
            "states": len(self.states),
            "edges": sum(len(state.edges) for state in self.states),
            "chunks": len(self.chunks),
        }


def merged_graph(
    slice_: Any,
    *,
    chunk_depth: int = 1,
    reading: Literal["trie", "skeleton"] = "trie",
    debug: dict | None = None,
) -> MergedGraph:
    """The planar state graph of every family whose skeletons are certified.

    ``reading`` picks the ring reading a skeleton's paths start from: the
    reading of its family's B trie (the cost-optimal cut and orientation of
    design §4.2, so closures land in the trie frame), or the skeleton's own
    reading (closures mapped into the trie frame per family frame map).

    Raises ``MergedUnavailable`` when no family qualifies or the graph fails
    its path certificate (module docstring).
    """

    plan = slice_.plan
    expansion = PlanarExpansion(slice_)
    targets = rx.trie_targets(plan)
    paths: list[PlanarPath] = []
    fallback: dict[FamilyKey, str] = {}
    families = set()
    for family, schedule in plan.schedules.items():
        found = []
        try:
            for skeleton, ring in schedule.trie.leaves:
                if reading == "trie":
                    contents = [plan.view.content(segment.label) for segment in ring]
                    relabel = None
                else:
                    contents, relabel = skeleton.contents, targets[skeleton][1]
                found += [
                    replace(path, relabel=relabel)
                    for path in expansion.paths(skeleton, contents)
                ]
        except MergedUnavailable as error:
            fallback[family] = str(error)
            continue
        families.add(family)
        paths.extend(found)
    if not paths:
        raise MergedUnavailable("no family has certified colour-ordered paths")
    entries: dict[Hashable, int] = {}
    entry_records: list[tuple[int, Orders, ExactColorCoefficient, Content]] = []
    index: dict[Hashable, int] = {}
    states: list[MergedState] = []
    edges: set[tuple[int, int, Hashable]] = set()
    for path in paths:
        skeleton = path.skeleton
        word = path.word
        cut = path.cut
        head = path.steps[:chunk_depth]
        chunk = (
            (
                tuple(step.content.mask for step in head),
                cut,
                tuple(step.content.loop_out for step in head),
            )
            if chunk_depth
            else ((),)
        )
        short = len(path.steps) <= 2
        total = skeleton.orders
        entry_key = (chunk, cut, short, word, total, path.seed, path.weight)
        source = entries.get(entry_key)
        if source is None:
            source = entries[entry_key] = len(entry_records)
            entry_records.append((path.seed, total, path.weight, path.steps[0].content))
        source = -1 - source
        dressed: tuple = ()
        orders: Orders = ()
        mask = 0
        remaining = len(word)
        for depth, step in enumerate(path.steps, start=1):
            content = step.content
            shift = mask
            mask |= content.mask
            dressed = (*dressed, (content.mask, content.loop_out))
            orders = add_orders(orders, content.orders)
            remaining -= len(step.tokens)
            rest = word[len(word) - remaining :] if remaining else ()
            key = (chunk, cut, short, dressed, rest, orders, total)
            state = index.get(key)
            if state is None:
                state = index[key] = len(states)
                states.append(MergedState(key, depth, shift, 0))
            if states[state].shift != shift:
                raise AssertionError("an M state with two dressing shifts")
            if (source, state, step.label) not in edges:
                edges.add((source, state, step.label))
                states[state].edges.append((source, step, step.coefficient))
            source = state
        closing = states[source]
        target = (skeleton.family, skeleton.orders, path.relabel)
        if closing.closure is None:
            closing.closure = target
        elif closing.closure != target:
            raise MergedUnavailable("an M closure merges two family frames")
    full = (1 << len(slice_.blocks.process.legs)) - 1
    for state in states:
        dressed = state.key[3]
        mask = 0
        for position_mask, _species in dressed:
            mask |= position_mask
        state.out_mask = full ^ mask

    def position(s: int) -> tuple:
        chunk, _cut, short, dressed, rest, orders, total = states[s].key
        return (chunk[0], repr(chunk[1:]), dressed, repr((rest, orders, total, short)))

    order = sorted(range(len(states)), key=position)
    new = {old: k for k, old in enumerate(order)}
    ordered = []
    chunks = []
    for k, old in enumerate(order):
        state = states[old]
        state.edges = [
            (source if source < 0 else new[source], step, weight)
            for source, step, weight in state.edges
        ]
        if k and state.key[0] != ordered[-1].key[0]:
            chunks.append(k)
        ordered.append(state)
    chunks.append(len(ordered))
    if debug is not None:
        debug.update(states=ordered, paths=paths, entries=entry_records)
    _certify(ordered, len(paths))
    _fold_weights(ordered)
    return MergedGraph(
        frozenset(families), len(paths), entry_records, ordered, chunks, fallback
    )


def _certify(states: Sequence[MergedState], paths: int) -> None:
    """Exactly one cut-to-closure path per planar path (module docstring)."""

    counts: list[int] = []
    for state in states:
        counts.append(
            sum(1 if source < 0 else counts[source] for source, *_ in state.edges)
        )
    closing = sum(
        count for count, state in zip(counts, states, strict=True) if state.closure
    )
    if closing != paths:
        raise MergedUnavailable(
            f"the merged graph has {closing} closing paths for {paths} planar paths"
        )
    for state in states:
        if state.closure is not None and state.out_mask:
            raise AssertionError("an M closure with undressed legs")


def _fold_weights(states: Sequence[MergedState]) -> None:
    """Move edge weights towards the entries, where they cost nothing.

    Dressing is linear, so the weight of an edge whose source state has no
    other reader can multiply every edge into that source instead. Walking
    the states backwards folds a whole chain into the edges that read
    entries (whose weights the executor multiplies into the entry weight);
    only edges out of shared states keep a weight, applied to their input.
    """

    readers = [0] * len(states)
    for state in states:
        for source, *_ in state.edges:
            if source >= 0:
                readers[source] += 1
    for state in reversed(states):
        for e, (source, step, weight) in enumerate(state.edges):
            if source < 0 or readers[source] != 1 or weight == _ONE:
                continue
            parent = states[source]
            parent.edges = [(s, t, w * weight) for s, t, w in parent.edges]
            state.edges[e] = (source, step, _ONE)
