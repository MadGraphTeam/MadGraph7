# SPDX-License-Identifier: 0BSD
"""Generic colour seeds of the loop engine (design §3.4, §7.1-§7.5, WP1.4).

Colour lives only in exact seed tables. Every skeleton has a *seed class*
kappa_s, the colour vector obtained by closing its colour flow: the product of
the vertex colour tensors of its positions and of the colour-flow strings of
its block words, contracted around the loop. kappa_s is a sum of external
colour structures (open strings, traces, powers of N_c), so it holds for any
model whose coloured vertices lower to colour flow; no Feynman diagram and no
decomposition theorem is used. The seed table is

    K[c, b] = <b | kappa_c>,

an exact Laurent polynomial in N_c for every seed class c and Born colour basis
element b of the tree engine. Loop colour is never decomposed into the Born
basis, so no Gram matrix is inverted (design §7.1).

Conventions a consumer relies on:

* **Generators.** Colour is written with tau = sqrt(2) t, tr(tau^a tau^b) =
  delta^ab, and SU(N_c) Fierz for every contracted adjoint index, so the U(1)
  pieces of internal gluons are explicit -1/N_c terms. Colour structures are
  the tree engine's literal tensors (``color.connections``): an open string
  ``OpenColorString(q, (a_1..a_k), qbar)`` is the block-word string
  ``q -> a_1 .. a_k -> qbar`` of ``loops.v2.blocks`` and a trace lists its
  word in the same order. Born basis elements are ``sector_color_tensor`` of
  the Born's full-colour plan (with its endpoint-pairing phase), so the Born
  colour vector is sum_b A0[b] |b>, the vector whose Gram matrix is the tree
  engine's Born metric.
* **Contents.** In this generic mode a content is one routing term in one
  colour state ``PositionColour``: the vertex rule, one of its colour
  structures, the UFO legs the loop and the blocks occupy, and one block word
  per attached block current. Its kinematics is the sum of the vertex terms of
  that colour structure on those legs (couplings times the colour-flow
  normalisation 2^(-p/2) of ``loops.v2.catalogue``) times the block currents of
  those words. Seeds absorb the matching 2^(p/2), so kappa carries no sqrt(2).
* **Signs.** kappa uses the plain colour-flow strings of the blocks.
  ``Block.wick_sign``, the skeleton's fermion sign epsilon_s, its automorphism
  weight and its i-factors belong to the leaf weight (design §3.3), never to K.
* **Projection** (design §7.4). P is the largest N_c degree of a nonzero cell
  of a squared-order sector. LC keeps degree P, NLC keeps P, P-1 and P-2 and FC
  keeps every power; N_c = 3 is substituted after projection and the external
  average after that.
* **Leading colour** (design §7.5). The face degree of a class is the U(N_c)
  index-loop count of its colour flow glued to the Born basis (integer cycle
  counting, no numerics). ``certify_leading_colour`` proves for a sector that
  the face rule keeps exactly the classes whose exact degree is P. It runs on
  the full seed table, after generation; nothing prunes skeletons before
  numerics yet. A generic content keeps its vertex's full UFO colour, so LC
  drops whole seed classes only (in QCD the quark loops); dropping gluon
  contents structurally needs colour-ordered contents (WP2.4).

LC and NLC need N_c-exact blocks: a process whose tree program uses numeric
U(1)-subtraction currents is refused (design §3.2 item 4); FC is always
available.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from itertools import count, product
from typing import Any, Literal

from pyamplicol.color.connections import (
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    OpenColorString,
    TensorTerm,
    contract_connected_tensor_nc_terms,
    overlap_traces,
)
from pyamplicol.color.contraction_trace import (
    combine_nc_power_terms,
    trace_duplicate_simplify_nc_power,
    trace_pair_simplify_nc_power,
)
from pyamplicol.color.contraction_types import NC
from pyamplicol.color.correlator_matrices import (
    process_color_legs,
    sector_color_tensor,
    selected_color_sectors,
)

from .blocks import Block, BlockColourFlow, TreeBlocks, _NativeFields
from .families import Orders, Position, _model_data, add_orders
from .skeletons import Skeleton

Label = Hashable
Laurent = Mapping[int, ExactColorCoefficient]
ColourAccuracy = Literal["lc", "nlc", "full"]
# (Born amplitude orders, loop amplitude orders): one pair of a squared-order sector.
SectorPair = tuple[Orders, Orders]

_ONE = ExactColorCoefficient(1)
_U1_SUBTRACTION = "u1-subtraction-color-flow-vector"

# --------------------------------------------------------------------- networks


@dataclass(frozen=True, slots=True)
class ColourString:
    """``(tau^word[0] ... tau^word[-1])`` with upper index ``row`` and lower
    index ``col`` (Spenso's orientation: the generator row is its ``cof`` slot)."""

    row: Label
    word: tuple[Label, ...]
    col: Label


@dataclass(frozen=True, slots=True)
class ColourProduct:
    """One product of strings and traces of tau generators.

    Labels occurring twice are contracted; ``root2_power`` counts factors of
    1/sqrt(2).
    """

    coefficient: ExactColorCoefficient = _ONE
    strings: tuple[ColourString, ...] = ()
    traces: tuple[tuple[Label, ...], ...] = ()
    nc_power: int = 0
    root2_power: int = 0

    def __mul__(self, other: ColourProduct) -> ColourProduct:
        return ColourProduct(
            self.coefficient * other.coefficient,
            self.strings + other.strings,
            self.traces + other.traces,
            self.nc_power + other.nc_power,
            self.root2_power + other.root2_power,
        )

    def relabel(self, rename: Callable[[Label], Label]) -> ColourProduct:
        return ColourProduct(
            self.coefficient,
            tuple(
                ColourString(rename(s.row), tuple(map(rename, s.word)), rename(s.col))
                for s in self.strings
            ),
            tuple(tuple(map(rename, trace)) for trace in self.traces),
            self.nc_power,
            self.root2_power,
        )


ColourNetwork = tuple[ColourProduct, ...]  # a sum of products


def multiply(*networks: ColourNetwork) -> ColourNetwork:
    """The expanded product of colour networks."""

    out: ColourNetwork = (ColourProduct(),)
    for network in networks:
        out = tuple(left * right for left in out for right in network)
    return out


def relabel(network: ColourNetwork, rename: Callable[[Label], Label]) -> ColourNetwork:
    return tuple(term.relabel(rename) for term in network)


# ------------------------------------------------------------------- Spenso input


def _exact_number(value: Any) -> ExactColorCoefficient:
    from symbolica import Expression

    real = ((value + value.conj()) / 2).expand()
    imag = ((value - value.conj()) / (2 * Expression.I)).expand()
    return ExactColorCoefficient(
        Fraction(real.to_canonical_string()), Fraction(imag.to_canonical_string())
    )


def _head(atom: Any) -> str:
    return atom.get_name().rsplit("::", 1)[-1]


def _index(atom: Any) -> tuple[str, str]:
    """(kind, name) of a Spenso slot: kind is 'adjoint', 'upper' or 'lower'."""

    from symbolica import AtomType

    head = _head(atom)
    if head == "dind":
        (inner,) = list(atom)
        kind, name = _index(inner)
        if kind != "upper":
            raise NotImplementedError(f"unsupported dual colour slot {atom}")
        return "lower", name
    if head not in ("coad", "cof"):
        raise NotImplementedError(f"unsupported colour slot {atom}")
    dimension, name = list(atom)
    if str(dimension) != ("8" if head == "coad" else "3"):
        raise NotImplementedError(f"colour slot {atom} is not an SU(3) slot")
    label = (
        name.get_name()
        if name.get_type() == AtomType.Var
        else name.to_canonical_string()
    )
    return ("adjoint" if head == "coad" else "upper"), label


def _structure_constant(a: Label, b: Label, c: Label) -> ColourNetwork:
    """f^abc = -(i/sqrt 2) [tr(tau^a tau^b tau^c) - tr(tau^a tau^c tau^b)]."""

    minus_i = ExactColorCoefficient(0, -1)
    return (
        ColourProduct(minus_i, traces=((a, b, c),), root2_power=1),
        ColourProduct(minus_i.scaled(-1), traces=((a, c, b),), root2_power=1),
    )


def _factor_network(atom: Any) -> ColourNetwork:
    from symbolica import AtomType

    kind = atom.get_type()
    if kind == AtomType.Num:
        return (ColourProduct(_exact_number(atom)),)
    if kind != AtomType.Fn:
        raise NotImplementedError(f"unsupported colour factor {atom}")
    head = _head(atom)
    arguments = list(atom)
    if head == "f":
        labels = []
        for argument in arguments:
            slot, name = _index(argument)
            if slot != "adjoint":
                raise NotImplementedError(f"f with a non-adjoint slot: {atom}")
            labels.append(name)
        return _structure_constant(*labels)
    if head == "t":
        (adjoint, a), (upper, row), (lower, col) = map(_index, arguments)
        if (adjoint, upper, lower) != ("adjoint", "upper", "lower"):
            raise NotImplementedError(f"unsupported generator slots: {atom}")
        return (ColourProduct(strings=(ColourString(row, (a,), col),), root2_power=1),)
    if head == "chain":
        (upper, row), (lower, col) = map(_index, arguments[:2])
        if (upper, lower) != ("upper", "lower"):
            raise NotImplementedError(f"unsupported chain ends: {atom}")
        word = []
        for matrix in arguments[2:]:
            if _head(matrix) != "t":
                raise NotImplementedError(f"unsupported chain matrix: {matrix}")
            slot, a = _index(next(iter(matrix)))
            if slot != "adjoint":
                raise NotImplementedError(f"unsupported chain matrix: {matrix}")
            word.append(a)
        return (
            ColourProduct(
                strings=(ColourString(row, tuple(word), col),), root2_power=len(word)
            ),
        )
    if head in ("g", "id"):
        (left_kind, left), (right_kind, right) = map(_index, arguments)
        if left_kind == right_kind == "adjoint":
            # delta^ab = tr(tau^a tau^b)
            return (ColourProduct(traces=((left, right),)),)
        slots = {left_kind: left, right_kind: right}
        if set(slots) != {"upper", "lower"}:
            raise NotImplementedError(f"unsupported colour metric {atom}")
        return (
            ColourProduct(strings=(ColourString(slots["upper"], (), slots["lower"]),)),
        )
    raise NotImplementedError(
        f"colour tensor {head!r} does not lower to colour flow: {atom}"
    )


def spenso_colour_network(expression: Any) -> ColourNetwork:
    """Lower a Spenso SU(3) colour expression to tau strings and traces.

    Accepts sums of products of numbers, ``f``, ``t``, ``chain`` and colour
    metrics, i.e. the ``color_expression`` of compiled vertex terms and the
    legacy engine's colour tensors. Index labels are the slot names.
    """

    from symbolica import AtomType, E, Expression

    if not isinstance(expression, Expression):
        expression = E(str(expression))
    expression = expression.expand()
    terms = list(expression) if expression.get_type() == AtomType.Add else [expression]
    out: list[ColourProduct] = []
    for term in terms:
        factors = list(term) if term.get_type() == AtomType.Mul else [term]
        out.extend(multiply(*(_factor_network(factor) for factor in factors)))
    return tuple(out)


# ------------------------------------------------------------------- reduction


def _join_strings(
    strings: Iterable[ColourString], internal: Callable[[Label], bool]
) -> tuple[list[ColourString], list[tuple[Label, ...]], int]:
    """Contract internal fundamental indices: open strings, traces, empty loops."""

    by_row: dict[Label, ColourString] = {}
    for string in strings:
        if string.row in by_row:
            raise ValueError(f"fundamental index {string.row!r} occurs twice as row")
        by_row[string.row] = string
    columns = [string.col for string in by_row.values()]
    if len(set(columns)) != len(columns):
        raise ValueError("a fundamental index occurs twice as column")
    starts = [row for row in by_row if not internal(row)]
    used: set[Label] = set()
    opened = []
    for start in starts:
        word: list[Label] = []
        current = by_row[start]
        used.add(start)
        while internal(current.col):
            word.extend(current.word)
            following = by_row.get(current.col)
            if following is None:
                raise ValueError(f"fundamental index {current.col!r} is not contracted")
            used.add(current.col)
            current = following
        word.extend(current.word)
        opened.append(ColourString(start, tuple(word), current.col))
    traces, empty = [], 0
    for row in list(by_row):
        if row in used:
            continue
        word = []
        current = row
        while current not in used:
            used.add(current)
            string = by_row[current]
            if not internal(string.col):
                raise ValueError(f"fundamental index {current!r} is not contracted")
            word.extend(string.word)
            current = string.col
        if current != row:
            raise ValueError("fundamental indices do not close into a loop")
        if word:
            traces.append(tuple(word))
        else:
            empty += 1
    return opened, traces, empty


@lru_cache(maxsize=1 << 16)
def _fierz(
    traces: tuple[tuple[int, ...], ...], sentinels: frozenset[int]
) -> tuple[tuple[Fraction, int, tuple[tuple[int, ...], ...]], ...]:
    """Remove every repeated adjoint label with SU(N_c) Fierz identities.

    Open strings ride along as traces closed by one matrix-unit ``sentinel``
    each, so the tree engine's trace rules apply unchanged; a trace of one
    sentinel is a Kronecker delta, not a vanishing tr(tau).
    """

    def trivial(terms):
        out = []
        for coefficient, power, rows in terms:
            if any(len(row) == 1 and row[0] not in sentinels for row in rows):
                continue  # tr(tau^a) = 0
            out.append(
                (
                    coefficient,
                    power + sum(not row for row in rows),
                    tuple(row for row in rows if row),
                )
            )
        return combine_nc_power_terms(out)

    terms = trivial([(Fraction(1), 0, traces)])
    while True:
        terms, paired = trace_pair_simplify_nc_power(terms)
        terms = trivial(terms)
        terms, duplicated = trace_duplicate_simplify_nc_power(terms)
        terms = trivial(terms)
        if not paired and not duplicated:
            return tuple(terms)


class _Numbering(dict):
    """Integer labels for the trace algebra; external legs keep theirs."""

    def __init__(self, external: Mapping[Label, int], first: int):
        super().__init__(external)
        self._next = count(first)

    def fresh(self) -> int:
        return next(self._next)

    def __missing__(self, label: Label) -> int:
        value = self[label] = self.fresh()
        return value


def reduce_network(
    network: ColourNetwork, legs: Sequence[ColorLeg], external: Mapping[Label, int]
) -> ConnectedColorTensor:
    """Contract every internal index of a colour network.

    ``external`` maps the network's external labels to process leg labels;
    every other label must be contracted. The result is in the tree engine's
    literal colour structures (module docstring).
    """

    def internal(label: Label) -> bool:
        return label not in external

    first = len(external) + 1 + max(external.values(), default=0)
    reduced: list[tuple[TensorTerm, int]] = []
    for term in network:
        if not term.coefficient:
            continue
        opened, joined, empty = _join_strings(term.strings, internal)
        number = _Numbering(external, first)
        traces = [
            tuple(number[label] for label in trace) for trace in (*term.traces, *joined)
        ]
        sentinels: dict[int, tuple[int, int]] = {}
        for string in opened:
            sentinel = number.fresh()
            sentinels[sentinel] = (external[string.row], external[string.col])
            traces.append((*(number[label] for label in string.word), sentinel))
        for coefficient, power, rows in _fierz(tuple(traces), frozenset(sentinels)):
            tensor = _decode(rows, sentinels, set(external.values()))
            reduced.append(
                (
                    TensorTerm(
                        term.coefficient.scaled(coefficient),
                        tensor,
                        term.nc_power + empty + power,
                    ),
                    term.root2_power,
                )
            )
    parities = {power % 2 for _term, power in reduced}
    if len(parities) > 1:
        raise ValueError("colour network mixes rational and sqrt(2) coefficients")
    target = parities.pop() if parities else 0
    return ConnectedColorTensor(
        tuple(legs),
        tuple(
            TensorTerm(
                term.coefficient.scaled(Fraction(2) ** ((target - power) // 2)),
                term.tensor,
                term.nc_power,
            )
            for term, power in reduced
        ),
        target,
    )


def _decode(
    rows: tuple[tuple[int, ...], ...],
    sentinel_ends: Mapping[int, tuple[int, int]],
    labels: set[int],
) -> ColorTensor:
    """Fierz output to literal structures (orientation of the module docstring)."""

    strings, traces = [], []
    for row in rows:
        positions = [i for i, label in enumerate(row) if label in sentinel_ends]
        if any(label not in labels and label not in sentinel_ends for label in row):
            raise ValueError("an internal adjoint index survived the Fierz reduction")
        if not positions:
            traces.append(tuple(reversed(row)))
            continue
        # tr(E_1 W_1 E_2 W_2 ...) with E_k = |col_k><row_k| is the product
        # of the strings (W_k)^{row_k}_{col_{k+1}}.
        rotated = row[positions[0] :] + row[: positions[0]]
        marks = [i for i, label in enumerate(rotated) if label in sentinel_ends]
        for index, start in enumerate(marks):
            stop = marks[index + 1] if index + 1 < len(marks) else len(rotated)
            upper, _ = sentinel_ends[rotated[start]]
            _, lower = sentinel_ends[rotated[stop % len(rotated)]]
            word = rotated[start + 1 : stop]
            strings.append(OpenColorString(lower, tuple(reversed(word)), upper))
    return ColorTensor(tuple(strings), tuple(traces))


# ------------------------------------------------------------------ seed classes


@dataclass(frozen=True, slots=True, order=True)
class PositionColour:
    """Colour state of one routing term of a position (generic mode).

    ``legs`` are the 1-based UFO legs of the vertex rule taken by the loop
    line entering and leaving the position and by each attached block, in the
    order of ``Position.blocks``; the kinematic segment uses the same legs.
    ``words`` holds the block word of each attached block current.
    """

    rule: str
    colour_index: int
    legs: tuple[int, ...]
    words: tuple[tuple[int, ...], ...]


def block_colour_network(
    flow: BlockColourFlow,
    representations: Mapping[int, int],
    field_representation: int,
    open_label: Label,
) -> ColourNetwork:
    """The colour of one block word (``loops.v2.blocks`` convention).

    A string s -> a_1 .. a_k -> e is (tau^a_k .. tau^a_1)^e_s; an adjoint field
    closes its open ends with (tau^X)^in_out. ``open_label`` names the field's
    colour index (X, or the open fundamental end).
    """

    incoming = open_label if field_representation == -3 else (open_label, "in")
    outgoing = open_label if field_representation == 3 else (open_label, "out")
    strings = []
    for labels in flow.strings:
        start = labels[0] if representations[labels[0]] == 3 else incoming
        end = labels[-1] if representations[labels[-1]] == -3 else outgoing
        adjoints = tuple(label for label in labels if representations[label] == 8)
        strings.append(ColourString(end, tuple(reversed(adjoints)), start))
    if field_representation == 8:
        strings.append(ColourString(incoming, (open_label,), outgoing))
    elif field_representation not in (1, 3, -3):
        raise NotImplementedError(f"block field colour {field_representation}")
    return (ColourProduct(strings=tuple(strings)),)


_UFO_SLOT = re.compile(r"ufo_c_(\d+)$")


def _vertex_label(name: str) -> Label:
    match = _UFO_SLOT.search(name)
    return ("leg", int(match.group(1))) if match else ("dummy", name)


class SeedColour:
    """Generic seed classes of skeleton contents (a ``skeletons.ColourModel``).

    ``blocks`` are the loop request's tree blocks and Born; ``native_model`` is
    the native model (or its JSON mapping) whose field and rule names the
    families use; ``vertex_terms`` are the compiled model's vertex terms
    (``CompiledModel.ir.vertex_terms``), whose Spenso colour expressions and
    colour-flow normalisation powers fix the colour of every vertex rule.
    """

    def __init__(
        self, blocks: TreeBlocks, native_model: Any, vertex_terms: Iterable[Any]
    ):
        data = _model_data(native_model)
        self._fields = _NativeFields.of(data)
        self._rules = {rule["name"]: rule for rule in data["vertex_rules"]}
        self.legs = process_color_legs(blocks.process)
        self._representations = {leg.label: leg.representation for leg in self.legs}
        self.blocks = blocks
        # (rule, orders) -> colour indices; (rule, colour index) -> colour.
        self._indices: dict[tuple[str, Orders], set[int]] = {}
        self._vertex: dict[tuple[str, int], ColourNetwork] = {}
        for term in vertex_terms:
            rule = self._rules.get(term.vertex)
            if rule is None:
                continue
            if tuple(rule["particles"]) != tuple(term.particles):
                raise ValueError(f"vertex {term.vertex} has different particle orders")
            orders = add_orders(tuple(term.coupling_orders))
            self._indices.setdefault((term.vertex, orders), set()).add(term.color_index)
            key = (term.vertex, term.color_index)
            if key in self._vertex:
                continue
            network = relabel(
                spenso_colour_network(term.color_expression),
                lambda name: _vertex_label(str(name)),
            )
            power = int(term.lc_color_normalization_power)
            # Seeds absorb the 2^(p/2) the kernels drop (module docstring).
            self._vertex[key] = tuple(
                ColourProduct(
                    p.coefficient,
                    p.strings,
                    p.traces,
                    p.nc_power,
                    p.root2_power - power,
                )
                for p in network
            )
        self._words: dict[tuple[int, str, Orders], dict[tuple[int, ...], Block]] = {}
        for block in blocks.blocks:
            key = block.key
            if key.auxiliary_kind is not None:
                continue
            field_name = self._fields.incoming(key.particle_id)["name"]
            self._words.setdefault(
                (key.external_mask, field_name, key.coupling_orders), {}
            ).setdefault(key.word, block)

    # -- ColourModel ---------------------------------------------------------

    def states(self, position: Position) -> tuple[PositionColour, ...]:
        vertex = position.vertex
        indices = self._indices.get((vertex.rule, add_orders(vertex.orders)))
        if not indices:
            raise ValueError(f"no colour structure of {vertex.rule} at {vertex.orders}")
        legs = self._canonical_legs(
            vertex.rule,
            vertex.loop_in,
            self._fields.by_name[vertex.loop_out]["antiname"],
            tuple(block.field for block in position.blocks),
        )
        options = [tuple(sorted(self._block_words(block))) for block in position.blocks]
        return tuple(
            PositionColour(vertex.rule, index, legs, words)
            for index in sorted(indices)
            for words in product(*options)
        )

    def reflected(self, state: PositionColour) -> PositionColour:
        particles = self._rules[state.rule]["particles"]
        in_leg, out_leg, *attached = state.legs
        legs = self._canonical_legs(
            state.rule,
            particles[out_leg - 1],
            particles[in_leg - 1],
            tuple(particles[leg - 1] for leg in attached),
        )
        return PositionColour(state.rule, state.colour_index, legs, state.words)

    # -- seeds ---------------------------------------------------------------

    def seed(self, skeleton: Skeleton) -> ConnectedColorTensor:
        """kappa_s: the closed colour flow of a skeleton (module docstring)."""

        count_ = len(skeleton.contents)
        factors: list[ColourNetwork] = []
        for position, content in enumerate(skeleton.contents):
            state = content.colour
            if not isinstance(state, PositionColour):
                raise TypeError("seed classes need PositionColour contents")
            term = content.terms[0]
            in_leg, out_leg, *attached = state.legs
            slots: dict[int, Label] = {
                in_leg: ("loop", (position - 1) % count_),
                out_leg: ("loop", position),
            }
            slots.update(
                (leg, ("block", position, j)) for j, leg in enumerate(attached)
            )

            def rename(label: Label, position=position, slots=slots) -> Label:
                kind, value = label
                return slots[value] if kind == "leg" else ("dummy", position, value)

            factors.append(
                relabel(self._vertex[(state.rule, state.colour_index)], rename)
            )
            for j, (current, word) in enumerate(
                zip(term.blocks, state.words, strict=True)
            ):
                block = self._words[(current.mask, current.field, current.orders)][word]
                factors.append(
                    block_colour_network(
                        block.colour_flow,
                        self._representations,
                        int(self._fields.by_pdg[block.key.particle_id]["color"]),
                        ("block", position, j),
                    )
                )
        external = {label: label for label in self._representations}
        return canonical_seed(reduce_network(multiply(*factors), self.legs, external))

    # -- helpers -------------------------------------------------------------

    def _block_words(self, block) -> tuple[tuple[int, ...], ...]:
        words = self._words.get((block.mask, block.field, block.orders))
        if not words:
            raise ValueError(f"no tree block for {block}")
        return tuple(words)

    def _canonical_legs(
        self, rule: str, loop_in: str, out_particle: str, attached: tuple[str, ...]
    ) -> tuple[int, ...]:
        """First free UFO leg of each role: loop in, loop out, then blocks."""

        particles = self._rules[rule]["particles"]
        free = list(range(1, len(particles) + 1))
        legs = []
        for name in (loop_in, out_particle, *attached):
            leg = next((leg for leg in free if particles[leg - 1] == name), None)
            if leg is None:
                raise ValueError(f"{rule} has no free leg for {name}")
            free.remove(leg)
            legs.append(leg)
        if free:
            raise ValueError(f"{rule} has unassigned legs {free}")
        return tuple(legs)

    @cached_property
    def formal_nc(self) -> bool:
        """Whether every tree current is exact in N_c (no numeric U(1) pieces)."""

        return not any(
            current.index.auxiliary_kind == _U1_SUBTRACTION
            for dag in (self.blocks.blocks_dag, self.blocks.born_dag)
            for current in dag.currents
        )


def canonical_seed(seed: ConnectedColorTensor) -> ConnectedColorTensor:
    """Fold sqrt(2) pairs into the coefficients: equal vectors compare equal."""

    power = seed.normalization_power
    if power < 2:
        return seed
    return ConnectedColorTensor(
        seed.legs,
        tuple(
            TensorTerm(
                term.coefficient.scaled(Fraction(1, 2 ** (power // 2))),
                term.tensor,
                term.nc_power,
            )
            for term in seed.terms
        ),
        power % 2,
    )


# ------------------------------------------------------------------ seed tables


@dataclass(frozen=True)
class BornBasis:
    """The Born's full-colour basis: literal tensors and amplitude orders.

    ``orders[b]`` are the exact coupling orders of the Born roots of basis
    element b (``TreeBlocks.born_roots``); A0[b] is bucketed by them.
    """

    legs: tuple[ColorLeg, ...]
    sector_ids: tuple[int, ...]
    tensors: tuple[ConnectedColorTensor, ...]
    orders: tuple[frozenset[Orders], ...]

    def __len__(self) -> int:
        return len(self.tensors)

    @cached_property
    def gram(self) -> tuple[tuple[Laurent, ...], ...]:
        """<b|b'> as exact Laurent polynomials (the tree engine's Born metric)."""

        return tuple(
            tuple(
                contract_connected_tensor_nc_terms(left, right)
                for right in self.tensors
            )
            for left in self.tensors
        )

    def degree(self, pairs: Iterable[SectorPair]) -> int | None:
        """Leading N_c degree of a Born squared-order sector (its own P)."""

        pairs = frozenset(pairs)
        return _max_degree(
            cell
            for b, row in enumerate(self.gram)
            for b2, cell in enumerate(row)
            if any((a, a2) in pairs for a in self.orders[b] for a2 in self.orders[b2])
        )


def born_basis(blocks: TreeBlocks) -> BornBasis:
    """The Born colour basis of a loop request (complete, full colour)."""

    plan = blocks.born_dag.color_plan
    sectors = selected_color_sectors(plan, None)
    orders: dict[int, set[Orders]] = {sector.id: set() for sector in sectors}
    for root in blocks.born_roots:
        sector_id = root.color_sector_id
        if sector_id is None:
            if len(sectors) != 1:
                raise ValueError("Born root without a colour sector")
            sector_id = sectors[0].id
        orders[sector_id].add(root.coupling_orders)
    legs = process_color_legs(blocks.process)
    return BornBasis(
        legs,
        tuple(sector.id for sector in sectors),
        tuple(
            ConnectedColorTensor(legs, (sector_color_tensor(sector, blocks.process),))
            for sector in sectors
        ),
        tuple(frozenset(orders[sector.id]) for sector in sectors),
    )


def _max_degree(cells: Iterable[Laurent]) -> int | None:
    return max(
        (power for cell in cells for power, value in cell.items() if value),
        default=None,
    )


def project(cell: Laurent, degree: int | None, accuracy: ColourAccuracy):
    """One cell at N_c = 3 after LC/NLC/FC truncation at leading degree P."""

    if accuracy not in ("lc", "nlc", "full"):
        raise ValueError(f"unknown colour accuracy {accuracy!r}")
    floor = None
    if accuracy != "full":
        if degree is None:
            return ExactColorCoefficient()
        floor = degree - (0 if accuracy == "lc" else 2)
    total = ExactColorCoefficient()
    for power, value in cell.items():
        if floor is None or floor <= power <= degree:
            total = total + value.scaled(Fraction(NC) ** power)
    return total


@dataclass(frozen=True)
class SeedTable:
    """Exact seed tables K[c, b] = <b|kappa_c> (design §7.1).

    ``classes`` are the distinct seed vectors, ``orders[c]`` the loop
    amplitude orders of the skeletons of class c and ``cells[c][b]`` the
    Laurent coefficients of K. Squared-order sectors are given as their
    (Born order, loop order) pairs. ``formal_nc`` is false when the tree
    currents carry numeric U(1) pieces; then only full colour is exact.
    """

    basis: BornBasis
    classes: tuple[ConnectedColorTensor, ...]
    orders: tuple[frozenset[Orders], ...]
    cells: tuple[tuple[Laurent, ...], ...]
    formal_nc: bool = True

    def admitted(self, pairs: Iterable[SectorPair]):
        """(class, loop order, basis element, Born order) entering a sector."""

        pairs = frozenset(pairs)
        for c, loop_orders in enumerate(self.orders):
            for order in sorted(loop_orders):
                for b, born_orders in enumerate(self.basis.orders):
                    for born_order in sorted(born_orders):
                        if (born_order, order) in pairs:
                            yield c, order, b, born_order

    def degree(self, pairs: Iterable[SectorPair]) -> int | None:
        """P: the leading N_c degree of the sector's nonzero cells (§7.4)."""

        return _max_degree(self.cells[c][b] for c, _o, b, _a in self.admitted(pairs))

    def class_degree(self, c: int) -> int | None:
        return _max_degree(self.cells[c])

    def seeds(
        self,
        pairs: Iterable[SectorPair],
        born: Mapping[tuple[int, Orders], Any],
        accuracy: ColourAccuracy = "full",
        *,
        cast: Callable[[ExactColorCoefficient], Any] = complex,
        conjugate: Callable[[Any], Any] = lambda value: value.conjugate(),
        degree: int | None = None,
    ) -> dict[tuple[int, Orders], Any]:
        """U0[c, loop order] = sum_b K[c, b] conj(A0[b, Born order]) (§7.3).

        ``born`` maps (basis element, Born order) to one helicity's Born
        amplitude in any carrier; ``cast`` converts exact cells into it. No
        external average is applied. ``degree`` overrides P, e.g. when the
        sector's counterterm and field cells raise it.
        """

        pairs = frozenset(pairs)
        if accuracy != "full":
            if not self.formal_nc:
                raise ValueError(
                    f"{accuracy.upper()} seeds need N_c-exact tree currents; this "
                    "process has numeric U(1)-subtraction currents (design §3.2)"
                )
            if degree is None:
                degree = self.degree(pairs)
        out: dict[tuple[int, Orders], Any] = {}
        for c, order, b, born_order in self.admitted(pairs):
            amplitude = born.get((b, born_order))
            weight = project(self.cells[c][b], degree, accuracy)
            if amplitude is None or not weight:
                continue
            term = cast(weight) * conjugate(amplitude)
            key = (c, order)
            out[key] = term if key not in out else out[key] + term
        return out


def build_seed_table(
    colour: SeedColour, skeletons: Iterable[Skeleton]
) -> tuple[SeedTable, dict[Skeleton, int]]:
    """Seed classes of ``skeletons`` and their exact table K."""

    basis = born_basis(colour.blocks)
    index: dict[ConnectedColorTensor, int] = {}
    orders: list[set[Orders]] = []
    class_of: dict[Skeleton, int] = {}
    for skeleton in skeletons:
        seed = colour.seed(skeleton)
        c = index.setdefault(seed, len(index))
        if c == len(orders):
            orders.append(set())
        orders[c].add(skeleton.orders)
        class_of[skeleton] = c
    classes = tuple(index)
    # Term-factored (design §7.2): <b|t> once per distinct structure t.
    overlaps: dict[ColorTensor, tuple[Laurent, ...]] = {}
    cells = []
    for seed in classes:
        if seed.normalization_power:
            raise ValueError("a seed class carries an irrational sqrt(2) factor")
        row: list[dict[int, ExactColorCoefficient]] = [{} for _ in basis.tensors]
        for term in seed.terms:
            if term.tensor not in overlaps:
                ket = ConnectedColorTensor(basis.legs, (TensorTerm(_ONE, term.tensor),))
                overlaps[term.tensor] = tuple(
                    contract_connected_tensor_nc_terms(b, ket) for b in basis.tensors
                )
            for cell, overlap in zip(row, overlaps[term.tensor], strict=True):
                for power, value in overlap.items():
                    key = power + term.nc_power
                    cell[key] = cell.get(key, ExactColorCoefficient()) + (
                        term.coefficient * value
                    )
        cells.append(
            tuple(
                {power: value for power, value in cell.items() if value} for cell in row
            )
        )
    cells = tuple(cells)
    return (
        SeedTable(
            basis, classes, tuple(map(frozenset, orders)), cells, colour.formal_nc
        ),
        class_of,
    )


# --------------------------------------------------------------- leading colour


def face_count(left: ColorTensor, right: ColorTensor) -> int:
    """U(N_c) index loops of <left|right>: its N_c power without U(1) parts.

    tau^a_ij tau^a_kl -> delta_il delta_kj glues the traces into index loops,
    the cycles of (next in trace) o (other occurrence of the label).
    """

    traces = overlap_traces(left, right)
    following, partner, seen = {}, {}, {}
    faces = 0
    for t, trace in enumerate(traces):
        if not trace:
            faces += 1
        for i, label in enumerate(trace):
            following[(t, i)] = (t, (i + 1) % len(trace))
            if label in seen:
                partner[(t, i)] = seen[label]
                partner[seen[label]] = (t, i)
            else:
                seen[label] = (t, i)
    if len(partner) != len(following):
        raise ValueError("an adjoint label is not contracted exactly twice")
    visited: set[tuple[int, int]] = set()
    for start in following:
        if start in visited:
            continue
        faces += 1
        current = start
        while current not in visited:
            visited.add(current)
            current = following[partner[current]]
    return faces


def face_degree(
    seed: ConnectedColorTensor, basis: BornBasis, elements=None
) -> int | None:
    """Structural (face-rule) N_c degree of a seed class against the Born basis."""

    elements = range(len(basis)) if elements is None else elements
    return max(
        (
            term.nc_power + born.nc_power + face_count(born.tensor, term.tensor)
            for b in elements
            for born in basis.tensors[b].terms
            for term in seed.terms
        ),
        default=None,
    )


@dataclass(frozen=True)
class LeadingColourCertificate:
    """LC skeleton classes of one sector, certified exactly (design §7.5).

    ``face_degrees`` are structural; ``exact_degrees`` come from the seed table;
    classes outside the sector have ``None``. ``leading`` are the classes the
    face rule keeps (face degree P), each proved to have exact degree P; every
    other class has exact degree below P, so pruning it before numerics is exact.
    """

    degree: int
    face_degrees: tuple[int | None, ...]
    exact_degrees: tuple[int | None, ...]
    leading: frozenset[int]


def certify_leading_colour(
    table: SeedTable, pairs: Iterable[SectorPair]
) -> LeadingColourCertificate:
    """Prove that the face rule selects exactly the degree-P classes."""

    pairs = frozenset(pairs)
    elements: dict[int, set[int]] = {}
    for c, _order, b, _born_order in table.admitted(pairs):
        elements.setdefault(c, set()).add(b)
    degree = table.degree(pairs)
    if degree is None:
        raise ValueError("the sector has no nonzero seed cell")
    faces: list[int | None] = []
    exact: list[int | None] = []
    for c, seed in enumerate(table.classes):
        if c not in elements:
            faces.append(None)
            exact.append(None)
            continue
        face = face_degree(seed, table.basis, sorted(elements[c]))
        value = _max_degree(table.cells[c][b] for b in elements[c])
        if value is not None and (face is None or value > face):
            raise ArithmeticError(f"seed class {c} exceeds its face degree")
        if face is not None and face > degree:
            raise ArithmeticError(f"seed class {c} has face degree {face} above P")
        if face == degree and value != degree:
            raise ArithmeticError(
                f"face rule keeps seed class {c} whose exact degree is {value}"
            )
        faces.append(face)
        exact.append(value)
    return LeadingColourCertificate(
        degree,
        tuple(faces),
        tuple(exact),
        frozenset(c for c, face in enumerate(faces) if face == degree),
    )


__all__ = [
    "BornBasis",
    "ColourNetwork",
    "ColourProduct",
    "ColourString",
    "LeadingColourCertificate",
    "PositionColour",
    "SeedColour",
    "SeedTable",
    "block_colour_network",
    "born_basis",
    "build_seed_table",
    "canonical_seed",
    "certify_leading_colour",
    "face_count",
    "face_degree",
    "multiply",
    "project",
    "reduce_network",
    "relabel",
    "spenso_colour_network",
]
