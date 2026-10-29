# SPDX-License-Identifier: 0BSD
"""Colour seeds of the loop engine (design §3.4, §7, WP1.4, WP2.4).

Colour lives only in exact seed tables. Every skeleton has a *seed class*
kappa_s, the colour vector obtained by closing its colour flow: the product of
the vertex colour tensors of its positions and of the colour-flow strings of
its block words, contracted around the loop. kappa_s is a sum of external
colour structures (open strings, traces, powers of N_c), so it holds for any
model whose coloured vertices lower to colour flow; no Feynman diagram is
used. The seed table is

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
* **Contents.** A content is one routing term in one colour state
  ``PositionColour``: the vertex rule, one of its colour structures, the UFO
  legs the loop and the blocks occupy, and one block word per attached block
  current. Its kinematics is the sum of the vertex terms of that colour
  structure on those legs (couplings times the colour-flow normalisation
  2^(-p/2) of ``loops.v2.catalogue``) times the block currents of those words.
  Seeds absorb the matching 2^(p/2), so kappa carries no sqrt(2).
* **Seeds.** A skeleton's seed is a combination sum_c x_c kappa_c of table
  classes (``build_seed_table``): its generic colour vector, or with
  certified compressions (``CompressedColour``, design §7.2) a combination of
  primitives: Tr_adj(F^sigma) for adjoint rings, tr(tau^sigma) for closed
  fundamental loops and quark rings for loops with quark and gluon arcs. Each
  compression carries a loop-level certificate (``CompressionCertificate``).
  Compression changes only the seed table, never the dressing; the family
  numerators equal the generic ones because pure-gluon blocks are Lie
  elements (``COMPRESSION_THEOREMS``). With compression, classes whose
  colour vectors are proportional share one table row.
* **Contraction** (design §7.3). ``seed_fourier`` certifies the regular
  symmetric-group orbits of a table exactly; ``SeedFourier.seeds``
  contracts them with the native S_m FFT in binary64, the rest directly.
* **N_c grades.** Blocks with U(1)-subtraction pieces enter with their grade
  (``BlockColourFlow.nc_grade``) and the Born amplitude A0[b] is split into
  grades (``SeedTable.seeds``), so seeds are exact in N_c (design §3.2 item
  4). Every block of a leading- or next-to-leading-colour request has one
  grade: trees of one word with different grades (identical-flavour lines)
  are separate blocks (``generation.dag_grades``, ``TreeBlocks.graded``),
  told apart by ``PositionColour.grades``. Full-colour blocks are named by
  their words alone, as every power is kept at N_c = 3. A table over blocks
  that mix grades (``formal_nc`` false; only from a DAG compiled without
  that pass) has only full colour.
* **Signs.** kappa uses the plain colour-flow strings of the blocks.
  ``Block.wick_sign``, the skeleton's fermion sign epsilon_s, its automorphism
  weight and its i-factors belong to the leaf weight (design §3.3), never to K.
* **Projection** (design §7.4). P is the largest N_c degree of a nonzero cell
  of a squared-order sector (Born grades included). LC keeps degree P, NLC
  keeps P, P-1 and P-2 and FC keeps every power (``kept``); N_c = 3 is
  substituted after projection and the external average after that.
* **Leading colour** (design §7.5). The face degree of a class is the U(N_c)
  index-loop count of its colour flow glued to the Born basis (integer cycle
  counting, no numerics); ``certify_leading_colour`` proves for a sector that
  the face rule keeps exactly the classes whose exact degree is P. On
  colour-ordered (planar) skeletons the rule is local
  (``PlanarLeadingColour``, against ``leading_born_flows``); the slice prunes
  every skeleton whose exact seed cells lie below the kept powers before any
  numerics (``assemble.LoopSlice``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache, cached_property, lru_cache
from itertools import count, permutations, product
from typing import Any, Literal

from pyamplicol.color.connections import (
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
    OpenColorString,
    TensorTerm,
    _literal_tensor_overlap_nc_terms,
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
from .families import ExternalLeg, Orders, Position, add_orders, model_data
from .skeletons import Content, Skeleton

Label = Hashable
Laurent = Mapping[int, ExactColorCoefficient]
ColourAccuracy = Literal["lc", "nlc", "full"]
# (Born amplitude orders, loop amplitude orders): one pair of a squared-order sector.
SectorPair = tuple[Orders, Orders]

_ONE = ExactColorCoefficient(1)

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
    if head == "trace":
        # Spenso's trace(cof(3), cyclic(t(coad(8, a), in, out), ...)) of
        # generators: tr(t^a1 ... t^ak) = 2^(-k/2) tr(tau^a1 ... tau^ak).
        representation, cyclic = arguments
        if _head(representation) != "cof" or _head(cyclic) != "cyclic":
            raise NotImplementedError(f"unsupported colour trace: {atom}")
        word = []
        for matrix in cyclic:
            if _head(matrix) != "t":
                raise NotImplementedError(f"unsupported trace matrix: {matrix}")
            slot, a = _index(next(iter(matrix)))
            if slot != "adjoint":
                raise NotImplementedError(f"unsupported trace matrix: {matrix}")
            word.append(a)
        return (ColourProduct(traces=(tuple(word),), root2_power=len(word)),)
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

    Accepts sums of products of numbers, ``f``, ``t``, ``chain``, ``trace``
    and colour metrics, i.e. the ``color_expression`` of compiled vertex
    terms, the legacy engine's colour tensors and the whole-rule continuation
    tensors. Index labels are the slot names.
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


def literal_network(
    tensor: ConnectedColorTensor, rename: Callable[[int], Label] = lambda label: label
) -> ColourNetwork:
    """A literal colour tensor as a colour network, labels renamed by
    ``rename`` (the inverse of ``reduce_network``'s decoding: a literal
    trace is the reversed network trace, ``OpenColorString(q, w, qbar)`` the
    network string with row ``qbar``, word reversed(w) and column ``q``).
    Renaming two labels to one internal label contracts them in
    ``reduce_network``. The tensor's 1/sqrt(2) normalisation is not carried."""

    out = []
    for term in tensor.terms:
        strings = tuple(
            ColourString(
                rename(string.antifundamental_label),
                tuple(rename(a) for a in reversed(string.adjoint_labels)),
                rename(string.fundamental_label),
            )
            for string in term.tensor.open_strings
        )
        traces = tuple(
            tuple(rename(a) for a in reversed(trace)) for trace in term.tensor.traces
        )
        out.append(ColourProduct(term.coefficient, strings, traces, term.nc_power))
    return tuple(out)


# ------------------------------------------------------------------ seed classes


@dataclass(frozen=True, slots=True, order=True)
class PositionColour:
    """Colour state of one routing term of a position (generic mode).

    ``legs`` are the 1-based UFO legs of the vertex rule taken by the loop
    line entering and leaving the position and by each attached block, in the
    order of ``Position.blocks``; the kinematic segment uses the same legs.
    ``words`` holds the block word of each attached block current. Over
    graded blocks (``TreeBlocks.graded``: leading and next-to-leading colour)
    ``grades`` holds its N_c grade (``BlockColourFlow.nc_grade``):
    identical-flavour lines reach one word with several grades, as separate
    blocks (``generation.dag_grades``), so a word and its grade name the
    block. Over full-colour blocks a word alone names its blocks (every power
    is kept at N_c = 3) and ``grades`` is empty.
    """

    rule: str
    colour_index: int
    legs: tuple[int, ...]
    words: tuple[tuple[int, ...], ...]
    grades: tuple[int, ...] = ()

    @property
    def block_names(self) -> tuple[BlockName, ...]:
        """The name of each attached block among the blocks of its block
        current (``block_name``; ``SeedColour.attached_blocks``, the block
        rows of ``reference_executor.BlockRows``): its word, or (word, N_c
        grade) over graded blocks."""

        if not self.grades:
            return self.words
        return tuple(zip(self.words, self.grades, strict=True))


# A block's name among the blocks of its block current: its word, or (word,
# N_c grade) over graded blocks (``block_name``).
BlockName = tuple[int, ...] | tuple[tuple[int, ...], int | None]


def block_name(blocks: TreeBlocks, block: Block) -> BlockName:
    """The name with which a colour state refers to ``block`` of ``blocks``
    (``PositionColour.block_names``): its word over full-colour blocks, (word,
    N_c grade) over graded blocks (``TreeBlocks.graded``)."""

    word = block.key.word
    return (word, block.colour_flow.nc_grade) if blocks.graded else word


def block_colour_network(
    flow: BlockColourFlow,
    representations: Mapping[int, int],
    field_representation: int,
    open_label: Label,
) -> ColourNetwork:
    """The colour of one block word (``loops.v2.blocks`` convention).

    A string s -> a_1 .. a_k -> e is (tau^a_k .. tau^a_1)^e_s; an adjoint field
    closes its open ends with (tau^X)^in_out. ``open_label`` names the field's
    colour index (X, or the open fundamental end). A block of N_c grade g
    (U(1)-subtraction pieces, whose rows carry 3^g) contributes N_c^g 3^(-g).

    A block whose trees mix grades (``nc_grade`` ``None``: only from a DAG
    compiled without the grade split, i.e. full colour) takes g = 0. That is
    exact where such a block can be evaluated: at N_c = 3, N_c^g 3^(-g) = 1
    for every g, and full colour keeps every power at N_c = 3. This network
    is built once for every accuracy of a seed table, so the accuracy is not
    known here; a projection of such a block is refused where it is
    (``SeedTable.seeds``; at generation ``refusals.require_nc_exact``, at
    export ``refusals.require_formal_nc_tables``).
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
    grade = flow.nc_grade or 0
    return (
        ColourProduct(
            ExactColorCoefficient(Fraction(NC) ** -grade),
            strings=tuple(strings),
            traces=flow.traces,
            nc_power=grade,
        ),
    )


_UFO_SLOT = re.compile(r"ufo_c_(\d+)$")


def _graded_name_order(name: tuple[tuple[int, ...], int]) -> tuple:
    """Graded block names in state order: words in order, then the grades of
    one word decreasing (0 first)."""

    word, grade = name
    return (word, -grade)


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
        data = model_data(native_model)
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
        # (mask, field, orders) -> block name (``block_name``) -> block. The
        # blocks of one name differ only by chirality or by the flavour flow
        # of their trees (one field current the tree engine splits; their
        # rows add, ``assemble._BlockRows``), which colour ignores: their
        # colour flows must agree up to the sign their rows carry.
        self._words: dict[tuple[int, str, Orders], dict[BlockName, Block]] = {}
        for block in blocks.blocks:
            key = block.key
            if key.auxiliary_kind is not None:
                continue
            if block.colour_flow.traces or key.color_state.lines:
                raise NotImplementedError(
                    f"loop seeds of block {key.word} with closed colour traces "
                    "(a colour-singlet vertex of two adjoint currents)"
                )
            if blocks.graded and block.colour_flow.nc_grade is None:
                # The grade split leaves every block one grade; a graded
                # table must never name a block by a mixed one.
                raise ValueError(
                    f"block {key.word} of a graded loop request mixes N_c "
                    "grades (generation.dag_grades left it mixed)"
                )
            field_name = self._fields.incoming(key.particle_id)["name"]
            first = self._words.setdefault(
                (key.external_mask, field_name, key.coupling_orders), {}
            ).setdefault(block_name(blocks, block), block)
            if first is not block and replace(first.colour_flow, sign=1) != replace(
                block.colour_flow, sign=1
            ):
                raise ValueError(
                    f"blocks {first.id} and {block.id} of word {key.word} and "
                    "field current differ in colour flow"
                )
        # The block names of each block current in state order, once.
        self._ordered: dict[tuple[int, str, Orders], tuple[BlockName, ...]] = {}

    # -- ColourModel ---------------------------------------------------------

    def states(self, position: Position) -> tuple[PositionColour, ...]:
        vertex = position.vertex
        indices = self._indices.get((vertex.rule, add_orders(vertex.orders)))
        if not indices:
            # The loop catalogue holds a vertex the tree model dropped (one
            # whose couplings vanish at the restriction is no loop vertex,
            # ``ct_sites.vanishing_loop_vertices``): refused, listed.
            from .refusals import LoopRequestUnsupported

            particles = " ".join(self._rules[vertex.rule]["particles"])
            raise LoopRequestUnsupported(
                "loop-vertex-outside-tree-model",
                f"{self.blocks.process.process!r}: the loop vertex {vertex.rule} "
                f"({particles}) at the coupling orders {dict(vertex.orders)} "
                "has no term in the request's tree model, and its couplings are "
                "not shown to vanish at the model's restriction",
            )
        legs = self._canonical_legs(
            vertex.rule,
            vertex.loop_in,
            self._fields.by_name[vertex.loop_out]["antiname"],
            tuple(block.field for block in position.blocks),
        )
        options = [self._ordered_names(block) for block in position.blocks]
        if not self.blocks.graded:
            return tuple(
                PositionColour(vertex.rule, index, legs, words)
                for index in sorted(indices)
                for words in product(*options)
            )
        return tuple(
            PositionColour(
                vertex.rule,
                index,
                legs,
                tuple(word for word, _grade in choice),
                tuple(grade for _word, grade in choice),
            )
            for index in sorted(indices)
            for choice in product(*options)
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
        return PositionColour(
            state.rule, state.colour_index, legs, state.words, state.grades
        )

    # -- seeds ---------------------------------------------------------------

    def attached_blocks(self, content: Content) -> tuple[Block, ...]:
        """The tree block of each attached block current of a content.

        Every routing term of a content must resolve to the same blocks: the
        content's colour (and so its seed) is that of any one term.
        """

        state = content.colour
        if not isinstance(state, PositionColour):
            raise TypeError("seed classes need PositionColour contents")
        found = None
        for term in content.terms:
            blocks = tuple(
                self._words[(current.mask, current.field, current.orders)][word]
                for current, word in zip(term.blocks, state.block_names, strict=True)
            )
            if found is None:
                found = blocks
            elif blocks != found:
                raise ValueError("the routing terms of a content differ in colour")
        assert found is not None
        return found

    def seed(self, skeleton: Skeleton) -> ConnectedColorTensor:
        """kappa_s: the closed colour flow of a skeleton (module docstring)."""

        count_ = len(skeleton.contents)
        factors: list[ColourNetwork] = []
        for position, content in enumerate(skeleton.contents):
            state = content.colour
            blocks = self.attached_blocks(content)
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

            factors.append(relabel(self.vertex_network(state), rename))
            for j, block in enumerate(blocks):
                factors.append(
                    block_colour_network(
                        block.colour_flow,
                        self._representations,
                        self.field_colour(block),
                        ("block", position, j),
                    )
                )
        external = {label: label for label in self._representations}
        return canonical_seed(reduce_network(multiply(*factors), self.legs, external))

    def vertex_network(self, state: PositionColour) -> ColourNetwork:
        """The colour of a content's vertex structure, labelled ("leg", i) by
        its UFO legs (seed normalisation, module docstring)."""

        return self._vertex[(state.rule, state.colour_index)]

    def field_colour(self, block: Block) -> int:
        return int(self._fields.by_pdg[block.key.particle_id]["color"])

    def species_colour(self, name: str) -> int:
        """Colour representation of a loop species (incoming field name)."""

        return int(self._fields.by_name[name]["color"])

    def all_adjoint_rules(self) -> tuple[tuple[str, int], ...]:
        """(rule, colour index) of every vertex structure whose legs are all
        adjoint: the vertices that pure-gluon tree blocks are built from."""

        return tuple(
            sorted(
                key
                for key in self._vertex
                if all(
                    int(self._fields.by_name[name]["color"]) == 8
                    for name in self._rules[key[0]]["particles"]
                )
            )
        )

    # -- helpers -------------------------------------------------------------

    def _block_words(self, block) -> tuple[BlockName, ...]:
        """The names (``block_name``) of a block current's tree blocks."""

        words = self._words.get((block.mask, block.field, block.orders))
        if not words:
            raise ValueError(f"no tree block for {block}")
        return tuple(words)

    def _ordered_names(self, block) -> tuple[BlockName, ...]:
        """``_block_words`` in state order (words in order; over graded
        blocks the grades of one word decreasing), cached per block
        current."""

        key = (block.mask, block.field, block.orders)
        found = self._ordered.get(key)
        if found is None:
            names = self._block_words(block)
            found = self._ordered[key] = tuple(
                sorted(names, key=_graded_name_order)
                if self.blocks.graded
                else sorted(names)
            )
        return found

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
        """Whether every block is exact in N_c: its numeric U(1)-subtraction
        pieces have one N_c grade (``BlockColourFlow.nc_grade``); the
        criterion of the LC/NLC generation refusal
        (``refusals.mixed_grade_blocks``)."""

        from .refusals import mixed_grade_blocks

        return not mixed_grade_blocks(self.blocks)


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


# ------------------------------------------------------- certified compressions

PrimitiveKind = Literal["adjoint-ring", "fundamental-trace", "quark-ring"]

# The theorem behind each certified compression (design §3.4, §7.2).
COMPRESSION_THEOREMS: Mapping[PrimitiveKind, str] = {
    "adjoint-ring": (
        "Del Duca, Dixon, Maltoni, hep-ph/9910563 sec. 3: a ring of adjoint "
        "lines whose vertices are products of structure constants is "
        "Tr(F^sigma) times the colour-ordered ring. A pure-gluon tree current "
        "sum_w J(w) tau^w is a Lie element (shuffle/Kleiss-Kuijf relations), "
        "and ad is a Lie homomorphism, so ad(sum_w J(w) tau^w) = "
        "sum_w J(w) F^w: each block word w enters as F^w."
    ),
    "fundamental-trace": (
        "A closed fundamental loop is Tr(prod tau^b) over its attachments. A "
        "pure-gluon tree current sum_w J(w) tau^w is a Lie element, hence "
        "traceless: its U(1) (-1/N_c Tr) piece vanishes and the block enters "
        "as tau^w, so the seed is Tr(tau^sigma) of the concatenated word."
    ),
    "quark-ring": (
        "Del Duca, Dixon, Maltoni, hep-ph/9910563 (one quark line, LoopColorqqNew) "
        "in its colour-flow form: on a ring with quark and gluon arcs, every "
        "pure-gluon block on a gluon arc enters as F^w and on a quark arc as "
        "tau^w (the Lie-element arguments of the two other classes); the "
        "positions where the loop line changes and every other block keep "
        "their exact colour, so the seed depends on the arcs' letter words."
    ),
}


@dataclass(frozen=True, slots=True, order=True)
class Primitive:
    """A certified primitive seed class (design §7.2).

    ``word`` lists external adjoint legs in matrix-product order:
    ``adjoint-ring`` is Tr_adj(F^w_1 ... F^w_n) with
    (F^a)_xy = tr(tau^x [tau^a, tau^y]) (the adjoint representation in the
    tau basis), ``fundamental-trace`` is tr(tau^w_1 ... tau^w_n). A
    ``quark-ring`` word lists one descriptor per ring position
    (``_ring_network``): ("adjoint", letters) for (F^letters)_{in out},
    ("fundamental", letters, forward) for a tau string and ("exact",
    network) for a position kept exactly. Words are canonical
    (``canonical_primitive``).
    """

    kind: PrimitiveKind
    word: tuple[int, ...]

    def un_terms(self) -> tuple[tuple[int, tuple[tuple[int, ...], ...], int], ...]:
        """The U(N) expansion: (sign, traces in natural order, N_c power).

        Tr_adj(prod_i ad tau^w_i) = sum_S (-1)^|not S| tr(tau^w_S)
        tr(tau^reversed(w_not S)) over the gl(N_c) action Y -> XY - YX (the
        identity direction carries no trace since ad 1 = 0); empty traces
        are N_c. Single-letter traces are kept: their U(N) values cancel the
        U(1) components, so U(N) contractions of the whole expansion are the
        SU(N) ones.
        """

        if self.kind == "fundamental-trace":
            return ((1, (self.word,), 0),)
        n = len(self.word)
        out = []
        for mask in range(1 << n):
            inside = tuple(a for i, a in enumerate(self.word) if mask >> i & 1)
            outside = tuple(
                reversed([a for i, a in enumerate(self.word) if not mask >> i & 1])
            )
            traces = tuple(trace for trace in (inside, outside) if trace)
            out.append((-1 if (n - len(inside)) % 2 else 1, traces, 2 - len(traces)))
        return tuple(out)

    def network(self) -> ColourNetwork:
        """The exact colour network (natural-order traces, explicit N_c)."""

        if self.kind == "quark-ring":
            return _ring_network(self.word)
        return tuple(
            ColourProduct(ExactColorCoefficient(sign), traces=traces, nc_power=power)
            for sign, traces, power in self.un_terms()
        )

    def tensor(self, legs: Sequence[ColorLeg]) -> ConnectedColorTensor:
        """The SU(N_c) literal tensor (single-letter traces vanish).

        Quark rings are reduced once per relabelling class of their external
        adjoint labels (``_ring_relabelling``) and relabelled back.
        """

        legs = tuple(legs)
        if self.kind == "quark-ring":
            adjoint = frozenset(leg.label for leg in legs if leg.representation == 8)
            canonical, back = _ring_relabelling(self, adjoint)
            return _relabelled(_reduced_primitive(canonical, legs), back)
        return _reduced_primitive(self, legs)


#: Quark-ring reductions made natively by the seed stage
#: (``native_seeds._prime_reduction``), taken by ``_reduced_primitive`` in
#: place of ``reduce_network`` so its cache holds them.
_NATIVE_REDUCED: dict[tuple[Primitive, tuple[ColorLeg, ...]], ConnectedColorTensor] = {}


@lru_cache(maxsize=1 << 14)
def _reduced_primitive(
    primitive: Primitive, legs: tuple[ColorLeg, ...]
) -> ConnectedColorTensor:
    native = _NATIVE_REDUCED.pop((primitive, legs), None)
    if native is not None:
        return native
    external = {leg.label: leg.label for leg in legs}
    return reduce_network(primitive.network(), legs, external)


def _ring_relabelling(
    primitive: Primitive, adjoint: frozenset[int]
) -> tuple[Primitive, dict[int, int]]:
    """(canonical quark ring, map from its adjoint labels back to the ring's).

    The ring's external adjoint labels, in order of first appearance, are
    renamed onto their sorted set; the colour depends on labels only through
    their names, so tensor(ring) = back(tensor(canonical)). A quark ring is a
    cyclic word, so the canonical one is also rotated (``canonical_primitive``).
    """

    seen: dict[int, None] = {}

    def visit(label: Label) -> None:
        if isinstance(label, int) and label in adjoint:
            seen.setdefault(label)

    for descriptor in primitive.word:
        if descriptor[0] == "exact":
            for term in descriptor[1]:
                for string in term.strings:
                    for label in (string.row, *string.word, string.col):
                        visit(label)
                for trace in term.traces:
                    for label in trace:
                        visit(label)
        else:
            for label in descriptor[1]:
                visit(label)
    forward = dict(zip(seen, sorted(seen), strict=True))
    if all(a == b for a, b in forward.items()):
        canonical, _sign = canonical_primitive("quark-ring", primitive.word)
        return canonical, {}

    def rename(label: Label) -> Label:
        return forward.get(label, label) if isinstance(label, int) else label

    word = []
    for descriptor in primitive.word:
        if descriptor[0] == "exact":
            word.append(("exact", relabel(descriptor[1], rename)))
        else:
            kind, letters, *rest = descriptor
            word.append((kind, tuple(map(rename, letters)), *rest))
    canonical, _sign = canonical_primitive("quark-ring", word)
    return canonical, {b: a for a, b in forward.items()}


def canonical_primitive(
    kind: PrimitiveKind, word: Sequence[int]
) -> tuple[Primitive, int]:
    """(canonical primitive, sign) with Primitive(kind, word) = sign x canonical.

    Both traces are cyclic; an adjoint ring also equals (-1)^n times its
    reflection (F^a is antisymmetric).
    """

    word = tuple(word)
    n = len(word)
    if kind == "quark-ring":
        best = min((word[i:] + word[:i] for i in range(n)), key=repr)
        return Primitive(kind, best), 1
    candidates = [(word[i:] + word[:i], 1) for i in range(n)] or [((), 1)]
    if kind == "adjoint-ring":
        mirror = tuple(reversed(word))
        sign = -1 if n % 2 else 1
        candidates += [(mirror[i:] + mirror[:i], sign) for i in range(n)]
    best, sign = min(candidates)
    return Primitive(kind, best), sign


def _ring_network(positions: Sequence[tuple]) -> ColourNetwork:
    """The colour of a ring of position descriptors (``Primitive.word``).

    Position p joins the loop labels ("loop", p-1) (its "in") and
    ("loop", p) (its "out"); external legs keep their integer labels and
    every other label of an exact position is local.
    """

    count_ = len(positions)
    factors = []
    for p, descriptor in enumerate(positions):
        in_label, out_label = ("loop", (p - 1) % count_), ("loop", p)
        kind = descriptor[0]
        if kind == "adjoint":
            factors.append(_adjoint_chain(descriptor[1], in_label, out_label, p))
        elif kind == "fundamental":
            _kind, letters, forward = descriptor
            row, col = (out_label, in_label) if forward else (in_label, out_label)
            factors.append((ColourProduct(strings=(ColourString(row, letters, col),)),))
        else:
            names = {"in": in_label, "out": out_label}

            def rename(label: Label, p=p, names=names) -> Label:
                if isinstance(label, int):
                    return label  # an external leg
                return names.get(label, ("local", p, label))

            factors.append(relabel(descriptor[1], rename))
    return multiply(*factors)


def _adjoint_chain(
    letters: Sequence[Label], row: Label, col: Label, tag: Hashable
) -> ColourNetwork:
    """(F^l_1 ... F^l_k)_{row col} with fresh dummy labels tagged ``tag``."""

    if not letters:
        return (ColourProduct(traces=((row, col),)),)
    labels = [row, *(("chain", tag, i) for i in range(len(letters) - 1)), col]
    factors = []
    for i, letter in enumerate(letters):
        x, y = labels[i], labels[i + 1]
        factors.append(
            (
                ColourProduct(traces=((x, letter, y),)),
                ColourProduct(ExactColorCoefficient(-1), traces=((x, y, letter),)),
            )
        )
    return multiply(*factors)


def _solve(
    columns: Sequence[ConnectedColorTensor], target: ConnectedColorTensor
) -> tuple[ExactColorCoefficient, ...] | None:
    """Exact coefficients x with sum_j x_j columns_j = target, or None.

    Columns have real rational coefficients; the real and imaginary parts
    of the target are solved separately by Fraction elimination.
    """

    if target.normalization_power or any(c.normalization_power for c in columns):
        return None
    keys = sorted(
        {(t.tensor, t.nc_power) for c in (*columns, target) for t in c.terms},
        key=repr,
    )
    index = {key: i for i, key in enumerate(keys)}

    def vector(tensor: ConnectedColorTensor, part: str) -> list[Fraction]:
        out = [Fraction(0)] * len(keys)
        for t in tensor.terms:
            out[index[(t.tensor, t.nc_power)]] += getattr(t.coefficient, part)
        return out

    matrix = [vector(column, "real") for column in columns]
    if any(
        any(column.imag for column in (t.coefficient for t in c.terms)) for c in columns
    ):
        return None
    solutions = []
    for part in ("real", "imag"):
        solution = _eliminate(matrix, vector(target, part))
        if solution is None:
            return None
        solutions.append(solution)
    return tuple(
        ExactColorCoefficient(re, im) for re, im in zip(*solutions, strict=True)
    )


def _eliminate(
    columns: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> list[Fraction] | None:
    """One exact solution of sum_j x_j columns_j = rhs (free unknowns zero)."""

    unknowns = len(columns)
    rows = [
        [columns[j][i] for j in range(unknowns)] + [rhs[i]] for i in range(len(rhs))
    ]
    pivots = []
    rank = 0
    for j in range(unknowns):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        head = rows[rank][j]
        rows[rank] = [value / head for value in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[j]:
                factor = row[j]
                rows[i] = [a - factor * b for a, b in zip(row, rows[rank], strict=True)]
        pivots.append(j)
        rank += 1
    if any(row[-1] for row in rows[rank:]):
        return None
    solution = [Fraction(0)] * unknowns
    for i, j in enumerate(pivots):
        solution[j] = rows[i][-1]
    return solution


@dataclass(frozen=True, slots=True)
class VertexDecomposition:
    """A vertex structure as a sum of products over its adjoint slots.

    Adjoint loop lines: V(in, out, b...) = sum c (F^b_pi1 ... F^b_pik)_{in out};
    fundamental loop lines: V = sum c (tau^b_pi1 ... tau^b_pik)^row_col with
    (row, col) = (out, in) if ``forward`` else (in, out). ``terms`` hold
    (c, pi) with pi indexing the adjoint slots.
    """

    kind: PrimitiveKind
    terms: tuple[tuple[ExactColorCoefficient, tuple[int, ...]], ...]
    forward: bool = True


def decompose_vertex(
    network: ColourNetwork,
    in_label: Label,
    out_label: Label,
    slots: Sequence[Label],
    kind: PrimitiveKind,
) -> VertexDecomposition | None:
    """Exact decomposition of a vertex network over ordered slot products.

    The colour-space identity behind a certified compression: the vertex is
    proved, exactly in N_c, to be a combination of the products of the
    compression's generators over its adjoint slots ``slots`` (every other
    label of ``network`` is internal or a colour-singlet leg).
    """

    legs, external = [], {}
    if kind == "adjoint-ring":
        ends = {in_label: 8, out_label: 8}
        forward = True
    else:
        rows = {s.row for p in network for s in p.strings}
        cols = {s.col for p in network for s in p.strings}
        if {in_label, out_label} != (rows | cols) & {in_label, out_label}:
            return None
        forward = out_label in rows and in_label in cols
        if not forward and not (in_label in rows and out_label in cols):
            return None
        row, col = (out_label, in_label) if forward else (in_label, out_label)
        ends = {row: -3, col: 3}
    for i, label in enumerate([*ends, *slots], start=1):
        external[label] = i
        legs.append(ColorLeg(i, ends.get(label, 8)))
    target = reduce_network(network, legs, external)
    orders = tuple(permutations(range(len(slots))))
    columns = []
    for order in orders:
        letters = [slots[k] for k in order]
        if kind == "adjoint-ring":
            candidate = _adjoint_chain(letters, in_label, out_label, "decompose")
        else:
            candidate = (
                ColourProduct(strings=(ColourString(row, tuple(letters), col),)),
            )
        columns.append(reduce_network(candidate, legs, external))
    solution = _solve(columns, target)
    if solution is None:
        return None
    return VertexDecomposition(
        kind,
        tuple((c, order) for c, order in zip(solution, orders, strict=True) if c),
        forward,
    )


Combination = tuple[tuple[Hashable, ExactColorCoefficient], ...]


class CompressedColour:
    """Certified compressions of generic seed classes (design §3.4, §7.2).

    A position compresses when its loop lines are both adjoint (gluons,
    ghosts) and every attached block is a pure-gluon current, or both
    fundamental and every block is pure-gluon or colour singlet, and its
    vertex decomposes exactly over the adjoint slots (``decompose_vertex``):
    the blocks then enter through their letters (F^w or tau^w). Every other
    position keeps its exact colour. A skeleton of compressed positions only
    is an adjoint ring or a closed fundamental loop; one that mixes them with
    exact positions (a quark line through the loop) is a quark ring. Its
    seed is sum_t c_t P_t over the primitives P (``combination``), equal to
    its generic seed summed over the words of its blocks by the theorems of
    ``COMPRESSION_THEOREMS``. ``certificate`` adds the tree vertices
    (pure-gluon blocks are Lie elements) and the smallest instance.
    """

    def __init__(self, colour: SeedColour):
        self.colour = colour
        self._vertices: dict[tuple, VertexDecomposition | None] = {}
        self._contents: dict[int, tuple[Content, tuple | None]] = {}
        self._products: dict[tuple, ExactColorCoefficient] = {}
        # A combination depends only on the kind and the positions' terms
        # (decomposition ids and descriptors), which many skeletons share.
        self._combinations: dict[tuple, Combination | None] = {}
        # Ring combinations of keys equal up to a renaming of the external
        # labels (``_renamed``), on the renamed key.
        self._renamed: dict[tuple, Combination | None] = {}

    def _letters(self, block: Block) -> tuple[int, ...] | None:
        """M_w = tau^w_k ... tau^w_1 of a pure-gluon block, in matrix order."""

        flow = block.colour_flow
        if (
            self.colour.field_colour(block) != 8
            or flow.nc_grade != 0
            or flow.singlet_labels
            or flow.traces
            or len(flow.strings) != 1
            or not (flow.incoming_open and flow.outgoing_open)
            or any(self.colour._representations[a] != 8 for a in flow.strings[0])
        ):
            return None
        return tuple(reversed(flow.strings[0]))

    def _singlet(self, block: Block) -> bool:
        return self.colour.field_colour(block) == 1 and not block.colour_flow.strings

    def decomposition(
        self, state: PositionColour, kind: PrimitiveKind, slots: tuple[int, ...]
    ) -> VertexDecomposition | None:
        """The vertex of ``state`` over its adjoint block legs ``slots``."""

        key = (state.rule, state.colour_index, state.legs, slots, kind)
        if key not in self._vertices:
            in_leg, out_leg, *_attached = state.legs
            self._vertices[key] = decompose_vertex(
                self.colour.vertex_network(state),
                ("leg", in_leg),
                ("leg", out_leg),
                [("leg", leg) for leg in slots],
                kind,
            )
        return self._vertices[key]

    def _content(self, content: Content) -> tuple:
        """(position kind, single-leg compressed blocks, terms) of a content.

        Kinds: "adjoint" (adjoint loop lines, pure-gluon blocks) and
        "fundamental" (quark loop lines, pure-gluon or singlet blocks) when the
        vertex decomposes, terms ((term id, descriptor), ...); otherwise
        "exact", whose one term keeps the vertex and block colour.
        """

        # Contents are shared by many skeletons; their hash is costly.
        cached = self._contents.get(id(content))
        if cached is not None and cached[0] is content:
            return cached[1]
        found = self._compressed(content) or (
            "exact",
            True,
            (((None, 0), ("exact", self._exact_network(content))),),
        )
        self._contents[id(content)] = (content, found)
        return found

    def _compressed(self, content: Content) -> tuple | None:
        colour = self.colour
        loop = {colour.species_colour(content.loop_in)}
        loop.add(colour.species_colour(content.loop_out))
        if loop == {8}:
            kind: PrimitiveKind = "adjoint-ring"
        elif loop <= {3, -3}:
            kind = "fundamental-trace"
        else:
            return None
        blocks = colour.attached_blocks(content)
        letters, slots = [], []
        _in, _out, *attached = content.colour.legs
        for leg, block in zip(attached, blocks, strict=True):
            word = self._letters(block)
            if word is not None:
                letters.append(word)
                slots.append(leg)
            elif not (kind == "fundamental-trace" and self._singlet(block)):
                return None
        decomposition = self.decomposition(content.colour, kind, tuple(slots))
        if decomposition is None:
            return None
        position = "adjoint" if kind == "adjoint-ring" else "fundamental"
        terms = []
        for t, (_c, order) in enumerate(decomposition.terms):
            word = tuple(a for k in order for a in letters[k])
            descriptor = (
                (position, word)
                if position == "adjoint"
                else (position, word, decomposition.forward)
            )
            terms.append(((id(decomposition), t), descriptor))
        return position, all(len(w) == 1 for w in letters), tuple(terms)

    def _exact_network(self, content: Content) -> ColourNetwork:
        """A content's vertex and block colour with labels "in", "out" for
        its loop lines (``_ring_network``)."""

        colour = self.colour
        state = content.colour
        in_leg, out_leg, *attached = state.legs
        slots: dict[int, Label] = {in_leg: "in", out_leg: "out"}
        slots.update((leg, ("block", j)) for j, leg in enumerate(attached))

        def rename(label: Label) -> Label:
            kind, value = label
            return slots[value] if kind == "leg" else ("dummy", value)

        factors = [relabel(colour.vertex_network(state), rename)]
        for j, block in enumerate(colour.attached_blocks(content)):
            factors.append(
                block_colour_network(
                    block.colour_flow,
                    colour._representations,
                    colour.field_colour(block),
                    ("block", j),
                )
            )
        return multiply(*factors)

    def _coefficient(self, ids: tuple) -> ExactColorCoefficient:
        found = self._products.get(ids)
        if found is None:
            by_id = {id(d): d for d in self._vertices.values() if d is not None}
            found = _ONE
            for vertex, t in ids:
                if vertex is not None:
                    found = found * by_id[vertex].terms[t][0]
            self._products[ids] = found
        return found

    def kind(self, skeleton: Skeleton) -> PrimitiveKind | None:
        """The compression class of a skeleton, or None (generic seed)."""

        kinds = {self._content(content)[0] for content in skeleton.contents}
        if kinds == {"adjoint"}:
            return "adjoint-ring"
        if kinds == {"fundamental"}:
            return "fundamental-trace"
        if "exact" in kinds and len(kinds) > 1:
            return "quark-ring"
        return None

    def combination(self, skeleton: Skeleton) -> Combination | None:
        """sum_t c_t P_t of a certified skeleton, or None (generic seed)."""

        kind = self.kind(skeleton)
        if kind is None:
            return None
        infos = [self._content(content) for content in skeleton.contents]
        key = (kind, tuple(info[2] for info in infos))
        if key in self._combinations:
            return self._combinations[key]
        if kind == "quark-ring":
            found = self._combination(kind, infos)
        else:
            found = self._ring_combination(kind, key)
        self._combinations[key] = found
        return found

    def _ring_combination(self, kind: PrimitiveKind, key: tuple) -> Combination | None:
        """An adjoint ring's or closed fundamental loop's combination from the
        one of its key with the external labels renamed 1, 2, ... in reading
        order: the products of position terms do not depend on the labels, so
        the renamed key's primitives, renamed back and made canonical again,
        are this key's (a primitive's sign changes with its canonical form).
        Rings of relabelled legs (identical gluons) share one renamed key."""

        rename: dict[int, int] = {}
        for terms in key[1]:
            for _ids, descriptor in terms:
                for label in descriptor[1]:
                    if label not in rename:
                        rename[label] = len(rename) + 1
        renamed = (
            kind,
            tuple(
                tuple(
                    (
                        ids,
                        (
                            descriptor[0],
                            tuple(rename[a] for a in descriptor[1]),
                            *descriptor[2:],
                        ),
                    )
                    for ids, descriptor in terms
                )
                for terms in key[1]
            ),
        )
        if renamed not in self._renamed:
            infos = [(None, None, terms) for terms in renamed[1]]
            self._renamed[renamed] = self._combination(kind, infos)
        found = self._renamed[renamed]
        if found is None:
            return None
        back = {new: old for old, new in rename.items()}
        out = []
        for primitive, coefficient in found:
            original, sign = canonical_primitive(
                kind, tuple(back[a] for a in primitive.word)
            )
            out.append((original, coefficient if sign == 1 else coefficient.scaled(-1)))
        return tuple(sorted(out, key=repr))

    def _combination(self, kind: PrimitiveKind, infos) -> Combination | None:
        out: dict[Primitive, ExactColorCoefficient] = {}
        for choice in product(*(info[2] for info in infos)):
            descriptors = tuple(descriptor for _ids, descriptor in choice)
            if kind == "quark-ring":
                primitive, sign = canonical_primitive(kind, descriptors)
            else:
                word = self._ring_word(kind, descriptors)
                if word is None:
                    return None
                primitive, sign = canonical_primitive(kind, word)
            coefficient = self._coefficient(tuple(ids for ids, _d in choice))
            out[primitive] = out.get(primitive, ExactColorCoefficient()) + (
                coefficient if sign == 1 else coefficient.scaled(-1)
            )
        return tuple(sorted(((p, c) for p, c in out.items() if c), key=repr))

    def single_legged(self, skeleton: Skeleton) -> bool:
        """Whether every compressed block of a certified skeleton is one
        external leg (its compression needs no Lie-element identity)."""

        return all(self._content(content)[1] for content in skeleton.contents)

    @staticmethod
    def _ring_word(kind: PrimitiveKind, descriptors) -> tuple[int, ...] | None:
        """The primitive word of one product of position terms around a ring."""

        if kind == "adjoint-ring":
            return tuple(a for _kind, word in descriptors for a in word)
        count_ = len(descriptors)
        strings = []
        for position, (_kind, word, forward) in enumerate(descriptors):
            in_label, out_label = ("loop", (position - 1) % count_), ("loop", position)
            row, col = (out_label, in_label) if forward else (in_label, out_label)
            strings.append(ColourString(row, word, col))
        try:
            opened, traces, empty = _join_strings(strings, lambda label: True)
        except ValueError:
            return None
        if opened or len(traces) + empty != 1:
            return None
        return traces[0] if traces else ()

    def certificate(
        self, combinations: Mapping[Skeleton, Combination]
    ) -> dict[PrimitiveKind, CompressionCertificate]:
        """Loop-level certificates of the compressions of ``combinations``.

        Per class: the smallest skeleton with single-leg blocks has its
        generic seed recomputed and compared exactly (a mismatch is an
        error). A class without such an instance, or a model whose
        all-adjoint tree vertices are not Lie-type, is not certified and
        keeps generic seeds.
        """

        trees = _lie_vertices(self.colour, self.colour.all_adjoint_rules())
        if trees is None:
            return {}  # pure-gluon blocks are not certified Lie elements
        instances: dict[PrimitiveKind, tuple] = {}
        for skeleton in combinations:
            if not self.single_legged(skeleton):
                continue
            kind = self.kind(skeleton)
            size = (len(skeleton.contents), repr(skeleton.contents))
            if kind not in instances or size < instances[kind][0]:
                instances[kind] = (size, skeleton)
        out = {}
        for kind, (_size, skeleton) in instances.items():
            generic = self.colour.seed(skeleton)
            compressed = _combination_tensor(combinations[skeleton], self.colour.legs)
            if canonical_seed(compressed) != generic:
                raise ArithmeticError(
                    f"{kind} compression differs from the generic seed of "
                    f"{skeleton.family}"
                )
            out[kind] = CompressionCertificate(
                kind,
                tuple(
                    key
                    for key, value in self._vertices.items()
                    if value is not None and kind in (key[-1], "quark-ring")
                ),
                trees,
                skeleton,
                COMPRESSION_THEOREMS[kind],
            )
        return out


def _combination_tensor(
    combination: Combination, legs: Sequence[ColorLeg]
) -> ConnectedColorTensor:
    terms = []
    for primitive, c in combination:
        terms.extend(
            TensorTerm(c * t.coefficient, t.tensor, t.nc_power)
            for t in primitive.tensor(legs).terms
        )
    return ConnectedColorTensor(tuple(legs), tuple(terms))


def _lie_vertices(
    colour: SeedColour, rules: Iterable[tuple[str, int]]
) -> tuple[tuple[str, int], ...] | None:
    """The all-adjoint tree vertex structures, each proved a Lie-type product
    (ad over its other legs), so pure-gluon tree currents are Lie elements;
    None if one is not."""

    out = []
    for rule, index in rules:
        network = colour._vertex[(rule, index)]
        count_ = len(colour._rules[rule]["particles"])
        legs = [("leg", i) for i in range(1, count_ + 1)]
        if (
            decompose_vertex(network, legs[0], legs[1], legs[2:], "adjoint-ring")
            is None
        ):
            return None
        out.append((rule, index))
    return tuple(out)


@dataclass(frozen=True)
class CompressionCertificate:
    """A loop-level certificate of one compression (design §7.2).

    ``vertices``: the loop vertex structures proved exact combinations of
    generator products (``decompose_vertex``); ``tree_vertices``: the
    all-adjoint tree vertices proved Lie-type; ``instance``: the smallest
    skeleton of the class with single-leg blocks, whose generic seed equals
    its compressed seed exactly; ``theorem``: the cited theorem.
    """

    kind: PrimitiveKind
    vertices: tuple[tuple, ...]
    tree_vertices: tuple[tuple[str, int], ...]
    instance: Skeleton
    theorem: str


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
        """<b|b'> as exact Laurent polynomials (the tree engine's Born metric).

        The external adjoint labels are summed, so <b|b'> = <pi b|pi b'> for
        every permutation pi of them. Row b is relabelled by the permutation
        that numbers its adjoint labels in reading order (``_adjoint_rename``),
        and each distinct relabelled pair is contracted once: a single-trace
        gluon basis of n legs contracts (n-1)! pairs instead of (n-1)!^2.
        """

        adjoint = sorted(
            leg.label for leg in self.legs if leg.representation == ADJOINT
        )
        cells: dict[tuple[Hashable, Hashable], Laurent] = {}
        rows = []
        for left in self.tensors:
            rename = _adjoint_rename(left, adjoint)
            key = _relabelled_key(left, rename)
            row = []
            for right in self.tensors:
                pair = (key, _relabelled_key(right, rename))
                found = cells.get(pair)
                if found is None:
                    found = cells[pair] = contract_colour(
                        _relabelled(left, rename), _relabelled(right, rename)
                    )
                row.append(found)
            rows.append(tuple(row))
        return tuple(rows)

    def degree(self, pairs: Iterable[SectorPair]) -> int | None:
        """Leading N_c degree of a Born squared-order sector (its own P)."""

        pairs = frozenset(pairs)
        return _max_degree(
            cell
            for b, row in enumerate(self.gram)
            for b2, cell in enumerate(row)
            if any((a, a2) in pairs for a in self.orders[b] for a2 in self.orders[b2])
        )


def contract_colour(left: ConnectedColorTensor, right: ConnectedColorTensor) -> Laurent:
    """``contract_connected_tensor_nc_terms`` (the exact Laurent coefficients
    of ``left.conjugate() * right``) with each literal overlap, a product of
    adjoint traces, counted natively: the subset sum over the labels that
    take the U(1) term of their Fierz identity, each term N_c to its index
    loops (``loop_v2::generate::cycles``). Products with too many labels for
    the subset sum keep the trace algebra of ``color/contraction_trace.py``,
    which is the oracle of both."""

    if left.legs != right.legs:
        raise ValueError(
            "bra and ket connections must have identical typed output legs"
        )
    power = left.normalization_power + right.normalization_power
    if power % 2:
        raise ValueError("bra/ket normalization has an odd inverse sqrt(2) power")
    normalization = Fraction(1, 2 ** (power // 2))
    pairs = [(a, b) for a in left.terms for b in right.terms]
    products = [overlap_traces(a.tensor, b.tensor) for a, b in pairs]
    counted = _trace_overlaps()(products)
    result: dict[int, ExactColorCoefficient] = {}
    for (a, b), laurent in zip(pairs, counted, strict=True):
        terms = (
            _literal_tensor_overlap_nc_terms(a.tensor, b.tensor).items()
            if laurent is None
            else ((p, Fraction(c)) for p, c in laurent)
        )
        coefficient = (a.coefficient.conjugate() * b.coefficient).scaled(normalization)
        for nc_power, factor in terms:
            exponent = nc_power + a.nc_power + b.nc_power
            result[exponent] = result.get(
                exponent, ExactColorCoefficient()
            ) + coefficient.scaled(factor)
    return {p: value for p, value in sorted(result.items()) if value}


@lru_cache(maxsize=1)
def _trace_overlaps() -> Callable[[list], list]:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native().trace_overlaps


ADJOINT = 8


def _adjoint_rename(
    tensor: ConnectedColorTensor, adjoint: Sequence[int]
) -> dict[int, int]:
    """The permutation of the adjoint labels ``adjoint`` (sorted) that numbers
    them in the order ``tensor`` reads them (terms, then open strings, then
    traces, as stored); labels it does not read keep the remaining ones."""

    order: list[int] = []
    seen: set[int] = set()
    members = set(adjoint)
    for term in tensor.terms:
        colour = term.tensor
        labels = [a for string in colour.open_strings for a in string.adjoint_labels]
        labels += [a for trace in colour.traces for a in trace]
        for label in labels:
            if label in members and label not in seen:
                seen.add(label)
                order.append(label)
    order += [label for label in adjoint if label not in seen]
    return dict(zip(order, adjoint, strict=True))


def _relabelled_key(
    tensor: ConnectedColorTensor, rename: Mapping[int, int]
) -> Hashable:
    """A key of ``_relabelled(tensor, rename)``: equal exactly for equal
    tensors (open strings and traces in canonical order and rotation)."""

    return tensor.normalization_power, frozenset(
        (term.coefficient, term.nc_power, _colour_key(term.tensor, rename))
        for term in tensor.terms
    )


def _colour_key(colour: ColorTensor, rename: Mapping[int, int]) -> Hashable:
    """A key of ``_relabelled_colour(colour, rename)``."""

    def label(a: int) -> int:
        return rename.get(a, a)

    return (
        tuple(
            sorted(
                (
                    string.fundamental_label,
                    tuple(map(label, string.adjoint_labels)),
                    string.antifundamental_label,
                )
                for string in colour.open_strings
            )
        ),
        tuple(
            sorted(
                _canonical_rotation(tuple(map(label, trace))) for trace in colour.traces
            )
        ),
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


# Powers below the leading degree P that each accuracy keeps (design §7.4).
_DEPTH = {"lc": 0, "nlc": 2}


def kept(power: int, degree: int | None, accuracy: ColourAccuracy) -> bool:
    """Whether a colour cell of N_c power ``power`` survives the projection
    of a sector of leading degree ``degree`` (P): LC keeps P, NLC keeps P,
    P-1 and P-2 (odd powers included), FC keeps every power."""

    if accuracy == "full":
        return True
    if accuracy not in _DEPTH:
        raise ValueError(f"unknown colour accuracy {accuracy!r}")
    return degree is not None and degree - _DEPTH[accuracy] <= power <= degree


def project(
    cell: Laurent, degree: int | None, accuracy: ColourAccuracy, shift: int = 0
) -> ExactColorCoefficient:
    """N_c^shift times one cell at N_c = 3 after LC/NLC/FC truncation at
    leading degree P (``shift`` is the N_c grade of the Born amplitude the
    cell multiplies)."""

    real = imag = Fraction(0)
    for power, value in cell.items():
        if kept(power + shift, degree, accuracy):
            factor = _nc_power(power + shift)
            real += value.real * factor
            imag += value.imag * factor
    return ExactColorCoefficient(real, imag)


@cache
def _nc_power(power: int) -> Fraction:
    return Fraction(NC) ** power


@dataclass(frozen=True)
class SeedTable:
    """Exact seed tables K[c, b] = <b|kappa_c> (design §7.1).

    ``classes`` are the distinct seed vectors (generic colour vectors or
    certified ``Primitive`` classes), ``orders[c]`` the loop amplitude orders
    of the skeletons using class c and ``cells[c][b]`` the Laurent
    coefficients of K. Squared-order sectors are given as their (Born order,
    loop order) pairs. ``formal_nc`` is false when the tree currents carry
    numeric U(1) pieces of mixed N_c grade (a DAG compiled without the grade
    split of ``generation.dag_grades``); then only full colour is exact.
    ``certificates`` holds the loop-level certificate of every compression
    the classes use.
    """

    basis: BornBasis
    classes: tuple[ConnectedColorTensor | Primitive, ...]
    orders: tuple[frozenset[Orders], ...]
    cells: tuple[tuple[Laurent, ...], ...]
    formal_nc: bool = True
    certificates: Mapping[PrimitiveKind, CompressionCertificate] = field(
        default_factory=dict
    )

    def tensor(self, c: int) -> ConnectedColorTensor:
        """The colour vector of class c (a primitive's SU(N_c) expansion)."""

        seed = self.classes[c]
        return seed.tensor(self.basis.legs) if isinstance(seed, Primitive) else seed

    def admitted(self, pairs: Iterable[SectorPair]):
        """(class, loop order, basis element, Born order) entering a sector."""

        pairs = frozenset(pairs)
        for c, loop_orders in enumerate(self.orders):
            for order in sorted(loop_orders):
                for b, born_orders in enumerate(self.basis.orders):
                    for born_order in sorted(born_orders):
                        if (born_order, order) in pairs:
                            yield c, order, b, born_order

    def degree(
        self,
        pairs: Iterable[SectorPair],
        born_grades: Mapping[tuple[int, Orders], Iterable[int]] | None = None,
    ) -> int | None:
        """P: the leading N_c degree of the sector's nonzero cells (§7.4).

        ``born_grades`` lists the N_c grades of the Born amplitude of each
        (basis element, Born order); by default every amplitude has grade 0.
        """

        return max(
            (
                power + grade
                for c, _o, b, a in self.admitted(pairs)
                for power, value in self.cells[c][b].items()
                if value
                for grade in (
                    (0,) if born_grades is None else born_grades.get((b, a), ())
                )
            ),
            default=None,
        )

    def class_degree(self, c: int) -> int | None:
        return _max_degree(self.cells[c])

    def certified(self, combination: SeedCombination) -> bool:
        """Whether a seed uses certified primitive classes only.

        This is the colour admissibility of schedule M at NLC and full colour
        (design §2.2, ``ReferenceExecutor.run(certified=...)``): every class
        is a ``Primitive`` whose compression carries a loop-level certificate
        in this table. A generic seed merged into such a class has exactly its
        colour vector and counts as certified.
        """

        return bool(combination) and all(
            isinstance(seed := self.classes[c], Primitive)
            and seed.kind in self.certificates
            for c, _x in combination
        )

    def seeds(
        self,
        pairs: Iterable[SectorPair],
        born: Mapping[tuple[int, Orders], Mapping[int, Any]],
        accuracy: ColourAccuracy = "full",
        *,
        cast: Callable[[ExactColorCoefficient], Any] = complex,
        conjugate: Callable[[Any], Any] = lambda value: value.conjugate(),
        degree: int | None = None,
        skip: Callable[[int, int], bool] | None = None,
    ) -> dict[tuple[int, Orders], Any]:
        """U0[c, loop order] = sum_b K[c, b] conj(A0[b, Born order]) (§7.3).

        ``born`` maps (basis element, Born order) to one helicity's Born
        amplitude split into N_c grades, ``{grade: amplitude}`` in any
        carrier: A0[b] = sum_g N_c^g A0[b, g]. ``cast`` converts exact cells
        into the carrier. The projection acts on the products K[c, b] N_c^g
        (N_c real, so conjugation leaves the grades alone). No external
        average is applied. ``degree`` overrides P, e.g. when the sector's
        counterterm and field cells raise it; by default it is the degree of
        the seed cells with the grades of ``born``. ``skip(c, b)`` leaves
        out the cells another contraction covers (``SeedFourier``).
        """

        pairs = frozenset(pairs)
        if accuracy != "full":
            if not self.formal_nc:
                raise ValueError(
                    f"{accuracy.upper()} seeds need N_c-exact tree currents; these "
                    "blocks mix U(1)-subtraction grades (a DAG compiled without "
                    "generation.dag_grades.split_u1_grades; design §3.2)"
                )
            if degree is None:
                degree = self.degree(
                    pairs, {key: tuple(value) for key, value in born.items()}
                )
        if accuracy == "full":
            # Every grade is kept: sum the Born at N_c = 3 first.
            born = {key: {0: _at_three(grades, cast)} for key, grades in born.items()}
        out: dict[tuple[int, Orders], Any] = {}
        for c, order, b, born_order in self.admitted(pairs):
            if skip is not None and skip(c, b):
                continue
            for grade, amplitude in born.get((b, born_order), {}).items():
                weight = project(self.cells[c][b], degree, accuracy, grade)
                if not weight:
                    continue
                term = cast(weight) * conjugate(amplitude)
                key = (c, order)
                out[key] = term if key not in out else out[key] + term
        return out


def _at_three(
    grades: Mapping[int, Any], cast: Callable[[ExactColorCoefficient], Any]
) -> Any:
    """sum_g 3^g A0[b, g] in the carrier (grades in their given order)."""

    total = None
    for grade, amplitude in grades.items():
        value = (
            amplitude
            if grade == 0
            else amplitude * cast(ExactColorCoefficient(Fraction(NC) ** grade))
        )
        total = value if total is None else total + value
    return total


SeedCombination = tuple[tuple[int, ExactColorCoefficient], ...]


def _native_seeds() -> Any:
    """The native seed stage module when it is switched on (WP3.1 B; the
    one flag ``native_seeds.NATIVE``), else None."""

    from . import native_seeds

    return native_seeds if native_seeds.NATIVE else None


def build_seed_table(
    colour: SeedColour,
    skeletons: Iterable[Skeleton],
    basis: BornBasis | None = None,
    *,
    compress: bool = False,
) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
    """Seed classes of ``skeletons`` and their exact table K.

    Returns the table and each skeleton's seed as a combination
    sum_c x_c kappa_c of table classes: one generic class with weight one,
    or with ``compress`` a combination of certified primitives
    (``CompressedColour``; each compression's loop-level certificate is
    checked on its smallest instance, ``SeedTable.certificates``). Skeletons
    outside every certified class keep their generic seed. ``basis``
    defaults to the tree engine's Born basis (``born_basis``); any set of
    literal colour tensors whose amplitudes the caller provides works.

    Runs natively (``native_seeds``, WP3.1 B) unless ``native_seeds.NATIVE``
    is off: this Python stage is the native stage's oracle.
    """

    if (native := _native_seeds()) is not None:
        return native.build_seed_table(colour, skeletons, basis, compress=compress)
    if basis is None:
        basis = born_basis(colour.blocks)
    skeletons = list(skeletons)
    table = SeedTable(basis, (), (), (), colour.formal_nc)
    compressed = CompressedColour(colour) if compress else None
    combinations: dict[Skeleton, Combination] = {}
    if compressed is not None:
        for skeleton in skeletons:
            combination = compressed.combination(skeleton)
            if combination is not None:
                combinations[skeleton] = combination
        certificates = compressed.certificate(combinations)
        table = replace(table, certificates=certificates)
        combinations = {
            skeleton: combination
            for skeleton, combination in combinations.items()
            if compressed.kind(skeleton) in certificates
        }
    return _extended(table, {}, colour, skeletons, compressed, combinations)


def extend_seed_table(
    table: SeedTable,
    class_of: Mapping[Skeleton, SeedCombination],
    colour: SeedColour,
    skeletons: Iterable[Skeleton],
    *,
    compress: bool = False,
) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
    """``table`` and ``class_of`` with the seeds of further skeletons (e.g.
    new readings of the same skeletons). Existing classes keep their index;
    compressions are used only where ``table`` holds their certificate.
    Native like ``build_seed_table``."""

    if (native := _native_seeds()) is not None:
        return native.extend_seed_table(
            table, class_of, colour, skeletons, compress=compress
        )
    return _extended(
        table,
        class_of,
        colour,
        [s for s in skeletons if s not in class_of],
        CompressedColour(colour) if compress else None,
        {},
    )


def _extended(
    table: SeedTable,
    class_of: Mapping[Skeleton, SeedCombination],
    colour: SeedColour,
    skeletons: Sequence[Skeleton],
    compressed: CompressedColour | None,
    combinations: Mapping[Skeleton, Combination],
) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
    basis = table.basis
    # With compression, classes are also merged by proportional tensors
    # (quark rings of different arc words with one colour vector, and
    # generic seeds equal to them), so each colour vector is seeded once.
    keys = _ClassKeys(basis.legs, merge=compressed is not None)
    index: dict[Hashable, int] = {}
    scales: list[ExactColorCoefficient] = []
    classes: list[ConnectedColorTensor | Primitive] = list(table.classes)
    for c, seed in enumerate(classes):
        key, scale = keys(seed)
        index.setdefault(key, c)
        scales.append(scale)
    orders: list[set[Orders]] = [set(row) for row in table.orders]
    class_of = dict(class_of)

    # x of a seed against its class's scale, by (scale, class scale).
    ratios: dict[tuple[ExactColorCoefficient, ExactColorCoefficient], Any] = {}

    def row(seed: Hashable, order: Orders) -> tuple[int, ExactColorCoefficient]:
        """(class, x) with seed = x kappa_class."""

        key, scale = keys(seed)
        c = index.get(key)
        if c is None:
            c = index[key] = len(classes)
            classes.append(seed)
            scales.append(scale)
            orders.append(set())
        orders[c].add(order)
        pair = (scale, scales[c])
        ratio = ratios.get(pair)
        if ratio is None:
            ratio = ratios[pair] = _ONE if pair[0] == pair[1] else _ratio(*pair)
        return c, ratio

    # Rows of shared (memoised) combinations, by identity and loop order.
    rows: dict[tuple[int, Orders], tuple[SeedCombination, Combination]] = {}
    for skeleton in skeletons:
        combination = combinations.get(skeleton)
        if (
            combination is None
            and compressed is not None
            and compressed.kind(skeleton) in table.certificates
        ):
            combination = compressed.combination(skeleton)
        if combination is None:
            combination = ((colour.seed(skeleton), _ONE),)
        shared = rows.get((id(combination), skeleton.orders))
        if shared is not None and shared[1] is combination:
            class_of[skeleton] = shared[0]
            continue
        merged: dict[int, ExactColorCoefficient] = {}
        for seed, x in combination:
            c, factor = row(seed, skeleton.orders)
            term = x if factor is _ONE else x * factor
            merged[c] = merged[c] + term if c in merged else term
        class_of[skeleton] = tuple(merged.items())
        rows[id(combination), skeleton.orders] = (class_of[skeleton], combination)
    classes = tuple(classes)
    cells = list(table.cells)
    cells.extend(seed_cells(basis, classes[len(cells) :]))
    return (
        replace(
            table,
            classes=classes,
            orders=tuple(map(frozenset, orders)),
            cells=tuple(cells),
        ),
        class_of,
    )


def seed_cells(
    basis: BornBasis,
    classes: Sequence[ConnectedColorTensor | Primitive],
    primitive_cells: _PrimitiveCells | None = None,
) -> tuple[tuple[Laurent, ...], ...]:
    """The cells <b|kappa_c> of ``classes`` against every tensor b of
    ``basis``: certified primitives by integer cycle counting
    (``_PrimitiveCells``, or the given one, whose caches it extends), generic
    classes term-factored (design §7.2: <b|t> once per distinct structure t).
    The tensors of ``basis`` may be any literal tensors on the Born legs,
    e.g. the operator images of a directed colour history (design §7.7)."""

    from . import native_cells  # WP3.1 C1 switch (rule 1): the "cells" stage

    if "cells" in native_cells.NATIVE_STAGES:
        return native_cells.switched_seed_cells(basis, classes, primitive_cells)
    if primitive_cells is None:
        primitive_cells = _PrimitiveCells(basis)
    elif primitive_cells.basis is not basis:
        raise ValueError("primitive cells of another basis")
    overlaps: dict[ColorTensor, tuple[Laurent, ...]] = {}
    out = []
    for seed in classes:
        if isinstance(seed, Primitive):
            out.append(primitive_cells(seed))
            continue
        if seed.normalization_power:
            raise ValueError("a seed class carries an irrational sqrt(2) factor")
        row_cells: list[dict[int, ExactColorCoefficient]] = [{} for _ in basis.tensors]
        for term in seed.terms:
            if term.tensor not in overlaps:
                ket = ConnectedColorTensor(basis.legs, (TensorTerm(_ONE, term.tensor),))
                overlaps[term.tensor] = tuple(
                    contract_colour(b, ket) for b in basis.tensors
                )
            for cell, overlap in zip(row_cells, overlaps[term.tensor], strict=True):
                for power, value in overlap.items():
                    key = power + term.nc_power
                    cell[key] = cell.get(key, ExactColorCoefficient()) + (
                        term.coefficient * value
                    )
        out.append(
            tuple(
                {power: value for power, value in cell.items() if value}
                for cell in row_cells
            )
        )
    return tuple(out)


class _ClassKeys:
    """Table keys of seed classes: (key, scale) with seed = scale x the
    class of that key. Adjoint rings and fundamental traces are keyed by
    their canonical word; with ``merge`` the other classes are keyed by
    their tensor up to a factor (``_proportional_key``)."""

    def __init__(self, legs: Sequence[ColorLeg], merge: bool):
        self.legs = tuple(legs)
        self.merge = merge
        self._adjoint = frozenset(leg.label for leg in legs if leg.representation == 8)
        self._cache: dict[Hashable, tuple[Hashable, ExactColorCoefficient]] = {}

    def __call__(self, seed) -> tuple[Hashable, ExactColorCoefficient]:
        if not self.merge or (
            isinstance(seed, Primitive) and seed.kind != "quark-ring"
        ):
            return seed, _ONE
        found = self._cache.get(seed)
        if found is None:
            if isinstance(seed, Primitive):
                # The key of the relabelled canonical ring (``Primitive.tensor``).
                canonical, back = _ring_relabelling(seed, self._adjoint)
                tensor = canonical_seed(_reduced_primitive(canonical, self.legs))
                found = _proportional_key(tensor, back)
            else:
                found = _proportional_key(canonical_seed(seed))
            self._cache[seed] = found
        return found


def _proportional_key(
    tensor: ConnectedColorTensor, rename: Mapping[int, int] | None = None
) -> tuple[Hashable, ExactColorCoefficient]:
    """(key, s): tensor = s x T_key, equal keys for proportional tensors
    (terms are canonically ordered, the first one is scaled to 1).

    ``rename`` gives the key of ``_relabelled(tensor, rename)`` without
    building it: a bijective relabelling merges no terms, so re-sorting the
    relabelled terms restores the canonical order.
    """

    if not tensor.terms:
        return ("zero", tensor.legs), _ONE
    terms = [(term.tensor, term.nc_power, term.coefficient) for term in tensor.terms]
    if rename:
        terms = sorted(
            ((_relabelled_colour(t, rename), power, c) for t, power, c in terms),
            key=lambda term: term[:2],
        )
    first = terms[0][2]
    inverse = _ratio(_ONE, first)
    return (
        tensor.legs,
        tensor.normalization_power,
        tuple((t, power, c * inverse) for t, power, c in terms),
    ), first


def _ratio(
    numerator: ExactColorCoefficient, denominator: ExactColorCoefficient
) -> ExactColorCoefficient:
    norm = (denominator * denominator.conjugate()).real
    return (numerator * denominator.conjugate()).scaled(1 / norm)


class _PrimitiveCells:
    """K[P, b] of certified primitives, by integer cycle counting.

    An adjoint ring is U(1)-free in every external label (ad 1 = 0), so its
    SU(N_c) overlaps are the U(N_c) ones of its expansion
    (``Primitive.un_terms``): N_c to the number of index loops
    (``face_count``) of each term. Fundamental traces and quark rings use the
    exact SU(N_c) contraction. Overlaps depend on the labels only through the
    relabelling that sends the primitive word to 1..n, so they are cached on
    the relabelled basis tensor (single-trace bases: (n-1)! distinct ones).
    """

    def __init__(self, basis: BornBasis):
        self.basis = basis
        self._cache: dict[Hashable, Laurent] = {}
        self._adjoint = frozenset(
            leg.label for leg in basis.legs if leg.representation == 8
        )
        # Quark rings: relabelled Born tensors and canonical rings by integer id.
        self._bras: list[ConnectedColorTensor] = []
        self._bra_ids: dict[ConnectedColorTensor, int] = {}
        self._relabellings: dict[tuple, tuple[int, ...]] = {}
        self._rings: dict[Primitive, int] = {}
        self._ring_cells: dict[tuple[int, int], Laurent] = {}
        # The native twin (``native``): the first one built is the root, held
        # in a list that the rebased copies share, and the others are rebased
        # from it, so their native caches are shared both ways.
        self._native_root: list[Any] = []
        self._native: Any = None

    def native(self) -> Any:
        """The native twin of these cells (``native_cells.NativeCells``) on
        this basis, sharing its caches with every rebased copy's twin."""

        if self._native is None:
            from .native_cells import NativeCells

            if not self._native_root:
                self._native_root.append(NativeCells(self.basis))
            root = self._native_root[0]
            same = root.basis is self.basis
            self._native = root if same else root.rebased(self.basis)
        return self._native

    def rebased(self, basis: BornBasis) -> _PrimitiveCells:
        """The cells against another basis on the same legs, sharing this
        one's caches: cached overlaps are keyed by the relabelled tensors
        themselves, not by their position in a basis (only the relabelling
        index of the basis is rebuilt)."""

        if basis.legs != self.basis.legs:
            raise ValueError("rebased primitive cells need the same colour legs")
        other = copy(self)
        other.basis = basis
        other._relabellings = {}
        other._native = None
        return other

    def __call__(self, primitive: Primitive) -> tuple[Laurent, ...]:
        if primitive.kind == "quark-ring":
            # <b|back(P')> = <back^-1(b)|P'> on the canonical ring P'.
            canonical, back = _ring_relabelling(primitive, self._adjoint)
            ring = self._rings.setdefault(canonical, len(self._rings))
            ket = None
            out = []
            for bra in self._relabelled_bras({b: a for a, b in back.items()}):
                found = self._ring_cells.get((ring, bra))
                if found is None:
                    if ket is None:
                        ket = canonical.tensor(self.basis.legs)
                    found = self._ring_cells[ring, bra] = contract_colour(
                        self._bras[bra], ket
                    )
                out.append(found)
            return tuple(out)
        rename = {label: i for i, label in enumerate(primitive.word, start=1)}
        return tuple(self._cell(primitive, b, rename) for b in self.basis.tensors)

    def _relabelled_bras(self, forward: Mapping[int, int]) -> tuple[int, ...]:
        """Ids of the Born tensors with their adjoint labels renamed."""

        key = tuple(sorted(forward.items()))
        found = self._relabellings.get(key)
        if found is None:
            ids = []
            for bra in self.basis.tensors:
                bra = _relabelled(bra, forward) if forward else bra
                index = self._bra_ids.setdefault(bra, len(self._bras))
                if index == len(self._bras):
                    self._bras.append(bra)
                ids.append(index)
            found = self._relabellings[key] = tuple(ids)
        return found

    def _cell(self, primitive: Primitive, bra: ConnectedColorTensor, rename) -> Laurent:
        key = _relabel_key(bra, rename)
        if key is None:
            return self._overlap(primitive.kind, primitive.word, bra)
        key = (primitive.kind, len(primitive.word), key)
        found = self._cache.get(key)
        if found is None:
            word = tuple(range(1, len(primitive.word) + 1))
            found = self._cache[key] = self._overlap(
                primitive.kind, word, _relabel_tensor(bra, rename)
            )
        return found

    @staticmethod
    def _overlap(
        kind: PrimitiveKind, word: tuple[int, ...], bra: ConnectedColorTensor
    ) -> Laurent:
        primitive = Primitive(kind, word)
        if kind == "fundamental-trace":
            return contract_colour(bra, primitive.tensor(bra.legs))
        if bra.normalization_power % 2:
            raise ValueError("a Born tensor has an odd inverse sqrt(2) power")
        scale = Fraction(1, 2 ** (bra.normalization_power // 2))
        out: dict[int, ExactColorCoefficient] = {}
        for sign, traces, power in primitive.un_terms():
            # Literal traces list tr(tau^w_k .. tau^w_1) (module docstring).
            ket = ColorTensor((), tuple(tuple(reversed(t)) for t in traces))
            for term in bra.terms:
                exponent = power + term.nc_power + face_count(term.tensor, ket)
                out[exponent] = out.get(exponent, ExactColorCoefficient()) + (
                    term.coefficient.conjugate().scaled(scale * sign)
                )
        return {p: value for p, value in out.items() if value}


def _relabel_key(tensor: ConnectedColorTensor, rename: Mapping[int, int]) -> Hashable:
    """A cheap key of ``_relabel_tensor(tensor, rename)`` (None if unmapped)."""

    if any(leg.representation not in (1, 8) for leg in tensor.legs):
        return None
    try:
        return tensor.normalization_power, tuple(
            (
                term.coefficient,
                term.nc_power,
                tuple(
                    sorted(
                        _canonical_rotation(tuple(rename[a] for a in trace))
                        for trace in term.tensor.traces
                    )
                ),
            )
            for term in tensor.terms
        )
    except KeyError:
        return None


def _canonical_rotation(trace: tuple[int, ...]) -> tuple[int, ...]:
    if not trace:
        return trace
    i = trace.index(min(trace))
    return trace[i:] + trace[:i]


def _relabel_tensor(
    tensor: ConnectedColorTensor, rename: Mapping[int, int]
) -> ConnectedColorTensor | None:
    """``tensor`` with adjoint labels renamed, or None if one is not mapped."""

    labels = {leg.label for leg in tensor.legs if leg.representation == 8}
    if not labels <= set(rename) or any(
        leg.representation not in (1, 8) for leg in tensor.legs
    ):
        return None
    legs = tuple(
        ColorLeg(rename[leg.label], 8) for leg in tensor.legs if leg.representation == 8
    )
    return ConnectedColorTensor(
        legs,
        tuple(
            TensorTerm(
                term.coefficient,
                ColorTensor(
                    (),
                    tuple(
                        tuple(rename[label] for label in trace)
                        for trace in term.tensor.traces
                    ),
                ),
                term.nc_power,
            )
            for term in tensor.terms
        ),
        tensor.normalization_power,
    )


# ---------------------------------------------------------- seed FFT (§7.3)

ContractionPolicy = Literal["auto", "direct", "fft"]
SYMMETRIC_GROUP_FFT = "symmetric-group-fft"


@dataclass(frozen=True)
class SeedFourier:
    """Certified symmetric-group orbits of a seed table (design §7.3).

    S_m acts on the external adjoint labels ``labels`` by x: labels[i] ->
    labels[x[i]]. Group elements are permutation tuples in lexicographic
    order (``itertools.permutations``), composed as (p o q)[i] = p[q[i]];
    element 0 is the identity.

    * ``born_orbits[a][g]`` is the basis element g.b_a. Born orbits are
      regular: every element of the orbit appears once.
    * ``loop_orbits[j][g]`` is (class, sign) with g.kappa_j = sign
      kappa_class. Stabilisers are allowed (an adjoint ring is fixed up to
      (-1)^n by the reflection of its word).
    * ``kernels[a, j][x]`` = sign_x K[class(x), b_a] for the certified pairs.

    Each kernel is certified on every entry, K[class(g), b(h)] sign_g =
    k(h^-1 o g). Hence sum_h K[c, b(h)] conj(A0_b(h)) over a pair is the group
    convolution that ``V2SeedFourier`` evaluates with the native S_m FFT, and
    it equals the direct contraction exactly before rounding. Every other
    (class, basis element) pair stays direct (``covers``).
    """

    degree: int
    labels: tuple[int, ...]
    born_orbits: tuple[tuple[int, ...], ...]
    loop_orbits: tuple[tuple[tuple[int, int], ...], ...]
    kernels: Mapping[tuple[int, int], tuple[Laurent, ...]]
    # Projected kernels per (pair, P, accuracy, grade) and native plans per
    # job: both are point-independent, so they are built once.
    _projected: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @cached_property
    def _orbit_of(self) -> tuple[dict[int, int], dict[int, int]]:
        born = {b: a for a, orbit in enumerate(self.born_orbits) for b in orbit}
        loop = {c: j for j, orbit in enumerate(self.loop_orbits) for c, _s in orbit}
        return born, loop

    def covers(self, c: int, b: int) -> bool:
        """Whether K[c, b] is contracted by a kernel, not directly."""

        born, loop = self._orbit_of
        a, j = born.get(b), loop.get(c)
        return a is not None and j is not None and (a, j) in self.kernels

    @cached_property
    def representatives(self) -> tuple[tuple[tuple[int, int, int], ...], ...]:
        """Per loop orbit: (class, first group element, sign) of each class."""

        out = []
        for orbit in self.loop_orbits:
            first: dict[int, tuple[int, int, int]] = {}
            for g, (c, sign) in enumerate(orbit):
                first.setdefault(c, (c, g, sign))
            out.append(tuple(first.values()))
        return tuple(out)

    def seeds(
        self,
        table: SeedTable,
        pairs: Iterable[SectorPair],
        rows: Sequence[Mapping[tuple[int, Orders], Mapping[int, complex]]],
        accuracy: ColourAccuracy,
        native: Any,
        *,
        degree: int | None = None,
    ) -> list[dict[tuple[int, Orders], complex]]:
        """``SeedTable.seeds`` in binary64 for a batch of Born rows (one per
        helicity): the certified pairs through the native S_m FFT, the rest
        directly. Kernels are the exact cells projected like the direct
        ones (per sector degree P and Born grade), rounded once; projections
        and native plans (whose constructor transforms the kernels) are kept
        across calls.
        """

        pairs = frozenset(pairs)
        if accuracy != "full" and degree is None:
            grades: dict[tuple[int, Orders], set[int]] = {}
            for row in rows:
                for key, value in row.items():
                    grades.setdefault(key, set()).update(value)
            degree = table.degree(pairs, grades)
        if accuracy == "full":
            # Every grade is kept: sum the Born at N_c = 3 first.
            rows = [
                {key: {0: _at_three(grades, complex)} for key, grades in row.items()}
                for row in rows
            ]
        out = [
            table.seeds(
                pairs, row, accuracy, cast=complex, degree=degree, skip=self.covers
            )
            for row in rows
        ]
        if not rows:
            return out
        # One native transform per (Born order, loop order, Born grade).
        jobs: dict[tuple[Orders, Orders, int], list] = {}
        for (a, j), kernel in self.kernels.items():
            born_orders = table.basis.orders[self.born_orbits[a][0]]
            loop_orders = table.orders[self.loop_orbits[j][0][0]]
            for born_order, loop_order in sorted(pairs):
                if born_order not in born_orders or loop_order not in loop_orders:
                    continue
                present = {
                    grade
                    for row in rows
                    for b in self.born_orbits[a]
                    for grade in row.get((b, born_order), ())
                }
                for grade in sorted(present):
                    projection = (a, j, degree, accuracy, grade)
                    values = self._projected.get(projection)
                    if values is None:
                        values = self._projected[projection] = [
                            complex(project(cell, degree, accuracy, grade))
                            for cell in kernel
                        ]
                    if any(values):
                        jobs.setdefault((born_order, loop_order, grade), []).append(
                            (a, j, values)
                        )
        group = len(self.born_orbits[0]) if self.born_orbits else 0
        width = len(self.born_orbits) * group
        for job, items in sorted(jobs.items()):
            born_order, loop_order, grade = job
            plan = (job, degree, accuracy, tuple((a, j) for a, j, _v in items))
            fourier = self._plans.get(plan)
            if fourier is None:
                fourier = self._plans[plan] = native.V2SeedFourier(
                    self.degree,
                    [[(v.real, v.imag) for v in values] for _a, _l, values in items],
                    [(k, a, j) for k, (a, j, _v) in enumerate(items)],
                    len(self.born_orbits),
                    len(self.loop_orbits),
                )
            values = [
                [repr(x.real), repr(x.imag)]
                for row in rows
                for orbit in self.born_orbits
                for x in (
                    complex(row.get((b, born_order), {}).get(grade, 0)) for b in orbit
                )
            ]
            batch = native.NativeBatch.from_decimal(
                values, len(rows), width, "double", None, list(range(len(rows)))
            )
            weights = fourier.weights(batch).to_decimal()
            loop_width = len(self.loop_orbits) * group
            for j in sorted({j for _a, j, _v in items}):
                for c, g, sign in self.representatives[j]:
                    key = (c, loop_order)
                    for r, seeds in enumerate(out):
                        real, imag = weights[r * loop_width + j * group + g]
                        term = sign * complex(float(real), float(imag))
                        seeds[key] = term if key not in seeds else seeds[key] + term
        return out


def seed_fourier(table: SeedTable) -> SeedFourier | None:
    """The certified symmetric-group orbits of ``table`` (None if none).

    The group permutes every external adjoint label, keeping the smallest
    one fixed when every coloured leg is adjoint (cyclic traces then form
    regular orbits). Born orbits must be regular and keep their Born orders;
    loop orbits must be complete and keep their loop orders. A Born orbit
    needs |S_m| distinct basis elements, so larger groups are never
    enumerated and the contraction stays direct (design §7.8).
    """

    basis = table.basis
    adjoint = sorted(leg.label for leg in basis.legs if leg.representation == 8)
    if all(leg.representation in (1, 8) for leg in basis.legs):
        adjoint = adjoint[1:]
    labels, degree = tuple(adjoint), len(adjoint)
    if degree < 2 or math.factorial(degree) > len(basis):
        return None
    group = list(permutations(range(degree)))
    renames = [{label: labels[x[i]] for i, label in enumerate(labels)} for x in group]

    born_index = {tensor: b for b, tensor in enumerate(basis.tensors)}
    born_orbits: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for b0, tensor in enumerate(basis.tensors):
        if b0 in seen:
            continue
        images = [born_index.get(_relabelled(tensor, r)) for r in renames]
        seen.update(i for i in images if i is not None)
        if (
            None not in images
            and len(set(images)) == len(group)
            and all(basis.orders[b] == basis.orders[b0] for b in images)
        ):
            born_orbits.append(tuple(images))
    if not born_orbits:
        return None

    keys = _ClassKeys(basis.legs, merge=True)
    class_index: dict[Hashable, int] = {}
    scales = []
    for c, seed in enumerate(table.classes):
        key, scale = keys(seed)
        class_index.setdefault(key, c)
        scales.append(scale)
    loop_orbits: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for c0, seed in enumerate(table.classes):
        if c0 in seen:
            continue
        words = isinstance(seed, Primitive) and seed.kind != "quark-ring"
        tensor = None if words else canonical_seed(table.tensor(c0))
        images = []
        for rename in renames:
            if words:
                key, sign = canonical_primitive(
                    seed.kind, (rename.get(a, a) for a in seed.word)
                )
                scale = ExactColorCoefficient(sign)
            else:
                key, scale = _proportional_key(tensor, rename)
            c = class_index.get(key)
            if c is None or table.orders[c] != table.orders[c0]:
                break
            # rename(kappa_c0) = factor kappa_c; orbits keep factors +-1.
            factor = _ratio(scale, scales[c])
            if factor.imag or abs(factor.real) != 1:
                break
            images.append((c, int(factor.real)))
        seen.update(c for c, _sign in images)
        if len(images) == len(group):
            loop_orbits.append(tuple(images))
    if not loop_orbits:
        return None

    rank = {x: i for i, x in enumerate(group)}
    inverse = [tuple(sorted(range(degree), key=x.__getitem__)) for x in group]
    # relative[h][g] = rank of h^-1 o g
    relative = [[rank[tuple(hinv[i] for i in g)] for g in group] for hinv in inverse]
    kernels: dict[tuple[int, int], tuple[Laurent, ...]] = {}
    for a, born in enumerate(born_orbits):
        for j, loop in enumerate(loop_orbits):
            kernel = tuple(_signed(table.cells[c][born[0]], sign) for c, sign in loop)
            if all(
                _signed(table.cells[c][b], sign) == kernel[relative[h][g]]
                for h, b in enumerate(born)
                for g, (c, sign) in enumerate(loop)
            ):
                kernels[a, j] = kernel
    if not kernels:
        return None
    return SeedFourier(degree, labels, tuple(born_orbits), tuple(loop_orbits), kernels)


def _signed(cell: Laurent, sign: int) -> Laurent:
    if sign == 1:
        return dict(cell)
    return {power: value.scaled(sign) for power, value in cell.items()}


def _relabelled_colour(tensor: ColorTensor, rename: Mapping[int, int]) -> ColorTensor:
    """A literal structure with its adjoint labels renamed."""

    def label(a: int) -> int:
        return rename.get(a, a)

    return ColorTensor(
        tuple(
            OpenColorString(
                string.fundamental_label,
                tuple(map(label, string.adjoint_labels)),
                string.antifundamental_label,
            )
            for string in tensor.open_strings
        ),
        tuple(tuple(map(label, trace)) for trace in tensor.traces),
    )


def _relabelled(
    tensor: ConnectedColorTensor, rename: Mapping[int, int]
) -> ConnectedColorTensor:
    """``tensor`` with its adjoint labels renamed (fundamental ones kept)."""

    return ConnectedColorTensor(
        tensor.legs,
        tuple(
            TensorTerm(
                term.coefficient,
                _relabelled_colour(term.tensor, rename),
                term.nc_power,
            )
            for term in tensor.terms
        ),
        tensor.normalization_power,
    )


# ------------------------------------------------------ planar leading colour
#
# The structural face rule of design §7.5 for colour-ordered skeletons. The
# slice's skeletons are not colour-ordered yet, so production LC/NLC pruning
# is ``LoopSlice._prune_colour`` on exact seed cells; this rule (and its flow
# helpers) serves the Appendix B cost model and its tests until the
# colour-ordered skeletons of schedule M (WP2.8) adopt it.

# (strings q -> adjoints -> qbar, sorted; traces, canonical rotation, sorted)
FlowKey = tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]


def outer_flow(
    labels: Sequence[int], representations: Mapping[int, int]
) -> FlowKey | None:
    """The colour flow of a cyclic colour word read forward (design §7.5).

    ``representations`` are outgoing colour representations (3 a quark,
    -3 an antiquark, 8 an adjoint, 1 a singlet, skipped). A string runs from
    a quark through the following adjoints to the next antiquark; adjoints
    between an antiquark and the next quark close into one trace (all of
    them for an all-adjoint word). None if a quark follows a quark before an
    antiquark (no planar flow).
    """

    coloured = [a for a in labels if representations[a] != 1]
    n = len(coloured)
    if not any(abs(representations[a]) == 3 for a in coloured):
        return ((), (_canonical_rotation(tuple(coloured)),) if coloured else ())
    strings, used = [], set()
    for i, label in enumerate(coloured):
        if representations[label] != 3:
            continue
        string = [label]
        j = (i + 1) % n
        while True:
            other = coloured[j]
            if representations[other] == 8:
                string.append(other)
            elif representations[other] == -3:
                string.append(other)
                break
            else:
                return None
            j = (j + 1) % n
        strings.append(tuple(string))
        used.update(string)
    leftover = tuple(a for a in coloured if a not in used)
    return (
        tuple(sorted(strings)),
        (_canonical_rotation(leftover),) if leftover else (),
    )


def sector_flow(sector: Any) -> FlowKey:
    """The ``outer_flow`` key of a tree-engine colour sector."""

    strings = tuple(sorted(line.coloured_labels for line in sector.open_color_lines))
    traces = (
        (_canonical_rotation(tuple(sector.trace_labels)),)
        if sector.trace_labels
        else ()
    )
    return strings, traces


def leading_born_flows(born_dag: Any) -> frozenset[FlowKey]:
    """Colour flows of the Born's leading sectors (design §7.5).

    A Born root's N_c grade is minus the number of U(1)-subtraction pieces
    of its trees (``blocks.u1_grades``); the leading sectors are those whose
    roots reach the largest grade.
    """

    from .blocks import u1_grades

    grades = u1_grades(born_dag)
    best: dict[int, int] = {}
    for root in born_dag.amplitude_roots:
        left, right = grades[root.left_id], grades[root.right_id]
        if left is None or right is None:
            raise ValueError("a Born root mixes N_c grades")
        sector = root.color_sector_id or 0
        best[sector] = max(best.get(sector, left + right), left + right)
    top = max(best.values())
    by_id = {sector.id: sector for sector in born_dag.color_plan.sectors}
    return frozenset(
        sector_flow(by_id[sector]) for sector, grade in best.items() if grade == top
    )


class PlanarLeadingColour:
    """The face rule of design §7.5 on colour-ordered (planar) skeletons.

    Skeleton contents carry colour words over letters (leg masks, as
    ``skeletons.PlanarWords``: a coloured leg, or a group of colour-singlet
    legs). A skeleton reaches the leading degree P iff, read forward or
    backward, (a) its word's outer flow is a leading Born flow
    (``leading_born_flows``) with every singlet letter in a colourless gap
    (an antiquark followed by a quark); and (b) its loop face is bounded by
    colour lines only: no closed quark loop, no coloured block attached
    inside a fermion arc, and every quark propagator in a colourless gap of
    that reading. Closed quark loops have degree P-1 and the other classes
    P-2 or less, which the exact seed tables certify (``certify_leading_
    colour``); the rule only selects, before any numerics.
    """

    def __init__(
        self,
        legs: Sequence[ExternalLeg],
        catalogue: Any,
        leading: Iterable[FlowKey],
    ):
        self.catalogue = catalogue
        self.leading = frozenset(leading)
        # Outgoing representations: legs carry the incoming field.
        self.representations = {
            leg.index + 1: (-leg.colour if abs(leg.colour) == 3 else leg.colour)
            for leg in legs
        }

    def _letter(self, letter: int) -> int | None:
        """The coloured leg label of a letter; None for a singlet letter."""

        labels = [i + 1 for i in range(letter.bit_length()) if letter >> i & 1]
        coloured = [a for a in labels if self.representations[a] != 1]
        if not coloured:
            return None
        if len(labels) != 1:
            raise ValueError(f"letter {letter:#b} groups coloured legs")
        return coloured[0]

    def _singlets_in_colourless_gaps(self, reading: Sequence[int | None]) -> bool:
        n, reps = len(reading), self.representations
        for i, label in enumerate(reading):
            if label is not None:
                continue
            before, after = (i - 1) % n, (i + 1) % n
            while reading[before] is None:
                before = (before - 1) % n
            while reading[after] is None:
                after = (after + 1) % n
            if reps[reading[before]] != -3 or reps[reading[after]] != 3:
                return False
        return True

    def _colourless_gaps(self, labels: Sequence[int | None]) -> set[int] | None:
        """Word gaps k (between letters k and k+1) that are colourless in a
        leading reading of the word, or None if no reading is leading."""

        n, reps = len(labels), self.representations
        for forward in (True, False):
            reading = list(labels) if forward else list(reversed(labels))
            flow = outer_flow([a for a in reading if a is not None], reps)
            if flow not in self.leading or not self._singlets_in_colourless_gaps(
                reading
            ):
                continue
            step = 1 if forward else -1
            gaps = set()
            for k in range(n):
                before, after = (k, k + 1) if forward else (k + 1, k)
                before, after = before % n, after % n
                while labels[before] is None:
                    before = (before - step) % n
                while labels[after] is None:
                    after = (after + step) % n
                if reps[labels[before]] == -3 and reps[labels[after]] == 3:
                    gaps.add(k)
            return gaps
        return None

    def keep(self, skeleton: Skeleton) -> bool:
        contents = skeleton.contents
        species = self.catalogue.species
        quark = [abs(species(c.loop_out).colour) == 3 for c in contents]
        if all(quark):
            return False  # closed quark loop: degree P-1
        word: list[int | None] = []
        ends = []
        for position, content in enumerate(contents):
            word.extend(self._letter(letter) for letter in content.colour)
            ends.append(len(word) - 1)
            if (
                quark[position]
                and quark[position - 1]
                and any(
                    abs(species(block.field).colour) != 1
                    for term in content.terms
                    for block in term.blocks
                )
            ):
                return False  # coloured attachment inside a fermion arc
        gaps = self._colourless_gaps(word)
        if gaps is None:
            return False
        return all(ends[i] in gaps for i in range(len(contents)) if quark[i])


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
    for c in range(len(table.classes)):
        if c not in elements:
            faces.append(None)
            exact.append(None)
            continue
        face = face_degree(table.tensor(c), table.basis, sorted(elements[c]))
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
    "COMPRESSION_THEOREMS",
    "SYMMETRIC_GROUP_FFT",
    "BlockName",
    "BornBasis",
    "ColourNetwork",
    "ColourProduct",
    "ColourString",
    "CompressedColour",
    "CompressionCertificate",
    "ContractionPolicy",
    "LeadingColourCertificate",
    "PositionColour",
    "Primitive",
    "PrimitiveKind",
    "SeedColour",
    "SeedCombination",
    "SeedFourier",
    "SeedTable",
    "VertexDecomposition",
    "block_colour_network",
    "block_name",
    "born_basis",
    "build_seed_table",
    "canonical_primitive",
    "canonical_seed",
    "certify_leading_colour",
    "decompose_vertex",
    "extend_seed_table",
    "face_count",
    "face_degree",
    "kept",
    "literal_network",
    "multiply",
    "project",
    "reduce_network",
    "relabel",
    "seed_cells",
    "seed_fourier",
    "spenso_colour_network",
]
