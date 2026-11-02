# SPDX-License-Identifier: 0BSD
"""Native exact colour cells (WP3.1 package C; ``RUST-MIGRATION.md`` §3.C).

Twins of the colour-cell stages of the Python generator, switched on inside
those stages (``NATIVE_STAGES``, DESIGN §7.9); the Python stages stay their
oracles (``native_stages(())`` runs them):

* **C1 cells.** ``seed_cells`` has the signature and output of
  ``colour.seed_cells``; ``NativeCells`` is the twin of
  ``colour._PrimitiveCells`` (relabelling caches, shared both ways with
  its ``rebased`` and ``directed`` engines; ``_PrimitiveCells.native`` is
  its twin in the switched path, ``switched_seed_cells``);
  ``contract_colour``/``contract_many`` and ``gram`` are the twins of
  ``colour.contract_colour`` and ``BornBasis.gram``. Cells stay native in a
  ``NativeCellTable`` handle (``cell_table``) that counts its decodes.
  Quark rings enter as their reduced canonical tensors
  (``colour._reduced_primitive``, cached by the oracle).
* **C2 projection.** ``project_records`` forms the plan's two-level seed
  records of ``plan_export._Exporter.seed_records`` natively, as the exact
  JSON bytes the plan holds (D's call site).
* **C3 degrees.** ``table_degree`` (``SeedTable.degree``), ``degrees``
  (``LoopSlice.degrees``, inclusive; ``slice_degrees`` is its switch),
  ``adjoint_orbits`` (``assemble._AdjointOrbits``; ``orbits_class`` is its
  switch) and ``live_classes`` (the liveness of ``LoopSlice._prune_colour``).
* **C4 dipoles.** ``dipole_orbits`` builds the IR dipole operators per
  relabelling orbit in rusticol-core (no Python, no licence call; the
  stored form and its binding are ``loops/v2/dipoles``); ``dipoles``
  expands them to the output of ``LoopSlice.dipoles``.
* **C5 directed colour.** ``directed_table`` (``DirectedColour._table``,
  operator images formed natively by ``operator_images``' Rust twin) and
  ``family_degrees`` (``ObservableSlice._family_degrees``).

Determinism (rule 5 of the plan): every native output is independent of the
thread count (``threads`` selects a private pool; None the global one).
"""

from __future__ import annotations

import hashlib
import weakref
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any

from pyamplicol.color.connections import (
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
)

from .assemble import _AdjointOrbits
from .colour import (
    BornBasis,
    Laurent,
    Primitive,
    _reduced_primitive,
    _ring_relabelling,
)
from .families import Orders, add_orders

__all__ = [
    "NativeCells",
    "adjoint_orbits",
    "cell_table",
    "contract_colour",
    "contract_many",
    "decode",
    "degrees",
    "dipole_orbits",
    "dipoles",
    "dipoles_digest",
    "face_count",
    "gram",
    "live_classes",
    "operator_images",
    "project_records",
    "seed_cells",
    "table_degree",
]

_KINDS = {"adjoint-ring": 0, "fundamental-trace": 1}

# Rule 1 (RUST-MIGRATION §2): the switched stages of package C, each one flag
# dispatched inside its Python stage, which stays the oracle:
#   "cells"   colour.seed_cells (the slice's, the companion's and the
#             prune's tables) and DirectedColour._table (directed tables);
#   "degrees" LoopSlice.degrees, the inclusive metric (_prune_colour and the
#             companion read it);
#   "orbits"  assemble._AdjointOrbits (classes and rename);
#   "dipoles" LoopSlice.dipoles (``dipoles``: orbit representatives);
#   "family"  ObservableSlice._family_degrees (complete-family degrees).
# Rule 2: no stage has a per-request Python fallback; a native refusal is
# raised (fail closed).
STAGES: frozenset[str] = frozenset({"cells", "degrees", "orbits", "dipoles", "family"})
NATIVE_STAGES: frozenset[str] = STAGES
_DECODED = 0


@contextmanager
def native_stages(stages: Iterable[str]) -> Iterator[None]:
    """``NATIVE_STAGES`` set to ``stages`` for the block and restored after
    it: ``native_stages(())`` is the oracle lane of the tests and tools."""

    global NATIVE_STAGES
    chosen = frozenset(stages)
    if unknown := chosen - STAGES:
        raise ValueError(f"unknown native colour stages {sorted(unknown)}")
    before = NATIVE_STAGES
    NATIVE_STAGES = chosen
    try:
        yield
    finally:
        NATIVE_STAGES = before


# The binding's plain-tuple encodings (``v2_colour.rs``).
RawStructure = tuple[list[tuple[int, list[int], int]], list[list[int]]]
RawTensor = tuple[list[tuple[int, int]], list[tuple[str, str, int, RawStructure]], int]
RawClass = tuple[int, list[int], int, list[tuple[int, int]], RawTensor | None]
Cells = tuple[tuple[Laurent, ...], ...]
KeySet = Iterable[tuple[ColorTensor, int]]


@cache
def _native() -> Any:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


# ------------------------------------------------------------------ encoding


def _structure(tensor: ColorTensor) -> RawStructure:
    return (
        [
            (s.fundamental_label, list(s.adjoint_labels), s.antifundamental_label)
            for s in tensor.open_strings
        ],
        [list(trace) for trace in tensor.traces],
    )


def _legs(legs: Iterable[Any]) -> list[tuple[int, int]]:
    return [(leg.label, leg.representation) for leg in legs]


def _tensor(tensor: ConnectedColorTensor) -> RawTensor:
    return (
        _legs(tensor.legs),
        [
            (
                str(term.coefficient.real),
                str(term.coefficient.imag),
                term.nc_power,
                _structure(term.tensor),
            )
            for term in tensor.terms
        ],
        tensor.normalization_power,
    )


def _texts(value: ExactColorCoefficient) -> tuple[str, str]:
    return str(value.real), str(value.imag)


class _Decoder:
    """Native Laurent cells to Python ones; equal values share one dict
    (cells are read-only, as ``_PrimitiveCells``'s cached cells)."""

    def __init__(self) -> None:
        self._coefficients: dict[tuple[str, str], ExactColorCoefficient] = {}

    def coefficient(self, re: str, im: str) -> ExactColorCoefficient:
        found = self._coefficients.get((re, im))
        if found is None:
            found = self._coefficients[re, im] = ExactColorCoefficient(
                Fraction(re), Fraction(im)
            )
        return found

    def cell(self, raw: Sequence[tuple[int, str, str]]) -> Laurent:
        return {power: self.coefficient(re, im) for power, re, im in raw}


def decode(table: Any, start: int = 0, stop: int | None = None) -> Cells:
    """The cells of a ``NativeCellTable`` (rows ``start:stop``) as
    ``seed_cells`` returns them."""

    values, ids = table.decode(start, stop)
    decoder = _Decoder()
    cells = [decoder.cell(value) for value in values]
    return tuple(tuple(cells[i] for i in row) for row in ids)


# ---------------------------------------------------------------- C1: cells


Relabelling = tuple[Primitive, dict[int, int]]

# ``colour._ring_relabelling`` of each quark-ring seed per basis (it depends
# only on the seed and the basis legs' adjoint labels), kept while the basis
# lives: a table rebuilt on the same basis and classes (the degrees' slice
# table, the LC prune's) canonicalises each ring once. Keyed by ``id`` and
# dropped by a finaliser when the basis dies.
_RELABELLINGS: dict[int, dict[Primitive, Relabelling]] = {}


def _relabellings(basis: BornBasis) -> dict[Primitive, Relabelling]:
    key = id(basis)
    found = _RELABELLINGS.get(key)
    if found is None:
        found = _RELABELLINGS[key] = {}
        weakref.finalize(basis, _RELABELLINGS.pop, key, None)
    return found


class NativeCells:
    """The twin of ``colour._PrimitiveCells`` (and of ``seed_cells``'s
    generic overlaps) against one basis: a native engine with the
    relabelling caches. ``rebased`` gives an engine against another basis
    on the same legs that shares them both ways (the native caches and the
    registered quark rings), as ``_PrimitiveCells.rebased`` does;
    ``directed`` one against the operator images of its basis under a
    colour history of gluon emissions, formed natively (its ``basis`` is
    None: the images stay native)."""

    def __init__(
        self,
        basis: BornBasis | None,
        *,
        _engine: Any = None,
        _rings: dict[Primitive, int] | None = None,
        _on_legs: Sequence[Any] | None = None,
        _relabelled: dict[Primitive, Relabelling] | None = None,
    ):
        self.basis = basis
        self.legs = tuple(basis.legs if basis is not None else _on_legs or ())
        if _engine is None:
            if basis is None:
                raise ValueError("native cells need a basis")
            _engine = _native().NativeCellEngine(
                _legs(basis.legs), [_tensor(b) for b in basis.tensors]
            )
        self._engine = _engine
        self._adjoint = frozenset(
            leg.label for leg in self.legs if leg.representation == 8
        )
        # Canonical quark ring -> native ring id, shared with the rebased
        # engines (the native ring registry is shared too).
        self._rings: dict[Primitive, int] = {} if _rings is None else _rings
        # Quark-ring seed -> its canonical ring and relabelling (the legs
        # fix the adjoint labels, so rebased and directed engines share it).
        if _relabelled is None:
            _relabelled = {} if basis is None else _relabellings(basis)
        self._relabelled = _relabelled

    def rebased(self, basis: BornBasis) -> NativeCells:
        """The cells against ``basis`` (same legs), sharing the caches."""

        if tuple(basis.legs) != self.legs:
            raise ValueError("rebased primitive cells need the same colour legs")
        engine = self._engine.rebased([_tensor(b) for b in basis.tensors])
        return NativeCells(
            basis, _engine=engine, _rings=self._rings, _relabelled=self._relabelled
        )

    def directed(
        self,
        born: Sequence[tuple[int, int]],
        loop_side: Sequence[tuple[int, int]],
        threads: int | None = None,
    ) -> NativeCells:
        """The cells against the operator images ket^dagger bra(b) of this
        basis (``DirectedColour.operator``) under a history whose Born and
        loop sides are the gluon emissions (emitter, emitted) ``born`` and
        ``loop_side``, formed natively; the caches are shared."""

        engine = self._engine.directed(list(born), list(loop_side), threads)
        return NativeCells(
            None,
            _engine=engine,
            _rings=self._rings,
            _on_legs=self.legs,
            _relabelled=self._relabelled,
        )

    def _class(self, seed: ConnectedColorTensor | Primitive) -> RawClass:
        if not isinstance(seed, Primitive):
            return (3, [], 0, [], _tensor(seed))
        if seed.kind != "quark-ring":
            return (_KINDS[seed.kind], list(seed.word), 0, [], None)
        # <b|back(P')> = <forward(b)|P'> on the canonical ring P'.
        found = self._relabelled.get(seed)
        if found is None:
            found = self._relabelled[seed] = _ring_relabelling(seed, self._adjoint)
        canonical, back = found
        ring = self._rings.get(canonical)
        if ring is None:
            reduced = _reduced_primitive(canonical, self.legs)
            ring = self._rings[canonical] = self._engine.register_ring(_tensor(reduced))
        return (2, [], ring, [(b, a) for a, b in back.items()], None)

    def table(
        self,
        classes: Sequence[ConnectedColorTensor | Primitive],
        threads: int | None = None,
    ) -> Any:
        """The cells of ``classes`` as a ``NativeCellTable``."""

        return self._engine.table([self._class(seed) for seed in classes], threads)

    def __call__(self, primitive: Primitive) -> tuple[Laurent, ...]:
        (row,) = decode(self.table([primitive]))
        return row

    @property
    def cache_sizes(self) -> tuple[int, int, int, int]:
        """(relabelled bras, quark rings, primitive overlaps, ring overlaps)
        cached natively, shared with the rebased engines."""

        sizes: tuple[int, int, int, int] = self._engine.cache_sizes
        return sizes


def cell_table(
    basis: BornBasis,
    classes: Sequence[ConnectedColorTensor | Primitive],
    primitive_cells: NativeCells | None = None,
    threads: int | None = None,
) -> Any:
    """The native cells <b|kappa_c> of ``classes`` (a ``NativeCellTable``)."""

    if primitive_cells is None:
        primitive_cells = NativeCells(basis)
    elif primitive_cells.basis is not basis:
        raise ValueError("primitive cells of another basis")
    return primitive_cells.table(classes, threads)


def seed_cells(
    basis: BornBasis,
    classes: Sequence[ConnectedColorTensor | Primitive],
    primitive_cells: NativeCells | None = None,
    threads: int | None = None,
) -> tuple[tuple[Laurent, ...], ...]:
    """``colour.seed_cells``, natively (same signature and output)."""

    return decode(cell_table(basis, classes, primitive_cells, threads))


def contract_many(
    pairs: Sequence[tuple[ConnectedColorTensor, ConnectedColorTensor]],
    threads: int | None = None,
) -> list[Laurent]:
    """``colour.contract_colour`` of every (bra, ket) pair, natively."""

    raw = _native().contract_colours(
        [(_tensor(a), _tensor(b)) for a, b in pairs], threads
    )
    decoder = _Decoder()
    return [decoder.cell(cell) for cell in raw]


def contract_colour(left: ConnectedColorTensor, right: ConnectedColorTensor) -> Laurent:
    """``colour.contract_colour``, natively."""

    (cell,) = contract_many([(left, right)])
    return cell


def face_count(left: ColorTensor, right: ColorTensor) -> int:
    """``colour.face_count``, natively."""

    (faces,) = _native().face_counts([(_structure(left), _structure(right))])
    return int(faces)


def gram(basis: BornBasis, threads: int | None = None) -> Cells:
    """``BornBasis.gram``, natively."""

    return decode(_native().colour_gram([_tensor(b) for b in basis.tensors], threads))


# ----------------------------------------------------------- C2: projection


class _Ranks:
    """Orders as ranks in their sorted order (sorting ranks sorts orders)."""

    def __init__(self, orders: Iterable[Orders]):
        self.orders = sorted(set(orders))
        self.rank = {order: i for i, order in enumerate(self.orders)}

    def __getitem__(self, order: Orders) -> int:
        return self.rank[order]


def _table_orders(table: Any, ranks: _Ranks) -> tuple[list[list[int]], list[list[int]]]:
    return (
        [sorted(ranks[o] for o in orders) for orders in table.orders],
        [sorted(ranks[o] for o in orders) for orders in table.basis.orders],
    )


def _table_handle(table: Any, handles: Mapping[int, Any] | None) -> Any:
    found = None if handles is None else handles.get(id(table))
    if found is not None:
        return found
    return _native().NativeCellTable.from_cells(
        [
            [[(p, *_texts(v)) for p, v in cell.items()] for cell in row]
            for row in table.cells
        ]
    )


def project_records(
    slice_: Any,
    columns: Sequence[Any],
    seeds: Iterable[tuple[tuple[Any, ...], Orders]],
    born_components: dict[tuple[int, Orders, int], int],
    tables: Mapping[int, Any] | None = None,
    threads: int | None = None,
) -> tuple[bytes, bytes]:
    """``_Exporter.seed_records``, natively: the JSON bytes of the seed class
    records and of the seed key records, byte-identical to their
    ``json.dumps(..., separators=(",", ":"))``. ``seeds`` are the exporter's
    seed keys (combination, loop order) in plan order; ``born_components``
    is extended in place in first-seen order. ``tables`` maps ``id(table)``
    of a column table to its ``NativeCellTable`` (default: encoded from its
    Python cells); ``threads`` selects a private pool (None: the global
    one; the bytes do not depend on it)."""

    from .refusals import require_formal_nc_tables

    seeds = list(seeds)
    accuracy = slice_.request.accuracy
    born_grades = slice_._born_grades()
    distinct: dict[int, Any] = {}
    for column in columns:
        distinct.setdefault(id(column.table), column.table)
    # Projected at formal N_c only over N_c-exact blocks (``refusals``).
    require_formal_nc_tables(accuracy, distinct.values(), "native seed records")
    orders: set[Orders] = {order for _c, order in seeds}
    orders |= {a for _b, a in born_grades} | {a for _b, a, _g in born_components}
    for table in distinct.values():
        orders |= {o for row in table.orders for o in row}
        orders |= {o for row in table.basis.orders for o in row}
    for column in columns:
        orders |= {o for pair in column.pairs for o in pair}
    ranks = _Ranks(orders)
    index = {key: t for t, key in enumerate(distinct)}
    raw_tables = [
        (_table_handle(table, tables), *_table_orders(table, ranks))
        for table in distinct.values()
    ]
    raw_columns = [
        (
            index[id(column.table)],
            [(ranks[a], ranks[b]) for a, b in column.pairs],
            column.degree,
            None if column.basis is None else sorted(column.basis),
        )
        for column in columns
    ]
    try:
        class_json, records_json, components = _native().project_records(
            raw_tables,
            raw_columns,
            [((b, ranks[a]), sorted(g)) for (b, a), g in born_grades.items()],
            accuracy,
            [(b, ranks[a], g) for b, a, g in born_components],
            [
                ([(c, *_texts(x)) for c, x in combination], ranks[order])
                for combination, order in seeds
            ],
            threads,
        )
    except ValueError as error:
        text = str(error)
        if "no sector admits class" in text:
            c, rank = text.rsplit("class ", 1)[1].split(" at order rank ")
            raise KeyError(
                f"no sector admits class {c} at {ranks.orders[int(rank)]}"
            ) from None
        raise
    for b, a, g in components[len(born_components) :]:
        born_components[(b, ranks.orders[a], g)] = len(born_components)
    return class_json, records_json


# -------------------------------------------------------------- C3: degrees


def table_degree(
    table: Any,
    pairs: Iterable[tuple[Orders, Orders]],
    born_grades: Mapping[tuple[int, Orders], Iterable[int]] | None = None,
    handle: Any = None,
) -> int | None:
    """``SeedTable.degree``, natively."""

    pairs = list(pairs)
    orders = {o for pair in pairs for o in pair}
    orders |= {o for row in table.orders for o in row}
    orders |= {o for row in table.basis.orders for o in row}
    if born_grades is not None:
        orders |= {a for _b, a in born_grades}
    ranks = _Ranks(orders)
    handle = _table_handle(table, None) if handle is None else handle
    class_orders, basis_orders = _table_orders(table, ranks)
    degree: int | None = handle.degree(
        class_orders,
        basis_orders,
        [(ranks[a], ranks[b]) for a, b in pairs],
        None
        if born_grades is None
        else [((b, ranks[a]), sorted(g)) for (b, a), g in born_grades.items()],
    )
    return degree


class _Orbits(_AdjointOrbits):
    """``assemble._AdjointOrbits`` with its classes and ``rename`` native
    (``site_key`` is the oracle's own)."""

    def __init__(self, legs: Sequence[Any], process: Any, born_keys: Iterable[KeySet]):
        pdg = {leg.label: leg.outgoing_pdg for leg in process.legs}
        adjoint = sorted(leg.label for leg in legs if leg.representation == 8)
        self.native = _native().NativeAdjointOrbits(
            [(label, pdg[label]) for label in adjoint],
            [[(_structure(t), p) for t, p in keys] for keys in born_keys],
        )
        self.classes = tuple(tuple(members) for members in self.native.classes)
        self._class = {
            label: c for c, members in enumerate(self.classes) for label in members
        }

    def rename(self, tensor: Any) -> dict[int, int]:
        if not self.classes:
            return {}
        return dict(self.native.rename(_structure(tensor)))


def adjoint_orbits(
    legs: Sequence[Any], process: Any, born_keys: Iterable[KeySet]
) -> _Orbits:
    """``assemble._AdjointOrbits``, with the classes built natively."""

    return _Orbits(legs, process, born_keys)


def _max(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


_CLOSURE_KEYS = ("native_cells", "closure-keys")


def _closure_keys(
    slice_: Any, layout: Mapping[Any, Any]
) -> dict[Orders, frozenset[tuple[ColorTensor, int]]]:
    """The degree keys of the CT program's closure terms per orders (the
    oracle's ``counterterms[item.orders] |= ...`` over
    ``counterterm_program.closure.terms``), cached on the slice for its
    program and coupling layout (the inclusive degrees and every complete
    family read them)."""

    from .assemble import WholeRuleTerm
    from .ct_sites import LAURENT_POWERS

    program = slice_.counterterm_program
    cached = slice_._cache.get(_CLOSURE_KEYS)
    if cached is not None and cached[0] is program and cached[1] is layout:
        found: dict[Orders, frozenset[tuple[ColorTensor, int]]] = cached[2]
        return found
    # The closure's terms (millions at seven legs, a few thousand colours)
    # are grouped by what their keys read: the colour (by identity: colours
    # are shared), the orders, the term's own grade and the coupling, or a
    # whole-rule term's coefficient grades (by identity).
    first: dict[tuple[int, Any, int, Orders], Any] = {}
    for item in program.closure.terms:
        term = item.term
        key = (
            id(term.colour),
            id(term.grades) if isinstance(term, WholeRuleTerm) else item.coupling,
            item.grade,
            item.orders,
        )
        if key not in first:
            first[key] = term
    grouped: dict[tuple[Orders, int], tuple[Any, set[int]]] = {}
    coupling_grades: dict[Any, tuple[int, ...]] = {}
    for (colour_id, coupling, grade, orders), term in first.items():
        if isinstance(term, WholeRuleTerm):
            grades = {d + grade for d, _e in term.grades}
        else:
            base = coupling_grades.get(coupling)
            if base is None:
                base = coupling_grades[coupling] = tuple(
                    g
                    for g, _c in layout.get(coupling, ())
                    if coupling[1] in LAURENT_POWERS
                )
            grades = {g + grade for g in base}
        entry = grouped.get((orders, colour_id))
        if entry is None:
            entry = grouped[orders, colour_id] = (term.colour, set())
        entry[1].update(grades)
    del first
    by_orders: dict[Orders, list[tuple[Any, set[int]]]] = defaultdict(list)
    for (orders, _colour_id), entry in grouped.items():
        by_orders[orders].append(entry)
    found = {
        orders: frozenset(
            (part.tensor, part.nc_power + grade)
            for colour, grades in entries
            for part in colour.terms
            for grade in grades
        )
        for orders, entries in by_orders.items()
    }
    slice_._cache[_CLOSURE_KEYS] = (program, layout, found)
    return found


class _DegreeKeys:
    """The key sets of ``LoopSlice.degrees``, collected as the oracle
    collects them (``_scalar_plan``'s grades, ``_terms_of`` per site, the CT
    program's closure terms): the Born key sets per Born order, the
    counterterm key sets per loop order and the field queries per sector.
    ``orbits`` closes one counterterm site per relabelling orbit (the
    inclusive metric); without it every site is closed (a directed colour
    history is not invariant under the relabellings)."""

    def __init__(self, slice_: Any, orbits: bool):
        from .assemble import WholeRuleTerm
        from .ct_sites import LAURENT_POWERS

        _plan, layout = slice_._scalar_plan(LAURENT_POWERS)

        def grades(name: str, powers: Iterable[int] = LAURENT_POWERS) -> set[int]:
            return {g for power in powers for g, _c in layout.get((name, power), ())}

        def keys(terms: Iterable[Any], grade_of: Any) -> set[tuple[ColorTensor, int]]:
            return {
                (item.tensor, item.nc_power + grade)
                for term in terms
                for item in term.colour.terms
                for grade in grade_of(term)
            }

        # The counterterm keys of one colour are its terms times the union
        # of its grades: grades are gathered per (order, colour) first and
        # every colour is expanded once (the oracle's sets, without
        # re-hashing a colour's tensors per term and grade).
        grouped: dict[Orders, dict[int, tuple[Any, set[int]]]] = defaultdict(dict)

        def gather(orders: Orders, colour: Any, found: Iterable[int]) -> None:
            entry = grouped[orders].get(id(colour))
            if entry is None:
                entry = grouped[orders][id(colour)] = (colour, set())
            entry[1].update(found)

        self.born = {
            order: frozenset(keys(terms, lambda t: grades(t.entry.coupling.name, (0,))))
            for order, terms in slice_.born_terms.items()
        }
        self.orbits = (
            _Orbits(slice_.basis.legs, slice_.blocks.process, self.born.values())
            if orbits
            else None
        )
        closed: set[Hashable] = set()
        site_grades: dict[str, set[int]] = {}
        for site in slice_.sites.sites:
            if self.orbits is not None:
                identity = self.orbits.site_key(site)
                if identity in closed:
                    continue
                closed.add(identity)
            grouped.setdefault(site.orders, {})
            for term in slice_._terms_of(site):
                if isinstance(term, WholeRuleTerm):
                    gather(site.orders, term.colour, {d for d, _e in term.grades})
                    continue
                name = term.entry.coupling.name
                named = site_grades.get(name)
                if named is None:
                    named = site_grades[name] = grades(name)
                gather(site.orders, term.colour, named)
        counterterms: dict[Orders, set[tuple[ColorTensor, int]]] = {
            orders: {
                (part.tensor, part.nc_power + grade)
                for colour, found in entries.values()
                for part in colour.terms
                for grade in found
            }
            for orders, entries in grouped.items()
        }
        if slice_.counterterm_program is not None:
            for orders, found_keys in _closure_keys(slice_, layout).items():
                counterterms[orders] = counterterms.get(orders, set()) | found_keys
        self.counterterms = counterterms
        # Per sector: (pairs, [(Born order a, ket key, offset)]) with the ket
        # keys ("born", order) or ("ct", order) (``_DegreeKeys.ket``).
        self.sectors: list[tuple[Orders, Any, list[tuple[Orders, Any, int]]]] = []
        for sector, pairs in slice_.sectors:
            found_queries = [(a, ("ct", b), 0) for a, b in pairs]
            for insertion in slice_.fields:
                if (
                    a := next((a for a, b in pairs if b == insertion.orders), None)
                ) is None or insertion.born_order not in self.born:
                    continue
                for grade in grades(slice_.field_names[insertion.identity]):
                    found_queries.append((a, ("born", insertion.born_order), grade))
            self.sectors.append((sector, pairs, found_queries))

    def ket(self, key: tuple[str, Orders]) -> KeySet:
        kind, order = key
        if kind == "born":
            return self.born[order]
        return self.counterterms.get(order, set())


def degrees(
    slice_: Any, table_handle: Any = None, threads: int | None = None
) -> dict[str, dict[Orders, int | None]]:
    """``LoopSlice.degrees()`` (the inclusive metric), with every overlap
    degree and the seed-table degrees evaluated natively in one batch.
    ``table_handle`` is the slice table's ``NativeCellTable`` (default: its
    cells built natively from its classes).

    The key sets are collected as the oracle collects them (``_DegreeKeys``,
    on its private members), so this copy follows the oracle's key rules;
    the oracle tests compare the two on every control."""

    keys = _DegreeKeys(slice_, orbits=True)
    born = keys.born
    assert keys.orbits is not None
    # Key sets and queries (bra set, ket set, offset).
    sets: list[list[tuple[RawStructure, int]]] = []
    set_ids: dict[Any, int] = {}

    def set_id(key: Any, values: KeySet) -> int:
        found = set_ids.get(key)
        if found is None:
            found = set_ids[key] = len(sets)
            sets.append([(_structure(t), p) for t, p in values])
        return found

    queries: list[tuple[int, int, int]] = []

    def query(bra: Orders, ket_key: Any, offset: int = 0) -> int:
        bra_set = set_id(("born", bra), born[bra])
        queries.append((bra_set, set_id(ket_key, keys.ket(ket_key)), offset))
        return len(queries) - 1

    born_queries = [
        (add_orders(a, b), query(a, ("born", b))) for a, b in product(born, repeat=2)
    ]
    born_grades = slice_._born_grades()
    table = slice_.table
    if table_handle is None:
        table_handle = cell_table(slice_.basis, table.classes, threads=threads)
    sector_queries = [
        (sector, pairs, [query(a, key, offset) for a, key, offset in found])
        for sector, pairs, found in keys.sectors
    ]
    values = _native().overlap_degrees(
        _legs(slice_.basis.legs), sets, queries, keys.orbits.native, threads
    )
    out: dict[str, dict[Orders, int | None]] = {"born": {}, "virtual": {}}
    for squared, q in born_queries:
        out["born"][squared] = _max(out["born"].get(squared), values[q])
    for sector, pairs, sector_found in sector_queries:
        value = table_degree(table, pairs, born_grades, table_handle)
        for q in sector_found:
            value = _max(value, values[q])
        out["virtual"][sector] = value
    return out


def live_classes(
    slice_: Any,
    observables: Iterable[tuple[Any, Mapping[Orders, Any]]] = (),
    degrees: Mapping[Orders, int | None] | None = None,
    handles: Mapping[int, Any] | None = None,
    threads: int | None = None,
) -> dict[tuple[int, Orders], bool]:
    """The liveness ``class_live(c, order)`` of ``LoopSlice._prune_colour``
    for every class of the slice table at each of its loop orders, over
    the slice table and the ``observables`` criteria (table, virtual sector
    degrees). ``degrees`` are the slice table's virtual sector degrees
    (default ``slice_.degrees()["virtual"]``); ``handles`` maps
    ``id(table)`` to its ``NativeCellTable``. Without a handle, the slice
    table's cells are built natively from its classes (they are
    ``seed_cells(table.basis, table.classes)`` by construction,
    ``colour._extended``); every other table is read from its own cells,
    as the oracle reads ``table.cells`` (an observable's directed table
    holds cells against operator images, not against its Born basis)."""

    accuracy = slice_.request.accuracy
    grades = slice_._born_grades()
    if degrees is None:
        degrees = slice_.degrees()["virtual"]
    criteria = [(slice_.table, degrees), *observables]
    sectors = list(slice_.sectors)
    orders = {o for row in slice_.table.orders for o in row}
    orders |= {o for row in slice_.basis.orders for o in row}
    orders |= {o for _s, pairs in sectors for pair in pairs for o in pair}
    orders |= {a for _b, a in grades}
    ranks = _Ranks(orders)

    def handle(table: Any) -> Any:
        found = None if handles is None else handles.get(id(table))
        if found is not None:
            return found
        if table is slice_.table:
            return cell_table(table.basis, table.classes, threads=threads)
        return _table_handle(table, None)

    queries = [
        (c, order) for c, row in enumerate(slice_.table.orders) for order in sorted(row)
    ]
    live = _native().live_classes(
        [
            (handle(table), [table_degrees[sector] for sector, _p in sectors])
            for table, table_degrees in criteria
        ],
        [[(ranks[a], ranks[b]) for a, b in pairs] for _s, pairs in sectors],
        [sorted(ranks[o] for o in born) for born in slice_.basis.orders],
        [((b, ranks[a]), sorted(g)) for (b, a), g in grades.items()],
        accuracy,
        [(c, ranks[order]) for c, order in queries],
        threads,
    )
    return dict(zip(queries, live, strict=True))


# ------------------------------------------------- C5: directed colour


# Native cell tables of the directed tables built under the "cells" stage,
# per ``DirectedColour`` and direction (``swapped``), for the family degrees.
_DIRECTED: weakref.WeakKeyDictionary[Any, dict[bool, Any]] = weakref.WeakKeyDictionary()


def _gluon_emissions(emissions: Iterable[Any]) -> list[tuple[int, int]] | None:
    """(emitter, emitted) of a colour connection's steps if they are all
    gluon emissions, else None (a quark-pair splitting)."""

    from pyamplicol.color.connections import EmitGluon

    out = []
    for emission in emissions:
        if not isinstance(emission, EmitGluon):
            return None
        out.append((emission.emitter_label, emission.emitted_label))
    return out


def directed_table(directed: Any, table: Any, swapped: bool) -> Any:
    """``DirectedColour._table`` of a direction whose ket steps are all gluon
    emissions, natively (the "cells" stage; None when it is off): the
    operator images ket^dagger bra(b) replace the Born basis, and the native
    engine of the shared ``_PrimitiveCells`` (``directed._cells``), rebased
    onto them, forms the cells; the relabelling caches are shared by every
    directed table of the slice, as in Python. The images are formed
    natively (``cells::operator_image``) when the Born side's steps are
    gluon emissions too, otherwise by the oracle's ``DirectedColour.operator``.
    The native table is kept for the family degrees (``family_degrees``)."""

    if "cells" not in NATIVE_STAGES:
        return None
    born, loop_side = directed._sides(swapped)
    shared = directed._cells
    if shared is None:
        base = NativeCells(table.basis)
    elif shared.basis is table.basis:
        base = shared.native()
    else:
        base = shared.native().rebased(table.basis)
    emissions = _gluon_emissions(born.emissions)
    absorbed = _gluon_emissions(loop_side.emissions)
    if emissions is not None and absorbed is not None:
        engine = base.directed(emissions, absorbed)
    else:
        images = replace(
            table.basis,
            tensors=tuple(directed.operator(b, swapped) for b in table.basis.tensors),
        )
        engine = base.rebased(images)
    handle = engine.table(table.classes)
    _DIRECTED.setdefault(directed, {})[swapped] = handle
    return replace(table, cells=_decoded(handle))


def _from_raw(raw: RawTensor) -> ConnectedColorTensor:
    from pyamplicol.color.connections import ColorLeg, OpenColorString, TensorTerm

    legs, terms, normalization = raw
    decoder = _Decoder()
    return ConnectedColorTensor(
        tuple(ColorLeg(label, representation) for label, representation in legs),
        tuple(
            TensorTerm(
                decoder.coefficient(re, im),
                ColorTensor(
                    tuple(OpenColorString(q, tuple(w), qb) for q, w, qb in strings),
                    tuple(tuple(trace) for trace in traces),
                ),
                power,
            )
            for re, im, power, (strings, traces) in terms
        ),
        normalization,
    )


def operator_images(
    tensors: Sequence[ConnectedColorTensor],
    born: Sequence[tuple[int, int]],
    loop_side: Sequence[tuple[int, int]],
) -> list[ConnectedColorTensor]:
    """``DirectedColour.operator`` of each tensor under a history whose Born
    and loop sides are the gluon emissions (emitter, emitted) ``born`` and
    ``loop_side``, formed natively (``cells::operator_image``): the same
    colour vectors as the oracle's images (equal overlaps with every
    tensor), not necessarily the same term lists."""

    raw = _native().operator_images(
        [_tensor(t) for t in tensors], list(born), list(loop_side)
    )
    return [_from_raw(image) for image in raw]


def _raw_single(legs: list[tuple[int, int]], tensor: ColorTensor) -> RawTensor:
    return (legs, [("1", "0", 0, _structure(tensor))], 0)


class _FamilyBatch:
    """The overlap queries of a complete family's history pairs, in two
    native batches: pairs of gluon-emission histories with their operator
    images formed natively (``directed_overlap_degrees``: literal bra and
    ket structures, one history per pair), the others with the connected
    images of the oracle (``tensor_overlap_degrees``)."""

    def __init__(self, slice_: Any):
        self.legs = slice_.basis.legs
        self.raw_legs = _legs(self.legs)
        self.keys = _DegreeKeys(slice_, orbits=False)
        # Gluon-emission pairs: shared literal sets, one history per pair.
        self.structure_sets: list[list[tuple[RawStructure, int]]] = []
        self._structure_ids: dict[Any, int] = {}
        self.histories: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        self.directed_queries: list[tuple[int, int, int, int]] = []
        # The other pairs: tensor sets.
        self.bra_sets: list[list[tuple[RawTensor, int]]] = []
        self.ket_sets: list[list[tuple[RawTensor, int]]] = []
        self.tensor_queries: list[tuple[int, int, int]] = []

    def _structures(self, key: Any) -> int:
        """A literal key set (("bra", order): the Born keys with their
        largest N_c power each; ("born" | "ct", order): the ket keys)."""

        found = self._structure_ids.get(key)
        if found is None:
            kind, order = key
            if kind == "bra":
                powers: dict[ColorTensor, int] = {}
                for t, p in self.keys.born[order]:
                    powers[t] = max(p, powers.get(t, p))
                values = [(_structure(t), p) for t, p in powers.items()]
            else:
                values = [(_structure(t), q) for t, q in self.keys.ket(key)]
            found = self._structure_ids[key] = len(self.structure_sets)
            self.structure_sets.append(values)
        return found

    def pair(self, pair: Any) -> tuple[list[tuple[Orders, Any]], list[Any]]:
        """The queries of one history pair (a ``DirectedColour``): its Born
        queries (squared order, query) and per sector (sector, pairs,
        queries), as ``LoopSlice.degrees`` forms them with its overlaps; a
        query is ("directed" | "tensor", index)."""

        bra_side, ket_side = pair._sides(False)
        emissions = _gluon_emissions(bra_side.emissions)
        absorbed = _gluon_emissions(ket_side.emissions)
        if emissions is not None and absorbed is not None:
            history = len(self.histories)
            self.histories.append((emissions, absorbed))

            def query(a: Orders, key: Any, offset: int = 0) -> tuple[str, int]:
                self.directed_queries.append(
                    (
                        history,
                        self._structures(("bra", a)),
                        self._structures(key),
                        offset,
                    )
                )
                return "directed", len(self.directed_queries) - 1

        else:
            query = self._tensor_queries(pair)
        born_queries = [
            (add_orders(a, b), query(a, ("born", b)))
            for a, b in product(self.keys.born, repeat=2)
        ]
        sector_queries = [
            (sector, pairs, [query(a, key, offset) for a, key, offset in found])
            for sector, pairs, found in self.keys.sectors
        ]
        return born_queries, sector_queries

    def _tensor_queries(self, pair: Any) -> Any:
        """The query former of a pair whose overlaps take the oracle's
        images: operator images of the Born keys against literal kets when
        its ket steps are gluon emissions, else the connected images of
        both sides (the generic contraction)."""

        from .observables import _literal

        absorbable = pair._absorbable(False)
        legs = self.legs

        def image(tensor: ColorTensor, bra: bool) -> Any:
            literal = _literal(tensor, legs)
            if absorbable:
                return pair.operator(literal) if bra else literal
            return pair._image(literal, bra)

        bras: dict[Orders, int] = {}
        for a, born in self.keys.born.items():
            powers: dict[ColorTensor, int] = {}
            for t, p in born:
                powers[t] = max(p, powers.get(t, p))
            bras[a] = len(self.bra_sets)
            self.bra_sets.append(
                [(_tensor(image(t, True)), p) for t, p in powers.items()]
            )
        kets: dict[Any, int] = {}

        def ket(key: Any) -> int:
            found = kets.get(key)
            if found is None:
                found = kets[key] = len(self.ket_sets)
                self.ket_sets.append(
                    [(_tensor(image(t, False)), q) for t, q in self.keys.ket(key)]
                )
            return found

        def query(a: Orders, key: Any, offset: int = 0) -> tuple[str, int]:
            self.tensor_queries.append((bras[a], ket(key), offset))
            return "tensor", len(self.tensor_queries) - 1

        return query

    def evaluate(self, threads: int | None = None) -> dict[str, list[int | None]]:
        native = _native()
        out: dict[str, list[int | None]] = {"directed": [], "tensor": []}
        if self.directed_queries:
            out["directed"] = native.directed_overlap_degrees(
                self.raw_legs,
                self.structure_sets,
                self.structure_sets,
                self.histories,
                self.directed_queries,
                threads,
            )
        if self.tensor_queries:
            out["tensor"] = native.tensor_overlap_degrees(
                self.bra_sets, self.ket_sets, self.tensor_queries, threads
            )
        return out


def family_degrees(
    observables: Any, family: Sequence[Any], tables: list[Any] | None = None
) -> dict[str, dict[Orders, int | None]] | None:
    """``ObservableSlice._family_degrees`` natively (the "family" stage; None
    when it is off): for every history pair (alpha, beta) of the complete
    family, the pair's directed table (``DirectedColour``, its table built
    natively under the "cells" stage) and the degrees of
    ``LoopSlice.degrees(overlaps=pair.overlap, table=pair.table)``, with
    every overlap degree of every pair in native batches
    (``_FamilyBatch``). ``tables`` collects every pair's table (the pruning
    criteria)."""

    if "family" not in NATIVE_STAGES:
        return None
    from pyamplicol.color.correlator_matrices import ColorCorrelator

    from .observables import DirectedColour

    slice_ = observables.slice
    born_grades = slice_._born_grades()
    batch = _FamilyBatch(slice_)
    # Per pair: (Born queries, sector queries, table, native table or None).
    pairs_out: list[tuple[list[Any], list[Any], Any, Any]] = []
    for alpha, beta in product(family, repeat=2):
        pair = DirectedColour(
            slice_.table,
            ColorCorrelator("family", alpha.emissions, beta.emissions),
            observables._cells,
            observables._literals,
        )
        if tables is not None:
            tables.append(pair.table)
        born_queries, sector_queries = batch.pair(pair)
        handle = _DIRECTED.get(pair, {}).get(False)
        pairs_out.append((born_queries, sector_queries, pair.table, handle))
    values = batch.evaluate()
    out: dict[str, dict[Orders, int | None]] = {"born": {}, "virtual": {}}
    for born_queries, sector_queries, table, handle in pairs_out:
        for squared, (kind, q) in born_queries:
            out["born"][squared] = _max(out["born"].get(squared), values[kind][q])
        for sector, sector_pairs, found in sector_queries:
            value = (
                table.degree(sector_pairs, born_grades)
                if handle is None
                else table_degree(table, sector_pairs, born_grades, handle)
            )
            for kind, q in found:
                value = _max(value, values[kind][q])
            out["virtual"][sector] = _max(out["virtual"].get(sector), value)
    return out


# --------------------------------------------------- phase-1 switches


def _decoded(handle: Any) -> Cells:
    """The cells of a native table, decoded and counted (rule 9)."""

    global _DECODED
    cells = decode(handle)
    _DECODED += sum(len(row) for row in cells)
    return cells


def decoded_cells() -> int:
    """Cells the switched stages decoded into Python objects (rule 9: every
    seed and directed table's own cells once, for their Python consumers;
    the degrees decode no cell)."""

    return _DECODED


def switched_seed_cells(
    basis: BornBasis,
    classes: Sequence[ConnectedColorTensor | Primitive],
    primitive_cells: Any = None,
) -> Cells:
    """``colour.seed_cells`` under the "cells" stage: the native engine of
    ``primitive_cells`` (a ``colour._PrimitiveCells``, whose native twin
    shares its caches with its rebased copies' twins) or a fresh one."""

    if primitive_cells is None:
        engine = NativeCells(basis)
    elif primitive_cells.basis is not basis:
        raise ValueError("primitive cells of another basis")
    else:
        engine = primitive_cells.native()
    return _decoded(engine.table(classes))


def slice_degrees(slice_: Any) -> dict[str, dict[Orders, int | None]] | None:
    """``LoopSlice.degrees()`` (inclusive) under the "degrees" stage, or None
    when it is off (the switch in ``LoopSlice.degrees`` caches it)."""

    if "degrees" not in NATIVE_STAGES:
        return None
    return degrees(slice_)


def orbits_class(cls: type) -> type:
    """The class ``assemble._AdjointOrbits(...)`` instantiates: the native
    ``_Orbits`` (classes and ``rename`` in rusticol-core) under the "orbits"
    stage."""

    return _Orbits if cls is _AdjointOrbits and "orbits" in NATIVE_STAGES else cls


# -------------------------------------------------------------- C4: dipoles


def dipole_orbits(slice_: Any, threads: int | None = None) -> Any:
    """The IR dipole operators of a slice's Born basis per relabelling orbit
    (``NativeDipoleOrbits``; built and checked exactly in rusticol-core).
    The candidate relabellings are the adjoint legs of one particle."""

    basis = slice_.basis
    pdg = {leg.label: leg.outgoing_pdg for leg in slice_.blocks.process.legs}
    classes: dict[int, list[int]] = {}
    for leg in basis.legs:
        if leg.representation == 8:
            classes.setdefault(pdg[leg.label], []).append(leg.label)
    return _native().NativeDipoleOrbits.build(
        _legs(basis.legs),
        [_tensor(b) for b in basis.tensors],
        [members for members in classes.values() if len(members) > 1],
        threads,
    )


def dipoles(
    slice_: Any, orbits: Any = None, threads: int | None = None
) -> dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]]:
    """``LoopSlice.dipoles()``, expanded from the native orbit build."""

    if orbits is None:
        orbits = dipole_orbits(slice_, threads)
    values: dict[str, Fraction] = {}

    def value(text: str) -> Fraction:
        found = values.get(text)
        if found is None:
            found = values[text] = Fraction(text)
        return found

    return {
        pair: tuple(tuple(value(v) for v in row) for row in rows)
        for pair, rows in orbits.expand()
    }


def dipoles_digest(
    dipoles: Mapping[tuple[int, int], Sequence[Sequence[Fraction]]],
) -> str:
    """SHA-256 of a ``LoopSlice.dipoles()`` output in its order: per pair a
    header line ``"(i, j) RxC"`` (the key's repr and the block shape), then
    one exact entry per line. It equals ``NativeDipoleOrbits.digest()`` iff
    both hold the same pairs and shapes in the same order with equal
    entries."""

    hasher = hashlib.sha256()
    for pair, rows in dipoles.items():
        columns = len(rows[0]) if rows else 0
        hasher.update(f"{tuple(pair)!r} {len(rows)}x{columns}\n".encode())
        for row in rows:
            for value in row:
                hasher.update(str(value).encode())
                hasher.update(b"\n")
    return hasher.hexdigest()
