# SPDX-License-Identifier: 0BSD
"""Native exact colour cells (WP3.1 package C; ``RUST-MIGRATION.md`` §3.C).

Twins of the colour-cell stages of the Python generator, which stay their
oracles (nothing here is switched on yet):

* **C1 cells.** ``seed_cells`` has the signature and output of
  ``colour.seed_cells``; ``NativeCells`` is the twin of
  ``colour._PrimitiveCells`` (relabelling caches, ``rebased``);
  ``contract_colour``/``contract_many`` and ``gram`` are the twins of
  ``colour.contract_colour`` and ``BornBasis.gram``. Cells stay native in a
  ``NativeCellTable`` handle (``cell_table``) that counts its decodes.
  Quark rings enter as their reduced canonical tensors
  (``colour._reduced_primitive``, cached by the oracle).
* **C2 projection.** ``project_records`` forms the plan's two-level seed
  records of ``plan_export._Exporter.seed_records`` natively, as the exact
  JSON bytes the plan holds.
* **C3 degrees.** ``table_degree`` (``SeedTable.degree``), ``degrees``
  (``LoopSlice.degrees``, inclusive), ``adjoint_orbits``
  (``assemble._AdjointOrbits``) and ``live_classes`` (the liveness of
  ``LoopSlice._prune_colour``).
* **C4 dipoles.** ``dipole_orbits`` builds the IR dipole operators per
  relabelling orbit in rusticol-core (no Python, no licence call; stored
  form ``NativeDipoleOrbits.to_json``); ``dipoles`` expands them to the
  output of ``LoopSlice.dipoles``.

Determinism (rule 5 of the plan): every native output is independent of the
thread count (``threads`` selects a private pool; None the global one).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any

from pyamplicol.color.connections import (
    ColorTensor,
    ConnectedColorTensor,
    ExactColorCoefficient,
)

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
    "face_count",
    "gram",
    "live_classes",
    "project_records",
    "seed_cells",
    "table_degree",
]

_KINDS = {"adjoint-ring": 0, "fundamental-trace": 1}


@cache
def _native() -> Any:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


# ------------------------------------------------------------------ encoding


def _structure(tensor: ColorTensor) -> tuple[list, list]:
    return (
        [
            (s.fundamental_label, list(s.adjoint_labels), s.antifundamental_label)
            for s in tensor.open_strings
        ],
        [list(trace) for trace in tensor.traces],
    )


def _legs(legs: Iterable[Any]) -> list[tuple[int, int]]:
    return [(leg.label, leg.representation) for leg in legs]


def _tensor(tensor: ConnectedColorTensor) -> tuple[list, list, int]:
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


def decode(table: Any, start: int = 0, stop: int | None = None) -> tuple:
    """The cells of a ``NativeCellTable`` (rows ``start:stop``) as
    ``seed_cells`` returns them."""

    values, ids = table.decode(start, stop)
    decoder = _Decoder()
    cells = [decoder.cell(value) for value in values]
    return tuple(tuple(cells[i] for i in row) for row in ids)


# ---------------------------------------------------------------- C1: cells


class NativeCells:
    """The twin of ``colour._PrimitiveCells`` (and of ``seed_cells``'s
    generic overlaps) against one basis: a native engine with the
    relabelling caches; ``rebased`` shares them with an engine against
    another basis on the same legs."""

    def __init__(self, basis: BornBasis, *, _engine: Any = None, _rings=None):
        self.basis = basis
        self._engine = (
            _native().NativeCellEngine(
                _legs(basis.legs), [_tensor(b) for b in basis.tensors]
            )
            if _engine is None
            else _engine
        )
        self._adjoint = frozenset(
            leg.label for leg in basis.legs if leg.representation == 8
        )
        self._rings: dict[Primitive, int] = dict(_rings or {})

    def rebased(self, basis: BornBasis) -> NativeCells:
        if basis.legs != self.basis.legs:
            raise ValueError("rebased primitive cells need the same colour legs")
        engine = self._engine.rebased([_tensor(b) for b in basis.tensors])
        return NativeCells(basis, _engine=engine, _rings=self._rings)

    def _class(self, seed: ConnectedColorTensor | Primitive) -> tuple:
        if not isinstance(seed, Primitive):
            return (3, [], 0, [], _tensor(seed))
        if seed.kind != "quark-ring":
            return (_KINDS[seed.kind], list(seed.word), 0, [], None)
        # <b|back(P')> = <forward(b)|P'> on the canonical ring P'.
        canonical, back = _ring_relabelling(seed, self._adjoint)
        ring = self._rings.get(canonical)
        if ring is None:
            reduced = _reduced_primitive(canonical, self.basis.legs)
            ring = self._rings[canonical] = self._engine.register_ring(
                _tensor(reduced)
            )
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
    return faces


def gram(basis: BornBasis, threads: int | None = None) -> tuple:
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


def _table_orders(table: Any, ranks: _Ranks) -> tuple[list, list]:
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
    seeds: Iterable[tuple[tuple, Orders]],
    born_components: dict[tuple[int, Orders, int], int],
    tables: Mapping[int, Any] | None = None,
) -> tuple[bytes, bytes]:
    """``_Exporter.seed_records``, natively: the JSON bytes of the seed class
    records and of the seed key records, byte-identical to their
    ``json.dumps(..., separators=(",", ":"))``. ``seeds`` are the exporter's
    seed keys (combination, loop order) in plan order; ``born_components``
    is extended in place in first-seen order. ``tables`` maps ``id(table)``
    of a column table to its ``NativeCellTable`` (default: encoded from its
    Python cells)."""

    seeds = list(seeds)
    accuracy = slice_.request.accuracy
    born_grades = slice_._born_grades()
    distinct: dict[int, Any] = {}
    for column in columns:
        distinct.setdefault(id(column.table), column.table)
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


def seed_records(
    slice_: Any,
    columns: Sequence[Any],
    seeds: Iterable[tuple[tuple, Orders]],
    born_components: dict[tuple[int, Orders, int], int],
    tables: Mapping[int, Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    """``_Exporter.seed_records``'s output (decoded ``project_records``)."""

    class_json, records_json = project_records(
        slice_, columns, seeds, born_components, tables
    )
    return json.loads(class_json), json.loads(records_json)


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
    return handle.degree(
        class_orders,
        basis_orders,
        [(ranks[a], ranks[b]) for a, b in pairs],
        None
        if born_grades is None
        else [((b, ranks[a]), sorted(g)) for (b, a), g in born_grades.items()],
    )


class _Orbits:
    """``assemble._AdjointOrbits`` with its classes built natively."""

    def __init__(self, legs: Sequence[Any], process: Any, born_keys: Iterable):
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

    def rename(self, tensor: ColorTensor) -> dict[int, int]:
        if not self.classes:
            return {}
        return dict(self.native.rename(_structure(tensor)))

    def site_key(self, site: Any) -> Hashable:
        def mask_key(mask: int) -> tuple:
            labels = [i + 1 for i in range(mask.bit_length()) if mask >> i & 1]
            counts = [0] * len(self.classes)
            fixed = []
            for label in labels:
                c = self._class.get(label)
                if c is None:
                    fixed.append(label)
                else:
                    counts[c] += 1
            return tuple(fixed), tuple(counts)

        return (
            site.rule_name,
            site.alternatives,
            tuple(
                (block.field, block.orders, mask_key(block.mask))
                for block in site.blocks
            ),
        )


def adjoint_orbits(legs: Sequence[Any], process: Any, born_keys: Iterable) -> _Orbits:
    """``assemble._AdjointOrbits``, with the classes built natively."""

    return _Orbits(legs, process, born_keys)


def _max(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


def degrees(
    slice_: Any, table_handle: Any = None, threads: int | None = None
) -> dict[str, dict[Orders, int | None]]:
    """``LoopSlice.degrees()`` (the inclusive metric), with every overlap
    degree and the seed-table degrees evaluated natively in one batch. The
    key sets are collected exactly as the oracle collects them.
    ``table_handle`` is the slice table's ``NativeCellTable`` (default: its
    cells built natively from its classes)."""

    from .assemble import LAURENT_POWERS, WholeRuleTerm

    _plan, layout = slice_._scalar_plan(LAURENT_POWERS)

    def grades(name: str, powers=LAURENT_POWERS) -> set[int]:
        return {g for power in powers for g, _c in layout.get((name, power), ())}

    def keys(terms, grade_of) -> set:
        return {
            (item.tensor, item.nc_power + grade)
            for term in terms
            for item in term.colour.terms
            for grade in grade_of(term)
        }

    born = {
        order: frozenset(keys(terms, lambda t: grades(t.entry.coupling.name, (0,))))
        for order, terms in slice_.born_terms.items()
    }
    orbits = _Orbits(slice_.basis.legs, slice_.blocks.process, born.values())
    counterterms: dict[Orders, set] = defaultdict(set)
    closed: set[Hashable] = set()
    for site in slice_.sites.sites:
        identity = orbits.site_key(site)
        if identity in closed:
            continue
        closed.add(identity)
        counterterms[site.orders] |= keys(
            slice_._terms_of(site),
            lambda t: (
                {d for d, _e in t.grades}
                if isinstance(t, WholeRuleTerm)
                else grades(t.entry.coupling.name)
            ),
        )
    if slice_.counterterm_program is not None:
        for item in slice_.counterterm_program.closure.terms:
            term = item.term
            if isinstance(term, WholeRuleTerm):
                found = {d + item.grade for d, _e in term.grades}
            else:
                found = {
                    g + item.grade
                    for g, _c in layout.get(item.coupling, ())
                    if item.coupling[1] in LAURENT_POWERS
                }
            counterterms[item.orders] |= {
                (part.tensor, part.nc_power + grade)
                for part in term.colour.terms
                for grade in found
            }
    # Key sets and queries (bra set, ket set, offset).
    sets: list[list] = []
    set_ids: dict[Any, int] = {}

    def set_id(key: Any, values: Iterable) -> int:
        found = set_ids.get(key)
        if found is None:
            found = set_ids[key] = len(sets)
            sets.append([(_structure(t), p) for t, p in values])
        return found

    queries: list[tuple[int, int, int]] = []

    def query(bra: Orders, ket_key: Any, ket: Iterable, offset: int = 0) -> int:
        queries.append((set_id(("born", bra), born[bra]), set_id(ket_key, ket), offset))
        return len(queries) - 1

    born_queries = [
        (add_orders(a, b), query(a, ("born", b), born[b]))
        for a, b in product(born, repeat=2)
    ]
    born_grades = slice_._born_grades()
    table = slice_.table
    if table_handle is None:
        table_handle = cell_table(slice_.basis, table.classes, threads=threads)
    sector_queries = []
    for sector, pairs in slice_.sectors:
        found = [
            query(a, ("ct", b), counterterms.get(b, set()))
            for a, b in pairs
        ]
        for insertion in slice_.fields:
            if (
                a := next((a for a, b in pairs if b == insertion.orders), None)
            ) is None or insertion.born_order not in born:
                continue
            for grade in grades(slice_.field_names[insertion.identity]):
                found.append(
                    query(
                        a,
                        ("born", insertion.born_order),
                        born[insertion.born_order],
                        grade,
                    )
                )
        sector_queries.append((sector, pairs, found))
    values = _native().overlap_degrees(
        _legs(slice_.basis.legs), sets, queries, orbits.native, threads
    )
    out: dict[str, dict[Orders, int | None]] = {"born": {}, "virtual": {}}
    for squared, q in born_queries:
        out["born"][squared] = _max(out["born"].get(squared), values[q])
    for sector, pairs, found in sector_queries:
        value = table_degree(table, pairs, born_grades, table_handle)
        for q in found:
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
    for every class of the slice table at each of its loop orders.
    ``degrees`` are the virtual sector degrees (default
    ``slice_.degrees()["virtual"]``); ``handles`` maps ``id(table)`` to its
    ``NativeCellTable`` (default: built natively from the table's classes)."""

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
        if found is None:
            found = cell_table(slice_.basis, table.classes, threads=threads)
        return found

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
