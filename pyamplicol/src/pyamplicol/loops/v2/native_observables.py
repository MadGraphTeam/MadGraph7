# SPDX-License-Identifier: 0BSD
"""Observable columns of a stored process, evaluated natively (WP3.1 I6).

A process generated with declared colour histories and spin classes serves
correlated, spin-correlated and helicity-resolved outputs
(``observables.NativeObservables``). Those column plans and their Born,
counterterm and field interferences were formed from the generated slice at
call time, so a loaded process regenerated its slice on first use. Here they
are exported at generation (``StoredObservables.export``) and stored with the
native process (PACBIN members ``point-inputs/observables*``), so no
observable call of a loaded process regenerates anything:

* **column sets**: ``correlated`` (the inclusive Born metric ``born`` and
  every declared history, ``ColumnPlan`` over ``Column(history)``) and
  ``resolved`` (one column per Born helicity row, and at leading colour per
  Born basis element; ``ObservableSlice.resolved_columns``). Per set: the
  seed records of its plan v4 (the slice's families and tries with the
  columns' seed weights; ``_seed_overlay`` over the stored process plan,
  which the set's first call completes, ``with_overlay``; projected
  natively, ``plan_export.NATIVE_COLUMN_RECORDS``), the plan's Born
  components in the tree program's numbering, the lanes of each column per
  sector (direct, swapped; one lane for a symmetric or self-adjoint history,
  ``observables.DirectedColour._self_adjoint``), and the same for the
  companion of the certified-scaleless families ("uv" and "ir");
* **interferences** per column (``loop_v2::trees::TreeProgram::
  set_columns``): the exact overlaps of the Born basis with the ket tensors
  of the counterterm closure under the column's colour operator (the Born
  metric, or a directed history's operator and its swap, which a
  self-adjoint history does not evaluate: ``_Metrics``), with the column's
  LC/NLC degrees, its kept Born rows and bra basis elements; metrics and
  entry lists are stored once per set and shared by the columns that read
  them (``_Programs``); the native trees form them per point next to the
  block rows and Born components;
* **spin sources**: per vector leg the source slots of the trees' and the
  counterterm program's states with their crossing phases, so a source
  vector replaces the leg's wavefunction natively (the Python path's
  ``correlated_exact._fill_correlated_sources``) and the leg keeps its fixed
  (lowest-helicity) row.

The assembly of each column (``observables._Assembly.result``: the bare
loop from the column's lanes, the local components from the native
interferences, the pole split) and the stability estimates
(``ColumnPlan.checked``) are the observables package's, on a structural view
of the stored process. Binary64 calls run the stored plane applications and
exact rationals only (no Symbolica state). The asymmetric Ward mode is a
validation tool of the generated slice and is not stored.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .families import Orders

# 2: a directed metric is stored as its distinct cells and per (ket tensor,
# basis element) the index of its cell (``_table``).
SCHEMA = 2
KINDS = ("correlated", "resolved")


def _orders(orders: Orders) -> list[list[Any]]:
    return [[name, power] for name, power in orders]


def _orders_of(record: Any) -> Orders:
    return tuple((str(name), int(power)) for name, power in record)


def _row(row: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(leg), int(h)) for leg, h in row)


def _text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator if value.denominator == 1 else value)


def _key_record(key: Hashable) -> Any:
    """A column key as JSON: a history id, or (row, basis element)."""

    if isinstance(key, tuple):
        row, b = key
        return [None if row is None else [list(x) for x in row], b]
    return key


def _key_of(record: Any) -> Hashable:
    if isinstance(record, list):
        row, b = record
        return (None if row is None else _row(row), b)
    return record


# --------------------------------------------------------------------- export


def _table(cells: Sequence[Sequence[Sequence[Any]]]) -> dict[str, Any]:
    """The stored form of one metric (``loop_v2::trees`` ``CellsFile``) from
    its cells per (ket tensor t, basis element b), each a list of nonzero
    [N_c power, re, im] by power: the distinct cells in first-seen order
    (t-major) and per t and b the index of its cell. Equal metrics have
    equal tables."""

    index: dict[tuple, int] = {}
    values: list[list[Any]] = []
    ids = []
    for row in cells:
        out = []
        for cell in row:
            key = tuple(map(tuple, cell))
            found = index.get(key)
            if found is None:
                found = index[key] = len(values)
                values.append([list(entry) for entry in cell])
            out.append(found)
        ids.append(out)
    return {"values": values, "ids": ids}


def _cells(basis_literals, ket_literals, overlap) -> dict[str, Any]:
    """The exact overlap cells (N_c power, re, im) of the Born basis with
    every ket tensor under ``overlap(left, right)`` (a dict power -> exact
    coefficient), as a table (``_table``): the oracle of ``_Metrics``."""

    return _table(
        [
            [
                [
                    [power, _text(cell.real), _text(cell.imag)]
                    for power, cell in sorted(overlap(left, right).items())
                    if cell
                ]
                for left in basis_literals
            ]
            for right in ket_literals
        ]
    )


class _Metrics:
    """The exact overlap cells of every colour metric the stored columns
    read, formed once per process: the Born metric is the counterterm
    record's own (``record["cells"]``, ``native_trees._counterterm_program``),
    stored as None (the native columns read the tree program's), a directed
    history's (and its swap) the history's operator overlaps, stored as a
    table (``_table``: a few dozen distinct cells and one index per (ket
    tensor, basis element); g g > g g g g: 29 distinct cells over 120 x 265
    pairs).

    A directed metric is formed natively (package C's colour cells, the
    "cells" stage of ``native_cells.NATIVE_STAGES``) when the direction's
    loop side is gluon emissions (``DirectedColour._absorbable``): the
    operator images ket^dagger bra(b) of the literal Born basis
    (``NativeCells.directed``, natively when the Born side is gluon
    emissions too, else the oracle's ``DirectedColour.operator``) against
    every ket tensor in one native table. ``DirectedColour.overlap`` per
    (b, t) pair is the oracle (``_cells``; the stage off or a quark-pair
    splitting on the loop side): one Python contraction per pair, 96 % of the
    export at five legs with every dipole declared."""

    def __init__(self, observables: Any, record: Mapping[str, Any], ket_literals):
        self.observables = observables
        self.record = record
        self.ket_literals = ket_literals
        slice_ = observables.slice
        self.basis = [tensor.terms[0].tensor for tensor in slice_.basis.tensors]
        self._found: dict[tuple[Any, bool], dict[str, Any]] = {}
        self._alias: dict[tuple[Any, bool], tuple[Any, bool]] = {}
        # A direct metric's native table until its swap is compared with it.
        self._raw: dict[tuple[Any, bool], tuple[list, list]] = {}
        self._engine: Any = None

    def cells(self, column: Any, swapped: bool) -> tuple[tuple[Any, bool], dict | None]:
        """(key, cells) of a column's direct or swapped metric; a swapped
        metric equal to its direct one has the direct one's key, so it is
        stored once: not evaluated for a symmetric or self-adjoint history
        (its swapped seed table is its direct one,
        ``observables.DirectedColour._self_adjoint``: a dipole T_i.T_j), else
        compared with the direct one."""

        key = (column.history.id if column.directed else None, swapped)
        if not column.directed:
            return key, None
        directed = self.observables.directed(column.history)
        if swapped and directed.swapped is directed.table:
            key, swapped = (key[0], False), False
        found = self._found.get(key)
        if found is None:
            direct = (key[0], False)
            raw = self._native(directed, swapped)
            if swapped and raw is not None and self._raw.pop(direct, None) == raw:
                self._alias[key] = direct
                found = self._found[direct]
            elif raw is not None:
                found = self._format(raw)
                if not swapped and directed.swapped is not directed.table:
                    self._raw[key] = raw
            else:

                def overlap(left, right):
                    return directed.overlap(left, right, swapped=swapped)

                found = _cells(self.basis, self.ket_literals, overlap)
            if swapped and key not in self._alias and self._found.get(direct) == found:
                self._alias[key] = direct
            self._found[key] = found
        return self._alias.get(key, key), found

    def _native(self, directed: Any, swapped: bool) -> tuple[list, list] | None:
        """The native table of one direction (class docstring) as its
        distinct cells and per ket tensor and basis element the cell's index
        (``NativeCellTable.decode``), or None where the oracle forms it."""

        from dataclasses import replace

        from . import native_cells

        if "cells" not in native_cells.NATIVE_STAGES or not directed._absorbable(
            swapped
        ):
            return None
        legs = self.observables.slice.basis.legs
        if self._engine is None:
            from .observables import _literal

            # The engine of the literal basis (unit coefficient, no
            # normalisation), as ``DirectedColour.overlap`` reads it, and
            # the ket tensors as generic classes, shared by every metric.
            basis = replace(
                self.observables.slice.basis,
                tensors=tuple(_literal(b, legs) for b in self.basis),
            )
            kets = [_literal(t, legs) for t in self.ket_literals]
            self._engine = (native_cells.NativeCells(basis), basis, kets)
        base, basis, kets = self._engine
        born, loop_side = directed._sides(swapped)
        emissions = native_cells._gluon_emissions(born.emissions)
        absorbed = native_cells._gluon_emissions(loop_side.emissions)
        if emissions is not None and absorbed is not None:
            engine = base.directed(emissions, absorbed)
        else:
            images = replace(
                basis,
                tensors=tuple(directed.operator(b, swapped) for b in basis.tensors),
            )
            engine = base.rebased(images)
        values, ids = engine.table(kets).decode(0, None)
        return values, ids

    def _format(self, raw: tuple[list, list]) -> dict[str, Any]:
        """The stored table (``_table``: ``_cells``' texts, by N_c power,
        nonzero only) of a native table's distinct cells and per (ket
        tensor, basis element) ids (``NativeCellTable.decode``, t-major):
        each distinct cell is formatted once, and the table is re-indexed in
        first-seen order (equal to the oracle's for equal cells)."""

        values, ids = raw
        texts: dict[str, str] = {}

        def text(value: str) -> str:
            found = texts.get(value)
            if found is None:
                found = texts[value] = _text(Fraction(value))
            return found

        # The decoded ids are in first-seen (t-major) order already; two
        # distinct cells that format alike take the first one's index.
        index: dict[tuple, int] = {}
        cells: list[list[Any]] = []
        remap = []
        for value in values:
            cell = []
            for power, re, im in sorted(value):
                re, im = text(re), text(im)
                if re != "0" or im != "0":
                    cell.append((power, re, im))
            key = tuple(cell)
            found = index.get(key)
            if found is None:
                found = index[key] = len(cells)
                cells.append([list(entry) for entry in cell])
            remap.append(found)
        if remap != list(range(len(remap))):
            ids = [[remap[k] for k in row] for row in ids]
        return {"values": cells, "ids": ids}


class _Programs:
    """The interference programs of a column set's columns, shared
    (``TreeProgram::set_columns``): distinct metrics (``cells``, from
    ``_Metrics``) and distinct entry lists (``programs``), each stored once,
    so the helicity-resolved columns of one metric store it once."""

    def __init__(
        self,
        observables: Any,
        metrics: _Metrics,
        outputs: Sequence[tuple[str, Orders, int]],
        born_orders: Sequence[Orders],
        squared_orders: Sequence[Orders],
    ):
        self.observables = observables
        self.metrics = metrics
        self.record = metrics.record
        self.outputs = outputs
        self.born_orders = born_orders
        self.squared_orders = squared_orders
        self.cells: list[dict[str, Any] | None] = []
        self.programs: list[dict[str, Any]] = []
        self._cells: dict[tuple[Any, bool], int] = {}
        self._programs: dict[str, int] = {}

    def _metric(self, column: Any, swapped: bool) -> int:
        """The index of a column's direct (or swapped) metric cells."""

        key, cells = self.metrics.cells(column, swapped)
        found = self._cells.get(key)
        if found is None:
            found = self._cells[key] = len(self.cells)
            self.cells.append(cells)
        return found

    def _program(self, column: Any) -> int:
        """The index of a column's entries: Born outputs per squared Born
        order (``observables._Assembly._local``'s ``born``), then the
        counterterm and field outputs of the inclusive record
        (``native_trees._counterterm_program``: its ``entries`` and
        ``outputs``) with the column's degrees."""

        observables = self.observables
        accuracy = observables.slice.request.accuracy
        degrees = observables.degrees(column.history) if accuracy != "full" else None
        born_index = {order: o for o, order in enumerate(self.born_orders)}
        entries: list[list[Any]] = []
        for s, squared in enumerate(self.squared_orders):
            degree = None if degrees is None else degrees["born"][squared]
            for left in self.born_orders:
                for right in self.born_orders:
                    if _add(left, right) == squared:
                        entries.append(
                            [
                                "born",
                                born_index[left],
                                born_index[right],
                                None,
                                0,
                                s,
                                degree,
                            ]
                        )
        shift = len(self.squared_orders)
        for kind, bra, ket, coupling, grade, output, _degree in self.record["entries"]:
            sector = self.outputs[output][1]
            degree = None if degrees is None else degrees["virtual"][sector]
            entries.append([kind, bra, ket, coupling, grade, output + shift, degree])
        program = {
            "entries": entries,
            "outputs": [0] * shift + [power for _c, _s, power in self.outputs],
            "born_outputs": shift,
        }
        key = json.dumps(program, separators=(",", ":"))
        found = self._programs.get(key)
        if found is None:
            found = self._programs[key] = len(self.programs)
            self.programs.append(program)
        return found

    def column(self, column: Any) -> dict[str, Any]:
        """One column's record: its program, metrics, rows and basis. A
        directed history whose swapped metric is its direct one stores no
        swapped metric: the native column then adds the direct value's
        conjugate (``TreeProgram::set_columns``), the same operations on the
        same bits as a swapped metric equal to the direct one."""

        rows = None
        if column.rows is not None:
            index = {row: k for k, row in enumerate(self.observables.rows)}
            rows = sorted(index[row] for row in column.rows)
        direct = self._metric(column, False)
        swapped = self._metric(column, True) if column.directed else None
        return {
            "program": self._program(column),
            "cells": direct,
            "swapped": None if swapped == direct else swapped,
            "rows": rows,
            "basis": None if column.basis is None else sorted(column.basis),
        }


def _add(left: Orders, right: Orders) -> Orders:
    from .families import add_orders

    return add_orders(left, right)


def _source_slots(source_fill: Mapping[str, Any]) -> list[list[Any]]:
    """(leg label, component start, phase re, phase im) of every vector
    source slot of a runtime schema's source fill (the spin sources'
    targets, ``correlated_exact._fill_correlated_sources``)."""

    from pyamplicol.runtime.symbolica_exact import (
        _crossing_phase,
        _validated_source_ir,
    )

    out = []
    for source in source_fill.get("sources", ()):
        slot = source["value_slot"]
        start, stop = int(slot["component_start"]), int(slot["component_stop"])
        source_ir, _identity, crossing = _validated_source_ir(source)
        if source_ir["wavefunction_family"] != "vector" or stop - start != 4:
            continue
        re, im = _crossing_phase(crossing)
        out.append([int(source["leg_label"]), start, str(re), str(im)])
    return out


# The plan fields an observable column set changes (``plan_export.
# export_plan(columns=...)``: its seed records and Born components).
SEED_FIELDS = (
    "seed_classes",
    "seeds",
    "columns",
    "column_rows",
    "column_weights",
    "born_components",
)


def _seed_overlay(base: Mapping[str, Any], columns: Mapping[str, Any]) -> dict:
    """The seed records of a column set's plan (``columns``, exported without
    trees from the same slice) that replace those of the process plan
    ``base`` (exported with its tree program, whose parameter program reads
    its kernel constants): ``SEED_FIELDS``. The two must read the same seed
    keys in every family leaf, with the same loop orders and Born rows; their
    kernels and contents may be numbered differently (an export's kernel
    numbering follows the slice's earlier exports), which the seed records
    do not read. Only the overlay is stored (``with_overlay`` forms the
    set's plan at its first use), not a second copy of the families."""

    def seeds(program):
        return [
            ([leaf["seed"] for leaf in family["leaves"]], family["orders"])
            for family in program["families"]
        ]

    for key in ("legs", "rows", "helicities", "average"):
        if base[key] != columns[key]:
            raise ValueError(f"a column plan's {key!r} differs from the process's")
    if seeds(base) != seeds(columns) or len(base["seeds"]) != len(columns["seeds"]):
        raise ValueError("a column plan reads other seed keys than the process plan")
    return {key: columns.get(key) for key in SEED_FIELDS}


def with_overlay(base_json: str, overlay: Mapping[str, Any]) -> str:
    """The plan JSON of a column set: the process (or companion) plan with
    the set's seed records (``_seed_overlay``, ``_overlay_of``): a field the
    overlay lacks keeps the plan's, a None field is absent."""

    merged = json.loads(base_json)
    for key in SEED_FIELDS:
        if key not in overlay:
            continue
        value = overlay[key]
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return json.dumps(merged, separators=(",", ":"))


def _plan_of(base_json: str, overlay_json: str) -> str:
    """A column set's plan: ``with_overlay``, or the plan itself when the
    set stores no seed records of its own (``null``)."""

    overlay = json.loads(overlay_json)
    return base_json if overlay is None else with_overlay(base_json, overlay)


def resolution_of(observables: Any) -> str:
    """The resolution of the stored resolved columns (the public
    ``evaluate_resolved``'s): per helicity row, and per Born basis element at
    leading colour."""

    return (
        "helicity-colour" if observables.slice.request.accuracy == "lc" else "helicity"
    )


def column_plans(observables: Any) -> dict[str, Any]:
    """The stored column sets of a process (``observables.ColumnPlan``):
    ``correlated`` (the Born metric and every declared history) and
    ``resolved`` (``resolution_of``)."""

    from .observables import INCLUSIVE, Column, ColumnPlan

    return {
        "correlated": ColumnPlan(
            observables,
            [
                Column(name, observables.history(name))
                for name in (INCLUSIVE, *(h.id for h in observables.histories))
            ],
        ),
        "resolved": ColumnPlan(
            observables, observables.resolved_columns(resolution_of(observables))
        ),
    }


def _overlay_of(seeds: Mapping[str, Any]) -> dict[str, Any] | None:
    """The plan fields (``SEED_FIELDS``) of one set's seed records from
    ``export_plan(column_sets=...)``; None when the plan's own columns serve
    the set (``"same"``)."""

    if seeds.get("same"):
        return None
    overlay = {
        "seed_classes": seeds["seed_classes"],
        "seeds": seeds["seeds"],
        "columns": seeds["columns"],
        "column_rows": seeds["column_rows"],
        "column_weights": seeds["column_weights"],
        "born_components": len(seeds["born_components"]),
    }
    if overlay["seed_classes"] is None:
        # The plan's own seed records (``export_plan``): not stored again.
        del overlay["seed_classes"], overlay["seeds"]
    return overlay


def _flows_record(slice_: Any) -> dict[str, Any]:
    """The leading-colour flows of an LC process (``selection.slice_flows``),
    which its runtime selects (``helicities``/``color_flows``); NLC and full
    colour store none (their record is unchanged)."""

    from . import selection

    if slice_.request.accuracy != "lc":
        return {}
    return {"color_flows": [flow.record() for flow in selection.slice_flows(slice_)]}


@dataclass(frozen=True)
class ObservableSet:
    """One stored column set: the seed records of its plan over the process
    plan (``_seed_overlay``, JSON), the plan's Born components in the tree
    program's numbering, its interference programs (JSON) and the
    companion's seed records and Born components (or None)."""

    overlay_json: str
    plan_components: tuple[int | None, ...]
    columns_json: str
    companion_overlay_json: str | None
    companion_components: tuple[int | None, ...] | None


@dataclass(frozen=True)
class StoredObservables:
    """The stored observable columns of a process (module docstring):
    ``record`` holds the column catalogue (keys, histories, rows, basis
    elements), the lanes of every column per sector, the Born rows, the
    spin classes; ``sets`` the column sets."""

    record: Mapping[str, Any]
    sets: Mapping[str, ObservableSet]

    @classmethod
    def export(
        cls,
        observables: Any,
        exported: Any,
        native: Any,
        plans: Mapping[str, Any],
        companion: Any = None,
    ) -> StoredObservables:
        """The observable columns of a generated process: ``observables``
        its ``ObservableSlice``, ``plans`` its column plans (``column_plans``),
        ``exported`` its plan with the tree program and the plans' seed
        records (``export_plan(column_sets=...)``), ``native`` its
        ``native_process.NativeProcess`` and ``companion`` the stored
        companion's exported plan with the seed records of the plans' twins
        (None: they are exported here, as when a twin runs on a rebuilt
        companion)."""

        from .plan_export import dumps, export_plan

        trees = exported.trees
        counterterms = trees.program.get("counterterms")
        if counterterms is None or not trees.ket_tensors:
            from .native_trees import NativeExportUnsupported

            raise NativeExportUnsupported(
                "observable-columns",
                "stored observable columns need the native counterterm components",
            )
        slice_ = observables.slice
        tree_index = {key: j for j, key in enumerate(trees.components)}
        companion_trees = None if native.scaleless is None else native.scaleless.trees
        companion_index = (
            None
            if companion_trees is None
            else {key: j for j, key in enumerate(companion_trees.components)}
        )
        rows = tuple(exported.rows)
        record: dict[str, Any] = {
            "schema": SCHEMA,
            **_flows_record(slice_),
            "rows": [[list(x) for x in row] for row in rows],
            "basis": len(slice_.basis.tensors),
            "spin_classes": [list(legs) for legs in observables.spin_classes],
            "resolution": resolution_of(observables),
            "sets": {},
            "sources": {
                "trees": _source_slots(trees.program["source_fill"]),
                "counterterms": _source_slots(
                    trees.program["counterterm_trees"]["source_fill"]
                )
                if trees.program.get("counterterm_trees")
                else [],
            },
        }
        metrics = _Metrics(observables, counterterms, trees.ket_tensors)
        sets = {}
        twin_base = None
        for kind, plan in plans.items():
            seeds = exported.column_seeds[kind]
            overlay = _overlay_of(seeds)
            components = tuple(tree_index.get(key) for key in seeds["born_components"])
            companion_overlay = companion_components = None
            twin = plan.scaleless()
            if twin is not None:
                if companion_index is None:
                    raise ValueError("a column companion without the stored companion")
                stored = (
                    None
                    if companion is None or companion.column_seeds is None
                    else companion.column_seeds.get(kind)
                )
                if stored is not None and twin.slice is slice_.scaleless:
                    companion_overlay = _overlay_of(stored)
                    keys = stored["born_components"]
                else:
                    # The twin runs on a rebuilt companion (LC/NLC with
                    # directed histories): its seed records over the stored
                    # companion plan, checked to read the same seed keys.
                    if twin_base is None:
                        twin_base = json.loads(native.scaleless.plan_json)
                    twin_plan = export_plan(
                        twin.slice, forward=False, columns=twin.seed_columns
                    )
                    companion_overlay = _seed_overlay(twin_base, twin_plan.program)
                    keys = twin_plan.born_components
                record["sources"].setdefault(
                    "companion",
                    _source_slots(companion_trees.program["source_fill"]),
                )
                companion_components = tuple(companion_index.get(key) for key in keys)
            programs = _Programs(
                observables,
                metrics,
                trees.counterterm_outputs,
                trees.born_orders,
                trees.squared_orders,
            )
            column_records = [programs.column(column) for column in plan.columns]
            columns_json = json.dumps(
                {
                    "cells": programs.cells,
                    "programs": programs.programs,
                    "columns": column_records,
                    "trees": record["sources"]["trees"],
                    "counterterms": record["sources"]["counterterms"],
                },
                separators=(",", ":"),
            )
            record["sets"][kind] = {
                "columns": [
                    {
                        "key": _key_record(column.key),
                        "history": None
                        if column.history is None
                        else column.history.id,
                        "rows": None
                        if column.rows is None
                        else [[list(x) for x in row] for row in sorted(column.rows)],
                        "basis": None if column.basis is None else sorted(column.basis),
                    }
                    for column in plan.columns
                ],
                "lanes": [
                    [_key_record(key), _orders(sector), direct, swapped]
                    for (key, sector), (direct, swapped) in plan.lanes.items()
                ],
                # Lane labels: the process plan's (and the stored
                # companion's) loop orders, whose families the sets run.
                "families": len(exported.families),
                "orders": [_orders(o) for o in exported.orders],
                "companion_families": None
                if twin is None
                else int(native.scaleless.record["families"]),
                "companion_orders": None
                if twin is None
                else native.scaleless.record["orders"],
                "plan_components": list(components),
                "companion_components": None
                if companion_components is None
                else list(companion_components),
            }
            sets[kind] = ObservableSet(
                dumps(overlay),
                components,
                columns_json,
                # "null" when the companion's own seed records serve the set.
                None if twin is None else dumps(companion_overlay),
                companion_components,
            )
        return cls(record, sets)

    # -- storage ---------------------------------------------------------------

    def members(self) -> dict[str, bytes]:
        out = {
            "observables.json": json.dumps(
                self.record, separators=(",", ":"), sort_keys=True
            ).encode()
        }
        for kind, value in self.sets.items():
            out[f"observables-{kind}-seeds.json"] = value.overlay_json.encode()
            out[f"observables-{kind}-columns.json"] = value.columns_json.encode()
            if value.companion_overlay_json is not None:
                out[f"observables-{kind}-companion-seeds.json"] = (
                    value.companion_overlay_json.encode()
                )
        return out

    @classmethod
    def from_members(cls, members: Mapping[str, bytes]) -> StoredObservables:
        record = json.loads(members["observables.json"])
        if record.get("schema") != SCHEMA:
            raise ValueError(
                "unsupported stored observables schema; regenerate the artifact"
            )
        sets = {}
        for kind, meta in record["sets"].items():
            companion = members.get(f"observables-{kind}-companion-seeds.json")
            sets[kind] = ObservableSet(
                bytes(members[f"observables-{kind}-seeds.json"]).decode(),
                tuple(meta["plan_components"]),
                bytes(members[f"observables-{kind}-columns.json"]).decode(),
                None if companion is None else bytes(companion).decode(),
                None
                if meta["companion_components"] is None
                else tuple(meta["companion_components"]),
            )
        return cls(record, sets)


# --------------------------------------------------------------------- runtime


class StoredObservableSlice:
    """The observables of a loaded process (the subset of
    ``observables.ObservableSlice`` that ``runtime.virtual.
    V2ObservablesRuntime`` reads), on the stored native process and its
    stored columns: no slice."""

    def __init__(
        self,
        process: Any,
        stored: StoredObservables,
        histories: Sequence[Any],
        native: Any,
    ):
        from .native_process import _ProcessView

        self.process = process
        self.stored = stored
        self.native = native
        record = process.record
        self.slice = _ProcessView(record, native)
        born = record["trees"]["born_orders"]
        self.slice.blocks.born_orders = tuple(_orders_of(o) for o in born)
        self.histories = tuple(histories)
        self.spin_classes = tuple(
            tuple(int(leg) for leg in legs) for legs in stored.record["spin_classes"]
        )
        self.rows = tuple(_row(row) for row in stored.record["rows"])
        self.components = tuple(range(int(stored.record["basis"])))
        from .selection import Flow

        self.flows = tuple(
            Flow.from_record(flow) for flow in stored.record.get("color_flows", ())
        )
        # One runner per stored column set, member and executor options: its
        # native executors are built once and serve every call that selects
        # some of its columns.
        self._sets: dict[tuple[Any, ...], StoredColumns] = {}

    def history(self, identifier: str) -> Any:
        from pyamplicol.color.correlator_matrices import ColorCorrelator

        from .observables import INCLUSIVE

        if identifier == INCLUSIVE:
            return ColorCorrelator(INCLUSIVE)
        for history in self.histories:
            if history.id == identifier:
                return history
        raise KeyError(f"undeclared colour correlation {identifier!r}")

    def native_columns(self, kind: str, value: Any, *, member=None, **options):
        """The native column runner of one call kind (``correlated``: the
        requested histories, served by the stored correlated set;
        ``resolved``: its resolution, which must be the stored one)."""

        from .selection import SELECTION

        if kind == SELECTION:
            # A leading-colour selection (``selection.column``): one column
            # set formed from the stored resolved set's records
            # (``selection_parts``), dressed and reduced once per sector.
            # Not kept here: the runtime keeps its most recently used
            # selections (``selection.SelectionCache``).
            found = StoredColumns(
                self, "resolved", member=member, selection=value, **options
            )
            grouped = len(found.plan.columns) > 1
            return _Selection(found, None, group=value.key if grouped else None)
        if kind not in KINDS:
            raise ValueError(f"unknown column set {kind!r}")
        key = (kind, id(member), tuple(sorted(options.items())))
        found = self._sets.get(key)
        if found is None:
            found = self._sets[key] = StoredColumns(
                self, kind, member=member, **options
            )
        if kind == "resolved":
            found.check_resolution(value)
            return _Selection(found, None)
        known = set(found.plan.index)
        wanted = tuple(dict.fromkeys(value))
        missing = [name for name in wanted if name not in known]
        if missing:
            raise KeyError(f"undeclared colour correlations {missing!r}")
        return _Selection(found, wanted)


class _Selection:
    """Some columns of a stored column set (``StoredColumns``): the requested
    histories of a correlated call, every resolved column, or (``group``, a
    leading-colour selection's key) the sum of some resolved columns as one
    output."""

    def __init__(
        self,
        columns: StoredColumns,
        wanted: tuple[Any, ...] | None,
        group: Any = None,
    ):
        self.columns = columns
        self.wanted = wanted
        self.group = group

    def evaluate(self, points, scales, **options):
        return self.columns.evaluate(
            points, scales, wanted=self.wanted, group=self.group, **options
        )


class _ColumnPlanView:
    """``observables.ColumnPlan``'s assembly and estimates on stored lanes
    (``_Assembly`` with the native interferences as its local part)."""

    def __init__(
        self,
        observables: StoredObservableSlice,
        kind: str,
        parts: _SelectionParts | None = None,
    ):
        from .observables import Column

        self.observables = observables
        self.slice = observables.slice
        if parts is not None:
            self.columns = list(parts.columns)
            self.lanes = dict(parts.lanes)
            self.index = {column.key: k for k, column in enumerate(self.columns)}
            return
        meta = observables.stored.record["sets"][kind]
        self.columns = [
            Column(
                _key_of(item["key"]),
                None
                if item["history"] is None
                else observables.history(item["history"]),
                None
                if item["rows"] is None
                else frozenset(_row(row) for row in item["rows"]),
                None if item["basis"] is None else frozenset(item["basis"]),
            )
            for item in meta["columns"]
        ]
        self.lanes = {
            (_key_of(key), _orders_of(sector)): (int(direct), int(swapped))
            for key, sector, direct, swapped in meta["lanes"]
        }
        self.index = {column.key: k for k, column in enumerate(self.columns)}


@dataclass(frozen=True)
class _SelectionParts:
    """A leading-colour selection's column set on a loaded process
    (``selection_parts``): its seed records and interference programs
    (``set``), the resolved set's lane labels (``meta``), its output columns
    and their lanes."""

    set: ObservableSet
    meta: Mapping[str, Any]
    columns: tuple[Any, ...]
    lanes: Mapping[tuple[Any, Orders], tuple[int, int]]
    # Per plan (process, companion) the families whose seeds the selection
    # reads (``live_families``); None keeps every family.
    families: tuple[tuple[int, ...] | None, ...] = (None, None)


def live_families(plan: Mapping[str, Any], overlay: Mapping[str, Any]) -> Any:
    """The families of ``plan`` with a leaf on a seed key that ``overlay``'s
    seed records (``SEED_FIELDS``) make nonzero in some column: the others
    dress zero seeds and contribute exact zeros. None when every family is
    live, or when the plan holds a forward or merged program, which reads
    the families by index."""

    if plan.get("forward") is not None or plan.get("merged") is not None:
        return None
    classes = [
        any(weights for weights in record["columns"])
        for record in overlay["seed_classes"]
    ]
    seeds = [any(classes[c] for c, *_x in key["classes"]) for key in overlay["seeds"]]
    kept = tuple(
        f
        for f, family in enumerate(plan["families"])
        if any(seeds[leaf["seed"]] for leaf in family["leaves"])
    )
    return None if len(kept) == len(plan["families"]) else kept


def _merged_weights(lists: Sequence[Sequence[Sequence[Any]]]) -> list[list[Any]]:
    """The sum of seed-weight lists (``[component, re, im]``, exact texts):
    their concatenation when no component repeats (one Born basis element
    per list), else the exact sum per component."""

    from .plan_export import _fraction_text

    terms = [term for weights in lists for term in weights]
    if len({int(term[0]) for term in terms}) == len(terms):
        return [list(term) for term in terms]
    total: dict[int, tuple[Fraction, Fraction]] = {}
    for j, re, im in terms:
        a, b = total.get(int(j), (Fraction(0), Fraction(0)))
        total[int(j)] = (a + Fraction(re), b + Fraction(im))
    return [
        [j, _fraction_text(re), _fraction_text(im)]
        for j, (re, im) in total.items()
        if re or im
    ]


def selection_parts(observables: StoredObservableSlice, column: Any) -> _SelectionParts:
    """The column set of a leading-colour selection (``selection.column``:
    Born helicity rows and basis elements) of a loaded process, formed from
    its stored resolved set (one column per (row, basis element), dressed
    and reduced each) without regeneration: one seed column per sector whose
    seed weights are the selected basis elements' (their seed-weight columns
    summed) on the selected rows, and one interference column with those
    rows and basis elements. The loop is linear in its seeds and the
    interferences in their bra, so this equals the sum of the selected
    resolved columns, at the cost of one inclusive column rather than of
    every resolved one. Rows whose selected resolved columns read other
    seed-weight columns get a column of their own, summed (``_Selection``
    group)."""

    from .observables import Column

    stored = observables.stored
    meta = stored.record["sets"]["resolved"]
    base = stored.sets["resolved"]
    items = meta["columns"]
    row_index = {row: k for k, row in enumerate(observables.rows)}
    rows, basis = column.rows, column.basis

    def chosen(item: Mapping[str, Any]) -> bool:
        (row,) = item["rows"]
        (b,) = item["basis"]
        return (rows is None or _row(row) in rows) and (basis is None or b in basis)

    kept = [k for k, item in enumerate(items) if chosen(item)]
    if not kept:
        raise ValueError("a selection keeps no resolved column")
    by_key = {json.dumps(item["key"]): k for k, item in enumerate(items)}
    sectors: dict[str, Orders] = {}
    seed_column: dict[tuple[int, str], int] = {}
    for key, sector, direct, swapped in meta["lanes"]:
        if direct != swapped:
            raise ValueError("a resolved column reads one seed column per sector")
        label = json.dumps(sector)
        sectors.setdefault(label, _orders_of(sector))
        seed_column[by_key[json.dumps(key)], label] = int(direct)
    overlays = [json.loads(base.overlay_json)]
    if base.companion_overlay_json is not None:
        overlays.append(json.loads(base.companion_overlay_json))
    if any(overlay is None for overlay in overlays):
        raise ValueError("the resolved set stores no seed records of its own")
    weight_of = [
        overlay.get("column_weights") or list(range(int(overlay["columns"])))
        for overlay in overlays
    ]
    # Per row, per sector and seed plan (process, companion): the seed-weight
    # columns of its selected basis elements; rows with equal ones share a
    # column.
    per_row: dict[int, dict[tuple[str, int], set[int]]] = {}
    for k in kept:
        (row,) = items[k]["rows"]
        h = row_index[_row(row)]
        for label in sectors:
            i = seed_column[k, label]
            for p, weights in enumerate(weight_of):
                per_row.setdefault(h, {}).setdefault((label, p), set()).add(weights[i])
    groups: dict[tuple[Any, ...], list[int]] = {}
    for h in sorted(per_row):
        signature = tuple(
            (label, p, tuple(sorted(found)))
            for (label, p), found in sorted(per_row[h].items())
        )
        groups.setdefault(signature, []).append(h)
    records = json.loads(base.columns_json)
    programs = {
        (
            records["columns"][k]["program"],
            records["columns"][k]["cells"],
            records["columns"][k]["swapped"],
        )
        for k in kept
    }
    if len(programs) != 1:
        raise ValueError("the selected resolved columns read different programs")
    ((program, cells, swapped),) = programs
    every = len(observables.rows)
    out_columns = []
    lanes: dict[tuple[Any, Orders], tuple[int, int]] = {}
    seeds: list[list[tuple[int, ...]]] = [[] for _ in overlays]
    seed_rows: list[list[int] | None] = []
    interference = []
    chosen_basis = None if basis is None else sorted(basis)
    for g, (signature, hs) in enumerate(groups.items()):
        kept_rows = (
            None if len(hs) == every else frozenset(observables.rows[h] for h in hs)
        )
        key = column.key if len(groups) == 1 else (column.key, g)
        out_columns.append(Column(key, None, kept_rows, basis))
        interference.append(
            {
                "program": program,
                "cells": cells,
                "swapped": swapped,
                "rows": None if kept_rows is None else list(hs),
                "basis": chosen_basis,
            }
        )
        found = {(label, p): ws for label, p, ws in signature}
        for label, orders in sectors.items():
            lanes[key, orders] = (len(seed_rows), len(seed_rows))
            seed_rows.append(None if kept_rows is None else list(hs))
            for p in range(len(overlays)):
                seeds[p].append(found[label, p])
    plans = [observables.process.plan_json]
    if len(overlays) > 1:
        plans.append(observables.process.scaleless.plan_json)
    formed = []
    for overlay, plan_json, weight_columns in zip(overlays, plans, seeds, strict=True):
        classes = overlay.get("seed_classes")
        keys = overlay.get("seeds")
        if classes is None:
            plan = json.loads(plan_json)
            classes, keys = plan["seed_classes"], plan["seeds"]
        formed.append(
            {
                "seed_classes": [
                    {
                        "columns": [
                            _merged_weights([record["columns"][w] for w in ws])
                            for ws in weight_columns
                        ]
                    }
                    for record in classes
                ],
                "seeds": keys,
                "columns": len(seed_rows),
                "column_rows": seed_rows
                if any(r is not None for r in seed_rows)
                else None,
                "column_weights": None,
                "born_components": overlay["born_components"],
            }
        )
    from .plan_export import dumps

    columns_json = json.dumps(
        {**records, "columns": interference}, separators=(",", ":")
    )
    families = tuple(
        live_families(json.loads(plan_json), overlay)
        for plan_json, overlay in zip(plans, formed, strict=True)
    )
    return _SelectionParts(
        ObservableSet(
            dumps(formed[0]),
            base.plan_components,
            columns_json,
            None if len(formed) == 1 else dumps(formed[1]),
            base.companion_components,
        ),
        meta,
        tuple(out_columns),
        lanes,
        families + (None,) * (2 - len(families)),
    )


def _plan_class():
    from .observables import ColumnPlan, _Assembly

    class _StoredAssembly(_Assembly):
        def _local(self, column):
            return self.evaluation.column_locals[self.plan.index[column.key]]

        def _local_uv(self, column):
            # The native interferences at the IR regulator 0 (a model with
            # pole regulators, ``StoredColumns._evaluations``), or None.
            found = getattr(self.evaluation, "column_locals_uv", None)
            if found is None:
                if self.slice.pole_regulators is not None:
                    raise ValueError(
                        "the uv and ir pole selections of a model with pole "
                        "regulators need the columns' interferences at the IR "
                        "regulator 0"
                    )
                return None
            return found[self.plan.index[column.key]][1]

    class _StoredPlan(ColumnPlan):
        def __init__(self, view: _ColumnPlanView):
            self.observables = view.observables
            self.slice = view.slice
            self.columns = view.columns
            self.seed_columns = []
            self.lanes = view.lanes
            self.index = view.index
            self._scaleless = [None]
            # Summed outputs (``results(group=...)``): key -> column keys.
            self.groups: dict[Any, tuple[Any, ...]] = {}

        def results(self, evaluation, wanted=None, group=None):
            """The outputs of the columns ``wanted`` (keys; None: every
            column); with ``group`` their sum as the one output ``group``."""

            columns = (
                self.columns
                if wanted is None
                else [self.columns[self.index[key]] for key in wanted]
            )
            with evaluation.numbers.scope():
                assembly = _StoredAssembly(self, evaluation)
                out = {column.key: assembly.result(column) for column in columns}
                if group is None:
                    return out
                from .observables import _sum_results

                self.groups[group] = tuple(out)
                return {group: _sum_results(list(out.values()))}

        def _lanes_of(self, key):
            """A group's lanes are its columns' (``ColumnPlan._lanes_of``)."""

            members = self.groups.get(key)
            if members is None:
                return super()._lanes_of(key)
            out = {}
            for member in members:
                out.update(super()._lanes_of(member))
            return out

    return _StoredPlan


class StoredColumns:
    """``observables.NativeObservables`` for a loaded process: one stored
    column set (every column dressed and reduced per batch: the set's plan
    holds them all), its outputs assembled for the selected columns."""

    def __init__(
        self,
        observables: StoredObservableSlice,
        kind: str,
        *,
        member: Any = None,
        lanes: int | None = None,
        threads: int | None = None,
        schedule: str = "B",
        selection: Any = None,
    ):
        from .plan_export import DEFAULT_LANES

        if kind not in KINDS:
            raise ValueError(f"unknown column set {kind!r}")
        if schedule != "B":
            raise ValueError("stored observable columns run schedule B")
        stored = observables.stored
        self.observables = observables
        self.kind = kind
        self.member = member
        self.lanes = lanes or DEFAULT_LANES
        self.threads = threads
        # ``selection``: a leading-colour selection's column, served by a set
        # formed from the resolved one (``selection_parts``).
        parts = None if selection is None else selection_parts(observables, selection)
        self.set = stored.sets[kind] if parts is None else parts.set
        self.meta = stored.record["sets"][kind] if parts is None else parts.meta
        self.plan = _plan_class()(_ColumnPlanView(observables, kind, parts))
        self.families = (None, None) if parts is None else parts.families
        self._native: dict[tuple[str, int | None, int, bool], Any] = {}
        self._plans: dict[bool, str] = {}
        self.last_report: dict[str, Any] | None = None

    def check_resolution(self, resolution: Any) -> None:
        stored = self.observables.stored.record["resolution"]
        if resolution != stored:
            raise ValueError(
                f"the stored resolved columns have resolution {stored!r}, "
                f"not {resolution!r}"
            )

    # -- native executors -------------------------------------------------------

    def _plan(self, twin: bool) -> str:
        """The set's plan JSON (or its companion's): the stored process (or
        companion) plan with the set's seed records, formed once."""

        found = self._plans.get(twin)
        if found is None:
            process = self.observables.process
            if twin:
                base = process.scaleless.plan_json
                overlay = self.set.companion_overlay_json
            else:
                base, overlay = process.plan_json, self.set.overlay_json
            if overlay is None:
                raise ValueError("the column set stores no companion seeds")
            found = _plan_of(base, overlay)
            kept = self.families[int(twin)]
            if kept is not None:
                # A selection's set: the families of its live seeds only
                # (``live_families``; lanes keep their family labels apart).
                plan = json.loads(found)
                plan["families"] = [plan["families"][f] for f in kept]
                found = json.dumps(plan, separators=(",", ":"))
            self._plans[twin] = found
        return found

    def _executor(self, arithmetic: str, digits: int | None, points: int, twin: bool):
        lanes = self.lanes
        if arithmetic != "double":
            lanes = 1
            while lanes < min(points, self.lanes):
                lanes *= 2
        key = (arithmetic, digits, lanes, twin)
        found = self._native.get(key)
        if found is None:
            process = self.observables.process
            native = self.observables.native
            if twin:
                trees = process.scaleless.trees
                sources = self.observables.stored.record["sources"]
                found = native.V2Executor(
                    self._plan(True),
                    arithmetic,
                    digits,
                    lanes=lanes,
                    threads=self.threads,
                    trees=trees.executor_argument(),
                    # The spin sources only (the companion adds lanes).
                    columns=json.dumps(
                        {
                            "cells": [],
                            "programs": [],
                            "columns": [],
                            "trees": sources.get("companion", []),
                            "counterterms": [],
                        }
                    ),
                    plan_components=list(self.set.companion_components),
                )
            else:
                found = native.V2Executor(
                    self._plan(False),
                    arithmetic,
                    digits,
                    lanes=lanes,
                    threads=self.threads,
                    trees=process.trees.executor_argument(),
                    columns=self.set.columns_json,
                    plan_components=list(self.set.plan_components),
                )
            self._native[key] = found
        return found

    # -- evaluation ---------------------------------------------------------------

    def evaluate(
        self,
        points,
        scales,
        *,
        wanted=None,
        group=None,
        spin=None,
        ward=None,
        parameters=None,
        precision=None,
        poles="all",
        stability=None,
    ):
        """``observables.NativeObservables.evaluate`` on the stored columns:
        the outputs of the columns ``wanted`` (keys; None: every column), or
        with ``group`` their sum as the one output ``group`` (its estimate
        from the budgets of all their lanes)."""

        from .assemble import Precision
        from .observables import (
            _member_key,
            _rotated_point,
            _rotated_spin,
        )
        from .plan_export import FailedPoints
        from .stability import ROTATION, StabilityChecks

        if ward is not None:
            raise NotImplementedError(
                "the asymmetric Ward mode is a validation tool of the generated "
                "slice; stored processes do not serve it"
            )
        stability = stability or StabilityChecks()
        spins = (
            list(spin)
            if spin is not None and not isinstance(spin, Mapping)
            else [spin] * len(points)
        )
        if len(spins) != len(points):
            raise ValueError("one spin source assignment per point")
        member = self.member
        if member is not None:
            points = [member.point(point) for point in points]
            spins = [member.spin(value) for value in spins]
        precision = precision or Precision.double()
        references, failures = self._evaluations(
            points, scales, spins, parameters, precision, poles, stability.frame
        )
        runs = []
        for k, rotation in enumerate(stability.rotations):
            evaluations, failed = self._evaluations(
                [_rotated_point(rotation, point) for point in points],
                scales,
                [_rotated_spin(rotation, value) for value in spins],
                parameters,
                precision,
                poles,
                None,
            )
            runs.append(evaluations)
            for index, reasons in failed.items():
                failures.setdefault(index, []).extend(
                    f"{ROTATION.format(k)}: {reason}" for reason in reasons
                )
        out = []
        for index, reference in enumerate(references):
            if index in failures:
                out.append(None)
                continue
            result = self.plan.results(reference, wanted, group)
            if stability.enabled:
                result = self.plan.checked(
                    result,
                    reference,
                    [
                        (run[index], self.plan.results(run[index], wanted, group))
                        for run in runs
                    ],
                    precision.unit_roundoff,
                    stability.reduction,
                )
            if member is not None:
                result = {_member_key(member, key): v for key, v in result.items()}
            out.append(result)
        if failures:
            raise FailedPoints(out, failures)
        return out

    def _evaluations(self, points, scales, spins, parameters, precision, poles, retest):
        """Point evaluations with every column's lanes reduced natively (and
        re-reduced in the frame ``retest``) and the native interferences;
        also the failures per point."""

        from .assemble import _fraction, _NativePoint, _Numbers
        from .native_trees import batch_parameters
        from .plan_export import NativeExecutor, _fraction_text
        from .poles import has_poles, pole_mode, reducer_poles, splits

        view = self.observables.slice
        process = self.observables.process
        mode = pole_mode(poles)
        evaluations = []
        for point, scale in zip(points, scales, strict=True):
            scale = _fraction(scale)
            values = view._point_parameters(dict(parameters or {}), scale, precision)
            evaluation = _NativePoint(
                view,
                tuple(tuple(_fraction(x) for x in p) for p in point),
                scale,
                values,
                _Numbers(precision),
            )
            evaluation.mode = mode
            evaluation.poles = has_poles(mode)
            evaluations.append(evaluation)
        arithmetic, digits = NativeExecutor._arithmetic(precision, False)
        inputs = {
            "momenta": [
                [_fraction_text(x) for p in e.incoming for x in p] for e in evaluations
            ],
            "scales": [_fraction_text(e.scale) for e in evaluations],
            "parameters": batch_parameters(process.trees, evaluations),
        }
        sources = [self._sources(spin) for spin in spins]
        if any(source is not None for source in sources):
            inputs["sources"] = [
                ([], None) if source is None else source for source in sources
            ]
        options = {
            "poles": reducer_poles(mode),
            "check": retest is not None,
            "rescale": retest == "rescaled",
        }
        native = self._executor(arithmetic, digits, len(evaluations), False)
        output = native.evaluate(inputs, schedule="B", **options)
        # The native report of the latest batch (stages and counts, as
        # ``plan_export.NativeExecutor.last_report``).
        self.last_report = output["report"]
        # A model with pole regulators: the split selections also form every
        # column's interferences with the IR regulator at 0, the
        # counterterms' UV parts (``poles``): one more pass of the tree
        # program, no loop dressing or reduction.
        regulated = None
        if splits(mode) and view.pole_regulators is not None:
            from .plan_export import _regulated

            regulators = view.pole_regulators
            regulated = native.tree_columns(
                {
                    **inputs,
                    "parameters": batch_parameters(
                        process.trees,
                        _regulated(evaluations, regulators, process.trees),
                    ),
                }
            )
            if regulated is None:
                raise ValueError("the stored tree program forms no observable columns")
        extra = None
        if splits(mode) and self.set.companion_overlay_json is not None:
            twin = self._executor(arithmetic, digits, len(evaluations), True)
            twin_inputs = {
                **inputs,
                "parameters": batch_parameters(process.scaleless.trees, evaluations),
            }
            extra = twin.evaluate(twin_inputs, schedule="B", **options)
        families = int(self.meta["families"])
        orders = [_orders_of(o) for o in self.meta["orders"]]
        twin_orders = (
            None
            if self.meta["companion_orders"] is None
            else [_orders_of(o) for o in self.meta["companion_orders"]]
        )
        failed: dict[int, list[str]] = {}
        for index, (evaluation, lanes, checks, failures) in enumerate(
            zip(
                evaluations,
                output["families"],
                output["check"] or [None] * len(evaluations),
                output["failures"],
                strict=True,
            )
        ):
            if extra is not None and extra["failures"][index]:
                failures = [*failures, *extra["failures"][index]]
            if failures:
                failed[index] = [str(failure) for failure in failures]
                continue
            numbers = evaluation.numbers

            def carrier(values, numbers=numbers):
                if numbers.double:
                    return [complex(float(re), float(im)) for re, im in values]
                return [numbers.pair(Decimal(re), Decimal(im)) for re, im in values]

            def read(runs, order_list, offset):
                if runs is None:
                    return None
                return {
                    (offset + family, order_list[order], column): carrier(values)
                    for (family, order, column), values in runs
                }

            scaleless = scaleless_check = None
            if extra is not None:
                scaleless = read(extra["families"][index], twin_orders, families)
                if extra["check"] is not None:
                    scaleless_check = read(extra["check"][index], twin_orders, families)
            with _Numbers(precision).scope():
                evaluation.column_locals = self._locals(
                    output["columns"][index], numbers, evaluation.poles
                )
                if regulated is not None:
                    evaluation.column_locals_uv = self._locals(
                        regulated[index], numbers, True
                    )
                evaluation.set_lanes(
                    read(lanes, orders, 0),
                    read(checks, orders, 0),
                    scaleless,
                    scaleless_check,
                )
        return evaluations, failed

    def _sources(self, spin: Mapping[int, Any] | None):
        """A point's native spin sources: per leg its literal vector (exact
        texts) and the Born rows it keeps (the source legs' fixed rows)."""

        if not spin:
            return None
        legs = tuple(sorted(spin))
        if legs not in self.observables.spin_classes:
            raise ValueError(f"spin legs {legs} are not a declared spin class")
        rows = self.observables.rows
        fixed = {}
        for leg in legs:
            values = {h for row in rows for i, h in row if i == leg - 1}
            if not values:
                raise ValueError(f"leg {leg} has no helicity rows")
            fixed[leg] = min(values)
        keep = [
            all(dict(row)[leg - 1] == h for leg, h in fixed.items()) for row in rows
        ]
        from .observables import _components

        vectors = [
            (int(leg), [[_text(re), _text(im)] for re, im in _components(spin[leg])])
            for leg in legs
        ]
        return (vectors, keep)

    def _locals(self, columns, numbers, poles: bool):
        """Per column the ``_Assembly._local`` pair: the Born per squared
        Born order (not divided by the average) and the counterterm and field
        values per (component, sector, epsilon power) (divided by it), from
        the native interferences."""

        from .ct_sites import LAURENT_POWERS

        trees = self.observables.process.trees
        squared = trees.squared_orders
        outputs = trees.counterterm_outputs
        out = []
        for values in columns:
            values = [_value(value, numbers) for value in values]
            born = {order: values[k] for k, order in enumerate(squared)}
            local = {}
            for k, (component, sector, power) in enumerate(outputs):
                if not poles and power != LAURENT_POWERS[0]:
                    continue
                local[component, sector, power] = values[len(squared) + k]
            out.append((born, local))
        return out


def _value(value, numbers):
    re, im = value
    if numbers.double:
        return complex(float(re), float(im))
    return numbers.pair(Decimal(re), Decimal(im))


__all__ = [
    "ObservableSet",
    "StoredColumns",
    "StoredObservableSlice",
    "StoredObservables",
]
