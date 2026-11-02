# SPDX-License-Identifier: 0BSD
"""Plan v4 export and the native v2 executor bridge (design §3.8, §5.1, WP2.1).

``export_plan`` writes the schedule program of a generated ``LoopSlice`` in the
form the native executor runs (``rusticol_core::loop_v2::plan``). It walks the
same objects as the Python reference executor (``reference_executor``), so the
two run identical programs:

* **contents**: per content its loop-line dimensions, structural q-degree and
  fused entries, and per routing term the kernel bindings that build its fused
  factor X(J) from block-current rows, block momenta and the loop shift, and
  its split vertex kernels with the id of their vertex tensor (shared by the
  terms of one routing key), from which the executor runs steps in the split
  form where gate G0's rule prefers it;
* **kernels**: every catalogue kernel as exact polynomial terms over its
  inputs (``reference_executor.ExactKernel``, in its canonical term order);
  the model factors of its coefficients are numbered kernel constants,
  valued per batch. A content whose tree reading flips the sign of its
  outgoing loop line's mass (``reference_executor.mass_flips``) binds its
  own copy of each kernel, whose constants are valued with the flipped mass;
* **seed classes and seeds**: per seed class and loop orders, and per
  observable column, the exact colour weights K[c, b] N_c^g of each Born
  component (b, Born order, N_c grade g), projected at the request's colour
  accuracy (``colour.SeedTable.seeds``); per seed key (class combination,
  loop orders) its exact weights x_c of the classes, so U0_c = sum_b K
  conj(A0) and U0 = sum_c x_c U0_c are formed natively;
* **families**: per family its trie frame reading, rank and loop orders, the
  leaves (seed key, leaf weight) and the schedule-B steps with their edges
  (child step or leaf, fused factor) and the legs each step leaves open; the
  native loader derives every step's helicity rows and row links from the
  Born rows (``helicities``: per leg its helicities, rows in product order)
  and each content's leg mask and row helicities (``links=True`` also writes
  them, which the loader then checks);
* **forward**: the schedule-F prefix tree with its row links and closures
  (seed key, weight, exact frame map into the family's trie frame).

Per batch, ``NativeExecutor`` evaluates the tree rows, Born amplitudes and
counterterms of every point in Python (``assemble``), hands block-current rows,
Born components, kernel constants, exact momenta, masses and scales to the
native ``V2Executor`` and receives the reduced family integrals, which the
slice's assembly turns into the six components and 19 totals.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any, Literal

from pyamplicol.color.connections import ExactColorCoefficient
from pyamplicol.config import PoleSelection

from . import reference_executor as rx
from .colour import SYMMETRIC_GROUP_FFT, project
from .families import FamilyKey, Orders
from .merged import MergedGraph, merged_graph
from .poly_affine import size
from .refusals import require_formal_nc_tables
from .schedules import forward_graph

PLAN_SCHEMA = 4
# The native report's count of seed-FFT jobs (``loop_execution::v2::
# FFT_SEED_JOBS``): present exactly when an evaluation ran the FFT.
FFT_SEED_JOBS = "symmetric-group-fft seed jobs"
_ONE = ExactColorCoefficient(1)
# Lanes of the native executor's SIMD batches (gate G0, design §5.4).
DEFAULT_LANES = 16
# Distinct points whose slice inputs ``NativeExecutor.family_numerators`` keeps.
_POINT_CACHE = 16

Arithmetic = Literal["double", "double-double", "arbitrary", "exact"]


def _fraction_text(value: Any) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _complex_text(value: Any) -> list[str]:
    value = rx.exact(value)
    return [_fraction_text(value.real), _fraction_text(value.imag)]


def _mass_text(value: Any) -> str | list[str]:
    """A squared plan mass as the executor reads it (``V2Batch``): an exact
    text, or an ``[re, im]`` pair for a complex value (a complex-mass
    scheme's pole mass, M^2 - i M Gamma)."""

    if isinstance(value, int | Fraction):
        return _fraction_text(value)
    value = rx.exact(value)
    if value.imag == 0:
        return _fraction_text(value.real)
    return [_fraction_text(value.real), _fraction_text(value.imag)]


def born_helicities(slice_: Any) -> tuple[tuple[int, ...], ...]:
    """Per leg (label order) its sorted helicities: the Born rows are their
    product (``born_rows``)."""

    labels = sorted(leg.label for leg in slice_.blocks.process.legs)
    if labels != list(range(1, len(labels) + 1)):
        raise ValueError("plan v4 needs legs labelled 1..n")
    helicities: dict[int, set[int]] = defaultdict(set)
    for block in slice_.blocks.blocks:
        if block.is_source:
            for row in block.rows:
                ((leg, value),) = row.helicities
                helicities[leg].add(value)
    return tuple(tuple(sorted(helicities[label])) for label in labels)


def born_rows(slice_: Any) -> tuple[rx.Row, ...]:
    """Born helicity rows in the order of ``assemble`` (the seed rows)."""

    return tuple(
        tuple(enumerate(choice)) for choice in product(*born_helicities(slice_))
    )


# --------------------------------------------------------------------- exporter


@dataclass
class ExportedPlan:
    """A plan v4 program and the Python tables that fill its batch inputs."""

    program: dict[str, Any]
    rows: tuple[rx.Row, ...]
    born_components: tuple[tuple[int, Orders, int], ...]
    block_rows: tuple[tuple[Any, Hashable, int, rx.Row, int], ...]
    # (constant key, mass flips) of every numbered kernel constant.
    constants: tuple[tuple[Hashable, rx.MassFlips], ...]
    masses: tuple[Any, ...]
    orders: tuple[Orders, ...]
    families: tuple[FamilyKey, ...]
    # Lane label of each observable column: its sector for the sector
    # columns, its index for other observable columns.
    columns: tuple[Hashable, ...]
    # The tree program of native per-point inputs (``native_trees``).
    trees: Any = None
    # Per named observable column set (``export_plan(column_sets=...)``) the
    # seed records that replace the plan's for that set: ``seed_classes`` and
    # ``seeds`` (None: the plan's own; ``RawJSON`` when projected natively,
    # ``NATIVE_COLUMN_RECORDS``), ``columns`` (count), ``column_rows``
    # and ``column_weights`` (or None) and the Born components they read
    # (``born_components``, keys); ``{"same": True, ...}`` when the plan's
    # own columns serve the set.
    column_seeds: Mapping[str, Mapping[str, Any]] | None = None
    # The certified seed FFT of the plan's seeds (``fourier_record``; JSON),
    # None when every seed is contracted directly.
    fourier: str | None = None

    def json(self) -> str:
        """The plan JSON, ``json.dumps(program, separators=(",", ":"))``,
        written by the native plan model (``native_lowering``; WP3.1 D2'),
        or by ``python_json`` when ``native_lowering.NATIVE_JSON`` is off."""

        from . import native_lowering

        if native_lowering.NATIVE_JSON:
            return native_lowering.plan_json(self).decode("ascii")
        return self.python_json()

    def python_json(self) -> str:
        """The plan JSON written by ``json`` (the native writer's oracle)."""

        return json.dumps(self.program, separators=(",", ":"))

    def binary(self) -> bytes:
        """The binary columnar plan member (``native_lowering``), read by
        ``loop_v2::plan::read_binary``: the plan's records for an artifact's
        PACBIN member (DESIGN §12.1)."""

        from . import native_lowering

        return native_lowering.plan_binary(self)


@dataclass
class _Object:
    """An open-loop object of the exported walk: a leaf seed or a step output."""

    rows: tuple[rx.Row, ...]
    d_alpha: int
    d_beta: int
    storage: int
    leaf: int = -1
    step: int = -1


def family_rank(trie_rank: int, steps: Iterable[Mapping[str, Any]]) -> int:
    """The rank a family is written with: its trie's (the largest summed
    degree of one ring) or, when larger, its largest step output rank (a
    step's largest source storage ``rank_in`` plus its largest segment
    ``degree``, ``_Exporter._step``), which can exceed every single ring's
    where a trie node joins sources and segments of different degrees (EW
    rings of degree-0 and degree-1 vertices: h h > h h, u u~ > z h,
    w+ w- > w+ w- [virt=QED]); the native loader refuses a step above the
    family rank. A family within its trie rank is written as before."""

    return max([trie_rank, *(step["rank_in"] + step["degree"] for step in steps)])


def _trace_groups(edges, rank: int, shape) -> list[list]:
    """The trace steps of one accumulator and loop species.

    A step reads its sources at the largest source storage and contracts its
    contents at the largest content degree, so its output rank is their sum,
    which the loader requires within the family rank (the largest degree sum
    of the family's rings). Two rings of one family can peak at different
    positions (a bubble whose rings have degree 1 at either propagator: rank
    1, while one step would take 1 + 1; u u~ > z z with QED corrections); the
    edges are then split by source storage into steps that each stay within
    the rank, which the accumulator sums. Otherwise one step, as before, so a
    plan the loader admits is exported byte for byte as before."""

    degree = max(shape(content).degree for content, _source in edges)
    storage = max(source.storage for _content, source in edges)
    if storage + degree <= rank:
        return [edges]
    groups: dict[int, list] = defaultdict(list)
    for content, source in edges:
        groups[source.storage].append((content, source))
    return [groups[key] for key in sorted(groups)]


@dataclass
class _Exporter:
    slice_: Any
    kinematics: rx.CatalogueKinematics
    rows: tuple[rx.Row, ...]
    contents: dict[Any, int] = field(default_factory=dict)
    content_records: list[dict] = field(default_factory=list)
    kernels: dict[tuple[int, rx.MassFlips], int] = field(default_factory=dict)
    kernel_records: list[dict] = field(default_factory=list)
    constants: dict[tuple[Hashable, rx.MassFlips], int] = field(default_factory=dict)
    block_rows: dict[tuple, int] = field(default_factory=dict)
    factors: dict[tuple[int, int], int] = field(default_factory=dict)
    vertices: dict[Hashable, int] = field(default_factory=dict)
    seeds: dict[Hashable, int] = field(default_factory=dict)
    born_components: dict[tuple[int, Orders, int], int] = field(default_factory=dict)
    orders: dict[Orders, int] = field(default_factory=dict)
    masses: dict[Any, int] = field(default_factory=dict)
    # Write the schedule-B row links and step rows (the loader derives them;
    # written, they are checked against the derivation).
    links: bool = False
    # (class, loop order) -> seed-class record of the latest ``seed_records``
    # (the seed Fourier record's loop orbits, ``fourier_record``).
    class_records: dict[tuple[int, Orders], int] = field(default_factory=dict)

    # -- kernels ------------------------------------------------------------

    def _kernel(self, binding, flips: rx.MassFlips) -> int:
        """A kernel record per (catalogue kernel, mass flips): the flips are
        part of its constants' keys (valued by ``NativeExecutor._inputs``), so
        one kernel read with and without flips has two records."""

        found = self.kernels.get((binding.kernel_id, flips))
        if found is not None:
            return found
        kernel = self.kinematics._kernel(binding)
        inputs = []
        for item in kernel.inputs:
            if item.role == "loop-momentum":
                inputs.append(["loop", int(item.component), 0])
            elif item.role == "block-current":
                inputs.append(["current", int(item.component), 0])
            elif item.role == "block-momentum":
                inputs.append(["momentum", int(item.component), 0])
            elif item.role == "model-parameter":
                inputs.append(["parameter", 0, 0])
            else:
                raise ValueError(f"segment kernel input of role {item.role!r}")
        outputs = [
            [
                [
                    self.constants.setdefault((key, flips), len(self.constants)),
                    list(monomial),
                ]
                for key, monomial in terms
            ]
            for terms in kernel.outputs
        ]
        found = self.kernels[(binding.kernel_id, flips)] = len(self.kernel_records)
        self.kernel_records.append({"inputs": inputs, "outputs": outputs})
        return found

    def _block_row(self, slot, part: rx.Row, dimension: int) -> int:
        key = (slot.block, slot.word, slot.chirality, part, dimension)
        return self.block_rows.setdefault(key, len(self.block_rows))

    # -- contents -----------------------------------------------------------

    def _binding(self, binding, index, split: bool, flips: rx.MassFlips) -> dict:
        """A kernel bound to a routing term's slots: resolved inputs and, per
        kernel output, its fused entry (and, for a split vertex binding, the
        flat block-component index nu it multiplies)."""

        kernel = self._kernel(binding, flips)
        current_slots, momentum_slots = [], []
        for slot, dimension in enumerate(binding.block_dimensions):
            current_slots += [(slot, k) for k in range(dimension)]
            momentum_slots += [(slot, mu) for mu in range(4)]
        inputs = []
        for kind, component, _ in self.kernel_records[kernel]["inputs"]:
            if kind == "current":
                if split:
                    raise ValueError("a split vertex kernel reads a block current")
                inputs.append(["current", *current_slots[component]])
            elif kind == "momentum":
                inputs.append(["momentum", *momentum_slots[component]])
            else:
                inputs.append([kind, component, 0])
        if split:
            entries = [
                [
                    index[(e.loop_in, e.out, e.exponents)],
                    rx._flat(e.blocks, binding.block_dimensions),
                ]
                for e in binding.entries
            ]
        else:
            entries = [index[(e.loop_in, e.out, e.exponents)] for e in binding.entries]
        return {"kernel": kernel, "inputs": inputs, "entries": entries}

    def content(self, content) -> tuple[int, rx.ContentShape]:
        shape = self.kinematics.shape(content)
        found = self.contents.get(content)
        if found is not None:
            return found, shape
        index = {entry: i for i, entry in enumerate(shape.fused)}
        row_index = {row: i for i, row in enumerate(shape.rows)}
        flips = rx.mass_flips(self.kinematics.reading, content)
        terms = []
        for term in self.kinematics.lowered(content):
            bindings = [self._binding(b, index, False, flips) for b in term.fused]
            split = [self._binding(b, index, True, flips) for b in term.split]
            dimensions = {tuple(b.block_dimensions) for b in (*term.fused, *term.split)}
            if len(dimensions) != 1:
                raise ValueError(f"content {content.label} mixes block dimensions")
            (dims,) = dimensions
            rows = [
                [
                    row_index[row],
                    [
                        self._block_row(slot, part, dimension)
                        for slot, part, dimension in zip(
                            term.slots, parts, dims, strict=True
                        )
                    ],
                ]
                for row, parts in term.rows
            ]
            terms.append(
                {
                    "slots": [slot.block.mask for slot in term.slots],
                    "dimensions": list(dims),
                    "bindings": bindings,
                    # Terms of one routing key share their split vertex
                    # tensor (``reference_executor.SplitShape.key``).
                    "vertex": self.vertices.setdefault(term.key, len(self.vertices))
                    if split
                    else -1,
                    "split": split,
                    "rows": rows,
                }
            )
        found = self.contents[content] = len(self.content_records)
        mask = content.mask
        for row in shape.rows:
            if sum(1 << leg for leg, _h in row) != mask or len(row) != mask.bit_count():
                raise ValueError(f"content {content.label} has a row off its legs")
        self.content_records.append(
            {
                "d_in": shape.loop_in,
                "d_out": shape.loop_out,
                "degree": shape.degree,
                "rows": len(shape.rows),
                "mask": mask,
                "row_helicities": [[h for _leg, h in row] for row in shape.rows],
                "fused": [[b, b2, *e] for b, b2, e in shape.fused],
                "terms": terms,
            }
        )
        return found, shape

    def factor(self, content: int, shift: int) -> int:
        return self.factors.setdefault((content, shift), len(self.factors))

    # -- seeds --------------------------------------------------------------

    def seed(self, skeleton, contents) -> int:
        slice_ = self.slice_
        reading = rx._reading(skeleton, contents)
        combination = slice_.class_of.get(reading)
        if combination is None:
            raise KeyError("reading without a seed class")
        key = (combination, skeleton.orders)
        found = self.seeds.get(key)
        if found is None:
            found = self.seeds[key] = len(self.seeds)
        return found

    def seed_records(
        self, columns: Sequence[Any]
    ) -> tuple[list[dict], list[dict], list[int] | None]:
        """The seed classes and seed keys of the plan: the projection of
        ``assemble.SectorSeeds`` (``SeedTable.seeds``) in two levels.

        A seed class record holds, per column, the exact projected weights
        K[c, b] of the Born components that the column admits for one class
        c at one loop order; a seed key record holds its combination
        sum_c x_c of class records (``U0 = sum_c x_c U0_c``, formed by the
        executor). A column (``SeedColumn``) has its own seed table (directed
        colour histories, on the classes of the slice's table), sector pairs,
        degree P and Born basis elements; its helicity rows are the plan's
        ``column_rows``. The admitted entries of each column are indexed by
        (class, loop order) once, in ``SeedTable.admitted`` order.

        Columns that differ only by their helicity rows have equal weights:
        they share one seed-weight column (the third value, per column its
        weight column, the plan's ``column_weights``; None when every column
        has its own, so the sector columns of the inclusive plan are written
        as before). A class record holds one weight list per weight column.
        """

        slice_ = self.slice_
        accuracy = slice_.request.accuracy
        # Projected at formal N_c only over N_c-exact blocks (``refusals``).
        require_formal_nc_tables(
            accuracy, (column.table for column in columns), "seed records"
        )
        born_grades = slice_._born_grades()
        admitted: set[tuple[int, Orders]] = set()
        # Per column, once: its table, degree and, per (class, loop order),
        # its admitted Born entries (table order).
        indexed = []
        keys, weight_of_column = weight_columns(columns)
        first: dict[int, int] = {}
        for k, w in enumerate(weight_of_column):
            first.setdefault(w, k)
        for w in range(len(keys)):
            column = columns[first[w]]
            table, degree = column.table, column.degree
            if accuracy != "full" and degree is None:
                degree = table.degree(column.pairs, born_grades)
            entries: dict[tuple[int, Orders], list[tuple[int, Orders]]] = {}
            for c, order, b, born_order in table.admitted(column.pairs):
                admitted.add((c, order))
                if column.basis is not None and b not in column.basis:
                    continue
                entries.setdefault((c, order), []).append((b, born_order))
            indexed.append((table, degree, entries))
        classes: dict[tuple[int, Orders], int] = {}
        self.class_records = classes
        class_records: list[dict] = []
        # Projected cells by (table, class, basis element, degree, grade):
        # columns and loop orders of one class share them.
        projected: dict[tuple[int, int, int, int | None, int], Any] = {}

        def weight_of(table, c: int, b: int, degree: int | None, grade: int):
            key = (id(table), c, b, degree, grade)
            found = projected.get(key)
            if found is None:
                found = projected[key] = project(
                    table.cells[c][b], degree, accuracy, grade
                )
            return found

        def class_record(c: int, order: Orders) -> int:
            found = classes.get((c, order))
            if found is not None:
                return found
            weights_of = []
            for table, degree, entries in indexed:
                weights: dict[int, ExactColorCoefficient] = {}
                for b, born_order in entries.get((c, order), ()):
                    for grade in sorted(born_grades.get((b, born_order), ())):
                        weight = weight_of(table, c, b, degree, grade)
                        if not weight:
                            continue
                        j = self.born_components.setdefault(
                            (b, born_order, grade), len(self.born_components)
                        )
                        weights[j] = weight + weights[j] if j in weights else weight
                weights_of.append(
                    [
                        [j, _fraction_text(w.real), _fraction_text(w.imag)]
                        for j, w in weights.items()
                        if w
                    ]
                )
            found = classes[(c, order)] = len(class_records)
            class_records.append({"columns": weights_of})
            return found

        records = []
        for combination, order in self.seeds:
            for c, _x in combination:
                if (c, order) not in admitted:
                    raise KeyError(f"no sector admits class {c} at {order}")
            records.append(
                {
                    "classes": [
                        [
                            class_record(c, order),
                            _fraction_text(x.real),
                            _fraction_text(x.imag),
                        ]
                        for c, x in combination
                        if x
                    ]
                }
            )
        shared = weight_of_column != list(range(len(weight_of_column)))
        return class_records, records, weight_of_column if shared else None

    # -- schedule B -----------------------------------------------------------

    def family(self, key: FamilyKey, schedule) -> dict:
        slice_ = self.slice_
        plan = slice_.plan
        trie = schedule.trie
        sequence = trie.frame.apply(key)
        before = [0]
        for mask, _mass in sequence:
            before.append(before[-1] | mask)
        full = before[-1]
        leaves, leaf_objects = [], []
        for skeleton, ring in trie.leaves:
            contents = [plan.view.content(segment.label) for segment in ring]
            dimension = self.kinematics.shape(contents[0]).loop_in
            leaves.append(
                {
                    "seed": self.seed(skeleton, contents),
                    "weight": _fraction_text(slice_.weights(skeleton, contents)),
                }
            )
            leaf_objects.append(
                _Object(self.rows, dimension, dimension, 0, leaf=len(leaves) - 1)
            )
        accumulators: dict[Orders, int] = {}
        steps: list[dict] = []
        values: dict[int, _Object] = {}
        for index, node in enumerate(trie.nodes):
            edges = []
            for segment, child in node.edges:
                source = leaf_objects[child] if node.depth == 1 else values.pop(child)
                edges.append((plan.view.content(segment.label), source))
            shift = before[node.depth - 1]
            if node.depth < len(trie.family):
                values[index] = self._step(
                    steps, edges, full ^ before[node.depth], shift, None
                )
                continue
            accumulator = accumulators.setdefault(node.orders, len(accumulators))
            cuts = defaultdict(list)
            for content, source in edges:
                cuts[content.loop_out].append((content, source))
            for species in sorted(cuts):
                for group in _trace_groups(
                    cuts[species], trie.rank, self.kinematics.shape
                ):
                    self._step(steps, group, 0, shift, accumulator)
        if values:
            raise AssertionError("a trie object without a consumer")
        return {
            "sequence": [[mask, self._mass(mass)] for mask, mass in sequence],
            "rank": family_rank(trie.rank, steps),
            "orders": [
                self.orders.setdefault(order, len(self.orders))
                for order in accumulators
            ],
            "leaves": leaves,
            "steps": steps,
        }

    def _mass(self, mass) -> int:
        return self.masses.setdefault(mass, len(self.masses))

    def _step(self, steps, edges, out_mask, shift, accumulator) -> _Object:
        """One schedule-B step. Its helicity rows and row links follow from
        the source rows and the contents' rows (``rx.restrict``): written only
        with ``links``, else the native loader derives them."""

        (d_alpha,) = {source.d_alpha for _content, source in edges}
        seeded = {source.leaf >= 0 for _content, source in edges}
        if len(seeded) != 1:
            raise AssertionError("a step mixes leaf seeds and open-loop objects")
        (seeded,) = seeded
        if accumulator is not None and out_mask:
            raise AssertionError("a closure with undressed legs")
        targets: dict[rx.Row, int] = {}
        records = []
        d_out = set()
        degree = 0
        storage = 0
        for content, source in edges:
            index, shape = self.content(content)
            if shape.loop_in != source.d_beta:
                raise AssertionError("a segment does not fit its open-loop current")
            d_out.add(shape.loop_out)
            degree = max(degree, shape.degree)
            storage = max(storage, source.storage)
            record = {
                "leaf": source.leaf,
                "child": source.step,
                "factor": self.factor(index, shift),
            }
            if self.links:
                block_index = {row: i for i, row in enumerate(shape.rows)}
                links = []
                for position, row in enumerate(source.rows):
                    block = block_index.get(rx.restrict(row, content.mask))
                    if block is not None:
                        target = targets.setdefault(
                            rx.restrict(row, out_mask), len(targets)
                        )
                        links += [target, position, block]
                record["links"] = links
            records.append(record)
        (d_out,) = d_out
        step = {
            "seeded": seeded,
            "trace": -1 if accumulator is None else accumulator,
            "out": out_mask,
            "d_alpha": d_alpha,
            "d_out": d_out,
            "degree": degree,
            "rank_in": storage,
            "edges": records,
        }
        if self.links:
            step["rows"] = len(targets)
        steps.append(step)
        return _Object(
            tuple(targets), d_alpha, d_out, storage + degree, step=len(steps) - 1
        )

    # -- schedule F -----------------------------------------------------------

    def forward(self, family_index: Mapping[FamilyKey, int], accumulator_of) -> dict:
        slice_ = self.slice_
        plan = slice_.plan
        skeletons = rx.plan_skeletons(plan)
        targets = rx.trie_targets(plan)
        graph = forward_graph(skeletons)
        contents = rx._edge_contents(skeletons, "F")
        children = defaultdict(list)
        for parent, child, label in graph.edges:
            children[parent].append((child, label))
        row_of = {row: i for i, row in enumerate(self.rows)}
        states: list[dict] = []
        stack = [
            (child, label, None, 0)
            for root in sorted(graph.entries)
            for child, label in sorted(children[root], key=repr)
        ]
        while stack:
            state, label, parent, dressed_mask = stack.pop()
            content = contents[label]
            index, shape = self.content(content)
            if parent is None:
                source_rows: tuple[rx.Row, ...] = ((),)
                d_alpha, storage = shape.loop_in, 0
            else:
                record = states[parent]
                source_rows = record["_rows"]
                d_alpha, storage = record["d_alpha"], record["rank"]
            links, rows = [], []
            for position, row in enumerate(source_rows):
                for block, block_row in enumerate(shape.rows):
                    links += [len(rows), position, block]
                    rows.append(rx.merge(row, block_row))
            closures = []
            for skeleton in graph.closures.get(state, ()):
                _frame, relabelled = targets[skeleton]
                family = family_index[skeleton.family]
                closures.append(
                    {
                        "family": family,
                        "accumulator": accumulator_of[family][skeleton.orders],
                        "seed": self.seed(skeleton, skeleton.contents),
                        "weight": _fraction_text(
                            slice_.weights(skeleton, skeleton.contents)
                        ),
                        "rows": [row_of.get(row, -1) for row in rows],
                        "relabel": None
                        if relabelled is None
                        else [relabelled.sign, relabelled.shift_mask],
                    }
                )
            states.append(
                {
                    "parent": -1 if parent is None else parent,
                    "factor": self.factor(index, dressed_mask),
                    "seeded": parent is None,
                    "rows": len(rows),
                    "d_alpha": d_alpha,
                    "d_out": shape.loop_out,
                    "degree": shape.degree,
                    "rank": storage + shape.degree,
                    "links": links,
                    "closures": closures,
                    "_rows": tuple(rows),
                }
            )
            mine = len(states) - 1
            for child, child_label in sorted(children[state], key=repr):
                stack.append((child, child_label, mine, dressed_mask | content.mask))
        for record in states:
            del record["_rows"]
        return {"states": states}

    # -- schedule M -----------------------------------------------------------

    def merged(
        self, graph: MergedGraph, family_index: Mapping[FamilyKey, int], accumulator_of
    ) -> dict:
        """The planar state graph of schedule M (``merged.merged_graph``):
        entries (seed key, exact weight), states in execution order with
        their edges (entry or earlier state, fused factor, exact weight,
        helicity row links) and closures (family accumulator, frame map into
        the family's trie frame), and the end of every chunk."""

        entries = []
        objects: list[_Object] = []
        for c, order, weight, content in graph.entries:
            dimension = self.kinematics.shape(content).loop_in
            seed = self.seeds.setdefault((((c, _ONE),), order), len(self.seeds))
            entries.append({"seed": seed, "weight": _complex_text(weight)})
            objects.append(
                _Object(self.rows, dimension, dimension, 0, leaf=len(entries) - 1)
            )
        states: list[dict] = []
        outputs: list[_Object] = []
        for state in graph.states:
            edges = []
            weights = []
            for source, step, weight in state.edges:
                edges.append(
                    (
                        step.content,
                        objects[-1 - source] if source < 0 else outputs[source],
                    )
                )
                weights.append(weight)
            record = self._state(edges, state.out_mask, state.shift)
            for edge, weight in zip(record["edges"], weights, strict=True):
                edge["weight"] = None if weight == _ONE else _complex_text(weight)
            closure = None
            if state.closure is not None:
                family_key, orders, relabelled = state.closure
                family = family_index[family_key]
                closure = {
                    "family": family,
                    "accumulator": accumulator_of[family][orders],
                    "relabel": None
                    if relabelled is None
                    else [relabelled.sign, relabelled.shift_mask],
                }
            record["closure"] = closure
            output = record.pop("_object")
            output.step = len(states)
            outputs.append(output)
            states.append(record)
        return {
            "families": sorted(family_index[key] for key in graph.families),
            "entries": entries,
            "states": states,
            "chunks": list(graph.chunks),
        }

    def _state(self, edges, out_mask, shift) -> dict:
        """One M state: the row links of its edges (as ``_step``)."""

        (d_alpha,) = {source.d_alpha for _content, source in edges}
        (seeded,) = {source.leaf >= 0 for _content, source in edges}
        targets: dict[rx.Row, int] = {}
        records = []
        d_out = set()
        degree = 0
        storage = 0
        for content, source in edges:
            index, shape = self.content(content)
            if shape.loop_in != source.d_beta:
                raise AssertionError("a segment does not fit its open-loop current")
            d_out.add(shape.loop_out)
            degree = max(degree, shape.degree)
            storage = max(storage, source.storage)
            block_index = {row: i for i, row in enumerate(shape.rows)}
            links = []
            for position, row in enumerate(source.rows):
                block = block_index.get(rx.restrict(row, content.mask))
                if block is not None:
                    target = targets.setdefault(
                        rx.restrict(row, out_mask), len(targets)
                    )
                    links += [target, position, block]
            records.append(
                {
                    "entry": source.leaf,
                    "parent": source.step,
                    "factor": self.factor(index, shift),
                    "links": links,
                }
            )
        (d_out,) = d_out
        return {
            "seeded": seeded,
            "rows": len(targets),
            "d_alpha": d_alpha,
            "d_out": d_out,
            "degree": degree,
            "rank_in": storage,
            "edges": records,
            "_object": _Object(
                tuple(targets), d_alpha, d_out, storage + degree, step=-1
            ),
        }


@dataclass(frozen=True)
class SeedColumn:
    """One observable column of a plan: its seed table, sector pairs and
    degree P, and the Born helicity rows and basis elements its seeds keep
    (``None``: all). The sector columns of the inclusive virtual are
    ``sector_columns``; ``observables.ColumnPlan`` builds the others."""

    table: Any
    pairs: frozenset
    degree: int | None
    rows: frozenset | None = None
    basis: frozenset | None = None


def weight_columns(
    columns: Sequence[SeedColumn],
) -> tuple[list[tuple[Any, ...]], list[int]]:
    """The distinct seed-weight columns of ``columns`` (keys: table identity,
    degree, sector pairs, basis elements; a column's helicity rows do not
    change its weights) in first-use order, and per column its weight
    column (``_Exporter.seed_records``, the plan's ``column_weights``)."""

    keys: dict[tuple[Any, ...], int] = {}
    of_column = []
    for column in columns:
        key = (id(column.table), column.degree, column.pairs, column.basis)
        of_column.append(keys.setdefault(key, len(keys)))
    return list(keys), of_column


# The seed records of the stored observable column sets
# (``export_plan(column_sets=...)``) are projected natively: package C's C2
# projection (``native_cells.project_records``) reads the directed tables'
# native cells (``native_cells._DIRECTED``) and writes the records' JSON
# bytes. False runs the exporter's Python projection
# (``_Exporter.seed_records``), the oracle; the bytes are equal. The plan's
# own seed records keep their call site (package D's D3 switch).
NATIVE_COLUMN_RECORDS = True


class RawJSON(bytes):
    """The compact JSON text of a value (``json.dumps(value,
    separators=(",", ":"))``), written into an enclosing record as is
    (``dumps``) instead of being parsed into Python objects first."""


def dumps(record: Mapping[str, Any] | None) -> str:
    """``json.dumps(record, separators=(",", ":"))`` of a mapping (or None)
    whose values may be ``RawJSON``: the same text, a raw value spliced as
    is."""

    if not isinstance(record, Mapping) or not any(
        isinstance(value, RawJSON) for value in record.values()
    ):
        return json.dumps(record, separators=(",", ":"))
    parts = []
    for key, value in record.items():
        text = (
            value.decode()
            if isinstance(value, RawJSON)
            else json.dumps(value, separators=(",", ":"))
        )
        parts.append(f"{json.dumps(key)}:{text}")
    return "{" + ",".join(parts) + "}"


def _native_tables(columns: Sequence[SeedColumn]) -> dict[int, Any]:
    """``id(table)`` -> its native cells for the columns' directed tables
    whose cells package C formed natively (``native_cells.directed_table``
    keeps them per history and direction), so the projection does not
    encode their Python cells again; the others are encoded from them."""

    from . import native_cells

    wanted = {id(column.table) for column in columns}
    out: dict[int, Any] = {}
    for directed, handles in list(native_cells._DIRECTED.items()):
        for swapped, handle in handles.items():
            table = vars(directed).get("swapped") if swapped else directed.table
            if table is not None and id(table) in wanted:
                out[id(table)] = handle
    return out


def _set_records(
    exporter: _Exporter, columns: Sequence[SeedColumn]
) -> tuple[Any, Any, list[int] | None]:
    """A column set's seed records, as ``_Exporter.seed_records`` returns
    them (class records, seed key records, per column its weight column or
    None); natively (``NATIVE_COLUMN_RECORDS``) the two record lists are
    their JSON bytes (``RawJSON``): the projection of each weight column's
    first column (``weight_columns``), ``born_components`` extended in place
    in the same order."""

    if not NATIVE_COLUMN_RECORDS:
        return exporter.seed_records(columns)
    from . import native_cells

    keys, of_column = weight_columns(columns)
    first: dict[int, int] = {}
    for k, w in enumerate(of_column):
        first.setdefault(w, k)
    distinct = [columns[first[w]] for w in range(len(keys))]
    class_json, records_json = native_cells.project_records(
        exporter.slice_,
        distinct,
        exporter.seeds,
        exporter.born_components,
        tables=_native_tables(distinct),
    )
    shared = of_column != list(range(len(of_column)))
    return RawJSON(class_json), RawJSON(records_json), of_column if shared else None


def sector_columns(slice_: Any) -> tuple[SeedColumn, ...]:
    """One column per squared-order sector (the inclusive virtual)."""

    degrees = slice_.degrees()["virtual"] if slice_.request.accuracy != "full" else {}
    return tuple(
        SeedColumn(slice_.table, pairs, degrees.get(sector))
        for sector, pairs in slice_.sectors
    )


FOURIER_SCHEMA = 1


def fourier_record(
    slice_: Any, exporter: _Exporter, columns: Sequence[SeedColumn]
) -> str | None:
    """The seed Fourier record of a plan (design §7.3; read and certified by
    ``loop_v2::fourier::SeedFourier::load``), or None without a certified
    pair the plan's seeds read.

    The orbits are those ``colour.seed_fourier`` detects and certifies
    structurally on the slice's exact seed cells (``slice_.fourier``). The
    record maps them into the plan: per job (weight column, Born order, loop
    order, Born N_c grade) the plan's Born component of each element of each
    Born orbit (-1: no seed weight reads it), the plan's seed-class record
    and sign of each element of each loop orbit (-1: no seed key reads the
    class at that loop order) and the certified pairs whose projected kernel
    is nonzero. It holds no weights: the native loader reads the kernels from
    the plan's weights and certifies every pair again on them. Only weight
    columns of the slice's own table over every basis element (the sector
    columns) are covered; directed and Born-component columns stay direct.
    """

    fourier = slice_.fourier
    if fourier is None:
        return None
    table = slice_.table
    accuracy = slice_.request.accuracy
    born_grades = slice_._born_grades()
    keys, weight_of_column = weight_columns(columns)
    first: dict[int, int] = {}
    for k, w in enumerate(weight_of_column):
        first.setdefault(w, k)
    records = exporter.class_records
    projected: dict[tuple[int, int, int | None, int], bool] = {}

    def live(a: int, j: int, degree: int | None, grade: int) -> bool:
        key = (a, j, degree, grade)
        found = projected.get(key)
        if found is None:
            found = projected[key] = any(
                project(cell, degree, accuracy, grade) for cell in fourier.kernels[a, j]
            )
        return found

    jobs = []
    for w in range(len(keys)):
        column = columns[first[w]]
        if column.table is not table or column.basis is not None:
            continue
        degree = column.degree
        if accuracy != "full" and degree is None:
            degree = table.degree(column.pairs, born_grades)
        for born_order, loop_order in sorted(column.pairs):
            pairs = [
                (a, j)
                for a, j in sorted(fourier.kernels)
                if born_order in table.basis.orders[fourier.born_orbits[a][0]]
                and loop_order in table.orders[fourier.loop_orbits[j][0][0]]
                and any(
                    (c, loop_order) in records for c, _sign in fourier.loop_orbits[j]
                )
            ]
            grades = sorted(
                {
                    grade
                    for a, _j in pairs
                    for b in fourier.born_orbits[a]
                    for grade in born_grades.get((b, born_order), ())
                }
            )
            for grade in grades:
                chosen = [
                    (a, j)
                    for a, j in pairs
                    if any(
                        grade in born_grades.get((b, born_order), ())
                        for b in fourier.born_orbits[a]
                    )
                    and live(a, j, degree, grade)
                ]
                if not chosen:
                    continue
                born = sorted({a for a, _j in chosen})
                loop = sorted({j for _a, j in chosen})
                jobs.append(
                    {
                        "column": w,
                        "born": [
                            [
                                exporter.born_components.get((b, born_order, grade), -1)
                                for b in fourier.born_orbits[a]
                            ]
                            for a in born
                        ],
                        "loop": [
                            [
                                [records.get((c, loop_order), -1), sign]
                                for c, sign in fourier.loop_orbits[j]
                            ]
                            for j in loop
                        ],
                        "pairs": [[born.index(a), loop.index(j)] for a, j in chosen],
                    }
                )
    if not jobs:
        return None
    return json.dumps(
        {"schema": FOURIER_SCHEMA, "degree": fourier.degree, "jobs": jobs},
        separators=(",", ":"),
    )


def export_plan(
    slice_: Any,
    *,
    forward: bool = True,
    columns: Sequence[Any] | None = None,
    merged: MergedGraph | None = None,
    trees: bool = False,
    born_squared: bool = False,
    links: bool = False,
    counterterms: bool = True,
    keep_kets: bool = False,
    column_sets: Mapping[str, Sequence[Any]] | None = None,
) -> ExportedPlan:
    """The plan v4 program of a generated slice (structure only, no numerics).

    ``columns`` are the observable columns (``SeedColumn``); by default one
    per squared-order sector. Columns that keep only some helicity rows
    (helicity-resolved outputs) are listed in the plan's ``column_rows``.
    ``merged`` adds schedule M's planar state graph (``merged.merged_graph``)
    for its certified families; schedule B keeps every family. Its entries
    are primitive seed keys, so they carry one weight list per column like
    every other seed. ``trees`` exports the tree program of native per-point
    inputs (``native_trees.export_trees``; its Born kernels join the plan's),
    with the Born colour metric when ``born_squared`` and the counterterm
    and external-field components when ``counterterms``. ``links`` writes the
    schedule-B helicity rows and row links, which the native loader otherwise
    derives (and with them checks). ``keep_kets`` keeps every counterterm
    ket component in the tree program (stored observable columns,
    ``native_observables``). ``column_sets`` (name -> ``SeedColumn``s) also
    forms the seed records of other column sets on the same families
    (``ExportedPlan.column_seeds``; the plan itself is unchanged): a set's
    plan is this plan with its seed records (``native_observables``), at the
    cost of its seed records only.
    """

    from .assemble import _BlockRows
    from .refusals import require_nc_exact

    # LC/NLC seed records, Born components and counterterm projections need
    # N_c-exact tree blocks: the generation refuses others
    # (``LoopSlice.generate``), and no export serves one (``refusals``).
    require_nc_exact(slice_.blocks, slice_.request.accuracy, slice_.request.process)
    kinematics = rx.CatalogueKinematics(
        slice_.catalogue,
        slice_.compiled.ir.vertex_terms,
        slice_.data,
        _BlockRows(slice_, None),
        (),
        rx.KernelConstants({}, rx.Carrier.f64()),
        reading=slice_.reading,
    )
    helicities = born_helicities(slice_)
    rows = born_rows(slice_)
    exporter = _Exporter(slice_, kinematics, rows, links=links)
    families, keys = [], []
    for key, schedule in slice_.plan.schedules.items():
        keys.append(key)
        families.append(exporter.family(key, schedule))
    orders = tuple(exporter.orders)
    accumulator_of = [
        {orders[o]: k for k, o in enumerate(record["orders"])} for record in families
    ]
    forward_program = (
        exporter.forward({key: i for i, key in enumerate(keys)}, accumulator_of)
        if forward
        else None
    )
    # ``merged`` registers its entries' primitive seed keys, so it runs
    # before the seed records are formed.
    merged_program = (
        exporter.merged(merged, {key: i for i, key in enumerate(keys)}, accumulator_of)
        if merged is not None
        else None
    )
    columns_given = columns
    columns = tuple(sector_columns(slice_) if columns is None else columns)
    seed_classes, seeds, column_weights = exporter.seed_records(columns)
    fourier = fourier_record(slice_, exporter, columns)
    native_trees = None
    if trees:
        from .native_trees import export_trees

        native_trees = export_trees(
            slice_,
            exporter,
            rows,
            born_squared=born_squared,
            with_counterterms=counterterms,
            keep_kets=keep_kets,
        )
    legs = len(slice_.blocks.process.legs)
    row_index = {row: i for i, row in enumerate(rows)}
    program = {
        "schema": PLAN_SCHEMA,
        "legs": legs,
        "rows": len(rows),
        "helicities": [list(h) for h in helicities],
        "columns": len(columns),
        "average": slice_.normalisation.average,
        "born_components": len(exporter.born_components),
        "constants": len(exporter.constants),
        "masses": len(exporter.masses),
        "block_rows": [key[-1] for key in exporter.block_rows],
        "kernels": exporter.kernel_records,
        "contents": exporter.content_records,
        "factors": [list(key) for key in exporter.factors],
        "seed_classes": seed_classes,
        "seeds": seeds,
        "families": families,
        "forward": forward_program,
        "merged": merged_program,
    }
    if any(column.rows is not None for column in columns):
        program["column_rows"] = [
            None
            if column.rows is None
            else sorted(row_index[row] for row in column.rows)
            for column in columns
        ]
    if column_weights is not None:
        program["column_weights"] = column_weights
    born_components = tuple(exporter.born_components)
    column_seeds = None
    if column_sets is not None:
        # After the plan and its tree program: a set's seed records may
        # register further Born components, which the plan does not read.
        column_seeds = {}

        def same(a: SeedColumn, b: SeedColumn) -> bool:
            return (
                a.table is b.table
                and a.pairs == b.pairs
                and a.degree == b.degree
                and a.rows == b.rows
                and a.basis == b.basis
            )

        plan_keys = weight_columns(columns)
        for name, set_columns in column_sets.items():
            set_columns = tuple(set_columns)
            if len(set_columns) == len(columns) and all(
                same(a, b) for a, b in zip(set_columns, columns, strict=True)
            ):
                # The plan's own columns (the Born metric of a process
                # without declared histories): the plan serves the set.
                column_seeds[name] = {"same": True, "born_components": born_components}
                continue
            set_keys, of_column = weight_columns(set_columns)
            if plan_keys == (set_keys, list(range(len(set_keys)))):
                # The plan's own weight columns under other columns (the
                # helicity-resolved columns at full colour): the plan's seed
                # records serve the set, which stores only its column fields.
                set_classes = set_seeds = None
                set_weights = (
                    None if of_column == list(range(len(of_column))) else of_column
                )
            else:
                set_classes, set_seeds, set_weights = _set_records(
                    exporter, set_columns
                )
            column_seeds[name] = {
                "seed_classes": set_classes,
                "seeds": set_seeds,
                "column_weights": set_weights,
                "columns": len(set_columns),
                "column_rows": [
                    None
                    if column.rows is None
                    else sorted(row_index[row] for row in column.rows)
                    for column in set_columns
                ]
                if any(column.rows is not None for column in set_columns)
                else None,
                "born_components": tuple(exporter.born_components),
            }
    return ExportedPlan(
        program=program,
        rows=rows,
        born_components=born_components,
        block_rows=tuple(exporter.block_rows),
        constants=tuple(exporter.constants),
        masses=tuple(exporter.masses),
        orders=orders,
        families=tuple(keys),
        columns=tuple(sector for sector, _pairs in slice_.sectors)
        if columns_given is None
        else tuple(range(len(columns))),
        trees=native_trees,
        column_seeds=column_seeds,
        fourier=fourier,
    )


# --------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class NativeNumerator:
    """One family's native numerator: per observable column, the dense
    coefficients in the family's trie frame (``FamilyNumerator`` layout)."""

    family: FamilyKey
    orders: Orders
    rank: int
    columns: tuple[tuple[Any, ...], ...]


class FailedPoints(ArithmeticError):
    """Some points of a native batch failed (a reduction lane's status, an
    ``ImpreciseResult`` master or a non-finite numerator, also in a rotation
    check's evaluation); the others are complete.

    ``results`` lists every point's ``PointResult`` in input order, ``None``
    for a failed point; ``failures`` maps a failed point's index to its
    reasons. The caller decides how to re-evaluate the failed points (e.g. in
    a higher-precision carrier); nothing is escalated automatically.
    """

    def __init__(self, results: list[Any], failures: Mapping[int, list[str]]):
        self.results = results
        self.failures = dict(failures)
        super().__init__(
            f"{len(self.failures)} of {len(results)} points failed: {self.failures}"
        )


def _scale_dependent(exported: Any, first: Any, other: Any) -> str:
    """The refusal of a Python-input batch whose points differ in a shared
    value (kernel constants, squared masses, the legs' squared masses): the
    points of a batch differ in their renormalisation scale only, so these
    depend on it (a running-coupling or MSbar-mass model; the native inputs
    record them, ``native_trees.scale_dependent_shared``). Names them as the
    native record does (``native_trees._output_label``)."""

    from .native_trees import _output_label

    labels = (
        [_output_label(("constant", key, flips)) for key, flips in exported.constants],
        [f"mass {name}" for name in exported.masses],
        [f"mass of leg {i + 1}" for i in range(len(first[2]))],
    )
    names = [
        label
        for kind, label_list in enumerate(labels)
        for label, a, b in zip(label_list, first[kind], other[kind], strict=True)
        if a != b
    ]
    return (
        f"{', '.join(names)} depend on the renormalisation scale: the points "
        "of one batch need equal kernel constants and masses (a batch takes "
        "one mu_R^2)"
    )


class NativeExecutor:
    """The native v2 executor of a generated slice (one plan, many batches).

    ``arithmetic`` selects the native carrier: ``double`` (binary64, the
    NinjaRS fast engine), ``double-double``, ``arbitrary`` (MPFR at the
    precision's working digits, the generic engine) or ``exact`` (rational
    numerators only, the algebraic oracle of the Python exact run).
    ``forms`` picks each schedule-B step's kernel form: gate G0's rule,
    or every step fused or split (where it can). ``merged`` adds schedule M
    (``merged.merged_graph``; True builds it with its default chunks), which
    ``schedule="M"`` runs for its certified families, and B for the others.

    ``inputs="native"`` forms every point's block rows and Born components in
    the executor (``native_trees``, WP3.1) instead of in Python: a batch then
    hands over only momenta, scales and the external model parameters, from
    which the executor binds the kernel constants, masses, couplings and the
    tree program's model parameters itself. Its results hold the Born (with
    ``born_squared``, at the request's colour accuracy), the ``bare``
    component and, with ``counterterms`` (default), the counterterm and
    external-field components and the complete virtual, formed natively from
    the slice's counterterm closure (``native_trees._counterterm_program``);
    without ``counterterms`` ``virtual`` is absent. Catani's pole prediction
    (``PointResult.ir_poles``) is not formed natively (WP3.1 I5c).

    ``evaluate(..., ward=...)`` runs the asymmetric Ward mode on either input
    path (``ward``): with native inputs the native trees read each point's
    loop-side vector (``loop_sources``), with Python inputs the Python lane
    forms them (the oracle); the slice needs the full tree kernels.
    """

    def __init__(
        self,
        slice_: Any,
        *,
        lanes: int = DEFAULT_LANES,
        threads: int | None = None,
        forward: bool = True,
        exported: ExportedPlan | None = None,
        forms: Literal["rule", "fused", "split"] = "rule",
        merged: MergedGraph | bool = False,
        plan_json: str | None = None,
        inputs: Literal["python", "native"] = "python",
        born_squared: bool = True,
        counterterms: bool = True,
        scaleless: tuple[ExportedPlan, str] | Callable[[], Any] | None = None,
        contraction: Literal["auto", "direct", "fft"] | None = None,
    ):
        if inputs not in ("python", "native"):
            raise ValueError(f"unknown input path {inputs!r}")
        if plan_json is not None and exported is None:
            raise ValueError("plan_json is the serialised exported plan")
        if isinstance(scaleless, tuple) and slice_.scaleless is None:
            raise ValueError("a scaleless companion plan needs the slice's companion")
        self.slice = slice_
        self.inputs = inputs
        # The seed contraction (design §7.3): the request's policy, or the
        # caller's (a companion plan contracts its seeds "auto").
        policy = contraction or slice_.request.colour_contraction
        if policy not in ("auto", "direct", "fft"):
            raise ValueError(f"unknown colour contraction {policy!r}")
        self.contraction = policy
        if merged is True:
            merged = merged_graph(slice_)
        self.exported = exported or export_plan(
            slice_,
            forward=forward,
            merged=merged or None,
            trees=inputs == "native",
            born_squared=born_squared,
            counterterms=counterterms,
        )
        if inputs == "native" and self.exported.trees is None:
            raise ValueError("native inputs need a plan exported with trees=True")
        if policy == "fft" and self.exported.fourier is None:
            # An explicit "fft" requires the FFT: refused, not ignored.
            raise ValueError(
                "colour contraction 'fft' needs a certified symmetric-group "
                "orbit pair among the plan's seeds; request 'auto' or 'direct'"
            )
        self.lanes = lanes
        self.threads = threads
        self.forms = forms
        # ``plan_json``: ``exported.json()`` when the caller already holds it.
        self._plan_json = self.exported.json() if plan_json is None else plan_json
        self._native: dict[tuple[str, int | None, int], Any] = {}
        self._scaleless: list[NativeExecutor | None] = []
        # (exported plan, its JSON) of the companion when the caller holds it
        # (an artifact's stored companion, ``artifact.GeneratedProcess``), or
        # a builder of the companion's executor (observable columns,
        # ``observables.NativeObservables``).
        self._scaleless_plan = scaleless
        # Point inputs of the latest ``family_numerators`` calls (oracle
        # comparisons run several schedules and carriers on one batch).
        self._point_inputs: OrderedDict[str, Any] = OrderedDict()

    def native(
        self,
        arithmetic: Arithmetic,
        digits: int | None = None,
        points: int | None = None,
    ) -> Any:
        """The native executor of one carrier (``digits`` for ``arbitrary``)
        for batches of ``points`` points (``lanes_for``)."""

        lanes = self.lanes_for(arithmetic, points)
        key = (arithmetic, digits, lanes)
        found = self._native.get(key)
        if found is None:
            trees = (
                self.exported.trees.executor_argument()
                if self.inputs == "native" and arithmetic != "exact"
                else None
            )
            found = self._native[key] = self.slice.native.V2Executor(
                self._plan_json,
                arithmetic,
                digits,
                lanes=lanes,
                threads=self.threads,
                forms=self.forms,
                trees=trees,
                # The seed FFT runs in every floating carrier; the exact one
                # (numerators only) and a direct policy contract directly.
                fourier=None
                if arithmetic == "exact" or self.contraction == "direct"
                else self.exported.fourier,
            )
        return found

    @property
    def scaleless(self) -> NativeExecutor | None:
        """The executor of the slice's certified-scaleless families
        (``LoopSlice.scaleless``), which only the "uv" and "ir" pole
        selections run; built on first use (from the companion plan or the
        builder given at construction, else exported then), schedule B
        only; None without such families."""

        if not self._scaleless:
            source = self._scaleless_plan
            built = None
            if callable(source):
                built = source()
            elif self.slice.scaleless is not None:
                exported, plan_json = source or (None, None)
                if self.inputs == "native" and (
                    exported is None or exported.trees is None
                ):
                    # With native inputs the companion forms its own block
                    # rows: its plan is exported with its tree program (no
                    # Born or counterterm outputs; it adds only lanes).
                    exported = plan_json = None
                built = NativeExecutor(
                    self.slice.scaleless,
                    lanes=self.lanes,
                    threads=self.threads,
                    forward=False,
                    exported=exported,
                    forms=self.forms,
                    plan_json=plan_json,
                    inputs=self.inputs,
                    born_squared=False,
                    counterterms=False,
                    # The companion's families carry pole residues only: its
                    # seeds take the FFT where certified, never required.
                    contraction="direct" if self.contraction == "direct" else "auto",
                )
            self._scaleless.append(built)
        return self._scaleless[0]

    def lanes_for(self, arithmetic: Arithmetic, points: int | None) -> int:
        """Lanes per chunk: the configured count in binary64 (SIMD lanes);
        in the other carriers, whose lanes are scalar loops that cost as much
        padded as filled, the smallest supported count holding the batch."""

        if arithmetic == "double" or points is None:
            return self.lanes
        lanes = 1
        while lanes < min(points, self.lanes):
            lanes *= 2
        return lanes

    # -- inputs ---------------------------------------------------------------

    def _points(self, points, scales, parameters, precision, poles, wards=None):
        """Python point inputs; ``wards`` (one Ward value or None per point,
        ``ward``) adds the loop-side vectors (``_PointEvaluation``'s
        ``loop_sources``)."""

        from .assemble import Precision, _fraction, _Numbers, _to_decimal

        precision = precision or Precision.double()
        wards = wards or [None] * len(points)
        evaluations = []
        for point, scale, ward in zip(points, scales, wards, strict=True):
            exact_point = tuple(tuple(_fraction(x) for x in p) for p in point)
            with _Numbers(precision).scope():
                extra = {}
                if ward is not None:
                    from .ward import loop_vector, ward_leg

                    leg = ward_leg(ward)
                    vector = loop_vector(ward, exact_point[leg - 1])
                    extra["loop_sources"] = {
                        leg: tuple(
                            (_to_decimal(re), _to_decimal(im)) for re, im in vector
                        )
                    }
                evaluations.append(
                    self.slice._point(
                        exact_point,
                        _fraction(scale),
                        dict(parameters or {}),
                        precision,
                        poles,
                        **extra,
                    )
                )
        return precision, evaluations

    def _cached_points(self, points, scales, parameters, precision):
        """``_points`` with poles, through a cache of the inputs of the last
        ``_POINT_CACHE`` distinct points (read only: ``family_numerators``
        never changes an evaluation)."""

        from .assemble import Precision

        precision = precision or Precision.double()
        common = repr((sorted((parameters or {}).items()), precision))
        keys = [
            repr((point, scale)) + common
            for point, scale in zip(points, scales, strict=True)
        ]
        cache = self._point_inputs
        missing = [k for k, key in enumerate(keys) if key not in cache]
        if missing:
            _precision, fresh = self._points(
                [points[k] for k in missing],
                [scales[k] for k in missing],
                parameters,
                precision,
                PoleSelection.ALL,
            )
            for k, evaluation in zip(missing, fresh, strict=True):
                cache[keys[k]] = evaluation
        evaluations = []
        for key in keys:
            cache.move_to_end(key)
            evaluations.append(cache[key])
        while len(cache) > max(_POINT_CACHE, len(keys)):
            cache.popitem(last=False)
        return precision, evaluations

    def _inputs(self, evaluations, arithmetic: Arithmetic) -> dict[str, Any]:
        """Batch inputs: exact momenta, scales, masses and the legs' squared
        masses (the on-shell invariants, DESIGN §11.1); carrier values of
        block rows, Born components and kernel constants.

        Every carrier value is formed in its point's Decimal context: block
        rows carry the tree engine's Wick signs and kernel constants round
        their irrational factors, both in the carrier's arithmetic.
        """

        from .assemble import _BlockRows, _leg_masses, _mass_values

        exported = self.exported
        slice_ = self.slice

        def value(x):
            if arithmetic == "double":
                z = complex(x)
                return [z.real, z.imag]
            if arithmetic == "exact":
                return _complex_text(x)
            return [str(x.real), str(x.imag)]

        def shared(evaluation):
            """Kernel constants (with their contents' mass flips), squared
            masses and the legs' squared masses of a point's parameters."""

            constants = evaluation.constants
            if arithmetic == "exact":
                constants = rx.KernelConstants(
                    constants.parameters,
                    rx.Carrier.exact(evaluation.numbers.precision.working_digits),
                )
            signed = {(): constants}

            def constant(key, flips):
                found = signed.get(flips)
                if found is None:
                    found = signed[flips] = rx.signed_constants(constants, flips)
                return found.constant(key)

            working = evaluation.numbers.precision.working_digits
            # A complex-mass scheme's pole mass squared is complex (an
            # [re, im] pair), as the native lane binds a ["complex", o] plan
            # mass.
            masses = _mass_values(
                slice_.compiled, evaluation.parameters, exported.masses, working
            )
            return (
                [value(constant(key, flips)) for key, flips in exported.constants],
                [_mass_text(masses[m]) for m in exported.masses],
                [
                    _fraction_text(m)
                    for m in _leg_masses(slice_, evaluation.parameters, working)
                ],
            )

        batch: dict[str, Any] = {
            "momenta": [
                [_fraction_text(x) for p in e.incoming for x in p] for e in evaluations
            ],
            "scales": [_fraction_text(e.scale) for e in evaluations],
            "block_rows": [],
            "born": [],
        }
        if self.inputs == "native":
            from .native_trees import batch_parameters

            # The executor binds kernel constants, masses, couplings and the
            # tree program's model parameters from the external parameters.
            batch["parameters"] = batch_parameters(exported.trees, evaluations)
            del batch["block_rows"], batch["born"]
            # Per point its loop-side vectors (exact texts; the Ward mode).
            loop = [getattr(e, "loop_sources", None) for e in evaluations]
            if any(loop):
                batch["loop_sources"] = [
                    [
                        (
                            leg,
                            [
                                [_fraction_text(re), _fraction_text(im)]
                                for re, im in vector
                            ],
                        )
                        for leg, vector in sorted((sources or {}).items())
                    ]
                    for sources in loop
                ]
            return batch
        for evaluation in evaluations:
            with evaluation.numbers.scope():
                common = shared(evaluation)
                source = _BlockRows(slice_, evaluation.rows)
                rows = [
                    value(x)
                    for block, word, chirality, row, dimension in exported.block_rows
                    for x in source.value(block, word, chirality, row, dimension)
                ]
            # The scale may change per point; kernels and masses may not.
            if batch.setdefault("shared", common) != common:
                raise ValueError(_scale_dependent(exported, batch["shared"], common))
            batch["block_rows"].append(rows)
            amplitudes = [_born_row(evaluation, row) for row in exported.rows]
            batch["born"].append(
                [
                    value(row.get((b, order), {}).get(grade, 0))
                    for row in amplitudes
                    for b, order, grade in exported.born_components
                ]
            )
        batch["constants"], batch["masses"], batch["leg_masses"] = batch.pop("shared")
        return batch

    @staticmethod
    def _arithmetic(precision, exact: bool) -> tuple[Arithmetic, int | None]:
        if exact:
            return "exact", None
        if precision.arithmetic in ("double", "double-double"):
            return precision.arithmetic, None
        return "arbitrary", precision.working_digits

    # -- runs -------------------------------------------------------------------

    def family_numerators(
        self,
        points: Sequence[Sequence[Sequence[Any]]],
        scales: Sequence[Any],
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Any = None,
        schedule: Literal["B", "F", "M"] = "B",
        exact: bool = False,
        double_double: bool = False,
    ) -> list[dict[tuple[FamilyKey, Orders], NativeNumerator]]:
        """Native family numerators per point (one column per sector), in the
        families' trie frames: the comparison with ``LoopSlice.family_numerators``.

        ``exact`` runs the native rational carrier on the carrier's tree rows
        and Born amplitudes read as exact rationals and the exact kernel
        constants (``reference_executor.KernelConstants`` stand-ins);
        ``double_double`` runs the native DoubleFloat carrier on the inputs
        of an arbitrary ``precision``. Calls on the same points, scales,
        parameters and precision share the slice's point inputs
        (``_cached_points``).
        """

        precision, evaluations = self._cached_points(
            points, scales, parameters, precision
        )
        arithmetic, digits = self._arithmetic(precision, exact)
        inputs = self._inputs(evaluations, arithmetic)
        if double_double:
            if arithmetic != "arbitrary":
                raise ValueError("double-double reads arbitrary-precision inputs")
            arithmetic, digits = "double-double", None
        native = self.native(arithmetic, digits, len(evaluations))
        raw = native.numerators(inputs, schedule)
        return [self._numerators(point, arithmetic) for point in raw]

    def _numerators(self, point, arithmetic) -> dict:
        exported = self.exported
        out = {}
        for family, order, rank, columns in point:
            key = exported.families[family]
            orders = exported.orders[order]
            out[(key, orders)] = NativeNumerator(
                key,
                orders,
                rank,
                tuple(
                    tuple(_carrier_value(z, arithmetic) for z in column)
                    for column in columns
                ),
            )
        return out

    def evaluate(
        self,
        points: Sequence[Sequence[Sequence[Any]]],
        scales: Sequence[Any],
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Any = None,
        poles: PoleSelection | str = PoleSelection.ALL,
        stability: Any = None,
        schedule: Literal["B", "M"] = "B",
        ward: Any = None,
    ) -> list[Any]:
        """The complete virtual of every point (``LoopSlice.evaluate``), with
        dressing and reduction in the native executor (``schedule`` B, or M
        for the plan's schedule-M families). ``poles`` is the pole selection
        (``PoleSelection`` or its value: ``"all"``, ``"none"``, ``"uv"``,
        ``"ir"``; ``poles``).

        ``ward`` is the asymmetric Ward mode (``ward``): one Ward value for
        every point or a list with one per point (None: physical), on a
        slice generated on the full tree kernels; each result then also
        holds ``loop_born``.

        Rotation checks re-evaluate the rotated points natively; the
        reduction check is the native re-test pass. A failed point fails
        alone: ``FailedPoints`` carries the complete results of the others.
        """

        from .assemble import _fraction
        from .stability import ROTATION, StabilityChecks, StabilityEstimate

        stability = stability or StabilityChecks()
        wards = None
        if ward is not None:
            from .ward import point_wards, require_full_kernels

            wards = point_wards(ward, len(points))
            if any(value is not None for value in wards):
                require_full_kernels(self.slice)
            else:
                wards = None
        precision, evaluations, failures = self._evaluated(
            points,
            scales,
            parameters,
            precision,
            poles,
            stability.frame,
            schedule,
            wards,
        )
        rotated = []
        for k, rotation in enumerate(stability.rotations):
            moved = [
                rotation.apply(tuple(tuple(_fraction(x) for x in p) for p in point))
                for point in points
            ]
            turned = None
            if wards is not None:
                from .ward import rotated_ward

                turned = [rotated_ward(rotation, value) for value in wards]
            _, runs, failed = self._evaluated(
                moved, scales, parameters, precision, poles, None, schedule, turned
            )
            rotated.append(runs)
            for index, reasons in failed.items():
                failures.setdefault(index, []).extend(
                    f"{ROTATION.format(k)}: {reason}" for reason in reasons
                )
        out: list[Any] = []
        for index, reference in enumerate(evaluations):
            if index in failures:
                out.append(None)
                continue
            if not stability.enabled:
                out.append(reference.result)
                continue
            budgets = {}
            for k, runs in enumerate(rotated):
                budgets[ROTATION.format(k)] = reference.rotation_budget(runs[index])
            if stability.reduction:
                budgets[stability.reduction] = reference.retest_budget
            components = reference.result.components
            estimate = StabilityEstimate.from_checks(
                components.get("virtual", components["bare"]),
                budgets,
                scale=reference.absolute,
                roundoff=precision.unit_roundoff,
                families=reference.family_deviations,
            )
            out.append(replace(reference.result, stability=estimate))
        if failures:
            raise FailedPoints(out, failures)
        return out

    def native_inputs(self, points, scales, parameters=None, precision=None):
        """The executor's native tree inputs of a batch (``inputs="native"``):
        per point ``(block_rows, born, amplitudes, born_squared, counterterms)``
        (the counterterm and field outputs with poles, or None)."""

        from .assemble import Precision

        precision = precision or Precision.double()
        arithmetic, digits = self._arithmetic(precision, False)
        evaluations = self._native_points(points, scales, parameters, precision)
        native = self.native(arithmetic, digits, len(evaluations))
        return native.tree_inputs(self._inputs(evaluations, arithmetic))

    def seed_values(
        self,
        points,
        scales,
        parameters=None,
        precision=None,
        contraction: Literal["auto", "direct", "fft"] = "auto",
    ) -> list[list[tuple[Any, ...]]]:
        """The seeds U0 of a batch (``inputs="native"``): per point, per seed
        key, its ``rows * columns`` values in the carrier (``complex`` in
        binary64, ``reference_executor.Complex`` of ``Decimal`` otherwise),
        contracted as ``evaluate`` with ``contraction`` contracts them: the
        oracle of the certified seed FFT (design §7.3), whose values equal
        the direct contraction's to the contraction's own rounding, without
        the reduction's amplification."""

        from .assemble import Precision

        precision = precision or Precision.double()
        arithmetic, digits = self._arithmetic(precision, False)
        evaluations = self._native_points(points, scales, parameters, precision)
        native = self.native(arithmetic, digits, len(evaluations))
        raw = native.seed_values(
            self._inputs(evaluations, arithmetic), contraction=contraction
        )
        return [
            [tuple(_carrier_value(z, arithmetic) for z in key) for key in point]
            for point in raw
        ]

    def bound_masses(self, points, scales, parameters=None, precision=None):
        """The squared masses the native lane binds for a batch
        (``inputs="native"``, ``TreeProgram::bind``): per plan mass and per
        leg (the on-shell invariants, DESIGN §11.1), exact ``Fraction``s; a
        complex plan mass (``["complex", o]``, a complex-mass scheme's pole
        mass) as its ``(real, imaginary)`` pair of ``Fraction``s."""

        from .assemble import Precision

        precision = precision or Precision.double()
        arithmetic, digits = self._arithmetic(precision, False)
        evaluations = self._native_points(points, scales, parameters, precision)
        native = self.native(arithmetic, digits, len(evaluations))
        masses, legs, imaginary = native.bound_masses(
            self._inputs(evaluations, arithmetic)
        )
        plan = [Fraction(m) for m in masses]
        if imaginary:
            plan = [
                (m, Fraction(i)) if Fraction(i) != 0 else m
                for m, i in zip(plan, imaginary, strict=True)
            ]
        return plan, [Fraction(m) for m in legs]

    def _native_points(self, points, scales, parameters, precision):
        from .assemble import _fraction, _NativePoint, _Numbers

        slice_ = self.slice
        out = []
        for point, scale in zip(points, scales, strict=True):
            scale = _fraction(scale)
            values = slice_._point_parameters(dict(parameters or {}), scale, precision)
            out.append(
                _NativePoint(
                    slice_,
                    tuple(tuple(_fraction(x) for x in p) for p in point),
                    scale,
                    values,
                    _Numbers(precision),
                )
            )
        return out

    def _evaluated(
        self,
        points,
        scales,
        parameters,
        precision,
        poles,
        retest,
        schedule="B",
        wards=None,
    ):
        """Evaluate a batch natively; returns the precision, the per-point
        evaluations (failed points unassembled) and the failures per point
        index. ``wards``: one Ward value or None per point (``ward``)."""

        from .assemble import Precision, _fraction, _Numbers

        if self.inputs == "native":
            from .poles import has_poles, pole_mode

            precision = precision or Precision.double()
            evaluations = self._native_points(points, scales, parameters, precision)
            for evaluation in evaluations:
                evaluation.mode = pole_mode(poles)
                evaluation.poles = has_poles(evaluation.mode)
            if wards is not None:
                from .ward import loop_vector, ward_leg

                for evaluation, point, ward in zip(
                    evaluations, points, wards, strict=True
                ):
                    if ward is not None:
                        leg = ward_leg(ward)
                        momentum = [_fraction(x) for x in point[leg - 1]]
                        evaluation.loop_sources = {leg: loop_vector(ward, momentum)}
        else:
            precision, evaluations = self._points(
                points, scales, parameters, precision, poles, wards
            )
        failed = self.reduce(evaluations, precision, poles, retest, schedule)
        # The executor's report names the seed contraction it ran: the
        # certified pairs by the symmetric-group FFT, or every weight directly.
        algorithm = (
            SYMMETRIC_GROUP_FFT
            if self.last_report["counts"].get(FFT_SEED_JOBS)
            else "direct"
        )
        for index, evaluation in enumerate(evaluations):
            if index in failed:
                continue
            with _Numbers(precision).scope():
                evaluation.colour_contraction = algorithm
                evaluation._assemble()
        return precision, evaluations, failed

    def reduce(
        self,
        evaluations,
        precision,
        poles: PoleSelection | str,
        retest: str | None,
        schedule: Literal["B", "M"] = "B",
    ) -> dict[int, list[str]]:
        """Dress and reduce prepared point evaluations (``assemble``'s
        ``_PointEvaluation``, or ``_NativePoint`` with native inputs) natively
        in ``precision``'s carrier and ``schedule`` (B, or M for the plan's
        schedule-M families): sets each evaluation's ``families`` (and
        ``retest_families`` in the re-test frame ``retest``,
        ``stability.FRAMES``), lanes labelled (family, loop orders, column
        label), and with native inputs its ``born`` and counterterm and field
        components. Returns the failures per point index; a failed point's
        evaluation is left without families. ``poles`` is the pole mode
        (``poles``): "uv" and "ir" also run the scaleless companion
        (``scaleless``: the slice's on the sector columns, or the one built
        for the plan's observable columns)."""

        from .assemble import _Numbers
        from .poles import pole_mode, reducer_poles, splits

        mode = pole_mode(poles)
        sectors = tuple(sector for sector, _pairs in self.slice.sectors)
        if (
            splits(mode)
            and tuple(self.exported.columns) != sectors
            and not callable(self._scaleless_plan)
        ):
            raise ValueError(
                "observable columns in the uv and ir pole selections need the "
                "companion of their column plan (observables.NativeObservables)"
            )
        arithmetic, digits = self._arithmetic(precision, False)
        native = self.native(arithmetic, digits, len(evaluations))
        options = {
            "poles": reducer_poles(mode),
            "check": retest is not None,
            "rescale": retest == "rescaled",
        }
        output = native.evaluate(
            self._inputs(evaluations, arithmetic),
            schedule=schedule,
            contraction=self.contraction,
            **options,
        )
        exported = self.exported
        self.last_report = output["report"]
        self.last_bare = output["bare"]
        # A model with pole regulators: in the "uv" and "ir" selections the
        # native trees also form the counterterm and field components with
        # the IR regulator at 0, the counterterms' UV parts (``poles``).
        regulated = None
        regulators = self.slice.pole_regulators
        if (
            splits(mode)
            and regulators is not None
            and self.inputs == "native"
            and output.get("counterterms") is not None
        ):
            # (block rows, Born, amplitudes, |A0|^2, counterterms) per batch.
            regulated = native.tree_inputs(
                self._inputs(
                    _regulated(evaluations, regulators, exported.trees), arithmetic
                )
            )[4]
        # The certified-scaleless families carry part of the bare UV and IR
        # poles; only the "uv" and "ir" selections dress them (schedule B).
        twin = self.scaleless if splits(mode) else None
        extra = None
        if twin is not None:
            extra = twin.native(arithmetic, digits, len(evaluations)).evaluate(
                twin._inputs(evaluations, arithmetic), schedule="B", **options
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

            def label(lane, exported=exported):
                family, order, column = lane
                return (
                    exported.families[family],
                    exported.orders[order],
                    exported.columns[column],
                )

            def read(runs, exported=exported):
                if runs is None:
                    return None
                return {label(lane, exported): carrier(values) for lane, values in runs}

            scaleless = scaleless_check = None
            if extra is not None:
                scaleless = read(extra["families"][index], twin.exported)
                if extra["check"] is not None:
                    scaleless_check = read(extra["check"][index], twin.exported)
            with _Numbers(precision).scope():
                if output.get("counterterms") is not None:
                    evaluation.counterterms = {
                        key: numbers.real(Decimal(str(re)))
                        for key, (re, _im) in zip(
                            exported.trees.counterterm_outputs,
                            output["counterterms"][index],
                            strict=True,
                        )
                    }
                    evaluation.native_counterterms = True
                if regulated is not None:
                    evaluation.counterterms_uv = {
                        key: numbers.real(Decimal(str(re)))
                        for key, (re, _im) in zip(
                            exported.trees.counterterm_outputs,
                            regulated[index],
                            strict=True,
                        )
                    }
                if output.get("born") is not None:
                    evaluation.born = {
                        order: numbers.real(Decimal(str(re)))
                        for order, (re, _im) in zip(
                            exported.trees.squared_orders,
                            output["born"][index],
                            strict=True,
                        )
                    }
                if output.get("loop_born") is not None and getattr(
                    evaluation, "loop_sources", None
                ):
                    evaluation.loop_born = {
                        order: numbers.real(Decimal(str(re)))
                        for order, (re, _im) in zip(
                            exported.trees.squared_orders,
                            output["loop_born"][index],
                            strict=True,
                        )
                    }
                evaluation.set_lanes(
                    read(lanes), read(checks), scaleless, scaleless_check
                )
        return failed


def _regulated(evaluations: Sequence[Any], regulators: Any, trees: Any) -> list[Any]:
    """Shallow copies of native point evaluations with the IR regulator at 0
    (``poles.regulated_parameters``): their native tree inputs (of the tree
    program ``trees``, which must bind the regulator: fail closed, as the
    SDK's ``V2Runtime``) are the counterterms' UV parts."""

    from .poles import regulated_parameters

    if regulators.ir not in trees.program["parameters"]["names"]:
        raise ValueError(
            f"the pole regulator {regulators.ir!r} is not a parameter of the "
            "tree program; regenerate the process"
        )
    out = []
    for evaluation in evaluations:
        twin = copy.copy(evaluation)
        twin.parameters = regulated_parameters(evaluation.parameters, regulators)
        out.append(twin)
    return out


def _born_row(evaluation, row) -> Mapping:
    """The Born amplitudes of one exported helicity row at a point: empty
    for a row the point's spin sources drop (a fixed leg with another
    helicity, ``assemble._PointEvaluation(fixed=...)``); any other missing
    row is an error, never a silent zero."""

    found = evaluation.amplitudes.get(row)
    if found is not None:
        return found
    fixed = evaluation.fixed
    if any(fixed.get(index + 1, helicity) != helicity for index, helicity in row):
        return {}
    raise KeyError(f"helicity row {row} is missing from the point evaluation")


def _carrier_value(z, arithmetic: Arithmetic) -> Any:
    re, im = z
    if arithmetic == "double":
        return complex(re, im)
    if arithmetic == "exact":
        return rx.Complex(Fraction(re), Fraction(im))
    return rx.Complex(Decimal(re), Decimal(im))


def compare_numerators(
    native: Mapping[tuple[FamilyKey, Orders], NativeNumerator],
    reference: Mapping[tuple[FamilyKey, Orders], Any],
    energy: float = 1.0,
    scale: Literal["family", "point"] = "family",
) -> float:
    """Largest coefficient deviation of two numerator sets (0 if identical);
    raises if the family sets differ.

    Coefficients of degree r are weighted by ``energy``^r, so with a typical
    energy (sqrt(s)) a deviation is that of N(q) at |q| ~ energy, the
    units-independent metric of oracle O2. ``scale="family"`` divides by
    each family column's largest weighted coefficient; ``"point"`` by the
    largest of the column over every family of the point, the scale of the
    numerators that enter one sector (families that vanish exactly, e.g. by
    Furry's theorem, are rounding noise on any other scale).
    """

    if native.keys() != reference.keys():
        raise AssertionError(
            f"family sets differ: {sorted(native.keys() ^ reference.keys())[:3]}"
        )

    def weighted(column):
        return [
            abs(complex(v)) * float(energy) ** _degree(k) for k, v in enumerate(column)
        ]

    point: dict[int, float] = defaultdict(float)
    for numerator in reference.values():
        for o, column in enumerate(numerator.columns):
            point[o] = max(point[o], max(weighted(column), default=0.0))
    worst = 0.0
    for key, mine in native.items():
        theirs = reference[key]
        for o, (x, y) in enumerate(zip(mine.columns, theirs.columns, strict=True)):
            width = max(len(x), len(y))
            x = list(x) + [0] * (width - len(x))
            y = list(y) + [0] * (width - len(y))
            size = point[o] if scale == "point" else max(weighted(y), default=0.0)
            for k, (a, b) in enumerate(zip(x, y, strict=True)):
                difference = abs(complex(rx.exact(a) - rx.exact(b)))
                difference *= float(energy) ** _degree(k)
                if difference:
                    worst = max(worst, difference / size if size else float("inf"))
    return worst


def _degree(index: int) -> int:
    """Degree of the monomial stored at ``index`` (graded layout)."""

    degree = 0
    while size(degree) <= index:
        degree += 1
    return degree
