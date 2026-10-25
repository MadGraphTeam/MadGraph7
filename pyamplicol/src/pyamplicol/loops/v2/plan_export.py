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

import json
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any, Literal

from pyamplicol.color.connections import ExactColorCoefficient
from pyamplicol.config import PoleSelection

from . import reference_executor as rx
from .colour import project
from .families import FamilyKey, Orders
from .merged import MergedGraph, merged_graph
from .poly_affine import size
from .schedules import forward_graph

PLAN_SCHEMA = 4
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
    # seed records that replace the plan's for that set: ``seed_classes``,
    # ``seeds``, ``columns`` (count), ``column_rows`` (or None) and the
    # Born components they read (``born_components``, keys).
    column_seeds: Mapping[str, Mapping[str, Any]] | None = None

    def json(self) -> str:
        return json.dumps(self.program, separators=(",", ":"))


@dataclass
class _Object:
    """An open-loop object of the exported walk: a leaf seed or a step output."""

    rows: tuple[rx.Row, ...]
    d_alpha: int
    d_beta: int
    storage: int
    leaf: int = -1
    step: int = -1


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

    def seed_records(self, columns: Sequence[Any]) -> tuple[list[dict], list[dict]]:
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
        """

        slice_ = self.slice_
        accuracy = slice_.request.accuracy
        born_grades = slice_._born_grades()
        admitted: set[tuple[int, Orders]] = set()
        # Per column, once: its table, degree and, per (class, loop order),
        # its admitted Born entries (table order).
        indexed = []
        for column in columns:
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
        return class_records, records

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
                self._step(steps, cuts[species], 0, shift, accumulator)
        if values:
            raise AssertionError("a trie object without a consumer")
        return {
            "sequence": [[mask, self._mass(mass)] for mask, mass in sequence],
            "rank": trie.rank,
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


def sector_columns(slice_: Any) -> tuple[SeedColumn, ...]:
    """One column per squared-order sector (the inclusive virtual)."""

    degrees = slice_.degrees()["virtual"] if slice_.request.accuracy != "full" else {}
    return tuple(
        SeedColumn(slice_.table, pairs, degrees.get(sector))
        for sector, pairs in slice_.sectors
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
    seed_classes, seeds = exporter.seed_records(columns)
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

        for name, set_columns in column_sets.items():
            set_columns = tuple(set_columns)
            if len(set_columns) == len(columns) and all(
                same(a, b) for a, b in zip(set_columns, columns, strict=True)
            ):
                # The plan's own columns (the Born metric of a process
                # without declared histories): the plan serves the set.
                column_seeds[name] = {"same": True, "born_components": born_components}
                continue
            set_classes, set_seeds = exporter.seed_records(set_columns)
            column_seeds[name] = {
                "seed_classes": set_classes,
                "seeds": set_seeds,
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
    ):
        if inputs not in ("python", "native"):
            raise ValueError(f"unknown input path {inputs!r}")
        if plan_json is not None and exported is None:
            raise ValueError("plan_json is the serialised exported plan")
        if isinstance(scaleless, tuple) and slice_.scaleless is None:
            raise ValueError("a scaleless companion plan needs the slice's companion")
        if slice_.request.colour_contraction == "fft":
            # The native seeds contract the exported exact weights directly
            # (``seed_records``); an explicit "fft" is refused, not ignored.
            raise ValueError(
                "the native executor contracts seeds directly; request colour "
                "contraction 'auto' or 'direct'"
            )
        self.slice = slice_
        self.inputs = inputs
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

    def _points(self, points, scales, parameters, precision, poles):
        from .assemble import Precision, _fraction, _Numbers

        precision = precision or Precision.double()
        evaluations = []
        for point, scale in zip(points, scales, strict=True):
            exact_point = tuple(tuple(_fraction(x) for x in p) for p in point)
            with _Numbers(precision).scope():
                evaluations.append(
                    self.slice._point(
                        exact_point,
                        _fraction(scale),
                        dict(parameters or {}),
                        precision,
                        poles,
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
        """Batch inputs: exact momenta, scales and masses; carrier values of
        block rows, Born components and kernel constants.

        Every carrier value is formed in its point's Decimal context: block
        rows carry the tree engine's Wick signs and kernel constants round
        their irrational factors, both in the carrier's arithmetic.
        """

        from .assemble import _BlockRows, _mass_values

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
            """Kernel constants (with their contents' mass flips) and squared
            masses of a point's parameters."""

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

            masses = _mass_values(
                slice_.compiled, evaluation.parameters, exported.masses
            )
            return (
                [value(constant(key, flips)) for key, flips in exported.constants],
                [_fraction_text(masses[m]) for m in exported.masses],
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
                raise ValueError(
                    "the points of one batch need equal kernel constants and masses"
                )
            batch["block_rows"].append(rows)
            amplitudes = [_born_row(evaluation, row) for row in exported.rows]
            batch["born"].append(
                [
                    value(row.get((b, order), {}).get(grade, 0))
                    for row in amplitudes
                    for b, order, grade in exported.born_components
                ]
            )
        batch["constants"], batch["masses"] = batch.pop("shared")
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
    ) -> list[Any]:
        """The complete virtual of every point (``LoopSlice.evaluate``), with
        dressing and reduction in the native executor (``schedule`` B, or M
        for the plan's schedule-M families). ``poles`` is the pole selection
        (``PoleSelection`` or its value: ``"all"``, ``"none"``, ``"uv"``,
        ``"ir"``; ``poles``).

        Rotation checks re-evaluate the rotated points natively; the
        reduction check is the native re-test pass. A failed point fails
        alone: ``FailedPoints`` carries the complete results of the others.
        """

        from .assemble import _fraction
        from .stability import ROTATION, StabilityChecks, StabilityEstimate

        stability = stability or StabilityChecks()
        precision, evaluations, failures = self._evaluated(
            points, scales, parameters, precision, poles, stability.frame, schedule
        )
        rotated = []
        for k, rotation in enumerate(stability.rotations):
            moved = [
                rotation.apply(tuple(tuple(_fraction(x) for x in p) for p in point))
                for point in points
            ]
            _, runs, failed = self._evaluated(
                moved, scales, parameters, precision, poles, None, schedule
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
        self, points, scales, parameters, precision, poles, retest, schedule="B"
    ):
        """Evaluate a batch natively; returns the precision, the per-point
        evaluations (failed points unassembled) and the failures per point
        index."""

        from .assemble import Precision, _Numbers

        if self.inputs == "native":
            from .poles import has_poles, pole_mode

            precision = precision or Precision.double()
            evaluations = self._native_points(points, scales, parameters, precision)
            for evaluation in evaluations:
                evaluation.mode = pole_mode(poles)
                evaluation.poles = has_poles(evaluation.mode)
        else:
            precision, evaluations = self._points(
                points, scales, parameters, precision, poles
            )
        failed = self.reduce(evaluations, precision, poles, retest, schedule)
        for index, evaluation in enumerate(evaluations):
            if index in failed:
                continue
            with _Numbers(precision).scope():
                # The native seeds contract the exported exact weights K[c, b]
                # with the Born components directly (``seed_records``).
                evaluation.colour_contraction = "direct"
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
            self._inputs(evaluations, arithmetic), schedule=schedule, **options
        )
        exported = self.exported
        self.last_report = output["report"]
        self.last_bare = output["bare"]
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
                if output.get("born") is not None:
                    evaluation.born = {
                        order: numbers.real(Decimal(str(re)))
                        for order, (re, _im) in zip(
                            exported.trees.squared_orders,
                            output["born"][index],
                            strict=True,
                        )
                    }
                evaluation.set_lanes(
                    read(lanes), read(checks), scaleless, scaleless_check
                )
        return failed


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
