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
  inputs (``reference_executor.ExactKernel``); the model factors of its
  coefficients are numbered kernel constants, valued per batch;
* **seed classes and seeds**: per seed class and loop orders, and per
  observable column, the exact colour weights K[c, b] N_c^g of each Born
  component (b, Born order, N_c grade g), projected at the request's colour
  accuracy (``colour.SeedTable.seeds``); per seed key (class combination,
  loop orders) its exact weights x_c of the classes, so U0_c = sum_b K
  conj(A0) and U0 = sum_c x_c U0_c are formed natively;
* **families**: per family its trie frame reading, rank and loop orders, the
  leaves (seed key, leaf weight) and the schedule-B steps with their edges
  (child step or leaf, fused factor) and helicity row links;
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
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any, Literal

from pyamplicol.color.connections import ExactColorCoefficient

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


def born_rows(slice_: Any) -> tuple[rx.Row, ...]:
    """Born helicity rows in the order of ``assemble`` (the seed rows)."""

    labels = sorted(leg.label for leg in slice_.blocks.process.legs)
    helicities: dict[int, set[int]] = defaultdict(set)
    for block in slice_.blocks.blocks:
        if block.is_source:
            for row in block.rows:
                ((leg, value),) = row.helicities
                helicities[leg].add(value)
    return tuple(
        tuple((label - 1, h) for label, h in zip(labels, choice, strict=True))
        for choice in product(*(sorted(helicities[label]) for label in labels))
    )


# --------------------------------------------------------------------- exporter


@dataclass
class ExportedPlan:
    """A plan v4 program and the Python tables that fill its batch inputs."""

    program: dict[str, Any]
    rows: tuple[rx.Row, ...]
    born_components: tuple[tuple[int, Orders, int], ...]
    block_rows: tuple[tuple[Any, Hashable, int, rx.Row, int], ...]
    constants: tuple[Hashable, ...]
    masses: tuple[Any, ...]
    orders: tuple[Orders, ...]
    families: tuple[FamilyKey, ...]
    sectors: tuple[Orders, ...]
    # The tree program of native per-point inputs (``native_trees``).
    trees: Any = None

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
    kernels: dict[int, int] = field(default_factory=dict)
    kernel_records: list[dict] = field(default_factory=list)
    constants: dict[Hashable, int] = field(default_factory=dict)
    block_rows: dict[tuple, int] = field(default_factory=dict)
    factors: dict[tuple[int, int], int] = field(default_factory=dict)
    vertices: dict[Hashable, int] = field(default_factory=dict)
    seeds: dict[Hashable, int] = field(default_factory=dict)
    born_components: dict[tuple[int, Orders, int], int] = field(default_factory=dict)
    orders: dict[Orders, int] = field(default_factory=dict)
    masses: dict[Any, int] = field(default_factory=dict)

    # -- kernels ------------------------------------------------------------

    def _kernel(self, binding) -> int:
        found = self.kernels.get(binding.kernel_id)
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
                [self.constants.setdefault(key, len(self.constants)), list(monomial)]
                for key, monomial in terms
            ]
            for terms in kernel.outputs
        ]
        found = self.kernels[binding.kernel_id] = len(self.kernel_records)
        self.kernel_records.append({"inputs": inputs, "outputs": outputs})
        return found

    def _block_row(self, slot, part: rx.Row, dimension: int) -> int:
        key = (slot.block, slot.word, slot.chirality, part, dimension)
        return self.block_rows.setdefault(key, len(self.block_rows))

    # -- contents -----------------------------------------------------------

    def _binding(self, binding, index, split: bool) -> dict:
        """A kernel bound to a routing term's slots: resolved inputs and, per
        kernel output, its fused entry (and, for a split vertex binding, the
        flat block-component index nu it multiplies)."""

        kernel = self._kernel(binding)
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
        terms = []
        for term in self.kinematics.lowered(content):
            bindings = [self._binding(b, index, False) for b in term.fused]
            split = [self._binding(b, index, True) for b in term.split]
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
        self.content_records.append(
            {
                "d_in": shape.loop_in,
                "d_out": shape.loop_out,
                "degree": shape.degree,
                "rows": len(shape.rows),
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

    def seed_records(self) -> tuple[list[dict], list[dict]]:
        """The seed classes and seed keys of the plan: the projection of
        ``assemble.SectorSeeds`` (``SeedTable.seeds``) in two levels.

        A seed class record holds, per column, the exact projected weights
        K[c, b] of the Born components that the sector admits for one class
        c at one loop order; a seed key record holds its combination
        sum_c x_c of class records (``U0 = sum_c x_c U0_c``, formed by the
        executor). The admitted entries of each sector are indexed by
        (class, loop order) once, in ``SeedTable.admitted`` order.
        """

        slice_ = self.slice_
        table = slice_.table
        accuracy = slice_.request.accuracy
        degrees = slice_.degrees()["virtual"] if accuracy != "full" else {}
        born_grades = slice_._born_grades()
        sectors = []
        for sector, pairs in slice_.sectors:
            degree = degrees.get(sector)
            if accuracy != "full" and degree is None:
                degree = table.degree(pairs, born_grades)
            index: dict[tuple[int, Orders], list[tuple[int, Orders]]] = defaultdict(
                list
            )
            for c, order, b, born_order in table.admitted(pairs):
                index[(c, order)].append((b, born_order))
            sectors.append((degree, index))
        admitted = {key for _degree, index in sectors for key in index}
        classes: dict[tuple[int, Orders], int] = {}
        class_records: list[dict] = []

        def class_record(c: int, order: Orders) -> int:
            found = classes.get((c, order))
            if found is not None:
                return found
            columns = []
            for degree, index in sectors:
                weights: dict[int, ExactColorCoefficient] = {}
                for b, born_order in index.get((c, order), ()):
                    for grade in sorted(born_grades.get((b, born_order), ())):
                        weight = project(table.cells[c][b], degree, accuracy, grade)
                        if not weight:
                            continue
                        j = self.born_components.setdefault(
                            (b, born_order, grade), len(self.born_components)
                        )
                        weights[j] = weight + weights[j] if j in weights else weight
                columns.append(
                    [
                        [j, _fraction_text(w.real), _fraction_text(w.imag)]
                        for j, w in weights.items()
                        if w
                    ]
                )
            found = classes[(c, order)] = len(class_records)
            class_records.append({"columns": columns})
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
        (d_alpha,) = {source.d_alpha for _content, source in edges}
        seeded = {source.leaf >= 0 for _content, source in edges}
        if len(seeded) != 1:
            raise AssertionError("a step mixes leaf seeds and open-loop objects")
        (seeded,) = seeded
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
                    "leaf": source.leaf,
                    "child": source.step,
                    "factor": self.factor(index, shift),
                    "links": links,
                }
            )
        (d_out,) = d_out
        if accumulator is not None and set(targets) - {()}:
            raise AssertionError("a closure with undressed legs")
        steps.append(
            {
                "seeded": seeded,
                "trace": -1 if accumulator is None else accumulator,
                "rows": len(targets),
                "d_alpha": d_alpha,
                "d_out": d_out,
                "degree": degree,
                "rank_in": storage,
                "edges": records,
            }
        )
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


def export_plan(
    slice_: Any,
    *,
    forward: bool = True,
    merged: MergedGraph | None = None,
    trees: bool = False,
    born_squared: bool = False,
) -> ExportedPlan:
    """The plan v4 program of a generated slice (structure only, no numerics).

    ``merged`` adds schedule M's planar state graph (``merged.merged_graph``)
    for its certified families; schedule B keeps every family. ``trees``
    exports the tree program of native per-point inputs
    (``native_trees.export_trees``; its Born kernels join the plan's), with
    the Born colour metric when ``born_squared``.
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
    rows = born_rows(slice_)
    exporter = _Exporter(slice_, kinematics, rows)
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
    merged_program = (
        exporter.merged(merged, {key: i for i, key in enumerate(keys)}, accumulator_of)
        if merged is not None
        else None
    )
    seed_classes, seeds = exporter.seed_records()
    native_trees = None
    if trees:
        from .native_trees import export_trees

        native_trees = export_trees(slice_, exporter, rows, born_squared=born_squared)
    legs = len(slice_.blocks.process.legs)
    program = {
        "schema": PLAN_SCHEMA,
        "legs": legs,
        "rows": len(rows),
        "columns": len(slice_.sectors),
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
    return ExportedPlan(
        program=program,
        rows=rows,
        born_components=tuple(exporter.born_components),
        block_rows=tuple(exporter.block_rows),
        constants=tuple(exporter.constants),
        masses=tuple(exporter.masses),
        orders=orders,
        families=tuple(keys),
        sectors=tuple(sector for sector, _pairs in slice_.sectors),
        trees=native_trees,
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
    hands over only parameter-dependent values. Its results hold the Born
    (with ``born_squared``, full colour) and the ``bare`` component; the
    counterterms and external fields join with the native counterterm
    program (WP3.1 I5, after WP2.2), so ``virtual`` is absent until then.
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
        inputs: Literal["python", "native"] = "python",
        born_squared: bool | None = None,
    ):
        if inputs not in ("python", "native"):
            raise ValueError(f"unknown input path {inputs!r}")
        if born_squared is None:
            born_squared = slice_.request.accuracy == "full"
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
        )
        if inputs == "native" and self.exported.trees is None:
            raise ValueError("native inputs need a plan exported with trees=True")
        self.lanes = lanes
        self.threads = threads
        self.forms = forms
        self._plan_json = self.exported.json()
        self._native: dict[tuple[str, int | None, int], Any] = {}
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
                True,
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
            """Kernel constants and squared masses of a point's parameters."""

            constants = evaluation.constants
            if arithmetic == "exact":
                constants = rx.KernelConstants(
                    constants.parameters,
                    rx.Carrier.exact(evaluation.numbers.precision.working_digits),
                )
            masses = _mass_values(
                slice_.compiled, evaluation.parameters, exported.masses
            )
            return (
                [value(constants.constant(key)) for key in exported.constants],
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
            from .native_trees import batch_inputs

            for evaluation in evaluations:
                with evaluation.numbers.scope():
                    common = shared(evaluation)
                if batch.setdefault("shared", common) != common:
                    raise ValueError(
                        "the points of one batch need equal kernel constants and masses"
                    )
            batch["constants"], batch["masses"] = batch.pop("shared")
            batch["trees"] = batch_inputs(slice_, exported.trees, evaluations, value)
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
            batch["born"].append(
                [
                    value(evaluation.amplitudes[row].get((b, order), {}).get(grade, 0))
                    for row in exported.rows
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
        poles: bool = True,
        stability: Any = None,
        schedule: Literal["B", "M"] = "B",
    ) -> list[Any]:
        """The complete virtual of every point (``LoopSlice.evaluate``), with
        dressing and reduction in the native executor (``schedule`` B, or M
        for the plan's schedule-M families).

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
        per point ``(block_rows, born, amplitudes, born_squared)``."""

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

        from .assemble import Laurent, Precision, _Numbers

        if self.inputs == "native":
            precision = precision or Precision.double()
            evaluations = self._native_points(points, scales, parameters, precision)
            for evaluation in evaluations:
                evaluation.poles = poles
        else:
            precision, evaluations = self._points(
                points, scales, parameters, precision, poles
            )
        arithmetic, digits = self._arithmetic(precision, False)
        native = self.native(arithmetic, digits, len(evaluations))
        output = native.evaluate(
            self._inputs(evaluations, arithmetic),
            poles=poles,
            check=retest is not None,
            rescale=retest == "rescaled",
            schedule=schedule,
        )
        exported = self.exported
        self.last_report = output["report"]
        self.last_bare = output["bare"]
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
            if failures:
                failed[index] = [str(failure) for failure in failures]
                continue
            numbers = evaluation.numbers

            def laurent(values, numbers=numbers):
                if numbers.double:
                    return Laurent(
                        *[complex(float(re), float(im)) for re, im in values]
                    )
                return Laurent(
                    *[numbers.pair(Decimal(re), Decimal(im)) for re, im in values]
                )

            def label(lane):
                family, order, column = lane
                return (
                    exported.families[family],
                    exported.orders[order],
                    exported.sectors[column],
                )

            with _Numbers(precision).scope():
                if output.get("born") is not None:
                    evaluation.born = {
                        order: numbers.real(Decimal(str(re)))
                        for order, (re, _im) in zip(
                            exported.trees.squared_orders,
                            output["born"][index],
                            strict=True,
                        )
                    }
                evaluation.families = {
                    label(lane): laurent(values) for lane, values in lanes
                }
                evaluation.retest_families = (
                    None
                    if checks is None
                    else {label(lane): laurent(values) for lane, values in checks}
                )
                # The native seeds contract the exported exact weights K[c, b]
                # with the Born components directly (``seed_records``).
                evaluation.colour_contraction = "direct"
                evaluation._assemble()
        return precision, evaluations, failed


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
