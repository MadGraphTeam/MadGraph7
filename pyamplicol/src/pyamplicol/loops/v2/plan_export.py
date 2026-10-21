# SPDX-License-Identifier: 0BSD
"""Plan v4 export and the native v2 executor bridge (design §3.8, §5.1, WP2.1).

``export_plan`` writes the schedule program of a generated ``LoopSlice`` in the
form the native executor runs (``rusticol_core::loop_v2::plan``). It walks the
same objects as the Python reference executor (``reference_executor``), so the
two run identical programs:

* **contents**: per content its loop-line dimensions, structural q-degree and
  fused entries, and per routing term the kernel bindings that build its fused
  factor X(J) from block-current rows, block momenta and the loop shift;
* **kernels**: every catalogue kernel as exact polynomial terms over its
  inputs (``reference_executor.ExactKernel``); the model factors of its
  coefficients are numbered kernel constants, valued per batch;
* **seeds**: per seed key (class, loop orders) and observable column the exact
  colour weights K[c, b] of each Born component (b, Born order), projected at
  the request's colour accuracy, so U0 = sum K conj(A0) is formed natively;
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
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any, Literal

from . import reference_executor as rx
from .colour import project
from .families import FamilyKey, Orders
from .poly_affine import size
from .schedules import forward_graph

PLAN_SCHEMA = 4
# Lanes of the native executor's SIMD batches (gate G0, design §5.4).
DEFAULT_LANES = 16

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
    born_components: tuple[tuple[int, Orders], ...]
    block_rows: tuple[tuple[Any, Hashable, int, rx.Row, int], ...]
    constants: tuple[Hashable, ...]
    masses: tuple[Any, ...]
    orders: tuple[Orders, ...]
    families: tuple[FamilyKey, ...]
    sectors: tuple[Orders, ...]

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
    seeds: dict[Hashable, int] = field(default_factory=dict)
    born_components: dict[tuple[int, Orders], int] = field(default_factory=dict)
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

    def content(self, content) -> tuple[int, rx.ContentShape]:
        shape = self.kinematics.shape(content)
        found = self.contents.get(content)
        if found is not None:
            return found, shape
        index = {entry: i for i, entry in enumerate(shape.fused)}
        row_index = {row: i for i, row in enumerate(shape.rows)}
        terms = []
        for term in self.kinematics.lowered(content):
            bindings = []
            for binding in term.fused:
                kernel = self._kernel(binding)
                current_slots, momentum_slots = [], []
                for slot, dimension in enumerate(binding.block_dimensions):
                    current_slots += [(slot, k) for k in range(dimension)]
                    momentum_slots += [(slot, mu) for mu in range(4)]
                inputs = []
                for kind, component, _ in self.kernel_records[kernel]["inputs"]:
                    if kind == "current":
                        inputs.append(["current", *current_slots[component]])
                    elif kind == "momentum":
                        inputs.append(["momentum", *momentum_slots[component]])
                    else:
                        inputs.append([kind, component, 0])
                bindings.append(
                    {
                        "kernel": kernel,
                        "inputs": inputs,
                        "entries": [
                            index[(e.loop_in, e.out, e.exponents)]
                            for e in binding.entries
                        ],
                        "dimensions": list(binding.block_dimensions),
                    }
                )
            dimensions = {tuple(b["dimensions"]) for b in bindings}
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
            for binding in bindings:
                del binding["dimensions"]
            terms.append(
                {
                    "slots": [slot.block.mask for slot in term.slots],
                    "bindings": bindings,
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
        c = slice_.class_of.get(reading)
        if c is None:
            raise KeyError("reading without a seed class")
        key = (c, skeleton.orders)
        found = self.seeds.get(key)
        if found is None:
            found = self.seeds[key] = len(self.seeds)
        return found

    def seed_records(self) -> list[dict]:
        slice_ = self.slice_
        table = slice_.table
        accuracy = slice_.request.accuracy
        degrees = slice_.degrees()["virtual"] if accuracy != "full" else {}
        records = []
        for c, order in self.seeds:
            columns = []
            admitted = False
            for sector, pairs in slice_.sectors:
                degree = degrees.get(sector)
                if accuracy != "full" and degree is None:
                    degree = table.degree(pairs)
                terms = []
                for cc, oo, b, born_order in table.admitted(pairs):
                    if (cc, oo) != (c, order):
                        continue
                    admitted = True
                    weight = project(table.cells[c][b], degree, accuracy)
                    if not weight:
                        continue
                    j = self.born_components.setdefault(
                        (b, born_order), len(self.born_components)
                    )
                    terms.append(
                        [j, _fraction_text(weight.real), _fraction_text(weight.imag)]
                    )
                columns.append(terms)
            if not admitted:
                raise KeyError(f"no sector admits class {c} at {order}")
            records.append({"columns": columns})
        return records

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


def export_plan(slice_: Any, *, forward: bool = True) -> ExportedPlan:
    """The plan v4 program of a generated slice (structure only, no numerics)."""

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
    seeds = exporter.seed_records()
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
        "seeds": seeds,
        "families": families,
        "forward": forward_program,
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


class NativeExecutor:
    """The native v2 executor of a generated slice (one plan, many batches).

    ``arithmetic`` selects the native carrier: ``double`` (binary64, the
    NinjaRS fast engine), ``double-double``, ``arbitrary`` (MPFR at the
    precision's working digits, the generic engine) or ``exact`` (rational
    numerators only, the algebraic oracle of the Python exact run).
    """

    def __init__(
        self,
        slice_: Any,
        *,
        lanes: int = DEFAULT_LANES,
        threads: int | None = None,
        forward: bool = True,
        exported: ExportedPlan | None = None,
    ):
        self.slice = slice_
        self.exported = exported or export_plan(slice_, forward=forward)
        self.lanes = lanes
        self.threads = threads
        self._plan_json = self.exported.json()
        self._native: dict[tuple[str, int | None], Any] = {}

    def native(self, arithmetic: Arithmetic, digits: int | None = None) -> Any:
        """The native executor of one carrier (``digits`` for ``arbitrary``)."""

        key = (arithmetic, digits)
        found = self._native.get(key)
        if found is None:
            found = self._native[key] = self.slice.native.V2Executor(
                self._plan_json,
                arithmetic,
                digits,
                lanes=self.lanes,
                threads=self.threads,
            )
        return found

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
                    value(evaluation.amplitudes[row].get(component, 0))
                    for row in exported.rows
                    for component in exported.born_components
                ]
            )
        batch["constants"], batch["masses"] = batch.pop("shared")
        return batch

    @staticmethod
    def _arithmetic(precision, exact: bool) -> tuple[Arithmetic, int | None]:
        if exact:
            return "exact", None
        if precision.arithmetic == "double":
            return "double", None
        return "arbitrary", precision.working_digits

    # -- runs -------------------------------------------------------------------

    def family_numerators(
        self,
        points: Sequence[Sequence[Sequence[Any]]],
        scales: Sequence[Any],
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Any = None,
        schedule: Literal["B", "F"] = "B",
        exact: bool = False,
    ) -> list[dict[tuple[FamilyKey, Orders], NativeNumerator]]:
        """Native family numerators per point (one column per sector), in the
        families' trie frames: the comparison with ``LoopSlice.family_numerators``.

        ``exact`` runs the native rational carrier on the carrier's tree rows
        and Born amplitudes read as exact rationals and the exact kernel
        constants (``reference_executor.KernelConstants`` stand-ins).
        """

        precision, evaluations = self._points(
            points, scales, parameters, precision, True
        )
        arithmetic, digits = self._arithmetic(precision, exact)
        native = self.native(arithmetic, digits)
        raw = native.numerators(self._inputs(evaluations, arithmetic), schedule)
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
    ) -> list[Any]:
        """The complete virtual of every point (``LoopSlice.evaluate``), with
        dressing and reduction in the native executor.

        Rotation checks re-evaluate the rotated points natively; the
        reduction-direction check is the native re-test pass.
        """

        from .stability import REDUCTION, ROTATION, StabilityChecks, StabilityEstimate

        stability = stability or StabilityChecks()
        precision, evaluations = self._evaluated(
            points, scales, parameters, precision, poles, stability.reduction_direction
        )
        if not stability.enabled:
            return [evaluation.result for evaluation in evaluations]
        rotated = []
        for rotation in stability.rotations:
            from .assemble import _fraction

            moved = [
                rotation.apply(tuple(tuple(_fraction(x) for x in p) for p in point))
                for point in points
            ]
            rotated.append(
                self._evaluated(moved, scales, parameters, precision, poles, False)[1]
            )
        out = []
        for index, reference in enumerate(evaluations):
            budgets = {}
            for k, runs in enumerate(rotated):
                budgets[ROTATION.format(k)] = reference.rotation_budget(runs[index])
            if stability.reduction_direction:
                budgets[REDUCTION] = reference.retest_budget
            estimate = StabilityEstimate.from_checks(
                reference.result.components["virtual"],
                budgets,
                scale=reference.absolute,
                roundoff=precision.unit_roundoff,
                families=reference.family_deviations,
            )
            out.append(replace(reference.result, stability=estimate))
        return out

    def _evaluated(self, points, scales, parameters, precision, poles, retest):
        from .assemble import Laurent, _Numbers

        precision, evaluations = self._points(
            points, scales, parameters, precision, poles
        )
        arithmetic, digits = self._arithmetic(precision, False)
        native = self.native(arithmetic, digits)
        output = native.evaluate(
            self._inputs(evaluations, arithmetic), poles=poles, check=retest
        )
        exported = self.exported
        self.last_report = output["report"]
        self.last_bare = output["bare"]
        for evaluation, lanes, checks, failures in zip(
            evaluations,
            output["families"],
            output["check"] or [None] * len(evaluations),
            output["failures"],
            strict=True,
        ):
            if failures:
                raise ArithmeticError(f"native reduction lanes failed: {failures}")
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
                evaluation.families = {
                    label(lane): laurent(values) for lane, values in lanes
                }
                evaluation.retest_families = (
                    None
                    if checks is None
                    else {label(lane): laurent(values) for lane, values in checks}
                )
                evaluation._assemble()
        return precision, evaluations


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
) -> float:
    """Largest coefficient deviation of two numerator sets, relative to each
    family column's largest coefficient (0 for identical sets); raises if the
    family sets differ.

    Coefficients of degree r are weighted by ``energy``^r, so with a typical
    energy (sqrt(s)) the deviation is that of N(q) at |q| ~ energy, the
    units-independent metric of oracle O2.
    """

    if native.keys() != reference.keys():
        raise AssertionError(
            f"family sets differ: {sorted(native.keys() ^ reference.keys())[:3]}"
        )
    worst = 0.0
    for key, mine in native.items():
        theirs = reference[key]
        for x, y in zip(mine.columns, theirs.columns, strict=True):
            width = max(len(x), len(y))
            x = list(x) + [0] * (width - len(x))
            y = list(y) + [0] * (width - len(y))
            weights = [float(energy) ** _degree(k) for k in range(width)]
            scale = max(
                (abs(complex(v)) * w for v, w in zip(y, weights, strict=True)),
                default=0.0,
            )
            for a, b, w in zip(x, y, weights, strict=True):
                difference = abs(complex(rx.exact(a) - rx.exact(b))) * w
                if difference:
                    worst = max(worst, difference / scale if scale else float("inf"))
    return worst


def _degree(index: int) -> int:
    """Degree of the monomial stored at ``index`` (graded layout)."""

    degree = 0
    while size(degree) <= index:
        degree += 1
    return degree
