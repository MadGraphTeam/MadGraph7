# SPDX-License-Identifier: 0BSD
"""Native per-point inputs of the v2 executor (WP3.1 I3/I4).

The slice forms every point's block rows and Born amplitudes in Python
(``assemble._PointEvaluation``: ``TreeRows`` replays the tree program in
Decimal, the Born root closure runs per helicity row). This module exports
the same program for the native executor (``rusticol-core``
``loop_v2::trees``), which then forms them without Python per point:

* **trees**: the slice's own tree stage program (``assemble.TreeRows``'
  runtime schema), each stage compiled in chunks of whole currents of at
  most ``CHUNK_OUTPUTS`` outputs and saved as exact Symbolica evaluators
  (any carrier), so only one chunk's expressions are alive at a time; with
  the runtime schema's sources, momentum slots and model-parameter layout.
  A plan block row is the Wick-signed state slice of its tree current;
* **Born**: every root-closure term (``assemble._born_terms``: an ordinary
  vertex holding leg 1 closed on 2-3 blocks) as a plan kernel on its blocks'
  rows and momenta, with its sign, its coupling per N_c grade and the exact
  colour weights of its literal tensors, accumulated per Born helicity row
  into the components A0[b, Born order, grade]; with ``born_squared`` the
  full-colour metric of the Born basis gives |A0|^2 per squared order.

Per batch only parameter-dependent values cross (``batch_inputs``): the model
parameters, the masses of massive external particles and the couplings. The
plan's kernel constants carry the Born kernels' model factors.

The exact carrier has no native trees (wavefunctions are irrational), and
leading/next-to-leading colour has no native Born metric yet (full colour
first, WP3.1 plan §4.3); the slice's Python inputs remain their path.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pyamplicol.color.contraction_types import NC

from .families import Orders, add_orders

TREE_SCHEMA = 1

# Outputs of one saved stage evaluator. The currents of a stage depend only on
# earlier stages, so a stage splits into chunks of whole currents, compiled
# and saved one at a time: the peak memory is one chunk's expressions, not
# the program's (g g > g g g g g: 475,440 outputs in four stages).
CHUNK_OUTPUTS = 1 << 15


@dataclass(frozen=True)
class NativeTrees:
    """A tree program for ``V2Executor(trees=...)`` and its batch layout."""

    program: dict[str, Any]
    stages: tuple[bytes, ...]
    couplings: tuple[tuple[str, int], ...]
    born_orders: tuple[Orders, ...]
    squared_orders: tuple[Orders, ...]
    masses: tuple[tuple[int, str], ...]
    # Every Born component (basis element, Born order, N_c grade), in the
    # order of the executor's ``amplitudes`` per Born row.
    components: tuple[tuple[int, Orders, int], ...] = ()

    def json(self) -> str:
        return json.dumps(self.program, separators=(",", ":"))

    def executor_argument(self) -> tuple[str, list[bytes]]:
        return self.json(), list(self.stages)


def _fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator if value.denominator == 1 else value)


def _saved_stage(stage: Any) -> bytes:
    from symbolica import Expression, FunctionDefinition

    return bytes(
        Expression.evaluator_multiple(
            list(stage.output_expressions),
            list(stage.parameter_symbols),
            functions=[
                FunctionDefinition(function, arguments, body, inlining="always")
                for function, arguments, body in stage.symbolica_functions
            ],
            jit_compile=False,
        ).save()
    )


def export_trees(
    slice_: Any,
    exporter: Any,
    rows: tuple[tuple[tuple[int, int], ...], ...],
    *,
    born_squared: bool,
) -> NativeTrees:
    """The tree program of ``slice_`` for the plan ``exporter`` is lowering.

    Call after the exporter's seed records (the plan's Born components are
    numbered there); the Born kernels and their constants join the plan.
    """

    from .assemble import LAURENT_POWERS, _BlockRows, _single

    if slice_.request.accuracy != "full" and born_squared:
        raise ValueError("the native Born metric is full colour (WP3.1 plan §4.3)")
    trees = slice_.tree_rows
    schema = trees.schema
    process = slice_.blocks.process
    labels = sorted(leg.label for leg in process.legs)
    if labels != list(range(1, len(labels) + 1)):
        raise ValueError("native trees need legs labelled 1..n")
    initial = [
        any(leg.label == label and leg.is_initial for leg in process.legs)
        for label in labels
    ]
    value_count = trees._value_count
    model_count = len(trees._records)
    state_size = max(trees._size, trees._offset + model_count)
    slots: dict[tuple[int, str], tuple[int, int]] = {}
    for record in schema["value_storage"]["value_slots"]:
        key = (int(record["current_id"]), str(record["variant"]))
        slots.setdefault(
            key, (int(record["component_start"]), int(record["dimension"]))
        )

    def slot(block: Any, entry: Any) -> tuple[int, int]:
        """(state component start, dimension) of a block row."""

        variant = "source" if block.is_source else "propagated"
        found = slots.get((entry.current_id, variant))
        if found is None:
            raise ValueError(f"tree current {entry.current_id} has no {variant} row")
        return found

    momentum_slots = []
    for record in schema["momentum_slots"]:
        mask = 0
        for label in record["external_labels"]:
            mask |= 1 << (int(label) - 1)
        momentum_slots.append([value_count + int(record["component_start"]), mask])
    stages, saved = _stream_stages(trees)
    index = _BlockRows(slice_, None)._index
    block_rows = []
    for current, word, chirality, row, dimension in exporter.block_rows:
        tree, entry = index[(current.mask, current.field, current.orders, word)][
            chirality
        ][row]
        start, size = slot(tree, entry)
        if size != dimension:
            raise ValueError("a block row's dimension differs from its tree current")
        block_rows.append([start, dimension, tree.wick_sign])
    # ---------------------------------------------------------------- Born
    _plan, layout = slice_._scalar_plan(LAURENT_POWERS)
    couplings: dict[tuple[str, int], int] = {}
    components: dict[tuple[int, Orders, int], int] = {}
    basis_index = {tensor: b for b, tensor in enumerate(slice_.basis.tensors)}
    legs = slice_.basis.legs
    block_ids: dict[int, int] = {}
    block_starts: list[list[int | None]] = []
    block_dimensions: list[int] = []

    def block_id(block: Any) -> int:
        found = block_ids.get(id(block))
        if found is not None:
            return found
        by_helicity = {row.helicities: row for row in block.rows}
        starts: list[int | None] = []
        dimensions = set()
        for born_row in rows:
            helicity = {leg + 1: h for leg, h in born_row}
            want = tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
            entry = by_helicity.get(want)
            if entry is None:
                starts.append(None)
                continue
            start, dimension = slot(block, entry)
            starts.append(start)
            dimensions.add(dimension)
        if len(dimensions) != 1:
            raise ValueError("a closure block's rows differ in dimension")
        found = block_ids[id(block)] = len(block_starts)
        block_starts.append(starts)
        block_dimensions.append(dimensions.pop())
        return found

    terms = []
    born_orders = tuple(sorted(slice_.born_terms))
    for order, order_terms in sorted(slice_.born_terms.items()):
        for term in order_terms:
            kernel = exporter._kernel(term.binding)
            ids = [block_id(block) for block in term.blocks]
            dimensions = [block_dimensions[ids[i]] for i in term.kernel_legs]
            current_slots = [
                (term.kernel_legs[leg], k)
                for leg, dimension in enumerate(dimensions)
                for k in range(dimension)
            ]
            inputs = []
            for kind, component, _ in exporter.kernel_records[kernel]["inputs"]:
                if kind == "current":
                    inputs.append(["current", *current_slots[component]])
                elif kind == "momentum":
                    inputs.append(["momentum", component // 4, component % 4])
                else:
                    raise ValueError(f"a Born closure kernel reads a {kind} input")
            contributions = []
            grades = layout.get((term.entry.coupling.name, 0), ())
            power = term.colour.normalization_power
            for grade, _column in grades:
                coupling = couplings.setdefault(
                    (term.entry.coupling.name, grade), len(couplings)
                )
                for item in term.colour.terms:
                    b = basis_index[_single(item.tensor, legs)]
                    component = components.setdefault(
                        (b, order, item.nc_power + grade), len(components)
                    )
                    contributions.append(
                        [
                            coupling,
                            component,
                            _fraction_text(item.coefficient.real),
                            _fraction_text(item.coefficient.imag),
                            power,
                        ]
                    )
            terms.append(
                {
                    "kernel": kernel,
                    "inputs": inputs,
                    "blocks": ids,
                    "masks": [block.key.external_mask for block in term.blocks],
                    "sign": term.sign,
                    "contributions": contributions,
                }
            )
    plan_components = [components.get(key) for key in exporter.born_components]
    orders_index = {order: o for o, order in enumerate(born_orders)}
    amplitudes: list[list[list[Any]]] = [
        [] for _ in range(len(slice_.basis.tensors) * len(born_orders))
    ]
    for (b, order, grade), j in components.items():
        amplitudes[b * len(born_orders) + orders_index[order]].append(
            [j, _fraction_text(Fraction(NC) ** grade)]
        )
    squared: dict[Orders, list[list[int]]] = defaultdict(list)
    for left in born_orders:
        for right in born_orders:
            squared[add_orders(left, right)].append(
                [orders_index[left], orders_index[right]]
            )
    squared_orders = tuple(sorted(squared))
    gram = None
    if born_squared:
        gram = []
        for b, row in enumerate(slice_.basis.gram):
            for b2, cell in enumerate(row):
                value = sum(
                    (
                        value.scaled(Fraction(NC) ** power)
                        for power, value in cell.items()
                    ),
                    start=_zero_colour(),
                )
                if value:
                    gram.append(
                        [
                            b,
                            b2,
                            _fraction_text(value.real),
                            _fraction_text(value.imag),
                        ]
                    )
    masses = _external_masses(slice_)
    program = {
        "schema": TREE_SCHEMA,
        "legs": len(labels),
        "initial": initial,
        "state_size": state_size,
        "model_offset": trees._offset,
        "model_count": model_count,
        "momentum_slots": momentum_slots,
        "source_fill": schema["source_fill"],
        "stages": stages,
        "block_rows": block_rows,
        "born": {
            "rows": len(rows),
            "components": len(components),
            "plan_components": plan_components,
            "blocks": block_starts,
            "couplings": len(couplings),
            "terms": terms,
            "amplitudes": amplitudes,
            "orders": len(born_orders),
            "squared": [squared[order] for order in squared_orders],
            "gram": gram,
            "average": str(slice_.normalisation.average),
        },
    }
    return NativeTrees(
        program,
        saved,
        tuple(couplings),
        born_orders,
        squared_orders,
        masses,
        tuple(components),
    )


def _stream_stages(trees: Any) -> tuple[list[dict[str, Any]], tuple[bytes, ...]]:
    """The stage records and saved evaluators of ``trees`` (a ``TreeRows``),
    compiled chunk by chunk (``CHUNK_OUTPUTS``) and released after saving."""

    from pyamplicol.generation.contracts import StageCompilationInput
    from pyamplicol.generation.stage_planning import (
        build_generic_stage_compiler_blueprint,
    )

    from .assemble import _exact_u1_subtraction_weights

    schema = dict(trees.schema)
    count = len(schema["stages"])
    schema["stages"] = _chunked_stages(trees.dag, trees.schema, CHUNK_OUTPUTS)
    records: list[dict[str, Any]] = []
    saved: list[bytes] = []
    blockers: list[str] = []

    def consume(stage: Any, index: int, total: int) -> None:
        if index == total:
            return  # the amplitude stage: the Born is the root closure
        if stage.blockers:
            blockers.extend(stage.blockers)
            return
        saved.append(_saved_stage(stage))
        records.append(
            {
                "parameters": int(stage.parameter_count),
                "inputs": [
                    [int(c.parameter_index), int(c.global_component)]
                    for c in stage.input_components
                ],
                "outputs": [
                    [
                        int(s.output_start),
                        int(s.output_stop - s.output_start),
                        int(s.component_start),
                    ]
                    for s in stage.output_slots
                ],
            }
        )

    with _exact_u1_subtraction_weights():
        build_generic_stage_compiler_blueprint(
            StageCompilationInput(trees.dag, trees.model, trees.runtime_schema),
            runtime_schema=schema,
            stage_consumer=consume,
            release_consumed_expressions=True,
        )
    if blockers:
        raise ValueError(f"tree blocks are not expression ready: {blockers}")
    if count and not records:
        raise ValueError("the tree program has no stage")
    return records, tuple(saved)


def _chunked_stages(dag: Any, schema: Any, limit: int) -> list[dict[str, Any]]:
    """The runtime schema's stages split into chunks of whole output currents
    of at most ``limit`` components (a larger current is a chunk alone),
    each with the interactions, inputs and slots of its currents."""

    slots = {
        int(slot["value_slot_id"]): slot
        for slot in schema["value_storage"]["value_slots"]
    }
    out: list[dict[str, Any]] = []
    for stage in schema["stages"]:
        if not stage.get("interactions_compacted"):
            raise ValueError("native trees chunk compacted runtime stages only")
        outputs: dict[int, list[int]] = defaultdict(list)
        for slot in stage["output_value_slot_ids"]:
            outputs[int(slots[int(slot)]["current_id"])].append(int(slot))
        inputs = {
            int(slots[int(slot)]["current_id"]): int(slot)
            for slot in stage["input_value_slot_ids"]
        }
        chunks: list[list[int]] = [[]]
        size = 0
        for current in sorted(outputs):
            dimension = sum(
                int(slots[slot]["component_stop"]) - int(slots[slot]["component_start"])
                for slot in outputs[current]
            )
            if chunks[-1] and size + dimension > limit:
                chunks.append([])
                size = 0
            chunks[-1].append(current)
            size += dimension
        chunk_of = {current: c for c, chunk in enumerate(chunks) for current in chunk}
        interactions: list[list[int]] = [[] for _ in chunks]
        for item in stage["interaction_ids"]:
            interactions[chunk_of[int(dag.interactions[int(item)].result_id)]].append(
                int(item)
            )
        for chunk, ids in zip(chunks, interactions, strict=True):
            used = sorted(
                {
                    current
                    for item in ids
                    for current in (
                        dag.interactions[item].left_id,
                        dag.interactions[item].right_id,
                    )
                }
            )
            out.append(
                {
                    **stage,
                    "input_current_ids": used,
                    "output_current_ids": list(chunk),
                    "input_value_slot_ids": sorted(inputs[c] for c in used),
                    "output_value_slot_ids": sorted(
                        slot for current in chunk for slot in outputs[current]
                    ),
                    "interaction_count": len(ids),
                    "interaction_ids": ids,
                }
            )
    return out


def _zero_colour():
    from pyamplicol.color.connections import ExactColorCoefficient

    return ExactColorCoefficient()


def _external_masses(slice_: Any) -> tuple[tuple[int, str], ...]:
    """(PDG code, mass parameter) of every massive external particle."""

    data = slice_.data
    by_pdg = {int(entry["pdg_code"]): entry for entry in data["particles"]}
    out = {}
    for leg in slice_.blocks.process.legs:
        entry = by_pdg[int(leg.pdg)]
        mass = entry.get("mass")
        if mass not in (None, "ZERO", "0"):
            out[abs(int(leg.pdg))] = str(mass)
    return tuple(sorted(out.items()))


def batch_inputs(
    slice_: Any,
    trees: NativeTrees,
    evaluations: Sequence[Any],
    value: Any,
) -> dict[str, Any]:
    """The parameter-only tree inputs of a batch (``inputs["trees"]``): per
    point the model parameters at the carrier's digits (the scale may change
    per point), and the batch's exact external masses and Born couplings,
    which may not (``value`` converts a carrier complex for the executor).
    The executor rounds the masses to its carrier, so the wavefunctions carry
    its precision."""

    from .assemble import _kernel_parameters

    model = []
    shared = None
    for evaluation in evaluations:
        numbers, parameters = evaluation.numbers, evaluation.parameters
        digits = numbers.precision.working_digits
        with numbers.scope():
            model.append(
                [str(x) for x in slice_.tree_rows._model_values(parameters, digits)]
            )
            scalars = slice_.scalars(parameters, numbers, True)
            couplings = [
                value(scalars[name, 0][grade]) for name, grade in trees.couplings
            ]
        values = _kernel_parameters(slice_.compiled, parameters)
        masses = [(pdg, _fraction_text(values[name])) for pdg, name in trees.masses]
        common = (couplings, masses)
        if shared is None:
            shared = common
        elif common != shared:
            raise ValueError(
                "the points of one batch need equal Born couplings and external masses"
            )
    if shared is None:
        raise ValueError("a batch needs points")
    couplings, masses = shared
    return {"model": model, "masses": masses, "couplings": couplings}
