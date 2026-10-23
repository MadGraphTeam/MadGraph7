# SPDX-License-Identifier: 0BSD
"""Native per-point inputs of the v2 executor (WP3.1 I3/I4).

The slice forms every point's block rows and Born amplitudes in Python
(``assemble._PointEvaluation``: ``TreeRows`` replays the tree program in
Decimal, the Born root closure runs per helicity row). This module exports
the same program for the native executor (``rusticol-core``
``loop_v2::trees``), which then forms them without Python per point:

* **trees**: the slice's blocks DAG as an interaction program over the
  runtime schema of ``assemble.TreeRows`` (``_interaction_program``): per
  stage, each interaction's exact colour weight times its vertex kernel on
  its input rows, summed per current, then each output slot copied or
  propagated. Kernels are the model's own component expressions, one per
  vertex and propagator shape, saved as exact Symbolica evaluators (any
  carrier; a few per process). With the runtime schema's sources, momentum
  slots and model-parameter layout. A plan block row is the Wick-signed
  state slice of its tree current. (The I2 decision, PLAN §10 item 18: this
  lane replaced the per-stage evaluators of the first native lane, which
  grew with the number of currents' outputs: about 200 GB at 8 legs.)
* **Born**: every root-closure term (``assemble._born_terms``: an ordinary
  vertex holding leg 1 closed on 2-3 blocks) as a plan kernel on its blocks'
  rows and momenta, with its sign, its coupling per N_c grade and the exact
  colour weights of its literal tensors, accumulated per Born helicity row
  into the components A0[b, Born order, grade]; with ``born_squared`` the
  full-colour metric of the Born basis gives |A0|^2 per squared order.

Per batch only parameter-dependent values cross (``batch_inputs``): the model
parameters, the masses of massive external particles and the couplings. The
plan's kernel constants carry the Born kernels' model factors.

At leading and next-to-leading colour the Born metric is per Born component
and squared Born order, keeping the N_c powers the projection keeps
(``_component_metric``). The exact carrier has no native trees (wavefunctions
are irrational); the slice's Python inputs remain its path.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from pyamplicol.color.contraction_types import NC

from .families import Orders, add_orders

TREE_SCHEMA = 2


@dataclass(frozen=True)
class NativeTrees:
    """A tree program for ``V2Executor(trees=...)`` and its batch layout."""

    program: dict[str, Any]
    # The interaction program's saved kernel evaluators (vertex kernels, then
    # propagator kernels).
    kernels: tuple[bytes, ...]
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
        return self.json(), list(self.kernels)


def _fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator if value.denominator == 1 else value)


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
    interactions, saved = _interaction_program(trees)
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
            kernel = exporter._kernel(term.binding, ())
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
    gram = metric = None
    if born_squared and slice_.request.accuracy != "full":
        metric = _component_metric(slice_, components, squared_orders)
    elif born_squared:
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
        "interactions": interactions,
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
            "metric": metric,
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


def _interaction_program(trees: Any) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """The blocks DAG of ``trees`` (a ``TreeRows``) as an interaction program.

    Per runtime-schema stage, every interaction adds its exact colour weight
    times its vertex kernel, evaluated on its two input rows and momenta, to
    its result current; each output slot then copies the sum or applies the
    current's propagator kernel. Kernels are the model's own component
    expressions (``vertex_component_expression``,
    ``propagator_component_expression``, as the stage program sums them per
    current) on formal inputs ``[left, right, p_left, p_right, model]`` and
    ``[value, p, model]``, one per distinct shape, saved as exact Symbolica
    evaluators. The program's size is that of the DAG; the saved evaluators
    come first for the kernels, then for the propagators."""

    from symbolica import Expression, FunctionDefinition, S

    from pyamplicol.generation.stage_parameters import (
        _logical_model_parameter_symbols,
        _model_symbolica_functions,
        _specialize_stage_symbolica_functions,
    )
    from pyamplicol.models._physics_ir import PropagatorIR

    from .assemble import _canonical_stage

    dag, schema = trees.dag, trees.schema
    records = trees._records
    model_symbols = [S(f"pyamplicol_v2_trees::m{i}") for i in range(len(records))]
    model = trees.model.with_runtime_parameters(
        _logical_model_parameter_symbols(
            records,
            {
                str(r["name"]): sym
                for r, sym in zip(records, model_symbols, strict=True)
            },
        )
    )
    value_count = trees._value_count
    value_slots = {
        int(slot["value_slot_id"]): slot
        for slot in schema["value_storage"]["value_slots"]
    }
    current_slots = {
        int(slot["current_id"]): slot
        for slot in schema["current_storage"]["current_slots"]
    }
    momentum_start = {
        int(slot["momentum_mask"]): value_count + int(slot["component_start"])
        for slot in schema["momentum_slots"]
    }
    saved: dict[str, list[bytes]] = {"kernels": [], "propagators": []}
    shapes: dict[str, list[list[int]]] = {"kernels": [], "propagators": []}
    ids: dict[tuple, int] = {}

    def symbols(name: str, count: int) -> list[Any]:
        return [S(f"pyamplicol_v2_trees::{name}{i}") for i in range(count)]

    def compile_kernel(kind: str, key: tuple, build) -> int:
        found = ids.get((kind, key))
        if found is not None:
            return found
        inputs, components = build()
        # The model's kernel functions materialise with their first use.
        outputs, retained = _specialize_stage_symbolica_functions(
            list(components), _model_symbolica_functions(model)
        )
        # On canonical symbols the kernel's term order, and so its binary64
        # rounding, does not depend on the symbols created earlier in the
        # interpreter (``assemble._canonical_stage``).
        outputs, parameters, functions = _canonical_stage(
            SimpleNamespace(
                output_expressions=list(outputs),
                parameter_symbols=[*inputs, *model_symbols],
                symbolica_functions=list(retained),
            )
        )
        evaluator = Expression.evaluator_multiple(
            outputs,
            parameters,
            functions=[
                FunctionDefinition(function, arguments, body, inlining="always")
                for function, arguments, body in functions
            ],
            jit_compile=False,
        )
        found = ids[(kind, key)] = len(saved[kind])
        saved[kind].append(bytes(evaluator.save()))
        shapes[kind].append([len(inputs) + len(model_symbols), len(outputs)])
        return found

    def vertex(interaction: Any) -> int:
        left = dag.currents[interaction.left_id]
        right = dag.currents[interaction.right_id]
        result = dag.currents[interaction.result_id]
        key = (
            int(interaction.vertex_kind),
            int(result.index.particle_id),
            int(result.index.chirality),
            int(left.index.chirality),
            int(right.index.chirality),
            int(left.dimension),
            int(right.dimension),
        )

        def build():
            lhs, rhs = symbols("l", left.dimension), symbols("r", right.dimension)
            pl, pr = symbols("pl", 4), symbols("pr", 4)
            return [*lhs, *rhs, *pl, *pr], model.vertex_component_expression(
                key[0],
                lhs,
                rhs,
                result_particle_id=key[1],
                result_chirality=key[2],
                left_chirality=key[3],
                right_chirality=key[4],
                coupling=interaction.coupling,
                left_momentum=pl,
                right_momentum=pr,
            )

        return compile_kernel("kernels", key, build)

    def propagator(current_id: int, slot: Mapping[str, Any]) -> int:
        current = current_slots[current_id]
        dimension = int(current["dimension"])
        key = (
            int(current["particle_id"]),
            int(current["chirality"]),
            dimension,
            json.dumps(slot["propagator"], sort_keys=True),
        )

        def build():
            value, momentum = symbols("v", dimension), symbols("p", 4)
            return [*value, *momentum], model.propagator_component_expression(
                key[0],
                value,
                momentum,
                chirality=key[1],
                propagator=PropagatorIR.from_json_dict(dict(slot["propagator"])),
            )

        return compile_kernel("propagators", key, build)

    steps = []
    for stage in schema["stages"]:
        if not stage.get("interactions_compacted"):
            raise ValueError("native trees read compacted runtime stages only")
        inputs = {
            int(value_slots[int(slot)]["current_id"]): value_slots[int(slot)]
            for slot in stage["input_value_slot_ids"]
        }
        accumulators: dict[int, int] = {}
        dimensions: list[int] = []
        for current in sorted(
            {
                int(value_slots[int(slot)]["current_id"])
                for slot in stage["output_value_slot_ids"]
            }
        ):
            accumulators[current] = len(dimensions)
            dimensions.append(int(current_slots[current]["dimension"]))
        rows = []
        for item in stage["interaction_ids"]:
            interaction = dag.interactions[int(item)]
            if complex(*interaction.evaluation_factor) == 0j:
                continue  # a certified zero attachment (as the stage program)
            left, right = inputs[interaction.left_id], inputs[interaction.right_id]
            weight = _exact_weight(complex(*interaction.color_weight))
            rows.append(
                [
                    vertex(interaction),
                    int(left["component_start"]),
                    int(left["dimension"]),
                    int(right["component_start"]),
                    int(right["dimension"]),
                    momentum_start[
                        dag.currents[interaction.left_id].index.momentum_mask
                    ],
                    momentum_start[
                        dag.currents[interaction.right_id].index.momentum_mask
                    ],
                    accumulators[int(interaction.result_id)],
                    _fraction_text(weight.real),
                    _fraction_text(weight.imag),
                ]
            )
        outputs = []
        for slot_id in stage["output_value_slot_ids"]:
            slot = value_slots[int(slot_id)]
            current = int(slot["current_id"])
            variant = str(slot["variant"])
            if variant not in ("propagated", "unpropagated"):
                raise ValueError(f"a stage output of variant {variant!r}")
            outputs.append(
                [
                    accumulators[current],
                    propagator(current, slot) if variant == "propagated" else -1,
                    momentum_start[int(current_slots[current]["momentum_mask"])],
                    int(slot["component_start"]),
                    int(slot["dimension"]),
                ]
            )
        steps.append(
            {"interactions": rows, "accumulators": dimensions, "outputs": outputs}
        )
    program = {
        "kernels": shapes["kernels"],
        "propagators": shapes["propagators"],
        "steps": steps,
    }
    return program, (*saved["kernels"], *saved["propagators"])


@dataclass(frozen=True)
class _ExactComplex:
    real: Fraction
    imag: Fraction


def _exact_weight(value: complex) -> _ExactComplex:
    """A stored binary64 colour weight, exactly: the U(1)-subtraction weight
    +-1/N_c as the rational (``assemble._exact_u1_subtraction_weights``),
    every other weight as its binary64 value (``models.expressions.
    _exact_complex_expression``)."""

    for sign in (1, -1):
        if value == complex(sign / NC):
            return _ExactComplex(Fraction(sign, NC), Fraction(0))
    return _ExactComplex(
        Fraction(*value.real.as_integer_ratio()),
        Fraction(*value.imag.as_integer_ratio()),
    )


def _component_metric(
    slice_: Any,
    components: dict[tuple[int, Orders, int], int],
    squared_orders: Sequence[Orders],
) -> list[list[Any]]:
    """The Born colour metric of a leading or next-to-leading colour request
    over Born components (b, Born order, N_c power), per squared Born order:
    <b|b'> keeping the N_c powers the projection keeps (``colour.kept``)
    below the order's leading degree, at N_c = 3, as the Python inputs'
    Born interference (``assemble._interfere``)."""

    from .colour import kept

    accuracy = slice_.request.accuracy
    degrees = slice_.degrees()["born"]
    index = {order: s for s, order in enumerate(squared_orders)}
    gram = slice_.basis.gram
    out = []
    for (b, left, p), j in components.items():
        for (b2, right, q), j2 in components.items():
            squared = add_orders(left, right)
            degree = degrees.get(squared)
            value = _zero_colour()
            for power, cell in gram[b][b2].items():
                exponent = power + p + q
                if kept(exponent, degree, accuracy):
                    value = value + cell.scaled(Fraction(NC) ** exponent)
            if value:
                out.append(
                    [
                        index[squared],
                        j,
                        j2,
                        _fraction_text(value.real),
                        _fraction_text(value.imag),
                    ]
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
