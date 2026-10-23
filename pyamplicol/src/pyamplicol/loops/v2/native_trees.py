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

* **parameters**: one saved exact evaluator from the external model
  parameters to every parameter-dependent value of the native lane
  (``_parameter_program``): the tree program's derived model parameters (the
  model's own definitions), the Born couplings per N_c grade (the slice's
  scalar plan, ``LoopSlice._scalar_plan``) and the plan's kernel constants
  (``reference_executor.KernelConstants`` keys, with their mass flips),
  plus the exact mass parameters of the external spinors and the loop
  propagators. The plan's kernel constants carry the Born kernels' model
  factors.

Per batch only momenta, scales (mu_R^2 per point) and the external parameter
values cross (``batch_parameters``); the executor binds everything else
natively, the scale parameter from each point's mu_R^2.

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
    # The parameter program's saved exact evaluator.
    parameters: bytes = b""
    # (component, sector, epsilon power) of each native counterterm and
    # external-field output (``_counterterm_program``); empty without them.
    counterterm_outputs: tuple[tuple[str, Orders, int], ...] = ()

    def json(self) -> str:
        return json.dumps(self.program, separators=(",", ":"))

    def executor_argument(self) -> tuple[str, list[bytes]]:
        return self.json(), [*self.kernels, self.parameters]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.program["parameters"]["names"])


def _fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator if value.denominator == 1 else value)


def export_trees(
    slice_: Any,
    exporter: Any,
    rows: tuple[tuple[tuple[int, int], ...], ...],
    *,
    born_squared: bool,
    with_counterterms: bool = True,
) -> NativeTrees:
    """The tree program of ``slice_`` for the plan ``exporter`` is lowering.

    Call after the exporter's seed records (the plan's Born components are
    numbered there); the Born kernels and their constants join the plan, as
    do the counterterm closure kernels with ``with_counterterms``
    (``_counterterm_program``).
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

    # Closure terms share their input layouts and colour weights (interned).
    layouts: dict[tuple[tuple[Any, ...], ...], int] = {}
    weights: dict[tuple[str, str, int], int] = {}

    def record(
        term: Any, kernel: int, inputs: list[list[Any]], contributions: list[list[Any]]
    ) -> dict[str, Any]:
        """A closure term's record: its kernel on the layout of its inputs,
        and per contribution (coupling, component, weight); ``contributions``
        are (coupling, component, weight re, weight im, sqrt(2) power)."""

        layout = layouts.setdefault(tuple(tuple(item) for item in inputs), len(layouts))
        return {
            "kernel": kernel,
            "layout": layout,
            "blocks": [block_id(block) for block in term.blocks],
            "masks": [block.key.external_mask for block in term.blocks],
            "sign": term.sign,
            "contributions": [
                [
                    coupling,
                    component,
                    weights.setdefault((real, imag, power), len(weights)),
                ]
                for coupling, component, real, imag, power in contributions
            ],
        }

    def closure_term(term: Any, contributions: list[list[Any]]) -> dict[str, Any]:
        """A root-closure term (``ct_sites.SiteTerm``: a vertex closed on
        tree blocks) as a plan kernel on its blocks' rows and momenta."""

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
                raise ValueError(f"a closure kernel reads a {kind} input")
        return record(term, kernel, inputs, contributions)

    def whole_term(term: Any, contributions: list[list[Any]]) -> dict[str, Any]:
        """A whole-rule continuation term (``assemble.WholeRuleTerm``): its
        sum of products of Minkowski contractions of block rows as a plan
        kernel on the blocks' four components."""

        ids = [block_id(block) for block in term.blocks]
        if any(block_dimensions[i] != 4 for i in ids):
            raise NotImplementedError("a whole-rule term on a non-vector block")
        polynomial: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
        for coefficient, pairs in term.pairings:
            partial = {(): Fraction(coefficient)}
            for a, b in pairs:
                expanded: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
                for monomial, value in partial.items():
                    for mu, metric in ((0, 1), (1, -1), (2, -1), (3, -1)):
                        key = tuple(sorted((*monomial, 4 * a + mu, 4 * b + mu)))
                        expanded[key] += value * metric
                partial = expanded
            for monomial, value in partial.items():
                polynomial[monomial] += value
        entries = tuple(
            (monomial, value) for monomial, value in sorted(polynomial.items()) if value
        )
        key = ("whole-rule", len(ids), entries)
        kernel = exporter.kernels.get(key)
        if kernel is None:
            outputs = []
            for monomial, value in entries:
                constant = exporter.constants.setdefault(
                    (((value, Fraction(0)), (), (), ()), ()), len(exporter.constants)
                )
                powers: dict[int, int] = defaultdict(int)
                for index in monomial:
                    powers[index] += 1
                outputs.append([constant, [[i, e] for i, e in sorted(powers.items())]])
            kernel = exporter.kernels[key] = len(exporter.kernel_records)
            exporter.kernel_records.append(
                {
                    "inputs": [["current", k, 0] for k in range(4 * len(ids))],
                    "outputs": [outputs],
                }
            )
        inputs = [["current", slot, k] for slot in range(len(ids)) for k in range(4)]
        return record(term, kernel, inputs, contributions)

    terms = []
    born_orders = tuple(sorted(slice_.born_terms))
    for order, order_terms in sorted(slice_.born_terms.items()):
        for term in order_terms:
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
            terms.append(closure_term(term, contributions))
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
    counterterms = None
    if with_counterterms:
        counterterms = _counterterm_program(
            slice_, closure_term, whole_term, components, born_orders
        )
    masses = _external_masses(slice_)
    parameters, parameter_evaluator = _parameter_program(
        slice_,
        exporter,
        trees,
        tuple(couplings),
        masses,
        () if counterterms is None else counterterms.couplings,
    )
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
        "parameters": parameters,
        "counterterms": None if counterterms is None else counterterms.record,
        # Per closure-term layout its kernel inputs: ("current", slot,
        # component) or ("momentum", slot, mu); per colour weight (re, im,
        # sqrt(2) power p): the weight times 2^(-p/2).
        "layouts": [[list(item) for item in layout] for layout in layouts],
        "weights": [list(weight) for weight in weights],
    }
    return NativeTrees(
        program,
        saved,
        tuple(couplings),
        born_orders,
        squared_orders,
        masses,
        tuple(components),
        parameter_evaluator,
        () if counterterms is None else counterterms.outputs,
    )


@dataclass(frozen=True)
class _Counterterms:
    record: dict[str, Any]
    # Per coupling slot ("scalar", coupling name, epsilon power, N_c grade),
    # or ("whole", N_c grade, coefficient expression, its scalar names) for
    # a whole-rule continuation's coefficient (``assemble.WholeRuleTerm``).
    couplings: tuple[tuple[Any, ...], ...]
    # (component, sector, epsilon power) per output.
    outputs: tuple[tuple[str, Orders, int], ...]


def _counterterm_program(
    slice_: Any,
    closure_term: Any,
    whole_term: Any,
    components: Mapping[tuple[int, Orders, int], int],
    born_orders: Sequence[Orders],
) -> _Counterterms:
    """The counterterm and external-field components of the native lane, as
    ``assemble._PointEvaluation._born_and_counterterms`` forms them.

    * **Counterterm amplitudes**: every term of the slice's counterterm
      closure (``LoopSlice.site_terms``: a counterterm vertex closed on tree
      blocks, ``ct_sites.SiteTerm``) is a closure term like the Born's; per
      epsilon power its coupling's N_c grades (the scalar plan) are coupling
      slots, and its literal colour tensors accumulate, times the site's
      multiplicity, into the ket components (tensor, vector, N_c power), one
      vector per (component, epsilon power, loop orders). The closure is the
      same for WP2.2's counterterm recursion, whose root terms are site terms
      on the counterterm program's blocks.
    * **Interferences**: per output (component, sector, epsilon power) the
      entries ``2 Re(c <A0(order)|M|ket>)`` of every Born order of the
      sector: a counterterm vector (c = 1), or an external-field insertion
      (``LoopSlice.fields``: the loop-side Born of its Born order, c its
      field coefficient per epsilon power and N_c grade). ``M`` is the exact
      overlap of the Born basis with the ket tensors, each cell's N_c power
      kept as the request's projection keeps it (``colour.kept``) below the
      sector's degree, at N_c = 3 (``assemble._interfere``).

    Whole-rule colour continuations (``assemble.WholeRuleTerm``, leading
    and next-to-leading colour) are closure terms whose kernel is their
    polynomial in the block rows (``whole_term``) and whose coupling slots
    are their N_c-graded coefficients.
    """

    from pyamplicol.color.connections import overlap_traces

    from .assemble import LAURENT_POWERS, WholeRuleTerm, _single
    from .colour import _DEPTH, _literal_tensor_overlap_nc_terms, _trace_overlaps
    from .fields import FIELD_COMPONENT

    _plan, layout = slice_._scalar_plan(LAURENT_POWERS)
    legs = slice_.basis.legs
    basis = slice_.basis.tensors
    whole_expressions: dict[int, Any] = {}
    tensors: dict[Any, int] = {tensor: b for b, tensor in enumerate(basis)}
    # The literal tensor of each ket tensor (``LoopSlice.overlaps`` reads
    # literal tensors).
    literal = [tensor.terms[0].tensor for tensor in basis]
    couplings: dict[tuple[Any, ...], int] = {}
    vectors: dict[tuple[str, int, Orders], int] = {}
    ket: dict[tuple[int, int, int], int] = {}
    terms = []

    # Ket tensor index per literal tensor (one ``_single`` per distinct one).
    index_of: dict[Any, int] = {}

    def tensor_index(tensor: Any) -> int:
        found = index_of.get(tensor)
        if found is None:
            found = index_of[tensor] = tensors.setdefault(
                _single(tensor, legs), len(tensors)
            )
            if found == len(literal):
                literal.append(tensor)
        return found

    def colour_items(colour: Any, multiplicity: int) -> list[tuple[Any, ...]]:
        """(ket tensor, N_c power, weight re, weight im) per literal tensor."""

        return [
            (
                tensor_index(item.tensor),
                item.nc_power,
                _fraction_text(item.coefficient.real * multiplicity),
                _fraction_text(item.coefficient.imag * multiplicity),
            )
            for item in colour.terms
        ]

    def contribute(out, slot, vector, grade, items, power2) -> None:
        for t, nc_power, real, imag in items:
            k = ket.setdefault((t, vector, nc_power + grade), len(ket))
            out.append([slot, k, real, imag, power2])

    for site, site_terms in slice_.site_terms.items():
        multiplicity = len(site.alternatives)
        for term in site_terms:
            contributions: list[list[Any]] = []
            items = colour_items(term.colour, multiplicity)
            power2 = term.colour.normalization_power
            if isinstance(term, WholeRuleTerm):
                # Only the eps^0 coefficient exists; its N_c grades are model
                # expressions of the named scalars' grade-0 values.
                vector = vectors.setdefault(
                    (site.component, 0, site.orders), len(vectors)
                )
                for degree, expression in term.grades:
                    key = (
                        "whole",
                        degree,
                        expression.to_canonical_string(),
                        tuple(term.parameters),
                    )
                    slot = couplings.get(key)
                    if slot is None:
                        slot = couplings[key] = len(couplings)
                        whole_expressions[slot] = expression
                    contribute(contributions, slot, vector, degree, items, power2)
                if contributions:
                    terms.append(whole_term(term, contributions))
                continue
            name = term.entry.coupling.name
            for power in LAURENT_POWERS:
                grades = layout.get((name, power), ())
                if not grades:
                    continue
                vector = vectors.setdefault(
                    (site.component, power, site.orders), len(vectors)
                )
                for grade, _column in grades:
                    slot = couplings.setdefault(
                        ("scalar", name, power, grade), len(couplings)
                    )
                    contribute(contributions, slot, vector, grade, items, power2)
            if contributions:
                terms.append(closure_term(term, contributions))
    born_index = {order: o for o, order in enumerate(born_orders)}
    sector_of = {
        pair: s for s, (_sector, pairs) in enumerate(slice_.sectors) for pair in pairs
    }
    accuracy = slice_.request.accuracy
    degrees = slice_.degrees()["virtual"] if accuracy != "full" else None

    def degree(s: int) -> int | None:
        return None if degrees is None else degrees[slice_.sectors[s][0]]

    outputs: dict[tuple[str, int, int], int] = {}
    entries = []
    for (component, power, loop_order), vector in vectors.items():
        for order in born_orders:
            s = sector_of.get((order, loop_order))
            if s is None:
                continue
            out = outputs.setdefault((component, s, power), len(outputs))
            entries.append(
                ["vector", born_index[order], vector, None, 0, out, degree(s)]
            )
    field_names = slice_.field_names
    for insertion in slice_.fields:
        name = field_names[insertion.identity]
        ket_order = born_index.get(insertion.born_order)
        if ket_order is None:
            continue
        for order in born_orders:
            s = sector_of.get((order, insertion.orders))
            if s is None:
                continue
            for power in LAURENT_POWERS:
                for grade, _column in layout.get((name, power), ()):
                    slot = couplings.setdefault(
                        ("scalar", name, power, grade), len(couplings)
                    )
                    out = outputs.setdefault((FIELD_COMPONENT, s, power), len(outputs))
                    entries.append(
                        [
                            "born",
                            born_index[order],
                            ket_order,
                            slot,
                            grade,
                            out,
                            degree(s),
                        ]
                    )
    # Exact overlaps of the Born basis with every ket tensor, per N_c power:
    # the basis Gram, and the other ket tensors' overlaps counted natively
    # in one batch (``colour.contract_colour`` of unit literal tensors).
    gram = slice_.basis.gram
    cells = []
    for b in range(len(basis)):
        for t in range(len(basis)):
            for power, cell in gram[b][t].items():
                if cell:
                    cells.append(
                        [
                            b,
                            t,
                            power,
                            _fraction_text(cell.real),
                            _fraction_text(cell.imag),
                        ]
                    )
    pairs = [(b, t) for b in range(len(basis)) for t in range(len(basis), len(literal))]
    if pairs:
        counted = _trace_overlaps()(
            [overlap_traces(literal[b], literal[t]) for b, t in pairs]
        )
        for (b, t), laurent in zip(pairs, counted, strict=True):
            nc_terms = (
                _literal_tensor_overlap_nc_terms(literal[b], literal[t]).items()
                if laurent is None
                else ((power, Fraction(value)) for power, value in laurent)
            )
            found: dict[int, Fraction] = defaultdict(Fraction)
            for power, value in nc_terms:
                found[power] += value
            cells.extend(
                [b, t, power, _fraction_text(value), "0"]
                for power, value in sorted(found.items())
                if value
            )
    record = {
        "couplings": len(couplings),
        "coupling_powers": [key[2] if key[0] == "scalar" else 0 for key in couplings],
        "ket": [list(key) for key in ket],
        "vectors": len(vectors),
        "tensors": len(tensors),
        "cells": cells,
        "born": [[b, born_index[order], grade] for b, order, grade in components],
        "terms": terms,
        "entries": entries,
        "outputs": [power for _component, _s, power in outputs],
        "depth": None if accuracy == "full" else _DEPTH[accuracy],
    }
    return _Counterterms(
        record,
        tuple(
            key if key[0] == "scalar" else (*key[:2], whole_expressions[slot], key[3])
            for key, slot in couplings.items()
        ),
        tuple(
            (component, slice_.sectors[s][0], power) for component, s, power in outputs
        ),
    )


def _parameter_program(
    slice_: Any,
    exporter: Any,
    trees: Any,
    couplings: Sequence[tuple[str, int]],
    masses: Sequence[tuple[int, str]],
    counterterm_couplings: Sequence[tuple[Any, ...]] = (),
) -> tuple[dict[str, Any], bytes]:
    """The parameter program of the native lane: its record and its saved
    exact evaluator (``loop_v2::trees`` binds a batch with it).

    Inputs are the tree runtime schema's external model parameters (real),
    in the schema's order. Outputs are complex: per derived model parameter
    of the tree program its definition (``TreeRows._definitions``, as
    ``TreeRows._model_values`` evaluates them in Decimal), per Born coupling
    slot (name, N_c grade) its scalar-plan coefficient, per counterterm or
    field coupling slot (name, epsilon power, N_c grade) its scalar-plan
    coefficient and the coefficient's active diagnostic (the executor checks
    it finite, as ``scalar_bindings.NativeScalarBindings`` does; the model's
    helper functions, e.g. ``cond`` and ``reglog``, are the scalar library's
    definitions), per whole-rule coefficient slot its grade expression on the
    named scalars' grade-0 coefficients (``assemble.WholeRuleTerm
    .coefficients``), per plan kernel constant its key's value (coefficient,
    exact and irrational factors, with the reading's mass flips, as
    ``reference_executor.KernelConstants`` forms it in Python) and every base
    a fractional power was distributed over (the executor checks it
    positive, as ``KernelConstants`` does). The evaluator is compiled on
    canonical symbols (``assemble._canonical_stage``): its term order does
    not depend on the process history.
    """

    from symbolica import E, Expression, FunctionDefinition, Replacement, S

    from pyamplicol._internal.physics.symbols import symbols

    from .assemble import LAURENT_POWERS, _canonical_stage

    records = trees._records
    names = [str(r["name"]) for r in records if r["kind"] == "external_parameter"]
    defaults = [
        _fraction_text(Fraction(repr(r["default"])))
        for r in records
        if r["kind"] == "external_parameter"
    ]
    inputs = [S(f"pyamplicol_v2_parameters::e{i}") for i in range(len(names))]
    index = {name: i for i, name in enumerate(names)}
    symbol_of = dict(zip(names, inputs, strict=True))
    outputs: list[Any] = []
    output_of: dict[Any, int] = {}

    def output(key: Any, build) -> int:
        found = output_of.get(key)
        if found is None:
            found = output_of[key] = len(outputs)
            outputs.append(build())
        return found

    # Derived model parameters, in the model's own symbols.
    model_symbols = symbols.model(slice_.compiled.name)
    externals = [
        Replacement(E(model_symbols.symbol(name).to_canonical_string()), symbol)
        for name, symbol in symbol_of.items()
    ]
    model = []
    for record in records:
        kind = record["kind"]
        if kind == "external_parameter":
            model.append(["input", index[str(record["name"])], False])
        elif kind == "derived_parameter_component":
            name = str(record["runtime_name"])
            o = output(
                ("model", name),
                lambda name=name: E(trees._definitions[name]).replace_multiple(
                    externals
                ),
            )
            model.append(["output", o, record["complex_component"] == "imag"])
        else:
            raise NotImplementedError(f"a tree model parameter of kind {kind!r}")
    # Born couplings per N_c grade, from the slice's scalar plan.
    plan, layout = slice_._scalar_plan(LAURENT_POWERS)
    plan_inputs = []
    for name, symbol in zip(plan.external_names, plan.input_symbols, strict=True):
        if name not in index:
            raise NotImplementedError(
                f"the scalar plan reads {name!r}, which the tree program has not"
            )
        plan_inputs.append(Replacement(symbol, symbol_of[name]))
    coupling_outputs = []
    for name, grade in couplings:
        (column,) = [c for degree, c in layout[(name, 0)] if degree == grade]
        coupling_outputs.append(
            output(
                ("coupling", name, grade),
                lambda column=column: plan.expressions[column].value.replace_multiple(
                    plan_inputs
                ),
            )
        )

    # Counterterm and field couplings per epsilon power and N_c grade; they
    # may read the renormalisation scale (one value per point).
    counterterm_outputs, finite = [], []

    def scalar(name: str, power: int, grade: int) -> Any:
        """The scalar-plan coefficient (and its diagnostic, checked finite)."""

        (column,) = [c for degree, c in layout[(name, power)] if degree == grade]
        expression = plan.expressions[column]
        if expression.diagnostic != 0:
            finite.append(
                output(
                    ("diagnostic", name, power, grade),
                    lambda e=expression: e.diagnostic.replace_multiple(plan_inputs),
                )
            )
        return expression.value

    for slot in counterterm_couplings:
        if slot[0] == "scalar":
            _kind, name, power, grade = slot
            value = scalar(name, power, grade)
            counterterm_outputs.append(
                output(
                    ("counterterm", name, power, grade),
                    lambda v=value: v.replace_multiple(plan_inputs),
                )
            )
            continue
        _kind, degree, coefficient, scalars = slot
        bound = coefficient.replace_multiple(
            [
                Replacement(symbols.qualified(f"UFO::{name}"), scalar(name, 0, 0))
                for name in scalars
            ]
        )
        counterterm_outputs.append(
            output(
                ("whole", degree, coefficient.to_canonical_string(), scalars),
                lambda v=bound: v.replace_multiple(plan_inputs),
            )
        )
    helpers = plan.program.library._bodies if finite or counterterm_outputs else {}

    # Plan kernel constants.
    def base(factor: Any, flips: Mapping[str, int]) -> Any:
        kind, value = factor
        if kind == "param":
            if value not in index:
                raise NotImplementedError(
                    f"a kernel constant reads {value!r}, which the tree program has not"
                )
            return symbol_of[value] * flips.get(value, 1)
        if kind == "prime":
            return Expression.num(int(value))
        if kind == "pi":
            return Expression.PI
        text, pairs = value
        return E(text).replace_multiple(
            [
                Replacement(E(symbol), base(("param", name), flips))
                for symbol, name in pairs
            ]
        )

    def power(factor: Any, exponent: Fraction, flips: Mapping[str, int]) -> Any:
        return base(factor, flips) ** E(f"{exponent.numerator}/{exponent.denominator}")

    constants, positive = [], []
    for key, flips in exporter.constants:
        coefficient, exact_factors, irrational, bases = key
        signs = dict(flips)

        def build(
            coefficient=coefficient,
            exact_factors=exact_factors,
            irrational=irrational,
            signs=signs,
        ):
            real, imag = coefficient
            value = (
                E(_fraction_text(Fraction(real)))
                + E(_fraction_text(Fraction(imag))) * Expression.I
            )
            for factor, exponent in exact_factors:
                value = value * power(factor, Fraction(exponent), signs)
            for factor, exponent in irrational:
                value = value * power(factor, Fraction(exponent), signs)
            return value

        constants.append(output(("constant", key, flips), build))
        for factor in bases:
            positive.append(
                output(
                    ("positive", factor, flips),
                    lambda factor=factor, signs=signs: base(factor, signs),
                )
            )
    # Exact masses: external spinors and loop propagators.
    for _pdg, name in masses:
        if name not in index:
            raise NotImplementedError(f"the external mass {name!r} is no input")
    plan_masses = []
    for mass in exporter.masses:
        if mass.massless:
            plan_masses.append(None)
        elif mass.mass not in index:
            raise NotImplementedError(f"the loop mass {mass.mass!r} is no input")
        else:
            plan_masses.append(index[mass.mass])
    canonical, parameters, functions = _canonical_stage(
        SimpleNamespace(
            output_expressions=outputs,
            parameter_symbols=inputs,
            symbolica_functions=[
                (head, tuple(arguments), body)
                for head, (arguments, body) in helpers.items()
            ],
        )
    )
    # Helper calls stay calls, as in the scalar library's own evaluator.
    evaluator = Expression.evaluator_multiple(
        canonical,
        parameters,
        functions=[
            FunctionDefinition(head, list(arguments), body, inlining="never")
            for head, arguments, body in functions
        ],
        jit_compile=False,
    )
    record = {
        "names": names,
        "defaults": defaults,
        # The renormalisation scale's parameter: sqrt(mu_R^2) per point.
        "scale": index.get(slice_.scale_parameter),
        "outputs": len(outputs),
        "model": model,
        "couplings": coupling_outputs,
        "counterterm_couplings": counterterm_outputs,
        "finite": finite,
        "constants": constants,
        "positive": sorted(set(positive)),
        "masses": [[int(pdg), index[name]] for pdg, name in masses],
        "plan_masses": plan_masses,
    }
    return record, bytes(evaluator.save())


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


def batch_parameters(trees: NativeTrees, evaluations: Sequence[Any]) -> list[str]:
    """The exact values of the tree program's external parameters for a
    batch (``inputs["parameters"]``): the points' values (``LoopSlice.
    _point_parameters``: the caller's and the request's fixed parameters),
    else the parameter's default. The renormalisation scale's parameter is
    formed natively from each point's mu_R^2 (``TreeProgram::bind``), so it
    is the one parameter the points may differ in."""

    record = trees.program["parameters"]
    scale = record["scale"]
    out = None
    for evaluation in evaluations:
        values = evaluation.parameters
        point = [
            "0" if index == scale else values.get(name, default)
            for index, (name, default) in enumerate(
                zip(record["names"], record["defaults"], strict=True)
            )
        ]
        if out is None:
            out = point
        elif point != out:
            raise ValueError("the points of one batch need equal model parameters")
    if out is None:
        raise ValueError("a batch needs points")
    return out
