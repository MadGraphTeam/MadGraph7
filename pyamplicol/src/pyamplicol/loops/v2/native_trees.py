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
  Born metric per pair of components gives |A0|^2 per squared order.
* **counterterms and external fields** (``_counterterm_program``): the
  counterterm closure's terms like the Born's, into ket vectors per
  (component, epsilon power, loop orders), and the field insertions, each
  interfered with the Born per sector (``loop_v2::interference``).

* **carriers**: every kernel and the parameter program also come as a
  binary64 SymJIT plane application (storage-v4 IR at -O2, the default
  ``jit_optimization_level``; ``_plane_applications``), compiled from the
  saved state's own binary64 evaluator and checked against it natively
  (``compile_v2_plane``); the native lane runs them in binary64, recompiled
  for the loading CPU, and touches no Symbolica state, so binary64 needs no
  licence (DESIGN §3.7); DoubleFloat and arbitrary precision run the exact
  states.

* **parameters**: one saved exact evaluator from the external model
  parameters to every parameter-dependent value of the native lane
  (``_parameter_program``): the tree program's derived model parameters (the
  model's own definitions), the Born, counterterm and field couplings per
  N_c grade (the slice's scalar plan, ``LoopSlice._scalar_plan``; the
  counterterm couplings per point, they read the scale) and the plan's
  kernel constants
  (``reference_executor.KernelConstants`` keys, with their mass flips),
  plus the exact mass parameters of the external spinors and the loop
  propagators. The plan's kernel constants carry the Born kernels' model
  factors.

Per batch only momenta, scales (mu_R^2 per point) and the external parameter
values cross (``batch_parameters``); the executor binds everything else
natively, the scale parameter from each point's mu_R^2.

The Born metric is per pair of Born components and squared Born order, each
weight exact (at leading and next-to-leading colour keeping the N_c powers the
projection keeps, ``_component_metric``). The exact carrier has no native
trees (wavefunctions are irrational); the slice's Python inputs remain its
path.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from pyamplicol.color.contraction_types import NC

from .families import Orders, add_orders

TREE_SCHEMA = 3

# Prune the exported tree program to what its outputs read
# (``prune_program``); False exports the unpruned program (the oracle of
# the pruning's bitwise equality, ``tests/unit/test_loop_v2_native_trees.py``).
PRUNE = True


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
    # The binary64 SymJIT plane application of each saved evaluator: the
    # kernels, then the parameter program (``_plane_applications``).
    planes: tuple[bytes, ...] = ()

    def json(self) -> str:
        return json.dumps(self.program, separators=(",", ":"))

    def executor_argument(self) -> tuple[str, list[bytes], list[bytes] | None]:
        if self.planes and len(self.planes) != len(self.kernels) + 1:
            raise ValueError("a plane application per saved evaluator")
        return (
            self.json(),
            [*self.kernels, self.parameters],
            list(self.planes) if self.planes else None,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.program["parameters"]["names"])


# The SymJIT optimisation level of the plane applications: always -O2, the
# default ``jit_optimization_level`` (``evaluators.symbolica_settings``) and a
# portable level (O0 and O3 SymJIT is target-specific), whatever the run's
# ``evaluator.jit.optimization_level`` says (docs/user/configuration.md).
_PLANE_OPTIMIZATION_LEVEL = 2
# Plane applications by saved state (SHA-256): the generation and staged
# validation of a run export the same kernels repeatedly.
_PLANES: OrderedDict[bytes, bytes] = OrderedDict()
_PLANES_KEPT = 4096


def _plane_applications(states: Sequence[bytes]) -> tuple[bytes, ...]:
    """The binary64 SymJIT plane application of each saved exact evaluator
    (storage-v4 IR; the native ``compile_v2_plane`` checks it against the
    evaluator and refuses a disagreement)."""

    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    out = []
    for state in states:
        key = hashlib.sha256(state).digest()
        plane = _PLANES.get(key)
        if plane is None:
            compiled, _inputs, _outputs = native.compile_v2_plane(
                bytes(state), _PLANE_OPTIMIZATION_LEVEL, False
            )
            plane = _PLANES[key] = bytes(compiled)
            while len(_PLANES) > _PLANES_KEPT:
                _PLANES.popitem(last=False)
        else:
            _PLANES.move_to_end(key)
        out.append(plane)
    return tuple(out)


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

    # The counterterm program (WP2.2, ``ct_closure``): a second interaction
    # program on a state of its own, which follows the trees' state in the
    # closure's index space (``loop_v2::trees``).
    ct_program = slice_.counterterm_program if with_counterterms else None
    ct = None if ct_program is None else _CountertermTrees.of(ct_program, state_size)

    def slot(block: Any, entry: Any) -> tuple[int, int]:
        """(state component start, dimension) of a block row; a block of the
        counterterm program reads its state, after the trees' state."""

        variant = "source" if block.is_source else "propagated"
        if ct is not None and id(block) in ct.blocks:
            found = ct.slots.get((entry.current_id, variant))
            if found is not None:
                found = (state_size + found[0], found[1])
        else:
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
        if ct is not None and id(block) in ct.blocks:
            by_helicity = {}
            for row in block.rows:
                by_helicity.setdefault(row.helicities, row)
        else:
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
    # The Born metric per pair of Born components, each weight exact (the
    # Gram cell times N_c to the components' powers, at N_c = 3): a colour-
    # orthogonal interference (u u~ > d d~'s QCD x QED order) cancels
    # exactly, as in ``assemble._interfere``, instead of leaving the rounding
    # of N_c^-1 folded into an amplitude.
    metric = (
        _component_metric(slice_, components, squared_orders) if born_squared else None
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
        ct,
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
            "metric": metric,
            "average": str(slice_.normalisation.average),
        },
        "parameters": parameters,
        "counterterms": None if counterterms is None else counterterms.record,
        "counterterm_trees": None if ct is None else ct.record,
        # Per closure-term layout its kernel inputs: ("current", slot,
        # component) or ("momentum", slot, mu); per colour weight (re, im,
        # sqrt(2) power p): the weight times 2^(-p/2).
        "layouts": [[list(item) for item in layout] for layout in layouts],
        "weights": [list(weight) for weight in weights],
    }
    ct_saved = () if ct is None else ct.saved
    if PRUNE:
        program, saved, ct_saved = prune_program(program, saved, ct_saved)
    # The kernels the pruned program still uses; their plane applications
    # are compiled after the pruning, so a dropped kernel compiles none.
    kernels = (*saved, *ct_saved)
    return NativeTrees(
        program,
        kernels,
        tuple(couplings),
        born_orders,
        squared_orders,
        masses,
        tuple(components),
        parameter_evaluator,
        () if counterterms is None else counterterms.outputs,
        _plane_applications((*kernels, parameter_evaluator)),
    )


@dataclass(frozen=True)
class _CountertermTrees:
    """The counterterm program of a slice (``ct_closure.CountertermProgram``,
    WP2.2) as the native lane's second interaction program.

    Its tree rows (``CountertermProgram.rows``) run like the trees', on a
    state of their own (sources, momentum slots, model parameters); its
    model parameters are the model's and the request's CT multiplicities and
    slot switches (``fixed``, every slot on). Its blocks' rows are read by
    the counterterm root terms at ``offset`` (the trees' state size) plus
    their start in its state.
    """

    rows: Any
    record: dict[str, Any]
    saved: tuple[bytes, ...]
    # (current id, variant) -> (state component start, dimension).
    slots: Mapping[tuple[int, str], tuple[int, int]]
    # ``id`` of every block of the program.
    blocks: frozenset[int]
    fixed: Mapping[str, int]

    @classmethod
    def of(cls, program: Any, offset: int) -> _CountertermTrees:
        rows = program.rows
        schema = rows.schema
        value_count = rows._value_count
        model_count = len(rows._records)
        slots: dict[tuple[int, str], tuple[int, int]] = {}
        for record in schema["value_storage"]["value_slots"]:
            key = (int(record["current_id"]), str(record["variant"]))
            slots.setdefault(
                key, (int(record["component_start"]), int(record["dimension"]))
            )
        momentum_slots = []
        for record in schema["momentum_slots"]:
            mask = 0
            for label in record["external_labels"]:
                mask |= 1 << (int(label) - 1)
            momentum_slots.append([value_count + int(record["component_start"]), mask])
        # Currents of a pole slot (an epsilon power other than 0): only pole
        # terms read them, so finite-only evaluations skip them.
        from pyamplicol.models.loading import counterterm_slot_order

        pole_orders = {
            counterterm_slot_order(*slot)
            for slot in program.recursion.slots
            if slot[1] != 0
        }
        pole_currents = frozenset(
            current.id
            for current in rows.dag.currents
            if any(
                power and name in pole_orders
                for name, power in current.index.coupling_orders
            )
        )
        interactions, saved = _interaction_program(rows, pole_currents)
        record = {
            "offset": offset,
            "state_size": max(rows._size, rows._offset + model_count),
            "model_offset": rows._offset,
            "model_count": model_count,
            "momentum_slots": momentum_slots,
            "source_fill": schema["source_fill"],
            "interactions": interactions,
        }
        return cls(
            rows,
            record,
            tuple(saved),
            slots,
            frozenset(id(block) for block in program.blocks.blocks.blocks),
            dict(program.recursion.parameter_values(poles=True)),
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
    tensors: dict[Any, int] = {tensor: b for b, tensor in enumerate(basis)}
    # The literal tensor of each ket tensor (``LoopSlice.overlaps`` reads
    # literal tensors).
    literal = [tensor.terms[0].tensor for tensor in basis]
    # Coupling slots: per key its index; per slot its value (``specs``) and
    # the epsilon power of the terms it weights (finite-only evaluations skip
    # the others).
    couplings: dict[tuple[Any, ...], int] = {}
    specs: list[tuple[Any, ...]] = []
    term_powers: list[int] = []

    def coupling_slot(key: tuple[Any, ...], spec: tuple[Any, ...], power: int) -> int:
        found = couplings.get(key)
        if found is None:
            found = couplings[key] = len(specs)
            specs.append(spec)
            term_powers.append(power)
        return found

    def scalar_slot(name: str, power: int, grade: int, term_power: int) -> int:
        """The slot of a coupling's (epsilon power, N_c grade) coefficient
        weighting terms of epsilon power ``term_power``."""

        spec = ("scalar", name, power, grade)
        return coupling_slot((*spec, term_power), spec, term_power)

    def whole_slot(term: Any, degree: int, expression: Any) -> int:
        """The slot of a whole-rule coefficient's N_c grade ``degree``."""

        parameters = tuple(term.parameters)
        key = ("whole", degree, expression.to_canonical_string(), parameters)
        return coupling_slot(key, ("whole", degree, expression, parameters), 0)

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

    # The closures share their closed colours (``SiteClosure._closed``), so
    # most terms repeat a (colour, multiplicity); the colour is kept with
    # its items so that its id stays its own.
    items_of: dict[tuple[int, int], tuple[Any, list[tuple[Any, ...]]]] = {}

    def colour_items(colour: Any, multiplicity: int) -> list[tuple[Any, ...]]:
        """(ket tensor, N_c power, weight re, weight im) per literal tensor."""

        found = items_of.get((id(colour), multiplicity))
        if found is not None:
            return found[1]
        items = [
            (
                tensor_index(item.tensor),
                item.nc_power,
                _fraction_text(item.coefficient.real * multiplicity),
                _fraction_text(item.coefficient.imag * multiplicity),
            )
            for item in colour.terms
        ]
        items_of[(id(colour), multiplicity)] = (colour, items)
        return items

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
                    slot = whole_slot(term, degree, expression)
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
                    slot = scalar_slot(name, power, grade, power)
                    contribute(contributions, slot, vector, grade, items, power2)
            if contributions:
                terms.append(closure_term(term, contributions))
    program = slice_.counterterm_program
    for item in () if program is None else program.closure.terms:
        # A root term of the counterterm recursion (``ct_closure.ClosureTerm``)
        # on the counterterm program's blocks: into the vector of its
        # (component, epsilon power, orders), with its multiplicity and its
        # coupling's N_c grades shifted by the inserted current's.
        term = item.term
        contributions = []
        items = colour_items(term.colour, item.multiplicity)
        power2 = term.colour.normalization_power
        vector = vectors.setdefault(
            (item.component, item.power, item.orders), len(vectors)
        )
        if isinstance(term, WholeRuleTerm):
            if item.scale is not None or item.power != 0:
                raise NotImplementedError("a scaled or pole whole-rule CT root term")
            for degree, expression in term.grades:
                slot = whole_slot(term, degree, expression)
                contribute(
                    contributions, slot, vector, degree + item.grade, items, power2
                )
            if contributions:
                terms.append(whole_term(term, contributions))
            continue
        if item.scale is not None:
            # A Weyl line's kappa (no loop_sm process has one): the writer
            # then keeps the plan alone and loads regenerate the slice.
            raise NotImplementedError(
                "native counterterm root terms with a Weyl-line factor"
            )
        name, power = item.coupling
        for grade, _column in layout.get(item.coupling, ()):
            slot = scalar_slot(name, power, grade, item.power)
            contribute(contributions, slot, vector, grade + item.grade, items, power2)
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
                    slot = scalar_slot(name, power, grade, power)
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
        "couplings": len(specs),
        "coupling_powers": term_powers,
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
        tuple(specs),
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
    counterterm_trees: _CountertermTrees | None = None,
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

    from symbolica import E, Expression, FunctionDefinition, Replacement

    from pyamplicol._internal.physics.symbols import symbols

    from .assemble import LAURENT_POWERS, _canonical_stage

    records = trees._records
    names = [str(r["name"]) for r in records if r["kind"] == "external_parameter"]
    defaults = [
        _fraction_text(Fraction(repr(r["default"])))
        for r in records
        if r["kind"] == "external_parameter"
    ]
    inputs = [
        symbols.qualified(f"pyamplicol_v2_parameters::e{i}") for i in range(len(names))
    ]
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
    # The counterterm program's model parameters: the model's, plus the CT
    # multiplicities and slot switches of the request (integers fixed per
    # process; every slot on, finite-only evaluations skip the pole terms).
    counterterm_model = []
    if counterterm_trees is not None:
        ct_rows = counterterm_trees.rows
        fixed = counterterm_trees.fixed
        ct_symbols = symbols.model(ct_rows.compiled.name)
        ct_externals = []
        for record in ct_rows._records:
            if record["kind"] != "external_parameter":
                continue
            name = str(record["name"])
            symbol = E(ct_symbols.symbol(name).to_canonical_string())
            if name in symbol_of:
                ct_externals.append(Replacement(symbol, symbol_of[name]))
            elif name in fixed:
                ct_externals.append(Replacement(symbol, Expression.num(fixed[name])))
            else:
                raise NotImplementedError(
                    f"the counterterm program reads {name!r}, which the trees have not"
                )
        for record in ct_rows._records:
            kind = record["kind"]
            if kind == "external_parameter":
                name = str(record["name"])
                if name in index:
                    counterterm_model.append(["input", index[name], False])
                    continue
                o = output(("ct-fixed", name), lambda v=fixed[name]: Expression.num(v))
                counterterm_model.append(["output", o, False])
            elif kind == "derived_parameter_component":
                name = str(record["runtime_name"])
                o = output(
                    ("ct-model", name),
                    lambda name=name: E(ct_rows._definitions[name]).replace_multiple(
                        ct_externals
                    ),
                )
                counterterm_model.append(
                    ["output", o, record["complex_component"] == "imag"]
                )
            else:
                raise NotImplementedError(
                    f"a counterterm program model parameter of kind {kind!r}"
                )

    # Born couplings per N_c grade, from the slice's scalar plans (the Born's
    # and the others', their columns concatenated in the layout).
    plans, layout = slice_._scalar_plan(LAURENT_POWERS)
    located = [
        (p, column)
        for p, plan in enumerate(plans)
        for column in range(len(plan.expressions))
    ]
    plan_inputs = []
    for plan in plans:
        replacements = []
        for name, symbol in zip(plan.external_names, plan.input_symbols, strict=True):
            if name not in index:
                raise NotImplementedError(
                    f"the scalar plan reads {name!r}, which the tree program has not"
                )
            replacements.append(Replacement(symbol, symbol_of[name]))
        plan_inputs.append(replacements)

    def column_value(column: int, diagnostic: bool = False) -> Any:
        """A scalar-plan column (or its diagnostic) on the program's inputs."""

        p, local = located[column]
        expression = plans[p].expressions[local]
        value = expression.diagnostic if diagnostic else expression.value
        return value.replace_multiple(plan_inputs[p])

    coupling_outputs = []
    for name, grade in couplings:
        (column,) = [c for degree, c in layout[(name, 0)] if degree == grade]
        coupling_outputs.append(
            output(
                ("coupling", name, grade), lambda column=column: column_value(column)
            )
        )

    # Counterterm and field couplings per epsilon power and N_c grade; they
    # may read the renormalisation scale (one value per point).
    counterterm_outputs, finite = [], []
    used: set[int] = set()

    def scalar(name: str, power: int, grade: int) -> Any:
        """The scalar-plan coefficient on the program's inputs (and its
        diagnostic, checked finite)."""

        (column,) = [c for degree, c in layout[(name, power)] if degree == grade]
        p, local = located[column]
        used.add(p)
        if plans[p].expressions[local].diagnostic != 0:
            finite.append(
                output(
                    ("diagnostic", name, power, grade),
                    lambda column=column: column_value(column, diagnostic=True),
                )
            )
        return column_value(column)

    every_input = [item for replacements in plan_inputs for item in replacements]
    for slot in counterterm_couplings:
        if slot[0] == "scalar":
            _kind, name, power, grade = slot
            value = scalar(name, power, grade)
            counterterm_outputs.append(
                output(("counterterm", name, power, grade), lambda v=value: v)
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
                lambda v=bound: v.replace_multiple(every_input),
            )
        )
    # The scalar library's helper functions (e.g. ``cond``, ``reglog``) the
    # counterterm and field couplings call; every plan shares the library.
    helpers: dict[Any, Any] = {}
    for p in sorted(used):
        for head, body in plans[p].program.library._bodies.items():
            if helpers.setdefault(head, body) != body:
                raise ValueError(f"the scalar plans define {head!r} differently")

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
    # The helpers in the order of their names, not the library's (which
    # follows the model's function bindings: a model compiled from its UFO
    # source and the same model read from the cache list them in different
    # orders), so their canonical names and the saved state do not depend on
    # how the model was obtained.
    canonical, parameters, functions = _canonical_stage(
        SimpleNamespace(
            output_expressions=outputs,
            parameter_symbols=inputs,
            symbolica_functions=[
                (head, tuple(arguments), body)
                for head, (arguments, body) in sorted(
                    helpers.items(), key=lambda item: item[0].to_canonical_string()
                )
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
        "counterterm_model": counterterm_model,
        "couplings": coupling_outputs,
        "counterterm_couplings": counterterm_outputs,
        "finite": finite,
        "constants": constants,
        "positive": sorted(set(positive)),
        "masses": [[int(pdg), index[name]] for pdg, name in masses],
        "plan_masses": plan_masses,
    }
    return record, bytes(evaluator.save())


def _interaction_program(
    trees: Any, pole_currents: frozenset[int] = frozenset()
) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """The blocks DAG of ``trees`` (a ``TreeRows``) as an interaction program.

    Per runtime-schema stage, every interaction adds its exact colour weight
    times its vertex kernel, evaluated on its two input rows and momenta, to
    its result current; each output slot then copies the sum or applies the
    current's propagator kernel. Kernels are the model's own component
    expressions (``vertex_component_expression``,
    ``propagator_component_expression``, as the stage program sums them per
    current) on formal inputs ``[left, right, p_left, p_right, model]`` and
    ``[value, p, model]``, where ``model`` is the model parameters the
    kernel's expressions hold (their indices into the state's model block,
    at ``model_offset``, are the third entry of the kernel's shape), one
    per distinct shape, saved as exact Symbolica evaluators. The program's
    size is that of the DAG; the saved evaluators come first for the
    kernels, then for the propagators.

    ``pole_currents`` are currents only epsilon-pole terms read (a
    counterterm program's pole slots): per step the indices of the
    interactions and outputs of such currents (``pole_interactions``,
    ``pole_outputs``), which finite-only evaluations skip."""

    from symbolica import Expression, FunctionDefinition

    from pyamplicol._internal.physics.symbols import symbols as registry
    from pyamplicol.generation.stage_parameters import (
        _logical_model_parameter_symbols,
        _model_symbolica_functions,
        _specialize_stage_symbolica_functions,
    )
    from pyamplicol.models._physics_ir import PropagatorIR

    from .assemble import _canonical_stage

    dag, schema = trees.dag, trees.schema
    records = trees._records
    model_symbols = [
        registry.qualified(f"pyamplicol_v2_trees::m{i}") for i in range(len(records))
    ]
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
        return [
            registry.qualified(f"pyamplicol_v2_trees::{name}{i}") for i in range(count)
        ]

    def compile_kernel(kind: str, key: tuple, build) -> int:
        found = ids.get((kind, key))
        if found is not None:
            return found
        inputs, components = build()
        # The model's kernel functions materialise with their first use.
        outputs, retained = _specialize_stage_symbolica_functions(
            list(components), _model_symbolica_functions(model)
        )
        # The kernel reads the model parameters its expressions hold, in the
        # model block's order: a counterterm program's model block holds every
        # coupling, multiplicity and slot switch of the request (160-190 at
        # six legs), a kernel a few, and each kernel row copied all of them.
        held = {
            symbol.to_canonical_string()
            for expression in (*outputs, *(body for _f, _a, body in retained))
            for symbol in expression.get_all_symbols()
        }
        model_inputs = [
            k
            for k, symbol in enumerate(model_symbols)
            if symbol.to_canonical_string() in held
        ]
        # On canonical symbols the kernel's term order, and so its binary64
        # rounding, does not depend on the symbols created earlier in the
        # interpreter (``assemble._canonical_stage``).
        outputs, parameters, functions = _canonical_stage(
            SimpleNamespace(
                output_expressions=list(outputs),
                parameter_symbols=[
                    *inputs,
                    *(model_symbols[k] for k in model_inputs),
                ],
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
        shapes[kind].append(
            [len(inputs) + len(model_inputs), len(outputs), model_inputs]
        )
        return found

    # Per current: (particle id, chirality, dimension) and its momentum
    # start; per stored colour weight: its exact (real, imaginary) texts. A
    # program repeats them for most of its interactions (g g > g g g g's
    # counterterm program: 0.5 M interactions, a handful of weights).
    current_keys: dict[int, tuple[int, int, int]] = {}
    momenta: dict[int, int] = {}
    weight_texts: dict[tuple[float, float], tuple[str, str]] = {}

    def current_key(current_id: int) -> tuple[int, int, int]:
        found = current_keys.get(current_id)
        if found is None:
            current = dag.currents[current_id]
            found = current_keys[current_id] = (
                int(current.index.particle_id),
                int(current.index.chirality),
                int(current.dimension),
            )
        return found

    def momentum_of(current_id: int) -> int:
        found = momenta.get(current_id)
        if found is None:
            mask = dag.currents[current_id].index.momentum_mask
            found = momenta[current_id] = momentum_start[mask]
        return found

    def weight_text(color_weight: tuple[float, float]) -> tuple[str, str]:
        found = weight_texts.get(color_weight)
        if found is None:
            weight = _exact_weight(complex(*color_weight))
            found = (_fraction_text(weight.real), _fraction_text(weight.imag))
            # Equal binary64 pairs (0.0 and -0.0 included) give equal texts.
            weight_texts[color_weight] = found
        return found

    def vertex(interaction: Any) -> int:
        left_key = current_key(interaction.left_id)
        right_key = current_key(interaction.right_id)
        result_key = current_key(interaction.result_id)
        key = (
            int(interaction.vertex_kind),
            result_key[0],
            result_key[1],
            left_key[1],
            right_key[1],
            left_key[2],
            right_key[2],
        )
        found = ids.get(("kernels", key))
        if found is not None:
            return found
        left = dag.currents[interaction.left_id]
        right = dag.currents[interaction.right_id]

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
        pole_rows = []
        for item in stage["interaction_ids"]:
            interaction = dag.interactions[int(item)]
            if complex(*interaction.evaluation_factor) == 0j:
                continue  # a certified zero attachment (as the stage program)
            if int(interaction.result_id) in pole_currents:
                pole_rows.append(len(rows))
            left, right = inputs[interaction.left_id], inputs[interaction.right_id]
            real, imag = weight_text(tuple(interaction.color_weight))
            rows.append(
                [
                    vertex(interaction),
                    int(left["component_start"]),
                    int(left["dimension"]),
                    int(right["component_start"]),
                    int(right["dimension"]),
                    momentum_of(interaction.left_id),
                    momentum_of(interaction.right_id),
                    accumulators[int(interaction.result_id)],
                    real,
                    imag,
                ]
            )
        outputs = []
        pole_outputs = []
        for slot_id in stage["output_value_slot_ids"]:
            slot = value_slots[int(slot_id)]
            current = int(slot["current_id"])
            if current in pole_currents:
                pole_outputs.append(len(outputs))
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
        step = {"interactions": rows, "accumulators": dimensions, "outputs": outputs}
        if pole_rows or pole_outputs:
            step["pole_interactions"] = pole_rows
            step["pole_outputs"] = pole_outputs
        steps.append(step)
    program = {
        "kernels": shapes["kernels"],
        "propagators": shapes["propagators"],
        "steps": steps,
    }
    return program, (*saved["kernels"], *saved["propagators"])


# ---------------------------------------------------------------- pruning


def prune_program(
    program: dict[str, Any],
    saved: Sequence[bytes],
    ct_saved: Sequence[bytes],
) -> tuple[dict[str, Any], tuple[bytes, ...], tuple[bytes, ...]]:
    """``program`` without the work no output of it reads; with the saved
    kernel evaluators ``saved`` (the trees') and ``ct_saved`` (the
    counterterm program's) of the kernels it still uses. ``program`` is
    pruned in place (``export_trees`` builds it for this call) and returned.

    1. **Counterterm ket components** no interference reads
       (``_live_ket``: at leading and next-to-leading colour every
       component whose N_c powers lie below the kept ones; at any accuracy
       a ket tensor without an overlap cell): their contributions leave the
       closure terms, and terms without contributions leave the closure.
    2. **Currents** no closure term (Born or counterterm, with their input
       layouts) and no plan block row reads, also through other currents:
       their interactions and output slots leave both interaction programs
       (``_prune_interactions``), then the kernels no interaction uses.

    Every value the executor returns is bitwise unchanged: a dropped ket
    component meets only zero metric cells (it adds ``0 * value``), every
    kept current sums the same interactions in the same order (kernel
    groups keep their order, kernel rows are independent), and the kept
    terms add their kept contributions in the same order. Only the work
    shrinks: at leading colour the counterterm program's N_c-graded slots
    that the projection drops (design §7.5).

    One linear pass over the program: a closure block's rows are read up to
    the widest slot of any term on it, marked once per block and Born row
    (not per term), and the interactions are filtered and renumbered
    without copies, so the pass costs a small fraction of the export
    (design §7.5, Table 7.5c)."""

    counterterms = program.get("counterterms")
    if counterterms is not None:
        _prune_counterterm_terms(counterterms)
    state_size = int(program["state_size"])
    ct = program.get("counterterm_trees")
    offset = state_size if ct is None else int(ct["offset"])
    tree_live = bytearray(state_size)
    ct_live = bytearray(0 if ct is None else int(ct["state_size"]))

    def read(start: int, length: int) -> None:
        if start < offset:
            _mark(tree_live, start, length)
        else:
            _mark(ct_live, start - offset, length)

    for start, size, _sign in program["block_rows"]:
        read(int(start), int(size))
    # Per closure block the widest prefix of its rows a term reads: per
    # layout the width of each slot (its largest current component + 1).
    born_blocks = program["born"]["blocks"]
    widths = []
    for layout in program["layouts"]:
        found: list[int] = []
        for kind, slot, component in layout:
            if kind != "current":
                continue
            if len(found) <= slot:
                found.extend([0] * (slot + 1 - len(found)))
            found[slot] = max(found[slot], int(component) + 1)
        widths.append(found)
    block_width = [0] * len(born_blocks)
    terms = program["born"]["terms"]
    if counterterms is not None:
        terms = [*terms, *counterterms["terms"]]
    for term in terms:
        blocks = term["blocks"]
        layout = widths[term["layout"]]
        if len(layout) > len(blocks):
            raise ValueError("a closure term's layout reads a slot it has no block for")
        for block, width in zip(blocks, layout, strict=False):
            if width > block_width[block]:
                block_width[block] = width
    for block, width in enumerate(block_width):
        if width:
            for start in born_blocks[block]:
                if start is not None:
                    read(start, width)
    tree_count = len(program["interactions"]["kernels"]) + len(
        program["interactions"]["propagators"]
    )
    if len(saved) != tree_count:
        raise ValueError("the trees' saved kernels and their program differ")
    saved = _prune_interactions(program["interactions"], tuple(saved), tree_live)
    if ct is not None:
        ct_saved = _prune_interactions(ct["interactions"], tuple(ct_saved), ct_live)
    return program, tuple(saved), tuple(ct_saved)


def _mark(live: bytearray, start: int, length: int) -> None:
    """Mark state components ``start .. start + length`` of ``live``."""

    if start < 0 or start + length > len(live):
        raise ValueError("a tree program read lies outside its state")
    live[start : start + length] = b"\x01" * length


class _Ones(dict[int, bytes]):
    """``b"\\x01" * n`` per length ``n``, built once."""

    def __missing__(self, length: int) -> bytes:
        found = self[length] = b"\x01" * length
        return found


def _kept_exponent(exponent: int, degree: int | None, depth: int | None) -> bool:
    """``loop_v2::interference::kept``: whether a cell of N_c exponent
    ``exponent`` survives the projection of a sector of degree ``degree``
    (``depth`` None: full colour; 0: LC; 2: NLC)."""

    if depth is None:
        return True
    return degree is not None and degree - depth <= exponent <= degree


def _live_ket(record: Mapping[str, Any]) -> list[bool]:
    """Per ket component (tensor ``t``, vector, N_c power ``q``) of a
    counterterm record whether an interference can read it: an entry of its
    vector (bra order, grade ``g``, degree ``P``), a Born component (basis
    element ``b``, power ``p``) of that order and an overlap cell
    ``<b|t>`` of power ``n`` that the projection keeps at ``n + p + q + g``
    (``loop_v2::interference``). Any other component meets only zero metric
    cells. Conservative: cells that cancel in a class count as kept."""

    bra: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for b, order, power in record["born"]:
        bra[int(order)].add((int(b), int(power)))
    cells: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for b, t, power, _real, _imag in record["cells"]:
        cells[int(t)][int(b)].add(int(power))
    entries: dict[int, set[tuple[int, int, int | None]]] = defaultdict(set)
    for kind, order, vector, _coupling, grade, _output, degree in record["entries"]:
        if kind == "vector":
            entries[int(vector)].add((int(order), int(grade), degree))
    depth = record["depth"]
    # Per (tensor, vector) the distinct (n + p + g, P) of its cells.
    exponents: dict[tuple[int, int], set[tuple[int, int | None]]] = {}
    live = []
    for t, vector, q in record["ket"]:
        key = (int(t), int(vector))
        found = exponents.get(key)
        if found is None:
            overlaps = cells.get(key[0], {})
            found = exponents[key] = {
                (n + p + grade, degree)
                for order, grade, degree in entries.get(key[1], ())
                for b, p in bra.get(order, ())
                for n in overlaps.get(b, ())
            }
        live.append(
            any(
                _kept_exponent(exponent + int(q), degree, depth)
                for exponent, degree in found
            )
        )
    return live


def _prune_counterterm_terms(record: dict[str, Any]) -> None:
    """Drop from ``record`` (in place) the ket components no interference
    reads (``_live_ket``), their contributions and the terms left without
    any; unchanged if every component is read."""

    live = _live_ket(record)
    if all(live) and all(term["contributions"] for term in record["terms"]):
        return
    index = [-1] * len(live)
    ket = []
    for k, (component, alive) in enumerate(zip(record["ket"], live, strict=True)):
        if alive:
            index[k] = len(ket)
            ket.append(component)
    terms = []
    for term in record["terms"]:
        contributions = [
            [coupling, index[component], weight]
            for coupling, component, weight in term["contributions"]
            if index[component] >= 0
        ]
        if contributions:
            term["contributions"] = contributions
            terms.append(term)
    record["ket"] = ket
    record["terms"] = terms


def _prune_interactions(
    interactions: dict[str, Any], saved: tuple[bytes, ...], live: bytearray
) -> tuple[bytes, ...]:
    """Drop from an interaction program (``_interaction_program``; in place)
    the outputs no state component marked in ``live`` (the program's
    readers' reads; marked further here) needs, directly or through the
    interactions of the outputs it needs; per step only the accumulators of
    kept outputs and their interactions stay, then only the kernels and
    propagators they use: their saved evaluators (``saved``: kernels, then
    propagators) are returned. Unchanged if a state component is the output
    of two slots (each is written once)."""

    steps = interactions["steps"]
    size = len(live)
    written = bytearray(size)
    for step in steps:
        for _accumulator, _propagator, _momentum, start, dimension in step["outputs"]:
            if written.find(1, start, start + dimension) >= 0:
                return saved
            _mark(written, start, dimension)
    # Backward: a step's outputs that a later reader reads, then the
    # interactions summed into their accumulators, whose inputs are read
    # (marked in place: a read past the state would grow ``live``).
    ones = _Ones()
    kept: list[tuple[list[int], list[int]]] = [([], [])] * len(steps)
    for index in reversed(range(len(steps))):
        step = steps[index]
        outputs = [
            k
            for k, (_a, _p, _m, start, dimension) in enumerate(step["outputs"])
            if live.find(1, start, start + dimension) >= 0
        ]
        wanted = bytearray(len(step["accumulators"]))
        for k in outputs:
            wanted[step["outputs"][k][0]] = 1
        interaction_rows = step["interactions"]
        rows = [k for k, row in enumerate(interaction_rows) if wanted[row[7]]]
        for k in rows:
            row = interaction_rows[k]
            live[row[1] : row[1] + row[2]] = ones[row[2]]
            live[row[3] : row[3] + row[4]] = ones[row[4]]
        kept[index] = (rows, outputs)
    if len(live) != size:
        raise ValueError("an interaction reads a current outside its state")
    vertex_kernels = len(interactions["kernels"])
    used = sorted(
        {
            row[0]
            for step, (rows, _outputs) in zip(steps, kept, strict=True)
            for row in (step["interactions"][k] for k in rows)
        }
    )
    propagators = sorted(
        {
            output[1]
            for step, (_rows, outputs) in zip(steps, kept, strict=True)
            for output in (step["outputs"][k] for k in outputs)
            if output[1] >= 0
        }
    )
    kernel_of = [-1] * vertex_kernels
    for new, old in enumerate(used):
        kernel_of[old] = new
    propagator_of = [-1] * len(interactions["propagators"])
    for new, old in enumerate(propagators):
        propagator_of[old] = new
    for step, (rows, outputs) in zip(steps, kept, strict=True):
        accumulators = sorted({step["outputs"][k][0] for k in outputs})
        accumulator_of = [-1] * len(step["accumulators"])
        for new, old in enumerate(accumulators):
            accumulator_of[old] = new
        pole_rows = set(step.get("pole_interactions", ()))
        pole_outputs = set(step.get("pole_outputs", ()))
        new_rows = []
        for k in rows:
            row = step["interactions"][k]
            row[0] = kernel_of[row[0]]
            row[7] = accumulator_of[row[7]]
            new_rows.append(row)
        new_outputs = []
        for k in outputs:
            output = step["outputs"][k]
            output[0] = accumulator_of[output[0]]
            if output[1] >= 0:
                output[1] = propagator_of[output[1]]
            new_outputs.append(output)
        new_pole_rows = [i for i, k in enumerate(rows) if k in pole_rows]
        new_pole_outputs = [i for i, k in enumerate(outputs) if k in pole_outputs]
        step["interactions"] = new_rows
        step["accumulators"] = [step["accumulators"][a] for a in accumulators]
        step["outputs"] = new_outputs
        if new_pole_rows or new_pole_outputs:
            step["pole_interactions"] = new_pole_rows
            step["pole_outputs"] = new_pole_outputs
        else:
            step.pop("pole_interactions", None)
            step.pop("pole_outputs", None)
    interactions["kernels"] = [interactions["kernels"][k] for k in used]
    interactions["propagators"] = [interactions["propagators"][p] for p in propagators]
    return (
        *(saved[k] for k in used),
        *(saved[vertex_kernels + p] for p in propagators),
    )


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
    """The Born colour metric over Born components (b, Born order, N_c
    power), per squared Born order: <b|b'> keeping the N_c powers the
    projection keeps (``colour.kept``: every power at full colour) below the
    order's leading degree, at N_c = 3, as the Python inputs' Born
    interference (``assemble._interfere``)."""

    from .colour import kept

    accuracy = slice_.request.accuracy
    degrees = slice_.degrees()["born"] if accuracy != "full" else {}
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
