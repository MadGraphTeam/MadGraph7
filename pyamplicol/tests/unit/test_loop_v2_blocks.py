# SPDX-License-Identifier: 0BSD
"""Tree blocks of loop requests (design §3.2, WP1.2).

FeynKit appears here only as the oracle: full-colour amputated tree sums and
one-loop diagram attachments. The production blocks come from the tree engine
in loop-request mode.
"""

from __future__ import annotations

import json
import math
import string
from collections import Counter
from decimal import Decimal

import numpy as np
import pytest

from pyamplicol.generation.dag_compiler import (
    AttachableLoopBlocks,
    GenericDAGCompiler,
)
from pyamplicol.loops.v2.blocks import (
    RequestedLoopBlocks,
    block_colour_flow,
    build_tree_blocks,
    requested_loop_blocks,
)
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelBlockTable,
    ModelLoopCatalogue,
    add_orders,
    generate_families,
)

from .loop_v2_stage_rows import stage_rows

QCD = {"QCD": 99, "QED": 0}
POINT_SEEDS = (7, 23)
# FeynKit evaluates its tensors and the model couplings in binary64.
TOLERANCE = 1e-12
# The tree engine's Weyl spinors of chirality +1 / -1 are the Spenso Dirac
# components 2,3 / 0,1 of the parity-mirrored kinematics; FeynKit therefore
# evaluates the mirrored process (mirrored momenta and vectors).
MIRROR = np.array([1.0, -1.0, -1.0, -1.0])


@pytest.fixture(scope="module")
def models(loop_sm_compiled_model):
    """(compiled loop_sm, tree-engine model, prepared native loop structure)."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    structure = prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 5, "QED": 2},
        loop_order_bounds={"QCD": 7, "QED": 2},
    ).structure
    return compiled, CompiledUFOModel(compiled), structure


def _process(compiled, text, accuracy="full"):
    from pyamplicol.processes.model import build_model_process_ir

    return build_model_process_ir(text, compiled.ir, color_accuracy=accuracy)


def _point(dag, model, seed):
    from pyamplicol.generation.validation import build_validation_point

    point = build_validation_point(dag, model, process_id="loop-v2-blocks", seed=seed)
    assert point.available, point.error
    return point.four_vectors


def _current_values(dag, model, point, precision=64, parameters=None):
    """Every value slot of every current, by (current id, variant).

    The shared interpreted stage evaluation of the tree runtime, as in the
    numerical-current warm-up; nothing is compiled. ``parameters`` overrides
    runtime model parameters (default: their binary64 defaults) by name.
    """

    overrides = parameters or {}

    def values(records):
        return [
            Decimal(overrides[record["name"]])
            if record["name"] in overrides
            else Decimal.from_float(float(record["default"]))
            for record in records
        ]

    momenta = [[Decimal.from_float(float(x)) for x in vector] for vector in point]
    rows = stage_rows(dag, model, momenta, values, precision)
    return {
        key: np.array([complex(float(re), float(im)) for re, im in row])
        for key, row in rows.items()
    }


# ----------------------------------------------------------------- FeynKit oracle


def _orient_spinor_chains(expression, starts):
    """Orient every spinor matrix along its chain, row first.

    Spenso contracts spinor indices by position, while FeynKit writes the
    matrices of a chain (gamma matrices, chiral projectors) in no particular
    orientation. Each chain is walked from a start slot and every matrix
    reached by its second spinor index is transposed back, so that a dense
    evaluation reads the chain row to column.
    """

    from symbolica import AtomType, E, S

    expression = expression.expand()
    terms = list(expression) if expression.get_type() == AtomType.Add else [expression]
    result = E("0")
    for term in terms:
        factors = list(term) if term.get_type() == AtomType.Mul else [term]
        gammas = {
            index: list(factor)
            for index, factor in enumerate(factors)
            if factor.get_type() == AtomType.Fn
            and len(list(factor)) >= 2
            and all(str(argument).startswith("bis(") for argument in list(factor)[:2])
        }
        used: set[int] = set()
        for start in starts:
            current = start
            while True:
                found = None
                for index, arguments in gammas.items():
                    if index in used:
                        continue
                    if str(arguments[1]) == current:
                        arguments[0], arguments[1] = arguments[1], arguments[0]
                    if str(arguments[0]) == current:
                        found = index
                        break
                if found is None:
                    break
                used.add(found)
                current = str(gammas[found][1])
        rebuilt = E("1")
        for index, factor in enumerate(factors):
            rebuilt *= (
                S(factor.get_name())(*gammas[index]) if index in gammas else factor
            )
        result += rebuilt
    return result


class _FeynKitTrees:
    """Amputated full-colour tree sums, all legs outgoing, one leg left open."""

    def __init__(self, native, model):
        from symbolica import E
        from symbolica.community import tensor as t

        self.native = native
        self.model = model
        self._expressions = {}
        library = t.TensorLibrary.hep_lib_atom()
        self.generators = self.numeric(
            t.TensorExpression(
                E(
                    "spenso::t(spenso::coad(8,x::a),spenso::cof(3,x::i),"
                    "spenso::dind(spenso::cof(3,x::j)))"
                )
            )
            .to_tensor(library)
            .to_dense()
        )

    def numeric(self, tensor):
        return tensor.map_components(
            lambda value: (
                complex(value.evaluate({}))
                if hasattr(value, "evaluate")
                else complex(value)
            ),
            dtype=complex,
        ).to_numpy()

    def amputated(self, names, orders, starts, momenta, states):
        """Dense tensor over the open slots with (leg, 'colour'|'spin') keys.

        The last leg stays open. The contraction is done once per process
        with symbolic momenta and external states, then evaluated per point
        and helicity.
        """

        key = (tuple(names), orders, tuple(starts))
        if key not in self._expressions:
            self._expressions[key] = self._contract(names, orders, starts)
        cached = self._expressions[key]
        if cached is None:
            return None, None
        evaluator, symbols, keys, couplings = cached
        values = dict(couplings)
        for index, momentum in enumerate(momenta):
            for axis, component in enumerate(momentum):
                values[symbols[("p", index, axis)]] = float(component)
        for index, state in enumerate(states):
            for axis, component in enumerate(state if state is not None else ()):
                values[symbols[("w", index, axis)]] = complex(component)
        inputs = np.array([[values[symbol] for symbol in evaluator.parameters]])
        return evaluator.evaluate_complex(inputs)[0].to_numpy(), keys

    def _contract(self, names, orders, starts):
        from symbolica import S
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        generated = self.native.process([], list(names)).generate_diagrams(
            loops=0,
            coupling_orders={name: (power, power) for name, power in orders},
            progress=None,
        )
        assert generated.report.completed
        if not generated.diagrams:
            return None
        amplitude = h.Amplitude(generated.diagrams)
        expression = amplitude.expression()
        start_slots = [
            str(slot.to_expression())
            for index, leg in enumerate(amplitude.legs)
            if index in starts
            for slot in leg.slots
            if str(slot).startswith("bis")
        ]
        if start_slots:
            expression = t.TensorExpression(
                _orient_spinor_chains(expression.to_expression(), start_slots)
            )
        # The structural model writes complexconjugate(p) as a scalar-program
        # function; its arguments here are real parameters (checked below).
        raw = expression.to_expression()
        conjugates = [
            symbol
            for symbol in raw.get_all_symbols(True)
            if symbol.get_name().startswith("pyamplicol_scalar_model")
        ]
        argument = S("blocks_oracle::argument_")
        for symbol in conjugates:
            raw = raw.replace(symbol(argument), argument)
        expression = t.TensorExpression(raw)
        library = t.TensorLibrary.hep_lib_atom()
        symbols = {}
        for index, leg in enumerate(amplitude.legs):
            components = [S(f"blocks_oracle::p{index}_{axis}") for axis in range(4)]
            symbols.update({("p", index, a): c for a, c in enumerate(components)})
            library.register(
                t.Tensor.dense(
                    t.TensorName("gammalooprs::P")(index, t.Representation.mink(4)),
                    components,
                )
            )
            if index == len(amplitude.legs) - 1:
                continue
            for slot in leg.slots:
                if not str(slot).startswith(("mink", "bis")):
                    continue
                components = [S(f"blocks_oracle::w{index}_{axis}") for axis in range(4)]
                symbols.update({("w", index, a): c for a, c in enumerate(components)})
                name = t.TensorName(f"blocks_oracle::state{index}")
                library.register(
                    t.Tensor.dense(name(slot.dual().representation), components)
                )
                expression = expression * t.TensorExpression(name(slot.dual()))
        tensor = expression.to_tensor(library).to_dense()
        keys = []
        for slot in tensor.axes:
            leg = int(str(slot).split("hedge(")[1].split(",")[0])
            keys.append(
                (leg, "spin" if str(slot).startswith(("mink", "bis")) else "colour")
            )
        # Model parameters at the tree runtime's defaults, in binary64.
        couplings = {
            symbol: complex(self.model._parameter_default(symbol.get_name()[5:]))
            for symbol in expression.to_expression().get_all_symbols(False)
            if symbol.get_name().startswith("UFO::")
        }
        assert not conjugates or all(value.imag == 0 for value in couplings.values())
        parameters = [*couplings, *symbols.values()]
        evaluator = tensor.evaluator(parameters, n_cores=1, jit_compile=False)
        return evaluator, symbols, keys, couplings


def _colour_tensor(word, x_representation, axes, representations, generators):
    """Full-colour factor of a block word over ``axes`` (labels and 'X').

    A string ``s -> a_1 ... a_k -> e`` contributes
    ``(tau^{a_k} ... tau^{a_1})_{e s}``; an adjoint current closes the open
    ends with ``tau^X_{in, out}`` (blocks module docstring), ``tau = sqrt(2) t``.
    """

    flow = block_colour_flow(word, representations, x_representation)
    letters = iter(string.ascii_letters)
    axis_letter = {axis: next(letters) for axis in axes}
    open_in = axis_letter["X"] if x_representation == -3 else next(letters)
    open_out = axis_letter["X"] if x_representation == 3 else next(letters)
    operands, subscripts = [], []
    for labels in flow.strings:
        start = axis_letter[labels[0]] if representations[labels[0]] == 3 else open_in
        end = axis_letter[labels[-1]] if representations[labels[-1]] == -3 else open_out
        adjoints = [label for label in labels if representations[label] == 8]
        if not adjoints:
            operands.append(np.eye(3))
            subscripts.append(end + start)
            continue
        row = end
        for position, label in enumerate(reversed(adjoints)):
            column = start if position == len(adjoints) - 1 else next(letters)
            operands.append(generators)
            subscripts.append(axis_letter[label] + row + column)
            row = column
    if x_representation == 8:
        operands.append(generators)
        subscripts.append(axis_letter["X"] + open_in + open_out)
    output = "".join(axis_letter[axis] for axis in axes)
    tensor = np.einsum(",".join(subscripts) + "->" + output, *operands)
    adjoint_count = sum(1 for label in word if representations[label] == 8)
    return tensor * math.sqrt(2) ** (adjoint_count + (x_representation == 8))


def _feynkit_ordering_sign(pdgs, model):
    """FeynKit's external-ordering sign of the identity fermion pairing.

    FeynKit signs an amplitude by the parity of its fermion lines listed as
    (antiparticle leg, particle leg) after crossing every leg incoming,
    relative to its leg order (``feynkit-generator`` ``generation.rs``,
    ``external_ordering_negative``). Here every leg is outgoing, so the
    antiparticle legs are the outgoing particles. Against the tree engine's
    ordering every pairing differs by the parity of its permutation, which the
    block signs carry; what remains is the identity pairing's parity.
    """

    fermions = [index for index, pdg in enumerate(pdgs) if model.is_fermion(pdg)]
    antiparticles = [index for index in fermions if pdgs[index] > 0]
    particles = [index for index in fermions if pdgs[index] < 0]
    sequence = [
        leg for pair in zip(antiparticles, particles, strict=True) for leg in pair
    ]
    inversions = sum(
        sequence[i] > sequence[j]
        for i in range(len(sequence))
        for j in range(i + 1, len(sequence))
    )
    return -1 if inversions % 2 else 1


def _assert_blocks_equal_feynkit(models, text, seeds, orders, min_legs=2):
    compiled, model, structure = models
    native = structure.native
    process = _process(compiled, text)
    blocks = build_tree_blocks(
        process,
        model,
        born_coupling_orders=orders,
        block_coupling_orders=orders,
    )
    dag = blocks.blocks_dag
    legs = {leg.label: leg for leg in process.legs}
    representations = {
        label: int(model.color_rep(leg.outgoing_pdg)) for label, leg in legs.items()
    }
    pdgs = {label: int(leg.outgoing_pdg) for label, leg in legs.items()}
    oracle = _FeynKitTrees(native, model)
    groups = {}
    for block in blocks.blocks:
        if len(block.key.word) < min_legs or block.key.auxiliary_kind is not None:
            continue
        group = (
            block.key.external_mask,
            block.key.particle_id,
            block.key.coupling_orders,
            block.key.chirality,
        )
        groups.setdefault(group, []).append(block)
    checked = 0
    for seed in seeds:
        point = _point(dag, model, seed)
        # FeynKit's structural model has width-free propagators.
        values = _current_values(dag, model, point, parameters={"WZ": 0, "WW": 0})
        momenta = {
            label: np.array(
                [-float(x) if legs[label].is_initial else float(x) for x in vector]
            )
            for label, vector in zip(sorted(legs), point, strict=True)
        }
        sources = {
            row.helicities[0]: (block.key.chirality, values[(row.current_id, "source")])
            for block in blocks.blocks
            if block.is_source
            for row in block.rows
        }
        for (_mask, particle_id, orders, chirality), members in groups.items():
            labels = members[0].key.labels
            x_representation = int(model.color_rep(particle_id))
            x_bar = model.particles[particle_id].anti_pdg

            names = [native.particle_by_pdg(pdgs[label]).name for label in labels] + [
                native.particle_by_pdg(x_bar).name
            ]
            # Spinor chains are read from the outgoing quarks, including the
            # open end of an antiquark current.
            starts = [
                index
                for index, pdg in enumerate([*(pdgs[label] for label in labels), x_bar])
                if model.is_fermion(pdg) and model.color_rep(pdg) == 3
            ]
            total = sum(momenta[label] for label in labels)
            # FeynKit signs every diagram by its external fermion ordering;
            # what the block signs leave over is the parity of its identity
            # pairing, which the export carries as ``pairing_sign``.
            feynkit_sign = _feynkit_ordering_sign(
                [*(pdgs[label] for label in labels), x_bar], model
            )
            assert {block.pairing_sign for block in members} == {feynkit_sign}
            comparisons = []
            for helicities in sorted(
                {row.helicities for block in members for row in block.rows}
            ):
                states = []
                for leg_helicity in helicities:
                    source_chirality, wave = sources[leg_helicity]
                    if len(wave) == 2:
                        wave = (
                            np.concatenate([[0, 0], wave])
                            if source_chirality == 1
                            else np.concatenate([wave, [0, 0]])
                        )
                    elif len(wave) == 4:
                        wave = wave * MIRROR
                    states.append(wave)
                fk, keys = oracle.amputated(
                    names,
                    orders,
                    starts,
                    [momenta[label] * MIRROR for label in labels] + [-total * MIRROR],
                    [*states, None],
                )
                assert fk is not None, (labels, particle_id, orders)
                axes = [
                    ("X" if kind == "colour" else "spin")
                    if leg == len(labels)
                    else labels[leg]
                    for leg, kind in keys
                ]
                colour_axes = [axis for axis in axes if axis != "spin"]
                fk = np.transpose(
                    fk,
                    [axes.index(axis) for axis in colour_axes] + [axes.index("spin")],
                )
                rows = [
                    (block, row)
                    for block in members
                    for row in block.rows
                    if row.helicities == helicities
                ]
                variant = (
                    "propagated"
                    if (rows[0][1].current_id, "propagated") in values
                    else "unpropagated"
                )
                if (
                    fk.shape[-1] == 4
                    and values[(rows[0][1].current_id, variant)].size == 2
                ):
                    # A Weyl current's vertex output is the opposite chirality
                    # block of the Dirac spinor.
                    fk = fk[..., 0:2] if chirality == 1 else fk[..., 2:4]
                elif fk.shape[-1] == 4:
                    fk = fk * MIRROR
                if variant == "propagated" and not model.is_fermion(particle_id):
                    # Width-free vector propagator; the P^mu P^nu / M^2 term
                    # of the unitary gauge drops on the conserved current.
                    mass = float(model.mass(particle_id))
                    fk = -1j * fk / (total[0] ** 2 - total[1:] @ total[1:] - mass**2)
                elif variant == "propagated":
                    flat = fk.reshape(-1, fk.shape[-1])
                    fk = np.array(
                        [
                            [
                                complex(value)
                                for value in model.propagator_component_expression(
                                    particle_id,
                                    list(map(complex, row)),
                                    list(map(float, total)),
                                    chirality=chirality,
                                )
                            ]
                            for row in flat
                        ]
                    ).reshape(fk.shape)
                predicted = sum(
                    block.wick_sign
                    * np.multiply.outer(
                        _colour_tensor(
                            block.key.word,
                            x_representation,
                            colour_axes,
                            representations,
                            oracle.generators,
                        ),
                        values[(row.current_id, variant)],
                    )
                    for block, row in rows
                )
                comparisons.append((fk, predicted))
            scale = max(np.abs(fk).max() for fk, _predicted in comparisons)
            assert scale > 0, (labels, particle_id, orders)
            error = max(np.abs(fk - mine).max() for fk, mine in comparisons)
            assert error <= TOLERANCE * scale, (labels, particle_id, orders, error)
            checked += len(comparisons)
    return checked


def _loop_attachments(native, model, process, orders):
    """(leg mask, particle id, coupling orders) of every tree a FeynKit
    one-loop diagram attaches to its loop (oracle only).

    The particle id is the tree engine's: the field the tree emits into the
    loop vertex, all outgoing; an edge entering the loop vertex carries its
    conjugate.

    Trees on all legs only feed one-attachment (VEV) tadpoles, which the loop
    selection rule never generates; they are left out.
    """

    from pyamplicol.loops.graphs import _rule_orders

    rule_orders = _rule_orders(native)
    incoming = [
        native.particle_by_pdg(model.particles[int(leg.outgoing_pdg)].anti_pdg).name
        for leg in process.legs
        if leg.is_initial
    ]
    outgoing = [
        native.particle_by_pdg(int(leg.outgoing_pdg)).name
        for leg in process.legs
        if not leg.is_initial
    ]
    generated = native.process(incoming, outgoing).generate_diagrams(
        loops=1,
        coupling_orders={name: (power, power) for name, power in orders.items()},
        allow_self_loops=True,
        allow_zero_flow_edges=True,
        maximum_bridges=None,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        progress=None,
    )
    assert generated.report.completed and generated.diagrams
    attachments = set()
    for diagram in generated.diagrams:
        cycles, _forest = diagram.cycle_basis()
        loop_edges = {edge.id for edge in cycles[0].edges}
        loop_vertices = {vertex.id for vertex in cycles[0].vertices}
        adjacent: dict[int, list] = {}
        for edge in diagram.edges:
            for end in (edge.source, edge.target):
                if end is not None:
                    adjacent.setdefault(end, []).append(edge)
        orders_by_vertex = {
            vertex.id: rule_orders[vertex.interaction] for vertex in diagram.vertices
        }
        for edge in diagram.edges:
            if edge.id in loop_edges:
                continue
            ends = [end for end in (edge.source, edge.target) if end is not None]
            if not any(end in loop_vertices for end in ends):
                continue
            mask, tree_orders = 0, ()
            if edge.is_external:
                mask = 1 << edge.external_index
            else:
                (root,) = [end for end in ends if end not in loop_vertices]
                seen_vertices, seen_edges, stack = {root}, {edge.id}, [root]
                while stack:
                    vertex = stack.pop()
                    tree_orders = add_orders(tree_orders, orders_by_vertex[vertex])
                    for other in adjacent[vertex]:
                        if other.id in seen_edges:
                            continue
                        seen_edges.add(other.id)
                        if other.is_external:
                            mask |= 1 << other.external_index
                            continue
                        (nxt,) = [
                            end for end in (other.source, other.target) if end != vertex
                        ] or [vertex]
                        if nxt not in seen_vertices:
                            seen_vertices.add(nxt)
                            stack.append(nxt)
            if mask != (1 << len(process.legs)) - 1:
                particle_id = (
                    edge.particle_pdg
                    if edge.source in loop_vertices
                    else model.particles[edge.particle_pdg].anti_pdg
                )
                attachments.add((mask, particle_id, tree_orders))
    return attachments


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "orders"),
    [("g g > g g", {"QCD": 2, "QED": 0}), ("u u~ > d d~ g", {"QCD": 3, "QED": 0})],
)
def test_block_currents_equal_feynkit_tree_sums(models, text, orders):
    assert _assert_blocks_equal_feynkit(models, text, POINT_SEEDS, orders) > 0


@pytest.mark.loop_nightly
def test_electroweak_block_currents_equal_feynkit_tree_sums(models):
    # Up to two electroweak couplings: photon, Z and W blocks and electroweak
    # exchanges inside quark blocks (chiral couplings fix the spinor map).
    orders = {"QCD": 3, "QED": 2}
    assert _assert_blocks_equal_feynkit(models, "u u~ > d d~ g", POINT_SEEDS, orders)


@pytest.mark.loop_nightly
def test_three_line_block_currents_equal_feynkit_tree_sums(models):
    # Five-leg blocks of six quarks join three fermion lines by two gluon
    # connections (``Block.connection_sign``).
    orders = {"QCD": 4, "QED": 0}
    text = "u u~ > d d~ s s~"
    assert _assert_blocks_equal_feynkit(models, text, POINT_SEEDS[:1], orders, 5)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "text",
    ["g g > g g g", "u u~ > d d~ g", "u u~ > u u~ g"],
)
def test_every_loop_position_content_is_a_block(models, text):
    compiled, model, structure = models
    native = structure.native
    process = _process(compiled, text)
    order = {"QCD": len(process.legs) - 2 + 2}
    blocks = build_tree_blocks(
        process,
        model,
        born_coupling_orders={**order, "QED": 0},
        block_coupling_orders={**order, "QED": 0},
    )
    available = {
        (block.key.external_mask, block.key.particle_id, block.key.coupling_orders)
        for block in blocks.blocks
    }
    required = _loop_attachments(native, model, process, {**order, "QED": 0})
    assert required
    missing = sorted(row for row in required if row not in available)
    assert not missing, missing
    # Coloured blocks carry every colour-flow word over their legs.
    representations = {
        leg.label: int(model.color_rep(leg.outgoing_pdg)) for leg in process.legs
    }
    if set(representations.values()) == {8}:
        words = {}
        for block in blocks.blocks:
            if block.key.particle_id == 21:
                words.setdefault(block.key.external_mask, set()).add(block.key.word)
        for mask, mask_words in words.items():
            assert len(mask_words) == math.factorial(mask.bit_count())


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "born"),
    [
        ("g g > g g", {"QCD": 2}),
        ("u u~ > d d~ g", {"QCD": 3}),
        ("g g > g g g", {"QCD": 3}),
        ("u u~ > u u~ g", {"QCD": 3}),
    ],
)
def test_family_recursion_runs_on_the_block_table(models, text, born):
    """WP1.3 consumes the blocks as its ``BlockTable`` and feeds back requests."""

    compiled, model, structure = models
    data = json.loads(structure.native.to_json())
    catalogue = ModelLoopCatalogue(
        data, species_orders=structure.perturbative_species_orders
    )
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({add_orders(tuple(born.items()), (("QCD", 2),))})
    )
    limits = {"QED": 0, **dict(selection.cap)}
    process = _process(compiled, text)
    table = build_tree_blocks(
        process, model, born_coupling_orders=limits, block_coupling_orders=limits
    ).block_table(data)
    native = structure.native
    incoming = [
        native.particle_by_pdg(model.particles[leg.outgoing_pdg].anti_pdg).name
        for leg in process.legs
        if leg.is_initial
    ]
    outgoing = [
        native.particle_by_pdg(leg.outgoing_pdg).name
        for leg in process.legs
        if not leg.is_initial
    ]
    reference = ModelBlockTable(data, incoming, outgoing, cap=selection.cap)
    assert table.legs == reference.legs
    for mask in range(1, 1 << len(table.legs)):
        assert table.currents(mask) == reference.currents(mask), mask
    families = generate_families(table, catalogue, selection)
    assert families.families
    assert families.families.keys() == (
        generate_families(reference, catalogue, selection).families.keys()
    )
    # The family recursion's requests drive the loop-aware reachability, and
    # the requested blocks alone reproduce the same families.
    requests = families.block_requests()
    requested = build_tree_blocks(
        process,
        model,
        born_coupling_orders=limits,
        block_coupling_orders=limits,
        reachability=requested_loop_blocks(requests, data),
    ).block_table(data)
    assert all(request in requested.currents(request.mask) for request in requests)
    again = generate_families(requested, catalogue, selection)
    assert again.families.keys() == families.families.keys()
    assert again.block_requests() == requests


@pytest.mark.loop_fast
def test_born_is_the_tree_program_bucketed_by_exact_order(models):
    compiled, model, _structure = models
    # The Born keeps the requested accuracy; this leading-colour tree has
    # both the QCD and the electroweak exchange.
    process = _process(compiled, "u u~ > d d~", "lc")
    born_limits = {"QCD": 2, "QED": 2}
    blocks = build_tree_blocks(
        process,
        model,
        born_coupling_orders=born_limits,
        block_coupling_orders={"QCD": 4, "QED": 2},
    )
    # Baseline of the plain tree program, recorded with compile_generic_dag on
    # 2590dc2c, before loop-request mode (WP1.2) touched the DAG compiler:
    # every root closes the (u u~ d) current with the d~ source.
    dag = blocks.born_dag
    assert len(dag.currents) == 30
    roots = Counter(
        (
            add_orders(
                dag.currents[root.left_id].index.coupling_orders,
                dag.currents[root.right_id].index.coupling_orders,
            ),
            root.color_sector_id,
            frozenset(
                (current.index.external_mask, current.index.particle_id)
                for current in (dag.currents[root.left_id], dag.currents[root.right_id])
            ),
        )
        for root in dag.amplitude_roots
    )
    closure = frozenset({(7, 1), (8, -1)})
    assert roots == {
        ((("QCD", 2),), 0, closure): 4,
        ((("QCD", 2),), 1, closure): 4,
        ((("QED", 2),), 0, closure): 4,
        ((("QED", 2),), 1, closure): 1,
    }
    assert blocks.born_orders == ((("QCD", 2),), (("QED", 2),))
    by_order = blocks.born_roots_by_order()
    assert sum(len(roots) for roots in by_order.values()) == len(
        blocks.born_dag.amplitude_roots
    )
    for orders, roots in by_order.items():
        for root in roots:
            dag_root = blocks.born_dag.amplitude_roots[root.root_id]
            left = blocks.born_dag.currents[dag_root.left_id].index
            right = blocks.born_dag.currents[dag_root.right_id].index
            assert add_orders(left.coupling_orders, right.coupling_orders) == (orders)
            assert [label for label, _helicity in root.helicities] == [1, 2, 3, 4]
    assert blocks.squared_born_orders() == {
        (("QCD", 2), ("QED", 2)): (
            ((("QCD", 2),), (("QED", 2),)),
            ((("QED", 2),), (("QCD", 2),)),
        ),
        (("QCD", 4),): (((("QCD", 2),), (("QCD", 2),)),),
        (("QED", 4),): (((("QED", 2),), (("QED", 2),)),),
    }
    # Blocks are exact colour flow whatever the requested accuracy.
    full = build_tree_blocks(
        _process(compiled, "u u~ > d d~"),
        model,
        born_coupling_orders=born_limits,
        block_coupling_orders={"QCD": 4, "QED": 2},
    )
    assert full.blocks_dag.process.color_accuracy == "full"
    assert {block.key for block in blocks.blocks} == {
        block.key for block in full.blocks
    }


@pytest.mark.loop_fast
def test_requested_blocks_keep_their_subcurrents_only(models):
    compiled, model, _structure = models
    process = _process(compiled, "g g > g g g")
    generic = GenericDAGCompiler(
        model=model, max_coupling_orders=QCD, loop_blocks=True
    ).compile(process)
    request = RequestedLoopBlocks(frozenset({(0b00111, 21, (("QCD", 2),))}))
    dag = GenericDAGCompiler(
        model=model,
        max_coupling_orders=QCD,
        loop_blocks=True,
        loop_block_reachability=request,
    ).compile(process)
    masks = {current.index.external_mask for current in dag.currents}
    assert masks <= {0b00011, 0b00101, 0b00110, 0b00111} | {1 << i for i in range(5)}
    requested = [
        current
        for current in dag.currents
        if current.index.external_mask == 0b00111 and current.index.particle_id == 21
    ]
    reference = [
        current
        for current in generic.currents
        if current.index.external_mask == 0b00111
        and current.index.particle_id == 21
        and current.index.coupling_orders == (("QCD", 2),)
    ]
    assert {c.index for c in requested} == {c.index for c in reference}
    assert len(requested) == 6 * 2**3


@pytest.mark.loop_fast
def test_generic_reachability_is_every_attachable_proper_subset(models):
    compiled, model, _structure = models
    process = _process(compiled, "u u~ > d d~ g")
    reachability = AttachableLoopBlocks.for_process(process, model)
    assert not reachability.admits_mask(0b11111)
    assert reachability.admits_mask(0b01111)
    dag = GenericDAGCompiler(model=model, loop_blocks=True).compile(process)
    assert all(current.index.external_mask != 0b11111 for current in dag.currents)
    assert not dag.amplitude_roots
    # The U(1)-subtraction piece of a closed quark line is a block too.
    assert any(
        current.index.auxiliary_kind == "u1-subtraction-color-flow-vector"
        and current.index.ordered_external_labels == (2, 1)
        for current in dag.currents
    )


@pytest.mark.loop_fast
def test_block_words_are_colour_flow_strings():
    representations = {1: -3, 2: 3, 3: 3, 4: -3, 5: 8, 6: 1}
    flow = block_colour_flow((3, 1, 2, 5, 6), representations, 3)
    assert flow.strings == ((3, 1), (2, 5))
    assert (flow.incoming_open, flow.outgoing_open) == (False, True)
    assert flow.singlet_labels == (6,)
    # Starts 2, 3 end at the open end (ranked last) and at 1: one inversion.
    assert flow.sign == -1
    gluon = block_colour_flow((4, 3, 5), representations, 8)
    assert gluon.strings == ((4,), (3, 5))
    assert (gluon.incoming_open, gluon.outgoing_open) == (True, True)
    assert gluon.sign == 1  # the open ends join: 3 -> 5 -> X -> 4
    # Two lines through an adjoint field: 2 -> X -> 1 and 3 -> 4.
    assert block_colour_flow((1, 3, 4, 2), representations, 8).sign == 1
    assert block_colour_flow((1, 2, 4, 3), representations, 8).sign == -1
    assert block_colour_flow((2, 1), representations, 1).sign == 1
    with pytest.raises(ValueError, match="not a colour-flow block word"):
        block_colour_flow((5, 2), representations, 8)
    with pytest.raises(ValueError, match="does not match a field"):
        block_colour_flow((2, 5), representations, -3)


@pytest.mark.loop_fast
def test_colour_octet_connections_flip_the_wick_sign(models):
    compiled, model, _structure = models
    limits = {"QCD": 3, "QED": 2}
    blocks = build_tree_blocks(
        _process(compiled, "u u~ > d d~ g"),
        model,
        born_coupling_orders=limits,
        block_coupling_orders=limits,
    )
    signs = {}
    for block in blocks.blocks:
        if block.key.external_mask == 0b00111 and block.key.particle_id == 1:
            signs.setdefault(block.key.coupling_orders, set()).add(
                (block.connection_sign, block.pairing_sign)
            )
    # The d-quark current of legs 1-3 joins the u-line by a gluon or by an
    # electroweak boson; identity pairing (2 -> 1, 3 -> open end) is odd.
    assert signs == {(("QCD", 2),): {(-1, -1)}, (("QED", 2),): {(1, -1)}}


@pytest.mark.loop_fast
def test_blocks_record_the_compiled_tree_kernel_variant(models):
    from pyamplicol.loops.v2.catalogue import compile_tree_kernel_variant
    from pyamplicol.models.external import CompiledUFOModel

    compiled, model, _structure = models
    process = _process(compiled, "g g > g g")
    for variant, tree_model in (
        ("transverse", model),
        ("full", CompiledUFOModel(compile_tree_kernel_variant(compiled, "full"))),
    ):
        blocks = build_tree_blocks(
            process, tree_model, born_coupling_orders=QCD, block_coupling_orders=QCD
        )
        assert blocks.kernel_variant == variant
        assert {block.kernel_variant for block in blocks.blocks} == {variant}


@pytest.mark.loop_fast
def test_loop_blocks_reject_tree_only_options(models):
    compiled, model, _structure = models
    with pytest.raises(ValueError, match="tree-only options: max_currents"):
        GenericDAGCompiler(model=model, loop_blocks=True, max_currents=10)
    with pytest.raises(ValueError, match="full-colour process"):
        GenericDAGCompiler(model=model, loop_blocks=True).compile(
            _process(compiled, "g g > g g", "lc")
        )
