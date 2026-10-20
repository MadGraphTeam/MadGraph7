# SPDX-License-Identifier: 0BSD
"""Tree blocks of loop requests (design §3.2, WP1.2).

FeynKit appears here only as the oracle: full-colour amputated tree sums and
one-loop diagram attachments. The production blocks come from the tree engine
in loop-request mode.
"""

from __future__ import annotations

import math
import string
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

np = pytest.importorskip("numpy")

from pyamplicol.generation.dag_compiler import (  # noqa: E402
    AttachableLoopBlocks,
    GenericDAGCompiler,
    compile_generic_dag,
)
from pyamplicol.loops.v2.blocks import (  # noqa: E402
    RequestedLoopBlocks,
    add_coupling_orders,
    block_colour_flow,
    build_tree_blocks,
)

QCD = {"QCD": 99, "QED": 0}
POINT_SEEDS = (7, 23)
# FeynKit evaluates its tensors and the model couplings in binary64.
TOLERANCE = 1e-12
# The tree engine's Weyl spinors of chirality +1 / -1 are the Spenso Dirac
# components 2,3 / 0,1 of the parity-mirrored kinematics; FeynKit therefore
# evaluates the mirrored process (mirrored momenta and vectors).
MIRROR = np.array([1.0, -1.0, -1.0, -1.0])


@pytest.fixture(scope="module")
def models(loop_sm_ufo, tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("tree blocks require licensed Symbolica")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_sm_ufo,
        cache_dir=tmp_path_factory.mktemp("loop-v2-blocks") / "model-cache",
        require_supported=False,
    )
    native = prepare_loop_request_model(
        replace(
            compiled,
            raw_loop_model=prepare_loop_sm_scalar_declarations(
                compiled.raw_loop_model
            ).derived,
            _serialized_path=None,
            _prepared_bundle=None,
        )
    ).structure.native
    return compiled, CompiledUFOModel(compiled), native


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

    The interpreted Symbolica stage evaluators of the tree runtime, as in the
    numerical-current warm-up; nothing is compiled. ``parameters`` overrides
    runtime model parameters by name.
    """

    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
    from pyamplicol.generation.contracts import StageCompilationInput
    from pyamplicol.generation.numerical_current_warmup import (
        _build_interpreted_stage_evaluator,
        _evaluate_interpreted_stage,
        _runtime_model_parameter_values,
    )
    from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
    from pyamplicol.generation.stage_planning import (
        build_generic_stage_compiler_blueprint,
    )
    from pyamplicol.runtime.symbolica_exact import _fill_momenta, _fill_sources

    runtime_schema = build_runtime_expression_schema(dag, model, process_id="blocks")
    schema = runtime_schema.to_mapping()
    blueprint = build_generic_stage_compiler_blueprint(
        StageCompilationInput(dag, model, runtime_schema)
    )
    assert blueprint.expression_ready, blueprint.blockers
    settings = SymbolicaEvaluatorSettings(
        iterations=1,
        cpe_iterations=0,
        n_cores=1,
        direct_translation=False,
        jit_direct_translation=False,
        jit_optimization_level=0,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        compiled_chunk_compile_workers=1,
    )
    layout = schema["parameter_layout"]
    value_count = int(layout["value_component_count"])
    momentum_count = int(layout["momentum_parameter_count"])
    overrides = parameters or {}
    parameters = tuple(
        Decimal(overrides[record["name"]]) if record["name"] in overrides else value
        for record, value in zip(
            schema["model_parameters"],
            _runtime_model_parameter_values(schema),
            strict=True,
        )
    )
    with localcontext() as context:
        context.prec = precision
        momenta = tuple(
            tuple(Decimal.from_float(float(x)) for x in momentum) for momentum in point
        )
        state = [(Decimal(0), Decimal(0))] * max(
            int(layout["parameter_count_if_flattened"]),
            value_count + momentum_count + len(parameters),
        )
        _fill_sources(state, momenta, schema, parameters)
        _fill_momenta(state, momenta, schema)
        for index, value in enumerate(parameters):
            state[value_count + momentum_count + index] = (value, Decimal(0))
        for stage in blueprint.stages:
            evaluator = _build_interpreted_stage_evaluator(
                stage,
                settings=settings,
                output_partitions=((0, stage.output_length),),
            )
            inputs = [None] * stage.parameter_count
            for component in stage.input_components:
                inputs[component.parameter_index] = state[component.global_component]
            outputs = _evaluate_interpreted_stage(
                evaluator, tuple(inputs), precision=precision
            )
            for slot in stage.output_slots:
                state[slot.component_start : slot.component_stop] = outputs[
                    slot.output_start : slot.output_stop
                ]
    return {
        (int(slot["current_id"]), str(slot["variant"])): np.array(
            [
                complex(float(real), float(imaginary))
                for real, imaginary in state[
                    slot["component_start"] : slot["component_stop"]
                ]
            ]
        )
        for slot in schema["value_storage"]["value_slots"]
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

    flow = block_colour_flow(word, representations)
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


def _colour_pairing_sign(word, x_representation, representations, pdgs):
    """Parity of the colour pairing relative to the fermion-line pairing.

    Colour-flow blocks own the ordering sign of coloured fermions: a block
    word whose strings pair (fundamental -> antifundamental) differently from
    the fermion lines carries the parity of that permutation.
    """

    flow = block_colour_flow(word, representations)
    colour = {}
    for labels in flow.strings:
        start = labels[0] if representations[labels[0]] == 3 else "in"
        end = labels[-1] if representations[labels[-1]] == -3 else "out"
        colour[start] = end
    starts = sorted(label for label in word if representations[label] == 3)
    ends = sorted(label for label in word if representations[label] == -3)
    fermion = {}
    for start in starts:
        partner = next(
            (
                end
                for end in ends
                if pdgs[end] == -pdgs[start] and end not in fermion.values()
            ),
            "out",
        )
        fermion[start] = partner
    if x_representation in (8, -3):
        fermion["in"] = next(
            (end for end in [*ends, "out"] if end not in fermion.values()), None
        )
    targets = list(fermion.values())
    permutation = [targets.index(colour[start]) for start in fermion]
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _assert_blocks_equal_feynkit(models, text, seeds, orders):
    compiled, model, native = models
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
        if block.is_source or block.key.auxiliary_kind is not None:
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
                words = {}
                for block, row in rows:
                    words[block.key.word] = np.multiply.outer(
                        _colour_tensor(
                            block.key.word,
                            x_representation,
                            colour_axes,
                            representations,
                            oracle.generators,
                        ),
                        values[(row.current_id, variant)],
                    )
                comparisons.append((fk, words))
            # Each colour-flow word enters the full current with a sign +-1,
            # the same for every helicity, colour component and point: fit
            # the signs once over all helicity rows.
            word_list = sorted({word for _fk, words in comparisons for word in words})
            target = np.concatenate([fk.reshape(-1) for fk, _words in comparisons])
            design = np.stack(
                [
                    np.concatenate(
                        [
                            words.get(word, np.zeros(fk.shape)).reshape(-1)
                            for fk, words in comparisons
                        ]
                    )
                    for word in word_list
                ],
                axis=1,
            )
            scale = np.abs(target).max()
            assert scale > 0, (labels, particle_id, orders)
            signs, *_ = np.linalg.lstsq(design, target, rcond=None)
            assert np.abs(design @ signs - target).max() <= TOLERANCE * scale, (
                labels,
                particle_id,
                orders,
            )
            assert np.allclose(np.abs(signs), 1, rtol=0, atol=1e-9), signs
            assert np.allclose(signs.imag, 0, rtol=0, atol=1e-9), signs
            if all(name == "QCD" for name, _power in orders):
                # With flavour-conserving lines the sign is the colour-pairing
                # parity, up to one overall sign: the free external-ordering
                # convention of each program.
                predicted = [
                    _colour_pairing_sign(word, x_representation, representations, pdgs)
                    for word in word_list
                ]
                ratios = signs.real / np.array(predicted)
                assert np.allclose(ratios, ratios[0], rtol=0, atol=1e-9), (
                    labels,
                    signs,
                    predicted,
                )
            checked += len(comparisons)
    return checked


def _loop_attachments(native, model, process, orders):
    """(leg mask, |field pdg|, coupling orders) of every tree a FeynKit
    one-loop diagram attaches to its loop (oracle only).

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
                    tree_orders = add_coupling_orders(
                        tree_orders, orders_by_vertex[vertex]
                    )
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
                attachments.add((mask, abs(edge.particle_pdg), tree_orders))
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


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "text",
    ["g g > g g g", "u u~ > d d~ g", "u u~ > u u~ g"],
)
def test_every_loop_position_content_is_a_block(models, text):
    compiled, model, native = models
    process = _process(compiled, text)
    order = {"QCD": len(process.legs) - 2 + 2}
    blocks = build_tree_blocks(
        process,
        model,
        born_coupling_orders={**order, "QED": 0},
        block_coupling_orders={**order, "QED": 0},
    )
    available = {
        (block.key.external_mask, abs(block.key.particle_id), block.key.coupling_orders)
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


@pytest.mark.loop_fast
def test_born_is_the_tree_program_bucketed_by_exact_order(models):
    compiled, model, _native = models
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
    assert blocks.born_dag == compile_generic_dag(
        process, model=model, max_coupling_orders=born_limits
    )
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
            assert add_coupling_orders(left.coupling_orders, right.coupling_orders) == (
                orders
            )
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
    compiled, model, _native = models
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
    compiled, model, _native = models
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
    flow = block_colour_flow((3, 1, 2, 5, 6), representations)
    assert flow.strings == ((3, 1), (2, 5))
    assert (flow.incoming_open, flow.outgoing_open) == (False, True)
    assert flow.singlet_labels == (6,)
    gluon = block_colour_flow((4, 3, 5), representations)
    assert gluon.strings == ((4,), (3, 5))
    assert (gluon.incoming_open, gluon.outgoing_open) == (True, True)
    with pytest.raises(ValueError, match="not a colour-flow block word"):
        block_colour_flow((5, 2), representations)


@pytest.mark.loop_fast
def test_loop_blocks_reject_tree_only_options(models):
    compiled, model, _native = models
    with pytest.raises(ValueError, match="tree-only options: max_currents"):
        GenericDAGCompiler(model=model, loop_blocks=True, max_currents=10)
    with pytest.raises(ValueError, match="full-colour process"):
        GenericDAGCompiler(model=model, loop_blocks=True).compile(
            _process(compiled, "g g > g g", "lc")
        )
