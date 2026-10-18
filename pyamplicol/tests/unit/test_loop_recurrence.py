# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import numpy as np
import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.kernels import CoefficientDAG
from pyamplicol.loops.recurrence import build_open_loop_recurrence, separate_local_color


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native graph worker requires a licensed Symbolica runtime")
    from symbolica.community import hepkit, tensor

    return symbolica, hepkit, tensor


@pytest.fixture(scope="module")
def graphs(native):
    _, hep, _ = native

    def build(model, incoming, outgoing, correction):
        return build_loop_process(
            model.process(incoming, outgoing),
            selected_corrections=[correction],
            perturbative_orders={correction: 1},
        )

    model = hep.Model.qcd_qed()
    return {
        "dy": build(model, ["u", "u~"], ["e-", "e+"], "QCD"),
        "qcd": build(model, ["u", "u~"], ["d", "d~"], "QCD"),
        "scalar": build(hep.Model.phi4(), ["phi", "phi"], ["phi", "phi"], "SCALAR"),
    }


def _oracle(recurrence, native):
    """Independent full graph contraction, used only for small test graphs.

    Native route_expression performs routing; no recurrence signature assembly,
    local color separation, coefficient extraction or current kernels are used.
    Explicit SU(3) materialization is confined to this independent oracle.
    """
    symbolica, hep, tensor = native
    graph = recurrence.graph.native
    library = tensor.TensorLibrary.hep_lib_atom()
    loop = tensor.TensorName.vector("loop_test::loop")
    external = [
        tensor.TensorName.vector(f"loop_test::p{i}")
        for i in range(len(recurrence.momentum_parameters))
    ]
    library.register(
        tensor.Tensor.dense(
            loop(tensor.Representation.mink(4)), recurrence.loop_components
        )
    )
    for name, components in zip(external, recurrence.momentum_parameters, strict=True):
        library.register(
            tensor.Tensor.dense(name(tensor.Representation.mink(4)), components)
        )
    expression = graph.loop_momentum_basis.route_expression(
        graph.numerator_expression(),
        loop_momenta=[loop],
        external_momenta=[
            external[i]
            for i, edge_id in enumerate(graph.loop_momentum_basis.external_edges)
            if edge_id not in graph.loop_momentum_basis.dependent_externals
        ],
    )
    states = dict(recurrence.external_states)
    for edge in graph.external_edges:
        if edge.external_index not in states:
            continue
        half = next(half for half in graph.half_edges if half.edge.id == edge.id)
        index = hep.Symbols.half_edge()(half.id, 1)
        axes = [
            slot.dual()
            for slot in expression.axes
            if slot.index == index
            and slot.representation
            in (tensor.Representation.bis(4), tensor.Representation.mink(4))
        ]
        name = tensor.TensorName(f"loop_test::state{edge.external_index}")
        library.register(
            tensor.Tensor.dense(
                name(*(slot.representation for slot in axes)),
                states[edge.external_index],
            )
        )
        expression *= name(*axes)
    expression *= graph.overall_factor_expression(evaluate=True)
    result = expression.to_tensor(library).to_dense()
    color = recurrence.color_expression.to_tensor(library).to_dense()
    color = color.permute_axes([color.axes.index(axis) for axis in result.axes])
    assert result.axes == color.axes
    params = [*recurrence.dag.parameters, *recurrence.dag.variables]
    evaluator = symbolica.Expression.evaluator_multiple(
        [result[i] for i in range(len(result))], params, n_cores=1, jit_compile=False
    )
    color_evaluator = symbolica.Expression.evaluator_multiple(
        [color[i] for i in range(len(color))], [], n_cores=1, jit_compile=False
    )
    return evaluator, color_evaluator


def _polynomial(recurrence, coefficients, loop_points):
    powers = np.asarray(
        [
            label.powers
            for label in recurrence.dag.nodes[recurrence.final_node].output_labels
        ]
    )
    monomials = np.prod(loop_points[:, None, :] ** powers[None, :, :], axis=2)
    return np.sum(coefficients * monomials, axis=1)


def _check_oracle(recurrence, native):
    rng = np.random.default_rng(8127)
    count = len(recurrence.dag.parameters)
    values = rng.normal(size=(3, count)) + 1j * rng.normal(size=(3, count))
    loop_points = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    coefficients = recurrence.dag.evaluate_batch(values)[recurrence.final_node]
    actual = _polynomial(recurrence, coefficients, loop_points)
    oracle, color = _oracle(recurrence, native)
    expected = oracle.evaluate_complex(np.concatenate([values, loop_points], axis=1))
    colors = color.evaluate_complex(np.empty((3, 0), dtype=complex))
    np.testing.assert_allclose(
        actual[:, None] * colors, expected, rtol=2e-12, atol=2e-10
    )
    return values, loop_points, actual


def test_drell_yan_local_dag_cut_independence_and_full_graph_oracle(native, graphs):
    graph = graphs["dy"].loops[0]
    results = []
    for cut in graph.loop_edges:
        recurrence = build_open_loop_recurrence(graph, cut_edge=cut)
        assert recurrence.dag.domain == "four-dimensional-model-rules"
        assert not recurrence.dag.direct_r2_certified
        assert recurrence.dag.nodes[-1].rank == 2
        assert len(recurrence.tree_denominators) == 1
        assert set(recurrence.propagator_history) == set(graph.loop_edges)
        assert recurrence.propagator_history[-1] == cut
        assert recurrence.graph.family is graph.family
        assert all(
            not node.axes or node.name != f"closure:{cut}"
            for node in recurrence.dag.nodes
        )
        # Local instructions consume coefficient placeholders, not recursively
        # substituted whole-process symbolic amplitudes.
        for node in recurrence.dag.nodes:
            assert node.evaluator.get_instructions().input_count == len(
                node.parameters
            ) + len(node.input_labels)
            assert node.evaluator_bytes()
            assert not any(
                s.contains(q)
                for s in node.expressions
                for q in recurrence.dag.variables
            )
        values, _loop_points, actual = _check_oracle(recurrence, native)
        results.append(actual)
        assert (
            recurrence.dag.evaluate_batch(values[:0])[recurrence.final_node].shape[0]
            == 0
        )
        with pytest.raises(TypeError, match="binary64"):
            recurrence.dag.evaluate_batch(values.astype(object))
    for actual in results[1:]:
        np.testing.assert_allclose(actual, results[0], rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("kind", ["ghost", "gauge", "fermion", "massive", "scalar"])
def test_gauge_ghost_fermion_closure_and_scalar_quartic(native, graphs, kind):
    if kind == "scalar":
        graph = graphs["scalar"].loops[0]
    elif kind == "ghost":
        graph = next(
            g
            for g in graphs["qcd"].loops
            if any(e.spin == -1 and e.id in g.loop_edges for e in g.edges)
        )
    elif kind == "gauge":
        graph = next(
            g
            for g in graphs["qcd"].loops
            if g.loop_species == (21,) and len(g.loop_edges) == 2
        )
    elif kind == "massive":
        graph = next(
            g
            for g in graphs["qcd"].loops
            if g.loop_species == (6,) and len(g.loop_edges) == 2
        )
    else:
        graph = next(
            g
            for g in graphs["qcd"].loops
            if len(g.loop_edges) == 2
            and all(e.spin == 2 for e in g.edges if e.id in g.loop_edges)
        )
    recurrence = build_open_loop_recurrence(graph)
    assert recurrence.dag.nodes[-1].rank == (0 if kind == "scalar" else 2)
    if kind in {"ghost", "fermion", "massive"}:
        assert "InternalFermionLoopSign(-1)" in graph.overall_factor
    if kind == "scalar":
        assert graph.evaluated_factor == "1/2"
    if kind == "massive":
        assert any(p.get_name() == "UFO::MT" for p in recurrence.dag.parameters)
    _check_oracle(recurrence, native)


def test_native_exact_polynomial_kernel_and_precise_storage(native):
    symbolica, _, tensor = native
    q = tuple(symbolica.S(f"loop_kernel_test::q{i}") for i in range(4))
    x = symbolica.S("loop_kernel_test::x")
    dag = CoefficientDAG(q)
    scalar = tensor.TensorExpression((x + q[0]) / 7).to_tensor()
    first = dag.add("seed", scalar)
    second = dag.add(
        "update",
        tensor.TensorExpression(q[1] + symbolica.E("1i")).to_tensor(),
        (first,),
    )
    assert dag.nodes[second].rank == 2
    assert any(
        symbolica.E("1/7") in node.evaluator.get_instructions().constants
        for node in dag.nodes
    )
    precise = dag.nodes[first].evaluator.evaluate_complex_with_prec(
        ["1.000000000000000000000000000000000001"], 60
    )
    assert any(
        "142857142857142857142857142857142857" in str(value) for value in precise
    )
    with pytest.raises(ValueError, match="not polynomial"):
        dag.add("bad", tensor.TensorExpression(1 / q[0]).to_tensor())


def test_multicolor_local_rule_rejects_without_su3_reconstruction(native):
    _, hep, _ = native
    process = hep.Model.qcd_qed().process(["g", "g"], ["g", "g"])
    born = process.generate_diagrams(loops=0, progress=None, threads=1).diagrams
    vertex = next(
        v for diagram in born for v in diagram.vertices if len(diagram.vertices) == 1
    )
    with pytest.raises(NotImplementedError, match="color-term DAG"):
        separate_local_color(vertex.numerator_expression())


def test_family_affine_shift_is_applied_before_coefficient_extraction(native, graphs):
    graph = next(
        g
        for g in graphs["qcd"].loops
        if any("P(0)" in str(image) for _, image in g.family.native.momentum_rules)
    )
    recurrence = build_open_loop_recurrence(graph, routing="family")
    assert recurrence.routing == "family"
    assert recurrence.loop_components != recurrence.dag.variables
    for axis, expression in enumerate(recurrence.loop_components):
        assert (
            expression
            == -recurrence.dag.variables[axis] + recurrence.momentum_parameters[0][axis]
        )
    _check_oracle(recurrence, native)


def test_coefficient_kernel_rejects_rounded_literal_before_serialization(native):
    symbolica, _, tensor = native
    q = tuple(symbolica.S(f"loop_float_test::q{i}") for i in range(4))
    dag = CoefficientDAG(q)
    for value in [0.1, 1j]:
        with pytest.raises(ValueError, match="floating numeric atoms"):
            dag.add(
                "rounded",
                tensor.TensorExpression(symbolica.Expression.num(value)).to_tensor(),
            )
