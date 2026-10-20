# SPDX-License-Identifier: 0BSD
"""Exact small graph oracles for local colour branching, never production traces."""

from __future__ import annotations

import pytest

from pyamplicol.loops.graph_basis import (
    build_color_loop_recurrence,
    build_color_tree_recurrence,
)
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.recurrence import build_open_loop_recurrence

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def native():
    s = pytest.importorskip("symbolica")
    if not s.is_licensed():
        pytest.skip("native graph preparation requires licensed Symbolica")
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    return s, h, t


@pytest.fixture(scope="module")
def process(native):
    _, h, _ = native
    return build_loop_process(
        h.Model.qcd().process(["g", "g"], ["g", "g"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )


def _exact_values(recurrence, s, seed):
    symbols = set(recurrence.dag.parameters) | set(recurrence.dag.variables)
    symbols.update(x for vector in recurrence.momentum_parameters for x in vector)
    symbols.update(x for _, vector in recurrence.external_states for x in vector)
    return {
        name: s.E(f"{((i + 2) * (seed + 3)) % 13 + 1}/{i % 5 + 1}")
        + s.E("1i") * ((i + seed) % 7 - 3)
        for i, name in enumerate(sorted(symbols, key=str))
    }


def _exact_dag_terms(recurrence, values, s):
    """Exact substitution of stored local kernels for this small audit only."""
    outputs = []
    parameter_rules = [s.Replacement(name, value) for name, value in values.items()]
    for node in recurrence.dag.nodes:
        input_rules = [
            s.Replacement(
                s.S(f"loop_coefficient::input_{position}_{column}"),
                outputs[dependency][column],
            )
            for position, dependency in enumerate(node.dependencies)
            for column, _ in enumerate(recurrence.dag.nodes[dependency].output_labels)
        ]
        outputs.append(
            tuple(
                expression.replace_multiple(parameter_rules + input_rules).expand()
                for expression in node.expressions
            )
        )
    terms = []
    for term in recurrence.terms:
        value = s.E("0")
        for coefficient, label in zip(
            outputs[term.node],
            recurrence.dag.nodes[term.node].output_labels,
            strict=True,
        ):
            for variable, power in zip(
                recurrence.dag.variables, label.powers, strict=True
            ):
                if power:
                    coefficient *= values[variable] ** power
            value += coefficient
        terms.append((value.expand(), term.color))
    return terms


def _color_projection(expression, graph, library, seed, native):
    s, h, t = native
    for edge in graph.external_edges:
        half = next(half for half in graph.half_edges if half.edge.id == edge.id)
        index = h.Symbols.half_edge()(half.id, 1)
        axes = [
            slot
            for slot in expression.axes
            if slot.index == index and slot.representation == t.Representation.coad(8)
        ]
        assert len(axes) == 1
        axis = axes[0]
        name = t.TensorName(f"graph_basis_oracle::color_{edge.external_index}")
        vector = [
            s.E(str(((edge.external_index + 2) * (j + seed + 1) + j * j) % 11 - 5))
            for j in range(8)
        ]
        library.register(t.Tensor.dense(name(axis.representation.dual()), vector))
        expression *= name(axis.dual())
    return expression.to_tensor(library)[0].expand()


def _full_graph_oracle(recurrence, values, seed, native):
    """Route and contract the unsplit native graph independently of the DAG."""
    s, h, t = native
    graph = recurrence.graph.native
    library = t.TensorLibrary.hep_lib_atom()
    rules = [s.Replacement(name, value) for name, value in values.items()]
    loop = t.TensorName.vector("graph_basis_oracle::loop")
    if graph.loop_count:
        library.register(
            t.Tensor.dense(
                loop(t.Representation.mink(4)),
                [x.replace_multiple(rules) for x in recurrence.loop_components],
            )
        )
    externals = [
        t.TensorName.vector(f"graph_basis_oracle::p{i}")
        for i in range(len(recurrence.momentum_parameters))
    ]
    for name, components in zip(externals, recurrence.momentum_parameters, strict=True):
        library.register(
            t.Tensor.dense(
                name(t.Representation.mink(4)), [values[x] for x in components]
            )
        )
    expression = graph.numerator_expression() * graph.overall_factor_expression(
        evaluate=True
    )
    if recurrence.tree_denominators_included:
        edge_id, momentum, mass, value = s.S(
            "graph_basis_oracle::e_",
            "graph_basis_oracle::p_",
            "graph_basis_oracle::m_",
            "graph_basis_oracle::v_",
        )
        for edge in graph.internal_edges:
            denominator = (
                edge.denominator_expression(dimension=4)
                .to_expression()
                .replace(h.Symbols.denominator()(edge_id, momentum, mass, value), value)
            )
            expression /= denominator
    expression = graph.loop_momentum_basis.route_expression(
        expression,
        loop_momenta=[loop] if graph.loop_count else [],
        external_momenta=[
            externals[i]
            for i, edge in enumerate(graph.loop_momentum_basis.external_edges)
            if edge not in graph.loop_momentum_basis.dependent_externals
        ],
    )
    states = dict(recurrence.external_states)
    for edge in graph.external_edges:
        half = next(half for half in graph.half_edges if half.edge.id == edge.id)
        index = h.Symbols.half_edge()(half.id, 1)
        axes = [
            slot.dual()
            for slot in expression.axes
            if slot.index == index and slot.representation == t.Representation.mink(4)
        ]
        assert len(axes) == 1
        name = t.TensorName(f"graph_basis_oracle::state{edge.external_index}")
        library.register(
            t.Tensor.dense(
                name(*(slot.representation for slot in axes)),
                [values[x] for x in states[edge.external_index]],
            )
        )
        expression *= name(*axes)
    expression = t.TensorExpression(expression.to_expression().replace_multiple(rules))
    return _color_projection(expression, graph, library, seed, native)


def _check_exact(recurrence, seed, native):
    s, _, t = native
    values = _exact_values(recurrence, s, seed)
    terms = _exact_dag_terms(recurrence, values, s)
    actual = sum(
        (
            _color_projection(
                color,
                recurrence.graph.native,
                t.TensorLibrary.hep_lib_atom(),
                seed,
                native,
            )
            * value
            for value, color in terms
        ),
        s.E("0"),
    ).expand()
    expected = _full_graph_oracle(recurrence, values, seed, native)
    assert actual != 0  # Avoid a vacuous component/colour projection.
    assert (actual - expected).expand() == 0
    return actual


@pytest.mark.parametrize("seed", [1, 4])
def test_all_gluon_born_graphs_exact_including_quartic_and_denominators(
    native, process, seed
):
    recurrences = [build_color_tree_recurrence(graph) for graph in process.born]
    assert sorted(len(r.terms) for r in recurrences) == [1, 1, 1, 3]
    for recurrence in recurrences:
        assert recurrence.current.closed and recurrence.tree_denominators_included
        assert all(
            recurrence.dag.nodes[node].rank == 0 for node in recurrence.final_nodes
        )
        _check_exact(recurrence, seed, native)


@pytest.mark.parametrize("pattern", [(2, 1), (2, 2), (3, 1)])
def test_quartic_loop_graphs_exact_every_cut_with_native_symmetry(
    native, process, pattern
):
    graph = next(
        graph
        for graph in process.loops
        if (
            len(graph.loop_edges),
            sum(
                sum(half.vertex == v.id for half in graph.native.half_edges) == 4
                for v in graph.native.vertices
            ),
        )
        == pattern
    )
    assert graph.loop_species == (21,)
    results = []
    for cut in graph.loop_edges:
        recurrence = build_color_loop_recurrence(graph, cut_edge=cut)
        assert len(recurrence.terms) == 3 ** pattern[1]
        assert recurrence.current.closed and not recurrence.tree_denominators_included
        assert recurrence.propagator_history[-1] == cut
        assert set(recurrence.propagator_history) == set(graph.loop_edges)
        assert len(recurrence.tree_denominators) == len(graph.bridge_edges)
        results.append(_check_exact(recurrence, 2, native))
    assert all(value == results[0] for value in results)
    with pytest.raises(NotImplementedError, match="build_color_loop_recurrence"):
        build_open_loop_recurrence(graph)
    # Each final branch receives precisely the dependency's graph factor. The
    # two-quartic bubble has a nontrivial symmetry denominator.
    if pattern == (2, 2):
        assert graph.symmetry_denominator == 2


def test_quartic_loop_family_routing_exact(native, process):
    graph = next(
        graph
        for graph in process.loops
        if len(graph.loop_edges) == 3
        and any(
            sum(half.vertex == v.id for half in graph.native.half_edges) == 4
            for v in graph.native.vertices
        )
    )
    recurrence = build_color_loop_recurrence(graph, routing="family")
    assert recurrence.routing == "family"
    _check_exact(recurrence, 3, native)


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "1e-12"),
        ("double-double", None, "1e-27"),
        ("arbitrary", 70, "1e-62"),
    ],
)
def test_quartic_alternatives_execute_as_native_precise_batches(
    loop_native, native, process, arithmetic, precision, tolerance
):
    from decimal import Decimal, localcontext
    from fractions import Fraction

    from pyamplicol.loops.kernels import NativeCoefficientDAG

    backend = loop_native
    s, _, _ = native
    graph = next(graph for graph in process.born if len(graph.vertices) == 1)
    recurrence = build_color_tree_recurrence(graph)
    schedule = NativeCoefficientDAG(
        recurrence.dag,
        arithmetic=arithmetic,
        precision=precision,
        output_nodes=recurrence.final_nodes,
        backend=backend,
    )
    rows, expected = [], []
    for lane in range(8):
        row, values = [], {q: s.E("0") for q in recurrence.dag.variables}
        for i, parameter in enumerate(recurrence.dag.parameters):
            real = f"{(i + lane) % 5 + 1}." + "0" * 62 + "1"
            imaginary = str((2 * i + lane) % 3 - 1)
            row.append([real, imaginary])
            values[parameter] = s.E(str(Fraction(real))) + s.E("1i") * int(imaginary)
        rows.extend(row)
        expected.extend(value for value, _ in _exact_dag_terms(recurrence, values, s))
    ids = [99 - lane * 7 for lane in range(8)]
    inputs = backend.NativeBatch.from_decimal(
        rows, 8, len(recurrence.dag.parameters), arithmetic, precision, ids
    )
    outputs = schedule.evaluate_batch(inputs)
    assert outputs.rows == 8 and outputs.columns == 3 and outputs.row_ids == ids
    with localcontext() as context:
        context.prec = 100
        for (real, imaginary), exact in zip(
            outputs.to_decimal(), expected, strict=True
        ):
            oracle = exact.evaluate({}, decimal_digit_precision=90)
            for actual, target in [
                (real, str(oracle.real)),
                (imaginary, str(oracle.imag)),
            ]:
                assert abs(Decimal(actual) - Decimal(target)) <= Decimal(
                    tolerance
                ) * max(1, abs(Decimal(target)))
    empty = schedule.evaluate_batch(inputs.take_rows([]))
    assert empty.rows == 0 and empty.columns == 3 and empty.row_ids == []
