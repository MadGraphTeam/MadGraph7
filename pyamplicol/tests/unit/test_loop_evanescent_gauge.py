# SPDX-License-Identifier: 0BSD
"""Dependency-native Yang--Mills state blocks and dimensional source oracles."""

from __future__ import annotations

import pytest

from pyamplicol.loops.color_terms import split_local_color_terms
from pyamplicol.loops.evanescent_gauge import DimensionalYangMillsKernels
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.recurrence import (
    attach_external_spin_components,
    build_open_loop_recurrence,
    momentum_component_library,
)
from tools.loop_reference.graph_r2_oracle import (
    closed_polynomial,
    exact_graph_coefficients,
    graph_full_dimension_numerator,
    sample_bindings,
)


@pytest.fixture(scope="module")
def process():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dimensional tensor runtime required")
    from symbolica.community import hepkit as h

    return build_loop_process(
        h.Model.qcd().process(["g", "g"], ["g", "g"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )


@pytest.fixture(scope="module")
def controls(process):
    return {
        (spin, size): next(
            graph
            for graph in process.loops
            if len(graph.loop_edges) == size
            and all(e.spin == spin for e in graph.edges if e.id in graph.loop_edges)
            and all(
                len(split_local_color_terms(v.numerator_expression())) == 1
                for v in graph.native.vertices
            )
        )
        for spin in (3, -1)
        for size in (2, 3)
    }


@pytest.fixture(scope="module")
def recurrences(controls):
    return {
        key: build_open_loop_recurrence(
            graph, routing="family", kernel_provider=DimensionalYangMillsKernels()
        )
        for key, graph in controls.items()
    }


def polynomial(recurrence):
    return closed_polynomial(
        recurrence,
        exact_graph_coefficients(recurrence, sample_bindings(recurrence)),
        even=False,
    )


@pytest.mark.parametrize("spin,size", [(3, 2), (3, 3), (-1, 2), (-1, 3)])
def test_weighted_local_gauge_and_ghost_match_independent_dimensional_source(
    recurrences, spin, size
):
    recurrence = recurrences[spin, size]
    mu, epsilon = recurrence.dag.variables[4:]
    actual = polynomial(recurrence)
    expected = graph_full_dimension_numerator(
        recurrence, sample_bindings(recurrence), dimension_value=4 - 2 * epsilon
    )
    assert (actual - expected).expand() == 0
    assert actual.replace(mu, -mu).expand() == actual
    baseline = build_open_loop_recurrence(
        recurrence.graph,
        routing="family",
        kernel_provider=DimensionalYangMillsKernels(4, dimensional_weight=False),
    )
    assert (
        actual.replace(epsilon, 0).replace(mu, 0) - polynomial(baseline)
    ).expand() == 0
    assert recurrence.dag.direct_r2_certified is False
    if spin == -1:
        assert not actual.contains(mu) and not actual.contains(epsilon)
    else:
        assert actual.coefficient(epsilon) != 0
        assert (actual - actual.replace(mu, 0)).expand() != 0


@pytest.mark.parametrize("size", (2, 3))
def test_spectator_weight_is_independent_of_cut_and_selected_cycle_edge(
    recurrences, size
):
    reference = recurrences[3, size]
    expected = polynomial(reference)
    for selected in reference.graph.loop_edges:
        changed = build_open_loop_recurrence(
            reference.graph,
            routing="family",
            cut_edge=selected,
            kernel_provider=DimensionalYangMillsKernels(weighted_edge=selected),
        )
        assert (polynomial(changed) - expected).expand() == 0
    changed_cut = build_open_loop_recurrence(
        reference.graph,
        routing="family",
        cut_edge=max(reference.graph.loop_edges),
        kernel_provider=DimensionalYangMillsKernels(
            weighted_edge=min(reference.graph.loop_edges)
        ),
    )
    assert (polynomial(changed_cut) - expected).expand() == 0
    five, six = [
        build_open_loop_recurrence(
            reference.graph,
            routing="family",
            kernel_provider=DimensionalYangMillsKernels(
                dimension, dimensional_weight=False
            ),
        )
        for dimension in (5, 6)
    ]
    n5, n6 = polynomial(five), polynomial(six)
    epsilon = reference.dag.variables[-1]
    assert (expected - n5 - (-1 - 2 * epsilon) * (n6 - n5)).expand() == 0


def local_library(graph, dimension, *, weighted=False):
    from symbolica import S

    q = tuple(S(f"gauge_local_q{i}") for i in range(4))
    momenta = tuple(
        tuple(S(f"gauge_local_p{leg}_{i}") for i in range(4)) for leg in range(4)
    )
    library = momentum_component_library(graph, (q,), momenta)
    local = {
        ("vertex", v.id): split_local_color_terms(v.numerator_expression())[0].lorentz
        for v in graph.native.vertices
    }
    attach_external_spin_components(graph, local, library)
    provider = DimensionalYangMillsKernels(dimension, dimensional_weight=weighted).bind(
        graph, q, q
    )
    provider._prepare_library(library)
    return provider, library


def local_cycle_matrix(graph, vertex, source, dimension):
    """Contract arbitrary physical attachments through dependency tensors."""
    from symbolica import E, S
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    provider, library = local_library(graph, dimension)
    provider.validate_factor("vertex", vertex.id, source)
    full = t.Representation.mink(dimension)
    cycle_ports = {
        h.Symbols.half_edge()(half.id, 1)
        for half in graph.native.half_edges
        if half.vertex == vertex.id and half.edge.id in graph.loop_edges
    }
    expression = source.with_lorentz_dimension(dimension)
    for position, slot in enumerate(expression.axes):
        if slot.index in cycle_ports:
            continue
        name = t.TensorName.vector(f"gauge_local_attachment::a{position}")
        values = [S(f"gauge_local_component::a{position}_{i}") for i in range(4)] + [
            E("0")
        ] * (dimension - 4)
        library.register(t.Tensor.dense(name(full), values))
        expression *= name(slot.dual())
    matrix = expression.to_tensor(library).to_dense()
    assert matrix.shape == (dimension, dimension)
    return matrix, provider.extra_variables[0]


def test_local_physical_aligned_spectator_blocks_include_quartic_terms(
    process, controls
):
    graphs = [controls[3, 2], controls[3, 3]]
    quartic = next(
        graph
        for graph in process.loops
        if all(e.spin == 3 for e in graph.edges)
        and any(
            len(split_local_color_terms(v.numerator_expression())) > 1
            for v in graph.native.vertices
        )
        and len(graph.loop_edges) >= 2
    )
    graphs.append(quartic)
    saw_quartic = saw_odd = False
    for graph in graphs:
        vertices = graph.native.subgraph(edges=list(graph.loop_edges)).vertices
        for vertex in vertices:
            for term in split_local_color_terms(vertex.numerator_expression()):
                six, mu = local_cycle_matrix(graph, vertex, term.lorentz, 6)
                seven, _ = local_cycle_matrix(graph, vertex, term.lorentz, 7)
                for coordinate in range(5):
                    assert (
                        six[5, coordinate].expand() == six[coordinate, 5].expand() == 0
                    )
                assert seven[5, 6].expand() == seven[6, 5].expand() == 0
                assert (six[5, 5] - seven[5, 5]).expand() == 0
                assert (seven[5, 5] - seven[6, 6]).expand() == 0
                assert not six[5, 5].expand().contains(mu)
                saw_quartic |= len(term.lorentz.axes) == 4
                saw_odd |= any(
                    six[i, 4].expand().coefficient(mu) != 0
                    or six[4, i].expand().coefficient(mu) != 0
                    for i in range(4)
                )
    assert saw_quartic and saw_odd


def test_weighted_projector_trace_uses_dependency_metric_sign(controls):
    from symbolica.community import tensor as t

    graph = controls[3, 2]
    provider, library = local_library(graph, 6, weighted=True)
    edge = next(
        e for e in graph.native.internal_edges if e.id == provider.weighted_edge
    )
    source = split_local_color_terms(edge.numerator_expression())[0].lorentz
    provider.validate_factor("edge", edge.id, source)
    matrix = provider.materialize("edge", edge.id, source, library)
    metric = t.TensorExpression.g(t.Representation.mink(6))(*matrix.axes).to_tensor()
    normalization = provider._normalizations[edge.id, source.to_expression().expand()]
    trace = (matrix * metric).to_tensor().scalar() / normalization
    assert (trace - (4 - 2 * provider.extra_variables[1])).expand() == 0


def test_source_guards_reject_longitudinal_and_higher_derivative_rules(controls):
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    graph = controls[3, 2]
    provider, _ = local_library(graph, 6, weighted=True)
    edge = next(e for e in graph.native.internal_edges if e.id in graph.loop_edges)
    source = split_local_color_terms(edge.numerator_expression())[0].lorentz
    momentum = h.Symbols.edge_momentum()
    longitudinal = t.TensorExpression(
        momentum(edge.id, source.axes[0].to_expression())
        * momentum(edge.id, source.axes[1].to_expression())
    )
    with pytest.raises(NotImplementedError, match="covariant metric"):
        provider.validate_factor("edge", edge.id, source + longitudinal)
    vertex = graph.native.vertices[0]
    source = split_local_color_terms(vertex.numerator_expression())[0].lorentz
    derivative = t.TensorExpression(1)
    for slot in source.axes:
        derivative *= momentum(edge.id, slot.to_expression())
    with pytest.raises(NotImplementedError, match="templates"):
        provider.validate_factor("vertex", vertex.id, source + derivative)


def test_discarding_odd_local_mu_transitions_loses_closed_even_terms(
    recurrences, monkeypatch
):
    from pyamplicol.loops.evanescent_gauge import BoundYangMillsKernels

    reference = recurrences[3, 2]
    original = BoundYangMillsKernels.materialize

    def wrongly_project_local(self, *args):
        tensor = original(self, *args)
        mu = self.extra_variables[0]
        return tensor.map_components(
            lambda value: ((value + value.replace(mu, -mu)) / 2).expand()
        )

    with monkeypatch.context() as temporary:
        temporary.setattr(BoundYangMillsKernels, "materialize", wrongly_project_local)
        wrong = build_open_loop_recurrence(
            reference.graph,
            routing="family",
            kernel_provider=DimensionalYangMillsKernels(),
        )
    difference = (polynomial(reference) - polynomial(wrong)).expand()
    assert difference != 0 and difference.contains(reference.dag.variables[4])


@pytest.fixture(scope="module")
def native_controls(recurrences):
    from symbolica import E, Expression, Replacement, S

    from pyamplicol.loops.evanescent_gauge import rational_sector_columns
    from pyamplicol.loops.kinematics import build_loop_kinematics

    coupling = S("gauge_native_coupling")
    result = {}

    def saved(values, parameters):
        return Expression.evaluator_multiple(
            values, parameters, n_cores=1, jit_compile=False
        ).save()

    for size in (2, 3):
        recurrence = recurrences[3, size]
        bindings = sample_bindings(recurrence)
        for symbol in recurrence.dag.parameters:
            if symbol.get_name().split("::")[-1] == "GC_10":
                bindings[symbol] *= coupling
        columns = rational_sector_columns(recurrence)
        labels = [
            recurrence.dag.nodes[recurrence.final_node].output_labels[i]
            for i in columns
        ]
        oracle = graph_full_dimension_numerator(
            recurrence, bindings, dimension_value=4 - 2 * recurrence.dag.variables[-1]
        )
        coefficients = {}
        for monomial, coefficient in oracle.coefficient_list(*recurrence.dag.variables):
            [(powers, _)] = monomial.to_polynomial(
                vars=recurrence.dag.variables
            ).coefficient_list(recurrence.dag.variables)
            coefficients[tuple(powers)] = coefficient
        names = [f"c{i}" for i in range(len(labels))]
        sectors = [[], [], []]
        for name, label in zip(names, labels, strict=True):
            terms = [name] + [
                f"q{i}^{power}" for i, power in enumerate(label.powers[:4]) if power
            ]
            if label.powers[4]:
                terms.append(f"mu2^{label.powers[4] // 2}")
            sectors[label.powers[5]].append("*".join(terms))
        kinematics = build_loop_kinematics(recurrence, family_order=True)
        substitutions = [Replacement(k, v) for k, v in bindings.items()]
        result[size] = (
            recurrence,
            columns,
            names,
            ["+".join(s) or "0" for s in sectors],
            saved([bindings[p] for p in recurrence.dag.parameters], [coupling]),
            saved(
                [coefficients.get(label.powers, E("0")) for label in labels], [coupling]
            ),
            saved(
                [
                    x.replace_multiple(substitutions)
                    for x in kinematics.exact_input_expressions
                ],
                [],
            ),
        )
    return result


@pytest.mark.parametrize("size", (2, 3))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "2e-9"),
        ("double-double", None, "2e-25"),
        ("arbitrary", 80, "2e-68"),
    ],
)
def test_dimensional_gauge_remainder_native_epsilon_sectors(
    native_controls, size, arithmetic, precision, tolerance
):
    from decimal import Decimal, localcontext

    from pyamplicol.loops.kernels import NativeCoefficientDAG

    backend = pytest.importorskip("_loop_reduction")
    recurrence, columns, names, sectors, inputs, oracle, kinematics = native_controls[
        size
    ]
    options = {"arithmetic": arithmetic, "precision": precision}
    program = NativeCoefficientDAG(
        recurrence.dag, output_nodes=(recurrence.final_node,), **options
    )
    binding = backend.CoefficientEvaluator(inputs, **options)
    expected_coefficients = backend.CoefficientEvaluator(oracle, **options)
    exact_kinematics = backend.ExactPolynomialEvaluator(kinematics)
    reducer = backend.PolynomialPlan(size, sectors, names, **options)
    with localcontext() as context:
        context.prec = 130
        tiny = (
            Decimal("0.125")
            if arithmetic == "double"
            else Decimal(2) ** -(80 if arithmetic == "double-double" else 180)
        )
        for rows in (1, 5, 0, 33, 1):
            ids = [91 + 7 * i for i in range(rows)]
            source = backend.NativeBatch.from_decimal(
                [[str(1 + tiny * i), "0"] for i in range(rows)],
                rows,
                1,
                row_ids=ids,
                **options,
            )
            actual = program.evaluate_batch(
                binding.evaluate_batch(source)
            ).take_columns(list(columns))
            expected = expected_coefficients.evaluate_batch(source)
            invariant = exact_kinematics.evaluate_decimal(
                [], rows, arithmetic, precision, ids
            )
            shifts = invariant.take_columns(list(range(4 * size)))
            matrix = invariant.take_columns(
                list(range(4 * size, 4 * size + size * size))
            )
            masses = backend.NativeBatch.from_decimal(
                [["0", "0"]] * (size * rows), rows, size, row_ids=ids, **options
            )
            scales = backend.NativeBatch.from_decimal(
                [["1", "0"]] * rows, rows, 1, row_ids=ids, **options
            )
            outputs = [
                reducer.evaluate_batch(shifts, masses, scales, coefficients, matrix)
                for coefficients in (actual, expected)
            ]
            for output in outputs:
                assert output.statuses == [(identity, "success") for identity in ids]
                assert output.values.row_ids == ids
            values = [output.values.to_decimal() for output in outputs]
            for left, right in zip(*values, strict=True):
                for a, b in zip(left, right, strict=True):
                    a, b = Decimal(a), Decimal(b)
                    assert abs(a - b) < Decimal(tolerance) * max(1, abs(b))
            if rows:
                assert any(abs(Decimal(x)) > Decimal("1e-3") for x in values[0][0])
            if rows > 1 and arithmetic != "double":
                axis = max(range(2), key=lambda i: abs(Decimal(values[0][0][i])))
                observed = Decimal(values[0][3][axis]) / Decimal(values[0][0][axis]) - 1
                expected = (1 + tiny) ** 4 - 1
                assert observed != 0 and abs(observed / expected - 1) < Decimal("0.002")
