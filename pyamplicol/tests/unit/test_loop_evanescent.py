# SPDX-License-Identifier: 0BSD
"""Actual native quark-cycle templates and independent dimensional trace checks."""

from __future__ import annotations

import pytest

from pyamplicol.loops.evanescent import (
    ClosedVectorlikeFermionKernels,
    closed_even_columns,
)
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.kinematics import build_loop_kinematics
from pyamplicol.loops.recurrence import build_open_loop_recurrence, separate_local_color
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
        pytest.skip("licensed tensor/graph runtime required")
    from symbolica.community import hepkit as h

    return build_loop_process(
        h.Model.qcd().process(["g", "g"], ["g", "g"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )


def attachment_word(graph):
    """Label a dependency-owned cycle traversal by its attached external sets."""
    cut = min(graph.loop_edges)
    edge = next(e for e in graph.native.edges if e.id == cut)
    cycle = graph.native.subgraph(edges=list(graph.loop_edges))
    path = cycle - graph.native.subgraph(edges=[cut])
    order = tuple(v.id for v in path.depth_first_traverse(edge.source).nodes)
    forest = graph.native.filter(edge=lambda e: e.id not in graph.loop_edges)
    word = []
    for vertex in order:
        nodes = {v.id for v in forest.depth_first_traverse(vertex).nodes}
        word.append(
            tuple(
                sorted(
                    e.external_index
                    for e in graph.native.external_edges
                    if e.source in nodes or e.target in nodes
                )
            )
        )
    return tuple(word)


def cyclic_word(word):
    return min(word[i:] + word[:i] for i in range(len(word)))


@pytest.fixture(scope="module")
def controls(process):
    result = {}
    for size in (2, 3, 4):
        candidates = [
            g
            for g in process.loops
            if len(g.loop_edges) == size
            and all(e.spin == 2 and e.pdg == 6 for e in g.edges if e.id in g.loop_edges)
        ]
        first = candidates[0]
        if size == 2:
            # The reverse two-letter cyclic word is equivalent; exercise both cuts.
            second = first
        else:
            reverse = cyclic_word(tuple(reversed(attachment_word(first))))
            second = next(
                g
                for g in candidates
                if g.id != first.id and cyclic_word(attachment_word(g)) == reverse
            )
        result[size] = (first, second)
    return result


@pytest.fixture(scope="module")
def recurrences(controls):
    result = {}
    for size, (left, right) in controls.items():
        result[size, 0] = build_open_loop_recurrence(
            left, kernel_provider=ClosedVectorlikeFermionKernels()
        )
        result[size, 1] = build_open_loop_recurrence(
            right,
            cut_edge=max(right.loop_edges) if size == 2 else None,
            routing="family",
            kernel_provider=ClosedVectorlikeFermionKernels(),
        )
    return result


@pytest.mark.parametrize("size", (2, 3, 4))
@pytest.mark.parametrize("orientation", (0, 1))
def test_graph_local_even_closure_matches_source_d_trace(
    recurrences, size, orientation
):
    recurrence = recurrences[size, orientation]
    bindings = sample_bindings(recurrence)
    coefficients = exact_graph_coefficients(recurrence, bindings)
    actual = closed_polynomial(recurrence, coefficients)
    expected = graph_full_dimension_numerator(recurrence, bindings)
    assert (actual - expected).expand() == 0
    assert len(recurrence.graph.bridge_edges) == 4 - size
    assert len(recurrence.dag.momentum_variables) == 4
    assert len(recurrence.dag.variables) == 6
    assert recurrence.dag.nodes[recurrence.final_node].rank == size
    columns = closed_even_columns(recurrence, rational_only=True)
    assert columns and any(coefficients[i] != 0 for i in columns)
    if size == 4:
        assert any(
            recurrence.dag.nodes[recurrence.final_node].output_labels[i].powers[4] == 4
            and coefficients[i] != 0
            for i in columns
        )
    plan = build_loop_kinematics(recurrence, family_order=orientation == 1)
    assert plan.propagators == size
    assert all(
        not value.contains(recurrence.dag.variables[4])
        for value in plan.exact_input_expressions
    )
    assert recurrence.dag.direct_r2_certified is False


@pytest.mark.parametrize("size", (3, 4))
def test_native_family_reversal_changes_odd_extra_slash_too(recurrences, size):
    from symbolica import E, Replacement

    family = recurrences[size, 1]
    assert family.kernel_provider.route_sign == E("-1")
    original = build_open_loop_recurrence(
        family.graph, kernel_provider=ClosedVectorlikeFermionKernels()
    )
    bindings = sample_bindings(original)
    unprojected = closed_polynomial(
        original, exact_graph_coefficients(original, bindings), even=False
    )
    actual = closed_polynomial(
        family, exact_graph_coefficients(family, bindings), even=False
    )
    physical = [Replacement(symbol, value) for symbol, value in bindings.items()]
    mu = original.dag.variables[4]
    changes = [
        Replacement(q, mapped.replace_multiple(physical))
        for q, mapped in zip(
            original.dag.momentum_variables, family.loop_components, strict=True
        )
    ]
    expected = unprojected.replace_multiple([*changes, Replacement(mu, -mu)]).expand()
    assert (actual - expected).expand() == 0
    # A closed vectorlike trace can already be even. Check the sign at the
    # open local factor, where dropping odd parity would hide a routing error.
    assert original.kernel_provider._extra
    for key, extra in original.kernel_provider._extra.items():
        assert extra.to_expression() != 0
        assert (
            family.kernel_provider._extra[key].to_expression() + extra.to_expression()
        ).expand() == 0


def test_provider_rejects_unproven_cycle_species_and_chiral_templates(
    process, controls
):
    from symbolica import E, S
    from symbolica.community import tensor as t

    q = tuple(S(f"evanescent_reject_q{i}") for i in range(4))
    unsuitable = next(
        g
        for g in process.loops
        if any(e.spin != 2 for e in g.edges if e.id in g.loop_edges)
    )
    with pytest.raises(NotImplementedError, match="Dirac fermions"):
        ClosedVectorlikeFermionKernels().bind(unsuitable, q, q)
    graph = controls[4][0]
    provider = ClosedVectorlikeFermionKernels().bind(graph, q, q)
    vertex = graph.native.vertices[0]
    lorentz, _ = separate_local_color(vertex.numerator_expression())
    _gamma, slots, _vector = provider._gamma(lorentz.to_expression())
    # A physical axial admixture is outside this auxiliary-matrix certificate.
    axial = t.TensorExpression.gamma5(4)(
        slots[0], S("axial_middle")
    ) * t.TensorExpression.dirac_gamma(4)(S("axial_middle"), slots[1], slots[2])
    with pytest.raises(NotImplementedError):
        provider.validate_factor("vertex", vertex.id, lorentz + axial)
    edge = next(e for e in graph.native.internal_edges if e.id in graph.loop_edges)
    numerator, _ = separate_local_color(edge.numerator_expression())
    with pytest.raises(NotImplementedError, match="source propagator"):
        provider.validate_factor(
            "edge",
            edge.id,
            numerator + E("1") * t.TensorExpression.gamma5(4)(*numerator.axes),
        )
    with pytest.raises(NotImplementedError, match="unit sign"):
        ClosedVectorlikeFermionKernels().bind(graph, tuple(2 * x for x in q), q)


@pytest.fixture(scope="module")
def native_controls(recurrences):
    from symbolica import E, Expression, Replacement, S

    result = {}
    mass, coupling = S("graph_native_mass", "graph_native_coupling")
    for size in (2, 4):
        recurrence = recurrences[size, 1]
        bindings = sample_bindings(recurrence)
        for symbol in recurrence.dag.parameters:
            name = symbol.get_name().split("::")[-1]
            if name == "MT":
                bindings[symbol] = mass
            elif name in ("GC_10", "GC_11"):
                bindings[symbol] *= coupling
        columns = closed_even_columns(recurrence, rational_only=True)
        labels = [
            recurrence.dag.nodes[recurrence.final_node].output_labels[i]
            for i in columns
        ]
        oracle = graph_full_dimension_numerator(recurrence, bindings)
        oracle_coefficients = {}
        for monomial, coefficient in oracle.coefficient_list(*recurrence.dag.variables):
            [(powers, _)] = monomial.to_polynomial(
                vars=recurrence.dag.variables
            ).coefficient_list(recurrence.dag.variables)
            oracle_coefficients[tuple(powers)] = coefficient

        def saved(values, parameters):
            return Expression.evaluator_multiple(
                values, parameters, n_cores=1, jit_compile=False
            ).save()

        kinematics = build_loop_kinematics(recurrence, family_order=True)
        substitutions = [Replacement(k, v) for k, v in bindings.items()]
        names = [f"c{i}" for i in range(len(labels))]
        terms = []
        for name, label in zip(names, labels, strict=True):
            factors = [name]
            factors.extend(
                f"q{component}^{power}"
                for component, power in enumerate(label.powers[:4])
                if power
            )
            factors.append(f"mu2^{label.powers[4] // 2}")
            terms.append("*".join(factors))
        result[size] = (
            recurrence,
            columns,
            names,
            "+".join(terms),
            saved([bindings[p] for p in recurrence.dag.parameters], [mass, coupling]),
            saved(
                [oracle_coefficients.get(label.powers, E("0")) for label in labels],
                [mass, coupling],
            ),
            saved(
                [
                    x.replace_multiple(substitutions)
                    for x in kinematics.exact_input_expressions
                ],
                [],
            ),
            saved(
                [x.replace_multiple(substitutions) for x in kinematics.masses_squared],
                [mass, coupling],
            ),
        )
    return result


@pytest.mark.parametrize("size", (2, 4))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "2e-9"),
        ("double-double", None, "2e-25"),
        ("arbitrary", 80, "2e-68"),
    ],
)
def test_graph_local_r2_native_reduction_matches_full_trace(
    native_controls, size, arithmetic, precision, tolerance
):
    from decimal import Decimal, localcontext

    from pyamplicol.loops.kernels import NativeCoefficientDAG

    backend = pytest.importorskip("_loop_reduction")
    recurrence, columns, names, polynomial, inputs, oracle, kinematics, masses = (
        native_controls[size]
    )
    options = {"arithmetic": arithmetic, "precision": precision}
    program = NativeCoefficientDAG(
        recurrence.dag, output_nodes=(recurrence.final_node,), **options
    )
    binding = backend.CoefficientEvaluator(inputs, **options)
    expected_coefficients = backend.CoefficientEvaluator(oracle, **options)
    mass_binding = backend.CoefficientEvaluator(masses, **options)
    exact_kinematics = backend.ExactPolynomialEvaluator(kinematics)
    reducer = backend.PolynomialPlan(size, [polynomial, "0", "0"], names, **options)
    with localcontext() as context:
        context.prec = 130
        coupling = str(
            1 + Decimal(2) ** -(80 if arithmetic == "double-double" else 180)
        )
        for rows in (1, 5, 0, 33, 1):
            ids = [101 + 9 * i for i in range(rows)]
            source = backend.NativeBatch.from_decimal(
                [
                    pair
                    for i in range(rows)
                    for pair in [["2", "-0.1" if i % 2 else "0"], [coupling, "0"]]
                ],
                rows,
                2,
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
            masses_squared = mass_binding.evaluate_batch(source)
            scales = backend.NativeBatch.from_decimal(
                [["1", "0"]] * rows, rows, 1, row_ids=ids, **options
            )
            outputs = [
                reducer.evaluate_batch(
                    shifts, masses_squared, scales, coefficients, matrix
                )
                for coefficients in (actual, expected)
            ]
            for output in outputs:
                assert output.statuses == [(identity, "success") for identity in ids]
                assert output.values.row_ids == ids
            values = [output.values.to_decimal() for output in outputs]
            for index, (left, right) in enumerate(zip(*values, strict=True)):
                for a, b in zip(left, right, strict=True):
                    a, b = Decimal(a), Decimal(b)
                    assert abs(a - b) < Decimal(tolerance) * max(1, abs(b))
                    if index % 3:
                        assert abs(a) < Decimal(tolerance)
            if rows:
                assert any(abs(Decimal(x)) > Decimal("1e-3") for x in values[0][0])
