# SPDX-License-Identifier: 0BSD
"""Complete retained gg source coverage and small independent D-dimensional oracles."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from pyamplicol.loops.direct_kernels import (
    closed_reduction_powers,
    dimensional_kernels,
)
from pyamplicol.loops.evanescent_gauge import DimensionalYangMillsKernels
from pyamplicol.loops.graph_basis import build_color_loop_recurrence
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.kernels import CoefficientLabel
from tools.loop_reference.graph_r2_oracle import (
    closed_polynomial,
    exact_node_coefficients,
    graph_full_dimension_numerator,
    project_external_gluon_colors,
    sample_bindings,
)


@pytest.fixture(scope="module")
def source():
    s = pytest.importorskip("symbolica")
    if not s.is_licensed():
        pytest.skip("licensed dimensional tensor runtime required")
    from symbolica.community import hepkit as h

    model = h.Model.qcd_qed()
    process = build_loop_process(
        model.process(["g", "g"], ["g", "g"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    return model, process


def pattern(graph):
    spins = {edge.spin for edge in graph.edges if edge.id in graph.loop_edges}
    assert len(spins) == 1
    quartic = sum(
        sum(half.vertex == vertex.id for half in graph.native.half_edges) == 4
        for vertex in graph.native.vertices
    )
    return spins.pop(), len(graph.loop_edges), quartic


def four_dimensional_coefficients(recurrence, model, *, dimensional):
    from symbolica import E

    outputs = exact_node_coefficients(
        recurrence, sample_bindings(recurrence, model=model)
    )
    values = {}
    for term in recurrence.terms:
        mapping = (
            closed_reduction_powers(recurrence, term.node)
            if dimensional
            else tuple(
                (*label.powers, 0, 0)
                for label in recurrence.dag.nodes[term.node].output_labels
            )
        )
        color = term.color.to_expression().expand()
        for powers, value in zip(mapping, outputs[term.node], strict=True):
            if powers is None or powers[-2:] != (0, 0):
                continue
            key = color, powers[:4]
            values[key] = (values.get(key, E("0")) + value).expand()
    return {key: value for key, value in values.items() if value != 0}


def test_all_retained_gluon_graphs_match_original_four_dimensional_coefficients(source):
    model, process = source
    counts, terms, largest_rank = Counter(), 0, 0
    for graph in process.loops:
        dimensional = build_color_loop_recurrence(
            graph, routing="family", kernel_provider=dimensional_kernels(graph)
        )
        ordinary = build_color_loop_recurrence(graph, routing="family")
        assert four_dimensional_coefficients(
            dimensional, model, dimensional=True
        ) == four_dimensional_coefficients(ordinary, model, dimensional=False), graph.id
        assert dimensional.dag.direct_r2_certified is False
        counts[pattern(graph)] += 1
        terms += len(dimensional.terms)
        largest_rank = max(largest_rank, *(node.rank for node in dimensional.dag.nodes))
    assert counts == {
        (-1, 2, 0): 3,
        (-1, 3, 0): 12,
        (-1, 4, 0): 6,
        (2, 2, 0): 18,
        (2, 3, 0): 72,
        (2, 4, 0): 36,
        (3, 2, 0): 3,
        (3, 2, 1): 18,
        (3, 2, 2): 3,
        (3, 3, 0): 6,
        (3, 3, 1): 6,
        (3, 4, 0): 3,
    }
    assert len(process.loops) == 186 and terms == 258 and largest_rank == 4


def projected_polynomial(recurrence, model, seed):
    from symbolica import E
    from symbolica.community import tensor as t

    outputs = exact_node_coefficients(
        recurrence, sample_bindings(recurrence, model=model)
    )
    library = t.TensorLibrary.hep_lib_atom()
    return sum(
        (
            closed_polynomial(
                recurrence, outputs[term.node], node=term.node, even=False
            )
            * project_external_gluon_colors(
                term.color, recurrence.graph.native, library, seed
            ).to_tensor(library)[0]
            for term in recurrence.terms
        ),
        E("0"),
    ).expand()


@pytest.mark.parametrize(
    "kind", [(3, 2, 1), (3, 2, 2), (3, 3, 1), (3, 4, 0), (-1, 4, 0)]
)
@pytest.mark.parametrize("seed", [1, 4])
def test_quartic_and_box_cycles_match_unsplit_full_dimensional_source(
    source, kind, seed
):
    model, process = source
    graph = next(graph for graph in process.loops if pattern(graph) == kind)
    recurrence = build_color_loop_recurrence(
        graph, routing="family", kernel_provider=dimensional_kernels(graph)
    )
    actual = projected_polynomial(recurrence, model, seed)
    epsilon = recurrence.dag.variables[-1]
    expected = graph_full_dimension_numerator(
        recurrence,
        sample_bindings(recurrence, model=model),
        dimension_value=4 - 2 * epsilon,
        color_seed=seed,
    )
    assert actual != 0
    assert (actual - expected).expand() == 0


def test_two_quartic_spectator_multiplicity_does_not_depend_on_cut_or_weighted_edge(
    source,
):
    model, process = source
    graph = next(graph for graph in process.loops if pattern(graph) == (3, 2, 2))
    values = []
    for cut in graph.loop_edges:
        for weighted in graph.loop_edges:
            recurrence = build_color_loop_recurrence(
                graph,
                routing="family",
                cut_edge=cut,
                kernel_provider=DimensionalYangMillsKernels(weighted_edge=weighted),
            )
            values.append(projected_polynomial(recurrence, model, 1))
    assert values[0] != 0
    assert all((value - values[0]).expand() == 0 for value in values)


def test_closed_sector_mapping_rejects_open_unweighted_and_odd_gauge_currents(source):
    _, process = source
    graph = next(graph for graph in process.loops if pattern(graph) == (3, 2, 1))
    recurrence = build_color_loop_recurrence(
        graph, routing="family", kernel_provider=dimensional_kernels(graph)
    )
    with pytest.raises(ValueError, match="multi-colour"):
        closed_reduction_powers(recurrence)
    open_node = next(node.id for node in recurrence.dag.nodes if node.axes)
    with pytest.raises(ValueError, match="closed scalar"):
        closed_reduction_powers(recurrence, open_node)
    final = recurrence.terms[0].node
    mapping = closed_reduction_powers(recurrence, final)
    assert mapping and all(item is not None for item in mapping)
    saved = recurrence.dag.nodes[final]
    recurrence.dag.nodes[final] = replace(
        saved, output_labels=(CoefficientLabel(0, (0, 0, 0, 0, 1, 0)),)
    )
    with pytest.raises(ValueError, match="reflection-even"):
        closed_reduction_powers(recurrence, final)
    recurrence.dag.nodes[final] = saved
    unweighted = build_color_loop_recurrence(
        graph,
        routing="family",
        kernel_provider=DimensionalYangMillsKernels(4, dimensional_weight=False),
    )
    with pytest.raises(ValueError, match="certified dimensional closure"):
        closed_reduction_powers(unweighted, unweighted.terms[0].node)
    with pytest.raises(NotImplementedError, match="one native cycle"):
        dimensional_kernels(process.born[0])


def test_closed_fermion_projection_is_explicit_and_survives_scalar_descendants(source):
    from symbolica import E
    from symbolica.community import tensor as t

    _, process = source
    graph = next(
        graph
        for graph in process.loops
        if pattern(graph) == (2, 4, 0)
        and any(edge.particle == "t" for edge in graph.edges)
    )
    recurrence = build_color_loop_recurrence(
        graph, routing="family", kernel_provider=dimensional_kernels(graph)
    )
    final = recurrence.final_nodes[0]
    mapping = closed_reduction_powers(recurrence)
    assert any(powers is None for powers in mapping)
    assert any(powers is not None and powers[-2] > 0 for powers in mapping)
    descendant = recurrence.dag.add(
        "physical-scalar", t.TensorExpression(E("3")).to_tensor(), (final,)
    )
    assert closed_reduction_powers(recurrence, descendant) == mapping
