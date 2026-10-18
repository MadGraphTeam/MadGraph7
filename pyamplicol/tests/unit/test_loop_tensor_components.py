# SPDX-License-Identifier: 0BSD
"""Exact 4D component conventions, independent of a dimensional prescription."""

import pytest


@pytest.fixture(scope="module")
def components():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("exact native tensor components require licensed Symbolica")
    from symbolica.community import tensor

    return symbolica, tensor


def test_native_epsilon_orientation_metric_and_exact_components(components):
    symbolica, t = components
    from pyamplicol.loops.tensor_components import physical_tensor_library

    labels = symbolica.S(*(f"epsilon_test::{label}" for label in "abcd"))
    epsilon = t.TensorExpression.levi_civita(t.Representation.mink(4))(*labels)
    value = epsilon.to_tensor(physical_tensor_library())
    slots = [t.Representation.mink(4)(label) for label in labels]
    value = value.permute_axes([list(value.axes).index(slot) for slot in slots])
    assert value[0, 1, 2, 3] == -symbolica.E("1i")
    assert value[1, 0, 2, 3] == symbolica.E("1i")
    assert value[0, 1, 2, 2] == 0
    assert all(
        component in (0, symbolica.E("1i"), -symbolica.E("1i")) for component in value
    )
    assert sum(component != 0 for component in value) == 24
    # Includes all four (+---) contraction metrics and Idenso's native -i.
    assert (epsilon * epsilon).to_tensor(physical_tensor_library())[0] == 24


def test_routed_component_libraries_keep_momenta_local(components):
    symbolica, t = components
    from symbolica.community import hepkit as h

    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.recurrence import momentum_component_library

    process = build_loop_process(
        h.Model.phi4().process(["phi", "phi"], ["phi", "phi"]),
        selected_corrections=["SCALAR"],
        perturbative_orders={"SCALAR": 1},
    )
    graph = process.loops[0]
    q = tuple(symbolica.S(*(f"epsilon_test::q{i}" for i in range(4))))
    zero = (symbolica.E("0"),) * 4
    first = momentum_component_library(graph, (q,), (zero,) * 4)
    second = momentum_component_library(
        graph, (tuple(2 * value for value in q),), (zero,) * 4
    )
    edge = next(
        edge for edge in graph.native.edges if any(edge.momentum_signature().loops)
    )
    name = t.TensorName(h.Symbols.edge_momentum().get_name())
    vector = name(edge.id, t.Representation.mink(4)(symbolica.S("epsilon_test::p")))
    a, b = vector.to_tensor(first), vector.to_tensor(second)
    assert list(b) == [2 * value for value in a]
    assert list(vector.to_tensor(first)) == list(a)
    epsilon = t.TensorExpression.levi_civita(t.Representation.mink(4))(0, 1, 2, 3)
    assert list(epsilon.to_tensor(first)) == list(epsilon.to_tensor(second))
