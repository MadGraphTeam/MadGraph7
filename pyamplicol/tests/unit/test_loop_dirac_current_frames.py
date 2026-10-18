# SPDX-License-Identifier: 0BSD
"""Directed FFS/propagator/FFV controls, separate from raw diagonal vertices."""

from types import SimpleNamespace

import pytest


def native_chain_fixture():
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    # Local factors captured from the actual Gmu nu/nu/charged-Goldstone graph.
    # They use native typed tensors and the unchanged source Dirac propagator.
    def hedge(index):
        return h.Symbols.half_edge()(index, 1)

    dummy = E("dirac_current_test::vertex_dummy")
    qdummy = E("dirac_current_test::edge_dummy")
    graph = SimpleNamespace(
        edges=tuple(
            SimpleNamespace(id=index, spin=spin)
            for index, spin in ((2, 2), (3, 2), (4, 1), (5, 2), (6, 2), (7, 3))
        ),
        half_edges=(
            (2, 1, 2, "source"),
            (3, 0, 3, "source"),
            (4, 1, 4, "source"),
            (5, 0, 4, "sink"),
            (6, 3, 5, "source"),
            (7, 0, 5, "sink"),
            (8, 1, 6, "source"),
            (9, 3, 6, "sink"),
            (11, 3, 7, "sink"),
        ),
    )
    scalar = (
        t.TensorExpression.projm(4)(hedge(3), hedge(7)),
        t.TensorExpression.projp(4)(hedge(8), hedge(2)),
    )
    vector = t.TensorExpression.dirac_gamma(4)(
        hedge(6), dummy, hedge(11)
    ) * t.TensorExpression.projm(4)(dummy, hedge(9))
    propagators = tuple(
        -E("1i")
        * t.TensorName(h.Symbols.edge_momentum().get_name())(
            edge, t.Representation.mink(4)(qdummy)
        )
        * t.TensorExpression.dirac_gamma(4)(hedge(a), hedge(b), qdummy)
        for edge, a, b in ((5, 6, 7), (6, 8, 9))
    )
    return graph, scalar, vector, propagators


def evaluate_chain(expression, edge, momentum, states):
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    library = t.TensorLibrary.hep_lib_atom()
    library.register(
        t.Tensor.dense(
            t.TensorName(h.Symbols.edge_momentum().get_name())(
                edge, t.Representation.mink(4)
            ),
            momentum,
        )
    )
    for half, components in states.items():
        axis = next(
            a for a in expression.axes if a.index == h.Symbols.half_edge()(half, 1)
        ).dual()
        name = t.TensorName(f"dirac_current_test::state{half}")
        library.register(t.Tensor.dense(name(axis.representation), components))
        expression *= name(axis)
    result = expression.to_tensor(library)
    assert not result.axes
    return result[0] / (momentum[0] ** 2 - sum(p * p for p in momentum[1:]))


@pytest.mark.parametrize("orientation", (0, 1))
def test_ffs_propagator_chiral_vector_matches_directed_aloha(orientation):
    pytest.importorskip("symbolica")
    from symbolica import E

    from pyamplicol.loops.vertex_frames import native_physical_vertex

    graph, scalars, vector, propagators = native_chain_fixture()
    q = tuple(map(E, ("3+1i/5", "1/3-1i/7", "-2/5+1i/11", "4/7-1i/13")))
    incoming = tuple(map(E, ("2/3+1i/5", "-3/7+1i/11", "5/13-2i/7", "-1/3-4i/9")))
    other = tuple(map(E, ("-4/7+3i/5", "1/7-2i/11", "4/13+1i/7", "-2/3+1i/9")))
    polarization = tuple(map(E, ("1/7+2i/3", "-2/5+1i/7", "3/11-1i/13", "-2/17+1i/19")))
    # Unmodified ALOHA FFS3_2 -> FFV2_0 and FFS1_1 -> FFV2_0 respectively.
    # Off-shell F1/F2 outputs attach to the opposite HELAS vertex argument;
    # the edge momentum reverses for the first charge flow. Couplings are one,
    # and only the final amplitude routine's common -i phase is removed.
    expected = (
        complex(-0.1094144491674145, -0.006340164187968814),
        complex(0.010726810530240573, 0.10258088316102629),
    )[orientation]
    input_half, other_half, edge, sign = ((3, 9, 5, -1), (2, 6, 6, 1))[orientation]
    states = {input_half: incoming, other_half: other, 11: polarization}
    momentum = tuple(sign * p for p in q)
    mapped_scalar = native_physical_vertex(graph, orientation, scalars[orientation])
    mapped_vector = native_physical_vertex(graph, 3, vector)
    actual = evaluate_chain(
        mapped_scalar * propagators[orientation] * mapped_vector, edge, momentum, states
    )
    assert complex(actual.evaluate({})) == pytest.approx(expected, abs=2e-15)
    # The previous endpoint-only FFS map passes diagonal _0 controls but kills
    # this physical chiral chain, making this a non-vacuous regression.
    previous = evaluate_chain(
        scalars[orientation] * propagators[orientation] * mapped_vector,
        edge,
        momentum,
        states,
    )
    assert previous == 0
    assert (
        native_physical_vertex(graph, orientation, mapped_scalar)
        == scalars[orientation]
    )
