# SPDX-License-Identifier: 0BSD
"""Bounded dimensional kernel selection and closed numerator interpretation.

This dispatch certifies only the local domains of its providers. A process
assembler must cover every contributing graph, retain the same regularization
scheme, and validate the complete result before enabling direct R2 generation.
"""

from __future__ import annotations

from .._internal.physics.symbols import symbols
from .evanescent import BoundClosedFermionKernels, ClosedVectorlikeFermionKernels
from .evanescent_gauge import BoundYangMillsKernels, DimensionalYangMillsKernels


def dimensional_kernels(graph):
    """Select an unbound provider; source-rule validation happens during binding."""
    if graph.loop_count != 1 or not graph.loop_edges:
        raise NotImplementedError("dimensional kernels require one native cycle")
    spins = {edge.spin for edge in graph.edges if edge.id in graph.loop_edges}
    if spins == {2}:
        return ClosedVectorlikeFermionKernels()
    if spins in ({3}, {-1}):
        return DimensionalYangMillsKernels()
    raise NotImplementedError("no certified dimensional provider for this cycle")


def closed_reduction_powers(recurrence, node=None):
    """Map closed output labels to (q0,q1,q2,q3,mu2,epsilon) powers.

    ``None`` projects an odd auxiliary gamma5 trace from the certified closed
    vectorlike fermion representation. It is never a projection of an open
    current. Gauge/ghost closures must already be reflection-even and reject
    odd labels. Scalar descendants, such as a tree-denominator multiplication,
    can be selected explicitly with ``node``. No numerical cancellation or
    small-coefficient threshold participates in this interpretation.
    """

    if node is None:
        if hasattr(recurrence, "final_node"):
            node = recurrence.final_node
        elif len(recurrence.final_nodes) == 1:
            node = recurrence.final_nodes[0]
        else:
            raise ValueError("select a closed node for a multi-colour recurrence")
    current = recurrence.dag.nodes[node]
    expected = (
        symbols.qualified("loop_evanescent::mu"),
        symbols.qualified("loop_evanescent::epsilon"),
    )
    if (
        current.axes
        or tuple(recurrence.dag.variables[4:]) != expected
        or recurrence.dag.variable_weights != (1, 1, 1, 1, 1, 0)
        or any(label.component != 0 for label in current.output_labels)
    ):
        raise ValueError("expected a closed scalar with the dimensional layout")
    provider = recurrence.kernel_provider
    fermion = isinstance(provider, BoundClosedFermionKernels)
    gauge = isinstance(provider, BoundYangMillsKernels)
    if not fermion and not (
        gauge and provider.dimension == 6 and provider.dimensional_weight
    ):
        raise ValueError("no certified dimensional closure for this provider")
    result = []
    for label in current.output_labels:
        q0, q1, q2, q3, mu, epsilon = label.powers
        if epsilon > (0 if fermion else 1):
            raise NotImplementedError("unsupported number of dimensional trace factors")
        if mu % 2:
            if not fermion:
                raise ValueError("closed gauge/ghost numerator is not reflection-even")
            result.append(None)
        else:
            result.append((q0, q1, q2, q3, mu // 2, epsilon))
    return tuple(result)
