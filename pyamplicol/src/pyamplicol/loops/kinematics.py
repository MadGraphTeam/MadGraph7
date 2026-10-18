# SPDX-License-Identifier: 0BSD
"""Exact graph kinematics prepared separately from numerical coefficient work.

HepKit provides momentum routing, and its tensor library supplies the Minkowski
pairing. Symbolica constructs polynomial evaluators. Invariant matrices must be
evaluated from original exact inputs before casting to the selected arithmetic;
forming them from already rounded shifts can change a scalar integral's branch.
This module does not snap nearly on-shell invariants or infer complex masses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exact import require_exact_numbers
from .recurrence import momentum_component_library


@dataclass(frozen=True, slots=True)
class LoopKinematicPlan:
    """Denominator bindings in native graph edge order, with explicit routing.

    ``S[i,j] = (V[i]-V[j])**2`` does not include masses. The exact evaluator
    returns flattened shifts followed by the row-major S matrix. Masses are
    separate model expressions; renormalization scale is a runtime input.
    """

    graph_id: str
    routing: str
    denominator_order: str
    edge_ids: tuple[int, ...]
    momentum_parameters: tuple[Any, ...]
    shifts: tuple[tuple[Any, ...], ...]
    masses_squared: tuple[Any, ...]
    s_matrix: tuple[Any, ...]

    @property
    def propagators(self) -> int:
        return len(self.edge_ids)

    @property
    def exact_input_expressions(self) -> tuple[Any, ...]:
        return (
            tuple(component for vector in self.shifts for component in vector)
            + self.s_matrix
        )

    def evaluator_bytes(self) -> bytes:
        from symbolica import Expression

        return Expression.evaluator_multiple(
            self.exact_input_expressions,
            self.momentum_parameters,
            n_cores=1,
            jit_compile=False,
        ).save()


def build_loop_kinematics(
    recurrence: Any, *, family_order: bool = False
) -> LoopKinematicPlan:
    """Bind the same graph/family loop variable used by a numerator recurrence.

    For ``p_i = a_i*q+b_i`` with ``a_i=+/-1``, the denominator is
    ``(q+a_i*b_i)**2-m_i**2-mu2``. Any other dependence is unsupported, never
    silently rescaled. Repeated cycle denominators are rejected explicitly.
    """
    from symbolica import E, Replacement
    from symbolica.community import tensor as t

    graph = recurrence.graph
    variables = recurrence.dag.momentum_variables
    momenta = recurrence.momentum_parameters
    if graph.loop_count != 1 or len(variables) != 4:
        raise ValueError("kinematic binding requires one four-component loop momentum")
    library = momentum_component_library(graph, (recurrence.loop_components,), momenta)
    shifts, masses, edge_ids, denominators = [], [], [], set()
    zero_loop = [Replacement(variable, E("0")) for variable in variables]
    edges = [
        edge for edge in graph.native.internal_edges if edge.id in graph.loop_edges
    ]
    if family_order:
        if recurrence.routing != "family" or graph.family is None:
            raise ValueError(
                "family denominator order requires verified family momentum routing"
            )
        mapping = graph.family.denominator_map
        if sorted(mapping) != list(range(len(edges))):
            raise NotImplementedError(
                "family binding requires a bijective denominator map"
            )
        edges = [edges[i] for i in sorted(range(len(edges)), key=mapping.__getitem__)]
    for edge in edges:
        vector = edge.momentum_expression(dimension=4).to_tensor(library).to_dense()
        components = tuple(vector[i] for i in range(4))
        direction = components[0].derivative(variables[0])
        if direction not in (E("-1"), E("1")):
            raise NotImplementedError(
                "loop propagator needs a unit loop-momentum coefficient"
            )
        shift = tuple(
            (direction * component.replace_multiple(zero_loop)).expand()
            for component in components
        )
        for component, variable, value in zip(
            components, variables, shift, strict=True
        ):
            if (component - direction * (variable + value)).expand() != 0:
                raise NotImplementedError(
                    "loop propagator momentum is not affine with unit coefficient"
                )
        mass = edge.particle.mass**2
        for expression in (*shift, mass):
            require_exact_numbers(expression)
        key = (*shift, mass.expand())
        if key in denominators:
            raise NotImplementedError(
                "repeated loop propagators require reducer support"
            )
        denominators.add(key)
        edge_ids.append(edge.id)
        shifts.append(shift)
        masses.append(mass)

    if len(edge_ids) != len(graph.loop_edges):
        raise ValueError("kinematic binding did not cover the native cycle")
    matrix = []
    for left in shifts:
        for right in shifts:
            difference = t.Tensor.dense(
                t.TensorName.vector("loop_kinematics::difference")(
                    t.Representation.mink(4)
                ),
                [a - b for a, b in zip(left, right, strict=True)],
            )
            matrix.append(difference.dot(difference).to_tensor()[0].expand())
    return LoopKinematicPlan(
        graph.id,
        recurrence.routing,
        "family" if family_order else "graph",
        tuple(edge_ids),
        tuple(component for vector in momenta for component in vector),
        tuple(shifts),
        tuple(masses),
        tuple(matrix),
    )
