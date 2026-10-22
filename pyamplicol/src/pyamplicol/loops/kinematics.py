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

from .._internal.physics.symbols import symbols
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
    kinematic_context: str = "raw"
    mass_expressions: tuple[Any, ...] = ()
    on_shell_bindings: tuple[tuple[int, int], ...] = ()

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


def _cycle_order(edges):
    """Positions of the one-loop cycle edges in traversal order (by incidence)."""
    incident = {}
    for position, edge in enumerate(edges):
        for vertex in {edge.source, edge.target}:
            incident.setdefault(vertex, []).append(position)
    if len(edges) < 3:
        return list(range(len(edges)))
    if any(len(positions) != 2 for positions in incident.values()):
        raise ValueError("loop edges do not form a simple cycle")
    order, vertex = [0], edges[0].target
    while len(order) < len(edges):
        position = next(p for p in incident[vertex] if p != order[-1])
        if position == order[0]:
            raise ValueError("loop edges do not form one simple cycle")
        order.append(position)
        edge = edges[position]
        vertex = edge.target if edge.source == vertex else edge.source
    return order


def build_loop_kinematics(
    recurrence: Any,
    *,
    family_order: bool = False,
    physical: bool = False,
    model_expressions: Any = None,
) -> LoopKinematicPlan:
    """Bind the same graph/family loop variable used by a numerator recurrence.

    For ``p_i = a_i*q+b_i`` with ``a_i=+/-1``, the denominator is
    ``(q+a_i*b_i)**2-m_i**2-mu2``. Any other dependence is unsupported, never
    silently rescaled. Repeated cycle denominators are rejected explicitly.
    ``family_order`` binds the family denominators in loop-cycle order,
    normalized on family indices; otherwise the native graph edge order is kept.
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
    shift_signatures = []
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
        # Family denominators follow the loop cycle: every box (and every box
        # pinched from a larger cycle) reaches the masters with consecutive
        # propagators adjacent, as in the conventional D0 argument order.
        # oneloopmaster 09b2c74 misclassified a soft-divergent massive box in a
        # non-adjacent labelling (finite in binary64, NaN in DoubleFloat);
        # 8ec256fc relabels D0 inside the numerical entry points pyAmpliCol
        # uses (NativeFloat, NinjaRS master callbacks). The order is kept.
        # The cycle order is unique up to rotation and reflection, so it is
        # normalized on family indices and identical for all family graphs.
        cycle = [mapping[i] for i in _cycle_order(edges)]
        start = cycle.index(0)
        forward = cycle[start:] + cycle[:start]
        order = min(forward, forward[:1] + forward[:0:-1])
        by_index = {mapping[i]: edge for i, edge in enumerate(edges)}
        edges = [by_index[index] for index in order]
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
        signature = edge.momentum_signature()
        sign = 1 if direction == E("1") else -1
        shift_signatures.append(tuple(sign * c for c in signature.external))

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
    mass_expressions, on_shell = list(masses), []
    if physical:
        # Dependency-provided routing signatures include the eliminated external
        # momentum. Match these exact identities, never sampled component values.
        external = {}
        for edge in graph.native.external_edges:
            signature = edge.momentum_signature()
            if any(signature.loops):
                raise ValueError("external momentum unexpectedly depends on the loop")
            key = tuple(signature.external)
            if not any(key):
                raise ValueError("external momentum has an empty routing signature")
            for signed in (key, tuple(-c for c in key)):
                mass = edge.particle.mass**2
                if signed in external and external[signed] != mass:
                    raise ValueError("external routing identifies incompatible masses")
                external[signed] = mass

        replacements = [
            Replacement(symbols.qualified(name), value)
            for name, value in (model_expressions or {}).items()
        ]

        def normalized(value):
            return value.replace_multiple(replacements).expand()

        normalized_masses = [normalized(mass) for mass in mass_expressions]
        n = len(shifts)
        for i, left in enumerate(shift_signatures):
            for j, right in enumerate(shift_signatures):
                if i == j:
                    continue
                key = tuple(a - b for a, b in zip(left, right, strict=True))
                mass = external.get(key)
                if mass is None:
                    continue
                require_exact_numbers(mass)
                identity = normalized(mass)
                if identity in normalized_masses:
                    column = normalized_masses.index(identity)
                else:
                    column = len(mass_expressions)
                    mass_expressions.append(mass)
                    normalized_masses.append(identity)
                on_shell.append((i * n + j, column))
    return LoopKinematicPlan(
        graph.id,
        recurrence.routing,
        "family" if family_order else "graph",
        tuple(edge_ids),
        tuple(component for vector in momenta for component in vector),
        tuple(shifts),
        tuple(masses),
        tuple(matrix),
        "physical" if physical else "raw",
        tuple(mass_expressions),
        tuple(on_shell),
    )
