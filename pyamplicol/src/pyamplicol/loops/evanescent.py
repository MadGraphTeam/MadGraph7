# SPDX-License-Identifier: 0BSD
"""Local single-evanescent-slash kernels for closed vectorlike fermion cycles.

This is a graph-numerator prototype, not process-wide direct-R2 eligibility.
Only source-proven scalar/vector cycle vertices and ordinary Dirac propagators
are admitted. Dependency tensors perform every contraction. Odd mu coefficients
must survive the open recurrence and are selected only after spin closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._internal.physics.symbols import symbols


@dataclass(frozen=True, slots=True)
class ClosedVectorlikeFermionKernels:
    """Reusable unbound provider; each recurrence gets independent validation."""

    def bind(self, graph, loop_components, q):
        from symbolica import E

        if graph.loop_count != 1 or not graph.loop_edges:
            raise NotImplementedError("closed fermion kernels require one native cycle")
        edges = {edge.id: edge for edge in graph.native.internal_edges}
        if any(
            edges[index].particle.spin != 2 or not edges[index].directed
            for index in graph.loop_edges
        ):
            raise NotImplementedError("cycle must contain only directed Dirac fermions")
        cycle = graph.native.subgraph(edges=list(graph.loop_edges))
        vertices = frozenset(vertex.id for vertex in cycle.vertices)
        for vertex in vertices:
            incident = [
                half
                for half in graph.native.half_edges
                if half.vertex == vertex and half.edge.id in graph.loop_edges
            ]
            if len(incident) != 2 or {half.flow for half in incident} != {
                "source",
                "sink",
            }:
                raise NotImplementedError("cycle vertex must join opposite Dirac flows")
        if len(q) != 4 or len(loop_components) != 4:
            raise ValueError("evanescent routing needs four physical loop components")
        sign = loop_components[0].derivative(q[0]).expand()
        if sign not in (E("1"), E("-1")) or any(
            (
                component.derivative(variable) - (sign if row == column else E("0"))
            ).expand()
            != 0
            for row, component in enumerate(loop_components)
            for column, variable in enumerate(q)
        ):
            raise NotImplementedError(
                "physical loop routing must have a common unit sign"
            )
        mu, epsilon = symbols.qualified(
            "loop_evanescent::mu", "loop_evanescent::epsilon"
        )
        return BoundClosedFermionKernels(graph, edges, vertices, sign, (mu, epsilon))


@dataclass(slots=True)
class BoundClosedFermionKernels:
    graph: Any
    edges: dict[int, Any]
    vertices: frozenset[int]
    route_sign: Any
    extra_variables: tuple[Any, Any]
    extra_weights: tuple[int, int] = (1, 0)
    domain: str = "closed-vectorlike-single-evanescent-slash-prototype"
    _extra: dict[tuple[int, Any], Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def _gamma(expression):
        from symbolica.community import tensor as t

        row, column, vector = symbols.qualified(
            "evanescent_row_", "evanescent_column_", "evanescent_vector_"
        )
        gamma = t.TensorName.dirac_gamma().to_expression()
        matches = [
            dict(match) for match in expression.match(gamma(row, column, vector))
        ]
        if len(matches) != 1:
            raise NotImplementedError("local rule must contain exactly one Dirac gamma")
        match = matches[0]
        atom = gamma(match[row], match[column], match[vector])
        slots = tuple(t.TensorExpression(atom).axes)
        if tuple(slot.representation for slot in slots) != (
            t.Representation.bis(4),
            t.Representation.bis(4),
            t.Representation.mink(4),
        ):
            raise NotImplementedError(
                "local rule requires four-dimensional Dirac/vector ports"
            )
        return atom, slots, match[vector]

    @staticmethod
    def _scalar(expression):
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        if (
            not t.TensorExpression(expression).is_scalar
            or h.Symbols.edge_momentum() in expression.get_all_symbols()
        ):
            raise NotImplementedError(
                "local normalization must be a momentum-independent scalar"
            )

    def validate_factor(self, kind, owner_id, lorentz):
        from symbolica import E
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        expression = lorentz.to_expression()
        if kind == "edge" and owner_id in self.graph.loop_edges:
            gamma, slots, vector = self._gamma(expression)
            momentum = h.Symbols.edge_momentum()(owner_id, vector)
            normalization = expression.coefficient(momentum * gamma)
            self._scalar(normalization)
            edge = self.edges[owner_id]
            identity = (
                t.Representation.bis(4)
                .id(slots[0].index, slots[1].index)
                .to_expression()
            )
            expected = normalization * (
                momentum * gamma + edge.particle.mass * identity
            )
            if normalization == 0 or (expression - expected).expand() != 0:
                raise NotImplementedError(
                    "source propagator is not normalization*(slash(P)+m)"
                )
            signature = edge.momentum_signature().loops
            if len(signature) != 1 or signature[0] not in (-1, 1):
                raise NotImplementedError(
                    "fermion propagator must carry one unit loop momentum"
                )
            mu = self.extra_variables[0]
            self._extra[owner_id, expression] = (
                normalization
                * E("1i")
                * signature[0]
                * self.route_sign
                * mu
                * t.TensorExpression.gamma5(4)(slots[0], slots[1])
            )
        elif kind == "vertex" and owner_id in self.vertices:
            spin = t.Representation.bis(4)
            physical = t.Representation.mink(4)
            axes = tuple(lorentz.axes)
            spin_axes = tuple(axis for axis in axes if axis.representation == spin)
            vector_axes = tuple(
                axis for axis in axes if axis.representation == physical
            )
            if (
                len(spin_axes) != 2
                or len(vector_axes) > 1
                or len(axes) != 2 + len(vector_axes)
            ):
                raise NotImplementedError(
                    "cycle vertex needs two spinor ports and at most one vector port"
                )
            if vector_axes:
                basis, _slots, _vector = self._gamma(expression)
            else:
                basis = spin.id(spin_axes[0].index, spin_axes[1].index).to_expression()
            normalization = expression.coefficient(basis)
            self._scalar(normalization)
            if normalization == 0 or (expression - normalization * basis).expand() != 0:
                raise NotImplementedError(
                    "cycle vertex is not a vectorlike gamma or scalar identity"
                )

    def materialize(self, kind, owner_id, lorentz, library):
        if kind == "edge" and owner_id in self.graph.loop_edges:
            key = owner_id, lorentz.to_expression()
            if key not in self._extra:
                raise ValueError(
                    "cycle propagator was not validated before materialization"
                )
            return (lorentz + self._extra[key]).to_tensor(library)
        return lorentz.to_tensor(library)


def closed_even_columns(recurrence, *, rational_only=False):
    """Choose closed even parity; never use this operation on an open current.

    ``rational_only`` selects positive mu powers for this epsilon-free source
    domain. It is not a generic R2 extractor for epsilon-dependent source rules.
    """

    node = recurrence.dag.nodes[recurrence.final_node]
    expected = (
        symbols.qualified("loop_evanescent::mu"),
        symbols.qualified("loop_evanescent::epsilon"),
    )
    if tuple(recurrence.dag.variables[4:]) != expected or node.axes:
        raise ValueError("expected a closed single-evanescent-slash recurrence")
    if any(label.powers[5] for label in node.output_labels):
        raise NotImplementedError(
            "epsilon-dependent sources require explicit sector completion"
        )
    return tuple(
        index
        for index, label in enumerate(node.output_labels)
        if label.powers[4] % 2 == 0 and (not rational_only or label.powers[4] > 0)
    )
