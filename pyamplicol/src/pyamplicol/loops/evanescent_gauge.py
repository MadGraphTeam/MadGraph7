# SPDX-License-Identifier: 0BSD
"""Bounded local dimensional continuation of massless Yang--Mills cycles.

The dependency supplies every metric, tensor contraction and polynomial rule.
Six explicit coordinates carry four physical directions, the common extra
momentum and one transverse spectator. One source propagator weights the last
state by D-5. This is an internal numerator prototype, not process eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DimensionalYangMillsKernels:
    dimension: int = 6
    dimensional_weight: bool = True
    weighted_edge: int | None = None

    def bind(self, graph, loop_components, q):
        from symbolica import E, S

        if self.dimension not in (4, 5, 6, 7):
            raise ValueError("prototype component dimensions are four through seven")
        if self.dimensional_weight and self.dimension != 6:
            raise ValueError("dimensional spectator weight requires six coordinates")
        if graph.loop_count != 1 or not graph.loop_edges:
            raise NotImplementedError("Yang--Mills kernels require one native cycle")
        edges = {e.id: e for e in graph.native.edges}
        species = {edges[i].particle.spin for i in graph.loop_edges}
        if species not in ({3}, {-1}):
            raise NotImplementedError("cycle must be purely vector or purely ghost")
        if any(
            e.particle.mass != 0
            or (e.id not in graph.loop_edges and e.particle.spin != 3)
            for e in edges.values()
        ):
            raise NotImplementedError(
                "only massless vector tree attachments are certified"
            )
        if len(q) != 4 or len(loop_components) != 4:
            raise ValueError("four physical loop components required")
        sign = loop_components[0].derivative(q[0]).expand()
        if sign not in (E("1"), E("-1")) or any(
            (component.derivative(variable) - (sign if i == j else E("0"))).expand()
            != 0
            for i, component in enumerate(loop_components)
            for j, variable in enumerate(q)
        ):
            raise NotImplementedError("loop routing needs one common unit sign")
        for edge in edges.values():
            signature = edge.momentum_signature().loops
            if len(signature) != 1 or signature[0] not in (-1, 0, 1):
                raise NotImplementedError("unsupported source momentum signature")
            if edge.id not in graph.loop_edges and signature[0]:
                raise NotImplementedError(
                    "tree attachments must carry physical momenta"
                )
        chosen = (
            min(graph.loop_edges) if self.weighted_edge is None else self.weighted_edge
        )
        if chosen not in graph.loop_edges:
            raise ValueError("spectator weight must belong to the native cycle")
        return BoundYangMillsKernels(
            graph,
            self.dimension,
            self.dimensional_weight,
            chosen,
            sign,
            species == {-1},
            (S("loop_evanescent::mu"), S("loop_evanescent::epsilon")),
        )


@dataclass(slots=True)
class BoundYangMillsKernels:
    graph: Any
    dimension: int
    dimensional_weight: bool
    weighted_edge: int
    route_sign: Any
    ghost_cycle: bool
    extra_variables: tuple[Any, Any]
    extra_weights: tuple[int, int] = (1, 0)
    domain: str = "massless-physical-attachment-yang-mills-prototype"
    _normalizations: dict[tuple[int, Any], Any] = field(
        default_factory=dict, repr=False
    )
    _prepared: bool = field(default=False, repr=False)

    @staticmethod
    def _scalar(expression):
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        if (
            not t.TensorExpression(expression).is_scalar
            or h.Symbols.edge_momentum() in expression.get_all_symbols()
        ):
            raise NotImplementedError(
                "source coefficient must be a momentum-independent scalar"
            )

    def validate_factor(self, kind, owner_id, lorentz):
        from symbolica import E
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        axes = tuple(lorentz.axes)
        if any(a.representation != t.Representation.mink(4) for a in axes):
            raise NotImplementedError(
                "Yang--Mills source ports must be physical vectors"
            )
        expression = lorentz.to_expression().expand()
        metric = t.TensorExpression.g(t.Representation.mink(4))
        if kind == "edge":
            source = next(
                e for e in self.graph.native.internal_edges if e.id == owner_id
            )
            if source.particle.spin == -1 and len(axes) == 0:
                self._scalar(expression)
                return
            if source.particle.spin != 3 or len(axes) != 2:
                raise NotImplementedError("vector propagator needs two ports")
            basis = metric(*axes).to_expression()
            normalization = expression.coefficient(basis)
            self._scalar(normalization)
            if normalization == 0 or (expression - normalization * basis).expand() != 0:
                raise NotImplementedError(
                    "only covariant metric propagators are certified"
                )
            self._normalizations[owner_id, expression] = normalization
            return
        if kind != "vertex" or len(axes) not in (1, 3, 4):
            raise NotImplementedError("unsupported Yang--Mills source vertex")
        incident = {
            half.edge.id
            for half in self.graph.native.half_edges
            if half.vertex == owner_id
        }
        vector_ports = {
            h.Symbols.half_edge()(half.id, 1)
            for half in self.graph.native.half_edges
            if half.vertex == owner_id and half.edge.particle.spin == 3
        }
        if {axis.index for axis in axes} != vector_ports:
            raise NotImplementedError(
                "source vertex does not expose its declared vector ports"
            )
        bases = []
        if len(axes) in (1, 3):
            for free in range(len(axes)):
                rest = [a for i, a in enumerate(axes) if i != free]
                pair = E("1") if not rest else metric(*rest).to_expression()
                bases.extend(
                    h.Symbols.edge_momentum()(edge, axes[free].to_expression()) * pair
                    for edge in incident
                )
        else:
            for partner in range(1, 4):
                rest = [a for i, a in enumerate(axes) if i not in (0, partner)]
                bases.append(
                    (metric(axes[0], axes[partner]) * metric(*rest)).to_expression()
                )
        expected = E("0")
        for basis in bases:
            coefficient = expression.coefficient(basis)
            self._scalar(coefficient)
            expected += coefficient * basis
        if (expression - expected).expand() != 0:
            raise NotImplementedError(
                "vertex is outside the metric/momentum Yang--Mills templates"
            )

    def _prepare_library(self, library):
        from symbolica import E
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        if self._prepared:
            return
        physical, full = t.Representation.mink(4), t.Representation.mink(self.dimension)
        head = t.TensorName(h.Symbols.edge_momentum().get_name())
        for edge in self.graph.native.edges:
            values = list(library[head(edge.id, physical)])
            if self.dimension > 4:
                values += [
                    edge.momentum_signature().loops[0]
                    * self.route_sign
                    * self.extra_variables[0]
                ]
                values += [E("0")] * (self.dimension - 5)
            library.register(t.Tensor.dense(head(edge.id, full), values))
        for edge in self.graph.native.external_edges:
            name = t.TensorName(f"loop_recurrence::state{edge.external_index}")
            values = list(library[name(physical)])
            library.register(
                t.Tensor.dense(name(full), values + [E("0")] * (self.dimension - 4))
            )
        self._prepared = True

    def materialize(self, kind, owner_id, lorentz, library):
        from symbolica import E
        from symbolica.community import tensor as t

        self._prepare_library(library)
        promoted = lorentz.with_lorentz_dimension(self.dimension)
        if (
            kind == "edge"
            and owner_id == self.weighted_edge
            and self.dimensional_weight
            and not self.ghost_cycle
        ):
            normalization = self._normalizations[
                owner_id, lorentz.to_expression().expand()
            ]
            full = t.Representation.mink(self.dimension)
            name = t.TensorName.vector("loop_evanescent::spectator")
            basis = t.Tensor.dense(name(full), [E("0")] * 5 + [E("1")])
            library.register(basis)
            norm = basis.dot(basis).to_tensor().scalar()
            left, right = promoted.axes
            projector = name(left) * name(right) / norm
            dimension = 4 - 2 * self.extra_variables[1]
            promoted += normalization * (dimension - 6) * projector
        return promoted.to_tensor(library)


def rational_sector_columns(recurrence):
    """Select the R2 difference only for the weighted physical-attachment domain.

    Its epsilon-free, mu-free closure is the original four-dimensional source
    numerator. The remaining even-mu columns and explicit epsilon columns form
    the local dimensional remainder. No open current may be projected this way.
    """
    provider = recurrence.kernel_provider
    node = recurrence.dag.nodes[recurrence.final_node]
    if (
        not isinstance(provider, BoundYangMillsKernels)
        or not provider.dimensional_weight
        or provider.dimension != 6
        or node.axes
    ):
        raise ValueError(
            "expected a closed dimensionally weighted Yang--Mills recurrence"
        )
    if any(label.powers[5] > 1 for label in node.output_labels):
        raise NotImplementedError(
            "more than one dimensional trace weight is unsupported"
        )
    return tuple(
        index
        for index, label in enumerate(node.output_labels)
        if label.powers[4] % 2 == 0 and (label.powers[4] > 0 or label.powers[5] > 0)
    )
