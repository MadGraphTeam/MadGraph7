# SPDX-License-Identifier: 0BSD
"""Native-graph open-loop coefficient recurrence prototype.

This is a bare four-dimensional numerator builder, not a virtual observable.
Tree propagator denominators, model counterterms, external wavefunctions and
color contraction remain explicit integration boundaries. Color tensors are
retained exactly and never replaced by numerical SU(3) components. The companion
``graph_basis`` builders retain multiple local colour terms; this module keeps
the original single-colour result interface for existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._internal.physics.symbols import symbols
from .graphs import GraphIR
from .kernels import CoefficientDAG


@dataclass(frozen=True, slots=True)
class LocalFactor:
    owner: tuple[str, int]
    lorentz: Any
    color: Any
    original: Any


@dataclass(frozen=True, slots=True)
class OpenLoopRecurrence:
    graph_id: str
    cut_edge: int
    cycle_vertices: tuple[int, ...]
    propagator_history: tuple[int, ...]
    final_node: int
    routing: str
    loop_components: tuple[Any, ...]
    dag: CoefficientDAG = field(repr=False, compare=False)
    factors: tuple[LocalFactor, ...] = field(repr=False, compare=False)
    color_expression: Any = field(repr=False, compare=False)
    tree_denominators: tuple[Any, ...] = field(repr=False, compare=False)
    momentum_parameters: tuple[tuple[Any, ...], ...]
    external_states: tuple[tuple[int, tuple[Any, ...]], ...]
    # Keep native graph, edge flow, affine family map and exact routing together.
    graph: GraphIR = field(repr=False, compare=False)
    kernel_provider: Any = field(default=None, repr=False, compare=False)
    stage: str = field(default="bare-coefficient-prototype", init=False)
    external_state_convention: str = field(
        default="independent-spin-components-source-projector-not-applied", init=False
    )


def separate_local_color(expression: Any) -> tuple[Any, Any]:
    """Prove a local factor is one Lorentz factor times one color factor.

    Only native local expansion is used, never whole-graph expansion. Unknown
    index spaces or distinct color terms are rejected rather than reconstructed
    from an evaluated SU(3) tensor. Original factors are retained by the caller.
    """
    from symbolica import AtomType, E
    from symbolica.community import tensor as t

    color_spaces = (
        t.Representation.cof(3),
        t.Representation.cof(3).dual(),
        t.Representation.coad(8),
    )
    spin_spaces = (t.Representation.mink(4), t.Representation.bis(4))
    expanded = expression.expand()
    terms = list(expanded) if expanded.get_type() == AtomType.Add else [expanded]
    groups: dict[Any, Any] = {}
    for term in terms:
        color, lorentz = E("1"), E("1")
        factors = list(term) if term.get_type() == AtomType.Mul else [term]
        for factor in factors:
            axes = t.TensorExpression(factor).axes
            spaces = tuple(slot.representation for slot in axes)
            if any(space not in color_spaces + spin_spaces for space in spaces):
                raise NotImplementedError(
                    "local factor contains an unsupported index space"
                )
            names = {symbol.get_name() for symbol in factor.get_all_symbols()}
            has_color = bool(names & {"spenso::cof", "spenso::coad"})
            if has_color:
                if any(space in spin_spaces for space in spaces):
                    raise NotImplementedError("one tensor mixes spin and color ports")
                color *= factor
            else:
                lorentz *= factor
        groups[color] = groups.get(color, E("0")) + lorentz
    if len(groups) != 1:
        raise NotImplementedError(
            "independent local color terms require a color-term DAG"
        )
    color, lorentz = next(iter(groups.items()))
    if (lorentz * color - expression).expand().to_expression() != E("0"):
        raise ValueError("native local color-factorization proof failed")
    return t.TensorExpression(lorentz), t.TensorExpression(color)


def build_open_loop_recurrence(
    graph: GraphIR,
    *,
    cut_edge: int | None = None,
    routing: str = "graph",
    kernel_provider: Any = None,
) -> OpenLoopRecurrence:
    """Compatibility interface for a graph with one final colour alternative.

    The shared native traversal also supports local multi-colour rules through
    ``graph_basis.build_color_loop_recurrence``. Use that explicit term interface
    when there is more than one final coefficient/colour pair. A supplied local
    kernel provider owns its dimensional-algebra validity boundary.
    """
    from .graph_basis import build_color_loop_recurrence

    result = build_color_loop_recurrence(
        graph, cut_edge=cut_edge, routing=routing, kernel_provider=kernel_provider
    )
    if len(result.terms) != 1:
        raise NotImplementedError(
            "independent colour alternatives require build_color_loop_recurrence"
        )
    term = result.terms[0]
    return OpenLoopRecurrence(
        graph.id,
        result.cut_edge,
        result.cycle_vertices,
        result.propagator_history,
        term.node,
        routing,
        result.loop_components,
        result.dag,
        result.factors,
        term.color,
        result.tree_denominators,
        result.momentum_parameters,
        result.external_states,
        graph,
        result.kernel_provider,
    )


def momentum_component_library(graph: GraphIR, loops: tuple, momenta: tuple):
    """Register exact routed edge components in the dependency tensor library."""
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    from .tensor_components import physical_tensor_library

    library = physical_tensor_library()
    for edge in graph.native.edges:
        signature = edge.momentum_signature()
        if len(signature.loops) != len(loops):
            raise ValueError("loop component vectors do not match the native basis")
        components = [
            sum(
                (
                    c * vector[axis]
                    for c, vector in zip(signature.loops, loops, strict=True)
                ),
                E("0"),
            )
            + sum(
                (
                    c * vector[axis]
                    for c, vector in zip(signature.external, momenta, strict=True)
                ),
                E("0"),
            )
            for axis in range(4)
        ]
        library.register(
            t.Tensor.dense(
                t.TensorName(h.Symbols.edge_momentum().get_name())(
                    edge.id, t.Representation.mink(4)
                ),
                components,
            )
        )
    return library


def attach_external_spin_components(graph: GraphIR, local: dict, library: Any):
    """Attach exact independent spin inputs in native half-edge slot order."""
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    external_states = []
    for edge in graph.native.external_edges:
        half = next(half for half in graph.native.half_edges if half.edge.id == edge.id)
        vertex = half.vertex
        label = h.Symbols.half_edge()(half.id, 1)
        axes = tuple(
            slot.dual() for slot in local["vertex", vertex].axes if slot.index == label
        )
        if not axes:
            continue
        from math import prod

        count = prod(int(slot.representation.dimension) for slot in axes)
        components = tuple(
            symbols.qualified(f"loop_recurrence::state{edge.external_index}_{i}")
            for i in range(count)
        )
        name = t.TensorName(f"loop_recurrence::state{edge.external_index}")
        library.register(
            t.Tensor.dense(name(*(slot.representation for slot in axes)), components)
        )
        local["vertex", vertex] *= name(*axes)
        external_states.append((edge.external_index, components))
    return tuple(external_states)
