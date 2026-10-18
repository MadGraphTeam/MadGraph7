# SPDX-License-Identifier: 0BSD
"""Native-graph open-loop coefficient recurrence prototype.

This is a bare four-dimensional numerator builder, not a virtual observable.
Tree propagator denominators, model counterterms, external wavefunctions and
color contraction remain explicit integration boundaries. Color tensors are
retained exactly and never replaced by numerical SU(3) components. Local rules
with multiple independent color terms require the forthcoming term-DAG lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

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
    graph: GraphIR, *, cut_edge: int | None = None, routing: str = "graph"
) -> OpenLoopRecurrence:
    """Build reusable local coefficient instructions for one native graph.

    The cut is a deterministic construction choice and adds no multiplicity.
    Native evaluated global signs/symmetry are included exactly once at closure.
    ``routing="family"`` lifts the verified native affine momentum map component
    by component before coefficient extraction; graph identity stays distinct.
    External spin tensors are independent exact component inputs. Their physical
    wavefunction provider is deliberately outside this algebraic prototype; the
    native graph projector is retained on ``graph.native`` and is not applied.
    """
    from symbolica import E, Replacement, S
    from symbolica.community import tensor as t

    if routing not in {"graph", "family"}:
        raise ValueError("routing must be graph or family")
    if graph.loop_count != 1:
        raise ValueError("an open-loop recurrence requires exactly one loop")
    if graph.native.numerator_prefactor_expression() != 1:
        raise NotImplementedError(
            "custom numerator prefactors need explicit local lowering"
        )
    if cut_edge is None:
        cut_edge = min(graph.loop_edges)
    if cut_edge not in graph.loop_edges:
        raise ValueError("cut edge must belong to the unique native cycle")
    edges = {edge.id: edge for edge in graph.native.edges}
    cut = edges[cut_edge]
    cycle = graph.native.subgraph(edges=list(graph.loop_edges))
    path = cycle - graph.native.subgraph(edges=[cut_edge])
    if len(graph.loop_edges) == 1:
        order = (cut.source,)
    else:
        order = tuple(
            vertex.id for vertex in path.depth_first_traverse(cut.source).nodes
        )
    if set(order) != {v.id for v in cycle.vertices} or len(order) != len(
        graph.loop_edges
    ):
        raise ValueError(
            "native cut traversal did not produce the complete simple cycle"
        )
    q = tuple(S(f"loop_recurrence::q{axis}") for axis in range(4))
    external_count = len(next(iter(edges.values())).momentum_signature().external)
    momenta = tuple(
        tuple(S(f"loop_recurrence::p{leg}_{axis}") for axis in range(4))
        for leg in range(external_count)
    )
    loop_components = q
    if routing == "family":
        if graph.family is None:
            raise ValueError("family routing requires a verified native family map")
        family = graph.native.propagator_family()
        source_loop = family.loop_momenta[0]
        image = dict(graph.family.native.momentum_rules)[source_loop]
        basis = graph.native.loop_momentum_basis
        independent = [
            momenta[i]
            for i, edge in enumerate(basis.external_edges)
            if edge not in basis.dependent_externals
        ]
        loop_components = tuple(
            image.replace_multiple(
                [
                    Replacement(source_loop, q[axis]),
                    *(
                        Replacement(name, components[axis])
                        for name, components in zip(
                            family.external_momenta, independent, strict=True
                        )
                    ),
                ]
            )
            for axis in range(4)
        )
    library = momentum_component_library(graph, (loop_components,), momenta)
    factors = []
    local = {}
    color = E("1")
    for kind, items in (
        ("vertex", graph.native.vertices),
        ("edge", graph.native.internal_edges),
    ):
        for item in items:
            original = item.numerator_expression()
            lorentz, color_factor = separate_local_color(original)
            factors.append(
                LocalFactor((kind, item.id), lorentz, color_factor, original)
            )
            local[kind, item.id] = lorentz
            color *= color_factor
    external_states = attach_external_spin_components(graph, local, library)
    dag = CoefficientDAG(q)
    forest = graph.native.filter(edge=lambda edge: edge.id not in graph.loop_edges)
    segments = {}
    used_vertices = set()
    for root in order:
        traversal = forest.depth_first_traverse(root)
        current = {}
        for vertex in reversed(traversal.nodes):
            if vertex.id in used_vertices:
                raise ValueError(
                    "tree attachment belongs to more than one cycle vertex"
                )
            used_vertices.add(vertex.id)
            children = tuple(
                current[child.id] for child in traversal.children(vertex.id)
            )
            node = dag.add(
                f"vertex:{vertex.id}",
                local["vertex", vertex.id].to_tensor(library),
                children,
            )
            parent = traversal.parent(vertex.id)
            if parent is not None:
                attachment = next(
                    edge
                    for edge in traversal.subgraph.internal_edges
                    if {edge.source, edge.target} == {parent.id, vertex.id}
                )
                node = dag.add(
                    f"tree-propagator:{attachment.id}",
                    local["edge", attachment.id].to_tensor(library),
                    (node,),
                )
            current[vertex.id] = node
        segments[root] = current[root]
    if used_vertices != {v.id for v in graph.native.vertices}:
        raise ValueError("native attachment traversals did not cover every vertex")
    current = segments[order[0]]
    history = []
    unit = t.TensorExpression(E("1")).to_tensor(library)
    for source, target in pairwise(order):
        propagator = next(
            edge
            for edge in path.internal_edges
            if {edge.source, edge.target} == {source, target}
        )
        history.append(propagator.id)
        current = dag.add(
            f"loop-propagator:{propagator.id}",
            local["edge", propagator.id].to_tensor(library),
            (current,),
        )
        current = dag.add(f"open-current:{target}", unit, (current, segments[target]))
    history.append(cut_edge)
    closing = local["edge", cut_edge] * graph.native.overall_factor_expression(
        evaluate=True
    )
    current = dag.add(f"closure:{cut_edge}", closing.to_tensor(library), (current,))
    if dag.nodes[current].axes:
        raise ValueError("loop closure left uncontracted Lorentz/spin ports")
    family = graph.native.propagator_family()
    tree_denominators = tuple(
        denominator
        for edge, denominator in zip(
            graph.native.internal_edges, family.denominators, strict=True
        )
        if edge.id not in graph.loop_edges
    )
    return OpenLoopRecurrence(
        graph.id,
        cut_edge,
        order,
        tuple(history),
        current,
        routing,
        loop_components,
        dag,
        tuple(factors),
        t.TensorExpression(color),
        tree_denominators,
        momenta,
        tuple(sorted(external_states)),
        graph,
    )


def momentum_component_library(graph: GraphIR, loops: tuple, momenta: tuple):
    """Register exact routed edge components in the dependency tensor library."""
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    library = t.TensorLibrary.hep_lib_atom()
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
    from symbolica import S
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
            S(f"loop_recurrence::state{edge.external_index}_{i}") for i in range(count)
        )
        name = t.TensorName(f"loop_recurrence::state{edge.external_index}")
        library.register(
            t.Tensor.dense(name(*(slot.representation for slot in axes)), components)
        )
        local["vertex", vertex] *= name(*axes)
        external_states.append((edge.external_index, components))
    return tuple(external_states)
