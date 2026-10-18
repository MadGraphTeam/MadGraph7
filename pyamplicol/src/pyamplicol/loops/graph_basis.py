# SPDX-License-Identifier: 0BSD
"""Native graph traversal with explicit local colour alternatives.

Only local rule expressions are expanded. Lorentz currents are coefficient-DAG
references, while colour tensors retain native graph ports. The dependency owns
traversal, routing, tensor contraction and polynomial extraction. External spin
inputs here are independent components, not physical wavefunctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .color_terms import (
    ColorCurrent,
    ColorTermDAG,
    LocalColorTerm,
    split_local_color_terms,
)
from .graphs import GraphIR
from .kernels import CoefficientDAG
from .recurrence import (
    LocalFactor,
    attach_external_spin_components,
    momentum_component_library,
)


@dataclass(frozen=True, slots=True)
class ColorGraphRecurrence:
    graph: GraphIR = field(repr=False, compare=False)
    current: ColorCurrent = field(repr=False, compare=False)
    factors: tuple[LocalFactor, ...] = field(repr=False, compare=False)
    momentum_parameters: tuple[tuple[Any, ...], ...]
    external_states: tuple[tuple[int, tuple[Any, ...]], ...]
    loop_components: tuple[Any, ...] = ()
    cut_edge: int | None = None
    cycle_vertices: tuple[int, ...] = ()
    propagator_history: tuple[int, ...] = ()
    routing: str = "graph"
    tree_denominators: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    tree_denominators_included: bool = False
    kernel_provider: Any = field(default=None, repr=False, compare=False)
    stage: str = field(default="bare-color-coefficient-prototype", init=False)
    external_state_convention: str = field(
        default="independent-spin-components-source-projector-not-applied", init=False
    )

    @property
    def graph_id(self):
        return self.graph.id

    @property
    def dag(self):
        return self.current.dag

    @property
    def terms(self):
        return self.current.terms

    @property
    def final_nodes(self):
        return tuple(term.node for term in self.terms)

    @property
    def color_expressions(self):
        return tuple(term.color for term in self.terms)


def _symbols(graph, variables=None, momenta=None):
    from symbolica import S

    q = (
        tuple(S(f"loop_recurrence::q{axis}") for axis in range(4))
        if variables is None
        else tuple(variables)
    )
    external_count = len(graph.native.edges[0].momentum_signature().external)
    if momenta is None:
        momenta = tuple(
            tuple(S(f"loop_recurrence::p{leg}_{axis}") for axis in range(4))
            for leg in range(external_count)
        )
    momenta = tuple(tuple(vector) for vector in momenta)
    if len(momenta) != external_count or any(len(vector) != 4 for vector in momenta):
        raise ValueError(
            "four momentum components per native external basis edge required"
        )
    return q, momenta


def _prepare_locals(graph, library, provider=None):
    from symbolica.community import hepkit as h

    factors, local = [], {}
    for kind, items in (
        ("vertex", graph.native.vertices),
        ("edge", graph.native.internal_edges),
    ):
        for item in items:
            original = item.numerator_expression()
            terms = split_local_color_terms(original)
            if not terms:
                raise NotImplementedError(
                    "identically zero local graph rules require graph pruning"
                )
            for term in terms:
                if provider is not None:
                    provider.validate_factor(kind, item.id, term.lorentz)
                factors.append(
                    LocalFactor((kind, item.id), term.lorentz, term.color, original)
                )
            local[kind, item.id] = terms
    # The existing state helper registers one shared component vector per leg.
    # Its multiplier is then attached to every term using identical native ports.
    representative = {key: terms[0].lorentz for key, terms in local.items()}
    original_representative = dict(representative)
    states = attach_external_spin_components(graph, representative, library)
    for edge in graph.native.external_edges:
        half = next(half for half in graph.native.half_edges if half.edge.id == edge.id)
        key = ("vertex", half.vertex)
        label = h.Symbols.half_edge()(half.id, 1)
        expected = {
            slot for slot in original_representative[key].axes if slot.index == label
        }
        if any(
            {slot for slot in term.lorentz.axes if slot.index == label} != expected
            for term in local[key]
        ):
            raise ValueError(
                "local colour alternatives disagree on external spin ports"
            )
    # Obtain the already registered state tensors directly from their native
    # slots; no division of Lorentz expressions or component reconstruction.
    from symbolica.community import tensor as t

    for edge in graph.native.external_edges:
        half = next(half for half in graph.native.half_edges if half.edge.id == edge.id)
        key = ("vertex", half.vertex)
        label = h.Symbols.half_edge()(half.id, 1)
        axes = tuple(
            slot.dual()
            for slot in original_representative[key].axes
            if slot.index == label
        )
        if axes:
            state = t.TensorName(f"loop_recurrence::state{edge.external_index}")(*axes)
            local[key] = tuple(
                LocalColorTerm(term.lorentz * state, term.color, term.source)
                for term in local[key]
            )
    return local, tuple(factors), tuple(sorted(states))


def _add(builder, local, owner, name, dependencies, library, provider=None):
    materialize = (
        None
        if provider is None
        else lambda term: provider.materialize(*owner, term.lorentz, library)
    )
    return builder.add(
        name, local[owner], dependencies, library=library, materialize=materialize
    )


def _tree_currents(
    graph,
    forest,
    roots,
    local,
    builder,
    library,
    *,
    include_denominators=False,
    provider=None,
):
    currents, used = {}, set()
    for root in roots:
        traversal = forest.depth_first_traverse(root)
        current = {}
        for vertex in reversed(traversal.nodes):
            if vertex.id in used:
                raise ValueError(
                    "tree attachment belongs to more than one cycle vertex"
                )
            used.add(vertex.id)
            children = tuple(
                current[child.id] for child in traversal.children(vertex.id)
            )
            value = _add(
                builder,
                local,
                ("vertex", vertex.id),
                f"vertex:{vertex.id}",
                children,
                library,
                provider,
            )
            parent = traversal.parent(vertex.id)
            if parent is not None:
                attachment = next(
                    edge
                    for edge in traversal.subgraph.internal_edges
                    if {edge.source, edge.target} == {parent.id, vertex.id}
                )
                value = _add(
                    builder,
                    local,
                    ("edge", attachment.id),
                    f"tree-propagator:{attachment.id}",
                    (value,),
                    library,
                    provider,
                )
                if include_denominators:
                    from symbolica import E
                    from symbolica.community import tensor as t

                    denominator = routed_edge_denominator(attachment, library)
                    value = builder.add(
                        f"tree-denominator:{attachment.id}",
                        (
                            LocalColorTerm(
                                t.TensorExpression(1 / denominator),
                                t.TensorExpression(E("1")),
                            ),
                        ),
                        (value,),
                        library=library,
                    )
            current[vertex.id] = value
        currents[root] = current[root]
    if used != {v.id for v in graph.native.vertices}:
        raise ValueError("native attachment traversals did not cover every vertex")
    return currents


def _global_factor(graph, builder, current, library):
    from symbolica import E
    from symbolica.community import tensor as t

    result = builder.add(
        "global-factor",
        (
            LocalColorTerm(
                t.TensorExpression(
                    graph.native.overall_factor_expression(evaluate=True)
                ),
                t.TensorExpression(E("1")),
            ),
        ),
        (current,),
        library=library,
    )
    if not result.closed:
        raise ValueError("graph closure left uncontracted Lorentz/spin ports")
    return result


def routed_edge_denominator(edge, library):
    """Materialize the dependency's exact propagator denominator locally."""
    from symbolica import S
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    edge_id, momentum, mass, value = S(
        "loop_denominator::e_",
        "loop_denominator::p_",
        "loop_denominator::m_",
        "loop_denominator::v_",
    )
    expression = (
        edge.denominator_expression(dimension=4)
        .to_expression()
        .replace(h.Symbols.denominator()(edge_id, momentum, mass, value), value)
    )
    return t.TensorExpression(expression).to_tensor(library)[0]


def build_color_tree_recurrence(
    graph: GraphIR, *, variables=None, momenta=None
) -> ColorGraphRecurrence:
    """Prepare every Born/CT colour alternative, including tree denominators."""
    if graph.loop_count != 0:
        raise ValueError("a tree recurrence requires a zero-loop graph")
    if graph.native.numerator_prefactor_expression() != 1:
        raise NotImplementedError(
            "custom numerator prefactors need explicit local lowering"
        )
    q, momenta = _symbols(graph, variables, momenta)
    library = momentum_component_library(graph, (), momenta)
    local, factors, states = _prepare_locals(graph, library)
    builder = ColorTermDAG(CoefficientDAG(q))
    root = graph.vertices[0].id
    currents = _tree_currents(
        graph, graph.native, (root,), local, builder, library, include_denominators=True
    )
    current = _global_factor(graph, builder, currents[root], library)
    return ColorGraphRecurrence(
        graph, current, factors, momenta, states, tree_denominators_included=True
    )


def build_color_loop_recurrence(
    graph: GraphIR,
    *,
    cut_edge: int | None = None,
    routing: str = "graph",
    kernel_provider=None,
) -> ColorGraphRecurrence:
    """Prepare local alternatives along the native cut cycle and its trees.

    Each term is one coefficient output and one exact colour tensor. Native
    signs/symmetry enter once per term, never multiplied by the number of basis
    alternatives. Loop denominators stay outside the numerator recurrence;
    attached tree denominators are explicitly returned for later weighting.
    An optional dependency-backed kernel provider can add certified local
    polynomial variables; this builder does not certify dimensional algebra.
    """
    from symbolica import E, Replacement
    from symbolica.community import tensor as t

    if routing not in {"graph", "family"}:
        raise ValueError("routing must be graph or family")
    if graph.loop_count != 1:
        raise ValueError("an open-loop recurrence requires exactly one loop")
    if graph.native.numerator_prefactor_expression() != 1:
        raise NotImplementedError(
            "custom numerator prefactors need explicit local lowering"
        )
    cut_edge = min(graph.loop_edges) if cut_edge is None else cut_edge
    if cut_edge not in graph.loop_edges:
        raise ValueError("cut edge must belong to the unique native cycle")
    cut = next(edge for edge in graph.native.edges if edge.id == cut_edge)
    cycle = graph.native.subgraph(edges=list(graph.loop_edges))
    path = cycle - graph.native.subgraph(edges=[cut_edge])
    order = (
        (cut.source,)
        if len(graph.loop_edges) == 1
        else tuple(vertex.id for vertex in path.depth_first_traverse(cut.source).nodes)
    )
    if set(order) != {v.id for v in cycle.vertices} or len(order) != len(
        graph.loop_edges
    ):
        raise ValueError(
            "native cut traversal did not produce the complete simple cycle"
        )
    q, momenta = _symbols(graph)
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
    provider = (
        None
        if kernel_provider is None
        else kernel_provider.bind(graph, loop_components, q)
    )
    library = momentum_component_library(graph, (loop_components,), momenta)
    local, factors, states = _prepare_locals(graph, library, provider)
    dag = CoefficientDAG(
        q,
        extra_variables=() if provider is None else tuple(provider.extra_variables),
        extra_weights=() if provider is None else tuple(provider.extra_weights),
    )
    builder = ColorTermDAG(dag)
    forest = graph.native.filter(edge=lambda edge: edge.id not in graph.loop_edges)
    segments = _tree_currents(
        graph, forest, order, local, builder, library, provider=provider
    )
    current, history = segments[order[0]], []
    unit = (LocalColorTerm(t.TensorExpression(E("1")), t.TensorExpression(E("1"))),)
    for source, target in pairwise(order):
        propagator = next(
            edge
            for edge in path.internal_edges
            if {edge.source, edge.target} == {source, target}
        )
        history.append(propagator.id)
        current = _add(
            builder,
            local,
            ("edge", propagator.id),
            f"loop-propagator:{propagator.id}",
            (current,),
            library,
            provider,
        )
        current = builder.add(
            f"open-current:{target}", unit, (current, segments[target]), library=library
        )
    history.append(cut_edge)
    current = _add(
        builder,
        local,
        ("edge", cut_edge),
        f"closure:{cut_edge}",
        (current,),
        library,
        provider,
    )
    current = _global_factor(graph, builder, current, library)
    family = graph.native.propagator_family()
    tree_denominators = tuple(
        denominator
        for edge, denominator in zip(
            graph.native.internal_edges, family.denominators, strict=True
        )
        if edge.id not in graph.loop_edges
    )
    return ColorGraphRecurrence(
        graph,
        current,
        factors,
        momenta,
        states,
        loop_components,
        cut_edge,
        order,
        tuple(history),
        routing,
        tree_denominators,
        kernel_provider=provider,
    )
