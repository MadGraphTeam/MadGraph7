# SPDX-License-Identifier: 0BSD
"""Dependency-backed Born/one-loop graph oracle for recurrence construction.

HepKit owns enumeration, graph identity, cycles, signs, routing and integral
family equivalence. This boundary keeps local Feynman-rule factors; it never
requests a contracted or expanded whole-process numerator. It is not a physical
virtual evaluator: counterterm lowering, covariant kernel certification and
complex-mass prescriptions remain separate prerequisites.

Selection follows MadLoop's ``filter_loop_for_perturbative_orders`` and
``is_wf_correction`` / ``is_vanishing_tadpole``. In particular, joint QCD+QED
selection can admit graphs absent from either singleton selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .orders import (
    BornLoopPair,
    LoopContribution,
    LoopEligibility,
    LoopOrderPlan,
    canonical_order,
    compile_order_plan,
)

if TYPE_CHECKING:
    from pyamplicol.models.base import CouplingOrders


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: int
    source: int | None
    target: int | None
    pdg: int
    particle: str
    directed: bool
    spin: int
    color: int
    mass: str
    width_parameter: str
    loop_momentum: tuple[int, ...]
    external_momentum: tuple[int, ...]
    external_index: int | None
    external_state: str | None
    denominator_template: str | None
    numerator: str


@dataclass(frozen=True, slots=True)
class GraphVertex:
    id: int
    interaction: str
    order: CouplingOrders
    numerator: str


@dataclass(frozen=True, slots=True)
class FamilyAssignment:
    family_index: int
    denominator_map: tuple[int, ...]
    momentum_rules: tuple[tuple[str, str], ...]
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class IntegralFamilyIR:
    """A reduction family, shared only after a native affine-map proof."""

    index: int
    denominators: tuple[str, ...]
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class GraphIR:
    id: str
    name: str
    order: CouplingOrders
    loop_count: int
    symmetry_denominator: int
    overall_factor: str
    evaluated_factor: str
    vertices: tuple[GraphVertex, ...]
    edges: tuple[GraphEdge, ...]
    # Each entry is (half-edge ID, vertex ID, edge ID, native flow label).
    half_edges: tuple[tuple[int, int, int, str], ...]
    loop_edges: tuple[int, ...]
    bridge_edges: tuple[int, ...]
    attachment_edges: tuple[int, ...]
    loop_species: tuple[int, ...]
    eligibility: LoopEligibility | None
    loop_basis_edges: tuple[int, ...]
    external_basis_edges: tuple[int, ...]
    family: FamilyAssignment | None
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RejectedGraph:
    id: str
    reason: str


@dataclass(frozen=True, slots=True)
class GraphSelectionConvention:
    """The graph policy used by this oracle, distinct from native defaults."""

    internal_bridges: str = "retained"
    internal_self_energies: str = "retained"
    external_wavefunction_corrections: str = "excluded-uv-insertions-required"
    one_attachment_tadpoles: str = "vev-subtracted"
    massless_tadpoles: str = "excluded"
    massive_quartic_tadpoles: str = "retained"
    single_coloured_loop_attachments: str = "excluded"
    vertex_orders: str = "homogeneous-required"
    denominators: str = "q2-minus-mass2-width-prescription-not-inferred"


@dataclass(frozen=True, slots=True)
class LoopProcessIR:
    model_name: str
    born: tuple[GraphIR, ...]
    loops: tuple[GraphIR, ...]
    families: tuple[IntegralFamilyIR, ...]
    orders: LoopOrderPlan
    rejected: tuple[RejectedGraph, ...]
    convention: GraphSelectionConvention = GraphSelectionConvention()
    stage: str = field(default="graph-oracle", init=False)


def _rule_orders(model: Any) -> dict[str, CouplingOrders]:
    """Avoid the native max-across-couplings convention for inhomogeneous rules."""
    result = {}
    for rule in model.vertex_rules:
        orders = {
            canonical_order(tuple(model.coupling(coupling).orders.items()))
            for row in rule.couplings
            for coupling in row
            if coupling is not None
        }
        if len(orders) != 1:
            raise NotImplementedError(
                f"vertex {rule.name!r} needs splitting into homogeneous coupling "
                "orders before loop graph generation"
            )
        result[rule.name] = next(iter(orders))
    return result


def _species_orders(
    model: Any, rule_orders: Mapping[str, CouplingOrders]
) -> dict[int, frozenset[str]]:
    # MadLoop Particle.is_perturbating ignores vertices with multiple order
    # names: a mixed photon-gluon-squark vertex does not charge the photon under
    # QCD. Use the full ordinary model, as MadLoop does, not only this process's
    # vertex allow-list.
    return _species_orders_from_rules(
        (
            tuple(
                abs(model.particle(particle).pdg_code) for particle in rule.particles
            ),
            rule_orders[rule.name],
        )
        for rule in model.vertex_rules
    )


def _species_orders_from_rules(rules):
    """Shared classification of complete ordinary species/order records."""
    names: dict[int, set[str]] = {}
    for particles, order in rules:
        if len(order) == 1:
            for pdg in particles:
                names.setdefault(pdg, set()).add(order[0][0])
    return {pdg: frozenset(orders) for pdg, orders in names.items()}


def _validated_species_orders(model, facts):
    import json

    if not isinstance(facts, Mapping):
        raise TypeError("perturbative species facts must be a mapping")
    particles = {abs(p.pdg_code) for p in model.particles}
    orders = {order["name"] for order in json.loads(model.to_json())["orders"]}
    result = {}
    for pdg, names in facts.items():
        if type(pdg) is not int or pdg <= 0 or pdg not in particles:
            raise ValueError("perturbative species facts name an unknown positive PDG")
        if isinstance(names, (str, bytes)):
            raise TypeError("perturbative species orders require a collection")
        names = frozenset(names)
        if not names <= orders:
            raise ValueError("perturbative species facts name an undeclared order")
        result[pdg] = names
    return result


def _summed_orders(orders: Iterable[CouplingOrders]) -> CouplingOrders:
    powers: dict[str, int] = {}
    for order in orders:
        for name, power in order:
            powers[name] = powers.get(name, 0) + power
    return canonical_order(tuple(powers.items()))


def _propagator_denominators(model: Any) -> dict[str, str]:
    from symbolica.community import hepkit as hep

    momentum_squared = hep.Symbols.ufo_momentum()(hep.Symbols.ufo_index()(1, 1)) ** 2
    result = {}
    for propagator in model.propagators:
        expected = momentum_squared - model.particle(propagator.particle).mass ** 2
        if (propagator.denominator - expected).expand() != 0:
            raise NotImplementedError(
                f"particle {propagator.particle!r} has a custom denominator; "
                "native propagator_family only supports q^2-m^2"
            )
        result[propagator.particle] = propagator.denominator.to_canonical_string()
    return result


def _snapshot(
    diagram: Any,
    rule_orders: Mapping[str, CouplingOrders],
    species_orders: Mapping[int, frozenset[str]],
    propagator_denominators: Mapping[str, str],
) -> GraphIR:
    if diagram.loop_count not in (0, 1):
        raise ValueError("the loop process oracle supports at most one loop")
    cycles, _forest = diagram.cycle_basis()
    if len(cycles) != diagram.loop_count:
        raise ValueError("native cycle basis disagrees with the diagram loop count")
    cycle_edges = () if not cycles else tuple(edge.id for edge in cycles[0].edges)
    cycle_vertices = set() if not cycles else {v.id for v in cycles[0].vertices}
    attachments = tuple(
        edge.id
        for edge in diagram.edges
        if edge.id not in cycle_edges
        and (edge.source in cycle_vertices or edge.target in cycle_vertices)
    )
    vertices = tuple(
        GraphVertex(v.id, v.interaction, rule_orders[v.interaction], v.numerator)
        for v in diagram.vertices
    )
    edges = []
    for edge in diagram.edges:
        if edge.is_dummy:
            raise NotImplementedError("dummy graph attachments need explicit lowering")
        signature = edge.momentum_signature()
        edges.append(
            GraphEdge(
                edge.id,
                edge.source,
                edge.target,
                edge.particle_pdg,
                edge.particle_name,
                edge.directed,
                edge.particle.spin,
                edge.particle.color,
                edge.particle.mass.to_canonical_string(),
                edge.particle.width_parameter,
                tuple(signature.loops),
                tuple(signature.external),
                edge.external_index if edge.is_external else None,
                edge.external_state if edge.is_external else None,
                None
                if edge.is_external
                else propagator_denominators[edge.particle_name],
                edge.numerator,
            )
        )
    loop_species = tuple(sorted({abs(e.pdg) for e in edges if e.id in cycle_edges}))
    eligibility = None
    if diagram.loop_count:
        eligibility = LoopEligibility(
            tuple(species_orders.get(pdg, frozenset()) for pdg in loop_species),
            frozenset(
                name
                for vertex in vertices
                if vertex.id in cycle_vertices
                for name, _power in vertex.order
            ),
        )
    return GraphIR(
        diagram.id,
        diagram.name,
        _summed_orders(v.order for v in vertices),
        diagram.loop_count,
        diagram.symmetry_factor,
        diagram.overall_factor,
        diagram.overall_factor_expression(evaluate=True).to_canonical_string(),
        vertices,
        tuple(edges),
        tuple((h.id, h.vertex, h.edge.id, h.flow) for h in diagram.half_edges),
        cycle_edges,
        tuple(e.id for e in diagram.bridges().internal_edges),
        attachments,
        loop_species,
        eligibility,
        tuple(diagram.loop_momentum_basis.loop_edges),
        tuple(diagram.loop_momentum_basis.external_edges),
        None,
        diagram,
    )


def _physical_rejection(graph: GraphIR) -> str | None:
    """MadLoop's amputated/VEV-subtracted one-loop diagram convention.

    This policy uses dependency-owned cycle/edge incidence. It does not reject
    internal self energies, massive quartic tadpoles, or other internal bridges.
    External wavefunction counterterms belong to the later UV insertion stage.
    """
    by_id = {edge.id: edge for edge in graph.edges}
    attachments = [by_id[index] for index in graph.attachment_edges]
    cycle = [by_id[index] for index in graph.loop_edges]
    if (
        len(cycle) in (1, 2)
        and len(attachments) == 2
        and any(edge.external_index is not None for edge in attachments)
    ):
        return "external-wavefunction-correction"
    if len(cycle) == 1:
        if len(attachments) <= 1:
            return "vev-subtracted-tadpole"
        if cycle[0].mass == "0":
            return "massless-tadpole"
    if sum(abs(edge.color) != 1 for edge in attachments) == 1:
        return "single-coloured-loop-attachment"
    return None


def _assign_families(
    graphs: tuple[GraphIR, ...],
) -> tuple[tuple[GraphIR, ...], tuple[IntegralFamilyIR, ...]]:
    from symbolica.community import hepkit as hep

    native_families = []
    for graph in graphs:
        full = graph.native.propagator_family()
        # propagator_family deliberately retains internal bridges. They are
        # tree factors, not integrated denominators: select cycle propagators
        # without rebuilding the dependency's routed expressions.
        denominators = [
            denominator
            for edge, denominator in zip(
                graph.native.internal_edges, full.denominators, strict=True
            )
            if edge.id in graph.loop_edges
        ]
        native_families.append(
            hep.IntegralFamily(
                full.loop_momenta,
                full.external_momenta,
                denominators,
                kinematics=full.kinematics,
            )
        )
    mappings = hep.IntegralFamily.find_mappings(native_families)
    representatives = sorted({index for index, _mapping in mappings})
    indices = {
        representative: index for index, representative in enumerate(representatives)
    }
    families = tuple(
        IntegralFamilyIR(
            indices[index],
            tuple(d.to_canonical_string() for d in native_families[index].denominators),
            native_families[index],
        )
        for index in representatives
    )
    assigned = tuple(
        replace(
            graph,
            family=FamilyAssignment(
                indices[representative],
                tuple(mapping.denominator_map),
                tuple(
                    (source.to_canonical_string(), target.to_canonical_string())
                    for source, target in mapping.momentum_rules
                ),
                mapping,
            ),
        )
        for graph, (representative, mapping) in zip(graphs, mappings, strict=True)
    )
    return assigned, families


def build_loop_process(
    process: Any,
    *,
    selected_corrections: Iterable[str],
    perturbative_orders: Mapping[str, int],
    allowed_pairs: Iterable[BornLoopPair] | None = None,
    born_order_bounds: Mapping[str, int | tuple[int, int | None]] | None = None,
    loop_order_bounds: Mapping[str, int | tuple[int, int | None]] | None = None,
    ordinary_model: Any = None,
    perturbative_species_orders: Mapping[int, frozenset[str]] | None = None,
) -> LoopProcessIR:
    """Generate a small native graph catalogue with explicit physical filters.

    ``perturbative_orders`` must come from the NLO model capability boundary;
    a normalized tree model cannot certify it. ``allowed_pairs=None`` authorizes
    every generated Born/selected-loop order pair. A provided list is exact,
    with no inference from order increments. Bounds use HepKit's exact/range
    coupling-power semantics. This oracle is intentionally not the production
    loop-generation dispatcher. For a model augmented with counterterm rules,
    supply its original ``ordinary_model``: perturbative species membership is
    defined by ordinary interactions, never by effective counterterm vertices.
    Species identities are signed PDG codes shared by the two models. A source
    compiler that prunes impossible local orders must supply the validated full
    ordinary species facts through ``perturbative_species_orders``; request
    pruning must not change correction eligibility.
    """
    names = tuple(sorted(set(selected_corrections)))
    if not names or any(perturbative_orders.get(name, 0) < 1 for name in names):
        raise ValueError("requested corrections require declared NLO model support")
    rule_orders = _rule_orders(process.model)
    species_model = process.model if ordinary_model is None else ordinary_model
    species_orders = (
        _species_orders(
            species_model,
            rule_orders if ordinary_model is None else _rule_orders(ordinary_model),
        )
        if perturbative_species_orders is None
        else _validated_species_orders(process.model, perturbative_species_orders)
    )
    propagator_denominators = _propagator_denominators(process.model)
    options = dict(
        maximum_bridges=None,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        allow_self_loops=True,
        allow_zero_flow_edges=True,
        factorized_loop_topologies_count_range=None,
        symmetrize_initial=False,
        symmetrize_final=False,
        numerator_grouping=None,
        filter_zero_color=False,
        threads=1,
        progress=None,
    )
    born_result = process.generate_diagrams(
        loops=0, coupling_orders=born_order_bounds, **options
    )
    loop_result = process.generate_diagrams(
        loops=1, coupling_orders=loop_order_bounds, **options
    )
    if not born_result.report.completed or not loop_result.report.completed:
        raise RuntimeError("native graph generation did not complete")
    born = tuple(
        _snapshot(diagram, rule_orders, species_orders, propagator_denominators)
        for diagram in sorted(born_result.diagrams, key=lambda d: d.id)
    )
    selected: dict[str, GraphIR] = {}
    rejected = []
    for diagram in sorted(loop_result.diagrams, key=lambda d: d.id):
        graph = _snapshot(diagram, rule_orders, species_orders, propagator_denominators)
        reason = _physical_rejection(graph)
        if reason is None and not graph.eligibility.admits(frozenset(names)):
            reason = "correction-ineligible"
        if reason is not None:
            rejected.append(RejectedGraph(graph.id, reason))
        else:
            selected[graph.id] = graph
    contributions = tuple(
        LoopContribution(graph.id, graph.order, graph.eligibility)
        for graph in selected.values()
    )
    if allowed_pairs is None:
        allowed_pairs = {
            BornLoopPair(tree.order, loop.order)
            for tree in born
            for loop in selected.values()
        }
    orders = compile_order_plan(
        selected_corrections=names,
        perturbative_orders=perturbative_orders,
        born_orders=(graph.order for graph in born),
        contributions=contributions,
        allowed_pairs=allowed_pairs,
    )
    loops, families = _assign_families(tuple(selected.values()))
    return LoopProcessIR(
        process.model.name, born, loops, families, orders, tuple(rejected)
    )
