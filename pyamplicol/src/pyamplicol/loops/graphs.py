# SPDX-License-Identifier: 0BSD
"""Graph records and coupling-order classification of HepKit diagrams.

The records (``GraphIR``, ``LoopProcessIR``) and the species/order
classification shared by the counterterm model and the test oracle
``tools/loop_reference/feynkit_loop_graphs.py``, which enumerates FeynKit's
one-loop diagrams (production never enumerates loop diagrams: the v2 engine
is recursive). This boundary keeps local Feynman-rule factors; it never
requests a contracted or expanded whole-process numerator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .orders import (
    LoopEligibility,
    LoopOrderPlan,
    canonical_order,
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
