# SPDX-License-Identifier: 0BSD
"""FeynKit (HepKit) one-loop diagram catalogue: a test oracle only.

Moved here from ``pyamplicol.loops.graphs`` when the per-diagram legacy loop
engine was removed (user ruling 2026-10-28, M-QCD Q3): production never
enumerates loop diagrams (the v2 engine is recursive); FeynKit's diagrams and
integral families stay an independent oracle of the v2 family sets
(``tests/unit/test_loop_v2_families.py``).

HepKit owns enumeration, graph identity, cycles, signs, routing and integral
family equivalence. Selection follows MadLoop's
``filter_loop_for_perturbative_orders`` and ``is_wf_correction`` /
``is_vanishing_tadpole``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from pyamplicol.loops.graphs import (
    FamilyAssignment,
    GraphIR,
    IntegralFamilyIR,
    LoopProcessIR,
    RejectedGraph,
    _propagator_denominators,
    _rule_orders,
    _snapshot,
    _species_orders,
)
from pyamplicol.loops.orders import (
    BornLoopPair,
    LoopContribution,
    compile_order_plan,
)


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
