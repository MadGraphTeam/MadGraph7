# SPDX-License-Identifier: 0BSD
"""Post-processing and selection transforms for generic DAGs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from ..color.plan import GenericColorPlan, build_color_plan
from ..models import Model, Vertex
from ..processes.ir import CanonicalProcessIR
from .dag_color import ColorEngine
from .dag_ordering import (
    _closure_candidate_splits,
    _closure_side_reachable_masks,
    _labels_mask,
    _lc_color_order_reachable_masks,
)
from .dag_reachability import (
    _closure_total_coupling_orders,
    _coupling_order_degree,
    _coupling_order_envelope,
    _normalize_coupling_order_limits,
)
from .dag_types import (
    AmplitudeRoot,
    CurrentNode,
    GenericDAG,
    InteractionNode,
)


def _normalize_generation_cap(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    return None if normalized < 0 else normalized


def infer_minimal_coupling_order_limits(
    process: CanonicalProcessIR,
    *,
    model: Model,
    max_color_sectors: int | None = None,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
) -> dict[str, int]:
    """Infer a generic lowest-order coupling envelope for a process.

    This is an opt-in generation accelerator.  It never recognizes a whole
    process family; it asks the model which local vertices can connect the
    external states, tracks UFO-style coupling orders, and returns the
    component-wise maximum over all closure paths with the lowest
    model-declared hierarchy-weighted order. The returned dictionary can be
    used as ordinary ``max_coupling_orders``. Model-declared orders absent from
    the minimal envelope are returned with a zero limit because an omitted
    limit means unrestricted to the DAG compiler.
    """

    active_model = model
    if not isinstance(process, CanonicalProcessIR):
        raise TypeError(
            "coupling-order inference requires a model-resolved CanonicalProcessIR"
        )
    process_ir = process
    color_plan = build_color_plan(
        process_ir,
        color_accuracy=process_ir.color_accuracy,
        max_sectors=max_color_sectors,
        fold_trace_reflections=(
            active_model.lc_trace_reflection_equivalence_is_proven(process_ir)
        ),
    )
    explicit_sector_ids = (
        None
        if selected_color_sector_ids is None
        else frozenset(int(sector_id) for sector_id in selected_color_sector_ids)
    )
    if explicit_sector_ids is not None and color_plan.color_accuracy == "lc":
        color_plan = GenericColorPlan(
            process=color_plan.process,
            color_accuracy=color_plan.color_accuracy,
            sectors=tuple(
                sector
                for sector in color_plan.sectors
                if sector.id in explicit_sector_ids
            ),
            diagnostics=color_plan.diagnostics,
            truncated=color_plan.truncated,
            idenso_required=color_plan.idenso_required,
            trace_reflections_folded=color_plan.trace_reflections_folded,
        )
    color_engine = ColorEngine(color_plan, active_model)
    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    closure_candidate_splits = _closure_candidate_splits(
        process_ir,
        active_model,
        color_engine,
    )
    closure_reachable_masks = (
        _closure_side_reachable_masks(full_mask, closure_candidate_splits)
        if closure_side_mask_pruning
        else None
    )
    color_order_reachable_masks = (
        _lc_color_order_reachable_masks(process_ir, color_plan, active_model)
        if color_order_mask_pruning
        else None
    )
    totals = _closure_total_coupling_orders(
        process_ir,
        active_model,
        color_engine,
        closure_candidate_splits,
        closure_reachable_masks,
        color_order_reachable_masks,
        max_coupling_orders=_normalize_coupling_order_limits(max_coupling_orders),
        ignored_particle_ids=frozenset(
            int(particle_id) for particle_id in (ignored_particle_ids or ())
        ),
        ignored_vertex_kinds=frozenset(
            int(kind) for kind in (ignored_vertex_kinds or ())
        ),
    )
    if not totals:
        return {}
    hierarchies = {
        str(name).upper(): max(1, int(value))
        for name, value in active_model.coupling_order_hierarchies().items()
    }
    minimum_total_order = min(
        _coupling_order_degree(total, hierarchies=hierarchies) for total in totals
    )
    minimal_totals = tuple(
        total
        for total in totals
        if _coupling_order_degree(total, hierarchies=hierarchies) == minimum_total_order
    )
    envelope = _coupling_order_envelope(minimal_totals)
    order_names = set(hierarchies) | set(envelope)
    for total in totals:
        order_names.update(name for name, _value in total)
    return {name: envelope.get(name, 0) for name in sorted(order_names)}


def prune_dag_to_amplitude_roots(dag: GenericDAG) -> GenericDAG:
    """Drop currents and interactions that cannot feed any amplitude root.

    The forward generic sweep intentionally over-generates valid local currents:
    it does not know which of them will survive closure until the full table is
    built.  Production artifacts should mirror AmpliCol's optimized library
    structure and keep only the backward-reachable sub-DAG feeding retained
    amplitude roots.
    """

    if not dag.amplitude_roots:
        return dag

    interactions_by_result: dict[int, list[InteractionNode]] = {}
    for interaction in dag.interactions:
        interactions_by_result.setdefault(interaction.result_id, []).append(interaction)

    required_current_ids: set[int] = set()
    stack: list[int] = []
    for root in dag.amplitude_roots:
        for current_id in (root.left_id, root.right_id):
            if current_id not in required_current_ids:
                required_current_ids.add(current_id)
                stack.append(current_id)

    required_interaction_ids: set[int] = set()
    while stack:
        current_id = stack.pop()
        for interaction in interactions_by_result.get(current_id, ()):
            if interaction.id in required_interaction_ids:
                continue
            required_interaction_ids.add(interaction.id)
            for parent_id in (interaction.left_id, interaction.right_id):
                if parent_id not in required_current_ids:
                    required_current_ids.add(parent_id)
                    stack.append(parent_id)

    if len(required_current_ids) == len(dag.currents) and len(
        required_interaction_ids
    ) == len(dag.interactions):
        return dag

    current_id_map = {
        old_id: new_id for new_id, old_id in enumerate(sorted(required_current_ids))
    }
    pruned_currents = tuple(
        CurrentNode(
            id=current_id_map[current.id],
            index=current.index,
            dimension=current.dimension,
            is_source=current.is_source,
            source_leg_label=current.source_leg_label,
            source_helicity=current.source_helicity,
        )
        for current in dag.currents
        if current.id in required_current_ids
    )
    interaction_id_map = {
        old_id: new_id for new_id, old_id in enumerate(sorted(required_interaction_ids))
    }
    pruned_interactions = tuple(
        InteractionNode(
            id=interaction_id_map[interaction.id],
            vertex_kind=interaction.vertex_kind,
            vertex_particles=interaction.vertex_particles,
            left_id=current_id_map[interaction.left_id],
            right_id=current_id_map[interaction.right_id],
            result_id=current_id_map[interaction.result_id],
            coupling=interaction.coupling,
            color_weight=interaction.color_weight,
            lowering_backend=interaction.lowering_backend,
            full_tensor_network_ready=interaction.full_tensor_network_ready,
            evaluation_group_id=interaction.evaluation_group_id,
            evaluation_factor=interaction.evaluation_factor,
        )
        for interaction in dag.interactions
        if interaction.id in required_interaction_ids
    )
    pruned_roots = tuple(
        AmplitudeRoot(
            id=new_id,
            kind=root.kind,
            left_id=current_id_map[root.left_id],
            right_id=current_id_map[root.right_id],
            color_weight=root.color_weight,
            color_sector_id=root.color_sector_id,
            vertex_kind=root.vertex_kind,
            vertex_particles=root.vertex_particles,
            coupling=root.coupling,
            contraction=root.contraction,
            helicity_weight=root.helicity_weight,
        )
        for new_id, root in enumerate(dag.amplitude_roots)
    )
    pruned_sources = tuple(
        current_id_map[source_id]
        for source_id in dag.sources
        if source_id in required_current_ids
    )
    return GenericDAG(
        process=dag.process,
        color_plan=dag.color_plan,
        currents=pruned_currents,
        sources=pruned_sources,
        interactions=pruned_interactions,
        amplitude_roots=pruned_roots,
        truncated=dag.truncated,
    )


def prune_global_helicity_flip_equivalent_roots(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Group roots when the model proves a global-helicity-flip identity.

    This is the safe, structural subset of AmpliCol's numerical helicity
    filtering. Keeping one proven representative with doubled helicity weight
    reduces amplitude roots, after which dead-tree pruning removes currents
    that fed only the discarded partner roots.
    """

    if not _global_helicity_flip_equivalence_safe(dag, model):
        return dag
    if not dag.amplitude_roots:
        return dag

    source_by_bit = _source_helicity_signature_by_bit(dag)
    pure_massless_adjoint = _pure_massless_adjoint_helicity_pruning_safe(dag, model)
    initial_leg_labels = {
        int(leg.label) for leg in dag.process.legs if leg.side == "initial"
    }
    roots_by_signature: dict[tuple[object, ...], list[AmplitudeRoot]] = {}
    zero_pruned = False
    for root in dag.amplitude_roots:
        signature = _root_physical_helicity_signature(dag, root, source_by_bit)
        if pure_massless_adjoint and _pure_massless_adjoint_helicity_signature_is_zero(
            signature,
            initial_leg_labels,
        ):
            zero_pruned = True
            continue
        roots_by_signature.setdefault(
            signature,
            [],
        ).append(root)
    if not roots_by_signature:
        return dag

    handled: set[tuple[object, ...]] = set()
    retained: list[AmplitudeRoot] = []
    changed = False
    for signature in sorted(roots_by_signature):
        if signature in handled:
            continue
        flipped = _flip_root_physical_helicity_signature(signature)
        partner = roots_by_signature.get(flipped)
        handled.add(signature)
        if partner is not None:
            handled.add(flipped)
        weight = 1.0
        if partner is not None and flipped != signature:
            weight = 2.0
            changed = True
        for root in roots_by_signature[signature]:
            retained.append(
                AmplitudeRoot(
                    id=len(retained),
                    kind=root.kind,
                    left_id=root.left_id,
                    right_id=root.right_id,
                    color_weight=root.color_weight,
                    color_sector_id=root.color_sector_id,
                    vertex_kind=root.vertex_kind,
                    vertex_particles=root.vertex_particles,
                    coupling=root.coupling,
                    contraction=root.contraction,
                    helicity_weight=root.helicity_weight * weight,
                )
            )

    if not changed and not zero_pruned:
        return dag
    return prune_dag_to_amplitude_roots(
        GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=dag.currents,
            sources=dag.sources,
            interactions=dag.interactions,
            amplitude_roots=tuple(retained),
            truncated=dag.truncated,
        )
    )


def _global_helicity_flip_equivalence_safe(
    dag: GenericDAG,
    model: Model,
) -> bool:
    if dag.process.color_accuracy not in {"lc", "nlc", "full"}:
        return False
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            return False
        pdg = int(leg.outgoing_pdg)
        if not (
            model.is_massless_adjoint_vector(pdg)
            or model.is_fundamental_colored_fermion(pdg)
        ):
            return False
        if model.mass(pdg) != 0.0:
            return False
    vertices = [
        Vertex(interaction.vertex_kind, interaction.vertex_particles)
        for interaction in dag.interactions
    ]
    vertices.extend(
        Vertex(root.vertex_kind, root.vertex_particles)
        for root in dag.amplitude_roots
        if root.vertex_kind is not None and root.vertex_particles is not None
    )
    return model.global_helicity_flip_equivalence_is_proven(vertices)


def _pure_massless_adjoint_helicity_pruning_safe(
    dag: GenericDAG,
    model: Model,
) -> bool:
    vertices = [
        Vertex(interaction.vertex_kind, interaction.vertex_particles)
        for interaction in dag.interactions
    ]
    vertices.extend(
        Vertex(root.vertex_kind, root.vertex_particles)
        for root in dag.amplitude_roots
        if root.vertex_kind is not None and root.vertex_particles is not None
    )
    return model.pure_massless_adjoint_helicity_zero_rule_is_proven(
        dag.process,
        vertices,
    )


def _pure_massless_adjoint_helicity_signature_is_zero(
    signature: tuple[object, ...],
    initial_leg_labels: set[int],
) -> bool:
    _sector_id, source_helicities = signature
    helicities = []
    for label, helicity in cast(Sequence[tuple[int, int]], source_helicities):
        value = int(helicity)
        if int(label) in initial_leg_labels:
            value = -value
        helicities.append(value)
    return helicities.count(1) < 2 or helicities.count(-1) < 2


def _root_physical_helicity_signature(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> tuple[object, ...]:
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
    source_helicities = tuple(
        sorted(source for bit, source in source_by_bit.items() if ancestry & bit)
    )
    sector_id = _amplitude_root_color_sector_id(dag, root)
    return (sector_id, source_helicities)


def _root_source_helicity_mapping(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> dict[int, int]:
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
    return {
        int(label): int(helicity)
        for bit, (label, helicity) in source_by_bit.items()
        if ancestry & bit
    }


def _source_helicity_signature_by_bit(
    dag: GenericDAG,
) -> dict[int, tuple[int, int]]:
    source_by_bit: dict[int, tuple[int, int]] = {}
    for current in dag.currents:
        if not current.is_source:
            continue
        source_by_bit[int(current.index.helicity_ancestry)] = (
            int(current.source_leg_label or 0),
            int(current.source_helicity or 0),
        )
    return source_by_bit


def _flip_root_physical_helicity_signature(
    signature: tuple[object, ...],
) -> tuple[object, ...]:
    sector_id, source_helicities = signature
    return (
        sector_id,
        tuple(
            sorted(
                (int(label), -int(helicity))
                for label, helicity in cast(
                    Sequence[tuple[int, int]],
                    source_helicities,
                )
            )
        ),
    )


def contributing_color_sector_ids(dag: GenericDAG) -> tuple[int, ...]:
    """Return LC colour sectors that actually contribute amplitude roots."""

    return tuple(
        sorted(
            {_amplitude_root_color_sector_id(dag, root) for root in dag.amplitude_roots}
        )
    )


def _amplitude_root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    left = dag.currents[root.left_id].index
    if left.color_state.accuracy in {"lc", "nlc", "full"}:
        return int(left.color_state.sector_id)
    return 0


def filter_dag_to_color_sectors(
    dag: GenericDAG,
    sector_ids: Iterable[int],
) -> GenericDAG:
    """Return a dense-current DAG restricted to the requested colour sectors.

    Full DAG construction remains useful for diagnostics, but schema-v3 runtime
    records expect dense current ids. This helper derives the runtime DAG by selecting
    roots whose LC colour sector is in ``sector_ids``, walking backward through
    the current DAG, and remapping the required currents/interactions densely.
    Root-based filtering is required for shared LC all-ordering DAGs where
    internal currents are sector-neutral but amplitude roots still carry the
    physical sector identity.
    """

    selected = set(sector_ids)
    if not selected:
        return GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=(),
            sources=(),
            interactions=(),
            amplitude_roots=(),
            truncated=dag.truncated,
        )

    selected_roots = tuple(
        root
        for root in dag.amplitude_roots
        if _amplitude_root_color_sector_id(dag, root) in selected
    )
    if not selected_roots:
        return GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=(),
            sources=(),
            interactions=(),
            amplitude_roots=(),
            truncated=dag.truncated,
        )

    interactions_by_result: dict[int, list[InteractionNode]] = {}
    for interaction in dag.interactions:
        interactions_by_result.setdefault(interaction.result_id, []).append(interaction)

    required_current_ids: set[int] = set()
    required_interaction_ids: set[int] = set()
    stack: list[int] = []
    for root in selected_roots:
        for current_id in (root.left_id, root.right_id):
            if current_id not in required_current_ids:
                required_current_ids.add(current_id)
                stack.append(current_id)

    while stack:
        current_id = stack.pop()
        for interaction in interactions_by_result.get(current_id, ()):
            if interaction.id in required_interaction_ids:
                continue
            required_interaction_ids.add(interaction.id)
            for parent_id in (interaction.left_id, interaction.right_id):
                if parent_id not in required_current_ids:
                    required_current_ids.add(parent_id)
                    stack.append(parent_id)

    current_id_map: dict[int, int] = {}
    currents: list[CurrentNode] = []
    for current in dag.currents:
        if current.id not in required_current_ids:
            continue
        new_id = len(currents)
        current_id_map[current.id] = new_id
        currents.append(
            CurrentNode(
                id=new_id,
                index=current.index,
                dimension=current.dimension,
                is_source=current.is_source,
                source_leg_label=current.source_leg_label,
                source_helicity=current.source_helicity,
            )
        )

    sources = tuple(
        current_id_map[source_id]
        for source_id in dag.sources
        if source_id in current_id_map
    )

    interactions: list[InteractionNode] = []
    for interaction in dag.interactions:
        if interaction.id not in required_interaction_ids:
            continue
        if (
            interaction.left_id not in current_id_map
            or interaction.right_id not in current_id_map
            or interaction.result_id not in current_id_map
        ):
            continue
        interactions.append(
            InteractionNode(
                id=len(interactions),
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                left_id=current_id_map[interaction.left_id],
                right_id=current_id_map[interaction.right_id],
                result_id=current_id_map[interaction.result_id],
                coupling=interaction.coupling,
                color_weight=interaction.color_weight,
                lowering_backend=interaction.lowering_backend,
                full_tensor_network_ready=interaction.full_tensor_network_ready,
                evaluation_group_id=interaction.evaluation_group_id,
                evaluation_factor=interaction.evaluation_factor,
            )
        )

    amplitude_roots: list[AmplitudeRoot] = []
    for root in selected_roots:
        if root.left_id not in current_id_map or root.right_id not in current_id_map:
            continue
        amplitude_roots.append(
            AmplitudeRoot(
                id=len(amplitude_roots),
                kind=root.kind,
                left_id=current_id_map[root.left_id],
                right_id=current_id_map[root.right_id],
                color_weight=root.color_weight,
                color_sector_id=root.color_sector_id,
                vertex_kind=root.vertex_kind,
                vertex_particles=root.vertex_particles,
                coupling=root.coupling,
                contraction=root.contraction,
                helicity_weight=root.helicity_weight,
            )
        )

    return GenericDAG(
        process=dag.process,
        color_plan=dag.color_plan,
        currents=tuple(currents),
        sources=sources,
        interactions=tuple(interactions),
        amplitude_roots=tuple(amplitude_roots),
        truncated=dag.truncated,
    )


def filter_dag_to_source_helicities(
    dag: GenericDAG,
    source_helicities: Mapping[int, int],
) -> GenericDAG:
    """Return a DAG restricted to roots matching fixed source helicities."""

    requested = {
        int(label): int(helicity) for label, helicity in source_helicities.items()
    }
    if not requested:
        return dag

    source_by_bit = _source_helicity_signature_by_bit(dag)
    selected_roots = tuple(
        root
        for root in dag.amplitude_roots
        if _root_source_helicity_mapping(dag, root, source_by_bit).items()
        >= requested.items()
    )
    if len(selected_roots) == len(dag.amplitude_roots):
        return dag
    return prune_dag_to_amplitude_roots(
        GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=dag.currents,
            sources=dag.sources,
            interactions=dag.interactions,
            amplitude_roots=selected_roots,
            truncated=dag.truncated,
        )
    )
