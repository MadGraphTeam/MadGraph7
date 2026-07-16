# SPDX-License-Identifier: 0BSD
"""Current-table construction for generic process DAGs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..color.plan import GenericColorPlan, build_color_plan
from ..models import (
    BuiltinSMModel,
    CouplingOrders,
    Model,
    QuantumFlow,
    Vertex,
    VertexEvaluationEquivalence,
)
from ..processes import ProcessOptions
from ..processes.ir import CanonicalProcessIR, build_process_ir
from .dag_algorithms import _normalize_generation_cap
from .dag_color import ColorEngine
from .dag_ordering import (
    _closure_candidate_splits,
    _closure_combination_matches_word,
    _closure_contraction_name,
    _closure_side_reachable_masks,
    _complex_weight_mul,
    _direct_contraction_kind,
    _labels_mask,
    _labels_projected_to_word,
    _lc_all_gluon_symmetry_order_variants,
    _lc_color_order_reachable_masks,
    _mask_labels,
)
from .dag_reachability import (
    _coupling_orders_within_limits,
    _lc_line_groups_within_limit,
    _mask_allowed_by_reachability,
    _masks_by_size,
    _normalize_coupling_order_limits,
    _ordered_splits,
    _right_particles_by_left,
    _state_allowed_by_reachability,
    _useful_states_by_mask,
)
from .dag_table import _CurrentTable
from .dag_types import (
    AmplitudeRoot,
    CurrentIndex,
    GenericDAG,
    InteractionNode,
)


class GenericDAGCompiler:
    """Compile a concrete process into a model-driven current DAG.

    The compiler never classifies the whole process as a family.  It sweeps
    external subsets, asks the model which local vertices are valid for two
    current particle ids, asks the colour engine whether their colour states
    combine, and deduplicates solely by ``CurrentIndex`` equality.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        color_accuracy: str = "lc",
        options: ProcessOptions | None = None,
        max_currents: int | None = None,
        max_color_sectors: int | None = None,
        reference_color_order: tuple[int, ...] | None = None,
        selected_color_sector_ids: Iterable[int] | None = None,
        max_coupling_orders: Mapping[str, int] | None = None,
        max_lc_current_line_groups: int | None = None,
        max_quark_pairs: int | None = None,
        closure_side_mask_pruning: bool = True,
        color_order_mask_pruning: bool = True,
        species_reachability_pruning: bool = True,
        ignored_particle_ids: Iterable[int] | None = None,
        ignored_vertex_kinds: Iterable[int] | None = None,
        selected_source_helicities: Mapping[int, int] | None = None,
        lc_all_ordering_symmetry: bool = True,
    ) -> None:
        self.model = model or BuiltinSMModel()
        self.color_accuracy = color_accuracy
        self.options = options
        self.max_currents = _normalize_generation_cap(max_currents)
        self.max_color_sectors = _normalize_generation_cap(max_color_sectors)
        self.reference_color_order = reference_color_order
        self.selected_color_sector_ids = (
            None
            if selected_color_sector_ids is None
            else frozenset(int(sector_id) for sector_id in selected_color_sector_ids)
        )
        self.max_coupling_orders = _normalize_coupling_order_limits(
            max_coupling_orders,
        )
        self.max_lc_current_line_groups = (
            None
            if max_lc_current_line_groups is None
            else max(0, int(max_lc_current_line_groups))
        )
        self.max_quark_pairs = (
            None if max_quark_pairs is None else max(0, int(max_quark_pairs))
        )
        self.closure_side_mask_pruning = bool(closure_side_mask_pruning)
        self.color_order_mask_pruning = bool(color_order_mask_pruning)
        self.species_reachability_pruning = bool(species_reachability_pruning)
        self.ignored_particle_ids = frozenset(
            int(particle_id) for particle_id in (ignored_particle_ids or ())
        )
        self.ignored_vertex_kinds = frozenset(
            int(kind) for kind in (ignored_vertex_kinds or ())
        )
        self.selected_source_helicities = (
            None
            if selected_source_helicities is None
            else {
                int(label): int(helicity)
                for label, helicity in selected_source_helicities.items()
            }
        )
        self.lc_all_ordering_symmetry = bool(lc_all_ordering_symmetry)

    def compile(self, process: str | CanonicalProcessIR) -> GenericDAG:
        process_ir = (
            process
            if isinstance(process, CanonicalProcessIR)
            else build_process_ir(
                process,
                color_accuracy=self.color_accuracy,
                options=self.options,
            )
        )
        color_plan = build_color_plan(
            process_ir,
            color_accuracy=process_ir.color_accuracy,
            options=self.options,
            max_sectors=self.max_color_sectors,
            reference_color_order=self.reference_color_order,
        )
        if (
            self.max_quark_pairs is not None
            and process_ir.quark_lines.quark_pair_count > self.max_quark_pairs
        ):
            return GenericDAG(
                process=process_ir,
                color_plan=color_plan,
                currents=(),
                sources=(),
                interactions=(),
                amplitude_roots=(),
                truncated=False,
            )
        if (
            self.selected_color_sector_ids is not None
            and color_plan.color_accuracy == "lc"
        ):
            selected_sectors = tuple(
                sector
                for sector in color_plan.sectors
                if sector.id in self.selected_color_sector_ids
            )
            missing_sector_ids = tuple(
                sorted(
                    int(sector_id)
                    for sector_id in self.selected_color_sector_ids
                    if all(sector.id != sector_id for sector in selected_sectors)
                )
            )
            diagnostics = color_plan.diagnostics
            if missing_sector_ids:
                diagnostics = (
                    *diagnostics,
                    "selected LC colour sector ids were not materialized: "
                    + ", ".join(str(sector_id) for sector_id in missing_sector_ids),
                )
            color_plan = GenericColorPlan(
                process=color_plan.process,
                color_accuracy=color_plan.color_accuracy,
                sectors=selected_sectors,
                diagnostics=diagnostics,
                truncated=bool(missing_sector_ids),
                idenso_required=color_plan.idenso_required,
            )
        color_engine = ColorEngine(
            color_plan,
            self.model,
            shared_lc_all_ordering_symmetry=(
                self.lc_all_ordering_symmetry and self.selected_color_sector_ids is None
            ),
        )
        table = _CurrentTable(self.model)
        sources = self._build_sources(process_ir, color_engine, table)
        interactions: list[InteractionNode] = []
        interaction_keys: set[tuple[object, ...]] = set()
        evaluation_group_by_key: dict[tuple[object, ...], int] = {}
        evaluation_equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
        evaluation_equivalence_for = self.model.vertex_evaluation_equivalence
        equivalence_record = VertexEvaluationEquivalence
        vertices_by_input: dict[tuple[int, int], tuple[Vertex, ...]] = {}
        right_particles_by_left = _right_particles_by_left(
            self.model,
            color_accuracy=process_ir.color_accuracy,
        )
        vertex_allowed_cache: dict[tuple[int, int, int, int], bool] = {}
        quantum_flow_cache: dict[tuple[object, ...], tuple[QuantumFlow, ...]] = {}
        coupling_order_cache: dict[
            tuple[CouplingOrders, CouplingOrders, int, tuple[int, int, int]],
            CouplingOrders,
        ] = {}
        state_allowed_cache: dict[
            tuple[int, int, CouplingOrders],
            bool,
        ] = {}
        full_mask = _labels_mask(leg.label for leg in process_ir.legs)
        gluon_labels = frozenset(
            leg.label
            for leg in process_ir.legs
            if leg.outgoing_pdg is not None and abs(int(leg.outgoing_pdg)) == 21
        )
        shared_lc_all_ordering_symmetry = color_engine.shared_lc_all_ordering_symmetry
        closure_candidate_splits = _closure_candidate_splits(
            process_ir,
            self.model,
            color_engine,
            reference_color_order=self.reference_color_order,
        )
        closure_reachable_masks = (
            _closure_side_reachable_masks(
                full_mask,
                closure_candidate_splits,
            )
            if self.closure_side_mask_pruning
            else None
        )
        color_order_reachable_masks = (
            _lc_color_order_reachable_masks(
                process_ir,
                color_plan,
                self.model,
            )
            if self.color_order_mask_pruning
            else None
        )
        useful_states_by_mask = (
            _useful_states_by_mask(
                process_ir,
                self.model,
                color_engine,
                closure_candidate_splits,
                closure_reachable_masks,
                color_order_reachable_masks,
                max_coupling_orders=self.max_coupling_orders,
                ignored_particle_ids=self.ignored_particle_ids,
                ignored_vertex_kinds=self.ignored_vertex_kinds,
            )
            if self.species_reachability_pruning
            else None
        )
        truncated = False
        if any(
            int(leg.outgoing_pdg or 0) in self.ignored_particle_ids
            for leg in process_ir.legs
        ):
            return GenericDAG(
                process=process_ir,
                color_plan=color_plan,
                currents=tuple(table.currents),
                sources=tuple(sources),
                interactions=(),
                amplitude_roots=(),
                truncated=False,
            )

        def state_allowed(
            mask: int,
            particle_id: int,
            coupling_orders: CouplingOrders,
        ) -> bool:
            if useful_states_by_mask is None:
                return True
            key = (mask, particle_id, coupling_orders)
            cached = state_allowed_cache.get(key)
            if cached is not None:
                return cached
            allowed = _state_allowed_by_reachability(
                useful_states_by_mask,
                mask,
                particle_id,
                coupling_orders,
            )
            state_allowed_cache[key] = allowed
            return allowed

        def combined_coupling_orders(
            left_index: CurrentIndex,
            right_index: CurrentIndex,
            vertex: Vertex,
        ) -> CouplingOrders:
            key = (
                left_index.coupling_orders,
                right_index.coupling_orders,
                vertex.kind,
                vertex.particles,
            )
            cached = coupling_order_cache.get(key)
            if cached is not None:
                return cached
            orders = self.model.combine_coupling_orders(
                left_index,
                right_index,
                vertex,
            )
            coupling_order_cache[key] = orders
            return orders

        def all_gluon_current(index: CurrentIndex) -> bool:
            labels = index.external_labels
            return bool(labels) and all(label in gluon_labels for label in labels)

        for mask in _masks_by_size(full_mask):
            if mask & (mask - 1) == 0:
                continue
            if mask == full_mask:
                continue
            if not _mask_allowed_by_reachability(
                mask,
                closure_reachable_masks,
                color_order_reachable_masks,
            ):
                continue
            if useful_states_by_mask is not None and mask not in useful_states_by_mask:
                continue
            labels = _mask_labels(mask)
            for left_mask, right_mask in _ordered_splits(mask):
                if not (
                    _mask_allowed_by_reachability(
                        left_mask,
                        closure_reachable_masks,
                        color_order_reachable_masks,
                    )
                    and _mask_allowed_by_reachability(
                        right_mask,
                        closure_reachable_masks,
                        color_order_reachable_masks,
                    )
                ):
                    continue
                if useful_states_by_mask is not None and (
                    left_mask not in useful_states_by_mask
                    or right_mask not in useful_states_by_mask
                ):
                    continue
                left_ids = table.ids_by_mask(left_mask)
                if not left_ids or not table.has_mask(right_mask):
                    continue
                for left_id in left_ids:
                    left = table.current(left_id)
                    if not state_allowed(
                        left_mask,
                        left.index.particle_id,
                        left.index.coupling_orders,
                    ):
                        continue
                    possible_right_particles = right_particles_by_left.get(
                        left.index.particle_id,
                    )
                    if not possible_right_particles:
                        continue
                    candidate_right_ids = table.ids_by_mask_and_particles(
                        right_mask,
                        possible_right_particles,
                        color_sector_id=(
                            None
                            if color_engine.shared_lc_orderings
                            else left.index.color_state.sector_id
                        ),
                    )
                    if not candidate_right_ids:
                        continue
                    for right_id in candidate_right_ids:
                        right = table.current(right_id)
                        if left.index.overlaps(right.index):
                            continue
                        left_all_gluon = (
                            shared_lc_all_ordering_symmetry
                            and all_gluon_current(left.index)
                        )
                        right_all_gluon = (
                            shared_lc_all_ordering_symmetry
                            and all_gluon_current(right.index)
                        )
                        if (
                            left_all_gluon
                            and right_all_gluon
                            and max(left.index.external_labels)
                            >= max(right.index.external_labels)
                        ):
                            continue
                        if not state_allowed(
                            right_mask,
                            right.index.particle_id,
                            right.index.coupling_orders,
                        ):
                            continue
                        vertex_lookup_key = (
                            left.index.particle_id,
                            right.index.particle_id,
                        )
                        if vertex_lookup_key in vertices_by_input:
                            vertices = vertices_by_input[vertex_lookup_key]
                        else:
                            vertices = self.model.vertices_accepting(
                                left.index.particle_id,
                                right.index.particle_id,
                                color_accuracy=process_ir.color_accuracy,
                            )
                            vertices_by_input[vertex_lookup_key] = vertices
                        for vertex in vertices:
                            if (
                                vertex.kind in self.ignored_vertex_kinds
                                or vertex.particles[2] in self.ignored_particle_ids
                            ):
                                continue
                            if useful_states_by_mask is not None and vertex.particles[
                                2
                            ] not in useful_states_by_mask.get(mask, {}):
                                continue
                            vertex_key = (vertex.kind, *vertex.particles)
                            if vertex_key in vertex_allowed_cache:
                                vertex_allowed = vertex_allowed_cache[vertex_key]
                            else:
                                vertex_allowed = color_engine.vertex_allowed(vertex)
                                vertex_allowed_cache[vertex_key] = vertex_allowed
                            if not vertex_allowed:
                                continue
                            if self.model.skip_duplicate_vertex_orientation(vertex):
                                continue
                            coupling_orders = combined_coupling_orders(
                                left.index,
                                right.index,
                                vertex,
                            )
                            if not _coupling_orders_within_limits(
                                coupling_orders,
                                self.max_coupling_orders,
                            ):
                                continue
                            if not state_allowed(
                                mask,
                                vertex.particles[2],
                                coupling_orders,
                            ):
                                continue
                            ordered_external_labels = (
                                color_engine.ordered_combination_labels(
                                    left.index,
                                    right.index,
                                    vertex,
                                )
                            )
                            if ordered_external_labels is None and not (
                                left_all_gluon or right_all_gluon
                            ):
                                continue
                            order_variants: tuple[
                                tuple[tuple[int, ...], tuple[float, float]],
                                ...,
                            ]
                            if left_all_gluon or right_all_gluon:
                                result_all_gluon = left_all_gluon and right_all_gluon
                                variants: list[
                                    tuple[tuple[int, ...], tuple[float, float]]
                                ] = []
                                for (
                                    proposed_labels,
                                    symmetry_weight,
                                ) in _lc_all_gluon_symmetry_order_variants(
                                    left.index.ordered_external_labels,
                                    right.index.ordered_external_labels,
                                    left_all_gluon=left_all_gluon,
                                    right_all_gluon=right_all_gluon,
                                ):
                                    projected = (
                                        color_engine.shared_lc_ordered_proposed_labels(
                                            proposed_labels,
                                            allow_reversed=result_all_gluon,
                                        )
                                    )
                                    if projected is None:
                                        continue
                                    variants.append((projected, symmetry_weight))
                                order_variants = tuple(variants)
                                if not order_variants:
                                    continue
                            else:
                                if ordered_external_labels is None:
                                    continue
                                order_variants = (
                                    (ordered_external_labels, (1.0, 0.0)),
                                )
                            quantum_flow_key = (
                                vertex.kind,
                                vertex.particles,
                                left.index.particle_id,
                                left.index.chirality,
                                left.index.flavour_flow,
                                right.index.particle_id,
                                right.index.chirality,
                                right.index.flavour_flow,
                            )
                            if quantum_flow_key in quantum_flow_cache:
                                quantum_flows = quantum_flow_cache[quantum_flow_key]
                            else:
                                quantum_flows = self.model.allowed_quantum_flows(
                                    vertex,
                                    left.index,
                                    right.index,
                                )
                                quantum_flow_cache[quantum_flow_key] = quantum_flows
                            for quantum_flow in quantum_flows:
                                for variant_index, (
                                    variant_ordered_labels,
                                    symmetry_weight,
                                ) in enumerate(order_variants):
                                    for color_flow in color_engine.combine(
                                        left.index.color_state,
                                        right.index.color_state,
                                        vertex,
                                        ordered_external_labels=(
                                            variant_ordered_labels
                                        ),
                                    ):
                                        if not _lc_line_groups_within_limit(
                                            color_flow.state,
                                            self.max_lc_current_line_groups,
                                        ):
                                            continue
                                        out_index = CurrentIndex(
                                            particle_id=vertex.particles[2],
                                            external_mask=mask,
                                            external_labels=labels,
                                            ordered_external_labels=variant_ordered_labels,
                                            helicity_ancestry=(
                                                left.index.helicity_ancestry
                                                | right.index.helicity_ancestry
                                            ),
                                            chirality=quantum_flow.chirality,
                                            spin_state=quantum_flow.spin_state,
                                            flavour_flow=quantum_flow.flavour_flow,
                                            charge_flow=quantum_flow.charge_flow,
                                            color_state=color_flow.state,
                                            momentum_mask=(
                                                left.index.momentum_mask
                                                | right.index.momentum_mask
                                            ),
                                            coupling_orders=coupling_orders,
                                            auxiliary_kind=self.model.auxiliary_kind(
                                                vertex.particles[2]
                                            ),
                                        )
                                        if not self.model.current_allowed(out_index):
                                            continue
                                        result = table.add_or_get(
                                            out_index,
                                            is_source=False,
                                        )
                                        signed_color_weight = _complex_weight_mul(
                                            _complex_weight_mul(
                                                color_flow.weight,
                                                self.model.vertex_color_weight(
                                                    vertex,
                                                    color_accuracy=process_ir.color_accuracy,
                                                ),
                                            ),
                                            symmetry_weight,
                                        )
                                        key = (
                                            vertex.kind,
                                            left_id,
                                            right_id,
                                            result.id,
                                            variant_index,
                                            signed_color_weight,
                                        )
                                        if key in interaction_keys:
                                            continue
                                        interaction_keys.add(key)
                                        rule = self.model.vertex_lowering_rule(
                                            vertex.kind
                                        )
                                        equivalence = (
                                            evaluation_equivalence_by_kind.get(
                                                vertex.kind
                                            )
                                        )
                                        if equivalence is None:
                                            equivalence = evaluation_equivalence_for(
                                                vertex.kind
                                            )
                                            if not equivalence.verified:
                                                model_type = (
                                                    f"{type(self.model).__module__}."
                                                    f"{type(self.model).__qualname__}"
                                                )
                                                equivalence = equivalence_record(
                                                    class_id=(
                                                        f"{model_type}:{int(vertex.kind)}"
                                                    )
                                                )
                                            evaluation_equivalence_by_kind[
                                                vertex.kind
                                            ] = equivalence
                                        canonical_inputs = (left_id, right_id)
                                        if equivalence.input_order == (1, 0):
                                            canonical_inputs = (right_id, left_id)
                                        evaluation_key = (
                                            equivalence.class_id,
                                            canonical_inputs,
                                            int(result.index.particle_id),
                                            int(result.index.chirality),
                                            quantum_flow.coupling,
                                        )
                                        evaluation_group_id = (
                                            evaluation_group_by_key.get(evaluation_key)
                                        )
                                        if evaluation_group_id is None:
                                            evaluation_group_id = len(
                                                evaluation_group_by_key
                                            )
                                            evaluation_group_by_key[evaluation_key] = (
                                                evaluation_group_id
                                            )
                                        interactions.append(
                                            InteractionNode(
                                                id=len(interactions),
                                                vertex_kind=vertex.kind,
                                                vertex_particles=vertex.particles,
                                                left_id=left_id,
                                                right_id=right_id,
                                                result_id=result.id,
                                                coupling=quantum_flow.coupling,
                                                color_weight=signed_color_weight,
                                                lowering_backend=rule.backend,
                                                full_tensor_network_ready=(
                                                    rule.full_tensor_network_ready
                                                ),
                                                evaluation_group_id=(
                                                    evaluation_group_id
                                                ),
                                                evaluation_factor=equivalence.factor,
                                            )
                                        )
                                        if (
                                            self.max_currents is not None
                                            and len(table.currents) > self.max_currents
                                        ):
                                            truncated = True
                                            return GenericDAG(
                                                process=process_ir,
                                                color_plan=color_plan,
                                                currents=tuple(table.currents),
                                                sources=tuple(sources),
                                                interactions=tuple(interactions),
                                                amplitude_roots=tuple(
                                                    self._build_amplitude_roots(
                                                        process_ir,
                                                        table,
                                                        color_engine,
                                                        candidate_splits=closure_candidate_splits,
                                                    )
                                                ),
                                                truncated=truncated,
                                            )

        return GenericDAG(
            process=process_ir,
            color_plan=color_plan,
            currents=tuple(table.currents),
            sources=tuple(sources),
            interactions=tuple(interactions),
            amplitude_roots=tuple(
                self._build_amplitude_roots(
                    process_ir,
                    table,
                    color_engine,
                    candidate_splits=closure_candidate_splits,
                )
            ),
            truncated=truncated,
        )

    def _build_sources(
        self,
        process_ir: CanonicalProcessIR,
        color_engine: ColorEngine,
        table: _CurrentTable,
    ) -> list[int]:
        sources: list[int] = []
        next_source_bit = 0
        for leg in process_ir.legs:
            if leg.outgoing_pdg is None:
                continue
            particle_id = int(leg.outgoing_pdg)
            for color_state in color_engine.source_states_for_leg(leg):
                if not _lc_line_groups_within_limit(
                    color_state,
                    self.max_lc_current_line_groups,
                ):
                    continue
                for source_state in self.model.source_spin_states(particle_id):
                    chirality = source_state.chirality
                    source_helicity = source_state.helicity
                    spin_state = source_state.spin_state
                    if leg.is_initial and self.model.is_chiral_eligible(particle_id):
                        chirality = -chirality
                        spin_state = chirality
                    elif leg.is_initial and self.model.is_gluon(particle_id):
                        source_helicity = -source_helicity
                        if not isinstance(spin_state, int):
                            raise TypeError(
                                "gluon source spin_state must be an integer"
                            )
                        spin_state = -spin_state
                    if (
                        self.selected_source_helicities is not None
                        and (
                            requested_helicity := self.selected_source_helicities.get(
                                leg.label
                            )
                        )
                        is not None
                        and int(source_helicity) != requested_helicity
                    ):
                        continue
                    helicity_ancestry = 1 << next_source_bit
                    next_source_bit += 1
                    index = CurrentIndex(
                        particle_id=particle_id,
                        external_mask=1 << (leg.label - 1),
                        external_labels=(leg.label,),
                        ordered_external_labels=(leg.label,),
                        helicity_ancestry=helicity_ancestry,
                        chirality=chirality,
                        spin_state=spin_state,
                        flavour_flow=(particle_id,),
                        charge_flow=self.model.charge_units(particle_id),
                        color_state=color_state,
                        momentum_mask=1 << (leg.label - 1),
                        coupling_orders=(),
                        auxiliary_kind=self.model.auxiliary_kind(particle_id),
                    )
                    current = table.add_or_get(
                        index,
                        is_source=True,
                        source_leg_label=leg.label,
                        source_helicity=source_helicity,
                    )
                    sources.append(current.id)
        return sources

    def _build_amplitude_roots(
        self,
        process_ir: CanonicalProcessIR,
        table: _CurrentTable,
        color_engine: ColorEngine,
        *,
        candidate_splits: tuple[tuple[int, int], ...] | None = None,
    ) -> list[AmplitudeRoot]:
        full_mask = _labels_mask(leg.label for leg in process_ir.legs)
        if candidate_splits is None:
            candidate_splits = _closure_candidate_splits(
                process_ir,
                self.model,
                color_engine,
            )
        roots: list[AmplitudeRoot] = []
        seen: set[tuple[object, ...]] = set()
        for left_mask, right_mask in candidate_splits:
            if left_mask == 0 or right_mask == 0:
                continue
            for left_id in table.ids_by_mask(left_mask):
                left = table.current(left_id)
                for right_id in table.ids_by_mask(right_mask):
                    right = table.current(right_id)
                    if left.index.overlaps(right.index):
                        continue
                    if not color_engine.ordered_closure_allowed(
                        left.index,
                        right.index,
                    ):
                        continue
                    if (
                        self.reference_color_order is not None
                        and process_ir.color_accuracy == "lc"
                    ):
                        sector = color_engine.color_plan.sector(
                            left.index.color_state.sector_id
                        )
                        if (
                            sector is not None
                            and self.reference_color_order in sector.compatibility_words
                            and not _closure_combination_matches_word(
                                _labels_projected_to_word(
                                    left.index.ordered_external_labels,
                                    self.reference_color_order,
                                ),
                                _labels_projected_to_word(
                                    right.index.ordered_external_labels,
                                    self.reference_color_order,
                                ),
                                self.reference_color_order,
                            )
                        ):
                            continue
                    color_flows = (
                        color_engine.shared_single_trace_closure_flows(
                            left.index,
                            right.index,
                        )
                        if color_engine.shared_single_trace
                        else color_engine.shared_lc_closure_flows(
                            left.index,
                            right.index,
                        )
                        if color_engine.shared_lc_orderings
                        else color_engine.closure_compatible(
                            left.index.color_state,
                            right.index.color_state,
                            full_mask=full_mask,
                        )
                    )
                    if not color_flows:
                        continue
                    direct = _direct_contraction_kind(
                        self.model,
                        left.index,
                        right.index,
                    )
                    for color_flow in color_flows:
                        if direct is not None:
                            direct_key: tuple[object, ...] = (
                                "direct",
                                direct,
                                left_id,
                                right_id,
                                color_flow.state,
                            )
                            if direct_key not in seen:
                                seen.add(direct_key)
                                roots.append(
                                    AmplitudeRoot(
                                        id=len(roots),
                                        kind="direct-contraction",
                                        left_id=left_id,
                                        right_id=right_id,
                                        color_weight=color_flow.weight,
                                        color_sector_id=color_flow.state.sector_id,
                                        contraction=direct,
                                    )
                                )
                        for vertex in self.model.vertices_accepting(
                            left.index.particle_id,
                            right.index.particle_id,
                            color_accuracy=process_ir.color_accuracy,
                        ):
                            if (
                                vertex.kind in self.ignored_vertex_kinds
                                or vertex.particles[2] in self.ignored_particle_ids
                            ):
                                continue
                            if not color_engine.vertex_allowed(vertex):
                                continue
                            if not self.model.vertex_closure_allowed(vertex):
                                continue
                            coupling_orders = self.model.combine_coupling_orders(
                                left.index,
                                right.index,
                                vertex,
                            )
                            if not _coupling_orders_within_limits(
                                coupling_orders,
                                self.max_coupling_orders,
                            ):
                                continue
                            closure_contraction = _closure_contraction_name(
                                self.model,
                                vertex.particles[2],
                            )
                            if closure_contraction != "scalar":
                                continue
                            if not self.model.allowed_quantum_flows(
                                vertex,
                                left.index,
                                right.index,
                            ):
                                continue
                            vertex_key: tuple[object, ...] = (
                                "vertex",
                                vertex.kind,
                                vertex.particles,
                                left_id,
                                right_id,
                                color_flow.state,
                            )
                            if vertex_key in seen:
                                continue
                            seen.add(vertex_key)
                            roots.append(
                                AmplitudeRoot(
                                    id=len(roots),
                                    kind="vertex-closure",
                                    left_id=left_id,
                                    right_id=right_id,
                                    color_weight=_complex_weight_mul(
                                        color_flow.weight,
                                        self.model.vertex_color_weight(
                                            vertex,
                                            color_accuracy=process_ir.color_accuracy,
                                        ),
                                    ),
                                    color_sector_id=color_flow.state.sector_id,
                                    vertex_kind=vertex.kind,
                                    vertex_particles=vertex.particles,
                                    coupling=vertex.coupling,
                                    contraction=closure_contraction,
                                )
                            )
        return roots


def compile_generic_dag(
    process: str | CanonicalProcessIR,
    *,
    model: Model | None = None,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
    max_currents: int | None = None,
    max_color_sectors: int | None = None,
    reference_color_order: tuple[int, ...] | None = None,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    max_lc_current_line_groups: int | None = None,
    max_quark_pairs: int | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    species_reachability_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
    selected_source_helicities: Mapping[int, int] | None = None,
    lc_all_ordering_symmetry: bool = True,
) -> GenericDAG:
    return GenericDAGCompiler(
        model=model,
        color_accuracy=color_accuracy,
        options=options,
        max_currents=max_currents,
        max_color_sectors=max_color_sectors,
        reference_color_order=reference_color_order,
        selected_color_sector_ids=selected_color_sector_ids,
        max_coupling_orders=max_coupling_orders,
        max_lc_current_line_groups=max_lc_current_line_groups,
        max_quark_pairs=max_quark_pairs,
        closure_side_mask_pruning=closure_side_mask_pruning,
        color_order_mask_pruning=color_order_mask_pruning,
        species_reachability_pruning=species_reachability_pruning,
        ignored_particle_ids=ignored_particle_ids,
        ignored_vertex_kinds=ignored_vertex_kinds,
        selected_source_helicities=selected_source_helicities,
        lc_all_ordering_symmetry=lc_all_ordering_symmetry,
    ).compile(process)
