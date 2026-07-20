# SPDX-License-Identifier: 0BSD
"""Amplitude grouping and color-contraction runtime records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..color import (
    ColorGroupDescriptor,
    build_color_contraction_plan,
)
from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import AmplitudeRoot, CurrentNode, GenericDAG


def build_runtime_amplitude_stage(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> dict[str, object]:
    """Build evaluator roots and physical coherent-amplitude metadata."""

    group_ids, descriptors = _amplitude_groups(dag, dag.amplitude_roots)
    color_contraction = build_color_contraction_plan(dag.color_plan, descriptors)
    multiple_lc_sectors = _has_multiple_lc_root_sectors(dag)
    roots: list[dict[str, object]] = []
    group_weights: dict[int, tuple[float, float]] = {}
    for output_index, root in enumerate(dag.amplitude_roots):
        group_id = group_ids[root.id]
        all_sector_weight = _root_all_sector_weight(
            dag,
            root,
            has_multiple_lc_root_sectors=multiple_lc_sectors,
        )
        group_weights.setdefault(
            group_id,
            (float(root.helicity_weight), float(all_sector_weight)),
        )
        roots.append(
            {
                "output_index": output_index,
                "root_id": output_index,
                "dag_root_id": root.id,
                "kind": root.kind,
                "left_current_id": root.left_id,
                "right_current_id": root.right_id,
                "left_slot": _current_slot_ref(current_slots[root.left_id]),
                "right_slot": _current_slot_ref(current_slots[root.right_id]),
                "left_value_slot": _value_slot_ref(
                    _amplitude_value_slot(dag.currents[root.left_id], value_slots)
                ),
                "right_value_slot": _value_slot_ref(
                    _amplitude_value_slot(dag.currents[root.right_id], value_slots)
                ),
                "vertex_kind": root.vertex_kind,
                "vertex_particles": (
                    None
                    if root.vertex_particles is None
                    else list(root.vertex_particles)
                ),
                "coupling": list(root.coupling),
                "coupling_parameter_names": (
                    None
                    if root.vertex_kind is None or root.vertex_particles is None
                    else runtime_coupling_parameter_names(
                        root.vertex_kind,
                        root.vertex_particles,
                        root.coupling,
                        model=model,
                    )
                ),
                "color_weight": list(root.color_weight),
                "color_sector_id": _root_color_sector_id(dag, root),
                "contraction": root.contraction,
                "contraction_ir": root.contraction_ir.to_json_dict(),
                "coherent_group_id": group_id,
                "helicity_weight": root.helicity_weight,
                "all_sector_weight": all_sector_weight,
            }
        )

    coherent_groups = []
    for descriptor in descriptors:
        helicity_weight, all_sector_weight = group_weights[descriptor.group_id]
        coherent_groups.append(
            {
                "group_id": descriptor.group_id,
                "helicities": _helicity_vector(dag, descriptor.helicity_key),
                "color_sector_id": descriptor.sector_id,
                "color_word": list(descriptor.word),
                "helicity_weight": helicity_weight,
                "all_sector_weight": all_sector_weight,
            }
        )

    return {
        "stage_kind": "amplitude-roots",
        "output_count": len(roots),
        "selected_color_sector_ids": None,
        "coherent_groups": coherent_groups,
        "roots": roots,
        "final_reduction": {
            "status": (
                "sparse-color-contraction"
                if color_contraction is not None
                else "coherent-leading-color-diagonal"
            ),
            "operation": (
                "sum root outputs into coherent helicity/color amplitudes, "
                "then apply the requested color contraction"
            ),
        },
        "color_contraction": (
            None if color_contraction is None else color_contraction.to_json_dict()
        ),
    }


def _amplitude_groups(
    dag: GenericDAG,
    roots: Sequence[AmplitudeRoot],
) -> tuple[dict[int, int], tuple[ColorGroupDescriptor, ...]]:
    ids: dict[tuple[object, ...], int] = {}
    result: dict[int, int] = {}
    descriptors: dict[int, ColorGroupDescriptor] = {}
    source_by_ancestry = {
        int(current.index.helicity_ancestry): (
            int(current.source_leg_label or 0),
            int(current.index.particle_id),
            int(current.index.chirality),
            current.index.spin_state,
            current.source_helicity,
        )
        for current in dag.currents
        if current.is_source
    }
    single_bit_ancestry = all(
        bit > 0 and bit & (bit - 1) == 0 for bit in source_by_ancestry
    )
    physical_sources_cache: dict[int, tuple[object, ...]] = {}

    def physical_sources(ancestry: int) -> tuple[object, ...]:
        cached = physical_sources_cache.get(ancestry)
        if cached is not None:
            return cached
        if single_bit_ancestry:
            sources: list[tuple[object, ...]] = []
            remaining = ancestry
            while remaining:
                bit = remaining & -remaining
                source = source_by_ancestry.get(bit)
                if source is not None:
                    sources.append(source)
                remaining ^= bit
            value: tuple[object, ...] = tuple(sorted(sources))
        else:
            value = tuple(
                sorted(
                    source
                    for bit, source in source_by_ancestry.items()
                    if ancestry & bit
                )
            )
        physical_sources_cache[ancestry] = value
        return value

    for root in roots:
        left = dag.currents[root.left_id].index
        right = dag.currents[root.right_id].index
        ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
        source_key = physical_sources(ancestry)
        sector_id = _root_color_sector_id(dag, root)
        sector = dag.color_plan.sector(sector_id)
        word = (
            tuple(sector.word_labels or sector.color_words[0])
            if sector is not None and sector.color_words
            else ()
        )
        color_key = (
            left.color_state.accuracy,
            word or sector_id,
            tuple(
                sorted(
                    set(left.color_state.basis_key) | set(right.color_state.basis_key)
                )
            ),
            tuple(root.color_weight),
        )
        key = (source_key or ancestry, color_key)
        group_id = ids.setdefault(key, len(ids))
        result[root.id] = group_id
        descriptors.setdefault(
            group_id,
            ColorGroupDescriptor(
                group_id=group_id,
                helicity_key=source_key or (ancestry,),
                sector_id=sector_id,
                word=word,
                helicity_weight=float(root.helicity_weight),
            ),
        )
    return result, tuple(descriptors[index] for index in sorted(descriptors))


def _helicity_vector(
    dag: GenericDAG,
    helicity_key: tuple[object, ...],
) -> list[int]:
    by_label: dict[int, int] = {}
    for item in helicity_key:
        if not isinstance(item, tuple) or len(item) < 5:
            continue
        by_label[int(item[0])] = int(item[4])
    return [by_label.get(leg.label, 0) for leg in dag.process.legs]


def _root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    return int(dag.currents[root.left_id].index.color_state.sector_id)


def _root_all_sector_weight(
    dag: GenericDAG,
    root: AmplitudeRoot,
    *,
    has_multiple_lc_root_sectors: bool,
) -> float:
    weight = float(root.helicity_weight)
    if dag.process.color_accuracy != "lc":
        return weight
    if dag.color_plan.process.color_endpoints.pair_count != 0:
        return weight
    if not dag.color_plan.trace_reflections_folded:
        return weight
    if not has_multiple_lc_root_sectors:
        return weight
    sector = dag.color_plan.sector(_root_color_sector_id(dag, root))
    if sector is None or sector.kind != "single-trace":
        return weight
    trace = tuple(int(label) for label in sector.trace_labels)
    if len(trace) <= 2 or trace[1:] == tuple(reversed(trace[1:])):
        return weight
    return 2.0 * weight


def _has_multiple_lc_root_sectors(dag: GenericDAG) -> bool:
    if dag.process.color_accuracy != "lc":
        return False
    if dag.color_plan.process.color_endpoints.pair_count != 0:
        return False
    return len({_root_color_sector_id(dag, root) for root in dag.amplitude_roots}) > 1


def _current_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "current_id",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


def _value_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "value_slot_id",
            "current_id",
            "variant",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


def _amplitude_value_slot(
    current: CurrentNode,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> Mapping[str, object]:
    variant = "source" if current.is_source else "unpropagated"
    try:
        return value_slots[(current.id, variant)]
    except KeyError as exc:
        raise ValueError(
            f"amplitude current {current.id} has no {variant} runtime value slot"
        ) from exc


__all__ = ["build_runtime_amplitude_stage"]
