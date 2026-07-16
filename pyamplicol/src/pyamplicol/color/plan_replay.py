# SPDX-License-Identifier: 0BSD
"""Topology grouping and replay safety for color sectors."""

from __future__ import annotations

from ..processes.ir import CanonicalProcessIR
from .plan_build import _ordered_open_line_blocks
from .plan_types import (
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorSectorTopologyGroup,
)


def _sector_topology_groups(
    process: CanonicalProcessIR,
    sectors: tuple[LCColorSector, ...],
) -> tuple[LCColorSectorTopologyGroup, ...]:
    by_signature: dict[tuple[object, ...], list[LCColorSector]] = {}
    for sector in sectors:
        by_signature.setdefault(
            _sector_topology_signature(process, sector),
            [],
        ).append(sector)

    groups: list[LCColorSectorTopologyGroup] = []
    for signature, sector_group in by_signature.items():
        representative = sector_group[0]
        representative_labels = _sector_topology_labels(representative)
        permutations: list[tuple[tuple[int, int], ...]] = []
        for sector in sector_group:
            sector_labels = _sector_topology_labels(sector)
            if len(sector_labels) != len(representative_labels):
                raise ValueError("isomorphic colour sectors have mismatched labels")
            permutations.append(
                tuple(zip(representative_labels, sector_labels, strict=True))
            )
        groups.append(
            LCColorSectorTopologyGroup(
                signature=signature,
                representative_sector_id=representative.id,
                sector_ids=tuple(sector.id for sector in sector_group),
                label_permutations=tuple(permutations),
            )
        )
    return tuple(groups)


def _sector_topology_signature(
    process: CanonicalProcessIR,
    sector: LCColorSector,
) -> tuple[object, ...]:
    pdg_by_label = {
        leg.label: leg.outgoing_pdg
        for leg in process.legs
        if leg.outgoing_pdg is not None
    }
    if sector.kind == "open-lines":
        line_by_coloured = {
            line.coloured_labels: line for line in sector.open_color_lines
        }
        ordered_blocks = (
            _ordered_open_line_blocks(sector.word_labels, sector.open_color_lines)
            if sector.word_labels
            else None
        )
        ordered_lines = (
            tuple(line_by_coloured[block] for block in ordered_blocks)
            if ordered_blocks is not None
            else sector.open_color_lines
        )
        return (
            sector.kind,
            tuple(
                (
                    pdg_by_label[line.fundamental_label],
                    tuple(pdg_by_label[label] for label in line.adjoint_labels),
                    pdg_by_label[line.antifundamental_label],
                    tuple(pdg_by_label[label] for label in line.singlet_labels),
                )
                for line in ordered_lines
            ),
        )
    if sector.kind == "single-trace":
        return (
            sector.kind,
            tuple(pdg_by_label[label] for label in sector.trace_labels),
            tuple(pdg_by_label[label] for label in sector.singlet_labels),
        )
    return (
        sector.kind,
        tuple(pdg_by_label[label] for label in sector.singlet_labels),
    )


def _sector_topology_labels(sector: LCColorSector) -> tuple[int, ...]:
    if sector.kind == "open-lines":
        line_by_coloured = {
            line.coloured_labels: line for line in sector.open_color_lines
        }
        ordered_blocks = (
            _ordered_open_line_blocks(sector.word_labels, sector.open_color_lines)
            if sector.word_labels
            else None
        )
        ordered_lines = (
            tuple(line_by_coloured[block] for block in ordered_blocks)
            if ordered_blocks is not None
            else sector.open_color_lines
        )
        return tuple(label for line in ordered_lines for label in line.line_labels)
    if sector.kind == "single-trace":
        return (*sector.trace_labels, *sector.singlet_labels)
    return sector.singlet_labels


def _jsonable_signature(signature: tuple[object, ...]) -> list[object]:
    def convert(value: object) -> object:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return [convert(item) for item in signature]


def lc_topology_replay_safe_groups(
    color_plan: GenericColorPlan,
) -> tuple[LCColorSectorTopologyGroup, ...]:
    """Return LC topology groups safe for physical-input replay.

    Runtime replay currently evaluates one representative sector with physical
    external momenta in the user's process ordering.  Therefore every replay
    permutation must preserve the initial-state label set.  Pure single-trace
    sectors are also kept out of this fast path for now because their trace
    symmetries need dedicated colour-convention validation.
    """

    if color_plan.color_accuracy != "lc":
        return ()
    return tuple(
        group
        for group in color_plan.topology_groups
        if _lc_topology_group_replay_safe(color_plan, group)
    )


def lc_topology_replay_partitions(
    color_plan: GenericColorPlan,
) -> tuple[LCColorSectorReplayPartition, ...]:
    """Partition LC topology groups into exact replay-safe representatives.

    The selected-sector ``lc_topology_replay_safe_groups`` helper only accepts
    groups whose full topology orbit preserves the initial-state label set.
    All-flow replay artifacts can be more general: they split one topology
    group into several initial-label-safe blocks and materialize one
    representative sidecar per block.  This also covers pure single-trace
    adjoint sectors without falling back to an all-sector artifact.
    """

    if color_plan.color_accuracy != "lc":
        return ()
    initial_labels = {leg.label for leg in color_plan.process.initial_legs}
    if not initial_labels:
        return ()
    partitions: list[LCColorSectorReplayPartition] = []
    for group in color_plan.topology_groups:
        sectors = tuple(
            sector
            for sector_id in group.sector_ids
            if (sector := color_plan.sector(sector_id)) is not None
        )
        if len(sectors) != len(group.sector_ids):
            continue
        if any(sector.kind not in {"open-lines", "single-trace"} for sector in sectors):
            continue
        base_maps = {
            int(sector_id): {
                int(representative_label): int(sector_label)
                for representative_label, sector_label in permutation
            }
            for sector_id, permutation in zip(
                group.sector_ids,
                group.label_permutations,
                strict=True,
            )
        }
        sector_ids_by_initial_preimage: dict[tuple[int, ...], list[int]] = {}
        complete_initial_maps = True
        for sector_id in group.sector_ids:
            sector_map = base_maps[int(sector_id)]
            inverse_sector_map = {
                sector_label: representative_label
                for representative_label, sector_label in sector_map.items()
            }
            if not initial_labels.issubset(inverse_sector_map):
                complete_initial_maps = False
                break
            initial_preimage = tuple(
                sorted(inverse_sector_map[label] for label in initial_labels)
            )
            sector_ids_by_initial_preimage.setdefault(initial_preimage, []).append(
                int(sector_id)
            )
        if not complete_initial_maps:
            continue
        partition_sector_ids = sorted(
            sector_ids_by_initial_preimage.values(),
            key=lambda sector_ids: min(sector_ids),
        )
        for grouped_sector_ids in partition_sector_ids:
            representative = min(grouped_sector_ids)
            representative_map = base_maps[representative]
            inverse_representative_map = {
                sector_label: representative_label
                for representative_label, sector_label in representative_map.items()
            }
            active_sector_ids: list[int] = []
            relative_permutations: list[tuple[tuple[int, int], ...]] = []
            replay_weights: list[float] = []
            grouped_sector_id_set = set(grouped_sector_ids)
            for sector_id in group.sector_ids:
                if int(sector_id) not in grouped_sector_id_set:
                    continue
                sector = color_plan.sector(int(sector_id))
                if sector is None:
                    continue
                sector_map = base_maps[int(sector_id)]
                relative_map = {
                    representative_label: sector_map[
                        inverse_representative_map[representative_label]
                    ]
                    for representative_label in sorted(representative_map.values())
                }
                if {relative_map[label] for label in initial_labels} != initial_labels:
                    raise RuntimeError(
                        "internal LC replay partitioning error: sectors grouped "
                        "by initial-label preimage do not preserve the initial set"
                    )
                active_sector_ids.append(int(sector_id))
                relative_permutations.append(tuple(sorted(relative_map.items())))
                replay_weights.append(
                    _lc_topology_replay_sector_weight(color_plan, sector)
                )
            if not active_sector_ids:
                raise RuntimeError(
                    "internal LC replay partitioning error: representative sector "
                    f"{representative} did not produce a non-empty replay block"
                )
            partitions.append(
                LCColorSectorReplayPartition(
                    representative_sector_id=representative,
                    active_sector_ids=tuple(active_sector_ids),
                    label_permutations=tuple(relative_permutations),
                    replay_weights=tuple(replay_weights),
                )
            )
    return tuple(partitions)


def _lc_topology_replay_sector_weight(
    color_plan: GenericColorPlan,
    sector: LCColorSector,
) -> float:
    """Return the LC multiplicity represented by a materialized sector.

    A model-proven pure single-trace LC plan may fold trace reflections during
    colour-sector enumeration.  Replaying such a materialized representative
    with weight two preserves the full ordering sum.
    """

    if (
        color_plan.color_accuracy == "lc"
        and color_plan.trace_reflections_folded
        and sector.kind == "single-trace"
        and len(sector.trace_labels) > 2
    ):
        return 2.0
    return 1.0


def lc_line_pairing_representative_ids(
    color_plan: GenericColorPlan,
) -> tuple[int, ...]:
    """Return one sector per LC open-line pairing/allocation.

    This is a generic colour-flow pruning helper.  It keeps distinct
    fundamental-antifundamental pairings and distinct adjoint/singlet
    attachments, but drops
    duplicate sectors that only permute complete open-line blocks in the colour
    word.  It deliberately does not group sectors with different flavour
    pairings or different particles assigned to a line.
    """

    if color_plan.color_accuracy != "lc":
        return ()
    representatives: list[int] = []
    seen: set[tuple[object, ...]] = set()
    pdg_by_label = {
        leg.label: leg.outgoing_pdg
        for leg in color_plan.process.legs
        if leg.outgoing_pdg is not None
    }
    for sector in color_plan.sectors:
        signature = _sector_line_pairing_signature(pdg_by_label, sector)
        if signature in seen:
            continue
        seen.add(signature)
        representatives.append(int(sector.id))
    return tuple(representatives)


def _sector_line_pairing_signature(
    pdg_by_label: dict[int, int | None],
    sector: LCColorSector,
) -> tuple[object, ...]:
    if sector.kind == "open-lines":
        return (
            sector.kind,
            tuple(
                sorted(
                    (
                        line.fundamental_label,
                        pdg_by_label.get(line.fundamental_label),
                        line.antifundamental_label,
                        pdg_by_label.get(line.antifundamental_label),
                        tuple(
                            (label, pdg_by_label.get(label))
                            for label in line.adjoint_labels
                        ),
                        tuple(
                            (label, pdg_by_label.get(label))
                            for label in line.singlet_labels
                        ),
                    )
                    for line in sector.open_color_lines
                )
            ),
        )
    if sector.kind == "single-trace":
        canonical_trace = min(sector.trace_labels, tuple(reversed(sector.trace_labels)))
        return (
            sector.kind,
            tuple((label, pdg_by_label.get(label)) for label in canonical_trace),
            tuple((label, pdg_by_label.get(label)) for label in sector.singlet_labels),
        )
    return (
        sector.kind,
        tuple((label, pdg_by_label.get(label)) for label in sector.singlet_labels),
    )


def _lc_topology_group_replay_safe(
    color_plan: GenericColorPlan,
    group: LCColorSectorTopologyGroup,
) -> bool:
    sectors = tuple(
        sector
        for sector_id in group.sector_ids
        if (sector := color_plan.sector(sector_id)) is not None
    )
    if len(sectors) != len(group.sector_ids):
        return False
    if any(sector.kind != "open-lines" for sector in sectors):
        return False
    initial_labels = {leg.label for leg in color_plan.process.initial_legs}
    if not initial_labels:
        return False
    for permutation in group.label_permutations:
        mapping = {
            int(representative_label): int(sector_label)
            for representative_label, sector_label in permutation
        }
        if {mapping.get(label) for label in initial_labels} != initial_labels:
            return False
    return True
