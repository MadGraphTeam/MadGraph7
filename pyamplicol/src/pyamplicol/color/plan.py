# SPDX-License-Identifier: 0BSD
"""Compatibility facade for color-sector planning."""

from __future__ import annotations

from .plan_build import build_color_plan
from .plan_replay import (
    lc_line_pairing_representative_ids,
    lc_topology_replay_partitions,
    lc_topology_replay_safe_groups,
)
from .plan_types import (
    ColorAccuracy,
    ColorSectorKind,
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorSectorTopologyGroup,
    LCQuarkLine,
)

__all__ = [
    "ColorAccuracy",
    "ColorSectorKind",
    "GenericColorPlan",
    "LCColorSector",
    "LCColorSectorReplayPartition",
    "LCColorSectorTopologyGroup",
    "LCQuarkLine",
    "build_color_plan",
    "lc_line_pairing_representative_ids",
    "lc_topology_replay_partitions",
    "lc_topology_replay_safe_groups",
]
