# SPDX-License-Identifier: 0BSD
"""Frozen color-contraction records."""

from __future__ import annotations

from dataclasses import dataclass

NC = 3


@dataclass(frozen=True)
class ColorContractionEntry:
    left_group_id: int
    right_group_id: int
    weight_re: float
    weight_im: float = 0.0
    symmetry_factor: float = 1.0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "left_group_id": self.left_group_id,
            "right_group_id": self.right_group_id,
            "weight": [self.weight_re, self.weight_im],
            "symmetry_factor": self.symmetry_factor,
        }


@dataclass(frozen=True)
class ColorContractionPlan:
    color_accuracy: str
    supported: bool
    reason: str | None
    group_count: int
    entries: tuple[ColorContractionEntry, ...]
    includes_color_factor: bool = True

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": "pyamplicol-color-contraction-plan",
            "color_accuracy": self.color_accuracy,
            "supported": self.supported,
            "reason": self.reason,
            "group_count": self.group_count,
            "includes_color_factor": self.includes_color_factor,
            "entry_count": len(self.entries),
            "storage": "upper-triangular sparse metric over coherent amplitude groups",
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class ColorGroupDescriptor:
    group_id: int
    helicity_key: tuple[object, ...]
    sector_id: int
    word: tuple[int, ...]
    helicity_weight: float
