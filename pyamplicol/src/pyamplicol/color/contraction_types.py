# SPDX-License-Identifier: 0BSD
"""Frozen color-contraction records."""

from __future__ import annotations

from collections.abc import Iterator
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
class ColorContractionTemplateEntry:
    left_group_index: int
    right_group_index: int
    weight_re: float
    weight_im: float = 0.0
    symmetry_factor: float = 1.0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "left_group_index": self.left_group_index,
            "right_group_index": self.right_group_index,
            "weight": [self.weight_re, self.weight_im],
            "symmetry_factor": self.symmetry_factor,
        }


@dataclass(frozen=True)
class RepeatedColorContractionBlock:
    component_count: int
    component_group_ids: tuple[int, ...]
    entries: tuple[ColorContractionTemplateEntry, ...]

    def __post_init__(self) -> None:
        if self.component_count < 2:
            raise ValueError(
                "repeated color contraction requires at least two components"
            )
        if not self.component_group_ids:
            raise ValueError("repeated color contraction group map is empty")
        if len(self.component_group_ids) % self.component_count != 0:
            raise ValueError(
                "repeated color contraction group IDs do not form a rectangular map"
            )
        if len(set(self.component_group_ids)) != len(self.component_group_ids):
            raise ValueError("repeated color contraction group IDs are not unique")
        local_group_count = self.local_group_count
        if any(
            entry.left_group_index < 0
            or entry.left_group_index >= local_group_count
            or entry.right_group_index < 0
            or entry.right_group_index >= local_group_count
            for entry in self.entries
        ):
            raise ValueError(
                "repeated color contraction entry references an unknown local group"
            )

    @property
    def local_group_count(self) -> int:
        return len(self.component_group_ids) // self.component_count

    def to_json_dict(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "component_group_ids": list(self.component_group_ids),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class ColorContractionPlan:
    color_accuracy: str
    supported: bool
    reason: str | None
    group_count: int
    entries: tuple[ColorContractionEntry, ...]
    repeated_block: RepeatedColorContractionBlock | None = None
    includes_color_factor: bool = True

    def __post_init__(self) -> None:
        if self.entries and self.repeated_block is not None:
            raise ValueError(
                "color contraction cannot mix expanded and repeated entries"
            )
        if (
            self.repeated_block is not None
            and len(self.repeated_block.component_group_ids) != self.group_count
        ):
            raise ValueError(
                "repeated color contraction group map does not match group count"
            )

    @property
    def logical_entry_count(self) -> int:
        if self.repeated_block is None:
            return len(self.entries)
        return self.repeated_block.component_count * len(self.repeated_block.entries)

    def iter_logical_entries(self) -> Iterator[ColorContractionEntry]:
        if self.repeated_block is None:
            yield from self.entries
            return
        block = self.repeated_block
        for component_index in range(block.component_count):
            for entry in block.entries:
                yield ColorContractionEntry(
                    left_group_id=block.component_group_ids[
                        entry.left_group_index * block.component_count + component_index
                    ],
                    right_group_id=block.component_group_ids[
                        entry.right_group_index * block.component_count
                        + component_index
                    ],
                    weight_re=entry.weight_re,
                    weight_im=entry.weight_im,
                    symmetry_factor=entry.symmetry_factor,
                )

    def to_json_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "pyamplicol-color-contraction-plan",
            "color_accuracy": self.color_accuracy,
            "supported": self.supported,
            "reason": self.reason,
            "group_count": self.group_count,
            "includes_color_factor": self.includes_color_factor,
            "entry_count": len(self.entries),
            "logical_entry_count": self.logical_entry_count,
            "storage": "upper-triangular sparse metric over coherent amplitude groups",
            "entries": [entry.to_json_dict() for entry in self.entries],
        }
        if self.repeated_block is not None:
            result["repeated_block"] = self.repeated_block.to_json_dict()
        return result


@dataclass(frozen=True)
class ColorGroupDescriptor:
    group_id: int
    helicity_key: tuple[object, ...]
    sector_id: int
    word: tuple[int, ...]
    helicity_weight: float
