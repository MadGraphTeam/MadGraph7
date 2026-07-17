# SPDX-License-Identifier: 0BSD
"""Frozen color-sector and color-plan records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import permutations
from typing import Literal

from ..processes.ir import CanonicalProcessIR

ColorAccuracy = Literal["lc", "nlc", "full"]
ColorSectorKind = Literal["singlet", "open-lines", "single-trace"]


@dataclass(frozen=True)
class LCOpenColorLine:
    """One leading-colour open line in all-outgoing conventions."""

    fundamental_label: int
    antifundamental_label: int
    adjoint_labels: tuple[int, ...]
    singlet_labels: tuple[int, ...] = ()

    @property
    def coloured_labels(self) -> tuple[int, ...]:
        return (
            self.fundamental_label,
            *self.adjoint_labels,
            self.antifundamental_label,
        )

    @property
    def line_labels(self) -> tuple[int, ...]:
        return (*self.coloured_labels, *self.singlet_labels)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fundamental_label": self.fundamental_label,
            "antifundamental_label": self.antifundamental_label,
            "adjoint_labels": list(self.adjoint_labels),
            "singlet_labels": list(self.singlet_labels),
            "line_labels": list(self.line_labels),
        }


@dataclass(frozen=True)
class LCColorSector:
    """One colour-flow sector produced during generation warmup."""

    id: int
    kind: ColorSectorKind
    open_color_lines: tuple[LCOpenColorLine, ...] = ()
    trace_labels: tuple[int, ...] = ()
    singlet_labels: tuple[int, ...] = ()
    word_labels: tuple[int, ...] = ()

    @property
    def coloured_label_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.kind == "open-lines":
            return tuple(line.coloured_labels for line in self.open_color_lines)
        if self.kind == "single-trace":
            return (self.trace_labels,)
        return ()

    @property
    def line_label_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.kind == "open-lines":
            return tuple(line.coloured_labels for line in self.open_color_lines)
        if self.kind == "single-trace":
            return ((*self.trace_labels, *self.singlet_labels),)
        if self.kind == "singlet":
            return (self.singlet_labels,)
        return ()

    @cached_property
    def color_words(self) -> tuple[tuple[int, ...], ...]:
        if self.word_labels:
            return (self.word_labels,)
        if self.kind == "open-lines":
            return (
                tuple(
                    label
                    for line in self.open_color_lines
                    for label in line.coloured_labels
                ),
            )
        if self.kind == "single-trace":
            return (self.trace_labels,)
        if self.kind == "singlet":
            return ((),)
        return ()

    @cached_property
    def admissible_traversal_words(self) -> tuple[tuple[int, ...], ...]:
        """Colour words accepted while constructing currents for this sector.

        The sector itself has one physical colour word.  During current
        construction, however, complete open-line blocks can be traversed
        in different intermediate orders.  This lets the generic recursion
        reproduce AmpliCol's colour-ordered current closures without naming a
        process family.
        """

        from .plan_build import _ordered_open_line_blocks

        if self.kind != "open-lines" or len(self.open_color_lines) < 2:
            return self.color_words
        primary = self.color_words[0]
        blocks = _ordered_open_line_blocks(primary, self.open_color_lines)
        if blocks is None or len(blocks) < 2:
            return self.color_words
        words: list[tuple[int, ...]] = [primary]
        seen = {primary}
        for block_permutation in permutations(blocks):
            word = tuple(label for block in block_permutation for label in block)
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
        return tuple(words)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "open_color_lines": [
                line.to_json_dict() for line in self.open_color_lines
            ],
            "trace_labels": list(self.trace_labels),
            "singlet_labels": list(self.singlet_labels),
            "word_labels": list(self.word_labels),
            "coloured_label_groups": [
                list(group) for group in self.coloured_label_groups
            ],
            "line_label_groups": [list(group) for group in self.line_label_groups],
            "color_words": [list(word) for word in self.color_words],
            "admissible_traversal_words": [
                list(word) for word in self.admissible_traversal_words
            ],
        }


@dataclass(frozen=True)
class LCColorSectorTopologyGroup:
    """Colour sectors that can share one compiled current topology.

    The group key is built from model/process data carried by the external
    labels, not from a process-family name.  Runtime reuse still evaluates each
    sector with its own external-label permutation, so this is a generation-time
    sharing plan rather than a physics approximation.
    """

    signature: tuple[object, ...]
    representative_sector_id: int
    sector_ids: tuple[int, ...]
    label_permutations: tuple[tuple[tuple[int, int], ...], ...]

    def to_json_dict(self) -> dict[str, object]:
        from .plan_replay import _jsonable_signature

        return {
            "signature": _jsonable_signature(self.signature),
            "representative_sector_id": self.representative_sector_id,
            "sector_ids": list(self.sector_ids),
            "label_permutations": [
                [[left, right] for left, right in permutation]
                for permutation in self.label_permutations
            ],
        }


@dataclass(frozen=True)
class LCColorSectorReplayPartition:
    """Initial-label-safe replay block inside one LC topology group."""

    representative_sector_id: int
    active_sector_ids: tuple[int, ...]
    label_permutations: tuple[tuple[tuple[int, int], ...], ...]
    replay_weights: tuple[float, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "representative_sector_id": self.representative_sector_id,
            "active_sector_ids": list(self.active_sector_ids),
            "label_permutations": [
                [[left, right] for left, right in permutation]
                for permutation in self.label_permutations
            ],
            "replay_weights": list(self.replay_weights),
        }


@dataclass(frozen=True)
class GenericColorPlan:
    """Colour-flow planning payload shared by future Python/Rust runtimes."""

    process: CanonicalProcessIR
    color_accuracy: str
    sectors: tuple[LCColorSector, ...]
    diagnostics: tuple[str, ...] = ()
    truncated: bool = False
    trace_reflections_folded: bool = False

    @property
    def sector_count(self) -> int:
        return len(self.sectors)

    @property
    def ready_for_leading_colour(self) -> bool:
        return self.color_accuracy == "lc" and bool(self.sectors) and not self.truncated

    @property
    def ready_for_requested_colour(self) -> bool:
        return bool(self.sectors) and not self.truncated

    @property
    def coloured_labels(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    *self.process.fundamental_labels,
                    *self.process.antifundamental_labels,
                    *self.process.adjoint_labels,
                }
            )
        )

    @cached_property
    def _sectors_by_id(self) -> dict[int, LCColorSector]:
        return {int(sector.id): sector for sector in self.sectors}

    def sector(self, color_sector: int) -> LCColorSector | None:
        return self._sectors_by_id.get(int(color_sector))

    @cached_property
    def topology_groups(self) -> tuple[LCColorSectorTopologyGroup, ...]:
        from .plan_replay import _sector_topology_groups

        return _sector_topology_groups(self.process, self.sectors)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process.to_json_dict(),
            "color_accuracy": self.color_accuracy,
            "sector_count": self.sector_count,
            "truncated": self.truncated,
            "trace_reflections_folded": self.trace_reflections_folded,
            "ready_for_leading_colour": self.ready_for_leading_colour,
            "ready_for_requested_colour": self.ready_for_requested_colour,
            "coloured_labels": list(self.coloured_labels),
            "diagnostics": list(self.diagnostics),
            "sectors": [sector.to_json_dict() for sector in self.sectors],
            "topology_groups": [group.to_json_dict() for group in self.topology_groups],
        }
