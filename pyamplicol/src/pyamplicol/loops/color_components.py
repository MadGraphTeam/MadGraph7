# SPDX-License-Identifier: 0BSD
"""Signed Born anchors from dependency-canonical external colour tensors.

This is a partition of coherent Born columns, not a minimal colour basis or a
set of positive tree flows. Canonicalization and colour algebra belong to the
tensor dependency; the partition only schedules already projected bra rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from .exact import require_exact_numbers


@dataclass(frozen=True, slots=True)
class BornColorComponents:
    ids: tuple[str, ...]
    members: tuple[tuple[int, ...], ...]
    kind: ClassVar[str] = "canonical-born-tensor-partition"

    def __post_init__(self):
        object.__setattr__(self, "ids", tuple(self.ids))
        object.__setattr__(self, "members", tuple(tuple(row) for row in self.members))
        if (
            not self.ids
            or len(self.ids) != len(self.members)
            or any(not isinstance(value, str) or not value for value in self.ids)
            or len(set(self.ids)) != len(self.ids)
            or any(not row for row in self.members)
            or any(
                type(index) is not int or index < 0
                for row in self.members
                for index in row
            )
        ):
            raise ValueError(
                "Born colour components require distinct IDs and nonempty index groups"
            )
        self.require_width(sum(map(len, self.members)))

    def require_width(self, width: int) -> None:
        indices = [index for row in self.members for index in row]
        if sorted(indices) != list(range(width)):
            raise ValueError(
                "Born colour components must partition every Born column exactly once"
            )

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ids": self.ids, "members": self.members}

    def mask_bra(
        self, metric: Sequence[Sequence[Any]]
    ) -> tuple[tuple[tuple[Any, ...], ...], ...]:
        """Partition an already projected metric without changing its degree.

        The sum of these rectangular matrices is exactly the supplied matrix.
        A Born component uses Re(B_anchor† G B), with no factor of two; an
        ordinary virtual component uses 2 Re only after loop integration.
        """
        from symbolica import E

        self.require_width(len(metric))
        rows = tuple(tuple(row) for row in metric)
        if len({len(row) for row in rows}) != 1:
            raise ValueError("colour metric must be rectangular")
        zero = E("0")
        return tuple(
            tuple(
                row if index in members else (zero,) * len(row)
                for index, row in enumerate(rows)
            )
            for members in map(frozenset, self.members)
        )


def prepare_born_color_components(tensors: Sequence[Any]) -> BornColorComponents:
    """Group equal native canonical tensors with fixed physical external labels.

    Equivalent dummy-index spellings share an anchor. The dependency does not
    claim that canonize solves general linear identities, so the resulting
    components can be overcomplete. Numeric equality at Nc=3 is never used.
    """
    from symbolica.community.tensor import TensorExpression

    if not tensors:
        raise ValueError("Born colour components require a Born amplitude")
    groups: dict[Any, list[int]] = {}
    axes = None
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, TensorExpression):
            raise TypeError("Born colour anchors require dependency tensors")
        if axes is None:
            axes = set(tensor.axes)
        elif set(tensor.axes) != axes:
            raise ValueError(
                "Born colour tensors have different physical external ports"
            )
        canonical = tensor.canonize().to_expression()
        require_exact_numbers(canonical)
        groups.setdefault(canonical, []).append(index)
    # Symbolica's canonical expressions retain their exact tensor names and
    # external labels. Ordering their printed normal forms avoids graph-order
    # dependence without introducing a separate canonicalization algorithm.
    ordered = sorted(groups, key=str)
    return BornColorComponents(
        tuple(f"born-{index}" for index in range(len(ordered))),
        tuple(tuple(groups[tensor]) for tensor in ordered),
    )
