# SPDX-License-Identifier: 0BSD
"""Structural order axes and precise results for internal bare assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .orders import LoopOrderPlan, VirtualOrderSector, canonical_order, squared_order

if TYPE_CHECKING:
    from pyamplicol.models.base import CouplingOrders


@dataclass(frozen=True, slots=True)
class BornSquaredSector:
    order: CouplingOrders
    pairs: tuple[tuple[CouplingOrders, CouplingOrders], ...]


@dataclass(frozen=True, slots=True)
class PhysicalOrderLayout:
    born: tuple[BornSquaredSector, ...]
    virtual: tuple[VirtualOrderSector, ...]
    correction_names: tuple[str, ...]

    @classmethod
    def from_plan(cls, plan: LoopOrderPlan):
        pairs = {}
        for left in plan.born_orders:
            for right in plan.born_orders:
                pairs.setdefault(squared_order(left, right), []).append((left, right))
        return cls(
            tuple(
                BornSquaredSector(order, tuple(pairs[order])) for order in sorted(pairs)
            ),
            plan.sectors,
            plan.correction_names,
        )

    @property
    def is_single(self):
        return len(self.born) == len(self.virtual) == 1


@dataclass(frozen=True, slots=True)
class SummedBareValues:
    """Explicit native sum over order axes, preserving decimal complex pairs."""

    born: Any
    virtual: Any
    rational: Any = None


@dataclass(frozen=True, slots=True)
class OrderedBareProcessResult:
    """Batch first, then squared order, then epsilon and decimal complex pair.

    Status columns correspond to ``channels`` entries (virtual sector, family).
    Structurally generated zero sectors remain on both order axes. Sums are
    computed in the selected native carrier before decimal transport.
    """

    born: Any
    virtual: Any
    orders: PhysicalOrderLayout
    channels: tuple[tuple[int, int], ...]
    statuses: Any
    arithmetic: str
    precision: int | None
    reduction_engine: str
    coefficient_engine: str
    _summed: SummedBareValues = field(repr=False)
    rational: Any = None
    epsilon_powers: tuple[int, ...] = (0, -1, -2)
    convention: str = "MadLoop-Cepsilon-bare-four-dimensional-no-R2-no-UV"

    def __post_init__(self):
        for values in (
            self.born,
            self.virtual,
            self.rational,
            self._summed.born,
            self._summed.virtual,
            self._summed.rational,
        ):
            if values is not None:
                values.setflags(write=False)

    @property
    def born_orders(self):
        return tuple(sector.order for sector in self.orders.born)

    @property
    def virtual_orders(self):
        return tuple(sector.order for sector in self.orders.virtual)

    @property
    def finite(self):
        return self.virtual[:, :, 0]

    @property
    def single_pole(self):
        return self.virtual[:, :, 1]

    @property
    def double_pole(self):
        return self.virtual[:, :, 2]

    def born_order(self, order):
        wanted = canonical_order(order)
        try:
            index = self.born_orders.index(wanted)
        except ValueError as error:
            raise ValueError(f"Born squared order {wanted!r} is absent") from error
        return self.born[:, index]

    def virtual_order(self, order):
        wanted = canonical_order(order)
        try:
            index = self.virtual_orders.index(wanted)
        except ValueError as error:
            raise ValueError(f"virtual squared order {wanted!r} is absent") from error
        return self.virtual[:, index]

    def sum_orders(self):
        return self._summed
