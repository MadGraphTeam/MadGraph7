# SPDX-License-Identifier: 0BSD
"""Structural order axes and precise results for internal virtual assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyamplicol.models.base import CouplingOrders


@dataclass(frozen=True, slots=True)
class BornSquaredSector:
    order: CouplingOrders
    pairs: tuple[tuple[CouplingOrders, CouplingOrders], ...]
