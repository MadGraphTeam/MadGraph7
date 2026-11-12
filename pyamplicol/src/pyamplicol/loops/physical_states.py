# SPDX-License-Identifier: 0BSD
"""Immutable external-state layout of a one-loop request (legs, helicities)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod


@dataclass(frozen=True, slots=True)
class ExternalStateLeg:
    index: int
    kind: str
    incoming: bool
    massless: bool
    offset: int

    @property
    def components(self):
        return 1 if self.kind == "scalar" else 4

    @property
    def helicities(self):
        if self.kind == "scalar":
            return (0,)
        if self.kind in {"epsilon", "epsilon_bar"} and not self.massless:
            return (-1, 0, 1)
        return (-1, 1)


@dataclass(frozen=True, slots=True)
class ExternalStateLayout:
    """Leg order, component offsets and physical helicities are fixed at build time.

    Masslessness is structural: an exact fixed zero may resolve a named mass,
    while an adjustable mass keeps its massive state space. A runtime update to
    zero does not silently change the helicity catalogue. Native longitudinal
    states at zero spatial momentum retain the dependency's explicit error.
    """

    legs: tuple[ExternalStateLeg, ...]

    @property
    def components(self):
        return sum(leg.components for leg in self.legs)

    @property
    def helicities(self):
        return tuple(product(*(leg.helicities for leg in self.legs)))

    @property
    def initial_spin_average(self):
        return prod(len(leg.helicities) for leg in self.legs if leg.incoming)
