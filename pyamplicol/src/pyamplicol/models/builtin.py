# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from .base import (
    CouplingOrders,
    Model,
    Vertex,
)
from .builtin_definitions import BuiltinSMDefinitionMixin
from .builtin_lowering import BuiltinSMLoweringMixin


@dataclass
class BuiltinSMModel(BuiltinSMLoweringMixin, BuiltinSMDefinitionMixin, Model):
    """Built-in Standard Model production path with pinned reference conventions."""

    name: str = "built-in-sm-leading-color"
    alpha_s_mz: float = 0.119
    alpha_s_me_check: float = 0.118
    alpha_ew: float = 0.007546771114
    sin_weak: float = 0.47143025548407230
    sqrt_s: float = 14000.0

    def __post_init__(self) -> None:
        self.particles = {
            particle.pdg: particle for particle in self._build_particles()
        }
        self.vertices = tuple(self._build_vertices())

    @cached_property
    def cos_weak(self) -> float:
        return math.sqrt(1.0 - self.sin_weak**2)

    def weak_coupling(self) -> float:
        return 1.0 / self.sin_weak

    def neutral_gauge_coupling(self) -> float:
        return self.weak_coupling() * self.cos_weak

    def charged_current_coupling(self) -> float:
        return self.weak_coupling() / math.sqrt(2.0)

    def weak_coupling_over_cosine(self) -> float:
        return self.weak_coupling() / self.cos_weak

    def photon_fermion_coupling(self, pdg: int) -> tuple[float, float]:
        particle = self.particle(pdg)
        return particle.charge, particle.charge

    def z_fermion_coupling(self, pdg: int) -> tuple[float, float]:
        particle = self.particle(pdg)
        charge = particle.charge
        left = particle.weak_isospin[0]
        right = particle.weak_isospin[1]
        prefactor = self.weak_coupling_over_cosine()
        return (
            prefactor * (left - charge * self.sin_weak**2),
            prefactor * (right - charge * self.sin_weak**2),
        )

    def leading_color_factor(self, process: Iterable[int]) -> int:
        exponent_twice = 0
        for pdg in process:
            if pdg == 21:
                exponent_twice += 2
            elif 1 <= abs(pdg) <= 6:
                exponent_twice += 1
        if exponent_twice % 2:
            raise ValueError(f"non-integer leading-color exponent for {tuple(process)}")
        return 3 ** (exponent_twice // 2)

    def skip_duplicate_vertex_orientation(self, vertex: Vertex) -> bool:
        """Skip mirrored model-table entries already covered by DAG sweeps."""

        return False

    def vertex_coupling_orders(self, vertex: Vertex) -> CouplingOrders:
        """Classify built-in SM vertices by UFO-style coupling order."""

        if vertex.kind in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
            return (("QCD", 1),)
        return (("QED", 1),)

    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {"QCD": 1, "QED": 2}


__all__ = ["BuiltinSMModel"]
