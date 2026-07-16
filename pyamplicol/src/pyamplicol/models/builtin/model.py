# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from ..base import (
    CouplingOrders,
    Model,
    PropagatorLoweringRule,
    QuantumNumberFlow,
    Vertex,
)
from .definitions import BuiltinSMDefinitionMixin
from .expressions import _builtin_vertex_result_chiralities
from .lowering import BuiltinSMLoweringMixin

_ELECTRIC_CHARGE_BY_PARTICLE = {
    1: "-1/3",
    2: "2/3",
    3: "-1/3",
    4: "2/3",
    5: "-1/3",
    6: "2/3",
    11: "-1",
    13: "-1",
    15: "-1",
    24: "1",
    26: "1",
}


class BuiltinModel(Model):
    """Legacy built-in-model conventions isolated from external UFO models."""

    @cached_property
    def _color_rep_by_pdg(self) -> dict[int, int]:
        reps: dict[int, int] = {}
        for pdg, particle in self._species_by_pdg.items():
            color = particle.color_rep
            if self._property_sign_by_pdg[pdg] < 0 and abs(color) == 3:
                reps[pdg] = -color
            else:
                reps[pdg] = color
        return reps

    def color_rep(self, pdg: int) -> int:
        try:
            return self._color_rep_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc

    def is_quark(self, pdg: int) -> bool:
        return 1 <= pdg <= 6

    def is_antiquark(self, pdg: int) -> bool:
        return -6 <= pdg <= -1

    def is_lepton(self, pdg: int) -> bool:
        return 11 <= pdg <= 16

    def is_antilepton(self, pdg: int) -> bool:
        return -16 <= pdg <= -11

    def is_fermion(self, pdg: int) -> bool:
        return (
            self.is_quark(pdg)
            or self.is_antiquark(pdg)
            or self.is_lepton(pdg)
            or self.is_antilepton(pdg)
        )

    def is_chiral_eligible(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.mass(pdg) == 0.0

    def _vertex_result_chiralities(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[int, ...]:
        return _builtin_vertex_result_chiralities(
            self,
            vertex,
            left_index,
            right_index,
        )

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        return self.is_quark(pdg) or self.is_antiquark(pdg)

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        return pdg == 21

    def is_gluon(self, pdg: int) -> bool:
        return pdg in (21, 99)

    def is_singlet(self, pdg: int) -> bool:
        return not (abs(pdg) <= 6 or pdg == 21)

    def is_tensor(self, pdg: int) -> bool:
        return pdg in (-21, -23, 26, -26)

    def is_massive_boson(self, pdg: int) -> bool:
        return pdg == 23 or abs(pdg) == 24

    def is_photon(self, pdg: int) -> bool:
        return pdg == 22

    def is_higgs(self, pdg: int) -> bool:
        return pdg == 25

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        particle = self.particle(particle_id)
        expression = _ELECTRIC_CHARGE_BY_PARTICLE.get(particle.pdg, "0")
        if expression != "0" and self._property_sign(particle_id) < 0:
            expression = (
                expression.removeprefix("-")
                if expression.startswith("-")
                else f"-{expression}"
            )
        return (("electric_charge", expression),)

    def propagator_lowering_rule(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorLoweringRule:
        if self.is_tensor(particle_id):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="identity",
                full_tensor_network_ready=True,
                applies_propagator=False,
                kernel="auxiliary_tensor_embedded_propagator",
                description=(
                    "auxiliary-tensor propagator factors are embedded in the "
                    "adjacent built-in-SM vertex kernels"
                ),
            )
        if particle_id in (125, 126, 127):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="identity",
                full_tensor_network_ready=True,
                applies_propagator=False,
                kernel="auxiliary_scalar_no_propagator",
                description=(
                    "Higgsor auxiliary scalar currents are non-propagating in "
                    "the built-in-SM model"
                ),
            )
        if particle_id in (21, 22, 99):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="massless_vector_feynman_gauge",
                description="massless vector propagator in mostly-minus metric",
            )
        if abs(particle_id) == 24 or particle_id == 23:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="massive_vector_unitary_gauge",
                description="massive vector propagator with width",
            )
        if self.is_chiral_eligible(particle_id) and chirality != 0:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="spenso",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="weyl_fermion",
                description="massless Weyl fermion propagator",
            )
        if self.is_fermion(particle_id) and self.mass(particle_id) != 0.0:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="massive_dirac_fermion",
                description="massive Dirac fermion propagator",
            )
        if self.is_higgs(particle_id):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="scalar_with_width",
                description="scalar propagator with optional width",
            )
        return PropagatorLoweringRule(
            particle_id=particle_id,
            chirality=chirality,
            backend="unimplemented",
            full_tensor_network_ready=False,
            applies_propagator=True,
            kernel="unknown",
            description="no built-in-model propagator lowering is registered",
        )

    def auxiliary_kind(self, particle_id: int) -> str | None:
        if particle_id == 99:
            return "u1-subtraction-color-flow-vector"
        if self.is_tensor(particle_id):
            return "antisymmetric-tensor"
        return None


@dataclass
class BuiltinSMModel(BuiltinSMLoweringMixin, BuiltinSMDefinitionMixin, BuiltinModel):
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

    def runtime_normalization_payload(self, dag: Any) -> dict[str, object]:
        """Preserve the established built-in-SM normalization convention."""

        initial = tuple(int(pdg) for pdg in dag.process.initial_pdgs)
        final = tuple(int(pdg) for pdg in dag.process.final_pdgs)
        average_factor = 1
        for pdg in initial:
            if pdg == 21:
                average_factor *= 16
            elif 1 <= abs(pdg) <= 6:
                average_factor *= 6
            else:
                average_factor *= 2
        identical_factor = math.prod(
            math.factorial(count) for count in Counter(final).values()
        )
        electroweak_power = (
            max(1, len(dag.process.singlet_labels)) if dag.process.singlet_labels else 0
        )
        qcd_power = max(0, len(dag.process.legs) - 2 - electroweak_power)
        return {
            "color_accuracy": dag.process.color_accuracy,
            "color_factor": int(self.leading_color_factor((*initial, *final))),
            "average_factor": average_factor,
            "identical_factor": identical_factor,
            "final_state_identical_factor": identical_factor,
            "quark_line_partner_factor": 1,
            "global_coupling_factor": (
                (4.0 * math.pi * self.alpha_s_me_check) ** qcd_power
                * (2.0 * 4.0 * math.pi * self.alpha_ew) ** electroweak_power
            ),
            "qcd_coupling_power": qcd_power,
            "electroweak_coupling_power": electroweak_power,
            "couplings_in_stage_evaluators": True,
            "coupling_policy": "stage evaluators include local vertex couplings",
        }

    def skip_duplicate_vertex_orientation(self, vertex: Vertex) -> bool:
        """Skip mirrored model-table entries already covered by DAG sweeps."""

        return False

    def vertex_coupling_orders(self, vertex: Vertex) -> CouplingOrders:
        """Classify built-in SM vertices by UFO-style coupling order."""

        if vertex.kind in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
            return (("QCD", 1),)
        return (("QED", 1),)

    def global_helicity_flip_equivalence_is_proven(
        self,
        vertices: Iterable[Vertex],
    ) -> bool:
        """Use the pinned built-in-SM QCD kernel inventory as the proof source."""

        for vertex in vertices:
            orders = self.vertex_coupling_orders(vertex)
            if not orders and self.vertex_is_internal_contact_fragment(vertex):
                continue
            if not orders or any(name != "QCD" for name, _value in orders):
                return False
        return True

    def pure_massless_adjoint_helicity_zero_rule_is_proven(
        self,
        process: Any,
        vertices: Iterable[Vertex],
    ) -> bool:
        """Prove the tree-level Yang--Mills helicity-zero rule.

        The proof is deliberately tied to the built-in model's registered
        massless-adjoint and auxiliary-tensor kernels. A QCD coupling-order
        label alone would also admit quark vertices and is not sufficient.
        """

        legs = tuple(getattr(process, "legs", ()))
        if not legs or any(
            getattr(leg, "outgoing_pdg", None) is None
            or not self.is_massless_adjoint_vector(int(leg.outgoing_pdg))
            or self.mass(int(leg.outgoing_pdg)) != 0.0
            for leg in legs
        ):
            return False

        pure_gauge_inventory = {
            vertex
            for vertex in self.vertices
            if all(
                self.is_massless_adjoint_vector(particle_id)
                or self.auxiliary_kind(particle_id) == "antisymmetric-tensor"
                for particle_id in vertex.particles
            )
        }
        used_vertices = tuple(vertices)
        return bool(used_vertices) and all(
            vertex in pure_gauge_inventory for vertex in used_vertices
        )

    def adjoint_current_reflection_phase(
        self,
        vertex: Vertex,
    ) -> tuple[float, float] | None:
        """Expose the pinned three-gauge-boson current antisymmetry."""

        if vertex.kind != 0 or not all(
            self.is_massless_adjoint_vector(particle_id)
            for particle_id in vertex.particles
        ):
            return None
        return (-1.0, 0.0)

    def lc_trace_reflection_equivalence_is_proven(self, process: Any) -> bool:
        """Prove reflection folding for the pinned pure-gluon implementation."""

        legs = tuple(getattr(process, "legs", ()))
        return bool(legs) and all(
            getattr(leg, "outgoing_pdg", None) is not None
            and self.is_massless_adjoint_vector(int(leg.outgoing_pdg))
            for leg in legs
        )

    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {"QCD": 1, "QED": 2}


__all__ = ["BuiltinModel", "BuiltinSMModel"]
