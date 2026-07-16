# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    Particle,
    QuantumFlow,
    SourceSpinState,
    Vertex,
    VertexEvaluationEquivalence,
    VertexLoweringRule,
)
from .tensors import (
    classify_trilinear_color_expression,
)

if TYPE_CHECKING:
    pass

from .external_helpers import _spin_dimension


class ExternalModelCatalogMixin:
    def particle(self, pdg: int) -> Particle:
        try:
            return self.particles[int(pdg)]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc

    def vertices_for_inputs(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        if color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError(f"unknown colour accuracy: {color_accuracy}")
        return self._compiled_vertices_by_input.get(
            (int(left_pdg), int(right_pdg)),
            (),
        )

    def anti_particle(self, pdg: int) -> int:
        return self.particle(pdg).anti_pdg

    def mass(self, pdg: int) -> Any:
        return self._real_parameter_value(
            self._particle_records_by_pdg[int(pdg)].mass,
            field="mass",
        )

    def width(self, pdg: int) -> Any:
        return self._real_parameter_value(
            self._particle_records_by_pdg[int(pdg)].width,
            field="width",
        )

    def spin(self, pdg: int) -> int:
        return self._particle_records_by_pdg[int(pdg)].spin

    def dimension(self, pdg: int) -> int:
        record = self._particle_records_by_pdg[int(pdg)]
        if record.component_dimension is not None:
            return record.component_dimension
        return _spin_dimension(record.spin)

    def current_dimension(self, particle_id: int, chirality: int = 0) -> int:
        if chirality != 0 and self.is_chiral_eligible(particle_id):
            return 2
        return self.dimension(particle_id)

    def color_rep(self, pdg: int) -> int:
        return self._particle_records_by_pdg[int(pdg)].color

    def color_dim(self, pdg: int) -> int:
        return abs(self.color_rep(pdg))

    def charge(self, pdg: int) -> float:
        return self._particle_records_by_pdg[int(pdg)].charge

    def is_fermion(self, pdg: int) -> bool:
        return self.spin(pdg) == 2

    def is_quark(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.color_rep(pdg) == 3

    def is_antiquark(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.color_rep(pdg) == -3

    def is_lepton(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.color_rep(pdg) == 1 and pdg > 0

    def is_antilepton(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.color_rep(pdg) == 1 and pdg < 0

    def is_chiral_eligible(self, pdg: int) -> bool:
        if not self.is_fermion(pdg):
            return False
        propagator = self._propagator_record(pdg)
        if propagator is not None and propagator.custom:
            return False
        particle = self._particle_records_by_pdg[int(pdg)]
        if particle.mass.upper() == "ZERO":
            return True
        record = self._parameter_records.get(particle.mass)
        return (
            record is not None
            and record.nature != "external"
            and self._parameter_default(particle.mass) == 0.0
        )

    def is_gluon(self, pdg: int) -> bool:
        return self.spin(pdg) == 3 and self.color_rep(pdg) == 8

    def is_singlet(self, pdg: int) -> bool:
        return self.color_rep(pdg) == 1

    def is_tensor(self, pdg: int) -> bool:
        return self.spin(pdg) == 5

    def is_massive_boson(self, pdg: int) -> bool:
        return (
            self.spin(pdg) in {3, 5}
            and self._parameter_default(self._particle_records_by_pdg[int(pdg)].mass)
            != 0.0
        )

    def is_photon(self, pdg: int) -> bool:
        return self.spin(pdg) == 3 and self.color_rep(pdg) == 1 and self.mass(pdg) == 0

    def is_higgs(self, pdg: int) -> bool:
        return self.spin(pdg) == 1

    def source_spin_states(self, particle_id: int) -> tuple[SourceSpinState, ...]:
        if self.is_chiral_eligible(particle_id):
            return super().source_spin_states(particle_id)
        spin = self.spin(particle_id)
        massive = (
            complex(
                self._parameter_default(
                    self._particle_records_by_pdg[int(particle_id)].mass
                )
            ).real
            != 0.0
        )
        if spin == 1:
            helicities = (0,)
        elif spin == 2:
            helicities = (-1, 1)
        elif spin == 3:
            helicities = (-1, 0, 1) if massive else (-1, 1)
        elif spin == 5:
            helicities = (-2, -1, 0, 1, 2) if massive else (-2, 2)
        else:
            raise ValueError(
                f"unsupported source spin {spin} for particle {particle_id}"
            )
        return tuple(
            SourceSpinState(helicity=helicity, chirality=0, spin_state=helicity)
            for helicity in helicities
        )

    def allowed_quantum_flows(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[QuantumFlow, ...]:
        result_particle = vertex.particles[2]
        left_chirality = int(getattr(left_index, "chirality", 0))
        right_chirality = int(getattr(right_index, "chirality", 0))
        result_chiralities = (
            (-1, 1) if self.is_chiral_eligible(result_particle) else (0,)
        )
        return tuple(
            QuantumFlow(
                chirality=result_chirality,
                spin_state=(
                    result_chirality if self.is_chiral_eligible(result_particle) else 0
                ),
                flavour_flow=self.combine_flavour_flow(
                    result_particle,
                    left_index,
                    right_index,
                ),
                charge_flow=self.charge_units(result_particle),
                coupling=(1.0, 0.0),
            )
            for result_chirality in result_chiralities
            if self._weyl_projection_is_nonzero(
                vertex.kind,
                left_chirality,
                right_chirality,
                result_chirality,
            )
        )

    def vertex_lowering_rule(self, kind: int) -> VertexLoweringRule:
        kernel = self._kernel(kind)
        return VertexLoweringRule(
            kind=kind,
            backend="spenso-ufo",
            tensor_names=(kernel.vertex,),
            expression_head="compiled_ufo_kernel",
            full_tensor_network_ready=True,
            description="typed and oriented UFO tensor kernel",
            kernel="compiled_ufo_kernel",
            input_roles=(kernel.particles[0], kernel.particles[1]),
            output_role=kernel.particles[2],
            coupling_mode="external-model-parameters",
        )

    def vertex_evaluation_equivalence(
        self,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        kernel = self._kernel(kind)
        if not kernel.evaluation_equivalence_verified or not kernel.evaluation_class:
            return super().vertex_evaluation_equivalence(kind)
        input_order = tuple(int(value) for value in kernel.evaluation_input_order)
        if input_order not in {(0, 1), (1, 0)}:
            raise ValueError(
                f"compiled UFO kernel {kind} has invalid evaluation input order "
                f"{input_order}"
            )
        return VertexEvaluationEquivalence(
            class_id=kernel.evaluation_class,
            factor=kernel.evaluation_factor,
            input_order=input_order,
            verified=True,
        )

    def vertex_coupling_orders(self, vertex: Vertex):
        return self._kernel(vertex.kind).coupling_orders

    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {
            str(order.name).upper(): max(1, int(order.hierarchy))
            for order in self.compiled.ir.orders
        }

    def vertex_color_weight(
        self,
        vertex: Vertex,
        *,
        color_accuracy: str,
    ) -> tuple[float, float]:
        if color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError(f"unknown colour accuracy: {color_accuracy}")
        power = self._kernel(vertex.kind).lc_color_normalization_power
        normalization = 2.0 ** (-0.5 * power)
        structure, coefficient = self._vertex_color_projection(vertex)
        if structure in {
            "adjoint-structure-constant",
            "adjoint-structure-constant-product",
        }:
            phase = (-1j) ** power
            weight = coefficient * normalization * phase
            return (weight.real, weight.imag)
        weight = coefficient * normalization
        return (weight.real, weight.imag)

    def vertex_color_structure(self, vertex: Vertex) -> str:
        return self._vertex_color_projection(vertex)[0]

    def _vertex_color_projection(self, vertex: Vertex) -> tuple[str, complex]:
        cached = self._color_projection_cache.get(vertex.kind)
        if cached is not None:
            return cached
        kernel = self._kernel(vertex.kind)
        if (
            kernel.color_projection_structure is not None
            and kernel.color_projection_coefficient is not None
        ):
            projected = (
                kernel.color_projection_structure,
                complex(*kernel.color_projection_coefficient),
            )
        else:
            projected = classify_trilinear_color_expression(
                kernel.color_expression,
                kernel.color_source,
                tuple(self.color_rep(particle_id) for particle_id in vertex.particles),
            )
        self._color_projection_cache[vertex.kind] = projected
        return projected

    def vertex_is_internal_contact_fragment(self, vertex: Vertex) -> bool:
        return "::contact-" in self._kernel(vertex.kind).vertex

    def vertex_closure_allowed(self, vertex: Vertex) -> bool:
        del vertex
        return False
