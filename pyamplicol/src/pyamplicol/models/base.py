# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

from ._physics_ir import (
    ContractionIR,
    CrossingIR,
    GoldstonePolicy,
    ParticleIdentityIR,
    ParticleOrientation,
    ParticleStatistics,
    PropagatorGauge,
    PropagatorIR,
    PropagatorKind,
    PropagatorMassClass,
    SourceIR,
    SourceStateIR,
    WavefunctionFamily,
)


@dataclass(frozen=True)
class Particle:
    pdg: int
    anti_pdg: int
    spin: int
    dimension: int
    color_rep: int
    mass: float = 0.0
    width: float = 0.0
    charge: float = 0.0
    weak_isospin: tuple[float, float] = (0.0, 0.0)
    weak_hypercharge: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Vertex:
    kind: int
    particles: tuple[int, int, int]
    coupling: tuple[float, float] = (1.0, 0.0)


CouplingOrders = tuple[tuple[str, int], ...]
QuantumNumberFlow = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SourceSpinState:
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]


@dataclass(frozen=True)
class QuantumFlow:
    chirality: int
    spin_state: int | tuple[int, ...]
    flavour_flow: tuple[int, ...]
    quantum_number_flow: QuantumNumberFlow
    coupling: tuple[float, float]


@dataclass(frozen=True)
class VertexLoweringRule:
    kind: int
    backend: str
    tensor_names: tuple[str, ...] = ()
    expression_head: str = ""
    full_tensor_network_ready: bool = False
    description: str = ""
    kernel: str = ""
    input_roles: tuple[str, str] = ("", "")
    output_role: str = ""
    coupling_mode: str = "none"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "tensor_names": list(self.tensor_names),
            "expression_head": self.expression_head,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "description": self.description,
            "kernel": self.kernel,
            "input_roles": list(self.input_roles),
            "output_role": self.output_role,
            "coupling_mode": self.coupling_mode,
        }


@dataclass(frozen=True)
class VertexEvaluationEquivalence:
    """Verified relation between oriented kernels used for evaluation reuse.

    ``factor`` states that the concrete oriented kernel equals ``factor`` times
    its equivalence-class representative after applying ``input_order``.
    Keeping this model-owned lets compiled UFO models persist relations proven
    from their actual lowered Symbolica expressions.
    """

    class_id: str
    factor: tuple[float, float] = (1.0, 0.0)
    input_order: tuple[int, int] = (0, 1)
    verified: bool = True

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("vertex evaluation equivalence class must be non-empty")
        if self.input_order not in {(0, 1), (1, 0)}:
            raise ValueError("vertex evaluation input order must be (0, 1) or (1, 0)")
        if not all(math.isfinite(component) for component in self.factor):
            raise ValueError("vertex evaluation equivalence factor must be finite")
        if self.factor == (0.0, 0.0):
            raise ValueError("vertex evaluation equivalence factor must be nonzero")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "factor": list(self.factor),
            "input_order": list(self.input_order),
            "verified": self.verified,
        }


@dataclass(frozen=True)
class PropagatorLoweringRule:
    particle_id: int
    chirality: int
    backend: str
    full_tensor_network_ready: bool
    applies_propagator: bool
    kernel: str
    kind: PropagatorKind
    mass_class: PropagatorMassClass
    gauge: PropagatorGauge | None = None
    numerator: str | None = None
    denominator: str | None = None
    custom_source: str | None = None
    auxiliary_policy: str | None = None
    goldstone_policy: GoldstonePolicy = "not-applicable"
    description: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "chirality": self.chirality,
            "backend": self.backend,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "applies_propagator": self.applies_propagator,
            "kernel": self.kernel,
            "kind": self.kind,
            "mass_class": self.mass_class,
            "gauge": self.gauge,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "custom_source": self.custom_source,
            "auxiliary_policy": self.auxiliary_policy,
            "goldstone_policy": self.goldstone_policy,
            "description": self.description,
        }


@dataclass(frozen=True)
class VertexLoweringCoverageEntry:
    kind: int
    vertex_count: int
    backend: str
    full_tensor_network_ready: bool
    tensor_names: tuple[str, ...]
    expression_head: str
    description: str
    kernel: str
    input_roles: tuple[str, str]
    output_role: str
    coupling_mode: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vertex_count": self.vertex_count,
            "backend": self.backend,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "tensor_names": list(self.tensor_names),
            "expression_head": self.expression_head,
            "description": self.description,
            "kernel": self.kernel,
            "input_roles": list(self.input_roles),
            "output_role": self.output_role,
            "coupling_mode": self.coupling_mode,
        }


@dataclass(frozen=True)
class VertexLoweringCoverageReport:
    model: str
    entries: tuple[VertexLoweringCoverageEntry, ...]

    @property
    def ready_kinds(self) -> tuple[int, ...]:
        return tuple(
            entry.kind for entry in self.entries if entry.full_tensor_network_ready
        )

    @property
    def pending_kinds(self) -> tuple[int, ...]:
        return tuple(
            entry.kind
            for entry in self.entries
            if entry.backend != "unimplemented" and not entry.full_tensor_network_ready
        )

    @property
    def unimplemented_kinds(self) -> tuple[int, ...]:
        return tuple(
            entry.kind for entry in self.entries if entry.backend == "unimplemented"
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "ready_kinds": list(self.ready_kinds),
            "pending_kinds": list(self.pending_kinds),
            "unimplemented_kinds": list(self.unimplemented_kinds),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


@dataclass
class Model:
    name: str
    compiled: Any | None = field(default=None, repr=False, compare=False)
    particles: dict[int, Particle] = field(default_factory=dict)
    vertices: tuple[Vertex, ...] = ()
    _source_ir_by_particle: dict[int, SourceIR] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _propagator_ir_by_state: dict[tuple[int, int], PropagatorIR] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _direct_contraction_ir_by_state: dict[
        tuple[int, int, int, int], ContractionIR | None
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _closure_contraction_ir_by_state: dict[tuple[int, int], ContractionIR | None] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
            compare=False,
        )
    )

    @cached_property
    def _species_by_pdg(self) -> dict[int, Particle]:
        species: dict[int, Particle] = {}
        for particle in self.particles.values():
            species.setdefault(particle.pdg, particle)
            species.setdefault(particle.anti_pdg, particle)
        return species

    @cached_property
    def _property_sign_by_pdg(self) -> dict[int, int]:
        signs: dict[int, int] = {}
        for particle in self.particles.values():
            signs.setdefault(particle.pdg, 1)
            if particle.anti_pdg != particle.pdg:
                signs.setdefault(particle.anti_pdg, -1)
            else:
                signs.setdefault(particle.anti_pdg, 1)
        return signs

    @cached_property
    def _vertices_by_input(self) -> dict[tuple[str, int, int], tuple[Vertex, ...]]:
        return {}

    def particle(self, pdg: int) -> Particle:
        species = self._species_by_pdg.get(pdg)
        if species is None:
            raise KeyError(f"particle not in model: {pdg}")
        return species

    def anti_particle(self, pdg: int) -> int:
        particle = self.particle(pdg)
        return particle.anti_pdg if particle.pdg == pdg else particle.pdg

    def mass(self, pdg: int) -> float:
        return self.particle(pdg).mass

    def width(self, pdg: int) -> float:
        return self.particle(pdg).width

    def spin(self, pdg: int) -> int:
        spin = self.particle(pdg).spin
        if spin < 0:
            raise ValueError(f"spin is ill-defined for particle {pdg}")
        return spin

    def dimension(self, pdg: int) -> int:
        return self.particle(pdg).dimension

    def charge(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).charge

    def weak_isospin_l(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).weak_isospin[0]

    def weak_isospin_r(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).weak_isospin[1]

    def color_rep(self, pdg: int) -> int:
        del pdg
        raise NotImplementedError("model does not define colour representations")

    def color_dim(self, pdg: int) -> int:
        return abs(self.color_rep(pdg))

    def is_fermion(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError("model does not define a fermion role")

    def is_chiral_eligible(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError("model does not define chiral source eligibility")

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError(
            "model does not define a fundamental colored-fermion role"
        )

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError(
            "model does not define a massless adjoint-vector role"
        )

    def is_singlet(self, pdg: int) -> bool:
        return self.color_rep(pdg) == 1

    def build_tensor_library(self) -> Any:
        raise NotImplementedError

    def vertex_lowering_rule(self, kind: int) -> VertexLoweringRule:
        raise NotImplementedError

    def vertex_evaluation_equivalence(
        self,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        """Return a conservative, always-safe evaluation equivalence class.

        Models with compiled symbolic kernels may override this with broader
        classes.  A class unique to one model type and vertex kind still enables
        exact fan-out reuse when several DAG attachments use the same kernel and
        input currents.
        """

        model_type = f"{type(self).__module__}.{type(self).__qualname__}"
        return VertexEvaluationEquivalence(class_id=f"{model_type}:{int(kind)}")

    def global_helicity_flip_equivalence_is_proven(
        self,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Return whether the model proves parity pairing for these vertices.

        Coupling-order names and particle labels are not such a proof. External
        models therefore remain fail-closed until their compiler records an
        expression-level equivalence certificate.
        """

        del vertices
        return False

    def pure_massless_adjoint_helicity_zero_rule_is_proven(
        self,
        process: Any,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Return whether all-equal and one-opposite helicities vanish.

        This tree-amplitude identity requires more than massless adjoint-vector
        external states. Generic and external models therefore remain
        fail-closed until their compiler can provide an expression-level proof.
        """

        del process, vertices
        return False

    def adjoint_current_reflection_phase(
        self,
        vertex: Vertex,
    ) -> tuple[float, float] | None:
        """Return a proven two-input adjoint-current reflection phase.

        A non-null result certifies the local off-shell identity obtained by
        exchanging the two inputs of ``vertex``.  This is deliberately
        separate from full-amplitude LC trace reflection: a pure gauge
        subcurrent may be reusable inside a mixed process even when the full
        process has no trace-reflection reduction.  Generic models remain
        fail-closed until their lowered component expressions prove the
        identity.
        """

        del vertex
        return None

    def lc_trace_reflection_equivalence_is_proven(self, process: Any) -> bool:
        """Return whether one LC trace may represent its reversed ordering.

        Reflection folding is an amplitude-level identity, not a consequence of
        adjoint colour alone.  External models therefore remain fail-closed
        unless their compiled model records a proof and overrides this hook.
        """

        del process
        return False

    def shared_single_trace_color_basis_is_proven(self, process: Any) -> bool:
        """Return whether NLC/full may use the shared single-trace recursion.

        Adjoint representation metadata alone does not prove that the model's
        color tensors obey the Yang--Mills trace relations assumed by that
        optimized recursion. Generic models therefore remain fail-closed.
        """

        del process
        return False

    def propagator_lowering_rule(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorLoweringRule:
        del particle_id, chirality
        raise NotImplementedError("model does not define propagator lowering")

    def source_spin_states(self, particle_id: int) -> tuple[SourceSpinState, ...]:
        if self.is_chiral_eligible(particle_id):
            return (
                SourceSpinState(helicity=-1, chirality=-1, spin_state=-1),
                SourceSpinState(helicity=1, chirality=1, spin_state=1),
            )
        spin = self.spin(particle_id)
        if spin == 1:
            return (SourceSpinState(helicity=0, chirality=0, spin_state=0),)
        if spin == 2:
            return (
                SourceSpinState(helicity=-1, chirality=0, spin_state=-1),
                SourceSpinState(helicity=1, chirality=0, spin_state=1),
            )
        if spin == 3:
            return (
                SourceSpinState(helicity=-1, chirality=0, spin_state=-1),
                SourceSpinState(helicity=0, chirality=0, spin_state=0),
                SourceSpinState(helicity=1, chirality=0, spin_state=1),
            )
        return (SourceSpinState(helicity=0, chirality=0, spin_state=0),)

    def source_wavefunction_kind(self, particle_id: int) -> str:
        dimension = self.current_dimension(particle_id)
        if self.is_fermion(particle_id):
            return "fermion"
        if dimension == 1:
            return "scalar"
        if dimension in {2, 4}:
            return "vector"
        if dimension == 16:
            return "spin2"
        return "unknown"

    def source_orientation(self, particle_id: int) -> str:
        """Return the source orientation from the model's anti-particle relation."""

        particle = self.particle(particle_id)
        antiparticle_id = self.anti_particle(particle_id)
        if self.anti_particle(antiparticle_id) != particle_id:
            raise ValueError(
                f"particle/antiparticle relation is not involutive for source "
                f"particle {particle_id}"
            )
        if antiparticle_id == particle_id:
            orientation = "self-conjugate"
        elif particle_id == particle.pdg and antiparticle_id == particle.anti_pdg:
            orientation = "particle"
        elif particle_id == particle.anti_pdg and antiparticle_id == particle.pdg:
            orientation = "antiparticle"
        else:
            raise ValueError(
                f"source particle {particle_id} is inconsistent with its model-owned "
                "particle/antiparticle relation"
            )
        if orientation == "self-conjugate" and self.is_fermion(particle_id):
            raise ValueError(
                f"unsupported self-conjugate fermion source {particle_id}: "
                "Majorana/FNV source wavefunctions are not implemented"
            )
        return orientation

    def _particle_identity_ir(self, particle_id: int) -> ParticleIdentityIR:
        """Return canonical oriented identity without assigning an SM role."""

        particle_id = int(particle_id)
        particle = self.particle(particle_id)
        anti_particle_id = int(self.anti_particle(particle_id))
        self_conjugate = particle_id == anti_particle_id
        canonical_id = f"model:{self.name}:state:{particle_id}"
        anti_canonical_id = f"model:{self.name}:state:{anti_particle_id}"
        return ParticleIdentityIR(
            canonical_id=canonical_id,
            species_id=f"model:{self.name}:species:{particle.pdg}",
            anti_canonical_id=anti_canonical_id,
            display_name=f"pdg_{particle_id}",
            anti_display_name=f"pdg_{anti_particle_id}",
            pdg_label=particle_id,
            anti_pdg_label=anti_particle_id,
            orientation=cast(ParticleOrientation, self.source_orientation(particle_id)),
            self_conjugate=self_conjugate,
        )

    def _source_crossing_ir(self, particle_id: int) -> CrossingIR:
        """Return the model-owned transform from outgoing to incoming source."""

        particle_id = int(particle_id)
        if self.is_chiral_eligible(particle_id):
            return CrossingIR(chirality_factor=-1, spin_state_factor=-1)
        if self.source_wavefunction_kind(particle_id) in {"vector", "spin2"}:
            return CrossingIR(helicity_factor=-1, spin_state_factor=-1)
        return CrossingIR()

    def _source_ir(self, particle_id: int) -> SourceIR:
        """Return the canonical source contract for one oriented particle.

        A compiled model's structural metadata is immutable during generation.
        Caching here ensures DAG construction, physics metadata, and execution
        schemas all consume the same contract object.
        """

        particle_id = int(particle_id)
        cache = self.__dict__.setdefault("_source_ir_by_particle", {})
        cached = cache.get(particle_id)
        if cached is not None:
            return cached
        source_ir = self._build_source_ir(particle_id)
        return cache.setdefault(particle_id, source_ir)

    def _build_source_ir(self, particle_id: int) -> SourceIR:
        """Build an uncached source contract for an oriented particle."""

        states = tuple(
            SourceStateIR(
                helicity=int(state.helicity),
                chirality=int(state.chirality),
                spin_state=state.spin_state,
            )
            for state in self.source_spin_states(particle_id)
        )
        dimensions = {
            int(self.current_dimension(particle_id, state.chirality))
            for state in states
        }
        if len(dimensions) != 1:
            raise ValueError(
                f"source {particle_id} has state-dependent component dimensions: "
                f"{sorted(dimensions)}"
            )
        wavefunction_family = self.source_wavefunction_kind(particle_id)
        if wavefunction_family not in {
            "scalar",
            "fermion",
            "vector",
            "spin2",
            "ghost",
            "auxiliary",
        }:
            raise ValueError(
                f"source {particle_id} has unsupported wavefunction family "
                f"{wavefunction_family!r}"
            )
        auxiliary = self.auxiliary_kind(particle_id)
        statistics: ParticleStatistics
        if auxiliary is not None:
            statistics = "auxiliary"
        elif self.spin(particle_id) < 0:
            statistics = "ghost"
        elif self.is_fermion(particle_id):
            statistics = "fermion"
        else:
            statistics = "boson"
        return SourceIR(
            identity=self._particle_identity_ir(particle_id),
            statistics=statistics,
            wavefunction_family=cast(WavefunctionFamily, wavefunction_family),
            component_dimension=dimensions.pop(),
            states=states,
            crossing=self._source_crossing_ir(particle_id),
            basis=self._current_basis(particle_id, states[0].chirality),
            mass_parameter=self._runtime_parameter_name(particle_id, "mass"),
            width_parameter=self._runtime_parameter_name(particle_id, "width"),
        )

    def _propagator_ir(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorIR:
        """Project the active propagator lowering into a typed model contract."""

        particle_id = int(particle_id)
        chirality = int(chirality)
        key = (particle_id, chirality)
        cache = self.__dict__.setdefault("_propagator_ir_by_state", {})
        cached = cache.get(key)
        if cached is not None:
            return cached
        rule = self.propagator_lowering_rule(particle_id, chirality)
        auxiliary_policy = rule.auxiliary_policy
        if not rule.applies_propagator:
            auxiliary_policy = (
                auxiliary_policy or self.auxiliary_kind(particle_id) or rule.kernel
            )
        default_numerator, default_denominator = _propagator_formula_metadata(
            rule.kind, rule.gauge
        )
        result = PropagatorIR(
            identity=self._particle_identity_ir(particle_id),
            chirality=chirality,
            kind=rule.kind,
            backend=rule.backend,
            basis=self._current_basis(particle_id, chirality),
            applies_propagator=rule.applies_propagator,
            kernel=rule.kernel,
            full_tensor_network_ready=rule.full_tensor_network_ready,
            mass_class=rule.mass_class,
            gauge=rule.gauge,
            numerator=rule.numerator or default_numerator,
            denominator=rule.denominator or default_denominator,
            mass_parameter=self._runtime_parameter_name(particle_id, "mass"),
            width_parameter=self._runtime_parameter_name(particle_id, "width"),
            custom_source=rule.custom_source,
            auxiliary_policy=auxiliary_policy,
            goldstone_policy=rule.goldstone_policy,
            description=rule.description,
        )
        return cache.setdefault(key, result)

    def _current_basis(self, particle_id: int, chirality: int = 0) -> str:
        dimension = int(self.current_dimension(particle_id, chirality))
        auxiliary = self.auxiliary_kind(particle_id)
        if auxiliary is not None:
            return f"auxiliary:{auxiliary}"
        if self.is_fermion(particle_id):
            return "weyl-chiral" if dimension == 2 else "dirac"
        family = self.source_wavefunction_kind(particle_id)
        if family == "scalar" and dimension == 1:
            return "scalar"
        if family == "vector" and dimension == 4:
            return "lorentz-vector"
        if family == "spin2" and dimension == 16:
            return "lorentz-rank-2"
        if dimension == 6:
            return "antisymmetric-lorentz-pair"
        return f"components:{dimension}"

    def _runtime_parameter_name(self, particle_id: int, kind: str) -> str | None:
        provider = getattr(self, f"runtime_{kind}_parameter_name", None)
        if not callable(provider):
            return None
        name = provider(int(particle_id))
        return None if name is None else str(name)

    def runtime_normalization_payload(self, dag: Any) -> dict[str, object]:
        """Return model-owned averaging, symmetry, color, and coupling factors."""

        del dag
        raise NotImplementedError("model does not define runtime normalization")

    def runtime_normalization_parameter_defaults(self) -> Mapping[str, float]:
        """Return mutable runtime parameters used by normalization only."""

        return {}

    def allowed_quantum_flows(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[QuantumFlow, ...]:
        result_particle = vertex.particles[2]
        chiralities = _model_vertex_result_chiralities(
            self,
            vertex,
            left_index,
            right_index,
        )
        flows: list[QuantumFlow] = []
        for chirality in chiralities:
            flows.append(
                QuantumFlow(
                    chirality=chirality,
                    spin_state=self.result_spin_state(result_particle, chirality),
                    flavour_flow=self.combine_flavour_flow(
                        result_particle,
                        left_index,
                        right_index,
                    ),
                    quantum_number_flow=self.quantum_number_flow(result_particle),
                    coupling=vertex.coupling,
                )
            )
        return tuple(flows)

    def combine_flavour_flow(
        self,
        result_particle: int,
        left_index: Any,
        right_index: Any,
    ) -> tuple[int, ...]:
        left_pdg = _index_particle_id(left_index)
        right_pdg = _index_particle_id(right_index)
        left_flow = _index_flavour_flow(left_index)
        right_flow = _index_flavour_flow(right_index)

        if self.is_fermion(result_particle):
            if self.is_fermion(left_pdg):
                return _append_flavour_transition(left_flow, result_particle)
            if self.is_fermion(right_pdg):
                return _append_flavour_transition(right_flow, result_particle)

        if self.is_fermion(left_pdg) and self.is_fermion(right_pdg):
            return (*left_flow, *right_flow, result_particle)

        return (result_particle,)

    def result_spin_state(self, particle_id: int, chirality: int) -> int:
        if self.is_fermion(particle_id):
            return chirality
        return 0

    def current_dimension(self, particle_id: int, chirality: int = 0) -> int:
        if chirality != 0 and self.is_chiral_eligible(particle_id):
            return 2
        try:
            return self.dimension(particle_id)
        except KeyError:
            return 0

    def direct_contraction_ir(
        self,
        left_particle_id: int,
        right_particle_id: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
    ) -> ContractionIR | None:
        """Return an explicitly installed contraction for two current states."""

        key = (
            int(left_particle_id),
            int(left_chirality),
            int(right_particle_id),
            int(right_chirality),
        )
        return self._direct_contraction_ir_by_state.get(key)

    def closure_contraction_ir(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> ContractionIR | None:
        """Return an explicitly installed one-current closure projection."""

        key = (int(particle_id), int(chirality))
        return self._closure_contraction_ir_by_state.get(key)

    def direct_contraction_possible(
        self,
        left_particle_id: int,
        right_particle_id: int,
    ) -> bool:
        """Over-approximate closure using only declared contraction records."""

        left = int(left_particle_id)
        right = int(right_particle_id)
        return any(
            value is not None and key[0] == left and key[2] == right
            for key, value in self._direct_contraction_ir_by_state.items()
        )

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        del particle_id
        raise NotImplementedError("model does not define exact quantum-number flow")

    def auxiliary_kind(self, particle_id: int) -> str | None:
        del particle_id
        return None

    def vertex_component_expression(
        self,
        kind: int,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        result_particle_id: int,
        result_chirality: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
        coupling: tuple[Any, Any] = (1.0, 0.0),
        left_momentum: Sequence[Any] | None = None,
        right_momentum: Sequence[Any] | None = None,
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        del propagator
        raise NotImplementedError

    def iter_vertices(self, *, color_accuracy: str = "lc") -> tuple[Vertex, ...]:
        if color_accuracy == "lc":
            return tuple(self.vertices)
        if color_accuracy in {"nlc", "full"}:
            return tuple(self.vertices)
        raise ValueError(f"unknown colour accuracy: {color_accuracy}")

    def vertices_for_inputs(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        key = (color_accuracy, int(left_pdg), int(right_pdg))
        if key not in self._vertices_by_input:
            self._vertices_by_input[key] = tuple(
                vertex
                for vertex in self.iter_vertices(color_accuracy=color_accuracy)
                if vertex.particles[0] == left_pdg and vertex.particles[1] == right_pdg
            )
        return self._vertices_by_input[key]

    def vertices_accepting(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        """Return model vertices for a local current-current combination.

        This is the process-generic name used by the DAG compiler.  It is a
        thin alias around the model table lookup, but keeping the name at the
        model boundary prevents production code from classifying whole process
        families before asking which local interactions are allowed.
        """

        return self.vertices_for_inputs(
            left_pdg,
            right_pdg,
            color_accuracy=color_accuracy,
        )

    def skip_duplicate_vertex_orientation(self, vertex: Vertex) -> bool:
        """Return whether a mirrored table entry should be skipped by DAG sweeps.

        Generic DAG generation asks the model because duplicated orientations are
        a model-table convention, not a process-family rule.
        """

        return False

    def vertex_closure_allowed(self, vertex: Vertex) -> bool:
        """Return whether a scalar vertex result represents a vacuum closure.

        The legacy built-in model uses auxiliary scalar result particles for a
        few contact interactions. Generic UFO particles are ordinary currents
        unless their compiled lowering explicitly introduces such an auxiliary.
        """

        return True

    def vertex_coupling_orders(self, vertex: Vertex) -> CouplingOrders:
        """Return model-generic coupling-order increments for one vertex.

        The keys intentionally mirror UFO-style coupling-order names.  They are
        used only as local model metadata, so DAG pruning can cap e.g. QCD or
        QED order without recognizing whole process families.
        """

        del vertex
        return ()

    def coupling_order_hierarchies(self) -> dict[str, int]:
        """Return UFO-style priorities used by minimal-order generation.

        A lower hierarchy value is preferred. Models without explicit order
        metadata conservatively assign equal priority to every observed order.
        """

        return {}

    def vertex_color_weight(
        self,
        vertex: Vertex,
        *,
        color_accuracy: str,
    ) -> tuple[float, float]:
        """Return the model-owned coefficient for one projected color vertex."""

        del vertex, color_accuracy
        return (1.0, 0.0)

    def vertex_color_structure(self, vertex: Vertex) -> str:
        """Return the model-level color tensor family for local projection.

        Built-in kernels already encode their projected color algebra. External
        tensor models override this hook so the generic color engine can apply
        flow-dependent identities without recognizing particles or processes.
        """

        del vertex
        return "model-defined"

    def vertex_is_internal_contact_fragment(self, vertex: Vertex) -> bool:
        del vertex
        return False

    def combine_coupling_orders(
        self,
        left_index: Any,
        right_index: Any,
        vertex: Vertex,
    ) -> CouplingOrders:
        totals: dict[str, int] = {}
        for orders in (
            _index_coupling_orders(left_index),
            _index_coupling_orders(right_index),
            self.vertex_coupling_orders(vertex),
        ):
            for name, value in orders:
                totals[str(name).upper()] = totals.get(str(name).upper(), 0) + int(
                    value
                )
        return tuple(sorted((name, value) for name, value in totals.items() if value))

    def current_allowed(self, index: Any) -> bool:
        """Return whether a generated current index is valid in this model."""

        try:
            particle_id = int(index.particle_id)
            chirality = int(getattr(index, "chirality", 0))
            if chirality != 0 and self.is_chiral_eligible(particle_id):
                return True
            return self.dimension(particle_id) > 0
        except (KeyError, TypeError, ValueError):
            return False

    def vertex_lowering_coverage(self) -> VertexLoweringCoverageReport:
        counts = Counter(vertex.kind for vertex in self.vertices)
        entries: list[VertexLoweringCoverageEntry] = []
        for kind, count in sorted(counts.items()):
            rule = self.vertex_lowering_rule(kind)
            entries.append(
                VertexLoweringCoverageEntry(
                    kind=kind,
                    vertex_count=count,
                    backend=rule.backend,
                    full_tensor_network_ready=rule.full_tensor_network_ready,
                    tensor_names=rule.tensor_names,
                    expression_head=rule.expression_head,
                    description=rule.description,
                    kernel=rule.kernel,
                    input_roles=rule.input_roles,
                    output_role=rule.output_role,
                    coupling_mode=rule.coupling_mode,
                )
            )
        return VertexLoweringCoverageReport(
            model=self.name,
            entries=tuple(entries),
        )

    def _species_particle(self, pdg: int) -> Particle | None:
        return self._species_by_pdg.get(pdg)

    def _property_sign(self, pdg: int) -> int:
        try:
            return self._property_sign_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc


def _propagator_formula_metadata(
    kind: PropagatorKind,
    gauge: PropagatorGauge | None,
) -> tuple[str | None, str | None]:
    """Return descriptive formulas for the implemented propagator kernel."""

    if kind == "identity":
        return ("identity", "1")
    if kind == "custom":
        return (None, None)
    if kind == "weyl-fermion":
        return ("i*weyl_slash(momentum)", "momentum_squared")
    if kind == "dirac-fermion":
        return (
            "i*(dirac_slash(momentum)+oriented_mass)",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "vector" and gauge == "feynman":
        return ("-i*metric", "momentum_squared")
    if kind == "vector" and gauge == "unitary":
        return (
            "-i*(metric-momentum_outer/mass_squared)",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "spin2":
        return (
            "i*spin2_projector",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "scalar":
        return ("i", "momentum_squared-mass_squared+i*mass*width")
    return (None, None)


from .expressions import (  # noqa: E402
    _append_flavour_transition,
    _index_coupling_orders,
    _index_flavour_flow,
    _index_particle_id,
    _model_vertex_result_chiralities,
)

__all__ = [
    "ContractionIR",
    "CouplingOrders",
    "Model",
    "Particle",
    "PropagatorLoweringRule",
    "QuantumFlow",
    "QuantumNumberFlow",
    "SourceSpinState",
    "Vertex",
    "VertexEvaluationEquivalence",
    "VertexLoweringRule",
]
