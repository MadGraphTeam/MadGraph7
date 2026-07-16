# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any


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
    charge_flow: int
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
    description: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "chirality": self.chirality,
            "backend": self.backend,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "applies_propagator": self.applies_propagator,
            "kernel": self.kernel,
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
    def _color_rep_by_pdg(self) -> dict[int, int]:
        reps: dict[int, int] = {}
        for pdg, particle in self._species_by_pdg.items():
            color = particle.color_rep
            if self._property_sign_by_pdg[pdg] < 0 and abs(color) == 3:
                reps[pdg] = -color
            else:
                reps[pdg] = color
        return reps

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
        try:
            return self._color_rep_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc

    def color_dim(self, pdg: int) -> int:
        return abs(self.color_rep(pdg))

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
            description="no pyamplicol propagator lowering is registered",
        )

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
                    charge_flow=self.charge_units(result_particle),
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

    def charge_units(self, particle_id: int) -> int:
        return round(3.0 * self.charge(particle_id))

    def auxiliary_kind(self, particle_id: int) -> str | None:
        if self.is_tensor(particle_id):
            return "antisymmetric-tensor"
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
    ) -> tuple[Any, ...]:
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

        return (("QED", 1),)

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

    def three_gluon_current_expression(
        self,
        *,
        left_slot: Any,
        right_slot: Any,
        output_slot: Any,
        left_momentum_tensor_name: str,
        right_momentum_tensor_name: str,
        dummy_prefix: str,
    ) -> Any:
        raise NotImplementedError

    def gluon_propagator_tensor_data(
        self,
        momentum: Sequence[Any],
    ) -> list[Any]:
        raise NotImplementedError

    def quark_weyl_propagator_tensor_data(
        self,
        momentum: Sequence[Any],
        *,
        chirality: int,
    ) -> list[Any]:
        raise NotImplementedError

    def _species_particle(self, pdg: int) -> Particle | None:
        return self._species_by_pdg.get(pdg)

    def _property_sign(self, pdg: int) -> int:
        try:
            return self._property_sign_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc


from .expressions import (  # noqa: E402
    _append_flavour_transition,
    _index_coupling_orders,
    _index_flavour_flow,
    _index_particle_id,
    _model_vertex_result_chiralities,
)

__all__ = [
    "CouplingOrders",
    "Model",
    "Particle",
    "PropagatorLoweringRule",
    "QuantumFlow",
    "SourceSpinState",
    "Vertex",
    "VertexEvaluationEquivalence",
    "VertexLoweringRule",
]
