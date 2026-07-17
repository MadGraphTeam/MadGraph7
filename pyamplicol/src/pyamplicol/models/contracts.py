# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .._internal.physics.symbols import symbols
from ._physics_ir import ContractionIR
from .base import QuantumNumberFlow
from .contact_decomposition import (
    CompiledContactDecompositionProof,
    CompiledContactDecompositionSplit,
    CompiledContactDummyIndexMapping,
    CompiledContactOrientationProof,
    CompiledContactUnsupportedReason,
)

PROPAGATOR_SOURCE_FIELD = "pyamplicol_source"
DEFAULT_FEYNMAN_PROPAGATOR_SOURCE = "default-feynman"
MODEL_SUPPLIED_PROPAGATOR_SOURCE = "model-supplied"

SUPPORTED_COLOR_REPRESENTATIONS = frozenset({-3, 1, 3, 8})


def validate_color_representation(value: int, *, context: str = "particle") -> int:
    representation = int(value)
    if representation not in SUPPORTED_COLOR_REPRESENTATIONS:
        raise ValueError(
            f"{context} uses unsupported UFO color representation {representation}"
        )
    return representation


def validate_quantum_number_flow(
    value: object,
    *,
    context: str = "particle",
) -> QuantumNumberFlow:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{context} quantum numbers must be a sequence")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(
                f"{context} quantum-number entries must be [name, expression] pairs"
            )
        name, expression = item
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{context} quantum-number names must be non-empty strings"
            )
        if not isinstance(expression, str) or not expression:
            raise ValueError(
                f"{context} quantum-number expressions must be non-empty strings"
            )
        _constant_quantum_number_expression(
            expression,
            context=f"{context} quantum number {name!r}",
        )
        result.append((name, expression))

    names = tuple(name for name, _expression in result)
    if names != tuple(sorted(set(names))):
        raise ValueError(f"{context} quantum-number names must be sorted and unique")
    return tuple(result)


def _constant_quantum_number_expression(
    expression: str,
    *,
    context: str,
) -> Any:
    from . import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    try:
        parsed = _sym.E(expression)
    except Exception as exc:
        raise ValueError(f"{context} is not a valid Symbolica expression") from exc
    if parsed.get_all_symbols(False):
        raise ValueError(f"{context} must be symbol-free")
    if not parsed.is_real():
        raise ValueError(f"{context} must be real")
    if not parsed.is_finite():
        raise ValueError(f"{context} must be a finite real constant")
    return parsed


@dataclass(frozen=True)
class CompiledCouplingOrder:
    name: str
    expansion_order: int
    hierarchy: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expansion_order": self.expansion_order,
            "hierarchy": self.hierarchy,
        }


@dataclass(frozen=True)
class CompiledParameterRecord:
    name: str
    nature: str
    parameter_type: str
    value: tuple[float, float] | None
    expression: str | None
    resolved_expression: str
    lhablock: str | None
    lhacode: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nature": self.nature,
            "parameter_type": self.parameter_type,
            "value": None if self.value is None else list(self.value),
            "expression": self.expression,
            "resolved_expression": self.resolved_expression,
            "lhablock": self.lhablock,
            "lhacode": list(self.lhacode),
        }


@dataclass(frozen=True)
class CompiledParticleRecord:
    name: str
    antiname: str
    pdg_code: int
    spin: int
    color: int
    mass: str
    width: str
    charge: float
    quantum_numbers: QuantumNumberFlow
    ghost_number: int
    propagating: bool
    goldstoneboson: bool
    propagator: str | None
    component_dimension: int | None = None
    auxiliary_kind: str | None = None
    statistics: str = ""
    wavefunction_family: str = ""
    color_role: str = ""
    self_conjugate: bool | None = None
    source_orientation: str = ""

    def __post_init__(self) -> None:
        if self.component_dimension is not None:
            if isinstance(self.component_dimension, bool) or not isinstance(
                self.component_dimension, int
            ):
                raise TypeError(
                    f"particle {self.name!r} component dimension must be an integer"
                )
            if self.component_dimension <= 0:
                raise ValueError(
                    f"particle {self.name!r} component dimension must be positive"
                )
        if not math.isfinite(float(self.charge)):
            raise ValueError(f"particle {self.name!r} charge must be finite")
        quantum_numbers = validate_quantum_number_flow(
            self.quantum_numbers,
            context=f"particle {self.name!r}",
        )
        if not any(name == "electric_charge" for name, _ in quantum_numbers):
            raise ValueError(
                f"particle {self.name!r} must declare exact electric_charge metadata"
            )
        object.__setattr__(
            self,
            "quantum_numbers",
            quantum_numbers,
        )
        derived = _particle_role_metadata(self)
        for field_name, expected in derived.items():
            supplied = getattr(self, field_name)
            if supplied in {"", None}:
                object.__setattr__(self, field_name, expected)
            elif supplied != expected:
                raise ValueError(
                    f"particle {self.name!r} has inconsistent {field_name}: "
                    f"{supplied!r}, expected {expected!r}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "antiname": self.antiname,
            "pdg_code": self.pdg_code,
            "spin": self.spin,
            "color": self.color,
            "mass": self.mass,
            "width": self.width,
            "charge": self.charge,
            "quantum_numbers": [list(item) for item in self.quantum_numbers],
            "ghost_number": self.ghost_number,
            "propagating": self.propagating,
            "goldstoneboson": self.goldstoneboson,
            "propagator": self.propagator,
            "component_dimension": self.component_dimension,
            "auxiliary_kind": self.auxiliary_kind,
            "statistics": self.statistics,
            "wavefunction_family": self.wavefunction_family,
            "color_role": self.color_role,
            "self_conjugate": self.self_conjugate,
            "source_orientation": self.source_orientation,
        }


def _particle_role_metadata(particle: CompiledParticleRecord) -> dict[str, object]:
    representation = validate_color_representation(
        particle.color,
        context=f"particle {particle.name!r}",
    )
    if particle.ghost_number != 0:
        statistics = "ghost"
    elif particle.auxiliary_kind is not None or particle.spin < 0:
        statistics = "auxiliary"
    elif particle.spin % 2 == 0:
        statistics = "fermion"
    else:
        statistics = "boson"

    if statistics == "fermion":
        wavefunction_family = "fermion"
    elif particle.spin == 1:
        wavefunction_family = "scalar"
    elif particle.spin == 3:
        wavefunction_family = "vector"
    elif particle.spin == 5:
        wavefunction_family = "spin2"
    elif statistics == "ghost":
        wavefunction_family = "ghost"
    else:
        wavefunction_family = "auxiliary"

    color_role = {
        -3: "antifundamental",
        1: "singlet",
        3: "fundamental",
        8: "adjoint",
    }[representation]
    self_conjugate = particle.name == particle.antiname
    if self_conjugate:
        source_orientation = "self-conjugate"
    elif particle.pdg_code > 0:
        source_orientation = "particle"
    elif particle.pdg_code < 0:
        source_orientation = "antiparticle"
    else:
        raise ValueError(
            f"non-self-conjugate particle {particle.name!r} cannot use PDG code zero"
        )
    return {
        "statistics": statistics,
        "wavefunction_family": wavefunction_family,
        "color_role": color_role,
        "self_conjugate": self_conjugate,
        "source_orientation": source_orientation,
    }


@dataclass(frozen=True)
class CompiledCouplingRecord:
    name: str
    expression: str
    resolved_expression: str
    value: tuple[float, float] | None
    orders: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expression": self.expression,
            "resolved_expression": self.resolved_expression,
            "value": None if self.value is None else list(self.value),
            "orders": [[name, value] for name, value in self.orders],
        }


@dataclass(frozen=True)
class CompiledPropagatorRecord:
    name: str
    particle: str
    numerator: str
    denominator: str
    custom: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "particle": self.particle,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "custom": self.custom,
        }


@dataclass(frozen=True)
class CompiledVertexTerm:
    id: int
    vertex: str
    particles: tuple[str, ...]
    color_index: int
    lorentz_index: int
    color_source: str
    color_expression: str
    lorentz_name: str
    lorentz_source: str
    lorentz_expression: str
    coupling: str
    coupling_expression: str
    coupling_orders: tuple[tuple[str, int], ...]
    backend: str = "ufo"
    lc_color_normalization_power: int = 0
    contact_decomposition_proof: CompiledContactDecompositionProof | None = None

    @property
    def valence(self) -> int:
        return len(self.particles)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "vertex": self.vertex,
            "particles": list(self.particles),
            "valence": self.valence,
            "color_index": self.color_index,
            "lorentz_index": self.lorentz_index,
            "color_source": self.color_source,
            "color_expression": self.color_expression,
            "lorentz_name": self.lorentz_name,
            "lorentz_source": self.lorentz_source,
            "lorentz_expression": self.lorentz_expression,
            "coupling": self.coupling,
            "coupling_expression": self.coupling_expression,
            "coupling_orders": [[name, value] for name, value in self.coupling_orders],
            "backend": self.backend,
            "lc_color_normalization_power": self.lc_color_normalization_power,
        }
        if self.contact_decomposition_proof is not None:
            payload["contact_decomposition_proof"] = (
                self.contact_decomposition_proof.to_dict()
            )
        return payload


@dataclass(frozen=True)
class CompiledOrientedKernel:
    kind: int
    term_id: int
    vertex: str
    particles: tuple[str, str, str]
    source_particle_legs: tuple[int, int, int]
    component_expressions: tuple[str, ...]
    coupling_expression: str
    coupling_orders: tuple[tuple[str, int], ...]
    runtime_parameters: tuple[str, ...]
    color_source: str
    color_expression: str
    color_projection_structure: str | None = None
    color_projection_coefficient: tuple[float, float] | None = None
    lc_color_normalization_power: int = 0
    term_ids: tuple[int, ...] = ()
    evaluation_class: str = ""
    evaluation_factor: tuple[float, float] = (1.0, 0.0)
    evaluation_input_order: tuple[int, int] = (0, 1)
    evaluation_equivalence_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "term_id": self.term_id,
            "vertex": self.vertex,
            "particles": list(self.particles),
            "source_particle_legs": list(self.source_particle_legs),
            "component_expressions": list(self.component_expressions),
            "coupling_expression": self.coupling_expression,
            "coupling_orders": [[name, value] for name, value in self.coupling_orders],
            "runtime_parameters": list(self.runtime_parameters),
            "color_source": self.color_source,
            "color_expression": self.color_expression,
            "color_projection_structure": self.color_projection_structure,
            "color_projection_coefficient": (
                None
                if self.color_projection_coefficient is None
                else list(self.color_projection_coefficient)
            ),
            "lc_color_normalization_power": self.lc_color_normalization_power,
            "term_ids": list(self.term_ids or (self.term_id,)),
            "evaluation_class": self.evaluation_class,
            "evaluation_factor": list(self.evaluation_factor),
            "evaluation_input_order": list(self.evaluation_input_order),
            "evaluation_equivalence_verified": (self.evaluation_equivalence_verified),
        }


@dataclass(frozen=True)
class _ContactTreeNode:
    legs: tuple[int, ...]
    particle: CompiledParticleRecord
    physical_leg: int | None = None
    left: _ContactTreeNode | None = None
    right: _ContactTreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.physical_leg is not None


@dataclass(frozen=True)
class CompiledDirectContractionRecord:
    left_particle: str
    left_chirality: int
    right_particle: str
    right_chirality: int
    contraction_ir: ContractionIR

    def __post_init__(self) -> None:
        _validate_particle_selector_name(self.left_particle, "left particle")
        _validate_particle_selector_name(self.right_particle, "right particle")
        _validate_chirality(self.left_chirality, "left chirality")
        _validate_chirality(self.right_chirality, "right chirality")
        if not isinstance(self.contraction_ir, ContractionIR):
            raise TypeError("direct contraction must contain a ContractionIR")

    @property
    def selector(self) -> tuple[str, int, str, int]:
        return (
            self.left_particle,
            self.left_chirality,
            self.right_particle,
            self.right_chirality,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left_particle": self.left_particle,
            "left_chirality": self.left_chirality,
            "right_particle": self.right_particle,
            "right_chirality": self.right_chirality,
            "contraction_ir": self.contraction_ir.to_json_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledDirectContractionRecord:
        fields = _strict_record_fields(
            payload,
            required={
                "left_particle",
                "left_chirality",
                "right_particle",
                "right_chirality",
                "contraction_ir",
            },
            context="compiled direct contraction",
        )
        return cls(
            left_particle=_strict_string(fields["left_particle"], "left particle"),
            left_chirality=_strict_integer(fields["left_chirality"], "left chirality"),
            right_particle=_strict_string(fields["right_particle"], "right particle"),
            right_chirality=_strict_integer(
                fields["right_chirality"], "right chirality"
            ),
            contraction_ir=ContractionIR.from_json_dict(
                _strict_mapping(fields["contraction_ir"], "contraction IR")
            ),
        )


@dataclass(frozen=True)
class CompiledClosureContractionRecord:
    particle: str
    chirality: int
    contraction_ir: ContractionIR

    def __post_init__(self) -> None:
        _validate_particle_selector_name(self.particle, "particle")
        _validate_chirality(self.chirality, "chirality")
        if not isinstance(self.contraction_ir, ContractionIR):
            raise TypeError("closure contraction must contain a ContractionIR")

    @property
    def selector(self) -> tuple[str, int]:
        return self.particle, self.chirality

    def to_dict(self) -> dict[str, object]:
        return {
            "particle": self.particle,
            "chirality": self.chirality,
            "contraction_ir": self.contraction_ir.to_json_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledClosureContractionRecord:
        fields = _strict_record_fields(
            payload,
            required={"particle", "chirality", "contraction_ir"},
            context="compiled closure contraction",
        )
        return cls(
            particle=_strict_string(fields["particle"], "particle"),
            chirality=_strict_integer(fields["chirality"], "chirality"),
            contraction_ir=ContractionIR.from_json_dict(
                _strict_mapping(fields["contraction_ir"], "contraction IR")
            ),
        )


@dataclass(frozen=True)
class CompiledModelIR:
    name: str
    orders: tuple[CompiledCouplingOrder, ...]
    parameters: tuple[CompiledParameterRecord, ...]
    particles: tuple[CompiledParticleRecord, ...]
    couplings: tuple[CompiledCouplingRecord, ...]
    propagators: tuple[CompiledPropagatorRecord, ...]
    vertex_terms: tuple[CompiledVertexTerm, ...]
    oriented_kernels: tuple[CompiledOrientedKernel, ...]
    direct_contractions: tuple[CompiledDirectContractionRecord, ...]
    closure_contractions: tuple[CompiledClosureContractionRecord, ...]

    def __post_init__(self) -> None:
        self._validate_particle_identities()
        self._validate_contractions()
        self._validate_contact_decomposition_proofs()
        for context, expression in self._executable_expressions():
            if "UFO::" in expression:
                raise ValueError(
                    f"{context} retains a process-global UFO symbol; "
                    "regenerate it through the model symbol registry"
                )

    def _validate_particle_identities(self) -> None:
        by_name: dict[str, CompiledParticleRecord] = {}
        by_pdg: dict[int, CompiledParticleRecord] = {}
        for particle in self.particles:
            if particle.name in by_name:
                raise ValueError(
                    f"compiled model contains duplicate particle name {particle.name!r}"
                )
            if particle.pdg_code in by_pdg:
                raise ValueError(
                    f"compiled model contains duplicate PDG code {particle.pdg_code}"
                )
            by_name[particle.name] = particle
            by_pdg[particle.pdg_code] = particle
        for particle in self.particles:
            anti = by_name.get(particle.antiname)
            if anti is None:
                raise ValueError(
                    f"particle {particle.name!r} refers to absent antiparticle "
                    f"{particle.antiname!r}"
                )
            if anti.antiname != particle.name:
                raise ValueError(
                    f"particle/antiparticle relation is not involutive for "
                    f"{particle.name!r} and {anti.name!r}"
                )
            if anti is not particle and anti.pdg_code != -particle.pdg_code:
                raise ValueError(
                    f"non-self-conjugate pair {particle.name!r}/{anti.name!r} must "
                    "use opposite signed PDG codes"
                )
            if anti is particle:
                for name, expression in particle.quantum_numbers:
                    parsed = _constant_quantum_number_expression(
                        expression,
                        context=(
                            f"self-conjugate particle {particle.name!r} quantum "
                            f"number {name!r}"
                        ),
                    )
                    if parsed.to_canonical_string() != "0":
                        raise ValueError(
                            f"self-conjugate particle {particle.name!r} must have "
                            f"zero quantum number {name!r}"
                        )
                continue
            particle_names = tuple(name for name, _ in particle.quantum_numbers)
            anti_names = tuple(name for name, _ in anti.quantum_numbers)
            if particle_names != anti_names:
                raise ValueError(
                    f"particle/antiparticle pair {particle.name!r}/{anti.name!r} "
                    "must declare the same quantum numbers"
                )
            for (name, expression), (_anti_name, anti_expression) in zip(
                particle.quantum_numbers,
                anti.quantum_numbers,
                strict=True,
            ):
                total = _constant_quantum_number_expression(
                    expression,
                    context=f"particle {particle.name!r} quantum number {name!r}",
                ) + _constant_quantum_number_expression(
                    anti_expression,
                    context=f"particle {anti.name!r} quantum number {name!r}",
                )
                if total.to_canonical_string() != "0":
                    raise ValueError(
                        f"particle/antiparticle pair {particle.name!r}/{anti.name!r} "
                        f"must have exactly negated quantum number {name!r}"
                    )

    def _validate_contact_decomposition_proofs(self) -> None:
        for term in self.vertex_terms:
            proof = term.contact_decomposition_proof
            requires_proof = (
                term.backend == "ufo"
                and term.valence == 4
                and "ufo_momentum_" not in term.lorentz_expression
            )
            if proof is None and requires_proof:
                raise ValueError(
                    f"UFO four-point term {term.id} has no contact decomposition proof"
                )
            if proof is None:
                continue
            if term.valence != 4:
                raise ValueError(
                    f"contact decomposition proof on non-four-point term {term.id}"
                )
            if not proof.matches(term):
                raise ValueError(
                    f"contact decomposition proof identity mismatch for term {term.id}"
                )

    def _validate_contractions(self) -> None:
        particles = {particle.name: particle for particle in self.particles}
        parameters = {parameter.name: parameter for parameter in self.parameters}
        propagators = {propagator.name: propagator for propagator in self.propagators}
        direct_selectors: set[tuple[str, int, str, int]] = set()
        for record in self.direct_contractions:
            if not isinstance(record, CompiledDirectContractionRecord):
                raise TypeError(
                    "compiled model direct contractions must contain typed records"
                )
            if record.selector in direct_selectors:
                raise ValueError(
                    f"compiled model contains duplicate direct contraction selector "
                    f"{record.selector!r}"
                )
            direct_selectors.add(record.selector)
            try:
                left = particles[record.left_particle]
                right = particles[record.right_particle]
            except KeyError as exc:
                raise ValueError(
                    f"direct contraction refers to absent particle {exc.args[0]!r}"
                ) from exc
            if left.antiname != right.name or right.antiname != left.name:
                raise ValueError(
                    f"direct contraction particles {left.name!r}/{right.name!r} "
                    "are not an antiparticle pair"
                )
            left_dimension = compiled_current_dimension(
                left,
                record.left_chirality,
                parameters=parameters,
                propagators=propagators,
            )
            right_dimension = compiled_current_dimension(
                right,
                record.right_chirality,
                parameters=parameters,
                propagators=propagators,
            )
            coefficient_count = len(record.contraction_ir.coefficients)
            if left_dimension != right_dimension or coefficient_count != left_dimension:
                raise ValueError(
                    f"direct contraction selector {record.selector!r} has "
                    f"{coefficient_count} coefficients for current dimensions "
                    f"{left_dimension} and {right_dimension}"
                )
            _validate_concrete_chirality_relation(
                record.contraction_ir,
                record.left_chirality,
                record.right_chirality,
                context=f"direct contraction selector {record.selector!r}",
            )

        closure_selectors: set[tuple[str, int]] = set()
        for record in self.closure_contractions:
            if not isinstance(record, CompiledClosureContractionRecord):
                raise TypeError(
                    "compiled model closure contractions must contain typed records"
                )
            if record.selector in closure_selectors:
                raise ValueError(
                    f"compiled model contains duplicate closure contraction selector "
                    f"{record.selector!r}"
                )
            closure_selectors.add(record.selector)
            try:
                particle = particles[record.particle]
            except KeyError as exc:
                raise ValueError(
                    f"closure contraction refers to absent particle {exc.args[0]!r}"
                ) from exc
            dimension = compiled_current_dimension(
                particle,
                record.chirality,
                parameters=parameters,
                propagators=propagators,
            )
            contraction = record.contraction_ir
            if (
                dimension != 1
                or len(contraction.coefficients) != 1
                or contraction.name != "scalar"
                or contraction.left_basis != "scalar"
                or contraction.right_basis != "scalar"
                or contraction.metric_signature is not None
            ):
                raise ValueError(
                    f"closure contraction selector {record.selector!r} must be a "
                    "one-component scalar projection"
                )
            _validate_concrete_chirality_relation(
                contraction,
                record.chirality,
                record.chirality,
                context=f"closure contraction selector {record.selector!r}",
            )

    def _executable_expressions(self) -> tuple[tuple[str, str], ...]:
        """Return scalar/evaluator expressions, excluding raw tensor source."""

        result: list[tuple[str, str]] = []
        for parameter in self.parameters:
            if parameter.expression is not None:
                result.append(
                    (f"parameter {parameter.name} expression", parameter.expression)
                )
            result.append(
                (
                    f"parameter {parameter.name} resolved expression",
                    parameter.resolved_expression,
                )
            )
        for coupling in self.couplings:
            result.extend(
                (
                    (f"coupling {coupling.name} expression", coupling.expression),
                    (
                        f"coupling {coupling.name} resolved expression",
                        coupling.resolved_expression,
                    ),
                )
            )
        for term in self.vertex_terms:
            result.append((f"vertex term {term.id} coupling", term.coupling_expression))
        for kernel in self.oriented_kernels:
            result.append(
                (f"oriented kernel {kernel.kind} coupling", kernel.coupling_expression)
            )
            result.extend(
                (f"oriented kernel {kernel.kind} component {index}", expression)
                for index, expression in enumerate(kernel.component_expressions)
            )
        return tuple(result)

    @property
    def max_vertex_valence(self) -> int:
        return max((term.valence for term in self.vertex_terms), default=0)

    @property
    def symbol_namespace(self) -> str:
        return symbols.model(self.name).namespace

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "symbol_namespace": self.symbol_namespace,
            "orders": [item.to_dict() for item in self.orders],
            "parameters": [item.to_dict() for item in self.parameters],
            "particles": [item.to_dict() for item in self.particles],
            "couplings": [item.to_dict() for item in self.couplings],
            "propagators": [item.to_dict() for item in self.propagators],
            "vertex_terms": [item.to_dict() for item in self.vertex_terms],
            "oriented_kernels": [item.to_dict() for item in self.oriented_kernels],
            "direct_contractions": [
                item.to_dict() for item in self.direct_contractions
            ],
            "closure_contractions": [
                item.to_dict() for item in self.closure_contractions
            ],
            "max_vertex_valence": self.max_vertex_valence,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, object]) -> CompiledModelIR:
        name = str(payload["name"])
        expected_namespace = symbols.model(name).namespace
        if payload.get("symbol_namespace") != expected_namespace:
            raise ValueError(
                "compiled model symbol namespace mismatch; regenerate the model"
            )
        return CompiledModelIR(
            name=name,
            orders=tuple(
                CompiledCouplingOrder(
                    name=str(item["name"]),
                    expansion_order=_integer(item["expansion_order"]),
                    hierarchy=_integer(item["hierarchy"]),
                )
                for item in _mappings(payload.get("orders"))
            ),
            parameters=tuple(
                CompiledParameterRecord(
                    name=str(item["name"]),
                    nature=str(item["nature"]),
                    parameter_type=str(item["parameter_type"]),
                    value=_optional_pair(item.get("value")),
                    expression=_optional_string(item.get("expression")),
                    resolved_expression=str(item["resolved_expression"]),
                    lhablock=_optional_string(item.get("lhablock")),
                    lhacode=tuple(
                        _integer(value) for value in _sequence(item.get("lhacode"))
                    ),
                )
                for item in _mappings(payload.get("parameters"))
            ),
            particles=tuple(
                CompiledParticleRecord(
                    name=str(item["name"]),
                    antiname=str(item["antiname"]),
                    pdg_code=_integer(item["pdg_code"]),
                    spin=_integer(item["spin"]),
                    color=_integer(item["color"]),
                    mass=str(item["mass"]),
                    width=str(item["width"]),
                    charge=_floating(item["charge"]),
                    quantum_numbers=validate_quantum_number_flow(
                        item["quantum_numbers"],
                        context=f"particle {str(item['name'])!r}",
                    ),
                    ghost_number=_integer(item["ghost_number"]),
                    propagating=bool(item["propagating"]),
                    goldstoneboson=bool(item["goldstoneboson"]),
                    propagator=_optional_string(item.get("propagator")),
                    component_dimension=cast(
                        int | None, item.get("component_dimension")
                    ),
                    auxiliary_kind=_optional_string(item.get("auxiliary_kind")),
                    statistics=str(item.get("statistics", "")),
                    wavefunction_family=str(item.get("wavefunction_family", "")),
                    color_role=str(item.get("color_role", "")),
                    self_conjugate=(
                        None
                        if item.get("self_conjugate") is None
                        else bool(item["self_conjugate"])
                    ),
                    source_orientation=str(item.get("source_orientation", "")),
                )
                for item in _mappings(payload.get("particles"))
            ),
            couplings=tuple(
                CompiledCouplingRecord(
                    name=str(item["name"]),
                    expression=str(item["expression"]),
                    resolved_expression=str(item["resolved_expression"]),
                    value=_optional_pair(item.get("value")),
                    orders=_orders(item.get("orders")),
                )
                for item in _mappings(payload.get("couplings"))
            ),
            propagators=tuple(
                CompiledPropagatorRecord(
                    name=str(item["name"]),
                    particle=str(item["particle"]),
                    numerator=str(item["numerator"]),
                    denominator=str(item["denominator"]),
                    custom=bool(item["custom"]),
                )
                for item in _mappings(payload.get("propagators"))
            ),
            vertex_terms=tuple(
                CompiledVertexTerm(
                    id=_integer(item["id"]),
                    vertex=str(item["vertex"]),
                    particles=tuple(
                        str(value) for value in _sequence(item["particles"])
                    ),
                    color_index=_integer(item["color_index"]),
                    lorentz_index=_integer(item["lorentz_index"]),
                    color_source=str(item["color_source"]),
                    color_expression=str(item["color_expression"]),
                    lorentz_name=str(item["lorentz_name"]),
                    lorentz_source=str(item["lorentz_source"]),
                    lorentz_expression=str(item["lorentz_expression"]),
                    coupling=str(item["coupling"]),
                    coupling_expression=str(item["coupling_expression"]),
                    coupling_orders=_orders(item.get("coupling_orders")),
                    backend=str(item.get("backend", "ufo")),
                    lc_color_normalization_power=_integer(
                        item.get("lc_color_normalization_power", 0)
                    ),
                    contact_decomposition_proof=(
                        None
                        if item.get("contact_decomposition_proof") is None
                        else CompiledContactDecompositionProof.from_dict(
                            _strict_mapping(
                                item["contact_decomposition_proof"],
                                "contact decomposition proof",
                            )
                        )
                    ),
                )
                for item in _mappings(payload.get("vertex_terms"))
            ),
            oriented_kernels=tuple(
                CompiledOrientedKernel(
                    kind=_integer(item["kind"]),
                    term_id=_integer(item["term_id"]),
                    vertex=str(item["vertex"]),
                    particles=cast_tuple3(item["particles"]),
                    source_particle_legs=cast_int_tuple3(item["source_particle_legs"]),
                    component_expressions=tuple(
                        str(value) for value in _sequence(item["component_expressions"])
                    ),
                    coupling_expression=str(item["coupling_expression"]),
                    coupling_orders=_orders(item.get("coupling_orders")),
                    runtime_parameters=tuple(
                        str(value) for value in _sequence(item["runtime_parameters"])
                    ),
                    color_source=str(
                        item.get("color_source", item["color_expression"])
                    ),
                    color_expression=str(item["color_expression"]),
                    color_projection_structure=_optional_string(
                        item.get("color_projection_structure")
                    ),
                    color_projection_coefficient=(
                        None
                        if item.get("color_projection_coefficient") is None
                        else _pair(item.get("color_projection_coefficient"))
                    ),
                    lc_color_normalization_power=_integer(
                        item.get("lc_color_normalization_power", 0)
                    ),
                    term_ids=tuple(
                        _integer(value)
                        for value in _sequence(
                            item.get("term_ids", [_integer(item["term_id"])])
                        )
                    ),
                    evaluation_class=str(
                        item.get(
                            "evaluation_class",
                            f"unverified-kernel-{_integer(item['kind'])}",
                        )
                    ),
                    evaluation_factor=_pair(item.get("evaluation_factor", (1.0, 0.0))),
                    evaluation_input_order=cast_int_tuple2(
                        item.get("evaluation_input_order", (0, 1))
                    ),
                    evaluation_equivalence_verified=bool(
                        item.get("evaluation_equivalence_verified", False)
                    ),
                )
                for item in _mappings(payload.get("oriented_kernels"))
            ),
            direct_contractions=tuple(
                CompiledDirectContractionRecord.from_dict(item)
                for item in _required_mappings(payload, "direct_contractions")
            ),
            closure_contractions=tuple(
                CompiledClosureContractionRecord.from_dict(item)
                for item in _required_mappings(payload, "closure_contractions")
            ),
        )


def compiled_particle_is_chiral_eligible(
    particle: CompiledParticleRecord,
    *,
    parameters: Mapping[str, CompiledParameterRecord],
    propagators: Mapping[str, CompiledPropagatorRecord],
) -> bool:
    """Return whether compilation proved a two-component Weyl current valid."""

    if (
        particle.statistics != "fermion"
        or particle.wavefunction_family != "fermion"
        or particle.self_conjugate
        or not particle.propagating
    ):
        return False
    if particle.component_dimension not in {None, 4}:
        return False
    if particle.propagator is not None:
        propagator = propagators.get(particle.propagator)
        if propagator is None or propagator.custom:
            return False
    if particle.mass.upper() == "ZERO":
        return True
    parameter = parameters.get(particle.mass)
    if parameter is None or parameter.nature.lower() == "external":
        return False
    from . import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    return (
        _sym.E(parameter.resolved_expression).expand().to_canonical_string()
        == _sym.E("0").to_canonical_string()
    )


def compiled_current_dimension(
    particle: CompiledParticleRecord,
    chirality: int,
    *,
    parameters: Mapping[str, CompiledParameterRecord],
    propagators: Mapping[str, CompiledPropagatorRecord],
) -> int:
    """Resolve one concrete compiled current state without model heuristics."""

    value = _validate_chirality(chirality, "chirality")
    if value != 0:
        if value not in {-1, 1}:
            raise ValueError(f"unsupported concrete chirality {value}")
        if not compiled_particle_is_chiral_eligible(
            particle,
            parameters=parameters,
            propagators=propagators,
        ):
            raise ValueError(
                f"particle {particle.name!r} does not support projected Weyl currents"
            )
        return 2
    if particle.component_dimension is not None:
        dimension = int(particle.component_dimension)
    else:
        try:
            dimension = {-1: 1, 1: 1, 2: 4, 3: 4, 5: 16}[particle.spin]
        except KeyError as exc:
            raise ValueError(
                f"particle {particle.name!r} has unsupported UFO spin code "
                f"{particle.spin}"
            ) from exc
    if dimension <= 0:
        raise ValueError(
            f"particle {particle.name!r} has invalid component dimension {dimension}"
        )
    return dimension


def _validate_concrete_chirality_relation(
    contraction: ContractionIR,
    left_chirality: int,
    right_chirality: int,
    *,
    context: str,
) -> None:
    relation = contraction.chirality_relation
    if relation == "equal" and left_chirality != right_chirality:
        raise ValueError(f"{context} violates equal chirality relation")
    if relation == "opposite" and (
        left_chirality == 0
        or right_chirality == 0
        or left_chirality != -right_chirality
    ):
        raise ValueError(f"{context} violates opposite chirality relation")


def _validate_particle_selector_name(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"compiled contraction {context} must be a string")
    if not value:
        raise ValueError(f"compiled contraction {context} must not be empty")
    return value


def _validate_chirality(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"compiled contraction {context} must be an integer")
    return value


def _strict_record_fields(
    payload: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping")
    fields = set(payload)
    missing = required - fields
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{context} is missing required fields: {names}")
    unknown = fields - required
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{context} has unknown fields: {names}")
    return payload


def _strict_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _strict_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _strict_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _required_mappings(
    payload: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    if field not in payload:
        raise ValueError(f"compiled model is missing required field {field!r}")
    value = payload[field]
    if isinstance(value, (str, bytes)) or not isinstance(value, list | tuple):
        raise TypeError(f"compiled model field {field!r} must be an array")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"compiled model field {field!r} item {index} must be a mapping"
            )
        result.append(item)
    return tuple(result)


def cast_tuple3(value: object) -> tuple[str, str, str]:
    values = tuple(str(item) for item in _sequence(value))
    if len(values) != 3:
        raise ValueError("oriented kernel particles must have length three")
    return values[0], values[1], values[2]


def cast_int_tuple3(value: object) -> tuple[int, int, int]:
    values = tuple(_integer(item) for item in _sequence(value))
    if len(values) != 3:
        raise ValueError("oriented kernel source legs must have length three")
    return values[0], values[1], values[2]


def cast_int_tuple2(value: object) -> tuple[int, int]:
    values = tuple(_integer(item) for item in _sequence(value))
    if len(values) != 2:
        raise ValueError("evaluation input order must have length two")
    return values[0], values[1]


def _orders(value: object) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for pair in _sequence(value):
        values = _sequence(pair)
        if len(values) != 2:
            raise ValueError("coupling order must be [name, value]")
        result.append((str(values[0]), _integer(values[1])))
    return tuple(result)


def _pair(value: object) -> tuple[float, float]:
    pair = _sequence(value)
    if len(pair) != 2:
        raise ValueError("complex value must be [real, imaginary]")
    return _floating(pair[0]), _floating(pair[1])


def _optional_pair(value: object) -> tuple[float, float] | None:
    return None if value is None else _pair(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _integer(value: object) -> int:
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected an integer-compatible value, got {value!r}")
    return int(value)


def _floating(value: object) -> float:
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected a numeric value, got {value!r}")
    return float(value)


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


def _mappings(value: object) -> list[dict[str, object]]:
    return [dict(item) for item in _sequence(value) if isinstance(item, Mapping)]


__all__ = [
    "SUPPORTED_COLOR_REPRESENTATIONS",
    "CompiledClosureContractionRecord",
    "CompiledContactDecompositionProof",
    "CompiledContactDecompositionSplit",
    "CompiledContactDummyIndexMapping",
    "CompiledContactOrientationProof",
    "CompiledContactUnsupportedReason",
    "CompiledCouplingOrder",
    "CompiledCouplingRecord",
    "CompiledDirectContractionRecord",
    "CompiledModelIR",
    "CompiledOrientedKernel",
    "CompiledParameterRecord",
    "CompiledParticleRecord",
    "CompiledPropagatorRecord",
    "CompiledVertexTerm",
    "compiled_current_dimension",
    "compiled_particle_is_chiral_eligible",
    "validate_color_representation",
    "validate_quantum_number_flow",
]
