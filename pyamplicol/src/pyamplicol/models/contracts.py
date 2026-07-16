# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._internal.physics.symbols import symbols
from .base import QuantumNumberFlow

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

    @property
    def valence(self) -> int:
        return len(self.particles)

    def to_dict(self) -> dict[str, object]:
        return {
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
class CompiledModelIR:
    name: str
    orders: tuple[CompiledCouplingOrder, ...]
    parameters: tuple[CompiledParameterRecord, ...]
    particles: tuple[CompiledParticleRecord, ...]
    couplings: tuple[CompiledCouplingRecord, ...]
    propagators: tuple[CompiledPropagatorRecord, ...]
    vertex_terms: tuple[CompiledVertexTerm, ...]
    oriented_kernels: tuple[CompiledOrientedKernel, ...]

    def __post_init__(self) -> None:
        self._validate_particle_identities()
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
                    component_dimension=(
                        None
                        if item.get("component_dimension") is None
                        else _integer(item["component_dimension"])
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
        )


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
    "CompiledCouplingOrder",
    "CompiledCouplingRecord",
    "CompiledModelIR",
    "CompiledOrientedKernel",
    "CompiledParameterRecord",
    "CompiledParticleRecord",
    "CompiledPropagatorRecord",
    "CompiledVertexTerm",
    "validate_color_representation",
    "validate_quantum_number_flow",
]
