# SPDX-License-Identifier: 0BSD
"""Canonical, model-owned physics contracts used by generation and runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

ParticleOrientation = Literal["particle", "antiparticle", "self-conjugate"]
ParticleStatistics = Literal["boson", "fermion", "ghost", "auxiliary"]
WavefunctionFamily = Literal[
    "scalar",
    "fermion",
    "vector",
    "spin2",
    "ghost",
    "auxiliary",
]
MomentumTransform = Literal["identity", "negate-four-momentum"]


@dataclass(frozen=True, slots=True)
class ParticleIdentityIR:
    """Identity of one oriented model state, independent of SM taxonomy."""

    canonical_id: str
    species_id: str
    anti_canonical_id: str
    display_name: str
    anti_display_name: str
    pdg_label: int
    anti_pdg_label: int
    orientation: ParticleOrientation
    self_conjugate: bool

    def __post_init__(self) -> None:
        for field_name in (
            "canonical_id",
            "species_id",
            "anti_canonical_id",
            "display_name",
            "anti_display_name",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"particle identity {field_name} must not be empty")
        for field_name in ("pdg_label", "anti_pdg_label"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"particle identity {field_name} must be an integer")
        if self.orientation not in {
            "particle",
            "antiparticle",
            "self-conjugate",
        }:
            raise ValueError(f"invalid particle orientation {self.orientation!r}")
        if self.self_conjugate != (self.canonical_id == self.anti_canonical_id):
            raise ValueError(
                "particle self-conjugacy must agree with its canonical anti relation"
            )
        if self.self_conjugate != (self.pdg_label == self.anti_pdg_label):
            raise ValueError(
                "particle self-conjugacy must agree with its PDG-label anti relation"
            )
        if (self.orientation == "self-conjugate") != self.self_conjugate:
            raise ValueError(
                "self-conjugate source orientation must agree with particle identity"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "species_id": self.species_id,
            "anti_canonical_id": self.anti_canonical_id,
            "display_name": self.display_name,
            "anti_display_name": self.anti_display_name,
            "pdg_label": self.pdg_label,
            "anti_pdg_label": self.anti_pdg_label,
            "orientation": self.orientation,
            "self_conjugate": self.self_conjugate,
        }


@dataclass(frozen=True, slots=True)
class SourceStateIR:
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("helicity", "chirality"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"source state {field_name} must be an integer")
        if isinstance(self.spin_state, bool) or not isinstance(
            self.spin_state, (int, tuple)
        ):
            raise TypeError("source spin state must be an integer or integer tuple")
        if isinstance(self.spin_state, tuple) and not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in self.spin_state
        ):
            raise TypeError("source spin-state tuples must contain only integers")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "helicity": self.helicity,
            "chirality": self.chirality,
            "spin_state": (
                list(self.spin_state)
                if isinstance(self.spin_state, tuple)
                else self.spin_state
            ),
        }


@dataclass(frozen=True, slots=True)
class CrossingIR:
    """Transform an outgoing source basis into an incoming physical leg."""

    momentum_transform: MomentumTransform = "negate-four-momentum"
    helicity_factor: int = 1
    chirality_factor: int = 1
    spin_state_factor: int = 1
    phase: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.momentum_transform not in {"identity", "negate-four-momentum"}:
            raise ValueError(
                f"unsupported crossing momentum transform {self.momentum_transform!r}"
            )
        for field_name in (
            "helicity_factor",
            "chirality_factor",
            "spin_state_factor",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"crossing {field_name} must be an integer")
            if value not in {-1, 1}:
                raise ValueError(f"crossing {field_name} must be -1 or 1")
        if not isinstance(self.phase, (tuple, list)) or len(self.phase) != 2:
            raise ValueError("crossing phase must be a finite complex pair")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.phase
        ):
            raise TypeError("crossing phase components must be real numbers")
        phase = (float(self.phase[0]), float(self.phase[1]))
        if not all(math.isfinite(value) for value in phase):
            raise ValueError("crossing phase must be a finite complex pair")
        if phase == (0.0, 0.0):
            raise ValueError("crossing phase must be nonzero")
        object.__setattr__(self, "phase", phase)

    @classmethod
    def identity(cls) -> CrossingIR:
        return cls(momentum_transform="identity")

    def apply(self, state: SourceStateIR) -> SourceStateIR:
        spin_state = state.spin_state
        if self.spin_state_factor != 1:
            if not isinstance(spin_state, int):
                raise TypeError(
                    "crossing cannot multiply a structured source spin state"
                )
            spin_state *= self.spin_state_factor
        return SourceStateIR(
            helicity=state.helicity * self.helicity_factor,
            chirality=state.chirality * self.chirality_factor,
            spin_state=spin_state,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "momentum_transform": self.momentum_transform,
            "helicity_factor": self.helicity_factor,
            "chirality_factor": self.chirality_factor,
            "spin_state_factor": self.spin_state_factor,
            "phase": list(self.phase),
        }


@dataclass(frozen=True, slots=True)
class SourceIR:
    identity: ParticleIdentityIR
    statistics: ParticleStatistics
    wavefunction_family: WavefunctionFamily
    component_dimension: int
    states: tuple[SourceStateIR, ...]
    crossing: CrossingIR
    basis: str
    mass_parameter: str | None = None
    width_parameter: str | None = None

    def __post_init__(self) -> None:
        if self.statistics not in {"boson", "fermion", "ghost", "auxiliary"}:
            raise ValueError(f"invalid source statistics {self.statistics!r}")
        if self.wavefunction_family not in {
            "scalar",
            "fermion",
            "vector",
            "spin2",
            "ghost",
            "auxiliary",
        }:
            raise ValueError(
                f"invalid source wavefunction family {self.wavefunction_family!r}"
            )
        expected_statistics: ParticleStatistics
        if self.wavefunction_family == "fermion":
            expected_statistics = "fermion"
        elif self.wavefunction_family == "ghost":
            expected_statistics = "ghost"
        elif self.wavefunction_family == "auxiliary":
            expected_statistics = "auxiliary"
        else:
            expected_statistics = "boson"
        if self.statistics != expected_statistics:
            raise ValueError(
                f"source wavefunction family {self.wavefunction_family!r} requires "
                f"statistics {expected_statistics!r}, got {self.statistics!r}"
            )
        if isinstance(self.component_dimension, bool) or not isinstance(
            self.component_dimension, int
        ):
            raise TypeError("source component dimension must be an integer")
        if self.component_dimension < 1:
            raise ValueError("source component dimension must be positive")
        if not self.states:
            raise ValueError("source metadata must declare at least one spin state")
        if not self.basis:
            raise ValueError("source basis must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_json_dict(),
            "statistics": self.statistics,
            "wavefunction_family": self.wavefunction_family,
            "component_dimension": self.component_dimension,
            "states": [state.to_json_dict() for state in self.states],
            "crossing": self.crossing.to_json_dict(),
            "basis": self.basis,
            "mass_parameter": self.mass_parameter,
            "width_parameter": self.width_parameter,
        }


@dataclass(frozen=True, slots=True)
class PropagatorIR:
    identity: ParticleIdentityIR
    chirality: int
    backend: str
    basis: str
    applies_propagator: bool
    kernel: str
    full_tensor_network_ready: bool
    gauge: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    mass_parameter: str | None = None
    width_parameter: str | None = None
    custom_source: str | None = None
    auxiliary_policy: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.backend or not self.basis or not self.kernel:
            raise ValueError("propagator backend, basis, and kernel must not be empty")
        if not self.applies_propagator and self.auxiliary_policy is None:
            raise ValueError(
                "a no-propagator current must declare its auxiliary policy"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_json_dict(),
            "particle_id": self.identity.pdg_label,
            "chirality": self.chirality,
            "backend": self.backend,
            "basis": self.basis,
            "applies_propagator": self.applies_propagator,
            "kernel": self.kernel,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "gauge": self.gauge,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "mass_parameter": self.mass_parameter,
            "width_parameter": self.width_parameter,
            "custom_source": self.custom_source,
            "auxiliary_policy": self.auxiliary_policy,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ContractionIR:
    name: str
    left_basis: str
    right_basis: str
    coefficients: tuple[tuple[float, float], ...]
    chirality_relation: Literal["any", "equal", "opposite"] = "any"
    metric_signature: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.left_basis or not self.right_basis:
            raise ValueError("contraction name and bases must not be empty")
        if not self.coefficients:
            raise ValueError("contraction must declare component coefficients")
        if not all(
            len(value) == 2 and all(math.isfinite(component) for component in value)
            for value in self.coefficients
        ):
            raise ValueError("contraction coefficients must be finite complex pairs")
        if self.chirality_relation not in {"any", "equal", "opposite"}:
            raise ValueError(
                f"invalid contraction chirality relation {self.chirality_relation!r}"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "left_basis": self.left_basis,
            "right_basis": self.right_basis,
            "coefficients": [list(value) for value in self.coefficients],
            "chirality_relation": self.chirality_relation,
            "metric_signature": self.metric_signature,
        }
