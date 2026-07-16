# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from pyamplicol.config import (
    BenchmarkConfig,
    ConfigClamp,
    GenerationConfig,
    RunConfig,
)

from .requests import ProcessRequest, ProcessSet


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(entry) for key, entry in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(entry) for entry in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class ExternalParticle:
    index: int
    label: int
    name: str
    pdg_id: int
    state: Literal["incoming", "outgoing"]
    momentum_slot: int

    def __post_init__(self) -> None:
        for name in ("index", "label", "momentum_slot"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"external particle {name} must be non-negative")
        if not self.name:
            raise ValueError("external particle name must not be empty")
        if isinstance(self.pdg_id, bool) or not isinstance(self.pdg_id, int):
            raise ValueError("external particle pdg_id must be an integer")
        if self.state not in ("incoming", "outgoing"):
            raise ValueError("external particle state must be 'incoming' or 'outgoing'")


@dataclass(frozen=True, slots=True)
class HelicityConfiguration:
    id: str
    index: int
    values: tuple[int, ...]
    computed: bool
    structural_zero: bool
    representative_id: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.id or not self.representative_id:
            raise ValueError("helicity IDs must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("helicity index must be non-negative")
        values = tuple(self.values)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise ValueError("helicity values must be integers")
        if not math.isfinite(self.coefficient):
            raise ValueError("helicity coefficient must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ColorFlow:
    id: str
    index: int
    word: tuple[int, ...]
    computed: bool
    representative_id: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.id or not self.representative_id:
            raise ValueError("color-flow IDs must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("color-flow index must be non-negative")
        word = tuple(self.word)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in word
        ):
            raise ValueError("color-flow words must contain non-negative integers")
        if not math.isfinite(self.coefficient):
            raise ValueError("color-flow coefficient must be finite")
        object.__setattr__(self, "word", word)


@dataclass(frozen=True, slots=True)
class ContractedColorComponent:
    id: str
    index: int
    description: str

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("contracted-color metadata must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("contracted-color index must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelParameter:
    name: str
    kind: str
    default_real: float
    default_imaginary: float
    mutable: bool

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("model parameter name and kind must not be empty")
        if not math.isfinite(self.default_real) or not math.isfinite(
            self.default_imaginary
        ):
            raise ValueError("model parameter defaults must be finite")


ColorComponent: TypeAlias = ColorFlow | ContractedColorComponent


@dataclass(frozen=True, slots=True)
class ReductionGroup:
    id: str
    representative_helicity_id: str
    representative_color_id: str
    physical_helicity_ids: tuple[str, ...]
    physical_color_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.representative_helicity_id:
            raise ValueError("reduction group and representative IDs must not be empty")
        if not self.representative_color_id:
            raise ValueError("reduction color representative ID must not be empty")
        helicities = tuple(self.physical_helicity_ids)
        colors = tuple(self.physical_color_ids)
        if not helicities or not colors:
            raise ValueError("reduction groups require physical helicity and color IDs")
        if len(set(helicities)) != len(helicities) or len(set(colors)) != len(colors):
            raise ValueError("reduction group member IDs must be unique")
        if self.representative_helicity_id not in helicities:
            raise ValueError("helicity representative must be a physical group member")
        if self.representative_color_id not in colors:
            raise ValueError("color representative must be a physical group member")
        object.__setattr__(self, "physical_helicity_ids", helicities)
        object.__setattr__(self, "physical_color_ids", colors)


@dataclass(frozen=True, slots=True)
class PhysicsReduction:
    kind: Literal["lc-diagonal", "contracted-color"]
    groups: tuple[ReductionGroup, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("lc-diagonal", "contracted-color"):
            raise ValueError("unsupported physics reduction kind")
        groups = tuple(self.groups)
        if not all(isinstance(group, ReductionGroup) for group in groups):
            raise TypeError("physics reduction contains invalid groups")
        if len({group.id for group in groups}) != len(groups):
            raise ValueError("physics reduction group IDs must be unique")
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    concrete_processes: tuple[ProcessRequest, ...]
    estimated_coverage: Mapping[str, object]
    requested_settings: RunConfig | GenerationConfig
    effective_settings: RunConfig | GenerationConfig
    adjustments: tuple[ConfigClamp, ...] = ()
    unsupported_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        concrete_processes = tuple(self.concrete_processes)
        if not all(
            isinstance(process, ProcessRequest) for process in concrete_processes
        ):
            raise TypeError("concrete_processes must contain ProcessRequest objects")
        for name in ("requested_settings", "effective_settings"):
            if not isinstance(getattr(self, name), (RunConfig, GenerationConfig)):
                raise TypeError(f"{name} must be RunConfig or GenerationConfig")
        adjustments = tuple(self.adjustments)
        if not all(isinstance(adjustment, ConfigClamp) for adjustment in adjustments):
            raise TypeError("adjustments must contain ConfigClamp objects")
        unsupported_features = tuple(self.unsupported_features)
        if not all(
            isinstance(feature, str) and feature for feature in unsupported_features
        ):
            raise ValueError("unsupported_features must contain non-empty strings")
        object.__setattr__(self, "concrete_processes", concrete_processes)
        object.__setattr__(
            self, "estimated_coverage", _freeze_mapping(self.estimated_coverage)
        )
        object.__setattr__(self, "adjustments", adjustments)
        object.__setattr__(self, "unsupported_features", unsupported_features)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    output: Path
    processes: ProcessSet
    mode: Literal["error", "append", "replace"]
    schema_version: int = 3
    files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        try:
            output = Path(os.fspath(self.output)).expanduser().resolve(strict=False)
            files = tuple(
                Path(os.fspath(path)).expanduser().resolve(strict=False)
                for path in self.files
            )
        except TypeError as exc:
            raise ValueError("generation result paths must be path-like") from exc
        object.__setattr__(self, "output", output)
        if self.mode not in ("error", "append", "replace"):
            raise ValueError(f"invalid generation mode {self.mode!r}")
        if self.schema_version != 3:
            raise ValueError("generated artifacts must use schema version 3")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ProcessPhysics:
    process_id: str
    process: str
    color_accuracy: Literal["lc", "nlc", "full"]
    helicity_coverage: str
    color_coverage: str
    color_kind: str
    structural_zero_helicity_count: int
    external_particles: tuple[ExternalParticle, ...]
    helicities: tuple[HelicityConfiguration, ...]
    color_flows: tuple[ColorFlow, ...]
    contracted_color_components: tuple[ContractedColorComponent, ...]
    reduction: PhysicsReduction
    model_parameters: tuple[ModelParameter, ...]
    selector_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.process_id or not self.process:
            raise ValueError("process ID and expression must not be empty")
        if self.color_accuracy not in ("lc", "nlc", "full"):
            raise ValueError("color accuracy must be 'lc', 'nlc', or 'full'")
        for name in ("helicity_coverage", "color_coverage", "color_kind"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if (
            isinstance(self.structural_zero_helicity_count, bool)
            or self.structural_zero_helicity_count < 0
        ):
            raise ValueError("structural-zero helicity count must be non-negative")
        particles = tuple(self.external_particles)
        helicities = tuple(self.helicities)
        color_flows = tuple(self.color_flows)
        contracted = tuple(self.contracted_color_components)
        reduction = self.reduction
        parameters = tuple(self.model_parameters)
        selector_capabilities = tuple(self.selector_capabilities)
        expected = (
            ("external_particles", particles, ExternalParticle),
            ("helicities", helicities, HelicityConfiguration),
            ("color_flows", color_flows, ColorFlow),
            ("contracted_color_components", contracted, ContractedColorComponent),
            ("model_parameters", parameters, ModelParameter),
        )
        for name, values, expected_type in expected:
            if not all(isinstance(value, expected_type) for value in values):
                raise TypeError(f"{name} contains invalid metadata")
        if not helicities:
            raise ValueError("physics metadata requires physical helicities")
        particle_indices = tuple(particle.index for particle in particles)
        if len(set(particle_indices)) != len(particles):
            raise ValueError("external particle indices must be unique")
        for name, identifiers in (
            ("helicities", tuple(value.id for value in helicities)),
            ("color_flows", tuple(value.id for value in color_flows)),
            (
                "contracted_color_components",
                tuple(value.id for value in contracted),
            ),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{name} IDs must be unique")
        parameter_names = tuple(parameter.name for parameter in parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("model parameter names must be unique")
        if self.color_accuracy == "lc" and not color_flows:
            raise ValueError("LC physics metadata requires physical color flows")
        if self.color_accuracy != "lc" and len(contracted) != 1:
            raise ValueError("NLC/full physics metadata requires one color component")
        if not isinstance(reduction, PhysicsReduction):
            raise TypeError("reduction must be PhysicsReduction metadata")
        expected_reduction = (
            "lc-diagonal" if self.color_accuracy == "lc" else "contracted-color"
        )
        if reduction.kind != expected_reduction:
            raise ValueError("reduction kind does not match color accuracy")
        physical_helicity_ids = {value.id for value in helicities}
        physical_color_ids = {value.id for value in color_flows} | {
            value.id for value in contracted
        }
        for group in reduction.groups:
            if not set(group.physical_helicity_ids) <= physical_helicity_ids:
                raise ValueError("reduction references unknown physical helicities")
            if not set(group.physical_color_ids) <= physical_color_ids:
                raise ValueError("reduction references unknown physical colors")
        if not all(isinstance(value, str) and value for value in selector_capabilities):
            raise ValueError("selector_capabilities must contain non-empty strings")
        if len(set(selector_capabilities)) != len(selector_capabilities):
            raise ValueError("selector_capabilities must be unique")
        object.__setattr__(self, "external_particles", particles)
        object.__setattr__(self, "helicities", helicities)
        object.__setattr__(self, "color_flows", color_flows)
        object.__setattr__(self, "contracted_color_components", contracted)
        object.__setattr__(self, "model_parameters", parameters)
        object.__setattr__(self, "selector_capabilities", selector_capabilities)

    @property
    def helicity_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.helicities)

    @property
    def color_flow_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.color_flows)

    @property
    def color_ids(self) -> tuple[str, ...]:
        components: tuple[ColorComponent, ...] = (
            self.color_flows
            if self.color_accuracy == "lc"
            else self.contracted_color_components
        )
        return tuple(item.id for item in components)


@dataclass(frozen=True, slots=True)
class ResolvedEvaluation:
    values: tuple[tuple[tuple[complex | Decimal, ...], ...], ...]
    helicity_ids: tuple[str, ...]
    color_ids: tuple[str, ...]
    accuracy: Literal["lc", "nlc", "full"] = "lc"

    def __post_init__(self) -> None:
        values = tuple(
            tuple(
                tuple(
                    entry if isinstance(entry, Decimal) else complex(entry)
                    for entry in colors
                )
                for colors in helicities
            )
            for helicities in self.values
        )
        helicity_ids = tuple(self.helicity_ids)
        color_ids = tuple(self.color_ids)
        expected_colors = len(color_ids)
        if self.accuracy != "lc" and expected_colors != 1:
            raise ValueError("NLC/full resolved output has one contracted color axis")
        for point in values:
            if len(point) != len(helicity_ids):
                raise ValueError(
                    "resolved values do not match the physical helicity dimension"
                )
            if any(len(colors) != expected_colors for colors in point):
                raise ValueError(
                    "resolved values do not match the contracted color dimension"
                )
        if self.accuracy not in ("lc", "nlc", "full"):
            raise ValueError("accuracy must be 'lc', 'nlc', or 'full'")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "helicity_ids", helicity_ids)
        object.__setattr__(self, "color_ids", color_ids)

    @property
    def color_flow_ids(self) -> tuple[str, ...]:
        return self.color_ids if self.accuracy == "lc" else ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.values), len(self.helicity_ids), len(self.color_ids))

    def total(self) -> tuple[complex | Decimal, ...]:
        totals: list[complex | Decimal] = []
        for point in self.values:
            entries = tuple(entry for helicity in point for entry in helicity)
            if entries and isinstance(entries[0], Decimal):
                if not all(isinstance(entry, Decimal) for entry in entries):
                    raise TypeError("resolved values must not mix Decimal and complex")
                totals.append(
                    sum(
                        (cast(Decimal, entry) for entry in entries),
                        start=Decimal(0),
                    )
                )
            else:
                totals.append(sum((complex(entry) for entry in entries), start=0j))
        return tuple(totals)


@dataclass(frozen=True, slots=True)
class BenchmarkStatistics:
    standard_deviation: float
    standard_error: float
    relative_standard_error: float

    def __post_init__(self) -> None:
        for name in (
            "standard_deviation",
            "standard_error",
            "relative_standard_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"benchmark {name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    requested_config: BenchmarkConfig
    effective_config: BenchmarkConfig
    sample_count: int
    wall_time_per_point: float
    evaluator_time_per_point: float | None
    uncertainty: BenchmarkStatistics
    environment: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("benchmark sample_count must be positive")
        if not math.isfinite(self.wall_time_per_point) or self.wall_time_per_point < 0:
            raise ValueError("benchmark wall_time_per_point must be non-negative")
        if self.evaluator_time_per_point is not None and (
            not math.isfinite(self.evaluator_time_per_point)
            or self.evaluator_time_per_point < 0
        ):
            raise ValueError("benchmark evaluator_time_per_point must be non-negative")
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))


__all__ = [
    "BenchmarkResult",
    "BenchmarkStatistics",
    "ColorComponent",
    "ColorFlow",
    "ContractedColorComponent",
    "ExternalParticle",
    "GenerationPlan",
    "GenerationResult",
    "HelicityConfiguration",
    "ModelParameter",
    "PhysicsReduction",
    "ProcessPhysics",
    "ReductionGroup",
    "ResolvedEvaluation",
]
