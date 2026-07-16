# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    DependencyError,
    EvaluationError,
)
from pyamplicol.api.protocols import ModelParameters, Momenta
from pyamplicol.api.results import (
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    ModelParameter,
    PhysicsReduction,
    ProcessPhysics,
    ReductionGroup,
    ResolvedEvaluation,
)
from pyamplicol.artifacts import load_manifest

if TYPE_CHECKING:
    from .symbolica_exact import SymbolicaExactExecutor

_Accuracy = Literal["lc", "nlc", "full"]
_ParticleState = Literal["incoming", "outgoing"]


def _manifest_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{description} must be an integer")
    return value


def _load_native_module() -> Any:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
    except ImportError as exc:
        raise DependencyError(
            "the pyamplicol._rusticol native runtime extension is unavailable"
        ) from exc
    if not hasattr(module, "Runtime"):
        raise DependencyError(
            "the pyamplicol._rusticol extension does not provide Runtime"
        )
    return module


def _translated_error(module: Any, error: Exception) -> Exception | None:
    mappings = (
        ("CompatibilityError", CompatibilityError),
        ("ArtifactError", ArtifactError),
        ("SelectorError", EvaluationError),
        ("ModelParameterError", EvaluationError),
        ("EvaluationError", EvaluationError),
        ("RusticolError", EvaluationError),
    )
    for native_name, public_type in mappings:
        native_type = getattr(module, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            return public_type(str(error))
    return None


def _invoke(module: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        translated = _translated_error(module, exc)
        if translated is None:
            raise
        raise translated from exc


def _physics_from_native(value: Any) -> ProcessPhysics:
    return ProcessPhysics(
        process_id=str(value.process_id),
        process=str(value.process),
        color_accuracy=cast(_Accuracy, str(value.color_accuracy)),
        helicity_coverage=str(value.helicity_coverage),
        color_coverage=str(value.color_coverage),
        color_kind=str(value.color_kind),
        structural_zero_helicity_count=int(value.structural_zero_helicity_count),
        external_particles=tuple(
            ExternalParticle(
                index=int(particle.index),
                label=int(particle.label),
                name=str(particle.name),
                pdg_id=int(particle.pdg_id),
                state=cast(_ParticleState, str(particle.state)),
                momentum_slot=int(particle.momentum_slot),
            )
            for particle in value.external_particles
        ),
        helicities=tuple(
            HelicityConfiguration(
                id=str(helicity.id),
                index=int(helicity.index),
                values=tuple(int(entry) for entry in helicity.values),
                computed=bool(helicity.computed),
                structural_zero=bool(helicity.structural_zero),
                representative_id=str(helicity.representative_id),
                coefficient=float(helicity.coefficient),
            )
            for helicity in value.helicities
        ),
        color_flows=tuple(
            ColorFlow(
                id=str(flow.id),
                index=int(flow.index),
                word=tuple(int(entry) for entry in flow.word),
                computed=bool(flow.computed),
                representative_id=str(flow.representative_id),
                coefficient=float(flow.coefficient),
            )
            for flow in value.color_flows
        ),
        contracted_color_components=tuple(
            ContractedColorComponent(
                id=str(component.id),
                index=int(component.index),
                description=str(component.description),
            )
            for component in value.contracted_color_components
        ),
        reduction=PhysicsReduction(
            kind=cast(
                Literal["lc-diagonal", "contracted-color"],
                str(value.reduction.kind),
            ),
            groups=tuple(
                ReductionGroup(
                    id=str(group.id),
                    representative_helicity_id=str(group.representative_helicity_id),
                    representative_color_id=str(group.representative_color_id),
                    physical_helicity_ids=tuple(
                        str(identifier) for identifier in group.physical_helicity_ids
                    ),
                    physical_color_ids=tuple(
                        str(identifier) for identifier in group.physical_color_ids
                    ),
                )
                for group in value.reduction.groups
            ),
        ),
        model_parameters=tuple(
            ModelParameter(
                name=str(parameter.name),
                kind=str(parameter.kind),
                default_real=float(parameter.default_real),
                default_imaginary=float(parameter.default_imaginary),
                mutable=bool(parameter.mutable),
            )
            for parameter in value.model_parameters
        ),
        selector_capabilities=tuple(
            str(capability) for capability in value.selector_capabilities
        ),
    )


def _scalar_from_native(value: Any) -> complex | Decimal:
    return value if isinstance(value, Decimal) else complex(value)


class RusticolRuntimeBackend:
    """Public-protocol adapter around ``pyamplicol._rusticol.Runtime``."""

    def __init__(self, runtime: Any, native_module: Any, artifact_path: Path) -> None:
        self._runtime = runtime
        self._native_module = native_module
        self._artifact_path = artifact_path
        self._exact_executor: SymbolicaExactExecutor | None = None

    @property
    def physics(self) -> ProcessPhysics:
        return _physics_from_native(self._runtime.physics)

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> tuple[complex | Decimal, ...]:
        if precision != 16:
            return (
                self._exact()
                .evaluate_resolved(
                    momenta,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=precision,
                )
                .total()
            )
        values = _invoke(
            self._native_module,
            self._runtime.evaluate,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )
        return tuple(_scalar_from_native(value) for value in values)

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> ResolvedEvaluation:
        if precision != 16:
            return self._exact().evaluate_resolved(
                momenta,
                helicities=helicities,
                color_flows=color_flows,
                precision=precision,
            )
        native = _invoke(
            self._native_module,
            self._runtime.evaluate_resolved,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )
        values = tuple(
            tuple(
                tuple(_scalar_from_native(entry) for entry in colors)
                for colors in helicities_at_point
            )
            for helicities_at_point in native.values
        )
        return ResolvedEvaluation(
            values=values,
            helicity_ids=tuple(str(value) for value in native.helicity_ids),
            color_ids=tuple(str(value) for value in native.color_ids),
            color_accuracy=cast(_Accuracy, str(native.color_accuracy)),
        )

    def _exact(self) -> SymbolicaExactExecutor:
        if self._exact_executor is None:
            from .symbolica_exact import SymbolicaExactExecutor

            self._exact_executor = SymbolicaExactExecutor(
                self._artifact_path,
                self.physics.process_id,
                self._runtime,
            )
        return self._exact_executor

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        _invoke(
            self._native_module,
            self._runtime.set_model_parameters,
            dict(mapping),
        )

    def mute_warnings(self) -> None:
        _invoke(self._native_module, self._runtime.mute_warnings)

    def unmute_warnings(self) -> None:
        _invoke(self._native_module, self._runtime.unmute_warnings)

    def take_warnings(self) -> tuple[str, ...]:
        values = _invoke(self._native_module, self._runtime.take_warnings)
        return tuple(str(value) for value in values)

    def validation_momenta(self) -> Momenta | None:
        """Return the selected process's verified deterministic artifact point."""

        manifest = load_manifest(self._artifact_path)
        selected_id = self.physics.process_id
        representative: Mapping[str, object] | None = None
        permutation: tuple[int, ...] | None = None
        for process in manifest.processes:
            if process["id"] == selected_id:
                representative = process
                break
            for alias in cast(Sequence[Mapping[str, object]], process["aliases"]):
                if alias["id"] == selected_id:
                    representative = process
                    permutation = tuple(
                        _manifest_integer(index, "alias external permutation entry")
                        for index in cast(
                            Sequence[object], alias["external_permutation"]
                        )
                    )
                    break
            if representative is not None:
                break
        if representative is None:
            raise ArtifactError(
                f"runtime selected process {selected_id!r} is absent from its artifact"
            )
        process_id = str(representative["id"])
        payloads = tuple(
            payload
            for payload in manifest.payloads
            if payload.role == "validation-momenta" and payload.process_id == process_id
        )
        if len(payloads) != 1:
            raise ArtifactError(
                f"process {process_id!r} must declare exactly one validation point"
            )
        try:
            raw = json.loads((manifest.root / payloads[0].path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"could not read validation point for process {process_id!r}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("available") is not True:
            return None
        points = raw.get("points")
        if not isinstance(points, list) or len(points) != 1:
            raise ArtifactError(
                f"process {process_id!r} has invalid validation-point metadata"
            )
        particles = points[0]
        expected_pdgs = tuple(
            _manifest_integer(value, "representative external PDG")
            for value in cast(Sequence[object], representative["external_pdgs"])
        )
        if not isinstance(particles, list) or len(particles) != len(expected_pdgs):
            raise ArtifactError(
                f"process {process_id!r} validation point has the wrong particle count"
            )
        vectors: list[tuple[float, float, float, float]] = []
        pdgs: list[int] = []
        for index, particle in enumerate(particles):
            if not isinstance(particle, dict):
                raise ArtifactError(
                    f"process {process_id!r} validation particle {index} is invalid"
                )
            momentum = particle.get("momentum")
            if not isinstance(momentum, list) or len(momentum) != 4:
                raise ArtifactError(
                    f"process {process_id!r} validation momentum {index} is invalid"
                )
            try:
                pdgs.append(int(particle["pdg"]))
                vectors.append(
                    cast(
                        tuple[float, float, float, float],
                        tuple(float(component) for component in momentum),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"process {process_id!r} validation particle {index} is invalid"
                ) from exc
        if tuple(pdgs) != expected_pdgs:
            raise ArtifactError(
                f"process {process_id!r} validation PDGs do not match its metadata"
            )
        if permutation is not None:
            reordered: list[tuple[float, float, float, float] | None] = [None] * len(
                vectors
            )
            for representative_index, alias_index in enumerate(permutation):
                reordered[alias_index] = vectors[representative_index]
            if any(vector is None for vector in reordered):
                raise ArtifactError(
                    f"process alias {selected_id!r} has an incomplete permutation"
                )
            vectors = cast(list[tuple[float, float, float, float]], reordered)
        return (tuple(vectors),)


def load_runtime_backend(
    artifact: os.PathLike[str] | str,
    *,
    process: str | None = None,
    model_parameters: ModelParameters | None = None,
    mute_warnings: bool = False,
) -> RusticolRuntimeBackend:
    """Load a schema-v3 artifact without importing the extension during discovery."""

    path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
    parameters = dict(model_parameters) if model_parameters is not None else None
    module = _load_native_module()
    runtime = _invoke(
        module,
        module.Runtime.load,
        path,
        process=process,
        model_parameters=parameters,
        mute_warnings=mute_warnings,
    )
    return RusticolRuntimeBackend(runtime, module, path)


__all__ = ["RusticolRuntimeBackend", "load_runtime_backend"]
