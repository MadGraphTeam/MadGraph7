# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pyamplicol._internal.versions import (
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EAGER_DAG_F64_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
    verify_native_module,
)
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
from pyamplicol.artifacts import MANIFEST_NAME, ArtifactManifest, load_manifest

if TYPE_CHECKING:
    from .eager_exact import EagerExactExecutor
    from .symbolica_exact import SymbolicaExactExecutor

    _ExactExecutor = EagerExactExecutor | SymbolicaExactExecutor

_Accuracy = Literal["lc", "nlc", "full"]
_ParticleState = Literal["incoming", "outgoing"]


def _manifest_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{description} must be an integer")
    return value


def _load_native_module() -> Any:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(module)
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


def _accepts_keyword_arguments(operation: object, *names: str) -> bool:
    if not callable(operation):
        return False
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    declared = {parameter.name for parameter in parameters}
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    return accepts_arbitrary or all(name in declared for name in names)


def _normalized_selectors(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(values) or None


def _selected_manifest_process(
    manifest: ArtifactManifest,
    selected_id: str,
) -> Mapping[str, object]:
    for process in manifest.processes:
        if process["id"] == selected_id:
            return process
        if any(
            alias["id"] == selected_id
            for alias in cast(Sequence[Mapping[str, object]], process["aliases"])
        ):
            return process
    raise ArtifactError(
        f"runtime selected process {selected_id!r} is absent from its artifact"
    )


def _has_reusable_runtime_selector_contract(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
) -> bool:
    relative = str(process["physics_path"])
    try:
        payload = json.loads((manifest.root / relative).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(
            f"could not read runtime selector metadata {relative}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"invalid runtime selector metadata {relative}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"runtime physics metadata {relative} must be an object")
    extensions = payload.get("extensions")
    if extensions is None:
        return False
    if not isinstance(extensions, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions is invalid"
        )
    selectors = extensions.get("runtime_selectors")
    if selectors is None:
        return False
    if not isinstance(selectors, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions.runtime_selectors "
            "is invalid"
        )
    axes = selectors.get("axes")
    if not isinstance(axes, Mapping):
        raise ArtifactError(
            f"runtime physics metadata {relative}.extensions.runtime_selectors.axes "
            "is invalid"
        )
    return any(
        isinstance(axis, Mapping)
        and axis.get("runtime_contract") == "complete-reusable"
        for axis in axes.values()
    )


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


def _native_runtime_metadata(runtime: Any) -> Mapping[str, object]:
    metadata_json = getattr(runtime, "metadata_json", None)
    if not callable(metadata_json):
        return {"execution_mode": "compiled"}
    try:
        payload = json.loads(str(metadata_json()))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"native runtime metadata is invalid: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("native runtime metadata must be an object")
    return dict(payload)


def _native_execution_mode(
    metadata: Mapping[str, object],
) -> Literal["compiled", "eager"]:
    payload = metadata
    mode = payload.get("execution_mode", "compiled")
    if mode not in {"compiled", "eager"}:
        raise CompatibilityError(f"unsupported runtime execution mode {mode!r}")
    return cast(Literal["compiled", "eager"], mode)


class RusticolRuntimeBackend:
    """Public-protocol adapter around ``pyamplicol._rusticol.Runtime``."""

    def __init__(
        self,
        runtime: Any,
        native_module: Any,
        artifact_path: Path,
        manifest: ArtifactManifest | None = None,
    ) -> None:
        self._runtime = runtime
        self._native_module = native_module
        self._artifact_path = artifact_path
        self._native_metadata = _native_runtime_metadata(runtime)
        self._execution_mode = _native_execution_mode(self._native_metadata)
        self._physics: ProcessPhysics | None = None
        self._exact_executor: _ExactExecutor | None = None
        self._required_runtime_capabilities: tuple[str, ...] = ()
        self._supports_per_point_selectors = _accepts_keyword_arguments(
            runtime.evaluate,
            "helicity_by_point",
            "color_flow_by_point",
        )
        if manifest is not None:
            self._validate_runtime_selector_capability(manifest)

    @property
    def physics(self) -> ProcessPhysics:
        if self._physics is None:
            self._physics = _physics_from_native(self._runtime.physics)
        return self._physics

    @property
    def execution_mode(self) -> Literal["compiled", "eager"]:
        """Return the native execution lane selected by the artifact."""

        return self._execution_mode

    @property
    def supports_profiling(self) -> bool:
        """Whether this installed native runtime exposes the optional profiler."""

        return callable(getattr(self._runtime, "profile", None))

    @property
    def supports_per_point_selectors(self) -> bool:
        """Whether this artifact/runtime pair supports per-point selectors."""

        return self._supports_per_point_selectors

    @property
    def required_runtime_capabilities(self) -> tuple[str, ...]:
        """Runtime capabilities declared by the selected artifact process."""

        return self._required_runtime_capabilities

    def _validate_runtime_selector_capability(
        self,
        manifest: ArtifactManifest,
    ) -> None:
        selected_id = self._native_metadata.get("process_key")
        if not isinstance(selected_id, str) or not selected_id:
            selected_id = self.physics.process_id
        process = _selected_manifest_process(manifest, selected_id)
        capabilities = tuple(
            str(value)
            for value in cast(
                Sequence[object], process["required_runtime_capabilities"]
            )
        )
        self._required_runtime_capabilities = capabilities
        reusable_selectors = _has_reusable_runtime_selector_contract(
            manifest,
            process,
        )
        if self._execution_mode == "compiled":
            declares_selector_capability = (
                COMPILED_RUNTIME_SELECTORS_CAPABILITY in capabilities
            )
            if reusable_selectors and not declares_selector_capability:
                raise CompatibilityError(
                    "compiled artifact declares reusable runtime selectors but does "
                    f"not require {COMPILED_RUNTIME_SELECTORS_CAPABILITY!r}; "
                    "regenerate the artifact with the current pyAmpliCol"
                )
            self._supports_per_point_selectors = declares_selector_capability
        else:
            self._supports_per_point_selectors = any(
                capability in capabilities
                for capability in (
                    EAGER_DAG_F64_RUNTIME_CAPABILITY,
                    EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
                )
            )
        if self._supports_per_point_selectors and not _accepts_keyword_arguments(
            self._runtime.evaluate,
            "helicity_by_point",
            "color_flow_by_point",
        ):
            required = (
                COMPILED_RUNTIME_SELECTORS_CAPABILITY
                if self._execution_mode == "compiled"
                else next(
                    capability
                    for capability in (
                        EAGER_DAG_F64_RUNTIME_CAPABILITY,
                        EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
                    )
                    if capability in capabilities
                )
            )
            raise CompatibilityError(
                f"artifact requires runtime capability {required!r}, but the "
                "installed native runtime does not accept per-point selectors"
            )

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
    ) -> tuple[complex | Decimal, ...]:
        helicities = _normalized_selectors(helicities)
        color_flows = _normalized_selectors(color_flows)
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = _normalized_selectors(color_flow_by_point)
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        color_flows = self._resolve_color_flows(color_flows)
        color_flow_by_point = self._resolve_color_flows(color_flow_by_point)
        if (
            helicity_by_point is not None or color_flow_by_point is not None
        ) and not self._supports_per_point_selectors:
            raise CompatibilityError(
                "the selected artifact/runtime does not support per-point "
                "helicity or color-flow selectors; regenerate a reusable-selector "
                "artifact with the current pyAmpliCol"
            )
        helicity_indices = self._point_selector_indices(
            helicity_by_point,
            self.physics.helicity_ids,
            "helicity_by_point",
        )
        color_indices = self._point_selector_indices(
            color_flow_by_point,
            self.physics.color_ids,
            "color_flow_by_point",
        )
        if precision != 16:
            if helicity_by_point is not None or color_flow_by_point is not None:
                return self._evaluate_exact_by_point(
                    momenta,
                    helicities=helicities,
                    color_flows=color_flows,
                    helicity_by_point=helicity_by_point,
                    color_flow_by_point=color_flow_by_point,
                    precision=precision,
                )
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
        selector_arguments: dict[str, object] = {}
        if helicity_indices is not None:
            selector_arguments["helicity_by_point"] = helicity_indices
        if color_indices is not None:
            selector_arguments["color_flow_by_point"] = color_indices
        values = _invoke(
            self._native_module,
            self._runtime.evaluate,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            **selector_arguments,
        )
        return tuple(_scalar_from_native(value) for value in values)

    @staticmethod
    def _point_selector_indices(
        values: Sequence[str] | None,
        available: Sequence[str],
        name: str,
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        index_by_id = {identifier: index for index, identifier in enumerate(available)}
        indices: list[int] = []
        for point_index, identifier in enumerate(values):
            try:
                indices.append(index_by_id[identifier])
            except KeyError as exc:
                raise EvaluationError(
                    f"{name}[{point_index}] references unknown selector {identifier!r}"
                ) from exc
        return tuple(indices)

    def _evaluate_exact_by_point(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        helicity_by_point: Sequence[str] | None,
        color_flow_by_point: Sequence[str] | None,
        precision: int,
    ) -> tuple[complex | Decimal, ...]:
        point_count = len(momenta)
        for name, selectors in (
            ("helicity_by_point", helicity_by_point),
            ("color_flow_by_point", color_flow_by_point),
        ):
            if selectors is not None and len(selectors) != point_count:
                raise EvaluationError(
                    f"{name} contains {len(selectors)} entries, expected one "
                    f"selector for each of {point_count} points"
                )
        grouped: dict[tuple[str | None, str | None], list[int]] = {}
        for point_index in range(point_count):
            key = (
                None if helicity_by_point is None else helicity_by_point[point_index],
                None
                if color_flow_by_point is None
                else color_flow_by_point[point_index],
            )
            grouped.setdefault(key, []).append(point_index)
        output: list[complex | Decimal | None] = [None] * point_count
        exact = self._exact()
        for (point_helicity, point_color), point_indices in grouped.items():
            selected_momenta = tuple(momenta[index] for index in point_indices)
            resolved = exact.evaluate_resolved(
                selected_momenta,
                helicities=(
                    (point_helicity,) if point_helicity is not None else helicities
                ),
                color_flows=(point_color,) if point_color is not None else color_flows,
                precision=precision,
            )
            for point_index, value in zip(point_indices, resolved.total(), strict=True):
                output[point_index] = value
        if any(value is None for value in output):
            raise EvaluationError("per-point exact selector evaluation was incomplete")
        return cast(tuple[complex | Decimal, ...], tuple(output))

    def _benchmark_f64_wall_time(
        self,
        momenta: Momenta,
        repetitions: int,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
    ) -> float:
        helicities = _normalized_selectors(helicities)
        color_flows = _normalized_selectors(color_flows)
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = _normalized_selectors(color_flow_by_point)
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        color_flows = self._resolve_color_flows(color_flows)
        color_flow_by_point = self._resolve_color_flows(color_flow_by_point)
        if (
            helicity_by_point is not None or color_flow_by_point is not None
        ) and not self._supports_per_point_selectors:
            raise CompatibilityError(
                "the selected artifact/runtime does not support per-point "
                "helicity or color-flow selectors"
            )
        helicity_indices = self._point_selector_indices(
            helicity_by_point,
            self.physics.helicity_ids,
            "helicity_by_point",
        )
        color_indices = self._point_selector_indices(
            color_flow_by_point,
            self.physics.color_ids,
            "color_flow_by_point",
        )
        timer = getattr(self._runtime, "_benchmark_f64_wall_time", None)
        if not callable(timer):
            raise EvaluationError("native Rusticol wall timer is unavailable")
        selector_arguments: dict[str, object] = {}
        if helicity_indices is not None:
            selector_arguments["helicity_by_point"] = helicity_indices
        if color_indices is not None:
            selector_arguments["color_flow_by_point"] = color_indices
        return float(
            _invoke(
                self._native_module,
                timer,
                momenta,
                repetitions,
                helicities=helicities,
                color_flows=color_flows,
                precision=precision,
                **selector_arguments,
            )
        )

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> ResolvedEvaluation:
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
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

    def profile(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = self._resolve_color_flows(
            _normalized_selectors(color_flow_by_point)
        )
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        if precision != 16:
            raise EvaluationError(
                "runtime profiling is available only for native f64 precision"
            )
        profiler = getattr(self._runtime, "profile", None)
        if not callable(profiler):
            raise EvaluationError("native runtime does not expose profiling")
        selector_arguments: dict[str, object] = {}
        if helicity_by_point is not None or color_flow_by_point is not None:
            if not self._supports_per_point_selectors or not _accepts_keyword_arguments(
                profiler,
                "helicity_by_point",
                "color_flow_by_point",
            ):
                raise CompatibilityError(
                    "the selected artifact/runtime does not support profiling "
                    "with per-point helicity or color-flow selectors"
                )
            helicity_indices = self._point_selector_indices(
                helicity_by_point,
                self.physics.helicity_ids,
                "helicity_by_point",
            )
            color_indices = self._point_selector_indices(
                color_flow_by_point,
                self.physics.color_ids,
                "color_flow_by_point",
            )
            if helicity_indices is not None:
                selector_arguments["helicity_by_point"] = helicity_indices
            if color_indices is not None:
                selector_arguments["color_flow_by_point"] = color_indices
        payload = _invoke(
            self._native_module,
            profiler,
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            include_values=include_values,
            **selector_arguments,
        )
        if not isinstance(payload, Mapping):
            raise EvaluationError("native runtime profile is not a mapping")
        return cast(Mapping[str, object], dict(payload))

    def profile_repeated(
        self,
        momenta: Momenta,
        repetitions: int,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        helicities = _normalized_selectors(helicities)
        color_flows = self._resolve_color_flows(_normalized_selectors(color_flows))
        helicity_by_point = _normalized_selectors(helicity_by_point)
        color_flow_by_point = self._resolve_color_flows(
            _normalized_selectors(color_flow_by_point)
        )
        if helicities is not None and helicity_by_point is not None:
            raise EvaluationError(
                "helicities and helicity_by_point are mutually exclusive"
            )
        if color_flows is not None and color_flow_by_point is not None:
            raise EvaluationError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        if precision != 16:
            raise EvaluationError(
                "runtime profiling is available only for native f64 precision"
            )
        profiler = getattr(self._runtime, "profile_repeated", None)
        if not callable(profiler):
            raise EvaluationError("native runtime does not expose repeated profiling")
        selector_arguments: dict[str, object] = {}
        if helicity_by_point is not None or color_flow_by_point is not None:
            if not self._supports_per_point_selectors or not _accepts_keyword_arguments(
                profiler,
                "helicity_by_point",
                "color_flow_by_point",
            ):
                raise CompatibilityError(
                    "the selected artifact/runtime does not support profiling "
                    "with per-point helicity or color-flow selectors"
                )
            helicity_indices = self._point_selector_indices(
                helicity_by_point,
                self.physics.helicity_ids,
                "helicity_by_point",
            )
            color_indices = self._point_selector_indices(
                color_flow_by_point,
                self.physics.color_ids,
                "color_flow_by_point",
            )
            if helicity_indices is not None:
                selector_arguments["helicity_by_point"] = helicity_indices
            if color_indices is not None:
                selector_arguments["color_flow_by_point"] = color_indices
        payload = _invoke(
            self._native_module,
            profiler,
            momenta,
            repetitions,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            include_values=include_values,
            **selector_arguments,
        )
        if not isinstance(payload, Mapping):
            raise EvaluationError("native repeated runtime profile is not a mapping")
        return cast(Mapping[str, object], dict(payload))

    def evaluate_profile(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        helicity_by_point: Sequence[str] | None = None,
        color_flow_by_point: Sequence[str] | None = None,
        precision: int = 16,
        include_values: bool = False,
    ) -> Mapping[str, object]:
        return self.profile(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            helicity_by_point=helicity_by_point,
            color_flow_by_point=color_flow_by_point,
            precision=precision,
            include_values=include_values,
        )

    def _resolve_color_flows(
        self,
        color_flows: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        if color_flows is None or not color_flows:
            return None
        available = self.physics.color_ids
        resolved: list[str] = []
        for requested in color_flows:
            if requested in available:
                resolved.append(requested)
                continue
            try:
                ordinal = int(requested, 10)
            except ValueError:
                resolved.append(requested)
                continue
            if (
                str(ordinal) != requested.strip()
                or ordinal < 1
                or ordinal > len(available)
            ):
                raise EvaluationError(
                    f"color-flow ordinal {requested!r} is out of range; choose "
                    f"1..{len(available)} or a stable color component ID"
                )
            resolved.append(available[ordinal - 1])
        return tuple(resolved)

    def _exact(self) -> _ExactExecutor:
        if self._exact_executor is None:
            if self._execution_mode == "eager":
                from .eager_exact import EagerExactExecutor

                self._exact_executor = EagerExactExecutor(
                    self._artifact_path, self.physics.process_id, self._runtime
                )
            else:
                from .symbolica_exact import SymbolicaExactExecutor

                self._exact_executor = SymbolicaExactExecutor(
                    self._artifact_path, self.physics.process_id, self._runtime
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
    # Rusticol is the authoritative runtime loader and verifies every declared
    # payload before returning.  Parse the manifest here only for the Python
    # adapter's selector-contract checks; hashing it again would add a complete
    # extra pass over large compact eager containers.
    manifest = (
        load_manifest(path, verify_payloads=False)
        if (path / MANIFEST_NAME).is_file()
        else None
    )
    runtime = _invoke(
        module,
        module.Runtime.load,
        path,
        process=process,
        model_parameters=parameters,
        mute_warnings=mute_warnings,
    )
    return RusticolRuntimeBackend(runtime, module, path, manifest)


__all__ = ["RusticolRuntimeBackend", "load_runtime_backend"]
