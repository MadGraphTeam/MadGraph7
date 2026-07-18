# SPDX-License-Identifier: 0BSD
"""Read-only summaries of generated process artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pyamplicol.api.errors import ArtifactError

from .manifest import ArtifactManifest, load_manifest


@dataclass(frozen=True, slots=True)
class ArtifactAliasInspection:
    id: str
    expression: str
    representative_id: str
    external_pdgs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArtifactProcessInspection:
    id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    default: bool
    physical_helicities: int
    computed_helicities: int
    physical_color_components: int
    computed_color_components: int
    helicity_coverage: str
    color_coverage: str
    aliases: tuple[ArtifactAliasInspection, ...]


@dataclass(frozen=True, slots=True)
class ArtifactDependencyInspection:
    name: str
    version: str
    license: str
    source: str


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    kind: str
    path: Path
    artifact_kind: str
    artifact_id: str
    created_utc: str
    producer_version: str
    target: str
    cpu_features: tuple[str, ...]
    model_name: str
    model_source: str
    model_restriction: str | None
    default_process_id: str | None
    runtime_engine: str
    runtime_version: str
    runtime_capabilities: tuple[str, ...]
    payload_count: int
    payload_size_bytes: int
    integrity: str
    processes: tuple[ArtifactProcessInspection, ...]
    dependencies: tuple[ArtifactDependencyInspection, ...]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} must be an array")
    return value


def _computed(value: object, context: str) -> bool:
    record = _mapping(value, context)
    if record.get("kind") == "contracted-color":
        return True
    computed = record.get("computed")
    if not isinstance(computed, bool):
        raise ArtifactError(f"{context}.computed must be a boolean")
    return computed


def _physics_counts(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
) -> tuple[int, int, int, int, str, str]:
    relative = str(process["physics_path"])
    path = manifest.root / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(
            f"cannot read runtime physics metadata {relative}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"invalid runtime physics metadata {relative}: {exc}"
        ) from exc

    physics = _mapping(payload, f"runtime physics metadata {relative}")
    helicities = _sequence(physics.get("helicities"), f"{relative}.helicities")
    colors = _sequence(physics.get("color_components"), f"{relative}.color_components")
    coverage = _mapping(physics.get("coverage"), f"{relative}.coverage")

    computed_helicities = sum(
        _computed(item, f"{relative}.helicities[{index}]")
        for index, item in enumerate(helicities)
    )
    computed_colors = sum(
        _computed(item, f"{relative}.color_components[{index}]")
        for index, item in enumerate(colors)
    )
    return (
        len(helicities),
        computed_helicities,
        len(colors),
        computed_colors,
        str(coverage.get("helicities", "unknown")),
        str(coverage.get("color", "unknown")),
    )


def _process_inspection(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
) -> ArtifactProcessInspection:
    process_id = str(process["id"])
    aliases = tuple(
        ArtifactAliasInspection(
            id=str(alias["id"]),
            expression=str(alias["expression"]),
            representative_id=process_id,
            external_pdgs=tuple(int(value) for value in alias["external_pdgs"]),
        )
        for alias in process["aliases"]
    )
    (
        physical_helicities,
        computed_helicities,
        physical_colors,
        computed_colors,
        helicity_coverage,
        color_coverage,
    ) = _physics_counts(manifest, process)
    return ArtifactProcessInspection(
        id=process_id,
        expression=str(process["expression"]),
        color_accuracy=str(process["color_accuracy"]),
        external_pdgs=tuple(int(value) for value in process["external_pdgs"]),
        default=process_id == manifest.default_process_id,
        physical_helicities=physical_helicities,
        computed_helicities=computed_helicities,
        physical_color_components=physical_colors,
        computed_color_components=computed_colors,
        helicity_coverage=helicity_coverage,
        color_coverage=color_coverage,
        aliases=aliases,
    )


def inspect_artifact(artifact: str | Path) -> ArtifactInspection:
    """Validate and summarize one generated artifact without loading evaluators."""

    manifest = load_manifest(artifact)
    target = _mapping(manifest.producer["target"], "producer.target")
    model = manifest.model
    runtime = manifest.runtime
    processes = tuple(
        _process_inspection(manifest, process) for process in manifest.processes
    )
    dependencies = tuple(
        ArtifactDependencyInspection(
            name=str(dependency["name"]),
            version=str(dependency["version"]),
            license=str(dependency["license"]),
            source=str(dependency["source"]),
        )
        for dependency in manifest.dependencies
    )
    restriction = model.get("restriction")
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=manifest.root,
        artifact_kind=manifest.kind,
        artifact_id=manifest.artifact_id,
        created_utc=manifest.created_utc,
        producer_version=str(manifest.producer["version"]),
        target=str(target["triple"]),
        cpu_features=tuple(str(value) for value in target["cpu_features"]),
        model_name=str(model["name"]),
        model_source=str(model["source_kind"]),
        model_restriction=None if restriction is None else str(restriction),
        default_process_id=manifest.default_process_id,
        runtime_engine=str(runtime["engine"]),
        runtime_version=str(runtime["engine_version"]),
        runtime_capabilities=tuple(
            str(value) for value in runtime["required_runtime_capabilities"]
        ),
        payload_count=len(manifest.payloads),
        payload_size_bytes=sum(payload.size_bytes for payload in manifest.payloads),
        integrity="verified",
        processes=processes,
        dependencies=dependencies,
    )


__all__ = [
    "ArtifactAliasInspection",
    "ArtifactDependencyInspection",
    "ArtifactInspection",
    "ArtifactProcessInspection",
    "inspect_artifact",
]
