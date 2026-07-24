# SPDX-License-Identifier: 0BSD
"""Transactional schema-v3 output for compiled concrete processes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import (
    ArtifactBuilder,
    ArtifactManifest,
    PayloadRecord,
    load_manifest,
)
from pyamplicol.config import (
    ConfigClamp,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
)

from .._internal.versions import (
    COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
    COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
    COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
    COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
    COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    EVALUATOR_RUNTIME_CAPABILITIES,
    PROCESS_ARTIFACT_SCHEMA_VERSION,
    PYTHON_API_VERSION,
    RUNTIME_PHYSICS_SCHEMA_VERSION,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMBOLICA_SERIALIZATION_ABI,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    TOML_SCHEMA_VERSION,
    package_version,
    verify_native_module,
)
from ..evaluators.execution_schema import evaluator_runtime_capabilities
from ..models.loading import COMPILED_MODEL_SCHEMA_VERSION, CompiledModel
from .contracts import RuntimeExpressionSchema
from .eager_columnar import EAGER_LOWERING_INPUT_ABI
from .eager_lowering import EAGER_RUNTIME_KIND, EagerExecutionTables
from .eager_tables import EAGER_KERNEL_ABI, EAGER_PLAN_ABI, EAGER_RUNTIME_CAPABILITY
from .evaluator_container import (
    PacbinIndex,
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin_atomic,
)
from .validation import ValidationPointRecord, validation_point_map

if TYPE_CHECKING:
    from pyamplicol.artifacts.transaction import ArtifactWriteMode

ApiBundleHook = Callable[
    [ArtifactBuilder, Mapping[str, Sequence[Sequence[float]]]],
    Sequence[object],
]

_CONFIG_REQUESTED_PATH = "config/requested.toml"
_CONFIG_EFFECTIVE_PATH = "config/effective.toml"
_COMPILED_MODEL_PATH = "model/compiled-model.json"
_MODEL_PARAMETERS_PATH = "model/parameters.json"
_EVALUATOR_SET_PATH = "processes/evaluators.json"
_EAGER_KERNEL_PACK_PATH = "model/eager-kernel-pack.json"
_EAGER_KERNEL_PAYLOAD_ROOT = "model/eager-kernels"
_HELICITY_SUM_PAYLOAD_ROOT = "helicity-sum"
_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT = "helicity-selector-union"
_COLOR_SELECTOR_PAYLOAD_ROOT = "color-selector"
_EAGER_PACK_IDENTITY_EXTENSION = "eager_prepared_pack"
_EAGER_PACK_IDENTITY_KIND = "pyamplicol-prepared-kernel-pack-identity"
_EAGER_PACK_IDENTITY_SCHEMA_VERSION = 1
_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION = "evaluator_payload_container"
_EVALUATOR_PAYLOAD_CONTAINER_PATH = "evaluators.pacbin"
_EVALUATOR_PAYLOAD_CONTAINER_KIND = "pyamplicol-evaluator-payload-container"
_EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION = 1
_EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI = "pacbin-v1"
EAGER_PLAN_V3_ABI = "pyamplicol-eager-plan-v3"
EAGER_RUNTIME_LAYOUT_ABI = "pyamplicol-eager-runtime-layout-v1"
EAGER_PLAN_V3_RUNTIME_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
EAGER_RUNTIME_CONTAINER_KIND = "pyamplicol-eager-runtime-container"
EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION = 1
EAGER_RUNTIME_STORAGE_ABI = "pacbin-v1"
_EAGER_RUNTIME_CONTAINER_PATH = "eager-runtime.pacbin"
_MAX_EAGER_EXECUTION_SUMMARY_BYTES = 1 << 20
_SAFE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_ARTIFACT_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    }
)


@dataclass(frozen=True, slots=True)
class CompiledExecutionArtifact:
    runtime_schema: RuntimeExpressionSchema
    stage_manifest: Mapping[str, object]
    model_parameter_evaluator: Mapping[str, object] | None
    dag_summary: Mapping[str, object]
    evaluator_root: Path
    color_selector_executions: tuple[CompiledColorSelectorExecutionArtifact, ...] = ()
    helicity_selector_executions: tuple[
        CompiledHelicitySelectorExecutionArtifact, ...
    ] = ()


@dataclass(frozen=True, slots=True)
class CompiledColorSelectorExecutionArtifact:
    materialized_sector_id: int
    execution: CompiledExecutionArtifact


@dataclass(frozen=True, slots=True)
class CompiledHelicitySelectorExecutionArtifact:
    selector_domain_ids: tuple[int, ...]
    execution: CompiledExecutionArtifact
    schedule_mode: str = "parent-closure"


@dataclass(frozen=True, slots=True)
class CompiledProcessArtifact:
    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    runtime_schema: RuntimeExpressionSchema
    stage_manifest: Mapping[str, object]
    model_parameter_evaluator: Mapping[str, object] | None
    dag_summary: Mapping[str, object]
    evaluator_root: Path
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]
    helicity_sum_execution: CompiledExecutionArtifact | None = None
    helicity_selector_executions: tuple[
        CompiledHelicitySelectorExecutionArtifact, ...
    ] = ()
    color_selector_executions: tuple[CompiledColorSelectorExecutionArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class EagerProcessArtifact:
    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    runtime_schema: RuntimeExpressionSchema | Mapping[str, object]
    eager_tables: EagerExecutionTables
    point_tile_size: int
    workspace_mib: int
    dag_summary: Mapping[str, object]
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EagerPlanV3ProcessArtifact:
    """One Rust-lowered eager runtime plus its bounded publication metadata."""

    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    aliases: tuple[Mapping[str, object], ...]
    physics: Mapping[str, object]
    eager_runtime_path: Path
    eager_runtime_size_bytes: int
    eager_runtime_sha256: str
    eager_runtime_member_count: int
    eager_runtime_unpacked_size_bytes: int
    eager_runtime_index_sha256: str
    lowering_input_sha256: str
    referenced_kernel_ids: frozenset[int]
    inspection_summary: Mapping[str, object]
    point_tile_size: int
    workspace_mib: int
    dag_summary: Mapping[str, object]
    validation_point: ValidationPointRecord
    generation_filters: Mapping[str, object]


ProcessArtifact = (
    CompiledProcessArtifact | EagerProcessArtifact | EagerPlanV3ProcessArtifact
)


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    output: Path
    files: tuple[Path, ...]
    validation_points: Mapping[str, Mapping[str, object]]
    api_bundle_path: str | None


@dataclass(frozen=True, slots=True)
class _GenerationConfigProvenance:
    requested: GenerationConfig | RunConfig
    effective: GenerationConfig | RunConfig
    adjustments: tuple[ConfigClamp, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
    ) -> _GenerationConfigProvenance:
        if isinstance(config, ConfigResolution):
            return cls(config.requested, config.effective, config.clamps)
        effective = GenerationConfig() if config is None else config
        return cls(effective, effective)


class _EvaluatorPayloadCollector:
    """Collect evaluator payloads and publish one root pacbin container."""

    def __init__(
        self,
        builder: ArtifactBuilder,
        *,
        existing: ArtifactManifest | None,
        target: Mapping[str, object],
    ) -> None:
        self._builder = builder
        self._existing = existing
        self._target = dict(target)
        self._new_sources: dict[str, PacbinMemberSource] = {}
        self._staged_loose_paths: set[str] = set()
        self._discarded_prefixes: set[str] = set()

    def discard_prefix(self, prefix: str) -> None:
        normalized = prefix.strip("/")
        if not normalized:
            raise ValueError("packed evaluator discard prefix must not be empty")
        self._discarded_prefixes.add(normalized)
        owned_prefix = normalized + "/"
        self._new_sources = {
            path: source
            for path, source in self._new_sources.items()
            if path != normalized and not path.startswith(owned_prefix)
        }

    def add_file(
        self,
        relative: str,
        source: Path,
        *,
        process_id: str | None,
    ) -> PayloadRecord:
        kind = _packed_evaluator_member_kind(relative)
        if kind is None:
            return self._builder.add_file(
                relative,
                source,
                role="evaluator-state",
                media_type=_media_type(source),
                target=self._target,
                process_id=process_id,
            )
            return
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"evaluator payload must be a regular file: {source}")
        # Evaluator builders may reuse their temporary output paths while
        # materializing nested selector lanes.  Snapshot each payload when it
        # is registered so a later write cannot silently change an earlier
        # logical container member before publication.
        record = self._builder.add_file(
            relative,
            source,
            role="evaluator-state",
            media_type=_media_type(source),
            target=self._target,
            process_id=process_id,
        )
        self._register_staged_source(relative, kind)
        return record

    def add_bytes(
        self,
        relative: str,
        content: bytes,
        *,
        process_id: str | None,
        media_type: str | None = None,
    ) -> None:
        kind = _packed_evaluator_member_kind(relative)
        if kind is None:
            self._builder.add_bytes(
                relative,
                content,
                role="evaluator-state",
                media_type=media_type or _media_type(Path(relative)),
                target=self._target,
                process_id=process_id,
            )
            return
        self._builder.add_bytes(
            relative,
            content,
            role="evaluator-state",
            media_type=media_type or _media_type(Path(relative)),
            target=self._target,
            process_id=process_id,
        )
        self._register_staged_source(relative, kind)

    def add_stream(
        self,
        relative: str,
        source: BinaryIO,
        *,
        process_id: str | None,
        media_type: str | None = None,
    ) -> None:
        kind = _packed_evaluator_member_kind(relative)
        self._builder.add_stream(
            relative,
            source,
            role="evaluator-state",
            media_type=media_type or _media_type(Path(relative)),
            target=self._target,
            process_id=process_id,
        )
        if kind is not None:
            self._register_staged_source(relative, kind)

    def publish(self) -> dict[str, object] | None:
        existing_container = _existing_evaluator_container_path(self._existing)
        if existing_container is None:
            return self._publish_with_old_sources(())

        container_path = self._builder.staged_path(existing_container)
        with PacbinReader.open(container_path, verify_payloads=True) as reader:
            old_sources = tuple(
                PacbinMemberSource(
                    member.logical_path,
                    member.kind,
                    cast("BinaryIO", reader.open_member_stream(member.logical_path)),
                )
                for member in reader.members
                if not self._is_discarded(member.logical_path)
                and member.logical_path not in self._new_sources
            )
            return self._publish_with_old_sources(old_sources)

    def _publish_with_old_sources(
        self,
        old_container_sources: Sequence[PacbinMemberSource],
    ) -> dict[str, object] | None:
        combined = {source.logical_path: source for source in old_container_sources}
        loose_paths: set[str] = set(self._staged_loose_paths)
        if self._existing is not None:
            for record in self._existing.payloads:
                kind = _packed_evaluator_member_kind(record.path)
                if kind is None or self._is_discarded(record.path):
                    continue
                path = self._builder.staged_path(record.path)
                if not path.is_file() or path.is_symlink():
                    continue
                loose_paths.add(record.path)
                combined[record.path] = PacbinMemberSource(
                    record.path,
                    kind,
                    path,
                )
        combined.update(self._new_sources)
        if not combined:
            self._builder.discard_payloads(_EVALUATOR_PAYLOAD_CONTAINER_PATH)
            return None

        destination = self._builder.staged_path(
            _EVALUATOR_PAYLOAD_CONTAINER_PATH,
            create_parent=True,
        )
        written = write_pacbin_atomic(destination, combined.values())
        with PacbinReader.open(destination, verify_payloads=True) as verified:
            if verified.index != written:
                raise ValueError("published evaluator container failed verification")
            index = verified.index

        for relative in sorted(loose_paths):
            self._builder.discard_payloads(relative)
        self._builder.register_staged_file(
            _EVALUATOR_PAYLOAD_CONTAINER_PATH,
            role="evaluator-state",
            media_type="application/octet-stream",
            target=self._target,
            process_id=None,
        )
        return _evaluator_payload_container_extension(index)

    def _register_staged_source(
        self,
        relative: str,
        kind: PacbinMemberKind,
    ) -> None:
        path = self._builder.staged_path(relative)
        self._new_sources[relative] = PacbinMemberSource(relative, kind, path)
        self._staged_loose_paths.add(relative)

    def _is_discarded(self, relative: str) -> bool:
        return any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in self._discarded_prefixes
        )


def _packed_evaluator_member_kind(relative: str) -> PacbinMemberKind | None:
    if relative.endswith(".symjit"):
        return PacbinMemberKind.SYMJIT_APPLICATION
    if relative.endswith(".evaluator.bin"):
        return PacbinMemberKind.SYMBOLICA_EXACT_STATE
    return None


def _existing_evaluator_container_path(
    existing: ArtifactManifest | None,
) -> str | None:
    if existing is None:
        return None
    raw = existing.extensions.get(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    if raw is None:
        return None
    extension = _mapping(raw)
    if str(extension.get("path")) != _EVALUATOR_PAYLOAD_CONTAINER_PATH:
        raise ValueError("append artifact has an incompatible evaluator container path")
    return _EVALUATOR_PAYLOAD_CONTAINER_PATH


def _evaluator_payload_container_extension(index: PacbinIndex) -> dict[str, object]:
    return {
        "kind": _EVALUATOR_PAYLOAD_CONTAINER_KIND,
        "schema_version": _EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION,
        "storage_abi": _EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI,
        "path": _EVALUATOR_PAYLOAD_CONTAINER_PATH,
        "member_count": len(index.members),
        "unpacked_size_bytes": sum(member.length for member in index.members),
        "index_sha256": index.index_sha256,
    }


def write_schema_v3_artifact(
    destination: str | Path,
    *,
    mode: Literal["error", "append", "replace"],
    source: ModelSource,
    compiled_model: CompiledModel,
    configuration: _GenerationConfigProvenance,
    processes: Sequence[ProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_hook: ApiBundleHook | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> ArtifactWriteResult:
    if not processes:
        raise ValueError("schema-v3 generation requires at least one concrete process")
    output = Path(destination).expanduser().resolve(strict=False)
    existing = load_manifest(output) if mode == "append" else None
    hook = api_bundle_hook or _default_api_bundle_hook()
    requested_config = _config_payload(configuration.requested)
    effective_config = _config_payload(configuration.effective)
    bundle_requested = _bundle_requested(configuration.effective)
    existing_bundle = _existing_bundle_path(existing)
    can_emit_bundle = hook is not None or existing_bundle is not None
    effective_config = _effective_config_payload(
        effective_config,
        disable_api_bundle=bundle_requested and not can_emit_bundle,
    )
    adjustments = [
        {"path": adjustment.path, "reason": adjustment.reason}
        for adjustment in configuration.adjustments
    ]
    if bundle_requested and not can_emit_bundle:
        adjustments.append(
            {
                "path": "generation.emit_api_bundle",
                "reason": "no root API-bundle emitter is installed",
            }
        )
    requested_bytes = _toml_bytes(requested_config)
    effective_bytes = _toml_bytes(effective_config)
    producer = _producer_metadata(configuration.effective)
    model = _model_metadata(source, compiled_model)
    dependencies = _dependency_metadata(source)
    eager_pack_identity = _eager_prepared_pack_identity(
        existing,
        compiled_model=compiled_model,
        processes=processes,
    )
    _validate_append_compatibility(
        existing,
        producer=producer,
        model=model,
        eager_pack_identity=eager_pack_identity,
        requested_bytes=requested_bytes,
        effective_bytes=effective_bytes,
        adjustments=adjustments,
        processes=processes,
    )
    if existing is not None:
        producer = _plain_mapping(existing.producer)
        model = _plain_mapping(existing.model)
        dependencies = tuple(_plain_mapping(item) for item in existing.dependencies)

    target = _mapping(producer["target"])
    process_records = _existing_process_records(existing)
    evaluator_entries = _existing_evaluator_entries(existing)
    execution_manifest_sha256_by_process: dict[str, str] = {}
    required_runtime_capabilities = set(
        _existing_required_runtime_capabilities(existing)
    )
    for process in processes:
        required_runtime_capabilities.update(_process_runtime_capabilities(process))
    canonical_runtime_capabilities = tuple(sorted(required_runtime_capabilities))
    validation_records = tuple(process.validation_point for process in processes)
    validations = validation_point_map(validation_records)
    bundle_points = _existing_bundle_points(existing)
    bundle_points.update(build_api_validation_points(processes))
    api_bundle_path = existing_bundle
    write_mode = cast("ArtifactWriteMode", mode)
    with ArtifactBuilder(
        output,
        mode=write_mode,
        expected_artifact_id=(existing.artifact_id if existing is not None else None),
    ) as builder:
        evaluator_payloads = _EvaluatorPayloadCollector(
            builder,
            existing=existing,
            target=target,
        )
        if existing is None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "global payloads",
                        "completed": 0,
                        "total": len(processes) + 2,
                    }
                )
            _write_global_payloads(
                builder,
                compiled_model=compiled_model,
                requested_bytes=requested_bytes,
                effective_bytes=effective_bytes,
            )
        eager_kernel_ids = _eager_kernel_ids(
            output,
            existing,
            compiled_model=compiled_model,
            processes=processes,
        )
        if eager_kernel_ids:
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "prepared kernel pack",
                        "completed": 1,
                        "total": len(processes) + 2,
                        "kernel_count": len(eager_kernel_ids),
                    }
                )
            _write_eager_kernel_pack(
                builder,
                compiled_model,
                kernel_ids=eager_kernel_ids,
                evaluator_payloads=evaluator_payloads,
            )
        for process_index, process in enumerate(processes, start=1):
            if progress_callback is not None:
                progress_callback(
                    {
                        "step": "process payloads",
                        "completed": process_index,
                        "total": len(processes) + 2,
                        "process": process.process_id,
                    }
                )
            record, evaluator_entry, execution_sha256 = _write_process_payloads(
                builder,
                process,
                evaluator_payloads=evaluator_payloads,
            )
            process_records.append(record)
            evaluator_entries.append(evaluator_entry)
            execution_manifest_sha256_by_process[process.process_id] = execution_sha256
        builder.add_json(
            _EVALUATOR_SET_PATH,
            {
                "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
                "kind": "pyamplicol-runtime-execution-set",
                "required_runtime_capabilities": list(canonical_runtime_capabilities),
                "processes": evaluator_entries,
            },
            role="evaluator-manifest",
        )
        if bundle_requested and hook is not None:
            api_bundle_path = _call_api_bundle_hook(builder, hook, bundle_points)
        evaluator_payload_container = evaluator_payloads.publish()
        extensions = _extensions(
            existing,
            processes=processes,
            timings=timings,
            api_bundle_requested=bundle_requested,
            api_bundle_path=api_bundle_path,
            eager_pack_identity=eager_pack_identity,
            execution_manifest_sha256_by_process=(execution_manifest_sha256_by_process),
            evaluator_payload_container=evaluator_payload_container,
        )
        builder.finalize(
            kind=(
                "pyamplicol-process"
                if len(process_records) == 1
                else "pyamplicol-process-set"
            ),
            producer=producer,
            model=model,
            configuration={
                "toml_schema_version": 1,
                "requested_path": _CONFIG_REQUESTED_PATH,
                "effective_path": _CONFIG_EFFECTIVE_PATH,
                "adjustments": adjustments,
            },
            processes=process_records,
            default_process_id=str(process_records[0]["id"]),
            runtime={
                "engine": "rusticol",
                "engine_version": str(producer["version"]),
                "evaluator_manifest_path": _EVALUATOR_SET_PATH,
                "api_bundle_path": api_bundle_path,
                "required_runtime_capabilities": list(canonical_runtime_capabilities),
            },
            dependencies=dependencies,
            extensions=extensions,
        )
        staged = load_manifest(builder.root)
        _validate_artifact_references(staged)
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "publishing artifact",
                    "completed": len(processes) + 2,
                    "total": len(processes) + 2,
                    "file_count": len(staged.payloads) + 1,
                }
            )

    manifest = load_manifest(output)
    _validate_artifact_references(manifest)
    files = tuple(
        [output / record.path for record in manifest.payloads]
        + [output / "artifact.json"]
    )
    return ArtifactWriteResult(
        output=output,
        files=files,
        validation_points=validations,
        api_bundle_path=api_bundle_path,
    )


def _write_global_payloads(
    builder: ArtifactBuilder,
    *,
    compiled_model: CompiledModel,
    requested_bytes: bytes,
    effective_bytes: bytes,
) -> None:
    builder.add_bytes(
        _CONFIG_REQUESTED_PATH,
        requested_bytes,
        role="configuration-requested",
        media_type="application/toml",
    )
    builder.add_bytes(
        _CONFIG_EFFECTIVE_PATH,
        effective_bytes,
        role="configuration-effective",
        media_type="application/toml",
    )
    builder.add_json(
        _COMPILED_MODEL_PATH,
        compiled_model.to_dict(),
        role="compiled-model",
    )
    builder.add_json(
        _MODEL_PARAMETERS_PATH,
        {
            "schema_version": 1,
            "kind": "pyamplicol-model-parameter-defaults",
            "parameters": {
                name: [float(value[0]), float(value[1])]
                for name, value in sorted(compiled_model.parameter_defaults.items())
            },
        },
        role="model-parameters",
    )


def _write_eager_kernel_pack(
    builder: ArtifactBuilder,
    compiled_model: CompiledModel,
    *,
    kernel_ids: frozenset[int],
    evaluator_payloads: _EvaluatorPayloadCollector,
) -> None:
    bundle = compiled_model.prepared_bundle
    if bundle is None:
        raise ValueError("eager artifact writing requires a prepared model bundle")
    selected = tuple(
        kernel
        for kernel in bundle.kernel_pack.kernels
        if kernel.kernel_id in kernel_ids
    )
    if {kernel.kernel_id for kernel in selected} != set(kernel_ids):
        missing = sorted(set(kernel_ids) - {kernel.kernel_id for kernel in selected})
        raise ValueError(f"prepared model omits referenced eager kernels {missing}")
    selected_variants = tuple(
        variant
        for variant in bundle.kernel_pack.kernel_variants
        if variant.base_kernel_id in kernel_ids
    )
    builder.discard_payloads(_EAGER_KERNEL_PACK_PATH)
    builder.discard_payloads(_EAGER_KERNEL_PAYLOAD_ROOT, recursive=True)
    evaluator_payloads.discard_prefix(_EAGER_KERNEL_PAYLOAD_ROOT)
    pack_payload = bundle.kernel_pack.to_dict()
    pack_payload["eager_kernel_abi"] = EAGER_KERNEL_ABI
    pack_payload["kernels"] = [kernel.to_dict() for kernel in selected]
    pack_payload["kernel_variants"] = [
        variant.to_dict() for variant in selected_variants
    ]
    pack_payload["resolver_manifest"] = _filtered_eager_resolver_manifest(
        bundle.kernel_pack.resolver_manifest,
        kernel_ids,
    )
    builder.add_json(
        _EAGER_KERNEL_PACK_PATH,
        pack_payload,
        role="evaluator-manifest",
        compact=True,
    )
    referenced_payloads = {
        path
        for record in (*selected, *selected_variants)
        for path in record.referenced_payload_paths
    }
    with zipfile.ZipFile(bundle.path, "r") as archive:
        for member_path in sorted(referenced_payloads):
            with archive.open(member_path, "r") as stream:
                evaluator_payloads.add_stream(
                    f"{_EAGER_KERNEL_PAYLOAD_ROOT}/{member_path}",
                    cast("BinaryIO", stream),
                    process_id=None,
                    media_type=_media_type(Path(member_path)),
                )


def _eager_kernel_ids(
    output: Path,
    existing: ArtifactManifest | None,
    *,
    compiled_model: CompiledModel,
    processes: Sequence[ProcessArtifact],
) -> frozenset[int]:
    has_eager_process = any(_is_eager_process(process) for process in processes)
    kernel_ids = {
        kernel_id
        for process in processes
        if _is_eager_process(process)
        for kernel_id in _eager_referenced_kernel_ids(process)
    }
    has_existing_pack = (
        existing is not None and (output / _EAGER_KERNEL_PACK_PATH).is_file()
    )
    if has_existing_pack:
        try:
            prior = json.loads(
                (output / _EAGER_KERNEL_PACK_PATH).read_text(encoding="utf-8")
            )
            kernel_ids.update(
                int(_mapping(item)["kernel_id"])
                for item in _sequence(_mapping(prior)["kernels"])
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "existing eager kernel pack is malformed; replace the artifact"
            ) from exc
    if has_eager_process or has_existing_pack:
        bundle = compiled_model.prepared_bundle
        if bundle is None:
            raise ValueError("eager artifact writing requires a prepared model bundle")
        parameter_kernel_id = bundle.kernel_pack.resolver_manifest.get(
            "model_parameter_kernel_id"
        )
        if parameter_kernel_id is not None:
            kernel_ids.add(int(parameter_kernel_id))
    return frozenset(kernel_ids)


def _eager_prepared_pack_identity(
    existing: ArtifactManifest | None,
    *,
    compiled_model: CompiledModel,
    processes: Sequence[ProcessArtifact],
) -> dict[str, object] | None:
    existing_uses_eager = existing is not None and (
        bool(
            {
                EAGER_RUNTIME_CAPABILITY,
                EAGER_PLAN_V3_RUNTIME_CAPABILITY,
            }.intersection(_required_runtime_capabilities(existing.runtime))
        )
        or any(record.path == _EAGER_KERNEL_PACK_PATH for record in existing.payloads)
    )
    incoming_uses_eager = any(_is_eager_process(process) for process in processes)
    if not existing_uses_eager and not incoming_uses_eager:
        return None
    bundle = compiled_model.prepared_bundle
    if bundle is None:
        raise ValueError("eager artifact writing requires a prepared model bundle")
    canonical_manifest = json.dumps(
        _deep_plain(bundle.manifest),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(
        b"pyamplicol-prepared-kernel-pack-identity-v1\x00" + canonical_manifest
    ).hexdigest()
    return {
        "kind": _EAGER_PACK_IDENTITY_KIND,
        "schema_version": _EAGER_PACK_IDENTITY_SCHEMA_VERSION,
        "eager_kernel_abi": EAGER_KERNEL_ABI,
        "identity_sha256": digest,
        "backend": bundle.kernel_pack.backend,
        "kernel_count": len(bundle.kernel_pack.kernels),
    }


def _is_eager_process(process: ProcessArtifact) -> bool:
    return isinstance(process, EagerProcessArtifact | EagerPlanV3ProcessArtifact)


def _eager_referenced_kernel_ids(
    process: ProcessArtifact,
) -> frozenset[int]:
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return process.referenced_kernel_ids
    if isinstance(process, EagerProcessArtifact):
        return process.eager_tables.referenced_kernel_ids
    raise TypeError("compiled process has no eager kernel references")


def _filtered_eager_resolver_manifest(
    manifest: Mapping[str, object],
    kernel_ids: frozenset[int],
) -> dict[str, object]:
    result = _plain_mapping(manifest)
    for field in (
        "vertex_bindings",
        "propagator_bindings",
        "closure_bindings",
    ):
        if field not in manifest:
            continue
        result[field] = [
            _plain_mapping(record)
            for item in _sequence(manifest[field])
            if (record := _mapping(item)).get("kernel_id") is None
            or int(record["kernel_id"]) in kernel_ids
        ]
    parameter_kernel_id = manifest.get("model_parameter_kernel_id")
    if parameter_kernel_id is not None and int(parameter_kernel_id) not in kernel_ids:
        result["model_parameter_kernel_id"] = None
    return result


def _write_process_payloads(
    builder: ArtifactBuilder,
    process: ProcessArtifact,
    *,
    evaluator_payloads: _EvaluatorPayloadCollector,
) -> tuple[dict[str, object], dict[str, object], str]:
    prefix = f"processes/{process.process_id}"
    physics_path = f"{prefix}/physics.json"
    execution_path = f"{prefix}/execution.json"
    validation_path = f"{prefix}/validation-momenta.json"
    schema: Mapping[str, object]
    if isinstance(process, EagerPlanV3ProcessArtifact):
        schema = {}
        physics = _mapping(process.physics)
    else:
        schema = _runtime_schema_mapping(process.runtime_schema)
        physics = _mapping(schema.get("physics"))
    if physics.get("process_id") != process.process_id:
        raise ValueError(
            f"runtime physics process ID does not match {process.process_id!r}"
        )
    if isinstance(process, EagerPlanV3ProcessArtifact) and (
        physics.get("schema_version") != RUNTIME_PHYSICS_SCHEMA_VERSION
        or physics.get("kind") != "pyamplicol-resolved-physics"
    ):
        raise ValueError("Rust eager lowering returned incompatible physics metadata")
    builder.add_json(
        physics_path,
        physics,
        role="runtime-physics",
        process_id=process.process_id,
        compact=True,
    )
    if isinstance(process, EagerPlanV3ProcessArtifact):
        runtime_path = f"{prefix}/{_EAGER_RUNTIME_CONTAINER_PATH}"
        runtime_record = evaluator_payloads.add_file(
            runtime_path,
            process.eager_runtime_path,
            process_id=process.process_id,
        )
        _validate_staged_eager_runtime(process, runtime_record)
        execution_record = builder.add_bytes(
            execution_path,
            _bounded_eager_execution_summary(process),
            role="evaluator-manifest",
            media_type="application/json",
            process_id=process.process_id,
        )
    else:
        execution_record = builder.add_json(
            execution_path,
            _execution_manifest(process, schema),
            role="evaluator-manifest",
            process_id=process.process_id,
            compact=True,
        )
    builder.add_json(
        validation_path,
        process.validation_point.to_mapping(),
        role="validation-momenta",
        process_id=process.process_id,
    )
    if isinstance(process, EagerProcessArtifact):
        for relative, content in sorted(process.eager_tables.binary_payloads().items()):
            evaluator_payloads.add_bytes(
                f"{prefix}/{relative}",
                content,
                media_type="application/octet-stream",
                process_id=process.process_id,
            )
    elif isinstance(process, CompiledProcessArtifact):
        _copy_evaluator_payloads(
            evaluator_payloads,
            process.evaluator_root,
            prefix=prefix,
            process_id=process.process_id,
        )
        _copy_color_selector_evaluator_payloads(
            evaluator_payloads,
            process.color_selector_executions,
            prefix=prefix,
            process_id=process.process_id,
        )
        if process.helicity_sum_execution is not None:
            _copy_evaluator_payloads(
                evaluator_payloads,
                process.helicity_sum_execution.evaluator_root,
                prefix=f"{prefix}/{_HELICITY_SUM_PAYLOAD_ROOT}",
                process_id=process.process_id,
            )
            _copy_color_selector_evaluator_payloads(
                evaluator_payloads,
                process.helicity_sum_execution.color_selector_executions,
                prefix=f"{prefix}/{_HELICITY_SUM_PAYLOAD_ROOT}",
                process_id=process.process_id,
            )
        _copy_helicity_selector_evaluator_payloads(
            evaluator_payloads,
            process.helicity_selector_executions,
            prefix=prefix,
            process_id=process.process_id,
        )
    return (
        {
            "id": process.process_id,
            "expression": process.expression,
            "color_accuracy": process.color_accuracy,
            "external_pdgs": list(process.external_pdgs),
            "physics_path": physics_path,
            "required_runtime_capabilities": list(
                _process_runtime_capabilities(process)
            ),
            "aliases": [dict(alias) for alias in process.aliases],
        },
        {
            "process_id": process.process_id,
            "manifest_path": f"{process.process_id}/execution.json",
            "required_runtime_capabilities": list(
                _process_runtime_capabilities(process)
            ),
        },
        execution_record.sha256,
    )


def _runtime_schema_mapping(
    schema: RuntimeExpressionSchema | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(schema, RuntimeExpressionSchema):
        return schema.to_mapping()
    return schema


def _validate_staged_eager_runtime(
    process: EagerPlanV3ProcessArtifact,
    record: PayloadRecord,
) -> None:
    if process.eager_runtime_size_bytes <= 0:
        raise ValueError("Rust eager runtime payload must not be empty")
    payload_sha256 = _canonical_sha256(
        process.eager_runtime_sha256,
        "Rust eager runtime payload SHA-256",
    )
    if (
        record.size_bytes != process.eager_runtime_size_bytes
        or record.sha256 != payload_sha256
    ):
        raise ValueError(
            "Rust eager runtime payload changed after lowering and before publication"
        )


def _bounded_eager_execution_summary(
    process: EagerPlanV3ProcessArtifact,
) -> bytes:
    try:
        content = (
            json.dumps(
                _eager_plan_v3_execution_manifest(process),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Rust eager execution summary is not canonical JSON: {exc}"
        ) from exc
    if len(content) >= _MAX_EAGER_EXECUTION_SUMMARY_BYTES:
        raise ValueError(
            "Rust eager execution summary must be smaller than 1 MiB; "
            f"received {len(content)} bytes"
        )
    return content


def _eager_plan_v3_execution_manifest(
    process: EagerPlanV3ProcessArtifact,
) -> dict[str, object]:
    lowering_input_sha256 = _canonical_sha256(
        process.lowering_input_sha256,
        "eager lowering input SHA-256",
    )
    payload_sha256 = _canonical_sha256(
        process.eager_runtime_sha256,
        "Rust eager runtime payload SHA-256",
    )
    index_sha256 = _canonical_sha256(
        process.eager_runtime_index_sha256,
        "Rust eager runtime index SHA-256",
    )
    member_count = _nonnegative_integer(
        process.eager_runtime_member_count,
        "Rust eager runtime member count",
        minimum=1,
    )
    unpacked_size = _nonnegative_integer(
        process.eager_runtime_unpacked_size_bytes,
        "Rust eager runtime unpacked size",
    )
    payload_size = _nonnegative_integer(
        process.eager_runtime_size_bytes,
        "Rust eager runtime payload size",
        minimum=1,
    )
    capabilities = [EAGER_PLAN_V3_RUNTIME_CAPABILITY]
    plan = {
        "kind": EAGER_RUNTIME_KIND,
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
        "lowering_input_sha256": lowering_input_sha256,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
        "required_runtime_capabilities": capabilities,
        "runtime_container": {
            "kind": EAGER_RUNTIME_CONTAINER_KIND,
            "schema_version": EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
            "storage_abi": EAGER_RUNTIME_STORAGE_ABI,
            "path": _EAGER_RUNTIME_CONTAINER_PATH,
            "size_bytes": payload_size,
            "sha256": payload_sha256,
            "member_count": member_count,
            "unpacked_size_bytes": unpacked_size,
            "index_sha256": index_sha256,
        },
        "inspection_summary": _deep_plain(process.inspection_summary),
    }
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": EAGER_RUNTIME_KIND,
        "required_runtime_capabilities": capabilities,
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "kernel_pack": {
            "manifest_path": _EAGER_KERNEL_PACK_PATH,
            "payload_root": _EAGER_KERNEL_PAYLOAD_ROOT,
        },
        "runtime_options": {
            "point_tile_size": process.point_tile_size,
            "workspace_mib": process.workspace_mib,
        },
        "plan": plan,
        "dag_summary": _dag_summary(process.dag_summary),
    }


def _canonical_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _nonnegative_integer(
    value: object,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _execution_manifest(
    process: ProcessArtifact,
    compiler_schema: Mapping[str, object],
) -> dict[str, object]:
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return _eager_plan_v3_execution_manifest(process)
    if isinstance(process, EagerProcessArtifact):
        topology_replay = compiler_schema.get("lc_topology_replay")
        required_runtime_capabilities = _eager_process_runtime_capabilities(process)
        plan = process.eager_tables.to_metadata()
        plan["required_runtime_capabilities"] = list(required_runtime_capabilities)
        return {
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": EAGER_RUNTIME_KIND,
            "required_runtime_capabilities": list(required_runtime_capabilities),
            "process": process.expression,
            "key": process.process_id,
            "color_accuracy": process.color_accuracy,
            "external_pdg_order": list(process.external_pdgs),
            "eager_plan_abi": EAGER_PLAN_ABI,
            "kernel_pack": {
                "manifest_path": _EAGER_KERNEL_PACK_PATH,
                "payload_root": _EAGER_KERNEL_PAYLOAD_ROOT,
            },
            "runtime_options": {
                "point_tile_size": process.point_tile_size,
                "workspace_mib": process.workspace_mib,
            },
            "plan": plan,
            "dag_summary": _dag_summary(process.dag_summary),
            "runtime_schema": _execution_plan(compiler_schema),
            **(
                {}
                if topology_replay is None
                else {"lc_topology_replay": _plain_mapping(_mapping(topology_replay))}
            ),
        }
    primary = _compiled_execution_lane_manifest(
        runtime_schema=compiler_schema,
        stage_manifest=process.stage_manifest,
        model_parameter_evaluator=process.model_parameter_evaluator,
        dag_summary=process.dag_summary,
        payload_prefix=None,
    )
    required_runtime_capabilities = set(_required_runtime_capabilities(primary))
    color_selector_executions = _compiled_color_selector_execution_manifests(
        process=process,
        executions=process.color_selector_executions,
        parent_payload_prefix=None,
    )
    if color_selector_executions:
        required_runtime_capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in color_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    helicity_sum_execution = process.helicity_sum_execution
    auxiliary: dict[str, object] | None = None
    if helicity_sum_execution is not None:
        auxiliary = _compiled_nested_execution_manifest(
            process=process,
            execution=helicity_sum_execution,
            payload_prefix=_HELICITY_SUM_PAYLOAD_ROOT,
        )
        required_runtime_capabilities.update(_required_runtime_capabilities(auxiliary))
        required_runtime_capabilities.add(COMPILED_HELICITY_DUAL_LANE_CAPABILITY)
    helicity_selector_executions = _compiled_helicity_selector_execution_manifests(
        process=process,
        executions=process.helicity_selector_executions,
        parent_payload_prefix=None,
    )
    if helicity_selector_executions:
        for record in helicity_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
        required_runtime_capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
    if _uses_primary_helicity_recurrence(process):
        required_runtime_capabilities.add(
            COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "compiled": primary["compiled"],
        "dag_summary": primary["dag_summary"],
        "runtime_schema": primary["runtime_schema"],
        **({} if auxiliary is None else {"helicity_sum_execution": auxiliary}),
        **(
            {}
            if not helicity_selector_executions
            else {"helicity_selector_executions": helicity_selector_executions}
        ),
        **(
            {}
            if not color_selector_executions
            else {"color_selector_executions": color_selector_executions}
        ),
    }


def _compiled_helicity_selector_execution_manifests(
    *,
    process: CompiledProcessArtifact,
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
    parent_payload_prefix: str | None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for lane_index, record in enumerate(
        _ordered_helicity_selector_executions(executions)
    ):
        execution = record.execution
        if execution.color_selector_executions:
            raise ValueError(
                "compiled helicity-selector closure execution cannot contain "
                "nested execution lanes"
            )
        lane_prefix = f"{_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT}/class-{lane_index}"
        payload_prefix = (
            lane_prefix
            if parent_payload_prefix is None
            else f"{parent_payload_prefix.rstrip('/')}/{lane_prefix}"
        )
        manifest = _compiled_nested_execution_manifest(
            process=process,
            execution=execution,
            payload_prefix=payload_prefix,
        )
        if (
            record.schedule_mode == "parent-closure"
            and COMPILED_RUNTIME_SELECTORS_CAPABILITY
            in _required_runtime_capabilities(manifest)
        ):
            raise ValueError(
                "compiled helicity-selector closure stage evaluators cannot require "
                "runtime selectors"
            )
        result.append(
            {
                "selector_domain_ids": list(record.selector_domain_ids),
                "schedule_mode": record.schedule_mode,
                "execution": manifest,
            }
        )
    return result


def _compiled_nested_execution_manifest(
    *,
    process: CompiledProcessArtifact,
    execution: CompiledExecutionArtifact,
    payload_prefix: str,
) -> dict[str, object]:
    runtime_schema = _runtime_schema_mapping(execution.runtime_schema)
    lane = _compiled_execution_lane_manifest(
        runtime_schema=runtime_schema,
        stage_manifest=execution.stage_manifest,
        model_parameter_evaluator=execution.model_parameter_evaluator,
        dag_summary=execution.dag_summary,
        payload_prefix=payload_prefix,
    )
    color_selector_executions = _compiled_color_selector_execution_manifests(
        process=process,
        executions=execution.color_selector_executions,
        parent_payload_prefix=payload_prefix,
    )
    required_runtime_capabilities = set(_required_runtime_capabilities(lane))
    if _runtime_schema_uses_primary_helicity_recurrence(
        runtime_schema,
        has_helicity_sum_execution=False,
    ):
        required_runtime_capabilities.add(
            COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        )
    helicity_selector_executions = _compiled_helicity_selector_execution_manifests(
        process=process,
        executions=execution.helicity_selector_executions,
        parent_payload_prefix=payload_prefix,
    )
    if helicity_selector_executions:
        required_runtime_capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in helicity_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    if color_selector_executions:
        required_runtime_capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in color_selector_executions:
            required_runtime_capabilities.update(
                _required_runtime_capabilities(_mapping(record["execution"]))
            )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "compiled": lane["compiled"],
        "dag_summary": lane["dag_summary"],
        "runtime_schema": lane["runtime_schema"],
        "physics_reduction": _plain_mapping(
            _mapping(_mapping(runtime_schema["physics"])["reduction"])
        ),
        **(
            {}
            if not color_selector_executions
            else {"color_selector_executions": color_selector_executions}
        ),
        **(
            {}
            if not helicity_selector_executions
            else {"helicity_selector_executions": (helicity_selector_executions)}
        ),
    }


def _compiled_color_selector_execution_manifests(
    *,
    process: CompiledProcessArtifact,
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
    parent_payload_prefix: str | None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in _ordered_color_selector_executions(executions):
        lane_prefix = _color_selector_payload_prefix(record.materialized_sector_id)
        payload_prefix = (
            lane_prefix
            if parent_payload_prefix is None
            else f"{parent_payload_prefix.rstrip('/')}/{lane_prefix}"
        )
        result.append(
            {
                "materialized_sector_id": record.materialized_sector_id,
                "execution": _compiled_nested_execution_manifest(
                    process=process,
                    execution=record.execution,
                    payload_prefix=payload_prefix,
                ),
            }
        )
    return result


def _compiled_execution_lane_manifest(
    *,
    runtime_schema: Mapping[str, object],
    stage_manifest: Mapping[str, object],
    model_parameter_evaluator: Mapping[str, object] | None,
    dag_summary: Mapping[str, object],
    payload_prefix: str | None,
) -> dict[str, object]:
    serialized_model_parameters = (
        None
        if model_parameter_evaluator is None
        else _model_parameter_evaluator(model_parameter_evaluator)
    )
    stage_evaluators = _stage_evaluator_set(stage_manifest)
    if payload_prefix is not None:
        stage_evaluators = _prefix_evaluator_payload_paths(
            stage_evaluators,
            payload_prefix,
        )
        if serialized_model_parameters is not None:
            serialized_model_parameters = _prefix_evaluator_payload_paths(
                serialized_model_parameters,
                payload_prefix,
            )
    required_runtime_capabilities = set(
        _required_runtime_capabilities(stage_evaluators)
    )
    if _runtime_schema_uses_walsh_color_contraction(runtime_schema):
        required_runtime_capabilities.add(
            COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY
        )
    if serialized_model_parameters is not None:
        required_runtime_capabilities.update(
            _required_runtime_capabilities(serialized_model_parameters)
        )
    compiled_manifest: dict[str, object] = {
        "kind": "generic-dag-stage-blueprint",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "model_parameter_evaluator": serialized_model_parameters,
        "stage_evaluators": stage_evaluators,
    }
    topology_replay = runtime_schema.get("lc_topology_replay")
    if topology_replay is not None:
        compiled_manifest["lc_topology_replay"] = _plain_mapping(
            _mapping(topology_replay)
        )
    helicity_recurrence = runtime_schema.get("helicity_recurrence")
    if helicity_recurrence is not None:
        compiled_manifest["helicity_recurrence"] = _plain_mapping(
            _mapping(helicity_recurrence)
        )
    return {
        "kind": "pyamplicol-runtime-compiled-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "compiled": compiled_manifest,
        "dag_summary": _dag_summary(dag_summary),
        "runtime_schema": _execution_plan(runtime_schema),
    }


def _prefix_evaluator_payload_paths(
    record: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    path_fields = {"application_path", "evaluator_state_path", "library_path"}

    def visit(value: object, *, field: str | None = None) -> object:
        if field in path_fields and value is not None:
            if not isinstance(value, str):
                raise TypeError(f"evaluator payload path {field!r} must be a string")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"evaluator payload path {value!r} is not artifact-relative"
                )
            return f"{prefix.rstrip('/')}/{path.as_posix()}"
        if isinstance(value, Mapping):
            return {
                str(key): visit(item, field=str(key)) for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return [visit(item) for item in value]
        return value

    result = visit(record)
    if not isinstance(result, dict):  # pragma: no cover - internal invariant
        raise TypeError("compiled evaluator manifest must be an object")
    return result


def _execution_plan(schema: Mapping[str, object]) -> dict[str, object]:
    source_fill = _mapping(schema["source_fill"])
    source_records = tuple(_mapping(item) for item in _sequence(source_fill["sources"]))
    source_count = (
        int(source_records[-1]["source_parameter_stop"]) if source_records else 0
    )
    parameter_layout = _mapping(schema["parameter_layout"])
    value_count = int(parameter_layout["value_component_count"])
    momentum_count = int(parameter_layout["momentum_parameter_count"])
    model_parameter_count = int(parameter_layout.get("model_parameter_count", 0))
    parameters = tuple(
        _execution_model_parameter(_mapping(item))
        for item in _sequence(schema.get("model_parameters", ()))
    )
    mass_parameters = {
        int(item["pdg"]): str(item["name"])
        for item in parameters
        if item["kind"] == "particle_mass" and item.get("pdg") is not None
    }
    model = _mapping(schema.get("model", {}))
    particles = tuple(_mapping(item) for item in _sequence(model.get("particles", ())))
    normalization = _mapping(schema.get("normalization", {}))
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-plan",
        "process_key": str(schema["process_key"]),
        "process": str(schema["process"]),
        "external_particles": [
            _select(
                _mapping(item),
                "label",
                "index",
                "pdg",
                "outgoing_pdg",
                "role",
                "momentum_slot",
            )
            for item in _sequence(schema["external_particles"])
        ],
        "model": {
            "particles": [
                {
                    "pdg": int(particle["pdg"]),
                    "mass": float(particle.get("mass", 0.0)),
                    "mass_parameter": (
                        str(particle["mass_parameter"])
                        if particle.get("mass_parameter") is not None
                        else mass_parameters.get(int(particle["pdg"]))
                    ),
                }
                for particle in particles
            ]
        },
        "model_parameters": list(parameters),
        "normalization": {
            "color_factor": float(normalization.get("color_factor", 1.0)),
            "global_coupling_factor": float(
                normalization.get("global_coupling_factor", 1.0)
            ),
            "average_factor": float(normalization.get("average_factor", 1.0)),
            "identical_factor": float(normalization.get("identical_factor", 1.0)),
            "qcd_coupling_power": int(normalization.get("qcd_coupling_power", 0)),
            "electroweak_coupling_power": int(
                normalization.get("electroweak_coupling_power", 0)
            ),
        },
        "parameter_layout": {
            "source_component_parameter_count": source_count,
            "momentum_parameter_count": momentum_count,
            "model_parameter_count": model_parameter_count,
            "parameter_count_if_flattened": (
                source_count + momentum_count + model_parameter_count
            ),
            "value_component_count": value_count,
            "source_components_complex": True,
            "momentum_components_real": True,
            "real_valued_inputs": list(
                range(
                    source_count,
                    source_count + momentum_count + model_parameter_count,
                )
            ),
        },
        "current_storage": _current_storage(_mapping(schema["current_storage"])),
        "value_storage": _value_storage(_mapping(schema["value_storage"])),
        "source_fill": {
            "source_count": int(source_fill["source_count"]),
            "sources": [_source_record(item) for item in source_records],
        },
        "momentum_slots": [
            _select(
                _mapping(item),
                "momentum_slot_id",
                "momentum_mask",
                "external_labels",
                "component_start",
                "component_stop",
                "real_valued",
            )
            for item in _sequence(schema["momentum_slots"])
        ],
        "stages": [
            _runtime_stage(_mapping(item)) for item in _sequence(schema["stages"])
        ],
        "amplitude_stage": _amplitude_stage(_mapping(schema["amplitude_stage"])),
        **(
            {}
            if schema.get("helicity_recurrence") is None
            else {
                "helicity_recurrence": _plain_mapping(
                    _mapping(schema["helicity_recurrence"])
                )
            }
        ),
    }


def _execution_model_parameter(record: Mapping[str, object]) -> dict[str, object]:
    result = _select(record, "name", "kind", "parameter_index", "default")
    for name in ("pdg", "runtime_name", "complex_component"):
        if record.get(name) is not None:
            result[name] = record[name]
    return result


def _current_storage(storage: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "current_id",
        "component_start",
        "component_stop",
        "dimension",
        "is_source",
        "particle_id",
        "external_mask",
        "external_labels",
        "helicity_ancestry",
        "chirality",
        "spin_state",
        "flavour_flow",
        "color_state",
        "momentum_mask",
        "auxiliary_kind",
    )
    return {
        "component_count": int(storage["component_count"]),
        "number_type": str(storage["number_type"]),
        "metadata_compacted": True,
        "current_slots": [
            _select(_mapping(item), *fields)
            for item in _sequence(storage["current_slots"])
        ],
    }


def _value_storage(storage: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "value_slot_id",
        "current_id",
        "variant",
        "component_start",
        "component_stop",
        "dimension",
        "current_component_start",
        "current_component_stop",
        "is_source",
        "applies_propagator",
        "particle_id",
        "external_mask",
        "external_labels",
        "momentum_mask",
        "chirality",
        "propagator",
    )
    return {
        "component_count": int(storage["component_count"]),
        "number_type": str(storage["number_type"]),
        "metadata_compacted": True,
        "value_slots": [
            _select(_mapping(item), *fields)
            for item in _sequence(storage["value_slots"])
        ],
    }


def _source_record(record: Mapping[str, object]) -> dict[str, object]:
    return _select(
        record,
        "source_id",
        "current_id",
        "current_component_start",
        "current_component_stop",
        "value_slot",
        "source_parameter_start",
        "source_parameter_stop",
        "leg_label",
        "input_momentum_slot",
        "side",
        "crossing",
        "physical_pdg",
        "outgoing_pdg",
        "particle_id",
        "anti_particle_id",
        "source_kind",
        "wavefunction_kind",
        "source_orientation",
        "source_basis",
        "source_ir",
        "applied_crossing",
        "source_helicity",
        "chirality",
        "spin_state",
        "dimension",
        "helicity_ancestry",
        "color_state",
    )


def _runtime_stage(stage: Mapping[str, object]) -> dict[str, object]:
    interactions_compacted = bool(stage.get("interactions_compacted", False))
    interaction_ids = (
        [int(value) for value in _sequence(stage.get("interaction_ids", []))]
        if interactions_compacted
        else [
            int(_mapping(item)["interaction_id"])
            for item in _sequence(stage["interactions"])
        ]
    )
    if len(interaction_ids) != int(stage["interaction_count"]):
        raise ValueError("runtime stage interaction count is inconsistent")
    return {
        **_select(
            stage,
            "stage_index",
            "stage_kind",
            "subset_size",
            "input_current_ids",
            "output_current_ids",
            "input_value_slot_ids",
            "output_value_slot_ids",
            "interaction_count",
        ),
        "interactions_compacted": True,
        "interaction_ids": interaction_ids,
        "interactions": [],
    }


def _amplitude_stage(stage: Mapping[str, object]) -> dict[str, object]:
    contraction = stage.get("color_contraction")
    return {
        "stage_kind": str(stage["stage_kind"]),
        "output_count": int(stage["output_count"]),
        "color_contraction": (
            None if contraction is None else _color_contraction(_mapping(contraction))
        ),
        "roots": [
            _select(
                _mapping(root),
                "output_index",
                "root_id",
                "kind",
                "left_current_id",
                "right_current_id",
                "left_slot",
                "right_slot",
                "left_value_slot",
                "right_value_slot",
                "vertex_kind",
                "vertex_particles",
                "coupling",
                "color_weight",
                "color_sector_id",
                "contraction",
                "contraction_ir",
                "coherent_group_id",
                "helicity_weight",
                "all_sector_weight",
            )
            for root in _sequence(stage["roots"])
        ],
    }


def _color_contraction(record: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        **_select(record, "supported", "reason", "group_count"),
        "includes_color_factor": bool(record.get("includes_color_factor", False)),
        "entries": [
            _select(
                _mapping(item),
                "left_group_id",
                "right_group_id",
                "weight",
                "symmetry_factor",
            )
            for item in _sequence(record["entries"])
        ],
    }
    repeated_block = record.get("repeated_block")
    if repeated_block is not None:
        repeated = _mapping(repeated_block)
        compact: dict[str, object] = {
            "component_count": int(repeated["component_count"]),
            "component_group_ids": [
                int(value) for value in _sequence(repeated["component_group_ids"])
            ],
            "entries": [
                _select(
                    _mapping(item),
                    "left_group_index",
                    "right_group_index",
                    "weight",
                    "symmetry_factor",
                )
                for item in _sequence(repeated["entries"])
            ],
        }
        factorized_block = repeated.get("factorized_block")
        if factorized_block is not None:
            factorized = _mapping(factorized_block)
            compact["factorized_block"] = {
                "kind": str(factorized["kind"]),
                "cosets": [
                    [int(value) for value in _sequence(coset)]
                    for coset in _sequence(factorized["cosets"])
                ],
            }
        result["repeated_block"] = compact
    return result


def _stage_evaluator_set(record: Mapping[str, object]) -> dict[str, object]:
    result = {
        **_select(
            record,
            "kind",
            "runtime_available",
            "runtime_unavailable_message",
            "parameter_count",
            "value_parameter_count",
            "momentum_parameter_count",
            "model_parameter_count",
            "real_valued_inputs",
            "parameter_layout",
            "stage_count",
            "required_runtime_capabilities",
        ),
        "stages": [
            _serialized_stage(_mapping(item)) for item in _sequence(record["stages"])
        ],
        "amplitude_stage": _serialized_stage(_mapping(record["amplitude_stage"])),
    }
    evaluator_manifests = [
        _mapping(_mapping(stage)["evaluator"])
        for stage in (*_sequence(result["stages"]), result["amplitude_stage"])
    ]
    actual = tuple(
        sorted(
            {
                capability
                for manifest in evaluator_manifests
                for capability in evaluator_runtime_capabilities(manifest)
            }
        )
    )
    declared = set(_required_runtime_capabilities(result))
    evaluator_capabilities = declared - {COMPILED_RUNTIME_SELECTORS_CAPABILITY}
    if set(actual) != evaluator_capabilities:
        raise ValueError(
            "stage evaluator runtime capabilities do not match evaluator payloads"
        )
    return result


def _serialized_stage(record: Mapping[str, object]) -> dict[str, object]:
    return {
        **_select(
            record,
            "stage_index",
            "stage_kind",
            "subset_size",
            "evaluator_label",
            "parameter_layout",
            "output_length",
            "output_slots",
            "input_value_slot_ids",
            "output_value_slot_ids",
            "interaction_ids",
            "input_components",
            "parameter_count",
            "value_parameter_count",
            "momentum_parameter_count",
            "model_parameter_count",
            "real_valued_inputs",
            "expression_ready",
            "blockers",
        ),
        "evaluator": _evaluator(_mapping(record["evaluator"])),
    }


def _model_parameter_evaluator(record: Mapping[str, object]) -> dict[str, object]:
    result = {
        **_select(
            record,
            "kind",
            "required_runtime_capabilities",
            "input_parameter_indices",
            "outputs",
        ),
        "evaluator": _evaluator(_mapping(record["evaluator"])),
    }
    if evaluator_runtime_capabilities(_mapping(result["evaluator"])) != (
        _required_runtime_capabilities(result)
    ):
        raise ValueError(
            "model-parameter evaluator runtime capabilities do not match its payload"
        )
    return result


def _evaluator(record: Mapping[str, object]) -> dict[str, object]:
    kind = str(record.get("kind", ""))
    if kind == "symjit-application-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "input_len",
            "output_len",
            "application_path",
            "application_abi",
            "element_layout",
            "batch_layout",
            "compiler_type",
            "translation_mode",
            "optimization_level",
            "word_bits",
            "endianness",
            "required_defuns",
            "evaluator_state_path",
            "evaluator_state_runtime_capability",
        )
        if result["runtime_capability"] != SYMJIT_F64_RUNTIME_CAPABILITY:
            raise ValueError(
                "direct SymJIT evaluator has an invalid runtime capability"
            )
        if result["application_abi"] != SYMJIT_APPLICATION_ABI:
            raise ValueError(
                "direct SymJIT evaluator has an incompatible application ABI"
            )
        if (
            result["evaluator_state_runtime_capability"]
            != SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
        ):
            raise ValueError(
                "direct SymJIT evaluator has an invalid fallback capability"
            )
        if result["element_layout"] != "complex-f64":
            raise ValueError("direct SymJIT evaluator has an invalid element layout")
        if (
            result["batch_layout"] != "row-major"
            or result["compiler_type"] != "native"
            or result["word_bits"] != 64
            or result["endianness"] != "little"
            or result["required_defuns"] != []
        ):
            raise ValueError("direct SymJIT evaluator has invalid execution metadata")
        if result["translation_mode"] not in {"direct", "indirect"}:
            raise ValueError("direct SymJIT evaluator has an invalid translation mode")
        optimization_level = result["optimization_level"]
        if (
            isinstance(optimization_level, bool)
            or not isinstance(optimization_level, int)
            or optimization_level not in {0, 1, 2, 3}
        ):
            raise ValueError(
                "direct SymJIT evaluator has an invalid optimization level"
            )
        return result
    if kind == "jit-symbolica-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "input_len",
            "output_len",
            "evaluator_state_path",
        )
        if result["runtime_capability"] != SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY:
            raise ValueError("legacy JIT evaluator has an invalid runtime capability")
        return result
    if kind == "compiled-complex-evaluator":
        result = _select(
            record,
            "kind",
            "runtime_capability",
            "function_name",
            "input_len",
            "output_len",
            "library_path",
            "evaluator_state_path",
            "number_type",
        )
        if result["runtime_capability"] not in {
            SYMBOLICA_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        }:
            raise ValueError("compiled evaluator has an invalid runtime capability")
        return result
    if kind == "chunked-symbolica-evaluator":
        result = {
            "kind": kind,
            "input_len": record["input_len"],
            "chunk_input_indices": [
                list(_sequence(indices))
                for indices in _sequence(record["chunk_input_indices"])
            ],
            "required_runtime_capabilities": list(
                _required_runtime_capabilities(record)
            ),
            "chunks": [
                _evaluator(_mapping(item)) for item in _sequence(record["chunks"])
            ],
        }
        actual = evaluator_runtime_capabilities(result)
        if actual != _required_runtime_capabilities(record):
            raise ValueError(
                "chunked evaluator required runtime capabilities do not match chunks"
            )
        return result
    raise ValueError(f"unsupported evaluator artifact kind {kind!r}")


def _dag_summary(record: Mapping[str, object]) -> dict[str, object]:
    return _select(
        record,
        "current_count",
        "source_count",
        "interaction_count",
        "amplitude_root_count",
        "truncated",
    )


def _select(record: Mapping[str, object], *names: str) -> dict[str, object]:
    missing = [name for name in names if name not in record]
    if missing:
        raise ValueError(
            "runtime execution record is missing fields: " + ", ".join(missing)
        )
    return {name: record[name] for name in names}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("runtime execution field must be a sequence")
    return value


def _copy_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    root: Path,
    *,
    prefix: str,
    process_id: str,
) -> None:
    source_root = root.expanduser().resolve(strict=True)
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        evaluator_payloads.add_file(
            f"{prefix}/{relative}",
            path,
            process_id=process_id,
        )


def _copy_color_selector_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
    *,
    prefix: str,
    process_id: str,
) -> None:
    for record in _ordered_color_selector_executions(executions):
        _copy_evaluator_payloads(
            evaluator_payloads,
            record.execution.evaluator_root,
            prefix=(
                f"{prefix}/{_color_selector_payload_prefix(record.materialized_sector_id)}"
            ),
            process_id=process_id,
        )


def _copy_helicity_selector_evaluator_payloads(
    evaluator_payloads: _EvaluatorPayloadCollector,
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
    *,
    prefix: str,
    process_id: str,
) -> None:
    for lane_index, record in enumerate(
        _ordered_helicity_selector_executions(executions)
    ):
        lane_prefix = (
            f"{prefix}/{_HELICITY_SELECTOR_UNION_PAYLOAD_ROOT}/class-{lane_index}"
        )
        _copy_evaluator_payloads(
            evaluator_payloads,
            record.execution.evaluator_root,
            prefix=lane_prefix,
            process_id=process_id,
        )
        _copy_helicity_selector_evaluator_payloads(
            evaluator_payloads,
            record.execution.helicity_selector_executions,
            prefix=lane_prefix,
            process_id=process_id,
        )


def _color_selector_payload_prefix(materialized_sector_id: int) -> str:
    if materialized_sector_id < 0:
        raise ValueError("materialized colour-sector ids must be non-negative")
    return f"{_COLOR_SELECTOR_PAYLOAD_ROOT}/sector-{materialized_sector_id}"


def _ordered_color_selector_executions(
    executions: Sequence[CompiledColorSelectorExecutionArtifact],
) -> tuple[CompiledColorSelectorExecutionArtifact, ...]:
    ordered = tuple(
        sorted(executions, key=lambda record: record.materialized_sector_id)
    )
    sector_ids = tuple(record.materialized_sector_id for record in ordered)
    if len(sector_ids) != len(set(sector_ids)):
        raise ValueError("compiled colour-selector lane ids must be unique")
    for record in ordered:
        if record.materialized_sector_id < 0:
            raise ValueError("materialized colour-sector ids must be non-negative")
        if record.execution.color_selector_executions:
            raise ValueError("compiled colour-selector execution lanes cannot nest")
    return ordered


def _ordered_helicity_selector_executions(
    executions: Sequence[CompiledHelicitySelectorExecutionArtifact],
) -> tuple[CompiledHelicitySelectorExecutionArtifact, ...]:
    ordered = tuple(sorted(executions, key=lambda record: record.selector_domain_ids))
    seen: set[int] = set()
    for record in ordered:
        domain_ids = tuple(sorted(set(record.selector_domain_ids)))
        if not domain_ids or domain_ids != record.selector_domain_ids:
            raise ValueError(
                "compiled helicity-selector domain ids must be non-empty, "
                "sorted, and unique"
            )
        overlap = seen.intersection(domain_ids)
        if overlap:
            raise ValueError(
                "compiled helicity-selector lanes overlap selector domains: "
                + ", ".join(str(item) for item in sorted(overlap))
            )
        seen.update(domain_ids)
        if record.schedule_mode not in {"parent-closure", "nested-runtime"}:
            raise ValueError(
                "compiled helicity-selector schedule mode must be "
                "'parent-closure' or 'nested-runtime'"
            )
        if record.execution.color_selector_executions:
            raise ValueError("compiled helicity-selector execution lanes cannot nest")
        children = record.execution.helicity_selector_executions
        if children:
            if record.schedule_mode != "nested-runtime":
                raise ValueError(
                    "only a nested-runtime helicity-selector execution may "
                    "contain closure lanes"
                )
            for child in _ordered_helicity_selector_executions(children):
                if (
                    child.schedule_mode != "parent-closure"
                    or child.execution.helicity_selector_executions
                    or child.execution.color_selector_executions
                ):
                    raise ValueError(
                        "nested helicity-selector closure lanes must be "
                        "terminal parent-closure executions"
                    )
    return ordered


def build_api_validation_points(
    processes: Sequence[ProcessArtifact],
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    """Return concrete and crossing-alias points in each public external order."""

    points: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    for process in processes:
        vectors = process.validation_point.four_vectors
        if not vectors:
            continue
        points[process.process_id] = vectors
        _add_alias_points(points, vectors=vectors, aliases=process.aliases)
    return points


def _existing_bundle_points(
    existing: ArtifactManifest | None,
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    if existing is None:
        return {}
    points: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    for process in existing.processes:
        process_id = str(process["id"])
        validation_path = (
            existing.root / f"processes/{process_id}/validation-momenta.json"
        )
        if not validation_path.is_file():
            continue
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        raw_points = payload.get("points") if isinstance(payload, Mapping) else None
        if not isinstance(raw_points, list) or not raw_points:
            continue
        raw_point = raw_points[0]
        if not isinstance(raw_point, list):
            raise ValueError(f"validation point for {process_id!r} is invalid")
        vectors = tuple(_validation_four_vector(item) for item in raw_point)
        points[process_id] = vectors
        aliases = process.get("aliases")
        if not isinstance(aliases, Sequence):
            raise ValueError(f"aliases for {process_id!r} are invalid")
        _add_alias_points(
            points,
            vectors=vectors,
            aliases=tuple(_mapping(alias) for alias in aliases),
        )
    return points


def _add_alias_points(
    points: dict[str, tuple[tuple[float, float, float, float], ...]],
    *,
    vectors: tuple[tuple[float, float, float, float], ...],
    aliases: Sequence[Mapping[str, object]],
) -> None:
    for alias in aliases:
        alias_id = str(alias["id"])
        raw_permutation = alias.get("external_permutation", ())
        if not isinstance(raw_permutation, Sequence):
            raise ValueError(f"alias {alias_id!r} permutation is invalid")
        permutation = tuple(int(index) for index in raw_permutation)
        if not permutation:
            permutation = tuple(range(len(vectors)))
        if sorted(permutation) != list(range(len(vectors))):
            raise ValueError(
                f"alias {alias_id!r} permutation does not match its external momenta"
            )
        if alias_id in points:
            raise ValueError(f"duplicate validation-point ID {alias_id!r}")
        alias_vectors: list[tuple[float, float, float, float] | None] = [
            None for _ in vectors
        ]
        for representative_index, alias_index in enumerate(permutation):
            alias_vectors[alias_index] = vectors[representative_index]
        if any(vector is None for vector in alias_vectors):
            raise ValueError(f"alias {alias_id!r} permutation is incomplete")
        points[alias_id] = tuple(
            vector for vector in alias_vectors if vector is not None
        )


def _validation_four_vector(
    record: object,
) -> tuple[float, float, float, float]:
    raw = _mapping(record).get("momentum")
    if not isinstance(raw, Sequence) or len(raw) != 4:
        raise ValueError("serialized validation momentum is not a four-vector")
    values = tuple(float(component) for component in raw)
    return values[0], values[1], values[2], values[3]


def _producer_metadata(config: GenerationConfig | RunConfig) -> dict[str, object]:
    version = package_version()
    target, c_abi = _target_metadata(config)
    return {
        "distribution": "pyamplicol",
        "version": version,
        "versions": {
            "python_api": PYTHON_API_VERSION,
            "toml": TOML_SCHEMA_VERSION,
            "compiled_model": COMPILED_MODEL_SCHEMA_VERSION,
            "process_artifact": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "runtime_physics": RUNTIME_PHYSICS_SCHEMA_VERSION,
            "symbolica_serialization": SYMBOLICA_SERIALIZATION_ABI,
            "c_abi": c_abi,
        },
        "target": target,
    }


def _target_metadata(
    config: GenerationConfig | RunConfig,
) -> tuple[dict[str, object], int]:
    requires_native_features = (
        isinstance(config, RunConfig)
        and str(config.evaluator.backend) != "jit"
        and config.evaluator.cpp.native_arch
    )
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
        info = rusticol.target_info()
        triple = str(info.triple)
        available_features = tuple(str(item) for item in info.cpu_features)
        if available_features != tuple(sorted(set(available_features))):
            raise RuntimeError(
                "Rusticol returned non-canonical target CPU feature metadata"
            )
        c_abi = int(rusticol.abi_version())
    except (
        AttributeError,
        ImportError,
        OSError,
        importlib.metadata.PackageNotFoundError,
    ) as rusticol_error:
        try:
            from pyamplicol._sdk import load_sdk_info

            sdk = load_sdk_info()
        except (
            ImportError,
            OSError,
            RuntimeError,
            importlib.metadata.PackageNotFoundError,
        ) as sdk_error:
            raise RuntimeError(
                "Rusticol target metadata is unavailable; install or build the "
                "pyamplicol native extension before generating process artifacts"
            ) from sdk_error
        if requires_native_features:
            raise RuntimeError(
                "native C++ evaluator generation requires Rusticol CPU-feature "
                "introspection; SDK target metadata alone is insufficient"
            ) from rusticol_error
        triple = sdk.target
        available_features = ()
        c_abi = int(sdk.abi_version)
    required_features = available_features if requires_native_features else ()
    if triple not in _SUPPORTED_ARTIFACT_TARGETS:
        raise RuntimeError(
            f"Rusticol process artifacts are not supported on target {triple!r}"
        )
    if requires_native_features and not required_features:
        raise RuntimeError(
            "Rusticol did not detect any CPU features for a native C++ evaluator; "
            "refusing to emit incomplete target metadata"
        )
    return {
        "triple": triple,
        "cpu_features": list(required_features),
    }, c_abi


def _model_metadata(
    source: ModelSource,
    compiled: CompiledModel,
) -> dict[str, object]:
    source_kind = {
        "built-in-sm": "built-in-sm",
        "ufo": "ufo",
        "json": "ufo-json",
        "compiled": "compiled-model",
        "prepared": "compiled-model",
    }[source.kind]
    digest = str(compiled.source.get("digest", ""))
    if source.kind == "compiled" and source.path is not None:
        digest = hashlib.sha256(source.path.read_bytes()).hexdigest()
    if len(digest) != 64:
        raise ValueError("compiled model source has no canonical SHA-256 digest")
    restriction = (
        (
            source.restriction.name
            if isinstance(source.restriction, Path)
            else source.restriction
        )
        if source.restriction is not None
        else "default"
        if source.kind in {"ufo", "json"}
        else None
    )
    return {
        "name": compiled.name,
        "source_kind": source_kind,
        "content_sha256": digest,
        "compiled_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
        "restriction": restriction,
    }


def _dependency_metadata(source: ModelSource) -> tuple[dict[str, object], ...]:
    dependencies: list[dict[str, object]] = [
        {
            "name": "symbolica",
            "version": _distribution_version("symbolica", "unknown"),
            "source": "https://symbolica.io/",
            "license": "Symbolica Software License Agreement",
        }
    ]
    if source.kind in {"ufo", "json"}:
        dependencies.append(
            {
                "name": "ufo-model-loader",
                "version": _distribution_version("ufo-model-loader", "unknown"),
                "source": "https://github.com/alphal00p/ufo_model_loader",
                "license": "MIT",
            }
        )
    return tuple(dependencies)


def _validate_append_compatibility(
    existing: ArtifactManifest | None,
    *,
    producer: Mapping[str, object],
    model: Mapping[str, object],
    eager_pack_identity: Mapping[str, object] | None,
    requested_bytes: bytes,
    effective_bytes: bytes,
    adjustments: Sequence[Mapping[str, str]],
    processes: Sequence[ProcessArtifact],
) -> None:
    if existing is None:
        return
    existing_uses_eager = EAGER_RUNTIME_CAPABILITY in _required_runtime_capabilities(
        existing.runtime
    ) or any(record.path == _EAGER_KERNEL_PACK_PATH for record in existing.payloads)
    existing_identity = existing.extensions.get(_EAGER_PACK_IDENTITY_EXTENSION)
    if existing_uses_eager or existing_identity is not None:
        if not isinstance(existing_identity, Mapping):
            raise ValueError(
                "append eager artifact has no canonical prepared-pack identity; "
                "replace the artifact"
            )
        if eager_pack_identity is None or _plain_mapping(
            existing_identity
        ) != _plain_mapping(eager_pack_identity):
            raise ValueError(
                "append prepared kernel pack identity differs from the existing "
                "artifact; replace the artifact"
            )
    if _plain_mapping(existing.model) != dict(model):
        raise ValueError("append model provenance differs from the existing artifact")
    if _plain_mapping(existing.producer).get("target") != producer.get("target"):
        raise ValueError("append target differs from the existing artifact")
    requested_path = existing.root / str(existing.configuration["requested_path"])
    effective_path = existing.root / str(existing.configuration["effective_path"])
    if requested_path.read_bytes() != requested_bytes:
        raise ValueError("append requested configuration differs from the artifact")
    if effective_path.read_bytes() != effective_bytes:
        raise ValueError("append effective configuration differs from the artifact")
    existing_adjustments: list[dict[str, str]] = []
    for item in _sequence(existing.configuration["adjustments"]):
        adjustment = _mapping(item)
        existing_adjustments.append(
            {
                "path": str(adjustment["path"]),
                "reason": str(adjustment["reason"]),
            }
        )
    if existing_adjustments != list(adjustments):
        raise ValueError("append configuration adjustments differ from the artifact")
    existing_ids = {str(record["id"]) for record in existing.processes}
    duplicates = existing_ids.intersection(process.process_id for process in processes)
    if duplicates:
        raise ValueError(
            "append process IDs already exist: " + ", ".join(sorted(duplicates))
        )


def _existing_process_records(
    existing: ArtifactManifest | None,
) -> list[dict[str, object]]:
    if existing is None:
        return []
    return [_plain_mapping(record) for record in existing.processes]


def _existing_evaluator_entries(
    existing: ArtifactManifest | None,
) -> list[dict[str, object]]:
    if existing is None:
        return []
    path = existing.root / str(existing.runtime["evaluator_manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROCESS_ARTIFACT_SCHEMA_VERSION
        or payload.get("kind") != "pyamplicol-runtime-execution-set"
    ):
        raise ValueError("append artifact has an incompatible evaluator-set manifest")
    entries = payload.get("processes")
    if not isinstance(entries, list) or not all(
        isinstance(item, dict) for item in entries
    ):
        raise ValueError("append evaluator-set process list is invalid")
    return [dict(item) for item in entries]


def _existing_required_runtime_capabilities(
    existing: ArtifactManifest | None,
) -> tuple[str, ...]:
    if existing is None:
        return ()
    return _required_runtime_capabilities(existing.runtime)


def _compiled_process_runtime_capabilities(
    process: CompiledProcessArtifact,
) -> tuple[str, ...]:
    capabilities = set(_required_runtime_capabilities(process.stage_manifest))
    if _runtime_schema_uses_walsh_color_contraction(
        _runtime_schema_mapping(process.runtime_schema)
    ):
        capabilities.add(COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY)
    if process.model_parameter_evaluator is not None:
        capabilities.update(
            _required_runtime_capabilities(process.model_parameter_evaluator)
        )
    if _uses_primary_helicity_recurrence(process):
        capabilities.add(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
    if process.color_selector_executions:
        capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in _ordered_color_selector_executions(
            process.color_selector_executions
        ):
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    auxiliary = process.helicity_sum_execution
    if auxiliary is not None:
        capabilities.add(COMPILED_HELICITY_DUAL_LANE_CAPABILITY)
        capabilities.update(_compiled_execution_runtime_capabilities(auxiliary))
    selector_lanes = _ordered_helicity_selector_executions(
        process.helicity_selector_executions
    )
    if selector_lanes:
        capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in selector_lanes:
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    return tuple(sorted(capabilities))


def _uses_primary_helicity_recurrence(process: CompiledProcessArtifact) -> bool:
    return _runtime_schema_uses_primary_helicity_recurrence(
        _runtime_schema_mapping(process.runtime_schema),
        has_helicity_sum_execution=process.helicity_sum_execution is not None,
    )


def _runtime_schema_uses_primary_helicity_recurrence(
    runtime_schema: Mapping[str, object],
    *,
    has_helicity_sum_execution: bool,
) -> bool:
    if has_helicity_sum_execution:
        return False
    recurrence = runtime_schema.get("helicity_recurrence")
    return isinstance(recurrence, Mapping) and isinstance(
        recurrence.get("materialization"), Mapping
    )


def _runtime_schema_uses_walsh_color_contraction(
    runtime_schema: Mapping[str, object],
) -> bool:
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        return False
    contraction = amplitude_stage.get("color_contraction")
    if not isinstance(contraction, Mapping):
        return False
    repeated_block = contraction.get("repeated_block")
    return isinstance(repeated_block, Mapping) and (
        repeated_block.get("factorized_block") is not None
    )


def _compiled_execution_runtime_capabilities(
    execution: CompiledExecutionArtifact,
) -> tuple[str, ...]:
    capabilities = set(_required_runtime_capabilities(execution.stage_manifest))
    if _runtime_schema_uses_walsh_color_contraction(
        _runtime_schema_mapping(execution.runtime_schema)
    ):
        capabilities.add(COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY)
    if _runtime_schema_uses_primary_helicity_recurrence(
        _runtime_schema_mapping(execution.runtime_schema),
        has_helicity_sum_execution=False,
    ):
        capabilities.add(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
    if execution.model_parameter_evaluator is not None:
        capabilities.update(
            _required_runtime_capabilities(execution.model_parameter_evaluator)
        )
    if execution.color_selector_executions:
        capabilities.add(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in _ordered_color_selector_executions(
            execution.color_selector_executions
        ):
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    selector_lanes = _ordered_helicity_selector_executions(
        execution.helicity_selector_executions
    )
    if selector_lanes:
        capabilities.add(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY)
        capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        for record in selector_lanes:
            capabilities.update(
                _compiled_execution_runtime_capabilities(record.execution)
            )
    return tuple(sorted(capabilities))


def _process_runtime_capabilities(
    process: ProcessArtifact,
) -> tuple[str, ...]:
    if isinstance(process, EagerPlanV3ProcessArtifact):
        return (EAGER_PLAN_V3_RUNTIME_CAPABILITY,)
    if isinstance(process, EagerProcessArtifact):
        return _eager_process_runtime_capabilities(process)
    return _compiled_process_runtime_capabilities(process)


def _eager_process_runtime_capabilities(
    process: EagerProcessArtifact,
) -> tuple[str, ...]:
    capabilities = {EAGER_RUNTIME_CAPABILITY}
    runtime_schema = _runtime_schema_mapping(process.runtime_schema)
    if runtime_schema.get("lc_topology_replay") is not None:
        capabilities.add(EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY)
    return tuple(sorted(capabilities))


def _required_runtime_capabilities(
    record: Mapping[str, object],
) -> tuple[str, ...]:
    raw = record.get("required_runtime_capabilities")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError("runtime capability metadata must be a sequence")
    values = tuple(str(item) for item in raw)
    if values != tuple(sorted(set(values))):
        raise ValueError("runtime capabilities must be sorted and unique")
    unknown = set(values) - (
        EVALUATOR_RUNTIME_CAPABILITIES | {EAGER_PLAN_V3_RUNTIME_CAPABILITY}
    )
    if unknown:
        raise ValueError(
            "unsupported evaluator runtime capabilities: " + ", ".join(sorted(unknown))
        )
    return values


def _extensions(
    existing: ArtifactManifest | None,
    *,
    processes: Sequence[ProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_requested: bool,
    api_bundle_path: str | None,
    eager_pack_identity: Mapping[str, object] | None,
    execution_manifest_sha256_by_process: Mapping[str, str],
    evaluator_payload_container: Mapping[str, object] | None,
) -> dict[str, object]:
    result = {} if existing is None else _plain_mapping(existing.extensions)
    previous = result.get("generation")
    generation = dict(previous) if isinstance(previous, Mapping) else {}
    concrete = generation.get("concrete_processes")
    process_records = list(concrete) if isinstance(concrete, list) else []
    for process in processes:
        record: dict[str, object] = {
            "id": process.process_id,
            "expression": process.expression,
            "validation_momenta_path": (
                f"processes/{process.process_id}/validation-momenta.json"
            ),
            "filters": dict(process.generation_filters),
        }
        if _is_eager_process(process):
            record["execution_manifest_sha256"] = execution_manifest_sha256_by_process[
                process.process_id
            ]
        else:
            record["runtime_schema_sha256"] = process.runtime_schema.sha256
            if process.helicity_sum_execution is not None:
                record["helicity_sum_runtime_schema_sha256"] = (
                    process.helicity_sum_execution.runtime_schema.sha256
                )
            if process.helicity_selector_executions:
                record["helicity_selector_runtime_schema_sha256s"] = [
                    lane.execution.runtime_schema.sha256
                    for lane in _ordered_helicity_selector_executions(
                        process.helicity_selector_executions
                    )
                ]
        process_records.append(record)
    generation.update(
        {
            "schema_version": 1,
            "concrete_processes": process_records,
            "phase_timings_seconds": {
                str(name): float(value) for name, value in timings.items()
            },
            "api_bundle": {
                "requested": api_bundle_requested,
                "emitted": api_bundle_path is not None,
                "path": api_bundle_path,
                "scope": "root-artifact",
            },
        }
    )
    result["generation"] = generation
    if eager_pack_identity is not None:
        result[_EAGER_PACK_IDENTITY_EXTENSION] = _plain_mapping(eager_pack_identity)
    if evaluator_payload_container is None:
        result.pop(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION, None)
    else:
        result[_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION] = _plain_mapping(
            evaluator_payload_container
        )
    return result


def _default_api_bundle_hook() -> ApiBundleHook | None:
    try:
        from pyamplicol.artifacts.api_bundle import emit_api_bundle
    except ImportError:
        return None
    return cast("ApiBundleHook", emit_api_bundle)


def _call_api_bundle_hook(
    builder: ArtifactBuilder,
    hook: ApiBundleHook,
    validation_points: Mapping[str, Sequence[Sequence[float]]],
) -> str:
    result = hook(builder, validation_points)
    paths = tuple(
        str(item["path"])
        if isinstance(item, Mapping)
        else str(getattr(item, "path", ""))
        for item in result
    )
    if not paths or any(not path.startswith("API/") for path in paths):
        raise ValueError("root API-bundle emitter returned an invalid payload set")
    return "API"


def _validate_artifact_references(manifest: ArtifactManifest) -> None:
    declared = {record.path for record in manifest.payloads}
    _validate_evaluator_payload_container(manifest)
    required = {
        str(manifest.configuration["requested_path"]),
        str(manifest.configuration["effective_path"]),
        str(manifest.runtime["evaluator_manifest_path"]),
        *(str(process["physics_path"]) for process in manifest.processes),
        *(
            f"processes/{process['id']}/execution.json"
            for process in manifest.processes
        ),
    }
    api_path = manifest.runtime.get("api_bundle_path")
    if api_path is not None:
        api_prefix = str(api_path).rstrip("/") + "/"
        if not any(path.startswith(api_prefix) for path in declared):
            raise ValueError("artifact API bundle has no declared payloads")
    missing = required - declared
    if missing:
        raise ValueError(
            "artifact references undeclared payloads: " + ", ".join(sorted(missing))
        )
    actual = {
        path.relative_to(manifest.root).as_posix()
        for path in manifest.root.rglob("*")
        if path.is_file() and path.name != "artifact.json"
    }
    undeclared = actual - declared
    if undeclared:
        raise ValueError(
            "artifact contains undeclared files: " + ", ".join(sorted(undeclared))
        )


def _validate_evaluator_payload_container(manifest: ArtifactManifest) -> None:
    raw = manifest.extensions.get(_EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    if raw is None:
        return
    extension = _mapping(raw)
    expected_fields = {
        "kind",
        "schema_version",
        "storage_abi",
        "path",
        "member_count",
        "unpacked_size_bytes",
        "index_sha256",
    }
    if set(extension) != expected_fields:
        raise ValueError("evaluator payload container extension fields are invalid")
    if extension != {
        **extension,
        "kind": _EVALUATOR_PAYLOAD_CONTAINER_KIND,
        "schema_version": _EVALUATOR_PAYLOAD_CONTAINER_SCHEMA_VERSION,
        "storage_abi": _EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI,
        "path": _EVALUATOR_PAYLOAD_CONTAINER_PATH,
    }:
        raise ValueError("evaluator payload container extension contract is invalid")
    records = {
        record.path: record
        for record in manifest.payloads
        if record.path == _EVALUATOR_PAYLOAD_CONTAINER_PATH
    }
    if len(records) != 1:
        raise ValueError("evaluator payload container is not a declared payload")
    record = records[_EVALUATOR_PAYLOAD_CONTAINER_PATH]
    if (
        record.role != "evaluator-state"
        or record.media_type != "application/octet-stream"
        or record.process_id is not None
        or record.target != manifest.producer["target"]
    ):
        raise ValueError("evaluator payload container manifest record is invalid")
    loose = sorted(
        payload.path
        for payload in manifest.payloads
        if payload.path != _EVALUATOR_PAYLOAD_CONTAINER_PATH
        and _packed_evaluator_member_kind(payload.path) is not None
    )
    if loose:
        raise ValueError(
            "artifact declares loose packed evaluator payloads: " + ", ".join(loose)
        )
    with PacbinReader.open(
        manifest.root / _EVALUATOR_PAYLOAD_CONTAINER_PATH,
        verify_payloads=False,
    ) as reader:
        expected = _evaluator_payload_container_extension(reader.index)
    if extension != expected:
        raise ValueError(
            "evaluator payload container metadata does not match its index"
        )


def _config_payload(config: GenerationConfig | RunConfig) -> dict[str, object]:
    payload = _plain_mapping(_mapping(config_to_dict(config)))
    if isinstance(config, GenerationConfig):
        return {"schema_version": 1, "generation": payload}
    return payload


def _effective_config_payload(
    requested: Mapping[str, object],
    *,
    disable_api_bundle: bool,
) -> dict[str, object]:
    result = _deep_plain(requested)
    if disable_api_bundle:
        generation = result.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("generation configuration section is missing")
        generation["emit_api_bundle"] = False
    return result


def _bundle_requested(config: GenerationConfig | RunConfig) -> bool:
    generation = config if isinstance(config, GenerationConfig) else config.generation
    return bool(generation.emit_api_bundle)


def _existing_bundle_path(existing: ArtifactManifest | None) -> str | None:
    if existing is None:
        return None
    value = existing.runtime.get("api_bundle_path")
    return None if value is None else str(value)


def _toml_bytes(payload: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    _write_toml_table(lines, (), payload, emit_header=False)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _write_toml_table(
    lines: list[str],
    path: tuple[str, ...],
    payload: Mapping[str, object],
    *,
    emit_header: bool,
) -> None:
    scalars = [
        (str(key), value)
        for key, value in payload.items()
        if value is not None and not isinstance(value, Mapping)
    ]
    tables = [
        (str(key), _mapping(value))
        for key, value in payload.items()
        if isinstance(value, Mapping)
    ]
    if emit_header:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
    lines.extend(f"{_toml_key(key)} = {_toml_value(value)}" for key, value in scalars)
    for key, table in tables:
        _write_toml_table(lines, (*path, key), table, emit_header=True)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, os.PathLike):
        return json.dumps(os.fspath(value), ensure_ascii=True)
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Mapping):
        entries = (
            f"{_toml_key(str(key))} = {_toml_value(entry)}"
            for key, entry in value.items()
            if entry is not None
        )
        return "{ " + ", ".join(entries) + " }"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"configuration value is not TOML serializable: {value!r}")


def _toml_key(value: str) -> str:
    return value if _SAFE_TOML_KEY.fullmatch(value) else json.dumps(value)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return "application/json"
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
        return "text/x-c++src"
    if suffix == ".symjit":
        return "application/vnd.symjit.application"
    return "application/octet-stream"


def _distribution_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("artifact metadata value must be an object")
    return {str(key): item for key, item in value.items()}


def _plain_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return _deep_plain(value)


def _deep_plain(value: Mapping[str, object]) -> dict[str, object]:
    def convert(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): convert(entry) for key, entry in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            return [convert(entry) for entry in item]
        return item

    return {str(key): convert(item) for key, item in value.items()}


__all__ = [
    "EAGER_PLAN_V3_ABI",
    "EAGER_PLAN_V3_RUNTIME_CAPABILITY",
    "EAGER_RUNTIME_CONTAINER_KIND",
    "EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION",
    "EAGER_RUNTIME_LAYOUT_ABI",
    "EAGER_RUNTIME_STORAGE_ABI",
    "ApiBundleHook",
    "ArtifactWriteResult",
    "CompiledExecutionArtifact",
    "CompiledProcessArtifact",
    "EagerPlanV3ProcessArtifact",
    "EagerProcessArtifact",
    "ProcessArtifact",
    "build_api_validation_points",
    "write_schema_v3_artifact",
]
