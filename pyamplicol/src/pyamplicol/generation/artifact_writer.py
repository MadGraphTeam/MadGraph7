# SPDX-License-Identifier: 0BSD
"""Transactional schema-v3 output for compiled concrete processes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import ArtifactBuilder, ArtifactManifest, load_manifest
from pyamplicol.config import (
    ConfigClamp,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
)

from .._internal.versions import (
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
)
from ..evaluators.execution_schema import evaluator_runtime_capabilities
from ..models.loading import COMPILED_MODEL_SCHEMA_VERSION, CompiledModel
from .contracts import RuntimeExpressionSchema
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
_SAFE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_ARTIFACT_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    }
)


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


def write_schema_v3_artifact(
    destination: str | Path,
    *,
    mode: Literal["error", "append", "replace"],
    source: ModelSource,
    compiled_model: CompiledModel,
    configuration: _GenerationConfigProvenance,
    processes: Sequence[CompiledProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_hook: ApiBundleHook | None = None,
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
    _validate_append_compatibility(
        existing,
        producer=producer,
        model=model,
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
    required_runtime_capabilities = set(
        _existing_required_runtime_capabilities(existing)
    )
    for process in processes:
        required_runtime_capabilities.update(
            _compiled_process_runtime_capabilities(process)
        )
    canonical_runtime_capabilities = tuple(sorted(required_runtime_capabilities))
    validation_records = tuple(process.validation_point for process in processes)
    validations = validation_point_map(validation_records)
    bundle_points = _existing_bundle_points(existing)
    bundle_points.update(build_api_validation_points(processes))
    api_bundle_path = existing_bundle
    write_mode = cast("ArtifactWriteMode", mode)
    with ArtifactBuilder(output, mode=write_mode) as builder:
        if existing is None:
            _write_global_payloads(
                builder,
                compiled_model=compiled_model,
                requested_bytes=requested_bytes,
                effective_bytes=effective_bytes,
            )
        for process in processes:
            record, evaluator_entry = _write_process_payloads(
                builder,
                process,
                target=target,
            )
            process_records.append(record)
            evaluator_entries.append(evaluator_entry)
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
        extensions = _extensions(
            existing,
            processes=processes,
            timings=timings,
            api_bundle_requested=bundle_requested,
            api_bundle_path=api_bundle_path,
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


def _write_process_payloads(
    builder: ArtifactBuilder,
    process: CompiledProcessArtifact,
    *,
    target: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    prefix = f"processes/{process.process_id}"
    physics_path = f"{prefix}/physics.json"
    execution_path = f"{prefix}/execution.json"
    validation_path = f"{prefix}/validation-momenta.json"
    schema = process.runtime_schema.to_mapping()
    physics = _mapping(schema.get("physics"))
    if physics.get("process_id") != process.process_id:
        raise ValueError(
            f"runtime physics process ID does not match {process.process_id!r}"
        )
    builder.add_json(
        physics_path,
        physics,
        role="runtime-physics",
        process_id=process.process_id,
        compact=True,
    )
    builder.add_json(
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
    _copy_evaluator_payloads(
        builder,
        process.evaluator_root,
        prefix=prefix,
        target=target,
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
                _compiled_process_runtime_capabilities(process)
            ),
            "aliases": [dict(alias) for alias in process.aliases],
        },
        {
            "process_id": process.process_id,
            "manifest_path": f"{process.process_id}/execution.json",
            "required_runtime_capabilities": list(
                _compiled_process_runtime_capabilities(process)
            ),
        },
    )


def _execution_manifest(
    process: CompiledProcessArtifact,
    compiler_schema: Mapping[str, object],
) -> dict[str, object]:
    model_parameter_evaluator = (
        None
        if process.model_parameter_evaluator is None
        else _model_parameter_evaluator(process.model_parameter_evaluator)
    )
    stage_evaluators = _stage_evaluator_set(process.stage_manifest)
    required_runtime_capabilities = set(
        _required_runtime_capabilities(stage_evaluators)
    )
    if model_parameter_evaluator is not None:
        required_runtime_capabilities.update(
            _required_runtime_capabilities(model_parameter_evaluator)
        )
    return {
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "process": process.expression,
        "key": process.process_id,
        "color_accuracy": process.color_accuracy,
        "external_pdg_order": list(process.external_pdgs),
        "compiled": {
            "kind": "generic-dag-stage-blueprint",
            "runtime_available": True,
            "runtime_unavailable_message": None,
            "model_parameter_evaluator": model_parameter_evaluator,
            "stage_evaluators": stage_evaluators,
        },
        "dag_summary": _dag_summary(process.dag_summary),
        "runtime_schema": _execution_plan(compiler_schema),
    }


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
            "identical_factor": float(
                normalization.get("identical_factor", 1.0)
            ),
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
    return {
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
    if actual != _required_runtime_capabilities(result):
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
    builder: ArtifactBuilder,
    root: Path,
    *,
    prefix: str,
    target: Mapping[str, object],
    process_id: str,
) -> None:
    source_root = root.expanduser().resolve(strict=True)
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        builder.add_file(
            f"{prefix}/{relative}",
            path,
            role="evaluator-state",
            media_type=_media_type(path),
            target=target,
            process_id=process_id,
        )


def build_api_validation_points(
    processes: Sequence[CompiledProcessArtifact],
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
    requested_bytes: bytes,
    effective_bytes: bytes,
    adjustments: Sequence[Mapping[str, str]],
    processes: Sequence[CompiledProcessArtifact],
) -> None:
    if existing is None:
        return
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
    if process.model_parameter_evaluator is not None:
        capabilities.update(
            _required_runtime_capabilities(process.model_parameter_evaluator)
        )
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
    unknown = set(values) - EVALUATOR_RUNTIME_CAPABILITIES
    if unknown:
        raise ValueError(
            "unsupported evaluator runtime capabilities: " + ", ".join(sorted(unknown))
        )
    return values


def _extensions(
    existing: ArtifactManifest | None,
    *,
    processes: Sequence[CompiledProcessArtifact],
    timings: Mapping[str, float],
    api_bundle_requested: bool,
    api_bundle_path: str | None,
) -> dict[str, object]:
    result = {} if existing is None else _plain_mapping(existing.extensions)
    previous = result.get("generation")
    generation = dict(previous) if isinstance(previous, Mapping) else {}
    concrete = generation.get("concrete_processes")
    process_records = list(concrete) if isinstance(concrete, list) else []
    process_records.extend(
        {
            "id": process.process_id,
            "expression": process.expression,
            "runtime_schema_sha256": process.runtime_schema.sha256,
            "validation_momenta_path": (
                f"processes/{process.process_id}/validation-momenta.json"
            ),
            "filters": dict(process.generation_filters),
        }
        for process in processes
    )
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
    "ApiBundleHook",
    "ArtifactWriteResult",
    "CompiledProcessArtifact",
    "build_api_validation_points",
    "write_schema_v3_artifact",
]
