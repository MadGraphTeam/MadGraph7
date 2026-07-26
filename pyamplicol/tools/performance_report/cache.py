# SPDX-License-Identifier: 0BSD
"""Canonical reset caches and strict compact measurement validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .catalog import REPORT_CATALOG, ReportCatalog
from .models import CellSpec, ResultStatus

CACHE_SCHEMA_VERSION = 3
REPORT_VERSION = "0.2.0"
_EXECUTION_TIMING_ABI = "pyamplicol-report-execution-timing-v1"
_COMPILED_ARENA_EXECUTION_TIME_SOURCE = (
    "runtime_profile_core_compiled_direct_arena_orchestration_time"
)
_LOADED_ORIGIN_OBSERVATION_FIELDS = frozenset(
    {
        "observed_module_count",
        "observations",
        "observations_sha256",
    }
)
_RUNTIME_POSTFLIGHT_FIELDS = frozenset(
    {
        "runtime_identity_sha256",
        "runtime_identity_stable_sha256",
        "runtime_identity_postflight_stable_sha256",
        "runtime_identity_postflight_loaded_module_origin_policy",
        "runtime_identity_postflight_match",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _stable_runtime_identity(identity: Mapping[str, object]) -> dict[str, object]:
    stable = dict(identity)
    raw_policy = stable.get("loaded_module_origin_policy")
    if isinstance(raw_policy, Mapping):
        stable["loaded_module_origin_policy"] = {
            field: value
            for field, value in raw_policy.items()
            if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
        }
    return stable


def _loaded_origin_policy(
    value: object,
    name: str,
) -> tuple[Mapping[str, object], list[object]]:
    policy = _required_mapping(value, name)
    observations = policy.get("observations")
    count = policy.get("observed_module_count")
    if (
        policy.get("kind") != "pyamplicol-loaded-module-origin-policy-v1"
        or policy.get("all_loaded_origins_authenticated") is not True
        or policy.get("native_image_origin_bound") is not True
        or policy.get("loaded_bytecode_eligible") is not False
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(observations, list)
        or len(observations) != count
        or policy.get("observations_sha256") != digest_json(observations)
    ):
        raise ValueError(f"{name} is not authenticated loaded-origin evidence")
    return policy, observations


def _validate_runtime_identity_postflight(
    provenance: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    """Require the initial/post-evaluation runtime identity binding."""

    raw_identity = provenance.get("runtime_identity")
    if raw_identity is None:
        if (
            provenance.get("method") == "original-amplicol-generated-library"
            or validation.get("method") == "independent-original-amplicol-oracle"
            or (
                "source_revision" not in provenance
                and not (_RUNTIME_POSTFLIGHT_FIELDS & provenance.keys())
            )
        ):
            return
        raise ValueError("successful pyAmpliCol measurement requires runtime_identity")
    identity = _required_mapping(raw_identity, "provenance.runtime_identity")
    if provenance.get("runtime_identity_sha256") != digest_json(identity):
        raise ValueError("provenance.runtime_identity_sha256 does not match")
    stable_digest = digest_json(_stable_runtime_identity(identity))
    if provenance.get("runtime_identity_stable_sha256") != stable_digest:
        raise ValueError("provenance.runtime_identity_stable_sha256 does not match")
    if provenance.get("runtime_identity_postflight_stable_sha256") != stable_digest:
        raise ValueError(
            "provenance.runtime_identity postflight stable SHA-256 differs"
        )
    if provenance.get("runtime_identity_postflight_match") is not True:
        raise ValueError("provenance.runtime_identity_postflight_match must be true")

    initial_policy, initial_observations = _loaded_origin_policy(
        identity.get("loaded_module_origin_policy"),
        "provenance.runtime_identity.loaded_module_origin_policy",
    )
    postflight_policy, postflight_observations = _loaded_origin_policy(
        provenance.get("runtime_identity_postflight_loaded_module_origin_policy"),
        ("provenance.runtime_identity_postflight_loaded_module_origin_policy"),
    )
    stable_initial_policy = {
        field: value
        for field, value in initial_policy.items()
        if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
    }
    stable_postflight_policy = {
        field: value
        for field, value in postflight_policy.items()
        if field not in _LOADED_ORIGIN_OBSERVATION_FIELDS
    }
    if stable_postflight_policy != stable_initial_policy:
        raise ValueError("provenance runtime postflight origin policy changed")
    postflight_keys = {_canonical_json(record) for record in postflight_observations}
    if any(
        _canonical_json(record) not in postflight_keys
        for record in initial_observations
    ):
        raise ValueError("provenance runtime postflight lost a loaded-module origin")


def empty_measurement() -> dict[str, object]:
    return {
        "status": ResultStatus.NOT_AVAILABLE.value,
        "generation_seconds": None,
        "wall_seconds_per_point": None,
        "execution_seconds_per_point": None,
        "matrix_element": None,
        "sample_count": None,
        "standard_error_seconds_per_point": None,
        "relative_standard_error": None,
        "artifact": None,
        "selector_contract": None,
        "validation": None,
        "resources": None,
        "provenance": None,
        "failure": None,
    }


def _measurement_spec_payload(cell: CellSpec) -> dict[str, object]:
    measurement = cell.measurement
    return {
        "execution_mode": measurement.execution_mode.value,
        "model": None if measurement.model is None else measurement.model.value,
        "accuracy": measurement.accuracy.value,
        "backend": measurement.backend,
        "jit_optimization_level": measurement.jit_optimization_level,
    }


def reset_entry(cell: CellSpec) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "variant": cell.variant,
        "workload": cell.workload.value,
        "measurement_spec": _measurement_spec_payload(cell),
        "measurement": empty_measurement(),
    }


def build_reset_cache(dataset_id: str, cells: Iterable[CellSpec]) -> dict[str, object]:
    selected = tuple(sorted(cells, key=lambda cell: cell.cell_id))
    if not selected:
        raise ValueError(f"dataset {dataset_id!r} has no cells")
    if any(cell.dataset_id != dataset_id for cell in selected):
        raise ValueError("cache cells must all belong to the requested dataset")
    descriptors = [_measurement_spec_payload(cell) for cell in selected]
    unique_descriptors = {
        json.dumps(descriptor, sort_keys=True) for descriptor in descriptors
    }
    return {
        "$schema": "./report-cache.schema.json",
        "schema_version": CACHE_SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
        "kind": "performance_measurements",
        "dataset_id": dataset_id,
        "measurement_specs": [
            json.loads(value) for value in sorted(unique_descriptors)
        ],
        "entries": [reset_entry(cell) for cell in selected],
    }


def build_reset_caches(
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[CellSpec]] = {}
    for cell in catalog.measurement_cells():
        grouped.setdefault(cell.dataset_id, []).append(cell)
    return {
        f"{dataset_id}.json": build_reset_cache(dataset_id, cells)
        for dataset_id, cells in sorted(grouped.items())
    }


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_number_or_none(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    number = float(value)
    if not number >= 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _validate_execution_timing(
    value: object,
    *,
    execution_seconds_per_point: float | None,
) -> None:
    timing = _required_mapping(value, "measurement.provenance.execution_timing")
    expected_fields = {
        "abi",
        "status",
        "ratio_eligible",
        "raw_seconds_per_point",
        "source",
        "compiled_direct_arena_active",
        "sample_count",
        "native_profile_points_per_sample",
        "sample_contract",
    }
    if set(timing) != expected_fields:
        raise ValueError(
            "measurement.provenance.execution_timing fields do not match contract"
        )
    raw = timing.get("raw_seconds_per_point")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) < 0.0
    ):
        raise ValueError("measurement.provenance.execution_timing raw time is invalid")
    sample_count = timing.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError(
            "measurement.provenance.execution_timing sample_count is invalid"
        )
    native_points = timing.get("native_profile_points_per_sample")
    if native_points is not None and (
        isinstance(native_points, bool)
        or not isinstance(native_points, int)
        or native_points < 1
    ):
        raise ValueError(
            "measurement.provenance.execution_timing native point count is invalid"
        )
    sample_contract = timing.get("sample_contract")
    if not isinstance(sample_contract, str) or not sample_contract:
        raise ValueError(
            "measurement.provenance.execution_timing sample contract is invalid"
        )
    if timing.get("abi") != _EXECUTION_TIMING_ABI:
        raise ValueError("measurement.provenance.execution_timing ABI is invalid")
    if timing.get("status") == "below_timer_resolution":
        if (
            execution_seconds_per_point is not None
            or float(raw) != 0.0
            or timing.get("ratio_eligible") is not False
            or timing.get("compiled_direct_arena_active") is not True
            or timing.get("source") != _COMPILED_ARENA_EXECUTION_TIME_SOURCE
            or native_points is None
        ):
            raise ValueError(
                "measurement.provenance.execution_timing below-resolution "
                "record is not an authenticated compiled Direct-Arena zero"
            )
        return
    if timing.get("status") != "measured":
        raise ValueError(
            "measurement.provenance.execution_timing status is unsupported"
        )
    if (
        execution_seconds_per_point is None
        or float(raw) != execution_seconds_per_point
        or timing.get("ratio_eligible") is not (execution_seconds_per_point > 0.0)
        or not isinstance(timing.get("compiled_direct_arena_active"), bool)
        or not isinstance(timing.get("source"), str)
        or not timing.get("source")
    ):
        raise ValueError(
            "measurement.provenance.execution_timing measured record is inconsistent"
        )


def validate_measurement(value: object) -> None:
    measurement = _required_mapping(value, "measurement")
    expected_keys = set(empty_measurement())
    if set(measurement) != expected_keys:
        missing = sorted(expected_keys - set(measurement))
        extra = sorted(set(measurement) - expected_keys)
        raise ValueError(
            f"measurement fields do not match schema; missing={missing}, extra={extra}"
        )
    try:
        status = ResultStatus(str(measurement["status"]))
    except ValueError as exc:
        raise ValueError("measurement.status is unsupported") from exc
    for field in (
        "generation_seconds",
        "wall_seconds_per_point",
        "execution_seconds_per_point",
        "standard_error_seconds_per_point",
        "relative_standard_error",
    ):
        _required_number_or_none(measurement[field], f"measurement.{field}")
    sample_count = measurement["sample_count"]
    if sample_count is not None and (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("measurement.sample_count must be a positive integer or null")
    if status is ResultStatus.NOT_AVAILABLE:
        expected = empty_measurement()
        if dict(measurement) != expected:
            raise ValueError("not_available measurement must be the canonical reset")
    elif status is ResultStatus.OK:
        for field in (
            "generation_seconds",
            "wall_seconds_per_point",
            "matrix_element",
            "sample_count",
            "artifact",
            "validation",
            "resources",
            "provenance",
        ):
            if measurement[field] is None:
                raise ValueError(f"successful measurement requires {field}")
        validation = _required_mapping(measurement["validation"], "validation")
        if validation.get("status") != ResultStatus.OK.value:
            raise ValueError("successful measurement requires successful validation")
        provenance = _required_mapping(
            measurement["provenance"], "measurement.provenance"
        )
        execution_seconds = _required_number_or_none(
            measurement["execution_seconds_per_point"],
            "measurement.execution_seconds_per_point",
        )
        raw_execution_timing = provenance.get("execution_timing")
        if raw_execution_timing is not None:
            _validate_execution_timing(
                raw_execution_timing,
                execution_seconds_per_point=execution_seconds,
            )
        elif execution_seconds is None and "source_revision" in provenance:
            raise ValueError(
                "successful pyAmpliCol measurement with unavailable execution "
                "timing requires below-resolution provenance"
            )
        _validate_runtime_identity_postflight(provenance, validation)
        if measurement["failure"] is not None:
            raise ValueError("successful measurement cannot contain failure metadata")
    elif measurement["failure"] is None:
        raise ValueError("non-success measurement requires failure metadata")


def validate_cache(
    payload: object,
    *,
    expected_cells: Iterable[CellSpec] | None = None,
) -> None:
    cache = _required_mapping(payload, "cache")
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema_version is unsupported")
    if cache.get("report_version") != REPORT_VERSION:
        raise ValueError("cache report_version is unsupported")
    if cache.get("kind") != "performance_measurements":
        raise ValueError("cache kind is unsupported")
    dataset_id = cache.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("cache dataset_id must be a non-empty string")
    entries = cache.get("entries")
    if not isinstance(entries, list):
        raise ValueError("cache entries must be a list")
    ids: list[str] = []
    for index, raw_entry in enumerate(entries):
        entry = _required_mapping(raw_entry, f"entries[{index}]")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"entries[{index}].cell_id must be a non-empty string")
        ids.append(cell_id)
        if entry.get("process_key") is not None and not isinstance(
            entry.get("process_key"), str
        ):
            raise ValueError(f"entries[{index}].process_key must be a string or null")
        if not isinstance(entry.get("process"), str):
            raise ValueError(f"entries[{index}].process must be a string")
        n_final = entry.get("n_final")
        if isinstance(n_final, bool) or not isinstance(n_final, int) or n_final < 1:
            raise ValueError(f"entries[{index}].n_final must be positive")
        validate_measurement(entry.get("measurement"))
    duplicate_ids = sorted(
        cell_id for cell_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"cache contains duplicate cell IDs: {duplicate_ids}")
    if expected_cells is not None:
        expected = sorted(cell.cell_id for cell in expected_cells)
        if sorted(ids) != expected:
            raise ValueError(f"cache coverage differs for dataset {dataset_id}")


def schema_document() -> dict[str, object]:
    statuses = [status.value for status in ResultStatus]
    nullable_number: dict[str, Any] = {"type": ["number", "null"], "minimum": 0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pyamplicol.dev/schema/report-cache-v3.json",
        "title": "pyAmpliCol three-mode performance cache",
        "type": "object",
        "required": [
            "$schema",
            "schema_version",
            "report_version",
            "kind",
            "dataset_id",
            "measurement_specs",
            "entries",
        ],
        "properties": {
            "$schema": {"const": "./report-cache.schema.json"},
            "schema_version": {"const": CACHE_SCHEMA_VERSION},
            "report_version": {"const": REPORT_VERSION},
            "kind": {"const": "performance_measurements"},
            "dataset_id": {"type": "string", "minLength": 1},
            "measurement_specs": {"type": "array", "items": {"type": "object"}},
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "cell_id",
                        "process_key",
                        "process",
                        "n_final",
                        "variant",
                        "workload",
                        "measurement_spec",
                        "measurement",
                    ],
                    "properties": {
                        "cell_id": {"type": "string", "minLength": 1},
                        "process_key": {"type": ["string", "null"]},
                        "process": {"type": "string", "minLength": 1},
                        "n_final": {"type": "integer", "minimum": 1},
                        "variant": {"type": ["string", "null"]},
                        "workload": {
                            "enum": ["selected-flow", "all-flow", "contracted"]
                        },
                        "measurement_spec": {"type": "object"},
                        "measurement": {
                            "type": "object",
                            "required": list(empty_measurement()),
                            "properties": {
                                "status": {"enum": statuses},
                                "generation_seconds": nullable_number,
                                "wall_seconds_per_point": nullable_number,
                                "execution_seconds_per_point": nullable_number,
                                "matrix_element": {
                                    "type": ["number", "string", "null"]
                                },
                                "sample_count": {
                                    "type": ["integer", "null"],
                                    "minimum": 1,
                                },
                                "standard_error_seconds_per_point": nullable_number,
                                "relative_standard_error": nullable_number,
                                "artifact": {"type": ["object", "null"]},
                                "selector_contract": {"type": ["object", "null"]},
                                "validation": {"type": ["object", "null"]},
                                "resources": {"type": ["object", "null"]},
                                "provenance": {"type": ["object", "null"]},
                                "failure": {"type": ["object", "null"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def write_reset_caches(
    results_dir: Path,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[Path, ...]:
    results_dir.mkdir(parents=True, exist_ok=True)
    caches = build_reset_caches(catalog)
    written: list[Path] = []
    schema_path = results_dir / "report-cache.schema.json"
    schema_path.write_bytes(_canonical_json(schema_document()) + b"\n")
    written.append(schema_path)
    for name, payload in caches.items():
        path = results_dir / name
        path.write_bytes(_canonical_json(payload) + b"\n")
        written.append(path)
    return tuple(written)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "REPORT_VERSION",
    "build_reset_cache",
    "build_reset_caches",
    "digest_json",
    "empty_measurement",
    "schema_document",
    "validate_cache",
    "validate_measurement",
    "write_reset_caches",
]
