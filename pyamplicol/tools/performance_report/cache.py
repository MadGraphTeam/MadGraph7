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

from .agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_ABI,
    INDEPENDENT_AUTHORITY_FIELD,
    LC_COMMON_COMPONENT_FIELD,
    OTF_COMPILED_CROSS_MODE,
    incoming_agreement_edges,
    independent_numerical_authorities,
    requires_independent_numerical_authority,
    validate_direct_agreement_records,
    validate_lc_common_component,
)
from .catalog import REPORT_CATALOG, ReportCatalog
from .models import Accuracy, CellSpec, ExecutionMode, ResultStatus
from .runner import (
    CONDITIONED_COMPARISON_ABI,
    CONVENTIONAL_WARMUP_FIELDS,
    CONVENTIONAL_WARMUP_TIMING_SCOPE,
    OTF_COLD_WARMUP_FIELDS,
    OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
    OTF_COLD_WARMUP_TIMING_SCOPE,
    OTF_DUAL_AUTHORITY_VALIDATION_ABI,
    OTF_DUAL_AUTHORITY_VALIDATION_FIELD,
    WARMUP_TIMER_SOURCE,
    validate_conditioned_comparison_record,
    validate_on_the_fly_dual_authority_validation_record,
    validate_profile_warmup_evidence,
    validate_resolved_sum_validation_record,
)
from .timing import (
    ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI,
    EVALUATOR_TOTAL_TIMING_KEY,
    MEASURED_EXECUTION_TIMING_ABI,
    evaluator_total_timing_record,
    unavailable_execution_timing_record,
)

CACHE_SCHEMA_VERSION = 4
REPORT_VERSION = "0.3.0"
_MEASURED_EXECUTION_TIMING_FIELDS = frozenset(
    {
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
_PRECISION_DIAGNOSTIC_ABI = (
    "pyamplicol-report-validation-failure-precision-diagnostic-v2"
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
    measurement_sample_count: int,
    arena_profile_evidence: object,
) -> None:
    timing = _required_mapping(value, "measurement.provenance.execution_timing")
    if timing.get("abi") == ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI:
        measurement = {
            "execution_seconds_per_point": execution_seconds_per_point,
            "provenance": {
                "execution_timing": timing,
                "arena_profile_evidence": arena_profile_evidence,
            },
        }
        if (
            unavailable_execution_timing_record(
                measurement,
                "execution_seconds_per_point",
            )
            is None
            or timing.get("sample_count") != measurement_sample_count
        ):
            raise ValueError(
                "measurement.provenance.execution_timing is not an "
                "authenticated Arena unavailable-attribution record"
            )
        return
    if set(timing) != _MEASURED_EXECUTION_TIMING_FIELDS:
        raise ValueError(
            "measurement.provenance.execution_timing fields do not match contract"
        )
    if timing.get("abi") != MEASURED_EXECUTION_TIMING_ABI:
        raise ValueError("measurement.provenance.execution_timing ABI is invalid")
    raw = timing.get("raw_seconds_per_point")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) <= 0.0
    ):
        raise ValueError("measurement.provenance.execution_timing raw time is invalid")
    sample_count = timing.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or sample_count != measurement_sample_count
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
    if timing.get("status") != "measured":
        raise ValueError(
            "measurement.provenance.execution_timing status is unsupported"
        )
    if (
        execution_seconds_per_point is None
        or float(raw) != execution_seconds_per_point
        or execution_seconds_per_point <= 0.0
        or timing.get("ratio_eligible") is not True
        or not isinstance(timing.get("compiled_direct_arena_active"), bool)
        or not isinstance(timing.get("source"), str)
        or not timing.get("source")
    ):
        raise ValueError(
            "measurement.provenance.execution_timing measured record is inconsistent"
        )


def _validate_conditioned_measurement_bindings(
    validation: Mapping[str, object],
    *,
    expected_cell: CellSpec | None,
    selector_contract: object,
) -> None:
    resolved = validation.get("resolved_sum")
    if not isinstance(resolved, Mapping):
        return
    point_identity = resolved.get("point_digest")
    resolved_source = resolved.get("resolved_source_sha256")
    high_resolved = validation.get("high_precision_resolved_sum")
    high_source = (
        high_resolved.get("resolved_source_sha256")
        if isinstance(high_resolved, Mapping)
        else None
    )
    component = validation.get(LC_COMMON_COMPONENT_FIELD)
    for field in ("pointwise", "high_precision"):
        record = validation.get(field)
        if not (
            isinstance(record, Mapping)
            and record.get("abi") == CONDITIONED_COMPARISON_ABI
        ):
            continue
        binding = _required_mapping(
            record.get("comparison_binding"),
            f"validation.{field}.comparison_binding",
        )
        identity = _required_mapping(
            binding.get("selector_component_identity"),
            f"validation.{field}.selector_component_identity",
        )
        if binding.get("point_digest") != point_identity:
            raise ValueError(f"validation.{field} is bound to a different point")
        value_kind = identity.get("value_kind")
        if expected_cell is not None:
            expected_identity = {
                "cell_id": expected_cell.cell_id,
                "accuracy": expected_cell.measurement.accuracy.value,
                "workload": expected_cell.workload.value,
                "selector_contract": selector_contract,
                "value_kind": value_kind,
            }
            for key, expected in expected_identity.items():
                if identity.get(key) != expected:
                    raise ValueError(
                        f"validation.{field} selector/component identity differs"
                    )
        if value_kind == LC_COMMON_COMPONENT_FIELD:
            if not isinstance(component, Mapping):
                raise ValueError(f"validation.{field} has no LC component scale source")
            component_identity = identity.get("component_identity")
            if not isinstance(component_identity, Mapping):
                raise ValueError(
                    f"validation.{field} LC component identity is unavailable"
                )
            for key in ("point_digest", "helicity_ids", "color_flow_ids"):
                if component_identity.get(key) != component.get(key):
                    raise ValueError(
                        f"validation.{field} LC component identity differs"
                    )
            expected_candidate_source = digest_json(component)
        else:
            expected_candidate_source = resolved_source
        if binding.get("candidate_source_sha256") != expected_candidate_source:
            raise ValueError(
                f"validation.{field} candidate scale source differs from resolved "
                "evidence"
            )
        if field == "high_precision" and (
            value_kind != "matrix-element-p16-versus-p32"
            or binding.get("baseline_source_sha256") != high_source
        ):
            raise ValueError(
                "validation.high_precision source binding differs from p32 evidence"
            )


def _validate_independent_authority_record(
    validation: Mapping[str, object],
    *,
    expected_cell: CellSpec,
    expected_status: str,
    catalog: ReportCatalog,
) -> None:
    if not requires_independent_numerical_authority(
        expected_cell,
        catalog=catalog,
    ):
        raise ValueError(
            "independent-authority evidence is not applicable to this catalog cell"
        )
    raw = validation.get(INDEPENDENT_AUTHORITY_FIELD)
    record = _required_mapping(raw, f"validation.{INDEPENDENT_AUTHORITY_FIELD}")
    expected_fields = {
        "abi",
        "expected_cell_ids",
        "selected_cell_id",
        "status",
        "reason",
        "same_artifact_diagnostics_are_authority",
    }
    canonical_ids = [
        cell.cell_id
        for cell in independent_numerical_authorities(
            expected_cell,
            catalog=catalog,
        )
    ]
    if (
        set(record) != expected_fields
        or record.get("abi") != INDEPENDENT_AUTHORITY_ABI
        or record.get("expected_cell_ids") != canonical_ids
        or record.get("status") != expected_status
        or record.get("same_artifact_diagnostics_are_authority") is not False
    ):
        raise ValueError("independent-authority record is invalid")
    selected = record.get("selected_cell_id")
    reason = record.get("reason")
    if expected_status == "unavailable":
        if selected is not None or reason != "no-successful-independent-authority":
            raise ValueError("unavailable independent-authority record is invalid")
    elif (
        not isinstance(selected, str)
        or selected not in canonical_ids
        or not isinstance(reason, str)
        or not reason
    ):
        raise ValueError("selected independent-authority record is invalid")


def _validate_unverified_precision_diagnostic(value: object) -> None:
    record = _required_mapping(value, "validation.precision_diagnostic")
    if (
        record.get("abi") != _PRECISION_DIAGNOSTIC_ABI
        or record.get("promotes_measurement") is not False
    ):
        raise ValueError("unverified precision diagnostic is invalid")
    if record.get("status") == "unavailable":
        error = _required_mapping(
            record.get("error"),
            "validation.precision_diagnostic.error",
        )
        if not isinstance(error.get("kind"), str) or not isinstance(
            error.get("message"), str
        ):
            raise ValueError("unavailable precision diagnostic has no error")
        return
    if record.get("status") != "diagnostic-only":
        raise ValueError("unverified precision diagnostic status is invalid")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or [
        attempt.get("precision_digits") if isinstance(attempt, Mapping) else None
        for attempt in attempts
    ] != [32, 200]:
        raise ValueError("unverified precision diagnostic must retain p32 and p200")
    if any(
        attempt.get("status") not in {"evaluated", "unavailable"}
        for attempt in attempts
        if isinstance(attempt, Mapping)
    ) or any(not isinstance(attempt, Mapping) for attempt in attempts):
        raise ValueError("unverified precision diagnostic attempt is invalid")


def _validate_standalone_internal_validation(
    measurement: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    """Validate the p16/resolved/p32 contract for standalone benchmarks."""

    if INDEPENDENT_AUTHORITY_FIELD in validation:
        raise ValueError(
            "standalone measurement cannot store independent-authority evidence"
        )
    resolved = _required_mapping(
        validation.get("resolved_sum"),
        "validation.resolved_sum",
    )
    high_resolved = _required_mapping(
        validation.get("high_precision_resolved_sum"),
        "validation.high_precision_resolved_sum",
    )
    high_precision = _required_mapping(
        validation.get("high_precision"),
        "validation.high_precision",
    )
    resolved_points = resolved.get("points")
    high_points = high_resolved.get("points")
    if (
        not isinstance(resolved_points, list)
        or len(resolved_points) != 1
        or not isinstance(resolved_points[0], Mapping)
        or not isinstance(high_points, list)
        or len(high_points) != 1
        or not isinstance(high_points[0], Mapping)
    ):
        raise ValueError(
            "standalone measurement requires one p16 and one p32 resolved point"
        )
    resolved_point = resolved_points[0]
    high_point = high_points[0]
    if (
        resolved.get("status") != ResultStatus.OK.value
        or resolved.get("precision_digits") != 16
        or high_resolved.get("status") != ResultStatus.OK.value
        or high_resolved.get("precision_digits") != 32
        or high_precision.get("status") != ResultStatus.OK.value
    ):
        raise ValueError(
            "standalone measurement requires successful resolved and p32 validation"
        )
    identity_fields = (
        "point_digest",
        "helicity_ids",
        "color_flow_ids",
        "resolved_ordering_sha256",
    )
    if any(
        resolved.get(field) != high_resolved.get(field) for field in identity_fields
    ):
        raise ValueError(
            "standalone p16 and p32 resolved evidence has different identity"
        )
    matrix_element = measurement.get("matrix_element")
    if (
        isinstance(matrix_element, bool)
        or not isinstance(matrix_element, (int, float))
        or not math.isfinite(float(matrix_element))
    ):
        raise ValueError("standalone measurement matrix element is not finite")
    expected_links = (
        (resolved_point.get("candidate"), matrix_element),
        (high_precision.get("candidate"), matrix_element),
        (high_precision.get("baseline"), high_point.get("candidate")),
        (
            high_precision.get("candidate_scale"),
            resolved_point.get("candidate_scale"),
        ),
        (
            high_precision.get("baseline_scale"),
            high_point.get("candidate_scale"),
        ),
        (high_precision.get("relative_tolerance"), resolved.get("relative_tolerance")),
        (
            high_precision.get("relative_tolerance"),
            high_resolved.get("relative_tolerance"),
        ),
    )
    if any(observed != expected for observed, expected in expected_links):
        raise ValueError(
            "standalone matrix element and p16/p32 evidence are not exactly linked"
        )
    if (
        high_precision.get("candidate_scale_source") != "resolved-component-l1-binary64"
        or high_precision.get("baseline_scale_source") != "resolved-component-l1-p32"
    ):
        raise ValueError("standalone p16/p32 scale-source labels are invalid")


def _validate_unverified_direct_agreement_coverage(
    validation: Mapping[str, object],
    *,
    expected_cell: CellSpec,
    catalog: ReportCatalog,
) -> None:
    direct_records = validation.get(DIRECT_AGREEMENT_FIELD)
    if not isinstance(direct_records, list):
        raise ValueError("unverified result has no direct-agreement records")
    observed_edges = {
        (
            record.get("edge_kind"),
            record.get("baseline_cell_id"),
            record.get("candidate_cell_id"),
        )
        for record in direct_records
        if isinstance(record, Mapping)
    }
    if any(
        record.get("status") != ResultStatus.OK.value
        for record in direct_records
        if isinstance(record, Mapping)
    ):
        raise ValueError("unverified result has a failed direct-agreement record")
    catalog_edges = incoming_agreement_edges(expected_cell, catalog=catalog)
    required_edges = {
        (edge.kind, edge.baseline.cell_id, edge.candidate.cell_id)
        for edge in catalog_edges
        if edge.required
    }
    allowed_edges = {
        (edge.kind, edge.baseline.cell_id, edge.candidate.cell_id)
        for edge in catalog_edges
    }
    if not required_edges.issubset(observed_edges) or not (
        observed_edges <= allowed_edges
    ):
        raise ValueError("unverified result direct-agreement coverage is incomplete")


def _validate_successful_otf_dual_authority(
    measurement: Mapping[str, object],
    validation: Mapping[str, object],
    *,
    expected_cell: CellSpec,
    catalog: ReportCatalog,
) -> None:
    if (
        expected_cell.measurement.execution_mode is not ExecutionMode.ON_THE_FLY
        or expected_cell.measurement.accuracy is not Accuracy.LC
    ):
        return
    recurrence = catalog.validation_baseline_cell(expected_cell)
    compiled_edges = tuple(
        edge
        for edge in incoming_agreement_edges(expected_cell, catalog=catalog)
        if edge.kind == OTF_COMPILED_CROSS_MODE
    )
    if (
        recurrence is None
        or recurrence.measurement.execution_mode is not ExecutionMode.RECURRENCE
        or len(compiled_edges) != 1
    ):
        raise ValueError("on-the-fly catalog authority chain is invalid")
    provenance = _required_mapping(
        measurement.get("provenance"), "measurement.provenance"
    )
    runtime_identity = _required_mapping(
        provenance.get("runtime_identity"),
        "measurement.provenance.runtime_identity",
    )
    candidate_artifact_id = runtime_identity.get("artifact_id")
    if not isinstance(candidate_artifact_id, str):
        raise ValueError("on-the-fly runtime identity has no artifact ID")
    validate_on_the_fly_dual_authority_validation_record(
        validation.get(OTF_DUAL_AUTHORITY_VALIDATION_FIELD),
        expected_cell=expected_cell,
        selector_contract=measurement.get("selector_contract"),
        candidate_artifact_id=candidate_artifact_id,
        expected_authorities=(
            ("recurrence", recurrence.cell_id),
            ("compiled", compiled_edges[0].baseline.cell_id),
        ),
    )


def _validate_successful_otf_warmup_evidence(
    measurement: Mapping[str, object],
    *,
    expected_cell: CellSpec,
) -> None:
    if expected_cell.measurement.execution_mode is not ExecutionMode.ON_THE_FLY:
        return
    provenance = _required_mapping(
        measurement.get("provenance"), "measurement.provenance"
    )
    if provenance.get("runtime_load_included_in_cold_warmup") is not False:
        raise ValueError(
            "on-the-fly runtime load must be excluded from cold warm-up timing"
        )
    if provenance.get("generation_timer_excludes_model_preparation") is not True:
        raise ValueError("on-the-fly generation timing must exclude model preparation")
    effective = _required_mapping(
        provenance.get("effective_config"),
        "measurement.provenance.effective_config",
    )
    benchmark = _required_mapping(
        effective.get("benchmark"),
        "measurement.provenance.effective_config.benchmark",
    )
    expected_batch_size = benchmark.get("batch_size")
    expected_warmup_runs = benchmark.get("warmup_runs")
    if (
        isinstance(expected_batch_size, bool)
        or not isinstance(expected_batch_size, int)
        or expected_batch_size < 1
    ):
        raise ValueError("on-the-fly effective benchmark batch_size must be positive")
    if (
        isinstance(expected_warmup_runs, bool)
        or not isinstance(expected_warmup_runs, int)
        or expected_warmup_runs < 0
    ):
        raise ValueError(
            "on-the-fly effective benchmark warmup_runs must be non-negative"
        )
    validate_profile_warmup_evidence(
        provenance.get("runtime_profile"),
        execution_mode=expected_cell.measurement.execution_mode,
        expected_batch_size=expected_batch_size,
        expected_warmup_run_count=expected_warmup_runs,
    )


def validate_measurement(
    value: object,
    *,
    expected_cell: CellSpec | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> None:
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
    elif status in {ResultStatus.OK, ResultStatus.UNVERIFIED}:
        retained_diagnostic = status is ResultStatus.UNVERIFIED
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
                raise ValueError(f"measured result requires {field}")
        validation = _required_mapping(measurement["validation"], "validation")
        expected_validation_status = (
            ResultStatus.UNVERIFIED.value
            if retained_diagnostic
            else ResultStatus.OK.value
        )
        if validation.get("status") != expected_validation_status:
            raise ValueError("measurement validation status is inconsistent")
        if "resolved_sum" in validation:
            validate_resolved_sum_validation_record(validation["resolved_sum"])
        for field in ("pointwise", "high_precision"):
            if field in validation:
                validate_conditioned_comparison_record(
                    validation[field],
                    require_binding=True,
                )
        if "high_precision_resolved_sum" in validation:
            validate_resolved_sum_validation_record(
                validation["high_precision_resolved_sum"]
            )
        _validate_conditioned_measurement_bindings(
            validation,
            expected_cell=expected_cell,
            selector_contract=measurement.get("selector_contract"),
        )
        validate_direct_agreement_records(
            validation.get(DIRECT_AGREEMENT_FIELD),
            expected_candidate_id=(
                None if expected_cell is None else expected_cell.cell_id
            ),
        )
        selector_contract = measurement.get("selector_contract")
        if expected_cell is not None:
            expects_lc = expected_cell.measurement.accuracy is Accuracy.LC
            if expects_lc and not isinstance(selector_contract, Mapping):
                raise ValueError("successful LC measurement requires selector_contract")
            if not expects_lc and selector_contract is not None:
                raise ValueError(
                    "successful non-LC measurement cannot contain selector_contract"
                )
        requires_lc_component = (
            expected_cell.measurement.accuracy is Accuracy.LC
            if expected_cell is not None
            else isinstance(selector_contract, Mapping)
        )
        if requires_lc_component:
            validate_lc_common_component(
                validation.get(LC_COMMON_COMPONENT_FIELD),
                expected_cell_id=(
                    None if expected_cell is None else expected_cell.cell_id
                ),
                selector_contract=selector_contract,
            )
        elif validation.get(LC_COMMON_COMPONENT_FIELD) is not None:
            raise ValueError("non-LC measurement cannot contain lc_common_component")
        if expected_cell is not None and not retained_diagnostic:
            _validate_successful_otf_dual_authority(
                measurement,
                validation,
                expected_cell=expected_cell,
                catalog=catalog,
            )
        if retained_diagnostic:
            if (
                expected_cell is None
                or expected_cell.measurement.execution_mode
                not in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
            ):
                raise ValueError(
                    "unverified timing requires a compiled/eager catalog cell"
                )
            _validate_independent_authority_record(
                validation,
                expected_cell=expected_cell,
                expected_status="unavailable",
                catalog=catalog,
            )
            _validate_unverified_direct_agreement_coverage(
                validation,
                expected_cell=expected_cell,
                catalog=catalog,
            )
            resolved = _required_mapping(
                validation.get("resolved_sum"),
                "validation.resolved_sum",
            )
            high_resolved = _required_mapping(
                validation.get("high_precision_resolved_sum"),
                "validation.high_precision_resolved_sum",
            )
            high_precision = _required_mapping(
                validation.get("high_precision"),
                "validation.high_precision",
            )
            if any(
                record.get("status") != ResultStatus.OK.value
                for record in (resolved, high_resolved, high_precision)
            ):
                raise ValueError(
                    "unverified result requires successful same-artifact diagnostics"
                )
            _validate_unverified_precision_diagnostic(
                validation.get("precision_diagnostic")
            )
        elif (
            expected_cell is not None
            and expected_cell.measurement.execution_mode
            in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
            and not requires_independent_numerical_authority(
                expected_cell,
                catalog=catalog,
            )
        ):
            _validate_standalone_internal_validation(measurement, validation)
        elif INDEPENDENT_AUTHORITY_FIELD in validation:
            if expected_cell is None:
                raise ValueError(
                    "independent-authority evidence requires a catalog cell"
                )
            _validate_independent_authority_record(
                validation,
                expected_cell=expected_cell,
                expected_status="verified",
                catalog=catalog,
            )
            pointwise = _required_mapping(
                validation.get("pointwise"),
                "validation.pointwise",
            )
            if pointwise.get("status") != ResultStatus.OK.value:
                raise ValueError(
                    "verified independent-authority comparison is not successful"
                )
            if "precision_diagnostic" in validation:
                _validate_unverified_precision_diagnostic(
                    validation.get("precision_diagnostic")
                )
        provenance = _required_mapping(
            measurement["provenance"], "measurement.provenance"
        )
        if expected_cell is not None:
            _validate_successful_otf_warmup_evidence(
                measurement,
                expected_cell=expected_cell,
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
                measurement_sample_count=int(measurement["sample_count"]),
                arena_profile_evidence=provenance.get("arena_profile_evidence"),
            )
        elif execution_seconds is None and "source_revision" in provenance:
            raise ValueError(
                "successful pyAmpliCol measurement with unavailable execution "
                "timing requires authenticated Arena provenance"
            )
        if (
            EVALUATOR_TOTAL_TIMING_KEY in provenance
            and evaluator_total_timing_record(measurement) is None
        ):
            raise ValueError(
                "measurement.provenance.evaluator_total_timing is not an "
                "authenticated accumulated evaluator-total record"
            )
        _validate_runtime_identity_postflight(provenance, validation)
        if retained_diagnostic:
            failure = _required_mapping(
                measurement["failure"],
                "measurement.failure",
            )
            if (
                set(failure) != {"kind", "message"}
                or failure.get("kind") != "IndependentAuthorityUnavailable"
                or not isinstance(failure.get("message"), str)
                or not failure.get("message")
            ):
                raise ValueError("unverified result failure metadata is invalid")
        elif measurement["failure"] is not None:
            raise ValueError("successful measurement cannot contain failure metadata")
    elif measurement["failure"] is None:
        raise ValueError("non-success measurement requires failure metadata")


def validate_cache(
    payload: object,
    *,
    expected_cells: Iterable[CellSpec] | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> None:
    expected_cell_list = None if expected_cells is None else tuple(expected_cells)
    expected_by_id = (
        {}
        if expected_cell_list is None
        else {cell.cell_id: cell for cell in expected_cell_list}
    )
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
        validate_measurement(
            entry.get("measurement"),
            expected_cell=expected_by_id.get(cell_id),
            catalog=catalog,
        )
    duplicate_ids = sorted(
        cell_id for cell_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"cache contains duplicate cell IDs: {duplicate_ids}")
    if expected_cell_list is not None:
        expected = sorted(cell.cell_id for cell in expected_cell_list)
        if sorted(ids) != expected:
            raise ValueError(f"cache coverage differs for dataset {dataset_id}")


def schema_document() -> dict[str, object]:
    statuses = [status.value for status in ResultStatus]
    nullable_number: dict[str, Any] = {"type": ["number", "null"], "minimum": 0}
    cold_warmup_required = sorted(OTF_COLD_WARMUP_FIELDS)
    otf_warmup_required = sorted(CONVENTIONAL_WARMUP_FIELDS | OTF_COLD_WARMUP_FIELDS)
    runtime_profile: dict[str, Any] = {
        "type": "object",
        "properties": {
            "cold_warmup_elapsed_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
            "cold_warmup_run_count": {"const": 1},
            "cold_warmup_batch_size": {"type": "integer", "minimum": 1},
            "cold_warmup_point_count": {"type": "integer", "minimum": 1},
            "cold_warmup_timer_source": {"const": WARMUP_TIMER_SOURCE},
            "cold_warmup_timing_scope": {"const": OTF_COLD_WARMUP_TIMING_SCOPE},
            "cold_warmup_runtime_freshness": {
                "const": OTF_COLD_WARMUP_RUNTIME_FRESHNESS
            },
            "cold_warmup_ratio_eligible": {"const": False},
            "cold_warmup_acceptance_eligible": {"const": False},
            "warmup_elapsed_seconds": {"type": "number", "minimum": 0},
            "warmup_configured_run_count": {
                "type": "integer",
                "minimum": 0,
            },
            "warmup_batch_size": {"type": "integer", "minimum": 1},
            "warmup_point_count": {"type": "integer", "minimum": 0},
            "warmup_run_outer_wall_seconds": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
            },
            "first_warmup_run_outer_wall_seconds": {
                "type": ["number", "null"],
                "minimum": 0,
            },
            "warmup_timer_source": {"const": WARMUP_TIMER_SOURCE},
            "warmup_timing_scope": {"const": CONVENTIONAL_WARMUP_TIMING_SCOPE},
        },
        "dependentRequired": {
            field: otf_warmup_required for field in cold_warmup_required
        },
    }
    provenance_record: dict[str, Any] = {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {
                    "runtime_profile": runtime_profile,
                    "runtime_load_included_in_cold_warmup": {"const": False},
                    "generation_timer_excludes_model_preparation": {"const": True},
                },
            },
        ]
    }
    direct_identity_properties: dict[str, Any] = {
        "edge_kind": {
            "enum": [
                "builtin-ufo-recurrence",
                "z-recurrence-cross-mode",
                "otf-compiled-cross-mode",
                "lc-cross-layout-component",
                "lc-legacy-pyamplicol-component",
            ]
        },
        "value_kind": {"enum": ["matrix_element", LC_COMMON_COMPONENT_FIELD]},
        "baseline_cell_id": {"type": "string", "minLength": 1},
        "candidate_cell_id": {"type": "string", "minLength": 1},
        "status": {"enum": statuses},
        "candidate": {"type": "number"},
        "baseline": {"type": "number"},
        "absolute_difference": {"type": "number", "minimum": 0},
        "relative_difference": {"type": "number", "minimum": 0},
        "relative_tolerance": {"type": "number", "minimum": 0},
    }
    direct_v1: dict[str, Any] = {
        "type": "object",
        "required": [
            "abi",
            "edge_kind",
            "value_kind",
            "baseline_cell_id",
            "candidate_cell_id",
            "status",
            "candidate",
            "baseline",
            "absolute_difference",
            "relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
        ],
        "properties": {
            "abi": {"const": "pyamplicol-report-direct-agreement-v1"},
            **direct_identity_properties,
            "absolute_tolerance": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    direct_v2_fields = [
        "abi",
        "edge_kind",
        "value_kind",
        "baseline_cell_id",
        "candidate_cell_id",
        "status",
        "candidate",
        "baseline",
        "candidate_scale",
        "baseline_scale",
        "candidate_scale_source",
        "baseline_scale_source",
        "comparison_scale",
        "absolute_difference",
        "relative_difference",
        "conditioned_residual",
        "error_bound",
        "relative_tolerance",
        "comparison_binding",
    ]
    direct_v2: dict[str, Any] = {
        "type": "object",
        "required": direct_v2_fields,
        "properties": {
            "abi": {"const": "pyamplicol-report-direct-agreement-v2"},
            **direct_identity_properties,
            "candidate_scale": {"type": "number", "minimum": 0},
            "baseline_scale": {"type": "number", "minimum": 0},
            "candidate_scale_source": {"type": "string", "minLength": 1},
            "baseline_scale_source": {"type": "string", "minLength": 1},
            "comparison_scale": {"type": "number", "minimum": 0},
            "conditioned_residual": {"type": "number", "minimum": 0},
            "error_bound": {"type": "number", "minimum": 0},
            "comparison_binding": {"type": "object", "minProperties": 1},
        },
        "additionalProperties": False,
    }
    direct_agreement_record: dict[str, Any] = {"oneOf": [direct_v1, direct_v2]}
    lc_common_component: dict[str, Any] = {
        "type": "object",
        "required": [
            "abi",
            "cell_id",
            "value",
            "point_digest",
            "helicity_ids",
            "color_flow_ids",
        ],
        "properties": {
            "abi": {"const": "pyamplicol-report-lc-common-component-v1"},
            "cell_id": {"type": "string", "minLength": 1},
            "value": {"type": "number"},
            "point_digest": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "helicity_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "color_flow_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "additionalProperties": False,
    }
    otf_total_summary: dict[str, Any] = {
        "type": "object",
        "required": [
            "check_count",
            "maximum_conditioned_residual",
            "maximum_absolute_delta",
        ],
        "properties": {
            "check_count": {"type": "integer", "minimum": 1},
            "maximum_conditioned_residual": {"type": "number", "minimum": 0},
            "maximum_absolute_delta": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    otf_component_summary: dict[str, Any] = {
        "type": "object",
        "required": [
            "check_count",
            "component_count",
            "maximum_conditioned_residual",
            "maximum_absolute_delta",
        ],
        "properties": {
            **otf_total_summary["properties"],
            "component_count": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    }
    otf_comparison: dict[str, Any] = {
        "type": "object",
        "required": ["authority_cell_id", "total", "resolved_components"],
        "properties": {
            "authority_cell_id": {"type": "string", "minLength": 1},
            "total": otf_total_summary,
            "resolved_components": otf_component_summary,
        },
        "additionalProperties": False,
    }
    otf_stage: dict[str, Any] = {
        "type": "object",
        "required": [
            "candidate_total_source_sha256",
            "candidate_resolved_total_source_sha256",
            "candidate_resolved_ordering_sha256",
            "candidate_resolved_source_sha256",
            "candidate_resolved_sum",
            "comparisons",
        ],
        "properties": {
            "candidate_total_source_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "candidate_resolved_total_source_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "candidate_resolved_ordering_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "candidate_resolved_source_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "candidate_resolved_sum": otf_total_summary,
            "comparisons": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": otf_comparison,
            },
        },
        "additionalProperties": False,
    }
    otf_authority: dict[str, Any] = {
        "type": "object",
        "required": [
            "role",
            "cell_id",
            "artifact_id",
            "total_source_sha256",
            "resolved_total_source_sha256",
            "resolved_ordering_sha256",
            "resolved_source_sha256",
            "resolved_sum",
        ],
        "properties": {
            "role": {"enum": ["recurrence", "compiled"]},
            "cell_id": {"type": "string", "minLength": 1},
            **{
                field: {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                for field in (
                    "artifact_id",
                    "total_source_sha256",
                    "resolved_total_source_sha256",
                    "resolved_ordering_sha256",
                    "resolved_source_sha256",
                )
            },
            "resolved_sum": otf_total_summary,
        },
        "additionalProperties": False,
    }
    otf_dual_authority: dict[str, Any] = {
        "type": "object",
        "required": [
            "abi",
            "status",
            "candidate_cell_id",
            "candidate_artifact_id",
            "workload",
            "precision_digits",
            "relative_tolerance",
            "point_digest",
            "selector_sha256",
            "point_count",
            "resolved_component_count",
            "resolved_check_count",
            "authorities",
            "before_clear",
            "after_clear",
            "lifecycle",
        ],
        "properties": {
            "abi": {"const": OTF_DUAL_AUTHORITY_VALIDATION_ABI},
            "status": {"const": ResultStatus.OK.value},
            "candidate_cell_id": {"type": "string", "minLength": 1},
            "candidate_artifact_id": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "workload": {"enum": ["selected-flow", "all-flow"]},
            "precision_digits": {"const": 16},
            "relative_tolerance": {"const": 1.0e-12},
            "point_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "selector_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "point_count": {"type": "integer", "minimum": 1},
            "resolved_component_count": {"type": "integer", "minimum": 1},
            "resolved_check_count": {"type": "integer", "minimum": 1},
            "authorities": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": otf_authority,
            },
            "before_clear": otf_stage,
            "after_clear": otf_stage,
            "lifecycle": {
                "type": "object",
                "required": [
                    "authority_artifacts_loaded_only",
                    "candidate_loaded_before_validation",
                    "validated_before_clear",
                    "validated_after_clear",
                    "clear_call_count",
                    "final_clear_before_profile",
                ],
                "properties": {
                    "authority_artifacts_loaded_only": {"const": True},
                    "candidate_loaded_before_validation": {"const": True},
                    "validated_before_clear": {"const": True},
                    "validated_after_clear": {"const": True},
                    "clear_call_count": {"const": 2},
                    "final_clear_before_profile": {"const": True},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    validation_properties = {
        "status": {"enum": statuses},
        DIRECT_AGREEMENT_FIELD: {
            "type": "array",
            "items": direct_agreement_record,
        },
        INDEPENDENT_AUTHORITY_FIELD: {
            "type": "object",
            "required": [
                "abi",
                "expected_cell_ids",
                "selected_cell_id",
                "status",
                "reason",
                "same_artifact_diagnostics_are_authority",
            ],
            "properties": {
                "abi": {"const": INDEPENDENT_AUTHORITY_ABI},
                "expected_cell_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "selected_cell_id": {"type": ["string", "null"]},
                "status": {
                    "enum": ["unavailable", "verified", "mismatch", "incompatible"]
                },
                "reason": {"type": "string", "minLength": 1},
                "same_artifact_diagnostics_are_authority": {"const": False},
            },
            "additionalProperties": False,
        },
        LC_COMMON_COMPONENT_FIELD: lc_common_component,
        OTF_DUAL_AUTHORITY_VALIDATION_FIELD: otf_dual_authority,
    }
    validation_record: dict[str, Any] = {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": validation_properties,
            },
        ]
    }
    successful_validation_record: dict[str, Any] = {
        "type": "object",
        "required": [DIRECT_AGREEMENT_FIELD],
        "properties": validation_properties,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pyamplicol.dev/schema/report-cache-v4.json",
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
                                "validation": validation_record,
                                "resources": {"type": ["object", "null"]},
                                "provenance": provenance_record,
                                "failure": {"type": ["object", "null"]},
                            },
                            "allOf": [
                                {
                                    "if": {
                                        "properties": {
                                            "status": {"const": ResultStatus.OK.value}
                                        },
                                        "required": ["status"],
                                    },
                                    "then": {
                                        "properties": {
                                            "validation": (successful_validation_record)
                                        }
                                    },
                                },
                                {
                                    "if": {
                                        "properties": {
                                            "status": {
                                                "const": ResultStatus.UNVERIFIED.value
                                            }
                                        },
                                        "required": ["status"],
                                    },
                                    "then": {
                                        "properties": {
                                            "validation": {
                                                "type": "object",
                                                "required": [
                                                    "status",
                                                    DIRECT_AGREEMENT_FIELD,
                                                    INDEPENDENT_AUTHORITY_FIELD,
                                                    "resolved_sum",
                                                    "high_precision_resolved_sum",
                                                    "high_precision",
                                                    "precision_diagnostic",
                                                ],
                                                "properties": validation_properties,
                                            },
                                            "failure": {"type": "object"},
                                        }
                                    },
                                },
                            ],
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
