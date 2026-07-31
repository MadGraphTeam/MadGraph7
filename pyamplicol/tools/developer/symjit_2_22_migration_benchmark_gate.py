#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare authenticated SymJIT 2.22 migration benchmark captures.

The recurrence z+6g harness deliberately records raw, independently warmed
subprocess measurements.  This gate consumes one topology-replay and one
all-flow-union capture for both the immutable pre-migration baseline and the
candidate.  It validates their workload identities, recomputes every statistic
used for acceptance from the retained raw samples, authenticates the outer
30-GiB watchdog report against the paired command/session/result, and emits one
content-addressed comparison record. The exact candidate revision, both native
build-input digests, and both prepared-model digests are independent required
inputs rather than identities inferred from the captures being judged.

This is a focused migration gate. It reports compiled/recurrence and eager
inventories separately, but applies the same regression acceptance rules to
all three required lanes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPTURE_KIND = "pyamplicol-recurrence-z6g-benchmark"
MINIMUM_CAPTURE_SCHEMA = 7
COMPARISON_KIND = "pyamplicol-symjit-2.22-migration-benchmark-comparison"
COMPARISON_SCHEMA = 2
BASELINE_REVISION = "172e58fd33a3c65563866c50cfbb5e1ddcd7b302"

LAYOUTS = ("topology-replay", "all-flow-union")
AUTHORITATIVE_MODES = ("compiled", "recurrence")
DIAGNOSTIC_MODES = ("eager",)
MODES = AUTHORITATIVE_MODES
CAPTURE_MODES = AUTHORITATIVE_MODES
BATCH_SIZES = (1, 128, 1024)
RATIO_BATCH_SIZES = (128, 1024)
MINIMUM_SUBPROCESS_SAMPLES = 7
MINIMUM_INTERNAL_SAMPLES = 7
MINIMUM_WARMUPS = 2
MINIMUM_TARGET_RUNTIME_SECONDS = 5.0

RUNTIME_RELATIVE_LIMIT = 1.03
RUNTIME_NOISE_MAD_MULTIPLIER = 3.0
COMPILED_RECURRENCE_RATIO_LIMIT = 1.15
GENERATION_CELL_LIMIT = 1.10
GENERATION_GEOMEAN_LIMIT = 1.05
RESOURCE_GROWTH_LIMIT = 1.03
RESOURCE_EXCEPTION_GAIN = 0.10

STRICT_SEMANTIC_COMPARISON_POLICY = "strict-exact-v1"
SELECTED_UNION_SEMANTIC_COMPARISON_POLICY = (
    "ddbar-z6g-all-flow-union-selected-helicity-v1"
)
SELECTED_UNION_PROCESS = "d d~ > Z g g g g g g"
SELECTED_UNION_HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
SELECTED_UNION_HELICITY_VALUES = (-1, 1, -1, 1, -1, 1, -1, 1, -1)
SELECTED_UNION_HELICITY_INDEX = 234
SELECTED_UNION_COLOR_COUNT = 720
PAIRED_CAMPAIGN_KIND = "pyamplicol-symjit-2.22-paired-benchmark-campaign"
PAIRED_CAMPAIGN_SCHEMA = 2
PAIRED_HARNESS_KIND = "pyamplicol-paired-benchmark-harness-identity"
PAIRED_HARNESS_SCHEMA = 1
PAIRED_HARNESS_RELATIVE_PATH = "tools/developer/recurrence_z6g_benchmark.py"
PAIRED_DRIVER_RELATIVE_PATH = "tools/developer/symjit_2_22_paired_benchmark.py"
RAW_NATIVE_WALL_BLOCK_KIND = "pyamplicol-raw-native-wall-blocks"
RAW_NATIVE_WALL_BLOCK_SCHEMA = 2
RAW_NATIVE_WALL_MEASUREMENT_CONTRACT = (
    "warmed-native-wall-minimum-duration-v1"
)
WATCHDOG_REPORT_KIND = "pyamplicol-memory-watchdog-execution-report"
WATCHDOG_REPORT_SCHEMA = 2
WATCHDOG_LIMIT_BYTES = 30 * 1024**3
WATCHDOG_SCOPE = "complete-orchestrator-process-tree-v1"
WATCHDOG_BINDING = "outer-command-session-result-v1"
WATCHDOG_RELATIVE_PATH = "tools/ci/memory_watchdog.py"

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40


class EvidenceError(RuntimeError):
    """Raised when a capture cannot support an acceptance decision."""


@dataclass(frozen=True)
class Distribution:
    """Raw values and statistics recomputed by this gate."""

    values: tuple[float, ...]
    median: float
    raw_mad: float

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": len(self.values),
            "values": list(self.values),
            "median": self.median,
            "raw_mad": self.raw_mad,
        }


@dataclass(frozen=True)
class CellEvidence:
    """One validated mode/batch cell from a capture."""

    runtime: Distribution
    cold_load: Distribution
    cold_load_rss: Distribution
    profiled_rss: Distribution
    by_round: Mapping[int, float]


@dataclass(frozen=True)
class GenerationEvidence:
    """Fresh generation metrics for one mode."""

    wall_seconds: float
    payload_size_bytes: int
    peak_rss_bytes: int


@dataclass(frozen=True)
class CaptureEvidence:
    """Validated evidence extracted from one recurrence-z6g capture."""

    input_identity: Mapping[str, object]
    layout: str
    logical_process: str
    source_revision: str
    build_identity: Mapping[str, object]
    host_identity: Mapping[str, object]
    campaign_identity: Mapping[str, object]
    workload_identities: Mapping[str, Mapping[str, object]]
    cells: Mapping[tuple[str, int], CellEvidence]
    generation: Mapping[str, GenerationEvidence]
    configured_modes: tuple[str, ...]
    paired_profile_coordination: Mapping[str, object]
    eager_diagnostic: Mapping[str, object]
    interpreter_path: str
    interpreter_resolved_path: str
    prepared_model_path: str
    prepared_model_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise EvidenceError("evidence is not canonical-JSON serializable") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvidenceError(f"cannot read capture: {path}") from error
    return digest.hexdigest()


def _input_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise EvidenceError(f"cannot inspect capture: {path}") from error
    if not resolved.is_file():
        raise EvidenceError(f"capture is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_json_mapping(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    identity = _input_identity(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} root is not an object: {path}")
    return value, identity


def _read_capture(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    return _read_json_mapping(path, label="capture")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} is missing or is not an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EvidenceError(f"{label} is missing or is not an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} is missing or is not a non-empty string")
    return value


def _exact_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} is missing or is not an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{label} must be at least {minimum}")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is missing or is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise EvidenceError(f"{label} must be a positive finite number")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is missing or is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} must be a finite non-negative number")
    return result


def _valid_sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != _SHA256_LENGTH:
        raise EvidenceError(f"{label} is not a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise EvidenceError(f"{label} is not a SHA-256 digest") from error
    if result != result.lower():
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _valid_revision(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != _REVISION_LENGTH:
        raise EvidenceError(f"{label} is not a full Git revision")
    try:
        int(result, 16)
    except ValueError as error:
        raise EvidenceError(f"{label} is not a Git revision") from error
    if result != result.lower():
        raise EvidenceError(f"{label} is not a lowercase Git revision")
    return result


def _validate_content_address(value: object, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    unsigned = dict(record)
    digest = _valid_sha256(
        unsigned.pop("content_sha256", None), f"{label}.content_sha256"
    )
    if digest != _sha256(unsigned):
        raise EvidenceError(f"{label} content address does not match its payload")
    return record


def _validate_nested_content_addresses(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        if "content_sha256" in value:
            record = _validate_content_address(value, label)
            for field, child in record.items():
                if field != "content_sha256":
                    _validate_nested_content_addresses(child, f"{label}.{field}")
            return
        for field, child in value.items():
            _validate_nested_content_addresses(child, f"{label}.{field}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _validate_nested_content_addresses(child, f"{label}[{index}]")


def _utc_timestamp(value: object, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} is not a UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise EvidenceError(f"{label} is not a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise EvidenceError(f"{label} is not a UTC timestamp")
    return parsed


def _distribution(values: Sequence[float], label: str) -> Distribution:
    if len(values) < MINIMUM_SUBPROCESS_SAMPLES:
        raise EvidenceError(
            f"{label} has {len(values)} subprocess samples; "
            f"at least {MINIMUM_SUBPROCESS_SAMPLES} are required"
        )
    checked = tuple(_positive_number(value, f"{label} raw sample") for value in values)
    median = float(statistics.median(checked))
    raw_mad = float(statistics.median(abs(value - median) for value in checked))
    return Distribution(values=checked, median=median, raw_mad=raw_mad)


def _rss_observation(value: object, label: str) -> float:
    peak = _mapping(value, label)
    if peak.get("source") != "resource.getrusage":
        raise EvidenceError(f"{label} has an unsupported RSS source")
    self_peak = _exact_int(
        peak.get("self_peak_bytes"), f"{label}.self_peak_bytes", minimum=0
    )
    child_peak = _exact_int(
        peak.get("maximum_child_peak_bytes"),
        f"{label}.maximum_child_peak_bytes",
        minimum=0,
    )
    observed = _exact_int(
        peak.get("observed_lower_bound_bytes"),
        f"{label}.observed_lower_bound_bytes",
        minimum=1,
    )
    if observed != max(self_peak, child_peak):
        raise EvidenceError(f"{label} is not a consistent resource peak")
    return float(observed)


def _recorded_statistic(
    observed: object,
    expected: float,
    label: str,
) -> None:
    value = _nonnegative_number(observed, label)
    if not math.isclose(value, expected, rel_tol=1.0e-14, abs_tol=0.0):
        raise EvidenceError(
            f"{label} is not reproducible from retained raw subprocess samples"
        )


def _logical_process(value: object, label: str) -> str:
    normalized = " ".join(_string(value, label).split()).casefold()
    supported = {
        "u u~ > z g g g g g g": "u u~ > Z g g g g g g",
        "d d~ > z + 6*g": "d d~ > Z g g g g g g",
        "d d~ > z g g g g g g": "d d~ > Z g g g g g g",
    }
    try:
        return supported[normalized]
    except KeyError as error:
        raise EvidenceError(
            f"{label} is not the supported u u~ or exact d d~ > Z + 6g route"
        ) from error


def _expected_semantic_comparison_policy(
    *,
    logical_process: str,
    layout: str,
    configuration: Mapping[str, Any],
    label: str,
) -> str:
    if logical_process == SELECTED_UNION_PROCESS and layout == "all-flow-union":
        if configuration.get("helicity_request") != SELECTED_UNION_HELICITY_ID:
            raise EvidenceError(
                f"{label} exact d all-flow-union route must select "
                f"{SELECTED_UNION_HELICITY_ID}"
            )
        return SELECTED_UNION_SEMANTIC_COMPARISON_POLICY
    return STRICT_SEMANTIC_COMPARISON_POLICY


def _validated_selected_union_projection(
    value: object,
    *,
    configuration: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    projection = _mapping(value, label)
    expected_fields = {
        "policy",
        "process_id",
        "process_expression",
        "physical_color_flows",
        "physical_helicities",
        "normalization_sha256",
        "model_common_physics_identity_sha256",
        "runtime_selector_semantics_sha256",
        "profile_selector",
        "resolved_color_ids_sha256",
    }
    if set(projection) != expected_fields:
        raise EvidenceError(f"{label} has an unexpected projection schema")
    if (
        projection.get("policy") != SELECTED_UNION_SEMANTIC_COMPARISON_POLICY
        or _logical_process(
            projection.get("process_expression"),
            f"{label}.process_expression",
        )
        != SELECTED_UNION_PROCESS
    ):
        raise EvidenceError(f"{label} has the wrong selected-union policy or process")
    _string(projection.get("process_id"), f"{label}.process_id")

    colors = _mapping(
        projection.get("physical_color_flows"),
        f"{label}.physical_color_flows",
    )
    if (
        set(colors)
        != {
            "count",
            "ordered_ids_sha256",
            "ordered_entries_sha256",
        }
        or _exact_int(
            colors.get("count"),
            f"{label}.physical_color_flows.count",
            minimum=1,
        )
        != SELECTED_UNION_COLOR_COUNT
    ):
        raise EvidenceError(f"{label} does not bind exactly 720 physical colors")
    for field in ("ordered_ids_sha256", "ordered_entries_sha256"):
        _valid_sha256(colors.get(field), f"{label}.physical_color_flows.{field}")

    helicities = _mapping(
        projection.get("physical_helicities"),
        f"{label}.physical_helicities",
    )
    if set(helicities) != {
        "count",
        "ordered_ids_sha256",
        "kinematic_entries_sha256",
        "selected_index",
        "selected_entry",
    }:
        raise EvidenceError(f"{label} has an invalid physical-helicity projection")
    helicity_count = _exact_int(
        helicities.get("count"),
        f"{label}.physical_helicities.count",
        minimum=SELECTED_UNION_HELICITY_INDEX + 1,
    )
    selected_index = _exact_int(
        helicities.get("selected_index"),
        f"{label}.physical_helicities.selected_index",
        minimum=0,
    )
    if selected_index != SELECTED_UNION_HELICITY_INDEX:
        raise EvidenceError(f"{label} selected helicity has the wrong ordered index")
    for field in ("ordered_ids_sha256", "kinematic_entries_sha256"):
        _valid_sha256(helicities.get(field), f"{label}.physical_helicities.{field}")
    selected = _mapping(
        helicities.get("selected_entry"),
        f"{label}.physical_helicities.selected_entry",
    )
    if (
        set(selected) != {"index", "id", "values", "coefficient", "structural_zero"}
        or selected.get("index") != selected_index
        or selected.get("id") != SELECTED_UNION_HELICITY_ID
        or selected.get("values") != list(SELECTED_UNION_HELICITY_VALUES)
        or selected.get("structural_zero") is not False
    ):
        raise EvidenceError(f"{label} selected helicity entry is not canonical/live")
    _positive_number(
        selected.get("coefficient"),
        f"{label}.physical_helicities.selected_entry.coefficient",
    )

    selector = _mapping(
        projection.get("profile_selector"),
        f"{label}.profile_selector",
    )
    if (
        set(selector)
        != {
            "color_flow_request",
            "resolved_color_flow_id",
            "helicity_request",
            "resolved_helicity_id",
            "color_flow_count",
            "helicity_count",
            "workload",
        }
        or selector.get("color_flow_request") != configuration.get("color_flow_request")
        or selector.get("resolved_color_flow_id") is not None
        or selector.get("helicity_request") != SELECTED_UNION_HELICITY_ID
        or selector.get("resolved_helicity_id") != SELECTED_UNION_HELICITY_ID
        or selector.get("color_flow_count") != SELECTED_UNION_COLOR_COUNT
        or selector.get("helicity_count") != helicity_count
        or selector.get("workload") != "all-flows/runtime-selected-single-helicity"
    ):
        raise EvidenceError(f"{label} has the wrong selected-union selector")
    for field in (
        "normalization_sha256",
        "model_common_physics_identity_sha256",
        "runtime_selector_semantics_sha256",
        "resolved_color_ids_sha256",
    ):
        _valid_sha256(projection.get(field), f"{label}.{field}")
    return projection


def _capture_semantic_projection(
    acceptance: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    logical_process: str,
    layout: str,
    label: str,
) -> Mapping[str, Any] | None:
    semantic_contract = _mapping(
        acceptance.get("artifact_semantic_contract"),
        f"{label}.artifact_semantic_contract",
    )
    lane_contracts = _mapping(
        semantic_contract.get("lane_contracts"),
        f"{label}.artifact_semantic_contract.lane_contracts",
    )
    expected_policy = _expected_semantic_comparison_policy(
        logical_process=logical_process,
        layout=layout,
        configuration=configuration,
        label=label,
    )
    if semantic_contract.get("comparison_policy") != expected_policy:
        raise EvidenceError(
            f"{label} artifact-semantic comparison policy does not match its route"
        )
    if expected_policy == STRICT_SEMANTIC_COMPARISON_POLICY:
        if any(
            isinstance(lane, Mapping) and "selected_union_workload_projection" in lane
            for lane in lane_contracts.values()
        ):
            raise EvidenceError(
                f"{label} strict route unexpectedly contains a "
                "selected-union projection"
            )
        return None

    common = _validated_selected_union_projection(
        semantic_contract.get("common_physics_contract"),
        configuration=configuration,
        label=f"{label}.artifact_semantic_contract.common_physics_contract",
    )
    for mode in CAPTURE_MODES:
        lane = _mapping(
            lane_contracts.get(mode),
            f"{label}.artifact_semantic_contract.lane_contracts.{mode}",
        )
        lane_projection = lane.get("selected_union_workload_projection")
        if lane_projection != common:
            raise EvidenceError(
                f"{label} {mode} selected-union projection differs from common"
            )
    return common


def _validate_capture_acceptance(
    payload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    expected_layout: str,
    subprocess_samples: int,
    internal_samples: int,
    warmups: int,
    configured_modes: Sequence[str],
) -> Mapping[str, Any]:
    label = f"{expected_layout}.capture_acceptance"
    if payload.get("complete") is not True or payload.get("passes") is not True:
        raise EvidenceError(
            f"{expected_layout} capture is not top-level complete and passing"
        )
    acceptance = _mapping(payload.get("capture_acceptance"), label)
    if (
        acceptance.get("kind") != "pyamplicol-three-lane-layout-capture"
        or _exact_int(
            acceptance.get("schema_version"),
            f"{label}.schema_version",
        )
        < 6
    ):
        raise EvidenceError(f"{label} has an unsupported identity")
    required_true = (
        "complete",
        "evidence_complete",
        "passes",
        "authoritative_eligible",
        "lane_self_validation_passes",
        "pairwise_validation_passes",
    )
    if any(acceptance.get(field) is not True for field in required_true):
        raise EvidenceError(f"{label} is not authoritative, complete, and passing")
    if (
        acceptance.get("authoritative_ineligibility_reasons") != []
        or acceptance.get("generation_specialized_axes_by_mode") != {}
        or acceptance.get("incomplete_physical_axes") != []
        or acceptance.get("missing_modes") != []
        or acceptance.get("missing_batch_sizes") != []
        or acceptance.get("generation_only") is not False
        or acceptance.get("layout") != expected_layout
    ):
        raise EvidenceError(f"{label} contains authoritative ineligibility evidence")
    if (
        list(
            _sequence(
                acceptance.get("required_modes"),
                f"{label}.required_modes",
            )
        )
        != list(CAPTURE_MODES)
        or list(
            _sequence(
                acceptance.get("observed_modes"),
                f"{label}.observed_modes",
            )
        )
        != list(CAPTURE_MODES)
        or set(
            _sequence(
                acceptance.get("required_batch_sizes"),
                f"{label}.required_batch_sizes",
            )
        )
        != set(BATCH_SIZES)
        or set(
            _sequence(
                acceptance.get("observed_batch_sizes"),
                f"{label}.observed_batch_sizes",
            )
        )
        != set(BATCH_SIZES)
    ):
        raise EvidenceError(f"{label} required/observed inventory is inconsistent")

    measurement = _mapping(
        acceptance.get("measurement_contract"),
        f"{label}.measurement_contract",
    )
    if (
        measurement.get("passes") is not True
        or measurement.get("root_processes_match") is not True
        or measurement.get("configured_internal_minimum_samples") != internal_samples
        or measurement.get("configured_subprocess_samples") != subprocess_samples
        or measurement.get("configured_warmup_runs") != warmups
        or _exact_int(
            measurement.get("minimum_authoritative_samples"),
            f"{label}.measurement_contract.minimum_authoritative_samples",
            minimum=MINIMUM_SUBPROCESS_SAMPLES,
        )
        > subprocess_samples
    ):
        raise EvidenceError(f"{label} measurement contract is inconsistent")
    schedule_contract = _mapping(
        measurement.get("schedule"),
        f"{label}.measurement_contract.schedule",
    )
    expected_entry_count = len(configured_modes) * len(BATCH_SIZES) * subprocess_samples
    if (
        schedule_contract.get("passes") is not True
        or schedule_contract.get("errors") != []
        or schedule_contract.get("entry_count") != expected_entry_count
        or schedule_contract.get("unique_worker_command_count") != expected_entry_count
        or schedule_contract.get("subprocess_samples_per_cell") != subprocess_samples
    ):
        raise EvidenceError(f"{label} schedule contract is incomplete")
    lane_contracts = _mapping(
        measurement.get("lanes"),
        f"{label}.measurement_contract.lanes",
    )
    if set(lane_contracts) != set(CAPTURE_MODES):
        raise EvidenceError(f"{label} measurement lane inventory is inconsistent")
    for mode in CAPTURE_MODES:
        lane = _mapping(
            lane_contracts.get(mode),
            f"{label}.measurement_contract.lanes.{mode}",
        )
        if (
            lane.get("passes") is not True
            or lane.get("errors") != []
            or set(
                _sequence(
                    lane.get("observed_batch_sizes"),
                    f"{label}.measurement_contract.lanes.{mode}.observed_batch_sizes",
                )
            )
            != set(BATCH_SIZES)
            or lane.get("missing_batch_sizes") != []
        ):
            raise EvidenceError(f"{label} {mode} measurement lane is incomplete")

    semantic_contract = _mapping(
        acceptance.get("artifact_semantic_contract"),
        f"{label}.artifact_semantic_contract",
    )
    semantic_lanes = _mapping(
        semantic_contract.get("lane_contracts"),
        f"{label}.artifact_semantic_contract.lane_contracts",
    )
    if (
        semantic_contract.get("passes") is not True
        or semantic_contract.get("errors") != []
        or semantic_contract.get("lanes_match") is not True
        or not isinstance(semantic_contract.get("common_physics_contract"), Mapping)
        or set(semantic_lanes) != set(CAPTURE_MODES)
    ):
        raise EvidenceError(f"{label} artifact-semantic contract is incomplete")

    eager = _mapping(
        acceptance.get("eager_diagnostic"),
        f"{label}.eager_diagnostic",
    )
    eager_measurement = _mapping(
        eager.get("measurement_contract"),
        f"{label}.eager_diagnostic.measurement_contract",
    )
    eager_semantics = _mapping(
        eager.get("artifact_semantic_contract"),
        f"{label}.eager_diagnostic.artifact_semantic_contract",
    )
    eager_validation = _mapping(
        eager.get("validation_summary"),
        f"{label}.eager_diagnostic.validation_summary",
    )
    if (
        eager.get("requested") is not True
        or eager.get("observed") is not True
        or eager.get("complete") is not True
        or eager.get("passes") is not True
        or eager.get("ineligibility_reasons") != []
        or eager_measurement.get("passes") is not True
        or eager_semantics.get("passes") is not True
        or eager_semantics.get("lanes_match") is not True
        or eager_validation.get("passes") is not True
    ):
        raise EvidenceError(f"{label} eager diagnostic is not complete and admissible")
    authoritative_validation = {
        "passes": acceptance.get("passes"),
        "lane_validation_passes": acceptance.get("lane_self_validation_passes"),
        "pairwise_validation_passes": acceptance.get("pairwise_validation_passes"),
    }
    if (
        authoritative_validation["passes"] is not True
        or authoritative_validation["lane_validation_passes"] is not True
        or authoritative_validation["pairwise_validation_passes"] is not True
    ):
        raise EvidenceError(
            f"{expected_layout} capture validation summary is not passing"
        )
    if set(
        _sequence(
            configuration.get("modes"),
            f"{expected_layout}.configuration.modes",
        )
    ) != set(configured_modes):
        raise EvidenceError(f"{label} does not match the configured lane inventory")
    return acceptance


def _semantic_digest(
    semantic: Mapping[str, Any],
    field: str,
    digest_field: str,
    label: str,
) -> tuple[object, str]:
    value = semantic.get(field)
    if value is None:
        raise EvidenceError(f"{label}.{field} is missing")
    digest = _valid_sha256(
        semantic.get(digest_field),
        f"{label}.{digest_field}",
    )
    if digest != _sha256(value):
        raise EvidenceError(f"{label}.{digest_field} does not match {field}")
    return value, digest


def _bind_selected_union_projection(
    projection: Mapping[str, Any],
    *,
    semantic: Mapping[str, Any],
    selector: Mapping[str, Any],
    validation: Mapping[str, Any],
    process_id: str,
    normalization_sha256: str,
    model_sha256: str,
    selector_semantics_sha256: str,
    label: str,
) -> None:
    if projection.get("process_id") != process_id:
        raise EvidenceError(f"{label} projected process ID differs from its profile")
    projected_selector = _mapping(
        projection.get("profile_selector"),
        f"{label}.selected_union_projection.profile_selector",
    )
    selector_fields = (
        "color_flow_request",
        "resolved_color_flow_id",
        "helicity_request",
        "resolved_helicity_id",
        "color_flow_count",
        "helicity_count",
        "workload",
    )
    if set(selector) != {*selector_fields, "structural_zero_helicity_count"} or {
        field: selector.get(field) for field in selector_fields
    } != dict(projected_selector):
        raise EvidenceError(
            f"{label} selector differs from its selected-union projection"
        )

    projected_colors = _mapping(
        projection.get("physical_color_flows"),
        f"{label}.selected_union_projection.physical_color_flows",
    )
    colors = _mapping(
        semantic.get("physical_color_flows"),
        f"{label}.physical_color_flows",
    )
    color_ids = list(
        _sequence(
            colors.get("ordered_ids"),
            f"{label}.physical_color_flows.ordered_ids",
        )
    )
    color_entries = list(
        _sequence(
            colors.get("ordered_entries"),
            f"{label}.physical_color_flows.ordered_entries",
        )
    )
    if (
        colors.get("count") != SELECTED_UNION_COLOR_COUNT
        or len(color_ids) != SELECTED_UNION_COLOR_COUNT
        or len(color_entries) != SELECTED_UNION_COLOR_COUNT
        or len(set(color_ids)) != SELECTED_UNION_COLOR_COUNT
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in color_ids
        )
        or _valid_sha256(
            colors.get("ordered_ids_sha256"),
            f"{label}.physical_color_flows.ordered_ids_sha256",
        )
        != _sha256(color_ids)
        or _valid_sha256(
            colors.get("ordered_entries_sha256"),
            f"{label}.physical_color_flows.ordered_entries_sha256",
        )
        != _sha256(color_entries)
        or projected_colors.get("count") != colors.get("count")
        or projected_colors.get("ordered_ids_sha256")
        != colors.get("ordered_ids_sha256")
        or projected_colors.get("ordered_entries_sha256")
        != colors.get("ordered_entries_sha256")
    ):
        raise EvidenceError(
            f"{label} physical colors differ from selected-union projection"
        )

    projected_helicities = _mapping(
        projection.get("physical_helicities"),
        f"{label}.selected_union_projection.physical_helicities",
    )
    helicities = _mapping(
        semantic.get("physical_helicities"),
        f"{label}.physical_helicities",
    )
    helicity_ids = list(
        _sequence(
            helicities.get("ordered_ids"),
            f"{label}.physical_helicities.ordered_ids",
        )
    )
    raw_helicity_entries = list(
        _sequence(
            helicities.get("ordered_entries"),
            f"{label}.physical_helicities.ordered_entries",
        )
    )
    helicity_count = _exact_int(
        helicities.get("count"),
        f"{label}.physical_helicities.count",
        minimum=SELECTED_UNION_HELICITY_INDEX + 1,
    )
    if (
        len(helicity_ids) != helicity_count
        or len(raw_helicity_entries) != helicity_count
        or len(set(helicity_ids)) != helicity_count
        or _valid_sha256(
            helicities.get("ordered_ids_sha256"),
            f"{label}.physical_helicities.ordered_ids_sha256",
        )
        != _sha256(helicity_ids)
        or projected_helicities.get("count") != helicity_count
        or projected_helicities.get("ordered_ids_sha256")
        != helicities.get("ordered_ids_sha256")
    ):
        raise EvidenceError(
            f"{label} physical helicities differ from selected-union projection"
        )
    kinematic_entries: list[dict[str, object]] = []
    selected_entry: dict[str, object] | None = None
    for index, raw_entry in enumerate(raw_helicity_entries):
        entry = _mapping(
            raw_entry,
            f"{label}.physical_helicities.ordered_entries[{index}]",
        )
        identifier = entry.get("id")
        values = entry.get("values")
        if (
            entry.get("index") != index
            or helicity_ids[index] != identifier
            or not isinstance(identifier, str)
            or not isinstance(values, list)
        ):
            raise EvidenceError(f"{label} physical helicity ordering is invalid")
        projected_entry: dict[str, object] = {
            "index": index,
            "id": identifier,
            "values": values,
        }
        if identifier == SELECTED_UNION_HELICITY_ID:
            projected_entry.update(
                {
                    "coefficient": entry.get("coefficient"),
                    "structural_zero": entry.get("structural_zero"),
                }
            )
            selected_entry = dict(entry)
        kinematic_entries.append(projected_entry)
    if (
        helicity_ids[SELECTED_UNION_HELICITY_INDEX] != SELECTED_UNION_HELICITY_ID
        or selected_entry is None
        or selected_entry != projected_helicities.get("selected_entry")
        or projected_helicities.get("selected_index") != SELECTED_UNION_HELICITY_INDEX
        or projected_helicities.get("kinematic_entries_sha256")
        != _sha256(kinematic_entries)
    ):
        raise EvidenceError(
            f"{label} selected helicity differs from selected-union projection"
        )

    resolved_colors = list(
        _sequence(
            validation.get("resolved_color_ids"),
            f"{label}.validation.resolved_color_ids",
        )
    )
    if (
        resolved_colors != color_ids
        or projection.get("resolved_color_ids_sha256") != _sha256(resolved_colors)
        or validation.get("resolved_helicity_ids") != [SELECTED_UNION_HELICITY_ID]
        or projection.get("normalization_sha256") != normalization_sha256
        or projection.get("model_common_physics_identity_sha256") != model_sha256
        or projection.get("runtime_selector_semantics_sha256")
        != selector_semantics_sha256
    ):
        raise EvidenceError(
            f"{label} selected-union physics/digests differ from projection"
        )


def _workload_identity(
    *,
    profile: Mapping[str, Any],
    generation: Mapping[str, Any],
    mode: str,
    layout: str,
    logical_process: str,
    expected_workload: str,
    selected_union_projection: Mapping[str, Any] | None,
    label: str,
) -> dict[str, object]:
    if profile.get("mode") != mode:
        raise EvidenceError(f"{label}.mode does not match its lane")
    if (
        _logical_process(
            profile.get("process_expression"), f"{label}.process_expression"
        )
        != logical_process
    ):
        raise EvidenceError(f"{label} process expression drifted from its capture")
    process_id = _string(profile.get("process_id"), f"{label}.process_id")
    selector = _mapping(profile.get("selector_contract"), f"{label}.selector_contract")
    if selector.get("workload") != expected_workload:
        raise EvidenceError(f"{label}.selector_contract has the wrong workload")
    validation = _mapping(profile.get("validation"), f"{label}.validation")
    if validation.get("passes") is not True:
        raise EvidenceError(f"{label} numerical validation did not pass")
    fixture = _mapping(validation.get("fixture"), f"{label}.validation.fixture")
    point_count = _exact_int(
        fixture.get("point_count"),
        f"{label}.validation.fixture.point_count",
        minimum=1,
    )
    points_sha256 = _valid_sha256(
        fixture.get("points_sha256"),
        f"{label}.validation.fixture.points_sha256",
    )

    semantic = _mapping(
        profile.get("artifact_semantic_identity"),
        f"{label}.artifact_semantic_identity",
    )
    semantic_digest = _valid_sha256(
        profile.get("artifact_semantic_identity_sha256"),
        f"{label}.artifact_semantic_identity_sha256",
    )
    if semantic_digest != _sha256(semantic):
        raise EvidenceError(
            f"{label} artifact semantic identity is not content-addressed"
        )
    generation_identity = _mapping(
        generation.get("artifact_identity"),
        f"{label}.generation.artifact_identity",
    )
    if (
        generation_identity.get("semantic_identity") != semantic
        or generation_identity.get("semantic_identity_sha256") != semantic_digest
    ):
        raise EvidenceError(
            f"{label} profile and generation artifact identities differ"
        )
    if semantic.get("generation_specialized_axes") != []:
        raise EvidenceError(f"{label} has generation-specialized physical axes")
    coverage = _mapping(semantic.get("coverage"), f"{label}.coverage")
    if coverage.get("complete_physical_axes") is not True:
        raise EvidenceError(f"{label} does not contain complete physical axes")
    reduction_coverage = _mapping(
        semantic.get("reduction_coverage"),
        f"{label}.reduction_coverage",
    )
    if reduction_coverage.get("complete") is not True:
        raise EvidenceError(f"{label} reduction coverage is incomplete")

    normalization, normalization_sha256 = _semantic_digest(
        semantic,
        "normalization",
        "normalization_sha256",
        label,
    )
    reduction_ordering, reduction_ordering_sha256 = _semantic_digest(
        semantic,
        "reduction_ordering",
        "reduction_ordering_sha256",
        label,
    )
    selector_semantics, selector_semantics_sha256 = _semantic_digest(
        semantic,
        "runtime_selector_semantics",
        "runtime_selector_semantics_sha256",
        label,
    )
    model_identity = _mapping(
        semantic.get("manifest_model_identity"),
        f"{label}.manifest_model_identity",
    )
    common_model = _mapping(
        model_identity.get("common_physics_identity"),
        f"{label}.manifest_model_identity.common_physics_identity",
    )
    common_model_sha256 = _valid_sha256(
        model_identity.get("common_physics_identity_sha256"),
        f"{label}.manifest_model_identity.common_physics_identity_sha256",
    )
    if common_model_sha256 != _sha256(common_model):
        raise EvidenceError(f"{label} common model identity is not content-addressed")
    physical_color_flows = semantic.get("physical_color_flows")
    physical_helicities = semantic.get("physical_helicities")
    if physical_color_flows is None or physical_helicities is None:
        raise EvidenceError(f"{label} physical-axis identity is incomplete")

    effective = _mapping(
        generation.get("effective_contract"),
        f"{label}.generation.effective_contract",
    )
    expected_optimization = 3 if mode == "compiled" else 2
    expected_effective = {
        "backend": "jit",
        "color_accuracy": "lc",
        "execution_mode": mode,
        "jit_optimization_level": expected_optimization,
        "lc_flow_layout": layout,
    }
    if any(effective.get(key) != value for key, value in expected_effective.items()):
        raise EvidenceError(
            f"{label} does not use {mode} JIT O{expected_optimization} "
            f"with the requested {layout} layout"
        )

    common_identity = {
        "process": logical_process,
        "process_id": process_id,
        "layout": layout,
        "mode": mode,
        "workload": expected_workload,
        "validation_fixture": {
            "point_count": point_count,
            "points_sha256": points_sha256,
        },
        "coverage_sha256": _sha256(coverage),
        "effective_contract": expected_effective,
    }
    if selected_union_projection is not None:
        _bind_selected_union_projection(
            selected_union_projection,
            semantic=semantic,
            selector=selector,
            validation=validation,
            process_id=process_id,
            normalization_sha256=normalization_sha256,
            model_sha256=common_model_sha256,
            selector_semantics_sha256=selector_semantics_sha256,
            label=label,
        )
        return {
            **common_identity,
            "semantic_comparison_policy": (SELECTED_UNION_SEMANTIC_COMPARISON_POLICY),
            "selected_union_projection_sha256": _sha256(selected_union_projection),
        }

    # Store digests for the large physical inventories so the comparison
    # remains compact while still binding every ordered entry.
    return {
        **common_identity,
        "selector_contract_sha256": _sha256(selector),
        "common_model_identity_sha256": common_model_sha256,
        "normalization_sha256": normalization_sha256,
        "normalization_payload_sha256": _sha256(normalization),
        "reduction_ordering_sha256": reduction_ordering_sha256,
        "reduction_ordering_payload_sha256": _sha256(reduction_ordering),
        "runtime_selector_semantics_sha256": selector_semantics_sha256,
        "runtime_selector_semantics_payload_sha256": _sha256(selector_semantics),
        "physical_color_flows_sha256": _sha256(physical_color_flows),
        "physical_helicities_sha256": _sha256(physical_helicities),
    }


def _validated_schedule_worker_records(
    schedule_entry: Mapping[str, Any],
    label: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    str,
    str,
]:
    invocation = _validate_content_address(
        schedule_entry.get("worker_invocation"),
        f"{label}.worker_invocation",
    )
    result = _validate_content_address(
        schedule_entry.get("worker_result_record"),
        f"{label}.worker_result_record",
    )
    invocation_sha256 = _valid_sha256(
        invocation.get("content_sha256"),
        f"{label}.worker_invocation.content_sha256",
    )
    result_sha256 = _valid_sha256(
        result.get("content_sha256"),
        f"{label}.worker_result_record.content_sha256",
    )
    process_sha256 = _valid_sha256(
        result.get("worker_process_record_sha256"),
        f"{label}.worker_result_record.worker_process_record_sha256",
    )
    if result.get("worker_invocation_sha256") != invocation_sha256:
        raise EvidenceError(
            f"{label} result is not bound to its scheduled worker invocation"
        )
    return (
        invocation,
        result,
        invocation_sha256,
        process_sha256,
        result_sha256,
    )


def _validate_sample_address(
    sample: Mapping[str, Any],
    schedule_entry: Mapping[str, Any],
    label: str,
    seen_addresses: set[str],
) -> None:
    invocation = _validate_content_address(
        sample.get("worker_invocation"), f"{label}.worker_invocation"
    )
    process_record = _validate_content_address(
        sample.get("worker_process_record"),
        f"{label}.worker_process_record",
    )
    result_record = _validate_content_address(
        sample.get("worker_result_record"),
        f"{label}.worker_result_record",
    )
    invocation_sha256 = _valid_sha256(
        invocation.get("content_sha256"),
        f"{label}.worker_invocation.content_sha256",
    )
    process_sha256 = _valid_sha256(
        process_record.get("content_sha256"),
        f"{label}.worker_process_record.content_sha256",
    )
    result_sha256 = _valid_sha256(
        result_record.get("content_sha256"),
        f"{label}.worker_result_record.content_sha256",
    )
    for address_name, address in (
        ("invocation", invocation_sha256),
        ("process", process_sha256),
        ("result", result_sha256),
    ):
        scoped = f"{address_name}:{address}"
        if scoped in seen_addresses:
            raise EvidenceError(f"{label} reuses a {address_name} subprocess identity")
        seen_addresses.add(scoped)
    if (
        result_record.get("worker_invocation_sha256") != invocation_sha256
        or result_record.get("worker_process_record_sha256") != process_sha256
    ):
        raise EvidenceError(f"{label} worker result is not bound to its subprocess")
    (
        scheduled_invocation,
        scheduled_result,
        scheduled_invocation_sha256,
        scheduled_process_sha256,
        scheduled_result_sha256,
    ) = _validated_schedule_worker_records(
        schedule_entry,
        f"{label}.profile_schedule_entry",
    )
    if (
        scheduled_invocation != invocation
        or scheduled_result != result_record
        or scheduled_invocation_sha256 != invocation_sha256
        or scheduled_process_sha256 != process_sha256
        or scheduled_result_sha256 != result_sha256
    ):
        raise EvidenceError(
            f"{label} sample worker provenance differs from its profile schedule entry"
        )
    completion = _validate_content_address(
        schedule_entry.get("paired_profile_completion"),
        f"{label}.paired_profile_completion",
    )
    if (
        completion.get("worker_invocation_sha256") != invocation_sha256
        or completion.get("worker_result_record_sha256") != result_sha256
        or completion.get("worker_started_at_utc") != invocation.get("started_at_utc")
        or completion.get("worker_finished_at_utc") != invocation.get("finished_at_utc")
    ):
        raise EvidenceError(
            f"{label} paired completion differs from its retained sample worker"
        )
    invocation_started = _utc_timestamp(
        invocation.get("started_at_utc"),
        f"{label}.worker_invocation.started_at_utc",
    )
    invocation_finished = _utc_timestamp(
        invocation.get("finished_at_utc"),
        f"{label}.worker_invocation.finished_at_utc",
    )
    if invocation_started > invocation_finished:
        raise EvidenceError(f"{label} worker invocation timestamps are inverted")
    unsigned_sample = {
        str(key): value
        for key, value in sample.items()
        if key != "worker_result_record"
    }
    addressed_payload = _valid_sha256(
        result_record.get("addressed_payload_sha256"),
        f"{label}.worker_result_record.addressed_payload_sha256",
    )
    if addressed_payload != _sha256(unsigned_sample):
        raise EvidenceError(f"{label} retained raw sample content address is invalid")
    if schedule_entry.get("worker_result_sha256") != addressed_payload:
        raise EvidenceError(f"{label} is not bound to its profile schedule entry")


def _validate_raw_native_wall_measurement(
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    internal_samples: int,
    warmups: int,
    target_runtime: float,
    sample_wall: float,
    fixture_points_sha256: str,
    label: str,
) -> None:
    raw = _mapping(
        sample.get("inner_native_wall_blocks"),
        f"{label}.inner_native_wall_blocks",
    )
    blocks = _sequence(
        raw.get("blocks"),
        f"{label}.inner_native_wall_blocks.blocks",
    )
    block_count = _exact_int(
        raw.get("block_count"),
        f"{label}.inner_native_wall_blocks.block_count",
        minimum=MINIMUM_INTERNAL_SAMPLES,
    )
    repetitions = _exact_int(
        raw.get("repetitions_per_block"),
        f"{label}.inner_native_wall_blocks.repetitions_per_block",
        minimum=1,
    )
    evaluation_count = _exact_int(
        raw.get("evaluation_count"),
        f"{label}.inner_native_wall_blocks.evaluation_count",
        minimum=1,
    )
    evaluated_point_count = _exact_int(
        raw.get("evaluated_point_count"),
        f"{label}.inner_native_wall_blocks.evaluated_point_count",
        minimum=1,
    )
    minimum_duration = _positive_number(
        raw.get("minimum_native_wall_seconds"),
        f"{label}.inner_native_wall_blocks.minimum_native_wall_seconds",
    )
    observed_duration = _positive_number(
        raw.get("observed_native_wall_seconds"),
        f"{label}.inner_native_wall_blocks.observed_native_wall_seconds",
    )
    observed_caller_duration = _positive_number(
        raw.get("observed_caller_elapsed_seconds"),
        f"{label}.inner_native_wall_blocks.observed_caller_elapsed_seconds",
    )
    invocation = _mapping(
        sample.get("worker_invocation"),
        f"{label}.worker_invocation",
    )
    process_record = _mapping(
        sample.get("worker_process_record"),
        f"{label}.worker_process_record",
    )
    invocation_wall = _positive_number(
        invocation.get("wall_seconds"),
        f"{label}.worker_invocation.wall_seconds",
    )
    process_wall = _positive_number(
        process_record.get("wall_seconds"),
        f"{label}.worker_process_record.wall_seconds",
    )
    calibration = _mapping(
        raw.get("calibration"),
        f"{label}.inner_native_wall_blocks.calibration",
    )
    runner_samples = _exact_int(
        calibration.get("benchmark_runner_sample_count"),
        f"{label}.inner_native_wall_blocks.calibration."
        "benchmark_runner_sample_count",
        minimum=1,
    )
    runner_repetitions = _exact_int(
        calibration.get("benchmark_runner_repetitions_per_sample"),
        f"{label}.inner_native_wall_blocks.calibration."
        "benchmark_runner_repetitions_per_sample",
        minimum=1,
    )
    runner_wall = _positive_number(
        calibration.get("benchmark_runner_wall_seconds_per_point"),
        f"{label}.inner_native_wall_blocks.calibration."
        "benchmark_runner_wall_seconds_per_point",
    )
    requested_block_count = _exact_int(
        calibration.get("requested_minimum_block_count"),
        f"{label}.inner_native_wall_blocks.calibration."
        "requested_minimum_block_count",
        minimum=MINIMUM_INTERNAL_SAMPLES,
    )
    scaled_repetitions = _exact_int(
        calibration.get("scaled_repetitions_per_block"),
        f"{label}.inner_native_wall_blocks.calibration."
        "scaled_repetitions_per_block",
        minimum=1,
    )
    headroom = _positive_number(
        calibration.get("duration_headroom_factor"),
        f"{label}.inner_native_wall_blocks.calibration."
        "duration_headroom_factor",
    )
    calibration_target = target_runtime * headroom
    calibration_denominator = requested_block_count * runner_wall * batch_size
    if (
        not math.isfinite(calibration_target)
        or not math.isfinite(calibration_denominator)
        or calibration_denominator <= 0.0
    ):
        raise EvidenceError(f"{label} raw native-wall calibration is not finite")
    expected_scaled_repetitions = max(
        runner_repetitions,
        math.ceil(calibration_target / calibration_denominator),
    )
    worker_measurement = _mapping(
        sample.get("worker_measurement"),
        f"{label}.worker_measurement",
    )
    if (
        raw.get("kind") != RAW_NATIVE_WALL_BLOCK_KIND
        or raw.get("schema_version") != RAW_NATIVE_WALL_BLOCK_SCHEMA
        or raw.get("measurement_contract")
        != RAW_NATIVE_WALL_MEASUREMENT_CONTRACT
        or raw.get("source") != "runtime._benchmark_f64_wall_time"
        or raw.get("fixture_points_sha256") != fixture_points_sha256
        or raw.get("minimum_duration_satisfied") is not True
        or minimum_duration != target_runtime
        or observed_duration < target_runtime
        or observed_caller_duration < target_runtime
        or observed_caller_duration < observed_duration
        or invocation_wall < observed_duration
        or invocation_wall < observed_caller_duration
        or process_wall < observed_duration
        or process_wall < observed_caller_duration
        or len(blocks) != block_count
        or evaluation_count != block_count * repetitions
        or evaluated_point_count != evaluation_count * batch_size
        or raw.get("blocks_sha256") != _sha256(list(blocks))
        or calibration.get("kind") != "benchmark-runner-wall-rate-calibration"
        or calibration.get("schema_version") != 1
        or calibration.get("benchmark_runner_total_repetitions")
        != runner_samples * runner_repetitions
        or requested_block_count != max(MINIMUM_INTERNAL_SAMPLES, internal_samples)
        or scaled_repetitions != repetitions
        or scaled_repetitions != expected_scaled_repetitions
        or _exact_int(
            calibration.get("preceded_by_benchmark_runner_warmup_runs"),
            f"{label}.inner_native_wall_blocks.calibration."
            "preceded_by_benchmark_runner_warmup_runs",
            minimum=MINIMUM_WARMUPS,
        )
        != warmups
        or headroom != 1.02
        or worker_measurement.get("batch_size") != batch_size
        or worker_measurement.get("sample_count") != block_count
        or worker_measurement.get("repetitions_per_sample") != repetitions
        or worker_measurement.get("evaluation_count") != evaluation_count
        or worker_measurement.get("evaluated_point_count")
        != evaluated_point_count
        or worker_measurement.get("wall_seconds_per_point") != sample_wall
        or worker_measurement.get("inner_native_wall_blocks") != raw
        or worker_measurement.get("benchmark_runner_sample_count")
        != runner_samples
        or worker_measurement.get("benchmark_runner_repetitions_per_sample")
        != runner_repetitions
        or worker_measurement.get("benchmark_runner_evaluation_count")
        != runner_samples * runner_repetitions
        or worker_measurement.get("benchmark_runner_evaluated_point_count")
        != runner_samples * runner_repetitions * batch_size
        or worker_measurement.get("benchmark_runner_wall_seconds_per_point")
        != runner_wall
    ):
        raise EvidenceError(
            f"{label} raw native-wall measurement contract is invalid"
        )

    wall_values: list[float] = []
    native_durations: list[float] = []
    caller_durations: list[float] = []
    previous_finished: dt.datetime | None = None
    for block_index, raw_block in enumerate(blocks):
        block_label = (
            f"{label}.inner_native_wall_blocks.blocks[{block_index}]"
        )
        block = _validate_content_address(raw_block, block_label)
        native = _positive_number(
            block.get("native_wall_seconds"),
            f"{block_label}.native_wall_seconds",
        )
        wall = _positive_number(
            block.get("wall_seconds_per_point"),
            f"{block_label}.wall_seconds_per_point",
        )
        caller = _positive_number(
            block.get("caller_elapsed_seconds"),
            f"{block_label}.caller_elapsed_seconds",
        )
        if (
            block.get("block_index") != block_index
            or block.get("repetitions") != repetitions
            or block.get("batch_size") != batch_size
            or block.get("evaluation_count") != repetitions
            or block.get("evaluated_point_count") != repetitions * batch_size
            or caller < native
            or not math.isclose(
                wall,
                native / (repetitions * batch_size),
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
        ):
            raise EvidenceError(f"{block_label} is internally inconsistent")
        block_started = _utc_timestamp(
            block.get("started_at_utc"),
            f"{block_label}.started_at_utc",
        )
        block_finished = _utc_timestamp(
            block.get("finished_at_utc"),
            f"{block_label}.finished_at_utc",
        )
        if (
            block_started > block_finished
            or (
                previous_finished is not None
                and previous_finished > block_started
            )
        ):
            raise EvidenceError(f"{block_label} timestamps are inverted")
        previous_finished = block_finished
        wall_values.append(wall)
        native_durations.append(native)
        caller_durations.append(caller)

    recomputed_median = statistics.median(wall_values)
    recomputed_mad = statistics.median(
        abs(value - recomputed_median) for value in wall_values
    )
    if (
        sum(native_durations) != observed_duration
        or sum(caller_durations) != observed_caller_duration
        or observed_duration < MINIMUM_TARGET_RUNTIME_SECONDS
        or _positive_number(
            raw.get("wall_seconds_per_point_median"),
            f"{label}.inner_native_wall_blocks.wall_seconds_per_point_median",
        )
        != recomputed_median
        or _nonnegative_number(
            raw.get("wall_seconds_per_point_mad"),
            f"{label}.inner_native_wall_blocks.wall_seconds_per_point_mad",
        )
        != recomputed_mad
        or sample_wall != recomputed_median
        or sample.get("internal_sample_count") != block_count
        or sample.get("repetitions_per_sample") != repetitions
        or sample.get("evaluation_count") != evaluation_count
        or sample.get("evaluated_point_count") != evaluated_point_count
    ):
        raise EvidenceError(
            f"{label} raw native-wall duration/statistics are not reproducible"
        )


def _extract_cell(
    *,
    aggregate: Mapping[str, Any],
    mode: str,
    batch_size: int,
    subprocess_samples: int,
    internal_samples: int,
    warmups: int,
    target_runtime: float,
    fixture_points_sha256: str,
    schedule_by_index: Mapping[int, Mapping[str, Any]],
    seen_addresses: set[str],
    label: str,
) -> CellEvidence:
    if aggregate.get("batch_size") != batch_size:
        raise EvidenceError(f"{label}.batch_size does not match its inventory slot")
    samples = _sequence(
        aggregate.get("subprocess_samples"), f"{label}.subprocess_samples"
    )
    if len(samples) != subprocess_samples:
        raise EvidenceError(
            f"{label} has {len(samples)} subprocess samples, but its campaign "
            f"declares {subprocess_samples}"
        )
    runtime_values: list[float] = []
    cold_values: list[float] = []
    cold_rss_values: list[float] = []
    profiled_rss_values: list[float] = []
    by_round: dict[int, float] = {}
    for sample_index, raw_sample in enumerate(samples):
        sample_label = f"{label}.subprocess_samples[{sample_index}]"
        sample = _mapping(raw_sample, sample_label)
        round_index = _exact_int(
            sample.get("round"), f"{sample_label}.round", minimum=0
        )
        if round_index in by_round:
            raise EvidenceError(f"{label} duplicates subprocess round {round_index}")
        schedule_index = _exact_int(
            sample.get("schedule_index"),
            f"{sample_label}.schedule_index",
            minimum=0,
        )
        schedule_entry = schedule_by_index.get(schedule_index)
        if schedule_entry is None:
            raise EvidenceError(
                f"{sample_label} has no matching profile schedule entry"
            )
        if (
            schedule_entry.get("mode") != mode
            or schedule_entry.get("batch_size") != batch_size
            or schedule_entry.get("round") != round_index
        ):
            raise EvidenceError(f"{sample_label} does not match its schedule slot")
        if sample.get("interrupted") is not False:
            raise EvidenceError(f"{sample_label} was interrupted")
        timing = _mapping(
            sample.get("timing_configuration"),
            f"{sample_label}.timing_configuration",
        )
        sample_warmups = _exact_int(
            timing.get("warmup_runs"),
            f"{sample_label}.timing_configuration.warmup_runs",
            minimum=MINIMUM_WARMUPS,
        )
        sample_target = _positive_number(
            timing.get("target_runtime_seconds"),
            f"{sample_label}.timing_configuration.target_runtime_seconds",
        )
        sample_internal_minimum = _exact_int(
            timing.get("minimum_internal_samples"),
            f"{sample_label}.timing_configuration.minimum_internal_samples",
            minimum=MINIMUM_INTERNAL_SAMPLES,
        )
        if (
            sample_warmups != warmups
            or sample_target != target_runtime
            or sample_internal_minimum != internal_samples
        ):
            raise EvidenceError(f"{sample_label} timing configuration drifted")
        _exact_int(
            sample.get("internal_sample_count"),
            f"{sample_label}.internal_sample_count",
            minimum=MINIMUM_INTERNAL_SAMPLES,
        )
        sources = _mapping(
            sample.get("timing_sources"), f"{sample_label}.timing_sources"
        )
        if sources.get("wall") != "runtime_core_repeated_wall_time":
            raise EvidenceError(
                f"{sample_label} is not an unprofiled native-wall sample"
            )

        runtime = _positive_number(
            sample.get("wall_seconds_per_point"),
            f"{sample_label}.wall_seconds_per_point",
        )
        _validate_raw_native_wall_measurement(
            sample,
            batch_size=batch_size,
            internal_samples=internal_samples,
            warmups=warmups,
            target_runtime=target_runtime,
            sample_wall=runtime,
            fixture_points_sha256=fixture_points_sha256,
            label=sample_label,
        )
        if "cold_load_seconds" not in sample:
            raise EvidenceError(
                f"{sample_label} lacks cold_load_seconds. The capture predates "
                "the resource-evidence contract; re-run the baseline/candidate "
                "recurrence_z6g campaign with the current harness before comparing."
            )
        cold = _positive_number(
            sample.get("cold_load_seconds"),
            f"{sample_label}.cold_load_seconds",
        )
        if (
            "peak_rss_after_cold_load" not in sample
            or "peak_rss_after_profile" not in sample
        ):
            raise EvidenceError(
                f"{sample_label} lacks cold-load/profile RSS evidence. Re-run "
                "the baseline/candidate recurrence_z6g campaign with the current "
                "harness before comparing."
            )
        cold_rss = _rss_observation(
            sample.get("peak_rss_after_cold_load"),
            f"{sample_label}.peak_rss_after_cold_load",
        )
        profiled_rss = _rss_observation(
            sample.get("peak_rss_after_profile"),
            f"{sample_label}.peak_rss_after_profile",
        )
        _validate_sample_address(sample, schedule_entry, sample_label, seen_addresses)
        runtime_values.append(runtime)
        cold_values.append(cold)
        cold_rss_values.append(cold_rss)
        profiled_rss_values.append(profiled_rss)
        by_round[round_index] = runtime
    expected_rounds = set(range(subprocess_samples))
    if set(by_round) != expected_rounds:
        raise EvidenceError(f"{label} does not retain exactly one sample per round")

    runtime_distribution = _distribution(runtime_values, f"{label}.runtime")
    cold_distribution = _distribution(cold_values, f"{label}.cold_load")
    cold_rss_distribution = _distribution(cold_rss_values, f"{label}.cold_load_rss")
    profiled_rss_distribution = _distribution(
        profiled_rss_values,
        f"{label}.profiled_rss",
    )
    if aggregate.get("statistics_contract") != "subprocess-median-and-raw-mad-v1":
        raise EvidenceError(f"{label} has an unsupported statistics contract")
    if (
        aggregate.get("resource_statistics_contract")
        != "subprocess-median-and-raw-mad-v1"
    ):
        raise EvidenceError(f"{label} has an unsupported resource statistics contract")
    if aggregate.get("interrupted") is not False:
        raise EvidenceError(f"{label} aggregate is interrupted")
    if (
        aggregate.get("sample_count") != subprocess_samples
        or aggregate.get("subprocess_sample_count") != subprocess_samples
    ):
        raise EvidenceError(f"{label} aggregate sample count is inconsistent")
    _recorded_statistic(
        aggregate.get("wall_seconds_per_point_median"),
        runtime_distribution.median,
        f"{label}.wall_seconds_per_point_median",
    )
    _recorded_statistic(
        aggregate.get("wall_seconds_per_point_mad"),
        runtime_distribution.raw_mad,
        f"{label}.wall_seconds_per_point_mad",
    )
    _recorded_statistic(
        aggregate.get("cold_load_seconds_median"),
        cold_distribution.median,
        f"{label}.cold_load_seconds_median",
    )
    _recorded_statistic(
        aggregate.get("cold_load_seconds_mad"),
        cold_distribution.raw_mad,
        f"{label}.cold_load_seconds_mad",
    )
    _recorded_statistic(
        aggregate.get("cold_load_peak_rss_bytes_median"),
        cold_rss_distribution.median,
        f"{label}.cold_load_peak_rss_bytes_median",
    )
    _recorded_statistic(
        aggregate.get("profiled_peak_rss_bytes_median"),
        profiled_rss_distribution.median,
        f"{label}.profiled_peak_rss_bytes_median",
    )
    return CellEvidence(
        runtime=runtime_distribution,
        cold_load=cold_distribution,
        cold_load_rss=cold_rss_distribution,
        profiled_rss=profiled_rss_distribution,
        by_round=by_round,
    )


def _validate_paired_profile_coordination(
    payload: Mapping[str, Any],
    *,
    role: str,
    layout: str,
    revision: str,
    schedule_by_index: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, object]:
    label = f"{role}/{layout}.paired_profile_coordination"
    coordination = _mapping(payload.get("paired_profile_coordination"), label)
    if (
        coordination.get("kind") != "pyamplicol-paired-profile-coordination"
        or coordination.get("schema_version") != 1
        or coordination.get("role") != role
        or coordination.get("layout") != layout
    ):
        raise EvidenceError(f"{label} has an unsupported identity")
    session_id = _string(coordination.get("session_id"), f"{label}.session_id")
    ready = _validate_content_address(
        coordination.get("ready_record"),
        f"{label}.ready_record",
    )
    plan = _sequence(
        ready.get("profile_schedule_plan"),
        f"{label}.ready_record.profile_schedule_plan",
    )
    expected_plan = [
        {
            field: schedule_by_index[index].get(field)
            for field in ("schedule_index", "round", "mode", "batch_size")
        }
        for index in sorted(schedule_by_index)
    ]
    if (
        ready.get("kind") != "pyamplicol-paired-profile-ready"
        or ready.get("schema_version") != 1
        or ready.get("session_id") != session_id
        or ready.get("role") != role
        or ready.get("layout") != layout
        or ready.get("source_revision") != revision
        or list(plan) != expected_plan
        or ready.get("profile_schedule_plan_sha256") != _sha256(expected_plan)
    ):
        raise EvidenceError(f"{label} ready record does not bind the capture")
    ready_at = _utc_timestamp(
        ready.get("ready_at_utc"),
        f"{label}.ready_record.ready_at_utc",
    )
    raw_completions = _sequence(
        coordination.get("completion_records"),
        f"{label}.completion_records",
    )
    if len(raw_completions) != len(schedule_by_index):
        raise EvidenceError(f"{label} completion inventory is incomplete")
    slots: dict[int, dict[str, object]] = {}
    for index, entry in schedule_by_index.items():
        token = _validate_content_address(
            entry.get("paired_profile_token"),
            f"{label}.slots[{index}].token",
        )
        completion = _validate_content_address(
            entry.get("paired_profile_completion"),
            f"{label}.slots[{index}].completion",
        )
        expected = {
            "session_id": session_id,
            "role": role,
            "layout": layout,
            "schedule_index": index,
            "round": entry.get("round"),
            "mode": entry.get("mode"),
            "batch_size": entry.get("batch_size"),
        }
        if (
            token.get("kind") != "pyamplicol-paired-profile-token"
            or token.get("schema_version") != 1
            or completion.get("kind") != "pyamplicol-paired-profile-completion"
            or completion.get("schema_version") != 1
            or any(token.get(field) != value for field, value in expected.items())
            or any(completion.get(field) != value for field, value in expected.items())
            or completion.get("token_sha256") != token.get("content_sha256")
        ):
            raise EvidenceError(f"{label} slot {index} changed paired coordinates")
        (
            invocation,
            _,
            invocation_sha256,
            process_sha256,
            result_sha256,
        ) = _validated_schedule_worker_records(
            entry,
            f"{label}.slots[{index}]",
        )
        if (
            completion.get("worker_invocation_sha256") != invocation_sha256
            or completion.get("worker_result_record_sha256") != result_sha256
            or completion.get("worker_started_at_utc")
            != invocation.get("started_at_utc")
            or completion.get("worker_finished_at_utc")
            != invocation.get("finished_at_utc")
        ):
            raise EvidenceError(
                f"{label} slot {index} completion is not bound to its worker"
            )
        issued = _utc_timestamp(
            token.get("issued_at_utc"),
            f"{label}.slots[{index}].token.issued_at_utc",
        )
        started = _utc_timestamp(
            completion.get("worker_started_at_utc"),
            f"{label}.slots[{index}].completion.worker_started_at_utc",
        )
        finished = _utc_timestamp(
            completion.get("worker_finished_at_utc"),
            f"{label}.slots[{index}].completion.worker_finished_at_utc",
        )
        recorded = _utc_timestamp(
            completion.get("recorded_at_utc"),
            f"{label}.slots[{index}].completion.recorded_at_utc",
        )
        if not ready_at <= issued <= started <= finished <= recorded:
            raise EvidenceError(f"{label} slot {index} timestamps are inverted")
        slots[index] = {
            "token": dict(token),
            "completion": dict(completion),
            "issued": issued,
            "started": started,
            "finished": finished,
            "recorded": recorded,
            "worker_invocation_sha256": invocation_sha256,
            "worker_process_record_sha256": process_sha256,
            "worker_result_record_sha256": result_sha256,
        }
    if list(raw_completions) != [slots[index]["completion"] for index in sorted(slots)]:
        raise EvidenceError(f"{label} completion summary differs from its schedule")
    return {
        "session_id": session_id,
        "role": role,
        "layout": layout,
        "slots": slots,
        "ready_at": ready_at,
        "ready_record": dict(ready),
    }


def _validate_capture(
    payload: Mapping[str, Any],
    input_identity: Mapping[str, object],
    *,
    role: str,
    expected_layout: str,
) -> CaptureEvidence:
    label = f"{role}/{expected_layout}"
    if payload.get("kind") != CAPTURE_KIND:
        raise EvidenceError(f"{label} has the wrong capture kind")
    schema = _exact_int(payload.get("schema_version"), f"{label}.schema_version")
    if schema < MINIMUM_CAPTURE_SCHEMA:
        raise EvidenceError(
            f"{label} schema {schema} predates the required raw-evidence contract"
        )
    logical_process = _logical_process(payload.get("process"), f"{label}.process")
    expected_workload = (
        "single-runtime-selected-flow/helicity-sum"
        if expected_layout == "topology-replay"
        else "all-flows/runtime-selected-single-helicity"
    )
    if payload.get("workload") != expected_workload:
        raise EvidenceError(f"{label} has the wrong workload")

    configuration = _mapping(payload.get("configuration"), f"{label}.configuration")
    if configuration.get("lc_flow_layout") != expected_layout:
        raise EvidenceError(f"{label} is in the wrong layout slot")
    if configuration.get("generation_only") is not False:
        raise EvidenceError(f"{label} is a generation-only capture")
    configured_modes = list(
        _sequence(configuration.get("modes"), f"{label}.configuration.modes")
    )
    allowed_modes = {*CAPTURE_MODES, *DIAGNOSTIC_MODES}
    if set(configured_modes) != allowed_modes or len(configured_modes) != len(
        set(configured_modes)
    ):
        raise EvidenceError(
            f"{label} must contain compiled, recurrence, and eager exactly once"
        )
    configured_batches = tuple(
        _exact_int(value, f"{label}.configuration.batch_sizes")
        for value in _sequence(
            configuration.get("batch_sizes"),
            f"{label}.configuration.batch_sizes",
        )
    )
    if len(configured_batches) != len(BATCH_SIZES) or set(configured_batches) != set(
        BATCH_SIZES
    ):
        raise EvidenceError(f"{label} must contain exactly batches {list(BATCH_SIZES)}")
    subprocess_samples = _exact_int(
        configuration.get("subprocess_samples"),
        f"{label}.configuration.subprocess_samples",
        minimum=MINIMUM_SUBPROCESS_SAMPLES,
    )
    if subprocess_samples != MINIMUM_SUBPROCESS_SAMPLES:
        raise EvidenceError(
            f"{label} must retain exactly {MINIMUM_SUBPROCESS_SAMPLES} "
            "independent subprocess pairs"
        )
    internal_samples = _exact_int(
        configuration.get("minimum_samples"),
        f"{label}.configuration.minimum_samples",
        minimum=MINIMUM_INTERNAL_SAMPLES,
    )
    warmups = _exact_int(
        configuration.get("warmup_runs"),
        f"{label}.configuration.warmup_runs",
        minimum=MINIMUM_WARMUPS,
    )
    if warmups != MINIMUM_WARMUPS:
        raise EvidenceError(
            f"{label} must use exactly {MINIMUM_WARMUPS} warmup runs"
        )
    target_runtime = _positive_number(
        configuration.get("target_runtime_seconds"),
        f"{label}.configuration.target_runtime_seconds",
    )
    if target_runtime < MINIMUM_TARGET_RUNTIME_SECONDS:
        raise EvidenceError(
            f"{label} target runtime is below {MINIMUM_TARGET_RUNTIME_SECONDS:g}s"
        )
    if configuration.get("jit_optimization_level") != 3:
        raise EvidenceError(f"{label} compiled JIT is not configured for O3")
    acceptance = _validate_capture_acceptance(
        payload,
        configuration,
        expected_layout=expected_layout,
        subprocess_samples=subprocess_samples,
        internal_samples=internal_samples,
        warmups=warmups,
        configured_modes=configured_modes,
    )
    selected_union_projection = _capture_semantic_projection(
        acceptance,
        configuration=configuration,
        logical_process=logical_process,
        layout=expected_layout,
        label=label,
    )

    source = _mapping(payload.get("source"), f"{label}.source")
    revision = _string(source.get("revision"), f"{label}.source.revision")
    if len(revision) != _REVISION_LENGTH:
        raise EvidenceError(f"{label}.source.revision is not a full Git revision")
    try:
        int(revision, 16)
    except ValueError as error:
        raise EvidenceError(f"{label}.source.revision is not a Git revision") from error
    if (
        source.get("dirty") is not False
        or source.get("untracked_files_checked") is not True
    ):
        raise EvidenceError(f"{label} source checkout is not clean and authenticated")

    provenance = _mapping(payload.get("provenance"), f"{label}.provenance")
    host = dict(_mapping(provenance.get("host"), f"{label}.provenance.host"))
    runtime = _mapping(payload.get("runtime_provenance"), f"{label}.runtime_provenance")
    interpreter = _mapping(
        runtime.get("interpreter"), f"{label}.runtime_provenance.interpreter"
    )
    native_extension = _mapping(
        runtime.get("native_extension"),
        f"{label}.runtime_provenance.native_extension",
    )
    build_info = _mapping(
        _mapping(
            runtime.get("active_build_info"),
            f"{label}.runtime_provenance.active_build_info",
        ).get("payload"),
        f"{label}.runtime_provenance.active_build_info.payload",
    )
    if build_info.get("source_revision") != revision:
        raise EvidenceError(
            f"{label} installed build does not match its source revision"
        )
    build_identity = {
        "source_revision": revision,
        "native_extension_sha256": _valid_sha256(
            native_extension.get("sha256"),
            f"{label}.runtime_provenance.native_extension.sha256",
        ),
        "native_build_inputs_sha256": _valid_sha256(
            native_extension.get("build_inputs_sha256"),
            f"{label}.runtime_provenance.native_extension.build_inputs_sha256",
        ),
        "package_version": _string(
            native_extension.get("package_version"),
            f"{label}.runtime_provenance.native_extension.package_version",
        ),
        "build_info_version": _string(
            build_info.get("version"),
            f"{label}.runtime_provenance.active_build_info.payload.version",
        ),
    }
    interpreter_identity = {
        "implementation": _string(
            interpreter.get("implementation"),
            f"{label}.runtime_provenance.interpreter.implementation",
        ),
        "python_version": _string(
            interpreter.get("python_version"),
            f"{label}.runtime_provenance.interpreter.python_version",
        ),
        "executable_sha256": _valid_sha256(
            interpreter.get("sha256"),
            f"{label}.runtime_provenance.interpreter.sha256",
        ),
    }
    interpreter_path = _string(
        interpreter.get("path"),
        f"{label}.runtime_provenance.interpreter.path",
    )
    interpreter_resolved_path = _string(
        interpreter.get("resolved_path"),
        f"{label}.runtime_provenance.interpreter.resolved_path",
    )
    model_identities = _mapping(
        configuration.get("model_identities"),
        f"{label}.configuration.model_identities",
    )
    if set(model_identities) != set(configured_modes):
        raise EvidenceError(
            f"{label} prepared-model identity inventory is incomplete"
        )
    prepared_paths: set[str] = set()
    prepared_hashes: set[str] = set()
    for mode in configured_modes:
        model_identity = _mapping(
            model_identities.get(mode),
            f"{label}.configuration.model_identities.{mode}",
        )
        if model_identity.get("kind") != "explicit-prepared-model":
            raise EvidenceError(
                f"{label} paired campaign did not use an explicit prepared model"
            )
        model_file = _mapping(
            model_identity.get("file"),
            f"{label}.configuration.model_identities.{mode}.file",
        )
        prepared_paths.add(
            _string(
                model_file.get("resolved_path"),
                f"{label}.configuration.model_identities.{mode}.file.resolved_path",
            )
        )
        prepared_hashes.add(
            _valid_sha256(
                model_file.get("sha256"),
                f"{label}.configuration.model_identities.{mode}.file.sha256",
            )
        )
    if len(prepared_paths) != 1 or len(prepared_hashes) != 1:
        raise EvidenceError(f"{label} modes use different prepared model packs")
    prepared_model_path = prepared_paths.pop()
    prepared_model_sha256 = prepared_hashes.pop()

    schedule = _mapping(payload.get("profile_schedule"), f"{label}.profile_schedule")
    if (
        schedule.get("kind") != "pyamplicol-interleaved-subprocess-profile-schedule"
        or schedule.get("schema_version") != 2
        or schedule.get("algorithm")
        != "round-major-cyclic-mode-and-batch-interleave-v1"
        or schedule.get("sample_unit") != "independent-profile-worker-subprocess"
        or schedule.get("subprocess_samples_per_cell") != subprocess_samples
        or set(
            _sequence(
                schedule.get("modes"),
                f"{label}.profile_schedule.modes",
            )
        )
        != set(configured_modes)
        or set(
            _sequence(
                schedule.get("batch_sizes"),
                f"{label}.profile_schedule.batch_sizes",
            )
        )
        != set(BATCH_SIZES)
    ):
        raise EvidenceError(f"{label} profile schedule identity is inconsistent")
    entries = _sequence(schedule.get("entries"), f"{label}.profile_schedule.entries")
    expected_entry_count = len(configured_modes) * len(BATCH_SIZES) * subprocess_samples
    if len(entries) != expected_entry_count:
        raise EvidenceError(
            f"{label} profile schedule has {len(entries)} entries; "
            f"{expected_entry_count} are required"
        )
    schedule_by_index: dict[int, Mapping[str, Any]] = {}
    for position, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"{label}.profile_schedule.entries[{position}]")
        index = _exact_int(
            entry.get("schedule_index"),
            f"{label}.profile_schedule.entries[{position}].schedule_index",
            minimum=0,
        )
        if index != position or index in schedule_by_index:
            raise EvidenceError(f"{label} profile schedule index is not canonical")
        schedule_by_index[index] = entry
    paired_profile_coordination = _validate_paired_profile_coordination(
        payload,
        role=role,
        layout=expected_layout,
        revision=revision,
        schedule_by_index=schedule_by_index,
    )

    generations = _mapping(payload.get("generation"), f"{label}.generation")
    raw_profiles = _mapping(payload.get("profiles"), f"{label}.profiles")
    if set(generations) != set(configured_modes) or set(raw_profiles) != set(
        configured_modes
    ):
        raise EvidenceError(
            f"{label} three-lane generation/profile inventory is incomplete"
        )
    generation_evidence: dict[str, GenerationEvidence] = {}
    workload_identities: dict[str, Mapping[str, object]] = {}
    cells: dict[tuple[str, int], CellEvidence] = {}
    seen_addresses: set[str] = set()
    eager_acceptance = _mapping(
        acceptance.get("eager_diagnostic"),
        f"{label}.capture_acceptance.eager_diagnostic",
    )
    eager_admissible = bool(
        "eager" in configured_modes
        and eager_acceptance.get("complete") is True
        and eager_acceptance.get("passes") is True
    )
    parsed_modes = (*CAPTURE_MODES, "eager") if eager_admissible else CAPTURE_MODES
    for mode in parsed_modes:
        generation = _mapping(generations.get(mode), f"{label}.generation.{mode}")
        if generation.get("mode") != mode:
            raise EvidenceError(f"{label}.generation.{mode} has the wrong mode")
        if generation.get("generation_reused") is not False:
            raise EvidenceError(
                f"{label}.generation.{mode} was reused; a fresh generation "
                "time is required"
            )
        generation_wall = _positive_number(
            generation.get("generation_wall_seconds"),
            f"{label}.generation.{mode}.generation_wall_seconds",
        )
        stats = _mapping(
            generation.get("artifact_stats"),
            f"{label}.generation.{mode}.artifact_stats",
        )
        _exact_int(
            stats.get("file_count"),
            f"{label}.generation.{mode}.artifact_stats.file_count",
            minimum=1,
        )
        payload_size = _exact_int(
            stats.get("size_bytes"),
            f"{label}.generation.{mode}.artifact_stats.size_bytes",
            minimum=1,
        )
        generation_peak = int(
            _rss_observation(
                generation.get("peak_rss"),
                f"{label}.generation.{mode}.peak_rss",
            )
        )
        generation_evidence[mode] = GenerationEvidence(
            wall_seconds=generation_wall,
            payload_size_bytes=payload_size,
            peak_rss_bytes=generation_peak,
        )

        profile = _mapping(raw_profiles.get(mode), f"{label}.profiles.{mode}")
        workload_identities[mode] = _workload_identity(
            profile=profile,
            generation=generation,
            mode=mode,
            layout=expected_layout,
            logical_process=logical_process,
            expected_workload=expected_workload,
            selected_union_projection=selected_union_projection,
            label=f"{label}.profiles.{mode}",
        )
        validation_fixture = _mapping(
            workload_identities[mode].get("validation_fixture"),
            f"{label}.profiles.{mode}.validation_fixture",
        )
        fixture_points_sha256 = _valid_sha256(
            validation_fixture.get("points_sha256"),
            f"{label}.profiles.{mode}.validation_fixture.points_sha256",
        )
        aggregates = _sequence(
            profile.get("profiles"),
            f"{label}.profiles.{mode}.profiles",
        )
        by_batch: dict[int, Mapping[str, Any]] = {}
        for aggregate_index, raw_aggregate in enumerate(aggregates):
            aggregate = _mapping(
                raw_aggregate,
                f"{label}.profiles.{mode}.profiles[{aggregate_index}]",
            )
            batch = _exact_int(
                aggregate.get("batch_size"),
                f"{label}.profiles.{mode}.profiles[{aggregate_index}].batch_size",
                minimum=1,
            )
            if batch in by_batch:
                raise EvidenceError(f"{label}.profiles.{mode} duplicates batch {batch}")
            by_batch[batch] = aggregate
        if set(by_batch) != set(BATCH_SIZES):
            raise EvidenceError(
                f"{label}.profiles.{mode} does not contain exactly {list(BATCH_SIZES)}"
            )
        for batch_size in BATCH_SIZES:
            cells[(mode, batch_size)] = _extract_cell(
                aggregate=by_batch[batch_size],
                mode=mode,
                batch_size=batch_size,
                subprocess_samples=subprocess_samples,
                internal_samples=internal_samples,
                warmups=warmups,
                target_runtime=target_runtime,
                fixture_points_sha256=fixture_points_sha256,
                schedule_by_index=schedule_by_index,
                seen_addresses=seen_addresses,
                label=f"{label}.profiles.{mode}.batch[{batch_size}]",
            )

    # Each round must retain exactly one independently scheduled subprocess for
    # every lane and batch. Compiled/recurrence ratios are paired by this
    # shared round identity, and eager is independently evaluated below.
    for batch_size in BATCH_SIZES:
        for round_index in range(subprocess_samples):
            pair = [
                entry
                for entry in schedule_by_index.values()
                if entry.get("batch_size") == batch_size
                and entry.get("round") == round_index
                and entry.get("mode") in configured_modes
            ]
            if len(pair) != len(configured_modes) or {
                entry.get("mode") for entry in pair
            } != set(configured_modes):
                raise EvidenceError(
                    f"{label} batch {batch_size} round {round_index} does not "
                    "contain one subprocess for every capture lane"
                )

    campaign_identity = {
        "logical_process": logical_process,
        "layout": expected_layout,
        "workload": expected_workload,
        "batch_sizes": list(BATCH_SIZES),
        "subprocess_samples": subprocess_samples,
        "minimum_internal_samples": configuration.get("minimum_samples"),
        "warmup_runs": warmups,
        "target_runtime_seconds": target_runtime,
        "color_flow_request": configuration.get("color_flow_request"),
        "helicity_request": configuration.get("helicity_request"),
        "validation_seed": configuration.get("validation_seed"),
        "point_tile_size": configuration.get("point_tile_size"),
        "interpreter": interpreter_identity,
        "configured_modes": list(configured_modes),
    }
    addressed_input_identity = dict(input_identity)
    addressed_input_identity["capture_acceptance_sha256"] = _sha256(acceptance)
    return CaptureEvidence(
        input_identity=addressed_input_identity,
        layout=expected_layout,
        logical_process=logical_process,
        source_revision=revision,
        build_identity=build_identity,
        host_identity=host,
        campaign_identity=campaign_identity,
        workload_identities=workload_identities,
        cells=cells,
        generation=generation_evidence,
        configured_modes=tuple(configured_modes),
        paired_profile_coordination=paired_profile_coordination,
        eager_diagnostic=dict(eager_acceptance),
        interpreter_path=interpreter_path,
        interpreter_resolved_path=interpreter_resolved_path,
        prepared_model_path=prepared_model_path,
        prepared_model_sha256=prepared_model_sha256,
    )


def _geometric_mean(values: Sequence[float], label: str) -> float:
    checked = [_positive_number(value, label) for value in values]
    if not checked:
        raise EvidenceError(f"{label} is empty")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def _runtime_cell_comparison(
    *,
    layout: str,
    mode: str,
    batch_size: int,
    baseline: CellEvidence,
    candidate: CellEvidence,
) -> dict[str, object]:
    relative_limit = RUNTIME_RELATIVE_LIMIT * baseline.runtime.median
    noise_limit = (
        baseline.runtime.median
        + RUNTIME_NOISE_MAD_MULTIPLIER * baseline.runtime.raw_mad
    )
    passes_relative = candidate.runtime.median <= relative_limit
    passes_noise = candidate.runtime.median <= noise_limit
    gain_at_least_ten_percent = (
        candidate.runtime.median
        <= (1.0 - RESOURCE_EXCEPTION_GAIN) * baseline.runtime.median
    )
    gain_beyond_noise = (
        candidate.runtime.median
        <= baseline.runtime.median
        - RUNTIME_NOISE_MAD_MULTIPLIER * baseline.runtime.raw_mad
    )
    return {
        "layout": layout,
        "mode": mode,
        "batch_size": batch_size,
        "baseline": baseline.runtime.as_dict(),
        "candidate": candidate.runtime.as_dict(),
        "candidate_to_baseline_ratio": (
            candidate.runtime.median / baseline.runtime.median
        ),
        "relative_limit_seconds_per_point": relative_limit,
        "noise_limit_seconds_per_point": noise_limit,
        "passes_relative_limit": passes_relative,
        "passes_noise_limit": passes_noise,
        "passes": passes_relative and passes_noise,
        "resource_exception_runtime_gain": {
            "gain_at_least_10_percent": gain_at_least_ten_percent,
            "gain_beyond_3_baseline_raw_mad": gain_beyond_noise,
            "passes": gain_at_least_ten_percent and gain_beyond_noise,
        },
    }


def _paired_ratio(
    *,
    layout: str,
    batch_size: int,
    capture: CaptureEvidence,
) -> dict[str, object]:
    compiled = capture.cells[("compiled", batch_size)].by_round
    recurrence = capture.cells[("recurrence", batch_size)].by_round
    if set(compiled) != set(recurrence):
        raise EvidenceError(
            f"{layout} batch {batch_size} compiled/recurrence rounds do not match"
        )
    rounds = sorted(compiled)
    ratios = [compiled[round_index] / recurrence[round_index] for round_index in rounds]
    distribution = _distribution(ratios, f"{layout}/{batch_size} paired ratio")
    upper_bound = (
        distribution.median + RUNTIME_NOISE_MAD_MULTIPLIER * distribution.raw_mad
    )
    return {
        "rounds": rounds,
        "compiled_to_recurrence_ratio": distribution.as_dict(),
        "median_plus_3_raw_mad": upper_bound,
    }


def _resource_comparison(
    *,
    layout: str,
    mode: str,
    metric: str,
    baseline: Distribution,
    candidate: Distribution,
    runtime_gain: Mapping[str, object],
    batch_size: int | None = None,
) -> dict[str, object]:
    ratio = candidate.median / baseline.median
    within_growth_limit = ratio <= RESOURCE_GROWTH_LIMIT
    exception = runtime_gain.get("passes") is True
    result: dict[str, object] = {
        "layout": layout,
        "mode": mode,
        "metric": metric,
        "baseline": baseline.as_dict(),
        "candidate": candidate.as_dict(),
        "candidate_to_baseline_ratio": ratio,
        "within_3_percent_growth_limit": within_growth_limit,
        "runtime_gain_exception": dict(runtime_gain),
        "passes": within_growth_limit or exception,
    }
    if batch_size is not None:
        result["batch_size"] = batch_size
    return result


def _single_resource_comparison(
    *,
    layout: str,
    mode: str,
    metric: str,
    baseline: float,
    candidate: float,
    runtime_gain: Mapping[str, object],
) -> dict[str, object]:
    baseline_distribution = Distribution((baseline,), baseline, 0.0)
    candidate_distribution = Distribution((candidate,), candidate, 0.0)
    return _resource_comparison(
        layout=layout,
        mode=mode,
        metric=metric,
        baseline=baseline_distribution,
        candidate=candidate_distribution,
        runtime_gain=runtime_gain,
    )


def _assert_role_consistency(
    captures: Mapping[str, CaptureEvidence],
    *,
    role: str,
) -> None:
    topology = captures["topology-replay"]
    union = captures["all-flow-union"]
    if topology.source_revision != union.source_revision:
        raise EvidenceError(f"{role} captures use different source revisions")
    if topology.build_identity != union.build_identity:
        raise EvidenceError(f"{role} captures use different native builds")
    if topology.host_identity != union.host_identity:
        raise EvidenceError(f"{role} captures were not measured on the same host")
    if topology.logical_process != union.logical_process:
        raise EvidenceError(f"{role} captures use different processes")
    if topology.paired_profile_coordination.get(
        "session_id"
    ) != union.paired_profile_coordination.get("session_id"):
        raise EvidenceError(f"{role} captures use different paired sessions")
    topology_campaign = dict(topology.campaign_identity)
    union_campaign = dict(union.campaign_identity)
    for field in ("layout", "workload", "color_flow_request", "helicity_request"):
        topology_campaign.pop(field, None)
        union_campaign.pop(field, None)
    if topology_campaign != union_campaign:
        raise EvidenceError(f"{role} capture campaign configuration drifted by layout")


def _assert_paired_role_order(
    baseline: CaptureEvidence,
    candidate: CaptureEvidence,
    *,
    layout: str,
) -> None:
    baseline_coordination = baseline.paired_profile_coordination
    candidate_coordination = candidate.paired_profile_coordination
    if baseline_coordination.get("session_id") != candidate_coordination.get(
        "session_id"
    ):
        raise EvidenceError(f"{layout} roles use different paired sessions")
    baseline_slots = _mapping(
        baseline_coordination.get("slots"),
        f"{layout}.baseline paired slots",
    )
    candidate_slots = _mapping(
        candidate_coordination.get("slots"),
        f"{layout}.candidate paired slots",
    )
    if set(baseline_slots) != set(candidate_slots):
        raise EvidenceError(f"{layout} paired role slots differ")
    previous_finished: dt.datetime | None = None
    for index in sorted(baseline_slots):
        baseline_slot = _mapping(
            baseline_slots[index], f"{layout}.baseline paired slot {index}"
        )
        candidate_slot = _mapping(
            candidate_slots[index], f"{layout}.candidate paired slot {index}"
        )
        baseline_token = _mapping(
            baseline_slot.get("token"), f"{layout}.baseline token {index}"
        )
        candidate_token = _mapping(
            candidate_slot.get("token"), f"{layout}.candidate token {index}"
        )
        coordinates = ("schedule_index", "round", "mode", "batch_size", "pair_index")
        if (
            any(
                baseline_token.get(field) != candidate_token.get(field)
                for field in coordinates
            )
            or baseline_token.get("pair_index") != index
        ):
            raise EvidenceError(f"{layout} paired slot {index} coordinates differ")
        round_index = _exact_int(
            baseline_token.get("round"),
            f"{layout}.paired slot {index}.round",
            minimum=0,
        )
        expected_order = (
            ("baseline", "candidate")
            if round_index % 2 == 0
            else ("candidate", "baseline")
        )
        by_role = {
            "baseline": baseline_slot,
            "candidate": candidate_slot,
        }
        order_coordinates = tuple(
            (
                _exact_int(
                    _mapping(
                        by_role[role].get("token"),
                        f"{layout}.{role} token {index}",
                    ).get("order_in_pair"),
                    f"{layout}.{role} token {index}.order_in_pair",
                    minimum=0,
                ),
                role,
            )
            for role in ("baseline", "candidate")
        )
        if {order for order, _ in order_coordinates} != {0, 1}:
            raise EvidenceError(
                f"{layout} paired slot {index} has invalid role positions"
            )
        observed_order = tuple(role for _, role in sorted(order_coordinates))
        if observed_order != expected_order:
            raise EvidenceError(
                f"{layout} paired slot {index} does not alternate role order"
            )
        for identity in (
            "worker_invocation_sha256",
            "worker_process_record_sha256",
            "worker_result_record_sha256",
        ):
            if baseline_slot.get(identity) == candidate_slot.get(identity):
                raise EvidenceError(
                    f"{layout} paired slot {index} reuses {identity} "
                    "identity across roles"
                )
        first = by_role[expected_order[0]]
        second = by_role[expected_order[1]]
        first_issued = first.get("issued")
        first_started = first.get("started")
        first_finished = first.get("finished")
        second_issued = second.get("issued")
        second_started = second.get("started")
        second_finished = second.get("finished")
        if not all(
            isinstance(value, dt.datetime)
            for value in (
                first_issued,
                first_started,
                first_finished,
                second_issued,
                second_started,
                second_finished,
            )
        ):
            raise EvidenceError(f"{layout} paired slot {index} lacks timestamps")
        assert isinstance(first_issued, dt.datetime)
        assert isinstance(first_started, dt.datetime)
        assert isinstance(first_finished, dt.datetime)
        assert isinstance(second_issued, dt.datetime)
        assert isinstance(second_started, dt.datetime)
        assert isinstance(second_finished, dt.datetime)
        if first_finished > second_issued:
            raise EvidenceError(
                f"{layout} paired slot {index} admitted the second role "
                "before the first worker finished"
            )
        if first_finished > second_started:
            raise EvidenceError(f"{layout} paired slot {index} workers overlap")
        if previous_finished is not None and previous_finished > first_issued:
            raise EvidenceError(f"{layout} paired slots overlap or reorder")
        previous_finished = second_finished


def _assert_cross_layout_sequence(
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> None:
    topology_finished: list[dt.datetime] = []
    for role, captures in (("baseline", baseline), ("candidate", candidate)):
        slots = _mapping(
            captures["topology-replay"].paired_profile_coordination.get("slots"),
            f"{role} topology paired slots",
        )
        for index, raw_slot in slots.items():
            finished = _mapping(
                raw_slot,
                f"{role} topology paired slot {index}",
            ).get("finished")
            if not isinstance(finished, dt.datetime):
                raise EvidenceError(
                    f"{role} topology paired slot {index} lacks a finish timestamp"
                )
            topology_finished.append(finished)
    union_ready = [
        captures["all-flow-union"].paired_profile_coordination.get("ready_at")
        for captures in (baseline, candidate)
    ]
    if not topology_finished or not all(
        isinstance(value, dt.datetime) for value in union_ready
    ):
        raise EvidenceError("paired campaign lacks cross-layout timestamps")
    if max(topology_finished) > min(union_ready):
        raise EvidenceError(
            "paired campaign layouts overlap or reorder; all-flow-union became "
            "ready before topology-replay completed"
        )


def _assert_global_worker_identity_separation(
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> None:
    for identity in (
        "worker_invocation_sha256",
        "worker_process_record_sha256",
        "worker_result_record_sha256",
    ):
        seen: dict[str, tuple[str, str, object]] = {}
        for role, captures in (("baseline", baseline), ("candidate", candidate)):
            for layout in LAYOUTS:
                slots = _mapping(
                    captures[layout].paired_profile_coordination.get("slots"),
                    f"{role}/{layout} paired slots",
                )
                for index, raw_slot in slots.items():
                    slot = _mapping(
                        raw_slot,
                        f"{role}/{layout} paired slot {index}",
                    )
                    digest = _valid_sha256(
                        slot.get(identity),
                        f"{role}/{layout} paired slot {index}.{identity}",
                    )
                    previous = seen.get(digest)
                    if previous is not None:
                        raise EvidenceError(
                            f"{role}/{layout} paired slot {index} reuses {identity} "
                            f"identity from {previous[0]}/{previous[1]} "
                            f"slot {previous[2]}"
                        )
                    seen[digest] = (role, layout, index)


def _assert_externally_pinned_role_inputs(
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
    *,
    expected_candidate_source_revision: str,
    expected_baseline_native_build_inputs_sha256: str,
    expected_candidate_native_build_inputs_sha256: str,
    expected_baseline_prepared_model_sha256: str,
    expected_candidate_prepared_model_sha256: str,
) -> dict[str, object]:
    expected_candidate_source_revision = _valid_revision(
        expected_candidate_source_revision,
        "expected candidate source revision",
    )
    expected = {
        "baseline": {
            "native_build_inputs_sha256": _valid_sha256(
                expected_baseline_native_build_inputs_sha256,
                "expected baseline native-build-input digest",
            ),
            "prepared_model_sha256": _valid_sha256(
                expected_baseline_prepared_model_sha256,
                "expected baseline prepared-model digest",
            ),
        },
        "candidate": {
            "native_build_inputs_sha256": _valid_sha256(
                expected_candidate_native_build_inputs_sha256,
                "expected candidate native-build-input digest",
            ),
            "prepared_model_sha256": _valid_sha256(
                expected_candidate_prepared_model_sha256,
                "expected candidate prepared-model digest",
            ),
        },
    }
    if expected_candidate_source_revision == BASELINE_REVISION:
        raise EvidenceError(
            "expected candidate source revision still identifies the baseline"
        )
    for role, captures in (("baseline", baseline), ("candidate", candidate)):
        for layout, capture in captures.items():
            if (
                role == "candidate"
                and capture.source_revision != expected_candidate_source_revision
            ):
                raise EvidenceError(
                    f"candidate {layout} capture does not match the externally "
                    "pinned source revision"
                )
            if (
                capture.build_identity.get("native_build_inputs_sha256")
                != expected[role]["native_build_inputs_sha256"]
            ):
                raise EvidenceError(
                    f"{role} {layout} capture does not match the externally "
                    "pinned native-build-input digest"
                )
            if (
                capture.prepared_model_sha256
                != expected[role]["prepared_model_sha256"]
            ):
                raise EvidenceError(
                    f"{role} {layout} capture does not match the externally "
                    "pinned prepared-model digest"
                )
    return {
        "candidate_source_revision": expected_candidate_source_revision,
        "roles": expected,
    }


def _assert_pair_identity(
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> None:
    if baseline["topology-replay"].source_revision != BASELINE_REVISION:
        raise EvidenceError(
            "baseline source revision is not the immutable migration baseline "
            f"{BASELINE_REVISION}"
        )
    if candidate["topology-replay"].source_revision == BASELINE_REVISION:
        raise EvidenceError("candidate capture still identifies the baseline revision")
    baseline_host = baseline["topology-replay"].host_identity
    candidate_host = candidate["topology-replay"].host_identity
    if baseline_host != candidate_host:
        raise EvidenceError(
            "baseline and candidate captures were not run on the same host"
        )
    _assert_global_worker_identity_separation(baseline, candidate)
    for layout in LAYOUTS:
        baseline_capture = baseline[layout]
        candidate_capture = candidate[layout]
        _assert_paired_role_order(
            baseline_capture,
            candidate_capture,
            layout=layout,
        )
        if baseline_capture.campaign_identity != candidate_capture.campaign_identity:
            raise EvidenceError(
                f"{layout} baseline/candidate campaign configurations differ"
            )
        modes = list(CAPTURE_MODES)
        if (
            baseline_capture.eager_diagnostic.get("complete") is True
            and baseline_capture.eager_diagnostic.get("passes") is True
            and candidate_capture.eager_diagnostic.get("complete") is True
            and candidate_capture.eager_diagnostic.get("passes") is True
        ):
            modes.append("eager")
        for mode in modes:
            if (
                baseline_capture.workload_identities[mode]
                != candidate_capture.workload_identities[mode]
            ):
                raise EvidenceError(
                    f"{layout}/{mode} baseline/candidate workload identities differ"
                )
    _assert_cross_layout_sequence(baseline, candidate)


def compare_captures(
    *,
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> dict[str, object]:
    """Build the content-addressed comparison for four validated captures."""

    if set(baseline) != set(LAYOUTS) or set(candidate) != set(LAYOUTS):
        raise EvidenceError("comparison requires both layouts for both revisions")
    _assert_role_consistency(baseline, role="baseline")
    _assert_role_consistency(candidate, role="candidate")
    _assert_pair_identity(baseline, candidate)

    runtime_cells: list[dict[str, object]] = []
    eager_runtime_cells: list[dict[str, object]] = []
    runtime_by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
    failures: list[str] = []
    eager_failures: list[str] = []
    all_captures = [
        captures[layout] for captures in (baseline, candidate) for layout in LAYOUTS
    ]
    eager_requested = all(
        "eager" in capture.configured_modes for capture in all_captures
    )
    eager_available = bool(
        eager_requested
        and all(
            capture.eager_diagnostic.get("complete") is True
            and capture.eager_diagnostic.get("passes") is True
            for capture in all_captures
        )
    )
    eager_inadmissibility_reasons = [
        {
            "role": role,
            "layout": layout,
            "reasons": capture.eager_diagnostic.get("ineligibility_reasons"),
        }
        for role, captures in (("baseline", baseline), ("candidate", candidate))
        for layout, capture in captures.items()
        if "eager" in capture.configured_modes
        and (
            capture.eager_diagnostic.get("complete") is not True
            or capture.eager_diagnostic.get("passes") is not True
        )
    ]
    for layout in LAYOUTS:
        for mode in CAPTURE_MODES:
            for batch_size in BATCH_SIZES:
                cell_result = _runtime_cell_comparison(
                    layout=layout,
                    mode=mode,
                    batch_size=batch_size,
                    baseline=baseline[layout].cells[(mode, batch_size)],
                    candidate=candidate[layout].cells[(mode, batch_size)],
                )
                runtime_cells.append(cell_result)
                runtime_by_key[(layout, mode, batch_size)] = cell_result
                if cell_result["passes"] is not True:
                    failures.append(
                        f"runtime regression: {layout}/{mode}/batch-{batch_size}"
                    )
    if eager_available:
        for layout in LAYOUTS:
            for batch_size in BATCH_SIZES:
                cell_result = _runtime_cell_comparison(
                    layout=layout,
                    mode="eager",
                    batch_size=batch_size,
                    baseline=baseline[layout].cells[("eager", batch_size)],
                    candidate=candidate[layout].cells[("eager", batch_size)],
                )
                eager_runtime_cells.append(cell_result)
                runtime_by_key[(layout, "eager", batch_size)] = cell_result
                if cell_result["passes"] is not True:
                    eager_failures.append(
                        "eager diagnostic runtime regression: "
                        f"{layout}/eager/batch-{batch_size}"
                    )

    paired_ratios: list[dict[str, object]] = []
    for layout in LAYOUTS:
        for batch_size in RATIO_BATCH_SIZES:
            baseline_ratio = _paired_ratio(
                layout=layout,
                batch_size=batch_size,
                capture=baseline[layout],
            )
            candidate_ratio = _paired_ratio(
                layout=layout,
                batch_size=batch_size,
                capture=candidate[layout],
            )
            candidate_upper = _positive_number(
                candidate_ratio["median_plus_3_raw_mad"],
                f"{layout}/{batch_size} candidate paired-ratio upper bound",
            )
            passes = candidate_upper <= COMPILED_RECURRENCE_RATIO_LIMIT
            paired_ratios.append(
                {
                    "layout": layout,
                    "batch_size": batch_size,
                    "baseline": baseline_ratio,
                    "candidate": candidate_ratio,
                    "limit": COMPILED_RECURRENCE_RATIO_LIMIT,
                    "passes": passes,
                }
            )
            if not passes:
                failures.append(
                    f"compiled/recurrence ratio regression: {layout}/batch-{batch_size}"
                )

    generation_cells: list[dict[str, object]] = []
    generation_ratios: list[float] = []
    for layout in LAYOUTS:
        for mode in MODES:
            baseline_wall = baseline[layout].generation[mode].wall_seconds
            candidate_wall = candidate[layout].generation[mode].wall_seconds
            ratio = candidate_wall / baseline_wall
            passes = ratio <= GENERATION_CELL_LIMIT
            generation_ratios.append(ratio)
            generation_cells.append(
                {
                    "layout": layout,
                    "mode": mode,
                    "baseline_wall_seconds": baseline_wall,
                    "candidate_wall_seconds": candidate_wall,
                    "candidate_to_baseline_ratio": ratio,
                    "limit": GENERATION_CELL_LIMIT,
                    "passes": passes,
                }
            )
            if not passes:
                failures.append(f"generation regression: {layout}/{mode}")
    generation_geomean = _geometric_mean(
        generation_ratios,
        "generation candidate/baseline ratios",
    )
    generation_geomean_passes = generation_geomean <= GENERATION_GEOMEAN_LIMIT
    if not generation_geomean_passes:
        failures.append("generation geometric-mean regression")

    eager_generation_cells: list[dict[str, object]] = []
    eager_generation_ratios: list[float] = []
    if eager_available:
        for layout in LAYOUTS:
            baseline_wall = baseline[layout].generation["eager"].wall_seconds
            candidate_wall = candidate[layout].generation["eager"].wall_seconds
            ratio = candidate_wall / baseline_wall
            eager_generation_ratios.append(ratio)
            passes = ratio <= GENERATION_CELL_LIMIT
            eager_generation_cells.append(
                {
                    "layout": layout,
                    "mode": "eager",
                    "baseline_wall_seconds": baseline_wall,
                    "candidate_wall_seconds": candidate_wall,
                    "candidate_to_baseline_ratio": ratio,
                    "limit": GENERATION_CELL_LIMIT,
                    "passes": passes,
                }
            )
            if not passes:
                eager_failures.append(
                    f"eager diagnostic generation regression: {layout}/eager"
                )
    eager_generation_geomean = (
        _geometric_mean(
            eager_generation_ratios,
            "eager generation candidate/baseline ratios",
        )
        if eager_available
        else None
    )
    eager_generation_geomean_passes = (
        eager_generation_geomean <= GENERATION_GEOMEAN_LIMIT
        if eager_generation_geomean is not None
        else None
    )
    if eager_generation_geomean_passes is False:
        eager_failures.append("eager diagnostic generation geometric-mean regression")

    # A mode-level payload or generation-RSS exception requires a >=10%
    # geometric-mean runtime gain and a gain beyond three baseline raw MADs in
    # every measured batch.  Cell-level cold-load/RSS exceptions use the
    # corresponding batch's runtime evidence directly.
    mode_runtime_gain: dict[tuple[str, str], dict[str, object]] = {}
    for layout in LAYOUTS:
        modes_with_diagnostic = (
            (*CAPTURE_MODES, "eager") if eager_available else CAPTURE_MODES
        )
        for mode in modes_with_diagnostic:
            comparisons = [
                runtime_by_key[(layout, mode, batch_size)] for batch_size in BATCH_SIZES
            ]
            ratios = [
                _positive_number(
                    comparison["candidate_to_baseline_ratio"],
                    f"{layout}/{mode} candidate/baseline runtime ratio",
                )
                for comparison in comparisons
            ]
            ratio_geomean = _geometric_mean(
                ratios,
                f"{layout}/{mode} runtime ratios",
            )
            beyond_noise_all = all(
                _mapping(
                    comparison["resource_exception_runtime_gain"],
                    "runtime gain evidence",
                ).get("gain_beyond_3_baseline_raw_mad")
                is True
                for comparison in comparisons
            )
            mode_runtime_gain[(layout, mode)] = {
                "runtime_ratio_geometric_mean": ratio_geomean,
                "gain_at_least_10_percent": (
                    ratio_geomean <= 1.0 - RESOURCE_EXCEPTION_GAIN
                ),
                "gain_beyond_3_baseline_raw_mad_in_every_batch": beyond_noise_all,
                "passes": (
                    ratio_geomean <= 1.0 - RESOURCE_EXCEPTION_GAIN and beyond_noise_all
                ),
            }

    resources: list[dict[str, object]] = []
    eager_resources: list[dict[str, object]] = []
    for layout in LAYOUTS:
        modes_with_diagnostic = (
            (*CAPTURE_MODES, "eager") if eager_available else CAPTURE_MODES
        )
        for mode in modes_with_diagnostic:
            baseline_generation = baseline[layout].generation[mode]
            candidate_generation = candidate[layout].generation[mode]
            aggregate_gain = mode_runtime_gain[(layout, mode)]
            mode_resources = eager_resources if mode == "eager" else resources
            mode_resources.extend(
                (
                    _single_resource_comparison(
                        layout=layout,
                        mode=mode,
                        metric="artifact_payload_size_bytes",
                        baseline=float(baseline_generation.payload_size_bytes),
                        candidate=float(candidate_generation.payload_size_bytes),
                        runtime_gain=aggregate_gain,
                    ),
                    _single_resource_comparison(
                        layout=layout,
                        mode=mode,
                        metric="generation_peak_rss_bytes",
                        baseline=float(baseline_generation.peak_rss_bytes),
                        candidate=float(candidate_generation.peak_rss_bytes),
                        runtime_gain=aggregate_gain,
                    ),
                )
            )
            for batch_size in BATCH_SIZES:
                baseline_cell = baseline[layout].cells[(mode, batch_size)]
                candidate_cell = candidate[layout].cells[(mode, batch_size)]
                gain = _mapping(
                    runtime_by_key[(layout, mode, batch_size)][
                        "resource_exception_runtime_gain"
                    ],
                    "runtime gain evidence",
                )
                mode_resources.extend(
                    (
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="cold_load_seconds",
                            baseline=baseline_cell.cold_load,
                            candidate=candidate_cell.cold_load,
                            runtime_gain=gain,
                        ),
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="cold_load_peak_rss_bytes",
                            baseline=baseline_cell.cold_load_rss,
                            candidate=candidate_cell.cold_load_rss,
                            runtime_gain=gain,
                        ),
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="profiled_peak_rss_bytes",
                            baseline=baseline_cell.profiled_rss,
                            candidate=candidate_cell.profiled_rss,
                            runtime_gain=gain,
                        ),
                    )
                )
    for diagnostic, inventory in (
        (False, resources),
        (True, eager_resources),
    ):
        for resource in inventory:
            if resource["passes"] is not True:
                batch_suffix = (
                    ""
                    if "batch_size" not in resource
                    else f"/batch-{resource['batch_size']}"
                )
                prefix = (
                    "eager diagnostic resource regression"
                    if diagnostic
                    else "resource regression"
                )
                failure_inventory = eager_failures if diagnostic else failures
                failure_inventory.append(
                    f"{prefix}: "
                    f"{resource['layout']}/{resource['mode']}{batch_suffix}/"
                    f"{resource['metric']}"
                )

    identity = {
        "logical_process": baseline["topology-replay"].logical_process,
        "host_identity_sha256": _sha256(baseline["topology-replay"].host_identity),
        "baseline_source_revision": baseline["topology-replay"].source_revision,
        "candidate_source_revision": candidate["topology-replay"].source_revision,
        "baseline_build_identity_sha256": _sha256(
            baseline["topology-replay"].build_identity
        ),
        "candidate_build_identity_sha256": _sha256(
            candidate["topology-replay"].build_identity
        ),
        "workloads": {
            layout: {
                mode: _sha256(baseline[layout].workload_identities[mode])
                for mode in CAPTURE_MODES
            }
            for layout in LAYOUTS
        },
    }
    eager_runtime_passes = (
        all(cell["passes"] is True for cell in eager_runtime_cells)
        if eager_available
        else None
    )
    eager_generation_passes = (
        eager_generation_geomean_passes is True
        and all(cell["passes"] is True for cell in eager_generation_cells)
        if eager_available
        else None
    )
    eager_resource_passes = (
        all(resource["passes"] is True for resource in eager_resources)
        if eager_available
        else None
    )
    if not eager_available:
        eager_failures.append("eager campaign is not admissible")
    failures.extend(eager_failures)
    comparison_result: dict[str, object] = {
        "kind": COMPARISON_KIND,
        "schema_version": COMPARISON_SCHEMA,
        "complete": True,
        "passes": not failures,
        "identity": identity,
        "identity_sha256": _sha256(identity),
        "inputs": {
            role: {layout: dict(captures[layout].input_identity) for layout in LAYOUTS}
            for role, captures in (
                ("baseline", baseline),
                ("candidate", candidate),
            )
        },
        "thresholds": {
            "runtime_candidate_median_relative_to_baseline": RUNTIME_RELATIVE_LIMIT,
            "runtime_baseline_raw_mad_multiplier": RUNTIME_NOISE_MAD_MULTIPLIER,
            "compiled_to_recurrence_ratio_median_plus_3_raw_mad": (
                COMPILED_RECURRENCE_RATIO_LIMIT
            ),
            "generation_cell_candidate_to_baseline": GENERATION_CELL_LIMIT,
            "generation_geometric_mean_candidate_to_baseline": (
                GENERATION_GEOMEAN_LIMIT
            ),
            "resource_growth_candidate_to_baseline": RESOURCE_GROWTH_LIMIT,
            "resource_exception_minimum_runtime_gain": RESOURCE_EXCEPTION_GAIN,
        },
        "runtime_cells": runtime_cells,
        "paired_compiled_recurrence_ratios": paired_ratios,
        "generation": {
            "cells": generation_cells,
            "candidate_to_baseline_ratio_geometric_mean": generation_geomean,
            "geometric_mean_limit": GENERATION_GEOMEAN_LIMIT,
            "geometric_mean_passes": generation_geomean_passes,
            "passes": (
                generation_geomean_passes
                and all(cell["passes"] is True for cell in generation_cells)
            ),
        },
        "payload_cold_load_rss": resources,
        "eager_diagnostic": {
            "requested": eager_requested,
            "admissible": eager_available,
            "inadmissibility_reasons": eager_inadmissibility_reasons,
            "runtime_cells": eager_runtime_cells,
            "runtime_passes": eager_runtime_passes,
            "generation": {
                "cells": eager_generation_cells,
                "candidate_to_baseline_ratio_geometric_mean": (
                    eager_generation_geomean
                ),
                "geometric_mean_limit": GENERATION_GEOMEAN_LIMIT,
                "geometric_mean_passes": eager_generation_geomean_passes,
                "passes": eager_generation_passes,
            },
            "payload_cold_load_rss": eager_resources,
            "resource_passes": eager_resource_passes,
            "paired_compiled_recurrence_ratio_applicable": False,
            "passes": (
                eager_runtime_passes
                and eager_generation_passes
                and eager_resource_passes
                if eager_available
                else None
            ),
            "failures": eager_failures,
        },
        "failures": failures,
    }
    comparison_result["content_sha256"] = _sha256(comparison_result)
    return comparison_result


def _validate_campaign_capture_identity(
    value: object,
    *,
    evidence: CaptureEvidence,
    label: str,
) -> None:
    identity = _mapping(value, label)
    if set(identity) != {"path", "resolved_path", "size_bytes", "sha256"}:
        raise EvidenceError(f"{label} has an unsupported capture identity")
    resolved_path = _string(identity.get("resolved_path"), f"{label}.resolved_path")
    if (
        _string(identity.get("path"), f"{label}.path")
        != evidence.input_identity.get("path")
        or resolved_path != evidence.input_identity.get("resolved_path")
        or _exact_int(
            identity.get("size_bytes"),
            f"{label}.size_bytes",
            minimum=1,
        )
        != evidence.input_identity.get("size_bytes")
        or _valid_sha256(identity.get("sha256"), f"{label}.sha256")
        != evidence.input_identity.get("sha256")
    ):
        raise EvidenceError(f"{label} does not identify its supplied capture file")


def _validate_paired_campaign(
    payload: Mapping[str, Any],
    input_identity: Mapping[str, object],
    *,
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> Mapping[str, object]:
    label = "paired campaign"
    campaign = _validate_content_address(payload, label)
    _validate_nested_content_addresses(campaign, label)
    expected_root_fields = {
        "kind",
        "schema_version",
        "complete",
        "session_id",
        "started_at_utc",
        "finished_at_utc",
        "process",
        "orchestrator",
        "harness",
        "roles",
        "configuration",
        "layouts",
        "content_sha256",
    }
    if (
        set(campaign) != expected_root_fields
        or campaign.get("kind") != PAIRED_CAMPAIGN_KIND
        or campaign.get("schema_version") != PAIRED_CAMPAIGN_SCHEMA
        or campaign.get("complete") is not True
    ):
        raise EvidenceError(f"{label} has an unsupported identity")
    session_id = _string(campaign.get("session_id"), f"{label}.session_id")
    started = _utc_timestamp(
        campaign.get("started_at_utc"),
        f"{label}.started_at_utc",
    )
    finished = _utc_timestamp(
        campaign.get("finished_at_utc"),
        f"{label}.finished_at_utc",
    )
    if started > finished:
        raise EvidenceError(f"{label} timestamps are inverted")
    logical_process = _logical_process(campaign.get("process"), f"{label}.process")
    if logical_process != baseline["topology-replay"].logical_process:
        raise EvidenceError(f"{label} process differs from its supplied captures")

    harness = _mapping(campaign.get("harness"), f"{label}.harness")
    expected_harness_fields = {
        "kind",
        "schema_version",
        "candidate_relative_path",
        "head_blob_sha256",
        "working_file_sha256",
        "head_blob_equals_working_file",
        "content_sha256",
    }
    harness_path = (
        Path(__file__).resolve().parents[2] / PAIRED_HARNESS_RELATIVE_PATH
    )
    harness_sha256 = _sha256_file(harness_path)
    if (
        set(harness) != expected_harness_fields
        or harness.get("kind") != PAIRED_HARNESS_KIND
        or harness.get("schema_version") != PAIRED_HARNESS_SCHEMA
        or harness.get("candidate_relative_path") != PAIRED_HARNESS_RELATIVE_PATH
        or harness.get("head_blob_equals_working_file") is not True
        or _valid_sha256(
            harness.get("head_blob_sha256"),
            f"{label}.harness.head_blob_sha256",
        )
        != harness_sha256
        or _valid_sha256(
            harness.get("working_file_sha256"),
            f"{label}.harness.working_file_sha256",
        )
        != harness_sha256
    ):
        raise EvidenceError(
            f"{label} harness is not the exact candidate benchmark driver"
        )
    orchestrator = _mapping(
        campaign.get("orchestrator"),
        f"{label}.orchestrator",
    )
    orchestrator_path = (
        Path(__file__).resolve().parents[2] / PAIRED_DRIVER_RELATIVE_PATH
    )
    orchestrator_sha256 = _sha256_file(orchestrator_path)
    if (
        set(orchestrator) != expected_harness_fields
        or orchestrator.get("kind") != PAIRED_HARNESS_KIND
        or orchestrator.get("schema_version") != PAIRED_HARNESS_SCHEMA
        or orchestrator.get("candidate_relative_path")
        != PAIRED_DRIVER_RELATIVE_PATH
        or orchestrator.get("head_blob_equals_working_file") is not True
        or _valid_sha256(
            orchestrator.get("head_blob_sha256"),
            f"{label}.orchestrator.head_blob_sha256",
        )
        != orchestrator_sha256
        or _valid_sha256(
            orchestrator.get("working_file_sha256"),
            f"{label}.orchestrator.working_file_sha256",
        )
        != orchestrator_sha256
    ):
        raise EvidenceError(
            f"{label} orchestrator is not the exact paired benchmark driver"
        )

    captures_by_role = {"baseline": baseline, "candidate": candidate}
    roles = _mapping(campaign.get("roles"), f"{label}.roles")
    if set(roles) != set(captures_by_role):
        raise EvidenceError(f"{label} role inventory is incomplete")
    expected_role_fields = {
        "source_root",
        "source_revision",
        "python",
        "python_resolved_target",
        "python_sha256",
        "prepared_model",
        "prepared_model_sha256",
    }
    for role, captures in captures_by_role.items():
        role_record = _mapping(roles.get(role), f"{label}.roles.{role}")
        reference = captures["topology-replay"]
        if any(
            capture.paired_profile_coordination.get("session_id") != session_id
            for capture in captures.values()
        ):
            raise EvidenceError(
                f"{label}.session_id differs from its {role} captures"
            )
        if set(role_record) != expected_role_fields:
            raise EvidenceError(f"{label}.roles.{role} has an unsupported identity")
        for field in (
            "source_root",
            "python",
            "python_resolved_target",
            "prepared_model",
        ):
            _string(role_record.get(field), f"{label}.roles.{role}.{field}")
        if (
            role_record.get("source_revision") != reference.source_revision
            or role_record.get("python") != reference.interpreter_path
            or role_record.get("python_resolved_target")
            != reference.interpreter_resolved_path
            or _valid_sha256(
                role_record.get("python_sha256"),
                f"{label}.roles.{role}.python_sha256",
            )
            != reference.campaign_identity["interpreter"]["executable_sha256"]
            or role_record.get("prepared_model") != reference.prepared_model_path
            or _valid_sha256(
                role_record.get("prepared_model_sha256"),
                f"{label}.roles.{role}.prepared_model_sha256",
            )
            != reference.prepared_model_sha256
        ):
            raise EvidenceError(
                f"{label}.roles.{role} differs from its capture provenance"
            )

    reference_campaign = baseline["topology-replay"].campaign_identity
    expected_configuration = {
        "authoritative_modes": list(AUTHORITATIVE_MODES),
        "diagnostic_modes": list(DIAGNOSTIC_MODES),
        "batch_sizes": list(BATCH_SIZES),
        "target_runtime_seconds": reference_campaign["target_runtime_seconds"],
        "minimum_samples": reference_campaign["minimum_internal_samples"],
        "subprocess_samples": reference_campaign["subprocess_samples"],
        "warmup_runs": reference_campaign["warmup_runs"],
        "watchdog": {
            "required": True,
            "report_kind": WATCHDOG_REPORT_KIND,
            "report_schema_version": WATCHDOG_REPORT_SCHEMA,
            "limit_bytes": WATCHDOG_LIMIT_BYTES,
            "scope": WATCHDOG_SCOPE,
            "binding": WATCHDOG_BINDING,
        },
    }
    configuration = _mapping(
        campaign.get("configuration"),
        f"{label}.configuration",
    )
    if dict(configuration) != expected_configuration:
        raise EvidenceError(f"{label} configuration differs from its captures")

    layouts = _mapping(campaign.get("layouts"), f"{label}.layouts")
    if set(layouts) != set(LAYOUTS):
        raise EvidenceError(f"{label} layout inventory is incomplete")
    for layout in LAYOUTS:
        layout_record = _mapping(layouts.get(layout), f"{label}.layouts.{layout}")
        if (
            set(layout_record)
            != {
                "layout",
                "ready_records",
                "paired_schedule_algorithm",
                "pairs",
                "captures",
            }
            or layout_record.get("layout") != layout
            or layout_record.get("paired_schedule_algorithm")
            != "slot-adjacent-round-alternating-role-order-v1"
        ):
            raise EvidenceError(f"{label}.layouts.{layout} has an unsupported identity")
        ready_records = _mapping(
            layout_record.get("ready_records"),
            f"{label}.layouts.{layout}.ready_records",
        )
        capture_identities = _mapping(
            layout_record.get("captures"),
            f"{label}.layouts.{layout}.captures",
        )
        if set(ready_records) != set(captures_by_role) or set(
            capture_identities
        ) != set(captures_by_role):
            raise EvidenceError(
                f"{label}.layouts.{layout} role inventory is incomplete"
            )
        for role, captures in captures_by_role.items():
            evidence = captures[layout]
            if (
                ready_records.get(role)
                != evidence.paired_profile_coordination["ready_record"]
            ):
                raise EvidenceError(
                    f"{label}.layouts.{layout}.{role} ready record differs "
                    "from its capture"
                )
            _validate_campaign_capture_identity(
                capture_identities.get(role),
                evidence=evidence,
                label=f"{label}.layouts.{layout}.captures.{role}",
            )
            ready_at = evidence.paired_profile_coordination.get("ready_at")
            if not isinstance(ready_at, dt.datetime) or not started <= ready_at:
                raise EvidenceError(
                    f"{label}.layouts.{layout}.{role} became ready before "
                    "the campaign started"
                )

        raw_pairs = _sequence(
            layout_record.get("pairs"),
            f"{label}.layouts.{layout}.pairs",
        )
        baseline_slots = _mapping(
            baseline[layout].paired_profile_coordination.get("slots"),
            f"{label}.layouts.{layout}.baseline slots",
        )
        candidate_slots = _mapping(
            candidate[layout].paired_profile_coordination.get("slots"),
            f"{label}.layouts.{layout}.candidate slots",
        )
        if len(raw_pairs) != len(baseline_slots):
            raise EvidenceError(
                f"{label}.layouts.{layout} pair inventory is incomplete"
            )
        for pair_index, raw_pair in enumerate(raw_pairs):
            pair = _mapping(
                raw_pair,
                f"{label}.layouts.{layout}.pairs[{pair_index}]",
            )
            baseline_slot = _mapping(
                baseline_slots[pair_index],
                f"{label}.layouts.{layout}.baseline slot {pair_index}",
            )
            candidate_slot = _mapping(
                candidate_slots[pair_index],
                f"{label}.layouts.{layout}.candidate slot {pair_index}",
            )
            token = _mapping(
                baseline_slot.get("token"),
                f"{label}.layouts.{layout}.baseline token {pair_index}",
            )
            round_index = _exact_int(
                token.get("round"),
                f"{label}.layouts.{layout}.pairs[{pair_index}].round",
                minimum=0,
            )
            expected_order = (
                ["baseline", "candidate"]
                if round_index % 2 == 0
                else ["candidate", "baseline"]
            )
            expected_coordinates = {
                field: token.get(field)
                for field in (
                    "schedule_index",
                    "round",
                    "mode",
                    "batch_size",
                )
            }
            completions = _mapping(
                pair.get("completions"),
                f"{label}.layouts.{layout}.pairs[{pair_index}].completions",
            )
            if (
                pair.get("kind") != "pyamplicol-paired-profile-pair"
                or pair.get("schema_version") != 1
                or pair.get("pair_index") != pair_index
                or any(
                    pair.get(field) != value
                    for field, value in expected_coordinates.items()
                )
                or pair.get("role_order") != expected_order
                or dict(completions)
                != {
                    "baseline": baseline_slot["completion"],
                    "candidate": candidate_slot["completion"],
                }
            ):
                raise EvidenceError(
                    f"{label}.layouts.{layout}.pairs[{pair_index}] differs "
                    "from its captures"
                )
            for role_slot in (baseline_slot, candidate_slot):
                recorded = role_slot.get("recorded")
                if not isinstance(recorded, dt.datetime) or recorded > finished:
                    raise EvidenceError(
                        f"{label}.layouts.{layout} completed after the campaign"
                    )

    result = dict(input_identity)
    result["campaign_content_sha256"] = _valid_sha256(
        campaign.get("content_sha256"),
        f"{label}.content_sha256",
    )
    result["harness_content_sha256"] = _valid_sha256(
        harness.get("content_sha256"),
        f"{label}.harness.content_sha256",
    )
    result["orchestrator_content_sha256"] = _valid_sha256(
        orchestrator.get("content_sha256"),
        f"{label}.orchestrator.content_sha256",
    )
    return result


def _validate_watchdog_report(
    payload: Mapping[str, Any],
    input_identity: Mapping[str, object],
    *,
    campaign: Mapping[str, Any],
    campaign_input_identity: Mapping[str, object],
) -> Mapping[str, object]:
    label = "outer memory watchdog report"
    report = _validate_content_address(payload, label)
    _validate_nested_content_addresses(report, label)
    expected_fields = {
        "kind",
        "schema_version",
        "complete",
        "passes",
        "watchdog",
        "working_directory",
        "execution",
        "enforcement",
        "result_binding",
        "content_sha256",
    }
    if (
        set(report) != expected_fields
        or report.get("kind") != WATCHDOG_REPORT_KIND
        or report.get("schema_version") != WATCHDOG_REPORT_SCHEMA
        or report.get("complete") is not True
        or report.get("passes") is not True
    ):
        raise EvidenceError(f"{label} is not complete and passing")

    watchdog_identity = _mapping(
        report.get("watchdog"),
        f"{label}.watchdog",
    )
    watchdog_path = Path(__file__).resolve().parents[2] / WATCHDOG_RELATIVE_PATH
    if (
        set(watchdog_identity) != {"path", "resolved_path", "size_bytes", "sha256"}
        or watchdog_identity.get("resolved_path") != str(watchdog_path.resolve())
        or _exact_int(
            watchdog_identity.get("size_bytes"),
            f"{label}.watchdog.size_bytes",
            minimum=1,
        )
        != watchdog_path.stat().st_size
        or _valid_sha256(
            watchdog_identity.get("sha256"),
            f"{label}.watchdog.sha256",
        )
        != _sha256_file(watchdog_path)
    ):
        raise EvidenceError(
            f"{label} was not emitted by the authenticated watchdog tool"
        )

    execution = _mapping(report.get("execution"), f"{label}.execution")
    if set(execution) != {
        "command",
        "command_sha256",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_wall_seconds",
        "child_pid",
        "child_exit_code",
        "watchdog_exit_code",
        "outcome",
        "reason",
    }:
        raise EvidenceError(f"{label} execution has an unsupported schema")
    command = list(_sequence(execution.get("command"), f"{label}.execution.command"))
    if (
        len(command) < 2
        or any(not isinstance(value, str) or not value for value in command)
        or _valid_sha256(
            execution.get("command_sha256"),
            f"{label}.execution.command_sha256",
        )
        != _sha256(command)
        or execution.get("child_exit_code") != 0
        or execution.get("watchdog_exit_code") != 0
        or execution.get("outcome") != "command-finished"
        or execution.get("reason") is not None
        or _exact_int(
            execution.get("child_pid"),
            f"{label}.execution.child_pid",
            minimum=1,
        )
        < 1
        or _positive_number(
            execution.get("elapsed_wall_seconds"),
            f"{label}.execution.elapsed_wall_seconds",
        )
        <= 0.0
    ):
        raise EvidenceError(f"{label} did not record a successful child execution")
    watchdog_started = _utc_timestamp(
        execution.get("started_at_utc"),
        f"{label}.execution.started_at_utc",
    )
    watchdog_finished = _utc_timestamp(
        execution.get("finished_at_utc"),
        f"{label}.execution.finished_at_utc",
    )
    campaign_started = _utc_timestamp(
        campaign.get("started_at_utc"),
        "paired campaign.started_at_utc",
    )
    campaign_finished = _utc_timestamp(
        campaign.get("finished_at_utc"),
        "paired campaign.finished_at_utc",
    )
    if not watchdog_started <= campaign_started <= campaign_finished <= watchdog_finished:
        raise EvidenceError(
            f"{label} timestamps do not enclose the paired campaign"
        )

    enforcement = _mapping(report.get("enforcement"), f"{label}.enforcement")
    if (
        set(enforcement)
        != {
            "scope",
            "limit_bytes",
            "poll_interval_seconds",
            "terminate_grace_seconds",
            "metric",
            "probe_sample_count",
            "probe_failure_count",
            "maximum_consecutive_probe_failures",
            "completed_under_retry_policy",
            "peak_rss_bytes",
            "peak_physical_footprint_bytes",
            "peak_guard_bytes",
            "peak_processes",
        }
        or enforcement.get("scope") != WATCHDOG_SCOPE
        or enforcement.get("limit_bytes") != WATCHDOG_LIMIT_BYTES
        or enforcement.get("metric")
        not in {
            "process-tree-rss",
            "max(process-tree-rss,darwin-process-tree-physical-footprint)",
        }
        or enforcement.get("completed_under_retry_policy") is not True
        or _exact_int(
            enforcement.get("probe_sample_count"),
            f"{label}.enforcement.probe_sample_count",
            minimum=1,
        )
        < 1
        or _positive_number(
            enforcement.get("poll_interval_seconds"),
            f"{label}.enforcement.poll_interval_seconds",
        )
        <= 0.0
        or _nonnegative_number(
            enforcement.get("terminate_grace_seconds"),
            f"{label}.enforcement.terminate_grace_seconds",
        )
        < 0.0
    ):
        raise EvidenceError(
            f"{label} does not prove successful 30-GiB enforcement"
        )
    probe_failure_count = _exact_int(
        enforcement.get("probe_failure_count"),
        f"{label}.enforcement.probe_failure_count",
        minimum=0,
    )
    maximum_consecutive_probe_failures = _exact_int(
        enforcement.get("maximum_consecutive_probe_failures"),
        f"{label}.enforcement.maximum_consecutive_probe_failures",
        minimum=0,
    )
    if (
        maximum_consecutive_probe_failures > 2
        or maximum_consecutive_probe_failures > probe_failure_count
        or (probe_failure_count == 0) != (maximum_consecutive_probe_failures == 0)
    ):
        raise EvidenceError(
            f"{label} exceeds the watchdog's recovered-probe retry policy"
        )
    peak_rss = _exact_int(
        enforcement.get("peak_rss_bytes"),
        f"{label}.enforcement.peak_rss_bytes",
        minimum=0,
    )
    peak_guard = _exact_int(
        enforcement.get("peak_guard_bytes"),
        f"{label}.enforcement.peak_guard_bytes",
        minimum=0,
    )
    _exact_int(
        enforcement.get("peak_processes"),
        f"{label}.enforcement.peak_processes",
        minimum=1,
    )
    physical_peak = enforcement.get("peak_physical_footprint_bytes")
    if physical_peak is not None:
        physical_peak = _exact_int(
            physical_peak,
            f"{label}.enforcement.peak_physical_footprint_bytes",
            minimum=0,
        )
    metric = enforcement.get("metric")
    if (
        peak_guard > WATCHDOG_LIMIT_BYTES
        or (
            metric == "process-tree-rss"
            and (physical_peak is not None or peak_guard != peak_rss)
        )
        or (
            metric
            == "max(process-tree-rss,darwin-process-tree-physical-footprint)"
            and (
                physical_peak is None
                or peak_guard != max(peak_rss, physical_peak)
            )
        )
    ):
        raise EvidenceError(f"{label} peak observations are internally inconsistent")

    working_directory = Path(
        _string(report.get("working_directory"), f"{label}.working_directory")
    )
    if not working_directory.is_absolute():
        raise EvidenceError(f"{label} working directory is not absolute")

    def option(name: str) -> str:
        indices = [index for index, value in enumerate(command) if value == name]
        if (
            len(indices) != 1
            or indices[0] + 1 >= len(command)
            or not isinstance(command[indices[0] + 1], str)
            or not command[indices[0] + 1]
        ):
            raise EvidenceError(f"{label} command has no unique {name}")
        return command[indices[0] + 1]

    roles = _mapping(campaign.get("roles"), "paired campaign.roles")
    candidate_role = _mapping(
        roles.get("candidate"),
        "paired campaign.roles.candidate",
    )
    if command[0] != candidate_role.get("python"):
        raise EvidenceError(
            f"{label} command did not use the candidate campaign interpreter"
        )
    command_driver = Path(command[1]).expanduser()
    if not command_driver.is_absolute():
        command_driver = working_directory / command_driver
    if command_driver.resolve() != (
        Path(__file__).resolve().parents[2] / PAIRED_DRIVER_RELATIVE_PATH
    ).resolve():
        raise EvidenceError(
            f"{label} command did not execute the paired benchmark driver"
        )
    if option("--session-id") != campaign.get("session_id"):
        raise EvidenceError(f"{label} command session differs from the campaign")
    result_argument = Path(option("--result-json")).expanduser()
    if not result_argument.is_absolute():
        result_argument = working_directory / result_argument
    result_resolved = str(result_argument.resolve())
    if result_resolved != campaign_input_identity.get("resolved_path"):
        raise EvidenceError(
            f"{label} command result path differs from the supplied campaign"
        )

    binding = _mapping(report.get("result_binding"), f"{label}.result_binding")
    if set(binding) != {"requested", "requested_path", "identity", "error"}:
        raise EvidenceError(f"{label} result binding has an unsupported schema")
    identity = _mapping(
        binding.get("identity"),
        f"{label}.result_binding.identity",
    )
    if (
        binding.get("requested") is not True
        or binding.get("error") is not None
        or set(identity) != {"path", "resolved_path", "size_bytes", "sha256"}
        or identity.get("path") != binding.get("requested_path")
        or str(Path(_string(identity.get("path"), f"{label}.result path")).expanduser())
        != str(Path(option("--result-json")).expanduser())
        or identity.get("resolved_path") != result_resolved
        or identity.get("resolved_path")
        != campaign_input_identity.get("resolved_path")
        or identity.get("size_bytes") != campaign_input_identity.get("size_bytes")
        or identity.get("sha256") != campaign_input_identity.get("sha256")
    ):
        raise EvidenceError(
            f"{label} does not content-address the supplied paired campaign"
        )

    result = dict(input_identity)
    result["report_content_sha256"] = _valid_sha256(
        report.get("content_sha256"),
        f"{label}.content_sha256",
    )
    result["command_sha256"] = _valid_sha256(
        execution.get("command_sha256"),
        f"{label}.execution.command_sha256",
    )
    result["bound_campaign_sha256"] = _valid_sha256(
        identity.get("sha256"),
        f"{label}.result_binding.identity.sha256",
    )
    result["watchdog_sha256"] = _valid_sha256(
        watchdog_identity.get("sha256"),
        f"{label}.watchdog.sha256",
    )
    return result


def compare_capture_files(
    *,
    campaign: Path,
    watchdog_report: Path,
    baseline_topology: Path,
    baseline_all_flow_union: Path,
    candidate_topology: Path,
    candidate_all_flow_union: Path,
    expected_candidate_source_revision: str,
    expected_baseline_native_build_inputs_sha256: str,
    expected_candidate_native_build_inputs_sha256: str,
    expected_baseline_prepared_model_sha256: str,
    expected_candidate_prepared_model_sha256: str,
) -> dict[str, object]:
    """Load, validate, and compare the four required capture files."""

    campaign_payload, campaign_input_identity = _read_json_mapping(
        campaign,
        label="paired campaign",
    )
    watchdog_payload, watchdog_input_identity = _read_json_mapping(
        watchdog_report,
        label="outer memory watchdog report",
    )
    paths = {
        "baseline": {
            "topology-replay": baseline_topology,
            "all-flow-union": baseline_all_flow_union,
        },
        "candidate": {
            "topology-replay": candidate_topology,
            "all-flow-union": candidate_all_flow_union,
        },
    }
    evidence: dict[str, dict[str, CaptureEvidence]] = {
        "baseline": {},
        "candidate": {},
    }
    for role in ("baseline", "candidate"):
        for layout in LAYOUTS:
            payload, input_identity = _read_capture(paths[role][layout])
            evidence[role][layout] = _validate_capture(
                payload,
                input_identity,
                role=role,
                expected_layout=layout,
            )
    expected_role_inputs = _assert_externally_pinned_role_inputs(
        evidence["baseline"],
        evidence["candidate"],
        expected_candidate_source_revision=expected_candidate_source_revision,
        expected_baseline_native_build_inputs_sha256=(
            expected_baseline_native_build_inputs_sha256
        ),
        expected_candidate_native_build_inputs_sha256=(
            expected_candidate_native_build_inputs_sha256
        ),
        expected_baseline_prepared_model_sha256=(
            expected_baseline_prepared_model_sha256
        ),
        expected_candidate_prepared_model_sha256=(
            expected_candidate_prepared_model_sha256
        ),
    )
    comparison = compare_captures(
        baseline=evidence["baseline"],
        candidate=evidence["candidate"],
    )
    campaign_identity = _validate_paired_campaign(
        campaign_payload,
        campaign_input_identity,
        baseline=evidence["baseline"],
        candidate=evidence["candidate"],
    )
    watchdog_identity = _validate_watchdog_report(
        watchdog_payload,
        watchdog_input_identity,
        campaign=campaign_payload,
        campaign_input_identity=campaign_input_identity,
    )
    unsigned = dict(comparison)
    unsigned.pop("content_sha256")
    inputs = dict(_mapping(unsigned.get("inputs"), "comparison.inputs"))
    inputs["externally_pinned_role_inputs"] = expected_role_inputs
    inputs["paired_campaign"] = dict(campaign_identity)
    inputs["outer_memory_watchdog"] = dict(watchdog_identity)
    unsigned["inputs"] = inputs
    unsigned["content_sha256"] = _sha256(unsigned)
    return unsigned


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_json_bytes(value) + b"\n"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise EvidenceError(f"cannot write comparison JSON: {path}") from error
    finally:
        temporary_path = locals().get("temporary")
        if isinstance(temporary_path, Path) and temporary_path.exists():
            temporary_path.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign", type=Path, required=True)
    result.add_argument("--watchdog-report", type=Path, required=True)
    result.add_argument("--baseline-topology", type=Path, required=True)
    result.add_argument("--baseline-all-flow-union", type=Path, required=True)
    result.add_argument("--candidate-topology", type=Path, required=True)
    result.add_argument("--candidate-all-flow-union", type=Path, required=True)
    result.add_argument("--expected-candidate-source-revision", required=True)
    result.add_argument(
        "--expected-baseline-native-build-inputs-sha256",
        required=True,
    )
    result.add_argument(
        "--expected-candidate-native-build-inputs-sha256",
        required=True,
    )
    result.add_argument(
        "--expected-baseline-prepared-model-sha256",
        required=True,
    )
    result.add_argument(
        "--expected-candidate-prepared-model-sha256",
        required=True,
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        comparison = compare_capture_files(
            campaign=arguments.campaign,
            watchdog_report=arguments.watchdog_report,
            baseline_topology=arguments.baseline_topology,
            baseline_all_flow_union=arguments.baseline_all_flow_union,
            candidate_topology=arguments.candidate_topology,
            candidate_all_flow_union=arguments.candidate_all_flow_union,
            expected_candidate_source_revision=(
                arguments.expected_candidate_source_revision
            ),
            expected_baseline_native_build_inputs_sha256=(
                arguments.expected_baseline_native_build_inputs_sha256
            ),
            expected_candidate_native_build_inputs_sha256=(
                arguments.expected_candidate_native_build_inputs_sha256
            ),
            expected_baseline_prepared_model_sha256=(
                arguments.expected_baseline_prepared_model_sha256
            ),
            expected_candidate_prepared_model_sha256=(
                arguments.expected_candidate_prepared_model_sha256
            ),
        )
        _write_json_atomic(arguments.output, comparison)
    except EvidenceError as error:
        print(f"benchmark evidence error: {error}", file=sys.stderr)
        return 2
    return 0 if comparison["passes"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
