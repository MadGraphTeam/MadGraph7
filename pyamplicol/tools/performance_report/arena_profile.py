# SPDX-License-Identifier: 0BSD
"""Fail-closed evidence for the private warmed-Arena profiling boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

ARENA_PROFILE_EVIDENCE_ABI = "pyamplicol-report-arena-profile-evidence-v1"
ARENA_PROFILE_PROTOCOL = "arena"
ARENA_PROFILE_SAMPLE_PASS = "runtime._profile_arena_repeated"
ARENA_PROFILE_BOUNDARY = (
    "warmed-direct-arena-borrowed-input-preallocated-output-v1"
)
ARENA_PHASE_TIMING_SCOPE = "coarse-arena-boundary-only-v1"
PAIRED_TIMING_SAMPLE_CONTRACT = (
    "paired_unprofiled_headline_profiled_attribution_v1"
)

ZERO_ARENA_COUNTER_FIELDS = (
    "native_input_container_allocation_count",
    "native_input_pack_bytes",
    "native_input_crossing_bytes",
    "stage_input_copy_component_count",
    "stage_leaf_input_copy_component_count",
    "stage_evaluator_output_gather_component_count",
    "stage_output_assign_component_count",
    "amplitude_input_copy_component_count",
    "amplitude_leaf_input_copy_component_count",
    "amplitude_evaluator_output_gather_component_count",
    "amplitude_output_remap_component_count",
    "selector_gather_point_count",
    "selector_gather_bytes",
    "selector_scatter_value_count",
    "observed_scratch_reallocation_count",
    "native_output_allocation_count",
)
ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS = (
    "compiled_direct_arena_boundary_input_bytes",
    "compiled_direct_arena_boundary_current_output_bytes",
    "compiled_direct_arena_boundary_amplitude_output_bytes",
)
COMPILED_ACTIVITY_COUNTER_FIELDS = (
    "compiled_direct_arena_engine_count",
    "compiled_direct_arena_call_count",
    "evaluator_backend_call_count",
)
ZERO_ARENA_PHASE_TIME_FIELDS = (
    "native_input_pack_time_s",
    "native_input_crossing_time_s",
    "state_prepare_time_s",
    "state_clear_time_s",
    "source_fill_time_s",
    "momentum_input_setup_time_s",
    "momentum_setup_time_s",
    "model_parameter_setup_time_s",
    "stage_input_pack_time_s",
    "stage_leaf_input_pack_time_s",
    "stage_evaluator_call_time_s",
    "stage_evaluator_time_s",
    "stage_backend_call_time_s",
    "stage_evaluator_output_gather_time_s",
    "output_assign_time_s",
    "amplitude_input_pack_time_s",
    "amplitude_leaf_input_pack_time_s",
    "amplitude_evaluator_call_time_s",
    "amplitude_backend_call_time_s",
    "amplitude_evaluator_output_gather_time_s",
    "amplitude_output_remap_time_s",
    "amplitude_evaluator_time_s",
    "reduction_time_s",
    "resolved_reduction_materialization_inclusive_time_s",
    "total_materialization_time_s",
    "final_output_copy_time_s",
    "eager_initialize_time_s",
    "eager_gather_time_s",
    "eager_kernel_call_time_s",
    "eager_invocation_scatter_time_s",
    "eager_finalization_time_s",
    "eager_scatter_finalization_time_s",
    "eager_closure_time_s",
    "eager_reduction_time_s",
    "eager_copy_out_time_s",
    "recurrence_momentum_fill_time_s",
    "recurrence_union_source_fill_time_s",
    "recurrence_schedule_time_s",
    "recurrence_source_kernel_time_s",
    "recurrence_contribution_kernel_time_s",
    "recurrence_finalization_time_s",
    "recurrence_closure_time_s",
    "recurrence_replay_output_mapping_time_s",
    "selector_planner_time_s",
    "selector_gather_time_s",
    "selector_scatter_time_s",
)
EMPTY_ARENA_PHASE_VECTOR_FIELDS = (
    "stage_input_pack_by_stage_time_s",
    "stage_leaf_input_pack_by_stage_time_s",
    "stage_evaluator_call_by_stage_time_s",
    "stage_backend_call_by_stage_time_s",
    "stage_evaluator_output_gather_by_stage_time_s",
    "stage_output_assign_by_stage_time_s",
)
ARENA_PROFILE_EVIDENCE_FIELDS = frozenset(
    {
        "abi",
        "execution_mode",
        "profile_count",
        "repetitions_per_profile",
        "batch_size",
        "evaluated_points_per_profile",
        "total_evaluated_points",
        "warmed_boundary_wall_seconds_per_point",
        "raw_profiles",
        "raw_profiles_sha256",
        "counter_totals",
        "counter_totals_sha256",
    }
)


class ArenaProfileEvidenceError(ValueError):
    """Raised when warmed-Arena profile evidence is incomplete or inconsistent."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ArenaProfileEvidenceError(
            "Arena profile evidence is not canonical JSON"
        ) from error


def digest_arena_profile_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json(value).decode("ascii"))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArenaProfileEvidenceError(f"{name} must be a positive integer")
    return value


def _integer(profile: Mapping[str, object], field: str) -> int:
    value = profile.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaProfileEvidenceError(
            f"Arena profile field {field!r} must be an integer"
        )
    return value


def _finite(profile: Mapping[str, object], field: str) -> float:
    value = profile.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ArenaProfileEvidenceError(
            f"Arena profile field {field!r} must be finite"
        )
    return float(value)


def _counter_fields(
    profiles: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return every scalar native count/traffic field exposed by the samples."""

    inventories = [
        {
            field
            for field, value in profile.items()
            if field == "points"
            or (
                (field.endswith("_count") or field.endswith("_bytes"))
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
        }
        for profile in profiles
    ]
    expected = inventories[0]
    if any(inventory != expected for inventory in inventories[1:]):
        raise ArenaProfileEvidenceError(
            "Arena profile counter inventory differs between raw samples"
        )
    return tuple(sorted(expected))


def _validate_raw_profile(
    profile: Mapping[str, object],
    *,
    execution_mode: str,
    evaluated_points: int,
) -> None:
    if profile.get("execution_mode") != execution_mode:
        raise ArenaProfileEvidenceError(
            "Arena profile execution mode does not match the measured runtime"
        )
    if (
        profile.get("profile_boundary") != ARENA_PROFILE_BOUNDARY
        or profile.get("borrowed_flat_input") is not True
        or profile.get("preallocated_output") is not True
        or profile.get("phase_timing_scope") != ARENA_PHASE_TIMING_SCOPE
        or profile.get("evaluator_timing_available") is not False
    ):
        raise ArenaProfileEvidenceError(
            "Arena profile does not authenticate the warmed borrowed-input, "
            "preallocated-output boundary"
        )
    if _integer(profile, "points") != evaluated_points:
        raise ArenaProfileEvidenceError(
            "Arena profile point count does not match batch repetitions"
        )
    wall = _finite(profile, "wall_time_s")
    orchestration = _finite(profile, "orchestration_time_s")
    if wall <= 0.0 or orchestration != wall:
        raise ArenaProfileEvidenceError(
            "Arena profile boundary wall time must be positive and fully "
            "accounted by orchestration"
        )
    zero_counter_fields = [
        *ZERO_ARENA_COUNTER_FIELDS,
        *ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS,
    ]
    for field in zero_counter_fields:
        if _integer(profile, field) != 0:
            raise ArenaProfileEvidenceError(
                f"Arena profile field {field!r} must be zero"
            )
    for field in ZERO_ARENA_PHASE_TIME_FIELDS:
        if _finite(profile, field) != 0.0:
            raise ArenaProfileEvidenceError(
                f"Arena profile field {field!r} must be zero"
            )
    for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS:
        if profile.get(field) != []:
            raise ArenaProfileEvidenceError(
                f"Arena profile field {field!r} must be an empty list"
            )
    if execution_mode == "compiled":
        engine_count = _integer(profile, "compiled_direct_arena_engine_count")
        call_count = _integer(profile, "compiled_direct_arena_call_count")
        backend_count = _integer(profile, "evaluator_backend_call_count")
        if engine_count <= 0 or call_count <= 0 or backend_count <= 0:
            raise ArenaProfileEvidenceError(
                "compiled Arena profile activity counters must be positive"
            )
        if call_count != backend_count:
            raise ArenaProfileEvidenceError(
                "compiled Arena calls do not cover every evaluator backend call"
            )


def build_arena_profile_evidence(
    raw_profiles: Sequence[Mapping[str, object]],
    *,
    execution_mode: str,
    repetitions_per_profile: int,
    batch_size: int,
) -> dict[str, object]:
    """Validate raw samples and build their independently recomputed evidence."""

    if execution_mode not in {"compiled", "eager"}:
        raise ArenaProfileEvidenceError("Arena profile execution mode is unsupported")
    repetitions = _positive_integer(
        repetitions_per_profile,
        "repetitions_per_profile",
    )
    points_per_batch = _positive_integer(batch_size, "batch_size")
    if (
        isinstance(raw_profiles, (str, bytes, bytearray))
        or not isinstance(raw_profiles, Sequence)
        or not raw_profiles
    ):
        raise ArenaProfileEvidenceError("Arena profile evidence requires raw samples")
    evaluated_points = repetitions * points_per_batch
    canonical_profiles: list[dict[str, object]] = []
    wall_per_point: list[float] = []
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, Mapping):
            raise ArenaProfileEvidenceError(
                f"Arena profile sample {index} is not an object"
            )
        copied = _json_copy(dict(raw_profile))
        if not isinstance(copied, dict):
            raise ArenaProfileEvidenceError(
                f"Arena profile sample {index} is not an object"
            )
        _validate_raw_profile(
            copied,
            execution_mode=execution_mode,
            evaluated_points=evaluated_points,
        )
        canonical_profiles.append(copied)
        wall_per_point.append(float(copied["wall_time_s"]) / evaluated_points)
    counter_totals = {
        field: 0 for field in _counter_fields(canonical_profiles)
    }
    for copied in canonical_profiles:
        for field in counter_totals:
            counter_totals[field] += _integer(copied, field)
    profile_count = len(canonical_profiles)
    warmed_wall = math.fsum(wall_per_point) / profile_count
    return {
        "abi": ARENA_PROFILE_EVIDENCE_ABI,
        "execution_mode": execution_mode,
        "profile_count": profile_count,
        "repetitions_per_profile": repetitions,
        "batch_size": points_per_batch,
        "evaluated_points_per_profile": evaluated_points,
        "total_evaluated_points": profile_count * evaluated_points,
        "warmed_boundary_wall_seconds_per_point": warmed_wall,
        "raw_profiles": canonical_profiles,
        "raw_profiles_sha256": digest_arena_profile_value(canonical_profiles),
        "counter_totals": counter_totals,
        "counter_totals_sha256": digest_arena_profile_value(counter_totals),
    }


def validate_arena_profile_evidence(
    value: object,
    *,
    execution_mode: str,
    sample_count: int,
    native_profile_points_per_sample: int,
) -> Mapping[str, object]:
    """Recompute and authenticate stored warmed-Arena evidence."""

    if not isinstance(value, Mapping) or set(value) != ARENA_PROFILE_EVIDENCE_FIELDS:
        raise ArenaProfileEvidenceError(
            "Arena profile evidence fields do not match the contract"
        )
    raw_profiles = value.get("raw_profiles")
    if not isinstance(raw_profiles, Sequence) or isinstance(
        raw_profiles,
        (str, bytes, bytearray),
    ):
        raise ArenaProfileEvidenceError("Arena profile raw samples are invalid")
    repetitions = _positive_integer(
        value.get("repetitions_per_profile"),
        "repetitions_per_profile",
    )
    batch_size = _positive_integer(value.get("batch_size"), "batch_size")
    rebuilt = build_arena_profile_evidence(
        raw_profiles,  # type: ignore[arg-type]
        execution_mode=execution_mode,
        repetitions_per_profile=repetitions,
        batch_size=batch_size,
    )
    if dict(value) != rebuilt:
        raise ArenaProfileEvidenceError(
            "Arena profile evidence differs from independently recomputed values"
        )
    if rebuilt["profile_count"] != sample_count:
        raise ArenaProfileEvidenceError(
            "Arena profile count does not match the measurement sample count"
        )
    if (
        rebuilt["evaluated_points_per_profile"]
        != native_profile_points_per_sample
    ):
        raise ArenaProfileEvidenceError(
            "Arena profile point count does not match timing provenance"
        )
    return value


__all__ = [
    "ARENA_PHASE_TIMING_SCOPE",
    "ARENA_PROFILE_BOUNDARY",
    "ARENA_PROFILE_EVIDENCE_ABI",
    "ARENA_PROFILE_EVIDENCE_FIELDS",
    "ARENA_PROFILE_PROTOCOL",
    "ARENA_PROFILE_SAMPLE_PASS",
    "COMPILED_ACTIVITY_COUNTER_FIELDS",
    "EMPTY_ARENA_PHASE_VECTOR_FIELDS",
    "PAIRED_TIMING_SAMPLE_CONTRACT",
    "ZERO_ARENA_COUNTER_FIELDS",
    "ZERO_ARENA_PHASE_TIME_FIELDS",
    "ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS",
    "ArenaProfileEvidenceError",
    "build_arena_profile_evidence",
    "digest_arena_profile_value",
    "validate_arena_profile_evidence",
]
