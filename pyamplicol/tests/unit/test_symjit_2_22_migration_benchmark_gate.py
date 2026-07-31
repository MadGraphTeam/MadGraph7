# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "symjit_2_22_migration_benchmark_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "symjit_2_22_migration_benchmark_gate",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

ALL_MODES = (*gate.CAPTURE_MODES, *gate.DIAGNOSTIC_MODES)
EXPECTED_CANDIDATE_REVISION = "2" * 40
EXPECTED_BASELINE_NATIVE_BUILD_INPUTS = "e" * 64
EXPECTED_CANDIDATE_NATIVE_BUILD_INPUTS = "f" * 64
EXPECTED_BASELINE_PREPARED_MODEL = "8" * 64
EXPECTED_CANDIDATE_PREPARED_MODEL = "9" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _address(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = _sha256(result)
    return result


def _peak(value: int) -> dict[str, object]:
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": value,
        "maximum_child_peak_bytes": 0,
        "observed_lower_bound_bytes": value,
        "semantics": "synthetic process high-water mark",
    }


def _statistics(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in values)
    return median, deviations[len(deviations) // 2]


_TEST_EPOCH = dt.datetime(2026, 7, 30, tzinfo=dt.UTC)


def _layout_epoch(layout: str) -> dt.datetime:
    return _TEST_EPOCH + (
        dt.timedelta(seconds=10_000)
        if layout == "all-flow-union"
        else dt.timedelta()
    )


def _slot_times(
    *,
    role: str,
    layout: str,
    schedule_index: int,
    round_index: int,
) -> tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]:
    expected_order = (
        ("baseline", "candidate")
        if round_index % 2 == 0
        else ("candidate", "baseline")
    )
    order = expected_order.index(role)
    pair_start = _layout_epoch(layout) + dt.timedelta(seconds=schedule_index * 10)
    issued = pair_start + dt.timedelta(seconds=order * 3)
    return (
        issued,
        issued + dt.timedelta(seconds=1),
        issued + dt.timedelta(seconds=2),
        issued + dt.timedelta(seconds=2, milliseconds=500),
    )


def _semantic_identity(
    layout: str,
    mode: str,
    *,
    selected_union: bool = False,
) -> dict[str, object]:
    normalization = {"average_factor": 36, "color_factor": 2187}
    reduction_ordering = {
        "kind": "lc-diagonal",
        "pair_order_abi": "helicity-major-color-minor-v1",
        **({"lane_internal": mode} if selected_union else {}),
    }
    selector_semantics = {
        "kind": "pyamplicol-runtime-selectors",
        "layout": layout,
    }
    common_model = {
        "name": "built-in-sm",
        "content_sha256": "9" * 64,
        "compiled_schema_version": 9,
        "restriction": None,
    }
    if selected_union:
        color_ids = [
            f"flow:synthetic:{index}"
            for index in range(gate.SELECTED_UNION_COLOR_COUNT)
        ]
        color_entries = [
            {
                "index": index,
                "id": identifier,
                "kind": "lc-flow",
                "word": [index + 1],
                "coefficient": 1.0,
            }
            for index, identifier in enumerate(color_ids)
        ]
        helicity_count = gate.SELECTED_UNION_HELICITY_INDEX + 1
        helicity_ids = [
            (
                gate.SELECTED_UNION_HELICITY_ID
                if index == gate.SELECTED_UNION_HELICITY_INDEX
                else f"h:synthetic:{index}"
            )
            for index in range(helicity_count)
        ]
        helicity_entries = [
            {
                "index": index,
                "id": identifier,
                "values": (
                    list(gate.SELECTED_UNION_HELICITY_VALUES)
                    if index == gate.SELECTED_UNION_HELICITY_INDEX
                    else [index]
                ),
                "coefficient": (
                    1.0
                    if index == gate.SELECTED_UNION_HELICITY_INDEX or mode != "compiled"
                    else 0.0
                ),
                "structural_zero": (
                    index != gate.SELECTED_UNION_HELICITY_INDEX and mode == "compiled"
                ),
            }
            for index, identifier in enumerate(helicity_ids)
        ]
        physical_colors = {
            "count": len(color_ids),
            "ordered_ids": color_ids,
            "ordered_ids_sha256": _sha256(color_ids),
            "ordered_entries": color_entries,
            "ordered_entries_sha256": _sha256(color_entries),
        }
        physical_helicities = {
            "count": len(helicity_ids),
            "ordered_ids": helicity_ids,
            "ordered_ids_sha256": _sha256(helicity_ids),
            "ordered_entries": helicity_entries,
            "ordered_entries_sha256": _sha256(helicity_entries),
        }
    else:
        physical_colors = {
            "count": 2,
            "ordered_entries": [{"index": 0}, {"index": 1}],
        }
        physical_helicities = {
            "count": 2,
            "ordered_entries": [{"index": 0}, {"index": 1}],
        }
    return {
        "kind": "pyamplicol-benchmark-artifact-semantic-identity",
        "schema_version": 3,
        "mode_internal_marker": mode,
        "generation_specialized_axes": [],
        "coverage": {
            "color": "complete",
            "helicities": "complete",
            "complete_physical_axes": True,
        },
        "reduction_coverage": {
            "complete": True,
            "errors": [],
            "expected_physical_pair_count": 4,
            "observed_physical_pair_count": 4,
        },
        "normalization": normalization,
        "normalization_sha256": _sha256(normalization),
        "reduction_ordering": reduction_ordering,
        "reduction_ordering_sha256": _sha256(reduction_ordering),
        "runtime_selector_semantics": selector_semantics,
        "runtime_selector_semantics_sha256": _sha256(selector_semantics),
        "manifest_model_identity": {
            "common_physics_identity": common_model,
            "common_physics_identity_sha256": _sha256(common_model),
        },
        "physical_color_flows": physical_colors,
        "physical_helicities": physical_helicities,
    }


def _selected_union_projection(
    semantic: dict[str, object],
    *,
    process_id: str,
    color_flow_request: str,
) -> dict[str, object]:
    colors = semantic["physical_color_flows"]
    helicities = semantic["physical_helicities"]
    assert isinstance(colors, dict)
    assert isinstance(helicities, dict)
    entries = helicities["ordered_entries"]
    assert isinstance(entries, list)
    kinematic_entries = [
        {
            "index": entry["index"],
            "id": entry["id"],
            "values": entry["values"],
            **(
                {
                    "coefficient": entry["coefficient"],
                    "structural_zero": entry["structural_zero"],
                }
                if entry["id"] == gate.SELECTED_UNION_HELICITY_ID
                else {}
            ),
        }
        for entry in entries
    ]
    selected_entry = entries[gate.SELECTED_UNION_HELICITY_INDEX]
    model = semantic["manifest_model_identity"]
    assert isinstance(model, dict)
    return {
        "policy": gate.SELECTED_UNION_SEMANTIC_COMPARISON_POLICY,
        "process_id": process_id,
        "process_expression": gate.SELECTED_UNION_PROCESS,
        "physical_color_flows": {
            "count": colors["count"],
            "ordered_ids_sha256": colors["ordered_ids_sha256"],
            "ordered_entries_sha256": colors["ordered_entries_sha256"],
        },
        "physical_helicities": {
            "count": helicities["count"],
            "ordered_ids_sha256": helicities["ordered_ids_sha256"],
            "kinematic_entries_sha256": _sha256(kinematic_entries),
            "selected_index": gate.SELECTED_UNION_HELICITY_INDEX,
            "selected_entry": selected_entry,
        },
        "normalization_sha256": semantic["normalization_sha256"],
        "model_common_physics_identity_sha256": model["common_physics_identity_sha256"],
        "runtime_selector_semantics_sha256": semantic[
            "runtime_selector_semantics_sha256"
        ],
        "profile_selector": {
            "color_flow_request": color_flow_request,
            "resolved_color_flow_id": None,
            "helicity_request": gate.SELECTED_UNION_HELICITY_ID,
            "resolved_helicity_id": gate.SELECTED_UNION_HELICITY_ID,
            "color_flow_count": gate.SELECTED_UNION_COLOR_COUNT,
            "helicity_count": helicities["count"],
            "workload": "all-flows/runtime-selected-single-helicity",
        },
        "resolved_color_ids_sha256": _sha256(colors["ordered_ids"]),
    }


def _sample(
    *,
    role: str,
    layout: str,
    identifier: int,
    schedule_index: int,
    round_index: int,
    batch_size: int,
    runtime: float,
    cold_load: float,
    cold_rss: int,
    profiled_rss: int,
) -> tuple[dict[str, object], dict[str, object]]:
    _, started, finished, recorded = _slot_times(
        role=role,
        layout=layout,
        schedule_index=schedule_index,
        round_index=round_index,
    )
    invocation = _address(
        {
            "kind": "invocation",
            "identifier": identifier,
            "role": role,
            "layout": layout,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "wall_seconds": 12.0,
        }
    )
    process_record = _address(
        {
            "kind": "worker-process",
            "identifier": identifier,
            "role": role,
            "layout": layout,
            "wall_seconds": 12.0,
        }
    )
    block_count = gate.MINIMUM_INTERNAL_SAMPLES
    repetitions = math.ceil(
        (5.0 * 1.02) / (block_count * runtime * batch_size)
    )
    blocks: list[dict[str, object]] = []
    for block_index in range(block_count):
        native = runtime * repetitions * batch_size
        block = _address(
            {
                "block_index": block_index,
                "started_at_utc": (
                    started + dt.timedelta(microseconds=block_index)
                ).isoformat(),
                "finished_at_utc": (
                    started + dt.timedelta(microseconds=block_index + 1)
                ).isoformat(),
                "caller_elapsed_seconds": native,
                "native_wall_seconds": native,
                "wall_seconds_per_point": runtime,
                "repetitions": repetitions,
                "batch_size": batch_size,
                "evaluation_count": repetitions,
                "evaluated_point_count": repetitions * batch_size,
            }
        )
        blocks.append(block)
    observed_native = sum(float(block["native_wall_seconds"]) for block in blocks)
    raw_native_wall = {
        "kind": gate.RAW_NATIVE_WALL_BLOCK_KIND,
        "schema_version": gate.RAW_NATIVE_WALL_BLOCK_SCHEMA,
        "measurement_contract": gate.RAW_NATIVE_WALL_MEASUREMENT_CONTRACT,
        "source": "runtime._benchmark_f64_wall_time",
        "fixture_points_sha256": "b" * 64,
        "minimum_native_wall_seconds": 5.0,
        "observed_native_wall_seconds": observed_native,
        "observed_caller_elapsed_seconds": observed_native,
        "minimum_duration_satisfied": True,
        "calibration": {
            "kind": "benchmark-runner-wall-rate-calibration",
            "schema_version": 1,
            "benchmark_runner_sample_count": 20,
            "benchmark_runner_repetitions_per_sample": 1,
            "benchmark_runner_total_repetitions": 20,
            "benchmark_runner_wall_seconds_per_point": runtime,
            "requested_minimum_block_count": block_count,
            "preceded_by_benchmark_runner_warmup_runs": 2,
            "duration_headroom_factor": 1.02,
            "scaled_repetitions_per_block": repetitions,
        },
        "block_count": block_count,
        "repetitions_per_block": repetitions,
        "evaluation_count": block_count * repetitions,
        "evaluated_point_count": block_count * repetitions * batch_size,
        "wall_seconds_per_point_median": runtime,
        "wall_seconds_per_point_mad": 0.0,
        "blocks": blocks,
        "blocks_sha256": _sha256(blocks),
    }
    worker_measurement = {
        "batch_size": batch_size,
        "sample_count": block_count,
        "repetitions_per_sample": repetitions,
        "evaluation_count": block_count * repetitions,
        "evaluated_point_count": block_count * repetitions * batch_size,
        "wall_seconds_per_point": runtime,
        "inner_native_wall_blocks": raw_native_wall,
        "benchmark_runner_sample_count": 20,
        "benchmark_runner_repetitions_per_sample": 1,
        "benchmark_runner_evaluation_count": 20,
        "benchmark_runner_evaluated_point_count": 20 * batch_size,
        "benchmark_runner_wall_seconds_per_point": runtime,
    }
    sample: dict[str, object] = {
        "schedule_index": schedule_index,
        "round": round_index,
        "wall_seconds_per_point": runtime,
        "cold_load_seconds": cold_load,
        "peak_rss_after_cold_load": _peak(cold_rss),
        "peak_rss_after_profile": _peak(profiled_rss),
        "internal_sample_count": block_count,
        "repetitions_per_sample": repetitions,
        "evaluation_count": block_count * repetitions,
        "evaluated_point_count": block_count * repetitions * batch_size,
        "inner_native_wall_blocks": raw_native_wall,
        "worker_measurement": worker_measurement,
        "interrupted": False,
        "timing_configuration": {
            "minimum_internal_samples": 7,
            "warmup_runs": 2,
            "target_runtime_seconds": 5.0,
        },
        "timing_sources": {"wall": "runtime_core_repeated_wall_time"},
        "worker_invocation": invocation,
        "worker_process_record": process_record,
    }
    result_record = {
        "kind": "pyamplicol-retained-profile-worker-result",
        "schema_version": 1,
        "recorded_at_utc": recorded.isoformat(),
        "addressed_payload_sha256": _sha256(sample),
        "upstream_worker_result_record_sha256": "a" * 64,
        "worker_process_record_sha256": process_record["content_sha256"],
        "worker_invocation_sha256": invocation["content_sha256"],
    }
    sample["worker_result_record"] = _address(result_record)
    schedule_entry = {
        "schedule_index": schedule_index,
        "round": round_index,
        "worker_result_sha256": result_record["addressed_payload_sha256"],
        "worker_invocation": invocation,
        "worker_result_record": sample["worker_result_record"],
    }
    return sample, schedule_entry


def _aggregate(batch_size: int, samples: list[dict[str, object]]) -> dict[str, object]:
    runtime = [float(sample["wall_seconds_per_point"]) for sample in samples]
    cold = [float(sample["cold_load_seconds"]) for sample in samples]
    cold_rss = [
        float(sample["peak_rss_after_cold_load"]["observed_lower_bound_bytes"])
        for sample in samples
    ]
    profiled_rss = [
        float(sample["peak_rss_after_profile"]["observed_lower_bound_bytes"])
        for sample in samples
    ]
    runtime_median, runtime_mad = _statistics(runtime)
    cold_median, cold_mad = _statistics(cold)
    cold_rss_median, _ = _statistics(cold_rss)
    profiled_rss_median, _ = _statistics(profiled_rss)
    return {
        "batch_size": batch_size,
        "sample_count": len(samples),
        "subprocess_sample_count": len(samples),
        "wall_seconds_per_point": runtime_median,
        "wall_seconds_per_point_median": runtime_median,
        "wall_seconds_per_point_mad": runtime_mad,
        "statistics_contract": "subprocess-median-and-raw-mad-v1",
        "cold_load_seconds_median": cold_median,
        "cold_load_seconds_mad": cold_mad,
        "cold_load_peak_rss_bytes_median": cold_rss_median,
        "profiled_peak_rss_bytes_median": profiled_rss_median,
        "resource_statistics_contract": "subprocess-median-and-raw-mad-v1",
        "subprocess_samples": samples,
        "interrupted": False,
    }


def _capture(
    *,
    role: str,
    layout: str,
    process: str,
    runtime_ratio: float = 1.0,
    compiled_to_recurrence_ratio: float = 1.05,
    generation_ratio: float = 1.0,
    resource_ratio: float = 1.0,
    eager_runtime_ratio: float = 1.0,
    eager_generation_ratio: float = 1.0,
    eager_resource_ratio: float = 1.0,
) -> dict[str, object]:
    candidate = role == "candidate"
    revision = "2" * 40 if candidate else gate.BASELINE_REVISION
    interpreter_path = "/synthetic/shared/bin/python"
    prepared_model_path = f"/synthetic/{role}/prepared-model.pack"
    prepared_model_sha256 = ("9" if candidate else "8") * 64
    runtime_role_ratio = runtime_ratio if candidate else 1.0
    generation_role_ratio = generation_ratio if candidate else 1.0
    resource_role_ratio = resource_ratio if candidate else 1.0
    eager_runtime_role_ratio = eager_runtime_ratio if candidate else 1.0
    eager_generation_role_ratio = eager_generation_ratio if candidate else 1.0
    eager_resource_role_ratio = eager_resource_ratio if candidate else 1.0
    workload = (
        "single-runtime-selected-flow/helicity-sum"
        if layout == "topology-replay"
        else "all-flows/runtime-selected-single-helicity"
    )
    selected_union = process.casefold().startswith("d ") and layout == "all-flow-union"
    selected_union_helicity_count = gate.SELECTED_UNION_HELICITY_INDEX + 1
    selector = {
        "color_flow_count": (gate.SELECTED_UNION_COLOR_COUNT if selected_union else 2),
        "helicity_count": selected_union_helicity_count if selected_union else 2,
        "workload": workload,
        "color_flow_request": "flow:2,1" if layout == "topology-replay" else "1",
        "helicity_request": (
            "1"
            if layout == "topology-replay"
            else gate.SELECTED_UNION_HELICITY_ID
            if selected_union
            else "h:-1,+1"
        ),
        "resolved_color_flow_id": ("flow:2,1" if layout == "topology-replay" else None),
        "resolved_helicity_id": (
            None
            if layout == "topology-replay"
            else gate.SELECTED_UNION_HELICITY_ID
            if selected_union
            else "h:-1,+1"
        ),
        "structural_zero_helicity_count": 0,
    }
    color_request = selector["color_flow_request"]
    helicity_request = selector["helicity_request"]
    generation: dict[str, object] = {}
    profiles: dict[str, object] = {}
    samples_by_cell: dict[tuple[str, int], list[dict[str, object]]] = {
        (mode, batch): [] for mode in ALL_MODES for batch in gate.BATCH_SIZES
    }
    schedule_entries: list[dict[str, object]] = []
    schedule_index = 0
    identifier = 1
    deviations = (-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003)
    for round_index, deviation in enumerate(deviations):
        offset = round_index % len(ALL_MODES)
        modes = ALL_MODES[offset:] + ALL_MODES[:offset]
        for batch_size in gate.BATCH_SIZES:
            for mode in modes:
                mode_ratio = (
                    compiled_to_recurrence_ratio
                    if mode == "compiled"
                    else 1.02
                    if mode == "eager"
                    else 1.0
                )
                base_runtime = (
                    1.0e-6
                    * (1.0 + 0.1 * gate.BATCH_SIZES.index(batch_size))
                    * mode_ratio
                )
                mode_runtime_ratio = (
                    runtime_role_ratio * eager_runtime_role_ratio
                    if mode == "eager"
                    else runtime_role_ratio
                )
                mode_resource_ratio = (
                    resource_role_ratio * eager_resource_role_ratio
                    if mode == "eager"
                    else resource_role_ratio
                )
                runtime = base_runtime * mode_runtime_ratio * (1.0 + deviation)
                sample, entry = _sample(
                    role=role,
                    layout=layout,
                    identifier=identifier,
                    schedule_index=schedule_index,
                    round_index=round_index,
                    batch_size=batch_size,
                    runtime=runtime,
                    cold_load=0.2 * mode_resource_ratio * (1.0 + deviation),
                    cold_rss=int(100_000_000 * mode_resource_ratio * (1.0 + deviation)),
                    profiled_rss=int(
                        120_000_000 * mode_resource_ratio * (1.0 + deviation)
                    ),
                )
                entry.update({"mode": mode, "batch_size": batch_size})
                schedule_entries.append(entry)
                samples_by_cell[(mode, batch_size)].append(sample)
                identifier += 1
                schedule_index += 1

    for mode in ALL_MODES:
        semantic = _semantic_identity(
            layout,
            mode,
            selected_union=selected_union,
        )
        semantic_sha256 = _sha256(semantic)
        optimization = 3 if mode == "compiled" else 2
        mode_generation_ratio = (
            generation_role_ratio * eager_generation_role_ratio
            if mode == "eager"
            else generation_role_ratio
        )
        mode_resource_ratio = (
            resource_role_ratio * eager_resource_role_ratio
            if mode == "eager"
            else resource_role_ratio
        )
        generation[mode] = {
            "mode": mode,
            "generation_wall_seconds": (
                (12.0 if mode == "compiled" else 11.0 if mode == "eager" else 10.0)
                * mode_generation_ratio
            ),
            "generation_reused": False,
            "peak_rss": _peak(int(400_000_000 * mode_resource_ratio)),
            "artifact_stats": {
                "file_count": 4,
                "size_bytes": int(10_000_000 * mode_resource_ratio),
            },
            "artifact_identity": {
                "semantic_identity": semantic,
                "semantic_identity_sha256": semantic_sha256,
            },
            "effective_contract": {
                "backend": "jit",
                "color_accuracy": "lc",
                "execution_mode": mode,
                "jit_optimization_level": optimization,
                "lc_flow_layout": layout,
            },
        }
        physical_colors = semantic["physical_color_flows"]
        assert isinstance(physical_colors, dict)
        mode_selector = dict(selector)
        if selected_union and mode == "compiled":
            mode_selector["structural_zero_helicity_count"] = (
                selected_union_helicity_count - 1
            )
        profiles[mode] = {
            "mode": mode,
            "process_expression": process,
            "process_id": "d_dbar_to_z_6g"
            if process.startswith("d ")
            else "u_ubar_to_z_6g",
            "selector_contract": mode_selector,
            "validation": {
                "passes": True,
                "fixture": {
                    "point_count": 1,
                    "points_sha256": "b" * 64,
                },
                **(
                    {
                        "resolved_color_ids": physical_colors["ordered_ids"],
                        "resolved_helicity_ids": [gate.SELECTED_UNION_HELICITY_ID],
                    }
                    if selected_union
                    else {}
                ),
            },
            "artifact_semantic_identity": semantic,
            "artifact_semantic_identity_sha256": semantic_sha256,
            "profiles": [
                _aggregate(batch, samples_by_cell[(mode, batch)])
                for batch in gate.BATCH_SIZES
            ],
        }

    native_sha = "d" * 64 if candidate else "c" * 64
    build_inputs_sha = "f" * 64 if candidate else "e" * 64
    lane_measurements = {
        mode: {
            "passes": True,
            "observed_batch_sizes": list(gate.BATCH_SIZES),
            "missing_batch_sizes": [],
            "errors": [],
        }
        for mode in gate.CAPTURE_MODES
    }
    if selected_union:
        selected_union_projections = {
            mode: _selected_union_projection(
                profiles[mode]["artifact_semantic_identity"],
                process_id="d_dbar_to_z_6g",
                color_flow_request=str(color_request),
            )
            for mode in gate.CAPTURE_MODES
        }
        assert (
            len({_sha256(value) for value in selected_union_projections.values()}) == 1
        )
        common_physics_contract = selected_union_projections["compiled"]
        artifact_lane_contracts = {
            mode: {
                "synthetic_lane_identity": mode,
                "selected_union_workload_projection": projection,
            }
            for mode, projection in selected_union_projections.items()
        }
        comparison_policy = gate.SELECTED_UNION_SEMANTIC_COMPARISON_POLICY
    else:
        common_physics_contract = {"synthetic": True}
        artifact_lane_contracts = {
            mode: {"synthetic_lane_identity": mode} for mode in gate.CAPTURE_MODES
        }
        comparison_policy = gate.STRICT_SEMANTIC_COMPARISON_POLICY
    eager_measurement = {
        "passes": True,
        "lanes": {
            "eager": {
                "passes": True,
                "observed_batch_sizes": list(gate.BATCH_SIZES),
                "missing_batch_sizes": [],
                "errors": [],
            }
        },
    }
    capture_acceptance = {
        "kind": "pyamplicol-three-lane-layout-capture",
        "schema_version": 6,
        "complete": True,
        "evidence_complete": True,
        "passes": True,
        "authoritative_eligible": True,
        "authoritative_ineligibility_reasons": [],
        "generation_specialized_axes_by_mode": {},
        "incomplete_physical_axes": [],
        "required_modes": list(gate.CAPTURE_MODES),
        "observed_modes": list(gate.CAPTURE_MODES),
        "missing_modes": [],
        "diagnostic_modes": list(gate.DIAGNOSTIC_MODES),
        "eager_diagnostic": {
            "requested": True,
            "observed": True,
            "complete": True,
            "passes": True,
            "measurement_contract": eager_measurement,
            "artifact_semantic_contract": {
                "passes": True,
                "errors": [],
                "lanes_match": True,
            },
            "validation_summary": {"passes": True},
            "ineligibility_reasons": [],
        },
        "required_batch_sizes": list(gate.BATCH_SIZES),
        "observed_batch_sizes": list(gate.BATCH_SIZES),
        "missing_batch_sizes": [],
        "measurement_contract": {
            "passes": True,
            "lanes": lane_measurements,
            "minimum_authoritative_samples": 7,
            "configured_internal_minimum_samples": 7,
            "configured_subprocess_samples": 7,
            "configured_warmup_runs": 2,
            "root_processes_match": True,
            "schedule": {
                "passes": True,
                "errors": [],
                "entry_count": len(schedule_entries),
                "unique_worker_command_count": len(schedule_entries),
                "subprocess_samples_per_cell": 7,
            },
        },
        "artifact_semantic_contract": {
            "passes": True,
            "errors": [],
            "lanes_match": True,
            "comparison_policy": comparison_policy,
            "lane_contracts": artifact_lane_contracts,
            "common_physics_contract": common_physics_contract,
        },
        "layout": layout,
        "generation_only": False,
        "lane_self_validation_passes": True,
        "pairwise_validation_passes": True,
    }
    session_id = "synthetic-paired-session"
    plan = [
        {
            field: entry[field]
            for field in ("schedule_index", "round", "mode", "batch_size")
        }
        for entry in schedule_entries
    ]
    ready = _address(
        {
            "kind": "pyamplicol-paired-profile-ready",
            "schema_version": 1,
            "session_id": session_id,
            "role": role,
            "layout": layout,
            "source_revision": revision,
            "ready_at_utc": (
                _layout_epoch(layout) - dt.timedelta(seconds=1)
            ).isoformat(),
            "profile_schedule_plan": plan,
            "profile_schedule_plan_sha256": _sha256(plan),
        }
    )
    completions: list[dict[str, object]] = []
    for entry in schedule_entries:
        index = int(entry["schedule_index"])
        round_index = int(entry["round"])
        expected_order = (
            ("baseline", "candidate")
            if round_index % 2 == 0
            else ("candidate", "baseline")
        )
        order = expected_order.index(role)
        issued, started, finished, recorded = _slot_times(
            role=role,
            layout=layout,
            schedule_index=index,
            round_index=round_index,
        )
        token = _address(
            {
                "kind": "pyamplicol-paired-profile-token",
                "schema_version": 1,
                "session_id": session_id,
                "role": role,
                "layout": layout,
                **{
                    field: entry[field]
                    for field in ("schedule_index", "round", "mode", "batch_size")
                },
                "pair_index": index,
                "order_in_pair": order,
                "issued_at_utc": issued.isoformat(),
            }
        )
        completion = _address(
            {
                "kind": "pyamplicol-paired-profile-completion",
                "schema_version": 1,
                "session_id": session_id,
                "role": role,
                "layout": layout,
                **{
                    field: entry[field]
                    for field in ("schedule_index", "round", "mode", "batch_size")
                },
                "token_sha256": token["content_sha256"],
                "worker_invocation_sha256": entry["worker_invocation"][
                    "content_sha256"
                ],
                "worker_result_record_sha256": entry["worker_result_record"][
                    "content_sha256"
                ],
                "worker_started_at_utc": started.isoformat(),
                "worker_finished_at_utc": finished.isoformat(),
                "recorded_at_utc": recorded.isoformat(),
            }
        )
        entry["paired_profile_token"] = token
        entry["paired_profile_completion"] = completion
        completions.append(completion)
    return {
        "kind": gate.CAPTURE_KIND,
        "schema_version": gate.MINIMUM_CAPTURE_SCHEMA,
        "complete": True,
        "passes": True,
        "capture_acceptance": capture_acceptance,
        "process": process,
        "process_name": ("ddbar_Z_6g" if process.startswith("d ") else "uubar_Z_6g"),
        "workload": workload,
        "source": {
            "revision": revision,
            "dirty": False,
            "untracked_files_checked": True,
        },
        "provenance": {
            "host": {
                "system": "TestOS",
                "machine": "test64",
                "cpu_model": "synthetic",
                "logical_cpu_count": 8,
            }
        },
        "runtime_provenance": {
            "interpreter": {
                "implementation": "CPython",
                "python_version": "3.12.6",
                "path": interpreter_path,
                "resolved_path": interpreter_path,
                "sha256": "1" * 64,
            },
            "native_extension": {
                "sha256": native_sha,
                "build_inputs_sha256": build_inputs_sha,
                "package_version": "0.1.0+candidate",
            },
            "active_build_info": {
                "payload": {
                    "source_revision": revision,
                    "version": "0.1.0+candidate",
                }
            },
        },
        "configuration": {
            "batch_sizes": list(gate.BATCH_SIZES),
            "target_runtime_seconds": 5.0,
            "minimum_samples": 7,
            "subprocess_samples": 7,
            "warmup_runs": 2,
            "color_flow_request": color_request,
            "helicity_request": helicity_request,
            "lc_flow_layout": layout,
            "validation_seed": 12345,
            "point_tile_size": 1024,
            "jit_optimization_level": 3,
            "generation_only": False,
            "modes": list(ALL_MODES),
            "model_identities": {
                mode: {
                    "kind": "explicit-prepared-model",
                    "resource_id": None,
                    "file": {
                        "path": prepared_model_path,
                        "resolved_path": prepared_model_path,
                        "size_bytes": 1234,
                        "sha256": prepared_model_sha256,
                    },
                    "compile_excluded_from_generation": True,
                }
                for mode in ALL_MODES
            },
        },
        "generation": generation,
        "profile_schedule": {
            "kind": "pyamplicol-interleaved-subprocess-profile-schedule",
            "schema_version": 2,
            "algorithm": "round-major-cyclic-mode-and-batch-interleave-v1",
            "sample_unit": "independent-profile-worker-subprocess",
            "subprocess_samples_per_cell": 7,
            "modes": list(ALL_MODES),
            "batch_sizes": list(gate.BATCH_SIZES),
            "entries": schedule_entries,
        },
        "profiles": profiles,
        "paired_profile_coordination": {
            "kind": "pyamplicol-paired-profile-coordination",
            "schema_version": 1,
            "session_id": session_id,
            "role": role,
            "layout": layout,
            "ready_record": ready,
            "completion_records": completions,
        },
        "validation_summary": {
            "passes": True,
            "selectors_match": True,
            "fixtures_match": True,
            "lane_validation_passes": True,
            "pairwise_validation_passes": True,
        },
        "selector_contracts_match": True,
        "validation_fixtures_match": True,
    }


def _write_campaign(
    tmp_path: Path,
    *,
    process: str = "d d~ > Z g g g g g g",
    runtime_ratio: float = 1.0,
    compiled_to_recurrence_ratio: float = 1.05,
    generation_ratio: float = 1.0,
    resource_ratio: float = 1.0,
    eager_runtime_ratio: float = 1.0,
    eager_generation_ratio: float = 1.0,
    eager_resource_ratio: float = 1.0,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for role in ("baseline", "candidate"):
        for layout in gate.LAYOUTS:
            payload = _capture(
                role=role,
                layout=layout,
                process=process,
                runtime_ratio=runtime_ratio,
                compiled_to_recurrence_ratio=compiled_to_recurrence_ratio,
                generation_ratio=generation_ratio,
                resource_ratio=resource_ratio,
                eager_runtime_ratio=eager_runtime_ratio,
                eager_generation_ratio=eager_generation_ratio,
                eager_resource_ratio=eager_resource_ratio,
            )
            path = tmp_path / f"{role}-{layout}.json"
            path.write_bytes(_canonical(payload) + b"\n")
            result[f"{role}-{layout}"] = path
    campaign_path = tmp_path / "paired-campaign.json"
    campaign_path.write_bytes(
        _canonical(_paired_campaign_record(result, process=process)) + b"\n"
    )
    result["campaign"] = campaign_path
    watchdog_path = tmp_path / "memory-watchdog-report.json"
    watchdog_path.write_bytes(
        _canonical(_watchdog_report_record(campaign_path)) + b"\n"
    )
    result["watchdog-report"] = watchdog_path
    return result


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _watchdog_report_record(campaign_path: Path) -> dict[str, object]:
    campaign = json.loads(campaign_path.read_text())
    result_identity = _file_identity(campaign_path)
    command = [
        campaign["roles"]["candidate"]["python"],
        str(ROOT / gate.PAIRED_DRIVER_RELATIVE_PATH),
        "--result-json",
        str(campaign_path),
        "--session-id",
        campaign["session_id"],
    ]
    return _address(
        {
            "kind": gate.WATCHDOG_REPORT_KIND,
            "schema_version": gate.WATCHDOG_REPORT_SCHEMA,
            "complete": True,
            "passes": True,
            "watchdog": _file_identity(ROOT / gate.WATCHDOG_RELATIVE_PATH),
            "working_directory": str(ROOT),
            "execution": {
                "command": command,
                "command_sha256": _sha256(command),
                "started_at_utc": (
                    dt.datetime.fromisoformat(campaign["started_at_utc"])
                    - dt.timedelta(seconds=1)
                ).isoformat(),
                "finished_at_utc": (
                    dt.datetime.fromisoformat(campaign["finished_at_utc"])
                    + dt.timedelta(seconds=1)
                ).isoformat(),
                "elapsed_wall_seconds": 20_000.0,
                "child_pid": 4242,
                "child_exit_code": 0,
                "watchdog_exit_code": 0,
                "outcome": "command-finished",
                "reason": None,
            },
            "enforcement": {
                "scope": gate.WATCHDOG_SCOPE,
                "limit_bytes": gate.WATCHDOG_LIMIT_BYTES,
                "poll_interval_seconds": 0.25,
                "terminate_grace_seconds": 5.0,
                "metric": "process-tree-rss",
                "probe_sample_count": 80_000,
                "probe_failure_count": 0,
                "maximum_consecutive_probe_failures": 0,
                "completed_under_retry_policy": True,
                "peak_rss_bytes": 10 * 1024**3,
                "peak_physical_footprint_bytes": None,
                "peak_guard_bytes": 10 * 1024**3,
                "peak_processes": 4,
            },
            "result_binding": {
                "requested": True,
                "requested_path": str(campaign_path),
                "identity": result_identity,
                "error": None,
            },
        }
    )


def _paired_campaign_record(
    paths: dict[str, Path],
    *,
    process: str,
) -> dict[str, object]:
    captures = {
        role: {
            layout: json.loads(paths[f"{role}-{layout}"].read_text())
            for layout in gate.LAYOUTS
        }
        for role in ("baseline", "candidate")
    }
    harness_path = ROOT / gate.PAIRED_HARNESS_RELATIVE_PATH
    harness_sha256 = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    harness = _address(
        {
            "kind": gate.PAIRED_HARNESS_KIND,
            "schema_version": gate.PAIRED_HARNESS_SCHEMA,
            "candidate_relative_path": gate.PAIRED_HARNESS_RELATIVE_PATH,
            "head_blob_sha256": harness_sha256,
            "working_file_sha256": harness_sha256,
            "head_blob_equals_working_file": True,
        }
    )
    orchestrator_path = ROOT / gate.PAIRED_DRIVER_RELATIVE_PATH
    orchestrator_sha256 = hashlib.sha256(orchestrator_path.read_bytes()).hexdigest()
    orchestrator = _address(
        {
            "kind": gate.PAIRED_HARNESS_KIND,
            "schema_version": gate.PAIRED_HARNESS_SCHEMA,
            "candidate_relative_path": gate.PAIRED_DRIVER_RELATIVE_PATH,
            "head_blob_sha256": orchestrator_sha256,
            "working_file_sha256": orchestrator_sha256,
            "head_blob_equals_working_file": True,
        }
    )
    layouts: dict[str, object] = {}
    latest_recorded = _TEST_EPOCH
    for layout in gate.LAYOUTS:
        pairs: list[dict[str, object]] = []
        baseline_entries = captures["baseline"][layout]["profile_schedule"]["entries"]
        candidate_entries = captures["candidate"][layout]["profile_schedule"]["entries"]
        for pair_index, (baseline_entry, candidate_entry) in enumerate(
            zip(baseline_entries, candidate_entries, strict=True)
        ):
            assert all(
                baseline_entry[field] == candidate_entry[field]
                for field in ("schedule_index", "round", "mode", "batch_size")
            )
            round_index = baseline_entry["round"]
            role_order = (
                ["baseline", "candidate"]
                if round_index % 2 == 0
                else ["candidate", "baseline"]
            )
            completions = {
                "baseline": baseline_entry["paired_profile_completion"],
                "candidate": candidate_entry["paired_profile_completion"],
            }
            for completion in completions.values():
                latest_recorded = max(
                    latest_recorded,
                    dt.datetime.fromisoformat(completion["recorded_at_utc"]),
                )
            pairs.append(
                _address(
                    {
                        "kind": "pyamplicol-paired-profile-pair",
                        "schema_version": 1,
                        "pair_index": pair_index,
                        **{
                            field: baseline_entry[field]
                            for field in (
                                "schedule_index",
                                "round",
                                "mode",
                                "batch_size",
                            )
                        },
                        "role_order": role_order,
                        "completions": completions,
                    }
                )
            )
        layouts[layout] = {
            "layout": layout,
            "ready_records": {
                role: captures[role][layout]["paired_profile_coordination"][
                    "ready_record"
                ]
                for role in ("baseline", "candidate")
            },
            "paired_schedule_algorithm": (
                "slot-adjacent-round-alternating-role-order-v1"
            ),
            "pairs": pairs,
            "captures": {
                role: _file_identity(paths[f"{role}-{layout}"])
                for role in ("baseline", "candidate")
            },
        }
    roles = {}
    for role in ("baseline", "candidate"):
        capture = captures[role]["topology-replay"]
        runtime = capture["runtime_provenance"]["interpreter"]
        model = capture["configuration"]["model_identities"]["recurrence"]["file"]
        roles[role] = {
            "source_root": f"/synthetic/{role}/source",
            "source_revision": capture["source"]["revision"],
            "python": runtime["path"],
            "python_resolved_target": runtime["resolved_path"],
            "python_sha256": runtime["sha256"],
            "prepared_model": model["resolved_path"],
            "prepared_model_sha256": model["sha256"],
        }
    topology_configuration = captures["baseline"]["topology-replay"]["configuration"]
    expanded_process = (
        "d d~ > Z g g g g g g"
        if process.casefold().startswith("d ")
        else "u u~ > Z g g g g g g"
    )
    return _address(
        {
            "kind": gate.PAIRED_CAMPAIGN_KIND,
            "schema_version": gate.PAIRED_CAMPAIGN_SCHEMA,
            "complete": True,
            "session_id": "synthetic-paired-session",
            "started_at_utc": (_TEST_EPOCH - dt.timedelta(seconds=2)).isoformat(),
            "finished_at_utc": (
                latest_recorded + dt.timedelta(seconds=1)
            ).isoformat(),
            "process": expanded_process,
            "orchestrator": orchestrator,
            "harness": harness,
            "roles": roles,
            "configuration": {
                "authoritative_modes": list(gate.AUTHORITATIVE_MODES),
                "diagnostic_modes": list(gate.DIAGNOSTIC_MODES),
                "batch_sizes": list(gate.BATCH_SIZES),
                "target_runtime_seconds": topology_configuration[
                    "target_runtime_seconds"
                ],
                "minimum_samples": topology_configuration["minimum_samples"],
                "subprocess_samples": topology_configuration["subprocess_samples"],
                "warmup_runs": topology_configuration["warmup_runs"],
                "watchdog": {
                    "required": True,
                    "report_kind": gate.WATCHDOG_REPORT_KIND,
                    "report_schema_version": gate.WATCHDOG_REPORT_SCHEMA,
                    "limit_bytes": gate.WATCHDOG_LIMIT_BYTES,
                    "scope": gate.WATCHDOG_SCOPE,
                    "binding": gate.WATCHDOG_BINDING,
                },
            },
            "layouts": layouts,
        }
    )


def _compare(
    paths: dict[str, Path],
    **expected_overrides: str,
) -> dict[str, object]:
    expected = {
        "expected_candidate_source_revision": EXPECTED_CANDIDATE_REVISION,
        "expected_baseline_native_build_inputs_sha256": (
            EXPECTED_BASELINE_NATIVE_BUILD_INPUTS
        ),
        "expected_candidate_native_build_inputs_sha256": (
            EXPECTED_CANDIDATE_NATIVE_BUILD_INPUTS
        ),
        "expected_baseline_prepared_model_sha256": (
            EXPECTED_BASELINE_PREPARED_MODEL
        ),
        "expected_candidate_prepared_model_sha256": (
            EXPECTED_CANDIDATE_PREPARED_MODEL
        ),
    }
    expected.update(expected_overrides)
    return gate.compare_capture_files(
        campaign=paths["campaign"],
        watchdog_report=paths["watchdog-report"],
        baseline_topology=paths["baseline-topology-replay"],
        baseline_all_flow_union=paths["baseline-all-flow-union"],
        candidate_topology=paths["candidate-topology-replay"],
        candidate_all_flow_union=paths["candidate-all-flow-union"],
        **expected,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(_canonical(payload) + b"\n")


def _readdress(record: dict[str, object]) -> dict[str, object]:
    unsigned = dict(record)
    unsigned.pop("content_sha256", None)
    return _address(unsigned)


def _entry_and_sample(
    payload: dict[str, object],
    schedule_index: int,
) -> tuple[dict[str, object], dict[str, object]]:
    entry = payload["profile_schedule"]["entries"][schedule_index]
    mode = entry["mode"]
    batch_size = entry["batch_size"]
    aggregate = next(
        aggregate
        for aggregate in payload["profiles"][mode]["profiles"]
        if aggregate["batch_size"] == batch_size
    )
    sample = next(
        sample
        for sample in aggregate["subprocess_samples"]
        if sample["schedule_index"] == schedule_index
    )
    return entry, sample


def _replace_completion(
    payload: dict[str, object],
    schedule_index: int,
    completion: dict[str, object],
) -> None:
    entry = payload["profile_schedule"]["entries"][schedule_index]
    entry["paired_profile_completion"] = completion
    payload["paired_profile_coordination"]["completion_records"][
        schedule_index
    ] = completion


def _replace_worker_provenance(
    payload: dict[str, object],
    schedule_index: int,
    *,
    invocation: dict[str, object],
    process_record: dict[str, object],
) -> None:
    entry, sample = _entry_and_sample(payload, schedule_index)
    old_result = sample.pop("worker_result_record")
    sample["worker_invocation"] = invocation
    sample["worker_process_record"] = process_record
    result_unsigned = dict(old_result)
    result_unsigned.pop("content_sha256")
    result_unsigned["addressed_payload_sha256"] = _sha256(sample)
    result_unsigned["worker_invocation_sha256"] = invocation["content_sha256"]
    result_unsigned["worker_process_record_sha256"] = process_record["content_sha256"]
    invocation_finished = dt.datetime.fromisoformat(invocation["finished_at_utc"])
    result_unsigned["recorded_at_utc"] = (
        invocation_finished + dt.timedelta(milliseconds=500)
    ).isoformat()
    result = _address(result_unsigned)
    sample["worker_result_record"] = result
    entry["worker_invocation"] = invocation
    entry["worker_result_record"] = result
    entry["worker_result_sha256"] = result_unsigned["addressed_payload_sha256"]
    completion_unsigned = dict(entry["paired_profile_completion"])
    completion_unsigned.pop("content_sha256")
    completion_unsigned["worker_invocation_sha256"] = invocation["content_sha256"]
    completion_unsigned["worker_result_record_sha256"] = result["content_sha256"]
    completion_unsigned["worker_started_at_utc"] = invocation["started_at_utc"]
    completion_unsigned["worker_finished_at_utc"] = invocation["finished_at_utc"]
    completion_unsigned["recorded_at_utc"] = result_unsigned["recorded_at_utc"]
    _replace_completion(
        payload,
        schedule_index,
        _address(completion_unsigned),
    )


def _replace_token_issued_at(
    payload: dict[str, object],
    schedule_index: int,
    issued_at: dt.datetime,
) -> None:
    entry = payload["profile_schedule"]["entries"][schedule_index]
    token = dict(entry["paired_profile_token"])
    token["issued_at_utc"] = issued_at.isoformat()
    token = _readdress(token)
    entry["paired_profile_token"] = token
    completion = dict(entry["paired_profile_completion"])
    completion["token_sha256"] = token["content_sha256"]
    _replace_completion(payload, schedule_index, _readdress(completion))


def _replace_ready_at(payload: dict[str, object], ready_at: dt.datetime) -> None:
    ready = dict(payload["paired_profile_coordination"]["ready_record"])
    ready["ready_at_utc"] = ready_at.isoformat()
    payload["paired_profile_coordination"]["ready_record"] = _readdress(ready)


def _rewrite_campaign(
    path: Path,
    mutation: object,
    *,
    readdress_harness: bool = False,
    readdress_outer: bool = True,
) -> None:
    payload = json.loads(path.read_text())
    assert callable(mutation)
    mutation(payload)
    if readdress_harness:
        payload["harness"] = _readdress(payload["harness"])
    if readdress_outer:
        payload = _readdress(payload)
    _write_payload(path, payload)


def _rewrite_watchdog_report(
    path: Path,
    mutation: object,
    *,
    readdress: bool = True,
) -> None:
    payload = json.loads(path.read_text())
    assert callable(mutation)
    mutation(payload)
    if readdress:
        payload = _readdress(payload)
    _write_payload(path, payload)


def _rebind_retained_sample(
    payload: dict[str, object],
    schedule_index: int,
) -> None:
    entry, sample = _entry_and_sample(payload, schedule_index)
    prior = sample.pop("worker_result_record")
    result = dict(prior)
    result.pop("content_sha256")
    result["addressed_payload_sha256"] = _sha256(sample)
    rebound = _address(result)
    sample["worker_result_record"] = rebound
    entry["worker_result_record"] = rebound
    entry["worker_result_sha256"] = result["addressed_payload_sha256"]
    completion = dict(entry["paired_profile_completion"])
    completion["worker_result_record_sha256"] = rebound["content_sha256"]
    _replace_completion(payload, schedule_index, _readdress(completion))


def _refresh_mode_semantic(
    payload: dict[str, object],
    mode: str,
) -> dict[str, object]:
    semantic = payload["profiles"][mode]["artifact_semantic_identity"]
    digest = _sha256(semantic)
    payload["profiles"][mode]["artifact_semantic_identity_sha256"] = digest
    payload["generation"][mode]["artifact_identity"]["semantic_identity"] = (
        copy.deepcopy(semantic)
    )
    payload["generation"][mode]["artifact_identity"]["semantic_identity_sha256"] = (
        digest
    )
    return semantic


def _rebuild_selected_union_projections(payload: dict[str, object]) -> None:
    color_request = payload["configuration"]["color_flow_request"]
    projections = {
        mode: _selected_union_projection(
            payload["profiles"][mode]["artifact_semantic_identity"],
            process_id="d_dbar_to_z_6g",
            color_flow_request=color_request,
        )
        for mode in gate.CAPTURE_MODES
    }
    assert len({_sha256(value) for value in projections.values()}) == 1
    contract = payload["capture_acceptance"]["artifact_semantic_contract"]
    contract["common_physics_contract"] = projections["compiled"]
    for mode, projection in projections.items():
        contract["lane_contracts"][mode]["selected_union_workload_projection"] = (
            projection
        )


def test_gate_rejects_split_schedule_and_sample_worker_provenance(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    entry, sample = _entry_and_sample(payload, 0)
    invocation = dict(sample["worker_invocation"])
    invocation["identifier"] = 999_001
    invocation = _readdress(invocation)
    result = dict(sample["worker_result_record"])
    result["worker_invocation_sha256"] = invocation["content_sha256"]
    result = _readdress(result)
    entry["worker_invocation"] = invocation
    entry["worker_result_record"] = result
    completion = dict(entry["paired_profile_completion"])
    completion["worker_invocation_sha256"] = invocation["content_sha256"]
    completion["worker_result_record_sha256"] = result["content_sha256"]
    _replace_completion(payload, 0, _readdress(completion))
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="sample worker provenance differs"):
        _compare(paths)


def test_gate_rejects_global_cross_role_worker_identity_reuse(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    baseline_path = paths["baseline-topology-replay"]
    candidate_path = paths["candidate-topology-replay"]
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    started = _TEST_EPOCH + dt.timedelta(seconds=14)
    invocation = _address(
        {
            "kind": "shared-forged-invocation",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": (started + dt.timedelta(seconds=1)).isoformat(),
            "wall_seconds": 12.0,
        }
    )
    process_record = _address(
        {"kind": "shared-forged-process", "wall_seconds": 12.0}
    )
    _replace_worker_provenance(
        baseline,
        0,
        invocation=invocation,
        process_record=process_record,
    )
    _replace_worker_provenance(
        candidate,
        1,
        invocation=invocation,
        process_record=process_record,
    )
    _write_payload(baseline_path, baseline)
    _write_payload(candidate_path, candidate)

    with pytest.raises(
        gate.EvidenceError,
        match=r"reuses worker_(invocation|process_record)_sha256 identity",
    ):
        _compare(paths)


def test_gate_rejects_second_token_before_first_worker_finishes(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    _replace_token_issued_at(
        payload,
        0,
        _TEST_EPOCH + dt.timedelta(seconds=1, milliseconds=500),
    )
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="admitted the second role"):
        _compare(paths)


def test_gate_rejects_ready_record_after_first_token(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    _replace_ready_at(
        payload,
        _TEST_EPOCH + dt.timedelta(seconds=3, milliseconds=500),
    )
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="timestamps are inverted"):
        _compare(paths)


def test_gate_rejects_cross_layout_overlap(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    _replace_ready_at(payload, _TEST_EPOCH + dt.timedelta(seconds=100))
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="layouts overlap or reorder"):
        _compare(paths)


def test_gate_binds_valid_outer_campaign_manifest(tmp_path: Path) -> None:
    comparison = _compare(_write_campaign(tmp_path))
    campaign = comparison["inputs"]["paired_campaign"]
    watchdog = comparison["inputs"]["outer_memory_watchdog"]
    assert campaign["campaign_content_sha256"]
    assert campaign["harness_content_sha256"]
    assert campaign["orchestrator_content_sha256"]
    assert watchdog["report_content_sha256"]
    assert watchdog["bound_campaign_sha256"] == campaign["sha256"]
    assert comparison["inputs"]["externally_pinned_role_inputs"] == {
        "candidate_source_revision": EXPECTED_CANDIDATE_REVISION,
        "roles": {
            "baseline": {
                "native_build_inputs_sha256": (
                    EXPECTED_BASELINE_NATIVE_BUILD_INPUTS
                ),
                "prepared_model_sha256": EXPECTED_BASELINE_PREPARED_MODEL,
            },
            "candidate": {
                "native_build_inputs_sha256": (
                    EXPECTED_CANDIDATE_NATIVE_BUILD_INPUTS
                ),
                "prepared_model_sha256": EXPECTED_CANDIDATE_PREPARED_MODEL,
            },
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "expected_candidate_source_revision",
            "3" * 40,
            "externally pinned source revision",
        ),
        (
            "expected_baseline_native_build_inputs_sha256",
            "3" * 64,
            "baseline .*capture.*native-build-input",
        ),
        (
            "expected_candidate_native_build_inputs_sha256",
            "4" * 64,
            "candidate .*capture.*native-build-input",
        ),
        (
            "expected_baseline_prepared_model_sha256",
            "5" * 64,
            "baseline .*capture.*prepared-model",
        ),
        (
            "expected_candidate_prepared_model_sha256",
            "6" * 64,
            "candidate .*capture.*prepared-model",
        ),
    ],
)
def test_gate_requires_externally_pinned_role_inputs(
    field: str,
    value: str,
    message: str,
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    with pytest.raises(gate.EvidenceError, match=message):
        _compare(paths, **{field: value})


@pytest.mark.parametrize(
    ("identity", "message"),
    (
        ("native", "candidate all-flow-union capture.*native-build-input"),
        ("prepared", "candidate all-flow-union capture.*prepared-model"),
    ),
)
def test_gate_applies_external_identity_pins_to_every_layout(
    identity: str,
    message: str,
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    if identity == "native":
        payload["runtime_provenance"]["native_extension"][
            "build_inputs_sha256"
        ] = "7" * 64
    else:
        for model in payload["configuration"]["model_identities"].values():
            model["file"]["sha256"] = "7" * 64
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match=message):
        _compare(paths)


def test_gate_accepts_watchdog_probe_failures_recovered_below_policy(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)

    def recovered(payload: dict[str, object]) -> None:
        enforcement = payload["enforcement"]
        enforcement["probe_failure_count"] = 7
        enforcement["maximum_consecutive_probe_failures"] = 1

    _rewrite_watchdog_report(paths["watchdog-report"], recovered)
    comparison = _compare(paths)
    assert comparison["passes"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("tool", "authenticated watchdog tool"),
        ("limit", "30-GiB enforcement"),
        ("retry", "recovered-probe retry policy"),
        ("session", "command session differs"),
        ("result", "content-address the supplied paired campaign"),
    ),
)
def test_gate_rejects_readdressed_watchdog_report_tampering(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    paths = _write_campaign(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        if mutation == "tool":
            payload["watchdog"]["sha256"] = "f" * 64
        elif mutation == "limit":
            payload["enforcement"]["limit_bytes"] -= 1
        elif mutation == "retry":
            payload["enforcement"]["probe_failure_count"] = 3
            payload["enforcement"]["maximum_consecutive_probe_failures"] = 3
        elif mutation == "session":
            command = payload["execution"]["command"]
            session_index = command.index("--session-id") + 1
            command[session_index] = "forged-session"
            payload["execution"]["command_sha256"] = _sha256(command)
        else:
            payload["result_binding"]["identity"]["sha256"] = "f" * 64

    _rewrite_watchdog_report(paths["watchdog-report"], tamper)
    with pytest.raises(gate.EvidenceError, match=expected):
        _compare(paths)


def test_gate_rejects_outer_campaign_address_tamper(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    _rewrite_campaign(
        paths["campaign"],
        lambda payload: payload.__setitem__("complete", False),
        readdress_outer=False,
    )

    with pytest.raises(gate.EvidenceError, match="content address"):
        _compare(paths)


def test_gate_rejects_campaign_capture_hash_drift(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    with paths["candidate-topology-replay"].open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(
        gate.EvidenceError,
        match="does not identify its supplied capture file",
    ):
        _compare(paths)


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("session", "session_id differs"),
        ("revision", "differs from its capture provenance"),
        ("pack", "differs from its capture provenance"),
        ("configuration", "configuration differs"),
    ),
)
def test_gate_rejects_semantically_forged_campaign_fields(
    tmp_path: Path,
    field: str,
    expected: str,
) -> None:
    paths = _write_campaign(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        if field == "session":
            payload["session_id"] = "forged-session"
        elif field == "revision":
            payload["roles"]["candidate"]["source_revision"] = "f" * 40
        elif field == "pack":
            payload["roles"]["candidate"]["prepared_model_sha256"] = "f" * 64
        else:
            payload["configuration"]["warmup_runs"] = 3

    _rewrite_campaign(paths["campaign"], mutate)
    with pytest.raises(gate.EvidenceError, match=expected):
        _compare(paths)


def test_gate_rejects_capture_with_more_than_two_warmups(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    payload["configuration"]["warmup_runs"] = 3
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="exactly 2 warmup"):
        _compare(paths)


def test_gate_rejects_eight_outer_subprocess_pairs(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    payload["configuration"]["subprocess_samples"] = 8
    _write_payload(path, payload)

    with pytest.raises(
        gate.EvidenceError,
        match="exactly 7 independent subprocess pairs",
    ):
        _compare(paths)


@pytest.mark.parametrize("field", ("candidate_relative_path", "head_blob_sha256"))
def test_gate_rejects_forged_campaign_harness(
    tmp_path: Path,
    field: str,
) -> None:
    paths = _write_campaign(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["harness"][field] = (
            "tools/developer/forged.py"
            if field == "candidate_relative_path"
            else "f" * 64
        )

    _rewrite_campaign(
        paths["campaign"],
        mutate,
        readdress_harness=True,
    )
    with pytest.raises(gate.EvidenceError, match="exact candidate benchmark driver"):
        _compare(paths)


@pytest.mark.parametrize(
    "process",
    (
        "d d~ > z + 6*g",
        "d d~ > Z g g g g g g",
        "u u~ > Z g g g g g g",
    ),
)
def test_gate_accepts_exact_d_route_and_existing_u_route(
    tmp_path: Path,
    process: str,
) -> None:
    comparison = _compare(_write_campaign(tmp_path, process=process))
    assert comparison["complete"] is True
    assert comparison["passes"] is True
    assert comparison["failures"] == []
    unsigned = dict(comparison)
    digest = unsigned.pop("content_sha256")
    assert digest == _sha256(unsigned)
    assert len(comparison["runtime_cells"]) == 12
    assert len(comparison["paired_compiled_recurrence_ratios"]) == 4
    assert len(comparison["eager_diagnostic"]["runtime_cells"]) == 6
    assert comparison["eager_diagnostic"]["passes"] is True
    assert (
        comparison["eager_diagnostic"]["paired_compiled_recurrence_ratio_applicable"]
        is False
    )


def test_selected_union_projection_allows_only_audited_lane_metadata_drift(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    payload = json.loads(paths["baseline-all-flow-union"].read_text())
    compiled = payload["profiles"]["compiled"]["artifact_semantic_identity"]
    recurrence = payload["profiles"]["recurrence"]["artifact_semantic_identity"]
    assert compiled["reduction_ordering"] != recurrence["reduction_ordering"]
    assert (
        compiled["physical_helicities"]["ordered_entries"]
        != recurrence["physical_helicities"]["ordered_entries"]
    )
    contract = payload["capture_acceptance"]["artifact_semantic_contract"]
    assert contract["comparison_policy"] == (
        gate.SELECTED_UNION_SEMANTIC_COMPARISON_POLICY
    )
    assert all(
        lane["selected_union_workload_projection"]
        == contract["common_physics_contract"]
        for lane in contract["lane_contracts"].values()
    )
    comparison = _compare(paths)
    assert comparison["passes"] is True


@pytest.mark.parametrize(
    "mutation",
    ("policy", "lane", "color", "model", "selector", "selected"),
)
def test_selected_union_projection_fails_closed_on_projection_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    contract = payload["capture_acceptance"]["artifact_semantic_contract"]
    common = contract["common_physics_contract"]
    if mutation == "policy":
        contract["comparison_policy"] = gate.STRICT_SEMANTIC_COMPARISON_POLICY
    elif mutation == "lane":
        lane = contract["lane_contracts"]["recurrence"][
            "selected_union_workload_projection"
        ]
        lane["physical_helicities"]["selected_entry"]["coefficient"] = 2.0
    else:
        if mutation == "color":
            common["physical_color_flows"]["ordered_ids_sha256"] = "0" * 64
        elif mutation == "model":
            common["model_common_physics_identity_sha256"] = "0" * 64
        elif mutation == "selector":
            common["profile_selector"]["resolved_color_flow_id"] = "flow:forged"
        else:
            common["physical_helicities"]["selected_entry"]["coefficient"] = 0.0
        for lane in contract["lane_contracts"].values():
            lane["selected_union_workload_projection"] = copy.deepcopy(common)
    _write_payload(path, payload)
    with pytest.raises(
        gate.EvidenceError,
        match=r"policy|projection|selector|helic|coefficient",
    ):
        _compare(paths)


def test_selected_union_projection_drift_across_captures_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    for mode in ALL_MODES:
        semantic = payload["profiles"][mode]["artifact_semantic_identity"]
        selected = semantic["physical_helicities"]["ordered_entries"][
            gate.SELECTED_UNION_HELICITY_INDEX
        ]
        selected["coefficient"] = 2.0
        semantic["physical_helicities"]["ordered_entries_sha256"] = _sha256(
            semantic["physical_helicities"]["ordered_entries"]
        )
        _refresh_mode_semantic(payload, mode)
    _rebuild_selected_union_projections(payload)
    _write_payload(path, payload)

    with pytest.raises(gate.EvidenceError, match="workload identities differ"):
        _compare(paths)


@pytest.mark.parametrize(
    ("process", "layout", "helicity"),
    (
        ("d d~ > Z g g g g g g", "topology-replay", "1"),
        ("u u~ > Z g g g g g g", "all-flow-union", "h:-1,+1"),
        ("d d~ > Z g g g g g g", "all-flow-union", "1"),
    ),
)
def test_selected_union_exception_is_unavailable_outside_exact_route(
    tmp_path: Path,
    process: str,
    layout: str,
    helicity: str,
) -> None:
    paths = _write_campaign(tmp_path, process=process)
    path = paths[f"candidate-{layout}"]
    payload = json.loads(path.read_text())
    payload["configuration"]["helicity_request"] = helicity
    payload["capture_acceptance"]["artifact_semantic_contract"]["comparison_policy"] = (
        gate.SELECTED_UNION_SEMANTIC_COMPARISON_POLICY
    )
    _write_payload(path, payload)
    with pytest.raises(gate.EvidenceError, match=r"policy|must select"):
        _compare(paths)


def test_strict_topology_still_rejects_full_axis_drift(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    semantic = payload["profiles"]["compiled"]["artifact_semantic_identity"]
    semantic["physical_helicities"]["ordered_entries"][0]["index"] = 99
    _refresh_mode_semantic(payload, "compiled")
    _write_payload(path, payload)
    with pytest.raises(gate.EvidenceError, match="workload identities differ"):
        _compare(paths)


def test_gate_recomputes_runtime_statistics_and_rejects_forged_aggregate(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    aggregate = payload["profiles"]["compiled"]["profiles"][0]
    aggregate["wall_seconds_per_point_median"] *= 0.5
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(gate.EvidenceError, match="not reproducible"):
        _compare(paths)


def test_gate_independently_revalidates_eager_raw_samples(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    aggregate = payload["profiles"]["eager"]["profiles"][0]
    aggregate["wall_seconds_per_point_median"] *= 0.5
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(gate.EvidenceError, match="not reproducible"):
        _compare(paths)


def test_gate_rejects_readdressed_sub_five_second_native_wall_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    _, sample = _entry_and_sample(payload, 0)
    raw = sample["inner_native_wall_blocks"]
    raw["observed_native_wall_seconds"] = 4.999
    raw["minimum_duration_satisfied"] = True
    _rebind_retained_sample(payload, 0)
    _write_payload(path, payload)

    with pytest.raises(
        gate.EvidenceError,
        match="raw native-wall measurement contract is invalid",
    ):
        _compare(paths)


def test_gate_rejects_readdressed_raw_calibration_tampering(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    _, sample = _entry_and_sample(payload, 0)
    raw = sample["inner_native_wall_blocks"]
    raw["calibration"]["duration_headroom_factor"] = 1.01
    sample["worker_measurement"]["inner_native_wall_blocks"] = copy.deepcopy(raw)
    _rebind_retained_sample(payload, 0)
    _write_payload(path, payload)

    with pytest.raises(
        gate.EvidenceError,
        match="raw native-wall measurement contract is invalid",
    ):
        _compare(paths)


def test_gate_reports_runtime_and_same_round_ratio_failures(tmp_path: Path) -> None:
    runtime = _compare(_write_campaign(tmp_path / "runtime", runtime_ratio=1.04))
    assert runtime["passes"] is False
    assert any(
        failure.startswith("runtime regression:") for failure in runtime["failures"]
    )

    ratio = _compare(
        _write_campaign(
            tmp_path / "ratio",
            compiled_to_recurrence_ratio=1.20,
        )
    )
    assert ratio["passes"] is False
    assert any(
        failure.startswith("compiled/recurrence ratio regression:")
        for failure in ratio["failures"]
    )


def test_gate_enforces_generation_cell_and_geometric_mean_limits(
    tmp_path: Path,
) -> None:
    comparison = _compare(_write_campaign(tmp_path, generation_ratio=1.11))
    assert comparison["passes"] is False
    assert comparison["generation"]["geometric_mean_passes"] is False
    assert "generation geometric-mean regression" in comparison["failures"]
    assert (
        sum(
            failure.startswith("generation regression:")
            for failure in comparison["failures"]
        )
        == 4
    )


def test_eager_diagnostic_runtime_generation_and_resource_failures(
    tmp_path: Path,
) -> None:
    runtime = _compare(
        _write_campaign(
            tmp_path / "runtime",
            eager_runtime_ratio=1.04,
        )
    )
    assert runtime["passes"] is False
    assert all(cell["passes"] is True for cell in runtime["runtime_cells"])
    assert runtime["eager_diagnostic"]["runtime_passes"] is False
    assert any(
        failure.startswith("eager diagnostic runtime regression:")
        for failure in runtime["eager_diagnostic"]["failures"]
    )
    assert any(
        failure.startswith("eager diagnostic runtime regression:")
        for failure in runtime["failures"]
    )

    generation = _compare(
        _write_campaign(
            tmp_path / "generation",
            eager_generation_ratio=1.11,
        )
    )
    assert generation["passes"] is False
    assert generation["generation"]["geometric_mean_passes"] is True
    assert generation["eager_diagnostic"]["generation"]["passes"] is False
    assert any(
        failure.startswith("eager diagnostic generation regression:")
        for failure in generation["eager_diagnostic"]["failures"]
    )
    assert any(
        failure.startswith("eager diagnostic generation regression:")
        for failure in generation["failures"]
    )

    resources = _compare(
        _write_campaign(
            tmp_path / "resources",
            eager_resource_ratio=1.04,
        )
    )
    assert resources["passes"] is False
    assert resources["eager_diagnostic"]["resource_passes"] is False
    assert any(
        failure.startswith("eager diagnostic resource regression:")
        for failure in resources["eager_diagnostic"]["failures"]
    )
    assert any(
        failure.startswith("eager diagnostic resource regression:")
        for failure in resources["failures"]
    )


def test_eager_generation_has_a_five_percent_geometric_mean_gate(
    tmp_path: Path,
) -> None:
    comparison = _compare(
        _write_campaign(
            tmp_path,
            eager_generation_ratio=1.06,
        )
    )
    eager_generation = comparison["eager_diagnostic"]["generation"]
    assert comparison["passes"] is False
    assert all(cell["passes"] is True for cell in eager_generation["cells"])
    assert eager_generation["geometric_mean_passes"] is False
    assert eager_generation["passes"] is False
    assert any(
        failure == "eager diagnostic generation geometric-mean regression"
        for failure in comparison["eager_diagnostic"]["failures"]
    )


def test_gate_requires_eager_present_and_admissible(tmp_path: Path) -> None:
    missing_paths = _write_campaign(tmp_path / "missing")
    missing_path = missing_paths["candidate-topology-replay"]
    missing = json.loads(missing_path.read_text())
    missing["configuration"]["modes"].remove("eager")
    _write_payload(missing_path, missing)
    with pytest.raises(gate.EvidenceError, match="compiled, recurrence, and eager"):
        _compare(missing_paths)

    inadmissible_paths = _write_campaign(tmp_path / "inadmissible")
    inadmissible_path = inadmissible_paths["candidate-topology-replay"]
    inadmissible = json.loads(inadmissible_path.read_text())
    eager = inadmissible["capture_acceptance"]["eager_diagnostic"]
    eager["complete"] = False
    eager["passes"] = None
    eager["ineligibility_reasons"] = ["synthetic failure"]
    _write_payload(inadmissible_path, inadmissible)
    with pytest.raises(gate.EvidenceError, match="not complete and admissible"):
        _compare(inadmissible_paths)


def test_eager_resource_growth_uses_the_runtime_gain_exception(
    tmp_path: Path,
) -> None:
    comparison = _compare(
        _write_campaign(
            tmp_path,
            eager_runtime_ratio=0.85,
            eager_resource_ratio=1.20,
        )
    )
    eager = comparison["eager_diagnostic"]
    assert comparison["passes"] is True
    assert eager["passes"] is True
    assert any(
        resource["within_3_percent_growth_limit"] is False
        and resource["runtime_gain_exception"]["passes"] is True
        for resource in eager["payload_cold_load_rss"]
    )


def test_resource_growth_needs_a_ten_percent_noise_clear_runtime_gain(
    tmp_path: Path,
) -> None:
    regression = _compare(_write_campaign(tmp_path / "regression", resource_ratio=1.04))
    assert regression["passes"] is False
    assert any(
        failure.startswith("resource regression:") for failure in regression["failures"]
    )

    exception = _compare(
        _write_campaign(
            tmp_path / "exception",
            runtime_ratio=0.85,
            resource_ratio=1.20,
        )
    )
    assert exception["passes"] is True
    assert all(
        resource["passes"] is True for resource in exception["payload_cold_load_rss"]
    )
    assert any(
        resource["within_3_percent_growth_limit"] is False
        and resource["runtime_gain_exception"]["passes"] is True
        for resource in exception["payload_cold_load_rss"]
    )


def test_gate_fails_closed_on_workload_identity_drift(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    payload["profiles"]["compiled"]["selector_contract"]["color_flow_count"] = 3
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match="selector differs from its selected-union projection",
    ):
        _compare(paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (("complete", False), ("passes", None)),
)
def test_gate_requires_top_level_complete_passing_captures(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match="not top-level complete and passing",
    ):
        _compare(paths)


def test_gate_requires_consistent_authoritative_capture_acceptance(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    payload["capture_acceptance"]["evidence_complete"] = False
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match="not authoritative, complete, and passing",
    ):
        _compare(paths)


def test_legacy_capture_without_cold_load_has_actionable_error(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["baseline-topology-replay"]
    payload = json.loads(path.read_text())
    del payload["profiles"]["compiled"]["profiles"][0]["subprocess_samples"][0][
        "cold_load_seconds"
    ]
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match=r"predates the resource-evidence contract.*re-run",
    ):
        _compare(paths)


def test_cli_writes_canonical_content_addressed_report(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    output = tmp_path / "comparison.json"
    exit_code = gate.main(
        [
            "--baseline-topology",
            str(paths["baseline-topology-replay"]),
            "--campaign",
            str(paths["campaign"]),
            "--watchdog-report",
            str(paths["watchdog-report"]),
            "--baseline-all-flow-union",
            str(paths["baseline-all-flow-union"]),
            "--candidate-topology",
            str(paths["candidate-topology-replay"]),
            "--candidate-all-flow-union",
            str(paths["candidate-all-flow-union"]),
            "--expected-candidate-source-revision",
            EXPECTED_CANDIDATE_REVISION,
            "--expected-baseline-native-build-inputs-sha256",
            EXPECTED_BASELINE_NATIVE_BUILD_INPUTS,
            "--expected-candidate-native-build-inputs-sha256",
            EXPECTED_CANDIDATE_NATIVE_BUILD_INPUTS,
            "--expected-baseline-prepared-model-sha256",
            EXPECTED_BASELINE_PREPARED_MODEL,
            "--expected-candidate-prepared-model-sha256",
            EXPECTED_CANDIDATE_PREPARED_MODEL,
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text())
    unsigned = copy.deepcopy(payload)
    digest = unsigned.pop("content_sha256")
    assert digest == _sha256(unsigned)
    assert output.read_bytes() == _canonical(payload) + b"\n"
