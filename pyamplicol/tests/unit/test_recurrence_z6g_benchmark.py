# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "recurrence_z6g_benchmark.py"
SPEC = importlib.util.spec_from_file_location("recurrence_z6g_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _arguments(*values: str) -> object:
    arguments = benchmark.parser().parse_args(list(values))
    arguments.modes = benchmark._normalize_modes(arguments)
    arguments.batch_size = benchmark._normalize_batch_sizes(arguments)
    return arguments


def test_modes_default_to_all_lanes_and_preserve_requested_order() -> None:
    default = benchmark.parser().parse_args([])
    assert benchmark._normalize_modes(default) == [
        "compiled",
        "eager",
        "recurrence",
    ]

    selected = benchmark.parser().parse_args(
        ["--mode", "recurrence", "--mode", "eager"]
    )
    assert benchmark._normalize_modes(selected) == ["recurrence", "eager"]


def test_modes_reject_duplicates_and_old_new_flag_mix() -> None:
    duplicate = benchmark.parser().parse_args(
        ["--mode", "compiled", "--mode", "compiled"]
    )
    with pytest.raises(benchmark.HarnessError, match="must be unique"):
        benchmark._normalize_modes(duplicate)

    mixed = benchmark.parser().parse_args(
        ["--mode", "compiled", "--only-mode", "recurrence"]
    )
    with pytest.raises(benchmark.HarnessError, match="cannot be combined"):
        benchmark._normalize_modes(mixed)


def test_default_batches_cover_milestone_zero_and_explicit_values_replace() -> None:
    assert benchmark.DEFAULT_BATCH_SIZES == (1, 128, 1024)
    defaults = benchmark.parser().parse_args([])
    assert defaults.minimum_samples == 7
    assert defaults.subprocess_samples == 7
    assert benchmark._normalize_batch_sizes(defaults) == [
        1,
        128,
        1024,
    ]
    assert benchmark._normalize_batch_sizes(
        benchmark.parser().parse_args(["--batch-size", "8", "--batch-size", "32"])
    ) == [8, 32]

    duplicate = benchmark.parser().parse_args(
        ["--batch-size", "8", "--batch-size", "8"]
    )
    with pytest.raises(benchmark.HarnessError, match="must be unique"):
        benchmark._normalize_batch_sizes(duplicate)


def test_incomplete_capture_requires_explicit_diagnostic_success() -> None:
    assert (
        benchmark._result_exit_code(
            {"complete": True, "passes": True},
            allow_diagnostic_incomplete_success=False,
        )
        == 0
    )
    assert (
        benchmark._result_exit_code(
            {"complete": True, "passes": False},
            allow_diagnostic_incomplete_success=True,
        )
        == 1
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": None},
            allow_diagnostic_incomplete_success=False,
        )
        == 1
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": None},
            allow_diagnostic_incomplete_success=True,
        )
        == 0
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": True},
            allow_diagnostic_incomplete_success=True,
        )
        == 1
    )


def _fixture(
    *,
    point_count: int = 2,
    points_sha256: str = "b" * 64,
    file_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "point_count": point_count,
        "points_sha256": points_sha256,
        "file": {
            "path": "/fixture.json",
            "resolved_path": "/fixture.json",
            "size_bytes": 123,
            "sha256": file_sha256,
        },
    }


def _artifact_semantic_identity(
    *,
    specialized_axes: tuple[str, ...] = (),
    complete: bool = True,
    model_name: str = "built-in-sm",
    model_content_sha256: str = "c" * 64,
    color_id: str = "flow:2,1",
    color_word: tuple[int, ...] = (2, 1),
    helicity_id: str = "h:-1,+1",
    helicity_values: tuple[int, ...] = (-1, 1),
) -> dict[str, object]:
    color_ids = [color_id]
    opposite_helicity_values = tuple(-value for value in helicity_values)
    opposite_helicity_id = "h:" + ",".join(
        f"{value:+d}" for value in opposite_helicity_values
    )
    helicity_ids = [helicity_id, opposite_helicity_id]
    color_entries = [
        {
            "index": 0,
            "id": color_id,
            "kind": "lc-flow",
            "word": list(color_word),
            "representative_id": color_id,
            "computed": True,
            "coefficient": 1.0,
            "structural_zero": None,
        }
    ]
    helicity_entries = [
        {
            "index": 0,
            "id": helicity_id,
            "values": list(helicity_values),
            "representative_id": helicity_id,
            "computed": True,
            "coefficient": 1.0,
            "structural_zero": False,
        },
        {
            "index": 1,
            "id": opposite_helicity_id,
            "values": list(opposite_helicity_values),
            "representative_id": helicity_id,
            "computed": False,
            "coefficient": 1.0,
            "structural_zero": False,
        },
    ]
    normalization = {"average_factor": 36, "color_factor": 3}
    model_common = {
        "name": model_name,
        "content_sha256": model_content_sha256,
        "compiled_schema_version": 9,
        "restriction": None,
    }
    model_manifest = {**model_common, "source_kind": "compiled-model"}
    model_identity = {
        "manifest": model_manifest,
        "manifest_sha256": benchmark._canonical_sha256(model_manifest),
        "common_physics_identity": model_common,
        "common_physics_identity_sha256": benchmark._canonical_sha256(model_common),
    }
    color_specialized = "color_flow" in specialized_axes
    helicity_specialized = "helicity" in specialized_axes
    selector_semantics = {
        "kind": "pyamplicol-runtime-selectors",
        "contract_version": 1,
        "axes": {
            "color_flow": {
                "generation_coverage": (
                    "selected" if color_specialized else "complete"
                ),
                "generation_selection": [0] if color_specialized else [],
                "runtime_contract": (
                    "generation-specialized"
                    if color_specialized
                    else "complete-reusable"
                ),
            },
            "helicity": {
                "generation_coverage": (
                    "selected" if helicity_specialized else "complete"
                ),
                "generation_selection": ({"1": -1} if helicity_specialized else {}),
                "runtime_contract": (
                    "generation-specialized"
                    if helicity_specialized
                    else "complete-reusable"
                ),
            },
        },
        "generation_specialized_axes": list(specialized_axes),
    }
    runtime_selectors = {
        **selector_semantics,
        "provenance": "test-runtime-selectors-v1",
    }
    reduction_ordering = {
        "kind": "lc-diagonal",
        "ordered_groups": [
            {
                "id": "reduction:0",
                "physical_color_ids": color_ids,
                "physical_helicity_ids": helicity_ids,
                "representative_color_id": color_ids[0],
                "representative_helicity_id": helicity_ids[0],
            }
        ],
    }
    execution_ordering = {
        "runtime_process_contract": {"id": "process-1"},
        "manifest_payload_order": [{"path": "plan.bin", "role": "runtime"}],
    }
    return {
        "kind": "pyamplicol-benchmark-artifact-semantic-identity",
        "schema_version": 2,
        "coverage": {
            "color": ("complete" if complete and not color_specialized else "selected"),
            "helicities": (
                "complete" if complete and not helicity_specialized else "selected"
            ),
            "complete_physical_axes": (
                complete and not color_specialized and not helicity_specialized
            ),
        },
        "physical_color_flows": {
            "count": len(color_ids),
            "ordered_ids": color_ids,
            "ordered_ids_sha256": benchmark._canonical_sha256(color_ids),
            "ordered_entries": color_entries,
            "ordered_entries_sha256": benchmark._canonical_sha256(color_entries),
        },
        "physical_helicities": {
            "count": len(helicity_ids),
            "ordered_ids": helicity_ids,
            "ordered_ids_sha256": benchmark._canonical_sha256(helicity_ids),
            "ordered_entries": helicity_entries,
            "ordered_entries_sha256": benchmark._canonical_sha256(helicity_entries),
        },
        "normalization": normalization,
        "normalization_sha256": benchmark._canonical_sha256(normalization),
        "manifest_model_identity": model_identity,
        "runtime_selector_semantics": selector_semantics,
        "runtime_selector_semantics_sha256": benchmark._canonical_sha256(
            selector_semantics
        ),
        "runtime_selectors": runtime_selectors,
        "runtime_selectors_sha256": benchmark._canonical_sha256(runtime_selectors),
        "reduction_ordering": reduction_ordering,
        "reduction_ordering_sha256": benchmark._canonical_sha256(reduction_ordering),
        "reduction_coverage": {
            "complete": True,
            "expected_physical_pair_count": 2,
            "observed_physical_pair_count": 2,
            "errors": [],
        },
        "execution_schedule_ordering": execution_ordering,
        "execution_schedule_ordering_sha256": benchmark._canonical_sha256(
            execution_ordering
        ),
        "generation_specialized_axes": list(specialized_axes),
    }


def _passing_schedule(
    *,
    modes: tuple[str, ...] = benchmark.EXECUTION_MODES,
    batch_sizes: tuple[int, ...] = benchmark.DEFAULT_BATCH_SIZES,
    subprocess_samples: int = benchmark.MIN_AUTHORITATIVE_SAMPLES,
    jit_optimization_level: int = 2,
    lc_flow_layout: str = "topology-replay",
) -> dict[str, object]:
    schedule: dict[str, object] = benchmark._build_profile_schedule(
        modes,
        batch_sizes,
        subprocess_samples=subprocess_samples,
    )
    entries = schedule["entries"]
    assert isinstance(entries, list)
    semantic_identity_sha256 = benchmark._canonical_sha256(
        _artifact_semantic_identity()
    )
    for entry in entries:
        assert isinstance(entry, dict)
        index = entry["schedule_index"]
        command = benchmark._command_identity(
            ("python", "profile-worker", "--schedule-index", str(index))
        )
        verification = {
            "kind": benchmark.WORKER_VERIFICATION_KIND,
            "schema_version": benchmark.WORKER_VERIFICATION_SCHEMA,
            "verified_at_utc": "2026-07-24T00:00:00+00:00",
            "expected": {
                field: (
                    semantic_identity_sha256
                    if field == "artifact_semantic_identity_sha256"
                    else "a" * 64
                )
                for field in benchmark._PROFILE_EXPECTATION_FIELDS
            },
            "observed": {
                field: (
                    semantic_identity_sha256
                    if field == "artifact_semantic_identity_sha256"
                    else "a" * 64
                )
                for field in benchmark._PROFILE_EXPECTATION_FIELDS
            },
            "artifact_semantic_identity_sha256": semantic_identity_sha256,
            "effective_contract": {
                "execution_mode": entry["mode"],
                "backend": "jit",
                "jit_optimization_level": jit_optimization_level,
                "color_accuracy": "lc",
                "lc_flow_layout": lc_flow_layout,
            },
        }
        invocation = {
            "started_at_utc": "2026-07-24T00:00:00+00:00",
            "finished_at_utc": "2026-07-24T00:00:01+00:00",
            "wall_seconds": 1.0,
            "command": command,
        }
        invocation["content_sha256"] = benchmark._canonical_sha256(invocation)
        addressed_payload_sha256 = hashlib.sha256(
            f"worker-payload-{index}".encode()
        ).hexdigest()
        result_record = {
            "kind": benchmark.RETAINED_WORKER_RESULT_KIND,
            "schema_version": benchmark.RETAINED_WORKER_RESULT_SCHEMA,
            "recorded_at_utc": "2026-07-24T00:00:01+00:00",
            "addressed_payload_sha256": addressed_payload_sha256,
            "upstream_worker_result_record_sha256": hashlib.sha256(
                f"upstream-result-{index}".encode()
            ).hexdigest(),
            "worker_process_record_sha256": hashlib.sha256(
                f"worker-process-{index}".encode()
            ).hexdigest(),
            "worker_invocation_sha256": invocation["content_sha256"],
        }
        result_record["content_sha256"] = benchmark._canonical_sha256(result_record)
        entry["worker_invocation"] = invocation
        entry["pre_timing_verification"] = verification
        entry["worker_result_record"] = result_record
        entry["worker_result_sha256"] = addressed_payload_sha256
    return schedule


def _raw_native_wall_blocks(
    *,
    wall_seconds_per_point: float,
    batch_size: int,
    fixture_points_sha256: str,
    repetitions_per_block: int = 1,
    block_count: int = benchmark.MIN_AUTHORITATIVE_SAMPLES,
) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for block_index in range(block_count):
        block = {
            "block_index": block_index,
            "started_at_utc": (f"2026-07-24T00:00:00.{block_index:06d}+00:00"),
            "finished_at_utc": (f"2026-07-24T00:00:00.{block_index + 1:06d}+00:00"),
            "caller_elapsed_seconds": (
                wall_seconds_per_point * repetitions_per_block * batch_size
            ),
            "native_wall_seconds": (
                wall_seconds_per_point * repetitions_per_block * batch_size
            ),
            "wall_seconds_per_point": wall_seconds_per_point,
            "repetitions": repetitions_per_block,
            "batch_size": batch_size,
            "evaluation_count": repetitions_per_block,
            "evaluated_point_count": repetitions_per_block * batch_size,
        }
        block["content_sha256"] = benchmark._canonical_sha256(block)
        blocks.append(block)
    return {
        "kind": "pyamplicol-raw-native-wall-blocks",
        "schema_version": 1,
        "source": "runtime._benchmark_f64_wall_time",
        "fixture_points_sha256": fixture_points_sha256,
        "block_count": block_count,
        "repetitions_per_block": repetitions_per_block,
        "evaluation_count": block_count * repetitions_per_block,
        "evaluated_point_count": (block_count * repetitions_per_block * batch_size),
        "wall_seconds_per_point_median": wall_seconds_per_point,
        "wall_seconds_per_point_mad": 0.0,
        "blocks": blocks,
        "blocks_sha256": benchmark._canonical_sha256(blocks),
    }


def _profile(
    mode: str,
    values: tuple[complex, ...],
    *,
    fixture: dict[str, object] | None = None,
    passes: bool = True,
    schedule: dict[str, object] | None = None,
    specialized_axes: tuple[str, ...] = (),
    semantic_identity: dict[str, object] | None = None,
    selector_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    schedule = _passing_schedule() if schedule is None else schedule
    validation_fixture = _fixture() if fixture is None else fixture
    selector_contract = (
        {
            "color_flow_request": "1",
            "resolved_color_flow_id": "flow:2,1",
            "helicity_request": "1",
            "resolved_helicity_id": None,
            "color_flow_count": 1,
            "helicity_count": 2,
            "structural_zero_helicity_count": 0,
            "workload": "single-runtime-selected-flow/helicity-sum",
        }
        if selector_contract is None
        else copy.deepcopy(selector_contract)
    )
    semantic_identity = (
        _artifact_semantic_identity(specialized_axes=specialized_axes)
        if semantic_identity is None
        else copy.deepcopy(semantic_identity)
    )
    semantic_identity_sha256 = benchmark._canonical_sha256(semantic_identity)
    complex_payloads = [[value.real, value.imag] for value in values]
    if selector_contract["workload"] == "all-flows/runtime-selected-single-helicity":
        resolved_helicity_ids = [selector_contract["resolved_helicity_id"]]
        resolved_color_ids = semantic_identity["physical_color_flows"]["ordered_ids"]
    else:
        resolved_helicity_ids = semantic_identity["physical_helicities"]["ordered_ids"]
        resolved_color_ids = [selector_contract["resolved_color_flow_id"]]
    component_count = len(resolved_helicity_ids) * len(resolved_color_ids)
    resolved_components = [
        [list(value), *([[0.0, 0.0]] * (component_count - 1))]
        for value in complex_payloads
    ]
    validation_record = {
        "passes": passes,
        "fixture": validation_fixture,
        "selected_totals": complex_payloads,
        "resolved_sums": copy.deepcopy(complex_payloads),
        "resolved_helicity_ids": resolved_helicity_ids,
        "resolved_color_ids": resolved_color_ids,
        "resolved_components": resolved_components,
        "point_comparisons": [
            {
                "point_index": index,
                "selected_total": value,
                "resolved_sum": list(value),
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
                "passes": True,
            }
            for index, value in enumerate(complex_payloads)
        ],
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
    }
    lane_contract = {
        "process_id": "process-1",
        "process_expression": "u u~ > Z g g g g g g",
        "selector_contract": selector_contract,
        "validation": validation_record,
        "artifact_semantic_identity": semantic_identity,
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
    }
    lane_contract_sha256 = benchmark._canonical_sha256(lane_contract)
    fixture_points_sha256 = validation_fixture["points_sha256"]
    assert isinstance(fixture_points_sha256, str)
    entries = schedule["entries"]
    assert isinstance(entries, list)
    measurements: list[dict[str, object]] = []
    for batch_size in benchmark.DEFAULT_BATCH_SIZES:
        cell_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry["mode"] == mode
            and entry["batch_size"] == batch_size
        ]
        samples = []
        for entry in cell_entries:
            wall = 1.0e-6 * (1.0 + 0.01 * int(entry["round"]))
            raw_blocks = _raw_native_wall_blocks(
                wall_seconds_per_point=wall,
                batch_size=batch_size,
                fixture_points_sha256=fixture_points_sha256,
            )
            verification = copy.deepcopy(entry["pre_timing_verification"])
            verification["artifact_semantic_identity_sha256"] = semantic_identity_sha256
            verification["expected"]["artifact_semantic_identity_sha256"] = (
                semantic_identity_sha256
            )
            verification["observed"]["artifact_semantic_identity_sha256"] = (
                semantic_identity_sha256
            )
            entry["pre_timing_verification"] = verification
            process_record = {
                "started_at_utc": "2026-07-24T00:00:00+00:00",
                "finished_at_utc": "2026-07-24T00:00:01+00:00",
                "wall_seconds": 1.0,
                "process_id": 1000 + int(entry["schedule_index"]),
                "operation": "profile",
                "mode": mode,
                "payload_sha256": hashlib.sha256(
                    f"profile-payload-{entry['schedule_index']}".encode()
                ).hexdigest(),
            }
            process_record["content_sha256"] = benchmark._canonical_sha256(
                process_record
            )
            timing_sources = {
                "wall": "runtime_core_repeated_wall_time",
                "evaluator": "diagnostic",
            }
            environment = {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "wall_time_sample_pass": "runtime._benchmark_f64_wall_time",
            }
            worker_measurement = {
                "batch_size": batch_size,
                "sample_count": benchmark.MIN_AUTHORITATIVE_SAMPLES,
                "repetitions_per_sample": 1,
                "evaluation_count": benchmark.MIN_AUTHORITATIVE_SAMPLES,
                "evaluated_point_count": (
                    benchmark.MIN_AUTHORITATIVE_SAMPLES * batch_size
                ),
                "wall_seconds_per_point": wall,
                "inner_native_wall_blocks": raw_blocks,
                "timing_sources": timing_sources,
                "environment": environment,
                "interrupted": False,
            }
            sample = {
                "schedule_index": entry["schedule_index"],
                "round": entry["round"],
                "worker_command": entry["worker_invocation"]["command"],
                "worker_invocation": entry["worker_invocation"],
                "worker_process_record": process_record,
                "pre_timing_verification": verification,
                "lane_contract_sha256": lane_contract_sha256,
                "timing_configuration": {
                    "minimum_internal_samples": (benchmark.MIN_AUTHORITATIVE_SAMPLES),
                    "warmup_runs": 2,
                    "target_runtime_seconds": 5.0,
                },
                "worker_measurement": worker_measurement,
                "internal_sample_count": benchmark.MIN_AUTHORITATIVE_SAMPLES,
                "repetitions_per_sample": 1,
                "evaluation_count": benchmark.MIN_AUTHORITATIVE_SAMPLES,
                "evaluated_point_count": (
                    benchmark.MIN_AUTHORITATIVE_SAMPLES * batch_size
                ),
                "wall_seconds_per_point": wall,
                "inner_native_wall_blocks": raw_blocks,
                "timing_sources": timing_sources,
                "environment": environment,
                "interrupted": False,
            }
            retained_result_record = benchmark._retained_profile_worker_result_record(
                sample,
                upstream_result_record=entry["worker_result_record"],
            )
            sample["worker_result_record"] = retained_result_record
            entry["worker_result_record"] = retained_result_record
            entry["worker_result_sha256"] = retained_result_record[
                "addressed_payload_sha256"
            ]
            samples.append(sample)
        wall_values = [float(sample["wall_seconds_per_point"]) for sample in samples]
        median = statistics.median(wall_values)
        mad = statistics.median(abs(value - median) for value in wall_values)
        measurements.append(
            {
                "batch_size": batch_size,
                "sample_count": len(samples),
                "subprocess_sample_count": len(samples),
                "wall_seconds_per_point": median,
                "wall_seconds_per_point_median": median,
                "wall_seconds_per_point_mad": mad,
                "statistics_contract": "subprocess-median-and-raw-mad-v1",
                "subprocess_samples": samples,
                "interrupted": False,
            }
        )
    return {
        "process_id": "process-1",
        "process_expression": "u u~ > Z g g g g g g",
        "selector_contract": selector_contract,
        "validation": validation_record,
        "mode": mode,
        "artifact_semantic_identity": semantic_identity,
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
        "profiles": measurements,
    }


def _passing_profiles(
    schedule: dict[str, object] | None = None,
    *,
    semantic_identity: dict[str, object] | None = None,
    selector_contract: dict[str, object] | None = None,
    fixture: dict[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    schedule = _passing_schedule() if schedule is None else schedule
    values = (1.0 + 2.0j, 3.0 + 4.0j)
    return {
        "compiled": _profile(
            "compiled",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
        "eager": _profile(
            "eager",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
        "recurrence": _profile(
            "recurrence",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
    }


def _rebind_sample_result(
    schedule: dict[str, object],
    sample: dict[str, Any],
) -> None:
    previous_record = sample["worker_result_record"]
    assert isinstance(previous_record, dict)
    retained_record = benchmark._retained_profile_worker_result_record(
        sample,
        upstream_result_record=previous_record,
    )
    sample["worker_result_record"] = retained_record
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if isinstance(item, dict) and item["schedule_index"] == sample["schedule_index"]
    )
    entry["worker_result_record"] = retained_record
    entry["worker_result_sha256"] = retained_record["addressed_payload_sha256"]


def test_preserved_worker_result_survives_driver_enrichment() -> None:
    command = benchmark._command_identity(("python", "worker"))
    invocation = {
        "started_at_utc": "2026-07-24T00:00:00+00:00",
        "finished_at_utc": "2026-07-24T00:00:03+00:00",
        "wall_seconds": 3.0,
        "command": command,
    }
    invocation["content_sha256"] = benchmark._canonical_sha256(invocation)
    operation_payload = {
        "mode": "compiled",
        "model_source": {"kind": "test"},
    }
    process_record = {
        "started_at_utc": "2026-07-24T00:00:01+00:00",
        "finished_at_utc": "2026-07-24T00:00:02+00:00",
        "wall_seconds": 1.0,
        "process_id": 123,
        "operation": "generate",
        "mode": "compiled",
        "payload_sha256": benchmark._canonical_sha256(operation_payload),
    }
    process_record["content_sha256"] = benchmark._canonical_sha256(process_record)
    payload = {
        **operation_payload,
        "worker_command": command,
        "worker_invocation": invocation,
        "worker_process_record": process_record,
    }
    result_record = {
        "recorded_at_utc": "2026-07-24T00:00:04+00:00",
        "addressed_payload_sha256": benchmark._canonical_sha256(payload),
        "worker_process_record_sha256": process_record["content_sha256"],
        "worker_invocation_sha256": invocation["content_sha256"],
    }
    result_record["content_sha256"] = benchmark._canonical_sha256(result_record)
    returned = {**payload, "worker_result_record": result_record}
    evidence = benchmark._preserved_worker_result_evidence(returned)
    forged = copy.deepcopy(returned)
    forged_process = forged["worker_process_record"]
    forged_process["payload_sha256"] = "b" * 64
    forged_process["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in forged_process.items() if key != "content_sha256"}
    )
    forged_record = forged["worker_result_record"]
    forged_record["worker_process_record_sha256"] = forged_process["content_sha256"]
    forged_payload = {
        key: value for key, value in forged.items() if key != "worker_result_record"
    }
    forged_record["addressed_payload_sha256"] = benchmark._canonical_sha256(
        forged_payload
    )
    forged_record["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in forged_record.items() if key != "content_sha256"}
    )
    with pytest.raises(
        benchmark.HarnessError,
        match="content-address contract",
    ):
        benchmark._preserved_worker_result_evidence(forged)
    returned["artifact_identity"] = {"later": "driver enrichment"}
    assert evidence["payload"] == payload
    assert evidence["worker_result_record"]["addressed_payload_sha256"] == (
        benchmark._canonical_sha256(evidence["payload"])
    )
    assert evidence["content_sha256"] == benchmark._canonical_sha256(
        {key: value for key, value in evidence.items() if key != "content_sha256"}
    )


def test_pairwise_validation_includes_eager_and_every_point() -> None:
    summary = benchmark._pairwise_profile_validation(_passing_profiles())
    comparisons = summary["comparisons"]
    assert set(comparisons) == {
        "compiled__eager",
        "compiled__recurrence",
        "eager__recurrence",
    }
    assert len(comparisons["compiled__eager"]["point_comparisons"]) == 2
    assert summary["selectors_match"]
    assert summary["fixtures_match"]
    assert summary["lane_validation_passes"]
    assert summary["pairwise_validation_passes"]
    assert summary["passes"]


def test_pairwise_validation_fails_when_only_a_later_eager_point_differs() -> None:
    profiles = _passing_profiles()
    profiles["eager"] = _profile("eager", (1.0 + 2.0j, 3.1 + 4.0j))
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["comparisons"]["compiled__eager"]
    assert comparison["point_comparisons"][0]["passes"]
    assert not comparison["point_comparisons"][1]["passes"]
    assert not summary["pairwise_validation_passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_resolved_component_difference() -> None:
    profiles = _passing_profiles()
    eager_components = profiles["eager"]["validation"]["resolved_components"]
    assert isinstance(eager_components, list)
    eager_components[0][0] = [0.5, 2.0]
    eager_components[0][1] = [0.5, 0.0]
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["resolved_component_comparisons"]["compiled__eager"]
    assert not comparison["passes"]
    assert not summary["pairwise_validation_passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_resolved_axis_difference() -> None:
    profiles = _passing_profiles()
    profiles["eager"]["validation"]["resolved_helicity_ids"][0] = "h:+1,+1"
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["resolved_component_comparisons"]["compiled__eager"]
    assert not comparison["axes_match"]
    assert not comparison["passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_fixture_mismatch() -> None:
    profiles = _passing_profiles()
    profiles["eager"] = _profile(
        "eager",
        (1.0 + 2.0j, 3.0 + 4.0j),
        fixture=_fixture(points_sha256="d" * 64),
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    assert not summary["fixtures_match"]
    assert not summary["passes"]


@pytest.mark.parametrize("value", ("false", 1))
def test_pairwise_validation_requires_boolean_lane_pass(value: object) -> None:
    profiles = _passing_profiles()
    profiles["compiled"]["validation"]["passes"] = value
    with pytest.raises(benchmark.HarnessError, match="summary is inconsistent"):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_requires_fixture_point_count() -> None:
    profiles = _passing_profiles()
    for profile in profiles.values():
        profile["validation"]["selected_totals"] = [[1.0, 2.0]]
    with pytest.raises(
        benchmark.HarnessError,
        match="inventory disagrees with fixture",
    ):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_rejects_boolean_complex_components() -> None:
    profiles = _passing_profiles()
    for profile in profiles.values():
        profile["validation"]["selected_totals"] = [
            [True, False],
            [True, False],
        ]
    with pytest.raises(
        benchmark.HarnessError,
        match="invalid complex value",
    ):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_recomputes_lane_point_evidence() -> None:
    profiles = _passing_profiles()
    profiles["compiled"]["validation"]["resolved_sums"][0] = [9.0, 9.0]
    with pytest.raises(
        benchmark.HarnessError,
        match="point evidence is inconsistent",
    ):
        benchmark._pairwise_profile_validation(profiles)


def test_partial_lane_capture_never_vacuously_passes() -> None:
    arguments = _arguments("--mode", "recurrence")
    schedule = _passing_schedule(modes=("recurrence",))
    profiles = {
        "recurrence": _profile(
            "recurrence",
            (1.0 + 2.0j, 3.0 + 4.0j),
            schedule=schedule,
        )
    }
    summary = benchmark._pairwise_profile_validation(profiles)
    assert summary["lane_validation_passes"]
    assert summary["pairwise_validation_passes"] is None

    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert capture["passes"] is None
    assert capture["missing_modes"] == ["compiled", "eager"]


def test_complete_three_lane_capture_can_pass_but_m0_remains_fail_closed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"]
    assert capture["passes"]
    assert capture["schema_version"] == 3
    assert schedule["schema_version"] == 2

    manifest = benchmark._milestone0_acceptance_manifest(arguments, capture)
    assert manifest["kind"] == benchmark.M0_ACCEPTANCE_KIND
    assert manifest["schema_version"] == benchmark.M0_ACCEPTANCE_SCHEMA
    assert not manifest["accepted"]
    assert manifest["status"] == "incomplete"
    assert {"kind": "layout_capture", "value": "all-flow-union"} in manifest[
        "missing_evidence"
    ]
    assert {"kind": "external_lane", "value": "amplicol"} in manifest[
        "missing_evidence"
    ]
    assert "separate fail-closed orchestrator" in manifest["integration_step"]


def test_generation_specialized_axes_are_not_authoritative() -> None:
    arguments = _arguments("--specialize-flow-at-generation")
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["recurrence"] = _profile(
        "recurrence",
        (1.0 + 2.0j, 3.0 + 4.0j),
        schedule=schedule,
        specialized_axes=("color_flow",),
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["evidence_complete"]
    assert not capture["authoritative_eligible"]
    assert not capture["complete"]
    assert capture["passes"] is None
    assert capture["generation_specialized_axes_by_mode"] == {
        "recurrence": ["color_flow"]
    }


def test_cross_lane_physical_or_normalization_identity_mismatch_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_identity["normalization"] = {"average_factor": 72, "color_factor": 3}
    eager_identity["normalization_sha256"] = benchmark._canonical_sha256(
        eager_identity["normalization"]
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["authoritative_eligible"]
    assert not capture["artifact_semantic_contract"]["lanes_match"]
    assert not capture["complete"]


def test_profile_semantic_identity_digest_is_required() -> None:
    profiles = _passing_profiles()
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_identity["normalization"]["average_factor"] = 72
    eager_identity["normalization_sha256"] = benchmark._canonical_sha256(
        eager_identity["normalization"]
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert any("not content-addressed" in error for error in contract["errors"])


def test_physics_relevant_axis_coefficient_is_cross_compared() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_axis = eager_identity["physical_helicities"]
    eager_entries = copy.deepcopy(eager_axis["ordered_entries"])
    eager_entries[1]["coefficient"] = 2.0
    eager_axis["ordered_entries"] = eager_entries
    eager_axis["ordered_entries_sha256"] = benchmark._canonical_sha256(eager_entries)
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert contract["lanes_match"] is False


def test_empty_runtime_selector_axes_are_ineligible() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_selector_semantics = copy.deepcopy(
        eager_identity["runtime_selector_semantics"]
    )
    eager_runtime_selectors = copy.deepcopy(eager_identity["runtime_selectors"])
    eager_selector_semantics["axes"] = {}
    eager_runtime_selectors["axes"] = {}
    eager_identity["runtime_selector_semantics"] = eager_selector_semantics
    eager_identity["runtime_selector_semantics_sha256"] = benchmark._canonical_sha256(
        eager_selector_semantics
    )
    eager_identity["runtime_selectors"] = eager_runtime_selectors
    eager_identity["runtime_selectors_sha256"] = benchmark._canonical_sha256(
        eager_runtime_selectors
    )
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert any(
        "model/selector/reduction semantic identity is invalid" in error
        for error in contract["errors"]
    )


def test_empty_profile_selector_workload_contract_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    for profile in profiles.values():
        profile["selector_contract"] = {}
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("color_flow_count", 99),
        ("resolved_color_flow_id", "flow:forged"),
        ("structural_zero_helicity_count", 1),
    ),
)
def test_profile_selector_reconciles_with_semantic_axes(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    profile = _profile("compiled", (1.0 + 2.0j, 3.0 + 4.0j))
    selector_contract = copy.deepcopy(profile["selector_contract"])
    assert isinstance(selector_contract, dict)
    selector_contract[field] = value
    assert (
        benchmark._profile_selector_contract_matches(
            arguments,
            selector_contract,
            profile["artifact_semantic_identity"],
        )
        is False
    )


def test_profile_root_process_must_match_semantic_process() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["eager"]["process_id"] = "wrong"
    profiles["eager"]["process_expression"] = "wrong > process"
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["root_processes_match"] is False


def test_runtime_selector_contract_version_rejects_boolean() -> None:
    identity = _artifact_semantic_identity()
    runtime_selectors = copy.deepcopy(identity["runtime_selectors"])
    assert isinstance(runtime_selectors, dict)
    runtime_selectors["contract_version"] = True
    with pytest.raises(
        benchmark.HarnessError,
        match="runtime-selector semantics are incomplete",
    ):
        benchmark._runtime_selector_semantic_identity(
            runtime_selectors,
            color_coverage="complete",
            helicity_coverage="complete",
            artifact=Path("<test-profile>"),
        )


def test_color_axis_rejects_empty_physics_word() -> None:
    identity = _artifact_semantic_identity()
    color_axis = identity["physical_color_flows"]
    assert isinstance(color_axis, dict)
    entries = copy.deepcopy(color_axis["ordered_entries"])
    entries[0]["word"] = []
    with pytest.raises(
        benchmark.HarnessError,
        match="not complete and ordered",
    ):
        benchmark._ordered_physical_axis(
            entries,
            label="color-flow",
            require_structural_zero=False,
        )


def test_missing_reduction_order_identity_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    del profiles["recurrence"]["artifact_semantic_identity"][
        "reduction_ordering_sha256"
    ]
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["artifact_semantic_contract"]["passes"]
    assert not capture["authoritative_eligible"]
    assert not capture["complete"]


def test_manifest_model_identity_is_cross_compared() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_model_identity = eager_identity["manifest_model_identity"]
    eager_manifest_model = copy.deepcopy(eager_model_identity["manifest"])
    eager_manifest_model["content_sha256"] = "d" * 64
    eager_common_model = {
        key: value
        for key, value in eager_manifest_model.items()
        if key != "source_kind"
    }
    eager_model_identity.update(
        {
            "manifest": eager_manifest_model,
            "manifest_sha256": benchmark._canonical_sha256(eager_manifest_model),
            "common_physics_identity": eager_common_model,
            "common_physics_identity_sha256": benchmark._canonical_sha256(
                eager_common_model
            ),
        }
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is False
    assert capture["artifact_semantic_contract"]["lanes_match"] is False
    assert capture["authoritative_eligible"] is False


def test_structural_zero_entry_mismatch_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_helicities = eager_identity["physical_helicities"]
    eager_entries = copy.deepcopy(eager_helicities["ordered_entries"])
    eager_entries[1]["structural_zero"] = True
    eager_helicities["ordered_entries"] = eager_entries
    eager_helicities["ordered_entries_sha256"] = benchmark._canonical_sha256(
        eager_entries
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is False
    assert capture["authoritative_eligible"] is False


def test_reduction_order_mismatch_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_reduction = copy.deepcopy(eager_identity["reduction_ordering"])
    eager_reduction["ordered_groups"][0]["physical_helicity_ids"].reverse()
    eager_identity["reduction_ordering"] = eager_reduction
    eager_identity["reduction_ordering_sha256"] = benchmark._canonical_sha256(
        eager_reduction
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is False
    assert capture["artifact_semantic_contract"]["lanes_match"] is False
    assert capture["authoritative_eligible"] is False


def test_reduction_members_must_map_to_the_group_representative() -> None:
    identity = _artifact_semantic_identity()
    helicity_axis = copy.deepcopy(identity["physical_helicities"])
    assert isinstance(helicity_axis, dict)
    helicity_entries = helicity_axis["ordered_entries"]
    assert isinstance(helicity_entries, list)
    assert isinstance(helicity_entries[1], dict)
    helicity_entries[1]["computed"] = True
    helicity_entries[1]["representative_id"] = "h:+1,-1"
    helicity_axis["ordered_entries_sha256"] = benchmark._canonical_sha256(
        helicity_entries
    )
    reduction = identity["reduction_ordering"]
    color_axis = identity["physical_color_flows"]
    assert isinstance(reduction, dict)
    assert isinstance(color_axis, dict)
    with pytest.raises(
        benchmark.HarnessError,
        match="not closed over physical axes",
    ):
        benchmark._reduction_ordering_identity(
            {
                "kind": reduction["kind"],
                "groups": reduction["ordered_groups"],
            },
            color_axis=color_axis,
            helicity_axis=helicity_axis,
            artifact=Path("<test-profile>"),
        )


@pytest.mark.parametrize(
    ("computed", "representative_id", "structural_zero", "coefficient"),
    (
        (True, "h:-1,+1", False, 1.0),
        (True, "h:+1,-1", True, 0.0),
        (False, "h:-1,+1", False, 0.0),
    ),
)
def test_helicity_axis_rejects_invalid_mapping_states(
    computed: bool,
    representative_id: str,
    structural_zero: bool,
    coefficient: float,
) -> None:
    helicity_axis = _artifact_semantic_identity()["physical_helicities"]
    assert isinstance(helicity_axis, dict)
    entries = copy.deepcopy(helicity_axis["ordered_entries"])
    assert isinstance(entries, list)
    assert isinstance(entries[1], dict)
    entries[1].update(
        {
            "computed": computed,
            "representative_id": representative_id,
            "structural_zero": structural_zero,
            "coefficient": coefficient,
        }
    )
    with pytest.raises(
        benchmark.HarnessError,
        match="representative mapping is not closed",
    ):
        benchmark._ordered_physical_axis(
            entries,
            label="helicity",
            require_structural_zero=True,
        )


def test_reduction_coverage_is_recomputed_from_physical_axes() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_reduction = copy.deepcopy(eager_identity["reduction_ordering"])
    eager_reduction["ordered_groups"][0]["physical_helicity_ids"].pop()
    eager_identity["reduction_ordering"] = eager_reduction
    eager_identity["reduction_ordering_sha256"] = benchmark._canonical_sha256(
        eager_reduction
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is False
    assert any(
        "model/selector/reduction semantic identity is invalid" in error
        for error in capture["artifact_semantic_contract"]["errors"]
    )
    assert capture["authoritative_eligible"] is False


def test_capture_rejects_missing_or_interrupted_timing_measurements() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["eager"]["profiles"] = profiles["eager"]["profiles"][1:]
    profiles["recurrence"]["profiles"][0]["interrupted"] = True
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert capture["passes"] is None
    contract = capture["measurement_contract"]["lanes"]
    assert contract["eager"]["missing_batch_sizes"] == [1]
    assert not contract["recurrence"]["passes"]


def test_single_worker_five_sample_timing_cannot_pass_authoritative_capture() -> None:
    arguments = _arguments(
        "--subprocess-samples",
        "1",
        "--minimum-samples",
        "5",
    )
    schedule = _passing_schedule(subprocess_samples=1)
    profiles = _passing_profiles(schedule)
    for profile in profiles.values():
        for measurement in profile["profiles"]:
            measurement["subprocess_samples"][0]["internal_sample_count"] = 5
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    contract = capture["measurement_contract"]
    assert not contract["passes"]
    assert contract["configured_subprocess_samples"] == 1
    assert contract["configured_internal_minimum_samples"] == 5


def test_aggregate_headline_sample_count_must_match_inventory() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["compiled"]["profiles"][0]["sample_count"] = False
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    "failure",
    (
        "native_wall",
        "warmup",
        "internal_samples",
        "repetitions",
        "evaluation_count",
        "raw_blocks",
        "worker_result_binding",
        "median_mad",
        "schedule",
        "result_record",
    ),
)
def test_authoritative_timing_contract_fails_closed(
    failure: str,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    measurement = profiles["compiled"]["profiles"][0]
    first_sample = measurement["subprocess_samples"][0]
    if failure == "native_wall":
        first_sample["timing_sources"]["wall"] = "runtime_evaluate_wall_time"
    elif failure == "warmup":
        first_sample["timing_configuration"]["warmup_runs"] = 0
    elif failure == "internal_samples":
        first_sample["internal_sample_count"] = 6
    elif failure == "repetitions":
        first_sample["repetitions_per_sample"] = 0
    elif failure == "evaluation_count":
        first_sample["evaluation_count"] = 0
    elif failure == "raw_blocks":
        first_sample["inner_native_wall_blocks"]["blocks"][0][
            "native_wall_seconds"
        ] *= 2.0
    elif failure == "worker_result_binding":
        raw_blocks = first_sample["inner_native_wall_blocks"]
        block = raw_blocks["blocks"][0]
        block["started_at_utc"] = "2026-07-24T00:00:00.5+00:00"
        block["content_sha256"] = benchmark._canonical_sha256(
            {key: value for key, value in block.items() if key != "content_sha256"}
        )
        raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    elif failure == "median_mad":
        measurement["wall_seconds_per_point_mad"] = None
    elif failure == "schedule":
        entries = schedule["entries"]
        assert isinstance(entries, list)
        del entries[0]["worker_invocation"]
    else:
        entries = schedule["entries"]
        assert isinstance(entries, list)
        entries[0]["worker_result_record"]["recorded_at_utc"] = (
            "2026-07-24T00:00:02+00:00"
        )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert not capture["measurement_contract"]["passes"]


def test_raw_block_boolean_counts_fail_even_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    block = raw_blocks["blocks"][0]
    for field in (
        "repetitions",
        "batch_size",
        "evaluation_count",
        "evaluated_point_count",
    ):
        block[field] = True
    block["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in block.items() if key != "content_sha256"}
    )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repetitions_per_block", True),
        ("evaluation_count", 7.0),
        ("evaluated_point_count", 7.0),
    ),
)
def test_raw_inventory_counts_require_positive_integers_when_readdressed(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks[field] = value
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_inventory_mad_rejects_boolean_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["wall_seconds_per_point_mad"] = False
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_inventory_schema_rejects_boolean_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["schema_version"] = True
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_block_timestamps_must_be_chronological_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    block = raw_blocks["blocks"][0]
    block["started_at_utc"] = "2026-07-24T00:00:02+00:00"
    block["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in block.items() if key != "content_sha256"}
    )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_block_inventory_must_be_chronological_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    first, second = raw_blocks["blocks"][:2]
    first["started_at_utc"] = "2026-07-24T00:00:00.800000+00:00"
    first["finished_at_utc"] = "2026-07-24T00:00:00.900000+00:00"
    second["started_at_utc"] = "2026-07-24T00:00:00.100000+00:00"
    second["finished_at_utc"] = "2026-07-24T00:00:00.200000+00:00"
    for block in (first, second):
        block["content_sha256"] = benchmark._canonical_sha256(
            {key: value for key, value in block.items() if key != "content_sha256"}
        )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_internal_samples", 0),
        ("warmup_runs", True),
        ("target_runtime_seconds", -1.0),
    ),
)
def test_retained_timing_configuration_must_match_cli(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    sample["timing_configuration"][field] = value
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "is not warmed" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    "contradiction",
    (
        "round",
        "measurement_batch",
        "worker_command",
        "verified_at",
        "effective_contract",
    ),
)
def test_retained_sample_contradictions_fail_after_readdressing(
    contradiction: str,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    if contradiction == "round":
        sample["round"] = False
    elif contradiction == "measurement_batch":
        sample["worker_measurement"]["batch_size"] = True
    elif contradiction == "worker_command":
        sample["worker_command"] = benchmark._command_identity(("forged-worker",))
    elif contradiction == "verified_at":
        sample["pre_timing_verification"]["verified_at_utc"] = "not-a-time"
    else:
        sample["pre_timing_verification"]["effective_contract"] = {
            "execution_mode": "eager",
            "backend": "not-jit",
        }
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schedule_index", False),
        ("round", False),
        ("batch_size", True),
    ),
)
def test_profile_schedule_rejects_boolean_slot_coordinates(
    field: str,
    value: object,
) -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entries[0][field] = value
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False
    assert any("invalid slot coordinates" in error for error in contract["errors"])


@pytest.mark.parametrize("record_kind", ("verification", "result"))
def test_profile_schedule_rejects_boolean_nested_schema_versions(
    record_kind: str,
) -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    if record_kind == "verification":
        entry["pre_timing_verification"]["schema_version"] = True
    else:
        result_record = entry["worker_result_record"]
        result_record["schema_version"] = True
        result_record["content_sha256"] = benchmark._canonical_sha256(
            {
                key: value
                for key, value in result_record.items()
                if key != "content_sha256"
            }
        )
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False


def test_profile_schedule_timestamps_must_be_chronological() -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    invocation = entry["worker_invocation"]
    invocation["started_at_utc"] = "2026-07-24T00:00:02+00:00"
    invocation["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in invocation.items() if key != "content_sha256"}
    )
    result_record = entry["worker_result_record"]
    result_record["worker_invocation_sha256"] = invocation["content_sha256"]
    result_record["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in result_record.items() if key != "content_sha256"}
    )
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False
    assert any(
        "inverted invocation timestamps" in error for error in contract["errors"]
    )


def test_profile_schedule_rejects_reordered_noninterleaved_lanes() -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entries[1]["mode"] = entries[0]["mode"]
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert not contract["passes"]
    assert any("interleaved" in error for error in contract["errors"])


@pytest.mark.parametrize("status", (" M tracked.py\n", "?? scratch.py\n"))
def test_source_identity_fails_closed_on_tracked_and_untracked_changes(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    commands: list[tuple[str, ...]] = []
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
            SimpleNamespace(returncode=0, stdout=status),
        )
    )

    def run(command: tuple[str, ...], **_kwargs: object) -> object:
        commands.append(command)
        return next(results)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(benchmark.HarnessError, match="source is dirty"):
        benchmark._git_source_identity()
    assert commands[1] == (
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )


def test_source_identity_accepts_only_a_full_clean_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
            SimpleNamespace(returncode=0, stdout=""),
        )
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )
    identity = benchmark._git_source_identity()
    assert identity["revision"] == "a" * 40
    assert identity["dirty"] is False
    assert identity["untracked_files_checked"] is True


def test_path_state_identity_records_optional_absence(tmp_path: Path) -> None:
    missing = tmp_path / "candidate-Cargo.lock"
    absent = benchmark._path_state_identity(missing)
    assert absent == {
        "present": False,
        "path": str(missing),
        "resolved_path": str(missing.resolve()),
    }

    missing.write_bytes(b"candidate lock")
    present = benchmark._path_state_identity(missing)
    assert present["present"] is True
    assert present["size_bytes"] == len(b"candidate lock")
    assert present["sha256"] == hashlib.sha256(b"candidate lock").hexdigest()


def _build_info(
    checkout: Path,
    *,
    revision: object = "a" * 40,
    digest: object = "b" * 64,
) -> dict[str, object]:
    return {
        "source_checkout": str(checkout),
        "source_revision": revision,
        "native_build_inputs_sha256": digest,
    }


def test_runtime_binding_requires_clean_matching_source_revision(
    tmp_path: Path,
) -> None:
    source = {
        "checkout": str(tmp_path),
        "revision": "a" * 40,
    }
    benchmark._validate_runtime_binding(
        source,
        _build_info(tmp_path),
        native_build_inputs_sha256="b" * 64,
    )

    with pytest.raises(benchmark.HarnessError, match="no clean source revision"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, revision=None),
            native_build_inputs_sha256="b" * 64,
        )
    with pytest.raises(benchmark.HarnessError, match="does not match benchmark HEAD"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, revision="c" * 40),
            native_build_inputs_sha256="b" * 64,
        )
    with pytest.raises(benchmark.HarnessError, match="build inputs"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, digest="c" * 64),
            native_build_inputs_sha256="b" * 64,
        )


def test_runtime_binding_rejects_a_different_checkout(
    tmp_path: Path,
) -> None:
    source_checkout = tmp_path / "source"
    other_checkout = tmp_path / "other"
    source_checkout.mkdir()
    other_checkout.mkdir()
    source = {
        "checkout": str(source_checkout),
        "revision": "a" * 40,
    }
    with pytest.raises(benchmark.HarnessError, match="different source checkout"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(other_checkout),
            native_build_inputs_sha256="b" * 64,
        )


@pytest.mark.parametrize("field", benchmark._PROFILE_EXPECTATION_FIELDS)
def test_profile_worker_rejects_each_identity_mismatch(field: str) -> None:
    expected = {name: "a" * 64 for name in benchmark._PROFILE_EXPECTATION_FIELDS}
    observed = dict(expected)
    observed[field] = "b" * 64
    with pytest.raises(benchmark.HarnessError, match="drifted before timing"):
        benchmark._validate_profile_worker_expectations(expected, observed)


@pytest.mark.parametrize("drift", ("source", "runtime", "artifact"))
def test_driver_post_worker_recheck_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    source = {"revision": "a" * 40, "checkout": str(tmp_path)}
    runtime = {"native_extension": {"sha256": "b" * 64}}
    artifact_identity = {"tree": {"sha256": "c" * 64}}
    reuse_signature = {"semantic_signature_sha256": "d" * 64}
    observed_source = dict(source)
    observed_runtime = dict(runtime)
    observed_artifact = dict(artifact_identity)
    if drift == "source":
        observed_source["revision"] = "e" * 40
    elif drift == "runtime":
        observed_runtime = {"native_extension": {"sha256": "e" * 64}}
    else:
        observed_artifact = {"tree": {"sha256": "e" * 64}}

    monkeypatch.setattr(
        benchmark,
        "_git_source_identity",
        lambda: observed_source,
    )
    monkeypatch.setattr(
        benchmark,
        "_runtime_provenance",
        lambda _source: observed_runtime,
    )
    monkeypatch.setattr(
        benchmark,
        "_require_reusable_artifact",
        lambda *_args, **_kwargs: (observed_artifact, reuse_signature),
    )
    monkeypatch.setattr(
        benchmark,
        "_path_identity",
        lambda _path: {"sha256": "f" * 64},
    )
    baselines = {
        "compiled": {
            "artifact": str(tmp_path / "artifact"),
            "generation_signature": {"kind": "generation"},
            "artifact_identity": artifact_identity,
            "reuse_signature": reuse_signature,
            "reuse_sidecar_sha256": "f" * 64,
        }
    }
    with pytest.raises(benchmark.HarnessError, match="drifted"):
        benchmark._recheck_driver_state(
            source,
            runtime,
            baselines,
            phase="after-test-worker",
        )


def test_explicit_model_identity_is_accurate_and_not_builtin(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "ufo-sm.pyamplicol-model"
    prepared.write_bytes(b"explicit model")
    arguments = _arguments("--prepared-model", str(prepared))
    identity = benchmark._selected_model_identity(
        arguments,
        mode="compiled",
        source_identity={"revision": "a" * 40},
    )
    assert identity["kind"] == "explicit-prepared-model"
    assert identity["resource_id"] is None
    assert identity["file"]["resolved_path"] == str(prepared.resolve())
    assert identity["file"]["sha256"] == hashlib.sha256(b"explicit model").hexdigest()
    benchmark._validate_worker_model_identity(
        identity,
        {
            "kind": "explicit-prepared-model",
            "resource_id": None,
            "compile_excluded_from_generation": True,
            "file": dict(identity["file"]),
        },
    )
    changed_file = dict(identity["file"])
    changed_file["sha256"] = "0" * 64
    with pytest.raises(benchmark.HarnessError, match="disagrees on sha256"):
        benchmark._validate_worker_model_identity(
            identity,
            {
                "kind": "explicit-prepared-model",
                "resource_id": None,
                "compile_excluded_from_generation": True,
                "file": changed_file,
            },
        )


def _write_fake_artifact(path: Path) -> None:
    path.mkdir()
    payload = path / "payload.bin"
    payload.write_bytes(b"payload")
    payload_sha256 = hashlib.sha256(b"payload").hexdigest()
    physics_path = path / "processes" / "process-1" / "physics.json"
    physics_path.parent.mkdir(parents=True)
    physics = {
        "coverage": {
            "color": "complete",
            "helicities": "complete",
        },
        "color_components": [
            {
                "index": 0,
                "id": "flow:2,1",
                "kind": "lc-flow",
                "word": [2, 1],
                "representative_id": "flow:2,1",
                "computed": True,
                "coefficient": 1.0,
            }
        ],
        "helicities": [
            {
                "index": 0,
                "id": "h:-1,+1",
                "values": [-1, 1],
                "representative_id": "h:-1,+1",
                "computed": True,
                "coefficient": 1.0,
                "structural_zero": False,
            },
            {
                "index": 1,
                "id": "h:+1,-1",
                "values": [1, -1],
                "representative_id": "h:-1,+1",
                "computed": False,
                "coefficient": 1.0,
                "structural_zero": False,
            },
        ],
        "extensions": {
            "normalization": {
                "average_factor": 36,
                "color_factor": 3,
                "coupling_policy": "stage-local",
            },
            "runtime_selectors": {
                "kind": "pyamplicol-runtime-selectors",
                "contract_version": 1,
                "generation_specialized_axes": [],
                "axes": {
                    "color_flow": {
                        "generation_coverage": "complete",
                        "generation_selection": [],
                        "runtime_contract": "complete-reusable",
                    },
                    "helicity": {
                        "generation_coverage": "complete",
                        "generation_selection": {},
                        "runtime_contract": "complete-reusable",
                    },
                },
            },
        },
        "reduction": {
            "kind": "lc-diagonal",
            "groups": [
                {
                    "id": "reduction:0",
                    "physical_color_ids": ["flow:2,1"],
                    "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
                    "representative_color_id": "flow:2,1",
                    "representative_helicity_id": "h:-1,+1",
                }
            ],
        },
    }
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    manifest = {
        "artifact_id": "artifact-1",
        "producer": {"name": "test"},
        "model": {
            "name": "built-in-sm",
            "content_sha256": "c" * 64,
            "compiled_schema_version": 9,
            "restriction": None,
            "source_kind": "compiled-model",
        },
        "extensions": {
            "generation": {
                "concrete_processes": [
                    {
                        "id": "process-1",
                        "runtime_schema_sha256": "a" * 64,
                        "reduction_schedule": "manifest-order",
                    }
                ]
            }
        },
        "processes": [
            {
                "id": "process-1",
                "expression": "d d~ > z g",
                "color_accuracy": "lc",
                "physics_path": "processes/process-1/physics.json",
            }
        ],
        "payloads": [
            {
                "path": "payload.bin",
                "size_bytes": len(b"payload"),
                "sha256": payload_sha256,
                "role": "runtime",
                "process_id": "process-1",
            },
            {
                "path": "processes/process-1/physics.json",
                "size_bytes": physics_path.stat().st_size,
                "sha256": hashlib.sha256(physics_path.read_bytes()).hexdigest(),
                "role": "physics",
                "process_id": "process-1",
            },
        ],
    }
    (path / "artifact.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def test_artifact_reuse_requires_exact_signature_and_tree(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    signature = {"kind": "test-signature", "process": "d d~ > z g"}
    identity = benchmark._artifact_identity(artifact)
    benchmark._write_reuse_signature(
        artifact,
        signature=signature,
        artifact_identity=identity,
        generation_command=benchmark._command_identity(("python", "generate-artifact")),
    )
    reused, provenance = benchmark._require_reusable_artifact(
        artifact,
        expected_signature=signature,
    )
    assert reused["tree"] == identity["tree"]
    assert provenance["generation_command"]["argv"] == [
        "python",
        "generate-artifact",
    ]
    assert (
        provenance["semantic_signature"]["artifact_semantic_identity"]
        == identity["semantic_identity"]
    )

    with pytest.raises(benchmark.HarnessError, match="generation request changed"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature={**signature, "process": "g g > g g"},
        )

    (artifact / "extra.bin").write_bytes(b"ambient mutation")
    with pytest.raises(benchmark.HarnessError, match="identity or tree changed"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature=signature,
        )


def test_artifact_reuse_rejects_missing_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    with pytest.raises(benchmark.HarnessError, match="reuse signature is missing"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature={"kind": "test-signature"},
        )


def test_artifact_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.bin"
    payload.write_bytes(b"payload")
    (artifact / "payload-link.bin").symlink_to(payload)
    with pytest.raises(benchmark.HarnessError, match="unsupported symlink"):
        benchmark._tree_identity(artifact)


def test_artifact_semantics_bind_axes_normalization_and_reduction_order(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    original = benchmark._artifact_semantic_identity(artifact, manifest, process)
    assert original["physical_color_flows"]["ordered_ids"] == ["flow:2,1"]
    assert original["physical_helicities"]["ordered_ids"] == [
        "h:-1,+1",
        "h:+1,-1",
    ]

    physics_path = artifact / "processes" / "process-1" / "physics.json"
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["extensions"]["normalization"]["average_factor"] = 72
    physics["reduction"]["groups"][0]["physical_helicity_ids"].reverse()
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    changed = benchmark._artifact_semantic_identity(artifact, manifest, process)
    assert changed["normalization_sha256"] != original["normalization_sha256"]
    assert (
        changed["reduction_ordering_sha256"] != (original["reduction_ordering_sha256"])
    )

    physics["helicities"].reverse()
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    with pytest.raises(benchmark.HarnessError, match="complete and ordered"):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_artifact_semantics_reject_model_free_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    del manifest["model"]
    with pytest.raises(benchmark.HarnessError, match="no model identity"):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_semantic_signature_covers_generation_configuration() -> None:
    source = {"revision": "a" * 40}
    runtime = {"native": {"sha256": "b" * 64}}
    model = {"kind": "built-in-sm-source", "source_revision": "a" * 40}
    default = _arguments()
    changed = _arguments("--point-tile-size", "2048")
    default_signature = benchmark._semantic_generation_signature(
        default,
        mode="recurrence",
        source_identity=source,
        runtime_provenance=runtime,
        model_identity=model,
    )
    changed_signature = benchmark._semantic_generation_signature(
        changed,
        mode="recurrence",
        source_identity=source,
        runtime_provenance=runtime,
        model_identity=model,
    )
    assert default_signature["point_tile_size"] == 1024
    assert changed_signature["point_tile_size"] == 2048
    assert benchmark._canonical_sha256(default_signature) != (
        benchmark._canonical_sha256(changed_signature)
    )


def test_validation_fixture_records_raw_and_semantic_digests(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / "processes" / "process-1"
    process_root.mkdir(parents=True)
    path = process_root / "validation-momenta.json"
    payload = {
        "points": [
            [
                {"momentum": [1.0, 0.0, 0.0, 1.0]},
                {"momentum": [1.0, 0.0, 0.0, -1.0]},
            ],
            [
                {"momentum": [2.0, 0.0, 0.0, 2.0]},
                {"momentum": [2.0, 0.0, 0.0, -2.0]},
            ],
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    points, identity = benchmark._validation_fixture(tmp_path, "process-1")
    assert len(points) == 2
    assert identity["point_count"] == 2
    assert identity["file"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["points_sha256"] == benchmark._canonical_sha256(points)


def test_profile_worker_evaluates_every_validation_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    process_root = artifact / "processes" / "process-1"
    process_root.mkdir(parents=True)
    points = [
        [
            {"momentum": [float(index + 1), 0.0, 0.0, 1.0]},
            {"momentum": [float(index + 1), 0.0, 0.0, -1.0]},
        ]
        for index in range(3)
    ]
    (process_root / "validation-momenta.json").write_text(
        json.dumps({"points": points}),
        encoding="utf-8",
    )
    calls: list[tuple[str, int]] = []
    benchmark_point_sets: list[object] = []
    events: list[str] = []

    def verify(_arguments: object) -> dict[str, object]:
        events.append("verify")
        return {
            "kind": benchmark.WORKER_VERIFICATION_KIND,
            "schema_version": benchmark.WORKER_VERIFICATION_SCHEMA,
            "artifact_semantic_identity": _artifact_semantic_identity(),
            "artifact_semantic_identity_sha256": "a" * 64,
        }

    monkeypatch.setattr(
        benchmark,
        "_verify_profile_worker_environment",
        verify,
    )

    class FakeResolved:
        def __init__(self, values: tuple[complex, ...]) -> None:
            self._values = values
            self.values = tuple(((value,),) for value in values)
            self.helicity_ids = ("h:-1,+1,-1,+1",)
            self.color_ids = ("flow:2,1",)

        def total(self) -> tuple[complex, ...]:
            return self._values

    class FakeRuntime:
        physics = SimpleNamespace(
            process="d d~ > z g",
            process_id="process-1",
            color_accuracy="lc",
            color_flows=(SimpleNamespace(id="flow:2,1"),),
            helicities=(SimpleNamespace(id="h:-1,+1,-1,+1"),),
            structural_zero_helicity_count=0,
        )

        def __init__(self) -> None:
            self._backend = SimpleNamespace(
                _benchmark_f64_wall_time=lambda batch, repetitions, **_selectors: (
                    1.0e-6 * len(batch) * repetitions
                )
            )

        def evaluate(
            self,
            batch: tuple[object, ...],
            **_selectors: object,
        ) -> tuple[complex, ...]:
            calls.append(("evaluate", len(batch)))
            return tuple(complex(index + 1) for index in range(len(batch)))

        def evaluate_resolved(
            self,
            batch: tuple[object, ...],
            **_selectors: object,
        ) -> FakeResolved:
            calls.append(("evaluate_resolved", len(batch)))
            return FakeResolved(
                tuple(complex(index + 1) for index in range(len(batch)))
            )

    class FakeRuntimeFactory:
        @staticmethod
        def load(_artifact: Path, *, process: str) -> FakeRuntime:
            events.append("load")
            assert process == "d d~ > z g"
            return FakeRuntime()

    class FakeBenchmarkConfig:
        batch_size: int

        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)
            batch_size = values["batch_size"]
            assert isinstance(batch_size, int)
            self.batch_size = batch_size

    class FakeBenchmarkRunner:
        def __init__(self, config: FakeBenchmarkConfig) -> None:
            self.config = config

        def run(self, _runtime: FakeRuntime, *, points: object) -> object:
            benchmark_point_sets.append(points)
            uncertainty = SimpleNamespace(
                standard_deviation=0.0,
                standard_error=0.0,
                relative_standard_error=0.0,
            )
            return SimpleNamespace(
                effective_config=SimpleNamespace(batch_size=self.config.batch_size),
                sample_count=7,
                repetitions_per_sample=1,
                evaluation_count=7,
                evaluated_point_count=7 * self.config.batch_size,
                wall_time_per_point=1.0e-6,
                evaluator_time_per_point=1.0e-6,
                uncertainty=uncertainty,
                evaluator_uncertainty=uncertainty,
                environment={
                    "wall_time_source": "runtime_core_repeated_wall_time",
                    "wall_time_sample_pass": "runtime._benchmark_f64_wall_time",
                },
                interrupted=False,
            )

    pyamplicol = ModuleType("pyamplicol")
    pyamplicol.BenchmarkRunner = FakeBenchmarkRunner  # type: ignore[attr-defined]
    pyamplicol.Runtime = FakeRuntimeFactory  # type: ignore[attr-defined]
    config = ModuleType("pyamplicol.config")
    config.BenchmarkConfig = FakeBenchmarkConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyamplicol", pyamplicol)
    monkeypatch.setitem(sys.modules, "pyamplicol.config", config)

    arguments = SimpleNamespace(
        artifact=artifact,
        process_expression="d d~ > z g",
        gluon_count=1,
        mode="compiled",
        color_flow="1",
        helicity="1",
        lc_flow_layout="topology-replay",
        validation_point_artifact=artifact,
        batch_size=[1],
        schedule_index=0,
        schedule_round=0,
        target_runtime=0.1,
        minimum_samples=7,
        warmup_runs=2,
        validation_samples=3,
    )
    result = benchmark._profile_worker(arguments)
    assert events[:2] == ["verify", "load"]
    assert calls[:2] == [("evaluate", 3), ("evaluate_resolved", 3)]
    assert len(benchmark_point_sets) == 1
    assert (
        benchmark._canonical_sha256(benchmark_point_sets[0])
        == (result["validation"]["fixture"]["points_sha256"])
    )
    assert result["validation"]["fixture"]["point_count"] == 3
    assert len(result["validation"]["selected_totals"]) == 3
    assert len(result["validation"]["resolved_components"]) == 3
    assert len(result["validation"]["point_comparisons"]) == 3
    raw_blocks = result["profiles"][0]["inner_native_wall_blocks"]
    assert (
        raw_blocks["fixture_points_sha256"]
        == (result["validation"]["fixture"]["points_sha256"])
    )
    assert len(raw_blocks["blocks"]) == benchmark.MIN_AUTHORITATIVE_SAMPLES
