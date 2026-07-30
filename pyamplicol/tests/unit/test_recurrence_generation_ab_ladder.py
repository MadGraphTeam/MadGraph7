# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer import recurrence_generation_ab_ladder as ladder


def test_process_family_and_approved_timeout_policy() -> None:
    assert ladder.process_expression(1) == "d d~ > Z"
    assert ladder.process_expression(4) == "d d~ > Z g g g"
    assert ladder.generation_timeout_seconds(2, "topology-replay") == 300.0
    assert ladder.generation_timeout_seconds(7, "all-flow-union") == 900.0
    assert ladder.generation_timeout_seconds(8, "topology-replay") == 3600.0
    assert ladder.generation_timeout_seconds(8, "all-flow-union") == 7200.0
    assert ladder.generation_timeout_seconds(9, "topology-replay") == 7200.0
    assert ladder.generation_timeout_seconds(9, "all-flow-union") == 21600.0
    with pytest.raises(ladder.LadderError):
        ladder.generation_timeout_seconds(10, "topology-replay")


def test_schedule_alternates_pair_order_and_selects_runtime_cells() -> None:
    schedule = ladder.build_schedule(
        (2, 6),
        ("topology-replay",),
        2,
        frozenset({6}),
    )

    assert [sample.variant_name for sample in schedule] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert [sample.pair_index for sample in schedule] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert all(not sample.runtime_enabled for sample in schedule[:4])
    assert all(sample.runtime_enabled for sample in schedule[4:])


def _spec(*, multiplicity: int, runtime_enabled: bool) -> ladder.SampleSpec:
    return ladder.SampleSpec(
        sequence_index=0,
        pair_index=0,
        order_in_pair=0,
        repetition=0,
        multiplicity=multiplicity,
        layout="all-flow-union",
        variant_name="baseline",
        runtime_enabled=runtime_enabled,
    )


def test_command_wraps_each_capture_and_runtime_workers_in_watchdog(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    variant = ladder.Variant(
        name="baseline",
        checkout=checkout,
        python=tmp_path / "python",
        pythonpath=tmp_path / "site-packages",
        prepared_model=tmp_path / "model.pack",
    )
    settings = ladder.RunnerSettings()
    sample_root = tmp_path / "sample"

    generation_command = ladder.build_sample_command(
        _spec(multiplicity=2, runtime_enabled=False),
        variant,
        sample_root,
        settings,
    )
    assert generation_command[:6] == [
        str(variant.python),
        str(checkout / ladder.WATCHDOG_RELATIVE_PATH),
        "--limit-gib",
        "30",
        "--",
        str(variant.python),
    ]
    assert str(checkout / ladder.HARNESS_RELATIVE_PATH) in generation_command
    assert generation_command.count("--mode") == 1
    assert "recurrence" in generation_command
    assert "--generation-only" in generation_command
    assert "--allow-diagnostic-incomplete-success" not in generation_command
    assert generation_command[generation_command.index("--process-expression") + 1] == (
        "d d~ > Z g"
    )
    validation_index = generation_command.index("--validation-samples")
    assert generation_command[validation_index : validation_index + 4] == [
        "--validation-samples",
        "10",
        "--point-tile-size",
        "1024",
    ]

    runtime_command = ladder.build_sample_command(
        _spec(multiplicity=6, runtime_enabled=True),
        variant,
        sample_root,
        settings,
    )
    assert "--generation-only" not in runtime_command
    assert [
        runtime_command[index + 1]
        for index, value in enumerate(runtime_command)
        if value == "--batch-size"
    ] == ["1", "128", "1024"]
    assert runtime_command[runtime_command.index("--subprocess-samples") + 1] == "7"
    assert runtime_command[runtime_command.index("--warmup-runs") + 1] == "2"
    assert runtime_command[runtime_command.index("--target-runtime") + 1] == "5"

    scouting_command = ladder.build_sample_command(
        _spec(multiplicity=9, runtime_enabled=False),
        variant,
        sample_root,
        ladder.RunnerSettings(allow_diagnostic_incomplete_success=True),
    )
    assert "--allow-diagnostic-incomplete-success" in scouting_command


def test_incomplete_diagnostic_capture_is_never_classified_as_passed() -> None:
    outcome = ladder.ProcessOutcome(
        exit_code=0,
        timed_out=False,
        wall_seconds=1.0,
        error=None,
    )
    watchdog = {"terminal_record": "command-finished"}
    incomplete = {"complete": None, "passes": None}

    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=incomplete,
            allow_diagnostic_incomplete_success=False,
        )
        == "failed-validation"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=incomplete,
            allow_diagnostic_incomplete_success=True,
        )
        == "censored"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary={"complete": True, "passes": True},
            allow_diagnostic_incomplete_success=False,
        )
        == "passed"
    )


def test_generation_timeout_is_censored_only_in_explicit_scouting_mode(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "recurrence-z6g-benchmark: recurrence generation worker exceeded "
        "21600 seconds\n"
        "memory-watchdog: command finished exit=2 peak_rss=1.250 GiB "
        "peak_physical_footprint=1.500 GiB peak_guard=1.500 GiB "
        "peak_processes=4\n",
        encoding="utf-8",
    )
    timeout = ladder.parse_harness_generation_timeout(stderr)
    assert timeout == {"configured_seconds": 21600.0}
    outcome = ladder.ProcessOutcome(
        exit_code=2,
        timed_out=False,
        wall_seconds=21600.0,
        error=None,
    )
    watchdog = {"terminal_record": "command-finished"}

    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=None,
            harness_generation_timeout=timeout,
            allow_diagnostic_incomplete_success=False,
        )
        == "timeout"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=None,
            harness_generation_timeout=timeout,
            allow_diagnostic_incomplete_success=True,
        )
        == "censored"
    )


def test_scouting_does_not_censor_arbitrary_or_memory_limit_failures(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "recurrence-z6g-benchmark: recurrence generation failed\n",
        encoding="utf-8",
    )
    assert ladder.parse_harness_generation_timeout(stderr) is None
    outcome = ladder.ProcessOutcome(
        exit_code=2,
        timed_out=False,
        wall_seconds=1.0,
        error=None,
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog={"terminal_record": "memory-limit-exceeded"},
            result_error=None,
            harness_summary=None,
            allow_diagnostic_incomplete_success=True,
        )
        == "failed"
    )


def test_watchdog_terminal_records_are_parsed() -> None:
    finished = ladder.parse_watchdog_text(
        "memory-watchdog: command finished exit=0"
        " peak_rss=1.250 GiB"
        " peak_physical_footprint=1.500 GiB"
        " peak_guard=1.500 GiB peak_processes=4\n"
    )
    assert finished["terminal_record"] == "command-finished"
    assert finished["child_exit_code"] == 0
    assert finished["peak_rss"]["bytes_rounded_from_watchdog"] == round(1.25 * 1024**3)
    assert finished["peak_guard"]["gib"] == 1.5

    exceeded = ladder.parse_watchdog_text(
        "memory-watchdog: memory limit exceeded"
        " reason=process-tree-rss-limit"
        " observed=30.125 GiB limit=30.000 GiB"
        " rss=30.125 GiB physical_footprint=unavailable"
        " processes=9; terminating tree\n"
    )
    assert exceeded["terminal_record"] == "memory-limit-exceeded"
    assert exceeded["limit_exceeded"] is True
    assert exceeded["reason"] == "process-tree-rss-limit"
    assert exceeded["process_count_at_limit"] == 9


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_harness_result_parser_extracts_phase_and_runtime_telemetry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    execution = artifact / "processes" / "process" / "execution.json"
    _write_json(
        execution,
        {
            "kind": ladder.EXECUTION_KIND,
            "schema_version": ladder.EXECUTION_SCHEMA_VERSION,
            "plan": {
                "inspection_summary": {
                    "generation_timings_seconds": {
                        "python_extraction": 1.0,
                        "direct_lowering": 2.0,
                        "native_total": 3.5,
                    },
                    "schedule": {"current_count": 11},
                    "schedule_digest": "a" * 64,
                }
            },
        },
    )
    result_path = tmp_path / "result.json"
    _write_json(
        result_path,
        {
            "kind": ladder.HARNESS_KIND,
            "schema_version": ladder.HARNESS_SCHEMA_VERSION,
            "complete": None,
            "passes": None,
            "process": "d d~ > Z g g g g g",
            "configuration": {"lc_flow_layout": "all-flow-union"},
            "source": {"git_revision": "1" * 40},
            "runtime_provenance": {"extension_sha256": "2" * 64},
            "provenance": {"wall_seconds": 12.0},
            "generation": {
                "recurrence": {
                    "artifact": str(artifact),
                    "artifact_identity": {"artifact_id": "3" * 64},
                    "artifact_semantic_identity_sha256": "4" * 64,
                    "generation_wall_seconds": 9.0,
                    "worker_process_record": {"wall_seconds": 8.0},
                    "peak_rss": {"observed_lower_bound_bytes": 1234},
                    "phase_timings_seconds": {
                        "model-loading": 1.5,
                        "recurrence-construction": 6.0,
                    },
                    "phase_total_seconds": 7.5,
                }
            },
            "profiles": {
                "recurrence": {
                    "process_id": "process",
                    "process_expression": "d d~ > Z g g g g g",
                    "profiles": [
                        {
                            "batch_size": 128,
                            "sample_count": 1,
                            "wall_seconds_per_point": 1.0e-6,
                            "wall_seconds_per_point_median": 1.0e-6,
                            "wall_seconds_per_point_mad": 1.0e-8,
                            "interrupted": False,
                            "subprocess_samples": [
                                {
                                    "schedule_index": 2,
                                    "round": 0,
                                    "wall_seconds_per_point": 1.0e-6,
                                    "internal_sample_count": 7,
                                    "repetitions_per_sample": 4,
                                    "evaluation_count": 28,
                                    "evaluated_point_count": 3584,
                                    "interrupted": False,
                                    "worker_process_record": {"wall_seconds": 5.0},
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    summary = ladder.parse_harness_result(result_path)

    assert summary["generation_wall_seconds"] == 9.0
    assert summary["generation_peak_rss"]["observed_lower_bound_bytes"] == 1234
    assert summary["native_generation_timings_seconds"] == {
        "python_extraction": 1.0,
        "direct_lowering": 2.0,
        "native_total": 3.5,
    }
    assert summary["native_inspection_summary"]["schedule"]["current_count"] == 11
    measurement = summary["runtime_profile"]["measurements"][0]
    assert measurement["batch_size"] == 128
    assert measurement["subprocess_samples"][0]["worker_wall_seconds"] == 5.0


def test_runtime_n_must_be_selected_in_generation_ladder() -> None:
    with pytest.raises(ladder.LadderError, match="runtime-n"):
        ladder.build_schedule(
            (2,),
            ("topology-replay",),
            1,
            frozenset({6}),
        )
