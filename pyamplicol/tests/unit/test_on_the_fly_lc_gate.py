# SPDX-License-Identifier: 0BSD
"""Gate-level contracts for the developer on-the-fly LC harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import on_the_fly_lc_gate as gate


def test_one_topology_artifact_has_two_dense_production_authority_workloads() -> None:
    config = gate._config()
    assert config.color.accuracy == "lc"
    assert config.color.lc_flow_layout == "topology-replay"
    assert config.evaluator.execution_mode == "recurrence"
    assert config.evaluator.backend == "jit"
    assert config.evaluator.jit.optimization_level == 2
    assert config.generation.relation_discovery.mode == "off"

    flow = SimpleNamespace(id=gate.FLOW_ID, word=gate.FLOW_WORD, index=7)
    helicity = SimpleNamespace(
        id=gate.HELICITY_ID,
        values=gate.HELICITY_VALUES,
        structural_zero=False,
    )
    assert gate._selectors(
        SimpleNamespace(color_flows=(flow,), helicities=(helicity,))
    ) == (flow, helicity)
    assert gate._query(flow, helicity).flow_index == 7

    authority = gate._dense_authority(SimpleNamespace(artifact_id="a" * 64), 8)
    assert authority["authority_kind"] == "validated_production_pyamplicol"
    assert authority["runtime_api"] == "Runtime.evaluate_resolved"
    assert authority["certifies"] == (
        "selected_flow_helicity_sum",
        "all_flow_single_helicity",
    )
    assert gate._sum(((1.0, 2.0), (3.0, 4.0)), 2) == (4.0, 6.0)
    assert gate._series((0.0,), (0.0,), "zero")["worst"]["status"] == "ok"
    with pytest.raises(gate.GateError, match="disagrees"):
        gate._series((1.0e-300,), (0.0,), "no absolute floor")


def test_amplicol_anchor_requires_exact_cell_and_point_digest(tmp_path: Path) -> None:
    def write(name: str, cell_id: str) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "matrix_element": 3.0,
                    "selector_contract": {
                        "selected_color_flow_ids": [gate.FLOW_ID],
                        "selected_color_words": [list(gate.FLOW_WORD)],
                    },
                    "validation": {
                        "lc_common_component": {
                            "cell_id": cell_id,
                            "point_digest": "same-point",
                            "helicity_ids": [gate.HELICITY_ID],
                            "color_flow_ids": [gate.FLOW_ID],
                            "value": 2.0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    wrong_cell = gate._anchor(write("wrong-cell.json", "another-cell"))
    contextual = gate._anchor_checks(
        wrong_cell,
        "same-point",
        public_component=99.0,
        hidden_component=98.0,
        public_sum=97.0,
        hidden_sum=96.0,
    )
    assert contextual["comparison_performed"] is False
    assert "cell identity differs" in str(contextual["reason"])

    exact = gate._anchor(write("exact.json", gate.AMPICOL_CELL_ID))
    different_point = gate._anchor_checks(
        exact,
        "different-point",
        public_component=99.0,
        hidden_component=98.0,
        public_sum=97.0,
        hidden_sum=96.0,
    )
    assert different_point["comparison_performed"] is False
    assert "point digest differs" in str(different_point["reason"])

    compared = gate._anchor_checks(
        exact,
        "same-point",
        public_component=2.0,
        hidden_component=2.0,
        public_sum=3.0,
        hidden_sum=3.0,
    )
    assert compared["comparison_performed"] is True


def test_hidden_timing_contract_counts_lookup_fill_execute_and_no_poison() -> None:
    def report(repetitions: int) -> dict[str, object]:
        benchmark = repetitions > 0
        cycles = gate.WARMUPS + repetitions if benchmark else 1
        elapsed = 0.25 if benchmark else None
        return {
            "process_id": gate.PROCESS_ID,
            "point_count": 2,
            "work_census_basis": gate.WORK_CENSUS_BASIS,
            "logical_current_count": 5,
            "resident_current_count": 5,
            "resident_current_component_count": 8,
            "source_operation_count": 2,
            "contribution_operation_count": 3,
            "finalization_operation_count": 1,
            "closure_operation_count": 1,
            "total_kernel_application_count": 7,
            "semantic_executor_binding_count": 4,
            "distinct_prepared_executor_count": 3,
            "trace_build_count": 1,
            "trace_cache_hit_count": cycles if benchmark else 0,
            "momentum_fill_count": cycles,
            "currents": [],
            "direct_plan_load_attempts": 0,
            "direct_plan_decode_attempts": 0,
            "direct_plan_materialization_attempts": 0,
            "established_builder_attempts": 0,
            "normalized_values": [1.0, 2.0],
            "benchmark_elapsed_seconds": elapsed,
            "benchmark_seconds_per_point": (
                None if elapsed is None else elapsed / (repetitions * 2)
            ),
        }

    assert gate._probe_values(report(0), 2) == (1.0, 2.0)
    assert gate._probe_values(report(5), 2, 5) == (1.0, 2.0)
    assert gate._work_census(report(5)) == {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "logical_current_count": 5,
        "resident_current_count": 5,
        "resident_current_component_count": 8,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
        "semantic_executor_binding_count": 4,
        "distinct_prepared_executor_count": 3,
    }
    assert gate._calibrate(0.25, 1.0) == 4

    poisoned = report(0)
    poisoned["direct_plan_load_attempts"] = 1
    with pytest.raises(gate.GateError, match="poison"):
        gate._probe_values(poisoned, 2)
    wrong_fill = report(5)
    wrong_fill["momentum_fill_count"] = 6
    with pytest.raises(gate.GateError, match="contract"):
        gate._probe_values(wrong_fill, 2, 5)
    inconsistent_work = report(0)
    inconsistent_work["total_kernel_application_count"] = 8
    with pytest.raises(gate.GateError, match="kernel and operation"):
        gate._probe_values(inconsistent_work, 2)


def test_workload_census_sums_operations_and_keeps_recurrence_calls_distinct() -> None:
    first = {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
    }
    second = {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "source_operation_count": 2,
        "contribution_operation_count": 5,
        "finalization_operation_count": 2,
        "closure_operation_count": 1,
        "total_kernel_application_count": 10,
    }
    assert gate._workload_operation_census(
        ({"work_census": first}, {"work_census": second})
    ) == {
        "aggregation_basis": "sum-one-execution-per-serialized-query-v1",
        "query_census_basis": gate.WORK_CENSUS_BASIS,
        "query_count": 2,
        "source_operation_count": 4,
        "contribution_operation_count": 8,
        "finalization_operation_count": 3,
        "closure_operation_count": 2,
        "total_kernel_application_count": 17,
    }

    counters = SimpleNamespace(
        normalization="mean_per_profiled_point_or_runtime_call_v1",
        recurrence_source_calls_per_call=2.0,
        recurrence_source_rows_per_call=4.0,
        recurrence_contribution_calls_per_call=3.0,
        recurrence_contribution_rows_per_call=8.0,
        recurrence_finalization_calls_per_call=1.0,
        recurrence_finalization_rows_per_call=3.0,
        recurrence_closure_calls_per_call=1.0,
        recurrence_closure_rows_per_call=2.0,
    )
    established = gate._public_recurrence_work(
        SimpleNamespace(timing_breakdown=SimpleNamespace(counters=counters))
    )
    assert established is not None
    assert established["source_calls_per_runtime_call"] == 2.0
    assert established["source_rows_per_runtime_call"] == 4.0
    assert "grouped prepared-backend invocations" in str(established["semantics"])
    assert gate._public_recurrence_work(SimpleNamespace(timing_breakdown=None)) is None


def test_query_family_census_matches_query_local_work_and_retains_destinations(
) -> None:
    observed: dict[str, object] = {}

    def probe(*args: object, **kwargs: object) -> dict[str, int]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "query_count": 2,
            "source_frame_partition_count": 1,
            "projection_applied_query_count": 2,
            "projection_pre_current_count": 12,
            "projection_pre_contribution_count": 10,
            "projection_pre_closure_count": 2,
            "projection_post_current_count": 10,
            "projection_post_contribution_count": 8,
            "projection_post_closure_count": 2,
            "dynamic_current_occurrence_count": 10,
            "dynamic_current_component_occurrence_count": 16,
            "dynamic_source_rows": 4,
            "dynamic_contribution_rows": 8,
            "dynamic_finalization_rows": 3,
            "dynamic_closure_rows": 2,
            "dynamic_source_calls": 4,
            "dynamic_contribution_calls": 8,
            "dynamic_finalization_calls": 3,
            "dynamic_closure_calls": 2,
            "union_unique_current_count": 6,
            "union_unique_current_component_count": 10,
            "union_source_rows": 2,
            "union_contribution_rows": 5,
            "union_finalization_rows": 2,
            "union_closure_rows": 2,
            "union_amplitude_destination_count": 2,
            "union_source_executor_call_groups": 2,
            "union_contribution_executor_call_groups": 3,
            "union_finalization_executor_call_groups": 1,
            "union_closure_executor_call_groups": 1,
        }

    retained = gate.RetainedInputs("builder", "template", b"{}", "a" * 64)
    queries = (
        gate.Query("flow:0", 0, "h:0", (1, -1)),
        gate.Query("flow:1", 1, "h:1", (-1, 1)),
    )
    hidden = {
        "queries": [
            {
                "work_census": {
                    "logical_current_count": 5,
                    "resident_current_component_count": 8,
                }
            },
            {
                "work_census": {
                    "logical_current_count": 5,
                    "resident_current_component_count": 8,
                }
            },
        ],
        "workload_operation_census": {
            "source_operation_count": 4,
            "contribution_operation_count": 8,
            "finalization_operation_count": 3,
            "closure_operation_count": 2,
        },
    }
    result = gate._query_family_census(probe, retained, queries, hidden, True)
    assert result["basis"] == "exact-current-core-key-query-family-union-v1"
    assert result["union_unique_current_count"] == 6
    assert result["union_amplitude_destination_count"] == 2
    assert observed["args"][4] == [0, 1]
    assert observed["args"][5] == [[1, -1], [-1, 1]]
    assert observed["kwargs"] == {"enable_color_projection": True}

    def stale(*_args: object, **_kwargs: object) -> dict[str, int]:
        value = probe(*_args, **_kwargs)
        value["dynamic_contribution_rows"] = 7
        return value

    with pytest.raises(gate.GateError, match="independently timed"):
        gate._query_family_census(stale, retained, queries, hidden, True)


def test_established_recurrence_schedule_census_reads_generated_dag(
    tmp_path: Path,
) -> None:
    process = tmp_path / "processes" / gate.PROCESS_ID
    process.mkdir(parents=True)
    payload = {
        "recurrence_summary": {
            "current_count": 7,
            "contribution_count": 5,
            "closure_term_count": 2,
        },
        "plan": {
            "inspection_summary": {
                "schedule": {
                    "source_row_count": 2,
                    "current_count": 7,
                    "contribution_count": 5,
                    "finalization_count": 3,
                    "closure_term_count": 2,
                    "amplitude_destination_count": 2,
                },
                "direct_arena": {"semantic_component_count": 11},
            }
        },
    }
    (process / "execution.json").write_text(json.dumps(payload), encoding="utf-8")
    result = gate._established_recurrence_schedule_census(tmp_path)
    assert result == {
        "basis": "persisted-recurrence-direct-schedule-v2",
        "semantics": (
            "currents and interaction rows in the generated recurrence DAG; "
            "runtime selector counters remain workload-specific"
        ),
        "source_row_count": 2,
        "current_count": 7,
        "contribution_count": 5,
        "finalization_count": 3,
        "closure_term_count": 2,
        "amplitude_destination_count": 2,
        "semantic_current_component_count": 11,
        "total_kernel_application_count": 12,
    }

    payload["recurrence_summary"]["current_count"] = 8
    (process / "execution.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.GateError, match="summaries disagree"):
        gate._established_recurrence_schedule_census(tmp_path)


def test_hidden_timing_serializes_one_query_census_and_workload_sum() -> None:
    def probe(*args: object, **kwargs: object) -> dict[str, object]:
        point_count = int(args[9])
        repetitions = int(kwargs["benchmark_repetitions"])
        cycles = gate.WARMUPS + repetitions
        elapsed = repetitions * point_count * 0.01
        return {
            "process_id": gate.PROCESS_ID,
            "point_count": point_count,
            "work_census_basis": gate.WORK_CENSUS_BASIS,
            "logical_current_count": 5,
            "resident_current_count": 5,
            "resident_current_component_count": 8,
            "source_operation_count": 2,
            "contribution_operation_count": 3,
            "finalization_operation_count": 1,
            "closure_operation_count": 1,
            "total_kernel_application_count": 7,
            "semantic_executor_binding_count": 4,
            "distinct_prepared_executor_count": 3,
            "trace_build_count": 1,
            "trace_cache_hit_count": cycles,
            "momentum_fill_count": cycles,
            "currents": [],
            "direct_plan_load_attempts": 0,
            "direct_plan_decode_attempts": 0,
            "direct_plan_materialization_attempts": 0,
            "established_builder_attempts": 0,
            "normalized_values": [1.0] * point_count,
            "benchmark_elapsed_seconds": elapsed,
            "benchmark_seconds_per_point": 0.01,
            "trace_digest": "a" * 64,
        }

    retained = gate.RetainedInputs(object(), object(), b"{}", "a" * 64)
    query = gate.Query("flow", 0, "helicity", (1, -1))
    result = gate._hidden_timing(
        probe,
        Path("artifact"),
        retained,
        (query,),
        (((1.0,),), ((2.0,),)),
        target=0.04,
    )
    row = result["queries"][0]
    assert row["work_census"]["logical_current_count"] == 5
    assert result["workload_operation_census"] == {
        "aggregation_basis": "sum-one-execution-per-serialized-query-v1",
        "query_census_basis": gate.WORK_CENSUS_BASIS,
        "query_count": 1,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
    }


def test_cli_launches_one_worker_with_cross_platform_30_gib_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_model = tmp_path / "built-in-sm.pyamplicol-model"
    prepared_model.write_bytes(b"prepared")
    arguments = gate._parser().parse_args(
        [
            "--output",
            "out",
            "--prepared-model",
            str(prepared_model),
            "--amplicol-result",
            "legacy.json",
            "--target-runtime",
            "2",
            "--batch-size",
            "64",
        ]
    )
    assert "--worker" not in gate._parser().format_help()
    command = gate._worker_command(arguments, Path("/tmp/gate"))
    assert command.count("--worker") == 1
    assert "all-flow-union" not in command
    assert command[command.index("--prepared-model") + 1] == str(
        prepared_model.resolve()
    )
    assert command[command.index("--amplicol-result") + 1] == str(
        Path("legacy.json").resolve()
    )
    assert arguments.bypass_color_projection is False
    assert "--bypass-color-projection" not in command
    bypass = arguments
    bypass.bypass_color_projection = True
    assert "--bypass-color-projection" in gate._worker_command(
        bypass, Path("/tmp/gate")
    )
    assert gate.WATCHDOG_BYTES == 30 * gate.GIB

    summary = gate._watchdog_summary(
        {
            "passes": True,
            "execution": {"outcome": "command-finished", "reason": None},
            "enforcement": {
                "limit_bytes": gate.WATCHDOG_BYTES,
                "peak_rss_bytes": 10,
                "peak_physical_footprint_bytes": 11,
                "peak_guard_bytes": 11,
                "peak_processes": 2,
            },
        }
    )
    assert summary["passes"] is True
    assert summary["peak_guard_bytes"] == 11

    def probe(_pids: object) -> dict[int, int]:
        return {}

    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gate, "DarwinPhysicalFootprintProbe", lambda: probe)
    assert gate._physical_footprint_probe() is probe
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    assert gate._physical_footprint_probe() is None


def test_prepared_model_load_precedes_generation_and_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "built-in-sm.pyamplicol-model"
    path.write_bytes(b"prepared")
    events: list[object] = []
    compiled = object()
    retained = gate.RetainedInputs(object(), object(), b"{}", "a" * 64)

    class Source:
        def compile(self) -> object:
            events.append("load")
            return compiled

    def from_path(candidate: Path) -> Source:
        events.append(("path", candidate))
        return Source()

    def generate(artifact: Path, model: object) -> tuple[float, gate.RetainedInputs]:
        events.append(("generate", artifact, model))
        return 2.5, retained

    monkeypatch.setattr(gate.ModelSource, "from_path", from_path)
    monkeypatch.setattr(gate, "_generate", generate)

    result = gate._generate_with_prepared_model(tmp_path / "artifact", path)

    assert events == [
        ("path", path.resolve()),
        "load",
        ("generate", tmp_path / "artifact", compiled),
    ]
    assert result[:3] == (2.5, retained, path.resolve())
    assert result[3] >= 0.0


def test_prepared_model_path_rejects_missing_non_file_and_wrong_suffix(
    tmp_path: Path,
) -> None:
    with pytest.raises(gate.GateError, match="does not exist"):
        gate._prepared_model_path(tmp_path / "missing.pyamplicol-model")

    directory = tmp_path / "directory.pyamplicol-model"
    directory.mkdir()
    with pytest.raises(gate.GateError, match="not a regular file"):
        gate._prepared_model_path(directory)

    wrong_suffix = tmp_path / "prepared.bin"
    wrong_suffix.write_bytes(b"prepared")
    with pytest.raises(gate.GateError, match="must end"):
        gate._prepared_model_path(wrong_suffix)
