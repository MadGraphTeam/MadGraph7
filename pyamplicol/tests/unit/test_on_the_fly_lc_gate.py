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
    assert gate._calibrate(0.25, 1.0) == 4

    poisoned = report(0)
    poisoned["direct_plan_load_attempts"] = 1
    with pytest.raises(gate.GateError, match="poison"):
        gate._probe_values(poisoned, 2)
    wrong_fill = report(5)
    wrong_fill["momentum_fill_count"] = 6
    with pytest.raises(gate.GateError, match="contract"):
        gate._probe_values(wrong_fill, 2, 5)


def test_cli_launches_one_worker_with_cross_platform_30_gib_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = gate._parser().parse_args(
        [
            "--output",
            "out",
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
    assert command[command.index("--amplicol-result") + 1] == str(
        Path("legacy.json").resolve()
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
