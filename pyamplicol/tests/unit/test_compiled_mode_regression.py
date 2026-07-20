# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from tools.developer import compiled_mode_regression as regression

WATCHDOG_STDERR = (
    "memory-watchdog: command finished exit=0 peak_rss=1.250 GiB peak_processes=2\n"
)


def _write_artifact(
    path: Path,
    *,
    artifact_id: str = "artifact-id",
    process: str = "d d~ > z",
    color: str = "lc",
    eager: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    capability = (
        "rusticol.eager-dag.complex-f64.v1"
        if eager
        else "symjit.application.complex-f64.v1"
    )
    (path / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "producer": {"version": "test"},
                "processes": [
                    {
                        "id": "d_dbar_to_z",
                        "expression": process,
                        "color_accuracy": color,
                        "required_runtime_capabilities": [capability],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _profile_payload(wall: float, *, source: str | None = None) -> dict[str, object]:
    return {
        "wall_time_per_point": wall,
        "sample_count": 5,
        "repetitions_per_sample": 11,
        "interrupted": False,
        "uncertainty": {"standard_deviation": wall / 100.0},
        "environment": {
            "wall_time_source": source or regression.NATIVE_WALL_TIME_SOURCE,
            "elapsed_seconds": 5.0,
            "measured_evaluation_count": 55,
            "interrupted": False,
        },
    }


def test_parser_exposes_required_lanes_and_plan_defaults(tmp_path: Path) -> None:
    arguments = regression.parser().parse_args(
        [
            "--baseline-python",
            str(tmp_path / "baseline-python"),
            "--current-python",
            str(tmp_path / "current-python"),
            "--output-root",
            str(tmp_path / "output"),
            "--process",
            "d d~ > z",
        ]
    )

    assert arguments.model == "built-in-sm"
    assert arguments.color == "lc"
    assert arguments.batch_size == 1024
    assert arguments.samples == 5
    assert arguments.target_runtime == 5.0
    assert arguments.minimum_samples == 5


def test_parser_rejects_fewer_than_five_outer_samples(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        regression.parser().parse_args(
            [
                "--baseline-python",
                str(tmp_path / "baseline-python"),
                "--current-python",
                str(tmp_path / "current-python"),
                "--output-root",
                str(tmp_path / "output"),
                "--process",
                "d d~ > z",
                "--samples",
                "4",
            ]
        )


def test_generation_command_is_cross_version_compiled_jit_o3() -> None:
    command = regression._generation_command(
        Path("baseline-python"),
        process="d d~ > z",
        artifact=Path("artifact"),
        model="built-in-sm",
        color="full",
    )

    assert command[:4] == (
        "baseline-python",
        "-m",
        "pyamplicol",
        "generate",
    )
    assert command[command.index("--backend") + 1] == "jit"
    assert command[command.index("--jit-optimization-level") + 1] == "3"
    assert command[command.index("--color-accuracy") + 1] == "full"
    assert command[command.index("--workers") + 1] == "1"
    assert "--execution-mode" not in command
    assert "--no-post-build-validation" in command


def test_profile_command_carries_sampling_and_selector_controls() -> None:
    command = regression._profile_command(
        Path("current-python"),
        artifact=Path("artifact"),
        process_id="process-id",
        batch_size=128,
        target_runtime=7.5,
        minimum_samples=7,
        warmup_runs=3,
        helicities=("h:-1,+1",),
        color_flows=("flow:1,2",),
    )

    assert command[3] == "profile"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--target-runtime") + 1] == "7.5"
    assert command[command.index("--minimum-samples") + 1] == "7"
    assert command[command.index("--warmup-runs") + 1] == "3"
    assert command[command.index("--helicity") + 1] == "h:-1,+1"
    assert command[command.index("--color-flow") + 1] == "flow:1,2"


def test_run_json_routes_the_child_through_fixed_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            captured["timeout"] = timeout
            return '{"ok": true}', WATCHDOG_STDERR

    def fake_popen(command: tuple[str, ...], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(regression.subprocess, "Popen", fake_popen)

    payload, _elapsed, stderr = regression._run_json(
        ("lane-python", "-m", "pyamplicol", "profile"),
        timeout=10.0,
        environment={"PATH": "/bin"},
    )

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[:6] == (
        sys.executable,
        str(regression.WATCHDOG),
        "--limit-gib",
        "30",
        "--",
        "lane-python",
    )
    assert payload == {"ok": True}
    assert stderr == WATCHDOG_STDERR
    assert captured["timeout"] == 10.0


def test_native_profile_sample_requires_rusticol_marker() -> None:
    sample = regression._native_profile_sample(
        _profile_payload(1.25e-6),
        minimum_samples=5,
    )

    assert sample["wall_seconds_per_point"] == 1.25e-6
    assert sample["native_wall_time_source"] == ("runtime_core_repeated_wall_time")
    assert sample["profile_timed_block_count"] == 5

    with pytest.raises(regression.RegressionError, match="native Rusticol"):
        regression._native_profile_sample(
            _profile_payload(1.25e-6, source="runtime_evaluate_wall_time"),
            minimum_samples=5,
        )


def test_distribution_reports_median_and_median_absolute_deviation() -> None:
    summary = regression._distribution((1.0, 2.0, 3.0, 4.0, 100.0))

    assert summary["samples_seconds_per_point"] == [1.0, 2.0, 3.0, 4.0, 100.0]
    assert summary["median_seconds_per_point"] == 3.0
    assert summary["mad_seconds_per_point"] == 1.0


def test_gate_requires_both_two_percent_and_three_baseline_mad() -> None:
    baseline = {
        "median_seconds_per_point": 100.0,
        "mad_seconds_per_point": 1.0,
    }

    passing = regression._regression_gate(
        baseline,
        {"median_seconds_per_point": 101.9},
    )
    relative_failure = regression._regression_gate(
        baseline,
        {"median_seconds_per_point": 102.1},
    )
    mad_failure = regression._regression_gate(
        {"median_seconds_per_point": 100.0, "mad_seconds_per_point": 0.1},
        {"median_seconds_per_point": 101.0},
    )

    assert passing["passes"] is True
    assert relative_failure["within_two_percent"] is False
    assert relative_failure["within_three_baseline_mad"] is True
    assert relative_failure["passes"] is False
    assert mad_failure["within_two_percent"] is True
    assert mad_failure["within_three_baseline_mad"] is False
    assert mad_failure["passes"] is False


def test_matching_artifact_cache_avoids_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = Path(sys.executable)
    artifact = tmp_path / "baseline" / "artifact"
    cache_path = tmp_path / "baseline" / "artifact-cache.json"
    _write_artifact(artifact)
    signature = regression._generation_signature(
        python,
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
    )
    regression._write_json_atomic(
        cache_path,
        {
            "kind": regression.CACHE_KIND,
            "schema_version": regression.SCHEMA_VERSION,
            "signature": signature,
            "artifact_id": "artifact-id",
        },
    )

    def unexpected_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("generation must be skipped for a matching cache")

    monkeypatch.setattr(regression, "_run_json", unexpected_run)

    result = regression._ensure_artifact(
        "baseline",
        python,
        output_root=tmp_path,
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        generation_timeout=300.0,
        regenerate=False,
        environment={},
    )

    assert result["reused"] is True
    assert result["generation"] is None
    assert result["artifact_id"] == "artifact-id"


def test_eager_artifact_is_never_accepted_as_compiled(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, eager=True)

    with pytest.raises(regression.RegressionError, match="not compiled mode"):
        regression._artifact_metadata(
            artifact,
            expected_process="d d~ > z",
            expected_color="lc",
        )


def test_run_regression_reports_independent_alternating_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_python = tmp_path / "baseline-python"
    current_python = tmp_path / "current-python"
    for python in (baseline_python, current_python):
        python.write_text("", encoding="utf-8")
        python.chmod(0o755)

    output_root = tmp_path / "output"
    artifacts = {
        lane: {
            "artifact_id": f"{lane}-artifact",
            "path": str(output_root / lane / "artifact"),
            "process_id": "d_dbar_to_z",
            "process_expression": "d d~ > z",
            "color_accuracy": "lc",
            "size_bytes": 100,
            "producer": {"version": lane},
            "reused": True,
            "generation": None,
            "cache_path": str(output_root / lane / "artifact-cache.json"),
        }
        for lane in ("baseline", "current")
    }

    def fake_ensure(lane: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        return artifacts[lane]

    baseline_values = iter((1.00, 1.01, 0.99, 1.005, 0.995))
    current_values = iter((1.005, 1.006, 1.004, 1.005, 1.005))
    observed_order: list[str] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        timeout: float,
        environment: dict[str, str],
    ) -> tuple[dict[str, object], float, str]:
        del timeout
        assert "PYTHONPATH" not in environment
        artifact = Path(command[4])
        lane = artifact.parent.name
        observed_order.append(lane)
        value = next(baseline_values if lane == "baseline" else current_values)
        return _profile_payload(value), 0.25, WATCHDOG_STDERR

    monkeypatch.setattr(regression, "_ensure_artifact", fake_ensure)
    monkeypatch.setattr(regression, "_run_json", fake_run)
    arguments = argparse.Namespace(
        baseline_python=baseline_python,
        current_python=current_python,
        output_root=output_root,
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        batch_size=1024,
        samples=5,
        target_runtime=5.0,
        minimum_samples=5,
        warmup_runs=2,
        helicity=[],
        color_flow=[],
        generation_timeout=300.0,
        profile_timeout=120.0,
        regenerate_artifacts=False,
    )

    result = regression.run_regression(arguments)

    assert observed_order == [
        "baseline",
        "current",
        "current",
        "baseline",
        "baseline",
        "current",
        "current",
        "baseline",
        "baseline",
        "current",
    ]
    assert result["complete"] is True
    assert result["passes"] is True
    assert result["distributions"]["baseline"]["sample_count"] == 5
    assert result["distributions"]["current"]["sample_count"] == 5
    assert result["resources"]["profile_subprocess_count"] == 10
    assert result["resources"]["generation_subprocess_count"] == 0
    assert all(
        measurement["native_wall_time_source"] == regression.NATIVE_WALL_TIME_SOURCE
        for measurement in result["measurements"]
    )
    persisted = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert persisted["complete"] is True
    assert persisted["gate"] == result["gate"]
