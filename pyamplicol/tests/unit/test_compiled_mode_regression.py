# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import compiled_mode_regression as regression
from tools.developer import compiled_mode_sample

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
    payload_path = path / "payload.bin"
    payload_path.write_bytes(b"compiled-payload")
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
                "payloads": [
                    {
                        "path": "payload.bin",
                        "role": "evaluator-state",
                        "sha256": regression._sha256_file(payload_path),
                        "size_bytes": payload_path.stat().st_size,
                    }
                ],
                "extensions": {
                    "generation": {
                        "concrete_processes": [
                            {
                                "id": "d_dbar_to_z",
                                "filters": {
                                    "lc_flow_layout": "topology-replay",
                                },
                            }
                        ]
                    }
                },
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


def _profile_payload(
    wall: float,
    *,
    source: str | None = None,
    sample_pass: str | None = None,
    timing_contract: str | None = None,
    batch_size: int = 1024,
    numerical_values: tuple[float, ...] = (2.5,),
    native_module_sha256: str = "a" * 64,
) -> dict[str, object]:
    sample_count = 5
    repetitions = 11
    evaluations = sample_count * repetitions
    points = evaluations * batch_size
    profiled_wall = wall * 1.1
    return {
        "kind": regression.NATIVE_SAMPLE_RESULT_KIND,
        "schema_version": regression.NATIVE_SAMPLE_SCHEMA_VERSION,
        "wall_time_per_point": wall,
        "wall_time_samples_seconds_per_point": [wall] * sample_count,
        "sample_count": sample_count,
        "repetitions_per_sample": repetitions,
        "interrupted": False,
        "uncertainty": {"standard_deviation": wall / 100.0},
        "evaluator_time_per_point": wall / 2.0,
        "evaluator_uncertainty": {"standard_deviation": wall / 200.0},
        "warmed_numerical_result": {
            "point_count": len(numerical_values),
            "batch_sha256": "d" * 64,
            "values_f64": list(numerical_values),
            "values_f64_hex": [value.hex() for value in numerical_values],
            "helicities": [],
            "color_flows": [],
        },
        "timing_breakdown": {
            "sample_count": sample_count,
            "execution_mode": "compiled",
            "wall_time": {
                "mean_seconds_per_point": profiled_wall,
                "sample_count": sample_count,
                "samples_seconds_per_point": [profiled_wall] * sample_count,
            },
            "raw_profile_samples": [
                {"wall_time_s": profiled_wall * batch_size * repetitions}
                for _ in range(sample_count)
            ],
        },
        "environment": {
            "wall_time_source": source or regression.NATIVE_WALL_TIME_SOURCE,
            "wall_time_sample_pass": (
                sample_pass or regression.NATIVE_WALL_TIME_SAMPLE_PASS
            ),
            "evaluator_time_sample_pass": (regression.PROFILE_ATTRIBUTION_SAMPLE_PASS),
            "timing_breakdown_sample_pass": (
                regression.PROFILE_ATTRIBUTION_SAMPLE_PASS
            ),
            "timing_sample_contract": (
                timing_contract or regression.PAIRED_TIMING_SAMPLE_CONTRACT
            ),
            "profile_attribution_paired_with_headline": True,
            "profile_attribution_identical_batch": True,
            "profile_attribution_identical_repetitions": True,
            "execution_mode": "compiled",
            "batch_size": batch_size,
            "completed_sample_count": sample_count,
            "planned_sample_count": sample_count,
            "measured_point_count": points,
            "native_profile_sample_count": sample_count,
            "native_profile_sample_limit": sample_count,
            "native_profile_repetitions_per_sample": repetitions,
            "native_profile_points_per_sample": repetitions * batch_size,
            "native_profile_calls_per_block": 1.0,
            "profile_attribution_evaluation_count": evaluations,
            "profile_attribution_point_count": points,
            "elapsed_seconds": 5.0,
            "measured_evaluation_count": evaluations,
            "interrupted": False,
            "batch_sha256": "c" * 64,
            "helicities": [],
            "color_flows": [],
            "runtime_identity": {
                "native_module": {
                    "sha256": native_module_sha256,
                }
            },
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
    assert arguments.lc_flow_layout == "topology-replay"
    assert arguments.shared_artifact is None
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
        lc_flow_layout="topology-replay",
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
    assert command[command.index("--lc-flow-layout") + 1] == "topology-replay"
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

    assert command[:2] == (
        "current-python",
        str(regression.NATIVE_SAMPLE_HELPER),
    )
    assert command[2] == "artifact"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--target-runtime") + 1] == "7.5"
    assert command[command.index("--minimum-samples") + 1] == "7"
    assert command[command.index("--warmup-runs") + 1] == "3"
    assert command[command.index("--helicity") + 1] == "h:-1,+1"
    assert command[command.index("--color-flow") + 1] == "flow:1,2"
    parsed = compiled_mode_sample.parser().parse_args(command[2:])
    assert parsed.artifact == Path("artifact")
    assert parsed.process == "process-id"


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

    monkeypatch.setattr(
        "tools.developer.compiled_mode_regression.subprocess.Popen",
        fake_popen,
    )

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
    assert sample["native_wall_time_sample_pass"] == (
        regression.NATIVE_WALL_TIME_SAMPLE_PASS
    )
    assert sample["timing_sample_contract"] == (
        regression.PAIRED_TIMING_SAMPLE_CONTRACT
    )
    assert sample["profile_timed_block_count"] == 5
    assert sample["measured_evaluation_count"] == 55
    assert sample["measured_point_count"] == 55 * 1024

    with pytest.raises(regression.RegressionError, match="native Rusticol"):
        regression._native_profile_sample(
            _profile_payload(1.25e-6, source="runtime_evaluate_wall_time"),
            minimum_samples=5,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            _profile_payload(
                1.25e-6,
                sample_pass="runtime.profile_repeated",
            ),
            "wall_time_sample_pass",
        ),
        (
            _profile_payload(
                1.25e-6,
                timing_contract="shared_native_repeated_profile_v1",
            ),
            "timing_sample_contract",
        ),
    ),
)
def test_native_profile_sample_rejects_profiled_or_unpaired_headlines(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(regression.RegressionError, match=message):
        regression._native_profile_sample(payload, minimum_samples=5)


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


def test_performance_authority_requires_shared_or_matching_payloads() -> None:
    matching_payloads = [
        {
            "path": "evaluators.pacbin",
            "role": "evaluator-state",
            "sha256": "a" * 64,
            "size_bytes": 123,
        }
    ]
    artifacts = {
        lane: {
            "process_id": "d_dbar_to_z",
            "color_accuracy": "lc",
            "lc_flow_layout": "topology-replay",
            "payload_digests": matching_payloads,
        }
        for lane in ("baseline", "current")
    }

    proven = regression._performance_authority(
        artifacts,
        shared_artifact=False,
    )
    shared = regression._performance_authority(
        artifacts,
        shared_artifact=True,
    )
    artifacts["current"] = {
        **artifacts["current"],
        "payload_digests": [
            {
                **matching_payloads[0],
                "sha256": "b" * 64,
            }
        ],
    }
    different = regression._performance_authority(
        artifacts,
        shared_artifact=False,
    )

    assert proven["authoritative"] is True
    assert proven["basis"] == "matching-performance-relevant-payload-identities"
    assert shared["authoritative"] is True
    assert shared["basis"] == "single-read-only-shared-artifact"
    assert different["authoritative"] is False
    assert "differ" in different["basis"]


def test_correctness_gate_compares_warmed_values_at_contract_tolerances() -> None:
    def measurement(
        lane: str,
        value: float,
        pair_index: int,
    ) -> dict[str, object]:
        return {
            "lane": lane,
            "pair_index": pair_index,
            "warmed_numerical_result": {
                "point_count": 1,
                "batch_sha256": "d" * 64,
                "values_f64": [value],
                "helicities": ["h:-1,+1"],
                "color_flows": ["flow:1,2"],
            },
        }

    passing = regression._correctness_gate(
        (
            measurement("baseline", 2.5, 1),
            measurement("current", 2.5 + 2.0e-12, 1),
        )
    )
    failing = regression._correctness_gate(
        (
            measurement("baseline", 2.5, 1),
            measurement("current", 2.5 + 3.0e-11, 1),
        )
    )

    assert passing["relative_tolerance"] == 1.0e-12
    assert passing["absolute_tolerance"] == 1.0e-15
    assert passing["passes"] is True
    assert failing["passes"] is False
    assert failing["comparisons"][1]["passes"] is False


def test_exact_identities_cover_file_tree_model_and_command_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "payload.bin").write_bytes(b"same bytes")

    first_tree = regression._tree_identity(first)
    second_tree = regression._tree_identity(second)
    assert first_tree == second_tree
    assert first_tree["file_count"] == 1

    model = tmp_path / "model.json"
    model.write_text('{"model": 1}\n', encoding="utf-8")
    model_identity = regression._model_identity(str(model))
    assert model_identity["sha256"] == regression._sha256_file(model)

    command = regression._command_identity(("python", "helper.py", "--value", "1"))
    assert command == regression._command_identity(
        ("python", "helper.py", "--value", "1")
    )
    assert command != regression._command_identity(
        ("python", "helper.py", "--value", "2")
    )

    (second / "nested" / "payload.bin").write_bytes(b"changed bytes")
    assert regression._tree_identity(second)["sha256"] != first_tree["sha256"]


def test_artifact_metadata_records_exact_tree_and_requires_requested_layout(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)

    metadata = regression._artifact_metadata(
        artifact,
        expected_process="d d~ > Z",
        expected_color="lc",
        expected_lc_flow_layout="topology-replay",
    )

    assert metadata["manifest_sha256"] == regression._sha256_file(
        artifact / "artifact.json"
    )
    assert metadata["lc_flow_layout"] == "topology-replay"
    assert metadata["size_bytes"] == metadata["tree_identity"]["size_bytes"]
    assert metadata["payload_digests"] == [
        {
            "path": "payload.bin",
            "role": "evaluator-state",
            "sha256": regression._sha256_file(artifact / "payload.bin"),
            "size_bytes": len(b"compiled-payload"),
        }
    ]
    with pytest.raises(regression.RegressionError, match="LC flow layout"):
        regression._artifact_metadata(
            artifact,
            expected_process="d d~ > z",
            expected_color="lc",
            expected_lc_flow_layout="all-flow-union",
        )
    (artifact / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(regression.RegressionError, match="declared digest"):
        regression._artifact_metadata(
            artifact,
            expected_process="d d~ > z",
            expected_color="lc",
            expected_lc_flow_layout="topology-replay",
        )


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
        installation_identity={
            "kind": regression.INSTALLATION_IDENTITY_KIND,
            "schema_version": regression.INSTALLATION_IDENTITY_SCHEMA_VERSION,
            "distribution_content": {"sha256": "e" * 64},
            "native_modules": [{"sha256": "f" * 64}],
        },
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        artifact=artifact,
    )
    metadata = regression._artifact_metadata(
        artifact,
        expected_process="d d~ > z",
        expected_color="lc",
        expected_lc_flow_layout="topology-replay",
    )
    tree_identity = metadata["tree_identity"]
    assert isinstance(tree_identity, dict)
    regression._write_json_atomic(
        cache_path,
        {
            "kind": regression.CACHE_KIND,
            "schema_version": regression.SCHEMA_VERSION,
            "signature": signature,
            "artifact_id": "artifact-id",
            "artifact_tree_sha256": tree_identity["sha256"],
        },
    )

    def unexpected_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("generation must be skipped for a matching cache")

    monkeypatch.setattr(regression, "_run_json", unexpected_run)
    monkeypatch.setattr(
        regression,
        "_installed_pyamplicol_identity",
        lambda *_args, **_kwargs: {
            "kind": regression.INSTALLATION_IDENTITY_KIND,
            "schema_version": regression.INSTALLATION_IDENTITY_SCHEMA_VERSION,
            "distribution_content": {"sha256": "e" * 64},
            "native_modules": [{"sha256": "f" * 64}],
        },
    )

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


def test_generation_signature_tracks_reinstalled_pyamplicol_content(
    tmp_path: Path,
) -> None:
    python = Path(sys.executable)
    common = {
        "kind": regression.INSTALLATION_IDENTITY_KIND,
        "schema_version": regression.INSTALLATION_IDENTITY_SCHEMA_VERSION,
        "native_modules": [{"sha256": "a" * 64}],
    }
    first = regression._generation_signature(
        python,
        installation_identity={
            **common,
            "distribution_content": {"sha256": "b" * 64},
            "build_info_files": [{"sha256": "c" * 64}],
        },
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        artifact=tmp_path / "artifact",
    )
    reinstalled = regression._generation_signature(
        python,
        installation_identity={
            **common,
            "distribution_content": {"sha256": "d" * 64},
            "build_info_files": [{"sha256": "e" * 64}],
        },
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        artifact=tmp_path / "artifact",
    )

    assert first["python"] == reinstalled["python"]
    assert first["installed_pyamplicol"] != reinstalled["installed_pyamplicol"]
    assert first != reinstalled


def test_eager_artifact_is_never_accepted_as_compiled(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, eager=True)

    with pytest.raises(regression.RegressionError, match="not compiled mode"):
        regression._artifact_metadata(
            artifact,
            expected_process="d d~ > z",
            expected_color="lc",
        )


def test_run_regression_reuses_one_read_only_shared_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_python = tmp_path / "baseline-python"
    current_python = tmp_path / "current-python"
    for python in (baseline_python, current_python):
        python.write_text("", encoding="utf-8")
        python.chmod(0o755)
    artifact = tmp_path / "retained-artifact"
    _write_artifact(artifact)
    original_payload = (artifact / "payload.bin").read_bytes()
    observed_commands: list[tuple[str, ...]] = []

    def unexpected_generation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shared artifact mode must not generate an artifact")

    def fake_run(
        command: tuple[str, ...],
        *,
        timeout: float,
        environment: dict[str, str],
    ) -> tuple[dict[str, object], float, str]:
        del timeout
        assert "PYTHONPATH" not in environment
        assert Path(command[2]) == artifact
        observed_commands.append(command)
        return _profile_payload(1.0, batch_size=4), 0.25, WATCHDOG_STDERR

    monkeypatch.setattr(regression, "_ensure_artifact", unexpected_generation)
    monkeypatch.setattr(regression, "_run_json", fake_run)
    arguments = argparse.Namespace(
        baseline_python=baseline_python,
        current_python=current_python,
        output_root=tmp_path / "output",
        shared_artifact=artifact,
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        lc_flow_layout="topology-replay",
        batch_size=4,
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

    assert len(observed_commands) == 10
    assert (artifact / "payload.bin").read_bytes() == original_payload
    assert result["configuration"]["shared_artifact"] == str(artifact)
    assert result["resources"]["generation_subprocess_count"] == 0
    assert {lane: value["path"] for lane, value in result["artifacts"].items()} == {
        "baseline": str(artifact),
        "current": str(artifact),
    }
    assert all(
        value["shared"] is True
        and value["payload_digest_count"] == 1
        and value["comparison_role"] == "shared-read-only-artifact"
        for value in result["artifacts"].values()
    )
    assert result["performance_authority"]["authoritative"] is True
    assert result["performance_result_authoritative"] is True
    assert result["gate"]["authoritative"] is True
    assert result["correctness_gate"]["passes"] is True


def test_run_regression_rejects_native_module_change_within_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_python = tmp_path / "baseline-python"
    current_python = tmp_path / "current-python"
    for python in (baseline_python, current_python):
        python.write_text("", encoding="utf-8")
        python.chmod(0o755)
    artifact = tmp_path / "retained-artifact"
    _write_artifact(artifact)
    lane_calls = {"baseline": 0, "current": 0}

    def fake_run(
        command: tuple[str, ...],
        *,
        timeout: float,
        environment: dict[str, str],
    ) -> tuple[dict[str, object], float, str]:
        del timeout, environment
        lane = "baseline" if Path(command[0]) == baseline_python else "current"
        lane_calls[lane] += 1
        native_sha256 = (
            "b" * 64 if lane == "baseline" and lane_calls[lane] > 1 else "a" * 64
        )
        return (
            _profile_payload(1.0, native_module_sha256=native_sha256),
            0.25,
            WATCHDOG_STDERR,
        )

    monkeypatch.setattr(regression, "_run_json", fake_run)
    arguments = argparse.Namespace(
        baseline_python=baseline_python,
        current_python=current_python,
        output_root=tmp_path / "output",
        shared_artifact=artifact,
        process="d d~ > z",
        model="built-in-sm",
        color="lc",
        lc_flow_layout="topology-replay",
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

    with pytest.raises(
        regression.RegressionError,
        match="baseline native module changed during sampling",
    ):
        regression.run_regression(arguments)


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
    artifacts: dict[str, dict[str, object]] = {
        lane: {
            "artifact_id": f"{lane}-artifact",
            "path": str(output_root / lane / "artifact"),
            "process_id": "d_dbar_to_z",
            "process_expression": "d d~ > z",
            "color_accuracy": "lc",
            "lc_flow_layout": "topology-replay",
            "manifest_sha256": ("a" if lane == "baseline" else "b") * 64,
            "tree_identity": {
                "sha256": ("c" if lane == "baseline" else "d") * 64,
            },
            "payload_digests": [],
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
        assert command[1] == str(regression.NATIVE_SAMPLE_HELPER)
        artifact = Path(command[2])
        lane = artifact.parent.name
        observed_order.append(lane)
        value = next(baseline_values if lane == "baseline" else current_values)
        return _profile_payload(value), 0.25, WATCHDOG_STDERR

    monkeypatch.setattr(regression, "_ensure_artifact", fake_ensure)
    monkeypatch.setattr(regression, "_run_json", fake_run)
    monkeypatch.setattr(
        regression,
        "_artifact_metadata",
        lambda artifact, **_kwargs: artifacts[artifact.parent.name],
    )
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
    assert result["passes"] is False
    assert result["performance_result_authoritative"] is False
    assert result["gate"]["measured_thresholds_pass"] is True
    assert result["gate"]["authoritative"] is False
    assert result["gate"]["passes"] is False
    assert result["performance_authority"]["comparison_mode"] == (
        "independently-generated-artifacts"
    )
    assert result["distributions"]["baseline"]["sample_count"] == 5
    assert result["distributions"]["current"]["sample_count"] == 5
    assert result["resources"]["profile_subprocess_count"] == 10
    assert result["resources"]["generation_subprocess_count"] == 0
    assert result["native_module_sha256_by_lane"] == {
        "baseline": "a" * 64,
        "current": "a" * 64,
    }
    assert result["configuration"]["lc_flow_layout"] == "topology-replay"
    assert result["configuration"]["native_wall_time_sample_pass"] == (
        regression.NATIVE_WALL_TIME_SAMPLE_PASS
    )
    assert all(
        measurement["native_wall_time_source"] == regression.NATIVE_WALL_TIME_SOURCE
        for measurement in result["measurements"]
    )
    assert all(
        measurement["command"]["argv"][1] == str(regression.NATIVE_SAMPLE_HELPER)
        for measurement in result["measurements"]
    )
    assert result["provenance"]["native_sample_helper"]["sha256"] == (
        regression._sha256_file(regression.NATIVE_SAMPLE_HELPER)
    )
    persisted = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert persisted["complete"] is True
    assert persisted["gate"] == result["gate"]


def test_run_regression_rehashes_generated_artifacts_after_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_python = tmp_path / "baseline-python"
    current_python = tmp_path / "current-python"
    for python in (baseline_python, current_python):
        python.write_text("", encoding="utf-8")
        python.chmod(0o755)
    output_root = tmp_path / "output"
    artifacts: dict[str, dict[str, object]] = {}
    for lane in ("baseline", "current"):
        path = output_root / lane / "artifact"
        _write_artifact(path, artifact_id=f"{lane}-artifact")
        artifacts[lane] = {
            **regression._artifact_metadata(
                path,
                expected_process="d d~ > z",
                expected_color="lc",
                expected_lc_flow_layout="topology-replay",
            ),
            "reused": True,
            "generation": None,
            "cache_path": str(output_root / lane / "artifact-cache.json"),
        }

    def fake_ensure(lane: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        return artifacts[lane]

    sample_calls = 0

    def fake_run(
        command: tuple[str, ...],
        *,
        timeout: float,
        environment: dict[str, str],
    ) -> tuple[dict[str, object], float, str]:
        nonlocal sample_calls
        del command, timeout, environment
        sample_calls += 1
        if sample_calls == 10:
            (output_root / "current" / "artifact" / "late-file.bin").write_bytes(
                b"mutation"
            )
        return _profile_payload(1.0), 0.25, WATCHDOG_STDERR

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

    with pytest.raises(
        regression.RegressionError,
        match="current artifact changed during sampling: tree_identity differs",
    ):
        regression.run_regression(arguments)


def test_native_sample_helper_pairs_direct_wall_and_profile_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_module = tmp_path / "native-module.so"
    native_module.write_bytes(b"native-module")

    class FakeRuntime:
        def __init__(self) -> None:
            self._native_module = SimpleNamespace(__file__=str(native_module))
            self.wall_calls: list[tuple[int, int, dict[str, object]]] = []
            self.profile_calls: list[tuple[int, int, dict[str, object]]] = []
            self.evaluate_calls: list[tuple[int, dict[str, object]]] = []

        def metadata_json(self) -> str:
            return '{"execution_mode": "compiled"}'

        def _benchmark_f64_wall_time(
            self,
            batch: tuple[object, ...],
            repetitions: int,
            **kwargs: object,
        ) -> float:
            self.wall_calls.append((id(batch), repetitions, dict(kwargs)))
            return repetitions * 0.01

        def profile_repeated(
            self,
            batch: tuple[object, ...],
            repetitions: int,
            **kwargs: object,
        ) -> dict[str, object]:
            self.profile_calls.append((id(batch), repetitions, dict(kwargs)))
            points = repetitions * len(batch)
            return {
                "execution_mode": "compiled",
                "wall_time_s": points * 0.007,
                "stage_evaluator_call_time_s": points * 0.003,
                "amplitude_evaluator_call_time_s": points * 0.001,
                "stage_input_copy_component_count": points * 17,
            }

        def evaluate(
            self,
            batch: tuple[object, ...],
            **kwargs: object,
        ) -> list[float]:
            self.evaluate_calls.append((id(batch), dict(kwargs)))
            return [2.5] * len(batch)

    runtime = FakeRuntime()
    monkeypatch.setattr(
        compiled_mode_sample,
        "_load_runtime",
        lambda *_args, **_kwargs: (
            runtime,
            {"native_module": {"sha256": "b" * 64}},
        ),
    )
    monkeypatch.setattr(
        compiled_mode_sample,
        "_validation_momenta",
        lambda *_args, **_kwargs: (((1.0, 0.0, 0.0, 1.0),),),
    )
    monkeypatch.setattr(
        compiled_mode_sample,
        "_resolve_color_flows",
        lambda _artifact, *, process, requested: tuple(requested),
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    arguments = argparse.Namespace(
        artifact=artifact,
        process="d_dbar_to_z",
        target_runtime=0.05,
        batch_size=2,
        minimum_samples=5,
        warmup_runs=1,
        helicity=["h:-1,+1"],
        color_flow=["flow:1,2"],
    )

    result = compiled_mode_sample.sample(arguments)

    assert result["wall_time_per_point"] == pytest.approx(0.005)
    breakdown = result["timing_breakdown"]
    assert isinstance(breakdown, dict)
    profile_wall = breakdown["wall_time"]
    assert isinstance(profile_wall, dict)
    assert profile_wall["mean_seconds_per_point"] == pytest.approx(0.007)
    measured = regression._native_profile_sample(
        result,
        minimum_samples=5,
        batch_size=2,
    )
    assert measured["wall_seconds_per_point"] == pytest.approx(0.005)
    assert measured["paired_profile_wall_seconds_per_point"] == pytest.approx(0.007)
    assert measured["warmed_numerical_result"]["values_f64"] == [2.5, 2.5]
    assert len(runtime.wall_calls) == 7
    assert len(runtime.profile_calls) == 6
    assert len(runtime.evaluate_calls) == 1
    assert runtime.evaluate_calls[0][1] == {
        "helicities": ("h:-1,+1",),
        "color_flows": ("flow:1,2",),
        "precision": 16,
    }
    for wall_call, profile_call in zip(
        runtime.wall_calls[-5:],
        runtime.profile_calls[-5:],
        strict=True,
    ):
        assert wall_call[:2] == profile_call[:2]
        assert wall_call[2] == {
            "helicities": ("h:-1,+1",),
            "color_flows": ("flow:1,2",),
            "precision": 16,
        }
        assert profile_call[2] == {
            **wall_call[2],
            "include_values": False,
        }
