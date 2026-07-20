# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import sys
from dataclasses import replace
from pathlib import Path

from pyamplicol.api import BenchmarkResult, BenchmarkStatistics
from pyamplicol.cli import run_cli
from pyamplicol.cli.handlers import _load_process_output, _process_set
from pyamplicol.config import (
    BenchmarkConfig,
    ConfigurationError,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.reporting import (
    CallbackProgressSink,
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)

ROOT = Path(__file__).resolve().parents[2]


class _Services:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.config: RunConfig | None = None

    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if self.fail:
            raise ConfigurationError("generation rejected")
        self.config = config
        return {
            "action": config.action,
            "workers": config.generation.workers,
            "requests": tuple(entry.expression for entry in config.process.entries),
        }

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object:
        raise AssertionError((config, progress))

    benchmark = evaluate
    inspect = evaluate
    model_inspect = evaluate
    model_compile = evaluate
    model_processes = evaluate


class _ProfileServices(_Services):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        self.config = config
        progress.emit(ProgressStart("runtime-benchmark", "Profiling runtime", 2))
        progress.emit(ProgressUpdate("runtime-benchmark", 1, 2, "sampled"))
        progress.emit(ProgressUpdate("runtime-benchmark", 2, 2, "sampled"))
        progress.emit(ProgressEnd("runtime-benchmark"))
        benchmark = BenchmarkConfig(
            target_runtime=config.benchmark.target_runtime,
            batch_size=config.benchmark.batch_size,
            minimum_samples=config.benchmark.minimum_samples,
        )
        uncertainty = BenchmarkStatistics(1.0e-7, 5.0e-8, 0.05)
        return BenchmarkResult(
            requested_config=benchmark,
            effective_config=benchmark,
            sample_count=2,
            wall_time_per_point=1.0e-6,
            evaluator_time_per_point=8.0e-7,
            uncertainty=uncertainty,
            environment={
                "elapsed_seconds": benchmark.target_runtime,
                "platform": "test",
                "wall_time_source": "runtime_evaluate_wall_time",
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            },
            repetitions_per_sample=3,
            evaluator_uncertainty=uncertainty,
            process_id="d_dbar_to_z_g",
            process_expression="d d~ > z g",
        )


class _InterruptedProfileServices(_Services):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        del config, progress
        raise KeyboardInterrupt


class _PartialProfileServices(_ProfileServices):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        result = super().benchmark(config, progress)
        assert isinstance(result, BenchmarkResult)
        return replace(result, interrupted=True)


def test_typed_config_entries_preserve_complete_process_set_behavior() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(
                ProcessEntry("d d~ > z g", "ddbar_zg"),
                ProcessEntry("u u~ > z g"),
            )
        ),
    )

    processes = _process_set(config)

    assert tuple(request.expression for request in processes.requests) == (
        "d d~ > z g",
        "u u~ > z g",
    )
    assert processes.requests[0].name == "ddbar_zg"
    assert processes.requests[1].name == "u_ubar_to_z_g"


def test_process_output_loading_emits_visible_timed_progress(tmp_path: Path) -> None:
    events: list[object] = []
    progress = CallbackProgressSink(events.append)

    result = _load_process_output(tmp_path / "artifact", progress, lambda: "loaded")

    assert result == "loaded"
    assert isinstance(events[0], ProgressStart)
    assert str((tmp_path / "artifact").resolve()) in events[0].description
    assert isinstance(events[-1], ProgressEnd)
    assert events[-1].success is True
    assert events[-1].elapsed_seconds is not None


def test_cli_dispatches_protocol_and_keeps_json_on_stdout(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _Services()
    sys.modules.pop("symbolica", None)
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "3",
            "--format",
            "json",
            "--progress",
            "off",
        ),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 0
    assert json.loads(stdout.getvalue()) == {
        "action": "generate",
        "requests": ["d d~ > z g"],
        "workers": 3,
    }
    assert stderr.getvalue() == ""
    assert services.config is not None
    assert "symbolica" not in sys.modules


def test_cli_failures_write_only_diagnostics_to_stderr(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--format",
            "json",
        ),
        services=_Services(fail=True),
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 2
    assert stdout.getvalue() == ""
    assert "generation rejected" in stderr.getvalue()


def test_profile_json_is_stdout_clean_and_uses_benchmark_service(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _ProfileServices()

    status = run_cli(
        (
            "profile",
            str(tmp_path / "artifact"),
            "--process",
            "d d~ > z g",
            "--target-runtime",
            "0.1",
            "--batch-size",
            "4",
            "--minimum-samples",
            "2",
            "--format",
            "json",
            "--progress",
            "log",
        ),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["process_id"] == "d_dbar_to_z_g"
    assert payload["repetitions_per_sample"] == 3
    assert "Runtime Profile" not in stdout.getvalue()
    assert "\x1b[" not in stdout.getvalue()
    assert "Profiling runtime" in stderr.getvalue()
    assert services.config is not None
    assert services.config.action == "benchmark"
    assert services.config.evaluation.process == "d d~ > z g"


def test_profile_interrupted_before_sampling_exits_without_traceback(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        ("profile", str(tmp_path / "artifact"), "--progress", "off"),
        services=_InterruptedProfileServices(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 130
    assert stdout.getvalue() == ""
    assert "interrupted before a complete result was available" in stderr.getvalue()


def test_partial_profile_prints_result_and_exits_as_interrupted(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        (
            "profile",
            str(tmp_path / "artifact"),
            "--format",
            "json",
            "--progress",
            "off",
        ),
        services=_PartialProfileServices(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 130
    assert json.loads(stdout.getvalue())["interrupted"] is True
    assert stderr.getvalue() == ""


def test_inspect_cli_lists_artifact_processes_as_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    artifact = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"

    status = run_cli(
        ("inspect", str(artifact), "--format", "json", "--progress", "off"),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["kind"] == "pyamplicol-artifact-inspection"
    assert payload["default_process_id"] == "d_dbar_to_z"
    assert payload["processes"][0]["expression"] == "d d~ > z"
    assert stderr.getvalue() == ""
