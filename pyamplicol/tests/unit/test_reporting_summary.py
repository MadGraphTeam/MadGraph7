# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from pyamplicol.api import (
    BenchmarkComponentTiming,
    BenchmarkResult,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    GenerationResult,
    ProcessRequest,
    ProcessSet,
)
from pyamplicol.artifacts import (
    ArtifactAliasInspection,
    ArtifactDependencyInspection,
    ArtifactInspection,
    ArtifactProcessInspection,
)
from pyamplicol.cli.main import write_result
from pyamplicol.config import BenchmarkConfig
from pyamplicol.reporting import render_summary


@dataclass(frozen=True)
class _Result:
    status: str
    generated_processes: int
    adjustments: tuple[str, ...]


def test_human_structured_results_use_aligned_prettytable() -> None:
    rendered = render_summary(_Result("complete", 2, ()), color=False)
    assert rendered is not None
    assert "field" in rendered
    assert "generated processes" in rendered
    assert "complete" in rendered


def test_generation_mode_is_labelled_as_existing_output_policy() -> None:
    rendered = render_summary(
        GenerationResult(
            output=Path("artifact"),
            processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
            mode="error",
        ),
        color=True,
    )

    assert rendered is not None
    assert "existing-output policy" in rendered
    assert "error" in rendered
    assert "\x1b[31merror" not in rendered


def test_generation_summary_compacts_file_inventory(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "artifact.json").write_bytes(b"x" * 24)
    manifest = SimpleNamespace(
        payloads=(
            SimpleNamespace(size_bytes=1000),
            SimpleNamespace(size_bytes=500),
        )
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    result = GenerationResult(
        output=artifact,
        processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
        mode="error",
        files=(artifact / "one", artifact / "two", artifact / "artifact.json"),
    )

    rendered = render_summary(result, color=False)

    assert rendered is not None
    assert "3 files; 1.49 KiB total" in rendered
    assert str(result.files) not in rendered


def test_json_result_never_contains_table_or_color_sequences() -> None:
    stream = io.StringIO()
    write_result(_Result("complete", 2, ()), format="json", stream=stream, color=True)
    assert json.loads(stream.getvalue()) == {
        "adjustments": [],
        "generated_processes": 2,
        "status": "complete",
    }
    assert "\x1b[" not in stream.getvalue()


def _benchmark_result() -> BenchmarkResult:
    config = BenchmarkConfig(target_runtime=1.0, batch_size=32, minimum_samples=8)
    wall_uncertainty = BenchmarkStatistics(0.2e-6, 0.05e-6, 0.02)
    evaluator_uncertainty = BenchmarkStatistics(0.1e-6, 0.025e-6, 0.0125)
    component = BenchmarkComponentTiming(0.5e-6, evaluator_uncertainty, 8)
    breakdown = BenchmarkTimingBreakdown(
        sample_count=8,
        wall_time=component,
        source_fill_time=component,
        momentum_setup_time=component,
        stage_input_pack_time=component,
        stage_evaluator_call_time=component,
        output_assign_time=component,
        amplitude_input_pack_time=component,
        amplitude_evaluator_call_time=component,
        reduction_time=component,
        other_core_time=component,
        stages=(BenchmarkStageTiming(1, component, component, component),),
    )
    return BenchmarkResult(
        requested_config=config,
        effective_config=config,
        sample_count=8,
        wall_time_per_point=2.5e-6,
        evaluator_time_per_point=2.0e-6,
        uncertainty=wall_uncertainty,
        environment={
            "target": "/tmp/artifact",
            "elapsed_seconds": 1.01,
            "platform": "test-platform",
            "wall_time_source": "runtime_evaluate_wall_time",
            "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            "execution_mode": "compiled",
            "color_workload": "all 1 generated physical LC flows",
            "helicity_workload": "all 24 generated helicity configurations",
        },
        repetitions_per_sample=50,
        evaluator_uncertainty=evaluator_uncertainty,
        process_id="ddbar_zg",
        process_expression="d d~ > z g",
        timing_breakdown=breakdown,
    )


def test_benchmark_result_uses_clear_runtime_profile_table() -> None:
    rendered = render_summary(_benchmark_result(), color=False)

    assert rendered is not None
    assert "Runtime Profile" in rendered
    assert "ddbar_zg (d d~ > z g)" in rendered
    assert "execution mode" in rendered
    assert "compiled" in rendered
    assert "all 1 generated physical LC flows" in rendered
    assert "all 24 generated helicity configurations" in rendered
    assert "2.5 +/- 0.05 us/point (standard error)" in rendered
    assert "8 blocks x 50 repetitions x 32 points" in rendered
    assert "timed points" in rendered
    assert "Rusticol Timing Breakdown" in rendered
    assert "Source fill" in rendered
    assert "Other Rusticol core" in rendered
    assert "Rusticol Stage Detail" in rendered
    assert "evaluator call" in rendered


def test_benchmark_profile_table_color_is_optional() -> None:
    rendered = render_summary(_benchmark_result(), color=True)

    assert rendered is not None
    assert "\x1b[" in rendered


def test_eager_benchmark_uses_eager_phase_labels() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    component = breakdown.stage_evaluator_call_time
    assert component is not None
    eager_breakdown = replace(
        breakdown,
        execution_mode="eager",
        stage_evaluator_call_time=None,
        eager_execution_time=component,
        eager_initialize_time=component,
        eager_gather_time=component,
        eager_kernel_call_time=component,
        eager_invocation_scatter_time=component,
        eager_finalization_time=component,
        eager_scatter_finalization_time=component,
        eager_closure_time=component,
        eager_copy_out_time=component,
        stages=(),
    )

    rendered = render_summary(
        replace(result, timing_breakdown=eager_breakdown), color=False
    )

    assert rendered is not None
    assert "Rusticol Eager Timing Breakdown" in rendered
    assert "Initialize" in rendered
    assert "Gather" in rendered
    assert "Kernel calls" in rendered
    assert "Invocation scatter" in rendered
    assert "Current finalization" in rendered
    assert "Amplitude closure" in rendered
    assert "Amplitude copy-out" in rendered
    assert "Rusticol Stage Detail" not in rendered


def test_eager_benchmark_labels_aggregate_until_native_phases_exist() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None

    rendered = render_summary(
        replace(
            result,
            timing_breakdown=replace(
                breakdown,
                execution_mode="eager",
                stage_evaluator_call_time=None,
                eager_execution_time=breakdown.stage_evaluator_call_time,
                stages=(),
            ),
        ),
        color=False,
    )

    assert rendered is not None
    assert "Eager execution (aggregate)" in rendered
    assert "Stage input pack" not in rendered
    assert "Output assign" not in rendered


def test_interrupted_benchmark_renders_partial_status() -> None:
    rendered = render_summary(
        replace(_benchmark_result(), interrupted=True), color=False
    )

    assert rendered is not None
    assert "interrupted - partial statistics from 8 complete blocks" in rendered


def _artifact_inspection() -> ArtifactInspection:
    alias = ArtifactAliasInspection(
        id="ddbar_zg_alias",
        expression="d d~ > g z",
        representative_id="ddbar_zg",
        external_pdgs=(1, -1, 21, 23),
    )
    process = ArtifactProcessInspection(
        id="ddbar_zg",
        expression="d d~ > z g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        default=True,
        physical_helicities=24,
        computed_helicities=12,
        physical_color_components=1,
        computed_color_components=1,
        helicity_coverage="complete",
        color_coverage="complete",
        aliases=(alias,),
    )
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=Path("/tmp/artifact"),
        artifact_kind="pyamplicol-process-set",
        artifact_id="a" * 64,
        created_utc="2026-07-16T00:00:00Z",
        producer_version="0.1.0",
        target="x86_64-unknown-linux-gnu",
        cpu_features=(),
        model_name="sm",
        model_source="ufo-json",
        model_restriction="default",
        default_process_id="ddbar_zg",
        runtime_engine="rusticol",
        runtime_version="0.1.0",
        runtime_capabilities=("symjit.application.complex-f64.v1",),
        payload_count=12,
        payload_size_bytes=2048,
        integrity="verified",
        processes=(process,),
        dependencies=(
            ArtifactDependencyInspection(
                name="symbolica",
                version="2.1.0",
                license="Symbolica Software License Agreement",
                source="https://symbolica.io/",
            ),
        ),
    )


def test_artifact_inspection_uses_process_and_alias_tables() -> None:
    rendered = render_summary(_artifact_inspection(), color=False)

    assert rendered is not None
    assert "Artifact" in rendered
    assert "Processes" in rendered
    assert "Aliases" in rendered
    assert "Dependencies" in rendered
    assert "Execution" in rendered
    assert "ddbar_zg" in rendered
    assert "d d~ > z g" in rendered
    assert "24 (12 eval.)" in rendered


def test_eager_artifact_inspection_reports_execution_contract() -> None:
    inspection = _artifact_inspection()
    eager_process = replace(
        inspection.processes[0],
        execution_mode="eager",
        prepared_backend="jit",
        prepared_kernel_count=12,
        referenced_kernel_count=9,
        invocation_count=80,
        attachment_count=100,
        evaluation_alias_count=20,
        maximum_fanout=4,
        finalization_count=30,
        closure_count=12,
        selector_closure_available=False,
        requested_point_tile_size=1024,
        effective_point_tile_size=None,
        workspace_limit_bytes=256 * 1024 * 1024,
        workspace_bytes=None,
        native_profile_phases=(
            "source-fill",
            "momentum-setup",
            "eager-execution-aggregate",
        ),
    )

    rendered = render_summary(
        replace(inspection, processes=(eager_process,)), color=False
    )

    assert rendered is not None
    assert "eager (jit)" in rendered
    assert "12 in pack; 9 referenced" in rendered
    assert "80 canonical; 100 attachments; 20 reused aliases; max fanout 4" in rendered
    assert "not emitted" in rendered
    assert "1024 / available after runtime load" in rendered
    assert "256.00 MiB / available after runtime load" in rendered


def test_artifact_inspection_color_is_optional() -> None:
    rendered = render_summary(_artifact_inspection(), color=True)

    assert rendered is not None
    assert "\x1b[" in rendered
