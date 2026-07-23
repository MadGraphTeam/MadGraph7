# SPDX-License-Identifier: 0BSD
"""Static external consumer of the installed inline public API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, assert_type

from pyamplicol import (
    BenchmarkComponentTiming,
    BenchmarkConfig,
    BenchmarkProfileCounters,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    ColorFlow,
    CompiledModel,
    ContractedColorComponent,
    EvaluationConfig,
    GenerationConfig,
    GenerationPlan,
    GenerationResult,
    Generator,
    HelicityConfiguration,
    ModelSource,
    ProcessPhysics,
    ProcessRequest,
    ProcessSet,
    ResolvedEvaluation,
    RunConfig,
    Runtime,
    benchmark,
    generate,
    load,
)
from pyamplicol.api import GeneratorBackend, ModelParameters, Momenta, RuntimeBackend
from pyamplicol.config import (
    CONFIG_SECTIONS,
    FIELD_REGISTRY,
    Action,
    ClampRequest,
    ConfigClamp,
    ConfigField,
    ConfigOverride,
    ConfigResolution,
    config_to_dict,
    config_to_toml,
    get_config_field,
    load_config,
    parse_override,
    resolution_to_dict,
    resolve_config,
)
from pyamplicol.reporting import NullProgressSink

MOMENTA: Momenta = (((10.0, 0.0, 0.0, 10.0),),)
PARAMETERS: ModelParameters = {"normalization.alpha_s_me_check": 0.118}


def exercise_generator(artifact: Path) -> None:
    processes = ProcessSet.from_expressions(("d d~ > z",), names=("ddbar_z",))
    assert_type(processes, ProcessSet)
    compiled = ModelSource.built_in_sm().compile(use_cache=False)
    assert_type(compiled, CompiledModel)
    generator = Generator(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(workers=1),
        ),
        progress=NullProgressSink(),
    )
    plan = generator.plan(processes, model=compiled)
    assert_type(plan, GenerationPlan)
    generated = generator.generate(processes, artifact, mode="replace")
    assert_type(generated, GenerationResult)
    assert_type(
        generate(
            ProcessRequest.parse("d d~ > z"),
            artifact,
            model=compiled,
            mode="append",
            config=GenerationConfig(workers=1),
        ),
        GenerationResult,
    )


def exercise_runtime(artifact: Path) -> None:
    runtime = Runtime.load(artifact, model_parameters=PARAMETERS, mute_warnings=True)
    assert_type(runtime, Runtime)
    assert_type(load(artifact, process="ddbar_z"), Runtime)
    physics = runtime.physics
    assert_type(physics, ProcessPhysics)
    assert_type(
        runtime.evaluate(
            MOMENTA,
            helicities=physics.helicities,
            color_flows=physics.color_flows,
        ),
        tuple[complex | Decimal, ...],
    )
    resolved = runtime.evaluate_resolved(MOMENTA, precision=32)
    assert_type(resolved, ResolvedEvaluation)
    assert_type(resolved.total(), tuple[complex | Decimal, ...])
    runtime.set_model_parameters(PARAMETERS)
    runtime.set_model_parameter("normalization.alpha_s_me_check", 0.119)
    runtime.mute_warnings()
    runtime.unmute_warnings()
    assert_type(BenchmarkRunner(BenchmarkConfig()).run(runtime), BenchmarkResult)
    assert_type(benchmark(runtime, points=MOMENTA), BenchmarkResult)


def exercise_configuration(card: Path) -> None:
    generation = GenerationConfig(workers=1)
    evaluation = EvaluationConfig(artifact=card, precision=32, resolved=True)
    benchmark_config = BenchmarkConfig(minimum_samples=3)
    run = RunConfig(
        action=Action.EVALUATE,
        generation=generation,
        evaluation=evaluation,
        benchmark=benchmark_config,
    )
    assert_type(generation, GenerationConfig)
    assert_type(evaluation, EvaluationConfig)
    assert_type(benchmark_config, BenchmarkConfig)
    assert_type(run, RunConfig)
    override = parse_override("generation.workers=2", base_dir=card.parent)
    assert_type(override, ConfigOverride)
    dedicated: Mapping[str, object] = {"generation": {"workers": 2}}
    resolution = resolve_config(
        card,
        action=Action.GENERATE,
        dedicated=dedicated,
        overrides=(override,),
        clamps=(ClampRequest("generation.workers", 1, "resource limit"),),
    )
    assert_type(resolution, ConfigResolution)
    assert_type(load_config(card, overrides=(override,)), RunConfig)
    assert_type(config_to_dict(run), dict[str, object])
    assert_type(config_to_toml(run), str)
    assert_type(resolution_to_dict(resolution), dict[str, object])
    assert_type(get_config_field("generation.workers"), ConfigField)
    assert_type(FIELD_REGISTRY, Mapping[str, ConfigField])
    assert_type(CONFIG_SECTIONS, tuple[str, ...])


def exercise_results(
    plan: GenerationPlan,
    generated: GenerationResult,
    physics: ProcessPhysics,
    resolved: ResolvedEvaluation,
    result: BenchmarkResult,
) -> None:
    assert_type(plan.concrete_processes, tuple[ProcessRequest, ...])
    assert_type(plan.estimated_coverage, Mapping[str, object])
    assert_type(plan.requested_settings, RunConfig | GenerationConfig)
    assert_type(plan.effective_settings, RunConfig | GenerationConfig)
    assert_type(plan.adjustments, tuple[ConfigClamp, ...])
    assert_type(plan.unsupported_features, tuple[str, ...])
    assert_type(generated.output, Path)
    assert_type(generated.processes, ProcessSet)
    assert_type(generated.mode, Literal["error", "append", "replace"])
    assert_type(physics.helicities, tuple[HelicityConfiguration, ...])
    assert_type(physics.color_flows, tuple[ColorFlow, ...])
    assert_type(
        physics.contracted_color_components,
        tuple[ContractedColorComponent, ...],
    )
    assert_type(physics.helicity_ids, tuple[str, ...])
    assert_type(resolved.shape, tuple[int, int, int])
    assert_type(result.uncertainty, BenchmarkStatistics)
    assert_type(result.evaluator_uncertainty, BenchmarkStatistics | None)
    assert_type(result.repetitions_per_sample, int)
    assert_type(result.evaluation_count, int)
    assert_type(result.evaluated_point_count, int)
    assert_type(result.process_id, str | None)
    assert_type(result.process_expression, str | None)
    assert_type(result.timing_breakdown, BenchmarkTimingBreakdown | None)
    if result.timing_breakdown is not None:
        assert_type(
            result.timing_breakdown.stage_evaluator_call_time,
            BenchmarkComponentTiming | None,
        )
        assert_type(result.timing_breakdown.stages, tuple[BenchmarkStageTiming, ...])
        assert_type(
            result.timing_breakdown.stage_backend_call_time,
            BenchmarkComponentTiming | None,
        )
        if result.timing_breakdown.stages:
            assert_type(
                result.timing_breakdown.stages[0].leaf_input_pack_time,
                BenchmarkComponentTiming | None,
            )
        assert_type(
            result.timing_breakdown.counters,
            BenchmarkProfileCounters | None,
        )
        if result.timing_breakdown.counters is not None:
            assert_type(
                result.timing_breakdown.counters.stage_input_copy_components_per_point,
                float | None,
            )
    assert_type(result.environment, Mapping[str, object])


def exercise_protocols(
    generator: GeneratorBackend,
    runtime: RuntimeBackend,
    processes: ProcessSet,
) -> None:
    assert_type(generator.plan(processes), GenerationPlan)
    assert_type(generator.generate(processes, "artifact"), GenerationResult)
    assert_type(runtime.physics, ProcessPhysics)
    assert_type(runtime.evaluate(MOMENTA), Sequence[complex | Decimal])
    assert_type(runtime.evaluate_resolved(MOMENTA), ResolvedEvaluation)
