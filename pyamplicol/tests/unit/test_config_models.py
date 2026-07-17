# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from pyamplicol.config import (
    FIELD_REGISTRY,
    Action,
    BenchmarkConfig,
    ColorAccuracy,
    ColorConfig,
    ConfigurationError,
    CppConfig,
    EvaluationConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ModelConfig,
    OutputConfig,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
    SymbolicaConfig,
)


def test_schema_v1_registry_contains_every_contract_leaf() -> None:
    assert len(FIELD_REGISTRY) == 63
    assert "evaluator.jit.direct_translation" not in FIELD_REGISTRY
    assert FIELD_REGISTRY["action"].required
    assert FIELD_REGISTRY["generation.workers"].default == "auto"
    assert FIELD_REGISTRY["evaluator.jit.optimization_level"].choices == (
        0,
        1,
        2,
        3,
    )
    assert FIELD_REGISTRY["process.multiparticles"].dynamic_kind == "list_str"
    assert FIELD_REGISTRY["process.entries"].kind == "process_entries"
    assert FIELD_REGISTRY["process.reference_color_order"].kind == "list_int"
    assert FIELD_REGISTRY["process.selected_color_sector_ids"].kind == "list_int"
    assert FIELD_REGISTRY["process.selected_source_helicities"].dynamic_kind == "int"
    assert "process.requests" not in FIELD_REGISTRY
    assert "process.names" not in FIELD_REGISTRY
    assert "evaluator.stage_local_parameter_layout" not in FIELD_REGISTRY
    assert {
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    }.isdisjoint(FIELD_REGISTRY)


def test_all_public_configuration_dataclasses_are_frozen() -> None:
    classes = (
        BenchmarkConfig,
        ColorConfig,
        CppConfig,
        EvaluationConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        ModelConfig,
        OutputConfig,
        ProcessConfig,
        ProcessEntry,
        RunConfig,
        SymbolicaConfig,
    )
    assert all(is_dataclass(cls) and cls.__dataclass_params__.frozen for cls in classes)
    config = RunConfig(action="generate")
    with pytest.raises(FrozenInstanceError):
        config.action = "evaluate"  # type: ignore[misc]


def test_configuration_collections_are_immutable() -> None:
    entry = ProcessEntry(expression="d d~ > z g", name="ddbar_zg")
    process = ProcessConfig(
        entries=(entry,),
        multiparticles={"j": ["u", "d", "g"]},
        max_coupling_orders={"QCD": 2},
    )
    assert process.entries == (entry,)
    assert process.multiparticles["j"] == ("u", "d", "g")
    with pytest.raises(FrozenInstanceError):
        entry.name = "renamed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        process.max_coupling_orders["QED"] = 1  # type: ignore[index]


def test_process_entries_validate_names_and_uniqueness() -> None:
    with pytest.raises(ConfigurationError, match="must start with a letter"):
        ProcessEntry(expression="d d~ > z", name="1-invalid")
    with pytest.raises(ConfigurationError, match="duplicates: 'same'"):
        ProcessConfig(
            entries=(
                ProcessEntry("d d~ > z", "same"),
                ProcessEntry("u u~ > z", "same"),
            )
        )


def test_schema_and_jit_levels_reject_boolean_integers() -> None:
    with pytest.raises(ConfigurationError, match="schema_version"):
        RunConfig(action="generate", schema_version=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="optimization_level"):
        JITConfig(optimization_level=True)  # type: ignore[arg-type]


def test_contract_defaults_are_typed() -> None:
    config = RunConfig(action="evaluate")
    assert config.action is Action.EVALUATE
    assert config.color.accuracy is ColorAccuracy.LC
    assert config.evaluator.backend is EvaluatorBackend.JIT
    assert config.schema_version == 1
    assert config.generation.validation.samples == 10
    assert config.evaluator.output_chunk_size == 128
    assert not config.evaluator.cpp.native_arch
    assert config.evaluator.optimization.max_common_pair_cache_entries == 5_000_000
    assert config.benchmark.target_runtime == 10.0
    assert config.benchmark.precision == 16
    assert config.output == OutputConfig()


def test_cpp_flags_cannot_hide_target_cpu_requirements() -> None:
    assert CppConfig(extra_flags=("-fno-math-errno",)).extra_flags == (
        "-fno-math-errno",
    )
    with pytest.raises(ConfigurationError, match="unrecorded target CPU"):
        CppConfig(extra_flags=("-march=native",))
    with pytest.raises(ConfigurationError, match="unsupported compiler arguments"):
        CppConfig(extra_flags=("-mavx2",))
