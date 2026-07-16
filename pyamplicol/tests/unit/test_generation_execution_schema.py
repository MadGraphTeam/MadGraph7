# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from pyamplicol.generation.artifact_writer import _execution_plan
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "models"


def test_execution_plan_is_strict_schema_v3_runtime_dto() -> None:
    schema = build_runtime_schema(
        compile_generic_dag("d d~ > z"),
        BuiltinSMModel(),
        process_id="ddbar_z",
    )

    plan = _execution_plan(schema)

    assert set(plan) == {
        "amplitude_stage",
        "current_storage",
        "external_particles",
        "kind",
        "model",
        "model_parameters",
        "momentum_slots",
        "normalization",
        "parameter_layout",
        "process",
        "process_key",
        "schema_version",
        "source_fill",
        "stages",
        "value_storage",
    }
    assert plan["schema_version"] == 3
    assert plan["kind"] == "pyamplicol-runtime-execution-plan"
    assert plan["process_key"] == "ddbar_z"
    assert "physics" not in plan
    assert "momentum_conventions" not in plan

    layout = plan["parameter_layout"]
    assert layout["parameter_count_if_flattened"] == (
        layout["source_component_parameter_count"]
        + layout["momentum_parameter_count"]
        + layout["model_parameter_count"]
    )
    assert layout["real_valued_inputs"] == list(
        range(
            layout["source_component_parameter_count"],
            layout["parameter_count_if_flattened"],
        )
    )

    for stage in plan["stages"]:
        assert "input_momentum_slot_ids" not in stage
        assert "interaction_evaluation_count" not in stage
        for interaction in stage["interactions"]:
            assert "coupling_parameter_names" not in interaction
            assert "evaluation_group_id" not in interaction
            assert "evaluation_factor" not in interaction

    amplitude = plan["amplitude_stage"]
    assert "coherent_groups" not in amplitude
    assert "final_reduction" not in amplitude
    for root in amplitude["roots"]:
        assert "dag_root_id" not in root
        assert "coupling_parameter_names" not in root


def test_external_particle_masses_link_to_runtime_parameters() -> None:
    compiled = compile_model_source(
        MODEL_ROOT / "json" / "sm" / "sm.json",
        restriction=str(
            (MODEL_ROOT / "json" / "sm" / "restrict_default.json").resolve()
        ),
        use_cache=False,
    )
    model = CompiledUFOModel(compiled)

    z_schema = build_runtime_schema(
        compile_generic_dag("d d~ > z", model=model),
        model,
        process_id="ddbar_z_external",
    )
    z_plan = _execution_plan(z_schema)
    z_record = next(
        particle for particle in z_plan["model"]["particles"] if particle["pdg"] == 23
    )
    assert z_record["mass_parameter"] == "MZ"

    w_schema = build_runtime_schema(
        compile_generic_dag("u d~ > w+", model=model),
        model,
        process_id="udbar_wp_external",
    )
    w_plan = _execution_plan(w_schema)
    w_record = next(
        particle for particle in w_plan["model"]["particles"] if particle["pdg"] == 24
    )
    assert w_record["mass_parameter"] == "MW"
    derived = {
        record["runtime_name"]
        for record in w_plan["model_parameters"]
        if record["kind"] == "derived_parameter_component"
    }
    assert "MW" in derived
    assert "MZ" in model.runtime_derived_parameter_definitions_for(("MW",))["MW"]
