# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pyamplicol.generation.runtime_schema as runtime_schema
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.physics_metadata import (
    _color_id,
    build_resolved_physics_from_dag,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_public_lc_color_ids_are_physical_and_sector_independent() -> None:
    assert _color_id(()) == "flow:singlet"
    assert _color_id((2, 4, 1)) == "flow:2,4,1"


@pytest.mark.parametrize("color_accuracy", ["lc", "full"])
def test_bounded_public_physics_matches_expanded_schema_without_building_it(
    color_accuracy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy=color_accuracy),
        model=model,
    )
    process_id = f"bounded-{color_accuracy}"
    expanded = runtime_schema.build_runtime_schema(
        dag,
        model,
        process_id=process_id,
    )["physics"]

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded physics called expanded runtime-schema builder")

    monkeypatch.setattr(runtime_schema, "build_runtime_schema_layout", forbidden)
    bounded = build_resolved_physics_from_dag(
        dag,
        model,
        process_id=process_id,
    )

    assert bounded == expanded
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "runtime-physics-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(bounded)
    assert len(bounded["external_particles"]) == len(dag.process.legs)
    assert bounded["helicities"]
    assert bounded["color_components"]
    assert bounded["reduction"]["groups"]
    assert bounded["model_parameters"]
    assert set(bounded["selectors"]) == {
        "helicity",
        "color_flow",
        "contracted_color",
    }
