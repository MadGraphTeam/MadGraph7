# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.generation.contracts import RuntimeExpressionSchema
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.stage_compiler import (
    _fanout_aware_current_order,
    build_generic_stage_compiler_blueprint,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _minimal_schema() -> dict[str, object]:
    return {
        "parameter_layout": {
            "value_component_count": 0,
            "momentum_parameter_count": 0,
            "model_parameter_count": 0,
        },
        "model_parameters": [],
        "current_storage": {"current_slots": []},
        "value_storage": {"value_slots": []},
        "momentum_slots": [],
        "stages": [],
        "amplitude_stage": {"roots": []},
    }


def test_generation_runtime_expression_schema_is_canonical_and_frozen() -> None:
    left = RuntimeExpressionSchema.from_mapping(_minimal_schema())
    right = RuntimeExpressionSchema.from_mapping(
        dict(reversed(tuple(_minimal_schema().items())))
    )
    assert left == right
    assert left.sha256 == right.sha256
    assert left.to_mapping()["stages"] == []


def test_generation_stage_compiler_requires_schema_and_local_parameters() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z"), model=model)
    with pytest.raises(ValueError, match="explicit runtime schema"):
        build_generic_stage_compiler_blueprint(dag)
    with pytest.raises(ValueError, match="stage-local parameter layout"):
        build_generic_stage_compiler_blueprint(
            dag,
            runtime_schema=_minimal_schema(),
            stage_local_parameter_layout=False,
        )


def test_generation_fanout_order_preserves_evaluation_reuse() -> None:
    order, before, after = _fanout_aware_current_order(
        (0, 1, 2, 3),
        output_size_by_current={current_id: 2 for current_id in range(4)},
        evaluation_groups_by_current={
            0: frozenset((1, 10)),
            1: frozenset((2, 20)),
            2: frozenset((1, 30)),
            3: frozenset((2, 40)),
        },
        chunk_size=4,
    )
    assert order in {(0, 2, 1, 3), (1, 3, 0, 2)}
    assert (before, after) == (8, 6)
