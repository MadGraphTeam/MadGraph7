# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.api import ProcessRequest, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import ColorConfig, EvaluatorConfig, RunConfig
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _backend(*, accuracy: str = "lc") -> GenerationBackend:
    return GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy=accuracy),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_recurrence_rejects_contracted_color_before_process_expansion(
    accuracy: str,
) -> None:
    processes = ProcessSet((ProcessRequest.parse("d d~ > z g"),))

    with pytest.raises(GenerationError, match="available only for LC generation"):
        _backend(accuracy=accuracy).plan(processes)


def test_recurrence_lc_is_fail_closed_before_generic_dag_construction() -> None:
    backend = _backend()
    process = build_process_ir("d d~ > z g")

    with pytest.raises(GenerationError, match="recurrence-template-v1"):
        backend._compile_concrete_process(process, BuiltinSMModel())
