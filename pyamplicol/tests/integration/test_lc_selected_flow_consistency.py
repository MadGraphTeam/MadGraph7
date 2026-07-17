# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pyamplicol import Generator, Runtime
from pyamplicol.config import (
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state


@pytest.mark.parametrize(
    ("process", "flow_word", "final_masses"),
    (
        ("g g > g g", (1, 2, 3, 4), (0.0, 0.0)),
        ("g g > t t~", (3, 1, 2, 4), (173.0, 173.0)),
        ("d d~ > t t~", (2, 1, 3, 4), (173.0, 173.0)),
    ),
)
def test_selected_lc_artifact_matches_complete_artifact_physical_flow(
    tmp_path: Path,
    process: str,
    flow_word: tuple[int, ...],
    final_masses: tuple[float, float],
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(jit=JITConfig(optimization_level=1))
    complete_path = tmp_path / "complete"
    selected_path = tmp_path / "selected"

    Generator(
        RunConfig(
            action="generate",
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, complete_path)
    Generator(
        RunConfig(
            action="generate",
            process=ProcessConfig(
                reference_color_order=flow_word,
                selected_color_sector_ids=(0,),
            ),
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, selected_path)

    final_state = massive_rambo_final_state(
        2,
        sqrt_s=1000.0,
        masses=final_masses,
        seed=731,
    )
    momenta = (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *final_state,
        ),
    )
    flow_id = "flow:" + ",".join(str(label) for label in flow_word)
    complete = Runtime.load(complete_path)
    selected = Runtime.load(selected_path)

    assert flow_id in {flow.id for flow in complete.physics.color_flows}
    assert tuple((flow.id, flow.word) for flow in selected.physics.color_flows) == (
        (flow_id, flow_word),
    )

    complete_component = complete.evaluate_resolved(
        momenta,
        color_flows=[flow_id],
    ).total()[0]
    selected_total = selected.evaluate(momenta)[0]
    assert selected_total.real == pytest.approx(
        complete_component.real,
        rel=1.0e-11,
        abs=1.0e-13,
    )
    assert selected_total.imag == pytest.approx(
        complete_component.imag,
        rel=1.0e-11,
        abs=1.0e-13,
    )

