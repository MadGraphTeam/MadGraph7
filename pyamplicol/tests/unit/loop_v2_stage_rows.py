# SPDX-License-Identifier: 0BSD
"""Tree-runtime rows of a ``GenericDAG`` for the loop-v2 block and CT oracles.

The interpreted Symbolica stage evaluators of the numerical-current warm-up;
nothing is compiled. Shared by ``test_loop_v2_blocks.py`` and
``test_loop_v2_counterterms.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from typing import Any

Row = tuple[tuple[Decimal, Decimal], ...]


def stage_rows(
    dag: Any,
    model: Any,
    momenta: Sequence[Sequence[Decimal]],
    parameters: Callable[[Sequence[Mapping[str, Any]]], Sequence[Decimal]],
    digits: int,
) -> dict[tuple[int, str], Row]:
    """Every value slot of every current: {(current id, variant): (re, im)s}.

    ``parameters`` maps the runtime model-parameter records, sorted by their
    parameter index, to their values. Every stage input is upcast to
    ``digits``, and the stages are evaluated at that precision.
    """

    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
    from pyamplicol.generation.contracts import StageCompilationInput
    from pyamplicol.generation.numerical_current_warmup import (
        _build_interpreted_stage_evaluator,
        _evaluate_interpreted_stage,
    )
    from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
    from pyamplicol.generation.stage_planning import (
        build_generic_stage_compiler_blueprint,
    )
    from pyamplicol.runtime.symbolica_exact import (
        _fill_momenta,
        _fill_sources,
        _upcast_complex_inputs,
    )

    runtime_schema = build_runtime_expression_schema(
        dag, model, process_id="loop-v2-oracle"
    )
    schema = runtime_schema.to_mapping()
    blueprint = build_generic_stage_compiler_blueprint(
        StageCompilationInput(dag, model, runtime_schema)
    )
    assert blueprint.expression_ready, blueprint.blockers
    settings = SymbolicaEvaluatorSettings(
        iterations=1,
        cpe_iterations=0,
        n_cores=1,
        direct_translation=False,
        jit_direct_translation=False,
        jit_optimization_level=0,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        compiled_chunk_compile_workers=1,
    )
    records = sorted(
        schema["model_parameters"], key=lambda record: record["parameter_index"]
    )
    values = tuple(parameters(records))
    layout = schema["parameter_layout"]
    value_count = int(layout["value_component_count"])
    offset = value_count + int(layout["momentum_parameter_count"])
    momenta = tuple(tuple(momentum) for momentum in momenta)
    state = [(Decimal(0), Decimal(0))] * max(
        int(layout["parameter_count_if_flattened"]), offset + len(values)
    )
    with localcontext() as context:
        context.prec = digits + 10
        _fill_sources(state, momenta, schema, values)
        _fill_momenta(state, momenta, schema)
    for index, value in enumerate(values):
        state[offset + index] = (value, Decimal(0))
    for stage in blueprint.stages:
        evaluator = _build_interpreted_stage_evaluator(
            stage, settings=settings, output_partitions=((0, stage.output_length),)
        )
        inputs = [None] * stage.parameter_count
        for component in stage.input_components:
            inputs[component.parameter_index] = state[component.global_component]
        outputs = _evaluate_interpreted_stage(
            evaluator, _upcast_complex_inputs(tuple(inputs), digits), precision=digits
        )
        for slot in stage.output_slots:
            state[slot.component_start : slot.component_stop] = outputs[
                slot.output_start : slot.output_stop
            ]
    return {
        (int(slot["current_id"]), str(slot["variant"])): tuple(
            state[slot["component_start"] : slot["component_stop"]]
        )
        for slot in schema["value_storage"]["value_slots"]
    }
