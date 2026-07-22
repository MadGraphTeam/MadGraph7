# SPDX-License-Identifier: 0BSD
"""Three-open-quark-line recurrence execution against an AmpliCol oracle."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime, _rusticol
from pyamplicol.color.plan import build_color_plan, build_lc_topology_replay_plan
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.dag_algorithms import infer_minimal_coupling_order_limits
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceNormalizationV1,
    build_recurrence_builder_input_v1,
)
from pyamplicol.generation.recurrence_projection import project_recurrence_process_v1
from pyamplicol.generation.recurrence_template_columnar import (
    build_recurrence_template_input_v1,
)
from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared_compile import prepare_model_bundle
from pyamplicol.processes.model import build_model_process_ir
from pyamplicol.runtime.symbolica_exact import _source_wavefunction

_EXPRESSION = "d d~ > u u~ s s~"
_LEGACY_HELICITY_SUM_BY_FLOW = {
    "flow:2,1,3,4,5,6": 1.7260373034739047e-11,
    "flow:2,1,3,6,5,4": 1.6570372188601389e-11,
    "flow:2,4,3,1,5,6": 1.0167883928661888e-10,
    "flow:2,4,3,6,5,1": 1.6445040193096446e-10,
    "flow:2,6,3,1,5,4": 6.5052811045824496e-10,
    "flow:2,6,3,4,5,1": 1.2411150983558527e-10,
}
_LEGACY_HELICITY_SUM_TOTAL = 1.0745996067347547e-9
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


def _prepared_parameter_defaults(model, catalog) -> list[tuple[float, float]]:
    values_by_name: dict[str, complex] = {}
    for provider_name in (
        "runtime_parameter_defaults",
        "runtime_derived_parameter_defaults",
    ):
        provider = getattr(model, provider_name, None)
        if not callable(provider):
            continue
        for name, raw_value in provider().items():
            values_by_name[str(name)] = (
                complex(*raw_value)
                if isinstance(raw_value, tuple)
                else complex(raw_value)
            )

    result = [(0.0, 0.0)] * len(catalog.parameters)
    for parameter in catalog.parameters:
        prepared_id = parameter.prepared_parameter_id
        if prepared_id is None:
            continue
        value = values_by_name.get(parameter.name)
        if value is None and parameter.default_value is not None:
            default = parameter.default_value
            value = complex(
                default.real_numerator / default.real_denominator,
                default.imag_numerator / default.imag_denominator,
            )
        assert value is not None, f"missing prepared default for {parameter.name}"
        result[prepared_id] = (value.real, value.imag)
    return result


def _normalization_factor(payload: dict[str, object]) -> float:
    return (
        float(payload["global_coupling_factor"])
        * float(payload["color_factor"])
        / (float(payload["average_factor"]) * float(payload["identical_factor"]))
    )


def _model_context(model_source: str):
    if model_source == "built-in":
        compiled_model = compile_model_source("built-in-sm", use_cache=True)
        return (
            compiled_model,
            BuiltinSMModel(),
            None,
            build_process_ir(
                _EXPRESSION,
                color_accuracy="lc",
            ),
        )

    model_path = _UFO_SM_ROOT / "sm.json"
    restriction_path = _UFO_SM_ROOT / "restrict_default.json"
    compiled_model = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=True,
    )
    return (
        compiled_model,
        CompiledUFOModel(compiled_model),
        ModelSource.from_path(model_path, restriction=restriction_path),
        build_model_process_ir(_EXPRESSION, compiled_model.ir),
    )


def _compiled_helicity_sum_by_flow(
    artifact: Path,
    point: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, float], dict[str, object]]:
    resolved = Runtime.load(artifact).evaluate_resolved((point,))
    values = {
        flow_id: float(
            sum(
                resolved.values[0][helicity_index][color_index].real
                for helicity_index in range(len(resolved.helicity_ids))
            )
        )
        for color_index, flow_id in enumerate(resolved.color_ids)
    }
    physics_files = tuple((artifact / "processes").glob("*/physics.json"))
    assert len(physics_files) == 1
    physics = json.loads(physics_files[0].read_text(encoding="utf-8"))
    normalization = physics["extensions"]["normalization"]
    assert isinstance(normalization, dict)
    return values, normalization


def _recurrence_helicity_sum_by_flow(
    *,
    tmp_path: Path,
    model_source: str,
    compiled_model,
    model,
    process,
    point: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, float], dict[str, object], tuple[str, ...]]:
    evaluator = EvaluatorConfig(jit=JITConfig(optimization_level=1))
    prepared = prepare_model_bundle(
        compiled_model,
        tmp_path / f"{model_source}-recurrence-jit-o1",
        evaluator=evaluator,
    )
    bundle = prepared.bundle
    catalog = bundle.kernel_pack.recurrence_template_catalog
    assert catalog is not None
    payload_root = tmp_path / f"{model_source}-recurrence-payloads"
    bundle.copy_referenced_payloads(payload_root)

    color_plan = build_color_plan(
        process,
        color_accuracy="lc",
        fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(process),
    )
    replay = build_lc_topology_replay_plan(color_plan, model)
    limits = infer_minimal_coupling_order_limits(process, model=model)
    logical = project_recurrence_process_v1(
        process,
        color_plan,
        catalog,
        layout="topology-replay",
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(1),
            "three-line-compiled-jit-o1-oracle-v1",
            "3" * 64,
        ),
        topology_replay=replay,
        coupling_order_limits=limits,
    )
    builder_input = build_recurrence_builder_input_v1(logical)
    template_input = build_recurrence_template_input_v1(catalog)
    validation = _rusticol._validate_recurrence_builder_input_v1(
        builder_input,
        template_input,
        construct_schedule=True,
    )
    source_layout = validation["inspection_summary"]["schedule"]["source_layout"]

    # The GenericDAG is used only to source independent wavefunction records.
    # Recurrence construction and execution use the compact pre-DAG schedule.
    oracle_dag = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        max_coupling_orders=limits,
    )
    runtime_schema = build_runtime_expression_schema(oracle_dag, model).to_mapping()
    decimal_point = tuple(
        tuple(Decimal(str(component)) for component in momentum) for momentum in point
    )
    parameter_defaults = tuple(
        Decimal(str(item["default"])) for item in runtime_schema["model_parameters"]
    )
    source_records = runtime_schema["source_fill"]["sources"]
    source_component_count = sum(int(item["component_count"]) for item in source_layout)
    source_values = [(0.0, 0.0)] * source_component_count
    for native in source_layout:
        source_slot = int(native["source_slot"])
        process_state = next(
            state
            for state in logical.external_legs[source_slot].source_states
            if state.source_template_id == int(native["source_template_id"])
        )
        record = next(
            item
            for item in source_records
            if int(item["leg_label"]) == source_slot + 1
            and int(item["source_helicity"]) == process_state.public_helicity
            and int(item["chirality"]) == process_state.chirality
            and int(item["spin_state"]) == process_state.spin_state
        )
        wavefunction = _source_wavefunction(
            record,
            decimal_point,
            runtime_schema,
            parameter_defaults,
        )
        start = int(native["component_start"])
        for component, (real, imaginary) in enumerate(wavefunction):
            source_values[start + component] = (float(real), float(imaginary))

    manifest = bundle.kernel_pack.to_dict()
    manifest.pop("recurrence_template", None)
    manifest["eager_kernel_abi"] = bundle.manifest["eager_kernel_abi"]
    result = _rusticol._evaluate_recurrence_helicity_sum_norm_sqr_v1(
        builder_input,
        template_input,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
        catalog.header.prepared_kernel_pack_digest,
        payload_root,
        source_values,
        [component for momentum in point for component in momentum],
        _prepared_parameter_defaults(model, catalog),
        point_count=1,
        point_tile_size=1,
    )

    normalization = runtime_schema["normalization"]
    factor = _normalization_factor(normalization)
    sector_by_flow = {
        "flow:" + ",".join(str(label) for label in sector.word_labels): int(sector.id)
        for sector in color_plan.sectors
    }
    return (
        {
            flow_id: float(result["norm_sqr"][sector_id]) * factor
            for flow_id, sector_id in sector_by_flow.items()
        },
        normalization,
        tuple(sector_by_flow),
    )


def test_three_line_topology_replay_matches_amplicol_and_compiled_per_flow(
    tmp_path: Path,
) -> None:
    """Compare both SM frontends to AmpliCol and fresh compiled artifacts."""

    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        *massive_rambo_final_state(
            4,
            sqrt_s=1000.0,
            masses=(0.0, 0.0, 0.0, 0.0),
            seed=731,
        ),
    )
    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(jit=JITConfig(optimization_level=1))
    by_model: dict[str, dict[str, float]] = {}
    mismatches: dict[str, dict[str, dict[str, float]]] = {}

    for model_source in ("built-in", "ufo-sm"):
        compiled_model, model, public_model, process = _model_context(model_source)
        artifact = tmp_path / f"{model_source}-compiled-jit-o1"
        Generator(
            RunConfig(
                action="generate",
                color=ColorConfig(accuracy="lc"),
                generation=generation,
                evaluator=evaluator,
            )
        ).generate(_EXPRESSION, artifact, model=public_model)
        compiled, artifact_normalization = _compiled_helicity_sum_by_flow(
            artifact,
            point,
        )
        recurrence, recurrence_normalization, recurrence_flow_ids = (
            _recurrence_helicity_sum_by_flow(
                tmp_path=tmp_path,
                model_source=model_source,
                compiled_model=compiled_model,
                model=model,
                process=process,
                point=point,
            )
        )

        assert tuple(compiled) == recurrence_flow_ids
        for key in (
            "global_coupling_factor",
            "color_factor",
            "average_factor",
            "identical_factor",
        ):
            assert float(artifact_normalization[key]) == float(
                recurrence_normalization[key]
            )
        assert _normalization_factor(artifact_normalization) == (
            _normalization_factor(recurrence_normalization)
        )
        model_mismatches: dict[str, dict[str, float]] = {}
        for flow_id, expected in compiled.items():
            obtained = recurrence[flow_id]
            absolute = abs(obtained - expected)
            relative = absolute / abs(expected) if expected else float("inf")
            if absolute > max(1e-15, 1e-12 * abs(expected)):
                model_mismatches[flow_id] = {
                    "recurrence": obtained,
                    "compiled_jit_o1": expected,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
        if model_mismatches:
            mismatches[model_source] = model_mismatches
        assert recurrence == pytest.approx(
            _LEGACY_HELICITY_SUM_BY_FLOW,
            rel=5e-12,
            abs=1e-15,
        )
        assert sum(recurrence.values()) == pytest.approx(
            _LEGACY_HELICITY_SUM_TOTAL,
            rel=5e-12,
            abs=1e-15,
        )
        by_model[model_source] = recurrence

    assert by_model["ufo-sm"] == pytest.approx(
        by_model["built-in"],
        rel=5e-12,
        abs=1e-15,
    )
    assert not mismatches, json.dumps(mismatches, indent=2, sort_keys=True)
