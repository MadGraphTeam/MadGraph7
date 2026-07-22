# SPDX-License-Identifier: 0BSD
"""Prepared-kernel numerical canary for compact recurrence execution."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import _rusticol
from pyamplicol.color.plan import (
    build_color_plan,
    build_lc_topology_replay_plan,
)
from pyamplicol.config import EvaluatorConfig
from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceNormalizationV1,
    build_recurrence_builder_input_v1,
)
from pyamplicol.generation.recurrence_projection import (
    project_recurrence_process_v1,
)
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

_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (
        504.1576256720017,
        270.45289818999487,
        -290.90819815599644,
        -296.79506445608888,
    ),
    (
        495.8423743279983,
        -270.45289818999214,
        290.90819815599195,
        296.79506445609007,
    ),
)
_HELICITY = (-1, 1, -1, -1)
_EXPECTED_RAW_AMPLITUDE = complex(
    -0.003152611916230649642,
    0.002930935033859867571,
)
_EXPECTED_NORMALIZED_COMPONENT = abs(_EXPECTED_RAW_AMPLITUDE) ** 2 * 0.2812502450332261
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
            if isinstance(raw_value, tuple):
                values_by_name[str(name)] = complex(*raw_value)
            else:
                values_by_name[str(name)] = complex(raw_value)

    prepared_count = len(catalog.parameters)
    result = [(0.0, 0.0)] * prepared_count
    for parameter in catalog.parameters:
        prepared_id = parameter.prepared_parameter_id
        if prepared_id is None:
            continue
        assert prepared_id < prepared_count
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


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_prepared_jit_recurrence_matches_compiled_raw_amplitude(
    tmp_path,
    model_source: str,
) -> None:
    """Lock crossing and terminal-current semantics against a compiled oracle."""

    if model_source == "built-in":
        compiled_model = compile_model_source("built-in-sm", use_cache=True)
        model = BuiltinSMModel()
        process = build_process_ir("d d~ > z g", color_accuracy="lc")
    else:
        compiled_model = compile_model_source(
            _UFO_SM_ROOT / "sm.json",
            restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
            use_cache=True,
        )
        model = CompiledUFOModel(compiled_model)
        process = build_model_process_ir("d d~ > z g", compiled_model.ir)
    prepared = prepare_model_bundle(
        compiled_model,
        tmp_path / f"{model_source}-jit-o3",
        evaluator=EvaluatorConfig(),
    )
    bundle = prepared.bundle
    catalog = bundle.kernel_pack.recurrence_template_catalog
    assert catalog is not None
    payload_root = tmp_path / "prepared-payloads"
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
            "prepared-jit-regression-v1",
            "d" * 64,
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

    # GenericDAG is used only to obtain an independent compiled-mode source-fill
    # oracle for this test. Recurrence generation itself never constructs it.
    oracle_dag = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        max_coupling_orders=limits,
    )
    runtime_schema = build_runtime_expression_schema(
        oracle_dag,
        model,
    ).to_mapping()
    point = tuple(tuple(Decimal(str(value)) for value in row) for row in _POINT)
    parameter_defaults = tuple(
        Decimal(str(item["default"])) for item in runtime_schema["model_parameters"]
    )
    prepared_parameter_defaults = _prepared_parameter_defaults(model, catalog)
    source_records = runtime_schema["source_fill"]["sources"]
    source_component_count = sum(
        int(item["component_count"]) for item in source_layout
    )
    source_values = [(0.0, 0.0)] * source_component_count
    all_source_values = [(0.0, 0.0)] * source_component_count
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
            point,
            runtime_schema,
            parameter_defaults,
        )
        start = int(native["component_start"])
        for component, (real, imaginary) in enumerate(wavefunction):
            value = (float(real), float(imaginary))
            all_source_values[start + component] = value
            if process_state.public_helicity == _HELICITY[source_slot]:
                source_values[start + component] = value

    manifest = bundle.kernel_pack.to_dict()
    manifest.pop("recurrence_template", None)
    manifest["eager_kernel_abi"] = bundle.manifest["eager_kernel_abi"]
    result = _rusticol._evaluate_recurrence_one_helicity_v1(
        builder_input,
        template_input,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
        catalog.header.prepared_kernel_pack_digest,
        payload_root,
        source_values,
        [component for row in _POINT for component in row],
        prepared_parameter_defaults,
        point_count=1,
        point_tile_size=1,
    )
    all_result = _rusticol._evaluate_recurrence_all_helicities_v1(
        builder_input,
        template_input,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
        catalog.header.prepared_kernel_pack_digest,
        payload_root,
        all_source_values,
        [component for row in _POINT for component in row],
        prepared_parameter_defaults,
        point_count=1,
        point_tile_size=1,
    )

    assert result["sector_count"] == 1
    amplitude = complex(*result["amplitudes"][0])
    normalized_component = abs(amplitude) ** 2 * float(
        runtime_schema["normalization"]["global_coupling_factor"]
    )
    assert normalized_component == pytest.approx(
        _EXPECTED_NORMALIZED_COMPONENT,
        rel=1e-12,
        abs=1e-15,
    )
    if model_source == "built-in":
        assert amplitude.real == pytest.approx(
            _EXPECTED_RAW_AMPLITUDE.real,
            rel=1e-12,
            abs=1e-15,
        )
        assert amplitude.imag == pytest.approx(
            _EXPECTED_RAW_AMPLITUDE.imag,
            rel=1e-12,
            abs=1e-15,
        )
    assert all_result["resolved"] is True
    assert len(all_result["resolved_helicities"]) == validation["inspection_summary"][
        "schedule"
    ]["resolved_helicity_count"]
    assert len(all_result["amplitude_destinations"]) == validation[
        "inspection_summary"
    ]["schedule"]["amplitude_destination_count"]
    selected_helicity_id = all_result["resolved_helicities"].index(list(_HELICITY))
    selected_destination = next(
        destination
        for destination in all_result["amplitude_destinations"]
        if destination["target_sector_id"] == 0
        and destination["target_helicity_id"] == selected_helicity_id
    )
    selected_amplitude = complex(
        *all_result["amplitudes"][selected_destination["id"]]
    )
    assert selected_amplitude.real == pytest.approx(
        amplitude.real,
        rel=1e-12,
        abs=1e-15,
    )
    assert selected_amplitude.imag == pytest.approx(
        amplitude.imag,
        rel=1e-12,
        abs=1e-15,
    )
    assert (
        sum(abs(complex(*value)) for value in all_result["amplitudes"])
        > abs(amplitude)
    )

    representative_indices = {
        0,
        selected_helicity_id,
        len(all_result["resolved_helicities"]) - 1,
    }
    for helicity_index in sorted(representative_indices):
        helicity = all_result["resolved_helicities"][helicity_index]
        sparse_values = [(0.0, 0.0)] * source_component_count
        for native in source_layout:
            source_slot = int(native["source_slot"])
            process_state = next(
                state
                for state in logical.external_legs[source_slot].source_states
                if state.source_template_id == int(native["source_template_id"])
            )
            if process_state.public_helicity != helicity[source_slot]:
                continue
            start = int(native["component_start"])
            stop = start + int(native["component_count"])
            sparse_values[start:stop] = all_source_values[start:stop]
        separate = _rusticol._evaluate_recurrence_one_helicity_v1(
            builder_input,
            template_input,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
            catalog.header.prepared_kernel_pack_digest,
            payload_root,
            sparse_values,
            [component for row in _POINT for component in row],
            prepared_parameter_defaults,
            point_count=1,
            point_tile_size=1,
        )
        separate_amplitude = complex(*separate["amplitudes"][0])
        destination = next(
            row
            for row in all_result["amplitude_destinations"]
            if row["target_sector_id"] == 0
            and row["target_helicity_id"] == helicity_index
        )
        combined_amplitude = complex(*all_result["amplitudes"][destination["id"]])
        assert combined_amplitude.real == pytest.approx(
            separate_amplitude.real,
            rel=1e-12,
            abs=1e-15,
        )
        assert combined_amplitude.imag == pytest.approx(
            separate_amplitude.imag,
            rel=1e-12,
            abs=1e-15,
        )


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_topology_replay_helicity_sum_matches_independent_two_flow_oracle(
    tmp_path: Path,
    model_source: str,
) -> None:
    """Replay every public flow and compare with the legacy generated-library oracle."""

    expression = "d d~ > z g g"
    if model_source == "built-in":
        compiled_model = compile_model_source("built-in-sm", use_cache=True)
        model = BuiltinSMModel()
        process = build_process_ir(expression, color_accuracy="lc")
    else:
        compiled_model = compile_model_source(
            _UFO_SM_ROOT / "sm.json",
            restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
            use_cache=True,
        )
        model = CompiledUFOModel(compiled_model)
        process = build_model_process_ir(expression, compiled_model.ir)
    prepared = prepare_model_bundle(
        compiled_model,
        tmp_path / f"{model_source}-zgg-jit-o3",
        evaluator=EvaluatorConfig(),
    )
    bundle = prepared.bundle
    catalog = bundle.kernel_pack.recurrence_template_catalog
    assert catalog is not None
    payload_root = tmp_path / f"{model_source}-zgg-prepared-payloads"
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
            "prepared-jit-two-flow-regression-v1",
            "e" * 64,
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

    oracle_dag = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        max_coupling_orders=limits,
    )
    runtime_schema = build_runtime_expression_schema(oracle_dag, model).to_mapping()
    final_state = massive_rambo_final_state(
        3,
        sqrt_s=1000.0,
        masses=(91.188, 0.0, 0.0),
        seed=731,
    )
    point_float = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        *final_state,
    )
    point = tuple(tuple(Decimal(str(value)) for value in row) for row in point_float)
    parameter_defaults = tuple(
        Decimal(str(item["default"])) for item in runtime_schema["model_parameters"]
    )
    prepared_parameter_defaults = _prepared_parameter_defaults(model, catalog)
    source_records = runtime_schema["source_fill"]["sources"]
    source_component_count = sum(int(item["component_count"]) for item in source_layout)
    all_source_values = [(0.0, 0.0)] * source_component_count
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
            point,
            runtime_schema,
            parameter_defaults,
        )
        start = int(native["component_start"])
        for component, (real, imaginary) in enumerate(wavefunction):
            all_source_values[start + component] = (float(real), float(imaginary))

    manifest = bundle.kernel_pack.to_dict()
    manifest.pop("recurrence_template", None)
    manifest["eager_kernel_abi"] = bundle.manifest["eager_kernel_abi"]
    result = _rusticol._evaluate_recurrence_helicity_sum_norm_sqr_v1(
        builder_input,
        template_input,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
        catalog.header.prepared_kernel_pack_digest,
        payload_root,
        all_source_values,
        [component for row in point_float for component in row],
        prepared_parameter_defaults,
        point_count=1,
        point_tile_size=1,
    )

    normalization_payload = runtime_schema["normalization"]
    normalization = (
        float(normalization_payload["global_coupling_factor"])
        * float(normalization_payload["color_factor"])
        / (
            float(normalization_payload["average_factor"])
            * float(normalization_payload["identical_factor"])
        )
    )
    sector_by_flow = {
        "flow:" + ",".join(str(label) for label in sector.word_labels): int(sector.id)
        for sector in color_plan.sectors
    }
    normalized = {
        flow_id: result["norm_sqr"][sector_id] * normalization
        for flow_id, sector_id in sector_by_flow.items()
    }
    assert result["sector_count"] == 2
    # The restricted UFO-SM input carries the same established 2.81e-12
    # parameter-rounding offset from the legacy oracle as compiled mode.  The
    # stricter recurrence-versus-compiled check above remains at 1e-12.
    legacy_relative_tolerance = 4e-12 if model_source == "ufo-sm" else 1e-12
    assert normalized == pytest.approx(
        {
            "flow:2,4,5,1": 2.2077590207957575e-7,
            "flow:2,5,4,1": 4.7670411360157456e-2,
        },
        rel=legacy_relative_tolerance,
        abs=1e-15,
    )
