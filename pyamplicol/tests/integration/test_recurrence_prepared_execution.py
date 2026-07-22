# SPDX-License-Identifier: 0BSD
"""Prepared-kernel numerical canary for compact recurrence execution."""

from __future__ import annotations

import json
from decimal import Decimal

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
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared_compile import prepare_model_bundle
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


def test_prepared_jit_recurrence_matches_compiled_raw_amplitude(tmp_path) -> None:
    """Lock crossing and terminal-current semantics against a compiled oracle."""

    compiled_model = compile_model_source("built-in-sm", use_cache=True)
    prepared = prepare_model_bundle(
        compiled_model,
        tmp_path / "built-in-sm-jit-o3",
        evaluator=EvaluatorConfig(),
    )
    bundle = prepared.bundle
    catalog = bundle.kernel_pack.recurrence_template_catalog
    assert catalog is not None
    payload_root = tmp_path / "prepared-payloads"
    bundle.copy_referenced_payloads(payload_root)

    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g", color_accuracy="lc")
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
    source_records = runtime_schema["source_fill"]["sources"]
    source_values = [(0.0, 0.0)] * sum(
        int(item["component_count"]) for item in source_layout
    )
    for source_slot, public_helicity in enumerate(_HELICITY):
        process_state = next(
            state
            for state in logical.external_legs[source_slot].source_states
            if state.public_helicity == public_helicity
        )
        native = next(
            item
            for item in source_layout
            if int(item["source_slot"]) == source_slot
            and int(item["source_template_id"]) == process_state.source_template_id
        )
        record = next(
            item
            for item in source_records
            if int(item["leg_label"]) == source_slot + 1
            and int(item["source_helicity"]) == public_helicity
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
            source_values[start + component] = (float(real), float(imaginary))

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
        [(0.0, 0.0)] * len(catalog.parameters),
        point_count=1,
        point_tile_size=1,
    )

    assert result["sector_count"] == 1
    amplitude = complex(*result["amplitudes"][0])
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
