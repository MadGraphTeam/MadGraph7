# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import load_manifest
from pyamplicol.config import Action, EvaluatorConfig, GenerationConfig, RunConfig
from pyamplicol.generation.artifact_writer import (
    EagerProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.eager_lowering import (
    MappingEagerKernelResolver,
    lower_eager_execution_tables,
)
from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
from pyamplicol.generation.validation import build_validation_point
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    write_prepared_model_bundle,
)


def _prepared_model(
    tmp_path: Path,
    *,
    kernel_ids: tuple[int, ...],
) -> tuple[Path, object]:
    source_model = compile_model_source("built-in-sm", use_cache=False)
    kernels = tuple(
        PreparedKernelRecord(
            kernel_id=kernel_id,
            contract_kind="vertex" if kernel_id < 1000 else "propagator",
            canonical_signature=f"test:eager-kernel:{kernel_id}",
            input_arity=1,
            output_arity=1,
            input_layout=("input",),
            input_contracts=(
                {
                    "role": "current",
                    "component": 0,
                    "symbol": "pyamplicol::input",
                    "model_parameter_name": None,
                    "model_parameter_index": None,
                },
            ),
            output_layout=("output",),
            exact_expressions=("pyamplicol::input",),
            exact_evaluator_state_path=f"kernels/{kernel_id}/exact.bin",
            f64_evaluator_manifest={
                "kind": "symjit-application-evaluator",
                "input_len": 1,
                "output_len": 1,
                "application_path": f"kernels/{kernel_id}/application.symjit",
                "evaluator_state_path": f"kernels/{kernel_id}/exact.bin",
            },
        )
        for kernel_id in kernel_ids
    )
    pack = PreparedKernelPack(
        backend="jit",
        optimization_settings={"optimization_level": 3},
        producer={"distribution": "pyamplicol", "version": "test"},
        dependency_abis={"symjit_application": "test-v1"},
        provenance={"compiled_model": "test"},
        target={
            "portable": False,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "test-target",
            "cpu_features": [],
        },
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "built-in-sm",
        },
        kernels=kernels,
    )
    payloads = {
        path: f"payload:{path}".encode()
        for kernel in kernels
        for path in kernel.referenced_payload_paths
    }
    bundle_path = write_prepared_model_bundle(
        tmp_path / "builtin",
        compiled_model=source_model.to_dict(),
        kernel_pack=pack,
        payloads=payloads,
    )
    return bundle_path, compile_model_source(bundle_path, use_cache=False)


def test_schema_v3_eager_artifact_owns_kernels_and_binary_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: ({"triple": "aarch64-apple-darwin", "cpu_features": []}, 1),
    )
    model = BuiltinSMModel()
    process_ir = build_process_ir("g g > g g")
    dag = compile_generic_dag(process_ir, model=model)
    runtime_schema = build_runtime_expression_schema(dag, model, process_id="gg_gg")
    schema = runtime_schema.to_mapping()
    propagated = {
        (int(slot["particle_id"]), int(slot["chirality"]))
        for slot in schema["value_storage"]["value_slots"]
        if slot["variant"] == "propagated"
    }
    vertex_kernels = {
        kind: 100 + kind for kind in sorted(dag.required_vertex_kinds)
    }
    propagator_kernels = {
        key: 1000 + index for index, key in enumerate(sorted(propagated))
    }
    resolver = MappingEagerKernelResolver(
        vertex_kernels=vertex_kernels,
        propagator_kernels=propagator_kernels,
        closure_kernels={},
    )
    tables = lower_eager_execution_tables(dag, model, schema, resolver)
    bundle_path, compiled_model = _prepared_model(
        tmp_path,
        kernel_ids=tuple(
            sorted({*vertex_kernels.values(), *propagator_kernels.values()})
        ),
    )
    process = EagerProcessArtifact(
        process_id="gg_gg",
        expression=process_ir.process,
        color_accuracy=process_ir.color_accuracy,
        external_pdgs=(*process_ir.initial_pdgs, *process_ir.final_pdgs),
        aliases=(),
        runtime_schema=runtime_schema,
        eager_tables=tables,
        point_tile_size=2048,
        workspace_mib=384,
        dag_summary={
            "current_count": len(dag.currents),
            "source_count": len(dag.sources),
            "interaction_count": len(dag.interactions),
            "amplitude_root_count": len(dag.amplitude_roots),
            "truncated": False,
        },
        validation_point=build_validation_point(
            dag,
            model,
            process_id="gg_gg",
            seed=7,
        ),
        generation_filters={},
    )
    run = RunConfig(
        action=Action.GENERATE,
        generation=GenerationConfig(emit_api_bundle=False),
        evaluator=EvaluatorConfig(execution_mode="eager"),
    )
    output = tmp_path / "artifact"

    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(bundle_path),
        compiled_model=compiled_model,
        configuration=_GenerationConfigProvenance.from_config(run),
        processes=(process,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    manifest = load_manifest(output)
    execution = json.loads(
        (output / "processes/gg_gg/execution.json").read_text(encoding="utf-8")
    )
    assert execution["kind"] == "pyamplicol-runtime-eager-execution"
    assert execution["eager_plan_abi"] == "pyamplicol-eager-plan-v1"
    assert execution["runtime_options"] == {
        "point_tile_size": 2048,
        "workspace_mib": 384,
    }
    assert execution["kernel_pack"] == {
        "manifest_path": "model/eager-kernel-pack.json",
        "payload_root": "model/eager-kernels",
    }
    assert manifest.runtime["required_runtime_capabilities"] == (
        "rusticol.eager-dag.complex-f64.v1",
    )
    declared = {payload.path for payload in manifest.payloads}
    assert "model/eager-kernel-pack.json" in declared
    assert "processes/gg_gg/eager/couplings.bin" in declared
    assert "processes/gg_gg/eager/closures.bin" in declared
    for kernel in compiled_model.prepared_bundle.kernel_pack.kernels:
        for path in kernel.referenced_payload_paths:
            assert f"model/eager-kernels/{path}" in declared
