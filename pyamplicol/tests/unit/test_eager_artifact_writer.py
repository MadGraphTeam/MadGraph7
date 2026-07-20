# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import pyamplicol.artifacts.inspection as artifact_inspection
import pyamplicol.generation.artifact_writer as artifact_writer
from pyamplicol.api.errors import ArtifactError
from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import ArtifactBuilder, inspect_artifact, load_manifest
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
    PREPARED_KERNEL_VARIANT_ABI,
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedKernelVariantRecord,
    prepared_expression_digest,
    prepared_input_contract_digest,
    prepared_optimization_settings_digest,
    prepared_output_contract_digest,
    write_prepared_model_bundle,
)


def _prepared_model(
    tmp_path: Path,
    *,
    kernel_ids: tuple[int, ...],
    bundle_name: str = "builtin",
    canonical_signatures: Mapping[int, str] | None = None,
    payload_tag: str = "payload",
    variant_kernel_ids: tuple[int, ...] = (),
) -> tuple[Path, object]:
    source_model = compile_model_source("built-in-sm", use_cache=False)
    kernels = tuple(
        PreparedKernelRecord(
            kernel_id=kernel_id,
            contract_kind="vertex" if kernel_id < 1000 else "propagator",
            canonical_signature=(
                canonical_signatures.get(
                    kernel_id,
                    f"test:eager-kernel:{kernel_id}",
                )
                if canonical_signatures is not None
                else f"test:eager-kernel:{kernel_id}"
            ),
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
    variants = tuple(
        PreparedKernelVariantRecord(
            variant_id="independent-block-4",
            variant_abi=PREPARED_KERNEL_VARIANT_ABI,
            kind="independent-block",
            block_size=4,
            lane_layout="lane-major",
            base_kernel_id=kernel.kernel_id,
            base_canonical_signature=kernel.canonical_signature,
            base_expression_digest=prepared_expression_digest(kernel.exact_expressions),
            base_input_contract_digest=prepared_input_contract_digest(
                kernel.input_layout,
                kernel.input_contracts,
            ),
            base_output_contract_digest=prepared_output_contract_digest(
                kernel.output_layout
            ),
            backend="jit",
            optimization_settings_digest=prepared_optimization_settings_digest(
                {"optimization_level": 3}
            ),
            input_arity=4 * kernel.input_arity,
            output_arity=4 * kernel.output_arity,
            input_lane_stride=kernel.input_arity,
            output_lane_stride=kernel.output_arity,
            input_layout=tuple(
                f"lane:{lane}:{item}"
                for lane in range(4)
                for item in kernel.input_layout
            ),
            output_layout=tuple(
                f"lane:{lane}:{item}"
                for lane in range(4)
                for item in kernel.output_layout
            ),
            f64_evaluator_manifest={
                "kind": "symjit-application-evaluator",
                "input_len": 4 * kernel.input_arity,
                "output_len": 4 * kernel.output_arity,
                "application_path": (
                    f"kernels/{kernel.kernel_id}/variants/"
                    "independent-block-4/application.symjit"
                ),
                "evaluator_state_path": (
                    f"kernels/{kernel.kernel_id}/variants/independent-block-4/exact.bin"
                ),
            },
        )
        for kernel in kernels
        if kernel.kernel_id in variant_kernel_ids
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
        kernel_variants=variants,
    )
    payloads = {
        path: f"{payload_tag}:{path}".encode() for path in pack.referenced_payload_paths
    }
    bundle_path = write_prepared_model_bundle(
        tmp_path / bundle_name,
        compiled_model=source_model.to_dict(),
        kernel_pack=pack,
        payloads=payloads,
    )
    return bundle_path, compile_model_source(bundle_path, use_cache=False)


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _inject_stale_eager_payload(output: Path) -> str:
    stale_path = "model/eager-kernels/superseded/application.symjit"
    manifest = load_manifest(output)
    with ArtifactBuilder(output, mode="append") as builder:
        builder.add_bytes(
            stale_path,
            b"superseded evaluator payload",
            role="evaluator-state",
            media_type="application/vnd.symjit.application",
            target=manifest.producer["target"],
        )
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            default_process_id=manifest.default_process_id,
            runtime=manifest.runtime,
            dependencies=manifest.dependencies,
            extensions=manifest.extensions,
        )
    assert any(record.path == stale_path for record in load_manifest(output).payloads)
    return stale_path


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
    vertex_kernels = {kind: 100 + kind for kind in sorted(dag.required_vertex_kinds)}
    propagator_kernels = {
        key: 1000 + index for index, key in enumerate(sorted(propagated))
    }
    resolver = MappingEagerKernelResolver(
        vertex_kernels=vertex_kernels,
        propagator_kernels=propagator_kernels,
        closure_kernels={},
    )
    tables = lower_eager_execution_tables(dag, model, schema, resolver)
    appended_vertex_kernels = {
        kind: 500 + kind for kind in sorted(dag.required_vertex_kinds)
    }
    appended_propagator_kernels = {
        key: 2000 + index for index, key in enumerate(sorted(propagated))
    }
    appended_resolver = MappingEagerKernelResolver(
        vertex_kernels=appended_vertex_kernels,
        propagator_kernels=appended_propagator_kernels,
        closure_kernels={},
    )
    appended_tables = lower_eager_execution_tables(
        dag,
        model,
        schema,
        appended_resolver,
    )
    selected_variant_kernel_id = min(vertex_kernels.values())
    bundle_path, compiled_model = _prepared_model(
        tmp_path,
        kernel_ids=tuple(
            sorted(
                {
                    *vertex_kernels.values(),
                    *propagator_kernels.values(),
                    *appended_vertex_kernels.values(),
                    *appended_propagator_kernels.values(),
                    4242,
                }
            )
        ),
        canonical_signatures={
            selected_variant_kernel_id: "a" * 64,
            4242: "b" * 64,
        },
        variant_kernel_ids=(selected_variant_kernel_id, 4242),
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
    progress_events: list[dict[str, object]] = []

    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(bundle_path),
        compiled_model=compiled_model,
        configuration=_GenerationConfigProvenance.from_config(run),
        processes=(process,),
        timings={"total": 0.1},
        api_bundle_hook=None,
        progress_callback=progress_events.append,
    )

    assert progress_events[0]["step"] == "global payloads"
    assert any(event["step"] == "process payloads" for event in progress_events)
    assert progress_events[-1]["step"] == "publishing artifact"

    with (
        pytest.raises(ValueError, match="changed before the transaction lock"),
        ArtifactBuilder(
            output,
            mode="append",
            expected_artifact_id="0" * 64,
        ),
    ):
        pytest.fail("a stale append snapshot must not enter its write body")
    assert not tuple(tmp_path.glob(".artifact.staging-*"))

    manifest = load_manifest(output)
    execution = json.loads(
        (output / "processes/gg_gg/execution.json").read_text(encoding="utf-8")
    )
    assert execution["kind"] == "pyamplicol-runtime-eager-execution"
    assert execution["eager_plan_abi"] == "pyamplicol-eager-plan-v2"
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
    inspection = inspect_artifact(output)
    inspected_process = inspection.processes[0]
    assert inspected_process.execution_mode == "eager"
    assert inspected_process.prepared_backend == "jit"
    assert inspected_process.invocation_count == tables.invocation_count
    assert inspected_process.attachment_count == tables.attachment_count
    assert inspected_process.evaluation_alias_count == (
        tables.attachment_count - tables.invocation_count
    )
    assert inspected_process.maximum_fanout == max(
        row.attachment_count for stage in tables.stages for row in stage.invocations
    )
    assert inspected_process.requested_point_tile_size == 2048
    assert inspected_process.effective_point_tile_size is None
    assert inspected_process.workspace_limit_bytes == 384 * 1024 * 1024
    assert inspected_process.workspace_bytes is None
    assert inspected_process.selector_closure_available
    assert tables.selector_closures is not None
    assert inspected_process.selector_domain_count == len(
        tables.selector_closures.domains
    )
    assert inspected_process.selector_domain_membership_count == len(
        tables.selector_closures.domain_group_ids
    )
    pack_identity = manifest.extensions["eager_prepared_pack"]
    assert pack_identity["kind"] == "pyamplicol-prepared-kernel-pack-identity"
    assert pack_identity["schema_version"] == 1
    assert pack_identity["eager_kernel_abi"] == "pyamplicol-eager-kernel-v1"
    assert pack_identity["backend"] == "jit"
    assert pack_identity["kernel_count"] == len(
        compiled_model.prepared_bundle.kernel_pack.kernels
    )
    assert len(pack_identity["identity_sha256"]) == 64
    declared = {payload.path for payload in manifest.payloads}
    assert "model/eager-kernel-pack.json" in declared
    assert "processes/gg_gg/eager/couplings.bin" in declared
    assert "processes/gg_gg/eager/closures.bin" in declared
    assert "processes/gg_gg/eager/selector-domains.bin" in declared
    assert "processes/gg_gg/eager/selector-domain-group-ids.bin" in declared
    assert "processes/gg_gg/eager/closure-domains.bin" in declared
    emitted_pack = json.loads(
        (output / "model/eager-kernel-pack.json").read_text(encoding="utf-8")
    )
    assert emitted_pack["eager_kernel_abi"] == "pyamplicol-eager-kernel-v1"
    assert {kernel["kernel_id"] for kernel in emitted_pack["kernels"]} == set(
        tables.referenced_kernel_ids
    )
    assert {
        variant["base_kernel_id"] for variant in emitted_pack["kernel_variants"]
    } == {selected_variant_kernel_id}
    for kernel in compiled_model.prepared_bundle.kernel_pack.kernels:
        for path in kernel.referenced_payload_paths:
            emitted = f"model/eager-kernels/{path}" in declared
            assert emitted is (kernel.kernel_id in tables.referenced_kernel_ids)
    for variant in compiled_model.prepared_bundle.kernel_pack.kernel_variants:
        for path in variant.referenced_payload_paths:
            emitted = f"model/eager-kernels/{path}" in declared
            assert emitted is (variant.base_kernel_id in tables.referenced_kernel_ids)

    execution_path = output / "processes/gg_gg/execution.json"
    closure_domains_path = output / "processes/gg_gg/eager/closure-domains.bin"
    original_closure_domains = closure_domains_path.read_bytes()
    closure_domains_path.write_bytes(
        struct.pack("<I", len(tables.selector_closures.domains))
        + original_closure_domains[4:]
    )
    try:
        with pytest.raises(ArtifactError, match="references unknown domain"):
            artifact_inspection._execution_inspection(manifest, execution_path)
    finally:
        closure_domains_path.write_bytes(original_closure_domains)

    appended_runtime_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id="gg_gg_appended",
    )
    appended_process = EagerProcessArtifact(
        process_id="gg_gg_appended",
        expression=process_ir.process,
        color_accuracy=process_ir.color_accuracy,
        external_pdgs=(*process_ir.initial_pdgs, *process_ir.final_pdgs),
        aliases=(),
        runtime_schema=appended_runtime_schema,
        eager_tables=appended_tables,
        point_tile_size=2048,
        workspace_mib=384,
        dag_summary=process.dag_summary,
        validation_point=build_validation_point(
            dag,
            model,
            process_id="gg_gg_appended",
            seed=11,
        ),
        generation_filters={},
    )
    stale_path = _inject_stale_eager_payload(output)
    write_schema_v3_artifact(
        output,
        mode="append",
        source=ModelSource.from_path(bundle_path),
        compiled_model=compiled_model,
        configuration=_GenerationConfigProvenance.from_config(run),
        processes=(appended_process,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    appended_manifest = load_manifest(output)
    appended_pack = json.loads(
        (output / "model/eager-kernel-pack.json").read_text(encoding="utf-8")
    )
    expected_kernel_ids = (
        tables.referenced_kernel_ids | appended_tables.referenced_kernel_ids
    )
    assert {kernel["kernel_id"] for kernel in appended_pack["kernels"]} == set(
        expected_kernel_ids
    )
    assert {record["id"] for record in appended_manifest.processes} == {
        "gg_gg",
        "gg_gg_appended",
    }
    assert appended_manifest.extensions["eager_prepared_pack"] == pack_identity
    appended_payloads = {payload.path for payload in appended_manifest.payloads}
    assert stale_path not in appended_payloads
    assert not (output / stale_path).exists()
    for kernel in compiled_model.prepared_bundle.kernel_pack.kernels:
        for path in kernel.referenced_payload_paths:
            emitted = f"model/eager-kernels/{path}" in appended_payloads
            assert emitted is (kernel.kernel_id in expected_kernel_ids)

    ordered_kernel_ids = tuple(
        kernel.kernel_id
        for kernel in compiled_model.prepared_bundle.kernel_pack.kernels
    )
    shifted_signatures = {
        kernel_id: (
            f"shifted:{ordered_kernel_ids[(index + 1) % len(ordered_kernel_ids)]}"
        )
        for index, kernel_id in enumerate(ordered_kernel_ids)
    }
    shifted_bundle_path, shifted_model = _prepared_model(
        tmp_path,
        kernel_ids=ordered_kernel_ids,
        bundle_name="shifted",
        canonical_signatures=shifted_signatures,
        payload_tag="shifted-payload",
    )
    rebound_id = "gg_gg_rebound"
    rebound_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id=rebound_id,
    )
    rebound_process = replace(
        appended_process,
        process_id=rebound_id,
        runtime_schema=rebound_schema,
        validation_point=build_validation_point(
            dag,
            model,
            process_id=rebound_id,
            seed=13,
        ),
    )
    before_failed_append = _tree_snapshot(output)
    with pytest.raises(
        ValueError,
        match="prepared kernel pack identity differs",
    ):
        write_schema_v3_artifact(
            output,
            mode="append",
            source=ModelSource.from_path(shifted_bundle_path),
            compiled_model=shifted_model,
            configuration=_GenerationConfigProvenance.from_config(run),
            processes=(rebound_process,),
            timings={"total": 0.1},
            api_bundle_hook=None,
        )
    assert _tree_snapshot(output) == before_failed_append
    assert not tuple(tmp_path.glob(".artifact.staging-*"))
