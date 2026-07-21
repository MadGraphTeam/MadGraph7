# SPDX-License-Identifier: 0BSD
"""Integration scaffold contracts for future Rust eager lowering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.generation.runtime_schema as runtime_schema
import pyamplicol.generation.service as generation_service
from pyamplicol.api.errors import GenerationError
from pyamplicol.api.requests import ModelSource, ProcessRequest
from pyamplicol.artifacts import ArtifactBuilder
from pyamplicol.config import Action, EvaluatorConfig, RunConfig
from pyamplicol.generation.artifact_writer import (
    EAGER_PLAN_V3_ABI,
    EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_CONTAINER_KIND,
    EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
    EAGER_RUNTIME_LAYOUT_ABI,
    EAGER_RUNTIME_STORAGE_ABI,
    EagerPlanV3ProcessArtifact,
    EagerProcessArtifact,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.eager_columnar import (
    EAGER_LOWERING_INPUT_ABI,
    EagerLoweringInputV1,
)
from pyamplicol.generation.eager_lowering import (
    MappingEagerKernelResolver,
    PreparedCatalogEagerKernelIndex,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir

_PROCESS_ID = "eager_v3_test"
_RUNTIME_BYTES = b"future-rust-eager-runtime-pacbin"
_INDEX_SHA256 = "3" * 64


def _resolver_for(dag: object, model: Model) -> MappingEagerKernelResolver:
    concrete = cast(generation_service.GenericDAG, dag)
    propagators: set[tuple[int, int]] = set()
    for current in concrete.currents:
        key = (current.index.particle_id, current.index.chirality)
        if model._propagator_ir(*key).applies_propagator:
            propagators.add(key)
    return MappingEagerKernelResolver(
        vertex_kernels={
            kind: 100 + index
            for index, kind in enumerate(sorted(concrete.required_vertex_kinds))
        },
        propagator_kernels={
            key: 1_000 + index for index, key in enumerate(sorted(propagators))
        },
        closure_kernels={
            (str(root.kind), root.vertex_kind): 2_000 + root.id
            for root in concrete.amplitude_roots
            if root.kind != "direct-contraction"
        },
    )


def _service_case() -> tuple[
    Model,
    generation_service._CompiledProcess,
    generation_service._ResolvedModel,
    MappingEagerKernelResolver,
]:
    model = BuiltinSMModel()
    process_ir = build_process_ir("d d~ > z g")
    dag = compile_generic_dag(process_ir, model=model)
    request = ProcessRequest.parse(process_ir.process, name=_PROCESS_ID)
    compiled = generation_service._CompiledProcess(
        expanded=generation_service._ExpandedProcess(request, process_ir),
        dag=dag,
        helicity_sum_dag=None,
        helicity_selector_union_dag=None,
        coverage={},
        filters={},
        validation_points=(
            ValidationPointRecord(
                process_id=_PROCESS_ID,
                process=process_ir.process,
                seed=1,
                error="not sampled in scaffold test",
            ),
        ),
    )
    resolved = generation_service._ResolvedModel(
        source=ModelSource.built_in_sm(),
        model=model,
        eager_kernel_index=cast(PreparedCatalogEagerKernelIndex, object()),
    )
    return model, compiled, resolved, _resolver_for(dag, model)


def _binding_result(lowering_input: EagerLoweringInputV1) -> dict[str, object]:
    return {
        "kind": "pyamplicol-eager-runtime-lowering-result",
        "schema_version": 1,
        "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
        "lowering_input_sha256": lowering_input.digest,
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
        "required_runtime_capabilities": [EAGER_PLAN_V3_RUNTIME_CAPABILITY],
        "runtime_container": {
            "kind": EAGER_RUNTIME_CONTAINER_KIND,
            "schema_version": EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
            "storage_abi": EAGER_RUNTIME_STORAGE_ABI,
            "member_count": 7,
            "unpacked_size_bytes": 4096,
            "index_sha256": _INDEX_SHA256,
        },
        "inspection_summary": {
            "stage_count": 3,
            "invocation_count": 17,
            "attachment_count": 19,
        },
    }


def _patch_binding(
    monkeypatch: pytest.MonkeyPatch,
    binding: object,
) -> None:
    native = SimpleNamespace(_lower_eager_runtime_v1=binding)
    original = generation_service.importlib.import_module

    def import_module(name: str) -> object:
        if name == "pyamplicol._rusticol":
            return native
        return original(name)

    monkeypatch.setattr(generation_service.importlib, "import_module", import_module)


def _forbidden(message: str):
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(message)

    return fail


def test_plan_v3_builds_columnar_input_without_schema_or_evaluator_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, compiled, resolved, resolver = _service_case()
    captured: list[EagerLoweringInputV1] = []

    def binding(lowering_input: EagerLoweringInputV1, destination: str) -> object:
        captured.append(lowering_input)
        Path(destination).write_bytes(_RUNTIME_BYTES)
        return _binding_result(lowering_input)

    _patch_binding(monkeypatch, binding)
    monkeypatch.setenv(generation_service._EAGER_PLAN_VERSION_ENV, "v3")
    monkeypatch.setattr(
        generation_service,
        "PreparedCatalogEagerKernelResolver",
        lambda *_args: resolver,
    )
    for name in (
        "build_runtime_expression_schema",
        "lower_fused_eager_execution",
        "build_and_write_generic_stage_evaluator_artifacts",
        "write_model_parameter_evaluator_artifact",
    ):
        monkeypatch.setattr(
            generation_service,
            name,
            _forbidden(f"plan-v3 called {name}"),
        )
    monkeypatch.setattr(
        runtime_schema,
        "build_runtime_schema_layout",
        _forbidden("plan-v3 called expanded runtime-schema construction"),
    )

    backend = generation_service.GenerationBackend(
        RunConfig(
            action=Action.GENERATE,
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )
    result = backend._construct_eager_artifact(
        compiled,
        model,
        resolved,
        tmp_path,
        PhaseHandle("test", None, 1),
    )

    assert isinstance(result, EagerPlanV3ProcessArtifact)
    assert len(captured) == 1
    assert captured[0].abi == EAGER_LOWERING_INPUT_ABI
    assert result.lowering_input_sha256 == captured[0].digest
    assert result.eager_runtime_path.read_bytes() == _RUNTIME_BYTES
    assert result.referenced_kernel_ids
    assert result.physics["schema_version"] == 1
    assert result.physics["kind"] == "pyamplicol-resolved-physics"
    assert result.physics["process_id"] == _PROCESS_ID
    assert result.physics["process"] == compiled.dag.process.process
    coverage = cast(dict[str, object], result.physics["coverage"])
    assert coverage["helicities"] == "complete"
    assert coverage["color"] == "complete"
    assert coverage["color_kind"] == "physical-lc-flows"
    assert len(cast(list[object], result.physics["external_particles"])) == 4
    helicities = cast(list[dict[str, object]], result.physics["helicities"])
    assert len(helicities) == 24
    assert coverage["structural_zero_helicity_count"] == sum(
        bool(record["structural_zero"]) for record in helicities
    )
    assert cast(list[object], result.physics["color_components"])
    assert cast(dict[str, object], result.physics["reduction"])["groups"]
    assert cast(list[object], result.physics["model_parameters"])
    assert result.physics["selectors"] == {
        "helicity": True,
        "color_flow": True,
        "contracted_color": False,
    }


def test_plan_v3_binding_failure_is_closed_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, compiled, _resolved, resolver = _service_case()
    lowering_input = generation_service.build_eager_lowering_input_v1(
        dag=compiled.dag,
        model=model,
        resolver=resolver,
        process_id=_PROCESS_ID,
    )
    destination = tmp_path / "eager-runtime.pacbin"

    def binding(_lowering_input: object, output: str) -> object:
        Path(output).write_bytes(b"partial")
        raise RuntimeError("native lowering stopped")

    _patch_binding(monkeypatch, binding)
    with pytest.raises(GenerationError, match="native lowering stopped"):
        generation_service._invoke_rust_eager_lowering_v1(
            lowering_input,
            destination,
        )

    assert not destination.exists()


def test_plan_v2_remains_default_and_compiled_mode_ignores_v3_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, compiled, resolved, resolver = _service_case()
    monkeypatch.delenv(generation_service._EAGER_PLAN_VERSION_ENV, raising=False)
    monkeypatch.setattr(
        generation_service,
        "PreparedCatalogEagerKernelResolver",
        lambda *_args: resolver,
    )
    monkeypatch.setattr(
        generation_service,
        "build_eager_lowering_input_v1",
        _forbidden("default plan-v2 built a v3 lowering input"),
    )
    monkeypatch.setattr(
        generation_service,
        "_invoke_rust_eager_lowering_v1",
        _forbidden("default plan-v2 invoked Rust lowering"),
    )

    class V2Tables:
        referenced_kernel_ids = frozenset({100})

    monkeypatch.setattr(
        generation_service,
        "lower_fused_eager_execution",
        lambda **_kwargs: (
            {
                "physics": {
                    "schema_version": 1,
                    "kind": "pyamplicol-resolved-physics",
                    "process_id": _PROCESS_ID,
                }
            },
            cast(object, V2Tables()),
        ),
    )
    eager_backend = generation_service.GenerationBackend(
        RunConfig(
            action=Action.GENERATE,
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )
    eager = eager_backend._construct_eager_artifact(
        compiled,
        model,
        resolved,
        tmp_path,
        PhaseHandle("v2", None, 1),
    )
    assert isinstance(eager, EagerProcessArtifact)

    monkeypatch.setenv(generation_service._EAGER_PLAN_VERSION_ENV, "v3")
    compiled_backend = generation_service.GenerationBackend(
        RunConfig(action=Action.GENERATE),
        None,
    )
    evaluator = compiled_backend._construct_evaluator(
        compiled,
        model,
        PhaseHandle("compiled", None, 1),
    )
    assert evaluator.compiled is compiled


def _writer_process(
    tmp_path: Path,
    *,
    inspection_summary: dict[str, object] | None = None,
) -> EagerPlanV3ProcessArtifact:
    payload = tmp_path / "native-eager-runtime.pacbin"
    payload.write_bytes(_RUNTIME_BYTES)
    return EagerPlanV3ProcessArtifact(
        process_id=_PROCESS_ID,
        expression="d d~ > z g",
        color_accuracy="full",
        external_pdgs=(1, -1, 23, 21),
        aliases=(),
        physics={
            "schema_version": 1,
            "kind": "pyamplicol-resolved-physics",
            "process_id": _PROCESS_ID,
        },
        eager_runtime_path=payload,
        eager_runtime_size_bytes=len(_RUNTIME_BYTES),
        eager_runtime_sha256=hashlib.sha256(_RUNTIME_BYTES).hexdigest(),
        eager_runtime_member_count=7,
        eager_runtime_unpacked_size_bytes=4096,
        eager_runtime_index_sha256=_INDEX_SHA256,
        lowering_input_sha256="4" * 64,
        referenced_kernel_ids=frozenset(),
        inspection_summary=inspection_summary
        or {
            "stage_count": 3,
            "invocation_count": 17,
            "attachment_count": 19,
        },
        point_tile_size=2048,
        workspace_mib=384,
        dag_summary={
            "current_count": 11,
            "source_count": 4,
            "interaction_count": 23,
            "amplitude_root_count": 5,
            "truncated": False,
        },
        validation_point=ValidationPointRecord(
            process_id=_PROCESS_ID,
            process="d d~ > z g",
            seed=1,
            error="not sampled in scaffold test",
        ),
        generation_filters={},
    )


def _finalize_test_artifact(
    builder: ArtifactBuilder,
    process_record: dict[str, object],
) -> None:
    builder.finalize(
        kind="pyamplicol-process",
        producer={},
        model={},
        configuration={},
        processes=(process_record,),
        default_process_id=_PROCESS_ID,
        runtime={},
    )


def test_plan_v3_runtime_and_exact_bounded_summary_publish_atomically(
    tmp_path: Path,
) -> None:
    process = _writer_process(tmp_path)
    output = tmp_path / "artifact"
    with ArtifactBuilder(output) as builder:
        collector = artifact_writer._EvaluatorPayloadCollector(
            builder,
            existing=None,
            target={"triple": "test-target", "cpu_features": []},
        )
        process_record, _entry, _sha256 = artifact_writer._write_process_payloads(
            builder,
            process,
            evaluator_payloads=collector,
        )
        assert not output.exists()
        assert builder.staged_path(
            f"processes/{_PROCESS_ID}/eager-runtime.pacbin"
        ).is_file()
        assert builder.staged_path(f"processes/{_PROCESS_ID}/execution.json").is_file()
        _finalize_test_artifact(builder, process_record)

    execution_path = output / f"processes/{_PROCESS_ID}/execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    capabilities = [EAGER_PLAN_V3_RUNTIME_CAPABILITY]
    assert execution == {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-eager-execution",
        "required_runtime_capabilities": capabilities,
        "process": "d d~ > z g",
        "key": _PROCESS_ID,
        "color_accuracy": "full",
        "external_pdg_order": [1, -1, 23, 21],
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "runtime_options": {"point_tile_size": 2048, "workspace_mib": 384},
        "plan": {
            "kind": "pyamplicol-runtime-eager-execution",
            "eager_plan_abi": EAGER_PLAN_V3_ABI,
            "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
            "lowering_input_sha256": "4" * 64,
            "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
            "required_runtime_capabilities": capabilities,
            "runtime_container": {
                "kind": EAGER_RUNTIME_CONTAINER_KIND,
                "schema_version": EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
                "storage_abi": EAGER_RUNTIME_STORAGE_ABI,
                "path": "eager-runtime.pacbin",
                "size_bytes": len(_RUNTIME_BYTES),
                "sha256": hashlib.sha256(_RUNTIME_BYTES).hexdigest(),
                "member_count": 7,
                "unpacked_size_bytes": 4096,
                "index_sha256": _INDEX_SHA256,
            },
            "inspection_summary": {
                "stage_count": 3,
                "invocation_count": 17,
                "attachment_count": 19,
            },
        },
        "dag_summary": {
            "current_count": 11,
            "source_count": 4,
            "interaction_count": 23,
            "amplitude_root_count": 5,
            "truncated": False,
        },
    }
    assert execution_path.stat().st_size < 1 << 20
    assert (
        output / f"processes/{_PROCESS_ID}/eager-runtime.pacbin"
    ).read_bytes() == _RUNTIME_BYTES
    assert not tuple(tmp_path.glob(".artifact.staging-*"))


def test_plan_v3_summary_failure_rolls_back_staged_runtime(
    tmp_path: Path,
) -> None:
    process = _writer_process(
        tmp_path,
        inspection_summary={"oversized": "x" * (1 << 20)},
    )
    output = tmp_path / "artifact"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"existing artifact")
    staged: list[str] = []

    class TrackingCollector(artifact_writer._EvaluatorPayloadCollector):
        def add_file(
            self,
            relative: str,
            source: Path,
            *,
            process_id: str | None,
        ):
            record = super().add_file(
                relative,
                source,
                process_id=process_id,
            )
            staged.append(record.path)
            return record

    with (
        pytest.raises(ValueError, match="smaller than 1 MiB"),
        ArtifactBuilder(output, mode="replace") as builder,
    ):
        collector = TrackingCollector(
            builder,
            existing=None,
            target={"triple": "test-target", "cpu_features": []},
        )
        artifact_writer._write_process_payloads(
            builder,
            process,
            evaluator_payloads=collector,
        )

    assert staged == [f"processes/{_PROCESS_ID}/eager-runtime.pacbin"]
    assert sentinel.read_bytes() == b"existing artifact"
    assert not (output / f"processes/{_PROCESS_ID}").exists()
    assert not tuple(tmp_path.glob(".artifact.staging-*"))
