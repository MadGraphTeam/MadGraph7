# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.cache import (
    build_reset_caches,
    digest_json,
    empty_measurement,
)
from tools.performance_report.final_audit import (
    ArtifactEvidence,
    FinalAuditError,
    _active_runtime_snapshot,
    _artifact_reference,
    _audit_below_resolution_execution_timing,
    _audit_compiled_execution,
    _audit_eager_execution,
    _audit_measurement,
    _audit_model_source,
    _audit_pointwise,
    _audit_recurrence_execution,
    _audit_recurrence_source_pack,
    _audit_runtime_identity,
    _audit_tex_table_reachability,
    _authenticated_effective_config,
    _ensure_exact_cli_python,
    _python_package_tree_identity,
    _real_nonnegative,
    _runtime_namespace_paths,
    _shared_artifact_contract,
    audit_final_report,
)
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    MeasurementSpec,
    ModelKey,
    Workload,
)
from tools.performance_report.publication import portable_publication_value
from tools.performance_report.service import ReportPaths, ReportService

_REVISION = "a" * 40
_LEGACY_REVISION = "79c96cecf2a722e50c3d2030b6894d755f96518a"
_ARTIFACT_ID = "b" * 64
_CAPABILITY = "rusticol.recurrence-direct-arena.complex-f64.v1"
_COLOR_CAPABILITY = "rusticol.recurrence-color.lc.v1"


def _cell(
    mode: ExecutionMode,
    *,
    backend: str = "jit",
    optimization_level: int | None = 3,
) -> CellSpec:
    return CellSpec(
        dataset_id=f"test_{mode.value}",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            mode,
            ModelKey.BUILTIN_SM,
            Accuracy.LC,
            backend,
            optimization_level,
        ),
        workload=Workload.SELECTED_FLOW,
    )


def _recurrence_binding_digest(binding: dict[str, object]) -> str:
    if binding["kind"] != "rusticol-intrinsic":
        payload = dict(binding)
        payload.pop("payload_digest", None)
        return digest_json(payload)
    fields = (
        "abi",
        "contribution_parent_permutation",
        "kind",
        "runtime_template",
    )
    return digest_json({field: binding.get(field) for field in fields})


def _refresh_recurrence_source_catalog(
    execution: dict[str, object],
    pack: dict[str, object],
    *,
    refresh_binding_digest: bool = True,
) -> None:
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        binding = template["payload_binding"]
        assert isinstance(binding, dict)
        if refresh_binding_digest:
            binding["payload_digest"] = _recurrence_binding_digest(binding)
        semantic = dict(template)
        semantic.pop("semantic_digest", None)
        template["semantic_digest"] = digest_json(semantic)
    semantic_catalog = dict(catalog)
    semantic_catalog.pop("catalog_digest", None)
    catalog["catalog_digest"] = digest_json(semantic_catalog)
    execution["direct_template_catalog_digest"] = catalog["catalog_digest"]
    plan = execution["plan"]
    assert isinstance(plan, dict)
    plan["direct_template_catalog_digest"] = catalog["catalog_digest"]


def _recurrence_source_fixture(
    source_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source_path = "kernels/000000/application-0.symjit"
    binding: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-payload-binding-v1",
        "contribution_parent_permutation": [0, 1],
        "destination_operation": "finalize-in-place",
        "direct_application_abi": "symjit-direct-application-storage-v1",
        "exact_factor_scalar_slots": [0, 1],
        "input_plane_count": 1,
        "input_plane_projections": [{"kind": "destination-current"}],
        "intrinsic_contract_digest": None,
        "kind": "prepared-direct-call",
        "output_alias_inputs": [0],
        "parameter_bindings": [{"index": 0, "kind": "plane"}],
        "payload_digest": "",
        "payload_paths": [source_path],
        "prepared_kernel_id": 0,
        "prepared_template_semantic_digest": "d" * 64,
        "role": "finalization",
        "runtime_template": None,
        "scalar_input_count": 2,
        "scalar_projections": [
            {"imaginary": False, "kind": "exact-factor"},
            {"imaginary": True, "kind": "exact-factor"},
        ],
        "source_application_abi": "symjit-application-storage-v3",
        "source_application_path": source_path,
        "source_application_sha256": source_sha256,
        "state_plane_indices": [],
    }
    template: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-template-v1",
        "alignment_bytes": 64,
        "backend": "jit",
        "coupling_slot_count": 0,
        "destination_aliasing": True,
        "destination_component_count": 1,
        "destination_operation": "finalize-in-place",
        "direct_executor_id": 0,
        "evaluator_binding_id": 0,
        "evaluator_resolver_key": "prepared-kernel-0",
        "exact_expression_digest": "e" * 64,
        "momentum_operand_count": 0,
        "optimization_level": 2,
        "parameter_slot_count": 0,
        "parent_arity": 1,
        "parent_component_counts": [1],
        "payload_binding": binding,
        "portable": True,
        "role": "finalization",
        "semantic_digest": "",
        "semantic_template_ids": ["prepared-finalization"],
        "simd_axis": "points-contiguous",
        "target_triple": "symjit-storage-v3-portable",
        "template_id": "template-0",
    }
    prepared_digest = "a" * 64
    catalog: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-template-v1",
        "backend": "jit",
        "backend_abi": "rusticol.recurrence-direct-backend.v1",
        "canonicalization_abi": "pyamplicol-canonical-json-v1",
        "catalog_digest": "",
        "compiled_model_digest": "b" * 64,
        "optimization_level": 2,
        "optimization_settings_digest": "c" * 64,
        "portable": True,
        "prepared_kernel_contract_digest": "4" * 64,
        "prepared_kernel_pack_digest": prepared_digest,
        "prepared_kernel_payload_digest": "5" * 64,
        "recurrence_template_catalog_digest": "6" * 64,
        "target_triple": "symjit-storage-v3-portable",
        "templates": [template],
    }
    plan: dict[str, object] = {
        "builder_input_abi": "pyamplicol-recurrence-builder-input-v2",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "inspection_summary": {
            "direct_arena": {
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0,
                "row_group_count": 4,
            }
        },
    }
    execution: dict[str, object] = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "builder_input_abi": "pyamplicol-recurrence-builder-input-v2",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "required_runtime_capabilities": [_COLOR_CAPABILITY, _CAPABILITY],
        "plan": plan,
    }
    pack: dict[str, object] = {
        "backend": "jit",
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "recurrence_direct_template": catalog,
        "kernel_variants": [],
        "kernels": [
            {
                "kernel_id": 0,
                "f64_evaluator_manifest": {
                    "kind": "symjit-application-evaluator",
                    "backend": "jit",
                    "runtime_capability": "symjit.application.complex-f64.v1",
                    "application_abi": "symjit-application-storage-v3",
                    "application_path": source_path,
                    "optimization_level": 2,
                },
            }
        ],
    }
    _refresh_recurrence_source_catalog(execution, pack)
    return execution, pack


def _legacy_cell() -> CellSpec:
    return CellSpec(
        dataset_id="reference_amplicol_lc",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            ExecutionMode.AMPLICOL,
            None,
            Accuracy.LC,
            "fortran",
            None,
        ),
        workload=Workload.SELECTED_FLOW,
    )


def _compiled_stage(
    *,
    optimization_level: int = 3,
    backend: str = "jit",
) -> dict[str, object]:
    source_application_abi = (
        "symjit-application-storage-v3"
        if backend == "jit"
        else "pyamplicol-native-compiled-direct-application-v1"
    )
    application_abi = (
        "symjit-direct-application-storage-v3"
        if backend == "jit"
        else "pyamplicol-native-compiled-direct-application-v1"
    )
    evaluator = (
        {
            "kind": "symjit-application-evaluator",
            "runtime_capability": "symjit.application.complex-f64.v1",
            "optimization_level": optimization_level,
        }
        if backend == "jit"
        else {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": (f"symbolica.compiled-{backend}.complex-f64.v1"),
        }
    )
    return {
        "parameter_count": 1,
        "output_length": 1,
        "evaluator": evaluator,
        "compiled_plane_arena": {
            "schema_version": 1,
            "kind": "compiled-plane-arena-stage",
            "application_abi": application_abi,
            "source_application_abi": source_application_abi,
            "element_layout": "split-complex-component-major",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "input_bindings": [{"parameter_index": 0}],
            "output_bindings": [{"output_index": 0}],
            "leaves": [
                {
                    "source_application_abi": source_application_abi,
                    "optimization_level": optimization_level,
                    "direct_codegen_optimization_level": 3,
                }
            ],
        },
    }


def test_compiled_arena_audits_fused_stages_and_parameter_exclusion() -> None:
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [_compiled_stage()],
                "amplitude_stage": _compiled_stage(),
            },
            "model_parameter_evaluator": {
                "evaluator": {"application_abi": "symjit-application-storage-v3"}
            },
        },
    }
    assert _audit_compiled_execution(
        execution,
        _cell(ExecutionMode.COMPILED),
    ) == (2, 2)
    jit_o1 = deepcopy(execution)
    for stage in (
        *jit_o1["compiled"]["stage_evaluators"]["stages"],  # type: ignore[index]
        jit_o1["compiled"]["stage_evaluators"]["amplitude_stage"],  # type: ignore[index]
    ):
        stage["compiled_plane_arena"]["leaves"][0]["optimization_level"] = 1
        stage["evaluator"]["optimization_level"] = 1
    assert _audit_compiled_execution(
        jit_o1,
        _cell(ExecutionMode.COMPILED, optimization_level=1),
    ) == (2, 2)

    invalid_direct_codegen = deepcopy(execution)
    invalid_direct_codegen["compiled"]["stage_evaluators"]["stages"][0][  # type: ignore[index]
        "compiled_plane_arena"
    ]["leaves"][0]["direct_codegen_optimization_level"] = 2
    with pytest.raises(
        FinalAuditError,
        match=r"leaf\.direct_codegen_optimization_level",
    ):
        _audit_compiled_execution(
            invalid_direct_codegen,
            _cell(ExecutionMode.COMPILED),
        )

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "compiled_plane_arena": {}
    }
    with pytest.raises(FinalAuditError, match="model-parameter"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "nested": {"native_direct_application": {}}
    }
    with pytest.raises(FinalAuditError, match="DirectApplication metadata"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))

    execution["compiled"]["model_parameter_evaluator"] = {  # type: ignore[index]
        "nested": {
            "application_abi": "pyamplicol-native-compiled-direct-application-v1"
        }
    }
    with pytest.raises(FinalAuditError, match="direct application ABI"):
        _audit_compiled_execution(execution, _cell(ExecutionMode.COMPILED))


@pytest.mark.parametrize(
    ("backend", "capability"),
    [
        ("jit", "symjit.application.complex-f64.v1"),
        ("cpp", "symbolica.compiled-cpp.complex-f64.v1"),
        ("asm", "symbolica.compiled-asm.complex-f64.v1"),
    ],
)
def test_compiled_arena_authenticates_source_runtime_capability(
    backend: str,
    capability: str,
) -> None:
    stage = _compiled_stage(backend=backend)
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": stage,
            },
        },
    }
    cell = _cell(
        ExecutionMode.COMPILED,
        backend=backend,
        optimization_level=3 if backend == "jit" else None,
    )
    assert _audit_compiled_execution(execution, cell) == (1, 1)
    assert stage["evaluator"]["runtime_capability"] == capability  # type: ignore[index]

    stage["evaluator"]["runtime_capability"] = "wrong.capability"  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="runtime_capability"):
        _audit_compiled_execution(execution, cell)


def test_final_replay_matrix_conversion_rejects_sign_and_complex_drift() -> None:
    assert _real_nonnegative(2.0 + 0.0j, "matrix") == 2.0
    assert _real_nonnegative(-1.0e-16 + 0.0j, "matrix") == 0.0
    with pytest.raises(FinalAuditError, match="materially negative"):
        _real_nonnegative(-1.0e-3 + 0.0j, "matrix")
    with pytest.raises(FinalAuditError, match="imaginary part"):
        _real_nonnegative(1.0 + 1.0e-3j, "matrix")


def test_shared_artifact_contract_binds_the_exact_physics_cell() -> None:
    cell = _cell(ExecutionMode.RECURRENCE, optimization_level=2)
    dataset_alias = replace(cell, dataset_id="another_dataset")
    different_process = replace(
        cell,
        process="u u~ > z",
        process_key="uu_z",
    )

    assert _shared_artifact_contract(dataset_alias) == _shared_artifact_contract(cell)
    assert _shared_artifact_contract(different_process) != _shared_artifact_contract(
        cell
    )


def test_effective_toml_is_payload_authenticated_and_model_bound(
    tmp_path: Path,
) -> None:
    relative = "config/effective.toml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    data = b'[model]\nsource = "built-in-sm"\n'
    path.write_bytes(data)
    manifest = SimpleNamespace(
        configuration={"effective_path": relative},
        model={"source_kind": "built-in-sm"},
        payloads=(
            SimpleNamespace(
                path=relative,
                role="configuration-effective",
                media_type="application/toml",
                process_id=None,
                executable=False,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )
    effective = _authenticated_effective_config(tmp_path, manifest)
    _audit_model_source(_cell(ExecutionMode.COMPILED), effective)

    path.write_bytes(b'[model]\nsource = "built-in-xx"\n')
    with pytest.raises(FinalAuditError, match="artifact manifest"):
        _authenticated_effective_config(tmp_path, manifest)


def test_eager_and_recurrence_arena_abis_are_audited(tmp_path: Path) -> None:
    pack_path = tmp_path / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_payload = {
        "backend": "jit",
        "optimization_settings": {
            "backend": "jit",
            "jit_optimization_level": 2,
        },
        "recurrence_direct_template": {
            "abi": "pyamplicol-recurrence-direct-template-v1",
            "backend": "jit",
            "optimization_level": 2,
            "portable": True,
            "templates": [{"backend": "jit", "optimization_level": 2}],
        },
        "kernel_variants": [],
        "kernels": [
            {
                "contract_kind": "vertex",
                "f64_evaluator_manifest": {
                    "kind": "symjit-application-evaluator",
                    "runtime_capability": "symjit.application.complex-f64.v1",
                    "optimization_level": 2,
                    "direct_table": {
                        "capability": "eager-direct-arena-v1",
                        "descriptor_abi": ("symjit-direct-table-descriptor-v1"),
                        "binding_abi": "symjit-direct-table-binding-v2",
                        "source_application_abi": ("symjit-application-storage-v3"),
                    },
                },
            },
            {
                "contract_kind": "model-parameter",
                "f64_evaluator_manifest": {
                    "kind": "symjit-application-evaluator",
                    "runtime_capability": "symjit.application.complex-f64.v1",
                    "optimization_level": 2,
                    "application_abi": "symjit-application-storage-v3",
                },
            },
        ],
    }
    payload_record = SimpleNamespace(
        path="model/eager-kernel-pack.json",
        role="evaluator-manifest",
        media_type="application/json",
        process_id=None,
        executable=False,
        size_bytes=0,
        sha256="",
    )

    def write_pack(value: object) -> None:
        data = json.dumps(value).encode("utf-8")
        pack_path.write_bytes(data)
        payload_record.size_bytes = len(data)
        payload_record.sha256 = hashlib.sha256(data).hexdigest()

    write_pack(pack_payload)
    manifest = SimpleNamespace(
        payloads=(payload_record,),
        extensions={
            "eager_prepared_pack": {
                "kind": "pyamplicol-prepared-kernel-pack-identity",
                "schema_version": 1,
                "backend": "jit",
                "eager_kernel_abi": "pyamplicol-eager-kernel-v1",
                "identity_sha256": "c" * 64,
                "kernel_count": 2,
            }
        },
    )
    eager = {
        "kind": "pyamplicol-runtime-eager-execution",
        "eager_plan_abi": "pyamplicol-eager-plan-v3",
        "plan": {
            "eager_plan_abi": "pyamplicol-eager-plan-v3",
            "lowering_input_abi": "pyamplicol-eager-lowering-input-v1",
            "runtime_layout_abi": "pyamplicol-eager-runtime-layout-v1",
        },
        "kernel_pack": {"manifest_path": "model/eager-kernel-pack.json"},
    }
    assert (
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(
                ExecutionMode.EAGER,
                optimization_level=2,
            ),
        )
        == 1
    )
    wrong_capability = deepcopy(pack_payload)
    wrong_capability["kernels"][0]["f64_evaluator_manifest"][  # type: ignore[index]
        "runtime_capability"
    ] = "symbolica.compiled-cpp.complex-f64.v1"
    write_pack(wrong_capability)
    with pytest.raises(FinalAuditError, match="runtime_capability"):
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(ExecutionMode.EAGER, optimization_level=2),
        )
    for nested_corruption in (
        {"nested": [{"direct_table": None}]},
        {"nested": {"capability": "eager-direct-arena-v1"}},
        {"nested": {"descriptor_abi": "symjit-direct-table-descriptor-v1"}},
        {"nested": {"binding_abi": "symjit-direct-table-binding-v2"}},
        {"nested": {"application_abi": "pyamplicol-eager-native-direct-table-v1"}},
    ):
        corrupted = deepcopy(pack_payload)
        corrupted["kernels"][1]["f64_evaluator_manifest"].update(  # type: ignore[index]
            nested_corruption
        )
        write_pack(corrupted)
        with pytest.raises(FinalAuditError, match="model-parameter"):
            _audit_eager_execution(
                tmp_path,
                manifest,
                eager,
                _cell(ExecutionMode.EAGER, optimization_level=2),
            )
    corrupted = deepcopy(pack_payload)
    corrupted["kernels"][1]["nested_kernel_metadata"] = {  # type: ignore[index]
        "direct_table": {}
    }
    write_pack(corrupted)
    with pytest.raises(FinalAuditError, match="model-parameter"):
        _audit_eager_execution(
            tmp_path,
            manifest,
            eager,
            _cell(ExecutionMode.EAGER, optimization_level=2),
        )
    source_relative = "model/eager-kernels/kernels/000000/application-0.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    recurrence, recurrence_pack = _recurrence_source_fixture(source_sha256)
    write_pack(recurrence_pack)
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_data)
    source_record = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        process_id=None,
        executable=False,
        size_bytes=len(source_data),
        sha256=source_sha256,
    )
    manifest.payloads = (*manifest.payloads, source_record)
    assert (
        _audit_recurrence_execution(
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        )
        == 4
    )
    source_identity = _audit_recurrence_source_pack(
        tmp_path,
        manifest,
        recurrence,
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        execution_manifest_path="processes/d_dbar_to_z/execution.json",
        execution_manifest_sha256="1" * 64,
    )
    assert source_identity == {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 1,
        "prepared_direct_template_count": 1,
        "source_evaluator_leaf_count": 1,
        "source_application_abi": "symjit-application-storage-v3",
        "direct_application_abi": "symjit-direct-application-storage-v1",
        "prepared_kernel_pack_digest": "a" * 64,
        "direct_template_catalog_digest": recurrence["direct_template_catalog_digest"],
        "execution_manifest_path": "processes/d_dbar_to_z/execution.json",
        "execution_manifest_sha256": "1" * 64,
        "kernel_pack_path": "model/eager-kernel-pack.json",
        "kernel_pack_sha256": payload_record.sha256,
    }
    wrong_o2 = deepcopy(recurrence_pack)
    wrong_o2["optimization_settings"]["jit_optimization_level"] = 1  # type: ignore[index]
    write_pack(wrong_o2)
    with pytest.raises(FinalAuditError, match="does not prove JIT O2"):
        _audit_recurrence_source_pack(
            tmp_path,
            manifest,
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
        )
    write_pack(pack_payload)
    recurrence["plan"]["inspection_summary"]["direct_arena"][  # type: ignore[index]
        "packed_input_bytes"
    ] = 8
    with pytest.raises(FinalAuditError, match="packed_input_bytes"):
        _audit_recurrence_execution(
            recurrence,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("execution_template_abi", "direct_template_abi"),
        ("plan_template_abi", "direct_template_abi"),
        ("execution_backend_abi", "direct_backend_abi"),
        ("plan_backend_abi", "direct_backend_abi"),
        ("plan_prepared_digest", "source digests"),
        ("plan_catalog_digest", "source digests"),
        ("pack_prepared_digest", "execution_digest_links"),
        ("execution_catalog_link", "execution_digest_links"),
        ("catalog_backend_abi", "does not prove JIT O2"),
        ("catalog_canonicalization_abi", "does not prove JIT O2"),
        ("catalog_digest", "catalog_digest"),
        ("template_portable", "template\\[0\\] contract"),
        ("template_abi", "template\\[0\\] contract"),
        ("binding_abi", "payload-binding contract"),
        ("source_application_abi", "source application"),
        ("direct_application_abi", "source application"),
        ("payload_digest", "payload-binding contract"),
        ("source_leaf_runtime_capability", "runtime_capability"),
        ("source_leaf_application_abi", "application_abi"),
        ("source_leaf_optimization", "optimization_level"),
        ("source_leaf_path", "payload is missing"),
        ("source_payload_digest_link", "not bound to its prepared kernel"),
    ),
)
def test_final_audit_recurrence_source_pack_rejects_every_broken_link(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    plan = execution["plan"]
    assert isinstance(plan, dict)
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    template = templates[0]
    assert isinstance(template, dict)
    binding = template["payload_binding"]
    assert isinstance(binding, dict)
    kernels = pack["kernels"]
    assert isinstance(kernels, list)
    kernel = kernels[0]
    assert isinstance(kernel, dict)
    source_leaf = kernel["f64_evaluator_manifest"]
    assert isinstance(source_leaf, dict)
    refresh_catalog: bool | None = None

    if corruption == "execution_template_abi":
        execution["direct_template_abi"] = "wrong"
    elif corruption == "plan_template_abi":
        plan["direct_template_abi"] = "wrong"
    elif corruption == "execution_backend_abi":
        execution["direct_backend_abi"] = "wrong"
    elif corruption == "plan_backend_abi":
        plan["direct_backend_abi"] = "wrong"
    elif corruption == "plan_prepared_digest":
        plan["prepared_kernel_pack_digest"] = "0" * 64
    elif corruption == "plan_catalog_digest":
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "pack_prepared_digest":
        catalog["prepared_kernel_pack_digest"] = "0" * 64
        refresh_catalog = True
    elif corruption == "execution_catalog_link":
        execution["direct_template_catalog_digest"] = "0" * 64
        plan["direct_template_catalog_digest"] = "0" * 64
    elif corruption == "catalog_backend_abi":
        catalog["backend_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_canonicalization_abi":
        catalog["canonicalization_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "catalog_digest":
        catalog["compiled_model_digest"] = "0" * 64
    elif corruption == "template_portable":
        template["portable"] = False
        refresh_catalog = True
    elif corruption == "template_abi":
        template["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "binding_abi":
        binding["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "source_application_abi":
        binding["source_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "direct_application_abi":
        binding["direct_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "payload_digest":
        binding["payload_digest"] = "0" * 64
        refresh_catalog = False
    elif corruption == "source_leaf_runtime_capability":
        source_leaf["runtime_capability"] = "wrong"
    elif corruption == "source_leaf_application_abi":
        source_leaf["application_abi"] = "wrong"
    elif corruption == "source_leaf_optimization":
        source_leaf["optimization_level"] = 1
    elif corruption == "source_leaf_path":
        source_leaf["application_path"] = "kernels/000000/missing.symjit"
    elif corruption == "source_payload_digest_link":
        binding["source_application_sha256"] = "0" * 64
        refresh_catalog = True
    else:
        pytest.fail(f"unknown corruption {corruption}")

    if refresh_catalog is not None:
        _refresh_recurrence_source_catalog(
            execution,
            pack,
            refresh_binding_digest=refresh_catalog,
        )
    pack_path = tmp_path / pack_relative
    pack_path.parent.mkdir(parents=True)
    pack_data = json.dumps(pack, sort_keys=True).encode("ascii")
    pack_path.write_bytes(pack_data)
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_data)
    manifest = SimpleNamespace(
        payloads=(
            SimpleNamespace(
                path=pack_relative,
                role="evaluator-manifest",
                media_type="application/json",
                process_id=None,
                executable=False,
                size_bytes=len(pack_data),
                sha256=hashlib.sha256(pack_data).hexdigest(),
            ),
            SimpleNamespace(
                path=source_relative,
                role="evaluator-state",
                media_type="application/octet-stream",
                process_id=None,
                executable=False,
                size_bytes=len(source_data),
                sha256=source_sha256,
            ),
        ),
        extensions={},
    )
    with pytest.raises(FinalAuditError, match=message):
        _audit_recurrence_source_pack(
            tmp_path,
            manifest,
            execution,
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
        )


def test_pointwise_audit_recomputes_arithmetic_and_status() -> None:
    record = {
        "status": "ok",
        "candidate": 1.0 + 1.0e-13,
        "baseline": 1.0,
        "absolute_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    _audit_pointwise(
        record,
        context="pointwise",
        expected_candidate=1.0 + 1.0e-13,
        expected_baseline=1.0,
        expected_relative_tolerance=1.0e-12,
    )
    record["relative_difference"] = 0.0
    with pytest.raises(FinalAuditError, match="relative difference"):
        _audit_pointwise(
            record,
            context="pointwise",
            expected_candidate=1.0 + 1.0e-13,
            expected_baseline=1.0,
            expected_relative_tolerance=1.0e-12,
        )


def test_python_package_tree_identity_is_canonical_and_ignores_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.performance_report import runtime_evidence

    cache_prefix = tmp_path / "absent-cache-prefix"
    monkeypatch.setattr(
        runtime_evidence,
        "_isolated_startup_flags",
        lambda: (True, True, True),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(cache_prefix))
    package = tmp_path / "pyamplicol"
    (package / "submodule").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "api.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "submodule" / "data.bin").write_bytes(b"\x00\x01")
    (package / "__pycache__" / "api.pyc").write_bytes(b"cache one")

    first = _python_package_tree_identity(package)
    assert first["kind"] == "pyamplicol-python-package-tree-v2"
    assert first["roots"] == [str(package)]
    assert first["file_count"] == 2
    assert first["total_bytes"] == 12

    (package / "__pycache__" / "api.pyc").write_bytes(b"changed cache")
    assert _python_package_tree_identity(package) == first

    (package / "api.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _python_package_tree_identity(package)
    assert changed["sha256"] != first["sha256"]


def test_runtime_namespace_uses_only_first_native_candidate(tmp_path: Path) -> None:
    from importlib.machinery import EXTENSION_SUFFIXES

    checkout = tmp_path / "checkout"
    source_package = checkout / "src" / "pyamplicol"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_bytes(b"")
    first_site = tmp_path / "first-site"
    second_site = tmp_path / "second-site"
    first_package = first_site / "pyamplicol"
    second_package = second_site / "pyamplicol"
    first_package.mkdir(parents=True)
    second_package.mkdir(parents=True)
    suffix = EXTENSION_SUFFIXES[0]
    first_native = first_package / f"_rusticol{suffix}"
    second_native = second_package / f"_rusticol{suffix}"
    first_native.write_bytes(b"first")
    second_native.write_bytes(b"stale")

    roots, native = _runtime_namespace_paths(
        checkout,
        search_path=(str(first_site), str(second_site)),
    )

    assert roots == (source_package.resolve(), first_package.resolve())
    assert native == first_native.resolve()
    assert second_package.resolve() not in roots


def test_runtime_namespace_rejects_ambiguous_first_native_root(
    tmp_path: Path,
) -> None:
    from importlib.machinery import EXTENSION_SUFFIXES

    checkout = tmp_path / "checkout"
    (checkout / "src" / "pyamplicol").mkdir(parents=True)
    site = tmp_path / "site"
    package = site / "pyamplicol"
    package.mkdir(parents=True)
    suffix = EXTENSION_SUFFIXES[0]
    (package / f"_rusticol{suffix}").write_bytes(b"one")
    (package / f"_rusticol_duplicate{suffix}").write_bytes(b"two")

    with pytest.raises(FinalAuditError, match="ambiguous"):
        _runtime_namespace_paths(checkout, search_path=(str(site),))


def test_direct_final_audit_cli_fails_closed_when_not_already_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.performance_report.runtime_evidence import RuntimeEvidenceError

    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        "tools.performance_report.final_audit.source_only_bytecode_policy",
        lambda: (_ for _ in ()).throw(RuntimeEvidenceError("not isolated")),
    )
    raw = ("--repo-root", str(checkout), "--expected-source-revision", _REVISION)
    with pytest.raises(
        FinalAuditError,
        match=r"docs/result_tables\.py final-audit",
    ):
        _ensure_exact_cli_python(checkout, raw)


def test_master_tex_dependency_graph_reaches_every_canonical_table(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    sections = docs / "sections"
    tables = docs / "tables"
    sections.mkdir(parents=True)
    tables.mkdir()
    master = docs / "pyAmpliCol.tex"
    section = sections / "performance.tex"
    first = tables / "result_first_table.tex"
    second = tables / "result_second_table.tex"
    unreachable = tables / "result_unreachable_table.tex"
    for table in (first, second, unreachable):
        table.write_text(f"% {table.name}\n", encoding="utf-8")
    master.write_text(
        "\\input{sections/performance}\n"
        "% \\\\input{tables/result_unreachable_table.tex}\n",
        encoding="utf-8",
    )
    section.write_text(
        "\\input{../tables/result_first_table.tex}\n"
        "\\include{../tables/result_second_table}\n",
        encoding="utf-8",
    )

    identity = _audit_tex_table_reachability(master, (first, second))
    assert identity["reachable_table_source_count"] == 2
    assert identity["reachable_tex_source_count"] == 4

    with pytest.raises(FinalAuditError, match="not reachable"):
        _audit_tex_table_reachability(master, (first, second, unreachable))


def test_active_runtime_snapshot_binds_exact_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol
    from pyamplicol._internal import versions

    native_path = tmp_path / "_rusticol.so"
    native_path.write_bytes(b"native")
    native = SimpleNamespace(
        __file__=str(native_path),
        package_version=lambda: pyamplicol.__version__,
        native_build_inputs_sha256=lambda: "e" * 64,
        target_info=lambda: SimpleNamespace(
            triple="aarch64-apple-darwin",
            cpu_features=(),
        ),
    )
    package_root = Path(str(pyamplicol.__file__)).parent.resolve()
    package_tree = {
        "kind": "pyamplicol-python-package-tree-v2",
        "root": str(package_root),
        "roots": [str(package_root)],
        "file_count": 1,
        "total_bytes": 1,
        "sha256": "9" * 64,
        "member_set_stable": True,
        "namespace_bound_to_root_fd": True,
        "bytecode_policy": _bytecode_policy(),
    }
    native_identity = {
        "path": str(native_path),
        "size": len(b"native"),
        "sha256": hashlib.sha256(b"native").hexdigest(),
    }
    preimport_identity = {
        "kind": "pyamplicol-preimport-runtime-identity-v1",
        "python_package_tree": package_tree,
        "native_extension": native_identity,
    }
    monkeypatch.setattr(
        "tools.performance_report.final_audit._prepare_exact_pyamplicol_namespace",
        lambda _checkout: (
            pyamplicol,
            (package_root,),
            native_path,
            preimport_identity,
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit._python_package_tree_identity",
        lambda roots: (
            package_tree
            if tuple(roots) == (package_root,)
            else pytest.fail(f"unexpected package roots: {roots}")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit.loaded_pyamplicol_origin_policy",
        lambda roots, **kwargs: (
            _loaded_origin_policy()
            if (
                tuple(roots) == (package_root,)
                and kwargs["native_extension"] == native_path
                and kwargs["expected_package_identity"] == package_tree
                and kwargs["expected_native_identity"] == native_identity
            )
            else pytest.fail("unexpected loaded-origin evidence request")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.final_audit.importlib.import_module",
        lambda name: native if name == "pyamplicol._rusticol" else None,
    )
    monkeypatch.setattr(versions, "verify_native_module", lambda _native: None)
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "schema_version": 1,
            "version": pyamplicol.__version__,
            "candidate_fingerprint": "candidate",
            "source_revision": _REVISION,
            "source_checkout": str(tmp_path),
            "native_build_inputs_sha256": "e" * 64,
            "publishable": False,
        },
    )

    snapshot = _active_runtime_snapshot(
        _REVISION,
        expected_checkout=tmp_path,
    )
    assert snapshot["native_build_inputs_sha256"] == "e" * 64
    assert snapshot["python_package_tree"]["kind"] == (  # type: ignore[index]
        "pyamplicol-python-package-tree-v2"
    )
    assert snapshot["loaded_module_origin_policy"] == _loaded_origin_policy()

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FinalAuditError, match="not bound"):
        _active_runtime_snapshot(_REVISION, expected_checkout=other)


@dataclass(frozen=True)
class _Flow:
    id: str
    word: tuple[int, ...]


@dataclass(frozen=True)
class _Helicity:
    id: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class _Particle:
    label: int


class _Resolved:
    def total(self) -> tuple[complex, ...]:
        return (1.0 + 0.0j,)


class _Runtime:
    artifact_id = _ARTIFACT_ID
    execution_mode = "recurrence"

    def __init__(self) -> None:
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            external_particles=(_Particle(1), _Particle(2), _Particle(3)),
            helicities=(_Helicity("h:-1,+1,-1", (-1, 1, -1)),),
            color_flows=(_Flow("flow:2,1", (2, 1)),),
        )

    def evaluate(
        self,
        _points: object,
        **_kwargs: object,
    ) -> tuple[complex, ...]:
        return (1.0 + 0.0j,)

    def evaluate_resolved(
        self,
        _points: object,
        **_kwargs: object,
    ) -> _Resolved:
        return _Resolved()


class _Catalog:
    def __init__(self, baseline: CellSpec, candidate: CellSpec) -> None:
        self.baseline = baseline
        self.candidate = candidate

    def measurement_cells(self) -> tuple[CellSpec, ...]:
        return (self.baseline, self.candidate)

    def baseline_cell(self, cell: CellSpec) -> CellSpec | None:
        return self.baseline if cell == self.candidate else None


def _selector() -> dict[str, object]:
    return {
        "selected_color_flow_ids": ["flow:2,1"],
        "selected_color_words": [[2, 1]],
        "all_flow_helicity_ids": ["h:-1,+1,-1"],
        "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
        "point_digest": (
            "20d4353b25501bdf6c980930babe948e7c1d0364a3169d1ef5dc2a9da427e05c"
        ),
    }


def _source_provenance() -> dict[str, object]:
    tree = "c" * 40
    return {
        "report_source_identity_schema": "pyamplicol-report-source-v1",
        "report_source_revision": _REVISION,
        "report_source_tree": tree,
        "report_measured_source_revision": _REVISION,
        "report_measured_source_tree": tree,
        "report_source_clean": True,
    }


def _baseline_measurement() -> dict[str, object]:
    result = empty_measurement()
    result.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 1.0e-6,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-9,
            "relative_standard_error": 1.0e-3,
            "artifact": {"path": "/legacy", "process_row": "row"},
            "selector_contract": _selector(),
            "validation": {
                "status": "ok",
                "method": "independent-original-amplicol-oracle",
                "point_digest": _selector()["point_digest"],
            },
            "resources": {"available": True},
            "provenance": {
                **_source_provenance(),
                "revision": _LEGACY_REVISION,
                "runtime_profile": {
                    "measurement": {
                        "target_runtime_seconds": 5.0,
                        "achieved_runtime_seconds": 5.0,
                        "target_runtime_achieved": True,
                        "chunk_count": 5,
                        "interrupted": False,
                    }
                },
            },
            "failure": None,
        }
    )
    return result


def _bytecode_policy() -> dict[str, object]:
    return {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }


def _loaded_origin_policy() -> dict[str, object]:
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 1,
            "sha256": "1" * 64,
        }
    ]
    return {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": digest_json(observations),
    }


def _runtime_identity_provenance(
    identity: Mapping[str, object],
    *,
    postflight_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stable_identity = deepcopy(dict(identity))
    stable_policy = stable_identity.get("loaded_module_origin_policy")
    assert isinstance(stable_policy, dict)
    for field in (
        "observed_module_count",
        "observations",
        "observations_sha256",
    ):
        stable_policy.pop(field)
    stable_digest = digest_json(stable_identity)
    return {
        "runtime_identity": dict(identity),
        "runtime_identity_sha256": digest_json(identity),
        "runtime_identity_stable_sha256": stable_digest,
        "runtime_identity_postflight_stable_sha256": stable_digest,
        "runtime_identity_postflight_loaded_module_origin_policy": deepcopy(
            identity["loaded_module_origin_policy"]
            if postflight_policy is None
            else postflight_policy
        ),
        "runtime_identity_postflight_match": True,
    }


def _active_runtime() -> dict[str, object]:
    candidate = {
        "schema_version": 1,
        "version": "0.1.0.dev0+candidate.test",
        "candidate_fingerprint": "test",
        "source_revision": _REVISION,
        "source_checkout": "/repo",
        "native_build_inputs_sha256": "e" * 64,
        "publishable": False,
    }
    return {
        "package_version": candidate["version"],
        "python_package_tree": {
            "kind": "pyamplicol-python-package-tree-v2",
            "root": "/runtime/pyamplicol",
            "roots": [
                "/runtime/pyamplicol",
                "/candidate/site-packages/pyamplicol",
            ],
            "file_count": 10,
            "total_bytes": 1000,
            "sha256": "9" * 64,
            "member_set_stable": True,
            "namespace_bound_to_root_fd": True,
            "bytecode_policy": _bytecode_policy(),
        },
        "loaded_module_origin_policy": _loaded_origin_policy(),
        "candidate_build_identity": candidate,
        "candidate_build_identity_sha256": digest_json(candidate),
        "native_build_inputs_sha256": "e" * 64,
        "native_extension": {
            "path": "/runtime/_rusticol.so",
            "sha256": "f" * 64,
            "package_version": candidate["version"],
        },
        "native_target": {
            "triple": "aarch64-apple-darwin",
            "cpu_features": [],
        },
    }


def _candidate_measurement(artifact: Path) -> dict[str, object]:
    active = _active_runtime()
    capabilities = [_COLOR_CAPABILITY, _CAPABILITY]
    identity = {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "artifact_id": _ARTIFACT_ID,
        "loaded_artifact_id": _ARTIFACT_ID,
        "artifact_identity_match": True,
        "process_id": "d_dbar_to_z",
        "execution_mode": "recurrence",
        "loaded_execution_mode": "recurrence",
        "backend": "jit",
        "required_arena_capability": _CAPABILITY,
        "expected_evaluator_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "expected_source_evaluator_abi": "symjit-application-storage-v3",
        "expected_source_evaluator_runtime_capability": (
            "symjit.application.complex-f64.v1"
        ),
        "source_jit_optimization_level": 2,
        "source_jit_identity": {
            "kind": "authenticated-recurrence-direct-template-source-v1",
            "optimization_level": 2,
            "direct_template_count": 4,
            "prepared_direct_template_count": 3,
            "source_evaluator_leaf_count": 5,
            "source_application_abi": "symjit-application-storage-v3",
            "direct_application_abi": "symjit-direct-application-storage-v1",
            "prepared_kernel_pack_digest": "7" * 64,
            "direct_template_catalog_digest": "8" * 64,
            "execution_manifest_path": "processes/d_dbar_to_z/execution.json",
            "execution_manifest_sha256": "1" * 64,
            "kernel_pack_path": "model/eager-kernel-pack.json",
            "kernel_pack_sha256": "2" * 64,
        },
        "process_required_runtime_capabilities": capabilities,
        "package_version": active["package_version"],
        "python_package_tree": active["python_package_tree"],
        "loaded_module_origin_policy": active["loaded_module_origin_policy"],
        "artifact_runtime_version": active["package_version"],
        "source_revision": _REVISION,
        "candidate_build_identity": active["candidate_build_identity"],
        "candidate_build_identity_sha256": active["candidate_build_identity_sha256"],
        "native_build_inputs_sha256": active["native_build_inputs_sha256"],
        "native_extension": active["native_extension"],
        "native_target": active["native_target"],
    }
    result = empty_measurement()
    result.update(
        {
            "status": "ok",
            "generation_seconds": 2.0,
            "wall_seconds_per_point": 2.0e-6,
            "execution_seconds_per_point": 1.5e-6,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-8,
            "relative_standard_error": 0.01,
            "artifact": {
                "path": str(artifact),
                "process_id": "d_dbar_to_z",
                "policy": "generated",
            },
            "selector_contract": _selector(),
            "validation": {
                "status": "ok",
                "resolved_sum": {
                    "status": "ok",
                    "maximum_absolute_difference": 0.0,
                    "maximum_relative_difference": 0.0,
                    "relative_tolerance": 1.0e-12,
                    "absolute_tolerance": 1.0e-15,
                },
                "pointwise": {
                    "status": "ok",
                    "candidate": 1.0,
                    "baseline": 1.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "relative_tolerance": 1.0e-8,
                    "absolute_tolerance": 1.0e-15,
                },
            },
            "resources": {"available": True},
            "provenance": {
                **_source_provenance(),
                "source_revision": _REVISION,
                "runtime_profile": {
                    "target_runtime_seconds": 5.0,
                    "achieved_runtime_seconds": 5.0,
                    "target_runtime_achieved": True,
                    "completed_sample_count": 5,
                    "planned_sample_count": 5,
                    "repetitions_per_sample": 128,
                    "measured_point_count": 640,
                    "interrupted": False,
                },
                "effective_config": {
                    "evaluator": {
                        "execution_mode": "recurrence",
                        "backend": "jit",
                        "jit": {"optimization_level": 2},
                    },
                    "color": {
                        "accuracy": "lc",
                        "lc_flow_layout": "topology-replay",
                    },
                    "generation": {
                        "validation": {
                            "enabled": True,
                            "samples": 10,
                            "seed": 12345,
                            "relative_tolerance": 1.0e-12,
                            "absolute_tolerance": 1.0e-300,
                            "post_build_validation": True,
                        }
                    },
                },
                **_runtime_identity_provenance(identity),
            },
            "failure": None,
        }
    )
    return result


def test_portable_artifact_locator_resolves_only_within_profile_root(
    tmp_path: Path,
) -> None:
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "profile-artifacts",
    )
    artifact = paths.artifact_root / "cells/example/artifact"
    artifact.mkdir(parents=True)
    measurement = {
        "artifact": {
            "path": ("${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/cells/example/artifact"),
            "process_id": "d_dbar_to_z",
        }
    }

    reference = _artifact_reference(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        measurement,
        report_paths=paths,
    )

    assert reference.path == artifact
    escaped = deepcopy(measurement)
    escaped["artifact"]["path"] = (  # type: ignore[index]
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/../outside"
    )
    with pytest.raises(FinalAuditError, match="not canonical"):
        _artifact_reference(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            escaped,
            report_paths=paths,
        )


def test_legacy_measurement_requires_pinned_revision_and_physical_point_digest() -> (
    None
):
    cell = _legacy_cell()
    measurement = _baseline_measurement()
    assert (
        _audit_measurement(
            cell,
            measurement,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )
        is None
    )

    wrong_revision = deepcopy(measurement)
    wrong_revision["provenance"]["revision"] = "d" * 40  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="pinned original-AmpliCol revision"):
        _audit_measurement(
            cell,
            wrong_revision,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )

    wrong_points = deepcopy(measurement)
    wrong_points["validation"]["point_digest"] = "0" * 64  # type: ignore[index]
    with pytest.raises(FinalAuditError, match="physical process"):
        _audit_measurement(
            cell,
            wrong_points,
            baseline=None,
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=None,
        )


@pytest.mark.parametrize("execution_seconds", [0.0, -1.0, float("inf")])
def test_candidate_measurement_requires_positive_finite_execution_time(
    tmp_path: Path,
    execution_seconds: float,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    measurement["execution_seconds_per_point"] = execution_seconds
    with pytest.raises(
        FinalAuditError,
        match="execution_seconds_per_point",
    ):
        _audit_measurement(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            measurement,
            baseline=_baseline_measurement(),
            expected_source_revision=_REVISION,
            expected_legacy_revision=_LEGACY_REVISION,
            active_runtime=_active_runtime(),
        )


def _below_resolution_execution_timing() -> dict[str, object]:
    return {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "below_timer_resolution",
        "ratio_eligible": False,
        "raw_seconds_per_point": 0.0,
        "source": ("runtime_profile_core_compiled_direct_arena_orchestration_time"),
        "compiled_direct_arena_active": True,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "sample_contract": "paired_unprofiled_headline_profiled_attribution_v1",
    }


def test_final_audit_accepts_only_authenticated_compiled_o3_timer_zero() -> None:
    provenance = {"execution_timing": _below_resolution_execution_timing()}
    _audit_below_resolution_execution_timing(
        _cell(ExecutionMode.COMPILED, optimization_level=3),
        provenance,
        measurement_sample_count=5,
        context="candidate",
    )

    for cell in (
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        _cell(ExecutionMode.EAGER, optimization_level=2),
        _cell(ExecutionMode.COMPILED, optimization_level=1),
        _cell(ExecutionMode.COMPILED, backend="cpp", optimization_level=None),
    ):
        with pytest.raises(FinalAuditError, match="only for compiled JIT O3"):
            _audit_below_resolution_execution_timing(
                cell,
                provenance,
                measurement_sample_count=5,
                context="candidate",
            )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("abi", "wrong"),
        ("status", "measured"),
        ("ratio_eligible", True),
        ("raw_seconds_per_point", 1.0e-30),
        ("raw_seconds_per_point", float("nan")),
        ("source", "runtime_profile_core_evaluator_call_time"),
        ("compiled_direct_arena_active", False),
        ("sample_count", 4),
        ("sample_count", True),
        ("native_profile_points_per_sample", 0),
        ("native_profile_points_per_sample", None),
        ("sample_contract", "separate_native_profile_diagnostic_v1"),
    ),
)
def test_final_audit_rejects_tampered_below_resolution_timing(
    field: str,
    value: object,
) -> None:
    timing = _below_resolution_execution_timing()
    timing[field] = value
    with pytest.raises(FinalAuditError, match="below-resolution record"):
        _audit_below_resolution_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            {"execution_timing": timing},
            measurement_sample_count=5,
            context="candidate",
        )


def test_final_audit_binds_below_resolution_timing_shape_and_sample_count() -> None:
    timing = _below_resolution_execution_timing()
    timing["unexpected"] = True
    with pytest.raises(FinalAuditError, match="fields do not match"):
        _audit_below_resolution_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            {"execution_timing": timing},
            measurement_sample_count=5,
            context="candidate",
        )

    with pytest.raises(FinalAuditError, match="below-resolution record"):
        _audit_below_resolution_execution_timing(
            _cell(ExecutionMode.COMPILED, optimization_level=3),
            {"execution_timing": _below_resolution_execution_timing()},
            measurement_sample_count=6,
            context="candidate",
        )


def test_runtime_identity_audit_distinguishes_source_and_direct_codegen_levels(
    tmp_path: Path,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    raw_provenance = measurement["provenance"]
    assert isinstance(raw_provenance, dict)
    raw_identity = raw_provenance["runtime_identity"]
    assert isinstance(raw_identity, dict)

    def provenance(identity: dict[str, object]) -> dict[str, object]:
        return _runtime_identity_provenance(identity)

    recurrence = _cell(ExecutionMode.RECURRENCE, optimization_level=2)
    _audit_runtime_identity(
        recurrence,
        provenance(raw_identity),
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )

    missing_source = deepcopy(raw_identity)
    del missing_source["source_jit_optimization_level"]
    with pytest.raises(FinalAuditError, match="source_jit_optimization_level"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_source),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    missing_source_identity = deepcopy(raw_identity)
    del missing_source_identity["source_jit_identity"]
    with pytest.raises(FinalAuditError, match="source_jit_identity"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_source_identity),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    wrong_source_capability = deepcopy(raw_identity)
    wrong_source_capability["expected_source_evaluator_runtime_capability"] = (
        "symbolica.compiled-cpp.complex-f64.v1"
    )
    with pytest.raises(
        FinalAuditError,
        match="expected_source_evaluator_runtime_capability",
    ):
        _audit_runtime_identity(
            recurrence,
            provenance(wrong_source_capability),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    missing_loaded_mode = deepcopy(raw_identity)
    del missing_loaded_mode["loaded_execution_mode"]
    with pytest.raises(FinalAuditError, match="loaded_execution_mode"):
        _audit_runtime_identity(
            recurrence,
            provenance(missing_loaded_mode),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    spurious_direct_codegen = deepcopy(raw_identity)
    spurious_direct_codegen["direct_codegen_optimization_level"] = 3
    with pytest.raises(FinalAuditError, match="direct_codegen_optimization_level"):
        _audit_runtime_identity(
            recurrence,
            provenance(spurious_direct_codegen),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )

    compiled_identity = deepcopy(raw_identity)
    del compiled_identity["source_jit_identity"]
    compiled_identity.update(
        {
            "execution_mode": "compiled",
            "loaded_execution_mode": "compiled",
            "required_arena_capability": "compiled-plane-arena-v1",
            "expected_evaluator_abi": "symjit-direct-application-storage-v3",
            "expected_source_evaluator_abi": "symjit-application-storage-v3",
            "source_jit_optimization_level": 1,
            "direct_codegen_optimization_level": 3,
            "direct_codegen_identity": {
                "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
                "optimization_level": 3,
                "source_optimization_level": 1,
                "leaf_count": 2,
                "execution_manifest_path": "execution.json",
                "execution_manifest_sha256": "8" * 64,
            },
        }
    )
    compiled = _cell(ExecutionMode.COMPILED, optimization_level=1)
    _audit_runtime_identity(
        compiled,
        provenance(compiled_identity),
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )
    compiled_effective = {"evaluator": {"execution_mode": "compiled"}}
    compiled_evidence = ArtifactEvidence(
        artifact_id=_ARTIFACT_ID,
        process_id="d_dbar_to_z",
        runtime_version=str(_active_runtime()["package_version"]),
        runtime_capabilities=(_COLOR_CAPABILITY, _CAPABILITY),
        execution_manifest_path="execution.json",
        execution_manifest_sha256="8" * 64,
        execution_mode="compiled",
        arena_record_count=2,
        direct_leaf_count=2,
        effective_config=compiled_effective,
        source_jit_identity=None,
    )
    compiled_provenance = provenance(compiled_identity)
    compiled_provenance["effective_config"] = compiled_effective
    _audit_runtime_identity(
        compiled,
        compiled_provenance,
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=compiled_evidence,
    )

    wrong_leaf_count = deepcopy(compiled_identity)
    wrong_leaf_count["direct_codegen_identity"]["leaf_count"] = 3  # type: ignore[index]
    wrong_leaf_provenance = provenance(wrong_leaf_count)
    wrong_leaf_provenance["effective_config"] = compiled_effective
    with pytest.raises(FinalAuditError, match=r"direct_codegen_identity\.leaf_count"):
        _audit_runtime_identity(
            compiled,
            wrong_leaf_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    wrong_path = deepcopy(compiled_identity)
    wrong_path["direct_codegen_identity"]["execution_manifest_path"] = (  # type: ignore[index]
        "other.json"
    )
    wrong_path_provenance = provenance(wrong_path)
    wrong_path_provenance["effective_config"] = compiled_effective
    with pytest.raises(
        FinalAuditError,
        match=r"direct_codegen_identity\.execution_manifest_path",
    ):
        _audit_runtime_identity(
            compiled,
            wrong_path_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    wrong_effective_provenance = provenance(compiled_identity)
    wrong_effective_provenance["effective_config"] = {
        "evaluator": {"execution_mode": "eager"}
    }
    with pytest.raises(FinalAuditError, match="effective_config"):
        _audit_runtime_identity(
            compiled,
            wrong_effective_provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=compiled_evidence,
        )

    missing_direct_codegen = deepcopy(compiled_identity)
    del missing_direct_codegen["direct_codegen_optimization_level"]
    with pytest.raises(FinalAuditError, match="direct_codegen_optimization_level"):
        _audit_runtime_identity(
            compiled,
            provenance(missing_direct_codegen),
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "runtime_identity_stable_sha256",
            "0" * 64,
            "runtime_identity_stable_sha256",
        ),
        (
            "runtime_identity_postflight_stable_sha256",
            "0" * 64,
            "postflight stable SHA-256",
        ),
        (
            "runtime_identity_postflight_match",
            False,
            "runtime_identity_postflight_match",
        ),
        (
            "runtime_identity_postflight_loaded_module_origin_policy",
            {},
            "loaded-origin policy",
        ),
    ],
)
def test_runtime_identity_audit_requires_per_cell_postflight_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value
    with pytest.raises(FinalAuditError, match=match):
        _audit_runtime_identity(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_runtime_identity_audit_accepts_only_monotonic_postflight_origins(
    tmp_path: Path,
) -> None:
    measurement = _candidate_measurement(tmp_path)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    raw_postflight = provenance[
        "runtime_identity_postflight_loaded_module_origin_policy"
    ]
    assert isinstance(raw_postflight, dict)
    observations = raw_postflight["observations"]
    assert isinstance(observations, list)
    observations.append(
        {
            "module": "pyamplicol.runtime",
            "kind": "package-member",
            "root_index": 0,
            "path": "runtime/__init__.py",
            "size": 2,
            "sha256": "2" * 64,
        }
    )
    raw_postflight["observed_module_count"] = len(observations)
    raw_postflight["observations_sha256"] = digest_json(observations)
    _audit_runtime_identity(
        _cell(ExecutionMode.RECURRENCE, optimization_level=2),
        provenance,
        expected_source_revision=_REVISION,
        active_runtime=_active_runtime(),
        artifact=None,
    )

    observations[0] = {
        "module": "pyamplicol.replaced",
        "kind": "package-member",
        "root_index": 0,
        "path": "replaced.py",
        "size": 3,
        "sha256": "3" * 64,
    }
    raw_postflight["observations_sha256"] = digest_json(observations)
    with pytest.raises(FinalAuditError, match="lost a loaded-module origin"):
        _audit_runtime_identity(
            _cell(ExecutionMode.RECURRENCE, optimization_level=2),
            provenance,
            expected_source_revision=_REVISION,
            active_runtime=_active_runtime(),
            artifact=None,
        )


def test_final_audit_authenticates_cache_store_and_replays_unique_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "contributor-lock.toml").write_text(
        f'[legacy_amplicol]\nrevision = "{_LEGACY_REVISION}"\n',
        encoding="utf-8",
    )
    baseline = _legacy_cell()
    candidate = CellSpec(
        dataset_id="matrix_recurrence_test",
        process="d d~ > z",
        n_final=1,
        process_key="dd_z",
        measurement=MeasurementSpec(
            ExecutionMode.RECURRENCE,
            ModelKey.BUILTIN_SM,
            Accuracy.LC,
            "jit",
            2,
        ),
        workload=Workload.SELECTED_FLOW,
    )
    catalog = _Catalog(baseline, candidate)
    paths = ReportPaths.from_repo(
        tmp_path,
        artifact_root=tmp_path / "artifacts",
        coordination_root=tmp_path / "locks",
    )
    service = ReportService(paths, catalog=catalog)  # type: ignore[arg-type]
    artifact = paths.artifact_root / "synthetic-artifact"
    artifact.mkdir()
    measurements = {
        baseline.cell_id: _baseline_measurement(),
        candidate.cell_id: _candidate_measurement(artifact),
    }
    for cell in catalog.measurement_cells():
        service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
            measurements[cell.cell_id]
        )
    caches = build_reset_caches(catalog)  # type: ignore[arg-type]
    for payload in caches.values():
        for entry in payload["entries"]:
            entry["measurement"] = portable_publication_value(
                measurements[entry["cell_id"]],
                paths,
            )
    paths.results_dir.mkdir(parents=True)
    for name, payload in caches.items():
        (paths.results_dir / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="ascii",
        )
    lock_state = {"held": False, "entries": 0}

    @contextmanager
    def named_lock(name: str):
        assert name == "report-writer"
        assert lock_state["held"] is False
        lock_state["held"] = True
        lock_state["entries"] += 1
        try:
            yield
        finally:
            lock_state["held"] = False

    def assert_locked() -> None:
        assert lock_state["held"] is True

    def render_auditor() -> dict[str, object]:
        assert_locked()
        return {"cache_render_match": True, "table_count": 1}

    monkeypatch.setattr(service.store, "named_lock", named_lock)
    monkeypatch.setattr(service, "audit", render_auditor)

    def artifact_auditor(
        _cell: CellSpec,
        path: Path,
        process_id: str,
    ) -> ArtifactEvidence:
        assert_locked()
        assert path == artifact
        candidate_provenance = measurements[candidate.cell_id]["provenance"]
        assert isinstance(candidate_provenance, dict)
        effective_config = candidate_provenance["effective_config"]
        assert isinstance(effective_config, dict)
        runtime_identity = candidate_provenance["runtime_identity"]
        assert isinstance(runtime_identity, dict)
        source_jit_identity = runtime_identity["source_jit_identity"]
        assert isinstance(source_jit_identity, dict)
        return ArtifactEvidence(
            artifact_id=_ARTIFACT_ID,
            process_id=process_id,
            runtime_version=str(_active_runtime()["package_version"]),
            runtime_capabilities=(_COLOR_CAPABILITY, _CAPABILITY),
            execution_manifest_path="processes/d_dbar_to_z/execution.json",
            execution_manifest_sha256="1" * 64,
            execution_mode="recurrence",
            arena_record_count=4,
            direct_leaf_count=0,
            effective_config=effective_config,
            source_jit_identity=source_jit_identity,
        )

    def runtime_loader(_path: Path, _process: str) -> _Runtime:
        assert_locked()
        return _Runtime()

    def source_auditor(_root: Path, _revision: str) -> None:
        assert_locked()

    def runtime_auditor(_revision: str, _root: Path) -> dict[str, object]:
        assert_locked()
        return _active_runtime()

    def pdf_auditor(_service: ReportService) -> dict[str, object]:
        assert_locked()
        return {
            "status": "ok",
            "published_sha256": "2" * 64,
        }

    result = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=True,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        runtime_loader=runtime_loader,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        pdf_auditor=pdf_auditor,
    )

    assert lock_state == {"held": False, "entries": 1}
    assert result["status"] == "incomplete"
    assert result["schema_version"] == 2
    assert result["measurement_source_revision"] == _REVISION
    assert result["publication_revision"] == _REVISION
    assert result["publication_lineage"]["relationship"] == "same-commit"
    assert result["authenticated_current_count"] == 2
    assert result["portable_publication_projection_count"] == 2
    assert result["publication_cache_role"] == ("portable-projection-of-current-result")
    assert result["cryptographic_audit_source"] == "immutable-current-result"
    assert result["numerically_evidenced_cell_count"] == 2
    assert result["legacy_fresh_oracle_count"] == 1
    assert result["legacy_oracles_with_inbound_agreement"] == 1
    assert result["legacy_pointwise_agreement_edge_count"] == 1
    assert result["unique_artifact_count"] == 1
    assert result["canonical_publication_scope"] is False
    assert result["final_gate_complete"] is False
    assert result["audit_scope"] == "diagnostic-incomplete"
    assert result["pyamplicol_replay_count"] == 1
    assert result["replayed_measurement_count"] == 1

    def wrong_runtime_loader(_path: Path, _process: str) -> _Runtime:
        assert_locked()
        runtime = _Runtime()
        runtime.execution_mode = "compiled"
        return runtime

    with pytest.raises(FinalAuditError, match="loaded execution mode"):
        audit_final_report(
            tmp_path,
            expected_source_revision=_REVISION,
            max_n_final=1,
            expected_cell_count=2,
            replay=True,
            catalog=catalog,  # type: ignore[arg-type]
            service=service,
            active_runtime=_active_runtime(),
            artifact_auditor=artifact_auditor,
            runtime_loader=wrong_runtime_loader,
            source_auditor=source_auditor,
            runtime_auditor=runtime_auditor,
            pdf_auditor=pdf_auditor,
        )
    assert lock_state == {"held": False, "entries": 2}

    structural = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=False,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        pdf_auditor=pdf_auditor,
    )
    assert structural["status"] == "incomplete"
    assert structural["final_gate_complete"] is False
    assert structural["audit_scope"] == "diagnostic-incomplete"
    assert structural["pyamplicol_replay_count"] == 0

    no_pdf = audit_final_report(
        tmp_path,
        expected_source_revision=_REVISION,
        max_n_final=1,
        expected_cell_count=2,
        replay=True,
        catalog=catalog,  # type: ignore[arg-type]
        service=service,
        active_runtime=_active_runtime(),
        artifact_auditor=artifact_auditor,
        runtime_loader=runtime_loader,
        source_auditor=source_auditor,
        runtime_auditor=runtime_auditor,
        verify_pdf=False,
    )
    assert no_pdf["status"] == "incomplete"
    assert no_pdf["final_gate_complete"] is False
    assert no_pdf["pdf_audit"] == {"status": "incomplete", "skipped": True}
    assert lock_state == {"held": False, "entries": 4}
