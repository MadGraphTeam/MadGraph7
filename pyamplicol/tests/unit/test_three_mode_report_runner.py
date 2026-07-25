# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    Workload,
)
from tools.performance_report.runner import (
    RunnerError,
    RunnerSettings,
    SelectorContract,
    _authenticated_direct_codegen_identity,
    _authenticated_effective_config,
    _authenticated_recurrence_source_identity,
    _benchmark_measurement,
    _real_nonnegative,
    _regular_file_identity,
    config_values,
    derive_selector_contract,
    point_digest,
    pointwise_validation,
    resolved_sum_validation,
    runtime_identity_payload,
    validate_artifact_contract,
    validate_runtime_contract,
    validate_selector_contract,
)


def _benchmark_fixture(
    *,
    evaluator_time: float | None,
    timing_status: str,
    compiled_direct_arena_active: bool,
) -> SimpleNamespace:
    raw_time = 0.0 if evaluator_time is None else evaluator_time
    return SimpleNamespace(
        uncertainty=SimpleNamespace(
            standard_error=1.0e-9,
            relative_standard_error=0.01,
        ),
        wall_time_per_point=1.0e-6,
        evaluator_time_per_point=evaluator_time,
        sample_count=5,
        environment={
            "evaluator_time_raw_seconds_per_point": raw_time,
            "evaluator_time_status": timing_status,
            "evaluator_time_ratio_eligible": evaluator_time is not None,
            "evaluator_time_source": (
                "runtime_profile_core_compiled_direct_arena_orchestration_time"
                if compiled_direct_arena_active
                else "runtime_profile_core_evaluator_call_time"
            ),
            "compiled_direct_arena_active": compiled_direct_arena_active,
            "evaluator_sample_count": 5,
            "native_profile_points_per_sample": 128,
            "timing_sample_contract": (
                "paired_unprofiled_headline_profiled_attribution_v1"
            ),
        },
    )


def test_benchmark_measurement_records_compiled_zero_below_timer_resolution() -> None:
    measurement = _benchmark_measurement(
        _benchmark_fixture(
            evaluator_time=None,
            timing_status="below_timer_resolution",
            compiled_direct_arena_active=True,
        ),
        matrix_element=2.0,
    )

    assert measurement["execution_seconds_per_point"] is None
    assert measurement["execution_timing"] == {
        "abi": "pyamplicol-report-execution-timing-v1",
        "status": "below_timer_resolution",
        "ratio_eligible": False,
        "raw_seconds_per_point": 0.0,
        "source": ("runtime_profile_core_compiled_direct_arena_orchestration_time"),
        "compiled_direct_arena_active": True,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
    }


def test_benchmark_measurement_rejects_unauthenticated_zero_execution() -> None:
    with pytest.raises(RunnerError, match="authenticated compiled Direct-Arena zero"):
        _benchmark_measurement(
            _benchmark_fixture(
                evaluator_time=None,
                timing_status="below_timer_resolution",
                compiled_direct_arena_active=False,
            ),
            matrix_element=2.0,
        )


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _binding_digest(binding: dict[str, object]) -> str:
    if binding["kind"] != "rusticol-intrinsic":
        payload = dict(binding)
        payload.pop("payload_digest", None)
        return _digest_json(payload)
    fields = (
        "abi",
        "contribution_parent_permutation",
        "kind",
        "runtime_template",
    )
    return _digest_json({field: binding.get(field) for field in fields})


def _refresh_recurrence_catalog(
    execution: dict[str, object],
    pack: dict[str, object],
    *,
    refresh_binding_digests: bool = True,
) -> None:
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        binding = template["payload_binding"]
        assert isinstance(binding, dict)
        if refresh_binding_digests:
            binding["payload_digest"] = _binding_digest(binding)
        semantic = dict(template)
        semantic.pop("semantic_digest", None)
        template["semantic_digest"] = _digest_json(semantic)
    semantic_catalog = dict(catalog)
    semantic_catalog.pop("catalog_digest", None)
    catalog["catalog_digest"] = _digest_json(semantic_catalog)
    execution["direct_template_catalog_digest"] = catalog["catalog_digest"]
    plan = execution["plan"]
    assert isinstance(plan, dict)
    plan["direct_template_catalog_digest"] = catalog["catalog_digest"]


def _recurrence_source_fixture(
    source_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source_path = "kernels/000000/application-0.symjit"
    prepared_binding: dict[str, object] = {
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
    intrinsic_binding: dict[str, object] = {
        "abi": "pyamplicol-recurrence-direct-payload-binding-v1",
        "contribution_parent_permutation": [0, 1],
        "destination_operation": None,
        "direct_application_abi": None,
        "exact_factor_scalar_slots": [],
        "input_plane_count": 0,
        "input_plane_projections": [],
        "intrinsic_contract_digest": None,
        "kind": "rusticol-intrinsic",
        "output_alias_inputs": [],
        "parameter_bindings": [],
        "payload_digest": "",
        "payload_paths": [],
        "prepared_kernel_id": None,
        "prepared_template_semantic_digest": None,
        "role": None,
        "runtime_template": "source-current",
        "scalar_input_count": 0,
        "scalar_projections": [],
        "source_application_abi": None,
        "source_application_path": None,
        "source_application_sha256": None,
        "state_plane_indices": [],
    }
    templates: list[dict[str, object]] = [
        {
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
            "payload_binding": prepared_binding,
            "portable": True,
            "role": "finalization",
            "semantic_digest": "",
            "semantic_template_ids": ["prepared-finalization"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "template-0",
        },
        {
            "abi": "pyamplicol-recurrence-direct-template-v1",
            "alignment_bytes": 64,
            "backend": "jit",
            "coupling_slot_count": 0,
            "destination_aliasing": False,
            "destination_component_count": 1,
            "destination_operation": "initialize",
            "direct_executor_id": 1,
            "evaluator_binding_id": 1,
            "evaluator_resolver_key": "source-current",
            "exact_expression_digest": "f" * 64,
            "momentum_operand_count": 0,
            "optimization_level": 2,
            "parameter_slot_count": 0,
            "parent_arity": 0,
            "parent_component_counts": [],
            "payload_binding": intrinsic_binding,
            "portable": True,
            "role": "source",
            "semantic_digest": "",
            "semantic_template_ids": ["source-current"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "template-1",
        },
    ]
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
        "templates": templates,
    }
    execution: dict[str, object] = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
        "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
        "prepared_kernel_pack_digest": prepared_digest,
        "direct_template_catalog_digest": "",
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "plan": {
            "direct_template_abi": "pyamplicol-recurrence-direct-template-v1",
            "direct_backend_abi": "rusticol.recurrence-direct-backend.v1",
            "prepared_kernel_pack_digest": prepared_digest,
            "direct_template_catalog_digest": "",
        },
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
    _refresh_recurrence_catalog(execution, pack)
    return execution, pack


def _write_json_payload(
    root: Path,
    relative: str,
    value: object,
    *,
    process_id: str | None,
) -> SimpleNamespace:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return SimpleNamespace(
        path=relative,
        role="evaluator-manifest",
        media_type="application/json",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        executable=False,
        process_id=process_id,
    )


def _runner_recurrence_artifact(
    root: Path,
) -> tuple[
    SimpleNamespace,
    dict[str, object],
    dict[str, object],
    SimpleNamespace,
    SimpleNamespace,
]:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_relative = f"processes/{process_id}/execution.json"
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": f"{process_id}/execution.json",
            }
        ],
    }
    index_payload = _write_json_payload(
        root,
        index_relative,
        index,
        process_id=None,
    )
    execution_payload = _write_json_payload(
        root,
        execution_relative,
        execution,
        process_id=process_id,
    )
    pack_payload = _write_json_payload(
        root,
        pack_relative,
        pack,
        process_id=None,
    )
    source_path = root / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_data)
    source_payload = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        size_bytes=len(source_data),
        sha256=source_sha256,
        executable=False,
        process_id=None,
    )
    manifest = SimpleNamespace(
        root=root,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(
            index_payload,
            execution_payload,
            pack_payload,
            source_payload,
        ),
        extensions={},
    )
    return manifest, execution, pack, execution_payload, pack_payload


def _rewrite_json_payload(
    root: Path,
    record: SimpleNamespace,
    value: object,
) -> None:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (root / record.path).write_bytes(data)
    record.size_bytes = len(data)
    record.sha256 = hashlib.sha256(data).hexdigest()


def test_checked_runtime_identity_rejects_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW is unavailable")
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises(RunnerError, match="checked fd"):
        _regular_file_identity(link)


def test_effective_config_is_read_from_its_authenticated_artifact_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative = "config/effective.toml"
    data = b'[evaluator]\nbackend = "jit"\nexecution_mode = "compiled"\n'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    manifest = SimpleNamespace(
        root=tmp_path,
        configuration={"effective_path": relative},
        payloads=(
            SimpleNamespace(
                path=relative,
                role="configuration-effective",
                media_type="application/toml",
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                executable=False,
                process_id=None,
            ),
        ),
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _path, *, verify_payloads: manifest,
    )

    assert _authenticated_effective_config(tmp_path) == {
        "evaluator": {
            "backend": "jit",
            "execution_mode": "compiled",
        }
    }

    path.write_bytes(data.replace(b"compiled", b"eager   "))
    with pytest.raises(RunnerError, match="artifact manifest"):
        _authenticated_effective_config(tmp_path)


@dataclass(frozen=True)
class Flow:
    id: str
    word: tuple[int, ...]


@dataclass(frozen=True)
class Helicity:
    id: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class Particle:
    label: int


class Resolved:
    def __init__(self, values: object, totals: tuple[complex, ...]) -> None:
        self.values = values
        self._totals = totals

    def total(self) -> tuple[complex, ...]:
        return self._totals


class FakeRuntime:
    def __init__(self) -> None:
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            external_particles=(Particle(1), Particle(2), Particle(3)),
            helicities=(
                Helicity("h:-1,-1,-1", (-1, -1, -1)),
                Helicity("h:-1,+1,-1", (-1, 1, -1)),
            ),
            color_flows=(
                Flow("flow:2,1,3", (2, 1, 3)),
                Flow("flow:1,2,3", (1, 2, 3)),
            ),
        )
        self.optimized = (3.0 + 0.0j,)
        self.resolved_total = (3.0 + 0.0j,)

    def evaluate(self, _points: object, **_selectors: object) -> tuple[complex, ...]:
        return self.optimized

    def evaluate_resolved(
        self,
        _points: object,
        **_selectors: object,
    ) -> Resolved:
        return Resolved(
            (
                (
                    (0.0 + 0.0j,),
                    (3.0 + 0.0j,),
                ),
            ),
            self.resolved_total,
        )


def _cell(
    mode: ExecutionMode,
    accuracy: Accuracy,
    workload: Workload,
):
    return next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is mode
        and cell.measurement.accuracy is accuracy
        and cell.workload is workload
    )


@pytest.mark.parametrize(
    ("mode", "workload", "expected_layout", "expected_level"),
    (
        (
            ExecutionMode.RECURRENCE,
            Workload.SELECTED_FLOW,
            "topology-replay",
            2,
        ),
        (
            ExecutionMode.RECURRENCE,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
        (
            ExecutionMode.COMPILED,
            Workload.SELECTED_FLOW,
            "topology-replay",
            3,
        ),
        (
            ExecutionMode.EAGER,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
    ),
)
def test_config_steers_complete_coverage_and_layout_only(
    mode: ExecutionMode,
    workload: Workload,
    expected_layout: str,
    expected_level: int,
) -> None:
    cell = _cell(mode, Accuracy.LC, workload)
    values = config_values(
        cell,
        RunnerSettings(worker_cores=1),
        repo_root=Path("/repo"),
    )

    assert values["color"]["lc_flow_layout"] == expected_layout  # type: ignore[index]
    assert values["evaluator"]["execution_mode"] == mode.value  # type: ignore[index]
    assert (
        values["evaluator"]["jit"]["optimization_level"]  # type: ignore[index]
        == expected_level
    )
    serialized = repr(values)
    assert "selected_color_sector_ids" not in serialized
    assert "selected_source_helicities" not in serialized
    assert "reference_color_order" not in serialized


def test_nlc_and_full_use_contracted_topology_replay_configuration() -> None:
    for accuracy in (Accuracy.NLC, Accuracy.FULL):
        cell = _cell(ExecutionMode.RECURRENCE, accuracy, Workload.CONTRACTED)
        values = config_values(
            cell,
            RunnerSettings(),
            repo_root=Path("/repo"),
        )
        assert values["color"] == {
            "accuracy": accuracy.value,
            "lc_flow_layout": "topology-replay",
        }


def test_selector_contract_uses_first_flow_and_first_nonzero_helicity() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)

    contract = derive_selector_contract(runtime, points)

    assert contract.selected_color_flow_ids == ("flow:2,1,3",)
    assert contract.selected_color_words == ((2, 1, 3),)
    assert contract.all_flow_helicity_ids == ("h:-1,+1,-1",)
    assert contract.all_flow_source_helicities == ((1, -1), (2, 1), (3, -1))
    assert contract.point_digest == point_digest(points)
    assert SelectorContract.from_mapping(contract.as_dict()) == contract
    validate_selector_contract(runtime, contract, points)


def test_selector_contract_rejects_changed_point_or_axis() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)

    with pytest.raises(RunnerError, match="measurement point differ"):
        validate_selector_contract(
            runtime,
            contract,
            (((2.0, 0.0, 0.0, 2.0),),),
        )

    runtime.physics.color_flows = (Flow("different", (2, 1, 3)),)
    with pytest.raises(RunnerError, match="selected physical flow"):
        validate_selector_contract(runtime, contract, points)


def test_runtime_contract_requires_both_lc_selector_axes() -> None:
    runtime = FakeRuntime()
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    validate_runtime_contract(cell, runtime)

    runtime.physics.selector_capabilities = ("helicity",)
    with pytest.raises(RunnerError, match="color_flow"):
        validate_runtime_contract(cell, runtime)


def test_artifact_contract_rejects_generation_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    process = SimpleNamespace(
        execution_mode="recurrence",
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_flow_layout="topology-replay",
    )
    inspection = SimpleNamespace(processes=(process,))
    inspection.runtime_capabilities = (
        "rusticol.recurrence-direct-arena.complex-f64.v1",
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact",
        lambda _path: inspection,
    )
    validate_artifact_contract(cell, Path("/artifact"))

    process.generation_specialized_axes = ("color_flow",)
    with pytest.raises(RunnerError, match="complete runtime coverage"):
        validate_artifact_contract(cell, Path("/artifact"))


@pytest.mark.parametrize(
    ("cell", "capability", "expected_evaluator_abi", "optimization_identity"),
    (
        (
            _cell(
                ExecutionMode.RECURRENCE,
                Accuracy.LC,
                Workload.SELECTED_FLOW,
            ),
            "rusticol.recurrence-direct-arena.complex-f64.v1",
            "pyamplicol-recurrence-runtime-layout-v2",
            {"source_jit_optimization_level": 2},
        ),
        (
            _cell(
                ExecutionMode.EAGER,
                Accuracy.LC,
                Workload.SELECTED_FLOW,
            ),
            "eager-direct-arena-v1",
            "symjit-direct-table-binding-v2",
            {"source_jit_optimization_level": 2},
        ),
        (
            next(
                cell
                for cell in REPORT_CATALOG.z_cells()
                if cell.variant == "jit_o1" and cell.workload is Workload.SELECTED_FLOW
            ),
            "compiled-plane-arena-v1",
            "symjit-direct-application-storage-v3",
            {
                "source_jit_optimization_level": 1,
                "direct_codegen_optimization_level": 3,
            },
        ),
        (
            next(
                cell
                for cell in REPORT_CATALOG.z_cells()
                if cell.variant == "cpp_o3" and cell.workload is Workload.SELECTED_FLOW
            ),
            "compiled-plane-arena-v1",
            "pyamplicol-native-compiled-direct-application-v1",
            {},
        ),
    ),
)
def test_runtime_identity_binds_native_artifact_and_arena(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cell: CellSpec,
    capability: str,
    expected_evaluator_abi: str,
    optimization_identity: dict[str, int],
) -> None:
    import pyamplicol

    artifact_id = "a" * 64
    manifest = SimpleNamespace(
        artifact_id=artifact_id,
        processes=(
            {
                "id": "process-1",
                "required_runtime_capabilities": [capability],
            },
        ),
        runtime={"engine_version": "0.1.0.test"},
    )
    source_revision = "d" * 40
    package_root = tmp_path / "pyamplicol"
    package_root.mkdir()
    package_init = package_root / "__init__.py"
    package_init.write_bytes(b"")
    (package_root / "api.py").write_bytes(b"VALUE=1\n")
    (package_root / "ignored.pyo").write_bytes(b"ignored")
    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    native_path = package_root / "_rusticol.so"
    native_path.write_bytes(b"native")
    native = SimpleNamespace(
        __file__=str(native_path),
        package_version=lambda: pyamplicol.__version__,
        native_build_inputs_sha256=lambda: "b" * 64,
        target_info=lambda: SimpleNamespace(
            triple="aarch64-apple-darwin",
            cpu_features=("neon",),
        ),
    )
    package_records = [
        {
            "root_index": 0,
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (package_init, native_path, package_root / "api.py")
    ]
    bytecode_policy = {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }
    package_tree_identity = {
        "kind": "pyamplicol-python-package-tree-v2",
        "root": str(package_root.resolve()),
        "roots": [str(package_root.resolve())],
        "file_count": 3,
        "total_bytes": sum(record["size"] for record in package_records),
        "sha256": hashlib.sha256(
            json.dumps(
                package_records,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "member_set_stable": True,
        "namespace_bound_to_root_fd": True,
        "bytecode_policy": bytecode_policy,
    }
    native_identity = {
        "path": str(native_path.resolve()),
        "size": native_path.stat().st_size,
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    loaded_origin_policy = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": 1,
        "observations": [
            {
                "module": "pyamplicol",
                "kind": "package-member",
                "root_index": 0,
                "path": "__init__.py",
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
    }
    loaded_origin_policy["observations_sha256"] = _digest_json(
        loaded_origin_policy["observations"]
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda _path, *, verify_payloads: manifest,
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.importlib.import_module",
        lambda name: native if name == "pyamplicol._rusticol" else None,
    )
    monkeypatch.setattr(
        "pyamplicol._internal.versions._active_build_info",
        lambda: {
            "schema_version": 1,
            "version": pyamplicol.__version__,
            "candidate_fingerprint": "candidate",
            "source_revision": source_revision,
            "source_checkout": "/repo",
            "native_build_inputs_sha256": "b" * 64,
            "publishable": False,
        },
    )
    monkeypatch.setattr(pyamplicol, "__file__", str(package_init))
    monkeypatch.setattr(pyamplicol, "__path__", [str(package_root)])
    monkeypatch.setattr(
        "tools.performance_report.runner.established_preimport_runtime_identity",
        lambda: {
            "kind": "pyamplicol-preimport-runtime-identity-v1",
            "python_package_tree": package_tree_identity,
            "native_extension": native_identity,
        },
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.python_package_tree_identity",
        lambda roots: (
            package_tree_identity
            if tuple(roots) == (package_root,)
            else pytest.fail(f"unexpected package roots: {roots}")
        ),
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.loaded_pyamplicol_origin_policy",
        lambda roots, **kwargs: (
            loaded_origin_policy
            if (
                tuple(roots) == (package_root,)
                and kwargs["native_extension"] == native_path
                and kwargs["expected_package_identity"] == package_tree_identity
                and kwargs["expected_native_identity"] == native_identity
            )
            else pytest.fail("unexpected loaded-origin evidence request")
        ),
    )

    def direct_identity(
        _manifest: object,
        *,
        process_id: str,
        source_optimization_level: int,
    ) -> dict[str, object]:
        assert process_id == "process-1"
        return {
            "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
            "optimization_level": 3,
            "source_optimization_level": source_optimization_level,
            "leaf_count": 2,
            "execution_manifest_path": "execution.json",
            "execution_manifest_sha256": "e" * 64,
        }

    monkeypatch.setattr(
        "tools.performance_report.runner._authenticated_direct_codegen_identity",
        direct_identity,
    )

    recurrence_identity = {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 5,
        "prepared_direct_template_count": 3,
        "source_evaluator_leaf_count": 4,
        "source_application_abi": "symjit-application-storage-v3",
        "direct_application_abi": "symjit-direct-application-storage-v1",
        "prepared_kernel_pack_digest": "7" * 64,
        "direct_template_catalog_digest": "8" * 64,
        "execution_manifest_path": "processes/process-1/execution.json",
        "execution_manifest_sha256": "f" * 64,
        "kernel_pack_path": "model/eager-kernel-pack.json",
        "kernel_pack_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        "tools.performance_report.runner._authenticated_recurrence_source_identity",
        lambda _manifest, *, process_id, source_optimization_level: (
            recurrence_identity
            if process_id == "process-1" and source_optimization_level == 2
            else pytest.fail("unexpected recurrence identity request")
        ),
    )

    identity = runtime_identity_payload(
        cell,
        SimpleNamespace(
            artifact_id=artifact_id,
            execution_mode=cell.measurement.execution_mode.value,
        ),
        Path("/artifact"),
        "process-1",
        expected_source_revision=source_revision,
    )

    assert identity["artifact_id"] == artifact_id
    assert identity["loaded_artifact_id"] == artifact_id
    assert identity["loaded_execution_mode"] == cell.measurement.execution_mode.value
    assert identity["required_arena_capability"] == capability
    assert identity["expected_evaluator_abi"] == expected_evaluator_abi
    expected_source_abi = (
        "pyamplicol-native-compiled-direct-application-v1"
        if (
            cell.measurement.execution_mode is ExecutionMode.COMPILED
            and cell.measurement.backend != "jit"
        )
        else {
            "jit": "symjit-application-storage-v3",
            "cpp": "symbolica.compiled-cpp.complex-f64.v1",
            "asm": "symbolica.compiled-asm.complex-f64.v1",
        }[cell.measurement.backend]
    )
    assert identity["expected_source_evaluator_abi"] == expected_source_abi
    assert (
        identity["expected_source_evaluator_runtime_capability"]
        == {
            "jit": "symjit.application.complex-f64.v1",
            "cpp": "symbolica.compiled-cpp.complex-f64.v1",
            "asm": "symbolica.compiled-asm.complex-f64.v1",
        }[cell.measurement.backend]
    )
    assert identity["native_build_inputs_sha256"] == "b" * 64
    package_tree = identity["python_package_tree"]
    assert package_tree == package_tree_identity
    assert identity["loaded_module_origin_policy"] == loaded_origin_policy
    for field in (
        "source_jit_optimization_level",
        "direct_codegen_optimization_level",
    ):
        if field in optimization_identity:
            assert identity[field] == optimization_identity[field]
        else:
            assert field not in identity
    if cell.measurement.execution_mode is ExecutionMode.COMPILED and (
        cell.measurement.backend == "jit"
    ):
        assert identity["direct_codegen_identity"] == {
            "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
            "optimization_level": 3,
            "source_optimization_level": cell.measurement.jit_optimization_level,
            "leaf_count": 2,
            "execution_manifest_path": "execution.json",
            "execution_manifest_sha256": "e" * 64,
        }
    else:
        assert "direct_codegen_identity" not in identity
    if cell.measurement.execution_mode is ExecutionMode.RECURRENCE:
        assert identity["source_jit_identity"] == recurrence_identity
    else:
        assert "source_jit_identity" not in identity

    with pytest.raises(RunnerError, match="loaded artifact identity"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(
                artifact_id="c" * 64,
                execution_mode=cell.measurement.execution_mode.value,
            ),
            Path("/artifact"),
            "process-1",
            expected_source_revision=source_revision,
        )

    mismatched_mode = (
        "compiled"
        if cell.measurement.execution_mode is not ExecutionMode.COMPILED
        else "eager"
    )
    with pytest.raises(RunnerError, match="loaded execution mode"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(
                artifact_id=artifact_id,
                execution_mode=mismatched_mode,
            ),
            Path("/artifact"),
            "process-1",
            expected_source_revision=source_revision,
        )

    with pytest.raises(RunnerError, match=r"does not expose.*execution mode"):
        runtime_identity_payload(
            cell,
            SimpleNamespace(artifact_id=artifact_id),
            Path("/artifact"),
            "process-1",
            expected_source_revision=source_revision,
        )


def test_direct_codegen_identity_follows_authenticated_process_index(
    tmp_path: Path,
) -> None:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_entry_relative = f"{process_id}/execution.json"
    execution_relative = f"processes/{execution_entry_relative}"
    execution = {
        "kind": "pyamplicol-runtime-execution",
        "compiled": {
            "stage_evaluators": {
                "amplitude_stage": {
                    "compiled_plane_arena": {
                        "leaves": [
                            {
                                "optimization_level": 1,
                                "direct_codegen_optimization_level": 3,
                            }
                        ]
                    }
                }
            }
        },
    }
    evaluator_index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": execution_entry_relative,
                "required_runtime_capabilities": ["compiled-plane-arena-v1"],
            }
        ],
    }

    def payload(
        relative: str,
        value: dict[str, object],
        *,
        payload_process_id: str | None,
    ) -> SimpleNamespace:
        data = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return SimpleNamespace(
            path=relative,
            role="evaluator-manifest",
            media_type="application/json",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            executable=False,
            process_id=payload_process_id,
        )

    index_payload = payload(
        index_relative,
        evaluator_index,
        payload_process_id=None,
    )
    execution_payload = payload(
        execution_relative,
        execution,
        payload_process_id=process_id,
    )
    manifest = SimpleNamespace(
        root=tmp_path,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(index_payload, execution_payload),
    )

    identity = _authenticated_direct_codegen_identity(
        manifest,
        process_id=process_id,
        source_optimization_level=1,
    )

    assert identity == {
        "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
        "optimization_level": 3,
        "source_optimization_level": 1,
        "leaf_count": 1,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_payload.sha256,
    }


def test_recurrence_source_identity_follows_authenticated_direct_template_pack(
    tmp_path: Path,
) -> None:
    process_id = "process-1"
    index_relative = "processes/evaluators.json"
    execution_relative = f"processes/{process_id}/execution.json"
    pack_relative = "model/eager-kernel-pack.json"
    source_relative = "model/eager-kernels/kernels/000000/application-0.symjit"
    source_data = b"authenticated-symjit-application"
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    execution, pack = _recurrence_source_fixture(source_sha256)
    evaluator_index = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-set",
        "processes": [
            {
                "process_id": process_id,
                "manifest_path": f"{process_id}/execution.json",
            }
        ],
    }

    def payload(
        relative: str,
        value: dict[str, object],
        *,
        payload_process_id: str | None,
    ) -> SimpleNamespace:
        data = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return SimpleNamespace(
            path=relative,
            role="evaluator-manifest",
            media_type="application/json",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            executable=False,
            process_id=payload_process_id,
        )

    index_payload = payload(
        index_relative,
        evaluator_index,
        payload_process_id=None,
    )
    execution_payload = payload(
        execution_relative,
        execution,
        payload_process_id=process_id,
    )
    pack_payload = payload(
        pack_relative,
        pack,
        payload_process_id=None,
    )
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_data)
    source_payload = SimpleNamespace(
        path=source_relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        size_bytes=len(source_data),
        sha256=source_sha256,
        executable=False,
        process_id=None,
    )
    manifest = SimpleNamespace(
        root=tmp_path,
        runtime={"evaluator_manifest_path": index_relative},
        payloads=(index_payload, execution_payload, pack_payload, source_payload),
        extensions={},
    )

    identity = _authenticated_recurrence_source_identity(
        manifest,
        process_id=process_id,
        source_optimization_level=2,
    )

    assert identity == {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": 2,
        "direct_template_count": 2,
        "prepared_direct_template_count": 1,
        "source_evaluator_leaf_count": 1,
        "source_application_abi": "symjit-application-storage-v3",
        "direct_application_abi": "symjit-direct-application-storage-v1",
        "prepared_kernel_pack_digest": "a" * 64,
        "direct_template_catalog_digest": execution["direct_template_catalog_digest"],
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_payload.sha256,
        "kernel_pack_path": pack_relative,
        "kernel_pack_sha256": pack_payload.sha256,
    }

    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    template = templates[1]
    assert isinstance(template, dict)
    template["optimization_level"] = 1
    _refresh_recurrence_catalog(execution, pack)
    replacement = json.dumps(
        pack,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / pack_relative).write_bytes(replacement)
    pack_payload.size_bytes = len(replacement)
    pack_payload.sha256 = hashlib.sha256(replacement).hexdigest()
    execution_replacement = json.dumps(
        execution,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / execution_relative).write_bytes(execution_replacement)
    execution_payload.size_bytes = len(execution_replacement)
    execution_payload.sha256 = hashlib.sha256(execution_replacement).hexdigest()
    with pytest.raises(RunnerError, match="template 1 contract"):
        _authenticated_recurrence_source_identity(
            manifest,
            process_id=process_id,
            source_optimization_level=2,
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
        ("pack_prepared_digest", "kernel-pack digests"),
        ("execution_catalog_link", "kernel-pack digests"),
        ("catalog_backend_abi", "optimization identity"),
        ("catalog_canonicalization_abi", "optimization identity"),
        ("catalog_digest", "catalog digest"),
        ("template_portable", "template 0 contract"),
        ("template_abi", "template 0 contract"),
        ("binding_abi", "payload-binding contract"),
        ("source_application_abi", "source application"),
        ("direct_application_abi", "source application"),
        ("payload_digest", "payload-binding contract"),
        ("intrinsic_source_application_abi", "intrinsic source contract"),
        ("intrinsic_direct_application_abi", "intrinsic source contract"),
        ("source_leaf_runtime_capability", "source evaluator leaf contract"),
        ("source_leaf_application_abi", "source evaluator leaf contract"),
        ("source_leaf_optimization", "source evaluator leaf contract"),
        ("source_leaf_path", "payload is missing"),
        ("source_payload_digest_link", "not bound to its prepared kernel"),
    ),
)
def test_recurrence_source_identity_rejects_every_broken_link(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    (
        manifest,
        execution,
        pack,
        execution_payload,
        pack_payload,
    ) = _runner_recurrence_artifact(tmp_path)
    plan = execution["plan"]
    assert isinstance(plan, dict)
    catalog = pack["recurrence_direct_template"]
    assert isinstance(catalog, dict)
    templates = catalog["templates"]
    assert isinstance(templates, list)
    prepared = templates[0]
    intrinsic = templates[1]
    assert isinstance(prepared, dict)
    assert isinstance(intrinsic, dict)
    prepared_binding = prepared["payload_binding"]
    intrinsic_binding = intrinsic["payload_binding"]
    assert isinstance(prepared_binding, dict)
    assert isinstance(intrinsic_binding, dict)
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
        prepared["portable"] = False
        refresh_catalog = True
    elif corruption == "template_abi":
        prepared["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "binding_abi":
        prepared_binding["abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "source_application_abi":
        prepared_binding["source_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "direct_application_abi":
        prepared_binding["direct_application_abi"] = "wrong"
        refresh_catalog = True
    elif corruption == "payload_digest":
        prepared_binding["payload_digest"] = "0" * 64
        refresh_catalog = False
    elif corruption == "intrinsic_source_application_abi":
        intrinsic_binding["source_application_abi"] = "symjit-application-storage-v3"
        refresh_catalog = True
    elif corruption == "intrinsic_direct_application_abi":
        intrinsic_binding["direct_application_abi"] = (
            "symjit-direct-application-storage-v1"
        )
        refresh_catalog = True
    elif corruption == "source_leaf_runtime_capability":
        source_leaf["runtime_capability"] = "wrong"
    elif corruption == "source_leaf_application_abi":
        source_leaf["application_abi"] = "wrong"
    elif corruption == "source_leaf_optimization":
        source_leaf["optimization_level"] = 1
    elif corruption == "source_leaf_path":
        source_leaf["application_path"] = "kernels/000000/missing.symjit"
    elif corruption == "source_payload_digest_link":
        prepared_binding["source_application_sha256"] = "0" * 64
        refresh_catalog = True
    else:
        pytest.fail(f"unknown corruption {corruption}")

    if refresh_catalog is not None:
        _refresh_recurrence_catalog(
            execution,
            pack,
            refresh_binding_digests=refresh_catalog,
        )
    _rewrite_json_payload(tmp_path, execution_payload, execution)
    _rewrite_json_payload(tmp_path, pack_payload, pack)
    with pytest.raises(RunnerError, match=message):
        _authenticated_recurrence_source_identity(
            manifest,
            process_id="process-1",
            source_optimization_level=2,
        )


def test_resolved_sum_validation_and_pointwise_tolerances() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )

    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "ok"
    )
    runtime.resolved_total = (2.0 + 0.0j,)
    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "validation_failed"
    )

    assert pointwise_validation(1.0 + 1.0e-13, 1.0)["status"] == "ok"
    assert pointwise_validation(2.0, 1.0)["status"] == "validation_failed"


def test_matrix_element_conversion_rejects_sign_and_complex_drift() -> None:
    assert _real_nonnegative(2.0 + 0.0j) == 2.0
    assert _real_nonnegative(-1.0e-16 + 0.0j) == 0.0
    with pytest.raises(RunnerError, match="materially negative"):
        _real_nonnegative(-1.0e-3 + 0.0j)
    with pytest.raises(RunnerError, match="imaginary part"):
        _real_nonnegative(1.0 + 1.0e-3j)
