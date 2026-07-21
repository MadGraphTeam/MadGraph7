# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.generation.service as service_module
from pyamplicol._internal.versions import (
    COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
    COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
    COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EVALUATOR_RUNTIME_CAPABILITIES,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.api import ModelSource, ProcessRequest
from pyamplicol.artifacts import load_manifest
from pyamplicol.config import GenerationConfig
from pyamplicol.generation.artifact_writer import (
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.evaluator_container import PacbinReader
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.service import _ProcessSelection
from pyamplicol.models import BuiltinSMModel, compile_model_source
from pyamplicol.models.builtin.process_ir import build_process_ir


def _evaluator_process(
    expression: str = "d d~ > z g",
    *,
    selection: _ProcessSelection | None = None,
) -> tuple[
    service_module.GenerationBackend,
    BuiltinSMModel,
    service_module._EvaluatorProcess,
]:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(),
        None,
        process_selection=selection,
    )
    process_ir = build_process_ir(expression, color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="dual_lane"),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )
    evaluator = backend._construct_evaluator(
        prepared,
        model,
        PhaseHandle("test", None, 1),
    )
    return backend, model, evaluator


def _symjit_stage_manifest(root: Path, *, label: str) -> dict[str, object]:
    evaluator_dir = root / "evaluators"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    application = evaluator_dir / f"{label}.symjit"
    state = evaluator_dir / f"{label}.evaluator.bin"
    application.write_bytes(f"application:{label}".encode())
    state.write_bytes(f"state:{label}".encode())
    evaluator = {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": 1,
        "output_len": 1,
        "application_path": application.relative_to(root).as_posix(),
        "application_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": state.relative_to(root).as_posix(),
        "evaluator_state_runtime_capability": (
            SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
        ),
    }
    amplitude_stage = {
        "stage_index": 0,
        "stage_kind": "amplitude",
        "subset_size": None,
        "evaluator_label": label,
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 1,
        "output_slots": [],
        "input_value_slot_ids": [],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [],
        "parameter_count": 1,
        "value_parameter_count": 0,
        "momentum_parameter_count": 1,
        "model_parameter_count": 0,
        "real_valued_inputs": [0],
        "expression_ready": True,
        "blockers": [],
        "evaluator": evaluator,
    }
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 1,
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    }


def _materialize_without_symbolica(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    expression: str = "d d~ > z g",
    selection: _ProcessSelection | None = None,
) -> service_module.CompiledProcessArtifact:
    backend, model, evaluator = _evaluator_process(
        expression,
        selection=selection,
    )
    calls: list[tuple[Path, int]] = []

    def compile_stages(
        stage_input: object,
        _runtime_schema: object,
        root: Path,
        **_kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        lane = "-".join(root.relative_to(tmp_path / "build").parts).replace(
            ".",
            "",
        )
        calls.append((root, len(stage_input.dag.currents)))  # type: ignore[attr-defined]
        return object(), _symjit_stage_manifest(root, label=lane)

    monkeypatch.setattr(
        service_module,
        "build_and_write_generic_stage_evaluator_artifacts",
        compile_stages,
    )
    monkeypatch.setattr(
        service_module,
        "write_model_parameter_evaluator_artifact",
        lambda *_args, **_kwargs: None,
    )
    artifact = backend._materialize_evaluator_unlocked(
        evaluator,
        model,
        tmp_path / "build",
        PhaseHandle("jit", None, None),
        PhaseHandle("process", None, None),
        backend="JIT",
    )

    expected_calls = [
        (tmp_path / "build" / "dual_lane", len(evaluator.compiled.dag.currents))
    ]
    if evaluator.compiled.helicity_sum_dag is not None:
        expected_calls.append(
            (
                tmp_path / "build" / ".helicity-sum" / "dual_lane",
                len(evaluator.compiled.helicity_sum_dag.currents),
            )
        )
    def add_selector_lane_call(
        lane: service_module._HelicitySelectorLane,
        root: Path,
    ) -> None:
        expected_calls.append((root, len(lane.dag.currents)))
        for child_index, child in enumerate(lane.child_lanes):
            add_selector_lane_call(
                child,
                root.parent / f"{root.name}-closure-{child_index}",
            )

    selector_root = (
        tmp_path / "build" / ".helicity-selector-union" / "dual_lane"
    )
    for lane_index, lane in enumerate(evaluator.helicity_selector_lanes):
        add_selector_lane_call(lane, selector_root / f"class-{lane_index}")
    selector_root = (
        ".helicity-sum-color-selector"
        if evaluator.compiled.helicity_sum_dag is not None
        else ".color-selector"
    )
    expected_calls.extend(
        (
            tmp_path
            / "build"
            / selector_root
            / "dual_lane"
            / f"sector-{lane.materialized_sector_id}",
            len(lane.dag.currents),
        )
        for lane in evaluator.color_selector_lanes
    )
    assert calls == expected_calls
    return artifact


def _payload_paths(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "application_path",
                "evaluator_state_path",
                "library_path",
            } and isinstance(item, str):
                result.add(item)
            else:
                result.update(_payload_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_payload_paths(item))
    return result


def test_generation_retains_selector_and_fused_helicity_sum_dags() -> None:
    _backend, _model, evaluator = _evaluator_process()
    prepared = evaluator.compiled
    assert prepared.helicity_sum_dag is not None
    assert prepared.helicity_sum_dag.helicity_recurrence is None
    assert prepared.helicity_sum_dag.helicity_materialization is None
    assert prepared.dag.helicity_materialization is not None
    assert (
        prepared.dag.helicity_materialization.strategy
        == "quotient"
    )
    assert prepared.dag.helicity_materialization.materialized_current_count == len(
        prepared.dag.currents
    )
    assert prepared.dag.helicity_materialization.materialized_root_count == len(
        prepared.dag.amplitude_roots
    )

    primary = evaluator.runtime_schema.to_mapping()
    assert evaluator.helicity_sum_runtime_schema is not None
    assert evaluator.helicity_sum_stage_input is not None
    assert "materialization" in primary["helicity_recurrence"]
    assert (
        primary["helicity_recurrence"]["materialization"]["strategy"]
        == "quotient"
    )
    assert "helicity_recurrence" not in (
        evaluator.helicity_sum_runtime_schema.to_mapping()
    )

    assert evaluator.helicity_selector_lanes
    materialization = prepared.dag.helicity_materialization
    assert materialization is not None
    nonzero_domains = {
        schedule.selector_domain_id
        for schedule in materialization.selector_schedules
        if not schedule.structural_zero
    }
    assert {
        domain_id
        for lane in evaluator.helicity_selector_lanes
        for domain_id in lane.selector_domain_ids
    } == nonzero_domains
    assert len(evaluator.helicity_selector_lanes) == len(
        {
            (schedule.active_current_ids, schedule.active_root_ids)
            for schedule in materialization.selector_schedules
            if not schedule.structural_zero
        }
    )
    for lane in evaluator.helicity_selector_lanes:
        selector_input = lane.stage_input
        assert selector_input.dag.currents == prepared.dag.currents
        assert selector_input.dag.interactions == prepared.dag.interactions
        assert selector_input.dag.amplitude_roots == prepared.dag.amplitude_roots
        assert selector_input.dag.helicity_recurrence is None
        assert selector_input.dag.helicity_materialization is None
        selector_schema = lane.runtime_schema.to_mapping()
        assert "helicity_recurrence" not in selector_schema
        assert all(
            int(stage["interaction_count"])
            < int(parent_stage["interaction_count"])
            for stage, parent_stage in zip(
                selector_schema["stages"],
                primary["stages"],
                strict=True,
            )
        )


def test_color_topology_lane_capability_is_publicly_supported() -> None:
    assert (
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY
        in EVALUATOR_RUNTIME_CAPABILITIES
    )


def test_helicity_selector_union_capability_is_publicly_supported() -> None:
    assert (
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY
        in EVALUATOR_RUNTIME_CAPABILITIES
    )


def test_complete_lc_union_dispatches_to_exact_helicity_closure_lanes() -> None:
    _backend, _model, evaluator = _evaluator_process("d d~ > z g g g")

    assert len(evaluator.helicity_selector_lanes) == 1
    union = evaluator.helicity_selector_lanes[0]
    assert union.schedule_mode == "nested-runtime"
    assert len(union.child_lanes) == 2
    assert {
        sum(
            int(stage["interaction_count"])
            for stage in child.runtime_schema.to_mapping()["stages"]
        )
        for child in union.child_lanes
    } == {97}
    assert all(
        child.schedule_mode == "parent-closure"
        and not child.child_lanes
        for child in union.child_lanes
    )


def test_nested_helicity_closures_are_written_with_owned_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g g",
    )
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    output = tmp_path / "artifact"
    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.built_in_sm(),
        compiled_model=compile_model_source("built-in-sm", use_cache=False),
        configuration=_GenerationConfigProvenance.from_config(
            GenerationConfig(emit_api_bundle=False)
        ),
        processes=(artifact,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    execution = json.loads(
        (output / "processes/dual_lane/execution.json").read_text(
            encoding="utf-8"
        )
    )
    outer = execution["helicity_selector_executions"][0]
    assert outer["schedule_mode"] == "nested-runtime"
    children = outer["execution"]["helicity_selector_executions"]
    assert len(children) == 2
    assert all(child["schedule_mode"] == "parent-closure" for child in children)

    referenced = _payload_paths(execution)
    nested_prefix = "helicity-selector-union/class-0/helicity-selector-union/"
    assert any(path.startswith(f"{nested_prefix}class-0/") for path in referenced)
    assert any(path.startswith(f"{nested_prefix}class-1/") for path in referenced)
    manifest = load_manifest(output)
    declared = {record.path for record in manifest.payloads}
    assert "evaluators.pacbin" in declared
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
    for relative in referenced:
        assert f"processes/dual_lane/{relative}" in members


def test_generation_specialized_color_artifact_has_no_topology_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g",
        selection=_ProcessSelection(
            selected_color_sector_ids=frozenset({0})
        ),
    )

    assert artifact.color_selector_executions == ()
    assert artifact.helicity_selector_executions
    if artifact.helicity_sum_execution is not None:
        assert artifact.helicity_sum_execution.color_selector_executions == ()
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert "color_selector_executions" not in manifest
    auxiliary = manifest.get("helicity_sum_execution")
    if isinstance(auxiliary, dict):
        assert "color_selector_executions" not in auxiliary
    assert COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY not in manifest[
        "required_runtime_capabilities"
    ]


def test_complete_lc_artifact_serializes_every_materialized_sector_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _backend, _model, evaluator = _evaluator_process("g g > g g")
    target_dag = (
        evaluator.compiled.helicity_sum_dag or evaluator.compiled.dag
    )
    expected_sector_ids = tuple(
        sorted(
            {
                int(root.color_sector_id)
                for root in target_dag.amplitude_roots
                if root.color_sector_id is not None
            }
        )
    )
    assert expected_sector_ids
    assert tuple(
        lane.materialized_sector_id for lane in evaluator.color_selector_lanes
    ) == expected_sector_ids

    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="g g > g g",
    )
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    parent = manifest.get("helicity_sum_execution", manifest)
    records = parent["color_selector_executions"]
    assert tuple(
        record["materialized_sector_id"] for record in records
    ) == expected_sector_ids
    assert all(
        "color_selector_executions" not in record["execution"]
        and "helicity_sum_execution" not in record["execution"]
        and "physics_reduction" in record["execution"]
        for record in records
    )


def test_fixed_helicity_complete_lc_attaches_color_lanes_to_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g",
        selection=_ProcessSelection(
            selected_source_helicities={
                1: -1,
                2: 1,
                3: -1,
                4: 1,
                5: -1,
            }
        ),
    )

    assert artifact.helicity_sum_execution is None
    assert artifact.helicity_selector_executions == ()
    assert artifact.color_selector_executions
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert "helicity_sum_execution" not in manifest
    assert manifest["color_selector_executions"]
    assert COMPILED_RUNTIME_SELECTORS_CAPABILITY in manifest[
        "required_runtime_capabilities"
    ]


def test_eager_process_artifacts_have_no_compiled_selector_union_lane() -> None:
    assert "helicity_selector_executions" not in (
        service_module.EagerProcessArtifact.__dataclass_fields__
    )


def test_replayed_physical_flows_compile_only_materialized_sector_lanes() -> None:
    _backend, _model, evaluator = _evaluator_process("d d~ > z g g")
    target_dag = (
        evaluator.compiled.helicity_sum_dag or evaluator.compiled.dag
    )
    replay = target_dag.lc_topology_replay

    assert replay is not None
    assert len(replay.physical_sector_ids) > len(replay.materialized_sector_ids)
    assert tuple(
        lane.materialized_sector_id for lane in evaluator.color_selector_lanes
    ) == replay.materialized_sector_ids


def test_color_selector_execution_lanes_cannot_nest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_sum_execution is not None
    record = artifact.helicity_sum_execution.color_selector_executions[0]
    recursive_record = replace(
        record,
        execution=replace(
            record.execution,
            color_selector_executions=(record,),
        ),
    )
    invalid = replace(
        artifact,
        helicity_sum_execution=replace(
            artifact.helicity_sum_execution,
            color_selector_executions=(recursive_record,),
        ),
    )

    with pytest.raises(ValueError, match="cannot nest"):
        artifact_writer._execution_manifest(
            invalid,
            invalid.runtime_schema.to_mapping(),
        )


def test_compiled_materialization_builds_primary_sum_and_selector_union_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_sum_execution is not None
    assert artifact.helicity_selector_executions
    assert artifact.dag_summary["current_count"] == (
        artifact.runtime_schema.to_mapping()["helicity_recurrence"][
            "materialization"
        ]["materialized_current_count"]
    )
    assert "helicity_recurrence" not in (
        artifact.helicity_sum_execution.runtime_schema.to_mapping()
    )
    primary_schema = artifact.runtime_schema.to_mapping()
    for record in artifact.helicity_selector_executions:
        selector = record.execution
        selector_schema = selector.runtime_schema.to_mapping()
        assert "helicity_recurrence" not in selector_schema
        assert selector_schema["physics"]["reduction"] == (
            primary_schema["physics"]["reduction"]
        )
        assert selector.dag_summary["current_count"] == len(
            selector_schema["current_storage"]["current_slots"]
        )
        assert selector.dag_summary["amplitude_root_count"] == (
            selector_schema["amplitude_stage"]["output_count"]
        )
        assert selector.dag_summary["interaction_count"] < (
            artifact.dag_summary["interaction_count"]
        )
        assert COMPILED_RUNTIME_SELECTORS_CAPABILITY not in (
            selector.stage_manifest["required_runtime_capabilities"]
        )
    selector_records = artifact_writer._execution_manifest(
        artifact,
        primary_schema,
    )["helicity_selector_executions"]
    assert len(selector_records) == len(artifact.helicity_selector_executions)
    for selector_record in selector_records:
        assert selector_record["selector_domain_ids"]
        selector_manifest = selector_record["execution"]
        assert selector_manifest["kind"] == "pyamplicol-runtime-execution"
        assert selector_manifest["physics_reduction"] == (
            primary_schema["physics"]["reduction"]
        )
        assert "helicity_sum_execution" not in selector_manifest
        assert "helicity_selector_executions" not in selector_manifest
        assert "color_selector_executions" not in selector_manifest


def test_helicity_selector_union_execution_cannot_nest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_selector_executions
    assert artifact.helicity_sum_execution is not None
    nested_color_lane = artifact.helicity_sum_execution.color_selector_executions[0]
    first = artifact.helicity_selector_executions[0]
    invalid_first = replace(
        first,
        execution=replace(
            first.execution,
            color_selector_executions=(nested_color_lane,),
        ),
    )
    invalid = replace(
        artifact,
        helicity_selector_executions=(
            invalid_first,
            *artifact.helicity_selector_executions[1:],
        ),
    )

    with pytest.raises(ValueError, match="cannot nest"):
        artifact_writer._execution_manifest(
            invalid,
            invalid.runtime_schema.to_mapping(),
        )


def test_compiled_process_capabilities_use_primary_execution_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact_writer._compiled_process_runtime_capabilities(artifact) == (
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    )


def test_writer_emits_selector_and_fused_sum_lanes_and_owns_all_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    primary_schema = artifact.runtime_schema.to_mapping()
    execution_manifest = artifact_writer._execution_manifest(
        artifact,
        primary_schema,
    )
    assert "helicity_sum_execution" in execution_manifest
    assert (
        execution_manifest["runtime_schema"]["helicity_recurrence"][
            "materialization"
        ]["strategy"]
        == "quotient"
    )

    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    output = tmp_path / "artifact"
    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.built_in_sm(),
        compiled_model=compile_model_source("built-in-sm", use_cache=False),
        configuration=_GenerationConfigProvenance.from_config(
            GenerationConfig(emit_api_bundle=False)
        ),
        processes=(artifact,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    execution = json.loads(
        (output / "processes/dual_lane/execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert "helicity_sum_execution" in execution
    assert "helicity_selector_executions" in execution
    assert execution["runtime_schema"]["process_key"] == "dual_lane"
    assert (
        execution["runtime_schema"]["helicity_recurrence"]["materialization"][
            "strategy"
        ]
        == "quotient"
    )
    primary_evaluator = execution["compiled"]["stage_evaluators"][
        "amplitude_stage"
    ]["evaluator"]
    assert primary_evaluator["optimization_level"] == 3
    helicity_sum_evaluator = execution["helicity_sum_execution"]["compiled"][
        "stage_evaluators"
    ]["amplitude_stage"]["evaluator"]
    assert helicity_sum_evaluator["optimization_level"] == 3
    assert "helicity_recurrence" not in execution["helicity_sum_execution"][
        "runtime_schema"
    ]
    selector_records = execution["helicity_selector_executions"]
    assert selector_records
    for record in selector_records:
        selector_lane = record["execution"]
        assert record["selector_domain_ids"]
        assert "helicity_recurrence" not in selector_lane["runtime_schema"]
        assert selector_lane["physics_reduction"] == primary_schema["physics"][
            "reduction"
        ]
        assert COMPILED_RUNTIME_SELECTORS_CAPABILITY not in selector_lane[
            "required_runtime_capabilities"
        ]
    color_selector_records = execution["helicity_sum_execution"][
        "color_selector_executions"
    ]
    assert artifact.helicity_sum_execution is not None
    assert [
        record["materialized_sector_id"]
        for record in color_selector_records
    ] == [0]
    color_selector_execution = color_selector_records[0]["execution"]
    assert "helicity_sum_execution" not in color_selector_execution
    assert "color_selector_executions" not in color_selector_execution
    assert color_selector_execution["physics_reduction"] == (
        artifact.helicity_sum_execution.color_selector_executions[
            0
        ].execution.runtime_schema.to_mapping()["physics"]["reduction"]
    )

    manifest = load_manifest(output)
    declared = {record.path for record in manifest.payloads}
    referenced = _payload_paths(execution)
    assert referenced
    assert any(path.startswith("helicity-sum/") for path in referenced)
    assert any(
        path.startswith("helicity-selector-union/") for path in referenced
    )
    assert any(
        path.startswith("helicity-sum/color-selector/sector-0/")
        for path in referenced
    )
    assert "evaluators.pacbin" in declared
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
    for relative in referenced:
        assert f"processes/dual_lane/{relative}" in members
    assert execution["required_runtime_capabilities"] == [
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    ]
    generation = manifest.extensions["generation"]["concrete_processes"][0]
    assert generation["runtime_schema_sha256"] == artifact.runtime_schema.sha256
    assert artifact.helicity_sum_execution is not None
    assert generation["helicity_sum_runtime_schema_sha256"] == (
        artifact.helicity_sum_execution.runtime_schema.sha256
    )
    assert artifact.helicity_selector_executions
    assert generation["helicity_selector_runtime_schema_sha256s"] == [
        record.execution.runtime_schema.sha256
        for record in artifact.helicity_selector_executions
    ]
