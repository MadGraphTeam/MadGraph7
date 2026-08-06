# SPDX-License-Identifier: 0BSD
"""Gate-level contracts for the developer on-the-fly LC harness."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    write_pacbin_atomic,
)
from tools.developer import on_the_fly_lc_gate as gate


def _write_execution(root: Path, payload: object) -> None:
    process = root / "processes" / "fixture_process"
    process.mkdir(parents=True)
    (process / "execution.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_on_the_fly_contract_artifact(
    root: Path,
    *,
    execution_field: str | None = None,
    extra_seed_member: bool = False,
    extra_sidecar: str | None = None,
) -> str:
    process_prefix = f"processes/{gate.PROCESS_ID}"
    target = {"triple": "portable-64le", "cpu_features": []}
    capabilities = list(gate.ON_THE_FLY_CAPABILITIES)
    pdgs = (1, -1, 6, -6, 21, 21)
    execution: dict[str, object] = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-on-the-fly-execution",
        "required_runtime_capabilities": capabilities,
        "process": gate.PROCESS,
        "key": gate.PROCESS_ID,
        "color_accuracy": "lc",
        "external_pdg_order": list(pdgs),
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "runtime_options": {"point_tile_size": 128},
        "selector_policy": {
            "color_coverage": "complete",
            "reference_color_word": None,
            "trace_reflections_folded": True,
        },
        "runtime_metadata": {
            "runtime_parameters": [],
            "prepared_parameter_defaults": [],
            "parameter_projection": [],
            "external_legs": [],
            "particle_masses": [],
            "normalization": {},
        },
        "runtime_container": {
            "kind": "pyamplicol-on-the-fly-runtime-container",
            "schema_version": 1,
            "storage_abi": "pacbin-v1",
            "path": "on-the-fly-runtime.pacbin",
            "seed_member_path": gate.ON_THE_FLY_SEED_MEMBER_PATH,
        },
    }
    if execution_field is not None:
        execution[execution_field] = {"materialized": True}

    with ArtifactBuilder(root) as builder:
        builder.add_bytes(
            "config/requested.toml",
            b"[evaluator]\nexecution_mode = 'on-the-fly'\n",
            role="configuration-requested",
            media_type="application/toml",
        )
        builder.add_bytes(
            "config/effective.toml",
            b"[evaluator]\nexecution_mode = 'on-the-fly'\n",
            role="configuration-effective",
            media_type="application/toml",
        )
        builder.add_json(
            "model/compiled-model.json",
            {"schema_version": 1},
            role="compiled-model",
        )
        builder.add_json(
            "model/eager-kernel-pack.json",
            {"kind": "fixture-prepared-kernel-pack"},
            role="evaluator-manifest",
        )
        builder.add_json(
            "processes/evaluators.json",
            {
                "schema_version": 3,
                "kind": "pyamplicol-runtime-execution-set",
                "required_runtime_capabilities": capabilities,
                "processes": [
                    {
                        "process_id": gate.PROCESS_ID,
                        "manifest_path": f"{gate.PROCESS_ID}/execution.json",
                        "required_runtime_capabilities": capabilities,
                    }
                ],
            },
            role="evaluator-manifest",
        )
        builder.add_json(
            f"{process_prefix}/execution.json",
            execution,
            role="evaluator-manifest",
            process_id=gate.PROCESS_ID,
            compact=True,
        )
        builder.add_json(
            f"{process_prefix}/physics.json",
            {
                "schema_version": 1,
                "kind": "pyamplicol-on-the-fly-public-metadata",
                "process_id": gate.PROCESS_ID,
                "process": gate.PROCESS,
                "color_accuracy": "lc",
                "external_particles": [
                    {
                        "index": index,
                        "label": index + 1,
                        "particle": name,
                        "pdg": pdg,
                        "role": "initial" if index < 2 else "final",
                        "momentum_slot": index,
                        "momentum_components": ["E", "px", "py", "pz"],
                    }
                    for index, (name, pdg) in enumerate(
                        (
                            ("d", 1),
                            ("d~", -1),
                            ("t", 6),
                            ("t~", -6),
                            ("g", 21),
                            ("g", 21),
                        )
                    )
                ],
                "model_parameters": [],
            },
            role="runtime-physics",
            process_id=gate.PROCESS_ID,
            compact=True,
        )
        builder.add_json(
            f"{process_prefix}/validation-momenta.json",
            {"available": False},
            role="validation-momenta",
            process_id=gate.PROCESS_ID,
            compact=True,
        )
        runtime_path = builder.staged_path(
            f"{process_prefix}/on-the-fly-runtime.pacbin",
            create_parent=True,
        )
        members = [
            PacbinMemberSource(
                gate.ON_THE_FLY_SEED_MEMBER_PATH,
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                io.BytesIO(b"compact-process-seed"),
            )
        ]
        if extra_seed_member:
            members.append(
                PacbinMemberSource(
                    "on-the-fly/materialized-direct-plan-v1.bin",
                    PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                    io.BytesIO(b"materialized-sidecar"),
                )
            )
        write_pacbin_atomic(runtime_path, members)
        builder.register_staged_file(
            f"{process_prefix}/on-the-fly-runtime.pacbin",
            role="evaluator-state",
            media_type="application/octet-stream",
            executable=False,
            target=target,
            process_id=gate.PROCESS_ID,
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer={
                "distribution": "pyamplicol",
                "version": "0.1.0",
                "versions": {
                    "python_api": 1,
                    "toml": 1,
                    "compiled_model": 1,
                    "process_artifact": 3,
                    "runtime_physics": 1,
                    "symbolica_serialization": "fixture",
                    "c_abi": 1,
                },
                "target": target,
            },
            model={
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": "1" * 64,
                "compiled_schema_version": 1,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "config/requested.toml",
                "effective_path": "config/effective.toml",
                "adjustments": [],
            },
            processes=[
                {
                    "id": gate.PROCESS_ID,
                    "expression": gate.PROCESS,
                    "color_accuracy": "lc",
                    "external_pdgs": list(pdgs),
                    "physics_path": f"{process_prefix}/physics.json",
                    "required_runtime_capabilities": capabilities,
                    "aliases": [],
                }
            ],
            default_process_id=gate.PROCESS_ID,
            runtime={
                "engine": "rusticol",
                "engine_version": "0.1.0",
                "evaluator_manifest_path": "processes/evaluators.json",
                "api_bundle_path": None,
                "required_runtime_capabilities": capabilities,
            },
        )

    if extra_sidecar is not None:
        sidecar = root / process_prefix / extra_sidecar
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(b"forbidden materialized sidecar")
    return load_manifest(root, verify_payloads=True).artifact_id


def _compiled_record(
    *,
    sources: int,
    currents: int,
    components: int,
    attachments: int,
    evaluations: int,
    roots: int,
) -> dict[str, object]:
    return {
        "kind": "pyamplicol-runtime-execution",
        "dag_summary": {
            "source_count": sources,
            "current_count": currents,
            "interaction_count": attachments,
            "interaction_evaluation_count": evaluations,
            "amplitude_root_count": roots,
        },
        "runtime_schema": {"current_storage": {"component_count": components}},
    }


def _runtime(
    mode: str,
    *,
    flows: tuple[SimpleNamespace, ...] = (),
    helicities: tuple[SimpleNamespace, ...] = (),
    artifact_id: str = "a" * 64,
    process: str = gate.PROCESS,
) -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode=mode,
        artifact_id=artifact_id,
        physics=SimpleNamespace(
            process_id="fixture_process",
            process=process,
            color_accuracy="lc",
            color_flows=flows,
            helicities=helicities,
        ),
    )


def test_real_on_the_fly_candidate_and_private_probe_carrier_are_separate() -> None:
    candidate = gate._on_the_fly_config()
    assert candidate.color.accuracy == "lc"
    assert candidate.color.lc_flow_layout == "topology-replay"
    assert candidate.evaluator.execution_mode == "on-the-fly"
    assert candidate.evaluator.backend == "jit"
    assert candidate.evaluator.jit.optimization_level == 2
    assert candidate.generation.relation_discovery.mode == "off"
    assert candidate.generation.validation.enabled is True
    assert candidate.generation.validation.post_build_validation is True

    carrier = gate._recurrence_probe_config()
    assert carrier.evaluator.execution_mode == "recurrence"
    assert carrier.generation.validation.enabled is False
    assert carrier.generation.validation.post_build_validation is False

    flow = SimpleNamespace(id=gate.FLOW_ID, word=gate.FLOW_WORD, index=7)
    helicity = SimpleNamespace(
        id=gate.HELICITY_ID,
        values=gate.HELICITY_VALUES,
        structural_zero=False,
    )
    assert gate._selectors(
        SimpleNamespace(color_flows=(flow,), helicities=(helicity,))
    ) == (flow, helicity)
    assert gate._query(flow, helicity).flow_index == 7

    authority = gate._dense_authority(
        SimpleNamespace(artifact_id="a" * 64),
        SimpleNamespace(artifact_id="b" * 64),
        8,
    )
    assert authority["authority_kind"] == "validated_production_pyamplicol"
    assert authority["runtime_api"] == "Runtime.evaluate_resolved"
    assert authority["selected_flow_artifact_id"] == "a" * 64
    assert authority["all_flow_artifact_id"] == "b" * 64
    assert authority["certifies"] == (
        "selected_flow_helicity_sum",
        "all_flow_single_helicity",
    )
    assert gate._sum(((1.0, 2.0), (3.0, 4.0)), 2) == (4.0, 6.0)
    assert gate._series((0.0,), (0.0,), "zero")["worst"]["status"] == "ok"
    with pytest.raises(gate.GateError, match="disagrees"):
        gate._series((1.0e-300,), (0.0,), "no absolute floor")


def test_resolved_authority_checks_each_component_not_only_the_total() -> None:
    candidate = SimpleNamespace(
        helicity_ids=("h:first", "h:second"),
        color_ids=("flow:only",),
        values=(((2.0,), (0.0,)),),
    )
    reference = SimpleNamespace(
        helicity_ids=("h:first", "h:second"),
        color_ids=("flow:only",),
        values=(((1.0,), (1.0,)),),
    )
    assert sum(value[0] for value in candidate.values[0]) == sum(
        value[0] for value in reference.values[0]
    )
    with pytest.raises(gate.GateError, match="disagrees"):
        gate._resolved_component_checks(candidate, reference, "canceling components")


def test_real_candidate_contract_and_profile_never_open_dense_physics(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact_id = _write_on_the_fly_contract_artifact(artifact)

    class Backend:
        def profile(self, _points: object, **kwargs: object) -> dict[str, object]:
            assert kwargs == {
                "helicities": None,
                "color_flows": (gate.FLOW_ID,),
                "precision": 16,
                "include_values": True,
            }
            return {
                "execution_mode": "on-the-fly",
                "values": [1.0, 2.0],
                "wall_time_s": 0.01,
                **{
                    name: 2 if name.endswith("row_count") else 1
                    for name in gate.PUBLIC_PROFILE_WORK_FIELDS
                },
            }

    class Candidate:
        execution_mode = "on-the-fly"
        representative_process_key = gate.PROCESS_ID
        _backend = Backend()

        @property
        def physics(self) -> object:
            raise AssertionError("candidate dense physics was opened")

    candidate = Candidate()
    candidate.artifact_id = artifact_id
    contract = gate._on_the_fly_artifact_contract(artifact, candidate)
    assert contract["execution_mode"] == "on-the-fly"
    assert contract["dense_physics_accessed"] is False
    assert contract["required_runtime_capabilities"] == gate.ON_THE_FLY_CAPABILITIES
    assert contract["process_evaluator_state_payload_count"] == 1
    assert contract["runtime_container_authenticated"] is True
    assert contract["runtime_container_member_count"] == 1
    assert contract["seed_member_kind"] == "ON_THE_FLY_PROCESS_SEED"
    profile = gate._on_the_fly_public_profile(
        candidate,
        (((1.0,),), ((2.0,),)),
        (1.0, 2.0),
        flows=(gate.FLOW_ID,),
    )
    assert profile["value_check"]["checks"] == 2
    assert profile["work"]["recurrence_source_call_count"] == 1
    assert profile["work"]["recurrence_source_row_count"] == 2

    with pytest.raises(gate.GateError, match="disagrees"):
        gate._on_the_fly_public_profile(
            candidate,
            (((1.0,),), ((2.0,),)),
            (1.0, 3.0),
            flows=(gate.FLOW_ID,),
        )


@pytest.mark.parametrize(
    "sidecar",
    (
        "dag.json",
        "color-plan.json",
        "eager-runtime.pacbin",
        "direct-plan.bin",
        "recurrence-runtime.pacbin",
        "compiled.so",
        "structural-source-proof.json",
    ),
)
def test_on_the_fly_contract_rejects_materialized_process_sidecars(
    tmp_path: Path,
    sidecar: str,
) -> None:
    artifact = tmp_path / sidecar.replace(".", "-")
    artifact_id = _write_on_the_fly_contract_artifact(
        artifact,
        extra_sidecar=sidecar,
    )
    runtime = SimpleNamespace(
        execution_mode="on-the-fly",
        representative_process_key=gate.PROCESS_ID,
        artifact_id=artifact_id,
    )
    with pytest.raises(gate.GateError, match="materialized sidecars"):
        gate._on_the_fly_artifact_contract(artifact, runtime)


def test_on_the_fly_contract_rejects_extra_pacbin_member(tmp_path: Path) -> None:
    artifact = tmp_path / "extra-member"
    artifact_id = _write_on_the_fly_contract_artifact(
        artifact,
        extra_seed_member=True,
    )
    runtime = SimpleNamespace(
        execution_mode="on-the-fly",
        representative_process_key=gate.PROCESS_ID,
        artifact_id=artifact_id,
    )
    with pytest.raises(gate.GateError, match="exactly one process seed"):
        gate._on_the_fly_artifact_contract(artifact, runtime)


@pytest.mark.parametrize(
    "field",
    (
        "dag",
        "color_plan",
        "eager_plan",
        "direct_plan",
        "recurrence_runtime",
        "compiled_execution",
        "structural_proof",
    ),
)
def test_on_the_fly_contract_rejects_materialization_fields(
    tmp_path: Path,
    field: str,
) -> None:
    artifact = tmp_path / field
    artifact_id = _write_on_the_fly_contract_artifact(
        artifact,
        execution_field=field,
    )
    runtime = SimpleNamespace(
        execution_mode="on-the-fly",
        representative_process_key=gate.PROCESS_ID,
        artifact_id=artifact_id,
    )
    with pytest.raises(gate.GateError, match="forbidden materialization field"):
        gate._on_the_fly_artifact_contract(artifact, runtime)


def test_amplicol_anchor_requires_exact_cell_and_point_digest(tmp_path: Path) -> None:
    def write(name: str, cell_id: str) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "matrix_element": 3.0,
                    "execution_seconds_per_point": 4.0e-6,
                    "wall_seconds_per_point": 5.0e-6,
                    "selector_contract": {
                        "selected_color_flow_ids": [gate.FLOW_ID],
                        "selected_color_words": [list(gate.FLOW_WORD)],
                    },
                    "validation": {
                        "lc_common_component": {
                            "cell_id": cell_id,
                            "point_digest": "same-point",
                            "helicity_ids": [gate.HELICITY_ID],
                            "color_flow_ids": [gate.FLOW_ID],
                            "value": 2.0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    wrong_cell = gate._anchor(write("wrong-cell.json", "another-cell"))
    contextual = gate._anchor_checks(
        wrong_cell,
        "same-point",
        public_component=99.0,
        hidden_component=98.0,
        public_sum=97.0,
        hidden_sum=96.0,
    )
    assert contextual["comparison_performed"] is False
    assert "cell identity differs" in str(contextual["reason"])

    exact = gate._anchor(write("exact.json", gate.AMPICOL_CELL_ID))
    different_point = gate._anchor_checks(
        exact,
        "different-point",
        public_component=99.0,
        hidden_component=98.0,
        public_sum=97.0,
        hidden_sum=96.0,
    )
    assert different_point["comparison_performed"] is False
    assert "point digest differs" in str(different_point["reason"])

    compared = gate._anchor_checks(
        exact,
        "same-point",
        public_component=2.0,
        hidden_component=2.0,
        public_sum=3.0,
        hidden_sum=3.0,
    )
    assert compared["comparison_performed"] is True
    assert compared["execution_seconds_per_point"] == 4.0e-6
    assert compared["wall_seconds_per_point"] == 5.0e-6
    assert "independent clocks" in compared["clock_attribution"]["relationship"]


def test_hidden_timing_contract_counts_lookup_fill_execute_and_no_poison() -> None:
    def report(repetitions: int) -> dict[str, object]:
        benchmark = repetitions > 0
        cycles = gate.WARMUPS + repetitions if benchmark else 1
        elapsed = 0.25 if benchmark else None
        return {
            "process_id": gate.PROCESS_ID,
            "point_count": 2,
            "work_census_basis": gate.WORK_CENSUS_BASIS,
            "logical_current_count": 5,
            "resident_current_count": 5,
            "resident_current_component_count": 8,
            "source_operation_count": 2,
            "contribution_operation_count": 3,
            "finalization_operation_count": 1,
            "closure_operation_count": 1,
            "total_kernel_application_count": 7,
            "semantic_executor_binding_count": 4,
            "distinct_prepared_executor_count": 3,
            "trace_build_count": 1,
            "trace_cache_hit_count": cycles if benchmark else 0,
            "momentum_fill_count": cycles,
            "currents": [],
            "direct_plan_load_attempts": 0,
            "direct_plan_decode_attempts": 0,
            "direct_plan_materialization_attempts": 0,
            "established_builder_attempts": 0,
            "normalized_values": [1.0, 2.0],
            "benchmark_elapsed_seconds": elapsed,
            "benchmark_seconds_per_point": (
                None if elapsed is None else elapsed / (repetitions * 2)
            ),
        }

    assert gate._probe_values(report(0), 2) == (1.0, 2.0)
    assert gate._probe_values(report(5), 2, 5) == (1.0, 2.0)
    assert gate._work_census(report(5)) == {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "logical_current_count": 5,
        "resident_current_count": 5,
        "resident_current_component_count": 8,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
        "semantic_executor_binding_count": 4,
        "distinct_prepared_executor_count": 3,
    }
    assert gate._calibrate(0.25, 1.0) == 4

    poisoned = report(0)
    poisoned["direct_plan_load_attempts"] = 1
    with pytest.raises(gate.GateError, match="poison"):
        gate._probe_values(poisoned, 2)
    wrong_fill = report(5)
    wrong_fill["momentum_fill_count"] = 6
    with pytest.raises(gate.GateError, match="contract"):
        gate._probe_values(wrong_fill, 2, 5)
    inconsistent_work = report(0)
    inconsistent_work["total_kernel_application_count"] = 8
    with pytest.raises(gate.GateError, match="kernel and operation"):
        gate._probe_values(inconsistent_work, 2)


def test_workload_census_sums_operations_and_keeps_recurrence_calls_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
    }
    second = {
        "work_census_basis": gate.WORK_CENSUS_BASIS,
        "source_operation_count": 2,
        "contribution_operation_count": 5,
        "finalization_operation_count": 2,
        "closure_operation_count": 1,
        "total_kernel_application_count": 10,
    }
    assert gate._workload_operation_census(
        ({"work_census": first}, {"work_census": second})
    ) == {
        "aggregation_basis": "sum-one-execution-per-serialized-query-v1",
        "query_census_basis": gate.WORK_CENSUS_BASIS,
        "query_count": 2,
        "source_operation_count": 4,
        "contribution_operation_count": 8,
        "finalization_operation_count": 3,
        "closure_operation_count": 2,
        "total_kernel_application_count": 17,
    }

    counters = SimpleNamespace(
        normalization="mean_per_profiled_point_or_runtime_call_v1",
        recurrence_source_calls_per_call=2.0,
        recurrence_source_rows_per_call=4.0,
        recurrence_contribution_calls_per_call=3.0,
        recurrence_contribution_rows_per_call=8.0,
        recurrence_finalization_calls_per_call=1.0,
        recurrence_finalization_rows_per_call=3.0,
        recurrence_closure_calls_per_call=1.0,
        recurrence_closure_rows_per_call=2.0,
    )
    established = gate._public_recurrence_work(
        SimpleNamespace(timing_breakdown=SimpleNamespace(counters=counters))
    )
    assert established is not None
    assert established["source_calls_per_runtime_call"] == 2.0
    assert established["source_rows_per_runtime_call"] == 4.0
    assert "grouped prepared-backend invocations" in str(established["semantics"])
    assert gate._public_recurrence_work(SimpleNamespace(timing_breakdown=None)) is None

    monkeypatch.setattr(gate.dataclasses, "asdict", lambda _value: {})
    public = gate._public_timing(
        SimpleNamespace(
            sample_count=3,
            repetitions_per_sample=4,
            wall_time_per_point=2.0,
            evaluator_time_per_point=1.5,
            evaluator_total_time_per_point=1.75,
            interrupted=False,
            effective_config=SimpleNamespace(),
            uncertainty=SimpleNamespace(),
            timing_breakdown=SimpleNamespace(
                counters=counters, execution_mode="recurrence"
            ),
        ),
        "recurrence",
    )
    assert public["execution_mode"] == "recurrence"
    assert public["evaluator_seconds_per_point"] == 1.5
    assert public["clock_attribution"]["evaluator_seconds_per_point"] == (
        "recurrence core"
    )
    assert "clocks are independent" in public["clock_attribution"]["relationship"]
    with pytest.raises(gate.GateError, match="matching timing breakdown"):
        gate._public_timing(
            SimpleNamespace(
                timing_breakdown=SimpleNamespace(execution_mode="compiled")
            ),
            "on-the-fly",
        )


def test_query_family_census_matches_query_local_work_and_retains_destinations() -> (
    None
):
    observed: dict[str, object] = {}

    def probe(*args: object, **kwargs: object) -> dict[str, int]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "query_count": 2,
            "source_frame_partition_count": 1,
            "projection_applied_query_count": 2,
            "projection_pre_current_count": 12,
            "projection_pre_contribution_count": 10,
            "projection_pre_closure_count": 2,
            "projection_post_current_count": 10,
            "projection_post_contribution_count": 8,
            "projection_post_closure_count": 2,
            "dynamic_current_occurrence_count": 10,
            "dynamic_current_component_occurrence_count": 16,
            "dynamic_source_rows": 4,
            "dynamic_contribution_rows": 8,
            "dynamic_finalization_rows": 3,
            "dynamic_closure_rows": 2,
            "dynamic_source_calls": 4,
            "dynamic_contribution_calls": 8,
            "dynamic_finalization_calls": 3,
            "dynamic_closure_calls": 2,
            "union_unique_current_count": 6,
            "union_unique_current_component_count": 10,
            "union_source_rows": 2,
            "union_contribution_rows": 5,
            "union_finalization_rows": 2,
            "union_closure_rows": 2,
            "union_amplitude_destination_count": 2,
            "union_source_executor_call_groups": 2,
            "union_contribution_executor_call_groups": 3,
            "union_finalization_executor_call_groups": 1,
            "union_closure_executor_call_groups": 1,
        }

    retained = gate.RetainedInputs("builder", "template", b"{}", "a" * 64)
    queries = (
        gate.Query("flow:0", 0, "h:0", (1, -1)),
        gate.Query("flow:1", 1, "h:1", (-1, 1)),
    )
    hidden = {
        "queries": [
            {
                "work_census": {
                    "logical_current_count": 5,
                    "resident_current_component_count": 8,
                }
            },
            {
                "work_census": {
                    "logical_current_count": 5,
                    "resident_current_component_count": 8,
                }
            },
        ],
        "workload_operation_census": {
            "source_operation_count": 4,
            "contribution_operation_count": 8,
            "finalization_operation_count": 3,
            "closure_operation_count": 2,
        },
    }
    result = gate._query_family_census(probe, retained, queries, hidden, True)
    assert result["basis"] == "exact-current-core-key-query-family-union-v1"
    assert result["union_unique_current_count"] == 6
    assert result["union_amplitude_destination_count"] == 2
    assert observed["args"][4] == [0, 1]
    assert observed["args"][5] == [[1, -1], [-1, 1]]
    assert observed["kwargs"] == {"enable_color_projection": True}

    def stale(*_args: object, **_kwargs: object) -> dict[str, int]:
        value = probe(*_args, **_kwargs)
        value["dynamic_contribution_rows"] = 7
        return value

    with pytest.raises(gate.GateError, match="independently timed"):
        gate._query_family_census(stale, retained, queries, hidden, True)


def test_executable_family_report_has_exact_union_work_and_ordered_outputs() -> None:
    queries = (
        gate.Query("flow:0", 0, "h:0", (1, -1)),
        gate.Query("flow:1", 1, "h:1", (-1, 1)),
    )
    census = {
        "query_count": 2,
        "source_frame_partition_count": 1,
        "projection_applied_query_count": 2,
        "projection_pre_current_count": 12,
        "projection_pre_contribution_count": 10,
        "projection_pre_closure_count": 2,
        "projection_post_current_count": 10,
        "projection_post_contribution_count": 8,
        "projection_post_closure_count": 2,
        "dynamic_current_occurrence_count": 10,
        "dynamic_current_component_occurrence_count": 16,
        "dynamic_source_rows": 4,
        "dynamic_contribution_rows": 8,
        "dynamic_finalization_rows": 3,
        "dynamic_closure_rows": 2,
        "dynamic_source_calls": 4,
        "dynamic_contribution_calls": 8,
        "dynamic_finalization_calls": 3,
        "dynamic_closure_calls": 2,
        "union_unique_current_count": 6,
        "union_unique_current_component_count": 10,
        "union_source_rows": 2,
        "union_contribution_rows": 5,
        "union_finalization_rows": 2,
        "union_closure_rows": 2,
        "union_amplitude_destination_count": 2,
        "union_source_executor_call_groups": 1,
        "union_contribution_executor_call_groups": 3,
        "union_finalization_executor_call_groups": 1,
        "union_closure_executor_call_groups": 1,
    }
    rows = [
        {
            "selected_public_flow_id": query.flow_index,
            "public_helicities": list(query.helicities),
            "query_digest": str(index) * 64,
            "raw_amplitudes": [[1.0 + index, 2.0], [3.0, 4.0 + index]],
            "normalized_values": [5.0 + index, 6.0 + index],
        }
        for index, query in enumerate(queries, start=1)
    ]
    report = {
        "process_id": gate.PROCESS_ID,
        "point_count": 2,
        "work_census_basis": gate.FAMILY_WORK_CENSUS_BASIS,
        "source_operation_count": 2,
        "contribution_operation_count": 5,
        "finalization_operation_count": 2,
        "closure_operation_count": 2,
        "total_kernel_application_count": 11,
        "trace_build_count": 2,
        "trace_cache_hit_count": 0,
        "momentum_fill_count": gate.WARMUPS + 3,
        "currents": [],
        "direct_plan_load_attempts": 0,
        "direct_plan_decode_attempts": 0,
        "direct_plan_materialization_attempts": 0,
        "established_builder_attempts": 0,
        "query_family": {
            "queries": rows,
            "census": census,
            "execution_cache_hit": True,
            "execution_source_calls": 1,
            "execution_source_rows": 2,
            "execution_contribution_calls": 3,
            "execution_contribution_rows": 5,
            "execution_finalization_calls": 1,
            "execution_finalization_rows": 2,
            "execution_closure_calls": 1,
            "execution_closure_rows": 2,
            "cold_prepare_seconds": 0.01,
            "private_warmed_elapsed_seconds": 0.12,
            "private_warmed_seconds_per_point": 0.02,
            "private_timing_excludes_source_crossing": True,
        },
    }
    parsed = gate._family_probe_result(report, queries, 2, 3)
    assert parsed["union_total_kernel_application_count"] == 11
    assert parsed["census"] == census
    gate._assert_executable_family_matches_structural_census(census, census)

    broken = dict(report)
    broken["total_kernel_application_count"] = 16
    with pytest.raises(gate.GateError, match="top-level execution census"):
        gate._family_probe_result(broken, queries, 2, 3)


def test_separate_recurrence_artifacts_route_exact_selector_certificates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "recurrence-selected"
    all_flow = tmp_path / "recurrence-all-flow"
    whole = {
        "source_row_count": 2,
        "current_count": 7,
        "semantic_component_count": 11,
        "contribution_count": 5,
        "finalization_count": 3,
        "closure_count": 2,
        "row_count": 12,
    }

    def recurrence(layout: str, representatives: list[object]) -> dict[str, object]:
        return {
            "kind": "pyamplicol-runtime-recurrence-execution",
            "recurrence_summary": {
                "current_count": 7,
                "contribution_count": 5,
                "closure_term_count": 2,
            },
            "runtime_metadata": {
                "public_color_flows": [
                    {"public_id": gate.FLOW_ID, "target_sector_id": 8}
                ]
            },
            "plan": {
                "inspection_summary": {
                    "lc_flow_layout": layout,
                    "schedule": {
                        "source_row_count": 2,
                        "current_count": 7,
                        "contribution_count": 5,
                        "finalization_count": 3,
                        "closure_term_count": 2,
                        "amplitude_destination_count": 1,
                    },
                    "direct_arena": {"semantic_component_count": 11},
                    "selector_work_certificate": {
                        "persisted_union": whole,
                        "representatives": representatives,
                    },
                }
            },
        }

    live = {
        "representative_sector_id": 8,
        "source_row_count": 1,
        "current_count": 4,
        "semantic_component_count": 6,
        "contribution_count": 2,
        "finalization_count": 1,
        "closure_count": 1,
        "amplitude_destination_count": 1,
        "row_count": 5,
    }
    _write_execution(selected, recurrence("topology-replay", [live]))
    _write_execution(all_flow, recurrence("all-flow-union", []))
    flow = SimpleNamespace(id=gate.FLOW_ID, index=8)
    helicity = SimpleNamespace(id=gate.HELICITY_ID, index=21)
    runtimes = {
        selected.resolve(): _runtime("recurrence", flows=(flow,)),
        all_flow.resolve(): _runtime("recurrence", helicities=(helicity,)),
    }
    loads: list[Path] = []

    def load(path: Path) -> SimpleNamespace:
        loads.append(path)
        return runtimes[path]

    monkeypatch.setattr(gate.Runtime, "load", load)
    selected_result = gate._recurrence_artifact_census(
        selected, layout="topology-replay"
    )
    all_result = gate._recurrence_artifact_census(all_flow, layout="all-flow-union")

    assert loads == [selected.resolve(), all_flow.resolve()]
    assert selected_result["selector_live"]["current_count"] == 4
    assert selected_result["selector_live"]["kernel_row_count"] == 5
    assert selected_result["whole"]["current_count"] == 7
    assert all_result["whole"] == all_result["selector_live"]
    assert all_result["selector"]["public_index"] == 21

    runtimes[selected.resolve()] = _runtime(
        "recurrence", flows=(flow,), process="g g > g g g g"
    )
    with pytest.raises(gate.GateError, match="wrong canonical process identity"):
        gate._recurrence_artifact_census(selected, layout="topology-replay")


def test_compiled_census_selects_one_exact_child_without_summing_alternatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "compiled-selected"
    all_flow = tmp_path / "compiled-all-flow"
    primary = _compiled_record(
        sources=8,
        currents=69,
        components=258,
        attachments=124,
        evaluations=112,
        roots=12,
    )
    program = _compiled_record(
        sources=12,
        currents=256,
        components=984,
        attachments=700,
        evaluations=650,
        roots=192,
    )
    chosen_leaf = _compiled_record(
        sources=12,
        currents=78,
        components=296,
        attachments=150,
        evaluations=150,
        roots=32,
    )
    alternative = _compiled_record(
        sources=99,
        currents=999,
        components=1999,
        attachments=999,
        evaluations=999,
        roots=999,
    )
    program.update(
        {
            "physics_reduction": {"groups": [{"physical_color_ids": [gate.FLOW_ID]}]},
            "color_selector_executions": [
                {"materialized_sector_id": 8, "execution": chosen_leaf},
                {"materialized_sector_id": 3, "execution": alternative},
            ],
        }
    )
    selected_payload = dict(primary)
    selected_payload.update(
        {
            "compiled": {
                "lc_topology_replay": {
                    "groups": [
                        {"active_sector_ids": [8, 11], "materialized_sector_id": 8},
                        {"active_sector_ids": [3], "materialized_sector_id": 3},
                    ]
                }
            },
            "helicity_sum_execution": program,
        }
    )
    _write_execution(selected, selected_payload)

    all_primary = _compiled_record(
        sources=8,
        currents=115,
        components=440,
        attachments=233,
        evaluations=189,
        roots=24,
    )
    middle = dict(all_primary)
    middle["helicity_selector_executions"] = [
        {"selector_domain_ids": [21], "execution": dict(all_primary)},
        {"selector_domain_ids": [5], "execution": alternative},
    ]
    all_payload = dict(all_primary)
    all_payload["helicity_selector_executions"] = [
        {"selector_domain_ids": [21, 22], "execution": middle},
        {"selector_domain_ids": [4], "execution": alternative},
    ]
    _write_execution(all_flow, all_payload)

    flow = SimpleNamespace(id=gate.FLOW_ID, index=8)
    helicity = SimpleNamespace(id=gate.HELICITY_ID, index=21)
    runtimes = {
        selected.resolve(): _runtime("compiled", flows=(flow,)),
        all_flow.resolve(): _runtime("compiled", helicities=(helicity,)),
    }
    loads: list[Path] = []

    def load(path: Path) -> SimpleNamespace:
        loads.append(path)
        return runtimes[path]

    monkeypatch.setattr(gate.Runtime, "load", load)
    selected_result = gate._compiled_artifact_census(
        selected, workload="selected_flow_helicity_sum"
    )
    all_result = gate._compiled_artifact_census(
        all_flow, workload="all_flow_single_helicity"
    )

    assert loads == [selected.resolve(), all_flow.resolve()]
    assert selected_result["levels"]["primary"]["current_count"] == 69
    assert selected_result["levels"]["program"]["current_count"] == 256
    assert selected_result["levels"]["executed_leaf"]["current_count"] == 78
    assert selected_result["levels"]["executed_leaf"]["current_component_count"] == 296
    assert all_result["levels"]["primary"]["current_count"] == 115
    assert all_result["levels"]["executed_leaf"]["current_count"] == 115
    assert all_result["selector"]["selector_depth"] == 2
    assert "never summed" in selected_result["semantics"]
    assert "999" not in json.dumps(selected_result)

    program["color_selector_executions"].append(
        {"materialized_sector_id": 8, "execution": alternative}
    )
    execution_path = selected / "processes" / "fixture_process" / "execution.json"
    execution_path.write_text(json.dumps(selected_payload), encoding="utf-8")
    with pytest.raises(gate.GateError, match="absent or ambiguous"):
        gate._compiled_artifact_census(selected, workload="selected_flow_helicity_sum")

    with pytest.raises(gate.GateError, match="absent or ambiguous"):
        gate._exact_public_index((), gate.FLOW_ID, "test")
    with pytest.raises(gate.GateError, match="absent or ambiguous"):
        gate._exact_public_index((flow, flow), gate.FLOW_ID, "test")


def test_hidden_timing_serializes_one_query_census_and_workload_sum() -> None:
    def probe(*args: object, **kwargs: object) -> dict[str, object]:
        point_count = int(args[9])
        repetitions = int(kwargs["benchmark_repetitions"])
        cycles = gate.WARMUPS + repetitions
        elapsed = repetitions * point_count * 0.01
        return {
            "process_id": gate.PROCESS_ID,
            "point_count": point_count,
            "work_census_basis": gate.WORK_CENSUS_BASIS,
            "logical_current_count": 5,
            "resident_current_count": 5,
            "resident_current_component_count": 8,
            "source_operation_count": 2,
            "contribution_operation_count": 3,
            "finalization_operation_count": 1,
            "closure_operation_count": 1,
            "total_kernel_application_count": 7,
            "semantic_executor_binding_count": 4,
            "distinct_prepared_executor_count": 3,
            "trace_build_count": 1,
            "trace_cache_hit_count": cycles,
            "momentum_fill_count": cycles,
            "currents": [],
            "direct_plan_load_attempts": 0,
            "direct_plan_decode_attempts": 0,
            "direct_plan_materialization_attempts": 0,
            "established_builder_attempts": 0,
            "normalized_values": [1.0] * point_count,
            "benchmark_elapsed_seconds": elapsed,
            "benchmark_seconds_per_point": 0.01,
            "trace_digest": "a" * 64,
        }

    retained = gate.RetainedInputs(object(), object(), b"{}", "a" * 64)
    query = gate.Query("flow", 0, "helicity", (1, -1))
    result = gate._hidden_timing(
        probe,
        Path("artifact"),
        retained,
        (query,),
        (((1.0,),), ((2.0,),)),
        target=0.04,
    )
    row = result["queries"][0]
    assert row["work_census"]["logical_current_count"] == 5
    assert result["workload_operation_census"] == {
        "aggregation_basis": "sum-one-execution-per-serialized-query-v1",
        "query_census_basis": gate.WORK_CENSUS_BASIS,
        "query_count": 1,
        "source_operation_count": 2,
        "contribution_operation_count": 3,
        "finalization_operation_count": 1,
        "closure_operation_count": 1,
        "total_kernel_application_count": 7,
    }


def test_cli_launches_one_worker_with_cross_platform_30_gib_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_model = tmp_path / "built-in-sm.pyamplicol-model"
    prepared_model.write_bytes(b"prepared")
    arguments = gate._parser().parse_args(
        [
            "--output",
            "out",
            "--prepared-model",
            str(prepared_model),
            "--amplicol-result",
            "legacy.json",
            "--recurrence-selected-artifact",
            "recurrence-selected",
            "--recurrence-all-flow-artifact",
            "recurrence-all-flow",
            "--compiled-selected-artifact",
            "compiled-selected",
            "--compiled-all-flow-artifact",
            "compiled-all-flow",
            "--target-runtime",
            "2",
            "--batch-size",
            "64",
        ]
    )
    assert "--worker" not in gate._parser().format_help()
    command = gate._worker_command(arguments, Path("/tmp/gate"))
    assert command.count("--worker") == 1
    assert "all-flow-union" not in command
    assert command[command.index("--prepared-model") + 1] == str(
        prepared_model.resolve()
    )
    assert command[command.index("--amplicol-result") + 1] == str(
        Path("legacy.json").resolve()
    )
    for option, name in (
        ("--recurrence-selected-artifact", "recurrence-selected"),
        ("--recurrence-all-flow-artifact", "recurrence-all-flow"),
        ("--compiled-selected-artifact", "compiled-selected"),
        ("--compiled-all-flow-artifact", "compiled-all-flow"),
    ):
        assert command[command.index(option) + 1] == str(Path(name).resolve())
    assert arguments.bypass_color_projection is False
    assert "--bypass-color-projection" not in command
    bypass = arguments
    bypass.bypass_color_projection = True
    assert "--bypass-color-projection" in gate._worker_command(
        bypass, Path("/tmp/gate")
    )
    assert gate.WATCHDOG_BYTES == 30 * gate.GIB

    summary = gate._watchdog_summary(
        {
            "passes": True,
            "execution": {"outcome": "command-finished", "reason": None},
            "enforcement": {
                "limit_bytes": gate.WATCHDOG_BYTES,
                "peak_rss_bytes": 10,
                "peak_physical_footprint_bytes": 11,
                "peak_guard_bytes": 11,
                "peak_processes": 2,
            },
        }
    )
    assert summary["passes"] is True
    assert summary["peak_guard_bytes"] == 11

    def probe(_pids: object) -> dict[int, int]:
        return {}

    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gate, "DarwinPhysicalFootprintProbe", lambda: probe)
    assert gate._physical_footprint_probe() is probe
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    assert gate._physical_footprint_probe() is None


def test_prepared_model_load_precedes_generation_and_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "built-in-sm.pyamplicol-model"
    path.write_bytes(b"prepared")
    events: list[object] = []
    compiled = object()
    retained = gate.RetainedInputs(object(), object(), b"{}", "a" * 64)

    class Source:
        def compile(self) -> object:
            events.append("load")
            return compiled

    def from_path(candidate: Path) -> Source:
        events.append(("path", candidate))
        return Source()

    def generate_candidate(
        artifact: Path, model: object
    ) -> tuple[float, dict[str, int]]:
        events.append(("generate-candidate", artifact, model))
        return 2.5, {
            "expanded_process_count": 1,
            "on_the_fly_seed_batch_binding_call_count": 1,
            "on_the_fly_seed_build_count": 1,
            "recurrence_lowering_call_count": 0,
            "materialized_process_lane_count": 0,
        }

    def generate_carrier(artifact: Path, model: object) -> gate.RetainedInputs:
        events.append(("generate-carrier", artifact, model))
        return retained

    monkeypatch.setattr(gate.ModelSource, "from_path", from_path)
    monkeypatch.setattr(gate, "_generate_on_the_fly", generate_candidate)
    monkeypatch.setattr(gate, "_generate_probe_carrier", generate_carrier)

    result = gate._generate_with_prepared_model(
        tmp_path / "artifact", tmp_path / "carrier", path
    )

    assert events == [
        ("path", path.resolve()),
        "load",
        ("generate-candidate", tmp_path / "artifact", compiled),
        ("generate-carrier", tmp_path / "carrier", compiled),
    ]
    assert result[0] == 2.5
    assert result[1]["on_the_fly_seed_batch_binding_call_count"] == 1
    assert result[1]["on_the_fly_seed_build_count"] == 1
    assert result[2:4] == (retained, path.resolve())
    assert result[4] >= 0.0


def test_candidate_generation_uses_seed_binding_and_never_recurrence_lowering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_modes: list[str] = []
    seed_inputs: list[tuple[object, ...]] = []

    def seed_builder(*args: object, **_kwargs: object) -> tuple[bytes, ...]:
        seed_inputs.append(args)
        return (b"seed",)

    class Generator:
        def __init__(self, config: object) -> None:
            seen_modes.append(config.evaluator.execution_mode)

        def generate(self, *_args: object, **_kwargs: object) -> None:
            gate.generation_service._invoke_rust_on_the_fly_seed_batch_builder_v1(
                (b'{"schema_version":1}',),
                b"templates",
                b"direct",
                "a" * 64,
            )

    monkeypatch.setattr(
        gate.generation_service,
        "_invoke_rust_on_the_fly_seed_batch_builder_v1",
        seed_builder,
    )
    monkeypatch.setattr(gate, "Generator", Generator)
    elapsed, census = gate._generate_on_the_fly(tmp_path / "artifact", object())
    assert elapsed >= 0.0
    assert seen_modes == ["on-the-fly"]
    assert len(seed_inputs) == 1
    assert census == {
        "expanded_process_count": 1,
        "on_the_fly_seed_batch_binding_call_count": 1,
        "on_the_fly_seed_build_count": 1,
        "recurrence_lowering_call_count": 0,
        "materialized_process_lane_count": 0,
    }


@pytest.mark.parametrize("symbol", gate.MATERIALIZED_PROCESS_LANE_SYMBOLS)
def test_candidate_generation_fails_fast_for_every_materialized_lane_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    seen_modes: list[str] = []

    class Generator:
        def __init__(self, config: object) -> None:
            seen_modes.append(config.evaluator.execution_mode)

        def generate(self, *_args: object, **_kwargs: object) -> None:
            if symbol == "GenerationBackend._prepare_process_construction":
                gate.generation_service.GenerationBackend._prepare_process_construction(
                    object()
                )
            else:
                getattr(gate.generation_service, symbol)()
            raise AssertionError("materialized lane poison returned to Generator")

    monkeypatch.setattr(gate, "Generator", Generator)
    with pytest.raises(gate.GateError, match="materialized process lane"):
        gate._generate_on_the_fly(tmp_path / symbol.replace(".", "-"), object())
    assert seen_modes == ["on-the-fly"]


def test_candidate_generation_rejects_materialized_source_projection_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_binding_called = False

    def seed_builder(*_args: object, **_kwargs: object) -> tuple[bytes, ...]:
        nonlocal native_binding_called
        native_binding_called = True
        return (b"seed",)

    class Generator:
        def __init__(self, _config: object) -> None:
            pass

        def generate(self, *_args: object, **_kwargs: object) -> None:
            gate.generation_service._invoke_rust_on_the_fly_seed_batch_builder_v1(
                (b'{"schema_version":1,"dag":{}}',),
                b"templates",
                b"direct",
                "a" * 64,
            )

    monkeypatch.setattr(
        gate.generation_service,
        "_invoke_rust_on_the_fly_seed_batch_builder_v1",
        seed_builder,
    )
    monkeypatch.setattr(gate, "Generator", Generator)
    with pytest.raises(gate.GateError, match="forbidden materialization field"):
        gate._generate_on_the_fly(tmp_path / "source-projection", object())
    assert native_binding_called is False


def test_public_candidate_mismatch_fails_before_any_private_probe_can_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    selected = tmp_path / "selected"
    all_flow = tmp_path / "all-flow"
    compiled_selected = tmp_path / "compiled-selected"
    compiled_all = tmp_path / "compiled-all"
    flow = SimpleNamespace(id=gate.FLOW_ID, word=gate.FLOW_WORD, index=0)
    helicity = SimpleNamespace(
        id=gate.HELICITY_ID,
        values=gate.HELICITY_VALUES,
        index=0,
        structural_zero=False,
    )
    physics = SimpleNamespace(color_flows=(flow,), helicities=(helicity,))

    class Resolved:
        helicity_ids = (gate.HELICITY_ID,)
        color_ids = (gate.FLOW_ID,)
        values = (((1.0,),),)

        @staticmethod
        def total() -> tuple[float, ...]:
            return (1.0,)

    class Authority:
        execution_mode = "recurrence"
        artifact_id = "b" * 64

        @staticmethod
        def evaluate_resolved(*_args: object, **_kwargs: object) -> Resolved:
            return Resolved()

        @staticmethod
        def evaluate(*_args: object, **_kwargs: object) -> tuple[float, ...]:
            return (1.0,)

    class Candidate:
        execution_mode = "on-the-fly"
        artifact_id = "a" * 64
        representative_process_key = gate.PROCESS_ID

        @property
        def physics(self) -> object:
            raise AssertionError("candidate dense physics was opened")

        @staticmethod
        def evaluate(*_args: object, **_kwargs: object) -> tuple[float, ...]:
            return (2.0,)

        @staticmethod
        def evaluate_resolved(*_args: object, **_kwargs: object) -> Resolved:
            return Resolved()

    candidate = Candidate()
    authority = Authority()
    authority.physics = physics
    retained = gate.RetainedInputs(object(), object(), b"{}", "c" * 64)
    monkeypatch.setattr(
        gate,
        "_generate_with_prepared_model",
        lambda *_args: (
            0.1,
            {
                "expanded_process_count": 1,
                "on_the_fly_seed_batch_binding_call_count": 1,
                "on_the_fly_seed_build_count": 1,
                "recurrence_lowering_call_count": 0,
                "materialized_process_lane_count": 0,
            },
            retained,
            tmp_path / "model.pyamplicol-model",
            0.01,
        ),
    )
    monkeypatch.setattr(gate, "_on_the_fly_artifact_contract", lambda *_args: {})
    monkeypatch.setattr(gate, "_points", lambda: ((((1.0,),),)))

    def load(path: Path, **_kwargs: object) -> object:
        path = Path(path)
        if path == output / "artifact":
            return candidate
        if path in (selected, all_flow):
            return authority
        if path.name == "recurrence-artifact":
            return authority
        raise AssertionError(f"unexpected load before mismatch: {path}")

    monkeypatch.setattr(gate.Runtime, "load", load)
    monkeypatch.setattr(
        gate,
        "_probe",
        lambda: (_ for _ in ()).throw(
            AssertionError("private probe ran before public authority check")
        ),
    )
    with pytest.raises(gate.GateError, match="disagrees"):
        gate._run(
            output,
            tmp_path / "model.pyamplicol-model",
            None,
            selected,
            all_flow,
            compiled_selected,
            compiled_all,
            0.1,
            1,
        )


def test_prepared_model_path_rejects_missing_non_file_and_wrong_suffix(
    tmp_path: Path,
) -> None:
    with pytest.raises(gate.GateError, match="does not exist"):
        gate._prepared_model_path(tmp_path / "missing.pyamplicol-model")

    directory = tmp_path / "directory.pyamplicol-model"
    directory.mkdir()
    with pytest.raises(gate.GateError, match="not a regular file"):
        gate._prepared_model_path(directory)

    wrong_suffix = tmp_path / "prepared.bin"
    wrong_suffix.write_bytes(b"prepared")
    with pytest.raises(gate.GateError, match="must end"):
        gate._prepared_model_path(wrong_suffix)
