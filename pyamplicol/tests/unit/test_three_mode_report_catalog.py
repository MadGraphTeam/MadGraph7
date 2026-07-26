# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from tools.performance_report.agreements import DIRECT_AGREEMENT_FIELD
from tools.performance_report.cache import (
    CACHE_SCHEMA_VERSION,
    REPORT_VERSION,
    build_reset_caches,
    digest_json,
    empty_measurement,
    validate_cache,
    validate_measurement,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)


def test_matrix_catalog_has_requested_twelve_datasets() -> None:
    datasets = REPORT_CATALOG.matrix_datasets

    assert len(datasets) == 12
    assert Counter(item.candidate.execution_mode for item in datasets) == {
        ExecutionMode.RECURRENCE: 6,
        ExecutionMode.COMPILED: 3,
        ExecutionMode.EAGER: 3,
    }
    assert Counter(item.candidate.model for item in datasets) == {
        ModelKey.BUILTIN_SM: 9,
        ModelKey.UFO_SM: 3,
    }
    assert all(
        item.baseline.execution_mode is ExecutionMode.AMPLICOL
        for item in datasets
        if item.candidate.execution_mode is ExecutionMode.RECURRENCE
    )
    assert all(
        item.baseline.execution_mode is ExecutionMode.RECURRENCE
        for item in datasets
        if item.candidate.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    )


def test_required_agreement_evidence_bumps_report_cache_contract() -> None:
    assert CACHE_SCHEMA_VERSION == 4
    assert REPORT_VERSION == "0.3.0"


def test_prepared_modes_are_portable_o2_and_compiled_is_o3() -> None:
    by_mode = {
        mode: {
            dataset.candidate.jit_optimization_level
            for dataset in REPORT_CATALOG.matrix_datasets
            if dataset.candidate.execution_mode is mode
        }
        for mode in (
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        )
    }

    assert by_mode == {
        ExecutionMode.RECURRENCE: {2},
        ExecutionMode.COMPILED: {3},
        ExecutionMode.EAGER: {2},
    }


def test_lc_cells_have_two_runtime_workloads_and_contracted_cells_have_one() -> None:
    counts = Counter(
        (cell.measurement.accuracy, cell.workload)
        for cell in REPORT_CATALOG.matrix_cells()
    )

    assert counts[(Accuracy.LC, Workload.SELECTED_FLOW)] == 388
    assert counts[(Accuracy.LC, Workload.ALL_FLOW)] == 388
    assert counts[(Accuracy.NLC, Workload.CONTRACTED)] == 180
    assert counts[(Accuracy.FULL, Workload.CONTRACTED)] == 180
    assert all(
        cell.workload is Workload.CONTRACTED
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.accuracy is not Accuracy.LC
    )


def test_n_le_four_new_matrix_smoke_has_384_logical_process_cells() -> None:
    cells = [cell for cell in REPORT_CATALOG.matrix_cells() if cell.n_final <= 4]
    logical = {(cell.dataset_id, cell.process_key, cell.n_final) for cell in cells}

    assert len(logical) == 384
    assert len(cells) == 512


def test_z_catalog_adds_recurrence_and_corrects_eager_label() -> None:
    variants = {variant.key: variant for variant in REPORT_CATALOG.z_variants}

    assert variants["eager_jit_o2"].label == "eager-DAG JIT O2"
    assert variants["recurrence_jit_o2"].label == "recurrence JIT O2"
    assert variants["recurrence_jit_o2"].jit_optimization_level == 2
    assert (
        len(
            [
                cell
                for cell in REPORT_CATALOG.z_cells()
                if cell.variant == "recurrence_jit_o2" and cell.n_final <= 4
            ]
        )
        == 16
    )


def test_scalar_ladders_are_canonical_resettable_cells() -> None:
    assert tuple(dataset.dataset_id for dataset in REPORT_CATALOG.scalar_datasets) == (
        "scalar_contact",
        "scalar_gravity",
    )
    cells = REPORT_CATALOG.scalar_cells()
    assert len(cells) == 10
    assert all(cell.workload is Workload.CONTRACTED for cell in cells)
    assert all(
        cell.measurement.execution_mode is ExecutionMode.COMPILED for cell in cells
    )
    assert all(REPORT_CATALOG.baseline_cell(cell) is None for cell in cells)


def test_baseline_dependencies_are_canonical_and_mode_ordered() -> None:
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    compiled = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.COMPILED
    )
    eager = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.EAGER
    )

    recurrence_baseline = REPORT_CATALOG.baseline_cell(recurrence)
    compiled_baseline = REPORT_CATALOG.baseline_cell(compiled)
    eager_baseline = REPORT_CATALOG.baseline_cell(eager)
    assert recurrence_baseline is not None
    assert recurrence_baseline.measurement.execution_mode is ExecutionMode.AMPLICOL
    assert compiled_baseline is not None
    assert compiled_baseline.measurement.execution_mode is ExecutionMode.RECURRENCE
    assert eager_baseline is not None
    assert eager_baseline.measurement.execution_mode is ExecutionMode.RECURRENCE
    assert REPORT_CATALOG.cell(compiled.cell_id) == compiled


def test_equivalent_cells_require_exact_generation_semantics() -> None:
    selected_recurrence = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 3
        and cell.workload is Workload.SELECTED_FLOW
    )
    equivalents = REPORT_CATALOG.equivalent_cells(selected_recurrence)

    assert equivalents
    assert all(
        candidate.cell_id != selected_recurrence.cell_id for candidate in equivalents
    )
    assert all(
        (
            candidate.process,
            candidate.n_final,
            candidate.process_key,
            candidate.measurement,
            candidate.workload,
        )
        == (
            selected_recurrence.process,
            selected_recurrence.n_final,
            selected_recurrence.process_key,
            selected_recurrence.measurement,
            selected_recurrence.workload,
        )
        for candidate in equivalents
    )
    assert {(candidate.dataset_id, candidate.variant) for candidate in equivalents} == {
        ("z_builtin_sm", "recurrence_jit_o2")
    }

    same_process_cells = [
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.process == selected_recurrence.process
        and cell.n_final == selected_recurrence.n_final
        and cell.process_key == selected_recurrence.process_key
    ]
    assert not {
        candidate.cell_id
        for candidate in same_process_cells
        if (
            candidate.measurement.model is ModelKey.UFO_SM
            or candidate.workload is Workload.ALL_FLOW
            or candidate.measurement.backend != selected_recurrence.measurement.backend
            or candidate.measurement.execution_mode
            is not selected_recurrence.measurement.execution_mode
        )
    } & {candidate.cell_id for candidate in equivalents}


def test_equivalent_cells_cover_matching_compiled_backend_but_never_amplicol() -> None:
    compiled = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 2
        and cell.workload is Workload.ALL_FLOW
    )
    equivalents = REPORT_CATALOG.equivalent_cells(compiled)

    assert {(candidate.dataset_id, candidate.variant) for candidate in equivalents} == {
        ("z_builtin_sm", "jit_o3")
    }
    assert all(candidate.measurement.backend == "jit" for candidate in equivalents)
    assert all(
        candidate.measurement.jit_optimization_level == 3 for candidate in equivalents
    )

    reference = next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.process_key == "dd_z_jets"
        and cell.n_final == 2
        and cell.workload is Workload.ALL_FLOW
    )
    assert REPORT_CATALOG.equivalent_cells(reference) == ()


def test_reset_caches_are_canonical_and_cover_every_measurement_cell() -> None:
    caches = build_reset_caches()
    cells_by_dataset: dict[str, list[object]] = {}
    for cell in REPORT_CATALOG.measurement_cells():
        cells_by_dataset.setdefault(cell.dataset_id, []).append(cell)

    assert set(caches) == {f"{dataset_id}.json" for dataset_id in cells_by_dataset}
    for name, payload in caches.items():
        dataset_id = name.removesuffix(".json")
        validate_cache(payload, expected_cells=cells_by_dataset[dataset_id])
        assert all(
            entry["measurement"] == empty_measurement() for entry in payload["entries"]
        )


def test_not_available_measurement_cannot_carry_stale_timing() -> None:
    measurement = empty_measurement()
    measurement["generation_seconds"] = 1.0

    with pytest.raises(ValueError, match="canonical reset"):
        validate_measurement(measurement)


def test_successful_measurement_requires_successful_validation() -> None:
    measurement = empty_measurement()
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "artifact": {},
            "validation": {"status": ResultStatus.VALIDATION_FAILED.value},
            "resources": {},
            "provenance": {},
        }
    )

    with pytest.raises(ValueError, match="successful validation"):
        validate_measurement(measurement)


def _candidate_measurement_with_runtime_postflight() -> dict[str, object]:
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 10,
            "sha256": "1" * 64,
        }
    ]
    policy = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": len(observations),
        "observations": observations,
        "observations_sha256": digest_json(observations),
    }
    identity = {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "loaded_module_origin_policy": policy,
    }
    stable_identity = deepcopy(identity)
    stable_policy = stable_identity["loaded_module_origin_policy"]
    assert isinstance(stable_policy, dict)
    for field in (
        "observed_module_count",
        "observations",
        "observations_sha256",
    ):
        stable_policy.pop(field)
    stable_digest = digest_json(stable_identity)
    measurement = empty_measurement()
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "artifact": {},
            "validation": {
                "status": ResultStatus.OK.value,
                DIRECT_AGREEMENT_FIELD: [],
            },
            "resources": {},
            "provenance": {
                "runtime_identity": identity,
                "runtime_identity_sha256": digest_json(identity),
                "runtime_identity_stable_sha256": stable_digest,
                "runtime_identity_postflight_stable_sha256": stable_digest,
                "runtime_identity_postflight_loaded_module_origin_policy": deepcopy(
                    policy
                ),
                "runtime_identity_postflight_match": True,
            },
        }
    )
    return measurement


def _below_resolution_candidate_measurement() -> dict[str, object]:
    measurement = _candidate_measurement_with_runtime_postflight()
    measurement["execution_seconds_per_point"] = None
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance["source_revision"] = "a" * 40
    provenance["execution_timing"] = {
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
    return measurement


def test_candidate_compiled_zero_requires_below_resolution_provenance() -> None:
    measurement = _below_resolution_candidate_measurement()
    validate_measurement(measurement)

    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("execution_timing")
    with pytest.raises(ValueError, match="below-resolution provenance"):
        validate_measurement(measurement)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ratio_eligible", True),
        ("raw_seconds_per_point", 1.0e-9),
        ("source", "runtime_profile_core_evaluator_call_time"),
        ("compiled_direct_arena_active", False),
        ("native_profile_points_per_sample", None),
    ),
)
def test_below_resolution_execution_provenance_is_fail_closed(
    field: str,
    value: object,
) -> None:
    measurement = _below_resolution_candidate_measurement()
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    timing = provenance["execution_timing"]
    assert isinstance(timing, dict)
    timing[field] = value

    with pytest.raises(ValueError, match="below-resolution record"):
        validate_measurement(measurement)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "runtime_identity",
            None,
            "requires runtime_identity",
        ),
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
            "runtime_identity_postflight_loaded_module_origin_policy",
            {},
            "loaded-origin evidence",
        ),
        (
            "runtime_identity_postflight_match",
            False,
            "runtime_identity_postflight_match",
        ),
    ],
)
def test_successful_candidate_measurement_requires_runtime_postflight(
    field: str,
    value: object,
    match: str,
) -> None:
    measurement = _candidate_measurement_with_runtime_postflight()
    validate_measurement(measurement)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value
    with pytest.raises(ValueError, match=match):
        validate_measurement(measurement)


def test_candidate_runtime_postflight_cannot_lose_initial_origin() -> None:
    measurement = _candidate_measurement_with_runtime_postflight()
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    postflight = provenance["runtime_identity_postflight_loaded_module_origin_policy"]
    assert isinstance(postflight, dict)
    observations = postflight["observations"]
    assert isinstance(observations, list)
    observations[0] = {
        "module": "pyamplicol.replaced",
        "kind": "package-member",
        "root_index": 0,
        "path": "replaced.py",
        "size": 10,
        "sha256": "2" * 64,
    }
    postflight["observations_sha256"] = digest_json(observations)
    with pytest.raises(ValueError, match="lost a loaded-module origin"):
        validate_measurement(measurement)
