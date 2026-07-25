from __future__ import annotations

from collections import Counter

import pytest

from tools.performance_report.cache import (
    build_reset_caches,
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
    cells = [
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.n_final <= 4
    ]
    logical = {
        (cell.dataset_id, cell.process_key, cell.n_final)
        for cell in cells
    }

    assert len(logical) == 384
    assert len(cells) == 512


def test_z_catalog_adds_recurrence_and_corrects_eager_label() -> None:
    variants = {variant.key: variant for variant in REPORT_CATALOG.z_variants}

    assert variants["eager_jit_o2"].label == "eager-DAG JIT O2"
    assert variants["recurrence_jit_o2"].label == "recurrence JIT O2"
    assert variants["recurrence_jit_o2"].jit_optimization_level == 2
    assert len(
        [
            cell
            for cell in REPORT_CATALOG.z_cells()
            if cell.variant == "recurrence_jit_o2" and cell.n_final <= 4
        ]
    ) == 16


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


def test_reset_caches_are_canonical_and_cover_every_measurement_cell() -> None:
    caches = build_reset_caches()
    cells_by_dataset: dict[str, list[object]] = {}
    for cell in REPORT_CATALOG.measurement_cells():
        cells_by_dataset.setdefault(cell.dataset_id, []).append(cell)

    assert set(caches) == {
        f"{dataset_id}.json" for dataset_id in cells_by_dataset
    }
    for name, payload in caches.items():
        dataset_id = name.removesuffix(".json")
        validate_cache(payload, expected_cells=cells_by_dataset[dataset_id])
        assert all(
            entry["measurement"] == empty_measurement()
            for entry in payload["entries"]
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
