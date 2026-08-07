# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_ABI,
    INDEPENDENT_AUTHORITY_FIELD,
    incoming_agreement_edges,
    independent_numerical_authorities,
)
from tools.performance_report.arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    EMPTY_ARENA_PHASE_VECTOR_FIELDS,
    ZERO_ARENA_COUNTER_FIELDS,
    ZERO_ARENA_PHASE_TIME_FIELDS,
    ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS,
    build_arena_profile_evidence,
    digest_arena_profile_value,
)
from tools.performance_report.cache import (
    CACHE_SCHEMA_VERSION,
    REPORT_VERSION,
    _validate_unverified_direct_agreement_coverage,
    build_reset_caches,
    digest_json,
    empty_measurement,
    schema_document,
    validate_cache,
    validate_measurement,
)
from tools.performance_report.catalog import (
    REPORT_CATALOG,
    STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6,
    STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6_DESCRIPTION,
    STATIC_NA_ON_THE_FLY_COLOR_ACCURACY,
    STATIC_NA_ON_THE_FLY_COLOR_ACCURACY_DESCRIPTION,
    STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT,
)
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
    ZVariant,
)
from tools.performance_report.runner import pointwise_validation
from tools.performance_report.service import ReportPaths, ReportService


def test_matrix_catalog_has_requested_fifteen_datasets() -> None:
    datasets = REPORT_CATALOG.matrix_datasets

    assert len(datasets) == 15
    assert Counter(item.candidate.execution_mode for item in datasets) == {
        ExecutionMode.RECURRENCE: 6,
        ExecutionMode.COMPILED: 3,
        ExecutionMode.EAGER: 3,
        ExecutionMode.ON_THE_FLY: 3,
    }
    assert Counter(item.candidate.model for item in datasets) == {
        ModelKey.BUILTIN_SM: 12,
        ModelKey.UFO_SM: 3,
    }
    assert all(
        item.baseline.execution_mode is ExecutionMode.AMPLICOL
        for item in datasets
        if item.candidate.execution_mode
        in {ExecutionMode.RECURRENCE, ExecutionMode.ON_THE_FLY}
    )
    assert all(
        item.baseline.execution_mode is ExecutionMode.RECURRENCE
        for item in datasets
        if item.candidate.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    )


def test_required_agreement_evidence_bumps_report_cache_contract() -> None:
    assert CACHE_SCHEMA_VERSION == 5
    assert REPORT_VERSION == "0.3.0"


def test_packaged_report_cache_schema_is_canonical() -> None:
    packaged = (
        Path(__file__).resolve().parents[2]
        / "src/pyamplicol/_profiling_campaign/results/report-cache.schema.json"
    )
    assert json.loads(packaged.read_text(encoding="ascii")) == schema_document()


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
            ExecutionMode.ON_THE_FLY,
        )
    }

    assert by_mode == {
        ExecutionMode.RECURRENCE: {2},
        ExecutionMode.COMPILED: {3},
        ExecutionMode.EAGER: {2},
        ExecutionMode.ON_THE_FLY: {2},
    }


def test_lc_cells_have_two_runtime_workloads_and_contracted_cells_have_one() -> None:
    counts = Counter(
        (cell.measurement.accuracy, cell.workload)
        for cell in REPORT_CATALOG.matrix_cells()
    )

    assert counts[(Accuracy.LC, Workload.SELECTED_FLOW)] == 461
    assert counts[(Accuracy.LC, Workload.ALL_FLOW)] == 461
    assert counts[(Accuracy.NLC, Workload.CONTRACTED)] == 250
    assert counts[(Accuracy.FULL, Workload.CONTRACTED)] == 250
    assert all(
        cell.workload is Workload.CONTRACTED
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.accuracy is not Accuracy.LC
    )


def test_extended_lc_families_are_declared_through_n9() -> None:
    process_keys = {
        "gg_tt_jets",
        "gg_gluons",
        "dd_3q_lines",
        "dd_4q_lines",
        "dd_3q_identical_lines",
    }
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.n_final == 9
        and cell.process_key in process_keys
        and cell.measurement.accuracy is Accuracy.LC
    )

    assert len(cells) == 50
    assert {
        (
            cell.process_key,
            cell.measurement.execution_mode,
            cell.measurement.model,
            cell.workload,
        )
        for cell in cells
    } == {
        (process_key, mode, model, workload)
        for process_key in process_keys
        for mode, model in {
            (ExecutionMode.AMPLICOL, None),
            (ExecutionMode.RECURRENCE, ModelKey.BUILTIN_SM),
            (ExecutionMode.RECURRENCE, ModelKey.UFO_SM),
            (ExecutionMode.COMPILED, ModelKey.BUILTIN_SM),
            (ExecutionMode.EAGER, ModelKey.BUILTIN_SM),
        }
        for workload in {Workload.SELECTED_FLOW, Workload.ALL_FLOW}
    }
    assert {
        cell.process_key
        for cell in cells
        if REPORT_CATALOG.static_na_reason(cell) is not None
    } == {"dd_4q_lines"}


def test_n_le_four_matrix_smoke_has_495_logical_process_cells() -> None:
    cells = [cell for cell in REPORT_CATALOG.matrix_cells() if cell.n_final <= 4]
    logical = {(cell.dataset_id, cell.process_key, cell.n_final) for cell in cells}

    assert len(logical) == 495
    assert len(cells) == 660


def test_identical_quark_line_family_has_canonical_order_and_full_coverage() -> None:
    family = next(
        family for family in REPORT_CATALOG.process_families if family.identifier == 15
    )

    assert family.key == "dd_3q_identical_lines"
    assert family.label_tex == r"$d\bar d\to u\bar u\,u\bar u+(n-4)g$"
    assert family.minimum_n == 4
    assert family.maximum_n(Accuracy.LC) == 9
    assert family.maximum_n(Accuracy.NLC) == 6
    assert family.maximum_n(Accuracy.FULL) == 6
    assert family.include_3qqbar is True
    assert family.process(3) is None
    assert family.process(4) == "d d~ > u u~ u u~"
    assert family.process(8) == "d d~ > u u~ u u~ g g g g"

    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.process_key == family.key
    )
    assert len(cells) == 98
    assert Counter(REPORT_CATALOG.static_na_reason(cell) for cell in cells) == {
        None: 92,
        STATIC_NA_ON_THE_FLY_COLOR_ACCURACY: 6,
    }
    assert (
        REPORT_CATALOG.cell(
            "matrix-recurrence-builtin-sm-full-n6-dd-3q-identical-lines-contracted"
        ).process
        == "d d~ > u u~ u u~ g g"
    )


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


def test_four_line_report_keeps_legacy_display_without_legacy_dependency() -> None:
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_lc"
        and cell.process_key == "dd_4q_lines"
        and cell.n_final == 6
        and cell.workload is Workload.SELECTED_FLOW
    )
    compiled = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_4q_lines"
        and cell.n_final == 6
        and cell.workload is Workload.SELECTED_FLOW
    )

    display_baseline = REPORT_CATALOG.baseline_cell(recurrence)
    assert display_baseline is not None
    assert display_baseline.measurement.execution_mode is ExecutionMode.AMPLICOL
    assert REPORT_CATALOG.legacy_reference_available(recurrence) is False
    assert REPORT_CATALOG.static_na_reason(display_baseline) == (
        "original-amplicol-open-quark-line-limit"
    )
    assert REPORT_CATALOG.validation_baseline_cell(recurrence) is None

    compiled_baseline = REPORT_CATALOG.validation_baseline_cell(compiled)
    assert compiled_baseline is not None
    assert compiled_baseline.measurement.execution_mode is ExecutionMode.RECURRENCE

    three_line_reference = REPORT_CATALOG.cell(
        "reference-amplicol-full-n6-dd-3q-lines-contracted"
    )
    assert REPORT_CATALOG.legacy_reference_available(three_line_reference)
    assert REPORT_CATALOG.static_na_reason(three_line_reference) is None


def test_canonical_static_na_census_is_exact() -> None:
    static_na = {
        cell.cell_id: REPORT_CATALOG.static_na_reason(cell)
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    }

    expected = {
        cell_id: STATIC_NA_ORIGINAL_AMPLICOL_OPEN_QUARK_LINE_LIMIT
        for cell_id in {
            "reference-amplicol-lc-n6-dd-4q-lines-selected-flow",
            "reference-amplicol-lc-n6-dd-4q-lines-all-flow",
            "reference-amplicol-lc-n7-dd-4q-lines-selected-flow",
            "reference-amplicol-lc-n7-dd-4q-lines-all-flow",
            "reference-amplicol-lc-n8-dd-4q-lines-selected-flow",
            "reference-amplicol-lc-n8-dd-4q-lines-all-flow",
            "reference-amplicol-lc-n9-dd-4q-lines-selected-flow",
            "reference-amplicol-lc-n9-dd-4q-lines-all-flow",
            "reference-amplicol-nlc-n6-dd-4q-lines-contracted",
            "reference-amplicol-full-n6-dd-4q-lines-contracted",
        }
    }
    expected.update(
        {
            (
                f"z-{model}-n{n_final}-dd-z-jets-{variant}-{workload}"
            ): STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6
            for model in ("builtin-sm", "external-sm")
            for n_final in (7, 8, 9)
            for variant in ("asm-o3", "cpp-o3")
            for workload in ("selected-flow", "all-flow")
        }
    )
    expected.update(
        {
            cell.cell_id: STATIC_NA_ON_THE_FLY_COLOR_ACCURACY
            for cell in REPORT_CATALOG.matrix_cells()
            if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
            and cell.measurement.accuracy in {Accuracy.NLC, Accuracy.FULL}
        }
    )

    assert static_na == expected
    for cell_id, reason in expected.items():
        if reason == STATIC_NA_ON_THE_FLY_COLOR_ACCURACY:
            cell = REPORT_CATALOG.cell(cell_id)
            assert (
                REPORT_CATALOG.static_na_description(cell)
                == STATIC_NA_ON_THE_FLY_COLOR_ACCURACY_DESCRIPTION
            )
            continue
        if reason != STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6:
            continue
        cell = REPORT_CATALOG.cell(cell_id)
        assert (
            REPORT_CATALOG.static_na_description(cell)
            == STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6_DESCRIPTION
        )


def test_z_variant_generation_cap_contract_is_paired_and_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ZVariant(
            "bad",
            "bad",
            ExecutionMode.COMPILED,
            "asm",
            maximum_generation_n_final=0,
            static_na_reason_code="reason-v1",
        )
    with pytest.raises(ValueError, match="requires a static_na_reason_code"):
        ZVariant(
            "bad",
            "bad",
            ExecutionMode.COMPILED,
            "asm",
            maximum_generation_n_final=6,
        )
    with pytest.raises(ValueError, match="requires maximum_generation_n_final"):
        ZVariant(
            "bad",
            "bad",
            ExecutionMode.COMPILED,
            "asm",
            static_na_reason_code="reason-v1",
        )


def test_contracted_multi_quark_coverage_reaches_n6_in_every_mode() -> None:
    process_keys = {
        "dd_3q_lines",
        "dd_3q_identical_lines",
        "dd_4q_lines",
    }
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.process_key in process_keys
        and cell.n_final == 6
        and cell.measurement.accuracy in {Accuracy.NLC, Accuracy.FULL}
    )

    assert len(cells) == 36
    assert {
        (
            cell.process_key,
            cell.measurement.accuracy,
            cell.measurement.execution_mode,
            cell.measurement.model,
        )
        for cell in cells
    } == {
        (process_key, accuracy, mode, model)
        for process_key in process_keys
        for accuracy in {Accuracy.NLC, Accuracy.FULL}
        for mode, model in {
            (ExecutionMode.AMPLICOL, None),
            (ExecutionMode.RECURRENCE, ModelKey.BUILTIN_SM),
            (ExecutionMode.RECURRENCE, ModelKey.UFO_SM),
            (ExecutionMode.COMPILED, ModelKey.BUILTIN_SM),
            (ExecutionMode.EAGER, ModelKey.BUILTIN_SM),
            (ExecutionMode.ON_THE_FLY, ModelKey.BUILTIN_SM),
        }
    }
    assert (
        REPORT_CATALOG.cell(
            "matrix-recurrence-builtin-sm-full-n6-dd-3q-lines-contracted"
        ).process
        == "d d~ > u u~ s s~ g g"
    )


def test_contracted_n6_catalog_impact_is_scoped_to_multi_quark_families() -> None:
    process_keys = {
        "dd_3q_lines",
        "dd_3q_identical_lines",
        "dd_4q_lines",
    }
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.process_key in process_keys
        and cell.n_final == 6
        and cell.measurement.accuracy is not Accuracy.LC
    )

    assert len(REPORT_CATALOG.measurement_cells()) == 1962
    assert len(REPORT_CATALOG.matrix_cells()) == 1422
    assert len(REPORT_CATALOG.reference_cells()) == 314
    assert len(cells) == 36
    assert {cell.process_key for cell in cells} == process_keys


def test_measurement_cells_reuses_one_immutable_catalog_materialization() -> None:
    fresh = replace(REPORT_CATALOG)
    cold_repr = repr(fresh)

    assert fresh._measurement_cells_cache is None

    first = fresh.measurement_cells()
    second = fresh.measurement_cells()

    assert second is first
    assert fresh._measurement_cells_cache is first
    assert isinstance(first, tuple)
    assert repr(fresh) == cold_repr
    assert "_measurement_cells_cache" not in repr(fresh)


def test_contracted_n6_rows_are_canonical_reset_cache_entries() -> None:
    caches = build_reset_caches()
    payload = caches["matrix_recurrence_builtin_sm_full.json"]
    expected = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_full"
    )

    validate_cache(payload, expected_cells=expected)
    entries = {
        entry["cell_id"]: entry for entry in payload["entries"] if entry["n_final"] == 6
    }
    assert set(entries) == {
        "matrix-recurrence-builtin-sm-full-n6-dd-3q-lines-contracted",
        ("matrix-recurrence-builtin-sm-full-n6-dd-3q-identical-lines-contracted"),
        "matrix-recurrence-builtin-sm-full-n6-dd-4q-lines-contracted",
    }
    assert all(
        entry["measurement"] == empty_measurement() for entry in entries.values()
    )


@pytest.mark.parametrize(
    "process_key",
    ("dd_3q_lines", "dd_3q_identical_lines"),
)
def test_three_line_reports_retain_the_original_amplicol_oracle(
    process_key: str,
) -> None:
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.dataset_id == "matrix_recurrence_builtin_sm_lc"
        and cell.process_key == process_key
        and cell.n_final == 4
        and cell.workload is Workload.SELECTED_FLOW
    )

    baseline = REPORT_CATALOG.validation_baseline_cell(recurrence)
    assert REPORT_CATALOG.legacy_reference_available(recurrence) is True
    assert baseline is not None
    assert baseline.measurement.execution_mode is ExecutionMode.AMPLICOL


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

    with pytest.raises(ValueError, match="validation status"):
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


def _unverified_candidate_measurement(cell: CellSpec) -> dict[str, object]:
    measurement = _candidate_measurement_with_runtime_postflight()
    point_identity = "a" * 64
    ordering = "b" * 64

    def resolved_record(precision: int, source: str) -> dict[str, object]:
        record = pointwise_validation(
            1.0,
            1.0,
            candidate_scale=1.0,
            baseline_scale=1.0,
            comparison_binding={
                "abi": "pyamplicol-report-resolved-component-scale-v1",
                "point_digest": point_identity,
                "helicity_ids": [],
                "color_flow_ids": [],
                "resolved_ordering_sha256": ordering,
                "resolved_source_sha256": source,
                "point_index": 0,
            },
        )
        return {
            "abi": "pyamplicol-report-resolved-sum-validation-v2",
            "status": ResultStatus.OK.value,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "maximum_conditioned_residual": 0.0,
            "relative_tolerance": 1.0e-12,
            "point_digest": point_identity,
            "helicity_ids": [],
            "color_flow_ids": [],
            "resolved_ordering_sha256": ordering,
            "resolved_source_sha256": source,
            "scale_source": "resolved-component-l1",
            "precision_digits": precision,
            "points": [record],
        }

    binary64 = resolved_record(16, "c" * 64)
    p32 = resolved_record(32, "d" * 64)
    selector_identity = {
        "cell_id": cell.cell_id,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "selector_contract": None,
        "value_kind": "matrix-element-p16-versus-p32",
    }
    high_precision = pointwise_validation(
        1.0,
        1.0,
        candidate_scale=1.0,
        baseline_scale=1.0,
        candidate_scale_source="resolved-component-l1-binary64",
        baseline_scale_source="resolved-component-l1-p32",
        comparison_binding={
            "point_digest": point_identity,
            "selector_component_identity": selector_identity,
            "selector_component_sha256": digest_json(selector_identity),
            "candidate_source_sha256": "c" * 64,
            "baseline_source_sha256": "d" * 64,
        },
    )
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    validation.update(
        {
            "status": ResultStatus.UNVERIFIED.value,
            "resolved_sum": binary64,
            "high_precision_resolved_sum": p32,
            "high_precision": high_precision,
            "precision_diagnostic": {
                "abi": ("pyamplicol-report-validation-failure-precision-diagnostic-v2"),
                "status": "unavailable",
                "promotes_measurement": False,
                "error": {"kind": "FixtureUnavailable", "message": "fixture"},
            },
            INDEPENDENT_AUTHORITY_FIELD: {
                "abi": INDEPENDENT_AUTHORITY_ABI,
                "expected_cell_ids": [
                    authority.cell_id
                    for authority in independent_numerical_authorities(cell)
                ],
                "selected_cell_id": None,
                "status": "unavailable",
                "reason": "no-successful-independent-authority",
                "same_artifact_diagnostics_are_authority": False,
            },
        }
    )
    measurement["status"] = ResultStatus.UNVERIFIED.value
    measurement["failure"] = {
        "kind": "IndependentAuthorityUnavailable",
        "message": "no successful independent numerical authority is available",
    }
    return measurement


def test_unverified_measurement_contract_is_strict_and_catalog_bound() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    measurement = _unverified_candidate_measurement(cell)
    validate_measurement(measurement, expected_cell=cell)

    validation = measurement["validation"]
    assert isinstance(validation, dict)
    authority = validation[INDEPENDENT_AUTHORITY_FIELD]
    assert isinstance(authority, dict)
    canonical = list(authority["expected_cell_ids"])
    for malformed in (
        canonical[:-1],
        list(reversed(canonical)),
        [*canonical, "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"],
    ):
        tampered = deepcopy(measurement)
        tampered["validation"][INDEPENDENT_AUTHORITY_FIELD][  # type: ignore[index]
            "expected_cell_ids"
        ] = malformed
        with pytest.raises(ValueError, match="independent-authority"):
            validate_measurement(tampered, expected_cell=cell)

    tampered = deepcopy(measurement)
    tampered["validation"]["precision_diagnostic"][  # type: ignore[index]
        "promotes_measurement"
    ] = True
    with pytest.raises(ValueError, match="precision diagnostic"):
        validate_measurement(tampered, expected_cell=cell)


@pytest.mark.parametrize(
    "cell_id",
    (
        "scalar-contact-n2-scalar-contact-contracted",
        "scalar-gravity-n2-scalar-gravity-contracted",
    ),
)
def test_standalone_scalar_cell_rejects_inapplicable_unverified_authority(
    cell_id: str,
) -> None:
    cell = REPORT_CATALOG.cell(cell_id)
    measurement = _unverified_candidate_measurement(cell)

    with pytest.raises(ValueError, match="not applicable"):
        validate_measurement(measurement, expected_cell=cell)


@pytest.mark.parametrize(
    "cell_id",
    (
        "scalar-contact-n2-scalar-contact-contracted",
        "scalar-gravity-n2-scalar-gravity-contracted",
    ),
)
def test_standalone_scalar_ok_requires_strict_internal_p32_contract(
    cell_id: str,
) -> None:
    cell = REPORT_CATALOG.cell(cell_id)
    measurement = _unverified_candidate_measurement(cell)
    measurement["status"] = ResultStatus.OK.value
    measurement["failure"] = None
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    validation["status"] = ResultStatus.OK.value
    validation.pop("precision_diagnostic")
    validation.pop(INDEPENDENT_AUTHORITY_FIELD)

    validate_measurement(measurement, expected_cell=cell)
    validation.pop("high_precision")
    with pytest.raises(ValueError, match="high_precision"):
        validate_measurement(measurement, expected_cell=cell)


@pytest.mark.parametrize(
    "mutation",
    (
        "point",
        "ordering",
        "binary-precision",
        "matrix-value",
        "p32-value",
    ),
)
def test_standalone_scalar_ok_rejects_cross_record_p32_tampering(
    mutation: str,
) -> None:
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    measurement = _unverified_candidate_measurement(cell)
    measurement["status"] = ResultStatus.OK.value
    measurement["failure"] = None
    validation = measurement["validation"]
    assert isinstance(validation, dict)
    validation["status"] = ResultStatus.OK.value
    validation.pop("precision_diagnostic")
    validation.pop(INDEPENDENT_AUTHORITY_FIELD)
    resolved = validation["resolved_sum"]
    high_resolved = validation["high_precision_resolved_sum"]
    assert isinstance(resolved, dict)
    assert isinstance(high_resolved, dict)
    high_points = high_resolved["points"]
    assert isinstance(high_points, list)
    high_point = high_points[0]
    assert isinstance(high_point, dict)
    high_binding = high_point["comparison_binding"]
    assert isinstance(high_binding, dict)

    if mutation == "point":
        high_resolved["point_digest"] = "e" * 64
        high_binding["point_digest"] = "e" * 64
    elif mutation == "ordering":
        high_resolved["resolved_ordering_sha256"] = "e" * 64
        high_binding["resolved_ordering_sha256"] = "e" * 64
    elif mutation == "binary-precision":
        resolved["precision_digits"] = 17
    elif mutation == "matrix-value":
        measurement["matrix_element"] = 2.0
    else:
        high_points[0] = pointwise_validation(
            2.0,
            2.0,
            candidate_scale=2.0,
            baseline_scale=2.0,
            candidate_scale_source=str(high_point["candidate_scale_source"]),
            baseline_scale_source=str(high_point["baseline_scale_source"]),
            comparison_binding=high_binding,
        )

    with pytest.raises(ValueError, match="standalone"):
        validate_measurement(measurement, expected_cell=cell)


def test_unverified_presentation_overlay_is_attempt_bound_and_json_isolated(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    docs = repo / "docs" / "performance_reports" / "manual"
    docs.mkdir(parents=True)
    service = ReportService(
        ReportPaths.from_repo(
            repo,
            docs_dir=docs,
            artifact_root=tmp_path / "campaign-artifacts",
            coordination_root=tmp_path / "campaign-artifacts" / "coordination",
        )
    )
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    revision = "a" * 40
    measurement = _unverified_candidate_measurement(cell)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance.update(
        {
            "report_source_revision": revision,
            "manual_campaign": {
                "cell_identity": {
                    "cell_id": cell.cell_id,
                    "dataset_id": cell.dataset_id,
                    "process_key": cell.process_key,
                    "process": cell.process,
                    "n_final": cell.n_final,
                    "workload": cell.workload.value,
                    "execution_mode": cell.measurement.execution_mode.value,
                    "model": cell.measurement.model.value,
                    "accuracy": cell.measurement.accuracy.value,
                    "backend": cell.measurement.backend,
                    "variant": cell.variant,
                }
            },
        }
    )
    validate_measurement(measurement, expected_cell=cell)

    attempt = service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE)
    attempt.write_json("worker-result.json", measurement)
    attempt.mark_failed(
        "independent authority unavailable",
        artifact_paths=("worker-result.json",),
    )
    outcome = manual_campaign.LightweightPresentationOutcome(
        profile="campaign-local-v1",
        cell_id=cell.cell_id,
        source_revision=revision,
        campaign_invocation_id="fixture",
        attempt_id=attempt.attempt_id,
        status=ResultStatus.UNVERIFIED.value,
        label="unverified",
        completed_at_ns=1,
    )

    publication_caches, publication_merged = (
        manual_campaign._merge_lightweight_snapshot(service, {})
    )
    render_caches, render_merged = manual_campaign._merge_lightweight_snapshot(
        service,
        {},
        {cell.cell_id: outcome},
    )
    assert publication_merged == 0
    assert render_merged == 1

    def cell_measurement(
        caches: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        payload = caches[f"{cell.dataset_id}.json"]
        entries = payload["entries"]
        assert isinstance(entries, list)
        entry = next(
            item
            for item in entries
            if isinstance(item, dict) and item.get("cell_id") == cell.cell_id
        )
        value = entry["measurement"]
        assert isinstance(value, dict)
        return value

    assert cell_measurement(render_caches)["status"] == ResultStatus.UNVERIFIED.value
    assert cell_measurement(render_caches)["generation_seconds"] == 1.0
    assert cell_measurement(publication_caches)["status"] == (
        ResultStatus.NOT_AVAILABLE.value
    )

    tables = service._render_tables(render_caches)
    assert (
        "unverified"
        in tables["result_matrix_compiled_builtin_sm_full_table.tex"].lower()
    )
    service._snapshot_files(publication_caches, tables)
    installed_payloads = tuple(
        path
        for path in service.paths.results_dir.glob("*.json")
        if path.name != "report-cache.schema.json"
    )
    assert installed_payloads
    assert all(
        '"status":"unverified"'
        not in json.dumps(
            json.loads(path.read_text(encoding="ascii")),
            separators=(",", ":"),
        )
        for path in installed_payloads
    )

    wrong_source = replace(outcome, source_revision="b" * 40)
    wrong_attempt = replace(
        outcome,
        attempt_id="00000000-0000-4000-8000-000000000000",
    )
    wrong_identity_measurement = deepcopy(measurement)
    wrong_identity_provenance = wrong_identity_measurement["provenance"]
    assert isinstance(wrong_identity_provenance, dict)
    wrong_identity_manual = wrong_identity_provenance["manual_campaign"]
    assert isinstance(wrong_identity_manual, dict)
    wrong_identity = wrong_identity_manual["cell_identity"]
    assert isinstance(wrong_identity, dict)
    wrong_identity["cell_id"] = "different-cell"
    wrong_identity_attempt = service.store.new_attempt(
        cell.cell_id,
        ArtifactPolicy.REGENERATE,
    )
    wrong_identity_attempt.write_json(
        "worker-result.json",
        wrong_identity_measurement,
    )
    wrong_identity_attempt.mark_failed(
        "independent authority unavailable",
        artifact_paths=("worker-result.json",),
    )
    wrong_cell = replace(outcome, attempt_id=wrong_identity_attempt.attempt_id)
    for malformed in (wrong_source, wrong_attempt, wrong_cell):
        fallback = manual_campaign._presentation_measurement(
            service,
            cell,
            malformed,
        )
        validate_measurement(fallback, expected_cell=cell)
        assert fallback["status"] == ResultStatus.FAILED.value
        assert fallback["generation_seconds"] is None
        failure = fallback["failure"]
        assert isinstance(failure, dict)
        assert failure["kind"] == "ManualCampaignOutcome:unverified"


def test_unverified_measurement_requires_every_hard_direct_agreement() -> None:
    cell = REPORT_CATALOG.cell("matrix-compiled-builtin-sm-lc-n1-dd-z-jets-all-flow")
    required = next(edge for edge in incoming_agreement_edges(cell) if edge.required)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        _validate_unverified_direct_agreement_coverage(
            {DIRECT_AGREEMENT_FIELD: []},
            expected_cell=cell,
            catalog=REPORT_CATALOG,
        )
    _validate_unverified_direct_agreement_coverage(
        {
            DIRECT_AGREEMENT_FIELD: [
                {
                    "edge_kind": required.kind,
                    "baseline_cell_id": required.baseline.cell_id,
                    "candidate_cell_id": cell.cell_id,
                    "status": ResultStatus.OK.value,
                }
            ]
        },
        expected_cell=cell,
        catalog=REPORT_CATALOG,
    )


def _arena_unavailable_candidate_measurement() -> dict[str, object]:
    measurement = _candidate_measurement_with_runtime_postflight()
    measurement["execution_seconds_per_point"] = None
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance["source_revision"] = "a" * 40
    raw_profile = {
        "execution_mode": "compiled",
        "profile_boundary": ARENA_PROFILE_BOUNDARY,
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
        "evaluator_timing_available": False,
        "points": 128,
        "wall_time_s": 128 * 1.1e-6,
        "orchestration_time_s": 128 * 1.1e-6,
        **{field: 0 for field in ZERO_ARENA_COUNTER_FIELDS},
        **{field: 0.0 for field in ZERO_ARENA_PHASE_TIME_FIELDS},
        **{field: [] for field in EMPTY_ARENA_PHASE_VECTOR_FIELDS},
        **{field: 0 for field in ZERO_COMPILED_BOUNDARY_COUNTER_FIELDS},
        "compiled_direct_arena_engine_count": 1,
        "compiled_direct_arena_call_count": 128,
        "evaluator_backend_call_count": 128,
    }
    arena_evidence = build_arena_profile_evidence(
        [raw_profile] * 5,
        execution_mode="compiled",
        repetitions_per_profile=1,
        batch_size=128,
    )
    provenance["arena_profile_evidence"] = arena_evidence
    provenance["execution_timing"] = {
        "abi": "pyamplicol-report-arena-execution-timing-v2",
        "status": "unavailable",
        "ratio_eligible": False,
        "raw_seconds_per_point": None,
        "sample_count": 5,
        "native_profile_points_per_sample": 128,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "sample_contract": ("paired_unprofiled_headline_profiled_attribution_v1"),
        "profile_protocol": "arena",
        "profile_sample_pass": "runtime._profile_arena_repeated",
        "profile_boundary": (
            "warmed-direct-arena-borrowed-input-preallocated-output-v1"
        ),
        "borrowed_flat_input": True,
        "preallocated_output": True,
        "phase_timing_scope": "coarse-arena-boundary-only-v1",
        "evaluator_timing_available": False,
        "paired_with_headline": True,
        "identical_batch": True,
        "identical_repetitions": True,
        "execution_mode": "compiled",
        "warmed_boundary_wall_seconds_per_point": 1.1e-6,
        "arena_profile_evidence_sha256": digest_arena_profile_value(arena_evidence),
    }
    return measurement


def test_candidate_null_execution_requires_authenticated_arena_provenance() -> None:
    measurement = _arena_unavailable_candidate_measurement()
    validate_measurement(measurement)

    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("execution_timing")
    with pytest.raises(ValueError, match="authenticated Arena provenance"):
        validate_measurement(measurement)


def test_future_evaluator_total_provenance_is_optional_and_fail_closed() -> None:
    measurement = _arena_unavailable_candidate_measurement()
    validate_measurement(measurement)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    provenance["evaluator_total_timing"] = {
        "abi": "pyamplicol-report-evaluator-total-timing-v1",
        "status": "measured",
        "ratio_eligible": False,
        "raw_seconds_per_point": 1.0e-6,
        "source": "runtime._benchmark_f64_wall_time.accumulated",
        "execution_mode": "compiled",
        "sample_contract": ("accumulated-repeated-warmed-evaluator-total-v1"),
        "sample_count": 5,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "points_per_sample": 128,
        "measured_point_count": 640,
        "accumulated_seconds": 6.4e-4,
    }
    validate_measurement(measurement)

    total = provenance["evaluator_total_timing"]
    assert isinstance(total, dict)
    total["accumulated_seconds"] = 1.0
    with pytest.raises(ValueError, match="evaluator_total_timing"):
        validate_measurement(measurement)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ratio_eligible", True),
        ("raw_seconds_per_point", 0.0),
        ("profile_protocol", "frozen-pre-arena"),
        ("profile_sample_pass", "runtime.profile_repeated"),
        ("borrowed_flat_input", False),
        ("evaluator_timing_available", True),
        ("native_profile_points_per_sample", None),
        ("warmed_boundary_wall_seconds_per_point", 0.0),
    ),
)
def test_arena_unavailable_execution_provenance_is_fail_closed(
    field: str,
    value: object,
) -> None:
    measurement = _arena_unavailable_candidate_measurement()
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    timing = provenance["execution_timing"]
    assert isinstance(timing, dict)
    timing[field] = value

    with pytest.raises(ValueError, match="unavailable-attribution record"):
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
