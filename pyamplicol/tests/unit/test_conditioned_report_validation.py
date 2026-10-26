# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.performance_report.agreements import (
    BUILTIN_UFO_RECURRENCE,
    DIRECT_AGREEMENT_V2_ABI,
    INDEPENDENT_AUTHORITY_ABI,
    INDEPENDENT_AUTHORITY_FIELD,
    independent_numerical_authorities,
    validate_direct_agreement_records,
)
from tools.performance_report.cache import _validate_conditioned_measurement_bindings
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.final_audit import (
    _stored_independent_pointwise_matches_baseline,
)
from tools.performance_report.measurement import (
    conditioned_measurement_comparison,
    reconcile_independent_authority,
)
from tools.performance_report.models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    MeasurementSpec,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.runner import (
    SelectorContract,
    pointwise_validation,
    resolved_sum_validation,
    validate_conditioned_comparison_record,
    validate_resolved_sum_validation_record,
)


def _source_binding(
    *,
    point_identity: str = "a" * 64,
    selector_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = selector_identity or {"cell_id": "candidate", "value_kind": "total"}
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "point_digest": point_identity,
        "selector_component_identity": identity,
        "selector_component_sha256": digest,
        "candidate_source_sha256": "b" * 64,
        "baseline_source_sha256": "c" * 64,
    }


def _cell() -> CellSpec:
    return CellSpec(
        dataset_id="test",
        process_key="test",
        process="d d~ > z",
        n_final=1,
        workload=Workload.CONTRACTED,
        measurement=MeasurementSpec(
            execution_mode=ExecutionMode.RECURRENCE,
            backend="jit",
            model=ModelKey.BUILTIN_SM,
            accuracy=Accuracy.FULL,
            jit_optimization_level=2,
        ),
    )


def test_conditioned_comparison_has_no_tiny_absolute_escape() -> None:
    failed = pointwise_validation(
        1.0e-16,
        0.0,
        comparison_binding=_source_binding(),
    )
    assert failed["status"] == "validation_failed"
    assert failed["comparison_scale"] == 1.0e-16
    assert failed["conditioned_residual"] == 1.0

    cancelled = pointwise_validation(
        1.0e-16,
        0.0,
        candidate_scale=1.0e-3,
        baseline_scale=1.0e-3,
        comparison_binding=_source_binding(),
    )
    assert cancelled["status"] == "ok"
    validate_conditioned_comparison_record(cancelled, require_binding=True)


def test_conditioned_zero_scale_and_scale_validation_are_fail_closed() -> None:
    exact = pointwise_validation(
        0.0,
        0.0,
        comparison_binding=_source_binding(),
    )
    assert exact["comparison_scale"] == 0.0
    assert exact["conditioned_residual"] == 0.0
    assert exact["error_bound"] == 0.0

    with pytest.raises(ValueError, match="smaller"):
        pointwise_validation(2.0, 2.0, candidate_scale=1.0)
    with pytest.raises(ValueError, match="finite"):
        pointwise_validation(1.0, 1.0, baseline_scale=float("inf"))


def test_legacy_comparison_reuse_rejects_floor_only_acceptance() -> None:
    floor_only = {
        "status": "ok",
        "candidate": 1.0e-16,
        "baseline": 0.0,
        "absolute_difference": 1.0e-16,
        "relative_difference": 1.0e-16 / 1.0e-300,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    with pytest.raises(ValueError, match="safely reusable"):
        validate_conditioned_comparison_record(floor_only, require_binding=False)

    safely_relative = {
        "status": "ok",
        "candidate": 1.0 + 1.0e-13,
        "baseline": 1.0,
        "absolute_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_difference": abs((1.0 + 1.0e-13) - 1.0),
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    validate_conditioned_comparison_record(safely_relative, require_binding=False)

    unbound_resolved = {
        "status": "ok",
        "maximum_absolute_difference": 1.0e-16,
        "maximum_relative_difference": 1.0,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    with pytest.raises(ValueError, match="floor-only or unbound"):
        validate_resolved_sum_validation_record(unbound_resolved)


def test_resolved_l1_scale_is_bound_and_tampering_is_rejected() -> None:
    class Runtime:
        def evaluate(self, _points, **_kwargs):
            return (1.0e-16,)

        def evaluate_resolved(self, _points, **_kwargs):
            return SimpleNamespace(
                values=(((1.0, -1.0 + 1.0e-16),),),
                helicity_ids=("h0",),
                color_ids=("c0", "c1"),
                total=lambda: (1.0e-16,),
            )

    record = resolved_sum_validation(
        Runtime(),
        (((1.0, 0.0, 0.0, 1.0),),),
        cell=_cell(),
        selector_contract=None,
    )
    assert record["status"] == "ok"
    point = record["points"][0]
    assert point["candidate_scale"] >= 1.0
    assert point["baseline_scale"] >= 1.0
    validate_resolved_sum_validation_record(record)

    point["comparison_binding"]["point_index"] = 1
    with pytest.raises(ValueError, match="binding differs"):
        validate_resolved_sum_validation_record(record)


def test_direct_agreement_v2_authenticates_conditioned_fields() -> None:
    comparison = pointwise_validation(
        1.0,
        1.0,
        comparison_binding=_source_binding(),
    )
    comparison.pop("abi")
    record = {
        "abi": DIRECT_AGREEMENT_V2_ABI,
        "edge_kind": BUILTIN_UFO_RECURRENCE,
        "value_kind": "matrix_element",
        "baseline_cell_id": "baseline",
        "candidate_cell_id": "candidate",
        **comparison,
    }
    validate_direct_agreement_records([record], expected_candidate_id="candidate")

    record["candidate_scale"] = 0.5
    with pytest.raises(ValueError, match="numerical record"):
        validate_direct_agreement_records([record], expected_candidate_id="candidate")

    legacy_floor = {
        "abi": "pyamplicol-report-direct-agreement-v1",
        "edge_kind": BUILTIN_UFO_RECURRENCE,
        "value_kind": "matrix_element",
        "baseline_cell_id": "baseline",
        "candidate_cell_id": "candidate",
        "status": "ok",
        "candidate": 1.0e-16,
        "baseline": 0.0,
        "absolute_difference": 1.0e-16,
        "relative_difference": 1.0e-16 / 1.0e-300,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
    }
    with pytest.raises(ValueError, match="numerical record"):
        validate_direct_agreement_records([legacy_floor])


def test_completed_unverified_candidate_can_be_compared_to_late_authority() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    candidate_value = 1.0 + 1.0e-13
    resolved_point = pointwise_validation(
        candidate_value,
        candidate_value,
        candidate_scale=candidate_value,
        baseline_scale=candidate_value,
        comparison_binding={
            "abi": "pyamplicol-report-resolved-component-scale-v1",
            "point_digest": "a" * 64,
            "helicity_ids": [],
            "color_flow_ids": [],
            "resolved_ordering_sha256": "c" * 64,
            "resolved_source_sha256": "b" * 64,
            "point_index": 0,
        },
    )
    resolved = {
        "abi": "pyamplicol-report-resolved-sum-validation-v2",
        "status": "ok",
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
        "maximum_conditioned_residual": 0.0,
        "relative_tolerance": 1.0e-12,
        "point_digest": "a" * 64,
        "helicity_ids": [],
        "color_flow_ids": [],
        "resolved_ordering_sha256": "c" * 64,
        "resolved_source_sha256": "b" * 64,
        "scale_source": "resolved-component-l1",
        "precision_digits": 16,
        "points": [resolved_point],
    }
    candidate = {
        "status": "unverified",
        "matrix_element": candidate_value,
        "selector_contract": None,
        "validation": {"resolved_sum": resolved},
        "provenance": {"report_momenta": [[[1.0, 0.0, 0.0, 1.0]]]},
    }
    authority = {
        "status": "ok",
        "matrix_element": 1.0,
        "selector_contract": None,
        "validation": {"point_digest": "a" * 64},
        "provenance": {"report_momenta": [[[1.0, 0.0, 0.0, 1.0]]]},
    }
    comparison = conditioned_measurement_comparison(cell, candidate, authority)
    assert comparison["status"] == "ok"
    assert comparison["candidate_scale_source"] == "resolved-component-l1"
    validate_conditioned_comparison_record(comparison, require_binding=True)
    candidate["validation"]["pointwise"] = comparison  # type: ignore[index]
    authority_cell = independent_numerical_authorities(cell)[0]
    assert _stored_independent_pointwise_matches_baseline(
        cell,
        candidate,
        authority_cell,
        authority,
    )
    comparison["comparison_binding"]["baseline_source_sha256"] = "f" * 64
    assert not _stored_independent_pointwise_matches_baseline(
        cell,
        candidate,
        authority_cell,
        authority,
    )
    candidate["validation"]["resolved_sum"]["point_digest"] = comparison[  # type: ignore[index]
        "comparison_binding"
    ]["point_digest"]
    candidate["validation"]["pointwise"] = comparison  # type: ignore[index]
    _validate_conditioned_measurement_bindings(
        candidate["validation"],  # type: ignore[arg-type]
        expected_cell=cell,
        selector_contract=None,
    )
    comparison["comparison_binding"]["candidate_source_sha256"] = "d" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="candidate scale source"):
        _validate_conditioned_measurement_bindings(
            candidate["validation"],  # type: ignore[arg-type]
            expected_cell=cell,
            selector_contract=None,
        )

    candidate["matrix_element"] = 0.5
    with pytest.raises(RuntimeError, match="differs from resolved-sum"):
        conditioned_measurement_comparison(cell, candidate, authority)


@pytest.mark.parametrize("authority_value", (1.0, 2.0))
def test_late_authority_reconciliation_promotes_or_fails(
    authority_value: float,
) -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    candidate = deepcopy(
        # Reuse the authenticated resolved-scale fixture built by the adjacent
        # test without relying on any campaign artifact.
        {
            "status": ResultStatus.UNVERIFIED.value,
            "matrix_element": 1.0,
            "selector_contract": None,
            "validation": {
                "point_digest": "a" * 64,
                INDEPENDENT_AUTHORITY_FIELD: {
                    "abi": INDEPENDENT_AUTHORITY_ABI,
                    "expected_cell_ids": [
                        item.cell_id
                        for item in independent_numerical_authorities(cell)
                    ],
                    "selected_cell_id": None,
                    "status": "unavailable",
                    "reason": "no-successful-independent-authority",
                    "same_artifact_diagnostics_are_authority": False,
                },
            },
            "failure": {
                "kind": "IndependentAuthorityUnavailable",
                "message": "no authority",
            },
        }
    )
    authority = {
        "status": ResultStatus.OK.value,
        "matrix_element": authority_value,
        "selector_contract": None,
        "validation": {"point_digest": "a" * 64},
        "provenance": {},
    }
    authority_id = independent_numerical_authorities(cell)[0].cell_id

    reconciled = reconcile_independent_authority(
        cell,
        candidate,
        authority,
        authority_cell_id=authority_id,
    )

    expected = (
        ResultStatus.OK.value
        if authority_value == 1.0
        else ResultStatus.VALIDATION_FAILED.value
    )
    assert reconciled["status"] == expected
    record = reconciled["validation"][INDEPENDENT_AUTHORITY_FIELD]
    assert record["selected_cell_id"] == authority_id
    assert record["status"] == (
        "verified" if authority_value == 1.0 else "mismatch"
    )


def test_late_lc_all_flow_authority_compares_only_common_component() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-lc-n4-dd-tt-jets-all-flow"
    )
    contract = SelectorContract(
        selected_color_flow_ids=("flow:1,2,3",),
        selected_color_words=((1, 2, 3),),
        all_flow_helicity_ids=("h:-1,+1,-1",),
        all_flow_source_helicities=((1, -1), (2, 1), (3, -1)),
        point_digest="a" * 64,
    ).as_dict()

    def component(cell_id: str, value: float) -> dict[str, object]:
        return {
            "abi": "pyamplicol-report-lc-common-component-v1",
            "cell_id": cell_id,
            "value": value,
            "point_digest": contract["point_digest"],
            "helicity_ids": contract["all_flow_helicity_ids"],
            "color_flow_ids": contract["selected_color_flow_ids"],
        }

    candidate = {
        "status": "unverified",
        "matrix_element": 99.0,
        "selector_contract": contract,
        "validation": {
            "lc_common_component": component(cell.cell_id, 2.0),
        },
    }
    authority_id = "reference-amplicol-lc-n4-dd-tt-jets-all-flow"
    authority = {
        "status": "ok",
        "matrix_element": 1.0,
        "selector_contract": contract,
        "validation": {
            "legacy_numerical_authority": {
                "abi": "pyamplicol-report-legacy-numerical-authority-v1",
                "source": "all-flow-selected-provider-replay",
            },
            "lc_common_component": component(authority_id, 2.0),
        },
    }
    comparison = conditioned_measurement_comparison(cell, candidate, authority)
    assert comparison["status"] == "ok"
    assert comparison["candidate"] == 2.0
    assert comparison["baseline"] == 2.0
    assert (
        comparison["comparison_binding"]["selector_component_identity"][  # type: ignore[index]
            "value_kind"
        ]
        == "lc_common_component"
    )

    authority["validation"]["lc_common_component"]["value"] = 3.0  # type: ignore[index]
    mismatch = conditioned_measurement_comparison(cell, candidate, authority)
    assert mismatch["status"] == "validation_failed"
