# SPDX-License-Identifier: 0BSD
"""One-cell measurement orchestration over the public pyAmpliCol Python API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import ReportPaths

from .agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_ABI,
    INDEPENDENT_AUTHORITY_FIELD,
    LC_COMMON_COMPONENT_FIELD,
    evaluate_lc_common_component,
    independent_numerical_authorities,
    requires_independent_numerical_authority,
    validate_lc_common_component,
)
from .cache import empty_measurement
from .catalog import REPORT_CATALOG, ReportCatalog
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .phase_state import WorkerPhaseReporter
from .runner import (
    INDEPENDENT_RELATIVE_TOLERANCE,
    RELATIVE_TOLERANCE,
    GeneratedArtifact,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    _real_nonnegative,
    _selector_kwargs,
    generate_artifact,
    point_digest,
    pointwise_validation,
    profile_runtime,
    provenance_payload,
    resolved_sum_validation,
    runtime_identity_payload,
    runtime_validation_points,
    validate_artifact_contract,
    validate_resolved_sum_validation_record,
)
from .source_identity import (
    ReportSourceIdentityError,
    inspect_report_source,
    require_eligible_report_source,
)


def shared_validation_points(process: str) -> object:
    from pyamplicol.models.builtin.validation import generic_validation_point

    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(process)
        ),
    )


def _reproduction_momenta(points: object) -> object:
    """Return the exact measured points in portable public-CLI JSON form."""

    to_list = getattr(points, "tolist", None)
    payload = to_list() if callable(to_list) else points
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RunnerError(
            "report profiling points cannot be materialized for reproduction"
        ) from error
    return json.loads(encoded)


def source_revision(repo_root: Path, *, require_clean: bool = False) -> str:
    try:
        identity = (
            require_eligible_report_source(repo_root)
            if require_clean
            else inspect_report_source(repo_root)
        )
    except ReportSourceIdentityError as error:
        if require_clean:
            raise RunnerError(
                "report source worktree contains changes outside generated "
                f"report outputs: {error}"
            ) from error
        return "unknown"
    return identity.revision


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_selector_contract(
    baseline: Mapping[str, object] | None,
) -> SelectorContract | None:
    if baseline is None:
        return None
    raw = baseline.get("selector_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RunnerError("baseline selector_contract must be an object")
    return SelectorContract.from_mapping(raw)


def _measurement_selector_contract(
    cell: CellSpec,
    baseline: Mapping[str, object] | None,
    selector_provider: Mapping[str, object] | None,
) -> SelectorContract | None:
    """Choose one canonical LC selector before candidate work starts."""

    if cell.measurement.accuracy is not Accuracy.LC:
        return None
    baseline_contract = _baseline_selector_contract(baseline)
    provider_contract: SelectorContract | None = None
    if selector_provider is not None:
        if selector_provider.get("status") != ResultStatus.OK.value:
            raise RunnerError("LC selector provider is not a successful measurement")
        try:
            provider_contract = _baseline_selector_contract(selector_provider)
        except (TypeError, ValueError) as error:
            raise RunnerError("LC selector provider has an invalid contract") from error
        if provider_contract is None:
            raise RunnerError("LC selector provider has no selector_contract")
    if (
        baseline_contract is not None
        and provider_contract is not None
        and baseline_contract != provider_contract
    ):
        raise RunnerError(
            "LC validation baseline and selected-flow selector provider disagree"
        )
    if baseline_contract is not None:
        return baseline_contract
    return provider_contract


def _baseline_matrix_element(
    baseline: Mapping[str, object] | None,
) -> float | None:
    if baseline is None:
        return None
    if baseline.get("status") != ResultStatus.OK.value:
        raise RunnerError("candidate baseline is not a valid completed measurement")
    value = baseline.get("matrix_element")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError("candidate baseline has no matrix element")
    return float(value)


_LEGACY_NUMERICAL_AUTHORITY_ABI = "pyamplicol-report-legacy-numerical-authority-v1"
_LEGACY_NUMERICAL_AUTHORITY_FIELD = "legacy_numerical_authority"
_LEGACY_ALL_FLOW_COMPONENT_AUTHORITY_SOURCE = "all-flow-selected-provider-replay"


def _independent_authority_record(
    expected_cell_ids: Sequence[str],
    *,
    selected_cell_id: str | None,
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "abi": INDEPENDENT_AUTHORITY_ABI,
        "expected_cell_ids": list(expected_cell_ids),
        "selected_cell_id": selected_cell_id,
        "status": status,
        "reason": reason,
        "same_artifact_diagnostics_are_authority": False,
    }


def baseline_uses_lc_common_component_authority(
    cell: CellSpec,
    baseline: Mapping[str, object] | None,
) -> bool:
    """Whether an LC all-flow baseline authenticates only its shared component."""

    if (
        baseline is None
        or cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
    ):
        return False
    validation = baseline.get("validation")
    authority = (
        validation.get(_LEGACY_NUMERICAL_AUTHORITY_FIELD)
        if isinstance(validation, Mapping)
        else None
    )
    if not (
        isinstance(authority, Mapping)
        and authority.get("abi") == _LEGACY_NUMERICAL_AUTHORITY_ABI
        and authority.get("source") == _LEGACY_ALL_FLOW_COMPONENT_AUTHORITY_SOURCE
    ):
        return False
    if baseline.get("status") != ResultStatus.OK.value:
        raise RunnerError("LC all-flow component authority is not successful")
    try:
        validate_lc_common_component(
            validation.get(LC_COMMON_COMPONENT_FIELD),
            selector_contract=baseline.get("selector_contract"),
        )
    except ValueError as error:
        raise RunnerError(
            "LC all-flow component authority is not bound to its selector"
        ) from error
    return True


def _attach_baseline_validation(
    cell: CellSpec,
    validation: dict[str, object],
    *,
    candidate_matrix_element: float,
    baseline: Mapping[str, object] | None,
    points: object = None,
    contract: SelectorContract | None = None,
) -> None:
    if baseline_uses_lc_common_component_authority(cell, baseline):
        # The worker's direct-agreement edge compares the authenticated shared
        # component.  Direct imode2's aggregate is diagnostic, not authority.
        return
    baseline_value = _baseline_matrix_element(baseline)
    if baseline_value is not None:
        resolved_sum = validation.get("resolved_sum")
        if not isinstance(resolved_sum, Mapping):
            raise RunnerError("candidate resolved-sum scale evidence is unavailable")
        point_records = resolved_sum.get("points")
        if not isinstance(point_records, list) or len(point_records) != 1:
            raise RunnerError("candidate comparison requires one resolved point")
        point_record = point_records[0]
        if not isinstance(point_record, Mapping):
            raise RunnerError("candidate resolved scale record is unavailable")
        candidate_scale = point_record.get("candidate_scale")
        candidate_source = resolved_sum.get("resolved_source_sha256")
        if (
            isinstance(candidate_scale, bool)
            or not isinstance(candidate_scale, (int, float))
            or not isinstance(candidate_source, str)
        ):
            raise RunnerError("candidate resolved scale evidence is invalid")
        baseline_source = _comparison_source_digest(
            {
                "matrix_element": baseline_value,
                "selector_contract": baseline.get("selector_contract"),
                "validation": baseline.get("validation"),
                "provenance": baseline.get("provenance"),
            }
        )
        validation["pointwise"] = pointwise_validation(
            candidate_matrix_element,
            baseline_value,
            relative_tolerance=_pointwise_tolerance(cell),
            candidate_scale=float(candidate_scale),
            baseline_scale=abs(baseline_value),
            candidate_scale_source="resolved-component-l1",
            baseline_scale_source="authority-value-magnitude",
            comparison_binding=_conditioned_binding(
                cell,
                points,
                contract,
                candidate_source_sha256=candidate_source,
                baseline_source_sha256=baseline_source,
                value_kind="matrix-element",
            ),
        )


def _stored_validation_point_digest(measurement: Mapping[str, object]) -> str:
    selector = measurement.get("selector_contract")
    if isinstance(selector, Mapping):
        raw = selector.get("point_digest")
        if isinstance(raw, str):
            return raw
    validation = measurement.get("validation")
    if isinstance(validation, Mapping):
        raw = validation.get("point_digest")
        if isinstance(raw, str):
            return raw
        resolved = validation.get("resolved_sum")
        if isinstance(resolved, Mapping):
            try:
                validate_resolved_sum_validation_record(resolved)
            except ValueError as error:
                raise RunnerError(
                    "measurement resolved-sum point identity is invalid"
                ) from error
            raw = resolved.get("point_digest")
            if isinstance(raw, str):
                return raw
    provenance = measurement.get("provenance")
    if isinstance(provenance, Mapping) and "report_momenta" in provenance:
        return point_digest(provenance["report_momenta"])
    raise RunnerError("measurement has no validation point identity")


def _stored_conditioned_scale(
    measurement: Mapping[str, object],
    value: float,
) -> tuple[float, str, str]:
    validation = measurement.get("validation")
    if isinstance(validation, Mapping):
        resolved = validation.get("resolved_sum")
        if isinstance(resolved, Mapping):
            try:
                validate_resolved_sum_validation_record(resolved)
            except ValueError as error:
                raise RunnerError(
                    "measurement resolved-sum scale evidence is invalid"
                ) from error
            records = resolved.get("points")
            source = resolved.get("resolved_source_sha256")
            if (
                isinstance(records, list)
                and len(records) == 1
                and isinstance(records[0], Mapping)
                and isinstance(source, str)
            ):
                raw_scale = records[0].get("candidate_scale")
                raw_candidate = records[0].get("candidate")
                if (
                    not isinstance(raw_scale, bool)
                    and isinstance(raw_scale, (int, float))
                    and math.isfinite(float(raw_scale))
                    and float(raw_scale) >= abs(value)
                    and not isinstance(raw_candidate, bool)
                    and isinstance(raw_candidate, (int, float))
                    and float(raw_candidate) == value
                ):
                    return float(raw_scale), "resolved-component-l1", source
                raise RunnerError(
                    "measurement matrix element differs from resolved-sum evidence"
                )
    source = _comparison_source_digest(
        {
            "matrix_element": value,
            "selector_contract": measurement.get("selector_contract"),
            "validation": validation,
            "provenance": measurement.get("provenance"),
        }
    )
    return abs(value), "authority-value-magnitude", source


def _stored_lc_common_component(
    measurement: Mapping[str, object],
    *,
    selector_contract: Mapping[str, object],
    expected_cell_id: str | None,
) -> tuple[float, Mapping[str, object], str]:
    validation = measurement.get("validation")
    observation = (
        validation.get(LC_COMMON_COMPONENT_FIELD)
        if isinstance(validation, Mapping)
        else None
    )
    try:
        validate_lc_common_component(
            observation,
            expected_cell_id=expected_cell_id,
            selector_contract=selector_contract,
        )
    except ValueError as error:
        raise RunnerError(
            "LC all-flow authority has no selector-bound common component"
        ) from error
    assert isinstance(observation, Mapping)
    raw_value = observation.get("value")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise RunnerError("LC all-flow common component is not numeric")
    value = float(raw_value)
    if not math.isfinite(value):
        raise RunnerError("LC all-flow common component is not finite")
    return value, observation, _comparison_source_digest(observation)


def conditioned_measurement_comparison(
    cell: CellSpec,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    """Compare a completed candidate to late independent authority evidence."""

    raw_contract = candidate.get("selector_contract")
    baseline_contract = baseline.get("selector_contract")
    if raw_contract != baseline_contract:
        raise RunnerError("candidate and authority selector contracts differ")
    contract = (
        None if raw_contract is None else SelectorContract.from_mapping(raw_contract)  # type: ignore[arg-type]
    )
    candidate_point = _stored_validation_point_digest(candidate)
    baseline_point = _stored_validation_point_digest(baseline)
    if candidate_point != baseline_point:
        raise RunnerError("candidate and authority validation points differ")

    if baseline_uses_lc_common_component_authority(cell, baseline):
        if not isinstance(raw_contract, Mapping) or contract is None:
            raise RunnerError("LC all-flow authority has no selector contract")
        candidate_value, candidate_component, candidate_source = (
            _stored_lc_common_component(
                candidate,
                selector_contract=raw_contract,
                expected_cell_id=cell.cell_id,
            )
        )
        baseline_value, baseline_component, baseline_source = (
            _stored_lc_common_component(
                baseline,
                selector_contract=raw_contract,
                expected_cell_id=None,
            )
        )
        component_identity = {
            "candidate_cell_id": candidate_component["cell_id"],
            "baseline_cell_id": baseline_component["cell_id"],
            "point_digest": candidate_component["point_digest"],
            "helicity_ids": candidate_component["helicity_ids"],
            "color_flow_ids": candidate_component["color_flow_ids"],
        }
        if any(
            candidate_component[field] != baseline_component[field]
            for field in ("point_digest", "helicity_ids", "color_flow_ids")
        ):
            raise RunnerError("LC all-flow common-component identities differ")
        selector_identity = {
            "cell_id": cell.cell_id,
            "accuracy": cell.measurement.accuracy.value,
            "workload": cell.workload.value,
            "selector_contract": contract.as_dict(),
            "value_kind": LC_COMMON_COMPONENT_FIELD,
            "component_identity": component_identity,
        }
        return pointwise_validation(
            candidate_value,
            baseline_value,
            relative_tolerance=_pointwise_tolerance(cell),
            candidate_scale=abs(candidate_value),
            baseline_scale=abs(baseline_value),
            candidate_scale_source="authority-component-magnitude",
            baseline_scale_source="authority-component-magnitude",
            comparison_binding={
                "point_digest": candidate_point,
                "selector_component_identity": selector_identity,
                "selector_component_sha256": _comparison_source_digest(
                    selector_identity
                ),
                "candidate_source_sha256": candidate_source,
                "baseline_source_sha256": baseline_source,
            },
        )

    raw_candidate_value = candidate.get("matrix_element")
    if isinstance(raw_candidate_value, bool) or not isinstance(
        raw_candidate_value, (int, float)
    ):
        raise RunnerError("candidate has no matrix element")
    candidate_value = float(raw_candidate_value)
    if not math.isfinite(candidate_value):
        raise RunnerError("candidate matrix element is not finite")
    baseline_value = _baseline_matrix_element(baseline)
    assert baseline_value is not None
    candidate_scale, candidate_scale_source, candidate_source = (
        _stored_conditioned_scale(candidate, candidate_value)
    )
    baseline_scale, baseline_scale_source, baseline_source = _stored_conditioned_scale(
        baseline, baseline_value
    )
    return pointwise_validation(
        candidate_value,
        baseline_value,
        relative_tolerance=_pointwise_tolerance(cell),
        candidate_scale=candidate_scale,
        baseline_scale=baseline_scale,
        candidate_scale_source=candidate_scale_source,
        baseline_scale_source=baseline_scale_source,
        comparison_binding={
            "point_digest": candidate_point,
            "selector_component_identity": {
                "cell_id": cell.cell_id,
                "accuracy": cell.measurement.accuracy.value,
                "workload": cell.workload.value,
                "selector_contract": None if contract is None else contract.as_dict(),
                "value_kind": "matrix-element",
            },
            "selector_component_sha256": _comparison_source_digest(
                {
                    "cell_id": cell.cell_id,
                    "accuracy": cell.measurement.accuracy.value,
                    "workload": cell.workload.value,
                    "selector_contract": (
                        None if contract is None else contract.as_dict()
                    ),
                    "value_kind": "matrix-element",
                }
            ),
            "candidate_source_sha256": candidate_source,
            "baseline_source_sha256": baseline_source,
        },
    )


def reconcile_independent_authority(
    cell: CellSpec,
    measurement: Mapping[str, object],
    authority: Mapping[str, object],
    *,
    authority_cell_id: str,
) -> dict[str, object]:
    """Resolve one durable unverified result against late authority evidence."""

    initial_status = measurement.get("status")
    if initial_status not in {
        ResultStatus.UNVERIFIED.value,
        ResultStatus.OK.value,
    }:
        raise RunnerError(
            "late authority reconciliation requires measured candidate input"
        )
    validation = measurement.get("validation")
    if not isinstance(validation, Mapping):
        raise RunnerError("unverified result has no validation evidence")
    record = validation.get(INDEPENDENT_AUTHORITY_FIELD)
    if not isinstance(record, Mapping):
        raise RunnerError("unverified result has no independent-authority record")
    expected_ids = tuple(record.get("expected_cell_ids", ()))
    canonical_ids = tuple(
        candidate.cell_id for candidate in independent_numerical_authorities(cell)
    )
    previous_selected = record.get("selected_cell_id")
    record_status = record.get("status")
    if (
        record.get("abi") != INDEPENDENT_AUTHORITY_ABI
        or expected_ids != canonical_ids
        or authority_cell_id not in canonical_ids
        or authority.get("status") != ResultStatus.OK.value
    ):
        raise RunnerError("late independent-authority contract is invalid")
    if initial_status == ResultStatus.UNVERIFIED.value:
        if previous_selected is not None or record_status != "unavailable":
            raise RunnerError("unverified independent-authority state is invalid")
    elif (
        not isinstance(previous_selected, str)
        or previous_selected not in canonical_ids
        or record_status != "verified"
        or canonical_ids.index(authority_cell_id)
        >= canonical_ids.index(previous_selected)
    ):
        raise RunnerError("late authority does not supersede the selected authority")

    updated = dict(measurement)
    mutable_validation = dict(validation)
    try:
        comparison = conditioned_measurement_comparison(cell, updated, authority)
    except Exception as error:
        mutable_validation[INDEPENDENT_AUTHORITY_FIELD] = (
            _independent_authority_record(
                canonical_ids,
                selected_cell_id=authority_cell_id,
                status="incompatible",
                reason="successful-independent-authority-contract-incompatible",
            )
        )
        mutable_validation["status"] = ResultStatus.VALIDATION_FAILED.value
        updated["status"] = ResultStatus.VALIDATION_FAILED.value
        updated["validation"] = mutable_validation
        updated["failure"] = {
            "kind": "IndependentAuthorityContractError",
            "message": str(error),
        }
        return updated

    mutable_validation["pointwise"] = comparison
    passed = comparison.get("status") == ResultStatus.OK.value
    mutable_validation[INDEPENDENT_AUTHORITY_FIELD] = _independent_authority_record(
        canonical_ids,
        selected_cell_id=authority_cell_id,
        status="verified" if passed else "mismatch",
        reason=(
            "independent-authority-agreement"
            if passed
            else "independent-authority-mismatch"
        ),
    )
    resolved_status = (
        ResultStatus.OK.value if passed else ResultStatus.VALIDATION_FAILED.value
    )
    mutable_validation["status"] = resolved_status
    updated["status"] = resolved_status
    updated["validation"] = mutable_validation
    updated["failure"] = (
        None
        if passed
        else {
            "kind": "IndependentAuthorityMismatch",
            "message": "candidate disagrees with independent numerical authority",
        }
    )
    return updated


_PRECISION_DIAGNOSTIC_ABI = (
    "pyamplicol-report-validation-failure-precision-diagnostic-v2"
)
_PRECISION_DIAGNOSTIC_CELL_IDS = frozenset(
    {
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted",
        "matrix-compiled-builtin-sm-nlc-n4-dd-tt-jets-contracted",
        "matrix-recurrence-builtin-sm-lc-n4-dd-tt-jets-all-flow",
    }
)


def _precision_diagnostic_enabled(cell: CellSpec) -> bool:
    return cell.cell_id in _PRECISION_DIAGNOSTIC_CELL_IDS


def _diagnostic_number(value: object) -> dict[str, object]:
    real = getattr(value, "real", value)
    if callable(real):
        real = real()
    return {"value": str(real), "binary64": _real_nonnegative(value)}


def _unavailable_precision_diagnostic(error: BaseException) -> dict[str, object]:
    return {
        "abi": _PRECISION_DIAGNOSTIC_ABI,
        "status": "unavailable",
        "promotes_measurement": False,
        "error": {"kind": type(error).__name__, "message": str(error)},
    }


def _diagnostic_timing_context(
    cell: CellSpec,
    measurement: Mapping[str, object] | None,
) -> dict[str, object]:
    """Copy independent measured clocks without recomputing or deriving them."""

    measurement = measurement or {}
    provenance = measurement.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    evaluator_total = provenance.get("evaluator_total_timing")
    execution = provenance.get("execution_timing")
    return {
        "source": "copied-from-measurement",
        "recomputed": False,
        "outer_wall_seconds_per_point": measurement.get("wall_seconds_per_point"),
        "evaluator_total_timing": (
            dict(evaluator_total) if isinstance(evaluator_total, Mapping) else None
        ),
        "recurrence_core_timing": (
            dict(execution)
            if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
            and isinstance(execution, Mapping)
            else None
        ),
    }


def _diagnostic_authorities(
    cell: CellSpec,
    measurements: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Extract the few already-stored values relevant to the retained canaries."""

    result: list[dict[str, object]] = []
    for label, measurement in measurements.items():
        validation = measurement.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        component_only = baseline_uses_lc_common_component_authority(cell, measurement)
        legacy = validation.get(_LEGACY_NUMERICAL_AUTHORITY_FIELD)
        source = (
            str(legacy.get("source"))
            if isinstance(legacy, Mapping)
            else "stored-successful-measurement"
        )

        common = validation.get(LC_COMMON_COMPONENT_FIELD)
        imode2 = validation.get("legacy_imode2_diagnostic")
        kind = LC_COMMON_COMPONENT_FIELD if component_only else "total"
        candidates = (
            ("matrix-element", "total", measurement.get("matrix_element"), source),
            (
                LC_COMMON_COMPONENT_FIELD,
                LC_COMMON_COMPONENT_FIELD,
                common.get("value") if isinstance(common, Mapping) else None,
                source,
            ),
            (
                "generated-library",
                kind,
                imode2.get("authoritative_value")
                if isinstance(imode2, Mapping)
                else None,
                str(imode2.get("authoritative_source", source))
                if isinstance(imode2, Mapping)
                else source,
            ),
            (
                "direct-imode2",
                kind,
                imode2.get("imode2_value") if isinstance(imode2, Mapping) else None,
                "legacy-direct-imode2-non-certifying",
            ),
        )
        for name, value_kind, value, value_source in candidates:
            if (
                (name == "matrix-element" and component_only)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            result.append(
                {
                    "authority": f"{label}:{name}",
                    "value_kind": value_kind,
                    "value": float(value),
                    "source": value_source,
                }
            )
    return result


def _diagnostic_evaluation(
    runtime: object,
    points: object,
    *,
    cell: CellSpec,
    contract: SelectorContract | None,
    precision: int,
    point_context: str,
) -> dict[str, object]:
    selectors = _selector_kwargs(cell, contract)
    values = runtime.evaluate(  # type: ignore[attr-defined]
        points, precision=precision, **selectors
    )
    resolved = runtime.evaluate_resolved(  # type: ignore[attr-defined]
        points, precision=precision, **selectors
    )
    totals = tuple(resolved.total())
    if len(values) != 1 or len(totals) != 1:
        raise RunnerError("precision diagnostic requires exactly one point")
    result = {
        "point_context": point_context,
        "total": _diagnostic_number(values[0]),
        "resolved_sum": _diagnostic_number(totals[0]),
        "internal_agreement": pointwise_validation(
            _real_nonnegative(values[0]),
            _real_nonnegative(totals[0]),
        ),
    }
    if (
        contract is not None
        and cell.measurement.accuracy is Accuracy.LC
        and cell.workload is Workload.ALL_FLOW
    ):
        component = runtime.evaluate_resolved(  # type: ignore[attr-defined]
            points,
            precision=precision,
            helicities=contract.runtime_all_flow_helicity_ids,
            color_flows=contract.selected_color_flow_ids,
        )
        if (
            tuple(component.helicity_ids) != contract.runtime_all_flow_helicity_ids
            or tuple(component.color_ids) != contract.selected_color_flow_ids
        ):
            raise RunnerError("diagnostic LC common-component axes changed")
        try:
            result[LC_COMMON_COMPONENT_FIELD] = _diagnostic_number(
                component.values[0][0][0]
            )
        except (AttributeError, IndexError, TypeError) as error:
            raise RunnerError("diagnostic LC common component is not scalar") from error
    return result


def _validation_failure_precision_diagnostic(
    runtime: object,
    points: object,
    *,
    cell: CellSpec,
    contract: SelectorContract | None,
    baseline: Mapping[str, object] | None,
    peers: Mapping[str, Mapping[str, object]] | None = None,
    measurement_context: Mapping[str, object] | None = None,
    complete_ladder: bool = False,
) -> dict[str, object]:
    """Attach a bounded, non-promoting p32/p200 explanation to three canaries."""

    try:
        measurements = dict(peers or {})
        if baseline is not None:
            measurements = {"baseline": baseline, **measurements}
        authorities = _diagnostic_authorities(cell, measurements)
        retained_digest = point_digest(points)
        projector = getattr(runtime, "_diagnostic_project_onshell", None)
        if not callable(projector):
            raise RunnerError("runtime does not expose diagnostic on-shell projection")
        diagnostic_points, raw_projection = projector(points, precision=200)
        if not isinstance(raw_projection, Mapping):
            raise RunnerError("runtime returned invalid diagnostic projection metadata")
        projection = dict(raw_projection)
        unchanged = projection.get("unchanged")
        if not isinstance(unchanged, bool):
            raise RunnerError(
                "diagnostic projection does not report whether it changed"
            )
        projected_digest = projection.get("projected_digest")
        if not isinstance(projected_digest, str) or not projected_digest:
            raise RunnerError("diagnostic projection has no projected point digest")
        projection["original_digest"] = retained_digest
        if unchanged:
            candidate_point_context = "retained-point-unchanged-by-projection"
            authority_point_context = candidate_point_context
        else:
            candidate_point_context = "projected-onshell-point"
            authority_point_context = "original-retained-point"
        result: dict[str, object] = {
            "abi": _PRECISION_DIAGNOSTIC_ABI,
            "status": "diagnostic-only",
            "promotes_measurement": False,
            "timings_unchanged": True,
            "retained_point": {
                "digest": retained_digest,
                "momenta": _reproduction_momenta(points),
            },
            "kinematic_projection": projection,
            "stored_peer_point_context": authority_point_context,
            "selector_contract": None if contract is None else contract.as_dict(),
            "execution_identity": {
                "cell_id": cell.cell_id,
                "process": cell.process,
                "multiplicity": cell.n_final,
                "workload": cell.workload.value,
                "layout": cell.workload.value,
                "execution_mode": cell.measurement.execution_mode.value,
                "backend": cell.measurement.backend,
                "model": (
                    None
                    if cell.measurement.model is None
                    else cell.measurement.model.value
                ),
                "accuracy": cell.measurement.accuracy.value,
                "variant": cell.variant,
            },
            "measurement_timing_context": _diagnostic_timing_context(
                cell, measurement_context
            ),
            "attempts": [],
        }
        attempts = result["attempts"]
        assert isinstance(attempts, list)
        for precision in (32, 200):
            attempt: dict[str, object] = {"precision_digits": precision}
            try:
                candidate = _diagnostic_evaluation(
                    runtime,
                    diagnostic_points,
                    cell=cell,
                    contract=contract,
                    precision=precision,
                    point_context=candidate_point_context,
                )
                comparisons = []
                for authority in authorities:
                    value_kind = str(authority["value_kind"])
                    candidate_fields = (
                        ("total", "resolved_sum")
                        if value_kind == "total"
                        else (value_kind,)
                    )
                    for candidate_field in candidate_fields:
                        candidate_value = candidate.get(candidate_field)
                        if not isinstance(candidate_value, Mapping):
                            continue
                        comparison = pointwise_validation(
                            float(candidate_value["binary64"]),
                            float(authority["value"]),
                            relative_tolerance=_pointwise_tolerance(cell),
                        )
                        comparison_payload = {
                            "candidate_value_kind": candidate_field,
                            "authority": authority["authority"],
                            "source": authority["source"],
                            "candidate_point_context": candidate_point_context,
                            "authority_point_context": authority_point_context,
                            "same_kinematic_point": unchanged,
                            "certifying": unchanged,
                            **comparison,
                        }
                        if not unchanged:
                            comparison_payload.pop("status", None)
                            comparison_payload["context_only"] = True
                        comparisons.append(comparison_payload)
                attempt.update(
                    {
                        "status": "evaluated",
                        "candidate": candidate,
                        "comparisons": comparisons,
                    }
                )
            except Exception as error:
                attempt.update(
                    {
                        "status": "unavailable",
                        "error": {
                            "kind": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                attempts.append(attempt)
                continue
            attempts.append(attempt)
            internal = candidate["internal_agreement"]
            if (
                not complete_ladder
                and
                precision == 32
                and isinstance(internal, Mapping)
                and internal.get("status") == ResultStatus.OK.value
                and comparisons
                and all(
                    comparison.get("status") == ResultStatus.OK.value
                    for comparison in comparisons
                )
            ):
                break
        return result
    except Exception as error:
        return _unavailable_precision_diagnostic(error)


def attach_validation_failure_precision_diagnostic(
    cell: CellSpec,
    measurement: dict[str, object],
    *,
    baseline: Mapping[str, object] | None,
    peers: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Add the same bounded diagnostic after late direct-agreement failures."""

    validation = measurement.get("validation")
    if (
        measurement.get("status") != ResultStatus.VALIDATION_FAILED.value
        or not _precision_diagnostic_enabled(cell)
        or not isinstance(validation, Mapping)
        or "precision_diagnostic" in validation
    ):
        return
    mutable_validation = dict(validation)
    try:
        artifact = measurement.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RunnerError("failed measurement has no generated artifact")
        path = artifact.get("path")
        process_id = artifact.get("process_id")
        if not isinstance(path, str) or not isinstance(process_id, str):
            raise RunnerError("failed measurement artifact is not loadable")
        runtime = _load_runtime(
            Path(path).expanduser().resolve(strict=False), process_id
        )
        points = shared_validation_points(cell.process)
        raw_contract = measurement.get("selector_contract")
        contract = (
            None
            if raw_contract is None
            else SelectorContract.from_mapping(raw_contract)
        )
        diagnostic = _validation_failure_precision_diagnostic(
            runtime,
            points,
            cell=cell,
            contract=contract,
            baseline=baseline,
            peers=peers,
            measurement_context=measurement,
        )
    except Exception as error:
        diagnostic = _unavailable_precision_diagnostic(error)
    mutable_validation["precision_diagnostic"] = diagnostic
    measurement["validation"] = mutable_validation


def _require_nonzero_lc_all_flow_baseline(
    cell: CellSpec,
    baseline: Mapping[str, object] | None,
) -> None:
    """Reject inherited selector contracts that authenticate only a zero lane."""

    if (
        baseline is None
        or cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
    ):
        return
    validation = baseline.get("validation")
    common = (
        validation.get(LC_COMMON_COMPONENT_FIELD)
        if isinstance(validation, Mapping)
        else None
    )
    value = common.get("value") if isinstance(common, Mapping) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RunnerError(
            "LC all-flow baseline selector is structural zero; remeasure the "
            "baseline with a nonzero fixed-helicity selector before generating "
            "a dependent candidate"
        )


def generated_artifact_from_measurement(
    measurement: Mapping[str, object],
) -> GeneratedArtifact:
    if measurement.get("status") != ResultStatus.OK.value:
        raise RunnerError("only a successful measurement can provide an artifact")
    artifact = measurement.get("artifact")
    provenance = measurement.get("provenance")
    generation_seconds = measurement.get("generation_seconds")
    if (
        not isinstance(artifact, Mapping)
        or not isinstance(provenance, Mapping)
        or isinstance(generation_seconds, bool)
        or not isinstance(generation_seconds, (int, float))
    ):
        raise RunnerError("measurement does not contain reusable artifact metadata")
    path = artifact.get("path")
    process_id = artifact.get("process_id")
    requested = provenance.get("requested_config")
    effective = provenance.get("effective_config")
    preparation_seconds = provenance.get("model_preparation_seconds", 0.0)
    preparation_reused = provenance.get("model_preparation_reused", True)
    generation_command_path = provenance.get("generation_command_path")
    numerical_relation_correctness = provenance.get("numerical_relation_correctness")
    numerical_relation_fallback = provenance.get("numerical_relation_fallback")
    if (
        not isinstance(path, str)
        or not isinstance(process_id, str)
        or not isinstance(requested, Mapping)
        or not isinstance(effective, Mapping)
        or isinstance(preparation_seconds, bool)
        or not isinstance(preparation_seconds, (int, float))
        or not isinstance(preparation_reused, bool)
        or (
            generation_command_path is not None
            and not isinstance(generation_command_path, str)
        )
        or (
            numerical_relation_correctness is not None
            and not isinstance(numerical_relation_correctness, Mapping)
        )
        or (
            numerical_relation_fallback is not None
            and not isinstance(numerical_relation_fallback, Mapping)
        )
    ):
        raise RunnerError("measurement reusable artifact metadata is malformed")
    artifact_path = Path(path).expanduser().resolve(strict=False)
    if not artifact_path.is_dir():
        raise RunnerError(f"reusable artifact directory is missing: {artifact_path}")
    return GeneratedArtifact(
        path=artifact_path,
        process_id=process_id,
        generation_seconds=float(generation_seconds),
        model_preparation_seconds=float(preparation_seconds),
        model_preparation_reused=preparation_reused,
        requested_config=dict(requested),
        effective_config=dict(effective),
        generation_command_path=generation_command_path,
        numerical_relation_correctness=(
            None
            if numerical_relation_correctness is None
            else dict(numerical_relation_correctness)
        ),
        numerical_relation_fallback=(
            None
            if numerical_relation_fallback is None
            else dict(numerical_relation_fallback)
        ),
    )


def _reuse_artifact_for_measurement(
    artifact: GeneratedArtifact,
    *,
    phase_reporter: WorkerPhaseReporter | None,
) -> GeneratedArtifact:
    """Close the local generation phase when no generation work is needed."""

    if phase_reporter is not None:
        # The inherited generation_seconds remains the cost of producing the
        # reusable artifact.  This empty local interval only gives the current
        # supervised worker an authenticated terminal phase state.
        with phase_reporter.generation():
            pass
    return artifact


def _pointwise_tolerance(cell: CellSpec) -> float:
    if cell.dataset_id.startswith("matrix_recurrence_") or cell.dataset_id.startswith(
        "z_"
    ):
        return INDEPENDENT_RELATIVE_TOLERANCE
    return RELATIVE_TOLERANCE


def _stable_runtime_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Remove only the growing loaded-module observation inventory."""

    stable = dict(identity)
    raw_policy = stable.get("loaded_module_origin_policy")
    if isinstance(raw_policy, Mapping):
        policy = dict(raw_policy)
        for field in (
            "observed_module_count",
            "observations",
            "observations_sha256",
        ):
            policy.pop(field, None)
        stable["loaded_module_origin_policy"] = policy
    return stable


def _runtime_identity_digest(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _comparison_source_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _conditioned_binding(
    cell: CellSpec,
    points: object,
    contract: SelectorContract | None,
    *,
    candidate_source_sha256: str,
    baseline_source_sha256: str,
    value_kind: str,
) -> dict[str, object]:
    selector_identity = {
        "cell_id": cell.cell_id,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "selector_contract": None if contract is None else contract.as_dict(),
        "value_kind": value_kind,
    }
    return {
        "point_digest": point_digest(points),
        "selector_component_identity": selector_identity,
        "selector_component_sha256": _comparison_source_digest(selector_identity),
        "candidate_source_sha256": candidate_source_sha256,
        "baseline_source_sha256": baseline_source_sha256,
    }


def _validate_runtime_identity_postflight(
    initial: Mapping[str, object],
    postflight: Mapping[str, object],
) -> None:
    if _stable_runtime_identity(initial) != _stable_runtime_identity(postflight):
        raise RunnerError(
            "candidate runtime identity changed during report measurement"
        )
    initial_policy = initial.get("loaded_module_origin_policy")
    postflight_policy = postflight.get("loaded_module_origin_policy")
    if not isinstance(initial_policy, Mapping) or not isinstance(
        postflight_policy,
        Mapping,
    ):
        raise RunnerError("report runtime has no loaded-module origin evidence")
    initial_observations = initial_policy.get("observations")
    postflight_observations = postflight_policy.get("observations")
    if not isinstance(initial_observations, list) or not isinstance(
        postflight_observations,
        list,
    ):
        raise RunnerError("report runtime loaded-module evidence is incomplete")
    postflight_keys = {
        json.dumps(
            observation,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for observation in postflight_observations
    }
    if any(
        json.dumps(
            observation,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        not in postflight_keys
        for observation in initial_observations
    ):
        raise RunnerError("report runtime lost an authenticated loaded-module origin")


def _load_runtime(artifact_path: Path, process_id: str) -> object:
    from pyamplicol.api import Runtime

    return Runtime.load(artifact_path, process=process_id)


def _resolution_benchmark_config(effective_config: Mapping[str, object]) -> object:
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import resolve_config

    return resolve_config(
        effective_config,
        action=Action.GENERATE,
        base_dir=Path.cwd(),
    ).effective.benchmark


def measure_pyamplicol_cell(
    cell: CellSpec,
    *,
    artifact_path: Path,
    settings: RunnerSettings,
    repo_root: Path,
    baseline: Mapping[str, object] | None,
    expected_authority_cell_ids: Sequence[str] = (),
    selected_authority_cell_id: str | None = None,
    validation_peers: Mapping[str, Mapping[str, object]] | None = None,
    selector_provider: Mapping[str, object] | None = None,
    prepared_model_path: Path | None = None,
    reused_artifact: GeneratedArtifact | None = None,
    phase_reporter: WorkerPhaseReporter | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, object]:
    """Generate or retime one complete-coverage pyAmpliCol artifact."""

    canonical_authority_ids = tuple(
        authority.cell_id
        for authority in independent_numerical_authorities(cell, catalog=catalog)
    )
    authority_ids = tuple(expected_authority_cell_ids)
    if authority_ids != canonical_authority_ids:
        raise RunnerError(
            "expected numerical authorities differ from the canonical catalog chain"
        )
    if selected_authority_cell_id is not None and (
        selected_authority_cell_id not in authority_ids or baseline is None
    ):
        raise RunnerError("selected numerical authority is invalid")
    if authority_ids and (baseline is not None) != (
        selected_authority_cell_id is not None
    ):
        raise RunnerError(
            "compiled/eager baseline and selected numerical authority differ"
        )

    _require_nonzero_lc_all_flow_baseline(cell, baseline)
    contract = _measurement_selector_contract(cell, baseline, selector_provider)
    generated = (
        generate_artifact(
            cell,
            artifact_path,
            settings=settings,
            repo_root=repo_root,
            prepared_model_path=prepared_model_path,
            phase_reporter=phase_reporter,
        )
        if reused_artifact is None
        else _reuse_artifact_for_measurement(
            reused_artifact,
            phase_reporter=phase_reporter,
        )
    )
    validate_artifact_contract(cell, generated.path)
    runtime = _load_runtime(generated.path, generated.process_id)
    revision = settings.source_revision_override or source_revision(
        repo_root,
        require_clean=True,
    )
    runtime_identity = runtime_identity_payload(
        cell,
        runtime,
        generated.path,
        generated.process_id,
        expected_source_revision=revision,
    )
    runtime_identity_sha256 = _runtime_identity_digest(runtime_identity)
    points = (
        runtime_validation_points(runtime)
        if cell.measurement.model in {ModelKey.SCALAR_CONTACT, ModelKey.SCALAR_GRAVITY}
        else shared_validation_points(cell.process)
    )
    if cell.measurement.accuracy is Accuracy.LC and contract is None:
        from .runner import derive_selector_contract

        contract = derive_selector_contract(runtime, points)
    if phase_reporter is not None:
        phase_reporter.profiling_started()
    profiling_deadlines = tuple(
        deadline
        for deadline in (
            settings.worker_deadline_monotonic,
            (
                None
                if settings.profiling_time_limit_seconds is None
                else time.monotonic() + settings.profiling_time_limit_seconds
            ),
        )
        if deadline is not None
    )
    profile = profile_runtime(
        runtime,
        points,
        cell=cell,
        benchmark_config=_resolution_benchmark_config(generated.effective_config),
        selector_contract=contract,
        progress=settings.progress,
        profiling_deadline_monotonic=(
            min(profiling_deadlines) if profiling_deadlines else None
        ),
    )
    execution_timing = profile.pop("execution_timing")
    evaluator_total_timing = profile.pop("evaluator_total_timing", None)
    arena_profile_evidence = profile.pop("arena_profile_evidence")
    benchmark_evidence = profile.pop("benchmark_evidence")
    if phase_reporter is not None:
        phase_reporter.validation_started()
    resolved_sum = profile.pop("resolved_sum_validation")
    validation: dict[str, object] = {
        "resolved_sum": resolved_sum,
        DIRECT_AGREEMENT_FIELD: [],
    }
    if cell.measurement.accuracy is Accuracy.LC:
        assert contract is not None
        validation[LC_COMMON_COMPONENT_FIELD] = evaluate_lc_common_component(
            runtime,
            points,
            cell=cell,
            contract=contract,
        )
    scalar = cell.measurement.model in {
        ModelKey.SCALAR_CONTACT,
        ModelKey.SCALAR_GRAVITY,
    }
    authority_expected = requires_independent_numerical_authority(
        cell,
        catalog=catalog,
    )
    diagnostic_without_authority = (
        authority_expected and selected_authority_cell_id is None
    )
    requires_high_precision = scalar or (
        baseline is None
        and cell.measurement.execution_mode
        in {
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        }
    )
    if requires_high_precision:
        high_precision_resolved = resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
            precision=32,
        )
        high_precision_records = high_precision_resolved.get("points")
        candidate_records = resolved_sum.get("points")
        if (
            not isinstance(high_precision_records, list)
            or len(high_precision_records) != 1
            or not isinstance(candidate_records, list)
            or len(candidate_records) != 1
            or not isinstance(high_precision_records[0], Mapping)
            or not isinstance(candidate_records[0], Mapping)
        ):
            raise RunnerError("high-precision validation requires one resolved point")
        high_record = high_precision_records[0]
        candidate_record = candidate_records[0]
        high_precision_value = float(high_record["candidate"])
        candidate_source = resolved_sum.get("resolved_source_sha256")
        baseline_source = high_precision_resolved.get("resolved_source_sha256")
        if not isinstance(candidate_source, str) or not isinstance(
            baseline_source, str
        ):
            raise RunnerError("high-precision resolved source binding is unavailable")
        validation["high_precision_resolved_sum"] = high_precision_resolved
        validation["high_precision"] = pointwise_validation(
            float(profile["matrix_element"]),
            high_precision_value,
            candidate_scale=float(candidate_record["candidate_scale"]),
            baseline_scale=float(high_record["candidate_scale"]),
            candidate_scale_source="resolved-component-l1-binary64",
            baseline_scale_source="resolved-component-l1-p32",
            comparison_binding=_conditioned_binding(
                cell,
                points,
                contract,
                candidate_source_sha256=candidate_source,
                baseline_source_sha256=baseline_source,
                value_kind="matrix-element-p16-versus-p32",
            ),
        )
    _attach_baseline_validation(
        cell,
        validation,
        candidate_matrix_element=float(profile["matrix_element"]),
        baseline=baseline,
        points=points,
        contract=contract,
    )
    authority_failure: dict[str, str] | None = None
    if authority_expected:
        if selected_authority_cell_id is None:
            validation[INDEPENDENT_AUTHORITY_FIELD] = (
                _independent_authority_record(
                    authority_ids,
                    selected_cell_id=None,
                    status="unavailable",
                    reason="no-successful-independent-authority",
                )
            )
        else:
            assert baseline is not None
            candidate_view = {
                "matrix_element": float(profile["matrix_element"]),
                "selector_contract": None if contract is None else contract.as_dict(),
                "validation": validation,
            }
            try:
                comparison = conditioned_measurement_comparison(
                    cell,
                    candidate_view,
                    baseline,
                )
            except Exception as error:
                validation[INDEPENDENT_AUTHORITY_FIELD] = (
                    _independent_authority_record(
                        authority_ids,
                        selected_cell_id=selected_authority_cell_id,
                        status="incompatible",
                        reason=(
                            "successful-independent-authority-contract-incompatible"
                        ),
                    )
                )
                validation["authority_contract_error"] = {
                    "kind": type(error).__name__,
                    "message": str(error),
                }
                authority_failure = {
                    "kind": "IndependentAuthorityContractError",
                    "message": str(error),
                }
            else:
                validation["pointwise"] = comparison
                passed = comparison.get("status") == ResultStatus.OK.value
                validation[INDEPENDENT_AUTHORITY_FIELD] = (
                    _independent_authority_record(
                        authority_ids,
                        selected_cell_id=selected_authority_cell_id,
                        status="verified" if passed else "mismatch",
                        reason=(
                            "independent-authority-agreement"
                            if passed
                            else "independent-authority-mismatch"
                        ),
                    )
                )
                if not passed:
                    authority_failure = {
                        "kind": "IndependentAuthorityMismatch",
                        "message": (
                            "candidate disagrees with independent numerical authority"
                        ),
                    }
    statuses = {
        str(record.get("status"))
        for record in validation.values()
        if isinstance(record, Mapping)
    }
    internal_failure = ResultStatus.VALIDATION_FAILED.value in statuses
    validation["status"] = (
        ResultStatus.VALIDATION_FAILED.value
        if internal_failure or authority_failure is not None
        else (
            ResultStatus.UNVERIFIED.value
            if diagnostic_without_authority
            else ResultStatus.OK.value
        )
    )
    if (
        validation["status"] == ResultStatus.VALIDATION_FAILED.value
        and _precision_diagnostic_enabled(cell)
    ) or validation["status"] == ResultStatus.UNVERIFIED.value:
        validation["precision_diagnostic"] = _validation_failure_precision_diagnostic(
            runtime,
            points,
            cell=cell,
            contract=contract,
            baseline=baseline,
            peers=validation_peers,
            measurement_context={
                "wall_seconds_per_point": profile.get("wall_seconds_per_point"),
                "provenance": {
                    "execution_timing": execution_timing,
                    "evaluator_total_timing": evaluator_total_timing,
                },
            },
            complete_ladder=(
                validation["status"] == ResultStatus.UNVERIFIED.value
            ),
        )
    status = str(profile["status"])
    if validation["status"] == ResultStatus.VALIDATION_FAILED.value:
        status = ResultStatus.VALIDATION_FAILED.value
    elif validation["status"] == ResultStatus.UNVERIFIED.value:
        status = ResultStatus.UNVERIFIED.value
    runtime_identity_postflight = runtime_identity_payload(
        cell,
        runtime,
        generated.path,
        generated.process_id,
        expected_source_revision=revision,
    )
    _validate_runtime_identity_postflight(
        runtime_identity,
        runtime_identity_postflight,
    )
    runtime_identity_stable_sha256 = _runtime_identity_digest(
        _stable_runtime_identity(runtime_identity)
    )
    runtime_identity_postflight_stable_sha256 = _runtime_identity_digest(
        _stable_runtime_identity(runtime_identity_postflight)
    )
    postflight_origin_policy = runtime_identity_postflight.get(
        "loaded_module_origin_policy"
    )
    if not isinstance(postflight_origin_policy, Mapping):
        raise RunnerError(
            "report runtime postflight has no loaded-module origin evidence"
        )

    measurement = empty_measurement()
    measurement.update(profile)
    measurement.update(
        {
            "status": status,
            "generation_seconds": generated.generation_seconds,
            "artifact": {
                "path": str(generated.path),
                "process_id": generated.process_id,
                "policy": "reused" if reused_artifact is not None else "generated",
            },
            "selector_contract": None if contract is None else contract.as_dict(),
            "validation": validation,
            "resources": None,
            "provenance": {
                **provenance_payload(),
                "source_revision": revision,
                "requested_config": dict(generated.requested_config),
                "effective_config": dict(generated.effective_config),
                "model_preparation_seconds": generated.model_preparation_seconds,
                "model_preparation_reused": generated.model_preparation_reused,
                "generation_timer_excludes_model_preparation": True,
                "generation_command_path": generated.generation_command_path,
                **(
                    {
                        "numerical_relation_correctness": dict(
                            generated.numerical_relation_correctness
                        )
                    }
                    if generated.numerical_relation_correctness is not None
                    else {}
                ),
                **(
                    {
                        "numerical_relation_fallback": dict(
                            generated.numerical_relation_fallback
                        )
                    }
                    if generated.numerical_relation_fallback is not None
                    else {}
                ),
                "report_momenta": _reproduction_momenta(points),
                "runtime_profile": benchmark_evidence,
                "execution_timing": execution_timing,
                **(
                    {"evaluator_total_timing": evaluator_total_timing}
                    if evaluator_total_timing is not None
                    else {}
                ),
                "arena_profile_evidence": arena_profile_evidence,
                "runtime_identity": runtime_identity,
                "runtime_identity_sha256": runtime_identity_sha256,
                "runtime_identity_stable_sha256": (runtime_identity_stable_sha256),
                "runtime_identity_postflight_stable_sha256": (
                    runtime_identity_postflight_stable_sha256
                ),
                "runtime_identity_postflight_loaded_module_origin_policy": dict(
                    postflight_origin_policy
                ),
                "runtime_identity_postflight_match": True,
            },
            "failure": None,
        }
    )
    if status == ResultStatus.UNVERIFIED.value:
        measurement["failure"] = {
            "kind": "IndependentAuthorityUnavailable",
            "message": "no successful independent numerical authority is available",
        }
    elif status != ResultStatus.OK.value:
        measurement["failure"] = authority_failure or {
            "kind": "MeasurementValidationError",
            "message": "candidate or same-artifact numerical validation failed",
        }
    return measurement


def failure_measurement(
    status: ResultStatus,
    error: BaseException | str,
    *,
    resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if status in {ResultStatus.NOT_AVAILABLE, ResultStatus.OK}:
        raise ValueError("failure measurement requires a terminal failure status")
    message = str(error)
    failure = empty_measurement()
    failure.update(
        {
            "status": status.value,
            "resources": None if resources is None else dict(resources),
            "failure": {
                "kind": (
                    type(error).__name__ if isinstance(error, BaseException) else None
                ),
                "message": message,
            },
        }
    )
    return failure


def _validate_portable_current_result_path(
    path: Path,
    paths: ReportPaths,
) -> Path:
    root_input = paths.artifact_root.expanduser()
    root_lexical = Path(os.path.abspath(root_input))
    try:
        root = root_input.resolve(strict=True)
    except OSError as error:
        raise ValueError("portable current artifact root is unavailable") from error
    if root_input.is_symlink() or root != root_lexical or not root.is_dir():
        raise ValueError("portable current artifact root is not canonical")

    source_input = path.expanduser()
    source_lexical = Path(os.path.abspath(source_input))
    try:
        source = source_input.resolve(strict=True)
    except OSError as error:
        raise ValueError("portable current result file is unavailable") from error
    if source_input.is_symlink() or source != source_lexical or not source.is_file():
        raise ValueError("portable current result path is not canonical")
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "portable current result is outside the artifact root"
        ) from error
    parts = relative.parts
    if (
        len(parts) != 5
        or parts[0] != "cells"
        or not parts[1]
        or parts[2] != "attempts"
        or parts[4] != "result.json"
    ):
        raise ValueError("portable current result path has an unsupported shape")
    try:
        attempt_id = str(uuid.UUID(parts[3]))
    except (AttributeError, ValueError) as error:
        raise ValueError("portable current result attempt ID is invalid") from error
    if attempt_id != parts[3]:
        raise ValueError("portable current result attempt ID is not canonical")
    return source


def load_measurement(
    path: Path,
    *,
    publication_paths: ReportPaths | None = None,
) -> dict[str, object]:
    source = (
        path
        if publication_paths is None
        else _validate_portable_current_result_path(path, publication_paths)
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("measurement file must contain an object")
    if publication_paths is not None:
        from .publication import materialize_current_value

        payload = materialize_current_value(payload, publication_paths)
        if not isinstance(payload, Mapping):  # pragma: no cover - parsed object above.
            raise ValueError("portable measurement did not materialize as an object")
    return dict(payload)


__all__ = [
    "attach_validation_failure_precision_diagnostic",
    "baseline_uses_lc_common_component_authority",
    "conditioned_measurement_comparison",
    "failure_measurement",
    "file_digest",
    "generated_artifact_from_measurement",
    "load_measurement",
    "measure_pyamplicol_cell",
    "reconcile_independent_authority",
    "shared_validation_points",
    "source_revision",
]
