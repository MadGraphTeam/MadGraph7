# SPDX-License-Identifier: 0BSD
"""Direct numerical-agreement edges required by the publication contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .catalog import REPORT_CATALOG, ReportCatalog
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .runner import (
    ABSOLUTE_TOLERANCE,
    CONDITIONED_COMPARISON_ABI,
    INDEPENDENT_RELATIVE_TOLERANCE,
    RELATIVE_TOLERANCE,
    SelectorContract,
    _real_nonnegative,
    point_digest,
    pointwise_validation,
    validate_conditioned_comparison_record,
)

DIRECT_AGREEMENT_ABI = "pyamplicol-report-direct-agreement-v1"
DIRECT_AGREEMENT_V1_ABI = DIRECT_AGREEMENT_ABI
DIRECT_AGREEMENT_V2_ABI = "pyamplicol-report-direct-agreement-v2"
LC_COMMON_COMPONENT_ABI = "pyamplicol-report-lc-common-component-v1"
BUILTIN_UFO_RECURRENCE = "builtin-ufo-recurrence"
Z_RECURRENCE_CROSS_MODE = "z-recurrence-cross-mode"
LC_CROSS_LAYOUT_COMPONENT = "lc-cross-layout-component"
LC_LEGACY_PYAMPLICOL_COMPONENT = "lc-legacy-pyamplicol-component"
DIRECT_AGREEMENT_KINDS = (
    BUILTIN_UFO_RECURRENCE,
    Z_RECURRENCE_CROSS_MODE,
    LC_CROSS_LAYOUT_COMPONENT,
    LC_LEGACY_PYAMPLICOL_COMPONENT,
)
DIRECT_AGREEMENT_FIELD = "direct_agreements"
LC_COMMON_COMPONENT_FIELD = "lc_common_component"
INDEPENDENT_AUTHORITY_ABI = "pyamplicol-report-independent-authority-v1"
INDEPENDENT_AUTHORITY_FIELD = "independent_authority"
STRICT_RELATIVE_TOLERANCE = RELATIVE_TOLERANCE
STRICT_ABSOLUTE_TOLERANCE = ABSOLUTE_TOLERANCE


class AgreementError(RuntimeError):
    """A required direct numerical comparison could not be established."""


@dataclass(frozen=True, slots=True)
class AgreementEdge:
    kind: str
    baseline: CellSpec
    candidate: CellSpec

    @property
    def value_kind(self) -> str:
        return (
            LC_COMMON_COMPONENT_FIELD
            if self.kind in {LC_CROSS_LAYOUT_COMPONENT, LC_LEGACY_PYAMPLICOL_COMPONENT}
            else "matrix_element"
        )

    @property
    def relative_tolerance(self) -> float:
        return agreement_relative_tolerance(self.kind)

    @property
    def required(self) -> bool:
        """Whether this peer is a hard measurement prerequisite.

        Original-AmpliCol LC agreement is useful when the legacy oracle is
        available, but its absence must not censor an otherwise independently
        validated pyAmpliCol measurement.  Z cross-mode recurrence is likewise
        an availability-only numerical authority.  Model-source and
        cross-layout pyAmpliCol edges remain mandatory.
        """

        return self.kind not in {
            LC_LEGACY_PYAMPLICOL_COMPONENT,
            Z_RECURRENCE_CROSS_MODE,
        }


def validation_baseline_is_required(
    cell: CellSpec,
    baseline: CellSpec | None,
) -> bool:
    """Return whether ``baseline`` is a hard numerical prerequisite.

    Recurrence has its own resolved-sum and high-precision validation path, so
    original AmpliCol is an optional comparison for recurrence.  Compiled and
    eager comparison surfaces use recurrence and original AmpliCol as ordered,
    availability-only numerical authorities rather than hard prerequisites.
    """

    if baseline is None:
        return False
    if cell.measurement.execution_mode in {
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    } and baseline.measurement.execution_mode in {
        ExecutionMode.RECURRENCE,
        ExecutionMode.AMPLICOL,
    }:
        return False
    if baseline.measurement.execution_mode is not ExecutionMode.AMPLICOL:
        return True
    return cell.measurement.execution_mode is not ExecutionMode.RECURRENCE


def _unique_cell(
    matches: Sequence[CellSpec],
    *,
    context: str,
) -> CellSpec:
    if len(matches) != 1:
        qualifier = "no" if not matches else "several"
        raise AgreementError(f"{context} has {qualifier} matching report peer cells")
    return matches[0]


def _builtin_recurrence_peer(
    cell: CellSpec,
    cells: Sequence[CellSpec],
) -> CellSpec | None:
    if (
        cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
        or cell.measurement.model is not ModelKey.UFO_SM
    ):
        return None
    return _unique_cell(
        tuple(
            candidate
            for candidate in cells
            if candidate.measurement.execution_mode is ExecutionMode.RECURRENCE
            and candidate.measurement.model is ModelKey.BUILTIN_SM
            and candidate.measurement.accuracy is cell.measurement.accuracy
            and candidate.process == cell.process
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.workload is cell.workload
            and (
                (
                    cell.dataset_id.startswith("matrix_recurrence_")
                    and candidate.dataset_id.startswith("matrix_recurrence_builtin_sm_")
                    and candidate.variant is None
                )
                or (
                    cell.dataset_id == "z_external_sm"
                    and candidate.dataset_id == "z_builtin_sm"
                    and candidate.variant == "recurrence_jit_o2"
                    and cell.variant == "recurrence_jit_o2"
                )
            )
        ),
        context=f"{cell.cell_id} built-in/UFO recurrence agreement",
    )


def _z_recurrence_peer(
    cell: CellSpec,
    cells: Sequence[CellSpec],
) -> CellSpec | None:
    if not cell.dataset_id.startswith("z_") or cell.measurement.execution_mode not in {
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    }:
        return None
    return _unique_cell(
        tuple(
            candidate
            for candidate in cells
            if candidate.dataset_id == cell.dataset_id
            and candidate.measurement.execution_mode is ExecutionMode.RECURRENCE
            and candidate.measurement.model is cell.measurement.model
            and candidate.measurement.accuracy is cell.measurement.accuracy
            and candidate.variant == "recurrence_jit_o2"
            and candidate.process == cell.process
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.workload is cell.workload
        ),
        context=f"{cell.cell_id} Z recurrence agreement",
    )


def _lc_layout_peer(
    cell: CellSpec,
    cells: Sequence[CellSpec],
    *,
    catalog: ReportCatalog,
) -> CellSpec | None:
    if (
        cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
        or (
            cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            and not catalog.legacy_reference_available(cell)
        )
    ):
        return None
    return _unique_cell(
        tuple(
            candidate
            for candidate in cells
            if candidate.dataset_id == cell.dataset_id
            and candidate.process == cell.process
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.measurement == cell.measurement
            and candidate.variant == cell.variant
            and candidate.workload is Workload.SELECTED_FLOW
        ),
        context=f"{cell.cell_id} LC cross-layout agreement",
    )


def _lc_legacy_peer(
    cell: CellSpec,
    cells: Sequence[CellSpec],
    *,
    catalog: ReportCatalog,
) -> CellSpec | None:
    if (
        cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        or cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
        or not catalog.legacy_reference_available(cell)
    ):
        return None
    return _unique_cell(
        tuple(
            candidate
            for candidate in cells
            if candidate.dataset_id == "reference_amplicol_lc"
            and candidate.measurement.execution_mode is ExecutionMode.AMPLICOL
            and candidate.measurement.accuracy is Accuracy.LC
            and candidate.process == cell.process
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.workload is Workload.ALL_FLOW
        ),
        context=f"{cell.cell_id} legacy/pyAmpliCol LC agreement",
    )


def agreement_relative_tolerance(kind: str) -> float:
    """Return the independent tolerance assigned to one direct-edge family."""

    if kind == LC_LEGACY_PYAMPLICOL_COMPONENT:
        return INDEPENDENT_RELATIVE_TOLERANCE
    if kind in DIRECT_AGREEMENT_KINDS:
        return STRICT_RELATIVE_TOLERANCE
    raise AgreementError(f"unsupported direct-agreement kind {kind!r}")


def incoming_agreement_edges(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[AgreementEdge, ...]:
    """Return the exact direct-comparison peers consumed by ``cell``."""

    cells = catalog.measurement_cells()
    edges: list[AgreementEdge] = []
    builtin = _builtin_recurrence_peer(cell, cells)
    if builtin is not None:
        edges.append(AgreementEdge(BUILTIN_UFO_RECURRENCE, builtin, cell))
    recurrence = _z_recurrence_peer(cell, cells)
    if recurrence is not None:
        edges.append(AgreementEdge(Z_RECURRENCE_CROSS_MODE, recurrence, cell))
    layout = _lc_layout_peer(cell, cells, catalog=catalog)
    if layout is not None:
        edges.append(AgreementEdge(LC_CROSS_LAYOUT_COMPONENT, layout, cell))
    legacy = _lc_legacy_peer(cell, cells, catalog=catalog)
    if legacy is not None:
        edges.append(AgreementEdge(LC_LEGACY_PYAMPLICOL_COMPONENT, legacy, cell))
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                DIRECT_AGREEMENT_KINDS.index(edge.kind),
                edge.baseline.cell_id,
            ),
        )
    )


def independent_numerical_authorities(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the canonical ordered independent-authority chain.

    This identity is shared by planning, the isolated worker, strict result
    validation, and late reconciliation.  It deliberately contains only
    availability-optional numerical authorities: recurrence first, followed
    by matching original AmpliCol when the catalog exposes it.
    """

    if cell.measurement.execution_mode not in {
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    }:
        return ()
    candidates: list[CellSpec] = []
    baseline = catalog.validation_baseline_cell(cell)
    recurrence_peer = next(
        (
            edge.baseline
            for edge in incoming_agreement_edges(cell, catalog=catalog)
            if edge.kind == Z_RECURRENCE_CROSS_MODE
        ),
        None,
    )
    recurrence = (
        recurrence_peer
        if recurrence_peer is not None
        else (
            baseline
            if baseline is not None
            and baseline.measurement.execution_mode is ExecutionMode.RECURRENCE
            else None
        )
    )
    if recurrence is not None:
        candidates.append(recurrence)
    legacy = (
        catalog.validation_baseline_cell(recurrence)
        if recurrence is not None
        else baseline
    )
    if (
        legacy is not None
        and legacy.measurement.execution_mode is ExecutionMode.AMPLICOL
        and legacy.cell_id not in {item.cell_id for item in candidates}
    ):
        candidates.append(legacy)
    return tuple(candidates)


def requires_independent_numerical_authority(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> bool:
    """Whether ``cell`` needs an external numerical authority to become OK.

    This is a success-classification rule, not a hard scheduling dependency.
    Standalone absolute benchmark surfaces have no catalog authority chain and
    are certified by their own strict internal validation contract.  Matrix
    and Z compiled/eager comparison surfaces do expose such a chain and remain
    unverified until one of its endpoints agrees.
    """

    return bool(independent_numerical_authorities(cell, catalog=catalog))


def agreement_edges(
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
    maximum_n_final: int | None = None,
) -> tuple[AgreementEdge, ...]:
    selected = tuple(
        cell
        for cell in catalog.measurement_cells()
        if maximum_n_final is None or cell.n_final <= maximum_n_final
    )
    selected_ids = {cell.cell_id for cell in selected}
    return tuple(
        edge
        for cell in sorted(selected, key=lambda item: item.cell_id)
        for edge in incoming_agreement_edges(cell, catalog=catalog)
        if edge.baseline.cell_id in selected_ids
    )


def _lc_component_record(
    cell: CellSpec,
    contract: SelectorContract,
    value: float,
) -> dict[str, object]:
    return {
        "abi": LC_COMMON_COMPONENT_ABI,
        "cell_id": cell.cell_id,
        "value": value,
        "point_digest": contract.point_digest,
        "helicity_ids": list(contract.all_flow_helicity_ids),
        "color_flow_ids": list(contract.selected_color_flow_ids),
    }


def evaluate_lc_common_component(
    runtime: object,
    points: object,
    *,
    cell: CellSpec,
    contract: SelectorContract,
) -> dict[str, object]:
    """Evaluate the selector intersection common to both LC layouts."""

    resolved = runtime.evaluate_resolved(  # type: ignore[attr-defined]
        points,
        precision=16,
        helicities=contract.runtime_all_flow_helicity_ids,
        color_flows=contract.selected_color_flow_ids,
    )
    helicity_ids = tuple(getattr(resolved, "helicity_ids", ()))
    color_ids = tuple(getattr(resolved, "color_ids", ()))
    if helicity_ids != contract.runtime_all_flow_helicity_ids:
        raise AgreementError(
            f"{cell.cell_id} LC common component has a different helicity axis"
        )
    if color_ids != contract.selected_color_flow_ids:
        raise AgreementError(
            f"{cell.cell_id} LC common component has a different color-flow axis"
        )
    values = getattr(resolved, "values", None)
    try:
        raw_value = values[0][0][0]
        if len(values) != 1 or len(values[0]) != 1 or len(values[0][0]) != 1:
            raise IndexError
    except (IndexError, TypeError) as error:
        raise AgreementError(
            f"{cell.cell_id} LC common component is not a scalar"
        ) from error
    return _lc_component_record(
        cell,
        contract,
        _real_nonnegative(raw_value),
    )


def legacy_lc_common_component(
    cell: CellSpec,
    contract: SelectorContract,
    value: float,
) -> dict[str, object]:
    """Shape an independently probed original-AmpliCol LC row component."""

    return _lc_component_record(cell, contract, float(value))


def _measurement_number(
    measurement: Mapping[str, object],
    *,
    edge: AgreementEdge,
    role: str,
) -> float:
    if edge.value_kind == "matrix_element":
        value = measurement.get("matrix_element")
    else:
        validation = measurement.get("validation")
        if not isinstance(validation, Mapping):
            raise AgreementError(
                f"{edge.candidate.cell_id} {role} validation is unavailable"
            )
        observation = validation.get(LC_COMMON_COMPONENT_FIELD)
        if not isinstance(observation, Mapping):
            raise AgreementError(
                f"{edge.candidate.cell_id} {role} LC common component is unavailable"
            )
        value = observation.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgreementError(
            f"{edge.candidate.cell_id} {role} {edge.value_kind} is not numeric"
        )
    number = float(value)
    if not math.isfinite(number):
        raise AgreementError(
            f"{edge.candidate.cell_id} {role} {edge.value_kind} is not finite"
        )
    return number


def _agreement_source_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _measurement_point_digest(measurement: Mapping[str, object]) -> str:
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
    provenance = measurement.get("provenance")
    if isinstance(provenance, Mapping) and "report_momenta" in provenance:
        return point_digest(provenance["report_momenta"])
    raise AgreementError("direct agreement measurement has no point identity")


def _measurement_scale(
    measurement: Mapping[str, object],
    *,
    edge: AgreementEdge,
    role: str,
    number: float,
) -> tuple[float, str, str]:
    validation = measurement.get("validation")
    if edge.value_kind == LC_COMMON_COMPONENT_FIELD and isinstance(validation, Mapping):
        component = validation.get(LC_COMMON_COMPONENT_FIELD)
        if isinstance(component, Mapping):
            return (
                abs(number),
                "authority-component-magnitude",
                _agreement_source_digest(component),
            )
    if edge.value_kind == "matrix_element" and isinstance(validation, Mapping):
        resolved = validation.get("resolved_sum")
        if isinstance(resolved, Mapping):
            records = resolved.get("points")
            source = resolved.get("resolved_source_sha256")
            if (
                isinstance(records, list)
                and len(records) == 1
                and isinstance(records[0], Mapping)
                and isinstance(source, str)
            ):
                raw_scale = records[0].get("candidate_scale")
                if (
                    not isinstance(raw_scale, bool)
                    and isinstance(raw_scale, (int, float))
                    and math.isfinite(float(raw_scale))
                    and float(raw_scale) >= abs(number)
                ):
                    return float(raw_scale), "resolved-component-l1", source
    stable_validation = (
        {key: item for key, item in validation.items() if key != DIRECT_AGREEMENT_FIELD}
        if isinstance(validation, Mapping)
        else validation
    )
    source = _agreement_source_digest(
        {
            "role": role,
            "value_kind": edge.value_kind,
            "value": number,
            "selector_contract": measurement.get("selector_contract"),
            "validation": stable_validation,
            "provenance": measurement.get("provenance"),
        }
    )
    return abs(number), "authority-value-magnitude", source


def direct_agreement_record(
    edge: AgreementEdge,
    *,
    candidate_measurement: Mapping[str, object],
    baseline_measurement: Mapping[str, object],
) -> dict[str, object]:
    candidate = _measurement_number(
        candidate_measurement,
        edge=edge,
        role="candidate",
    )
    baseline = _measurement_number(
        baseline_measurement,
        edge=edge,
        role="baseline",
    )
    candidate_scale, candidate_scale_source, candidate_source = _measurement_scale(
        candidate_measurement,
        edge=edge,
        role="candidate",
        number=candidate,
    )
    baseline_scale, baseline_scale_source, baseline_source = _measurement_scale(
        baseline_measurement,
        edge=edge,
        role="baseline",
        number=baseline,
    )
    candidate_point = _measurement_point_digest(candidate_measurement)
    baseline_point = _measurement_point_digest(baseline_measurement)
    if candidate_point != baseline_point:
        raise AgreementError("direct agreement points differ")
    selector_identity = {
        "candidate_cell_id": edge.candidate.cell_id,
        "baseline_cell_id": edge.baseline.cell_id,
        "candidate_accuracy": edge.candidate.measurement.accuracy.value,
        "baseline_accuracy": edge.baseline.measurement.accuracy.value,
        "candidate_workload": edge.candidate.workload.value,
        "baseline_workload": edge.baseline.workload.value,
        "value_kind": edge.value_kind,
        "candidate_selector": candidate_measurement.get("selector_contract"),
        "baseline_selector": baseline_measurement.get("selector_contract"),
    }
    comparison = pointwise_validation(
        candidate,
        baseline,
        relative_tolerance=edge.relative_tolerance,
        candidate_scale=candidate_scale,
        baseline_scale=baseline_scale,
        candidate_scale_source=candidate_scale_source,
        baseline_scale_source=baseline_scale_source,
        comparison_binding={
            "point_digest": candidate_point,
            "selector_component_identity": selector_identity,
            "selector_component_sha256": _agreement_source_digest(selector_identity),
            "candidate_source_sha256": candidate_source,
            "baseline_source_sha256": baseline_source,
        },
    )
    comparison.pop("abi")
    return {
        "abi": DIRECT_AGREEMENT_V2_ABI,
        "edge_kind": edge.kind,
        "value_kind": edge.value_kind,
        "baseline_cell_id": edge.baseline.cell_id,
        "candidate_cell_id": edge.candidate.cell_id,
        **comparison,
    }


def validate_direct_agreement_records(
    value: object,
    *,
    expected_candidate_id: str | None = None,
) -> None:
    """Validate the complete per-measurement direct-agreement wire contract."""

    if not isinstance(value, list):
        raise ValueError(f"{DIRECT_AGREEMENT_FIELD} must be an array")
    identity_fields = {
        "abi",
        "edge_kind",
        "value_kind",
        "baseline_cell_id",
        "candidate_cell_id",
    }
    v1_fields = identity_fields | {
        "status",
        "candidate",
        "baseline",
        "absolute_difference",
        "relative_difference",
        "relative_tolerance",
        "absolute_tolerance",
    }
    v2_fields = identity_fields | {
        "status",
        "candidate",
        "baseline",
        "candidate_scale",
        "baseline_scale",
        "candidate_scale_source",
        "baseline_scale_source",
        "comparison_scale",
        "absolute_difference",
        "relative_difference",
        "conditioned_residual",
        "error_bound",
        "relative_tolerance",
        "comparison_binding",
    }
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{DIRECT_AGREEMENT_FIELD}[{index}] must be an object")
        abi = raw.get("abi")
        expected_fields = v2_fields if abi == DIRECT_AGREEMENT_V2_ABI else v1_fields
        if set(raw) != expected_fields:
            raise ValueError(
                f"{DIRECT_AGREEMENT_FIELD}[{index}] fields differ from the ABI"
            )
        kind = raw.get("edge_kind")
        if not isinstance(kind, str) or kind not in DIRECT_AGREEMENT_KINDS:
            raise ValueError(
                f"{DIRECT_AGREEMENT_FIELD}[{index}] edge kind is unsupported"
            )
        expected_value_kind = (
            LC_COMMON_COMPONENT_FIELD
            if kind in {LC_CROSS_LAYOUT_COMPONENT, LC_LEGACY_PYAMPLICOL_COMPONENT}
            else "matrix_element"
        )
        baseline_id = raw.get("baseline_cell_id")
        candidate_id = raw.get("candidate_cell_id")
        if (
            abi not in {DIRECT_AGREEMENT_V2_ABI, DIRECT_AGREEMENT_V1_ABI}
            or raw.get("value_kind") != expected_value_kind
            or not isinstance(baseline_id, str)
            or not baseline_id
            or not isinstance(candidate_id, str)
            or not candidate_id
            or (
                expected_candidate_id is not None
                and candidate_id != expected_candidate_id
            )
        ):
            raise ValueError(f"{DIRECT_AGREEMENT_FIELD}[{index}] identity is invalid")
        key = (kind, baseline_id, candidate_id)
        if key in seen:
            raise ValueError(f"{DIRECT_AGREEMENT_FIELD}[{index}] duplicates an edge")
        seen.add(key)
        expected_relative_tolerance = agreement_relative_tolerance(kind)
        comparison = {
            key: item for key, item in raw.items() if key not in identity_fields
        }
        if abi == DIRECT_AGREEMENT_V2_ABI:
            comparison["abi"] = CONDITIONED_COMPARISON_ABI
        try:
            validate_conditioned_comparison_record(
                comparison,
                require_binding=abi == DIRECT_AGREEMENT_V2_ABI,
            )
        except ValueError as error:
            raise ValueError(
                f"{DIRECT_AGREEMENT_FIELD}[{index}] numerical record is invalid"
            ) from error
        if float(comparison["relative_tolerance"]) != expected_relative_tolerance:
            raise ValueError(f"{DIRECT_AGREEMENT_FIELD}[{index}] tolerance is invalid")
        if (
            abi == DIRECT_AGREEMENT_V1_ABI
            and float(comparison["absolute_tolerance"]) != STRICT_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                f"{DIRECT_AGREEMENT_FIELD}[{index}] legacy tolerance is invalid"
            )
        if raw.get("status") != ResultStatus.OK.value:
            raise ValueError(
                f"{DIRECT_AGREEMENT_FIELD}[{index}] agreement is not successful"
            )


def validate_lc_common_component(
    value: object,
    *,
    expected_cell_id: str | None = None,
    selector_contract: object = None,
) -> None:
    """Validate one LC component record before it is admitted to the cache."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{LC_COMMON_COMPONENT_FIELD} must be an object")
    expected_fields = {
        "abi",
        "cell_id",
        "value",
        "point_digest",
        "helicity_ids",
        "color_flow_ids",
    }
    if set(value) != expected_fields:
        raise ValueError(f"{LC_COMMON_COMPONENT_FIELD} fields differ from the ABI")
    cell_id = value.get("cell_id")
    number = value.get("value")
    point_digest = value.get("point_digest")
    helicity_ids = value.get("helicity_ids")
    color_flow_ids = value.get("color_flow_ids")
    if (
        value.get("abi") != LC_COMMON_COMPONENT_ABI
        or not isinstance(cell_id, str)
        or not cell_id
        or (expected_cell_id is not None and cell_id != expected_cell_id)
        or isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
        or not isinstance(point_digest, str)
        or len(point_digest) != 64
        or any(character not in "0123456789abcdef" for character in point_digest)
        or not isinstance(helicity_ids, list)
        or not helicity_ids
        or any(not isinstance(item, str) or not item for item in helicity_ids)
        or len(set(helicity_ids)) != len(helicity_ids)
        or not isinstance(color_flow_ids, list)
        or not color_flow_ids
        or any(not isinstance(item, str) or not item for item in color_flow_ids)
        or len(set(color_flow_ids)) != len(color_flow_ids)
    ):
        raise ValueError(f"{LC_COMMON_COMPONENT_FIELD} record is invalid")
    if selector_contract is not None:
        try:
            selector = SelectorContract.from_mapping(
                selector_contract  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("selector contract is invalid") from error
        if (
            point_digest != selector.point_digest
            or helicity_ids != list(selector.all_flow_helicity_ids)
            or color_flow_ids != list(selector.selected_color_flow_ids)
        ):
            raise ValueError(
                f"{LC_COMMON_COMPONENT_FIELD} is not bound to the selector"
            )


def attach_direct_agreements(
    cell: CellSpec,
    measurement: dict[str, object],
    peers: Mapping[str, Mapping[str, object]],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> None:
    """Attach every incoming direct edge and make any failure terminal."""

    initial_status = measurement.get("status")
    if initial_status not in {
        ResultStatus.OK.value,
        ResultStatus.UNVERIFIED.value,
    }:
        return
    validation = measurement.get("validation")
    if not isinstance(validation, Mapping):
        raise AgreementError(f"{cell.cell_id} has no validation record")
    mutable_validation = dict(validation)
    expected = incoming_agreement_edges(cell, catalog=catalog)
    expected_peer_ids = {edge.baseline.cell_id for edge in expected}
    required_peer_ids = {edge.baseline.cell_id for edge in expected if edge.required}
    observed_peer_ids = set(peers)
    if not required_peer_ids.issubset(observed_peer_ids) or not (
        observed_peer_ids <= expected_peer_ids
    ):
        raise AgreementError(
            f"{cell.cell_id} direct-agreement peers differ: "
            f"required={sorted(required_peer_ids)}, "
            f"allowed={sorted(expected_peer_ids)}, "
            f"observed={sorted(observed_peer_ids)}"
        )
    records: list[dict[str, object]] = []
    for edge in expected:
        if edge.baseline.cell_id not in peers:
            assert not edge.required
            continue
        peer = peers[edge.baseline.cell_id]
        if peer.get("status") != ResultStatus.OK.value:
            raise AgreementError(
                f"{cell.cell_id} peer {edge.baseline.cell_id} is not successful"
            )
        if measurement.get("selector_contract") != peer.get("selector_contract"):
            raise AgreementError(
                f"{cell.cell_id} selector contract differs from direct peer "
                f"{edge.baseline.cell_id}"
            )
        records.append(
            direct_agreement_record(
                edge,
                candidate_measurement=measurement,
                baseline_measurement=peer,
            )
        )
    mutable_validation[DIRECT_AGREEMENT_FIELD] = records
    failed = any(record["status"] != ResultStatus.OK.value for record in records)
    if failed:
        mutable_validation["status"] = ResultStatus.VALIDATION_FAILED.value
        measurement["status"] = ResultStatus.VALIDATION_FAILED.value
        measurement["failure"] = {
            "kind": "MeasurementValidationError",
            "message": "required direct numerical agreement failed",
        }
    elif initial_status == ResultStatus.UNVERIFIED.value:
        # Hard model/layout/provider agreements remain mandatory even when the
        # independent recurrence/AmpliCol numerical authority is unavailable.
        # Passing them is necessary but deliberately cannot promote a
        # diagnostic candidate to a reusable success.
        mutable_validation["status"] = ResultStatus.UNVERIFIED.value
    measurement["validation"] = mutable_validation


__all__ = [
    "BUILTIN_UFO_RECURRENCE",
    "DIRECT_AGREEMENT_ABI",
    "DIRECT_AGREEMENT_FIELD",
    "DIRECT_AGREEMENT_KINDS",
    "DIRECT_AGREEMENT_V2_ABI",
    "INDEPENDENT_AUTHORITY_ABI",
    "INDEPENDENT_AUTHORITY_FIELD",
    "LC_COMMON_COMPONENT_ABI",
    "LC_COMMON_COMPONENT_FIELD",
    "LC_CROSS_LAYOUT_COMPONENT",
    "LC_LEGACY_PYAMPLICOL_COMPONENT",
    "STRICT_ABSOLUTE_TOLERANCE",
    "STRICT_RELATIVE_TOLERANCE",
    "Z_RECURRENCE_CROSS_MODE",
    "AgreementEdge",
    "AgreementError",
    "agreement_edges",
    "agreement_relative_tolerance",
    "attach_direct_agreements",
    "direct_agreement_record",
    "evaluate_lc_common_component",
    "incoming_agreement_edges",
    "independent_numerical_authorities",
    "legacy_lc_common_component",
    "requires_independent_numerical_authority",
    "validate_direct_agreement_records",
    "validate_lc_common_component",
    "validation_baseline_is_required",
]
