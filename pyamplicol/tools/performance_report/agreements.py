# SPDX-License-Identifier: 0BSD
"""Direct numerical-agreement edges required by the publication contract."""

from __future__ import annotations

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
    RELATIVE_TOLERANCE,
    SelectorContract,
    _real_nonnegative,
    pointwise_validation,
)

DIRECT_AGREEMENT_ABI = "pyamplicol-report-direct-agreement-v1"
LC_COMMON_COMPONENT_ABI = "pyamplicol-report-lc-common-component-v1"
BUILTIN_UFO_RECURRENCE = "builtin-ufo-recurrence"
Z_RECURRENCE_CROSS_MODE = "z-recurrence-cross-mode"
LC_CROSS_LAYOUT_COMPONENT = "lc-cross-layout-component"
DIRECT_AGREEMENT_KINDS = (
    BUILTIN_UFO_RECURRENCE,
    Z_RECURRENCE_CROSS_MODE,
    LC_CROSS_LAYOUT_COMPONENT,
)
DIRECT_AGREEMENT_FIELD = "direct_agreements"
LC_COMMON_COMPONENT_FIELD = "lc_common_component"
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
            if self.kind == LC_CROSS_LAYOUT_COMPONENT
            else "matrix_element"
        )


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
                    and candidate.dataset_id.startswith(
                        "matrix_recurrence_builtin_sm_"
                    )
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
    if (
        not cell.dataset_id.startswith("z_")
        or cell.measurement.execution_mode
        not in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    ):
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
) -> CellSpec | None:
    if (
        cell.measurement.accuracy is not Accuracy.LC
        or cell.workload is not Workload.ALL_FLOW
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
    layout = _lc_layout_peer(cell, cells)
    if layout is not None:
        edges.append(AgreementEdge(LC_CROSS_LAYOUT_COMPONENT, layout, cell))
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                DIRECT_AGREEMENT_KINDS.index(edge.kind),
                edge.baseline.cell_id,
            ),
        )
    )


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
        helicities=contract.all_flow_helicity_ids,
        color_flows=contract.selected_color_flow_ids,
    )
    helicity_ids = tuple(getattr(resolved, "helicity_ids", ()))
    color_ids = tuple(getattr(resolved, "color_ids", ()))
    if helicity_ids != contract.all_flow_helicity_ids:
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
        if (
            len(values) != 1
            or len(values[0]) != 1
            or len(values[0][0]) != 1
        ):
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


def _direct_record(
    edge: AgreementEdge,
    *,
    candidate: float,
    baseline: float,
) -> dict[str, object]:
    return {
        "abi": DIRECT_AGREEMENT_ABI,
        "edge_kind": edge.kind,
        "value_kind": edge.value_kind,
        "baseline_cell_id": edge.baseline.cell_id,
        "candidate_cell_id": edge.candidate.cell_id,
        **pointwise_validation(
            candidate,
            baseline,
            relative_tolerance=STRICT_RELATIVE_TOLERANCE,
            absolute_tolerance=STRICT_ABSOLUTE_TOLERANCE,
        ),
    }


def attach_direct_agreements(
    cell: CellSpec,
    measurement: dict[str, object],
    peers: Mapping[str, Mapping[str, object]],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> None:
    """Attach every incoming direct edge and make any failure terminal."""

    if measurement.get("status") != ResultStatus.OK.value:
        return
    validation = measurement.get("validation")
    if not isinstance(validation, Mapping):
        raise AgreementError(f"{cell.cell_id} has no validation record")
    mutable_validation = dict(validation)
    expected = incoming_agreement_edges(cell, catalog=catalog)
    expected_peer_ids = {edge.baseline.cell_id for edge in expected}
    if set(peers) != expected_peer_ids:
        raise AgreementError(
            f"{cell.cell_id} direct-agreement peers differ: "
            f"expected={sorted(expected_peer_ids)}, observed={sorted(peers)}"
        )
    records: list[dict[str, object]] = []
    for edge in expected:
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
            _direct_record(
                edge,
                candidate=_measurement_number(
                    measurement,
                    edge=edge,
                    role="candidate",
                ),
                baseline=_measurement_number(
                    peer,
                    edge=edge,
                    role="baseline",
                ),
            )
        )
    mutable_validation[DIRECT_AGREEMENT_FIELD] = records
    failed = any(
        record["status"] != ResultStatus.OK.value for record in records
    )
    if failed:
        mutable_validation["status"] = ResultStatus.VALIDATION_FAILED.value
        measurement["status"] = ResultStatus.VALIDATION_FAILED.value
        measurement["failure"] = {
            "kind": "MeasurementValidationError",
            "message": "required direct numerical agreement failed",
        }
    measurement["validation"] = mutable_validation


__all__ = [
    "BUILTIN_UFO_RECURRENCE",
    "DIRECT_AGREEMENT_ABI",
    "DIRECT_AGREEMENT_FIELD",
    "DIRECT_AGREEMENT_KINDS",
    "LC_COMMON_COMPONENT_ABI",
    "LC_COMMON_COMPONENT_FIELD",
    "LC_CROSS_LAYOUT_COMPONENT",
    "STRICT_ABSOLUTE_TOLERANCE",
    "STRICT_RELATIVE_TOLERANCE",
    "Z_RECURRENCE_CROSS_MODE",
    "AgreementEdge",
    "AgreementError",
    "agreement_edges",
    "attach_direct_agreements",
    "evaluate_lc_common_component",
    "incoming_agreement_edges",
    "legacy_lc_common_component",
]
