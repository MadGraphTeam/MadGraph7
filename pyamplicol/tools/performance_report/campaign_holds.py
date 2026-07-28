# SPDX-License-Identifier: 0BSD
"""Fail-closed re-admission of dependency-held campaign cells."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .agreements import incoming_agreement_edges
from .artifacts import ArtifactStore, CurrentRecord
from .cache import validate_measurement
from .catalog import REPORT_CATALOG, ReportCatalog
from .models import CellSpec, ResultStatus

PRIOR_HELD_DISPOSITION_ABI = "pyamplicol-prior-held-disposition-v1"
PRIOR_HELD_HISTORY_ABI = "pyamplicol-prior-held-history-v1"
DEPENDENCY_HOLD_REASON = "authenticated non-ok dependency"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")


class PriorHeldEligibilityError(RuntimeError):
    """A historical hold could not be classified without guessing."""


@dataclass(frozen=True, slots=True)
class PriorHeldDisposition:
    """One authenticated historical-hold eligibility decision."""

    cell_id: str
    eligible: bool
    reason: str
    historical_reasons: tuple[str, ...]
    historical_observations: tuple[tuple[str, str, str], ...]
    target_attempt_ids: tuple[str, ...]
    prerequisite_ids: tuple[str, ...]
    blocking_prerequisites: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "abi": PRIOR_HELD_DISPOSITION_ABI,
            "cell_id": self.cell_id,
            "eligible": self.eligible,
            "reason": self.reason,
            "historical_reasons": list(self.historical_reasons),
            "historical_observations": [
                {
                    "reason": reason,
                    "summary_path": summary_path,
                    "summary_sha256": summary_sha256,
                }
                for summary_path, summary_sha256, reason in (
                    self.historical_observations
                )
            ],
            "target_attempt_ids": list(self.target_attempt_ids),
            "prerequisite_ids": list(self.prerequisite_ids),
            "blocking_prerequisites": [
                {"cell_id": cell_id, "state": state}
                for cell_id, state in self.blocking_prerequisites
            ],
        }


def prior_held_history_record(
    cell_id: str,
    observations: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Build one complete, deterministic historical-hold record."""

    if not isinstance(cell_id, str) or not cell_id:
        raise PriorHeldEligibilityError(
            "historical hold record has an invalid catalog cell identity"
        )
    normalized: list[dict[str, str]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {
            "reason",
            "summary_path",
            "summary_sha256",
        }:
            raise PriorHeldEligibilityError(
                f"{cell_id} historical hold observation {index} is invalid"
            )
        reason = observation.get("reason")
        summary_path = observation.get("summary_path")
        summary_sha256 = observation.get("summary_sha256")
        if (
            not isinstance(reason, str)
            or not reason
            or not isinstance(summary_path, str)
            or not summary_path
            or not isinstance(summary_sha256, str)
            or _SHA256_RE.fullmatch(summary_sha256) is None
        ):
            raise PriorHeldEligibilityError(
                f"{cell_id} historical hold observation {index} is invalid"
            )
        normalized.append(
            {
                "reason": reason,
                "summary_path": summary_path,
                "summary_sha256": summary_sha256,
            }
        )
    if not normalized:
        raise PriorHeldEligibilityError(f"{cell_id} historical hold history is empty")
    normalized.sort(key=lambda observation: observation["summary_path"])
    summary_paths = [observation["summary_path"] for observation in normalized]
    if len(set(summary_paths)) != len(summary_paths):
        raise PriorHeldEligibilityError(
            f"{cell_id} historical hold history repeats a summary"
        )
    return {
        "abi": PRIOR_HELD_HISTORY_ABI,
        "cell_id": cell_id,
        "observations": normalized,
        "reasons": [observation["reason"] for observation in normalized],
    }


def _validated_historical_history(
    cell_id: str,
    raw_record: object,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    if not isinstance(raw_record, Mapping):
        raise PriorHeldEligibilityError(
            f"{cell_id} historical hold record must be an object"
        )
    observations = raw_record.get("observations")
    if (
        raw_record.get("abi") != PRIOR_HELD_HISTORY_ABI
        or raw_record.get("cell_id") != cell_id
        or not isinstance(observations, list)
    ):
        raise PriorHeldEligibilityError(f"{cell_id} historical hold record is invalid")
    rebuilt = prior_held_history_record(cell_id, observations)
    if dict(raw_record) != rebuilt:
        raise PriorHeldEligibilityError(
            f"{cell_id} historical hold record is inconsistent"
        )
    reasons = rebuilt["reasons"]
    assert isinstance(reasons, list)
    normalized = rebuilt["observations"]
    assert isinstance(normalized, list)
    return (
        tuple(str(reason) for reason in reasons),
        tuple(
            (
                str(observation["summary_path"]),
                str(observation["summary_sha256"]),
                str(observation["reason"]),
            )
            for observation in normalized
        ),
    )


def catalog_prerequisite_cells(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return direct baseline and agreement/equivalence prerequisites."""

    baseline = catalog.validation_baseline_cell(cell)
    peers = tuple(
        edge.baseline for edge in incoming_agreement_edges(cell, catalog=catalog)
    )
    return tuple(
        sorted(
            {
                prerequisite.cell_id: prerequisite
                for prerequisite in (
                    *((baseline,) if baseline is not None else ()),
                    *peers,
                )
            }.values(),
            key=lambda prerequisite: prerequisite.cell_id,
        )
    )


def catalog_prerequisite_closure(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    """Return the complete acyclic prerequisite closure for one cell."""

    resolved: dict[str, CellSpec] = {}
    visiting: set[str] = set()

    def visit(candidate: CellSpec) -> None:
        if candidate.cell_id in visiting:
            raise PriorHeldEligibilityError(
                f"report prerequisite cycle reaches {candidate.cell_id!r}"
            )
        visiting.add(candidate.cell_id)
        try:
            for prerequisite in catalog_prerequisite_cells(
                candidate,
                catalog=catalog,
            ):
                if prerequisite.cell_id not in resolved:
                    visit(prerequisite)
                    resolved[prerequisite.cell_id] = prerequisite
        finally:
            visiting.remove(candidate.cell_id)

    visit(cell)
    return tuple(resolved[cell_id] for cell_id in sorted(resolved))


def _catalog_cells_by_id(catalog: ReportCatalog) -> dict[str, CellSpec]:
    cells = catalog.measurement_cells()
    by_id = {cell.cell_id: cell for cell in cells}
    if len(by_id) != len(cells):
        raise PriorHeldEligibilityError(
            "report catalog contains duplicate cell identities"
        )
    return by_id


def _validated_current_state(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    expected_source_revision: str | None,
    expected_source_tree: str | None,
    authenticate_current: Callable[[CellSpec, CurrentRecord], bool] | None,
) -> str:
    current = store.load_current(cell.cell_id, missing_ok=True)
    if current is None:
        return "missing"
    try:
        validate_measurement(current.result, expected_cell=cell)
    except ValueError as error:
        raise PriorHeldEligibilityError(
            f"{cell.cell_id} current measurement is invalid: {error}"
        ) from error
    raw_status = current.result.get("status")
    try:
        status = ResultStatus(str(raw_status))
    except ValueError as error:
        raise PriorHeldEligibilityError(
            f"{cell.cell_id} current status is unknown"
        ) from error
    if status is not ResultStatus.OK:
        return status.value
    if expected_source_revision is not None:
        provenance = current.result.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("report_source_revision") != expected_source_revision
            or provenance.get("report_measured_source_revision")
            != expected_source_revision
            or (
                expected_source_tree is not None
                and (
                    provenance.get("report_source_tree") != expected_source_tree
                    or provenance.get("report_measured_source_tree")
                    != expected_source_tree
                )
            )
        ):
            return "source-stale"
    if authenticate_current is not None:
        authenticated = authenticate_current(cell, current)
        if authenticated is not True:
            return "unauthenticated"
    return ResultStatus.OK.value


def classify_prior_held_cells(
    store: ArtifactStore,
    prior_held_records: Mapping[str, object],
    *,
    active_scoped_hold_ids: Iterable[str] = (),
    catalog: ReportCatalog = REPORT_CATALOG,
    expected_source_revision: str | None = None,
    expected_source_tree: str | None = None,
    authenticate_current: Callable[[CellSpec, CurrentRecord], bool] | None = None,
) -> dict[str, PriorHeldDisposition]:
    """Classify historical holds without permanently excluding resolved cells.

    Only histories in which every reason is
    :data:`DEPENDENCY_HOLD_REASON` can become eligible. Eligibility also
    requires an explicit source-authentication policy, a target that has never
    acquired an attempt or current, no active scoped defect hold, and an
    immutable, schema-valid ``ok`` current for every transitive baseline or
    direct agreement/equivalence prerequisite.
    """

    prior_ids = tuple(prior_held_records)
    active_ids = frozenset(active_scoped_hold_ids)
    if any(
        not isinstance(cell_id, str) or not cell_id
        for cell_id in (*prior_ids, *active_ids)
    ):
        raise PriorHeldEligibilityError(
            "hold input contains an invalid catalog cell identity"
        )
    if (expected_source_revision is None) != (expected_source_tree is None):
        raise PriorHeldEligibilityError(
            "expected source revision and tree must be provided together"
        )
    if expected_source_revision is not None and (
        not isinstance(expected_source_revision, str)
        or _GIT_OBJECT_RE.fullmatch(expected_source_revision) is None
        or not isinstance(expected_source_tree, str)
        or _GIT_OBJECT_RE.fullmatch(expected_source_tree) is None
    ):
        raise PriorHeldEligibilityError(
            "expected source revision and tree must be Git object identities"
        )
    if expected_source_revision is None and authenticate_current is None:
        raise PriorHeldEligibilityError(
            "current source authentication policy is required"
        )
    by_id = _catalog_cells_by_id(catalog)
    unknown = sorted((set(prior_ids) | active_ids) - set(by_id))
    if unknown:
        raise PriorHeldEligibilityError(
            f"hold input contains unknown catalog cells: {unknown}"
        )

    dispositions: dict[str, PriorHeldDisposition] = {}
    for cell_id in sorted(prior_ids):
        cell = by_id[cell_id]
        historical_reasons, historical_observations = _validated_historical_history(
            cell_id,
            prior_held_records[cell_id],
        )
        with store.cell_lock(cell_id):
            current = store.load_current(cell_id, missing_ok=True)
            attempt_ids = store.cell_attempt_ids(cell_id)
        if current is not None and current.attempt_id not in attempt_ids:
            raise PriorHeldEligibilityError(
                f"{cell_id} current is absent from its attempt inventory"
            )
        prerequisites = catalog_prerequisite_closure(cell, catalog=catalog)
        prerequisite_ids = tuple(prerequisite.cell_id for prerequisite in prerequisites)

        if attempt_ids or current is not None:
            dispositions[cell_id] = PriorHeldDisposition(
                cell_id,
                False,
                "target-already-attempted",
                historical_reasons,
                historical_observations,
                attempt_ids,
                prerequisite_ids,
                (),
            )
            continue
        if cell_id in active_ids:
            dispositions[cell_id] = PriorHeldDisposition(
                cell_id,
                False,
                "active-scoped-defect-hold",
                historical_reasons,
                historical_observations,
                (),
                prerequisite_ids,
                (),
            )
            continue
        if any(reason != DEPENDENCY_HOLD_REASON for reason in historical_reasons):
            dispositions[cell_id] = PriorHeldDisposition(
                cell_id,
                False,
                "historical-hold-not-dependency",
                historical_reasons,
                historical_observations,
                (),
                prerequisite_ids,
                (),
            )
            continue

        blocking: list[tuple[str, str]] = []
        for prerequisite in prerequisites:
            if prerequisite.cell_id in active_ids:
                blocking.append(
                    (
                        prerequisite.cell_id,
                        "active-scoped-defect-hold",
                    )
                )
                continue
            state = _validated_current_state(
                store,
                prerequisite,
                expected_source_revision=expected_source_revision,
                expected_source_tree=expected_source_tree,
                authenticate_current=authenticate_current,
            )
            if state != ResultStatus.OK.value:
                blocking.append((prerequisite.cell_id, state))
        dispositions[cell_id] = PriorHeldDisposition(
            cell_id,
            not blocking,
            "eligible" if not blocking else "prerequisite-not-ok",
            historical_reasons,
            historical_observations,
            (),
            prerequisite_ids,
            tuple(blocking),
        )
    return dispositions


def active_prior_held_ids(
    store: ArtifactStore,
    prior_held_records: Mapping[str, object],
    *,
    active_scoped_hold_ids: Iterable[str] = (),
    catalog: ReportCatalog = REPORT_CATALOG,
    expected_source_revision: str | None = None,
    expected_source_tree: str | None = None,
    authenticate_current: Callable[[CellSpec, CurrentRecord], bool] | None = None,
) -> frozenset[str]:
    """Return only historical holds that are still fail-closed."""

    dispositions = classify_prior_held_cells(
        store,
        prior_held_records,
        active_scoped_hold_ids=active_scoped_hold_ids,
        catalog=catalog,
        expected_source_revision=expected_source_revision,
        expected_source_tree=expected_source_tree,
        authenticate_current=authenticate_current,
    )
    return frozenset(
        cell_id
        for cell_id, disposition in dispositions.items()
        if not disposition.eligible
    )


__all__ = [
    "DEPENDENCY_HOLD_REASON",
    "PRIOR_HELD_DISPOSITION_ABI",
    "PRIOR_HELD_HISTORY_ABI",
    "PriorHeldDisposition",
    "PriorHeldEligibilityError",
    "active_prior_held_ids",
    "catalog_prerequisite_cells",
    "catalog_prerequisite_closure",
    "classify_prior_held_cells",
    "prior_held_history_record",
]
