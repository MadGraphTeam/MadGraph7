# SPDX-License-Identifier: 0BSD
"""Authenticated contracts for narrowly scoped performance studies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .cache import validate_measurement
from .campaign_policy import (
    GENERATION_PHASE_EVIDENCE_ABI,
    MACBOOK_M3_MEMORY_LIMIT_BYTES,
    MACBOOK_M3_PROFILE,
    MACBOOK_M3_Z_TABLE_F_GENERATION_LIMIT_SECONDS,
    MACBOOK_M3_Z_TABLE_F_POLICY,
    PolicyMeasurementState,
    policy_status_label,
    validate_policy_measurement,
)
from .catalog import REPORT_CATALOG
from .models import Accuracy, ResultStatus, Workload
from .publication import publication_measurement_matches_current
from .resources import PROCESS_TREE_MEMORY_METRIC_ABI
from .source_identity import require_eligible_report_source

if TYPE_CHECKING:
    from .service import ReportService

Z_TABLE_F_STUDY_CONTRACT_SCHEMA = "pyamplicol-z-table-f-study-contract-v1"
Z_TABLE_F_STUDY_ID = "macbook-m3-z-table-f"
Z_TABLE_F_CONTRACT_MINIMUM_N = 8
Z_TABLE_F_SELECTION_ABI = "pyamplicol-explicit-single-cell-selection-v1"
Z_TABLE_F_ATTEMPT_BINDING_ABI = (
    "pyamplicol-z-table-f-attempt-binding-v1"
)
Z_TABLE_F_MEMORY_GUARD = (
    "max(aggregate-process-tree-rss,darwin-physical-footprint)"
)
_SHA40 = frozenset("0123456789abcdef")
_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "study_id",
        "campaign_policy",
        "policy_profile",
        "measured_source",
        "policy_wrapper",
        "allowed_cell_ids",
        "selection_abi",
        "memory_guard",
        "generation_guard",
        "retained_prior_evidence",
        "sha256",
    }
)


class StudyContractError(ValueError):
    """A study contract is missing, stale, or internally inconsistent."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _git_value(root: Path, expression: str) -> str:
    completed = subprocess.run(
        ("git", "-C", os.fspath(root), "rev-parse", expression),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in _SHA40 for character in value)
    ):
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise StudyContractError(
            f"cannot authenticate Git identity {expression!r}: {detail}"
        )
    return value


def _clean_git_identity(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve(strict=True)
    completed = subprocess.run(
        (
            "git",
            "-C",
            os.fspath(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise StudyContractError(
            "cannot inspect policy-wrapper worktree cleanliness"
        )
    if completed.stdout:
        raise StudyContractError(
            "policy-wrapper worktree must be clean before contract creation"
        )
    return {
        "revision": _git_value(resolved, "HEAD^{commit}"),
        "tree": _git_value(resolved, "HEAD^{tree}"),
    }


def z_table_f_cell_ids() -> tuple[str, ...]:
    """Return the exact newly supervised Z-table declaration scope."""

    return tuple(
        sorted(
            cell.cell_id
            for cell in REPORT_CATALOG.measurement_cells()
            if cell.dataset_id
            in {
                "z_builtin_sm",
                "reference_amplicol_lc",
            }
            and cell.process_key == "dd_z_jets"
            and Z_TABLE_F_CONTRACT_MINIMUM_N <= cell.n_final <= 9
            and cell.measurement.accuracy is Accuracy.LC
            and cell.workload
            in {
                Workload.SELECTED_FLOW,
                Workload.ALL_FLOW,
            }
        )
    )


def create_z_table_f_study_contract(
    measured_root: Path,
    policy_wrapper_root: Path,
) -> dict[str, object]:
    """Create a digest-bound contract from two clean, separate Git identities."""

    source = require_eligible_report_source(
        measured_root.expanduser().resolve(strict=True)
    )
    wrapper = _clean_git_identity(policy_wrapper_root)
    body: dict[str, object] = {
        "schema": Z_TABLE_F_STUDY_CONTRACT_SCHEMA,
        "study_id": Z_TABLE_F_STUDY_ID,
        "campaign_policy": MACBOOK_M3_Z_TABLE_F_POLICY.as_manifest(),
        "policy_profile": MACBOOK_M3_PROFILE,
        "measured_source": {
            "revision": source.revision,
            "tree": source.tree,
        },
        "policy_wrapper": wrapper,
        "allowed_cell_ids": list(z_table_f_cell_ids()),
        "selection_abi": Z_TABLE_F_SELECTION_ABI,
        "memory_guard": {
            "metric_abi": PROCESS_TREE_MEMORY_METRIC_ABI,
            "metric": Z_TABLE_F_MEMORY_GUARD,
            "limit_bytes": MACBOOK_M3_MEMORY_LIMIT_BYTES,
        },
        "generation_guard": {
            "evidence_abi": GENERATION_PHASE_EVIDENCE_ABI,
            "limit_seconds": (
                MACBOOK_M3_Z_TABLE_F_GENERATION_LIMIT_SECONDS
            ),
            "scope": "every legacy, selected-flow, and all-flow generation",
        },
        "retained_prior_evidence": {
            "n_final": list(range(1, Z_TABLE_F_CONTRACT_MINIMUM_N)),
            "treatment": (
                "outside this contract; retain each original attempt, "
                "provenance record, and resource ABI unchanged"
            ),
            "memory_guard_interpretation": (
                "legacy RSS-only evidence remains legacy and is not "
                "relabeled as Darwin physical-footprint or exact-decimal-"
                "30GB evidence"
            ),
        },
    }
    return {**body, "sha256": _canonical_sha256(body)}


def write_z_table_f_study_contract(
    path: Path,
    measured_root: Path,
    policy_wrapper_root: Path,
) -> dict[str, object]:
    """Write a new contract without overwriting prior study evidence."""

    destination = path.expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise StudyContractError(
            f"study contract already exists: {destination}"
        )
    payload = create_z_table_f_study_contract(
        measured_root,
        policy_wrapper_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.new")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, destination)
    return payload


def load_z_table_f_study_contract(
    path: Path,
    measured_root: Path,
    policy_wrapper_root: Path,
) -> dict[str, object]:
    """Load a contract and reauthenticate both source and wrapper identities."""

    try:
        raw = json.loads(
            path.expanduser().resolve(strict=True).read_text(
                encoding="ascii"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyContractError(
            f"cannot read Z-table study contract: {error}"
        ) from error
    if not isinstance(raw, dict) or set(raw) != _CONTRACT_FIELDS:
        raise StudyContractError("Z-table study contract fields are invalid")
    body = {key: value for key, value in raw.items() if key != "sha256"}
    if raw.get("sha256") != _canonical_sha256(body):
        raise StudyContractError("Z-table study contract digest differs")

    expected = create_z_table_f_study_contract(
        measured_root,
        policy_wrapper_root,
    )
    if raw != expected:
        raise StudyContractError(
            "Z-table study contract differs from active source or policy"
        )
    return raw


def require_z_table_f_explicit_cell(
    contract: Mapping[str, object],
    cell_id: str,
) -> None:
    """Reject broad selectors and any cell outside the dedicated Z table."""

    allowed = contract.get("allowed_cell_ids")
    if (
        not isinstance(cell_id, str)
        or not cell_id
        or not isinstance(allowed, list)
        or cell_id not in allowed
    ):
        raise StudyContractError(
            f"cell {cell_id!r} is outside the authenticated Z-table scope"
        )


def z_table_f_attempt_binding(
    study_contract_sha256: str,
) -> dict[str, str]:
    """Return the exact provenance record for one contracted attempt."""

    if (
        len(study_contract_sha256) != 64
        or any(character not in _SHA40 for character in study_contract_sha256)
    ):
        raise StudyContractError("study contract SHA-256 is invalid")
    return {
        "abi": Z_TABLE_F_ATTEMPT_BINDING_ABI,
        "study_id": Z_TABLE_F_STUDY_ID,
        "study_contract_sha256": study_contract_sha256,
    }


def bind_z_table_f_attempt(
    measurement: dict[str, object],
    study_contract_sha256: str,
) -> None:
    """Bind a scheduler-produced attempt result to its study contract."""

    expected = z_table_f_attempt_binding(study_contract_sha256)
    provenance = measurement.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise StudyContractError("measurement provenance is invalid")
    active = (
        provenance.get("study_contract")
        if isinstance(provenance, Mapping)
        else None
    )
    if active is not None and active != expected:
        raise StudyContractError(
            "measurement is already bound to a different study contract"
        )
    measurement["provenance"] = {
        **({} if provenance is None else dict(provenance)),
        "study_contract": expected,
    }


def require_z_table_f_attempt_binding(
    measurement: Mapping[str, object],
    study_contract_sha256: str,
) -> None:
    """Reject a current or publication transplanted from another study."""

    provenance = measurement.get("provenance")
    active = (
        provenance.get("study_contract")
        if isinstance(provenance, Mapping)
        else None
    )
    if active != z_table_f_attempt_binding(study_contract_sha256):
        raise StudyContractError(
            "measurement is not bound to the active study contract"
        )


def audit_z_table_f_policy_projection(
    contract: Mapping[str, object],
    service: ReportService,
    *,
    maximum_n: int,
) -> dict[str, object]:
    """Audit current-to-publication policy identity for a completed Z tier."""

    if maximum_n < Z_TABLE_F_CONTRACT_MINIMUM_N or maximum_n > 9:
        raise StudyContractError(
            "maximum_n must be between 8 and 9 for contracted evidence"
        )
    source = contract.get("measured_source")
    if not isinstance(source, Mapping):
        raise StudyContractError("study contract source identity is missing")
    revision = source.get("revision")
    tree = source.get("tree")
    if not isinstance(revision, str) or not isinstance(tree, str):
        raise StudyContractError("study contract source identity is invalid")
    contract_sha256 = contract.get("sha256")
    if not isinstance(contract_sha256, str):
        raise StudyContractError("study contract digest is invalid")
    z_table_f_attempt_binding(contract_sha256)

    allowed = contract.get("allowed_cell_ids")
    if not isinstance(allowed, list):
        raise StudyContractError("study contract cell scope is invalid")
    allowed_ids = frozenset(str(value) for value in allowed)
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.cell_id in allowed_ids and cell.n_final <= maximum_n
    )
    if not cells:
        raise StudyContractError("study contract selects no audit cells")

    published_by_cell: dict[str, Mapping[str, object]] = {}
    for payload in service.load_caches().values():
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            cell_id = entry.get("cell_id")
            measurement = entry.get("measurement")
            if isinstance(cell_id, str) and isinstance(
                measurement,
                Mapping,
            ):
                published_by_cell[cell_id] = measurement

    errors: list[str] = []
    states: dict[str, int] = {}
    static_na = 0
    for cell in cells:
        published = published_by_cell.get(cell.cell_id)
        if published is None:
            errors.append(f"{cell.cell_id}: publication entry is missing")
            continue
        static_reason = REPORT_CATALOG.static_na_reason(cell)
        if static_reason is not None:
            static_na += 1
            if published.get("status") != ResultStatus.NOT_AVAILABLE.value:
                errors.append(
                    f"{cell.cell_id}: static N/A publication was overwritten"
                )
            continue

        current = service.store.load_current(
            cell.cell_id,
            missing_ok=True,
        )
        if current is None:
            errors.append(f"{cell.cell_id}: authenticated current is missing")
            continue
        try:
            validate_measurement(current.result, expected_cell=cell)
            validate_measurement(published, expected_cell=cell)
            require_z_table_f_attempt_binding(
                current.result,
                contract_sha256,
            )
            require_z_table_f_attempt_binding(
                published,
                contract_sha256,
            )
            current_state = validate_policy_measurement(
                MACBOOK_M3_Z_TABLE_F_POLICY,
                MACBOOK_M3_PROFILE,
                cell,
                current.result,
                expected_source_revision=revision,
                expected_source_tree=tree,
            )
            published_state = validate_policy_measurement(
                MACBOOK_M3_Z_TABLE_F_POLICY,
                MACBOOK_M3_PROFILE,
                cell,
                published,
                expected_source_revision=revision,
                expected_source_tree=tree,
            )
            if current_state is not published_state:
                raise StudyContractError(
                    "current and publication policy states differ"
                )
            if not publication_measurement_matches_current(
                published,
                current.result,
                service.paths,
            ):
                raise StudyContractError(
                    "publication is not the portable current projection"
                )
            expected_label = {
                PolicyMeasurementState.GENERATION_LIMIT: ">1h",
                PolicyMeasurementState.MEMORY_LIMIT: ">30GB",
            }.get(current_state)
            if (
                expected_label is not None
                and policy_status_label(published) != expected_label
            ):
                raise StudyContractError(
                    f"terminal label is not {expected_label}"
                )
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"{cell.cell_id}: {error}")
            continue
        states[current_state.value] = states.get(current_state.value, 0) + 1

    if errors:
        preview = "; ".join(errors[:20])
        raise StudyContractError(
            f"Z-table policy projection audit failed: {preview}"
        )
    return {
        "schema": "pyamplicol-z-table-f-policy-audit-v1",
        "status": "ok",
        "study_contract_sha256": contract_sha256,
        "campaign_policy": MACBOOK_M3_Z_TABLE_F_POLICY.as_manifest(),
        "policy_profile": MACBOOK_M3_PROFILE,
        "measured_source": dict(source),
        "maximum_n": maximum_n,
        "declared_cell_count": len(cells),
        "static_na_cell_count": static_na,
        "policy_state_counts": dict(sorted(states.items())),
    }


__all__ = [
    "Z_TABLE_F_ATTEMPT_BINDING_ABI",
    "Z_TABLE_F_CONTRACT_MINIMUM_N",
    "Z_TABLE_F_MEMORY_GUARD",
    "Z_TABLE_F_SELECTION_ABI",
    "Z_TABLE_F_STUDY_CONTRACT_SCHEMA",
    "Z_TABLE_F_STUDY_ID",
    "StudyContractError",
    "audit_z_table_f_policy_projection",
    "bind_z_table_f_attempt",
    "create_z_table_f_study_contract",
    "load_z_table_f_study_contract",
    "require_z_table_f_attempt_binding",
    "require_z_table_f_explicit_cell",
    "write_z_table_f_study_contract",
    "z_table_f_attempt_binding",
    "z_table_f_cell_ids",
]
