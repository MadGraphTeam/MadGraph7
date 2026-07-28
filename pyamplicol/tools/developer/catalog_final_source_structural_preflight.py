#!/usr/bin/env python3
"""Assemble and authenticate the all-catalog final-source restart manifest.

This producer is intentionally structural-only: it never benchmarks or derives
proof claims from timings.  Each successful candidate artifact must contain a
``structural-preflight-proof.json`` emitted by generation, and each comparable
legacy artifact must contain ``legacy-structural-proof.json`` emitted by the
legacy probe.  The producer authenticates those sidecars against their source
revision and every persisted object before assembling the exact 1,136 rows
consumed by :mod:`tools.developer.catalog_restart_parity_gate`.

Missing sidecars are reported as missing producer hooks.  Counts or digests are
never guessed from a summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.developer.catalog_restart_parity_gate import SCHEMA
from tools.developer.catalog_structural_parity_audit import (
    CatalogParityError,
    _candidate_counts,
    _load_currents,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode, ResultStatus

CELL_PROOF_SCHEMA = "pyamplicol-cell-final-source-structural-proof-v1"
LEGACY_PROOF_SCHEMA = "pyamplicol-legacy-final-structural-proof-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MODES = {
    ExecutionMode.RECURRENCE,
    ExecutionMode.COMPILED,
    ExecutionMode.EAGER,
}


class FinalSourceProducerError(RuntimeError):
    """Exact restart evidence is absent, stale, or unauthenticated."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FinalSourceProducerError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FinalSourceProducerError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(result: dict[str, Any], cell_id: str) -> Path:
    artifact = result.get("artifact")
    raw = artifact.get("path") if isinstance(artifact, dict) else None
    if not isinstance(raw, str) or not raw:
        raise FinalSourceProducerError(f"{cell_id}: result has no artifact path")
    path = Path(raw)
    if not path.is_dir():
        raise FinalSourceProducerError(f"{cell_id}: artifact is absent: {path}")
    return path


def _counts(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise FinalSourceProducerError(f"{label} must be an object")
    fields = (
        "source_current_count",
        "produced_current_count",
        "kernel_evaluation_count",
        "attachment_count",
        "amplitude_destination_count",
    )
    output: dict[str, int] = {}
    for field in fields:
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise FinalSourceProducerError(
                f"{label}.{field} must be a non-negative integer"
            )
        output[field] = item
    if output["source_current_count"] + output["produced_current_count"] <= 0:
        raise FinalSourceProducerError(f"{label} has no logical currents")
    return output


def _authenticate_inventory(
    artifact: Path,
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise FinalSourceProducerError(f"{label} inventory is not complete")
    objects = value.get("objects")
    roles = value.get("roles")
    if not isinstance(objects, list) or not objects or not isinstance(roles, list):
        raise FinalSourceProducerError(f"{label} inventory is malformed")
    ids: set[str] = set()
    authenticated: list[dict[str, Any]] = []
    for index, raw in enumerate(objects):
        if not isinstance(raw, dict):
            raise FinalSourceProducerError(f"{label} object {index} is malformed")
        object_id = raw.get("object_id")
        relative = raw.get("path")
        expected = raw.get("content_sha256")
        if (
            not isinstance(object_id, str)
            or not object_id
            or object_id in ids
            or not isinstance(relative, str)
            or not relative
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
        ):
            raise FinalSourceProducerError(f"{label} object {index} is invalid")
        ids.add(object_id)
        path = (artifact / relative).resolve()
        try:
            path.relative_to(artifact.resolve())
        except ValueError as error:
            raise FinalSourceProducerError(
                f"{label} object escapes artifact: {relative}"
            ) from error
        if not path.is_file() or _sha256(path) != expected:
            raise FinalSourceProducerError(
                f"{label} object is absent or changed: {relative}"
            )
        authenticated.append(
            {
                "object_id": object_id,
                "content_sha256": expected,
                "counts": _counts(raw.get("counts"), f"{label} object {object_id}"),
            }
        )
    valid_roles = [
        role
        for role in roles
        if isinstance(role, dict)
        and isinstance(role.get("role"), str)
        and bool(role["role"])
        and isinstance(role.get("object_id"), str)
    ]
    referenced = {role["object_id"] for role in valid_roles}
    if referenced != ids or len(valid_roles) != len(roles):
        raise FinalSourceProducerError(f"{label} role coverage is not exact")
    canonical = json.dumps(
        {"objects": authenticated, "roles": roles},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "status": "complete",
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "objects": authenticated,
        "roles": roles,
    }


def _proof_digest_fields(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalSourceProducerError(f"{label} is absent")
    output = dict(value)
    for field in (
        "current_member_map_sha256",
        "interaction_row_map_sha256",
        "closure_map_sha256",
        "source_contract_sha256",
    ):
        item = output.get(field)
        if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
            raise FinalSourceProducerError(f"{label}.{field} is not exact")
    return output


def _authenticate_evidence_files(
    artifact: Path,
    value: object,
    *,
    label: str,
) -> None:
    if not isinstance(value, list) or not value:
        raise FinalSourceProducerError(f"{label} evidence-file inventory is absent")
    root = artifact.resolve()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise FinalSourceProducerError(
                f"{label} evidence file {index} is malformed"
            )
        relative = raw.get("path")
        expected = raw.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
        ):
            raise FinalSourceProducerError(
                f"{label} evidence file {index} is invalid"
            )
        path = (artifact / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise FinalSourceProducerError(
                f"{label} evidence file escapes artifact: {relative}"
            ) from error
        if not path.is_file() or _sha256(path) != expected:
            raise FinalSourceProducerError(
                f"{label} evidence file is absent or changed: {relative}"
            )


def _candidate_record(
    cell: Any,
    result: dict[str, Any],
    revision: str,
) -> dict[str, Any]:
    if result.get("status") != ResultStatus.OK.value:
        raise FinalSourceProducerError(f"{cell.cell_id}: candidate current is not ok")
    artifact = _artifact(result, cell.cell_id)
    proof_path = artifact / "structural-preflight-proof.json"
    if not proof_path.is_file():
        raise FinalSourceProducerError(
            f"{cell.cell_id}: missing generation hook {proof_path.name}"
        )
    proof = _read(proof_path)
    if (
        proof.get("schema") != CELL_PROOF_SCHEMA
        or proof.get("cell_id") != cell.cell_id
        or proof.get("source_revision") != revision
        or proof.get("color_accuracy") != cell.measurement.accuracy.value
        or proof.get("workload") != cell.workload.value
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: structural proof is stale or has wrong identity"
        )
    candidate = _candidate_counts(result, workload=cell.workload)
    if candidate.source_revision != revision:
        raise FinalSourceProducerError(
            f"{cell.cell_id}: measured source is not the final source revision"
        )
    phases = {
        name: _counts(proof.get(name), f"{cell.cell_id} candidate.{name}")
        for name in ("active", "final_materialized", "peak_materialized")
    }
    expected = {
        "active": (candidate.active, candidate.active_closure_count),
        "final_materialized": (
            candidate.final_materialized,
            candidate.final_materialized_closure_count,
        ),
        "peak_materialized": (
            candidate.peak_materialized,
            candidate.peak_materialized_closure_count,
        ),
    }
    for name, (structural, closure_count) in expected.items():
        phase = phases[name]
        if (
            phase["source_current_count"] + phase["produced_current_count"]
            != structural.current_count
            or phase["kernel_evaluation_count"] != structural.evaluation_count
            or phase["attachment_count"] != structural.attachment_count
            or (
                closure_count is not None
                and phase["amplitude_destination_count"] != closure_count
            )
        ):
            raise FinalSourceProducerError(
                f"{cell.cell_id}: {name} proof disagrees with runtime structure"
            )
    semantic = _proof_digest_fields(
        proof.get("semantic_proof"),
        f"{cell.cell_id} semantic proof",
    )
    if (
        semantic.get("status") != "proven"
        or semantic.get("strength") not in {"exact-symbolic", "exact-reconstructed"}
        or semantic.get("source_revision") != revision
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: semantic proof is not exact and source-bound"
        )
    numerical = proof.get("numerical_validation")
    if (
        not isinstance(numerical, dict)
        or numerical.get("status") != "ok"
        or not isinstance(numerical.get("precision_decimal_digits"), int)
        or numerical["precision_decimal_digits"] < 50
        or not isinstance(numerical.get("comparison_sha256"), str)
        or _SHA256.fullmatch(numerical["comparison_sha256"]) is None
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: precision>=50 numerical proof hook is missing"
        )
    return {
        "source_revision": revision,
        **phases,
        "semantic_proof": semantic,
        "numerical_validation": numerical,
        "persisted_lane_inventory": _authenticate_inventory(
            artifact,
            proof.get("persisted_lane_inventory"),
            label=f"{cell.cell_id} candidate",
        ),
    }


def _legacy_record(cell: Any, result: dict[str, Any] | None) -> dict[str, Any]:
    if not REPORT_CATALOG.legacy_reference_available(cell):
        return {
            "scope": "unavailable",
            "reason": "original-amplicol-open-quark-line-limit",
        }
    if result is None or result.get("status") != ResultStatus.OK.value:
        raise FinalSourceProducerError(
            f"{cell.cell_id}: authenticated legacy current is absent"
        )
    artifact = _artifact(result, cell.cell_id)
    path = artifact / "legacy-structural-proof.json"
    if not path.is_file():
        raise FinalSourceProducerError(
            f"{cell.cell_id}: missing legacy probe hook {path.name}"
        )
    proof = _read(path)
    if proof.get("schema") != LEGACY_PROOF_SCHEMA:
        raise FinalSourceProducerError(
            f"{cell.cell_id}: legacy structural proof schema is unsupported"
        )
    _authenticate_evidence_files(
        artifact,
        proof.get("evidence_files"),
        label=f"{cell.cell_id} legacy",
    )
    active = _counts(proof.get("active"), f"{cell.cell_id} legacy.active")
    static = _counts(proof.get("static"), f"{cell.cell_id} legacy.static")
    mapping = proof.get("object_mapping")
    multiplicity = proof.get("row_multiplicity")
    if not isinstance(mapping, dict) or mapping.get("status") != "exact":
        raise FinalSourceProducerError(
            f"{cell.cell_id}: exact legacy object mapping is absent"
        )
    _proof_digest_fields(
        {
            "current_member_map_sha256": mapping.get("current_object_map_sha256"),
            "interaction_row_map_sha256": mapping.get("kernel_term_map_sha256"),
            "closure_map_sha256": mapping.get("combine_route_map_sha256"),
            "source_contract_sha256": mapping.get("source_contract_sha256"),
        },
        f"{cell.cell_id} legacy mapping",
    )
    if (
        not isinstance(multiplicity, dict)
        or multiplicity.get("status") != "exact"
        or not isinstance(multiplicity.get("histogram_sha256"), str)
        or _SHA256.fullmatch(multiplicity["histogram_sha256"]) is None
        or (
            cell.workload.value == "contracted"
            and (
                not isinstance(multiplicity.get("call_count"), int)
                or multiplicity["call_count"] < 0
            )
        )
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: exact legacy row histogram is absent"
        )
    return {
        "scope": "available",
        "active": active,
        "static": static,
        "object_mapping": mapping,
        "row_multiplicity": multiplicity,
    }


def produce(
    artifact_root: Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    if _REVISION.fullmatch(source_revision) is None:
        raise FinalSourceProducerError("source revision must be a full Git SHA")
    try:
        currents = _load_currents(artifact_root)
    except CatalogParityError as error:
        raise FinalSourceProducerError(str(error)) from error
    references = {
        (
            cell.process_key,
            cell.n_final,
            cell.measurement.accuracy,
            cell.workload,
        ): cell
        for cell in REPORT_CATALOG.reference_cells()
    }
    rows: list[dict[str, Any]] = []
    for cell in REPORT_CATALOG.matrix_cells():
        if cell.measurement.execution_mode not in _MODES:
            continue
        result = currents.get(cell.cell_id)
        if result is None:
            raise FinalSourceProducerError(
                f"{cell.cell_id}: final-source candidate current is absent"
            )
        reference = references[
            (
                cell.process_key,
                cell.n_final,
                cell.measurement.accuracy,
                cell.workload,
            )
        ]
        comparable = REPORT_CATALOG.legacy_reference_available(cell)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "process_key": cell.process_key,
                "n_final": cell.n_final,
                "mode": cell.measurement.execution_mode.value,
                "model": cell.measurement.model.value,
                "accuracy": cell.measurement.accuracy.value,
                "workload": cell.workload.value,
                "classification": (
                    "certified-parity"
                    if comparable
                    else "legacy-scope-unavailable"
                ),
                "candidate": _candidate_record(cell, result, source_revision),
                "legacy": _legacy_record(cell, currents.get(reference.cell_id)),
            }
        )
    if len(rows) != 1136:
        raise FinalSourceProducerError(
            f"catalog producer expected 1136 rows, generated {len(rows)}"
        )
    return {"schema": SCHEMA, "source_revision": source_revision, "cells": rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = produce(
            args.artifact_root,
            source_revision=args.source_revision,
        )
    except FinalSourceProducerError as error:
        print(f"final-source structural preflight unavailable: {error}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(payload['cells'])} authenticated catalog rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
