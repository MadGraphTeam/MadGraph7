#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Assemble and authenticate the all-catalog final-source restart manifest.

This producer is intentionally structural-only: it never benchmarks or derives
proof claims from timings.  Each successful candidate artifact must contain a
``structural-preflight-proof.json`` emitted by generation, and each comparable
legacy artifact must contain ``legacy-structural-proof.json`` emitted by the
legacy probe.  The producer authenticates those sidecars against their source
revision and every persisted object before assembling the exact 1,356 rows
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

from pyamplicol.artifacts import load_manifest
from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    validate_generation_structural_proof,
)
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
            raise FinalSourceProducerError(f"{label} evidence file {index} is invalid")
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
    manifest = load_manifest(artifact)
    artifact_identity = proof.get("artifact_identity")
    if (
        not isinstance(artifact_identity, dict)
        or artifact_identity.get("artifact_id") != manifest.artifact_id
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: cell proof artifact identity is stale"
        )
    structural_records = [
        record
        for record in manifest.payloads
        if record.role == STRUCTURAL_SOURCE_PROOF_ROLE
        and record.process_id == artifact_identity.get("process_id")
    ]
    if len(structural_records) != 1:
        raise FinalSourceProducerError(
            f"{cell.cell_id}: generation structural proof payload is ambiguous"
        )
    structural_record = structural_records[0]
    if (
        artifact_identity.get("structural_proof_path") != structural_record.path
        or artifact_identity.get("structural_proof_sha256") != structural_record.sha256
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: generation structural proof identity is stale"
        )
    generation_proof = _read(artifact / structural_record.path)
    producer_native_inputs = manifest.producer.get("native_build_inputs_sha256")
    if not isinstance(producer_native_inputs, str):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: artifact native-input identity is absent"
        )
    try:
        authenticated_generation_proof = validate_generation_structural_proof(
            generation_proof,
            artifact_root=artifact,
            expected_process_id=str(artifact_identity["process_id"]),
            expected_source_revision=revision,
            expected_native_build_inputs_sha256=producer_native_inputs,
        )
    except ValueError as error:
        raise FinalSourceProducerError(
            f"{cell.cell_id}: generation structural proof does not recompute"
        ) from error
    if (
        proof.get("persisted_lane_inventory")
        != (authenticated_generation_proof["physical_lane_inventory"])
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: cell proof changed the generation lane inventory"
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
        "evidence_root": str(artifact.resolve()),
        "artifact_identity": artifact_identity,
        **phases,
        "semantic_proof": semantic,
        "numerical_validation": numerical,
        "generation_structural_proof": authenticated_generation_proof,
        "persisted_lane_inventory": proof["persisted_lane_inventory"],
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
    provenance = result.get("provenance")
    result_revision = (
        provenance.get("revision") if isinstance(provenance, dict) else None
    )
    if (
        proof.get("cell_id") != cell.cell_id
        or proof.get("accuracy") != cell.measurement.accuracy.value
        or proof.get("workload") != cell.workload.value
        or not isinstance(result_revision, str)
        or _REVISION.fullmatch(result_revision) is None
        or proof.get("source_revision") != result_revision
    ):
        raise FinalSourceProducerError(
            f"{cell.cell_id}: legacy structural proof is stale or has wrong identity"
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
                    "certified-parity" if comparable else "legacy-scope-unavailable"
                ),
                "candidate": _candidate_record(cell, result, source_revision),
                "legacy": _legacy_record(reference, currents.get(reference.cell_id)),
            }
        )
    if len(rows) != 1356:
        raise FinalSourceProducerError(
            f"catalog producer expected 1356 rows, generated {len(rows)}"
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
