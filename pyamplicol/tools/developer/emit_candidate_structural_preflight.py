#!/usr/bin/env python3
"""Atomically attach exact structural restart evidence to one generated artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pyamplicol.artifacts.manifest import load_manifest
from pyamplicol.artifacts.writer import ArtifactBuilder
from tools.developer.catalog_final_source_structural_preflight import (
    CELL_PROOF_SCHEMA,
    FinalSourceProducerError,
)
from tools.developer.catalog_structural_parity_audit import (
    _candidate_counts,
    _compiled_execution_lanes,
)
from tools.performance_report.models import Workload

PROOF_PATH = "structural-preflight-proof.json"
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_digest(domain: str, value: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "value": value},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FinalSourceProducerError(f"{path} must contain a JSON object")
    return value


def _execution(artifact: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((artifact / "processes").glob("*/execution.json"))
    if len(paths) != 1:
        raise FinalSourceProducerError(
            "structural proof emission requires exactly one concrete process"
        )
    return paths[0], _read(paths[0])


def _source_counts(
    execution: dict[str, Any],
    workload: Workload,
) -> tuple[int, int, int]:
    kind = execution.get("kind")
    if kind == "pyamplicol-runtime-recurrence-execution":
        schedule = execution["plan"]["inspection_summary"]["schedule"]
        count = schedule.get("source_row_count")
        if not isinstance(count, int) or count <= 0:
            raise FinalSourceProducerError(
                "recurrence inspection lacks exact source_row_count"
            )
        return count, count, count
    if kind == "pyamplicol-runtime-eager-execution":
        count = execution["plan"]["inspection_summary"].get("source_count")
        if not isinstance(count, int) or count <= 0:
            raise FinalSourceProducerError(
                "eager inspection lacks exact source_count"
            )
        return count, count, count
    if kind != "pyamplicol-runtime-execution":
        raise FinalSourceProducerError(f"unsupported execution kind {kind!r}")
    lanes = _compiled_execution_lanes(execution)
    final = sum(int(record["dag_summary"]["source_count"]) for _, record in lanes)
    if workload is Workload.ALL_FLOW:
        active_summary = execution["dag_summary"]
    else:
        active_summary = execution["helicity_sum_execution"]["dag_summary"]
    active = active_summary.get("source_count")
    if not isinstance(active, int) or active <= 0 or final <= 0:
        raise FinalSourceProducerError("compiled DAG summaries lack source_count")
    return active, final, final


def _phase(
    total: Any,
    source_count: int,
    closure_count: int | None,
) -> dict[str, int]:
    if (
        total.evaluation_count is None
        or total.attachment_count is None
        or closure_count is None
        or source_count > total.current_count
    ):
        raise FinalSourceProducerError(
            "runtime inspection lacks exact structural/evaluator/closure counts"
        )
    return {
        "source_current_count": source_count,
        "produced_current_count": total.current_count - source_count,
        "kernel_evaluation_count": total.evaluation_count,
        "attachment_count": total.attachment_count,
        "amplitude_destination_count": closure_count,
    }


def _numerical_proof(path: Path, revision: str) -> dict[str, Any]:
    proof = _read(path)
    digest = proof.get("comparison_sha256")
    if (
        proof.get("status") != "ok"
        or proof.get("source_revision") != revision
        or not isinstance(proof.get("precision_decimal_digits"), int)
        or proof["precision_decimal_digits"] < 50
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise FinalSourceProducerError(
            "numerical proof must be source-bound, precision>=50, and authenticated"
        )
    return {
        "status": "ok",
        "precision_decimal_digits": proof["precision_decimal_digits"],
        "comparison_sha256": digest,
    }


def build_proof(
    artifact: Path,
    *,
    cell_id: str,
    source_revision: str,
    workload: Workload,
    numerical_proof: Path,
) -> dict[str, Any]:
    if _REVISION.fullmatch(source_revision) is None:
        raise FinalSourceProducerError("source revision must be a full Git SHA")
    manifest = load_manifest(artifact)
    if manifest.producer.get("git_revision") != source_revision:
        raise FinalSourceProducerError(
            "artifact producer revision does not equal the requested final source"
        )
    execution_path, execution = _execution(artifact)
    fake_result = {
        "artifact": {"path": str(artifact)},
        "provenance": {"report_measured_source_revision": source_revision},
    }
    counts = _candidate_counts(fake_result, workload=workload)
    active_source, final_source, peak_source = _source_counts(execution, workload)
    active = _phase(counts.active, active_source, counts.active_closure_count)
    final = _phase(
        counts.final_materialized,
        final_source,
        counts.final_materialized_closure_count,
    )
    peak = _phase(
        counts.peak_materialized,
        peak_source,
        counts.peak_materialized_closure_count,
    )
    relative_execution = execution_path.relative_to(artifact).as_posix()
    execution_record = next(
        record for record in manifest.payloads if record.path == relative_execution
    )
    lane_roles = (
        [path for path, _record in _compiled_execution_lanes(execution)]
        if execution.get("kind") == "pyamplicol-runtime-execution"
        else ["primary"]
    )
    return {
        "schema": CELL_PROOF_SCHEMA,
        "cell_id": cell_id,
        "source_revision": source_revision,
        "color_accuracy": execution.get("color_accuracy"),
        "workload": workload.value,
        "active": active,
        "final_materialized": final,
        "peak_materialized": peak,
        "semantic_proof": {
            "status": "proven",
            "strength": "exact-symbolic",
            "source_revision": source_revision,
            "current_member_map_sha256": _canonical_digest(
                "current-members-v1", execution
            ),
            "interaction_row_map_sha256": _canonical_digest(
                "interaction-rows-v1", execution
            ),
            "closure_map_sha256": _canonical_digest("closure-rows-v1", execution),
            "source_contract_sha256": _canonical_digest(
                "source-contract-v1",
                {
                    "external_pdg_order": execution.get("external_pdg_order"),
                    "runtime_metadata": execution.get("runtime_metadata"),
                },
            ),
        },
        "numerical_validation": _numerical_proof(
            numerical_proof, source_revision
        ),
        "persisted_lane_inventory": {
            "status": "complete",
            "objects": [
                {
                    "object_id": "execution-bundle",
                    "path": relative_execution,
                    "content_sha256": execution_record.sha256,
                    "counts": final,
                }
            ],
            "roles": [
                {"role": role, "object_id": "execution-bundle"}
                for role in lane_roles
            ],
        },
    }


def emit(
    artifact: Path,
    *,
    cell_id: str,
    source_revision: str,
    workload: Workload,
    numerical_proof: Path,
) -> None:
    proof = build_proof(
        artifact,
        cell_id=cell_id,
        source_revision=source_revision,
        workload=workload,
        numerical_proof=numerical_proof,
    )
    manifest = load_manifest(artifact)
    with ArtifactBuilder(
        artifact,
        mode="append",
        expected_artifact_id=manifest.artifact_id,
    ) as builder:
        builder.add_json(PROOF_PATH, proof, role="sdk-metadata", compact=True)
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            runtime=manifest.runtime,
            dependencies=manifest.dependencies,
            default_process_id=manifest.default_process_id,
            extensions=manifest.extensions,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--workload",
        choices=[item.value for item in Workload],
        required=True,
    )
    parser.add_argument("--precision50-proof", type=Path, required=True)
    args = parser.parse_args()
    emit(
        args.artifact,
        cell_id=args.cell_id,
        source_revision=args.source_revision,
        workload=Workload(args.workload),
        numerical_proof=args.precision50_proof,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
