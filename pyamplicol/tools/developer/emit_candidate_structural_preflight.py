#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Atomically attach exact structural restart evidence to one generated artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pyamplicol.artifacts.manifest import load_manifest
from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    SEMANTIC_MAP_DOMAINS,
    validate_generation_structural_proof,
)
from tools.developer.catalog_final_source_structural_preflight import (
    CELL_PROOF_SCHEMA,
    FinalSourceProducerError,
)
from tools.developer.catalog_structural_parity_audit import (
    _candidate_counts,
    _compiled_execution_lanes,
)
from tools.developer.final_source_numerical_truth import (
    TruthProducerError,
    validate_witness_payload,
)
from tools.performance_report.models import Workload

PROOF_PATH = "structural-preflight-proof.json"
_REVISION = re.compile(r"[0-9a-f]{40}")


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
            raise FinalSourceProducerError("eager inspection lacks exact source_count")
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


def _execution_mode(execution: dict[str, Any]) -> str:
    try:
        return {
            "pyamplicol-runtime-recurrence-execution": "recurrence",
            "pyamplicol-runtime-execution": "compiled",
            "pyamplicol-runtime-eager-execution": "eager",
        }[str(execution.get("kind"))]
    except KeyError as error:
        raise FinalSourceProducerError("execution mode is unsupported") from error


def _numerical_proof(
    path: Path,
    revision: str,
    *,
    cell_id: str | None = None,
    mode: str | None = None,
    accuracy: str | None = None,
    workload: str | None = None,
    candidate_artifact_id: str | None = None,
    structural_proof_sha256: str | None = None,
) -> dict[str, Any]:
    proof = _read(path)
    try:
        return validate_witness_payload(
            proof,
            expected_source_revision=revision,
            expected_cell_id=cell_id,
            expected_mode=mode,
            expected_accuracy=accuracy,
            expected_workload=workload,
            expected_candidate_artifact_id=candidate_artifact_id,
            expected_structural_proof_sha256=structural_proof_sha256,
        )
    except TruthProducerError as error:
        raise FinalSourceProducerError(
            "numerical proof is not a recomputable independent precision>=50 witness"
        ) from error


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
    structural_records = [
        record
        for record in manifest.payloads
        if record.role == STRUCTURAL_SOURCE_PROOF_ROLE
        and record.process_id == str(execution.get("key"))
    ]
    if len(structural_records) != 1:
        raise FinalSourceProducerError(
            "artifact must contain exactly one generation structural proof"
        )
    structural_record = structural_records[0]
    structural_path = artifact / structural_record.path
    structural = _read(structural_path)
    native_inputs = manifest.producer.get("native_build_inputs_sha256")
    if not isinstance(native_inputs, str):
        raise FinalSourceProducerError(
            "artifact producer native-input identity is absent"
        )
    try:
        generation_proof = validate_generation_structural_proof(
            structural,
            artifact_root=artifact,
            expected_process_id=str(execution.get("key")),
            expected_source_revision=source_revision,
            expected_native_build_inputs_sha256=native_inputs,
            expected_execution_path=relative_execution,
            expected_execution_sha256=execution_record.sha256,
            execution=execution,
        )
    except ValueError as error:
        raise FinalSourceProducerError(
            "generation structural proof does not recompute"
        ) from error
    maps = generation_proof["semantic_maps"]
    assert isinstance(maps, dict)
    semantic_witnesses = {name: maps[name]["rows"] for name in SEMANTIC_MAP_DOMAINS}
    semantic_digests = {name: maps[name]["sha256"] for name in SEMANTIC_MAP_DOMAINS}
    process_id = str(execution.get("key"))
    if process_id != structural_record.process_id:
        raise FinalSourceProducerError(
            "generation structural proof process identity changed"
        )
    physical_inventory = generation_proof["physical_lane_inventory"]
    assert isinstance(physical_inventory, dict)
    proof = {
        "schema": CELL_PROOF_SCHEMA,
        "cell_id": cell_id,
        "source_revision": source_revision,
        "artifact_identity": {
            "artifact_id": manifest.artifact_id,
            "process_id": process_id,
            "structural_proof_path": structural_record.path,
            "structural_proof_sha256": structural_record.sha256,
        },
        "color_accuracy": execution.get("color_accuracy"),
        "workload": workload.value,
        "active": active,
        "final_materialized": final,
        "peak_materialized": peak,
        "semantic_proof": {
            "status": "proven",
            "strength": "exact-symbolic",
            "source_revision": source_revision,
            "current_member_map_sha256": semantic_digests["current_member_map"],
            "interaction_row_map_sha256": semantic_digests["interaction_row_map"],
            "closure_map_sha256": semantic_digests["closure_map"],
            "source_contract_sha256": semantic_digests["source_contract"],
            "witnesses": semantic_witnesses,
        },
        "numerical_validation": _numerical_proof(
            numerical_proof,
            source_revision,
            cell_id=cell_id,
            mode=_execution_mode(execution),
            accuracy=str(execution.get("color_accuracy")),
            workload=workload.value,
            candidate_artifact_id=manifest.artifact_id,
            structural_proof_sha256=structural_record.sha256,
        ),
        "persisted_lane_inventory": physical_inventory,
    }
    return proof


def emit(
    artifact: Path,
    *,
    cell_id: str,
    source_revision: str,
    workload: Workload,
    numerical_proof: Path,
    output: Path | None = None,
) -> Path:
    proof = build_proof(
        artifact,
        cell_id=cell_id,
        source_revision=source_revision,
        workload=workload,
        numerical_proof=numerical_proof,
    )
    destination = artifact / PROOF_PATH if output is None else output
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                proof,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


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
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "atomic per-cell sidecar destination; report workers should place "
            "this beside the immutable attempt/result"
        ),
    )
    args = parser.parse_args()
    output = emit(
        args.artifact,
        cell_id=args.cell_id,
        source_revision=args.source_revision,
        workload=Workload(args.workload),
        numerical_proof=args.precision50_proof,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "cell_id": args.cell_id,
                "output": str(output.resolve()),
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
