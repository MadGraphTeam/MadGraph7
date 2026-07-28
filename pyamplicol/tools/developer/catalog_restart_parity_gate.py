#!/usr/bin/env python3
"""Fail-closed restart gate for a final-source structural parity preflight.

This validator deliberately does not generate or optimize amplitudes.  It is an
independent consumer of a final-source preflight manifest.  In particular, it
does not trust a producer's summary boolean or precomputed ratios.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode, Workload

SCHEMA = "pyamplicol-final-source-structural-preflight-v1"
OUTPUT_SCHEMA = "pyamplicol-independent-restart-parity-gate-v1"
MAX_RATIO = 1.05
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_EXACT_PROOF_STRENGTHS = {"exact-symbolic", "exact-reconstructed"}
_CANDIDATE_MODES = {
    ExecutionMode.RECURRENCE,
    ExecutionMode.COMPILED,
    ExecutionMode.EAGER,
}


class RestartParityError(ValueError):
    """The final-source preflight manifest is malformed."""


@dataclass(frozen=True)
class Counts:
    source_current_count: int
    produced_current_count: int
    kernel_evaluation_count: int
    attachment_count: int
    amplitude_destination_count: int

    @property
    def logical_current_count(self) -> int:
        return self.source_current_count + self.produced_current_count

    def add(self, other: Counts) -> Counts:
        return Counts(
            self.source_current_count + other.source_current_count,
            self.produced_current_count + other.produced_current_count,
            self.kernel_evaluation_count + other.kernel_evaluation_count,
            self.attachment_count + other.attachment_count,
            self.amplitude_destination_count + other.amplitude_destination_count,
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RestartParityError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RestartParityError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RestartParityError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise RestartParityError(f"{label} must be a lowercase SHA-256")
    return text


def _revision(value: object, label: str) -> str:
    text = _text(value, label)
    if _REVISION.fullmatch(text) is None:
        raise RestartParityError(f"{label} must be a full lowercase Git revision")
    return text


def _count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RestartParityError(f"{label} must be a non-negative integer")
    return value


def _counts(value: object, label: str) -> Counts:
    payload = _mapping(value, label)
    counts = Counts(
        _count(payload.get("source_current_count"), f"{label}.source_current_count"),
        _count(
            payload.get("produced_current_count"),
            f"{label}.produced_current_count",
        ),
        _count(
            payload.get("kernel_evaluation_count"),
            f"{label}.kernel_evaluation_count",
        ),
        _count(payload.get("attachment_count"), f"{label}.attachment_count"),
        _count(
            payload.get("amplitude_destination_count"),
            f"{label}.amplitude_destination_count",
        ),
    )
    if counts.logical_current_count == 0:
        raise RestartParityError(f"{label} has no logical currents")
    return counts


def _candidate_cells() -> tuple[Any, ...]:
    return tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode in _CANDIDATE_MODES
    )


def _identity_reasons(cell: Any, record: Mapping[str, Any]) -> list[str]:
    expected = {
        "process_key": cell.process_key,
        "n_final": cell.n_final,
        "mode": cell.measurement.execution_mode.value,
        "model": cell.measurement.model.value,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
    }
    return [
        f"identity-{name}-mismatch"
        for name, value in expected.items()
        if record.get(name) != value
    ]


def _proof_reasons(candidate: Mapping[str, Any], revision: str) -> list[str]:
    reasons: list[str] = []
    try:
        proof = _mapping(candidate.get("semantic_proof"), "candidate.semantic_proof")
        if proof.get("status") != "proven":
            reasons.append("semantic-proof-not-proven")
        if proof.get("strength") not in _EXACT_PROOF_STRENGTHS:
            reasons.append("semantic-proof-not-exact")
        if proof.get("source_revision") != revision:
            reasons.append("semantic-proof-source-revision-mismatch")
        for field in (
            "current_member_map_sha256",
            "interaction_row_map_sha256",
            "closure_map_sha256",
            "source_contract_sha256",
        ):
            _digest(proof.get(field), f"candidate.semantic_proof.{field}")
    except RestartParityError:
        reasons.append("semantic-proof-incomplete")

    discovery = candidate.get("numerical_relation_discovery")
    if discovery is not None:
        try:
            discovery_map = _mapping(
                discovery,
                "candidate.numerical_relation_discovery",
            )
            if discovery_map.get("used_for_generation") is True:
                if discovery_map.get("reconstructed_exact") is not True:
                    reasons.append("numerical-only-relation-used-for-generation")
                else:
                    _digest(
                        discovery_map.get("exact_proof_sha256"),
                        "candidate.numerical_relation_discovery.exact_proof_sha256",
                    )
        except RestartParityError:
            reasons.append("numerical-relation-discovery-malformed")

    try:
        numerical = _mapping(
            candidate.get("numerical_validation"),
            "candidate.numerical_validation",
        )
        if numerical.get("status") != "ok":
            reasons.append("numerical-validation-not-ok")
        precision = _count(
            numerical.get("precision_decimal_digits"),
            "candidate.numerical_validation.precision_decimal_digits",
        )
        if precision < 50:
            reasons.append("numerical-validation-below-precision-50")
        _digest(
            numerical.get("comparison_sha256"),
            "candidate.numerical_validation.comparison_sha256",
        )
    except RestartParityError:
        reasons.append("numerical-validation-incomplete")
    return reasons


def _candidate_counts_and_inventory(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Counts], list[str]]:
    reasons: list[str] = []
    phases: dict[str, Counts] = {}
    try:
        for phase in ("active", "final_materialized", "peak_materialized"):
            phases[phase] = _counts(candidate.get(phase), f"candidate.{phase}")
    except RestartParityError:
        return {}, ["candidate-structural-counts-incomplete"]

    try:
        inventory = _mapping(
            candidate.get("persisted_lane_inventory"),
            "candidate.persisted_lane_inventory",
        )
        if inventory.get("status") != "complete":
            reasons.append("persisted-lane-inventory-incomplete")
        _digest(
            inventory.get("inventory_sha256"),
            "candidate.persisted_lane_inventory.inventory_sha256",
        )
        objects = _array(
            inventory.get("objects"),
            "candidate.persisted_lane_inventory.objects",
        )
        roles = _array(
            inventory.get("roles"),
            "candidate.persisted_lane_inventory.roles",
        )
        if not objects or not roles:
            reasons.append("persisted-lane-inventory-empty")
        totals = Counts(0, 0, 0, 0, 0)
        object_ids: set[str] = set()
        for index, raw in enumerate(objects):
            item = _mapping(raw, f"persisted object {index}")
            object_id = _text(item.get("object_id"), f"persisted object {index}.id")
            if object_id in object_ids:
                reasons.append("persisted-object-id-duplicate")
            object_ids.add(object_id)
            _digest(
                item.get("content_sha256"),
                f"persisted object {index}.content_sha256",
            )
            totals = totals.add(
                _counts(item.get("counts"), f"persisted object {index}")
            )
        referenced: set[str] = set()
        for index, raw in enumerate(roles):
            role = _mapping(raw, f"persisted role {index}")
            _text(role.get("role"), f"persisted role {index}.role")
            referenced.add(
                _text(role.get("object_id"), f"persisted role {index}.object")
            )
        if referenced != object_ids:
            reasons.append("persisted-role-object-coverage-incomplete")
        if totals != phases["final_materialized"]:
            reasons.append("persisted-object-counts-do-not-reconcile")
    except RestartParityError:
        reasons.append("persisted-lane-inventory-malformed")
    return phases, reasons


def _legacy_counts(
    legacy: Mapping[str, Any],
    *,
    workload: Workload,
) -> tuple[Counts, Counts, list[str]]:
    reasons: list[str] = []
    try:
        active = _counts(legacy.get("active"), "legacy.active")
        static = _counts(legacy.get("static"), "legacy.static")
        mapping = _mapping(legacy.get("object_mapping"), "legacy.object_mapping")
        if mapping.get("status") != "exact":
            reasons.append("legacy-object-mapping-not-exact")
        for field in (
            "current_object_map_sha256",
            "kernel_term_map_sha256",
            "combine_route_map_sha256",
        ):
            _digest(mapping.get(field), f"legacy.object_mapping.{field}")
        multiplicity = _mapping(
            legacy.get("row_multiplicity"),
            "legacy.row_multiplicity",
        )
        if multiplicity.get("status") != "exact":
            reasons.append("legacy-row-multiplicity-not-exact")
        _digest(
            multiplicity.get("histogram_sha256"),
            "legacy.row_multiplicity.histogram_sha256",
        )
        if workload is Workload.CONTRACTED and multiplicity.get("call_count") is None:
            reasons.append("legacy-contracted-call-histogram-missing")
        if workload is Workload.CONTRACTED:
            _count(
                multiplicity.get("call_count"),
                "legacy.row_multiplicity.call_count",
            )
    except RestartParityError:
        return Counts(0, 0, 0, 0, 0), Counts(0, 0, 0, 0, 0), [
            "legacy-structural-evidence-incomplete"
        ]
    return active, static, reasons


def _phase_ratios(candidate: Counts, legacy: Counts) -> dict[str, float]:
    values = {
        "logical_current": (
            candidate.logical_current_count,
            legacy.logical_current_count,
        ),
        "kernel_evaluation": (
            candidate.kernel_evaluation_count,
            legacy.kernel_evaluation_count,
        ),
        "attachment": (candidate.attachment_count, legacy.attachment_count),
    }
    ratios: dict[str, float] = {}
    for name, (numerator, denominator) in values.items():
        if denominator <= 0:
            raise RestartParityError(f"legacy {name} budget is not positive")
        ratio = numerator / denominator
        if not math.isfinite(ratio):
            raise RestartParityError(f"{name} ratio is not finite")
        ratios[name] = ratio
    return ratios


def _exception_reasons(record: Mapping[str, Any]) -> list[str]:
    try:
        exception = _mapping(
            record.get("representation_exception"),
            "representation_exception",
        )
        if exception.get("status") != "reviewed":
            return ["representation-exception-not-reviewed"]
        _revision(exception.get("review_revision"), "exception.review_revision")
        _digest(exception.get("object_mapping_sha256"), "exception.object_mapping")
        _text(exception.get("reason"), "exception.reason")
        return []
    except RestartParityError:
        return ["representation-exception-incomplete"]


def _validate_row(
    cell: Any,
    record: Mapping[str, Any],
    revision: str,
) -> dict[str, Any]:
    reasons = _identity_reasons(cell, record)
    classification = record.get("classification")
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        return {
            "cell_id": cell.cell_id,
            "classification": classification,
            "status": "failed",
            "reason_codes": sorted({*reasons, "candidate-evidence-missing"}),
            "ratios": {},
        }
    if candidate.get("source_revision") != revision:
        reasons.append("candidate-source-revision-mismatch")
    reasons.extend(_proof_reasons(candidate, revision))
    phases, inventory_reasons = _candidate_counts_and_inventory(candidate)
    reasons.extend(inventory_reasons)

    legacy_available = REPORT_CATALOG.legacy_reference_available(cell)
    ratios: dict[str, dict[str, float]] = {}
    if not legacy_available:
        if classification != "legacy-scope-unavailable":
            reasons.append("legacy-scope-classification-mismatch")
        legacy = record.get("legacy")
        if not isinstance(legacy, Mapping) or legacy.get("reason") != (
            "original-amplicol-open-quark-line-limit"
        ):
            reasons.append("legacy-scope-unavailable-reason-missing")
    else:
        if classification not in {
            "certified-parity",
            "reviewed-representation-exception",
        }:
            reasons.append("comparable-row-not-classified")
        legacy = record.get("legacy")
        if not isinstance(legacy, Mapping) or legacy.get("scope") != "available":
            reasons.append("legacy-evidence-missing")
        else:
            active, static, legacy_reasons = _legacy_counts(
                legacy,
                workload=cell.workload,
            )
            reasons.extend(legacy_reasons)
            if phases and not legacy_reasons:
                try:
                    ratios = {
                        "active": _phase_ratios(phases["active"], active),
                        "final_materialized": _phase_ratios(
                            phases["final_materialized"],
                            static,
                        ),
                        "peak_materialized": _phase_ratios(
                            phases["peak_materialized"],
                            static,
                        ),
                    }
                except RestartParityError:
                    reasons.append("structural-ratio-unavailable")
                violations = [
                    f"{phase}-{metric}-exceeds-1.05"
                    for phase, metrics in ratios.items()
                    for metric, ratio in metrics.items()
                    if ratio > MAX_RATIO
                ]
                if violations:
                    if classification == "reviewed-representation-exception":
                        reasons.extend(_exception_reasons(record))
                    else:
                        reasons.extend(violations)
                elif classification == "reviewed-representation-exception":
                    reasons.append("representation-exception-unnecessary")

    reason_codes = sorted(set(reasons))
    return {
        "cell_id": cell.cell_id,
        "classification": classification,
        "status": "ok" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "ratios": ratios,
    }


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    revision = _revision(expected_revision, "expected revision")
    if payload.get("schema") != SCHEMA:
        raise RestartParityError(f"manifest schema must be {SCHEMA!r}")
    if payload.get("source_revision") != revision:
        raise RestartParityError("manifest is not bound to the expected final revision")
    records = _array(payload.get("cells"), "manifest cells")
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, raw in enumerate(records):
        record = _mapping(raw, f"manifest cell {index}")
        cell_id = _text(record.get("cell_id"), f"manifest cell {index}.cell_id")
        if cell_id in by_id:
            duplicate_ids.add(cell_id)
        by_id[cell_id] = record

    rows: list[dict[str, Any]] = []
    expected_ids: set[str] = set()
    for cell in _candidate_cells():
        expected_ids.add(cell.cell_id)
        record = by_id.get(cell.cell_id)
        if record is None:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "classification": None,
                    "status": "failed",
                    "reason_codes": ["missing-record"],
                    "ratios": {},
                }
            )
            continue
        row = _validate_row(cell, record, revision)
        if cell.cell_id in duplicate_ids:
            row["status"] = "failed"
            row["reason_codes"] = sorted(
                {*row["reason_codes"], "duplicate-record"}
            )
        rows.append(row)
    for cell_id in sorted(set(by_id) - expected_ids):
        rows.append(
            {
                "cell_id": cell_id,
                "classification": by_id[cell_id].get("classification"),
                "status": "failed",
                "reason_codes": ["unexpected-record"],
                "ratios": {},
            }
        )

    failures = [row for row in rows if row["status"] != "ok"]
    classifications = Counter(
        str(row["classification"]) for row in rows if row["status"] == "ok"
    )
    reason_counts = Counter(
        reason for row in failures for reason in row["reason_codes"]
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "source_revision": revision,
        "summary": {
            "expected_cell_count": len(_candidate_cells()),
            "reported_row_count": len(rows),
            "passed_row_count": len(rows) - len(failures),
            "failed_row_count": len(failures),
            "classification_counts": dict(sorted(classifications.items())),
            "failure_reason_counts": dict(sorted(reason_counts.items())),
            "restart_ready": not failures,
        },
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = json.loads(args.manifest.read_text())
    result = validate_manifest(
        _mapping(payload, "manifest"),
        expected_revision=args.expected_source_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["restart_ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
