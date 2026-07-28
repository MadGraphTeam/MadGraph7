#!/usr/bin/env python3
"""Audit report-catalog structural work against original AmpliCol evidence.

The performance report stores immutable per-cell artifacts.  This tool reads
those artifacts without importing a native pyAmpliCol runtime and emits one
machine-readable row for every matrix candidate cell.  It deliberately keeps
two comparisons separate:

* active/dynamic work per phase-space point;
* final and peak static materialization during artifact generation.

A candidate can beat AmpliCol dynamically while still materializing far more
work.  Neither comparison is silently substituted for the other.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode, ResultStatus, Workload


class CatalogParityError(RuntimeError):
    """The artifact tree cannot produce an authenticated census."""


_MODULE_CURRENT = re.compile(
    r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*val_c\b",
    re.IGNORECASE,
)
_MODULE_INTERACTION = re.compile(
    r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*int_c\b",
    re.IGNORECASE,
)
_AFTER_FILTER = re.compile(
    r"Total number of currents, vertices and amplitudes after filter"
    r"\s+(\d+)\s+(\d+)\s+(\d+)",
)
_DIRECT_CURRENT = re.compile(r"AMPICOL_COLOR_PROBE_CURRENTS\s+(\d+)")
_DIRECT_INTERACTION = re.compile(r"AMPICOL_COLOR_PROBE_VERTICES\s+(\d+)")
_DIRECT_AMPLITUDE = re.compile(r"AMPICOL_COLOR_PROBE_AMPLITUDES\s+(\d+)")
_PROCESS_ROW = re.compile(r"group:(\d+):integral:(\d+)")

MAX_RATIO = 1.05


@dataclass(frozen=True)
class WorkCounts:
    current_count: int
    interaction_count: int


@dataclass(frozen=True)
class LegacyCounts:
    evidence_kind: str
    active: WorkCounts | None
    static: WorkCounts
    generated_module_count: int
    retained_amplitude_count: int | None
    module_shapes: tuple[tuple[int, int], ...]
    selected_module: str | None
    limitation: str | None


@dataclass(frozen=True)
class CandidateCounts:
    mode: str
    active: WorkCounts
    final_materialized: WorkCounts
    peak_materialized: WorkCounts
    active_evidence_kind: str
    selector_certificate_available: bool
    source_revision: str | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogParityError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CatalogParityError(f"JSON root is not an object: {path}")
    return payload


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CatalogParityError(f"{label} must be a positive integer")
    return value


def _single_execution(artifact: Path) -> Path:
    executions = sorted((artifact / "processes").glob("*/execution.json"))
    if len(executions) != 1:
        raise CatalogParityError(
            f"{artifact} must contain one process execution; found {len(executions)}"
        )
    return executions[0]


def _module_shape(path: Path) -> tuple[int, int]:
    text = path.read_text()
    current = _MODULE_CURRENT.search(text)
    interaction = _MODULE_INTERACTION.search(text)
    if current is None or interaction is None:
        raise CatalogParityError(f"legacy module lacks val_c/int_c dimensions: {path}")
    return int(current.group(1)), int(interaction.group(1))


def _legacy_counts(
    result: dict[str, Any],
    *,
    workload: Workload,
) -> LegacyCounts:
    artifact_payload = result.get("artifact")
    if not isinstance(artifact_payload, dict):
        raise CatalogParityError("legacy result has no artifact")
    artifact_raw = artifact_payload.get("path")
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise CatalogParityError("legacy artifact path is absent")
    artifact = Path(artifact_raw)
    modules = sorted(artifact.rglob("amp*_lib.f03"))
    shapes = tuple(_module_shape(module) for module in modules)

    if workload is Workload.ALL_FLOW:
        log = artifact / "legacy.log"
        text = log.read_text()
        current = _DIRECT_CURRENT.search(text)
        interaction = _DIRECT_INTERACTION.search(text)
        amplitude = _DIRECT_AMPLITUDE.search(text)
        if current is None or interaction is None or amplitude is None:
            raise CatalogParityError("all-flow legacy log lacks structural counters")
        counts = WorkCounts(int(current.group(1)), int(interaction.group(1)))
        return LegacyCounts(
            evidence_kind="direct-fixed-helicity-all-flow-probe",
            active=counts,
            static=counts,
            generated_module_count=0,
            retained_amplitude_count=int(amplitude.group(1)),
            module_shapes=(),
            selected_module=None,
            limitation=None,
        )

    if not modules:
        raise CatalogParityError("generated-library legacy artifact has no modules")

    if workload is Workload.SELECTED_FLOW:
        row = artifact_payload.get("process_row")
        if not isinstance(row, str):
            raise CatalogParityError("selected-flow legacy result lacks process row")
        match = _PROCESS_ROW.fullmatch(row)
        if match is None:
            raise CatalogParityError(f"malformed selected-flow process row: {row!r}")
        selected_name = f"amp{match.group(1)}_{match.group(2)}_lib.f03"
        selected = [module for module in modules if module.name == selected_name]
        if len(selected) != 1:
            raise CatalogParityError(
                f"selected-flow module {selected_name} occurs {len(selected)} times"
            )
        current, interaction = _module_shape(selected[0])
        active = WorkCounts(current, interaction)
        static = WorkCounts(
            sum(module_current for module_current, _ in shapes),
            sum(module_interaction for _, module_interaction in shapes),
        )
        return LegacyCounts(
            evidence_kind="generated-library-selected-row",
            active=active,
            # The selected-flow timing executes one row, but original
            # AmpliCol first materializes the complete generated library.
            # Preserve that distinction when comparing generation work.
            static=static,
            generated_module_count=len(modules),
            retained_amplitude_count=1,
            module_shapes=shapes,
            selected_module=selected_name,
            limitation=None,
        )

    probes = sorted(artifact.rglob("amplicol_color_library_probe.output"))
    if len(probes) != 1:
        raise CatalogParityError(
            "contracted legacy artifact must contain one color-library probe"
        )
    match = _AFTER_FILTER.search(probes[0].read_text())
    if match is None:
        raise CatalogParityError("contracted legacy probe lacks after-filter counts")
    retained = int(match.group(3))
    static = WorkCounts(
        sum(current for current, _ in shapes),
        sum(interaction for _, interaction in shapes),
    )
    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        current, interaction = next(iter(unique_shapes))
        active = WorkCounts(current * retained, interaction * retained)
        limitation = None
    else:
        active = None
        limitation = (
            "generated modules have nonuniform shapes and the immutable probe "
            "does not record the color-row-to-module replay multiplicities"
        )
    return LegacyCounts(
        evidence_kind="generated-library-contracted-color-replay",
        active=active,
        static=static,
        generated_module_count=len(modules),
        retained_amplitude_count=retained,
        module_shapes=shapes,
        selected_module=None,
        limitation=limitation,
    )


def _candidate_counts(
    result: dict[str, Any],
    *,
    workload: Workload,
) -> CandidateCounts:
    artifact_payload = result.get("artifact")
    if not isinstance(artifact_payload, dict):
        raise CatalogParityError("candidate result has no artifact")
    artifact_raw = artifact_payload.get("path")
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise CatalogParityError("candidate artifact path is absent")
    execution = _read_json(_single_execution(Path(artifact_raw)))
    kind = execution.get("kind")
    selector_available = False
    active_kind = "persisted-plan"

    if kind == "pyamplicol-runtime-recurrence-execution":
        mode = ExecutionMode.RECURRENCE.value
        try:
            inspection = execution["plan"]["inspection_summary"]
            schedule = inspection["schedule"]
            construction = inspection["construction"]
        except (KeyError, TypeError) as error:
            raise CatalogParityError(
                "recurrence execution summary is incomplete"
            ) from error
        final = WorkCounts(
            _positive(schedule.get("current_count"), "recurrence current count"),
            _positive(
                schedule.get("contribution_count"),
                "recurrence contribution count",
            ),
        )
        peak = WorkCounts(
            _positive(
                construction.get("peak_current_count"),
                "recurrence peak current count",
            ),
            _positive(
                construction.get("peak_contribution_count"),
                "recurrence peak contribution count",
            ),
        )
        active = final
        selector = inspection.get("selector_work_certificate")
        if workload is Workload.SELECTED_FLOW and isinstance(selector, dict):
            representatives = selector.get("representatives")
            if isinstance(representatives, list) and representatives:
                parsed = [
                    WorkCounts(
                        _positive(item.get("current_count"), "selector current count"),
                        _positive(
                            item.get("contribution_count"),
                            "selector contribution count",
                        ),
                    )
                    for item in representatives
                    if isinstance(item, dict)
                ]
                if parsed:
                    # The report selector can target any physical flow.  The
                    # worst representative is the fail-closed catalog gate.
                    active = WorkCounts(
                        max(item.current_count for item in parsed),
                        max(item.interaction_count for item in parsed),
                    )
                    selector_available = True
                    active_kind = "selector-active-worst-representative"
    elif kind == "pyamplicol-runtime-execution":
        mode = ExecutionMode.COMPILED.value
        if workload is Workload.ALL_FLOW:
            # The all-flow report workload fixes one physical helicity and
            # sums color.  Its executed lane is the primary DAG.  Unlike the
            # selected-flow (one color, helicity sum) and contracted lanes,
            # it therefore need not expose an auxiliary helicity-sum DAG.
            summary = execution.get("dag_summary")
            active_kind = "compiled-primary-fixed-helicity-all-flow"
        else:
            summary = execution.get("helicity_sum_execution", {}).get("dag_summary")
            active_kind = "compiled-helicity-sum"
        if not isinstance(summary, dict):
            raise CatalogParityError("compiled execution lacks DAG summary")
        final = WorkCounts(
            _positive(summary.get("current_count"), "compiled current count"),
            _positive(summary.get("interaction_count"), "compiled interaction count"),
        )
        peak = final
        active = final
    elif kind == "pyamplicol-runtime-eager-execution":
        mode = ExecutionMode.EAGER.value
        try:
            inspection = execution["plan"]["inspection_summary"]
        except (KeyError, TypeError) as error:
            raise CatalogParityError(
                "eager execution lacks inspection summary"
            ) from error
        final = WorkCounts(
            _positive(inspection.get("current_count"), "eager current count"),
            _positive(inspection.get("attachment_count"), "eager attachment count"),
        )
        peak = final
        active = final
    else:
        raise CatalogParityError(f"unsupported candidate execution kind {kind!r}")

    provenance = result.get("provenance")
    source_revision = None
    if isinstance(provenance, dict):
        value = provenance.get("report_measured_source_revision")
        if isinstance(value, str):
            source_revision = value
    return CandidateCounts(
        mode=mode,
        active=active,
        final_materialized=final,
        peak_materialized=peak,
        active_evidence_kind=active_kind,
        selector_certificate_available=selector_available,
        source_revision=source_revision,
    )


def _ratio(candidate: int, legacy: int) -> float:
    value = candidate / legacy
    if not math.isfinite(value):
        raise CatalogParityError("non-finite structural-work ratio")
    return value


def _load_currents(artifact_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for pointer_path in sorted((artifact_root / "cells").glob("*/current.json")):
        pointer = _read_json(pointer_path)
        cell_id = pointer.get("cell_id")
        manifest_raw = pointer.get("manifest_path")
        if not isinstance(cell_id, str) or not isinstance(manifest_raw, str):
            raise CatalogParityError(f"malformed current pointer {pointer_path}")
        manifest_path = pointer_path.parent / manifest_raw
        manifest = _read_json(manifest_path)
        result_raw = manifest.get("result_path")
        if not isinstance(result_raw, str):
            raise CatalogParityError(f"manifest lacks result path: {manifest_path}")
        result_path = manifest_path.parent / result_raw
        result = _read_json(result_path)
        result["_audit_result_path"] = str(result_path)
        results[cell_id] = result
    return results


def _cell_row(
    cell: Any,
    currents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "cell_id": cell.cell_id,
        "dataset_id": cell.dataset_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "mode": cell.measurement.execution_mode.value,
        "model": None
        if cell.measurement.model is None
        else cell.measurement.model.value,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "limit": MAX_RATIO,
    }
    candidate_result = currents.get(cell.cell_id)
    if candidate_result is None:
        return base | {"status": "candidate-unavailable"}
    candidate_status = candidate_result.get("status")
    if candidate_status != ResultStatus.OK.value:
        return base | {
            "status": "candidate-non-ok",
            "candidate_status": candidate_status,
            "candidate_result_path": candidate_result.get("_audit_result_path"),
        }

    reference = next(
        candidate
        for candidate in REPORT_CATALOG.reference_cells()
        if candidate.process_key == cell.process_key
        and candidate.n_final == cell.n_final
        and candidate.measurement.accuracy == cell.measurement.accuracy
        and candidate.workload == cell.workload
    )
    if not REPORT_CATALOG.legacy_reference_available(cell):
        return base | {
            "status": "legacy-unsupported",
            "candidate_result_path": candidate_result.get("_audit_result_path"),
        }
    legacy_result = currents.get(reference.cell_id)
    if legacy_result is None:
        return base | {
            "status": "legacy-unavailable",
            "legacy_cell_id": reference.cell_id,
            "candidate_result_path": candidate_result.get("_audit_result_path"),
        }
    legacy_status = legacy_result.get("status")
    if legacy_status != ResultStatus.OK.value:
        return base | {
            "status": "legacy-non-ok",
            "legacy_cell_id": reference.cell_id,
            "legacy_status": legacy_status,
            "candidate_result_path": candidate_result.get("_audit_result_path"),
            "legacy_result_path": legacy_result.get("_audit_result_path"),
        }

    try:
        candidate_counts = _candidate_counts(
            candidate_result,
            workload=cell.workload,
        )
        legacy_counts = _legacy_counts(legacy_result, workload=cell.workload)
    except CatalogParityError as error:
        return base | {
            "status": "evidence-unavailable",
            "error": str(error),
            "legacy_cell_id": reference.cell_id,
            "candidate_result_path": candidate_result.get("_audit_result_path"),
            "legacy_result_path": legacy_result.get("_audit_result_path"),
        }

    ratios: dict[str, float | None] = {
        "active_current": None,
        "active_interaction": None,
        "final_static_current": _ratio(
            candidate_counts.final_materialized.current_count,
            legacy_counts.static.current_count,
        ),
        "final_static_interaction": _ratio(
            candidate_counts.final_materialized.interaction_count,
            legacy_counts.static.interaction_count,
        ),
        "peak_static_current": _ratio(
            candidate_counts.peak_materialized.current_count,
            legacy_counts.static.current_count,
        ),
        "peak_static_interaction": _ratio(
            candidate_counts.peak_materialized.interaction_count,
            legacy_counts.static.interaction_count,
        ),
    }
    if legacy_counts.active is not None:
        ratios["active_current"] = _ratio(
            candidate_counts.active.current_count,
            legacy_counts.active.current_count,
        )
        ratios["active_interaction"] = _ratio(
            candidate_counts.active.interaction_count,
            legacy_counts.active.interaction_count,
        )
    violations = tuple(
        key for key, value in ratios.items() if value is not None and value > MAX_RATIO
    )
    unknown = tuple(key for key, value in ratios.items() if value is None)
    status = (
        "ok"
        if not violations and not unknown
        else ("exceeds-1.05" if violations else "active-parity-unresolved")
    )
    return base | {
        "status": status,
        "legacy_cell_id": reference.cell_id,
        "candidate": asdict(candidate_counts),
        "legacy": asdict(legacy_counts),
        "ratios": ratios,
        "violations": violations,
        "unresolved_metrics": unknown,
        "candidate_result_path": candidate_result.get("_audit_result_path"),
        "legacy_result_path": legacy_result.get("_audit_result_path"),
    }


def audit(artifact_root: Path) -> dict[str, Any]:
    currents = _load_currents(artifact_root)
    cells = [
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode
        in {ExecutionMode.RECURRENCE, ExecutionMode.COMPILED, ExecutionMode.EAGER}
    ]
    rows = [_cell_row(cell, currents) for cell in cells]
    status_counts = Counter(row["status"] for row in rows)
    comparable = [
        row
        for row in rows
        if row["status"] in {"ok", "exceeds-1.05", "active-parity-unresolved"}
    ]
    violations = [row for row in comparable if row.get("violations")]
    active_violations = [
        row
        for row in violations
        if any(metric.startswith("active_") for metric in row["violations"])
    ]
    final_static_violations = [
        row
        for row in violations
        if any(metric.startswith("final_static_") for metric in row["violations"])
    ]
    peak_static_violations = [
        row
        for row in violations
        if any(metric.startswith("peak_static_") for metric in row["violations"])
    ]
    return {
        "schema": "pyamplicol-catalog-structural-parity-census-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_root": str(artifact_root.resolve()),
        "policy": {
            "maximum_unreviewed_ratio_to_original_amplicol": MAX_RATIO,
            "active_and_static_materialization_are_independent_gates": True,
            "compiled_and_eager_peak_defaults_to_final_materialization": True,
            "selected_recurrence_active_work_uses_worst_selector_representative": True,
        },
        "summary": {
            "catalog_candidate_cell_count": len(rows),
            "current_pointer_count": len(currents),
            "status_counts": dict(sorted(status_counts.items())),
            "comparable_cell_count": len(comparable),
            "violation_cell_count": len(violations),
            "active_violation_cell_count": len(active_violations),
            "final_static_violation_cell_count": len(final_static_violations),
            "peak_static_violation_cell_count": len(peak_static_violations),
            "fully_certified_catalog_parity": (
                len(comparable) == len(rows) and not violations
            ),
            "violation_by_metric": dict(
                sorted(
                    Counter(
                        metric for row in violations for metric in row["violations"]
                    ).items()
                )
            ),
            "violation_by_mode": dict(
                sorted(Counter(row["mode"] for row in violations).items())
            ),
            "violation_by_process": dict(
                sorted(Counter(row["process_key"] for row in violations).items())
            ),
        },
        "cells": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-complete-parity",
        action="store_true",
        help=(
            "exit nonzero unless every catalog candidate has complete evidence "
            "and all active/final/peak ratios are <=1.05"
        ),
    )
    return parser


def _parity_exit_code(payload: dict[str, Any], *, required: bool) -> int:
    if not required:
        return 0
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return 2
    return 0 if summary.get("fully_certified_catalog_parity") is True else 2


def main() -> int:
    args = _parser().parse_args()
    payload = audit(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return _parity_exit_code(payload, required=args.require_complete_parity)


if __name__ == "__main__":
    raise SystemExit(main())
