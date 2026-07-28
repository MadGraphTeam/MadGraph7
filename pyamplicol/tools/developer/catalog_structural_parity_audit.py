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
_LIBRARY_CALL = re.compile(
    r"AMPICOL_COLOR_PROBE_LIBRARY_CALLS\s+(\d+)\s+(\d+)\s+(\d+)"
)
_PROCESS_ROW = re.compile(r"group:(\d+):integral:(\d+)")
_SUBROUTINE = re.compile(
    r"^\s*subroutine\s+(\w+)\b(.*?)^\s*end\s+subroutine\s+\1\b",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_EXTERNAL_CURRENT = re.compile(
    r"call\s+ext_\w+\s*\(.*?val_c\s*\(\s*1\s*,\s*(\d+)\s*\).*?\)",
    re.IGNORECASE | re.DOTALL,
)
_INT1_ARRAY = re.compile(
    r"integer\s*,\s*parameter\s*,\s*dimension\s*\(([^)]*)\)\s*::\s*"
    r"int1\s*=\s*(?:reshape\s*\(\s*)?\[\s*&?(.*?)\]"
    r"(?:\s*,\s*shape\s*=\s*\[[^]]*\]\s*\))?",
    re.IGNORECASE | re.DOTALL,
)
_COMBINE_SHAPE = re.compile(r"0:(\d+),(\d+)")

MAX_RATIO = 1.05
_DYNAMIC_COLOR_PROJECTION_ABI = (
    "pyamplicol-generic-dag-dynamic-color-projection-v2"
)


@dataclass(frozen=True)
class WorkCounts:
    current_count: int
    evaluation_count: int | None
    attachment_count: int | None


@dataclass(frozen=True)
class LegacyObjectMapping:
    """Exact generated-module mapping to the structural comparison objects."""

    abi: str
    declared_current_count: int
    declared_interaction_count: int
    source_current_count: int
    produced_current_count: int
    vertex_term_count: int
    combine_route_count: int
    current_ids_complete: bool
    interaction_ids_complete: bool
    contribution_references_complete: bool


@dataclass(frozen=True)
class LegacyCounts:
    evidence_kind: str
    active: WorkCounts | None
    static: WorkCounts
    generated_module_count: int
    retained_amplitude_count: int | None
    module_shapes: tuple[tuple[int, int], ...]
    selected_module: str | None
    selected_module_object_mapping: LegacyObjectMapping | None
    limitation: str | None


@dataclass(frozen=True)
class CandidateCounts:
    mode: str
    active: WorkCounts
    final_materialized: WorkCounts
    peak_materialized: WorkCounts
    active_evidence_kind: str
    active_evidence_exact: bool
    selector_certificate_available: bool
    source_revision: str | None
    active_closure_count: int | None
    final_materialized_closure_count: int | None
    peak_materialized_closure_count: int | None


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


def _optional_positive(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive(value, label)


def _optional_nonnegative(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CatalogParityError(f"{label} must be a nonnegative integer")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CatalogParityError(f"{label} must be boolean")
    return value


def _compiled_execution_lanes(
    execution: dict[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Enumerate every physically persisted compiled DAG lane exactly once."""

    lanes: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[object, ...]] = set()

    def payload_paths(value: object) -> tuple[str, ...]:
        paths: list[str] = []

        def visit(item: object, field: str | None = None) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, str(key))
            elif isinstance(item, list):
                for child in item:
                    visit(child, field)
            elif (
                isinstance(item, str)
                and field is not None
                and (field == "path" or field.endswith("_path"))
            ):
                paths.append(item)

        visit(value)
        return tuple(sorted(set(paths)))

    def add(record: object, path: str) -> None:
        if not isinstance(record, dict):
            raise CatalogParityError(f"compiled execution lane {path} is malformed")
        summary = record.get("dag_summary")
        if not isinstance(summary, dict):
            raise CatalogParityError(
                f"compiled execution lane {path} lacks DAG summary"
            )
        physical_paths = payload_paths(record.get("compiled"))
        identity: tuple[object, ...] = (
            ("payloads", *physical_paths)
            if physical_paths
            else ("manifest-path", path)
        )
        if identity in seen:
            return
        seen.add(identity)
        lanes.append((path, record))
        helicity_sum = record.get("helicity_sum_execution")
        if helicity_sum is not None:
            add(helicity_sum, f"{path}.helicity_sum_execution")
        for key in ("color_selector_executions", "helicity_selector_executions"):
            raw = record.get(key, [])
            if raw is None:
                continue
            if not isinstance(raw, list):
                raise CatalogParityError(f"compiled {path}.{key} is malformed")
            for index, wrapper in enumerate(raw):
                if not isinstance(wrapper, dict):
                    raise CatalogParityError(
                        f"compiled {path}.{key}[{index}] is malformed"
                    )
                add(wrapper.get("execution"), f"{path}.{key}[{index}].execution")

    add(execution, "primary")
    return tuple(lanes)


def _sum_compiled_materialization(
    lanes: tuple[tuple[str, dict[str, Any]], ...],
) -> tuple[WorkCounts, int | None]:
    summaries = [record["dag_summary"] for _path, record in lanes]
    evaluations = [
        _optional_positive(
            summary.get("interaction_evaluation_count"),
            "compiled interaction evaluation count",
        )
        for summary in summaries
    ]
    closures = [
        _optional_nonnegative(
            summary.get("amplitude_root_count"),
            "compiled amplitude-root count",
        )
        for summary in summaries
    ]
    return (
        WorkCounts(
            sum(
                _positive(summary.get("current_count"), "compiled current count")
                for summary in summaries
            ),
            (
                None
                if any(value is None for value in evaluations)
                else sum(value for value in evaluations if value is not None)
            ),
            sum(
                _positive(
                    summary.get("interaction_count"),
                    "compiled interaction attachment count",
                )
                for summary in summaries
            ),
        ),
        (
            None
            if any(value is None for value in closures)
            else sum(value for value in closures if value is not None)
        ),
    )


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


def _module_object_mapping(path: Path) -> LegacyObjectMapping:
    """Prove val_c/int_c are the same objects counted by candidate schedules.

    Original AmpliCol's generated module stores external and produced currents
    in ``val_c``.  Every vertex term is stored in ``int_c`` and then referenced
    by a current-combination routine.  The proof below rejects sparse arrays or
    dangling terms instead of treating a maximum array index as work.
    """

    text = path.read_text()
    declared_currents, declared_interactions = _module_shape(path)
    source_ids = {int(value) for value in _EXTERNAL_CURRENT.findall(text)}
    produced_ids: set[int] = set()
    vertex_term_ids: set[int] = set()
    contribution_reference_ids: set[int] = set()
    combine_route_count = 0

    for match in _SUBROUTINE.finditer(text):
        name = match.group(1).lower()
        body = match.group(2)
        array = _INT1_ARRAY.search(body)
        if array is None:
            continue
        values = [int(value) for value in re.findall(r"\d+", array.group(2))]
        if name.startswith("vertex_"):
            vertex_term_ids.update(values)
            continue
        if not name.startswith("combine_currents"):
            continue
        shape = _COMBINE_SHAPE.fullmatch(array.group(1).replace(" ", ""))
        if shape is None:
            raise CatalogParityError(
                f"legacy current-combination int1 shape is unsupported: {path}"
            )
        stride = int(shape.group(1)) + 1
        column_count = int(shape.group(2))
        if len(values) != stride * column_count:
            raise CatalogParityError(
                f"legacy current-combination int1 payload has the wrong size: {path}"
            )
        for start in range(0, len(values), stride):
            produced_ids.add(values[start])
            contribution_ids = values[start + 1 : start + stride]
            contribution_reference_ids.update(contribution_ids)
            combine_route_count += len(contribution_ids)

    expected_currents = set(range(1, declared_currents + 1))
    expected_interactions = set(range(1, declared_interactions + 1))
    mapping = LegacyObjectMapping(
        abi="pyamplicol-legacy-module-object-mapping-v1",
        declared_current_count=declared_currents,
        declared_interaction_count=declared_interactions,
        source_current_count=len(source_ids),
        produced_current_count=len(produced_ids),
        vertex_term_count=len(vertex_term_ids),
        combine_route_count=combine_route_count,
        current_ids_complete=(source_ids | produced_ids) == expected_currents,
        interaction_ids_complete=vertex_term_ids == expected_interactions,
        contribution_references_complete=(
            contribution_reference_ids == expected_interactions
        ),
    )
    if not (
        mapping.current_ids_complete
        and mapping.interaction_ids_complete
        and mapping.contribution_references_complete
    ):
        raise CatalogParityError(
            f"legacy module has sparse or dangling current/interaction objects: {path}"
        )
    return mapping


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
    module_mappings = tuple(_module_object_mapping(module) for module in modules)

    if workload is Workload.ALL_FLOW:
        log = artifact / "legacy.log"
        text = log.read_text()
        current = _DIRECT_CURRENT.search(text)
        interaction = _DIRECT_INTERACTION.search(text)
        amplitude = _DIRECT_AMPLITUDE.search(text)
        if current is None or interaction is None or amplitude is None:
            raise CatalogParityError("all-flow legacy log lacks structural counters")
        counts = WorkCounts(
            int(current.group(1)),
            int(interaction.group(1)),
            None,
        )
        return LegacyCounts(
            evidence_kind="direct-fixed-helicity-all-flow-probe",
            active=counts,
            static=counts,
            generated_module_count=0,
            retained_amplitude_count=int(amplitude.group(1)),
            module_shapes=(),
            selected_module=None,
            selected_module_object_mapping=None,
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
        object_mapping = _module_object_mapping(selected[0])
        active = WorkCounts(
            current,
            interaction,
            object_mapping.combine_route_count,
        )
        static = WorkCounts(
            sum(module_current for module_current, _ in shapes),
            sum(module_interaction for _, module_interaction in shapes),
            sum(mapping.combine_route_count for mapping in module_mappings),
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
            selected_module_object_mapping=object_mapping,
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
    row = artifact_payload.get("process_row")
    if not isinstance(row, str):
        raise CatalogParityError("contracted legacy result lacks process row")
    row_match = _PROCESS_ROW.fullmatch(row)
    if row_match is None:
        raise CatalogParityError(f"malformed contracted process row: {row!r}")
    selected_name = f"amp{row_match.group(1)}_{row_match.group(2)}_lib.f03"
    selected = [module for module in modules if module.name == selected_name]
    if len(selected) != 1:
        raise CatalogParityError(
            f"contracted module {selected_name} occurs {len(selected)} times"
        )
    object_mapping = _module_object_mapping(selected[0])
    retained = int(match.group(3))
    static = WorkCounts(
        sum(current for current, _ in shapes),
        sum(interaction for _, interaction in shapes),
        sum(mapping.combine_route_count for mapping in module_mappings),
    )
    calls = {
        (int(group), int(integral)): int(count)
        for group, integral, count in _LIBRARY_CALL.findall(probes[0].read_text())
    }
    active: WorkCounts | None = None
    limitation = (
        "contracted probe predates exact generated-library call histogram"
    )
    if calls:
        by_name = {module.name: module for module in modules}
        missing = [
            f"amp{group}_{integral}_lib.f03"
            for group, integral in calls
            if f"amp{group}_{integral}_lib.f03" not in by_name
        ]
        if missing:
            raise CatalogParityError(
                "contracted probe call histogram references absent modules: "
                + ", ".join(sorted(missing))
            )
        active = WorkCounts(
            sum(
                _module_shape(by_name[f"amp{group}_{integral}_lib.f03"])[0] * count
                for (group, integral), count in calls.items()
            ),
            sum(
                _module_shape(by_name[f"amp{group}_{integral}_lib.f03"])[1] * count
                for (group, integral), count in calls.items()
            ),
            sum(
                _module_object_mapping(
                    by_name[f"amp{group}_{integral}_lib.f03"]
                ).combine_route_count
                * count
                for (group, integral), count in calls.items()
            ),
        )
        limitation = None
    return LegacyCounts(
        evidence_kind=(
            "generated-library-contracted-exact-call-histogram"
            if active is not None
            else "generated-library-contracted-call-histogram-unavailable"
        ),
        active=active,
        static=static,
        generated_module_count=len(modules),
        retained_amplitude_count=retained,
        module_shapes=shapes,
        selected_module=selected_name,
        selected_module_object_mapping=object_mapping,
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
    active_exact = workload is Workload.CONTRACTED
    active_closures: int | None = None
    final_closures: int | None = None
    peak_closures: int | None = None

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
            _positive(
                construction.get("peak_contribution_count"),
                "recurrence peak contribution count",
            ),
        )
        active = final
        final_closures = _optional_nonnegative(
            schedule.get("closure_term_count"),
            "recurrence closure-term count",
        )
        active_closures = final_closures
        peak_closures = _optional_nonnegative(
            construction.get("peak_closure_term_count"),
            "recurrence peak closure-term count",
        )
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
                        max(
                            item.evaluation_count
                            for item in parsed
                            if item.evaluation_count is not None
                        ),
                        max(
                            item.attachment_count
                            for item in parsed
                            if item.attachment_count is not None
                        ),
                    )
                    selector_available = True
                    active_exact = True
                    active_kind = "selector-active-worst-representative"
                    active_closures = max(
                        (
                            _optional_nonnegative(
                                item.get("closure_count"),
                                "selector closure count",
                            )
                            or 0
                        )
                        for item in representatives
                        if isinstance(item, dict)
                    )
    elif kind == "pyamplicol-runtime-execution":
        mode = ExecutionMode.COMPILED.value
        if workload is Workload.ALL_FLOW:
            # The all-flow report workload fixes one physical helicity and
            # sums color.  Its executed lane is the primary DAG.  Unlike the
            # selected-flow (one color, helicity sum) and contracted lanes,
            # it therefore need not expose an auxiliary helicity-sum DAG.
            summary = execution.get("dag_summary")
            active_kind = "compiled-primary-fixed-helicity-all-flow"
            selector_records = execution.get("helicity_selector_executions")
        else:
            helicity_sum = execution.get("helicity_sum_execution")
            if not isinstance(helicity_sum, dict):
                raise CatalogParityError("compiled execution lacks helicity-sum lane")
            summary = helicity_sum.get("dag_summary")
            active_kind = "compiled-helicity-sum"
            selector_records = (
                helicity_sum.get("color_selector_executions")
                if workload is Workload.SELECTED_FLOW
                else None
            )
        if not isinstance(summary, dict):
            raise CatalogParityError("compiled execution lacks DAG summary")
        active = WorkCounts(
            _positive(summary.get("current_count"), "compiled current count"),
            _optional_positive(
                summary.get("interaction_evaluation_count"),
                "compiled interaction evaluation count",
            ),
            _positive(
                summary.get("interaction_count"),
                "compiled interaction attachment count",
            ),
        )
        active_closures = _optional_nonnegative(
            summary.get("amplitude_root_count"),
            "compiled active amplitude-root count",
        )
        final, final_closures = _sum_compiled_materialization(
            _compiled_execution_lanes(execution)
        )
        peak = final
        peak_closures = final_closures
        if isinstance(selector_records, list) and selector_records:
            selector_counts: list[WorkCounts] = []
            for record in selector_records:
                if not isinstance(record, dict):
                    continue
                selector_execution = record.get("execution")
                if not isinstance(selector_execution, dict):
                    continue
                selector_summary = selector_execution.get("dag_summary")
                if not isinstance(selector_summary, dict):
                    continue
                selector_counts.append(
                    WorkCounts(
                        _positive(
                            selector_summary.get("current_count"),
                            "compiled selector current count",
                        ),
                        _optional_positive(
                            selector_summary.get("interaction_evaluation_count"),
                            "compiled selector interaction evaluation count",
                        ),
                        _positive(
                            selector_summary.get("interaction_count"),
                            "compiled selector interaction attachment count",
                        ),
                    )
                )
            if selector_counts:
                active = WorkCounts(
                    max(item.current_count for item in selector_counts),
                    (
                        None
                        if any(
                            item.evaluation_count is None for item in selector_counts
                        )
                        else max(
                            item.evaluation_count
                            for item in selector_counts
                            if item.evaluation_count is not None
                        )
                    ),
                    max(
                        item.attachment_count
                        for item in selector_counts
                        if item.attachment_count is not None
                    ),
                )
                selector_available = True
                active_exact = True
                active_kind = (
                    "compiled-selected-color-worst-sector"
                    if workload is Workload.SELECTED_FLOW
                    else "compiled-fixed-helicity-worst-class"
                )
                active_closures = max(
                    (
                        _optional_nonnegative(
                            record.get("execution", {})
                            .get("dag_summary", {})
                            .get("amplitude_root_count"),
                            "compiled selector amplitude-root count",
                        )
                        or 0
                    )
                    for record in selector_records
                    if isinstance(record, dict)
                )
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
            _optional_positive(
                inspection.get("invocation_count"),
                "eager invocation count",
            ),
            _positive(inspection.get("attachment_count"), "eager attachment count"),
        )
        peak = final
        active = final
        final_closures = _optional_nonnegative(
            inspection.get("closure_count"),
            "eager closure count",
        )
        active_closures = final_closures
        peak_closures = final_closures
        selector_work = inspection.get("selector_work")
        if isinstance(selector_work, dict):
            if selector_work.get("abi") != "pyamplicol-eager-selector-work-v1":
                raise CatalogParityError("eager selector-work ABI is unsupported")
            prefix = {
                Workload.SELECTED_FLOW: "selected_flow",
                Workload.ALL_FLOW: "all_flow",
                Workload.CONTRACTED: "contracted",
            }[workload]
            active = WorkCounts(
                _positive(
                    selector_work.get(f"{prefix}_current_count"),
                    f"eager {prefix} active current count",
                ),
                _positive(
                    selector_work.get(f"{prefix}_evaluation_count"),
                    f"eager {prefix} active evaluation count",
                ),
                _positive(
                    selector_work.get(f"{prefix}_attachment_count"),
                    f"eager {prefix} active attachment count",
                ),
            )
            selector_available = True
            active_exact = True
            active_kind = f"eager-{prefix.replace('_', '-')}-selector-census"
            active_closures = _optional_nonnegative(
                selector_work.get(f"{prefix}_closure_count"),
                f"eager {prefix} active closure count",
            )
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
        active_evidence_exact=active_exact,
        selector_certificate_available=selector_available,
        source_revision=source_revision,
        active_closure_count=active_closures,
        final_materialized_closure_count=final_closures,
        peak_materialized_closure_count=peak_closures,
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
        "active_evaluation": None,
        "active_attachment": None,
        "final_static_current": _ratio(
            candidate_counts.final_materialized.current_count,
            legacy_counts.static.current_count,
        ),
        "final_static_evaluation": None,
        "final_static_attachment": None,
        "peak_static_current": _ratio(
            candidate_counts.peak_materialized.current_count,
            legacy_counts.static.current_count,
        ),
        "peak_static_evaluation": None,
        "peak_static_attachment": None,
    }
    if (
        candidate_counts.final_materialized.evaluation_count is not None
        and legacy_counts.static.evaluation_count is not None
    ):
        ratios["final_static_evaluation"] = _ratio(
            candidate_counts.final_materialized.evaluation_count,
            legacy_counts.static.evaluation_count,
        )
    if (
        candidate_counts.final_materialized.attachment_count is not None
        and legacy_counts.static.attachment_count is not None
    ):
        ratios["final_static_attachment"] = _ratio(
            candidate_counts.final_materialized.attachment_count,
            legacy_counts.static.attachment_count,
        )
    if (
        candidate_counts.peak_materialized.evaluation_count is not None
        and legacy_counts.static.evaluation_count is not None
    ):
        ratios["peak_static_evaluation"] = _ratio(
            candidate_counts.peak_materialized.evaluation_count,
            legacy_counts.static.evaluation_count,
        )
    if (
        candidate_counts.peak_materialized.attachment_count is not None
        and legacy_counts.static.attachment_count is not None
    ):
        ratios["peak_static_attachment"] = _ratio(
            candidate_counts.peak_materialized.attachment_count,
            legacy_counts.static.attachment_count,
        )
    if legacy_counts.active is not None and candidate_counts.active_evidence_exact:
        ratios["active_current"] = _ratio(
            candidate_counts.active.current_count,
            legacy_counts.active.current_count,
        )
        if (
            candidate_counts.active.evaluation_count is not None
            and legacy_counts.active.evaluation_count is not None
        ):
            ratios["active_evaluation"] = _ratio(
                candidate_counts.active.evaluation_count,
                legacy_counts.active.evaluation_count,
            )
        if (
            candidate_counts.active.attachment_count is not None
            and legacy_counts.active.attachment_count is not None
        ):
            ratios["active_attachment"] = _ratio(
                candidate_counts.active.attachment_count,
                legacy_counts.active.attachment_count,
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


def _source_work_counts(value: object, label: str) -> WorkCounts:
    if not isinstance(value, dict):
        raise CatalogParityError(f"{label} must be an object")
    return WorkCounts(
        _positive(value.get("current_count"), f"{label} current count"),
        _positive(value.get("evaluation_count"), f"{label} evaluation count"),
        _positive(value.get("attachment_count"), f"{label} attachment count"),
    )


def _require_exact_projection_certificate(
    value: object,
    label: str,
    *,
    expected_source_revision: str,
    projection_applied: bool,
) -> None:
    if value is None and not projection_applied:
        return
    if not isinstance(value, dict):
        raise CatalogParityError(f"{label} projection certificate is malformed")
    required_digests = (
        "source_semantics_sha256",
        "selector_domains_sha256",
        "current_class_members_sha256",
        "current_remap_sha256",
        "interaction_groups_sha256",
        "closure_groups_sha256",
        "rectangle_cardinalities_sha256",
        "row_identity_sha256",
    )
    remap = value.get("old_to_new_current_ids")
    if (
        value.get("abi") != _DYNAMIC_COLOR_PROJECTION_ABI
        or value.get("source_revision") != expected_source_revision
        or value.get("applied") is not projection_applied
        or value.get("equality_check_status")
        not in {
            "passed-exact-cartesian-products-no-duplicates",
            "passed-no-projectable-classes",
        }
        or not isinstance(remap, list)
        or not remap
        or any(not isinstance(item, int) or item < 0 for item in remap)
        or any(
            not isinstance(value.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None
            for field in required_digests
        )
    ):
        raise CatalogParityError(
            f"{label} lacks a complete exact rectangular projection proof"
        )


def audit_source_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on a source-derived, timing-free all-catalog census."""

    if payload.get("schema") != "pyamplicol-catalog-source-structural-evidence-v1":
        raise CatalogParityError("source structural evidence schema is unsupported")
    source_revision = payload.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise CatalogParityError("source structural evidence revision is invalid")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise CatalogParityError("source structural evidence cells must be a list")
    expected_cells = tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode
        in {ExecutionMode.RECURRENCE, ExecutionMode.COMPILED, ExecutionMode.EAGER}
    )
    expected_by_id = {cell.cell_id: cell for cell in expected_cells}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_cells:
        if not isinstance(raw, dict) or not isinstance(raw.get("cell_id"), str):
            raise CatalogParityError("source structural evidence has malformed cell")
        cell_id = raw["cell_id"]
        if cell_id in rows_by_id:
            raise CatalogParityError(
                f"source structural evidence repeats cell {cell_id}"
            )
        rows_by_id[cell_id] = raw
    missing = sorted(set(expected_by_id) - set(rows_by_id))
    extra = sorted(set(rows_by_id) - set(expected_by_id))
    if missing or extra:
        raise CatalogParityError(
            "source structural evidence does not cover the exact catalog "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    audited: list[dict[str, Any]] = []
    violation_count = 0
    unresolved_count = 0
    scope_unavailable_count = 0
    for cell_id, cell in expected_by_id.items():
        raw = rows_by_id[cell_id]
        if raw.get("source_revision") != source_revision:
            raise CatalogParityError(
                f"source structural evidence {cell_id} revision mismatch"
            )
        if raw.get("proof_strength") != "exact-source-construction":
            raise CatalogParityError(
                f"source structural evidence {cell_id} is not an exact proof"
            )
        candidate = raw.get("candidate")
        if not isinstance(candidate, dict):
            raise CatalogParityError(f"{cell_id} candidate evidence is absent")
        candidate_active = _source_work_counts(
            candidate.get("active"),
            f"{cell_id} candidate active",
        )
        candidate_final = _source_work_counts(
            candidate.get("final_materialized"),
            f"{cell_id} candidate final",
        )
        candidate_peak = _source_work_counts(
            candidate.get("peak_materialized"),
            f"{cell_id} candidate peak",
        )
        _optional_nonnegative(
            candidate.get("active_closure_count"),
            f"{cell_id} candidate active closure count",
        )
        _optional_nonnegative(
            candidate.get("final_closure_count"),
            f"{cell_id} candidate final closure count",
        )
        _optional_nonnegative(
            candidate.get("peak_closure_count"),
            f"{cell_id} candidate peak closure count",
        )
        _require_exact_projection_certificate(
            candidate.get("dynamic_color_projection"),
            cell_id,
            expected_source_revision=source_revision,
            projection_applied=_require_bool(
                candidate.get("dynamic_color_projection_applied"),
                f"{cell_id} candidate dynamic-color projection applied",
            ),
        )

        if not REPORT_CATALOG.legacy_reference_available(cell):
            if (
                raw.get("status") != "legacy-scope-unavailable"
                or not isinstance(raw.get("scope_reason"), str)
                or raw.get("legacy") is not None
            ):
                raise CatalogParityError(
                    f"{cell_id} must explicitly record legacy-scope-unavailable"
                )
            scope_unavailable_count += 1
            audited.append(
                {
                    "cell_id": cell_id,
                    "status": "legacy-scope-unavailable",
                    "scope_reason": raw["scope_reason"],
                    "candidate": {
                        "active": asdict(candidate_active),
                        "final_materialized": asdict(candidate_final),
                        "peak_materialized": asdict(candidate_peak),
                    },
                    "ratios": None,
                    "violations": (),
                }
            )
            continue

        if raw.get("status") != "ok":
            raise CatalogParityError(
                f"{cell_id} comparable source evidence status must be ok"
            )
        legacy = raw.get("legacy")
        if not isinstance(legacy, dict):
            raise CatalogParityError(f"{cell_id} legacy evidence is absent")
        legacy_active = _source_work_counts(
            legacy.get("active"),
            f"{cell_id} legacy active",
        )
        legacy_static = _source_work_counts(
            legacy.get("static"),
            f"{cell_id} legacy static",
        )
        ratios = {
            "active_current": _ratio(
                candidate_active.current_count, legacy_active.current_count
            ),
            "active_evaluation": _ratio(
                candidate_active.evaluation_count or 0,
                legacy_active.evaluation_count or 0,
            ),
            "active_attachment": _ratio(
                candidate_active.attachment_count or 0,
                legacy_active.attachment_count or 0,
            ),
            "final_static_current": _ratio(
                candidate_final.current_count, legacy_static.current_count
            ),
            "final_static_evaluation": _ratio(
                candidate_final.evaluation_count or 0,
                legacy_static.evaluation_count or 0,
            ),
            "final_static_attachment": _ratio(
                candidate_final.attachment_count or 0,
                legacy_static.attachment_count or 0,
            ),
            "peak_static_current": _ratio(
                candidate_peak.current_count, legacy_static.current_count
            ),
            "peak_static_evaluation": _ratio(
                candidate_peak.evaluation_count or 0,
                legacy_static.evaluation_count or 0,
            ),
            "peak_static_attachment": _ratio(
                candidate_peak.attachment_count or 0,
                legacy_static.attachment_count or 0,
            ),
        }
        violations = tuple(
            name for name, ratio in ratios.items() if ratio > MAX_RATIO
        )
        violation_count += bool(violations)
        audited.append(
            {
                "cell_id": cell_id,
                "status": "ok" if not violations else "exceeds-1.05",
                "candidate": {
                    "active": asdict(candidate_active),
                    "final_materialized": asdict(candidate_final),
                    "peak_materialized": asdict(candidate_peak),
                },
                "legacy": {
                    "active": asdict(legacy_active),
                    "static": asdict(legacy_static),
                },
                "ratios": ratios,
                "violations": violations,
            }
        )
    comparable_count = len(audited) - scope_unavailable_count
    return {
        "schema": "pyamplicol-catalog-source-structural-parity-preflight-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision,
        "policy": {
            "maximum_ratio_to_original_amplicol": MAX_RATIO,
            "currents_evaluations_and_attachments_are_independent_gates": True,
            "amplitude_roots_are_reported_but_not_conflated_with_int_c": True,
            "legacy_scope_unavailable_is_explicit_and_has_no_fabricated_ratio": True,
        },
        "summary": {
            "catalog_candidate_cell_count": len(expected_cells),
            "comparable_cell_count": comparable_count,
            "legacy_scope_unavailable_cell_count": scope_unavailable_count,
            "violation_cell_count": violation_count,
            "unresolved_cell_count": unresolved_count,
            "complete_catalog_coverage": True,
            "all_comparable_cells_within_limit": violation_count == 0,
            "fully_certified_catalog_parity": violation_count == 0,
        },
        "cells": audited,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-root", type=Path)
    source.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "timing-free source-construction evidence covering every matrix "
            "candidate cell"
        ),
    )
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
    payload = (
        audit_source_manifest(_read_json(args.source_manifest))
        if args.source_manifest is not None
        else audit(args.artifact_root)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return _parity_exit_code(payload, required=args.require_complete_parity)


if __name__ == "__main__":
    raise SystemExit(main())
