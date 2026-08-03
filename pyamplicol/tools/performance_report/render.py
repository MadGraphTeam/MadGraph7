# SPDX-License-Identifier: 0BSD
"""Dynamic baseline joins and fixed-block TeX rendering for the report tables."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .agreements import (
    DIRECT_AGREEMENT_ABI,
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
    incoming_agreement_edges,
)
from .cache import empty_measurement
from .campaign_policy import (
    STRICT_POLICY,
    CampaignPolicy,
    PolicyMeasurementState,
    policy_measurement_state_hint,
    policy_status_label,
)
from .catalog import (
    REPORT_CATALOG,
    STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6,
    STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6_DESCRIPTION,
    ReportCatalog,
    matrix_multiplicities,
    z_dataset_id,
)
from .display_contract import report_display_accounting
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    MatrixDataset,
    ModelKey,
    ProcessFamily,
    ResultStatus,
    ScalarDataset,
    Workload,
    ZVariant,
)
from .timing import (
    evaluator_total_seconds_per_point,
    evaluator_total_timing_record,
    recurrence_core_seconds_per_point,
    unavailable_execution_timing_record,
)
from .validation_summary import (
    SUMMARY_TABLE_NAME,
    render_validation_summary,
)

Measurement = Mapping[str, object]
CachePayload = Mapping[str, object]

_NA = empty_measurement()
_MATRIX_BLOCK_SIZE = 3
_Z_BLOCK_SIZE = 3
_BEST_MODE_CODES = {
    ExecutionMode.RECURRENCE: "r",
    ExecutionMode.COMPILED: "c",
    ExecutionMode.EAGER: "e",
}
_BEST_MODE_ORDER = tuple(_BEST_MODE_CODES)
_BEST_MODE_TABLE_NAMES = {
    Accuracy.LC: "result_matrix_best_builtin_sm_lc_table.tex",
    Accuracy.NLC: "result_matrix_best_builtin_sm_nlc_table.tex",
    Accuracy.FULL: "result_matrix_best_builtin_sm_full_table.tex",
}
_PRESENTATION_FAILURE_KIND_PREFIX = "ManualCampaignOutcome:"
_SAFE_OUTCOME_SLUG = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_TIME_CAP_LABEL = re.compile(r">(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>h|s)")
_RAM_CAP_LABEL = re.compile(r">(?P<value>[0-9]+)GB")
_POLICY_PRESENTATION_OUTCOME_SLUGS = frozenset(
    state.value
    for state in PolicyMeasurementState
    if state is not PolicyMeasurementState.SUCCESS
)
_MAX_BEST_MODE_TERMINAL_LABEL_CHARS = 48
_TERMINAL_LABEL_DIGEST_CHARS = 6
_KNOWN_PRESENTATION_TERMINAL_LABELS = {
    "blocked_dependency": "blocked dependency",
    "cancelled": "interrupted",
    "error": "error",
    "failed": "failed",
    "generation_limit": "generation limit",
    "interrupted": "interrupted",
    "memory_limit": "memory limit",
    "preparation_error": "preparation error",
    "preparation_failed": "preparation failed",
    "profiling_timeout": "profiling timeout",
    "resource_frontier": "resource frontier",
    "skip": "skip",
    "static_na": "static na",
    "unsupported": "unsupported",
    "validation_failed": "validation failed",
    "validation_timeout": "validation timeout",
    "worker_timeout": "worker timeout",
}
_KNOWN_RESULT_TERMINAL_LABELS = frozenset(
    {"N/A", "t/o", "RAM", "skip", "validation failed", "unsupported", "failed", "error"}
)
_POLICY_TERMINAL_DISPLAY_LABEL = re.compile(
    r"(?:(?:worker|profile|validation) >[0-9]{1,4}(?:\.[0-9]{1,4})?(?:h|s)"
    r"|>[0-9]{1,4}(?:\.[0-9]{1,4})?(?:h|s)|>[0-9]{1,4}GB"
    r"|dependency(?: (?:>[0-9]{1,4}(?:\.[0-9]{1,4})?(?:h|s)"
    r"|>[0-9]{1,4}GB|blocked))?)"
)
_COMPACT_OUTCOME_COUNT_LABEL = re.compile(
    r"[1-9][0-9]* outcomes \[(?P<digest>[0-9a-f]{6})\]"
)


def _chunks(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(values[start : start + size]) for start in range(0, len(values), size)
    )


def _cache_dataset_id(payload: CachePayload) -> str:
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("cache payload has no dataset_id")
    return dataset_id


def _entry_key(
    dataset_id: str,
    entry: Mapping[str, object],
) -> tuple[str, str | None, int, str, str | None]:
    process_key = entry.get("process_key")
    if process_key is not None and not isinstance(process_key, str):
        raise ValueError("cache process_key must be a string or null")
    n_final = entry.get("n_final")
    if isinstance(n_final, bool) or not isinstance(n_final, int):
        raise ValueError("cache n_final must be an integer")
    workload = entry.get("workload")
    if not isinstance(workload, str):
        raise ValueError("cache workload must be a string")
    variant = entry.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ValueError("cache variant must be a string or null")
    return dataset_id, process_key, n_final, workload, variant


class MeasurementIndex:
    """Read-only index over mode-owned caches.

    The index never copies measurements between datasets. Report views locate the
    candidate and baseline independently and compute ratios while rendering.
    """

    def __init__(self, caches: Mapping[str, CachePayload] | Iterable[CachePayload]):
        payloads = caches.values() if isinstance(caches, Mapping) else caches
        entries: dict[
            tuple[str, str | None, int, str, str | None],
            Measurement,
        ] = {}
        for payload in payloads:
            dataset_id = _cache_dataset_id(payload)
            raw_entries = payload.get("entries")
            if not isinstance(raw_entries, list):
                raise ValueError(f"cache {dataset_id!r} has no entries list")
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, Mapping):
                    raise ValueError(
                        f"cache {dataset_id!r} contains a non-object entry"
                    )
                key = _entry_key(dataset_id, raw_entry)
                if key in entries:
                    raise ValueError(f"duplicate report measurement key {key!r}")
                measurement = raw_entry.get("measurement")
                if not isinstance(measurement, Mapping):
                    raise ValueError(f"cache entry {key!r} has no measurement object")
                entries[key] = measurement
        self._entries = entries

    def get(
        self,
        dataset_id: str,
        process_key: str | None,
        n_final: int,
        workload: Workload,
        *,
        variant: str | None = None,
    ) -> Measurement:
        return self._entries.get(
            (dataset_id, process_key, n_final, workload.value, variant),
            _NA,
        )

    def for_cell(self, cell: CellSpec) -> Measurement:
        """Return the current measurement owned by one canonical catalog cell."""

        return self.get(
            cell.dataset_id,
            cell.process_key,
            cell.n_final,
            cell.workload,
            variant=cell.variant,
        )


@dataclass(frozen=True, slots=True)
class JoinedWorkload:
    workload: Workload
    baseline: Measurement
    candidate: Measurement
    comparison_linked: bool = False


@dataclass(frozen=True, slots=True)
class JoinedMatrixCell:
    dataset: MatrixDataset
    process_family: ProcessFamily
    n_final: int
    applicable: bool
    workloads: tuple[JoinedWorkload, ...]


@dataclass(frozen=True, slots=True)
class _TerminalOutcome:
    """One terminal result with identity and color kept separate from its label."""

    identity: str
    label: str
    color: str


@dataclass(frozen=True, slots=True)
class _TerminalSummary:
    """A bounded display label retaining every represented terminal outcome."""

    outcomes: tuple[_TerminalOutcome, ...]
    label: str
    color: str


@dataclass(frozen=True, slots=True)
class BestModeWorkload:
    workload: Workload
    baseline: Measurement
    candidate: Measurement
    mode: ExecutionMode | None
    terminal_label: _TerminalSummary | None
    comparison_linked: bool = False


@dataclass(frozen=True, slots=True)
class BestModeMatrixCell:
    process_family: ProcessFamily
    n_final: int
    accuracy: Accuracy
    applicable: bool
    workloads: tuple[BestModeWorkload, ...]


@dataclass(frozen=True, slots=True)
class _BestModeComparisonLayout:
    """One comparison token in both inline and column-aligned forms."""

    inline: str
    prefix: str
    primary: str


@dataclass(frozen=True, slots=True)
class _BestModeCellRows:
    """Physical generation/runtime rows for one multiplicity cell."""

    generation: tuple[str, ...]
    runtime: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MatrixCellRows:
    """Physical generation/runtime rows for one detailed matrix cell."""

    generation: tuple[str, ...]
    runtime: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisibleCompleteness:
    """Render-level evidence for the publication's measured-cell contract."""

    maximum_n_final: int
    required_measurement_count: int
    rendered_required_measurement_count: int
    structurally_not_applicable_display_slot_count: int
    not_exposed_display_slot_count: int
    applicable_na_display_slots: tuple[str, ...]
    missing_rendered_cell_ids: tuple[str, ...]
    contract_errors: tuple[str, ...]
    declared_measurement_cell_count: int = 0
    catalog_static_na_cell_count: int = 0
    rendered_catalog_static_na_cell_count: int = 0

    @property
    def complete(self) -> bool:
        return not (
            self.applicable_na_display_slots
            or self.missing_rendered_cell_ids
            or self.contract_errors
            or self.declared_measurement_cell_count
            != (self.required_measurement_count + self.catalog_static_na_cell_count)
            or self.rendered_required_measurement_count
            != self.required_measurement_count
            or self.rendered_catalog_static_na_cell_count
            != self.catalog_static_na_cell_count
        )

    def as_dict(self) -> dict[str, object]:
        """Return compact, JSON-safe final-audit evidence."""

        preview_limit = 50
        return {
            "kind": "pyamplicol-report-visible-completeness",
            "schema_version": 2,
            "status": "ok" if self.complete else "incomplete",
            "maximum_n_final": self.maximum_n_final,
            "declared_measurement_cell_count": (self.declared_measurement_cell_count),
            "required_measurement_count": self.required_measurement_count,
            "rendered_required_measurement_count": (
                self.rendered_required_measurement_count
            ),
            "structurally_not_applicable_display_slot_count": (
                self.structurally_not_applicable_display_slot_count
            ),
            "not_exposed_display_slot_count": self.not_exposed_display_slot_count,
            "catalog_static_na_cell_count": self.catalog_static_na_cell_count,
            "rendered_catalog_static_na_cell_count": (
                self.rendered_catalog_static_na_cell_count
            ),
            "applicable_na_display_slot_count": len(self.applicable_na_display_slots),
            "applicable_na_display_slots": list(
                self.applicable_na_display_slots[:preview_limit]
            ),
            "missing_rendered_cell_count": len(self.missing_rendered_cell_ids),
            "missing_rendered_cell_ids": list(
                self.missing_rendered_cell_ids[:preview_limit]
            ),
            "contract_errors": list(self.contract_errors[:preview_limit]),
        }


class BaselineCandidateAdapter:
    """Join candidate measurements to their canonical baseline at render time."""

    def __init__(
        self,
        caches: Mapping[str, CachePayload] | Iterable[CachePayload],
        *,
        catalog: ReportCatalog = REPORT_CATALOG,
    ):
        self.catalog = catalog
        self.index = MeasurementIndex(caches)
        self._cells = {
            (
                cell.dataset_id,
                cell.process_key,
                cell.n_final,
                cell.workload,
                cell.variant,
            ): cell
            for cell in catalog.measurement_cells()
        }

    def _cell(
        self,
        dataset_id: str,
        process_key: str,
        n_final: int,
        workload: Workload,
        *,
        variant: str | None = None,
    ) -> CellSpec:
        return self._cells[
            (dataset_id, process_key, n_final, workload, variant)
        ]

    @staticmethod
    def _finite_number(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    @classmethod
    def _exact_number(cls, value: object, expected: object) -> bool:
        if (
            not cls._finite_number(value)
            or not cls._finite_number(expected)
        ):
            return False
        return float(value) == float(expected)

    @staticmethod
    def _selectors_match(
        candidate_cell: CellSpec,
        candidate: Measurement,
        baseline: Measurement,
    ) -> bool:
        candidate_selector = candidate.get("selector_contract")
        baseline_selector = baseline.get("selector_contract")
        if candidate_cell.measurement.accuracy is Accuracy.LC:
            return (
                isinstance(candidate_selector, Mapping)
                and isinstance(baseline_selector, Mapping)
                and candidate_selector == baseline_selector
            )
        return candidate_selector is None and baseline_selector is None

    @staticmethod
    def _component(
        cell: CellSpec,
        measurement: Measurement,
    ) -> Mapping[str, object] | None:
        validation = measurement.get("validation")
        component = (
            validation.get(LC_COMMON_COMPONENT_FIELD)
            if isinstance(validation, Mapping)
            else None
        )
        if not isinstance(component, Mapping):
            return None
        if (
            component.get("abi") != LC_COMMON_COMPONENT_ABI
            or component.get("cell_id") != cell.cell_id
            or not BaselineCandidateAdapter._finite_number(component.get("value"))
        ):
            return None
        return component

    def _direct_agreement_link(
        self,
        candidate_cell: CellSpec,
        candidate: Measurement,
        baseline_cell: CellSpec,
        baseline: Measurement,
    ) -> bool:
        matching_edges = tuple(
            edge
            for edge in incoming_agreement_edges(
                candidate_cell,
                catalog=self.catalog,
            )
            if edge.baseline.cell_id == baseline_cell.cell_id
        )
        if not matching_edges:
            return False
        validation = candidate.get("validation")
        records = (
            validation.get(DIRECT_AGREEMENT_FIELD)
            if isinstance(validation, Mapping)
            else None
        )
        if not isinstance(records, list):
            return False
        for edge in matching_edges:
            record = next(
                (
                    item
                    for item in records
                    if isinstance(item, Mapping)
                    and item.get("abi") == DIRECT_AGREEMENT_ABI
                    and item.get("edge_kind") == edge.kind
                    and item.get("value_kind") == edge.value_kind
                    and item.get("candidate_cell_id") == candidate_cell.cell_id
                    and item.get("baseline_cell_id") == baseline_cell.cell_id
                    and item.get("status") == ResultStatus.OK.value
                ),
                None,
            )
            if record is None:
                continue
            if edge.value_kind == "matrix_element":
                candidate_value = candidate.get("matrix_element")
                baseline_value = baseline.get("matrix_element")
            else:
                candidate_component = self._component(candidate_cell, candidate)
                baseline_component = self._component(baseline_cell, baseline)
                if candidate_component is None or baseline_component is None:
                    continue
                identity_fields = (
                    "point_digest",
                    "helicity_ids",
                    "color_flow_ids",
                )
                if any(
                    candidate_component.get(field)
                    != baseline_component.get(field)
                    for field in identity_fields
                ):
                    continue
                candidate_value = candidate_component.get("value")
                baseline_value = baseline_component.get("value")
            if self._exact_number(
                record.get("candidate"),
                candidate_value,
            ) and self._exact_number(
                record.get("baseline"),
                baseline_value,
            ):
                return True
        return False

    def _direct_link(
        self,
        candidate_cell: CellSpec,
        candidate: Measurement,
        baseline_cell: CellSpec,
        baseline: Measurement,
    ) -> bool:
        if not (_ok(candidate) and _ok(baseline)) or not self._selectors_match(
            candidate_cell,
            candidate,
            baseline,
        ):
            return False
        matching_edges = tuple(
            edge
            for edge in incoming_agreement_edges(
                candidate_cell,
                catalog=self.catalog,
            )
            if edge.baseline.cell_id == baseline_cell.cell_id
        )
        if matching_edges:
            return self._direct_agreement_link(
                candidate_cell,
                candidate,
                baseline_cell,
                baseline,
            )
        canonical_baseline = self.catalog.validation_baseline_cell(candidate_cell)
        if (
            canonical_baseline is None
            or canonical_baseline.cell_id != baseline_cell.cell_id
        ):
            return False
        validation = candidate.get("validation")
        pointwise = (
            validation.get("pointwise")
            if isinstance(validation, Mapping)
            else None
        )
        return (
            isinstance(pointwise, Mapping)
            and pointwise.get("status") == ResultStatus.OK.value
            and self._exact_number(
                pointwise.get("candidate"),
                candidate.get("matrix_element"),
            )
            and self._exact_number(
                pointwise.get("baseline"),
                baseline.get("matrix_element"),
            )
        )

    def _recurrence_bridge(self, candidate_cell: CellSpec) -> CellSpec | None:
        baseline = self.catalog.validation_baseline_cell(candidate_cell)
        if (
            baseline is not None
            and baseline.measurement.execution_mode is ExecutionMode.RECURRENCE
        ):
            return baseline
        peers = tuple(
            edge.baseline
            for edge in incoming_agreement_edges(
                candidate_cell,
                catalog=self.catalog,
            )
            if edge.baseline.measurement.execution_mode is ExecutionMode.RECURRENCE
        )
        return peers[0] if len(peers) == 1 else None

    def _comparison_linked(
        self,
        candidate_cell: CellSpec,
        candidate: Measurement,
        baseline_cell: CellSpec,
        baseline: Measurement,
    ) -> bool:
        if (
            baseline_cell.measurement.execution_mode is ExecutionMode.AMPLICOL
            and candidate_cell.measurement.execution_mode
            in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
        ):
            recurrence_cell = self._recurrence_bridge(candidate_cell)
            if recurrence_cell is None:
                return False
            recurrence = self.index.for_cell(recurrence_cell)
            return self._direct_link(
                candidate_cell,
                candidate,
                recurrence_cell,
                recurrence,
            ) and self._direct_link(
                recurrence_cell,
                recurrence,
                baseline_cell,
                baseline,
            )
        return self._direct_link(
            candidate_cell,
            candidate,
            baseline_cell,
            baseline,
        )

    def matrix_cell(
        self,
        dataset: MatrixDataset,
        family: ProcessFamily,
        n_final: int,
    ) -> JoinedMatrixCell:
        process = family.process(n_final)
        applicable = (
            process is not None
            and n_final in dataset.multiplicities
            and n_final <= family.maximum_n(dataset.candidate.accuracy)
        )
        workloads = (
            (Workload.SELECTED_FLOW, Workload.ALL_FLOW)
            if dataset.candidate.accuracy is Accuracy.LC
            else (Workload.CONTRACTED,)
        )
        if not applicable:
            joined = tuple(JoinedWorkload(workload, _NA, _NA) for workload in workloads)
        else:
            joined_items: list[JoinedWorkload] = []
            for workload in workloads:
                candidate_cell = self._cell(
                    dataset.dataset_id,
                    family.key,
                    n_final,
                    workload,
                )
                baseline_cell = self.catalog.baseline_cell(candidate_cell)
                assert baseline_cell is not None
                candidate = self.index.for_cell(candidate_cell)
                baseline = self.index.for_cell(baseline_cell)
                joined_items.append(
                    JoinedWorkload(
                        workload,
                        baseline,
                        candidate,
                        self._comparison_linked(
                            candidate_cell,
                            candidate,
                            baseline_cell,
                            baseline,
                        ),
                    )
                )
            joined = tuple(joined_items)
        return JoinedMatrixCell(dataset, family, n_final, applicable, joined)

    def z_workload(
        self,
        *,
        model: ModelKey,
        n_final: int,
        variant: ZVariant,
        workload: Workload,
    ) -> JoinedWorkload:
        baseline = self.index.get(
            "reference_amplicol_lc",
            "dd_z_jets",
            n_final,
            workload,
        )
        if variant.execution_mode is ExecutionMode.AMPLICOL:
            return JoinedWorkload(workload, baseline, baseline)
        candidate_cell = self._cell(
            z_dataset_id(model),
            "dd_z_jets",
            n_final,
            workload,
            variant=variant.key,
        )
        baseline_cell = self.catalog.baseline_cell(candidate_cell)
        assert baseline_cell is not None
        candidate = self.index.for_cell(candidate_cell)
        return JoinedWorkload(
            workload,
            baseline,
            candidate,
            self._comparison_linked(
                candidate_cell,
                candidate,
                baseline_cell,
                baseline,
            ),
        )

    def best_mode_cell(
        self,
        accuracy: Accuracy,
        family: ProcessFamily,
        n_final: int,
    ) -> BestModeMatrixCell:
        """Join AmpliCol to the fastest valid built-in-SM mode per workload."""

        process = family.process(n_final)
        multiplicities = matrix_multiplicities(accuracy)
        applicable = (
            process is not None
            and n_final in multiplicities
            and n_final <= family.maximum_n(accuracy)
        )
        workloads = (
            (Workload.SELECTED_FLOW, Workload.ALL_FLOW)
            if accuracy is Accuracy.LC
            else (Workload.CONTRACTED,)
        )
        if not applicable:
            joined = tuple(
                BestModeWorkload(workload, _NA, _NA, None, None)
                for workload in workloads
            )
            return BestModeMatrixCell(
                family,
                n_final,
                accuracy,
                applicable,
                joined,
            )

        baseline_dataset = f"reference_amplicol_{accuracy.value}"
        joined_workloads: list[BestModeWorkload] = []
        for workload in workloads:
            baseline_cell = self._cell(
                baseline_dataset,
                family.key,
                n_final,
                workload,
            )
            baseline = self.index.for_cell(baseline_cell)
            candidates = tuple(
                (
                    mode,
                    self._cell(
                        f"matrix_{mode.value}_builtin_sm_{accuracy.value}",
                        family.key,
                        n_final,
                        workload,
                    ),
                )
                for mode in _BEST_MODE_ORDER
            )
            measured_candidates = tuple(
                (mode, cell, self.index.for_cell(cell))
                for mode, cell in candidates
            )
            eligible = tuple(
                (mode, cell, measurement)
                for mode, cell, measurement in measured_candidates
                if _runtime_value(measurement) is not None
            )
            if eligible:
                winner_mode, winner_cell, winner = min(
                    eligible,
                    key=lambda item: (
                        _runtime_value(item[2]),
                        _BEST_MODE_ORDER.index(item[0]),
                    ),
                )
                terminal_label = None
                comparison_linked = self._comparison_linked(
                    winner_cell,
                    winner,
                    baseline_cell,
                    baseline,
                )
            else:
                winner_mode, winner = None, _NA
                terminal_label = _best_mode_terminal_label(
                    tuple(
                        measurement
                        for _mode, _cell, measurement in measured_candidates
                    )
                )
                comparison_linked = False
            joined_workloads.append(
                BestModeWorkload(
                    workload,
                    baseline,
                    winner,
                    winner_mode,
                    terminal_label,
                    comparison_linked,
                )
            )
        return BestModeMatrixCell(
            family,
            n_final,
            accuracy,
            applicable,
            tuple(joined_workloads),
        )


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _compact(value: float) -> str:
    return f"{value:.3g}"


def _timing_value(value: float) -> str:
    return f"{value:.6g}"


def _best_mode_value(value: float) -> str:
    """Render exactly three significant digits in compact overview tables."""

    rendered = f"{value:#.3g}"
    if "e" not in rendered and "E" not in rendered and rendered.endswith("."):
        return rendered[:-1]
    return rendered


def _best_mode_time(value: object, *, microseconds: bool = False) -> str:
    if value is None:
        return r"\matrixna{ReportMuted}"
    number = float(value)
    if not math.isfinite(number):
        return r"\matrixna{ReportMuted}"
    if microseconds:
        number *= 1.0e6
    return rf"\texttt{{{_best_mode_value(number)}}}"


def _best_mode_ratio_color(value: float) -> str:
    if value < 1.0:
        return "ReportGreen"
    if value < 2.0:
        return "ReportOrange"
    return "ReportRed"


def _metric_group_span(accuracy: Accuracy) -> int:
    return 8 if accuracy is Accuracy.LC else 3


def _metric_group_column_spec(accuracy: Accuracy) -> str:
    """Physical value columns shared by detailed and best-mode tables.

    Ordinary inter-column padding is intentional: ``colortbl`` paints it as
    part of a coloured row.  Non-zero ``@{...}`` inserts leave unpainted seams
    through alternating row bands, so ``@{}`` is used only where two fragments
    form one visually continuous value.
    """

    font = r"\matrixentryfontlc" if accuracy is Accuracy.LC else r"\matrixentryfont"
    if accuracy is Accuracy.LC:
        return (
            rf">{{{font}}}l"
            rf">{{{font}}}c"
            rf">{{{font}}}l"
            rf">{{{font}}}r"
            r"@{}"
            rf">{{{font}}}l"
            rf">{{{font}}}c"
            rf">{{{font}}}r"
            r"@{}"
            rf">{{{font}}}l"
        )
    return (
        rf">{{{font}}}l"
        rf">{{{font}}}r"
        r"@{}"
        rf">{{{font}}}l"
    )


def _metric_table_column_spec(
    accuracy: Accuracy,
    multiplicity_count: int,
) -> str:
    multiplicity_groups = "".join(
        _metric_group_column_spec(accuracy) for _ in range(multiplicity_count)
    )
    return r"@{}rL{1.45in}l" + multiplicity_groups + r"@{}"


def _metric_group_cell(
    content: str,
    accuracy: Accuracy,
    *,
    alignment: str = "c",
) -> str:
    if alignment not in {"c", "l", "r"}:
        raise ValueError(f"unsupported metric-group alignment: {alignment}")
    return (
        rf"\multicolumn{{{_metric_group_span(accuracy)}}}"
        rf"{{{alignment}}}{{{content}}}"
    )


def _metric_summary_group_cell(
    content: str,
    accuracy: Accuracy,
    *,
    group_index: int,
) -> str:
    """Render one centred summary group with block-local band shading."""

    if group_index % 2 == 0:
        content = rf"\cellcolor{{refblue}}{content}"
    return _metric_group_cell(content, accuracy, alignment="c")


def _metric_summary_header_row(
    multiplicities: Sequence[int],
    accuracy: Accuracy,
    *,
    annotations: Sequence[str] | None = None,
) -> str:
    """Repeat the multiplicity headings immediately above summary rows."""

    if annotations is None:
        annotations = ("",) * len(multiplicities)
    if len(annotations) != len(multiplicities):
        raise ValueError("summary header annotations must match multiplicities")

    return (
        r"\multicolumn{3}{@{}l}{} & "
        + " & ".join(
            _metric_summary_group_cell(
                (
                    rf"\textbf{{n={n_final}}}"
                    + (
                        rf"\hspace{{0.08in}}{annotation}"
                        if annotation
                        else ""
                    )
                ),
                accuracy,
                group_index=group_index,
            )
            for group_index, (n_final, annotation) in enumerate(
                zip(multiplicities, annotations, strict=True)
            )
        )
        + r" \\"
    )


def _matrix_wall_summary_label(accuracy: Accuracy) -> str:
    if accuracy is Accuracy.LC:
        return (
            r"\matrixsummaryworkloads"
            r"{\textbf{summary: run single-flow, hel. sum}}"
            r"{\textbf{summary: run all-flows, single hel.}}"
        )
    return r"\textbf{summary: wall}"


def _metric_row_label(*, runtime: bool) -> str:
    label = r"run [\(\mu\mathrm{s}/\mathrm{pt}\)]" if runtime else r"gen. [s]"
    return rf"\textcolor{{ReportMuted}}{{\scriptsize {label}}}"


def _time(value: object, *, microseconds: bool = False) -> str:
    if value is None:
        return r"\matrixna{ReportMuted}"
    number = float(value)
    if not math.isfinite(number):
        return r"\matrixna{ReportMuted}"
    if microseconds:
        number *= 1.0e6
    return rf"\texttt{{{_timing_value(number)}}}"


def _unavailable_time(
    measurement: Measurement,
    field: str,
    *,
    microseconds: bool = False,
) -> str | None:
    record = unavailable_execution_timing_record(measurement, field)
    if record is None:
        return None
    total_record = evaluator_total_timing_record(measurement)
    if total_record is not None:
        return (
            r"\matrixtotalevaluator{"
            + _time(
                total_record["raw_seconds_per_point"],
                microseconds=microseconds,
            )
            + "}"
        )
    return _not_exposed()


def _presentation_outcome(
    measurement: Measurement,
) -> tuple[str, str] | None:
    failure = measurement.get("failure")
    if not isinstance(failure, Mapping):
        return None
    kind = failure.get("kind")
    message = failure.get("message")
    if (
        not isinstance(kind, str)
        or not kind.startswith(_PRESENTATION_FAILURE_KIND_PREFIX)
        or not isinstance(message, str)
        or not message
    ):
        return None
    slug = kind.removeprefix(_PRESENTATION_FAILURE_KIND_PREFIX)
    if _SAFE_OUTCOME_SLUG.fullmatch(slug) is None:
        return None
    return slug, message


def _visible_status(measurement: Measurement) -> _TerminalOutcome:
    """Return a visible label whose classification does not depend on its text."""

    status = str(measurement.get("status", ResultStatus.NOT_AVAILABLE.value))
    policy_label = policy_status_label(measurement)
    if policy_label is not None:
        policy_state = policy_measurement_state_hint(measurement)
        display_label = (
            "dependency"
            if policy_state
            in {
                PolicyMeasurementState.DEPENDENCY,
                PolicyMeasurementState.RESOURCE_FRONTIER,
            }
            else policy_label
        )
        return _TerminalOutcome(
            identity=(
                f"policy:{policy_state.value}"
                if policy_state is not None
                else f"policy-label:{policy_label}"
            ),
            label=display_label,
            color="ReportOrange",
        )
    presentation = _presentation_outcome(measurement)
    if presentation is not None:
        slug, label = presentation
        return _TerminalOutcome(
            identity=f"presentation:{slug}",
            label=label,
            color=(
                "ReportOrange"
                if slug in _POLICY_PRESENTATION_OUTCOME_SLUGS
                else "ReportRed"
            ),
        )
    labels = {
        ResultStatus.NOT_AVAILABLE.value: "N/A",
        ResultStatus.TIMEOUT.value: "t/o",
        ResultStatus.MEMORY_LIMIT.value: "RAM",
        ResultStatus.SKIP.value: "skip",
        ResultStatus.VALIDATION_FAILED.value: "validation failed",
        ResultStatus.UNSUPPORTED.value: "unsupported",
        ResultStatus.FAILED.value: "failed",
        ResultStatus.ERROR.value: "error",
    }
    label = labels.get(status, status)
    color = "ReportMuted" if status == ResultStatus.NOT_AVAILABLE.value else "ReportRed"
    return _TerminalOutcome(
        identity=f"result:{status}",
        label=label,
        color=color,
    )


def _visible_status_label(measurement: Measurement) -> tuple[str, str]:
    visible = _visible_status(measurement)
    return _compact_terminal_display(visible), visible.color


def _compact_terminal_display(outcome: _TerminalOutcome) -> str:
    """Bound hostile/future labels while retaining one stable identity token."""

    identity_kind, _, identity_value = outcome.identity.partition(":")
    known_presentation = (
        identity_kind == "presentation"
        and _KNOWN_PRESENTATION_TERMINAL_LABELS.get(identity_value) == outcome.label
    )
    known_result = (
        identity_kind == "result"
        and outcome.label in _KNOWN_RESULT_TERMINAL_LABELS
    )
    known_policy = (
        (
            identity_kind in {"policy", "policy-label"}
            or (
                identity_kind == "presentation"
                and identity_value in _POLICY_PRESENTATION_OUTCOME_SLUGS
            )
        )
        and _POLICY_TERMINAL_DISPLAY_LABEL.fullmatch(outcome.label) is not None
        and len(outcome.label) <= 18
    )
    compact_count = (
        _COMPACT_OUTCOME_COUNT_LABEL.fullmatch(outcome.label)
        if identity_kind == "summary"
        else None
    )
    if (
        known_presentation
        or known_result
        or known_policy
    ):
        return outcome.label
    if compact_count is not None:
        return f"N out.[{compact_count.group('digest')}]"
    if identity_kind in {"policy", "presentation"} and _SAFE_OUTCOME_SLUG.fullmatch(
        identity_value
    ):
        slug_words = tuple(
            word for word in re.split(r"[_-]+", identity_value) if word
        )
        if len(slug_words) == 1 and len(slug_words[0]) <= 12:
            return slug_words[0]
        if 2 <= len(slug_words) <= 3:
            return " ".join(word[:4] for word in slug_words)
    digest = hashlib.sha256(
        f"{outcome.identity}\0{outcome.label}\0{outcome.color}".encode()
    ).hexdigest()[:_TERMINAL_LABEL_DIGEST_CHARS]
    if " | " in outcome.label:
        return f"N out.[{digest}]"
    prefix = "".join(
        character
        for character in outcome.label
        if character.isascii() and character.isalnum()
    )[:4]
    return f"{prefix or 'term'}..{digest}"


def _status(measurement: Measurement) -> str:
    label, color = _visible_status_label(measurement)
    return rf"\matrixstatus{{{color}}}{{{_tex_escape(label)}}}"


def _ok(measurement: Measurement) -> bool:
    return measurement.get("status") == ResultStatus.OK.value


def _runtime_value(measurement: Measurement) -> float | None:
    """Return a comparable wall time for best-mode selection."""

    if not _ok(measurement):
        return None
    value = measurement.get("wall_seconds_per_point")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _metric(measurement: Measurement, field: str, *, microseconds: bool = False) -> str:
    if not _ok(measurement):
        return _status(measurement)
    unavailable = _unavailable_time(
        measurement,
        field,
        microseconds=microseconds,
    )
    if unavailable is not None:
        return unavailable
    return _time(measurement.get(field), microseconds=microseconds)


def _best_mode_metric(
    measurement: Measurement,
    field: str,
    *,
    microseconds: bool = False,
) -> str:
    if not _ok(measurement):
        return _status(measurement)
    unavailable = _unavailable_time(
        measurement,
        field,
        microseconds=microseconds,
    )
    if unavailable is not None:
        return unavailable
    return _best_mode_time(measurement.get(field), microseconds=microseconds)


def _ratio_value(
    candidate: Measurement,
    baseline: Measurement,
    field: str,
) -> float | None:
    if not (_ok(candidate) and _ok(baseline)):
        return None
    if field == "execution_seconds_per_point":
        for measurement in (candidate, baseline):
            provenance = measurement.get("provenance")
            if not isinstance(provenance, Mapping):
                continue
            timing = provenance.get("execution_timing")
            if isinstance(timing, Mapping) and timing.get("ratio_eligible") is not True:
                return None
    if (
        unavailable_execution_timing_record(candidate, field) is not None
        or unavailable_execution_timing_record(baseline, field) is not None
    ):
        return None
    numerator = candidate.get(field)
    denominator = baseline.get(field)
    if numerator is None or denominator is None:
        return None
    numerator_number = float(numerator)
    denominator_number = float(denominator)
    if (
        not math.isfinite(numerator_number)
        or not math.isfinite(denominator_number)
        or denominator_number <= 0.0
    ):
        return None
    return numerator_number / denominator_number


def _positive_timing_value(
    measurement: Measurement,
    field: str,
) -> float | None:
    """Return one finite positive timing without changing its clock identity."""

    if not _ok(measurement):
        return None
    value = measurement.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _secondary_runtime_ratio(
    candidate: Measurement,
    baseline: Measurement,
    *,
    baseline_mode: ExecutionMode,
) -> float | None:
    """Return the documented supplementary runtime ratio, if available.

    The clock hierarchy is deliberately strict: compare exposed execution
    attribution first, then authenticated evaluator totals.  Only the legacy
    AmpliCol reference may supply its direct execution (or wall) clock when it
    has no evaluator-total record; a recurrence core is never used as the
    denominator for another backend's evaluator total.
    """

    execution_ratio = _ratio_value(
        candidate,
        baseline,
        "execution_seconds_per_point",
    )
    if execution_ratio is not None:
        return execution_ratio

    candidate_total = evaluator_total_seconds_per_point(candidate)
    if candidate_total is None:
        return None
    baseline_total = evaluator_total_seconds_per_point(baseline)
    if baseline_total is not None:
        return candidate_total / baseline_total
    if baseline_mode is not ExecutionMode.AMPLICOL:
        return None

    legacy_direct = _positive_timing_value(
        baseline,
        "execution_seconds_per_point",
    )
    if legacy_direct is None:
        legacy_direct = _positive_timing_value(
            baseline,
            "wall_seconds_per_point",
        )
    if legacy_direct is None:
        return None
    return candidate_total / legacy_direct


def _ratio(candidate: Measurement, baseline: Measurement, field: str) -> str:
    value = _ratio_value(candidate, baseline, field)
    if value is None:
        if not _ok(candidate):
            return _status(candidate)
        if (
            unavailable_execution_timing_record(candidate, field) is not None
            or unavailable_execution_timing_record(baseline, field) is not None
        ):
            return _not_exposed()
        return r"\matrixnaratio{ReportMuted}"
    color = (
        "ReportGreen" if value < 1.0 else "ReportOrange" if value < 2.0 else "ReportRed"
    )
    return rf"\matrixratio{{{color}}}{{{_compact(value)}}}"


def _ratio_or_absolute(
    candidate: Measurement,
    baseline: Measurement,
    field: str,
    *,
    absolute: bool,
    microseconds: bool = False,
) -> str:
    if absolute and _ok(candidate):
        return (
            r"\matrixncabsolute{"
            + _metric(candidate, field, microseconds=microseconds)
            + "}"
        )
    return _ratio(candidate, baseline, field)


def _evaluator_total_clock(measurement: Measurement) -> str:
    total = evaluator_total_seconds_per_point(measurement)
    value = _not_exposed() if total is None else _time(total, microseconds=True)
    return r"\matrixtotalevaluator{" + value + "}"


def _recurrence_core_clock(measurement: Measurement) -> str:
    core = recurrence_core_seconds_per_point(measurement)
    value = _not_exposed() if core is None else _time(core, microseconds=True)
    return r"\matrixrecurrencecore{" + value + "}"


def _ratio_pair(
    candidate: Measurement,
    baseline: Measurement,
    *,
    candidate_mode: ExecutionMode,
) -> str:
    wall = _ratio_value(candidate, baseline, "wall_seconds_per_point")
    if not _ok(candidate):
        return _status(candidate)
    if wall is None:
        wall_clock = r"\matrixna{ReportMuted}"
    else:
        color = (
            "ReportGreen"
            if wall < 1.0
            else "ReportOrange"
            if wall < 2.0
            else "ReportRed"
        )
        wall_clock = rf"\matrixwallclock{{{color}}}{{x{_compact(wall)}}}"
    total_clock = _evaluator_total_clock(candidate)
    if candidate_mode is not ExecutionMode.RECURRENCE:
        return rf"\matrixruntimepair{{{wall_clock}}}{{{total_clock}}}"
    return (
        rf"\matrixruntimetriple{{{wall_clock}}}{{{total_clock}}}"
        rf"{{{_recurrence_core_clock(candidate)}}}"
    )


def _ratio_pair_or_absolute(
    candidate: Measurement,
    baseline: Measurement,
    *,
    candidate_mode: ExecutionMode,
    absolute: bool,
) -> str:
    if absolute and _ok(candidate):
        wall = _metric(candidate, "wall_seconds_per_point", microseconds=True)
        wall_clock = r"\matrixncabsolute{\matrixwallabsolute{" + wall + "}}"
        total_clock = _evaluator_total_clock(candidate)
        if candidate_mode is not ExecutionMode.RECURRENCE:
            return rf"\matrixruntimepair{{{wall_clock}}}{{{total_clock}}}"
        return (
            rf"\matrixruntimetriple{{{wall_clock}}}{{{total_clock}}}"
            rf"{{{_recurrence_core_clock(candidate)}}}"
        )
    return _ratio_pair(
        candidate,
        baseline,
        candidate_mode=candidate_mode,
    )


def _matrix_macros() -> list[str]:
    return [
        r"\providecommand{\matrixentryfont}{\fontsize{6.8pt}{7.8pt}\selectfont}",
        r"\providecommand{\matrixentryfontlc}{\fontsize{6.5pt}{7.5pt}\selectfont}",
        r"\providecommand{\matrixsummaryfont}{\fontsize{6.2pt}{7.4pt}\selectfont}",
        r"\providecommand{\matrixpunct}[1]{\textcolor{black}{\texttt{#1}}}",
        (
            r"\providecommand{\matrixratio}[2]{\matrixpunct{(}"
            r"\textcolor{#1}{\texttt{x#2}}\matrixpunct{)}}"
        ),
        r"\providecommand{\matrixna}[1]{\textcolor{#1}{\texttt{N/A}}}",
        (
            r"\providecommand{\matrixnotapplicable}[1]{"
            r"\textcolor{#1}{\textsc{not applicable}}}"
        ),
        (
            r"\providecommand{\matrixnotexposed}[1]{"
            r"\textcolor{#1}{\textsc{not exposed}}}"
        ),
        (
            r"\providecommand{\matrixstaticna}[1]{"
            r"\textcolor{#1}{\textsc{static N/A}}}"
        ),
        (
            r"\providecommand{\matrixtotalevaluator}[1]{"
            r"\textcolor{ReportBlue}{\texttt{T }}#1}"
        ),
        (
            r"\providecommand{\matrixrecurrencecore}[1]{"
            r"\textcolor{ReportOrange}{\texttt{C }}#1}"
        ),
        (
            r"\providecommand{\matrixzrecurrenceclocks}[2]{"
            r"\shortstack[r]{#1\\#2}}"
        ),
        (
            r"\providecommand{\matrixwallclock}[2]{"
            r"\textcolor{#1}{\texttt{W #2}}}"
        ),
        (
            r"\providecommand{\matrixwallabsolute}[1]{"
            r"\textcolor{ReportBlue}{\texttt{W }}#1}"
        ),
        (
            r"\providecommand{\matrixruntimepair}[2]{"
            r"\shortstack[l]{#1\\#2}}"
        ),
        (
            r"\providecommand{\matrixruntimetriple}[3]{"
            r"\shortstack[l]{#1\\#2\matrixpunct{ | }#3}}"
        ),
        r"\providecommand{\matrixbelow}[1]{\textcolor{ReportMuted}{\texttt{<#1}}}",
        (
            r"\providecommand{\matrixnaratio}[1]{\matrixpunct{(}"
            r"\matrixna{#1}\matrixpunct{)}}"
        ),
        r"\providecommand{\matrixstatus}[2]{\textcolor{#1}{\textsc{#2}}}",
        (
            r"\providecommand{\matrixncabsolute}[1]{"
            r"#1}"
        ),
        (
            r"\providecommand{\matrixsummaryratio}[2]{"
            r"\textcolor{#1}{\texttt{x#2}}}"
        ),
        (
            r"\providecommand{\matrixsummaryratiohighlight}[2]{"
            r"\begingroup\setlength{\fboxsep}{0.9pt}"
            r"\setlength{\fboxrule}{0.35pt}"
            r"\fbox{\textcolor{#1}{{\usefont{T1}{lmtt}{b}{n}x#2}}}"
            r"\endgroup}"
        ),
        # Keep the five statistic anchors identical on every summary line.
        # Max accommodates scientific notation; average accommodates its frame.
        (
            r"\providecommand{\matrixsummarystats}[5]{"
            r"\begingroup\matrixsummaryfont"
            r"\begin{tabular}[t]{@{}l"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}l"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}l"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}l"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}l@{}}"
            r"\makebox[3.6em][l]{#1}&"
            r"\makebox[4.6em][l]{#2}&"
            r"\makebox[3.6em][l]{#3}&"
            r"\makebox[4.2em][l]{#4}&"
            r"\makebox[3.6em][l]{#5}"
            r"\end{tabular}\endgroup}"
        ),
        (
            r"\providecommand{\matrixsummaryworkloads}[2]{"
            r"\begin{tabular}[t]{@{}l@{}}#1\\[-0.10em]#2\end{tabular}}"
        ),
    ]


def _compact_comparison_macros() -> list[str]:
    """Return the compact aligned comparison vocabulary shared by matrices."""

    return [
        (
            r"\providecommand{\bestmoderatio}[2]{"
            r"\textcolor{#1}{\texttt{x#2}}}"
        ),
        (
            r"\providecommand{\bestmodecompactratio}[3]{"
            r"\matrixpunct{(}"
            r"\textcolor{ReportMuted}{\texttt{[x#1]}}"
            r"\hspace{0.04in}"
            r"\textcolor{#2}{\texttt{x#3}}"
            r"\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\bestmodewallratio}[2]{"
            r"\matrixpunct{(}\textcolor{#1}{\texttt{x#2}}\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\bestmodecompactprefix}[1]{"
            r"\matrixpunct{(}\textcolor{ReportMuted}{\texttt{[x#1]}}"
            r"\hspace{0.04in}}"
        ),
        (
            r"\providecommand{\bestmodeopenprefix}{"
            r"\matrixpunct{(}\hspace{0.04in}}"
        ),
        (
            r"\providecommand{\bestmodeprimaryratio}[2]{"
            r"\textcolor{#1}{\texttt{x#2}}\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\bestmodeabsoluteprefix}[1]{"
            r"#1\hspace{0.04in}}"
        ),
    ]


def _not_applicable() -> str:
    return r"\matrixnotapplicable{ReportMuted}"


def _not_exposed() -> str:
    return r"\matrixnotexposed{ReportMuted}"


def _static_na() -> str:
    return r"\matrixstaticna{ReportMuted}"


def _legacy_baseline_static_na(
    *,
    catalog: ReportCatalog,
    accuracy: Accuracy,
    process_key: str,
    n_final: int,
    workload: Workload,
) -> bool:
    baseline = next(
        cell
        for cell in catalog.measurement_cells()
        if cell.dataset_id == f"reference_amplicol_{accuracy.value}"
        and cell.process_key == process_key
        and cell.n_final == n_final
        and cell.workload is workload
    )
    return catalog.static_na_reason(baseline) is not None


def _legacy_baseline_unavailable(
    view: JoinedMatrixCell,
    *,
    catalog: ReportCatalog,
) -> bool:
    return (
        view.dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
        and _legacy_baseline_static_na(
            catalog=catalog,
            accuracy=view.dataset.baseline.accuracy,
            process_key=view.process_family.key,
            n_final=view.n_final,
            workload=view.workloads[0].workload,
        )
    )


def _fixed_mode_generation_comparison_layout(
    joined: JoinedWorkload,
    *,
    baseline_mode: ExecutionMode,
    baseline_static_na: bool = False,
    comparable: bool = True,
) -> _BestModeComparisonLayout:
    """Render one fixed-engine generation comparison without a mode letter."""

    if not _ok(joined.candidate):
        status = _status(joined.candidate)
        return _BestModeComparisonLayout(status, "", status)
    if (
        baseline_static_na
        or not comparable
        or not joined.comparison_linked
        or (
            baseline_mode is ExecutionMode.AMPLICOL
            and not _ok(joined.baseline)
        )
    ):
        absolute = _best_mode_metric(joined.candidate, "generation_seconds")
        return _BestModeComparisonLayout(
            absolute,
            rf"\bestmodeabsoluteprefix{{{absolute}}}",
            "",
        )
    if not _ok(joined.baseline):
        status = _status(joined.baseline)
        return _BestModeComparisonLayout(status, "", status)
    generation_ratio = _ratio_value(
        joined.candidate,
        joined.baseline,
        "generation_seconds",
    )
    if generation_ratio is None:
        unavailable = r"\matrixna{ReportMuted}"
        return _BestModeComparisonLayout(unavailable, "", unavailable)
    ratio = (
        rf"\bestmoderatio{{{_best_mode_ratio_color(generation_ratio)}}}"
        rf"{{{_best_mode_value(generation_ratio)}}}"
    )
    return _BestModeComparisonLayout(ratio, "", ratio)


def _fixed_mode_runtime_comparison_layout(
    joined: JoinedWorkload,
    *,
    candidate_mode: ExecutionMode,
    baseline_mode: ExecutionMode,
    baseline_static_na: bool = False,
) -> _BestModeComparisonLayout:
    """Use the best-table compact clock layout for one fixed engine."""

    return _best_mode_runtime_comparison_layout(
        BestModeWorkload(
            workload=joined.workload,
            baseline=joined.baseline,
            candidate=joined.candidate,
            mode=candidate_mode,
            terminal_label=None,
            comparison_linked=joined.comparison_linked,
        ),
        baseline_mode=baseline_mode,
        baseline_static_na=baseline_static_na,
    )


def _lc_cell(
    view: JoinedMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> _MatrixCellRows:
    selected, all_flow = view.workloads
    legacy_baseline_unavailable = _legacy_baseline_unavailable(
        view,
        catalog=catalog,
    )
    selected_baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(selected.baseline, "generation_seconds")
    )
    all_flow_baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(all_flow.baseline, "generation_seconds")
    )
    selected_generation = _fixed_mode_generation_comparison_layout(
        selected,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
    )
    all_flow_generation = _fixed_mode_generation_comparison_layout(
        all_flow,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
        comparable=not (
            view.dataset.candidate.execution_mode is ExecutionMode.RECURRENCE
            and view.dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
        ),
    )
    selected_baseline_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(
            selected.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    all_flow_baseline_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(
            all_flow.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    selected_runtime = _fixed_mode_runtime_comparison_layout(
        selected,
        candidate_mode=view.dataset.candidate.execution_mode,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
    )
    all_flow_runtime = _fixed_mode_runtime_comparison_layout(
        all_flow,
        candidate_mode=view.dataset.candidate.execution_mode,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
    )
    return _MatrixCellRows(
        generation=(
            selected_baseline_generation,
            r"\matrixpunct{|}",
            all_flow_baseline_generation,
            *_best_mode_comparison_columns(selected_generation),
            r"\matrixpunct{|}",
            *_best_mode_comparison_columns(all_flow_generation),
        ),
        runtime=(
            selected_baseline_runtime,
            r"\matrixpunct{|}",
            all_flow_baseline_runtime,
            *_best_mode_comparison_columns(selected_runtime),
            r"\matrixpunct{|}",
            *_best_mode_comparison_columns(all_flow_runtime),
        ),
    )


def _contracted_cell(
    view: JoinedMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> _MatrixCellRows:
    joined = view.workloads[0]
    legacy_baseline_unavailable = _legacy_baseline_unavailable(
        view,
        catalog=catalog,
    )
    candidate_generation = _fixed_mode_generation_comparison_layout(
        joined,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
    )
    candidate_runtime = _fixed_mode_runtime_comparison_layout(
        joined,
        candidate_mode=view.dataset.candidate.execution_mode,
        baseline_mode=view.dataset.baseline.execution_mode,
        baseline_static_na=legacy_baseline_unavailable,
    )
    baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(joined.baseline, "generation_seconds")
    )
    baseline_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _best_mode_metric(
            joined.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    return _MatrixCellRows(
        generation=(
            baseline_generation,
            *_best_mode_comparison_columns(candidate_generation),
        ),
        runtime=(
            baseline_runtime,
            *_best_mode_comparison_columns(candidate_runtime),
        ),
    )


def _summary_pair(
    views: Sequence[JoinedMatrixCell],
    workload: Workload,
    field: str,
) -> str:
    joined = [
        next(item for item in view.workloads if item.workload is workload)
        for view in views
        if view.applicable
    ]
    valid = [
        item
        for item in joined
        if item.comparison_linked
        and _ok(item.baseline)
        and _ok(item.candidate)
        and item.baseline.get(field) is not None
        and item.candidate.get(field) is not None
        and unavailable_execution_timing_record(item.baseline, field) is None
        and unavailable_execution_timing_record(item.candidate, field) is None
        and math.isfinite(float(item.baseline[field]))
        and math.isfinite(float(item.candidate[field]))
        and float(item.baseline[field]) > 0.0
        and float(item.candidate[field]) >= 0.0
    ]
    if not valid:
        if field == "execution_seconds_per_point" and any(
            unavailable_execution_timing_record(measurement, field) is not None
            for item in joined
            for measurement in (item.baseline, item.candidate)
        ):
            return _not_exposed()
        for item in joined:
            if not _ok(item.candidate) and policy_status_label(item.candidate):
                return _status(item.candidate)
            if not _ok(item.baseline) and policy_status_label(item.baseline):
                return _status(item.baseline)
        return r"\matrixna{ReportMuted}"
    return _ratio_statistics_tex(
        tuple(
            (float(item.baseline[field]), float(item.candidate[field]))
            for item in valid
        )
    )


def _ratio_statistics_tex(
    timings: Sequence[tuple[float, float]],
) -> str:
    """Render min/max/median/mean/ratio-of-sums for timing pairs."""

    ratios = tuple(candidate / baseline for baseline, candidate in timings)
    ordered = tuple(sorted(ratios))
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    average = math.fsum(ratios) / len(ratios)
    baseline_sum = math.fsum(baseline for baseline, _candidate in timings)
    candidate_sum = math.fsum(candidate for _baseline, candidate in timings)
    weighted_average = candidate_sum / baseline_sum
    statistics = (
        ordered[0],
        ordered[-1],
        median,
        average,
        weighted_average,
    )
    rendered = tuple(
        (
            rf"\matrixsummaryratiohighlight"
            rf"{{{_best_mode_ratio_color(value)}}}"
            rf"{{{_best_mode_value(value)}}}"
            if index == 3
            else rf"\matrixsummaryratio{{{_best_mode_ratio_color(value)}}}"
            rf"{{{_best_mode_value(value)}}}"
        )
        for index, value in enumerate(statistics)
    )
    return r"\matrixsummarystats{" + "}{".join(rendered) + "}"


def _matrix_block(
    adapter: BaselineCandidateAdapter,
    dataset: MatrixDataset,
    multiplicities: tuple[int, ...],
    *,
    block_index: int,
    block_count: int,
) -> list[str]:
    if len(multiplicities) not in (2, 3):
        raise ValueError("matrix blocks must contain two or three multiplicities")
    accuracy = dataset.candidate.accuracy
    column_spec = _metric_table_column_spec(
        accuracy,
        len(multiplicities),
    )
    lines = [
        r"\clearpage",
        r"\noindent\begin{minipage}{\linewidth}",
        (
            rf"\subsubsection*{{{_tex_escape(dataset.title)}"
            + (
                ""
                if block_count == 1
                else rf" (block {block_index + 1} of {block_count})"
            )
            + "}"
        ),
        r"\begingroup",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\makebox[\linewidth][c]{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            r"\textbf{ID} & \textbf{base process} & \textbf{metric} & "
            + " & ".join(
                _metric_group_cell(
                    rf"\textbf{{n={n_final}}}",
                    accuracy,
                )
                for n_final in multiplicities
            )
            + r" \\"
        ),
        r"\specialrule{0.85pt}{0pt}{0pt}",
    ]
    views_by_n: dict[int, list[JoinedMatrixCell]] = {
        n_final: [] for n_final in multiplicities
    }
    for row_index, family in enumerate(adapter.catalog.process_families):
        generation_row = [
            rf"\texttt{{{family.identifier}}}",
            family.label_tex,
            _metric_row_label(runtime=False),
        ]
        runtime_row = ["", "", _metric_row_label(runtime=True)]
        for n_final in multiplicities:
            view = adapter.matrix_cell(dataset, family, n_final)
            views_by_n[n_final].append(view)
            if not view.applicable:
                generation_row.append(
                    _metric_group_cell(
                        _not_applicable(),
                        accuracy,
                    )
                )
                runtime_row.append(
                    _metric_group_cell(
                        "",
                        accuracy,
                    )
                )
            elif accuracy is Accuracy.LC:
                rendered = _lc_cell(view, catalog=adapter.catalog)
                generation_row.extend(rendered.generation)
                runtime_row.extend(rendered.runtime)
            else:
                rendered = _contracted_cell(view, catalog=adapter.catalog)
                generation_row.extend(rendered.generation)
                runtime_row.extend(rendered.runtime)
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(generation_row) + r" \\")
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(runtime_row) + r" \\")
        lines.append(r"\addlinespace[0.05em]")
    lines.extend(
        [
            r"\specialrule{1.05pt}{0.22em}{0.18em}",
            _metric_summary_header_row(multiplicities, accuracy),
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _metric_summary_group_cell(
                        _matrix_generation_summary(
                            views_by_n[n_final],
                            dataset,
                        ),
                        accuracy,
                        group_index=group_index,
                    )
                    for group_index, n_final in enumerate(multiplicities)
                )
                + r" \\[0.08em]"
            ),
            (
                rf"\multicolumn{{3}}{{@{{}}l}}{{"
                rf"{_matrix_wall_summary_label(accuracy)}}} & "
                + " & ".join(
                    _metric_summary_group_cell(
                        _matrix_wall_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        group_index=group_index,
                    )
                    for group_index, n_final in enumerate(multiplicities)
                )
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\endgroup",
            _matrix_legend(dataset),
            r"\end{minipage}",
        ]
    )
    return lines


def _matrix_generation_summary(
    views: Sequence[JoinedMatrixCell],
    dataset: MatrixDataset,
) -> str:
    if dataset.candidate.accuracy is Accuracy.LC:
        return _summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "generation_seconds",
        )
    return _summary_pair(views, Workload.CONTRACTED, "generation_seconds")


def _matrix_wall_summary(
    views: Sequence[JoinedMatrixCell],
    accuracy: Accuracy,
) -> str:
    if accuracy is Accuracy.LC:
        selected = _summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "wall_seconds_per_point",
        )
        all_flow = _summary_pair(
            views,
            Workload.ALL_FLOW,
            "wall_seconds_per_point",
        )
        return rf"\matrixsummaryworkloads{{{selected}}}{{{all_flow}}}"
    return _summary_pair(
        views,
        Workload.CONTRACTED,
        "wall_seconds_per_point",
    )


def _mode_label(mode: ExecutionMode) -> str:
    return {
        ExecutionMode.AMPLICOL: "original AmpliCol",
        ExecutionMode.RECURRENCE: "recurrence JIT O2",
        ExecutionMode.COMPILED: "compiled JIT O3",
        ExecutionMode.EAGER: "eager-DAG JIT O2",
    }[mode]


def _matrix_legend(dataset: MatrixDataset) -> str:
    baseline = _mode_label(dataset.baseline.execution_mode)
    candidate = _mode_label(dataset.candidate.execution_mode)
    if dataset.candidate.accuracy is Accuracy.LC:
        if (
            dataset.candidate.execution_mode is ExecutionMode.RECURRENCE
            and dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
        ):
            detail = (
                "Each cell shows the selected-flow library-generation and "
                "all-flow direct-setup baseline quantities. The selected-flow "
                "entry carries a generation ratio; the all-flow entry carries "
                "the absolute pyAmpliCol process-generation time. Its setup "
                "boundary differs from the reference, so no generation ratio "
                "is formed."
            )
        else:
            detail = (
                "Each cell shows the topology-replay/all-flow-union baseline "
                "generation times and wall times, followed by layout-matched "
                "candidate/baseline generation and wall-time ratios."
            )
    else:
        detail = (
            "Each cell shows the baseline generation and wall time, followed by "
            "candidate/baseline generation and wall-time ratios."
        )
    clock_detail = (
        " Runtime comparisons use ([xS]xW). The muted supplementary xS first "
        "compares compatible exposed execution attributions; otherwise it "
        "compares authenticated evaluator totals. Only when original AmpliCol "
        "has no evaluator-total clock may a candidate evaluator total use the "
        "legacy direct execution clock (or its wall clock when direct execution "
        "is absent) as denominator. An evaluator total is never divided by a "
        "recurrence core. This evaluator-total fallback is diagnostic only; the "
        "execution/execution ratio remains the compatible supplementary "
        "attribution. The coloured xW is always the candidate/baseline wall "
        "multiplier and remains the primary figure of merit. When no compatible "
        "supplementary clock exists, only (xW) is shown. No clock is fabricated "
        "from another. Every displayed number uses exactly three significant "
        "digits."
    )
    legacy_scope_detail = (
        " Original AmpliCol is static N/A beyond three open quark lines; "
        "unavailable, terminal, or unlinked baselines keep their status while "
        "successful candidate timings remain absolute and leave summaries."
        if dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
        else ""
    )
    summary_detail = (
        " Summary rows contain multipliers only in min, max, median, average, "
        "and weighted-average order; the weighted average is the ratio of "
        "timing sums. The framed bold entry is the arithmetic average of the "
        "per-cell multipliers. LC generation uses selected flow only, while LC "
        "runtime shows separate selected-flow and all-flow wall-only lines."
        if dataset.candidate.accuracy is Accuracy.LC
        else " Summary rows contain multipliers only in min, max, median, average, "
        "and weighted-average order; the weighted average is the ratio of "
        "timing sums, and runtime statistics use wall time only. The framed "
        "bold entry is the arithmetic average of the per-cell multipliers."
    )
    comparison_identity_detail = (
        " Ratios and summaries require exact stored linkage to the current "
        "denominator."
        if dataset.baseline.execution_mode is not ExecutionMode.AMPLICOL
        else ""
    )
    return (
        r"\ReportTableNote{{\scriptsize Baseline: "
        + _tex_escape(baseline)
        + "; candidate: "
        + _tex_escape(candidate)
        + ". "
        + _tex_escape(detail)
        + _tex_escape(clock_detail)
        + _tex_escape(legacy_scope_detail)
        + _tex_escape(comparison_identity_detail)
        + _tex_escape(summary_detail)
        + " "
        + _tex_escape(
            "Not applicable marks a process/multiplicity combination outside "
            "the process-family definition. Fixed-engine tables intentionally "
            "omit mode letters because the selected engine is named in the "
            "table heading."
        )
        + "}}"
    )


def render_matrix_table(
    dataset: MatrixDataset,
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> str:
    """Render one matrix dataset into nonbreaking page-sized TeX blocks."""

    adapter = BaselineCandidateAdapter(caches, catalog=catalog)
    blocks = _chunks(dataset.multiplicities, _MATRIX_BLOCK_SIZE)
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by tools/performance_report/render.py; do not edit.",
        *_matrix_macros(),
        (
            r"\providecommand{\matrixpair}[2]{"
            r"\begin{tabular}[t]{@{}r@{\hspace{0.025in}"
            r"\matrixpunct{|}\hspace{0.025in}}r@{}}"
            r"#1&#2\end{tabular}}"
        ),
        *_compact_comparison_macros(),
        r"\clearpage",
        rf"\subsection{{{_tex_escape(dataset.title)}}}",
    ]
    for block_index, multiplicities in enumerate(blocks):
        block = _matrix_block(
            adapter,
            dataset,
            multiplicities,
            block_index=block_index,
            block_count=len(blocks),
        )
        lines.extend(block[1:] if block_index == 0 else block)
    return "\n".join(lines) + "\n"


def render_all_matrix_tables(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, str]:
    """Render all twelve matrices in canonical catalog order."""

    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        dataset.table_name: render_matrix_table(
            dataset,
            cache_source,
            catalog=catalog,
        )
        for dataset in catalog.matrix_datasets
    }


def _canonical_best_mode_terminal_label(
    terminals: Sequence[_TerminalOutcome | _TerminalSummary | None],
) -> _TerminalSummary | None:
    """Return one deterministic, bounded label for represented terminals."""

    if not terminals:
        return None
    represented: list[_TerminalOutcome] = []
    for terminal in terminals:
        if terminal is None:
            continue
        outcomes = (
            terminal.outcomes
            if isinstance(terminal, _TerminalSummary)
            else (terminal,)
        )
        for outcome in outcomes:
            key = (outcome.identity, outcome.label, outcome.color)
            if all(
                (item.identity, item.label, item.color) != key
                for item in represented
            ):
                represented.append(outcome)

    def order(outcome: _TerminalOutcome) -> tuple[int, float, str, str]:
        time_cap = (
            _TIME_CAP_LABEL.fullmatch(outcome.label)
            if outcome.color == "ReportOrange"
            else None
        )
        if time_cap is not None:
            seconds = float(time_cap.group("value")) * (
                3600.0 if time_cap.group("unit") == "h" else 1.0
            )
            return (0, seconds, outcome.label, outcome.identity)
        ram_cap = (
            _RAM_CAP_LABEL.fullmatch(outcome.label)
            if outcome.color == "ReportOrange"
            else None
        )
        if ram_cap is not None:
            return (1, float(ram_cap.group("value")), outcome.label, outcome.identity)
        if outcome.color == "ReportOrange" and outcome.label == "dependency":
            return (2, 0.0, outcome.label, outcome.identity)
        return (3, 0.0, outcome.label, outcome.identity)

    outcomes = tuple(sorted(represented, key=order))
    if not outcomes:
        return None
    labels = tuple(_compact_terminal_display(outcome) for outcome in outcomes)
    label = " | ".join(labels)
    if len(label) > _MAX_BEST_MODE_TERMINAL_LABEL_CHARS:
        separators = 3 * (len(labels) - 1)
        item_budget = max(
            _TERMINAL_LABEL_DIGEST_CHARS + 3,
            (_MAX_BEST_MODE_TERMINAL_LABEL_CHARS - separators) // len(labels),
        )
        compacted: list[str] = []
        for outcome in outcomes:
            if len(outcome.label) <= item_budget:
                compacted.append(outcome.label)
                continue
            digest = hashlib.sha256(
                f"{outcome.identity}\0{outcome.label}".encode()
            ).hexdigest()[:_TERMINAL_LABEL_DIGEST_CHARS]
            prefix_chars = item_budget - len(digest) - 2
            compacted.append(f"{outcome.label[:prefix_chars]}..{digest}")
        label = " | ".join(compacted)
    if len(label) > _MAX_BEST_MODE_TERMINAL_LABEL_CHARS:
        identity_digest = hashlib.sha256(
            "\0".join(
                f"{outcome.identity}\0{outcome.label}\0{outcome.color}"
                for outcome in outcomes
            ).encode()
        ).hexdigest()[:_TERMINAL_LABEL_DIGEST_CHARS]
        label = f"{len(outcomes)} outcomes [{identity_digest}]"
    color = (
        "ReportOrange"
        if all(outcome.color == "ReportOrange" for outcome in outcomes)
        else "ReportRed"
    )
    return _TerminalSummary(outcomes=outcomes, label=label, color=color)


def _terminal_measurement_label(measurement: Measurement) -> _TerminalOutcome | None:
    status = measurement.get("status")
    if status in {ResultStatus.NOT_AVAILABLE.value, ResultStatus.OK.value}:
        return None
    return _visible_status(measurement)


def _best_mode_terminal_label(
    measurements: Sequence[Measurement],
) -> _TerminalSummary | None:
    """Return one fail-closed summary when no candidate mode succeeded."""

    return _canonical_best_mode_terminal_label(
        tuple(_terminal_measurement_label(measurement) for measurement in measurements)
    )


def _best_mode_terminal_status(joined: BestModeWorkload) -> str:
    if joined.terminal_label is None:
        return _status(joined.candidate)
    display_label = _compact_terminal_summary_display(joined.terminal_label)
    return (
        rf"\matrixstatus{{{joined.terminal_label.color}}}"
        rf"{{{_tex_escape(display_label)}}}"
    )


def _compact_terminal_summary_display(summary: _TerminalSummary) -> str:
    if len(summary.outcomes) == 1:
        return _compact_terminal_display(summary.outcomes[0])
    return _compact_terminal_display(
        _TerminalOutcome(
            identity="summary:"
            + "|".join(outcome.identity for outcome in summary.outcomes),
            label=summary.label,
            color=summary.color,
        )
    )


def _best_mode_generation_comparison_layout(
    joined: BestModeWorkload,
    *,
    baseline_static_na: bool = False,
    comparable: bool = True,
) -> _BestModeComparisonLayout:
    if joined.mode is None:
        status = _best_mode_terminal_status(joined)
        return _BestModeComparisonLayout(status, "", status)
    if not _ok(joined.candidate):
        status = _status(joined.candidate)
        return _BestModeComparisonLayout(status, "", status)
    assert joined.mode is not None
    mode_code = _BEST_MODE_CODES[joined.mode]
    if (
        baseline_static_na
        or not comparable
        or not joined.comparison_linked
        or not _ok(joined.baseline)
    ):
        absolute = _best_mode_metric(joined.candidate, "generation_seconds")
        return _BestModeComparisonLayout(
            rf"\bestmodeabsolutechoice{{{absolute}}}{{{mode_code}}}",
            rf"\bestmodeabsoluteprefix{{{absolute}}}",
            rf"\bestmodecode{{{mode_code}}}",
        )
    if not _ok(joined.baseline):
        status = _status(joined.baseline)
        return _BestModeComparisonLayout(status, "", status)
    generation_ratio = _ratio_value(
        joined.candidate,
        joined.baseline,
        "generation_seconds",
    )
    if generation_ratio is None:
        unavailable = r"\matrixna{ReportMuted}"
        return _BestModeComparisonLayout(unavailable, "", unavailable)
    generation_color = _best_mode_ratio_color(generation_ratio)
    ratio = (
        rf"\bestmoderatio{{{generation_color}}}"
        rf"{{{_best_mode_value(generation_ratio)}}}"
    )
    return _BestModeComparisonLayout(
        rf"\bestmodecodeprefix{{{mode_code}}}{ratio}",
        rf"\bestmodecodeprefix{{{mode_code}}}",
        ratio,
    )


def _best_mode_generation_comparison(
    joined: BestModeWorkload,
    *,
    baseline_static_na: bool = False,
    comparable: bool = True,
) -> str:
    return _best_mode_generation_comparison_layout(
        joined,
        baseline_static_na=baseline_static_na,
        comparable=comparable,
    ).inline


def _best_mode_runtime_comparison_layout(
    joined: BestModeWorkload,
    *,
    baseline_mode: ExecutionMode = ExecutionMode.AMPLICOL,
    baseline_static_na: bool = False,
) -> _BestModeComparisonLayout:
    if joined.mode is None:
        status = _best_mode_terminal_status(joined)
        return _BestModeComparisonLayout(status, "", status)
    if not _ok(joined.candidate):
        status = _status(joined.candidate)
        return _BestModeComparisonLayout(status, "", status)
    if baseline_static_na or not joined.comparison_linked or (
        baseline_mode is ExecutionMode.AMPLICOL and not _ok(joined.baseline)
    ):
        wall = _best_mode_metric(
            joined.candidate,
            "wall_seconds_per_point",
            microseconds=True,
        )
        return _BestModeComparisonLayout(
            wall,
            rf"\bestmodeabsoluteprefix{{{wall}}}",
            "",
        )
    if not _ok(joined.baseline):
        status = _status(joined.baseline)
        return _BestModeComparisonLayout(status, "", status)
    wall_ratio = _ratio_value(
        joined.candidate,
        joined.baseline,
        "wall_seconds_per_point",
    )
    if wall_ratio is None:
        unavailable = r"\matrixnaratio{ReportMuted}"
        return _BestModeComparisonLayout(unavailable, "", unavailable)
    wall_color = _best_mode_ratio_color(wall_ratio)
    secondary_ratio = _secondary_runtime_ratio(
        joined.candidate,
        joined.baseline,
        baseline_mode=baseline_mode,
    )
    if secondary_ratio is None:
        inline = (
            rf"\bestmodewallratio{{{wall_color}}}"
            rf"{{{_best_mode_value(wall_ratio)}}}"
        )
        return _BestModeComparisonLayout(
            inline,
            "",
            inline,
        )
    inline = (
        rf"\bestmodecompactratio{{{_best_mode_value(secondary_ratio)}}}"
        rf"{{{wall_color}}}{{{_best_mode_value(wall_ratio)}}}"
    )
    return _BestModeComparisonLayout(
        inline,
        rf"\bestmodecompactprefix{{{_best_mode_value(secondary_ratio)}}}",
        (
            rf"\bestmodeprimaryratio{{{wall_color}}}"
            rf"{{{_best_mode_value(wall_ratio)}}}"
        ),
    )


def _best_mode_runtime_comparison(
    joined: BestModeWorkload,
    *,
    baseline_static_na: bool = False,
) -> str:
    return _best_mode_runtime_comparison_layout(
        joined,
        baseline_static_na=baseline_static_na,
    ).inline


def _best_mode_comparison_columns(
    layout: _BestModeComparisonLayout,
) -> tuple[str, str]:
    """Expose a comparison's prefix and primary anchor as table columns."""

    return layout.prefix, layout.primary


def _best_mode_summary_terminal_label(
    joined: BestModeWorkload,
) -> _TerminalOutcome | _TerminalSummary | None:
    if joined.mode is None:
        return joined.terminal_label
    if not _ok(joined.baseline):
        return _terminal_measurement_label(joined.baseline)
    if not _ok(joined.candidate):
        return _terminal_measurement_label(joined.candidate)
    return None


def _best_mode_lc_cell(
    view: BestModeMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> _BestModeCellRows:
    selected, all_flow = view.workloads
    baseline_static_na = _legacy_baseline_static_na(
        catalog=catalog,
        accuracy=view.accuracy,
        process_key=view.process_family.key,
        n_final=view.n_final,
        workload=selected.workload,
    )
    selected_baseline_generation = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(selected.baseline, "generation_seconds")
    )
    all_flow_baseline_generation = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(all_flow.baseline, "generation_seconds")
    )
    selected_baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(
            selected.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    all_flow_baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(
            all_flow.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    selected_generation = _best_mode_generation_comparison_layout(
        selected,
        baseline_static_na=baseline_static_na,
    )
    all_flow_generation = _best_mode_generation_comparison_layout(
        all_flow,
        baseline_static_na=baseline_static_na,
        comparable=False,
    )
    selected_runtime = _best_mode_runtime_comparison_layout(
        selected,
        baseline_static_na=baseline_static_na,
    )
    all_flow_runtime = _best_mode_runtime_comparison_layout(
        all_flow,
        baseline_static_na=baseline_static_na,
    )
    return _BestModeCellRows(
        generation=(
            selected_baseline_generation,
            r"\matrixpunct{|}",
            all_flow_baseline_generation,
            *_best_mode_comparison_columns(selected_generation),
            r"\matrixpunct{|}",
            *_best_mode_comparison_columns(all_flow_generation),
        ),
        runtime=(
            selected_baseline_runtime,
            r"\matrixpunct{|}",
            all_flow_baseline_runtime,
            *_best_mode_comparison_columns(selected_runtime),
            r"\matrixpunct{|}",
            *_best_mode_comparison_columns(all_flow_runtime),
        ),
    )


def _best_mode_contracted_cell(
    view: BestModeMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> _BestModeCellRows:
    joined = view.workloads[0]
    baseline_static_na = _legacy_baseline_static_na(
        catalog=catalog,
        accuracy=view.accuracy,
        process_key=view.process_family.key,
        n_final=view.n_final,
        workload=joined.workload,
    )
    baseline_generation = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(joined.baseline, "generation_seconds")
    )
    baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _best_mode_metric(
            joined.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    generation_comparison = _best_mode_generation_comparison_layout(
        joined,
        baseline_static_na=baseline_static_na,
    )
    runtime_comparison = _best_mode_runtime_comparison_layout(
        joined,
        baseline_static_na=baseline_static_na,
    )
    return _BestModeCellRows(
        generation=(
            baseline_generation,
            *_best_mode_comparison_columns(generation_comparison),
        ),
        runtime=(
            baseline_runtime,
            *_best_mode_comparison_columns(runtime_comparison),
        ),
    )


def _best_mode_summary_pair(
    views: Sequence[BestModeMatrixCell],
    workload: Workload,
    field: str,
) -> str:
    joined, valid = _best_mode_summary_items(views, workload, field)
    if not valid:
        if field == "execution_seconds_per_point" and any(
            unavailable_execution_timing_record(measurement, field) is not None
            for item in joined
            for measurement in (item.baseline, item.candidate)
        ):
            return _not_exposed()
        terminal_label = _canonical_best_mode_terminal_label(
            tuple(_best_mode_summary_terminal_label(item) for item in joined)
        )
        if terminal_label is not None:
            display_label = _compact_terminal_summary_display(terminal_label)
            return (
                rf"\matrixstatus{{{terminal_label.color}}}"
                rf"{{{_tex_escape(display_label)}}}"
            )
        return r"\matrixna{ReportMuted}"
    return _ratio_statistics_tex(
        tuple(
            (float(item.baseline[field]), float(item.candidate[field]))
            for item in valid
        )
    )


def _best_mode_summary_items(
    views: Sequence[BestModeMatrixCell],
    workload: Workload,
    field: str,
) -> tuple[tuple[BestModeWorkload, ...], tuple[BestModeWorkload, ...]]:
    """Return joined and ratio-eligible workloads for one summary metric."""

    joined = tuple(
        next(item for item in view.workloads if item.workload is workload)
        for view in views
        if view.applicable
    )
    valid = tuple(
        item
        for item in joined
        if item.comparison_linked
        and item.mode is not None
        and _ok(item.baseline)
        and _ok(item.candidate)
        and item.baseline.get(field) is not None
        and item.candidate.get(field) is not None
        and unavailable_execution_timing_record(item.baseline, field) is None
        and unavailable_execution_timing_record(item.candidate, field) is None
        and math.isfinite(float(item.baseline[field]))
        and math.isfinite(float(item.candidate[field]))
        and float(item.baseline[field]) > 0.0
        and float(item.candidate[field]) >= 0.0
    )
    return joined, valid


def _best_mode_generation_mode_mix(
    views: Sequence[BestModeMatrixCell],
    accuracy: Accuracy,
) -> str:
    workload = (
        Workload.SELECTED_FLOW if accuracy is Accuracy.LC else Workload.CONTRACTED
    )
    _joined, valid = _best_mode_summary_items(
        views,
        workload,
        "generation_seconds",
    )
    if not valid:
        return ""
    counts = {
        mode: sum(item.mode is mode for item in valid) for mode in _BEST_MODE_ORDER
    }
    return (
        r"\bestmodemix{"
        + "|".join(
            f"{_BEST_MODE_CODES[mode]}:{counts[mode]}" for mode in _BEST_MODE_ORDER
        )
        + "}"
    )


def _best_mode_generation_summary(
    views: Sequence[BestModeMatrixCell],
    accuracy: Accuracy,
) -> str:
    if accuracy is Accuracy.LC:
        return _best_mode_summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "generation_seconds",
        )
    return _best_mode_summary_pair(
        views,
        Workload.CONTRACTED,
        "generation_seconds",
    )


def _best_mode_wall_summary(
    views: Sequence[BestModeMatrixCell],
    accuracy: Accuracy,
) -> str:
    if accuracy is Accuracy.LC:
        selected = _best_mode_summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "wall_seconds_per_point",
        )
        all_flow = _best_mode_summary_pair(
            views,
            Workload.ALL_FLOW,
            "wall_seconds_per_point",
        )
        return rf"\matrixsummaryworkloads{{{selected}}}{{{all_flow}}}"
    return _best_mode_summary_pair(
        views,
        Workload.CONTRACTED,
        "wall_seconds_per_point",
    )


def _best_mode_block(
    adapter: BaselineCandidateAdapter,
    accuracy: Accuracy,
    multiplicities: tuple[int, ...],
    *,
    block_index: int,
    block_count: int,
) -> list[str]:
    accuracy_label = {
        Accuracy.LC: "LC",
        Accuracy.NLC: "NLC",
        Accuracy.FULL: "full-colour",
    }[accuracy]
    if len(multiplicities) not in (2, 3):
        raise ValueError("matrix blocks must contain two or three multiplicities")
    column_spec = _metric_table_column_spec(
        accuracy,
        len(multiplicities),
    )
    lines = [
        r"\clearpage",
        r"\noindent\begin{minipage}{\linewidth}",
        (
            rf"\subsubsection*{{Built-in SM best measured mode versus AmpliCol "
            rf"{accuracy_label}"
            + (
                ""
                if block_count == 1
                else rf" (block {block_index + 1} of {block_count})"
            )
            + "}"
        ),
        r"\begingroup",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.00}",
        r"\makebox[\linewidth][c]{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            r"\textbf{ID} & \textbf{base process} & \textbf{metric} & "
            + " & ".join(
                _metric_group_cell(
                    rf"\textbf{{n={n_final}}}",
                    accuracy,
                )
                for n_final in multiplicities
            )
            + r" \\"
        ),
        r"\specialrule{0.85pt}{0pt}{0pt}",
    ]
    views_by_n: dict[int, list[BestModeMatrixCell]] = {
        n_final: [] for n_final in multiplicities
    }
    for row_index, family in enumerate(adapter.catalog.process_families):
        generation_row = [
            rf"\texttt{{{family.identifier}}}",
            family.label_tex,
            _metric_row_label(runtime=False),
        ]
        runtime_row = ["", "", _metric_row_label(runtime=True)]
        for n_final in multiplicities:
            view = adapter.best_mode_cell(accuracy, family, n_final)
            views_by_n[n_final].append(view)
            if not view.applicable:
                generation_row.append(
                    _metric_group_cell(
                        _not_applicable(),
                        accuracy,
                    )
                )
                runtime_row.append(
                    _metric_group_cell(
                        "",
                        accuracy,
                    )
                )
            elif accuracy is Accuracy.LC:
                rendered = _best_mode_lc_cell(
                    view,
                    catalog=adapter.catalog,
                )
                generation_row.extend(rendered.generation)
                runtime_row.extend(rendered.runtime)
            else:
                rendered = _best_mode_contracted_cell(
                    view,
                    catalog=adapter.catalog,
                )
                generation_row.extend(rendered.generation)
                runtime_row.extend(rendered.runtime)
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(generation_row) + r" \\")
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(runtime_row) + r" \\")
        lines.append(r"\addlinespace[0.05em]")
    boundary_note = (
        r" For the LC all-flow workload, AmpliCol direct-evaluation setup and "
        r"pyAmpliCol process generation have different boundaries, so the "
        r"candidate generation value is shown absolutely; no "
        r"misleading multiplier is formed."
        if accuracy is Accuracy.LC
        else ""
    )
    lines.extend(
        [
            r"\specialrule{1.05pt}{0.22em}{0.18em}",
            _metric_summary_header_row(
                multiplicities,
                accuracy,
                annotations=tuple(
                    _best_mode_generation_mode_mix(
                        views_by_n[n_final],
                        accuracy,
                    )
                    for n_final in multiplicities
                ),
            ),
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _metric_summary_group_cell(
                        _best_mode_generation_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        group_index=group_index,
                    )
                    for group_index, n_final in enumerate(multiplicities)
                )
                + r" \\[0.08em]"
            ),
            (
                rf"\multicolumn{{3}}{{@{{}}l}}{{"
                rf"{_matrix_wall_summary_label(accuracy)}}} & "
                + " & ".join(
                    _metric_summary_group_cell(
                        _best_mode_wall_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        group_index=group_index,
                    )
                    for group_index, n_final in enumerate(multiplicities)
                )
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\endgroup",
            (
                r"\ReportTableNote{{\scriptsize The candidate is selected "
                r"independently in "
                r"each cell and workload by the smallest validated wall time. "
                r"The upper row is generation in seconds and the lower row is "
                r"runtime in \(\mu\mathrm{s}/\mathrm{pt}\). In LC cells, each "
                r"pair is non-union-flow \texttt{|} union-flow. Generation "
                r"multipliers are coloured directly. Each runtime workload "
                r"uses \texttt{([xS]xW)}. The muted supplementary xS first "
                r"compares compatible exposed execution attributions; otherwise "
                r"it uses authenticated evaluator totals. Because original "
                r"AmpliCol exposes no evaluator-total clock, a candidate total "
                r"then uses the AmpliCol direct execution clock, or its wall "
                r"clock when direct execution is absent, as denominator. An "
                r"evaluator total is "
                r"never divided by a recurrence core. This evaluator-total "
                r"fallback is diagnostic only; execution/execution remains the "
                r"compatible supplementary attribution. The coloured xW is always "
                r"the candidate/AmpliCol wall-time multiplier used for selection "
                r"and as the primary figure of merit. If no compatible "
                r"supplementary clock exists, only \texttt{(xW)} is shown. No "
                r"clock is fabricated from another. Every "
                r"displayed number uses exactly three significant digits. "
                r"Original-AmpliCol "
                r"rows beyond its three-open-quark-line scope are catalog "
                r"static N/A entries; their candidates are shown absolutely and "
                r"excluded from ratio summaries. Unavailable baselines keep "
                r"their status; candidates remain absolute."
                + boundary_note
                + r" Only exact stored same-point links enter ratios."
                + r" Summary rows contain multipliers only in min, max, "
                r"median, average, and weighted-average order; the weighted "
                r"average is the ratio of timing sums. The framed bold entry "
                r"is the arithmetic average of the per-cell multipliers. "
                r"Muted (r), (c), and (e) labels on the generation row "
                r"identify the recurrence, compiled, and eager winner selected "
                r"by wall time for each workload; the runtime row does not "
                r"repeat them. LC generation uses "
                r"non-union flow only, while LC runtime keeps separate "
                r"non-union and union wall-only lines. Summary mode counts "
                r"use r|c|e for recurrence JIT O2, compiled JIT O3, and "
                r"eager-DAG JIT O2, respectively.}}"
            ),
            r"\end{minipage}",
        ]
    )
    return lines


def render_best_mode_table(
    accuracy: Accuracy,
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> str:
    """Render the fastest valid built-in mode against AmpliCol."""

    adapter = BaselineCandidateAdapter(caches, catalog=catalog)
    multiplicities = matrix_multiplicities(accuracy)
    blocks = _chunks(multiplicities, _MATRIX_BLOCK_SIZE)
    accuracy_label = {
        Accuracy.LC: "LC",
        Accuracy.NLC: "NLC",
        Accuracy.FULL: "full-colour",
    }[accuracy]
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by tools/performance_report/render.py; do not edit.",
        *_matrix_macros(),
        (
            r"\providecommand{\matrixpair}[2]{"
            r"\begin{tabular}[t]{@{}r@{\hspace{0.025in}"
            r"\matrixpunct{|}\hspace{0.025in}}r@{}}"
            r"#1&#2\end{tabular}}"
        ),
        *_compact_comparison_macros(),
        (
            r"\providecommand{\bestmodecode}[1]{"
            r"\textcolor{ReportMuted}{\texttt{(#1)}}}"
        ),
        (
            r"\providecommand{\bestmodecodeprefix}[1]{"
            r"\bestmodecode{#1}\hspace{0.025in}}"
        ),
        (
            r"\providecommand{\bestmodeabsolutechoice}[2]{"
            r"#1\hspace{0.025in}\bestmodecode{#2}}"
        ),
        (
            r"\providecommand{\bestmodemix}[1]{"
            r"\textcolor{ReportBlue}{\texttt{[#1]}}}"
        ),
        r"\clearpage",
        (
            r"\subsection{Built-in SM best measured mode versus AmpliCol: "
            f"{accuracy_label}}}"
        ),
    ]
    for block_index, block in enumerate(blocks):
        rendered = _best_mode_block(
            adapter,
            accuracy,
            block,
            block_index=block_index,
            block_count=len(blocks),
        )
        lines.extend(rendered[1:] if block_index == 0 else rendered)
    return "\n".join(lines) + "\n"


def render_all_best_mode_tables(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, str]:
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        _BEST_MODE_TABLE_NAMES[accuracy]: render_best_mode_table(
            accuracy,
            cache_source,
            catalog=catalog,
        )
        for accuracy in Accuracy
    }


def _z_setup(variant: ZVariant) -> str:
    labels = {
        "reference": r"\AC{} reference",
        "jit_o1": r"compiled JIT O1",
        "asm_o3": r"compiled ASM O3",
        "cpp_o3": r"compiled C++ O3",
        "jit_o3": r"compiled JIT O3",
        "eager_jit_o2": r"eager-DAG JIT O2",
        "recurrence_jit_o2": r"recurrence JIT O2",
    }
    return labels.get(variant.key, _tex_escape(variant.label))


def _z_value(
    joined: JoinedWorkload,
    field: str,
    *,
    reference: bool,
    microseconds: bool = False,
    comparable: bool = True,
    static_na: bool = False,
) -> str:
    if static_na:
        return _static_na()
    measurement = joined.baseline if reference else joined.candidate
    absolute = _metric(measurement, field, microseconds=microseconds)
    if reference or not _ok(measurement):
        return absolute
    if (
        unavailable_execution_timing_record(measurement, field) is not None
        or unavailable_execution_timing_record(joined.baseline, field) is not None
    ):
        if evaluator_total_timing_record(measurement) is not None:
            return absolute
        return _not_exposed()
    if not comparable or not joined.comparison_linked or not _ok(joined.baseline):
        return rf"\matrixncabsolute{{{absolute}}}"
    return absolute + r"\," + _ratio(measurement, joined.baseline, field)


def _z_evaluator_total(
    joined: JoinedWorkload,
    *,
    reference: bool,
    static_na: bool = False,
) -> str:
    if static_na:
        return _static_na()
    measurement = joined.baseline if reference else joined.candidate
    if reference:
        return _not_exposed()
    if not _ok(measurement):
        return _status(measurement)
    total = evaluator_total_seconds_per_point(measurement)
    if total is None:
        return _not_exposed()
    absolute = _time(total, microseconds=True)
    if not joined.comparison_linked or not _ok(joined.baseline):
        return rf"\matrixncabsolute{{{absolute}}}"
    baseline_wall = joined.baseline.get("wall_seconds_per_point")
    if (
        isinstance(baseline_wall, bool)
        or not isinstance(baseline_wall, (int, float))
        or not math.isfinite(float(baseline_wall))
        or float(baseline_wall) <= 0.0
    ):
        return absolute + r"\," + r"\matrixnaratio{ReportMuted}"
    ratio = total / float(baseline_wall)
    return (
        absolute
        + r"\,"
        + rf"\matrixratio{{{_best_mode_ratio_color(ratio)}}}"
        + rf"{{{_compact(ratio)}}}"
    )


def _z_block(
    adapter: BaselineCandidateAdapter,
    *,
    model: ModelKey,
    multiplicities: tuple[int, ...],
    block_index: int,
    block_count: int,
) -> list[str]:
    model_label = "Built-in SM" if model is ModelKey.BUILTIN_SM else "UFO-SM"
    candidate_cells = {
        (cell.n_final, cell.variant, cell.workload): cell
        for cell in adapter.catalog.z_cells()
        if cell.dataset_id == z_dataset_id(model)
    }
    lines = [
        r"\clearpage",
        r"\noindent\begin{minipage}{\linewidth}",
        (
            rf"\subsubsection*{{{model_label} \(d\bar d\to Z+\) gluon ladder"
            + (
                ""
                if block_count == 1
                else rf" (block {block_index + 1} of {block_count})"
            )
            + "}"
        ),
        r"\begingroup",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        (
            r"\begin{tabular}{@{}r L{1.50in} "
            r"R{1.26in} R{1.26in} R{1.26in} "
            r"@{\hspace{0.08in}}"
            r"R{1.26in} R{1.26in} R{1.26in}@{}}"
        ),
        r"\toprule",
        (
            r"\textbf{n} & \textbf{setup} & "
            r"\multicolumn{3}{c}{\textbf{selected flow, helicity sum}} & "
            r"\multicolumn{3}{c}{\textbf{all flows, single helicity}} \\"
        ),
        (
            r"& & \textbf{gen [s]} & "
            r"\textbf{wall [\(\mu\mathrm{s}/\mathrm{pt}\)]} & "
            r"\textbf{eval [\(\mu\mathrm{s}/\mathrm{pt}\)]} & "
            r"\textbf{gen [s]} & "
            r"\textbf{wall [\(\mu\mathrm{s}/\mathrm{pt}\)]} & "
            r"\textbf{eval [\(\mu\mathrm{s}/\mathrm{pt}\)]} \\"
        ),
        r"\midrule",
    ]
    for n_final in multiplicities:
        for variant in adapter.catalog.z_variants:
            selected = adapter.z_workload(
                model=model,
                n_final=n_final,
                variant=variant,
                workload=Workload.SELECTED_FLOW,
            )
            all_flow = adapter.z_workload(
                model=model,
                n_final=n_final,
                variant=variant,
                workload=Workload.ALL_FLOW,
            )
            reference = variant.execution_mode is ExecutionMode.AMPLICOL
            selected_static_na = False
            all_flow_static_na = False
            if not reference:
                selected_cell = candidate_cells[
                    (n_final, variant.key, Workload.SELECTED_FLOW)
                ]
                all_flow_cell = candidate_cells[
                    (n_final, variant.key, Workload.ALL_FLOW)
                ]
                selected_static_na = (
                    adapter.catalog.static_na_reason(selected_cell) is not None
                )
                all_flow_static_na = (
                    adapter.catalog.static_na_reason(all_flow_cell) is not None
                )
            if reference:
                lines.append(r"\rowcolor{refblue}")
            elif variant.key in {
                "jit_o3",
                "eager_jit_o2",
                "recurrence_jit_o2",
            }:
                lines.append(r"\rowcolor{ReportGreen!12}")
            row = (
                str(n_final),
                _z_setup(variant),
                _z_value(
                    selected,
                    "generation_seconds",
                    reference=reference,
                    static_na=selected_static_na,
                ),
                _z_value(
                    selected,
                    "wall_seconds_per_point",
                    reference=reference,
                    microseconds=True,
                    static_na=selected_static_na,
                ),
                _z_evaluator_total(
                    selected,
                    reference=reference,
                    static_na=selected_static_na,
                ),
                _z_value(
                    all_flow,
                    "generation_seconds",
                    reference=reference,
                    comparable=False,
                    static_na=all_flow_static_na,
                ),
                _z_value(
                    all_flow,
                    "wall_seconds_per_point",
                    reference=reference,
                    microseconds=True,
                    static_na=all_flow_static_na,
                ),
                _z_evaluator_total(
                    all_flow,
                    reference=reference,
                    static_na=all_flow_static_na,
                ),
            )
            lines.append(" & ".join(row) + r" \\")
        if n_final != multiplicities[-1]:
            lines.append(r"\midrule[0.45pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            (
                r"\ReportTableNote{Original AmpliCol is the denominator only "
                r"when successful and exactly linked by stored same-point "
                r"evidence, directly or through current recurrence. Otherwise "
                r"its status remains visible and successful pyAmpliCol values "
                r"are absolute. "
                r"Each pyAmpliCol row reports separate topology-replay and "
                r"all-flow-union generation and runtime measurements. "
                r"Parenthesized values are candidate/reference ratios. "
                r"All-flow generation shows the absolute pyAmpliCol value "
                r"without a ratio because its setup boundary differs from the "
                r"reference. The wall time is the common runtime observable. "
                r"Not exposed means that a successful wall measurement has no "
                r"dedicated authenticated evaluator-total timing. Authenticated "
                r"accumulated warmed evaluator totals appear in the eval column "
                r"with enough precision to remain distinct from wall time. Its "
                r"parenthesized multiplier uses the original-AmpliCol wall "
                r"measurement as denominator because the legacy reference has "
                r"no separate evaluator-total clock. Recurrence rows retain the "
                r"independently measured narrower recurrence core in detailed "
                r"evidence; it is not relabeled as evaluator total. Neither "
                r"evaluator total nor recurrence core is derived from wall "
                r"time or from the other. Older entries "
                r"without authenticated total evidence remain marked not "
                r"exposed; it is not a missing measurement. "
                + _tex_escape(STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6_DESCRIPTION)
                + " ("
                + _tex_escape(STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6)
                + r").}"
            ),
            r"\end{minipage}",
        ]
    )
    return lines


def render_z_ladder(
    model: ModelKey,
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> str:
    if model not in {ModelKey.BUILTIN_SM, ModelKey.UFO_SM}:
        raise ValueError("Z ladder model must be built-in SM or UFO-SM")
    adapter = BaselineCandidateAdapter(caches, catalog=catalog)
    model_label = "Built-in SM" if model is ModelKey.BUILTIN_SM else "UFO-SM"
    blocks = _chunks(tuple(range(1, 10)), _Z_BLOCK_SIZE)
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by tools/performance_report/render.py; do not edit.",
        *_matrix_macros(),
        r"\clearpage",
        rf"\subsection{{{model_label} dedicated \(d\bar d\to Z+\) gluon performance}}",
    ]
    for block_index, multiplicities in enumerate(blocks):
        block = _z_block(
            adapter,
            model=model,
            multiplicities=multiplicities,
            block_index=block_index,
            block_count=len(blocks),
        )
        lines.extend(block[1:] if block_index == 0 else block)
    return "\n".join(lines) + "\n"


def render_all_z_ladders(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, str]:
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        "result_z_builtin_sm_table.tex": render_z_ladder(
            ModelKey.BUILTIN_SM,
            cache_source,
            catalog=catalog,
        ),
        "result_z_external_sm_table.tex": render_z_ladder(
            ModelKey.UFO_SM,
            cache_source,
            catalog=catalog,
        ),
    }


def _scalar_value(measurement: Measurement, field: str) -> str:
    if field == "evaluator_total_seconds_per_point":
        if not _ok(measurement):
            return _status(measurement)
        total = evaluator_total_seconds_per_point(measurement)
        if total is None:
            return _not_exposed()
        return _time(total, microseconds=True)
    if field == "matrix_element":
        if not _ok(measurement):
            return _status(measurement)
        value = measurement.get(field)
        if value is None:
            return r"\matrixna{ReportMuted}"
        try:
            return rf"\texttt{{{_compact(float(value))}}}"
        except (TypeError, ValueError):
            return rf"\texttt{{{_tex_escape(str(value))}}}"
    return _metric(
        measurement,
        field,
        microseconds=field in {"wall_seconds_per_point", "execution_seconds_per_point"},
    )


def _scalar_relative_difference(measurement: Measurement) -> str:
    if not _ok(measurement):
        return _status(measurement)
    validation = measurement.get("validation")
    if not isinstance(validation, Mapping):
        return r"\matrixna{ReportMuted}"
    high_precision = validation.get("high_precision")
    if not isinstance(high_precision, Mapping):
        return r"\matrixna{ReportMuted}"
    value = high_precision.get("relative_difference")
    if value is None:
        return r"\matrixna{ReportMuted}"
    return rf"\texttt{{{_compact(float(value))}}}"


def render_scalar_ladder(
    dataset: ScalarDataset,
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
) -> str:
    index = MeasurementIndex(caches)
    dense = len(dataset.multiplicities) > 4
    label_width = "1.00in" if dense else "1.08in"
    value_width = "0.72in" if dense else "0.82in"
    tab_column_separation = "2.2pt" if dense else "3.2pt"
    column_spec = (
        rf"@{{}}L{{{label_width}}}"
        + "".join(rf"L{{{value_width}}}" for _ in dataset.multiplicities)
        + r"@{}"
    )
    rows = (
        ("generation [s]", "generation_seconds"),
        (r"wall [\(\mu\mathrm{s}/\mathrm{pt}\)]", "wall_seconds_per_point"),
        (
            r"evaluator total [\(\mu\mathrm{s}/\mathrm{pt}\)]",
            "evaluator_total_seconds_per_point",
        ),
        ("matrix element", "matrix_element"),
    )
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by tools/performance_report/render.py; do not edit.",
        *_matrix_macros(),
        rf"\subsection{{{_tex_escape(dataset.title)}}}",
        r"\begingroup",
        r"\scriptsize",
        rf"\setlength{{\tabcolsep}}{{{tab_column_separation}}}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            rf"\multicolumn{{{len(dataset.multiplicities) + 1}}}{{c}}"
            rf"{{\texttt{{{_tex_escape(dataset.process_template)}}}}} \\"
        ),
        (
            r"\textbf{metric} & "
            + " & ".join(
                rf"\textbf{{\texttt{{n={n_final}}}}}"
                for n_final in dataset.multiplicities
            )
            + r" \\"
        ),
        r"\midrule",
    ]
    for row_index, (label, field) in enumerate(rows):
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        values = [
            _scalar_value(
                index.get(
                    dataset.dataset_id,
                    dataset.dataset_id,
                    n_final,
                    Workload.CONTRACTED,
                ),
                field,
            )
            for n_final in dataset.multiplicities
        ]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    relative_values = [
        _scalar_relative_difference(
            index.get(
                dataset.dataset_id,
                dataset.dataset_id,
                n_final,
                Workload.CONTRACTED,
            )
        )
        for n_final in dataset.multiplicities
    ]
    lines.extend(
        [
            r"\rowcolor{ReportOrange!8}",
            "relative diff. vs hp & " + " & ".join(relative_values) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\endgroup",
            (
                r"\ReportTableNote{Wall time is the common runtime observable. "
                r"Evaluator total is the independently measured accumulated "
                r"warmed evaluator clock from its dedicated authenticated "
                r"record; it is never copied from or derived from wall time. "
                r"\textsc{not exposed} denotes a successful wall measurement, "
                r"and older entries without dedicated evaluator-total evidence "
                r"remain valid; it does not denote a missing result.}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def render_all_scalar_ladders(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, str]:
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        dataset.table_name: render_scalar_ladder(dataset, cache_source)
        for dataset in catalog.scalar_datasets
    }


def _renders_na(value: str) -> bool:
    return r"\matrixna{" in value or r"\matrixnaratio{" in value or "{N/A}" in value


def _measurements_by_cell_id(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
) -> dict[str, Measurement]:
    payloads = caches.values() if isinstance(caches, Mapping) else caches
    measurements: dict[str, Measurement] = {}
    for payload in payloads:
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("report cache has no entries list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("report cache contains a non-object entry")
            cell_id = entry.get("cell_id")
            measurement = entry.get("measurement")
            if not isinstance(cell_id, str) or not isinstance(measurement, Mapping):
                raise ValueError("report cache entry is missing its measurement")
            if cell_id in measurements:
                raise ValueError(f"duplicate report cell {cell_id!r}")
            measurements[cell_id] = measurement
    return measurements


def summarize_visible_completeness(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
    max_n_final: int = 9,
    policy: CampaignPolicy = STRICT_POLICY,
) -> VisibleCompleteness:
    """Audit every logical table slot that represents a required measurement.

    The check invokes the same field-level rendering helpers as the tables. It
    therefore catches an ``ok`` cache record whose missing submetric would still
    appear as ``N/A`` in TeX, as well as an applicable reset record. Structural
    matrix positions and reference-only execution fields are counted separately
    and are required to use their dedicated visible markers.
    """

    if max_n_final < 1:
        raise ValueError("max_n_final must be positive")
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    measurements = _measurements_by_cell_id(cache_source)
    declared_cells = tuple(
        cell for cell in catalog.measurement_cells() if cell.n_final <= max_n_final
    )
    static_na_cells = tuple(
        cell for cell in declared_cells if catalog.static_na_reason(cell) is not None
    )
    required_cells = tuple(
        cell for cell in declared_cells if catalog.static_na_reason(cell) is None
    )
    required_by_id = {cell.cell_id: cell for cell in required_cells}
    if len(required_by_id) != len(required_cells):
        raise ValueError("report catalog contains duplicate required cell IDs")
    cell_index = {
        (
            cell.dataset_id,
            cell.process_key,
            cell.n_final,
            cell.workload,
            cell.variant,
        ): cell
        for cell in catalog.measurement_cells()
    }
    rendered_cell_ids: set[str] = set()
    rendered_static_na_cell_ids: set[str] = set()
    na_slots: list[str] = []
    contract_errors: list[str] = []
    for cell in required_cells:
        measurement = measurements.get(cell.cell_id, _NA)
        status = measurement.get("status", ResultStatus.NOT_AVAILABLE.value)
        policy_terminal = (
            policy.allow_terminal_censors
            and policy_status_label(measurement) is not None
        )
        if status != ResultStatus.OK.value and not policy_terminal:
            contract_errors.append(
                f"{cell.cell_id}: required measurement status is "
                f"{status!r}, expected {ResultStatus.OK.value!r}"
            )
    for cell in static_na_cells:
        measurement = measurements.get(cell.cell_id)
        if measurement != _NA:
            contract_errors.append(
                f"{cell.cell_id}: catalog static N/A measurement differs "
                "from the canonical reset record"
            )

    def inspect(
        location: str,
        fragments: Sequence[str],
    ) -> None:
        for index, fragment in enumerate(fragments):
            if _renders_na(fragment):
                na_slots.append(f"{location}[{index}]")

    def record(
        cell: CellSpec,
        location: str,
        fragments: Sequence[str],
    ) -> None:
        if cell.n_final > max_n_final:
            return
        rendered_cell_ids.add(cell.cell_id)
        if catalog.static_na_reason(cell) is not None:
            rendered_static_na_cell_ids.add(cell.cell_id)
            if any(fragment != _static_na() for fragment in fragments):
                contract_errors.append(
                    f"{location} ({cell.cell_id}): catalog static N/A slot "
                    "does not use the dedicated marker"
                )
            return
        inspect(
            f"{location} ({cell.cell_id})",
            fragments,
        )

    def locate(
        dataset_id: str,
        process_key: str | None,
        n_final: int,
        workload: Workload,
        variant: str | None = None,
    ) -> CellSpec | None:
        cell = cell_index.get((dataset_id, process_key, n_final, workload, variant))
        if cell is None:
            contract_errors.append(
                f"renderer has no catalog cell for {dataset_id}/n{n_final}/"
                f"{process_key}/{variant}/{workload.value}"
            )
        return cell

    adapter = BaselineCandidateAdapter(cache_source, catalog=catalog)
    structural_seen = 0
    matrix_datasets = getattr(catalog, "matrix_datasets", ())
    process_families = getattr(catalog, "process_families", ())
    for dataset in matrix_datasets:
        views_by_n: dict[int, list[JoinedMatrixCell]] = {}
        for family in process_families:
            for n_final in dataset.multiplicities:
                if n_final > max_n_final:
                    continue
                view = adapter.matrix_cell(dataset, family, n_final)
                views_by_n.setdefault(n_final, []).append(view)
                if not view.applicable:
                    structural_seen += 1
                    if _renders_na(_not_applicable()):
                        contract_errors.append(
                            f"{dataset.table_name}/n{n_final}/{family.key}: "
                            "structural slot uses the missing-measurement marker"
                        )
                    continue
                for joined in view.workloads:
                    candidate = locate(
                        dataset.dataset_id,
                        family.key,
                        n_final,
                        joined.workload,
                    )
                    if candidate is None:
                        continue
                    baseline = catalog.baseline_cell(candidate)
                    if baseline is None:
                        contract_errors.append(
                            f"{candidate.cell_id}: matrix baseline is missing"
                        )
                        continue
                    baseline_static_na = catalog.static_na_reason(baseline) is not None
                    record(
                        baseline,
                        (
                            f"{dataset.table_name}/n{n_final}/{family.key}/"
                            f"{joined.workload.value}/baseline"
                        ),
                        (
                            (
                                _static_na()
                                if baseline_static_na
                                else _best_mode_metric(
                                    joined.baseline,
                                    "generation_seconds",
                                )
                            ),
                            (
                                _static_na()
                                if baseline_static_na
                                else _best_mode_metric(
                                    joined.baseline,
                                    "wall_seconds_per_point",
                                    microseconds=True,
                                )
                            ),
                        ),
                    )
                    candidate_generation = _fixed_mode_generation_comparison_layout(
                        joined,
                        baseline_mode=dataset.baseline.execution_mode,
                        baseline_static_na=baseline_static_na,
                        comparable=not (
                            joined.workload is Workload.ALL_FLOW
                            and dataset.candidate.execution_mode
                            is ExecutionMode.RECURRENCE
                            and dataset.baseline.execution_mode
                            is ExecutionMode.AMPLICOL
                        ),
                    )
                    record(
                        candidate,
                        (
                            f"{dataset.table_name}/n{n_final}/{family.key}/"
                            f"{joined.workload.value}/candidate"
                        ),
                        (
                            candidate_generation.inline,
                            _fixed_mode_runtime_comparison_layout(
                                joined,
                                candidate_mode=(dataset.candidate.execution_mode),
                                baseline_mode=(dataset.baseline.execution_mode),
                                baseline_static_na=baseline_static_na,
                            ).inline,
                        ),
                    )
        for n_final, views in views_by_n.items():
            if not any(view.applicable for view in views):
                continue
            for label, fragment in (
                (
                    "generation-summary",
                    _matrix_generation_summary(views, dataset),
                ),
                (
                    "wall-summary",
                    _matrix_wall_summary(views, dataset.candidate.accuracy),
                ),
            ):
                if _renders_na(fragment):
                    na_slots.append(f"{dataset.table_name}/n{n_final}/{label}")

    for accuracy in Accuracy:
        table_name = _BEST_MODE_TABLE_NAMES[accuracy]
        multiplicities = matrix_multiplicities(accuracy)
        views_by_n: dict[int, list[BestModeMatrixCell]] = {}
        for family in process_families:
            for n_final in multiplicities:
                if n_final > max_n_final:
                    continue
                view = adapter.best_mode_cell(accuracy, family, n_final)
                views_by_n.setdefault(n_final, []).append(view)
                if not view.applicable:
                    continue
                for joined in view.workloads:
                    baseline_static_na = _legacy_baseline_static_na(
                        catalog=catalog,
                        accuracy=accuracy,
                        process_key=family.key,
                        n_final=n_final,
                        workload=joined.workload,
                    )
                    inspect(
                        (
                            f"{table_name}/n{n_final}/{family.key}/"
                            f"{joined.workload.value}"
                        ),
                        (
                            (
                                _static_na()
                                if baseline_static_na
                                else _best_mode_metric(
                                    joined.baseline,
                                    "generation_seconds",
                                )
                            ),
                            _best_mode_generation_comparison(
                                joined,
                                baseline_static_na=baseline_static_na,
                                comparable=not (
                                    accuracy is Accuracy.LC
                                    and joined.workload is Workload.ALL_FLOW
                                ),
                            ),
                            (
                                _static_na()
                                if baseline_static_na
                                else _best_mode_metric(
                                    joined.baseline,
                                    "wall_seconds_per_point",
                                    microseconds=True,
                                )
                            ),
                            _best_mode_runtime_comparison(
                                joined,
                                baseline_static_na=baseline_static_na,
                            ),
                        ),
                    )
        for n_final, views in views_by_n.items():
            if not any(view.applicable for view in views):
                continue
            inspect(
                f"{table_name}/n{n_final}/best-mode-summaries",
                (
                    _best_mode_generation_summary(views, accuracy),
                    _best_mode_wall_summary(views, accuracy),
                ),
            )

    not_exposed_seen = 0
    if getattr(catalog, "z_variants", ()):
        for model in (ModelKey.BUILTIN_SM, ModelKey.UFO_SM):
            if model not in getattr(catalog, "models", {}):
                continue
            for n_final in range(1, min(max_n_final, 9) + 1):
                for variant in catalog.z_variants:
                    for workload in (
                        Workload.SELECTED_FLOW,
                        Workload.ALL_FLOW,
                    ):
                        joined = adapter.z_workload(
                            model=model,
                            n_final=n_final,
                            variant=variant,
                            workload=workload,
                        )
                        reference = variant.execution_mode is ExecutionMode.AMPLICOL
                        if reference:
                            cell = locate(
                                "reference_amplicol_lc",
                                "dd_z_jets",
                                n_final,
                                workload,
                            )
                            if cell is None:
                                continue
                            fragments = (
                                _z_value(
                                    joined,
                                    "generation_seconds",
                                    reference=True,
                                    comparable=workload is not Workload.ALL_FLOW,
                                ),
                                _z_value(
                                    joined,
                                    "wall_seconds_per_point",
                                    reference=True,
                                    microseconds=True,
                                ),
                            )
                            not_exposed_seen += 1
                            if _renders_na(_not_exposed()):
                                contract_errors.append(
                                    "reference execution slot uses the "
                                    "missing-measurement marker"
                                )
                        else:
                            cell = locate(
                                z_dataset_id(model),
                                "dd_z_jets",
                                n_final,
                                workload,
                                variant.key,
                            )
                            if cell is None:
                                continue
                            static_na = catalog.static_na_reason(cell) is not None
                            fragments = (
                                _z_value(
                                    joined,
                                    "generation_seconds",
                                    reference=False,
                                    comparable=workload is not Workload.ALL_FLOW,
                                    static_na=static_na,
                                ),
                                _z_value(
                                    joined,
                                    "wall_seconds_per_point",
                                    reference=False,
                                    microseconds=True,
                                    static_na=static_na,
                                ),
                                _z_evaluator_total(
                                    joined,
                                    reference=False,
                                    static_na=static_na,
                                ),
                            )
                        record(
                            cell,
                            (
                                f"result_z_{model.value}/n{n_final}/"
                                f"{variant.key}/{workload.value}"
                            ),
                            fragments,
                        )

    for dataset in getattr(catalog, "scalar_datasets", ()):
        for n_final in dataset.multiplicities:
            if n_final > max_n_final:
                continue
            cell = locate(
                dataset.dataset_id,
                dataset.dataset_id,
                n_final,
                Workload.CONTRACTED,
            )
            if cell is None:
                continue
            measurement = measurements.get(cell.cell_id, _NA)
            record(
                cell,
                f"{dataset.table_name}/n{n_final}",
                (
                    _scalar_value(measurement, "generation_seconds"),
                    _scalar_value(measurement, "wall_seconds_per_point"),
                    _scalar_value(
                        measurement,
                        "evaluator_total_seconds_per_point",
                    ),
                    _scalar_value(measurement, "matrix_element"),
                    _scalar_relative_difference(measurement),
                ),
            )

    if not (matrix_datasets or getattr(catalog, "scalar_datasets", ())):
        # Small injected catalogs used by the final-audit unit boundary do not
        # carry renderer metadata. Their cells still receive the core field-level
        # visibility check, while the canonical catalog exercises every table.
        for cell in required_cells:
            measurement = measurements.get(cell.cell_id, _NA)
            fragments = [
                _metric(measurement, "generation_seconds"),
                _metric(
                    measurement,
                    "wall_seconds_per_point",
                    microseconds=True,
                ),
            ]
            if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL:
                fragments.append(
                    _metric(
                        measurement,
                        "execution_seconds_per_point",
                        microseconds=True,
                    )
                )
            record(cell, f"injected-catalog/{cell.cell_id}", fragments)
        for cell in static_na_cells:
            record(
                cell,
                f"injected-catalog/{cell.cell_id}",
                (_static_na(), _static_na()),
            )

    accounting = report_display_accounting(
        catalog=catalog,
        max_n_final=max_n_final,
    )
    if structural_seen != (accounting.structurally_not_applicable_display_slot_count):
        contract_errors.append(
            "structural display-slot enumeration differs from the catalog: "
            f"{structural_seen}/"
            f"{accounting.structurally_not_applicable_display_slot_count}"
        )
    if not_exposed_seen != accounting.not_exposed_display_slot_count:
        contract_errors.append(
            "not-exposed display-slot enumeration differs from the catalog: "
            f"{not_exposed_seen}/{accounting.not_exposed_display_slot_count}"
        )
    required_ids = set(required_by_id)
    missing = tuple(sorted(required_ids - rendered_cell_ids))
    static_na_ids = {cell.cell_id for cell in static_na_cells}
    missing_static_na = tuple(sorted(static_na_ids - rendered_static_na_cell_ids))
    if missing_static_na:
        contract_errors.append(
            "catalog static N/A cells are not rendered: " + ", ".join(missing_static_na)
        )
    return VisibleCompleteness(
        maximum_n_final=max_n_final,
        declared_measurement_cell_count=(accounting.declared_measurement_cell_count),
        required_measurement_count=accounting.required_measurement_count,
        rendered_required_measurement_count=len(rendered_cell_ids & required_ids),
        structurally_not_applicable_display_slot_count=structural_seen,
        not_exposed_display_slot_count=not_exposed_seen,
        applicable_na_display_slots=tuple(sorted(set(na_slots))),
        missing_rendered_cell_ids=missing,
        contract_errors=tuple(sorted(set(contract_errors))),
        catalog_static_na_cell_count=accounting.catalog_static_na_cell_count,
        rendered_catalog_static_na_cell_count=len(rendered_static_na_cell_ids),
    )


def render_all_tables(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
    authenticated_source_lineage: tuple[str, str] | None = None,
) -> dict[str, str]:
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        SUMMARY_TABLE_NAME: render_validation_summary(
            cache_source,
            catalog=catalog,
            authenticated_source_lineage=authenticated_source_lineage,
        ),
        **render_all_best_mode_tables(cache_source, catalog=catalog),
        **render_all_matrix_tables(cache_source, catalog=catalog),
        **render_all_z_ladders(cache_source, catalog=catalog),
        **render_all_scalar_ladders(cache_source, catalog=catalog),
    }


__all__ = [
    "BaselineCandidateAdapter",
    "JoinedMatrixCell",
    "JoinedWorkload",
    "MeasurementIndex",
    "VisibleCompleteness",
    "render_all_best_mode_tables",
    "render_all_matrix_tables",
    "render_all_scalar_ladders",
    "render_all_tables",
    "render_all_z_ladders",
    "render_best_mode_table",
    "render_matrix_table",
    "render_scalar_ladder",
    "render_validation_summary",
    "render_z_ladder",
    "summarize_visible_completeness",
]
