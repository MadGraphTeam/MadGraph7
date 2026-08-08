# SPDX-License-Identifier: 0BSD
"""Publication-facing numerical-validation coverage for the report."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .agreements import (
    DIRECT_AGREEMENT_FIELD,
    INDEPENDENT_AUTHORITY_FIELD,
    LC_LEGACY_PYAMPLICOL_COMPONENT,
    MADGRAPH_COMPARISON_FIELD,
)
from .campaign_policy import policy_status_label
from .catalog import REPORT_CATALOG, ReportCatalog
from .display_contract import report_display_accounting
from .models import CellSpec, ExecutionMode, ResultStatus

Measurement = Mapping[str, object]
CachePayload = Mapping[str, object]

SUMMARY_TABLE_NAME = "result_validation_summary.tex"
MAX_PUBLICATION_MULTIPLICITY = 9


def _number(record: Mapping[str, object], field: str) -> float | None:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _maximum(records: Iterable[Mapping[str, object]], field: str) -> float | None:
    values = tuple(
        value for record in records if (value := _number(record, field)) is not None
    )
    return max(values) if values else None


def _validation_record(
    measurement: Measurement,
    name: str,
) -> Mapping[str, object] | None:
    validation = measurement.get("validation")
    if not isinstance(validation, Mapping):
        return None
    record = validation.get(name)
    return record if isinstance(record, Mapping) else None


def _successful(measurement: Measurement) -> bool:
    validation = measurement.get("validation")
    return (
        measurement.get("status") == ResultStatus.OK.value
        and isinstance(validation, Mapping)
        and validation.get("status") == ResultStatus.OK.value
    )


def _source_revision(measurement: Measurement) -> str | None:
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    for field in (
        "report_measured_source_revision",
        "report_source_revision",
        "source_revision",
    ):
        value = provenance.get(field)
        if isinstance(value, str) and value and value != "unknown":
            return value
    return None


def _cell_measurements(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
) -> dict[str, Measurement]:
    payloads = caches.values() if isinstance(caches, Mapping) else caches
    result: dict[str, Measurement] = {}
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
            if cell_id in result:
                raise ValueError(f"duplicate report cell {cell_id!r}")
            result[cell_id] = measurement
    return result


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Numerical evidence across the complete declared publication range."""

    declared_by_n: tuple[tuple[int, int], ...]
    expected_by_n: tuple[tuple[int, int], ...]
    static_na_by_n: tuple[tuple[int, int], ...]
    passed_by_n: tuple[tuple[int, int], ...]
    policy_terminal_by_n: tuple[tuple[int, int], ...]
    status_counts: tuple[tuple[str, int], ...]
    madgraph_reference_count: int
    madgraph_comparison_count: int
    madgraph_comparison_maximum_relative_difference: float | None
    legacy_reference_count: int
    legacy_comparison_pass_count: int
    legacy_comparison_failure_count: int
    legacy_comparison_maximum_relative_difference: float | None
    cross_mode_count: int
    cross_mode_maximum_relative_difference: float | None
    resolved_count: int
    resolved_maximum_relative_difference: float | None
    high_precision_count: int
    high_precision_maximum_relative_difference: float | None
    source_revisions: tuple[str, ...]
    successful_source_identity_count: int

    @property
    def legacy_comparison_count(self) -> int:
        return self.legacy_comparison_pass_count + self.legacy_comparison_failure_count

    @property
    def declared_total(self) -> int:
        return sum(count for _, count in self.declared_by_n)

    @property
    def expected_total(self) -> int:
        return sum(count for _, count in self.expected_by_n)

    @property
    def static_na_total(self) -> int:
        return sum(count for _, count in self.static_na_by_n)

    @property
    def passed_total(self) -> int:
        return sum(count for _, count in self.passed_by_n)

    @property
    def complete(self) -> bool:
        return self.passed_total == self.expected_total

    @property
    def policy_terminal_total(self) -> int:
        return sum(count for _, count in self.policy_terminal_by_n)

    @property
    def policy_complete(self) -> bool:
        return self.passed_total + self.policy_terminal_total == self.expected_total

    @property
    def uniform_source_revision(self) -> str | None:
        if (
            len(self.source_revisions) == 1
            and self.successful_source_identity_count == self.passed_total
        ):
            return self.source_revisions[0]
        return None


def summarize_validation(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> ValidationSummary:
    """Summarize validation evidence without treating missing cells as passing."""

    measurements = _cell_measurements(caches)
    cells = tuple(
        cell
        for cell in catalog.measurement_cells()
        if cell.n_final <= MAX_PUBLICATION_MULTIPLICITY
    )
    declared = Counter(cell.n_final for cell in cells)
    static_na = Counter(
        cell.n_final for cell in cells if catalog.static_na_reason(cell) is not None
    )
    expected = declared - static_na
    passed: Counter[int] = Counter()
    policy_terminal: Counter[int] = Counter()
    statuses: Counter[str] = Counter()
    madgraph_reference_count = 0
    madgraph_comparisons: list[Mapping[str, object]] = []
    legacy_reference_count = 0
    legacy_comparisons: list[Mapping[str, object]] = []
    cross_mode: list[Mapping[str, object]] = []
    resolved: list[Mapping[str, object]] = []
    high_precision: list[Mapping[str, object]] = []
    revisions: list[str] = []

    for cell in cells:
        if catalog.static_na_reason(cell) is not None:
            statuses["static-na"] += 1
            continue
        measurement = measurements.get(cell.cell_id, {})
        status = str(measurement.get("status", ResultStatus.NOT_AVAILABLE.value))
        statuses[status] += 1
        if not _successful(measurement):
            if policy_status_label(measurement) is not None:
                policy_terminal[cell.n_final] += 1
            continue
        passed[cell.n_final] += 1
        revision = _source_revision(measurement)
        if revision is not None:
            revisions.append(revision)
        validation = measurement["validation"]
        assert isinstance(validation, Mapping)
        method = validation.get("method")
        if method == "independent-madgraph-tree-level-oracle":
            madgraph_reference_count += 1
        elif method == "independent-original-amplicol-oracle":
            legacy_reference_count += 1
        if (
            comparison := _validation_record(
                measurement,
                MADGRAPH_COMPARISON_FIELD,
            )
        ) is not None:
            madgraph_comparisons.append(comparison)
        if (pointwise := _validation_record(measurement, "pointwise")) is not None:
            reference_mode = _pointwise_reference_mode(
                cell,
                validation,
                catalog=catalog,
            )
            if reference_mode is ExecutionMode.AMPLICOL:
                legacy_comparisons.append(pointwise)
            elif reference_mode is ExecutionMode.RECURRENCE:
                cross_mode.append(pointwise)
        legacy_comparisons.extend(_additional_legacy_comparisons(validation))
        if (record := _validation_record(measurement, "resolved_sum")) is not None:
            resolved.append(record)
        if (record := _validation_record(measurement, "high_precision")) is not None:
            high_precision.append(record)

    multiplicities = tuple(sorted(declared))
    return ValidationSummary(
        declared_by_n=tuple((n_final, declared[n_final]) for n_final in multiplicities),
        expected_by_n=tuple((n_final, expected[n_final]) for n_final in multiplicities),
        static_na_by_n=tuple(
            (n_final, static_na[n_final]) for n_final in multiplicities
        ),
        passed_by_n=tuple((n_final, passed[n_final]) for n_final in multiplicities),
        policy_terminal_by_n=tuple(
            (n_final, policy_terminal[n_final]) for n_final in multiplicities
        ),
        status_counts=tuple(sorted(statuses.items())),
        madgraph_reference_count=madgraph_reference_count,
        madgraph_comparison_count=len(madgraph_comparisons),
        madgraph_comparison_maximum_relative_difference=_maximum(
            madgraph_comparisons,
            "relative_difference",
        ),
        legacy_reference_count=legacy_reference_count,
        legacy_comparison_pass_count=sum(
            record.get("status") == ResultStatus.OK.value
            for record in legacy_comparisons
        ),
        legacy_comparison_failure_count=sum(
            record.get("status") != ResultStatus.OK.value
            for record in legacy_comparisons
        ),
        legacy_comparison_maximum_relative_difference=_maximum(
            legacy_comparisons,
            "relative_difference",
        ),
        cross_mode_count=len(cross_mode),
        cross_mode_maximum_relative_difference=_maximum(
            cross_mode,
            "relative_difference",
        ),
        resolved_count=len(resolved),
        resolved_maximum_relative_difference=_maximum(
            resolved,
            "maximum_relative_difference",
        ),
        high_precision_count=len(high_precision),
        high_precision_maximum_relative_difference=_maximum(
            high_precision,
            "relative_difference",
        ),
        source_revisions=tuple(sorted(set(revisions))),
        successful_source_identity_count=len(revisions),
    )


def _pointwise_reference_mode(
    cell: CellSpec,
    validation: Mapping[str, object],
    *,
    catalog: ReportCatalog,
) -> ExecutionMode | None:
    """Identify the actual endpoint of a stored pointwise comparison."""

    authority = validation.get(INDEPENDENT_AUTHORITY_FIELD)
    if isinstance(authority, Mapping):
        selected = authority.get("selected_cell_id")
        if isinstance(selected, str):
            try:
                return catalog.cell(selected).measurement.execution_mode
            except KeyError:
                return None
    baseline = catalog.validation_baseline_cell(cell)
    return None if baseline is None else baseline.measurement.execution_mode


def _additional_legacy_comparisons(
    validation: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Return retained legacy records not represented by ``pointwise``."""

    records: list[Mapping[str, object]] = []
    direct = validation.get(DIRECT_AGREEMENT_FIELD)
    if isinstance(direct, list):
        records.extend(
            record
            for record in direct
            if isinstance(record, Mapping)
            and record.get("edge_kind") == LC_LEGACY_PYAMPLICOL_COMPONENT
        )
    for field in (
        "legacy_amplicol_comparison",
        "legacy_comparison",
        "amplicol_comparison",
    ):
        raw = validation.get(field)
        if not isinstance(raw, Mapping):
            continue
        nested = raw.get("pointwise")
        if isinstance(nested, Mapping):
            records.append(nested)
        elif "status" in raw:
            records.append(raw)
    return tuple(records)


def _scientific(value: float | None) -> str:
    if value is None:
        return r"\textcolor{ReportMuted}{---}"
    if value == 0.0:
        return r"\texttt{0}"
    coefficient, exponent = f"{value:.3e}".split("e")
    return rf"\(\texttt{{{coefficient}}}\times10^{{{int(exponent)}}}\)"


def _coverage_status(passed: int, expected: int, policy_terminal: int = 0) -> str:
    if passed == expected:
        return r"\textcolor{ReportGreen}{verified}"
    if passed + policy_terminal == expected:
        return (
            r"\textcolor{ReportOrange}{policy-complete ("
            f"{policy_terminal} censored"
            r")}"
        )
    open_count = expected - passed - policy_terminal
    return (
        r"\textcolor{ReportOrange}{incomplete ("
        f"{policy_terminal} censored, {open_count} open"
        r")}"
    )


def render_validation_summary(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
    authenticated_source_lineage: tuple[str, str] | None = None,
) -> str:
    """Render a compact summary immediately before the performance tables."""

    summary = summarize_validation(caches, catalog=catalog)
    display = report_display_accounting(
        catalog=catalog,
        max_n_final=MAX_PUBLICATION_MULTIPLICITY,
    )
    declared_by_n = dict(summary.declared_by_n)
    static_na_by_n = dict(summary.static_na_by_n)
    passed_by_n = dict(summary.passed_by_n)
    terminal_by_n = dict(summary.policy_terminal_by_n)
    total_coverage = _coverage_status(
        summary.passed_total,
        summary.expected_total,
        summary.policy_terminal_total,
    )
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by tools/performance_report/validation_summary.py; do not edit.",
        r"\begin{samepage}",
        r"\subsection*{Numerical validation summary}",
        (
            r"Only measurements whose numerical checks pass are counted below. "
            r"The scope comprises every applicable report cell across the "
            r"complete declared multiplicity range."
        ),
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}l r r r l@{}}",
        r"\toprule",
        (
            r"\textbf{final-state multiplicity} & \textbf{declared} & "
            r"\textbf{static N/A} & \textbf{verified} & "
            r"\textbf{measurable coverage} \\"
        ),
        r"\midrule",
    ]
    for n_final, expected in summary.expected_by_n:
        passed = passed_by_n[n_final]
        lines.append(
            f"{n_final} & {declared_by_n[n_final]} & "
            f"{static_na_by_n[n_final]} & {passed} & "
            f"{_coverage_status(passed, expected, terminal_by_n[n_final])}"
            r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"\textbf{total} & "
                f"{summary.declared_total} & {summary.static_na_total} & "
                f"{summary.passed_total} & "
                f"{total_coverage}"
                r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{samepage}",
            (
                r"\ReportTableNote{Full-catalog display accounting through \(n=9\): "
                f"{display.declared_measurement_cell_count} declared cells, comprising "
                f"{display.required_measurement_count} measurable cells and "
                f"{display.catalog_static_na_cell_count} catalog-authenticated "
                r"static N/A rows; "
                f"{display.structurally_not_applicable_display_slot_count} "
                r"matrix process/multiplicity positions marked either "
                r"\textsc{not applicable} or \textsc{not run}; and "
                f"{display.not_exposed_display_slot_count} "
                r"reference execution fields marked \textsc{not exposed}. "
                r"The latter two categories are intentional table structure, "
                r"not unfilled measurements.}"
            ),
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{@{}l r r@{}}",
            r"\toprule",
            (
                r"\textbf{validation evidence} & \textbf{records} & "
                r"\textbf{max. relative difference} \\"
            ),
            r"\midrule",
            (
                r"MadGraph standalone tree-level reference records"
                f" & {summary.madgraph_reference_count} & "
                r"\textcolor{ReportMuted}{reference} \\"
            ),
            (
                r"\PAC{} versus MadGraph binary64 (p200; OTF p16)"
                f" & {summary.madgraph_comparison_count} & "
                f"{_scientific(summary.madgraph_comparison_maximum_relative_difference)}"
                r" \\"
            ),
            (
                r"original-\AC{} legacy diagnostic records"
                f" & {summary.legacy_reference_count} & "
                r"\textcolor{ReportMuted}{diagnostic} \\"
            ),
            (
                r"\PAC{} versus original-\AC{} legacy diagnostic"
                f" & {summary.legacy_comparison_pass_count} pass, "
                f"{summary.legacy_comparison_failure_count} mismatch & "
                f"{_scientific(summary.legacy_comparison_maximum_relative_difference)}"
                r" \\"
            ),
            (
                r"compiled/eager/OTF versus recurrence"
                f" & {summary.cross_mode_count} & "
                f"{_scientific(summary.cross_mode_maximum_relative_difference)}"
                r" \\"
            ),
            (
                r"optimized result versus resolved sum"
                f" & {summary.resolved_count} & "
                f"{_scientific(summary.resolved_maximum_relative_difference)}"
                r" \\"
            ),
            (
                r"binary64 result versus p32"
                f" & {summary.high_precision_count} & "
                f"{_scientific(summary.high_precision_maximum_relative_difference)}"
                r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )
    revision = summary.uniform_source_revision
    if (
        authenticated_source_lineage is not None
        and summary.successful_source_identity_count == summary.passed_total
        and set(summary.source_revisions).issubset(set(authenticated_source_lineage))
    ):
        ancestor, descendant = authenticated_source_lineage
        source_text = (
            rf"\nolinkurl{{{ancestor}}} \(\rightarrow\) "
            rf"\nolinkurl{{{descendant}}} "
            r"\textcolor{ReportGreen}{(authenticated Class-C lineage)}"
        )
    elif revision is None:
        source_text = (
            r"\textcolor{ReportOrange}{source identity is incomplete or mixed}"
        )
    else:
        source_text = rf"\nolinkurl{{{revision}}}"
    lines.extend(
        [
            (
                r"\ReportTableNote{Profile: \texttt{\ReportProfileName}. "
                r"Measured source: "
                + source_text
                + r". The authoritative \PAC{} versus MadGraph binary64 "
                r"full-colour comparison uses p200 for recurrence, compiled, "
                r"and eager candidates and the supported p16 lane for OTF; "
                r"all use a relative tolerance "
                r"of \(10^{-10}\). Original-\AC{} comparisons are retained "
                r"non-authoritative legacy diagnostics at \(10^{-8}\); a "
                r"legacy mismatch does not invalidate a candidate; cross-mode "
                r"(including OTF/recurrence), resolved-sum, and p32 "
                r"comparisons use \(10^{-12}\).}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "MAX_PUBLICATION_MULTIPLICITY",
    "SUMMARY_TABLE_NAME",
    "ValidationSummary",
    "render_validation_summary",
    "summarize_validation",
]
