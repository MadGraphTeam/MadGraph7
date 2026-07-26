# SPDX-License-Identifier: 0BSD
"""Publication-facing numerical-validation coverage for the report."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .campaign_policy import policy_status_label
from .catalog import REPORT_CATALOG, ReportCatalog
from .display_contract import report_display_accounting
from .models import CellSpec, ResultStatus

Measurement = Mapping[str, object]
CachePayload = Mapping[str, object]

SUMMARY_TABLE_NAME = "result_validation_summary.tex"
MAX_PUBLICATION_MULTIPLICITY = 4


def _number(record: Mapping[str, object], field: str) -> float | None:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _maximum(records: Iterable[Mapping[str, object]], field: str) -> float | None:
    values = tuple(
        value
        for record in records
        if (value := _number(record, field)) is not None
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
    """Numerical evidence summarized over every applicable ``n <= 4`` cell."""

    expected_by_n: tuple[tuple[int, int], ...]
    passed_by_n: tuple[tuple[int, int], ...]
    policy_terminal_by_n: tuple[tuple[int, int], ...]
    status_counts: tuple[tuple[str, int], ...]
    oracle_count: int
    independent_count: int
    independent_maximum_relative_difference: float | None
    cross_mode_count: int
    cross_mode_maximum_relative_difference: float | None
    resolved_count: int
    resolved_maximum_relative_difference: float | None
    high_precision_count: int
    high_precision_maximum_relative_difference: float | None
    source_revisions: tuple[str, ...]
    successful_source_identity_count: int

    @property
    def expected_total(self) -> int:
        return sum(count for _, count in self.expected_by_n)

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
        return (
            self.passed_total + self.policy_terminal_total
            == self.expected_total
        )

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
    expected = Counter(cell.n_final for cell in cells)
    passed: Counter[int] = Counter()
    policy_terminal: Counter[int] = Counter()
    statuses: Counter[str] = Counter()
    oracle_count = 0
    independent: list[Mapping[str, object]] = []
    cross_mode: list[Mapping[str, object]] = []
    resolved: list[Mapping[str, object]] = []
    high_precision: list[Mapping[str, object]] = []
    revisions: list[str] = []

    for cell in cells:
        measurement = measurements.get(cell.cell_id, {})
        status = str(
            measurement.get("status", ResultStatus.NOT_AVAILABLE.value)
        )
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
        if validation.get("method") == "independent-original-amplicol-oracle":
            oracle_count += 1
        if (pointwise := _validation_record(measurement, "pointwise")) is not None:
            if _uses_independent_reference(cell):
                independent.append(pointwise)
            else:
                cross_mode.append(pointwise)
        if (record := _validation_record(measurement, "resolved_sum")) is not None:
            resolved.append(record)
        if (record := _validation_record(measurement, "high_precision")) is not None:
            high_precision.append(record)

    multiplicities = tuple(sorted(expected))
    return ValidationSummary(
        expected_by_n=tuple((n_final, expected[n_final]) for n_final in multiplicities),
        passed_by_n=tuple((n_final, passed[n_final]) for n_final in multiplicities),
        policy_terminal_by_n=tuple(
            (n_final, policy_terminal[n_final])
            for n_final in multiplicities
        ),
        status_counts=tuple(sorted(statuses.items())),
        oracle_count=oracle_count,
        independent_count=len(independent),
        independent_maximum_relative_difference=_maximum(
            independent,
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


def _uses_independent_reference(cell: CellSpec) -> bool:
    return (
        cell.dataset_id.startswith("matrix_recurrence_")
        or cell.dataset_id.startswith("z_")
    )


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
) -> str:
    """Render a compact summary immediately before the performance tables."""

    summary = summarize_validation(caches, catalog=catalog)
    display = report_display_accounting(
        catalog=catalog,
        max_n_final=MAX_PUBLICATION_MULTIPLICITY,
    )
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
        r"\subsection*{Numerical validation summary}",
        (
            r"Only measurements whose numerical checks pass are counted below. "
            r"The scope comprises every applicable report cell with at most "
            r"four final-state particles."
        ),
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}r r r L{1.65in}@{}}",
        r"\toprule",
        (
            r"\textbf{final-state multiplicity} & \textbf{required} & "
            r"\textbf{verified} & \textbf{coverage} \\"
        ),
        r"\midrule",
    ]
    for n_final, expected in summary.expected_by_n:
        passed = passed_by_n[n_final]
        lines.append(
            f"{n_final} & {expected} & {passed} & "
            f"{_coverage_status(passed, expected, terminal_by_n[n_final])}"
            r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"\textbf{total} & "
                f"{summary.expected_total} & {summary.passed_total} & "
                f"{total_coverage}"
                r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                r"\ReportTableNote{Display accounting through \(n=4\): "
                f"{display.required_measurement_count} required measured cells; "
                f"{display.structurally_not_applicable_display_slot_count} "
                r"matrix process/multiplicity positions marked "
                r"\textsc{not applicable}; and "
                f"{display.not_exposed_display_slot_count} "
                r"reference execution fields marked \textsc{not exposed}. "
                r"The latter two categories are intentional table structure, "
                r"not unfilled measurements.}"
            ),
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{@{}L{3.65in}r R{1.45in}@{}}",
            r"\toprule",
            (
                r"\textbf{validation comparison} & \textbf{passed} & "
                r"\textbf{largest relative difference} \\"
            ),
            r"\midrule",
            (
                r"independent original-\AC{} reference records"
                f" & {summary.oracle_count} & "
                r"\textcolor{ReportMuted}{reference} \\"
            ),
            (
                r"\PAC{} versus independent reference"
                f" & {summary.independent_count} & "
                f"{_scientific(summary.independent_maximum_relative_difference)}"
                r" \\"
            ),
            (
                r"compiled/eager versus recurrence"
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
                r"scalar result versus higher precision"
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
    if revision is None:
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
                + r". Independent-reference comparisons use a relative "
                r"tolerance of \(10^{-8}\); cross-mode, resolved-sum, and "
                r"higher-precision comparisons use \(10^{-12}\).}"
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
