# SPDX-License-Identifier: 0BSD
"""Dynamic baseline joins and fixed-block TeX rendering for the report tables."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .cache import empty_measurement
from .campaign_policy import (
    STRICT_POLICY,
    CampaignPolicy,
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
        process_key: str,
        n_final: int,
        workload: Workload,
        *,
        variant: str | None = None,
    ) -> Measurement:
        return self._entries.get(
            (dataset_id, process_key, n_final, workload.value, variant),
            _NA,
        )


@dataclass(frozen=True, slots=True)
class JoinedWorkload:
    workload: Workload
    baseline: Measurement
    candidate: Measurement


@dataclass(frozen=True, slots=True)
class JoinedMatrixCell:
    dataset: MatrixDataset
    process_family: ProcessFamily
    n_final: int
    applicable: bool
    workloads: tuple[JoinedWorkload, ...]


@dataclass(frozen=True, slots=True)
class BestModeWorkload:
    workload: Workload
    baseline: Measurement
    candidate: Measurement
    mode: ExecutionMode | None
    terminal_label: str | None


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

    @staticmethod
    def _baseline_dataset(dataset: MatrixDataset) -> str:
        accuracy = dataset.candidate.accuracy.value
        if dataset.baseline.execution_mode is ExecutionMode.AMPLICOL:
            return f"reference_amplicol_{accuracy}"
        if dataset.baseline.execution_mode is ExecutionMode.RECURRENCE:
            return f"matrix_recurrence_builtin_sm_{accuracy}"
        raise ValueError(
            "matrix baseline must be original AmpliCol or built-in recurrence"
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
            baseline_dataset = self._baseline_dataset(dataset)
            joined = tuple(
                JoinedWorkload(
                    workload,
                    self.index.get(
                        baseline_dataset,
                        family.key,
                        n_final,
                        workload,
                    ),
                    self.index.get(
                        dataset.dataset_id,
                        family.key,
                        n_final,
                        workload,
                    ),
                )
                for workload in workloads
            )
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
        candidate = (
            baseline
            if variant.execution_mode is ExecutionMode.AMPLICOL
            else self.index.get(
                z_dataset_id(model),
                "dd_z_jets",
                n_final,
                workload,
                variant=variant.key,
            )
        )
        return JoinedWorkload(workload, baseline, candidate)

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
            baseline = self.index.get(
                baseline_dataset,
                family.key,
                n_final,
                workload,
            )
            candidates = tuple(
                (
                    mode,
                    self.index.get(
                        f"matrix_{mode.value}_builtin_sm_{accuracy.value}",
                        family.key,
                        n_final,
                        workload,
                    ),
                )
                for mode in _BEST_MODE_ORDER
            )
            eligible = tuple(
                (mode, measurement)
                for mode, measurement in candidates
                if _runtime_value(measurement) is not None
            )
            if eligible:
                winner_mode, winner = min(
                    eligible,
                    key=lambda item: (
                        _runtime_value(item[1]),
                        _BEST_MODE_ORDER.index(item[0]),
                    ),
                )
                terminal_label = None
            else:
                winner_mode, winner = None, _NA
                terminal_label = _best_mode_terminal_label(
                    tuple(measurement for _mode, measurement in candidates)
                )
            joined_workloads.append(
                BestModeWorkload(
                    workload,
                    baseline,
                    winner,
                    winner_mode,
                    terminal_label,
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


def _status(measurement: Measurement) -> str:
    status = str(measurement.get("status", ResultStatus.NOT_AVAILABLE.value))
    policy_label = policy_status_label(measurement)
    if policy_label is not None:
        return r"\matrixstatus{ReportOrange}{" + _tex_escape(policy_label) + "}"
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
        (
            r"\providecommand{\matrixsummarystats}[5]{"
            r"\begingroup\matrixsummaryfont"
            r"\begin{tabular}[t]{@{}r"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}r"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}r"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}r"
            r"@{\hspace{0.014in}\matrixpunct{|}\hspace{0.014in}}r@{}}"
            r"#1&#2&#3&#4&#5\end{tabular}\endgroup}"
        ),
        (
            r"\providecommand{\matrixsummaryworkloads}[2]{"
            r"\begin{tabular}[t]{@{}l@{}}#1\\[-0.10em]#2\end{tabular}}"
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
        else _metric(selected.baseline, "generation_seconds")
    )
    all_flow_baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(all_flow.baseline, "generation_seconds")
    )
    selected_generation_ratio = _ratio_or_absolute(
        selected.candidate,
        selected.baseline,
        "generation_seconds",
        absolute=legacy_baseline_unavailable,
    )
    all_flow_generation_ratio = (
        (
            r"\matrixncabsolute{"
            + _metric(all_flow.candidate, "generation_seconds")
            + "}"
        )
        if (
            view.dataset.candidate.execution_mode is ExecutionMode.RECURRENCE
            and view.dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
            and _ok(all_flow.candidate)
            and (_ok(all_flow.baseline) or legacy_baseline_unavailable)
        )
        else _ratio_or_absolute(
            all_flow.candidate,
            all_flow.baseline,
            "generation_seconds",
            absolute=legacy_baseline_unavailable,
        )
    )
    selected_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(
            selected.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    all_flow_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(
            all_flow.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    selected_ratio = _ratio_pair_or_absolute(
        selected.candidate,
        selected.baseline,
        candidate_mode=view.dataset.candidate.execution_mode,
        absolute=legacy_baseline_unavailable,
    )
    all_flow_ratio = _ratio_pair_or_absolute(
        all_flow.candidate,
        all_flow.baseline,
        candidate_mode=view.dataset.candidate.execution_mode,
        absolute=legacy_baseline_unavailable,
    )
    return _MatrixCellRows(
        generation=(
            selected_baseline_generation,
            r"\matrixpunct{|}",
            all_flow_baseline_generation,
            "",
            selected_generation_ratio,
            r"\matrixpunct{|}",
            "",
            all_flow_generation_ratio,
        ),
        runtime=(
            selected_runtime,
            r"\matrixpunct{|}",
            all_flow_runtime,
            "",
            selected_ratio,
            r"\matrixpunct{|}",
            "",
            all_flow_ratio,
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
    candidate_generation = _ratio_or_absolute(
        joined.candidate,
        joined.baseline,
        "generation_seconds",
        absolute=legacy_baseline_unavailable,
    )
    candidate_runtime = _ratio_pair_or_absolute(
        joined.candidate,
        joined.baseline,
        candidate_mode=view.dataset.candidate.execution_mode,
        absolute=legacy_baseline_unavailable,
    )
    baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(joined.baseline, "generation_seconds")
    )
    baseline_runtime = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(
            joined.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    return _MatrixCellRows(
        generation=(baseline_generation, "", candidate_generation),
        runtime=(baseline_runtime, "", candidate_runtime),
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
        if _ok(item.baseline)
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
        r"\renewcommand{\arraystretch}{1.10}",
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
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _metric_group_cell(
                        _matrix_generation_summary(
                            views_by_n[n_final],
                            dataset,
                        ),
                        accuracy,
                        alignment="l",
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\addlinespace[0.08em]",
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: wall}} & "
                + " & ".join(
                    _metric_group_cell(
                        _matrix_wall_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        alignment="l",
                    )
                    for n_final in multiplicities
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
        " Runtime cells mark the candidate/baseline wall-time multiplier W, "
        "the independent absolute evaluator total T from its dedicated "
        "authenticated record, and the narrower authenticated recurrence core "
        "C. Neither T nor C is derived from wall time or from the other, and C "
        "is never relabeled as evaluator total."
        if dataset.candidate.execution_mode is ExecutionMode.RECURRENCE
        else " Runtime cells mark the candidate/baseline wall-time multiplier W and "
        "the independent absolute evaluator-total value marked T from its "
        "dedicated authenticated record. Compiled and eager cells do not "
        "fabricate a recurrence-core C value, and T is never copied from or "
        "derived from wall time."
    )
    legacy_scope_detail = (
        " Original AmpliCol supports at most three open quark lines; beyond "
        "that scope its declared catalog entry is marked static N/A, requires "
        "no measurement, and valid candidate generation and runtime values are "
        "shown as absolute quantities without a baseline ratio."
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
    return (
        r"\ReportTableNote{Baseline: "
        + _tex_escape(baseline)
        + "; candidate: "
        + _tex_escape(candidate)
        + ". "
        + _tex_escape(detail)
        + _tex_escape(clock_detail)
        + _tex_escape(legacy_scope_detail)
        + _tex_escape(summary_detail)
        + " "
        + _tex_escape(
            "Not applicable marks a process/multiplicity combination outside "
            "the process-family definition. Not exposed means that a successful "
            "wall-time measurement has no dedicated authenticated evidence for "
            "the separately labeled T or C clock. Older entries remain marked "
            "not exposed; neither label denotes an unfilled measurement."
        )
        + "}"
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
    labels: Sequence[str | None],
) -> str | None:
    """Return one concise label only when every represented outcome is terminal."""

    if not labels:
        return None
    canonical: list[str] = []
    for label in labels:
        if label is None:
            return None
        if label.startswith("dependency"):
            normalized = ("dependency",)
        else:
            normalized = tuple(
                item.strip()
                for item in label.replace("|", "/").split("/")
                if item.strip()
            )
            if not normalized or any(
                item not in {">2h", "dependency"}
                and not (
                    item.startswith(">")
                    and item.endswith("GB")
                    and item[1:-2].isdigit()
                )
                for item in normalized
            ):
                return None
        for item in normalized:
            if item not in canonical:
                canonical.append(item)

    def order(item: str) -> tuple[int, int]:
        if item == ">2h":
            return (0, 0)
        if item.endswith("GB"):
            return (1, int(item[1:-2]))
        return (2, 0)

    return " | ".join(sorted(canonical, key=order)) or None


def _best_mode_terminal_label(
    measurements: Sequence[Measurement],
) -> str | None:
    """Return one fail-closed summary of an all-terminal candidate set."""

    return _canonical_best_mode_terminal_label(
        tuple(policy_status_label(measurement) for measurement in measurements)
    )


def _best_mode_terminal_status(joined: BestModeWorkload) -> str:
    if joined.terminal_label is None:
        return _status(joined.candidate)
    return r"\matrixstatus{ReportOrange}{" + _tex_escape(joined.terminal_label) + "}"


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
    if baseline_static_na or not comparable:
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
    baseline_static_na: bool = False,
) -> _BestModeComparisonLayout:
    if joined.mode is None:
        status = _best_mode_terminal_status(joined)
        return _BestModeComparisonLayout(status, "", status)
    if not _ok(joined.candidate):
        status = _status(joined.candidate)
        return _BestModeComparisonLayout(status, "", status)
    if baseline_static_na:
        wall = _best_mode_metric(
            joined.candidate,
            "wall_seconds_per_point",
            microseconds=True,
        )
        return _BestModeComparisonLayout(
            wall,
            wall,
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
    secondary_ratio = None
    if (
        unavailable_execution_timing_record(
            joined.candidate,
            "execution_seconds_per_point",
        )
        is None
        and unavailable_execution_timing_record(
            joined.baseline,
            "execution_seconds_per_point",
        )
        is None
    ):
        secondary_ratio = _ratio_value(
            joined.candidate,
            joined.baseline,
            "execution_seconds_per_point",
        )
    if secondary_ratio is None:
        inline = (
            rf"\bestmodewallratio{{{wall_color}}}"
            rf"{{{_best_mode_value(wall_ratio)}}}"
        )
        return _BestModeComparisonLayout(
            inline,
            r"\bestmodeopenprefix",
            (
                rf"\bestmodeprimaryratio{{{wall_color}}}"
                rf"{{{_best_mode_value(wall_ratio)}}}"
            ),
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
) -> str | None:
    if joined.mode is None:
        return joined.terminal_label
    if not _ok(joined.baseline):
        return policy_status_label(joined.baseline)
    if not _ok(joined.candidate):
        return policy_status_label(joined.candidate)
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
    *,
    show_mode_mix: bool = False,
) -> str:
    joined = tuple(
        next(item for item in view.workloads if item.workload is workload)
        for view in views
        if view.applicable
    )
    valid = tuple(
        item
        for item in joined
        if item.mode is not None
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
            return r"\matrixstatus{ReportOrange}{" + _tex_escape(terminal_label) + "}"
        return r"\matrixna{ReportMuted}"
    statistics = _ratio_statistics_tex(
        tuple(
            (float(item.baseline[field]), float(item.candidate[field]))
            for item in valid
        )
    )
    if show_mode_mix:
        counts = {
            mode: sum(item.mode is mode for item in valid) for mode in _BEST_MODE_ORDER
        }
        return (
            r"\bestmodesummarystats{"
            + statistics
            + r"}{\bestmodemix{"
            + "|".join(
                f"{_BEST_MODE_CODES[mode]}:{counts[mode]}" for mode in _BEST_MODE_ORDER
            )
            + "}}"
        )
    return statistics


def _best_mode_generation_summary(
    views: Sequence[BestModeMatrixCell],
    accuracy: Accuracy,
) -> str:
    if accuracy is Accuracy.LC:
        return _best_mode_summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "generation_seconds",
            show_mode_mix=True,
        )
    return _best_mode_summary_pair(
        views,
        Workload.CONTRACTED,
        "generation_seconds",
        show_mode_mix=True,
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
        r"\renewcommand{\arraystretch}{1.10}",
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
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _metric_group_cell(
                        _best_mode_generation_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        alignment="l",
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\addlinespace[0.08em]",
            (
                r"\multicolumn{3}{@{}l}{\textbf{summary: wall}} & "
                + " & ".join(
                    _metric_group_cell(
                        _best_mode_wall_summary(
                            views_by_n[n_final],
                            accuracy,
                        ),
                        accuracy,
                        alignment="l",
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\endgroup",
            (
                r"\ReportTableNote{The candidate is selected independently in "
                r"each cell and workload by the smallest validated wall time. "
                r"The upper row is generation in seconds and the lower row is "
                r"runtime in \(\mu\mathrm{s}/\mathrm{pt}\). In LC cells, each "
                r"pair is non-union-flow \texttt{|} union-flow. Generation "
                r"multipliers are coloured directly. Each runtime workload "
                r"uses \texttt{([xS]xW)}: the muted bracket is the separately "
                r"measured, ratio-eligible execution attribution when exposed, "
                r"and the coloured value is the candidate/AmpliCol wall-time "
                r"multiplier used as the figure of merit. Independent absolute "
                r"T and recurrence-core C clocks remain in the detailed "
                r"evidence and are never fabricated from wall time. Every "
                r"displayed number uses exactly three significant digits. "
                r"Original-AmpliCol "
                r"rows beyond its three-open-quark-line scope are catalog "
                r"static N/A entries; their candidates are shown absolutely and "
                r"excluded from ratio summaries."
                + boundary_note
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
                r"eager-DAG JIT O2, respectively.}"
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
            r"\providecommand{\bestmodecode}[1]{"
            r"\textcolor{ReportMuted}{\texttt{(#1)}}}"
        ),
        (
            r"\providecommand{\bestmodecodeprefix}[1]{"
            r"\bestmodecode{#1}\hspace{0.025in}}"
        ),
        (
            r"\providecommand{\bestmodeabsoluteprefix}[1]{"
            r"#1\hspace{0.04in}}"
        ),
        (
            r"\providecommand{\bestmodeabsolutechoice}[2]{"
            r"#1\hspace{0.025in}\bestmodecode{#2}}"
        ),
        (
            r"\providecommand{\bestmodemix}[1]{"
            r"\begingroup\matrixsummaryfont"
            r"\textcolor{ReportBlue}{\texttt{[#1]}}\endgroup}"
        ),
        (
            r"\providecommand{\bestmodesummarystats}[2]{"
            r"\begin{tabular}[t]{@{}l@{}}"
            r"#1\\[-0.12em]#2\end{tabular}}"
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
    if not comparable:
        return rf"\matrixncabsolute{{{absolute}}}"
    return absolute + r"\," + _ratio(measurement, joined.baseline, field)


def _z_evaluator_total(
    joined: JoinedWorkload,
    *,
    reference: bool,
    recurrence: bool = False,
    static_na: bool = False,
) -> str:
    if static_na:
        return _static_na()
    measurement = joined.baseline if reference else joined.candidate
    if reference:
        return _not_exposed()
    if not _ok(measurement):
        return _status(measurement)
    total_text = _evaluator_total_clock(measurement)
    if not recurrence:
        return total_text
    return (
        r"\matrixzrecurrenceclocks{"
        + total_text
        + "}{"
        + _recurrence_core_clock(measurement)
        + "}"
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
            r"\shortstack{\textbf{eval total T}\\"
            r"\textbf{rec. core C}\\"
            r"\textbf{[\(\mu\mathrm{s}/\mathrm{pt}\)]}} & "
            r"\textbf{gen [s]} & "
            r"\textbf{wall [\(\mu\mathrm{s}/\mathrm{pt}\)]} & "
            r"\shortstack{\textbf{eval total T}\\"
            r"\textbf{rec. core C}\\"
            r"\textbf{[\(\mu\mathrm{s}/\mathrm{pt}\)]}} \\"
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
                    recurrence=(variant.execution_mode is ExecutionMode.RECURRENCE),
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
                    recurrence=(variant.execution_mode is ExecutionMode.RECURRENCE),
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
                r"\ReportTableNote{Original AmpliCol is the denominator. "
                r"Each pyAmpliCol row reports separate topology-replay and "
                r"all-flow-union generation and runtime measurements. "
                r"Parenthesized values are candidate/reference ratios. "
                r"All-flow generation shows the absolute pyAmpliCol value "
                r"without a ratio because its setup boundary differs from the "
                r"reference. The wall time is the common runtime observable. "
                r"Not exposed means that a successful wall measurement has no "
                r"dedicated authenticated evaluator-total timing. Authenticated "
                r"accumulated warmed evaluator totals are marked T; T is not "
                r"an attribution ratio. Recurrence rows also retain the "
                r"independently measured narrower recurrence core marked C. "
                r"Neither T nor C is derived from wall time or from the other, "
                r"and C is never relabeled as evaluator total. Older entries "
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
                                else _metric(
                                    joined.baseline,
                                    "generation_seconds",
                                )
                            ),
                            (
                                _static_na()
                                if baseline_static_na
                                else _metric(
                                    joined.baseline,
                                    "wall_seconds_per_point",
                                    microseconds=True,
                                )
                            ),
                        ),
                    )
                    candidate_generation = (
                        _ratio_or_absolute(
                            joined.candidate,
                            joined.baseline,
                            "generation_seconds",
                            absolute=baseline_static_na,
                        )
                        if baseline_static_na
                        else _metric(joined.candidate, "generation_seconds")
                        if (
                            joined.workload is Workload.ALL_FLOW
                            and dataset.candidate.execution_mode
                            is ExecutionMode.RECURRENCE
                            and dataset.baseline.execution_mode
                            is ExecutionMode.AMPLICOL
                        )
                        else _ratio(
                            joined.candidate,
                            joined.baseline,
                            "generation_seconds",
                        )
                    )
                    record(
                        candidate,
                        (
                            f"{dataset.table_name}/n{n_final}/{family.key}/"
                            f"{joined.workload.value}/candidate"
                        ),
                        (
                            candidate_generation,
                            _ratio_pair_or_absolute(
                                joined.candidate,
                                joined.baseline,
                                candidate_mode=(dataset.candidate.execution_mode),
                                absolute=baseline_static_na,
                            ),
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
                                    recurrence=(
                                        variant.execution_mode
                                        is ExecutionMode.RECURRENCE
                                    ),
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
