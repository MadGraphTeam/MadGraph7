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
    ExecutionMode.RECURRENCE: "A",
    ExecutionMode.COMPILED: "B",
    ExecutionMode.EAGER: "C",
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
            != (
                self.required_measurement_count
                + self.catalog_static_na_cell_count
            )
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
            "declared_measurement_cell_count": (
                self.declared_measurement_cell_count
            ),
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


def _time(value: object, *, microseconds: bool = False) -> str:
    if value is None:
        return r"\matrixna{ReportMuted}"
    number = float(value)
    if not math.isfinite(number):
        return r"\matrixna{ReportMuted}"
    if microseconds:
        number *= 1.0e6
    return rf"\texttt{{{_compact(number)}}}"


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
        return (
            r"\matrixstatus{ReportOrange}{"
            + _tex_escape(policy_label)
            + "}"
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
            if (
                isinstance(timing, Mapping)
                and timing.get("ratio_eligible") is not True
            ):
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


def _ratio_pair(candidate: Measurement, baseline: Measurement) -> str:
    wall = _ratio_value(candidate, baseline, "wall_seconds_per_point")
    execution = _ratio_value(candidate, baseline, "execution_seconds_per_point")
    if not _ok(candidate):
        return _status(candidate)
    execution_not_exposed = any(
        unavailable_execution_timing_record(
            measurement,
            "execution_seconds_per_point",
        )
        is not None
        for measurement in (candidate, baseline)
    )

    def field(value: float | None, name: str) -> tuple[str, str]:
        if value is None:
            return "ReportMuted", "N/A"
        color = (
            "ReportGreen"
            if value < 1.0
            else "ReportOrange"
            if value < 2.0
            else "ReportRed"
        )
        return color, f"x{_compact(value)}"

    wall_color, wall_text = field(wall, "wall_seconds_per_point")
    if execution_not_exposed:
        total_record = evaluator_total_timing_record(candidate)
        if total_record is not None:
            total_time = _time(
                total_record["raw_seconds_per_point"],
                microseconds=True,
            )
            return (
                rf"\matrixratiopairtotalevaluator{{{wall_color}}}"
                f"{{{wall_text}}}{{{total_time}}}"
            )
        return (
            rf"\matrixratiopairnotexposed{{{wall_color}}}"
            f"{{{wall_text}}}"
        )
    execution_color, execution_text = field(
        execution,
        "execution_seconds_per_point",
    )
    return (
        rf"\matrixratiopair{{{wall_color}}}{{{wall_text}}}"
        rf"{{{execution_color}}}{{{execution_text}}}"
    )


def _ratio_pair_or_absolute(
    candidate: Measurement,
    baseline: Measurement,
    *,
    absolute: bool,
) -> str:
    if absolute and _ok(candidate):
        wall = _metric(candidate, "wall_seconds_per_point", microseconds=True)
        execution = _metric(
            candidate,
            "execution_seconds_per_point",
            microseconds=True,
        )
        return (
            r"\matrixncabsolute{"
            rf"\matrixpair{{{wall}}}{{{execution}}}"
            "}"
        )
    return _ratio_pair(candidate, baseline)


def _matrix_macros() -> list[str]:
    return [
        r"\providecommand{\matrixentryfont}{\fontsize{6.8pt}{7.8pt}\selectfont}",
        r"\providecommand{\matrixentryfontlc}{\fontsize{6.5pt}{7.5pt}\selectfont}",
        r"\providecommand{\matrixpunct}[1]{\textcolor{black}{\texttt{#1}}}",
        (
            r"\providecommand{\matrixratio}[2]{\matrixpunct{(}"
            r"\textcolor{#1}{\texttt{x#2}}\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\matrixratiopair}[4]{\matrixpunct{(}"
            r"\textcolor{#1}{\texttt{#2}}\matrixpunct{|}"
            r"\textcolor{#3}{\texttt{#4}}\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\matrixratiopairnotexposed}[2]{\matrixpunct{(}"
            r"\textcolor{#1}{\texttt{#2}}\matrixpunct{|}"
            r"\matrixnotexposed{ReportMuted}\matrixpunct{)}}"
        ),
        (
            r"\providecommand{\matrixratiopairtotalevaluator}[3]{"
            r"\matrixpunct{(}\textcolor{#1}{\texttt{#2}}"
            r"\matrixpunct{|}\matrixtotalevaluator{#3}\matrixpunct{)}}"
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
        r"\providecommand{\matrixbelow}[1]{\textcolor{ReportMuted}{\texttt{<#1}}}",
        (
            r"\providecommand{\matrixnaratio}[1]{\matrixpunct{(}"
            r"\matrixna{#1}\matrixpunct{)}}"
        ),
        r"\providecommand{\matrixstatus}[2]{\textcolor{#1}{\textsc{#2}}}",
        (
            r"\providecommand{\matrixncabsolute}[1]{"
            r"\textcolor{ReportBlue}{\texttt{C }}#1"
            r"\matrixpunct{; }\matrixstatus{ReportMuted}{n.c.}}"
        ),
        (
            r"\providecommand{\matrixcelllc}[6]{"
            r"\begingroup\matrixentryfontlc"
            r"\scalebox{0.94}{"
            r"\begin{tabular}[t]{"
            r"@{}l@{\hspace{0.03in}}l@{\hspace{0.03in}}l@{}}"
            r"#1&#2&#3\\#4&#5&#6"
            r"\end{tabular}}\endgroup}"
        ),
        (
            r"\providecommand{\matrixcellcontracted}[4]{"
            r"\begingroup\matrixentryfont"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}"
            r"#1&#2\\#3&#4"
            r"\end{tabular}\endgroup}"
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
) -> str:
    selected, all_flow = view.workloads
    legacy_baseline_unavailable = _legacy_baseline_unavailable(
        view,
        catalog=catalog,
    )
    selected_baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(selected.baseline, 'generation_seconds')
    )
    all_flow_baseline_generation = (
        _static_na()
        if legacy_baseline_unavailable
        else _metric(all_flow.baseline, 'generation_seconds')
    )
    baseline_generation = (
        rf"\matrixpair{{{selected_baseline_generation}}}"
        rf"{{{all_flow_baseline_generation}}}"
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
            and (
                _ok(all_flow.baseline)
                or legacy_baseline_unavailable
            )
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
    baseline_runtime = (
        rf"\matrixpair{{{selected_runtime}}}"
        rf"{{{all_flow_runtime}}}"
    )
    selected_ratio = _ratio_pair_or_absolute(
        selected.candidate,
        selected.baseline,
        absolute=legacy_baseline_unavailable,
    )
    all_flow_ratio = _ratio_pair_or_absolute(
        all_flow.candidate,
        all_flow.baseline,
        absolute=legacy_baseline_unavailable,
    )
    return (
        r"\matrixcelllc"
        f"{{{baseline_generation}}}"
        f"{{{selected_generation_ratio}}}"
        f"{{{all_flow_generation_ratio}}}"
        f"{{{baseline_runtime}}}"
        f"{{{selected_ratio}}}"
        f"{{{all_flow_ratio}}}"
    )


def _contracted_cell(
    view: JoinedMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> str:
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
    return (
        r"\matrixcellcontracted"
        f"{{{baseline_generation}}}"
        f"{{{candidate_generation}}}"
        f"{{{baseline_runtime}}}"
        f"{{{candidate_runtime}}}"
    )


def _summary_pair(
    views: Sequence[JoinedMatrixCell],
    workload: Workload,
    field: str,
    *,
    microseconds: bool = False,
    comparable: bool = True,
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
    baseline_sum = sum(float(item.baseline[field]) for item in valid)
    candidate_sum = sum(float(item.candidate[field]) for item in valid)
    baseline_mean = baseline_sum / len(valid)
    baseline_text = _time(baseline_mean, microseconds=microseconds)
    if not comparable:
        candidate_mean = candidate_sum / len(valid)
        candidate_text = _time(candidate_mean, microseconds=microseconds)
        return (
            rf"\matrixsummarypair{{{baseline_text}}}"
            rf"{{\matrixncabsolute{{{candidate_text}}}}}"
        )
    ratio = candidate_sum / baseline_sum if baseline_sum > 0.0 else math.nan
    if not math.isfinite(ratio):
        ratio_text = r"\matrixnaratio{ReportMuted}"
    else:
        color = (
            "ReportGreen"
            if ratio < 1.0
            else "ReportOrange"
            if ratio < 2.0
            else "ReportRed"
        )
        ratio_text = rf"\matrixratio{{{color}}}{{{_compact(ratio)}}}"
    return rf"\matrixsummarypair{{{baseline_text}}}{{{ratio_text}}}"


def _matrix_block(
    adapter: BaselineCandidateAdapter,
    dataset: MatrixDataset,
    multiplicities: tuple[int, ...],
    *,
    block_index: int,
    block_count: int,
) -> list[str]:
    # Three-column LC blocks have the widest nested entries.  Leave a little
    # landscape-page breathing room so populated compiled rows cannot extend
    # into the right crop boundary.
    value_width = {2: "3.50in", 3: "2.42in"}.get(len(multiplicities))
    if value_width is None:
        raise ValueError(
            "matrix blocks must contain two or three multiplicities"
        )
    column_spec = (
        r"@{}r@{\hspace{0.04in}}L{1.65in}"
        + "".join(
            rf"@{{\hspace{{0.06in}}}}L{{{value_width}}}"
            for _ in multiplicities
        )
        + r"@{}"
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
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            r"\textbf{ID} & \textbf{base process} & "
            + " & ".join(rf"\textbf{{n={n_final}}}" for n_final in multiplicities)
            + r" \\"
        ),
        r"\specialrule{0.85pt}{0pt}{0pt}",
    ]
    views_by_n: dict[int, list[JoinedMatrixCell]] = {
        n_final: [] for n_final in multiplicities
    }
    for row_index, family in enumerate(adapter.catalog.process_families):
        row = [rf"\texttt{{{family.identifier}}}", family.label_tex]
        for n_final in multiplicities:
            view = adapter.matrix_cell(dataset, family, n_final)
            views_by_n[n_final].append(view)
            if not view.applicable:
                row.append(_not_applicable())
            elif dataset.candidate.accuracy is Accuracy.LC:
                row.append(_lc_cell(view, catalog=adapter.catalog))
            else:
                row.append(_contracted_cell(view, catalog=adapter.catalog))
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace[0.05em]")
    lines.extend(
        [
            r"\specialrule{1.05pt}{0.22em}{0.18em}",
            (
                r"\multicolumn{2}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _matrix_generation_summary(
                        views_by_n[n_final],
                        dataset,
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\addlinespace[0.08em]",
            (
                r"\multicolumn{2}{@{}l}{\textbf{summary: wall}} & "
                + " & ".join(
                    _matrix_wall_summary(
                        views_by_n[n_final],
                        dataset.candidate.accuracy,
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
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
        selected = _summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "generation_seconds",
        )
        all_flow = _summary_pair(
            views,
            Workload.ALL_FLOW,
            "generation_seconds",
            comparable=not (
                dataset.candidate.execution_mode is ExecutionMode.RECURRENCE
                and dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
            ),
        )
        return rf"\matrixpair{{{selected}}}{{{all_flow}}}"
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
            microseconds=True,
        )
        all_flow = _summary_pair(
            views,
            Workload.ALL_FLOW,
            "wall_seconds_per_point",
            microseconds=True,
        )
        return rf"\matrixpair{{{selected}}}{{{all_flow}}}"
    return _summary_pair(
        views,
        Workload.CONTRACTED,
        "wall_seconds_per_point",
        microseconds=True,
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
                "the absolute pyAmpliCol process-generation time marked n.c. "
                "(not comparable); no generation ratio is formed. Runtime "
                "comparisons use native wall time; "
                "a separate execution-attribution ratio appears only when both "
                "measurements expose one."
            )
        else:
            detail = (
                "Each cell shows the topology-replay/all-flow-union baseline "
                "generation times and wall times, followed by layout-matched "
                "candidate/baseline generation and wall-time ratios. A separate "
                "execution-attribution ratio appears only when both "
                "measurements expose one."
            )
    else:
        detail = (
            "Each cell shows the baseline generation and wall time, followed by "
            "candidate/baseline generation and wall-time ratios. A separate "
            "execution-attribution ratio appears only when both measurements "
            "expose one."
        )
    legacy_scope_detail = (
        " Original AmpliCol supports at most three open quark lines; beyond "
        "that scope its declared catalog entry is marked static N/A, requires "
        "no measurement, and valid candidate generation and runtime values are "
        "shown as absolute n.c. quantities. Here n.c. means not comparable."
        if dataset.baseline.execution_mode is ExecutionMode.AMPLICOL
        else ""
    )
    return (
        r"\ReportTableNote{Baseline: "
        + _tex_escape(baseline)
        + "; candidate: "
        + _tex_escape(candidate)
        + ". "
        + _tex_escape(detail)
        + _tex_escape(legacy_scope_detail)
        + " "
        + _tex_escape(
            "Not applicable marks a process/multiplicity combination outside "
            "the process-family definition. Not exposed means that a successful "
            "wall-time measurement has no separately reported execution "
            "attribution; for compiled and eager rows, the wall value remains "
            "the authenticated warmed total-evaluator boundary. Future compiled "
            "and eager entries additionally show an accumulated absolute "
            "evaluator-total value marked T; it is not an execution-attribution "
            "ratio. Older entries remain marked not exposed; neither label "
            "denotes an unfilled measurement."
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
            r"\begin{tabular}[t]{@{}l@{\hspace{0.025in}/\hspace{0.025in}}l@{}}"
            r"#1&#2\end{tabular}}"
        ),
        (
            r"\providecommand{\matrixsummarypair}[2]{"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}#1&#2\end{tabular}}"
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


def _best_mode_code(mode: ExecutionMode | None) -> str:
    if mode is None:
        return ""
    return rf"\bestmodecode{{{_BEST_MODE_CODES[mode]}}}"


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
            normalized = tuple(label.split("/"))
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

    return "/".join(sorted(canonical, key=order)) or None


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
    return (
        r"\matrixstatus{ReportOrange}{"
        + _tex_escape(joined.terminal_label)
        + "}"
    )


def _best_mode_ratio(
    joined: BestModeWorkload,
    field: str,
    *,
    baseline_static_na: bool = False,
) -> str:
    if joined.mode is None:
        return _best_mode_terminal_status(joined)
    if baseline_static_na:
        return _ratio_or_absolute(
            joined.candidate,
            joined.baseline,
            field,
            absolute=True,
        ) + _best_mode_code(joined.mode)
    if not _ok(joined.baseline):
        return _status(joined.baseline)
    return _ratio(joined.candidate, joined.baseline, field) + _best_mode_code(
        joined.mode
    )


def _best_mode_runtime_ratio(
    joined: BestModeWorkload,
    *,
    baseline_static_na: bool = False,
) -> str:
    if joined.mode is None:
        return _best_mode_terminal_status(joined)
    if baseline_static_na:
        return _ratio_or_absolute(
            joined.candidate,
            joined.baseline,
            "wall_seconds_per_point",
            absolute=True,
            microseconds=True,
        )
    if not _ok(joined.baseline):
        return _status(joined.baseline)
    return _ratio(
        joined.candidate,
        joined.baseline,
        "wall_seconds_per_point",
    )


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
) -> str:
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
        else _metric(selected.baseline, "generation_seconds")
    )
    all_flow_baseline_generation = (
        _static_na()
        if baseline_static_na
        else _metric(all_flow.baseline, "generation_seconds")
    )
    baseline_generation = (
        rf"\matrixpair{{{selected_baseline_generation}}}"
        rf"{{{all_flow_baseline_generation}}}"
    )
    selected_baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _metric(
            selected.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    all_flow_baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _metric(
            all_flow.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    baseline_runtime = (
        rf"\matrixpair{{{selected_baseline_runtime}}}"
        rf"{{{all_flow_baseline_runtime}}}"
    )
    selected_generation_ratio = _best_mode_ratio(
        selected,
        "generation_seconds",
        baseline_static_na=baseline_static_na,
    )
    all_flow_generation_ratio = _best_mode_ratio(
        all_flow,
        "generation_seconds",
        baseline_static_na=baseline_static_na,
    )
    selected_runtime_ratio = _best_mode_runtime_ratio(
        selected,
        baseline_static_na=baseline_static_na,
    )
    all_flow_runtime_ratio = _best_mode_runtime_ratio(
        all_flow,
        baseline_static_na=baseline_static_na,
    )
    return (
        r"\matrixcelllc"
        f"{{{baseline_generation}}}"
        f"{{{selected_generation_ratio}}}"
        f"{{{all_flow_generation_ratio}}}"
        f"{{{baseline_runtime}}}"
        f"{{{selected_runtime_ratio}}}"
        f"{{{all_flow_runtime_ratio}}}"
    )


def _best_mode_contracted_cell(
    view: BestModeMatrixCell,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> str:
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
        else _metric(joined.baseline, "generation_seconds")
    )
    baseline_runtime = (
        _static_na()
        if baseline_static_na
        else _metric(
            joined.baseline,
            "wall_seconds_per_point",
            microseconds=True,
        )
    )
    generation_ratio = _best_mode_ratio(
        joined,
        "generation_seconds",
        baseline_static_na=baseline_static_na,
    )
    runtime_ratio = _best_mode_runtime_ratio(
        joined,
        baseline_static_na=baseline_static_na,
    )
    return (
        r"\matrixcellcontracted"
        f"{{{baseline_generation}}}"
        f"{{{generation_ratio}}}"
        f"{{{baseline_runtime}}}"
        f"{{{runtime_ratio}}}"
    )


def _best_mode_summary_pair(
    views: Sequence[BestModeMatrixCell],
    workload: Workload,
    field: str,
    *,
    microseconds: bool = False,
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
    )
    if not valid:
        if field == "execution_seconds_per_point" and any(
            unavailable_execution_timing_record(measurement, field) is not None
            for item in joined
            for measurement in (item.baseline, item.candidate)
        ):
            return _not_exposed()
        terminal_label = _canonical_best_mode_terminal_label(
            tuple(
                _best_mode_summary_terminal_label(item)
                for item in joined
            )
        )
        if terminal_label is not None:
            return (
                r"\matrixstatus{ReportOrange}{"
                + _tex_escape(terminal_label)
                + "}"
            )
        return r"\matrixna{ReportMuted}"
    baseline_sum = math.fsum(float(item.baseline[field]) for item in valid)
    candidate_sum = math.fsum(float(item.candidate[field]) for item in valid)
    baseline_mean = baseline_sum / len(valid)
    ratio = candidate_sum / baseline_sum if baseline_sum > 0.0 else math.nan
    baseline_text = _time(baseline_mean, microseconds=microseconds)
    ratio_text = (
        r"\matrixnaratio{ReportMuted}"
        if not math.isfinite(ratio)
        else _ratio(
            {
                "status": ResultStatus.OK.value,
                field: candidate_sum,
            },
            {
                "status": ResultStatus.OK.value,
                field: baseline_sum,
            },
            field,
        )
    )
    if show_mode_mix:
        counts = {
            mode: sum(item.mode is mode for item in valid)
            for mode in _BEST_MODE_ORDER
        }
        return (
            r"\bestmodesummarypair{"
            + baseline_text
            + "}{"
            + ratio_text
            + r"}{\bestmodemix{"
            + "/".join(
                f"{_BEST_MODE_CODES[mode]}:{counts[mode]}"
                for mode in _BEST_MODE_ORDER
            )
            + "}}"
        )
    return rf"\matrixsummarypair{{{baseline_text}}}{{{ratio_text}}}"


def _best_mode_generation_summary(
    views: Sequence[BestModeMatrixCell],
    accuracy: Accuracy,
) -> str:
    if accuracy is Accuracy.LC:
        selected = _best_mode_summary_pair(
            views,
            Workload.SELECTED_FLOW,
            "generation_seconds",
            show_mode_mix=True,
        )
        all_flow = _best_mode_summary_pair(
            views,
            Workload.ALL_FLOW,
            "generation_seconds",
            show_mode_mix=True,
        )
        return (
            rf"\matrixpair{{{selected}}}{{{all_flow}}}"
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
            microseconds=True,
        )
        all_flow = _best_mode_summary_pair(
            views,
            Workload.ALL_FLOW,
            "wall_seconds_per_point",
            microseconds=True,
        )
        return (
            rf"\matrixpair{{{selected}}}{{{all_flow}}}"
        )
    return _best_mode_summary_pair(
        views,
        Workload.CONTRACTED,
        "wall_seconds_per_point",
        microseconds=True,
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
    value_width = {2: "3.50in", 3: "2.42in"}.get(len(multiplicities))
    if value_width is None:
        raise ValueError(
            "matrix blocks must contain two or three multiplicities"
        )
    column_spec = (
        r"@{}r@{\hspace{0.04in}}L{1.65in}"
        + "".join(
            rf"@{{\hspace{{0.06in}}}}L{{{value_width}}}"
            for _ in multiplicities
        )
        + r"@{}"
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
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            r"\textbf{ID} & \textbf{base process} & "
            + " & ".join(rf"\textbf{{n={n_final}}}" for n_final in multiplicities)
            + r" \\"
        ),
        r"\specialrule{0.85pt}{0pt}{0pt}",
    ]
    views_by_n: dict[int, list[BestModeMatrixCell]] = {
        n_final: [] for n_final in multiplicities
    }
    for row_index, family in enumerate(adapter.catalog.process_families):
        row = [rf"\texttt{{{family.identifier}}}", family.label_tex]
        for n_final in multiplicities:
            view = adapter.best_mode_cell(accuracy, family, n_final)
            views_by_n[n_final].append(view)
            if not view.applicable:
                row.append(_not_applicable())
            elif accuracy is Accuracy.LC:
                row.append(_best_mode_lc_cell(view, catalog=adapter.catalog))
            else:
                row.append(
                    _best_mode_contracted_cell(view, catalog=adapter.catalog)
                )
        if row_index % 2 == 0:
            lines.append(r"\rowcolor{refblue}")
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace[0.05em]")
    boundary_note = (
        r" For the LC all-flow workload, the generation multiplier compares "
        r"pyAmpliCol process generation with AmpliCol direct-evaluation setup; "
        r"it is a setup-cost indicator across different boundaries, not a "
        r"like-for-like compiler benchmark."
        if accuracy is Accuracy.LC
        else ""
    )
    lines.extend(
        [
            r"\specialrule{1.05pt}{0.22em}{0.18em}",
            (
                r"\multicolumn{2}{@{}l}{\textbf{summary: generation}} & "
                + " & ".join(
                    _best_mode_generation_summary(
                        views_by_n[n_final],
                        accuracy,
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\addlinespace[0.08em]",
            (
                r"\multicolumn{2}{@{}l}{\textbf{summary: wall}} & "
                + " & ".join(
                    _best_mode_wall_summary(
                        views_by_n[n_final],
                        accuracy,
                    )
                    for n_final in multiplicities
                )
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            (
                r"\ReportTableNote{The candidate is selected independently in "
                r"each cell and workload by the smallest validated wall time. "
                r"Generation multipliers identify that runtime winner: "
                r"\texttt{(A)} recurrence JIT O2, \texttt{(B)} compiled JIT O3, "
                r"and \texttt{(C)} eager-DAG JIT O2. Runtime entries are "
                r"winner/AmpliCol wall-time multipliers. Original-AmpliCol "
                r"rows beyond its three-open-quark-line scope are catalog "
                r"static N/A entries; their candidate values are absolute "
                r"n.c. quantities and are excluded from ratio summaries."
                + boundary_note
                + r" Summary mode counts use the order A/B/C.}"
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
            r"\begin{tabular}[t]{@{}l@{\hspace{0.025in}/\hspace{0.025in}}l@{}}"
            r"#1&#2\end{tabular}}"
        ),
        (
            r"\providecommand{\matrixsummarypair}[2]{"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}#1&#2\end{tabular}}"
        ),
        (
            r"\providecommand{\bestmodecode}[1]{"
            r"\hspace{0.025in}\textcolor{ReportBlue}{\texttt{(#1)}}}"
        ),
        (
            r"\providecommand{\bestmodemix}[1]{"
            r"\textcolor{ReportBlue}{\texttt{[#1]}}}"
        ),
        (
            r"\providecommand{\bestmodesummarypair}[3]{"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}"
            r"#1&#2\\[-0.16em]\multicolumn{2}{@{}l@{}}{#3}\end{tabular}}"
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
    return r"\matrixtotalevaluator{" + _time(total, microseconds=True) + "}"


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
            r"& & \textbf{gen [s]} & \textbf{wall [us/pt]} & "
            r"\textbf{eval total [us/pt]} & \textbf{gen [s]} & "
            r"\textbf{wall [us/pt]} & \textbf{eval total [us/pt]} \\"
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
                r"\ReportTableNote{Original AmpliCol is the denominator. "
                r"Each pyAmpliCol row reports separate topology-replay and "
                r"all-flow-union generation and runtime measurements. "
                r"Parenthesized values are candidate/reference ratios. "
                r"All-flow generation shows the absolute pyAmpliCol value "
                r"marked n.c.; n.c. means not comparable because its setup "
                r"boundary differs from the "
                r"reference. The wall time is the common runtime observable. "
                r"Not exposed means that a successful wall measurement has no "
                r"separately reported evaluator-total timing. Every "
                r"pyAmpliCol row shows the authenticated accumulated warmed "
                r"evaluator total marked T; T is not an attribution ratio. "
                r"Recurrence core/execution attribution remains a separate "
                r"metric in raw evidence and is not relabeled as evaluator "
                r"total. Older entries without authenticated total evidence "
                r"remain marked not exposed; it is not a missing measurement. "
                + _tex_escape(
                    STATIC_NA_NATIVE_BACKEND_GENERATION_CAP_N6_DESCRIPTION
                )
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
        microseconds=field
        in {"wall_seconds_per_point", "execution_seconds_per_point"},
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
        + "".join(
            rf"L{{{value_width}}}" for _ in dataset.multiplicities
        )
        + r"@{}"
    )
    rows = (
        ("generation [s]", "generation_seconds"),
        (r"wall [$\mu$s/pt]", "wall_seconds_per_point"),
        (r"execution [$\mu$s/pt]", "execution_seconds_per_point"),
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
                r"Execution is a separately measured attribution when exposed; "
                r"future compiled and eager entries show the accumulated "
                r"absolute evaluator total marked T when narrower attribution "
                r"is unavailable. "
                r"\textsc{not exposed} denotes a successful wall measurement, "
                r"and older entries without T remain valid; neither denotes a "
                r"missing result.}"
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
    return (
        r"\matrixna{" in value
        or r"\matrixnaratio{" in value
        or "{N/A}" in value
    )


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
        cell
        for cell in declared_cells
        if catalog.static_na_reason(cell) is not None
    )
    required_cells = tuple(
        cell
        for cell in declared_cells
        if catalog.static_na_reason(cell) is None
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
                    baseline_static_na = (
                        catalog.static_na_reason(baseline) is not None
                    )
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
                                else _metric(
                                    joined.baseline,
                                    "generation_seconds",
                                )
                            ),
                            _best_mode_ratio(
                                joined,
                                "generation_seconds",
                                baseline_static_na=baseline_static_na,
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
                            _best_mode_runtime_ratio(
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
                            static_na = (
                                catalog.static_na_reason(cell) is not None
                            )
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
                    _scalar_value(measurement, "execution_seconds_per_point"),
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
    missing_static_na = tuple(
        sorted(static_na_ids - rendered_static_na_cell_ids)
    )
    if missing_static_na:
        contract_errors.append(
            "catalog static N/A cells are not rendered: "
            + ", ".join(missing_static_na)
        )
    return VisibleCompleteness(
        maximum_n_final=max_n_final,
        declared_measurement_cell_count=(
            accounting.declared_measurement_cell_count
        ),
        required_measurement_count=accounting.required_measurement_count,
        rendered_required_measurement_count=len(rendered_cell_ids & required_ids),
        structurally_not_applicable_display_slot_count=structural_seen,
        not_exposed_display_slot_count=not_exposed_seen,
        applicable_na_display_slots=tuple(sorted(set(na_slots))),
        missing_rendered_cell_ids=missing,
        contract_errors=tuple(sorted(set(contract_errors))),
        catalog_static_na_cell_count=accounting.catalog_static_na_cell_count,
        rendered_catalog_static_na_cell_count=len(
            rendered_static_na_cell_ids
        ),
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
