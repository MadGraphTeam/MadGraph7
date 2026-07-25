# SPDX-License-Identifier: 0BSD
"""Dynamic baseline joins and fixed-block TeX rendering for the report tables."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .cache import empty_measurement
from .catalog import REPORT_CATALOG, ReportCatalog, z_dataset_id
from .models import (
    Accuracy,
    ExecutionMode,
    MatrixDataset,
    ModelKey,
    ProcessFamily,
    ResultStatus,
    ScalarDataset,
    Workload,
    ZVariant,
)
from .timing import below_resolution_record
from .validation_summary import (
    SUMMARY_TABLE_NAME,
    render_validation_summary,
)

Measurement = Mapping[str, object]
CachePayload = Mapping[str, object]

_NA = empty_measurement()
_MATRIX_BLOCK_SIZE = 3
_Z_BLOCK_SIZE = 3


def _chunks(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
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
            joined = tuple(
                JoinedWorkload(workload, _NA, _NA) for workload in workloads
            )
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


def _below_resolution_time(
    measurement: Measurement,
    field: str,
    *,
    microseconds: bool = False,
) -> str | None:
    record = below_resolution_record(measurement, field)
    if record is None:
        return None
    return r"\matrixstatus{ReportMuted}{below res.}"


def _status(measurement: Measurement) -> str:
    status = str(measurement.get("status", ResultStatus.NOT_AVAILABLE.value))
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


def _metric(measurement: Measurement, field: str, *, microseconds: bool = False) -> str:
    if not _ok(measurement):
        return _status(measurement)
    below = _below_resolution_time(
        measurement,
        field,
        microseconds=microseconds,
    )
    if below is not None:
        return below
    return _time(measurement.get(field), microseconds=microseconds)


def _ratio_value(
    candidate: Measurement,
    baseline: Measurement,
    field: str,
) -> float | None:
    if not (_ok(candidate) and _ok(baseline)):
        return None
    if (
        below_resolution_record(candidate, field) is not None
        or below_resolution_record(baseline, field) is not None
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
            below_resolution_record(candidate, field) is not None
            or below_resolution_record(baseline, field) is not None
        ):
            return r"\matrixstatus{ReportMuted}{below res.}"
        return r"\matrixnaratio{ReportMuted}"
    color = (
        "ReportGreen"
        if value < 1.0
        else "ReportOrange"
        if value < 2.0
        else "ReportRed"
    )
    return rf"\matrixratio{{{color}}}{{{_compact(value)}}}"


def _ratio_pair(candidate: Measurement, baseline: Measurement) -> str:
    wall = _ratio_value(candidate, baseline, "wall_seconds_per_point")
    execution = _ratio_value(candidate, baseline, "execution_seconds_per_point")
    if not _ok(candidate):
        return _status(candidate)

    def field(value: float | None, name: str) -> tuple[str, str]:
        if value is None:
            if (
                below_resolution_record(candidate, name) is not None
                or below_resolution_record(baseline, name) is not None
            ):
                return "ReportMuted", "below res."
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
    execution_color, execution_text = field(
        execution,
        "execution_seconds_per_point",
    )
    return (
        rf"\matrixratiopair{{{wall_color}}}{{{wall_text}}}"
        rf"{{{execution_color}}}{{{execution_text}}}"
    )


def _matrix_macros() -> list[str]:
    return [
        r"\providecommand{\matrixentryfont}{\fontsize{6.2pt}{7.0pt}\selectfont}",
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
        r"\providecommand{\matrixna}[1]{\textcolor{#1}{\texttt{N/A}}}",
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
            r"\begingroup\matrixentryfont"
            r"\begin{tabular}[t]{"
            r"@{}l@{\hspace{0.035in}}l@{\hspace{0.035in}}l@{}}"
            r"#1&#2&#3\\#4&#5&#6"
            r"\end{tabular}\endgroup}"
        ),
        (
            r"\providecommand{\matrixcellcontracted}[4]{"
            r"\begingroup\matrixentryfont"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.04in}}l@{}}"
            r"#1&#2\\#3&#4"
            r"\end{tabular}\endgroup}"
        ),
    ]


def _lc_cell(view: JoinedMatrixCell) -> str:
    selected, all_flow = view.workloads
    baseline_generation = (
        rf"\matrixpair{{{_metric(selected.baseline, 'generation_seconds')}}}"
        rf"{{{_metric(all_flow.baseline, 'generation_seconds')}}}"
    )
    selected_generation_ratio = _ratio(
        selected.candidate,
        selected.baseline,
        "generation_seconds",
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
            and _ok(all_flow.baseline)
        )
        else _ratio(
            all_flow.candidate,
            all_flow.baseline,
            "generation_seconds",
        )
    )
    selected_runtime = _metric(
        selected.baseline,
        "wall_seconds_per_point",
        microseconds=True,
    )
    all_flow_runtime = _metric(
        all_flow.baseline,
        "wall_seconds_per_point",
        microseconds=True,
    )
    baseline_runtime = (
        rf"\matrixpair{{{selected_runtime}}}"
        rf"{{{all_flow_runtime}}}"
    )
    selected_ratio = _ratio_pair(selected.candidate, selected.baseline)
    all_flow_ratio = _ratio_pair(all_flow.candidate, all_flow.baseline)
    return (
        r"\matrixcelllc"
        f"{{{baseline_generation}}}"
        f"{{{selected_generation_ratio}}}"
        f"{{{all_flow_generation_ratio}}}"
        f"{{{baseline_runtime}}}"
        f"{{{selected_ratio}}}"
        f"{{{all_flow_ratio}}}"
    )


def _contracted_cell(view: JoinedMatrixCell) -> str:
    joined = view.workloads[0]
    return (
        r"\matrixcellcontracted"
        f"{{{_metric(joined.baseline, 'generation_seconds')}}}"
        f"{{{_ratio(joined.candidate, joined.baseline, 'generation_seconds')}}}"
        f"{{{_metric(joined.baseline, 'wall_seconds_per_point', microseconds=True)}}}"
        f"{{{_ratio_pair(joined.candidate, joined.baseline)}}}"
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
        and below_resolution_record(item.baseline, field) is None
        and below_resolution_record(item.candidate, field) is None
    ]
    if not valid:
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
    column_spec = (
        r"@{}r@{\hspace{0.04in}}L{1.65in}"
        + "".join(r"@{\hspace{0.06in}}L{2.49in}" for _ in multiplicities)
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
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.06}",
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
                row.append(r"\matrixna{ReportMuted}")
            elif dataset.candidate.accuracy is Accuracy.LC:
                row.append(_lc_cell(view))
            else:
                row.append(_contracted_cell(view))
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
                "(not comparable). Both runtime entries retain their "
                "(wall|native execution) ratios."
            )
        else:
            detail = (
                "Each cell shows the topology-replay/all-flow-union baseline "
                "generation times and wall times, followed by layout-matched "
                "candidate/baseline generation and "
                "(wall|native execution) ratios."
            )
    else:
        detail = (
            "Each cell shows the baseline generation and wall time, followed by "
            "candidate/baseline generation and (wall|native execution) ratios."
        )
    return (
        r"\ReportTableNote{Baseline: "
        + _tex_escape(baseline)
        + "; candidate: "
        + _tex_escape(candidate)
        + ". "
        + _tex_escape(detail)
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
) -> str:
    measurement = joined.baseline if reference else joined.candidate
    absolute = _metric(measurement, field, microseconds=microseconds)
    if reference or not _ok(measurement):
        return absolute
    if not comparable:
        return rf"\matrixncabsolute{{{absolute}}}"
    return absolute + r"\," + _ratio(measurement, joined.baseline, field)


def _z_block(
    adapter: BaselineCandidateAdapter,
    *,
    model: ModelKey,
    multiplicities: tuple[int, ...],
    block_index: int,
    block_count: int,
) -> list[str]:
    model_label = (
        "Built-in SM" if model is ModelKey.BUILTIN_SM else "UFO-SM"
    )
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
        r"\tiny",
        r"\setlength{\tabcolsep}{1.4pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        (
            r"\begin{tabular}{@{}r L{1.18in} "
            r"R{0.82in} R{0.82in} R{0.82in} "
            r"@{\hspace{0.08in}}"
            r"R{0.82in} R{0.82in} R{0.82in}@{}}"
        ),
        r"\toprule",
        (
            r"\textbf{n} & \textbf{setup} & "
            r"\multicolumn{3}{c}{\textbf{selected flow, helicity sum}} & "
            r"\multicolumn{3}{c}{\textbf{all flows, single helicity}} \\"
        ),
        (
            r"& & \textbf{gen [s]} & \textbf{wall [us/pt]} & "
            r"\textbf{exec [us/pt]} & \textbf{gen [s]} & "
            r"\textbf{wall [us/pt]} & \textbf{exec [us/pt]} \\"
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
                ),
                _z_value(
                    selected,
                    "wall_seconds_per_point",
                    reference=reference,
                    microseconds=True,
                ),
                (
                    r"\matrixna{ReportMuted}"
                    if reference
                    else _z_value(
                        selected,
                        "execution_seconds_per_point",
                        reference=False,
                        microseconds=True,
                    )
                ),
                _z_value(
                    all_flow,
                    "generation_seconds",
                    reference=reference,
                    comparable=False,
                ),
                _z_value(
                    all_flow,
                    "wall_seconds_per_point",
                    reference=reference,
                    microseconds=True,
                ),
                (
                    r"\matrixna{ReportMuted}"
                    if reference
                    else _z_value(
                        all_flow,
                        "execution_seconds_per_point",
                        reference=False,
                        microseconds=True,
                    )
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
                r"marked n.c. because its setup boundary differs from the "
                r"reference.}"
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
            "relative diff. vs hp & "
            + " & ".join(relative_values)
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\endgroup",
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


def render_all_tables(
    caches: Mapping[str, CachePayload] | Iterable[CachePayload],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, str]:
    cache_source = caches if isinstance(caches, Mapping) else tuple(caches)
    return {
        SUMMARY_TABLE_NAME: render_validation_summary(
            cache_source,
            catalog=catalog,
        ),
        **render_all_matrix_tables(cache_source, catalog=catalog),
        **render_all_z_ladders(cache_source, catalog=catalog),
        **render_all_scalar_ladders(cache_source, catalog=catalog),
    }


__all__ = [
    "BaselineCandidateAdapter",
    "JoinedMatrixCell",
    "JoinedWorkload",
    "MeasurementIndex",
    "render_all_matrix_tables",
    "render_all_scalar_ladders",
    "render_all_tables",
    "render_all_z_ladders",
    "render_matrix_table",
    "render_scalar_ladder",
    "render_validation_summary",
    "render_z_ladder",
]
