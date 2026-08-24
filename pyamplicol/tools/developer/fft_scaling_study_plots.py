#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Render the FullColor scaling-study report as six publication-style PNGs.

The input report may be complete or still in progress.  Positive measured
values are plotted; failed, not-applicable, and skipped cells are summarized
below each relevant figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "fft_scaling_study_plots.py requires matplotlib; run it with a Python "
        "environment that provides matplotlib"
    ) from error


@dataclass(frozen=True, slots=True)
class Metric:
    key: str
    slug: str
    title: str
    axis_label: str
    scale: float = 1.0


METRICS = (
    Metric(
        "generation_seconds",
        "generation",
        "Setup time",
        "Time [s]",
    ),
    Metric(
        "warm_seconds_per_point",
        "warm-runtime",
        "Warmed runtime per point",
        "Time per point [s]",
    ),
    Metric("max_rss_kib", "rss", "Peak RSS", "Peak RSS [MiB]", 1 / 1024),
)
FAMILIES = ("gg", "ddbar")
FAMILY_TITLES = {
    "gg": r"$g g \;\to\; g g + (n-2)g$",
    "ddbar": r"$d \bar{d} \;\to\; d \bar{d} + (n-2)g$",
}
FAMILY_BASELINES = {"gg": "reference-fft", "ddbar": "amplicol"}
MODE_LABELS = {
    "reference-fft": "Reference FFT",
    "amplicol": "AmpliCol",
    "recurrence-direct": "pyAmpliCol - recurrence",
    "recurrence-fft": "pyAmpliCol - recurrence - FFT",
    "otf-direct": "pyAmpliCol - OTF",
    "otf-fft": "pyAmpliCol - OTF - FFT",
    "compiled-direct": "pyAmpliCol - Compiled",
    "compiled-fft": "pyAmpliCol - Compiled - FFT",
    "madgraph-standalone": "MadGraph standalone (fixed h)",
}
MODE_STYLES = {
    "reference-fft": ("#000000", "o", "-"),
    "amplicol": ("#F0E442", "D", "-"),
    "recurrence-direct": ("#009E73", "o", "--"),
    "recurrence-fft": ("#009E73", "s", "-"),
    "otf-direct": ("#0072B2", "^", "--"),
    "otf-fft": ("#0072B2", "v", "-"),
    "compiled-direct": ("#D55E00", "P", "--"),
    "compiled-fft": ("#7A3E9D", "X", "-"),
    "madgraph-standalone": ("#CC3311", "*", "-"),
}
LEGEND_MODE_ORDER = {
    # Matplotlib fills a multi-column legend down each column.  Keep each
    # Direct/FFT strategy pair together, with FFT (solid) above Direct
    # (dashed), so the encoding can be read at a glance.
    "gg": (
        "reference-fft",
        "amplicol",
        "recurrence-fft",
        "recurrence-direct",
        "otf-fft",
        "otf-direct",
        "madgraph-standalone",
    ),
    "ddbar": (
        "amplicol",
        "madgraph-standalone",
        "recurrence-fft",
        "recurrence-direct",
        "otf-fft",
        "otf-direct",
    ),
}
STATUS_ORDER = {"failed": 0, "not-applicable": 1, "skipped": 2}
PROVISIONAL_WARM_STATUS = "excluded-provisional-warm-quality"
MODE_PUBLICATION_MAX_MULTIPLICITY = {
    "otf-direct": 6,
    "otf-fft": 6,
}
EXIT_STATUS = re.compile(r"command failed with status\s+(-?\d+)")
PLOT_MANIFEST_NAME = "plot-manifest.json"
PLOT_FILENAMES = tuple(
    f"fullcolor-{family}-{metric.slug}.png"
    for family in FAMILIES
    for metric in METRICS
)


class PlotError(RuntimeError):
    """The report cannot be rendered without guessing its meaning."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="scaling-study report.json")
    parser.add_argument("output_directory", type=Path, help="directory for six PNGs")
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution in dots per inch (default: 160)",
    )
    return parser


def _load_report_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlotError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), dict):
        raise PlotError("report must be a JSON object containing a cells object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    return _load_report_snapshot(path)[0]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mode_order(report: Mapping[str, Any], family: str) -> tuple[str, ...]:
    policy = _mapping(report.get("policy"))
    family_policy = _mapping(_mapping(policy.get("process_families")).get(family))
    requested = family_policy.get("modes")
    if isinstance(requested, Sequence) and not isinstance(requested, str):
        modes = tuple(str(mode) for mode in requested)
        if modes:
            return modes
    family_cells = _mapping(_mapping(report.get("cells")).get(family))
    return tuple(family_cells)


def _multiplicities(report: Mapping[str, Any]) -> tuple[int, ...]:
    policy = _mapping(report.get("policy"))
    requested = policy.get("final_state_multiplicities")
    if isinstance(requested, Sequence) and not isinstance(requested, str):
        values = tuple(sorted({int(value) for value in requested}))
        if values:
            return values
    observed: set[int] = set()
    for family in FAMILIES:
        for cells in _mapping(_mapping(report.get("cells")).get(family)).values():
            observed.update(int(value) for value in _mapping(cells))
    if not observed:
        raise PlotError("report has no requested or observed multiplicities")
    return tuple(sorted(observed))


def _source_boundary(
    report: Mapping[str, Any], metric_key: str | None = None
) -> tuple[float, str] | None:
    policy = _mapping(report.get("policy"))
    plot = _mapping(policy.get("plot"))
    metric_boundaries = _mapping(plot.get("metric_source_boundaries"))
    if metric_key is not None and metric_key in metric_boundaries:
        metric_boundary = metric_boundaries[metric_key]
        if metric_boundary is None:
            return None
        boundary_policy = _mapping(metric_boundary)
    else:
        boundary_policy = plot
    raw_boundary = boundary_policy.get("source_boundary_after_n")
    if raw_boundary is None:
        return None
    if isinstance(raw_boundary, bool) or not isinstance(raw_boundary, int):
        raise PlotError("policy.plot.source_boundary_after_n must be an integer")
    multiplicities = _multiplicities(report)
    if raw_boundary not in multiplicities or raw_boundary + 1 not in multiplicities:
        raise PlotError(
            "policy.plot.source_boundary_after_n must separate two plotted "
            "multiplicities"
        )
    raw_label = boundary_policy.get("source_boundary_label")
    label = (
        raw_label
        if isinstance(raw_label, str) and raw_label.strip()
        else f"source boundary after n={raw_boundary}"
    )
    return raw_boundary + 0.5, label


def _protocol_summary(report: Mapping[str, Any]) -> str:
    policy = _mapping(report.get("policy"))
    measurement = _mapping(policy.get("measurement"))
    if policy.get("fft_enabled") is not True:
        return "Main and ratio panels: logarithmic Y scale"
    details = ["Campaign --fft enabled"]
    if measurement.get("generation_helicity_coverage") == "all":
        details.append("artifacts: all runtime helicities")
    workload = _helicity_workload(report)
    if workload == "fixed":
        details.append("setup/warm workload: one fixed helicity")
    elif workload == "sum":
        details.append("setup/warm workload: complete helicity sum")
        details.append("AmpliCol: create-raw bulk H family")
    batch_size = measurement.get("warm_benchmark_batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise PlotError("measurement warm_benchmark_batch_size must be positive")
    details.append(f"pyAmpliCol CLI profile: batch {batch_size} (cyclic 10 points)")
    details.append("Reference/AmpliCol: scalar aggregates normalized per point")
    if measurement.get("compiled_fft_enabled") is False:
        details.append("compiled FFT disabled")
    details.append("log-scale Y axes")
    return " | ".join(details)


def _protocol_summary_lines(
    report: Mapping[str, Any], *, max_line_length: int = 110
) -> tuple[str, ...]:
    """Wrap the protocol at semantic separators for a stable plot header."""

    if max_line_length < 1:
        raise ValueError("max_line_length must be positive")
    sections = _protocol_summary(report).split(" | ")
    lines: list[str] = []
    current = ""
    for section in sections:
        candidate = section if not current else f"{current} | {section}"
        if current and len(candidate) > max_line_length:
            lines.append(current)
            current = section
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


def _helicity_workload(report: Mapping[str, Any]) -> str:
    """Return the explicitly measured warm-helicity workload.

    Older selected-scalar reports predate ``helicity_workload`` and are
    identified by their existing ``warm_fixed_helicity`` contract.
    """

    policy = _mapping(report.get("policy"))
    measurement = _mapping(policy.get("measurement"))
    measurement_declared = measurement.get("helicity_workload")
    policy_declared = policy.get("helicity_workload")
    if (
        measurement_declared is not None
        and policy_declared is not None
        and str(measurement_declared) != str(policy_declared)
    ):
        raise PlotError(
            "policy and measurement helicity_workload declarations disagree"
        )
    declared = (
        measurement_declared
        if measurement_declared is not None
        else policy_declared
    )
    if declared is not None:
        value = str(declared)
        if value not in {"fixed", "sum"}:
            raise PlotError("policy helicity_workload must be 'fixed' or 'sum'")
        if value == "sum" and (
            measurement.get("warm_fixed_helicity") is not False
            or measurement.get("warm_helicity_sum") is not True
        ):
            raise PlotError(
                "summed policy must record warm_fixed_helicity=false and "
                "warm_helicity_sum=true"
            )
        if value == "fixed" and (
            measurement.get("warm_fixed_helicity") is not True
            or measurement.get("warm_helicity_sum") is True
        ):
            raise PlotError(
                "fixed policy must record warm_fixed_helicity=true without "
                "warm_helicity_sum=true"
            )
        return value
    if (
        measurement.get("warm_helicity_sum") is True
        and measurement.get("warm_fixed_helicity") is True
    ):
        raise PlotError("campaign helicity workload markers are contradictory")
    if measurement.get("warm_helicity_sum") is True:
        return "sum"
    if measurement.get("warm_fixed_helicity") is True:
        return "fixed"
    return "fixed"


def _metric_title(report: Mapping[str, Any], metric: Metric) -> str:
    if metric.key == "warm_seconds_per_point" and _helicity_workload(report) == "sum":
        return "Warmed helicity-summed runtime per point"
    return metric.title


def _metric_axis_label(report: Mapping[str, Any], metric: Metric) -> str:
    if metric.key == "warm_seconds_per_point" and _helicity_workload(report) == "sum":
        return "Time per helicity-summed point [s]"
    return metric.axis_label


def _label(family_cells: Mapping[str, Any], mode: str) -> str:
    for cell in _mapping(family_cells.get(mode)).values():
        label = _mapping(cell).get("label")
        if isinstance(label, str) and label:
            return label
    return MODE_LABELS.get(mode, mode)


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _series(
    family_cells: Mapping[str, Any], mode: str, metric: Metric
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for raw_n, raw_cell in _mapping(family_cells.get(mode)).items():
        multiplicity = int(raw_n)
        maximum = MODE_PUBLICATION_MAX_MULTIPLICITY.get(mode)
        if maximum is not None and multiplicity > maximum:
            continue
        cell = _mapping(raw_cell)
        if cell.get("status") != "measured":
            continue
        exclusion = _mapping(_mapping(cell.get("excluded_metrics")).get(metric.key))
        exclusion_status = exclusion.get("status")
        if isinstance(exclusion_status, str) and exclusion_status.startswith(
            "excluded-"
        ):
            # Compatibility with already-written rolling snapshots: the
            # superseded 3% advisory gate moved otherwise valid measurements
            # under excluded_metrics.  Plot those values exactly like any
            # other measured warm point so curves and ratios stay continuous.
            if (
                metric.key == "warm_seconds_per_point"
                and exclusion_status == PROVISIONAL_WARM_STATUS
            ):
                value = _positive_number(exclusion.get("value"))
            else:
                continue
        else:
            value = _positive_number(_mapping(cell.get("metrics")).get(metric.key))
        if value is not None:
            values.append((multiplicity, value * metric.scale))
    return sorted(values)


def _family_cells_for_metric(
    report: Mapping[str, Any], family: str, metric: Metric
) -> tuple[dict[str, Any], tuple[str, ...]]:
    family_cells = dict(_mapping(_mapping(report.get("cells")).get(family)))
    modes = _mode_order(report, family)
    runtime_series = _mapping(_mapping(report.get("runtime_series")).get(family))
    for mode, raw_cells in runtime_series.items():
        mode = str(mode)
        if mode in family_cells or mode in modes:
            raise PlotError(f"external series duplicates {family}/{mode}")
        cells = _mapping(raw_cells)
        family_cells[mode] = {
            str(raw_n): dict(_mapping(raw_cell)) for raw_n, raw_cell in cells.items()
        }
        modes = (*modes, mode)
    _validate_cell_workloads(report, family, family_cells)
    excluded_modes = _metric_series_exclusions(report, family, metric)
    if excluded_modes:
        modes = tuple(mode for mode in modes if mode not in excluded_modes)
        for mode in excluded_modes:
            family_cells.pop(mode, None)
    return family_cells, modes


def _declared_cell_workload(cell: Mapping[str, Any]) -> str | None:
    markers: set[str] = set()
    declared = cell.get("helicity_workload")
    if declared is not None:
        workload = str(declared)
        if workload not in {"fixed", "sum"}:
            raise PlotError("measured cell helicity_workload must be 'fixed' or 'sum'")
        markers.add(workload)
    if cell.get("warm_helicity_sum") is True:
        markers.add("sum")
    if cell.get("warm_fixed_helicity") is True:
        markers.add("fixed")
    protocol = _mapping(cell.get("protocol"))
    helicity_summed = protocol.get("helicity_summed")
    if helicity_summed is True:
        markers.add("sum")
    if helicity_summed is False or protocol.get("warm_fixed_helicity") is True:
        markers.add("fixed")
    if len(markers) > 1:
        raise PlotError("measured cell helicity workload markers are contradictory")
    return next(iter(markers), None)


def _validate_cell_workloads(
    report: Mapping[str, Any],
    family: str,
    family_cells: Mapping[str, Any],
) -> None:
    expected = _helicity_workload(report)
    for mode, raw_cells in family_cells.items():
        for raw_n, raw_cell in _mapping(raw_cells).items():
            cell = _mapping(raw_cell)
            if cell.get("status") != "measured":
                continue
            observed = _declared_cell_workload(cell)
            context = f"{family}/{mode}/n={raw_n}"
            if expected == "sum" and observed != "sum":
                raise PlotError(
                    f"{context} is not authenticated as a helicity-summed cell"
                )
            if expected == "fixed" and observed == "sum":
                raise PlotError(
                    f"{context} is summed data in a fixed-helicity campaign"
                )


def _metric_series_exclusions(
    report: Mapping[str, Any], family: str, metric: Metric
) -> Mapping[str, Any]:
    plot = _mapping(_mapping(report.get("policy")).get("plot"))
    metric_exclusions = _mapping(plot.get("metric_series_exclusions"))
    family_exclusions = _mapping(
        _mapping(metric_exclusions.get(metric.key)).get(family)
    )
    return {str(mode): reason for mode, reason in family_exclusions.items()}


def _plot_policy_notes(
    report: Mapping[str, Any],
    family: str,
    metric: Metric,
    original_family_cells: Mapping[str, Any],
) -> list[str]:
    plot = _mapping(_mapping(report.get("policy")).get("plot"))
    notes: list[str] = []
    raw_notes = plot.get("notes")
    if isinstance(raw_notes, Sequence) and not isinstance(raw_notes, str):
        notes.extend(
            note.strip()
            for raw_note in raw_notes
            if isinstance(raw_note, str) and (note := raw_note.strip())
        )

    grouped: dict[str, list[str]] = {}
    for mode, raw_reason in _metric_series_exclusions(report, family, metric).items():
        reason = (
            raw_reason.strip()
            if isinstance(raw_reason, str) and raw_reason.strip()
            else "excluded by the report's metric protocol policy"
        )
        grouped.setdefault(reason, []).append(_label(original_family_cells, mode))
    for reason, labels in grouped.items():
        notes.append(f"{', '.join(labels)}: {metric.title.lower()} omitted ({reason}).")
    archived_modes = [
        mode
        for mode, maximum in MODE_PUBLICATION_MAX_MULTIPLICITY.items()
        if any(
            int(raw_n) > maximum and _mapping(raw_cell).get("status") == "measured"
            for raw_n, raw_cell in _mapping(original_family_cells.get(mode)).items()
        )
    ]
    if archived_modes:
        labels = ", ".join(
            _label(original_family_cells, mode) for mode in archived_modes
        )
        notes.append(
            f"{labels}: final publication shown through n=6; higher-n source "
            "measurements remain archived and are omitted."
        )
    return notes


def _consecutive_runs(
    values: Sequence[tuple[int, float]],
) -> Iterable[list[tuple[int, float]]]:
    run: list[tuple[int, float]] = []
    for point in values:
        if run and point[0] != run[-1][0] + 1:
            yield run
            run = []
        run.append(point)
    if run:
        yield run


def _log_limits(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        return low / 2.0, high * 2.0
    span = math.log10(high) - math.log10(low)
    padding = max(0.10, span * 0.06)
    return 10 ** (math.log10(low) - padding), 10 ** (math.log10(high) + padding)


def _format_n(values: Sequence[int]) -> str:
    ordered = sorted(set(values))
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _timeout_label(report: Mapping[str, Any], policy_key: str) -> str:
    policy = _mapping(report.get("policy"))
    measurement = _mapping(policy.get("measurement"))
    seconds = _positive_number(measurement.get(policy_key))
    return "configured" if seconds is None else f"{seconds / 60:g} min"


def _failure_detail(report: Mapping[str, Any], cell: Mapping[str, Any]) -> str:
    short = cell.get("failure_reason_short")
    if isinstance(short, str) and short.strip():
        return short.strip()
    category = cell.get("failure_category")
    reason = cell.get("failure_reason")
    reason_text = reason if isinstance(reason, str) else ""
    if category == "memory-limit":
        return "memory watchdog limit"
    if category == "legacy-amplicol-structural-limit":
        return "legacy AmpliCol structural limit"
    if category in {"runtime-time-limit", "runtime-timeout"}:
        return f"{_timeout_label(report, 'runtime_timeout_seconds')} runtime limit"
    if category == "setup-or-runtime-time-limit":
        return (
            f"{_timeout_label(report, 'generation_timeout_seconds')} "
            "setup/runtime limit"
        )
    if category in {"generation-time-limit", "generation-timeout", "timeout"}:
        return (
            f"{_timeout_label(report, 'generation_timeout_seconds')} "
            "setup-time limit"
        )
    match = EXIT_STATUS.search(reason_text)
    if match:
        return f"generation error (exit {match.group(1)})"
    if reason_text and len(reason_text) <= 90:
        return reason_text.rstrip(".")
    return str(category or "generation error")


def _status_notes(
    report: Mapping[str, Any],
    family: str,
    modes: Sequence[str],
    *,
    family_cells: Mapping[str, Any] | None = None,
) -> list[str]:
    if family_cells is None:
        family_cells = _mapping(_mapping(report.get("cells")).get(family))
    notes: list[str] = []
    for mode in modes:
        grouped: dict[tuple[str, str], list[int]] = {}
        for raw_n, raw_cell in _mapping(family_cells.get(mode)).items():
            multiplicity = int(raw_n)
            maximum = MODE_PUBLICATION_MAX_MULTIPLICITY.get(mode)
            if maximum is not None and multiplicity > maximum:
                continue
            cell = _mapping(raw_cell)
            status = cell.get("status")
            if status not in STATUS_ORDER:
                continue
            if status == "failed":
                detail = _failure_detail(report, cell)
            else:
                reason = cell.get("failure_reason")
                detail = reason.rstrip(".") if isinstance(reason, str) else status
            grouped.setdefault((str(status), detail), []).append(multiplicity)
        if not grouped:
            continue
        fragments = []
        for (status, detail), values in sorted(
            grouped.items(), key=lambda item: (STATUS_ORDER[item[0][0]], item[1])
        ):
            display_status = "N/A" if status == "not-applicable" else status
            fragments.append(f"n={_format_n(values)} {display_status} ({detail})")
        notes.append(f"{_label(family_cells, mode)}: " + "; ".join(fragments) + ".")
    report_status = report.get("status")
    if not isinstance(report_status, str) or not report_status.startswith("complete"):
        status_reason = report.get("status_reason")
        if report_status == "stopped-protocol-investigation":
            reason = (
                status_reason.strip()
                if isinstance(status_reason, str) and status_reason.strip()
                else "measurement protocol under investigation"
            )
            notes.append(
                f"Campaign stopped: {reason}. Absent cells are unattempted and omitted."
            )
        else:
            notes.append(
                "Campaign is still running; absent cells are unattempted and omitted."
            )
    return notes


def _wrapped_notes(notes: Sequence[str], width: int = 138) -> list[str]:
    lines: list[str] = []
    for note in notes:
        wrapped = textwrap.wrap(
            note,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent="   ",
        )
        if wrapped:
            lines.extend(("• " + wrapped[0], *wrapped[1:]))
    return lines


def _style_axes(axis: Axes) -> None:
    axis.grid(True, which="major", color="#D7DCE2", linewidth=0.7, alpha=0.85)
    axis.grid(True, which="minor", axis="y", color="#E9ECF0", linewidth=0.5, alpha=0.7)
    for spine in axis.spines.values():
        spine.set_color("#75808C")
        spine.set_linewidth(0.8)
    axis.tick_params(colors="#20262D", labelsize=9)


def _draw_run(
    axis: Axes,
    run: Sequence[tuple[int, float]],
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    x_values = [point[0] for point in run]
    y_values = [point[1] for point in run]
    axis.plot(
        x_values,
        y_values,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.7,
        markersize=5.3,
        markeredgewidth=0.8,
        markeredgecolor="white" if marker not in {"+", "x"} else color,
        zorder=3,
    )


def _plot_metric(
    report: Mapping[str, Any],
    family: str,
    metric: Metric,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    original_family_cells = _mapping(_mapping(report.get("cells")).get(family))
    family_cells, modes = _family_cells_for_metric(report, family, metric)
    multiplicities = _multiplicities(report)
    baseline = FAMILY_BASELINES[family]
    report_status = report.get("status")
    allow_empty = not (
        isinstance(report_status, str) and report_status.startswith("complete")
    )
    if baseline not in modes and not allow_empty:
        raise PlotError(
            f"{family} mode list does not contain ratio baseline {baseline!r}"
        )

    series = {mode: _series(family_cells, mode, metric) for mode in modes}
    main_values = [value for values in series.values() for _, value in values]
    if not main_values and not allow_empty:
        raise PlotError(f"no measured positive {metric.key} values for {family}")
    baseline_values = dict(series.get(baseline, ()))
    ratios: dict[str, list[tuple[int, float]]] = {}
    for mode, values in series.items():
        ratios[mode] = [
            (n, value / baseline_values[n])
            for n, value in values
            if n in baseline_values and baseline_values[n] > 0.0
        ]
    ratio_values = [value for values in ratios.values() for _, value in values]
    if not ratio_values and not allow_empty:
        raise PlotError(f"no ratios can be formed for {family} {metric.key}")

    notes = _wrapped_notes(
        [
            *_status_notes(report, family, modes, family_cells=family_cells),
            *_plot_policy_notes(report, family, metric, original_family_cells),
        ]
    )
    # Reserve enough space below the ratio axes for its tick labels and x-axis
    # title before the failure-note block begins.  Keeping these as separate
    # vertical bands avoids collisions in partial campaigns with several notes.
    notes_height = 1.15 + 0.20 * max(1, len(notes))
    figure_height = 8.07 + 0.20 * max(0, len(notes) - 1)
    bottom = notes_height / figure_height
    figure = plt.figure(figsize=(12.2, figure_height), facecolor="white")
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(3.0, 1.35),
        hspace=0.08,
        left=0.105,
        right=0.975,
        top=0.765,
        bottom=bottom,
    )
    main_axis = figure.add_subplot(grid[0])
    ratio_axis = figure.add_subplot(grid[1], sharex=main_axis)

    legend_entries: dict[str, tuple[Line2D, str]] = {}
    for mode in modes:
        color, marker, linestyle = MODE_STYLES.get(mode, ("#59636E", "o", "-"))
        if series[mode]:
            legend_entries[mode] = (
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.7,
                    markersize=5.3,
                    markeredgewidth=0.8,
                ),
                _label(family_cells, mode),
            )
        for run in _consecutive_runs(series[mode]):
            _draw_run(
                main_axis,
                run,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )
        for run in _consecutive_runs(ratios[mode]):
            _draw_run(
                ratio_axis,
                run,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )

    main_axis.set_yscale("log")
    if main_values:
        main_axis.set_ylim(*_log_limits(main_values))
    else:
        main_axis.set_ylim(0.5, 2.0)
        main_axis.set_yticks([])
        main_axis.text(
            0.5,
            0.5,
            "No protocol-valid measurements are currently available\n"
            "for this process and metric.",
            transform=main_axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#4B5560",
        )
    main_axis.set_ylabel(_metric_axis_label(report, metric), fontsize=10)
    main_axis.tick_params(axis="x", labelbottom=False)
    ratio_axis.set_yscale("log")
    ratio_axis.axhline(1.0, color="#59636E", linewidth=0.9, linestyle=":", zorder=1)
    ratio_axis.set_ylim(*_log_limits([*ratio_values, 1.0]))
    ratio_axis.set_ylabel(
        f"Ratio to\n{_label(original_family_cells, baseline)}", fontsize=9
    )
    if not ratio_values:
        ratio_axis.text(
            0.5,
            0.5,
            "Ratios unavailable until the baseline is remeasured.",
            transform=ratio_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4B5560",
        )
    ratio_axis.set_xlabel("Final-state multiplicity n", fontsize=10)
    ratio_axis.set_xticks(multiplicities)
    ratio_axis.set_xlim(min(multiplicities) - 0.25, max(multiplicities) + 0.25)
    _style_axes(main_axis)
    _style_axes(ratio_axis)

    source_boundary = _source_boundary(report, metric.key)
    if source_boundary is not None:
        boundary_x, boundary_label = source_boundary
        for axis in (main_axis, ratio_axis):
            axis.axvline(
                boundary_x,
                color="#75808C",
                linewidth=1.0,
                linestyle=(0, (2.5, 2.5)),
                zorder=2,
            )
        main_axis.text(
            boundary_x,
            0.975,
            boundary_label,
            transform=main_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.8,
            color="#59636E",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=4,
        )

    figure.suptitle(
        f"FullColor {_metric_title(report, metric)}: {FAMILY_TITLES[family]}",
        x=0.54,
        y=0.975,
        fontsize=15,
        fontweight="semibold",
    )
    protocol_lines = _protocol_summary_lines(report)
    figure.text(
        0.54,
        0.945,
        "\n".join(protocol_lines),
        ha="center",
        va="top",
        fontsize=8.7,
        color="#59636E",
        linespacing=1.15,
    )
    if legend_entries:
        preferred_modes = LEGEND_MODE_ORDER[family]
        legend_modes = [mode for mode in preferred_modes if mode in legend_entries]
        legend_modes.extend(
            mode
            for mode in modes
            if mode in legend_entries and mode not in preferred_modes
        )
        handles = [legend_entries[mode][0] for mode in legend_modes]
        labels = [legend_entries[mode][1] for mode in legend_modes]
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.91 - 0.011 * len(protocol_lines)),
            ncol=4 if family == "gg" else 3,
            frameon=False,
            fontsize=8.8,
            handlelength=2.5,
            columnspacing=1.35,
        )

    if notes:
        figure.text(
            0.105,
            (notes_height - 0.68) / figure_height,
            "Failures and omitted points",
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="semibold",
            color="#20262D",
        )
        line_step = 0.20 / figure_height
        y = (notes_height - 0.96) / figure_height
        for line in notes:
            figure.text(
                0.105,
                y,
                line,
                ha="left",
                va="top",
                fontsize=8.2,
                color="#4B5560",
            )
            y -= line_step

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    figure.savefig(
        temporary_path,
        format="png",
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "pyAmpliCol fft_scaling_study_plots.py"},
    )
    plt.close(figure)
    temporary_path.replace(output_path)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "axes.labelcolor": "#20262D",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#20262D",
        }
    )


def _render(report: Mapping[str, Any], output_directory: Path, *, dpi: int) -> None:
    if dpi < 72 or dpi > 600:
        raise PlotError("--dpi must be between 72 and 600")
    output_directory.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    for family in FAMILIES:
        for metric in METRICS:
            output_path = output_directory / f"fullcolor-{family}-{metric.slug}.png"
            _plot_metric(report, family, metric, output_path, dpi=dpi)
            print(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_plot_manifest(
    report_sha256: str,
    report: Mapping[str, Any],
    output_directory: Path,
) -> Path:
    """Bind a rendered plot set to its exact source report and workload."""

    plot_hashes: dict[str, str] = {}
    for filename in PLOT_FILENAMES:
        plot_path = output_directory / filename
        if not plot_path.is_file():
            raise PlotError(f"rendered plot is missing: {plot_path}")
        plot_hashes[filename] = _sha256(plot_path)
    manifest = {
        "kind": "pyamplicol-fft-scaling-plots",
        "schema_version": 1,
        "report_sha256": report_sha256,
        "helicity_workload": _helicity_workload(report),
        "plots": plot_hashes,
    }
    manifest_path = output_directory / PLOT_MANIFEST_NAME
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    print(manifest_path)
    return manifest_path


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report, report_sha256 = _load_report_snapshot(arguments.report)
        _render(report, arguments.output_directory, dpi=arguments.dpi)
        _write_plot_manifest(
            report_sha256,
            report,
            arguments.output_directory,
        )
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
