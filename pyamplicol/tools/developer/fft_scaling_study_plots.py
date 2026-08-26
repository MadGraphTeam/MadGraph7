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


@dataclass(frozen=True, slots=True)
class AxisOptions:
    main_y_range: tuple[float, float] | None = None
    ratio_y_range: tuple[float, float] | None = None
    ratio_y_scale: str = "log"


@dataclass(frozen=True, slots=True)
class LineFilterOptions:
    main_include_lines: tuple[str, ...] | None = None
    main_veto_lines: tuple[str, ...] = ()
    ratio_include_lines: tuple[str, ...] | None = None
    ratio_veto_lines: tuple[str, ...] = ()


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
RECOLA_MODE = "recola"
RECOLA_LABEL = "Recola"
RECOLA_FAMILY_MAP = {
    "all_gluon": "gg",
    "down_quark_qcd": "ddbar",
}
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
    RECOLA_MODE: RECOLA_LABEL,
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
    RECOLA_MODE: ("#AA4499", "h", "-"),
}
PLOT_LINE_SELECTOR_MODES = {
    "reference-fft": ("reference-fft",),
    "amplicol": ("amplicol",),
    "pyamplicol-recurrence": ("recurrence-direct", "recurrence-fft"),
    "pyamplicol-otf": ("otf-direct", "otf-fft"),
    "madgraph": ("madgraph-standalone",),
    "recurrence-direct": ("recurrence-direct",),
    "recurrence-fft": ("recurrence-fft",),
    "otf-direct": ("otf-direct",),
    "otf-fft": ("otf-fft",),
    "madgraph-standalone": ("madgraph-standalone",),
    "recola": (RECOLA_MODE,),
    "compiled-direct": ("compiled-direct",),
    "compiled-fft": ("compiled-fft",),
}
PLOT_LINE_CHOICES = tuple(PLOT_LINE_SELECTOR_MODES)
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
        RECOLA_MODE,
    ),
    "ddbar": (
        "amplicol",
        "madgraph-standalone",
        "recurrence-fft",
        "recurrence-direct",
        "otf-fft",
        "otf-direct",
        RECOLA_MODE,
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
PLOT_DATA_NAME = "plot-data.json"
PLOT_FILENAMES = tuple(
    f"fullcolor-{family}-{metric.slug}.png"
    for family in FAMILIES
    for metric in METRICS
)
DEFAULT_AXIS_OPTIONS = AxisOptions()
DEFAULT_LINE_FILTER_OPTIONS = LineFilterOptions()


class PlotError(RuntimeError):
    """The report cannot be rendered without guessing its meaning."""


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


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
    parser.add_argument(
        "--recola-results",
        type=Path,
        help="overlay a Recola profiling JSON as an external Recola series",
    )
    parser.add_argument(
        "--main-y-range",
        type=_finite_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="force the main-panel log y-range for every plot",
    )
    parser.add_argument(
        "--ratio-y-range",
        type=_finite_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="force the ratio-panel y-range for every plot",
    )
    parser.add_argument(
        "--ratio-y-scale",
        choices=("log", "linear"),
        default="log",
        help="ratio-panel y-axis scale (default: log)",
    )
    parser.add_argument(
        "--main-include-lines",
        nargs="+",
        choices=PLOT_LINE_CHOICES,
        metavar="LINE",
        help="main-panel line ids to include; overrides --main-veto-lines",
    )
    parser.add_argument(
        "--main-veto-lines",
        nargs="+",
        choices=PLOT_LINE_CHOICES,
        metavar="LINE",
        help="main-panel line ids to hide when --main-include-lines is absent",
    )
    parser.add_argument(
        "--ratio-include-lines",
        nargs="+",
        choices=PLOT_LINE_CHOICES,
        metavar="LINE",
        help="ratio-panel line ids to include; overrides --ratio-veto-lines",
    )
    parser.add_argument(
        "--ratio-veto-lines",
        nargs="+",
        choices=PLOT_LINE_CHOICES,
        metavar="LINE",
        help="ratio-panel line ids to hide when --ratio-include-lines is absent",
    )
    return parser


def _validate_y_range(
    values: Sequence[float] | None,
    *,
    option: str,
    positive: bool,
) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise PlotError(f"{option} requires exactly two values")
    low, high = (float(values[0]), float(values[1]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise PlotError(f"{option} values must be finite")
    if high <= low:
        raise PlotError(f"{option} requires MIN < MAX")
    if positive and low <= 0.0:
        raise PlotError(f"{option} must be positive for a logarithmic y-axis")
    return low, high


def _axis_options_from_arguments(arguments: argparse.Namespace) -> AxisOptions:
    ratio_y_scale = str(arguments.ratio_y_scale)
    if ratio_y_scale not in {"log", "linear"}:
        raise PlotError("--ratio-y-scale must be 'log' or 'linear'")
    return AxisOptions(
        main_y_range=_validate_y_range(
            arguments.main_y_range,
            option="--main-y-range",
            positive=True,
        ),
        ratio_y_range=_validate_y_range(
            arguments.ratio_y_range,
            option="--ratio-y-range",
            positive=ratio_y_scale == "log",
        ),
        ratio_y_scale=ratio_y_scale,
    )


def _validated_line_names(
    values: Sequence[str] | None,
    *,
    option: str,
    default: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return default
    selected = tuple(str(value) for value in values)
    if len(set(selected)) != len(selected):
        raise PlotError(f"{option} must not contain duplicates")
    invalid = sorted(set(selected) - set(PLOT_LINE_CHOICES))
    if invalid:
        raise PlotError(f"{option} has unknown line id(s): {', '.join(invalid)}")
    return selected


def _line_filter_options_from_arguments(
    arguments: argparse.Namespace,
) -> LineFilterOptions:
    return LineFilterOptions(
        main_include_lines=_validated_line_names(
            arguments.main_include_lines,
            option="--main-include-lines",
            default=None,
        ),
        main_veto_lines=_validated_line_names(
            arguments.main_veto_lines,
            option="--main-veto-lines",
            default=(),
        )
        or (),
        ratio_include_lines=_validated_line_names(
            arguments.ratio_include_lines,
            option="--ratio-include-lines",
            default=None,
        ),
        ratio_veto_lines=_validated_line_names(
            arguments.ratio_veto_lines,
            option="--ratio-veto-lines",
            default=(),
        )
        or (),
    )


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


def _protocol_summary(
    report: Mapping[str, Any], *, ratio_y_scale: str = "log"
) -> str:
    policy = _mapping(report.get("policy"))
    measurement = _mapping(policy.get("measurement"))
    if policy.get("fft_enabled") is not True:
        return f"main Y log; ratio Y {ratio_y_scale}"
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
    details.append(f"main Y log; ratio Y {ratio_y_scale}")
    return " | ".join(details)


def _protocol_summary_lines(
    report: Mapping[str, Any],
    *,
    max_line_length: int = 110,
    ratio_y_scale: str = "log",
) -> tuple[str, ...]:
    """Wrap the protocol at semantic separators for a stable plot header."""

    if max_line_length < 1:
        raise ValueError("max_line_length must be positive")
    sections = _protocol_summary(report, ratio_y_scale=ratio_y_scale).split(" | ")
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


def _load_recola_results(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlotError(f"cannot read Recola results {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise PlotError("Recola results must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _recola_cell_workload(
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    context: str,
) -> str:
    polarized = cell.get("polarized")
    if not isinstance(polarized, bool):
        polarized = config.get("polarized")
    if not isinstance(polarized, bool):
        raise PlotError(f"{context} lacks a boolean Recola polarized marker")
    return "fixed" if polarized else "sum"


def _recola_metrics(cell: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    generation_seconds = _positive_number(cell.get("gen_time_s"))
    if generation_seconds is not None:
        metrics["generation_seconds"] = generation_seconds
    warm_seconds = _positive_number(cell.get("run_time_s"))
    if warm_seconds is not None:
        metrics["warm_seconds_per_point"] = warm_seconds
    rss_values = [
        value
        for field in (
            "peak_generation_rss_mib",
            "ram_after_generation_mib",
            "ram_after_profile_mib",
        )
        if (value := _positive_number(cell.get(field))) is not None
    ]
    if rss_values:
        metrics["max_rss_kib"] = max(rss_values) * 1024.0
    return metrics


def _recola_failure_category(reason: str) -> str:
    if "RSS" in reason or "MAX_RSS" in reason:
        return "memory-limit"
    if "generation time" in reason or "MAX_GEN_T" in reason:
        return "generation-time-limit"
    return "generation-timeout"


def _recola_source_metadata(
    path: Path,
    digest: str,
    cells_by_family: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    return {
        "kind": "recola-profile-results",
        "path": str(path.expanduser().resolve(strict=False)),
        "sha256": digest,
        "mode": RECOLA_MODE,
        "label": RECOLA_LABEL,
        "family_map": dict(RECOLA_FAMILY_MAP),
        "metrics": {
            "generation_seconds": "gen_time_s",
            "warm_seconds_per_point": "run_time_s",
            "max_rss_kib": (
                "max(peak_generation_rss_mib, ram_after_generation_mib, "
                "ram_after_profile_mib) * 1024"
            ),
        },
        "multiplicities": {
            family: list(multiplicities)
            for family, multiplicities in cells_by_family.items()
        },
    }


def _attach_recola_results(report: dict[str, Any], path: Path) -> dict[str, Any]:
    payload, digest = _load_recola_results(path)
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise PlotError("Recola results must contain a results object")
    config = _mapping(payload.get("config"))
    expected_workload = _helicity_workload(report)
    config_polarized = config.get("polarized")
    if isinstance(config_polarized, bool):
        config_workload = "fixed" if config_polarized else "sum"
        if config_workload != expected_workload:
            raise PlotError(
                f"Recola results are {config_workload} workload but the campaign "
                f"report is {expected_workload}"
            )
    series: dict[str, dict[str, dict[str, Any]]] = {}
    multiplicities_by_family: dict[str, list[int]] = {family: [] for family in FAMILIES}
    measured_count = 0

    for raw_family, raw_family_results in results.items():
        if not isinstance(raw_family_results, Mapping):
            raise PlotError(f"Recola results family {raw_family!r} must be an object")
        for raw_n, raw_cell in raw_family_results.items():
            if raw_cell is None:
                continue
            if not isinstance(raw_cell, Mapping):
                raise PlotError(f"Recola {raw_family}/n={raw_n} must be an object")
            try:
                multiplicity = int(raw_n)
            except (TypeError, ValueError) as error:
                raise PlotError(
                    f"Recola multiplicity {raw_family}/n={raw_n!r} is not an integer"
                ) from error
            source_family = str(raw_cell.get("family") or raw_family)
            target_family = RECOLA_FAMILY_MAP.get(source_family)
            if target_family is None:
                continue
            context = f"Recola {source_family}/n={multiplicity}"
            status = raw_cell.get("status")
            if status == "limit_reached":
                reason = raw_cell.get("error")
                reason_text = (
                    reason
                    if isinstance(reason, str) and reason.strip()
                    else "Recola generation limit reached"
                )
                cell_payload = {
                    "status": "failed",
                    "label": RECOLA_LABEL,
                    "failure_category": _recola_failure_category(reason_text),
                    "failure_reason": reason_text,
                }
            else:
                metrics = _recola_metrics(raw_cell)
                if not metrics:
                    continue
                observed_workload = _recola_cell_workload(
                    raw_cell,
                    config,
                    context=context,
                )
                if observed_workload != expected_workload:
                    raise PlotError(
                        f"{context} is {observed_workload} workload but the "
                        f"campaign report is {expected_workload}"
                    )
                cell_payload = {
                    "status": "measured",
                    "label": RECOLA_LABEL,
                    "metrics": metrics,
                    "helicity_workload": observed_workload,
                    "warm_fixed_helicity": observed_workload == "fixed",
                    "warm_helicity_sum": observed_workload == "sum",
                    "process": raw_cell.get("process"),
                    "helicity": raw_cell.get("helicity"),
                    "profiled_call": raw_cell.get("profiled_call"),
                }
                measured_count += 1
            series.setdefault(target_family, {})[str(multiplicity)] = cell_payload
            multiplicities_by_family[target_family].append(multiplicity)

    if measured_count == 0:
        raise PlotError("Recola results contain no measured cells")

    runtime_series = report.setdefault("runtime_series", {})
    if not isinstance(runtime_series, dict):
        raise PlotError("report runtime_series must be an object when present")
    for family, cells in series.items():
        family_runtime = runtime_series.setdefault(family, {})
        if not isinstance(family_runtime, dict):
            raise PlotError(f"report runtime_series.{family} must be an object")
        family_cells = _mapping(_mapping(report.get("cells")).get(family))
        if RECOLA_MODE in family_runtime or RECOLA_MODE in family_cells:
            raise PlotError(f"external series duplicates {family}/{RECOLA_MODE}")
        family_runtime[RECOLA_MODE] = {
            raw_n: cells[raw_n] for raw_n in sorted(cells, key=int)
        }

    return _recola_source_metadata(
        path,
        digest,
        {
            family: sorted(set(multiplicities))
            for family, multiplicities in multiplicities_by_family.items()
            if multiplicities
        },
    )


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
    *,
    visible_modes: Sequence[str] | None = None,
) -> list[str]:
    visible_mode_set = set(visible_modes) if visible_modes is not None else None
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
        if visible_mode_set is not None and mode not in visible_mode_set:
            continue
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
        if visible_mode_set is None or mode in visible_mode_set
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


def _line_selector_modes(lines: Sequence[str]) -> set[str]:
    modes: set[str] = set()
    for line in lines:
        modes.update(PLOT_LINE_SELECTOR_MODES[line])
    return modes


def _filtered_modes(
    modes: Sequence[str],
    *,
    include_lines: Sequence[str] | None,
    veto_lines: Sequence[str],
) -> tuple[str, ...]:
    if include_lines is not None:
        selected = _line_selector_modes(include_lines)
        return tuple(mode for mode in modes if mode in selected)
    vetoed = _line_selector_modes(veto_lines)
    return tuple(mode for mode in modes if mode not in vetoed)


def _visible_modes(
    modes: Sequence[str],
    *,
    line_filters: LineFilterOptions,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    main_modes = _filtered_modes(
        modes,
        include_lines=line_filters.main_include_lines,
        veto_lines=line_filters.main_veto_lines,
    )
    ratio_modes = _filtered_modes(
        modes,
        include_lines=line_filters.ratio_include_lines,
        veto_lines=line_filters.ratio_veto_lines,
    )
    visible = tuple(
        mode for mode in modes if mode in set(main_modes) or mode in set(ratio_modes)
    )
    return main_modes, ratio_modes, visible


def _line_filter_payload(line_filters: LineFilterOptions) -> dict[str, Any]:
    return {
        "main": {
            "include_lines": (
                None
                if line_filters.main_include_lines is None
                else list(line_filters.main_include_lines)
            ),
            "veto_lines": list(line_filters.main_veto_lines),
        },
        "ratio": {
            "include_lines": (
                None
                if line_filters.ratio_include_lines is None
                else list(line_filters.ratio_include_lines)
            ),
            "veto_lines": list(line_filters.ratio_veto_lines),
        },
        "precedence": "include-lines override veto-lines per panel",
    }


def _log_limits(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        return low / 2.0, high * 2.0
    span = math.log10(high) - math.log10(low)
    padding = max(0.10, span * 0.06)
    return 10 ** (math.log10(low) - padding), 10 ** (math.log10(high) + padding)


def _linear_limits(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        padding = max(abs(low) * 0.5, 0.5)
    else:
        padding = max((high - low) * 0.08, 0.05)
    return low - padding, high + padding


def _main_y_limits(
    values: Sequence[float], axis_options: AxisOptions
) -> tuple[float, float]:
    if axis_options.main_y_range is not None:
        return axis_options.main_y_range
    if values:
        return _log_limits(values)
    return 0.5, 2.0


def _ratio_y_limits(
    values: Sequence[float], axis_options: AxisOptions
) -> tuple[float, float]:
    if axis_options.ratio_y_range is not None:
        return axis_options.ratio_y_range
    scale_values = [*values, 1.0]
    if axis_options.ratio_y_scale == "linear":
        return _linear_limits(scale_values)
    return _log_limits(scale_values)


def _axis_payload(
    *,
    scale: str,
    limits: tuple[float, float],
) -> dict[str, Any]:
    return {"scale": scale, "range": [limits[0], limits[1]]}


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
    axis_options: AxisOptions = DEFAULT_AXIS_OPTIONS,
    line_filters: LineFilterOptions = DEFAULT_LINE_FILTER_OPTIONS,
) -> None:
    original_family_cells = _mapping(_mapping(report.get("cells")).get(family))
    family_cells, modes = _family_cells_for_metric(report, family, metric)
    main_modes, ratio_modes, visible_modes = _visible_modes(
        modes,
        line_filters=line_filters,
    )
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
    all_main_values = [value for values in series.values() for _, value in values]
    main_values = [
        value for mode in main_modes for _, value in series.get(mode, ())
    ]
    if not all_main_values and not allow_empty:
        raise PlotError(f"no measured positive {metric.key} values for {family}")
    baseline_values = dict(series.get(baseline, ()))
    ratios: dict[str, list[tuple[int, float]]] = {}
    for mode, values in series.items():
        ratios[mode] = [
            (n, value / baseline_values[n])
            for n, value in values
            if n in baseline_values and baseline_values[n] > 0.0
        ]
    all_ratio_values = [value for values in ratios.values() for _, value in values]
    ratio_values = [
        value for mode in ratio_modes for _, value in ratios.get(mode, ())
    ]
    if not all_ratio_values and not allow_empty:
        raise PlotError(f"no ratios can be formed for {family} {metric.key}")

    notes = _wrapped_notes(
        [
            *_status_notes(report, family, visible_modes, family_cells=family_cells),
            *_plot_policy_notes(
                report,
                family,
                metric,
                original_family_cells,
                visible_modes=visible_modes,
            ),
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
        if mode in visible_modes and (series[mode] or ratios[mode]):
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
        if mode in main_modes:
            for run in _consecutive_runs(series[mode]):
                _draw_run(
                    main_axis,
                    run,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )
        if mode in ratio_modes:
            for run in _consecutive_runs(ratios[mode]):
                _draw_run(
                    ratio_axis,
                    run,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )

    main_axis.set_yscale("log")
    main_axis.set_ylim(*_main_y_limits(main_values, axis_options))
    if not main_values:
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
    ratio_axis.set_yscale(axis_options.ratio_y_scale)
    ratio_axis.axhline(1.0, color="#59636E", linewidth=0.9, linestyle=":", zorder=1)
    ratio_axis.set_ylim(*_ratio_y_limits(ratio_values, axis_options))
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
    protocol_lines = _protocol_summary_lines(
        report,
        ratio_y_scale=axis_options.ratio_y_scale,
    )
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
            ncol=4 if family == "gg" or len(labels) > 6 else 3,
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


def _point_payload(point: tuple[int, float]) -> dict[str, float | int]:
    return {"n": point[0], "value": point[1]}


def _source_boundary_payload(
    report: Mapping[str, Any], metric: Metric
) -> dict[str, float | str] | None:
    boundary = _source_boundary(report, metric.key)
    if boundary is None:
        return None
    x_position, label = boundary
    return {"x": x_position, "label": label}


def _metric_data_payload(
    report: Mapping[str, Any],
    family: str,
    metric: Metric,
    *,
    axis_options: AxisOptions = DEFAULT_AXIS_OPTIONS,
    line_filters: LineFilterOptions = DEFAULT_LINE_FILTER_OPTIONS,
) -> dict[str, Any]:
    family_cells, modes = _family_cells_for_metric(report, family, metric)
    main_modes, ratio_modes, visible_modes = _visible_modes(
        modes,
        line_filters=line_filters,
    )
    series = {mode: _series(family_cells, mode, metric) for mode in modes}
    main_values = [
        value for mode in main_modes for _, value in series.get(mode, ())
    ]
    baseline = FAMILY_BASELINES[family]
    baseline_values = dict(series.get(baseline, ()))
    mode_payloads: dict[str, Any] = {}
    ratio_values: list[float] = []
    for mode in visible_modes:
        values = series[mode]
        ratios = [
            (n, value / baseline_values[n])
            for n, value in values
            if n in baseline_values and baseline_values[n] > 0.0
        ]
        if mode in ratio_modes:
            ratio_values.extend(value for _, value in ratios)
        mode_payloads[mode] = {
            "label": _label(family_cells, mode),
            "visible_in_main": mode in main_modes,
            "visible_in_ratio": mode in ratio_modes,
            "series": (
                [_point_payload(point) for point in values]
                if mode in main_modes
                else []
            ),
            "ratio_to_baseline": (
                [_point_payload(point) for point in ratios]
                if mode in ratio_modes
                else []
            ),
        }
    return {
        "source_metric_key": metric.key,
        "slug": metric.slug,
        "title": _metric_title(report, metric),
        "axis_label": _metric_axis_label(report, metric),
        "scale_from_report": metric.scale,
        "source_boundary": _source_boundary_payload(report, metric),
        "baseline_mode": baseline,
        "main_axis": _axis_payload(
            scale="log",
            limits=_main_y_limits(main_values, axis_options),
        ),
        "ratio_axis": _axis_payload(
            scale=axis_options.ratio_y_scale,
            limits=_ratio_y_limits(ratio_values, axis_options),
        ),
        "modes": mode_payloads,
    }


def _plot_data_payload(
    report_sha256: str,
    report: Mapping[str, Any],
    *,
    axis_options: AxisOptions = DEFAULT_AXIS_OPTIONS,
    line_filters: LineFilterOptions = DEFAULT_LINE_FILTER_OPTIONS,
    external_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": "pyamplicol-fft-scaling-plot-data",
        "schema_version": 1,
        "report_sha256": report_sha256,
        "helicity_workload": _helicity_workload(report),
        "line_filters": _line_filter_payload(line_filters),
        "families": {
            family: {
                "title": FAMILY_TITLES[family],
                "metrics": {
                    metric.slug: _metric_data_payload(
                        report,
                        family,
                        metric,
                        axis_options=axis_options,
                        line_filters=line_filters,
                    )
                    for metric in METRICS
                },
            }
            for family in FAMILIES
        },
    }
    if external_sources:
        payload["external_sources"] = dict(external_sources)
    return payload


def _write_plot_data(
    report_sha256: str,
    report: Mapping[str, Any],
    output_directory: Path,
    *,
    axis_options: AxisOptions = DEFAULT_AXIS_OPTIONS,
    line_filters: LineFilterOptions = DEFAULT_LINE_FILTER_OPTIONS,
    external_sources: Mapping[str, Any] | None = None,
) -> Path:
    data_path = output_directory / PLOT_DATA_NAME
    temporary_path = data_path.with_suffix(data_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _plot_data_payload(
                report_sha256,
                report,
                axis_options=axis_options,
                line_filters=line_filters,
                external_sources=external_sources,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(data_path)
    return data_path


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


def _render(
    report: Mapping[str, Any],
    output_directory: Path,
    *,
    dpi: int,
    axis_options: AxisOptions = DEFAULT_AXIS_OPTIONS,
    line_filters: LineFilterOptions = DEFAULT_LINE_FILTER_OPTIONS,
) -> None:
    if dpi < 72 or dpi > 600:
        raise PlotError("--dpi must be between 72 and 600")
    output_directory.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    for family in FAMILIES:
        for metric in METRICS:
            output_path = output_directory / f"fullcolor-{family}-{metric.slug}.png"
            _plot_metric(
                report,
                family,
                metric,
                output_path,
                dpi=dpi,
                axis_options=axis_options,
                line_filters=line_filters,
            )
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
    *,
    external_sources: Mapping[str, Any] | None = None,
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
    if external_sources:
        manifest["external_sources"] = dict(external_sources)
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
        axis_options = _axis_options_from_arguments(arguments)
        line_filters = _line_filter_options_from_arguments(arguments)
        report, report_sha256 = _load_report_snapshot(arguments.report)
        external_sources: dict[str, Any] = {}
        if arguments.recola_results is not None:
            external_sources["recola_results"] = _attach_recola_results(
                report,
                arguments.recola_results,
            )
        _render(
            report,
            arguments.output_directory,
            dpi=arguments.dpi,
            axis_options=axis_options,
            line_filters=line_filters,
        )
        _write_plot_manifest(
            report_sha256,
            report,
            arguments.output_directory,
            external_sources=external_sources,
        )
        _write_plot_data(
            report_sha256,
            report,
            arguments.output_directory,
            axis_options=axis_options,
            line_filters=line_filters,
            external_sources=external_sources,
        )
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
