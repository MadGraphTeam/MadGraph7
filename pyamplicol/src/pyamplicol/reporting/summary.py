# SPDX-License-Identifier: 0BSD
"""Compact PrettyTable summaries for human-facing CLI results."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyamplicol.api.results import (
        BenchmarkComponentTiming,
        BenchmarkResult,
        BenchmarkStatistics,
    )


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(entry) for entry in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _value_text(value: object) -> str:
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, sort_keys=True, separators=(", ", ": "))
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _colored(text: str, *, key: str, enabled: bool) -> str:
    if not enabled:
        return text
    normalized = text.casefold()
    try:
        colorama: Any = importlib.import_module("colorama")
    except ImportError:
        return text
    if normalized in {"yes", "ok", "passed", "complete", "completed", "valid"}:
        color = colorama.Fore.GREEN
    elif normalized in {"no", "failed", "error", "invalid"}:
        color = colorama.Fore.RED
    elif key.casefold() in {"warning", "warnings", "adjustments"}:
        color = colorama.Fore.YELLOW
    else:
        return text
    return f"{color}{text}{colorama.Style.RESET_ALL}"


def _paint(text: str, color_name: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    try:
        colorama: Any = importlib.import_module("colorama")
    except ImportError:
        return text
    return f"{getattr(colorama.Fore, color_name)}{text}{colorama.Style.RESET_ALL}"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _byte_size(value: object) -> str:
    size = _integer(value, "payload size")
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(size)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable byte-size unit")


def _component_count(process: Mapping[str, object], prefix: str) -> str:
    physical = _integer(process[f"physical_{prefix}"], f"physical {prefix}")
    computed = _integer(process[f"computed_{prefix}"], f"computed {prefix}")
    if physical == computed:
        return str(physical)
    return f"{physical} ({computed} eval.)"


def _duration_unit(seconds: float) -> tuple[float, str]:
    magnitude = abs(seconds)
    if magnitude < 1.0e-6:
        return 1.0e9, "ns"
    if magnitude < 1.0e-3:
        return 1.0e6, "us"
    if magnitude < 1.0:
        return 1.0e3, "ms"
    return 1.0, "s"


def _timing_text(
    seconds: float,
    uncertainty: BenchmarkStatistics,
) -> str:
    scale, unit = _duration_unit(seconds)
    return (
        f"{seconds * scale:.6g} +/- "
        f"{uncertainty.standard_error * scale:.3g} {unit}/point (standard error)"
    )


def _benchmark_timing_text(
    seconds: float,
    uncertainty: BenchmarkStatistics,
    *,
    sample_count: int,
) -> str:
    if sample_count > 1:
        return _timing_text(seconds, uncertainty)
    return f"{_seconds_text(seconds)}/point (uncertainty needs at least 2 blocks)"


def _seconds_text(seconds: float) -> str:
    scale, unit = _duration_unit(seconds)
    return f"{seconds * scale:.6g} {unit}"


def _component_timing_text(timing: BenchmarkComponentTiming | None) -> str:
    if timing is None:
        return "N/A"
    scale, unit = _duration_unit(timing.mean_seconds_per_point)
    if timing.sample_count < 2:
        return f"{timing.mean_seconds_per_point * scale:.6g} {unit}/point (SE pending)"
    return (
        f"{timing.mean_seconds_per_point * scale:.6g} +/- "
        f"{timing.uncertainty.standard_error * scale:.3g} {unit}/point"
    )


def _benchmark_summary(
    result: BenchmarkResult,
    *,
    prettytable: Any,
    color: bool,
) -> str:
    environment = result.environment
    process = result.process_id or "N/A"
    if result.process_expression is not None:
        process = f"{process} ({result.process_expression})"
    elapsed_value = environment.get("elapsed_seconds")
    elapsed = (
        float(elapsed_value)
        if isinstance(elapsed_value, (float, int))
        and not isinstance(elapsed_value, bool)
        else None
    )
    evaluator_source = str(environment.get("evaluator_time_source", "unavailable"))
    evaluator_text = "N/A"
    if result.evaluator_time_per_point is not None:
        if result.evaluator_uncertainty is None:
            evaluator_text = f"{_seconds_text(result.evaluator_time_per_point)}/point"
        else:
            evaluator_text = _benchmark_timing_text(
                result.evaluator_time_per_point,
                result.evaluator_uncertainty,
                sample_count=result.sample_count,
            )

    relative_error = result.uncertainty.relative_standard_error
    if relative_error <= 0.01:
        uncertainty_color = "GREEN"
    elif relative_error <= 0.05:
        uncertainty_color = "YELLOW"
    else:
        uncertainty_color = "RED"
    status = (
        f"interrupted - partial statistics from {result.sample_count} complete blocks"
        if result.interrupted
        else "complete"
    )
    rows: list[tuple[str, str, str | None]] = [
        ("process", process, None),
        ("artifact", _value_text(environment.get("target")), None),
        ("status", status, "YELLOW" if result.interrupted else "GREEN"),
        (
            "wall time",
            _benchmark_timing_text(
                result.wall_time_per_point,
                result.uncertainty,
                sample_count=result.sample_count,
            ),
            "GREEN",
        ),
        ("evaluator time", evaluator_text, "CYAN"),
        (
            "wall variability",
            (
                (
                    f"SD {_seconds_text(result.uncertainty.standard_deviation)}/point; "
                    f"relative SE {relative_error:.3%}"
                )
                if result.sample_count > 1
                else "not estimable from one complete block"
            ),
            uncertainty_color if result.sample_count > 1 else "YELLOW",
        ),
        (
            "sampling",
            (
                f"{result.sample_count} blocks x "
                f"{result.repetitions_per_sample} repetitions x "
                f"{result.effective_config.batch_size} points"
            ),
            None,
        ),
        ("timed evaluations", str(result.evaluation_count), None),
        ("timed points", str(result.evaluated_point_count), None),
        (
            "target / measured",
            (
                f"{result.effective_config.target_runtime:.6g} s / {elapsed:.6g} s"
                if elapsed is not None
                else f"{result.effective_config.target_runtime:.6g} s / N/A"
            ),
            None,
        ),
        ("precision", str(result.effective_config.precision), None),
        (
            "timing sources",
            (
                f"wall={environment.get('wall_time_source', 'unavailable')}; "
                f"evaluator={evaluator_source}"
            ),
            None,
        ),
        ("platform", _value_text(environment.get("platform")), None),
    ]

    table = prettytable.PrettyTable(("metric", "value"))
    table.title = _paint("Runtime Profile", "CYAN", enabled=color)
    table.align["metric"] = "l"
    table.align["value"] = "l"
    table.max_width["metric"] = 24
    table.max_width["value"] = 96
    table.hrules = prettytable.HRuleStyle.HEADER
    for metric, value, color_name in rows:
        table.add_row(
            (
                metric,
                _paint(value, color_name, enabled=color)
                if color_name is not None
                else value,
            )
        )
    sections = [cast(str, table.get_string())]
    breakdown = result.timing_breakdown
    if breakdown is None:
        return sections[0]

    component_table = prettytable.PrettyTable(("component", "mean +/- SE", "rel. SE"))
    eager_profile = breakdown.execution_mode == "eager"
    component_table.title = _paint(
        "Rusticol Eager Timing Breakdown"
        if eager_profile
        else "Rusticol Timing Breakdown",
        "CYAN",
        enabled=color,
    )
    component_table.align["component"] = "l"
    component_table.align["mean +/- SE"] = "r"
    component_table.align["rel. SE"] = "r"
    component_table.hrules = prettytable.HRuleStyle.HEADER
    detailed_eager_profile = eager_profile and any(
        timing is not None
        for timing in (
            breakdown.eager_gather_time,
            breakdown.eager_kernel_call_time,
            breakdown.eager_scatter_finalization_time,
            breakdown.eager_closure_time,
        )
    )
    component_rows: tuple[
        tuple[str, BenchmarkComponentTiming | None], ...
    ]
    if detailed_eager_profile:
        component_rows = (
            ("Profile wall", breakdown.wall_time),
            ("Source fill", breakdown.source_fill_time),
            ("Momentum setup", breakdown.momentum_setup_time),
            ("Eager execution (aggregate)", breakdown.eager_execution_time),
            ("Gather", breakdown.eager_gather_time),
            ("Kernel calls", breakdown.eager_kernel_call_time),
            (
                "Scatter / finalization",
                breakdown.eager_scatter_finalization_time,
            ),
            ("Amplitude closure", breakdown.eager_closure_time),
            ("Reduction", breakdown.reduction_time),
        )
    elif eager_profile:
        component_rows = (
            ("Profile wall", breakdown.wall_time),
            ("Source fill", breakdown.source_fill_time),
            ("Momentum setup", breakdown.momentum_setup_time),
            ("Eager execution (aggregate)", breakdown.eager_execution_time),
        )
    else:
        component_rows = (
            ("Profile wall", breakdown.wall_time),
            ("Source fill", breakdown.source_fill_time),
            ("Momentum setup", breakdown.momentum_setup_time),
            ("Stage input pack", breakdown.stage_input_pack_time),
            ("Stage evaluator calls", breakdown.stage_evaluator_call_time),
            ("Output assign", breakdown.output_assign_time),
            ("Amplitude input pack", breakdown.amplitude_input_pack_time),
            ("Amplitude evaluator call", breakdown.amplitude_evaluator_call_time),
            ("Reduction", breakdown.reduction_time),
        )
    for label, timing in component_rows:
        relative = (
            "N/A"
            if timing is None
            else f"{timing.uncertainty.relative_standard_error:.3%}"
        )
        value = _component_timing_text(timing)
        if label in {
            "Stage evaluator calls",
            "Amplitude evaluator call",
            "Eager execution (aggregate)",
            "Kernel calls",
        }:
            value = _paint(value, "CYAN", enabled=color)
        component_table.add_row((label, value, relative))
    sections.append(cast(str, component_table.get_string()))

    if breakdown.stages:
        stage_table = prettytable.PrettyTable(
            ("stage", "input pack", "evaluator call", "output assign")
        )
        stage_table.title = _paint("Rusticol Stage Detail", "CYAN", enabled=color)
        stage_table.align["stage"] = "r"
        for column in ("input pack", "evaluator call", "output assign"):
            stage_table.align[column] = "r"
        stage_table.hrules = prettytable.HRuleStyle.HEADER
        for stage in breakdown.stages:
            stage_table.add_row(
                (
                    stage.stage_index,
                    _component_timing_text(stage.input_pack_time),
                    _paint(
                        _component_timing_text(stage.evaluator_call_time),
                        "CYAN",
                        enabled=color,
                    ),
                    _component_timing_text(stage.output_assign_time),
                )
            )
        sections.append(cast(str, stage_table.get_string()))
    return "\n\n".join(sections)


def _artifact_inspection_summary(
    plain: Mapping[str, object],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    processes = cast(list[Mapping[str, object]], plain["processes"])
    dependencies = cast(list[Mapping[str, object]], plain["dependencies"])
    aliases = [
        cast(Mapping[str, object], alias)
        for process in processes
        for alias in cast(list[object], process["aliases"])
    ]

    model = str(plain["model_name"])
    restriction = plain.get("model_restriction")
    if restriction is not None:
        model = f"{model} ({restriction})"
    target = str(plain["target"])
    features = cast(list[object], plain["cpu_features"])
    if features:
        target = f"{target} [{', '.join(str(feature) for feature in features)}]"
    runtime_capabilities = cast(
        Sequence[object],
        plain["runtime_capabilities"],
    )

    summary = prettytable.PrettyTable(("field", "value"))
    summary.align["field"] = "l"
    summary.align["value"] = "l"
    summary.max_width["field"] = 24
    summary.max_width["value"] = 88
    summary.hrules = prettytable.HRuleStyle.HEADER
    summary_rows = (
        ("path", plain["path"]),
        ("artifact type", str(plain["artifact_kind"]).removeprefix("pyamplicol-")),
        ("artifact ID", plain["artifact_id"]),
        ("created", plain["created_utc"]),
        ("producer", f"pyamplicol {plain['producer_version']}"),
        ("target", target),
        ("model", f"{model}; {plain['model_source']}"),
        ("default process", plain.get("default_process_id")),
        ("contents", f"{len(processes)} processes, {len(aliases)} aliases"),
        (
            "runtime",
            f"{plain['runtime_engine']} {plain['runtime_version']}",
        ),
        (
            "capabilities",
            ", ".join(str(value) for value in runtime_capabilities),
        ),
        (
            "payloads",
            (
                f"{plain['payload_count']} files; "
                f"{_byte_size(plain['payload_size_bytes'])}"
            ),
        ),
        ("integrity", plain["integrity"]),
    )
    for key, value in summary_rows:
        text = _value_text(value)
        summary.add_row((key, _colored(text, key=key, enabled=color)))

    process_table = prettytable.PrettyTable(
        (
            "default",
            "stable ID",
            "concrete process",
            "color",
            "helicities",
            "color outputs",
            "coverage (hel./color)",
            "aliases",
        )
    )
    process_table.align = "l"
    process_table.align["default"] = "c"
    process_table.align["color"] = "c"
    process_table.align["helicities"] = "r"
    process_table.align["color outputs"] = "r"
    process_table.align["aliases"] = "r"
    process_table.max_width["stable ID"] = 30
    process_table.max_width["concrete process"] = 48
    process_table.hrules = prettytable.HRuleStyle.HEADER
    accuracy_colors = {"lc": "CYAN", "nlc": "YELLOW", "full": "MAGENTA"}
    for process in processes:
        accuracy = str(process["color_accuracy"])
        marker = "*" if bool(process["default"]) else ""
        coverage = f"{process['helicity_coverage']} / {process['color_coverage']}"
        coverage_text = (
            _paint(coverage, "GREEN", enabled=color)
            if coverage == "complete / complete"
            else coverage
        )
        process_table.add_row(
            (
                _paint(marker, "GREEN", enabled=color),
                process["id"],
                process["expression"],
                _paint(
                    accuracy,
                    accuracy_colors.get(accuracy, "WHITE"),
                    enabled=color,
                ),
                _component_count(process, "helicities"),
                _component_count(process, "color_components"),
                coverage_text,
                len(cast(list[object], process["aliases"])),
            )
        )

    sections = [
        _paint("Artifact", "CYAN", enabled=color),
        cast(str, summary.get_string()),
        _paint("Processes", "CYAN", enabled=color),
        cast(str, process_table.get_string()),
    ]

    execution_table = prettytable.PrettyTable(("process", "execution field", "value"))
    execution_table.align = "l"
    execution_table.max_width["process"] = 30
    execution_table.max_width["execution field"] = 26
    execution_table.max_width["value"] = 72
    execution_table.hrules = prettytable.HRuleStyle.HEADER
    for process in processes:
        process_id = str(process["id"])
        mode = str(process.get("execution_mode", "compiled"))
        backend = process.get("prepared_backend")
        mode_text = mode if backend is None else f"{mode} ({backend})"
        execution_table.add_row((process_id, "mode / backend", mode_text))
        phases = cast(Sequence[object], process.get("native_profile_phases", ()))
        execution_table.add_row(
            (
                process_id,
                "native profile phases",
                ", ".join(str(phase) for phase in phases) or "unavailable",
            )
        )
        if mode != "eager":
            continue
        pack_kernels = process.get("prepared_kernel_count")
        referenced_kernels = process.get("referenced_kernel_count")
        execution_table.add_row(
            (
                process_id,
                "prepared kernels",
                f"{pack_kernels} in pack; {referenced_kernels} referenced",
            )
        )
        invocation_count = process.get("invocation_count")
        attachment_count = process.get("attachment_count")
        alias_count = process.get("evaluation_alias_count")
        maximum_fanout = process.get("maximum_fanout")
        execution_table.add_row(
            (
                process_id,
                "invocations / reuse",
                (
                    f"{invocation_count} canonical; {attachment_count} attachments; "
                    f"{alias_count} reused aliases; max fanout {maximum_fanout}"
                ),
            )
        )
        execution_table.add_row(
            (
                process_id,
                "finalization / closure",
                (
                    f"{process.get('finalization_count')} currents; "
                    f"{process.get('closure_count')} amplitudes"
                ),
            )
        )
        execution_table.add_row(
            (
                process_id,
                "selector closures",
                (
                    "available"
                    if bool(process.get("selector_closure_available"))
                    else "not emitted"
                ),
            )
        )
        requested_tile = process.get("requested_point_tile_size")
        effective_tile = process.get("effective_point_tile_size")
        execution_table.add_row(
            (
                process_id,
                "point tile requested / effective",
                (
                    f"{requested_tile} / "
                    + (
                        str(effective_tile)
                        if effective_tile is not None
                        else "available after runtime load"
                    )
                ),
            )
        )
        workspace_limit = process.get("workspace_limit_bytes")
        workspace_used = process.get("workspace_bytes")
        execution_table.add_row(
            (
                process_id,
                "workspace limit / allocated",
                (
                    f"{_byte_size(workspace_limit)} / "
                    + (
                        _byte_size(workspace_used)
                        if workspace_used is not None
                        else "available after runtime load"
                    )
                ),
            )
        )
    sections.extend(
        (
            _paint("Execution", "CYAN", enabled=color),
            cast(str, execution_table.get_string()),
        )
    )

    if aliases:
        alias_table = prettytable.PrettyTable(
            ("stable alias ID", "concrete process", "representative ID")
        )
        alias_table.align = "l"
        alias_table.max_width["stable alias ID"] = 30
        alias_table.max_width["concrete process"] = 52
        alias_table.max_width["representative ID"] = 30
        alias_table.hrules = prettytable.HRuleStyle.HEADER
        for alias in aliases:
            alias_table.add_row(
                (alias["id"], alias["expression"], alias["representative_id"])
            )
        sections.extend(
            (
                _paint("Aliases", "CYAN", enabled=color),
                cast(str, alias_table.get_string()),
            )
        )

    if dependencies:
        dependency_table = prettytable.PrettyTable(("dependency", "version", "license"))
        dependency_table.align = "l"
        dependency_table.max_width["license"] = 48
        dependency_table.hrules = prettytable.HRuleStyle.HEADER
        for dependency in dependencies:
            dependency_table.add_row(
                (dependency["name"], dependency["version"], dependency["license"])
            )
        sections.extend(
            (
                _paint("Dependencies", "CYAN", enabled=color),
                cast(str, dependency_table.get_string()),
            )
        )
    return "\n\n".join(sections)


def render_summary(value: object, *, color: bool = False) -> str | None:
    """Return a two-column table for structured results, if applicable."""

    plain = _plain(value)
    if not isinstance(plain, Mapping):
        return None
    try:
        prettytable: Any = importlib.import_module("prettytable")
    except ImportError:
        return None
    from pyamplicol.api.results import BenchmarkResult

    if isinstance(value, BenchmarkResult):
        return _benchmark_summary(value, prettytable=prettytable, color=color)
    if plain.get("kind") == "pyamplicol-artifact-inspection":
        return _artifact_inspection_summary(
            plain,
            prettytable=prettytable,
            color=color,
        )
    table = prettytable.PrettyTable(("field", "value"))
    table.align["field"] = "l"
    table.align["value"] = "l"
    table.max_width["field"] = 30
    table.max_width["value"] = 88
    table.hrules = prettytable.HRuleStyle.HEADER
    for key, entry in plain.items():
        text = _value_text(entry)
        table.add_row((key.replace("_", " "), _colored(text, key=key, enabled=color)))
    return cast(str, table.get_string())


__all__ = ["render_summary"]
