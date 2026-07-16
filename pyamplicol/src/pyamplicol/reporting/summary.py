# SPDX-License-Identifier: 0BSD
"""Compact PrettyTable summaries for human-facing CLI results."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast


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
            ", ".join(str(value) for value in plain["runtime_capabilities"]),
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
