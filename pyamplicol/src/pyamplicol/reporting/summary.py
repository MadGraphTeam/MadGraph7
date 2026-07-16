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


def render_summary(value: object, *, color: bool = False) -> str | None:
    """Return a two-column table for structured results, if applicable."""

    plain = _plain(value)
    if not isinstance(plain, Mapping):
        return None
    try:
        prettytable: Any = importlib.import_module("prettytable")
    except ImportError:
        return None
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
