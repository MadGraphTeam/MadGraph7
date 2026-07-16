# SPDX-License-Identifier: 0BSD
"""Lazy compatibility facade for built-in symbolic-lowering reports."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.builtin import lowering_reports as _builtin_lowering_reports

    build_symbolic_lowering_report = (
        _builtin_lowering_reports.build_symbolic_lowering_report
    )

__all__ = ["build_symbolic_lowering_report"]


def _builtin_module() -> ModuleType:
    return import_module("..models.builtin.lowering_reports", __package__)


def __getattr__(name: str) -> Any:
    value = getattr(_builtin_module(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_builtin_module())})
