# SPDX-License-Identifier: 0BSD
"""Generic lowering types with lazy built-in-SM compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .lowering_types import (
    ColorAlgebraProbe,
    RecursionLoweringPlan,
    SymbolicLoweringReport,
    TensorNetworkBlueprint,
    TensorNetworkProbe,
    VertexLoweringReport,
    VertexLoweringStep,
)

if TYPE_CHECKING:
    from ..models.builtin.lowering_reports import build_symbolic_lowering_report
    from ..models.builtin.lowering_tensor import (
        _GraphTensorExpressionBuilder as _GraphTensorExpressionBuilder,
    )
    from ..models.builtin.lowering_tensor import (
        build_interleaved_tensor_network_scalar_bundle,
        build_tensor_network_scalar_bundle,
    )

_LAZY_EXPORTS = {
    "_GraphTensorExpressionBuilder": (
        "..models.builtin.lowering_tensor",
        "_GraphTensorExpressionBuilder",
    ),
    "build_interleaved_tensor_network_scalar_bundle": (
        "..models.builtin.lowering_tensor",
        "build_interleaved_tensor_network_scalar_bundle",
    ),
    "build_symbolic_lowering_report": (
        "..models.builtin.lowering_reports",
        "build_symbolic_lowering_report",
    ),
    "build_tensor_network_scalar_bundle": (
        "..models.builtin.lowering_tensor",
        "build_tensor_network_scalar_bundle",
    ),
}

__all__ = [
    "ColorAlgebraProbe",
    "RecursionLoweringPlan",
    "SymbolicLoweringReport",
    "TensorNetworkBlueprint",
    "TensorNetworkProbe",
    "VertexLoweringReport",
    "VertexLoweringStep",
    "build_interleaved_tensor_network_scalar_bundle",
    "build_symbolic_lowering_report",
    "build_tensor_network_scalar_bundle",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __package__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
