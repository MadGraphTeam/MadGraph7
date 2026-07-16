# SPDX-License-Identifier: 0BSD
"""Compatibility facade for symbolic recursion lowering."""

from __future__ import annotations

from .lowering_reports import build_symbolic_lowering_report
from .lowering_tensor import (
    _GraphTensorExpressionBuilder as _GraphTensorExpressionBuilder,
)
from .lowering_tensor import (
    build_interleaved_tensor_network_scalar_bundle,
    build_tensor_network_scalar_bundle,
)
from .lowering_types import (
    ColorAlgebraProbe,
    RecursionLoweringPlan,
    SymbolicLoweringReport,
    TensorNetworkBlueprint,
    TensorNetworkProbe,
    VertexLoweringReport,
    VertexLoweringStep,
)

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
