# SPDX-License-Identifier: 0BSD
"""Lazy compatibility facade for built-in tensor-network lowering."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.builtin import lowering_tensor as _builtin_lowering_tensor

    _GraphTensorExpressionBuilder = (
        _builtin_lowering_tensor._GraphTensorExpressionBuilder
    )
    build_interleaved_tensor_network_scalar_bundle = (
        _builtin_lowering_tensor.build_interleaved_tensor_network_scalar_bundle
    )
    build_tensor_network_scalar_bundle = (
        _builtin_lowering_tensor.build_tensor_network_scalar_bundle
    )

__all__ = [
    "build_interleaved_tensor_network_scalar_bundle",
    "build_tensor_network_scalar_bundle",
]


def _builtin_module() -> ModuleType:
    return import_module("..models.builtin.lowering_tensor", __package__)


def __getattr__(name: str) -> Any:
    value = getattr(_builtin_module(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_builtin_module())})
