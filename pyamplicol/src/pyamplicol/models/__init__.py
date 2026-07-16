# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
from typing import Any

from .base import (
    CouplingOrders,
    Model,
    Particle,
    PropagatorLoweringRule,
    QuantumFlow,
    SourceSpinState,
    Vertex,
    VertexEvaluationEquivalence,
    VertexLoweringRule,
)
from .builtin import BuiltinSMModel

_LAZY_EXPORTS = {
    "CompiledModel": (".loading", "CompiledModel"),
    "CompiledUFOModel": (".external", "CompiledUFOModel"),
    "ModelCompileOptions": (".loading", "ModelCompileOptions"),
    "compile_builtin_model_ir": (".compiler", "compile_builtin_model_ir"),
    "compile_model_source": (".loading", "compile_model_source"),
    "compile_ufo_model_ir": (".compiler", "compile_ufo_model_ir"),
    "load_compiled_model": (".loading", "load_compiled_model"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BuiltinSMModel",
    "CompiledModel",
    "CompiledUFOModel",
    "CouplingOrders",
    "Model",
    "ModelCompileOptions",
    "Particle",
    "PropagatorLoweringRule",
    "QuantumFlow",
    "SourceSpinState",
    "Vertex",
    "VertexEvaluationEquivalence",
    "VertexLoweringRule",
    "compile_builtin_model_ir",
    "compile_model_source",
    "compile_ufo_model_ir",
    "load_compiled_model",
]
