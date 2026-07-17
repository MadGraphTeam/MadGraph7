# SPDX-License-Identifier: 0BSD
"""Public Python interface for pyAmpliCol.

Importing this module intentionally does not import Symbolica or model tooling. Heavy
dependencies are loaded only when generation or runtime services are first used.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._internal.versions import package_version

if TYPE_CHECKING:
    from .api import (
        ArtifactError,
        BenchmarkComponentTiming,
        BenchmarkResult,
        BenchmarkRunner,
        BenchmarkStageTiming,
        BenchmarkStatistics,
        BenchmarkTimingBreakdown,
        ColorComponent,
        ColorFlow,
        CompatibilityError,
        CompiledModel,
        CompiledModelCapabilities,
        CompiledModelInfo,
        CompiledModelSource,
        ConfigurationError,
        ContractedColorComponent,
        DependencyError,
        EvaluationError,
        ExternalParticle,
        GenerationError,
        GenerationPlan,
        GenerationResult,
        Generator,
        HelicityConfiguration,
        ModelCompilationIssue,
        ModelCompilationPhase,
        ModelError,
        ModelParameter,
        ModelSource,
        PhysicsReduction,
        ProcessAlias,
        ProcessPhysics,
        ProcessRequest,
        ProcessSet,
        PyAmpliColError,
        ReductionGroup,
        ResolvedEvaluation,
        Runtime,
        benchmark,
        generate,
        load,
    )
    from .config import BenchmarkConfig, EvaluationConfig, GenerationConfig, RunConfig

__version__ = package_version()

__all__ = [
    "ArtifactError",
    "BenchmarkComponentTiming",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "ColorComponent",
    "ColorFlow",
    "CompatibilityError",
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ConfigurationError",
    "ContractedColorComponent",
    "DependencyError",
    "EvaluationConfig",
    "EvaluationError",
    "ExternalParticle",
    "GenerationConfig",
    "GenerationError",
    "GenerationPlan",
    "GenerationResult",
    "Generator",
    "HelicityConfiguration",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
    "ModelError",
    "ModelParameter",
    "ModelSource",
    "PhysicsReduction",
    "ProcessAlias",
    "ProcessPhysics",
    "ProcessRequest",
    "ProcessSet",
    "PyAmpliColError",
    "ReductionGroup",
    "ResolvedEvaluation",
    "RunConfig",
    "Runtime",
    "__version__",
    "benchmark",
    "generate",
    "load",
]

_ERROR_EXPORTS = (
    "ArtifactError",
    "CompatibilityError",
    "ConfigurationError",
    "DependencyError",
    "EvaluationError",
    "GenerationError",
    "ModelError",
    "PyAmpliColError",
)
_REQUEST_EXPORTS = (
    "ModelSource",
    "ProcessAlias",
    "ProcessRequest",
    "ProcessSet",
)
_MODEL_EXPORTS = (
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
)
_RESULT_EXPORTS = (
    "BenchmarkComponentTiming",
    "BenchmarkResult",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "ColorComponent",
    "ColorFlow",
    "ContractedColorComponent",
    "ExternalParticle",
    "GenerationPlan",
    "GenerationResult",
    "HelicityConfiguration",
    "ModelParameter",
    "PhysicsReduction",
    "ProcessPhysics",
    "ReductionGroup",
    "ResolvedEvaluation",
)
_SERVICE_EXPORTS = (
    "BenchmarkRunner",
    "Generator",
    "Runtime",
    "benchmark",
    "generate",
    "load",
)
_CONFIG_EXPORTS = (
    "BenchmarkConfig",
    "EvaluationConfig",
    "GenerationConfig",
    "RunConfig",
)
_PUBLIC_EXPORTS = {
    **{name: (".api.errors", name) for name in _ERROR_EXPORTS},
    **{name: (".api.models", name) for name in _MODEL_EXPORTS},
    **{name: (".api.requests", name) for name in _REQUEST_EXPORTS},
    **{name: (".api.results", name) for name in _RESULT_EXPORTS},
    **{name: (".api.services", name) for name in _SERVICE_EXPORTS},
    **{name: (".config", name) for name in _CONFIG_EXPORTS},
}


def __getattr__(name: str) -> Any:
    """Load a public object only when that object is requested."""

    export = _PUBLIC_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = export
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
