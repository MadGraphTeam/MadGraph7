# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyamplicol.models.loading import CompiledModel

    from .errors import (
        ArtifactError,
        CompatibilityError,
        ConfigurationError,
        DependencyError,
        EvaluationError,
        GenerationError,
        ModelError,
        PyAmpliColError,
    )
    from .protocols import (
        BenchmarkBackend,
        BenchmarkFactory,
        GeneratorBackend,
        GeneratorFactory,
        ModelParameters,
        Momenta,
        RuntimeBackend,
        RuntimeLoader,
    )
    from .requests import (
        ModelSource,
        ModelSourceKind,
        ProcessAlias,
        ProcessRequest,
        ProcessSet,
    )
    from .results import (
        BenchmarkResult,
        BenchmarkStatistics,
        ColorComponent,
        ColorFlow,
        ContractedColorComponent,
        ExternalParticle,
        GenerationPlan,
        GenerationResult,
        HelicityConfiguration,
        ModelParameter,
        PhysicsReduction,
        ProcessPhysics,
        ReductionGroup,
        ResolvedEvaluation,
    )
    from .services import (
        BenchmarkRunner,
        Generator,
        Runtime,
        benchmark,
        generate,
        install_backend_factories,
        load,
    )

__all__ = [
    "ArtifactError",
    "BenchmarkBackend",
    "BenchmarkFactory",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkStatistics",
    "ColorComponent",
    "ColorFlow",
    "CompatibilityError",
    "CompiledModel",
    "ConfigurationError",
    "ContractedColorComponent",
    "DependencyError",
    "EvaluationError",
    "ExternalParticle",
    "GenerationError",
    "GenerationPlan",
    "GenerationResult",
    "Generator",
    "GeneratorBackend",
    "GeneratorFactory",
    "HelicityConfiguration",
    "ModelError",
    "ModelParameter",
    "ModelParameters",
    "ModelSource",
    "ModelSourceKind",
    "Momenta",
    "PhysicsReduction",
    "ProcessAlias",
    "ProcessPhysics",
    "ProcessRequest",
    "ProcessSet",
    "PyAmpliColError",
    "ReductionGroup",
    "ResolvedEvaluation",
    "Runtime",
    "RuntimeBackend",
    "RuntimeLoader",
    "benchmark",
    "generate",
    "install_backend_factories",
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
_PROTOCOL_EXPORTS = (
    "BenchmarkBackend",
    "BenchmarkFactory",
    "GeneratorBackend",
    "GeneratorFactory",
    "ModelParameters",
    "Momenta",
    "RuntimeBackend",
    "RuntimeLoader",
)
_REQUEST_EXPORTS = (
    "ModelSource",
    "ModelSourceKind",
    "ProcessAlias",
    "ProcessRequest",
    "ProcessSet",
)
_RESULT_EXPORTS = (
    "BenchmarkResult",
    "BenchmarkStatistics",
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
    "install_backend_factories",
    "load",
)
_PUBLIC_EXPORTS = {
    **{name: (".errors", name) for name in _ERROR_EXPORTS},
    **{name: (".protocols", name) for name in _PROTOCOL_EXPORTS},
    **{name: (".requests", name) for name in _REQUEST_EXPORTS},
    **{name: (".results", name) for name in _RESULT_EXPORTS},
    **{name: (".services", name) for name in _SERVICE_EXPORTS},
    "CompiledModel": ("pyamplicol.models.loading", "CompiledModel"),
}


def __getattr__(name: str) -> Any:
    """Load a public object only when that object is requested."""

    export = _PUBLIC_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = export
    package = __name__ if module_name.startswith(".") else None
    value = getattr(import_module(module_name, package), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
