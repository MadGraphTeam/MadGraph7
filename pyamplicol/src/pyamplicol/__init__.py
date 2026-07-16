# SPDX-License-Identifier: 0BSD
"""Public Python interface for pyAmpliCol.

Importing this module intentionally does not import Symbolica or model tooling. Heavy
dependencies are loaded only when generation or runtime services are first used.
"""

from __future__ import annotations

from ._internal.versions import package_version
from .api import (
    ArtifactError,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStatistics,
    ColorComponent,
    ColorFlow,
    CompatibilityError,
    CompiledModel,
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
    "BenchmarkConfig",
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
    "EvaluationConfig",
    "EvaluationError",
    "ExternalParticle",
    "GenerationConfig",
    "GenerationError",
    "GenerationPlan",
    "GenerationResult",
    "Generator",
    "HelicityConfiguration",
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
