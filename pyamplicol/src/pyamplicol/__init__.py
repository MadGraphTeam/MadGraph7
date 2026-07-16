# SPDX-License-Identifier: 0BSD
"""Public Python interface for pyAmpliCol.

Importing this module intentionally does not import Symbolica or model tooling. Heavy
dependencies are loaded only when generation or runtime services are first used.
"""

from __future__ import annotations

import json
from importlib import resources
from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("pyamplicol")
except PackageNotFoundError:
    try:
        _build_info = json.loads(
            resources.files(__package__)
            .joinpath("_build_info.json")
            .read_text(encoding="utf-8")
        )
        __version__ = str(_build_info["version"])
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        __version__ = "0.1.0"

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
