# SPDX-License-Identifier: 0BSD
"""Public Python interface for pyAmpliCol.

The package's Symbolica library key is tried before any symbolic work.
Personal licenses and restricted mode remain available if registration fails.
Model tooling is loaded only when generation or runtime services are first used.
"""

from __future__ import annotations

import os as _os
from contextlib import suppress as _suppress
from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import _native_loader

# First: pyamplicol._rusticol is loaded only after the running glibc is checked
# against the library's ELF version requirements (NativePlatformError).
_native_loader.install()

from ._internal.versions import package_version  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType as _ModuleType

# Registration precedes symbolic work, including in spawned Python workers.
# Keep startup quiet before CLI output/banner policy can be resolved, without
# changing the caller's banner or personal-license environment.
_previous_banner = _os.environ.get("SYMBOLICA_HIDE_BANNER")
_os.environ["SYMBOLICA_HIDE_BANNER"] = "1"
_symbolica: _ModuleType | None
try:
    try:
        import symbolica as _symbolica
    except ModuleNotFoundError as _error:
        if _error.name != "symbolica":
            raise
        # Keep configuration/source-only tooling usable without native deps.
        _symbolica = None
    except ImportError as _error:
        # A Symbolica build that needs a newer glibc: name both versions.
        _translated = _native_loader.translate_loader_error(_error)
        if _translated is None:
            raise
        raise _translated from _error

    _set_library_key = getattr(_symbolica, "set_library_key", None)
    if callable(_set_library_key):
        # Symbolica's public is_licensed() remains the authority: a personal
        # key can still license this process, otherwise use restricted mode.
        with _suppress(ValueError, PermissionError):
            _set_library_key(
                "SU-419-pyamplicol-2028.01.01-"
                "KJBO3RLZJ3H5FKU5CL756QLKZCURABBCKRGHZRK5OVIUQXZBCLJTOUQPHXYVTANIX"
                "SCKI6YTZDWGH2NWHQIBCPDUO5GVW2STVMP3MCQ"
            )
finally:
    if _previous_banner is None:
        _os.environ.pop("SYMBOLICA_HIDE_BANNER", None)
    else:
        _os.environ["SYMBOLICA_HIDE_BANNER"] = _previous_banner
del _os, _suppress, _previous_banner, _symbolica, _set_library_key

if TYPE_CHECKING:
    from .api import (
        ArtifactError,
        BenchmarkComponentTiming,
        BenchmarkProfileCounters,
        BenchmarkResult,
        BenchmarkRunner,
        BenchmarkStageTiming,
        BenchmarkStatistics,
        BenchmarkTimingBreakdown,
        BornAnchoredColorComponent,
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
        NativePlatformError,
        PhysicsReduction,
        ProcessAlias,
        ProcessPhysics,
        ProcessRequest,
        ProcessSet,
        PyAmpliColError,
        ReductionGroup,
        ResolvedEvaluation,
        Runtime,
        TreeOrderEvaluation,
        WarmUpResult,
        benchmark,
        generate,
        group_processes,
        load,
    )
    from .api.virtual_results import (
        LaurentCoefficients,
        ResolvedVirtualEvaluation,
        VirtualArithmeticSpec,
        VirtualColorContraction,
        VirtualEvaluation,
        VirtualFailureDiagnostic,
        VirtualLaneDiagnostic,
        VirtualStabilityPolicy,
        VirtualVerificationAttempt,
        VirtualVerificationDiagnostic,
    )
    from .config import (
        BenchmarkConfig,
        ColorConfig,
        ColorFFTBasis,
        EvaluationConfig,
        GenerationConfig,
        LoopConfig,
        R2Strategy,
        RunConfig,
    )
    from .correlators import (
        ColorCorrelator,
        CorrelatedRequest,
        CorrelatedValue,
        CorrelatorConfig,
        EmitGluon,
        SplitGluon,
    )

__version__ = package_version()

__all__ = [
    "ArtifactError",
    "BenchmarkComponentTiming",
    "BenchmarkConfig",
    "BenchmarkProfileCounters",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "BornAnchoredColorComponent",
    "ColorComponent",
    "ColorConfig",
    "ColorCorrelator",
    "ColorFFTBasis",
    "ColorFlow",
    "CompatibilityError",
    "CompiledModel",
    "CompiledModelCapabilities",
    "CompiledModelInfo",
    "CompiledModelSource",
    "ConfigurationError",
    "ContractedColorComponent",
    "CorrelatedRequest",
    "CorrelatedValue",
    "CorrelatorConfig",
    "DependencyError",
    "EmitGluon",
    "EvaluationConfig",
    "EvaluationError",
    "ExternalParticle",
    "GenerationConfig",
    "GenerationError",
    "GenerationPlan",
    "GenerationResult",
    "Generator",
    "HelicityConfiguration",
    "LaurentCoefficients",
    "LoopConfig",
    "ModelCompilationIssue",
    "ModelCompilationPhase",
    "ModelError",
    "ModelParameter",
    "ModelSource",
    "NativePlatformError",
    "PhysicsReduction",
    "ProcessAlias",
    "ProcessPhysics",
    "ProcessRequest",
    "ProcessSet",
    "PyAmpliColError",
    "R2Strategy",
    "ReductionGroup",
    "ResolvedEvaluation",
    "ResolvedVirtualEvaluation",
    "RunConfig",
    "Runtime",
    "SplitGluon",
    "TreeOrderEvaluation",
    "VirtualArithmeticSpec",
    "VirtualColorContraction",
    "VirtualEvaluation",
    "VirtualFailureDiagnostic",
    "VirtualLaneDiagnostic",
    "VirtualStabilityPolicy",
    "VirtualVerificationAttempt",
    "VirtualVerificationDiagnostic",
    "WarmUpResult",
    "__version__",
    "benchmark",
    "generate",
    "group_processes",
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
    "NativePlatformError",
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
    "BenchmarkProfileCounters",
    "BenchmarkResult",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "BornAnchoredColorComponent",
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
    "TreeOrderEvaluation",
    "WarmUpResult",
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
    "ColorConfig",
    "ColorFFTBasis",
    "EvaluationConfig",
    "GenerationConfig",
    "LoopConfig",
    "R2Strategy",
    "RunConfig",
)
_VIRTUAL_RESULT_EXPORTS = (
    "LaurentCoefficients",
    "ResolvedVirtualEvaluation",
    "VirtualEvaluation",
    "VirtualFailureDiagnostic",
    "VirtualLaneDiagnostic",
    "VirtualColorContraction",
    "VirtualArithmeticSpec",
    "VirtualStabilityPolicy",
    "VirtualVerificationAttempt",
    "VirtualVerificationDiagnostic",
)
_PUBLIC_EXPORTS = {
    **{name: (".api.virtual_results", name) for name in _VIRTUAL_RESULT_EXPORTS},
    **{
        name: (".correlators", name)
        for name in (
            "ColorCorrelator",
            "CorrelatedRequest",
            "CorrelatedValue",
            "CorrelatorConfig",
            "EmitGluon",
            "SplitGluon",
        )
    },
    **{name: (".api.errors", name) for name in _ERROR_EXPORTS},
    **{name: (".api.models", name) for name in _MODEL_EXPORTS},
    **{name: (".api.requests", name) for name in _REQUEST_EXPORTS},
    **{name: (".api.results", name) for name in _RESULT_EXPORTS},
    **{name: (".api.services", name) for name in _SERVICE_EXPORTS},
    **{name: (".config", name) for name in _CONFIG_EXPORTS},
    "group_processes": (".api.process_groups", "group_processes"),
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
