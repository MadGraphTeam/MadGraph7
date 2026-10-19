# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from typing import TYPE_CHECKING

from pyamplicol.config.errors import ConfigurationError, PyAmpliColError

if TYPE_CHECKING:
    from .virtual_results import VirtualFailureDiagnostic, VirtualVerificationDiagnostic


class ModelError(PyAmpliColError):
    """A model source or compiled model is invalid."""


class GenerationError(PyAmpliColError):
    """Planning or artifact generation failed."""


class ArtifactError(PyAmpliColError):
    """An artifact is malformed, incomplete, or unavailable."""


class CompatibilityError(ArtifactError):
    """An artifact is incompatible with the active target or ABI."""


class EvaluationError(PyAmpliColError):
    """Runtime loading, selection, or evaluation failed."""

    def __init__(
        self,
        message: str,
        *,
        failures: tuple[VirtualFailureDiagnostic, ...] = (),
        verification: tuple[VirtualVerificationDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failures = failures
        self.verification = verification


class DependencyError(PyAmpliColError):
    """A required optional or native dependency is unavailable."""


__all__ = [
    "ArtifactError",
    "CompatibilityError",
    "ConfigurationError",
    "DependencyError",
    "EvaluationError",
    "GenerationError",
    "ModelError",
    "PyAmpliColError",
]
