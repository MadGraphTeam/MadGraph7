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


class NativePlatformError(DependencyError):
    """The running platform cannot load one of pyAmpliCol's native libraries.

    Raised before the library is loaded when its ELF version requirements need
    a newer glibc release than the running one (the loop engine's release
    natives can need ``GLIBC_ABI_GNU2_TLS``: glibc 2.43 or newer, or a glibc
    2.42 build with the stable-branch update that added it), or a glibc on a
    platform without one, and in place of the dynamic loader's
    "version `GLIBC_...' not found" error (for instance on a glibc 2.42 build
    without that update). It is not an ImportError: no fallback can run the
    library on this platform.

    Attributes: ``library`` (path, if known), ``required`` (the oldest glibc
    release that can provide the version, ``"2.42"``; None if unknown),
    ``found`` (the running glibc, None if the C library is not glibc) and
    ``version_name`` (``"GLIBC_ABI_GNU2_TLS"``). ``found`` at or above
    ``required`` means that this glibc build does not define an ABI marker
    its release can define (a glibc 2.42 without that update).
    """

    def __init__(
        self,
        message: str,
        *,
        library: str | None = None,
        required: str | None = None,
        found: str | None = None,
        version_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.library = library
        self.required = required
        self.found = found
        self.version_name = version_name


__all__ = [
    "ArtifactError",
    "CompatibilityError",
    "ConfigurationError",
    "DependencyError",
    "EvaluationError",
    "GenerationError",
    "ModelError",
    "NativePlatformError",
    "PyAmpliColError",
]
