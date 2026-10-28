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

    On Linux a native library needs ELF symbol versions from glibc
    (``GLIBC_2.38``, or an ABI marker such as ``GLIBC_ABI_GNU2_TLS``). Raised
    before ``pyamplicol._rusticol`` is loaded when the running glibc does not
    define one of them, or when the process does not run on glibc at all; and
    in place of the dynamic loader's "version `GLIBC_...' not found" error when
    it loads ``pyamplicol._rusticol`` (a library it depends on included) or
    Symbolica's extension at ``import pyamplicol``; other native code reports
    the loader's error unchanged. The dynamic loader would refuse the library
    for the same reason. It is not an ImportError, so code that treats a
    missing optional module as absent does not hide it.

    Attributes: ``library`` (path, if known), ``missing`` (the version names
    the running glibc does not define, numbered ones first), ``required`` (the
    newest glibc release among the missing numbered versions, ``"2.38"``; None
    when only ABI markers are missing, which no release number guarantees) and
    ``found`` (the running glibc release, None if the C library is not glibc).
    """

    def __init__(
        self,
        message: str,
        *,
        library: str | None = None,
        required: str | None = None,
        found: str | None = None,
        missing: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.library = library
        self.required = required
        self.found = found
        self.missing = missing


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
