# SPDX-License-Identifier: 0BSD
"""Independent public, serialization, and evaluator runtime contracts."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

PYTHON_API_VERSION = 1
TOML_SCHEMA_VERSION = 1
COMPILED_MODEL_SCHEMA_VERSION = 6
PROCESS_ARTIFACT_SCHEMA_VERSION = 3
RUNTIME_PHYSICS_SCHEMA_VERSION = 1
C_ABI_VERSION = 1

# Symbolica evaluator states are not compatible merely because the Python
# distribution version matches. This identifier follows the pinned candidate
# serialization contract recorded in dependencies/release-lock.toml.
SYMBOLICA_SERIALIZATION_ABI = "candidate-e4167e7-bincode2"

# This digest is derived only from the SymJIT Application storage contract,
# candidate revision, and ordered patch hashes recorded in release-lock.toml.
# It deliberately contains no checkout or build-machine path.
_SYMJIT_APPLICATION_ABI_INPUT = (
    "symjit-application-abi-v1",
    "storage-version=3",
    "revision=7fb09d1cb2a943c25a6fd71a208af44fcc6d813d",
    "patch=805635ca033cb3852a8862f64d7a52ce1860c42173cc59c202b7a8f7bdc2e504",
    "patch=87adc77dc734dbe96ad68e1efcff6829df6d315f51a7f6916f43d24048771123",
    "patch=1b1b84b7b1e8b66b57e628c06a7d5f2e47cf69597a04b793231eb3d79260dff1",
    "patch=c2e0cf9247930f006fbc7280b3ed60319ebc388b4f4137268e8eb3f9b08bd102",
)
_SYMJIT_APPLICATION_ABI_DIGEST = hashlib.sha256(
    ("\n".join(_SYMJIT_APPLICATION_ABI_INPUT) + "\n").encode("ascii")
).hexdigest()
SYMJIT_APPLICATION_ABI = f"symjit-app-v3-sha256:{_SYMJIT_APPLICATION_ABI_DIGEST}"

SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY = (
    "symbolica.legacy-jit-container.complex-f64.v1"
)
SYMBOLICA_CPP_RUNTIME_CAPABILITY = "symbolica.compiled-cpp.complex-f64.v1"
SYMBOLICA_ASM_RUNTIME_CAPABILITY = "symbolica.compiled-asm.complex-f64.v1"
EVALUATOR_RUNTIME_CAPABILITIES = frozenset(
    {
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }
)

_PACKAGE_BUILD_INFO_PATH = Path(__file__).resolve().parents[1] / "_build_info.json"
_SOURCE_BUILD_INFO_PATH = (
    Path(__file__).resolve().parents[3]
    / ".artifacts"
    / "source-runtime"
    / "_build_info.json"
)


def package_version(default: str = "0.1.0") -> str:
    """Return the wheel/source-runtime version without importing heavy modules."""

    for path in (_PACKAGE_BUILD_INFO_PATH, _SOURCE_BUILD_INFO_PATH):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload["version"]
            if isinstance(value, str) and value:
                return value
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            pass
    try:
        return metadata.version("pyamplicol")
    except metadata.PackageNotFoundError:
        return default


__all__ = [
    "COMPILED_MODEL_SCHEMA_VERSION",
    "C_ABI_VERSION",
    "EVALUATOR_RUNTIME_CAPABILITIES",
    "PROCESS_ARTIFACT_SCHEMA_VERSION",
    "PYTHON_API_VERSION",
    "RUNTIME_PHYSICS_SCHEMA_VERSION",
    "SYMBOLICA_ASM_RUNTIME_CAPABILITY",
    "SYMBOLICA_CPP_RUNTIME_CAPABILITY",
    "SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY",
    "SYMBOLICA_SERIALIZATION_ABI",
    "SYMJIT_APPLICATION_ABI",
    "SYMJIT_F64_RUNTIME_CAPABILITY",
    "TOML_SCHEMA_VERSION",
    "package_version",
]
