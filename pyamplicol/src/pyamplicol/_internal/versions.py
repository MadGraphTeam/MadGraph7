# SPDX-License-Identifier: 0BSD
"""Independent public, serialization, and evaluator runtime contracts."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

PYTHON_API_VERSION = 1
TOML_SCHEMA_VERSION = 1
COMPILED_MODEL_SCHEMA_VERSION = 9
PROCESS_ARTIFACT_SCHEMA_VERSION = 3
RUNTIME_PHYSICS_SCHEMA_VERSION = 1
C_ABI_VERSION = 1

# These are project-owned wire-format identifiers. Exact contributor source
# revisions and patch hashes live only in dependencies/contributor-lock.toml.
SYMBOLICA_SERIALIZATION_ABI = "symbolica-bincode2-v1"
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"

SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY = (
    "symbolica.legacy-jit-container.complex-f64.v1"
)
SYMBOLICA_CPP_RUNTIME_CAPABILITY = "symbolica.compiled-cpp.complex-f64.v1"
SYMBOLICA_ASM_RUNTIME_CAPABILITY = "symbolica.compiled-asm.complex-f64.v1"
EAGER_DAG_F64_RUNTIME_CAPABILITY = "rusticol.eager-dag.complex-f64.v1"
EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY = "rusticol.eager-dag.lc-topology-replay.v1"
COMPILED_RUNTIME_SELECTORS_CAPABILITY = "rusticol.compiled.runtime-selectors.v1"
COMPILED_HELICITY_DUAL_LANE_CAPABILITY = "rusticol.compiled.helicity-dual-lane.v1"
COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY = (
    "rusticol.compiled.helicity-selector-union.v1"
)
COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY = (
    "rusticol.compiled.helicity-primary-recurrence.v1"
)
COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY = "rusticol.compiled.color-topology-lanes.v1"
EVALUATOR_RUNTIME_CAPABILITIES = frozenset(
    {
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        EAGER_DAG_F64_RUNTIME_CAPABILITY,
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
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
    "COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY",
    "COMPILED_HELICITY_DUAL_LANE_CAPABILITY",
    "COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY",
    "COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY",
    "COMPILED_MODEL_SCHEMA_VERSION",
    "COMPILED_RUNTIME_SELECTORS_CAPABILITY",
    "C_ABI_VERSION",
    "EAGER_DAG_F64_RUNTIME_CAPABILITY",
    "EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY",
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
