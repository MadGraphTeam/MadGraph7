# SPDX-License-Identifier: 0BSD
"""Independent public, serialization, and evaluator runtime contracts."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

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
EAGER_RUNTIME_LAYOUT_F64_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
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
        EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }
)

_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_BUILD_INFO_PATH = _SOURCE_PACKAGE_ROOT / "_build_info.json"
_SOURCE_RUNTIME_ROOT = _SOURCE_ROOT / ".artifacts" / "source-runtime"
_SOURCE_BUILD_INFO_PATH = _SOURCE_RUNTIME_ROOT / "_build_info.json"
_SOURCE_RUNTIME_STAGING_PATH = _SOURCE_RUNTIME_ROOT / ".staging"

_NATIVE_BUILD_INPUT_FILES = (
    Path("Cargo.lock"),
    Path("Cargo.toml"),
    Path("pyproject.toml"),
    Path("rust-toolchain.toml"),
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/contributor-lock.toml"),
    Path("dependencies/install-state.json"),
    Path("dependencies/release-lock.toml"),
)
_NATIVE_BUILD_INPUT_TREES = (Path("build_backend"), Path("rust"))
_NATIVE_BUILD_INPUT_SUFFIXES = {
    ".f90",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
}
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")


def _native_build_inputs_digest(root: Path) -> str:
    """Hash the small set of checkout inputs that determines the native build."""

    paths = [root / relative for relative in _NATIVE_BUILD_INPUT_FILES]
    for relative in _NATIVE_BUILD_INPUT_TREES:
        tree = root / relative
        if not tree.is_dir():
            continue
        paths.extend(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and not {"__pycache__", "target"}.intersection(path.relative_to(tree).parts)
            and path.suffix in _NATIVE_BUILD_INPUT_SUFFIXES
        )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _native_extensions(package_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in package_root.glob("_rusticol.*")
                if path.is_file() and path.name.endswith(_NATIVE_EXTENSION_SUFFIXES)
            ),
            key=lambda path: path.name,
        )
    )


def _read_build_info(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"{description} is unreadable; rerun `just dev-install`"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is invalid; rerun `just dev-install`")
    return payload


def _is_source_checkout(package_root: Path, source_root: Path) -> bool:
    return (
        package_root == source_root / "src" / "pyamplicol"
        and (source_root / "pyproject.toml").is_file()
    )


def _verify_source_runtime(
    payload: dict[str, Any],
    *,
    package_root: Path | None = None,
    source_root: Path | None = None,
) -> None:
    contract = payload.get("source_runtime")
    if not isinstance(contract, dict):
        raise RuntimeError(
            "source runtime provenance is missing; rerun `just dev-install`"
        )
    extension_name = contract.get("extension_name")
    extension_sha256 = contract.get("extension_sha256")
    native_digest = contract.get("native_build_inputs_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (extension_name, extension_sha256, native_digest)
    ):
        raise RuntimeError(
            "source runtime provenance is incomplete; rerun `just dev-install`"
        )
    package_root = package_root or _SOURCE_PACKAGE_ROOT
    extensions = _native_extensions(package_root)
    if len(extensions) != 1 or extensions[0].name != extension_name:
        raise RuntimeError(
            "source runtime extension inventory is ambiguous or stale; "
            "rerun `just dev-install`"
        )
    if hashlib.sha256(extensions[0].read_bytes()).hexdigest() != extension_sha256:
        raise RuntimeError(
            "source runtime extension is stale or was replaced; "
            "rerun `just dev-install`"
        )
    source_root = source_root or _SOURCE_ROOT
    if _native_build_inputs_digest(source_root) != native_digest:
        raise RuntimeError(
            "native build inputs changed after the source runtime was staged; "
            "rerun `just dev-install`"
        )


def _verify_candidate_install(payload: dict[str, Any]) -> None:
    raw_root = payload.get("source_checkout")
    native_digest = payload.get("native_build_inputs_sha256")
    if (
        not isinstance(raw_root, str)
        or not raw_root
        or not isinstance(native_digest, str)
        or len(native_digest) != 64
    ):
        raise RuntimeError(
            "candidate wheel provenance is incomplete; rerun `just dev-install`"
        )
    source_root = Path(raw_root)
    if not source_root.is_absolute() or not (source_root / "pyproject.toml").is_file():
        raise RuntimeError(
            "candidate wheel source checkout is unavailable; rerun `just dev-install`"
        )
    if _native_build_inputs_digest(source_root) != native_digest:
        raise RuntimeError(
            "installed candidate wheel is stale for this checkout; "
            "rerun `just dev-install`"
        )


def _active_build_info() -> dict[str, Any] | None:
    if (
        _is_source_checkout(_SOURCE_PACKAGE_ROOT, _SOURCE_ROOT)
        and _SOURCE_BUILD_INFO_PATH.exists()
    ):
        return _read_build_info(
            _SOURCE_BUILD_INFO_PATH,
            "source runtime provenance",
        )
    if _PACKAGE_BUILD_INFO_PATH.exists():
        return _read_build_info(_PACKAGE_BUILD_INFO_PATH, "wheel build provenance")
    return None


def verify_native_module(module: Any, *, expected_version: str | None = None) -> None:
    """Reject a stale native extension in contributor builds.

    Published wheels retain the normal package-manager import path; the extra
    build-ID check is limited to non-publishable candidate wheels and staged
    source runtimes.
    """

    # Unit tests and external adapters may provide a duck-typed stand-in. Real
    # extension modules always expose an on-disk module path.
    if not getattr(module, "__file__", None):
        return
    operation = getattr(module, "package_version", None)
    if not callable(operation):
        raise RuntimeError(
            "native runtime has no package-version contract; reinstall pyAmpliCol"
        )
    observed = operation()
    expected = expected_version or package_version()
    if not isinstance(observed, str) or observed.replace("-dev.", ".dev") != expected:
        raise RuntimeError(
            "native runtime version does not match the Python package "
            f"({observed!r} != {expected!r}); rerun `just dev-install`"
        )
    build_info = _active_build_info()
    if build_info is None or build_info.get("publishable") is not False:
        return
    native_digest = build_info.get("native_build_inputs_sha256")
    native_operation = getattr(module, "native_build_inputs_sha256", None)
    if not isinstance(native_digest, str) or not callable(native_operation):
        raise RuntimeError(
            "native runtime build-ID contract is incomplete; rerun `just dev-install`"
        )
    if native_operation() != native_digest:
        raise RuntimeError(
            "native runtime was built from different source inputs; "
            "rerun `just dev-install`"
        )


def package_version(default: str = "0.1.0") -> str:
    """Return the wheel/source-runtime version without importing heavy modules."""

    if _is_source_checkout(_SOURCE_PACKAGE_ROOT, _SOURCE_ROOT):
        source_runtime_present = (
            _SOURCE_BUILD_INFO_PATH.exists()
            or _SOURCE_RUNTIME_STAGING_PATH.exists()
            or bool(_native_extensions(_SOURCE_PACKAGE_ROOT))
        )
        if source_runtime_present:
            if _SOURCE_RUNTIME_STAGING_PATH.exists():
                raise RuntimeError(
                    "source runtime staging is incomplete; rerun `just dev-install`"
                )
            payload = _read_build_info(
                _SOURCE_BUILD_INFO_PATH,
                "source runtime provenance",
            )
            _verify_source_runtime(payload)
            value = payload.get("version")
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    "source runtime version provenance is invalid; "
                    "rerun `just dev-install`"
                )
            return value

    if _PACKAGE_BUILD_INFO_PATH.exists():
        payload = _read_build_info(
            _PACKAGE_BUILD_INFO_PATH,
            "wheel build provenance",
        )
        value = payload.get("version")
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                "wheel build version provenance is invalid; reinstall pyAmpliCol"
            )
        if payload.get("publishable") is False:
            _verify_candidate_install(payload)
        return value
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
    "EAGER_RUNTIME_LAYOUT_F64_CAPABILITY",
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
    "verify_native_module",
]
