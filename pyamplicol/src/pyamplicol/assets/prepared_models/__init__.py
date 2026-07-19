# SPDX-License-Identifier: 0BSD
"""Discovery and validation for wheel-owned prepared model bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyamplicol.models.prepared import PreparedModelBundle

BUILTIN_SM_JIT_O3 = "built-in-sm-jit-o3"
_METADATA_NAME = f"{BUILTIN_SM_JIT_O3}.metadata.json"
_BUNDLE_NAME = f"{BUILTIN_SM_JIT_O3}.pyamplicol-model"
_KNOWN_MODELS = (BUILTIN_SM_JIT_O3,)
_METADATA_KEYS = frozenset(
    {
        "backend",
        "bundle",
        "bundle_sha256",
        "bundle_size",
        "build_contract",
        "dependencies",
        "eager_kernel_abi",
        "id",
        "jit_optimization_level",
        "kernel_count",
        "model",
        "prepared_model_bundle_schema",
        "producer",
        "schema_version",
        "target",
    }
)


class PackagedPreparedModelError(RuntimeError):
    """Raised when an installed prepared-model resource is absent or stale."""


def available_prepared_models() -> tuple[str, ...]:
    """Return stable identifiers for prepared models shipped in this wheel."""

    return _KNOWN_MODELS


@contextmanager
def packaged_prepared_model_path(identifier: str) -> Iterator[Path]:
    """Materialize and validate one packaged bundle for path-based consumers."""

    if identifier not in _KNOWN_MODELS:
        choices = ", ".join(_KNOWN_MODELS)
        raise PackagedPreparedModelError(
            f"unknown packaged prepared model {identifier!r}; available: {choices}"
        )
    root = resources.files(__package__)
    metadata_resource = root.joinpath(_METADATA_NAME)
    bundle_resource = root.joinpath(_BUNDLE_NAME)
    try:
        metadata = _metadata(json.loads(metadata_resource.read_text(encoding="utf-8")))
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise PackagedPreparedModelError(
            f"cannot read packaged prepared-model metadata: {error}"
        ) from error
    with resources.as_file(bundle_resource) as path:
        bundle_path = Path(path)
        if not bundle_path.is_file() or bundle_path.is_symlink():
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle is not a regular file"
            )
        data = bundle_path.read_bytes()
        if metadata.get("bundle_size") != len(data):
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle size does not match metadata"
            )
        if metadata.get("bundle_sha256") != hashlib.sha256(data).hexdigest():
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle SHA-256 does not match metadata"
            )
        _validate_bundle(bundle_path, metadata)
        yield bundle_path


@contextmanager
def open_packaged_prepared_model(
    identifier: str = BUILTIN_SM_JIT_O3,
) -> Iterator[PreparedModelBundle]:
    """Open a validated bundle while keeping zip-installed resources alive."""

    from pyamplicol.models.prepared import load_prepared_model_bundle

    with packaged_prepared_model_path(identifier) as path:
        yield load_prepared_model_bundle(path)


def _validate_bundle(path: Path, metadata: Mapping[str, object]) -> None:
    from pyamplicol._internal.versions import (
        COMPILED_MODEL_SCHEMA_VERSION,
        SYMBOLICA_SERIALIZATION_ABI,
        SYMJIT_APPLICATION_ABI,
        package_version,
    )
    from pyamplicol.models.builtin.adapters import source_digest
    from pyamplicol.models.loading import MODEL_COMPILER_VERSION, compiler_fingerprint
    from pyamplicol.models.prepared import (
        EAGER_KERNEL_ABI,
        PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
        load_prepared_model_bundle,
    )

    try:
        bundle = load_prepared_model_bundle(path)
    except (OSError, TypeError, ValueError) as error:
        raise PackagedPreparedModelError(
            f"packaged prepared-model bundle is invalid: {error}"
        ) from error
    pack = bundle.kernel_pack
    producer = _mapping(metadata.get("producer"), "metadata.producer")
    dependencies = _mapping(metadata.get("dependencies"), "metadata.dependencies")
    compiled_producer = _mapping(
        bundle.compiled_model.get("producer"), "compiled_model.producer"
    )
    compiled_source = _mapping(
        bundle.compiled_model.get("source"), "compiled_model.source"
    )
    fingerprint = compiler_fingerprint()
    expected = {
        "package_version": package_version(),
        "compiled_model_schema": COMPILED_MODEL_SCHEMA_VERSION,
        "model_compiler_version": MODEL_COMPILER_VERSION,
        "model_compiler_sha256": fingerprint["model_compiler_sha256"],
        "model_source_digest": _compiled_source_digest(source_digest()),
    }
    for key, value in expected.items():
        if producer.get(key) != value:
            raise PackagedPreparedModelError(
                f"packaged prepared-model producer {key} is stale: "
                f"expected {value!r}, got {producer.get(key)!r}"
            )
    if compiled_producer.get("pyamplicol") != expected["package_version"]:
        raise PackagedPreparedModelError(
            "prepared compiled-model package version is stale"
        )
    if (
        compiled_producer.get("compiled_model_schema_version")
        != expected["compiled_model_schema"]
    ):
        raise PackagedPreparedModelError("prepared compiled-model schema is stale")
    if (
        compiled_producer.get("model_compiler_version")
        != expected["model_compiler_version"]
    ):
        raise PackagedPreparedModelError("prepared model compiler version is stale")
    if (
        compiled_producer.get("model_compiler_sha256")
        != expected["model_compiler_sha256"]
    ):
        raise PackagedPreparedModelError("prepared model compiler digest is stale")
    if compiled_source.get("digest") != expected["model_source_digest"]:
        raise PackagedPreparedModelError("prepared built-in model source is stale")
    if pack.producer.get("version") != expected["package_version"]:
        raise PackagedPreparedModelError("prepared kernel-pack version is stale")
    dependency_expected = {
        "symbolica_version": fingerprint["symbolica"],
        "ufo_model_loader_version": fingerprint["ufo_model_loader"],
        "symbolica_serialization_abi": SYMBOLICA_SERIALIZATION_ABI,
        "symjit_application_abi": SYMJIT_APPLICATION_ABI,
    }
    for key, value in dependency_expected.items():
        if dependencies.get(key) != value:
            raise PackagedPreparedModelError(
                f"packaged prepared-model dependency {key} is stale"
            )
    pack_dependencies = {
        "symbolica_version": dependencies["symbolica_version"],
        "symbolica_serialization": dependencies["symbolica_serialization_abi"],
        "symjit_application": dependencies["symjit_application_abi"],
    }
    for key, value in pack_dependencies.items():
        if pack.dependency_abis.get(key) != value:
            raise PackagedPreparedModelError(
                f"prepared kernel-pack dependency {key} is stale"
            )
    if metadata.get("eager_kernel_abi") != EAGER_KERNEL_ABI:
        raise PackagedPreparedModelError("packaged eager kernel ABI is stale")
    if (
        metadata.get("prepared_model_bundle_schema")
        != PREPARED_MODEL_BUNDLE_SCHEMA_VERSION
    ):
        raise PackagedPreparedModelError("packaged model bundle schema is stale")
    if metadata.get("backend") != "jit" or bundle.backend != "jit":
        raise PackagedPreparedModelError("packaged built-in model is not JIT-backed")
    if metadata.get("jit_optimization_level") != 3:
        raise PackagedPreparedModelError("packaged built-in model is not JIT O3")
    if pack.optimization_settings.get("jit_optimization_level") != 3:
        raise PackagedPreparedModelError("prepared kernel pack is not JIT O3")
    if metadata.get("kernel_count") != len(pack.kernels) or not pack.kernels:
        raise PackagedPreparedModelError(
            "packaged prepared-model kernel count is invalid"
        )
    target = _plain_json(pack.target)
    if target != metadata.get("target") or target != {
        "portable": True,
        "word_bits": 64,
        "endianness": "little",
        "target_triple": "portable-symjit-mir",
        "cpu_features": [],
    }:
        raise PackagedPreparedModelError(
            "packaged prepared model is not portable 64-bit little-endian MIR"
        )


def _metadata(value: object) -> Mapping[str, object]:
    metadata = _mapping(value, "prepared-model metadata")
    missing = _METADATA_KEYS - metadata.keys()
    unknown = metadata.keys() - _METADATA_KEYS
    if missing or unknown:
        raise PackagedPreparedModelError(
            "packaged prepared-model metadata fields are invalid"
        )
    if metadata.get("schema_version") != 1:
        raise PackagedPreparedModelError(
            "unsupported packaged prepared-model metadata schema"
        )
    if metadata.get("id") != BUILTIN_SM_JIT_O3:
        raise PackagedPreparedModelError("packaged prepared-model identity is invalid")
    if metadata.get("model") != "built-in-sm" or metadata.get("bundle") != _BUNDLE_NAME:
        raise PackagedPreparedModelError("packaged prepared-model resource is invalid")
    return metadata


def _compiled_source_digest(implementation_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"built-in-sm\0")
    digest.update(implementation_digest.encode("ascii"))
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PackagedPreparedModelError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    return value


__all__ = [
    "BUILTIN_SM_JIT_O3",
    "PackagedPreparedModelError",
    "available_prepared_models",
    "open_packaged_prepared_model",
    "packaged_prepared_model_path",
]
