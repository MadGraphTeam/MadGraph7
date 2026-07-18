# SPDX-License-Identifier: 0BSD
"""Deterministic prepared-model bundle container contract.

This module owns the portable container format only. Kernel compilation and
backend-specific manifest interpretation belong to the model compiler and
runtime respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

PREPARED_MODEL_BUNDLE_KIND = "pyamplicol-prepared-model"
PREPARED_MODEL_BUNDLE_SCHEMA_VERSION = 1
PREPARED_MODEL_BUNDLE_SUFFIX = ".pyamplicol-model"
EAGER_KERNEL_ABI = "pyamplicol-eager-kernel-v1"
EAGER_KERNEL_ABI_VERSION = 1

PREPARED_MODEL_MANIFEST_PATH = "manifest.json"
PREPARED_MODEL_COMPILED_MODEL_PATH = "model/model.pyAmplicol-model.json"

PreparedBackend: TypeAlias = Literal["jit", "asm", "cpp"]
KernelContractKind: TypeAlias = Literal[
    "vertex",
    "propagator",
    "closure",
    "model-parameter",
]
PayloadSource: TypeAlias = bytes | bytearray | memoryview | Path

_BACKENDS = frozenset(("jit", "asm", "cpp"))
_CONTRACT_KINDS = frozenset(("vertex", "propagator", "closure", "model-parameter"))
_PATH_FIELDS = frozenset(
    (
        "application_path",
        "evaluator_state_path",
        "library_path",
        "payload_path",
        "source_path",
    )
)
_PATH_LIST_FIELDS = frozenset(("payload_paths",))
_SHA256_LENGTH = 64
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = (stat.S_IFREG | 0o644) << 16


class PreparedModelBundleError(ValueError):
    """Raised when a prepared-model bundle violates its container contract."""


def _freeze_json(value: object, context: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreparedModelBundleError(
                f"{context} must not contain NaN or infinity"
            )
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PreparedModelBundleError(f"{context} keys must be strings")
            frozen[key] = _freeze_json(item, f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise PreparedModelBundleError(
        f"{context} contains unsupported value {type(value).__name__}"
    )


def _freeze_mapping(value: Mapping[str, object], context: str) -> Mapping[str, object]:
    frozen = _freeze_json(value, context)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by annotation
        raise PreparedModelBundleError(f"{context} must be an object")
    return cast(Mapping[str, object], frozen)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PreparedModelBundleError(f"{context} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PreparedModelBundleError(f"{context} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    context: str,
    expected: frozenset[str],
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise PreparedModelBundleError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise PreparedModelBundleError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreparedModelBundleError(f"{context} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreparedModelBundleError(f"{context} must be an integer >= 0")
    return value


def _positive_integer(value: object, context: str) -> int:
    result = _nonnegative_integer(value, context)
    if result == 0:
        raise PreparedModelBundleError(f"{context} must be an integer >= 1")
    return result


def _normalized_member_path(value: object, context: str) -> str:
    path = _nonempty_string(value, context)
    if "\\" in path or "\x00" in path:
        raise PreparedModelBundleError(
            f"{context} must be a normalized relative POSIX path"
        )
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or path.endswith("/")
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PreparedModelBundleError(
            f"{context} must be a normalized relative POSIX path"
        )
    return path


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    try:
        serialized = json.dumps(
            _thaw_json(payload),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PreparedModelBundleError(
            f"payload is not canonical JSON: {error}"
        ) from error
    return (serialized + "\n").encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: object, context: str) -> str:
    digest = _nonempty_string(value, context)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PreparedModelBundleError(f"{context} must be a lowercase SHA-256")
    return digest


def _collect_manifest_paths(value: object, *, context: str) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(item: object, item_context: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in _PATH_FIELDS:
                    if child is not None:
                        paths.append(
                            _normalized_member_path(child, f"{item_context}.{key}")
                        )
                elif key in _PATH_LIST_FIELDS:
                    for index, path in enumerate(
                        _sequence(child, f"{item_context}.{key}")
                    ):
                        paths.append(
                            _normalized_member_path(
                                path,
                                f"{item_context}.{key}[{index}]",
                            )
                        )
                else:
                    visit(child, f"{item_context}.{key}")
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, child in enumerate(item):
                visit(child, f"{item_context}[{index}]")

    visit(value, context)
    return tuple(sorted(set(paths)))


@dataclass(frozen=True, slots=True)
class PreparedKernelRecord:
    """One canonical eager kernel and its exact/f64 evaluator contracts."""

    kernel_id: int
    contract_kind: KernelContractKind
    canonical_signature: str
    input_arity: int
    output_arity: int
    input_layout: tuple[str, ...]
    input_contracts: tuple[Mapping[str, object], ...]
    output_layout: tuple[str, ...]
    exact_expressions: tuple[str, ...]
    exact_evaluator_state_path: str
    f64_evaluator_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_id",
            _nonnegative_integer(self.kernel_id, "kernel.kernel_id"),
        )
        if self.contract_kind not in _CONTRACT_KINDS:
            raise PreparedModelBundleError(
                f"unsupported kernel contract kind {self.contract_kind!r}"
            )
        _nonempty_string(self.canonical_signature, "kernel.canonical_signature")
        input_arity = _nonnegative_integer(self.input_arity, "kernel.input_arity")
        output_arity = _nonnegative_integer(self.output_arity, "kernel.output_arity")
        input_layout = tuple(
            _nonempty_string(item, f"kernel.input_layout[{index}]")
            for index, item in enumerate(self.input_layout)
        )
        output_layout = tuple(
            _nonempty_string(item, f"kernel.output_layout[{index}]")
            for index, item in enumerate(self.output_layout)
        )
        if len(input_layout) != input_arity:
            raise PreparedModelBundleError(
                "kernel input layout length must equal input arity"
            )
        if len(output_layout) != output_arity:
            raise PreparedModelBundleError(
                "kernel output layout length must equal output arity"
            )
        object.__setattr__(self, "input_layout", input_layout)
        object.__setattr__(self, "output_layout", output_layout)
        input_contracts = tuple(
            _freeze_mapping(item, f"kernel.input_contracts[{index}]")
            for index, item in enumerate(self.input_contracts)
        )
        if len(input_contracts) != input_arity:
            raise PreparedModelBundleError(
                "kernel input contract count must equal input arity"
            )
        for index, contract in enumerate(input_contracts):
            _require_exact_keys(
                contract,
                f"kernel.input_contracts[{index}]",
                frozenset(
                    (
                        "role",
                        "component",
                        "symbol",
                        "model_parameter_name",
                        "model_parameter_index",
                    )
                ),
            )
            _nonempty_string(
                contract.get("role"), f"kernel.input_contracts[{index}].role"
            )
            _nonnegative_integer(
                contract.get("component"),
                f"kernel.input_contracts[{index}].component",
            )
            _nonempty_string(
                contract.get("symbol"),
                f"kernel.input_contracts[{index}].symbol",
            )
        object.__setattr__(self, "input_contracts", input_contracts)
        exact_expressions = tuple(
            _nonempty_string(item, f"kernel.exact_expressions[{index}]")
            for index, item in enumerate(self.exact_expressions)
        )
        if len(exact_expressions) != output_arity:
            raise PreparedModelBundleError(
                "kernel exact expression count must equal output arity"
            )
        object.__setattr__(self, "exact_expressions", exact_expressions)
        object.__setattr__(
            self,
            "exact_evaluator_state_path",
            _normalized_member_path(
                self.exact_evaluator_state_path,
                "kernel.exact_evaluator_state_path",
            ),
        )
        if not self.f64_evaluator_manifest:
            raise PreparedModelBundleError(
                "kernel.f64_evaluator_manifest must not be empty"
            )
        object.__setattr__(
            self,
            "f64_evaluator_manifest",
            _freeze_mapping(
                self.f64_evaluator_manifest,
                "kernel.f64_evaluator_manifest",
            ),
        )

    @property
    def referenced_payload_paths(self) -> tuple[str, ...]:
        paths = {
            self.exact_evaluator_state_path,
            *_collect_manifest_paths(
                self.f64_evaluator_manifest,
                context="kernel.f64_evaluator_manifest",
            ),
        }
        return tuple(sorted(paths))

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "contract_kind": self.contract_kind,
            "canonical_signature": self.canonical_signature,
            "input_arity": self.input_arity,
            "output_arity": self.output_arity,
            "input_layout": list(self.input_layout),
            "input_contracts": [
                _thaw_json(contract) for contract in self.input_contracts
            ],
            "output_layout": list(self.output_layout),
            "exact_expressions": list(self.exact_expressions),
            "exact_evaluator_state_path": self.exact_evaluator_state_path,
            "f64_evaluator_manifest": _thaw_json(self.f64_evaluator_manifest),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedKernelRecord:
        expected = frozenset(
            (
                "kernel_id",
                "contract_kind",
                "canonical_signature",
                "input_arity",
                "output_arity",
                "input_layout",
                "input_contracts",
                "output_layout",
                "exact_expressions",
                "exact_evaluator_state_path",
                "f64_evaluator_manifest",
            )
        )
        _require_exact_keys(value, "kernel", expected)
        contract_kind = _nonempty_string(
            value.get("contract_kind"),
            "kernel.contract_kind",
        )
        return cls(
            kernel_id=_nonnegative_integer(value.get("kernel_id"), "kernel.kernel_id"),
            contract_kind=cast(KernelContractKind, contract_kind),
            canonical_signature=_nonempty_string(
                value.get("canonical_signature"),
                "kernel.canonical_signature",
            ),
            input_arity=_nonnegative_integer(
                value.get("input_arity"), "kernel.input_arity"
            ),
            output_arity=_nonnegative_integer(
                value.get("output_arity"), "kernel.output_arity"
            ),
            input_layout=tuple(
                _nonempty_string(item, f"kernel.input_layout[{index}]")
                for index, item in enumerate(
                    _sequence(value.get("input_layout"), "kernel.input_layout")
                )
            ),
            input_contracts=tuple(
                _mapping(item, f"kernel.input_contracts[{index}]")
                for index, item in enumerate(
                    _sequence(
                        value.get("input_contracts"), "kernel.input_contracts"
                    )
                )
            ),
            output_layout=tuple(
                _nonempty_string(item, f"kernel.output_layout[{index}]")
                for index, item in enumerate(
                    _sequence(value.get("output_layout"), "kernel.output_layout")
                )
            ),
            exact_expressions=tuple(
                _nonempty_string(item, f"kernel.exact_expressions[{index}]")
                for index, item in enumerate(
                    _sequence(
                        value.get("exact_expressions"),
                        "kernel.exact_expressions",
                    )
                )
            ),
            exact_evaluator_state_path=_normalized_member_path(
                value.get("exact_evaluator_state_path"),
                "kernel.exact_evaluator_state_path",
            ),
            f64_evaluator_manifest=_mapping(
                value.get("f64_evaluator_manifest"),
                "kernel.f64_evaluator_manifest",
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedKernelPack:
    """The single prepared backend pack embedded in a model bundle."""

    backend: PreparedBackend
    optimization_settings: Mapping[str, object]
    producer: Mapping[str, object]
    dependency_abis: Mapping[str, object]
    provenance: Mapping[str, object]
    target: Mapping[str, object]
    resolver_manifest: Mapping[str, object]
    kernels: tuple[PreparedKernelRecord, ...]

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise PreparedModelBundleError(
                f"unsupported prepared backend {self.backend!r}"
            )
        for name in (
            "optimization_settings",
            "producer",
            "dependency_abis",
            "provenance",
            "target",
            "resolver_manifest",
        ):
            value = cast(Mapping[str, object], getattr(self, name))
            if not value:
                raise PreparedModelBundleError(f"kernel_pack.{name} must not be empty")
            object.__setattr__(
                self,
                name,
                _freeze_mapping(value, f"kernel_pack.{name}"),
            )
        target = self.target
        target_fields = frozenset(
            ("portable", "word_bits", "endianness", "target_triple", "cpu_features")
        )
        _require_exact_keys(target, "kernel_pack.target", target_fields)
        if not isinstance(target.get("portable"), bool):
            raise PreparedModelBundleError(
                "kernel_pack.target.portable must be a boolean"
            )
        _positive_integer(target.get("word_bits"), "kernel_pack.target.word_bits")
        if target.get("endianness") not in {"little", "big"}:
            raise PreparedModelBundleError(
                "kernel_pack.target.endianness must be 'little' or 'big'"
            )
        _nonempty_string(
            target.get("target_triple"),
            "kernel_pack.target.target_triple",
        )
        cpu_features = tuple(
            _nonempty_string(item, f"kernel_pack.target.cpu_features[{index}]")
            for index, item in enumerate(
                _sequence(
                    target.get("cpu_features"),
                    "kernel_pack.target.cpu_features",
                )
            )
        )
        if cpu_features != tuple(sorted(set(cpu_features))):
            raise PreparedModelBundleError(
                "kernel_pack.target.cpu_features must be sorted and unique"
            )
        kernels = tuple(sorted(self.kernels, key=lambda kernel: kernel.kernel_id))
        if not kernels:
            raise PreparedModelBundleError("kernel_pack.kernels must not be empty")
        kernel_ids = tuple(kernel.kernel_id for kernel in kernels)
        signatures = tuple(kernel.canonical_signature for kernel in kernels)
        if len(set(kernel_ids)) != len(kernel_ids):
            raise PreparedModelBundleError("kernel IDs must be unique")
        if len(set(signatures)) != len(signatures):
            raise PreparedModelBundleError("kernel canonical signatures must be unique")
        object.__setattr__(self, "kernels", kernels)

    @property
    def referenced_payload_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    path
                    for kernel in self.kernels
                    for path in kernel.referenced_payload_paths
                }
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "optimization_settings": _thaw_json(self.optimization_settings),
            "producer": _thaw_json(self.producer),
            "dependency_abis": _thaw_json(self.dependency_abis),
            "provenance": _thaw_json(self.provenance),
            "target": _thaw_json(self.target),
            "resolver_manifest": _thaw_json(self.resolver_manifest),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedKernelPack:
        expected = frozenset(
            (
                "backend",
                "optimization_settings",
                "producer",
                "dependency_abis",
                "provenance",
                "target",
                "resolver_manifest",
                "kernels",
            )
        )
        _require_exact_keys(value, "kernel_pack", expected)
        backend = _nonempty_string(value.get("backend"), "kernel_pack.backend")
        kernels = tuple(
            PreparedKernelRecord.from_dict(
                _mapping(item, f"kernel_pack.kernels[{index}]")
            )
            for index, item in enumerate(
                _sequence(value.get("kernels"), "kernel_pack.kernels")
            )
        )
        return cls(
            backend=cast(PreparedBackend, backend),
            optimization_settings=_mapping(
                value.get("optimization_settings"),
                "kernel_pack.optimization_settings",
            ),
            producer=_mapping(value.get("producer"), "kernel_pack.producer"),
            dependency_abis=_mapping(
                value.get("dependency_abis"),
                "kernel_pack.dependency_abis",
            ),
            provenance=_mapping(value.get("provenance"), "kernel_pack.provenance"),
            target=_mapping(value.get("target"), "kernel_pack.target"),
            resolver_manifest=_mapping(
                value.get("resolver_manifest"),
                "kernel_pack.resolver_manifest",
            ),
            kernels=kernels,
        )


@dataclass(frozen=True, slots=True)
class PreparedModelBundle:
    """A validated prepared-model archive and its immutable metadata."""

    path: Path
    compiled_model: Mapping[str, object]
    kernel_pack: PreparedKernelPack
    manifest: Mapping[str, object]
    _member_digests: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        object.__setattr__(
            self,
            "compiled_model",
            _freeze_mapping(self.compiled_model, "compiled_model"),
        )
        object.__setattr__(
            self,
            "manifest",
            _freeze_mapping(self.manifest, "manifest"),
        )
        object.__setattr__(
            self,
            "_member_digests",
            MappingProxyType(dict(self._member_digests)),
        )

    @property
    def backend(self) -> PreparedBackend:
        return self.kernel_pack.backend

    def compiled_model_payload(self) -> dict[str, object]:
        """Return a detached plain-JSON payload for the model deserializer."""

        payload = _thaw_json(self.compiled_model)
        if not isinstance(payload, dict):  # pragma: no cover - constructor invariant
            raise PreparedModelBundleError("compiled model root must be an object")
        return payload

    def read_payload(self, member_path: str) -> bytes:
        """Read and revalidate one kernel payload from the archive."""
        normalized = _normalized_member_path(member_path, "member_path")
        if normalized not in self.kernel_pack.referenced_payload_paths:
            raise PreparedModelBundleError(
                f"{normalized!r} is not referenced by the prepared kernel pack"
            )
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                data = archive.read(normalized)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise PreparedModelBundleError(
                f"could not read prepared payload {normalized!r}: {error}"
            ) from error
        expected = self._member_digests[normalized]
        if _sha256(data) != expected:
            raise PreparedModelBundleError(
                f"prepared payload {normalized!r} has a SHA-256 mismatch"
            )
        return data

    def copy_referenced_payloads(self, destination: Path) -> tuple[Path, ...]:
        """Atomically copy all referenced backend payloads below *destination*."""
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise PreparedModelBundleError("payload destination must not be a symlink")
        root = destination.resolve()
        outputs: list[Path] = []
        for member_path in self.kernel_pack.referenced_payload_paths:
            output = root.joinpath(*PurePosixPath(member_path).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                output.parent.resolve().relative_to(root)
            except ValueError as error:
                raise PreparedModelBundleError(
                    f"payload destination for {member_path!r} escapes its root"
                ) from error
            if output.is_symlink():
                raise PreparedModelBundleError(
                    f"payload destination {output} must not be a symlink"
                )
            data = self.read_payload(member_path)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(data)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(output)
        return tuple(outputs)


def _read_payload_source(source: PayloadSource, context: str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, Path):
        if source.is_symlink():
            raise PreparedModelBundleError(f"{context} must not be a symlink")
        if not source.is_file():
            raise PreparedModelBundleError(f"{context} must be a regular file")
        return source.read_bytes()
    raise PreparedModelBundleError(f"{context} has an unsupported payload source")


def _bundle_output_path(path: Path) -> Path:
    text = str(path)
    if not text.endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        text += PREPARED_MODEL_BUNDLE_SUFFIX
    return Path(text).expanduser().resolve()


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = _ZIP_FILE_MODE
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.comment = b""
            for member_path in (
                PREPARED_MODEL_MANIFEST_PATH,
                *sorted(
                    path for path in members if path != PREPARED_MODEL_MANIFEST_PATH
                ),
            ):
                archive.writestr(_zip_info(member_path), members[member_path])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prepared_model_bundle(
    path: Path,
    *,
    compiled_model: Mapping[str, object],
    kernel_pack: PreparedKernelPack,
    payloads: Mapping[str, PayloadSource],
) -> Path:
    """Write one deterministic, self-contained prepared-model bundle."""
    frozen_model = _freeze_mapping(compiled_model, "compiled_model")
    payload_data: dict[str, bytes] = {}
    for raw_path, source in payloads.items():
        member_path = _normalized_member_path(raw_path, "payload path")
        if member_path in {
            PREPARED_MODEL_MANIFEST_PATH,
            PREPARED_MODEL_COMPILED_MODEL_PATH,
        }:
            raise PreparedModelBundleError(
                f"payload path {member_path!r} is reserved by the bundle"
            )
        if member_path in payload_data:
            raise PreparedModelBundleError(f"duplicate payload path {member_path!r}")
        payload_data[member_path] = _read_payload_source(
            source,
            f"payload {member_path!r}",
        )
    referenced = set(kernel_pack.referenced_payload_paths)
    supplied = set(payload_data)
    if referenced != supplied:
        missing = sorted(referenced - supplied)
        unexpected = sorted(supplied - referenced)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unreferenced: " + ", ".join(unexpected))
        raise PreparedModelBundleError(
            "prepared payload set does not match kernel references ("
            + "; ".join(details)
            + ")"
        )
    hashed_members = {
        PREPARED_MODEL_COMPILED_MODEL_PATH: _canonical_json(frozen_model),
        **payload_data,
    }
    member_records = [
        {
            "path": member_path,
            "size": len(data),
            "sha256": _sha256(data),
        }
        for member_path, data in sorted(hashed_members.items())
    ]
    manifest: dict[str, object] = {
        "kind": PREPARED_MODEL_BUNDLE_KIND,
        "schema_version": PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
        "eager_kernel_abi": EAGER_KERNEL_ABI,
        "kernel_pack": kernel_pack.to_dict(),
        "members": member_records,
    }
    members = {
        PREPARED_MODEL_MANIFEST_PATH: _canonical_json(manifest),
        **hashed_members,
    }
    output = _bundle_output_path(path)
    _write_zip(output, members)
    return output


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for index, info in enumerate(archive.infolist()):
        member_path = _normalized_member_path(info.filename, f"archive member {index}")
        if member_path in members:
            raise PreparedModelBundleError(
                f"prepared-model archive contains duplicate member {member_path!r}"
            )
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_IFMT(mode) == stat.S_IFLNK:
            raise PreparedModelBundleError(
                f"prepared-model member {member_path!r} must be a regular file"
            )
        members[member_path] = info
    return members


def _load_json_member(
    archive: zipfile.ZipFile,
    member_path: str,
    context: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(archive.read(member_path))
    except KeyError as error:
        raise PreparedModelBundleError(
            f"prepared-model archive is missing {member_path!r}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparedModelBundleError(
            f"{context} is not valid JSON: {error}"
        ) from error
    return _mapping(payload, context)


def _load_member_records(value: object) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(_sequence(value, "manifest.members")):
        record = _mapping(item, f"manifest.members[{index}]")
        _require_exact_keys(
            record,
            f"manifest.members[{index}]",
            frozenset(("path", "size", "sha256")),
        )
        path = _normalized_member_path(
            record.get("path"),
            f"manifest.members[{index}].path",
        )
        if path == PREPARED_MODEL_MANIFEST_PATH:
            raise PreparedModelBundleError("manifest.json must not hash itself")
        if path in records:
            raise PreparedModelBundleError(
                f"manifest contains duplicate member record {path!r}"
            )
        size = _nonnegative_integer(
            record.get("size"),
            f"manifest.members[{index}].size",
        )
        digest = _valid_sha256(
            record.get("sha256"),
            f"manifest.members[{index}].sha256",
        )
        records[path] = (size, digest)
    return records


def load_prepared_model_bundle(path: Path) -> PreparedModelBundle:
    """Load and validate a prepared-model bundle without extracting it."""
    requested_path = Path(path).expanduser()
    if requested_path.is_symlink():
        raise PreparedModelBundleError("prepared-model bundle must not be a symlink")
    archive_path = requested_path.resolve()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = _archive_members(archive)
            if PREPARED_MODEL_MANIFEST_PATH not in archive_members:
                raise PreparedModelBundleError(
                    "prepared-model archive is missing 'manifest.json'"
                )
            manifest = _load_json_member(
                archive,
                PREPARED_MODEL_MANIFEST_PATH,
                "manifest",
            )
            _require_exact_keys(
                manifest,
                "manifest",
                frozenset(
                    (
                        "kind",
                        "schema_version",
                        "eager_kernel_abi",
                        "kernel_pack",
                        "members",
                    )
                ),
            )
            if manifest.get("kind") != PREPARED_MODEL_BUNDLE_KIND:
                raise PreparedModelBundleError("invalid prepared-model bundle kind")
            if manifest.get("schema_version") != PREPARED_MODEL_BUNDLE_SCHEMA_VERSION:
                raise PreparedModelBundleError(
                    "unsupported prepared-model bundle schema"
                )
            if manifest.get("eager_kernel_abi") != EAGER_KERNEL_ABI:
                raise PreparedModelBundleError("unsupported eager kernel ABI")
            kernel_pack = PreparedKernelPack.from_dict(
                _mapping(manifest.get("kernel_pack"), "manifest.kernel_pack")
            )
            records = _load_member_records(manifest.get("members"))
            expected_paths = {PREPARED_MODEL_MANIFEST_PATH, *records}
            actual_paths = set(archive_members)
            if expected_paths != actual_paths:
                missing = sorted(expected_paths - actual_paths)
                unexpected = sorted(actual_paths - expected_paths)
                details: list[str] = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unexpected:
                    details.append("unexpected: " + ", ".join(unexpected))
                raise PreparedModelBundleError(
                    "prepared-model archive members do not match manifest ("
                    + "; ".join(details)
                    + ")"
                )
            if PREPARED_MODEL_COMPILED_MODEL_PATH not in records:
                raise PreparedModelBundleError(
                    "prepared-model archive is missing its compiled model"
                )
            for member_path, (expected_size, expected_digest) in records.items():
                data = archive.read(member_path)
                if len(data) != expected_size:
                    raise PreparedModelBundleError(
                        f"prepared-model member {member_path!r} has a size mismatch"
                    )
                if _sha256(data) != expected_digest:
                    raise PreparedModelBundleError(
                        f"prepared-model member {member_path!r} has a SHA-256 mismatch"
                    )
            payload_paths = set(records) - {PREPARED_MODEL_COMPILED_MODEL_PATH}
            referenced_paths = set(kernel_pack.referenced_payload_paths)
            if payload_paths != referenced_paths:
                missing = sorted(referenced_paths - payload_paths)
                unexpected = sorted(payload_paths - referenced_paths)
                details = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unexpected:
                    details.append("unreferenced: " + ", ".join(unexpected))
                raise PreparedModelBundleError(
                    "prepared payload members do not match kernel references ("
                    + "; ".join(details)
                    + ")"
                )
            compiled_model = _load_json_member(
                archive,
                PREPARED_MODEL_COMPILED_MODEL_PATH,
                "compiled_model",
            )
    except PreparedModelBundleError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise PreparedModelBundleError(
            f"could not open prepared-model bundle: {error}"
        ) from error
    return PreparedModelBundle(
        path=archive_path,
        compiled_model=compiled_model,
        kernel_pack=kernel_pack,
        manifest=manifest,
        _member_digests={
            member_path: digest for member_path, (_, digest) in records.items()
        },
    )


def read_prepared_model_bundle(path: Path) -> PreparedModelBundle:
    """Alias for :func:`load_prepared_model_bundle` for reader-style callers."""
    return load_prepared_model_bundle(path)


__all__ = [
    "EAGER_KERNEL_ABI",
    "EAGER_KERNEL_ABI_VERSION",
    "PREPARED_MODEL_BUNDLE_KIND",
    "PREPARED_MODEL_BUNDLE_SCHEMA_VERSION",
    "PREPARED_MODEL_BUNDLE_SUFFIX",
    "PREPARED_MODEL_COMPILED_MODEL_PATH",
    "PREPARED_MODEL_MANIFEST_PATH",
    "PreparedKernelPack",
    "PreparedKernelRecord",
    "PreparedModelBundle",
    "PreparedModelBundleError",
    "load_prepared_model_bundle",
    "read_prepared_model_bundle",
    "write_prepared_model_bundle",
]
