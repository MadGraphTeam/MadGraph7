# SPDX-License-Identifier: 0BSD
"""Validate source-owned prepared models before wheel staging."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_ASSET_DIRECTORY = Path("src/pyamplicol/assets/prepared_models")
_EXPECTED_ARCHITECTURES = ("aarch64", "x86_64")
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
_EXPECTED_ID = "built-in-sm-jit-o3"


def _asset_names(architecture: str) -> tuple[str, str]:
    stem = f"{_EXPECTED_ID}-{architecture}"
    return f"{stem}.metadata.json", f"{stem}.pyamplicol-model"


_EXPECTED_FILES = frozenset(
    (
        "__init__.py",
        *(
            name
            for architecture in _EXPECTED_ARCHITECTURES
            for name in _asset_names(architecture)
        ),
    )
)


def stage_packaged_prepared_models(overlay: Path, mode: str) -> None:
    """Fail closed unless the wheel-owned prepared model matches the overlay."""

    if mode not in {"candidate", "release"}:
        raise RuntimeError(f"unsupported prepared-model build mode: {mode}")
    package_root = overlay / "src" / "pyamplicol"
    asset_root = overlay / _ASSET_DIRECTORY
    if not asset_root.is_dir() or asset_root.is_symlink():
        raise RuntimeError("wheel build input has no packaged prepared-model assets")
    actual_files = {path.name for path in asset_root.iterdir()}
    unsafe = [path for path in asset_root.iterdir() if path.is_symlink()]
    if unsafe:
        raise RuntimeError(
            "packaged prepared-model assets may not be symlinks: "
            + ", ".join(str(path) for path in sorted(unsafe))
        )
    if actual_files != _EXPECTED_FILES:
        missing = sorted(_EXPECTED_FILES - actual_files)
        unexpected = sorted(actual_files - _EXPECTED_FILES)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise RuntimeError(
            "packaged prepared-model asset set is invalid ("
            + "; ".join(details)
            + ")"
        )

    contract = _load_prepared_contract(package_root / "models" / "prepared.py")
    for architecture in _EXPECTED_ARCHITECTURES:
        metadata_name, expected_bundle_name = _asset_names(architecture)
        metadata = _load_json(
            asset_root / metadata_name,
            f"{architecture} prepared-model metadata",
        )
        _require_exact_keys(metadata, _METADATA_KEYS, "prepared-model metadata")
        if metadata.get("schema_version") != 1:
            raise RuntimeError("unsupported packaged prepared-model metadata schema")
        if (
            metadata.get("id") != _EXPECTED_ID
            or metadata.get("model") != "built-in-sm"
        ):
            raise RuntimeError("packaged prepared-model identity is invalid")
        bundle_name = _required_string(metadata.get("bundle"), "metadata.bundle")
        if bundle_name != expected_bundle_name:
            raise RuntimeError("packaged prepared-model bundle name is invalid")
        bundle_path = asset_root / bundle_name
        if not bundle_path.is_file() or bundle_path.is_symlink():
            raise RuntimeError("packaged prepared-model bundle must be a regular file")
        bundle_bytes = bundle_path.read_bytes()
        if metadata.get("bundle_size") != len(bundle_bytes):
            raise RuntimeError(
                "packaged prepared-model bundle size does not match metadata"
            )
        if metadata.get("bundle_sha256") != hashlib.sha256(bundle_bytes).hexdigest():
            raise RuntimeError(
                "packaged prepared-model bundle SHA-256 does not match metadata"
            )
        try:
            bundle = contract.load_prepared_model_bundle(bundle_path)
        except Exception as error:
            raise RuntimeError(
                f"packaged prepared-model bundle is invalid: {error}"
            ) from error
        _validate_bundle(
            bundle,
            metadata=metadata,
            expected_target={
                "portable": False,
                "word_bits": 64,
                "endianness": "little",
                "target_triple": f"symjit-storage-v3-{architecture}",
                "cpu_features": [],
            },
            package_root=package_root,
            overlay=overlay,
            mode=mode,
        )


def write_candidate_packaged_prepared_model_asset(
    overlay: Path,
    bundle_path: Path,
    output_directory: Path,
    *,
    architecture: str,
) -> tuple[Path, Path]:
    """Write one source-ready architecture pack and its derived metadata."""

    if architecture not in _EXPECTED_ARCHITECTURES:
        raise RuntimeError(f"unsupported prepared-model architecture: {architecture}")
    package_root = overlay / "src" / "pyamplicol"
    contributor_path = overlay / "dependencies" / "contributor-lock.toml"
    release_path = overlay / "dependencies" / "release-lock.toml"
    with contributor_path.open("rb") as stream:
        contributor = tomllib.load(stream)
    with release_path.open("rb") as stream:
        release = tomllib.load(stream)

    contract = _load_prepared_contract(package_root / "models" / "prepared.py")
    source_bundle = bundle_path.resolve(strict=True)
    bundle_bytes = source_bundle.read_bytes()
    try:
        bundle = contract.load_prepared_model_bundle(source_bundle)
    except Exception as error:
        raise RuntimeError(f"prepared-model bundle is invalid: {error}") from error
    pack = bundle.kernel_pack
    expected_target = {
        "portable": False,
        "word_bits": 64,
        "endianness": "little",
        "target_triple": f"symjit-storage-v3-{architecture}",
        "cpu_features": [],
    }
    if _plain_json(pack.target) != expected_target:
        raise RuntimeError("prepared-model bundle target does not match architecture")
    if bundle.backend != "jit" or pack.backend != "jit":
        raise RuntimeError("packaged built-in prepared model must use JIT")
    optimization = dict(pack.optimization_settings)
    if optimization.get("jit_optimization_level") != 3:
        raise RuntimeError("packaged built-in prepared model must use JIT O3")

    compiled = dict(bundle.compiled_model)
    compiled_producer = _mapping(
        compiled.get("producer"), "compiled_model.producer"
    )
    compiled_source = _mapping(compiled.get("source"), "compiled_model.source")
    package_version = _required_string(
        compiled_producer.get("pyamplicol"), "compiled_model producer version"
    )
    marker = "+candidate."
    if marker not in package_version:
        raise RuntimeError("source-ready prepared assets require a candidate bundle")
    candidate_fingerprint = package_version.rsplit(marker, maxsplit=1)[1]
    if len(candidate_fingerprint) != 12 or any(
        character not in "0123456789abcdef" for character in candidate_fingerprint
    ):
        raise RuntimeError("prepared bundle has an invalid candidate fingerprint")
    if pack.producer.get("version") != package_version:
        raise RuntimeError("prepared kernel-pack package version is inconsistent")

    compiled_schema = _literal_assignment(
        package_root / "_internal" / "versions.py",
        "COMPILED_MODEL_SCHEMA_VERSION",
    )
    model_compiler_version = _literal_assignment(
        package_root / "models" / "loading.py", "MODEL_COMPILER_VERSION"
    )
    model_compiler_digest = _model_compiler_digest(package_root)
    prepared_pack_compiler_digest = _prepared_pack_compiler_digest(package_root)
    source_digest = _built_in_source_digest(package_root)
    expected_compiled = {
        "compiled_model_schema_version": compiled_schema,
        "model_compiler_version": model_compiler_version,
        "model_compiler_sha256": model_compiler_digest,
    }
    for key, expected in expected_compiled.items():
        if compiled_producer.get(key) != expected:
            raise RuntimeError(f"prepared compiled-model producer {key} is stale")
    if compiled_source.get("kind") != "built-in-sm":
        raise RuntimeError("prepared bundle does not contain the built-in SM")
    if compiled_source.get("digest") != source_digest:
        raise RuntimeError("prepared built-in model source digest is stale")

    dependencies = {
        "symbolica_serialization_abi": _literal_assignment(
            package_root / "_internal" / "versions.py",
            "SYMBOLICA_SERIALIZATION_ABI",
        ),
        "symbolica_version": contributor["symbolica"]["candidate_version"],
        "symjit_application_abi": _literal_assignment(
            package_root / "_internal" / "versions.py",
            "SYMJIT_APPLICATION_ABI",
        ),
        "symjit_version": contributor["symjit"]["candidate_version"],
        "ufo_model_loader_version": release["ufo_model_loader"][
            "required_version"
        ],
    }
    expected_pack_dependencies = {
        "symbolica_serialization": dependencies["symbolica_serialization_abi"],
        "symbolica_version": dependencies["symbolica_version"],
        "symjit_application": dependencies["symjit_application_abi"],
    }
    for key, expected in expected_pack_dependencies.items():
        if pack.dependency_abis.get(key) != expected:
            raise RuntimeError(f"prepared kernel-pack dependency {key} is stale")

    metadata_name, bundle_name = _asset_names(architecture)
    metadata: dict[str, object] = {
        "backend": "jit",
        "build_contract": {
            "candidate_fingerprint": candidate_fingerprint,
            "mode": "candidate",
            "sources": {
                "symbolica": contributor["symbolica"]["candidate_revision"],
                "symbolica-community": contributor["symbolica"][
                    "community_revision"
                ],
                "symjit": contributor["symjit"]["candidate_revision"],
            },
        },
        "bundle": bundle_name,
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bundle_size": len(bundle_bytes),
        "dependencies": dependencies,
        "eager_kernel_abi": _literal_assignment(
            package_root / "models" / "prepared.py", "EAGER_KERNEL_ABI"
        ),
        "id": _EXPECTED_ID,
        "jit_optimization_level": 3,
        "kernel_count": len(pack.kernels),
        "model": "built-in-sm",
        "prepared_model_bundle_schema": _literal_assignment(
            package_root / "models" / "prepared.py",
            "PREPARED_MODEL_BUNDLE_SCHEMA_VERSION",
        ),
        "producer": {
            "compiled_model_schema": compiled_schema,
            "model_compiler_sha256": model_compiler_digest,
            "model_compiler_version": model_compiler_version,
            "model_source_digest": source_digest,
            "package_version": package_version,
            "prepared_pack_compiler_sha256": prepared_pack_compiler_digest,
        },
        "schema_version": 1,
        "target": expected_target,
    }

    destination = output_directory.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    output_bundle = destination / bundle_name
    output_metadata = destination / metadata_name
    for path in (output_bundle, output_metadata):
        if path.exists():
            raise RuntimeError(f"source-ready prepared asset already exists: {path}")
    temporary_bundle = output_bundle.with_name(f".{output_bundle.name}.tmp")
    temporary_metadata = output_metadata.with_name(f".{output_metadata.name}.tmp")
    temporary_bundle.write_bytes(bundle_bytes)
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_bundle.replace(output_bundle)
    temporary_metadata.replace(output_metadata)
    return output_metadata, output_bundle


def _validate_bundle(
    bundle: Any,
    *,
    metadata: Mapping[str, object],
    expected_target: Mapping[str, object],
    package_root: Path,
    overlay: Path,
    mode: str,
) -> None:
    pack = bundle.kernel_pack
    if bundle.backend != "jit" or metadata.get("backend") != "jit":
        raise RuntimeError("packaged built-in prepared model must use the JIT backend")
    if metadata.get("jit_optimization_level") != 3:
        raise RuntimeError("packaged built-in prepared model must use JIT O3")
    optimization = dict(pack.optimization_settings)
    if (
        optimization.get("backend") != "jit"
        or optimization.get("jit_optimization_level") != 3
    ):
        raise RuntimeError("prepared kernel pack does not record JIT O3 settings")
    target = _plain_json(pack.target)
    if target != expected_target or metadata.get("target") != expected_target:
        raise RuntimeError(
            "packaged prepared model target does not match its architecture asset"
        )
    kernel_count = len(pack.kernels)
    if metadata.get("kernel_count") != kernel_count or kernel_count == 0:
        raise RuntimeError("packaged prepared-model kernel count is invalid")

    eager_abi = _literal_assignment(
        package_root / "models" / "prepared.py", "EAGER_KERNEL_ABI"
    )
    bundle_schema = _literal_assignment(
        package_root / "models" / "prepared.py",
        "PREPARED_MODEL_BUNDLE_SCHEMA_VERSION",
    )
    if metadata.get("eager_kernel_abi") != eager_abi:
        raise RuntimeError("packaged prepared-model eager kernel ABI is stale")
    if metadata.get("prepared_model_bundle_schema") != bundle_schema:
        raise RuntimeError("packaged prepared-model container schema is stale")
    if bundle.manifest.get("eager_kernel_abi") != eager_abi:
        raise RuntimeError("prepared bundle declares an incompatible eager kernel ABI")
    if bundle.manifest.get("schema_version") != bundle_schema:
        raise RuntimeError("prepared bundle declares an incompatible container schema")

    build_contract = _mapping(
        metadata.get("build_contract"), "metadata.build_contract"
    )
    if build_contract.get("mode") != mode:
        raise RuntimeError(
            "packaged prepared model was built for "
            f"{build_contract.get('mode')!r}, not the active {mode!r} dependency mode; "
            "regenerate it with the active exact dependency lock"
        )
    expected_fingerprint: str | None = None
    if mode == "candidate":
        info = _load_json(
            package_root / "_build_info.json", "candidate package build info"
        )
        expected_fingerprint = _required_string(
            info.get("candidate_fingerprint"),
            "candidate package build info fingerprint",
        )
    if build_contract.get("candidate_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "packaged prepared model does not match the active exact dependency lock; "
            "regenerate the built-in JIT O3 prepared-model asset"
        )

    compiled = dict(bundle.compiled_model)
    producer = _mapping(compiled.get("producer"), "compiled_model.producer")
    source = _mapping(compiled.get("source"), "compiled_model.source")
    model = _mapping(compiled.get("model"), "compiled_model.model")
    if model.get("name") != "built-in-sm" or source.get("kind") != "built-in-sm":
        raise RuntimeError("packaged prepared model does not contain the built-in SM")

    compiled_schema = _literal_assignment(
        package_root / "_internal" / "versions.py",
        "COMPILED_MODEL_SCHEMA_VERSION",
    )
    compiler_version = _literal_assignment(
        package_root / "models" / "loading.py", "MODEL_COMPILER_VERSION"
    )
    compiler_digest = _model_compiler_digest(package_root)
    prepared_pack_compiler_digest = _prepared_pack_compiler_digest(package_root)
    source_digest = _built_in_source_digest(package_root)
    packaged_producer = _mapping(metadata.get("producer"), "metadata.producer")
    expected_producer = {
        "compiled_model_schema": compiled_schema,
        "model_compiler_version": compiler_version,
        "model_compiler_sha256": compiler_digest,
        "prepared_pack_compiler_sha256": prepared_pack_compiler_digest,
        "model_source_digest": source_digest,
        "package_version": _expected_package_version(overlay, mode),
    }
    for key, expected in expected_producer.items():
        if packaged_producer.get(key) != expected:
            raise RuntimeError(
                f"packaged prepared-model producer {key} is stale: "
                f"expected {expected!r}, got {packaged_producer.get(key)!r}"
            )
    if producer.get("compiled_model_schema_version") != compiled_schema:
        raise RuntimeError("prepared compiled-model schema is stale")
    if producer.get("model_compiler_version") != compiler_version:
        raise RuntimeError("prepared model compiler version is stale")
    if producer.get("model_compiler_sha256") != compiler_digest:
        raise RuntimeError("prepared model compiler digest is stale")
    if source.get("digest") != source_digest:
        raise RuntimeError("prepared built-in model source digest is stale")
    if producer.get("pyamplicol") != expected_producer["package_version"]:
        raise RuntimeError("prepared compiled-model package version is stale")
    if pack.producer.get("version") != expected_producer["package_version"]:
        raise RuntimeError("prepared kernel-pack package version is stale")
    if pack.producer.get("compiled_model_schema") != compiled_schema:
        raise RuntimeError("prepared kernel-pack compiled-model schema is stale")
    if pack.producer.get("model_compiler_version") != compiler_version:
        raise RuntimeError("prepared kernel-pack model compiler version is stale")
    if pack.provenance.get("model_name") != "built-in-sm":
        raise RuntimeError("prepared kernel-pack model provenance is invalid")
    if pack.provenance.get("compiled_model_digest") != source_digest:
        raise RuntimeError("prepared kernel-pack source provenance is stale")

    dependencies = _mapping(metadata.get("dependencies"), "metadata.dependencies")
    _validate_dependency_contract(
        dependencies,
        pack_dependency_abis=pack.dependency_abis,
        package_root=package_root,
        overlay=overlay,
        mode=mode,
    )
    for kernel in pack.kernels:
        manifest = kernel.f64_evaluator_manifest
        expected = {
            "kind": "symjit-application-evaluator",
            "backend": "jit",
            "runtime_capability": "symjit.application.complex-f64.v1",
            "application_abi": dependencies["symjit_application_abi"],
            "compiler_type": "native",
            "translation_mode": "indirect",
            "optimization_level": 3,
            "word_bits": 64,
            "endianness": "little",
            "required_defuns": (),
        }
        for key, expected_value in expected.items():
            actual = manifest.get(key)
            if key == "required_defuns" and isinstance(actual, Sequence):
                actual = tuple(actual)
            if actual != expected_value:
                raise RuntimeError(
                    f"prepared kernel {kernel.kernel_id} has incompatible {key}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )


def _validate_dependency_contract(
    dependencies: Mapping[str, object],
    *,
    pack_dependency_abis: Mapping[str, object],
    package_root: Path,
    overlay: Path,
    mode: str,
) -> None:
    with (overlay / "dependencies" / "release-lock.toml").open("rb") as stream:
        release = tomllib.load(stream)
    symbolica_version = release["symbolica"]["python_version"]
    if mode == "candidate":
        with (overlay / "dependencies" / "contributor-lock.toml").open(
            "rb"
        ) as stream:
            contributor = tomllib.load(stream)
        symbolica_version = contributor["symbolica"]["candidate_version"]
    expected = {
        "symbolica_version": symbolica_version,
        "ufo_model_loader_version": release["ufo_model_loader"]["required_version"],
        "symjit_version": _symjit_version(overlay),
        "symbolica_serialization_abi": _literal_assignment(
            package_root / "_internal" / "versions.py",
            "SYMBOLICA_SERIALIZATION_ABI",
        ),
        "symjit_application_abi": _literal_assignment(
            package_root / "_internal" / "versions.py",
            "SYMJIT_APPLICATION_ABI",
        ),
    }
    for key, value in expected.items():
        if dependencies.get(key) != value:
            raise RuntimeError(
                f"packaged prepared-model dependency {key} is stale: "
                f"expected {value!r}, got {dependencies.get(key)!r}"
            )
    pack_expected = {
        "symbolica_version": expected["symbolica_version"],
        "symbolica_serialization": expected["symbolica_serialization_abi"],
        "symjit_application": expected["symjit_application_abi"],
    }
    for key, value in pack_expected.items():
        if pack_dependency_abis.get(key) != value:
            raise RuntimeError(
                f"prepared kernel-pack dependency ABI {key} is stale"
            )


def _symjit_version(overlay: Path) -> str:
    with (overlay / "Cargo.lock").open("rb") as stream:
        cargo_lock = tomllib.load(stream)
    packages = cargo_lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("Cargo.lock has no package array")
    matches = [
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == "symjit"
    ]
    if len(matches) != 1:
        raise RuntimeError("Cargo.lock must contain exactly one SymJIT package")
    return _required_string(matches[0], "Cargo.lock SymJIT version")


def _expected_package_version(overlay: Path, mode: str) -> str:
    if mode == "candidate":
        info = _load_json(
            overlay / "src" / "pyamplicol" / "_build_info.json",
            "candidate package build info",
        )
        return _required_string(info.get("version"), "candidate package version")
    with (overlay / "Cargo.toml").open("rb") as stream:
        cargo = tomllib.load(stream)
    return _required_string(
        cargo["workspace"]["package"]["version"], "Cargo package version"
    )


def _model_compiler_digest(package_root: Path) -> str:
    paths = sorted(
        (
            *(package_root / "models").glob("*.py"),
            *(package_root / "_internal" / "physics").glob("*.py"),
            package_root / "processes" / "core_syntax.py",
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prepared_pack_compiler_digest(package_root: Path) -> str:
    paths = sorted(
        {
            *(package_root / "models").glob("prepared*.py"),
            *(package_root / "evaluators").glob("symbolica*.py"),
            *(package_root / "config").glob("*.py"),
            package_root / "_internal" / "physics" / "symbols.py",
            package_root / "_internal" / "versions.py",
        }
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _built_in_source_digest(package_root: Path) -> str:
    implementation = hashlib.sha256()
    for path in sorted((package_root / "models" / "builtin").glob("*.py")):
        implementation.update(path.name.encode("utf-8") + b"\0")
        implementation.update(path.read_bytes())
        implementation.update(b"\0")
    digest = hashlib.sha256()
    digest.update(b"built-in-sm\0")
    digest.update(implementation.hexdigest().encode("ascii"))
    return digest.hexdigest()


def _literal_assignment(path: Path, name: str) -> str | int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            break
        return value
    raise RuntimeError(f"could not read literal {name} from {path}")


def _load_prepared_contract(path: Path) -> ModuleType:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    name = f"_pyamplicol_build_prepared_contract_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prepared-model contract from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_json(path: Path, context: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _required_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a nonempty string")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown: " + ", ".join(sorted(unknown)))
        raise RuntimeError(f"{context} fields are invalid ({'; '.join(details)})")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    return value


__all__ = [
    "stage_packaged_prepared_models",
    "write_candidate_packaged_prepared_model_asset",
]
