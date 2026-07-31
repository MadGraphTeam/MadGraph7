# SPDX-License-Identifier: 0BSD
"""Independent public, serialization, and evaluator runtime contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path, PurePosixPath
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
SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
NATIVE_COMPILED_DIRECT_APPLICATION_ABI = (
    "pyamplicol-native-compiled-direct-application-v1"
)
NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI = "pyamplicol-eager-native-direct-table-v1"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "pyamplicol-compiled-plane-kernel-v2"
EAGER_DIRECT_TABLE_DESCRIPTOR_ABI = "pyamplicol-eager-plane-table-descriptor-v1"
EAGER_DIRECT_TABLE_BINDING_ABI = "pyamplicol-eager-plane-table-binding-v2"
RECURRENCE_DIRECT_BINDING_ABI = "pyamplicol-recurrence-plane-binding-v2"

SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY = (
    "symbolica.legacy-jit-container.complex-f64.v1"
)
SYMBOLICA_CPP_RUNTIME_CAPABILITY = "symbolica.compiled-cpp.complex-f64.v1"
SYMBOLICA_ASM_RUNTIME_CAPABILITY = "symbolica.compiled-asm.complex-f64.v1"
EAGER_DAG_F64_RUNTIME_CAPABILITY = "rusticol.eager-dag.complex-f64.v1"
EAGER_RUNTIME_LAYOUT_F64_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY = "eager-direct-arena-v1"
EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY = "rusticol.eager-dag.lc-topology-replay.v1"
RECURRENCE_BUILDER_INPUT_ABI = "pyamplicol-recurrence-builder-input-v2"
RECURRENCE_PLAN_ABI = "pyamplicol-recurrence-plan-v2"
RECURRENCE_RUNTIME_LAYOUT_ABI = "pyamplicol-recurrence-runtime-layout-v2"
RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY = (
    "rusticol.recurrence-direct-arena.complex-f64.v1"
)
RECURRENCE_COLOR_RUNTIME_CAPABILITY = "rusticol.recurrence-color.lc.v1"
RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY = (
    "rusticol.recurrence-color.contracted.v1"
)
COMPILED_RUNTIME_SELECTORS_CAPABILITY = "rusticol.compiled.runtime-selectors.v1"
COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY = "compiled-plane-arena-v1"
COMPILED_HELICITY_DUAL_LANE_CAPABILITY = "rusticol.compiled.helicity-dual-lane.v1"
COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY = (
    "rusticol.compiled.helicity-selector-union.v1"
)
COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY = (
    "rusticol.compiled.helicity-primary-recurrence.v1"
)
COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY = "rusticol.compiled.color-topology-lanes.v1"
COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY = (
    "rusticol.compiled.color-contraction-walsh.v1"
)
COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY = (
    "rusticol.compiled.color-contraction-walsh-c2k.v1"
)
EVALUATOR_RUNTIME_CAPABILITIES = frozenset(
    {
        COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
        COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
        RECURRENCE_COLOR_RUNTIME_CAPABILITY,
        RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }
)
KNOWN_EVALUATOR_RUNTIME_CAPABILITIES = EVALUATOR_RUNTIME_CAPABILITIES | frozenset(
    {
        # Historical eager plan-v2 artifacts remain parseable for inspection
        # and actionable migration errors, but are not executable capabilities.
        EAGER_DAG_F64_RUNTIME_CAPABILITY,
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
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
    Path("dependencies/python-runtime-lock.toml"),
    Path("dependencies/release-lock.toml"),
)
_NATIVE_BUILD_INPUT_TREES = (
    Path("build_backend"),
    Path("dependencies/patches"),
    Path("rust"),
)
# Keep this release-sdist projection identical to
# build_backend/native_build_identity.py without importing build-only modules.
_RELEASE_OMITTED_NATIVE_BUILD_INPUTS = frozenset(
    {
        Path("build_backend/python_lock.py"),
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/contributor-lock.toml"),
        Path("dependencies/install-state.json"),
        Path("dependencies/python-runtime-lock.toml"),
    }
)
_NATIVE_BUILD_INPUT_SUFFIXES = {
    ".f90",
    ".h",
    ".hpp",
    ".json",
    ".patch",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
}
_CANDIDATE_CARGO_CONFIG = Path("dependencies/candidate-cargo-config.toml")
_CANDIDATE_INSTALL_STATE = Path("dependencies/install-state.json")
_CANDIDATE_PATCH_TARGETS = {
    "graphica": "dependencies/checkouts/symbolica/lib/graphica",
    "numerica": "dependencies/checkouts/symbolica/lib/numerica",
    "symbolica": "dependencies/checkouts/symbolica",
    "symjit": "dependencies/checkouts/symjit",
}
_CANDIDATE_STATE_LOCKS = {
    "candidate_lock_sha256": Path("dependencies/candidate-Cargo.lock"),
    "contributor_lock_sha256": Path("dependencies/contributor-lock.toml"),
    "python_runtime_lock_sha256": Path("dependencies/python-runtime-lock.toml"),
    "release_lock_sha256": Path("dependencies/release-lock.toml"),
}
_CANDIDATE_SOURCE_FIELDS = {
    "gammaloop": ("url", "revision", "worktree_sha256"),
    "ratatui-ffi": ("url", "revision", "worktree_sha256"),
    "symbolica": ("url", "revision", "worktree_sha256"),
    "symbolica-community": ("url", "revision", "worktree_sha256"),
    "symjit": (
        "url",
        "revision",
        "version",
        "archive_sha256",
        "patch_sha256",
        "worktree_sha256",
    ),
}
_CANDIDATE_PATCH_FIELDS = (
    "name",
    "target",
    "path",
    "sha256",
    "applies_to_revision",
)
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"candidate native build identity is not canonical JSON: {error}"
        ) from error


def _canonical_checkout_path(raw: str) -> str:
    normalized = raw.replace("\\", "/").rstrip("/")
    marker = "/dependencies/checkouts/"
    if marker in normalized:
        suffix = normalized.rsplit(marker, maxsplit=1)[1]
        if suffix:
            return f"dependencies/checkouts/{suffix}"
    prefix = "dependencies/checkouts/"
    if normalized.startswith(prefix) and len(normalized) > len(prefix):
        return normalized
    raise RuntimeError(
        "candidate Cargo patch paths must resolve below dependencies/checkouts"
    )


def _candidate_cargo_config_bytes(data: bytes) -> bytes:
    """Return the exact path-independent candidate Cargo patch contract."""

    try:
        payload = tomllib.loads(data.decode("utf-8"))
        patch = payload["patch"]
        crates_io = patch["crates-io"]
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        raise RuntimeError(f"invalid candidate Cargo config: {error}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"patch"}
        or not isinstance(patch, dict)
        or set(patch) != {"crates-io"}
        or not isinstance(crates_io, dict)
        or set(crates_io) != set(_CANDIDATE_PATCH_TARGETS)
    ):
        raise RuntimeError(
            "candidate Cargo config must contain exactly the locked crates.io "
            "patch table"
        )
    canonical: dict[str, Any] = {
        "patch": {"crates-io": {}}
    }
    canonical_patches = canonical["patch"]["crates-io"]
    for name, expected in _CANDIDATE_PATCH_TARGETS.items():
        entry = crates_io[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path"}
            or not isinstance(entry["path"], str)
        ):
            raise RuntimeError(
                f"candidate Cargo patch {name} must contain exactly one path"
            )
        observed = _canonical_checkout_path(entry["path"])
        if observed != expected:
            raise RuntimeError(
                f"candidate Cargo patch {name} must resolve to {expected}"
            )
        canonical_patches[name] = {"path": observed}
    return _canonical_json_bytes(canonical)


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{description} {key} must be a non-empty string")
    return value


def _required_hex(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
    description: str,
) -> str:
    value = _required_string(payload, key, description=description)
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(
            f"{description} {key} must be a lowercase {length}-digit hex value"
        )
    return value


def _validate_candidate_contributor_contract(
    root: Path,
    *,
    patches: list[dict[str, str]],
    sources: dict[str, dict[str, str]],
) -> None:
    contributor_path = root / "dependencies/contributor-lock.toml"
    try:
        with contributor_path.open("rb") as stream:
            contributor = tomllib.load(stream)
        expected_patches = contributor["patches"]
        symbolica = contributor["symbolica"]
        symjit = contributor["symjit"]
        gammaloop = contributor["gammaloop_candidate"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"candidate contributor contract is invalid: {error}"
        ) from error
    if (
        not isinstance(expected_patches, list)
        or any(not isinstance(patch, Mapping) for patch in expected_patches)
        or not isinstance(symbolica, Mapping)
        or not isinstance(symjit, Mapping)
        or not isinstance(gammaloop, Mapping)
    ):
        raise RuntimeError("candidate contributor contract has invalid tables")
    if expected_patches != patches:
        raise RuntimeError(
            "candidate installer patches do not match contributor-lock.toml"
        )
    dependency_root = (root / "dependencies").resolve()
    for patch in patches:
        path = root / "dependencies" / Path(*PurePosixPath(patch["path"]).parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(dependency_root)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"candidate installer patch is unavailable: {patch['path']}"
            ) from error
        if path.is_symlink() or not resolved.is_file():
            raise RuntimeError(
                f"candidate installer patch is not a regular file: {patch['path']}"
            )
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != patch["sha256"]:
            raise RuntimeError(
                f"candidate installer patch digest is stale: {patch['path']}"
            )
    patch_closure = hashlib.sha256(
        json.dumps(
            patches,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    try:
        expected_sources = {
            "gammaloop": {
                "revision": gammaloop["revision"],
                "url": gammaloop["source_url"],
            },
            "symbolica": {
                "revision": symbolica["candidate_revision"],
                "url": symbolica["source_url"],
            },
            "symbolica-community": {
                "revision": symbolica["community_revision"],
                "url": symbolica["community_url"],
            },
            "symjit": {
                "archive_sha256": symjit["archive_sha256"],
                "patch_sha256": patch_closure,
                "revision": symjit["candidate_revision"],
                "url": symjit["source_url"],
                "version": symjit["candidate_version"],
                "worktree_sha256": symjit["candidate_tree_sha256"],
            },
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"candidate contributor contract is incomplete: {error}"
        ) from error
    for name, expected in expected_sources.items():
        if any(sources[name].get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"candidate installer source {name} does not match "
                "contributor-lock.toml"
            )


def _candidate_install_state_bytes(
    root: Path,
    data: bytes,
    *,
    cargo_config_data: bytes,
) -> bytes:
    """Validate installer attestations and remove install-only volatility."""

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid candidate installer state: {error}") from error
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or payload.get("publishable") is not False
    ):
        raise RuntimeError(
            "candidate installer state must be a non-publishable schema-1 object"
        )
    created_utc = payload.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise RuntimeError("candidate installer state has no creation timestamp")

    cargo_config_sha256 = _required_hex(
        payload,
        "cargo_config_sha256",
        length=64,
        description="candidate installer state",
    )
    if cargo_config_sha256 != hashlib.sha256(cargo_config_data).hexdigest():
        raise RuntimeError(
            "candidate installer state does not match candidate-cargo-config.toml"
        )
    for key, relative in _CANDIDATE_STATE_LOCKS.items():
        expected = _required_hex(
            payload,
            key,
            length=64,
            description="candidate installer state",
        )
        path = root / relative
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(
                f"candidate installer state lock is unavailable: {relative}"
            ) from error
        if observed != expected:
            raise RuntimeError(
                f"candidate installer state does not match {relative.as_posix()}"
            )

    raw_patches = payload.get("patches")
    if not isinstance(raw_patches, list):
        raise RuntimeError("candidate installer state patches must be a list")
    patches: list[dict[str, str]] = []
    for index, raw_patch in enumerate(raw_patches):
        description = f"candidate installer patch {index}"
        if not isinstance(raw_patch, Mapping) or set(raw_patch) != set(
            _CANDIDATE_PATCH_FIELDS
        ):
            raise RuntimeError(
                f"{description} must contain exactly the patch contract fields"
            )
        patch = {
            field: _required_string(
                raw_patch,
                field,
                description=description,
            )
            for field in _CANDIDATE_PATCH_FIELDS
        }
        _required_hex(
            raw_patch,
            "sha256",
            length=64,
            description=description,
        )
        _required_hex(
            raw_patch,
            "applies_to_revision",
            length=40,
            description=description,
        )
        patch_path = PurePosixPath(patch["path"])
        if (
            patch_path.is_absolute()
            or not patch_path.parts
            or any(part in {"", ".", ".."} for part in patch_path.parts)
        ):
            raise RuntimeError(f"{description} path must be workspace-relative")
        patches.append(patch)

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, Mapping):
        raise RuntimeError("candidate installer state sources must be an object")
    source_names = set(raw_sources)
    allowed_source_names = set(_CANDIDATE_SOURCE_FIELDS) | {"legacy-amplicol"}
    if (
        not set(_CANDIDATE_SOURCE_FIELDS).issubset(source_names)
        or not source_names.issubset(allowed_source_names)
    ):
        raise RuntimeError(
            "candidate installer state has an incomplete or unexpected source map"
        )
    sources: dict[str, dict[str, str]] = {}
    for name, fields in _CANDIDATE_SOURCE_FIELDS.items():
        raw_source = raw_sources[name]
        description = f"candidate installer source {name}"
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(f"{description} must be an object")
        source = {
            field: _required_string(
                raw_source,
                field,
                description=description,
            )
            for field in fields
        }
        _required_hex(
            raw_source,
            "revision",
            length=40,
            description=description,
        )
        _required_hex(
            raw_source,
            "worktree_sha256",
            length=64,
            description=description,
        )
        if name == "symjit":
            _required_hex(
                raw_source,
                "archive_sha256",
                length=64,
                description=description,
            )
            _required_hex(
                raw_source,
                "patch_sha256",
                length=64,
                description=description,
            )
            if source["version"] != "2.22.0":
                raise RuntimeError(
                    "candidate installer source symjit must use version 2.22.0"
                )
        sources[name] = source

    _validate_candidate_contributor_contract(
        root,
        patches=patches,
        sources=sources,
    )
    return _canonical_json_bytes(
        {
            "patches": patches,
            "publishable": False,
            "schema_version": 1,
            "sources": sources,
        }
    )


def _release_cargo_lock_bytes(root: Path, data: bytes) -> bytes:
    """Canonicalize the two authenticated release SymJIT lock representations."""

    lock_path = root / "dependencies" / "release-lock.toml"
    try:
        with lock_path.open("rb") as stream:
            release = tomllib.load(stream)
        symjit = release["symjit"]
        version = symjit["version"]
        repository = symjit["repository"]
        revision = symjit["revision"]
        expected = symjit["release_cargo_lock_sha256"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"release native build identity has an invalid dependency lock: {error}"
        ) from error
    if (
        version != "2.22.0"
        or not isinstance(repository, str)
        or not repository
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError(
            "release native build identity has an invalid SymJIT 2.22 lock"
        )
    if hashlib.sha256(data).hexdigest() == expected:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("release Cargo.lock must be UTF-8") from error
    source = f"git+{repository}?rev={revision}#{revision}"
    marker = (
        "[[package]]\n"
        'name = "symjit"\n'
        f'version = "{version}"\n'
        f'source = "{source}"\n'
    )
    replacement = (
        "[[package]]\n"
        'name = "symjit"\n'
        f'version = "{version}"\n'
    )
    if text.count(marker) != 1:
        raise RuntimeError(
            "release Cargo.lock is neither the immutable SymJIT Git lock nor "
            "its authenticated path projection"
        )
    projected = text.replace(marker, replacement, 1).encode("utf-8")
    if hashlib.sha256(projected).hexdigest() != expected:
        raise RuntimeError(
            "release Cargo.lock path projection does not match release-lock.toml"
        )
    return projected


def _native_build_inputs_digest(
    root: Path,
    *,
    normalize_release_cargo_lock: bool = False,
) -> str:
    """Hash the small set of checkout inputs that determines the native build."""

    candidate_config = root / _CANDIDATE_CARGO_CONFIG
    candidate_state = root / _CANDIDATE_INSTALL_STATE
    canonical_candidate_inputs: dict[Path, bytes] = {}
    if not normalize_release_cargo_lock:
        if candidate_config.is_file() != candidate_state.is_file():
            raise RuntimeError(
                "candidate native build identity requires both "
                "candidate-cargo-config.toml and install-state.json"
            )
        if candidate_config.is_file():
            cargo_config_data = candidate_config.read_bytes()
            canonical_candidate_inputs[candidate_config] = (
                _candidate_cargo_config_bytes(cargo_config_data)
            )
            canonical_candidate_inputs[candidate_state] = (
                _candidate_install_state_bytes(
                    root,
                    candidate_state.read_bytes(),
                    cargo_config_data=cargo_config_data,
                )
            )

    paths = [
        root / relative
        for relative in _NATIVE_BUILD_INPUT_FILES
        if not (
            normalize_release_cargo_lock
            and relative in _RELEASE_OMITTED_NATIVE_BUILD_INPUTS
        )
    ]
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
            and not (
                normalize_release_cargo_lock
                and path.relative_to(root) in _RELEASE_OMITTED_NATIVE_BUILD_INPUTS
            )
        )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = canonical_candidate_inputs.get(path, path.read_bytes())
        if normalize_release_cargo_lock and path == root / "Cargo.lock":
            data = _release_cargo_lock_bytes(root, data)
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
    mode = contract.get("mode")
    native_digest = contract.get("native_build_inputs_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (extension_name, extension_sha256, native_digest)
    ) or mode not in {"candidate", "release"}:
        raise RuntimeError(
            "source runtime provenance is incomplete; rerun `just dev-install`"
        )
    valid_build_mode = (
        mode == "candidate" and payload.get("publishable") is False
    ) or (
        mode == "release"
        and (
            payload.get("publishable") is True
            or (
                payload.get("publishable") is False
                and payload.get("release_prepared_model_bootstrap") is True
            )
        )
    )
    if not valid_build_mode:
        raise RuntimeError(
            "source runtime provenance has an inconsistent build mode; "
            "rerun `just dev-install`"
        )
    if (
        len(extension_sha256) != 64
        or any(character not in "0123456789abcdef" for character in extension_sha256)
        or len(native_digest) != 64
        or any(character not in "0123456789abcdef" for character in native_digest)
        or payload.get("native_build_inputs_sha256") != native_digest
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
    if (
        _native_build_inputs_digest(
            source_root,
            normalize_release_cargo_lock=mode == "release",
        )
        != native_digest
    ):
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


def active_native_source_identity() -> tuple[str, str]:
    """Return the authenticated source revision and native-input digest."""

    build_info = _active_build_info()
    if build_info is None:
        raise RuntimeError(
            "active native build provenance is unavailable; reinstall pyAmpliCol"
        )
    source_revision = build_info.get("source_revision")
    native_digest = build_info.get("native_build_inputs_sha256")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
        or not isinstance(native_digest, str)
        or len(native_digest) != 64
        or any(character not in "0123456789abcdef" for character in native_digest)
    ):
        raise RuntimeError(
            "active native source identity is incomplete; reinstall pyAmpliCol"
        )
    return source_revision, native_digest


def active_source_revision() -> str | None:
    """Return the revision bound to the active wheel/source runtime, if any."""

    build_info = _active_build_info()
    if build_info is None:
        return None
    revision = build_info.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        return None
    return revision


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
    if build_info is None:
        return
    source_runtime = build_info.get("source_runtime")
    if (
        build_info.get("publishable") is not False
        and not isinstance(source_runtime, dict)
    ):
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
    "COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY",
    "COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY",
    "COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY",
    "COMPILED_HELICITY_DUAL_LANE_CAPABILITY",
    "COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY",
    "COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY",
    "COMPILED_MODEL_SCHEMA_VERSION",
    "COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY",
    "COMPILED_PLANE_DIRECT_APPLICATION_ABI",
    "COMPILED_RUNTIME_SELECTORS_CAPABILITY",
    "C_ABI_VERSION",
    "EAGER_DAG_F64_RUNTIME_CAPABILITY",
    "EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "EAGER_DIRECT_TABLE_BINDING_ABI",
    "EAGER_DIRECT_TABLE_DESCRIPTOR_ABI",
    "EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY",
    "EAGER_RUNTIME_LAYOUT_F64_CAPABILITY",
    "EVALUATOR_RUNTIME_CAPABILITIES",
    "KNOWN_EVALUATOR_RUNTIME_CAPABILITIES",
    "PROCESS_ARTIFACT_SCHEMA_VERSION",
    "PYTHON_API_VERSION",
    "RECURRENCE_BUILDER_INPUT_ABI",
    "RECURRENCE_COLOR_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_BACKEND_ABI",
    "RECURRENCE_DIRECT_BINDING_ABI",
    "RECURRENCE_DIRECT_TEMPLATE_ABI",
    "RECURRENCE_PLAN_ABI",
    "RECURRENCE_RUNTIME_LAYOUT_ABI",
    "RUNTIME_PHYSICS_SCHEMA_VERSION",
    "SYMBOLICA_ASM_RUNTIME_CAPABILITY",
    "SYMBOLICA_CPP_RUNTIME_CAPABILITY",
    "SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY",
    "SYMBOLICA_SERIALIZATION_ABI",
    "SYMJIT_APPLICATION_ABI",
    "SYMJIT_F64_RUNTIME_CAPABILITY",
    "SYMJIT_PLANE_APPLICATION_ABI",
    "TOML_SCHEMA_VERSION",
    "active_native_source_identity",
    "active_source_revision",
    "package_version",
    "verify_native_module",
]
