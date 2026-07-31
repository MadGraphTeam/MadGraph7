# SPDX-License-Identifier: 0BSD
"""Canonical source identity for the native pyAmpliCol build."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

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
# Release sdists deliberately omit contributor-only resolution state. It is
# still part of the unchanged candidate identity, but not of the reproducible
# release-native source closure.
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


def canonical_candidate_cargo_config_bytes(data: bytes) -> bytes:
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


def canonical_candidate_install_state_bytes(
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


def canonical_release_cargo_lock_bytes(root: Path, data: bytes) -> bytes:
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


def native_build_inputs_digest(
    root: Path,
    *,
    normalize_release_cargo_lock: bool = False,
) -> str:
    """Hash every checked-out input that can change the native runtime build."""

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
                canonical_candidate_cargo_config_bytes(cargo_config_data)
            )
            canonical_candidate_inputs[candidate_state] = (
                canonical_candidate_install_state_bytes(
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
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        data = canonical_candidate_inputs.get(path, path.read_bytes())
        if normalize_release_cargo_lock and path == root / "Cargo.lock":
            data = canonical_release_cargo_lock_bytes(root, data)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


__all__ = [
    "canonical_candidate_cargo_config_bytes",
    "canonical_candidate_install_state_bytes",
    "canonical_release_cargo_lock_bytes",
    "native_build_inputs_digest",
]
