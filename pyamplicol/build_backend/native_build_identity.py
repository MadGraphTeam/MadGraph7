# SPDX-License-Identifier: 0BSD
"""Canonical source identity for the native pyAmpliCol build."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
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
_CANDIDATE_SOURCE_NAMES = frozenset(
    {
        "gammaloop",
        "ratatui-ffi",
        "symbolica",
        "symbolica-community",
        "symjit",
    }
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


def _canonical_candidate_sources(raw_sources: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw_sources, Mapping):
        raise RuntimeError("candidate installer state sources must be an object")
    source_names = set(raw_sources)
    if any(not isinstance(name, str) or not name for name in source_names):
        raise RuntimeError("candidate installer source names must be non-empty strings")
    allowed_source_names = _CANDIDATE_SOURCE_NAMES | {"legacy-amplicol"}
    if (
        not _CANDIDATE_SOURCE_NAMES.issubset(source_names)
        or not source_names.issubset(allowed_source_names)
    ):
        raise RuntimeError(
            "candidate installer state has an incomplete or unexpected source map"
        )
    sources: dict[str, dict[str, str]] = {}
    for name in sorted(source_names):
        raw_source = raw_sources[name]
        description = f"candidate installer source {name}"
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(f"{description} must be an object")
        allowed_fields = {"url", "revision", "branch"}
        required_fields = {"url", "revision"}
        if not required_fields.issubset(raw_source) or not set(raw_source).issubset(
            allowed_fields
        ):
            raise RuntimeError(
                f"{description} must contain url/revision and optional branch only"
            )
        source = {
            "url": _required_string(raw_source, "url", description=description),
            "revision": _required_hex(
                raw_source,
                "revision",
                length=40,
                description=description,
            ),
        }
        if "branch" in raw_source:
            source["branch"] = _required_string(
                raw_source,
                "branch",
                description=description,
            )
        if name in _CANDIDATE_SOURCE_NAMES:
            sources[name] = source
    return sources


def canonical_candidate_install_state_bytes(
    root: Path,
    data: bytes,
    *,
    cargo_config_data: bytes,
) -> bytes:
    """Validate and canonicalize the minimal immutable-source install state."""

    del root, cargo_config_data

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid candidate installer state: {error}") from error
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if not isinstance(payload, dict) or set(payload) != {
        "publishable",
        "schema_version",
        "sources",
    }:
        raise RuntimeError(
            "candidate installer state must contain only schema_version, "
            "publishable, and sources"
        )
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or payload["publishable"] is not False
    ):
        raise RuntimeError(
            "candidate installer state must be a non-publishable schema-1 object"
        )
    sources = _canonical_candidate_sources(payload["sources"])
    return _canonical_json_bytes(
        {
            "publishable": False,
            "schema_version": 1,
            "sources": sources,
        }
    )


def canonical_release_cargo_lock_bytes(root: Path, data: bytes) -> bytes:
    """Return the ordinary immutable-Git Cargo lock without projection."""

    del root
    return data


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
