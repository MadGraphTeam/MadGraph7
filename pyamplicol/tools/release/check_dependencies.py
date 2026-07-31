#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate the small published or contributor dependency contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "release-lock.toml"
CONTRIBUTOR_LOCK_PATH = ROOT / "dependencies" / "contributor-lock.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CARGO_LOCK_PATH = ROOT / "Cargo.lock"
RUST_TOOLCHAIN_PATH = ROOT / "rust-toolchain.toml"
STATE_PATH = ROOT / "dependencies" / "install-state.json"
CANDIDATE_LOCK_PATH = ROOT / "dependencies" / "candidate-Cargo.lock"
CARGO_CONFIG_PATH = ROOT / "dependencies" / "candidate-cargo-config.toml"
CHECKOUTS_PATH = ROOT / "dependencies" / "checkouts"

_REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_LOCAL_CRATES = {"rusticol-capi", "rusticol-core", "rusticol-python"}
_CANDIDATE_LOCAL_CRATES = {
    *_LOCAL_CRATES,
    "graphica",
    "numerica",
    "symbolica",
    "symjit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_CANONICAL_NAME = re.compile(r"[-_.]+")
_SOURCE_TREE_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
}
_CANDIDATE_ABIS = {
    "symbolica_serialization": "symbolica-bincode2-v1",
    "symjit_application": "symjit-application-storage-v3",
    "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
}


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


def canonicalize_name(value: str) -> str:
    return _CANONICAL_NAME.sub("-", value).lower()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return payload


def _load_lock() -> dict[str, Any]:
    payload = _load_toml(LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError("dependencies/release-lock.toml must use schema_version = 1")
    return payload


def _load_contributor_lock() -> dict[str, Any]:
    payload = _load_toml(CONTRIBUTOR_LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "dependencies/contributor-lock.toml must use schema_version = 1"
        )
    return payload


def _locked_python_dependencies(lock: dict[str, Any]) -> dict[str, str]:
    raw = lock.get("python_dependencies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("release lock must list exact Python dependencies")
    dependencies: dict[str, str] = {}
    ordered_names: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"distribution", "version"}:
            raise ValueError(
                f"python_dependencies[{index}] must contain only distribution/version"
            )
        distribution = entry["distribution"]
        version = entry["version"]
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ValueError(
                f"python_dependencies[{index}] needs string distribution/version"
            )
        name = canonicalize_name(distribution)
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ValueError(f"invalid locked version for {name}: {version}") from error
        if name in dependencies:
            raise ValueError(f"release lock repeats Python dependency {name}")
        ordered_names.append(name)
        dependencies[name] = normalized_version
    if ordered_names != sorted(ordered_names):
        raise ValueError("release-lock Python dependencies must be name-sorted")
    return dependencies


def _project_python_dependencies() -> dict[str, str]:
    project = _load_toml(PYPROJECT_PATH).get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table")
    raw = project.get("dependencies")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("pyproject runtime dependencies must be strings")
    dependencies: dict[str, str] = {}
    for value in raw:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as error:
            raise ValueError(f"invalid pyproject dependency: {value}") from error
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            raise ValueError(
                f"release runtime dependency must be one exact published pin: {value}"
            )
        try:
            version = str(Version(specifiers[0].version))
        except InvalidVersion as error:
            raise ValueError(
                f"invalid pyproject dependency version: {value}"
            ) from error
        if name in dependencies:
            raise ValueError(f"pyproject repeats runtime dependency {name}")
        dependencies[name] = version
    return dependencies


def _release_contract_issues(lock: dict[str, Any]) -> list[GateIssue]:
    issues: list[GateIssue] = []
    abis = lock.get("abis")
    if not isinstance(abis, dict) or any(
        abis.get(name) != value for name, value in _CANDIDATE_ABIS.items()
    ):
        issues.append(
            GateIssue(
                "release-abi-contract",
                "release-lock.toml must pin the Symbolica serialization, SymJIT "
                "storage, and SymJIT plane-application ABIs",
            )
        )
    forbidden_tables = {"legal_status", "python_runtime_lock"}.intersection(lock)
    if forbidden_tables:
        issues.append(
            GateIssue(
                "release-lock-scope",
                "release-lock.toml contains retired release machinery: "
                + ", ".join(sorted(forbidden_tables)),
            )
        )
    try:
        locked = _locked_python_dependencies(lock)
        project = _project_python_dependencies()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [*issues, GateIssue("python-dependency-contract", str(error))]
    if locked != project:
        issues.append(
            GateIssue(
                "python-dependency-contract",
                "pyproject exact runtime pins disagree with release-lock.toml; "
                f"lock={locked}, pyproject={project}",
            )
        )

    symbolica = lock.get("symbolica")
    symjit = lock.get("symjit")
    loader = lock.get("ufo_model_loader")
    if (
        not isinstance(symbolica, dict)
        or not isinstance(symjit, dict)
        or not isinstance(loader, dict)
    ):
        issues.append(
            GateIssue(
                "published-dependency-contract",
                "release lock needs Symbolica, SymJIT, and ufo-model-loader "
                "compatibility data",
            )
        )
        return issues
    allowed_symbolica = {
        "python_distribution",
        "python_version",
        "rust_crate",
        "rust_version",
        "serialization_abi",
    }
    allowed_symjit = {
        "version",
        "repository",
        "revision",
        "source_url",
        "archive_prefix",
        "archive_sha256",
        "source_tree_sha256",
        "configured_tree_sha256",
        "release_cargo_lock_sha256",
        "patches",
    }
    allowed_loader = {
        "python_distribution",
        "required_version",
    }
    if (
        set(symbolica) != allowed_symbolica
        or set(symjit) != allowed_symjit
        or set(loader) != allowed_loader
    ):
        issues.append(
            GateIssue(
                "release-lock-scope",
                "release dependency sections contain fields outside their approved "
                "compatibility and source contracts",
            )
        )
    if (
        not isinstance(symjit.get("version"), str)
        or not isinstance(symjit.get("repository"), str)
        or not str(symjit["repository"]).startswith("https://github.com/")
        or not str(symjit["repository"]).endswith(".git")
        or not isinstance(symjit.get("revision"), str)
        or _GIT_REVISION.fullmatch(str(symjit["revision"])) is None
    ):
        issues.append(
            GateIssue(
                "symjit-source-contract",
                "SymJIT must use one HTTPS Git repository and immutable full revision",
            )
        )
    symbolica_name = canonicalize_name(str(symbolica.get("python_distribution", "")))
    loader_name = canonicalize_name(str(loader.get("python_distribution", "")))
    if locked.get(symbolica_name) != str(symbolica.get("python_version", "")):
        issues.append(
            GateIssue(
                "symbolica-pin",
                "Symbolica compatibility data disagrees with the exact Python pin",
            )
        )
    if locked.get(loader_name) != str(loader.get("required_version", "")):
        issues.append(
            GateIssue(
                "ufo-loader-pin",
                "ufo-model-loader compatibility data disagrees with the exact pin",
            )
        )
    issues.extend(_release_symjit_source_issues(lock))
    return issues


def _cargo_packages(path: Path) -> list[dict[str, Any]]:
    payload = _load_toml(path)
    packages = payload.get("package")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        raise ValueError(f"{path} has no Cargo package array")
    return packages


def _registry_source_issues(
    packages: list[dict[str, Any]],
    *,
    local_crates: set[str],
    prefix: str,
    exact_git_sources: dict[str, tuple[str, str]] | None = None,
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    exact_git_sources = exact_git_sources or {}
    for package in packages:
        name = str(package.get("name", ""))
        source = package.get("source")
        if name in local_crates:
            if source is not None:
                issues.append(
                    GateIssue(
                        f"{prefix}-cargo-local-source",
                        f"local crate {name} unexpectedly has source {source}",
                    )
                )
            continue
        git_contract = exact_git_sources.get(name)
        if git_contract is not None:
            expected_version, expected_source = git_contract
            if (
                str(package.get("version")) != expected_version
                or source != expected_source
                or package.get("checksum") is not None
            ):
                issues.append(
                    GateIssue(
                        f"{prefix}-cargo-git-source",
                        f"Cargo.lock package {name} does not match its exact "
                        "immutable Git contract",
                    )
                )
            continue
        if source != _REGISTRY_SOURCE:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-nonregistry",
                    f"Cargo.lock package {name} is not an exact crates.io package",
                )
            )
            continue
        checksum = package.get("checksum")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-checksum",
                    f"Cargo.lock package {name} has no crates.io checksum",
                )
            )
    return issues


def _cargo_manifest_pin_issues(lock: dict[str, Any]) -> list[GateIssue]:
    symbolica = lock["symbolica"]
    symjit = lock["symjit"]
    try:
        root = _load_toml(ROOT / "Cargo.toml")
        core = _load_toml(ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml")
        root_dependencies = root["workspace"]["dependencies"]
        core_dependencies = core["dependencies"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-manifest", f"invalid Cargo manifest: {error}")]
    issues: list[GateIssue] = []
    root_patch = root.get("patch")
    expected_patch = {
        "git": symjit["repository"],
        "rev": symjit["revision"],
    }
    if (
        not isinstance(root_patch, dict)
        or not isinstance(root_patch.get("crates-io"), dict)
        or set(root_patch["crates-io"]) != {"symjit"}
        or root_patch["crates-io"].get("symjit") != expected_patch
    ):
        issues.append(
            GateIssue(
                "release-cargo-patch",
                "release Cargo.toml must redirect crates.io SymJIT to the exact "
                "release-lock Git revision",
            )
        )
    symbolica_version = f"={symbolica['rust_version']}"
    for table in (root_dependencies, core_dependencies):
        entry = table.get("symbolica") if isinstance(table, dict) else None
        if not isinstance(entry, dict) or entry.get("version") != symbolica_version:
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    "Cargo manifest must require "
                    f"symbolica {symbolica_version} exactly",
                )
            )
        elif "git" in entry or "path" in entry:
            issues.append(
                GateIssue(
                    "release-cargo-source",
                    "release Cargo dependency symbolica must use crates.io",
                )
            )
    symjit_entry = (
        core_dependencies.get("symjit") if isinstance(core_dependencies, dict) else None
    )
    if (
        not isinstance(symjit_entry, dict)
        or symjit_entry.get("version") != f"={symjit['version']}"
        or "git" in symjit_entry
        or "rev" in symjit_entry
        or "path" in symjit_entry
    ):
        issues.append(
            GateIssue(
                "release-cargo-source",
                "rusticol-core must require the exact release-lock SymJIT version; "
                "the workspace patch owns its Git source",
            )
        )
    return issues


def _release_cargo_lock_issues(lock: dict[str, Any]) -> list[GateIssue]:
    try:
        packages = _cargo_packages(CARGO_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-lock", str(error))]
    symjit = lock.get("symjit", {})
    symjit_source = (
        f"git+{symjit.get('repository', '')}?rev={symjit.get('revision', '')}"
        f"#{symjit.get('revision', '')}"
    )
    lock_bytes = CARGO_LOCK_PATH.read_bytes()
    local_lock = (
        hashlib.sha256(lock_bytes).hexdigest()
        == symjit.get("release_cargo_lock_sha256")
    )
    if local_lock:
        issues = _registry_source_issues(
            packages,
            local_crates={*_LOCAL_CRATES, "symjit"},
            prefix="release",
        )
        matches = [
            package
            for package in packages
            if package.get("name") == "symjit"
            and package.get("version") == symjit.get("version")
            and package.get("source") is None
        ]
        if len(matches) != 1:
            issues.append(
                GateIssue(
                    "release-cargo-local-source",
                    "release-local Cargo.lock must resolve exactly one authenticated "
                    "SymJIT path package",
                )
            )
    else:
        issues = _registry_source_issues(
            packages,
            local_crates=_LOCAL_CRATES,
            prefix="release",
            exact_git_sources={
                "symjit": (str(symjit.get("version", "")), symjit_source)
            },
        )
        marker = (
            "[[package]]\n"
            'name = "symjit"\n'
            f'version = "{symjit.get("version", "")}"\n'
            f'source = "{symjit_source}"\n'
        )
        try:
            text = lock_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if text.count(marker) == 1:
            projected = text.replace(
                marker,
                (
                    "[[package]]\n"
                    'name = "symjit"\n'
                    f'version = "{symjit.get("version", "")}"\n'
                ),
                1,
            ).encode("utf-8")
            projected_sha256 = hashlib.sha256(projected).hexdigest()
        else:
            projected_sha256 = ""
        if projected_sha256 != symjit.get("release_cargo_lock_sha256"):
            issues.append(
                GateIssue(
                    "release-cargo-lock-projection",
                    "canonical Cargo.lock does not project to the authenticated "
                    "release-local SymJIT resolution",
                )
            )
    symbolica = lock.get("symbolica", {})
    name = str(symbolica.get("rust_crate", "symbolica"))
    version = str(symbolica.get("rust_version", ""))
    matches = [
        package
        for package in packages
        if package.get("name") == name and package.get("version") == version
    ]
    if len(matches) != 1 or matches[0].get("source") != _REGISTRY_SOURCE:
        issues.append(
            GateIssue(
                "release-cargo-pin",
                f"Cargo.lock must resolve published {name}=={version} exactly",
            )
        )
    return [*issues, *_cargo_manifest_pin_issues(lock)]


def _toolchain_issues(lock: dict[str, Any]) -> list[GateIssue]:
    toolchain = lock.get("toolchain")
    if not isinstance(toolchain, dict):
        return [GateIssue("toolchain-contract", "release lock has no toolchain table")]
    issues: list[GateIssue] = []
    try:
        rust = _load_toml(RUST_TOOLCHAIN_PATH)["toolchain"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("toolchain-contract", f"invalid rust-toolchain.toml: {error}")
        ]
    if not isinstance(rust, dict) or rust.get("channel") != toolchain.get(
        "rust_toolchain"
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "rust-toolchain.toml disagrees with release-lock.toml",
            )
        )
    digest = toolchain.get("manylinux_image_digest")
    image = toolchain.get("manylinux_image")
    if (
        not isinstance(image, str)
        or not image
        or not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "manylinux image and digest must be pinned exactly",
            )
        )
    build_system = _load_toml(PYPROJECT_PATH).get("build-system", {})
    requirements = (
        build_system.get("requires", []) if isinstance(build_system, dict) else []
    )
    expected = {
        "maturin": str(toolchain.get("maturin", "")),
        "packaging": str(toolchain.get("packaging", "")),
    }
    observed: dict[str, str] = {}
    for raw in requirements if isinstance(requirements, list) else []:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            observed[canonicalize_name(requirement.name)] = specifiers[0].version
    if observed != expected:
        issues.append(
            GateIssue(
                "toolchain-contract",
                f"build-system pins disagree with release lock: {observed}",
            )
        )
    return issues


def _candidate_revisions(lock: dict[str, Any]) -> dict[str, str]:
    return {
        "gammaloop": str(lock["gammaloop_candidate"]["revision"]),
        "symbolica": str(lock["symbolica"]["candidate_revision"]),
        "symbolica-community": str(lock["symbolica"]["community_revision"]),
        "symjit": str(lock["symjit"]["candidate_revision"]),
    }


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_tree_sha256(root: Path) -> str:
    """Match the installer's deterministic candidate source-tree fingerprint."""

    digest = hashlib.sha256()
    for raw_directory, raw_directories, raw_files in os.walk(root, topdown=True):
        directory = Path(raw_directory)
        directories = sorted(
            name for name in raw_directories if name not in _SOURCE_TREE_EXCLUDES
        )
        raw_directories[:] = [
            name for name in directories if not (directory / name).is_symlink()
        ]
        entries = [
            *(
                directory / name
                for name in directories
                if (directory / name).is_symlink()
            ),
            *(
                directory / name
                for name in sorted(raw_files)
                if name not in _SOURCE_TREE_EXCLUDES
                and not name.endswith((".pyc", ".pyo"))
            ),
        ]
        for path in entries:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            mode = path.lstat().st_mode & 0o111
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(mode.to_bytes(2, "big"))
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"L")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            elif path.is_file():
                digest.update(b"F")
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            else:
                digest.update(b"O")
    return digest.hexdigest()


def _candidate_patch_contract(
    contributor: dict[str, Any],
    *,
    lock_path: Path | None = None,
    issue_code: str = "candidate-patch-contract",
    label: str = "contributor",
) -> tuple[list[dict[str, str]], list[GateIssue]]:
    lock_path = CONTRIBUTOR_LOCK_PATH if lock_path is None else lock_path
    raw_patches = contributor.get("patches")
    if not isinstance(raw_patches, list):
        return [], [
            GateIssue(
                issue_code,
                f"{label} SymJIT patches must be an ordered list",
            )
        ]
    allowed_keys = {
        "name",
        "target",
        "path",
        "sha256",
        "applies_to_revision",
    }
    symjit = contributor.get("symjit")
    revision = symjit.get("candidate_revision") if isinstance(symjit, dict) else None
    dependency_root = lock_path.parent.resolve()
    state: list[dict[str, str]] = []
    issues: list[GateIssue] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, entry in enumerate(raw_patches):
        if not isinstance(entry, dict) or set(entry) != allowed_keys:
            issues.append(
                GateIssue(
                    issue_code,
                    f"{label} patch {index} has an invalid field set",
                )
            )
            continue
        if not all(
            isinstance(entry[key], str) and entry[key] for key in allowed_keys
        ):
            issues.append(
                GateIssue(
                    issue_code,
                    f"{label} patch {index} fields must be nonempty strings",
                )
            )
            continue
        normalized = {key: str(entry[key]) for key in allowed_keys}
        name = normalized["name"]
        target = normalized["target"]
        relative = normalized["path"]
        digest = normalized["sha256"]
        applies_to = normalized["applies_to_revision"]
        pure = PurePosixPath(relative)
        invalid_path = (
            pure.is_absolute()
            or not pure.parts
            or pure.parts[0] != "patches"
            or pure.suffix != ".patch"
            or any(part in {"", ".", ".."} for part in pure.parts)
        )
        if (
            name in seen_names
            or relative in seen_paths
            or target != "symjit"
            or applies_to != revision
            or _SHA256.fullmatch(digest) is None
            or invalid_path
        ):
            issues.append(
                GateIssue(
                    issue_code,
                    f"{label} patch {name!r} has an invalid identity contract",
                )
            )
            continue
        path = dependency_root.joinpath(*pure.parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(dependency_root)
        except (OSError, ValueError):
            issues.append(
                GateIssue(
                    issue_code,
                    f"{label} patch {name!r} is missing or escapes dependencies",
                )
            )
            continue
        current = dependency_root
        symlinked = False
        for part in pure.parts:
            current /= part
            symlinked = symlinked or current.is_symlink()
        if (
            symlinked
            or not resolved.is_file()
            or hashlib.sha256(resolved.read_bytes()).hexdigest() != digest
        ):
            issues.append(
                GateIssue(
                    issue_code,
                    f"{label} patch {name!r} is not an authenticated regular file",
                )
            )
            continue
        seen_names.add(name)
        seen_paths.add(relative)
        state.append(
            {
                "name": name,
                "target": target,
                "path": pure.as_posix(),
                "sha256": digest,
                "applies_to_revision": applies_to,
            }
        )
    return state, issues


def _release_symjit_source_issues(lock: dict[str, Any]) -> list[GateIssue]:
    """Validate the source, generic patch, and local-lock release contract."""

    symjit = lock.get("symjit")
    if not isinstance(symjit, dict):
        return [
            GateIssue(
                "release-symjit-source",
                "release-lock.toml has no SymJIT source contract",
            )
        ]
    issues: list[GateIssue] = []
    revision = symjit.get("revision")
    digests: dict[str, str] = {}
    for key in (
        "archive_sha256",
        "source_tree_sha256",
        "configured_tree_sha256",
        "release_cargo_lock_sha256",
    ):
        digest = symjit.get(key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            issues.append(
                GateIssue(
                    "release-symjit-source",
                    f"release SymJIT {key} must be a SHA-256 digest",
                )
            )
        else:
            digests[key] = digest
    if (
        not isinstance(revision, str)
        or _GIT_REVISION.fullmatch(revision) is None
        or not isinstance(symjit.get("source_url"), str)
        or not str(symjit["source_url"]).endswith(f"/{revision}.tar.gz")
        or not isinstance(symjit.get("archive_prefix"), str)
        or revision not in str(symjit["archive_prefix"])
    ):
        issues.append(
            GateIssue(
                "release-symjit-source",
                "release SymJIT archive must identify the immutable Git revision",
            )
        )
    pseudo_contributor = {
        "patches": symjit.get("patches"),
        "symjit": {"candidate_revision": revision},
    }
    patch_state, patch_issues = _candidate_patch_contract(
        pseudo_contributor,
        lock_path=LOCK_PATH,
        issue_code="release-symjit-patch",
        label="release SymJIT",
    )
    issues.extend(patch_issues)
    if not patch_state:
        issues.append(
            GateIssue(
                "release-symjit-patch",
                "release SymJIT requires its authenticated generic patch closure",
            )
        )
    if (
        patch_state
        and digests.get("source_tree_sha256")
        == digests.get("configured_tree_sha256")
    ):
        issues.append(
            GateIssue(
                "release-symjit-source",
                "patched SymJIT source and configured tree identities must differ",
            )
        )

    if CONTRIBUTOR_LOCK_PATH.is_file():
        try:
            contributor = _load_contributor_lock()
            contributor_symjit = contributor["symjit"]
            shared = {
                "version": (
                    symjit.get("version"),
                    contributor_symjit.get("candidate_version"),
                ),
                "repository": (
                    symjit.get("repository"),
                    contributor_symjit.get("repository"),
                ),
                "revision": (
                    revision,
                    contributor_symjit.get("candidate_revision"),
                ),
                "source_url": (
                    symjit.get("source_url"),
                    contributor_symjit.get("source_url"),
                ),
                "archive_prefix": (
                    symjit.get("archive_prefix"),
                    contributor_symjit.get("archive_prefix"),
                ),
                "archive_sha256": (
                    symjit.get("archive_sha256"),
                    contributor_symjit.get("archive_sha256"),
                ),
                "source_tree_sha256": (
                    symjit.get("source_tree_sha256"),
                    contributor_symjit.get("source_tree_sha256"),
                ),
                "configured_tree_sha256": (
                    symjit.get("configured_tree_sha256"),
                    contributor_symjit.get("candidate_tree_sha256"),
                ),
                "patches": (
                    symjit.get("patches"),
                    contributor.get("patches"),
                ),
            }
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
            issues.append(
                GateIssue(
                    "release-contributor-symjit",
                    f"could not compare release and contributor SymJIT locks: {error}",
                )
            )
        else:
            drift = sorted(name for name, pair in shared.items() if pair[0] != pair[1])
            if drift:
                issues.append(
                    GateIssue(
                        "release-contributor-symjit",
                        "release and contributor builds do not share the same "
                        "authenticated SymJIT source: " + ", ".join(drift),
                    )
                )
    return issues


def _patch_closure_sha256(patches: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        patches,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_contributor_contract_issues(
    contributor: dict[str, Any],
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    patch_state, patch_issues = _candidate_patch_contract(contributor)
    issues.extend(patch_issues)
    if contributor.get("abis") != _CANDIDATE_ABIS:
        issues.append(
            GateIssue(
                "candidate-abi-contract",
                "contributor-lock.toml has unexpected candidate ABI identities",
            )
        )
    symjit = contributor.get("symjit")
    if not isinstance(symjit, dict):
        return [
            *issues,
            GateIssue(
                "candidate-source-tree",
                "contributor-lock.toml has no SymJIT source contract",
            ),
        ]
    tree_digests: dict[str, str] = {}
    for key in ("source_tree_sha256", "candidate_tree_sha256"):
        digest = symjit.get(key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            issues.append(
                GateIssue(
                    "candidate-source-tree",
                    f"contributor lock has no valid SymJIT {key}",
                )
            )
        else:
            tree_digests[key] = digest
    if (
        len(tree_digests) == 2
        and not patch_state
        and tree_digests["source_tree_sha256"]
        != tree_digests["candidate_tree_sha256"]
    ):
        issues.append(
            GateIssue(
                "candidate-source-tree",
                "patchless SymJIT source and candidate tree identities must match",
            )
        )
    return issues


def _candidate_config_issues() -> list[GateIssue]:
    try:
        config = _load_toml(CARGO_CONFIG_PATH)
        patch_tables = config["patch"]
        registry_patches = patch_tables["crates-io"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("candidate-cargo-config", f"invalid Cargo patch config: {error}")
        ]
    if (
        not isinstance(config, dict)
        or set(config) != {"patch"}
        or not isinstance(patch_tables, dict)
        or set(patch_tables) != {"crates-io"}
        or not isinstance(registry_patches, dict)
        or set(registry_patches)
        != {"graphica", "numerica", "symbolica", "symjit"}
    ):
        return [
            GateIssue(
                "candidate-cargo-config",
                "candidate Cargo patch table does not match the locked source "
                "overrides",
            )
        ]
    issues: list[GateIssue] = []
    checkout_root = CHECKOUTS_PATH.resolve()
    expected_paths = {
        "graphica": checkout_root / "symbolica" / "lib" / "graphica",
        "numerica": checkout_root / "symbolica" / "lib" / "numerica",
        "symbolica": checkout_root / "symbolica",
        "symjit": checkout_root / "symjit",
    }
    patches = registry_patches
    for name, entry in patches.items():
        path_value = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path"}
            or not isinstance(path_value, str)
        ):
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} must contain exactly one "
                    "local path",
                )
            )
            continue
        path = Path(path_value).resolve()
        try:
            path.relative_to(checkout_root)
        except ValueError:
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} escapes dependencies/checkouts",
                )
            )
            continue
        if path != expected_paths[name]:
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} does not resolve to its "
                    "locked checkout",
                )
            )
            continue
        if not path.exists():
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} path is missing: {path}",
                )
            )
    return issues


def _candidate_issues(_release_lock: dict[str, Any]) -> list[GateIssue]:
    required = (
        CONTRIBUTOR_LOCK_PATH,
        STATE_PATH,
        CANDIDATE_LOCK_PATH,
        CARGO_CONFIG_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        return [
            GateIssue(
                "candidate-input-missing",
                "run 'just dev-install' before a candidate build; missing: "
                + ", ".join(str(path) for path in missing),
            )
        ]
    try:
        contributor = _load_contributor_lock()
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        packages = _cargo_packages(CANDIDATE_LOCK_PATH)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        return [GateIssue("candidate-input-invalid", str(error))]
    issues = _registry_source_issues(
        packages,
        local_crates=_CANDIDATE_LOCAL_CRATES,
        prefix="candidate",
    )
    issues.extend(_candidate_contributor_contract_issues(contributor))
    patch_state, _ = _candidate_patch_contract(contributor)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        issues.append(
            GateIssue("candidate-state-invalid", "installer state must use schema 1")
        )
        return issues
    if state.get("publishable") is not False:
        issues.append(
            GateIssue(
                "candidate-state-publishable",
                "candidate installer state must explicitly be non-publishable",
            )
        )
    contributor_digest = hashlib.sha256(CONTRIBUTOR_LOCK_PATH.read_bytes()).hexdigest()
    if state.get("contributor_lock_sha256") != contributor_digest:
        issues.append(
            GateIssue(
                "candidate-state-lock",
                "installer state is not bound to the current contributor lock",
            )
        )
    if state.get("patches") != patch_state:
        issues.append(
            GateIssue(
                "candidate-state-patches",
                "installer state does not match the authenticated patch contract",
            )
        )
    sources = state.get("sources")
    revisions = _candidate_revisions(contributor)
    if not isinstance(sources, dict):
        issues.append(
            GateIssue("candidate-state-invalid", "installer state has no source map")
        )
    else:
        for name, revision in revisions.items():
            entry = sources.get(name)
            checkout = CHECKOUTS_PATH / name
            if not isinstance(entry, dict) or entry.get("revision") != revision:
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate source {name} does not match contributor-lock.toml",
                    )
                )
            if name == "symjit":
                symjit = contributor["symjit"]
                expected_tree = symjit.get("candidate_tree_sha256")
                archive_matches = (
                    isinstance(entry, dict)
                    and entry.get("version") == symjit.get("candidate_version")
                    and entry.get("archive_sha256") == symjit.get("archive_sha256")
                )
                if not checkout.is_dir() or not archive_matches:
                    issues.append(
                        GateIssue(
                            "candidate-source-revision",
                            "candidate SymJIT archive does not match "
                            "contributor-lock.toml",
                        )
                    )
                if (
                    not isinstance(expected_tree, str)
                    or _SHA256.fullmatch(expected_tree) is None
                ):
                    issues.append(
                        GateIssue(
                            "candidate-source-tree",
                            "contributor lock has no valid SymJIT candidate tree "
                            "SHA-256",
                        )
                    )
                elif isinstance(entry, dict):
                    if entry.get("worktree_sha256") != expected_tree:
                        issues.append(
                            GateIssue(
                                "candidate-source-tree",
                                "installer state has the wrong SymJIT source-tree "
                                "digest",
                            )
                        )
                    if entry.get("patch_sha256") != _patch_closure_sha256(
                        patch_state
                    ):
                        issues.append(
                            GateIssue(
                                "candidate-source-patch",
                                "installer SymJIT source entry does not match the "
                                "authenticated patch closure",
                            )
                        )
                    if checkout.is_dir():
                        try:
                            actual_tree = _source_tree_sha256(checkout)
                        except OSError as error:
                            issues.append(
                                GateIssue(
                                    "candidate-source-tree",
                                    f"could not fingerprint managed SymJIT: {error}",
                                )
                            )
                        else:
                            if actual_tree != expected_tree:
                                issues.append(
                                    GateIssue(
                                        "candidate-source-tree",
                                        "managed SymJIT source tree does not match "
                                        "contributor-lock.toml",
                                    )
                                )
            elif not checkout.is_dir() or _git_head(checkout) != revision:
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate checkout {name} is not at {revision}",
                    )
                )
    return [*issues, *_candidate_config_issues()]


def check(*, candidate: bool) -> list[GateIssue]:
    try:
        lock = _load_lock()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-lock-invalid", str(error))]
    issues = [
        *_release_contract_issues(lock),
        *_release_cargo_lock_issues(lock),
        *_toolchain_issues(lock),
    ]
    if candidate:
        issues.extend(_candidate_issues(lock))
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="validate source-checkout candidate inputs as non-publishable",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = check(candidate=args.candidate)
    payload = {
        "mode": "candidate" if args.candidate else "release",
        "ready": not issues,
        "issues": [{"code": issue.code, "message": issue.message} for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.message}", file=sys.stderr)
    else:
        print(f"{payload['mode']} dependency gate passed")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
