#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate the immutable dependency contract used by release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.tags import compatible_tags, cpython_tags, mac_platforms
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the bootstrap fallback
    from pip._vendor.packaging.markers import (  # type: ignore[no-redef]
        default_environment,
    )
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.tags import (  # type: ignore[no-redef]
        compatible_tags,
        cpython_tags,
        mac_platforms,
    )
    from pip._vendor.packaging.utils import (  # type: ignore[no-redef]
        InvalidWheelFilename,
        parse_wheel_filename,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

from python_lock import (  # noqa: E402
    PythonRuntimeLock,
    canonicalize_name,
    load_python_runtime_lock,
)

LOCK_PATH = ROOT / "dependencies" / "release-lock.toml"
PYTHON_LOCK_PATH = ROOT / "dependencies" / "python-runtime-lock.toml"
CARGO_LOCK_PATH = ROOT / "Cargo.lock"
CARGO_MANIFEST_PATH = ROOT / "Cargo.toml"
RUST_TOOLCHAIN_PATH = ROOT / "rust-toolchain.toml"
RUST_WORKSPACE_PATH = ROOT / "rust"
STATE_PATH = ROOT / "dependencies" / "install-state.json"
CANDIDATE_LOCK_PATH = ROOT / "dependencies" / "candidate-Cargo.lock"
CARGO_CONFIG_PATH = ROOT / "dependencies" / "candidate-cargo-config.toml"
CHECKOUTS_PATH = ROOT / "dependencies" / "checkouts"
WORKFLOW_PATHS = (
    ROOT / ".github" / "workflows" / "candidate.yml",
    ROOT / ".github" / "workflows" / "release-artifacts.yml",
)

_WORKSPACE_CRATES = frozenset(
    {
        "rusticol-capi",
        "rusticol-core",
        "rusticol-python",
    }
)
_CANDIDATE_PATH_CRATES = frozenset(
    {
        "graphica",
        "numerica",
        "symbolica",
        "symjit",
    }
)
_CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CARGO_ENVIRONMENT_NAMES = {
    "CARGO",
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTFLAGS",
}
_CARGO_ENVIRONMENT_PREFIXES = ("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_")

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


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


def _toolchain_issues(lock: dict[str, Any]) -> list[GateIssue]:
    toolchain = lock.get("toolchain")
    if not isinstance(toolchain, dict):
        return [GateIssue("toolchain-lock", "release lock has no toolchain table")]
    expected_rust = str(toolchain.get("rust_toolchain", ""))
    rust_action_sha = str(toolchain.get("rust_toolchain_action_sha", ""))
    image = str(toolchain.get("manylinux_image", ""))
    digest = str(toolchain.get("manylinux_image_digest", ""))
    issues: list[GateIssue] = []
    expected_build_requirements = {
        "maturin": str(toolchain.get("maturin", "")),
        "packaging": str(toolchain.get("packaging", "")),
    }
    if any(
        re.fullmatch(r"\d+(?:\.\d+)+", version) is None
        for version in expected_build_requirements.values()
    ):
        issues.append(
            GateIssue(
                "build-system-pin",
                "toolchain Maturin and packaging versions must be exact",
            )
        )
    try:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            raw_build_requirements = tomllib.load(stream)["build-system"]["requires"]
        observed_build_requirements: dict[str, str] = {}
        for raw in raw_build_requirements:
            requirement = Requirement(str(raw))
            specifiers = list(requirement.specifier)
            if (
                requirement.marker is not None
                or requirement.extras
                or requirement.url
                or len(specifiers) != 1
                or specifiers[0].operator != "=="
            ):
                raise ValueError(f"build requirement is not an exact pin: {raw}")
            observed_build_requirements[canonicalize_name(requirement.name)] = (
                specifiers[0].version
            )
    except (
        InvalidRequirement,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as error:
        issues.append(
            GateIssue(
                "build-system-pin",
                f"cannot validate build-system requirements: {error}",
            )
        )
    else:
        if observed_build_requirements != expected_build_requirements:
            issues.append(
                GateIssue(
                    "build-system-pin",
                    "pyproject build-system requirements disagree with the "
                    "release lock",
                )
            )
    if re.fullmatch(r"\d+\.\d+\.\d+", expected_rust) is None:
        issues.append(
            GateIssue(
                "rust-toolchain-pin",
                "toolchain.rust_toolchain must be an exact stable Rust version",
            )
        )
    if re.fullmatch(r"[0-9a-f]{40}", rust_action_sha) is None:
        issues.append(
            GateIssue(
                "rust-toolchain-pin",
                "toolchain.rust_toolchain_action_sha must be a full commit SHA",
            )
        )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None or not image:
        issues.append(
            GateIssue(
                "manylinux-image-pin",
                "toolchain manylinux image and SHA-256 digest must be immutable",
            )
        )
    try:
        with RUST_TOOLCHAIN_PATH.open("rb") as stream:
            rust_toolchain = tomllib.load(stream)["toolchain"]
        with CARGO_MANIFEST_PATH.open("rb") as stream:
            workspace = tomllib.load(stream)["workspace"]["package"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        issues.append(GateIssue("rust-toolchain-pin", f"invalid Rust pin: {exc}"))
    else:
        observed = {
            "rust-toolchain.toml": rust_toolchain.get("channel"),
            "Cargo.toml rust-version": workspace.get("rust-version"),
            "release-lock rust_minimum": toolchain.get("rust_minimum"),
        }
        for source, version in observed.items():
            if version != expected_rust:
                issues.append(
                    GateIssue(
                        "rust-toolchain-pin",
                        f"{source} must equal pinned Rust {expected_rust!r}, "
                        f"found {version!r}",
                    )
                )
    expected_image = f"{image}@{digest}"
    try:
        workflows = "\n".join(
            path.read_text(encoding="utf-8") for path in WORKFLOW_PATHS
        )
    except OSError as exc:
        issues.append(
            GateIssue("toolchain-workflow", f"cannot read release workflows: {exc}")
        )
    else:
        if (
            "rust-toolchain@stable" in workflows
            or "default-toolchain stable" in workflows
        ):
            issues.append(
                GateIssue(
                    "rust-toolchain-pin",
                    "release workflows contain a mutable stable Rust toolchain",
                )
            )
        if (
            workflows.count(f"rust-toolchain@{rust_action_sha}") != 6
            or workflows.count(f"default-toolchain {expected_rust}") != 2
        ):
            issues.append(
                GateIssue(
                    "rust-toolchain-pin",
                    "release workflows do not use the exact Rust pin everywhere",
                )
            )
        if ":latest" in workflows or workflows.count(expected_image) != 2:
            issues.append(
                GateIssue(
                    "manylinux-image-pin",
                    "candidate/release workflows do not use the exact manylinux digest",
                )
            )
    return issues


def _cargo_lock_packages(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    packages = payload.get("package")
    if payload.get("version") != 4 or not isinstance(packages, list):
        raise ValueError(f"{path} must use Cargo lock format version 4")
    return packages


def _release_cargo_lock_issues(lock: dict[str, Any]) -> list[GateIssue]:
    try:
        packages = _cargo_lock_packages(CARGO_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-lock-invalid", str(error))]

    issues: list[GateIssue] = []
    for package in packages:
        name = str(package.get("name", "<unnamed>"))
        version = str(package.get("version", "<unknown>"))
        source = package.get("source")
        checksum = package.get("checksum")
        if name in _WORKSPACE_CRATES and source is None and checksum is None:
            continue
        if source != _CRATES_IO_SOURCE:
            issues.append(
                GateIssue(
                    "release-cargo-nonregistry",
                    f"Cargo.lock package {name} {version} has non-registry "
                    f"source {source!r}",
                )
            )
            continue
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            issues.append(
                GateIssue(
                    "release-cargo-checksum",
                    f"Cargo.lock package {name} {version} has no valid checksum",
                )
            )

    symbolica = lock["symbolica"]
    expected = {
        "symbolica": (
            str(symbolica["rust_version"]),
            str(symbolica["rust_checksum"]),
        ),
        "symjit": (
            str(symbolica["published_symjit_version"]),
            str(symbolica["published_symjit_checksum"]),
        ),
    }
    for name, (expected_version, expected_checksum) in expected.items():
        matches = [package for package in packages if package.get("name") == name]
        if len(matches) != 1:
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    f"Cargo.lock must contain exactly one published {name} package",
                )
            )
            continue
        package = matches[0]
        if (
            package.get("version") != expected_version
            or package.get("checksum") != expected_checksum
            or package.get("source") != _CRATES_IO_SOURCE
        ):
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    f"Cargo.lock {name} entry does not match release-lock.toml "
                    f"({expected_version}, {expected_checksum})",
                )
            )
    return issues


def _candidate_cargo_lock_issues() -> list[GateIssue]:
    try:
        packages = _cargo_lock_packages(CANDIDATE_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("candidate-cargo-lock-invalid", str(error))]

    local_names = {
        str(package.get("name"))
        for package in packages
        if package.get("source") is None and package.get("checksum") is None
    }
    expected_local = _WORKSPACE_CRATES | _CANDIDATE_PATH_CRATES
    issues: list[GateIssue] = []
    if local_names != expected_local:
        issues.append(
            GateIssue(
                "candidate-cargo-sources",
                "candidate-Cargo.lock local packages differ from the managed set: "
                f"expected {sorted(expected_local)}, found {sorted(local_names)}",
            )
        )
    for package in packages:
        name = str(package.get("name", "<unnamed>"))
        if name in expected_local:
            continue
        checksum = package.get("checksum")
        if (
            package.get("source") != _CRATES_IO_SOURCE
            or not isinstance(checksum, str)
            or _SHA256_PATTERN.fullmatch(checksum) is None
        ):
            issues.append(
                GateIssue(
                    "candidate-cargo-sources",
                    f"candidate-Cargo.lock package {name} is neither a managed "
                    "path package nor a checksummed crates.io package",
                )
            )
    return issues


def _cargo_metadata_issue(
    *,
    mode: str,
    lock_path: Path,
    config_path: Path | None,
) -> GateIssue | None:
    """Validate a lock in a sibling-free overlay with no checkout config leakage."""

    cargo = shutil.which("cargo")
    if cargo is None:
        return GateIssue(
            f"{mode}-cargo-missing",
            "cargo is required for lock validation",
        )
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"pyamplicol-{mode}-cargo-metadata-"
        ) as raw:
            overlay = Path(raw)
            shutil.copy2(CARGO_MANIFEST_PATH, overlay / "Cargo.toml")
            shutil.copy2(lock_path, overlay / "Cargo.lock")
            shutil.copytree(RUST_WORKSPACE_PATH, overlay / "rust")
            if mode == "candidate":
                _rewrite_candidate_symjit_requirement(overlay)
            if config_path is not None:
                target = overlay / ".cargo" / "config.toml"
                target.parent.mkdir(parents=True)
                shutil.copy2(config_path, target)
            elif (overlay / ".cargo" / "config.toml").exists():
                return GateIssue(
                    "release-cargo-config-leak",
                    "release Cargo metadata overlay unexpectedly contains a config",
                )
            environment = {
                name: value
                for name, value in os.environ.items()
                if name not in _CARGO_ENVIRONMENT_NAMES
                and not name.startswith(_CARGO_ENVIRONMENT_PREFIXES)
            }
            environment["CARGO_TARGET_DIR"] = str(overlay / "target")
            completed = subprocess.run(
                [cargo, "metadata", "--locked", "--format-version", "1"],
                cwd=overlay,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
    except OSError as error:
        return GateIssue(
            f"{mode}-cargo-metadata",
            f"could not construct clean Cargo metadata overlay: {error}",
        )
    if completed.returncode == 0:
        return None
    diagnostic = (completed.stderr or completed.stdout).strip().splitlines()
    detail = " | ".join(diagnostic[-4:]) if diagnostic else "unknown Cargo error"
    return GateIssue(
        f"{mode}-cargo-metadata",
        f"cargo metadata --locked failed in the clean {mode} overlay: {detail}",
    )


def _rewrite_candidate_symjit_requirement(root: Path) -> None:
    """Project the release manifest onto the managed candidate SymJIT."""

    with LOCK_PATH.open("rb") as stream:
        lock = tomllib.load(stream)
    published = str(lock["symbolica"]["published_symjit_version"])
    candidate = str(lock["symjit"]["candidate_version"])
    manifest = root / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    pattern = rf'(?m)^(symjit\s*=\s*\{{\s*version\s*=\s*)"={re.escape(published)}"'
    updated, count = re.subn(pattern, rf'\g<1>"={candidate}"', text, count=1)
    if count != 1:
        raise RuntimeError(
            "could not project rusticol-core from the published SymJIT "
            f"requirement {published} to candidate {candidate}"
        )
    manifest.write_text(updated, encoding="utf-8")


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("rb") as stream:
        payload = tomllib.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError("dependencies/release-lock.toml must use schema_version = 1")
    return payload


def _candidate_python_names(lock: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        {
            canonicalize_name(str(lock["symbolica"]["python_distribution"])),
            canonicalize_name(str(lock["ufo_model_loader"]["python_distribution"])),
        }
    )


def _python_runtime_lock(
    release_lock: dict[str, Any],
) -> tuple[PythonRuntimeLock | None, list[GateIssue]]:
    contract = release_lock.get("python_runtime_lock")
    if not isinstance(contract, dict):
        return None, [
            GateIssue(
                "python-lock-contract",
                "release lock has no python_runtime_lock table",
            )
        ]
    relative = contract.get("path")
    digest = contract.get("sha256")
    if relative != "dependencies/python-runtime-lock.toml":
        return None, [
            GateIssue(
                "python-lock-contract",
                "python runtime lock path must be "
                "dependencies/python-runtime-lock.toml",
            )
        ]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        return None, [
            GateIssue(
                "python-lock-contract",
                "python runtime lock digest must be a lowercase SHA-256",
            )
        ]
    if not PYTHON_LOCK_PATH.is_file():
        return None, [
            GateIssue(
                "python-lock-missing",
                f"Python runtime lock is missing: {PYTHON_LOCK_PATH}",
            )
        ]
    observed = _sha256_path(PYTHON_LOCK_PATH)
    if observed != digest:
        return None, [
            GateIssue(
                "python-lock-digest",
                f"Python runtime lock has SHA-256 {observed}, expected {digest}",
            )
        ]
    try:
        runtime_lock = load_python_runtime_lock(PYTHON_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return None, [GateIssue("python-lock-invalid", str(error))]
    return runtime_lock, []


def _direct_python_contract_issues(
    release_lock: dict[str, Any], runtime_lock: PythonRuntimeLock
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    release_entries = release_lock.get("python_dependencies")
    if not isinstance(release_entries, list):
        return [
            GateIssue(
                "python-direct-contract",
                "release lock has no direct Python dependency list",
            )
        ]
    expected: dict[str, tuple[str, str]] = {}
    for entry in release_entries:
        if not isinstance(entry, dict):
            issues.append(
                GateIssue(
                    "python-direct-contract",
                    "direct Python dependency entries must be tables",
                )
            )
            continue
        name = canonicalize_name(str(entry.get("distribution", "")))
        value = (str(entry.get("version", "")), str(entry.get("license", "")))
        if not name or name in expected:
            issues.append(
                GateIssue(
                    "python-direct-contract",
                    f"direct Python dependency is empty or repeated: {name!r}",
                )
            )
            continue
        expected[name] = value
    observed = {
        package.name: (package.version, package.license)
        for package in runtime_lock.direct_packages
    }
    if observed != expected:
        issues.append(
            GateIssue(
                "python-direct-contract",
                "release-lock direct Python dependencies disagree with the full "
                f"runtime lock: expected={expected}, observed={observed}",
            )
        )

    try:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            raw_project = tomllib.load(stream)["project"]
        requirements = raw_project["dependencies"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        issues.append(
            GateIssue(
                "python-direct-contract", f"cannot load pyproject dependencies: {error}"
            )
        )
        return issues
    if not isinstance(requirements, list):
        issues.append(
            GateIssue(
                "python-direct-contract",
                "pyproject project.dependencies must be a list",
            )
        )
        return issues
    pyproject: dict[str, str] = {}
    for raw in requirements:
        try:
            requirement = Requirement(str(raw))
        except InvalidRequirement as error:
            issues.append(
                GateIssue(
                    "python-direct-contract",
                    f"invalid pyproject runtime requirement {raw!r}: {error}",
                )
            )
            continue
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if (
            requirement.marker is not None
            or requirement.extras
            or requirement.url
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            issues.append(
                GateIssue(
                    "python-direct-contract",
                    f"pyproject dependency must be one unconditional exact pin: {raw}",
                )
            )
            continue
        pyproject[name] = specifiers[0].version
    expected_versions = {name: value[0] for name, value in expected.items()}
    if pyproject != expected_versions:
        issues.append(
            GateIssue(
                "python-direct-contract",
                "pyproject runtime requirements disagree with release-lock direct "
                f"dependencies: expected={expected_versions}, observed={pyproject}",
            )
        )
    return issues


def _target_platforms(platform_tag: str) -> tuple[str, ...]:
    mac = re.fullmatch(r"macosx_(\d+)_(\d+)_(arm64|x86_64)", platform_tag)
    if mac is not None:
        version = (int(mac.group(1)), int(mac.group(2)))
        return tuple(mac_platforms(version=version, arch=mac.group(3)))
    if platform_tag == "manylinux_2_28_x86_64":
        return tuple(
            [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)]
            + ["manylinux2014_x86_64"]
        )
    raise ValueError(f"unsupported Python artifact target: {platform_tag}")


def _supported_tags(python: str, platform_tag: str) -> frozenset[object]:
    major, minor = (int(value) for value in python.split(".", 1))
    platforms = _target_platforms(platform_tag)
    interpreter = f"cp{major}{minor}"
    return frozenset(
        [
            *cpython_tags((major, minor), platforms=platforms),
            *compatible_tags(
                (major, minor), interpreter=interpreter, platforms=platforms
            ),
        ]
    )


def _runtime_marker_environments(
    supported_python: tuple[str, ...], platforms: list[str]
) -> tuple[dict[str, str], ...]:
    environments: list[dict[str, str]] = []
    for python in supported_python:
        for platform in platforms:
            environment = default_environment()
            environment.update(
                {
                    "extra": "",
                    "python_version": python,
                    "python_full_version": f"{python}.0",
                    "implementation_version": f"{python}.0",
                    "sys_platform": "darwin"
                    if platform.startswith("macosx")
                    else "linux",
                    "os_name": "posix",
                    "platform_system": (
                        "Darwin" if platform.startswith("macosx") else "Linux"
                    ),
                    "platform_machine": (
                        "arm64" if platform.endswith("arm64") else "x86_64"
                    ),
                }
            )
            environments.append(environment)
    return tuple(environments)


def _pypi_graph_issues(
    package_name: str,
    package_version: str,
    declared_dependencies: tuple[str, ...],
    payload: dict[str, Any],
    runtime_lock: PythonRuntimeLock,
    environments: tuple[dict[str, str], ...],
) -> list[GateIssue]:
    info = payload.get("info")
    if not isinstance(info, dict):
        return [
            GateIssue(
                "python-package-metadata",
                f"PyPI returned no metadata object for {package_name}",
            )
        ]
    raw_requirements = info.get("requires_dist") or []
    if not isinstance(raw_requirements, list):
        return [
            GateIssue(
                "python-package-metadata",
                f"PyPI returned an invalid dependency list for {package_name}",
            )
        ]
    observed: set[str] = set()
    issues: list[GateIssue] = []
    locked = runtime_lock.by_name
    for raw in raw_requirements:
        try:
            requirement = Requirement(str(raw))
        except InvalidRequirement as error:
            issues.append(
                GateIssue(
                    "python-package-metadata",
                    f"invalid PyPI requirement for {package_name}: {raw!r}: {error}",
                )
            )
            continue
        if requirement.marker is not None and not any(
            requirement.marker.evaluate(environment) for environment in environments
        ):
            continue
        dependency = canonicalize_name(requirement.name)
        observed.add(dependency)
        locked_dependency = locked.get(dependency)
        if locked_dependency is None:
            issues.append(
                GateIssue(
                    "python-closure-incomplete",
                    f"{package_name}=={package_version} requires unlocked package "
                    f"{dependency}",
                )
            )
            continue
        try:
            locked_version = Version(locked_dependency.version)
        except InvalidVersion as error:
            issues.append(
                GateIssue(
                    "python-lock-invalid",
                    f"invalid locked version for {dependency}: {error}",
                )
            )
            continue
        if requirement.specifier and locked_version not in requirement.specifier:
            issues.append(
                GateIssue(
                    "python-closure-version",
                    f"locked {dependency}=={locked_version} does not satisfy "
                    f"{package_name}'s requirement {requirement}",
                )
            )
    if observed != set(declared_dependencies):
        issues.append(
            GateIssue(
                "python-closure-graph",
                f"locked dependency edges for {package_name} disagree with PyPI; "
                f"declared={sorted(declared_dependencies)}, "
                f"observed={sorted(observed)}",
            )
        )
    return issues


def _python_artifact_issues(
    release_lock: dict[str, Any],
    runtime_lock: PythonRuntimeLock,
    *,
    candidate: bool,
    online: bool,
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    candidates = _candidate_python_names(release_lock) if candidate else frozenset()
    targets = release_lock.get("targets")
    if not isinstance(targets, list) or not targets:
        return [GateIssue("python-artifact-targets", "release lock has no targets")]
    platforms: list[str] = []
    for target in targets:
        platform = target.get("platform_tag") if isinstance(target, dict) else None
        if not isinstance(platform, str):
            issues.append(
                GateIssue(
                    "python-artifact-targets",
                    "release target has no Python platform tag",
                )
            )
        else:
            platforms.append(platform)
    marker_environments = _runtime_marker_environments(
        runtime_lock.supported_python, platforms
    )

    for package in runtime_lock.packages:
        artifact_tags: list[frozenset[object]] = []
        for artifact in package.artifacts:
            try:
                name, version, _build, tags = parse_wheel_filename(artifact.filename)
            except InvalidWheelFilename as error:
                issues.append(
                    GateIssue(
                        "python-artifact-filename",
                        f"invalid locked wheel {artifact.filename}: {error}",
                    )
                )
                continue
            if (
                canonicalize_name(str(name)) != package.name
                or str(version) != package.version
            ):
                issues.append(
                    GateIssue(
                        "python-artifact-identity",
                        f"locked wheel {artifact.filename} does not belong to "
                        f"{package.name}=={package.version}",
                    )
                )
            artifact_tags.append(frozenset(tags))
        if not artifact_tags and package.name not in candidates:
            issues.append(
                GateIssue(
                    "python-artifact-missing",
                    f"{package.name}=={package.version} has no locked wheel artifacts",
                )
            )
            continue
        if package.name not in candidates:
            for python in runtime_lock.supported_python:
                for platform in platforms:
                    try:
                        supported = _supported_tags(python, platform)
                    except ValueError as error:
                        issues.append(GateIssue("python-artifact-targets", str(error)))
                        continue
                    if not any(tags & supported for tags in artifact_tags):
                        issues.append(
                            GateIssue(
                                "python-artifact-coverage",
                                f"{package.name}=={package.version} has no admitted "
                                f"wheel for CPython {python} on {platform}",
                            )
                        )

        if not online:
            continue
        url = f"https://pypi.org/pypi/{package.distribution}/{package.version}/json"
        request = urllib.request.Request(
            url, headers={"User-Agent": "pyamplicol-release-gate/0.1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            issues.append(
                GateIssue(
                    "python-package-unpublished",
                    f"cannot verify {package.name}=={package.version} on PyPI: {error}",
                )
            )
            continue
        urls = payload.get("urls") if isinstance(payload, dict) else None
        if not isinstance(urls, list):
            issues.append(
                GateIssue(
                    "python-package-metadata",
                    f"PyPI returned no artifact list for {package.name}",
                )
            )
            continue
        published = {
            str(item.get("filename")): str(item.get("digests", {}).get("sha256", ""))
            for item in urls
            if isinstance(item, dict) and isinstance(item.get("digests"), dict)
        }
        for artifact in package.artifacts:
            if published.get(artifact.filename) != artifact.sha256:
                issues.append(
                    GateIssue(
                        "python-artifact-pypi",
                        f"PyPI artifact/hash mismatch for {artifact.filename}",
                    )
                )
        issues.extend(
            _pypi_graph_issues(
                package.name,
                package.version,
                package.dependencies,
                payload,
                runtime_lock,
                marker_environments,
            )
        )
    return issues


def _python_lock_issues(
    release_lock: dict[str, Any], *, candidate: bool, online: bool
) -> list[GateIssue]:
    runtime_lock, issues = _python_runtime_lock(release_lock)
    if runtime_lock is None:
        return issues
    return [
        *issues,
        *_direct_python_contract_issues(release_lock, runtime_lock),
        *_python_artifact_issues(
            release_lock,
            runtime_lock,
            candidate=candidate,
            online=online,
        ),
    ]


def _published(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pyamplicol-release-gate/0.1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _check_patch(entry: dict[str, Any]) -> GateIssue | None:
    relative = Path(str(entry["path"]))
    path = ROOT / "dependencies" / relative
    if not path.is_file():
        return GateIssue("missing-patch", f"missing dependency patch: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(entry["sha256"])
    if digest != expected:
        return GateIssue(
            "patch-digest",
            f"{relative} has SHA-256 {digest}, expected {expected}",
        )
    return None


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_revisions(lock: dict[str, Any]) -> dict[str, str]:
    symbolica = lock["symbolica"]
    return {
        "symbolica": str(symbolica["candidate_revision"]),
        "symbolica-community": str(symbolica["community_revision"]),
        "symjit": str(lock["symjit"]["candidate_revision"]),
        "ufo-model-loader": str(lock["ufo_model_loader"]["candidate_revision"]),
        "gammaloop": str(lock["gammaloop_candidate"]["revision"]),
    }


def _git_output(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _source_tree_sha256(root: Path) -> str:
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


def _candidate_issues(lock: dict[str, Any]) -> list[GateIssue]:
    required = {
        "installer_state": STATE_PATH,
        "candidate_lock": CANDIDATE_LOCK_PATH,
        "cargo_config": CARGO_CONFIG_PATH,
    }
    issues = [
        GateIssue(
            "candidate-input-missing",
            f"candidate {name.replace('_', ' ')} is missing: {path}",
        )
        for name, path in required.items()
        if not path.is_file()
    ]
    if issues:
        return issues
    cargo_lock_issues = _candidate_cargo_lock_issues()
    issues.extend(cargo_lock_issues)
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [GateIssue("candidate-state-invalid", f"invalid installer state: {exc}")]
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        return [
            GateIssue(
                "candidate-state-invalid",
                "installer state must be a schema-v1 object",
            )
        ]
    checks = {
        "release_lock_sha256": _sha256_path(LOCK_PATH),
        "python_runtime_lock_sha256": _sha256_path(PYTHON_LOCK_PATH),
        "candidate_lock_sha256": _sha256_path(CANDIDATE_LOCK_PATH),
        "cargo_config_sha256": _sha256_path(CARGO_CONFIG_PATH),
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            issues.append(
                GateIssue(
                    "candidate-state-digest",
                    f"installer state {key} does not match the active candidate input",
                )
            )
    if state.get("publishable") is not False:
        issues.append(
            GateIssue(
                "candidate-state-publishable",
                "candidate installer state must explicitly be non-publishable",
            )
        )
    sources = state.get("sources")
    if not isinstance(sources, dict):
        return [
            *issues,
            GateIssue("candidate-state-invalid", "installer state has no source map"),
        ]
    for name, revision in _candidate_revisions(lock).items():
        checkout = CHECKOUTS_PATH / name
        entry = sources.get(name)
        if not checkout.is_dir() or not isinstance(entry, dict):
            issues.append(
                GateIssue(
                    "candidate-source-missing",
                    f"candidate source {name} is absent from checkout/state",
                )
            )
            continue
        try:
            head = _git_output(checkout, "rev-parse", "HEAD").strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            issues.append(
                GateIssue(
                    "candidate-source-invalid",
                    f"cannot inspect candidate source {name}: {exc}",
                )
            )
            continue
        snapshot = _source_tree_sha256(checkout)
        if head != revision or entry.get("revision") != revision:
            issues.append(
                GateIssue(
                    "candidate-source-revision",
                    f"candidate source {name} is not at {revision}",
                )
            )
        if entry.get("worktree_sha256") != snapshot:
            issues.append(
                GateIssue(
                    "candidate-source-worktree",
                    f"candidate source {name} differs from the installed snapshot",
                )
            )
    expected_patches = [
        {
            "dependency": entry["dependency"],
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        for entry in lock["patches"]
    ]
    if state.get("patches") != expected_patches:
        issues.append(
            GateIssue(
                "candidate-patch-state",
                "installer patch inventory does not match release-lock.toml",
            )
        )
    if not cargo_lock_issues:
        metadata_issue = _cargo_metadata_issue(
            mode="candidate",
            lock_path=CANDIDATE_LOCK_PATH,
            config_path=CARGO_CONFIG_PATH,
        )
        if metadata_issue is not None:
            issues.append(metadata_issue)
    return issues


def check(*, candidate: bool, online: bool) -> list[GateIssue]:
    lock = _load_lock()
    issues = [
        *_toolchain_issues(lock),
        *_python_lock_issues(lock, candidate=candidate, online=online),
        *(
            issue
            for entry in lock.get("patches", [])
            if (issue := _check_patch(entry)) is not None
        ),
    ]

    cargo_lock_issues = _release_cargo_lock_issues(lock)
    issues.extend(cargo_lock_issues)
    if not cargo_lock_issues:
        metadata_issue = _cargo_metadata_issue(
            mode="release",
            lock_path=CARGO_LOCK_PATH,
            config_path=None,
        )
        if metadata_issue is not None:
            issues.append(metadata_issue)

    if candidate:
        return [*issues, *_candidate_issues(lock)]

    symbolica = lock["symbolica"]
    if symbolica["release_status"] != "verified":
        issues.append(
            GateIssue(
                "symbolica-unverified",
                "published Symbolica Python/Rust/serialization compatibility "
                "has not been marked verified",
            )
        )

    loader = lock["ufo_model_loader"]
    if loader["release_status"] != "verified":
        issues.append(
            GateIssue(
                "ufo-loader-unverified",
                f"{loader['python_distribution']} "
                f"{loader['required_version']} has not been marked verified",
            )
        )

    if online:
        py_package = str(symbolica["python_distribution"])
        py_version = str(symbolica["python_version"])
        if not _published(f"https://pypi.org/pypi/{py_package}/{py_version}/json"):
            issues.append(
                GateIssue(
                    "symbolica-pypi",
                    f"{py_package} {py_version} is unavailable on PyPI",
                )
            )

        crate = str(symbolica["rust_crate"])
        crate_version = str(symbolica["rust_version"])
        if not _published(f"https://crates.io/api/v1/crates/{crate}/{crate_version}"):
            issues.append(
                GateIssue(
                    "symbolica-crates-io",
                    f"{crate} {crate_version} is unavailable on crates.io",
                )
            )

        loader_package = str(loader["python_distribution"])
        loader_version = str(loader["required_version"])
        if not _published(
            f"https://pypi.org/pypi/{loader_package}/{loader_version}/json"
        ):
            issues.append(
                GateIssue(
                    "ufo-loader-pypi",
                    f"{loader_package} {loader_version} is unavailable on PyPI",
                )
            )

    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="validate local candidate inputs without claiming release readiness",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip package-index availability checks",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = check(candidate=args.candidate, online=not args.offline)
    payload = {
        "mode": "candidate" if args.candidate else "release",
        "ready": not issues,
        "issues": [issue.__dict__ for issue in issues],
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
