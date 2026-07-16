#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Audit and compare pyAmpliCol release artifacts."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote

try:
    from packaging.markers import Variable
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.markers import Variable  # type: ignore[no-redef]
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

from _common import (
    ROOT,
    ReleaseError,
    external_temporary_directory,
    run,
    safe_extract_sdist,
    sha256,
)
from audit_sdist import REQUIRED_SDIST_MEMBERS, missing_required_sdist_members

MAX_WHEEL_BYTES = 95_000_000
EXPECTED_DISTRIBUTION = "pyamplicol"
EXPECTED_RELEASE_VERSION = "0.1.0"
EXPECTED_PYTHON_TAG = "cp311"
EXPECTED_ABI_TAG = "abi3"
RELEASE_TARGETS = {
    "macosx_11_0_arm64": "aarch64-apple-darwin",
    "macosx_11_0_x86_64": "x86_64-apple-darwin",
    "manylinux_2_28_x86_64": "x86_64-unknown-linux-gnu",
}
_LINUX_PLATFORM_TAGS = {
    "manylinux_2_28_x86_64",
}
_REQUIRED_SDK_PATHS = {
    "pyamplicol/_sdk/include/rusticol.h",
    "pyamplicol/_sdk/include/rusticol.hpp",
    "pyamplicol/_sdk/fortran/rusticol.f90",
    "pyamplicol/_sdk/lib/librusticol_capi.a",
    "pyamplicol/_sdk/config.py",
    "pyamplicol/_sdk/metadata.json",
    "pyamplicol/_sdk/link.json",
    "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json",
}


def _pypi_purl(name: str, version: str) -> str:
    return f"pkg:pypi/{quote(name, safe='.-_~')}@{quote(version, safe='.-_~')}"


_FORBIDDEN_MEMBER_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "PYPI_DEPLOYMENT_TEST",
    "target",
}
_FORBIDDEN_PATH_MARKERS = (
    b"/tmp/",
    b"/var/tmp/",
    b"/Users/",
    b"/home/",
    b"/private/var/folders/",
    b"/opt/homebrew/",
    b"/opt/local/",
    b"/opt/python/",
    b"/github/workspace/",
    b"C:\\Users\\",
    os.fsencode(str(ROOT.resolve())),
)
_FORBIDDEN_NATIVE_RELATIVE_MARKERS = (b"../", b"..\\")
_NATIVE_MEMBER_SUFFIXES = (".a", ".dylib", ".lib", ".pyd", ".so")
_FORBIDDEN_CAPI_SYMBOLS = (
    "pyobject",
    "pyexc_",
    "pygilstate",
    "pyerr_",
    "pylong_",
    "pyunicode_",
    "pymem_",
    "pytype_",
    "pytuple_",
    "pylist_",
    "pydict_",
    "pymodule_",
    "pyimport_",
    "pybytes_",
    "pycapsule_",
    "pyfloat_",
    "pybool_",
    "pythread_",
    "pyo3",
    "numpy",
    "python3",
)
_FORBIDDEN_CAPI_UNDEFINED_PREFIXES = (
    *_FORBIDDEN_CAPI_SYMBOLS,
    "py_",
    "python",
    "libpython",
)
_FORBIDDEN_STATIC_RUNTIME_MARKERS = (
    "__gmp",
    "gmp-mpfr-sys",
    "malachite-base",
    "malachite-nz",
    "malachite-q",
    "mpfr_",
)
_COMMON_SYSTEM_LIBRARIES = {"c", "dl", "m", "pthread", "resolv", "rt", "util"}
_SYSTEM_LIBRARIES = {
    "aarch64-apple-darwin": _COMMON_SYSTEM_LIBRARIES | {"System", "gcc_s", "iconv"},
    "x86_64-apple-darwin": _COMMON_SYSTEM_LIBRARIES | {"System", "gcc_s", "iconv"},
    "x86_64-unknown-linux-gnu": _COMMON_SYSTEM_LIBRARIES | {"gcc_s"},
}
_MACOS_FRAMEWORKS = {"CoreFoundation", "IOKit", "Security", "SystemConfiguration"}
_DEPENDENCY_LOCK_MEMBER = "pyamplicol/assets/release/release-lock.toml"
_CANONICAL_DEPENDENCY_LOCK = Path("dependencies/release-lock.toml")
_PYTHON_RUNTIME_LOCK_MEMBER = "pyamplicol/assets/release/python-runtime-lock.toml"
_CANONICAL_PYTHON_RUNTIME_LOCK = Path("dependencies/python-runtime-lock.toml")
_RUST_LEGAL_INVENTORY = Path("licenses/RUST_THIRD_PARTY.toml")
_REQUIRED_PACKAGE_RESOURCES = {
    "pyamplicol/assets/schemas/README.md",
    "pyamplicol/assets/schemas/artifact-manifest-v3.schema.json",
    "pyamplicol/assets/schemas/runtime-physics-v1.schema.json",
}
_ALLOWED_REPAIR_ROOTS = {"pyamplicol.libs"}
_DISTRIBUTION_SBOM_NAME = "rusticol-python.cyclonedx.json"
_SDK_SBOM_NAME = "pyamplicol/_sdk/sboms/rusticol-capi.cyclonedx.json"
_CANONICAL_SDIST_SINGLE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "justfile",
    "pyproject.toml",
    "rust-toolchain.toml",
}
_CANONICAL_SDIST_TREES = {
    "build_backend",
    "docs",
    "examples",
    "licenses",
    "rust",
    "schemas",
    "src",
    "tests",
    "tools/developer",
    "tools/release",
    "tools/typing",
}
_CANONICAL_SDIST_EXACT_PATHS = {
    "config/release-dependencies.toml",
    "dependencies/install_dependencies.py",
    "dependencies/python-runtime-lock.toml",
    "dependencies/release-lock.toml",
}
_CANONICAL_SDIST_EXCLUDED_PARTS = {
    "__pycache__",
    ".DS_Store",
    ".agent-work",
    ".artifacts",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".trash",
    ".venv",
    "PYPI_DEPLOYMENT_TEST",
    "archive",
    "build",
    "checkouts",
    "dist",
    "outputs",
    "target",
    "venv",
    "wheelhouse",
}
_CANONICAL_SDIST_EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".mod",
    ".out",
    ".pyc",
    ".pyd",
    ".pyo",
    ".run.xml",
    ".synctex.gz",
    ".toc",
    ".whl",
}


class ArtifactError(ReleaseError):
    """An artifact violates the packaging contract."""


@dataclass(frozen=True)
class WheelReport:
    filename: str
    version: str
    size: int
    sha256: str
    python_tag: str
    abi_tag: str
    target: str
    rust_target: str
    sdk_archive_sha256: str
    native_scan: bool


@dataclass(frozen=True)
class SdistReport:
    filename: str
    version: str
    size: int
    sha256: str


def canonicalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _source_inventory_excluded(relative: Path) -> bool:
    if any(part in _CANONICAL_SDIST_EXCLUDED_PARTS for part in relative.parts):
        return True
    name = relative.name
    if name.endswith(tuple(_CANONICAL_SDIST_EXCLUDED_SUFFIXES)):
        return True
    if name.endswith(".egg-info") or any(
        part.endswith(".egg-info") for part in relative.parts
    ):
        return True
    if relative in {
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/install-state.json"),
        Path("src/pyamplicol/_sdk/link.json"),
        Path("src/pyamplicol/_sdk/metadata.json"),
    }:
        return True
    generated_trees = (
        Path("src/pyamplicol/_sdk/fortran"),
        Path("src/pyamplicol/_sdk/include"),
        Path("src/pyamplicol/_sdk/lib"),
        Path("src/pyamplicol/_sdk/sboms"),
    )
    if any(relative.is_relative_to(tree) for tree in generated_trees):
        return True
    selftest_root = Path("src/pyamplicol/assets/selftest")
    if relative.is_relative_to(selftest_root):
        selftest_parts = relative.relative_to(selftest_root).parts
        if selftest_parts and selftest_parts[0] != "portable-64le":
            return True
    return (
        relative.parent == Path("src/pyamplicol")
        and relative.name.startswith("_rusticol")
        and relative.suffix in {".dylib", ".pyd", ".so"}
    )


def _regular_source_members(root: Path, relative_root: Path) -> set[str]:
    members: set[str] = set()
    source = root / relative_root
    if not source.exists():
        return members
    if source.is_symlink():
        raise ArtifactError(f"canonical source inventory contains a symlink: {source}")
    candidates = [source] if source.is_file() else source.rglob("*")
    for candidate in candidates:
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(root)
        if _source_inventory_excluded(relative):
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ArtifactError(
                f"canonical source inventory contains a non-regular file: {candidate}"
            )
        members.add(relative.as_posix())
    return members


@lru_cache(maxsize=1)
def _canonical_sdist_members() -> frozenset[str]:
    members = set(_CANONICAL_SDIST_SINGLE_FILES)
    members.update(_CANONICAL_SDIST_EXACT_PATHS)
    for tree in _CANONICAL_SDIST_TREES:
        members.update(_regular_source_members(ROOT, Path(tree)))
    members.update(_regular_source_members(ROOT, Path("dependencies/patches")))
    members.update(REQUIRED_SDIST_MEMBERS)
    members.add("PKG-INFO")
    return frozenset(members)


@lru_cache(maxsize=1)
def _canonical_wheel_package_members() -> frozenset[str]:
    members: set[str] = set()
    package_root = Path("src/pyamplicol")
    for name in _regular_source_members(ROOT, package_root):
        relative = Path(name).relative_to(package_root)
        if relative.parts[0] in {"_examples"}:
            continue
        if relative == Path("_build_info.json") or relative == Path("_rusticol.pyi"):
            continue
        if relative.is_relative_to(Path("assets/selftest")):
            continue
        if relative.is_relative_to(Path("assets/release")) or relative.is_relative_to(
            Path("assets/schemas")
        ):
            continue
        members.add((Path("pyamplicol") / relative).as_posix())
    for name in _regular_source_members(ROOT, Path("examples")):
        relative = Path(name).relative_to("examples")
        members.add((Path("pyamplicol/_examples") / relative).as_posix())
    stub = Path("rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi")
    if not (ROOT / stub).is_file():
        raise ArtifactError(f"canonical Rusticol stub is missing: {stub}")
    members.add("pyamplicol/_rusticol.pyi")
    members.update(_REQUIRED_SDK_PATHS)
    members.update(_REQUIRED_PACKAGE_RESOURCES)
    members.add(_DEPENDENCY_LOCK_MEMBER)
    members.add(_PYTHON_RUNTIME_LOCK_MEMBER)
    return frozenset(members)


def _safe_member_name(name: str, *, archive: str) -> PurePosixPath:
    stripped = name.rstrip("/")
    if not stripped or "\\" in stripped:
        raise ArtifactError(f"{archive} contains an unsafe member name: {name!r}")
    relative = PurePosixPath(stripped)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ArtifactError(f"{archive} contains an unsafe member path: {name}")
    if _FORBIDDEN_MEMBER_PARTS.intersection(relative.parts):
        raise ArtifactError(f"{archive} contains forbidden build state: {name}")
    return relative


def _wheel_entries(path: Path) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"invalid wheel archive {path}: {error}") from error
    with archive:
        for info in archive.infolist():
            relative = _safe_member_name(info.filename, archive="wheel")
            name = relative.as_posix()
            if info.is_dir():
                continue
            if name in entries:
                raise ArtifactError(f"wheel contains duplicate member: {name}")
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ArtifactError(f"wheel contains a symbolic link: {name}")
            if info.flag_bits & 0x1:
                raise ArtifactError(f"wheel contains an encrypted member: {name}")
            entries[name] = archive.read(info)
    return entries


def _single_name(entries: dict[str, bytes], suffix: str) -> str:
    matches = sorted(name for name in entries if name.endswith(suffix))
    if len(matches) != 1:
        raise ArtifactError(
            f"wheel must contain exactly one {suffix}, found {len(matches)}"
        )
    return matches[0]


def _parse_wheel_filename(path: Path) -> tuple[str, str, str, str]:
    if path.suffix != ".whl":
        raise ArtifactError(f"not a wheel filename: {path.name}")
    fields = path.name[:-4].rsplit("-", 3)
    if len(fields) != 4:
        raise ArtifactError(f"invalid wheel filename: {path.name}")
    prefix, python_tag, abi_tag, platform_tag = fields
    distribution_version = prefix.rsplit("-", 1)
    if len(distribution_version) != 2:
        raise ArtifactError(f"invalid wheel distribution/version: {path.name}")
    distribution, filename_version = distribution_version
    if canonicalize_name(distribution) != EXPECTED_DISTRIBUTION:
        raise ArtifactError(f"wheel distribution is not pyamplicol: {path.name}")
    return filename_version, python_tag, abi_tag, platform_tag


def _canonical_target(platform_tag: str) -> tuple[str, str]:
    components = set(platform_tag.split("."))
    if components and components <= _LINUX_PLATFORM_TAGS:
        target = "manylinux_2_28_x86_64"
        return target, RELEASE_TARGETS[target]
    if len(components) == 1:
        target = next(iter(components))
        rust_target = RELEASE_TARGETS.get(target)
        if rust_target is not None:
            return target, rust_target
    raise ArtifactError(
        "wheel platform tag must be macosx_11_0_arm64, "
        "macosx_11_0_x86_64, or manylinux_2_28_x86_64; "
        f"found {platform_tag}"
    )


def _message(data: bytes, description: str):
    try:
        return BytesParser(policy=policy.default).parsebytes(data)
    except Exception as error:  # pragma: no cover - email parser is defensive
        raise ArtifactError(f"could not parse {description}: {error}") from error


def _reject_direct_requirements(
    requirements: list[str], *, description: str = "wheel metadata"
) -> None:
    direct = re.compile(
        r"(?i)(?:\s@\s|(?:file|git\+[^:]+|https?)://|(?:^|[ (])\.\.?[/\\])"
    )
    invalid = [
        requirement for requirement in requirements if direct.search(requirement)
    ]
    if invalid:
        raise ArtifactError(
            f"{description} contains direct URL or path dependencies: "
            + ", ".join(invalid)
        )


def _project_requirements(pyproject: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise ArtifactError("sdist pyproject.toml [project] must be a table")
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise ArtifactError("sdist project dependencies must be strings")
    requirements.extend(dependencies)
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict) or not all(
        isinstance(group, list) and all(isinstance(item, str) for item in group)
        for group in optional.values()
    ):
        raise ArtifactError("sdist optional dependencies must be string lists")
    for group in optional.values():
        requirements.extend(group)
    build_system = pyproject.get("build-system", {})
    if not isinstance(build_system, dict):
        raise ArtifactError("sdist pyproject.toml [build-system] must be a table")
    build_requirements = build_system.get("requires", [])
    if not isinstance(build_requirements, list) or not all(
        isinstance(item, str) for item in build_requirements
    ):
        raise ArtifactError("sdist build requirements must be strings")
    requirements.extend(build_requirements)
    return requirements


def _dependency_lock(entries: dict[str, bytes]) -> dict[str, Any]:
    if _DEPENDENCY_LOCK_MEMBER not in entries:
        raise ArtifactError(
            f"wheel is missing packaged dependency lock: {_DEPENDENCY_LOCK_MEMBER}"
        )
    packaged = entries[_DEPENDENCY_LOCK_MEMBER]
    canonical_path = ROOT / _CANONICAL_DEPENDENCY_LOCK
    if not canonical_path.is_file():
        raise ArtifactError(f"canonical dependency lock is missing: {canonical_path}")
    if packaged != canonical_path.read_bytes():
        raise ArtifactError(
            "wheel packaged dependency lock differs from canonical lock"
        )
    try:
        payload = tomllib.loads(packaged.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(f"wheel dependency lock is invalid: {error}") from error
    if payload.get("schema_version") != 1:
        raise ArtifactError("wheel dependency lock must use schema_version = 1")
    return payload


def _python_runtime_lock(
    entries: dict[str, bytes], release_lock: dict[str, Any]
) -> dict[str, Any]:
    if _PYTHON_RUNTIME_LOCK_MEMBER not in entries:
        raise ArtifactError(
            "wheel is missing packaged Python runtime lock: "
            f"{_PYTHON_RUNTIME_LOCK_MEMBER}"
        )
    packaged = entries[_PYTHON_RUNTIME_LOCK_MEMBER]
    canonical_path = ROOT / _CANONICAL_PYTHON_RUNTIME_LOCK
    if not canonical_path.is_file():
        raise ArtifactError(
            f"canonical Python runtime lock is missing: {canonical_path}"
        )
    if packaged != canonical_path.read_bytes():
        raise ArtifactError(
            "wheel packaged Python runtime lock differs from canonical lock"
        )
    contract = release_lock.get("python_runtime_lock")
    if (
        not isinstance(contract, dict)
        or contract.get("path") != "dependencies/python-runtime-lock.toml"
        or contract.get("sha256") != hashlib.sha256(packaged).hexdigest()
    ):
        raise ArtifactError(
            "wheel Python runtime lock is not hash-bound by release-lock.toml"
        )
    try:
        payload = tomllib.loads(packaged.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(f"wheel Python runtime lock is invalid: {error}") from error
    if payload.get("schema_version") != 1:
        raise ArtifactError("wheel Python runtime lock must use schema_version = 1")
    return payload


def _validate_sdist_python_runtime_lock(relative_files: dict[str, Path]) -> None:
    release_path = relative_files.get("dependencies/release-lock.toml")
    runtime_path = relative_files.get("dependencies/python-runtime-lock.toml")
    if release_path is None or runtime_path is None:
        raise ArtifactError("sdist is missing its Python dependency lock pair")
    try:
        with release_path.open("rb") as stream:
            release_lock = tomllib.load(stream)
        with runtime_path.open("rb") as stream:
            runtime_lock = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(
            f"sdist contains an invalid dependency lock: {error}"
        ) from error
    contract = release_lock.get("python_runtime_lock")
    if (
        release_lock.get("schema_version") != 1
        or runtime_lock.get("schema_version") != 1
        or not isinstance(contract, dict)
        or contract.get("path") != "dependencies/python-runtime-lock.toml"
        or contract.get("sha256")
        != hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    ):
        raise ArtifactError("sdist Python runtime lock is not hash-bound correctly")


def _validate_wheel_resource_layout(entries: dict[str, bytes]) -> None:
    dist_info_roots = {
        PurePosixPath(name).parts[0]
        for name in entries
        if PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if len(dist_info_roots) != 1:
        raise ArtifactError(
            "wheel must contain exactly one top-level .dist-info directory"
        )
    allowed_roots = {"pyamplicol", *dist_info_roots, *_ALLOWED_REPAIR_ROOTS}
    misplaced = sorted(
        name for name in entries if PurePosixPath(name).parts[0] not in allowed_roots
    )
    if misplaced:
        raise ArtifactError(
            "wheel contains members outside pyamplicol, its .dist-info, or an "
            "approved repair-library root: " + ", ".join(misplaced)
        )
    missing = sorted(_REQUIRED_PACKAGE_RESOURCES - entries.keys())
    if missing:
        raise ArtifactError(
            "wheel is missing required package-owned resources: " + ", ".join(missing)
        )


def _required_legal_files() -> tuple[str, ...]:
    path = ROOT / _RUST_LEGAL_INVENTORY
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(
            f"could not read legal inventory {path}: {error}"
        ) from error
    raw = payload.get("required_release_files")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) for item in raw)
    ):
        raise ArtifactError(
            f"legal inventory {path} has no required_release_files string list"
        )
    normalized = tuple(
        _safe_member_name(item, archive="legal inventory").as_posix() for item in raw
    )
    if len(normalized) != len(set(normalized)):
        raise ArtifactError(f"legal inventory {path} repeats a release file")
    return normalized


@lru_cache(maxsize=1)
def _legal_cargo_inventory() -> frozenset[tuple[str, str]]:
    path = ROOT / _RUST_LEGAL_INVENTORY
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(
            f"could not read legal inventory {path}: {error}"
        ) from error
    identities: list[tuple[str, str]] = []
    for table in ("first_party", "package"):
        entries = payload.get(table)
        if not isinstance(entries, list) or not entries:
            raise ArtifactError(f"legal inventory {path} has no {table} entries")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ArtifactError(
                    f"legal inventory {path} {table}[{index}] is not a table"
                )
            name = entry.get("name")
            version = entry.get("version")
            if not isinstance(name, str) or not name or not isinstance(version, str):
                raise ArtifactError(
                    f"legal inventory {path} {table}[{index}] lacks name/version"
                )
            identities.append((name, version))
    if len(identities) != len(set(identities)):
        raise ArtifactError(f"legal inventory {path} repeats a package identity")
    return frozenset(identities)


@cache
def _target_cargo_inventory(
    package: str, rust_target: str
) -> frozenset[tuple[str, str]]:
    try:
        from check_rust_licenses import (  # imported lazily by the release audit
            SHIPPED_CARGO_ARTIFACTS,
            CargoClosureError,
            cargo_dependency_closure,
        )
    except ImportError as error:  # pragma: no cover - source layout is mandatory
        raise ArtifactError(
            "release audit cannot import the Cargo closure verifier"
        ) from error
    matches = [spec for spec in SHIPPED_CARGO_ARTIFACTS if spec.package == package]
    if len(matches) != 1:
        raise ArtifactError(f"release audit has no unique Cargo spec for {package}")
    try:
        closure = cargo_dependency_closure(ROOT, matches[0], rust_target)
    except CargoClosureError as error:
        raise ArtifactError(
            f"cannot resolve release Cargo closure for {package} on {rust_target}: "
            f"{error}"
        ) from error
    identities = frozenset((item.name, item.version) for item in closure.packages)
    if not identities or len(identities) != len(closure.packages):
        raise ArtifactError(
            f"Cargo closure for {package} on {rust_target} has ambiguous identities"
        )
    unknown = identities - _legal_cargo_inventory()
    if unknown:
        raise ArtifactError(
            "Cargo closure disagrees with the legal inventory: "
            + ", ".join(f"{name}@{version}" for name, version in sorted(unknown))
        )
    return identities


def _candidate_cargo_inventory(
    *, version: str, lock: dict[str, Any]
) -> frozenset[tuple[str, str]]:
    cargo_version = version.replace(".dev0+", "-dev.0+")
    legal = set(_legal_cargo_inventory())
    first_party = {"rusticol-capi", "rusticol-core", "rusticol-python"}
    legal.update((name, cargo_version) for name in first_party)
    symjit = lock.get("symjit")
    candidate_version = (
        symjit.get("candidate_version") if isinstance(symjit, dict) else None
    )
    if not isinstance(candidate_version, str) or not candidate_version:
        raise ArtifactError("candidate dependency lock has no SymJIT candidate version")
    legal.add(("symjit", candidate_version))
    return frozenset(legal)


def _cargo_purl_identity(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    decoded = unquote(value)
    if "?" in decoded or "#" in decoded:
        return None
    return decoded


def _validate_cyclonedx_graph(
    document: dict[str, Any],
    *,
    description: str,
    root_name: str,
    version: str,
    rust_target: str,
    mode: str,
    lock: dict[str, Any],
) -> frozenset[tuple[str, str]]:
    metadata = document.get("metadata")
    if (
        document.get("bomFormat") != "CycloneDX"
        or document.get("specVersion") != "1.5"
        or document.get("version") != 1
        or not isinstance(metadata, dict)
    ):
        raise ArtifactError(f"{description} is not a CycloneDX 1.5 document")
    root = metadata.get("component")
    components = document.get("components")
    dependencies = document.get("dependencies")
    if not isinstance(root, dict):
        raise ArtifactError(f"{description} has no root component")
    if not isinstance(components, list) or not components:
        raise ArtifactError(f"{description} must contain a nonempty component list")
    if not isinstance(dependencies, list) or not dependencies:
        raise ArtifactError(f"{description} must contain a nonempty dependency graph")

    cargo_version = (
        version if mode == "release" else version.replace(".dev0+", "-dev.0+")
    )
    records = [root, *components]
    references: dict[str, tuple[str, str]] = {}
    identities: set[tuple[str, str]] = set()
    purls: set[str] = set()
    for index, component in enumerate(records):
        label = "root" if index == 0 else f"component[{index - 1}]"
        if not isinstance(component, dict):
            raise ArtifactError(f"{description} {label} is not an object")
        name = component.get("name")
        component_version = component.get("version")
        reference = component.get("bom-ref")
        purl = _cargo_purl_identity(component.get("purl"))
        if (
            component.get("type") != "library"
            or component.get("scope") not in {"required", "excluded"}
            or not isinstance(name, str)
            or not name
            or not isinstance(component_version, str)
            or not component_version
            or not isinstance(reference, str)
            or not reference
            or purl != f"pkg:cargo/{name}@{component_version}"
        ):
            raise ArtifactError(
                f"{description} {label} has invalid Cargo component identity"
            )
        if index == 0 and component.get("scope") != "required":
            raise ArtifactError(f"{description} root component must be required")
        if reference in references:
            raise ArtifactError(f"{description} repeats component bom-ref {reference}")
        identity = (name, component_version)
        if identity in identities:
            raise ArtifactError(
                f"{description} repeats component identity {name}@{component_version}"
            )
        if purl in purls:
            raise ArtifactError(f"{description} repeats component purl {purl}")
        references[reference] = identity
        identities.add(identity)
        purls.add(purl)

    root_reference = str(root["bom-ref"])
    if (
        root.get("name") != root_name
        or root.get("version") != cargo_version
        or _cargo_purl_identity(root_reference)
        != f"pkg:cargo/{root_name}@{cargo_version}"
    ):
        raise ArtifactError(f"{description} root identity/version/purl is inconsistent")

    edges: dict[str, set[str]] = {}
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            raise ArtifactError(f"{description} dependency[{index}] is not an object")
        reference = dependency.get("ref")
        raw_children = dependency.get("dependsOn", [])
        if not isinstance(reference, str) or not reference:
            raise ArtifactError(f"{description} dependency[{index}] has no ref")
        if reference in edges:
            raise ArtifactError(f"{description} repeats dependency ref {reference}")
        if not isinstance(raw_children, list) or not all(
            isinstance(child, str) and child for child in raw_children
        ):
            raise ArtifactError(
                f"{description} dependency[{index}] has invalid dependsOn"
            )
        if len(raw_children) != len(set(raw_children)):
            raise ArtifactError(
                f"{description} dependency[{index}] repeats a dependency edge"
            )
        children = set(raw_children)
        if reference in children:
            raise ArtifactError(f"{description} contains a self dependency")
        edges[reference] = children
    reference_set = set(references)
    if set(edges) != reference_set:
        missing = sorted(reference_set - set(edges))
        extra = sorted(set(edges) - reference_set)
        raise ArtifactError(
            f"{description} dependency/component inventory differs; "
            f"missing={missing}, extra={extra}"
        )
    dangling = sorted(
        child
        for children in edges.values()
        for child in children
        if child not in references
    )
    if dangling:
        raise ArtifactError(
            f"{description} dependency graph has dangling references: {dangling}"
        )
    reachable: set[str] = set()
    pending = [root_reference]
    while pending:
        reference = pending.pop()
        if reference in reachable:
            continue
        reachable.add(reference)
        pending.extend(edges[reference] - reachable)
    if reachable != reference_set:
        raise ArtifactError(
            f"{description} dependency graph has unreachable components: "
            f"{sorted(reference_set - reachable)}"
        )

    identity_set = frozenset(identities)
    allowed = (
        _legal_cargo_inventory()
        if mode == "release"
        else _candidate_cargo_inventory(version=version, lock=lock)
    )
    unknown = identity_set - allowed
    if unknown:
        raise ArtifactError(
            f"{description} disagrees with the Cargo legal inventory: "
            + ", ".join(
                f"{name}@{item_version}" for name, item_version in sorted(unknown)
            )
        )
    if mode == "release":
        expected = _target_cargo_inventory(root_name, rust_target)
        if identity_set != expected:
            missing = sorted(expected - identity_set)
            extra = sorted(identity_set - expected)
            raise ArtifactError(
                f"{description} disagrees with the target Cargo closure; "
                f"missing={missing}, extra={extra}"
            )
    return identity_set


def _component_license_values(component: dict[str, Any]) -> set[str]:
    raw = component.get("licenses")
    if not isinstance(raw, list) or not raw:
        return set()
    values: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            return set()
        expression = item.get("expression")
        if isinstance(expression, str) and expression:
            values.add(expression)
            continue
        license_record = item.get("license")
        if not isinstance(license_record, dict):
            return set()
        name = license_record.get("name")
        if not isinstance(name, str) or not name:
            return set()
        values.add(name)
    return values


def _component_properties(component: dict[str, Any]) -> dict[str, list[str]]:
    raw = component.get("properties", [])
    if not isinstance(raw, list):
        raise ArtifactError("distribution SBOM component properties must be a list")
    properties: dict[str, list[str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ArtifactError("distribution SBOM component property is not an object")
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ArtifactError("distribution SBOM component property is invalid")
        properties.setdefault(name, []).append(value)
    return properties


def _locked_python_graph(
    runtime_lock: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_packages = runtime_lock.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ArtifactError("Python runtime lock has no package graph")
    packages: dict[str, dict[str, Any]] = {}
    artifact_names: set[str] = set()
    for index, raw in enumerate(raw_packages):
        if not isinstance(raw, dict):
            raise ArtifactError(f"Python runtime package[{index}] is not a table")
        distribution = raw.get("distribution")
        version = raw.get("version")
        license_name = raw.get("license")
        direct = raw.get("direct")
        dependencies = raw.get("dependencies")
        artifacts = raw.get("artifacts", [])
        if (
            not isinstance(distribution, str)
            or not distribution
            or not isinstance(version, str)
            or not version
            or not isinstance(license_name, str)
            or not license_name
            or not isinstance(direct, bool)
            or not isinstance(dependencies, list)
            or not all(isinstance(item, str) and item for item in dependencies)
            or not isinstance(artifacts, list)
        ):
            raise ArtifactError(f"Python runtime package[{index}] is invalid")
        name = canonicalize_name(distribution)
        if name in packages:
            raise ArtifactError(f"Python runtime lock repeats package {name}")
        dependency_names = tuple(canonicalize_name(item) for item in dependencies)
        if len(dependency_names) != len(set(dependency_names)):
            raise ArtifactError(f"Python runtime package {name} repeats a dependency")
        locked_artifacts: list[dict[str, str]] = []
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ArtifactError(
                    f"Python runtime package {name} artifact[{artifact_index}] "
                    "is invalid"
                )
            filename = artifact.get("filename")
            digest = artifact.get("sha256")
            if (
                not isinstance(filename, str)
                or not filename
                or "/" in filename
                or "\\" in filename
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ArtifactError(
                    f"Python runtime package {name} has an invalid artifact"
                )
            if filename in artifact_names:
                raise ArtifactError(f"Python runtime lock repeats artifact {filename}")
            artifact_names.add(filename)
            locked_artifacts.append({"filename": filename, "sha256": digest})
        packages[name] = {
            "distribution": distribution,
            "version": version,
            "license": license_name,
            "direct": direct,
            "dependencies": dependency_names,
            "artifacts": tuple(
                sorted(
                    locked_artifacts,
                    key=lambda item: (item["filename"], item["sha256"]),
                )
            ),
        }
    for name, package in packages.items():
        missing = sorted(set(package["dependencies"]) - set(packages))
        if missing:
            raise ArtifactError(
                f"Python runtime package {name} references unlocked packages: "
                + ", ".join(missing)
            )
    return packages


def _validate_distribution_cyclonedx_graph(
    document: dict[str, Any],
    *,
    version: str,
    rust_target: str,
    mode: str,
    release_lock: dict[str, Any],
    runtime_lock: dict[str, Any],
) -> frozenset[tuple[str, str]]:
    metadata = document.get("metadata")
    components = document.get("components")
    dependencies = document.get("dependencies")
    if (
        document.get("bomFormat") != "CycloneDX"
        or document.get("specVersion") != "1.5"
        or document.get("version") != 1
        or not isinstance(metadata, dict)
        or not isinstance(components, list)
        or not components
        or not isinstance(dependencies, list)
        or not dependencies
    ):
        raise ArtifactError(
            "distribution SBOM is not a complete CycloneDX 1.5 document"
        )
    root = metadata.get("component")
    if not isinstance(root, dict):
        raise ArtifactError("distribution SBOM has no root component")

    records = [root, *components]
    by_reference: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str]] = set()
    kinds: dict[str, str] = {}
    for index, component in enumerate(records):
        label = "root" if index == 0 else f"component[{index - 1}]"
        if not isinstance(component, dict):
            raise ArtifactError(f"distribution SBOM {label} is not an object")
        name = component.get("name")
        item_version = component.get("version")
        reference = component.get("bom-ref")
        purl = component.get("purl")
        if (
            component.get("type") != "library"
            or component.get("scope") not in {"required", "excluded"}
            or not isinstance(name, str)
            or not name
            or not isinstance(item_version, str)
            or not item_version
            or not isinstance(reference, str)
            or not reference
            or not isinstance(purl, str)
            or reference != purl
        ):
            raise ArtifactError(
                f"distribution SBOM {label} has invalid component identity"
            )
        decoded = unquote(reference)
        match = re.fullmatch(r"pkg:(cargo|pypi)/([^@]+)@(.+)", decoded)
        if match is None:
            raise ArtifactError(
                f"distribution SBOM {label} has an unsupported package URL"
            )
        kind, purl_name, purl_version = match.groups()
        expected_name = name if kind == "cargo" else canonicalize_name(name)
        if purl_name != expected_name or purl_version != item_version:
            raise ArtifactError(
                f"distribution SBOM {label} has an inconsistent package URL"
            )
        identity = (kind, purl_name, item_version)
        if reference in by_reference or identity in identities:
            raise ArtifactError("distribution SBOM repeats a component identity")
        by_reference[reference] = component
        identities.add(identity)
        kinds[reference] = kind

    root_reference = str(root["bom-ref"])
    expected_root = _pypi_purl(EXPECTED_DISTRIBUTION, version)
    if (
        root_reference != expected_root
        or root.get("name") != EXPECTED_DISTRIBUTION
        or root.get("version") != version
        or root.get("scope") != "required"
        or _component_license_values(root) != {"0BSD"}
    ):
        raise ArtifactError("distribution SBOM root identity is inconsistent")

    edges: dict[str, set[str]] = {}
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            raise ArtifactError(
                f"distribution SBOM dependency[{index}] is not an object"
            )
        reference = dependency.get("ref")
        raw_children = dependency.get("dependsOn", [])
        if (
            not isinstance(reference, str)
            or not reference
            or reference in edges
            or not isinstance(raw_children, list)
            or not all(isinstance(child, str) and child for child in raw_children)
            or len(raw_children) != len(set(raw_children))
            or reference in raw_children
        ):
            raise ArtifactError(f"distribution SBOM dependency[{index}] is invalid")
        edges[reference] = set(raw_children)
    references = set(by_reference)
    if set(edges) != references:
        raise ArtifactError(
            "distribution SBOM component and dependency inventories differ"
        )
    dangling = sorted(
        child
        for children in edges.values()
        for child in children
        if child not in references
    )
    if dangling:
        raise ArtifactError(
            f"distribution SBOM has dangling dependency references: {dangling}"
        )
    reachable: set[str] = set()
    pending = [root_reference]
    while pending:
        reference = pending.pop()
        if reference in reachable:
            continue
        reachable.add(reference)
        pending.extend(edges[reference] - reachable)
    if reachable != references:
        raise ArtifactError(
            "distribution SBOM has unreachable components: "
            f"{sorted(references - reachable)}"
        )

    cargo_version = (
        version if mode == "release" else version.replace(".dev0+", "-dev.0+")
    )
    cargo_root_ref = f"pkg:cargo/rusticol-python@{cargo_version}"
    cargo_references = {
        reference for reference, kind in kinds.items() if kind == "cargo"
    }
    if cargo_root_ref not in cargo_references:
        raise ArtifactError("distribution SBOM has no Rusticol-Python Cargo root")
    for reference in cargo_references:
        non_cargo = edges[reference] - cargo_references
        if non_cargo:
            raise ArtifactError(
                "distribution SBOM Cargo graph depends on Python components"
            )
    cargo_document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": by_reference[cargo_root_ref]},
        "components": [
            by_reference[reference]
            for reference in sorted(cargo_references - {cargo_root_ref})
        ],
        "dependencies": [
            {"ref": reference, "dependsOn": sorted(edges[reference])}
            for reference in sorted(cargo_references)
        ],
    }
    cargo_inventory = _validate_cyclonedx_graph(
        cargo_document,
        description="distribution SBOM Cargo graph",
        root_name="rusticol-python",
        version=version,
        rust_target=rust_target,
        mode=mode,
        lock=release_lock,
    )

    packages = _locked_python_graph(runtime_lock)
    expected_python_refs = {
        name: _pypi_purl(name, str(package["version"]))
        for name, package in packages.items()
    }
    actual_python_refs = {
        reference
        for reference, kind in kinds.items()
        if kind == "pypi" and reference != root_reference
    }
    if actual_python_refs != set(expected_python_refs.values()):
        raise ArtifactError(
            "distribution SBOM Python inventory disagrees with the runtime lock"
        )
    for name, package in packages.items():
        reference = expected_python_refs[name]
        component = by_reference[reference]
        if (
            component.get("name") != package["distribution"]
            or component.get("version") != package["version"]
            or _component_license_values(component) != {package["license"]}
        ):
            raise ArtifactError(
                f"distribution SBOM component for {name} disagrees with the lock"
            )
        properties = _component_properties(component)
        expected_artifacts = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in package["artifacts"]
        }
        expected_property_names = {"pyamplicol:pypi:hashes-verified"}
        if expected_artifacts:
            expected_property_names.add("pyamplicol:pypi:artifact")
        actual_artifacts = set(properties.get("pyamplicol:pypi:artifact", []))
        if (
            set(properties) != expected_property_names
            or actual_artifacts != expected_artifacts
        ):
            raise ArtifactError(
                f"distribution SBOM artifact records for {name} disagree with the lock"
            )
        expected_verified = str(bool(package["artifacts"])).lower()
        if properties["pyamplicol:pypi:hashes-verified"] != [expected_verified]:
            raise ArtifactError(
                f"distribution SBOM hash status for {name} disagrees with the lock"
            )
        expected_children = {
            expected_python_refs[dependency] for dependency in package["dependencies"]
        }
        if edges[reference] != expected_children:
            raise ArtifactError(
                f"distribution SBOM dependency edges for {name} disagree with the lock"
            )

    expected_root_children = {
        cargo_root_ref,
        *(
            expected_python_refs[name]
            for name, package in packages.items()
            if package["direct"]
        ),
    }
    if edges[root_reference] != expected_root_children:
        raise ArtifactError(
            "distribution SBOM root dependencies disagree with the runtime lock"
        )
    return cargo_inventory


def _locked_python_dependencies(lock: dict[str, Any]) -> dict[str, str]:
    raw_dependencies = lock.get("python_dependencies")
    if not isinstance(raw_dependencies, list):
        raise ArtifactError("wheel dependency lock has no Python dependency list")
    dependencies: dict[str, str] = {}
    for entry in raw_dependencies:
        if not isinstance(entry, dict):
            raise ArtifactError("wheel dependency lock entries must be tables")
        distribution = entry.get("distribution")
        version = entry.get("version")
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ArtifactError("wheel dependency lock entries need name and version")
        name = canonicalize_name(distribution)
        if name in dependencies:
            raise ArtifactError(f"wheel dependency lock repeats {name}")
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ArtifactError(
                f"wheel dependency lock has invalid version for {name}: {version}"
            ) from error
        dependencies[name] = normalized_version
    return dependencies


def _exact_locked_dependencies(
    lock: dict[str, Any], dependencies: dict[str, str]
) -> dict[str, str]:
    symbolica = lock.get("symbolica")
    loader = lock.get("ufo_model_loader")
    if not isinstance(symbolica, dict) or not isinstance(loader, dict):
        raise ArtifactError("wheel dependency lock lacks patched dependency contracts")
    raw_exact = (
        (symbolica.get("python_distribution"), symbolica.get("python_version")),
        (loader.get("python_distribution"), loader.get("required_version")),
    )
    exact: dict[str, str] = {}
    for distribution, version in raw_exact:
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ArtifactError(
                "wheel dependency lock has invalid exact dependency data"
            )
        name = canonicalize_name(distribution)
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ArtifactError(
                f"wheel dependency lock has invalid exact version for {name}: {version}"
            ) from error
        if dependencies.get(name) != normalized_version:
            raise ArtifactError(
                f"wheel dependency lock disagrees about exact {name} version"
            )
        exact[name] = normalized_version
    return exact


def _validate_runtime_requirements(
    raw_requirements: list[str],
    lock: dict[str, Any],
    *,
    require_exact: bool,
) -> None:
    dependencies = _locked_python_dependencies(lock)
    exact = _exact_locked_dependencies(lock, dependencies)
    requirements: dict[str, Requirement] = {}

    def marker_uses_extra(items: Any) -> bool:
        if isinstance(items, tuple):
            return any(
                isinstance(item, Variable) and item.value == "extra" for item in items
            )
        if isinstance(items, list):
            return any(marker_uses_extra(item) for item in items)
        return False

    for raw in raw_requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            raise ArtifactError(f"wheel has invalid Requires-Dist: {raw}") from error
        if requirement.marker is not None and marker_uses_extra(
            requirement.marker._markers
        ):
            continue
        if requirement.marker is not None:
            raise ArtifactError(
                "locked runtime Requires-Dist entries may not use environment markers: "
                f"{raw}"
            )
        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ArtifactError(f"wheel metadata repeats runtime dependency {name}")
        if requirement.extras or requirement.url:
            raise ArtifactError(
                f"wheel runtime dependency must be a plain version requirement: {raw}"
            )
        requirements[name] = requirement
    if set(requirements) != set(dependencies):
        missing = sorted(set(dependencies) - set(requirements))
        extra = sorted(set(requirements) - set(dependencies))
        raise ArtifactError(
            "wheel Requires-Dist inventory disagrees with dependency lock; "
            f"missing={missing}, extra={extra}"
        )
    for name, locked_version in dependencies.items():
        requirement = requirements[name]
        if (
            not requirement.specifier
            or Version(locked_version) not in requirement.specifier
        ):
            raise ArtifactError(
                f"wheel Requires-Dist for {name} excludes locked version "
                f"{locked_version}"
            )
        if require_exact or name in exact:
            specifiers = list(requirement.specifier)
            exact_match = False
            if len(specifiers) == 1 and specifiers[0].operator == "==":
                try:
                    exact_match = str(Version(specifiers[0].version)) == locked_version
                except InvalidVersion:
                    exact_match = False
            if not exact_match:
                raise ArtifactError(
                    f"wheel Requires-Dist must pin {name}=={locked_version} exactly"
                )


def _validate_legal_members(
    entries: dict[str, bytes], metadata_name: str, metadata: Any
) -> None:
    license_files = list(metadata.get_all("License-File", []))
    declared: list[str] = []
    for raw_name in license_files:
        relative = _safe_member_name(str(raw_name), archive="License-File")
        declared.append(relative.as_posix())
    if len(declared) != len(set(declared)):
        raise ArtifactError("wheel metadata repeats a License-File declaration")
    required = _required_legal_files()
    if set(declared) != set(required):
        missing = sorted(set(required) - set(declared))
        extra = sorted(set(declared) - set(required))
        raise ArtifactError(
            "wheel License-File inventory disagrees with the legal manifest; "
            f"missing={missing}, extra={extra}"
        )
    dist_info = metadata_name.removesuffix("METADATA")
    for relative in required:
        member = f"{dist_info}licenses/{relative}"
        if member not in entries or not entries[member].strip():
            raise ArtifactError(f"wheel is missing nonempty legal member: {member}")


def _record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _validate_record(entries: dict[str, bytes], record_name: str) -> None:
    try:
        rows = list(csv.reader(io.StringIO(entries[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ArtifactError(f"invalid wheel RECORD: {error}") from error
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ArtifactError(f"wheel RECORD row must have three fields: {row!r}")
        name, digest, size = row
        safe_name = _safe_member_name(name, archive="RECORD").as_posix()
        if safe_name in records:
            raise ArtifactError(f"wheel RECORD repeats {safe_name}")
        records[safe_name] = (digest, size)
    if set(records) != set(entries):
        missing = sorted(set(entries) - set(records))
        extra = sorted(set(records) - set(entries))
        raise ArtifactError(
            f"wheel RECORD inventory mismatch; missing={missing}, extra={extra}"
        )
    for name, data in entries.items():
        digest, size = records[name]
        if name == record_name:
            if digest or size:
                raise ArtifactError("wheel RECORD must not hash itself")
            continue
        expected_digest = f"sha256={_record_digest(data)}"
        if digest != expected_digest or size != str(len(data)):
            raise ArtifactError(f"wheel RECORD hash/size mismatch for {name}")


def _json_object(entries: dict[str, bytes], name: str) -> dict[str, Any]:
    try:
        payload = json.loads(entries[name])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid JSON resource {name}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"JSON resource must be an object: {name}")
    return payload


def _sanitized_native_bytes(data: bytes, *, allow_local_rustup: bool) -> bytes:
    scanned = re.sub(
        rb"/rustc/[0-9a-f]+/library/std/src/\.\./\.\./backtrace/",
        b"/rustc/standard-library-backtrace/",
        data,
    )
    scanned = re.sub(
        rb"library/std/src/\.\./\.\./backtrace/",
        b"library/standard-library-backtrace/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:Users|home)/runner/work/rust/rust/",
        b"/rust/toolchain-source/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:private/)?var/folders/[^/\x00]+/[^/\x00]+/T/"
        rb"pyamplicol-build-[^/\x00]+/cargo-home/registry/src/",
        b"/cargo/registry/src/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:tmp|var/tmp)/pyamplicol-build-[^/\x00]+/"
        rb"cargo-home/registry/src/",
        b"/cargo/registry/src/",
        scanned,
    )
    if allow_local_rustup:
        scanned = re.sub(
            rb"/(?:Users|home)/[^/\x00]+/\.rustup/toolchains/[^/\x00]+/"
            rb"lib/rustlib/src/rust/library/",
            b"/rust/sysroot/library/",
            scanned,
        )
    return scanned


def _scan_embedded_paths(
    entries: dict[str, bytes], *, allow_local_rustup: bool = False
) -> None:
    for name, data in entries.items():
        scanned = data
        if name.lower().endswith(_NATIVE_MEMBER_SUFFIXES):
            scanned = _sanitized_native_bytes(
                data, allow_local_rustup=allow_local_rustup
            )
        lowered = scanned.lower()
        markers = _FORBIDDEN_PATH_MARKERS
        if name.lower().endswith((".dylib", ".pyd", ".so")):
            markers += _FORBIDDEN_NATIVE_RELATIVE_MARKERS
        for marker in markers:
            if marker and marker.lower() in lowered:
                raise ArtifactError(
                    f"wheel member {name} embeds non-relocatable path marker "
                    f"{os.fsdecode(marker)!r}"
                )


def _validate_sdk(
    entries: dict[str, bytes],
    *,
    version: str,
    rust_target: str,
    mode: str,
    lock: dict[str, Any],
) -> tuple[str, frozenset[tuple[str, str]]]:
    missing = sorted(_REQUIRED_SDK_PATHS - set(entries))
    if missing:
        raise ArtifactError("wheel is missing SDK resources: " + ", ".join(missing))
    metadata = _json_object(entries, "pyamplicol/_sdk/metadata.json")
    link = _json_object(entries, "pyamplicol/_sdk/link.json")
    if metadata.get("schema_version") != 1 or link.get("schema_version") != 1:
        raise ArtifactError("unsupported SDK metadata schema")
    if metadata.get("abi_version") != 1:
        raise ArtifactError("Rusticol SDK must provide C ABI version 1")
    if metadata.get("target") != rust_target or link.get("target") != rust_target:
        raise ArtifactError("SDK target does not match the wheel platform")
    metadata_version = str(metadata.get("version", ""))
    if version != metadata_version:
        raise ArtifactError("SDK version does not match wheel metadata")
    archive = PurePosixPath(str(metadata.get("archive", "")))
    if archive.is_absolute() or any(part in {"", ".", ".."} for part in archive.parts):
        raise ArtifactError("SDK archive path must be relative and confined")
    archive_name = (PurePosixPath("pyamplicol/_sdk") / archive).as_posix()
    if archive_name != "pyamplicol/_sdk/lib/librusticol_capi.a":
        raise ArtifactError(f"unexpected SDK archive path: {archive}")
    actual_digest = hashlib.sha256(entries[archive_name]).hexdigest()
    if metadata.get("archive_sha256") != actual_digest:
        raise ArtifactError("SDK archive SHA-256 does not match metadata.json")
    sbom = PurePosixPath(str(metadata.get("sbom", "")))
    if sbom.as_posix() != "sboms/rusticol-capi.cyclonedx.json":
        raise ArtifactError(f"unexpected SDK SBOM path: {sbom}")
    sbom_name = (PurePosixPath("pyamplicol/_sdk") / sbom).as_posix()
    if sbom_name != _SDK_SBOM_NAME:
        raise ArtifactError(f"unexpected SDK SBOM member: {sbom_name}")
    sbom_bytes = entries[sbom_name]
    if metadata.get("sbom_sha256") != hashlib.sha256(sbom_bytes).hexdigest():
        raise ArtifactError("SDK SBOM SHA-256 does not match metadata.json")
    sbom_document = _json_object(entries, sbom_name)
    sbom_metadata = sbom_document.get("metadata")
    properties = (
        sbom_metadata.get("properties") if isinstance(sbom_metadata, dict) else None
    )
    if not isinstance(properties, list) or {
        str(item.get("value"))
        for item in properties
        if isinstance(item, dict) and item.get("name") == "pyamplicol:rust-target"
    } != {rust_target}:
        raise ArtifactError("SDK SBOM target metadata is inconsistent")
    if b"file://" in sbom_bytes or b"path+file:" in sbom_bytes:
        raise ArtifactError("SDK SBOM contains a local Cargo source reference")
    identities = _validate_cyclonedx_graph(
        sbom_document,
        description="SDK SBOM",
        root_name="rusticol-capi",
        version=version,
        rust_target=rust_target,
        mode=mode,
        lock=lock,
    )
    typed_values: dict[str, list[str]] = {}
    for key in ("system_libraries", "frameworks"):
        values = link.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ArtifactError(f"SDK {key} must be a list of nonempty strings")
        if len(values) != len(set(values)):
            raise ArtifactError(f"SDK {key} contains duplicate entries")
        if any(
            "/" in value
            or "\\" in value
            or value.startswith("-")
            or any(character.isspace() for character in value)
            for value in values
        ):
            raise ArtifactError(f"SDK {key} contains a path")
        typed_values[key] = values
    unexpected_libraries = (
        set(typed_values["system_libraries"]) - _SYSTEM_LIBRARIES[rust_target]
    )
    allowed_frameworks = _MACOS_FRAMEWORKS if "apple-darwin" in rust_target else set()
    unexpected_frameworks = set(typed_values["frameworks"]) - allowed_frameworks
    if unexpected_libraries or unexpected_frameworks:
        raise ArtifactError(
            "SDK link metadata is not target-allowlisted: "
            f"libraries={sorted(unexpected_libraries)}, "
            f"frameworks={sorted(unexpected_frameworks)}"
        )
    return actual_digest, identities


def _validate_distribution_sbom(
    entries: dict[str, bytes],
    *,
    metadata_name: str,
    version: str,
    rust_target: str,
    mode: str,
    lock: dict[str, Any],
    runtime_lock: dict[str, Any],
) -> frozenset[tuple[str, str]]:
    dist_info = metadata_name.removesuffix("METADATA")
    expected = f"{dist_info}sboms/{_DISTRIBUTION_SBOM_NAME}"
    all_sboms = sorted(name for name in entries if name.endswith(".cyclonedx.json"))
    if all_sboms != sorted([expected, _SDK_SBOM_NAME]):
        raise ArtifactError(
            "wheel must contain exactly one Maturin distribution SBOM and one "
            f"SDK SBOM; found {all_sboms}"
        )
    data = entries[expected]
    if b"file://" in data or b"path+file:" in data:
        raise ArtifactError("distribution SBOM contains a local Cargo source reference")
    document = _json_object(entries, expected)
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactError("distribution SBOM has no metadata")
    properties = _component_properties(metadata)
    required_properties = {
        "pyamplicol:build-mode": [mode],
        "pyamplicol:release-lock:sha256": [
            hashlib.sha256(entries[_DEPENDENCY_LOCK_MEMBER]).hexdigest()
        ],
        "pyamplicol:python-runtime-lock:sha256": [
            hashlib.sha256(entries[_PYTHON_RUNTIME_LOCK_MEMBER]).hexdigest()
        ],
    }
    for name, values in required_properties.items():
        if properties.get(name) != values:
            raise ArtifactError(
                f"distribution SBOM metadata property {name} is inconsistent"
            )
    return _validate_distribution_cyclonedx_graph(
        document,
        version=version,
        rust_target=rust_target,
        mode=mode,
        release_lock=lock,
        runtime_lock=runtime_lock,
    )


def _validate_selftest_fixture(
    entries: dict[str, bytes], *, version: str, rust_target: str
) -> None:
    root = PurePosixPath("pyamplicol/assets/selftest")
    fixture_members = [
        PurePosixPath(name)
        for name in entries
        if PurePosixPath(name).is_relative_to(root)
    ]
    targets = {
        path.parts[len(root.parts)]
        for path in fixture_members
        if len(path.parts) > len(root.parts)
    }
    if targets != {rust_target}:
        raise ArtifactError(
            "wheel must contain only its target self-test fixture; "
            f"expected {rust_target!r}, found {sorted(targets)!r}"
        )
    prefix = (root / rust_target).as_posix()
    expected_name = f"{prefix}/expected.json"
    manifest_name = f"{prefix}/artifact/artifact.json"
    expected = _json_object(entries, expected_name)
    manifest = _json_object(entries, manifest_name)
    if (
        expected.get("schema_version") != 1
        or expected.get("target") != rust_target
        or expected.get("artifact_path") != "artifact"
    ):
        raise ArtifactError("wheel self-test expectation is invalid")
    if manifest.get("schema_version") != 3:
        raise ArtifactError("wheel self-test artifact must use schema v3")
    producer = manifest.get("producer")
    runtime = manifest.get("runtime")
    if not isinstance(producer, dict) or not isinstance(runtime, dict):
        raise ArtifactError("wheel self-test producer/runtime metadata is invalid")
    target = producer.get("target")
    if (
        producer.get("version") != version
        or runtime.get("engine_version") != version
        or not isinstance(target, dict)
        or target.get("triple") != rust_target
    ):
        raise ArtifactError("wheel self-test artifact version/target is inconsistent")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise ArtifactError("wheel self-test payload inventory is invalid")
    tagged_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("target") is not None
    ]
    if not tagged_payloads or any(
        not isinstance(payload.get("target"), dict)
        or payload["target"].get("triple") != rust_target
        or payload["target"].get("cpu_features") != []
        for payload in tagged_payloads
    ):
        raise ArtifactError("wheel self-test evaluator target metadata is invalid")
    content = dict(manifest)
    claimed_id = content.pop("artifact_id", None)
    canonical = (
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if claimed_id != hashlib.sha256(canonical).hexdigest():
        raise ArtifactError("wheel self-test artifact identity is invalid")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ArtifactError("wheel self-test artifact has no payload inventory")
    declared: set[str] = set()
    direct_symjit = 0
    artifact_prefix = f"{prefix}/artifact/"
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ArtifactError(f"wheel self-test payload {index} is invalid")
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise ArtifactError(f"wheel self-test payload {index} has no path")
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ArtifactError(f"wheel self-test payload path is unsafe: {relative}")
        if relative in declared:
            raise ArtifactError(f"wheel self-test payload is repeated: {relative}")
        declared.add(relative)
        member = f"{artifact_prefix}{relative}"
        data = entries.get(member)
        if data is None:
            raise ArtifactError(f"wheel self-test payload is missing: {relative}")
        if (
            payload.get("size_bytes") != len(data)
            or payload.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ArtifactError(
                f"wheel self-test payload digest is invalid: {relative}"
            )
        if relative.endswith(".evaluator.bin"):
            raise ArtifactError(
                "wheel self-test must not retain Symbolica fallback state"
            )
        if payload.get("media_type") == "application/vnd.symjit.application":
            direct_symjit += 1
    actual = {
        name.removeprefix(artifact_prefix)
        for name in entries
        if name.startswith(artifact_prefix) and name != manifest_name
    }
    if actual != declared:
        raise ArtifactError("wheel self-test artifact payload inventory is incomplete")
    if direct_symjit == 0:
        raise ArtifactError("wheel self-test artifact has no direct SymJIT application")


def _validate_wheel_inventory(
    entries: dict[str, bytes],
    *,
    metadata_name: str,
    wheel_name: str,
    record_name: str,
    extension_name: str,
    rust_target: str,
    mode: str,
) -> None:
    expected = set(_canonical_wheel_package_members())
    expected.add(extension_name)
    if mode == "candidate":
        expected.add("pyamplicol/_build_info.json")
    selftest_root = f"pyamplicol/assets/selftest/{rust_target}/"
    expected.update(name for name in entries if name.startswith(selftest_root))

    dist_info = metadata_name.removesuffix("METADATA")
    expected.update(
        {
            metadata_name,
            wheel_name,
            record_name,
            f"{dist_info}entry_points.txt",
            f"{dist_info}sboms/{_DISTRIBUTION_SBOM_NAME}",
        }
    )
    expected.update(
        f"{dist_info}licenses/{relative}" for relative in _required_legal_files()
    )

    repair_members: set[str] = set()
    for name in entries:
        root = PurePosixPath(name).parts[0]
        if root not in _ALLOWED_REPAIR_ROOTS:
            continue
        basename = PurePosixPath(name).name
        if not re.search(r"(?:\.dylib|\.dll|\.so(?:\.[0-9]+)*)$", basename):
            raise ArtifactError(
                f"wheel repair-library root contains a non-library member: {name}"
            )
        repair_members.add(name)

    actual = set(entries) - repair_members
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactError(
            "wheel inventory disagrees with the canonical package manifest; "
            f"missing={missing}, extra={extra}"
        )


def _host_matches(target: str) -> bool:
    machine = platform.machine().lower()
    if target == "macosx_11_0_arm64":
        return sys.platform == "darwin" and machine in {"arm64", "aarch64"}
    if target == "macosx_11_0_x86_64":
        return sys.platform == "darwin" and machine in {"x86_64", "amd64"}
    return sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}


def _command_output(command: list[str], *, cwd: Path) -> str:
    completed = run(command, cwd=cwd, capture_output=True)
    return completed.stdout + completed.stderr


def _reject_native_paths(output: str, description: str) -> None:
    lowered = output.lower()
    for marker in _FORBIDDEN_PATH_MARKERS:
        decoded = os.fsdecode(marker).lower()
        if decoded and decoded in lowered:
            raise ArtifactError(f"{description} embeds non-relocatable path {decoded}")


def _reject_native_path_bytes(data: bytes, description: str) -> None:
    lowered = data.lower()
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker and marker.lower() in lowered:
            raise ArtifactError(
                f"{description} embeds non-relocatable path "
                f"{os.fsdecode(marker).lower()}"
            )


def _scan_capi_archive(path: Path, *, allow_local_rustup: bool) -> bool:
    archive = path.read_bytes()
    lowered = archive.lower()
    found = [marker for marker in _FORBIDDEN_CAPI_SYMBOLS if marker.encode() in lowered]
    if found:
        raise ArtifactError(
            "Rusticol C API archive references Python/PyO3/NumPy symbols: "
            + ", ".join(found)
        )
    scanned = _sanitized_native_bytes(archive, allow_local_rustup=allow_local_rustup)
    _reject_native_path_bytes(scanned, "Rusticol C API archive")

    candidates: list[str] = []
    for candidate in (
        os.environ.get("LLVM_NM"),
        shutil.which("llvm-nm"),
        shutil.which("nm"),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for nm in candidates:
        completed = subprocess.run(
            [nm, "-u", str(path)],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        forbidden_symbols: list[str] = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[-2] not in {"U", "u", "W", "w", "V", "v"}:
                continue
            symbol = fields[-1]
            normalized = symbol.removeprefix("_").lower()
            if normalized.startswith(_FORBIDDEN_CAPI_UNDEFINED_PREFIXES):
                forbidden_symbols.append(symbol)
        if forbidden_symbols:
            raise ArtifactError(
                "Rusticol C API archive has undefined Python/PyO3/NumPy symbols: "
                + ", ".join(sorted(set(forbidden_symbols)))
            )
        if completed.returncode == 0:
            return True
    return False


def _native_c_compiler(rust_target: str) -> list[str]:
    configured = os.environ.get("CC")
    if configured:
        command = shlex.split(configured)
        if command:
            return command
    names = ("clang", "cc") if "apple-darwin" in rust_target else ("cc", "clang")
    for name in names:
        compiler = shutil.which(name)
        if compiler:
            return [compiler]
    raise ArtifactError("a C compiler is required for the native SDK audit")


def _validate_capi_linkage(
    entries: dict[str, bytes],
    *,
    archive: Path,
    directory: Path,
    rust_target: str,
) -> None:
    header_data = entries["pyamplicol/_sdk/include/rusticol.h"]
    exports = tuple(
        dict.fromkeys(re.findall(rb"\b(rusticol_[a-z0-9_]+)\s*\(", header_data))
    )
    decoded_exports = tuple(item.decode("ascii") for item in exports)
    if not decoded_exports or "rusticol_abi_version" not in decoded_exports:
        raise ArtifactError("wheel Rusticol C header has no complete public ABI")
    header = directory / "rusticol.h"
    source = directory / "probe.c"
    binary = directory / "probe"
    header.write_bytes(header_data)
    table = ",\n".join(f"    (rusticol_probe_fn){name}" for name in decoded_exports)
    source.write_text(
        f"""\
#include <stdint.h>
#include \"rusticol.h\"

typedef void (*rusticol_probe_fn)(void);
static rusticol_probe_fn volatile rusticol_api[] = {{
{table}
}};

int main(void) {{
    uint32_t (*abi_version)(void) = (uint32_t (*)(void))rusticol_api[0];
    return abi_version() == RUSTICOL_ABI_VERSION ? 0 : 1;
}}
""",
        encoding="utf-8",
    )
    link = _json_object(entries, "pyamplicol/_sdk/link.json")
    if link.get("target") != rust_target:
        raise ArtifactError("wheel SDK link metadata targets the wrong platform")
    flags = [f"-l{item}" for item in link.get("system_libraries", [])]
    for framework in link.get("frameworks", []):
        flags.extend(("-framework", str(framework)))
    completed = subprocess.run(
        [
            *_native_c_compiler(rust_target),
            "-std=c11",
            "-O0",
            "-I",
            str(directory),
            str(source),
            str(archive),
            *flags,
            "-o",
            str(binary),
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise ArtifactError(
            "wheel Rusticol archive failed the complete C ABI link probe" + suffix
        )
    completed = subprocess.run(
        [str(binary)],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ArtifactError("wheel Rusticol archive failed the C ABI execution probe")


def _scan_static_runtime_families(path: Path, description: str) -> None:
    lowered = path.read_bytes().lower()
    found = [
        marker
        for marker in _FORBIDDEN_STATIC_RUNTIME_MARKERS
        if marker.encode() in lowered
    ]
    if found:
        raise ArtifactError(
            f"{description} contains a forbidden static arithmetic runtime: "
            + ", ".join(found)
        )


def _allowed_runtime_path(value: str) -> bool:
    return value.startswith(
        ("$ORIGIN", "@loader_path", "@rpath", "/usr/lib/", "/System/Library/", "/lib/")
    )


def _scan_extension(path: Path, target: str) -> None:
    file_output = _command_output(["file", str(path)], cwd=path.parent).lower()
    if target.endswith("arm64") and not any(
        marker in file_output for marker in ("arm64", "aarch64")
    ):
        raise ArtifactError(
            "extension binary does not contain the required arm64 slice"
        )
    if target.endswith("x86_64") and not any(
        marker in file_output for marker in ("x86-64", "x86_64")
    ):
        raise ArtifactError("extension binary is not x86_64")

    if target.startswith("macosx"):
        libraries = _command_output(["otool", "-L", str(path)], cwd=path.parent)
        for line in libraries.splitlines()[1:]:
            dependency = line.strip().split(" ", 1)[0]
            if dependency and not _allowed_runtime_path(dependency):
                raise ArtifactError(f"extension links a non-system path: {dependency}")
            if "python" in dependency.lower():
                raise ArtifactError(f"extension links Python directly: {dependency}")
        load_commands = _command_output(["otool", "-l", str(path)], cwd=path.parent)
        lines = load_commands.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "cmd LC_RPATH":
                for candidate in lines[index + 1 : index + 5]:
                    match = re.match(r"\s*path\s+(\S+)", candidate)
                    if match and not _allowed_runtime_path(match.group(1)):
                        raise ArtifactError(
                            f"extension has a non-system RPATH: {match.group(1)}"
                        )
                        break
        inspected = (libraries + load_commands).replace(str(path), "<extension>")
        _reject_native_paths(inspected, "extension load metadata")
        return

    dynamic = _command_output(["readelf", "-d", str(path)], cwd=path.parent)
    for line in dynamic.splitlines():
        if "(RPATH)" in line or "(RUNPATH)" in line:
            match = re.search(r"\[([^]]*)\]", line)
            if match:
                for value in match.group(1).split(":"):
                    if value and not _allowed_runtime_path(value):
                        raise ArtifactError(
                            f"extension has a non-system RPATH: {value}"
                        )
        if "(NEEDED)" in line:
            match = re.search(r"\[([^]]+)\]", line)
            if match and ("/" in match.group(1) or "python" in match.group(1).lower()):
                raise ArtifactError(
                    f"extension has a forbidden dynamic dependency: {match.group(1)}"
                )
    _reject_native_paths(
        dynamic.replace(str(path), "<extension>"), "extension dynamic metadata"
    )


def _native_scan(
    entries: dict[str, bytes],
    extension: str,
    target: str,
    *,
    allow_local_rustup: bool,
) -> None:
    with external_temporary_directory("pyamplicol-native-audit-") as temporary:
        extension_path = temporary / Path(extension).name
        archive_path = temporary / "librusticol_capi.a"
        extension_path.write_bytes(entries[extension])
        archive_path.write_bytes(entries["pyamplicol/_sdk/lib/librusticol_capi.a"])
        _scan_extension(extension_path, target)
        _scan_capi_archive(archive_path, allow_local_rustup=allow_local_rustup)
        link = _json_object(entries, "pyamplicol/_sdk/link.json")
        rust_target = str(link.get("target", ""))
        _validate_capi_linkage(
            entries,
            archive=archive_path,
            directory=temporary,
            rust_target=rust_target,
        )
        _scan_static_runtime_families(extension_path, "Rusticol Python extension")
        _scan_static_runtime_families(archive_path, "Rusticol C API archive")


def audit_wheel(
    path: Path,
    *,
    mode: str,
    native_scan: bool | None = None,
) -> WheelReport:
    """Validate a wheel's metadata, content inventory, SDK, and native binaries."""

    path = path.resolve()
    if mode not in {"candidate", "release"}:
        raise ArtifactError(f"unsupported wheel audit mode: {mode}")
    size = path.stat().st_size
    if size > MAX_WHEEL_BYTES:
        raise ArtifactError(
            f"compressed wheel is {size} bytes; limit is {MAX_WHEEL_BYTES} bytes"
        )
    filename_version, python_tag, abi_tag, platform_tag = _parse_wheel_filename(path)
    if python_tag != EXPECTED_PYTHON_TAG or abi_tag != EXPECTED_ABI_TAG:
        raise ArtifactError(
            f"wheel must use {EXPECTED_PYTHON_TAG}-{EXPECTED_ABI_TAG}, found "
            f"{python_tag}-{abi_tag}"
        )
    target, rust_target = _canonical_target(platform_tag)
    entries = _wheel_entries(path)
    _validate_wheel_resource_layout(entries)
    metadata_name = _single_name(entries, ".dist-info/METADATA")
    wheel_name = _single_name(entries, ".dist-info/WHEEL")
    record_name = _single_name(entries, ".dist-info/RECORD")
    metadata = _message(entries[metadata_name], "METADATA")
    name = canonicalize_name(str(metadata.get("Name", "")))
    version = str(metadata.get("Version", ""))
    if name != EXPECTED_DISTRIBUTION:
        raise ArtifactError(f"wheel metadata Name is {name!r}, expected pyamplicol")
    if filename_version != version:
        raise ArtifactError(
            "wheel filename version disagrees with metadata: "
            f"{filename_version!r} != {version!r}"
        )
    raw_requirements = list(metadata.get_all("Requires-Dist", []))
    _reject_direct_requirements(raw_requirements)
    dependency_lock = _dependency_lock(entries)
    runtime_lock = _python_runtime_lock(entries, dependency_lock)
    _validate_runtime_requirements(
        raw_requirements,
        dependency_lock,
        require_exact=mode == "release",
    )
    _validate_legal_members(entries, metadata_name, metadata)
    if str(metadata.get("Requires-Python", "")) != ">=3.11":
        raise ArtifactError("wheel metadata must require Python >=3.11")

    candidate_info = [name for name in entries if name.endswith("/_build_info.json")]
    if mode == "candidate":
        if not re.fullmatch(r"0\.1\.0\.dev0\+candidate\.[0-9a-f]{12}", version):
            raise ArtifactError(f"candidate wheel has invalid version: {version}")
        if candidate_info != ["pyamplicol/_build_info.json"]:
            raise ArtifactError("candidate wheel must contain one _build_info.json")
        build_info = _json_object(entries, candidate_info[0])
        if (
            build_info.get("publishable") is not False
            or build_info.get("version") != version
        ):
            raise ArtifactError("candidate build marker is missing or inconsistent")
    else:
        if version != EXPECTED_RELEASE_VERSION:
            raise ArtifactError(f"release wheel has invalid version: {version}")
        if candidate_info or b"candidate" in entries[metadata_name].lower():
            raise ArtifactError("release wheel contains candidate markers")

    wheel_metadata = _message(entries[wheel_name], "WHEEL")
    if str(wheel_metadata.get("Root-Is-Purelib", "")).lower() != "false":
        raise ArtifactError("binary wheel must set Root-Is-Purelib: false")
    wheel_tags = list(wheel_metadata.get_all("Tag", []))
    if not wheel_tags:
        raise ArtifactError("WHEEL metadata contains no Tag fields")
    for tag in wheel_tags:
        fields = tag.rsplit("-", 2)
        if len(fields) != 3 or fields[0] != python_tag or fields[1] != abi_tag:
            raise ArtifactError(f"WHEEL metadata has an unexpected tag: {tag}")
        tag_target, _ = _canonical_target(fields[2])
        if tag_target != target:
            raise ArtifactError(f"WHEEL tag disagrees with filename: {tag}")

    extensions = sorted(
        name
        for name in entries
        if re.fullmatch(r"pyamplicol/_rusticol(?:\.[^.]+)*\.so", name)
    )
    if len(extensions) != 1 or "abi3" not in Path(extensions[0]).name:
        raise ArtifactError(
            "wheel must contain exactly one pyamplicol/_rusticol*.abi3.so"
        )
    sdk_digest, sdk_inventory = _validate_sdk(
        entries,
        version=version,
        rust_target=rust_target,
        mode=mode,
        lock=dependency_lock,
    )
    distribution_inventory = _validate_distribution_sbom(
        entries,
        metadata_name=metadata_name,
        version=version,
        rust_target=rust_target,
        mode=mode,
        lock=dependency_lock,
        runtime_lock=runtime_lock,
    )
    cargo_version = (
        version if mode == "release" else version.replace(".dev0+", "-dev.0+")
    )
    sdk_dependencies = sdk_inventory - {("rusticol-capi", cargo_version)}
    if not sdk_dependencies <= distribution_inventory:
        raise ArtifactError(
            "SDK SBOM dependencies are not represented by the distribution SBOM: "
            + ", ".join(
                f"{name}@{item_version}"
                for name, item_version in sorted(
                    sdk_dependencies - distribution_inventory
                )
            )
        )
    _validate_selftest_fixture(entries, version=version, rust_target=rust_target)
    _validate_wheel_inventory(
        entries,
        metadata_name=metadata_name,
        wheel_name=wheel_name,
        record_name=record_name,
        extension_name=extensions[0],
        rust_target=rust_target,
        mode=mode,
    )
    _validate_record(entries, record_name)
    _scan_embedded_paths(entries, allow_local_rustup=mode == "candidate")

    perform_native_scan = _host_matches(target) if native_scan is None else native_scan
    if perform_native_scan:
        if not _host_matches(target):
            raise ArtifactError(f"cannot perform native scan for {target} on this host")
        _native_scan(
            entries,
            extensions[0],
            target,
            allow_local_rustup=mode == "candidate",
        )
    return WheelReport(
        filename=path.name,
        version=version,
        size=size,
        sha256=sha256(path),
        python_tag=python_tag,
        abi_tag=abi_tag,
        target=target,
        rust_target=rust_target,
        sdk_archive_sha256=sdk_digest,
        native_scan=perform_native_scan,
    )


def _validate_sdist_inventory(members: set[str], *, mode: str) -> None:
    expected = set(_canonical_sdist_members())
    if mode == "candidate":
        expected.add("src/pyamplicol/_build_info.json")
    if members != expected:
        missing = sorted(expected - members)
        extra = sorted(members - expected)
        raise ArtifactError(
            "sdist inventory disagrees with the canonical source manifest; "
            f"missing={missing}, extra={extra}"
        )


def audit_sdist(path: Path, *, mode: str) -> SdistReport:
    """Validate sdist structure and candidate/release identity."""

    path = path.resolve()
    if mode not in {"candidate", "release"}:
        raise ArtifactError(f"unsupported sdist audit mode: {mode}")
    with external_temporary_directory("pyamplicol-sdist-audit-") as temporary:
        source = safe_extract_sdist(path, temporary / "source")
        relative_files = {
            item.relative_to(source).as_posix(): item
            for item in source.rglob("*")
            if item.is_file()
        }
        _validate_sdist_inventory(set(relative_files), mode=mode)
        missing = missing_required_sdist_members(relative_files)
        if missing:
            raise ArtifactError(
                "sdist is missing required files: " + ", ".join(missing)
            )
        cargo = relative_files["Cargo.toml"].read_text(encoding="utf-8")
        match = re.search(r'(?m)^version = "([^"]+)"$', cargo)
        if match is None:
            raise ArtifactError("sdist Cargo.toml has no workspace version")
        cargo_version = match.group(1)
        build_info_path = relative_files.get("src/pyamplicol/_build_info.json")
        if mode == "release":
            if cargo_version != EXPECTED_RELEASE_VERSION or build_info_path is not None:
                raise ArtifactError("release sdist contains candidate identity")
            version = EXPECTED_RELEASE_VERSION
        else:
            candidate_match = re.fullmatch(
                r"0\.1\.0-dev\.0\+candidate\.([0-9a-f]{12})", cargo_version
            )
            if candidate_match is None or build_info_path is None:
                raise ArtifactError("candidate sdist is not marked non-publishable")
            build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
            version = f"0.1.0.dev0+candidate.{candidate_match.group(1)}"
            if (
                build_info.get("publishable") is not False
                or build_info.get("version") != version
            ):
                raise ArtifactError("candidate sdist build marker is inconsistent")

        pyproject = relative_files["pyproject.toml"].read_text(encoding="utf-8")
        try:
            pyproject_data = tomllib.loads(pyproject)
        except tomllib.TOMLDecodeError as error:
            raise ArtifactError(
                f"sdist contains invalid pyproject.toml: {error}"
            ) from error
        _reject_direct_requirements(
            _project_requirements(pyproject_data),
            description="sdist project metadata",
        )
        _validate_sdist_python_runtime_lock(relative_files)
        scan_names = {
            name
            for name in relative_files
            if name
            in {
                "Cargo.lock",
                "Cargo.toml",
                "dependencies/python-runtime-lock.toml",
                "dependencies/release-lock.toml",
                "pyproject.toml",
                "src/pyamplicol/_build_info.json",
            }
            or ("MANIFEST" in Path(name).name and name.endswith(".toml"))
            or (
                name.startswith("tests/fixtures/reference/")
                and name.endswith(".json")
            )
            or name.startswith(".cargo/")
        }
        for name in sorted(scan_names):
            item = relative_files[name]
            data = item.read_bytes()
            for marker in _FORBIDDEN_PATH_MARKERS:
                if marker and marker.lower() in data.lower():
                    raise ArtifactError(
                        f"sdist member {name} embeds non-relocatable path marker "
                        f"{os.fsdecode(marker)!r}"
                    )
        expected_base = f"pyamplicol-{version}"
        if source.name != expected_base or path.name != f"{expected_base}.tar.gz":
            raise ArtifactError(
                "sdist filename/root does not match package identity: "
                f"{path.name}, {source.name}"
            )
    return SdistReport(
        filename=path.name,
        version=version,
        size=path.stat().st_size,
        sha256=sha256(path),
    )


def _normalized_record(data: bytes) -> bytes:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(sorted(rows, key=lambda row: tuple(row)))
    return output.getvalue().encode("utf-8")


def normalized_wheel_contents(path: Path) -> dict[str, bytes]:
    """Return wheel payloads independent of ZIP ordering and timestamps."""

    entries = _wheel_entries(path.resolve())
    record_name = _single_name(entries, ".dist-info/RECORD")
    entries[record_name] = _normalized_record(entries[record_name])
    return entries


def compare_wheels(source_wheel: Path, sdist_wheel: Path) -> None:
    """Require byte-identical payloads after normalizing ZIP and RECORD ordering."""

    source = normalized_wheel_contents(source_wheel)
    rebuilt = normalized_wheel_contents(sdist_wheel)
    if set(source) != set(rebuilt):
        missing = sorted(set(source) - set(rebuilt))
        extra = sorted(set(rebuilt) - set(source))
        raise ArtifactError(
            "wheel inventory differs after sdist rebuild; "
            f"missing={missing}, extra={extra}"
        )
    mismatches = [name for name in sorted(source) if source[name] != rebuilt[name]]
    if mismatches:
        detail = ", ".join(mismatches[:20])
        raise ArtifactError(
            "wheel payload differs after sdist rebuild (metadata, RECORD, package "
            f"resources, SDK, and native binaries are compared): {detail}"
        )


def artifact_record(report: WheelReport | SdistReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["kind"] = "wheel" if isinstance(report, WheelReport) else "sdist"
    return payload


def write_manifest(
    directory: Path,
    *,
    mode: str,
    wheels: list[WheelReport],
    sdists: list[SdistReport],
    parity: str,
    retained_sdist: SdistReport | None = None,
    source_commit: str | None = None,
    source_tag: str | None = None,
) -> Path:
    """Write deterministic exact-hash metadata for uploadable artifacts."""

    if mode not in {"candidate", "release"}:
        raise ArtifactError(f"unsupported manifest mode: {mode}")
    if (source_commit is None) != (source_tag is None):
        raise ArtifactError("release source commit and tag must be supplied together")
    if source_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise ArtifactError(
                "release source commit must be a full lowercase Git SHA"
            )
        if source_tag != f"v{EXPECTED_RELEASE_VERSION}":
            raise ArtifactError(
                f"release source tag must be v{EXPECTED_RELEASE_VERSION}, "
                f"found {source_tag!r}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    reports: list[WheelReport | SdistReport] = sorted(
        [*wheels, *sdists], key=lambda report: report.filename
    )
    wheel_targets = [report.target for report in wheels]
    complete_release = (
        mode == "release"
        and parity == "verified"
        and len(wheels) == len(RELEASE_TARGETS)
        and len(wheel_targets) == len(set(wheel_targets))
        and set(wheel_targets) == set(RELEASE_TARGETS)
        and len(sdists) == 1
        and retained_sdist is not None
        and retained_sdist in sdists
        and all(report.native_scan for report in wheels)
        and all(
            report.version == EXPECTED_RELEASE_VERSION
            and report.python_tag == EXPECTED_PYTHON_TAG
            and report.abi_tag == EXPECTED_ABI_TAG
            and report.rust_target == RELEASE_TARGETS.get(report.target)
            for report in wheels
        )
        and sdists[0].version == EXPECTED_RELEASE_VERSION
        and source_commit is not None
        and source_tag is not None
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "distribution": EXPECTED_DISTRIBUTION,
        "mode": mode,
        "publishable": complete_release,
        "sdist_wheel_parity": parity,
        "release_targets": sorted(wheel_targets),
        "artifacts": [artifact_record(report) for report in reports],
    }
    if retained_sdist is not None:
        payload["retained_sdist"] = {
            "filename": retained_sdist.filename,
            "sha256": retained_sdist.sha256,
            "size": retained_sdist.size,
        }
    if source_commit is not None and source_tag is not None:
        payload["source"] = {"commit": source_commit, "tag": source_tag}
    manifest = directory / "release-manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = directory / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{report.sha256}  {report.filename}\n" for report in reports),
        encoding="ascii",
    )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid release manifest {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ArtifactError(f"unsupported release manifest schema: {path}")
    return payload


def verify_manifest(
    directory: Path,
    *,
    require_release: bool,
    require_all_targets: bool,
) -> dict[str, Any]:
    """Verify that a flat bundle exactly matches its manifest and checksums."""

    directory = directory.resolve()
    manifest_path = directory / "release-manifest.json"
    checksums_path = directory / "SHA256SUMS"
    payload = _load_manifest(manifest_path)
    if payload.get("distribution") != EXPECTED_DISTRIBUTION:
        raise ArtifactError("release manifest has the wrong distribution")
    if payload.get("mode") not in {"candidate", "release"}:
        raise ArtifactError("release manifest has an unsupported mode")
    if payload.get("mode") == "candidate" and payload.get("publishable") is not False:
        raise ArtifactError("candidate manifest must be explicitly non-publishable")
    if require_release:
        if payload.get("mode") != "release" or payload.get("publishable") is not True:
            raise ArtifactError("manifest is not a complete publishable release")
        if payload.get("sdist_wheel_parity") != "verified":
            raise ArtifactError("release manifest does not prove sdist/wheel parity")
    source = payload.get("source")
    if source is not None and (
        not isinstance(source, dict)
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))) is None
        or source.get("tag") != f"v{EXPECTED_RELEASE_VERSION}"
        or set(source) != {"commit", "tag"}
    ):
        raise ArtifactError("release manifest has invalid source identity")
    if require_release and source is None:
        raise ArtifactError("publishable release manifest has no source identity")
    artifact_entries = payload.get("artifacts")
    if not isinstance(artifact_entries, list):
        raise ArtifactError("release manifest artifacts must be a list")
    expected: dict[str, str] = {}
    wheel_entries: list[dict[str, Any]] = []
    sdist_entries: list[dict[str, Any]] = []
    for entry in artifact_entries:
        if not isinstance(entry, dict):
            raise ArtifactError("release manifest artifact entries must be objects")
        filename = str(entry.get("filename", ""))
        if (
            Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or filename in expected
        ):
            raise ArtifactError(f"unsafe or duplicate manifest filename: {filename}")
        kind = entry.get("kind")
        if kind == "wheel" and filename.endswith(".whl"):
            wheel_entries.append(entry)
        elif kind == "sdist" and filename.endswith(".tar.gz"):
            sdist_entries.append(entry)
        else:
            raise ArtifactError(f"manifest artifact kind disagrees with {filename}")
        artifact = directory / filename
        if not artifact.is_file():
            raise ArtifactError(f"manifest artifact is missing: {filename}")
        digest = sha256(artifact)
        if digest != entry.get("sha256") or artifact.stat().st_size != entry.get(
            "size"
        ):
            raise ArtifactError(f"manifest hash/size mismatch: {filename}")
        expected[filename] = digest
    actual = {
        path.name
        for pattern in ("*.whl", "*.tar.gz")
        for path in directory.glob(pattern)
    }
    if actual != set(expected):
        raise ArtifactError(
            f"bundle artifact inventory differs from manifest: {sorted(actual)}"
        )
    checksum_lines = checksums_path.read_text(encoding="ascii").splitlines()
    rendered = [f"{digest}  {name}" for name, digest in sorted(expected.items())]
    if checksum_lines != rendered:
        raise ArtifactError("SHA256SUMS does not exactly match release-manifest.json")
    targets = set(map(str, payload.get("release_targets", [])))
    wheel_targets = [str(entry.get("target", "")) for entry in wheel_entries]
    if payload.get("release_targets") != sorted(wheel_targets):
        raise ArtifactError("release target inventory disagrees with wheel entries")
    if len(wheel_targets) != len(set(wheel_targets)):
        raise ArtifactError("release manifest contains duplicate platform targets")
    if require_release:
        if (
            len(wheel_entries) != len(RELEASE_TARGETS)
            or targets != set(RELEASE_TARGETS)
            or len(sdist_entries) != 1
        ):
            raise ArtifactError(
                "release bundle must contain exactly one wheel for every release "
                "target and one sdist"
            )
        if any(entry.get("native_scan") is not True for entry in wheel_entries):
            raise ArtifactError("release manifest lacks native-scan evidence")
        invalid_wheels = [
            str(entry.get("filename", ""))
            for entry in wheel_entries
            if entry.get("version") != EXPECTED_RELEASE_VERSION
            or entry.get("python_tag") != EXPECTED_PYTHON_TAG
            or entry.get("abi_tag") != EXPECTED_ABI_TAG
            or entry.get("rust_target")
            != RELEASE_TARGETS.get(str(entry.get("target", "")))
        ]
        if (
            invalid_wheels
            or sdist_entries[0].get("version") != EXPECTED_RELEASE_VERSION
        ):
            raise ArtifactError(
                "release manifest contains an artifact with inconsistent release "
                f"identity: wheels={invalid_wheels}"
            )
        retained = payload.get("retained_sdist")
        sdist_entry = sdist_entries[0]
        if not isinstance(retained, dict) or any(
            retained.get(key) != sdist_entry.get(key)
            for key in ("filename", "sha256", "size")
        ):
            raise ArtifactError("retained sdist identity disagrees with the bundle")
    if (require_release or require_all_targets) and targets != set(RELEASE_TARGETS):
        raise ArtifactError(
            "release bundle target set is incomplete: " + ", ".join(sorted(targets))
        )
    return payload


def collect_unique_artifacts(directory: Path) -> list[Path]:
    """Collect recursive artifacts, rejecting same-name hash disagreements."""

    by_name: dict[str, Path] = {}
    for pattern in ("*.whl", "*.tar.gz"):
        for path in sorted(directory.rglob(pattern)):
            existing = by_name.get(path.name)
            if existing is not None and sha256(existing) != sha256(path):
                raise ArtifactError(
                    f"conflicting artifacts share filename {path.name}: "
                    f"{existing} and {path}"
                )
            by_name.setdefault(path.name, path.resolve())
    if not by_name:
        raise ArtifactError(f"no wheels or sdist found under {directory}")
    return [by_name[name] for name in sorted(by_name)]


def verify_parity_evidence(source: Path, wheel: Path, sdist: Path) -> None:
    """Require a platform manifest proving parity against the retained sdist."""

    wheel_digest = sha256(wheel)
    sdist_digest = sha256(sdist)
    for manifest_path in source.rglob("release-manifest.json"):
        payload = _load_manifest(manifest_path)
        artifacts = payload.get("artifacts", [])
        has_wheel = any(
            isinstance(entry, dict)
            and entry.get("filename") == wheel.name
            and entry.get("sha256") == wheel_digest
            and entry.get("native_scan") is True
            for entry in artifacts
        )
        retained = payload.get("retained_sdist")
        if (
            has_wheel
            and payload.get("mode") == "release"
            and payload.get("sdist_wheel_parity") == "verified"
            and isinstance(retained, dict)
            and retained.get("filename") == sdist.name
            and retained.get("sha256") == sdist_digest
        ):
            return
    raise ArtifactError(
        f"no release manifest proves {wheel.name} was compared with {sdist.name}"
    )


def copy_artifacts(paths: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in paths:
        target = destination / source.name
        if target.exists():
            raise ArtifactError(f"bundle destination already contains {target.name}")
        shutil.copy2(source, target)
        copied.append(target)
    return copied


__all__ = [
    "MAX_WHEEL_BYTES",
    "RELEASE_TARGETS",
    "ArtifactError",
    "SdistReport",
    "WheelReport",
    "audit_sdist",
    "audit_wheel",
    "collect_unique_artifacts",
    "compare_wheels",
    "copy_artifacts",
    "normalized_wheel_contents",
    "verify_manifest",
    "verify_parity_evidence",
    "write_manifest",
]
