#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Verify the Cargo.lock legal inventory and release-source license files."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_RELATIVE = Path("licenses/RUST_THIRD_PARTY.toml")
COMPLIANCE_RELATIVE = Path("licenses/STATIC_LINK_COMPLIANCE.toml")
RELEASE_LOCK_RELATIVE = Path("dependencies/release-lock.toml")
RUST_CORPUS_RELATIVE = Path("licenses/rust")

_CORPUS_BEGIN = "# BEGIN GENERATED RUST LICENSE CORPUS"
_CORPUS_END = "# END GENERATED RUST LICENSE CORPUS"
_LEGAL_TEXT_BASENAME = re.compile(
    r"(?:licen[cs]e|unlicense|copying|copyright|notice|authors|patents)"
    r"(?:$|[._-].*)",
    re.IGNORECASE,
)
_CRATE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_SPDX_LICENSE_IDS = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-2.0-only",
    "ISC",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MIT",
    "MPL-2.0",
    "MPL-2.0-or-later",
    "Unicode-3.0",
    "Unlicense",
    "Zlib",
}
_SPDX_EXCEPTIONS = {"LLVM-exception"}
_SPDX_LICENSE_REFS = {
    "LicenseRef-Symbolica-Proprietary",
}
_TOKEN = re.compile(r"\s*(\(|\)|AND\b|OR\b|WITH\b|[A-Za-z0-9][A-Za-z0-9.-]*)")

_SPECIAL_CRATES = {
    "symbolica": (
        "LicenseRef-Symbolica-Proprietary",
        "licenses/Symbolica.txt",
    ),
    "symjit": ("MIT", "licenses/SymJIT.txt"),
}
_STATIC_ARTIFACT_ROOTS = {
    "rusticol-capi": "librusticol_capi.a",
    "rusticol-python": "pyamplicol._rusticol",
}
_STATIC_LGPL_FAMILIES = {
    "gmp": frozenset({"gmp-mpfr-sys", "rug"}),
    "malachite": frozenset(
        {
            "malachite",
            "malachite-base",
            "malachite-float",
            "malachite-nz",
            "malachite-q",
        }
    ),
}
_RELINKING_EVIDENCE_ROLES = frozenset(
    {
        "corresponding-source",
        "legal-review-record",
        "license-text",
        "rebuild-instructions",
        "relink-instructions",
        "relink-test-record",
        "relinkable-application-material",
    }
)
_REQUIRED_TEXT = {
    "THIRD_PARTY_NOTICES.md": (
        "## Symbolica",
        "express authorization",
        "## SymJIT",
        "MIT License",
        "## GammaLoop Model Assets",
        "LicenseRef-GammaLoop-No-Restrictions",
        "## Native Runtime Feature Boundary",
        "do not link Symbolica, Rug/GMP",
        "Malachite",
    ),
    "licenses/Symbolica.txt": (
        "not permitted to copy or distribute",
        "Express redistribution permission for pyAmpliCol",
        "Ben Ruijl",
    ),
    "licenses/SymJIT.txt": (
        "MIT License",
        "Copyright (c) 2026 siravan",
    ),
    "licenses/GammaLoop-model-assets.txt": (
        "LicenseRef-GammaLoop-No-Restrictions",
        "There are no usage restrictions for `GammaLoop`.",
    ),
}


@dataclass(frozen=True)
class LicenseIssue:
    code: str
    message: str
    release_only: bool = False


@dataclass(frozen=True)
class CargoArtifactSpec:
    """Exact Cargo feature selection used for one native build product."""

    cargo_root: str
    artifact: str
    package: str
    manifest: Path
    features: tuple[str, ...] = ()
    no_default_features: bool = False
    shipped: bool = True


@dataclass(frozen=True, order=True)
class CargoClosurePackage:
    """One package activated in a target-specific Cargo dependency closure."""

    name: str
    version: str
    source: str
    package_id: str


@dataclass(frozen=True)
class CargoClosure:
    """Auditable dependency evidence for one artifact and target triple."""

    spec: CargoArtifactSpec
    target: str
    command: tuple[str, ...]
    packages: tuple[CargoClosurePackage, ...]
    activated_features: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, order=True)
class CargoPackageKey:
    """Exact registry identity shared by Cargo.lock, metadata, and the corpus."""

    name: str
    version: str
    source: str


@dataclass(frozen=True)
class CorpusFallback:
    """Pinned standard text used only when a package archive has no legal file."""

    package_path: str
    source_name: str
    source_version: str
    source_path: str


@dataclass(frozen=True)
class CorpusFile:
    """One retained file and its Cargo-registry provenance."""

    package_path: str
    retained_path: str
    sha256: str
    source_kind: str
    source_name: str
    source_version: str
    source_path: str
    source_checksum: str
    data: bytes


@dataclass(frozen=True)
class CorpusPackage:
    """Complete conventional legal-file set for one locked registry package."""

    key: CargoPackageKey
    checksum: str
    fallback: bool
    files: tuple[CorpusFile, ...]


class CargoClosureError(RuntimeError):
    """Raised when Cargo cannot provide trustworthy dependency evidence."""


class RustCorpusError(RuntimeError):
    """Raised when Cargo registry data cannot produce a trustworthy corpus."""


CargoMetadataRunner = Callable[[tuple[str, ...], Path], str]
CargoFetchRunner = Callable[[tuple[str, ...], Path], None]


_CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_CORPUS_FALLBACKS: dict[CargoPackageKey, tuple[CorpusFallback, ...]] = {
    CargoPackageKey("hugepage-rs", "0.1.1", _CRATES_IO_SOURCE): (
        CorpusFallback("LICENSE-APACHE", "anyhow", "1.0.103", "LICENSE-APACHE"),
        CorpusFallback("LICENSE-MIT", "anyhow", "1.0.103", "LICENSE-MIT"),
    ),
    CargoPackageKey("spec_math", "0.1.6", _CRATES_IO_SOURCE): (
        CorpusFallback("LICENSE-APACHE", "anyhow", "1.0.103", "LICENSE-APACHE"),
        CorpusFallback("LICENSE-MIT", "anyhow", "1.0.103", "LICENSE-MIT"),
    ),
    CargoPackageKey(
        "wasmtime-jit-icache-coherence",
        "34.0.2",
        _CRATES_IO_SOURCE,
    ): (CorpusFallback("LICENSE", "target-lexicon", "0.13.5", "LICENSE"),),
}


SHIPPED_CARGO_ARTIFACTS = (
    CargoArtifactSpec(
        cargo_root="rusticol-capi",
        artifact="librusticol_capi.a",
        package="rusticol-capi",
        manifest=Path("rust/crates/rusticol-capi/Cargo.toml"),
    ),
    CargoArtifactSpec(
        cargo_root="rusticol-python",
        artifact="pyamplicol._rusticol",
        package="rusticol-python",
        manifest=Path("rust/crates/rusticol-python/Cargo.toml"),
        features=("extension-module",),
    ),
)

# This closure is deliberately not shipped. Keeping its feature selection here
# makes tests and legal review able to distinguish optional Symbolica tooling
# from the two f64-only release artifacts.
OPTIONAL_SYMBOLICA_CLOSURES = (
    CargoArtifactSpec(
        cargo_root="rusticol-core-symbolica-runtime",
        artifact="developer-only Symbolica runtime",
        package="rusticol-core",
        manifest=Path("rust/crates/rusticol-core/Cargo.toml"),
        features=("symbolica-runtime",),
        no_default_features=True,
        shipped=False,
    ),
)


def _release_issue(code: str, message: str) -> LicenseIssue:
    return LicenseIssue(code, message, release_only=True)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a TOML table")
    return payload


def _lock_registry_packages(
    lock: dict[str, Any],
) -> dict[CargoPackageKey, dict[str, Any]]:
    packages: dict[CargoPackageKey, dict[str, Any]] = {}
    for package in _table_list(lock, "package"):
        source = package.get("source")
        if not isinstance(source, str) or not source:
            continue
        key = CargoPackageKey(
            str(package.get("name", "")),
            str(package.get("version", "")),
            source,
        )
        if key in packages:
            raise RustCorpusError(f"Cargo.lock repeats registry package {key}")
        packages[key] = package
    return packages


def _cargo_home() -> Path:
    configured = os.environ.get("CARGO_HOME")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.home() / ".cargo").resolve()
    )


def _registry_archive(
    key: CargoPackageKey,
    checksum: str,
) -> Path:
    if not _SHA256.fullmatch(checksum):
        raise RustCorpusError(f"{key.name}@{key.version} has invalid lock checksum")
    if not key.source.startswith("registry+"):
        raise RustCorpusError(
            f"{key.name}@{key.version} uses unsupported source {key.source!r}"
        )
    cache = _cargo_home() / "registry" / "cache"
    filename = f"{key.name}-{key.version}.crate"
    candidates = (
        sorted(
            directory / filename
            for directory in cache.iterdir()
            if directory.is_dir() and (directory / filename).is_file()
        )
        if cache.is_dir()
        else []
    )
    matching: list[Path] = []
    observed: list[str] = []
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        observed.append(actual)
        if actual == checksum:
            matching.append(candidate)
    if not matching:
        detail = f"; observed checksums={sorted(set(observed))}" if observed else ""
        raise RustCorpusError(
            f"active Cargo home has no checksum-matching archive for "
            f"{key.name}@{key.version}{detail}"
        )
    return matching[0]


def _safe_package_path(value: str, *, context: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RustCorpusError(f"unsafe package path {value!r} in {context}")
    normalized = path.as_posix()
    if "\\" in value or normalized in {"", "."}:
        raise RustCorpusError(f"unsafe package path {value!r} in {context}")
    return normalized


def _archive_legal_files(
    archive: Path,
    key: CargoPackageKey,
) -> dict[str, bytes]:
    archive_root = f"{key.name}-{key.version}"
    try:
        package = tarfile.open(archive, mode="r:gz")  # noqa: SIM115
    except (OSError, tarfile.TarError) as exc:
        raise RustCorpusError(
            f"cannot read Cargo archive for {key.name}@{key.version}: {exc}"
        ) from exc

    with package:
        members: dict[str, tarfile.TarInfo] = {}
        for member in package.getmembers():
            raw = PurePosixPath(member.name)
            if raw.is_absolute() or ".." in raw.parts or not raw.parts:
                raise RustCorpusError(
                    f"Cargo archive for {key.name}@{key.version} has unsafe member "
                    f"{member.name!r}"
                )
            if raw.parts[0] != archive_root:
                raise RustCorpusError(
                    f"Cargo archive for {key.name}@{key.version} has member outside "
                    f"{archive_root!r}: {member.name!r}"
                )
            if len(raw.parts) == 1:
                continue
            relative = PurePosixPath(*raw.parts[1:]).as_posix()
            if relative in members:
                raise RustCorpusError(
                    f"Cargo archive for {key.name}@{key.version} repeats {relative!r}"
                )
            members[relative] = member

        selected = {
            relative
            for relative in members
            if _LEGAL_TEXT_BASENAME.fullmatch(PurePosixPath(relative).name)
        }
        for manifest_name in ("Cargo.toml.orig", "Cargo.toml"):
            manifest_member = members.get(manifest_name)
            if manifest_member is None or not manifest_member.isfile():
                continue
            manifest_stream = package.extractfile(manifest_member)
            if manifest_stream is None:
                raise RustCorpusError(
                    f"cannot extract {manifest_name} from {key.name}@{key.version}"
                )
            try:
                manifest = tomllib.loads(manifest_stream.read().decode("utf-8"))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise RustCorpusError(
                    f"cannot parse {manifest_name} from {key.name}@{key.version}: {exc}"
                ) from exc
            package_table = manifest.get("package", {})
            if isinstance(package_table, dict):
                license_file = package_table.get("license-file")
                if isinstance(license_file, str):
                    selected.add(
                        _safe_package_path(
                            license_file,
                            context=f"{key.name}@{key.version} license-file",
                        )
                    )

        casefolded: set[str] = set()
        legal_files: dict[str, bytes] = {}
        for relative in sorted(selected):
            folded = relative.casefold()
            if folded in casefolded:
                raise RustCorpusError(
                    f"{key.name}@{key.version} legal paths collide by case: "
                    f"{relative!r}"
                )
            casefolded.add(folded)
            member = members.get(relative)
            if member is None or not member.isfile():
                raise RustCorpusError(
                    f"{key.name}@{key.version} legal source is missing or not a "
                    f"regular file: {relative}"
                )
            stream = package.extractfile(member)
            if stream is None:
                raise RustCorpusError(
                    f"cannot extract {relative} from {key.name}@{key.version}"
                )
            data = stream.read()
            if not data:
                raise RustCorpusError(
                    f"{key.name}@{key.version} has empty legal source {relative}"
                )
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RustCorpusError(
                    f"{key.name}@{key.version} legal source is not UTF-8: {relative}"
                ) from exc
            legal_files[relative] = data
    return legal_files


def _corpus_retained_path(key: CargoPackageKey, package_path: str) -> str:
    component = f"{key.name}-{key.version}"
    if not _CRATE_COMPONENT.fullmatch(key.name) or not _CRATE_COMPONENT.fullmatch(
        key.version
    ):
        raise RustCorpusError(
            f"unsafe Cargo identity for retained corpus path: {key.name}@{key.version}"
        )
    relative = _safe_package_path(
        package_path,
        context=f"{key.name}@{key.version} retained corpus",
    )
    return (RUST_CORPUS_RELATIVE / component / relative).as_posix()


def _corpus_package_from_archive(
    key: CargoPackageKey,
    locked: dict[CargoPackageKey, dict[str, Any]],
) -> CorpusPackage:
    lock_entry = locked.get(key)
    if lock_entry is None:
        raise RustCorpusError(
            f"Cargo metadata package is absent from Cargo.lock: "
            f"{key.name}@{key.version} ({key.source})"
        )
    checksum = lock_entry.get("checksum")
    if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
        raise RustCorpusError(
            f"{key.name}@{key.version} has no lowercase SHA-256 in Cargo.lock"
        )
    archive = _registry_archive(key, checksum)
    try:
        direct = _archive_legal_files(archive, key)
    except (OSError, tarfile.TarError) as exc:
        raise RustCorpusError(
            f"cannot inspect Cargo archive for {key.name}@{key.version}: {exc}"
        ) from exc
    fallbacks = _CORPUS_FALLBACKS.get(key, ())
    if direct and fallbacks:
        raise RustCorpusError(
            f"stale fallback policy for {key.name}@{key.version}: "
            "the package archive now contains legal files"
        )
    if not direct and not fallbacks:
        raise RustCorpusError(
            f"{key.name}@{key.version} publishes no conventional legal text and "
            "has no reviewed locked-registry fallback"
        )

    files: list[CorpusFile] = []
    if direct:
        for package_path, data in sorted(direct.items()):
            files.append(
                CorpusFile(
                    package_path=package_path,
                    retained_path=_corpus_retained_path(key, package_path),
                    sha256=hashlib.sha256(data).hexdigest(),
                    source_kind="package-archive",
                    source_name=key.name,
                    source_version=key.version,
                    source_path=package_path,
                    source_checksum=checksum,
                    data=data,
                )
            )
    else:
        for fallback in fallbacks:
            source_key = CargoPackageKey(
                fallback.source_name,
                fallback.source_version,
                _CRATES_IO_SOURCE,
            )
            source_entry = locked.get(source_key)
            if source_entry is None:
                raise RustCorpusError(
                    f"fallback donor is absent from Cargo.lock: "
                    f"{source_key.name}@{source_key.version}"
                )
            source_checksum = source_entry.get("checksum")
            if not isinstance(source_checksum, str) or not _SHA256.fullmatch(
                source_checksum
            ):
                raise RustCorpusError(
                    f"fallback donor {source_key.name}@{source_key.version} has "
                    "no lowercase SHA-256"
                )
            source_archive = _registry_archive(source_key, source_checksum)
            try:
                source_files = _archive_legal_files(source_archive, source_key)
            except (OSError, tarfile.TarError) as exc:
                raise RustCorpusError(
                    f"cannot inspect fallback donor archive for "
                    f"{source_key.name}@{source_key.version}: {exc}"
                ) from exc
            data = source_files.get(fallback.source_path)
            if data is None:
                raise RustCorpusError(
                    f"fallback donor {source_key.name}@{source_key.version} lacks "
                    f"{fallback.source_path}"
                )
            files.append(
                CorpusFile(
                    package_path=fallback.package_path,
                    retained_path=_corpus_retained_path(
                        key,
                        fallback.package_path,
                    ),
                    sha256=hashlib.sha256(data).hexdigest(),
                    source_kind="locked-registry-fallback",
                    source_name=source_key.name,
                    source_version=source_key.version,
                    source_path=fallback.source_path,
                    source_checksum=source_checksum,
                    data=data,
                )
            )
    return CorpusPackage(
        key=key,
        checksum=checksum,
        fallback=not direct,
        files=tuple(files),
    )


def _required_corpus_keys(
    closures: tuple[CargoClosure, ...],
) -> tuple[CargoPackageKey, ...]:
    keys = {
        CargoPackageKey(package.name, package.version, package.source)
        for closure in closures
        for package in closure.packages
        if package.source
    }
    unsupported = sorted(key for key in keys if not key.source.startswith("registry+"))
    if unsupported:
        labels = ", ".join(
            f"{key.name}@{key.version} ({key.source})" for key in unsupported
        )
        raise RustCorpusError(f"unsupported release dependency sources: {labels}")
    return tuple(sorted(keys))


def _corpus_packages(
    lock: dict[str, Any],
    closures: tuple[CargoClosure, ...],
) -> tuple[CorpusPackage, ...]:
    locked = _lock_registry_packages(lock)
    packages = tuple(
        _corpus_package_from_archive(key, locked)
        for key in _required_corpus_keys(closures)
    )
    retained = [file.retained_path for package in packages for file in package.files]
    if len(retained) != len(set(retained)) or len(retained) != len(
        {path.casefold() for path in retained}
    ):
        raise RustCorpusError("retained corpus paths are not uniquely portable")
    return packages


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _corpus_file_record(file: CorpusFile) -> dict[str, str]:
    return {
        "package_path": file.package_path,
        "retained_path": file.retained_path,
        "sha256": file.sha256,
        "source_kind": file.source_kind,
        "source_name": file.source_name,
        "source_version": file.source_version,
        "source_path": file.source_path,
        "source_checksum": file.source_checksum,
    }


def _render_corpus(packages: tuple[CorpusPackage, ...]) -> str:
    lines = [
        _CORPUS_BEGIN,
        "[rust_license_corpus]",
        "schema_version = 1",
        f"root = {_toml_string(RUST_CORPUS_RELATIVE.as_posix())}",
        'selection = "union of ordinary registry packages in shipped '
        'artifact/target closures"',
        'source = "checksum-verified Cargo registry package archives"',
        'fallback_policy = "exact reviewed package identities with empty '
        'archive legal-file sets"',
    ]
    for package in packages:
        lines.extend(
            (
                "",
                "[[rust_license_corpus.package]]",
                f"name = {_toml_string(package.key.name)}",
                f"version = {_toml_string(package.key.version)}",
                f"source = {_toml_string(package.key.source)}",
                f"checksum = {_toml_string(package.checksum)}",
                f"fallback = {'true' if package.fallback else 'false'}",
            )
        )
        for file in package.files:
            lines.append("")
            lines.append("[[rust_license_corpus.package.file]]")
            for field, value in _corpus_file_record(file).items():
                lines.append(f"{field} = {_toml_string(value)}")
    lines.extend(("", _CORPUS_END, ""))
    return "\n".join(lines)


def _replace_required_release_files(text: str, required: list[str]) -> str:
    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n") == "required_release_files = ["
    ]
    if len(starts) != 1:
        raise RustCorpusError(
            "inventory must contain exactly one required_release_files block"
        )
    start = starts[0]
    try:
        end = next(
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip() == "]"
        )
    except StopIteration as exc:
        raise RustCorpusError("required_release_files block is unterminated") from exc
    replacement = ["required_release_files = [\n"]
    replacement.extend(f"  {_toml_string(path)},\n" for path in required)
    replacement.append("]\n")
    return "".join((*lines[:start], *replacement, *lines[end + 1 :]))


def _replace_corpus_section(text: str, rendered: str) -> str:
    begin = text.find(_CORPUS_BEGIN)
    end = text.find(_CORPUS_END)
    if begin == -1 and end == -1:
        return text.rstrip() + "\n\n" + rendered
    if begin == -1 or end == -1 or end < begin:
        raise RustCorpusError("inventory has malformed generated corpus markers")
    after = end + len(_CORPUS_END)
    while after < len(text) and text[after] in "\r\n":
        after += 1
    return text[:begin] + rendered + text[after:]


def refresh_rust_license_corpus(
    root: Path = ROOT,
    *,
    runner: CargoMetadataRunner | None = None,
) -> tuple[int, int]:
    """Rebuild retained legal texts from exact archives in the active Cargo home."""

    root = root.resolve()
    lock = _read_toml(root / "Cargo.lock")
    inventory_path = root / INVENTORY_RELATIVE
    inventory_text = inventory_path.read_text(encoding="utf-8")
    release_lock = _read_toml(root / RELEASE_LOCK_RELATIVE)
    targets = _release_targets(release_lock)
    if not targets:
        raise RustCorpusError("release lock declares no native target triples")
    closures = collect_cargo_closures(
        root,
        SHIPPED_CARGO_ARTIFACTS,
        targets,
        runner=runner,
    )
    packages = _corpus_packages(lock, closures)

    retained_paths = {
        file.retained_path for package in packages for file in package.files
    }
    legal_paths = {"LICENSE", "THIRD_PARTY_NOTICES.md", *retained_paths}
    licenses_root = root / "licenses"
    if licenses_root.is_dir():
        for path in licenses_root.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(root)
            if relative.is_relative_to(RUST_CORPUS_RELATIVE):
                continue
            legal_paths.add(relative.as_posix())
    required = sorted(legal_paths)
    rendered_inventory = _replace_required_release_files(inventory_text, required)
    rendered_inventory = _replace_corpus_section(
        rendered_inventory,
        _render_corpus(packages),
    )
    try:
        tomllib.loads(rendered_inventory)
    except tomllib.TOMLDecodeError as exc:
        raise RustCorpusError(f"generated inventory is invalid TOML: {exc}") from exc

    corpus_root = root / RUST_CORPUS_RELATIVE
    if corpus_root.exists():
        if corpus_root.is_symlink() or not corpus_root.is_dir():
            raise RustCorpusError(f"unsafe retained corpus root: {corpus_root}")
        shutil.rmtree(corpus_root)
    for package in packages:
        for file in package.files:
            destination = root / file.retained_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.data)
    inventory_path.write_text(rendered_inventory, encoding="utf-8")
    return len(packages), len(retained_paths)


def _table_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key!r} must be an array of TOML tables")
    return value


def _tokens(expression: str) -> list[str] | None:
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if match is None:
            return None
        tokens.append(match.group(1))
        position = match.end()
    return tokens


def _valid_spdx(expression: str) -> bool:
    tokens = _tokens(expression)
    if not tokens:
        return False
    position = 0

    def primary() -> bool:
        nonlocal position
        if position >= len(tokens):
            return False
        token = tokens[position]
        if token == "(":
            position += 1
            if not disjunction() or position >= len(tokens) or tokens[position] != ")":
                return False
            position += 1
            return True
        if token not in _SPDX_LICENSE_IDS and token not in _SPDX_LICENSE_REFS:
            return False
        position += 1
        return True

    def with_exception() -> bool:
        nonlocal position
        if not primary():
            return False
        if position < len(tokens) and tokens[position] == "WITH":
            position += 1
            if position >= len(tokens) or tokens[position] not in _SPDX_EXCEPTIONS:
                return False
            position += 1
        return True

    def conjunction() -> bool:
        nonlocal position
        if not with_exception():
            return False
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            if not with_exception():
                return False
        return True

    def disjunction() -> bool:
        nonlocal position
        if not conjunction():
            return False
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            if not conjunction():
                return False
        return True

    return disjunction() and position == len(tokens)


def _package_key(package: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(package.get("name", "")),
        str(package.get("version", "")),
        str(package.get("source", "")),
    )


def _first_party_key(package: dict[str, Any]) -> tuple[str, str]:
    return str(package.get("name", "")), str(package.get("version", ""))


def _static_family_for(name: str, license_expression: str) -> str | None:
    for family, members in _STATIC_LGPL_FAMILIES.items():
        if name in members:
            return family
    if license_expression.startswith("LGPL-"):
        return f"crate:{name}"
    return None


def _inventory_issues(
    lock: dict[str, Any], inventory: dict[str, Any]
) -> list[LicenseIssue]:
    issues: list[LicenseIssue] = []
    if inventory.get("schema_version") != 1:
        issues.append(LicenseIssue("inventory-schema", "schema_version must be 1"))
    if inventory.get("cargo_lock") != "Cargo.lock":
        issues.append(
            LicenseIssue("inventory-lock-path", "cargo_lock must be 'Cargo.lock'")
        )
    if inventory.get("cargo_lock_format") != lock.get("version"):
        issues.append(
            LicenseIssue(
                "inventory-lock-format",
                "cargo_lock_format does not match Cargo.lock",
            )
        )

    lock_packages = _table_list(lock, "package")
    third_party = [package for package in lock_packages if package.get("source")]
    first_party = [package for package in lock_packages if not package.get("source")]
    inventory_packages = _table_list(inventory, "package")
    inventory_first_party = _table_list(inventory, "first_party")

    expected_first_party = {_first_party_key(package) for package in first_party}
    actual_first_party = [
        _first_party_key(package) for package in inventory_first_party
    ]
    if len(actual_first_party) != len(set(actual_first_party)):
        issues.append(
            LicenseIssue(
                "first-party-duplicate", "first-party inventory has duplicates"
            )
        )
    if set(actual_first_party) != expected_first_party:
        missing_first_party = sorted(expected_first_party - set(actual_first_party))
        extra_first_party = sorted(set(actual_first_party) - expected_first_party)
        issues.append(
            LicenseIssue(
                "first-party-mismatch",
                "first-party package mismatch; "
                f"missing={missing_first_party}, extra={extra_first_party}",
            )
        )

    expected = {_package_key(package): package for package in third_party}
    actual_keys = [_package_key(package) for package in inventory_packages]
    if len(actual_keys) != len(set(actual_keys)):
        issues.append(
            LicenseIssue("package-duplicate", "inventory has duplicate crates")
        )
    if actual_keys != sorted(actual_keys):
        issues.append(
            LicenseIssue("package-order", "crate tables must use Cargo identity order")
        )

    actual = {_package_key(package): package for package in inventory_packages}
    missing_packages = sorted(set(expected) - set(actual))
    extra_packages = sorted(set(actual) - set(expected))
    if missing_packages:
        issues.append(
            LicenseIssue(
                "package-missing",
                "Cargo.lock crates absent from inventory: "
                + ", ".join(
                    f"{name}@{version}" for name, version, _ in missing_packages
                ),
            )
        )
    if extra_packages:
        issues.append(
            LicenseIssue(
                "package-extra",
                "inventory crates absent from Cargo.lock: "
                + ", ".join(f"{name}@{version}" for name, version, _ in extra_packages),
            )
        )

    seen_special: set[str] = set()
    for key in sorted(set(expected) & set(actual)):
        lock_package = expected[key]
        entry = actual[key]
        name, version, _ = key
        expected_checksum = lock_package.get("checksum")
        actual_checksum = entry.get("checksum")
        if actual_checksum != expected_checksum:
            issues.append(
                LicenseIssue(
                    "checksum-mismatch",
                    f"{name}@{version} checksum is {actual_checksum!r}, "
                    f"expected {expected_checksum!r}",
                )
            )
        expression = entry.get("license")
        if not isinstance(expression, str) or not _valid_spdx(expression):
            issues.append(
                LicenseIssue(
                    "license-expression",
                    f"{name}@{version} has invalid curated SPDX expression "
                    f"{expression!r}",
                )
            )
        if isinstance(expression, str):
            family = _static_family_for(name, expression)
            if family is not None and (
                entry.get("linkage_risk") != "static-lgpl"
                or entry.get("compliance_family") != family
            ):
                issues.append(
                    LicenseIssue(
                        "static-lgpl-inventory",
                        f"{name}@{version} must declare linkage_risk="
                        f"'static-lgpl' and compliance_family={family!r}",
                    )
                )
        if name in _SPECIAL_CRATES:
            seen_special.add(name)
            special_license, special_file = _SPECIAL_CRATES[name]
            if (
                expression != special_license
                or entry.get("license_file") != special_file
            ):
                issues.append(
                    LicenseIssue(
                        "special-license",
                        f"{name}@{version} must use {special_license} "
                        f"and {special_file}",
                    )
                )
    for name in sorted(set(_SPECIAL_CRATES) - seen_special):
        issues.append(
            LicenseIssue("special-crate-missing", f"Cargo.lock does not contain {name}")
        )
    return issues


def _safe_relative(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return path


def _corpus_table(inventory: dict[str, Any]) -> dict[str, Any] | None:
    table = inventory.get("rust_license_corpus")
    return table if isinstance(table, dict) else None


def _corpus_package_record(package: CorpusPackage) -> dict[str, Any]:
    return {
        "name": package.key.name,
        "version": package.key.version,
        "source": package.key.source,
        "checksum": package.checksum,
        "fallback": package.fallback,
        "file": [_corpus_file_record(file) for file in package.files],
    }


def _corpus_key_from_table(entry: dict[str, Any]) -> CargoPackageKey:
    return CargoPackageKey(
        str(entry.get("name", "")),
        str(entry.get("version", "")),
        str(entry.get("source", "")),
    )


def _rust_license_corpus_file_issues(
    root: Path,
    inventory: dict[str, Any],
) -> list[LicenseIssue]:
    issues: list[LicenseIssue] = []
    corpus = _corpus_table(inventory)
    if corpus is None:
        return [
            LicenseIssue(
                "rust-license-corpus-schema",
                "rust_license_corpus must be a TOML table",
            )
        ]
    expected_header = {
        "schema_version": 1,
        "root": RUST_CORPUS_RELATIVE.as_posix(),
        "selection": (
            "union of ordinary registry packages in shipped artifact/target closures"
        ),
        "source": "checksum-verified Cargo registry package archives",
        "fallback_policy": (
            "exact reviewed package identities with empty archive legal-file sets"
        ),
    }
    for field, expected in expected_header.items():
        if corpus.get(field) != expected:
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-schema",
                    f"rust_license_corpus.{field} must be {expected!r}",
                )
            )
    unexpected_header = sorted(set(corpus) - {*expected_header, "package"})
    if unexpected_header:
        issues.append(
            LicenseIssue(
                "rust-license-corpus-schema",
                f"rust_license_corpus has unexpected fields {unexpected_header}",
            )
        )

    raw_packages = corpus.get("package")
    if not isinstance(raw_packages, list) or not all(
        isinstance(entry, dict) for entry in raw_packages
    ):
        return [
            *issues,
            LicenseIssue(
                "rust-license-corpus-schema",
                "rust_license_corpus.package must be an array of tables",
            ),
        ]

    package_entries = list(raw_packages)
    package_keys = [_corpus_key_from_table(entry) for entry in package_entries]
    if package_keys != sorted(package_keys):
        issues.append(
            LicenseIssue(
                "rust-license-corpus-order",
                "corpus packages must use Cargo identity order",
            )
        )
    if len(package_keys) != len(set(package_keys)):
        issues.append(
            LicenseIssue(
                "rust-license-corpus-duplicate",
                "corpus contains duplicate package identities",
            )
        )

    try:
        canonical_lock = _read_toml(root / "Cargo.lock")
        locked = _lock_registry_packages(canonical_lock)
    except (
        OSError,
        ValueError,
        RustCorpusError,
        tomllib.TOMLDecodeError,
    ) as exc:
        return [
            *issues,
            LicenseIssue(
                "rust-license-corpus-lock",
                f"cannot read canonical Cargo.lock for corpus verification: {exc}",
            ),
        ]

    declared_paths: list[str] = []
    for entry, key in zip(package_entries, package_keys, strict=True):
        actual_checksum = entry.get("checksum")
        lock_entry = locked.get(key)
        if lock_entry is None:
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-package-extra",
                    f"corpus package is absent from Cargo.lock: "
                    f"{key.name}@{key.version} ({key.source})",
                )
            )
        elif actual_checksum != lock_entry.get("checksum"):
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-checksum",
                    f"{key.name}@{key.version} corpus checksum "
                    f"{actual_checksum!r} does not match Cargo.lock",
                )
            )

        raw_files = entry.get("file")
        if (
            not isinstance(raw_files, list)
            or not raw_files
            or not all(isinstance(file, dict) for file in raw_files)
        ):
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-schema",
                    f"{key.name}@{key.version} needs a nonempty file table array",
                )
            )
            raw_files = []

        package_paths = [str(file.get("package_path", "")) for file in raw_files]
        if package_paths != sorted(package_paths):
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-order",
                    f"{key.name}@{key.version} files must use package-path order",
                )
            )
        if len(package_paths) != len(set(package_paths)):
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-duplicate",
                    f"{key.name}@{key.version} repeats a package legal path",
                )
            )

        if lock_entry is not None:
            try:
                expected_package = _corpus_package_from_archive(key, locked)
            except RustCorpusError as exc:
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-source-archive",
                        f"cannot reproduce {key.name}@{key.version} corpus: {exc}",
                    )
                )
            else:
                expected_record = _corpus_package_record(expected_package)
                if entry != expected_record:
                    issues.append(
                        LicenseIssue(
                            "rust-license-corpus-source-drift",
                            f"{key.name}@{key.version} corpus records do not match "
                            "its checksum-verified Cargo source",
                        )
                    )

        for file in raw_files:
            retained_path = file.get("retained_path")
            if isinstance(retained_path, str):
                declared_paths.append(retained_path)
            relative = _safe_relative(retained_path)
            expected_prefix = (
                "licenses",
                "rust",
                f"{key.name}-{key.version}",
            )
            if relative is None or relative.parts[:3] != expected_prefix:
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-path",
                        f"unsafe or unstable retained path for "
                        f"{key.name}@{key.version}: {retained_path!r}",
                    )
                )
                continue
            expected_hash = file.get("sha256")
            if not isinstance(expected_hash, str) or not _SHA256.fullmatch(
                expected_hash
            ):
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-hash",
                        f"{retained_path} needs a lowercase SHA-256",
                    )
                )
            path = root / relative
            if path.is_symlink() or not path.is_file():
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-file",
                        f"retained legal text is missing or a symlink: {retained_path}",
                    )
                )
                continue
            data = path.read_bytes()
            if not data:
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-file",
                        f"retained legal text is empty: {retained_path}",
                    )
                )
            actual_hash = hashlib.sha256(data).hexdigest()
            if isinstance(expected_hash, str) and actual_hash != expected_hash:
                issues.append(
                    LicenseIssue(
                        "rust-license-corpus-hash",
                        f"{retained_path} hash is {actual_hash}, "
                        f"expected {expected_hash}",
                    )
                )

    if len(declared_paths) != len(set(declared_paths)) or len(declared_paths) != len(
        {path.casefold() for path in declared_paths}
    ):
        issues.append(
            LicenseIssue(
                "rust-license-corpus-duplicate",
                "retained corpus paths must be uniquely portable",
            )
        )

    corpus_root = root / RUST_CORPUS_RELATIVE
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        issues.append(
            LicenseIssue(
                "rust-license-corpus-root",
                f"retained corpus root is missing or a symlink: {RUST_CORPUS_RELATIVE}",
            )
        )
        discovered_paths: set[str] = set()
    else:
        discovered_paths = {
            path.relative_to(root).as_posix()
            for path in corpus_root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
    declared_set = set(declared_paths)
    if discovered_paths != declared_set:
        missing = sorted(declared_set - discovered_paths)
        extra = sorted(discovered_paths - declared_set)
        issues.append(
            LicenseIssue(
                "rust-license-corpus-file-set",
                f"retained corpus file mismatch; missing={missing}, extra={extra}",
            )
        )

    configured = inventory.get("required_release_files", [])
    configured_corpus = {
        path
        for path in configured
        if isinstance(path, str) and path.startswith("licenses/rust/")
    }
    if configured_corpus != declared_set:
        issues.append(
            LicenseIssue(
                "rust-license-corpus-release-files",
                "required_release_files does not exactly list the retained corpus",
            )
        )
    return issues


def _rust_license_corpus_closure_issues(
    inventory: dict[str, Any],
    closures: tuple[CargoClosure, ...],
) -> list[LicenseIssue]:
    corpus = _corpus_table(inventory)
    if corpus is None:
        return [
            LicenseIssue(
                "rust-license-corpus-schema",
                "cannot verify release closures without rust_license_corpus",
            )
        ]
    raw_packages = corpus.get("package")
    if not isinstance(raw_packages, list) or not all(
        isinstance(entry, dict) for entry in raw_packages
    ):
        return [
            LicenseIssue(
                "rust-license-corpus-schema",
                "cannot verify release closures without corpus package tables",
            )
        ]
    try:
        expected = set(_required_corpus_keys(closures))
    except RustCorpusError as exc:
        return [LicenseIssue("rust-license-corpus-closure", str(exc))]
    actual = {_corpus_key_from_table(entry) for entry in raw_packages}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    issues: list[LicenseIssue] = []
    if missing:
        issues.append(
            LicenseIssue(
                "rust-license-corpus-package-missing",
                "release Cargo closures lack retained legal text for: "
                + ", ".join(f"{key.name}@{key.version}" for key in missing),
            )
        )
    if extra:
        issues.append(
            LicenseIssue(
                "rust-license-corpus-package-extra",
                "retained corpus has packages outside release Cargo closures: "
                + ", ".join(f"{key.name}@{key.version}" for key in extra),
            )
        )
    return issues


def _legal_file_issues(root: Path, inventory: dict[str, Any]) -> list[LicenseIssue]:
    issues = _rust_license_corpus_file_issues(root, inventory)
    configured = inventory.get("required_release_files")
    if not isinstance(configured, list) or not all(
        isinstance(value, str) for value in configured
    ):
        return [
            *issues,
            LicenseIssue(
                "legal-file-inventory",
                "required_release_files must be an array of paths",
            ),
        ]
    configured_paths = list(configured)
    if configured_paths != sorted(configured_paths) or len(configured_paths) != len(
        set(configured_paths)
    ):
        issues.append(
            LicenseIssue(
                "legal-file-order",
                "required_release_files must be unique and sorted",
            )
        )

    licenses_root = root / "licenses"
    discovered = ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    if licenses_root.is_dir():
        discovered.extend(
            path.relative_to(root).as_posix()
            for path in licenses_root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
    discovered = sorted(discovered)
    if configured_paths != discovered:
        issues.append(
            LicenseIssue(
                "legal-file-inventory",
                f"legal file mismatch; configured={configured_paths}, "
                f"discovered={discovered}",
            )
        )

    for raw_path in configured_paths:
        relative = _safe_relative(raw_path)
        if relative is None:
            issues.append(
                LicenseIssue("legal-file-path", f"unsafe legal file path: {raw_path!r}")
            )
            continue
        path = root / relative
        if path.is_symlink():
            issues.append(
                LicenseIssue(
                    "legal-file-symlink", f"legal file is a symlink: {raw_path}"
                )
            )
        elif not path.is_file():
            issues.append(
                LicenseIssue("legal-file-missing", f"missing legal file: {raw_path}")
            )
        elif not path.read_bytes():
            issues.append(
                LicenseIssue("legal-file-empty", f"empty legal file: {raw_path}")
            )

    package_tables = inventory.get("package", [])
    required_set = set(configured_paths)
    if isinstance(package_tables, list):
        for entry in package_tables:
            if not isinstance(entry, dict) or "license_file" not in entry:
                continue
            license_file = entry["license_file"]
            if license_file not in required_set:
                issues.append(
                    LicenseIssue(
                        "crate-license-file",
                        f"{entry.get('name')} references unlisted {license_file!r}",
                    )
                )
    return issues


def _packaging_issues(root: Path, inventory: dict[str, Any]) -> list[LicenseIssue]:
    issues: list[LicenseIssue] = []
    pyproject = _read_toml(root / "pyproject.toml")
    configured = inventory.get("required_release_files", [])
    legal_files = [str(value) for value in configured if isinstance(value, str)]

    project = pyproject.get("project", {})
    pep_patterns = project.get("license-files", []) if isinstance(project, dict) else []
    if not isinstance(pep_patterns, list):
        pep_patterns = []

    tool = pyproject.get("tool", {})
    maturin = tool.get("maturin", {}) if isinstance(tool, dict) else {}
    includes = maturin.get("include", []) if isinstance(maturin, dict) else []
    sdist_patterns: list[str] = []
    if isinstance(includes, list):
        for entry in includes:
            if isinstance(entry, str):
                sdist_patterns.append(entry)
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            formats = entry.get("format", [])
            if isinstance(formats, list) and "sdist" in formats:
                sdist_patterns.append(entry["path"])

    for path in legal_files:
        if not any(
            isinstance(pattern, str) and _packaging_pattern_matches(path, pattern)
            for pattern in pep_patterns
        ):
            issues.append(
                LicenseIssue(
                    "pep639-license-file",
                    f"{path} is not covered by project.license-files",
                )
            )
        if not any(
            _packaging_pattern_matches(path, pattern) for pattern in sdist_patterns
        ):
            issues.append(
                LicenseIssue(
                    "sdist-license-file",
                    f"{path} is not covered by a Maturin sdist include",
                )
            )
    return issues


def _packaging_pattern_matches(path: str, pattern: str) -> bool:
    """Match Maturin/PEP 639 globs where ``**/`` may match no directory."""

    candidate = pattern
    while True:
        if fnmatch.fnmatchcase(path, candidate):
            return True
        if "**/" not in candidate:
            return False
        candidate = candidate.replace("**/", "", 1)


def _notice_issues(root: Path) -> list[LicenseIssue]:
    issues: list[LicenseIssue] = []
    for relative, markers in _REQUIRED_TEXT.items():
        path = root / relative
        if not path.is_file():
            issues.append(
                LicenseIssue(
                    "notice-missing", f"required notice is missing: {relative}"
                )
            )
            continue
        text = path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        for marker in markers:
            if " ".join(marker.split()) not in normalized_text:
                issues.append(
                    LicenseIssue(
                        "notice-content",
                        f"{relative} is missing required text {marker!r}",
                    )
                )
    return issues


def cargo_metadata_command(
    root: Path,
    spec: CargoArtifactSpec,
    target: str,
) -> tuple[str, ...]:
    """Build the non-mutating Cargo query for an artifact's exact feature set."""

    if (
        spec.manifest.is_absolute()
        or ".." in spec.manifest.parts
        or not spec.manifest.parts
    ):
        raise CargoClosureError(f"unsafe Cargo manifest path: {spec.manifest}")
    if not target:
        raise CargoClosureError("Cargo closure target must be nonempty")
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
        "--manifest-path",
        str((root / spec.manifest).resolve()),
        "--filter-platform",
        target,
    ]
    if spec.no_default_features:
        command.append("--no-default-features")
    if spec.features:
        command.extend(("--features", ",".join(spec.features)))
    return tuple(command)


def cargo_fetch_commands(root: Path) -> tuple[tuple[str, ...], ...]:
    """Build locked fetch commands for every declared native release target.

    The legal gate intentionally resolves Cargo metadata with ``--offline``.
    CI primes a fresh, job-local Cargo home first with these target-specific
    commands, so the gate neither inherits an ambient cache nor reaches the
    network while deciding release readiness.
    """

    root = root.resolve()
    release_lock = _read_toml(root / RELEASE_LOCK_RELATIVE)
    targets = _release_targets(release_lock)
    if not targets:
        raise CargoClosureError("release lock declares no native target triples")
    manifest = (root / "Cargo.toml").resolve()
    if not manifest.is_file():
        raise CargoClosureError(f"Cargo workspace manifest is missing: {manifest}")
    return tuple(
        (
            "cargo",
            "fetch",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--target",
            target,
        )
        for target in targets
    )


def _run_cargo_fetch(command: tuple[str, ...], root: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CargoClosureError(
            f"Cargo fetch exited with {completed.returncode}: {detail}"
        )


def fetch_locked_release_targets(
    root: Path = ROOT,
    *,
    runner: CargoFetchRunner | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Populate the active Cargo home for all locked release targets."""

    root = root.resolve()
    commands = cargo_fetch_commands(root)
    selected_runner = runner or _run_cargo_fetch
    for command in commands:
        selected_runner(command, root)
    return commands


def _run_cargo_metadata(command: tuple[str, ...], root: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CargoClosureError(
            f"Cargo metadata exited with {completed.returncode}: {detail}"
        )
    return completed.stdout


def cargo_dependency_closure(
    root: Path,
    spec: CargoArtifactSpec,
    target: str,
    *,
    runner: CargoMetadataRunner | None = None,
) -> CargoClosure:
    """Resolve activated normal/build dependencies for one native artifact.

    Cargo performs feature and target resolution. The local traversal excludes
    dev-only edges while retaining build dependencies because their code is
    part of the shipped build product's static supply chain.
    """

    root = root.resolve()
    command = cargo_metadata_command(root, spec, target)
    output = (runner or _run_cargo_metadata)(command, root)
    try:
        metadata = json.loads(output)
    except json.JSONDecodeError as exc:
        raise CargoClosureError(f"Cargo metadata returned invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CargoClosureError("Cargo metadata root must be an object")
    raw_packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(raw_packages, list) or not isinstance(resolve, dict):
        raise CargoClosureError("Cargo metadata has no package/resolve graph")
    raw_nodes = resolve.get("nodes")
    if not isinstance(raw_nodes, list):
        raise CargoClosureError("Cargo metadata resolve graph has no nodes")

    packages: dict[str, dict[str, Any]] = {}
    for package in raw_packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise CargoClosureError("Cargo metadata contains an invalid package")
        packages[package["id"]] = package
    nodes: dict[str, dict[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise CargoClosureError("Cargo metadata contains an invalid resolve node")
        nodes[node["id"]] = node

    expected_manifest = (root / spec.manifest).resolve()
    declared_root = resolve.get("root")
    root_id: str | None = None
    if isinstance(declared_root, str):
        package = packages.get(declared_root)
        if package is not None and package.get("name") == spec.package:
            root_id = declared_root
    if root_id is None:
        matches = [
            package_id
            for package_id, package in packages.items()
            if package.get("name") == spec.package
            and isinstance(package.get("manifest_path"), str)
            and Path(package["manifest_path"]).resolve() == expected_manifest
        ]
        if len(matches) != 1:
            raise CargoClosureError(
                f"expected one metadata root for {spec.package!r}, found {matches}"
            )
        root_id = matches[0]

    stack = [root_id]
    visited: set[str] = set()
    while stack:
        package_id = stack.pop()
        if package_id in visited:
            continue
        if package_id not in packages or package_id not in nodes:
            raise CargoClosureError(
                f"resolve graph references unknown package {package_id!r}"
            )
        visited.add(package_id)
        dependencies = nodes[package_id].get("deps", [])
        if not isinstance(dependencies, list):
            raise CargoClosureError(
                f"resolve node {package_id!r} has invalid dependencies"
            )
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(
                dependency.get("pkg"), str
            ):
                raise CargoClosureError(
                    f"resolve node {package_id!r} has an invalid dependency edge"
                )
            dependency_kinds = dependency.get("dep_kinds", [])
            if not isinstance(dependency_kinds, list):
                raise CargoClosureError(
                    f"resolve node {package_id!r} has invalid dependency kinds"
                )
            active_for_binary = not dependency_kinds or any(
                isinstance(kind, dict) and kind.get("kind") in (None, "normal", "build")
                for kind in dependency_kinds
            )
            if active_for_binary:
                stack.append(dependency["pkg"])

    closure_packages: list[CargoClosurePackage] = []
    activated_features: list[tuple[str, tuple[str, ...]]] = []
    for package_id in visited:
        package = packages[package_id]
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if not isinstance(name, str) or not isinstance(version, str):
            raise CargoClosureError(f"package {package_id!r} lacks name/version")
        closure_packages.append(
            CargoClosurePackage(
                name=name,
                version=version,
                source=source if isinstance(source, str) else "",
                package_id=package_id,
            )
        )
        features = nodes[package_id].get("features", [])
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise CargoClosureError(f"resolve node {package_id!r} has invalid features")
        activated_features.append((package_id, tuple(sorted(features))))

    return CargoClosure(
        spec=spec,
        target=target,
        command=command,
        packages=tuple(sorted(closure_packages)),
        activated_features=tuple(sorted(activated_features)),
    )


def collect_cargo_closures(
    root: Path,
    specs: tuple[CargoArtifactSpec, ...],
    targets: tuple[str, ...],
    *,
    runner: CargoMetadataRunner | None = None,
) -> tuple[CargoClosure, ...]:
    """Collect deterministic per-artifact/per-target dependency evidence."""

    return tuple(
        cargo_dependency_closure(root, spec, target, runner=runner)
        for spec in specs
        for target in targets
    )


def sensitive_dependency_families(
    closure: CargoClosure,
) -> dict[str, tuple[str, ...]]:
    """Identify Symbolica and known static-LGPL families in a closure."""

    found: dict[str, set[str]] = {}
    for package in closure.packages:
        family: str | None = None
        if package.name == "symbolica":
            family = "symbolica"
        elif package.name in _STATIC_LGPL_FAMILIES["gmp"]:
            family = "gmp"
        elif package.name.startswith("malachite"):
            family = "malachite"
        if family is not None:
            found.setdefault(family, set()).add(f"{package.name}@{package.version}")
    return {
        family: tuple(sorted(packages)) for family, packages in sorted(found.items())
    }


def _inventory_license_expression(
    package: CargoClosurePackage,
    inventory: dict[str, Any],
) -> str:
    candidates = [
        entry
        for entry in _table_list(inventory, "package")
        if entry.get("name") == package.name and entry.get("version") == package.version
    ]
    exact = [entry for entry in candidates if entry.get("source") == package.source]
    selected = exact if exact else candidates
    if len(selected) != 1:
        return ""
    expression = selected[0].get("license")
    return expression if isinstance(expression, str) else ""


def _static_lgpl_graph(
    closures: tuple[CargoClosure, ...],
    inventory: dict[str, Any],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Aggregate target-specific LGPL evidence by shipped Cargo root."""

    found_by_root: dict[str, dict[str, set[str]]] = {
        spec.cargo_root: {} for spec in SHIPPED_CARGO_ARTIFACTS
    }
    for closure in closures:
        found = found_by_root.setdefault(closure.spec.cargo_root, {})
        sensitive = sensitive_dependency_families(closure)
        for family in ("gmp", "malachite"):
            for package_label in sensitive.get(family, ()):  # known-family policy
                found.setdefault(family, set()).add(package_label)
        for closure_package in closure.packages:
            expression = _inventory_license_expression(closure_package, inventory)
            inventory_family = _static_family_for(closure_package.name, expression)
            if inventory_family is not None:
                found.setdefault(inventory_family, set()).add(
                    f"{closure_package.name}@{closure_package.version}"
                )
    return {
        cargo_root: {
            family: tuple(sorted(packages))
            for family, packages in sorted(families.items())
        }
        for cargo_root, families in sorted(found_by_root.items())
    }


def _shipped_closure_issues(
    closures: tuple[CargoClosure, ...],
) -> list[LicenseIssue]:
    issues: list[LicenseIssue] = []
    for closure in closures:
        if not closure.spec.shipped:
            continue
        forbidden = sensitive_dependency_families(closure)
        if forbidden:
            detail = ", ".join(
                f"{family}=[{', '.join(packages)}]"
                for family, packages in forbidden.items()
            )
            issues.append(
                _release_issue(
                    "static-native-forbidden-dependency",
                    f"{closure.spec.artifact} ({closure.spec.package}, "
                    f"target {closure.target}) must be Symbolica-free for direct "
                    f"SymJIT f64 evaluation but reaches {detail}",
                )
            )
    return issues


def _release_targets(release_lock: dict[str, Any]) -> tuple[str, ...]:
    targets = _table_list(release_lock, "targets")
    triples = [str(target.get("triple", "")) for target in targets]
    return tuple(sorted(triple for triple in triples if triple))


def _evidence_issues(
    root: Path,
    evidence_tables: list[dict[str, Any]],
) -> tuple[list[LicenseIssue], dict[str, dict[str, Any]]]:
    issues: list[LicenseIssue] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for evidence in evidence_tables:
        evidence_id = evidence.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-schema",
                    "evidence entries require a nonempty id",
                )
            )
            continue
        if evidence_id in evidence_by_id:
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-schema",
                    f"duplicate compliance evidence id {evidence_id!r}",
                )
            )
            continue
        evidence_by_id[evidence_id] = evidence
        role = evidence.get("role")
        if role not in _RELINKING_EVIDENCE_ROLES:
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-schema",
                    f"evidence {evidence_id!r} has unsupported role {role!r}",
                )
            )
        relative = _safe_relative(evidence.get("path"))
        if relative is None or relative.parts[:2] != ("licenses", "compliance"):
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-path",
                    f"evidence {evidence_id!r} must use a safe "
                    "licenses/compliance path",
                )
            )
            continue
        path = root / relative
        expected_hash = evidence.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-hash",
                    f"evidence {evidence_id!r} needs a lowercase SHA-256",
                )
            )
        if path.is_symlink() or not path.is_file() or not path.read_bytes():
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-file",
                    f"evidence {evidence_id!r} is missing, empty, or a symlink: "
                    f"{relative}",
                )
            )
        elif isinstance(expected_hash, str):
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                issues.append(
                    _release_issue(
                        "static-lgpl-evidence-hash",
                        f"evidence {evidence_id!r} hash is {actual_hash}, "
                        f"expected {expected_hash}",
                    )
                )
    return issues, evidence_by_id


def _static_link_compliance_issues(
    root: Path,
    lock: dict[str, Any],
    inventory: dict[str, Any],
    *,
    runner: CargoMetadataRunner | None = None,
    verify_corpus: bool = True,
) -> list[LicenseIssue]:
    # Candidate checks still pass their projected lock through this API. Cargo,
    # rather than lockfile package-name edges, now determines actual features
    # and target-specific reachability for the manifests in the build tree.
    del lock
    issues: list[LicenseIssue] = []
    try:
        release_lock = _read_toml(root / RELEASE_LOCK_RELATIVE)
        targets = _release_targets(release_lock)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        issues = [
            _release_issue(
                "static-native-closure-read",
                f"cannot determine native release targets: {exc}",
            )
        ]
        if verify_corpus:
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-closure",
                    f"cannot determine corpus release targets: {exc}",
                )
            )
        return issues
    if not targets:
        issues = [
            _release_issue(
                "static-native-closure-targets",
                "release lock declares no native target triples",
            )
        ]
        if verify_corpus:
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-closure",
                    "release lock declares no corpus target triples",
                )
            )
        return issues
    try:
        closures = collect_cargo_closures(
            root,
            SHIPPED_CARGO_ARTIFACTS,
            targets,
            runner=runner,
        )
    except CargoClosureError as exc:
        issues = [
            _release_issue(
                "static-native-closure",
                f"cannot resolve shipped native dependency closure: {exc}",
            )
        ]
        if verify_corpus:
            issues.append(
                LicenseIssue(
                    "rust-license-corpus-closure",
                    f"cannot resolve corpus dependency closure: {exc}",
                )
            )
        return issues

    if verify_corpus:
        issues.extend(_rust_license_corpus_closure_issues(inventory, closures))
    issues.extend(_shipped_closure_issues(closures))
    graph = _static_lgpl_graph(closures, inventory)
    detected = {
        (artifact_root, family)
        for artifact_root, families in graph.items()
        for family in families
    }

    try:
        compliance = _read_toml(root / COMPLIANCE_RELATIVE)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return [
            *issues,
            _release_issue(
                "static-lgpl-policy-read",
                f"cannot read static-link compliance policy: {exc}",
            ),
        ]

    if compliance.get("schema_version") != 1:
        issues.append(
            _release_issue(
                "static-lgpl-policy-schema", "compliance schema_version must be 1"
            )
        )
    if compliance.get("candidate_builds_allowed") is not True:
        issues.append(
            _release_issue(
                "static-lgpl-candidate-policy",
                "compliance policy must explicitly allow non-publishable "
                "candidate builds",
            )
        )

    review = inventory.get("review", {})
    if not isinstance(review, dict) or review.get("static_link_policy") != str(
        COMPLIANCE_RELATIVE
    ):
        issues.append(
            _release_issue(
                "static-lgpl-policy-reference",
                f"Rust inventory must reference {COMPLIANCE_RELATIVE}",
            )
        )

    legal_status = release_lock.get("legal_status", {})
    if not isinstance(legal_status, dict):
        legal_status = {}
    expected_status = compliance.get("status")
    expected_release_ready = compliance.get("release_ready")
    expected_lock_fields = {
        "candidate_builds_allowed": True,
        "static_lgpl_compliance_manifest": str(COMPLIANCE_RELATIVE),
        "static_lgpl_release_ready": expected_release_ready,
        "static_lgpl_status": expected_status,
    }
    for field, expected in expected_lock_fields.items():
        if legal_status.get(field) != expected:
            issues.append(
                _release_issue(
                    "static-lgpl-release-lock",
                    f"release-lock legal_status.{field} must be {expected!r}",
                )
            )

    try:
        coverage_tables = _table_list(compliance, "coverage")
        evidence_tables = _table_list(compliance, "evidence")
    except ValueError as exc:
        return [
            *issues,
            _release_issue("static-lgpl-policy-schema", str(exc)),
        ]

    evidence_issues, evidence_by_id = _evidence_issues(root, evidence_tables)
    issues.extend(evidence_issues)
    coverage_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for coverage_entry in coverage_tables:
        key = (
            str(coverage_entry.get("cargo_root", "")),
            str(coverage_entry.get("family", "")),
        )
        if key in coverage_by_key:
            issues.append(
                _release_issue(
                    "static-lgpl-coverage-schema",
                    f"duplicate compliance coverage for {key[0]}/{key[1]}",
                )
            )
        coverage_by_key[key] = coverage_entry

    complete_claim = (
        compliance.get("status") == "complete"
        and compliance.get("release_ready") is True
    )
    if not detected:
        if not complete_claim:
            issues.append(
                _release_issue(
                    "static-native-binary-audit-pending",
                    "Cargo feature closures contain no Symbolica, Rug/GMP, or "
                    "Malachite code in shipped artifacts, but release binaries and "
                    "archives still require target-by-target dependency/symbol scans "
                    "before legal clearance",
                )
            )
        return issues

    actual_coverage = set(coverage_by_key)
    if actual_coverage != detected:
        missing = sorted(detected - actual_coverage)
        extra = sorted(actual_coverage - detected)
        issues.append(
            _release_issue(
                "static-lgpl-coverage",
                f"compliance coverage mismatch; missing={missing}, extra={extra}",
            )
        )

    if not complete_claim:
        summary = "; ".join(
            f"{root_name}: "
            + ", ".join(
                f"{family}=[{', '.join(crates)}]"
                for family, crates in graph[root_name].items()
            )
            for root_name in sorted(graph)
            if graph[root_name]
        )
        issues.append(
            _release_issue(
                "static-lgpl-compliance-incomplete",
                "release readiness is blocked for statically reachable LGPL code; "
                + summary,
            )
        )
        return issues

    for key in sorted(detected):
        selected_coverage = coverage_by_key.get(key)
        if selected_coverage is None:
            continue
        artifact_root, family = key
        if selected_coverage.get("artifact") != _STATIC_ARTIFACT_ROOTS[artifact_root]:
            issues.append(
                _release_issue(
                    "static-lgpl-coverage-schema",
                    f"{artifact_root}/{family} names the wrong shipped artifact",
                )
            )
        if selected_coverage.get("status") != "complete" or selected_coverage.get(
            "strategy"
        ) != ("relinking-source-kit"):
            issues.append(
                _release_issue(
                    "static-lgpl-coverage-incomplete",
                    f"{artifact_root}/{family} needs a complete relinking-source-kit",
                )
            )
        coverage_targets = selected_coverage.get("targets")
        if (
            not isinstance(coverage_targets, list)
            or tuple(sorted(str(target) for target in coverage_targets)) != targets
        ):
            issues.append(
                _release_issue(
                    "static-lgpl-target-coverage",
                    f"{artifact_root}/{family} must cover release targets "
                    f"{list(targets)}",
                )
            )
        source_offer = selected_coverage.get("source_offer_url")
        if not isinstance(source_offer, str) or not source_offer.startswith("https://"):
            issues.append(
                _release_issue(
                    "static-lgpl-source-offer",
                    f"{artifact_root}/{family} needs a persistent HTTPS source offer",
                )
            )
        evidence_ids = selected_coverage.get("evidence")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(evidence_id, str) for evidence_id in evidence_ids
        ):
            issues.append(
                _release_issue(
                    "static-lgpl-coverage-schema",
                    f"{artifact_root}/{family} evidence must be a list of IDs",
                )
            )
            continue
        selected = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]
        roles = {str(evidence.get("role", "")) for evidence in selected}
        missing_roles = sorted(_RELINKING_EVIDENCE_ROLES - roles)
        if missing_roles:
            issues.append(
                _release_issue(
                    "static-lgpl-evidence-incomplete",
                    f"{artifact_root}/{family} lacks evidence roles {missing_roles}",
                )
            )
        for role in sorted(_RELINKING_EVIDENCE_ROLES):
            covered_targets: set[str] = set()
            for evidence in selected:
                if evidence.get("role") != role:
                    continue
                evidence_targets = evidence.get("targets", [])
                if isinstance(evidence_targets, list):
                    covered_targets.update(str(target) for target in evidence_targets)
            if set(targets) - covered_targets:
                issues.append(
                    _release_issue(
                        "static-lgpl-evidence-targets",
                        f"{artifact_root}/{family} role {role!r} does not cover "
                        f"all release targets",
                    )
                )
    return issues


def blocking_issues(issues: list[LicenseIssue], *, mode: str) -> list[LicenseIssue]:
    if mode == "release":
        return list(issues)
    if mode == "candidate":
        return [issue for issue in issues if not issue.release_only]
    raise ValueError(f"unsupported legal-check mode {mode!r}")


def check_repository(root: Path = ROOT) -> list[LicenseIssue]:
    try:
        lock = _read_toml(root / "Cargo.lock")
        inventory = _read_toml(root / INVENTORY_RELATIVE)
        issues = [
            *_inventory_issues(lock, inventory),
            *_legal_file_issues(root, inventory),
            *_packaging_issues(root, inventory),
            *_notice_issues(root),
            *_static_link_compliance_issues(root, lock, inventory),
        ]
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        issues = [LicenseIssue("inventory-read", str(exc))]
    return sorted(issues, key=lambda issue: (issue.code, issue.message))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (defaults to the script's checkout)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--mode",
        choices=("candidate", "release"),
        default="release",
        help=("candidate permits release-only legal findings; release is strict"),
    )
    parser.add_argument(
        "--fetch-locked-release-targets",
        action="store_true",
        help=(
            "populate the active Cargo home for every target in the release lock "
            "and exit; the subsequent legal check remains offline"
        ),
    )
    parser.add_argument(
        "--refresh-license-corpus",
        action="store_true",
        help=(
            "replace licenses/rust and its generated inventory section from "
            "checksum-verified archives in the active Cargo home"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fetch_locked_release_targets and args.refresh_license_corpus:
        raise SystemExit(
            "--fetch-locked-release-targets and --refresh-license-corpus "
            "must be separate steps"
        )
    if args.fetch_locked_release_targets:
        if args.json:
            raise SystemExit(
                "--json cannot be combined with --fetch-locked-release-targets"
            )
        try:
            commands = fetch_locked_release_targets(args.root.resolve())
        except (CargoClosureError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            print(f"cannot prepare Cargo legal cache: {exc}", file=sys.stderr)
            return 1
        print(f"Fetched locked Cargo inputs for {len(commands)} release targets")
        return 0
    if args.refresh_license_corpus:
        if args.json:
            raise SystemExit("--json cannot be combined with --refresh-license-corpus")
        try:
            package_count, file_count = refresh_rust_license_corpus(args.root.resolve())
        except (
            CargoClosureError,
            OSError,
            RustCorpusError,
            ValueError,
            tomllib.TOMLDecodeError,
        ) as exc:
            print(f"cannot refresh Rust license corpus: {exc}", file=sys.stderr)
            return 1
        print(
            f"Retained {file_count} legal texts for {package_count} "
            "release Rust packages"
        )
        return 0
    issues = check_repository(args.root.resolve())
    candidate_blockers = blocking_issues(issues, mode="candidate")
    release_blockers = blocking_issues(issues, mode="release")
    requested_blockers = blocking_issues(issues, mode=args.mode)
    if args.json:
        print(
            json.dumps(
                {
                    "candidate_ready": not candidate_blockers,
                    "issues": [asdict(issue) for issue in issues],
                    "release_ready": not release_blockers,
                    "requested_mode": args.mode,
                    "requested_mode_ready": not requested_blockers,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif requested_blockers:
        for issue in requested_blockers:
            print(f"[{issue.code}] {issue.message}", file=sys.stderr)
    elif release_blockers:
        print("Candidate legal checks passed; release readiness is FALSE")
        for issue in release_blockers:
            print(f"[{issue.code}] {issue.message}")
    else:
        print("Rust third-party license inventory passed")
    return 0 if not requested_blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
