# SPDX-License-Identifier: 0BSD
"""Build the wheel's combined Rust and Python distribution SBOM."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote

_EPOCH = "1970-01-01T00:00:00Z"
_ZERO_SERIAL = "urn:uuid:00000000-0000-4000-8000-000000000000"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DISTRIBUTION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_ARTIFACT_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_REQUIREMENT = re.compile(
    r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"\s*(?:==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*))?\s*"
)
_LOCAL_MARKERS = (
    "path+file:",
    "file://",
    "/Users/",
    "/home/",
    "/tmp/",
    "/var/tmp/",
    "/private/var/",
    "/opt/homebrew/",
    "/opt/local/",
    "/github/workspace/",
)


@dataclass(frozen=True)
class _Artifact:
    filename: str
    sha256: str


@dataclass(frozen=True)
class _PythonPackage:
    distribution: str
    name: str
    version: str
    license: str
    dependencies: tuple[tuple[str, str | None], ...]
    artifacts: tuple[_Artifact, ...]
    hashes_verified: bool
    direct: bool | None

    @property
    def purl(self) -> str:
        return _pypi_purl(self.name, self.version)


def _object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{description} must be an object")
    return value


def _array(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{description} must be an array")
    return value


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{description} must be a non-empty string")
    return value


def _canonical_name(value: str) -> str:
    if _DISTRIBUTION.fullmatch(value) is None:
        raise RuntimeError(f"invalid Python distribution name: {value!r}")
    return re.sub(r"[-_.]+", "-", value).lower()


def _pypi_purl(name: str, version: str) -> str:
    return f"pkg:pypi/{quote(name, safe='.-_~')}@{quote(version, safe='.-_~')}"


def _artifact_records(
    entry: Mapping[str, object], description: str
) -> tuple[_Artifact, ...]:
    keys = [key for key in ("artifacts", "files") if key in entry]
    if len(keys) > 1:
        raise RuntimeError(f"{description} defines both artifacts and files")
    raw: object = entry.get(keys[0], []) if keys else []
    if "filename" in entry or "sha256" in entry:
        if keys:
            raise RuntimeError(f"{description} mixes package and nested artifact data")
        raw = [{"filename": entry.get("filename"), "sha256": entry.get("sha256")}]
    records: list[_Artifact] = []
    filenames: set[str] = set()
    for index, item in enumerate(_array(raw, f"{description} artifacts")):
        artifact = _object(item, f"{description} artifact[{index}]")
        filename = _string(
            artifact.get("filename"), f"{description} artifact[{index}].filename"
        )
        digest = _string(
            artifact.get("sha256"), f"{description} artifact[{index}].sha256"
        )
        if (
            _ARTIFACT_FILENAME.fullmatch(filename) is None
            or not filename.endswith(".whl")
            or filename in {".", ".."}
        ):
            raise RuntimeError(f"{description} has an unsafe artifact filename")
        if _SHA256.fullmatch(digest) is None:
            raise RuntimeError(f"{description} has an invalid artifact SHA-256")
        if filename in filenames:
            raise RuntimeError(f"{description} repeats artifact {filename}")
        filenames.add(filename)
        records.append(_Artifact(filename, digest))
    return tuple(sorted(records, key=lambda item: (item.filename, item.sha256)))


def _dependency_record(value: object, description: str) -> tuple[str, str | None]:
    if isinstance(value, str):
        match = _REQUIREMENT.fullmatch(value)
        if match is None:
            raise RuntimeError(f"{description} must be a name or exact requirement")
        return _canonical_name(match.group(1)), match.group(2)
    entry = _object(value, description)
    distribution = _string(entry.get("distribution"), f"{description}.distribution")
    version = entry.get("version")
    if version is not None and (not isinstance(version, str) or not version):
        raise RuntimeError(f"{description}.version must be a non-empty string")
    return _canonical_name(distribution), version


def _python_packages(
    lock: Mapping[str, object],
    *,
    package_key: str,
    global_hashes_verified: bool,
) -> dict[str, _PythonPackage]:
    raw_packages = _array(lock.get(package_key), f"Python lock {package_key}")
    if not raw_packages:
        raise RuntimeError("Python lock has no runtime dependencies")
    packages: dict[str, _PythonPackage] = {}
    for index, item in enumerate(raw_packages):
        description = f"Python lock {package_key}[{index}]"
        entry = _object(item, description)
        distribution = _string(entry.get("distribution"), f"{description}.distribution")
        name = _canonical_name(distribution)
        if name in packages:
            raise RuntimeError(f"release lock repeats Python distribution {name}")
        version = _string(entry.get("version"), f"{description}.version")
        license_name = _string(entry.get("license"), f"{description}.license")
        dependency_keys = [key for key in ("dependencies", "requires") if key in entry]
        if len(dependency_keys) > 1:
            raise RuntimeError(f"{description} defines both dependencies and requires")
        raw_dependencies: object = (
            entry.get(dependency_keys[0], []) if dependency_keys else []
        )
        dependencies = tuple(
            sorted(
                (
                    _dependency_record(value, f"{description} dependency[{dep_index}]")
                    for dep_index, value in enumerate(
                        _array(raw_dependencies, f"{description} dependencies")
                    )
                ),
                key=lambda value: (value[0], value[1] or ""),
            )
        )
        if len({name for name, _version in dependencies}) != len(dependencies):
            raise RuntimeError(f"{description} repeats a dependency")
        verified = entry.get("hashes_verified", global_hashes_verified)
        if not isinstance(verified, bool):
            raise RuntimeError(f"{description}.hashes_verified must be Boolean")
        direct = entry.get("direct")
        if direct is not None and not isinstance(direct, bool):
            raise RuntimeError(f"{description}.direct must be Boolean")
        packages[name] = _PythonPackage(
            distribution=distribution,
            name=name,
            version=version,
            license=license_name,
            dependencies=dependencies,
            artifacts=_artifact_records(entry, description),
            hashes_verified=verified,
            direct=direct,
        )

    for package in packages.values():
        for dependency_name, dependency_version in package.dependencies:
            dependency = packages.get(dependency_name)
            if dependency is None:
                raise RuntimeError(
                    "Python dependency graph references unlocked package "
                    f"{dependency_name}"
                )
            if (
                dependency_version is not None
                and dependency_version != dependency.version
            ):
                raise RuntimeError(
                    "Python dependency graph version mismatch for "
                    f"{dependency_name}: {dependency_version} != {dependency.version}"
                )
            if dependency_name == package.name:
                raise RuntimeError(
                    f"Python dependency graph contains a self edge for {package.name}"
                )
    artifact_filenames = [
        artifact.filename
        for package in packages.values()
        for artifact in package.artifacts
    ]
    if len(artifact_filenames) != len(set(artifact_filenames)):
        raise RuntimeError("Python dependency lock repeats an artifact filename")
    return packages


def _parse_toml(data: bytes, description: str) -> dict[str, Any]:
    try:
        payload = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"{description} is invalid") from error
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"{description} has an unsupported schema")
    return payload


def _runtime_packages(
    release: Mapping[str, object], runtime_lock: bytes | None
) -> tuple[dict[str, _PythonPackage], str | None, bool]:
    reference = release.get("python_runtime_lock")
    if reference is None:
        toolchain = _object(release.get("toolchain"), "release lock toolchain")
        globally_verified = toolchain.get("artifact_hashes_verified")
        if not isinstance(globally_verified, bool):
            raise RuntimeError("release lock artifact_hashes_verified must be Boolean")
        return (
            _python_packages(
                release,
                package_key="python_dependencies",
                global_hashes_verified=globally_verified,
            ),
            None,
            globally_verified,
        )

    record = _object(reference, "release lock python_runtime_lock")
    path = _string(record.get("path"), "release lock Python runtime lock path")
    expected_hash = _string(
        record.get("sha256"), "release lock Python runtime lock SHA-256"
    ).lower()
    if path != "dependencies/python-runtime-lock.toml":
        raise RuntimeError("release lock references an unexpected Python runtime lock")
    if _SHA256.fullmatch(expected_hash) is None:
        raise RuntimeError("release lock has an invalid Python runtime lock SHA-256")
    if runtime_lock is None:
        raise RuntimeError("wheel is missing the packaged Python runtime lock")
    actual_hash = hashlib.sha256(runtime_lock).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("packaged Python runtime lock fails its SHA-256 binding")
    runtime = _parse_toml(runtime_lock, "packaged Python runtime lock")
    project = _object(release.get("project"), "release lock project")
    if runtime.get("requires_python") != project.get("python_requires"):
        raise RuntimeError(
            "release and Python runtime locks disagree about requires-python"
        )
    packages = _python_packages(
        runtime,
        package_key="packages",
        global_hashes_verified=True,
    )

    declared: dict[str, str] = {}
    for index, item in enumerate(
        _array(
            release.get("python_dependencies"),
            "release lock python_dependencies",
        )
    ):
        entry = _object(item, f"release lock python_dependencies[{index}]")
        name = _canonical_name(
            _string(
                entry.get("distribution"),
                f"release lock python_dependencies[{index}].distribution",
            )
        )
        version = _string(
            entry.get("version"),
            f"release lock python_dependencies[{index}].version",
        )
        if name in declared:
            raise RuntimeError(f"release lock repeats Python distribution {name}")
        declared[name] = version
    direct = {
        package.name: package.version
        for package in packages.values()
        if package.direct is True
    }
    if declared != direct:
        raise RuntimeError(
            "release lock direct Python dependencies disagree with the runtime lock"
        )
    return packages, actual_hash, True


def _root_requirements(
    requirements: Sequence[str] | None,
    packages: Mapping[str, _PythonPackage],
) -> tuple[str, ...]:
    if requirements is None:
        direct_values = {package.direct for package in packages.values()}
        if direct_values <= {None}:
            return tuple(sorted(packages))
        if None in direct_values:
            raise RuntimeError(
                "Python dependency lock mixes explicit and implicit direct flags"
            )
        roots = sorted(package.name for package in packages.values() if package.direct)
        if not roots:
            raise RuntimeError(
                "Python dependency lock has no direct runtime dependencies"
            )
        return tuple(roots)

    roots: dict[str, str | None] = {}
    for index, raw in enumerate(requirements):
        requirement, separator, marker = raw.partition(";")
        if separator:
            if re.search(r"\bextra\b", marker):
                continue
            raise RuntimeError(
                f"runtime requirement[{index}] uses an unsupported marker: {raw}"
            )
        name, version = _dependency_record(requirement, f"runtime requirement[{index}]")
        if name in roots:
            raise RuntimeError(f"wheel metadata repeats runtime requirement {name}")
        roots[name] = version
    if set(roots) - set(packages):
        missing = sorted(set(roots) - set(packages))
        raise RuntimeError(
            "wheel metadata references unlocked Python dependencies: "
            + ", ".join(missing)
        )
    for name, version in roots.items():
        if version is not None and version != packages[name].version:
            raise RuntimeError(
                f"wheel metadata and release lock disagree about {name}: "
                f"{version} != {packages[name].version}"
            )
    if not roots:
        raise RuntimeError("wheel metadata has no locked runtime requirements")
    direct_values = {package.direct for package in packages.values()}
    if None not in direct_values:
        locked_roots = {
            package.name for package in packages.values() if package.direct is True
        }
        if set(roots) != locked_roots:
            raise RuntimeError(
                "wheel metadata and Python lock direct dependencies disagree"
            )
    return tuple(sorted(roots))


def _license_choice(value: str) -> dict[str, object]:
    if " " not in value or any(
        operator in value for operator in (" AND ", " OR ", " WITH ")
    ):
        return {"expression": value}
    return {"license": {"name": value}}


def _python_component(package: _PythonPackage) -> dict[str, object]:
    component: dict[str, object] = {
        "type": "library",
        "bom-ref": package.purl,
        "name": package.distribution,
        "version": package.version,
        "scope": "required",
        "purl": package.purl,
        "licenses": [_license_choice(package.license)],
        "properties": [
            {
                "name": "pyamplicol:pypi:hashes-verified",
                "value": str(
                    package.hashes_verified and bool(package.artifacts)
                ).lower(),
            },
            *(
                {
                    "name": "pyamplicol:pypi:artifact",
                    "value": json.dumps(
                        {"filename": artifact.filename, "sha256": artifact.sha256},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for artifact in package.artifacts
            ),
        ],
    }
    hashes = sorted({artifact.sha256 for artifact in package.artifacts})
    if hashes:
        component["hashes"] = [
            {"alg": "SHA-256", "content": digest} for digest in hashes
        ]
    return component


def _canonical_cargo_purl(component: Mapping[str, object], description: str) -> str:
    name = _string(component.get("name"), f"{description}.name")
    version = _string(component.get("version"), f"{description}.version")
    purl = _string(component.get("purl"), f"{description}.purl")
    canonical = purl.partition("?")[0]
    if not canonical.startswith("pkg:cargo/") or "#" in canonical:
        raise RuntimeError(f"{description} has an invalid Cargo purl")
    identity = canonical.removeprefix("pkg:cargo/")
    encoded_name, separator, encoded_version = identity.rpartition("@")
    if (
        not separator
        or unquote(encoded_name) != name
        or unquote(encoded_version) != version
    ):
        raise RuntimeError(f"{description} Cargo identity and purl disagree")
    return canonical


def _valid_hashes(component: Mapping[str, object], description: str) -> bool:
    raw_hashes = component.get("hashes", [])
    hashes = _array(raw_hashes, f"{description}.hashes")
    for index, item in enumerate(hashes):
        record = _object(item, f"{description}.hashes[{index}]")
        if (
            record.get("alg") != "SHA-256"
            or _SHA256.fullmatch(str(record.get("content", "")).lower()) is None
        ):
            raise RuntimeError(f"{description} contains an invalid package hash")
    return bool(hashes)


def _replace_references(value: object, replacements: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {
            key: _replace_references(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_references(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _rust_graph(
    document: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, tuple[str, ...]]]:
    metadata = _object(document.get("metadata"), "Maturin SBOM metadata")
    root = _object(metadata.get("component"), "Maturin SBOM root component")
    raw_components = _array(document.get("components"), "Maturin SBOM components")
    components = [
        _object(item, f"Maturin SBOM component[{index}]")
        for index, item in enumerate(raw_components)
    ]
    replacements: dict[str, str] = {}
    owners: dict[str, str] = {}
    records = [root, *components]
    for index, component in enumerate(records):
        description = (
            "Maturin SBOM root"
            if index == 0
            else f"Maturin SBOM component[{index - 1}]"
        )
        old_reference = _string(component.get("bom-ref"), f"{description}.bom-ref")
        reference = _canonical_cargo_purl(component, description)
        owner = owners.setdefault(reference, old_reference)
        if owner != old_reference:
            raise RuntimeError(f"Maturin SBOM has ambiguous Cargo purl {reference}")
        replacements[old_reference] = reference
        if not isinstance(component.get("licenses"), list) or not component["licenses"]:
            raise RuntimeError(f"{description} has no license declaration")
        has_hash = _valid_hashes(component, description)
        if old_reference.startswith("registry+") and not has_hash:
            raise RuntimeError(f"{description} has no registry package hash")

    normalized = _replace_references(document, replacements)
    assert isinstance(normalized, dict)
    normalized_metadata = _object(normalized.get("metadata"), "Maturin SBOM metadata")
    normalized_root = _object(
        normalized_metadata.get("component"), "Maturin SBOM root component"
    )
    normalized_components = [
        _object(item, f"Maturin SBOM component[{index}]")
        for index, item in enumerate(
            _array(normalized.get("components"), "Maturin SBOM components")
        )
    ]
    for component in [normalized_root, *normalized_components]:
        component["purl"] = str(component["bom-ref"])
        # Maturin nests the extension target beneath its Cargo package using a
        # checkout-relative ``file://.#src/lib.rs`` purl. It repeats the parent
        # package rather than adding a dependency or shipped component, so the
        # distribution SBOM keeps the canonical Cargo package and drops this
        # build-target-only record.
        component.pop("components", None)

    references = {
        _string(component.get("bom-ref"), "Maturin SBOM component bom-ref")
        for component in [normalized_root, *normalized_components]
    }
    raw_dependencies = _array(
        normalized.get("dependencies"), "Maturin SBOM dependencies"
    )
    edges: dict[str, tuple[str, ...]] = {}
    for index, item in enumerate(raw_dependencies):
        dependency = _object(item, f"Maturin SBOM dependency[{index}]")
        reference = _string(
            dependency.get("ref"), f"Maturin SBOM dependency[{index}].ref"
        )
        raw_children = _array(
            dependency.get("dependsOn", []),
            f"Maturin SBOM dependency[{index}].dependsOn",
        )
        children = tuple(
            sorted(_string(child, "Maturin dependency ref") for child in raw_children)
        )
        if reference in edges or len(children) != len(set(children)):
            raise RuntimeError("Maturin SBOM repeats a dependency record or edge")
        if reference in children:
            raise RuntimeError("Maturin SBOM contains a self dependency")
        edges[reference] = children
    if set(edges) != references:
        raise RuntimeError("Maturin SBOM component and dependency inventories differ")
    dangling = sorted(
        child
        for children in edges.values()
        for child in children
        if child not in references
    )
    if dangling:
        raise RuntimeError(
            f"Maturin SBOM has dangling dependency references: {dangling}"
        )
    root_reference = _string(
        normalized_root.get("bom-ref"), "Maturin SBOM root bom-ref"
    )
    _require_reachable(
        root_reference, references, edges, "Maturin Rust dependency graph"
    )
    return normalized_root, normalized_components, edges


def _require_reachable(
    root: str,
    references: set[str],
    edges: Mapping[str, Iterable[str]],
    description: str,
) -> None:
    reachable: set[str] = set()
    pending = [root]
    while pending:
        reference = pending.pop()
        if reference in reachable:
            continue
        reachable.add(reference)
        pending.extend(child for child in edges[reference] if child not in reachable)
    if reachable != references:
        unreachable = sorted(references - reachable)
        raise RuntimeError(f"{description} has unreachable components: {unreachable}")


def _validate_identity(root: Mapping[str, object], *, version: str, mode: str) -> str:
    cargo_version = (
        version if mode == "release" else version.replace(".dev0+", "-dev.0+")
    )
    if root.get("name") != "rusticol-python" or root.get("version") != cargo_version:
        raise RuntimeError("Maturin SBOM has an unexpected Rusticol-Python root")
    expected = f"pkg:cargo/rusticol-python@{cargo_version}"
    if root.get("bom-ref") != expected or root.get("purl") != expected:
        raise RuntimeError("Maturin SBOM Rusticol-Python purl is inconsistent")
    return expected


def _deterministic_serial(document: dict[str, object]) -> None:
    document["serialNumber"] = _ZERO_SERIAL
    identity = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    identifier = bytearray(hashlib.sha256(identity).digest()[:16])
    identifier[6] = (identifier[6] & 0x0F) | 0x40
    identifier[8] = (identifier[8] & 0x3F) | 0x80
    document["serialNumber"] = f"urn:uuid:{uuid.UUID(bytes=bytes(identifier))}"


def _reject_local_paths(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _reject_local_paths(child)
        return
    if isinstance(value, list):
        for child in value:
            _reject_local_paths(child)
        return
    if not isinstance(value, str):
        return
    decoded = unquote(value)
    if any(marker.lower() in decoded.lower() for marker in _LOCAL_MARKERS):
        raise RuntimeError("distribution SBOM retains a local path reference")
    if decoded.startswith("/") or re.match(r"[A-Za-z]:[\\/]", decoded):
        raise RuntimeError("distribution SBOM retains an absolute path reference")


def build_distribution_sbom(
    cargo_sbom: bytes,
    release_lock: bytes,
    python_runtime_lock: bytes | None,
    *,
    distribution_name: str,
    distribution_version: str,
    runtime_requirements: Sequence[str] | None,
    mode: str,
) -> bytes:
    """Merge Maturin's Cargo graph with the locked Python runtime graph."""

    if mode not in {"candidate", "release"}:
        raise RuntimeError(f"unsupported distribution SBOM mode: {mode}")
    try:
        document = json.loads(cargo_sbom)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Maturin generated an invalid CycloneDX SBOM") from error
    document = _object(document, "Maturin SBOM")
    if (
        document.get("bomFormat") != "CycloneDX"
        or document.get("specVersion") != "1.5"
        or document.get("version") != 1
    ):
        raise RuntimeError("Maturin generated an unsupported CycloneDX SBOM")
    lock = _parse_toml(release_lock, "packaged release lock")
    project = _object(lock.get("project"), "release lock project")
    locked_name = _string(project.get("distribution"), "release lock distribution")
    locked_version = _string(project.get("version"), "release lock project version")
    project_license = _string(project.get("license"), "release lock project license")
    if _canonical_name(distribution_name) != _canonical_name(locked_name):
        raise RuntimeError("wheel metadata and release lock distributions disagree")
    if mode == "release":
        if distribution_version != locked_version:
            raise RuntimeError("release wheel version disagrees with the release lock")
    elif (
        re.fullmatch(
            rf"{re.escape(locked_version)}\.dev0\+candidate\.[0-9a-f]+",
            distribution_version,
        )
        is None
    ):
        raise RuntimeError(
            "candidate wheel version is not derived from the release lock"
        )

    packages, runtime_lock_hash, globally_verified = _runtime_packages(
        lock, python_runtime_lock
    )
    if mode == "release":
        if not globally_verified:
            raise RuntimeError("release lock artifact hashes are not verified")
        incomplete = sorted(
            package.name
            for package in packages.values()
            if not package.hashes_verified or not package.artifacts
        )
        if incomplete:
            raise RuntimeError(
                "release lock lacks verified Python artifact records: "
                + ", ".join(incomplete)
            )
    root_names = _root_requirements(runtime_requirements, packages)

    rust_root, rust_components, rust_edges = _rust_graph(document)
    rust_root_ref = _validate_identity(
        rust_root, version=distribution_version, mode=mode
    )
    root_purl = _pypi_purl(_canonical_name(locked_name), distribution_version)
    distribution_root: dict[str, object] = {
        "type": "library",
        "bom-ref": root_purl,
        "name": locked_name,
        "version": distribution_version,
        "scope": "required",
        "purl": root_purl,
        "licenses": [_license_choice(project_license)],
    }

    metadata = _object(document.get("metadata"), "Maturin SBOM metadata")
    metadata = dict(metadata)
    metadata["timestamp"] = _EPOCH
    metadata["component"] = distribution_root
    raw_properties = metadata.get("properties", [])
    properties = [
        _object(item, f"Maturin metadata property[{index}]")
        for index, item in enumerate(
            _array(raw_properties, "Maturin metadata properties")
        )
    ]
    properties.extend(
        [
            {"name": "pyamplicol:build-mode", "value": mode},
            {
                "name": "pyamplicol:release-lock:sha256",
                "value": hashlib.sha256(release_lock).hexdigest(),
            },
        ]
    )
    if runtime_lock_hash is not None:
        properties.append(
            {
                "name": "pyamplicol:python-runtime-lock:sha256",
                "value": runtime_lock_hash,
            }
        )
    metadata["properties"] = sorted(
        properties,
        key=lambda item: (str(item.get("name", "")), str(item.get("value", ""))),
    )

    python_components = [_python_component(packages[name]) for name in sorted(packages)]
    all_components = [rust_root, *rust_components, *python_components]
    references = {
        _string(component.get("bom-ref"), "distribution component bom-ref")
        for component in all_components
    }
    if len(references) != len(all_components) or root_purl in references:
        raise RuntimeError("distribution SBOM contains duplicate component references")

    edges: dict[str, tuple[str, ...]] = dict(rust_edges)
    for package in packages.values():
        edges[package.purl] = tuple(
            sorted(packages[name].purl for name, _version in package.dependencies)
        )
    edges[root_purl] = tuple(
        sorted({rust_root_ref, *(packages[name].purl for name in root_names)})
    )
    all_references = references | {root_purl}
    if set(edges) != all_references:
        raise RuntimeError("distribution SBOM graph and component inventories differ")
    _require_reachable(
        root_purl, all_references, edges, "distribution dependency graph"
    )

    merged: dict[str, object] = {
        key: value
        for key, value in document.items()
        if key not in {"components", "dependencies", "metadata", "serialNumber"}
    }
    merged["metadata"] = metadata
    merged["components"] = sorted(all_components, key=lambda item: str(item["bom-ref"]))
    merged["dependencies"] = [
        {"ref": reference, "dependsOn": list(edges[reference])}
        for reference in sorted(edges)
    ]
    _deterministic_serial(merged)
    _reject_local_paths(merged)
    return (
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
