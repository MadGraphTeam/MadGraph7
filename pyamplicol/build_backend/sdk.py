# SPDX-License-Identifier: 0BSD
"""Build, validate, and stage Rusticol's native SDK."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

SDK_SOURCES = (
    ("rust/crates/rusticol-capi/include/rusticol.h", "include/rusticol.h"),
    ("rust/crates/rusticol-capi/include/rusticol.hpp", "include/rusticol.hpp"),
    ("rust/crates/rusticol-capi/fortran/rusticol.f90", "fortran/rusticol.f90"),
)
CAPI_PACKAGE = "rusticol-capi"
CAPI_MANIFEST = Path("rust/crates/rusticol-capi/Cargo.toml")
SDK_SBOM = Path("sboms/rusticol-capi.cyclonedx.json")
NATIVE_MARKER = "native-static-libs:"
FORBIDDEN_SYMBOLS = (
    "PyObject",
    "PyExc_",
    "PyGILState",
    "PyErr_",
    "PyLong_",
    "PyUnicode_",
    "PyMem_",
    "PyType_",
    "PyTuple_",
    "PyList_",
    "PyDict_",
    "PyModule_",
    "PyImport_",
    "PyBytes_",
    "PyCapsule_",
    "PyFloat_",
    "PyBool_",
    "PyThread_",
    "libpython",
    "numpy",
    "pyo3",
    "python3",
)
FORBIDDEN_UNDEFINED_SYMBOL_PREFIXES = (
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
    "py_",
    "python",
    "libpython",
    "pyo3",
    "numpy",
)
FORBIDDEN_BINARY_MARKERS = (
    b"/opt/homebrew",
    b"/opt/local",
    b"/site-packages/",
    b"/.venv/",
    b"/venv/",
    b"PREPARE_STANDALONE_PYAMPLICOL",
    b"/pyAmpliCol/pyAmpliCol",
)
COMMON_LIBRARIES = {
    "c",
    "dl",
    "m",
    "pthread",
    "resolv",
    "rt",
    "util",
}
MACOS_LIBRARIES = COMMON_LIBRARIES | {"System", "gcc_s", "iconv"}
LINUX_LIBRARIES = COMMON_LIBRARIES | {"gcc_s"}
MACOS_FRAMEWORKS = {
    "CoreFoundation",
    "IOKit",
    "Security",
    "SystemConfiguration",
}


@dataclass(frozen=True)
class _CargoPackage:
    name: str
    version: str
    manifest_path: Path
    source: str | None


def _host_target(root: Path) -> str:
    completed = subprocess.run(
        ["rustc", "-vV"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise RuntimeError("rustc -vV did not report a host target")


def _requested_target(host: str) -> str:
    requested = (
        os.environ.get("CARGO_BUILD_TARGET")
        or os.environ.get("MATURIN_BUILD_TARGET")
        or host
    )
    if requested != host:
        raise RuntimeError(
            "the native SDK bootstrap does not permit cross-target builds: "
            f"host={host}, requested={requested}"
        )
    return requested


def _cargo_messages(stdout: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"cargo emitted non-JSON output before SDK validation: {line}"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError("cargo JSON messages must be objects")
        messages.append(payload)
    return messages


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"cargo metadata {label} must be an object")
    return cast(dict[str, object], value)


def _json_array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"cargo metadata {label} must be an array")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"cargo metadata {label} must be a non-empty string")
    return value


def _cargo_metadata(root: Path, target_dir: Path, target: str) -> dict[str, object]:
    manifest = root / CAPI_MANIFEST
    if not manifest.is_file():
        raise RuntimeError(f"missing Rusticol C API manifest: {manifest}")
    completed = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--offline",
            "--filter-platform",
            target,
            "--manifest-path",
            str(manifest),
        ],
        cwd=root,
        env=dict(os.environ, CARGO_TARGET_DIR=str(target_dir)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostics = completed.stderr.strip() or completed.stdout.strip()
        detail = f"\n{diagnostics}" if diagnostics else ""
        raise RuntimeError(
            f"cargo metadata failed with exit code {completed.returncode}{detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cargo metadata emitted invalid JSON") from error
    return _json_object(payload, "output")


def _cargo_fetch(root: Path, target_dir: Path, target: str) -> None:
    completed = subprocess.run(
        ["cargo", "fetch", "--locked", "--target", target],
        cwd=root,
        env=dict(os.environ, CARGO_TARGET_DIR=str(target_dir)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostics = completed.stderr.strip() or completed.stdout.strip()
        detail = f"\n{diagnostics}" if diagnostics else ""
        raise RuntimeError(
            f"cargo fetch failed with exit code {completed.returncode}{detail}"
        )


def _cargo_purl(package: _CargoPackage) -> str:
    name = quote(package.name, safe=".-_~")
    version = quote(package.version, safe=".-_~")
    return f"pkg:cargo/{name}@{version}"


def _component(package: _CargoPackage, reference: str) -> dict[str, str]:
    return {
        "type": "library",
        "bom-ref": reference,
        "name": package.name,
        "version": package.version,
        "scope": "required",
        "purl": reference,
    }


def _cyclonedx_sbom(
    metadata: object,
    *,
    root_manifest: Path,
    target: str,
) -> bytes:
    payload = _json_object(metadata, "output")
    if type(payload.get("version")) is not int or payload["version"] != 1:
        raise RuntimeError("cargo metadata output has an unsupported format version")
    if (
        not target
        or any(character.isspace() for character in target)
        or "/" in target
        or "\\" in target
    ):
        raise RuntimeError("cargo metadata target is invalid")

    packages: dict[str, _CargoPackage] = {}
    for index, item in enumerate(_json_array(payload.get("packages"), "packages")):
        package = _json_object(item, f"packages[{index}]")
        package_id = _required_string(package.get("id"), f"packages[{index}].id")
        if package_id in packages:
            raise RuntimeError(f"cargo metadata repeats package id {package_id}")
        manifest_path = Path(
            _required_string(
                package.get("manifest_path"),
                f"packages[{index}].manifest_path",
            )
        )
        if not manifest_path.is_absolute():
            raise RuntimeError("cargo metadata package manifest paths must be absolute")
        if "source" not in package:
            raise RuntimeError(f"cargo metadata packages[{index}].source is missing")
        source = package["source"]
        if source is not None and (not isinstance(source, str) or not source):
            raise RuntimeError(f"cargo metadata packages[{index}].source is invalid")
        packages[package_id] = _CargoPackage(
            name=_required_string(package.get("name"), f"packages[{index}].name"),
            version=_required_string(
                package.get("version"), f"packages[{index}].version"
            ),
            manifest_path=manifest_path,
            source=source,
        )

    workspace_items = _json_array(payload.get("workspace_members"), "workspace_members")
    workspace_members = {
        _required_string(item, f"workspace_members[{index}]")
        for index, item in enumerate(workspace_items)
    }
    if len(workspace_members) != len(workspace_items):
        raise RuntimeError("cargo metadata repeats a workspace member")
    if not workspace_members <= packages.keys():
        raise RuntimeError("cargo metadata has an unknown workspace member")

    resolve = _json_object(payload.get("resolve"), "resolve")
    root_id = _required_string(resolve.get("root"), "resolve.root")
    if root_id not in workspace_members:
        raise RuntimeError("cargo metadata root is not a workspace member")

    nodes: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(_json_array(resolve.get("nodes"), "resolve.nodes")):
        node = _json_object(item, f"resolve.nodes[{index}]")
        node_id = _required_string(node.get("id"), f"resolve.nodes[{index}].id")
        if node_id in nodes:
            raise RuntimeError(f"cargo metadata repeats resolve node {node_id}")
        nodes[node_id] = node
    if packages.keys() != nodes.keys():
        raise RuntimeError("cargo metadata package and resolve-node sets differ")

    edges: dict[str, set[str]] = {}
    for node_id, node in nodes.items():
        required: set[str] = set()
        for index, item in enumerate(
            _json_array(node.get("deps"), f"resolve node {node_id} deps")
        ):
            dependency = _json_object(item, f"resolve node {node_id} deps[{index}]")
            _required_string(
                dependency.get("name"), f"resolve node {node_id} dependency name"
            )
            dependency_id = _required_string(
                dependency.get("pkg"), f"resolve node {node_id} dependency package"
            )
            if dependency_id not in nodes:
                raise RuntimeError(
                    f"cargo metadata node {node_id} references unknown package "
                    f"{dependency_id}"
                )
            kinds = _json_array(
                dependency.get("dep_kinds"),
                f"resolve node {node_id} dependency kinds",
            )
            if not kinds:
                raise RuntimeError("cargo metadata dependency has no dependency kind")
            include = False
            for kind_index, kind_item in enumerate(kinds):
                kind = _json_object(
                    kind_item,
                    f"resolve node {node_id} dependency kind {kind_index}",
                )
                if "kind" not in kind or "target" not in kind:
                    raise RuntimeError("cargo metadata dependency kind is incomplete")
                kind_name = kind["kind"]
                if kind_name is not None and (
                    not isinstance(kind_name, str) or kind_name not in {"build", "dev"}
                ):
                    raise RuntimeError(
                        f"cargo metadata dependency kind is invalid: {kind_name!r}"
                    )
                kind_target = kind["target"]
                if kind_target is not None and not isinstance(kind_target, str):
                    raise RuntimeError("cargo metadata dependency target is invalid")
                include = include or kind_name in {None, "build"}
            if include:
                required.add(dependency_id)
        edges[node_id] = required

    root_package = packages[root_id]
    if root_package.name != CAPI_PACKAGE or root_package.source is not None:
        raise RuntimeError(
            "cargo metadata did not resolve the local rusticol-capi root"
        )
    if root_package.manifest_path.resolve() != root_manifest.resolve():
        raise RuntimeError(
            "cargo metadata resolved an unexpected rusticol-capi manifest"
        )

    reachable: set[str] = set()
    pending = [root_id]
    while pending:
        package_id = pending.pop()
        if package_id in reachable:
            continue
        reachable.add(package_id)
        pending.extend(edges[package_id] - reachable)

    references: dict[str, str] = {}
    owners: dict[str, str] = {}
    for package_id in sorted(reachable):
        reference = _cargo_purl(packages[package_id])
        owner = owners.setdefault(reference, package_id)
        if owner != package_id:
            raise RuntimeError(
                f"cargo metadata closure has ambiguous package reference {reference}"
            )
        references[package_id] = reference

    ordered = sorted(reachable, key=references.__getitem__)
    document: dict[str, object] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": _component(root_package, references[root_id]),
            "properties": [
                {"name": "pyamplicol:rust-target", "value": target},
            ],
        },
        "components": [
            _component(packages[package_id], references[package_id])
            for package_id in ordered
            if package_id != root_id
        ],
        "dependencies": [
            {
                "ref": references[package_id],
                "dependsOn": sorted(
                    references[dependency_id] for dependency_id in edges[package_id]
                ),
            }
            for package_id in ordered
        ],
    }
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    forbidden = (b"file://", b"path+file:")
    if any(marker in encoded for marker in forbidden):
        raise RuntimeError("generated SDK SBOM retains a local Cargo reference")
    return encoded


def _static_library(messages: list[dict[str, Any]]) -> Path:
    candidates: list[Path] = []
    for message in messages:
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target")
        if not isinstance(target, dict) or target.get("name") != "rusticol_capi":
            continue
        for filename in message.get("filenames", []):
            path = Path(str(filename))
            if path.suffix in {".a", ".lib"}:
                candidates.append(path)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "expected one rusticol-capi static archive, found "
            + ", ".join(map(str, unique))
        )
    return unique[0]


def _native_tokens(stderr: str) -> list[str]:
    matches = [
        line.split(NATIVE_MARKER, 1)[1].strip()
        for line in stderr.splitlines()
        if NATIVE_MARKER in line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one rustc native-static-libs diagnostic, found {len(matches)}"
        )
    return shlex.split(matches[0])


def _typed_link_arguments(tokens: list[str], target: str) -> dict[str, Any]:
    libraries = MACOS_LIBRARIES if "apple-darwin" in target else LINUX_LIBRARIES
    result: dict[str, Any] = {
        "schema_version": 1,
        "target": target,
        "system_libraries": [],
        "frameworks": [],
    }
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-framework":
            if "apple-darwin" not in target or index + 1 >= len(tokens):
                raise RuntimeError(f"invalid native framework token: {token}")
            framework = tokens[index + 1]
            if framework not in MACOS_FRAMEWORKS:
                raise RuntimeError(f"native framework is not allowlisted: {framework}")
            result["frameworks"].append(framework)
            index += 2
            continue
        if token.startswith("-l") and len(token) > 2:
            library = token[2:]
            if library not in libraries:
                raise RuntimeError(
                    f"native system library is not allowlisted: {library}"
                )
            result["system_libraries"].append(library)
            index += 1
            continue
        if (
            token.startswith(("-L", "-Wl,"))
            or "rpath" in token.lower()
            or Path(token).is_absolute()
        ):
            raise RuntimeError(f"non-relocatable native link token: {token}")
        raise RuntimeError(f"unsupported native link token: {token}")
    return result


def _scan_archive(path: Path) -> None:
    # Retain a byte-level scan as defense in depth for crate names and embedded
    # strings which do not appear in the undefined-symbol table.
    archive_bytes = path.read_bytes()
    lowered_archive = archive_bytes.lower()
    for marker in FORBIDDEN_SYMBOLS:
        if marker.lower().encode() in lowered_archive:
            raise RuntimeError(
                f"Rusticol static archive unexpectedly references {marker}"
            )
    for binary_marker in FORBIDDEN_BINARY_MARKERS:
        if binary_marker.lower() in lowered_archive:
            raise RuntimeError(
                "Rusticol static archive embeds a non-relocatable path marker: "
                f"{binary_marker.decode(errors='replace')}"
            )


def _scan_archive_symbols(path: Path) -> bool:
    """Inspect undefined symbols when a compatible object reader is available.

    Rust tracks upstream LLVM more closely than Apple's command-line tools. An
    older ``nm`` can therefore reject a valid Rust archive before the platform
    linker does. Returning ``False`` in that case delegates the authoritative
    check to :func:`_validate_archive_linkage`; a successfully parsed table is
    still checked for forbidden Python dependencies here.
    """

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
            check=False,
            capture_output=True,
            text=True,
        )
        forbidden: list[str] = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[-2] not in {"U", "u", "W", "w", "V", "v"}:
                continue
            symbol = fields[-1]
            normalized = symbol.removeprefix("_").lower()
            if normalized.startswith(FORBIDDEN_UNDEFINED_SYMBOL_PREFIXES):
                forbidden.append(symbol)
        if forbidden:
            raise RuntimeError(
                "Rusticol static archive has undefined Python/PyO3/NumPy symbols: "
                + ", ".join(sorted(set(forbidden)))
            )
        if completed.returncode == 0:
            return True
    return False


def _native_c_compiler(target: str) -> list[str]:
    configured = os.environ.get("CC")
    if configured:
        command = shlex.split(configured)
        if command:
            return command
    names = ("clang", "cc") if "apple-darwin" in target else ("cc", "clang")
    for name in names:
        compiler = shutil.which(name)
        if compiler:
            return [compiler]
    raise RuntimeError("a C compiler is required to validate the Rusticol SDK")


def _c_api_exports(header: Path) -> tuple[str, ...]:
    exports = tuple(
        dict.fromkeys(
            re.findall(
                r"\b(rusticol_[a-z0-9_]+)\s*\(",
                header.read_text(encoding="utf-8"),
            )
        )
    )
    if not exports or "rusticol_abi_version" not in exports:
        raise RuntimeError(
            f"Rusticol C header declares no complete public ABI: {header}"
        )
    return exports


def _native_link_flags(link: Mapping[str, Any], target: str) -> list[str]:
    if link.get("target") != target:
        raise RuntimeError("Rusticol native link metadata targets the wrong platform")
    flags: list[str] = []
    libraries = link.get("system_libraries")
    frameworks = link.get("frameworks")
    if not isinstance(libraries, list) or not all(
        isinstance(item, str) for item in libraries
    ):
        raise RuntimeError("Rusticol native system-library metadata is invalid")
    if not isinstance(frameworks, list) or not all(
        isinstance(item, str) for item in frameworks
    ):
        raise RuntimeError("Rusticol native framework metadata is invalid")
    flags.extend(f"-l{library}" for library in libraries)
    for framework in frameworks:
        flags.extend(("-framework", framework))
    return flags


def _validate_archive_linkage(
    archive: Path,
    *,
    header: Path,
    link: Mapping[str, Any],
    target: str,
) -> None:
    """Link and execute a probe that references every declared C ABI export."""

    exports = _c_api_exports(header)
    entries = ",\n".join(f"    (rusticol_probe_fn){name}" for name in exports)
    source = f"""\
#include <stdint.h>
#include \"rusticol.h\"

typedef void (*rusticol_probe_fn)(void);
static rusticol_probe_fn volatile rusticol_api[] = {{
{entries}
}};

int main(void) {{
    uint32_t (*abi_version)(void) = (uint32_t (*)(void))rusticol_api[0];
    return abi_version() == RUSTICOL_ABI_VERSION ? 0 : 1;
}}
"""
    with tempfile.TemporaryDirectory(prefix="pyamplicol-sdk-link-") as temporary:
        directory = Path(temporary)
        probe_source = directory / "probe.c"
        probe_binary = directory / "probe"
        probe_source.write_text(source, encoding="utf-8")
        command = [
            *_native_c_compiler(target),
            "-std=c11",
            "-O0",
            "-I",
            str(header.parent),
            str(probe_source),
            str(archive),
            *_native_link_flags(link, target),
            "-o",
            str(probe_binary),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                "Rusticol static archive failed the complete C ABI link probe" + suffix
            )
        completed = subprocess.run(
            [str(probe_binary)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                "Rusticol static archive failed the C ABI execution probe" + suffix
            )


def _package_version(root: Path) -> str:
    candidate = root / "src" / "pyamplicol" / "_build_info.json"
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        version = payload.get("version")
        if isinstance(version, str) and version:
            return version
        raise RuntimeError("candidate build information has no package version")
    with (root / "Cargo.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    version = payload.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("Cargo workspace has no package version")
    return version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_sdk(root: Path, target_dir: Path) -> Path:
    host = _host_target(root)
    target = _requested_target(host)
    environment = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    _cargo_fetch(root, target_dir, target)
    completed = subprocess.run(
        [
            "cargo",
            "rustc",
            "--locked",
            "--offline",
            "--release",
            "--package",
            "rusticol-capi",
            "--target",
            target,
            "--message-format=json-render-diagnostics",
            "--",
            "--print",
            "native-static-libs",
            "-C",
            "lto=off",
            "-C",
            "embed-bitcode=no",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    sbom = _cyclonedx_sbom(
        _cargo_metadata(root, target_dir, target),
        root_manifest=root / CAPI_MANIFEST,
        target=target,
    )
    messages = _cargo_messages(completed.stdout)
    archive = _static_library(messages)
    if not archive.is_file():
        raise RuntimeError(f"cargo reported a missing static archive: {archive}")
    link = _typed_link_arguments(_native_tokens(completed.stderr), target)
    _scan_archive(archive)
    _scan_archive_symbols(archive)
    _validate_archive_linkage(
        archive,
        header=root / "rust/crates/rusticol-capi/include/rusticol.h",
        link=link,
        target=target,
    )

    staging = target_dir.parent / "wheel-data" / "_sdk"
    for source_name, destination_name in SDK_SOURCES:
        source = root / source_name
        if not source.is_file():
            raise RuntimeError(f"missing Rusticol SDK source: {source}")
        destination = staging / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    archive_name = (
        "rusticol_capi.lib" if archive.suffix == ".lib" else "librusticol_capi.a"
    )
    library = staging / "lib" / archive_name
    library.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, library)
    (staging / "link.json").write_text(
        json.dumps(link, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sbom_path = staging / SDK_SBOM
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    sbom_path.write_bytes(sbom)
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": _package_version(root),
                "target": target,
                "archive": f"lib/{archive_name}",
                "archive_sha256": _sha256(library),
                "sbom": SDK_SBOM.as_posix(),
                "sbom_sha256": _sha256(sbom_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return staging
