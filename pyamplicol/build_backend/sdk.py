# SPDX-License-Identifier: 0BSD
"""Build, validate, and stage Rusticol's native SDK."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SDK_SOURCES = (
    ("rust/crates/rusticol-capi/include/rusticol.h", "include/rusticol.h"),
    ("rust/crates/rusticol-capi/include/rusticol.hpp", "include/rusticol.hpp"),
    ("rust/crates/rusticol-capi/fortran/rusticol.f90", "fortran/rusticol.f90"),
)
RUST_SDK_SOURCE = "src/pyamplicol/_sdk/rust/rusticol.rs"
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
    rust_source = root / RUST_SDK_SOURCE
    if not rust_source.is_file():
        raise RuntimeError(f"missing Rusticol SDK source: {rust_source}")
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
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": _package_version(root),
                "target": target,
                "archive": f"lib/{archive_name}",
                "rust_source": "rust/rusticol.rs",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return staging
