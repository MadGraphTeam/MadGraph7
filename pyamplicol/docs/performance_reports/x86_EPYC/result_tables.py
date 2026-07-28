#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compatibility entry point for the performance-report service."""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import sysconfig
import tempfile
import uuid
from pathlib import Path

_EXACT_REEXEC_MARKER = "PYAMPLICOL_EXACT_PYTHON_REEXEC"
_EXACT_IMPORT_PATHS = "PYAMPLICOL_EXACT_IMPORT_PATHS"
_PUBLICATION_ONLY_COMMANDS = {
    "init-profile",
    "export-profile",
    "validate",
    "audit",
    "audit-source-bridge",
    "reset",
    "render",
    "recover",
    "seal-existing-worker-result",
    "publish-snapshot",
    "validate-snapshot",
}
_GLOBAL_OPTIONS_WITH_VALUE = {
    "--repo-root",
    "--report-profile",
    "--docs-dir",
    "--artifact-root",
    "--coordination-root",
}


def _repository_root(entrypoint: Path) -> Path:
    for candidate in entrypoint.parents:
        if (
            (candidate / "tools/performance_report").is_dir()
            and (candidate / "src/pyamplicol").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "result_tables.py must run from a pyAmpliCol source checkout"
    )


def _embedded_profile(entrypoint: Path, repo_root: Path) -> str | None:
    profile_parent = repo_root / "docs/performance_reports"
    try:
        relative = entrypoint.parent.relative_to(profile_parent)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) == 1 else None


def _has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _subcommand(arguments: list[str]) -> str | None:
    offset = 0
    while offset < len(arguments):
        argument = arguments[offset]
        if argument in _GLOBAL_OPTIONS_WITH_VALUE:
            offset += 2
            continue
        if any(
            argument.startswith(f"{option}=")
            for option in _GLOBAL_OPTIONS_WITH_VALUE
        ):
            offset += 1
            continue
        if argument.startswith("-"):
            offset += 1
            continue
        return argument
    return None


ENTRYPOINT = Path(__file__).resolve()
REPOSITORY_ROOT = _repository_root(ENTRYPOINT)
EMBEDDED_PROFILE = _embedded_profile(ENTRYPOINT, REPOSITORY_ROOT)
ARGUMENTS = list(sys.argv[1:])
if (
    EMBEDDED_PROFILE is not None
    and not _has_option(ARGUMENTS, "--report-profile")
    and not _has_option(ARGUMENTS, "--docs-dir")
):
    ARGUMENTS[:0] = ("--report-profile", EMBEDDED_PROFILE)
COMMAND = _subcommand(ARGUMENTS)
RUNTIME_REQUIRED = COMMAND not in _PUBLICATION_ONLY_COMMANDS


def _native_package_dir(search_path: object = sys.path) -> Path | None:
    if not isinstance(search_path, (list, tuple)):
        raise RuntimeError("Python import path is invalid")
    for entry in search_path:
        if not isinstance(entry, str):
            continue
        package_dir = Path(entry) / "pyamplicol"
        try:
            native = any(
                path.is_file()
                and path.name.startswith("_rusticol")
                and path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
                for path in package_dir.iterdir()
            )
        except OSError:
            continue
        if native:
            return package_dir
    return None


def _bootstrap_exact_python(arguments: list[str]) -> None:
    """Re-exec before importing any repository or candidate module."""

    if os.environ.get(_EXACT_REEXEC_MARKER) == "1":
        if (
            sys.flags.isolated != 1
            or sys.flags.no_site != 1
            or sys.flags.ignore_environment != 1
            or sys.flags.dont_write_bytecode != 1
            or not sys.dont_write_bytecode
        ):
            raise RuntimeError("exact Python re-exec flags are incomplete")
        raw_paths = json.loads(os.environ[_EXACT_IMPORT_PATHS])
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) and Path(path).is_absolute() for path in raw_paths
        ):
            raise RuntimeError("exact Python import paths are invalid")
        cache_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
        if (
            not isinstance(cache_prefix, str)
            or not Path(cache_prefix).is_absolute()
            or Path(cache_prefix).exists()
        ):
            raise RuntimeError("exact Python cache prefix is not absent")
        for import_path in reversed(raw_paths):
            if import_path not in sys.path:
                sys.path.insert(0, import_path)
        # ``-I`` ignores PYTHON* variables. Install the authenticated absent
        # namespace explicitly before any repository or candidate import.
        sys.pycache_prefix = cache_prefix
        return

    prefix = Path(tempfile.gettempdir()) / (
        f".pyamplicol-no-bytecode-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if prefix.exists():
        raise RuntimeError(
            f"isolated Python cache prefix unexpectedly exists: {prefix}"
        )
    import_paths: list[str] = []
    native_package = _native_package_dir()
    if native_package is not None:
        import_paths.append(str(native_package.parent.resolve(strict=True)))
    for scheme_name in ("purelib", "platlib"):
        scheme_path = sysconfig.get_path(scheme_name)
        if not isinstance(scheme_path, str):
            continue
        try:
            resolved = str(Path(scheme_path).resolve(strict=True))
        except OSError:
            continue
        if resolved not in import_paths:
            import_paths.append(resolved)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    environment[_EXACT_REEXEC_MARKER] = "1"
    environment[_EXACT_IMPORT_PATHS] = json.dumps(import_paths)
    os.execve(
        sys.executable,
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ENTRYPOINT.resolve(strict=True)),
            *arguments,
        ),
        environment,
    )
    raise RuntimeError("exact Python re-exec returned unexpectedly")


if __name__ == "__main__":
    _bootstrap_exact_python(ARGUMENTS)

for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
NATIVE_PACKAGE_DIR = _native_package_dir()

if __name__ == "__main__":
    from tools.performance_report.runtime_evidence import (
        RuntimeEvidenceError,
        source_only_bytecode_policy,
    )

    try:
        source_only_bytecode_policy()
    except RuntimeEvidenceError as error:
        raise RuntimeError(
            "could not establish source-only Python bytecode isolation"
        ) from error

if __name__ == "__main__" and RUNTIME_REQUIRED:
    from tools.performance_report.runtime_evidence import (
        native_extension_in_package,
        preimport_python_runtime_identity,
    )

    if NATIVE_PACKAGE_DIR is not None:
        _EXACT_SOURCE_PACKAGE = REPOSITORY_ROOT / "src" / "pyamplicol"
        _EXACT_PACKAGE_ROOTS = (
            (_EXACT_SOURCE_PACKAGE,)
            if NATIVE_PACKAGE_DIR == _EXACT_SOURCE_PACKAGE
            else (_EXACT_SOURCE_PACKAGE, NATIVE_PACKAGE_DIR)
        )
        _EXACT_NATIVE_EXTENSION = native_extension_in_package(NATIVE_PACKAGE_DIR)
        preimport_python_runtime_identity(
            _EXACT_PACKAGE_ROOTS,
            native_extension=_EXACT_NATIVE_EXTENSION,
        )

    import pyamplicol

    if (
        NATIVE_PACKAGE_DIR is not None
        and str(NATIVE_PACKAGE_DIR) not in pyamplicol.__path__
    ):
        # Match normal Python import precedence exactly. Additional native-bearing
        # installations remain ineligible rather than expanding this namespace.
        pyamplicol.__path__.append(str(NATIVE_PACKAGE_DIR))

from tools.performance_report.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(ARGUMENTS))
