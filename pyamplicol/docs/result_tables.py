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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXACT_REEXEC_MARKER = "PYAMPLICOL_EXACT_PYTHON_REEXEC"
_EXACT_IMPORT_PATHS = "PYAMPLICOL_EXACT_IMPORT_PATHS"


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


def _bootstrap_exact_python() -> None:
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
            str(Path(__file__).resolve(strict=True)),
            *sys.argv[1:],
        ),
        environment,
    )
    raise RuntimeError("exact Python re-exec returned unexpectedly")


if __name__ == "__main__":
    _bootstrap_exact_python()

NATIVE_PACKAGE_DIR = _native_package_dir()
for source_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

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

from tools.performance_report.runtime_evidence import (  # noqa: E402
    native_extension_in_package,
    preimport_python_runtime_identity,
)

if __name__ == "__main__" and NATIVE_PACKAGE_DIR is not None:
    _EXACT_PACKAGE_ROOTS = (
        REPOSITORY_ROOT / "src" / "pyamplicol",
        NATIVE_PACKAGE_DIR,
    )
    _EXACT_NATIVE_EXTENSION = native_extension_in_package(NATIVE_PACKAGE_DIR)
    preimport_python_runtime_identity(
        _EXACT_PACKAGE_ROOTS,
        native_extension=_EXACT_NATIVE_EXTENSION,
    )

import pyamplicol  # noqa: E402

if (
    NATIVE_PACKAGE_DIR is not None
    and str(NATIVE_PACKAGE_DIR) not in pyamplicol.__path__
):
    # Match normal Python import precedence exactly. Additional native-bearing
    # installations later on sys.path remain ineligible rather than silently
    # expanding this exact-source package namespace.
    pyamplicol.__path__.append(str(NATIVE_PACKAGE_DIR))

from tools.performance_report.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
