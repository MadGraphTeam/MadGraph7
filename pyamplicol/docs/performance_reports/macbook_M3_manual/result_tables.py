#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compatibility entry point for the performance-report service."""

from __future__ import annotations

import importlib.machinery
import json
import os
import subprocess
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
    "publish-snapshot",
    "validate-snapshot",
    "snapshot-cell-boundary",
    "accept-cell-boundary",
}
_GLOBAL_OPTIONS_WITH_VALUE = {
    "--repo-root",
    "--report-profile",
    "--docs-dir",
    "--artifact-root",
    "--coordination-root",
    "--class-c-ancestor-runtime-root",
    "--measurement-source-root",
    "--expected-measurement-source-revision",
    "--expected-measurement-source-tree",
    "--expected-policy-wrapper-revision",
    "--expected-policy-wrapper-tree",
    "--expected-policy-entrypoint-sha256",
    "--expected-legacy-adapter-sha256",
    "--study-contract-sha256",
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


def _option_path(arguments: list[str], option: str) -> Path | None:
    """Read one early bootstrap path option without importing the CLI."""

    values: list[str] = []
    offset = 0
    while offset < len(arguments):
        argument = arguments[offset]
        if argument == option:
            if offset + 1 >= len(arguments):
                raise RuntimeError(f"{option} requires one absolute path")
            values.append(arguments[offset + 1])
            offset += 2
            continue
        prefix = f"{option}="
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        offset += 1
    if not values:
        return None
    if len(values) != 1:
        raise RuntimeError(f"{option} must be specified exactly once")
    path = Path(values[0])
    if not path.is_absolute():
        raise RuntimeError(f"{option} must be an absolute path")
    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{option} is unavailable") from error
    if not (root / "src/pyamplicol").is_dir():
        raise RuntimeError(f"{option} has no pyamplicol source package")
    return root


def _option_value(arguments: list[str], option: str) -> str | None:
    """Read one exact early-bootstrap value without importing the CLI."""

    values: list[str] = []
    offset = 0
    while offset < len(arguments):
        argument = arguments[offset]
        if argument == option:
            if offset + 1 >= len(arguments):
                raise RuntimeError(f"{option} requires one value")
            values.append(arguments[offset + 1])
            offset += 2
            continue
        prefix = f"{option}="
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        offset += 1
    if not values:
        return None
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"{option} must be specified exactly once")
    return values[0]


def _git_value(root: Path, expression: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), "rev-parse", expression),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"cannot authenticate worker checkout identity {expression!r}"
        )
    return value


def _require_git_identity(
    root: Path,
    *,
    revision: str,
    tree: str,
    require_clean: bool,
    require_tracked_clean: bool = False,
    untracked_source_paths: tuple[str, ...] = (),
) -> None:
    if (
        _git_value(root, "HEAD^{commit}") != revision
        or _git_value(root, "HEAD^{tree}") != tree
    ):
        raise RuntimeError("worker checkout Git identity differs")
    if require_clean:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or completed.stdout:
            raise RuntimeError("policy-wrapper checkout is not clean")
    if require_tracked_clean:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "diff",
                "--quiet",
                "--no-ext-diff",
                "HEAD",
                "--",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 1:
            raise RuntimeError(
                "measured-source checkout has tracked changes"
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "cannot inspect measured-source tracked cleanliness"
            )
    if untracked_source_paths:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *untracked_source_paths,
            ),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "cannot inspect measured-source untracked files"
            )
        if completed.stdout:
            raise RuntimeError(
                "measured-source checkout has untracked files in "
                "imported source roots"
            )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measured_venv_site_paths(measured_root: Path) -> tuple[Path, ...]:
    """Recover the measured checkout's venv even under ``-I -S``."""

    expected_venv = measured_root / ".venv"
    if expected_venv.is_symlink() or not expected_venv.is_dir():
        raise RuntimeError(
            "split worker requires a regular measured .venv directory"
        )
    venv_root = expected_venv.resolve(strict=True)
    executable = Path(os.path.abspath(sys.executable))
    if executable.parent.parent.resolve(strict=True) != venv_root:
        raise RuntimeError(
            "split worker interpreter is not the measured .venv interpreter"
        )
    configuration = expected_venv / "pyvenv.cfg"
    if configuration.is_symlink() or not configuration.is_file():
        raise RuntimeError("measured .venv has no regular pyvenv.cfg")
    fields: dict[str, str] = {}
    for line in configuration.read_text(
        encoding="utf-8",
        errors="strict",
    ).splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip().lower()] = value.strip()
    if fields.get("include-system-site-packages", "").lower() != "false":
        raise RuntimeError(
            "measured .venv must exclude system site-packages"
        )

    variables = dict(sysconfig.get_config_vars())
    variables.update(
        {
            "base": str(venv_root),
            "platbase": str(venv_root),
            "installed_base": str(venv_root),
            "installed_platbase": str(venv_root),
        }
    )
    paths: list[Path] = []
    for name in ("purelib", "platlib"):
        value = sysconfig.get_path(name, vars=variables)
        if not isinstance(value, str):
            raise RuntimeError(f"measured .venv has no {name} path")
        candidate = Path(value)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(venv_root)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"measured .venv {name} path is unavailable"
            ) from error
        if resolved not in paths:
            paths.append(resolved)
    if not paths:
        raise RuntimeError("measured .venv has no site-packages path")
    return tuple(paths)


def _native_extensions(package_dir: Path) -> tuple[Path, ...]:
    try:
        candidates = tuple(
            sorted(
                (
                    path
                    for path in package_dir.iterdir()
                    if path.name.startswith("_rusticol")
                    and path.name.endswith(
                        tuple(importlib.machinery.EXTENSION_SUFFIXES)
                    )
                ),
                key=lambda path: path.name,
            )
        )
    except OSError:
        return ()
    if any(path.is_symlink() or not path.is_file() for path in candidates):
        raise RuntimeError(
            "measured native extension must be a regular file"
        )
    return candidates


def _measured_split_runtime(
    measured_root: Path,
) -> tuple[tuple[Path, ...], Path]:
    """Return exact venv import roots and one canonical measured native root."""

    site_paths = _measured_venv_site_paths(measured_root)
    source_package = (measured_root / "src/pyamplicol").resolve(strict=True)
    package_dirs = (
        source_package,
        *(
            site_path / "pyamplicol"
            for site_path in site_paths
            if (site_path / "pyamplicol").is_dir()
        ),
    )
    candidates: list[tuple[Path, Path, str]] = []
    for package_dir in package_dirs:
        extensions = _native_extensions(package_dir)
        if len(extensions) > 1:
            raise RuntimeError(
                "measured package has multiple native extension candidates"
            )
        if extensions:
            extension = extensions[0]
            candidates.append(
                (package_dir, extension, _sha256_file(extension))
            )
    if not candidates:
        raise RuntimeError(
            "split worker found no native extension in measured source/venv"
        )
    if len({digest for _package, _path, digest in candidates}) != 1:
        raise RuntimeError(
            "measured source and venv native extensions differ"
        )
    canonical_package = next(
        (
            package
            for package, _path, _digest in candidates
            if package == source_package
        ),
        candidates[0][0],
    )
    return site_paths, canonical_package


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
CLASS_C_ANCESTOR_RUNTIME_ROOT = _option_path(
    ARGUMENTS,
    "--class-c-ancestor-runtime-root",
)
MEASUREMENT_SOURCE_ROOT = _option_path(
    ARGUMENTS,
    "--measurement-source-root",
)
_SPLIT_WORKER_OPTIONS = {
    "measured_revision": _option_value(
        ARGUMENTS,
        "--expected-measurement-source-revision",
    ),
    "measured_tree": _option_value(
        ARGUMENTS,
        "--expected-measurement-source-tree",
    ),
    "wrapper_revision": _option_value(
        ARGUMENTS,
        "--expected-policy-wrapper-revision",
    ),
    "wrapper_tree": _option_value(
        ARGUMENTS,
        "--expected-policy-wrapper-tree",
    ),
    "entrypoint_sha256": _option_value(
        ARGUMENTS,
        "--expected-policy-entrypoint-sha256",
    ),
    "legacy_adapter_sha256": _option_value(
        ARGUMENTS,
        "--expected-legacy-adapter-sha256",
    ),
    "study_contract_sha256": _option_value(
        ARGUMENTS,
        "--study-contract-sha256",
    ),
}
_SPLIT_WORKER_SPECIFIED = (
    MEASUREMENT_SOURCE_ROOT is not None
    or any(value is not None for value in _SPLIT_WORKER_OPTIONS.values())
)
if _SPLIT_WORKER_SPECIFIED:
    if MEASUREMENT_SOURCE_ROOT is None or any(
        value is None for value in _SPLIT_WORKER_OPTIONS.values()
    ):
        raise RuntimeError(
            "split worker wrapper/source options must be specified together"
        )
    if COMMAND not in {"_prepare", "_worker"}:
        raise RuntimeError(
            "split worker wrapper/source options are restricted to workers"
        )
    requested_repo_root = _option_path(ARGUMENTS, "--repo-root")
    if requested_repo_root != MEASUREMENT_SOURCE_ROOT:
        raise RuntimeError(
            "worker --repo-root must equal --measurement-source-root"
        )
    measured_revision = str(_SPLIT_WORKER_OPTIONS["measured_revision"])
    measured_tree = str(_SPLIT_WORKER_OPTIONS["measured_tree"])
    wrapper_revision = str(_SPLIT_WORKER_OPTIONS["wrapper_revision"])
    wrapper_tree = str(_SPLIT_WORKER_OPTIONS["wrapper_tree"])
    _require_git_identity(
        MEASUREMENT_SOURCE_ROOT,
        revision=measured_revision,
        tree=measured_tree,
        require_clean=False,
        require_tracked_clean=True,
        # ``src`` is the measured checkout's only Python import root.
        untracked_source_paths=("src",),
    )
    _require_git_identity(
        REPOSITORY_ROOT,
        revision=wrapper_revision,
        tree=wrapper_tree,
        require_clean=True,
    )
    if _sha256_file(ENTRYPOINT) != _SPLIT_WORKER_OPTIONS[
        "entrypoint_sha256"
    ]:
        raise RuntimeError("policy-wrapper entrypoint digest differs")
    if _sha256_file(
        REPOSITORY_ROOT / "tools/performance_report/legacy.py"
    ) != _SPLIT_WORKER_OPTIONS["legacy_adapter_sha256"]:
        raise RuntimeError("policy-wrapper legacy adapter digest differs")
if (
    CLASS_C_ANCESTOR_RUNTIME_ROOT is not None
    and COMMAND != "prepare-class-c-bridge"
):
    raise RuntimeError(
        "--class-c-ancestor-runtime-root is restricted to "
        "prepare-class-c-bridge"
    )
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
        if MEASUREMENT_SOURCE_ROOT is not None:
            measured_paths, _native_package = _measured_split_runtime(
                MEASUREMENT_SOURCE_ROOT
            )
            if raw_paths != [str(path) for path in measured_paths]:
                raise RuntimeError(
                    "exact Python import paths differ from measured .venv"
                )
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
    if MEASUREMENT_SOURCE_ROOT is not None:
        measured_paths, _native_package = _measured_split_runtime(
            MEASUREMENT_SOURCE_ROOT
        )
        import_paths = [str(path) for path in measured_paths]
    else:
        import_paths = []
        native_package = _native_package_dir()
        if native_package is not None:
            import_paths.append(
                str(native_package.parent.resolve(strict=True))
            )
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

_PACKAGE_SOURCE_ROOT = (
    CLASS_C_ANCESTOR_RUNTIME_ROOT
    if CLASS_C_ANCESTOR_RUNTIME_ROOT is not None
    else (
        REPOSITORY_ROOT
        if MEASUREMENT_SOURCE_ROOT is None
        else MEASUREMENT_SOURCE_ROOT
    )
)
_IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    _PACKAGE_SOURCE_ROOT / "src",
)
for source_root in reversed(_IMPORT_ROOTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
NATIVE_PACKAGE_DIR = (
    _native_package_dir()
    if MEASUREMENT_SOURCE_ROOT is None
    else _measured_split_runtime(MEASUREMENT_SOURCE_ROOT)[1]
)

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
        _EXACT_SOURCE_PACKAGE = _PACKAGE_SOURCE_ROOT / "src" / "pyamplicol"
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
    if MEASUREMENT_SOURCE_ROOT is not None:
        expected_package_roots = {
            path.resolve(strict=True)
            for path in _EXACT_PACKAGE_ROOTS
        }
        observed_package_roots = {
            Path(path).resolve(strict=True)
            for path in pyamplicol.__path__
        }
        if (
            observed_package_roots
            - expected_package_roots
            or _EXACT_SOURCE_PACKAGE.resolve(strict=True)
            not in observed_package_roots
        ):
            raise RuntimeError(
                "pyamplicol namespace escaped measured source/venv roots"
            )

from tools.performance_report.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(ARGUMENTS))
