#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Authenticate and bind the native-x86 Arena acceptance evidence.

The ``preflight`` phase runs after the candidate wheel deployment check and
authenticates the clean source checkout, installed Python package, native
extension, build provenance, and candidate wheel before any ``pyamplicol``
import.  The ``audit`` phase repeats that authentication after the Arena gates,
validates their complete numerical contracts, and writes one content-addressed
acceptance manifest that binds every input evidence file.

Both phases re-exec with ``-I -S -B`` and an absent external bytecode-cache
prefix.  This is the same source-only bootstrap policy used by the compiled
Arena gate programs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import io
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_EXACT_REEXEC_MARKER = "PYAMPLICOL_X86_ACCEPTANCE_EXACT_REEXEC"
_EXACT_IMPORT_PATHS = "PYAMPLICOL_X86_ACCEPTANCE_IMPORT_PATHS"


def _bootstrap_source_only_python() -> None:
    """Re-exec before importing repository or candidate runtime modules."""

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
        sys.pycache_prefix = cache_prefix
        return

    prefix = Path(tempfile.gettempdir()) / (
        f".pyamplicol-x86-acceptance-no-bytecode-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if prefix.exists():
        raise RuntimeError(
            f"isolated Python cache prefix unexpectedly exists: {prefix}"
        )
    import_paths: list[str] = []
    native_root = next(
        (
            candidate
            for entry in sys.path
            if (candidate := Path(entry) / "pyamplicol").is_dir()
            and any(
                path.is_file()
                and path.name.startswith("_rusticol")
                and path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
                for path in candidate.iterdir()
            )
        ),
        None,
    )
    if native_root is not None:
        import_paths.append(str(native_root.parent.resolve(strict=True)))
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
    _bootstrap_source_only_python()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.performance_report.runtime_evidence import (  # noqa: E402
    RuntimeEvidenceError,
    established_preimport_runtime_identity,
    loaded_pyamplicol_origin_policy,
    native_extension_in_package,
    preimport_python_runtime_identity,
    source_only_bytecode_policy,
)

PREFLIGHT_KIND = "pyamplicol-arena-native-x86-runtime-preflight"
ACCEPTANCE_KIND = "pyamplicol-arena-native-x86-acceptance"
ALL_JIT_KIND = "pyamplicol-compiled-all-jit-direct-arena-gate"
FOUR_QUARK_KIND = "pyamplicol-four-quark-compiled-direct-arena-gate"
COLOR_MATRIX_KIND = "pyamplicol-eager-benchmark-matrix"
SCHEMA_VERSION = 1
COLOR_MATRIX_SCHEMA_VERSION = 3
EXPECTED_TARGET = "x86_64-unknown-linux-gnu"
EXPECTED_POINT_COUNT = 3
EXPECTED_MEMORY_LIMIT_GIB = 30
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
HASH_CHUNK_BYTES = 1024 * 1024
CONTENT_IDENTITY_ALGORITHM = "sha256-canonical-json-body-v1"
PREFLIGHT_FILENAME = "runtime-identity-preflight.json"
ACCEPTANCE_FILENAME = "arena-native-x86-acceptance.json"
ALL_JIT_RELATIVE_PATH = Path(
    "compiled-all-jit",
    "compiled-all-jit-arena-gate.json",
)
FOUR_QUARK_RELATIVE_PATH = Path(
    "four-quark",
    "four-quark-compiled-gate.json",
)
COLOR_MATRIX_RELATIVE_PATH = Path("eager-compiled-color", "result.json")

_ALL_JIT_LEVELS = {
    "jit-o0": 0,
    "jit-o1": 1,
    "jit-o2": 2,
    "jit-o3": 3,
}
_ALL_JIT_POINT_SEEDS = (846_731, 846_739, 846_763)
_ALL_JIT_COMPARISONS = {
    "f64_vs_precision32_total",
    "f64_evaluate_vs_resolved_total",
    "precision32_evaluate_vs_resolved_total",
    "f64_vs_precision32_resolved_components",
}
_CROSS_LEVEL_COMPARISONS = {
    "f64_totals",
    "precision32_totals",
    "f64_resolved_components",
    "precision32_resolved_components",
}
_FOUR_QUARK_LANES = {
    "lc-topology-replay": ("lc", "topology-replay", False),
    "lc-all-flow-union": ("lc", "all-flow-union", False),
    "nlc-contracted": ("nlc", "topology-replay", True),
    "full-contracted": ("full", "topology-replay", True),
}
_FOUR_QUARK_POINT_SEEDS = (443_041, 443_099, 443_137)
_FOUR_QUARK_BASE_COMPARISONS = {
    "f64_vs_precision32_total",
    "f64_vs_precision32_components",
    "f64_evaluate_vs_resolved_total",
    "precision32_evaluate_vs_resolved_total",
    "helicity_selector_evaluate_vs_resolved_total",
}
_CROSS_LC_COMPARISONS = {
    "f64_totals",
    "precision32_totals",
    "f64_resolved_components",
    "precision32_resolved_components",
    "color_selector",
    "helicity_selector",
}
_SMOKE_CASES = {
    "dd_z_3g": ("d d~ > z g g g", 4),
    "dd_3q_1g": ("d d~ > u u~ s s~ g", 5),
}
_COLORS = ("lc", "nlc", "full")
_COLOR_MATRIX_GATES = {
    "matrix_scope_complete",
    "correctness",
    "per_point_selector_patterns",
    "builtin_ufo_topology_parity",
    "smoke_under_five_minutes",
    "n4_n5_reusable_compiled_at_most_1_25x_specialized",
    "nlc_full_each_at_least_7x",
    "lc_no_generation_regression",
    "per_process_geometric_mean_at_least_7x",
}
_CORRECTNESS_FIELDS = {
    "total",
    "resolved_f64",
    "resolved_precision32",
    "eager_resolved_sum",
    "compiled_resolved_sum",
    "passes",
}
_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class AcceptanceError(RuntimeError):
    """Raised when native-x86 acceptance evidence cannot be trusted."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AcceptanceError("evidence is not canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["content_identity"] = {
        "algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sha256": _canonical_sha256(body),
    }
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AcceptanceError(f"{label} must be an array")
    return value


def _require_true(value: object, *, label: str) -> None:
    if value is not True:
        raise AcceptanceError(f"{label} must be true")


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AcceptanceError(f"{label} must be a positive integer")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AcceptanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_content_identity(payload: Mapping[str, object], *, label: str) -> None:
    identity = _mapping(
        payload.get("content_identity"),
        label=f"{label}.content_identity",
    )
    if identity.get("algorithm") != CONTENT_IDENTITY_ALGORITHM:
        raise AcceptanceError(f"{label} has the wrong content-identity algorithm")
    body = dict(payload)
    body.pop("content_identity")
    if identity.get("sha256") != _canonical_sha256(body):
        raise AcceptanceError(f"{label} content identity is invalid")


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field) for field in _STABLE_STAT_FIELDS
    )


def _checked_file_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    """Read one stable regular file through a no-follow descriptor."""

    try:
        parent = path.parent.expanduser().resolve(strict=True)
    except OSError as error:
        raise AcceptanceError(f"cannot resolve {label} parent: {path}") from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        namespace_before = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(namespace_before.st_mode):
            raise AcceptanceError(f"{label} is not a regular file: {path}")
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        opened_before = os.fstat(descriptor)
        if not _same_stat(namespace_before, opened_before):
            raise AcceptanceError(f"{label} changed while it was opened: {path}")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            while block := stream.read(HASH_CHUNK_BYTES):
                digest.update(block)
                blocks.append(block)
                byte_count += len(block)
            opened_after = os.fstat(stream.fileno())
        namespace_after = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise AcceptanceError(f"cannot authenticate {label}: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)
    if (
        byte_count != opened_before.st_size
        or not _same_stat(opened_before, opened_after)
        or not _same_stat(opened_after, namespace_after)
    ):
        raise AcceptanceError(f"{label} changed while it was hashed: {path}")
    return b"".join(blocks), {
        "path": str(parent / path.name),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _checked_file_identity(path: Path, *, label: str) -> dict[str, object]:
    return _checked_file_bytes(path, label=label)[1]


def _reject_json_constant(value: str) -> None:
    raise AcceptanceError(f"JSON contains a non-finite number: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def _checked_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    encoded, identity = _checked_file_bytes(path, label=label)
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"{label} is not strict UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"{label} must contain a JSON object")
    _canonical_json(payload)
    return payload, identity


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise AcceptanceError(f"cannot write acceptance evidence: {path}") from error


def _remove_stale_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise AcceptanceError(f"cannot inspect stale evidence: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise AcceptanceError(f"refusing non-regular stale evidence: {path}")
    try:
        path.unlink()
    except OSError as error:
        raise AcceptanceError(f"cannot remove stale evidence: {path}") from error


def _evidence_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    checkout = ROOT.resolve(strict=True)
    if root == Path(root.anchor) or root == checkout:
        raise AcceptanceError("refusing a broad evidence root")
    try:
        root.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise AcceptanceError("Arena evidence root must be outside the checkout")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _git_output(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AcceptanceError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _source_identity(
    *,
    expected_revision: str,
    expected_workspace: Path,
) -> dict[str, object]:
    if REVISION_PATTERN.fullmatch(expected_revision) is None:
        raise AcceptanceError("expected revision must be a full lowercase Git SHA")
    try:
        workspace = expected_workspace.expanduser().resolve(strict=True)
        checkout = ROOT.resolve(strict=True)
    except OSError as error:
        raise AcceptanceError("expected source workspace is unavailable") from error
    if workspace != checkout:
        raise AcceptanceError("acceptance helper is executing from another workspace")
    revision = _git_output(("rev-parse", "--verify", "HEAD"))
    status = _git_output(("status", "--porcelain=v1", "--untracked-files=all"))
    if revision != expected_revision or status:
        raise AcceptanceError("candidate checkout is not the clean dispatched revision")
    return {
        "checkout": str(checkout),
        "revision": revision,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _ensure_source_only_python() -> None:
    try:
        source_only_bytecode_policy()
    except RuntimeEvidenceError as error:
        raise AcceptanceError(
            "source-only Python bootstrap did not complete before authentication"
        ) from error


def _establish_preimport_runtime_identity() -> None:
    candidates = []
    for entry in sys.path:
        package_root = Path(entry) / "pyamplicol"
        if not package_root.is_dir():
            continue
        if any(
            path.is_file()
            and path.name.startswith("_rusticol")
            and path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
            for path in package_root.iterdir()
        ):
            resolved = package_root.resolve(strict=True)
            if resolved not in candidates:
                candidates.append(resolved)
    if len(candidates) != 1:
        raise AcceptanceError(
            "exactly one native pyamplicol package must be importable before audit"
        )
    try:
        native = native_extension_in_package(candidates[0])
        preimport_python_runtime_identity(
            (candidates[0],),
            native_extension=native,
        )
    except RuntimeEvidenceError as error:
        raise AcceptanceError(str(error)) from error


def _wheel_build_info(encoded: bytes) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            matches = [
                info
                for info in archive.infolist()
                if info.filename == "pyamplicol/_build_info.json"
            ]
            if len(matches) != 1:
                raise AcceptanceError(
                    "candidate wheel must contain one pyamplicol/_build_info.json"
                )
            raw = archive.read(matches[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise AcceptanceError("candidate wheel is not a valid wheel archive") from error
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError("candidate wheel build info is invalid JSON") from error
    if not isinstance(value, dict):
        raise AcceptanceError("candidate wheel build info is not an object")
    return value


def _runtime_identity(
    source: Mapping[str, object],
) -> dict[str, object]:
    try:
        pyamplicol = importlib.import_module("pyamplicol")
        native = importlib.import_module("pyamplicol._rusticol")
        versions = importlib.import_module("pyamplicol._internal.versions")
    except ImportError as error:
        raise AcceptanceError("candidate native runtime is unavailable") from error
    package_roots = tuple(
        Path(str(path)).expanduser().resolve(strict=True)
        for path in pyamplicol.__path__
    )
    candidate_root = (ROOT / ".venv").resolve(strict=True)
    if len(package_roots) != 1 or not package_roots[0].is_relative_to(candidate_root):
        raise AcceptanceError("pyamplicol was not imported from the candidate venv")
    native_path_raw = getattr(native, "__file__", None)
    if not isinstance(native_path_raw, str):
        raise AcceptanceError("candidate native extension has no path")
    native_path = Path(native_path_raw).expanduser().resolve(strict=True)
    if native_path.parent != package_roots[0]:
        raise AcceptanceError("native extension is outside the candidate package")
    target = str(native.target_info().triple)
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or "x86_64" not in sysconfig.get_platform()
        or target != EXPECTED_TARGET
    ):
        raise AcceptanceError("Arena acceptance requires native Linux x86_64")
    native_inputs = native.native_build_inputs_sha256()
    _require_sha256(native_inputs, label="native build-input identity")
    build_info = versions._active_build_info()
    if not isinstance(build_info, Mapping):
        raise AcceptanceError("candidate exposes no strict build provenance")
    try:
        build_checkout = (
            Path(str(build_info.get("source_checkout")))
            .expanduser()
            .resolve(strict=True)
        )
    except OSError as error:
        raise AcceptanceError("candidate build checkout is unavailable") from error
    if (
        build_info.get("source_revision") != source.get("revision")
        or build_checkout != ROOT.resolve(strict=True)
        or build_info.get("native_build_inputs_sha256") != native_inputs
        or build_info.get("publishable") is not False
        or build_info.get("selftest_fixture_bootstrap") is not False
    ):
        raise AcceptanceError("candidate build provenance is not exact")
    artifact_directory = (ROOT / ".artifacts" / "candidate").resolve(strict=True)
    wheels = tuple(sorted(artifact_directory.glob("*.whl")))
    if len(wheels) != 1:
        raise AcceptanceError("candidate build must produce exactly one wheel")
    wheel_bytes, wheel_identity = _checked_file_bytes(
        wheels[0],
        label="candidate wheel",
    )
    wheel_build_info = _wheel_build_info(wheel_bytes)
    if wheel_build_info != dict(build_info):
        raise AcceptanceError(
            "candidate wheel and installed runtime expose different build provenance"
        )
    preimport = established_preimport_runtime_identity()
    try:
        loaded_policy = loaded_pyamplicol_origin_policy(
            package_roots,
            native_extension=native_path,
            expected_package_identity=preimport["python_package_tree"],
            expected_native_identity=preimport["native_extension"],
        )
    except RuntimeEvidenceError as error:
        raise AcceptanceError(str(error)) from error
    interpreter_identity = _checked_file_identity(
        Path(sys.executable).resolve(strict=True),
        label="candidate interpreter",
    )
    return {
        "kind": "pyamplicol-arena-native-x86-runtime-identity",
        "schema_version": 1,
        "interpreter": {
            **interpreter_identity,
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_platform": sysconfig.get_platform(),
            "target": target,
        },
        "preimport_runtime_identity": preimport,
        "package": {
            "roots": [str(path) for path in package_roots],
            "version": str(getattr(pyamplicol, "__version__", "")),
        },
        "loaded_module_origin_policy": loaded_policy,
        "native_extension": {
            **_checked_file_identity(native_path, label="candidate native extension"),
            "target": target,
            "build_inputs_sha256": native_inputs,
        },
        "active_build_info": {
            "payload": dict(build_info),
            "canonical_sha256": _canonical_sha256(build_info),
        },
        "candidate_wheel": {
            **wheel_identity,
            "build_info_canonical_sha256": _canonical_sha256(wheel_build_info),
            "matches_active_build_info": True,
        },
        "passes": True,
    }


def _stable_runtime_identity(value: Mapping[str, object]) -> dict[str, object]:
    stable = dict(value)
    raw_policy = stable.get("loaded_module_origin_policy")
    if isinstance(raw_policy, Mapping):
        policy = dict(raw_policy)
        for field in (
            "observed_module_count",
            "observations",
            "observations_sha256",
        ):
            policy.pop(field, None)
        stable["loaded_module_origin_policy"] = policy
    return stable


def _runtime_identity_sha256(value: Mapping[str, object]) -> str:
    return _canonical_sha256(_stable_runtime_identity(value))


def _cross_process_runtime_binding(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Normalize the byte identities shared by helper and gate processes."""

    preimport = value.get("preimport_runtime_identity")
    if isinstance(preimport, Mapping):
        package_tree = _mapping(
            preimport.get("python_package_tree"),
            label=f"{label}.preimport_runtime_identity.python_package_tree",
        )
    else:
        package_tree = _mapping(
            value.get("python_package_tree"),
            label=f"{label}.python_package_tree",
        )
    interpreter = _mapping(
        value.get("interpreter"),
        label=f"{label}.interpreter",
    )
    native = _mapping(
        value.get("native_extension"),
        label=f"{label}.native_extension",
    )
    active = _mapping(
        value.get("active_build_info"),
        label=f"{label}.active_build_info",
    )
    package = _mapping(value.get("package"), label=f"{label}.package")
    return {
        "interpreter": {
            name: interpreter.get(name)
            for name in (
                "path",
                "size_bytes",
                "sha256",
                "python_version",
                "implementation",
            )
        },
        "python_package_tree": dict(package_tree),
        "native_extension": {
            name: native.get(name)
            for name in ("path", "size_bytes", "sha256", "build_inputs_sha256")
        },
        "active_build_info": dict(active),
        "package_version": package.get("version"),
    }


def _require_gate_provenance(
    payload: Mapping[str, object],
    *,
    label: str,
    expected_revision: str,
    expected_workspace: Path,
    expected_runtime: Mapping[str, object] | None = None,
) -> None:
    source = _mapping(payload.get("source_identity"), label=f"{label}.source_identity")
    source_postflight = _mapping(
        payload.get("source_identity_postflight"),
        label=f"{label}.source_identity_postflight",
    )
    _require_true(payload.get("source_identity_match"), label=f"{label}.source match")
    expected_source = {
        "checkout": str(expected_workspace.resolve(strict=True)),
        "revision": expected_revision,
        "dirty": False,
        "untracked_files_checked": True,
    }
    if dict(source) != expected_source or dict(source_postflight) != expected_source:
        raise AcceptanceError(f"{label} source identity is not exact")
    runtime = _mapping(
        payload.get("runtime_identity"),
        label=f"{label}.runtime_identity",
    )
    runtime_postflight = _mapping(
        payload.get("runtime_identity_postflight"),
        label=f"{label}.runtime_identity_postflight",
    )
    _require_true(payload.get("runtime_identity_match"), label=f"{label}.runtime match")
    if _stable_runtime_identity(runtime) != _stable_runtime_identity(
        runtime_postflight
    ):
        raise AcceptanceError(f"{label} runtime identity changed")
    if expected_runtime is not None and _cross_process_runtime_binding(
        runtime,
        label=f"{label}.runtime_identity",
    ) != _cross_process_runtime_binding(
        expected_runtime,
        label="acceptance runtime identity",
    ):
        raise AcceptanceError(f"{label} did not use the preflight candidate runtime")
    active = _mapping(
        runtime.get("active_build_info"),
        label=f"{label}.runtime_identity.active_build_info",
    )
    build_info = _mapping(
        active.get("payload"),
        label=f"{label}.runtime_identity.active_build_info.payload",
    )
    if build_info.get("source_revision") != expected_revision or active.get(
        "canonical_sha256"
    ) != _canonical_sha256(build_info):
        raise AcceptanceError(f"{label} runtime build provenance is invalid")
    script = _mapping(
        payload.get("gate_script_identity"),
        label=f"{label}.gate_script_identity",
    )
    script_postflight = _mapping(
        payload.get("gate_script_identity_postflight"),
        label=f"{label}.gate_script_identity_postflight",
    )
    _require_true(
        payload.get("gate_script_identity_match"),
        label=f"{label}.gate script match",
    )
    if dict(script) != dict(script_postflight):
        raise AcceptanceError(f"{label} gate script identity changed")
    _require_sha256(script.get("sha256"), label=f"{label}.gate script SHA-256")


def _require_comparison(value: object, *, label: str) -> None:
    comparison = _mapping(value, label=label)
    _require_true(comparison.get("passes"), label=f"{label}.passes")
    if (
        "failing_value_count" in comparison
        and comparison.get("failing_value_count") != 0
    ):
        raise AcceptanceError(f"{label} has failing numerical values")
    counts = [
        comparison.get(name)
        for name in ("value_count", "component_count")
        if name in comparison
    ]
    if counts and not any(
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        for count in counts
    ):
        raise AcceptanceError(f"{label} comparison is empty")


def _require_comparison_mapping(
    value: object,
    *,
    label: str,
    expected_names: set[str],
) -> None:
    comparisons = _mapping(value, label=label)
    if set(comparisons) != expected_names:
        raise AcceptanceError(f"{label} has incomplete comparison coverage")
    for name, comparison in comparisons.items():
        _require_comparison(comparison, label=f"{label}.{name}")


def _validate_all_jit(
    payload: Mapping[str, object],
    *,
    expected_revision: str,
    expected_workspace: Path,
    point_count: int,
    expected_runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    label = "all-JIT evidence"
    _require_content_identity(payload, label=label)
    if (
        payload.get("kind") != ALL_JIT_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "ok"
    ):
        raise AcceptanceError(f"{label} has the wrong kind, schema, or status")
    _require_true(payload.get("passes"), label=f"{label}.passes")
    if payload.get("failures") != []:
        raise AcceptanceError(f"{label} contains failures")
    _require_gate_provenance(
        payload,
        label=label,
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
        expected_runtime=expected_runtime,
    )
    request = _mapping(payload.get("request"), label=f"{label}.request")
    if (
        request.get("process") != "g g > g g"
        or request.get("jit_optimization_levels") != [0, 1, 2, 3]
        or request.get("point_count") != point_count
        or request.get("point_seeds") != list(_ALL_JIT_POINT_SEEDS[:point_count])
        or request.get("relative_tolerance") != 1.0e-12
        or request.get("absolute_tolerance") != 1.0e-15
        or request.get("memory_limit_gib") != EXPECTED_MEMORY_LIMIT_GIB
    ):
        raise AcceptanceError(f"{label} request is not the required request")
    momenta = _mapping(
        request.get("momenta_identity"),
        label=f"{label}.request.momenta_identity",
    )
    if (
        momenta.get("algorithm") != "sha256-float-hex-momenta-v1"
        or momenta.get("point_count") != point_count
        or momenta.get("external_particle_count") != 4
    ):
        raise AcceptanceError(f"{label} momenta identity is invalid")
    _require_sha256(momenta.get("sha256"), label=f"{label}.momenta SHA-256")
    _require_nonempty_string(
        request.get("mandatory_watchdog_command"),
        label=f"{label}.mandatory_watchdog_command",
    )
    levels = _mapping(payload.get("levels"), label=f"{label}.levels")
    if set(levels) != set(_ALL_JIT_LEVELS):
        raise AcceptanceError(f"{label} does not cover JIT O0 through O3")
    for name, optimization_level in _ALL_JIT_LEVELS.items():
        lane = _mapping(levels[name], label=f"{label}.levels.{name}")
        _require_true(lane.get("passes"), label=f"{label}.levels.{name}.passes")
        expected_configuration = {
            "process": "g g > g g",
            "color_accuracy": "lc",
            "lc_flow_layout": "topology-replay",
            "backend": "jit",
            "execution_mode": "compiled",
            "jit_optimization_level": optimization_level,
        }
        if lane.get("configuration") != expected_configuration:
            raise AcceptanceError(f"{label}.{name} configuration drifted")
        generation = _mapping(
            lane.get("generation"),
            label=f"{label}.{name}.generation",
        )
        worker = _mapping(
            generation.get("worker_provenance"),
            label=f"{label}.{name}.generation.worker_provenance",
        )
        if (
            worker.get("source_revision") != expected_revision
            or worker.get("postflight_identity_match") is not True
        ):
            raise AcceptanceError(f"{label}.{name} worker provenance drifted")
        artifact = _mapping(
            lane.get("artifact_identity"),
            label=f"{label}.{name}.artifact_identity",
        )
        _require_positive_int(
            artifact.get("payload_count"),
            label=f"{label}.{name}.artifact payload count",
        )
        arena_audit = _mapping(
            artifact.get("direct_arena_audit"),
            label=f"{label}.{name}.artifact direct Arena audit",
        )
        _require_true(
            arena_audit.get("passes"),
            label=f"{label}.{name}.artifact direct Arena audit passes",
        )
        numerical = _mapping(
            lane.get("numerical_validation"),
            label=f"{label}.{name}.numerical_validation",
        )
        _require_true(
            numerical.get("passes"),
            label=f"{label}.{name}.numerical_validation.passes",
        )
        if (
            numerical.get("execution_mode") != "compiled"
            or numerical.get("point_count") != point_count
            or numerical.get("point_seeds") != list(_ALL_JIT_POINT_SEEDS[:point_count])
        ):
            raise AcceptanceError(f"{label}.{name} numerical request drifted")
        _require_comparison_mapping(
            numerical.get("comparisons"),
            label=f"{label}.{name}.comparisons",
            expected_names=_ALL_JIT_COMPARISONS,
        )
        native_profile = _mapping(
            numerical.get("native_profile"),
            label=f"{label}.{name}.native_profile",
        )
        _require_true(
            native_profile.get("passes"),
            label=f"{label}.{name}.native_profile.passes",
        )
        if native_profile.get("execution_mode") != "compiled":
            raise AcceptanceError(f"{label}.{name} did not profile compiled execution")
        engine = _mapping(
            native_profile.get("direct_arena_engine_counter"),
            label=f"{label}.{name}.direct_arena_engine_counter",
        )
        calls = _mapping(
            native_profile.get("direct_arena_call_counter"),
            label=f"{label}.{name}.direct_arena_call_counter",
        )
        _require_positive_int(engine.get("value"), label=f"{label}.{name}.engines")
        _require_positive_int(calls.get("value"), label=f"{label}.{name}.calls")
    cross = _mapping(
        payload.get("cross_level_numerical_parity"),
        label=f"{label}.cross_level_numerical_parity",
    )
    _require_true(cross.get("passes"), label=f"{label}.cross-level passes")
    if cross.get("baseline_optimization_level") != 3:
        raise AcceptanceError(f"{label} cross-level baseline is not O3")
    cross_comparisons = _mapping(
        cross.get("comparisons"),
        label=f"{label}.cross-level comparisons",
    )
    if set(cross_comparisons) != {"o0_vs_o3", "o1_vs_o3", "o2_vs_o3"}:
        raise AcceptanceError(f"{label} cross-level coverage is incomplete")
    for name, comparisons in cross_comparisons.items():
        _require_comparison_mapping(
            comparisons,
            label=f"{label}.cross-level comparisons.{name}",
            expected_names=_CROSS_LEVEL_COMPARISONS,
        )
    return {
        "kind": ALL_JIT_KIND,
        "schema_version": SCHEMA_VERSION,
        "point_count": point_count,
        "levels": sorted(levels),
        "numerical_validation": True,
        "passes": True,
    }


def _validate_four_quark(
    payload: Mapping[str, object],
    *,
    expected_revision: str,
    expected_workspace: Path,
    point_count: int,
    expected_runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    label = "four-quark evidence"
    _require_content_identity(payload, label=label)
    if (
        payload.get("kind") != FOUR_QUARK_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "ok"
    ):
        raise AcceptanceError(f"{label} has the wrong kind, schema, or status")
    _require_true(payload.get("passes"), label=f"{label}.passes")
    if payload.get("failures") != []:
        raise AcceptanceError(f"{label} contains failures")
    _require_gate_provenance(
        payload,
        label=label,
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
        expected_runtime=expected_runtime,
    )
    request = _mapping(payload.get("request"), label=f"{label}.request")
    if (
        request.get("process") != "d d~ > u u~ s s~ c c~"
        or request.get("independent_quark_line_count") != 4
        or request.get("expected_external_pdgs") != [-4, -3, -2, -1, 1, 2, 3, 4]
        or request.get("max_quark_lines") != 4
        or request.get("point_count") != point_count
        or request.get("point_seeds") != list(_FOUR_QUARK_POINT_SEEDS[:point_count])
        or request.get("relative_tolerance") != 1.0e-12
        or request.get("absolute_tolerance") != 1.0e-300
        or request.get("memory_limit_gib") != EXPECTED_MEMORY_LIMIT_GIB
    ):
        raise AcceptanceError(f"{label} request is not the required request")
    momenta = _mapping(
        request.get("momenta_identity"),
        label=f"{label}.request.momenta_identity",
    )
    if (
        momenta.get("algorithm") != "sha256-float-hex-momenta-v1"
        or momenta.get("point_count") != point_count
        or momenta.get("external_particle_count") != 8
    ):
        raise AcceptanceError(f"{label} momenta identity is invalid")
    _require_sha256(momenta.get("sha256"), label=f"{label}.momenta SHA-256")
    _require_nonempty_string(
        request.get("mandatory_watchdog_command"),
        label=f"{label}.mandatory_watchdog_command",
    )
    rejections = _list(
        payload.get("invalid_union_configurations"),
        label=f"{label}.invalid_union_configurations",
    )
    rejected_colors: set[str] = set()
    for index, raw in enumerate(rejections):
        rejection = _mapping(
            raw,
            label=f"{label}.invalid_union_configurations[{index}]",
        )
        color = rejection.get("color_accuracy")
        if (
            color not in {"nlc", "full"}
            or rejection.get("lc_flow_layout") != "all-flow-union"
            or rejection.get("rejected") is not True
            or not isinstance(rejection.get("message"), str)
            or "all-flow-union" not in str(rejection.get("message"))
        ):
            raise AcceptanceError(f"{label} invalid-union rejection is incomplete")
        rejected_colors.add(str(color))
    if rejected_colors != {"nlc", "full"} or len(rejections) != 2:
        raise AcceptanceError(f"{label} did not reject both invalid union requests")
    lanes = _mapping(payload.get("lanes"), label=f"{label}.lanes")
    if set(lanes) != set(_FOUR_QUARK_LANES):
        raise AcceptanceError(f"{label} lane coverage is incomplete")
    for name, (color, layout, contracted) in _FOUR_QUARK_LANES.items():
        lane = _mapping(lanes[name], label=f"{label}.lanes.{name}")
        _require_true(lane.get("passes"), label=f"{label}.{name}.passes")
        expected_configuration = {
            "process": "d d~ > u u~ s s~ c c~",
            "max_quark_lines": 4,
            "color_accuracy": color,
            "lc_flow_layout": layout,
            "contracted_color": contracted,
            "backend": "jit",
            "execution_mode": "compiled",
            "jit_optimization_level": 3,
        }
        if lane.get("configuration") != expected_configuration:
            raise AcceptanceError(f"{label}.{name} configuration drifted")
        generation = _mapping(
            lane.get("generation"),
            label=f"{label}.{name}.generation",
        )
        worker = _mapping(
            generation.get("worker_provenance"),
            label=f"{label}.{name}.generation.worker_provenance",
        )
        if (
            worker.get("source_revision") != expected_revision
            or worker.get("postflight_identity_match") is not True
        ):
            raise AcceptanceError(f"{label}.{name} worker provenance drifted")
        artifact = _mapping(
            lane.get("artifact_identity"),
            label=f"{label}.{name}.artifact_identity",
        )
        _require_positive_int(
            artifact.get("payload_count"),
            label=f"{label}.{name}.artifact payload count",
        )
        arena_audit = _mapping(
            artifact.get("direct_arena_audit"),
            label=f"{label}.{name}.artifact direct Arena audit",
        )
        _require_true(
            arena_audit.get("passes"),
            label=f"{label}.{name}.artifact direct Arena audit passes",
        )
        numerical = _mapping(
            lane.get("numerical_validation"),
            label=f"{label}.{name}.numerical_validation",
        )
        _require_true(
            numerical.get("passes"),
            label=f"{label}.{name}.numerical_validation.passes",
        )
        if numerical.get("point_count") != point_count or numerical.get(
            "point_seeds"
        ) != list(_FOUR_QUARK_POINT_SEEDS[:point_count]):
            raise AcceptanceError(f"{label}.{name} numerical request drifted")
        expected_comparisons = set(_FOUR_QUARK_BASE_COMPARISONS)
        if not contracted:
            expected_comparisons.add("color_selector_evaluate_vs_resolved_total")
        _require_comparison_mapping(
            numerical.get("comparisons"),
            label=f"{label}.{name}.comparisons",
            expected_names=expected_comparisons,
        )
        execution = _mapping(
            numerical.get("runtime_execution"),
            label=f"{label}.{name}.runtime_execution",
        )
        if execution.get("execution_mode") != "compiled":
            raise AcceptanceError(f"{label}.{name} did not execute compiled mode")
        profiles = _mapping(
            execution.get("direct_arena_profiles"),
            label=f"{label}.{name}.direct_arena_profiles",
        )
        expected_profiles = {"complete", "helicity_selector"}
        if not contracted:
            expected_profiles.update({"color_selector", "combined_selector"})
        if set(profiles) != expected_profiles:
            raise AcceptanceError(f"{label}.{name} profile coverage is incomplete")
        for profile_name, raw_profile in profiles.items():
            profile = _mapping(
                raw_profile,
                label=f"{label}.{name}.profiles.{profile_name}",
            )
            _require_true(
                profile.get("passes"),
                label=f"{label}.{name}.profiles.{profile_name}.passes",
            )
            _require_positive_int(
                profile.get("compiled_direct_arena_engine_count"),
                label=f"{label}.{name}.{profile_name}.engines",
            )
            _require_positive_int(
                profile.get("compiled_direct_arena_call_count"),
                label=f"{label}.{name}.{profile_name}.calls",
            )
            if profile.get("legacy_boundary_component_total") != 0:
                raise AcceptanceError(
                    f"{label}.{name}.{profile_name} used a legacy boundary"
                )
    cross = _mapping(
        payload.get("lc_cross_layout_parity"),
        label=f"{label}.lc_cross_layout_parity",
    )
    _require_true(cross.get("passes"), label=f"{label}.LC cross-layout passes")
    _require_true(
        cross.get("physical_axes_match"),
        label=f"{label}.LC physical axes match",
    )
    _require_comparison_mapping(
        cross.get("comparisons"),
        label=f"{label}.LC cross-layout comparisons",
        expected_names=_CROSS_LC_COMPARISONS,
    )
    return {
        "kind": FOUR_QUARK_KIND,
        "schema_version": SCHEMA_VERSION,
        "point_count": point_count,
        "independent_quark_line_count": 4,
        "lanes": sorted(lanes),
        "invalid_union_requests_rejected": sorted(rejected_colors),
        "numerical_validation": True,
        "passes": True,
    }


def _require_pointwise_items(value: object, *, label: str) -> None:
    items = _list(value, label=label)
    if not items:
        raise AcceptanceError(f"{label} must not be empty")
    for index, item in enumerate(items):
        comparison = _mapping(item, label=f"{label}[{index}]")
        _require_true(comparison.get("passes"), label=f"{label}[{index}].passes")


def _validate_correctness(
    value: object,
    *,
    label: str,
    require_specialized: bool,
) -> None:
    correctness = _mapping(value, label=label)
    expected_fields = set(_CORRECTNESS_FIELDS)
    if require_specialized:
        expected_fields.add("specialized_compiled")
    if set(correctness) != expected_fields:
        raise AcceptanceError(f"{label} has incomplete numerical fields")
    _require_true(correctness.get("passes"), label=f"{label}.passes")
    for name in ("total", "eager_resolved_sum", "compiled_resolved_sum"):
        _require_pointwise_items(correctness.get(name), label=f"{label}.{name}")
    for name in ("resolved_f64", "resolved_precision32"):
        resolved = _mapping(correctness.get(name), label=f"{label}.{name}")
        _require_true(resolved.get("passes"), label=f"{label}.{name}.passes")
        _require_positive_int(
            resolved.get("point_count"),
            label=f"{label}.{name}.point_count",
        )
        _require_positive_int(
            resolved.get("component_count"),
            label=f"{label}.{name}.component_count",
        )
        for match_field in (
            "helicity_ids_match",
            "color_ids_match",
            "shape_matches",
        ):
            _require_true(
                resolved.get(match_field),
                label=f"{label}.{name}.{match_field}",
            )
    if require_specialized:
        _validate_correctness(
            correctness.get("specialized_compiled"),
            label=f"{label}.specialized_compiled",
            require_specialized=False,
        )


def _validate_color_matrix(
    payload: Mapping[str, object],
    *,
    expected_revision: str,
    expected_runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    label = "eager/compiled color matrix"
    if (
        payload.get("kind") != COLOR_MATRIX_KIND
        or payload.get("schema_version") != COLOR_MATRIX_SCHEMA_VERSION
        or payload.get("suite") != "smoke"
    ):
        raise AcceptanceError(f"{label} has the wrong kind, schema, or suite")
    _require_true(payload.get("complete"), label=f"{label}.complete")
    _require_true(payload.get("passes"), label=f"{label}.passes")
    if payload.get("source_revision") != expected_revision:
        raise AcceptanceError(f"{label} came from another source revision")
    if expected_runtime is not None:
        interpreter = _mapping(
            expected_runtime.get("interpreter"),
            label="acceptance runtime identity.interpreter",
        )
        observed_platform = payload.get("platform")
        if (
            payload.get("python") != interpreter.get("python_version")
            or not isinstance(observed_platform, str)
            or "Linux" not in observed_platform
            or "x86_64" not in observed_platform
        ):
            raise AcceptanceError(f"{label} did not run on the candidate x86 host")
    configuration = _mapping(
        payload.get("configuration"),
        label=f"{label}.configuration",
    )
    expected_configuration = {
        "batch_sizes": [128, 1024],
        "colors": ["lc", "nlc", "full"],
        "generation_timeout": 300.0,
        "memory_limit_gib": 30.0,
        "minimum_samples": 5,
        "target_runtime": 5.0,
        "selector_target_runtime": 1.0,
        "selector_seed": 0xC0FFEE,
    }
    if dict(configuration) != expected_configuration:
        raise AcceptanceError(f"{label} request configuration drifted")
    gates = _mapping(payload.get("gates"), label=f"{label}.gates")
    if set(gates) != _COLOR_MATRIX_GATES:
        raise AcceptanceError(f"{label} hard-gate coverage is incomplete")
    for name, value in gates.items():
        _require_true(value, label=f"{label}.gates.{name}")
    records = _list(payload.get("records"), label=f"{label}.records")
    expected_cells = {
        (case, "built-in", color) for case in _SMOKE_CASES for color in _COLORS
    }
    observed_cells: set[tuple[str, str, str]] = set()
    workload_count = 0
    for index, raw_record in enumerate(records):
        record_label = f"{label}.records[{index}]"
        record = _mapping(raw_record, label=record_label)
        case = _mapping(record.get("case"), label=f"{record_label}.case")
        case_key = case.get("key")
        model = record.get("model")
        color = record.get("color")
        if not all(isinstance(value, str) for value in (case_key, model, color)):
            raise AcceptanceError(f"{record_label} cell identity is invalid")
        cell = (str(case_key), str(model), str(color))
        if cell not in expected_cells or cell in observed_cells:
            raise AcceptanceError(f"{record_label} is unexpected or duplicated")
        observed_cells.add(cell)
        expected_process, expected_n_final = _SMOKE_CASES[str(case_key)]
        if (
            case.get("process") != expected_process
            or case.get("n_final") != expected_n_final
            or case.get("smoke") is not True
        ):
            raise AcceptanceError(f"{record_label} process metadata drifted")
        _require_nonempty_string(
            record.get("process_id"),
            label=f"{record_label}.process_id",
        )
        _require_true(
            record.get("compiled_generation_under_hard_limit"),
            label=f"{record_label}.compiled_generation_under_hard_limit",
        )
        generation = _mapping(
            record.get("generation"),
            label=f"{record_label}.generation",
        )
        if set(generation) != {"compiled", "eager"}:
            raise AcceptanceError(f"{record_label} generation coverage is incomplete")
        for mode, raw_generation in generation.items():
            mode_generation = _mapping(
                raw_generation,
                label=f"{record_label}.generation.{mode}",
            )
            core_seconds = mode_generation.get("core_phase_seconds")
            if (
                isinstance(core_seconds, bool)
                or not isinstance(core_seconds, (int, float))
                or float(core_seconds) <= 0
            ):
                raise AcceptanceError(
                    f"{record_label}.generation.{mode} has no core timing"
                )
        workloads = _list(
            record.get("workloads"),
            label=f"{record_label}.workloads",
        )
        expected_workloads = (
            {"single-flow-helicity-sum", "all-flow-single-helicity"}
            if color == "lc"
            else {"summed"}
        )
        if {workload.get("name") for workload in workloads} != expected_workloads:
            raise AcceptanceError(f"{record_label} workload scope is incomplete")
        if len(workloads) != len(expected_workloads):
            raise AcceptanceError(f"{record_label} contains duplicate workloads")
        for workload_index, raw_workload in enumerate(workloads):
            workload_count += 1
            workload_label = f"{record_label}.workloads[{workload_index}]"
            workload = _mapping(raw_workload, label=workload_label)
            _validate_correctness(
                workload.get("correctness"),
                label=f"{workload_label}.correctness",
                require_specialized=color == "lc",
            )
            profiles = _list(
                workload.get("profiles"),
                label=f"{workload_label}.profiles",
            )
            if {profile.get("batch_size") for profile in profiles} != {128, 1024}:
                raise AcceptanceError(f"{workload_label} batch coverage is incomplete")
            if len(profiles) != 2:
                raise AcceptanceError(f"{workload_label} contains duplicate batches")
            expected_profile_modes = {"compiled_complete", "eager_complete"}
            if color == "lc":
                expected_profile_modes.add("compiled_specialized")
            for profile_index, raw_profile in enumerate(profiles):
                profile_label = f"{workload_label}.profiles[{profile_index}]"
                profile = _mapping(raw_profile, label=profile_label)
                batch_size = profile.get("batch_size")
                for mode in expected_profile_modes:
                    timing = _mapping(
                        profile.get(mode),
                        label=f"{profile_label}.{mode}",
                    )
                    result = _mapping(
                        timing.get("result"),
                        label=f"{profile_label}.{mode}.result",
                    )
                    sample_count = _require_positive_int(
                        result.get("sample_count"),
                        label=f"{profile_label}.{mode}.result.sample_count",
                    )
                    wall = result.get("wall_time_per_point")
                    if (
                        sample_count < 1
                        or isinstance(wall, bool)
                        or not isinstance(wall, (int, float))
                        or not math.isfinite(float(wall))
                        or float(wall) < 0
                        or result.get("interrupted") is not False
                    ):
                        raise AcceptanceError(
                            f"{profile_label}.{mode} runtime profile is invalid"
                        )
                    effective = _mapping(
                        result.get("effective_config"),
                        label=f"{profile_label}.{mode}.result.effective_config",
                    )
                    if effective.get("batch_size") != batch_size:
                        raise AcceptanceError(
                            f"{profile_label}.{mode} profiled another batch size"
                        )
        selector_profiles = record.get("selector_pattern_profiles")
        if color == "lc":
            selectors = _mapping(
                selector_profiles,
                label=f"{record_label}.selector_pattern_profiles",
            )
            if set(selectors) != {"compiled", "eager"}:
                raise AcceptanceError(
                    f"{record_label} selector-profile coverage is incomplete"
                )
            for mode, raw_profile in selectors.items():
                profile = _mapping(
                    raw_profile,
                    label=f"{record_label}.selector_pattern_profiles.{mode}",
                )
                result = _mapping(
                    profile.get("result"),
                    label=f"{record_label}.selector_pattern_profiles.{mode}.result",
                )
                _require_true(
                    result.get("passes"),
                    label=(
                        f"{record_label}.selector_pattern_profiles.{mode}.result.passes"
                    ),
                )
        elif selector_profiles is not None:
            raise AcceptanceError(
                f"{record_label} contracted color has unexpected selector profiles"
            )
    if observed_cells != expected_cells or len(records) != len(expected_cells):
        raise AcceptanceError(f"{label} matrix cell scope is incomplete")
    return {
        "kind": COLOR_MATRIX_KIND,
        "schema_version": COLOR_MATRIX_SCHEMA_VERSION,
        "suite": "smoke",
        "cells": [
            {"case": case, "model": model, "color": color}
            for case, model, color in sorted(observed_cells)
        ],
        "cell_count": len(observed_cells),
        "workload_count": workload_count,
        "hard_gates": sorted(gates),
        "numerical_validation": True,
        "passes": True,
    }


def _validate_preflight(
    payload: Mapping[str, object],
    *,
    expected_revision: str,
    expected_workspace: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    label = "runtime preflight"
    _require_content_identity(payload, label=label)
    if (
        payload.get("kind") != PREFLIGHT_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "ok"
    ):
        raise AcceptanceError(f"{label} has the wrong kind, schema, or status")
    _require_true(payload.get("passes"), label=f"{label}.passes")
    if payload.get("failures") != []:
        raise AcceptanceError(f"{label} contains failures")
    request = _mapping(payload.get("request"), label=f"{label}.request")
    expected_request = {
        "expected_revision": expected_revision,
        "expected_workspace": str(expected_workspace.resolve(strict=True)),
        "expected_target": EXPECTED_TARGET,
    }
    if dict(request) != expected_request:
        raise AcceptanceError(f"{label} request drifted")
    source = _mapping(payload.get("source_identity"), label=f"{label}.source_identity")
    source_postflight = _mapping(
        payload.get("source_identity_postflight"),
        label=f"{label}.source_identity_postflight",
    )
    _require_true(payload.get("source_identity_match"), label=f"{label}.source match")
    if dict(source) != dict(source_postflight):
        raise AcceptanceError(f"{label} source identity changed")
    if (
        source.get("revision") != expected_revision
        or source.get("checkout") != str(expected_workspace.resolve(strict=True))
        or source.get("dirty") is not False
        or source.get("untracked_files_checked") is not True
    ):
        raise AcceptanceError(f"{label} source identity is not exact")
    runtime = _mapping(
        payload.get("runtime_identity"),
        label=f"{label}.runtime_identity",
    )
    runtime_postflight = _mapping(
        payload.get("runtime_identity_postflight"),
        label=f"{label}.runtime_identity_postflight",
    )
    _require_true(payload.get("runtime_identity_match"), label=f"{label}.runtime match")
    if _stable_runtime_identity(runtime) != _stable_runtime_identity(
        runtime_postflight
    ):
        raise AcceptanceError(f"{label} runtime identity changed")
    if payload.get("stable_runtime_identity_sha256") != _runtime_identity_sha256(
        runtime
    ):
        raise AcceptanceError(f"{label} stable runtime identity digest is invalid")
    platform_identity = _mapping(
        runtime.get("platform"),
        label=f"{label}.runtime_identity.platform",
    )
    if platform_identity.get("target") != EXPECTED_TARGET:
        raise AcceptanceError(f"{label} has the wrong native target")
    return source, runtime


def _preflight(
    *,
    evidence_root: Path,
    expected_revision: str,
    expected_workspace: Path,
) -> dict[str, object]:
    root = _evidence_root(evidence_root)
    destination = root / PREFLIGHT_FILENAME
    _remove_stale_file(destination)
    _remove_stale_file(root / ACCEPTANCE_FILENAME)
    source = _source_identity(
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
    )
    runtime = _runtime_identity(source)
    source_postflight = _source_identity(
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
    )
    runtime_postflight = _runtime_identity(source_postflight)
    if source_postflight != source:
        raise AcceptanceError("source identity changed during native-x86 preflight")
    if _stable_runtime_identity(runtime_postflight) != _stable_runtime_identity(
        runtime
    ):
        raise AcceptanceError("runtime identity changed during native-x86 preflight")
    payload = _attach_content_identity(
        {
            "kind": PREFLIGHT_KIND,
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "passes": True,
            "request": {
                "expected_revision": expected_revision,
                "expected_workspace": str(expected_workspace.resolve(strict=True)),
                "expected_target": EXPECTED_TARGET,
            },
            "source_identity": source,
            "source_identity_postflight": source_postflight,
            "source_identity_match": True,
            "runtime_identity": runtime,
            "runtime_identity_postflight": runtime_postflight,
            "runtime_identity_match": True,
            "stable_runtime_identity_sha256": _runtime_identity_sha256(runtime),
            "failures": [],
        }
    )
    _write_json_atomic(destination, payload)
    return payload


def _audit(
    *,
    evidence_root: Path,
    expected_revision: str,
    expected_workspace: Path,
    point_count: int,
) -> dict[str, object]:
    root = _evidence_root(evidence_root)
    destination = root / ACCEPTANCE_FILENAME
    _remove_stale_file(destination)
    source = _source_identity(
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
    )
    runtime = _runtime_identity(source)

    preflight, preflight_file = _checked_json(
        root / PREFLIGHT_FILENAME,
        label="runtime preflight evidence",
    )
    preflight_source, preflight_runtime = _validate_preflight(
        preflight,
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
    )
    if dict(preflight_source) != source:
        raise AcceptanceError("source identity differs from the preflight")
    if _stable_runtime_identity(preflight_runtime) != _stable_runtime_identity(runtime):
        raise AcceptanceError("runtime identity differs from the preflight")

    all_jit, all_jit_file = _checked_json(
        root / ALL_JIT_RELATIVE_PATH,
        label="all-JIT evidence",
    )
    all_jit_summary = _validate_all_jit(
        all_jit,
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
        point_count=point_count,
        expected_runtime=runtime,
    )
    four_quark, four_quark_file = _checked_json(
        root / FOUR_QUARK_RELATIVE_PATH,
        label="four-quark evidence",
    )
    four_quark_summary = _validate_four_quark(
        four_quark,
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
        point_count=point_count,
        expected_runtime=runtime,
    )
    color_matrix, color_matrix_file = _checked_json(
        root / COLOR_MATRIX_RELATIVE_PATH,
        label="eager/compiled color matrix evidence",
    )
    color_matrix_summary = _validate_color_matrix(
        color_matrix,
        expected_revision=expected_revision,
        expected_runtime=runtime,
    )

    source_postflight = _source_identity(
        expected_revision=expected_revision,
        expected_workspace=expected_workspace,
    )
    runtime_postflight = _runtime_identity(source_postflight)
    if source_postflight != source:
        raise AcceptanceError("source identity changed during final evidence audit")
    if _stable_runtime_identity(runtime_postflight) != _stable_runtime_identity(
        runtime
    ):
        raise AcceptanceError("runtime identity changed during final evidence audit")
    payload = _attach_content_identity(
        {
            "kind": ACCEPTANCE_KIND,
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "passes": True,
            "request": {
                "expected_revision": expected_revision,
                "expected_workspace": str(expected_workspace.resolve(strict=True)),
                "expected_target": EXPECTED_TARGET,
                "point_count": point_count,
                "required_evidence": {
                    "runtime_preflight": PREFLIGHT_FILENAME,
                    "compiled_all_jit": ALL_JIT_RELATIVE_PATH.as_posix(),
                    "four_quark": FOUR_QUARK_RELATIVE_PATH.as_posix(),
                    "eager_compiled_color": COLOR_MATRIX_RELATIVE_PATH.as_posix(),
                },
            },
            "source_identity": {
                "preflight": dict(preflight_source),
                "audit": source,
                "postflight": source_postflight,
                "all_match": True,
            },
            "runtime_identity": {
                "stable_sha256": _runtime_identity_sha256(runtime),
                "preflight_stable_sha256": _runtime_identity_sha256(preflight_runtime),
                "audit": runtime,
                "postflight": runtime_postflight,
                "all_match": True,
            },
            "evidence": {
                "runtime_preflight": {
                    "file_identity": preflight_file,
                    "semantic_validation": {
                        "kind": PREFLIGHT_KIND,
                        "schema_version": SCHEMA_VERSION,
                        "passes": True,
                    },
                },
                "compiled_all_jit": {
                    "file_identity": all_jit_file,
                    "semantic_validation": all_jit_summary,
                },
                "four_quark": {
                    "file_identity": four_quark_file,
                    "semantic_validation": four_quark_summary,
                },
                "eager_compiled_color": {
                    "file_identity": color_matrix_file,
                    "semantic_validation": color_matrix_summary,
                },
            },
            "validation": {
                "exact_source_revision": True,
                "source_clean_preflight_and_postflight": True,
                "source_only_python_runtime": True,
                "native_linux_x86_64_target": True,
                "runtime_preimport_bytes_unchanged": True,
                "candidate_wheel_matches_loaded_runtime": True,
                "compiled_all_jit_numerical_gate": True,
                "four_quark_all_valid_color_lanes_numerical_gate": True,
                "eager_compiled_full_smoke_color_matrix_numerical_gate": True,
                "all_evidence_files_content_bound": True,
            },
            "failures": [],
        }
    )
    _write_json_atomic(destination, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("preflight", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--evidence-root", type=Path, required=True)
        child.add_argument("--expected-revision", required=True)
        child.add_argument("--expected-workspace", type=Path, required=True)
        if command == "audit":
            child.add_argument(
                "--point-count",
                type=int,
                choices=(2, 3),
                default=EXPECTED_POINT_COUNT,
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_source_only_python()
    _establish_preimport_runtime_identity()
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "preflight":
            payload = _preflight(
                evidence_root=arguments.evidence_root,
                expected_revision=arguments.expected_revision,
                expected_workspace=arguments.expected_workspace,
            )
        else:
            payload = _audit(
                evidence_root=arguments.evidence_root,
                expected_revision=arguments.expected_revision,
                expected_workspace=arguments.expected_workspace,
                point_count=arguments.point_count,
            )
    except (AcceptanceError, OSError, RuntimeError, ValueError) as error:
        print(f"arena-native-x86-acceptance: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
