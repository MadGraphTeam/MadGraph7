#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate compiled Direct-Arena execution for four independent quark lines.

The physical process is fixed to ``d d~ > u u~ s s~ c c~``.  A successful
run generates exactly these compiled/JIT/O3 artifacts:

* LC topology replay;
* LC all-flow union;
* NLC contracted color;
* full contracted color.

The gate also proves that NLC/full all-flow-union configurations fail closed,
evaluates two or three deterministic phase-space points in f64 and precision
32, checks resolved sums, compares the two LC layouts component by component
and through runtime selectors, and recursively authenticates every compiled
plane-arena descriptor.

Run the complete gate from a clean, exact-source candidate:

```
.venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  .venv/bin/python tools/developer/four_quark_compiled_gate.py \
  --output-root /private/tmp/pyamplicol-four-quark-compiled-gate
```

The outer watchdog is mandatory.  Generation workers are descendants of the
guarded process and intentionally do not start nested watchdogs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import tomllib
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_EXACT_REEXEC_MARKER = "PYAMPLICOL_EXACT_PYTHON_REEXEC"
_EXACT_IMPORT_PATHS = "PYAMPLICOL_EXACT_IMPORT_PATHS"


def _bootstrap_source_only_python() -> None:
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
    python_package_tree_identity,
    source_only_bytecode_policy,
)

PROCESS = "d d~ > u u~ s s~ c c~"
EXPECTED_EXTERNAL_PDGS = (-4, -3, -2, -1, 1, 2, 3, 4)
MAX_QUARK_LINES = 4
VALIDATION_SEED = 20260725
POINT_SEEDS = (443_041, 443_099, 443_137)
SQRT_S = 2_000.0
RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-300
COMPILED_PLANE_ARENA_CAPABILITY = "compiled-plane-arena-v1"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v3"
NATIVE_COMPILED_DIRECT_APPLICATION_ABI = (
    "pyamplicol-native-compiled-direct-application-v1"
)
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
SYMJIT_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
RESULT_KIND = "pyamplicol-four-quark-compiled-direct-arena-gate"
SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WATCHDOG_INVOCATION = (
    ".venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- "
    ".venv/bin/python tools/developer/four_quark_compiled_gate.py "
    "--output-root /private/tmp/pyamplicol-four-quark-compiled-gate"
)
_ZERO_PROFILE_COUNTERS = (
    "stage_input_copy_component_count",
    "stage_leaf_input_copy_component_count",
    "stage_evaluator_output_gather_component_count",
    "stage_output_assign_component_count",
    "amplitude_input_copy_component_count",
    "amplitude_leaf_input_copy_component_count",
    "amplitude_evaluator_output_gather_component_count",
    "amplitude_output_remap_component_count",
    "selector_gather_point_count",
    "selector_gather_bytes",
    "selector_scatter_value_count",
    "compiled_direct_arena_boundary_input_bytes",
    "compiled_direct_arena_boundary_current_output_bytes",
    "compiled_direct_arena_boundary_amplitude_output_bytes",
)
_ZERO_PROFILE_TIMES = (
    "stage_input_pack_time_s",
    "stage_leaf_input_pack_time_s",
    "stage_evaluator_output_gather_time_s",
    "output_assign_time_s",
    "amplitude_input_pack_time_s",
    "amplitude_leaf_input_pack_time_s",
    "amplitude_evaluator_output_gather_time_s",
    "amplitude_output_remap_time_s",
    "selector_gather_time_s",
    "selector_scatter_time_s",
)


class GateError(RuntimeError):
    """Raised when four-quark evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Lane:
    name: str
    color_accuracy: str
    lc_flow_layout: str
    contracted: bool


LANES = (
    Lane("lc-topology-replay", "lc", "topology-replay", False),
    Lane("lc-all-flow-union", "lc", "all-flow-union", False),
    Lane("nlc-contracted", "nlc", "topology-replay", True),
    Lane("full-contracted", "full", "topology-replay", True),
)


@dataclass(slots=True)
class EvaluationCapture:
    """Non-serializable values retained only for cross-lane comparisons."""

    physics_axes: dict[str, object]
    f64_total: tuple[complex, ...]
    exact_total: tuple[complex, ...]
    f64_components: tuple[complex, ...]
    exact_components: tuple[complex, ...]
    color_probe: tuple[complex, ...] | None
    helicity_probe: tuple[complex, ...]


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
        raise GateError("evidence is not canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _regular_file_identity(
    path: Path,
    *,
    aggregate: Any | None = None,
    tree_relative: bytes | None = None,
) -> tuple[int, str]:
    """Hash one stable regular file through one no-follow descriptor."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateError(f"identity target is not a regular file: {path}")
        if aggregate is not None:
            if tree_relative is None:
                raise GateError("tree hashing requires a relative path")
            aggregate.update(len(tree_relative).to_bytes(8, "big"))
            aggregate.update(tree_relative)
            aggregate.update(before.st_size.to_bytes(8, "big"))
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            while block := stream.read(HASH_CHUNK_BYTES):
                digest.update(block)
                if aggregate is not None:
                    aggregate.update(block)
                byte_count += len(block)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise GateError(
            f"cannot hash regular file through a checked fd: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if byte_count != before.st_size or any(
        getattr(before, name) != getattr(after, name) for name in stable_fields
    ):
        raise GateError(f"identity target changed while it was hashed: {path}")
    return byte_count, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _regular_file_identity(path)[1]


def _file_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError(f"cannot identify file: {path}") from error
    size, sha256 = _regular_file_identity(resolved)
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": sha256,
    }


def _tree_identity(path: Path) -> dict[str, object]:
    """Hash relative names, sizes, and contents for a relocatable tree ID."""

    try:
        root = path.expanduser().resolve(strict=True)
        candidates = list(root.rglob("*"))
    except OSError as error:
        raise GateError(f"cannot inspect identity tree: {path}") from error
    if not root.is_dir():
        raise GateError(f"identity target is not a directory: {root}")
    for candidate in candidates:
        if candidate.is_symlink():
            raise GateError(
                f"identity tree contains an unsupported symlink: {candidate}"
            )
    members = sorted(
        (candidate for candidate in candidates if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        for member in members:
            relative = member.relative_to(root).as_posix().encode("utf-8")
            size, _ = _regular_file_identity(
                member,
                aggregate=digest,
                tree_relative=relative,
            )
            size_bytes += size
    except OSError as error:
        raise GateError(f"cannot hash identity tree: {root}") from error
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(members),
        "size_bytes": size_bytes,
    }


def _python_package_tree_identity(path: Path) -> dict[str, object]:
    try:
        return python_package_tree_identity(path)
    except RuntimeEvidenceError as error:
        raise GateError(str(error)) from error


def _ensure_source_only_python() -> None:
    """Verify the stdlib-only bootstrap completed before repository imports."""

    try:
        source_only_bytecode_policy()
    except RuntimeEvidenceError as error:
        raise GateError(
            "source-only Python bootstrap did not complete before gate imports"
        ) from error


def _establish_preimport_runtime_identity() -> None:
    native_root = next(
        (
            candidate
            for entry in sys.path
            if (candidate := Path(entry) / "pyamplicol").is_dir()
            and any(
                path.name.startswith("_rusticol")
                and path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
                for path in candidate.iterdir()
            )
        ),
        None,
    )
    if native_root is None:
        raise GateError("candidate Python path has no native pyamplicol package")
    native = native_extension_in_package(native_root)
    preimport_python_runtime_identity((native_root,), native_extension=native)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object: {path}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GateError(f"{label} must be an array")
    return value


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
        raise GateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _source_identity() -> dict[str, object]:
    revision = _git_output(("rev-parse", "--verify", "HEAD"))
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise GateError("source checkout has no full Git revision")
    status = _git_output(("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise GateError(
            "source checkout is dirty; build and run the gate from one committed SHA"
        )
    return {
        "checkout": str(ROOT.resolve(strict=True)),
        "revision": revision,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _runtime_identity(source: Mapping[str, object]) -> dict[str, object]:
    try:
        pyamplicol = importlib.import_module("pyamplicol")
        native = importlib.import_module("pyamplicol._rusticol")
        versions = importlib.import_module("pyamplicol._internal.versions")
    except ImportError as error:
        raise GateError(
            "the candidate pyamplicol native runtime is unavailable"
        ) from error
    native_path_raw = getattr(native, "__file__", None)
    if not isinstance(native_path_raw, str):
        raise GateError("native extension has no filesystem path")
    native_digest_function = getattr(native, "native_build_inputs_sha256", None)
    if not callable(native_digest_function):
        raise GateError("native extension exposes no build-input digest")
    native_digest = native_digest_function()
    if (
        not isinstance(native_digest, str)
        or SHA256_PATTERN.fullmatch(native_digest) is None
    ):
        raise GateError("native extension build-input digest is invalid")
    build_info_function = getattr(versions, "_active_build_info", None)
    build_info = build_info_function() if callable(build_info_function) else None
    if not isinstance(build_info, Mapping):
        raise GateError("candidate exposes no strict build provenance")
    if build_info.get("source_revision") != source.get("revision"):
        raise GateError(
            "candidate native runtime was built from another source revision"
        )
    if build_info.get("native_build_inputs_sha256") != native_digest:
        raise GateError("candidate build info and native build inputs disagree")
    checkout = build_info.get("source_checkout")
    if not isinstance(checkout, str):
        raise GateError("candidate build info has no source checkout")
    try:
        bound_checkout = Path(checkout).expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError("candidate source checkout is unavailable") from error
    if bound_checkout != ROOT.resolve(strict=True):
        raise GateError("candidate was built from another source checkout")
    package_path_raw = getattr(pyamplicol, "__file__", None)
    if not isinstance(package_path_raw, str):
        raise GateError("pyamplicol package has no filesystem identity")
    package_roots = tuple(Path(str(path)) for path in pyamplicol.__path__)
    preimport_identity = established_preimport_runtime_identity()
    return {
        "interpreter": {
            **_file_identity(Path(sys.executable)),
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "package": {
            **_file_identity(Path(package_path_raw)),
            "version": str(getattr(pyamplicol, "__version__", "")),
        },
        "python_package_tree": _python_package_tree_identity(package_roots),
        "loaded_module_origin_policy": loaded_pyamplicol_origin_policy(
            package_roots,
            native_extension=Path(native_path_raw),
            expected_package_identity=preimport_identity["python_package_tree"],
            expected_native_identity=preimport_identity["native_extension"],
        ),
        "native_extension": {
            **_file_identity(Path(native_path_raw)),
            "build_inputs_sha256": native_digest,
        },
        "active_build_info": {
            "payload": dict(build_info),
            "canonical_sha256": _canonical_sha256(build_info),
        },
    }


def _stable_runtime_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Exclude only the monotonically growing loaded-module observation set."""

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


def _peak_rss_gib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported Unix platforms use KiB.
    divisor = float(1024**3 if platform.system() == "Darwin" else 1024**2)
    return raw / divisor


def _generation_worker(
    artifact: Path,
    lane: Lane,
    *,
    expected_revision: str,
    expected_runtime_sha256: str,
    expected_script_sha256: str,
) -> dict[str, object]:
    source = _source_identity()
    if source["revision"] != expected_revision:
        raise GateError("generation worker source revision changed")
    runtime = _runtime_identity(source)
    if _runtime_identity_sha256(runtime) != expected_runtime_sha256:
        raise GateError("generation worker candidate runtime identity changed")
    script = _file_identity(Path(__file__))
    if script["sha256"] != expected_script_sha256:
        raise GateError("generation worker gate script identity changed")

    from pyamplicol import Generator
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        OutputConfig,
        ProcessConfig,
        RunConfig,
    )

    run = RunConfig(
        action="generate",
        process=ProcessConfig(max_quark_lines=MAX_QUARK_LINES),
        color=ColorConfig(
            accuracy=lane.color_accuracy,
            lc_flow_layout=lane.lc_flow_layout,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=True,
                samples=2,
                seed=VALIDATION_SEED,
                relative_tolerance=RELATIVE_TOLERANCE,
                absolute_tolerance=ABSOLUTE_TOLERANCE,
                post_build_validation=True,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=3),
        ),
        output=OutputConfig(format="json", color="never", progress="off"),
    )
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    result = Generator(run).generate(PROCESS, artifact, mode="error")
    elapsed_wall = time.monotonic() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    source_postflight = _source_identity()
    runtime_postflight = _runtime_identity(source_postflight)
    script_postflight = _file_identity(Path(__file__))
    if source_postflight != source:
        raise GateError("generation worker source identity changed during generation")
    if _stable_runtime_identity(runtime_postflight) != _stable_runtime_identity(
        runtime
    ):
        raise GateError("generation worker runtime identity changed during generation")
    if script_postflight != script:
        raise GateError("generation worker script identity changed during generation")
    return {
        "lane": lane.name,
        "artifact": str(result.output),
        "worker_script": script,
        "worker_provenance": {
            "source_revision": source["revision"],
            "runtime_identity_sha256": _runtime_identity_sha256(runtime),
            "gate_script_sha256": script["sha256"],
            "postflight_identity_match": True,
        },
        "process_ids": [process.name for process in result.processes.requests],
        "generation": {
            "wall_seconds": elapsed_wall,
            "process_cpu_seconds": elapsed_cpu,
            "worker_peak_rss_gib": _peak_rss_gib(),
            "workers": 1,
            "validation_samples": 2,
            "validation_seed": VALIDATION_SEED,
            "post_build_validation": True,
        },
    }


def _worker_command(
    artifact: Path,
    lane: Lane,
    *,
    expected_revision: str,
    expected_runtime_sha256: str,
    expected_script_sha256: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(Path(__file__).resolve()),
        "--_generate-one",
        "--artifact",
        str(artifact),
        "--lane",
        lane.name,
        "--expected-revision",
        expected_revision,
        "--expected-runtime-sha256",
        expected_runtime_sha256,
        "--expected-script-sha256",
        expected_script_sha256,
    )


def _invoke_generation_worker(
    artifact: Path,
    lane: Lane,
    *,
    timeout: float,
    expected_revision: str,
    expected_runtime_sha256: str,
    expected_script_sha256: str,
) -> dict[str, object]:
    command = _worker_command(
        artifact,
        lane,
        expected_revision=expected_revision,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_script_sha256=expected_script_sha256,
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise GateError(
            f"{lane.name} generation exceeded {timeout:.1f} seconds"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(f"{lane.name} generation failed:\n{detail[-6000:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GateError(f"{lane.name} worker emitted invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("lane") != lane.name:
        raise GateError(f"{lane.name} worker returned the wrong lane")
    worker_script = _mapping(
        payload.get("worker_script"),
        label=f"{lane.name} worker script identity",
    )
    if worker_script.get("sha256") != expected_script_sha256:
        raise GateError(f"{lane.name} worker executed another gate script")
    worker_provenance = _mapping(
        payload.get("worker_provenance"),
        label=f"{lane.name} worker provenance",
    )
    if (
        worker_provenance.get("source_revision") != expected_revision
        or worker_provenance.get("runtime_identity_sha256") != expected_runtime_sha256
        or worker_provenance.get("gate_script_sha256") != expected_script_sha256
        or worker_provenance.get("postflight_identity_match") is not True
    ):
        raise GateError(f"{lane.name} worker provenance drifted")
    payload["orchestration_wall_seconds"] = time.monotonic() - started
    payload["command"] = {
        "argv": list(command),
        "canonical_sha256": _canonical_sha256(list(command)),
    }
    if completed.stderr.strip():
        payload["worker_stderr_sha256"] = hashlib.sha256(
            completed.stderr.encode()
        ).hexdigest()
    return payload


def _assert_union_rejections() -> list[dict[str, object]]:
    from pyamplicol.config import ColorConfig, ConfigurationError

    result: list[dict[str, object]] = []
    for accuracy in ("nlc", "full"):
        try:
            ColorConfig(accuracy=accuracy, lc_flow_layout="all-flow-union")
        except ConfigurationError as error:
            message = str(error)
            if "all-flow-union" not in message or "requires" not in message:
                raise GateError(
                    f"{accuracy} union rejection was not actionable: {message}"
                ) from error
            result.append(
                {
                    "color_accuracy": accuracy,
                    "lc_flow_layout": "all-flow-union",
                    "rejected": True,
                    "error_type": type(error).__name__,
                    "message": message,
                }
            )
        else:
            raise GateError(f"{accuracy} all-flow-union did not fail closed")
    return result


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise GateError(f"{label} must be artifact-relative")
    return value


def _source_evaluator_leaves(
    evaluator: Mapping[str, Any],
    *,
    label: str,
) -> list[Mapping[str, Any]]:
    kind = evaluator.get("kind")
    if kind == "symjit-application-evaluator":
        return [evaluator]
    if kind != "chunked-symbolica-evaluator":
        raise GateError(f"{label} has unsupported evaluator kind {kind!r}")
    chunks = _sequence(evaluator.get("chunks"), label=f"{label}.chunks")
    leaves = [
        _mapping(chunk, label=f"{label}.chunks[{index}]")
        for index, chunk in enumerate(chunks)
    ]
    if not leaves:
        raise GateError(f"{label} has no evaluator leaves")
    return leaves


def _audit_direct_descriptor(
    descriptor: Mapping[str, Any],
    *,
    stage: Mapping[str, Any],
    label: str,
) -> dict[str, int]:
    if descriptor.get("kind") != "compiled-plane-arena-stage":
        raise GateError(f"{label}.compiled_plane_arena has the wrong kind")
    if descriptor.get("schema_version") != 1:
        raise GateError(f"{label}.compiled_plane_arena has the wrong schema")
    expected_scalars = {
        "application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "source_application_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "split-complex-component-major",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
        "output_operation": "overwrite",
        "output_factor": "identity",
    }
    for name, expected in expected_scalars.items():
        if descriptor.get(name) != expected:
            raise GateError(f"{label}.compiled_plane_arena.{name} is not {expected!r}")
    inputs = _sequence(
        descriptor.get("input_bindings"),
        label=f"{label}.compiled_plane_arena.input_bindings",
    )
    input_indices = [
        _mapping(item, label=f"{label}.input_bindings[{index}]").get("parameter_index")
        for index, item in enumerate(inputs)
    ]
    if input_indices != list(range(len(inputs))):
        raise GateError(f"{label} Direct-Arena input bindings are not contiguous")
    outputs = _sequence(
        descriptor.get("output_bindings"),
        label=f"{label}.compiled_plane_arena.output_bindings",
    )
    output_indices = [
        _mapping(item, label=f"{label}.output_bindings[{index}]").get("output_index")
        for index, item in enumerate(outputs)
    ]
    if output_indices != list(range(len(outputs))):
        raise GateError(f"{label} Direct-Arena output bindings are not contiguous")
    evaluator = _mapping(stage.get("evaluator"), label=f"{label}.evaluator")
    source_leaves = _source_evaluator_leaves(evaluator, label=f"{label}.evaluator")
    direct_leaves = _sequence(
        descriptor.get("leaves"),
        label=f"{label}.compiled_plane_arena.leaves",
    )
    if len(direct_leaves) != len(source_leaves) or not direct_leaves:
        raise GateError(f"{label} Direct-Arena/source leaf counts disagree")
    output_cursor = 0
    for index, (direct_raw, source) in enumerate(
        zip(direct_leaves, source_leaves, strict=True)
    ):
        direct = _mapping(direct_raw, label=f"{label}.leaves[{index}]")
        source_path = _safe_relative_path(
            source.get("application_path"),
            label=f"{label}.source_leaves[{index}].application_path",
        )
        if (
            direct.get("application_path") != source_path
            or direct.get("source_application_abi") != SYMJIT_APPLICATION_ABI
            or source.get("application_abi") != SYMJIT_APPLICATION_ABI
            or source.get("runtime_capability") != SYMJIT_RUNTIME_CAPABILITY
        ):
            raise GateError(f"{label} Direct-Arena leaf identity is inconsistent")
        if (
            direct.get("optimization_level") != 3
            or source.get("optimization_level") != 3
        ):
            raise GateError(f"{label} Direct-Arena leaf is not JIT O3")
        if direct.get("direct_codegen_optimization_level") != 3:
            raise GateError(
                f"{label} Direct-Arena leaf does not authenticate fixed O3 "
                "direct code generation"
            )
        indices = _sequence(
            direct.get("input_indices"),
            label=f"{label}.leaves[{index}].input_indices",
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in indices
            )
            or len(indices) != direct.get("input_len")
            or len(indices) != source.get("input_len")
        ):
            raise GateError(f"{label} Direct-Arena leaf inputs are invalid")
        output_start = direct.get("output_start")
        output_stop = direct.get("output_stop")
        output_len = direct.get("output_len")
        if (
            output_start != output_cursor
            or isinstance(output_stop, bool)
            or not isinstance(output_stop, int)
            or isinstance(output_len, bool)
            or not isinstance(output_len, int)
            or output_stop - output_start != output_len
            or output_len != source.get("output_len")
        ):
            raise GateError(f"{label} Direct-Arena leaf outputs are invalid")
        output_cursor = output_stop
    if output_cursor != len(outputs) or output_cursor != stage.get("output_length"):
        raise GateError(f"{label} Direct-Arena leaves do not cover stage outputs")
    return {
        "leaf_count": len(direct_leaves),
        "input_binding_count": len(inputs),
        "output_binding_count": len(outputs),
    }


def _assert_model_parameter_not_direct(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        if value.get("kind") == "compiled-plane-arena-stage":
            raise GateError(
                f"{label} illegally uses a compiled plane DirectApplication"
            )
        if "compiled_plane_arena" in value or "native_direct_application" in value:
            raise GateError(f"{label} illegally carries DirectApplication metadata")
        if value.get("application_abi") in {
            COMPILED_PLANE_DIRECT_APPLICATION_ABI,
            NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
        }:
            raise GateError(f"{label} illegally uses the direct application ABI")
        for name, child in value.items():
            _assert_model_parameter_not_direct(child, label=f"{label}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_model_parameter_not_direct(child, label=f"{label}[{index}]")


def _walk_mappings(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        yield path, value
        for name, child in value.items():
            yield from _walk_mappings(child, path=(*path, str(name)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _walk_mappings(child, path=(*path, str(index)))


def _audit_execution_payload(payload: Mapping[str, Any]) -> dict[str, object]:
    capability_declarations = 0
    stage_set_count = 0
    stage_count = 0
    descriptor_count = 0
    leaf_count = 0
    model_parameter_slot_count = 0
    model_parameter_evaluator_count = 0
    for path, record in _walk_mappings(payload):
        label = ".".join(path) or "execution"
        capabilities = record.get("required_runtime_capabilities")
        if (
            isinstance(capabilities, Sequence)
            and not isinstance(capabilities, (str, bytes))
            and COMPILED_PLANE_ARENA_CAPABILITY in capabilities
        ):
            capability_declarations += 1
        if "model_parameter_evaluator" in record:
            model_parameter_slot_count += 1
            model_parameters = record["model_parameter_evaluator"]
            if model_parameters is not None:
                model_parameter_evaluator_count += 1
                _assert_model_parameter_not_direct(
                    model_parameters,
                    label=f"{label}.model_parameter_evaluator",
                )
        if record.get("kind") != "generic-dag-stage-evaluator-artifacts":
            continue
        stage_set_count += 1
        if record.get("runtime_available") is not True:
            raise GateError(f"{label} compiled stage runtime is unavailable")
        capabilities = _sequence(
            record.get("required_runtime_capabilities"),
            label=f"{label}.required_runtime_capabilities",
        )
        if COMPILED_PLANE_ARENA_CAPABILITY not in capabilities:
            raise GateError(f"{label} does not require compiled-plane-arena-v1")
        stages = list(_sequence(record.get("stages"), label=f"{label}.stages"))
        amplitude = _mapping(
            record.get("amplitude_stage"),
            label=f"{label}.amplitude_stage",
        )
        all_stages = (*stages, amplitude)
        if record.get("stage_count") != len(all_stages):
            raise GateError(f"{label}.stage_count does not cover all stages")
        for index, raw_stage in enumerate(all_stages):
            stage = _mapping(raw_stage, label=f"{label}.stages[{index}]")
            descriptor = _mapping(
                stage.get("compiled_plane_arena"),
                label=f"{label}.stages[{index}].compiled_plane_arena",
            )
            counts = _audit_direct_descriptor(
                descriptor,
                stage=stage,
                label=f"{label}.stages[{index}]",
            )
            stage_count += 1
            descriptor_count += 1
            leaf_count += counts["leaf_count"]
    if capability_declarations < 1:
        raise GateError("execution payload does not declare compiled-plane-arena-v1")
    if stage_set_count < 1 or descriptor_count < 1 or leaf_count < 1:
        raise GateError("execution payload has no complete Direct-Arena stage set")
    if model_parameter_slot_count < 1:
        raise GateError("execution payload has no model-parameter evaluator slot")
    return {
        "capability": COMPILED_PLANE_ARENA_CAPABILITY,
        "capability_declaration_count": capability_declarations,
        "stage_set_count": stage_set_count,
        "stage_count": stage_count,
        "descriptor_count": descriptor_count,
        "leaf_count": leaf_count,
        "model_parameter_slot_count": model_parameter_slot_count,
        "model_parameter_evaluator_count": model_parameter_evaluator_count,
        "model_parameter_direct_application_count": 0,
        "passes": True,
    }


def _artifact_identity_and_audit(
    artifact: Path,
    lane: Lane,
) -> tuple[dict[str, object], str]:
    manifest_path = artifact / "artifact.json"
    manifest = _json_object(manifest_path, label="artifact manifest")
    artifact_id = manifest.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or SHA256_PATTERN.fullmatch(artifact_id) is None
    ):
        raise GateError(f"{lane.name} artifact ID is invalid")
    processes = _sequence(manifest.get("processes"), label="artifact.processes")
    if len(processes) != 1:
        raise GateError(f"{lane.name} artifact must contain exactly one process")
    process = _mapping(processes[0], label="artifact.processes[0]")
    if process.get("expression") != PROCESS:
        raise GateError(f"{lane.name} artifact contains the wrong process")
    if sorted(process.get("external_pdgs", ())) != list(EXPECTED_EXTERNAL_PDGS):
        raise GateError(f"{lane.name} does not contain four quark/antiquark pairs")
    if process.get("color_accuracy") != lane.color_accuracy:
        raise GateError(f"{lane.name} artifact color accuracy is wrong")
    capabilities = _sequence(
        process.get("required_runtime_capabilities"),
        label="artifact.processes[0].required_runtime_capabilities",
    )
    if COMPILED_PLANE_ARENA_CAPABILITY not in capabilities:
        raise GateError(f"{lane.name} artifact lacks compiled-plane-arena-v1")
    process_id = process.get("id")
    if not isinstance(process_id, str) or not process_id:
        raise GateError(f"{lane.name} artifact process ID is invalid")

    generation = _mapping(
        _mapping(manifest.get("extensions"), label="artifact.extensions").get(
            "generation"
        ),
        label="artifact.extensions.generation",
    )
    concrete = _sequence(
        generation.get("concrete_processes"),
        label="artifact.extensions.generation.concrete_processes",
    )
    if len(concrete) != 1:
        raise GateError(f"{lane.name} generation metadata is ambiguous")
    filters = _mapping(
        _mapping(concrete[0], label="concrete process").get("filters"),
        label="concrete process filters",
    )
    if lane.color_accuracy == "lc":
        if filters.get("lc_flow_layout") != lane.lc_flow_layout:
            raise GateError(f"{lane.name} generation layout metadata is wrong")
    elif "lc_flow_layout" in filters:
        raise GateError(
            f"{lane.name} generation metadata illegally carries an LC layout"
        )

    try:
        effective = tomllib.loads(
            (artifact / "config" / "effective.toml").read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateError(f"{lane.name} effective configuration is invalid") from error
    expected_config = (
        (effective.get("process", {}).get("max_quark_lines"), MAX_QUARK_LINES),
        (effective.get("color", {}).get("accuracy"), lane.color_accuracy),
        (effective.get("color", {}).get("lc_flow_layout"), lane.lc_flow_layout),
        (effective.get("evaluator", {}).get("backend"), "jit"),
        (effective.get("evaluator", {}).get("execution_mode"), "compiled"),
        (
            effective.get("evaluator", {}).get("jit", {}).get("optimization_level"),
            3,
        ),
    )
    if any(observed != expected for observed, expected in expected_config):
        raise GateError(f"{lane.name} effective compiled/JIT/O3 configuration drifted")

    payloads = _sequence(manifest.get("payloads"), label="artifact.payloads")
    payload_bytes = 0
    for index, raw_payload in enumerate(payloads):
        payload = _mapping(raw_payload, label=f"artifact.payloads[{index}]")
        relative = _safe_relative_path(
            payload.get("path"),
            label=f"artifact.payloads[{index}].path",
        )
        expected_size = payload.get("size_bytes")
        expected_digest = payload.get("sha256")
        path = artifact / relative
        try:
            size = path.stat().st_size
        except OSError as error:
            raise GateError(f"{lane.name} payload is missing: {relative}") from error
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or size != expected_size
            or not isinstance(expected_digest, str)
            or _sha256_file(path) != expected_digest
        ):
            raise GateError(f"{lane.name} payload identity failed: {relative}")
        payload_bytes += size

    execution_path = artifact / "processes" / process_id / "execution.json"
    execution = _json_object(execution_path, label="process execution")
    if execution.get("color_accuracy") != lane.color_accuracy:
        raise GateError(f"{lane.name} execution color accuracy is wrong")
    arena_audit = _audit_execution_payload(execution)
    identity = {
        "path": str(artifact.resolve(strict=True)),
        "artifact_id": artifact_id,
        "manifest": _file_identity(manifest_path),
        "tree": _tree_identity(artifact),
        "payload_count": len(payloads),
        "payload_bytes": payload_bytes,
        "effective_configuration": {
            "max_quark_lines": MAX_QUARK_LINES,
            "color_accuracy": lane.color_accuracy,
            "lc_flow_layout": lane.lc_flow_layout,
            "backend": "jit",
            "execution_mode": "compiled",
            "jit_optimization_level": 3,
        },
        "direct_arena_audit": arena_audit,
    }
    return identity, process_id


def _deterministic_points(count: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    from pyamplicol.generation.phase_space import massive_rambo_final_state

    if count not in (2, 3):
        raise GateError("the numerical gate requires exactly two or three points")
    incoming = (
        (SQRT_S / 2.0, 0.0, 0.0, SQRT_S / 2.0),
        (SQRT_S / 2.0, 0.0, 0.0, -SQRT_S / 2.0),
    )
    return tuple(
        (
            *incoming,
            *massive_rambo_final_state(
                6,
                sqrt_s=SQRT_S,
                masses=(0.0,) * 6,
                seed=seed,
            ),
        )
        for seed in POINT_SEEDS[:count]
    )


def _as_complex(value: object) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GateError(
            f"evaluation returned a non-numeric value: {value!r}"
        ) from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise GateError("evaluation returned a non-finite value")
    return result


def _flatten_resolved(resolved: object) -> tuple[complex, ...]:
    return tuple(
        _as_complex(value)
        for point in resolved.values
        for helicity in point
        for value in helicity
    )


def _resolved_value_digest(resolved: object) -> str:
    return _canonical_sha256(
        [
            [[str(value) for value in helicity] for helicity in point]
            for point in resolved.values
        ]
    )


def _points_identity(
    points: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object]:
    hexadecimal = [
        [[float(component).hex() for component in momentum] for momentum in point]
        for point in points
    ]
    maximum_conservation_residual = 0.0
    maximum_mass_squared = 0.0
    for point in points:
        incoming = point[:2]
        outgoing = point[2:]
        for component in range(4):
            residual = sum(momentum[component] for momentum in incoming) - sum(
                momentum[component] for momentum in outgoing
            )
            maximum_conservation_residual = max(
                maximum_conservation_residual,
                float(abs(residual)),
            )
        for momentum in point:
            mass_squared = momentum[0] ** 2 - sum(value**2 for value in momentum[1:])
            maximum_mass_squared = max(
                maximum_mass_squared,
                float(abs(mass_squared)),
            )
    if maximum_conservation_residual > 1.0e-9 or maximum_mass_squared > 1.0e-8:
        raise GateError("deterministic validation momenta are not physical")
    return {
        "algorithm": "sha256-float-hex-momenta-v1",
        "sha256": _canonical_sha256(hexadecimal),
        "point_count": len(points),
        "external_particle_count": len(points[0]) if points else 0,
        "maximum_momentum_conservation_residual": maximum_conservation_residual,
        "maximum_abs_mass_squared": maximum_mass_squared,
    }


def _comparison(
    left: Sequence[complex],
    right: Sequence[complex],
    *,
    label: str,
) -> dict[str, object]:
    if len(left) != len(right) or not left:
        raise GateError(f"{label} vectors have incompatible or empty shapes")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failing = 0
    for left_value, right_value in zip(left, right, strict=True):
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(left_value), abs(right_value), ABSOLUTE_TOLERANCE)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > ABSOLUTE_TOLERANCE and relative > RELATIVE_TOLERANCE:
            failing += 1
    result = {
        "label": label,
        "value_count": len(left),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "failing_value_count": failing,
        "passes": failing == 0,
    }
    if failing:
        raise GateError(
            f"{label} failed for {failing}/{len(left)} values "
            f"(max relative {maximum_relative:.3e})"
        )
    return result


def _complex_vector_digest(values: Sequence[complex]) -> str:
    return _canonical_sha256([[value.real.hex(), value.imag.hex()] for value in values])


def _physics_axes(physics: object) -> dict[str, object]:
    return {
        "process_id": physics.process_id,
        "process": physics.process,
        "color_accuracy": physics.color_accuracy,
        "external_pdgs": [particle.pdg_id for particle in physics.external_particles],
        "helicities": [
            {
                "id": item.id,
                "values": list(item.values),
                "computed": item.computed,
                "representative_id": item.representative_id,
                "coefficient": item.coefficient,
            }
            for item in physics.helicities
        ],
        "color_flows": [
            {
                "id": item.id,
                "word": list(item.word),
                "computed": item.computed,
                "representative_id": item.representative_id,
                "coefficient": item.coefficient,
            }
            for item in physics.color_flows
        ],
        "contracted_color_ids": [
            item.id for item in physics.contracted_color_components
        ],
        "selector_capabilities": list(physics.selector_capabilities),
        "reduction_kind": physics.reduction.kind,
    }


def _select_helicity_probe(physics: Any, *, lane_name: str) -> tuple[Any, int]:
    eligible = tuple(
        helicity
        for helicity in physics.helicities
        if helicity.computed
        and not helicity.structural_zero
        and helicity.coefficient != 0.0
        and helicity.representative_id == helicity.id
    )
    if not eligible:
        raise GateError(f"{lane_name} has no executable helicity selector probe")
    return eligible[len(eligible) // 2], len(eligible)


def _evaluate_artifact(
    artifact: Path,
    *,
    process_id: str,
    lane: Lane,
    points: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[dict[str, object], EvaluationCapture]:
    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process=process_id)
    if runtime.execution_mode != "compiled":
        raise GateError(f"{lane.name} runtime did not load the compiled lane")
    if runtime.artifact_id != _json_object(
        artifact / "artifact.json", label="artifact manifest"
    ).get("artifact_id"):
        raise GateError(f"{lane.name} native runtime authenticated another artifact")
    physics = runtime.physics
    observed_pdgs = sorted(particle.pdg_id for particle in physics.external_particles)
    if observed_pdgs != list(EXPECTED_EXTERNAL_PDGS):
        raise GateError(f"{lane.name} runtime does not expose four quark lines")
    if physics.color_accuracy != lane.color_accuracy:
        raise GateError(f"{lane.name} runtime color accuracy is wrong")
    if lane.contracted:
        if (
            physics.color_flows
            or len(physics.contracted_color_components) != 1
            or physics.reduction.kind != "contracted-color"
        ):
            raise GateError(f"{lane.name} did not expose contracted color")
    elif (
        not physics.color_flows
        or physics.contracted_color_components
        or physics.reduction.kind != "lc-diagonal"
    ):
        raise GateError(f"{lane.name} did not expose physical LC flows")

    direct_profiles = {
        "complete": _profile_compiled_runtime(runtime, points),
    }
    f64_total = tuple(_as_complex(value) for value in runtime.evaluate(points))
    exact_total_raw = runtime.evaluate(points, precision=32)
    exact_total = tuple(_as_complex(value) for value in exact_total_raw)
    f64_resolved = runtime.evaluate_resolved(points)
    exact_resolved = runtime.evaluate_resolved(points, precision=32)
    if (
        tuple(f64_resolved.helicity_ids) != physics.helicity_ids
        or tuple(exact_resolved.helicity_ids) != physics.helicity_ids
        or tuple(f64_resolved.color_ids) != physics.color_ids
        or tuple(exact_resolved.color_ids) != physics.color_ids
    ):
        raise GateError(f"{lane.name} resolved physical axes drifted")
    f64_components = _flatten_resolved(f64_resolved)
    exact_components = _flatten_resolved(exact_resolved)
    f64_resolved_total = tuple(_as_complex(value) for value in f64_resolved.total())
    exact_resolved_total = tuple(_as_complex(value) for value in exact_resolved.total())

    comparisons = {
        "f64_vs_precision32_total": _comparison(
            f64_total, exact_total, label=f"{lane.name} f64/precision32 totals"
        ),
        "f64_vs_precision32_components": _comparison(
            f64_components,
            exact_components,
            label=f"{lane.name} f64/precision32 resolved components",
        ),
        "f64_evaluate_vs_resolved_total": _comparison(
            f64_total,
            f64_resolved_total,
            label=f"{lane.name} f64 evaluate/resolved total",
        ),
        "precision32_evaluate_vs_resolved_total": _comparison(
            exact_total,
            exact_resolved_total,
            label=f"{lane.name} precision32 evaluate/resolved total",
        ),
    }

    helicity, eligible_helicity_count = _select_helicity_probe(
        physics, lane_name=lane.name
    )
    helicity_id = helicity.id
    helicity_probe = tuple(
        _as_complex(value)
        for value in runtime.evaluate(points, helicities=(helicity_id,))
    )
    helicity_resolved = runtime.evaluate_resolved(points, helicities=(helicity_id,))
    comparisons["helicity_selector_evaluate_vs_resolved_total"] = _comparison(
        helicity_probe,
        tuple(_as_complex(value) for value in helicity_resolved.total()),
        label=f"{lane.name} helicity selector evaluate/resolved total",
    )
    direct_profiles["helicity_selector"] = _profile_compiled_runtime(
        runtime,
        points,
        helicities=(helicity_id,),
    )
    color_id: str | None = None
    color_probe: tuple[complex, ...] | None = None
    if physics.color_flow_ids:
        color_id = physics.color_flow_ids[len(physics.color_flow_ids) // 2]
        color_probe = tuple(
            _as_complex(value)
            for value in runtime.evaluate(points, color_flows=(color_id,))
        )
        color_resolved = runtime.evaluate_resolved(points, color_flows=(color_id,))
        comparisons["color_selector_evaluate_vs_resolved_total"] = _comparison(
            color_probe,
            tuple(_as_complex(value) for value in color_resolved.total()),
            label=f"{lane.name} color selector evaluate/resolved total",
        )
        direct_profiles["color_selector"] = _profile_compiled_runtime(
            runtime,
            points,
            color_flows=(color_id,),
        )
        direct_profiles["combined_selector"] = _profile_compiled_runtime(
            runtime,
            points,
            helicities=(helicity_id,),
            color_flows=(color_id,),
        )

    axes = _physics_axes(physics)
    capture = EvaluationCapture(
        physics_axes=axes,
        f64_total=f64_total,
        exact_total=exact_total,
        f64_components=f64_components,
        exact_components=exact_components,
        color_probe=color_probe,
        helicity_probe=helicity_probe,
    )
    evidence = {
        "point_count": len(points),
        "point_seeds": list(POINT_SEEDS[: len(points)]),
        "sqrt_s": SQRT_S,
        "physics_axes": {
            **axes,
            "canonical_sha256": _canonical_sha256(axes),
        },
        "f64_total": [[value.real, value.imag] for value in f64_total],
        "precision32_total_decimal": [str(value) for value in exact_total_raw],
        "resolved": {
            "shape": list(f64_resolved.shape),
            "component_count": len(f64_components),
            "f64_component_digest": _complex_vector_digest(f64_components),
            "precision32_component_decimal_digest": _resolved_value_digest(
                exact_resolved
            ),
            "precision32_decimal_digits": 32,
        },
        "selector_probes": {
            "helicity_id": helicity_id,
            "helicity": {
                "id": helicity.id,
                "index": helicity.index,
                "values": list(helicity.values),
                "computed": helicity.computed,
                "structural_zero": helicity.structural_zero,
                "representative_id": helicity.representative_id,
                "coefficient": helicity.coefficient,
            },
            "eligible_helicity_count": eligible_helicity_count,
            "color_flow_id": color_id,
        },
        "runtime_execution": {
            "execution_mode": runtime.execution_mode,
            "direct_arena_profiles": direct_profiles,
        },
        "comparisons": comparisons,
        "finite_value_count": (
            len(f64_total)
            + len(exact_total)
            + len(f64_components)
            + len(exact_components)
            + len(helicity_probe)
            + (0 if color_probe is None else len(color_probe))
        ),
        "passes": True,
    }
    return evidence, capture


def _profile_compiled_runtime(
    runtime: object,
    points: object,
    *,
    helicities: tuple[str, ...] | None = None,
    color_flows: tuple[str, ...] | None = None,
) -> dict[str, object]:
    profiler = getattr(runtime, "profile", None)
    surface = "Runtime.profile"
    if not callable(profiler):
        backend = getattr(runtime, "_backend", None)
        profiler = getattr(backend, "profile", None)
        surface = "native runtime adapter profile"
    if not callable(profiler):
        raise GateError(
            "installed public/runtime-adapter API exposes no profiler; "
            "compiled Direct-Arena activity cannot be proven"
        )
    payload = profiler(
        points,
        helicities=helicities,
        color_flows=color_flows,
        precision=16,
        include_values=False,
    )
    return {
        **_audit_compiled_direct_profile(payload),
        "surface": surface,
        "helicity_selector_count": 0 if helicities is None else len(helicities),
        "color_selector_count": 0 if color_flows is None else len(color_flows),
    }


def _audit_compiled_direct_profile(profile: object) -> dict[str, object]:
    """Require observed Direct-Arena work and no legacy evaluator boundaries."""

    payload = _mapping(profile, label="compiled runtime profile")
    if payload.get("execution_mode") != "compiled":
        raise GateError("compiled runtime profile reports another execution mode")
    engines = payload.get("compiled_direct_arena_engine_count")
    if isinstance(engines, bool) or not isinstance(engines, int) or engines <= 0:
        raise GateError("compiled Direct-Arena profile observed no Arena engines")
    calls = payload.get("compiled_direct_arena_call_count")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        raise GateError("compiled Direct-Arena profile observed no backend calls")
    backend_calls = payload.get("evaluator_backend_call_count")
    if (
        isinstance(backend_calls, bool)
        or not isinstance(backend_calls, int)
        or backend_calls <= 0
        or backend_calls != calls
    ):
        raise GateError(
            "compiled Direct-Arena calls do not cover every evaluator backend call"
        )
    boundary_counts: dict[str, int] = {}
    for field in _ZERO_PROFILE_COUNTERS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise GateError(
                "compiled Direct-Arena profile observed legacy boundary traffic: "
                f"{field}={value!r}"
            )
        boundary_counts[field] = value
    boundary_times: dict[str, float] = {}
    for field in _ZERO_PROFILE_TIMES:
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
        ):
            raise GateError(
                "compiled Direct-Arena profile observed legacy boundary timing: "
                f"{field}={value!r}"
            )
        boundary_times[field] = float(value)
    return {
        "compiled_direct_arena_engine_count": engines,
        "compiled_direct_arena_call_count": calls,
        "evaluator_backend_call_count": backend_calls,
        "legacy_boundary_component_counts": boundary_counts,
        "legacy_boundary_component_total": sum(boundary_counts.values()),
        "legacy_boundary_times_seconds": boundary_times,
        "passes": True,
    }


def _cross_lc_parity(
    topology: EvaluationCapture,
    union: EvaluationCapture,
) -> dict[str, object]:
    axes_match = topology.physics_axes == union.physics_axes
    if not axes_match:
        raise GateError("LC topology-replay and union physical axes differ")
    if topology.color_probe is None or union.color_probe is None:
        raise GateError("LC selector comparison has no physical color probe")
    comparisons = {
        "f64_totals": _comparison(
            topology.f64_total,
            union.f64_total,
            label="LC topology/union f64 totals",
        ),
        "precision32_totals": _comparison(
            topology.exact_total,
            union.exact_total,
            label="LC topology/union precision32 totals",
        ),
        "f64_resolved_components": _comparison(
            topology.f64_components,
            union.f64_components,
            label="LC topology/union f64 resolved components",
        ),
        "precision32_resolved_components": _comparison(
            topology.exact_components,
            union.exact_components,
            label="LC topology/union precision32 resolved components",
        ),
        "color_selector": _comparison(
            topology.color_probe,
            union.color_probe,
            label="LC topology/union color selector",
        ),
        "helicity_selector": _comparison(
            topology.helicity_probe,
            union.helicity_probe,
            label="LC topology/union helicity selector",
        ),
    }
    return {
        "physical_axes_match": axes_match,
        "physical_axes_sha256": _canonical_sha256(topology.physics_axes),
        "comparisons": comparisons,
        "passes": True,
    }


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["content_identity"] = {
        "algorithm": "sha256-canonical-json-body-v1",
        "sha256": _canonical_sha256(body),
    }
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise GateError(f"cannot write evidence: {path}") from error


def _prepare_output_root(output_root: Path, *, replace: bool) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    if root == ROOT or root == Path(root.anchor):
        raise GateError("refusing a broad output root")
    if root.exists():
        if not root.is_dir():
            raise GateError(f"output root is not a directory: {root}")
        occupied = [lane for lane in LANES if (root / lane.name).exists()]
        if occupied and not replace:
            names = ", ".join(lane.name for lane in occupied)
            raise GateError(f"artifact lanes already exist ({names}); pass --replace")
        if replace:
            for lane in LANES:
                target = root / lane.name
                if target.exists():
                    shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_gate(arguments: argparse.Namespace) -> dict[str, object]:
    source = _source_identity()
    runtime = _runtime_identity(source)
    script = _file_identity(Path(__file__))
    runtime_sha256 = _runtime_identity_sha256(runtime)
    union_rejections = _assert_union_rejections()
    output_root = _prepare_output_root(
        arguments.output_root,
        replace=arguments.replace,
    )
    points = _deterministic_points(arguments.points)
    points_identity = _points_identity(points)
    lanes: dict[str, dict[str, object]] = {}
    captures: dict[str, EvaluationCapture] = {}
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    for lane in LANES:
        artifact = output_root / lane.name
        try:
            generation = _invoke_generation_worker(
                artifact,
                lane,
                timeout=arguments.generation_timeout,
                expected_revision=str(source["revision"]),
                expected_runtime_sha256=runtime_sha256,
                expected_script_sha256=str(script["sha256"]),
            )
            identity, process_id = _artifact_identity_and_audit(artifact, lane)
            numerical, capture = _evaluate_artifact(
                artifact,
                process_id=process_id,
                lane=lane,
                points=points,
            )
            lanes[lane.name] = {
                "configuration": {
                    "process": PROCESS,
                    "max_quark_lines": MAX_QUARK_LINES,
                    "color_accuracy": lane.color_accuracy,
                    "lc_flow_layout": lane.lc_flow_layout,
                    "contracted_color": lane.contracted,
                    "backend": "jit",
                    "execution_mode": "compiled",
                    "jit_optimization_level": 3,
                },
                "generation": generation,
                "artifact_identity": identity,
                "numerical_validation": numerical,
                "passes": True,
            }
            captures[lane.name] = capture
        except (GateError, OSError, RuntimeError, ValueError) as error:
            failures.append(
                {
                    "lane": lane.name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            lanes[lane.name] = {
                "configuration": {
                    "process": PROCESS,
                    "max_quark_lines": MAX_QUARK_LINES,
                    "color_accuracy": lane.color_accuracy,
                    "lc_flow_layout": lane.lc_flow_layout,
                    "contracted_color": lane.contracted,
                    "backend": "jit",
                    "execution_mode": "compiled",
                    "jit_optimization_level": 3,
                },
                "passes": False,
            }
    cross_lc: dict[str, object] | None = None
    if {"lc-topology-replay", "lc-all-flow-union"} <= captures.keys():
        try:
            cross_lc = _cross_lc_parity(
                captures["lc-topology-replay"],
                captures["lc-all-flow-union"],
            )
        except GateError as error:
            failures.append(
                {
                    "lane": "lc-cross-layout",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            cross_lc = {"passes": False, "message": str(error)}
    else:
        cross_lc = {
            "passes": False,
            "message": "both LC captures are required",
        }
    source_postflight = _source_identity()
    runtime_postflight = _runtime_identity(source_postflight)
    script_postflight = _file_identity(Path(__file__))
    if source_postflight != source:
        raise GateError("source identity changed during the four-quark gate")
    if _stable_runtime_identity(runtime_postflight) != _stable_runtime_identity(
        runtime
    ):
        raise GateError("candidate runtime identity changed during the four-quark gate")
    if script_postflight != script:
        raise GateError("gate script identity changed during the four-quark gate")
    passed = (
        not failures
        and all(lane.get("passes") is True for lane in lanes.values())
        and cross_lc.get("passes") is True
    )
    body = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if passed else "validation_failed",
        "passes": passed,
        "request": {
            "process": PROCESS,
            "independent_quark_line_count": 4,
            "expected_external_pdgs": list(EXPECTED_EXTERNAL_PDGS),
            "max_quark_lines": MAX_QUARK_LINES,
            "point_count": arguments.points,
            "point_seeds": list(POINT_SEEDS[: arguments.points]),
            "momenta_identity": points_identity,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "mandatory_watchdog_command": WATCHDOG_INVOCATION,
            "memory_limit_gib": 30,
        },
        "source_identity": source,
        "source_identity_postflight": source_postflight,
        "source_identity_match": True,
        "runtime_identity": runtime,
        "runtime_identity_postflight": runtime_postflight,
        "runtime_identity_match": True,
        "gate_script_identity": script,
        "gate_script_identity_postflight": script_postflight,
        "gate_script_identity_match": True,
        "invalid_union_configurations": union_rejections,
        "lanes": lanes,
        "lc_cross_layout_parity": cross_lc,
        "resources": {
            "orchestration_wall_seconds": time.monotonic() - started,
            "orchestrator_peak_rss_gib": _peak_rss_gib(),
            "mandatory_process_tree_limit_gib": 30,
        },
        "failures": failures,
    }
    return _attach_content_identity(body)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Mandatory invocation: {WATCHDOG_INVOCATION}",
    )
    result.add_argument("--output-root", type=Path)
    result.add_argument(
        "--evidence",
        type=Path,
        help="default: OUTPUT_ROOT/four-quark-compiled-gate.json",
    )
    result.add_argument(
        "--points",
        type=int,
        choices=(2, 3),
        default=2,
        help="number of deterministic physical points (default: 2)",
    )
    result.add_argument(
        "--generation-timeout",
        type=float,
        default=10_800.0,
        help="per-lane generation timeout in seconds",
    )
    result.add_argument(
        "--replace",
        action="store_true",
        help="remove only the four named lane directories before generation",
    )
    result.add_argument("--_generate-one", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    result.add_argument(
        "--lane",
        choices=tuple(lane.name for lane in LANES),
        help=argparse.SUPPRESS,
    )
    result.add_argument(
        "--expected-script-sha256",
        help=argparse.SUPPRESS,
    )
    result.add_argument("--expected-revision", help=argparse.SUPPRESS)
    result.add_argument("--expected-runtime-sha256", help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    _establish_preimport_runtime_identity()
    if arguments._generate_one:
        if arguments.artifact is None or arguments.lane is None:
            parser().error("--_generate-one requires --artifact and --lane")
        if (
            not isinstance(arguments.expected_script_sha256, str)
            or SHA256_PATTERN.fullmatch(arguments.expected_script_sha256) is None
            or not isinstance(arguments.expected_revision, str)
            or REVISION_PATTERN.fullmatch(arguments.expected_revision) is None
            or not isinstance(arguments.expected_runtime_sha256, str)
            or SHA256_PATTERN.fullmatch(arguments.expected_runtime_sha256) is None
        ):
            parser().error(
                "--_generate-one requires full source/runtime/script identities"
            )
        if _sha256_file(Path(__file__)) != arguments.expected_script_sha256:
            print(
                "four-quark-compiled-gate worker: gate script identity mismatch",
                file=sys.stderr,
            )
            return 2
        lane = next(item for item in LANES if item.name == arguments.lane)
        try:
            payload = _generation_worker(
                arguments.artifact,
                lane,
                expected_revision=arguments.expected_revision,
                expected_runtime_sha256=arguments.expected_runtime_sha256,
                expected_script_sha256=arguments.expected_script_sha256,
            )
        except (GateError, OSError, RuntimeError, ValueError) as error:
            print(f"four-quark-compiled-gate worker: {error}", file=sys.stderr)
            return 2
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
        return 0
    if arguments.output_root is None:
        parser().error("--output-root is required")
    if arguments.generation_timeout <= 0 or not math.isfinite(
        arguments.generation_timeout
    ):
        parser().error("--generation-timeout must be positive and finite")
    evidence_path = (
        arguments.evidence
        if arguments.evidence is not None
        else arguments.output_root / "four-quark-compiled-gate.json"
    )
    try:
        result = run_gate(arguments)
        _write_json_atomic(evidence_path.expanduser().resolve(strict=False), result)
    except (GateError, OSError, RuntimeError, ValueError) as error:
        print(f"four-quark-compiled-gate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    _ensure_source_only_python()
    raise SystemExit(main())
