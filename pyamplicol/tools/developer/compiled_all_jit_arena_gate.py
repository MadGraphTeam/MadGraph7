#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""End-to-end compiled Direct-Arena gate for every JIT optimization level.

The gate generates four independent ``g g > g g`` artifacts, one for each
supported JIT source optimization level (O0, O1, O2, and O3).  Every artifact
must be built and evaluated by a candidate installed from the exact clean
source revision.  The gate then:

* authenticates every artifact payload and the native loader's artifact ID;
* recursively requires ``compiled-plane-arena-v1`` for every fused stage;
* requires every source and Direct-Arena leaf to retain the requested JIT
  optimization level;
* rejects DirectApplication metadata below every model-parameter evaluator;
* evaluates deterministic physical points through native f64 and retained
  precision-32 execution;
* compares ``evaluate()`` with ``evaluate_resolved().total()``;
* compares totals and resolved components pointwise across O0/O1/O2/O3; and
* when profiling is exposed, requires nonzero Direct-Arena backend calls and
  zero legacy pack/gather/scatter/remap traffic.

Run this only from a clean, exact-source candidate.  The outer 30 GiB watchdog
is mandatory and covers this process plus every generation worker:

```
.venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  .venv/bin/python tools/developer/compiled_all_jit_arena_gate.py \
  --output-root /private/tmp/pyamplicol-compiled-all-jit-arena-gate
```
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

PROCESS = "g g > g g"
EXPECTED_EXTERNAL_PDGS = (21, 21, 21, 21)
JIT_LEVELS = (0, 1, 2, 3)
POINT_SEEDS = (846_731, 846_739, 846_763)
VALIDATION_SEED = 20260725
SQRT_S = 1_000.0
RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-15
COMPILED_PLANE_ARENA_CAPABILITY = "compiled-plane-arena-v1"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "pyamplicol-compiled-plane-kernel-v2"
NATIVE_COMPILED_DIRECT_APPLICATION_ABI = (
    "pyamplicol-native-compiled-direct-application-v1"
)
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
SYMJIT_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
RESULT_KIND = "pyamplicol-compiled-all-jit-direct-arena-gate"
SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WATCHDOG_INVOCATION = (
    ".venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- "
    ".venv/bin/python tools/developer/compiled_all_jit_arena_gate.py "
    "--output-root /private/tmp/pyamplicol-compiled-all-jit-arena-gate"
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
_DIRECT_ENGINE_COUNTER = "compiled_direct_arena_engine_count"
_DIRECT_CALL_COUNTER = "compiled_direct_arena_call_count"


class GateError(RuntimeError):
    """Raised when all-JIT Direct-Arena evidence cannot be trusted."""


@dataclass(slots=True)
class EvaluationCapture:
    """Non-serializable values retained for cross-level comparisons."""

    physics_axes: dict[str, object]
    f64_total: tuple[complex, ...]
    precision32_total: tuple[complex, ...]
    f64_components: tuple[complex, ...]
    precision32_components: tuple[complex, ...]


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


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise GateError(f"{label} must be artifact-relative")
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
    divisor = float(1024**3 if platform.system() == "Darwin" else 1024**2)
    return raw / divisor


def _generation_worker(
    artifact: Path,
    optimization_level: int,
    expected_revision: str,
    expected_script_sha256: str,
) -> dict[str, object]:
    source = _source_identity()
    if source["revision"] != expected_revision:
        raise GateError("generation worker source revision changed")
    script = _file_identity(Path(__file__))
    if script["sha256"] != expected_script_sha256:
        raise GateError("generation worker gate script identity changed")
    runtime = _runtime_identity(source)

    from pyamplicol import Generator
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        OutputConfig,
        RunConfig,
    )

    run = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
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
            jit=JITConfig(optimization_level=optimization_level),
        ),
        output=OutputConfig(format="json", color="never", progress="off"),
    )
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    result = Generator(run).generate(PROCESS, artifact, mode="error")
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
        "optimization_level": optimization_level,
        "artifact": str(result.output),
        "process_ids": [process.name for process in result.processes.requests],
        "worker_provenance": {
            "source_revision": source["revision"],
            "runtime_identity_sha256": _runtime_identity_sha256(runtime),
            "native_extension_sha256": runtime["native_extension"]["sha256"],
            "native_build_inputs_sha256": runtime["native_extension"][
                "build_inputs_sha256"
            ],
            "gate_script_sha256": script["sha256"],
            "postflight_identity_match": True,
        },
        "generation": {
            "wall_seconds": time.monotonic() - started_wall,
            "process_cpu_seconds": time.process_time() - started_cpu,
            "worker_peak_rss_gib": _peak_rss_gib(),
            "workers": 1,
            "validation_samples": 2,
            "validation_seed": VALIDATION_SEED,
            "post_build_validation": True,
        },
    }


def _worker_command(
    artifact: Path,
    optimization_level: int,
    expected_revision: str,
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
        "--jit-level",
        str(optimization_level),
        "--expected-revision",
        expected_revision,
        "--expected-script-sha256",
        expected_script_sha256,
    )


def _invoke_generation_worker(
    artifact: Path,
    optimization_level: int,
    expected_revision: str,
    expected_script_sha256: str,
    *,
    timeout: float,
) -> dict[str, object]:
    command = _worker_command(
        artifact,
        optimization_level,
        expected_revision,
        expected_script_sha256,
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
            f"JIT O{optimization_level} generation exceeded {timeout:.1f} seconds"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(
            f"JIT O{optimization_level} generation failed:\n{detail[-6000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            f"JIT O{optimization_level} worker emitted invalid JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("optimization_level") != optimization_level
    ):
        raise GateError(f"JIT O{optimization_level} worker returned the wrong level")
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
    leaves: list[Mapping[str, Any]] = []
    for index, raw_chunk in enumerate(chunks):
        chunk = _mapping(raw_chunk, label=f"{label}.chunks[{index}]")
        leaves.extend(
            _source_evaluator_leaves(
                chunk,
                label=f"{label}.chunks[{index}]",
            )
        )
    if not leaves:
        raise GateError(f"{label} has no evaluator leaves")
    return leaves


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


def _audit_direct_descriptor(
    descriptor: Mapping[str, Any],
    *,
    stage: Mapping[str, Any],
    optimization_level: int,
    label: str,
) -> dict[str, int]:
    if descriptor.get("kind") != "compiled-plane-arena-stage":
        raise GateError(f"{label}.compiled_plane_arena has the wrong kind")
    if descriptor.get("schema_version") != 1:
        raise GateError(f"{label}.compiled_plane_arena has the wrong schema")
    expected_scalars = {
        "application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "source_application_abi": SYMJIT_PLANE_APPLICATION_ABI,
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
    if [
        _mapping(item, label=f"{label}.input_bindings[{index}]").get("parameter_index")
        for index, item in enumerate(inputs)
    ] != list(range(len(inputs))):
        raise GateError(f"{label} Direct-Arena input bindings are not contiguous")

    outputs = _sequence(
        descriptor.get("output_bindings"),
        label=f"{label}.compiled_plane_arena.output_bindings",
    )
    if [
        _mapping(item, label=f"{label}.output_bindings[{index}]").get("output_index")
        for index, item in enumerate(outputs)
    ] != list(range(len(outputs))):
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
        if (
            source.get("application_abi") != SYMJIT_APPLICATION_ABI
            or source.get("runtime_capability") != SYMJIT_RUNTIME_CAPABILITY
        ):
            raise GateError(
                f"{label} ordinary SymJIT fallback leaf identity is inconsistent"
            )
        _safe_relative_path(
            source.get("application_path"),
            label=f"{label}.source_leaves[{index}].application_path",
        )
        plane = _mapping(
            source.get("plane_application"),
            label=f"{label}.source_leaves[{index}].plane_application",
        )
        source_path = _safe_relative_path(
            plane.get("application_path"),
            label=(
                f"{label}.source_leaves[{index}].plane_application.application_path"
            ),
        )
        if (
            direct.get("application_path") != source_path
            or direct.get("source_application_abi")
            != SYMJIT_PLANE_APPLICATION_ABI
            or plane.get("application_abi") != SYMJIT_PLANE_APPLICATION_ABI
            or plane.get("storage_abi") != SYMJIT_APPLICATION_ABI
            or plane.get("translation_mode")
            != "symbolica-structured-instructions"
            or plane.get("direct_arena") is not True
            or not isinstance(plane.get("source_digest"), str)
            or SHA256_PATTERN.fullmatch(str(plane.get("source_digest"))) is None
        ):
            raise GateError(f"{label} Direct-Arena leaf identity is inconsistent")
        if (
            direct.get("optimization_level") != optimization_level
            or source.get("optimization_level") != optimization_level
            or plane.get("optimization_level") != optimization_level
        ):
            raise GateError(
                f"{label} Direct-Arena/source leaf does not retain JIT "
                f"O{optimization_level}"
            )
        if direct.get("direct_codegen_optimization_level") != optimization_level:
            raise GateError(
                f"{label} Direct-Arena leaf direct codegen does not match its "
                "plane application"
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


def _audit_execution_payload(
    payload: Mapping[str, Any],
    optimization_level: int,
) -> dict[str, object]:
    capability_declarations = 0
    stage_set_count = 0
    stage_count = 0
    descriptor_count = 0
    leaf_count = 0
    model_parameter_slot_count = 0
    model_parameter_evaluator_count = 0
    audited_stage_sets: list[dict[str, object]] = []
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
        required = _sequence(
            record.get("required_runtime_capabilities"),
            label=f"{label}.required_runtime_capabilities",
        )
        if COMPILED_PLANE_ARENA_CAPABILITY not in required:
            raise GateError(f"{label} does not require compiled-plane-arena-v1")
        stages = list(_sequence(record.get("stages"), label=f"{label}.stages"))
        amplitude = _mapping(
            record.get("amplitude_stage"),
            label=f"{label}.amplitude_stage",
        )
        all_stages = (*stages, amplitude)
        if record.get("stage_count") != len(all_stages):
            raise GateError(f"{label}.stage_count does not cover all stages")
        set_leaf_count = 0
        for index, raw_stage in enumerate(all_stages):
            stage = _mapping(raw_stage, label=f"{label}.stages[{index}]")
            descriptor = _mapping(
                stage.get("compiled_plane_arena"),
                label=f"{label}.stages[{index}].compiled_plane_arena",
            )
            counts = _audit_direct_descriptor(
                descriptor,
                stage=stage,
                optimization_level=optimization_level,
                label=f"{label}.stages[{index}]",
            )
            stage_count += 1
            descriptor_count += 1
            leaf_count += counts["leaf_count"]
            set_leaf_count += counts["leaf_count"]
        audited_stage_sets.append(
            {
                "path": label,
                "stage_count": len(all_stages),
                "leaf_count": set_leaf_count,
                "source_optimization_level": optimization_level,
            }
        )
    if capability_declarations < 1:
        raise GateError("execution payload does not declare compiled-plane-arena-v1")
    if stage_set_count < 1 or descriptor_count < 1 or leaf_count < 1:
        raise GateError("execution payload has no complete Direct-Arena stage set")
    if model_parameter_slot_count < 1:
        raise GateError("execution payload has no model-parameter evaluator slot")
    return {
        "capability": COMPILED_PLANE_ARENA_CAPABILITY,
        "source_optimization_level": optimization_level,
        "capability_declaration_count": capability_declarations,
        "stage_set_count": stage_set_count,
        "stage_count": stage_count,
        "descriptor_count": descriptor_count,
        "leaf_count": leaf_count,
        "model_parameter_slot_count": model_parameter_slot_count,
        "model_parameter_evaluator_count": model_parameter_evaluator_count,
        "model_parameter_direct_application_count": 0,
        "stage_sets": audited_stage_sets,
        "passes": True,
    }


def _artifact_identity_and_audit(
    artifact: Path,
    optimization_level: int,
) -> tuple[dict[str, object], str]:
    manifest_path = artifact / "artifact.json"
    manifest = _json_object(manifest_path, label="artifact manifest")
    artifact_id = manifest.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or SHA256_PATTERN.fullmatch(artifact_id) is None
    ):
        raise GateError(f"JIT O{optimization_level} artifact ID is invalid")
    processes = _sequence(manifest.get("processes"), label="artifact.processes")
    if len(processes) != 1:
        raise GateError(
            f"JIT O{optimization_level} artifact must contain exactly one process"
        )
    process = _mapping(processes[0], label="artifact.processes[0]")
    if process.get("expression") != PROCESS:
        raise GateError(f"JIT O{optimization_level} artifact has the wrong process")
    if tuple(process.get("external_pdgs", ())) != EXPECTED_EXTERNAL_PDGS:
        raise GateError(f"JIT O{optimization_level} external PDGs drifted")
    if process.get("color_accuracy") != "lc":
        raise GateError(f"JIT O{optimization_level} color accuracy drifted")
    capabilities = _sequence(
        process.get("required_runtime_capabilities"),
        label="artifact.processes[0].required_runtime_capabilities",
    )
    if COMPILED_PLANE_ARENA_CAPABILITY not in capabilities:
        raise GateError(
            f"JIT O{optimization_level} artifact lacks compiled-plane-arena-v1"
        )
    process_id = process.get("id")
    if not isinstance(process_id, str) or not process_id:
        raise GateError(f"JIT O{optimization_level} process ID is invalid")

    try:
        effective = tomllib.loads(
            (artifact / "config" / "effective.toml").read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateError(
            f"JIT O{optimization_level} effective configuration is invalid"
        ) from error
    expected_config = (
        (effective.get("color", {}).get("accuracy"), "lc"),
        (
            effective.get("color", {}).get("lc_flow_layout"),
            "topology-replay",
        ),
        (effective.get("evaluator", {}).get("backend"), "jit"),
        (effective.get("evaluator", {}).get("execution_mode"), "compiled"),
        (
            effective.get("evaluator", {}).get("jit", {}).get("optimization_level"),
            optimization_level,
        ),
    )
    if any(observed != expected for observed, expected in expected_config):
        raise GateError(
            f"JIT O{optimization_level} effective compiled configuration drifted"
        )

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
            raise GateError(
                f"JIT O{optimization_level} payload is missing: {relative}"
            ) from error
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or size != expected_size
            or not isinstance(expected_digest, str)
            or _sha256_file(path) != expected_digest
        ):
            raise GateError(
                f"JIT O{optimization_level} payload identity failed: {relative}"
            )
        payload_bytes += size

    execution_path = artifact / "processes" / process_id / "execution.json"
    execution = _json_object(execution_path, label="process execution")
    if execution.get("color_accuracy") != "lc":
        raise GateError(f"JIT O{optimization_level} execution color drifted")
    arena_audit = _audit_execution_payload(execution, optimization_level)
    return (
        {
            "path": str(artifact.resolve(strict=True)),
            "artifact_id": artifact_id,
            "manifest": _file_identity(manifest_path),
            "tree": _tree_identity(artifact),
            "payload_count": len(payloads),
            "payload_bytes": payload_bytes,
            "effective_configuration": {
                "process": PROCESS,
                "color_accuracy": "lc",
                "lc_flow_layout": "topology-replay",
                "backend": "jit",
                "execution_mode": "compiled",
                "jit_optimization_level": optimization_level,
            },
            "direct_arena_audit": arena_audit,
        },
        process_id,
    )


def _deterministic_points(
    count: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
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
                2,
                sqrt_s=SQRT_S,
                masses=(0.0, 0.0),
                seed=seed,
            ),
        )
        for seed in POINT_SEEDS[:count]
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
        "maximum_momentum_conservation_residual": (maximum_conservation_residual),
        "maximum_abs_mass_squared": maximum_mass_squared,
    }


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
        relative = absolute / max(
            abs(left_value),
            abs(right_value),
            ABSOLUTE_TOLERANCE,
        )
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
        "selector_capabilities": list(physics.selector_capabilities),
        "reduction_kind": physics.reduction.kind,
    }


def _required_nonnegative_profile_integer(
    profile: Mapping[str, object],
    name: str,
) -> int:
    value = profile.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"native profile field {name!r} is not a non-negative integer")
    return value


def _audit_profile(profile: Mapping[str, object]) -> dict[str, object]:
    if profile.get("execution_mode") != "compiled":
        raise GateError("native profile did not execute the compiled lane")
    for name in _ZERO_PROFILE_COUNTERS:
        value = _required_nonnegative_profile_integer(profile, name)
        if value != 0:
            raise GateError(f"compiled Direct-Arena profile observed {name}={value}")
    for name in _ZERO_PROFILE_TIMES:
        value = profile.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
        ):
            raise GateError(
                f"compiled Direct-Arena profile observed nonzero/invalid {name}"
            )

    engine_count = _required_nonnegative_profile_integer(
        profile,
        _DIRECT_ENGINE_COUNTER,
    )
    if engine_count < 1:
        raise GateError("native profile reports no compiled Direct-Arena engines")
    call_count = _required_nonnegative_profile_integer(
        profile,
        _DIRECT_CALL_COUNTER,
    )
    if call_count < 1:
        raise GateError("native profile reports no compiled Direct-Arena calls")
    backend_call_count = _required_nonnegative_profile_integer(
        profile,
        "evaluator_backend_call_count",
    )
    if backend_call_count != call_count:
        raise GateError(
            "compiled Direct-Arena call count does not cover every evaluator "
            "backend call"
        )

    counters = {
        name: _required_nonnegative_profile_integer(profile, name)
        for name in (
            *_ZERO_PROFILE_COUNTERS,
            _DIRECT_ENGINE_COUNTER,
            _DIRECT_CALL_COUNTER,
            "evaluator_backend_call_count",
        )
    }
    return {
        "available": True,
        "execution_mode": "compiled",
        "direct_arena_engine_counter": {
            "name": _DIRECT_ENGINE_COUNTER,
            "value": engine_count,
        },
        "direct_arena_call_counter": {
            "name": _DIRECT_CALL_COUNTER,
            "value": call_count,
        },
        "zero_legacy_traffic_counters": counters,
        "zero_legacy_traffic_times": {
            name: float(profile[name]) for name in _ZERO_PROFILE_TIMES
        },
        "passes": True,
    }


def _runtime_execution_mode(runtime: object) -> tuple[str, str]:
    value = getattr(runtime, "execution_mode", None)
    if isinstance(value, str):
        return value, "Runtime.execution_mode"
    backend = getattr(runtime, "_backend", None)
    value = getattr(backend, "execution_mode", None)
    if isinstance(value, str):
        return value, "native runtime adapter execution_mode"
    raise GateError("loaded runtime exposes no execution-mode identity")


def _profile_runtime(
    runtime: object,
    points: Sequence[Sequence[Sequence[float]]],
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
            "Direct-Arena activity cannot be proven"
        )
    payload = profiler(points, precision=16, include_values=False)
    if not isinstance(payload, Mapping):
        raise GateError("native runtime profile is not an object")
    return {**_audit_profile(payload), "surface": surface}


def _evaluate_artifact(
    artifact: Path,
    *,
    process_id: str,
    optimization_level: int,
    points: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[dict[str, object], EvaluationCapture]:
    from pyamplicol import Runtime

    manifest_id = _json_object(
        artifact / "artifact.json",
        label="artifact manifest",
    ).get("artifact_id")
    runtime = Runtime.load(artifact, process=process_id)
    if runtime.artifact_id != manifest_id:
        raise GateError(
            f"JIT O{optimization_level} native runtime authenticated another artifact"
        )
    execution_mode, execution_mode_surface = _runtime_execution_mode(runtime)
    if execution_mode != "compiled":
        raise GateError(f"JIT O{optimization_level} loaded a non-compiled runtime")
    physics = runtime.physics
    if physics.process != PROCESS:
        raise GateError(f"JIT O{optimization_level} runtime process drifted")
    if tuple(particle.pdg_id for particle in physics.external_particles) != (
        EXPECTED_EXTERNAL_PDGS
    ):
        raise GateError(f"JIT O{optimization_level} runtime external PDGs drifted")
    if physics.color_accuracy != "lc":
        raise GateError(f"JIT O{optimization_level} runtime color accuracy drifted")

    f64_total = tuple(_as_complex(value) for value in runtime.evaluate(points))
    precision32_raw = runtime.evaluate(points, precision=32)
    precision32_total = tuple(_as_complex(value) for value in precision32_raw)
    f64_resolved = runtime.evaluate_resolved(points)
    precision32_resolved = runtime.evaluate_resolved(points, precision=32)
    if (
        tuple(f64_resolved.helicity_ids) != physics.helicity_ids
        or tuple(precision32_resolved.helicity_ids) != physics.helicity_ids
        or tuple(f64_resolved.color_ids) != physics.color_ids
        or tuple(precision32_resolved.color_ids) != physics.color_ids
    ):
        raise GateError(f"JIT O{optimization_level} resolved physical axes drifted")
    f64_components = _flatten_resolved(f64_resolved)
    precision32_components = _flatten_resolved(precision32_resolved)
    comparisons = {
        "f64_vs_precision32_total": _comparison(
            f64_total,
            precision32_total,
            label=f"JIT O{optimization_level} f64/precision32 totals",
        ),
        "f64_evaluate_vs_resolved_total": _comparison(
            f64_total,
            tuple(_as_complex(value) for value in f64_resolved.total()),
            label=f"JIT O{optimization_level} f64 evaluate/resolved total",
        ),
        "precision32_evaluate_vs_resolved_total": _comparison(
            precision32_total,
            tuple(_as_complex(value) for value in precision32_resolved.total()),
            label=(f"JIT O{optimization_level} precision32 evaluate/resolved total"),
        ),
        "f64_vs_precision32_resolved_components": _comparison(
            f64_components,
            precision32_components,
            label=f"JIT O{optimization_level} f64/precision32 components",
        ),
    }
    axes = _physics_axes(physics)
    capture = EvaluationCapture(
        physics_axes=axes,
        f64_total=f64_total,
        precision32_total=precision32_total,
        f64_components=f64_components,
        precision32_components=precision32_components,
    )
    return (
        {
            "artifact_id": runtime.artifact_id,
            "execution_mode": execution_mode,
            "execution_mode_surface": execution_mode_surface,
            "point_count": len(points),
            "point_seeds": list(POINT_SEEDS[: len(points)]),
            "physics_axes": {
                **axes,
                "canonical_sha256": _canonical_sha256(axes),
            },
            "f64_total": [[value.real, value.imag] for value in f64_total],
            "precision32_total_decimal": [str(value) for value in precision32_raw],
            "resolved": {
                "shape": list(f64_resolved.shape),
                "component_count": len(f64_components),
                "f64_component_sha256": _complex_vector_digest(f64_components),
                "precision32_component_sha256": _complex_vector_digest(
                    precision32_components
                ),
            },
            "comparisons": comparisons,
            "native_profile": _profile_runtime(runtime, points),
            "passes": True,
        },
        capture,
    )


def _cross_level_parity(
    captures: Mapping[int, EvaluationCapture],
) -> dict[str, object]:
    if set(captures) != set(JIT_LEVELS):
        raise GateError("cross-level parity requires O0, O1, O2, and O3")
    baseline_level = 3
    baseline = captures[baseline_level]
    comparisons: dict[str, dict[str, object]] = {}
    for optimization_level in JIT_LEVELS:
        if optimization_level == baseline_level:
            continue
        candidate = captures[optimization_level]
        if candidate.physics_axes != baseline.physics_axes:
            raise GateError(
                f"JIT O{optimization_level}/O{baseline_level} physical axes differ"
            )
        comparisons[f"o{optimization_level}_vs_o{baseline_level}"] = {
            "f64_totals": _comparison(
                candidate.f64_total,
                baseline.f64_total,
                label=f"JIT O{optimization_level}/O{baseline_level} f64 totals",
            ),
            "precision32_totals": _comparison(
                candidate.precision32_total,
                baseline.precision32_total,
                label=(
                    f"JIT O{optimization_level}/O{baseline_level} precision32 totals"
                ),
            ),
            "f64_resolved_components": _comparison(
                candidate.f64_components,
                baseline.f64_components,
                label=(
                    f"JIT O{optimization_level}/O{baseline_level} "
                    "f64 resolved components"
                ),
            ),
            "precision32_resolved_components": _comparison(
                candidate.precision32_components,
                baseline.precision32_components,
                label=(
                    f"JIT O{optimization_level}/O{baseline_level} "
                    "precision32 resolved components"
                ),
            ),
        }
    return {
        "baseline_optimization_level": baseline_level,
        "physical_axes_sha256": _canonical_sha256(baseline.physics_axes),
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


def _lane_name(optimization_level: int) -> str:
    return f"jit-o{optimization_level}"


def _prepare_output_root(output_root: Path, *, replace: bool) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    if root == ROOT or root == Path(root.anchor):
        raise GateError("refusing a broad output root")
    try:
        root.relative_to(ROOT.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise GateError(
            "output root must be outside the source checkout so every isolated "
            "generation worker can reauthenticate a clean revision"
        )
    if root.exists():
        if not root.is_dir():
            raise GateError(f"output root is not a directory: {root}")
        occupied = [
            _lane_name(level)
            for level in JIT_LEVELS
            if (root / _lane_name(level)).exists()
        ]
        if occupied and not replace:
            raise GateError(
                f"artifact lanes already exist ({', '.join(occupied)}); pass --replace"
            )
        if replace:
            for level in JIT_LEVELS:
                target = root / _lane_name(level)
                if target.exists():
                    shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_gate(arguments: argparse.Namespace) -> dict[str, object]:
    source = _source_identity()
    runtime = _runtime_identity(source)
    script = _file_identity(Path(__file__))
    output_root = _prepare_output_root(
        arguments.output_root,
        replace=arguments.replace,
    )
    points = _deterministic_points(arguments.points)
    points_identity = _points_identity(points)
    levels: dict[str, dict[str, object]] = {}
    captures: dict[int, EvaluationCapture] = {}
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    expected_revision = str(source["revision"])
    expected_runtime_sha = _runtime_identity_sha256(runtime)

    for optimization_level in JIT_LEVELS:
        lane_name = _lane_name(optimization_level)
        artifact = output_root / lane_name
        try:
            generation = _invoke_generation_worker(
                artifact,
                optimization_level,
                expected_revision,
                str(script["sha256"]),
                timeout=arguments.generation_timeout,
            )
            worker = _mapping(
                generation.get("worker_provenance"),
                label=f"{lane_name}.worker_provenance",
            )
            if (
                worker.get("source_revision") != expected_revision
                or worker.get("runtime_identity_sha256") != expected_runtime_sha
                or worker.get("gate_script_sha256") != script["sha256"]
                or worker.get("postflight_identity_match") is not True
            ):
                raise GateError(f"{lane_name} worker provenance drifted")
            identity, process_id = _artifact_identity_and_audit(
                artifact,
                optimization_level,
            )
            numerical, capture = _evaluate_artifact(
                artifact,
                process_id=process_id,
                optimization_level=optimization_level,
                points=points,
            )
            levels[lane_name] = {
                "configuration": {
                    "process": PROCESS,
                    "color_accuracy": "lc",
                    "lc_flow_layout": "topology-replay",
                    "backend": "jit",
                    "execution_mode": "compiled",
                    "jit_optimization_level": optimization_level,
                },
                "generation": generation,
                "artifact_identity": identity,
                "numerical_validation": numerical,
                "passes": True,
            }
            captures[optimization_level] = capture
        except (GateError, OSError, RuntimeError, ValueError) as error:
            failures.append(
                {
                    "lane": lane_name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            levels[lane_name] = {
                "configuration": {
                    "process": PROCESS,
                    "backend": "jit",
                    "execution_mode": "compiled",
                    "jit_optimization_level": optimization_level,
                },
                "passes": False,
            }

    try:
        cross_level = _cross_level_parity(captures)
    except GateError as error:
        failures.append(
            {
                "lane": "cross-level",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        cross_level = {"passes": False, "message": str(error)}
    source_postflight = _source_identity()
    runtime_postflight = _runtime_identity(source_postflight)
    script_postflight = _file_identity(Path(__file__))
    if source_postflight != source:
        raise GateError("source identity changed during the all-JIT gate")
    if _stable_runtime_identity(runtime_postflight) != _stable_runtime_identity(
        runtime
    ):
        raise GateError("candidate runtime identity changed during the all-JIT gate")
    if script_postflight != script:
        raise GateError("gate script identity changed during the all-JIT gate")
    passed = (
        not failures
        and all(level.get("passes") is True for level in levels.values())
        and cross_level.get("passes") is True
    )
    return _attach_content_identity(
        {
            "kind": RESULT_KIND,
            "schema_version": SCHEMA_VERSION,
            "status": "ok" if passed else "validation_failed",
            "passes": passed,
            "request": {
                "process": PROCESS,
                "jit_optimization_levels": list(JIT_LEVELS),
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
            "levels": levels,
            "cross_level_numerical_parity": cross_level,
            "resources": {
                "orchestration_wall_seconds": time.monotonic() - started,
                "orchestrator_peak_rss_gib": _peak_rss_gib(),
                "mandatory_process_tree_limit_gib": 30,
            },
            "failures": failures,
        }
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Mandatory invocation: {WATCHDOG_INVOCATION}",
    )
    result.add_argument("--output-root", type=Path)
    result.add_argument(
        "--evidence",
        type=Path,
        help="default: OUTPUT_ROOT/compiled-all-jit-arena-gate.json",
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
        default=3_600.0,
        help="per-level generation timeout in seconds",
    )
    result.add_argument(
        "--replace",
        action="store_true",
        help="remove only the four named JIT-level directories before generation",
    )
    result.add_argument("--_generate-one", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    result.add_argument(
        "--jit-level",
        type=int,
        choices=JIT_LEVELS,
        help=argparse.SUPPRESS,
    )
    result.add_argument("--expected-revision", help=argparse.SUPPRESS)
    result.add_argument("--expected-script-sha256", help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    _establish_preimport_runtime_identity()
    if arguments._generate_one:
        if (
            arguments.artifact is None
            or arguments.jit_level is None
            or arguments.expected_revision is None
            or arguments.expected_script_sha256 is None
        ):
            parser().error(
                "--_generate-one requires --artifact, --jit-level, and "
                "--expected-revision and --expected-script-sha256"
            )
        if SHA256_PATTERN.fullmatch(arguments.expected_script_sha256) is None:
            parser().error("--expected-script-sha256 must be a SHA-256 digest")
        try:
            payload = _generation_worker(
                arguments.artifact,
                arguments.jit_level,
                arguments.expected_revision,
                arguments.expected_script_sha256,
            )
        except (GateError, OSError, RuntimeError, ValueError) as error:
            print(f"compiled-all-jit-arena worker: {error}", file=sys.stderr)
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
        else arguments.output_root / "compiled-all-jit-arena-gate.json"
    )
    try:
        result = run_gate(arguments)
        _write_json_atomic(evidence_path.expanduser().resolve(strict=False), result)
    except (GateError, OSError, RuntimeError, ValueError) as error:
        print(f"compiled-all-jit-arena: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    _ensure_source_only_python()
    raise SystemExit(main())
