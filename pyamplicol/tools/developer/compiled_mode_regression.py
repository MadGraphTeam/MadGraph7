#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare compiled-mode runtime against a frozen baseline interpreter.

The driver generates one compiled JIT O3 artifact per interpreter and reuses
it while its cache signature remains valid.  Each reported outer sample comes
from a separate repository-owned sampling helper run by that interpreter. The
accepted headline timing comes directly from the warmed, unprofiled
``_benchmark_f64_wall_time`` pass, independent of the installed benchmark
coordinator version. Native profile attribution is collected in a paired pass
over the byte-identical batch and repetition count, and never substitutes for
the headline. A performance gate is authoritative only for one shared artifact
or byte-identical performance-relevant payloads, and warmed native values are
an independent numerical-correctness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools" / "ci" / "memory_watchdog.py"
NATIVE_SAMPLE_HELPER = ROOT / "tools" / "developer" / "compiled_mode_sample.py"
MEMORY_LIMIT_GIB = 30.0
NATIVE_WALL_TIME_SOURCE = "runtime_core_repeated_wall_time"
NATIVE_WALL_TIME_SAMPLE_PASS = "runtime._benchmark_f64_wall_time"
PAIRED_TIMING_SAMPLE_CONTRACT = "paired_unprofiled_headline_profiled_attribution_v1"
PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime.profile_repeated"
NATIVE_SAMPLE_RESULT_KIND = "pyamplicol-compiled-mode-native-sample"
NATIVE_SAMPLE_SCHEMA_VERSION = 2
INSTALLATION_IDENTITY_KIND = "pyamplicol-installed-distribution-identity"
INSTALLATION_IDENTITY_SCHEMA_VERSION = 1
RESULT_KIND = "pyamplicol-compiled-mode-regression"
CACHE_KIND = "pyamplicol-compiled-mode-regression-artifact-cache"
SCHEMA_VERSION = 3
DEFAULT_GENERATION_TIMEOUT = 300.0
DEFAULT_PROFILE_TIMEOUT = 120.0
DEFAULT_TARGET_RUNTIME = 5.0
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WARMUP_RUNS = 2
RELATIVE_TOLERANCE = 0.02
MAD_MULTIPLIER = 3.0
VALIDATION_SEED = 20260719
DEFAULT_LC_FLOW_LAYOUT = "topology-replay"
CORRECTNESS_RELATIVE_TOLERANCE = 1.0e-12
CORRECTNESS_ABSOLUTE_TOLERANCE = 1.0e-15
PERFORMANCE_RELEVANT_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
REQUIRED_PERFORMANCE_PAYLOAD_ROLES = frozenset({"evaluator-state"})
_HASH_CHUNK_BYTES = 1024 * 1024


class RegressionError(RuntimeError):
    """Raised when the regression measurement cannot be trusted."""


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _at_least_five(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least five")
    return parsed


def _absolute_path(path: Path) -> Path:
    """Return an absolute invocation path without resolving venv symlinks."""

    return Path(os.path.abspath(path.expanduser()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise RegressionError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _tree_identity(path: Path) -> dict[str, object]:
    """Hash relative names and bytes for an exact, location-independent tree ID."""

    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot resolve identity tree: {path}") from error
    if not root.is_dir():
        raise RegressionError(f"identity tree is not a directory: {path}")
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    try:
        members = sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        for member in members:
            relative = member.relative_to(root).as_posix().encode("utf-8")
            size = member.stat().st_size
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            with member.open("rb") as stream:
                while chunk := stream.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
            file_count += 1
            size_bytes += size
    except OSError as error:
        raise RegressionError(f"cannot hash identity tree: {path}") from error
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _command_identity(command: Sequence[str]) -> dict[str, object]:
    argv = [str(value) for value in command]
    canonical = json.dumps(
        argv,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "argv": argv,
        "argv_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_argument(value: str) -> str:
    candidate = Path(value).expanduser()
    return str(_absolute_path(candidate)) if candidate.exists() else value


def _require_interpreter(path: Path, *, lane: str) -> Path:
    path = _absolute_path(path)
    if not path.is_file():
        raise RegressionError(f"{lane} Python does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise RegressionError(f"{lane} Python is not executable: {path}")
    return path


def _guarded_command(command: Sequence[str]) -> tuple[str, ...]:
    if not WATCHDOG.is_file():
        raise RegressionError(f"memory watchdog does not exist: {WATCHDOG}")
    if not NATIVE_SAMPLE_HELPER.is_file():
        raise RegressionError(
            f"native sampling helper does not exist: {NATIVE_SAMPLE_HELPER}"
        )
    return (
        sys.executable,
        str(WATCHDOG),
        "--limit-gib",
        f"{MEMORY_LIMIT_GIB:g}",
        "--",
        *command,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_json(
    command: Sequence[str],
    *,
    timeout: float,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], float, str]:
    """Run a JSON-producing generation/sample command under the watchdog."""

    guarded = _guarded_command(command)
    started = time.monotonic()
    process = subprocess.Popen(
        guarded,
        cwd=ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RegressionError(
            f"command exceeded {timeout:.1f}s: {' '.join(command)}"
        ) from error
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit code {process.returncode}"
        raise RegressionError(f"command failed: {' '.join(command)}\n{detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RegressionError(
            f"command did not emit JSON: {' '.join(command)}\n{stdout[-2000:]}"
        ) from error
    if not isinstance(payload, dict):
        raise RegressionError("command JSON result must be an object")
    return payload, elapsed, stderr


def _generation_command(
    python: Path,
    *,
    process: str,
    artifact: Path,
    model: str,
    color: str,
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
) -> tuple[str, ...]:
    # Do not pass --execution-mode: the frozen pre-branch CLI has no such flag,
    # and compiled is the default in both interpreters.
    return (
        str(python),
        "-m",
        "pyamplicol",
        "generate",
        process,
        str(artifact),
        "--model",
        model,
        "--backend",
        "jit",
        "--jit-optimization-level",
        "3",
        "--color-accuracy",
        color,
        "--lc-flow-layout",
        lc_flow_layout,
        "--mode",
        "replace",
        "--workers",
        "1",
        "--validation",
        "--validation-samples",
        "1",
        "--validation-seed",
        str(VALIDATION_SEED),
        "--no-post-build-validation",
        "--no-emit-api-bundle",
        "--progress",
        "off",
        "--format",
        "json",
    )


def _profile_command(
    python: Path,
    *,
    artifact: Path,
    process_id: str,
    batch_size: int,
    target_runtime: float,
    minimum_samples: int,
    warmup_runs: int,
    helicities: Sequence[str],
    color_flows: Sequence[str],
) -> tuple[str, ...]:
    command = [
        str(python),
        str(NATIVE_SAMPLE_HELPER),
        str(artifact),
        "--process",
        process_id,
        "--target-runtime",
        str(target_runtime),
        "--batch-size",
        str(batch_size),
        "--warmup-runs",
        str(warmup_runs),
        "--minimum-samples",
        str(minimum_samples),
    ]
    for helicity in helicities:
        command.extend(("--helicity", helicity))
    for color_flow in color_flows:
        command.extend(("--color-flow", color_flow))
    return tuple(command)


def _watchdog_peak_gib(stderr: str) -> float | None:
    matches = re.findall(r"peak_rss=([0-9]+(?:\.[0-9]+)?) GiB", stderr)
    return float(matches[-1]) if matches else None


def _watchdog_marker(stderr: str) -> str | None:
    markers = [
        line.strip()
        for line in stderr.splitlines()
        if line.startswith("memory-watchdog: command finished")
    ]
    return markers[-1] if markers else None


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegressionError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RegressionError(f"{label} must be a JSON object: {path}")
    return payload


def _artifact_lc_flow_layout(
    manifest: Mapping[str, Any],
    *,
    process_id: str,
) -> str | None:
    extensions = manifest.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    generation = extensions.get("generation")
    if not isinstance(generation, Mapping):
        return None
    processes = generation.get("concrete_processes")
    if not isinstance(processes, list):
        return None
    for process in processes:
        if not isinstance(process, Mapping) or process.get("id") != process_id:
            continue
        filters = process.get("filters")
        if not isinstance(filters, Mapping):
            return None
        layout = filters.get("lc_flow_layout")
        return layout if isinstance(layout, str) and layout else None
    return None


def _artifact_payload_digests(
    manifest: Mapping[str, Any],
    *,
    artifact: Path,
) -> list[dict[str, object]]:
    payloads = manifest.get("payloads")
    if payloads is None:
        return []
    if not isinstance(payloads, list):
        raise RegressionError("artifact payload inventory must be a list")
    try:
        artifact_root = artifact.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot resolve artifact root: {artifact}") from error
    result: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise RegressionError("artifact payload inventory entry is invalid")
        path = payload.get("path")
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise RegressionError("artifact payload digest entry is invalid")
        try:
            payload_path = (artifact / path).resolve(strict=True)
            payload_path.relative_to(artifact_root)
        except (OSError, ValueError) as error:
            raise RegressionError(
                f"artifact payload path is invalid: {path!r}"
            ) from error
        observed_size = payload_path.stat().st_size
        observed_sha256 = _sha256_file(payload_path)
        if observed_size != size_bytes or observed_sha256 != sha256:
            raise RegressionError(
                f"artifact payload does not match its declared digest: {path!r}"
            )
        identity: dict[str, object] = {
            "path": path,
            "sha256": observed_sha256,
            "size_bytes": observed_size,
            "role": payload.get("role"),
        }
        process_id = payload.get("process_id")
        if process_id is not None:
            identity["process_id"] = process_id
        result.append(identity)
    return sorted(result, key=lambda entry: str(entry["path"]))


def _artifact_metadata(
    artifact: Path,
    *,
    expected_process: str,
    expected_color: str,
    expected_lc_flow_layout: str | None = None,
) -> dict[str, Any]:
    manifest_path = artifact / "artifact.json"
    outer = _json_object(manifest_path, label="artifact manifest")
    processes = outer.get("processes")
    if not isinstance(processes, list) or len(processes) != 1:
        raise RegressionError(
            f"regression artifact must contain exactly one process: {artifact}"
        )
    record = processes[0]
    if not isinstance(record, dict):
        raise RegressionError(f"artifact process metadata is invalid: {artifact}")
    process_id = record.get("id")
    expression = record.get("expression")
    color = record.get("color_accuracy")
    if not isinstance(process_id, str) or not process_id:
        raise RegressionError(f"artifact process ID is invalid: {artifact}")
    if not isinstance(expression, str) or (
        " ".join(expression.split()).casefold()
        != " ".join(expected_process.split()).casefold()
    ):
        raise RegressionError(
            f"artifact process does not match {expected_process!r}: {artifact}"
        )
    if color != expected_color:
        raise RegressionError(
            f"artifact color accuracy does not match {expected_color!r}: {artifact}"
        )
    capabilities = record.get("required_runtime_capabilities", ())
    if not isinstance(capabilities, list) or any(
        "eager-dag" in str(capability) for capability in capabilities
    ):
        raise RegressionError(f"artifact is not compiled mode: {artifact}")
    artifact_id = outer.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise RegressionError(f"artifact ID is invalid: {artifact}")
    lc_flow_layout = _artifact_lc_flow_layout(outer, process_id=process_id)
    if (
        expected_lc_flow_layout is not None
        and lc_flow_layout != expected_lc_flow_layout
    ):
        raise RegressionError(
            "artifact LC flow layout does not match "
            f"{expected_lc_flow_layout!r}: got {lc_flow_layout!r} in {artifact}"
        )
    tree_identity = _tree_identity(artifact)
    payload_digests = _artifact_payload_digests(outer, artifact=artifact)
    return {
        "artifact_id": artifact_id,
        "path": str(artifact),
        "process_id": process_id,
        "process_expression": expression,
        "color_accuracy": color,
        "lc_flow_layout": lc_flow_layout,
        "manifest_sha256": _sha256_file(manifest_path),
        "tree_identity": tree_identity,
        "payload_digests": payload_digests,
        "payload_digest_count": len(payload_digests),
        "size_bytes": tree_identity["size_bytes"],
        "producer": outer.get("producer"),
    }


def _path_identity(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot inspect path identity: {path}") from error
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _model_identity(model: str) -> dict[str, object]:
    path = Path(model).expanduser()
    if not path.exists():
        return {"argument": model, "path_backed": False}
    try:
        absolute = _absolute_path(path)
        resolved = absolute.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise RegressionError(f"cannot inspect model identity: {model}") from error
    result: dict[str, object] = {
        "argument": model,
        "path_backed": True,
        "path": str(absolute),
        "resolved_path": str(resolved),
        "is_directory": resolved.is_dir(),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }
    if resolved.is_dir():
        tree = _tree_identity(resolved)
        result["tree_identity"] = tree
        result["size_bytes"] = tree["size_bytes"]
    elif resolved.is_file():
        result["sha256"] = _sha256_file(resolved)
    else:
        raise RegressionError(f"model path is not a file or directory: {model}")
    return result


def _generation_signature(
    python: Path,
    *,
    installation_identity: Mapping[str, Any],
    process: str,
    model: str,
    color: str,
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
    artifact: Path | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "python": _path_identity(python),
        "installed_pyamplicol": dict(installation_identity),
        "process": process,
        "model": _model_identity(model),
        "color_accuracy": color,
        "lc_flow_layout": lc_flow_layout,
        "execution_mode": "compiled-default",
        "backend": "jit",
        "jit_optimization_level": 3,
        "workers": 1,
        "validation_samples": 1,
        "validation_seed": VALIDATION_SEED,
        "post_build_validation": False,
        "emit_api_bundle": False,
    }
    if artifact is not None:
        result["generation_command"] = _command_identity(
            _generation_command(
                python,
                process=process,
                artifact=artifact,
                model=model,
                color=color,
                lc_flow_layout=lc_flow_layout,
            )
        )
    return result


def _installed_pyamplicol_identity(
    python: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = (str(python), str(NATIVE_SAMPLE_HELPER), "--installation-identity")
    payload, _elapsed, _stderr = _run_json(
        command,
        timeout=DEFAULT_PROFILE_TIMEOUT,
        environment=environment,
    )
    _require_equal(
        payload,
        "kind",
        INSTALLATION_IDENTITY_KIND,
        label="installed pyamplicol identity kind",
    )
    _require_equal(
        payload,
        "schema_version",
        INSTALLATION_IDENTITY_SCHEMA_VERSION,
        label="installed pyamplicol identity schema_version",
    )
    distribution_content = payload.get("distribution_content")
    if not isinstance(distribution_content, Mapping):
        raise RegressionError(
            "installed pyamplicol identity has no distribution content identity"
        )
    digest = distribution_content.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RegressionError(
            "installed pyamplicol identity has an invalid distribution SHA-256"
        )
    native_modules = payload.get("native_modules")
    if (
        not isinstance(native_modules, Sequence)
        or isinstance(native_modules, (str, bytes))
        or len(native_modules) != 1
    ):
        raise RegressionError(
            "installed pyamplicol identity must contain one native module"
        )
    return dict(payload)


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_artifact(
    lane: str,
    python: Path,
    *,
    output_root: Path,
    process: str,
    model: str,
    color: str,
    generation_timeout: float,
    regenerate: bool,
    environment: Mapping[str, str],
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
) -> dict[str, Any]:
    lane_root = output_root / lane
    artifact = lane_root / "artifact"
    cache_path = lane_root / "artifact-cache.json"
    generation_command = _generation_command(
        python,
        process=process,
        artifact=artifact,
        model=model,
        color=color,
        lc_flow_layout=lc_flow_layout,
    )
    command_identity = _command_identity(generation_command)
    installation_identity = _installed_pyamplicol_identity(
        python,
        environment=environment,
    )
    signature = _generation_signature(
        python,
        installation_identity=installation_identity,
        process=process,
        model=model,
        color=color,
        lc_flow_layout=lc_flow_layout,
        artifact=artifact,
    )
    cache = None if regenerate else _read_cache(cache_path)
    if (
        cache is not None
        and cache.get("kind") == CACHE_KIND
        and cache.get("schema_version") == SCHEMA_VERSION
        and cache.get("signature") == signature
    ):
        try:
            metadata = _artifact_metadata(
                artifact,
                expected_process=process,
                expected_color=color,
                expected_lc_flow_layout=lc_flow_layout,
            )
        except RegressionError:
            pass
        else:
            tree_identity = metadata["tree_identity"]
            assert isinstance(tree_identity, Mapping)
            if cache.get("artifact_id") == metadata["artifact_id"] and cache.get(
                "artifact_tree_sha256"
            ) == tree_identity.get("sha256"):
                return {
                    **metadata,
                    "reused": True,
                    "generation": None,
                    "generation_command": command_identity,
                    "cache_path": str(cache_path),
                }

    lane_root.mkdir(parents=True, exist_ok=True)
    payload, elapsed, stderr = _run_json(
        generation_command,
        timeout=generation_timeout,
        environment=environment,
    )
    metadata = _artifact_metadata(
        artifact,
        expected_process=process,
        expected_color=color,
        expected_lc_flow_layout=lc_flow_layout,
    )
    tree_identity = metadata["tree_identity"]
    assert isinstance(tree_identity, Mapping)
    _write_json_atomic(
        cache_path,
        {
            "kind": CACHE_KIND,
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "artifact_id": metadata["artifact_id"],
            "artifact_tree_sha256": tree_identity["sha256"],
        },
    )
    return {
        **metadata,
        "reused": False,
        "generation": {
            "command": command_identity,
            "guarded_command": _command_identity(_guarded_command(generation_command)),
            "command_elapsed_seconds": elapsed,
            "peak_rss_gib": _watchdog_peak_gib(stderr),
            "watchdog": _watchdog_marker(stderr),
            "output": payload.get("output"),
            "schema_version": payload.get("schema_version"),
        },
        "generation_command": command_identity,
        "cache_path": str(cache_path),
    }


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RegressionError(f"{label} must be positive and finite")
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RegressionError(f"{label} must be finite and non-negative")
    return result


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RegressionError(f"{label} must be finite")
    return result


def _positive_mapping_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegressionError(f"{label} must be a positive integer")
    return value


def _require_equal(
    mapping: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    label: str,
) -> None:
    value = mapping.get(key)
    if value != expected or type(value) is not type(expected):
        raise RegressionError(f"{label} must be {expected!r}, got {value!r}")


def _native_profile_sample(
    payload: Mapping[str, Any],
    *,
    minimum_samples: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    _require_equal(
        payload,
        "kind",
        NATIVE_SAMPLE_RESULT_KIND,
        label="native sample kind",
    )
    _require_equal(
        payload,
        "schema_version",
        NATIVE_SAMPLE_SCHEMA_VERSION,
        label="native sample schema_version",
    )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise RegressionError("profile result has no environment metadata")
    source = environment.get("wall_time_source")
    if source != NATIVE_WALL_TIME_SOURCE:
        raise RegressionError(
            f"profile did not use the native Rusticol repeated wall timer: {source!r}"
        )
    _require_equal(
        environment,
        "wall_time_sample_pass",
        NATIVE_WALL_TIME_SAMPLE_PASS,
        label="profile wall_time_sample_pass",
    )
    _require_equal(
        environment,
        "timing_sample_contract",
        PAIRED_TIMING_SAMPLE_CONTRACT,
        label="profile timing_sample_contract",
    )
    for key in ("evaluator_time_sample_pass", "timing_breakdown_sample_pass"):
        _require_equal(
            environment,
            key,
            PROFILE_ATTRIBUTION_SAMPLE_PASS,
            label=f"profile {key}",
        )
    for key in (
        "profile_attribution_paired_with_headline",
        "profile_attribution_identical_batch",
        "profile_attribution_identical_repetitions",
    ):
        _require_equal(
            environment,
            key,
            True,
            label=f"profile {key}",
        )
    _require_equal(
        environment,
        "execution_mode",
        "compiled",
        label="profile execution_mode",
    )
    sample_count = payload.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < minimum_samples
    ):
        raise RegressionError(
            "profile result contains fewer complete timing blocks than requested"
        )
    headline_samples = payload.get("wall_time_samples_seconds_per_point")
    if not isinstance(headline_samples, Sequence) or isinstance(
        headline_samples,
        (str, bytes),
    ):
        raise RegressionError("native sample has no headline wall sample vector")
    if len(headline_samples) != sample_count:
        raise RegressionError(
            "native headline wall sample vector does not match sample_count"
        )
    headline_values = [
        _finite_positive(value, label="native headline wall sample")
        for value in headline_samples
    ]
    _require_equal(
        payload,
        "interrupted",
        False,
        label="profile interrupted",
    )
    _require_equal(
        environment,
        "interrupted",
        False,
        label="profile environment interrupted",
    )
    repetitions = _positive_mapping_int(
        payload,
        "repetitions_per_sample",
        label="profile repetitions_per_sample",
    )
    reported_batch_size = _positive_mapping_int(
        environment,
        "batch_size",
        label="profile environment batch_size",
    )
    if batch_size is not None and reported_batch_size != batch_size:
        raise RegressionError(
            "profile environment batch_size does not match the requested batch: "
            f"{reported_batch_size} != {batch_size}"
        )
    expected_evaluations = sample_count * repetitions
    expected_points = expected_evaluations * reported_batch_size
    expected_points_per_sample = repetitions * reported_batch_size
    exact_counts = {
        "completed_sample_count": sample_count,
        "planned_sample_count": sample_count,
        "measured_evaluation_count": expected_evaluations,
        "measured_point_count": expected_points,
        "native_profile_sample_count": sample_count,
        "native_profile_sample_limit": sample_count,
        "native_profile_repetitions_per_sample": repetitions,
        "native_profile_points_per_sample": expected_points_per_sample,
        "profile_attribution_evaluation_count": expected_evaluations,
        "profile_attribution_point_count": expected_points,
    }
    for key, expected in exact_counts.items():
        _require_equal(
            environment,
            key,
            expected,
            label=f"profile environment {key}",
        )
    calls_per_block = _finite_positive(
        environment.get("native_profile_calls_per_block"),
        label="profile environment native_profile_calls_per_block",
    )
    if calls_per_block != 1.0:
        raise RegressionError(
            "profile native attribution must contain exactly one paired call "
            f"per headline block, got {calls_per_block!r}"
        )
    timing_breakdown = payload.get("timing_breakdown")
    if not isinstance(timing_breakdown, Mapping):
        raise RegressionError("profile result has no paired native timing breakdown")
    _require_equal(
        timing_breakdown,
        "sample_count",
        sample_count,
        label="profile timing breakdown sample_count",
    )
    _require_equal(
        timing_breakdown,
        "execution_mode",
        "compiled",
        label="profile timing breakdown execution_mode",
    )
    profile_wall = timing_breakdown.get("wall_time")
    if not isinstance(profile_wall, Mapping):
        raise RegressionError("profile timing breakdown has no profiled wall component")
    profile_wall_seconds_per_point = _finite_positive(
        profile_wall.get("mean_seconds_per_point"),
        label="paired profiled wall mean_seconds_per_point",
    )
    profile_evaluator_seconds_per_point = _finite_nonnegative(
        payload.get("evaluator_time_per_point"),
        label="paired profile evaluator_time_per_point",
    )
    headline_mean = _finite_positive(
        payload.get("wall_time_per_point"),
        label="profile wall_time_per_point",
    )
    computed_headline_mean = float(statistics.fmean(headline_values))
    if not math.isclose(
        headline_mean,
        computed_headline_mean,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    ):
        raise RegressionError(
            "native headline mean does not match its unprofiled sample vector"
        )
    raw_profiles = timing_breakdown.get("raw_profile_samples")
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
        raise RegressionError("paired timing breakdown has no raw profile samples")
    if len(raw_profiles) != sample_count:
        raise RegressionError(
            "paired raw profile sample vector does not match sample_count"
        )
    runtime_identity = environment.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise RegressionError("native sample has no runtime identity")
    native_module_identity = runtime_identity.get("native_module")
    if not isinstance(native_module_identity, Mapping):
        raise RegressionError("native sample has no native module identity")
    native_module_sha256 = native_module_identity.get("sha256")
    if (
        not isinstance(native_module_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", native_module_sha256) is None
    ):
        raise RegressionError("native sample has an invalid native module SHA-256")
    batch_sha256 = environment.get("batch_sha256")
    if (
        not isinstance(batch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", batch_sha256) is None
    ):
        raise RegressionError("native sample has an invalid batch SHA-256")
    helicities = environment.get("helicities")
    color_flows = environment.get("color_flows")
    if (
        not isinstance(helicities, Sequence)
        or isinstance(helicities, (str, bytes))
        or not all(isinstance(value, str) for value in helicities)
    ):
        raise RegressionError("native sample has invalid helicity selectors")
    if (
        not isinstance(color_flows, Sequence)
        or isinstance(color_flows, (str, bytes))
        or not all(isinstance(value, str) for value in color_flows)
    ):
        raise RegressionError("native sample has invalid color-flow selectors")
    warmed_numerical_result = payload.get("warmed_numerical_result")
    if not isinstance(warmed_numerical_result, Mapping):
        raise RegressionError("native sample has no warmed numerical result")
    numerical_point_count = _positive_mapping_int(
        warmed_numerical_result,
        "point_count",
        label="warmed numerical result point_count",
    )
    if numerical_point_count > reported_batch_size:
        raise RegressionError(
            "warmed numerical result contains more points than the sampled batch"
        )
    numerical_batch_sha256 = warmed_numerical_result.get("batch_sha256")
    if (
        not isinstance(numerical_batch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", numerical_batch_sha256) is None
    ):
        raise RegressionError("warmed numerical result has an invalid batch SHA-256")
    numerical_values = warmed_numerical_result.get("values_f64")
    numerical_hex = warmed_numerical_result.get("values_f64_hex")
    if (
        not isinstance(numerical_values, Sequence)
        or isinstance(numerical_values, (str, bytes))
        or not isinstance(numerical_hex, Sequence)
        or isinstance(numerical_hex, (str, bytes))
        or len(numerical_values) != numerical_point_count
        or len(numerical_hex) != numerical_point_count
    ):
        raise RegressionError("warmed numerical result has invalid value vectors")
    validated_values: list[float] = []
    validated_hex: list[str] = []
    for raw_value, raw_hex in zip(numerical_values, numerical_hex, strict=True):
        value = _finite_number(
            raw_value,
            label="warmed numerical result value",
        )
        if not isinstance(raw_hex, str) or raw_hex != value.hex():
            raise RegressionError(
                "warmed numerical result has an invalid raw f64 representation"
            )
        validated_values.append(value)
        validated_hex.append(raw_hex)
    for key, selector_expected in (
        ("helicities", list(helicities)),
        ("color_flows", list(color_flows)),
    ):
        if warmed_numerical_result.get(key) != selector_expected:
            raise RegressionError(
                f"warmed numerical result {key} do not match timed selectors"
            )
    return {
        "wall_seconds_per_point": headline_mean,
        "wall_samples_seconds_per_point": headline_values,
        "native_wall_time_source": source,
        "native_wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
        "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
        "profile_timed_block_count": sample_count,
        "repetitions_per_timed_block": repetitions,
        "batch_size": reported_batch_size,
        "measured_evaluation_count": expected_evaluations,
        "measured_point_count": expected_points,
        "native_elapsed_seconds": environment.get("elapsed_seconds"),
        "profile_uncertainty": payload.get("uncertainty"),
        "paired_profile_wall_seconds_per_point": (profile_wall_seconds_per_point),
        "paired_profile_evaluator_seconds_per_point": (
            profile_evaluator_seconds_per_point
        ),
        "paired_profile_evaluator_uncertainty": payload.get("evaluator_uncertainty"),
        "paired_profile_timing_breakdown": timing_breakdown,
        "runtime_identity": runtime_identity,
        "native_module_sha256": native_module_sha256,
        "batch_sha256": batch_sha256,
        "helicities": list(helicities),
        "color_flows": list(color_flows),
        "warmed_numerical_result": {
            "point_count": numerical_point_count,
            "batch_sha256": numerical_batch_sha256,
            "values_f64": validated_values,
            "values_f64_hex": validated_hex,
            "helicities": list(helicities),
            "color_flows": list(color_flows),
        },
    }


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        raise RegressionError("timing distribution must contain samples")
    values = tuple(_finite_positive(value, label="timing sample") for value in samples)
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return {
        "sample_count": len(values),
        "samples_seconds_per_point": list(values),
        "median_seconds_per_point": median,
        "mad_seconds_per_point": mad,
        "minimum_seconds_per_point": min(values),
        "maximum_seconds_per_point": max(values),
    }


def _regression_gate(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_median = _finite_positive(
        baseline.get("median_seconds_per_point"),
        label="baseline median",
    )
    current_median = _finite_positive(
        current.get("median_seconds_per_point"),
        label="current median",
    )
    raw_mad = baseline.get("mad_seconds_per_point")
    if isinstance(raw_mad, bool) or not isinstance(raw_mad, (float, int)):
        raise RegressionError("baseline MAD must be numeric")
    baseline_mad = float(raw_mad)
    if not math.isfinite(baseline_mad) or baseline_mad < 0.0:
        raise RegressionError("baseline MAD must be finite and non-negative")
    relative_limit = baseline_median * (1.0 + RELATIVE_TOLERANCE)
    mad_limit = baseline_median + MAD_MULTIPLIER * baseline_mad
    within_relative = current_median <= relative_limit
    within_mad = current_median <= mad_limit
    return {
        "relative_tolerance": RELATIVE_TOLERANCE,
        "mad_multiplier": MAD_MULTIPLIER,
        "baseline_median_seconds_per_point": baseline_median,
        "baseline_mad_seconds_per_point": baseline_mad,
        "current_median_seconds_per_point": current_median,
        "current_over_baseline": current_median / baseline_median,
        "relative_change": (current_median - baseline_median) / baseline_median,
        "two_percent_upper_bound_seconds_per_point": relative_limit,
        "three_baseline_mad_upper_bound_seconds_per_point": mad_limit,
        "within_two_percent": within_relative,
        "within_three_baseline_mad": within_mad,
        "passes": within_relative and within_mad,
    }


def _performance_payload_identity(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    payloads = artifact.get("payload_digests")
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        return None
    relevant: list[dict[str, object]] = []
    observed_roles: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, Mapping):
            return None
        role = payload.get("role")
        if role not in PERFORMANCE_RELEVANT_PAYLOAD_ROLES:
            continue
        path = payload.get("path")
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes")
        if (
            not isinstance(role, str)
            or not isinstance(path, str)
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            return None
        observed_roles.add(role)
        relevant.append(
            {
                "path": path,
                "role": role,
                "process_id": payload.get("process_id"),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if not REQUIRED_PERFORMANCE_PAYLOAD_ROLES.issubset(observed_roles):
        return None
    relevant.sort(key=lambda value: (str(value["role"]), str(value["path"])))
    identity_basis = {
        "process_id": artifact.get("process_id"),
        "color_accuracy": artifact.get("color_accuracy"),
        "lc_flow_layout": artifact.get("lc_flow_layout"),
        "payloads": relevant,
    }
    return {
        "algorithm": "sha256-performance-relevant-artifact-payloads-v1",
        "sha256": _canonical_sha256(identity_basis),
        "roles": sorted(observed_roles),
        "payload_count": len(relevant),
        "identity_basis": identity_basis,
    }


def _performance_authority(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    shared_artifact: bool,
) -> dict[str, Any]:
    if shared_artifact:
        return {
            "authoritative": True,
            "basis": "single-read-only-shared-artifact",
            "comparison_mode": "shared-artifact",
            "payload_identities": None,
        }
    identities = {
        lane: _performance_payload_identity(artifact)
        for lane, artifact in artifacts.items()
    }
    if any(identity is None for identity in identities.values()):
        return {
            "authoritative": False,
            "basis": "per-lane-artifact-performance-payload-identity-unproven",
            "comparison_mode": "independently-generated-artifacts",
            "payload_identities": identities,
        }
    digests = {
        str(identity["sha256"])
        for identity in identities.values()
        if identity is not None
    }
    matches = len(digests) == 1
    return {
        "authoritative": matches,
        "basis": (
            "matching-performance-relevant-payload-identities"
            if matches
            else "per-lane-artifact-performance-payload-identities-differ"
        ),
        "comparison_mode": "independently-generated-artifacts",
        "payload_identities": identities,
    }


def _correctness_gate(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_reference = next(
        (
            measurement
            for measurement in measurements
            if measurement.get("lane") == "baseline"
        ),
        None,
    )
    if baseline_reference is None:
        raise RegressionError("correctness comparison has no baseline measurement")
    reference = baseline_reference.get("warmed_numerical_result")
    if not isinstance(reference, Mapping):
        raise RegressionError("baseline measurement has no warmed numerical result")
    reference_values = reference.get("values_f64")
    if not isinstance(reference_values, Sequence) or isinstance(
        reference_values,
        (str, bytes),
    ):
        raise RegressionError("baseline warmed numerical result is invalid")
    comparisons: list[dict[str, Any]] = []
    all_pass = True
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for measurement in measurements:
        candidate = measurement.get("warmed_numerical_result")
        if not isinstance(candidate, Mapping):
            raise RegressionError("measurement has no warmed numerical result")
        for key in ("point_count", "batch_sha256", "helicities", "color_flows"):
            if candidate.get(key) != reference.get(key):
                raise RegressionError(
                    "warmed numerical results do not share identical inputs: "
                    f"{key} differs"
                )
        candidate_values = candidate.get("values_f64")
        if (
            not isinstance(candidate_values, Sequence)
            or isinstance(candidate_values, (str, bytes))
            or len(candidate_values) != len(reference_values)
        ):
            raise RegressionError("warmed numerical result shape differs")
        comparison_passes = True
        comparison_max_absolute = 0.0
        comparison_max_relative = 0.0
        for raw_expected, raw_observed in zip(
            reference_values,
            candidate_values,
            strict=True,
        ):
            expected = _finite_number(
                raw_expected,
                label="baseline correctness value",
            )
            observed = _finite_number(
                raw_observed,
                label="candidate correctness value",
            )
            absolute_error = abs(observed - expected)
            relative_error = absolute_error / max(
                abs(expected),
                CORRECTNESS_ABSOLUTE_TOLERANCE,
            )
            comparison_max_absolute = max(
                comparison_max_absolute,
                absolute_error,
            )
            comparison_max_relative = max(
                comparison_max_relative,
                relative_error,
            )
            comparison_passes &= math.isclose(
                observed,
                expected,
                rel_tol=CORRECTNESS_RELATIVE_TOLERANCE,
                abs_tol=CORRECTNESS_ABSOLUTE_TOLERANCE,
            )
        maximum_absolute_error = max(
            maximum_absolute_error,
            comparison_max_absolute,
        )
        maximum_relative_error = max(
            maximum_relative_error,
            comparison_max_relative,
        )
        all_pass &= comparison_passes
        comparisons.append(
            {
                "lane": measurement.get("lane"),
                "pair_index": measurement.get("pair_index"),
                "passes": comparison_passes,
                "maximum_absolute_error": comparison_max_absolute,
                "maximum_relative_error": comparison_max_relative,
            }
        )
    return {
        "relative_tolerance": CORRECTNESS_RELATIVE_TOLERANCE,
        "absolute_tolerance": CORRECTNESS_ABSOLUTE_TOLERANCE,
        "reference_lane": "baseline",
        "reference_pair_index": baseline_reference.get("pair_index"),
        "comparison_count": len(comparisons),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "comparisons": comparisons,
        "passes": all_pass,
    }


def _resource_summary(
    artifacts: Mapping[str, Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    peaks: list[float] = []
    for artifact in artifacts.values():
        generation = artifact.get("generation")
        if isinstance(generation, Mapping):
            peak = generation.get("peak_rss_gib")
            if isinstance(peak, (float, int)) and not isinstance(peak, bool):
                peaks.append(float(peak))
    for measurement in measurements:
        peak = measurement.get("peak_rss_gib")
        if isinstance(peak, (float, int)) and not isinstance(peak, bool):
            peaks.append(float(peak))
    return {
        "memory_limit_gib": MEMORY_LIMIT_GIB,
        "maximum_observed_peak_rss_gib": max(peaks, default=None),
        "generation_subprocess_count": sum(
            not bool(artifact.get("reused")) for artifact in artifacts.values()
        ),
        "sample_subprocess_count": len(measurements),
        "profile_subprocess_count": len(measurements),
        "elapsed_seconds": elapsed_seconds,
    }


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONHASHSEED"] = "0"
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    return environment


def run_regression(arguments: argparse.Namespace) -> dict[str, Any]:
    baseline_python = _require_interpreter(
        arguments.baseline_python,
        lane="baseline",
    )
    current_python = _require_interpreter(arguments.current_python, lane="current")
    output_root = _absolute_path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = _model_argument(arguments.model)
    lc_flow_layout = getattr(
        arguments,
        "lc_flow_layout",
        DEFAULT_LC_FLOW_LAYOUT,
    )
    shared_artifact_argument = getattr(arguments, "shared_artifact", None)
    if shared_artifact_argument is not None and arguments.regenerate_artifacts:
        raise RegressionError(
            "--shared-artifact and --regenerate-artifacts are mutually exclusive"
        )
    shared_artifact = (
        None
        if shared_artifact_argument is None
        else _absolute_path(shared_artifact_argument)
    )
    environment = _environment()
    started = time.monotonic()

    interpreters = {
        "baseline": baseline_python,
        "current": current_python,
    }
    provenance = {
        "driver": _path_identity(Path(__file__)),
        "watchdog": _path_identity(WATCHDOG),
        "native_sample_helper": _path_identity(NATIVE_SAMPLE_HELPER),
        "interpreters": {
            lane: _path_identity(python) for lane, python in interpreters.items()
        },
        "model": _model_identity(model),
    }
    if shared_artifact is None:
        artifacts = {
            lane: _ensure_artifact(
                lane,
                python,
                output_root=output_root,
                process=arguments.process,
                model=model,
                color=arguments.color,
                generation_timeout=arguments.generation_timeout,
                regenerate=arguments.regenerate_artifacts,
                environment=environment,
                lc_flow_layout=lc_flow_layout,
            )
            for lane, python in interpreters.items()
        }
        for artifact in artifacts.values():
            artifact["comparison_role"] = "independently-generated-lane-artifact"
        shared_artifact_identity: dict[str, Any] | None = None
    else:
        shared_artifact_identity = _artifact_metadata(
            shared_artifact,
            expected_process=arguments.process,
            expected_color=arguments.color,
            expected_lc_flow_layout=lc_flow_layout,
        )
        artifacts = {
            lane: {
                **shared_artifact_identity,
                "shared": True,
                "reused": True,
                "generation": None,
                "generation_command": None,
                "cache_path": None,
                "comparison_role": "shared-read-only-artifact",
            }
            for lane in interpreters
        }
    partial: dict[str, Any] = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "provenance": provenance,
        "artifacts": artifacts,
        "measurements": [],
    }
    result_path = output_root / "result.json"
    _write_json_atomic(result_path, partial)

    measurements: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {"baseline": [], "current": []}
    native_module_sha256_by_lane: dict[str, str] = {}
    pair_orders: list[list[str]] = []
    for pair_index in range(arguments.samples):
        order = (
            ("baseline", "current") if pair_index % 2 == 0 else ("current", "baseline")
        )
        pair_orders.append(list(order))
        for order_index, lane in enumerate(order):
            artifact = artifacts[lane]
            profile_command = _profile_command(
                interpreters[lane],
                artifact=Path(str(artifact["path"])),
                process_id=str(artifact["process_id"]),
                batch_size=arguments.batch_size,
                target_runtime=arguments.target_runtime,
                minimum_samples=arguments.minimum_samples,
                warmup_runs=arguments.warmup_runs,
                helicities=arguments.helicity,
                color_flows=arguments.color_flow,
            )
            payload, command_elapsed, stderr = _run_json(
                profile_command,
                timeout=arguments.profile_timeout,
                environment=environment,
            )
            sample = _native_profile_sample(
                payload,
                minimum_samples=arguments.minimum_samples,
                batch_size=arguments.batch_size,
            )
            native_module_sha256 = str(sample["native_module_sha256"])
            expected_native_module_sha256 = native_module_sha256_by_lane.setdefault(
                lane,
                native_module_sha256,
            )
            if native_module_sha256 != expected_native_module_sha256:
                raise RegressionError(
                    f"{lane} native module changed during sampling: "
                    f"{native_module_sha256} != {expected_native_module_sha256}"
                )
            if measurements:
                reference = measurements[0]
                for key in ("batch_sha256", "helicities", "color_flows"):
                    if sample[key] != reference[key]:
                        raise RegressionError(
                            "native samples do not share byte-identical inputs: "
                            f"{key} differs in {lane} pair {pair_index + 1}"
                        )
            wall = float(sample["wall_seconds_per_point"])
            values[lane].append(wall)
            measurements.append(
                {
                    "pair_index": pair_index + 1,
                    "measurement_order": order_index + 1,
                    "lane": lane,
                    "command": _command_identity(profile_command),
                    "guarded_command": _command_identity(
                        _guarded_command(profile_command)
                    ),
                    **sample,
                    "command_elapsed_seconds": command_elapsed,
                    "peak_rss_gib": _watchdog_peak_gib(stderr),
                    "watchdog": _watchdog_marker(stderr),
                }
            )
        partial["measurements"] = measurements
        partial["pair_orders"] = pair_orders
        _write_json_atomic(result_path, partial)

    if shared_artifact_identity is not None:
        assert shared_artifact is not None
        observed_identity = _artifact_metadata(
            shared_artifact,
            expected_process=arguments.process,
            expected_color=arguments.color,
            expected_lc_flow_layout=lc_flow_layout,
        )
        for key in (
            "artifact_id",
            "manifest_sha256",
            "tree_identity",
            "payload_digests",
        ):
            if observed_identity[key] != shared_artifact_identity[key]:
                raise RegressionError(
                    f"shared artifact changed during sampling: {key} differs"
                )
    else:
        for lane, initial_identity in artifacts.items():
            observed_identity = _artifact_metadata(
                Path(str(initial_identity["path"])),
                expected_process=arguments.process,
                expected_color=arguments.color,
                expected_lc_flow_layout=lc_flow_layout,
            )
            for key in (
                "artifact_id",
                "manifest_sha256",
                "tree_identity",
                "payload_digests",
            ):
                if observed_identity[key] != initial_identity[key]:
                    raise RegressionError(
                        f"{lane} artifact changed during sampling: {key} differs"
                    )
    distributions = {
        lane: _distribution(lane_values) for lane, lane_values in values.items()
    }
    authority = _performance_authority(
        artifacts,
        shared_artifact=shared_artifact_identity is not None,
    )
    measured_gate = _regression_gate(
        distributions["baseline"],
        distributions["current"],
    )
    measured_thresholds_pass = bool(measured_gate.pop("passes"))
    gate = {
        **measured_gate,
        "measured_thresholds_pass": measured_thresholds_pass,
        "authoritative": bool(authority["authoritative"]),
        "authority_basis": authority["basis"],
        "passes": (bool(authority["authoritative"]) and measured_thresholds_pass),
    }
    correctness_gate = _correctness_gate(measurements)
    elapsed = time.monotonic() - started
    result = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "performance_result_authoritative": bool(authority["authoritative"]),
        "passes": bool(gate["passes"] and correctness_gate["passes"]),
        "platform": platform.platform(),
        "provenance": provenance,
        "configuration": {
            "baseline_python": str(baseline_python),
            "current_python": str(current_python),
            "output_root": str(output_root),
            "process": arguments.process,
            "model": model,
            "color_accuracy": arguments.color,
            "lc_flow_layout": lc_flow_layout,
            "shared_artifact": (
                None if shared_artifact is None else str(shared_artifact)
            ),
            "batch_size": arguments.batch_size,
            "independent_samples_per_lane": arguments.samples,
            "target_runtime_per_profile_seconds": arguments.target_runtime,
            "target_runtime_per_native_sample_seconds": arguments.target_runtime,
            "minimum_native_timed_blocks_per_profile": arguments.minimum_samples,
            "warmup_runs_per_profile": arguments.warmup_runs,
            "generation_timeout_seconds": arguments.generation_timeout,
            "profile_timeout_seconds": arguments.profile_timeout,
            "native_sample_timeout_seconds": arguments.profile_timeout,
            "helicities": list(arguments.helicity),
            "color_flows": list(arguments.color_flow),
            "native_wall_time_source": NATIVE_WALL_TIME_SOURCE,
            "native_wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
            "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
        },
        "artifacts": artifacts,
        "pair_orders": pair_orders,
        "measurements": measurements,
        "native_module_sha256_by_lane": native_module_sha256_by_lane,
        "distributions": distributions,
        "performance_authority": authority,
        "gate": gate,
        "correctness_gate": correctness_gate,
        "resources": _resource_summary(
            artifacts,
            measurements,
            elapsed_seconds=elapsed,
        ),
    }
    _write_json_atomic(result_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The gate passes only when the current median is no more than 2% "
            "above the baseline median and no more than three baseline MAD above it."
        ),
    )
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--current-python", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--shared-artifact",
        type=Path,
        help=(
            "sample one existing read-only artifact in both lanes instead of "
            "generating per-lane artifacts"
        ),
    )
    result.add_argument("--process", required=True)
    result.add_argument("--model", default="built-in-sm")
    result.add_argument(
        "--color",
        "--color-accuracy",
        dest="color",
        choices=("lc", "nlc", "full"),
        default="lc",
    )
    result.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default=DEFAULT_LC_FLOW_LAYOUT,
        help="LC artifact layout generated identically in both lanes",
    )
    result.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    result.add_argument(
        "--samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help="independent native sampling subprocesses per interpreter (minimum: 5)",
    )
    result.add_argument(
        "--target-runtime",
        type=_positive_float,
        default=DEFAULT_TARGET_RUNTIME,
        help="native headline timing target for each sampling subprocess",
    )
    result.add_argument(
        "--minimum-samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help="minimum native timed blocks inside each sampling subprocess",
    )
    result.add_argument(
        "--warmup-runs",
        type=_positive_int,
        default=DEFAULT_WARMUP_RUNS,
    )
    result.add_argument("--helicity", action="append", default=[])
    result.add_argument("--color-flow", action="append", default=[])
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=DEFAULT_GENERATION_TIMEOUT,
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=DEFAULT_PROFILE_TIMEOUT,
    )
    result.add_argument(
        "--regenerate-artifacts",
        action="store_true",
        help="replace both cached compiled artifacts before sampling",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = run_regression(arguments)
    except (RegressionError, OSError, ValueError) as error:
        print(f"compiled-mode-regression: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
