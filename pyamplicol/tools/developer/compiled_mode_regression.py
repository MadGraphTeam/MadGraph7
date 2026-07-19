#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare compiled-mode runtime against a frozen baseline interpreter.

The driver generates one compiled JIT O3 artifact per interpreter and reuses
it while its cache signature remains valid.  Each reported outer sample comes
from a separate ``pyamplicol profile`` process.  The accepted timing is the
profile result marked ``runtime_core_repeated_wall_time`` by Rusticol, which
excludes Python and NumPy momentum packing from the timed region.
"""

from __future__ import annotations

import argparse
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
MEMORY_LIMIT_GIB = 30.0
NATIVE_WALL_TIME_SOURCE = "runtime_core_repeated_wall_time"
RESULT_KIND = "pyamplicol-compiled-mode-regression"
CACHE_KIND = "pyamplicol-compiled-mode-regression-artifact-cache"
SCHEMA_VERSION = 1
DEFAULT_GENERATION_TIMEOUT = 300.0
DEFAULT_PROFILE_TIMEOUT = 120.0
DEFAULT_TARGET_RUNTIME = 5.0
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WARMUP_RUNS = 2
RELATIVE_TOLERANCE = 0.02
MAD_MULTIPLIER = 3.0
VALIDATION_SEED = 20260719


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
    """Run a JSON-producing generation/profile command under the watchdog."""

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
        "-m",
        "pyamplicol",
        "profile",
        str(artifact),
        "--process",
        process_id,
        "--target-runtime",
        str(target_runtime),
        "--batch-size",
        str(batch_size),
        "--precision",
        "16",
        "--warmup-runs",
        str(warmup_runs),
        "--minimum-samples",
        str(minimum_samples),
    ]
    for helicity in helicities:
        command.extend(("--helicity", helicity))
    for color_flow in color_flows:
        command.extend(("--color-flow", color_flow))
    command.extend(("--progress", "off", "--format", "json"))
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


def _artifact_size_bytes(artifact: Path) -> int:
    return sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())


def _artifact_metadata(
    artifact: Path,
    *,
    expected_process: str,
    expected_color: str,
) -> dict[str, Any]:
    outer = _json_object(artifact / "artifact.json", label="artifact manifest")
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
        " ".join(expression.split()) != " ".join(expected_process.split())
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
    return {
        "artifact_id": artifact_id,
        "path": str(artifact),
        "process_id": process_id,
        "process_expression": expression,
        "color_accuracy": color,
        "size_bytes": _artifact_size_bytes(artifact),
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
    }


def _model_identity(model: str) -> dict[str, object]:
    path = Path(model)
    if not path.exists():
        return {"argument": model, "path_backed": False}
    stat = path.stat()
    return {
        "argument": model,
        "path_backed": True,
        "is_directory": path.is_dir(),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }


def _generation_signature(
    python: Path,
    *,
    process: str,
    model: str,
    color: str,
) -> dict[str, object]:
    return {
        "python": _path_identity(python),
        "process": process,
        "model": _model_identity(model),
        "color_accuracy": color,
        "execution_mode": "compiled-default",
        "backend": "jit",
        "jit_optimization_level": 3,
        "workers": 1,
        "validation_samples": 1,
        "validation_seed": VALIDATION_SEED,
        "post_build_validation": False,
        "emit_api_bundle": False,
    }


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
) -> dict[str, Any]:
    lane_root = output_root / lane
    artifact = lane_root / "artifact"
    cache_path = lane_root / "artifact-cache.json"
    signature = _generation_signature(
        python,
        process=process,
        model=model,
        color=color,
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
            )
        except RegressionError:
            pass
        else:
            if cache.get("artifact_id") == metadata["artifact_id"]:
                return {
                    **metadata,
                    "reused": True,
                    "generation": None,
                    "cache_path": str(cache_path),
                }

    lane_root.mkdir(parents=True, exist_ok=True)
    payload, elapsed, stderr = _run_json(
        _generation_command(
            python,
            process=process,
            artifact=artifact,
            model=model,
            color=color,
        ),
        timeout=generation_timeout,
        environment=environment,
    )
    metadata = _artifact_metadata(
        artifact,
        expected_process=process,
        expected_color=color,
    )
    _write_json_atomic(
        cache_path,
        {
            "kind": CACHE_KIND,
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "artifact_id": metadata["artifact_id"],
        },
    )
    return {
        **metadata,
        "reused": False,
        "generation": {
            "command_elapsed_seconds": elapsed,
            "peak_rss_gib": _watchdog_peak_gib(stderr),
            "watchdog": _watchdog_marker(stderr),
            "output": payload.get("output"),
            "schema_version": payload.get("schema_version"),
        },
        "cache_path": str(cache_path),
    }


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RegressionError(f"{label} must be positive and finite")
    return result


def _native_profile_sample(
    payload: Mapping[str, Any],
    *,
    minimum_samples: int,
) -> dict[str, Any]:
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise RegressionError("profile result has no environment metadata")
    source = environment.get("wall_time_source")
    if source != NATIVE_WALL_TIME_SOURCE:
        raise RegressionError(
            f"profile did not use the native Rusticol repeated wall timer: {source!r}"
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
    if payload.get("interrupted") is True or environment.get("interrupted") is True:
        raise RegressionError("profile result was interrupted")
    repetitions = payload.get("repetitions_per_sample")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise RegressionError("profile repetitions_per_sample is invalid")
    if repetitions <= 0:
        raise RegressionError("profile repetitions_per_sample must be positive")
    return {
        "wall_seconds_per_point": _finite_positive(
            payload.get("wall_time_per_point"),
            label="profile wall_time_per_point",
        ),
        "native_wall_time_source": source,
        "profile_timed_block_count": sample_count,
        "repetitions_per_timed_block": repetitions,
        "measured_evaluation_count": environment.get("measured_evaluation_count"),
        "native_elapsed_seconds": environment.get("elapsed_seconds"),
        "profile_uncertainty": payload.get("uncertainty"),
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
    environment = _environment()
    started = time.monotonic()

    interpreters = {
        "baseline": baseline_python,
        "current": current_python,
    }
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
        )
        for lane, python in interpreters.items()
    }
    partial: dict[str, Any] = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "artifacts": artifacts,
        "measurements": [],
    }
    result_path = output_root / "result.json"
    _write_json_atomic(result_path, partial)

    measurements: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {"baseline": [], "current": []}
    pair_orders: list[list[str]] = []
    for pair_index in range(arguments.samples):
        order = (
            ("baseline", "current") if pair_index % 2 == 0 else ("current", "baseline")
        )
        pair_orders.append(list(order))
        for order_index, lane in enumerate(order):
            artifact = artifacts[lane]
            payload, command_elapsed, stderr = _run_json(
                _profile_command(
                    interpreters[lane],
                    artifact=Path(str(artifact["path"])),
                    process_id=str(artifact["process_id"]),
                    batch_size=arguments.batch_size,
                    target_runtime=arguments.target_runtime,
                    minimum_samples=arguments.minimum_samples,
                    warmup_runs=arguments.warmup_runs,
                    helicities=arguments.helicity,
                    color_flows=arguments.color_flow,
                ),
                timeout=arguments.profile_timeout,
                environment=environment,
            )
            sample = _native_profile_sample(
                payload,
                minimum_samples=arguments.minimum_samples,
            )
            wall = float(sample["wall_seconds_per_point"])
            values[lane].append(wall)
            measurements.append(
                {
                    "pair_index": pair_index + 1,
                    "measurement_order": order_index + 1,
                    "lane": lane,
                    **sample,
                    "command_elapsed_seconds": command_elapsed,
                    "peak_rss_gib": _watchdog_peak_gib(stderr),
                    "watchdog": _watchdog_marker(stderr),
                }
            )
        partial["measurements"] = measurements
        partial["pair_orders"] = pair_orders
        _write_json_atomic(result_path, partial)

    distributions = {
        lane: _distribution(lane_values) for lane, lane_values in values.items()
    }
    gate = _regression_gate(distributions["baseline"], distributions["current"])
    elapsed = time.monotonic() - started
    result = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "passes": bool(gate["passes"]),
        "platform": platform.platform(),
        "configuration": {
            "baseline_python": str(baseline_python),
            "current_python": str(current_python),
            "output_root": str(output_root),
            "process": arguments.process,
            "model": model,
            "color_accuracy": arguments.color,
            "batch_size": arguments.batch_size,
            "independent_samples_per_lane": arguments.samples,
            "target_runtime_per_profile_seconds": arguments.target_runtime,
            "minimum_native_timed_blocks_per_profile": arguments.minimum_samples,
            "warmup_runs_per_profile": arguments.warmup_runs,
            "generation_timeout_seconds": arguments.generation_timeout,
            "profile_timeout_seconds": arguments.profile_timeout,
            "helicities": list(arguments.helicity),
            "color_flows": list(arguments.color_flow),
            "native_wall_time_source": NATIVE_WALL_TIME_SOURCE,
        },
        "artifacts": artifacts,
        "pair_orders": pair_orders,
        "measurements": measurements,
        "distributions": distributions,
        "gate": gate,
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
    result.add_argument("--process", required=True)
    result.add_argument("--model", default="built-in-sm")
    result.add_argument(
        "--color",
        "--color-accuracy",
        dest="color",
        choices=("lc", "nlc", "full"),
        default="lc",
    )
    result.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    result.add_argument(
        "--samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help="independent profile subprocesses per interpreter (minimum: 5)",
    )
    result.add_argument(
        "--target-runtime",
        type=_positive_float,
        default=DEFAULT_TARGET_RUNTIME,
        help="native timing target for each independent profile subprocess",
    )
    result.add_argument(
        "--minimum-samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help="minimum native timed blocks inside each profile subprocess",
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
        help="replace both cached compiled artifacts before profiling",
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
