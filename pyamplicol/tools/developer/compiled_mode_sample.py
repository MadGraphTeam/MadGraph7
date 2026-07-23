#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Emit one version-independent compiled-runtime timing sample as JSON.

The helper is executed by the interpreter being measured.  It deliberately
does not use :class:`pyamplicol.benchmarking.BenchmarkBackend`, whose timing
policy can differ between the frozen and current installations.  Every
headline block calls the native ``_benchmark_f64_wall_time`` entry point, whose
timer starts after it has packed the batch. A separate ``profile_repeated`` call then
uses the same batch object, selector arguments, and repetition count solely
for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESULT_KIND = "pyamplicol-compiled-mode-native-sample"
SCHEMA_VERSION = 2
INSTALLATION_IDENTITY_KIND = "pyamplicol-installed-distribution-identity"
INSTALLATION_IDENTITY_SCHEMA_VERSION = 1
NATIVE_WALL_TIME_SOURCE = "runtime_core_repeated_wall_time"
NATIVE_WALL_TIME_SAMPLE_PASS = "runtime._benchmark_f64_wall_time"
PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime.profile_repeated"
PAIRED_TIMING_SAMPLE_CONTRACT = "paired_unprofiled_headline_profiled_attribution_v1"
MAX_CALIBRATION_BLOCKS = 4
MAX_REPETITIONS = 1_000_000_000
MAX_CORRECTNESS_POINTS = 8
_HASH_CHUNK_BYTES = 1024 * 1024


class SampleError(RuntimeError):
    """Raised when a trustworthy paired native sample cannot be produced."""


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


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SampleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SampleError(f"{label} must be positive and finite")
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SampleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SampleError(f"{label} must be finite and non-negative")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_distribution_identity() -> dict[str, object]:
    """Return a content identity for the installed generator/runtime wheel."""

    try:
        distribution = importlib.metadata.distribution("pyamplicol")
    except importlib.metadata.PackageNotFoundError as error:
        raise SampleError(
            "the selected interpreter has no pyamplicol installation"
        ) from error
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    native_modules: list[dict[str, object]] = []
    build_info_files: list[dict[str, object]] = []
    for entry in sorted(distribution.files or (), key=str):
        relative = str(entry)
        path = Path(str(distribution.locate_file(entry)))
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
        entry_bytes = relative.encode("utf-8")
        digest.update(len(entry_bytes).to_bytes(8, "big"))
        digest.update(entry_bytes)
        digest.update(size.to_bytes(8, "big"))
        with resolved.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        file_count += 1
        size_bytes += size
        name = resolved.name
        identity = {
            "relative_path": relative,
            **_path_identity(resolved),
        }
        if name.startswith("_rusticol.") and resolved.suffix in {
            ".so",
            ".pyd",
            ".dylib",
        }:
            native_modules.append(identity)
        if relative.endswith("pyamplicol/_build_info.json"):
            build_info_files.append(identity)
    if len(native_modules) != 1:
        raise SampleError(
            "the selected pyamplicol installation must contain exactly one "
            f"native Rusticol module, found {len(native_modules)}"
        )
    return {
        "kind": INSTALLATION_IDENTITY_KIND,
        "schema_version": INSTALLATION_IDENTITY_SCHEMA_VERSION,
        "package_version": distribution.version,
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "size_bytes": size_bytes,
        },
        "native_modules": native_modules,
        "build_info_files": build_info_files,
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleError("native profile contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(entry) for entry in value]
    raise SampleError(
        f"native profile contains a non-JSON value of type {type(value).__name__}"
    )


def _statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise SampleError("timing statistics require at least one sample")
    mean = float(statistics.fmean(values))
    deviation = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    error = deviation / math.sqrt(len(values))
    return {
        "standard_deviation": deviation,
        "standard_error": error,
        "relative_standard_error": error / mean if mean > 0.0 else 0.0,
    }


def _component(values: Sequence[float]) -> dict[str, object]:
    return {
        "mean_seconds_per_point": float(statistics.fmean(values)),
        "uncertainty": _statistics(values),
        "sample_count": len(values),
        "samples_seconds_per_point": list(values),
    }


def _benchmark_batch(points: Sequence[object], batch_size: int) -> tuple[object, ...]:
    source = tuple(points)
    if not source:
        raise SampleError("artifact has no deterministic validation point")
    return tuple(source[index % len(source)] for index in range(batch_size))


def _load_runtime(
    artifact: Path,
    *,
    process: str,
) -> tuple[object, dict[str, object]]:
    try:
        distribution = importlib.metadata.distribution("pyamplicol")
    except importlib.metadata.PackageNotFoundError as error:
        raise SampleError(
            "the selected interpreter has no pyamplicol installation"
        ) from error
    native_paths = [
        Path(str(distribution.locate_file(entry)))
        for entry in distribution.files or ()
        if Path(str(entry)).name.startswith("_rusticol.")
        and Path(str(entry)).suffix in {".so", ".pyd", ".dylib"}
    ]
    if len(native_paths) != 1:
        raise SampleError(
            "the selected pyamplicol installation must contain exactly one "
            f"native Rusticol module, found {len(native_paths)}"
        )
    native_path = native_paths[0].resolve(strict=True)
    specification = importlib.util.spec_from_file_location("_rusticol", native_path)
    if specification is None or specification.loader is None:
        raise SampleError(f"cannot load native Rusticol module: {native_path}")
    native_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(native_module)
    runtime_type = getattr(native_module, "Runtime", None)
    loader = getattr(runtime_type, "load", None)
    if not callable(loader):
        raise SampleError("native Rusticol module has no Runtime.load operation")
    runtime = loader(
        artifact,
        process=process,
        model_parameters=None,
        mute_warnings=False,
    )
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": _path_identity(Path(sys.executable)),
        "package_version": distribution.version,
        "native_module": _path_identity(native_path),
    }
    build_info_entries = [
        Path(str(distribution.locate_file(entry)))
        for entry in distribution.files or ()
        if str(entry).endswith("pyamplicol/_build_info.json")
    ]
    if len(build_info_entries) == 1:
        build_info_path = build_info_entries[0].resolve(strict=True)
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        if not isinstance(build_info, dict):
            raise SampleError("installed pyamplicol build provenance is not an object")
        result["build_info"] = {
            **_path_identity(build_info_path),
            "payload": build_info,
        }
        native_digest_operation = getattr(
            native_module,
            "native_build_inputs_sha256",
            None,
        )
        if callable(native_digest_operation):
            native_digest = native_digest_operation()
            result["native_build_inputs_sha256"] = native_digest
            expected_digest = build_info.get("native_build_inputs_sha256")
            if isinstance(expected_digest, str) and native_digest != expected_digest:
                raise SampleError(
                    "installed native Rusticol module does not match build provenance"
                )
    native_version_operation = getattr(native_module, "package_version", None)
    if callable(native_version_operation):
        result["native_package_version"] = native_version_operation()
    return runtime, result


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SampleError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SampleError(f"{label} must be a JSON object: {path}")
    return payload


def _artifact_process(
    manifest: Mapping[str, Any],
    *,
    process: str,
) -> Mapping[str, Any]:
    records = manifest.get("processes")
    if not isinstance(records, list):
        raise SampleError("artifact manifest has no process records")
    normalized = " ".join(process.split())
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and (
            record.get("id") == process
            or (
                isinstance(record.get("expression"), str)
                and " ".join(str(record["expression"]).split()) == normalized
            )
        )
    ]
    if len(matches) != 1:
        raise SampleError(
            f"artifact must contain exactly one process matching {process!r}"
        )
    return matches[0]


def _validation_momenta(
    artifact: Path,
    *,
    process: str,
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    manifest = _json_object(artifact / "artifact.json", label="artifact manifest")
    process_record = _artifact_process(manifest, process=process)
    process_id = process_record.get("id")
    if not isinstance(process_id, str) or not process_id:
        raise SampleError("artifact process has no stable ID")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise SampleError("artifact manifest has no payload inventory")
    candidates = [
        payload
        for payload in payloads
        if isinstance(payload, Mapping)
        and payload.get("role") == "validation-momenta"
        and payload.get("process_id") == process_id
    ]
    if len(candidates) != 1 or not isinstance(candidates[0].get("path"), str):
        raise SampleError(
            f"artifact process {process_id!r} has no unique validation momentum payload"
        )
    validation = _json_object(
        artifact / str(candidates[0]["path"]),
        label="validation momenta",
    )
    points = validation.get("points")
    if validation.get("available") is not True or not isinstance(points, list):
        raise SampleError("artifact has no deterministic validation momenta")
    expected_pdgs = process_record.get("external_pdgs")
    result: list[tuple[tuple[float, float, float, float], ...]] = []
    for point in points:
        if not isinstance(point, list):
            raise SampleError("validation momentum point is invalid")
        vectors: list[tuple[float, float, float, float]] = []
        pdgs: list[int] = []
        for particle in point:
            if not isinstance(particle, Mapping):
                raise SampleError("validation momentum particle is invalid")
            raw_momentum = particle.get("momentum")
            if not isinstance(raw_momentum, list) or len(raw_momentum) != 4:
                raise SampleError("validation four-momentum is invalid")
            try:
                vector = tuple(float(component) for component in raw_momentum)
                pdg = int(particle["pdg"])
            except (KeyError, TypeError, ValueError) as error:
                raise SampleError("validation momentum particle is invalid") from error
            if len(vector) != 4 or not all(math.isfinite(value) for value in vector):
                raise SampleError("validation four-momentum is non-finite")
            vectors.append((vector[0], vector[1], vector[2], vector[3]))
            pdgs.append(pdg)
        if isinstance(expected_pdgs, list) and pdgs != expected_pdgs:
            raise SampleError("validation momentum PDG order does not match artifact")
        result.append(tuple(vectors))
    if not result:
        raise SampleError("artifact validation momentum payload is empty")
    return tuple(result)


def _resolve_color_flows(
    artifact: Path,
    *,
    process: str,
    requested: Sequence[str],
) -> tuple[str, ...]:
    if not requested:
        return ()
    manifest = _json_object(artifact / "artifact.json", label="artifact manifest")
    process_record = _artifact_process(manifest, process=process)
    physics_path = process_record.get("physics_path")
    if not isinstance(physics_path, str) or not physics_path:
        raise SampleError("artifact process has no runtime physics payload")
    physics = _json_object(
        artifact / physics_path,
        label="runtime physics",
    )
    components = physics.get("color_components")
    if not isinstance(components, list):
        raise SampleError("runtime physics has no color component inventory")
    available = tuple(
        str(component["id"])
        for component in components
        if isinstance(component, Mapping) and isinstance(component.get("id"), str)
    )
    resolved: list[str] = []
    for value in requested:
        if value in available:
            resolved.append(value)
            continue
        try:
            ordinal = int(value, 10)
        except ValueError:
            resolved.append(value)
            continue
        if str(ordinal) != value.strip() or ordinal < 1 or ordinal > len(available):
            raise SampleError(
                f"color-flow ordinal {value!r} is out of range; "
                f"choose 1..{len(available)} or a stable color component ID"
            )
        resolved.append(available[ordinal - 1])
    return tuple(resolved)


def _execution_mode(runtime: object) -> str:
    metadata_operation = getattr(runtime, "metadata_json", None)
    if not callable(metadata_operation):
        raise SampleError("native runtime has no metadata operation")
    try:
        metadata_payload = json.loads(metadata_operation())
    except (TypeError, json.JSONDecodeError) as error:
        raise SampleError("native runtime metadata is invalid") from error
    if not isinstance(metadata_payload, Mapping):
        raise SampleError("native runtime metadata is not an object")
    mode = metadata_payload.get("execution_mode")
    if not isinstance(mode, str):
        raise SampleError("native runtime metadata has no execution mode")
    return mode


def _calibrate_repetitions(
    timer: Any,
    batch: tuple[object, ...],
    *,
    target_seconds: float,
    sample_count: int,
    selector_arguments: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    target_per_block = target_seconds / sample_count
    repetitions = 1
    observed = _finite_positive(
        timer(batch, repetitions, **selector_arguments),
        label="native wall calibration duration",
    )
    blocks = [(repetitions, observed)]
    for _ in range(MAX_CALIBRATION_BLOCKS):
        estimate = math.ceil(repetitions * target_per_block / observed)
        candidate = min(max(estimate, 1), MAX_REPETITIONS)
        if candidate == repetitions:
            break
        repetitions = candidate
        observed = _finite_positive(
            timer(batch, repetitions, **selector_arguments),
            label="native wall calibration duration",
        )
        blocks.append((repetitions, observed))
        ratio = observed / target_per_block
        if 0.75 <= ratio <= 1.5:
            break
    return repetitions, {
        "target_seconds_per_block": target_per_block,
        "block_count": len(blocks),
        "evaluation_count": sum(block_repetitions for block_repetitions, _ in blocks),
        "blocks": [
            {
                "repetitions": block_repetitions,
                "duration_seconds": duration,
            }
            for block_repetitions, duration in blocks
        ],
    }


def _profile_components(
    profile: Mapping[str, object],
    *,
    evaluated_points: int,
) -> tuple[float, float]:
    execution_mode = profile.get("execution_mode")
    if execution_mode not in (None, "compiled"):
        raise SampleError(
            f"native repeated profile used execution mode {execution_mode!r}"
        )
    wall = _finite_positive(
        profile.get("wall_time_s"),
        label="native repeated profile wall_time_s",
    )
    stage = _finite_nonnegative(
        profile.get("stage_evaluator_call_time_s"),
        label="native repeated profile stage_evaluator_call_time_s",
    )
    amplitude = _finite_nonnegative(
        profile.get("amplitude_evaluator_call_time_s"),
        label="native repeated profile amplitude_evaluator_call_time_s",
    )
    return wall / evaluated_points, (stage + amplitude) / evaluated_points


def _warmed_numerical_result(
    evaluator: Any,
    batch: tuple[object, ...],
    *,
    selector_arguments: Mapping[str, object],
) -> dict[str, object]:
    selector_values: dict[str, list[str]] = {}
    for key in ("helicities", "color_flows"):
        raw_selectors = selector_arguments.get(key)
        if raw_selectors is None:
            selector_values[key] = []
        elif (
            isinstance(raw_selectors, Sequence)
            and not isinstance(
                raw_selectors,
                (str, bytes, bytearray),
            )
            and all(isinstance(value, str) for value in raw_selectors)
        ):
            selector_values[key] = list(raw_selectors)
        else:
            raise SampleError(f"native warmed evaluation has invalid {key}")
    validation_batch = batch[:MAX_CORRECTNESS_POINTS]
    raw_values = evaluator(validation_batch, **selector_arguments)
    if not isinstance(raw_values, Sequence) or isinstance(
        raw_values,
        (str, bytes, bytearray),
    ):
        raise SampleError("native warmed evaluation did not return a value sequence")
    values: list[float] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (float, int)):
            raise SampleError("native warmed evaluation returned a non-numeric value")
        value = float(raw_value)
        if not math.isfinite(value):
            raise SampleError("native warmed evaluation returned a non-finite value")
        values.append(value)
    if len(values) != len(validation_batch):
        raise SampleError(
            "native warmed evaluation returned an unexpected number of values"
        )
    return {
        "point_count": len(validation_batch),
        "batch_sha256": _canonical_sha256(validation_batch),
        "values_f64": values,
        "values_f64_hex": [value.hex() for value in values],
        "helicities": selector_values["helicities"],
        "color_flows": selector_values["color_flows"],
    }


def sample(arguments: argparse.Namespace) -> dict[str, object]:
    artifact = arguments.artifact.expanduser().resolve(strict=True)
    runtime, runtime_identity = _load_runtime(
        artifact,
        process=arguments.process,
    )
    if _execution_mode(runtime) != "compiled":
        raise SampleError("regression sampling requires a compiled runtime")
    timer = getattr(runtime, "_benchmark_f64_wall_time", None)
    profiler = getattr(runtime, "profile_repeated", None)
    evaluate_once = getattr(runtime, "evaluate", None)
    if not callable(timer):
        raise SampleError("native unprofiled wall timer is unavailable")
    if not callable(profiler):
        raise SampleError("native repeated profiler is unavailable")
    if not callable(evaluate_once):
        raise SampleError("native f64 evaluator is unavailable")
    points = _validation_momenta(artifact, process=arguments.process)
    batch = _benchmark_batch(points, arguments.batch_size)
    helicities = tuple(arguments.helicity) or None
    resolved_color_flows = _resolve_color_flows(
        artifact,
        process=arguments.process,
        requested=arguments.color_flow,
    )
    color_flows = resolved_color_flows or None
    selector_arguments: dict[str, object] = {
        "helicities": helicities,
        "color_flows": color_flows,
        "precision": 16,
    }
    profile_arguments = {**selector_arguments, "include_values": False}

    for _ in range(arguments.warmup_runs):
        _finite_positive(
            timer(batch, 1, **selector_arguments),
            label="native wall warmup duration",
        )
        warmup_profile = profiler(batch, 1, **profile_arguments)
        if not isinstance(warmup_profile, Mapping):
            raise SampleError("native repeated warmup profile is not a mapping")

    repetitions, calibration = _calibrate_repetitions(
        timer,
        batch,
        target_seconds=arguments.target_runtime,
        sample_count=arguments.minimum_samples,
        selector_arguments=selector_arguments,
    )
    evaluated_points_per_block = repetitions * len(batch)
    headline_samples: list[float] = []
    profiled_wall_samples: list[float] = []
    evaluator_samples: list[float] = []
    raw_profiles: list[object] = []
    headline_durations: list[float] = []
    attribution_elapsed = 0.0
    for _ in range(arguments.minimum_samples):
        duration = _finite_positive(
            timer(batch, repetitions, **selector_arguments),
            label="native headline wall duration",
        )
        headline_durations.append(duration)
        headline_samples.append(duration / evaluated_points_per_block)

        profile_started = time.perf_counter()
        raw_profile = profiler(batch, repetitions, **profile_arguments)
        attribution_elapsed += time.perf_counter() - profile_started
        if not isinstance(raw_profile, Mapping):
            raise SampleError("native repeated profile is not a mapping")
        profile_wall, evaluator_time = _profile_components(
            raw_profile,
            evaluated_points=evaluated_points_per_block,
        )
        profiled_wall_samples.append(profile_wall)
        evaluator_samples.append(evaluator_time)
        raw_profiles.append(_json_value(raw_profile))

    numerical_result = _warmed_numerical_result(
        evaluate_once,
        batch,
        selector_arguments=selector_arguments,
    )
    sample_count = len(headline_samples)
    measured_evaluations = sample_count * repetitions
    measured_points = measured_evaluations * len(batch)
    headline_mean = float(statistics.fmean(headline_samples))
    evaluator_mean = float(statistics.fmean(evaluator_samples))
    timing_breakdown = {
        "sample_count": sample_count,
        "execution_mode": "compiled",
        "wall_time": _component(profiled_wall_samples),
        "evaluator_call_time": _component(evaluator_samples),
        "raw_profile_samples": raw_profiles,
    }
    return {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "wall_time_per_point": headline_mean,
        "wall_time_samples_seconds_per_point": headline_samples,
        "sample_count": sample_count,
        "repetitions_per_sample": repetitions,
        "interrupted": False,
        "uncertainty": _statistics(headline_samples),
        "evaluator_time_per_point": evaluator_mean,
        "evaluator_uncertainty": _statistics(evaluator_samples),
        "warmed_numerical_result": numerical_result,
        "timing_breakdown": timing_breakdown,
        "environment": {
            "wall_time_source": NATIVE_WALL_TIME_SOURCE,
            "wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
            "evaluator_time_sample_pass": PROFILE_ATTRIBUTION_SAMPLE_PASS,
            "timing_breakdown_sample_pass": PROFILE_ATTRIBUTION_SAMPLE_PASS,
            "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            "profile_attribution_paired_with_headline": True,
            "profile_attribution_identical_batch": True,
            "profile_attribution_identical_repetitions": True,
            "execution_mode": "compiled",
            "batch_size": len(batch),
            "precision": 16,
            "interrupted": False,
            "completed_sample_count": sample_count,
            "planned_sample_count": arguments.minimum_samples,
            "measured_evaluation_count": measured_evaluations,
            "measured_point_count": measured_points,
            "native_profile_sample_count": len(raw_profiles),
            "native_profile_sample_limit": arguments.minimum_samples,
            "native_profile_repetitions_per_sample": repetitions,
            "native_profile_points_per_sample": evaluated_points_per_block,
            "native_profile_calls_per_block": 1.0,
            "profile_attribution_evaluation_count": measured_evaluations,
            "profile_attribution_point_count": measured_points,
            "elapsed_seconds": sum(headline_durations),
            "headline_block_durations_seconds": headline_durations,
            "profile_attribution_elapsed_seconds": attribution_elapsed,
            "batch_sha256": _canonical_sha256(batch),
            "helicities": list(helicities or ()),
            "color_flows": list(color_flows or ()),
            "calibration": calibration,
            "runtime_identity": runtime_identity,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("--process", required=True)
    result.add_argument("--target-runtime", type=_positive_float, required=True)
    result.add_argument("--batch-size", type=_positive_int, required=True)
    result.add_argument("--minimum-samples", type=_at_least_five, required=True)
    result.add_argument("--warmup-runs", type=_positive_int, required=True)
    result.add_argument("--helicity", action="append", default=[])
    result.add_argument("--color-flow", action="append", default=[])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if tuple(sys.argv[1:] if argv is None else argv) == ("--installation-identity",):
        try:
            identity = _installed_distribution_identity()
        except (OSError, SampleError, ValueError) as error:
            print(f"compiled-mode-sample: {error}", file=sys.stderr)
            return 2
        print(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False))
        return 0
    arguments = parser().parse_args(argv)
    try:
        result = sample(arguments)
    except (OSError, SampleError, ValueError) as error:
        print(f"compiled-mode-sample: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
