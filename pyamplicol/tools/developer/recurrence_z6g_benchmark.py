#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Benchmark topology-replay recurrence against compiled JIT O2 for qq_Zng.

This is a developer harness, independent of ``docs/result_tables.py``. Run the
whole command behind ``tools/ci/memory_watchdog.py`` when generating the large
artifacts. Generation and profiling run in isolated worker processes so each
phase has a meaningful process-level ``resource.getrusage`` peak-RSS record.

Example::

    .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
      .venv/bin/python tools/developer/recurrence_z6g_benchmark.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREPARED_MODEL_ID = "built-in-sm-jit-o2"
DEFAULT_BATCH_SIZES = (128, 1024)
RESULT_KIND = "pyamplicol-recurrence-z6g-benchmark"
RESULT_SCHEMA = 1
_WORKER_MARKER = "PYAMPLICOL_RECURRENCE_Z6G_WORKER_RESULT="


class HarnessError(RuntimeError):
    """Raised when the benchmark contract cannot be completed."""


def _process(gluon_count: int) -> str:
    return "u u~ > Z" + " g" * gluon_count


def _process_name(gluon_count: int) -> str:
    return f"uubar_Z_{gluon_count}g"


def _selected_process(arguments: argparse.Namespace) -> str:
    return (
        _process(arguments.gluon_count)
        if arguments.process_expression is None
        else arguments.process_expression
    )


def _selected_process_name(arguments: argparse.Namespace) -> str:
    return (
        _process_name(arguments.gluon_count)
        if arguments.process_expression is None
        else "custom_process"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _rss_bytes(usage: resource.struct_rusage) -> int:
    """Normalize ru_maxrss (bytes on macOS, KiB on Linux) to bytes."""

    raw = float(usage.ru_maxrss)
    if platform.system() == "Darwin":
        return int(raw)
    return int(raw * 1024.0)


def _resource_peak() -> dict[str, object]:
    self_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF))
    children_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": self_bytes,
        "maximum_child_peak_bytes": children_bytes,
        "observed_lower_bound_bytes": max(self_bytes, children_bytes),
        "semantics": (
            "self high-water mark and maximum completed-child high-water mark; "
            "not an aggregate process-tree sample"
        ),
    }


def _artifact_stats(path: Path) -> dict[str, int]:
    file_count = 0
    size_bytes = 0
    for root, _directories, files in os.walk(path):
        directory = Path(root)
        for name in files:
            candidate = directory / name
            if candidate.is_file():
                file_count += 1
                size_bytes += candidate.stat().st_size
    return {"file_count": file_count, "size_bytes": size_bytes}


def _artifact_phases(path: Path) -> dict[str, float]:
    try:
        artifact = json.loads((path / "artifact.json").read_text(encoding="utf-8"))
        raw = artifact["extensions"]["generation"]["phase_timings_seconds"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise HarnessError(
            f"artifact has no valid generation timings: {path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise HarnessError(f"artifact generation timings are not an object: {path}")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or isinstance(value, bool):
            raise HarnessError(f"artifact has an invalid generation phase: {path}")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise HarnessError(f"artifact has an invalid phase duration: {path}")
        result[name] = seconds
    return result


def _effective_contract(path: Path) -> dict[str, object]:
    try:
        payload = tomllib.loads(
            (path / "config" / "effective.toml").read_text(encoding="utf-8")
        )
        color = payload["color"]
        evaluator = payload["evaluator"]
        jit = evaluator["jit"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise HarnessError(
            f"artifact has no valid effective configuration: {path}"
        ) from error
    return {
        "execution_mode": str(evaluator["execution_mode"]),
        "backend": str(evaluator["backend"]),
        "jit_optimization_level": int(jit["optimization_level"]),
        "color_accuracy": str(color["accuracy"]),
        "lc_flow_layout": str(color["lc_flow_layout"]),
    }


def _validation_points(
    artifact: Path, process_id: str
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    path = artifact / "processes" / process_id / "validation-momenta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload["points"]
        return tuple(
            tuple(
                tuple(float(component) for component in particle["momentum"])
                for particle in point
            )
            for point in points
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarnessError(f"invalid validation momenta at {path}") from error


def _complex_payload(value: object) -> list[float]:
    converted = complex(value)
    return [converted.real, converted.imag]


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(entry) for entry in value]
    return value


def _statistics_payload(value: Any) -> dict[str, float]:
    return {
        "standard_deviation_seconds_per_point": float(value.standard_deviation),
        "standard_error_seconds_per_point": float(value.standard_error),
        "relative_standard_error": float(value.relative_standard_error),
    }


def _benchmark_payload(result: Any) -> dict[str, object]:
    evaluator_uncertainty = result.evaluator_uncertainty
    return {
        "batch_size": int(result.effective_config.batch_size),
        "sample_count": int(result.sample_count),
        "repetitions_per_sample": int(result.repetitions_per_sample),
        "evaluation_count": int(result.evaluation_count),
        "evaluated_point_count": int(result.evaluated_point_count),
        "wall_seconds_per_point": float(result.wall_time_per_point),
        "evaluator_seconds_per_point": (
            None
            if result.evaluator_time_per_point is None
            else float(result.evaluator_time_per_point)
        ),
        "wall_uncertainty": _statistics_payload(result.uncertainty),
        "evaluator_uncertainty": (
            None
            if evaluator_uncertainty is None
            else _statistics_payload(evaluator_uncertainty)
        ),
        "timing_sources": {
            "wall": result.environment.get("wall_time_source"),
            "evaluator": result.environment.get("evaluator_time_source"),
        },
        "environment": _plain(result.environment),
        "interrupted": bool(result.interrupted),
    }


def _reference_color_order(gluon_count: int) -> tuple[int, ...]:
    return (2, *range(4, 4 + gluon_count), 1, 3)


def _generation_config(
    execution_mode: str,
    *,
    validation_samples: int,
    lc_flow_layout: str,
    point_tile_size: int,
    jit_optimization_level: int,
    gluon_count: int | None = None,
) -> Any:
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        ProcessConfig,
        RecurrenceEvaluatorConfig,
        RunConfig,
    )

    if execution_mode not in {"compiled", "eager", "recurrence"}:
        raise HarnessError(f"unsupported generation mode {execution_mode!r}")
    if lc_flow_layout not in {"topology-replay", "all-flow-union"}:
        raise HarnessError(f"unsupported LC flow layout {lc_flow_layout!r}")
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout=lc_flow_layout),
        process=ProcessConfig(
            reference_color_order=(
                () if gluon_count is None else _reference_color_order(gluon_count)
            ),
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=True,
                samples=validation_samples,
                seed=12345,
                relative_tolerance=1.0e-12,
                absolute_tolerance=1.0e-300,
                post_build_validation=True,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode=execution_mode,
            batch_size=64,
            output_chunk_size=512,
            optimization=EvaluatorOptimizationConfig(
                horner_iterations=10,
                cores=1,
                max_horner_variables=1000,
                max_common_pair_cache_entries=5_000_000,
                max_common_pair_distance=1000,
                collect_factors="auto",
            ),
            # Recurrence process generation consumes its prepared pack without
            # compiling. Compiled mode performs process-specific JIT work in
            # the measured generation interval.
            jit=JITConfig(optimization_level=jit_optimization_level),
            recurrence=RecurrenceEvaluatorConfig(
                point_tile_size=point_tile_size,
            ),
        ),
    )


def _generate_worker(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import Generator, ModelSource
    from pyamplicol.assets.prepared_models import (
        BUILTIN_SM_JIT_O2,
        packaged_prepared_model_path,
    )
    from pyamplicol.reporting import StreamProgressSink

    artifact = arguments.artifact.resolve()
    prepared_access_started = time.perf_counter()
    prepared_context = (
        contextlib.nullcontext(None)
        if arguments.mode == "compiled"
        else (
            packaged_prepared_model_path(BUILTIN_SM_JIT_O2)
            if arguments.prepared_model is None
            else contextlib.nullcontext(arguments.prepared_model.resolve())
        )
    )
    try:
        with prepared_context as prepared_model:
            if prepared_model is None:
                model_source = ModelSource.built_in_sm()
                model_record: dict[str, object] = {
                    "kind": "built-in-sm",
                    "prepared_model": None,
                    "compile_excluded_from_generation": False,
                }
            else:
                if not prepared_model.is_file():
                    raise HarnessError(
                        f"prepared model does not exist: {prepared_model}"
                    )
                model_source = ModelSource.from_path(prepared_model)
                model_record = {
                    "kind": "prepared-model",
                    "prepared_model": (
                        PREPARED_MODEL_ID
                        if arguments.prepared_model is None
                        else str(prepared_model)
                    ),
                    "bundle_size_bytes": prepared_model.stat().st_size,
                    "compile_excluded_from_generation": True,
                }
            prepared_access_seconds = time.perf_counter() - prepared_access_started
            generation_started = time.perf_counter()
            Generator(
                _generation_config(
                    arguments.mode,
                    validation_samples=arguments.validation_samples,
                    lc_flow_layout=arguments.lc_flow_layout,
                    point_tile_size=arguments.point_tile_size,
                    jit_optimization_level=arguments.jit_optimization_level,
                    gluon_count=(
                        arguments.gluon_count
                        if arguments.process_expression is None
                        else None
                    ),
                ),
                progress=StreamProgressSink(sys.stderr),
            ).generate(
                _selected_process(arguments),
                artifact,
                model=model_source,
                mode=arguments.write_mode,
            )
    except Exception as error:
        if arguments.mode == "recurrence":
            raise HarnessError(
                "recurrence generation is unavailable or failed; install a native "
                "build with recurrence support and a current built-in prepared "
                f"model pack: {error}"
            ) from error
        raise
    return {
        "mode": arguments.mode,
        "generation_wall_seconds": time.perf_counter() - generation_started,
        "generation_reused": False,
        "peak_rss": _resource_peak(),
        "model_source": {
            **model_record,
            "access_seconds": prepared_access_seconds,
        },
    }


def _profile_worker(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import BenchmarkRunner, Runtime
    from pyamplicol.config import BenchmarkConfig

    artifact = arguments.artifact.resolve()
    load_started = time.perf_counter()
    try:
        process = _selected_process(arguments)
        runtime = Runtime.load(artifact, process=process)
    except Exception as error:
        if arguments.mode == "recurrence":
            raise HarnessError(
                "the installed runtime cannot load the recurrence artifact; "
                f"rebuild pyAmpliCol with recurrence support: {error}"
            ) from error
        raise
    cold_load_seconds = time.perf_counter() - load_started
    peak_after_load = _resource_peak()
    physics = runtime.physics
    if physics.process.casefold() != process.casefold():
        raise HarnessError(
            f"artifact resolved process {physics.process!r}, expected {process!r}"
        )
    if physics.color_accuracy != "lc" or not physics.color_flows:
        raise HarnessError("benchmark artifact does not expose physical LC flows")

    def resolve_axis(
        requested: str,
        available_ids: tuple[str, ...],
        *,
        label: str,
    ) -> str:
        try:
            ordinal = int(requested, 10)
        except ValueError:
            ordinal = 0
        if ordinal:
            if ordinal < 1 or ordinal > len(available_ids):
                raise HarnessError(
                    f"{label} ordinal is outside 1..{len(available_ids)}"
                )
            return available_ids[ordinal - 1]
        if requested not in set(available_ids):
            raise HarnessError(
                f"unknown {label} {requested!r}; artifact exposes "
                f"{len(available_ids)} values"
            )
        return requested

    color_flow_id = resolve_axis(
        arguments.color_flow,
        tuple(flow.id for flow in physics.color_flows),
        label="color flow",
    )
    helicity_id = resolve_axis(
        arguments.helicity,
        tuple(helicity.id for helicity in physics.helicities),
        label="helicity",
    )
    union_workload = arguments.lc_flow_layout == "all-flow-union"
    selectors = (
        {"helicities": (helicity_id,)}
        if union_workload
        else {"color_flows": (color_flow_id,)}
    )

    validation_point_artifact = arguments.validation_point_artifact.resolve()
    validation_points = _validation_points(
        validation_point_artifact,
        physics.process_id,
    )
    if not validation_points:
        raise HarnessError("artifact contains no deterministic validation point")
    selected_points = validation_points[:1]
    selected_total = runtime.evaluate(selected_points, **selectors)[0]
    selected_resolved = runtime.evaluate_resolved(selected_points, **selectors)
    resolved_total = selected_resolved.total()[0]
    absolute = abs(complex(selected_total) - complex(resolved_total))
    relative = absolute / max(
        abs(complex(selected_total)), abs(complex(resolved_total)), 1.0e-300
    )

    profiles: list[dict[str, object]] = []
    for batch_size in arguments.batch_size:
        try:
            result = BenchmarkRunner(
                BenchmarkConfig(
                    target_runtime=arguments.target_runtime,
                    batch_size=batch_size,
                    precision=16,
                    warmup_runs=arguments.warmup_runs,
                    minimum_samples=arguments.minimum_samples,
                    color_flow_ids=(() if union_workload else (color_flow_id,)),
                    helicity_ids=((helicity_id,) if union_workload else ()),
                )
            ).run(runtime)
        except Exception as error:
            if arguments.mode == "recurrence":
                raise HarnessError(
                    "the installed runtime cannot profile recurrence execution at "
                    f"batch {batch_size}: {error}"
                ) from error
            raise
        profiles.append(_benchmark_payload(result))

    return {
        "mode": arguments.mode,
        "cold_load_seconds": cold_load_seconds,
        "peak_rss_after_cold_load": peak_after_load,
        "peak_rss_after_profile": _resource_peak(),
        "process_id": physics.process_id,
        "process_expression": physics.process,
        "selector_contract": {
            "color_flow_request": arguments.color_flow,
            "resolved_color_flow_id": (None if union_workload else color_flow_id),
            "helicity_request": arguments.helicity,
            "resolved_helicity_id": helicity_id if union_workload else None,
            "color_flow_count": len(physics.color_flows),
            "helicity_count": len(physics.helicities),
            "structural_zero_helicity_count": (physics.structural_zero_helicity_count),
            "workload": (
                "all-flows/runtime-selected-single-helicity"
                if union_workload
                else "single-runtime-selected-flow/helicity-sum"
            ),
        },
        "validation": {
            "point_source_artifact": str(validation_point_artifact),
            "selected_total": _complex_payload(selected_total),
            "resolved_sum": _complex_payload(resolved_total),
            "absolute_difference": absolute,
            "relative_difference": relative,
            "passes": absolute <= 1.0e-15 or relative <= 1.0e-12,
        },
        "profiles": profiles,
    }


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=("generate", "profile"))
    parser.add_argument(
        "--mode",
        choices=("compiled", "eager", "recurrence"),
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--validation-point-artifact", type=Path)
    parser.add_argument("--write-mode", choices=("error", "replace"), default="error")
    parser.add_argument("--batch-size", type=_positive_int, action="append", default=[])
    parser.add_argument("--target-runtime", type=_positive_float, default=5.0)
    parser.add_argument("--minimum-samples", type=_positive_int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--color-flow", default="1")
    parser.add_argument("--helicity", default="1")
    parser.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default="topology-replay",
    )
    parser.add_argument("--gluon-count", type=_positive_int, default=6)
    parser.add_argument("--process-expression")
    parser.add_argument("--validation-samples", type=_positive_int, default=10)
    parser.add_argument("--point-tile-size", type=_positive_int, default=1024)
    parser.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
    )
    parser.add_argument("--prepared-model", type=Path)
    return parser


def _worker_main(argv: Sequence[str]) -> int:
    arguments = _worker_parser().parse_args(argv)
    if arguments.warmup_runs < 0:
        raise HarnessError("warmup runs must be non-negative")
    if arguments.operation == "profile" and arguments.validation_point_artifact is None:
        raise HarnessError(
            "profile workers require an explicit --validation-point-artifact"
        )
    operation = (
        _generate_worker if arguments.operation == "generate" else _profile_worker
    )
    payload = operation(arguments)
    print(_WORKER_MARKER + json.dumps(payload, sort_keys=True))
    return 0


def _run_worker(
    arguments: Sequence[str],
    *,
    mode: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = (sys.executable, str(Path(__file__).resolve()), "_worker", *arguments)
    environment = os.environ.copy()
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    environment.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise HarnessError(
            f"{mode} {phase} worker exceeded {timeout_seconds:g} seconds"
        ) from error
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        stderr_text = completed.stderr or ""
        detail = (
            "\n".join(
                section
                for section in (
                    completed.stdout.strip()[-4000:],
                    stderr_text.strip()[-4000:],
                )
                if section
            )
            or "no worker output"
        )
        raise HarnessError(
            f"{mode} {phase} worker failed with exit {completed.returncode}: {detail}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_MARKER):
            payload = json.loads(line.removeprefix(_WORKER_MARKER))
            if isinstance(payload, dict):
                return payload
            break
    raise HarnessError(f"{mode} {phase} worker did not emit a JSON result")


def _comparison(left: object, right: object) -> dict[str, object]:
    left_value = complex(*left) if isinstance(left, list) else complex(left)
    right_value = complex(*right) if isinstance(right, list) else complex(right)
    absolute = abs(left_value - right_value)
    relative = absolute / max(abs(left_value), abs(right_value), 1.0e-300)
    return {
        "recurrence": _complex_payload(left_value),
        "compiled": _complex_payload(right_value),
        "absolute_difference": absolute,
        "relative_difference": relative,
        "passes": absolute <= 1.0e-15 or relative <= 1.0e-12,
    }


def _git_head() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_json = (
        arguments.result_json.resolve()
        if arguments.result_json is not None
        else output_root / "result.json"
    )
    layout_suffix = arguments.lc_flow_layout
    all_artifacts = {
        "recurrence": output_root / f"recurrence-{layout_suffix}",
        "compiled": (
            output_root
            / f"compiled-jit-o{arguments.jit_optimization_level}-{layout_suffix}"
        ),
        "eager": (
            output_root
            / f"eager-jit-o{arguments.jit_optimization_level}-{layout_suffix}"
        ),
    }
    artifacts = (
        {arguments.only_mode: all_artifacts[arguments.only_mode]}
        if arguments.only_mode is not None
        else all_artifacts
    )
    generation: dict[str, dict[str, object]] = {}
    for mode, artifact in artifacts.items():
        if artifact.exists() and not arguments.force:
            if not artifact.is_dir():
                raise HarnessError(f"artifact path is not a directory: {artifact}")
            record: dict[str, object] = {
                "mode": mode,
                "generation_wall_seconds": None,
                "generation_reused": True,
                "peak_rss": None,
            }
        else:
            if arguments.reuse_only:
                raise HarnessError(f"required artifact does not exist: {artifact}")
            print(f"Generating {mode} artifact at {artifact}", file=sys.stderr)
            record = _run_worker(
                (
                    "generate",
                    "--mode",
                    mode,
                    "--artifact",
                    str(artifact),
                    "--write-mode",
                    "replace" if arguments.force else "error",
                    "--gluon-count",
                    str(arguments.gluon_count),
                    *(
                        ()
                        if arguments.process_expression is None
                        else ("--process-expression", arguments.process_expression)
                    ),
                    "--validation-samples",
                    str(arguments.validation_samples),
                    "--point-tile-size",
                    str(arguments.point_tile_size),
                    "--jit-optimization-level",
                    str(arguments.jit_optimization_level),
                    "--lc-flow-layout",
                    arguments.lc_flow_layout,
                    *(
                        ()
                        if arguments.prepared_model is None
                        else ("--prepared-model", str(arguments.prepared_model))
                    ),
                ),
                mode=mode,
                phase="generation",
                timeout_seconds=arguments.generation_timeout,
            )
        phases = _artifact_phases(artifact)
        record.update(
            {
                "artifact": str(artifact),
                "artifact_stats": _artifact_stats(artifact),
                "phase_timings_seconds": phases,
                "phase_total_seconds": sum(phases.values()),
                "effective_contract": _effective_contract(artifact),
            }
        )
        generation[mode] = record

    profiles: dict[str, dict[str, Any]] = {}
    validation_point_artifact = artifacts.get(
        "recurrence",
        next(iter(artifacts.values())),
    )
    worker_profile_arguments: list[str] = [
        "profile",
        "--validation-point-artifact",
        str(validation_point_artifact),
        "--target-runtime",
        str(arguments.target_runtime),
        "--minimum-samples",
        str(arguments.minimum_samples),
        "--warmup-runs",
        str(arguments.warmup_runs),
        "--color-flow",
        arguments.color_flow,
        "--helicity",
        arguments.helicity,
        "--lc-flow-layout",
        arguments.lc_flow_layout,
        "--gluon-count",
        str(arguments.gluon_count),
        *(
            ()
            if arguments.process_expression is None
            else ("--process-expression", arguments.process_expression)
        ),
        "--validation-samples",
        str(arguments.validation_samples),
    ]
    for batch_size in arguments.batch_size:
        worker_profile_arguments.extend(("--batch-size", str(batch_size)))
    if not arguments.generation_only:
        for mode, artifact in artifacts.items():
            print(f"Loading and profiling {mode} artifact", file=sys.stderr)
            profiles[mode] = _run_worker(
                (
                    *worker_profile_arguments,
                    "--mode",
                    mode,
                    "--artifact",
                    str(artifact),
                ),
                mode=mode,
                phase="profile",
                timeout_seconds=arguments.profile_timeout,
            )

    comparison: dict[str, object] | None = None
    selectors_match: bool | None = None
    passes: bool | None = None
    if {"recurrence", "compiled"}.issubset(profiles):
        recurrence_validation = profiles["recurrence"]["validation"]
        compiled_validation = profiles["compiled"]["validation"]
        if not isinstance(recurrence_validation, Mapping) or not isinstance(
            compiled_validation, Mapping
        ):
            raise HarnessError("profile workers returned invalid validation metadata")
        comparison = _comparison(
            recurrence_validation["selected_total"],
            compiled_validation["selected_total"],
        )
        selectors_match = (
            profiles["recurrence"]["selector_contract"]
            == profiles["compiled"]["selector_contract"]
        )
        passes = (
            bool(recurrence_validation["passes"])
            and bool(compiled_validation["passes"])
            and bool(comparison["passes"])
            and selectors_match
        )
    payload: dict[str, object] = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA,
        "complete": not arguments.generation_only,
        "passes": passes,
        "source_revision": _git_head(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "process": _selected_process(arguments),
        "process_name": _selected_process_name(arguments),
        "workload": (
            "all-flows/runtime-selected-single-helicity"
            if arguments.lc_flow_layout == "all-flow-union"
            else "single-runtime-selected-flow/helicity-sum"
        ),
        "configuration": {
            "batch_sizes": list(arguments.batch_size),
            "target_runtime_seconds": arguments.target_runtime,
            "minimum_samples": arguments.minimum_samples,
            "warmup_runs": arguments.warmup_runs,
            "generation_timeout_seconds": arguments.generation_timeout,
            "profile_timeout_seconds": arguments.profile_timeout,
            "color_flow_request": arguments.color_flow,
            "helicity_request": arguments.helicity,
            "lc_flow_layout": arguments.lc_flow_layout,
            "gluon_count": arguments.gluon_count,
            "validation_samples": arguments.validation_samples,
            "point_tile_size": arguments.point_tile_size,
            "jit_optimization_level": arguments.jit_optimization_level,
            "validation_point_artifact": str(validation_point_artifact),
            "generation_only": arguments.generation_only,
            "only_mode": arguments.only_mode,
            "prepared_model": PREPARED_MODEL_ID,
            "prepared_model_path": (
                None
                if arguments.prepared_model is None
                else str(arguments.prepared_model.resolve())
            ),
            "external_watchdog_required_for_long_runs": True,
        },
        "generation": generation,
        "profiles": profiles,
        "selector_contracts_match": selectors_match,
        "selected_flow_validation": comparison,
    }
    _write_json_atomic(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".artifacts" / "developer" / "recurrence-z6g",
        help="artifact/result directory",
    )
    result.add_argument("--result-json", type=Path)
    result.add_argument("--gluon-count", type=_positive_int, default=6)
    result.add_argument(
        "--process-expression",
        help="explicit diagnostic process expression instead of the qq_Zng family",
    )
    result.add_argument("--validation-samples", type=_positive_int, default=10)
    result.add_argument(
        "--point-tile-size",
        type=_positive_int,
        default=1024,
        help="recurrence workspace point stride used for cache-tiling experiments",
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="JIT optimization level used for generated artifacts (default: 2)",
    )
    result.add_argument(
        "--prepared-model",
        type=Path,
        help="explicit prepared-model bundle used by both execution modes",
    )
    result.add_argument(
        "--generation-only",
        action="store_true",
        help="generate and report artifact statistics without runtime profiling",
    )
    result.add_argument(
        "--only-mode",
        choices=("compiled", "eager", "recurrence"),
        help="restrict generation/profiling to one execution mode",
    )
    result.add_argument(
        "--batch-size",
        type=_positive_int,
        action="append",
        default=None,
        help="native profiling batch size; repeat (default: 128 and 1024)",
    )
    result.add_argument("--target-runtime", type=_positive_float, default=5.0)
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=900.0,
        help="maximum seconds for either generation worker (default: 900)",
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=300.0,
        help="maximum seconds for either profiling worker (default: 300)",
    )
    result.add_argument("--minimum-samples", type=_positive_int, default=5)
    result.add_argument("--warmup-runs", type=int, default=2)
    result.add_argument(
        "--color-flow",
        default="1",
        help="one-based flow ordinal or stable flow ID (default: 1)",
    )
    result.add_argument(
        "--helicity",
        default="1",
        help="one-based helicity ordinal or stable helicity ID (default: 1)",
    )
    result.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default="topology-replay",
        help="recurrence/compiled LC layout and matching benchmark workload",
    )
    result.add_argument(
        "--force",
        action="store_true",
        help="replace both artifacts before profiling",
    )
    result.add_argument(
        "--reuse-only",
        action="store_true",
        help="fail rather than generate when either artifact is missing",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if values[:1] == ["_worker"]:
            return _worker_main(values[1:])
        arguments = parser().parse_args(values)
        if arguments.force and arguments.reuse_only:
            raise HarnessError("--force and --reuse-only are mutually exclusive")
        if arguments.warmup_runs < 0:
            raise HarnessError("warmup runs must be non-negative")
        if arguments.batch_size is None:
            arguments.batch_size = list(DEFAULT_BATCH_SIZES)
        elif len(set(arguments.batch_size)) != len(arguments.batch_size):
            raise HarnessError("batch sizes must be unique")
        result = run(arguments)
    except (
        HarnessError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"recurrence-z6g-benchmark: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["passes"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
