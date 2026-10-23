# SPDX-License-Identifier: 0BSD
"""Directly measured phase evidence for the manual campaign dashboard."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from .models import ExecutionMode

PHASE_TIMELINE_SCHEMA = "pyamplicol-manual-campaign-phase-timeline-v1"


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _legacy_profile_outer_seconds(measurement: Mapping[str, object]) -> float | None:
    chunks = measurement.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None
    durations: list[float] = []
    for chunk in chunks:
        values = _mapping(chunk)
        elapsed = _seconds(values.get("elapsed_seconds"))
        if elapsed is None:
            return None
        durations.append(elapsed)
    return math.fsum(durations)


def _entry(
    key: str,
    label: str,
    seconds: float | None,
    *,
    status: str,
    source: str,
    detail: str | None = None,
    included_in: str | None = None,
    cpu_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
) -> dict[str, object]:
    return {
        "key": key,
        "label": label,
        "seconds": seconds,
        "status": status,
        "source": source,
        "detail": detail,
        "included_in": included_in,
        "cpu_seconds": cpu_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _progress_generation_children(path: Path | None) -> tuple[dict[str, object], ...]:
    """Read direct progress clocks for generation children once, if available."""

    if path is None or not path.is_file():
        return ()
    starts: dict[str, tuple[str, str | None]] = {}
    ended: dict[str, float] = {}
    order: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, Mapping):
                    continue
                task_id = event.get("task_id")
                if not isinstance(task_id, str):
                    continue
                if event.get("event") == "start":
                    description = event.get("description")
                    parent = event.get("parent_task_id")
                    starts[task_id] = (
                        description if isinstance(description, str) else task_id,
                        parent if isinstance(parent, str) else None,
                    )
                    order.append(task_id)
                elif event.get("event") == "end" and event.get("success") is not False:
                    elapsed = _seconds(event.get("elapsed_seconds"))
                    if elapsed is not None:
                        ended[task_id] = elapsed
    except OSError:
        return ()

    rows: list[dict[str, object]] = []
    for task_id in order:
        description, parent = starts[task_id]
        if parent != "generation" or task_id not in ended:
            continue
        rows.append(
            _entry(
                f"generation-step:{task_id.removeprefix('generation:')}",
                description,
                ended[task_id],
                status="measured",
                source="worker progress direct elapsed clock",
                detail="included in Generation",
                included_in="generation",
            )
        )
    return tuple(rows)


def build_phase_timeline(
    result: Mapping[str, object],
    *,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Build a compact timeline without deriving one clock from another."""

    provenance = _mapping(result.get("provenance"))
    manual = _mapping(provenance.get("manual_campaign"))
    identity = _mapping(manual.get("cell_identity"))
    runtime = _mapping(provenance.get("runtime_profile"))
    resources = _mapping(result.get("resources"))
    execution_mode = str(identity.get("execution_mode") or "")
    entries: list[dict[str, object]] = []

    preparation = _seconds(provenance.get("model_preparation_seconds"))
    preparation_reused = provenance.get("model_preparation_reused") is True
    if execution_mode == ExecutionMode.AMPLICOL.value:
        entries.append(
            _entry(
                "model-preparation",
                "Model preparation",
                None,
                status="not_applicable",
                source="legacy adapter",
                detail="not applicable to original AmpliCol",
            )
        )
    else:
        entries.append(
            _entry(
                "model-preparation",
                "Model preparation",
                preparation,
                status=(
                    "reused"
                    if preparation_reused
                    else ("measured" if preparation is not None else "unavailable")
                ),
                source="result provenance model_preparation_seconds",
                detail=(
                    "prepared model reused"
                    if preparation_reused
                    else (
                        None
                        if preparation is not None
                        else "not recorded by this result"
                    )
                ),
            )
        )

    generation = _seconds(result.get("generation_seconds"))
    entries.append(
        _entry(
            "generation",
            "Generation",
            generation,
            status="measured" if generation is not None else "unavailable",
            source="result generation_seconds",
        )
    )
    entries.extend(_progress_generation_children(progress_path))

    if execution_mode == ExecutionMode.AMPLICOL.value:
        warmup = _mapping(runtime.get("warmup"))
        warmup_seconds = _seconds(warmup.get("elapsed_seconds"))
        measurement = _mapping(runtime.get("measurement"))
        measurement_seconds = _legacy_profile_outer_seconds(measurement)
        entries.extend(
            (
                _entry(
                    "runtime-warmup",
                    "Runtime warm-up",
                    warmup_seconds,
                    status=(
                        "measured" if warmup_seconds is not None else "unavailable"
                    ),
                    source="legacy warmup subprocess wall clock",
                ),
                _entry(
                    "runtime-calibration",
                    "Runtime calibration",
                    None,
                    status="not_applicable",
                    source="legacy adaptive point calculation",
                    detail="no timed calibration phase",
                ),
                _entry(
                    "runtime-measurement",
                    "Timed profiling",
                    measurement_seconds,
                    status=(
                        "measured" if measurement_seconds is not None else "unavailable"
                    ),
                    source="sum of legacy profiling subprocess wall clocks",
                    detail=(
                        None
                        if _count(measurement.get("chunk_count")) is None
                        else f"{_count(measurement.get('chunk_count'))} chunks"
                    ),
                ),
            )
        )
    else:
        warmup_seconds = _seconds(runtime.get("warmup_elapsed_seconds"))
        calibration_seconds = _seconds(runtime.get("calibration_outer_elapsed_seconds"))
        if calibration_seconds is None and execution_mode in {
            ExecutionMode.COMPILED.value,
            ExecutionMode.EAGER.value,
        }:
            calibration_seconds = _seconds(runtime.get("calibration_elapsed_seconds"))
        measurement_seconds = _seconds(runtime.get("achieved_runtime_seconds"))
        if measurement_seconds is None:
            measurement_seconds = _seconds(
                runtime.get("measurement_phase_elapsed_seconds")
            )
        profile_total = _seconds(runtime.get("profile_total_elapsed_seconds"))
        attribution = _seconds(runtime.get("profile_attribution_elapsed_seconds"))
        if attribution is None:
            attribution = _seconds(runtime.get("evaluator_elapsed_seconds"))
        sample_count = _count(runtime.get("completed_sample_count"))
        entries.extend(
            (
                _entry(
                    "runtime-warmup",
                    "Runtime warm-up",
                    warmup_seconds,
                    status=(
                        "measured" if warmup_seconds is not None else "unavailable"
                    ),
                    source="benchmark warm-up wall clock",
                ),
                _entry(
                    "runtime-calibration",
                    "Runtime calibration",
                    calibration_seconds,
                    status=(
                        "measured" if calibration_seconds is not None else "unavailable"
                    ),
                    source="benchmark calibration wall clock",
                    detail=(
                        None
                        if _count(runtime.get("calibration_block_count")) is None
                        else (
                            f"{_count(runtime.get('calibration_block_count'))} blocks"
                        )
                    ),
                ),
                _entry(
                    "runtime-measurement",
                    "Timed headline measurement",
                    measurement_seconds,
                    status=(
                        "measured" if measurement_seconds is not None else "unavailable"
                    ),
                    source="accumulated benchmark headline outer wall clock",
                    detail=(
                        None if sample_count is None else f"{sample_count} samples"
                    ),
                ),
                _entry(
                    "runtime-attribution",
                    "Profiling attribution",
                    attribution,
                    status="measured" if attribution is not None else "unavailable",
                    source="native profile-call outer wall clock",
                    detail="separate from wall/evaluator-total/core metrics",
                ),
                _entry(
                    "runtime-profile-total",
                    "Complete runtime profile",
                    profile_total,
                    status=("measured" if profile_total is not None else "unavailable"),
                    source="profile operation outer wall clock",
                    detail="contains warm-up, calibration, measurement and attribution",
                ),
            )
        )

    validation_completed = result.get("validation") is not None
    entries.extend(
        (
            _entry(
                "post-profile-validation",
                "Post-profile validation",
                None,
                status="completed" if validation_completed else "unavailable",
                source="result validation status",
                detail=(
                    "completed; duration not separately timed"
                    if validation_completed
                    else "status and duration not separately recorded"
                ),
            ),
            _entry(
                "attempt-publication",
                "Result publication",
                None,
                status="completed",
                source="atomic attempt publication",
                detail="completed; duration not separately timed",
            ),
        )
    )

    worker_wall = _seconds(resources.get("wall_seconds"))
    peak_rss = _count(resources.get("peak_rss_bytes"))
    peak_guard = _count(resources.get("peak_guard_bytes"))
    cpu_seconds = _seconds(resources.get("cpu_seconds"))
    entries.append(
        _entry(
            "worker-supervision",
            "Worker supervision",
            worker_wall,
            status="observed" if worker_wall is not None else "unavailable",
            source="resource supervisor sample",
            detail="overall observed worker wall; phase rows may overlap",
            cpu_seconds=cpu_seconds,
            peak_memory_bytes=peak_guard or peak_rss,
        )
    )
    return {
        "schema": PHASE_TIMELINE_SCHEMA,
        "entries": entries,
        "total_worker_wall_seconds": worker_wall,
        "peak_rss_bytes": peak_rss,
        "peak_guard_bytes": peak_guard,
    }


__all__ = ["PHASE_TIMELINE_SCHEMA", "build_phase_timeline"]
