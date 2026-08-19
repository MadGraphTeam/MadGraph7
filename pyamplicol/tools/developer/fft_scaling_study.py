#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Resumable FullColor scaling study for direct and FFT contractions.

The study intentionally measures one fixed, nonzero helicity per process.  OTF
generation retains its complete helicity coverage, but its runtime query is the
same fixed helicity used by every other curve.  Every external command that can
grow with multiplicity is run through the repository memory watchdog.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import (  # noqa: E402
    fft_gluon_performance_acceptance as performance,
)
from tools.developer import legacy_amplicol  # noqa: E402

KIND = "pyamplicol-fullcolor-fft-scaling-study"
SCHEMA_VERSION = 1
STUDY_ROOT = ROOT / ".artifacts" / "fft-scaling-study"
FINAL_MULTIPLICITIES = tuple(range(2, 8))
POINT_COUNT = performance.POINT_COUNT
WARM_SAMPLES = performance.WARM_SAMPLE_COUNT
MEMORY_LIMIT_GIB = 30.0
MAX_GENERATION_SECONDS = 30.0 * 60.0
DEFAULT_ALPHA_S = 0.118
NUMERICAL_RELATIVE_TOLERANCE = 1.0e-10
AMPLI_COL_MAX_POINTS = 1_000_000_000
FORTRAN_FLOAT = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?"
AMPLI_COL_TIMING_ROW = re.compile(
    rf"^\s*(generation setup|total)\s+({FORTRAN_FLOAT})\s+",
    re.MULTILINE,
)
RSS_MARKER = performance.DARWIN_RSS_MARKER


class StudyError(RuntimeError):
    """The bounded scaling-study contract could not be satisfied."""


@dataclass(frozen=True, slots=True)
class Mode:
    key: str
    label: str
    kind: str
    execution_mode: str | None = None
    contraction: str | None = None


MODES = (
    Mode("reference-fft", "Reference FFT", "reference"),
    Mode("amplicol", "AmpliCol", "amplicol"),
    Mode(
        "recurrence-direct",
        "pyAmpliCol - recurrence",
        "candidate",
        "recurrence",
        "direct",
    ),
    Mode(
        "recurrence-fft",
        "pyAmpliCol - recurrence - FFT",
        "candidate",
        "recurrence",
        "symmetric-group-fft",
    ),
    Mode("otf-direct", "pyAmpliCol - OTF", "candidate", "on-the-fly", "direct"),
    Mode(
        "otf-fft",
        "pyAmpliCol - OTF - FFT",
        "candidate",
        "on-the-fly",
        "symmetric-group-fft",
    ),
    Mode("compiled-direct", "pyAmpliCol - Compiled", "candidate", "compiled", "direct"),
    Mode(
        "compiled-fft",
        "pyAmpliCol - Compiled - FFT",
        "candidate",
        "compiled",
        "symmetric-group-fft",
    ),
)
MODE_BY_KEY = {mode.key: mode for mode in MODES}
FAMILIES = ("gg", "ddbar")


def process_expression(family: str, final_multiplicity: int) -> str:
    if final_multiplicity not in FINAL_MULTIPLICITIES:
        raise StudyError(f"unsupported final-state multiplicity {final_multiplicity}")
    extra = " ".join("g" for _ in range(final_multiplicity - 2))
    if family == "gg":
        return "g g > g g" + (f" {extra}" if extra else "")
    if family == "ddbar":
        return "d d~ > d d~" + (f" {extra}" if extra else "")
    raise StudyError(f"unknown process family {family!r}")


def process_key(family: str, final_multiplicity: int) -> str:
    return f"{family}_n{final_multiplicity}"


def fixed_ddbar_helicity(final_multiplicity: int) -> tuple[int, ...]:
    return tuple(-1 if index % 2 == 0 else 1 for index in range(final_multiplicity + 2))


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _parse_candidate_rows(output: str) -> dict[str, object]:
    """Parse the common V4 probe wire format, including compiled mode."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[0] != "FFT_CANDIDATE_PROBE_V4":
        raise StudyError("candidate probe header is missing or invalid")
    scalar: dict[str, str] = {}
    points: dict[int, float] = {}
    warm: dict[int, float] = {}
    for line in lines[1:]:
        fields = line.split()
        if fields[0] == "POINT_VALUE":
            if len(fields) != 3 or int(fields[1]) in points:
                raise StudyError("candidate point-value rows are malformed")
            points[int(fields[1])] = float(fields[2])
        elif fields[0] == "WARM_CELL_SECONDS":
            if len(fields) != 4 or int(fields[1]) in warm or int(fields[2]) != 1:
                raise StudyError("candidate warm-cell rows are malformed")
            warm[int(fields[1])] = float(fields[3])
        elif fields[0] == "CALIBRATION_CELL":
            continue
        else:
            if len(fields) < 2 or fields[0] in scalar:
                raise StudyError("candidate scalar rows are malformed")
            scalar[fields[0]] = " ".join(fields[1:])
    required = {
        "PROCESS",
        "EXECUTION_MODE",
        "TIMER_SOURCE",
        "HELICITY_COVERAGE_COUNT",
        "SELECTED_HELICITY_ID",
        "POINT_COUNT",
        "LOAD_SECONDS",
        "FIRST_WARM_SECONDS",
        "WARM_UP_API_SECONDS",
        "MIN_ABSOLUTE_VALUE",
        "MAX_RSS_KIB",
    }
    if set(scalar) != required:
        raise StudyError("candidate probe scalar fields differ from the V4 contract")
    if set(points) != set(range(1, POINT_COUNT + 1)):
        raise StudyError("candidate probe did not report all ten points")
    if set(warm) != set(range(1, WARM_SAMPLES + 1)):
        raise StudyError("candidate probe did not report ten warm samples")
    warm_values = tuple(warm[index] for index in range(1, WARM_SAMPLES + 1))
    point_values = tuple(points[index] for index in range(1, POINT_COUNT + 1))
    max_rss_kib = int(scalar["MAX_RSS_KIB"])
    if (
        scalar["EXECUTION_MODE"] not in {"recurrence", "on-the-fly", "compiled"}
        or scalar["TIMER_SOURCE"] != "process-cpu-time"
        or int(scalar["POINT_COUNT"]) != POINT_COUNT
        or int(scalar["HELICITY_COVERAGE_COUNT"]) < 1
        or not all(_positive_finite(value) for value in warm_values)
        or not all(math.isfinite(value) for value in point_values)
        or max_rss_kib < 1
    ):
        raise StudyError("candidate probe contains invalid measurement values")
    return {
        "process": scalar["PROCESS"],
        "execution_mode": scalar["EXECUTION_MODE"],
        "helicity_coverage_count": int(scalar["HELICITY_COVERAGE_COUNT"]),
        "selected_helicity_id": scalar["SELECTED_HELICITY_ID"],
        "point_values": list(point_values),
        "load_seconds": float(scalar["LOAD_SECONDS"]),
        "first_warm_seconds": float(scalar["FIRST_WARM_SECONDS"]),
        "warm_up_api_seconds": float(scalar["WARM_UP_API_SECONDS"]),
        "warm_samples_seconds": list(warm_values),
        "warm_median_seconds": statistics.median(warm_values),
        "max_rss_kib": max_rss_kib,
    }


def parse_candidate_load_only_output(output: str) -> dict[str, object]:
    """Parse the small loader A/B wire format emitted by ``--load-only``."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[0] != "FFT_CANDIDATE_LOAD_ONLY_V1":
        raise StudyError("candidate load-only header is missing or invalid")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(" ")
        if not separator or key in fields:
            raise StudyError("candidate load-only rows are malformed")
        fields[key] = value.strip()
    required = {
        "PROCESS",
        "EXECUTION_MODE",
        "TIMER_SOURCE",
        "ALPHA_S",
        "LOAD_SECONDS",
        "MAX_RSS_KIB",
    }
    if set(fields) != required:
        raise StudyError("candidate load-only fields differ from the V1 contract")
    alpha_s = float(fields["ALPHA_S"])
    load_seconds = float(fields["LOAD_SECONDS"])
    max_rss_kib = int(fields["MAX_RSS_KIB"])
    if (
        fields["EXECUTION_MODE"] not in {"recurrence", "on-the-fly", "compiled"}
        or fields["TIMER_SOURCE"] != "process-cpu-time"
        or not _positive_finite(alpha_s)
        or not _positive_finite(load_seconds)
        or max_rss_kib < 1
    ):
        raise StudyError("candidate load-only values are invalid")
    return {
        "process": fields["PROCESS"],
        "execution_mode": fields["EXECUTION_MODE"],
        "alpha_s": alpha_s,
        "load_seconds": load_seconds,
        "max_rss_kib": max_rss_kib,
    }


def _write_event(
    path: Path,
    momenta: Sequence[Sequence[float]],
    helicity: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "PYAMPLICOL_SCALING_EVENT_V1",
        "BEGIN_MOMENTA",
        *(" ".join(format(float(value), ".17e") for value in row) for row in momenta),
        "END_MOMENTA",
        "NHELICITIES 1",
        "BEGIN_HELICITIES",
        " ".join(f"{int(value):+d}" for value in helicity),
        "END_HELICITIES",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_event(path: Path) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    momenta: list[tuple[float, ...]] = []
    helicity: list[int] = []
    in_momenta = False
    in_helicities = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "BEGIN_MOMENTA":
            in_momenta = True
        elif stripped == "END_MOMENTA":
            in_momenta = False
        elif stripped == "BEGIN_HELICITIES":
            in_helicities = True
        elif stripped == "END_HELICITIES":
            in_helicities = False
        elif in_momenta:
            row = tuple(float(value) for value in stripped.split())
            if len(row) != 4:
                raise StudyError(f"invalid momentum row in {path}")
            momenta.append(row)
        elif in_helicities:
            helicity.extend(int(value) for value in stripped.split())
    if len(momenta) != len(helicity) or any(value not in (-1, 1) for value in helicity):
        raise StudyError(f"invalid fixed-helicity event {path}")
    return tuple(momenta), tuple(helicity)


def _ddbar_events(run_root: Path, final_multiplicity: int) -> tuple[Path, ...]:
    from pyamplicol.generation.phase_space import massive_rambo_final_state

    helicity = fixed_ddbar_helicity(final_multiplicity)
    incoming = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
    )
    result: list[Path] = []
    for point in range(1, POINT_COUNT + 1):
        path = (
            run_root
            / "events"
            / "ddbar"
            / f"n{final_multiplicity}"
            / f"point-{point:02d}.event"
        )
        if not path.is_file():
            outgoing = massive_rambo_final_state(
                final_multiplicity,
                sqrt_s=1000.0,
                masses=(0.0,) * final_multiplicity,
                seed=91_000 + 100 * final_multiplicity + point,
            )
            _write_event(path, (*incoming, *outgoing), helicity)
        result.append(path)
    return tuple(result)


def _candidate_generation_command(
    *,
    python: str,
    family: str,
    final_multiplicity: int,
    mode: Mode,
    helicity: Sequence[int],
    artifact: Path,
) -> tuple[str, ...]:
    assert mode.execution_mode is not None and mode.contraction is not None
    total_external = final_multiplicity + 2
    base_lane = (
        "recurrence" if mode.execution_mode == "compiled" else mode.execution_mode
    )
    command = list(
        performance.candidate_generation_command(
            python=python,
            lane=base_lane,
            total_gluons=total_external,
            helicities=helicity,
            artifact=artifact,
        )
    )
    command[4] = process_expression(family, final_multiplicity)
    command[command.index("--name") + 1] = process_key(family, final_multiplicity)
    command[command.index("--color-contraction") + 1] = mode.contraction
    command[command.index("--execution-mode") + 1] = mode.execution_mode
    if mode.execution_mode == "compiled":
        level = command.index("evaluator.jit.optimization_level=2")
        command[level] = "evaluator.jit.optimization_level=3"
    return tuple(command)


def _candidate_probe_command(
    *,
    probe: Path,
    artifact: Path,
    process: str,
    alpha_s: float,
    target_seconds: float,
    events: Sequence[Path],
) -> tuple[str, ...]:
    return (
        str(probe),
        str(artifact),
        process,
        "--alpha-s",
        f"{alpha_s:.17g}",
        "--target-seconds",
        f"{target_seconds:.17g}",
        "--samples",
        str(WARM_SAMPLES),
        *(str(path) for path in events),
    )


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(observed), abs(expected), 1.0e-300)


def _require_bounded_rss(max_rss_kib: int) -> int:
    if max_rss_kib < 1 or max_rss_kib > int(MEMORY_LIMIT_GIB * 1024**2):
        raise StudyError("runtime RSS is outside the 30 GiB study bound")
    return max_rss_kib


def _numerical_evidence(
    *,
    family: str,
    final_multiplicity: int,
    alpha_s: float,
    observed: Sequence[float],
    baseline: Mapping[str, object] | None,
) -> dict[str, object]:
    if baseline is None or baseline.get("status") != "measured":
        return {"available": False, "passes": None, "reason": "baseline unavailable"}
    baseline_values = baseline.get("point_values")
    if not isinstance(baseline_values, list) or not baseline_values:
        raise StudyError("baseline cell has no numerical values")
    if family == "gg":
        if len(observed) not in {1, len(baseline_values)}:
            raise StudyError("reference/candidate point counts differ")
        exponent = final_multiplicity
        factor = (
            performance.INITIAL_GLUON_AVERAGE_FACTOR
            * math.factorial(exponent)
            / (4.0 * math.pi * alpha_s) ** exponent
        )
        errors = [
            _relative_error(float(value) * factor, float(reference))
            for value, reference in zip(
                observed, baseline_values[: len(observed)], strict=True
            )
        ]
    else:
        errors = [_relative_error(float(observed[0]), float(baseline_values[0]))]
        factor = 1.0
    maximum = max(errors)
    return {
        "available": True,
        "normalization_factor": factor,
        "maximum_relative_error": maximum,
        "relative_tolerance": NUMERICAL_RELATIVE_TOLERANCE,
        "passes": maximum <= NUMERICAL_RELATIVE_TOLERANCE,
    }


def _timed_command(command: Sequence[str]) -> tuple[tuple[str, ...], Any]:
    if sys.platform == "darwin":
        return (
            "/usr/bin/time",
            "-l",
            *command,
        ), performance._synthesize_darwin_rss_markers
    return ("/usr/bin/time", f"--format={RSS_MARKER} %M", *command), None


def _parse_rss_marker(stderr: str) -> int:
    matches = re.findall(rf"^{re.escape(RSS_MARKER)}\s+(\d+)\s*$", stderr, re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) < 1:
        raise StudyError("timed process did not report one positive RSS marker")
    return int(matches[0])


def _exec_in_command(
    python: str, directory: Path, command: Sequence[str]
) -> tuple[str, ...]:
    return (python, str(Path(__file__).resolve()), "_exec-in", str(directory), *command)


def _parse_amplicol_timing(output: str) -> tuple[float, float, int]:
    rows: dict[str, float] = {}
    for label, raw_value in AMPLI_COL_TIMING_ROW.findall(output):
        if label in rows:
            raise StudyError(f"AmpliCol timing row {label!r} is duplicated")
        rows[label] = float(raw_value.replace("D", "E").replace("d", "e"))
    point_matches = re.findall(r"^points\s+(\d+)\s*$", output, re.MULTILINE)
    if set(rows) != {"generation setup", "total"} or len(point_matches) != 1:
        raise StudyError("AmpliCol timing output is incomplete")
    points = int(point_matches[0])
    if (
        points < 1
        or rows["generation setup"] < 0.0
        or not _positive_finite(rows["total"])
    ):
        raise StudyError("AmpliCol timing output contains invalid values")
    return rows["generation setup"], rows["total"] / points, points


def _empty_report(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "policy": dry_run_plan(arguments),
        "cells": {
            family: {
                mode.key: {}
                for mode in MODES
                if not (family == "ddbar" and mode.kind == "reference")
            }
            for family in FAMILIES
        },
    }


def _load_report(path: Path, arguments: argparse.Namespace) -> dict[str, object]:
    if not path.is_file():
        return _empty_report(arguments)
    if not arguments.resume:
        raise StudyError(f"study report already exists; pass --resume: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != KIND or payload.get("schema_version") != SCHEMA_VERSION:
        raise StudyError("existing scaling-study report has the wrong schema")
    if not isinstance(payload.get("cells"), dict):
        raise StudyError("existing scaling-study report has no cells")
    if payload.get("policy") != dry_run_plan(arguments):
        raise StudyError("resume arguments differ from the stored study policy")
    payload["status"] = "running"
    return payload


def _record(
    report: dict[str, object],
    report_path: Path,
    family: str,
    mode: str,
    final_multiplicity: int,
    cell: Mapping[str, object],
) -> None:
    cells = report["cells"]
    assert isinstance(cells, dict)
    family_cells = cells[family]
    assert isinstance(family_cells, dict)
    curve = family_cells[mode]
    assert isinstance(curve, dict)
    curve[str(final_multiplicity)] = dict(cell)
    performance._write_report(report_path, report)


def _curve(report: Mapping[str, object], family: str, mode: str) -> dict[str, object]:
    cells = report["cells"]
    assert isinstance(cells, dict)
    family_cells = cells[family]
    assert isinstance(family_cells, dict)
    result = family_cells[mode]
    assert isinstance(result, dict)
    return result


def _cell_base(family: str, mode: Mode, final_multiplicity: int) -> dict[str, object]:
    return {
        "family": family,
        "mode": mode.key,
        "label": mode.label,
        "n": final_multiplicity,
        "total_external": final_multiplicity + 2,
        "process": process_expression(family, final_multiplicity),
        "color_accuracy": "full",
    }


def _failure_cell(
    base: Mapping[str, object], error: Exception, cell_root: Path
) -> dict[str, object]:
    detail = str(error)
    evidence = detail
    for path in sorted(cell_root.rglob("*.json")):
        with contextlib.suppress(OSError):
            evidence += "\n" + path.read_text(encoding="utf-8")
    category = "error"
    if "setup/runtime probe" in detail and (
        "timed out" in evidence or "deadline" in evidence
    ):
        category = "setup-or-runtime-time-limit"
    elif "runtime probe" in detail and (
        "timed out" in evidence or "deadline" in evidence
    ):
        category = "runtime-time-limit"
    elif "timed out" in evidence or "deadline" in evidence:
        category = "generation-time-limit"
    elif (
        "memory limit exceeded" in evidence
        or "memory-limit-exceeded" in evidence
        or "RSS is outside the 30 GiB" in evidence
    ):
        category = "memory-limit"
    return dict(base) | {
        "status": "failed",
        "censors_higher_multiplicities": True,
        "failure_category": category,
        "failure_reason": detail,
    }


def _reference_cell(
    *,
    arguments: argparse.Namespace,
    reference: Any,
    run_root: Path,
    environment: Mapping[str, str],
    final_multiplicity: int,
) -> dict[str, object]:
    original_run_watched = performance._run_watched
    original_build_executables = reference.build_executables
    deadline = time.perf_counter() + arguments.generation_timeout
    generation_active = True

    def bounded_build_executables(*args: Any, **kwargs: Any) -> Any:
        nonlocal generation_active
        try:
            return original_build_executables(*args, **kwargs)
        finally:
            generation_active = False

    def bounded_run_watched(
        command: Sequence[str], **kwargs: Any
    ) -> performance.WatchedCompletedProcess:
        if generation_active:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                raise StudyError(
                    "reference cell exhausted the aggregate generation cap"
                )
            kwargs["timeout_seconds"] = min(float(kwargs["timeout_seconds"]), remaining)
        else:
            kwargs["timeout_seconds"] = min(
                float(kwargs["timeout_seconds"]), arguments.runtime_timeout
            )
        return original_run_watched(command, **kwargs)

    performance._run_watched = bounded_run_watched
    reference.build_executables = bounded_build_executables
    try:
        result = performance._run_reference(
            reference=reference,
            total_gluons=final_multiplicity + 2,
            run_root=run_root,
            python=arguments.python,
            environment=environment,
            fc=arguments.fc,
            target_seconds=arguments.target_seconds,
            timeout_seconds=arguments.generation_timeout,
        )
    finally:
        performance._run_watched = original_run_watched
        reference.build_executables = original_build_executables
    metrics = result.metrics
    if metrics.clean_build_seconds > arguments.generation_timeout:
        raise StudyError("reference generation exceeded the aggregate generation cap")
    return _cell_base("gg", MODE_BY_KEY["reference-fft"], final_multiplicity) | {
        "status": "measured",
        "helicity": list(metrics.selected_helicity),
        "event_paths": [str(path) for path in result.event_paths],
        "point_values": list(metrics.matrix_elements),
        "metrics": {
            "generation_seconds": metrics.clean_build_seconds,
            "warm_seconds_per_point": metrics.warm_median_seconds,
            "max_rss_kib": _require_bounded_rss(metrics.max_rss_kib),
        },
        "reference": asdict(metrics),
    }


def _amplicol_cell(
    *,
    arguments: argparse.Namespace,
    family: str,
    final_multiplicity: int,
    events: Sequence[Path],
    helicity: Sequence[int],
    baseline: Mapping[str, object] | None,
    run_root: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    base = _cell_base(family, MODE_BY_KEY["amplicol"], final_multiplicity)
    cell_root = run_root / "amplicol" / family / f"n{final_multiplicity}"
    setup = cell_root / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    generation = performance._run_watched(
        _exec_in_command(
            arguments.python,
            setup,
            (
                arguments.python,
                str(arguments.amplicol_repository / "process_list.py"),
                "--serial",
                process_expression(family, final_multiplicity),
            ),
        ),
        python=arguments.python,
        environment=environment,
        timeout_seconds=arguments.generation_timeout,
        log_path=cell_root / "process-generation.json",
    )
    process_file = setup / "processes.txt"
    entries = legacy_amplicol.parse_process_file(process_file)
    source_pdgs = legacy_amplicol.process_pdgs(
        process_expression(family, final_multiplicity)
    )
    entry, matches = legacy_amplicol.select_generated_process_entry(
        entries,
        generated_process=process_expression(family, final_multiplicity),
        wanted_pdgs=source_pdgs,
    )
    momenta, event_helicity = _read_event(events[0])
    if tuple(helicity) != event_helicity:
        raise StudyError(
            "AmpliCol event helicity differs from the frozen cell helicity"
        )
    permutation = legacy_amplicol._permutation(source_pdgs, entry.process_pdgs)
    ordered_momenta = legacy_amplicol._ordered_binary64_momenta(
        source_pdgs, entry.process_pdgs, momenta
    )
    ordered_helicity = tuple(int(helicity[index]) for index in permutation)
    momentum_path = setup / "momenta.dat"
    momentum_path.write_text(
        "\n".join(
            " ".join(format(float(component), ".17g") for component in row)
            for row in ordered_momenta
        )
        + "\n",
        encoding="utf-8",
    )
    probe_command = _exec_in_command(
        arguments.python,
        setup,
        (
            str((arguments.amplicol_repository / "amplicol_color_probe").resolve()),
            str(AMPLI_COL_MAX_POINTS),
            str(entry.group),
            str(entry.integral),
            "full",
            str(process_file),
            str(momentum_path),
            *(str(value) for value in ordered_helicity),
        ),
    )
    timed_command, normalizer = _timed_command(probe_command)
    probe_environment = dict(environment)
    probe_environment["AMPICOL_COLOR_PROBE_TARGET_RUNTIME_S"] = (
        f"{arguments.target_seconds:.17g}"
    )
    remaining = arguments.generation_timeout - generation.elapsed_seconds
    if remaining <= 0.0:
        raise StudyError("AmpliCol process generation exhausted the generation cap")
    try:
        completed = performance._run_watched(
            timed_command,
            python=arguments.python,
            environment=probe_environment,
            timeout_seconds=remaining,
            log_path=cell_root / "probe.json",
            normalize_completed=normalizer,
        )
    except performance.AcceptanceError as error:
        raise StudyError(f"AmpliCol setup/runtime probe failed: {error}") from error
    generation_setup, warm_per_point, points = _parse_amplicol_timing(completed.stdout)
    generation_seconds = generation.elapsed_seconds + generation_setup
    if generation_seconds > arguments.generation_timeout:
        raise StudyError("AmpliCol generation exceeded the aggregate generation cap")
    probe = legacy_amplicol._parse_probe_output(completed.stdout)
    if not math.isfinite(probe.value) or abs(probe.value) <= 1.0e-300:
        raise StudyError("AmpliCol frozen helicity is zero or non-finite")
    numerical = _numerical_evidence(
        family=family,
        final_multiplicity=final_multiplicity,
        alpha_s=arguments.alpha_s,
        observed=(probe.value,),
        baseline=baseline,
    )
    if numerical["passes"] is False:
        raise StudyError(
            "AmpliCol numerical comparison failed: relative error "
            f"{numerical['maximum_relative_error']:.6g}"
        )
    max_rss_kib = _require_bounded_rss(_parse_rss_marker(completed.stderr))
    return base | {
        "status": "measured",
        "helicity": list(helicity),
        "event_paths": [str(path) for path in events],
        "point_values": [probe.value],
        "metrics": {
            "generation_seconds": generation_seconds,
            "warm_seconds_per_point": warm_per_point,
            "max_rss_kib": max_rss_kib,
        },
        "adaptive_runtime_points": points,
        "process_entry": asdict(entry) | {"matching_row_count": len(matches)},
        "numerical": numerical,
    }


def _candidate_cell(
    *,
    arguments: argparse.Namespace,
    family: str,
    final_multiplicity: int,
    mode: Mode,
    events: Sequence[Path],
    helicity: Sequence[int],
    baseline: Mapping[str, object] | None,
    probe: Path,
    run_root: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    assert mode.execution_mode is not None
    base = _cell_base(family, mode, final_multiplicity)
    cell_root = run_root / "candidate" / mode.key / family / f"n{final_multiplicity}"
    artifact = cell_root / "artifact"
    candidate_environment = dict(environment)
    candidate_environment["PYAMPLICOL_CACHE_DIR"] = str(
        cell_root / "disabled-model-cache"
    )
    generation = performance._run_watched(
        _candidate_generation_command(
            python=arguments.python,
            family=family,
            final_multiplicity=final_multiplicity,
            mode=mode,
            helicity=helicity,
            artifact=artifact,
        ),
        python=arguments.python,
        environment=candidate_environment,
        timeout_seconds=arguments.generation_timeout,
        log_path=cell_root / "generation.json",
    )
    try:
        completed = performance._run_watched(
            _candidate_probe_command(
                probe=probe,
                artifact=artifact,
                process=process_key(family, final_multiplicity),
                alpha_s=arguments.alpha_s,
                target_seconds=arguments.target_seconds,
                events=events,
            ),
            python=arguments.python,
            environment=candidate_environment,
            timeout_seconds=arguments.runtime_timeout,
            log_path=cell_root / "probe.json",
        )
    except performance.AcceptanceError as error:
        raise StudyError(f"candidate runtime probe failed: {error}") from error
    parsed = _parse_candidate_rows(completed.stdout)
    if parsed["process"] != process_key(family, final_multiplicity):
        raise StudyError("candidate probe loaded the wrong process")
    if parsed["execution_mode"] != mode.execution_mode:
        raise StudyError("candidate probe loaded the wrong execution mode")
    coverage = int(parsed["helicity_coverage_count"])
    if mode.execution_mode == "on-the-fly":
        if coverage <= 1:
            raise StudyError("OTF generation did not retain complete helicity coverage")
    elif coverage != 1:
        raise StudyError("specialized candidate retained more than one helicity")
    values = parsed["point_values"]
    assert isinstance(values, list)
    if not any(abs(float(value)) > 1.0e-300 for value in values):
        raise StudyError("candidate frozen helicity is structurally zero")
    numerical = _numerical_evidence(
        family=family,
        final_multiplicity=final_multiplicity,
        alpha_s=arguments.alpha_s,
        observed=[float(value) for value in values],
        baseline=baseline,
    )
    if numerical["passes"] is False:
        raise StudyError(
            "candidate numerical comparison failed: relative error "
            f"{numerical['maximum_relative_error']:.6g}"
        )
    return base | {
        "status": "measured",
        "helicity": list(helicity),
        "event_paths": [str(path) for path in events],
        "artifact": str(artifact),
        "point_values": values,
        "metrics": {
            "generation_seconds": generation.elapsed_seconds,
            "warm_seconds_per_point": parsed["warm_median_seconds"],
            "max_rss_kib": _require_bounded_rss(int(parsed["max_rss_kib"])),
        },
        "probe": parsed,
        "numerical": numerical,
        "otf_generation_helicity_coverage": "complete"
        if mode.execution_mode == "on-the-fly"
        else "selected",
    }


def _inputs(
    report: Mapping[str, object], run_root: Path, family: str, final_multiplicity: int
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    if family == "ddbar":
        return _ddbar_events(run_root, final_multiplicity), fixed_ddbar_helicity(
            final_multiplicity
        )
    reference = _curve(report, "gg", "reference-fft").get(str(final_multiplicity))
    if not isinstance(reference, dict) or reference.get("status") != "measured":
        raise StudyError("shared FFT reference cell is unavailable")
    raw_events = reference.get("event_paths")
    raw_helicity = reference.get("helicity")
    if not isinstance(raw_events, list) or not isinstance(raw_helicity, list):
        raise StudyError("shared FFT reference cell has incomplete inputs")
    events = tuple(Path(str(path)) for path in raw_events)
    if len(events) != POINT_COUNT or not all(path.is_file() for path in events):
        raise StudyError("shared FFT reference events are unavailable")
    return events, tuple(int(value) for value in raw_helicity)


def _baseline(
    report: Mapping[str, object], family: str, final_multiplicity: int
) -> Mapping[str, object] | None:
    key = "reference-fft" if family == "gg" else "amplicol"
    value = _curve(report, family, key).get(str(final_multiplicity))
    return value if isinstance(value, dict) else None


def _prepare_support_directories() -> None:
    for path in (
        performance.PERFORMANCE_ROOT / "tmp",
        performance.PERFORMANCE_ROOT / "cache" / "cargo-home",
        performance.PERFORMANCE_ROOT / "cache" / "cargo-target",
        performance.PERFORMANCE_ROOT / "cache" / "pip-cache",
        performance.PERFORMANCE_ROOT / "cache" / "xdg-cache",
        performance.PERFORMANCE_ROOT / "cache" / "python-cache",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _archive_interrupted_cell(cell_root: Path) -> None:
    """Preserve an unrecorded partial attempt before a resumed cell reruns."""

    if not cell_root.exists():
        return
    for attempt in range(1, 10_000):
        archived = cell_root.with_name(f"{cell_root.name}.interrupted-{attempt}")
        if not archived.exists():
            cell_root.rename(archived)
            return
    raise StudyError(f"too many interrupted attempts beside {cell_root}")


def _campaign(arguments: argparse.Namespace) -> dict[str, object]:
    run_root = STUDY_ROOT / "runs" / arguments.run_id
    report_path = run_root / "report.json"
    run_root.mkdir(parents=True, exist_ok=True)
    report = _load_report(report_path, arguments)
    performance._write_report(report_path, report)
    _prepare_support_directories()
    environment = performance._workspace_environment(arguments.python)
    os.environ.update(environment)

    # Rebuilding this tiny probe on resume avoids trusting a partially linked
    # executable left by a hard interruption.
    probe, _ = performance._build_probe(
        run_root=run_root,
        python=arguments.python,
        cxx=arguments.cxx,
        environment=environment,
    )
    legacy_amplicol.validate_checkout(arguments.amplicol_repository)
    amplicol_probe = arguments.amplicol_repository / "amplicol_color_probe"
    if not amplicol_probe.is_file():
        if not arguments.build_amplicol:
            raise StudyError(
                "AmpliCol probe is missing; rerun with --build-amplicol: "
                f"{amplicol_probe}"
            )
        performance._run_watched(
            (
                "make",
                "-C",
                str(arguments.amplicol_repository),
                "-j1",
                "PDF_BACKEND=internal",
                "amplicol_color_probe",
            ),
            python=arguments.python,
            environment=environment,
            timeout_seconds=arguments.generation_timeout,
            log_path=run_root / "logs" / "amplicol-probe-build.json",
        )

    reference_module = performance._load_reference_module()
    for family in FAMILIES:
        for mode in MODES:
            if family == "ddbar" and mode.kind == "reference":
                continue
            curve = _curve(report, family, mode.key)
            failed_at: int | None = next(
                (
                    n
                    for n in FINAL_MULTIPLICITIES
                    if isinstance(curve.get(str(n)), dict)
                    and curve[str(n)].get("status") == "failed"
                    and curve[str(n)].get("censors_higher_multiplicities") is True
                ),
                None,
            )
            for final_multiplicity in FINAL_MULTIPLICITIES:
                if str(final_multiplicity) in curve:
                    continue
                base = _cell_base(family, mode, final_multiplicity)
                if failed_at is not None and final_multiplicity > failed_at:
                    _record(
                        report,
                        report_path,
                        family,
                        mode.key,
                        final_multiplicity,
                        base
                        | {
                            "status": "skipped",
                            "failure_reason": (
                                f"curve censored after failure at n={failed_at}"
                            ),
                            "failed_at_n": failed_at,
                        },
                    )
                    continue
                if (
                    family == "ddbar"
                    and mode.key == "compiled-fft"
                    and final_multiplicity < 4
                ):
                    _record(
                        report,
                        report_path,
                        family,
                        mode.key,
                        final_multiplicity,
                        base
                        | {
                            "status": "not-applicable",
                            "failure_reason": (
                                "compiled FFT requires a nontrivial "
                                "symmetric-group block"
                            ),
                            "censors_higher_multiplicities": False,
                        },
                    )
                    continue
                if mode.kind == "reference":
                    cell_root = run_root / "reference" / f"N{final_multiplicity + 2}"
                elif mode.kind == "amplicol":
                    cell_root = (
                        run_root / "amplicol" / family / f"n{final_multiplicity}"
                    )
                else:
                    cell_root = (
                        run_root
                        / "candidate"
                        / mode.key
                        / family
                        / f"n{final_multiplicity}"
                    )
                _archive_interrupted_cell(cell_root)
                try:
                    if mode.kind == "reference":
                        cell = _reference_cell(
                            arguments=arguments,
                            reference=reference_module,
                            run_root=run_root,
                            environment=environment,
                            final_multiplicity=final_multiplicity,
                        )
                    else:
                        events, helicity = _inputs(
                            report, run_root, family, final_multiplicity
                        )
                        baseline = _baseline(report, family, final_multiplicity)
                        if mode.kind == "amplicol":
                            cell = _amplicol_cell(
                                arguments=arguments,
                                family=family,
                                final_multiplicity=final_multiplicity,
                                events=events,
                                helicity=helicity,
                                baseline=baseline if family == "gg" else None,
                                run_root=run_root,
                                environment=environment,
                            )
                        else:
                            cell = _candidate_cell(
                                arguments=arguments,
                                family=family,
                                final_multiplicity=final_multiplicity,
                                mode=mode,
                                events=events,
                                helicity=helicity,
                                baseline=baseline,
                                probe=probe,
                                run_root=run_root,
                                environment=environment,
                            )
                except (
                    StudyError,
                    performance.AcceptanceError,
                    legacy_amplicol.LegacyOracleError,
                    OSError,
                    ValueError,
                ) as error:
                    cell = _failure_cell(base, error, cell_root)
                    failed_at = final_multiplicity
                _record(report, report_path, family, mode.key, final_multiplicity, cell)

    failures = sum(
        cell.get("status") == "failed"
        for family in FAMILIES
        for curve in (
            value.values()
            for key, value in report["cells"][family].items()  # type: ignore[index,union-attr]
            if key in MODE_BY_KEY
        )
        for cell in curve
        if isinstance(cell, dict)
    )
    report["status"] = "complete-with-failures" if failures else "complete"
    report["failure_count"] = failures
    performance._write_report(report_path, report)
    return report


def dry_run_plan(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "run_root": str(STUDY_ROOT / "runs" / arguments.run_id),
        "tools": {
            "python": arguments.python,
            "cxx": arguments.cxx,
            "fc": arguments.fc,
            "amplicol_repository": str(arguments.amplicol_repository.resolve()),
        },
        "process_families": {
            "gg": {
                "expression": "g g > g g + (n-2)*g",
                "ratio_reference": "reference-fft",
                "modes": [mode.key for mode in MODES],
            },
            "ddbar": {
                "expression": "d d~ > d d~ + (n-2)*g",
                "ratio_reference": "amplicol",
                "modes": [mode.key for mode in MODES if mode.kind != "reference"],
            },
        },
        "final_state_multiplicities": list(FINAL_MULTIPLICITIES),
        "total_external_particles": [n + 2 for n in FINAL_MULTIPLICITIES],
        "measurement": {
            "color_accuracy": "full",
            "n_definition": "final-state-particle-count",
            "fixed_helicity": True,
            "otf_generation_helicity_coverage": "complete",
            "warm_samples": WARM_SAMPLES,
            "calibration_target_seconds": arguments.target_seconds,
            "generation_timeout_seconds": arguments.generation_timeout,
            "runtime_timeout_seconds": arguments.runtime_timeout,
            "memory_watchdog_gib": MEMORY_LIMIT_GIB,
            "alpha_s": arguments.alpha_s,
            "failure_censors_higher_n_in_same_curve": True,
            "ddbar_compiled_fft_n2_n3": "not-applicable-without-censoring",
            "generation_metric": {
                "reference-fft": "clean backend build",
                "amplicol": (
                    "process-list wall time plus Fortran CPU-time color-object setup"
                ),
                "pyamplicol": "artifact generation wall time",
            },
            "warm_timing_metric": {
                "reference-fft": "median of 10 calibrated process-CPU cells",
                "amplicol": (
                    "one >=0.25 s adaptive process-CPU aggregate divided by points"
                ),
                "pyamplicol": "median of 10 calibrated process-CPU cells",
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default="fullcolor-n2-n7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--alpha-s", type=float, default=DEFAULT_ALPHA_S)
    parser.add_argument("--target-seconds", type=float, default=0.25)
    parser.add_argument(
        "--generation-timeout", type=float, default=MAX_GENERATION_SECONDS
    )
    parser.add_argument("--runtime-timeout", type=float, default=MAX_GENERATION_SECONDS)
    parser.add_argument(
        "--amplicol-repository",
        type=Path,
        default=legacy_amplicol.DEFAULT_REPOSITORY,
    )
    parser.add_argument("--build-amplicol", action="store_true")
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if performance._RUN_ID.fullmatch(arguments.run_id) is None or arguments.run_id in {
        ".",
        "..",
    }:
        raise StudyError("--run-id is not a safe workspace-local name")
    if arguments.alpha_s != DEFAULT_ALPHA_S:
        raise StudyError(
            "this FullColor study requires --alpha-s 0.118 to match pinned AmpliCol"
        )
    if (
        not _positive_finite(arguments.target_seconds)
        or arguments.target_seconds < 0.25
    ):
        raise StudyError("--target-seconds must be at least 0.25")
    if (
        not _positive_finite(arguments.generation_timeout)
        or arguments.generation_timeout > MAX_GENERATION_SECONDS
    ):
        raise StudyError("--generation-timeout must be in (0, 1800]")
    if not _positive_finite(arguments.runtime_timeout):
        raise StudyError("--runtime-timeout must be positive")


def _exec_in(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        raise StudyError("_exec-in requires a directory and command")
    directory = Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    command = list(argv[1:])
    os.execvpe(command[0], command, os.environ)
    raise AssertionError("os.execvpe returned")


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        if raw and raw[0] == "_exec-in":
            return _exec_in(raw[1:])
        arguments = _parser().parse_args(raw)
        _validate_arguments(arguments)
        if arguments.dry_run:
            print(json.dumps(dry_run_plan(arguments), indent=2, sort_keys=True))
            return 0
        result = _campaign(arguments)
        print(
            json.dumps(
                {"status": result["status"], "failure_count": result["failure_count"]},
                indent=2,
            )
        )
        return 0
    except (StudyError, performance.AcceptanceError, OSError, ValueError) as error:
        print(f"FFT scaling study: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
