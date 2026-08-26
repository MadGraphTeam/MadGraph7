#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Paired pure-gluon FFT performance, RSS, and cold-start acceptance.

The authoritative reference implementation is loaded from the pinned
developer checkout's ``Benchmark/run_benchmark.py`` and called directly; none
of its implementation is copied here.  Candidate timing uses the public
Rusticol C ABI probe in this directory.  Linux uses the reference driver's GNU
process wrappers.  Darwin translates those wrappers to the study's getrusage
worker; the common watchdog and subprocess timeout retain the same hard bounds.

This file intentionally is not wired into a default target.  Use ``--dry-run``
to inspect the complete bounded campaign without compiling or writing files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.performance_report.source_identity import (  # noqa: E402
    ReportSourceIdentity,
    ReportSourceIdentityError,
    require_eligible_report_source,
)

PERFORMANCE_ROOT = ROOT / ".artifacts" / "fft-performance"
DEFAULT_REFERENCE_ROOT = ROOT / "dependencies" / "checkouts" / "reference-fft"
REFERENCE_ROOT = DEFAULT_REFERENCE_ROOT
REFERENCE_DRIVER = REFERENCE_ROOT / "Benchmark" / "run_benchmark.py"
REFERENCE_REVISION = "9c3cb4fb4658200884553bab796e85bd5e7fe7a9"
PROBE_SOURCE = ROOT / "tools" / "developer" / "fft_gluon_candidate_probe.cpp"
SCALING_STUDY_DRIVER = ROOT / "tools" / "developer" / "fft_scaling_study.py"
WATCHDOG = ROOT / "tools" / "ci" / "memory_watchdog.py"

KIND = "pyamplicol-pure-gluon-fft-performance-acceptance"
SCHEMA_VERSION = 8
REFERENCE_BACKEND = "AmpliGluonTraceDefaultBG"
REFERENCE_BUILD_PROGRAM = "benchmark_ampligluon_trace"
REFERENCE_CLEAN_BUILD_SCOPE = "ampligluon-trace-backend-only"
# Recurrence is generated for one known-nonzero helicity.  OTF retains complete
# runtime-selector coverage and is measured through one fixed-helicity query.
# Both are valid, lane-specific acceptance workloads.  Running recurrence first
# prevents OTF from warming its cold path.
LANES = ("recurrence", "on-the-fly")
MANDATORY_MULTIPLICITIES = tuple(range(4, 10))
OPTIONAL_MULTIPLICITIES = (10, 11)
POINT_COUNT = 10
REPRESENTATIVE_POINT = 1
BASE_SEED = 1729
WARM_SAMPLE_COUNT = 10
MINIMUM_CALIBRATION_SECONDS = 0.25
MEMORY_LIMIT_GIB = 30.0
OPTIONAL_COLD_LIMIT_SECONDS = 15.0 * 60.0
MANDATORY_LANE_TIMEOUT_SECONDS = 60.0 * 60.0
WARM_RATIO_LIMIT = 1.25
RSS_RATIO_LIMIT = 2.0
COLD_RATIO_LIMIT = 5.0
NUMERICAL_RELATIVE_TOLERANCE = 1.0e-10
INITIAL_GLUON_AVERAGE_FACTOR = 256
UNIT_COUPLING_ALPHA_S = 1.0 / (4.0 * math.pi)
DARWIN_RSS_MARKER = "FFT_MAX_RSS_KIB"
REFERENCE_RSS_MARKER = "BENCHMARK_MAX_RSS_KIB"
WATCHDOG_CLEANUP_WAIT_SECONDS = 7.0
SINGLE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ACCELERATE_MAX_THREADS",
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class AcceptanceError(RuntimeError):
    """Raised when the paired campaign or its evidence is invalid."""


def configure_reference_root(path: Path) -> None:
    """Select the independent pinned Reference FFT checkout for this process."""

    global REFERENCE_ROOT, REFERENCE_DRIVER
    REFERENCE_ROOT = path.expanduser().resolve()
    REFERENCE_DRIVER = REFERENCE_ROOT / "Benchmark" / "run_benchmark.py"


class NumericalParityError(AcceptanceError):
    """Raised when an evaluated candidate disagrees with the reference."""


class ReferenceColdLimitError(AcceptanceError):
    """Raised when Reference setup exhausts its aggregate cold-cell budget."""


@dataclass(frozen=True)
class ReferenceSourceIdentity:
    """Pinned revision and deterministic identity of the reference inputs."""

    revision: str
    content_sha256: str
    file_count: int

    def provenance(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "required_revision": REFERENCE_REVISION,
            "content_sha256": self.content_sha256,
            "tracked_and_nonignored_file_count": self.file_count,
        }


@dataclass(frozen=True)
class CandidateProbeMetrics:
    process: str
    execution_mode: str
    helicity_coverage_count: int
    selected_helicity_id: str
    point_count: int
    point_values: tuple[float, ...]
    load_seconds: float
    first_warm_seconds: float
    warm_up_api_seconds: float
    calibration_calls: tuple[int, ...]
    calibration_seconds: tuple[float, ...]
    warm_cell_seconds: tuple[tuple[float, ...], ...]
    warm_samples_seconds: tuple[float, ...]
    warm_median_seconds: float
    minimum_absolute_value: float
    max_rss_kib: int


@dataclass(frozen=True)
class NumericalParityEvidence:
    normalization_alpha_s_me_check: float
    candidate_scale_factor: int
    relative_tolerance: float
    maximum_relative_error: float
    maximum_relative_error_point: int
    passes: bool


@dataclass(frozen=True)
class CandidateMetrics:
    lane: str
    total_gluons: int
    generator_seed: int
    generation_seconds: float
    load_seconds: float
    first_warm_seconds: float
    max_rss_kib: int
    probe: CandidateProbeMetrics
    numerical_parity: NumericalParityEvidence

    @property
    def cold_to_ready_seconds(self) -> float:
        return self.generation_seconds + self.load_seconds + self.first_warm_seconds


@dataclass(frozen=True)
class CandidateFirstReadyMetrics:
    process: str
    execution_mode: str
    helicity_coverage_count: int
    selected_helicity_id: str
    point_count: int
    load_seconds: float
    first_warm_seconds: float
    warm_up_api_seconds: float
    minimum_absolute_value: float
    max_rss_kib: int


class WatchedCompletedProcess(subprocess.CompletedProcess[str]):
    """Completed watchdog process with parent-side timing evidence."""

    def __init__(
        self,
        args: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        elapsed_seconds: float,
        log_write_seconds: float,
        timed_out: bool,
        timeout_cleanup: str,
        peak_guard_kib: int | None = None,
        peak_rss_kib: int | None = None,
    ) -> None:
        super().__init__(args, returncode, stdout, stderr)
        self.elapsed_seconds = elapsed_seconds
        self.log_write_seconds = log_write_seconds
        self.peak_guard_kib = peak_guard_kib
        self.peak_rss_kib = peak_rss_kib
        self.timed_out = timed_out
        self.timeout_cleanup = timeout_cleanup


@dataclass(frozen=True)
class ReferenceMetrics:
    total_gluons: int
    generator_seed: int
    backend: str
    clean_build_scope: str
    clean_build_command_count: int
    clean_build_seconds: float
    setup_to_driver_seconds: float
    initialization_seconds: float
    first_pass_seconds: float
    warm_samples_seconds: tuple[float, ...]
    warm_median_seconds: float
    max_rss_kib: int
    selected_helicity: tuple[int, ...]
    selected_path: str
    event_paths: tuple[str, ...]
    matrix_elements: tuple[float, ...]
    driver_max_rss_kib: int | None = None
    watchdog_max_rss_kib: int | None = None
    direct_command_watchdog_max_rss_kib: int | None = None
    translated_child_max_rss_kib: int | None = None
    watchdog_max_peak_guard_kib: int | None = None
    warm_repetitions: tuple[int, ...] = ()
    warm_repetition_quantum: int | None = None
    warm_calibration_seconds: tuple[float, ...] = ()
    helicity_workload: str = "fixed"
    helicity_coverage_count: int = 1
    timed_helicity_count: int = 1
    active_helicity_count: int = 1
    exhaustive_event_paths: tuple[str, ...] = ()

    @property
    def cold_to_ready_seconds(self) -> float:
        return (
            self.clean_build_seconds
            + self.initialization_seconds
            + self.first_pass_seconds
        )

    @property
    def setup_to_ready_seconds(self) -> float:
        return (
            self.setup_to_driver_seconds
            + self.initialization_seconds
            + self.first_pass_seconds
        )


@dataclass(frozen=True)
class GateResult:
    total_gluons: int
    lane: str
    warm_ratio: float
    rss_ratio: float
    cold_ratio: float
    warm_passes: bool
    rss_passes: bool
    cold_passes: bool
    numerical_passes: bool

    @property
    def passes(self) -> bool:
        return (
            self.warm_passes
            and self.rss_passes
            and self.cold_passes
            and self.numerical_passes
        )


@dataclass(frozen=True)
class _ReferenceRun:
    metrics: ReferenceMetrics
    event_paths: tuple[Path, ...]


def generator_seed(total_gluons: int) -> int:
    if total_gluons < 4:
        raise AcceptanceError(f"unsupported total-gluon multiplicity {total_gluons}")
    return BASE_SEED + total_gluons


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def candidate_reference_scale_factor(total_gluons: int) -> int:
    """Undo pyAmpliCol's initial-state average and final-gluon symmetry factor."""

    if total_gluons < 2:
        raise AcceptanceError("total-gluon multiplicity must be at least two")
    return INITIAL_GLUON_AVERAGE_FACTOR * math.factorial(total_gluons - 2)


def compare_candidate_to_reference(
    *,
    total_gluons: int,
    candidate_values: Sequence[float],
    reference_values: Sequence[float],
) -> NumericalParityEvidence:
    """Compare the same ordered points after aligning public ME conventions."""

    if len(candidate_values) != POINT_COUNT or len(reference_values) != POINT_COUNT:
        raise AcceptanceError("numerical parity requires exactly 10 paired values")
    if not all(_finite_number(value) for value in candidate_values):
        raise AcceptanceError("candidate numerical parity values must be finite")
    if not all(_positive_finite(value) for value in reference_values):
        raise AcceptanceError("reference numerical parity values must be positive")
    scale_factor = candidate_reference_scale_factor(total_gluons)
    errors: list[float] = []
    for candidate, reference in zip(candidate_values, reference_values, strict=True):
        scaled_candidate = float(candidate) * scale_factor
        scale = max(abs(scaled_candidate), abs(float(reference)))
        errors.append(abs(scaled_candidate - float(reference)) / scale)
    maximum = max(errors)
    maximum_point = errors.index(maximum) + 1
    return NumericalParityEvidence(
        normalization_alpha_s_me_check=UNIT_COUPLING_ALPHA_S,
        candidate_scale_factor=scale_factor,
        relative_tolerance=NUMERICAL_RELATIVE_TOLERANCE,
        maximum_relative_error=maximum,
        maximum_relative_error_point=maximum_point,
        passes=maximum <= NUMERICAL_RELATIVE_TOLERANCE,
    )


def parse_candidate_probe_output(output: str) -> CandidateProbeMetrics:
    """Parse per-cell process-CPU evidence emitted by the public-C probe."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[0] != "FFT_CANDIDATE_PROBE_V4":
        raise AcceptanceError("candidate probe header is missing or invalid")
    scalar: dict[str, str] = {}
    point_values: dict[int, float] = {}
    calibration: dict[int, tuple[int, float]] = {}
    warm_cells: dict[tuple[int, int], float] = {}
    for line in lines[1:]:
        fields = line.split()
        if fields[0] == "POINT_VALUE":
            if len(fields) != 3:
                raise AcceptanceError("candidate point-value row is malformed")
            point = int(fields[1])
            if point in point_values:
                raise AcceptanceError("candidate point value is duplicated")
            point_values[point] = float(fields[2])
            continue
        if fields[0] == "CALIBRATION_CELL":
            if len(fields) != 4:
                raise AcceptanceError("candidate calibration cell is malformed")
            point = int(fields[1])
            if point in calibration:
                raise AcceptanceError("candidate calibration cell is duplicated")
            calibration[point] = (int(fields[2]), float(fields[3]))
            continue
        if fields[0] == "WARM_CELL_SECONDS":
            if len(fields) != 4:
                raise AcceptanceError("candidate warm cell row is malformed")
            key = (int(fields[1]), int(fields[2]))
            if key in warm_cells:
                raise AcceptanceError("candidate warm cell is duplicated")
            warm_cells[key] = float(fields[3])
            continue
        if len(fields) < 2:
            raise AcceptanceError("candidate scalar row is malformed")
        key = fields[0]
        if key in scalar:
            raise AcceptanceError(f"candidate scalar {key} is duplicated")
        scalar[key] = " ".join(fields[1:])
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
        missing = sorted(required - set(scalar))
        extra = sorted(set(scalar) - required)
        raise AcceptanceError(
            f"candidate probe scalar fields differ: missing={missing}, extra={extra}"
        )
    expected_points = set(range(1, POINT_COUNT + 1))
    if set(point_values) != expected_points:
        raise AcceptanceError("candidate probe must report all 10 point values")
    if set(calibration) != {REPRESENTATIVE_POINT}:
        raise AcceptanceError(
            "candidate probe must calibrate only representative point 1"
        )
    expected_warm_cells = {
        (sample, REPRESENTATIVE_POINT) for sample in range(1, WARM_SAMPLE_COUNT + 1)
    }
    if set(warm_cells) != expected_warm_cells:
        raise AcceptanceError(
            "candidate probe must contain one point-1 cell in each of 10 batches"
        )
    execution_mode = scalar["EXECUTION_MODE"]
    if execution_mode not in LANES:
        raise AcceptanceError("candidate probe reported an unknown execution mode")
    if scalar["TIMER_SOURCE"] != "process-cpu-time":
        raise AcceptanceError("candidate probe did not use process CPU time")
    point_count = int(scalar["POINT_COUNT"])
    helicity_coverage_count = int(scalar["HELICITY_COVERAGE_COUNT"])
    max_rss_kib = int(scalar["MAX_RSS_KIB"])
    numeric = {
        key: float(scalar[key])
        for key in (
            "LOAD_SECONDS",
            "FIRST_WARM_SECONDS",
            "WARM_UP_API_SECONDS",
            "MIN_ABSOLUTE_VALUE",
        )
    }
    calibration_calls = (calibration[REPRESENTATIVE_POINT][0],)
    calibration_seconds = (calibration[REPRESENTATIVE_POINT][1],)
    ordered_point_values = tuple(
        point_values[index] for index in range(1, POINT_COUNT + 1)
    )
    warm_cell_rows = tuple(
        (warm_cells[(sample, REPRESENTATIVE_POINT)],)
        for sample in range(1, WARM_SAMPLE_COUNT + 1)
    )
    warm_samples = tuple(row[0] for row in warm_cell_rows)
    if point_count != POINT_COUNT:
        raise AcceptanceError("candidate probe did not use 10 phase-space points")
    if helicity_coverage_count < 1 or not scalar["SELECTED_HELICITY_ID"].startswith(
        "h:"
    ):
        raise AcceptanceError("candidate selected-helicity metadata is invalid")
    if any(value < 1 for value in calibration_calls):
        raise AcceptanceError("candidate calibration call count is not positive")
    if any(value < MINIMUM_CALIBRATION_SECONDS for value in calibration_seconds):
        raise AcceptanceError(
            "every candidate calibration cell must reach 0.25 seconds"
        )
    if max_rss_kib < 1:
        raise AcceptanceError("candidate peak RSS is not positive")
    warm_up_api_seconds = numeric["WARM_UP_API_SECONDS"]
    if execution_mode == "recurrence":
        if warm_up_api_seconds != 0.0:
            raise AcceptanceError(
                "recurrence candidate called the OTF-only warm-up API"
            )
    elif not _positive_finite(warm_up_api_seconds):
        raise AcceptanceError("OTF candidate warm-up API timing is not positive")
    positive_numeric = (
        numeric["LOAD_SECONDS"],
        numeric["FIRST_WARM_SECONDS"],
        *calibration_seconds,
        *(value for row in warm_cell_rows for value in row),
    )
    if not all(_positive_finite(value) for value in positive_numeric):
        raise AcceptanceError("candidate probe contains a non-positive timing")
    if (
        not _finite_number(numeric["MIN_ABSOLUTE_VALUE"])
        or numeric["MIN_ABSOLUTE_VALUE"] < 0.0
        or not all(_finite_number(value) for value in ordered_point_values)
    ):
        raise AcceptanceError("candidate probe contains an invalid matrix element")
    return CandidateProbeMetrics(
        process=scalar["PROCESS"],
        execution_mode=execution_mode,
        helicity_coverage_count=helicity_coverage_count,
        selected_helicity_id=scalar["SELECTED_HELICITY_ID"],
        point_count=point_count,
        point_values=ordered_point_values,
        load_seconds=numeric["LOAD_SECONDS"],
        first_warm_seconds=numeric["FIRST_WARM_SECONDS"],
        warm_up_api_seconds=warm_up_api_seconds,
        calibration_calls=calibration_calls,
        calibration_seconds=calibration_seconds,
        warm_cell_seconds=warm_cell_rows,
        warm_samples_seconds=warm_samples,
        warm_median_seconds=statistics.median(warm_samples),
        minimum_absolute_value=numeric["MIN_ABSOLUTE_VALUE"],
        max_rss_kib=max_rss_kib,
    )


def parse_candidate_first_ready_output(output: str) -> CandidateFirstReadyMetrics:
    """Parse a cold-only probe that exits before calibration and warm batches."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[0] != "FFT_CANDIDATE_FIRST_READY_V1":
        raise AcceptanceError("candidate first-ready header is missing or invalid")
    scalar: dict[str, str] = {}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 2 or fields[0] in scalar:
            raise AcceptanceError("candidate first-ready scalar rows are malformed")
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
        raise AcceptanceError("candidate first-ready scalar fields differ")
    execution_mode = scalar["EXECUTION_MODE"]
    if execution_mode not in LANES or scalar["TIMER_SOURCE"] != "process-cpu-time":
        raise AcceptanceError(
            "candidate first-ready execution/timer metadata is invalid"
        )
    point_count = int(scalar["POINT_COUNT"])
    coverage = int(scalar["HELICITY_COVERAGE_COUNT"])
    max_rss_kib = int(scalar["MAX_RSS_KIB"])
    load_seconds = float(scalar["LOAD_SECONDS"])
    first_warm_seconds = float(scalar["FIRST_WARM_SECONDS"])
    warm_up_api_seconds = float(scalar["WARM_UP_API_SECONDS"])
    minimum_absolute_value = float(scalar["MIN_ABSOLUTE_VALUE"])
    if (
        point_count != POINT_COUNT
        or coverage < 1
        or not scalar["SELECTED_HELICITY_ID"].startswith("h:")
        or max_rss_kib < 1
        or not all(
            _positive_finite(value) for value in (load_seconds, first_warm_seconds)
        )
        or not _finite_number(minimum_absolute_value)
        or minimum_absolute_value < 0.0
    ):
        raise AcceptanceError("candidate first-ready evidence is invalid")
    if execution_mode == "recurrence":
        if warm_up_api_seconds != 0.0:
            raise AcceptanceError(
                "recurrence candidate called the OTF-only warm-up API"
            )
    elif not _positive_finite(warm_up_api_seconds):
        raise AcceptanceError("OTF first-ready warm-up API timing is not positive")
    return CandidateFirstReadyMetrics(
        process=scalar["PROCESS"],
        execution_mode=execution_mode,
        helicity_coverage_count=coverage,
        selected_helicity_id=scalar["SELECTED_HELICITY_ID"],
        point_count=point_count,
        load_seconds=load_seconds,
        first_warm_seconds=first_warm_seconds,
        warm_up_api_seconds=warm_up_api_seconds,
        minimum_absolute_value=minimum_absolute_value,
        max_rss_kib=max_rss_kib,
    )


def select_global_lane(
    references: Mapping[int, ReferenceMetrics],
    candidates: Mapping[str, Mapping[int, CandidateMetrics]],
) -> str:
    """Choose the fastest globally consistent lane that passes every gate."""

    eligibility = lane_eligibility(candidates)
    scores: list[tuple[float, str]] = []
    for lane in LANES:
        lane_policy = eligibility[lane]
        if lane_policy["eligible"] and all(
            gate.passes
            for gate in _lane_gate_results(
                lane,
                references,
                candidates[lane],
                MANDATORY_MULTIPLICITIES,
            )
        ):
            scores.append((float(lane_policy["warm_geometric_mean_seconds"]), lane))
    if not scores:
        raise AcceptanceError("no eligible lane passes every mandatory acceptance gate")
    # The lane name is a stable tie-breaker; selection never changes per N.
    return min(scores)[1]


def lane_eligibility(
    candidates: Mapping[str, Mapping[int, CandidateMetrics]],
) -> dict[str, dict[str, object]]:
    """Summarize mandatory evidence and acceptance eligibility per lane."""

    result: dict[str, dict[str, object]] = {}
    mandatory = set(MANDATORY_MULTIPLICITIES)
    for lane in LANES:
        lane_records = candidates.get(lane)
        if lane_records is None or not mandatory <= set(lane_records):
            raise AcceptanceError(
                f"lane {lane} does not cover every mandatory multiplicity"
            )
        records = tuple(lane_records[total] for total in MANDATORY_MULTIPLICITIES)
        if any(
            record.lane != lane
            or record.total_gluons != total
            or record.probe.execution_mode != lane
            for total, record in zip(MANDATORY_MULTIPLICITIES, records, strict=True)
        ):
            raise AcceptanceError(f"lane {lane} contains inconsistent evidence")
        values = tuple(record.probe.warm_median_seconds for record in records)
        if not all(_positive_finite(value) for value in values):
            raise AcceptanceError(f"lane {lane} has an invalid warm measurement")
        workload = _lane_workload_eligibility(lane, records)
        result[lane] = {
            **workload,
            "mandatory_helicity_coverage_count": {
                str(total): record.probe.helicity_coverage_count
                for total, record in zip(MANDATORY_MULTIPLICITIES, records, strict=True)
            },
            "warm_geometric_mean_seconds": math.exp(
                statistics.fmean(math.log(value) for value in values)
            ),
        }
    return result


def _lane_workload_eligibility(
    lane: str,
    records: Sequence[CandidateMetrics],
) -> dict[str, object]:
    """Apply the lane-specific helicity workload contract once."""

    if lane not in LANES or not records:
        raise AcceptanceError("lane workload evidence is invalid")
    coverage = tuple(record.probe.helicity_coverage_count for record in records)
    generation_specialized = all(count == 1 for count in coverage)
    complete_helicity_coverage = all(count > 1 for count in coverage)
    if lane == "recurrence":
        eligible = generation_specialized
        contract = "generation-specialized-known-nonzero"
        reason = (
            "eligible: one generation-selected known-nonzero helicity"
            if eligible
            else "ineligible: recurrence artifact is not generation-specialized"
        )
    else:
        eligible = complete_helicity_coverage
        contract = "complete-runtime-selector-fixed-helicity-query"
        reason = (
            "eligible: complete runtime-selector axis with one fixed query"
            if eligible
            else "ineligible: OTF artifact does not retain complete selector coverage"
        )
    return {
        "eligible": eligible,
        "reason": reason,
        "coverage_contract": contract,
        "generation_specialized": generation_specialized,
        "complete_helicity_coverage": complete_helicity_coverage,
    }


def evaluate_gates(
    winner: str,
    references: Mapping[int, ReferenceMetrics],
    candidates: Mapping[str, Mapping[int, CandidateMetrics]],
    *,
    multiplicities: Sequence[int] = MANDATORY_MULTIPLICITIES,
) -> tuple[GateResult, ...]:
    """Apply all ratios to the one global winner without lane mixing."""

    if winner not in LANES or winner not in candidates:
        raise AcceptanceError("global winning lane is unavailable")
    if not lane_eligibility(candidates)[winner]["eligible"]:
        raise AcceptanceError(
            f"lane {winner} does not satisfy its helicity workload contract"
        )
    return _lane_gate_results(winner, references, candidates[winner], multiplicities)


def _lane_gate_results(
    lane: str,
    references: Mapping[int, ReferenceMetrics],
    records: Mapping[int, CandidateMetrics],
    multiplicities: Sequence[int],
) -> tuple[GateResult, ...]:
    results: list[GateResult] = []
    for total_gluons in multiplicities:
        if total_gluons not in references or total_gluons not in records:
            raise AcceptanceError(
                f"paired winner/reference evidence is missing for N={total_gluons}"
            )
        results.append(
            _paired_gate_result(
                lane,
                references[total_gluons],
                records[total_gluons],
            )
        )
    return tuple(results)


def _paired_gate_result(
    lane: str,
    reference: ReferenceMetrics,
    candidate: CandidateMetrics,
) -> GateResult:
    if candidate.lane != lane or candidate.total_gluons != reference.total_gluons:
        raise AcceptanceError("paired acceptance evidence is inconsistent")
    warm_ratio = candidate.probe.warm_median_seconds / reference.warm_median_seconds
    rss_ratio = candidate.max_rss_kib / reference.max_rss_kib
    cold_ratio = candidate.cold_to_ready_seconds / reference.cold_to_ready_seconds
    if not all(
        _positive_finite(value) for value in (warm_ratio, rss_ratio, cold_ratio)
    ):
        raise AcceptanceError("paired acceptance ratio is invalid")
    return GateResult(
        total_gluons=reference.total_gluons,
        lane=lane,
        warm_ratio=warm_ratio,
        rss_ratio=rss_ratio,
        cold_ratio=cold_ratio,
        warm_passes=warm_ratio <= WARM_RATIO_LIMIT,
        rss_passes=rss_ratio <= RSS_RATIO_LIMIT,
        cold_passes=cold_ratio <= COLD_RATIO_LIMIT,
        numerical_passes=candidate.numerical_parity.passes,
    )


def _observed_mandatory_gate_viability(
    references: Mapping[int, ReferenceMetrics],
    candidates: Mapping[str, Mapping[int, CandidateMetrics]],
    *,
    multiplicities: Sequence[int],
) -> dict[str, object]:
    """Report whether one lane can still pass every completed mandatory row."""

    completed = tuple(multiplicities)
    if not completed or any(total not in references for total in completed):
        raise AcceptanceError("observed mandatory reference evidence is incomplete")
    lanes: dict[str, object] = {}
    viable_lanes: list[str] = []
    for lane in LANES:
        records = candidates.get(lane)
        if records is None or any(total not in records for total in completed):
            raise AcceptanceError(
                f"observed mandatory candidate evidence is incomplete for {lane}"
            )
        selected = tuple(records[total] for total in completed)
        if any(
            record.lane != lane
            or record.total_gluons != total
            or record.probe.execution_mode != lane
            for total, record in zip(completed, selected, strict=True)
        ):
            raise AcceptanceError(f"lane {lane} contains inconsistent evidence")
        workload = _lane_workload_eligibility(lane, selected)
        eligible = bool(workload["eligible"])
        gates = _lane_gate_results(lane, references, records, completed)
        viable = eligible and all(gate.passes for gate in gates)
        if viable:
            viable_lanes.append(lane)
        lanes[lane] = {
            **workload,
            "viable": viable,
            "gates": [_plain_gate(gate) for gate in gates],
        }
    return {
        "completed_multiplicities": list(completed),
        "lanes": lanes,
        "viable_lanes": viable_lanes,
    }


def optional_candidate_is_feasible(candidate: CandidateMetrics) -> bool:
    return (
        candidate.generation_seconds < OPTIONAL_COLD_LIMIT_SECONDS
        and candidate.first_warm_seconds < OPTIONAL_COLD_LIMIT_SECONDS
        and candidate.max_rss_kib <= int(MEMORY_LIMIT_GIB * 1024**2)
    )


def process_expression(total_gluons: int) -> str:
    return "g g > " + " ".join("g" for _ in range(total_gluons - 2))


def process_name(total_gluons: int) -> str:
    return f"gg_N{total_gluons}"


def _toml_helicity_map(helicities: Sequence[int]) -> str:
    if not helicities or any(value not in (-1, 1) for value in helicities):
        raise AcceptanceError("generation-selected helicity must contain only +/-1")
    return (
        "{"
        + ",".join(
            f"{index}={value}" for index, value in enumerate(helicities, start=1)
        )
        + "}"
    )


def candidate_generation_command(
    *,
    python: str,
    lane: str,
    total_gluons: int,
    helicities: Sequence[int],
    artifact: Path,
) -> tuple[str, ...]:
    if lane not in LANES:
        raise AcceptanceError(f"unknown candidate lane {lane}")
    if len(helicities) != total_gluons:
        raise AcceptanceError("candidate helicity multiplicity is inconsistent")
    common = (
        python,
        "-m",
        "pyamplicol",
        "generate",
        process_expression(total_gluons),
        str(artifact),
        "--name",
        process_name(total_gluons),
        "--model",
        "built-in-sm",
        "--color-accuracy",
        "full",
        "--color-contraction",
        "symmetric-group-fft",
        "--execution-mode",
        lane,
        "--workers",
        "1",
        "--mode",
        "error",
        "--no-emit-api-bundle",
        "--set",
        "generation.validation.enabled=false",
        "--set",
        "generation.validation.post_build_validation=false",
        "--set",
        "model.cache=false",
        "--set",
        "evaluator.batch_size=1",
        "--set",
        "evaluator.optimization.cores=1",
        "--set",
        "evaluator.jit.optimization_level=2",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    if lane == "on-the-fly":
        return common
    insertion = common.index("--no-emit-api-bundle") + 1
    return (
        *common[:insertion],
        "--set",
        f"process.selected_source_helicities={_toml_helicity_map(helicities)}",
        *common[insertion:],
    )


def _watchdog_command(
    command: Sequence[str],
    *,
    python: str,
    memory_limit_gib: float = MEMORY_LIMIT_GIB,
    report_json: Path | None = None,
) -> tuple[str, ...]:
    command_prefix = (
        python,
        str(WATCHDOG),
        "--limit-gib",
        f"{memory_limit_gib:g}",
    )
    if report_json is not None:
        command_prefix = (*command_prefix, "--report-json", str(report_json))
    return (*command_prefix, "--", *command)


def _read_watchdog_usage_kib(report_path: Path) -> tuple[int, int]:
    """Read exact RSS and enforced-guard peaks from one successful report."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        enforcement = payload["enforcement"]
        peak_rss_bytes = enforcement["peak_rss_bytes"]
        peak_guard_bytes = enforcement["peak_guard_bytes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise AcceptanceError(
            f"watchdog report is missing or malformed: {report_path}"
        ) from error
    if (
        payload.get("complete") is not True
        or payload.get("passes") is not True
        or not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes < 0
        or not isinstance(peak_guard_bytes, int)
        or isinstance(peak_guard_bytes, bool)
        or peak_guard_bytes < 0
    ):
        raise AcceptanceError(f"watchdog report is not successful: {report_path}")
    return (
        (peak_rss_bytes + 1023) // 1024,
        (peak_guard_bytes + 1023) // 1024,
    )


def _workspace_environment(python: str) -> dict[str, str]:
    cache_root = PERFORMANCE_ROOT / "cache"
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": str(PERFORMANCE_ROOT / "tmp"),
            "CARGO_HOME": str(cache_root / "cargo-home"),
            "CARGO_TARGET_DIR": str(cache_root / "cargo-target"),
            "PIP_CACHE_DIR": str(cache_root / "pip-cache"),
            "XDG_CACHE_HOME": str(cache_root / "xdg-cache"),
            "PYTHONPYCACHEPREFIX": str(cache_root / "python-cache"),
            "CARGO_NET_OFFLINE": "true",
            "PIP_NO_INDEX": "1",
            **{name: "1" for name in SINGLE_THREAD_ENVIRONMENT},
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SYMBOLICA_HIDE_BANNER": "1",
            "FFT_ACCEPTANCE_PYTHON": python,
        }
    )
    return environment


def _create_workspace_directories(run_root: Path) -> None:
    if run_root.exists():
        raise AcceptanceError(
            f"acceptance run already exists; choose another --run-id: {run_root}"
        )
    for path in (
        PERFORMANCE_ROOT / "tmp",
        PERFORMANCE_ROOT / "cache" / "cargo-home",
        PERFORMANCE_ROOT / "cache" / "cargo-target",
        PERFORMANCE_ROOT / "cache" / "pip-cache",
        PERFORMANCE_ROOT / "cache" / "xdg-cache",
        PERFORMANCE_ROOT / "cache" / "python-cache",
        run_root / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _translate_darwin_reference_command(
    command: Sequence[str],
    *,
    python: str = sys.executable,
) -> tuple[tuple[str, ...], bool]:
    """Use the study's getrusage wrapper instead of unavailable Darwin sysctls."""

    original = tuple(str(item) for item in command)
    if original[:2] == ("/usr/bin/time", "-l"):
        payload = original[2:]
        if not payload:
            raise AcceptanceError("Darwin time wrapper has no payload")
        return (python, str(SCALING_STUDY_DRIVER), "_time-rss", *payload), True
    if not original or original[0] != "/usr/bin/timeout":
        return original, False
    if (
        len(original) < 10
        or original[1:3] != ("--signal=TERM", "--kill-after=5s")
        or not original[3].endswith("s")
        or original[4] != "/usr/bin/time"
        or original[5] != f"--format={REFERENCE_RSS_MARKER} %M"
        or original[6] != "/usr/bin/prlimit"
        or not original[7].startswith("--as=")
        or original[8] != "--"
    ):
        raise AcceptanceError("unrecognized GNU reference measurement wrapper")
    try:
        timeout_value = float(original[3][:-1])
        memory_bytes = int(original[7].partition("=")[2])
    except ValueError as error:
        raise AcceptanceError("invalid GNU reference measurement bound") from error
    if not _positive_finite(timeout_value) or memory_bytes < 1:
        raise AcceptanceError("invalid GNU reference measurement bound")
    payload = original[9:]
    if not payload:
        raise AcceptanceError("GNU reference measurement wrapper has no payload")
    return (python, str(SCALING_STUDY_DRIVER), "_time-rss", *payload), True


def _resolve_gnu_tool(name: str, fallback: str) -> str:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    if Path(fallback).is_file():
        return fallback
    raise AcceptanceError(
        f"required GNU utility is unavailable: {name} "
        f"(also checked {fallback})"
    )


def _translate_gnu_reference_command(
    command: Sequence[str],
    *,
    python: str = sys.executable,
) -> tuple[tuple[str, ...], bool]:
    """Measure Linux reference commands with getrusage and resolved wrappers."""

    original = tuple(str(item) for item in command)
    if original[:2] == ("/usr/bin/time", "-l"):
        payload = original[2:]
        if not payload:
            raise AcceptanceError("time wrapper has no payload")
        return (python, str(SCALING_STUDY_DRIVER), "_time-rss", *payload), True
    if not original or original[0] != "/usr/bin/timeout":
        return original, False
    if (
        len(original) < 10
        or original[1:3] != ("--signal=TERM", "--kill-after=5s")
        or not original[3].endswith("s")
        or original[4] != "/usr/bin/time"
        or original[5] != f"--format={REFERENCE_RSS_MARKER} %M"
        or original[6] != "/usr/bin/prlimit"
        or not original[7].startswith("--as=")
        or original[8] != "--"
    ):
        raise AcceptanceError("unrecognized GNU reference measurement wrapper")
    try:
        timeout_value = float(original[3][:-1])
        memory_bytes = int(original[7].partition("=")[2])
    except ValueError as error:
        raise AcceptanceError("invalid GNU reference measurement bound") from error
    if not _positive_finite(timeout_value) or memory_bytes < 1:
        raise AcceptanceError("invalid GNU reference measurement bound")
    return (
        python,
        str(SCALING_STUDY_DRIVER),
        "_time-rss",
        _resolve_gnu_tool("timeout", original[0]),
        *original[1:4],
        _resolve_gnu_tool("prlimit", original[6]),
        *original[7:],
    ), True


def _synthesize_reference_rss_marker(
    completed: subprocess.CompletedProcess[str],
) -> subprocess.CompletedProcess[str]:
    """Copy the getrusage wrapper's exact FFT RSS into the reference protocol."""

    matches: list[int] = []
    for line in completed.stderr.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == DARWIN_RSS_MARKER:
            matches.append(int(fields[1]))
    if len(matches) != 1 or matches[0] < 1:
        raise AcceptanceError("Darwin getrusage wrapper did not report exact RSS")
    stderr = completed.stderr.rstrip("\n")
    return subprocess.CompletedProcess(
        args=completed.args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=f"{stderr}\n{REFERENCE_RSS_MARKER} {matches[0]}\n",
    )


def _parse_translated_reference_child_rss_kib(stderr: str) -> int:
    markers: dict[str, list[int]] = {
        DARWIN_RSS_MARKER: [],
        REFERENCE_RSS_MARKER: [],
    }
    for line in stderr.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in markers:
            try:
                markers[fields[0]].append(int(fields[1]))
            except ValueError as error:
                raise AcceptanceError(
                    "translated Reference RSS marker is malformed"
                ) from error
    fft_values = markers[DARWIN_RSS_MARKER]
    benchmark_values = markers[REFERENCE_RSS_MARKER]
    if (
        len(fft_values) != 1
        or len(benchmark_values) != 1
        or fft_values[0] < 1
        or fft_values != benchmark_values
    ):
        raise AcceptanceError(
            "translated Reference command lacks one matching exact child RSS marker"
        )
    return fft_values[0]


def _formal_reference_evaluator_rss_kib(
    driver_child_rss_kib: int,
    direct_command_watchdog_rss_kib: int,
    translated_child_rss_kib: int,
) -> int:
    values = (
        driver_child_rss_kib,
        direct_command_watchdog_rss_kib,
        translated_child_rss_kib,
    )
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        )
        or driver_child_rss_kib < 1
    ):
        raise AcceptanceError("Reference evaluator RSS evidence is invalid")
    # The formal ratio compares the two fresh evaluator processes.  Compiler
    # and wrapper peaks remain watchdog/cap telemetry, not part of this value.
    return driver_child_rss_kib


def _run_watched(
    command: Sequence[str],
    *,
    python: str,
    environment: Mapping[str, str],
    timeout_seconds: float,
    log_path: Path,
    memory_limit_gib: float = MEMORY_LIMIT_GIB,
    watchdog_report_path: Path | None = None,
    normalize_completed: (
        Callable[[subprocess.CompletedProcess[str]], subprocess.CompletedProcess[str]]
        | None
    ) = None,
) -> WatchedCompletedProcess:
    """Run one watchdog and give it time to reap its separate-session child."""

    if not _positive_finite(timeout_seconds):
        raise AcceptanceError("watched command timeout must be positive")
    if not _positive_finite(memory_limit_gib):
        raise AcceptanceError("watched command memory limit must be positive")
    if watchdog_report_path is not None:
        watchdog_report_path.parent.mkdir(parents=True, exist_ok=True)
        if watchdog_report_path.exists():
            raise AcceptanceError(
                f"watchdog report already exists: {watchdog_report_path}"
            )
    watched_command = _watchdog_command(
        command,
        python=python,
        memory_limit_gib=memory_limit_gib,
        report_json=watchdog_report_path,
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        watched_command,
        cwd=ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = False
    timeout_cleanup = "not-required"
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        timeout_cleanup = "sigterm-complete"
        try:
            stdout, stderr = process.communicate(timeout=WATCHDOG_CLEANUP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            timeout_cleanup = "sigkill-fallback"
    elapsed_seconds = time.perf_counter() - started
    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=watched_command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    normalization_error: Exception | None = None
    if normalize_completed is not None and not timed_out and completed.returncode == 0:
        try:
            completed = normalize_completed(completed)
        except (
            Exception
        ) as error:  # Preserve the raw child evidence before surfacing it.
            normalization_error = error
    peak_guard_kib: int | None = None
    peak_rss_kib: int | None = None
    watchdog_report_error: Exception | None = None
    if watchdog_report_path is not None and not timed_out and completed.returncode == 0:
        try:
            peak_rss_kib, peak_guard_kib = _read_watchdog_usage_kib(
                watchdog_report_path
            )
        except Exception as error:
            watchdog_report_error = error
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_started = time.perf_counter()
    log_path.write_text(
        json.dumps(
            {
                "command": list(command),
                "watchdog_command": list(watched_command),
                "elapsed_seconds": elapsed_seconds,
                "elapsed_excludes_log_write": True,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": timed_out,
                "timeout_cleanup": timeout_cleanup,
                "normalization_error": (
                    None
                    if normalization_error is None
                    else f"{type(normalization_error).__name__}: {normalization_error}"
                ),
                "watchdog_report": (
                    None if watchdog_report_path is None else str(watchdog_report_path)
                ),
                "watchdog_peak_guard_kib": peak_guard_kib,
                "watchdog_peak_rss_kib": peak_rss_kib,
                "watchdog_report_error": (
                    None
                    if watchdog_report_error is None
                    else f"{type(watchdog_report_error).__name__}: "
                    f"{watchdog_report_error}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log_write_seconds = time.perf_counter() - log_started
    result = WatchedCompletedProcess(
        completed.args,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        elapsed_seconds=elapsed_seconds,
        log_write_seconds=log_write_seconds,
        peak_guard_kib=peak_guard_kib,
        peak_rss_kib=peak_rss_kib,
        timed_out=timed_out,
        timeout_cleanup=timeout_cleanup,
    )
    if timed_out:
        raise AcceptanceError(
            f"command timed out after {timeout_seconds:g}s "
            f"(watchdog cleanup: {timeout_cleanup}): {shlex.join(command)}; "
            f"see {log_path}"
        )
    if result.returncode != 0:
        raise AcceptanceError(
            f"command failed with status {result.returncode}: "
            f"{shlex.join(command)}; see {log_path}"
        )
    if normalization_error is not None:
        raise normalization_error
    if watchdog_report_error is not None:
        raise watchdog_report_error
    return result


def _load_reference_module() -> ModuleType:
    if not REFERENCE_DRIVER.is_file():
        raise AcceptanceError(f"reference driver is missing: {REFERENCE_DRIVER}")
    name = "_pyamplicol_fft_reference_benchmark"
    specification = importlib.util.spec_from_file_location(name, REFERENCE_DRIVER)
    if specification is None or specification.loader is None:
        raise AcceptanceError("cannot load bundled AmpliGluonTrace benchmark")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@contextmanager
def _arguments(arguments: Sequence[str]):
    previous = sys.argv
    sys.argv = [str(REFERENCE_DRIVER), *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _reference_arguments(
    reference: ModuleType,
    *,
    total_gluons: int,
    build_dir: Path,
    fc: str,
    target_seconds: float,
    timeout_seconds: float,
    memory_limit_gib: float = MEMORY_LIMIT_GIB,
    repetition_quantum: int | None = None,
    warm_sample_count: int = WARM_SAMPLE_COUNT,
) -> argparse.Namespace:
    if (
        isinstance(warm_sample_count, bool)
        or not isinstance(warm_sample_count, int)
        or warm_sample_count < 1
    ):
        raise AcceptanceError("reference warm_sample_count must be positive")
    argv = (
        "--min-gluons",
        str(total_gluons),
        "--max-gluons",
        str(total_gluons),
        "--points",
        str(POINT_COUNT),
        "--seed",
        str(BASE_SEED),
        "--mhv-samples",
        "1",
        "--non-mhv-samples",
        "1",
        "--backend",
        REFERENCE_BACKEND,
        "--target-seconds",
        f"{target_seconds:.17g}",
        "--batches",
        str(warm_sample_count),
        "--initialization-runs",
        "1",
        "--skip-initialization-preflight",
        "--max-memory-gib",
        f"{memory_limit_gib:g}",
        "--backend-timeout",
        f"{timeout_seconds:g}",
        "--build-dir",
        str(build_dir),
        "--fc",
        fc,
    )
    if repetition_quantum is not None:
        argv = (*argv, "--repetition-quantum", str(repetition_quantum))
    with _arguments(argv):
        parsed = reference.parse_arguments()
    reference.validate_arguments(parsed)
    return parsed


def _reference_representative_warm_samples(
    cell_timings: Mapping[tuple[int, int, int], float],
    *,
    warm_sample_count: int = WARM_SAMPLE_COUNT,
) -> tuple[float, ...]:
    """Validate DefaultBG's one representative event/configuration per batch."""

    expected = {
        (batch, REPRESENTATIVE_POINT, 1)
        for batch in range(1, warm_sample_count + 1)
    }
    if set(cell_timings) != expected:
        raise AcceptanceError(
            "reference must report one point-1 cell in each requested batch"
        )
    values = tuple(
        cell_timings[(batch, REPRESENTATIVE_POINT, 1)]
        for batch in range(1, warm_sample_count + 1)
    )
    if not all(_positive_finite(value) for value in values):
        raise AcceptanceError("reference representative timing cells are not positive")
    return values


def _bounded_reference_cold_timeout(
    effective_timeout: float,
    cold_limit_seconds: float | None,
    cold_started: float,
) -> float:
    if cold_limit_seconds is None:
        return effective_timeout
    remaining_cold = cold_limit_seconds - (time.perf_counter() - cold_started)
    if remaining_cold <= 0.0:
        raise ReferenceColdLimitError(
            "reference campaign exhausted its aggregate cold deadline"
        )
    return min(effective_timeout, remaining_cold)


def _is_reference_backend_build_command(
    command: Sequence[str],
    *,
    build_dir: Path,
    build_phase_active: bool,
) -> bool:
    """Identify compiler/linker commands belonging only to AmpliGluonTrace."""

    if not build_phase_active:
        return False
    backend_dir = (build_dir / REFERENCE_BUILD_PROGRAM).resolve()
    backend_prefix = f"{backend_dir}{os.sep}"
    include_prefixes = (
        backend_prefix,
        f"-I{backend_prefix}",
        f"-J{backend_prefix}",
    )
    return any(
        str(token) == str(backend_dir) or str(token).startswith(include_prefixes)
        for token in command
    )


def _run_reference(
    *,
    reference: ModuleType,
    total_gluons: int,
    run_root: Path,
    python: str,
    environment: Mapping[str, str],
    fc: str,
    target_seconds: float,
    timeout_seconds: float,
    cold_limit_seconds: float | None = None,
    memory_limit_gib: float = MEMORY_LIMIT_GIB,
    repetition_quantum: int | None = None,
    warm_sample_count: int = WARM_SAMPLE_COUNT,
    sum_helicities: bool = False,
) -> _ReferenceRun:
    reference_root = run_root / "reference" / f"N{total_gluons}"
    build_dir = reference_root / "build"
    if build_dir.exists():
        raise AcceptanceError("reference build directory is not clean")
    effective_repetition_quantum = 1 if sum_helicities else repetition_quantum
    arguments = _reference_arguments(
        reference,
        total_gluons=total_gluons,
        build_dir=build_dir,
        fc=fc,
        target_seconds=target_seconds,
        timeout_seconds=timeout_seconds,
        memory_limit_gib=memory_limit_gib,
        repetition_quantum=effective_repetition_quantum,
        warm_sample_count=warm_sample_count,
    )
    command_index = 0
    command_log_write_seconds = 0.0
    watchdog_max_rss_kib = 0
    direct_command_watchdog_max_rss_kib = 0
    translated_child_max_rss_kib = 0
    watchdog_max_peak_guard_kib = 0
    clean_build_seconds = 0.0
    clean_build_command_count = 0
    build_phase_active = False
    original_run_command = reference.run_command
    campaign_timeout_seconds = timeout_seconds
    cold_started = time.perf_counter()

    def watched_reference_command(
        command: Sequence[str],
        description: str,
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal command_index, command_log_write_seconds
        nonlocal watchdog_max_peak_guard_kib, watchdog_max_rss_kib
        nonlocal direct_command_watchdog_max_rss_kib
        nonlocal translated_child_max_rss_kib
        nonlocal clean_build_command_count, clean_build_seconds
        command_index += 1
        safe_description = re.sub(r"[^A-Za-z0-9._-]+", "-", description).strip("-")
        translated = False
        command_to_run = tuple(str(item) for item in command)
        is_clean_backend_build = _is_reference_backend_build_command(
            command_to_run,
            build_dir=build_dir,
            build_phase_active=build_phase_active,
        )
        if sys.platform == "darwin":
            command_to_run, translated = _translate_darwin_reference_command(
                command_to_run,
                python=python,
            )
        else:
            command_to_run, translated = _translate_gnu_reference_command(
                command_to_run,
                python=python,
            )
        effective_timeout = (
            campaign_timeout_seconds + 30.0
            if timeout_seconds is None
            else timeout_seconds + 30.0
        )
        if translated and timeout_seconds is not None:
            # run_driver already supplies backend_timeout + its cleanup grace.
            effective_timeout = timeout_seconds
        effective_timeout = _bounded_reference_cold_timeout(
            effective_timeout, cold_limit_seconds, cold_started
        )
        log_path = reference_root / "logs" / f"{command_index:03d}-{safe_description}"
        completed = _run_watched(
            command_to_run,
            python=python,
            environment=environment,
            timeout_seconds=effective_timeout,
            log_path=log_path.with_suffix(".json"),
            memory_limit_gib=memory_limit_gib,
            watchdog_report_path=log_path.with_suffix(".watchdog.json"),
            normalize_completed=(
                _synthesize_reference_rss_marker if translated else None
            ),
        )
        if completed.peak_rss_kib is None or completed.peak_guard_kib is None:
            raise AcceptanceError("reference command has no watchdog RSS evidence")
        watchdog_max_rss_kib = max(watchdog_max_rss_kib, completed.peak_rss_kib)
        if translated:
            translated_child_max_rss_kib = max(
                translated_child_max_rss_kib,
                _parse_translated_reference_child_rss_kib(completed.stderr),
            )
        else:
            direct_command_watchdog_max_rss_kib = max(
                direct_command_watchdog_max_rss_kib,
                completed.peak_rss_kib,
            )
        watchdog_max_peak_guard_kib = max(
            watchdog_max_peak_guard_kib, completed.peak_guard_kib
        )
        if is_clean_backend_build:
            clean_build_seconds += completed.elapsed_seconds
            clean_build_command_count += 1
        command_log_write_seconds += completed.log_write_seconds
        return completed

    reference.run_command = watched_reference_command
    try:
        compiler = shlex.split(arguments.fc)
        flags = shlex.split(arguments.fflags)
        if not compiler:
            raise AcceptanceError("reference Fortran compiler is empty")
        compiler_identity = reference.compiler_version(compiler).strip()
        build_phase_active = True
        try:
            rambo, proxy, executables = reference.build_executables(
                arguments,
                compiler,
                compiler_identity,
                flags,
            )
        finally:
            build_phase_active = False
        reference_executable = executables.get(REFERENCE_BACKEND)
        expected_build_dir = (build_dir / REFERENCE_BUILD_PROGRAM).resolve()
        if (
            reference_executable is None
            or expected_build_dir not in reference_executable.resolve().parents
            or clean_build_command_count < 1
            or not _positive_finite(clean_build_seconds)
        ):
            raise AcceptanceError(
                "reference clean backend build boundary is incomplete"
            )
        events, _, exhaustive_helicities = reference.generate_events(
            rambo,
            total_gluons,
            arguments,
        )
        weights, _, _, _ = reference.run_helicity_proxy(
            proxy,
            events,
            exhaustive_helicities,
            arguments,
        )
        _, _, selected = reference.select_helicity_samples(
            events,
            exhaustive_helicities,
            weights,
            total_gluons,
            arguments,
        )
        representative = selected[0]
        if representative.path == "zero" or representative.proxy_weight <= 0.0:
            raise AcceptanceError("reference selected an analytically zero helicity")
        uniform_events: list[Path] = []
        event_dir = reference_root / "selected-helicity-events"
        for point, source_event in enumerate(events, start=1):
            destination = event_dir / f"N{total_gluons}-point-{point:02d}.event"
            reference.write_single_helicity_event(
                source_event,
                destination,
                representative.helicities,
            )
            uniform_events.append(destination)
        setup_to_driver_seconds = (
            time.perf_counter() - cold_started - command_log_write_seconds
        )
        if not _positive_finite(setup_to_driver_seconds):
            raise AcceptanceError("reference setup timing is not positive")
        if sum_helicities:
            run = reference.run_driver(
                REFERENCE_BACKEND,
                reference_executable,
                events,
                arguments,
                False,
                sum_helicities=True,
            )
        else:
            run = reference.run_driver(
                REFERENCE_BACKEND,
                reference_executable,
                uniform_events,
                arguments,
                False,
            )
    finally:
        reference.run_command = original_run_command

    batch_values = _reference_representative_warm_samples(
        run.cell_timings,
        warm_sample_count=warm_sample_count,
    )
    warm_repetitions: tuple[int, ...] = ()
    warm_calibration_seconds: tuple[float, ...] = ()
    if effective_repetition_quantum is not None:
        expected_repetition_keys = {
            (batch, REPRESENTATIVE_POINT, 1)
            for batch in range(1, warm_sample_count + 1)
        }
        if (
            run.cell_repetitions is None
            or set(run.cell_repetitions) != expected_repetition_keys
            or any(
                value < effective_repetition_quantum
                or value % effective_repetition_quantum != 0
                for value in run.cell_repetitions.values()
            )
        ):
            raise AcceptanceError(
                "reference did not use the requested repetition quantum"
            )
        warm_repetitions = tuple(
            run.cell_repetitions[(batch, REPRESENTATIVE_POINT, 1)]
            for batch in range(1, warm_sample_count + 1)
        )
        if len(set(warm_repetitions)) != 1:
            raise AcceptanceError("reference changed repetitions after calibration")
        expected_calibration_keys = {(REPRESENTATIVE_POINT, 1)}
        if (
            run.cell_calibration_seconds is None
            or set(run.cell_calibration_seconds) != expected_calibration_keys
            or any(
                value < target_seconds
                for value in run.cell_calibration_seconds.values()
            )
        ):
            raise AcceptanceError(
                "reference repetition calibration did not reach its timing target"
            )
        warm_calibration_seconds = tuple(run.cell_calibration_seconds.values())
    if (
        run.first_helicity_sweep is None
        or not _positive_finite(run.first_helicity_sweep)
        or run.peak_rss_kib is None
        or run.peak_rss_kib < 1
    ):
        raise AcceptanceError("reference cold/RSS metrics are incomplete")
    if sum_helicities:
        exhaustive_keys = set(exhaustive_helicities)
        if set(run.matrix_elements) != exhaustive_keys or any(
            not math.isfinite(value) or value < 0.0
            for value in run.matrix_elements.values()
        ):
            raise AcceptanceError(
                "reference did not return the exhaustive helicity grid"
            )
        helicity_coverage_count = len(
            [key for key in exhaustive_keys if key[0] == REPRESENTATIVE_POINT]
        )
        active_helicity_count = sum(
            not reference.is_analytic_zero_helicity(helicities)
            for (point, _), helicities in exhaustive_helicities.items()
            if point == REPRESENTATIVE_POINT
        )
        if (
            helicity_coverage_count < 1
            or active_helicity_count < 1
            or active_helicity_count > helicity_coverage_count
        ):
            raise AcceptanceError("reference exhaustive helicity census is invalid")
        matrix_elements = tuple(
            math.fsum(
                run.matrix_elements[key]
                for key in sorted(exhaustive_keys)
                if key[0] == point
            )
            for point in range(1, POINT_COUNT + 1)
        )
        if any(not _positive_finite(value) for value in matrix_elements):
            raise AcceptanceError("reference helicity sum is zero or non-finite")
    else:
        expected_matrix_element_keys = {
            (point, 1) for point in range(1, POINT_COUNT + 1)
        }
        if set(run.matrix_elements) != expected_matrix_element_keys or any(
            not _positive_finite(value) for value in run.matrix_elements.values()
        ):
            raise AcceptanceError(
                "reference did not confirm the selected helicity at all 10 points"
            )
        matrix_elements = tuple(
            run.matrix_elements[(point, 1)] for point in range(1, POINT_COUNT + 1)
        )
        helicity_coverage_count = 1
        active_helicity_count = 1
    if watchdog_max_peak_guard_kib > int(memory_limit_gib * 1024**2):
        raise AcceptanceError("reference exceeded the watchdog peak-guard bound")
    metrics = ReferenceMetrics(
        total_gluons=total_gluons,
        generator_seed=generator_seed(total_gluons),
        backend=REFERENCE_BACKEND,
        clean_build_scope=REFERENCE_CLEAN_BUILD_SCOPE,
        clean_build_command_count=clean_build_command_count,
        clean_build_seconds=clean_build_seconds,
        setup_to_driver_seconds=setup_to_driver_seconds,
        initialization_seconds=run.initialization,
        first_pass_seconds=run.first_helicity_sweep,
        warm_samples_seconds=batch_values,
        warm_median_seconds=statistics.median(batch_values),
        max_rss_kib=_formal_reference_evaluator_rss_kib(
            run.peak_rss_kib,
            direct_command_watchdog_max_rss_kib,
            translated_child_max_rss_kib,
        ),
        selected_helicity=tuple(representative.helicities),
        selected_path=representative.path,
        event_paths=tuple(str(path) for path in uniform_events),
        matrix_elements=matrix_elements,
        driver_max_rss_kib=run.peak_rss_kib,
        watchdog_max_rss_kib=watchdog_max_rss_kib,
        direct_command_watchdog_max_rss_kib=(
            direct_command_watchdog_max_rss_kib or None
        ),
        translated_child_max_rss_kib=translated_child_max_rss_kib or None,
        watchdog_max_peak_guard_kib=watchdog_max_peak_guard_kib,
        warm_repetitions=warm_repetitions,
        warm_repetition_quantum=effective_repetition_quantum,
        warm_calibration_seconds=warm_calibration_seconds,
        helicity_workload="sum" if sum_helicities else "fixed",
        helicity_coverage_count=helicity_coverage_count,
        timed_helicity_count=active_helicity_count,
        active_helicity_count=active_helicity_count,
        exhaustive_event_paths=(
            tuple(str(path) for path in events) if sum_helicities else ()
        ),
    )
    if (
        cold_limit_seconds is not None
        and metrics.setup_to_ready_seconds > cold_limit_seconds
    ):
        raise ReferenceColdLimitError(
            "reference setup/initialization/first pass exceeded its cold deadline"
        )
    return _ReferenceRun(metrics=metrics, event_paths=tuple(uniform_events))


def _build_probe(
    *,
    run_root: Path,
    python: str,
    cxx: str,
    environment: Mapping[str, str],
    memory_limit_gib: float = MEMORY_LIMIT_GIB,
) -> tuple[Path, dict[str, Any]]:
    sdk = _run_watched(
        (python, "-m", "pyamplicol._sdk.config", "--json"),
        python=python,
        environment=environment,
        timeout_seconds=60.0,
        log_path=run_root / "logs" / "sdk-config.json",
        memory_limit_gib=memory_limit_gib,
    )
    try:
        sdk_info = json.loads(sdk.stdout)
    except json.JSONDecodeError as error:
        raise AcceptanceError("Rusticol SDK query returned invalid JSON") from error
    required = {"cflags", "link_flags", "target", "package_version"}
    if not isinstance(sdk_info, dict) or not required <= set(sdk_info):
        raise AcceptanceError("Rusticol SDK query is incomplete")
    executable = run_root / "candidate-probe" / "fft-gluon-candidate-probe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(cxx)
    if not compiler:
        raise AcceptanceError("C++ compiler command is empty")
    command = (
        *compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        *map(str, sdk_info["cflags"]),
        str(PROBE_SOURCE),
        "-o",
        str(executable),
        *map(str, sdk_info["link_flags"]),
        *_candidate_probe_executable_link_flags(sys.platform),
    )
    _run_watched(
        command,
        python=python,
        environment=environment,
        timeout_seconds=600.0,
        log_path=run_root / "logs" / "candidate-probe-build.json",
        memory_limit_gib=memory_limit_gib,
    )
    return executable, sdk_info


def _candidate_probe_executable_link_flags(platform_name: str) -> tuple[str, ...]:
    """Discard SDK code that is unreachable from this standalone probe."""

    if platform_name == "darwin":
        # Apple ld keeps exported symbols as dead-strip roots.  A standalone
        # benchmark executable has no plugin ABI, so retaining the static
        # archive's thousands of Rust globals defeats ordinary link-time GC.
        return ("-Wl,-dead_strip", "-Wl,-no_exported_symbols")
    return ()


def _candidate_probe_command(
    *,
    probe: Path,
    artifact: Path,
    total_gluons: int,
    target_seconds: float,
    event_paths: Sequence[Path],
    first_ready_only: bool,
) -> tuple[str, ...]:
    return (
        str(probe),
        str(artifact),
        process_name(total_gluons),
        "--target-seconds",
        f"{target_seconds:.17g}",
        "--samples",
        str(WARM_SAMPLE_COUNT),
        *(("--first-ready-only",) if first_ready_only else ()),
        *(str(path) for path in event_paths),
    )


def _validate_candidate_probe_identity(
    metrics: CandidateProbeMetrics | CandidateFirstReadyMetrics,
    *,
    lane: str,
    total_gluons: int,
    expected_helicities: Sequence[int],
) -> None:
    if metrics.execution_mode != lane:
        raise AcceptanceError("candidate probe execution mode differs from its lane")
    if metrics.process != process_name(total_gluons):
        raise AcceptanceError("candidate probe process differs from its paired process")
    if lane == "recurrence" and metrics.helicity_coverage_count != 1:
        raise AcceptanceError(
            "recurrence candidate did not preserve generation helicity specialization"
        )
    if lane == "on-the-fly" and metrics.helicity_coverage_count <= 1:
        raise AcceptanceError(
            "on-the-fly candidate did not preserve complete runtime selector coverage"
        )
    expected_helicity_id = "h:" + ",".join(
        f"{helicity:+d}" for helicity in expected_helicities
    )
    if metrics.selected_helicity_id != expected_helicity_id:
        raise AcceptanceError(
            "candidate probe helicity differs from the generation-selected "
            "known-nonzero helicity"
        )


def _run_candidate(
    *,
    lane: str,
    total_gluons: int,
    reference: _ReferenceRun,
    probe: Path,
    run_root: Path,
    python: str,
    environment: Mapping[str, str],
    target_seconds: float,
    timeout_seconds: float,
    continuation_stage_limit_seconds: float | None = None,
) -> CandidateMetrics:
    candidate_root = run_root / "candidate" / lane / f"N{total_gluons}"
    artifact = candidate_root / "artifact"
    candidate_environment = dict(environment)
    candidate_environment["PYAMPLICOL_CACHE_DIR"] = str(
        candidate_root / "disabled-model-cache"
    )
    command = candidate_generation_command(
        python=python,
        lane=lane,
        total_gluons=total_gluons,
        helicities=reference.metrics.selected_helicity,
        artifact=artifact,
    )
    lane_wall_started = time.perf_counter()
    generation_started = lane_wall_started
    generation = _run_watched(
        command,
        python=python,
        environment=candidate_environment,
        timeout_seconds=(
            min(timeout_seconds, continuation_stage_limit_seconds)
            if continuation_stage_limit_seconds is not None
            else timeout_seconds
        ),
        log_path=candidate_root / "generation.json",
    )
    generation_seconds = (
        time.perf_counter() - generation_started - generation.log_write_seconds
    )
    if not _positive_finite(generation_seconds):
        raise AcceptanceError("candidate generation timing is not positive")
    if (
        continuation_stage_limit_seconds is not None
        and generation_seconds >= continuation_stage_limit_seconds
    ):
        raise AcceptanceError(
            "candidate generation did not remain below the 15 minute continuation cap"
        )

    cold_metrics: CandidateFirstReadyMetrics | None = None
    if continuation_stage_limit_seconds is not None:
        first_ready = _run_watched(
            _candidate_probe_command(
                probe=probe,
                artifact=artifact,
                total_gluons=total_gluons,
                target_seconds=target_seconds,
                event_paths=reference.event_paths,
                first_ready_only=True,
            ),
            python=python,
            environment=candidate_environment,
            # The continuation contract limits the reported first warm-up,
            # independently of artifact load.  Give the fresh process the
            # enclosing lane deadline so a valid sub-15-minute warm-up is not
            # killed merely because loading preceded it.
            timeout_seconds=timeout_seconds,
            log_path=candidate_root / "probe-first-ready.json",
        )
        cold_metrics = parse_candidate_first_ready_output(first_ready.stdout)
        _validate_candidate_probe_identity(
            cold_metrics,
            lane=lane,
            total_gluons=total_gluons,
            expected_helicities=reference.metrics.selected_helicity,
        )
        if cold_metrics.first_warm_seconds >= continuation_stage_limit_seconds:
            raise AcceptanceError(
                "candidate first warm-up did not remain below the 15 minute "
                "continuation cap"
            )
        if cold_metrics.max_rss_kib > int(MEMORY_LIMIT_GIB * 1024**2):
            raise AcceptanceError(
                "candidate first-ready process exceeded the 30 GiB RSS cap"
            )

    if continuation_stage_limit_seconds is None:
        remaining_warm = timeout_seconds - (time.perf_counter() - lane_wall_started)
        if remaining_warm <= 0.0:
            raise AcceptanceError("candidate generation exhausted its lane timeout")
    else:
        # Optional feasibility has already been decided by the fresh cold-only
        # process.  The warmed campaign gets an independent bounded window.
        remaining_warm = timeout_seconds
    completed = _run_watched(
        _candidate_probe_command(
            probe=probe,
            artifact=artifact,
            total_gluons=total_gluons,
            target_seconds=target_seconds,
            event_paths=reference.event_paths,
            first_ready_only=False,
        ),
        python=python,
        environment=candidate_environment,
        timeout_seconds=remaining_warm,
        log_path=(
            candidate_root / "probe-warm.json"
            if cold_metrics is not None
            else candidate_root / "probe.json"
        ),
    )
    probe_metrics = parse_candidate_probe_output(completed.stdout)
    _validate_candidate_probe_identity(
        probe_metrics,
        lane=lane,
        total_gluons=total_gluons,
        expected_helicities=reference.metrics.selected_helicity,
    )
    load_seconds = (
        cold_metrics.load_seconds
        if cold_metrics is not None
        else probe_metrics.load_seconds
    )
    first_warm_seconds = (
        cold_metrics.first_warm_seconds
        if cold_metrics is not None
        else probe_metrics.first_warm_seconds
    )
    max_rss_kib = max(
        probe_metrics.max_rss_kib,
        cold_metrics.max_rss_kib if cold_metrics is not None else 0,
    )
    numerical_parity = compare_candidate_to_reference(
        total_gluons=total_gluons,
        candidate_values=probe_metrics.point_values,
        reference_values=reference.metrics.matrix_elements,
    )
    return CandidateMetrics(
        lane=lane,
        total_gluons=total_gluons,
        generator_seed=generator_seed(total_gluons),
        generation_seconds=generation_seconds,
        load_seconds=load_seconds,
        first_warm_seconds=first_warm_seconds,
        max_rss_kib=max_rss_kib,
        probe=probe_metrics,
        numerical_parity=numerical_parity,
    )


def _cpu_policy(platform_name: str) -> dict[str, object]:
    thread_environment = {name: "1" for name in SINGLE_THREAD_ENVIRONMENT}
    if platform_name == "darwin":
        return {
            "requested_cpu_cores": 1,
            "method": "single-thread-environment",
            "thread_environment": thread_environment,
            "affinity_available": False,
            "affinity_enforced": False,
            "cpu": None,
        }
    if platform_name.startswith("linux"):
        return {
            "requested_cpu_cores": 1,
            "method": "sched-affinity-and-single-thread-environment",
            "thread_environment": thread_environment,
            "affinity_available": bool(
                hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
            ),
            "affinity_enforced": False,
            "cpu": None,
        }
    raise AcceptanceError(f"unsupported performance host platform {platform_name!r}")


def _enforce_one_core(platform_name: str) -> dict[str, object]:
    policy = _cpu_policy(platform_name)
    if platform_name == "darwin":
        return policy
    if not policy["affinity_available"]:
        raise AcceptanceError("Linux host does not expose sched affinity")
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise AcceptanceError("the process has no allowed CPU")
        core = allowed[0]
        os.sched_setaffinity(0, {core})
    except OSError as error:
        raise AcceptanceError("could not enforce Linux CPU affinity") from error
    return policy | {"affinity_enforced": True, "cpu": core}


def _plain_candidate(candidate: CandidateMetrics) -> dict[str, object]:
    payload = asdict(candidate)
    payload["cold_to_ready_seconds"] = candidate.cold_to_ready_seconds
    return payload


def _plain_reference(reference: ReferenceMetrics) -> dict[str, object]:
    payload = asdict(reference)
    payload["cold_to_ready_seconds"] = reference.cold_to_ready_seconds
    payload["setup_to_ready_seconds"] = reference.setup_to_ready_seconds
    return payload


def _plain_gate(gate: GateResult) -> dict[str, object]:
    return asdict(gate) | {"passes": gate.passes}


def _gate_report(gates: Sequence[GateResult]) -> dict[str, object]:
    """Serialize every completed gate and aggregate its acceptance result."""

    return {
        "gates": [_plain_gate(gate) for gate in gates],
        # Infeasible optional rows are skipped before producing a GateResult.
        # Every row that was measured is therefore acceptance-authoritative.
        "passes": all(gate.passes for gate in gates),
    }


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reference_git_output(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(REFERENCE_ROOT), *arguments),
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise AcceptanceError(
            f"cannot inspect FFT reference source: {error}"
        ) from error
    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip()
        raise AcceptanceError(
            "cannot inspect FFT reference source" + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _hash_identity_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _reference_source_identity() -> ReferenceSourceIdentity:
    """Hash every tracked or nonignored reference input at the pinned HEAD."""

    repository_root = Path(
        os.fsdecode(_reference_git_output("rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if repository_root != REFERENCE_ROOT.resolve():
        raise AcceptanceError(
            f"FFT reference source is not an independent repository: {REFERENCE_ROOT}"
        )
    revision = os.fsdecode(
        _reference_git_output("rev-parse", "--verify", "HEAD")
    ).strip()
    if revision != REFERENCE_REVISION:
        raise AcceptanceError(
            "FFT reference source is at the wrong revision: "
            f"expected {REFERENCE_REVISION}, found {revision}"
        )

    raw_paths = _reference_git_output(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"
    )
    paths = sorted(path for path in raw_paths.split(b"\0") if path)
    if not paths:
        raise AcceptanceError("FFT reference source contains no authenticated inputs")

    digest = hashlib.sha256()
    _hash_identity_field(digest, b"pyamplicol-fft-reference-content-v1")
    for raw_path in paths:
        relative = Path(os.fsdecode(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise AcceptanceError("FFT reference source reported an unsafe path")
        path = REFERENCE_ROOT / relative
        try:
            before = path.lstat()
            if stat.S_ISREG(before.st_mode):
                kind = b"file+x" if before.st_mode & 0o111 else b"file"
                content = path.read_bytes()
            elif stat.S_ISLNK(before.st_mode):
                kind = b"symlink"
                content = os.fsencode(os.readlink(path))
            else:
                raise AcceptanceError(
                    f"FFT reference input has unsupported file type: {relative}"
                )
            after = path.lstat()
        except OSError as error:
            raise AcceptanceError(
                f"cannot authenticate FFT reference input {relative}: {error}"
            ) from error
        before_identity = (
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ino,
        )
        after_identity = (
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        )
        if before_identity != after_identity:
            raise AcceptanceError(
                f"FFT reference input changed while being authenticated: {relative}"
            )
        _hash_identity_field(digest, raw_path)
        _hash_identity_field(digest, kind)
        _hash_identity_field(digest, content)
    return ReferenceSourceIdentity(revision, digest.hexdigest(), len(paths))


def _require_reference_source_identity(
    expected: ReferenceSourceIdentity | None = None,
) -> ReferenceSourceIdentity:
    current = _reference_source_identity()
    if expected is not None and current != expected:
        raise AcceptanceError(
            "FFT reference source identity changed during the campaign: "
            f"started at {expected.revision}/{expected.content_sha256}, now "
            f"{current.revision}/{current.content_sha256}"
        )
    return current


def _require_campaign_source_identity(
    expected: ReportSourceIdentity | None = None,
) -> ReportSourceIdentity:
    """Authenticate a clean HEAD before measuring or retaining any result."""

    try:
        current = require_eligible_report_source(ROOT)
    except ReportSourceIdentityError as error:
        raise AcceptanceError(
            f"FFT performance source is ineligible: {error}"
        ) from error
    if expected is not None and current != expected:
        raise AcceptanceError(
            "FFT performance source identity changed during the campaign: "
            f"started at {expected.revision}/{expected.tree}, now "
            f"{current.revision}/{current.tree}"
        )
    return current


def dry_run_plan(
    *,
    run_id: str,
    python: str,
    include_optional: bool,
    platform_name: str | None = None,
) -> dict[str, object]:
    host_platform = sys.platform if platform_name is None else platform_name
    cpu_policy = _cpu_policy(host_platform)
    multiplicities = (
        (*MANDATORY_MULTIPLICITIES, *OPTIONAL_MULTIPLICITIES)
        if include_optional
        else MANDATORY_MULTIPLICITIES
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "host_platform": host_platform,
        "run_root": str(PERFORMANCE_ROOT / "runs" / run_id),
        "reference_driver": str(REFERENCE_DRIVER),
        "reference": {
            "backend": REFERENCE_BACKEND,
            "source_revision": REFERENCE_REVISION,
            "source_identity": (
                "pinned-head-plus-tracked-and-nonignored-content-sha256"
            ),
            "color_contraction": "fft",
            "kinematics": "default-bg",
            "clean_build_per_multiplicity": True,
            "clean_build_scope": REFERENCE_CLEAN_BUILD_SCOPE,
            "clean_build_program": REFERENCE_BUILD_PROGRAM,
            "clean_build_timing": (
                "sum of watched compiler/linker wall times for the "
                "AmpliGluonTrace backend; RAMBO and helicity-proxy builds excluded"
            ),
            "initialization_and_first_pass_separate": True,
            "formal_cold_to_ready_metric": (
                "clean AmpliGluonTrace backend build plus initialization plus "
                "first complete pass"
            ),
            "scaling_setup_to_ready_metric": (
                "all setup through driver plus initialization plus first complete pass"
            ),
            "process_measurement": (
                "darwin-getrusage-child-plus-watchdog"
                if host_platform == "darwin"
                else "reference-gnu-time-timeout-prlimit-plus-watchdog"
            ),
            "rss_marker": DARWIN_RSS_MARKER,
            "rss_metric": (
                "fresh evaluator-process peak RSS from the reference driver; "
                "compiler and wrapper RSS are excluded from the formal ratio"
            ),
            "memory_guard_metric": (
                "max(process-tree RSS, Darwin physical footprint); enforcement only"
            ),
        },
        "candidate_probe_source": str(PROBE_SOURCE),
        "multiplicities": [
            {
                "total_gluons": total,
                "generator_seed": generator_seed(total),
                "required": total in MANDATORY_MULTIPLICITIES,
                "point_count": POINT_COUNT,
                "candidate_lanes": list(LANES if total <= 9 else ("global-winner",)),
                "selected_helicity": "reference-proxy-selected-nonzero",
            }
            for total in multiplicities
        ],
        "measurement": {
            "cpu_cores": 1,
            "batch_size": 1,
            "calibration_seconds_minimum": MINIMUM_CALIBRATION_SECONDS,
            "calibration_scope": "representative-point-1-only",
            "candidate_timer_source": "process-cpu-time",
            "warm_batch_reduction": "point-1-cell-direct",
            "timed_event_cells_per_batch": 1,
            "warm_samples": WARM_SAMPLE_COUNT,
            "warm_excludes_first_call": True,
            "memory_limit_gib": MEMORY_LIMIT_GIB,
            "fresh_process_same_os_rss": True,
            "model_cache": "disabled-per-lane-and-multiplicity",
            "mandatory_lane_order": list(LANES),
            "optional_cold_and_warm_processes_separate": True,
            "cpu_policy": cpu_policy,
        },
        "helicity_policy": {
            "recurrence": "generation-specialized-known-nonzero",
            "on-the-fly": "complete-runtime-selector-fixed-helicity-query",
        },
        "numerical_parity": {
            "reference_values": "ordered-10-AmpliGluonTrace-matrix-elements",
            "candidate_values": (
                "recurrence-first-pass-all-10;"
                "otf-first-pass-points-2-10-plus-point-1-calibration"
            ),
            "candidate_runtime_parameter": "normalization.alpha_s_me_check",
            "candidate_runtime_parameter_value": UNIT_COUPLING_ALPHA_S,
            "candidate_to_reference_scale": "256*factorial(N-2)",
            "relative_error_scale": "max(abs(candidate),abs(reference))",
            "relative_tolerance": NUMERICAL_RELATIVE_TOLERANCE,
            "failure_is_fatal": True,
            "extra_evaluations": 0,
        },
        "global_lane_policy": (
            "recurrence and on-the-fly are eligible under their respective "
            "helicity workloads; the fastest lane passing every mandatory gate "
            "supplies every reported metric and optional continuation"
        ),
        "thresholds": {
            "warm_ratio_maximum": WARM_RATIO_LIMIT,
            "rss_ratio_maximum": RSS_RATIO_LIMIT,
            "cold_to_ready_ratio_maximum": COLD_RATIO_LIMIT,
            "optional_generation_seconds_maximum_exclusive": (
                OPTIONAL_COLD_LIMIT_SECONDS
            ),
            "optional_first_warm_seconds_maximum_exclusive": (
                OPTIONAL_COLD_LIMIT_SECONDS
            ),
            "optional_continuation_deadlines_are_independent": True,
        },
        "candidate_generation_template": list(
            candidate_generation_command(
                python=python,
                lane="recurrence",
                total_gluons=4,
                helicities=(1, 1, 1, 1),
                artifact=PERFORMANCE_ROOT
                / "runs"
                / run_id
                / "candidate"
                / "LANE"
                / "N",
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument(
        "--reference-fft-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=(
            "pinned AllGluonsMultipletFFT checkout used by Reference FFT "
            "(default path is populated by "
            "dev-install --with-reference-fft)"
        ),
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=MINIMUM_CALIBRATION_SECONDS,
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if _RUN_ID.fullmatch(arguments.run_id) is None or arguments.run_id in {".", ".."}:
        raise AcceptanceError("--run-id is not a safe workspace-local name")
    if (
        not math.isfinite(arguments.target_seconds)
        or arguments.target_seconds < MINIMUM_CALIBRATION_SECONDS
    ):
        raise AcceptanceError("--target-seconds must be at least 0.25")
    if not arguments.python or not arguments.cxx or not arguments.fc:
        raise AcceptanceError("compiler and Python commands must be nonempty")


def _campaign(arguments: argparse.Namespace) -> dict[str, object]:
    source_identity = _require_campaign_source_identity()
    reference_source_identity = _require_reference_source_identity()
    run_root = PERFORMANCE_ROOT / "runs" / arguments.run_id
    _create_workspace_directories(run_root)
    environment = _workspace_environment(arguments.python)
    os.environ.update(environment)
    cpu_policy = _enforce_one_core(sys.platform)
    probe, sdk_info = _build_probe(
        run_root=run_root,
        python=arguments.python,
        cxx=arguments.cxx,
        environment=environment,
    )
    reference_module = _load_reference_module()
    references: dict[int, ReferenceMetrics] = {}
    candidates: dict[str, dict[int, CandidateMetrics]] = {lane: {} for lane in LANES}
    optional_status: dict[str, object] = {}
    report_path = run_root / "report.json"
    execution_policy = dry_run_plan(
        run_id=arguments.run_id,
        python=arguments.python,
        include_optional=arguments.include_optional,
    )
    execution_policy["dry_run"] = False

    def partial(status: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "dry_run": False,
            "status": status,
            "terminal": status != "running",
            "source_identity": source_identity.provenance(),
            "reference_source_identity": reference_source_identity.provenance(),
            "policy": execution_policy,
            "environment": {
                "platform": sys.platform,
                "threads": 1,
                "cpu_policy": cpu_policy,
                "sdk_target": sdk_info["target"],
                "sdk_package_version": sdk_info["package_version"],
            },
            "reference": {
                str(total): _plain_reference(record)
                for total, record in sorted(references.items())
            },
            "candidates": {
                lane: {
                    str(total): _plain_candidate(record)
                    for total, record in sorted(records.items())
                }
                for lane, records in candidates.items()
            },
            "optional": optional_status,
        }
        return payload

    def require_numerical_parity(candidate: CandidateMetrics) -> None:
        evidence = candidate.numerical_parity
        if evidence.passes:
            return
        failed = partial("failed-numerical-parity")
        failed.update(
            {
                "passes": False,
                "failure": {
                    "kind": "numerical-parity",
                    "lane": candidate.lane,
                    "total_gluons": candidate.total_gluons,
                    "maximum_relative_error": evidence.maximum_relative_error,
                    "relative_tolerance": evidence.relative_tolerance,
                },
            }
        )
        _write_report(report_path, failed)
        raise NumericalParityError(
            "candidate/reference matrix-element mismatch for "
            f"{candidate.lane} N={candidate.total_gluons} at point "
            f"{evidence.maximum_relative_error_point}: relative error "
            f"{evidence.maximum_relative_error:.17g} exceeds "
            f"{evidence.relative_tolerance:.17g}"
        )

    completed_mandatory: list[int] = []
    for total_gluons in MANDATORY_MULTIPLICITIES:
        reference = _run_reference(
            reference=reference_module,
            total_gluons=total_gluons,
            run_root=run_root,
            python=arguments.python,
            environment=environment,
            fc=arguments.fc,
            target_seconds=arguments.target_seconds,
            timeout_seconds=MANDATORY_LANE_TIMEOUT_SECONDS,
        )
        references[total_gluons] = reference.metrics
        for lane in LANES:
            candidate = _run_candidate(
                lane=lane,
                total_gluons=total_gluons,
                reference=reference,
                probe=probe,
                run_root=run_root,
                python=arguments.python,
                environment=environment,
                target_seconds=arguments.target_seconds,
                timeout_seconds=MANDATORY_LANE_TIMEOUT_SECONDS,
            )
            candidates[lane][total_gluons] = candidate
            require_numerical_parity(candidate)
        completed_mandatory.append(total_gluons)
        viability = _observed_mandatory_gate_viability(
            references,
            candidates,
            multiplicities=completed_mandatory,
        )
        if not viability["viable_lanes"]:
            failed = partial("failed-performance-gates")
            failed.update(
                {
                    "passes": False,
                    "failure": {
                        "kind": "no-global-lane-remains",
                        "message": (
                            "no eligible lane can satisfy every completed "
                            "mandatory warm-runtime, RSS, cold-to-ready, and "
                            "numerical gate"
                        ),
                    },
                    "observed_gate_viability": viability,
                }
            )
            _write_report(report_path, failed)
            return failed
        running = partial("running")
        running["observed_gate_viability"] = viability
        _write_report(report_path, running)

    winner = select_global_lane(references, candidates)
    mandatory_gates = evaluate_gates(winner, references, candidates)
    optional_gates: tuple[GateResult, ...] = ()
    if arguments.include_optional:
        optional_totals: list[int] = []
        for total_gluons in OPTIONAL_MULTIPLICITIES:
            try:
                reference = _run_reference(
                    reference=reference_module,
                    total_gluons=total_gluons,
                    run_root=run_root,
                    python=arguments.python,
                    environment=environment,
                    fc=arguments.fc,
                    target_seconds=arguments.target_seconds,
                    timeout_seconds=OPTIONAL_COLD_LIMIT_SECONDS,
                )
                candidate = _run_candidate(
                    lane=winner,
                    total_gluons=total_gluons,
                    reference=reference,
                    probe=probe,
                    run_root=run_root,
                    python=arguments.python,
                    environment=environment,
                    target_seconds=arguments.target_seconds,
                    timeout_seconds=MANDATORY_LANE_TIMEOUT_SECONDS,
                    continuation_stage_limit_seconds=OPTIONAL_COLD_LIMIT_SECONDS,
                )
                if not candidate.numerical_parity.passes:
                    references[total_gluons] = reference.metrics
                    candidates[winner][total_gluons] = candidate
                    optional_status[str(total_gluons)] = {
                        "status": "failed-numerical-parity"
                    }
                    require_numerical_parity(candidate)
                if not optional_candidate_is_feasible(candidate):
                    optional_status[str(total_gluons)] = {
                        "status": "skipped",
                        "reason": "candidate exceeded the 15 minute or 30 GiB cap",
                    }
                    break
                references[total_gluons] = reference.metrics
                candidates[winner][total_gluons] = candidate
                optional_totals.append(total_gluons)
                optional_status[str(total_gluons)] = {"status": "measured"}
                _write_report(report_path, partial("running"))
            except NumericalParityError:
                raise
            except AcceptanceError as error:
                optional_status[str(total_gluons)] = {
                    "status": "skipped",
                    "reason": str(error),
                }
                break
        optional_gates = evaluate_gates(
            winner,
            references,
            candidates,
            multiplicities=optional_totals,
        )

    gates = (*mandatory_gates, *optional_gates)
    final = partial("complete")
    final.update(
        {
            "global_winning_lane": winner,
            "lane_eligibility": lane_eligibility(candidates),
            "optional": optional_status,
        }
    )
    final.update(_gate_report(gates))
    _write_report(report_path, final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        configure_reference_root(arguments.reference_fft_root)
        _validate_arguments(arguments)
        if arguments.dry_run:
            print(
                json.dumps(
                    dry_run_plan(
                        run_id=arguments.run_id,
                        python=arguments.python,
                        include_optional=arguments.include_optional,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = _campaign(arguments)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passes"] is True else 1
    except (AcceptanceError, OSError, ValueError) as error:
        print(f"fft performance acceptance: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
