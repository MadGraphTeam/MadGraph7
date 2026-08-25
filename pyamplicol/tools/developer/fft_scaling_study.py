#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Resumable FullColor scaling study for direct and FFT contractions.

Every pyAmpliCol artifact retains complete helicity coverage.  By default the
warm runtime query fixes the same nonzero helicity used by the comparison
curves; ``--compare-helicity-sums`` instead times the complete physical sum.
Every external command that can grow with multiplicity is run through the
repository memory watchdog.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci import memory_watchdog  # noqa: E402
from tools.developer import (  # noqa: E402
    fft_gluon_performance_acceptance as performance,
)
from tools.developer import legacy_amplicol  # noqa: E402
from tools.performance_report import legacy as legacy_report  # noqa: E402
from tools.performance_report import (  # noqa: E402
    legacy_structure as legacy_structure_tools,
)
from tools.performance_report.legacy_structure import (  # noqa: E402
    legacy_structural_probe_lock,
)
from tools.performance_report.models import (  # noqa: E402
    Accuracy,
    CellSpec,
    ExecutionMode,
    MeasurementSpec,
    Workload,
)

KIND = "pyamplicol-fullcolor-fft-scaling-study"
SCHEMA_VERSION = 1
STUDY_ROOT = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS" / "fft-scaling-study" / "raw"
FINAL_MULTIPLICITIES = tuple(range(2, 8))
POINT_COUNT = performance.POINT_COUNT
WARM_SAMPLES = performance.WARM_SAMPLE_COUNT
BENCHMARK_BATCH_SIZE = 128
PROFILE_WARMUP_RUNS = 2
PROFILE_PRECISION = 16
REQUESTED_MEMORY_LIMIT_GIB = 30.0
MAX_MEMORY_LIMIT_GIB = REQUESTED_MEMORY_LIMIT_GIB
DEFAULT_TIME_LIMIT_SECONDS = 60.0 * 60.0
MAX_TIME_LIMIT_SECONDS = 60.0 * 60.0
DEFAULT_ALPHA_S = 0.118
NUMERICAL_RELATIVE_TOLERANCE = 1.0e-10
AMPLI_COL_MAX_POINTS = 1_000_000_000
AMPLI_COL_CALIBRATION_HEADROOM = 1.05
FORTRAN_FLOAT = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?"
AMPLI_COL_TIMING_ROW = re.compile(
    rf"^\s*(generation setup|total)\s+({FORTRAN_FLOAT})\s+",
    re.MULTILINE,
)
RSS_MARKER = performance.DARWIN_RSS_MARKER
WATCHDOG_PEAK_GUARD = re.compile(r"peak_guard=([0-9]+(?:\.[0-9]+)?) GiB")
WATCHDOG_PEAK_RSS = re.compile(r"peak_rss=([0-9]+(?:\.[0-9]+)?) GiB")
JSON_PATH_REFERENCE = re.compile(
    r"(?P<path>(?:/|\.{1,2}/)[^\s'\"<>]+?\.json)(?=$|[\s,;:)])"
)
SYMBOLICA_LICENSE_BUSY_FRAGMENT = (
    "cannot start new unlicensed symbolica instance since there is already another "
    "one running on the machine"
)
SYMBOLICA_GENERATION_LOCK_ENV = "PYAMPICOL_FFT_SYMBOLICA_GENERATION_LOCK"
SYMBOLICA_BUSY_RETRY_SECONDS = 5.0
SYMBOLICA_LOCK_POLL_SECONDS = 0.25
LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY = "legacy-amplicol-structural-limit"
_LEGACY_AMPLICOL_PROBE_NAMES = frozenset(
    {"amplicol_color_probe", "amplicol_color_library_probe"}
)


class StudyError(RuntimeError):
    """The bounded scaling-study contract could not be satisfied."""


_RESOURCE_FAILURE_CATEGORIES = frozenset(
    {
        "generation-time-limit",
        "memory-limit",
        "runtime-time-limit",
        "setup-or-runtime-time-limit",
    }
)
_CENSORING_FAILURE_CATEGORIES = _RESOURCE_FAILURE_CATEGORIES | frozenset(
    {LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY}
)


class _CellResourceLimitError(StudyError):
    """A cell reached one explicitly enforced time or memory limit."""

    def __init__(self, message: str, *, category: str) -> None:
        if category not in _RESOURCE_FAILURE_CATEGORIES:
            raise ValueError(f"invalid cell resource-limit category {category!r}")
        super().__init__(message)
        self.category = category


class _CellCommandError(StudyError):
    """A watched command failed in a stage with a known timeout category."""

    def __init__(self, message: str, *, timeout_category: str) -> None:
        if timeout_category not in {
            "generation-time-limit",
            "runtime-time-limit",
            "setup-or-runtime-time-limit",
        }:
            raise ValueError(f"invalid command timeout category {timeout_category!r}")
        super().__init__(message)
        self.timeout_category = timeout_category


def report_helicity_workload(report: Mapping[str, Any]) -> str:
    """Return the strictly declared fixed or summed report workload."""

    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        raise StudyError("campaign report has no policy object")
    measurement = policy.get("measurement")
    if not isinstance(measurement, Mapping):
        raise StudyError("campaign report has no measurement policy")
    policy_declared = policy.get("helicity_workload")
    measurement_declared = measurement.get("helicity_workload")
    if (
        policy_declared is not None
        and measurement_declared is not None
        and str(policy_declared) != str(measurement_declared)
    ):
        raise StudyError(
            "policy and measurement helicity_workload declarations disagree"
        )
    declared = (
        measurement_declared
        if measurement_declared is not None
        else policy_declared
    )
    if declared is None:
        if (
            measurement.get("warm_fixed_helicity") is True
            and measurement.get("warm_helicity_sum") is True
        ):
            raise StudyError("campaign helicity workload markers are contradictory")
        if measurement.get("warm_helicity_sum") is True:
            declared = "sum"
        elif measurement.get("warm_fixed_helicity") is True:
            declared = "fixed"
        else:
            raise StudyError("campaign report does not declare its helicity workload")
    workload = str(declared)
    if workload not in {"fixed", "sum"}:
        raise StudyError("campaign helicity_workload must be 'fixed' or 'sum'")
    if workload == "sum" and (
        measurement.get("warm_fixed_helicity") is not False
        or measurement.get("warm_helicity_sum") is not True
    ):
        raise StudyError(
            "summed campaign must record warm_fixed_helicity=false and "
            "warm_helicity_sum=true"
        )
    if workload == "fixed" and (
        measurement.get("warm_fixed_helicity") is not True
        or measurement.get("warm_helicity_sum") is True
    ):
        raise StudyError(
            "fixed campaign must record warm_fixed_helicity=true without "
            "warm_helicity_sum=true"
        )
    return workload


@dataclass(frozen=True, slots=True)
class Mode:
    key: str
    label: str
    kind: str
    execution_mode: str | None = None
    contraction: str | None = None


@dataclass(frozen=True, slots=True)
class AmpliColWarmSamples:
    repetitions: int
    samples_seconds: tuple[float, ...]
    sample_totals_seconds: tuple[float, ...]
    sample_set_attempts: int


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
_CANDIDATE_MODE_BY_EXECUTION_AND_CONTRACTION = {
    (mode.execution_mode, mode.contraction): mode
    for mode in MODES
    if mode.kind == "candidate"
}


def process_expression(family: str, final_multiplicity: int) -> str:
    if final_multiplicity < 2:
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


def _multiplicities(arguments: argparse.Namespace) -> tuple[int, ...]:
    if arguments.multiplicities is not None:
        return tuple(sorted(set(arguments.multiplicities)))
    return tuple(range(arguments.min_n, arguments.max_n + 1))


def _fill_multiplicities(arguments: argparse.Namespace) -> tuple[int, ...]:
    """Return the policy multiplicities selected for this campaign invocation.

    ``--multiplicity`` remains part of the persisted measurement policy.  The
    separate fill selector lets an orchestrator populate that fixed policy in
    bounded slices without changing resume identity.
    """

    policy = _multiplicities(arguments)
    requested = arguments.fill_multiplicities
    if requested is None:
        return policy
    return tuple(sorted(set(requested)))


def _study_root(arguments: argparse.Namespace) -> Path:
    return arguments.study_root.expanduser().resolve(strict=False)


def _selected_families(arguments: argparse.Namespace) -> tuple[str, ...]:
    requested = set(arguments.families or FAMILIES)
    return tuple(family for family in FAMILIES if family in requested)


def _selected_modes(arguments: argparse.Namespace, family: str) -> tuple[Mode, ...]:
    requested = (
        set(MODE_BY_KEY) - {"compiled-fft"}
        if arguments.modes is None
        else set(arguments.modes)
    )
    if arguments.fft_enabled and not getattr(arguments, "explicit_modes", False):
        selected_execution_modes = {
            MODE_BY_KEY[key].execution_mode
            for key in requested
            if MODE_BY_KEY[key].kind == "candidate"
        }
        for execution_mode in selected_execution_modes:
            direct = _CANDIDATE_MODE_BY_EXECUTION_AND_CONTRACTION.get(
                (execution_mode, "direct")
            )
            if direct is not None:
                requested.add(direct.key)
            # Compiled FFT is presently a selected-helicity diagnostic lane,
            # so it is not auto-added to this helicity-general campaign.
            if execution_mode == "compiled":
                continue
            fft = _CANDIDATE_MODE_BY_EXECUTION_AND_CONTRACTION.get(
                (execution_mode, "symmetric-group-fft")
            )
            if fft is not None:
                requested.add(fft.key)
    return tuple(
        mode
        for mode in MODES
        if mode.key in requested
        and not (family == "ddbar" and mode.kind == "reference")
    )


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


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
    artifact: Path,
    batch_size: int,
    optimization_cores: int | None = None,
) -> tuple[str, ...]:
    assert mode.execution_mode is not None and mode.contraction is not None
    command = (
        python,
        "-m",
        "pyamplicol",
        "generate",
        process_expression(family, final_multiplicity),
        str(artifact),
        "--name",
        process_key(family, final_multiplicity),
        "--model",
        "built-in-sm",
        "--color-accuracy",
        "full",
        "--color-contraction",
        mode.contraction,
        "--execution-mode",
        mode.execution_mode,
        "--set",
        f"evaluator.batch_size={batch_size}",
    )
    if optimization_cores is not None:
        command += (
            "--set",
            f"evaluator.optimization.cores={optimization_cores}",
        )
    return command


def _helicity_identifier(helicity: Sequence[int]) -> str:
    values = tuple(int(value) for value in helicity)
    if not values or any(value not in (-1, 1) for value in values):
        raise StudyError("candidate fixed helicity is invalid")
    return "h:" + ",".join(f"{value:+d}" for value in values)


def _write_candidate_cli_inputs(
    *,
    cell_root: Path,
    events: Sequence[Path],
    alpha_s: float,
) -> tuple[Path, Path]:
    momenta = [_read_event(path)[0] for path in events]
    if len(momenta) != POINT_COUNT:
        raise StudyError("candidate CLI evaluation requires exactly ten shared points")
    momenta_path = cell_root / "cli-momenta.json"
    parameters_path = cell_root / "cli-model-parameters.json"
    momenta_path.write_text(
        json.dumps(momenta, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    parameters_path.write_text(
        json.dumps(
            {"normalization.alpha_s_me_check": alpha_s},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return momenta_path, parameters_path


def _candidate_evaluate_command(
    *,
    python: str,
    artifact: Path,
    process: str,
    momenta: Path,
    model_parameters: Path,
    helicity: Sequence[int],
    sum_helicities: bool,
) -> tuple[str, ...]:
    command = (
        python,
        "-m",
        "pyamplicol",
        "evaluate",
        str(artifact),
        "--process",
        process,
        "--momenta",
        str(momenta),
        "--model-parameters",
        str(model_parameters),
        "--precision",
        str(PROFILE_PRECISION),
    )
    if not sum_helicities:
        command += ("--helicity", _helicity_identifier(helicity))
    return (
        *command,
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )


def _candidate_profile_command(
    *,
    python: str,
    artifact: Path,
    process: str,
    momenta: Path,
    helicity: Sequence[int],
    sum_helicities: bool,
    target_seconds: float,
    batch_size: int,
) -> tuple[str, ...]:
    command = (
        python,
        "-m",
        "pyamplicol",
        "profile",
        str(artifact),
        "--process",
        process,
        "--momenta",
        str(momenta),
        "--target-runtime",
        f"{target_seconds:.17g}",
        "--batch-size",
        str(batch_size),
        "--precision",
        str(PROFILE_PRECISION),
        "--warmup-runs",
        str(PROFILE_WARMUP_RUNS),
        "--minimum-samples",
        str(WARM_SAMPLES),
    )
    if not sum_helicities:
        command += ("--helicity", _helicity_identifier(helicity))
    return (
        *command,
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )


def _parse_candidate_evaluate_json(output: str) -> list[float]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise StudyError("candidate CLI evaluate output is not JSON") from error
    if not isinstance(payload, list) or len(payload) != POINT_COUNT:
        raise StudyError("candidate CLI evaluate did not return ten point values")
    values: list[float] = []
    for entry in payload:
        if not isinstance(entry, Mapping) or set(entry) != {"real", "imag"}:
            raise StudyError(
                "candidate CLI evaluate value is not a complex JSON scalar"
            )
        try:
            real = float(entry["real"])
            imaginary = float(entry["imag"])
        except (TypeError, ValueError) as error:
            raise StudyError("candidate CLI evaluate value is not numeric") from error
        if (
            not math.isfinite(real)
            or not math.isfinite(imaginary)
            or abs(imaginary) > 1.0e-12 * max(abs(real), 1.0e-300)
        ):
            raise StudyError("candidate CLI evaluate value is not finite and real")
        values.append(real)
    return values


def _parse_candidate_profile_json(
    output: str,
    *,
    artifact: Path,
    family: str,
    final_multiplicity: int,
    execution_mode: str,
    helicity: Sequence[int],
    sum_helicities: bool,
    target_seconds: float,
    batch_size: int,
) -> dict[str, object]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise StudyError("candidate CLI profile output is not JSON") from error
    if not isinstance(payload, dict):
        raise StudyError("candidate CLI profile output is not an object")
    expected_helicities = [] if sum_helicities else [_helicity_identifier(helicity)]
    expected_config = {
        "batch_size": batch_size,
        "color_flow_ids": [],
        "helicity_ids": expected_helicities,
        "minimum_samples": WARM_SAMPLES,
        "precision": PROFILE_PRECISION,
        "target_runtime": target_seconds,
        "warmup_runs": PROFILE_WARMUP_RUNS,
    }
    requested = payload.get("requested_config")
    effective = payload.get("effective_config")
    environment = payload.get("environment")
    if requested != expected_config or effective != expected_config:
        raise StudyError("candidate CLI profile configuration differs from the request")
    if not isinstance(environment, Mapping):
        raise StudyError("candidate CLI profile has no environment evidence")
    try:
        wall_time = float(payload["wall_time_per_point"])
        sample_count = int(payload["sample_count"])
        repetitions = int(payload["repetitions_per_sample"])
        completed_samples = int(environment["completed_sample_count"])
        measured_points = int(environment["measured_point_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise StudyError("candidate CLI profile timing fields are malformed") from error
    target = environment.get("target")
    if (
        payload.get("process_id") != process_key(family, final_multiplicity)
        or payload.get("process_expression")
        != process_expression(family, final_multiplicity)
        or environment.get("execution_mode") != execution_mode
        or environment.get("color_accuracy") != "full"
        or environment.get("batch_size") != batch_size
        or environment.get("precision") != PROFILE_PRECISION
        or environment.get("selected_color_ids") != []
        or environment.get("selected_helicity_ids") != expected_helicities
        or not isinstance(target, str)
        or Path(target).expanduser().resolve(strict=False)
        != artifact.expanduser().resolve(strict=False)
        or payload.get("interrupted") is not False
        or environment.get("interrupted") is not False
        or sample_count < WARM_SAMPLES
        or completed_samples != sample_count
        or repetitions < 1
        or measured_points != sample_count * repetitions * batch_size
        or not _positive_finite(wall_time)
    ):
        raise StudyError("candidate CLI profile provenance is inconsistent")
    if execution_mode == "on-the-fly" and (
        not _positive_finite(environment.get("cold_warmup_elapsed_seconds"))
        or environment.get("cold_warmup_runtime_cold_before_first_evaluation")
        is not True
        or environment.get("cold_warmup_runtime_retained_after_first_evaluation")
        is not True
    ):
        raise StudyError("OTF CLI profile lacks authenticated cold warm-up evidence")
    return payload


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(observed), abs(expected), 1.0e-300)


def _require_bounded_rss(max_rss_kib: int, memory_limit_gib: float) -> int:
    if max_rss_kib < 1:
        raise StudyError("runtime RSS evidence is not positive")
    if max_rss_kib >= int(memory_limit_gib * 1024**2):
        raise _CellResourceLimitError(
            f"runtime RSS reached the strict <{memory_limit_gib:g} GiB bound",
            category="memory-limit",
        )
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


def _timed_command(
    command: Sequence[str], *, python: str
) -> tuple[tuple[str, ...], Any]:
    return (python, str(Path(__file__).resolve()), "_time-rss", *command), None


def _parse_rss_marker(stderr: str) -> int:
    matches = re.findall(rf"^{re.escape(RSS_MARKER)}\s+(\d+)\s*$", stderr, re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) < 1:
        raise StudyError("timed process did not report one positive RSS marker")
    return int(matches[0])


def _parse_watchdog_peak_guard_kib(stderr: str) -> int:
    matches = WATCHDOG_PEAK_GUARD.findall(stderr)
    if len(matches) != 1:
        raise StudyError("watched process did not report one peak_guard value")
    value = float(matches[0])
    if not math.isfinite(value) or value < 0.0:
        raise StudyError("watched process peak_guard is invalid")
    return round(value * 1024**2)


def _parse_watchdog_peak_rss_kib(stderr: str) -> int:
    matches = WATCHDOG_PEAK_RSS.findall(stderr)
    if len(matches) != 1:
        raise StudyError("watched process did not report one peak_rss value")
    value = float(matches[0])
    # Very short children can finish before the watchdog's first full sample;
    # its GiB-formatted diagnostic then rounds a sub-MiB observation to 0.000.
    if not math.isfinite(value) or value < 0.0:
        raise StudyError("watched process peak_rss is invalid")
    return round(value * 1024**2)


def _exec_in_command(
    python: str, directory: Path, command: Sequence[str]
) -> tuple[str, ...]:
    return (python, str(Path(__file__).resolve()), "_exec-in", str(directory), *command)


def _parse_amplicol_timing_aggregate(output: str) -> tuple[float, float, int]:
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
    return rows["generation setup"], rows["total"], points


def _parse_amplicol_timing(output: str) -> tuple[float, float, int]:
    generation_setup, total, points = _parse_amplicol_timing_aggregate(output)
    return generation_setup, total / points, points


def _amplicol_fixed_repetitions(
    points: int, total_seconds: float, target_seconds: float
) -> int:
    if points < 1 or points > AMPLI_COL_MAX_POINTS:
        raise StudyError("AmpliCol calibration point count is outside its bound")
    if not _positive_finite(total_seconds) or not _positive_finite(target_seconds):
        raise StudyError("AmpliCol calibration timing is invalid")
    if total_seconds >= target_seconds:
        return points
    required = max(
        points + 1,
        math.ceil(
            points * target_seconds / total_seconds * AMPLI_COL_CALIBRATION_HEADROOM
        ),
    )
    if required > AMPLI_COL_MAX_POINTS:
        raise StudyError(
            "AmpliCol requires more than its bounded point count to reach the "
            "warm calibration target"
        )
    return required


def _collect_amplicol_warm_samples(
    *,
    seed_points: int,
    seed_total_seconds: float,
    target_seconds: float,
    run_sample: Callable[[int, int, int], str],
) -> AmpliColWarmSamples:
    """Collect ten equal-repetition aggregates that each meet the time floor."""

    repetitions = _amplicol_fixed_repetitions(
        seed_points, seed_total_seconds, target_seconds
    )
    sample_set_attempt = 0
    while True:
        sample_set_attempt += 1
        samples: list[float] = []
        totals: list[float] = []
        for sample_index in range(1, WARM_SAMPLES + 1):
            output = run_sample(sample_set_attempt, sample_index, repetitions)
            _, total, observed_points = _parse_amplicol_timing_aggregate(output)
            if observed_points != repetitions:
                raise StudyError(
                    "AmpliCol fixed-repetition probe reported a different point count"
                )
            if total < target_seconds:
                repetitions = _amplicol_fixed_repetitions(
                    repetitions, total, target_seconds
                )
                break
            totals.append(total)
            samples.append(total / repetitions)
        else:
            return AmpliColWarmSamples(
                repetitions=repetitions,
                samples_seconds=tuple(samples),
                sample_totals_seconds=tuple(totals),
                sample_set_attempts=sample_set_attempt,
            )


def _empty_report(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "policy": dry_run_plan(arguments),
        "cells": {
            family: {mode.key: {} for mode in _selected_modes(arguments, family)}
            for family in _selected_families(arguments)
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
    normalize_loaded_failure_cells(payload, path.parent)
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


def _failure_count(report: Mapping[str, object]) -> int:
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return 0
    return sum(
        cell.get("status") == "failed"
        for family_cells in cells.values()
        if isinstance(family_cells, Mapping)
        for curve in family_cells.values()
        if isinstance(curve, Mapping)
        for cell in curve.values()
        if isinstance(cell, Mapping)
    )


def selected_cells_complete(
    report: Mapping[str, object],
    *,
    family: str,
    modes: Sequence[str],
    multiplicities: Sequence[int],
) -> bool:
    """Return whether the requested cells have authoritative terminal records."""

    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return False
    family_cells = cells.get(family)
    if not isinstance(family_cells, Mapping):
        return False
    for mode in modes:
        curve = family_cells.get(mode)
        if not isinstance(curve, Mapping):
            return False
        for final_multiplicity in multiplicities:
            cell = curve.get(str(final_multiplicity))
            if not isinstance(cell, Mapping):
                return False
            status = cell.get("status")
            if status in {"measured", "skipped"}:
                continue
            if status == "failed" and cell.get("censors_higher_multiplicities") is True:
                continue
            return False
    return True


def resource_frontier_inversions(
    report: Mapping[str, object],
) -> tuple[tuple[str, str, int, int], ...]:
    """Find measured cells retained above an earlier resource frontier."""

    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return ()
    inversions: list[tuple[str, str, int, int]] = []
    for family, family_cells in cells.items():
        if not isinstance(family, str) or not isinstance(family_cells, Mapping):
            continue
        for mode, curve in family_cells.items():
            if not isinstance(mode, str) or not isinstance(curve, Mapping):
                continue
            frontier: int | None = None
            for raw_n, cell in sorted(
                curve.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
            ):
                if not str(raw_n).isdigit() or not isinstance(cell, Mapping):
                    continue
                final_multiplicity = int(raw_n)
                if cell.get("censors_higher_multiplicities") is True and cell.get(
                    "status"
                ) in {"failed", "skipped"}:
                    failed_at = cell.get("failed_at_n")
                    frontier = min(
                        frontier if frontier is not None else final_multiplicity,
                        int(failed_at)
                        if isinstance(failed_at, int)
                        else final_multiplicity,
                    )
                elif cell.get("status") == "measured" and frontier is not None:
                    inversions.append((family, mode, frontier, final_multiplicity))
    return tuple(inversions)


def compose_report(
    arguments: argparse.Namespace,
    curve_sources: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    halt_reason: str | None = None,
) -> dict[str, object]:
    """Compose one canonical study report from already-authenticated curves.

    The orchestration layer may schedule isolated curve shards, but report
    shape, completeness, failure accounting, and terminal status remain owned
    here at the scaling-study boundary.
    """

    report = _empty_report(arguments)
    cells = report["cells"]
    assert isinstance(cells, dict)
    policy_multiplicities = set(_multiplicities(arguments))
    for family in _selected_families(arguments):
        source_family = curve_sources.get(family, {})
        target_family = cells[family]
        assert isinstance(target_family, dict)
        for mode in _selected_modes(arguments, family):
            source_curve = source_family.get(mode.key, {})
            if not isinstance(source_curve, Mapping):
                raise StudyError(
                    f"composed {family}/{mode.key} curve must be an object"
                )
            malformed = [
                raw_n
                for raw_n in source_curve
                if not isinstance(raw_n, str) or not raw_n.isdigit()
            ]
            unexpected = sorted(
                int(raw_n)
                for raw_n in source_curve
                if isinstance(raw_n, str)
                and raw_n.isdigit()
                and int(raw_n) not in policy_multiplicities
            )
            if unexpected or malformed:
                raise StudyError(
                    f"composed {family}/{mode.key} curve is outside its policy"
                )
            target_family[mode.key] = copy.deepcopy(dict(source_curve))

    failures = _failure_count(report)
    complete = all(
        selected_cells_complete(
            report,
            family=family,
            modes=tuple(mode.key for mode in _selected_modes(arguments, family)),
            multiplicities=_multiplicities(arguments),
        )
        for family in _selected_families(arguments)
    )
    inversions = resource_frontier_inversions(report)
    if inversions:
        report["resource_frontier_inversions"] = [list(item) for item in inversions]
    else:
        report.pop("resource_frontier_inversions", None)
    if halt_reason is not None:
        report["status"] = "stopped-correctness-failure"
        report["status_reason"] = halt_reason
    elif complete:
        report["status"] = "complete-with-failures" if failures else "complete"
        report.pop("status_reason", None)
    else:
        report["status"] = "running"
        report.pop("status_reason", None)
    report["failure_count"] = failures
    return report


def _gg_reference_dependency_skip(
    report: Mapping[str, object], final_multiplicity: int
) -> dict[str, object] | None:
    reference = _curve(report, "gg", "reference-fft").get(str(final_multiplicity))
    if not isinstance(reference, Mapping):
        return None
    status = reference.get("status")
    if status == "failed":
        if reference.get("censors_higher_multiplicities") is not True:
            return None
        failed_at = final_multiplicity
    elif status == "skipped" and isinstance(reference.get("failed_at_n"), int):
        failed_at = int(reference["failed_at_n"])
    else:
        return None
    return {
        "status": "skipped",
        "failure_category": "dependency-unavailable",
        "failure_reason": (
            "Reference FFT baseline/input unavailable because its curve was "
            f"resource-censored at n={failed_at}"
        ),
        "censors_higher_multiplicities": False,
        "dependency": {
            "family": "gg",
            "mode": "reference-fft",
            "n": final_multiplicity,
            "status": status,
            "resource_failure_at_n": failed_at,
        },
    }


def _amplicol_gg_dense_index_preflight(
    final_multiplicity: int, memory_limit_gib: float
) -> dict[str, object]:
    total_external = final_multiplicity + 2
    color_orders = math.factorial(total_external - 1)
    lower_bound_bytes = 2 * color_orders**2
    memory_limit_bytes = int(memory_limit_gib * 1024**3)
    return {
        "formula": "2*((total_external-1)!)^2 bytes",
        "color_orders": color_orders,
        "lower_bound_bytes": lower_bound_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "feasible": lower_bound_bytes < memory_limit_bytes,
    }


def _cell_base(
    family: str,
    mode: Mode,
    final_multiplicity: int,
    *,
    sum_helicities: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "family": family,
        "mode": mode.key,
        "label": mode.label,
        "n": final_multiplicity,
        "total_external": final_multiplicity + 2,
        "process": process_expression(family, final_multiplicity),
        "color_accuracy": "full",
    }
    if sum_helicities:
        result.update(
            {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
            }
        )
    if mode.kind == "candidate":
        result["execution_mode"] = mode.execution_mode
        result["color_contraction"] = mode.contraction
        result["generation_helicity_coverage"] = "all"
        if not sum_helicities:
            result["warm_fixed_helicity"] = True
    return result


def _compiled_fft_not_applicable_cell(
    *,
    family: str,
    mode: Mode,
    final_multiplicity: int,
    sum_helicities: bool = False,
) -> dict[str, object] | None:
    """Exclude the selected-helicity-only compiled FFT diagnostic lane."""

    if mode.key != "compiled-fft":
        return None
    return _cell_base(
        family,
        mode,
        final_multiplicity,
        sum_helicities=sum_helicities,
    ) | {
        "status": "not-applicable",
        "failure_category": "publication-requires-helicity-general-artifact",
        "failure_reason": (
            "compiled FFT is disabled in this publication campaign because its "
            "current backend supports only a generation-selected diagnostic "
            "helicity, while publication artifacts must support all runtime "
            "helicities; compiled-direct remains supported"
        ),
        "failure_reason_short": (
            "selected-helicity diagnostic only; compiled-direct remains supported"
        ),
        "censors_higher_multiplicities": False,
        "applicability": {
            "compiled_direct": True,
            "compiled_fft": False,
            "reason_code": "publication-requires-all-runtime-helicities",
        },
    }


def otf_protocol_scope_cell(
    *,
    family: str,
    mode: Mode,
    final_multiplicity: int,
    sum_helicities: bool = False,
) -> dict[str, object] | None:
    """Apply the workload-specific OTF publication frontier."""

    maximum_multiplicity = 6
    if (
        mode.execution_mode != "on-the-fly"
        or final_multiplicity <= maximum_multiplicity
    ):
        return None
    return _cell_base(
        family,
        mode,
        final_multiplicity,
        sum_helicities=sum_helicities,
    ) | {
        "status": "skipped",
        "failure_category": "publication-protocol-scope",
        "failure_reason": (
            "the final scaling scan deliberately limits on-the-fly curves to "
            f"n<={maximum_multiplicity} for this helicity workload; beyond that "
            "it retains recurrence, AmpliCol, and Reference FFT where applicable"
        ),
        "censors_higher_multiplicities": True,
        "failed_at_n": maximum_multiplicity + 1,
    }


def apply_protocol_scope_cells(
    report: dict[str, object],
    *,
    family: str,
    modes: Sequence[str],
    multiplicities: Sequence[int],
) -> bool:
    """Populate deliberate non-measurement OTF cells for an orchestrated shard."""

    sum_helicities = report_helicity_workload(report) == "sum"
    changed = False
    for mode_key in modes:
        mode = MODE_BY_KEY[mode_key]
        curve = _curve(report, family, mode_key)
        for final_multiplicity in multiplicities:
            cell = otf_protocol_scope_cell(
                family=family,
                mode=mode,
                final_multiplicity=final_multiplicity,
                sum_helicities=sum_helicities,
            )
            if cell is not None and str(final_multiplicity) not in curve:
                curve[str(final_multiplicity)] = cell
                changed = True
    return changed


def _referenced_command_records(
    detail: str, cell_root: Path
) -> tuple[tuple[Path, Mapping[str, object]], ...]:
    """Load only watched-command records explicitly cited by the exception."""

    candidate_paths: dict[str, Path] = {}
    for match in JSON_PATH_REFERENCE.finditer(detail):
        raw_path = Path(match.group("path"))
        path = raw_path if raw_path.is_absolute() else cell_root / raw_path
        candidate_paths[str(path.resolve(strict=False))] = path
    records: list[tuple[Path, Mapping[str, object]]] = []
    try:
        local_paths = sorted(cell_root.rglob("*.json"))
    except OSError:
        local_paths = []
    for path in local_paths:
        rendered_paths = {str(path)}
        with contextlib.suppress(OSError):
            rendered_paths.add(str(path.resolve()))
        if not any(rendered in detail for rendered in rendered_paths):
            continue
        candidate_paths[str(path.resolve(strict=False))] = path
    for path in sorted(candidate_paths.values(), key=str):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        command = payload.get("command")
        watchdog_command = payload.get("watchdog_command")
        timed_out = payload.get("timed_out")
        returncode = payload.get("returncode")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) for value in command)
            or not isinstance(watchdog_command, list)
            or not watchdog_command
            or not all(isinstance(value, str) for value in watchdog_command)
            or not isinstance(timed_out, bool)
            or not isinstance(returncode, int)
            or isinstance(returncode, bool)
        ):
            continue
        records.append((path, payload))
    return tuple(records)


def _legacy_amplicol_probe_command(record: Mapping[str, object]) -> bool:
    command = record.get("command")
    if not isinstance(command, list):
        return False
    return any(
        isinstance(value, str) and Path(value).name in _LEGACY_AMPLICOL_PROBE_NAMES
        for value in command
    )


def _legacy_amplicol_structural_crash(record: Mapping[str, object]) -> bool:
    if not _legacy_amplicol_probe_command(record):
        return False
    output = "\n".join(
        str(record.get(field, "")) for field in ("stdout", "stderr")
    ).lower()
    return (
        ("program received signal sigsegv" in output or "segmentation fault" in output)
        and "init_col" in output
    )


def _mark_legacy_amplicol_structural_limit(cell: dict[str, object]) -> None:
    cell["failure_category"] = LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY
    cell["failure_reason_short"] = "legacy AmpliCol structural colour-init limit"
    cell["censors_higher_multiplicities"] = True


def normalize_loaded_failure_cells(
    report: dict[str, object], run_root: Path
) -> bool:
    """Upgrade historical legacy-probe colour-init crashes to censoring cells."""

    cells = report.get("cells")
    if not isinstance(cells, dict):
        return False
    changed = False
    for family, family_cells in cells.items():
        if not isinstance(family, str) or not isinstance(family_cells, dict):
            continue
        curve = family_cells.get("amplicol")
        if not isinstance(curve, dict):
            continue
        for raw_n, cell in curve.items():
            if (
                not isinstance(raw_n, str)
                or not raw_n.isdigit()
                or not isinstance(cell, dict)
                or cell.get("status") != "failed"
                or cell.get("failure_category")
                == LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY
            ):
                continue
            reason = cell.get("failure_reason")
            if not isinstance(reason, str):
                continue
            cell_root = run_root / "amplicol" / family / f"n{raw_n}"
            if any(
                _legacy_amplicol_structural_crash(record)
                for _, record in _referenced_command_records(reason, cell_root)
            ):
                _mark_legacy_amplicol_structural_limit(cell)
                changed = True
    return changed


def _is_memory_watchdog_record(record: Mapping[str, object]) -> bool:
    command = record["command"]
    watchdog_command = record["watchdog_command"]
    assert isinstance(command, list)
    assert isinstance(watchdog_command, list)
    try:
        separator = watchdog_command.index("--")
    except ValueError:
        return False
    return (
        str(performance.WATCHDOG) in watchdog_command[:separator]
        and watchdog_command[separator + 1 :] == command
    )


def _watchdog_outcome(
    record: Mapping[str, object], cell_root: Path
) -> str | None:
    raw_report_path = record.get("watchdog_report")
    if not isinstance(raw_report_path, str) or not raw_report_path:
        return None
    report_path = Path(raw_report_path)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        resolved_report = report_path.resolve(strict=True)
        resolved_report.relative_to(cell_root.resolve(strict=False))
        payload = json.loads(resolved_report.read_text(encoding="utf-8"))
        execution = payload["execution"]
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != memory_watchdog.WATCHDOG_REPORT_KIND
        or payload.get("schema_version") != memory_watchdog.WATCHDOG_REPORT_SCHEMA
        or payload.get("complete") is not True
        or not isinstance(execution, Mapping)
        or not isinstance(execution.get("outcome"), str)
    ):
        return None
    return str(execution["outcome"])


def _typed_command_failure_category(
    error: Exception,
    cell_root: Path,
    records: Sequence[tuple[Path, Mapping[str, object]]],
) -> str:
    timeout_category = (
        error.timeout_category
        if isinstance(error, _CellCommandError)
        else "generation-time-limit"
    )
    if any(record["timed_out"] is True for _, record in records):
        return timeout_category
    for _, record in records:
        watchdog_outcome = _watchdog_outcome(record, cell_root)
        if watchdog_outcome == "memory-limit-exceeded":
            return "memory-limit"
        if watchdog_outcome is not None:
            continue
        if (
            record["returncode"] == memory_watchdog.MEMORY_LIMIT_EXIT_CODE
            and _is_memory_watchdog_record(record)
        ):
            return "memory-limit"
    if any(_legacy_amplicol_structural_crash(record) for _, record in records):
        return LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY
    return "error"


def _failure_cell(
    base: Mapping[str, object], error: Exception, cell_root: Path
) -> dict[str, object]:
    detail = str(error)
    command_records = _referenced_command_records(detail, cell_root)
    disk_evidence = "\n".join(
        (
            detail,
            *(
                str(record.get(field, ""))
                for _, record in command_records
                for field in ("stdout", "stderr")
            ),
        )
    ).lower()
    category = "error"
    if getattr(error, "errno", None) == 28 or any(
        marker in disk_evidence
        for marker in ("no space left on device", "errno 28", "enospc")
    ):
        category = "disk-infrastructure-failure"
    elif isinstance(error, _CellResourceLimitError):
        category = error.category
    elif isinstance(error, legacy_report.ProfilingTimeLimitError):
        category = "runtime-time-limit"
    else:
        category = _typed_command_failure_category(
            error, cell_root, command_records
        )
    censors_higher_multiplicities = category in _CENSORING_FAILURE_CATEGORIES
    cell = dict(base) | {
        "status": "failed",
        "censors_higher_multiplicities": censors_higher_multiplicities,
        "failure_category": category,
        "failure_reason": detail,
    }
    if category == LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY:
        _mark_legacy_amplicol_structural_limit(cell)
    return cell


def _reference_cell(
    *,
    arguments: argparse.Namespace,
    reference: Any,
    run_root: Path,
    environment: Mapping[str, str],
    final_multiplicity: int,
) -> dict[str, object]:
    try:
        result = performance._run_reference(
            reference=reference,
            total_gluons=final_multiplicity + 2,
            run_root=run_root,
            python=arguments.python,
            environment=environment,
            fc=arguments.fc,
            target_seconds=arguments.target_seconds,
            timeout_seconds=arguments.runtime_timeout,
            cold_limit_seconds=arguments.generation_timeout,
            memory_limit_gib=arguments.memory_limit_gib,
            repetition_quantum=arguments.batch_size,
            sum_helicities=arguments.compare_helicity_sums,
        )
    except performance.ReferenceColdLimitError as error:
        raise _CellResourceLimitError(
            str(error), category="generation-time-limit"
        ) from error
    metrics = result.metrics
    if metrics.setup_to_ready_seconds >= arguments.generation_timeout:
        raise _CellResourceLimitError(
            "reference generation reached the strict generation cap",
            category="generation-time-limit",
        )
    shared_event_paths = tuple(str(path) for path in result.event_paths)
    if tuple(metrics.event_paths) != shared_event_paths:
        raise StudyError("reference shared event-path provenance is inconsistent")
    timed_event_paths = (
        tuple(str(path) for path in metrics.exhaustive_event_paths)
        if arguments.compare_helicity_sums
        else shared_event_paths
    )
    if (
        len(shared_event_paths) != POINT_COUNT
        or len(timed_event_paths) != POINT_COUNT
    ):
        raise StudyError("reference event-path provenance must contain ten points")
    return _cell_base(
        "gg",
        MODE_BY_KEY["reference-fft"],
        final_multiplicity,
        sum_helicities=arguments.compare_helicity_sums,
    ) | {
        "status": "measured",
        "helicity": list(metrics.selected_helicity),
        # Compatibility: candidate/shared kinematics continue to consume
        # event_paths.  The explicit aliases prevent a summed Reference timing
        # run from falsely claiming that those fixed-H files were timed.
        "event_paths": list(shared_event_paths),
        "shared_event_paths": list(shared_event_paths),
        "timed_event_paths": list(timed_event_paths),
        "event_path_semantics": {
            "shared": "fixed-helicity event files reused by comparison curves",
            "timed": (
                "exhaustive-helicity Reference input files"
                if arguments.compare_helicity_sums
                else "same fixed-helicity event files as shared"
            ),
            "event_paths_compatibility_alias": "shared_event_paths",
        },
        "point_values": list(metrics.matrix_elements),
        "metrics": {
            "generation_seconds": metrics.setup_to_ready_seconds,
            "warm_seconds_per_point": metrics.warm_median_seconds,
            "max_rss_kib": _require_bounded_rss(
                metrics.max_rss_kib, arguments.memory_limit_gib
            ),
        },
        "reference": asdict(metrics),
        **(
            {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
                "helicity_coverage_count": metrics.helicity_coverage_count,
                "timed_helicity_count": metrics.timed_helicity_count,
                "active_helicity_count": metrics.active_helicity_count,
                "helicity_count_semantics": {
                    "coverage": "complete physical input-helicity axis",
                    "active": "analytic-nonzero configurations",
                    "timed": "analytic-nonzero configurations evaluated per sum",
                },
            }
            if arguments.compare_helicity_sums
            else {}
        ),
    }


class _WatchedLegacyExecutor:
    """Route the authoritative legacy-library workflow through cell limits."""

    def __init__(
        self,
        *,
        arguments: argparse.Namespace,
        environment: Mapping[str, str],
        log_root: Path,
    ) -> None:
        self.arguments = arguments
        self.environment = dict(environment)
        self.log_root = log_root
        self.command_index = 0
        self.generation_deadline = time.monotonic() + arguments.generation_timeout
        self.runtime_deadline: float | None = None
        self.maximum_rss_kib = 0
        self.maximum_guard_kib = 0

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> legacy_report.CommandResult:
        rendered = tuple(os.fspath(item) for item in args)
        if not rendered:
            raise StudyError("legacy AmpliCol requested an empty command")
        executable = Path(rendered[0]).name
        if (
            executable == "amplicol_color_library_probe"
            and self.runtime_deadline is None
        ):
            self.runtime_deadline = time.monotonic() + self.arguments.runtime_timeout
        deadline = (
            self.runtime_deadline
            if self.runtime_deadline is not None
            else self.generation_deadline
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            stage = "runtime" if self.runtime_deadline is not None else "generation"
            raise _CellResourceLimitError(
                f"AmpliCol generated-library {stage} deadline expired",
                category=f"{stage}-time-limit",
            )
        self.command_index += 1
        command_environment = {
            **self.environment,
            **({} if environment is None else dict(environment)),
        }
        log_path = self.log_root / f"{self.command_index:03d}-{executable}.json"
        try:
            completed = performance._run_watched(
                _exec_in_command(
                    self.arguments.python,
                    cwd,
                    rendered,
                ),
                python=self.arguments.python,
                environment=command_environment,
                timeout_seconds=remaining,
                log_path=log_path,
                memory_limit_gib=self.arguments.memory_limit_gib,
                watchdog_report_path=log_path.with_suffix(".watchdog.json"),
            )
        except performance.AcceptanceError as error:
            stage = "runtime" if self.runtime_deadline is not None else "generation"
            raise _CellCommandError(
                f"AmpliCol generated-library {stage} command failed: {error}",
                timeout_category=f"{stage}-time-limit",
            ) from error
        if completed.peak_rss_kib is None or completed.peak_guard_kib is None:
            raise StudyError("AmpliCol generated-library RSS evidence is missing")
        self.maximum_rss_kib = max(self.maximum_rss_kib, completed.peak_rss_kib)
        self.maximum_guard_kib = max(self.maximum_guard_kib, completed.peak_guard_kib)
        return legacy_report.CommandResult(
            args=rendered,
            cwd=cwd,
            elapsed_seconds=completed.elapsed_seconds,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            environment={} if environment is None else dict(environment),
        )


_SUMMED_AMPLICOL_PROBE_OVERLAY_ABI = (
    "pyamplicol-summed-amplicol-probe-source-overlay-v1"
)
_SUMMED_AMPLICOL_FERMION_GUARD = (
    "          ! No fermion permutation sign is represented here.  Keep fermions\n"
    "          ! fixed and use this fallback only for identical bosonic legs.\n"
    "          if (label.ne.local_label .and. &\n"
    "               phys_model%is_fermion(pgl(jgroup)%processes(label,jint))) then\n"
)
_SUMMED_AMPLICOL_UNIQUE_FERMION_GUARD = (
    "          ! Relabelling a fermion is sign-free only when its exact PDG is\n"
    "          ! unique on this side. Repeated identical fermions remain fixed.\n"
    "          if (.not.fermion_relabel_is_unique("
    "jgroup,jint,label,local_label)) then\n"
)
_SUMMED_AMPLICOL_FALLBACK_GUARD = """             if (pos.ne.local_label .and. &
                  phys_model%is_fermion(pgl(jgroup)%processes(pos,jint))) cycle
"""
_SUMMED_AMPLICOL_UNIQUE_FALLBACK_GUARD = (
    "             if (.not.fermion_relabel_is_unique("
    "jgroup,jint,pos,local_label)) cycle\n"
)
_SUMMED_AMPLICOL_HELPER_ANCHOR = (
    "  logical function is_pure_gluon_word(jgroup,jint,word,nord)\n"
)
_SUMMED_AMPLICOL_HELPER = """  logical function fermion_relabel_is_unique( &
       jgroup,jint,label,local_label)
    implicit none
    integer,intent(in) :: jgroup,jint,label,local_label
    integer :: position, generated_count, local_count, particle
    fermion_relabel_is_unique = .true.
    if (label.eq.local_label) return
    particle = pgl(jgroup)%processes(label,jint)
    if (.not.phys_model%is_fermion(particle)) return
    generated_count = 0
    local_count = 0
    do position=1,n
       if ((position.le.2).eqv.(label.le.2)) then
          if (pgl(jgroup)%processes(position,jint).eq.particle) &
               generated_count = generated_count + 1
       endif
       if ((position.le.2).eqv.(local_label.le.2)) then
          if (local_part(position,1).eq.local_part(local_label,1)) &
               local_count = local_count + 1
       endif
    enddo
    fermion_relabel_is_unique = generated_count.eq.1 .and. local_count.eq.1
  end function fermion_relabel_is_unique

"""


def _replace_summed_amplicol_probe_anchor(
    source: str, old: str, new: str, *, label: str
) -> str:
    count = source.count(old)
    if count != 1:
        raise StudyError(
            f"pinned summed AmpliCol probe anchor {label!r} occurs {count} times, "
            "expected exactly once"
        )
    return source.replace(old, new, 1)


def _transform_summed_amplicol_probe_source(payload: bytes) -> bytes:
    """Apply the exact unique-same-side-fermion row-map fix to pinned source."""

    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StudyError("pinned AmpliCol library probe source is not UTF-8") from error
    source = _replace_summed_amplicol_probe_anchor(
        source,
        _SUMMED_AMPLICOL_FERMION_GUARD,
        _SUMMED_AMPLICOL_UNIQUE_FERMION_GUARD,
        label="colour-row fermion guard",
    )
    source = _replace_summed_amplicol_probe_anchor(
        source,
        _SUMMED_AMPLICOL_FALLBACK_GUARD,
        _SUMMED_AMPLICOL_UNIQUE_FALLBACK_GUARD,
        label="singlet-row fermion guard",
    )
    source = _replace_summed_amplicol_probe_anchor(
        source,
        _SUMMED_AMPLICOL_HELPER_ANCHOR,
        _SUMMED_AMPLICOL_HELPER + _SUMMED_AMPLICOL_HELPER_ANCHOR,
        label="unique-fermion helper insertion",
    )
    return source.encode("utf-8")


def _summed_amplicol_source_stat(value: os.stat_result) -> dict[str, int]:
    return {
        "mode": value.st_mode,
        "size": value.st_size,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "atime_ns": value.st_atime_ns,
        "mtime_ns": value.st_mtime_ns,
    }


@contextlib.contextmanager
def _summed_amplicol_probe_source_overlay(
    repository: Path,
    artifact_root: Path,
) -> Any:
    """Temporarily install the summed-probe fix into one clean pinned checkout."""

    repository = repository.expanduser().resolve()
    source = repository / "amplicol_color_library_probe.f03"
    with legacy_structural_probe_lock(repository):
        # Validate under the same lock as the temporary edit. This rejects a
        # modified checkout rather than treating arbitrary source as an overlay input.
        legacy_amplicol.validate_checkout(repository)
        if not source.is_file():
            raise StudyError(
                f"pinned AmpliCol library probe source is absent: {source}"
            )
        original_stat = source.stat()
        original = source.read_bytes()
        patched = _transform_summed_amplicol_probe_source(original)
        if patched == original:
            raise StudyError("summed AmpliCol probe overlay made no source change")
        snapshot = artifact_root / "amplicol_color_library_probe.f03"
        manifest = artifact_root / "provenance.json"
        record: dict[str, object] = {
            "abi": _SUMMED_AMPLICOL_PROBE_OVERLAY_ABI,
            "scope": "summed-generated-library-amplicol-cells-only",
            "checkout_validation": "clean-pinned-revision-under-structural-probe-lock",
            "revision": legacy_amplicol.expected_revision(),
            "repository": str(repository),
            "source": source.name,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "patched_sha256": hashlib.sha256(patched).hexdigest(),
            "patched_source_snapshot": str(snapshot),
            "manifest": str(manifest),
            "source_stat_before": _summed_amplicol_source_stat(original_stat),
            "restoration": {"status": "pending"},
        }
        completed = False
        try:
            legacy_structure_tools._atomic_write(snapshot, patched)
            legacy_structure_tools._atomic_json(manifest, record)
            legacy_structure_tools._atomic_write(source, patched)
            os.chmod(source, original_stat.st_mode)
            yield record
            completed = True
        finally:
            legacy_structure_tools._atomic_write(source, original)
            os.chmod(source, original_stat.st_mode)
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            restored = source.read_bytes()
            # Undo any atime update caused by the byte-for-byte verification.
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            restored_stat = source.stat()
            stat_after = _summed_amplicol_source_stat(restored_stat)
            stat_before = _summed_amplicol_source_stat(original_stat)
            if restored != original or stat_after != stat_before:
                raise StudyError(
                    "summed AmpliCol probe overlay failed to restore its source"
                )
            record["restoration"] = {
                "status": "restored",
                "body_exit": "success" if completed else "exception",
                "sha256": hashlib.sha256(restored).hexdigest(),
                "source_stat_after": stat_after,
            }
            legacy_structure_tools._atomic_json(manifest, record)


def _amplicol_summed_cell(
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
    """Measure the create-raw bulk-family complete helicity sum."""

    base = _cell_base(
        family,
        MODE_BY_KEY["amplicol"],
        final_multiplicity,
        sum_helicities=True,
    )
    cell_root = run_root / "amplicol" / family / f"n{final_multiplicity}"
    setup = cell_root / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    momenta, event_helicity = _read_event(events[0])
    if tuple(helicity) != event_helicity:
        raise StudyError("AmpliCol event helicity differs from the shared input")

    cell = CellSpec(
        dataset_id="fft-scaling-helicity-sum",
        process=process_expression(family, final_multiplicity),
        n_final=final_multiplicity,
        process_key=family,
        measurement=MeasurementSpec(
            execution_mode=ExecutionMode.AMPLICOL,
            model=None,
            accuracy=Accuracy.FULL,
            backend="original-amplicol-generated-library",
            jit_optimization_level=None,
        ),
        workload=Workload.CONTRACTED,
    )
    executor = _WatchedLegacyExecutor(
        arguments=arguments,
        environment=environment,
        log_root=cell_root / "commands",
    )
    adapter = legacy_report.LegacyMeasurementAdapter(
        executor=executor,
        structural_proof=False,
    )
    settings = legacy_report.LegacySettings(
        target_runtime_seconds=arguments.target_seconds,
        warmup_points=1,
        minimum_points=1,
        maximum_points=AMPLI_COL_MAX_POINTS,
        minimum_profile_chunks=WARM_SAMPLES,
        maximum_profile_chunks=max(64, WARM_SAMPLES),
        jobs=1,
        repository=arguments.amplicol_repository,
        validate_checkout=False,
        profiling_time_limit_seconds=arguments.runtime_timeout,
    )
    commands: list[dict[str, object]] = []
    log_path = cell_root / "legacy-generated-library.log"
    generation_started = time.perf_counter()
    with _summed_amplicol_probe_source_overlay(
        arguments.amplicol_repository,
        cell_root / "summed-probe-source-overlay",
    ) as probe_source_overlay:
        context = adapter._prepare_process(
            cell,
            repository=arguments.amplicol_repository,
            artifact_path=setup,
            commands=commands,
            log_path=log_path,
        )
        context = replace(
            context,
            momenta=tuple(tuple(float(value) for value in row) for row in momenta),
            points=(tuple(tuple(float(value) for value in row) for row in momenta),),
        )
        adapter._generate_library(
            context=context,
            repository=arguments.amplicol_repository,
            raw_color=True,
            n_final=final_multiplicity,
            settings=settings,
            commands=commands,
            log_path=log_path,
        )
        adapter._run(
            ("make", "-j1", "amplicol_color_library_probe"),
            cwd=arguments.amplicol_repository,
            commands=commands,
            log_path=log_path,
        )
        generated = adapter.snapshotter.snapshot(
            arguments.amplicol_repository,
            cell_root / "contracted-generated-library",
            executables=("amplicol_color_library_probe",),
            process_file=context.process_file,
        )
        ordered = adapter.api.ordered_momenta(
            context.source_pdgs,
            context.entry.process_pdgs,
            context.momenta,
        )
        momentum_path = generated / "momenta.dat"
        momentum_path.write_text(
            "\n".join(
                " ".join(format(component, ".17g") for component in vector)
                for vector in ordered
            )
            + "\n",
            encoding="utf-8",
        )
        generation_seconds = time.perf_counter() - generation_started
        if generation_seconds >= arguments.generation_timeout:
            raise _CellResourceLimitError(
                "AmpliCol generated-library generation reached its cap",
                category="generation-time-limit",
            )
        library_environment = legacy_report._library_environment(generated)
        profile = adapter._profile(
            lambda count: adapter._invoke_command(
                (
                    "./amplicol_color_library_probe",
                    str(count),
                    str(context.entry.group),
                    str(context.entry.integral),
                    Accuracy.FULL.value,
                    momentum_path.name,
                ),
                cwd=generated,
                environment=library_environment,
                commands=commands,
                log_path=log_path,
            ),
            settings=settings,
            timing_labels=("total",),
        )
        validation_result, _, _ = adapter._invoke_command(
            (
                "./amplicol_color_library_probe",
                "1",
                str(context.entry.group),
                str(context.entry.integral),
                Accuracy.FULL.value,
                momentum_path.name,
            ),
            cwd=generated,
            environment=library_environment,
            commands=commands,
            log_path=log_path,
        )
    matrix_element = legacy_report._parse_generated_library_color_probe_output(
        validation_result.stdout + "\n" + validation_result.stderr,
        expected_accuracy=Accuracy.FULL.value,
        expected_group=int(context.entry.group),
        expected_integral=int(context.entry.integral),
    )
    if not _positive_finite(matrix_element):
        raise StudyError("AmpliCol generated library returned a zero/non-finite sum")
    numerical = _numerical_evidence(
        family=family,
        final_multiplicity=final_multiplicity,
        alpha_s=arguments.alpha_s,
        observed=(matrix_element,),
        baseline=baseline,
    )
    if numerical["passes"] is False:
        raise StudyError(
            "AmpliCol generated-library numerical comparison failed: relative "
            f"error {numerical['maximum_relative_error']:.6g}"
        )
    helicity_coverage_count = 2 ** len(context.source_pdgs)
    warm_seconds_per_point = profile.seconds / profile.points
    return base | {
        "status": "measured",
        "helicity": list(helicity),
        "event_paths": [str(path) for path in events],
        "point_values": [matrix_element],
        "metrics": {
            "generation_seconds": generation_seconds,
            "warm_seconds_per_point": warm_seconds_per_point,
            "max_rss_kib": _require_bounded_rss(
                executor.maximum_rss_kib,
                arguments.memory_limit_gib,
            ),
        },
        "helicity_workload": "sum",
        "warm_fixed_helicity": False,
        "warm_helicity_sum": True,
        "helicity_coverage_count": helicity_coverage_count,
        "helicity_count_semantics": {
            "coverage": "complete physical input-helicity axis",
            "timed": None,
            "timed_unavailable_reason": (
                "the create-raw probe prunes combinations with no mapped colour-"
                "row amplitude but does not expose its retained count"
            ),
        },
        "helicity_compaction": (
            "probe-local no-mapped-amplitude pruning only; create-raw does not "
            "apply the generated library's hel_fac filter"
        ),
        "generated_library": str(generated),
        "runtime_profile": {
            "points": profile.points,
            "seconds": profile.seconds,
            "standard_error_seconds_per_point": (
                profile.standard_error_seconds_per_point
            ),
            "relative_standard_error": profile.relative_standard_error,
            "record": dict(profile.record),
            "warmup": dict(profile.warmup_record),
        },
        "resource_peaks_kib": {
            "maximum_child_rss": executor.maximum_rss_kib,
            "maximum_child_guard": executor.maximum_guard_kib,
        },
        "process_entry": asdict(context.entry)
        | {"matching_row_count": context.matching_rows},
        "numerical": numerical,
        "provenance": {
            "implementation": (
                "original-amplicol-create-raw-bulk-family-complete-helicity-sum"
            ),
            "helicity_selector": None,
            "probe_source_overlay": probe_source_overlay,
            "commands": commands,
        },
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
    if arguments.compare_helicity_sums:
        return _amplicol_summed_cell(
            arguments=arguments,
            family=family,
            final_multiplicity=final_multiplicity,
            events=events,
            helicity=helicity,
            baseline=baseline,
            run_root=run_root,
            environment=environment,
        )
    base = _cell_base(
        family,
        MODE_BY_KEY["amplicol"],
        final_multiplicity,
        sum_helicities=arguments.compare_helicity_sums,
    )
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
        memory_limit_gib=arguments.memory_limit_gib,
        watchdog_report_path=cell_root / "process-generation-watchdog.json",
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

    def probe_command(points: int) -> tuple[str, ...]:
        return _exec_in_command(
            arguments.python,
            setup,
            (
                str((arguments.amplicol_repository / "amplicol_color_probe").resolve()),
                str(points),
                str(entry.group),
                str(entry.integral),
                "full",
                str(process_file),
                str(momentum_path),
                *(str(value) for value in ordered_helicity),
            ),
        )

    probe_wall_budget = min(
        arguments.runtime_timeout,
        arguments.generation_timeout - generation.elapsed_seconds,
    )
    if probe_wall_budget <= 0.0:
        raise _CellResourceLimitError(
            "AmpliCol process generation exhausted the generation cap",
            category="generation-time-limit",
        )
    probe_wall_elapsed = 0.0
    runtime_self_rss_kib = 0
    runtime_watchdog_rss_kib = 0
    runtime_guard_rss_kib = 0

    def run_probe(
        points: int, probe_environment: Mapping[str, str], log_path: Path
    ) -> performance.WatchedCompletedProcess:
        nonlocal probe_wall_elapsed
        nonlocal runtime_guard_rss_kib
        nonlocal runtime_self_rss_kib
        nonlocal runtime_watchdog_rss_kib
        remaining = probe_wall_budget - probe_wall_elapsed
        if remaining <= 0.0:
            raise _CellResourceLimitError(
                "AmpliCol warm sampling exhausted the per-cell time cap",
                category="setup-or-runtime-time-limit",
            )
        timed_command, normalizer = _timed_command(
            probe_command(points), python=arguments.python
        )
        try:
            completed = performance._run_watched(
                timed_command,
                python=arguments.python,
                environment=probe_environment,
                timeout_seconds=remaining,
                log_path=log_path,
                memory_limit_gib=arguments.memory_limit_gib,
                watchdog_report_path=log_path.with_suffix(".watchdog.json"),
                normalize_completed=normalizer,
            )
        except performance.AcceptanceError as error:
            raise _CellCommandError(
                f"AmpliCol setup/runtime probe failed: {error}",
                timeout_category="setup-or-runtime-time-limit",
            ) from error
        probe_wall_elapsed += completed.elapsed_seconds
        if completed.peak_rss_kib is None or completed.peak_guard_kib is None:
            raise StudyError("AmpliCol runtime watchdog peak evidence is missing")
        runtime_self_rss_kib = max(
            runtime_self_rss_kib, _parse_rss_marker(completed.stderr)
        )
        runtime_watchdog_rss_kib = max(runtime_watchdog_rss_kib, completed.peak_rss_kib)
        runtime_guard_rss_kib = max(runtime_guard_rss_kib, completed.peak_guard_kib)
        return completed

    adaptive_environment = dict(environment)
    adaptive_environment["AMPICOL_COLOR_PROBE_TARGET_RUNTIME_S"] = (
        f"{arguments.target_seconds:.17g}"
    )
    completed = run_probe(
        AMPLI_COL_MAX_POINTS, adaptive_environment, cell_root / "probe.json"
    )
    generation_setup, seed_total_seconds, adaptive_points = (
        _parse_amplicol_timing_aggregate(completed.stdout)
    )
    generation_seconds = generation.elapsed_seconds + generation_setup
    if generation_seconds >= arguments.generation_timeout:
        raise _CellResourceLimitError(
            "AmpliCol generation reached the strict generation cap",
            category="generation-time-limit",
        )

    fixed_environment = dict(environment)
    fixed_environment.pop("AMPICOL_COLOR_PROBE_TARGET_RUNTIME_S", None)

    def run_sample(sample_set: int, sample: int, repetitions: int) -> str:
        return run_probe(
            repetitions,
            fixed_environment,
            cell_root / f"warm-sample-set-{sample_set:02d}-sample-{sample:02d}.json",
        ).stdout

    warm = _collect_amplicol_warm_samples(
        seed_points=adaptive_points,
        seed_total_seconds=seed_total_seconds,
        target_seconds=arguments.target_seconds,
        run_sample=run_sample,
    )
    warm_per_point = statistics.median(warm.samples_seconds)
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
    if generation.peak_rss_kib is None or generation.peak_guard_kib is None:
        raise StudyError("AmpliCol generation watchdog peak evidence is missing")
    generation_rss_kib = generation.peak_rss_kib
    generation_guard_rss_kib = generation.peak_guard_kib
    max_rss_kib = _require_bounded_rss(
        max(runtime_self_rss_kib, generation_rss_kib), arguments.memory_limit_gib
    )
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
        "adaptive_runtime_points": adaptive_points,
        "warm_repetitions": [warm.repetitions] * WARM_SAMPLES,
        "warm_samples_seconds": list(warm.samples_seconds),
        "warm_sample_totals_seconds": list(warm.sample_totals_seconds),
        "warm_sample_set_attempts": warm.sample_set_attempts,
        "warm_calibration_seed": {
            "points": adaptive_points,
            "total_seconds": seed_total_seconds,
            "seconds_per_point": seed_total_seconds / adaptive_points,
        },
        "resource_peaks_kib": {
            "generation": generation_rss_kib,
            "generation_guard": generation_guard_rss_kib,
            "runtime_self": runtime_self_rss_kib,
            "runtime_watchdog": runtime_watchdog_rss_kib,
            "runtime_guard": runtime_guard_rss_kib,
            "runtime_watchdog_included_in_plotted_rss": False,
        },
        "process_entry": asdict(entry) | {"matching_row_count": len(matches)},
        "numerical": numerical,
    }


def _load_successful_candidate_generation(
    cell_root: Path,
) -> performance.WatchedCompletedProcess | None:
    """Reuse one completed generation phase after a later probe failure."""

    log_path = cell_root / "generation.json"
    if not (cell_root / "artifact" / "artifact.json").is_file():
        return None
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    command = payload.get("command")
    elapsed_seconds = payload.get("elapsed_seconds")
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    timeout_cleanup = payload.get("timeout_cleanup")
    if (
        payload.get("returncode") != 0
        or payload.get("timed_out") is not False
        or not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or not _positive_finite(elapsed_seconds)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or not isinstance(timeout_cleanup, str)
    ):
        return None
    return performance.WatchedCompletedProcess(
        command,
        0,
        stdout,
        stderr,
        elapsed_seconds=float(elapsed_seconds),
        log_write_seconds=0.0,
        timed_out=False,
        timeout_cleanup=timeout_cleanup,
    )


def _candidate_cli_failure_allows_generation_reuse(
    cell: Mapping[str, object],
) -> bool:
    reason = cell.get("failure_reason")
    return isinstance(reason, str) and reason.startswith(
        ("candidate CLI evaluate", "candidate CLI profile", "OTF CLI profile")
    )


def _symbolica_generation_lock_path(
    environment: Mapping[str, str], *, force: bool = False
) -> Path | None:
    configured = environment.get(SYMBOLICA_GENERATION_LOCK_ENV)
    if configured is not None:
        if configured.strip().lower() in {"", "0", "false", "no", "off", "none"}:
            return None
        return Path(configured).expanduser().resolve(strict=False)
    if not force and environment.get("SYMBOLICA_LICENSE", "").strip():
        return None
    uid = str(os.getuid()) if hasattr(os, "getuid") else "unknown"
    root = Path("/tmp") if os.name == "posix" else Path(tempfile.gettempdir())
    return root / f"pyamplicol-symbolica-generation-{uid}.lock"


@contextlib.contextmanager
def _bounded_symbolica_generation_lock(
    lock_path: Path | None, timeout_seconds: float
) -> Iterator[None]:
    if lock_path is None:
        yield
        return
    deadline = time.monotonic() + timeout_seconds
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise _CellResourceLimitError(
                        "candidate artifact generation waited for the Symbolica "
                        "generation lock until the strict generation cap",
                        category="generation-time-limit",
                    ) from error
                time.sleep(min(SYMBOLICA_LOCK_POLL_SECONDS, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _symbolica_license_busy_log(log_path: Path) -> bool:
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    output = "\n".join(
        str(payload.get(field, "")) for field in ("stdout", "stderr")
    ).lower()
    return SYMBOLICA_LICENSE_BUSY_FRAGMENT in output


def _candidate_generation_time_exhausted() -> _CellResourceLimitError:
    return _CellResourceLimitError(
        "candidate artifact generation exhausted the strict generation cap",
        category="generation-time-limit",
    )


def _run_candidate_generation(
    *,
    arguments: argparse.Namespace,
    family: str,
    final_multiplicity: int,
    mode: Mode,
    artifact: Path,
    cell_root: Path,
    environment: Mapping[str, str],
) -> performance.WatchedCompletedProcess:
    command = _candidate_generation_command(
        python=arguments.python,
        family=family,
        final_multiplicity=final_multiplicity,
        mode=mode,
        artifact=artifact,
        batch_size=arguments.batch_size,
        optimization_cores=arguments.optimization_cores,
    )
    log_path = cell_root / "generation.json"
    started = time.monotonic()
    lock_path = _symbolica_generation_lock_path(environment)
    attempts = 0
    maximum_attempts = (
        max(2, math.ceil(arguments.generation_timeout / SYMBOLICA_BUSY_RETRY_SECONDS))
        + 1
    )
    while True:
        attempts += 1
        remaining = arguments.generation_timeout - (time.monotonic() - started)
        if remaining <= 0.0:
            raise _candidate_generation_time_exhausted()
        try:
            with _bounded_symbolica_generation_lock(lock_path, remaining):
                remaining = arguments.generation_timeout - (time.monotonic() - started)
                if remaining <= 0.0:
                    raise _candidate_generation_time_exhausted()
                return performance._run_watched(
                    command,
                    python=arguments.python,
                    environment=environment,
                    timeout_seconds=remaining,
                    log_path=log_path,
                    memory_limit_gib=arguments.memory_limit_gib,
                )
        except performance.AcceptanceError as error:
            if not _symbolica_license_busy_log(log_path):
                raise
            if attempts >= maximum_attempts:
                raise _candidate_generation_time_exhausted() from error
            lock_path = lock_path or _symbolica_generation_lock_path(
                environment, force=True
            )
            remaining = arguments.generation_timeout - (time.monotonic() - started)
            if remaining <= SYMBOLICA_BUSY_RETRY_SECONDS:
                raise _candidate_generation_time_exhausted() from error
            time.sleep(min(SYMBOLICA_BUSY_RETRY_SECONDS, remaining))


def _validate_publication_candidate_artifact(
    artifact: Path,
    process: str,
) -> int:
    """Require an authoritative all-runtime-helicity artifact contract."""

    execution_path = artifact / "processes" / process / "execution.json"
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        execution = None
    on_the_fly_kind = "pyamplicol-runtime-on-the-fly-execution"
    if isinstance(execution, Mapping) and execution.get("kind") == on_the_fly_kind:
        try:
            selector_policy = execution["selector_policy"]
            selector_census = selector_policy["selector_census"]
            physical_helicity_count = selector_census["physical_helicity_count"]
            runtime_metadata = execution["runtime_metadata"]
            process_seed_identity = runtime_metadata["process_seed_identity"]
            external_sources = process_seed_identity["external_sources"]
        except (KeyError, TypeError) as error:
            raise StudyError(
                "candidate on-the-fly artifact has a malformed helicity contract"
            ) from error
        if (
            not isinstance(execution.get("schema_version"), int)
            or isinstance(execution.get("schema_version"), bool)
            or execution.get("schema_version") != 3
            or execution.get("kind") != on_the_fly_kind
            or not isinstance(selector_policy, Mapping)
            or selector_policy.get("color_coverage") != "contracted"
            or not isinstance(selector_census, Mapping)
            or not isinstance(physical_helicity_count, int)
            or isinstance(physical_helicity_count, bool)
            or physical_helicity_count <= 1
            or not isinstance(runtime_metadata, Mapping)
            or not isinstance(process_seed_identity, Mapping)
            or not isinstance(external_sources, list)
            or not external_sources
        ):
            raise StudyError(
                "candidate on-the-fly artifact has an invalid helicity contract"
            )

        derived_helicity_count = 1
        source_slots: set[int] = set()
        for source_index, source in enumerate(external_sources):
            if not isinstance(source, Mapping):
                raise StudyError(
                    "candidate on-the-fly artifact has a malformed source identity"
                )
            source_slot = source.get("source_slot")
            states = source.get("states")
            if (
                not isinstance(source_slot, int)
                or isinstance(source_slot, bool)
                or source_slot != source_index
                or source_slot in source_slots
                or not isinstance(states, list)
                or not states
            ):
                raise StudyError(
                    "candidate on-the-fly artifact has an invalid source identity"
                )
            source_slots.add(source_slot)
            state_indices: set[int] = set()
            public_helicities: set[int] = set()
            for state in states:
                if not isinstance(state, Mapping):
                    raise StudyError(
                        "candidate on-the-fly artifact has a malformed source state"
                    )
                state_index = state.get("state_index")
                public_helicity = state.get("public_helicity")
                if (
                    not isinstance(state_index, int)
                    or isinstance(state_index, bool)
                    or state_index < 0
                    or state_index in state_indices
                    or not isinstance(public_helicity, int)
                    or isinstance(public_helicity, bool)
                    or public_helicity in public_helicities
                ):
                    raise StudyError(
                        "candidate on-the-fly artifact has an invalid source state"
                    )
                state_indices.add(state_index)
                public_helicities.add(public_helicity)
            derived_helicity_count *= len(states)
        if derived_helicity_count != physical_helicity_count:
            raise StudyError(
                "candidate on-the-fly artifact helicity census disagrees with "
                "its process seed"
            )
        return physical_helicity_count

    physics_path = artifact / "processes" / process / "physics.json"
    try:
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        coverage = physics["coverage"]
        selectors = physics["extensions"]["runtime_selectors"]
        helicity_axis = selectors["axes"]["helicity"]
        specialized_axes = selectors["generation_specialized_axes"]
        helicities = physics["helicities"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise StudyError(
            "candidate artifact has no readable helicity-coverage contract"
        ) from error
    if (
        not isinstance(physics, Mapping)
        or not isinstance(coverage, Mapping)
        or not isinstance(selectors, Mapping)
        or not isinstance(helicity_axis, Mapping)
        or not isinstance(specialized_axes, list)
        or not isinstance(helicities, list)
        or len(helicities) <= 1
        or coverage.get("helicities") != "complete"
        or helicity_axis.get("generation_coverage") != "complete"
        or helicity_axis.get("generation_selection") != {}
        or helicity_axis.get("runtime_contract") != "complete-reusable"
        or "helicity" in specialized_axes
    ):
        raise StudyError(
            "candidate artifact is not reusable for all runtime helicities"
        )
    return len(helicities)


def _candidate_cell(
    *,
    arguments: argparse.Namespace,
    family: str,
    final_multiplicity: int,
    mode: Mode,
    events: Sequence[Path],
    helicity: Sequence[int],
    baseline: Mapping[str, object] | None,
    cell_root: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    assert mode.execution_mode is not None
    base = _cell_base(
        family,
        mode,
        final_multiplicity,
        sum_helicities=arguments.compare_helicity_sums,
    )
    artifact = cell_root / "artifact"
    generation = _load_successful_candidate_generation(cell_root)
    if generation is None:
        generation = _run_candidate_generation(
            arguments=arguments,
            family=family,
            final_multiplicity=final_multiplicity,
            mode=mode,
            artifact=artifact,
            cell_root=cell_root,
            environment=environment,
        )
    generation_helicity_count = _validate_publication_candidate_artifact(
        artifact,
        process_key(family, final_multiplicity),
    )
    momenta_path, parameters_path = _write_candidate_cli_inputs(
        cell_root=cell_root,
        events=events,
        alpha_s=arguments.alpha_s,
    )
    evaluate_command = _candidate_evaluate_command(
        python=arguments.python,
        artifact=artifact,
        process=process_key(family, final_multiplicity),
        momenta=momenta_path,
        model_parameters=parameters_path,
        helicity=helicity,
        sum_helicities=arguments.compare_helicity_sums,
    )
    evaluate_timeout = arguments.generation_timeout - generation.elapsed_seconds
    if evaluate_timeout <= 0.0:
        raise _CellResourceLimitError(
            "candidate artifact generation exhausted the strict generation cap",
            category="generation-time-limit",
        )
    try:
        evaluated = performance._run_watched(
            evaluate_command,
            python=arguments.python,
            environment=environment,
            timeout_seconds=evaluate_timeout,
            log_path=cell_root / "evaluate.json",
            memory_limit_gib=arguments.memory_limit_gib,
        )
    except performance.AcceptanceError as error:
        raise _CellCommandError(
            f"candidate CLI evaluate failed: {error}",
            timeout_category="generation-time-limit",
        ) from error
    values = _parse_candidate_evaluate_json(evaluated.stdout)
    generation_seconds = generation.elapsed_seconds + evaluated.elapsed_seconds
    if generation_seconds >= arguments.generation_timeout:
        raise _CellResourceLimitError(
            "candidate generation/fresh CLI evaluation reached the strict "
            "generation cap",
            category="generation-time-limit",
        )
    generation_rss_kib = _parse_watchdog_peak_rss_kib(generation.stderr)
    generation_guard_rss_kib = _parse_watchdog_peak_guard_kib(generation.stderr)
    evaluate_rss_kib = _parse_watchdog_peak_rss_kib(evaluated.stderr)
    evaluate_guard_rss_kib = _parse_watchdog_peak_guard_kib(evaluated.stderr)
    if not any(abs(float(value)) > 1.0e-300 for value in values):
        raise StudyError("candidate CLI evaluation is structurally zero")
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
    profile_command = _candidate_profile_command(
        python=arguments.python,
        artifact=artifact,
        process=process_key(family, final_multiplicity),
        momenta=momenta_path,
        helicity=helicity,
        sum_helicities=arguments.compare_helicity_sums,
        target_seconds=arguments.target_seconds,
        batch_size=arguments.batch_size,
    )
    try:
        profiled = performance._run_watched(
            profile_command,
            python=arguments.python,
            environment=environment,
            timeout_seconds=arguments.runtime_timeout,
            log_path=cell_root / "profile.json",
            memory_limit_gib=arguments.memory_limit_gib,
        )
    except performance.AcceptanceError as error:
        raise _CellCommandError(
            f"candidate CLI profile failed: {error}",
            timeout_category="runtime-time-limit",
        ) from error
    profile = _parse_candidate_profile_json(
        profiled.stdout,
        artifact=artifact,
        family=family,
        final_multiplicity=final_multiplicity,
        execution_mode=mode.execution_mode,
        helicity=helicity,
        sum_helicities=arguments.compare_helicity_sums,
        target_seconds=arguments.target_seconds,
        batch_size=arguments.batch_size,
    )
    profile_rss_kib = _parse_watchdog_peak_rss_kib(profiled.stderr)
    profile_guard_rss_kib = _parse_watchdog_peak_guard_kib(profiled.stderr)
    measured_rss_kib = max(
        generation_rss_kib,
        evaluate_rss_kib,
        profile_rss_kib,
    )
    return base | {
        "status": "measured",
        "helicity": list(helicity),
        "event_paths": [str(path) for path in events],
        "artifact": str(artifact),
        "point_values": values,
        "metrics": {
            "generation_seconds": generation_seconds,
            "warm_seconds_per_point": profile["wall_time_per_point"],
            "max_rss_kib": _require_bounded_rss(
                measured_rss_kib,
                arguments.memory_limit_gib,
            ),
        },
        "cli_evaluate": {
            "command": list(evaluate_command),
            "elapsed_seconds": evaluated.elapsed_seconds,
            "result": json.loads(evaluated.stdout),
            "model_parameters": {
                "normalization.alpha_s_me_check": arguments.alpha_s
            },
        },
        "runtime_profile_command": list(profile_command),
        "runtime_profile": profile,
        "resource_peaks_kib": {
            "generation": generation_rss_kib,
            "generation_guard": generation_guard_rss_kib,
            "evaluate_watchdog": evaluate_rss_kib,
            "evaluate_guard": evaluate_guard_rss_kib,
            "profile_watchdog": profile_rss_kib,
            "profile_guard": profile_guard_rss_kib,
            "all_process_tree_peaks_included_in_plotted_rss": True,
        },
        "numerical": numerical,
        "generation_helicity_coverage": "all",
        "generation_helicity_count": generation_helicity_count,
        **(
            {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
                "timed_helicity_count": generation_helicity_count,
                "helicity_count_semantics": {
                    "coverage": "complete physical runtime input-helicity axis",
                    "timed": (
                        "complete physical axis requested once through the null "
                        "helicity selector; internal structural-zero elimination "
                        "does not change the API workload"
                    ),
                },
            }
            if arguments.compare_helicity_sums
            else {"warm_fixed_helicity": True}
        ),
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
    raw_events = reference.get("shared_event_paths", reference.get("event_paths"))
    legacy_events = reference.get("event_paths")
    raw_helicity = reference.get("helicity")
    if (
        not isinstance(raw_events, list)
        or not isinstance(legacy_events, list)
        or raw_events != legacy_events
        or not isinstance(raw_helicity, list)
    ):
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


def _prepare_support_directories(study_root: Path | None = None) -> None:
    study_root = STUDY_ROOT if study_root is None else study_root
    cache_root = study_root / "cache"
    for path in (
        study_root / "tmp",
        cache_root / "cargo-home",
        cache_root / "cargo-target",
        cache_root / "pip-cache",
        cache_root / "xdg-cache",
        cache_root / "python-cache",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _study_environment(python: str, study_root: Path | None = None) -> dict[str, str]:
    study_root = STUDY_ROOT if study_root is None else study_root
    environment = performance._workspace_environment(python)
    cache_root = study_root / "cache"
    environment.update(
        {
            "TMPDIR": str(study_root / "tmp"),
            "CARGO_HOME": str(cache_root / "cargo-home"),
            "CARGO_TARGET_DIR": str(cache_root / "cargo-target"),
            "PIP_CACHE_DIR": str(cache_root / "pip-cache"),
            "XDG_CACHE_HOME": str(cache_root / "xdg-cache"),
            "PYTHONPYCACHEPREFIX": str(cache_root / "python-cache"),
        }
    )
    return environment


def _clear_campaign_cell(path: Path) -> None:
    """Remove only this campaign's exact stale cell/staging path."""

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _candidate_staging_root(cell_root: Path) -> Path:
    return cell_root.with_name(f".{cell_root.name}.staging")


def _adopt_interrupted_candidate_generation(
    *,
    staging_root: Path,
    cell_root: Path,
    expected_command: Sequence[str],
    process: str,
) -> bool:
    """Atomically retain an interrupted cell's verified generation phase."""

    if cell_root.exists() or not staging_root.is_dir():
        return False
    generation = _load_successful_candidate_generation(staging_root)
    if generation is None or tuple(generation.args) != tuple(expected_command):
        return False
    try:
        _validate_publication_candidate_artifact(staging_root / "artifact", process)
    except StudyError:
        return False
    staging_root.replace(cell_root)
    return True


def _discard_candidate_artifact(cell_root: Path) -> None:
    """Keep bounded failure logs but not an unpublishable artifact payload."""

    artifact = cell_root / "artifact"
    if artifact.is_dir():
        shutil.rmtree(artifact)
    elif artifact.exists():
        artifact.unlink()


def _promote_candidate_cell(
    staging_root: Path,
    cell_root: Path,
    cell: dict[str, object],
) -> None:
    """Atomically publish one completed cell and normalize its artifact path."""

    if not staging_root.is_dir() or cell_root.exists():
        raise StudyError("candidate cell staging cannot be promoted atomically")
    artifact = cell.get("artifact")
    if isinstance(artifact, str):
        expected = staging_root / "artifact"
        if Path(artifact) != expected:
            raise StudyError("candidate cell recorded an unexpected staged artifact")
    staging_root.replace(cell_root)
    if isinstance(artifact, str):
        cell["artifact"] = str(cell_root / "artifact")


def _acquire_campaign_lock(lock_root: Path | None = None) -> Any:
    """Prevent two scaling-study campaign drivers from overlapping."""

    lock_root = STUDY_ROOT if lock_root is None else lock_root
    lock_path = lock_root / ".campaign.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise StudyError(f"another campaign driver holds {lock_path}") from error
    return handle


def _campaign(arguments: argparse.Namespace) -> dict[str, object]:
    global STUDY_ROOT
    study_root = _study_root(arguments)
    # A driver process owns one campaign.  Resolve the selected root once so
    # all legacy no-argument helper seams and their lock agree with --study-root.
    STUDY_ROOT = study_root
    run_root = study_root / "runs" / arguments.run_id
    report_path = run_root / "report.json"
    run_root.mkdir(parents=True, exist_ok=True)
    campaign_lock = _acquire_campaign_lock()
    report = _load_report(report_path, arguments)
    performance._write_report(report_path, report)
    _prepare_support_directories()
    environment = _study_environment(arguments.python)
    os.environ.update(environment)
    multiplicities = _multiplicities(arguments)
    fill_multiplicities = _fill_multiplicities(arguments)
    families = _selected_families(arguments)
    modes_by_family = {
        family: _selected_modes(arguments, family) for family in families
    }
    selected_modes = {
        mode.key: mode for modes in modes_by_family.values() for mode in modes
    }

    if any(mode.kind == "amplicol" for mode in selected_modes.values()):
        legacy_amplicol.validate_checkout(arguments.amplicol_repository)
        amplicol_probe = arguments.amplicol_repository / "amplicol_color_probe"
        if not arguments.compare_helicity_sums and not amplicol_probe.is_file():
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
                memory_limit_gib=arguments.memory_limit_gib,
            )

    reference_module = (
        performance._load_reference_module()
        if any(mode.kind == "reference" for mode in selected_modes.values())
        else None
    )
    for final_multiplicity in fill_multiplicities:
        for family in families:
            for mode in modes_by_family[family]:
                curve = _curve(report, family, mode.key)
                candidate_cell_root = (
                    run_root
                    / "candidate"
                    / mode.key
                    / family
                    / f"n{final_multiplicity}"
                    if mode.kind == "candidate"
                    else None
                )
                reuse_candidate_generation = False
                existing = curve.get(str(final_multiplicity))
                if isinstance(existing, dict):
                    higher_measurement_exists = any(
                        isinstance(other, dict)
                        and other.get("status") == "measured"
                        and int(other_n) > final_multiplicity
                        for other_n, other in curve.items()
                        if str(other_n).isdigit()
                    )
                    retry_lower_resource_frontier = (
                        existing.get("status") == "failed"
                        and existing.get("censors_higher_multiplicities") is True
                        and higher_measurement_exists
                    )
                    if (
                        existing.get("status") == "failed"
                        and (
                            existing.get("censors_higher_multiplicities") is False
                            or retry_lower_resource_frontier
                        )
                    ):
                        reuse_candidate_generation = (
                            not retry_lower_resource_frontier
                            and candidate_cell_root is not None
                            and _candidate_cli_failure_allows_generation_reuse(
                                existing
                            )
                            and _load_successful_candidate_generation(
                                candidate_cell_root
                            )
                            is not None
                        )
                        del curve[str(final_multiplicity)]
                        performance._write_report(report_path, report)
                    else:
                        continue
                censored_at: int | None = next(
                    (
                        n
                        for n in multiplicities
                        if isinstance(curve.get(str(n)), dict)
                        and curve[str(n)].get("censors_higher_multiplicities") is True
                    ),
                    None,
                )
                base = _cell_base(
                    family,
                    mode,
                    final_multiplicity,
                    sum_helicities=arguments.compare_helicity_sums,
                )
                if censored_at is not None and final_multiplicity > censored_at:
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
                                "curve censored after resource limit at "
                                f"n={censored_at}"
                            ),
                            "failed_at_n": censored_at,
                            "censors_higher_multiplicities": True,
                        },
                    )
                    continue
                compiled_fft_na = _compiled_fft_not_applicable_cell(
                    family=family,
                    mode=mode,
                    final_multiplicity=final_multiplicity,
                    sum_helicities=arguments.compare_helicity_sums,
                )
                if compiled_fft_na is not None:
                    _record(
                        report,
                        report_path,
                        family,
                        mode.key,
                        final_multiplicity,
                        compiled_fft_na,
                    )
                    continue
                otf_scope = otf_protocol_scope_cell(
                    family=family,
                    mode=mode,
                    final_multiplicity=final_multiplicity,
                    sum_helicities=arguments.compare_helicity_sums,
                )
                if otf_scope is not None:
                    _record(
                        report,
                        report_path,
                        family,
                        mode.key,
                        final_multiplicity,
                        otf_scope,
                    )
                    continue
                if family == "gg" and mode.kind != "reference":
                    dependency_skip = _gg_reference_dependency_skip(
                        report, final_multiplicity
                    )
                    if dependency_skip is not None:
                        _record(
                            report,
                            report_path,
                            family,
                            mode.key,
                            final_multiplicity,
                            base | dependency_skip,
                        )
                        continue
                if family == "gg" and mode.kind == "amplicol":
                    preflight = _amplicol_gg_dense_index_preflight(
                        final_multiplicity, arguments.memory_limit_gib
                    )
                    if preflight["feasible"] is False:
                        lower_bound_gib = int(preflight["lower_bound_bytes"]) / 1024**3
                        _record(
                            report,
                            report_path,
                            family,
                            mode.key,
                            final_multiplicity,
                            base
                            | {
                                "status": "skipped",
                                "failure_category": "structural-memory-limit",
                                "failure_reason": (
                                    "AmpliCol dense symmetric color-index lower bound "
                                    f"is {lower_bound_gib:.3f} GiB, above the "
                                    f"{arguments.memory_limit_gib:g} GiB cap"
                                ),
                                "censors_higher_multiplicities": True,
                                "preflight": preflight,
                            },
                        )
                        continue
                if mode.kind == "reference":
                    cell_root = run_root / "reference" / f"N{final_multiplicity + 2}"
                    candidate_staging_root = None
                elif mode.kind == "amplicol":
                    cell_root = (
                        run_root / "amplicol" / family / f"n{final_multiplicity}"
                    )
                    candidate_staging_root = None
                else:
                    assert candidate_cell_root is not None
                    interrupted_staging_root = _candidate_staging_root(
                        candidate_cell_root
                    )
                    if not reuse_candidate_generation and arguments.resume:
                        reuse_candidate_generation = (
                            _adopt_interrupted_candidate_generation(
                                staging_root=interrupted_staging_root,
                                cell_root=candidate_cell_root,
                                expected_command=_candidate_generation_command(
                                    python=arguments.python,
                                    family=family,
                                    final_multiplicity=final_multiplicity,
                                    mode=mode,
                                    artifact=interrupted_staging_root / "artifact",
                                    batch_size=arguments.batch_size,
                                    optimization_cores=arguments.optimization_cores,
                                ),
                                process=process_key(family, final_multiplicity),
                            )
                        )
                    candidate_staging_root = (
                        None if reuse_candidate_generation else interrupted_staging_root
                    )
                    cell_root = candidate_staging_root or candidate_cell_root
                try:
                    if not reuse_candidate_generation:
                        if candidate_cell_root is not None:
                            _clear_campaign_cell(candidate_cell_root)
                            assert candidate_staging_root is not None
                            _clear_campaign_cell(candidate_staging_root)
                            candidate_staging_root.mkdir(parents=True)
                        else:
                            _clear_campaign_cell(cell_root)
                    if mode.kind == "reference":
                        assert reference_module is not None
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
                                cell_root=cell_root,
                                environment=environment,
                            )
                    if candidate_staging_root is not None:
                        assert candidate_cell_root is not None
                        _promote_candidate_cell(
                            candidate_staging_root,
                            candidate_cell_root,
                            cell,
                        )
                except (
                    StudyError,
                    performance.AcceptanceError,
                    legacy_amplicol.LegacyOracleError,
                    legacy_report.LegacyAdapterError,
                    legacy_report.ProfilingTimeLimitError,
                    OSError,
                    ValueError,
                ) as error:
                    cell = _failure_cell(base, error, cell_root)
                    if candidate_staging_root is not None:
                        assert candidate_cell_root is not None
                        if (
                            cell["censors_higher_multiplicities"] is True
                            or not _candidate_cli_failure_allows_generation_reuse(
                                cell
                            )
                        ):
                            _discard_candidate_artifact(candidate_staging_root)
                        if candidate_staging_root.is_dir():
                            _promote_candidate_cell(
                                candidate_staging_root,
                                candidate_cell_root,
                                cell,
                            )
                    if cell["censors_higher_multiplicities"] is False:
                        report["status"] = "stopped-correctness-failure"
                        report["failure_count"] = _failure_count(report) + 1
                        _record(
                            report,
                            report_path,
                            family,
                            mode.key,
                            final_multiplicity,
                            cell,
                        )
                        raise StudyError(
                            "non-resource failure recorded at "
                            f"{family}/{mode.key}/n{final_multiplicity}; "
                            "repair it, then resume to retry that cell"
                        ) from error
                _record(
                    report,
                    report_path,
                    family,
                    mode.key,
                    final_multiplicity,
                    cell,
                )

    failures = _failure_count(report)
    inversions = resource_frontier_inversions(report)
    if inversions:
        report["resource_frontier_inversions"] = [list(item) for item in inversions]
    else:
        report.pop("resource_frontier_inversions", None)
    policy_complete = all(
        str(final_multiplicity) in _curve(report, family, mode.key)
        for final_multiplicity in multiplicities
        for family in families
        for mode in modes_by_family[family]
    )
    report["status"] = (
        ("complete-with-failures" if failures else "complete")
        if policy_complete
        else "running"
    )
    report.pop("status_reason", None)
    report["failure_count"] = failures
    performance._write_report(report_path, report)
    campaign_lock.close()
    return report


def dry_run_plan(arguments: argparse.Namespace) -> dict[str, object]:
    multiplicities = _multiplicities(arguments)
    families = _selected_families(arguments)
    modes_by_family = {
        family: _selected_modes(arguments, family) for family in families
    }
    selected_candidate_modes = {
        mode.key: mode
        for modes in modes_by_family.values()
        for mode in modes
        if mode.kind == "candidate" and mode.key != "compiled-fft"
    }
    contractions_by_execution_mode = {
        execution_mode: [
            contraction
            for contraction in ("direct", "symmetric-group-fft")
            if any(
                mode.execution_mode == execution_mode
                and mode.contraction == contraction
                for mode in selected_candidate_modes.values()
            )
        ]
        for execution_mode in ("recurrence", "on-the-fly", "compiled")
        if any(
            mode.execution_mode == execution_mode
            for mode in selected_candidate_modes.values()
        )
    }
    candidate_generation_metric = (
        "artifact-generation wall time plus a fresh public `pyamplicol evaluate` "
        "process loading the artifact and evaluating the ten shared points, "
        "including OTF selected-family warm-up when applicable"
    )
    candidate_warm_metric = (
        "public `pyamplicol profile --json` wall_time_per_point from true "
        f"{arguments.batch_size}-point calls cycling the ten shared points; two "
        f"warm-up calls, at least {WARM_SAMPLES} independent timed blocks, and "
        f"{arguments.target_seconds:g}s cumulative target runtime"
    )
    plan = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "run_root": str(_study_root(arguments) / "runs" / arguments.run_id),
        "fft_enabled": bool(arguments.fft_enabled),
        "selected_pyamplicol_color_contractions": contractions_by_execution_mode,
        "tools": {
            "python": arguments.python,
            "cxx": arguments.cxx,
            "fc": arguments.fc,
            "amplicol_repository": str(arguments.amplicol_repository.resolve()),
            "reference_fft_root": str(arguments.reference_fft_root.resolve()),
        },
        "process_families": {
            family: {
                "expression": (
                    "g g > g g + (n-2)*g" if family == "gg" else "d d~ > d d~ + (n-2)*g"
                ),
                "ratio_reference": ("reference-fft" if family == "gg" else "amplicol"),
                "modes": [mode.key for mode in modes_by_family[family]],
            }
            for family in families
        },
        "final_state_multiplicities": list(multiplicities),
        "total_external_particles": [n + 2 for n in multiplicities],
        "measurement": {
            "color_accuracy": "full",
            "n_definition": "final-state-particle-count",
            "generation_helicity_coverage": "all",
            "warm_fixed_helicity": True,
            "warm_samples": WARM_SAMPLES,
            "calibration_target_seconds": arguments.target_seconds,
            "candidate_profile_target_runtime_seconds": arguments.target_seconds,
            "candidate_profile_warmup_runs": PROFILE_WARMUP_RUNS,
            "candidate_profile_minimum_samples": WARM_SAMPLES,
            "warm_benchmark_batch_size": arguments.batch_size,
            "warm_sample_count": WARM_SAMPLES,
            "candidate_optimization_cores": (
                arguments.optimization_cores
                if arguments.optimization_cores is not None
                else "configured-default"
            ),
            "generation_timeout_seconds": arguments.generation_timeout,
            "runtime_timeout_seconds": arguments.runtime_timeout,
            "requested_memory_ceiling_gib": arguments.memory_limit_gib,
            "memory_watchdog_gib": arguments.memory_limit_gib,
            "memory_policy": (
                "per-cell-strictly-below-publication-ceiling"
                if arguments.memory_limit_gib == REQUESTED_MEMORY_LIMIT_GIB
                else "per-cell-strictly-below-configured-ceiling"
            ),
            "cell_admission_limits": {
                "generation_seconds": {
                    "operator": "<",
                    "limit": arguments.generation_timeout,
                },
                "runtime_seconds": {
                    "operator": "<",
                    "limit": arguments.runtime_timeout,
                },
                "peak_rss_gib": {
                    "operator": "<",
                    "limit": arguments.memory_limit_gib,
                },
            },
            "schedule_order": "multiplicity-then-family-then-mode",
            "alpha_s": arguments.alpha_s,
            "resource_failure_censors_higher_n_in_same_curve": True,
            "non_resource_failure_policy": "record-and-abort-for-repair",
            "amplicol_gg_dense_index_preflight": (
                "skip and censor only AmpliCol gg when "
                "2*((total_external-1)!)^2 bytes exceeds the memory watchdog cap"
            ),
            "compiled_fft_applicability": (
                "not applicable to this helicity-general publication campaign: "
                "the current compiled FFT backend is a generation-selected "
                "diagnostic lane; compiled-direct remains supported"
            ),
            "compiled_fft_enabled": False,
            "generation_metric": {
                "reference-fft": (
                    "compiler identity/build, event generation, helicity proxy/"
                    "selection, selected-event writing, initialization, and first pass"
                ),
                "amplicol": (
                    "process-list wall time plus Fortran CPU-time color-object setup"
                ),
                "pyamplicol": candidate_generation_metric,
            },
            "candidate_rss_metric": (
                "maximum process-tree peak RSS across public generation, fresh "
                "CLI evaluate, and fresh CLI profile children"
            ),
            "candidate_numerical_metric": (
                "ten-point public `pyamplicol evaluate --json` with the study "
                "alpha_s runtime parameter; fixed-H passes one stable selector and "
                "the summed workload omits the selector"
            ),
            "candidate_profile_model_parameters": (
                "artifact defaults; the profile CLI has no model-parameter option "
                "and timing control flow is value-independent at alpha_s=0.118"
            ),
            "amplicol_rss_metric": (
                "maximum of process-generation actual peak RSS and Fortran runtime "
                "child peak RSS; watchdog process-tree RSS/guard are cap telemetry"
            ),
            "warm_timing_metric": {
                "reference-fft": (
                    "median of 10 process-CPU samples after calibrating whole "
                    f"{arguments.batch_size}-call scalar repetition groups; "
                    "normalized per evaluation "
                    "(not a vectorized batch API)"
                ),
                "amplicol": (
                    "median of 10 independent process-CPU aggregates with one "
                    "fixed scalar repetition count; every retained aggregate meets "
                    "the calibration target and is normalized per evaluation"
                ),
                "pyamplicol": candidate_warm_metric,
            },
        },
    }
    if arguments.compare_helicity_sums:
        measurement = plan["measurement"]
        assert isinstance(measurement, dict)
        measurement.update(
            {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
                "candidate_timed_helicity_contract": (
                    "complete null-selector physical-helicity axis; authenticated "
                    "structural zeros may be eliminated internally"
                ),
                "helicity_count_semantics": {
                    "candidate": (
                        "coverage and timed count are the complete physical API "
                        "axis requested by a null helicity selector"
                    ),
                    "reference_fft": (
                        "coverage is the complete physical axis; active/timed is "
                        "the exact analytic-nonzero subset evaluated per sum"
                    ),
                    "amplicol": (
                        "coverage is the complete physical axis; create-raw "
                        "evaluates the all-helicity amplitude family in bulk and "
                        "the probe prunes no-row combinations, but it does not "
                        "expose that retained count or apply the hel_fac filter"
                    ),
                },
            }
        )
        generation_metric = measurement["generation_metric"]
        assert isinstance(generation_metric, dict)
        generation_metric["reference-fft"] = (
            "compiler identity/build, exhaustive event generation, exact "
            "nonzero-helicity sweep initialization, and first complete sum"
        )
        generation_metric["amplicol"] = (
            "process-list generation, raw contracted-library generation/build, "
            "and immutable generated-library snapshot"
        )
        warm_metric = measurement["warm_timing_metric"]
        assert isinstance(warm_metric, dict)
        warm_metric["reference-fft"] = (
            "median of 10 calibrated process-CPU samples of one complete "
            "analytic-nonzero helicity sum at the representative point"
        )
        warm_metric["amplicol"] = (
            "adaptive aggregate of at least 10 independently timed create-raw "
            "bulk-family evaluations followed by a complete physical-helicity "
            "sum with probe-local no-row pruning, normalized per point"
        )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compare-helicity-sums",
        action="store_true",
        help=(
            "measure the complete physical-helicity sum instead of the fixed "
            "selected-helicity workload; this is part of report/cache identity"
        ),
    )
    parser.add_argument("--run-id", default="fullcolor-n2-n7")
    parser.add_argument(
        "--study-root",
        type=Path,
        default=STUDY_ROOT,
        help=(
            "isolated cache/run root (default: repository FFT scaling raw root); "
            "use distinct roots for deliberately parallel campaign shards"
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=min(FINAL_MULTIPLICITIES),
        help="first plotted final-state multiplicity (default: 2)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=max(FINAL_MULTIPLICITIES),
        help="last plotted final-state multiplicity (default: 7)",
    )
    parser.add_argument(
        "--multiplicity",
        dest="multiplicities",
        action="append",
        type=int,
        help=(
            "exact final-state multiplicity to run; repeat for a sparse scan "
            "(overrides --min-n/--max-n)"
        ),
    )
    parser.add_argument(
        "--fill-multiplicity",
        dest="fill_multiplicities",
        action="append",
        type=int,
        help=(
            "populate only this multiplicity in the fixed --multiplicity policy; "
            "repeat for a bounded resumable fill (not part of report identity)"
        ),
    )
    parser.add_argument(
        "--family",
        dest="families",
        action="append",
        choices=FAMILIES,
        help="process family to run; repeat to select both (default: all)",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=tuple(MODE_BY_KEY),
        help="curve to run; repeat as needed (default: all applicable curves)",
    )
    parser.add_argument(
        "--fft",
        dest="fft_enabled",
        action="store_true",
        help=(
            "for every selected applicable pyAmpliCol execution mode, retain "
            "its direct curve and add its symmetric-group FFT curve"
        ),
    )
    parser.add_argument(
        "--explicit-modes",
        action="store_true",
        help="honor repeated --mode values exactly without FFT companion expansion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BENCHMARK_BATCH_SIZE,
        help=(
            "candidate public CLI profile batch size and Reference scalar "
            "repetition quantum (default: 128)"
        ),
    )
    parser.add_argument(
        "--optimization-cores",
        type=int,
        help=(
            "explicit evaluator.optimization.cores for candidate generation; "
            "the configured default is retained when omitted"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--alpha-s", type=float, default=DEFAULT_ALPHA_S)
    parser.add_argument("--target-seconds", type=float, default=0.25)
    parser.add_argument(
        "--generation-timeout", type=float, default=DEFAULT_TIME_LIMIT_SECONDS
    )
    parser.add_argument(
        "--runtime-timeout", type=float, default=DEFAULT_TIME_LIMIT_SECONDS
    )
    parser.add_argument("--memory-limit-gib", type=float, default=MAX_MEMORY_LIMIT_GIB)
    parser.add_argument(
        "--amplicol-repository",
        type=Path,
        default=legacy_amplicol.DEFAULT_REPOSITORY,
    )
    parser.add_argument(
        "--reference-fft-root",
        type=Path,
        default=performance.DEFAULT_REFERENCE_ROOT,
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
    if arguments.multiplicities is not None and (
        not arguments.multiplicities or min(arguments.multiplicities) < 2
    ):
        raise StudyError("--multiplicity values must be integers >=2")
    if arguments.multiplicities is None and (
        arguments.min_n < 2 or arguments.max_n < arguments.min_n
    ):
        raise StudyError(
            "--min-n/--max-n must define a nonempty range starting at n>=2"
        )
    if arguments.fill_multiplicities is not None:
        if min(arguments.fill_multiplicities, default=1) < 2:
            raise StudyError("--fill-multiplicity values must be integers >=2")
        outside_policy = sorted(
            set(arguments.fill_multiplicities) - set(_multiplicities(arguments))
        )
        if outside_policy:
            raise StudyError(
                "--fill-multiplicity values must belong to the fixed policy: "
                + ", ".join(str(value) for value in outside_policy)
            )
    if arguments.batch_size < 1:
        raise StudyError("--batch-size must be positive")
    if (
        not _positive_finite(arguments.target_seconds)
        or arguments.target_seconds < 0.25
    ):
        raise StudyError("--target-seconds must be at least 0.25")
    if not _positive_finite(arguments.generation_timeout):
        raise StudyError("--generation-timeout must be positive and finite")
    if not _positive_finite(arguments.runtime_timeout):
        raise StudyError("--runtime-timeout must be positive and finite")
    if not _positive_finite(arguments.memory_limit_gib):
        raise StudyError("--memory-limit-gib must be positive and finite")
    if arguments.optimization_cores is not None and arguments.optimization_cores < 1:
        raise StudyError("--optimization-cores must be positive")
    families = _selected_families(arguments)
    if not families:
        raise StudyError("at least one --family is required")
    modes_by_family = {
        family: _selected_modes(arguments, family) for family in families
    }
    if any(not modes for modes in modes_by_family.values()):
        raise StudyError("the selected --mode values leave a family with no curves")
    gg_modes = {mode.key for mode in modes_by_family.get("gg", ())}
    if gg_modes - {"reference-fft"} and "reference-fft" not in gg_modes:
        raise StudyError("gg candidate/AmpliCol curves require --mode reference-fft")
    ddbar_modes = {mode.key for mode in modes_by_family.get("ddbar", ())}
    if any(MODE_BY_KEY[key].kind == "candidate" for key in ddbar_modes) and (
        "amplicol" not in ddbar_modes
    ):
        raise StudyError("ddbar candidate curves require --mode amplicol")
    selected_modes = {
        mode.key: mode for modes in modes_by_family.values() for mode in modes
    }
    if arguments.batch_size == 1 and any(
        mode.execution_mode == "compiled" and mode.key != "compiled-fft"
        for mode in selected_modes.values()
    ):
        raise StudyError("--batch-size 1 is not supported for compiled curves")


def _exec_in(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        raise StudyError("_exec-in requires a directory and command")
    directory = Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    command = list(argv[1:])
    os.execvpe(command[0], command, os.environ)
    raise AssertionError("os.execvpe returned")


def _time_rss(argv: Sequence[str]) -> int:
    """Run one child and report its getrusage high-water mark."""

    if not argv:
        raise StudyError("_time-rss requires a command")
    completed = subprocess.run(tuple(argv), check=False)
    maximum_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    maximum_rss_kib = maximum_rss // 1024 if sys.platform == "darwin" else maximum_rss
    if maximum_rss_kib < 1:
        raise StudyError("_time-rss child did not report a positive RSS")
    print(f"{RSS_MARKER} {maximum_rss_kib}", file=sys.stderr)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        if raw and raw[0] == "_exec-in":
            return _exec_in(raw[1:])
        if raw and raw[0] == "_time-rss":
            return _time_rss(raw[1:])
        arguments = _parser().parse_args(raw)
        performance.configure_reference_root(arguments.reference_fft_root)
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
    except (
        StudyError,
        performance.AcceptanceError,
        legacy_report.LegacyAdapterError,
        legacy_report.ProfilingTimeLimitError,
        OSError,
        ValueError,
    ) as error:
        print(f"FFT scaling study: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
