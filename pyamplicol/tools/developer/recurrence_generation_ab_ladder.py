#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the guarded recurrence-generation baseline/candidate ladder.

This developer-only orchestrator deliberately delegates artifact generation
and runtime validation to ``recurrence_z6g_benchmark.py``.  Every scheduled
baseline/candidate capture is a separate process-tree invocation of
``memory_watchdog.py --limit-gib 30`` with unique output, temporary, bytecode,
and XDG cache roots.

The default generation ladder covers n=2 through n=9 for both LC layouts with
three repetitions.  Baseline-first and candidate-first pairs alternate.  Add
``--runtime-n 6 --runtime-n 7`` to perform the approved runtime validation
cells with the harness defaults made explicit here: batches 1/128/1024,
seven subprocesses, seven native blocks, two warm-ups, and a five-second
measurement target.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_PARENT = ROOT / ".artifacts" / "recurrence-generation-opt"
CAMPAIGN_PARENT = ALLOWED_OUTPUT_PARENT / "benchmark-campaigns"
HARNESS_RELATIVE_PATH = Path("tools/developer/recurrence_z6g_benchmark.py")
WATCHDOG_RELATIVE_PATH = Path("tools/ci/memory_watchdog.py")

CAMPAIGN_KIND = "pyamplicol-recurrence-generation-ab-ladder"
CAMPAIGN_SCHEMA_VERSION = 1
HARNESS_KIND = "pyamplicol-recurrence-z6g-benchmark"
HARNESS_SCHEMA_VERSION = 6
EXECUTION_KIND = "pyamplicol-runtime-recurrence-execution"
EXECUTION_SCHEMA_VERSION = 3

LAYOUTS = ("topology-replay", "all-flow-union")
DEFAULT_MULTIPLICITIES = tuple(range(2, 10))
DEFAULT_BATCH_SIZES = (1, 128, 1024)
WATCHDOG_LIMIT_GIB = 30.0
GIB = 1024**3

_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_WATCHDOG_FINISHED_RE = re.compile(
    r"memory-watchdog: command finished"
    r" exit=(?P<exit>\d+)"
    r" peak_rss=(?P<rss>\d+(?:[.]\d+)?) GiB"
    r" peak_physical_footprint="
    r"(?P<physical>unavailable|\d+(?:[.]\d+)? GiB)"
    r" peak_guard=(?P<guard>\d+(?:[.]\d+)?) GiB"
    r" peak_processes=(?P<processes>\d+)"
)
_WATCHDOG_LIMIT_RE = re.compile(
    r"memory-watchdog: memory limit exceeded"
    r" reason=(?P<reason>[^ ]+)"
    r" observed=(?P<observed>\d+(?:[.]\d+)?) GiB"
    r" limit=(?P<limit>\d+(?:[.]\d+)?) GiB"
    r" rss=(?P<rss>\d+(?:[.]\d+)?) GiB"
    r" physical_footprint="
    r"(?P<physical>unavailable|\d+(?:[.]\d+)? GiB)"
    r" processes=(?P<processes>\d+);"
)
_HARNESS_GENERATION_TIMEOUT_RE = re.compile(
    r"^recurrence-z6g-benchmark: recurrence generation worker exceeded "
    r"(?P<seconds>\d+(?:[.]\d+)?) seconds$",
    re.MULTILINE,
)


class LadderError(RuntimeError):
    """Raised when a campaign or one of its captured results is invalid."""


@dataclass(frozen=True, slots=True)
class Variant:
    """One independently installed baseline or candidate environment."""

    name: str
    checkout: Path
    python: Path
    pythonpath: Path | None = None
    prepared_model: Path | None = None


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """One outer watchdog-guarded harness capture."""

    sequence_index: int
    pair_index: int
    order_in_pair: int
    repetition: int
    multiplicity: int
    layout: str
    variant_name: str
    runtime_enabled: bool

    @property
    def process_expression(self) -> str:
        return process_expression(self.multiplicity)

    @property
    def sample_id(self) -> str:
        layout = self.layout.replace("-", "_")
        return (
            f"{self.sequence_index:04d}-p{self.pair_index:03d}"
            f"-r{self.repetition + 1:02d}-{self.variant_name}"
            f"-n{self.multiplicity:02d}-{layout}"
        )


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    """Stable harness settings shared by every campaign sample."""

    validation_samples: int = 10
    point_tile_size: int = 1024
    jit_optimization_level: int = 2
    profile_timeout_seconds: float = 300.0
    minimum_samples: int = 7
    subprocess_samples: int = 7
    warmup_runs: int = 2
    target_runtime_seconds: float = 5.0
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    allow_diagnostic_incomplete_success: bool = False


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Minimal outcome of an outer watchdog process."""

    exit_code: int | None
    timed_out: bool
    wall_seconds: float
    error: str | None


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def process_expression(multiplicity: int) -> str:
    """Return exactly ``d d~ > Z + (n-1)*g`` in CLI process syntax."""

    if isinstance(multiplicity, bool) or multiplicity < 1:
        raise LadderError("process multiplicity n must be a positive integer")
    return " ".join(("d", "d~", ">", "Z", *(("g",) * (multiplicity - 1))))


def generation_timeout_seconds(multiplicity: int, layout: str) -> float:
    """Return the approved per-cell generation timeout."""

    if layout not in LAYOUTS:
        raise LadderError(f"unsupported LC flow layout: {layout}")
    if multiplicity < 1:
        raise LadderError("process multiplicity n must be positive")
    if multiplicity <= 4:
        return 5.0 * 60.0
    if multiplicity <= 7:
        return 15.0 * 60.0
    if multiplicity == 8:
        return (1.0 if layout == "topology-replay" else 2.0) * 60.0 * 60.0
    if multiplicity == 9:
        return (2.0 if layout == "topology-replay" else 6.0) * 60.0 * 60.0
    raise LadderError(
        "the approved timeout policy covers only process multiplicities n=1..9"
    )


def _outer_timeout_seconds(
    spec: SampleSpec,
    settings: RunnerSettings,
) -> float:
    generation = generation_timeout_seconds(spec.multiplicity, spec.layout)
    if not spec.runtime_enabled:
        return generation + 120.0
    profile_budget = (
        settings.profile_timeout_seconds
        * settings.subprocess_samples
        * len(settings.batch_sizes)
    )
    return generation + profile_budget + 300.0


def build_schedule(
    multiplicities: Sequence[int],
    layouts: Sequence[str],
    repetitions: int,
    runtime_multiplicities: frozenset[int],
) -> list[SampleSpec]:
    """Build deterministic pairs with alternating A/B order."""

    if repetitions <= 0:
        raise LadderError("repetitions must be positive")
    normalized_multiplicities = tuple(dict.fromkeys(multiplicities))
    normalized_layouts = tuple(dict.fromkeys(layouts))
    if not normalized_multiplicities:
        raise LadderError("at least one process multiplicity is required")
    if not normalized_layouts:
        raise LadderError("at least one flow layout is required")
    for multiplicity in normalized_multiplicities:
        generation_timeout_seconds(multiplicity, "topology-replay")
    for layout in normalized_layouts:
        if layout not in LAYOUTS:
            raise LadderError(f"unsupported LC flow layout: {layout}")
    if not runtime_multiplicities.issubset(normalized_multiplicities):
        raise LadderError("--runtime-n values must also be selected with --n")

    schedule: list[SampleSpec] = []
    pair_index = 0
    sequence_index = 0
    for multiplicity in normalized_multiplicities:
        for layout in normalized_layouts:
            for repetition in range(repetitions):
                order = (
                    ("baseline", "candidate")
                    if pair_index % 2 == 0
                    else ("candidate", "baseline")
                )
                for order_in_pair, variant_name in enumerate(order):
                    schedule.append(
                        SampleSpec(
                            sequence_index=sequence_index,
                            pair_index=pair_index,
                            order_in_pair=order_in_pair,
                            repetition=repetition,
                            multiplicity=multiplicity,
                            layout=layout,
                            variant_name=variant_name,
                            runtime_enabled=(multiplicity in runtime_multiplicities),
                        )
                    )
                    sequence_index += 1
                pair_index += 1
    return schedule


def build_sample_command(
    spec: SampleSpec,
    variant: Variant,
    sample_root: Path,
    settings: RunnerSettings,
) -> list[str]:
    """Construct the exact watchdog and harness command for one sample."""

    watchdog = variant.checkout / WATCHDOG_RELATIVE_PATH
    harness = variant.checkout / HARNESS_RELATIVE_PATH
    harness_output = sample_root / "harness"
    result_json = sample_root / "harness-result.json"
    generation_timeout = generation_timeout_seconds(
        spec.multiplicity,
        spec.layout,
    )
    harness_command = [
        str(variant.python),
        str(harness),
        "--output-root",
        str(harness_output),
        "--result-json",
        str(result_json),
        "--gluon-count",
        str(max(1, spec.multiplicity - 1)),
        "--process-expression",
        spec.process_expression,
        "--validation-samples",
        str(settings.validation_samples),
        "--point-tile-size",
        str(settings.point_tile_size),
        "--jit-optimization-level",
        str(settings.jit_optimization_level),
        "--mode",
        "recurrence",
        "--lc-flow-layout",
        spec.layout,
        "--generation-timeout",
        f"{generation_timeout:g}",
        "--profile-timeout",
        f"{settings.profile_timeout_seconds:g}",
        "--minimum-samples",
        str(settings.minimum_samples),
        "--subprocess-samples",
        str(settings.subprocess_samples),
        "--warmup-runs",
        str(settings.warmup_runs),
        "--target-runtime",
        f"{settings.target_runtime_seconds:g}",
    ]
    if settings.allow_diagnostic_incomplete_success:
        harness_command.append("--allow-diagnostic-incomplete-success")
    if variant.prepared_model is not None:
        harness_command.extend(("--prepared-model", str(variant.prepared_model)))
    if spec.runtime_enabled:
        for batch_size in settings.batch_sizes:
            harness_command.extend(("--batch-size", str(batch_size)))
    else:
        harness_command.append("--generation-only")

    return [
        str(variant.python),
        str(watchdog),
        "--limit-gib",
        f"{WATCHDOG_LIMIT_GIB:g}",
        "--",
        *harness_command,
    ]


def _sample_environment(
    variant: Variant,
    sample_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    cache_root = sample_root / "cold-cache"
    temporary_root = sample_root / "tmp"
    pycache_root = sample_root / "pycache"
    matplotlib_root = cache_root / "matplotlib"
    for path in (cache_root, temporary_root, pycache_root, matplotlib_root):
        path.mkdir(parents=True, exist_ok=False)

    overrides = {
        "MPLCONFIGDIR": str(matplotlib_root),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_root),
        "SYMBOLICA_HIDE_BANNER": "1",
        "TEMP": str(temporary_root),
        "TMP": str(temporary_root),
        "TMPDIR": str(temporary_root),
        "XDG_CACHE_HOME": str(cache_root),
    }
    environment = os.environ.copy()
    environment.update(overrides)
    if variant.pythonpath is None:
        environment.pop("PYTHONPATH", None)
    else:
        overrides["PYTHONPATH"] = str(variant.pythonpath)
        environment["PYTHONPATH"] = str(variant.pythonpath)
    return environment, overrides


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> ProcessOutcome:
    started = time.perf_counter()
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                return ProcessOutcome(
                    exit_code=None,
                    timed_out=True,
                    wall_seconds=time.perf_counter() - started,
                    error=f"outer sample timeout after {timeout_seconds:g} seconds",
                )
    except OSError as error:
        if process is not None:
            _terminate_process_tree(process)
        return ProcessOutcome(
            exit_code=None,
            timed_out=False,
            wall_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
    return ProcessOutcome(
        exit_code=exit_code,
        timed_out=False,
        wall_seconds=time.perf_counter() - started,
        error=None,
    )


def _gib_value(text: str) -> dict[str, float | int]:
    gib = float(text.removesuffix(" GiB"))
    return {
        "gib": gib,
        "bytes_rounded_from_watchdog": round(gib * GIB),
    }


def parse_watchdog_text(text: str) -> dict[str, Any]:
    """Extract the watchdog's terminal process-tree memory observation."""

    finished = list(_WATCHDOG_FINISHED_RE.finditer(text))
    exceeded = list(_WATCHDOG_LIMIT_RE.finditer(text))
    if finished:
        match = finished[-1]
        physical = match.group("physical")
        return {
            "terminal_record": "command-finished",
            "limit_gib": WATCHDOG_LIMIT_GIB,
            "limit_exceeded": False,
            "child_exit_code": int(match.group("exit")),
            "peak_rss": _gib_value(match.group("rss")),
            "peak_physical_footprint": (
                None if physical == "unavailable" else _gib_value(physical)
            ),
            "peak_guard": _gib_value(match.group("guard")),
            "peak_process_count": int(match.group("processes")),
        }
    if exceeded:
        match = exceeded[-1]
        physical = match.group("physical")
        return {
            "terminal_record": "memory-limit-exceeded",
            "limit_gib": float(match.group("limit")),
            "limit_exceeded": True,
            "reason": match.group("reason"),
            "observed_guard": _gib_value(match.group("observed")),
            "rss_at_limit": _gib_value(match.group("rss")),
            "physical_footprint_at_limit": (
                None if physical == "unavailable" else _gib_value(physical)
            ),
            "process_count_at_limit": int(match.group("processes")),
        }
    return {
        "terminal_record": None,
        "limit_gib": WATCHDOG_LIMIT_GIB,
        "limit_exceeded": None,
    }


def parse_watchdog_log(path: Path) -> dict[str, Any]:
    """Parse at most the final MiB of a potentially large stderr log."""

    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 1024 * 1024))
            raw = stream.read()
    except OSError as error:
        raise LadderError(f"cannot read watchdog stderr log: {path}") from error
    return parse_watchdog_text(raw.decode("utf-8", errors="replace"))


def parse_harness_generation_timeout(path: Path) -> dict[str, float] | None:
    """Return an exact inner generation-timeout record, if present."""

    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 1024 * 1024))
            raw = stream.read()
    except OSError as error:
        raise LadderError(f"cannot read harness stderr log: {path}") from error
    text = raw.decode("utf-8", errors="replace")
    matches = tuple(_HARNESS_GENERATION_TIMEOUT_RE.finditer(text))
    if not matches:
        return None
    if len(matches) != 1:
        raise LadderError("harness stderr contains multiple generation timeouts")
    return {"configured_seconds": float(matches[0].group("seconds"))}


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LadderError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise LadderError(f"{description} must be a JSON object: {path}")
    return value


def _required_mapping(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LadderError(f"{description} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LadderError(f"cannot hash result file: {path}") from error
    return digest.hexdigest()


def _runtime_profile_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    profile = _required_mapping(value, description="recurrence runtime profile")
    raw_measurements = profile.get("profiles")
    if not isinstance(raw_measurements, list):
        raise LadderError("recurrence runtime profile has no measurement list")
    measurements: list[dict[str, Any]] = []
    for raw_measurement in raw_measurements:
        measurement = _required_mapping(
            raw_measurement,
            description="runtime batch measurement",
        )
        raw_samples = measurement.get("subprocess_samples")
        if not isinstance(raw_samples, list):
            raise LadderError("runtime batch measurement has no subprocess samples")
        subprocess_samples = []
        for raw_sample in raw_samples:
            sample = _required_mapping(
                raw_sample,
                description="runtime subprocess sample",
            )
            process_record = sample.get("worker_process_record")
            worker_wall = (
                process_record.get("wall_seconds")
                if isinstance(process_record, Mapping)
                else None
            )
            subprocess_samples.append(
                {
                    "schedule_index": sample.get("schedule_index"),
                    "round": sample.get("round"),
                    "wall_seconds_per_point": sample.get("wall_seconds_per_point"),
                    "internal_sample_count": sample.get("internal_sample_count"),
                    "repetitions_per_sample": sample.get("repetitions_per_sample"),
                    "evaluation_count": sample.get("evaluation_count"),
                    "evaluated_point_count": sample.get("evaluated_point_count"),
                    "interrupted": sample.get("interrupted"),
                    "worker_wall_seconds": worker_wall,
                    "peak_rss_after_cold_load": sample.get("peak_rss_after_cold_load"),
                    "peak_rss_after_profile": sample.get("peak_rss_after_profile"),
                }
            )
        measurements.append(
            {
                "batch_size": measurement.get("batch_size"),
                "sample_count": measurement.get("sample_count"),
                "wall_seconds_per_point": measurement.get("wall_seconds_per_point"),
                "wall_seconds_per_point_median": measurement.get(
                    "wall_seconds_per_point_median"
                ),
                "wall_seconds_per_point_mad": measurement.get(
                    "wall_seconds_per_point_mad"
                ),
                "interrupted": measurement.get("interrupted"),
                "subprocess_samples": subprocess_samples,
            }
        )
    return {
        "process_id": profile.get("process_id"),
        "process_expression": profile.get("process_expression"),
        "measurements": measurements,
    }


def parse_harness_result(path: Path) -> dict[str, Any]:
    """Extract compact generation/native/runtime evidence from a harness result."""

    result = _json_object(path, description="recurrence benchmark result")
    if (
        result.get("kind") != HARNESS_KIND
        or result.get("schema_version") != HARNESS_SCHEMA_VERSION
    ):
        raise LadderError(
            "recurrence benchmark result kind/schema does not match this runner"
        )
    generation = _required_mapping(
        result.get("generation"),
        description="benchmark generation",
    )
    recurrence = _required_mapping(
        generation.get("recurrence"),
        description="recurrence generation record",
    )
    artifact_raw = recurrence.get("artifact")
    if not isinstance(artifact_raw, str):
        raise LadderError("recurrence generation record has no artifact path")
    artifact = Path(artifact_raw).expanduser().resolve(strict=True)
    execution_paths = sorted(artifact.glob("processes/*/execution.json"))
    if len(execution_paths) != 1:
        raise LadderError(
            "exact-process recurrence artifact must have one execution manifest"
        )
    execution = _json_object(
        execution_paths[0],
        description="recurrence execution manifest",
    )
    if (
        execution.get("kind") != EXECUTION_KIND
        or execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
    ):
        raise LadderError("recurrence execution manifest kind/schema mismatch")
    plan = _required_mapping(
        execution.get("plan"),
        description="recurrence execution plan",
    )
    inspection = _required_mapping(
        plan.get("inspection_summary"),
        description="native recurrence inspection summary",
    )
    native_timings = _required_mapping(
        inspection.get("generation_timings_seconds"),
        description="native generation timings",
    )
    phases = _required_mapping(
        recurrence.get("phase_timings_seconds"),
        description="generation phase timings",
    )
    profiles = _required_mapping(
        result.get("profiles"),
        description="benchmark profiles",
    )
    artifact_identity = _required_mapping(
        recurrence.get("artifact_identity"),
        description="artifact identity",
    )
    process_record = recurrence.get("worker_process_record")
    worker_wall_seconds = (
        process_record.get("wall_seconds")
        if isinstance(process_record, Mapping)
        else None
    )
    provenance = result.get("provenance")
    driver_wall_seconds = (
        provenance.get("wall_seconds") if isinstance(provenance, Mapping) else None
    )
    return {
        "harness_result_sha256": _sha256_file(path),
        "process": result.get("process"),
        "layout": (
            result.get("configuration", {}).get("lc_flow_layout")
            if isinstance(result.get("configuration"), Mapping)
            else None
        ),
        "source": result.get("source"),
        "runtime_provenance": result.get("runtime_provenance"),
        "artifact": str(artifact),
        "artifact_id": artifact_identity.get("artifact_id"),
        "artifact_semantic_identity_sha256": recurrence.get(
            "artifact_semantic_identity_sha256"
        ),
        "generation_wall_seconds": recurrence.get("generation_wall_seconds"),
        "generation_worker_wall_seconds": worker_wall_seconds,
        "harness_driver_wall_seconds": driver_wall_seconds,
        "generation_peak_rss": recurrence.get("peak_rss"),
        "generation_phase_timings_seconds": dict(phases),
        "generation_phase_total_seconds": recurrence.get("phase_total_seconds"),
        "native_generation_timings_seconds": dict(native_timings),
        "native_inspection_summary": dict(inspection),
        "runtime_profile": _runtime_profile_summary(profiles.get("recurrence")),
        "complete": result.get("complete"),
        "passes": result.get("passes"),
    }


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise LadderError(f"cannot write campaign result: {path}") from error


def _variant_payload(variant: Variant) -> dict[str, str | None]:
    return {
        "checkout": str(variant.checkout),
        "python": str(variant.python),
        "pythonpath": (None if variant.pythonpath is None else str(variant.pythonpath)),
        "prepared_model": (
            None if variant.prepared_model is None else str(variant.prepared_model)
        ),
    }


def _sample_plan_payload(
    spec: SampleSpec,
    *,
    variant: Variant,
    campaign_root: Path,
    settings: RunnerSettings,
) -> dict[str, Any]:
    sample_root = campaign_root / "samples" / spec.sample_id
    return {
        "sample_id": spec.sample_id,
        "sequence_index": spec.sequence_index,
        "pair_index": spec.pair_index,
        "order_in_pair": spec.order_in_pair,
        "repetition": spec.repetition + 1,
        "variant": spec.variant_name,
        "multiplicity": spec.multiplicity,
        "process_expression": spec.process_expression,
        "layout": spec.layout,
        "runtime_enabled": spec.runtime_enabled,
        "generation_timeout_seconds": generation_timeout_seconds(
            spec.multiplicity,
            spec.layout,
        ),
        "outer_timeout_seconds": _outer_timeout_seconds(spec, settings),
        "sample_root": str(sample_root),
        "command": build_sample_command(
            spec,
            variant,
            sample_root,
            settings,
        ),
    }


def _execute_sample(
    spec: SampleSpec,
    *,
    variant: Variant,
    campaign_root: Path,
    settings: RunnerSettings,
) -> dict[str, Any]:
    sample_root = campaign_root / "samples" / spec.sample_id
    sample_root.mkdir(parents=True, exist_ok=False)
    environment, environment_overrides = _sample_environment(
        variant,
        sample_root,
    )
    command = build_sample_command(spec, variant, sample_root, settings)
    stdout_path = sample_root / "stdout.log"
    stderr_path = sample_root / "stderr.log"
    result_path = sample_root / "harness-result.json"
    outer_timeout = _outer_timeout_seconds(spec, settings)
    started_at = _utc_now()
    outcome = _run_process(
        command,
        cwd=variant.checkout,
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=outer_timeout,
    )
    finished_at = _utc_now()
    watchdog = parse_watchdog_log(stderr_path)
    harness_generation_timeout = parse_harness_generation_timeout(stderr_path)
    harness_summary: dict[str, Any] | None = None
    result_error: str | None = None
    if result_path.is_file():
        try:
            harness_summary = parse_harness_result(result_path)
        except LadderError as error:
            result_error = str(error)
    elif outcome.exit_code == 0 and not outcome.timed_out:
        result_error = "watchdog exited successfully without a harness result"

    status = _sample_status(
        outcome,
        watchdog=watchdog,
        result_error=result_error,
        harness_summary=harness_summary,
        harness_generation_timeout=harness_generation_timeout,
        allow_diagnostic_incomplete_success=(
            settings.allow_diagnostic_incomplete_success
        ),
    )
    return {
        **_sample_plan_payload(
            spec,
            variant=variant,
            campaign_root=campaign_root,
            settings=settings,
        ),
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_seconds": outcome.wall_seconds,
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        "process_error": outcome.error,
        "result_error": result_error,
        "environment_overrides": environment_overrides,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "harness_result": str(result_path),
        "watchdog": watchdog,
        "harness_generation_timeout": harness_generation_timeout,
        "telemetry": harness_summary,
    }


def _sample_status(
    outcome: ProcessOutcome,
    *,
    watchdog: Mapping[str, object],
    result_error: str | None,
    harness_summary: Mapping[str, object] | None,
    harness_generation_timeout: Mapping[str, object] | None = None,
    allow_diagnostic_incomplete_success: bool,
) -> str:
    """Classify authoritative success separately from censored diagnostics."""

    if outcome.timed_out:
        return "censored" if allow_diagnostic_incomplete_success else "timeout"
    if outcome.error is not None:
        return "launch-error"
    if harness_generation_timeout is not None:
        if watchdog.get("terminal_record") != "command-finished":
            return "invalid-watchdog-evidence"
        return "censored" if allow_diagnostic_incomplete_success else "timeout"
    if outcome.exit_code != 0:
        return "failed"
    if result_error is not None or harness_summary is None:
        return "invalid-result"
    if watchdog.get("terminal_record") != "command-finished":
        return "invalid-watchdog-evidence"
    complete = harness_summary.get("complete")
    passes = harness_summary.get("passes")
    if complete is True and passes is True:
        return "passed"
    if allow_diagnostic_incomplete_success and complete is not True and passes is None:
        return "censored"
    return "failed-validation"


def _resolve_existing_path(path: Path, *, description: str) -> Path:
    try:
        result = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise LadderError(f"{description} does not exist: {path}") from error
    return result


def _resolve_python(path: Path, *, description: str) -> Path:
    """Validate while preserving a virtual-environment interpreter symlink."""

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if not absolute.exists():
        raise LadderError(f"{description} does not exist: {path}")
    if not absolute.is_file():
        raise LadderError(f"{description} is not a regular file")
    return absolute.absolute()


def _resolve_variant(
    name: str,
    *,
    checkout: Path,
    python: Path,
    pythonpath: Path | None,
    prepared_model: Path | None,
) -> Variant:
    resolved_checkout = _resolve_existing_path(
        checkout,
        description=f"{name} checkout",
    )
    if not resolved_checkout.is_dir():
        raise LadderError(f"{name} checkout is not a directory")
    resolved_python = _resolve_python(
        python,
        description=f"{name} Python",
    )
    resolved_pythonpath = (
        None
        if pythonpath is None
        else _resolve_existing_path(
            pythonpath,
            description=f"{name} Python path",
        )
    )
    resolved_model = (
        None
        if prepared_model is None
        else _resolve_existing_path(
            prepared_model,
            description=f"{name} prepared model",
        )
    )
    for relative, description in (
        (WATCHDOG_RELATIVE_PATH, "memory watchdog"),
        (HARNESS_RELATIVE_PATH, "recurrence benchmark harness"),
    ):
        path = resolved_checkout / relative
        if not path.is_file():
            raise LadderError(f"{name} checkout has no {description}: {path}")
    return Variant(
        name=name,
        checkout=resolved_checkout,
        python=resolved_python,
        pythonpath=resolved_pythonpath,
        prepared_model=resolved_model,
    )


def _campaign_root(output_root: Path, campaign_id: str) -> Path:
    if _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None or campaign_id in {".", ".."}:
        raise LadderError(
            "campaign ID must contain only letters, digits, '.', '_', and '-'"
        )
    allowed_parent = ALLOWED_OUTPUT_PARENT.resolve()
    requested_parent = output_root.expanduser().resolve()
    try:
        requested_parent.relative_to(allowed_parent)
    except ValueError as error:
        raise LadderError(
            f"campaign output must remain under {ALLOWED_OUTPUT_PARENT}"
        ) from error
    return requested_parent / campaign_id


def _default_campaign_id() -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--candidate-python", type=Path, required=True)
    result.add_argument("--baseline-checkout", type=Path, default=ROOT)
    result.add_argument("--candidate-checkout", type=Path, default=ROOT)
    result.add_argument("--baseline-pythonpath", type=Path)
    result.add_argument("--candidate-pythonpath", type=Path)
    result.add_argument("--baseline-prepared-model", type=Path)
    result.add_argument("--candidate-prepared-model", type=Path)
    result.add_argument(
        "--output-root",
        type=Path,
        default=CAMPAIGN_PARENT,
        help=f"campaign parent below {CAMPAIGN_PARENT}",
    )
    result.add_argument("--campaign-id", default=None)
    result.add_argument(
        "--n",
        dest="multiplicities",
        action="append",
        type=_positive_int,
        help="process multiplicity; repeat (default: n=2 through n=9)",
    )
    result.add_argument(
        "--layout",
        dest="layouts",
        action="append",
        choices=LAYOUTS,
        help="LC flow layout; repeat (default: both layouts)",
    )
    result.add_argument(
        "--repetitions",
        type=_positive_int,
        default=3,
        help="paired baseline/candidate repetitions per cell (default: 3)",
    )
    result.add_argument(
        "--runtime-n",
        action="append",
        type=_positive_int,
        default=None,
        help=(
            "enable runtime profiling for this selected n; repeat "
            "(approved validation: 6 and 7)"
        ),
    )
    result.add_argument(
        "--batch-size",
        action="append",
        type=_positive_int,
        default=None,
    )
    result.add_argument(
        "--validation-samples",
        type=_positive_int,
        default=10,
    )
    result.add_argument(
        "--point-tile-size",
        type=_positive_int,
        default=1024,
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=300.0,
    )
    result.add_argument(
        "--minimum-samples",
        type=_positive_int,
        default=7,
    )
    result.add_argument(
        "--subprocess-samples",
        type=_positive_int,
        default=7,
    )
    result.add_argument(
        "--warmup-runs",
        type=_nonnegative_int,
        default=2,
    )
    result.add_argument(
        "--target-runtime",
        type=_positive_float,
        default=5.0,
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="write the complete campaign plan without starting subprocesses",
    )
    result.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop after the first failed or timed-out sample",
    )
    result.add_argument(
        "--allow-diagnostic-incomplete-success",
        action="store_true",
        help=(
            "retain explicitly censored diagnostic captures without counting "
            "them as passing acceptance samples"
        ),
    )
    return result


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    multiplicities = tuple(
        dict.fromkeys(
            DEFAULT_MULTIPLICITIES
            if arguments.multiplicities is None
            else arguments.multiplicities
        )
    )
    layouts = tuple(
        dict.fromkeys(LAYOUTS if arguments.layouts is None else arguments.layouts)
    )
    runtime_multiplicities = frozenset(arguments.runtime_n or ())
    batch_sizes = tuple(
        DEFAULT_BATCH_SIZES
        if arguments.batch_size is None
        else dict.fromkeys(arguments.batch_size)
    )
    settings = RunnerSettings(
        validation_samples=arguments.validation_samples,
        point_tile_size=arguments.point_tile_size,
        jit_optimization_level=arguments.jit_optimization_level,
        profile_timeout_seconds=arguments.profile_timeout,
        minimum_samples=arguments.minimum_samples,
        subprocess_samples=arguments.subprocess_samples,
        warmup_runs=arguments.warmup_runs,
        target_runtime_seconds=arguments.target_runtime,
        batch_sizes=batch_sizes,
        allow_diagnostic_incomplete_success=(
            arguments.allow_diagnostic_incomplete_success
        ),
    )
    variants = {
        "baseline": _resolve_variant(
            "baseline",
            checkout=arguments.baseline_checkout,
            python=arguments.baseline_python,
            pythonpath=arguments.baseline_pythonpath,
            prepared_model=arguments.baseline_prepared_model,
        ),
        "candidate": _resolve_variant(
            "candidate",
            checkout=arguments.candidate_checkout,
            python=arguments.candidate_python,
            pythonpath=arguments.candidate_pythonpath,
            prepared_model=arguments.candidate_prepared_model,
        ),
    }
    schedule = build_schedule(
        multiplicities,
        layouts,
        arguments.repetitions,
        runtime_multiplicities,
    )
    campaign_id = arguments.campaign_id or _default_campaign_id()
    campaign_root = _campaign_root(arguments.output_root, campaign_id)
    try:
        campaign_root.mkdir(parents=True, exist_ok=False)
        (campaign_root / "samples").mkdir()
    except FileExistsError as error:
        raise LadderError(
            f"campaign directory already exists; choose a new ID: {campaign_root}"
        ) from error
    campaign_path = campaign_root / "campaign.json"
    plan_payloads = [
        _sample_plan_payload(
            spec,
            variant=variants[spec.variant_name],
            campaign_root=campaign_root,
            settings=settings,
        )
        for spec in schedule
    ]
    started_at = _utc_now()
    campaign: dict[str, Any] = {
        "kind": CAMPAIGN_KIND,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "status": "planned" if arguments.dry_run else "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "campaign_root": str(campaign_root),
        "configuration": {
            "multiplicities": list(multiplicities),
            "layouts": list(layouts),
            "repetitions": arguments.repetitions,
            "runtime_multiplicities": sorted(runtime_multiplicities),
            "watchdog_limit_gib": WATCHDOG_LIMIT_GIB,
            "validation_samples": settings.validation_samples,
            "point_tile_size": settings.point_tile_size,
            "jit_optimization_level": settings.jit_optimization_level,
            "profile_timeout_seconds": settings.profile_timeout_seconds,
            "minimum_samples": settings.minimum_samples,
            "subprocess_samples": settings.subprocess_samples,
            "warmup_runs": settings.warmup_runs,
            "target_runtime_seconds": settings.target_runtime_seconds,
            "batch_sizes": list(settings.batch_sizes),
            "stop_on_failure": arguments.stop_on_failure,
            "allow_diagnostic_incomplete_success": (
                settings.allow_diagnostic_incomplete_success
            ),
            "cold_cache_policy": "unique-roots-per-outer-sample-v1",
            "ordering_policy": "alternating-baseline-candidate-pairs-v1",
            "watchdog_policy": "outer-process-tree-guard-covers-all-workers-v1",
        },
        "variants": {
            name: _variant_payload(variant) for name, variant in variants.items()
        },
        "schedule": plan_payloads,
        "samples": [],
        "summary": None,
    }
    _atomic_write_json(campaign_path, campaign)
    if arguments.dry_run:
        campaign["finished_at_utc"] = _utc_now()
        _atomic_write_json(campaign_path, campaign)
        return campaign

    campaign_started = time.perf_counter()
    for spec in schedule:
        sample = _execute_sample(
            spec,
            variant=variants[spec.variant_name],
            campaign_root=campaign_root,
            settings=settings,
        )
        campaign["samples"].append(sample)
        _atomic_write_json(campaign_path, campaign)
        acceptable_statuses = (
            {"passed", "censored"}
            if settings.allow_diagnostic_incomplete_success
            else {"passed"}
        )
        if arguments.stop_on_failure and sample["status"] not in acceptable_statuses:
            break

    statuses = [
        sample["status"]
        for sample in campaign["samples"]
        if isinstance(sample, Mapping)
    ]
    completed = len(campaign["samples"])
    scheduled = len(schedule)
    passed = sum(status == "passed" for status in statuses)
    censored = sum(status == "censored" for status in statuses)
    failed = completed - passed - censored
    campaign["finished_at_utc"] = _utc_now()
    if completed != scheduled or failed != 0:
        campaign["status"] = "failed"
    elif censored:
        campaign["status"] = "completed-with-censoring"
    else:
        campaign["status"] = "passed"
    campaign["summary"] = {
        "scheduled_sample_count": scheduled,
        "completed_sample_count": completed,
        "passed_sample_count": passed,
        "censored_sample_count": censored,
        "failed_sample_count": failed,
        "campaign_wall_seconds": time.perf_counter() - campaign_started,
    }
    _atomic_write_json(campaign_path, campaign)
    return campaign


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
        result = run(arguments)
    except LadderError as error:
        print(f"recurrence-generation-ab-ladder: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "campaign_root": result["campaign_root"],
                "status": result["status"],
                "summary": result["summary"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return (
        0
        if result["status"] in {"planned", "passed", "completed-with-censoring"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
