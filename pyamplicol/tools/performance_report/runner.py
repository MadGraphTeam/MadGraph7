# SPDX-License-Identifier: 0BSD
"""Direct Python-API generation, profiling, and validation for report cells."""

from __future__ import annotations

import hashlib
import importlib
import importlib.resources
import io
import json
import math
import os
import platform
import re
import stat
import statistics
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyamplicol.config import BenchmarkConfig
    from pyamplicol.reporting import ProgressSink

from .arena_profile import (
    ARENA_PHASE_TIMING_SCOPE,
    ARENA_PROFILE_BOUNDARY,
    ARENA_PROFILE_PROTOCOL,
    ARENA_PROFILE_SAMPLE_PASS,
    PAIRED_TIMING_SAMPLE_CONTRACT,
    ArenaProfileEvidenceError,
    build_arena_profile_evidence,
    digest_arena_profile_value,
    validate_arena_profile_evidence,
)
from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .phase_state import WorkerPhaseReporter
from .runtime_evidence import (
    RuntimeEvidenceError,
    established_preimport_runtime_identity,
    loaded_pyamplicol_origin_policy,
    python_package_tree_identity,
)
from .selector_policy import (
    SelectorPolicyError,
    canonical_lc_flow_word,
    canonical_lc_selector_word,
    fixed_selector_helicity,
    selector_color_flow_id,
    selector_helicity_id,
)
from .timing import (
    ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI,
    EVALUATOR_TOTAL_SAMPLE_CONTRACT,
    EVALUATOR_TOTAL_TIMING_ABI,
    EVALUATOR_TOTAL_TIMING_KEY,
    EVALUATOR_TOTAL_TIMING_SOURCE,
    MEASURED_EXECUTION_TIMING_ABI,
    UNAVAILABLE_STATUS,
    evaluator_total_timing_record,
)

RELATIVE_TOLERANCE = 1.0e-12
INDEPENDENT_RELATIVE_TOLERANCE = 1.0e-8
MADGRAPH_RELATIVE_TOLERANCE = 1.0e-10
ABSOLUTE_TOLERANCE = 1.0e-15
CONDITIONED_COMPARISON_ABI = "pyamplicol-report-conditioned-comparison-v2"
RESOLVED_SUM_VALIDATION_ABI = "pyamplicol-report-resolved-sum-validation-v2"
RESOLVED_COMPONENT_SCALE_ABI = "pyamplicol-report-resolved-component-scale-v1"
OTF_RECURRENCE_AUTHORITY_VALIDATION_ABI = (
    "pyamplicol-report-on-the-fly-recurrence-authority-validation-v1"
)
OTF_RECURRENCE_AUTHORITY_VALIDATION_FIELD = "on_the_fly_recurrence_authority"
OTF_COLD_WARMUP_FIELDS = frozenset(
    {
        "cold_warmup_elapsed_seconds",
        "cold_warmup_run_count",
        "cold_warmup_batch_size",
        "cold_warmup_point_count",
        "cold_warmup_timer_source",
        "cold_warmup_timing_scope",
        "cold_warmup_runtime_freshness",
        "cold_warmup_runtime_state_evidence",
        "cold_warmup_runtime_state_before",
        "cold_warmup_runtime_state_after",
        "cold_warmup_runtime_cold_before_first_evaluation",
        "cold_warmup_runtime_retained_before_first_evaluation",
        "cold_warmup_runtime_retained_after_first_evaluation",
        "cold_warmup_ratio_eligible",
        "cold_warmup_acceptance_eligible",
    }
)
CONVENTIONAL_WARMUP_FIELDS = frozenset(
    {
        "warmup_elapsed_seconds",
        "warmup_configured_run_count",
        "warmup_batch_size",
        "warmup_point_count",
        "warmup_run_outer_wall_seconds",
        "first_warmup_run_outer_wall_seconds",
        "warmup_timer_source",
        "warmup_timing_scope",
    }
)
WARMUP_TIMER_SOURCE = "python_outer_time.perf_counter"
OTF_COLD_WARMUP_TIMING_SCOPE = (
    "one initial requested-selector Runtime evaluation on the full benchmark "
    "batch; artifact generation and Runtime/artifact load are excluded"
)
OTF_COLD_WARMUP_RUNTIME_FRESHNESS = "authenticated-cold"
OTF_COLD_WARMUP_RUNTIME_RETAINED_FRESHNESS = "authenticated-already-retained"
OTF_COLD_WARMUP_RUNTIME_FRESHNESSES = frozenset(
    {
        OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
        OTF_COLD_WARMUP_RUNTIME_RETAINED_FRESHNESS,
    }
)
OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE = "authenticated-native-otf-census-v1"
OTF_RUNTIME_STATE_CENSUS_KIND = "rusticol-on-the-fly-runtime-state-census-v1"
OTF_RUNTIME_STATE_FAMILY_CACHE_POLICY = "last-family-only"
OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT = 1
OTF_RUNTIME_STATE_COUNT_FIELDS = (
    "process_preparation_count",
    "retained_family_count",
    "pending_family_count",
    "retained_selection_count",
    "retained_request_count",
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
    "semantic_executor_binding_count",
)
OTF_RUNTIME_STATE_RETAINED_BASE_POSITIVE_FIELDS = (
    "process_preparation_count",
    "retained_family_count",
    "retained_selection_count",
    "retained_request_count",
)
OTF_RUNTIME_STATE_RETAINED_EXECUTABLE_FIELDS = (
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "semantic_executor_binding_count",
)
OTF_ACTIVE_FAMILY_COUNT_FIELDS = (
    "query_count",
    "union_unique_current_count",
    "union_unique_current_component_count",
    "union_source_rows",
    "union_contribution_rows",
    "union_finalization_rows",
    "union_closure_rows",
    "union_amplitude_destination_count",
    "union_source_executor_call_groups",
    "union_contribution_executor_call_groups",
    "union_finalization_executor_call_groups",
    "union_closure_executor_call_groups",
)
OTF_OPERATION_ROLES = ("source", "contribution", "finalization", "closure")
CONVENTIONAL_WARMUP_TIMING_SCOPE = (
    "configured benchmark warm-up iteration outer wall; includes the headline "
    "evaluation and optional native-profile warm-up; artifact generation and "
    "Runtime/artifact load are excluded"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GENERATION_VALIDATION_SEED = 12345
DEFAULT_TARGET_RUNTIME_SECONDS = 5.0
REPORT_BENCHMARK_BATCH_SIZE = 128
REPORT_BENCHMARK_WARMUP_RUNS = 2
_ARENA_MINIMUM_SAMPLES = 5
_ARENA_MAX_SAMPLES = 64
_ARENA_MAX_CALIBRATION_BLOCKS = 4
_ARENA_MAX_REPETITIONS = 1_000_000_000
_ARENA_MINIMUM_TARGET_FRACTION = 0.95
_MIB = 1024 * 1024
_RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
_RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
_RECURRENCE_DIRECT_CANONICALIZATION_ABI = "pyamplicol-canonical-json-v1"
_RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI = "pyamplicol-recurrence-plane-binding-v2"
_SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
_SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
_RECURRENCE_JIT_SOURCE_APPLICATION_ABI = _SYMJIT_PLANE_APPLICATION_ABI
_RECURRENCE_JIT_DIRECT_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
_RECURRENCE_JIT_SOURCE_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"

# These labels make the public-command boundary, and its two report-only
# exceptions, explicit in returned generation/profile evidence.
PUBLIC_CLI_COMMAND_PATH = "pyamplicol-cli-parse-resolve-dispatch-v1"
LOADED_RUNTIME_PROFILE_COMMAND_PATH = (
    "pyamplicol-profile-loaded-runtime-and-points-injection-v1"
)
PAIRED_ARENA_PROFILE_COMMAND_PATH = "pyamplicol-report-private-paired-arena-profile-v1"
PRECOMPILED_GENERATION_COMMAND_PATH = (
    "pyamplicol-generate-precompiled-model-injection-v1"
)


class RunnerError(RuntimeError):
    """Raised when a cell cannot satisfy the report measurement contract."""


class ProfilingTimeLimitError(RunnerError):
    """Raised before a profiling chunk cannot fit its remaining stage budget."""


ProfilingChunkGuard = Callable[[float | None, str], None]


def profiling_chunk_guard(
    deadline_monotonic: float | None,
    *,
    clock: Callable[[], float] | None = None,
) -> ProfilingChunkGuard | None:
    """Return a cheap pre-launch guard for one absolute profiling deadline."""

    if deadline_monotonic is None:
        return None
    if not math.isfinite(deadline_monotonic) or deadline_monotonic <= 0.0:
        raise ValueError("profiling deadline must be finite and positive")
    selected_clock = time.monotonic if clock is None else clock

    def guard(estimated_seconds: float | None, description: str) -> None:
        if not isinstance(description, str) or not description:
            raise ValueError("profiling chunk description must be non-empty")
        if estimated_seconds is not None and (
            not math.isfinite(estimated_seconds) or estimated_seconds < 0.0
        ):
            raise ValueError("profiling chunk estimate must be finite and non-negative")
        remaining = deadline_monotonic - selected_clock()
        if remaining <= 0.0 or (
            estimated_seconds is not None and estimated_seconds > remaining
        ):
            estimate = (
                "unknown" if estimated_seconds is None else f"{estimated_seconds:.6g}s"
            )
            raise ProfilingTimeLimitError(
                "profiling stage has insufficient remaining budget for "
                f"{description}: remaining={max(remaining, 0.0):.6g}s, "
                f"estimated={estimate}"
            )

    return guard


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS
    batch_size: int = REPORT_BENCHMARK_BATCH_SIZE
    worker_cores: int = 1
    memory_limit_bytes: int | None = None
    warmup_runs: int = REPORT_BENCHMARK_WARMUP_RUNS
    minimum_samples: int = 5
    model_cache_dir: Path | None = None
    progress: ProgressSink | None = None
    source_revision_override: str | None = None
    profiling_time_limit_seconds: float | None = None
    worker_deadline_monotonic: float | None = None

    def __post_init__(self) -> None:
        if self.target_runtime_seconds <= 0.0:
            raise ValueError("target_runtime_seconds must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.worker_cores < 1:
            raise ValueError("worker_cores must be positive")
        if self.memory_limit_bytes is not None and (
            isinstance(self.memory_limit_bytes, bool)
            or not isinstance(self.memory_limit_bytes, int)
            or self.memory_limit_bytes < _MIB
        ):
            raise ValueError("memory_limit_bytes must be at least one MiB")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.minimum_samples < _ARENA_MINIMUM_SAMPLES:
            raise ValueError(
                f"minimum_samples must be at least {_ARENA_MINIMUM_SAMPLES}"
            )
        if self.progress is not None and not callable(
            getattr(self.progress, "emit", None)
        ):
            raise TypeError("progress must implement ProgressSink.emit(event)")
        if (
            self.source_revision_override is not None
            and not self.source_revision_override
        ):
            raise ValueError("source_revision_override must be non-empty")
        for name, value in (
            ("profiling_time_limit_seconds", self.profiling_time_limit_seconds),
            ("worker_deadline_monotonic", self.worker_deadline_monotonic),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when specified")


@dataclass(frozen=True, slots=True)
class SelectorContract:
    selected_color_flow_ids: tuple[str, ...]
    selected_color_words: tuple[tuple[int, ...], ...]
    all_flow_helicity_ids: tuple[str, ...]
    all_flow_source_helicities: tuple[tuple[int, int], ...]
    point_digest: str

    def __post_init__(self) -> None:
        if not self.selected_color_flow_ids:
            raise ValueError("selector contract requires a selected color flow")
        if len(self.selected_color_flow_ids) != len(self.selected_color_words):
            raise ValueError("color-flow IDs and words must have equal length")
        if len(set(self.selected_color_flow_ids)) != len(self.selected_color_flow_ids):
            raise ValueError("selector contract color-flow IDs must be unique")
        if len(self.all_flow_helicity_ids) != 1:
            raise ValueError("selector contract requires one fixed helicity")
        if not self.all_flow_source_helicities:
            raise ValueError("selector contract requires source helicities")
        if len(self.point_digest) != 64:
            raise ValueError("selector contract point digest must be SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_color_flow_ids": list(self.selected_color_flow_ids),
            "selected_color_words": [list(word) for word in self.selected_color_words],
            "all_flow_helicity_ids": list(self.all_flow_helicity_ids),
            "all_flow_source_helicities": {
                str(label): value for label, value in self.all_flow_source_helicities
            },
            "point_digest": self.point_digest,
        }

    @property
    def runtime_all_flow_helicity_ids(self) -> tuple[str, ...]:
        """Return exact public runtime IDs for the stored physical states.

        Historical original-AmpliCol contracts encoded a zero-helicity state as
        ``0`` while generated runtime axes use the canonical signed spelling
        ``+0``.  Treat only that textual alias as equivalent: the parsed ID must
        reproduce every stored source state before it is canonicalized.
        """

        states = tuple(
            state for _label, state in sorted(self.all_flow_source_helicities)
        )
        resolved: list[str] = []
        for identifier in self.all_flow_helicity_ids:
            if not identifier.startswith("h:"):
                resolved.append(identifier)
                continue
            try:
                encoded = tuple(
                    int(token) for token in identifier.removeprefix("h:").split(",")
                )
            except ValueError:
                resolved.append(identifier)
                continue
            if encoded != states:
                resolved.append(identifier)
                continue
            resolved.append("h:" + ",".join(f"{state:+d}" for state in states))
        return tuple(resolved)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectorContract:
        raw_ids = value.get("selected_color_flow_ids")
        raw_words = value.get("selected_color_words")
        raw_helicities = value.get("all_flow_helicity_ids")
        raw_sources = value.get("all_flow_source_helicities")
        point_digest = value.get("point_digest")
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_words, Sequence)
            or isinstance(raw_words, (str, bytes))
            or not isinstance(raw_helicities, Sequence)
            or isinstance(raw_helicities, (str, bytes))
            or not isinstance(raw_sources, Mapping)
            or not isinstance(point_digest, str)
        ):
            raise ValueError("selector contract has an invalid shape")
        return cls(
            selected_color_flow_ids=tuple(str(item) for item in raw_ids),
            selected_color_words=tuple(
                tuple(int(label) for label in word)  # type: ignore[arg-type]
                for word in raw_words
            ),
            all_flow_helicity_ids=tuple(str(item) for item in raw_helicities),
            all_flow_source_helicities=tuple(
                sorted((int(label), int(state)) for label, state in raw_sources.items())
            ),
            point_digest=point_digest,
        )


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    path: Path
    process_id: str
    generation_seconds: float
    model_preparation_seconds: float
    model_preparation_reused: bool
    requested_config: Mapping[str, object]
    effective_config: Mapping[str, object]
    generation_command_path: str | None = None
    numerical_relation_correctness: Mapping[str, object] | None = None
    numerical_relation_fallback: Mapping[str, object] | None = None


class RuntimeLike(Protocol):
    @property
    def physics(self) -> object: ...

    def evaluate(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> Sequence[object]: ...

    def evaluate_resolved(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> object: ...

    def clear(self) -> None: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda item: item.tolist(),
    ).encode("ascii")


def _regular_file_identity(path: Path) -> tuple[int, str]:
    """Hash one unchanged regular file through a single no-follow descriptor."""

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
            raise RunnerError(f"identity target is not a regular file: {path}")
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise RunnerError(
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
        raise RunnerError(f"identity target changed while it was hashed: {path}")
    return byte_count, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _regular_file_identity(path)[1]


def point_digest(points: object) -> str:
    return hashlib.sha256(_canonical_json(points)).hexdigest()


def _scale_decimal(value: object) -> Decimal:
    """Return one finite non-negative magnitude without binary64 accumulation."""

    try:
        magnitude = abs(value)  # type: ignore[arg-type]
        decimal = Decimal(str(magnitude))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RunnerError("resolved component magnitude is not numeric") from error
    if not decimal.is_finite() or decimal < 0:
        raise RunnerError("resolved component magnitude is not finite")
    return decimal


def _resolved_l1(value: object) -> Decimal:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum((_resolved_l1(item) for item in value), Decimal(0))
    return _scale_decimal(value)


def _exact_number_text(value: object) -> str:
    """Serialize a real component deterministically for a source binding."""

    if isinstance(value, bool):
        raise RunnerError("resolved component is boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerError("resolved component is not finite")
        return value.hex()
    number = float(value)  # validates arbitrary-precision real values cheaply
    if not math.isfinite(number):
        raise RunnerError("resolved component is not finite")
    return str(value)


def _resolved_component_payload(value: object) -> object:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolved_component_payload(item) for item in value]
    real = getattr(value, "real", value)
    imag = getattr(value, "imag", 0)
    if callable(real):
        real = real()
    if callable(imag):
        imag = imag()
    return {"real": _exact_number_text(real), "imag": _exact_number_text(imag)}


def resolved_component_scale_evidence(
    resolved: object,
    points: object,
) -> dict[str, object]:
    """Bind per-point L1 scales to exact axes, ordering, values and momenta."""

    values = getattr(resolved, "values", None)
    to_list = getattr(values, "tolist", None)
    if callable(to_list):
        values = to_list()
    if not isinstance(values, Sequence) or not values:
        raise RunnerError("resolved evaluation does not expose point components")
    helicity_ids = tuple(str(item) for item in getattr(resolved, "helicity_ids", ()))
    color_flow_ids = tuple(str(item) for item in getattr(resolved, "color_ids", ()))
    ordered = {
        "helicity_ids": list(helicity_ids),
        "color_flow_ids": list(color_flow_ids),
        "values": _resolved_component_payload(values),
    }
    source_digest = hashlib.sha256(_canonical_json(ordered)).hexdigest()
    ordering_digest = hashlib.sha256(
        _canonical_json(
            {
                "helicity_ids": list(helicity_ids),
                "color_flow_ids": list(color_flow_ids),
            }
        )
    ).hexdigest()
    scales: list[float] = []
    for point_values in values:
        scale = float(_resolved_l1(point_values))
        if not math.isfinite(scale) or scale < 0.0:
            raise RunnerError("resolved component L1 scale is not finite")
        scales.append(scale)
    return {
        "abi": RESOLVED_COMPONENT_SCALE_ABI,
        "point_digest": point_digest(points),
        "helicity_ids": list(helicity_ids),
        "color_flow_ids": list(color_flow_ids),
        "resolved_ordering_sha256": ordering_digest,
        "resolved_source_sha256": source_digest,
        "scales": scales,
    }


def validate_conditioned_comparison_record(
    value: object,
    *,
    require_binding: bool,
) -> None:
    """Authenticate a v2 conditioned comparison or a safely reusable v1 row."""

    if not isinstance(value, Mapping):
        raise ValueError("conditioned comparison must be an object")
    if value.get("abi") != CONDITIONED_COMPARISON_ABI:
        # Schema-free v1 rows can be reused only when the stored numbers
        # recompute and satisfy the symmetric, floor-free criterion.
        expected_v1 = {
            "status",
            "candidate",
            "baseline",
            "absolute_difference",
            "relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
        }
        if set(value) != expected_v1:
            raise ValueError("conditioned comparison ABI is unsupported")
        numeric: dict[str, float] = {}
        for field in expected_v1 - {"status"}:
            raw = value.get(field)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError(f"legacy comparison {field} is not finite")
            numeric[field] = float(raw)
        delta = abs(numeric["candidate"] - numeric["baseline"])
        relative = delta / max(abs(numeric["baseline"]), 1.0e-300)
        old_passed = (
            delta <= numeric["absolute_tolerance"]
            or relative <= numeric["relative_tolerance"]
        )
        safe = delta <= numeric["relative_tolerance"] * max(
            abs(numeric["candidate"]), abs(numeric["baseline"])
        )
        expected_status = (
            ResultStatus.OK.value
            if old_passed
            else ResultStatus.VALIDATION_FAILED.value
        )
        if numeric["absolute_difference"] != delta:
            raise ValueError("legacy absolute difference was not recomputed")
        if numeric["relative_difference"] != relative:
            raise ValueError("legacy relative difference was not recomputed")
        if value.get("status") != expected_status:
            raise ValueError("legacy comparison status is inconsistent")
        if expected_status == ResultStatus.OK.value and not safe:
            raise ValueError("legacy comparison is not safely reusable")
        return

    expected_v2 = {
        "abi",
        "status",
        "candidate",
        "baseline",
        "candidate_scale",
        "baseline_scale",
        "candidate_scale_source",
        "baseline_scale_source",
        "comparison_scale",
        "absolute_difference",
        "relative_difference",
        "conditioned_residual",
        "error_bound",
        "relative_tolerance",
        "comparison_binding",
    }
    if set(value) != expected_v2:
        raise ValueError("conditioned comparison fields differ from the v2 ABI")
    numeric = {}
    for field in (
        "candidate",
        "baseline",
        "candidate_scale",
        "baseline_scale",
        "comparison_scale",
        "absolute_difference",
        "relative_difference",
        "conditioned_residual",
        "error_bound",
        "relative_tolerance",
    ):
        raw = value.get(field)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"conditioned comparison {field} is not finite")
        numeric[field] = float(raw)
    if any(
        numeric[field] < 0.0
        for field in (
            "candidate_scale",
            "baseline_scale",
            "comparison_scale",
            "absolute_difference",
            "relative_difference",
            "conditioned_residual",
            "error_bound",
            "relative_tolerance",
        )
    ):
        raise ValueError("conditioned comparison contains a negative magnitude")
    for field in ("candidate_scale_source", "baseline_scale_source"):
        raw = value.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"conditioned comparison {field} is invalid")
    binding = value.get("comparison_binding")
    if require_binding and (not isinstance(binding, Mapping) or not binding):
        raise ValueError("conditioned comparison is not bound to its source")
    if binding is not None and not isinstance(binding, Mapping):
        raise ValueError("conditioned comparison binding is invalid")
    if require_binding:
        assert isinstance(binding, Mapping)
        point_identity = binding.get("point_digest")
        if not isinstance(point_identity, str) or not _SHA256_PATTERN.fullmatch(
            point_identity
        ):
            raise ValueError("conditioned comparison point binding is invalid")
        if binding.get("abi") == RESOLVED_COMPONENT_SCALE_ABI:
            for field in ("resolved_ordering_sha256", "resolved_source_sha256"):
                raw_digest = binding.get(field)
                if not isinstance(raw_digest, str) or not _SHA256_PATTERN.fullmatch(
                    raw_digest
                ):
                    raise ValueError(
                        f"conditioned comparison {field} binding is invalid"
                    )
        else:
            expected_binding_fields = {
                "point_digest",
                "selector_component_identity",
                "selector_component_sha256",
                "candidate_source_sha256",
                "baseline_source_sha256",
            }
            if set(binding) != expected_binding_fields:
                raise ValueError("conditioned comparison source binding fields differ")
            selector_identity = binding.get("selector_component_identity")
            if not isinstance(selector_identity, Mapping) or not selector_identity:
                raise ValueError(
                    "conditioned comparison selector/component binding is invalid"
                )
            expected_selector_digest = hashlib.sha256(
                _canonical_json(selector_identity)
            ).hexdigest()
            if binding.get("selector_component_sha256") != expected_selector_digest:
                raise ValueError(
                    "conditioned comparison selector/component digest differs"
                )
            for field in ("candidate_source_sha256", "baseline_source_sha256"):
                raw_digest = binding.get(field)
                if not isinstance(raw_digest, str) or not _SHA256_PATTERN.fullmatch(
                    raw_digest
                ):
                    raise ValueError(
                        f"conditioned comparison {field} binding is invalid"
                    )
    candidate = numeric["candidate"]
    baseline = numeric["baseline"]
    candidate_scale = numeric["candidate_scale"]
    baseline_scale = numeric["baseline_scale"]
    if candidate_scale < abs(candidate) or baseline_scale < abs(baseline):
        raise ValueError("conditioned comparison scale is below its value")
    delta = abs(candidate - baseline)
    raw_relative = delta / max(abs(baseline), 1.0e-300)
    scale = max(abs(candidate), abs(baseline), candidate_scale, baseline_scale)
    if scale == 0.0:
        if candidate != 0.0 or baseline != 0.0 or delta != 0.0:
            raise ValueError("zero scale comparison contains a nonzero value")
        residual = 0.0
    else:
        residual = delta / scale
    bound = numeric["relative_tolerance"] * scale
    status = (
        ResultStatus.OK.value
        if delta <= bound
        else ResultStatus.VALIDATION_FAILED.value
    )
    if (
        numeric["comparison_scale"] != scale
        or numeric["absolute_difference"] != delta
        or numeric["relative_difference"] != raw_relative
        or numeric["conditioned_residual"] != residual
        or numeric["error_bound"] != bound
        or value.get("status") != status
    ):
        raise ValueError("conditioned comparison numerical evidence is inconsistent")


def _real_nonnegative(value: object) -> float:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise RunnerError("matrix element is not finite")
    if abs(number.imag) > 1.0e-9 * max(abs(number.real), 1.0):
        raise RunnerError(
            f"matrix element has a non-negligible imaginary part: {value}"
        )
    result = float(number.real)
    if result < -ABSOLUTE_TOLERANCE:
        raise RunnerError(f"matrix element is materially negative: {value}")
    return max(result, 0.0)


def _model_source_path(repo_root: Path, model: ModelKey) -> Path | None:
    if model is ModelKey.BUILTIN_SM:
        return None
    relative = {
        ModelKey.UFO_SM: Path("sm/sm.json"),
        ModelKey.SCALAR_CONTACT: Path("scalars/scalars.json"),
        ModelKey.SCALAR_GRAVITY: Path("scalar_gravity/scalar_gravity.json"),
    }.get(model)
    if relative is None:
        raise RunnerError(f"model {model.value!r} is not supported by process matrices")
    source_path = repo_root / "src/pyamplicol/assets/models/json" / relative
    if source_path.is_file():
        return source_path
    resource = importlib.resources.files("pyamplicol").joinpath(
        "assets",
        "models",
        "json",
        *relative.parts,
    )
    if not isinstance(resource, os.PathLike):
        raise RunnerError("installed model resources are not filesystem-backed")
    try:
        return Path(os.fspath(resource)).resolve(strict=True)
    except OSError as error:
        raise RunnerError(
            f"installed model resource is unavailable: {relative.as_posix()}"
        ) from error


def config_values(
    cell: CellSpec,
    settings: RunnerSettings,
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Return complete generation/benchmark settings for one pyAmpliCol cell."""

    measurement = cell.measurement
    if measurement.execution_mode is ExecutionMode.AMPLICOL:
        raise RunnerError("original AmpliCol does not use the pyAmpliCol runner")
    if measurement.model is None:
        raise RunnerError("pyAmpliCol measurement requires an explicit model")
    on_the_fly = measurement.execution_mode is ExecutionMode.ON_THE_FLY
    if on_the_fly and (
        measurement.backend != "jit" or measurement.jit_optimization_level != 2
    ):
        raise RunnerError("on-the-fly campaign measurements require prepared JIT O2")
    model_path = _model_source_path(repo_root, measurement.model)
    layout = (
        "topology-replay"
        if on_the_fly
        else (
            "all-flow-union"
            if cell.workload is Workload.ALL_FLOW
            else "topology-replay"
        )
    )
    values: dict[str, object] = {
        "model": {
            "source": "built-in-sm" if model_path is None else os.fspath(model_path),
            "cache": True,
            "cache_dir": (
                None
                if settings.model_cache_dir is None
                else os.fspath(settings.model_cache_dir)
            ),
        },
        "color": {
            "accuracy": measurement.accuracy.value,
            "lc_flow_layout": layout,
        },
        "generation": {
            "workers": settings.worker_cores,
            "emit_api_bundle": not on_the_fly,
            "validation": {
                "enabled": not on_the_fly,
                "samples": 1,
                "seed": GENERATION_VALIDATION_SEED,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "absolute_tolerance": 1.0e-300,
                # The campaign immediately loads the artifact and performs its
                # own resolved, high-precision, and authority validation.
                "post_build_validation": False,
            },
            **({"relation_discovery": {"mode": "off"}} if on_the_fly else {}),
        },
        "evaluator": {
            "backend": measurement.backend,
            "execution_mode": measurement.execution_mode.value,
            "batch_size": settings.batch_size,
            "output_chunk_size": 512,
            "optimization": {
                "horner_iterations": 10,
                "cpe_iterations": None,
                "cores": settings.worker_cores,
                "max_horner_variables": 1000,
                "max_common_pair_cache_entries": 5_000_000,
                "max_common_pair_distance": 1000,
            },
            "jit": {
                "optimization_level": measurement.jit_optimization_level or 2,
            },
            "cpp": {"optimization": "O3"},
        },
        "benchmark": {
            "target_runtime": settings.target_runtime_seconds,
            "batch_size": settings.batch_size,
            "warmup_runs": settings.warmup_runs,
            "minimum_samples": settings.minimum_samples,
        },
        "output": {"format": "json", "progress": "off"},
    }
    if settings.memory_limit_bytes is not None and measurement.execution_mode in {
        ExecutionMode.EAGER,
        ExecutionMode.RECURRENCE,
    }:
        workspace_mib = settings.memory_limit_bytes // _MIB
        evaluator = values["evaluator"]
        assert isinstance(evaluator, dict)
        evaluator[measurement.execution_mode.value] = {"workspace_mib": workspace_mib}
    return values


def _cli_override_value(value: object) -> str:
    """Encode one config leaf using the public ``--set`` value grammar."""

    if value is None:
        return "null"
    if isinstance(value, Path):
        value = os.fspath(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _cli_overrides(
    values: Mapping[str, object],
    *,
    prefix: str = "",
) -> tuple[str, ...]:
    """Flatten config values into deterministic public ``--set`` arguments."""

    arguments: list[str] = []
    for key in sorted(values):
        path = f"{prefix}.{key}" if prefix else key
        value = values[key]
        if isinstance(value, Mapping):
            arguments.extend(_cli_overrides(value, prefix=path))
            continue
        arguments.extend(("--set", f"{path}={_cli_override_value(value)}"))
    return tuple(arguments)


def _generation_cli_argv(
    cell: CellSpec,
    destination: Path,
    values: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the exact public ``pyamplicol generate`` argument vector."""

    command_values = {key: value for key, value in values.items() if key != "process"}
    return (
        "generate",
        cell.process,
        os.fspath(destination),
        *_cli_overrides(command_values),
    )


def _physics_ids(physics: object, name: str) -> tuple[str, ...]:
    return tuple(str(item.id) for item in getattr(physics, name, ()))


def _on_the_fly_compact_context(
    cell: CellSpec,
    runtime: RuntimeLike,
) -> tuple[int, int]:
    """Validate OTF process identity and return compact selector counts."""

    backend = getattr(runtime, "_backend", runtime)
    operation = getattr(backend, "_on_the_fly_benchmark_context", None)
    if not callable(operation):
        raise RunnerError("on-the-fly report runtime has no compact selector context")
    try:
        raw = operation(())
    except Exception as error:
        raise RunnerError(
            "on-the-fly report runtime compact selector context is unavailable"
        ) from error
    if not isinstance(raw, Mapping):
        raise RunnerError(
            "on-the-fly report runtime compact selector context is invalid"
        )
    process_id = raw.get("process_id")
    selected_color_ids = raw.get("selected_color_ids")
    if (
        not isinstance(process_id, str)
        or not process_id
        or raw.get("process_expression") != cell.process
        or raw.get("color_accuracy") != cell.measurement.accuracy.value
        or not isinstance(selected_color_ids, Sequence)
        or isinstance(selected_color_ids, (str, bytes))
        or selected_color_ids
    ):
        raise RunnerError("on-the-fly report runtime compact selector identity differs")
    counts: list[int] = []
    for field in ("helicity_count", "color_count"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RunnerError(f"on-the-fly report runtime compact {field} is invalid")
        counts.append(value)
    return counts[0], counts[1]


def _on_the_fly_selector_ordinals(
    runtime: RuntimeLike,
    values: Sequence[str],
    *,
    name: str,
    count: int,
) -> tuple[int, ...]:
    backend = getattr(runtime, "_backend", runtime)
    operation = getattr(backend, "_point_selector_indices", None)
    if not callable(operation):
        raise RunnerError("on-the-fly report runtime has no compact selector resolver")
    try:
        raw = operation(values, name)
    except Exception as error:
        raise RunnerError(f"on-the-fly report runtime cannot resolve {name}") from error
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != len(values)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= count
            for value in raw
        )
        or len(set(raw)) != len(raw)
    ):
        raise RunnerError(f"on-the-fly report runtime returned invalid {name} ordinals")
    return tuple(raw)


def validate_runtime_contract(cell: CellSpec, runtime: RuntimeLike) -> None:
    runtime_is_on_the_fly = (
        _runtime_execution_mode(runtime) == ExecutionMode.ON_THE_FLY.value
    )
    cell_is_on_the_fly = cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    if runtime_is_on_the_fly:
        if not cell_is_on_the_fly:
            raise RunnerError("on-the-fly report runtime has the wrong execution mode")
        _helicity_count, color_count = _on_the_fly_compact_context(cell, runtime)
        if cell.measurement.accuracy is Accuracy.LC:
            if cell.workload not in {Workload.SELECTED_FLOW, Workload.ALL_FLOW}:
                raise RunnerError("LC on-the-fly measurement has an invalid workload")
        elif (
            cell.measurement.accuracy not in {Accuracy.NLC, Accuracy.FULL}
            or cell.workload is not Workload.CONTRACTED
            or color_count != 1
        ):
            raise RunnerError(
                "contracted on-the-fly measurement has an invalid public color axis"
            )
        return
    if cell_is_on_the_fly:
        raise RunnerError("on-the-fly report runtime has the wrong execution mode")
    physics = runtime.physics
    accuracy = str(getattr(physics, "color_accuracy", ""))
    if accuracy != cell.measurement.accuracy.value:
        raise RunnerError(
            f"artifact color accuracy {accuracy!r} does not match "
            f"{cell.measurement.accuracy.value!r}"
        )
    capabilities = set(getattr(physics, "selector_capabilities", ()))
    helicity_ids = _physics_ids(physics, "helicities")
    if not helicity_ids:
        raise RunnerError("artifact exposes no physical helicities")
    if cell.measurement.accuracy is Accuracy.LC:
        color_ids = _physics_ids(physics, "color_flows")
        if not color_ids:
            raise RunnerError("LC artifact exposes no physical color flows")
        missing = {"helicity", "color_flow"} - capabilities
        if missing:
            raise RunnerError(
                "LC artifact does not retain complete runtime selectors: "
                + ", ".join(sorted(missing))
            )
    elif cell.workload is not Workload.CONTRACTED:
        raise RunnerError("NLC/full measurements must use the contracted workload")


def validate_artifact_contract(cell: CellSpec, artifact_path: Path) -> None:
    from pyamplicol._internal.versions import (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_RUNTIME_CAPABILITY,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    )
    from pyamplicol.artifacts import inspect_artifact

    inspection = inspect_artifact(artifact_path)
    if len(inspection.processes) != 1:
        raise RunnerError("report artifacts must contain exactly one process")
    process = inspection.processes[0]
    if process.execution_mode != cell.measurement.execution_mode.value:
        raise RunnerError(
            f"artifact execution mode {process.execution_mode!r} does not match "
            f"{cell.measurement.execution_mode.value!r}"
        )
    if process.generation_specialized_axes:
        raise RunnerError(
            "report artifacts must retain complete runtime coverage; specialized "
            f"axes: {process.generation_specialized_axes}"
        )
    if process.selected_source_helicities or process.selected_color_sector_ids:
        raise RunnerError("report artifact contains forbidden generation selectors")
    expected_runtime_capability = {
        ExecutionMode.COMPILED: COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        ExecutionMode.EAGER: EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        ExecutionMode.RECURRENCE: RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        ExecutionMode.ON_THE_FLY: ON_THE_FLY_RUNTIME_CAPABILITY,
    }[cell.measurement.execution_mode]
    runtime_capabilities = set(inspection.runtime_capabilities)
    if expected_runtime_capability not in runtime_capabilities:
        raise RunnerError(
            "report artifact does not require its authenticated execution lane "
            f"{expected_runtime_capability!r}"
        )
    if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY:
        expected_color_capability = (
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY
            if cell.measurement.accuracy is Accuracy.LC
            else ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        )
        if len(inspection.runtime_capabilities) != 2 or runtime_capabilities != {
            ON_THE_FLY_RUNTIME_CAPABILITY,
            expected_color_capability,
        }:
            raise RunnerError(
                "on-the-fly report artifact does not expose exactly its two "
                "accuracy-specific runtime capabilities"
            )
        if cell.measurement.accuracy is not Accuracy.LC and (
            cell.workload is not Workload.CONTRACTED
            or process.recurrence_color_accuracy != cell.measurement.accuracy.value
            or process.recurrence_color_storage != "expanded"
            or process.recurrence_color_component_count != 1
            or process.recurrence_color_group_count is None
            or process.recurrence_color_group_count < 1
            or process.recurrence_color_destination_count
            != process.recurrence_color_group_count
        ):
            raise RunnerError(
                "contracted on-the-fly report artifact has no authenticated "
                "expanded color payload"
            )
    if cell.measurement.accuracy is Accuracy.LC:
        expected_layout = (
            "compact/query-local"
            if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
            else (
                "all-flow-union"
                if cell.workload is Workload.ALL_FLOW
                else "topology-replay"
            )
        )
        if process.lc_flow_layout != expected_layout:
            raise RunnerError(
                f"artifact LC layout {process.lc_flow_layout!r} does not match "
                f"{expected_layout!r}"
            )


def runtime_identity_payload(
    cell: CellSpec,
    runtime: object,
    artifact_path: Path,
    process_id: str,
    *,
    expected_source_revision: str,
) -> dict[str, object]:
    """Return the exact artifact/native-Arena identity used by one report cell."""

    import pyamplicol
    from pyamplicol._internal.versions import (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        EAGER_DIRECT_TABLE_BINDING_ABI,
        NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
        ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_RUNTIME_CAPABILITY,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMJIT_PLANE_APPLICATION_ABI,
        _active_build_info,
        verify_native_module,
    )
    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(artifact_path, verify_payloads=False)
    matches = [
        process for process in manifest.processes if process.get("id") == process_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"report runtime process {process_id!r} is missing or ambiguous"
        )
    process = matches[0]
    loaded_artifact_id = getattr(runtime, "artifact_id", None)
    if loaded_artifact_id != manifest.artifact_id:
        raise RunnerError(
            "native runtime loaded artifact identity does not match the report "
            "artifact manifest"
        )
    try:
        loaded_execution_mode = runtime.execution_mode  # type: ignore[attr-defined]
    except Exception as exc:
        raise RunnerError(
            "native runtime does not expose an authenticated execution mode"
        ) from exc
    expected_execution_mode = cell.measurement.execution_mode.value
    if loaded_execution_mode != expected_execution_mode:
        raise RunnerError(
            f"native runtime loaded execution mode {loaded_execution_mode!r} "
            f"does not match the requested report mode {expected_execution_mode!r}"
        )
    required_capabilities = tuple(
        str(value) for value in process["required_runtime_capabilities"]
    )
    if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY and (
        len(required_capabilities) != 2
        or set(required_capabilities)
        != {
            ON_THE_FLY_RUNTIME_CAPABILITY,
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        }
    ):
        raise RunnerError(
            "on-the-fly report runtime identity requires exactly its two "
            "compact LC capabilities"
        )
    arena_capability, evaluator_abi, source_evaluator_abi = {
        ExecutionMode.EAGER: (
            EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            SYMJIT_PLANE_APPLICATION_ABI,
        ),
        ExecutionMode.RECURRENCE: (
            RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
            RECURRENCE_RUNTIME_LAYOUT_ABI,
            (
                SYMJIT_PLANE_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else {
                    "cpp": SYMBOLICA_CPP_RUNTIME_CAPABILITY,
                    "asm": SYMBOLICA_ASM_RUNTIME_CAPABILITY,
                }.get(cell.measurement.backend)
            ),
        ),
        ExecutionMode.ON_THE_FLY: (
            ON_THE_FLY_RUNTIME_CAPABILITY,
            "pyamplicol-runtime-on-the-fly-execution",
            SYMJIT_PLANE_APPLICATION_ABI,
        ),
    }.get(
        cell.measurement.execution_mode,
        (
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            (
                COMPILED_PLANE_DIRECT_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else NATIVE_COMPILED_DIRECT_APPLICATION_ABI
            ),
            (
                SYMJIT_PLANE_APPLICATION_ABI
                if cell.measurement.backend == "jit"
                else NATIVE_COMPILED_DIRECT_APPLICATION_ABI
            ),
        ),
    )
    if arena_capability not in required_capabilities:
        raise RunnerError(
            f"report artifact process does not require {arena_capability!r}"
        )
    source_evaluator_runtime_capability = {
        "jit": SYMJIT_F64_RUNTIME_CAPABILITY,
        "cpp": SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        "asm": SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }.get(cell.measurement.backend)
    if source_evaluator_runtime_capability is None:
        raise RunnerError(
            f"report runtime has unsupported backend {cell.measurement.backend!r}"
        )
    native = importlib.import_module("pyamplicol._rusticol")
    verify_native_module(native)
    native_digest = native.native_build_inputs_sha256()
    target = native.target_info()
    build_info = _active_build_info()
    if not isinstance(build_info, Mapping):
        raise RunnerError("report runtime has no exact wheel build identity")
    publishable = build_info.get("publishable")
    if publishable is False:
        build_identity_fields = (
            "schema_version",
            "version",
            "candidate_fingerprint",
            "source_revision",
            "source_checkout",
            "native_build_inputs_sha256",
            "publishable",
        )
        candidate_build_identity = {
            field: build_info.get(field) for field in build_identity_fields
        }
        if candidate_build_identity[
            "native_build_inputs_sha256"
        ] != native_digest or not all(
            candidate_build_identity[field] is not None
            for field in build_identity_fields
        ):
            raise RunnerError(
                "report runtime is not a complete compatible non-publishable "
                "candidate build"
            )
    elif publishable is True:
        build_identity_fields = (
            "schema_version",
            "version",
            "source_revision",
            "publishable",
            "selftest_fixture_bootstrap",
        )
        candidate_build_identity = {
            field: build_info.get(field) for field in build_identity_fields
        }
        if (
            candidate_build_identity["source_revision"] != expected_source_revision
            or candidate_build_identity["selftest_fixture_bootstrap"] is not False
            or not all(
                candidate_build_identity[field] is not None
                for field in build_identity_fields
                if field != "selftest_fixture_bootstrap"
            )
        ):
            raise RunnerError(
                "report runtime is not a complete compatible release-wheel build"
            )
    else:
        raise RunnerError("report runtime build kind is unavailable")
    if (
        not isinstance(native_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", native_digest) is None
    ):
        raise RunnerError("native runtime build-input identity is invalid")
    native_path = Path(str(native.__file__)).resolve(strict=True)
    native_extension_sha256 = _sha256_file(native_path)
    package_roots = tuple(Path(str(path)) for path in pyamplicol.__path__)
    try:
        if __package__.startswith("pyamplicol."):
            package_tree = python_package_tree_identity(package_roots)
            loaded_origin_policy = loaded_pyamplicol_origin_policy(
                package_roots,
                native_extension=native_path,
            )
        else:
            preimport_identity = established_preimport_runtime_identity()
            expected_package_tree = preimport_identity.get("python_package_tree")
            expected_native_identity = preimport_identity.get("native_extension")
            if (
                preimport_identity.get("kind")
                != "pyamplicol-preimport-runtime-identity-v1"
                or not isinstance(expected_package_tree, Mapping)
                or not isinstance(expected_native_identity, Mapping)
            ):
                raise RuntimeEvidenceError(
                    "preimport runtime identity has an invalid contract"
                )
            package_tree = python_package_tree_identity(package_roots)
            if dict(package_tree) != dict(expected_package_tree):
                raise RuntimeEvidenceError(
                    "report runtime package tree differs from its preimport identity"
                )
            loaded_origin_policy = loaded_pyamplicol_origin_policy(
                package_roots,
                native_extension=native_path,
                expected_package_identity=dict(expected_package_tree),
                expected_native_identity=dict(expected_native_identity),
            )
    except RuntimeEvidenceError as error:
        raise RunnerError(
            "report runtime Python namespace cannot be authenticated"
        ) from error
    candidate_build_identity_sha256 = hashlib.sha256(
        _canonical_json(candidate_build_identity)
    ).hexdigest()
    optimization_identity: dict[str, int] = {}
    direct_codegen_identity: dict[str, object] = {}
    source_jit_identity: dict[str, object] = {}
    if cell.measurement.backend == "jit":
        source_level = cell.measurement.jit_optimization_level
        if (
            isinstance(source_level, bool)
            or not isinstance(source_level, int)
            or source_level not in {0, 1, 2, 3}
        ):
            raise RunnerError(
                "JIT Arena report runtime has no valid source optimization level"
            )
        if (
            cell.measurement.execution_mode
            in {
                ExecutionMode.EAGER,
                ExecutionMode.RECURRENCE,
                ExecutionMode.ON_THE_FLY,
            }
            and source_level != 2
        ):
            raise RunnerError(
                f"{cell.measurement.execution_mode.value} JIT Arena report "
                "runtime must use its authenticated portable O2 source"
            )
        optimization_identity["source_jit_optimization_level"] = source_level
        if cell.measurement.execution_mode is ExecutionMode.COMPILED:
            direct_codegen_identity = _authenticated_direct_codegen_identity(
                manifest,
                process_id=process_id,
                source_optimization_level=source_level,
            )
            optimization_identity["direct_codegen_optimization_level"] = int(
                direct_codegen_identity["optimization_level"]
            )
        elif cell.measurement.execution_mode is ExecutionMode.RECURRENCE:
            source_jit_identity = _authenticated_recurrence_source_identity(
                manifest,
                process_id=process_id,
                source_optimization_level=source_level,
            )
    elif cell.measurement.jit_optimization_level is not None:
        raise RunnerError(
            "non-JIT Arena report runtime unexpectedly declares a JIT "
            "optimization level"
        )
    return {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "artifact_id": manifest.artifact_id,
        "loaded_artifact_id": loaded_artifact_id,
        "artifact_identity_match": True,
        "process_id": process_id,
        "execution_mode": expected_execution_mode,
        "loaded_execution_mode": loaded_execution_mode,
        "backend": cell.measurement.backend,
        "required_arena_capability": arena_capability,
        "expected_evaluator_abi": evaluator_abi,
        "expected_source_evaluator_abi": source_evaluator_abi,
        "expected_source_evaluator_runtime_capability": (
            source_evaluator_runtime_capability
        ),
        "process_required_runtime_capabilities": list(required_capabilities),
        "package_version": pyamplicol.__version__,
        "python_package_tree": package_tree,
        "loaded_module_origin_policy": loaded_origin_policy,
        "artifact_runtime_version": manifest.runtime["engine_version"],
        "source_revision": expected_source_revision,
        "candidate_build_identity": candidate_build_identity,
        "candidate_build_identity_sha256": candidate_build_identity_sha256,
        "native_build_inputs_sha256": native_digest,
        "native_extension": {
            "path": str(native_path),
            "sha256": native_extension_sha256,
            "package_version": str(native.package_version()),
        },
        "native_target": {
            "triple": str(target.triple),
            "cpu_features": [str(value) for value in target.cpu_features],
        },
        **optimization_identity,
        **(
            {"direct_codegen_identity": direct_codegen_identity}
            if direct_codegen_identity
            else {}
        ),
        **({"source_jit_identity": source_jit_identity} if source_jit_identity else {}),
    }


def _authenticated_process_execution(
    manifest: object,
    *,
    process_id: str,
) -> tuple[Mapping[str, object], str, str, Path, Sequence[object]]:
    runtime = getattr(manifest, "runtime", None)
    root = getattr(manifest, "root", None)
    payloads = getattr(manifest, "payloads", None)
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(root, Path)
        or not isinstance(payloads, Sequence)
    ):
        raise RunnerError("artifact exposes no authenticated execution payload")
    relative = runtime.get("evaluator_manifest_path")
    if not isinstance(relative, str) or not relative:
        raise RunnerError("artifact runtime has no evaluator manifest path")
    evaluator_index, _ = _authenticated_json_payload(
        root,
        payloads,
        relative,
        expected_process_id=None,
        label="evaluator index",
    )
    if (
        evaluator_index.get("kind") != "pyamplicol-runtime-execution-set"
        or type(evaluator_index.get("schema_version")) is not int
        or evaluator_index.get("schema_version") != 3
    ):
        raise RunnerError("authenticated evaluator index has an invalid contract")
    entries = evaluator_index.get("processes")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise RunnerError("authenticated evaluator index has no process entries")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("process_id") == process_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"authenticated evaluator index has no unique entry for {process_id!r}"
        )
    entry_relative = matches[0].get("manifest_path")
    if not isinstance(entry_relative, str) or not entry_relative:
        raise RunnerError("authenticated evaluator index entry has no manifest path")
    entry_logical = PurePosixPath(entry_relative)
    if (
        entry_logical.is_absolute()
        or not entry_logical.parts
        or any(part in {"", ".", ".."} for part in entry_logical.parts)
        or entry_logical.as_posix() != entry_relative
    ):
        raise RunnerError(
            "authenticated evaluator index entry has a noncanonical manifest path"
        )
    execution_relative = (PurePosixPath(relative).parent / entry_logical).as_posix()
    execution, actual_sha256 = _authenticated_json_payload(
        root,
        payloads,
        execution_relative,
        expected_process_id=process_id,
        label="process execution manifest",
    )
    return execution, execution_relative, actual_sha256, root, payloads


def _authenticated_symjit_plane_leaves(
    value: object,
    *,
    source_optimization_level: int,
    context: str,
) -> list[tuple[str, str]]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{context} source evaluator is not an object")
    if value.get("kind") == "chunked-symbolica-evaluator":
        chunks = value.get("chunks")
        if (
            isinstance(chunks, (str, bytes))
            or not isinstance(chunks, Sequence)
            or not chunks
        ):
            raise RunnerError(f"{context} chunked source evaluator has no leaves")
        leaves: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            leaves.extend(
                _authenticated_symjit_plane_leaves(
                    chunk,
                    source_optimization_level=source_optimization_level,
                    context=f"{context}.chunks[{index}]",
                )
            )
        return leaves
    legacy_backend = value.get("backend")
    input_len = value.get("input_len")
    output_len = value.get("output_len")
    if (
        value.get("kind") != "symjit-application-evaluator"
        or ("backend" in value and legacy_backend != "jit")
        or value.get("runtime_capability") != _RECURRENCE_JIT_SOURCE_RUNTIME_CAPABILITY
        or value.get("application_abi") != _SYMJIT_APPLICATION_ABI
        or value.get("element_layout") != "complex-f64"
        or value.get("batch_layout") != "row-major"
        or value.get("compiler_type") != "native"
        or value.get("translation_mode") not in {"direct", "indirect"}
        or value.get("optimization_level") != source_optimization_level
        or value.get("word_bits") != 64
        or value.get("endianness") != "little"
        or value.get("required_defuns") != []
        or isinstance(input_len, bool)
        or not isinstance(input_len, int)
        or input_len < 0
        or isinstance(output_len, bool)
        or not isinstance(output_len, int)
        or output_len < 0
    ):
        raise RunnerError(f"{context} source evaluator leaf contract drifted")
    _canonical_recurrence_path(
        value.get("application_path"),
        f"{context} ordinary source evaluator application",
    )
    fallback_path = value.get("evaluator_state_path")
    fallback_capability = value.get("evaluator_state_runtime_capability")
    if fallback_path is None and fallback_capability is None:
        pass
    elif (
        not isinstance(fallback_path, str)
        or fallback_capability != "symbolica.legacy-jit-container.complex-f64.v1"
    ):
        raise RunnerError(f"{context} source evaluator fallback contract drifted")
    else:
        _canonical_recurrence_path(
            fallback_path,
            f"{context} fallback source evaluator state",
        )
    plane = value.get("plane_application")
    if not isinstance(plane, Mapping):
        raise RunnerError(f"{context} source evaluator has no plane application")
    target = plane.get("target")
    if (
        plane.get("application_abi") != _SYMJIT_PLANE_APPLICATION_ABI
        or plane.get("storage_abi") != _SYMJIT_APPLICATION_ABI
        or plane.get("element_layout") != "split-complex-plane-major"
        or plane.get("descriptor_order") != "inputs-re-im-then-outputs-re-im"
        or plane.get("input_complex_count") != input_len
        or plane.get("output_complex_count") != output_len
        or plane.get("input_plane_count") != 2 * input_len
        or plane.get("output_plane_count") != 2 * output_len
        or plane.get("compiler_type") != "native"
        or plane.get("translation_mode") != "symbolica-structured-instructions"
        or plane.get("simd") is not True
        or plane.get("complex") is not True
        or plane.get("fast_math") is not True
        or plane.get("fast_complex") is not False
        or not isinstance(plane.get("compression"), bool)
        or plane.get("threading") is not False
        or plane.get("direct_arena") is not True
        or plane.get("optimization_level") != source_optimization_level
        or not _valid_symjit_plane_target(target)
    ):
        raise RunnerError(f"{context} plane application contract drifted")
    plane_path = _canonical_recurrence_path(
        plane.get("application_path"),
        f"{context} plane application",
    )
    source_digest = _lowercase_sha256(
        plane.get("source_digest"),
        f"{context} plane source digest",
    )
    return [(plane_path, source_digest)]


def _valid_symjit_plane_target(value: object) -> bool:
    """Match Rusticol's canonical SymJIT plane-target object contract."""

    if not isinstance(value, Mapping):
        return False
    allowed_fields = {"word_bits", "endianness", "triple", "cpu_features"}
    if set(value) - allowed_fields:
        return False
    word_bits = value.get("word_bits")
    if isinstance(word_bits, bool) or not isinstance(word_bits, int) or word_bits != 64:
        return False
    if value.get("endianness") != "little":
        return False
    if "triple" in value:
        triple = value.get("triple")
        if not isinstance(triple, str) or not triple or "\0" in triple:
            return False
    if "cpu_features" in value:
        features = value.get("cpu_features")
        if not isinstance(features, list):
            return False
        previous: str | None = None
        for feature in features:
            if (
                not isinstance(feature, str)
                or not feature
                or "\0" in feature
                or (previous is not None and previous >= feature)
            ):
                return False
            previous = feature
    return True


def _authenticated_direct_codegen_identity(
    manifest: object,
    *,
    process_id: str,
    source_optimization_level: int,
) -> dict[str, object]:
    """Authenticate P-kernel lowering in one artifact-ID-bound process payload."""

    execution, execution_relative, actual_sha256, _, _ = (
        _authenticated_process_execution(manifest, process_id=process_id)
    )

    leaf_count = 0

    def walk(value: object) -> None:
        nonlocal leaf_count
        if isinstance(value, Mapping):
            arena = value.get("compiled_plane_arena")
            if isinstance(arena, Mapping):
                source_leaves = _authenticated_symjit_plane_leaves(
                    value.get("evaluator"),
                    source_optimization_level=source_optimization_level,
                    context="compiled plane-Arena evaluator",
                )
                leaves = arena.get("leaves")
                if (
                    isinstance(leaves, (str, bytes))
                    or not isinstance(leaves, Sequence)
                    or not leaves
                ):
                    raise RunnerError(
                        "compiled plane-Arena descriptor has no authenticated leaves"
                    )
                if len(leaves) != len(source_leaves):
                    raise RunnerError(
                        "compiled plane-Arena descriptor does not cover its "
                        "authenticated plane applications"
                    )
                for raw_leaf, (plane_path, _source_digest) in zip(
                    leaves,
                    source_leaves,
                    strict=True,
                ):
                    if not isinstance(raw_leaf, Mapping):
                        raise RunnerError(
                            "compiled plane-Arena descriptor has an invalid leaf"
                        )
                    if (
                        raw_leaf.get("application_path") != plane_path
                        or raw_leaf.get("source_application_abi")
                        != _SYMJIT_PLANE_APPLICATION_ABI
                        or raw_leaf.get("optimization_level")
                        != source_optimization_level
                        or raw_leaf.get("direct_codegen_optimization_level")
                        != source_optimization_level
                    ):
                        raise RunnerError(
                            "compiled plane-Arena leaf optimization identity drifted"
                        )
                    leaf_count += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                walk(child)

    walk(execution)
    if leaf_count == 0:
        raise RunnerError("execution payload contains no compiled plane-Arena leaves")
    return {
        "kind": "authenticated-compiled-plane-arena-direct-codegen-v1",
        "optimization_level": source_optimization_level,
        "source_optimization_level": source_optimization_level,
        "leaf_count": leaf_count,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": actual_sha256,
    }


def _authenticated_recurrence_source_identity(
    manifest: object,
    *,
    process_id: str,
    source_optimization_level: int,
) -> dict[str, object]:
    """Bind recurrence JIT O2 to its exact source and direct-template payloads."""

    execution, execution_relative, execution_sha256, root, payloads = (
        _authenticated_process_execution(manifest, process_id=process_id)
    )
    if execution.get("kind") != "pyamplicol-runtime-recurrence-execution":
        raise RunnerError("authenticated recurrence execution kind is invalid")
    plan = execution.get("plan")
    if not isinstance(plan, Mapping):
        raise RunnerError("authenticated recurrence execution has no plan")
    expected_execution_abis = {
        "direct_template_abi": _RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct_backend_abi": _RECURRENCE_DIRECT_BACKEND_ABI,
    }
    for field, expected in expected_execution_abis.items():
        if execution.get(field) != expected or plan.get(field) != expected:
            raise RunnerError(
                f"authenticated recurrence execution/plan {field} drifted"
            )
    prepared_kernel_pack_digest = _lowercase_sha256(
        execution.get("prepared_kernel_pack_digest"),
        "authenticated recurrence execution prepared-kernel pack digest",
    )
    direct_template_catalog_digest = _lowercase_sha256(
        execution.get("direct_template_catalog_digest"),
        "authenticated recurrence execution direct-template catalog digest",
    )
    if (
        plan.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or plan.get("direct_template_catalog_digest") != direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence execution/plan source digests drifted"
        )
    kernel_pack = execution.get("kernel_pack")
    if not isinstance(kernel_pack, Mapping):
        raise RunnerError("authenticated recurrence execution has no kernel pack")
    pack_relative = kernel_pack.get("manifest_path")
    if not isinstance(pack_relative, str) or not pack_relative:
        raise RunnerError("authenticated recurrence execution has no kernel-pack path")
    payload_root = _canonical_recurrence_path(
        kernel_pack.get("payload_root"),
        "authenticated recurrence kernel payload root",
    )
    pack, pack_sha256 = _authenticated_json_payload(
        root,
        payloads,
        pack_relative,
        expected_process_id=None,
        label="recurrence kernel pack",
    )
    settings = pack.get("optimization_settings")
    direct_catalog = pack.get("recurrence_direct_template")
    if not isinstance(settings, Mapping) or not isinstance(direct_catalog, Mapping):
        raise RunnerError("authenticated recurrence kernel pack is incomplete")
    templates = direct_catalog.get("templates")
    if (
        pack.get("backend") != "jit"
        or settings.get("backend") != "jit"
        or settings.get("jit_optimization_level") != source_optimization_level
        or direct_catalog.get("abi") != _RECURRENCE_DIRECT_TEMPLATE_ABI
        or direct_catalog.get("backend") != "jit"
        or direct_catalog.get("backend_abi") != _RECURRENCE_DIRECT_BACKEND_ABI
        or direct_catalog.get("canonicalization_abi")
        != _RECURRENCE_DIRECT_CANONICALIZATION_ABI
        or direct_catalog.get("optimization_level") != source_optimization_level
        or direct_catalog.get("portable") is not True
        or isinstance(templates, (str, bytes))
        or not isinstance(templates, Sequence)
        or not templates
    ):
        raise RunnerError(
            "authenticated recurrence direct-template optimization identity drifted"
        )
    if (
        direct_catalog.get("prepared_kernel_pack_digest") != prepared_kernel_pack_digest
        or direct_catalog.get("catalog_digest") != direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence execution and kernel-pack digests differ"
        )
    if _mapping_digest_without(direct_catalog, "catalog_digest") != (
        direct_template_catalog_digest
    ):
        raise RunnerError(
            "authenticated recurrence direct-template catalog digest is invalid"
        )

    evaluator_members = _authenticated_evaluator_container_members(
        manifest,
        root=root,
        payloads=payloads,
    )
    authenticated_source_payloads: dict[str, str] = {}

    def authenticate_source_payload(relative: str, *, label: str) -> str:
        cached = authenticated_source_payloads.get(relative)
        if cached is not None:
            return cached
        result = _authenticated_recurrence_source_payload(
            root,
            payloads,
            evaluator_members,
            relative,
            label=label,
        )
        authenticated_source_payloads[relative] = result
        return result

    kernels = pack.get("kernels")
    variants = pack.get("kernel_variants")
    if (
        isinstance(kernels, (str, bytes))
        or not isinstance(kernels, Sequence)
        or not kernels
        or isinstance(variants, (str, bytes))
        or not isinstance(variants, Sequence)
    ):
        raise RunnerError(
            "authenticated recurrence kernel pack has an invalid source inventory"
        )
    source_leaves_by_kernel: dict[int, set[tuple[str, str]]] = {}
    source_evaluator_leaf_count = 0
    for index, raw_kernel in enumerate(kernels):
        if not isinstance(raw_kernel, Mapping):
            raise RunnerError(
                f"authenticated recurrence kernel {index} is not an object"
            )
        kernel_id = raw_kernel.get("kernel_id")
        if (
            isinstance(kernel_id, bool)
            or not isinstance(kernel_id, int)
            or kernel_id < 0
            or kernel_id in source_leaves_by_kernel
        ):
            raise RunnerError(
                f"authenticated recurrence kernel {index} has an invalid kernel ID"
            )
        leaves = _authenticated_recurrence_source_leaves(
            raw_kernel.get("f64_evaluator_manifest"),
            source_optimization_level=source_optimization_level,
            payload_root=payload_root,
            authenticate_payload=authenticate_source_payload,
            context=f"authenticated recurrence kernel {index}",
        )
        source_evaluator_leaf_count += len(leaves)
        source_leaves_by_kernel[kernel_id] = set(leaves)
    for index, raw_variant in enumerate(variants):
        if not isinstance(raw_variant, Mapping):
            raise RunnerError(
                f"authenticated recurrence kernel variant {index} is not an object"
            )
        source_evaluator_leaf_count += len(
            _authenticated_recurrence_source_leaves(
                raw_variant.get("f64_evaluator_manifest"),
                source_optimization_level=source_optimization_level,
                payload_root=payload_root,
                authenticate_payload=authenticate_source_payload,
                context=f"authenticated recurrence kernel variant {index}",
            )
        )

    prepared_direct_template_count = 0
    for index, raw_template in enumerate(templates):
        if (
            not isinstance(raw_template, Mapping)
            or raw_template.get("abi") != _RECURRENCE_DIRECT_TEMPLATE_ABI
            or raw_template.get("backend") != "jit"
            or raw_template.get("portable") is not True
            or raw_template.get("optimization_level") != source_optimization_level
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} contract drifted"
            )
        if _mapping_digest_without(raw_template, "semantic_digest") != (
            raw_template.get("semantic_digest")
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} digest is invalid"
            )
        binding = raw_template.get("payload_binding")
        if not isinstance(binding, Mapping):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} has no "
                "payload binding"
            )
        if (
            binding.get("abi") != _RECURRENCE_DIRECT_PAYLOAD_BINDING_ABI
            or _recurrence_payload_binding_digest(binding)
            != binding.get("payload_digest")
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} "
                "payload-binding contract is invalid"
            )
        binding_kind = binding.get("kind")
        if binding_kind == "rusticol-intrinsic":
            if (
                binding.get("source_application_abi") is not None
                or binding.get("direct_application_abi") is not None
                or binding.get("source_application_path") is not None
                or binding.get("source_application_sha256") is not None
                or binding.get("payload_paths") != []
                or not isinstance(binding.get("runtime_template"), str)
                or not binding.get("runtime_template")
            ):
                raise RunnerError(
                    f"authenticated recurrence direct-template {index} "
                    "intrinsic source contract is invalid"
                )
            continue
        if binding_kind != "prepared-direct-call":
            raise RunnerError(
                f"authenticated recurrence direct-template {index} is not executable"
            )
        prepared_direct_template_count += 1
        source_path = _canonical_recurrence_path(
            binding.get("source_application_path"),
            (f"authenticated recurrence direct-template {index} source application"),
        )
        source_sha256 = _lowercase_sha256(
            binding.get("source_application_sha256"),
            (
                f"authenticated recurrence direct-template {index} "
                "source application digest"
            ),
        )
        prepared_kernel_id = binding.get("prepared_kernel_id")
        if (
            binding.get("source_application_abi")
            != _RECURRENCE_JIT_SOURCE_APPLICATION_ABI
            or binding.get("direct_application_abi")
            != _RECURRENCE_JIT_DIRECT_APPLICATION_ABI
            or binding.get("payload_paths") != [source_path]
            or isinstance(prepared_kernel_id, bool)
            or not isinstance(prepared_kernel_id, int)
            or prepared_kernel_id < 0
            or (source_path, source_sha256)
            not in source_leaves_by_kernel.get(prepared_kernel_id, set())
        ):
            raise RunnerError(
                f"authenticated recurrence direct-template {index} source "
                "application is not bound to its prepared kernel"
            )
    if prepared_direct_template_count == 0 or source_evaluator_leaf_count == 0:
        raise RunnerError(
            "authenticated recurrence direct-template pack has no source "
            "application evidence"
        )
    return {
        "kind": "authenticated-recurrence-direct-template-source-v1",
        "optimization_level": source_optimization_level,
        "direct_template_count": len(templates),
        "prepared_direct_template_count": prepared_direct_template_count,
        "source_evaluator_leaf_count": source_evaluator_leaf_count,
        "source_application_abi": _RECURRENCE_JIT_SOURCE_APPLICATION_ABI,
        "direct_application_abi": _RECURRENCE_JIT_DIRECT_APPLICATION_ABI,
        "prepared_kernel_pack_digest": prepared_kernel_pack_digest,
        "direct_template_catalog_digest": direct_template_catalog_digest,
        "execution_manifest_path": execution_relative,
        "execution_manifest_sha256": execution_sha256,
        "kernel_pack_path": pack_relative,
        "kernel_pack_sha256": pack_sha256,
    }


def _lowercase_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunnerError(f"{label} is not a lowercase SHA-256")
    return value


def _mapping_digest_without(value: Mapping[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _recurrence_payload_binding_digest(binding: Mapping[str, object]) -> str:
    if binding.get("kind") != "rusticol-intrinsic":
        return _mapping_digest_without(binding, "payload_digest")
    if binding.get("role") == "contribution":
        fields = (
            "abi",
            "destination_operation",
            "contribution_parent_permutation",
            "intrinsic_contract_digest",
            "kind",
            "role",
            "runtime_template",
            "scalar_input_count",
            "scalar_projections",
        )
        return hashlib.sha256(
            _canonical_json({field: binding.get(field) for field in fields})
        ).hexdigest()
    # Source/finalization intrinsics have an opaque Rusticol implementation
    # digest whose derivation includes semantic-catalog fields not duplicated
    # in the payload binding.  Its exact value remains transitively bound by
    # the recomputed template and catalog digests.
    return _lowercase_sha256(
        binding.get("payload_digest"),
        "authenticated recurrence intrinsic payload digest",
    )


def _canonical_recurrence_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{label} path is missing")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != value
    ):
        raise RunnerError(f"{label} path is not canonical")
    return value


def _authenticated_evaluator_container_members(
    manifest: object,
    *,
    root: Path,
    payloads: Sequence[object],
) -> Mapping[str, tuple[int, str]]:
    extensions = getattr(manifest, "extensions", None)
    if extensions is None:
        return {}
    if not isinstance(extensions, Mapping):
        raise RunnerError("artifact evaluator payload extensions are invalid")
    raw_container = extensions.get("evaluator_payload_container")
    if raw_container is None:
        return {}
    if not isinstance(raw_container, Mapping):
        raise RunnerError("artifact evaluator payload container is invalid")
    required = {
        "kind",
        "schema_version",
        "storage_abi",
        "path",
        "member_count",
        "unpacked_size_bytes",
        "index_sha256",
    }
    if (
        set(raw_container) != required
        or raw_container.get("kind") != "pyamplicol-evaluator-payload-container"
        or raw_container.get("schema_version") != 1
        or raw_container.get("storage_abi") != "pacbin-v1"
    ):
        raise RunnerError("artifact evaluator payload container contract drifted")
    member_count = raw_container.get("member_count")
    unpacked_size = raw_container.get("unpacked_size_bytes")
    index_sha256 = _lowercase_sha256(
        raw_container.get("index_sha256"),
        "artifact evaluator payload container index digest",
    )
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 0
        or isinstance(unpacked_size, bool)
        or not isinstance(unpacked_size, int)
        or unpacked_size < 0
    ):
        raise RunnerError("artifact evaluator payload container inventory is invalid")
    container_path = _canonical_recurrence_path(
        raw_container.get("path"),
        "artifact evaluator payload container",
    )
    data, _ = _authenticated_payload_bytes(
        root,
        payloads,
        container_path,
        expected_process_id=None,
        expected_role="evaluator-state",
        expected_media_type="application/octet-stream",
        label="evaluator payload container",
    )
    try:
        from pyamplicol.generation.evaluator_container import PacbinReader

        with PacbinReader.open(io.BytesIO(data), verify_payloads=True) as reader:
            index = reader.index
            members = tuple(reader.members)
    except (OSError, TypeError, ValueError) as error:
        raise RunnerError(
            "authenticated evaluator payload container is invalid"
        ) from error
    if (
        len(members) != member_count
        or sum(member.length for member in members) != unpacked_size
        or index.index_sha256 != index_sha256
    ):
        raise RunnerError(
            "authenticated evaluator payload container index metadata drifted"
        )
    return {
        member.logical_path: (int(member.kind), member.sha256) for member in members
    }


def _authenticated_recurrence_source_payload(
    root: Path,
    payloads: Sequence[object],
    evaluator_members: Mapping[str, tuple[int, str]],
    relative: str,
    *,
    label: str,
) -> str:
    canonical = _canonical_recurrence_path(relative, label)
    loose = [
        payload for payload in payloads if getattr(payload, "path", None) == canonical
    ]
    packed = evaluator_members.get(canonical)
    if len(loose) + (packed is not None) != 1:
        raise RunnerError(f"{label} payload is missing or ambiguous")
    if loose:
        _, sha256 = _authenticated_payload_bytes(
            root,
            payloads,
            canonical,
            expected_process_id=None,
            expected_role="evaluator-state",
            expected_media_type="application/octet-stream",
            label=label,
        )
        return sha256
    assert packed is not None
    kind, sha256 = packed
    if kind != 1:
        raise RunnerError(f"{label} is not a packed SymJIT application")
    return _lowercase_sha256(sha256, f"{label} packed member digest")


def _authenticated_recurrence_source_leaves(
    value: object,
    *,
    source_optimization_level: int,
    payload_root: str,
    authenticate_payload: object,
    context: str,
) -> list[tuple[str, str]]:
    plane_leaves = _authenticated_symjit_plane_leaves(
        value,
        source_optimization_level=source_optimization_level,
        context=context,
    )
    if not callable(authenticate_payload):
        raise RunnerError("recurrence source payload authenticator is invalid")
    result: list[tuple[str, str]] = []
    for application_path, _source_digest in plane_leaves:
        payload_path = (
            PurePosixPath(payload_root) / PurePosixPath(application_path)
        ).as_posix()
        sha256 = authenticate_payload(  # type: ignore[operator]
            payload_path,
            label=f"{context} plane application",
        )
        if not isinstance(sha256, str):
            raise RunnerError(f"{context} plane application digest is invalid")
        result.append((application_path, sha256))
    return result


def _authenticated_json_payload(
    root: Path,
    payloads: Sequence[object],
    relative: str,
    *,
    expected_process_id: str | None,
    label: str,
) -> tuple[Mapping[str, object], str]:
    """Read one declared JSON payload through one checked file descriptor."""

    data, actual_sha256 = _authenticated_payload_bytes(
        root,
        payloads,
        relative,
        expected_process_id=expected_process_id,
        expected_role="evaluator-manifest",
        expected_media_type="application/json",
        label=label,
    )
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise RunnerError(f"authenticated {label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise RunnerError(f"authenticated {label} is not a JSON object")
    return value, actual_sha256


def _authenticated_payload_bytes(
    root: Path,
    payloads: Sequence[object],
    relative: str,
    *,
    expected_process_id: str | None,
    expected_role: str,
    expected_media_type: str,
    label: str,
) -> tuple[bytes, str]:
    """Read exact manifest-declared bytes through one no-follow descriptor."""

    logical = PurePosixPath(relative)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != relative
    ):
        raise RunnerError(f"{label} path is not canonical")
    matches = [
        payload for payload in payloads if getattr(payload, "path", None) == relative
    ]
    if len(matches) != 1:
        raise RunnerError(f"{label} payload is missing or ambiguous")
    payload = matches[0]
    expected_size = getattr(payload, "size_bytes", None)
    expected_sha256 = getattr(payload, "sha256", None)
    if (
        getattr(payload, "role", None) != expected_role
        or getattr(payload, "media_type", None) != expected_media_type
        or getattr(payload, "executable", None) is not False
        or getattr(payload, "process_id", None) != expected_process_id
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise RunnerError(f"{label} has an invalid artifact declaration")
    path = root.joinpath(*logical.parts)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise RunnerError(f"{label} does not match its artifact declaration")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            data = stream.read(expected_size + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise RunnerError(f"cannot read authenticated {label}") from error
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
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise RunnerError(f"authenticated {label} changed while it was read")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_sha256 != expected_sha256:
        raise RunnerError(f"{label} does not match the artifact manifest")
    return data, actual_sha256


def _authenticated_effective_config(artifact_path: Path) -> Mapping[str, object]:
    """Return the exact effective TOML declared by a generated artifact."""

    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(artifact_path, verify_payloads=False)
    root = getattr(manifest, "root", None)
    payloads = getattr(manifest, "payloads", None)
    configuration = getattr(manifest, "configuration", None)
    if (
        not isinstance(root, Path)
        or not isinstance(payloads, Sequence)
        or not isinstance(configuration, Mapping)
    ):
        raise RunnerError("generated artifact exposes no effective configuration")
    relative = configuration.get("effective_path")
    if not isinstance(relative, str) or not relative:
        raise RunnerError("generated artifact has no effective configuration path")
    data, _ = _authenticated_payload_bytes(
        root,
        payloads,
        relative,
        expected_process_id=None,
        expected_role="configuration-effective",
        expected_media_type="application/toml",
        label="effective configuration",
    )
    try:
        value = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RunnerError(
            "authenticated effective configuration is invalid TOML"
        ) from error
    if not isinstance(value, Mapping):
        raise RunnerError("authenticated effective configuration is not a TOML table")
    return value


def _artifact_numerical_relation_metadata(
    artifact_path: Path,
    process_id: str,
) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
    """Read compact correctness and fallback state from one manifest load."""

    from pyamplicol.artifacts import load_manifest
    from pyamplicol.generation.recurrence_numerical_current_warmup import (
        recurrence_numerical_relation_correctness_summary,
        recurrence_numerical_relation_fallback_summary,
    )

    manifest = load_manifest(artifact_path, verify_payloads=False)
    generation = manifest.extensions.get("generation")
    concrete = (
        generation.get("concrete_processes")
        if isinstance(generation, Mapping)
        else None
    )
    if not isinstance(concrete, Sequence) or isinstance(concrete, (str, bytes)):
        raise RunnerError("generated artifact has no concrete-process metadata")
    matches = tuple(
        entry
        for entry in concrete
        if isinstance(entry, Mapping) and entry.get("id") == process_id
    )
    if len(matches) != 1:
        raise RunnerError(
            "generated artifact numerical relation process is missing or ambiguous"
        )
    filters = matches[0].get("filters")
    relation_report = (
        filters.get("relation_discovery") if isinstance(filters, Mapping) else None
    )
    if not isinstance(relation_report, Mapping):
        raise RunnerError("generated artifact has no numerical relation report")
    try:
        return (
            recurrence_numerical_relation_correctness_summary(relation_report),
            recurrence_numerical_relation_fallback_summary(relation_report),
        )
    except ValueError as error:
        raise RunnerError(str(error)) from error


def _selector_kwargs(
    cell: CellSpec,
    contract: SelectorContract | None,
) -> dict[str, tuple[str, ...] | None]:
    if cell.measurement.accuracy is not Accuracy.LC:
        return {"helicities": None, "color_flows": None}
    if contract is None:
        raise RunnerError("LC measurement requires a selector contract")
    if cell.workload is Workload.SELECTED_FLOW:
        return {
            "helicities": None,
            "color_flows": contract.selected_color_flow_ids,
        }
    if cell.workload is Workload.ALL_FLOW:
        return {
            "helicities": contract.runtime_all_flow_helicity_ids,
            "color_flows": None,
        }
    raise RunnerError("LC measurement has an invalid workload")


def derive_selector_contract(
    runtime: RuntimeLike,
    points: object,
) -> SelectorContract:
    """Select one backend-independent labelled LC flow and helicity."""

    if _runtime_execution_mode(runtime) == ExecutionMode.ON_THE_FLY.value:
        raise RunnerError(
            "on-the-fly selector contracts must be inherited from a dense authority"
        )
    physics = runtime.physics
    color_flows = tuple(getattr(physics, "color_flows", ()))
    helicities = tuple(getattr(physics, "helicities", ()))
    particles = tuple(getattr(physics, "external_particles", ()))
    if not color_flows or not helicities or not particles:
        raise RunnerError("LC selector derivation requires complete physical axes")

    try:
        words = tuple(canonical_lc_flow_word(flow.word) for flow in color_flows)
        selected_word = canonical_lc_selector_word(words)
    except (AttributeError, SelectorPolicyError) as error:
        raise RunnerError(str(error)) from error
    selected_matches = tuple(
        flow
        for flow, word in zip(color_flows, words, strict=True)
        if word == selected_word
    )
    if len(selected_matches) != 1:
        raise RunnerError("LC selector did not identify exactly one physical flow")
    selected_flow = selected_matches[0]
    expected_flow_id = selector_color_flow_id(selected_word)
    if str(selected_flow.id) != expected_flow_id:
        raise RunnerError("LC selector flow ID does not encode its physical word")

    try:
        pdgs = tuple(int(particle.pdg_id) for particle in particles)
        states = fixed_selector_helicity(pdgs)
    except (AttributeError, SelectorPolicyError, TypeError, ValueError) as error:
        raise RunnerError(
            "LC selector derivation requires external particle PDGs"
        ) from error
    helicity_matches = tuple(
        helicity
        for helicity in helicities
        if tuple(int(value) for value in helicity.values) == states
    )
    if len(helicity_matches) != 1:
        raise RunnerError(
            "artifact does not expose exactly one deterministic selector helicity"
        )
    helicity = helicity_matches[0]
    expected_helicity_id = selector_helicity_id(states)
    if str(helicity.id) != expected_helicity_id:
        raise RunnerError("LC selector helicity ID does not encode its physical states")

    resolved = runtime.evaluate_resolved(
        points,
        color_flows=(expected_flow_id,),
        helicities=(expected_helicity_id,),
    )
    components = tuple(
        complex(component)
        for point in getattr(resolved, "values", ())
        for helicity_row in point
        for component in helicity_row
    )
    if not components:
        raise RunnerError("fixed-helicity selector evaluation is empty")
    if any(
        not math.isfinite(component.real) or not math.isfinite(component.imag)
        for component in components
    ):
        raise RunnerError("fixed-helicity selector contains a non-finite component")
    if not any(abs(component) > 0.0 for component in components):
        raise RunnerError("canonical fixed-helicity selector is zero at report point")

    labels = tuple(int(particle.label) for particle in particles)
    if len(labels) != len(states):
        raise RunnerError(
            "helicity source-state axis does not match external particles"
        )
    return SelectorContract(
        selected_color_flow_ids=(expected_flow_id,),
        selected_color_words=(selected_word,),
        all_flow_helicity_ids=(expected_helicity_id,),
        all_flow_source_helicities=tuple(zip(labels, states, strict=True)),
        point_digest=point_digest(points),
    )


def validate_selector_contract(
    runtime: RuntimeLike,
    contract: SelectorContract,
    points: object,
    *,
    cell: CellSpec | None = None,
) -> None:
    if point_digest(points) != contract.point_digest:
        raise RunnerError("selector contract and measurement point differ")
    runtime_is_on_the_fly = (
        _runtime_execution_mode(runtime) == ExecutionMode.ON_THE_FLY.value
    )
    cell_is_on_the_fly = (
        cell is not None and cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    )
    if runtime_is_on_the_fly:
        if cell is None:
            raise RunnerError("on-the-fly selector validation requires its report cell")
        if not cell_is_on_the_fly:
            raise RunnerError("on-the-fly report runtime has the wrong execution mode")
        helicity_count, color_count = _on_the_fly_compact_context(
            cell,
            runtime,
        )
        for identifier, word in zip(
            contract.selected_color_flow_ids,
            contract.selected_color_words,
            strict=True,
        ):
            try:
                canonical_word = canonical_lc_flow_word(word)
            except SelectorPolicyError as error:
                raise RunnerError(str(error)) from error
            if canonical_word != word or selector_color_flow_id(word) != identifier:
                raise RunnerError(
                    f"artifact does not expose selected physical flow {identifier!r}"
                )
        expected_states = tuple(
            state for _label, state in sorted(contract.all_flow_source_helicities)
        )
        runtime_helicity_ids = contract.runtime_all_flow_helicity_ids
        if runtime_helicity_ids != (selector_helicity_id(expected_states),):
            raise RunnerError(
                "artifact does not expose selected physical helicity "
                f"{contract.all_flow_helicity_ids[0]!r}"
            )
        _on_the_fly_selector_ordinals(
            runtime,
            contract.selected_color_flow_ids,
            name="color_flow_by_point",
            count=color_count,
        )
        _on_the_fly_selector_ordinals(
            runtime,
            runtime_helicity_ids,
            name="helicity_by_point",
            count=helicity_count,
        )
        return
    if cell_is_on_the_fly:
        raise RunnerError("on-the-fly report runtime has the wrong execution mode")
    physics = runtime.physics
    colors = {
        str(flow.id): tuple(int(label) for label in flow.word)
        for flow in getattr(physics, "color_flows", ())
    }
    for identifier, word in zip(
        contract.selected_color_flow_ids,
        contract.selected_color_words,
        strict=True,
    ):
        if colors.get(identifier) != word:
            raise RunnerError(
                f"artifact does not expose selected physical flow {identifier!r}"
            )
    helicities = {
        str(item.id): tuple(int(value) for value in item.values)
        for item in getattr(physics, "helicities", ())
    }
    labels = tuple(
        int(particle.label) for particle in getattr(physics, "external_particles", ())
    )
    expected_states = dict(contract.all_flow_source_helicities)
    expected = tuple(expected_states[label] for label in labels)
    for identifier, runtime_identifier in zip(
        contract.all_flow_helicity_ids,
        contract.runtime_all_flow_helicity_ids,
        strict=True,
    ):
        if helicities.get(runtime_identifier) != expected:
            raise RunnerError(
                f"artifact does not expose selected physical helicity {identifier!r}"
            )


@dataclass(frozen=True, slots=True)
class _OtfAuthorityCapture:
    totals: tuple[float, ...]
    helicity_ids: tuple[str, ...]
    color_flow_ids: tuple[str, ...]
    components: tuple[tuple[tuple[float, ...], ...], ...]
    total_source_sha256: str
    resolved_total_source_sha256: str
    resolved_ordering_sha256: str
    resolved_source_sha256: str
    resolved_sum: dict[str, object]


def _finite_real(value: object, label: str) -> float:
    try:
        number = complex(value)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError) as error:
        raise RunnerError(f"{label} is not numeric") from error
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise RunnerError(f"{label} is not finite")
    if number.imag != 0.0:
        raise RunnerError(f"{label} is not real")
    return float(number.real)


def _otf_selector_digest(cell: CellSpec, contract: SelectorContract) -> str:
    selectors = _selector_kwargs(cell, contract)
    return hashlib.sha256(
        _canonical_json(
            {
                "accuracy": cell.measurement.accuracy.value,
                "workload": cell.workload.value,
                "selector_contract": contract.as_dict(),
                "helicities": (
                    None
                    if selectors["helicities"] is None
                    else list(selectors["helicities"])
                ),
                "color_flows": (
                    None
                    if selectors["color_flows"] is None
                    else list(selectors["color_flows"])
                ),
            }
        )
    ).hexdigest()


def _otf_runtime_artifact_id(runtime: object, label: str) -> str:
    artifact_id = getattr(runtime, "artifact_id", None)
    if not isinstance(artifact_id, str) or not _SHA256_PATTERN.fullmatch(artifact_id):
        raise RunnerError(f"{label} runtime has no authenticated artifact ID")
    return artifact_id


def _runtime_execution_mode(runtime: object) -> str:
    value = getattr(runtime, "execution_mode", None)
    return value.value if isinstance(value, ExecutionMode) else str(value)


def _capture_otf_authority_values(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    contract: SelectorContract,
    label: str,
) -> _OtfAuthorityCapture:
    selectors = _selector_kwargs(cell, contract)
    raw_totals = tuple(runtime.evaluate(points, precision=16, **selectors))
    try:
        expected_point_count = len(points)  # type: ignore[arg-type]
    except TypeError as error:
        raise RunnerError("on-the-fly authority points have no finite axis") from error
    if expected_point_count < 1 or len(raw_totals) != expected_point_count:
        raise RunnerError(f"{label} total point axis is empty or differs")
    totals = tuple(_real_nonnegative(value) for value in raw_totals)

    resolved = runtime.evaluate_resolved(points, precision=16, **selectors)
    helicity_ids = tuple(str(item) for item in getattr(resolved, "helicity_ids", ()))
    color_flow_ids = tuple(str(item) for item in getattr(resolved, "color_ids", ()))
    if (
        not helicity_ids
        or not color_flow_ids
        or any(not item for item in helicity_ids + color_flow_ids)
        or len(set(helicity_ids)) != len(helicity_ids)
        or len(set(color_flow_ids)) != len(color_flow_ids)
    ):
        raise RunnerError(f"{label} resolved axes are empty or ambiguous")
    if cell.workload is Workload.SELECTED_FLOW and (
        len(color_flow_ids) != len(contract.selected_color_flow_ids)
        or set(color_flow_ids) != set(contract.selected_color_flow_ids)
    ):
        raise RunnerError(f"{label} resolved output ignored the selected flow")
    if cell.workload is Workload.ALL_FLOW and (
        len(helicity_ids) != len(contract.runtime_all_flow_helicity_ids)
        or set(helicity_ids) != set(contract.runtime_all_flow_helicity_ids)
    ):
        raise RunnerError(f"{label} resolved output ignored the selected helicity")

    raw_components = getattr(resolved, "values", None)
    to_list = getattr(raw_components, "tolist", None)
    if callable(to_list):
        raw_components = to_list()
    if (
        not isinstance(raw_components, Sequence)
        or isinstance(raw_components, (str, bytes, bytearray))
        or len(raw_components) != expected_point_count
    ):
        raise RunnerError(f"{label} resolved point axis is invalid")
    components: list[tuple[tuple[float, ...], ...]] = []
    for point_index, raw_point in enumerate(raw_components):
        if (
            not isinstance(raw_point, Sequence)
            or isinstance(raw_point, (str, bytes, bytearray))
            or len(raw_point) != len(helicity_ids)
        ):
            raise RunnerError(f"{label} resolved helicity axis is invalid")
        point_rows: list[tuple[float, ...]] = []
        for helicity_index, raw_row in enumerate(raw_point):
            if (
                not isinstance(raw_row, Sequence)
                or isinstance(raw_row, (str, bytes, bytearray))
                or len(raw_row) != len(color_flow_ids)
            ):
                raise RunnerError(f"{label} resolved color-flow axis is invalid")
            point_rows.append(
                tuple(
                    _finite_real(
                        value,
                        (
                            f"{label} resolved component "
                            f"{point_index}/{helicity_index}/{color_index}"
                        ),
                    )
                    for color_index, value in enumerate(raw_row)
                )
            )
        components.append(tuple(point_rows))
    frozen_components = tuple(components)
    resolved_total = getattr(resolved, "total", None)
    if not callable(resolved_total):
        raise RunnerError(f"{label} resolved result does not expose total()")
    raw_resolved_totals = tuple(resolved_total())
    if len(raw_resolved_totals) != expected_point_count:
        raise RunnerError(f"{label} resolved total point axis differs")
    resolved_totals = tuple(_real_nonnegative(value) for value in raw_resolved_totals)
    resolved_sum = _otf_compact_series_summary(
        totals,
        resolved_totals,
        label=f"{label} optimized/resolved sum",
    )
    ordering = {
        "helicity_ids": list(helicity_ids),
        "color_flow_ids": list(color_flow_ids),
    }
    return _OtfAuthorityCapture(
        totals=totals,
        helicity_ids=helicity_ids,
        color_flow_ids=color_flow_ids,
        components=frozen_components,
        total_source_sha256=hashlib.sha256(
            _canonical_json(_resolved_component_payload(totals))
        ).hexdigest(),
        resolved_total_source_sha256=hashlib.sha256(
            _canonical_json(_resolved_component_payload(resolved_totals))
        ).hexdigest(),
        resolved_ordering_sha256=hashlib.sha256(_canonical_json(ordering)).hexdigest(),
        resolved_source_sha256=hashlib.sha256(
            _canonical_json(
                {
                    **ordering,
                    "values": _resolved_component_payload(frozen_components),
                }
            )
        ).hexdigest(),
        resolved_sum=resolved_sum,
    )


def _otf_compact_series_summary(
    candidate: Sequence[float],
    authority: Sequence[float],
    *,
    label: str,
    candidate_scales: Sequence[float] | None = None,
    authority_scales: Sequence[float] | None = None,
) -> dict[str, object]:
    if not candidate or len(candidate) != len(authority):
        raise RunnerError(f"{label} comparison axes are empty or differ")
    if (candidate_scales is None) != (authority_scales is None):
        raise RunnerError(f"{label} comparison scales must be paired")
    if candidate_scales is not None and (
        len(candidate_scales) != len(candidate)
        or authority_scales is None
        or len(authority_scales) != len(authority)
    ):
        raise RunnerError(f"{label} comparison scale axes differ")

    if candidate_scales is None:
        scale_pairs: Sequence[tuple[float | None, float | None]] = (
            (None, None),
        ) * len(candidate)
    else:
        assert authority_scales is not None
        scale_pairs = tuple(zip(candidate_scales, authority_scales, strict=True))
    records = tuple(
        pointwise_validation(
            candidate_value,
            authority_value,
            relative_tolerance=RELATIVE_TOLERANCE,
            candidate_scale=candidate_scale,
            baseline_scale=authority_scale,
            candidate_scale_source=(
                "value-magnitude"
                if candidate_scale is None
                else "resolved-component-l1"
            ),
            baseline_scale_source=(
                "value-magnitude"
                if authority_scale is None
                else "resolved-component-l1"
            ),
        )
        for (candidate_value, authority_value), (
            candidate_scale,
            authority_scale,
        ) in zip(
            zip(candidate, authority, strict=True),
            scale_pairs,
            strict=True,
        )
    )
    failed = next(
        (
            index
            for index, record in enumerate(records)
            if record["status"] != ResultStatus.OK.value
        ),
        None,
    )
    if failed is not None:
        raise RunnerError(
            f"{label} disagrees at check {failed}: conditioned residual "
            f"{records[failed]['conditioned_residual']!r}"
        )
    return {
        "check_count": len(records),
        "maximum_conditioned_residual": max(
            float(record["conditioned_residual"]) for record in records
        ),
        "maximum_absolute_delta": max(
            float(record["absolute_difference"]) for record in records
        ),
    }


def _otf_capture_comparison(
    candidate: _OtfAuthorityCapture,
    authority: _OtfAuthorityCapture,
    *,
    authority_cell_id: str,
    label: str,
) -> dict[str, object]:
    total = _otf_compact_series_summary(
        candidate.totals,
        authority.totals,
        label=f"{label} totals",
    )
    if (
        len(candidate.helicity_ids) != len(authority.helicity_ids)
        or set(candidate.helicity_ids) != set(authority.helicity_ids)
        or len(candidate.color_flow_ids) != len(authority.color_flow_ids)
        or set(candidate.color_flow_ids) != set(authority.color_flow_ids)
    ):
        raise RunnerError(f"{label} resolved semantic axes differ")
    authority_helicities = {
        identifier: index for index, identifier in enumerate(authority.helicity_ids)
    }
    authority_colors = {
        identifier: index for index, identifier in enumerate(authority.color_flow_ids)
    }
    candidate_values: list[float] = []
    authority_values: list[float] = []
    candidate_scales: list[float] = []
    authority_scales: list[float] = []
    component_count = len(candidate.helicity_ids) * len(candidate.color_flow_ids)
    for point_index in range(len(candidate.components)):
        candidate_point = candidate.components[point_index]
        authority_point = tuple(
            tuple(
                authority.components[point_index][authority_helicities[helicity_id]][
                    authority_colors[color_id]
                ]
                for color_id in candidate.color_flow_ids
            )
            for helicity_id in candidate.helicity_ids
        )
        candidate_scales.extend(
            [float(_resolved_l1(candidate_point))] * component_count
        )
        authority_scales.extend(
            [float(_resolved_l1(authority_point))] * component_count
        )
        for candidate_row, authority_row in zip(
            candidate_point,
            authority_point,
            strict=True,
        ):
            for candidate_value, authority_value in zip(
                candidate_row,
                authority_row,
                strict=True,
            ):
                candidate_values.append(candidate_value)
                authority_values.append(authority_value)
    resolved = _otf_compact_series_summary(
        candidate_values,
        authority_values,
        label=f"{label} resolved components",
        candidate_scales=candidate_scales,
        authority_scales=authority_scales,
    )
    resolved["component_count"] = component_count
    return {
        "authority_cell_id": authority_cell_id,
        "total": total,
        "resolved_components": resolved,
    }


def _otf_stage_record(
    candidate: _OtfAuthorityCapture,
    authority_cell: CellSpec,
    authority: _OtfAuthorityCapture,
    *,
    stage: str,
) -> dict[str, object]:
    return {
        "candidate_total_source_sha256": candidate.total_source_sha256,
        "candidate_resolved_total_source_sha256": (
            candidate.resolved_total_source_sha256
        ),
        "candidate_resolved_ordering_sha256": (candidate.resolved_ordering_sha256),
        "candidate_resolved_source_sha256": candidate.resolved_source_sha256,
        "candidate_resolved_sum": candidate.resolved_sum,
        "comparison": _otf_capture_comparison(
            candidate,
            authority,
            authority_cell_id=authority_cell.cell_id,
            label=f"on-the-fly {stage}/recurrence",
        ),
    }


def on_the_fly_recurrence_authority_validation(
    candidate: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    selector_contract: SelectorContract,
    authority: tuple[CellSpec, RuntimeLike],
) -> dict[str, object]:
    """Validate cold OTF against recurrence, then reset it for profiling."""

    if (
        cell.measurement.execution_mode is not ExecutionMode.ON_THE_FLY
        or cell.measurement.accuracy is not Accuracy.LC
    ):
        raise RunnerError(
            "recurrence-authority preflight requires an on-the-fly LC cell"
        )
    validate_runtime_contract(cell, candidate)
    if _runtime_execution_mode(candidate) != ExecutionMode.ON_THE_FLY.value:
        raise RunnerError("on-the-fly candidate runtime has the wrong execution mode")
    validate_selector_contract(
        candidate,
        selector_contract,
        points,
        cell=cell,
    )
    candidate_artifact_id = _otf_runtime_artifact_id(candidate, "on-the-fly candidate")

    authority_cell, authority_runtime = authority
    if (
        authority_cell.measurement.execution_mode is not ExecutionMode.RECURRENCE
        or authority_cell.measurement.accuracy is not Accuracy.LC
        or authority_cell.process_key != cell.process_key
        or authority_cell.n_final != cell.n_final
        or authority_cell.workload is not cell.workload
    ):
        raise RunnerError("on-the-fly recurrence authority cell is incompatible")
    validate_runtime_contract(authority_cell, authority_runtime)
    if _runtime_execution_mode(authority_runtime) != ExecutionMode.RECURRENCE.value:
        raise RunnerError("on-the-fly recurrence authority has the wrong mode")
    validate_selector_contract(authority_runtime, selector_contract, points)
    authority_capture = _capture_otf_authority_values(
        authority_runtime,
        points,
        cell=authority_cell,
        contract=selector_contract,
        label="on-the-fly recurrence authority",
    )
    authority_record = {
        "cell_id": authority_cell.cell_id,
        "artifact_id": _otf_runtime_artifact_id(
            authority_runtime, "on-the-fly recurrence authority"
        ),
        "total_source_sha256": authority_capture.total_source_sha256,
        "resolved_total_source_sha256": (
            authority_capture.resolved_total_source_sha256
        ),
        "resolved_ordering_sha256": authority_capture.resolved_ordering_sha256,
        "resolved_source_sha256": authority_capture.resolved_source_sha256,
        "resolved_sum": authority_capture.resolved_sum,
    }

    before = _capture_otf_authority_values(
        candidate,
        points,
        cell=cell,
        contract=selector_contract,
        label="on-the-fly candidate before clear",
    )
    before_record = _otf_stage_record(
        before,
        authority_cell,
        authority_capture,
        stage="before-clear",
    )
    clear = getattr(candidate, "clear", None)
    if not callable(clear):
        raise RunnerError("on-the-fly candidate does not expose Runtime.clear()")
    clear()
    try:
        after = _capture_otf_authority_values(
            candidate,
            points,
            cell=cell,
            contract=selector_contract,
            label="on-the-fly candidate after clear",
        )
        after_record = _otf_stage_record(
            after,
            authority_cell,
            authority_capture,
            stage="after-clear",
        )
    finally:
        # The public profiler must own the next evaluation and its cold clock.
        clear()

    point_count = len(before.totals)
    component_count = len(before.helicity_ids) * len(before.color_flow_ids)
    record = {
        "abi": OTF_RECURRENCE_AUTHORITY_VALIDATION_ABI,
        "status": ResultStatus.OK.value,
        "candidate_cell_id": cell.cell_id,
        "candidate_artifact_id": candidate_artifact_id,
        "workload": cell.workload.value,
        "precision_digits": 16,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "point_digest": point_digest(points),
        "selector_sha256": _otf_selector_digest(cell, selector_contract),
        "point_count": point_count,
        "resolved_component_count": component_count,
        "resolved_check_count": point_count * component_count,
        "authority": authority_record,
        "before_clear": before_record,
        "after_clear": after_record,
        "lifecycle": {
            "authority_artifact_loaded_only": True,
            "candidate_loaded_before_validation": True,
            "validated_before_clear": True,
            "validated_after_clear": True,
            "clear_call_count": 2,
            "final_clear_before_profile": True,
        },
    }
    validate_on_the_fly_recurrence_authority_validation_record(
        record,
        expected_cell=cell,
        selector_contract=selector_contract.as_dict(),
        candidate_artifact_id=candidate_artifact_id,
        expected_authority_cell_id=authority_cell.cell_id,
    )
    return record


def _require_otf_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"on-the-fly recurrence-authority {field} is not SHA-256")
    return value


def _validate_otf_compact_summary(
    value: object,
    *,
    field: str,
    expected_checks: int,
    expected_components: int | None = None,
    relative_tolerance: float,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"on-the-fly recurrence-authority {field} is not an object")
    expected = {
        "check_count",
        "maximum_conditioned_residual",
        "maximum_absolute_delta",
    }
    if expected_components is not None:
        expected.add("component_count")
    check_count = value.get("check_count")
    if (
        set(value) != expected
        or isinstance(check_count, bool)
        or not isinstance(check_count, int)
        or check_count != expected_checks
    ):
        raise ValueError(f"on-the-fly recurrence-authority {field} counts differ")
    if expected_components is not None:
        component_count = value.get("component_count")
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, int)
            or component_count != expected_components
        ):
            raise ValueError(
                f"on-the-fly recurrence-authority {field} component count differs"
            )
    for name in ("maximum_conditioned_residual", "maximum_absolute_delta"):
        raw = value.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ValueError(
                f"on-the-fly recurrence-authority {field}.{name} is invalid"
            )
    if float(value["maximum_conditioned_residual"]) > relative_tolerance:
        raise ValueError(f"on-the-fly recurrence-authority {field} is not successful")


def validate_on_the_fly_recurrence_authority_validation_record(
    value: object,
    *,
    expected_cell: CellSpec | None = None,
    selector_contract: object = None,
    candidate_artifact_id: str | None = None,
    expected_authority_cell_id: str | None = None,
) -> None:
    """Validate the compact, success-only OTF numerical preflight record."""

    if not isinstance(value, Mapping):
        raise ValueError("on-the-fly recurrence-authority validation must be an object")
    expected_fields = {
        "abi",
        "status",
        "candidate_cell_id",
        "candidate_artifact_id",
        "workload",
        "precision_digits",
        "relative_tolerance",
        "point_digest",
        "selector_sha256",
        "point_count",
        "resolved_component_count",
        "resolved_check_count",
        "authority",
        "before_clear",
        "after_clear",
        "lifecycle",
    }
    if set(value) != expected_fields:
        raise ValueError("on-the-fly recurrence-authority validation fields differ")
    if (
        value.get("abi") != OTF_RECURRENCE_AUTHORITY_VALIDATION_ABI
        or value.get("status") != ResultStatus.OK.value
        or value.get("precision_digits") != 16
        or value.get("relative_tolerance") != RELATIVE_TOLERANCE
    ):
        raise ValueError("on-the-fly recurrence-authority validation header is invalid")
    _require_otf_sha256(value.get("candidate_artifact_id"), "candidate_artifact_id")
    _require_otf_sha256(value.get("point_digest"), "point_digest")
    _require_otf_sha256(value.get("selector_sha256"), "selector_sha256")
    point_count = value.get("point_count")
    component_count = value.get("resolved_component_count")
    check_count = value.get("resolved_check_count")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 1
        or isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count < 1
        or check_count != point_count * component_count
    ):
        raise ValueError(
            "on-the-fly recurrence-authority validation counts are invalid"
        )
    if (
        candidate_artifact_id is not None
        and value.get("candidate_artifact_id") != candidate_artifact_id
    ):
        raise ValueError("on-the-fly candidate artifact ID differs")

    if expected_cell is not None:
        if (
            expected_cell.measurement.execution_mode is not ExecutionMode.ON_THE_FLY
            or expected_cell.measurement.accuracy is not Accuracy.LC
            or value.get("candidate_cell_id") != expected_cell.cell_id
            or value.get("workload") != expected_cell.workload.value
        ):
            raise ValueError("on-the-fly recurrence-authority candidate cell differs")
        if not isinstance(selector_contract, Mapping):
            raise ValueError(
                "on-the-fly recurrence-authority selector contract is absent"
            )
        try:
            contract = SelectorContract.from_mapping(selector_contract)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "on-the-fly recurrence-authority selector contract is invalid"
            ) from error
        if value.get("point_digest") != contract.point_digest or value.get(
            "selector_sha256"
        ) != _otf_selector_digest(expected_cell, contract):
            raise ValueError("on-the-fly recurrence-authority selector binding differs")
    elif (
        not isinstance(value.get("candidate_cell_id"), str)
        or not value.get("candidate_cell_id")
        or value.get("workload")
        not in {
            Workload.SELECTED_FLOW.value,
            Workload.ALL_FLOW.value,
        }
    ):
        raise ValueError(
            "on-the-fly recurrence-authority candidate identity is invalid"
        )

    authority = value.get("authority")
    authority_fields = {
        "cell_id",
        "artifact_id",
        "total_source_sha256",
        "resolved_total_source_sha256",
        "resolved_ordering_sha256",
        "resolved_source_sha256",
        "resolved_sum",
    }
    if not isinstance(authority, Mapping) or set(authority) != authority_fields:
        raise ValueError("on-the-fly recurrence-authority fields differ")
    authority_cell_id = authority.get("cell_id")
    if not isinstance(authority_cell_id, str) or not authority_cell_id:
        raise ValueError("on-the-fly recurrence-authority identity is invalid")
    if (
        expected_authority_cell_id is not None
        and authority_cell_id != expected_authority_cell_id
    ):
        raise ValueError("on-the-fly recurrence-authority catalog identity differs")
    for field in (
        "artifact_id",
        "total_source_sha256",
        "resolved_total_source_sha256",
        "resolved_ordering_sha256",
        "resolved_source_sha256",
    ):
        _require_otf_sha256(authority.get(field), f"authority.{field}")
    _validate_otf_compact_summary(
        authority.get("resolved_sum"),
        field="authority.resolved_sum",
        expected_checks=point_count,
        relative_tolerance=RELATIVE_TOLERANCE,
    )

    for stage_name in ("before_clear", "after_clear"):
        stage = value.get(stage_name)
        expected_stage_fields = {
            "candidate_total_source_sha256",
            "candidate_resolved_total_source_sha256",
            "candidate_resolved_ordering_sha256",
            "candidate_resolved_source_sha256",
            "candidate_resolved_sum",
            "comparison",
        }
        if not isinstance(stage, Mapping) or set(stage) != expected_stage_fields:
            raise ValueError(
                f"on-the-fly recurrence-authority {stage_name} fields differ"
            )
        for field in (
            "candidate_total_source_sha256",
            "candidate_resolved_total_source_sha256",
            "candidate_resolved_ordering_sha256",
            "candidate_resolved_source_sha256",
        ):
            _require_otf_sha256(stage.get(field), f"{stage_name}.{field}")
        _validate_otf_compact_summary(
            stage.get("candidate_resolved_sum"),
            field=f"{stage_name}.candidate_resolved_sum",
            expected_checks=point_count,
            relative_tolerance=RELATIVE_TOLERANCE,
        )
        comparison = stage.get("comparison")
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "authority_cell_id",
            "total",
            "resolved_components",
        }:
            raise ValueError(
                f"on-the-fly recurrence-authority {stage_name} comparison fields differ"
            )
        if comparison.get("authority_cell_id") != authority_cell_id:
            raise ValueError(
                f"on-the-fly recurrence-authority {stage_name} identity differs"
            )
        _validate_otf_compact_summary(
            comparison.get("total"),
            field=f"{stage_name}.comparison.total",
            expected_checks=point_count,
            relative_tolerance=RELATIVE_TOLERANCE,
        )
        _validate_otf_compact_summary(
            comparison.get("resolved_components"),
            field=f"{stage_name}.comparison.resolved_components",
            expected_checks=check_count,
            expected_components=component_count,
            relative_tolerance=RELATIVE_TOLERANCE,
        )

    expected_lifecycle = {
        "authority_artifact_loaded_only": True,
        "candidate_loaded_before_validation": True,
        "validated_before_clear": True,
        "validated_after_clear": True,
        "clear_call_count": 2,
        "final_clear_before_profile": True,
    }
    if value.get("lifecycle") != expected_lifecycle:
        raise ValueError("on-the-fly recurrence-authority lifecycle is invalid")


def resolved_sum_validation(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    selector_contract: SelectorContract | None,
    precision: int = 16,
) -> dict[str, object]:
    selectors = _selector_kwargs(cell, selector_contract)
    optimized = runtime.evaluate(points, precision=precision, **selectors)
    resolved = runtime.evaluate_resolved(points, precision=precision, **selectors)
    totals = tuple(resolved.total())
    if len(optimized) != len(totals):
        raise RunnerError("optimized and resolved evaluations have different lengths")
    scale_evidence = resolved_component_scale_evidence(resolved, points)
    scales = scale_evidence["scales"]
    if not isinstance(scales, list) or len(scales) != len(totals):
        raise RunnerError("resolved scale evidence differs from evaluated points")

    records: list[dict[str, object]] = []
    for point_index, (optimized_value, resolved_value, raw_scale) in enumerate(
        zip(optimized, totals, scales, strict=True)
    ):
        candidate = _real_nonnegative(optimized_value)
        baseline = _real_nonnegative(resolved_value)
        scale = float(raw_scale)
        binding = {
            field: scale_evidence[field]
            for field in (
                "abi",
                "point_digest",
                "helicity_ids",
                "color_flow_ids",
                "resolved_ordering_sha256",
                "resolved_source_sha256",
            )
        }
        binding["point_index"] = point_index
        records.append(
            pointwise_validation(
                candidate,
                baseline,
                candidate_scale=max(scale, abs(candidate)),
                baseline_scale=max(scale, abs(baseline)),
                candidate_scale_source="resolved-component-l1",
                baseline_scale_source="resolved-component-l1",
                comparison_binding=binding,
            )
        )
    maximum_absolute = max(
        (float(record["absolute_difference"]) for record in records), default=0.0
    )
    maximum_relative = max(
        (float(record["relative_difference"]) for record in records), default=0.0
    )
    maximum_conditioned = max(
        (float(record["conditioned_residual"]) for record in records), default=0.0
    )
    passed = all(record["status"] == ResultStatus.OK.value for record in records)
    return {
        "abi": RESOLVED_SUM_VALIDATION_ABI,
        "status": (
            ResultStatus.OK.value if passed else ResultStatus.VALIDATION_FAILED.value
        ),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "maximum_conditioned_residual": maximum_conditioned,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "point_digest": point_digest(points),
        "helicity_ids": list(getattr(resolved, "helicity_ids", ())),
        "color_flow_ids": list(getattr(resolved, "color_ids", ())),
        "resolved_ordering_sha256": scale_evidence["resolved_ordering_sha256"],
        "resolved_source_sha256": scale_evidence["resolved_source_sha256"],
        "scale_source": "resolved-component-l1",
        "precision_digits": precision,
        "points": records,
    }


def pointwise_validation(
    candidate: float,
    baseline: float,
    *,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float | None = None,
    candidate_scale: float | None = None,
    baseline_scale: float | None = None,
    candidate_scale_source: str = "value-magnitude",
    baseline_scale_source: str = "value-magnitude",
    comparison_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a scale-conditioned comparison without an absolute escape hatch.

    ``absolute_tolerance`` remains an accepted keyword solely so callers from
    the v1 API fail over without a signature break.  It is deliberately not
    used for acceptance and must be either absent or finite/non-negative.
    """

    numbers = (candidate, baseline, relative_tolerance)
    if any(not math.isfinite(float(value)) for value in numbers):
        raise ValueError("pointwise values and tolerance must be finite")
    if relative_tolerance < 0.0:
        raise ValueError("pointwise tolerance must be non-negative")
    if absolute_tolerance is not None and (
        not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0
    ):
        raise ValueError("legacy absolute tolerance must be finite and non-negative")
    candidate_scale = abs(candidate) if candidate_scale is None else candidate_scale
    baseline_scale = abs(baseline) if baseline_scale is None else baseline_scale
    if any(
        not math.isfinite(float(value)) or value < 0.0
        for value in (candidate_scale, baseline_scale)
    ):
        raise ValueError("comparison scales must be finite and non-negative")
    if candidate_scale < abs(candidate) or baseline_scale < abs(baseline):
        raise ValueError("comparison scale is smaller than its value magnitude")
    absolute = abs(candidate - baseline)
    relative = absolute / max(abs(baseline), 1.0e-300)
    comparison_scale = max(
        abs(candidate), abs(baseline), candidate_scale, baseline_scale
    )
    if comparison_scale == 0.0:
        if candidate != 0.0 or baseline != 0.0 or absolute != 0.0:
            raise ValueError("zero comparison scale requires exact zero values")
        conditioned_residual = 0.0
    else:
        conditioned_residual = absolute / comparison_scale
    error_bound = relative_tolerance * comparison_scale
    passed = absolute <= error_bound
    return {
        "abi": CONDITIONED_COMPARISON_ABI,
        "status": (
            ResultStatus.OK.value if passed else ResultStatus.VALIDATION_FAILED.value
        ),
        "candidate": candidate,
        "baseline": baseline,
        "candidate_scale": candidate_scale,
        "baseline_scale": baseline_scale,
        "candidate_scale_source": candidate_scale_source,
        "baseline_scale_source": baseline_scale_source,
        "comparison_scale": comparison_scale,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "conditioned_residual": conditioned_residual,
        "error_bound": error_bound,
        "relative_tolerance": relative_tolerance,
        "comparison_binding": (
            None if comparison_binding is None else dict(comparison_binding)
        ),
    }


def validate_resolved_sum_validation_record(value: object) -> None:
    """Validate v2 resolved-scale evidence or an exact-zero legacy record."""

    if not isinstance(value, Mapping):
        raise ValueError("resolved-sum validation must be an object")
    if value.get("abi") != RESOLVED_SUM_VALIDATION_ABI:
        expected_v1 = {
            "status",
            "maximum_absolute_difference",
            "maximum_relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
        }
        if set(value) != expected_v1:
            raise ValueError("resolved-sum validation ABI is unsupported")
        numbers: dict[str, float] = {}
        for field in expected_v1 - {"status"}:
            raw = value.get(field)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                raise ValueError(f"legacy resolved-sum {field} is invalid")
            numbers[field] = float(raw)
        old_passed = (
            numbers["maximum_absolute_difference"] <= numbers["absolute_tolerance"]
            or numbers["maximum_relative_difference"] <= numbers["relative_tolerance"]
        )
        if value.get("status") != (
            ResultStatus.OK.value
            if old_passed
            else ResultStatus.VALIDATION_FAILED.value
        ):
            raise ValueError("legacy resolved-sum status is inconsistent")
        # v1 did not retain either side of each comparison, so a nonzero
        # difference cannot be authenticated against the new symmetric scale.
        if old_passed and numbers["maximum_absolute_difference"] != 0.0:
            raise ValueError("legacy resolved-sum evidence is floor-only or unbound")
        return

    expected = {
        "abi",
        "status",
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "maximum_conditioned_residual",
        "relative_tolerance",
        "point_digest",
        "helicity_ids",
        "color_flow_ids",
        "resolved_ordering_sha256",
        "resolved_source_sha256",
        "scale_source",
        "precision_digits",
        "points",
    }
    if set(value) != expected:
        raise ValueError("resolved-sum validation fields differ from the v2 ABI")
    for field in ("point_digest", "resolved_ordering_sha256", "resolved_source_sha256"):
        raw = value.get(field)
        if (
            not isinstance(raw, str)
            or len(raw) != 64
            or any(character not in "0123456789abcdef" for character in raw)
        ):
            raise ValueError(f"resolved-sum {field} is not SHA-256")
    for field in ("helicity_ids", "color_flow_ids"):
        raw = value.get(field)
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item for item in raw
        ):
            raise ValueError(f"resolved-sum {field} is invalid")
    if value.get("scale_source") != "resolved-component-l1":
        raise ValueError("resolved-sum scale source is unsupported")
    precision = value.get("precision_digits")
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
        raise ValueError("resolved-sum precision is invalid")
    relative_tolerance = value.get("relative_tolerance")
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(float(relative_tolerance))
        or float(relative_tolerance) < 0.0
    ):
        raise ValueError("resolved-sum relative tolerance is invalid")
    records = value.get("points")
    if not isinstance(records, list) or not records:
        raise ValueError("resolved-sum points are unavailable")
    for point_index, record in enumerate(records):
        validate_conditioned_comparison_record(record, require_binding=True)
        assert isinstance(record, Mapping)
        if float(record["relative_tolerance"]) != float(relative_tolerance):
            raise ValueError("resolved-sum point tolerance differs")
        binding = record.get("comparison_binding")
        if not isinstance(binding, Mapping):
            raise ValueError("resolved-sum point binding is unavailable")
        expected_binding = {
            "abi": RESOLVED_COMPONENT_SCALE_ABI,
            "point_digest": value["point_digest"],
            "helicity_ids": value["helicity_ids"],
            "color_flow_ids": value["color_flow_ids"],
            "resolved_ordering_sha256": value["resolved_ordering_sha256"],
            "resolved_source_sha256": value["resolved_source_sha256"],
            "point_index": point_index,
        }
        if dict(binding) != expected_binding:
            raise ValueError("resolved-sum point binding differs from its source")
    maximum_absolute = max(float(record["absolute_difference"]) for record in records)
    maximum_relative = max(float(record["relative_difference"]) for record in records)
    maximum_conditioned = max(
        float(record["conditioned_residual"]) for record in records
    )
    for field, expected_value in (
        ("maximum_absolute_difference", maximum_absolute),
        ("maximum_relative_difference", maximum_relative),
        ("maximum_conditioned_residual", maximum_conditioned),
    ):
        raw = value.get(field)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) != expected_value
        ):
            raise ValueError(f"resolved-sum {field} is inconsistent")
    status = (
        ResultStatus.OK.value
        if all(record["status"] == ResultStatus.OK.value for record in records)
        else ResultStatus.VALIDATION_FAILED.value
    )
    if value.get("status") != status:
        raise ValueError("resolved-sum status is inconsistent")


@dataclass(frozen=True, slots=True)
class _ArenaStatistics:
    standard_deviation: float
    standard_error: float
    relative_standard_error: float


@dataclass(frozen=True, slots=True)
class _ArenaBenchmarkResult:
    effective_config: object
    sample_count: int
    wall_time_per_point: float
    evaluator_time_per_point: None
    evaluator_total_time_per_point: float
    uncertainty: _ArenaStatistics
    environment: Mapping[str, object]
    timing_breakdown: Mapping[str, object]
    arena_profile_evidence: Mapping[str, object]
    evaluator_uncertainty: None = None


def _positive_duration(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RunnerError(f"{name} must be finite and positive")
    return float(value)


def _arena_benchmark_batch(points: object, batch_size: int) -> tuple[object, ...]:
    if (
        not isinstance(points, Sequence)
        or isinstance(points, (str, bytes, bytearray))
        or not points
    ):
        raise RunnerError("Arena report benchmark requires validation points")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise RunnerError("Arena report benchmark batch size must be positive")
    source = tuple(points)
    return tuple(source[index % len(source)] for index in range(batch_size))


def _arena_statistics(samples: Sequence[float]) -> _ArenaStatistics:
    if len(samples) < _ARENA_MINIMUM_SAMPLES:
        raise RunnerError(
            f"Arena report benchmark requires at least {_ARENA_MINIMUM_SAMPLES} "
            "timed blocks"
        )
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    error = deviation / math.sqrt(len(samples))
    relative = error / mean
    return _ArenaStatistics(
        standard_deviation=deviation,
        standard_error=error,
        relative_standard_error=relative,
    )


def _calibrate_arena_repetitions(
    timer: object,
    batch: tuple[object, ...],
    *,
    target_runtime: float,
    sample_count: int,
    selector_arguments: Mapping[str, object],
    progress: ProgressSink | None = None,
    task_id: str = "report-profile:calibration",
) -> tuple[int, list[dict[str, object]]]:
    if not callable(timer):
        raise RunnerError("native unprofiled wall timer is unavailable")
    if progress is not None:
        from pyamplicol.reporting import ProgressUpdate

    target_per_block = target_runtime / sample_count
    repetitions = 1
    observed = _positive_duration(
        timer(batch, repetitions, **selector_arguments),
        "Arena wall calibration duration",
    )
    blocks = [{"repetitions": repetitions, "duration_seconds": observed}]
    minimum_repetition_confirmed = False
    if progress is not None:
        progress.emit(
            ProgressUpdate(
                task_id,
                1,
                _ARENA_MAX_CALIBRATION_BLOCKS + 1,
                "initial timing block",
                {
                    "duration_seconds": observed,
                    "repetitions": repetitions,
                },
            )
        )
    for _ in range(_ARENA_MAX_CALIBRATION_BLOCKS):
        estimate = math.ceil(repetitions * target_per_block / observed)
        candidate = min(max(estimate, 1), _ARENA_MAX_REPETITIONS)
        if candidate == repetitions:
            # A busy worker can make the first one-repetition probe arbitrarily
            # slow.  At the repetition floor that outlier otherwise terminates
            # calibration immediately, even when the warmed steady-state call
            # is orders of magnitude faster.  Confirm the floor once and use
            # the faster observation: scheduling contention can delay a probe,
            # but it cannot make the native work complete spuriously early.
            if (
                repetitions == 1
                and observed >= target_per_block
                and not minimum_repetition_confirmed
            ):
                confirmation = _positive_duration(
                    timer(batch, repetitions, **selector_arguments),
                    "Arena wall calibration confirmation duration",
                )
                blocks.append(
                    {
                        "repetitions": repetitions,
                        "duration_seconds": confirmation,
                    }
                )
                observed = min(observed, confirmation)
                minimum_repetition_confirmed = True
                if progress is not None:
                    progress.emit(
                        ProgressUpdate(
                            task_id,
                            len(blocks),
                            _ARENA_MAX_CALIBRATION_BLOCKS + 1,
                            "confirming minimum-repetition timing",
                            {
                                "duration_seconds": confirmation,
                                "selected_duration_seconds": observed,
                                "repetitions": repetitions,
                            },
                        )
                    )
                continue
            break
        repetitions = candidate
        observed = _positive_duration(
            timer(batch, repetitions, **selector_arguments),
            "Arena wall calibration duration",
        )
        blocks.append({"repetitions": repetitions, "duration_seconds": observed})
        if progress is not None:
            progress.emit(
                ProgressUpdate(
                    task_id,
                    len(blocks),
                    _ARENA_MAX_CALIBRATION_BLOCKS + 1,
                    "refining repetitions",
                    {
                        "duration_seconds": observed,
                        "repetitions": repetitions,
                    },
                )
            )
        ratio = observed / target_per_block
        if 0.75 <= ratio <= 1.5:
            break
    return repetitions, blocks


def _run_arena_benchmark(
    runtime: object,
    points: object,
    *,
    execution_mode: str,
    benchmark_config: object,
    selectors: Mapping[str, tuple[str, ...] | None],
    progress: ProgressSink | None = None,
    chunk_guard: ProfilingChunkGuard | None = None,
) -> _ArenaBenchmarkResult:
    """Measure native wall time and authenticate a paired private Arena profile."""

    if execution_mode not in {"compiled", "eager"}:
        raise RunnerError(
            "private warmed Arena profiling is supported only for eager and "
            "compiled report cells"
        )
    timer = getattr(runtime, "_benchmark_f64_wall_time", None)
    profiler = getattr(runtime, "_profile_arena_repeated", None)
    if not callable(timer):
        raise RunnerError(
            "current report timing requires runtime._benchmark_f64_wall_time"
        )
    if not callable(profiler):
        raise RunnerError(
            "current report timing requires runtime._profile_arena_repeated"
        )
    target_runtime = float(getattr(benchmark_config, "target_runtime", 0.0))
    batch_size = getattr(benchmark_config, "batch_size", None)
    warmup_runs = getattr(benchmark_config, "warmup_runs", None)
    minimum_samples = getattr(benchmark_config, "minimum_samples", None)
    precision = getattr(benchmark_config, "precision", None)
    if not math.isfinite(target_runtime) or target_runtime <= 0.0:
        raise RunnerError("Arena report target runtime must be finite and positive")
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise RunnerError("Arena report warmup count is invalid")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < _ARENA_MINIMUM_SAMPLES
    ):
        raise RunnerError(
            f"Arena report timing requires at least {_ARENA_MINIMUM_SAMPLES} samples"
        )
    if precision != 16:
        raise RunnerError("Arena report timing requires native f64 precision")
    if progress is not None:
        from pyamplicol.reporting import ProgressEnd, ProgressStart, ProgressUpdate

    batch = _arena_benchmark_batch(points, batch_size)
    selector_arguments: dict[str, object] = {
        "helicities": selectors.get("helicities"),
        "color_flows": selectors.get("color_flows"),
        "precision": 16,
    }
    profile_arguments = {**selector_arguments, "include_values": False}
    timer_seconds_per_repetition: float | None = None
    profiler_seconds_per_repetition: float | None = None

    def guarded_timer(repetitions: int) -> float:
        nonlocal timer_seconds_per_repetition
        estimate = (
            None
            if timer_seconds_per_repetition is None
            else timer_seconds_per_repetition * repetitions
        )
        if chunk_guard is not None:
            chunk_guard(estimate, f"{execution_mode} evaluator timing chunk")
        observed = _positive_duration(
            timer(batch, repetitions, **selector_arguments),
            "Arena evaluator duration",
        )
        timer_seconds_per_repetition = observed / repetitions
        return observed

    def guarded_profiler(
        repetitions: int,
        *,
        measure_elapsed: bool = True,
    ) -> tuple[Mapping[str, object], float]:
        nonlocal profiler_seconds_per_repetition
        estimated_per_repetition = (
            profiler_seconds_per_repetition
            if profiler_seconds_per_repetition is not None
            else timer_seconds_per_repetition
        )
        estimate = (
            None
            if estimated_per_repetition is None
            else estimated_per_repetition * repetitions
        )
        if chunk_guard is not None:
            chunk_guard(estimate, f"{execution_mode} attribution chunk")
        started = time.perf_counter() if measure_elapsed else None
        raw = profiler(batch, repetitions, **profile_arguments)
        observed = (
            (time.perf_counter() - started)
            if started is not None
            else (estimate or 0.0)
        )
        if not isinstance(raw, Mapping):
            raise RunnerError("native warmed Arena profile is not an object")
        if observed > 0.0:
            profiler_seconds_per_repetition = observed / repetitions
        return raw, observed

    parent_task_id = "report-profile"
    warmup_task_id = f"{parent_task_id}:warmup"
    calibration_task_id = f"{parent_task_id}:calibration"
    samples_task_id = f"{parent_task_id}:samples"
    started = time.perf_counter()
    active_task_id: str | None = None
    if progress is not None:
        progress.emit(
            ProgressStart(
                parent_task_id,
                f"Profiling {execution_mode} runtime",
                None,
                unit="stages",
                details={
                    "target_runtime_seconds": target_runtime,
                    "minimum_samples": minimum_samples,
                },
            )
        )
    try:
        active_task_id = warmup_task_id
        if progress is not None:
            progress.emit(
                ProgressStart(
                    warmup_task_id,
                    "Warming the native runtime",
                    warmup_runs,
                    parent_task_id=parent_task_id,
                    unit="runs",
                )
            )
        warmup_started = time.perf_counter()
        for index in range(warmup_runs):
            guarded_timer(1)
            warmup_profile, _warmup_profile_elapsed = guarded_profiler(
                1,
                measure_elapsed=False,
            )
            try:
                build_arena_profile_evidence(
                    (warmup_profile,),
                    execution_mode=execution_mode,
                    repetitions_per_profile=1,
                    batch_size=len(batch),
                )
            except ArenaProfileEvidenceError as error:
                raise RunnerError(f"invalid warmed Arena profile: {error}") from error
            if progress is not None:
                progress.emit(
                    ProgressUpdate(
                        warmup_task_id,
                        index + 1,
                        warmup_runs,
                        "runtime warmed",
                    )
                )
        warmup_elapsed = time.perf_counter() - warmup_started
        if progress is not None:
            progress.emit(
                ProgressEnd(
                    warmup_task_id,
                    elapsed_seconds=warmup_elapsed,
                )
            )
        active_task_id = calibration_task_id
        if progress is not None:
            progress.emit(
                ProgressStart(
                    calibration_task_id,
                    "Calibrating repetitions",
                    _ARENA_MAX_CALIBRATION_BLOCKS + 1,
                    parent_task_id=parent_task_id,
                    unit="blocks",
                )
            )
        calibration_started = time.perf_counter()
        repetitions, calibration_blocks = _calibrate_arena_repetitions(
            lambda _batch, repetitions, **_arguments: guarded_timer(repetitions),
            batch,
            target_runtime=target_runtime,
            sample_count=minimum_samples,
            selector_arguments=selector_arguments,
            progress=progress,
            task_id=calibration_task_id,
        )
        calibration_elapsed = time.perf_counter() - calibration_started
        if progress is not None:
            progress.emit(
                ProgressEnd(
                    calibration_task_id,
                    elapsed_seconds=calibration_elapsed,
                    details={
                        "blocks": len(calibration_blocks),
                        "repetitions": repetitions,
                    },
                )
            )
        active_task_id = samples_task_id
        if progress is not None:
            progress.emit(
                ProgressStart(
                    samples_task_id,
                    "Measuring paired runtime samples",
                    _ARENA_MAX_SAMPLES,
                    parent_task_id=parent_task_id,
                    unit="samples",
                    details={
                        "minimum_samples": minimum_samples,
                        "target_runtime_seconds": target_runtime,
                    },
                )
            )
        samples_started = time.perf_counter()
        evaluated_points = repetitions * len(batch)
        headline_samples: list[float] = []
        headline_durations: list[float] = []
        evaluator_total_durations: list[float] = []
        raw_profiles: list[Mapping[str, object]] = []
        profile_elapsed = 0.0
        while (
            len(headline_samples) < minimum_samples
            or math.fsum(headline_durations) < target_runtime
        ):
            if len(headline_samples) >= _ARENA_MAX_SAMPLES:
                raise RunnerError(
                    "Arena report benchmark could not reach its target duration "
                    f"within {_ARENA_MAX_SAMPLES} samples"
                )
            headline_started = time.perf_counter()
            evaluator_total_duration = guarded_timer(repetitions)
            wall_duration = _positive_duration(
                time.perf_counter() - headline_started,
                "Arena headline wall duration",
            )
            headline_durations.append(wall_duration)
            evaluator_total_durations.append(evaluator_total_duration)
            headline_samples.append(wall_duration / evaluated_points)
            raw_profile, raw_profile_elapsed = guarded_profiler(repetitions)
            profile_elapsed += raw_profile_elapsed
            raw_profiles.append(raw_profile)
            if progress is not None:
                progress.emit(
                    ProgressUpdate(
                        samples_task_id,
                        len(headline_samples),
                        _ARENA_MAX_SAMPLES,
                        "sample complete",
                        {
                            "elapsed_seconds": math.fsum(headline_durations),
                            "minimum_samples": minimum_samples,
                            "target_runtime_seconds": target_runtime,
                        },
                    )
                )
        samples_elapsed = time.perf_counter() - samples_started
        if progress is not None:
            progress.emit(
                ProgressEnd(
                    samples_task_id,
                    elapsed_seconds=samples_elapsed,
                    details={
                        "samples": len(headline_samples),
                        "measured_seconds": math.fsum(headline_durations),
                    },
                )
            )
        active_task_id = None
        try:
            arena_evidence = build_arena_profile_evidence(
                raw_profiles,
                execution_mode=execution_mode,
                repetitions_per_profile=repetitions,
                batch_size=len(batch),
            )
        except ArenaProfileEvidenceError as error:
            raise RunnerError(f"invalid warmed Arena profile: {error}") from error
        sample_count = len(headline_samples)
        uncertainty = _arena_statistics(headline_samples)
    except BaseException as error:
        if progress is not None:
            if active_task_id is not None:
                progress.emit(
                    ProgressEnd(active_task_id, success=False, message=str(error))
                )
            progress.emit(
                ProgressEnd(
                    parent_task_id,
                    success=False,
                    message=str(error),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        raise
    mean_wall = statistics.fmean(headline_samples)
    achieved_runtime = math.fsum(headline_durations)
    evaluator_total_accumulated = math.fsum(evaluator_total_durations)
    measured_evaluations = sample_count * repetitions
    measured_points = measured_evaluations * len(batch)
    evaluator_total_time_per_point = evaluator_total_accumulated / measured_points
    profile_total_elapsed = time.perf_counter() - started
    if progress is not None:
        progress.emit(
            ProgressEnd(
                parent_task_id,
                elapsed_seconds=profile_total_elapsed,
            )
        )
    return _ArenaBenchmarkResult(
        effective_config=benchmark_config,
        sample_count=sample_count,
        wall_time_per_point=mean_wall,
        evaluator_time_per_point=None,
        evaluator_total_time_per_point=evaluator_total_time_per_point,
        uncertainty=uncertainty,
        environment={
            "wall_time_source": "runtime_core_repeated_wall_time",
            "wall_time_sample_pass": "runtime._benchmark_f64_wall_time",
            "evaluator_time_raw_seconds_per_point": None,
            "evaluator_time_status": UNAVAILABLE_STATUS,
            "evaluator_time_ratio_eligible": False,
            "evaluator_time_sample_pass": ARENA_PROFILE_SAMPLE_PASS,
            "evaluator_total_time_raw_seconds_per_point": (
                evaluator_total_time_per_point
            ),
            "evaluator_total_time_status": "measured",
            "evaluator_total_time_ratio_eligible": False,
            "evaluator_total_time_source": EVALUATOR_TOTAL_TIMING_SOURCE,
            "evaluator_total_time_sample_contract": (EVALUATOR_TOTAL_SAMPLE_CONTRACT),
            "evaluator_total_accumulated_seconds": (evaluator_total_accumulated),
            "timing_breakdown_sample_pass": ARENA_PROFILE_SAMPLE_PASS,
            "profile_protocol": ARENA_PROFILE_PROTOCOL,
            "report_command_path": PAIRED_ARENA_PROFILE_COMMAND_PATH,
            "report_public_cli_path": None,
            "profile_attribution_boundary": ARENA_PROFILE_BOUNDARY,
            "profile_attribution_borrowed_flat_input": True,
            "profile_attribution_preallocated_output": True,
            "profile_attribution_phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
            "profile_attribution_evaluator_timing_available": False,
            "profile_attribution_paired_with_headline": True,
            "profile_attribution_identical_batch": True,
            "profile_attribution_identical_repetitions": True,
            "execution_mode": execution_mode,
            "evaluator_sample_count": sample_count,
            "native_profile_sample_count": sample_count,
            "native_profile_points_per_sample": evaluated_points,
            "native_profile_repetitions_per_sample": repetitions,
            "native_profile_batch_size": len(batch),
            "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            "elapsed_seconds": achieved_runtime,
            "warmup_elapsed_seconds": warmup_elapsed,
            "calibration_elapsed_seconds": calibration_elapsed,
            "calibration_outer_elapsed_seconds": calibration_elapsed,
            "measurement_phase_elapsed_seconds": samples_elapsed,
            "profile_total_elapsed_seconds": profile_total_elapsed,
            "completed_sample_count": sample_count,
            "planned_sample_count": sample_count,
            "repetitions_per_sample": repetitions,
            "measured_point_count": measured_points,
            "profile_attribution_evaluation_count": measured_evaluations,
            "profile_attribution_point_count": measured_points,
            "profile_attribution_elapsed_seconds": profile_elapsed,
            "headline_block_durations_seconds": headline_durations,
            "calibration": {
                "target_seconds_per_block": (target_runtime / minimum_samples),
                "blocks": calibration_blocks,
            },
            "interrupted": False,
        },
        timing_breakdown={
            "sample_count": sample_count,
            "execution_mode": execution_mode,
            "wall_time": {
                "mean_seconds_per_point": arena_evidence[
                    "warmed_boundary_wall_seconds_per_point"
                ],
            },
            "evaluator_call_time": None,
            "raw_profile_samples": arena_evidence["raw_profiles"],
        },
        arena_profile_evidence=arena_evidence,
    )


def _profile_cli_argv(
    benchmark_config: BenchmarkConfig,
) -> tuple[str, ...]:
    """Return public ``pyamplicol profile`` arguments for one loaded runtime."""

    arguments = [
        "profile",
        ".pyamplicol-loaded-runtime",
        "--target-runtime",
        str(benchmark_config.target_runtime),
        "--batch-size",
        str(benchmark_config.batch_size),
        "--precision",
        str(benchmark_config.precision),
        "--warmup-runs",
        str(benchmark_config.warmup_runs),
        "--minimum-samples",
        str(benchmark_config.minimum_samples),
        "--format",
        "json",
        "--progress",
        "off",
    ]
    for identifier in benchmark_config.helicity_ids:
        arguments.extend(("--helicity", str(identifier)))
    for identifier in benchmark_config.color_flow_ids:
        arguments.extend(("--color-flow", str(identifier)))
    return tuple(arguments)


def _run_report_benchmark(
    runtime: object,
    points: object,
    *,
    execution_mode: ExecutionMode,
    benchmark_config: BenchmarkConfig,
    selectors: Mapping[str, tuple[str, ...] | None],
    progress: ProgressSink | None = None,
    chunk_guard: ProfilingChunkGuard | None = None,
) -> object:
    """Select the authenticated timing command path without fallback.

    Recurrence and on-the-fly use the public ``profile`` parser/config/dispatch
    path.  Their narrowly scoped service override supplies the already
    authenticated runtime and exact in-memory point set, avoiding an artifact
    reload inside the measured campaign.  Compiled/eager retain the report-only
    paired private Arena protocol because the public profiler does not expose
    that paired headline/attribution boundary.
    """

    if execution_mode in {ExecutionMode.EAGER, ExecutionMode.COMPILED}:
        backend = getattr(runtime, "_backend", None)
        if backend is None:
            raise RunnerError(
                "eager/compiled report timing requires the runtime's private "
                "native backend"
            )
        return _run_arena_benchmark(
            backend,
            points,
            execution_mode=execution_mode.value,
            benchmark_config=benchmark_config,
            selectors=selectors,
            progress=progress,
            chunk_guard=chunk_guard,
        )
    if execution_mode in {ExecutionMode.RECURRENCE, ExecutionMode.ON_THE_FLY}:
        from pyamplicol.api import BenchmarkRunner
        from pyamplicol.cli import CliInvocation, parse_cli
        from pyamplicol.cli.handlers import DefaultCliServices, dispatch
        from pyamplicol.config import RunConfig
        from pyamplicol.reporting import NullProgressSink

        selected_progress = progress or NullProgressSink()
        invocation = parse_cli(_profile_cli_argv(benchmark_config))
        if not isinstance(invocation, CliInvocation):
            raise RunnerError("public profile parser returned a non-command invocation")
        resolution = invocation.resolve()
        if resolution.effective.benchmark != benchmark_config:
            raise RunnerError(
                "public profile arguments do not preserve report benchmark settings"
            )
        expected_helicities = tuple(selectors["helicities"] or ())
        expected_color_flows = tuple(selectors["color_flows"] or ())
        if (
            resolution.effective.benchmark.helicity_ids != expected_helicities
            or resolution.effective.benchmark.color_flow_ids != expected_color_flows
        ):
            raise RunnerError(
                "public profile arguments do not preserve report selectors"
            )

        class _LoadedRuntimeProfileServices(DefaultCliServices):
            """Inject only the authenticated loaded runtime and exact points.

            Parsing, typed resolution, progress propagation, benchmark settings,
            and dispatch remain the public ``pyamplicol profile`` path.
            """

            def benchmark(
                self,
                config: RunConfig,
                command_progress: ProgressSink,
            ) -> object:
                profile_started = time.perf_counter()
                runner_arguments: dict[str, object] = {
                    "progress": command_progress,
                }
                if chunk_guard is not None:
                    runner_arguments["_chunk_guard"] = chunk_guard
                result = BenchmarkRunner(config, **runner_arguments).run(
                    runtime,
                    points=points,
                )
                profile_total_elapsed = time.perf_counter() - profile_started
                environment = getattr(result, "environment", None)
                if not isinstance(environment, Mapping):
                    raise RunnerError(
                        "public profile result does not retain environment evidence"
                    )
                return replace(
                    result,
                    environment={
                        **environment,
                        "report_command_path": (LOADED_RUNTIME_PROFILE_COMMAND_PATH),
                        "report_public_cli_path": PUBLIC_CLI_COMMAND_PATH,
                        "profile_total_elapsed_seconds": profile_total_elapsed,
                    },
                )

        return dispatch(
            resolution.effective,
            _LoadedRuntimeProfileServices(),
            selected_progress,
            dry_run=invocation.dry_run,
        )
    raise RunnerError(
        f"unsupported pyAmpliCol report execution mode: {execution_mode.value}"
    )


def _warmup_finite_number(
    value: object,
    field: str,
    *,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) <= 0.0 if positive else float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"runtime profile {field} must be a finite {qualifier} number")
    return float(value)


def _warmup_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"runtime profile {field} must be a non-negative integer")
    return value


def _warmup_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"runtime profile {field} must be a boolean")
    return value


def _warmup_plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _warmup_plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_warmup_plain_value(item) for item in value)
    if isinstance(value, list):
        return [_warmup_plain_value(item) for item in value]
    return value


def _validate_otf_runtime_state_census(
    value: object,
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime profile {field} must be a mapping")
    expected_fields = {
        "kind",
        "process_id",
        "family_cache_policy",
        "family_cache_limit",
        "active_family_union_census",
        *OTF_RUNTIME_STATE_COUNT_FIELDS,
    }
    if set(value) != expected_fields:
        raise ValueError(f"runtime profile {field} has invalid runtime census fields")
    if value.get("kind") != OTF_RUNTIME_STATE_CENSUS_KIND:
        raise ValueError(f"runtime profile {field} has an invalid runtime census kind")
    if value.get("family_cache_policy") != OTF_RUNTIME_STATE_FAMILY_CACHE_POLICY:
        raise ValueError(
            f"runtime profile {field} has an unsupported family cache policy"
        )
    family_cache_limit = value.get("family_cache_limit")
    if (
        isinstance(family_cache_limit, bool)
        or not isinstance(family_cache_limit, int)
        or family_cache_limit != OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT
    ):
        raise ValueError(
            f"runtime profile {field} must use family cache limit "
            f"{OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT}"
        )
    process_id = value.get("process_id")
    if not isinstance(process_id, str) or not process_id:
        raise ValueError(
            f"runtime profile {field} runtime census has an invalid process ID"
        )
    for count_field in OTF_RUNTIME_STATE_COUNT_FIELDS:
        _warmup_nonnegative_integer(value[count_field], f"{field}.{count_field}")
    if value["retained_family_count"] > OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT:
        raise ValueError(
            f"runtime profile {field} retained family count exceeds its cache limit"
        )

    active = value.get("active_family_union_census")
    if active is None:
        return value
    if not isinstance(active, Mapping):
        raise ValueError(
            f"runtime profile {field} active-family census must be a mapping or null"
        )
    expected_active_fields = {
        "basis",
        "scope",
        *OTF_ACTIVE_FAMILY_COUNT_FIELDS,
    }
    if set(active) != expected_active_fields:
        raise ValueError(
            f"runtime profile {field} has invalid active-family census fields"
        )
    if (
        active.get("basis") != "shared-query-family-union-v1"
        or active.get("scope") != "active-family-union"
    ):
        raise ValueError(
            f"runtime profile {field} has an invalid active-family census identity"
        )
    for count_field in OTF_ACTIVE_FAMILY_COUNT_FIELDS:
        _warmup_nonnegative_integer(
            active[count_field],
            f"{field}.active_family_union_census.{count_field}",
        )
    if (
        active["query_count"] < 1
        or active["union_unique_current_count"]
        > active["union_unique_current_component_count"]
        or active["union_amplitude_destination_count"] < 1
        or active["union_amplitude_destination_count"] > active["query_count"]
        or active["query_count"] > value["retained_request_count"]
        or active["union_amplitude_destination_count"]
        > value["retained_amplitude_destination_count"]
    ):
        raise ValueError(
            f"runtime profile {field} active-family census is inconsistent"
        )
    for role in OTF_OPERATION_ROLES:
        if active[f"union_{role}_executor_call_groups"] > active[f"union_{role}_rows"]:
            raise ValueError(
                f"runtime profile {field} active-family {role} call-group "
                "count exceeds its row count"
            )
    return value


def _otf_runtime_state_is_cold(value: Mapping[str, object]) -> bool:
    return all(value[field] == 0 for field in OTF_RUNTIME_STATE_COUNT_FIELDS) and (
        value["active_family_union_census"] is None
    )


def _otf_runtime_state_is_retained(value: Mapping[str, object]) -> bool:
    if (
        value["pending_family_count"] != 0
        or value["process_preparation_count"] != 1
        or value["retained_family_count"] != OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT
        or value["retained_selection_count"] != OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT
        or _warmup_nonnegative_integer(
            value["retained_request_count"],
            "retained_request_count",
        )
        < 1
    ):
        return False
    executable_counts = tuple(
        _warmup_nonnegative_integer(value[field], field)
        for field in OTF_RUNTIME_STATE_RETAINED_EXECUTABLE_FIELDS
    )
    active = value["active_family_union_census"]
    if isinstance(active, Mapping):
        return (
            executable_counts[0] > 0
            and executable_counts[1] == OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT
            and executable_counts[2] > 0
        )
    return active is None and all(count == 0 for count in executable_counts)


def validate_profile_warmup_evidence(
    value: object,
    *,
    execution_mode: ExecutionMode | str,
    expected_batch_size: int | None = None,
    expected_warmup_run_count: int | None = None,
) -> None:
    """Validate bounded OTF cold preparation and conventional warm-ups.

    The fields are the flat environment evidence emitted by
    :class:`BenchmarkBackend`; this validator deliberately does not introduce a
    second timing record or reinterpret the backend's clocks.
    """

    if not isinstance(value, Mapping):
        raise ValueError("runtime profile warm-up evidence must be a mapping")
    mode = (
        execution_mode.value
        if isinstance(execution_mode, ExecutionMode)
        else execution_mode
    )
    is_otf = mode == ExecutionMode.ON_THE_FLY.value
    conventional_present = CONVENTIONAL_WARMUP_FIELDS & value.keys()
    cold_present = OTF_COLD_WARMUP_FIELDS & value.keys()
    if is_otf and conventional_present != CONVENTIONAL_WARMUP_FIELDS:
        missing = sorted(CONVENTIONAL_WARMUP_FIELDS - value.keys())
        raise ValueError(
            "on-the-fly runtime profile has incomplete conventional warm-up "
            f"evidence; missing={missing}"
        )
    if is_otf and cold_present != OTF_COLD_WARMUP_FIELDS:
        missing = sorted(OTF_COLD_WARMUP_FIELDS - value.keys())
        raise ValueError(
            "on-the-fly runtime profile has incomplete cold warm-up evidence; "
            f"missing={missing}"
        )
    if not is_otf and cold_present:
        raise ValueError(
            "non-on-the-fly runtime profile cannot contain cold warm-up evidence"
        )
    if (
        not is_otf
        and conventional_present
        and conventional_present != CONVENTIONAL_WARMUP_FIELDS
    ):
        if conventional_present != {"warmup_elapsed_seconds"}:
            missing = sorted(CONVENTIONAL_WARMUP_FIELDS - value.keys())
            raise ValueError(
                "runtime profile has unsupported partial conventional warm-up "
                f"evidence; missing={missing}"
            )
        _warmup_finite_number(
            value["warmup_elapsed_seconds"],
            "warmup_elapsed_seconds",
        )
        return
    if not conventional_present:
        return

    warmup_elapsed = _warmup_finite_number(
        value["warmup_elapsed_seconds"],
        "warmup_elapsed_seconds",
    )
    warmup_count = _warmup_nonnegative_integer(
        value["warmup_configured_run_count"],
        "warmup_configured_run_count",
    )
    if expected_warmup_run_count is not None and (
        isinstance(expected_warmup_run_count, bool)
        or not isinstance(expected_warmup_run_count, int)
        or expected_warmup_run_count < 0
        or warmup_count != expected_warmup_run_count
    ):
        raise ValueError(
            "runtime profile warmup_configured_run_count does not match the "
            "effective benchmark configuration"
        )
    warmup_batch_size = _warmup_nonnegative_integer(
        value["warmup_batch_size"],
        "warmup_batch_size",
    )
    if warmup_batch_size < 1:
        raise ValueError("runtime profile warmup_batch_size must be positive")
    warmup_point_count = _warmup_nonnegative_integer(
        value["warmup_point_count"],
        "warmup_point_count",
    )
    if warmup_point_count != warmup_count * warmup_batch_size:
        raise ValueError(
            "runtime profile warmup_point_count does not equal configured runs "
            "times batch size"
        )
    raw_runs = value["warmup_run_outer_wall_seconds"]
    if isinstance(raw_runs, (str, bytes)) or not isinstance(raw_runs, Sequence):
        raise ValueError(
            "runtime profile warmup_run_outer_wall_seconds must be a sequence"
        )
    runs = tuple(
        _warmup_finite_number(item, "warmup_run_outer_wall_seconds[]")
        for item in raw_runs
    )
    if len(runs) != warmup_count:
        raise ValueError(
            "runtime profile warmup_run_outer_wall_seconds length does not match "
            "the configured run count"
        )
    warmup_sum = sum(runs)
    consistency_tolerance = max(
        1.0e-12,
        max(abs(warmup_elapsed), abs(warmup_sum)) * 1.0e-12,
    )
    if abs(warmup_sum - warmup_elapsed) > consistency_tolerance:
        raise ValueError(
            "runtime profile warmup_elapsed_seconds does not equal its per-run "
            "outer-wall timing sum"
        )
    first_run = value["first_warmup_run_outer_wall_seconds"]
    if warmup_count == 0:
        if first_run is not None:
            raise ValueError(
                "runtime profile first_warmup_run_outer_wall_seconds must be null "
                "when no warm-up runs are configured"
            )
    elif (
        _warmup_finite_number(
            first_run,
            "first_warmup_run_outer_wall_seconds",
        )
        != runs[0]
    ):
        raise ValueError(
            "runtime profile first warm-up timing does not match the first per-run "
            "timing"
        )
    if value["warmup_timer_source"] != WARMUP_TIMER_SOURCE:
        raise ValueError("runtime profile warmup_timer_source is unsupported")
    if value["warmup_timing_scope"] != CONVENTIONAL_WARMUP_TIMING_SCOPE:
        raise ValueError("runtime profile warmup_timing_scope is unsupported")

    if not is_otf:
        return
    cold_elapsed = _warmup_finite_number(
        value["cold_warmup_elapsed_seconds"],
        "cold_warmup_elapsed_seconds",
        positive=True,
    )
    if cold_elapsed <= 0.0:  # Defensive clarity for static type narrowing.
        raise ValueError("on-the-fly cold warm-up timing must be positive")
    cold_runs = _warmup_nonnegative_integer(
        value["cold_warmup_run_count"],
        "cold_warmup_run_count",
    )
    cold_batch_size = _warmup_nonnegative_integer(
        value["cold_warmup_batch_size"],
        "cold_warmup_batch_size",
    )
    cold_point_count = _warmup_nonnegative_integer(
        value["cold_warmup_point_count"],
        "cold_warmup_point_count",
    )
    if cold_runs != 1:
        raise ValueError("on-the-fly cold warm-up must contain exactly one run")
    if (
        cold_batch_size < 1
        or cold_point_count != cold_batch_size
        or cold_batch_size != warmup_batch_size
    ):
        raise ValueError(
            "on-the-fly cold warm-up must cover exactly the full benchmark batch"
        )
    if expected_batch_size is not None and (
        isinstance(expected_batch_size, bool)
        or not isinstance(expected_batch_size, int)
        or expected_batch_size < 1
        or cold_batch_size != expected_batch_size
    ):
        raise ValueError(
            "on-the-fly cold warm-up batch does not match the effective "
            "benchmark configuration"
        )
    if value["cold_warmup_timer_source"] != WARMUP_TIMER_SOURCE:
        raise ValueError("on-the-fly cold warm-up timer source is unsupported")
    if value["cold_warmup_timing_scope"] != OTF_COLD_WARMUP_TIMING_SCOPE:
        raise ValueError(
            "on-the-fly cold warm-up scope must exclude artifact generation and "
            "Runtime/artifact load"
        )
    if (
        value["cold_warmup_runtime_state_evidence"]
        != OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE
    ):
        raise ValueError(
            "on-the-fly cold warm-up runtime state evidence is unsupported"
        )
    state_before = _validate_otf_runtime_state_census(
        value["cold_warmup_runtime_state_before"],
        "cold_warmup_runtime_state_before",
    )
    state_after = _validate_otf_runtime_state_census(
        value["cold_warmup_runtime_state_after"],
        "cold_warmup_runtime_state_after",
    )
    if state_before["process_id"] != state_after["process_id"]:
        raise ValueError(
            "on-the-fly cold warm-up runtime state snapshots identify different "
            "processes"
        )
    runtime_was_cold = _otf_runtime_state_is_cold(state_before)
    runtime_was_retained = _otf_runtime_state_is_retained(state_before)
    runtime_is_retained = _otf_runtime_state_is_retained(state_after)
    if runtime_was_cold == runtime_was_retained:
        raise ValueError(
            "on-the-fly runtime state before the first evaluation must be exactly "
            "cold or strictly retained"
        )
    if not runtime_is_retained:
        raise ValueError(
            "on-the-fly runtime state after the first evaluation must be strictly "
            "retained"
        )
    if (
        _warmup_boolean(
            value["cold_warmup_runtime_cold_before_first_evaluation"],
            "cold_warmup_runtime_cold_before_first_evaluation",
        )
        is not runtime_was_cold
        or _warmup_boolean(
            value["cold_warmup_runtime_retained_before_first_evaluation"],
            "cold_warmup_runtime_retained_before_first_evaluation",
        )
        is not runtime_was_retained
        or _warmup_boolean(
            value["cold_warmup_runtime_retained_after_first_evaluation"],
            "cold_warmup_runtime_retained_after_first_evaluation",
        )
        is not runtime_is_retained
    ):
        raise ValueError(
            "on-the-fly cold warm-up runtime state booleans do not match the "
            "authenticated census snapshots"
        )
    expected_freshness = (
        OTF_COLD_WARMUP_RUNTIME_FRESHNESS
        if runtime_was_cold
        else OTF_COLD_WARMUP_RUNTIME_RETAINED_FRESHNESS
    )
    if value["cold_warmup_runtime_freshness"] != expected_freshness:
        raise ValueError(
            "on-the-fly cold warm-up freshness does not match the authenticated "
            "runtime state"
        )
    if (
        value["cold_warmup_ratio_eligible"] is not False
        or value["cold_warmup_acceptance_eligible"] is not False
    ):
        raise ValueError(
            "on-the-fly cold warm-up must be ineligible for ratios and acceptance"
        )


def _benchmark_measurement(
    benchmark: object,
    *,
    matrix_element: float,
) -> dict[str, object]:
    uncertainty = benchmark.uncertainty
    environment = getattr(benchmark, "environment", None)
    if not isinstance(environment, Mapping):
        raise RunnerError("benchmark did not retain timing provenance")
    evaluator_time = benchmark.evaluator_time_per_point
    raw_evaluator_time = environment.get("evaluator_time_raw_seconds_per_point")
    timing_status = environment.get("evaluator_time_status")
    ratio_eligible = environment.get("evaluator_time_ratio_eligible")
    evaluator_sample_count = environment.get(
        "evaluator_sample_count",
        environment.get("native_profile_sample_count", benchmark.sample_count),
    )
    if (
        isinstance(evaluator_sample_count, bool)
        or not isinstance(evaluator_sample_count, int)
        or evaluator_sample_count < 1
    ):
        raise RunnerError("benchmark has invalid evaluator timing sample count")
    raw_points_per_sample = environment.get("native_profile_points_per_sample")
    if (
        isinstance(raw_points_per_sample, bool)
        or not isinstance(raw_points_per_sample, int)
        or raw_points_per_sample < 1
    ):
        raise RunnerError("benchmark has invalid native-profile point count")
    timing_sample_contract = environment.get("timing_sample_contract")
    arena_profile_evidence: Mapping[str, object] | None = None
    if timing_status == UNAVAILABLE_STATUS:
        breakdown = getattr(benchmark, "timing_breakdown", None)
        wall_component = (
            breakdown.get("wall_time")
            if isinstance(breakdown, Mapping)
            else getattr(breakdown, "wall_time", None)
        )
        warmed_wall = (
            wall_component.get("mean_seconds_per_point")
            if isinstance(wall_component, Mapping)
            else getattr(wall_component, "mean_seconds_per_point", None)
        )
        evaluator_component = (
            breakdown.get("evaluator_call_time")
            if isinstance(breakdown, Mapping)
            else getattr(breakdown, "evaluator_call_time", None)
        )
        raw_arena_evidence = getattr(benchmark, "arena_profile_evidence", None)
        profile_repetitions = environment.get("native_profile_repetitions_per_sample")
        profile_batch_size = environment.get("native_profile_batch_size")
        try:
            arena_profile_evidence = validate_arena_profile_evidence(
                raw_arena_evidence,
                execution_mode=str(environment.get("execution_mode")),
                sample_count=evaluator_sample_count,
                native_profile_points_per_sample=raw_points_per_sample,
            )
        except ArenaProfileEvidenceError as error:
            raise RunnerError(
                f"benchmark has invalid warmed Arena profile evidence: {error}"
            ) from error
        if (
            evaluator_time is not None
            or getattr(benchmark, "evaluator_uncertainty", None) is not None
            or raw_evaluator_time is not None
            or ratio_eligible is not False
            or environment.get("profile_protocol") != ARENA_PROFILE_PROTOCOL
            or environment.get("evaluator_time_sample_pass")
            != ARENA_PROFILE_SAMPLE_PASS
            or environment.get("timing_breakdown_sample_pass")
            != ARENA_PROFILE_SAMPLE_PASS
            or environment.get("profile_attribution_boundary") != ARENA_PROFILE_BOUNDARY
            or environment.get("profile_attribution_borrowed_flat_input") is not True
            or environment.get("profile_attribution_preallocated_output") is not True
            or environment.get("profile_attribution_phase_timing_scope")
            != ARENA_PHASE_TIMING_SCOPE
            or environment.get("profile_attribution_evaluator_timing_available")
            is not False
            or environment.get("profile_attribution_paired_with_headline") is not True
            or environment.get("profile_attribution_identical_batch") is not True
            or environment.get("profile_attribution_identical_repetitions") is not True
            or timing_sample_contract != PAIRED_TIMING_SAMPLE_CONTRACT
            or environment.get("execution_mode") not in {"compiled", "eager"}
            or evaluator_sample_count != benchmark.sample_count
            or isinstance(profile_repetitions, bool)
            or not isinstance(profile_repetitions, int)
            or profile_repetitions < 1
            or isinstance(profile_batch_size, bool)
            or not isinstance(profile_batch_size, int)
            or profile_batch_size < 1
            or profile_repetitions * profile_batch_size != raw_points_per_sample
            or arena_profile_evidence.get("repetitions_per_profile")
            != profile_repetitions
            or arena_profile_evidence.get("batch_size") != profile_batch_size
            or evaluator_component is not None
            or isinstance(warmed_wall, bool)
            or not isinstance(warmed_wall, (int, float))
            or not math.isfinite(float(warmed_wall))
            or float(warmed_wall) <= 0.0
        ):
            raise RunnerError(
                "benchmark unavailable execution timing is not authenticated "
                "by the warmed Arena profile boundary"
            )
        execution_timing = {
            "abi": ARENA_UNAVAILABLE_EXECUTION_TIMING_ABI,
            "status": UNAVAILABLE_STATUS,
            "ratio_eligible": False,
            "raw_seconds_per_point": None,
            "sample_count": evaluator_sample_count,
            "native_profile_points_per_sample": raw_points_per_sample,
            "repetitions_per_sample": profile_repetitions,
            "batch_size": profile_batch_size,
            "sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            "profile_protocol": ARENA_PROFILE_PROTOCOL,
            "profile_sample_pass": ARENA_PROFILE_SAMPLE_PASS,
            "profile_boundary": ARENA_PROFILE_BOUNDARY,
            "borrowed_flat_input": True,
            "preallocated_output": True,
            "phase_timing_scope": ARENA_PHASE_TIMING_SCOPE,
            "evaluator_timing_available": False,
            "paired_with_headline": True,
            "identical_batch": True,
            "identical_repetitions": True,
            "execution_mode": environment["execution_mode"],
            "warmed_boundary_wall_seconds_per_point": float(warmed_wall),
            "arena_profile_evidence_sha256": digest_arena_profile_value(
                arena_profile_evidence
            ),
        }
    elif timing_status == "measured":
        time_source = environment.get("evaluator_time_source")
        compiled_direct_arena_active = environment.get(
            "compiled_direct_arena_active",
            False,
        )
        if (
            isinstance(raw_evaluator_time, bool)
            or not isinstance(raw_evaluator_time, (int, float))
            or not math.isfinite(float(raw_evaluator_time))
            or float(raw_evaluator_time) <= 0.0
            or isinstance(evaluator_time, bool)
            or not isinstance(evaluator_time, (int, float))
            or not math.isfinite(float(evaluator_time))
            or float(evaluator_time) != float(raw_evaluator_time)
            or ratio_eligible is not True
            or not isinstance(time_source, str)
            or not time_source
            or not isinstance(compiled_direct_arena_active, bool)
            or not isinstance(timing_sample_contract, str)
            or not timing_sample_contract
        ):
            raise RunnerError("benchmark measured execution timing is inconsistent")
        execution_timing = {
            "abi": MEASURED_EXECUTION_TIMING_ABI,
            "status": "measured",
            "ratio_eligible": True,
            "raw_seconds_per_point": float(raw_evaluator_time),
            "source": time_source,
            "compiled_direct_arena_active": compiled_direct_arena_active,
            "sample_count": evaluator_sample_count,
            "native_profile_points_per_sample": raw_points_per_sample,
            "sample_contract": timing_sample_contract,
        }
    else:
        raise RunnerError("benchmark has unsupported evaluator timing status")
    total_environment_fields = (
        "evaluator_total_time_raw_seconds_per_point",
        "evaluator_total_time_status",
        "evaluator_total_time_ratio_eligible",
        "evaluator_total_time_source",
        "evaluator_total_time_sample_contract",
        "evaluator_total_accumulated_seconds",
    )
    total_fields_present = tuple(
        field in environment for field in total_environment_fields
    )
    evaluator_total_timing: Mapping[str, object] | None = None
    if any(total_fields_present):
        if not all(total_fields_present):
            raise RunnerError(
                "benchmark has incomplete accumulated evaluator-total timing"
            )
        repetitions = environment.get("native_profile_repetitions_per_sample")
        batch_size = environment.get("native_profile_batch_size")
        measured_point_count = environment.get("measured_point_count")
        raw_total = environment["evaluator_total_time_raw_seconds_per_point"]
        evaluator_total_timing = {
            "abi": EVALUATOR_TOTAL_TIMING_ABI,
            "status": environment["evaluator_total_time_status"],
            "ratio_eligible": environment["evaluator_total_time_ratio_eligible"],
            "raw_seconds_per_point": raw_total,
            "source": environment["evaluator_total_time_source"],
            "execution_mode": environment.get("execution_mode"),
            "sample_contract": environment["evaluator_total_time_sample_contract"],
            "sample_count": benchmark.sample_count,
            "repetitions_per_sample": repetitions,
            "batch_size": batch_size,
            "points_per_sample": raw_points_per_sample,
            "measured_point_count": measured_point_count,
            "accumulated_seconds": environment["evaluator_total_accumulated_seconds"],
        }
        total_measurement = {
            "status": ResultStatus.OK.value,
            "wall_seconds_per_point": benchmark.wall_time_per_point,
            "sample_count": benchmark.sample_count,
            "provenance": {
                EVALUATOR_TOTAL_TIMING_KEY: evaluator_total_timing,
            },
        }
        result_total = getattr(
            benchmark,
            "evaluator_total_time_per_point",
            None,
        )
        if (
            evaluator_total_timing_record(total_measurement) is None
            or isinstance(result_total, bool)
            or not isinstance(result_total, (int, float))
            or not math.isfinite(float(result_total))
            or float(result_total) != float(raw_total)
        ):
            raise RunnerError(
                "benchmark accumulated evaluator-total timing is inconsistent"
            )
    target_runtime = float(benchmark.effective_config.target_runtime)
    if not math.isfinite(target_runtime) or target_runtime <= 0.0:
        raise RunnerError("benchmark has an invalid target timing duration")
    achieved_runtime = environment.get("elapsed_seconds")
    if (
        isinstance(achieved_runtime, bool)
        or not isinstance(achieved_runtime, (int, float))
        or not math.isfinite(float(achieved_runtime))
        or float(achieved_runtime) < 0.0
    ):
        raise RunnerError("benchmark did not report its achieved timing duration")
    command_evidence: dict[str, object] = {}
    if "report_command_path" in environment:
        command_path = environment.get("report_command_path")
        if not isinstance(command_path, str) or not command_path:
            raise RunnerError("benchmark has an invalid report command path")
        command_evidence["report_command_path"] = command_path
    if "report_public_cli_path" in environment:
        public_path = environment.get("report_public_cli_path")
        if public_path is not None and (
            not isinstance(public_path, str) or not public_path
        ):
            raise RunnerError("benchmark has an invalid public CLI command path")
        command_evidence["report_public_cli_path"] = public_path
    phase_evidence: dict[str, object] = {}
    raw_execution_mode = environment.get("execution_mode")
    benchmark_batch_size: int | None = None
    benchmark_warmup_runs: int | None = None
    if raw_execution_mode == ExecutionMode.ON_THE_FLY.value:
        raw_batch_size = environment.get("batch_size")
        effective_batch_size = getattr(
            benchmark.effective_config,
            "batch_size",
            None,
        )
        effective_warmup_runs = getattr(
            benchmark.effective_config,
            "warmup_runs",
            None,
        )
        if (
            isinstance(raw_batch_size, bool)
            or not isinstance(raw_batch_size, int)
            or raw_batch_size < 1
            or isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or effective_batch_size < 1
            or raw_batch_size != effective_batch_size
        ):
            raise RunnerError(
                "on-the-fly benchmark batch size does not match its effective "
                "configuration"
            )
        if (
            isinstance(effective_warmup_runs, bool)
            or not isinstance(effective_warmup_runs, int)
            or effective_warmup_runs < 0
        ):
            raise RunnerError(
                "on-the-fly benchmark has an invalid effective warm-up run count"
            )
        benchmark_batch_size = effective_batch_size
        benchmark_warmup_runs = effective_warmup_runs
    try:
        validate_profile_warmup_evidence(
            environment,
            execution_mode=(
                raw_execution_mode
                if isinstance(raw_execution_mode, str)
                else "<unknown>"
            ),
            expected_batch_size=benchmark_batch_size,
            expected_warmup_run_count=benchmark_warmup_runs,
        )
    except ValueError as error:
        raise RunnerError(str(error)) from error
    for field in (*sorted(OTF_COLD_WARMUP_FIELDS), *sorted(CONVENTIONAL_WARMUP_FIELDS)):
        if field in environment:
            phase_evidence[field] = _warmup_plain_value(environment[field])
    for field in (
        "calibration_elapsed_seconds",
        "calibration_outer_elapsed_seconds",
        "measurement_phase_elapsed_seconds",
        "profile_total_elapsed_seconds",
        "profile_attribution_elapsed_seconds",
        "evaluator_elapsed_seconds",
        "calibration_probe_seconds",
    ):
        if field not in environment:
            continue
        value = environment[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RunnerError(f"benchmark has invalid {field.replace('_', ' ')}")
        phase_evidence[field] = float(value)
    for field in (
        "calibration_block_count",
        "calibration_evaluation_count",
    ):
        if field not in environment:
            continue
        value = environment[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError(f"benchmark has invalid {field.replace('_', ' ')}")
        phase_evidence[field] = value
    if "measurement_phase_elapsed_seconds" not in phase_evidence:
        phase_evidence["measurement_phase_elapsed_seconds"] = float(achieved_runtime)
    return {
        "status": ResultStatus.OK.value,
        "wall_seconds_per_point": float(benchmark.wall_time_per_point),
        "execution_seconds_per_point": (
            None if evaluator_time is None else float(evaluator_time)
        ),
        "execution_timing": execution_timing,
        "evaluator_total_timing": evaluator_total_timing,
        "arena_profile_evidence": arena_profile_evidence,
        "matrix_element": matrix_element,
        "sample_count": int(benchmark.sample_count),
        "standard_error_seconds_per_point": float(uncertainty.standard_error),
        "relative_standard_error": float(uncertainty.relative_standard_error),
        "benchmark_evidence": {
            **command_evidence,
            **phase_evidence,
            "target_runtime_seconds": target_runtime,
            "achieved_runtime_seconds": float(achieved_runtime),
            "target_runtime_achieved": (
                float(achieved_runtime) >= 0.95 * target_runtime
            ),
            "completed_sample_count": environment.get("completed_sample_count"),
            "planned_sample_count": environment.get("planned_sample_count"),
            "repetitions_per_sample": environment.get("repetitions_per_sample"),
            "measured_point_count": environment.get("measured_point_count"),
            "interrupted": bool(environment.get("interrupted", False)),
        },
    }


def profile_runtime(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    benchmark_config: BenchmarkConfig,
    selector_contract: SelectorContract | None,
    progress: ProgressSink | None = None,
    profiling_deadline_monotonic: float | None = None,
) -> dict[str, object]:
    validate_runtime_contract(cell, runtime)
    if selector_contract is not None:
        validate_selector_contract(
            runtime,
            selector_contract,
            points,
            cell=cell,
        )
    selectors = _selector_kwargs(cell, selector_contract)
    values = (
        None
        if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
        else runtime.evaluate(points, **selectors)
    )
    if values is not None and not values:
        raise RunnerError("runtime returned no matrix elements")
    selected_config = replace(
        benchmark_config,
        helicity_ids=tuple(selectors["helicities"] or ()),
        color_flow_ids=tuple(selectors["color_flows"] or ()),
    )
    benchmark = _run_report_benchmark(
        runtime,
        points,
        execution_mode=cell.measurement.execution_mode,
        benchmark_config=selected_config,
        selectors=selectors,
        progress=progress,
        chunk_guard=profiling_chunk_guard(profiling_deadline_monotonic),
    )
    # The public OTF profiler owns the first evaluation so that its separately
    # reported cold warm-up is not destroyed by campaign-side validation.
    if values is None:
        values = runtime.evaluate(points, **selectors)
    if not values:
        raise RunnerError("runtime returned no matrix elements")
    result = _benchmark_measurement(
        benchmark,
        matrix_element=_real_nonnegative(values[0]),
    )
    result["resolved_sum_validation"] = resolved_sum_validation(
        runtime,
        points,
        cell=cell,
        selector_contract=selector_contract,
    )
    if result["resolved_sum_validation"]["status"] != ResultStatus.OK.value:
        result["status"] = ResultStatus.VALIDATION_FAILED.value
    return result


def runtime_validation_points(runtime: object) -> object:
    backend = getattr(runtime, "_backend", None)
    operation = getattr(backend, "validation_momenta", None)
    if not callable(operation):
        raise RunnerError("artifact does not retain deterministic validation momenta")
    points = operation()
    if points is None:
        raise RunnerError("artifact validation momenta are unavailable")
    return points


def _single_process_id(artifact_path: Path, fallback: str) -> str:
    manifest_path = artifact_path / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        processes = manifest["processes"]
        if isinstance(processes, list) and len(processes) == 1:
            identifier = processes[0]["id"]
            if isinstance(identifier, str) and identifier:
                return identifier
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        pass
    return fallback


def generate_artifact(
    cell: CellSpec,
    destination: Path,
    *,
    settings: RunnerSettings,
    repo_root: Path,
    prepared_model_path: Path | None = None,
    phase_reporter: WorkerPhaseReporter | None = None,
) -> GeneratedArtifact:
    """Generate one complete-coverage artifact and time process generation only."""

    from pyamplicol.api import Generator, ModelSource
    from pyamplicol.cli import CliInvocation, parse_cli
    from pyamplicol.cli.handlers import DefaultCliServices, dispatch
    from pyamplicol.config import Action, RunConfig
    from pyamplicol.config.resolver import config_to_dict, resolve_config
    from pyamplicol.reporting import NullProgressSink

    values = config_values(cell, settings, repo_root=repo_root)
    generation_values = values["generation"]
    assert isinstance(generation_values, dict)
    generation_values.update(
        {
            "output": os.fspath(destination),
            "mode": "replace",
        }
    )
    values["process"] = {"entries": [{"expression": cell.process}]}
    if prepared_model_path is not None:
        if not prepared_model_path.is_file():
            raise RunnerError(f"prepared model does not exist: {prepared_model_path}")
        model_values = values["model"]
        assert isinstance(model_values, dict)
        model_values["source"] = os.fspath(prepared_model_path)
    expected_resolution = resolve_config(
        values,
        action=Action.GENERATE,
        base_dir=repo_root,
    )
    invocation = parse_cli(_generation_cli_argv(cell, destination, values))
    if not isinstance(invocation, CliInvocation):
        raise RunnerError("public generate parser returned a non-command invocation")
    resolution = invocation.resolve()
    if (
        resolution.requested != expected_resolution.requested
        or resolution.effective != expected_resolution.effective
    ):
        raise RunnerError(
            "public generate arguments do not preserve report generation settings"
        )
    generation_phase = (
        nullcontext() if phase_reporter is None else phase_reporter.generation()
    )
    with generation_phase:
        # The process-tree watchdog owns the complete preparation+generation
        # interval, while the published generation timer below intentionally
        # continues to measure dispatch/generation only.
        assert cell.measurement.model is not None
        source_path = _model_source_path(repo_root, cell.measurement.model)
        source = (
            ModelSource.built_in_sm()
            if source_path is None
            else ModelSource.from_path(source_path)
        )
        model_started = time.perf_counter()
        prepared_execution = cell.measurement.execution_mode in {
            ExecutionMode.EAGER,
            ExecutionMode.RECURRENCE,
            ExecutionMode.ON_THE_FLY,
        }
        if prepared_model_path is not None:
            model = ModelSource.from_path(prepared_model_path)
            preparation_reused = True
        elif prepared_execution and cell.measurement.model is ModelKey.BUILTIN_SM:
            # Omitting the explicit model lets the generation service select the
            # validated wheel-owned built-in-SM JIT O2 prepared pack.
            model = None
            preparation_reused = True
        elif prepared_execution:
            raise RunnerError(
                f"{cell.measurement.model.value} "
                f"{cell.measurement.execution_mode.value} generation requires "
                "a prepared model path"
            )
        else:
            model = source.compile(
                cache_dir=settings.model_cache_dir,
                use_cache=True,
                require_supported=True,
            )
            preparation_reused = False
        model_seconds = time.perf_counter() - model_started
        generation_started = time.perf_counter()
        selected_progress = settings.progress or NullProgressSink()
        if prepared_execution:
            command_services = DefaultCliServices(resolution=resolution)
            generation_command_path = PUBLIC_CLI_COMMAND_PATH
        else:

            class _PreparedGenerationServices(DefaultCliServices):
                """Inject only the already compiled model into public dispatch.

                The public parser, typed resolution, progress path, and dispatch
                are retained.  This report-only handler exception is necessary
                to exclude model preparation from the process-generation timer.
                """

                def generate(
                    self,
                    config: RunConfig,
                    command_progress: ProgressSink,
                ) -> object:
                    return Generator(
                        resolution,
                        progress=command_progress,
                    ).generate(
                        cell.process,
                        destination,
                        model=model,
                        mode="replace",
                    )

            command_services = _PreparedGenerationServices(resolution=resolution)
            generation_command_path = PRECOMPILED_GENERATION_COMMAND_PATH

        dispatch(
            resolution.effective,
            command_services,
            selected_progress,
            dry_run=invocation.dry_run,
        )
        generation_seconds = time.perf_counter() - generation_started
    effective_config = _authenticated_effective_config(destination)
    process_id = _single_process_id(destination, cell.process)
    if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY:
        numerical_relation_correctness = None
        numerical_relation_fallback = None
    else:
        (
            numerical_relation_correctness,
            numerical_relation_fallback,
        ) = _artifact_numerical_relation_metadata(destination, process_id)
    return GeneratedArtifact(
        path=destination,
        process_id=process_id,
        generation_seconds=generation_seconds,
        model_preparation_seconds=model_seconds,
        model_preparation_reused=preparation_reused,
        requested_config=config_to_dict(resolution.requested),
        effective_config=effective_config,
        generation_command_path=generation_command_path,
        numerical_relation_correctness=numerical_relation_correctness,
        numerical_relation_fallback=numerical_relation_fallback,
    )


def provenance_payload() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "CONDITIONED_COMPARISON_ABI",
    "INDEPENDENT_RELATIVE_TOLERANCE",
    "MADGRAPH_RELATIVE_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "RESOLVED_COMPONENT_SCALE_ABI",
    "RESOLVED_SUM_VALIDATION_ABI",
    "GeneratedArtifact",
    "ProfilingTimeLimitError",
    "RunnerError",
    "RunnerSettings",
    "SelectorContract",
    "config_values",
    "derive_selector_contract",
    "generate_artifact",
    "point_digest",
    "pointwise_validation",
    "profile_runtime",
    "profiling_chunk_guard",
    "provenance_payload",
    "resolved_component_scale_evidence",
    "resolved_sum_validation",
    "runtime_identity_payload",
    "runtime_validation_points",
    "validate_artifact_contract",
    "validate_conditioned_comparison_record",
    "validate_resolved_sum_validation_record",
    "validate_runtime_contract",
    "validate_selector_contract",
]
