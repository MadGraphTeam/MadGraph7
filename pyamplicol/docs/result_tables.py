# SPDX-License-Identifier: 0BSD
"""Typed cache, table, and PDF services for the public performance report.

This module never generates physics artifacts on import or during a normal
render. Measurements are supplied explicitly by a caller after using the
public ``Generator`` and ``BenchmarkRunner`` APIs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as _dt
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol

if TYPE_CHECKING:
    pass


REPORT_VERSION = "0.1.0"
CACHE_SCHEMA_VERSION = 2
SPDX_LICENSE = "0BSD"
NA_STATUS = "not_available"
DEFAULT_ARTIFACT_ROOT = Path(".artifacts/performance-report")
DEFAULT_DEV_PYTHON = Path(".venv/bin/python")
REPORT_TEX_INPUTS = (
    "section_zgg_example.tex",
    "section_zgg_dag.tex",
    "section_ufo_support.tex",
)
DEFAULT_LIMIT_GIB = 800.0
DEFAULT_GENERATION_TIMEOUT_SECONDS = 3600.0
DEFAULT_WORKERS = 50
DEFAULT_PARALLEL_CELL_CORES = 1
DEFAULT_REPORT_TARGET_RUNTIME_SECONDS = 20.0
LEGACY_PROFILE_POLICY = "target_runtime_warmup_v1"
PYAMPLICOL_GENERATION_PROFILE_POLICY = "precompiled_model_before_generation_v1"
DEFAULT_LEGACY_PROFILE_WARMUP_POINTS = 100
DEFAULT_LEGACY_PROFILE_MIN_POINTS = 100
DEFAULT_LEGACY_PROFILE_MAX_POINTS = 100_000
LEGACY_LC_ALL_FLOW_GENERATION_SOURCE = "shared_generated_library_build"
ALLOW_PARALLEL_SYMBOLICA_ENV = "PYAMPLICOL_REPORT_ALLOW_PARALLEL_SYMBOLICA"
VALIDATION_RELATIVE_TOLERANCE = 1.0e-8
VALIDATION_ABSOLUTE_TOLERANCE = 1.0e-15
LEGACY_REFERENCE_COMPATIBLE_REVISIONS = frozenset(
    {
        "38937fc4a0a66ae14c55e77ba455de8c6170547b",
        "60443f327c2203cf92625da2bf0969c27e68a4ac",
        "754064d751224ec96c182d5f5d21fd6a11ad28f6",
    }
)
BUILTIN_SM_COMPATIBLE_COMPILED_MODEL_SCHEMAS = frozenset({8, 9})
_ACTIVE_WORKER_LOCK = threading.Lock()
_ACTIVE_WORKER_PROCESSES: set[subprocess.Popen[bytes]] = set()

REPORT_CONFIG_OVERRIDES: Mapping[str, object] = {
    "evaluator.backend": "jit",
    "evaluator.batch_size": 128,
    "evaluator.output_chunk_size": 128,
    "evaluator.optimization.horner_iterations": 10,
    "evaluator.optimization.cpe_iterations": None,
    "evaluator.optimization.max_horner_variables": 1000,
    "evaluator.optimization.max_common_pair_cache_entries": 5_000_000,
    "evaluator.optimization.max_common_pair_distance": 1000,
    "evaluator.jit.optimization_level": 3,
    "benchmark.target_runtime": DEFAULT_REPORT_TARGET_RUNTIME_SECONDS,
    "benchmark.batch_size": 128,
    "benchmark.warmup_runs": 2,
    "benchmark.minimum_samples": 5,
}


class CacheKind(StrEnum):
    PROCESS_MATRIX = "process_matrix"
    PERFORMANCE_LADDER = "performance_ladder"
    MODEL_LADDER = "model_ladder"


class ResultStatus(StrEnum):
    NOT_AVAILABLE = NA_STATUS
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    VALIDATION_FAILED = "validation_failed"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class BenchmarkStatisticsLike(Protocol):
    standard_deviation: float
    standard_error: float
    relative_standard_error: float


class BenchmarkResultLike(Protocol):
    requested_config: object
    effective_config: object
    sample_count: int
    wall_time_per_point: float
    evaluator_time_per_point: float | None
    uncertainty: BenchmarkStatisticsLike
    environment: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    """Serializable subset of the standalone ``BenchmarkResult`` contract."""

    sample_count: int
    wall_seconds_per_point: float
    evaluator_seconds_per_point: float | None
    standard_deviation_seconds_per_point: float
    standard_error_seconds_per_point: float
    relative_standard_error: float
    requested_config: Mapping[str, object]
    effective_config: Mapping[str, object]
    environment: Mapping[str, object]

    @classmethod
    def from_result(cls, result: BenchmarkResultLike) -> BenchmarkObservation:
        return cls(
            sample_count=result.sample_count,
            wall_seconds_per_point=result.wall_time_per_point,
            evaluator_seconds_per_point=result.evaluator_time_per_point,
            standard_deviation_seconds_per_point=(
                result.uncertainty.standard_deviation
            ),
            standard_error_seconds_per_point=result.uncertainty.standard_error,
            relative_standard_error=result.uncertainty.relative_standard_error,
            requested_config=_mapping_from_config(result.requested_config),
            effective_config=_mapping_from_config(result.effective_config),
            environment={
                str(key): _json_compatible(value)
                for key, value in result.environment.items()
            },
        )

    def as_cache_fields(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "wall_seconds_per_point": self.wall_seconds_per_point,
            "evaluator_seconds_per_point": self.evaluator_seconds_per_point,
            "standard_deviation_seconds_per_point": (
                self.standard_deviation_seconds_per_point
            ),
            "standard_error_seconds_per_point": (self.standard_error_seconds_per_point),
            "relative_standard_error": self.relative_standard_error,
            "requested_config": dict(self.requested_config),
            "effective_config": dict(self.effective_config),
            "environment": dict(self.environment),
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    profile: str
    label: str
    source_kind: Literal["built-in-sm", "json"]

    def as_json(self) -> dict[str, str]:
        return {
            "profile": self.profile,
            "label": self.label,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True, slots=True)
class ProcessFamily:
    identifier: int
    key: str
    label_tex: str
    initial_state: tuple[str, ...]
    base_final_state: tuple[str, ...]
    maximum_lc_n: int
    include_3qqbar: bool = False
    include_cc: bool = False
    include_resonance: bool = False

    @property
    def minimum_n(self) -> int:
        return len(self.base_final_state)

    def maximum_n(self, accuracy: str) -> int:
        return self.maximum_lc_n if accuracy == "lc" else 5

    def process(self, n_final: int) -> str | None:
        extra_gluons = n_final - self.minimum_n
        if extra_gluons < 0:
            return None
        final_state = (*self.base_final_state, *("g" for _ in range(extra_gluons)))
        return f"{' '.join(self.initial_state)} > {' '.join(final_state)}"

    def as_json(self, accuracy: str) -> dict[str, object]:
        return {
            "id": self.identifier,
            "key": self.key,
            "label_tex": self.label_tex,
            "initial_state": list(self.initial_state),
            "base_final_state": list(self.base_final_state),
            "minimum_n": self.minimum_n,
            "maximum_n": self.maximum_n(accuracy),
            "legacy_process_list_flags": self.legacy_process_list_flags(),
        }

    def legacy_process_list_flags(self) -> list[str]:
        flags: list[str] = []
        if self.include_3qqbar:
            flags.append("-3")
        if self.include_cc:
            flags.append("-cc")
        if self.include_resonance:
            flags.append("-res")
        return flags


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    dataset_id: str
    cache_name: str
    table_name: str
    title: str
    model: ModelSpec
    color_accuracy: Literal["lc", "nlc", "full"]
    multiplicities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VariantSpec:
    key: str
    label: str
    config_overrides: Mapping[str, object]

    def as_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "config_overrides": dict(self.config_overrides),
        }


@dataclass(frozen=True, slots=True)
class LadderSpec:
    dataset_id: str
    cache_name: str
    table_name: str
    title: str
    kind: Literal[CacheKind.PERFORMANCE_LADDER, CacheKind.MODEL_LADDER]
    model: ModelSpec
    multiplicities: tuple[int, ...]
    process_family: str
    final_particle: str
    variants: tuple[VariantSpec, ...] = ()

    def process(self, n_final: int) -> str:
        if self.dataset_id.startswith("z_"):
            final = ("z", *("g" for _ in range(n_final - 1)))
            return f"d d~ > {' '.join(final)}"
        final = " ".join(self.final_particle for _ in range(n_final))
        return f"scalar_0 scalar_0 > {final}"


@dataclass(frozen=True, slots=True)
class ReportPaths:
    docs_dir: Path

    @classmethod
    def default(cls) -> ReportPaths:
        return cls(Path(__file__).resolve().parent)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "docs_dir",
            self.docs_dir.expanduser().resolve(strict=False),
        )

    @property
    def results_dir(self) -> Path:
        return self.docs_dir / "results"

    @property
    def schema_path(self) -> Path:
        return self.results_dir / "report-cache.schema.json"

    @property
    def report_tex(self) -> Path:
        return self.docs_dir / "pyAmpliCol.tex"

    @property
    def report_pdf(self) -> Path:
        return self.docs_dir / "pyAmpliCol.pdf"


@dataclass(frozen=True, slots=True)
class CampaignCell:
    kind: Literal["matrix", "performance_ladder", "model_ladder"]
    cache_name: str
    dataset_id: str
    n_final: int
    process: str
    process_key: str | None = None
    variant: str | None = None
    priority: tuple[int, int, int, str] = (0, 0, 0, "")

    @property
    def cell_id(self) -> str:
        parts = [self.dataset_id, f"n{self.n_final}"]
        if self.process_key is not None:
            parts.append(self.process_key)
        if self.variant is not None:
            parts.append(self.variant)
        return "-".join(part.replace("_", "-") for part in parts)

    def as_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "cache_name": self.cache_name,
            "dataset_id": self.dataset_id,
            "n_final": self.n_final,
            "process": self.process,
            "process_key": self.process_key,
            "variant": self.variant,
            "cell_id": self.cell_id,
            "priority": list(self.priority),
        }


BUILTIN_SM = ModelSpec("built-in-sm", "Built-in Standard Model", "built-in-sm")
EXTERNAL_SM = ModelSpec(
    "external-sm",
    "UFO-SM",
    "json",
)
SCALAR_CONTACT = ModelSpec(
    "scalar-contact",
    "Massless scalar contact model",
    "json",
)
SCALAR_GRAVITY = ModelSpec(
    "scalar-gravity",
    "Scalar-gravity model",
    "json",
)


PROCESS_FAMILIES: tuple[ProcessFamily, ...] = (
    ProcessFamily(1, "dd_z_jets", r"$d\bar d\to Z+(n-1)g$", ("d", "d~"), ("z",), 9),
    ProcessFamily(
        2,
        "ud_w_jets",
        r"$u\bar d\to W^++(n-1)g$",
        ("u", "d~"),
        ("w+",),
        9,
        include_cc=True,
    ),
    ProcessFamily(
        3, "dd_epem_jets", r"$d\bar d\to e^+e^-+(n-2)g$", ("d", "d~"), ("e+", "e-"), 9
    ),
    ProcessFamily(
        4,
        "ud_epve_jets",
        r"$u\bar d\to e^+\nu_e+(n-2)g$",
        ("u", "d~"),
        ("e+", "ve"),
        9,
        include_cc=True,
    ),
    ProcessFamily(
        5, "dd_zz_jets", r"$d\bar d\to ZZ+(n-2)g$", ("d", "d~"), ("z", "z"), 9
    ),
    ProcessFamily(
        6, "gg_tt_jets", r"$gg\to t\bar t+(n-2)g$", ("g", "g"), ("t", "t~"), 8
    ),
    ProcessFamily(
        7, "dd_tt_jets", r"$d\bar d\to t\bar t+(n-2)g$", ("d", "d~"), ("t", "t~"), 9
    ),
    ProcessFamily(8, "gg_gluons", r"$gg\to gg+(n-2)g$", ("g", "g"), ("g", "g"), 8),
    ProcessFamily(
        9, "dd_zzz_jets", r"$d\bar d\to ZZZ+(n-3)g$", ("d", "d~"), ("z", "z", "z"), 9
    ),
    ProcessFamily(
        10,
        "dd_epemzh_jets",
        r"$d\bar d\to e^+e^-ZH+(n-4)g$",
        ("d", "d~"),
        ("e+", "e-", "z", "h"),
        9,
    ),
    ProcessFamily(
        11,
        "dd_ttzh_jets",
        r"$d\bar d\to t\bar t ZH+(n-4)g$",
        ("d", "d~"),
        ("t", "t~", "z", "h"),
        9,
    ),
    ProcessFamily(
        12,
        "dd_4l_jets",
        r"$d\bar d\to e^+e^-e^+e^-+(n-4)g$",
        ("d", "d~"),
        ("e+", "e-", "e+", "e-"),
        9,
    ),
    ProcessFamily(
        13,
        "dd_3q_lines",
        r"$d\bar d\to u\bar u\,s\bar s+(n-4)g$",
        ("d", "d~"),
        ("u", "u~", "s", "s~"),
        8,
        include_3qqbar=True,
    ),
    ProcessFamily(
        14,
        "dd_4q_lines",
        r"$d\bar d\to u\bar u\,s\bar s\,c\bar c+(n-6)g$",
        ("d", "d~"),
        ("u", "u~", "s", "s~", "c", "c~"),
        8,
        include_3qqbar=True,
    ),
)


def _process_family_by_key(key: str | None) -> ProcessFamily | None:
    for family in PROCESS_FAMILIES:
        if family.key == key:
            return family
    return None


def _matrix_spec(
    model_key: str,
    model: ModelSpec,
    accuracy: Literal["lc", "nlc", "full"],
) -> MatrixSpec:
    accuracy_label = {"lc": "LC", "nlc": "NLC", "full": "full-colour"}[accuracy]
    model_title = "Built-in SM" if model is BUILTIN_SM else "UFO-SM"
    multiplicities = tuple(range(1, 10 if accuracy == "lc" else 6))
    stem = f"matrix_{model_key}_{accuracy}"
    return MatrixSpec(
        dataset_id=stem,
        cache_name=f"{stem}.json",
        table_name=f"result_{stem}_table.tex",
        title=f"{model_title} {accuracy_label} process matrix",
        model=model,
        color_accuracy=accuracy,
        multiplicities=multiplicities,
    )


MATRIX_SPECS: tuple[MatrixSpec, ...] = tuple(
    _matrix_spec(model_key, model, accuracy)
    for model_key, model in (("builtin_sm", BUILTIN_SM), ("external_sm", EXTERNAL_SM))
    for accuracy in ("lc", "nlc", "full")
)

Z_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("reference", "Independent reference", {}),
    VariantSpec(
        "jit_o1",
        "JIT level 1",
        {"evaluator.backend": "jit", "evaluator.jit.optimization_level": 1},
    ),
    VariantSpec("asm_o3", "ASM O3", {"evaluator.backend": "asm"}),
    VariantSpec(
        "cpp_o3",
        "C++ O3",
        {"evaluator.backend": "cpp", "evaluator.cpp.optimization": "O3"},
    ),
    VariantSpec(
        "jit_o3",
        "JIT level 3",
        {"evaluator.backend": "jit", "evaluator.jit.optimization_level": 3},
    ),
)

LADDER_SPECS: tuple[LadderSpec, ...] = (
    LadderSpec(
        "z_builtin_sm",
        "z_builtin_sm.json",
        "result_z_builtin_sm_table.tex",
        (
            r"Built-in SM \texorpdfstring{$d\bar d\to Z+(n-1)g$}"
            r"{d dbar to Z plus (n-1)g} ladder"
        ),
        CacheKind.PERFORMANCE_LADDER,
        BUILTIN_SM,
        tuple(range(1, 10)),
        "d d~ > z + (n-1)g",
        "g",
        Z_VARIANTS,
    ),
    LadderSpec(
        "z_external_sm",
        "z_external_sm.json",
        "result_z_external_sm_table.tex",
        (
            r"UFO-SM \texorpdfstring{$d\bar d\to Z+(n-1)g$}"
            r"{d dbar to Z plus (n-1)g} ladder"
        ),
        CacheKind.PERFORMANCE_LADDER,
        EXTERNAL_SM,
        tuple(range(1, 7)),
        "d d~ > z + (n-1)g",
        "g",
        Z_VARIANTS,
    ),
    LadderSpec(
        "scalar_contact",
        "scalar_contact.json",
        "result_scalar_contact_table.tex",
        "Scalar-contact ladder",
        CacheKind.MODEL_LADDER,
        SCALAR_CONTACT,
        tuple(range(2, 9)),
        "scalar_0 scalar_0 > n*scalar_0",
        "scalar_0",
    ),
    LadderSpec(
        "scalar_gravity",
        "scalar_gravity.json",
        "result_scalar_gravity_table.tex",
        "Scalar-gravity ladder",
        CacheKind.MODEL_LADDER,
        SCALAR_GRAVITY,
        tuple(range(2, 5)),
        "scalar_0 scalar_0 > n*graviton",
        "graviton",
    ),
)

TABLE_INPUTS: tuple[str, ...] = tuple(
    spec.table_name for spec in (*MATRIX_SPECS, *LADDER_SPECS)
)


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(entry) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(entry) for entry in value]
    raise TypeError(f"cannot serialize report value of type {type(value).__name__}")


def _mapping_from_config(value: object) -> Mapping[str, object]:
    serialized = _json_compatible(value)
    if not isinstance(serialized, Mapping):
        raise TypeError("benchmark configuration must serialize to an object")
    return {str(key): entry for key, entry in serialized.items()}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _empty_measurement() -> dict[str, object]:
    return {
        "status": NA_STATUS,
        "generation_seconds": None,
        "sample_count": None,
        "wall_seconds_per_point": None,
        "evaluator_seconds_per_point": None,
        "standard_deviation_seconds_per_point": None,
        "standard_error_seconds_per_point": None,
        "relative_standard_error": None,
        "matrix_element": None,
        "requested_config": None,
        "effective_config": None,
        "environment": {},
        "failure_kind": None,
        "failure_message": None,
        "artifact_path": None,
        "log_path": None,
        "manifest_path": None,
        "peak_rss_gib": None,
        "limit_gib": None,
        "timeout_seconds": None,
        "command": None,
        "metadata": {},
    }


def _empty_validation() -> dict[str, object]:
    return {
        "status": NA_STATUS,
        "reference_matrix_element": None,
        "pyamplicol_matrix_element": None,
        "absolute_difference": None,
        "relative_difference": None,
        "all_flow_status": NA_STATUS,
        "all_flow_reference_matrix_element": None,
        "all_flow_pyamplicol_matrix_element": None,
        "all_flow_absolute_difference": None,
        "all_flow_relative_difference": None,
        "relative_tolerance": VALIDATION_RELATIVE_TOLERANCE,
        "absolute_tolerance": VALIDATION_ABSOLUTE_TOLERANCE,
        "point_source": None,
        "message": None,
    }


def _empty_parameter_alignment() -> dict[str, object]:
    return {
        "status": NA_STATUS,
        "built_in_sm_source": "built-in-sm",
        "ufo_sm_source": None,
        "snapshot_path": None,
        "message": None,
    }


def _safe_divide(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0.0:
        return None
    return top / bottom


def _optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _measurement_status(value: Mapping[str, object]) -> str:
    return str(value.get("status", NA_STATUS))


def _measurement_ok(value: Mapping[str, object]) -> bool:
    return _measurement_status(value) == ResultStatus.OK.value


def _failure_measurement(
    status: ResultStatus | str,
    message: str,
    *,
    failure_kind: str | None = None,
    artifact_path: Path | str | None = None,
    log_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    limit_gib: float | None = None,
    timeout_seconds: float | None = None,
    command: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    measurement = _empty_measurement()
    measurement.update(
        {
            "status": str(status),
            "failure_kind": failure_kind or str(status),
            "failure_message": str(message),
            "artifact_path": (
                None if artifact_path is None else os.fspath(artifact_path)
            ),
            "log_path": None if log_path is None else os.fspath(log_path),
            "manifest_path": (
                None if manifest_path is None else os.fspath(manifest_path)
            ),
            "limit_gib": limit_gib,
            "timeout_seconds": timeout_seconds,
            "command": None if command is None else list(command),
            "metadata": dict(metadata or {}),
        }
    )
    return measurement


def _common_payload(
    *, kind: CacheKind, dataset_id: str, model: ModelSpec
) -> dict[str, object]:
    return {
        "$schema": "report-cache.schema.json",
        "schema_version": CACHE_SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
        "spdx_license_identifier": SPDX_LICENSE,
        "kind": kind.value,
        "dataset_id": dataset_id,
        "created_by": "docs/result_tables.py",
        "updated_at": None,
        "model": model.as_json(),
        "multiplicity_convention": "final_state",
    }


def _benchmark_contract() -> dict[str, object]:
    return {
        "observable": "summed_matrix_element",
        "runtime_api": "Runtime.evaluate",
        "target_runtime_seconds": DEFAULT_REPORT_TARGET_RUNTIME_SECONDS,
        "batch_size": 128,
        "warmup_runs": 2,
        "minimum_samples": 5,
        "config_overrides": dict(REPORT_CONFIG_OVERRIDES),
        "legacy_profile": _legacy_profile_requested_config(
            DEFAULT_REPORT_TARGET_RUNTIME_SECONDS
        ),
    }


def build_matrix_cache(spec: MatrixSpec) -> dict[str, object]:
    payload = _common_payload(
        kind=CacheKind.PROCESS_MATRIX,
        dataset_id=spec.dataset_id,
        model=spec.model,
    )
    entries: list[dict[str, object]] = []
    for family in PROCESS_FAMILIES:
        maximum_n = family.maximum_n(spec.color_accuracy)
        for n_final in spec.multiplicities:
            applicable = family.minimum_n <= n_final <= maximum_n
            entries.append(
                {
                    "process_key": family.key,
                    "n_final": n_final,
                    "process": family.process(n_final),
                    "applicable": applicable,
                    "status": NA_STATUS,
                    "legacy_amplicol": _empty_measurement(),
                    "pyamplicol_jit_o3": _empty_measurement(),
                    "reference": _empty_measurement(),
                    "pyamplicol": _empty_measurement(),
                    "generation_multiplier": None,
                    "runtime_multiplier": None,
                    "pointwise_validation": _empty_validation(),
                    "parameter_alignment": _empty_parameter_alignment(),
                    "relative_difference": None,
                }
            )
    payload.update(
        {
            "color_accuracy": spec.color_accuracy,
            "multiplicities": list(spec.multiplicities),
            "process_families": [
                family.as_json(spec.color_accuracy) for family in PROCESS_FAMILIES
            ],
            "benchmark_contract": _benchmark_contract(),
            "entries": entries,
        }
    )
    validate_cache(payload)
    return payload


def build_ladder_cache(spec: LadderSpec) -> dict[str, object]:
    payload = _common_payload(
        kind=CacheKind(spec.kind),
        dataset_id=spec.dataset_id,
        model=spec.model,
    )
    entries: list[dict[str, object]] = []
    if spec.kind == CacheKind.PERFORMANCE_LADDER:
        for n_final in spec.multiplicities:
            for variant in spec.variants:
                entries.append(
                    {
                        "n_final": n_final,
                        "process": spec.process(n_final),
                        "variant": variant.key,
                        "status": NA_STATUS,
                        "measurement": _empty_measurement(),
                    }
                )
    else:
        for n_final in spec.multiplicities:
            entries.append(
                {
                    "n_final": n_final,
                    "process": spec.process(n_final),
                    "status": NA_STATUS,
                    "measurement": _empty_measurement(),
                    "high_precision_matrix_element": None,
                    "relative_difference": None,
                }
            )
    payload.update(
        {
            "multiplicities": list(spec.multiplicities),
            "process_family": spec.process_family,
            "color_accuracy": "lc",
            "benchmark_contract": _benchmark_contract(),
            "entries": entries,
        }
    )
    if spec.variants:
        payload["variants"] = [variant.as_json() for variant in spec.variants]
    validate_cache(payload)
    return payload


def build_reset_caches() -> dict[str, dict[str, object]]:
    caches = {spec.cache_name: build_matrix_cache(spec) for spec in MATRIX_SPECS}
    caches.update({spec.cache_name: build_ladder_cache(spec) for spec in LADDER_SPECS})
    return caches


def _normalize_measurement(value: object) -> dict[str, object]:
    measurement = dict(_empty_measurement())
    if isinstance(value, Mapping):
        measurement.update(
            {str(key): _json_compatible(entry) for key, entry in value.items()}
        )
    return measurement


def _normalize_validation(value: object) -> dict[str, object]:
    payload = dict(_empty_validation())
    if isinstance(value, Mapping):
        payload.update(
            {str(key): _json_compatible(entry) for key, entry in value.items()}
        )
    return payload


def _normalize_parameter_alignment(value: object) -> dict[str, object]:
    payload = dict(_empty_parameter_alignment())
    if isinstance(value, Mapping):
        payload.update(
            {str(key): _json_compatible(entry) for key, entry in value.items()}
        )
    return payload


def _refresh_matrix_derived_fields(entry: dict[str, object]) -> None:
    legacy = _normalize_measurement(
        entry.get("legacy_amplicol", entry.get("reference"))
    )
    pyamplicol = _normalize_measurement(
        entry.get("pyamplicol_jit_o3", entry.get("pyamplicol"))
    )
    entry["legacy_amplicol"] = legacy
    entry["pyamplicol_jit_o3"] = pyamplicol
    # Compatibility aliases retained for older report tooling.
    entry["reference"] = dict(legacy)
    entry["pyamplicol"] = dict(pyamplicol)
    entry["generation_multiplier"] = _safe_divide(
        legacy.get("generation_seconds"),
        pyamplicol.get("generation_seconds"),
    )
    entry["runtime_multiplier"] = _safe_divide(
        legacy.get("wall_seconds_per_point"),
        pyamplicol.get("wall_seconds_per_point"),
    )
    validation = _normalize_validation(entry.get("pointwise_validation"))
    entry["pointwise_validation"] = validation
    entry["parameter_alignment"] = _normalize_parameter_alignment(
        entry.get("parameter_alignment")
    )
    relative_difference = validation.get("relative_difference")
    entry["relative_difference"] = relative_difference
    if not bool(entry.get("applicable", False)):
        entry["status"] = NA_STATUS
        return
    statuses = {
        _measurement_status(legacy),
        _measurement_status(pyamplicol),
        str(validation.get("status", NA_STATUS)),
    }
    if ResultStatus.VALIDATION_FAILED.value in statuses:
        entry["status"] = ResultStatus.VALIDATION_FAILED.value
    elif ResultStatus.MEMORY_LIMIT.value in statuses:
        entry["status"] = ResultStatus.MEMORY_LIMIT.value
    elif ResultStatus.TIMEOUT.value in statuses:
        entry["status"] = ResultStatus.TIMEOUT.value
    elif ResultStatus.ERROR.value in statuses or ResultStatus.FAILED.value in statuses:
        entry["status"] = ResultStatus.ERROR.value
    elif ResultStatus.UNSUPPORTED.value in statuses:
        entry["status"] = ResultStatus.UNSUPPORTED.value
    elif _measurement_ok(legacy) and _measurement_ok(pyamplicol):
        entry["status"] = ResultStatus.OK.value
    elif statuses == {NA_STATUS}:
        entry["status"] = NA_STATUS
    else:
        entry["status"] = str(entry.get("status", NA_STATUS))


def normalize_cache_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a schema-current cache payload, migrating old checked-in caches."""

    normalized = {str(key): _json_compatible(value) for key, value in payload.items()}
    normalized["schema_version"] = CACHE_SCHEMA_VERSION
    if normalized.get("kind") in {member.value for member in CacheKind}:
        normalized["benchmark_contract"] = _benchmark_contract()
    entries = normalized.get("entries")
    if not isinstance(entries, list):
        return normalized
    if normalized.get("kind") == CacheKind.PROCESS_MATRIX.value:
        families = normalized.get("process_families")
        if isinstance(families, list):
            for raw_family in families:
                if not isinstance(raw_family, dict):
                    continue
                family = _process_family_by_key(str(raw_family.get("key")))
                raw_family.setdefault(
                    "legacy_process_list_flags",
                    [] if family is None else family.legacy_process_list_flags(),
                )
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            _refresh_matrix_derived_fields(raw_entry)
    else:
        if normalized.get("kind") == CacheKind.PERFORMANCE_LADDER.value:
            dataset_id = str(normalized.get("dataset_id"))
            for spec in LADDER_SPECS:
                if (
                    spec.dataset_id == dataset_id
                    and spec.kind == CacheKind.PERFORMANCE_LADDER
                ):
                    normalized["variants"] = [
                        variant.as_json() for variant in spec.variants
                    ]
                    break
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            raw_entry["measurement"] = _normalize_measurement(
                raw_entry.get("measurement")
            )
    return normalized


def _require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], required: Iterable[str], context: str
) -> None:
    missing = set(required).difference(value)
    if missing:
        raise ValueError(f"{context} is missing keys: {', '.join(sorted(missing))}")


def _validate_measurement(value: object, context: str) -> None:
    measurement = _require_object(value, context)
    numeric_fields = (
        "generation_seconds",
        "wall_seconds_per_point",
        "evaluator_seconds_per_point",
        "standard_deviation_seconds_per_point",
        "standard_error_seconds_per_point",
        "relative_standard_error",
        "matrix_element",
        "peak_rss_gib",
        "limit_gib",
        "timeout_seconds",
    )
    _require_exact_keys(
        measurement,
        (
            "status",
            "sample_count",
            "requested_config",
            "effective_config",
            "environment",
            "failure_kind",
            "failure_message",
            "artifact_path",
            "log_path",
            "manifest_path",
            "command",
            "metadata",
            *numeric_fields,
        ),
        context,
    )
    status = measurement["status"]
    if status not in {member.value for member in ResultStatus}:
        raise ValueError(f"{context}.status is invalid: {status!r}")
    sample_count = measurement["sample_count"]
    if sample_count is not None and (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError(f"{context}.sample_count must be positive or null")
    for name in numeric_fields:
        field_value = measurement[name]
        if field_value is not None and (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
            or float(field_value) < 0.0
        ):
            raise ValueError(f"{context}.{name} must be non-negative or null")
    if status == NA_STATUS:
        if sample_count is not None or any(
            measurement[name] is not None for name in numeric_fields
        ):
            raise ValueError(f"{context} N/A measurement contains measured values")
        if measurement["requested_config"] is not None:
            raise ValueError(f"{context} N/A requested_config must be null")
        if measurement["effective_config"] is not None:
            raise ValueError(f"{context} N/A effective_config must be null")
        if measurement["environment"] != {}:
            raise ValueError(f"{context} N/A environment must be empty")
    for name in (
        "failure_kind",
        "failure_message",
        "artifact_path",
        "log_path",
        "manifest_path",
    ):
        field_value = measurement[name]
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"{context}.{name} must be a string or null")
    command = measurement["command"]
    if command is not None and not (
        isinstance(command, list)
        and all(isinstance(part, str) and part for part in command)
    ):
        raise TypeError(f"{context}.command must be a list of strings or null")
    if not isinstance(measurement["metadata"], Mapping):
        raise TypeError(f"{context}.metadata must be an object")


def validate_cache(payload: Mapping[str, object]) -> None:
    _require_exact_keys(
        payload,
        (
            "$schema",
            "schema_version",
            "report_version",
            "spdx_license_identifier",
            "kind",
            "dataset_id",
            "created_by",
            "updated_at",
            "model",
            "multiplicity_convention",
            "multiplicities",
            "entries",
        ),
        "cache",
    )
    if payload["$schema"] != "report-cache.schema.json":
        raise ValueError("cache uses an unexpected schema reference")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema version is not supported")
    if payload["report_version"] != REPORT_VERSION:
        raise ValueError("cache report version is not supported")
    if payload["spdx_license_identifier"] != SPDX_LICENSE:
        raise ValueError("cache SPDX identifier must be 0BSD")
    try:
        kind = CacheKind(str(payload["kind"]))
    except ValueError as error:
        raise ValueError(f"unknown cache kind {payload['kind']!r}") from error
    multiplicities = payload["multiplicities"]
    if not isinstance(multiplicities, list) or not multiplicities:
        raise ValueError("cache multiplicities must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in multiplicities
    ):
        raise ValueError("cache multiplicities must be positive integers")
    if multiplicities != sorted(set(multiplicities)):
        raise ValueError("cache multiplicities must be sorted and unique")
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise TypeError("cache entries must be a list")
    if kind == CacheKind.PROCESS_MATRIX:
        _validate_matrix_cache(payload, entries)
    else:
        _validate_ladder_cache(payload, entries, kind)


def _validate_optional_nonnegative_number(
    value: object, context: str, *, allow_zero: bool = True
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{context} must be {qualifier} or null")
    number = float(value)
    minimum_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{context} must be {qualifier} or null")


def _validate_pointwise_validation(value: object, context: str) -> None:
    payload = _require_object(value, context)
    _require_exact_keys(
        payload,
        (
            "status",
            "reference_matrix_element",
            "pyamplicol_matrix_element",
            "absolute_difference",
            "relative_difference",
            "all_flow_status",
            "all_flow_reference_matrix_element",
            "all_flow_pyamplicol_matrix_element",
            "all_flow_absolute_difference",
            "all_flow_relative_difference",
            "relative_tolerance",
            "absolute_tolerance",
            "point_source",
            "message",
        ),
        context,
    )
    if payload["status"] not in {member.value for member in ResultStatus}:
        raise ValueError(f"{context}.status is invalid")
    if payload["all_flow_status"] not in {member.value for member in ResultStatus}:
        raise ValueError(f"{context}.all_flow_status is invalid")
    for field_name in (
        "reference_matrix_element",
        "pyamplicol_matrix_element",
        "absolute_difference",
        "relative_difference",
        "all_flow_reference_matrix_element",
        "all_flow_pyamplicol_matrix_element",
        "all_flow_absolute_difference",
        "all_flow_relative_difference",
        "relative_tolerance",
        "absolute_tolerance",
    ):
        _validate_optional_nonnegative_number(
            payload[field_name], f"{context}.{field_name}"
        )
    for field_name in ("point_source", "message"):
        field_value = payload[field_name]
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"{context}.{field_name} must be a string or null")


def _validate_parameter_alignment(value: object, context: str) -> None:
    payload = _require_object(value, context)
    _require_exact_keys(
        payload,
        (
            "status",
            "built_in_sm_source",
            "ufo_sm_source",
            "snapshot_path",
            "message",
        ),
        context,
    )
    if payload["status"] not in {member.value for member in ResultStatus}:
        raise ValueError(f"{context}.status is invalid")
    for field_name in (
        "built_in_sm_source",
        "ufo_sm_source",
        "snapshot_path",
        "message",
    ):
        field_value = payload[field_name]
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"{context}.{field_name} must be a string or null")


def _validate_matrix_cache(
    payload: Mapping[str, object], entries: list[object]
) -> None:
    _require_exact_keys(
        payload,
        ("color_accuracy", "process_families", "benchmark_contract"),
        "process matrix",
    )
    if payload["color_accuracy"] not in ("lc", "nlc", "full"):
        raise ValueError("process matrix color_accuracy is invalid")
    families = payload["process_families"]
    multiplicities = payload["multiplicities"]
    assert isinstance(multiplicities, list)
    if not isinstance(families, list) or not families:
        raise ValueError("process matrix families must be a non-empty list")
    family_keys: set[str] = set()
    family_bounds: dict[str, tuple[int, int]] = {}
    for index, raw_family in enumerate(families):
        family = _require_object(raw_family, f"process_families[{index}]")
        _require_exact_keys(
            family,
            (
                "id",
                "key",
                "label_tex",
                "initial_state",
                "base_final_state",
                "minimum_n",
                "maximum_n",
                "legacy_process_list_flags",
            ),
            f"process_families[{index}]",
        )
        key = family["key"]
        if not isinstance(key, str) or not key or key in family_keys:
            raise ValueError("process family keys must be non-empty and unique")
        minimum_n = family["minimum_n"]
        maximum_n = family["maximum_n"]
        if (
            isinstance(minimum_n, bool)
            or not isinstance(minimum_n, int)
            or isinstance(maximum_n, bool)
            or not isinstance(maximum_n, int)
            or minimum_n < 1
            or maximum_n < 1
        ):
            raise ValueError("process family multiplicity bounds are invalid")
        family_keys.add(key)
        family_bounds[key] = (minimum_n, maximum_n)
        flags = family["legacy_process_list_flags"]
        if not isinstance(flags, list) or any(
            not isinstance(flag, str) or not flag.startswith("-") for flag in flags
        ):
            raise TypeError(
                f"process_families[{index}].legacy_process_list_flags "
                "must be a list of option strings"
            )
    expected = {(key, n_final) for key in family_keys for n_final in multiplicities}
    actual: set[tuple[str, int]] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(raw_entry, f"entries[{index}]")
        _require_exact_keys(
            entry,
            (
                "process_key",
                "n_final",
                "process",
                "applicable",
                "status",
                "legacy_amplicol",
                "pyamplicol_jit_o3",
                "reference",
                "pyamplicol",
                "generation_multiplier",
                "runtime_multiplier",
                "pointwise_validation",
                "parameter_alignment",
                "relative_difference",
            ),
            f"entries[{index}]",
        )
        key = entry["process_key"]
        n_final = entry["n_final"]
        if not isinstance(key, str) or key not in family_keys:
            raise ValueError(f"entries[{index}] has an unknown process key")
        if isinstance(n_final, bool) or not isinstance(n_final, int):
            raise TypeError(f"entries[{index}].n_final must be an integer")
        cell_key = (key, n_final)
        if cell_key in actual:
            raise ValueError(f"duplicate process matrix entry {cell_key!r}")
        actual.add(cell_key)
        if not isinstance(entry["applicable"], bool):
            raise TypeError(f"entries[{index}].applicable must be a boolean")
        minimum_n, maximum_n = family_bounds[key]
        expected_applicability = minimum_n <= n_final <= maximum_n
        if entry["applicable"] != expected_applicability:
            raise ValueError(f"entries[{index}].applicable contradicts family bounds")
        process = entry["process"]
        if n_final < minimum_n:
            if process is not None:
                raise ValueError(
                    f"entries[{index}].process must be null below minimum_n"
                )
        elif not isinstance(process, str) or not process:
            raise ValueError(f"entries[{index}].process must be a non-empty string")
        if entry["status"] not in {member.value for member in ResultStatus}:
            raise ValueError(f"entries[{index}].status is invalid")
        _validate_measurement(
            entry["legacy_amplicol"], f"entries[{index}].legacy_amplicol"
        )
        _validate_measurement(
            entry["pyamplicol_jit_o3"], f"entries[{index}].pyamplicol_jit_o3"
        )
        _validate_measurement(entry["reference"], f"entries[{index}].reference")
        _validate_measurement(entry["pyamplicol"], f"entries[{index}].pyamplicol")
        for field_name in (
            "generation_multiplier",
            "runtime_multiplier",
            "relative_difference",
        ):
            field_value = entry[field_name]
            if field_value is not None and (
                isinstance(field_value, bool)
                or not isinstance(field_value, (int, float))
                or not math.isfinite(float(field_value))
                or float(field_value) < 0.0
            ):
                raise ValueError(
                    f"entries[{index}].{field_name} must be non-negative or null"
                )
        _validate_pointwise_validation(
            entry["pointwise_validation"], f"entries[{index}].pointwise_validation"
        )
        _validate_parameter_alignment(
            entry["parameter_alignment"], f"entries[{index}].parameter_alignment"
        )
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            f"process matrix is incomplete; missing={missing}, extra={extra}"
        )


def _validate_ladder_cache(
    payload: Mapping[str, object], entries: list[object], kind: CacheKind
) -> None:
    _require_exact_keys(
        payload,
        ("process_family", "color_accuracy", "benchmark_contract"),
        "ladder",
    )
    multiplicities = payload["multiplicities"]
    assert isinstance(multiplicities, list)
    variants: list[str] = []
    if kind == CacheKind.PERFORMANCE_LADDER:
        raw_variants = payload.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError("performance ladder variants must be a non-empty list")
        for index, raw_variant in enumerate(raw_variants):
            variant = _require_object(raw_variant, f"variants[{index}]")
            _require_exact_keys(
                variant,
                ("key", "label", "config_overrides"),
                f"variants[{index}]",
            )
            key = variant["key"]
            if not isinstance(key, str) or not key or key in variants:
                raise ValueError("ladder variant keys must be non-empty and unique")
            variants.append(key)
        expected: set[tuple[int, str | None]] = {
            (n_final, variant) for n_final in multiplicities for variant in variants
        }
    else:
        expected = {(n_final, None) for n_final in multiplicities}
    actual: set[tuple[int, str | None]] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(raw_entry, f"entries[{index}]")
        _require_exact_keys(
            entry,
            ("n_final", "process", "status", "measurement"),
            f"entries[{index}]",
        )
        n_final = entry["n_final"]
        if isinstance(n_final, bool) or not isinstance(n_final, int):
            raise TypeError(f"entries[{index}].n_final must be an integer")
        variant_value = entry.get("variant")
        variant = str(variant_value) if variant_value is not None else None
        key = (n_final, variant)
        if key in actual:
            raise ValueError(f"duplicate ladder entry {key!r}")
        actual.add(key)
        if entry["status"] not in {member.value for member in ResultStatus}:
            raise ValueError(f"entries[{index}].status is invalid")
        _validate_measurement(entry["measurement"], f"entries[{index}].measurement")
        if kind == CacheKind.MODEL_LADDER:
            _require_exact_keys(
                entry,
                ("high_precision_matrix_element", "relative_difference"),
                f"entries[{index}]",
            )
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"ladder is incomplete; missing={missing}, extra={extra}")


def schema_document() -> dict[str, object]:
    """Return the formal JSON Schema shipped beside every cache."""

    nullable_number = {"type": ["number", "null"], "minimum": 0}
    measurement = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "generation_seconds",
            "sample_count",
            "wall_seconds_per_point",
            "evaluator_seconds_per_point",
            "standard_deviation_seconds_per_point",
            "standard_error_seconds_per_point",
            "relative_standard_error",
            "matrix_element",
            "requested_config",
            "effective_config",
            "environment",
            "failure_kind",
            "failure_message",
            "artifact_path",
            "log_path",
            "manifest_path",
            "peak_rss_gib",
            "limit_gib",
            "timeout_seconds",
            "command",
            "metadata",
        ],
        "properties": {
            "status": {"enum": [member.value for member in ResultStatus]},
            "generation_seconds": nullable_number,
            "sample_count": {"type": ["integer", "null"], "minimum": 1},
            "wall_seconds_per_point": nullable_number,
            "evaluator_seconds_per_point": nullable_number,
            "standard_deviation_seconds_per_point": nullable_number,
            "standard_error_seconds_per_point": nullable_number,
            "relative_standard_error": nullable_number,
            "matrix_element": nullable_number,
            "requested_config": {"type": ["object", "null"]},
            "effective_config": {"type": ["object", "null"]},
            "environment": {"type": "object"},
            "failure_kind": {"type": ["string", "null"]},
            "failure_message": {"type": ["string", "null"]},
            "artifact_path": {"type": ["string", "null"]},
            "log_path": {"type": ["string", "null"]},
            "manifest_path": {"type": ["string", "null"]},
            "peak_rss_gib": nullable_number,
            "limit_gib": nullable_number,
            "timeout_seconds": nullable_number,
            "command": {
                "type": ["array", "null"],
                "items": {"type": "string", "minLength": 1},
            },
            "metadata": {"type": "object"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$comment": "SPDX-License-Identifier: 0BSD",
        "$id": "report-cache.schema.json",
        "title": "pyAmpliCol 0.1.0 performance report cache",
        "type": "object",
        "required": [
            "$schema",
            "schema_version",
            "report_version",
            "spdx_license_identifier",
            "kind",
            "dataset_id",
            "created_by",
            "updated_at",
            "model",
            "multiplicity_convention",
            "multiplicities",
            "entries",
        ],
        "properties": {
            "$schema": {"const": "report-cache.schema.json"},
            "schema_version": {"const": CACHE_SCHEMA_VERSION},
            "report_version": {"const": REPORT_VERSION},
            "spdx_license_identifier": {"const": SPDX_LICENSE},
            "kind": {"enum": [member.value for member in CacheKind]},
            "dataset_id": {"type": "string", "minLength": 1},
            "created_by": {"const": "docs/result_tables.py"},
            "updated_at": {"type": ["string", "null"], "format": "date-time"},
            "model": {
                "type": "object",
                "additionalProperties": False,
                "required": ["profile", "label", "source_kind"],
                "properties": {
                    "profile": {"type": "string", "minLength": 1},
                    "label": {"type": "string", "minLength": 1},
                    "source_kind": {"enum": ["built-in-sm", "json"]},
                },
            },
            "multiplicity_convention": {"const": "final_state"},
            "multiplicities": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 1},
            },
            "entries": {"type": "array"},
            "color_accuracy": {"enum": ["lc", "nlc", "full"]},
            "process_families": {"type": "array"},
            "process_family": {"type": "string"},
            "benchmark_contract": {"type": "object"},
            "variants": {"type": "array"},
        },
        "$defs": {"measurement": measurement},
        "allOf": [
            {
                "if": {"properties": {"kind": {"const": "process_matrix"}}},
                "then": {
                    "required": [
                        "color_accuracy",
                        "process_families",
                        "benchmark_contract",
                    ]
                },
            },
            {
                "if": {"properties": {"kind": {"const": "performance_ladder"}}},
                "then": {
                    "required": [
                        "color_accuracy",
                        "process_family",
                        "benchmark_contract",
                        "variants",
                    ]
                },
            },
            {
                "if": {"properties": {"kind": {"const": "model_ladder"}}},
                "then": {
                    "required": [
                        "color_accuracy",
                        "process_family",
                        "benchmark_contract",
                    ]
                },
            },
        ],
    }


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_number(value: object, *, digits: int = 3) -> str:
    if value is None:
        return r"\ReportNA"
    number = float(value)
    return rf"\num{{{number:.{digits}g}}}"


def _status_label(status: object, *, limit_gib: object = None) -> str:
    value = str(status)
    if value == ResultStatus.OK.value:
        return r"\textcolor{ReportGreen}{\textsc{ok}}"
    if value == NA_STATUS:
        return r"\ReportNA"
    if value == ResultStatus.TIMEOUT.value:
        return r"\ReportStatus{t/o}"
    if value == ResultStatus.MEMORY_LIMIT.value:
        suffix = "" if limit_gib is None else f">{float(limit_gib):.0f}G"
        return rf"\ReportStatus{{RAM{suffix}}}"
    if value == ResultStatus.VALIDATION_FAILED.value:
        return r"\ReportStatus{VALIDATION FAILED}"
    if value == ResultStatus.UNSUPPORTED.value:
        return r"\ReportStatus{UNSUPPORTED}"
    return r"\ReportStatus{ERROR}"


def _measurement_label(measurement: Mapping[str, object]) -> str:
    return _status_label(
        measurement.get("status", NA_STATUS),
        limit_gib=measurement.get("limit_gib"),
    )


def _multiplier_latex(value: object) -> str:
    if value is None:
        return r"\ReportNA"
    number = float(value)
    if number >= 1.0:
        color = "ReportGreen"
    elif number >= 0.5:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return rf"\textcolor{{{color}}}{{\num{{{number:.3g}}}$\times$}}"


def _measurement_cell(measurement: Mapping[str, object]) -> str:
    if measurement["status"] != ResultStatus.OK.value:
        return _measurement_label(measurement)
    return (
        r"\ReportMetricCell"
        f"{{{_format_number(measurement['generation_seconds'])}}}"
        f"{{{_format_number(measurement['wall_seconds_per_point'])}}}"
    )


def _z_compact_number(value: float) -> str:
    return _matrix_compact_number(value)


def _z_monospace(value: str) -> str:
    return rf"\texttt{{{value}}}"


def _z_time_seconds(value: object) -> str:
    if value is None:
        return r"\ReportNA"
    return _z_monospace(_z_compact_number(float(value)))


def _z_time_microseconds(value: object) -> str:
    if value is None:
        return r"\ReportNA"
    return _z_monospace(_z_compact_number(1.0e6 * float(value)))


def _z_ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return r"\ReportNA"
    if value < 1.0:
        color = "ReportGreen"
    elif value < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return rf"\textcolor{{{color}}}{{\texttt{{(x{_z_compact_number(value)})}}}}"


def _z_metric_with_ratio(
    measurement: Mapping[str, object],
    field: str,
    formatter,
    reference: object,
) -> str:
    value = measurement.get(field)
    if value is None:
        return r"\ReportNA"
    ratio = _safe_divide(value, reference)
    return formatter(value) + " " + _z_ratio(ratio)


def _z_evaluator_value(measurement: Mapping[str, object]) -> object:
    evaluator = measurement.get("evaluator_seconds_per_point")
    if evaluator is None:
        evaluator = measurement.get("wall_seconds_per_point")
    return evaluator


def _z_status_note(measurement: Mapping[str, object]) -> str:
    status = str(measurement.get("status", NA_STATUS))
    if status == ResultStatus.OK.value:
        return ""
    if status == NA_STATUS:
        return r"\ReportNA"
    return _measurement_label(measurement)


def _z_variant_route(variant: VariantSpec) -> str:
    return "reference" if variant.key == "reference" else "Rusticol"


def _z_variant_setup(variant: VariantSpec) -> str:
    labels = {
        "reference": r"\AC",
        "jit_o1": r"\PAC\ JIT \(\mathrm{O}1\)",
        "jit_o3": r"\PAC\ JIT \(\mathrm{O}3\)",
        "asm_o3": r"\PAC\ ASM \(\mathrm{O}3\)",
        "cpp_o3": r"\PAC\ C++ \(\mathrm{O}3\)",
    }
    return labels.get(variant.key, _tex_escape(variant.label))


def _z_variant_notes(
    variant: VariantSpec,
    measurement: Mapping[str, object],
) -> str:
    if measurement.get("status") != ResultStatus.OK.value:
        return _z_status_note(measurement)
    batch = measurement.get("environment", {})
    batch_size = None
    if isinstance(batch, Mapping):
        batch_size = batch.get("batch_size")
    notes = {
        "reference": r"Fortran \AC; legacy oracle",
        "jit_o1": "JIT; O1",
        "jit_o3": "JIT; O3",
        "asm_o3": "ASM; O3",
        "cpp_o3": "C++; O3",
    }.get(variant.key, _tex_escape(variant.label))
    if batch_size is not None and variant.key != "reference":
        notes += rf"; batch={_tex_escape(str(batch_size))}"
    return notes


def _z_table_rows(
    *,
    spec: LadderSpec,
    entries: Mapping[tuple[int, str], Mapping[str, object]],
    multiplicities: Sequence[int],
) -> list[str]:
    lines: list[str] = []
    for n_final in multiplicities:
        reference_entry = entries.get((n_final, "reference"), {})
        reference_measurement = (
            reference_entry.get("measurement", {})
            if isinstance(reference_entry, Mapping)
            else {}
        )
        if not isinstance(reference_measurement, Mapping):
            reference_measurement = {}
        reference_generation = (
            reference_measurement.get("generation_seconds")
            if _measurement_ok(reference_measurement)
            else None
        )
        reference_runtime = (
            _z_evaluator_value(reference_measurement)
            if _measurement_ok(reference_measurement)
            else None
        )
        for variant in spec.variants:
            entry = entries[(n_final, variant.key)]
            measurement = entry["measurement"]
            assert isinstance(measurement, Mapping)
            if variant.key == "reference":
                generation = (
                    _z_time_seconds(measurement.get("generation_seconds"))
                    if _measurement_ok(measurement)
                    else _measurement_label(measurement)
                )
                wall = r"\ReportNA"
                evaluator = (
                    _z_time_microseconds(_z_evaluator_value(measurement))
                    if _measurement_ok(measurement)
                    else r"\ReportNA"
                )
                lines.append(r"\rowcolor{refblue}")
            elif _measurement_ok(measurement):
                generation = _z_metric_with_ratio(
                    measurement,
                    "generation_seconds",
                    _z_time_seconds,
                    reference_generation,
                )
                wall = _z_metric_with_ratio(
                    measurement,
                    "wall_seconds_per_point",
                    _z_time_microseconds,
                    reference_runtime,
                )
                evaluator_value = _z_evaluator_value(measurement)
                evaluator = (
                    r"\ReportNA"
                    if evaluator_value is None
                    else _z_time_microseconds(evaluator_value)
                    + " "
                    + _z_ratio(_safe_divide(evaluator_value, reference_runtime))
                )
                if variant.key == "jit_o3":
                    lines.append(r"\rowcolor{ReportGreen!12}")
            else:
                generation = _measurement_label(measurement)
                wall = r"\ReportNA"
                evaluator = r"\ReportNA"
            cells = (
                str(n_final),
                _z_variant_route(variant),
                _z_variant_setup(variant),
                generation,
                wall,
                evaluator,
                _z_variant_notes(variant, measurement),
            )
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[0.55em]")
    return lines


def _z_old_mode_key(variant_key: str) -> str:
    return {
        "reference": "amplicol",
        "asm_o3": "asm",
    }.get(variant_key, variant_key)


def _z_old_process_for_n(n_final: int) -> str:
    return f"d d~ > Z + ({n_final}-1)*g"


def _z_old_explicit_process_for_n(n_final: int) -> str:
    return "d d~ > " + " ".join(("z", *("g" for _ in range(n_final - 1))))


def _z_reference_color_order_for_n(n_final: int) -> list[int]:
    return [2, *range(4, n_final + 3), 1, 3]


def _z_reference_color_order_cli(n_final: int) -> str:
    return ",".join(str(label) for label in _z_reference_color_order_for_n(n_final))


def _z_old_status(value: object) -> str:
    status = str(value)
    if status == ResultStatus.OK.value:
        return "ok"
    if status in {NA_STATUS, "missing", "not_run"}:
        return "missing"
    if status in {ResultStatus.TIMEOUT.value, "timeout"}:
        return "timeout"
    if status in {ResultStatus.MEMORY_LIMIT.value, "ram_limit"}:
        return "ram_limit"
    if status == ResultStatus.VALIDATION_FAILED.value:
        return "validation_failed"
    if status == ResultStatus.UNSUPPORTED.value:
        return "unsupported"
    return "error"


def _z_old_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _z_old_missing(*, color: str = "speedred") -> str:
    return rf"\textcolor{{{color}}}{{\texttt{{N/A}}}}"


def _z_old_format_number(value: float) -> str:
    text = f"{value:.3g}"
    return text.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _z_old_format_plain(value: float | None) -> str:
    if value is None:
        return _z_old_missing()
    return rf"\texttt{{{_z_old_format_number(value)}}}"


def _z_old_format_ratio(value: float) -> str:
    if value < 0.095:
        return f"{value:.2f}"
    if value < 9.95:
        return f"{value:.2f}"
    return _z_old_format_number(value)


def _z_old_format_with_ratio(value: float | None, reference: float | None) -> str:
    if value is None:
        return _z_old_missing()
    text = _z_old_format_plain(value)
    if reference is None or reference <= 0.0:
        return text
    ratio = value / reference
    color = (
        "speedgreen" if ratio < 1.0 else "speedorange" if ratio < 2.0 else "speedred"
    )
    return (
        text + rf" \textcolor{{{color}}}{{\texttt{{(x{_z_old_format_ratio(ratio)})}}}}"
    )


def _z_old_status_cell(status: str) -> str:
    labels = {
        "timeout": "t/o",
        "ram_limit": "RAM>800G",
        "validation_failed": "VALIDATION FAILED",
        "unsupported": "UNSUPPORTED",
        "error": "ERROR",
    }
    label = labels.get(status, status.upper())
    return rf"\textcolor{{speedred}}{{\texttt{{{_tex_escape(label)}}}}}"


def _z_old_ratio_against_built_in(
    row: Mapping[str, object],
    built_in_row: Mapping[str, object],
    *,
    mode_key: str,
    status_key: str,
    value_key: str,
) -> str:
    if mode_key == "amplicol":
        return ""
    if row.get(status_key) != "ok" or built_in_row.get(status_key) != "ok":
        return _z_old_missing(color="black!45")
    value = _z_old_optional_float(row.get(value_key))
    reference = _z_old_optional_float(built_in_row.get(value_key))
    if value is None or reference is None or reference <= 0.0:
        return _z_old_missing(color="black!45")
    ratio = value / reference
    color = (
        "speedgreen" if ratio < 1.0 else "speedorange" if ratio < 2.0 else "speedred"
    )
    return rf"\textcolor{{{color}}}{{\texttt{{x{_z_old_format_ratio(ratio)}}}}}"


def _z_old_render_timing_triplet(
    row: Mapping[str, object],
    *,
    mode_key: str,
    status: str,
    generation_key: str,
    wall_key: str,
    runtime_key: str,
    ref_generation: float | None,
    ref_runtime: float | None,
) -> list[str]:
    if status == "missing":
        if mode_key == "amplicol":
            return [
                _z_old_missing(),
                _z_old_missing(color="black!45"),
                _z_old_missing(),
            ]
        return [_z_old_missing(), _z_old_missing(), _z_old_missing()]
    if status in {"timeout", "ram_limit", "validation_failed", "unsupported", "error"}:
        return [
            _z_old_status_cell(status),
            _z_old_missing(color="black!45")
            if mode_key == "amplicol"
            else _z_old_missing(),
            _z_old_missing(),
        ]
    generation = _z_old_optional_float(row.get(generation_key))
    wall = _z_old_optional_float(row.get(wall_key))
    runtime = _z_old_optional_float(row.get(runtime_key))
    if mode_key == "amplicol":
        return [
            _z_old_format_plain(generation),
            _z_old_format_plain(runtime),
            _z_old_missing(color="black!45"),
        ]
    return [
        _z_old_format_with_ratio(generation, ref_generation),
        _z_old_format_with_ratio(wall, ref_runtime),
        _z_old_format_with_ratio(runtime, ref_runtime),
    ]


def _z_old_render_mode_cells(
    row: Mapping[str, object],
    *,
    mode_key: str,
    ref_generation: float | None,
    ref_runtime: float | None,
    ref_all_flow_generation: float | None,
    ref_all_flow_runtime: float | None,
) -> list[str]:
    status = str(row.get("status", "missing"))
    all_flow_status = str(row.get("all_flow_status", "missing"))
    if status == "missing":
        selected = (
            [_z_old_missing(), _z_old_missing(color="black!45"), _z_old_missing()]
            if mode_key == "amplicol"
            else [_z_old_missing(), _z_old_missing(), _z_old_missing()]
        )
    elif status in {
        "timeout",
        "ram_limit",
        "validation_failed",
        "unsupported",
        "error",
    }:
        selected = [
            _z_old_status_cell(status),
            _z_old_missing(color="black!45")
            if mode_key == "amplicol"
            else _z_old_missing(),
            _z_old_missing(),
        ]
    else:
        generation = _z_old_optional_float(row.get("generation_s"))
        wall = _z_old_optional_float(row.get("wall_us_per_point"))
        runtime = _z_old_optional_float(row.get("runtime_us_per_point"))
        if mode_key == "amplicol":
            selected = [
                _z_old_format_plain(generation),
                _z_old_format_plain(runtime),
                _z_old_missing(color="black!45"),
            ]
        else:
            selected = [
                _z_old_format_with_ratio(generation, ref_generation),
                _z_old_format_with_ratio(wall, ref_runtime),
                _z_old_format_with_ratio(runtime, ref_runtime),
            ]
    return [
        *selected,
        *_z_old_render_timing_triplet(
            row,
            mode_key=mode_key,
            status=all_flow_status,
            generation_key="all_flow_generation_s",
            wall_key="all_flow_wall_us_per_point",
            runtime_key="all_flow_runtime_us_per_point",
            ref_generation=ref_all_flow_generation,
            ref_runtime=ref_all_flow_runtime,
        ),
    ]


def _z_old_row_from_measurement(
    measurement: Mapping[str, object],
    *,
    variant_key: str,
) -> dict[str, object]:
    fields = _measurement_old_matrix_fields(measurement)
    if fields:
        selected_status = _z_old_status(fields.get("status", measurement.get("status")))
    else:
        selected_status = _z_old_status(
            measurement.get("status")
            if measurement.get("status") != ResultStatus.OK.value
            else NA_STATUS
        )
    row: dict[str, object] = {
        "status": selected_status,
        "all_flow_status": _z_old_status(fields.get("all_flow_status", NA_STATUS)),
    }
    for target, source in (
        ("generation_s", "generation_s"),
        ("wall_us_per_point", "wall_us_per_point"),
        ("runtime_us_per_point", "runtime_us_per_point"),
        ("all_flow_generation_s", "all_flow_generation_s"),
        ("all_flow_wall_us_per_point", "all_flow_wall_us_per_point"),
        ("all_flow_runtime_us_per_point", "all_flow_runtime_us_per_point"),
    ):
        if source in fields and fields[source] is not None:
            row[target] = fields[source]
    if variant_key == "reference":
        row["wall_us_per_point"] = row.get("runtime_us_per_point")
        row["all_flow_wall_us_per_point"] = row.get("all_flow_runtime_us_per_point")
    return row


def _z_old_rows_by_variant(
    entries: Mapping[tuple[int, str], Mapping[str, object]],
    *,
    n_final: int,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for variant in Z_VARIANTS:
        entry = entries.get((n_final, variant.key), {})
        measurement = entry.get("measurement", {}) if isinstance(entry, Mapping) else {}
        if not isinstance(measurement, Mapping):
            measurement = {}
        row = _z_old_row_from_measurement(
            measurement,
            variant_key=variant.key,
        )
        metadata = measurement.get("metadata")
        validation = (
            metadata.get("pointwise_validation")
            if isinstance(metadata, Mapping)
            else None
        )
        if isinstance(validation, Mapping):
            if validation.get("status") == ResultStatus.VALIDATION_FAILED.value:
                row["status"] = ResultStatus.VALIDATION_FAILED.value
            if (
                validation.get("all_flow_status")
                == ResultStatus.VALIDATION_FAILED.value
            ):
                row["all_flow_status"] = ResultStatus.VALIDATION_FAILED.value
        rows[variant.key] = row
    return rows


def _z_old_reproduction_n(
    entries: Mapping[tuple[int, str], Mapping[str, object]],
    multiplicities: Sequence[int],
) -> int | None:
    for n_final in (7, 9, 8, 6, 5, 4, 3, 2, 1):
        if n_final not in multiplicities:
            continue
        row = _z_old_rows_by_variant(entries, n_final=n_final).get("jit_o3", {})
        if row.get("status") == "ok":
            return n_final
    return None


def _chunks(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
    step = max(1, int(size))
    return tuple(
        tuple(values[index : index + step]) for index in range(0, len(values), step)
    )


def _matrix_compact_number(value: float) -> str:
    text = f"{float(value):.3g}"
    return text.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _matrix_compact_summary_number(value: float) -> str:
    number = float(value)
    magnitude = abs(number)
    if magnitude and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
        text = f"{number:.1e}"
        return text.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return _matrix_compact_number(number)


def _matrix_format_seconds(value: object) -> str:
    return rf"\texttt{{{_matrix_compact_number(float(value))} s}}"


def _matrix_format_microseconds(value: object) -> str:
    return rf"\texttt{{{_matrix_compact_number(1.0e6 * float(value))} us}}"


def _matrix_na(color: str = "ReportMuted") -> str:
    return rf"\matrixna{{{color}}}"


def _matrix_missing_ratio() -> str:
    return r"\matrixnaratio{ReportMuted}"


def _matrix_failure_label(measurement: Mapping[str, object]) -> str:
    status = str(measurement.get("status", NA_STATUS))
    if status == NA_STATUS:
        return _matrix_na()
    if status == ResultStatus.TIMEOUT.value:
        return r"\textcolor{ReportRed}{\texttt{t/o}}"
    if status == ResultStatus.MEMORY_LIMIT.value:
        limit = measurement.get("limit_gib")
        suffix = "" if limit is None else f">{float(limit):.0f}G"
        return rf"\textcolor{{ReportRed}}{{\texttt{{RAM{suffix}}}}}"
    if status == ResultStatus.VALIDATION_FAILED.value:
        return r"\textcolor{ReportRed}{\textbf{VALIDATION FAILED}}"
    if status == ResultStatus.UNSUPPORTED.value:
        return r"\textcolor{ReportMuted}{\texttt{UNSUPPORTED}}"
    return r"\textcolor{ReportRed}{\texttt{ERROR}}"


def _matrix_reference_metric(
    measurement: Mapping[str, object],
    field: str,
    formatter,
) -> str:
    if _measurement_ok(measurement):
        value = measurement.get(field)
        return _matrix_na() if value is None else formatter(value)
    return _matrix_failure_label(measurement)


def _matrix_multiplier_fragment(value: object) -> str:
    color, text = _matrix_multiplier_inner(value)
    if text == "N/A":
        return _matrix_missing_ratio()
    return rf"\matrixratio{{{color}}}{{{text}}}"


def _matrix_multiplier_pair_text(value: object) -> tuple[str, str]:
    color, text = _matrix_multiplier_inner(value)
    return color, text


def _matrix_multiplier_inner(value: object) -> tuple[str, str]:
    if value is None:
        return ("ReportRed", "N/A")
    number = float(value)
    if not math.isfinite(number):
        return ("ReportRed", "N/A")
    if number < 1.0:
        color = "ReportGreen"
    elif number < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return (color, _matrix_compact_number(number))


def _matrix_jit_generation(
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
) -> str:
    if not _measurement_ok(pyamplicol):
        return _matrix_failure_label(pyamplicol)
    py_generation = pyamplicol.get("generation_seconds")
    if py_generation is None:
        return _matrix_na()
    if _measurement_ok(legacy):
        return _matrix_multiplier_fragment(
            _safe_divide(py_generation, legacy.get("generation_seconds"))
        )
    return _matrix_format_seconds(py_generation)


def _matrix_runtime_pair(
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
) -> str:
    if not _measurement_ok(pyamplicol):
        return _matrix_failure_label(pyamplicol)
    wall = pyamplicol.get("wall_seconds_per_point")
    evaluator = pyamplicol.get("evaluator_seconds_per_point")
    if evaluator is None:
        evaluator = wall
    if _measurement_ok(legacy):
        reference = legacy.get("wall_seconds_per_point")
        wall_color, wall_text = _matrix_multiplier_pair_text(
            _safe_divide(wall, reference)
        )
        evaluator_color, evaluator_text = _matrix_multiplier_pair_text(
            _safe_divide(evaluator, reference)
        )
        if wall_text != "N/A":
            wall_text = f"x{wall_text}"
        return (
            r"\matrixratiopair"
            f"{{{wall_color}}}{{{wall_text}}}"
            f"{{{evaluator_color}}}{{{evaluator_text}}}"
        )
    wall_text = _matrix_na() if wall is None else _matrix_format_microseconds(wall)
    evaluator_text = (
        _matrix_na() if evaluator is None else _matrix_format_microseconds(evaluator)
    )
    return (
        r"\begin{tabular}[t]{@{}l@{\hspace{0.006in}\matrixpunct{|}\hspace{0.006in}}l@{}}"
        + wall_text
        + "&"
        + evaluator_text
        + r"\end{tabular}"
    )


def _matrix_old_value(
    measurement: Mapping[str, object],
    key: str,
    fallback_key: str | None = None,
) -> object:
    fields = _measurement_old_matrix_fields(measurement)
    if key in fields:
        return fields[key]
    if fallback_key is not None:
        return measurement.get(fallback_key)
    return None


def _matrix_scaled_old_value(
    measurement: Mapping[str, object],
    key: str,
    *,
    value_scale: float = 1.0,
    fallback_key: str | None = None,
    fallback_scale: float = 1.0,
) -> float | None:
    fields = _measurement_old_matrix_fields(measurement)
    if key in fields:
        value = fields[key]
        scale = value_scale
    elif fallback_key is not None:
        value = measurement.get(fallback_key)
        scale = fallback_scale
    else:
        return None
    if value is None:
        return None
    return float(value) * scale


def _matrix_reference_pair(
    measurement: Mapping[str, object],
    selected_key: str,
    all_flow_key: str,
    formatter,
    *,
    selected_fallback_key: str | None = None,
    selected_fallback_scale: float = 1.0,
) -> str:
    if not _measurement_ok(measurement):
        return _matrix_failure_label(measurement)
    selected = _matrix_scaled_old_value(
        measurement,
        selected_key,
        fallback_key=selected_fallback_key,
        fallback_scale=selected_fallback_scale,
    )
    all_flow = _matrix_old_value(measurement, all_flow_key)
    if selected is None and all_flow is None:
        return _matrix_na("ReportRed")
    if selected is None:
        return formatter(all_flow)
    if all_flow is None:
        return formatter(selected)
    return rf"\matrixrefpair{{{formatter(selected)}}}{{{formatter(all_flow)}}}"


def _matrix_plain_number(value: object) -> str:
    return rf"\texttt{{{_matrix_compact_number(float(value))}}}"


def _matrix_py_over_ref_ratio(value: object) -> str:
    if value is None:
        return _matrix_missing_ratio()
    number = float(value)
    if not math.isfinite(number):
        return _matrix_missing_ratio()
    if number < 1.0:
        color = "ReportGreen"
    elif number < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return rf"\matrixratio{{{color}}}{{{_matrix_compact_number(number)}}}"


def _matrix_py_over_ref_pair(
    wall_value: object,
    evaluator_value: object,
) -> str:
    def fragment(value: object) -> tuple[str, str]:
        if value is None:
            return ("ReportMuted", "N/A")
        number = float(value)
        if not math.isfinite(number):
            return ("ReportMuted", "N/A")
        if number < 1.0:
            color = "ReportGreen"
        elif number < 2.0:
            color = "ReportOrange"
        else:
            color = "ReportRed"
        return (color, _matrix_compact_number(number))

    wall_color, wall_text = fragment(wall_value)
    evaluator_color, evaluator_text = fragment(evaluator_value)
    if wall_text != "N/A":
        wall_text = f"x{wall_text}"
    return (
        r"\matrixratiopair"
        f"{{{wall_color}}}{{{wall_text}}}"
        f"{{{evaluator_color}}}{{{evaluator_text}}}"
    )


def _matrix_lc_generation_ratio(
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
    *,
    legacy_key: str,
    py_key: str,
    legacy_fallback_key: str | None = None,
    py_fallback_key: str | None = None,
) -> str:
    if not _measurement_ok(pyamplicol):
        return _matrix_failure_label(pyamplicol)
    reference = _matrix_old_value(legacy, legacy_key, legacy_fallback_key)
    py_value = _matrix_old_value(pyamplicol, py_key, py_fallback_key)
    return _matrix_py_over_ref_ratio(_safe_divide(py_value, reference))


def _matrix_lc_runtime_ratio(
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
    *,
    legacy_key: str,
    py_wall_key: str,
    py_eval_key: str,
    legacy_scale: float = 1.0e-6,
    py_scale: float = 1.0e-6,
    py_wall_fallback_key: str | None = None,
    py_eval_fallback_key: str | None = None,
) -> str:
    if not _measurement_ok(pyamplicol):
        return _matrix_failure_label(pyamplicol)
    reference = _matrix_scaled_old_value(
        legacy,
        legacy_key,
        value_scale=legacy_scale,
    )
    py_wall = _matrix_scaled_old_value(
        pyamplicol,
        py_wall_key,
        value_scale=py_scale,
        fallback_key=py_wall_fallback_key,
    )
    py_eval = _matrix_scaled_old_value(
        pyamplicol,
        py_eval_key,
        value_scale=py_scale,
        fallback_key=py_eval_fallback_key,
    )
    return _matrix_py_over_ref_pair(
        _safe_divide(py_wall, reference),
        _safe_divide(py_eval, reference),
    )


def _matrix_validation_marker(entry: Mapping[str, object]) -> str:
    validation = entry.get("pointwise_validation", {})
    if not isinstance(validation, Mapping):
        return ""
    status = str(validation.get("status", NA_STATUS))
    if status == ResultStatus.VALIDATION_FAILED.value:
        return r"\\[-0.1em]\textcolor{ReportRed}{\scriptsize\textbf{VALIDATION FAILED}}"
    return ""


def _matrix_cell(entry: Mapping[str, object], *, color_accuracy: str) -> str:
    if not bool(entry.get("applicable", False)):
        return _matrix_na()
    legacy = entry["legacy_amplicol"]
    pyamplicol = entry["pyamplicol_jit_o3"]
    assert isinstance(legacy, Mapping)
    assert isinstance(pyamplicol, Mapping)
    if (
        legacy["status"] == NA_STATUS
        and pyamplicol["status"] == NA_STATUS
        and entry.get("status") == NA_STATUS
    ):
        return _matrix_na()
    if color_accuracy == "lc":
        reference_generation = _matrix_reference_metric(
            legacy,
            "generation_seconds",
            _matrix_plain_number,
        )
        generation_selected = _matrix_lc_generation_ratio(
            legacy,
            pyamplicol,
            legacy_key="generation_s",
            py_key="selected_generation_s",
            legacy_fallback_key="generation_seconds",
            py_fallback_key="generation_seconds",
        )
        generation_all_flow = _matrix_lc_generation_ratio(
            legacy,
            pyamplicol,
            legacy_key="generation_s",
            py_key="all_flow_generation_s",
            legacy_fallback_key="generation_seconds",
            py_fallback_key="generation_seconds",
        )
        reference_runtime = _matrix_reference_pair(
            legacy,
            "runtime_us_per_point",
            "all_flow_runtime_us_per_point",
            _matrix_plain_number,
            selected_fallback_key="wall_seconds_per_point",
            selected_fallback_scale=1.0e6,
        )
        runtime_selected = _matrix_lc_runtime_ratio(
            legacy,
            pyamplicol,
            legacy_key="runtime_us_per_point",
            py_wall_key="wall_us_per_point",
            py_eval_key="runtime_us_per_point",
            py_wall_fallback_key="wall_seconds_per_point",
            py_eval_fallback_key="evaluator_seconds_per_point",
        )
        runtime_all_flow = _matrix_lc_runtime_ratio(
            legacy,
            pyamplicol,
            legacy_key="all_flow_runtime_us_per_point",
            py_wall_key="all_flow_wall_us_per_point",
            py_eval_key="all_flow_runtime_us_per_point",
        )
        cell = (
            rf"\matrixcell{{{reference_generation}}}{{{generation_selected}}}"
            rf"{{{generation_all_flow}}}{{{reference_runtime}}}"
            rf"{{{runtime_selected}}}{{{runtime_all_flow}}}"
        )
        marker = _matrix_validation_marker(entry)
        if marker:
            return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{cell}{marker}\end{{tabular}}"
        return cell
    reference_generation = _matrix_reference_metric(
        legacy,
        "generation_seconds",
        _matrix_format_seconds,
    )
    reference_runtime = _matrix_reference_metric(
        legacy,
        "wall_seconds_per_point",
        _matrix_format_microseconds,
    )
    cell = (
        r"\matrixcelljitothree"
        f"{{{reference_generation}}}"
        f"{{{_matrix_jit_generation(legacy, pyamplicol)}}}"
        f"{{{reference_runtime}}}"
        f"{{{_matrix_runtime_pair(legacy, pyamplicol)}}}"
    )
    marker = _matrix_validation_marker(entry)
    if marker:
        return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{cell}{marker}\end{{tabular}}"
    return cell


def _matrix_table_macros() -> list[str]:
    return [
        r"\providecommand{\matrixentryfont}{\fontsize{6.3pt}{7.1pt}\selectfont}",
        r"\providecommand{\matrixsummaryfont}{\fontsize{5.8pt}{6.4pt}\selectfont}",
        r"\providecommand{\matrixpunct}[1]{\textcolor{black}{\texttt{#1}}}",
        r"\providecommand{\matrixratio}[2]{\matrixpunct{(}\textcolor{#1}{\texttt{x#2}}\matrixpunct{)}}",
        r"\providecommand{\matrixratioinner}[2]{#2}",
        r"\providecommand{\matrixna}[1]{\textcolor{#1}{\texttt{N/A}}}",
        r"\providecommand{\matrixnaratio}[1]{\matrixpunct{(}\matrixna{#1}\matrixpunct{)}}",
        (
            r"\providecommand{\matrixratiopair}[4]{"
            r"\matrixpunct{(}"
            r"\textcolor{#1}{\texttt{#2}}"
            r"\matrixpunct{|}"
            r"\textcolor{#3}{\texttt{#4}}"
            r"\matrixpunct{)}}"
        ),
        r"\providecommand{\matrixslot}[2]{\makebox[#1][l]{#2}}",
        (
            r"\providecommand{\matrixcelljitothree}[4]{"
            r"\begingroup\matrixentryfont"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.006in}}l@{}}"
            r"\matrixslot{1.20in}{#1}&\matrixslot{1.12in}{#2}\\"
            r"\matrixslot{1.20in}{#3}&\matrixslot{1.12in}{#4}"
            r"\end{tabular}\endgroup}"
        ),
        (
            r"\providecommand{\matrixcell}[6]{"
            r"\begingroup\matrixentryfont"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.004in}}l@{\hspace{0.004in}}l@{}}"
            r"\matrixslot{0.86in}{#1}&\matrixslot{0.68in}{#2}&"
            r"\matrixslot{0.68in}{#3}\\"
            r"\matrixslot{0.86in}{#4}&\matrixslot{0.68in}{#5}&"
            r"\matrixslot{0.68in}{#6}"
            r"\end{tabular}\endgroup}"
        ),
        r"\providecommand{\matrixrefslot}[1]{\makebox[0.27in][l]{#1}}",
        (
            r"\providecommand{\matrixrefpair}[2]{"
            r"\begin{tabular}[t]{@{}l@{\hspace{0.006in}\matrixpunct{/}\hspace{0.012in}}l@{}}"
            r"\matrixrefslot{#1}&\matrixrefslot{#2}"
            r"\end{tabular}}"
        ),
        (
            r"\providecommand{\matrixsummarycell}[2]{"
            r"\begingroup\matrixsummaryfont"
            r"\begin{tabular}[t]{@{}l@{}}#1\\#2\end{tabular}\endgroup}"
        ),
        r"\providecommand{\matrixsummaryfield}[1]{\makebox[0.38in][r]{#1}}",
        (
            r"\providecommand{\matrixsummaryfour}[4]{"
            r"\matrixsummaryfield{#1}\matrixpunct{|}"
            r"\matrixsummaryfield{#2}\matrixpunct{|}"
            r"\matrixsummaryfield{#3}\matrixpunct{|}"
            r"\matrixsummaryfield{#4}}"
        ),
        (
            r"\providecommand{\matrixsummaryfive}[5]{"
            r"\matrixsummaryfield{#1}\matrixpunct{|}"
            r"\matrixsummaryfield{#2}\matrixpunct{|}"
            r"\matrixsummaryfield{#3}\matrixpunct{|}"
            r"\matrixsummaryfield{#4}\matrixpunct{|}"
            r"\matrixsummaryfield{#5}}"
        ),
    ]


def _summary_stats(values: Sequence[float]) -> tuple[float, float, float, float] | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    midpoint = len(finite) // 2
    median = (
        finite[midpoint]
        if len(finite) % 2
        else 0.5 * (finite[midpoint - 1] + finite[midpoint])
    )
    return (finite[0], finite[-1], math.fsum(finite) / len(finite), median)


def _summary_numeric_stats_line(values: Sequence[float]) -> str:
    stats = _summary_stats(values)
    if stats is None:
        return _matrix_na()
    return (
        r"\matrixsummaryfour{"
        + "}{".join(rf"\texttt{{{_matrix_compact_number(value)}}}" for value in stats)
        + "}"
    )


def _summed_ratio(
    numerators: Sequence[float],
    denominators: Sequence[float],
) -> float | None:
    pairs = [
        (numerator, denominator)
        for numerator, denominator in zip(numerators, denominators, strict=False)
        if math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0.0
    ]
    if not pairs:
        return None
    denominator_sum = math.fsum(denominator for _, denominator in pairs)
    if denominator_sum <= 0.0:
        return None
    return math.fsum(numerator for numerator, _ in pairs) / denominator_sum


def _summary_multiplier_stats_line(
    values: Sequence[float],
    summed: float | None,
) -> str:
    stats = _summary_stats(values)
    if stats is None:
        return _matrix_na()
    fields = [_summary_multiplier_fragment(value) for value in stats]
    fields.append(_summary_multiplier_fragment(summed))
    return r"\matrixsummaryfive{" + "}{".join(fields) + "}"


def _summary_multiplier_fragment(value: object) -> str:
    if value is None:
        return _matrix_missing_ratio()
    number = float(value)
    if not math.isfinite(number):
        return _matrix_missing_ratio()
    if number < 1.0:
        color = "ReportGreen"
    elif number < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return rf"\matrixratio{{{color}}}{{{_matrix_compact_summary_number(number)}}}"


def _matrix_column_summary(
    entries: Mapping[tuple[str, int], Mapping[str, object]],
    n_final: int,
    *,
    color_accuracy: str,
) -> dict[str, list[float]]:
    if color_accuracy == "lc":
        return _matrix_lc_column_summary(entries, n_final)
    summary: dict[str, list[float]] = {
        "reference_generation": [],
        "jit_generation_multiplier": [],
        "jit_generation_reference_paired": [],
        "jit_generation_paired": [],
        "reference_runtime_wall_us": [],
        "jit_runtime_wall_multiplier": [],
        "jit_runtime_wall_reference_paired": [],
        "jit_runtime_wall_paired": [],
        "jit_runtime_evaluator_multiplier": [],
        "jit_runtime_evaluator_reference_paired": [],
        "jit_runtime_evaluator_paired": [],
    }
    for family in PROCESS_FAMILIES:
        if n_final < family.minimum_n:
            continue
        entry = entries.get((family.key, n_final))
        if not isinstance(entry, Mapping) or not bool(entry.get("applicable", False)):
            continue
        legacy = entry.get("legacy_amplicol")
        pyamplicol = entry.get("pyamplicol_jit_o3")
        if not isinstance(legacy, Mapping) or not isinstance(pyamplicol, Mapping):
            continue
        if not _measurement_ok(legacy):
            continue
        reference_generation = legacy.get("generation_seconds")
        if reference_generation is not None:
            reference_generation_float = float(reference_generation)
            summary["reference_generation"].append(reference_generation_float)
            py_generation = pyamplicol.get("generation_seconds")
            if _measurement_ok(pyamplicol) and py_generation is not None:
                py_generation_float = float(py_generation)
                multiplier = _safe_divide(
                    py_generation_float,
                    reference_generation_float,
                )
                if multiplier is not None:
                    summary["jit_generation_multiplier"].append(multiplier)
                    summary["jit_generation_reference_paired"].append(
                        reference_generation_float
                    )
                    summary["jit_generation_paired"].append(py_generation_float)
        reference_runtime = legacy.get("wall_seconds_per_point")
        if reference_runtime is None:
            continue
        reference_runtime_float = float(reference_runtime)
        summary["reference_runtime_wall_us"].append(1.0e6 * reference_runtime_float)
        if not _measurement_ok(pyamplicol):
            continue
        py_wall = pyamplicol.get("wall_seconds_per_point")
        if py_wall is not None:
            py_wall_float = float(py_wall)
            multiplier = _safe_divide(py_wall_float, reference_runtime_float)
            if multiplier is not None:
                summary["jit_runtime_wall_multiplier"].append(multiplier)
                summary["jit_runtime_wall_reference_paired"].append(
                    reference_runtime_float
                )
                summary["jit_runtime_wall_paired"].append(py_wall_float)
        py_evaluator = pyamplicol.get("evaluator_seconds_per_point")
        if py_evaluator is None:
            py_evaluator = pyamplicol.get("wall_seconds_per_point")
        if py_evaluator is not None:
            py_evaluator_float = float(py_evaluator)
            multiplier = _safe_divide(py_evaluator_float, reference_runtime_float)
            if multiplier is not None:
                summary["jit_runtime_evaluator_multiplier"].append(multiplier)
                summary["jit_runtime_evaluator_reference_paired"].append(
                    reference_runtime_float
                )
                summary["jit_runtime_evaluator_paired"].append(py_evaluator_float)
    return summary


def _matrix_lc_column_summary(
    entries: Mapping[tuple[str, int], Mapping[str, object]],
    n_final: int,
) -> dict[str, list[float]]:
    summary: dict[str, list[float]] = {
        "amplicol_generation_one_flow": [],
        "jit_generation_one_flow_ratio": [],
        "jit_generation_one_flow_paired": [],
        "jit_generation_one_flow_ref_paired": [],
        "amplicol_generation_all_flow": [],
        "jit_generation_all_flow_ratio": [],
        "jit_generation_all_flow_paired": [],
        "jit_generation_all_flow_ref_paired": [],
        "amplicol_runtime_one_flow": [],
        "jit_runtime_one_flow_ratio": [],
        "jit_runtime_one_flow_paired": [],
        "jit_runtime_one_flow_ref_paired": [],
        "amplicol_runtime_all_flow": [],
        "jit_runtime_all_flow_ratio": [],
        "jit_runtime_all_flow_paired": [],
        "jit_runtime_all_flow_ref_paired": [],
    }
    for family in PROCESS_FAMILIES:
        if n_final < family.minimum_n:
            continue
        entry = entries.get((family.key, n_final))
        if not isinstance(entry, Mapping) or not bool(entry.get("applicable", False)):
            continue
        legacy = entry.get("legacy_amplicol")
        pyamplicol = entry.get("pyamplicol_jit_o3")
        if not isinstance(legacy, Mapping) or not isinstance(pyamplicol, Mapping):
            continue
        if not _measurement_ok(legacy):
            continue
        ref_generation = _optional_positive_float(
            _matrix_old_value(legacy, "generation_s", "generation_seconds")
        )
        if ref_generation is not None:
            summary["amplicol_generation_one_flow"].append(ref_generation)
            py_generation = _optional_positive_float(
                _matrix_old_value(
                    pyamplicol,
                    "selected_generation_s",
                    "generation_seconds",
                )
            )
            if _measurement_ok(pyamplicol) and py_generation is not None:
                summary["jit_generation_one_flow_ratio"].append(
                    py_generation / ref_generation
                )
                summary["jit_generation_one_flow_paired"].append(py_generation)
                summary["jit_generation_one_flow_ref_paired"].append(ref_generation)
        if ref_generation is not None:
            py_all_generation = _optional_positive_float(
                _matrix_old_value(
                    pyamplicol,
                    "all_flow_generation_s",
                    "generation_seconds",
                )
            )
            if _measurement_ok(pyamplicol) and py_all_generation is not None:
                summary["amplicol_generation_all_flow"].append(ref_generation)
                summary["jit_generation_all_flow_ratio"].append(
                    py_all_generation / ref_generation
                )
                summary["jit_generation_all_flow_paired"].append(py_all_generation)
                summary["jit_generation_all_flow_ref_paired"].append(ref_generation)
        ref_runtime = _optional_positive_float(
            _matrix_old_value(legacy, "runtime_us_per_point")
        )
        if ref_runtime is not None:
            summary["amplicol_runtime_one_flow"].append(ref_runtime)
            py_runtime = _optional_positive_float(
                _matrix_old_value(pyamplicol, "wall_us_per_point")
            )
            if py_runtime is None:
                seconds = _optional_positive_float(
                    _matrix_old_value(pyamplicol, "wall_seconds_per_point")
                )
                py_runtime = None if seconds is None else 1.0e6 * seconds
            if _measurement_ok(pyamplicol) and py_runtime is not None:
                summary["jit_runtime_one_flow_ratio"].append(py_runtime / ref_runtime)
                summary["jit_runtime_one_flow_paired"].append(py_runtime)
                summary["jit_runtime_one_flow_ref_paired"].append(ref_runtime)
        ref_all_runtime = _optional_positive_float(
            _matrix_old_value(legacy, "all_flow_runtime_us_per_point")
        )
        if ref_all_runtime is not None:
            summary["amplicol_runtime_all_flow"].append(ref_all_runtime)
            py_all_runtime = _optional_positive_float(
                _matrix_old_value(pyamplicol, "all_flow_wall_us_per_point")
            )
            if _measurement_ok(pyamplicol) and py_all_runtime is not None:
                summary["jit_runtime_all_flow_ratio"].append(
                    py_all_runtime / ref_all_runtime
                )
                summary["jit_runtime_all_flow_paired"].append(py_all_runtime)
                summary["jit_runtime_all_flow_ref_paired"].append(ref_all_runtime)
    return summary


def _matrix_summary_cell(numeric: str, multipliers: str) -> str:
    return rf"\matrixsummarycell{{{numeric}}}{{{multipliers}}}"


def _matrix_summary_rows(
    entries: Mapping[tuple[str, int], Mapping[str, object]],
    chunk: Sequence[int],
    *,
    color_accuracy: str,
) -> list[str]:
    if color_accuracy == "lc":
        return _matrix_lc_summary_rows(entries, chunk)
    generation_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{\textbf{summary: gen}}"
    ]
    runtime_wall_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{\textbf{summary: run wall}}"
    ]
    runtime_evaluator_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{\textbf{summary: run eval}}"
    ]
    for n_final in chunk:
        summary = _matrix_column_summary(
            entries,
            n_final,
            color_accuracy=color_accuracy,
        )
        generation_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["reference_generation"]),
                _summary_multiplier_stats_line(
                    summary["jit_generation_multiplier"],
                    _summed_ratio(
                        summary["jit_generation_paired"],
                        summary["jit_generation_reference_paired"],
                    ),
                ),
            )
        )
        runtime_wall_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["reference_runtime_wall_us"]),
                _summary_multiplier_stats_line(
                    summary["jit_runtime_wall_multiplier"],
                    _summed_ratio(
                        summary["jit_runtime_wall_paired"],
                        summary["jit_runtime_wall_reference_paired"],
                    ),
                ),
            )
        )
        runtime_evaluator_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["reference_runtime_wall_us"]),
                _summary_multiplier_stats_line(
                    summary["jit_runtime_evaluator_multiplier"],
                    _summed_ratio(
                        summary["jit_runtime_evaluator_paired"],
                        summary["jit_runtime_evaluator_reference_paired"],
                    ),
                ),
            )
        )
    return [
        r"\specialrule{1.05pt}{0.25em}{0.20em}",
        " & ".join(generation_cells) + r" \\",
        r"\addlinespace[0.12em]",
        " & ".join(runtime_wall_cells) + r" \\",
        r"\addlinespace[0.12em]",
        " & ".join(runtime_evaluator_cells) + r" \\",
        r"\addlinespace[0.12em]",
    ]


def _summary_py_over_ref_stats_line(
    values: Sequence[float],
    summed: float | None,
) -> str:
    stats = _summary_stats(values)
    if stats is None:
        return _matrix_na()
    fields = [_summary_py_over_ref_fragment(value) for value in stats]
    fields.append(_summary_py_over_ref_fragment(summed))
    return r"\matrixsummaryfive{" + "}{".join(fields) + "}"


def _summary_py_over_ref_fragment(value: object) -> str:
    if value is None:
        return r"\textcolor{ReportRed}{\texttt{xN/A}}"
    number = float(value)
    if not math.isfinite(number):
        return r"\textcolor{ReportRed}{\texttt{xN/A}}"
    if number < 1.0:
        color = "ReportGreen"
    elif number < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return (
        rf"\textcolor{{{color}}}"
        rf"{{\texttt{{x{_matrix_compact_summary_number(number)}}}}}"
    )


def _matrix_py_over_ref_inner(value: object) -> tuple[str, str]:
    if value is None:
        return ("ReportRed", "N/A")
    number = float(value)
    if not math.isfinite(number):
        return ("ReportRed", "N/A")
    if number < 1.0:
        color = "ReportGreen"
    elif number < 2.0:
        color = "ReportOrange"
    else:
        color = "ReportRed"
    return (color, _matrix_compact_number(number))


def _matrix_lc_summary_rows(
    entries: Mapping[tuple[str, int], Mapping[str, object]],
    chunk: Sequence[int],
) -> list[str]:
    generation_one_flow_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{"
        r"\textbf{gen O3 one-flow, hel. sum}}"
    ]
    generation_all_flow_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{"
        r"\textbf{gen O3 all-flows, one-hel}}"
    ]
    runtime_one_flow_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{"
        r"\textbf{run O3 one-flow, hel. sum}}"
    ]
    runtime_all_flow_cells = [
        r"\multicolumn{2}{@{}L{1.74in}@{\hspace{0.075in}}}{"
        r"\textbf{run O3 all-flows, one-hel}}"
    ]
    for n_final in chunk:
        summary = _matrix_column_summary(
            entries,
            n_final,
            color_accuracy="lc",
        )
        generation_one_flow_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["amplicol_generation_one_flow"]),
                _summary_py_over_ref_stats_line(
                    summary["jit_generation_one_flow_ratio"],
                    _summed_ratio(
                        summary["jit_generation_one_flow_paired"],
                        summary["jit_generation_one_flow_ref_paired"],
                    ),
                ),
            )
        )
        generation_all_flow_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["amplicol_generation_all_flow"]),
                _summary_py_over_ref_stats_line(
                    summary["jit_generation_all_flow_ratio"],
                    _summed_ratio(
                        summary["jit_generation_all_flow_paired"],
                        summary["jit_generation_all_flow_ref_paired"],
                    ),
                ),
            )
        )
        runtime_one_flow_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["amplicol_runtime_one_flow"]),
                _summary_py_over_ref_stats_line(
                    summary["jit_runtime_one_flow_ratio"],
                    _summed_ratio(
                        summary["jit_runtime_one_flow_paired"],
                        summary["jit_runtime_one_flow_ref_paired"],
                    ),
                ),
            )
        )
        runtime_all_flow_cells.append(
            _matrix_summary_cell(
                _summary_numeric_stats_line(summary["amplicol_runtime_all_flow"]),
                _summary_py_over_ref_stats_line(
                    summary["jit_runtime_all_flow_ratio"],
                    _summed_ratio(
                        summary["jit_runtime_all_flow_paired"],
                        summary["jit_runtime_all_flow_ref_paired"],
                    ),
                ),
            )
        )
    return [
        r"\specialrule{1.05pt}{0.25em}{0.20em}",
        " & ".join(generation_one_flow_cells) + r" \\",
        r"\addlinespace[0.16em]",
        " & ".join(generation_all_flow_cells) + r" \\",
        r"\addlinespace[0.12em]",
        " & ".join(runtime_one_flow_cells) + r" \\",
        r"\addlinespace[0.12em]",
        " & ".join(runtime_all_flow_cells) + r" \\",
        r"\addlinespace[0.12em]",
    ]


def render_matrix_table(spec: MatrixSpec, payload: Mapping[str, object]) -> str:
    validate_cache(payload)
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    entries = {
        (str(entry["process_key"]), int(entry["n_final"])): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    chunks = _chunks(spec.multiplicities, 3)
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
    ]
    lines.extend(_matrix_table_macros())
    for chunk_index, chunk in enumerate(chunks):
        multiplicity_columns = r"@{\hspace{0.055in}}".join("L{2.51in}" for _ in chunk)
        column_spec = (
            r"@{}r@{\hspace{0.055in}}L{1.42in}@{\hspace{0.075in}}"
            + multiplicity_columns
            + r"@{}"
        )
        if chunk_index:
            lines.append(r"\clearpage")
        title = spec.title if chunk_index == 0 else f"{spec.title} (continued)"
        heading = (
            rf"\subsection*{{{title}}}" if chunk_index else rf"\subsection{{{title}}}"
        )
        lines.extend(
            [
                heading,
                r"\begingroup",
                r"\scriptsize",
                r"\setlength{\LTpre}{0.15em}",
                r"\setlength{\LTpost}{0.15em}",
                r"\setlength{\tabcolsep}{2.2pt}",
                r"\renewcommand{\arraystretch}{1.06}",
                rf"\begin{{longtable}}{{{column_spec}}}",
                r"\toprule",
                r"\textbf{ID} & \textbf{base process}"
                + "".join(rf" & \textbf{{n={n}}}" for n in chunk)
                + r" \\",
                r"\specialrule{0.85pt}{0pt}{0pt}",
                r"\endfirsthead",
                r"\toprule",
                r"\textbf{ID} & \textbf{base process}"
                + "".join(rf" & \textbf{{n={n}}}" for n in chunk)
                + r" \\",
                r"\specialrule{0.85pt}{0pt}{0pt}",
                r"\endhead",
            ]
        )
        for row_index, family in enumerate(PROCESS_FAMILIES):
            cells = [rf"\texttt{{{family.identifier}}}", family.label_tex]
            for n_final in chunk:
                entry = entries[(family.key, n_final)]
                cells.append(_matrix_cell(entry, color_accuracy=spec.color_accuracy))
            if row_index % 2 == 0:
                lines.append(r"\rowcolor{refblue}")
            lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace[0.06em]")
        lines.extend(
            _matrix_summary_rows(
                entries,
                chunk,
                color_accuracy=spec.color_accuracy,
            )
        )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{longtable}",
                r"\endgroup",
                "",
            ]
        )
    return "\n".join(lines)


def render_performance_ladder(
    spec: LadderSpec,
    payload: Mapping[str, object],
    *,
    built_in_payload: Mapping[str, object] | None = None,
) -> str:
    validate_cache(payload)
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    entries = {
        (int(entry["n_final"]), str(entry["variant"])): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    compare_to_built_in = spec.model.profile != BUILTIN_SM.profile
    built_in_entries: dict[tuple[int, str], Mapping[str, object]] = {}
    if compare_to_built_in and built_in_payload is not None:
        validate_cache(built_in_payload)
        raw_built_in_entries = built_in_payload["entries"]
        assert isinstance(raw_built_in_entries, list)
        built_in_entries = {
            (int(entry["n_final"]), str(entry["variant"])): entry
            for entry in raw_built_in_entries
            if isinstance(entry, Mapping)
        }
    display_n_values = tuple(spec.multiplicities)
    section_prefix = "UFO-SM " if compare_to_built_in else ""
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
        (
            rf"\subsection{{\texorpdfstring{{{section_prefix}Dedicated "
            r"\(d\bar d\to Z+\) Gluon Performance}"
            rf"{{{section_prefix}Dedicated d dbar to Z plus Gluon Performance}}}}"
        ),
        r"\begingroup",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.25pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        (
            r"\begin{longtable}{@{}r "
            + (
                r"L{0.78in} L{0.40in} L{0.94in} "
                r"R{0.62in} R{0.52in} R{0.62in} R{0.52in} R{0.62in} "
                r"@{\hspace{0.025in}}p{0.22in}@{\hspace{0.025in}} "
                r"R{0.62in} R{0.52in} R{0.62in} R{0.52in} R{0.62in}@{\hspace{0.035in}}}"
                if compare_to_built_in
                else (
                    r"L{0.92in} L{0.48in} L{0.86in} "
                    r"R{0.66in} R{0.66in} R{0.66in} "
                    r"@{\hspace{0.025in}}p{0.22in}@{\hspace{0.025in}} "
                    r"R{0.66in} R{0.66in} R{0.66in}@{\hspace{0.035in}}}"
                )
            )
        ),
        r"\toprule",
        (
            r"\textbf{n} & \textbf{process} & \textbf{route} & \textbf{setup} "
            + (
                rf"& \multicolumn{{{5 if compare_to_built_in else 3}}}{{c}}{{"
                r"\textbf{selected flow, helicity sum}} "
            )
            + r"& "
            + (
                rf"& \multicolumn{{{5 if compare_to_built_in else 3}}}{{c}}{{"
                r"\textbf{all flows, fixed helicity}} "
            )
            + r"\\"
        ),
        (
            r"& & & & \textbf{gen [s]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{wall [us/pt]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{eval [us/pt]} & & \textbf{gen [s]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{wall [us/pt]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{eval [us/pt]} \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (
            r"\textbf{n} & \textbf{process} & \textbf{route} & \textbf{setup} "
            + (
                rf"& \multicolumn{{{5 if compare_to_built_in else 3}}}{{c}}{{"
                r"\textbf{selected flow, helicity sum}} "
            )
            + r"& "
            + (
                rf"& \multicolumn{{{5 if compare_to_built_in else 3}}}{{c}}{{"
                r"\textbf{all flows, fixed helicity}} "
            )
            + r"\\"
        ),
        (
            r"& & & & \textbf{gen [s]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{wall [us/pt]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{eval [us/pt]} & & \textbf{gen [s]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{wall [us/pt]} "
            + (r"& \textbf{vs blt-in} " if compare_to_built_in else "")
            + r"& \textbf{eval [us/pt]} \\"
        ),
        r"\midrule",
        r"\endhead",
    ]
    for n_index, n_final in enumerate(display_n_values):
        rows = _z_old_rows_by_variant(entries, n_final=n_final)
        reference = rows.get("reference", {})
        ref_generation = _z_old_optional_float(reference.get("generation_s"))
        ref_runtime = _z_old_optional_float(reference.get("runtime_us_per_point"))
        ref_all_flow_generation = _z_old_optional_float(
            reference.get("all_flow_generation_s")
        )
        ref_all_flow_runtime = _z_old_optional_float(
            reference.get("all_flow_runtime_us_per_point")
        )
        built_in_rows = (
            _z_old_rows_by_variant(built_in_entries, n_final=n_final)
            if compare_to_built_in
            else {}
        )
        for variant in spec.variants:
            mode_key = _z_old_mode_key(variant.key)
            row = rows.get(variant.key, {})
            row_color = "refblue" if variant.key == "reference" else None
            if variant.key == "jit_o3":
                row_color = "bestgreen"
            if row_color is not None:
                lines.append(rf"\rowcolor{{{row_color}}}")
            cells = _z_old_render_mode_cells(
                row,
                mode_key=mode_key,
                ref_generation=ref_generation,
                ref_runtime=ref_runtime,
                ref_all_flow_generation=ref_all_flow_generation,
                ref_all_flow_runtime=ref_all_flow_runtime,
            )
            if compare_to_built_in:
                built_in_row = built_in_rows.get(variant.key, {})
                cells = [
                    cells[0],
                    _z_old_ratio_against_built_in(
                        row,
                        built_in_row,
                        mode_key=mode_key,
                        status_key="status",
                        value_key="generation_s",
                    ),
                    cells[1],
                    _z_old_ratio_against_built_in(
                        row,
                        built_in_row,
                        mode_key=mode_key,
                        status_key="status",
                        value_key="wall_us_per_point",
                    ),
                    cells[2],
                    cells[3],
                    _z_old_ratio_against_built_in(
                        row,
                        built_in_row,
                        mode_key=mode_key,
                        status_key="all_flow_status",
                        value_key="all_flow_generation_s",
                    ),
                    cells[4],
                    _z_old_ratio_against_built_in(
                        row,
                        built_in_row,
                        mode_key=mode_key,
                        status_key="all_flow_status",
                        value_key="all_flow_wall_us_per_point",
                    ),
                    cells[5],
                ]
            cells = (
                [*cells[:5], "", *cells[5:10]]
                if compare_to_built_in
                else [*cells[:3], "", *cells[3:6]]
            )
            lines.append(
                " & ".join(
                    [
                        rf"\textbf{{{n_final}}}"
                        if variant.key == "reference"
                        else str(n_final),
                        rf"\texttt{{{_tex_escape(_z_old_process_for_n(n_final))}}}",
                        _z_variant_route(variant),
                        _z_variant_setup(variant),
                        *cells,
                    ]
                )
                + r" \\"
            )
        if n_index != len(display_n_values) - 1:
            lines.append(r"\midrule[0.45pt]")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    lines.append("")
    return "\n".join(lines)


def render_model_ladder(spec: LadderSpec, payload: Mapping[str, object]) -> str:
    validate_cache(payload)
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    entries = {
        int(entry["n_final"]): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    multiplicity_count = len(spec.multiplicities)
    column_spec = "@{}L{0.94in}" + "L{0.72in}" * multiplicity_count + "@{}"
    process_family = _tex_escape(spec.process_family)
    measurements = {
        n_final: entries[n_final]["measurement"] for n_final in spec.multiplicities
    }
    if not all(
        isinstance(measurement, Mapping) for measurement in measurements.values()
    ):
        raise TypeError("model-ladder measurements must be objects")

    def model_missing() -> str:
        return _z_old_missing()

    def model_status_label(measurement: Mapping[str, object]) -> str:
        status = _z_old_status(measurement.get("status", NA_STATUS))
        if status == "missing":
            return model_missing()
        if status == "ok":
            return r"\textcolor{speedgreen}{\texttt{ok}}"
        return _z_old_status_cell(status)

    def measurement_value(n_final: int, field: str, *, scale: float = 1.0) -> str:
        measurement = measurements[n_final]
        assert isinstance(measurement, Mapping)
        if str(measurement.get("status", NA_STATUS)) != ResultStatus.OK.value:
            return model_status_label(measurement)
        value = measurement.get(field)
        if value is None:
            return model_missing()
        return _matrix_plain_number(scale * float(value))

    def measurement_row(
        label: str,
        field: str,
        *,
        scale: float = 1.0,
        color: str | None = None,
    ) -> list[str]:
        values = (
            measurement_value(n_final, field, scale=scale)
            for n_final in spec.multiplicities
        )
        row = f"{label} & " + " & ".join(values) + r" \\"
        return ([] if color is None else [rf"\rowcolor{{{color}}}"]) + [row]

    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
        rf"\subsection{{{spec.title}}}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            rf"\multicolumn{{{multiplicity_count + 1}}}{{c}}"
            rf"{{\texttt{{{process_family}}}}}" + r" \\"
        ),
        r"\textbf{metric} & "
        + " & ".join(rf"\textbf{{\texttt{{n={n}}}}}" for n in spec.multiplicities)
        + r" \\",
        r"\midrule",
    ]
    lines.extend(
        measurement_row(
            r"generation [s]",
            "generation_seconds",
            scale=1.0,
            color="ReportGreen!8",
        )
    )
    lines.extend(
        measurement_row(
            r"wall [$\mu$s/pt]",
            "wall_seconds_per_point",
            scale=1.0e6,
            color="ReportBlue!7",
        )
    )
    lines.extend(
        measurement_row(
            r"evaluator [$\mu$s/pt]",
            "evaluator_seconds_per_point",
            scale=1.0e6,
            color="ReportBlue!7",
        )
    )
    lines.extend(
        measurement_row("matrix element", "matrix_element", color="refblue")
    )
    lines.extend(
        [
            r"\rowcolor{ReportOrange!8}",
            "relative diff. vs hp & "
            + " & ".join(
                model_missing()
                if entries[n_final]["relative_difference"] is None
                else _matrix_plain_number(entries[n_final]["relative_difference"])
                for n_final in spec.multiplicities
            )
            + r" \\",
        ]
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def render_tables(
    caches: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    tables: dict[str, str] = {}
    for spec in MATRIX_SPECS:
        tables[spec.table_name] = render_matrix_table(spec, caches[spec.cache_name])
    for spec in LADDER_SPECS:
        payload = caches[spec.cache_name]
        if spec.kind == CacheKind.PERFORMANCE_LADDER:
            built_in_payload = (
                caches["z_builtin_sm.json"]
                if spec.model.profile != BUILTIN_SM.profile
                else None
            )
            tables[spec.table_name] = render_performance_ladder(
                spec,
                payload,
                built_in_payload=built_in_payload,
            )
        else:
            tables[spec.table_name] = render_model_ladder(spec, payload)
    return tables


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_rev_parse(ref: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision or None


@cache
def _report_source_provenance() -> dict[str, object]:
    schema_version: int | None
    compiler_version: int | None
    try:
        schema_version, compiler_version = _current_compiled_model_contract()
    except Exception:
        schema_version, compiler_version = None, None
    return {
        "repository": os.fspath(_repo_root()),
        "head": _git_rev_parse("HEAD"),
        "origin_main": _git_rev_parse("origin/main"),
        "report_version": REPORT_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "compiled_model_schema_version": schema_version,
        "model_compiler_version": compiler_version,
    }


def _ensure_repo_root_on_path() -> None:
    root = os.fspath(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _generation_slice_tools() -> tuple[type[object], Callable[..., object]]:
    _ensure_repo_root_on_path()
    from tools.developer.generation_slice import GenerationSlice, generate_slice

    return GenerationSlice, generate_slice


def _model_source_path(model: ModelSpec) -> Path | None:
    root = _repo_root()
    if model is EXTERNAL_SM or model.profile == EXTERNAL_SM.profile:
        return root / "src/pyamplicol/assets/models/json/sm/sm.json"
    if model is SCALAR_CONTACT or model.profile == SCALAR_CONTACT.profile:
        return root / "src/pyamplicol/assets/models/json/scalars/scalars.json"
    if model is SCALAR_GRAVITY or model.profile == SCALAR_GRAVITY.profile:
        return (
            root
            / "src/pyamplicol/assets/models/json/scalar_gravity/scalar_gravity.json"
        )
    return None


def _spec_by_dataset() -> dict[str, MatrixSpec | LadderSpec]:
    result: dict[str, MatrixSpec | LadderSpec] = {}
    for spec in (*MATRIX_SPECS, *LADDER_SPECS):
        result[spec.dataset_id] = spec
    return result


def _campaign_cells() -> tuple[CampaignCell, ...]:
    cells: list[CampaignCell] = []
    model_order = {BUILTIN_SM.profile: 0, EXTERNAL_SM.profile: 1}
    accuracy_order = {"lc": 0, "nlc": 1, "full": 2}
    for spec in MATRIX_SPECS:
        for family in PROCESS_FAMILIES:
            for n_final in spec.multiplicities:
                if family.process(n_final) is None:
                    continue
                if n_final > family.maximum_n(spec.color_accuracy):
                    continue
                cells.append(
                    CampaignCell(
                        kind="matrix",
                        cache_name=spec.cache_name,
                        dataset_id=spec.dataset_id,
                        n_final=n_final,
                        process=family.process(n_final) or "",
                        process_key=family.key,
                        priority=(
                            n_final,
                            0,
                            model_order.get(spec.model.profile, 9),
                            f"{accuracy_order[spec.color_accuracy]}-{family.identifier:02d}",
                        ),
                    )
                )
    for spec in LADDER_SPECS:
        if spec.kind == CacheKind.PERFORMANCE_LADDER:
            variant_order = {
                variant.key: index for index, variant in enumerate(spec.variants)
            }
            for n_final in spec.multiplicities:
                for variant in spec.variants:
                    cells.append(
                        CampaignCell(
                            kind="performance_ladder",
                            cache_name=spec.cache_name,
                            dataset_id=spec.dataset_id,
                            n_final=n_final,
                            process=spec.process(n_final),
                            variant=variant.key,
                            priority=(
                                n_final,
                                1,
                                model_order.get(spec.model.profile, 9),
                                f"{variant_order[variant.key]:02d}",
                            ),
                        )
                    )
        else:
            for n_final in spec.multiplicities:
                cells.append(
                    CampaignCell(
                        kind="model_ladder",
                        cache_name=spec.cache_name,
                        dataset_id=spec.dataset_id,
                        n_final=n_final,
                        process=spec.process(n_final),
                        priority=(n_final, 2, 0, spec.dataset_id),
                    )
                )
    return tuple(sorted(cells, key=lambda cell: cell.priority))


def _normalized_process_expression(process: str) -> str:
    return " ".join(process.split())


def _select_cells(
    *,
    datasets: set[str] | None = None,
    cell_ids: set[str] | None = None,
    processes: set[str] | None = None,
    process_keys: set[str] | None = None,
    variants: set[str] | None = None,
    n_values: set[int] | None = None,
    limit: int | None = None,
    missing_only: bool = False,
    caches: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[CampaignCell, ...]:
    cache_payloads = caches if missing_only else None
    process_filters = (
        {_normalized_process_expression(process) for process in processes}
        if processes is not None
        else None
    )
    selected: list[CampaignCell] = []
    for cell in _campaign_cells():
        if datasets is not None and cell.dataset_id not in datasets:
            continue
        if cell_ids is not None and cell.cell_id not in cell_ids:
            continue
        if (
            process_filters is not None
            and _normalized_process_expression(cell.process) not in process_filters
        ):
            continue
        if process_keys is not None and cell.process_key not in process_keys:
            continue
        if variants is not None and cell.variant not in variants:
            continue
        if n_values is not None and cell.n_final not in n_values:
            continue
        if cache_payloads is not None and not _campaign_cell_needs_measurement(
            cell,
            cache_payloads,
        ):
            continue
        selected.append(cell)
        if limit is not None and len(selected) >= limit:
            break
    return tuple(selected)


def _known_cell_ids() -> set[str]:
    return {cell.cell_id for cell in _campaign_cells()}


def _known_process_expressions() -> set[str]:
    return {_normalized_process_expression(cell.process) for cell in _campaign_cells()}


def _campaign_cell_needs_measurement(
    cell: CampaignCell,
    caches: Mapping[str, Mapping[str, object]],
) -> bool:
    payload = caches.get(cell.cache_name)
    if payload is None:
        return True
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return True
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if cell.kind == "matrix":
            if (
                entry.get("process_key") != cell.process_key
                or entry.get("n_final") != cell.n_final
            ):
                continue
            if not bool(entry.get("applicable", False)):
                return False
            status = str(entry.get("status", NA_STATUS))
            if status == NA_STATUS:
                return True
            if status in {
                ResultStatus.ERROR.value,
                ResultStatus.FAILED.value,
                ResultStatus.MEMORY_LIMIT.value,
                ResultStatus.TIMEOUT.value,
                ResultStatus.UNSUPPORTED.value,
                ResultStatus.VALIDATION_FAILED.value,
            }:
                return True
            legacy = entry.get("legacy_amplicol")
            if (
                isinstance(legacy, Mapping)
                and str(legacy.get("status", NA_STATUS)) == ResultStatus.OK.value
                and not (
                    _legacy_measurement_revision_current(legacy)
                    and _legacy_measurement_profile_current(legacy)
                )
            ):
                return True
            if cell.dataset_id.endswith("_lc"):
                pyamplicol = entry.get("pyamplicol_jit_o3")
                return not (
                    isinstance(legacy, Mapping)
                    and isinstance(pyamplicol, Mapping)
                    and bool(_measurement_old_matrix_fields(legacy))
                    and _legacy_lc_measurement_contract_current(legacy)
                    and bool(_measurement_old_matrix_fields(pyamplicol))
                    and _pyamplicol_timing_profile_current(pyamplicol)
                    and _pyamplicol_generation_profile_current(pyamplicol)
                    and _pyamplicol_artifacts_current(
                        pyamplicol,
                        require_current_compiled_model_contract=_cell_uses_external_model(
                            cell
                        ),
                    )
                )
            pyamplicol = entry.get("pyamplicol_jit_o3")
            if (
                isinstance(pyamplicol, Mapping)
                and str(pyamplicol.get("status", NA_STATUS)) == ResultStatus.OK.value
            ):
                return not (
                    _pyamplicol_timing_profile_current(pyamplicol)
                    and _pyamplicol_generation_profile_current(pyamplicol)
                    and _pyamplicol_artifacts_current(
                        pyamplicol,
                        require_current_compiled_model_contract=_cell_uses_external_model(
                            cell
                        ),
                    )
                )
            return False
        if cell.kind == "performance_ladder":
            if (
                entry.get("n_final") != cell.n_final
                or entry.get("variant") != cell.variant
            ):
                continue
            measurement = entry.get("measurement")
            if not isinstance(measurement, Mapping):
                return True
            status = str(measurement.get("status", NA_STATUS))
            if status == NA_STATUS:
                return True
            if status in {
                ResultStatus.ERROR.value,
                ResultStatus.FAILED.value,
                ResultStatus.MEMORY_LIMIT.value,
                ResultStatus.TIMEOUT.value,
                ResultStatus.UNSUPPORTED.value,
                ResultStatus.VALIDATION_FAILED.value,
            }:
                return True
            if (
                str(entry.get("status", NA_STATUS))
                == ResultStatus.VALIDATION_FAILED.value
            ):
                return True
            if not bool(_measurement_old_matrix_fields(measurement)):
                return True
            if cell.variant == "reference" and status == ResultStatus.OK.value:
                return not (
                    _legacy_measurement_revision_current(measurement)
                    and _legacy_lc_measurement_contract_current(measurement)
                )
            if cell.variant != "reference" and status == ResultStatus.OK.value:
                return not (
                    _pyamplicol_timing_profile_current(measurement)
                    and _pyamplicol_generation_profile_current(measurement)
                    and _pyamplicol_artifacts_current(
                        measurement,
                        require_current_compiled_model_contract=_cell_uses_external_model(
                            cell
                        ),
                    )
                )
            return False
        if entry.get("n_final") != cell.n_final:
            continue
        measurement = entry.get("measurement")
        if not isinstance(measurement, Mapping):
            return True
        status = str(measurement.get("status", NA_STATUS))
        if status == NA_STATUS:
            return True
        if status in {
            ResultStatus.ERROR.value,
            ResultStatus.FAILED.value,
            ResultStatus.MEMORY_LIMIT.value,
            ResultStatus.TIMEOUT.value,
            ResultStatus.UNSUPPORTED.value,
            ResultStatus.VALIDATION_FAILED.value,
        }:
            return True
        if status == ResultStatus.OK.value:
            return not (
                _pyamplicol_generation_profile_current(measurement)
                and _pyamplicol_artifacts_current(
                    measurement,
                    require_current_compiled_model_contract=_cell_uses_external_model(
                        cell
                    ),
                )
            )
        return False


def _cell_uses_external_model(cell: CampaignCell) -> bool:
    if cell.kind == "model_ladder":
        return True
    return cell.dataset_id.startswith("matrix_external_sm") or (
        cell.dataset_id == "z_external_sm"
    )


def _pyamplicol_timing_profile_current(measurement: Mapping[str, object]) -> bool:
    environment = measurement.get("environment")
    if not isinstance(environment, Mapping):
        return False
    return (
        environment.get("wall_time_source") == "runtime_core_repeated_wall_time"
        and environment.get("evaluator_time_source")
        == "runtime_profile_core_evaluator_call_time"
    )


def _pyamplicol_generation_profile_current(
    measurement: Mapping[str, object],
) -> bool:
    metadata = measurement.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return (
        metadata.get("model_precompile_policy")
        == PYAMPLICOL_GENERATION_PROFILE_POLICY
        and metadata.get("generation_timer_excludes_model_compile") is True
    )


@cache
def _current_compiled_model_contract() -> tuple[int, int]:
    _ensure_repo_root_on_path()
    from pyamplicol._internal.versions import COMPILED_MODEL_SCHEMA_VERSION
    from pyamplicol.models.loading import MODEL_COMPILER_VERSION

    return int(COMPILED_MODEL_SCHEMA_VERSION), int(MODEL_COMPILER_VERSION)


def _compiled_model_cache_dir(artifact_root: Path) -> Path:
    schema_version, compiler_version = _current_compiled_model_contract()
    return (
        artifact_root
        / f"model-cache-schema{schema_version}-compiler{compiler_version}"
    )


def _pyamplicol_artifact_subdir(artifact_subdir: str) -> Path:
    schema_version, compiler_version = _current_compiled_model_contract()
    parts = Path(artifact_subdir).parts
    if parts and parts[0] == "pyamplicol":
        root = Path(
            f"pyamplicol-schema{schema_version}-compiler{compiler_version}"
        )
        return root.joinpath(*parts[1:])
    return Path(artifact_subdir)


def _pyamplicol_artifacts_current(
    measurement: Mapping[str, object],
    *,
    require_current_compiled_model_contract: bool,
) -> bool:
    paths = tuple(_measurement_artifact_paths(measurement))
    if not paths:
        return False
    return all(
        _artifact_compiled_model_current(
            path,
            require_current_compiled_model_contract=require_current_compiled_model_contract,
        )
        for path in paths
    )


def _previous_cache_entry_for_cell(cell: CampaignCell) -> Mapping[str, object] | None:
    try:
        payload = load_caches(ReportPaths.default()).get(cell.cache_name)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if cell.kind == "matrix":
            if (
                entry.get("process_key") == cell.process_key
                and entry.get("n_final") == cell.n_final
            ):
                return entry
        elif cell.kind == "performance_ladder":
            if (
                entry.get("n_final") == cell.n_final
                and entry.get("variant") == cell.variant
            ):
                return entry
        elif entry.get("n_final") == cell.n_final:
            return entry
    return None


def _reusable_pyamplicol_generation_seconds(
    cell: CampaignCell,
    artifact_dir: Path,
    previous_measurement: Mapping[str, object] | None,
) -> float | None:
    if previous_measurement is None:
        return None
    previous_generation_seconds = _optional_positive_float(
        previous_measurement.get("generation_seconds")
    )
    if previous_generation_seconds is None:
        return None
    if not _pyamplicol_generation_profile_current(previous_measurement):
        return None
    if not (artifact_dir / "artifact.json").is_file():
        return None
    if not _artifact_compiled_model_current(
        artifact_dir,
        require_current_compiled_model_contract=_cell_uses_external_model(cell),
    ):
        return None
    return previous_generation_seconds


def _measurement_artifact_paths(
    value: object,
    *,
    _seen: set[int] | None = None,
) -> Iterator[Path]:
    if _seen is None:
        _seen = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            return
        _seen.add(marker)
        raw_path = value.get("artifact_path")
        if isinstance(raw_path, str) and raw_path:
            yield Path(raw_path)
        metadata = value.get("metadata")
        if isinstance(metadata, Mapping):
            yield from _measurement_artifact_paths(metadata, _seen=_seen)
        for key in ("selected_flow_measurement", "all_flow_measurement"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                yield from _measurement_artifact_paths(nested, _seen=_seen)


def _artifact_compiled_model_current(
    artifact_path: Path,
    *,
    require_current_compiled_model_contract: bool,
) -> bool:
    model_path = artifact_path / "model" / "compiled-model.json"
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    if not isinstance(payload, Mapping):
        return False
    schema_version, compiler_version = _current_compiled_model_contract()
    if require_current_compiled_model_contract:
        return (
            payload.get("schema_version") == schema_version
            and payload.get("model_compiler_version") == compiler_version
        )
    payload_schema = payload.get("schema_version")
    return payload_schema == schema_version or (
        schema_version == 9
        and payload_schema in BUILTIN_SM_COMPATIBLE_COMPILED_MODEL_SCHEMAS
    )


def _legacy_measurement_revision_current(
    measurement: Mapping[str, object],
) -> bool:
    environment = measurement.get("environment")
    if not isinstance(environment, Mapping):
        return False
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    revision = environment.get("revision")
    expected = legacy_amplicol.expected_revision()
    if revision == expected:
        return True
    return (
        isinstance(revision, str)
        and revision in LEGACY_REFERENCE_COMPATIBLE_REVISIONS
        and expected in LEGACY_REFERENCE_COMPATIBLE_REVISIONS
    )


def _legacy_profile_requested_config(target_runtime: float) -> dict[str, object]:
    return {
        "legacy_profile_policy": LEGACY_PROFILE_POLICY,
        "legacy_profile_target_seconds": float(target_runtime),
        "legacy_profile_warmup_points": DEFAULT_LEGACY_PROFILE_WARMUP_POINTS,
        "legacy_profile_min_points": DEFAULT_LEGACY_PROFILE_MIN_POINTS,
        "legacy_profile_max_points": DEFAULT_LEGACY_PROFILE_MAX_POINTS,
    }


def _legacy_measurement_profile_current(
    measurement: Mapping[str, object],
) -> bool:
    requested_config = measurement.get("requested_config")
    if not isinstance(requested_config, Mapping):
        return False
    if requested_config.get("legacy_profile_policy") != LEGACY_PROFILE_POLICY:
        return False
    target = _optional_positive_float(
        requested_config.get("legacy_profile_target_seconds")
    )
    if target is None:
        return False
    for key, expected in (
        ("legacy_profile_warmup_points", DEFAULT_LEGACY_PROFILE_WARMUP_POINTS),
        ("legacy_profile_min_points", DEFAULT_LEGACY_PROFILE_MIN_POINTS),
        ("legacy_profile_max_points", DEFAULT_LEGACY_PROFILE_MAX_POINTS),
    ):
        value = requested_config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            return False
    return True


def _legacy_lc_measurement_contract_current(
    measurement: Mapping[str, object],
) -> bool:
    if not _legacy_measurement_profile_current(measurement):
        return False
    fields = _measurement_old_matrix_fields(measurement)
    if not fields:
        return False
    all_flow_status = str(fields.get("all_flow_status", NA_STATUS))
    if all_flow_status == ResultStatus.OK.value:
        if (
            fields.get("all_flow_generation_source")
            != LEGACY_LC_ALL_FLOW_GENERATION_SOURCE
        ):
            return False
        selected_generation = _optional_positive_float(
            fields.get("generation_s", measurement.get("generation_seconds"))
        )
        all_flow_generation = _optional_positive_float(
            fields.get("all_flow_generation_s")
        )
        if selected_generation is None or all_flow_generation is None:
            return False
        if not math.isclose(
            selected_generation,
            all_flow_generation,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            return False
    return True


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def load_caches(paths: ReportPaths | None = None) -> dict[str, dict[str, object]]:
    selected_paths = paths or ReportPaths.default()
    caches: dict[str, dict[str, object]] = {}
    names = [spec.cache_name for spec in (*MATRIX_SPECS, *LADDER_SPECS)]
    for name in names:
        path = selected_paths.results_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        payload = normalize_cache_payload(payload)
        validate_cache(payload)
        caches[name] = payload
    return caches


class ReportGenerationTimeout(TimeoutError):
    """Raised when a report cell exceeds its generation time budget."""


@contextmanager
def _generation_timeout(seconds: float | None) -> Iterable[None]:
    if seconds is None or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise ReportGenerationTimeout(
            f"generation exceeded {float(seconds):.0f} seconds"
        )

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _set_nested(mapping: dict[str, object], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _run_config_values(
    *,
    model: ModelSpec,
    color_accuracy: str,
    variant_overrides: Mapping[str, object],
    process_overrides: Mapping[str, object] | None = None,
    benchmark_overrides: Mapping[str, object] | None = None,
    artifact_root: Path | None = None,
    target_runtime: float,
    cell_cores: int,
) -> dict[str, object]:
    model_path = _model_source_path(model)
    model_config: dict[str, object] = {
        "source": "built-in-sm" if model_path is None else os.fspath(model_path),
        "cache": True,
    }
    if model_path is not None and artifact_root is not None:
        model_config["cache_dir"] = os.fspath(_compiled_model_cache_dir(artifact_root))
    values: dict[str, object] = {
        "model": model_config,
        "color": {"accuracy": color_accuracy},
        "generation": {
            "workers": max(1, cell_cores),
            "emit_api_bundle": True,
            "validation": {
                "enabled": True,
                "samples": 10,
                "seed": 12345,
                "relative_tolerance": 1.0e-12,
                "absolute_tolerance": 1.0e-300,
                "post_build_validation": True,
            },
        },
        "evaluator": {
            "backend": "jit",
            "batch_size": 128,
            "output_chunk_size": 128,
            "optimization": {
                "horner_iterations": 10,
                "cpe_iterations": None,
                "cores": max(1, cell_cores),
                "max_horner_variables": 1000,
                "max_common_pair_cache_entries": 5_000_000,
                "max_common_pair_distance": 1000,
            },
            "jit": {"optimization_level": 3},
            "cpp": {"optimization": "O3"},
        },
        "benchmark": {
            "target_runtime": target_runtime,
            "batch_size": 128,
            "warmup_runs": 2,
            "minimum_samples": 5,
        },
        "output": {"format": "json", "progress": "off"},
    }
    overrides = {
        **REPORT_CONFIG_OVERRIDES,
        **(process_overrides or {}),
        **(benchmark_overrides or {}),
        **variant_overrides,
    }
    for dotted, value in overrides.items():
        if dotted == "benchmark.target_runtime":
            value = target_runtime
        _set_nested(values, str(dotted), value)
    _set_nested(values, "generation.workers", max(1, cell_cores))
    _set_nested(values, "evaluator.optimization.cores", max(1, cell_cores))
    return values


def _runtime_validation_momenta(runtime: object) -> object | None:
    backend = getattr(runtime, "_backend", None)
    loader = getattr(backend, "validation_momenta", None)
    if callable(loader):
        return loader()
    return None


def _single_artifact_process_id(artifact_dir: Path, fallback: str) -> str:
    manifest_path = artifact_dir / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    processes = manifest.get("processes")
    if not isinstance(processes, list) or len(processes) != 1:
        return fallback
    process = processes[0]
    if not isinstance(process, Mapping):
        return fallback
    process_id = process.get("id")
    return process_id if isinstance(process_id, str) and process_id else fallback


def _real_nonnegative_scalar(value: object) -> float:
    if isinstance(value, complex):
        if abs(value.imag) > 1.0e-9 * max(abs(value.real), 1.0):
            raise ValueError(
                f"matrix element has non-negligible imaginary part {value}"
            )
        number = float(value.real)
    else:
        number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError("matrix element is not finite")
    return abs(number)


def _model_source_for_api(model: ModelSpec) -> object:
    from pyamplicol.api import ModelSource

    model_path = _model_source_path(model)
    if model_path is None:
        return ModelSource.built_in_sm()
    return ModelSource.from_path(model_path)


def _precompile_model_for_generation(
    model: ModelSpec,
    config_values: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    source = _model_source_for_api(model)
    model_config = config_values.get("model")
    cache_dir: Path | None = None
    use_cache = True
    if isinstance(model_config, Mapping):
        raw_cache_dir = model_config.get("cache_dir")
        if isinstance(raw_cache_dir, str) and raw_cache_dir:
            cache_dir = Path(raw_cache_dir).expanduser().resolve(strict=False)
        raw_use_cache = model_config.get("cache")
        if isinstance(raw_use_cache, bool):
            use_cache = raw_use_cache
    compile_model = getattr(source, "compile", None)
    if not callable(compile_model):
        return source, {
            "model_precompile_policy": PYAMPLICOL_GENERATION_PROFILE_POLICY,
            "model_precompile_seconds": 0.0,
            "model_precompile_cache_dir": (
                None if cache_dir is None else os.fspath(cache_dir)
            ),
            "model_precompile_used_cache": use_cache,
            "model_precompile_source_kind": None,
            "generation_timer_excludes_model_compile": True,
        }
    started = time.perf_counter()
    compiled = compile_model(
        cache_dir=cache_dir,
        use_cache=use_cache,
        require_supported=True,
    )
    precompile_seconds = time.perf_counter() - started
    return compiled, {
        "model_precompile_policy": PYAMPLICOL_GENERATION_PROFILE_POLICY,
        "model_precompile_seconds": precompile_seconds,
        "model_precompile_cache_dir": (
            None if cache_dir is None else os.fspath(cache_dir)
        ),
        "model_precompile_used_cache": use_cache,
        "model_precompile_source_kind": getattr(source, "kind", None),
        "generation_timer_excludes_model_compile": True,
    }


def _model_resolved_process_ir(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec,
    color_accuracy: str,
    artifact_root: Path,
) -> object:
    if spec.model is BUILTIN_SM or spec.model.source_kind == "built-in-sm":
        from pyamplicol.models.builtin.process_ir import build_process_ir

        return build_process_ir(process, color_accuracy=color_accuracy)

    from pyamplicol.api import ProcessSet
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import resolve_config
    from pyamplicol.generation.service import GenerationBackend

    config = _run_config_values(
        model=spec.model,
        color_accuracy=color_accuracy,
        variant_overrides={
            "evaluator.backend": "jit",
            "evaluator.jit.optimization_level": 3,
        },
        artifact_root=artifact_root,
        target_runtime=1.0,
        cell_cores=1,
    )
    resolution = resolve_config(
        config,
        action=Action.GENERATE,
        base_dir=_repo_root(),
    )
    backend = GenerationBackend(resolution, None)
    resolved = backend._resolve_model(_model_source_for_api(spec.model))
    expanded = backend._expand_process_set(
        ProcessSet.from_expressions((process,)),
        resolved,
        _NullGenerationPhase(),
    )
    if not expanded:
        raise RuntimeError(
            f"model process expansion produced no process for {process!r}"
        )
    if len(expanded) != 1:
        raise RuntimeError(
            f"model process expansion produced {len(expanded)} processes for "
            f"{process!r}; report cells require one concrete process"
        )
    return expanded[0].process_ir


def _reference_coloured_word(
    color_plan: object,
    reference_order: Sequence[int],
) -> tuple[int, ...]:
    coloured = set(getattr(color_plan, "coloured_labels", ()) or ())
    return tuple(int(label) for label in reference_order if int(label) in coloured)


def _lc_sector_preserves_reference_singlet_blocks(
    sector: object,
    reference_order: Sequence[int],
) -> bool:
    if getattr(sector, "kind", "") != "open-lines":
        return True
    coloured_labels = set(getattr(sector, "word_labels", ()) or ())
    if not coloured_labels:
        return True
    blocks: list[set[int]] = []
    current: list[int] = []
    for raw_label in reference_order:
        label = int(raw_label)
        if label in coloured_labels:
            if current:
                blocks.append(set(current))
                current = []
            continue
        current.append(label)
    if current:
        blocks.append(set(current))
    if not blocks:
        return True
    line_singlets = [
        set(getattr(line, "singlet_labels", ()) or ())
        for line in getattr(sector, "open_color_lines", ())
        if getattr(line, "singlet_labels", ()) or ()
    ]
    if not line_singlets:
        sector_singlets = set(getattr(sector, "singlet_labels", ()) or ())
        return all(block.issubset(sector_singlets) for block in blocks)
    return all(
        any(block.issubset(singlets) for singlets in line_singlets) for block in blocks
    )


def _lc_colored_word_sibling_sector_ids(
    color_plan: object,
    sector: object,
    *,
    reference_order: Sequence[int] | None = None,
) -> set[int]:
    word = tuple(getattr(sector, "word_labels", ()) or ())
    if not word:
        return {int(sector.id)}
    siblings: set[int] = set()
    for candidate in getattr(color_plan, "sectors", ()) or ():
        if tuple(getattr(candidate, "word_labels", ()) or ()) != word:
            continue
        if (
            reference_order is not None
            and not _lc_sector_preserves_reference_singlet_blocks(
                candidate,
                reference_order,
            )
        ):
            continue
        siblings.add(int(candidate.id))
    return siblings or {int(sector.id)}


def _selected_lc_sector_ids_for_reference_order(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec,
    reference_order: Sequence[int] | None,
    artifact_root: Path,
) -> set[int] | None:
    from pyamplicol.color.plan import build_color_plan

    process_ir = _model_resolved_process_ir(
        process,
        spec=spec,
        color_accuracy="lc",
        artifact_root=artifact_root,
    )
    color_plan = build_color_plan(
        process_ir,
        color_accuracy="lc",
        max_sectors=None,
        reference_color_order=reference_order,
    )
    if color_plan.color_accuracy != "lc" or color_plan.sector_count <= 1:
        return None
    if reference_order is not None:
        wanted = tuple(int(label) for label in reference_order)
        wanted_coloured = _reference_coloured_word(color_plan, wanted)
        if wanted_coloured:
            for sector in color_plan.sectors:
                if tuple(getattr(sector, "word_labels", ()) or ()) == wanted_coloured:
                    if wanted_coloured == wanted:
                        return {int(sector.id)}
                    return _lc_colored_word_sibling_sector_ids(
                        color_plan,
                        sector,
                        reference_order=wanted,
                    )
        for sector in color_plan.sectors:
            if wanted in sector.color_words:
                if wanted_coloured == wanted:
                    return {int(sector.id)}
                return _lc_colored_word_sibling_sector_ids(
                    color_plan,
                    sector,
                    reference_order=wanted,
                )
            if wanted in getattr(sector, "legacy_order_words", ()):
                if wanted_coloured == wanted:
                    return {int(sector.id)}
                return _lc_colored_word_sibling_sector_ids(
                    color_plan,
                    sector,
                    reference_order=wanted,
                )
        for sector in color_plan.sectors:
            if wanted in getattr(sector, "admissible_traversal_words", ()):
                return _lc_colored_word_sibling_sector_ids(color_plan, sector)
        raise ValueError(
            "LC reference colour order does not match any generated colour sector: "
            f"{wanted}"
        )
    return {0}


def _selected_lc_reference_partition_words(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec,
    reference_order: Sequence[int],
    artifact_root: Path,
) -> tuple[tuple[int, ...], ...]:
    from pyamplicol.color.plan import build_color_plan

    process_ir = _model_resolved_process_ir(
        process,
        spec=spec,
        color_accuracy="lc",
        artifact_root=artifact_root,
    )
    color_plan = build_color_plan(
        process_ir,
        color_accuracy="lc",
        max_sectors=None,
        reference_color_order=reference_order,
    )
    selected_ids = _selected_lc_sector_ids_for_reference_order(
        process,
        spec=spec,
        reference_order=reference_order,
        artifact_root=artifact_root,
    )
    wanted_coloured = _reference_coloured_word(
        color_plan,
        tuple(int(label) for label in reference_order),
    )
    words: list[tuple[int, ...]] = []
    for sector in color_plan.sectors:
        if selected_ids is not None and int(sector.id) not in selected_ids:
            continue
        for raw_word in (
            getattr(sector, "word_labels", ()) or (),
            *getattr(sector, "color_words", ()),
            *getattr(sector, "legacy_order_words", ()),
            *getattr(sector, "admissible_traversal_words", ()),
        ):
            word = tuple(int(label) for label in raw_word)
            if word and word not in words:
                words.append(word)
    if wanted_coloured and wanted_coloured not in words:
        words.insert(0, wanted_coloured)
    return tuple(words or (tuple(int(label) for label in reference_order),))


def _generation_slice_snapshot(selection: object | None) -> dict[str, object] | None:
    if selection is None:
        return None
    selected_helicities = getattr(selection, "selected_source_helicities", None)
    return {
        "reference_color_order": [
            int(label) for label in getattr(selection, "reference_color_order", ())
        ],
        "selected_color_sector_ids": [
            int(sector_id)
            for sector_id in getattr(selection, "selected_color_sector_ids", ())
        ],
        "selected_source_helicities": (
            None
            if not selected_helicities
            else {
                str(label): int(helicity)
                for label, helicity in selected_helicities.items()
            }
        ),
    }


def _measure_pyamplicol(
    *,
    cell: CampaignCell,
    spec: MatrixSpec | LadderSpec,
    color_accuracy: str,
    variant_overrides: Mapping[str, object],
    process_overrides: Mapping[str, object] | None = None,
    benchmark_overrides: Mapping[str, object] | None = None,
    artifact_root: Path,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
    generation_slice: object | None = None,
    high_precision: bool = False,
    points_override: object | None = None,
    artifact_subdir: str = "pyamplicol",
    log_name: str = "pyamplicol.log",
    previous_measurement: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], object | None]:
    from pyamplicol.api import BenchmarkRunner, CompatibilityError, Generator, Runtime
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import config_to_dict, resolve_config

    cell_root = artifact_root / "cells" / cell.cell_id
    artifact_dir = cell_root / _pyamplicol_artifact_subdir(artifact_subdir)
    log_path = cell_root / "logs" / log_name
    manifest_path = artifact_dir / "manifest.json"
    snapshot_path = cell_root / "inputs" / "pyamplicol-inputs.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    reusable_generation_seconds = _reusable_pyamplicol_generation_seconds(
        cell,
        artifact_dir,
        previous_measurement,
    )
    artifact_reusable = reusable_generation_seconds is not None
    config_values = _run_config_values(
        model=spec.model,
        color_accuracy=color_accuracy,
        variant_overrides=variant_overrides,
        process_overrides=process_overrides,
        benchmark_overrides=benchmark_overrides,
        artifact_root=artifact_root,
        target_runtime=target_runtime,
        cell_cores=cell_cores,
    )
    resolution = resolve_config(
        config_values,
        action=Action.GENERATE,
        base_dir=_repo_root(),
    )
    snapshot = {
        "cell": cell.as_json(),
        "model": spec.model.as_json(),
        "model_source": config_values["model"],
        "color_accuracy": color_accuracy,
        "variant_overrides": dict(variant_overrides),
        "process_overrides": dict(process_overrides or {}),
        "benchmark_overrides": dict(benchmark_overrides or {}),
        "generation_slice": _generation_slice_snapshot(generation_slice),
        "target_runtime": target_runtime,
        "cell_cores": cell_cores,
        "artifact_reused_for_timing": artifact_reusable,
        "source_provenance": _report_source_provenance(),
        "captured_at": _utc_now(),
    }
    snapshot_path.write_text(_json_text(snapshot), encoding="utf-8")
    command = [
        os.fspath(DEFAULT_DEV_PYTHON),
        "docs/result_tables.py",
        "measure-cell",
        "--dataset-id",
        cell.dataset_id,
        "--n-final",
        str(cell.n_final),
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# pyAmpliCol cell {cell.cell_id} started {_utc_now()}\n")
            log.flush()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model_for_generation: object | None = None
                model_precompile_metadata: dict[str, object] = {
                    "model_precompile_policy": PYAMPLICOL_GENERATION_PROFILE_POLICY,
                    "model_precompile_seconds": None,
                    "model_precompile_cache_dir": None,
                    "model_precompile_used_cache": None,
                    "model_precompile_source_kind": None,
                    "generation_timer_excludes_model_compile": True,
                    "model_precompile_skipped": "artifact_reused_for_timing",
                }
                if artifact_reusable:
                    generation_seconds = float(reusable_generation_seconds)
                else:
                    model_for_generation, model_precompile_metadata = (
                        _precompile_model_for_generation(spec.model, config_values)
                    )
                    started = time.perf_counter()
                    with _generation_timeout(generation_timeout_seconds):
                        if generation_slice is None:
                            Generator(resolution).generate(
                                cell.process,
                                artifact_dir,
                                model=model_for_generation,
                                mode="replace",
                            )
                        else:
                            _GenerationSlice, generate_slice = _generation_slice_tools()

                            generate_slice(
                                cell.process,
                                artifact_dir,
                                selection=generation_slice,  # type: ignore[arg-type]
                                model=model_for_generation,
                                mode="replace",
                                config=resolution,
                            )
                    generation_seconds = time.perf_counter() - started
                runtime_process = _single_artifact_process_id(
                    artifact_dir,
                    fallback=cell.process,
                )
                runtime = Runtime.load(artifact_dir, process=runtime_process)
                points = (
                    points_override
                    if points_override is not None
                    else _runtime_validation_momenta(runtime)
                )
                values = runtime.evaluate(points) if points is not None else ()
                matrix_element = _real_nonnegative_scalar(values[0]) if values else None
                high_precision_value: float | None = None
                high_precision_relative_difference: float | None = None
                if high_precision and points is not None:
                    precise = runtime.evaluate(points, precision=32)
                    if precise:
                        high_precision_value = _real_nonnegative_scalar(precise[0])
                        high_precision_relative_difference = _safe_divide(
                            abs((matrix_element or 0.0) - high_precision_value),
                            max(abs(high_precision_value), 1.0e-300),
                        )
                benchmark = BenchmarkRunner(resolution.effective).run(
                    runtime,
                    points=points,  # type: ignore[arg-type]
                )
        observation = BenchmarkObservation.from_result(benchmark).as_cache_fields()
        metadata = {
            "cell": cell.as_json(),
            "model_source": snapshot["model_source"],
            "input_snapshot_path": os.fspath(snapshot_path),
            "runtime_process": runtime_process,
            "artifact_reused_for_timing": artifact_reusable,
            "generation_seconds_source": (
                "previous_measurement" if artifact_reusable else "fresh_generation"
            ),
            "source_provenance": _report_source_provenance(),
            "high_precision_matrix_element": high_precision_value,
            "high_precision_relative_difference": high_precision_relative_difference,
            "generation_slice": snapshot["generation_slice"],
            **model_precompile_metadata,
        }
        measurement = {
            **_empty_measurement(),
            **observation,
            "status": ResultStatus.OK.value,
            "generation_seconds": generation_seconds,
            "matrix_element": matrix_element,
            "requested_config": config_to_dict(resolution.requested),
            "effective_config": config_to_dict(resolution.effective),
            "artifact_path": os.fspath(artifact_dir),
            "log_path": os.fspath(log_path),
            "manifest_path": os.fspath(manifest_path),
            "limit_gib": None,
            "timeout_seconds": generation_timeout_seconds,
            "command": command,
            "metadata": metadata,
        }
        manifest = {
            "cell": cell.as_json(),
            "measurement": measurement,
            "input_snapshot_path": os.fspath(snapshot_path),
            "source_provenance": _report_source_provenance(),
            "captured_at": _utc_now(),
        }
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        return measurement, points
    except ReportGenerationTimeout as exc:
        return (
            _failure_measurement(
                ResultStatus.TIMEOUT,
                str(exc),
                failure_kind="generation_timeout",
                artifact_path=artifact_dir,
                log_path=log_path,
                manifest_path=manifest_path,
                timeout_seconds=generation_timeout_seconds,
                command=command,
                metadata={
                    "cell": cell.as_json(),
                    "input_snapshot_path": os.fspath(snapshot_path),
                    "source_provenance": _report_source_provenance(),
                },
            ),
            None,
        )
    except Exception as exc:
        status = (
            ResultStatus.UNSUPPORTED
            if isinstance(exc, CompatibilityError)
            else ResultStatus.ERROR
        )
        return (
            _failure_measurement(
                status,
                str(exc),
                failure_kind=type(exc).__name__,
                artifact_path=artifact_dir,
                log_path=log_path,
                manifest_path=manifest_path,
                timeout_seconds=generation_timeout_seconds,
                command=command,
                metadata={
                    "cell": cell.as_json(),
                    "input_snapshot_path": os.fspath(snapshot_path),
                    "source_provenance": _report_source_provenance(),
                },
            ),
            None,
        )


def _measurement_old_matrix_fields(
    measurement: Mapping[str, object],
) -> Mapping[str, object]:
    metadata = measurement.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    fields = metadata.get("old_matrix_format")
    return fields if isinstance(fields, Mapping) else {}


def _measure_pyamplicol_lc_two_workloads(
    *,
    cell: CampaignCell,
    spec: MatrixSpec | LadderSpec,
    variant_overrides: Mapping[str, object],
    legacy: Mapping[str, object] | None,
    artifact_root: Path,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
    points: object | None,
    fixed_helicity: Mapping[str, object] | None = None,
    previous_measurement: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], object | None]:
    GenerationSlice, _generate_slice = _generation_slice_tools()

    old_legacy = _measurement_old_matrix_fields(legacy or {})
    previous_metadata = (
        previous_measurement.get("metadata")
        if isinstance(previous_measurement, Mapping)
        else None
    )
    previous_selected = (
        previous_metadata.get("selected_flow_measurement")
        if isinstance(previous_metadata, Mapping)
        else None
    )
    previous_all_flow = (
        previous_metadata.get("all_flow_measurement")
        if isinstance(previous_metadata, Mapping)
        else None
    )
    reference_order = _selected_flow_reference_color_order(old_legacy, cell)
    selected_slice: object | None = None
    if isinstance(reference_order, list) and reference_order:
        normalized_reference_order = [int(label) for label in reference_order]
        selected_sector_ids = _selected_lc_sector_ids_for_reference_order(
            cell.process,
            spec=spec,
            reference_order=normalized_reference_order,
            artifact_root=artifact_root,
        )
        selected_slice = GenerationSlice(
            reference_color_order=tuple(normalized_reference_order),
            selected_color_sector_ids=tuple(sorted(selected_sector_ids or ())),
        )
    selected, selected_points = _measure_pyamplicol(
        cell=cell,
        spec=spec,
        color_accuracy="lc",
        variant_overrides=variant_overrides,
        benchmark_overrides={
            "benchmark.batch_size": 64,
            "evaluator.batch_size": 64,
        },
        artifact_root=artifact_root,
        generation_timeout_seconds=generation_timeout_seconds,
        target_runtime=target_runtime,
        cell_cores=cell_cores,
        generation_slice=selected_slice,
        points_override=points,
        artifact_subdir="pyamplicol/selected-flow",
        log_name="pyamplicol-selected-flow.log",
        previous_measurement=(
            previous_selected if isinstance(previous_selected, Mapping) else None
        ),
    )

    if fixed_helicity is None:
        fixed_helicity = _fixed_source_helicity_choice(
            cell.process,
            spec=spec,
            artifact_root=artifact_root,
        )
    all_flow: dict[str, object]
    if _lc_all_flow_supported(cell.process):
        all_flow, _all_flow_points = _measure_pyamplicol(
            cell=cell,
            spec=spec,
            color_accuracy="lc",
            variant_overrides=variant_overrides,
            benchmark_overrides={
                "benchmark.batch_size": 64,
                "evaluator.batch_size": 64,
                "evaluator.output_chunk_size": 8192,
            },
            artifact_root=artifact_root,
            generation_timeout_seconds=generation_timeout_seconds,
            target_runtime=target_runtime,
            cell_cores=cell_cores,
            generation_slice=GenerationSlice(
                selected_source_helicities={
                    int(label): int(helicity)
                    for label, helicity in fixed_helicity[
                        "source_helicities"
                    ].items()
                }
            ),
            points_override=points,
            artifact_subdir="pyamplicol/all-flows",
            log_name="pyamplicol-all-flows.log",
            previous_measurement=(
                previous_all_flow if isinstance(previous_all_flow, Mapping) else None
            ),
        )
    else:
        all_flow_artifact_dir = (
            artifact_root
            / "cells"
            / cell.cell_id
            / _pyamplicol_artifact_subdir("pyamplicol/all-flows")
        )
        all_flow = _failure_measurement(
            ResultStatus.UNSUPPORTED,
            "LC all-flow fixed-helicity timing is unsupported for more than "
            "two quark lines",
            artifact_path=all_flow_artifact_dir,
            metadata={"cell": cell.as_json()},
        )

    combined = dict(selected)
    metadata = dict(
        combined.get("metadata")
        if isinstance(combined.get("metadata"), Mapping)
        else {}
    )
    jit_level = variant_overrides.get("evaluator.jit.optimization_level")
    backend = variant_overrides.get("evaluator.backend", "jit")
    old_fields = {
        "status": selected.get("status"),
        "generation_s": selected.get("generation_seconds"),
        "selected_generation_s": selected.get("generation_seconds"),
        "runtime_us_per_point": (
            None
            if selected.get("evaluator_seconds_per_point") is None
            else 1.0e6 * float(selected["evaluator_seconds_per_point"])
        ),
        "wall_us_per_point": (
            None
            if selected.get("wall_seconds_per_point") is None
            else 1.0e6 * float(selected["wall_seconds_per_point"])
        ),
        "selected_backend": backend,
        "selected_jit_optimization_level": jit_level,
        "selected_output_dir": selected.get("artifact_path"),
        "all_flow_status": all_flow.get("status"),
        "all_flow_generation_s": all_flow.get("generation_seconds"),
        "all_flow_runtime_us_per_point": (
            None
            if all_flow.get("evaluator_seconds_per_point") is None
            else 1.0e6 * float(all_flow["evaluator_seconds_per_point"])
        ),
        "all_flow_matrix_element": all_flow.get("matrix_element"),
        "all_flow_wall_us_per_point": (
            None
            if all_flow.get("wall_seconds_per_point") is None
            else 1.0e6 * float(all_flow["wall_seconds_per_point"])
        ),
        "all_flow_backend": backend,
        "all_flow_jit_optimization_level": jit_level,
        "all_flow_output_dir": all_flow.get("artifact_path"),
        "all_flow_error": all_flow.get("failure_message"),
        "all_flow_helicity_mode": fixed_helicity["mode"],
        "all_flow_helicity_selection_source": fixed_helicity.get(
            "selection_source"
        ),
        "all_flow_source_helicities": fixed_helicity["source_helicities"],
        "all_flow_amplicol_helicities": fixed_helicity["amplicol_helicities"],
        "all_flow_validation_note": fixed_helicity["validation_note"],
    }
    metadata["old_matrix_format"] = old_fields
    metadata["selected_flow_measurement"] = selected
    metadata["all_flow_measurement"] = all_flow
    combined["metadata"] = metadata
    return combined, selected_points


def _selected_flow_reference_color_order(
    old_legacy: Mapping[str, object],
    cell: CampaignCell,
) -> object:
    reference_order = old_legacy.get("reference_color_order")
    if not isinstance(reference_order, list):
        reference_order = old_legacy.get("reference_color_order_process_file")
    if not isinstance(reference_order, list) and cell.dataset_id.startswith("z_"):
        reference_order = _z_reference_color_order_for_n(cell.n_final)
    return reference_order


def _measure_pyamplicol_matrix_jit_o3(
    *,
    cell: CampaignCell,
    spec: MatrixSpec,
    legacy: Mapping[str, object],
    artifact_root: Path,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
    points: object | None,
    fixed_helicity: Mapping[str, object] | None = None,
    previous_measurement: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], object | None]:
    variant = {
        "evaluator.backend": "jit",
        "evaluator.jit.optimization_level": 3,
    }
    if spec.color_accuracy != "lc":
        return _measure_pyamplicol(
            cell=cell,
            spec=spec,
            color_accuracy=spec.color_accuracy,
            variant_overrides=variant,
            artifact_root=artifact_root,
            generation_timeout_seconds=generation_timeout_seconds,
            target_runtime=target_runtime,
            cell_cores=cell_cores,
            points_override=points,
            previous_measurement=previous_measurement,
        )
    return _measure_pyamplicol_lc_two_workloads(
        cell=cell,
        spec=spec,
        variant_overrides=variant,
        legacy=legacy,
        artifact_root=artifact_root,
        generation_timeout_seconds=generation_timeout_seconds,
        target_runtime=target_runtime,
        cell_cores=cell_cores,
        points=points,
        fixed_helicity=fixed_helicity,
        previous_measurement=previous_measurement,
    )


def _legacy_momenta_from_pyamplicol(
    points: object,
) -> tuple[tuple[float, float, float, float], ...] | None:
    if not isinstance(points, tuple) or not points:
        return None
    first = points[0]
    if not isinstance(first, tuple):
        return None
    rows: list[tuple[float, float, float, float]] = []
    for row in first:
        if not isinstance(row, tuple) or len(row) != 4:
            return None
        rows.append(
            tuple(float(component) for component in row)  # type: ignore[arg-type]
        )
    return tuple(rows)


def _shared_validation_particles(process: str) -> tuple[object, ...]:
    from pyamplicol.models.builtin.validation import generic_validation_point

    return tuple(generic_validation_point(process))


def _pyamplicol_points_from_particles(particles: Sequence[object]) -> object:
    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in particles
        ),
    )


def _legacy_momenta_from_particles(
    particles: Sequence[object],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in particles
    )


def _legacy_pdgs_from_particles(particles: Sequence[object]) -> tuple[int, ...]:
    return tuple(int(particle.pdg) for particle in particles)


class _NullGenerationPhase:
    def update(self, *_args: object, **_kwargs: object) -> None:
        pass

    def advance(self, *_args: object, **_kwargs: object) -> None:
        pass


def _fixed_source_helicity_choice(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec | None = None,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    fallback = _alternating_fixed_source_helicity_choice(process)
    if spec is None or artifact_root is None:
        return fallback
    try:
        if _source_helicity_choice_has_amplitudes(
            process,
            fallback["source_helicities"],
            spec=spec,
            artifact_root=artifact_root,
        ):
            return {
                **fallback,
                "selection_source": "pyamplicol-dag-selected-helicity-probe",
            }
        chiral = _chiral_fermion_fixed_source_helicity_choice(process)
        if (
            chiral is not None
            and _source_helicity_choice_has_amplitudes(
                process,
                chiral["source_helicities"],
                spec=spec,
                artifact_root=artifact_root,
            )
        ):
            return chiral
        supported = _first_model_supported_source_helicity_choice(
            process,
            spec=spec,
            artifact_root=artifact_root,
            preferred=fallback["source_helicities"],
        )
        if supported is not None:
            return supported
    except Exception as exc:
        note = str(fallback["validation_note"])
        return {
            **fallback,
            "selection_source": "alternating-fallback-after-dag-probe-error",
            "selection_error": str(exc),
            "validation_note": f"{note}; DAG helicity probe failed: {exc}",
        }
    return fallback


def _alternating_fixed_source_helicity_choice(process: str) -> dict[str, object]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    source_helicities: dict[str, int] = {}
    for index, pdg in enumerate(legacy_amplicol.process_pdgs(process), start=1):
        helicities = _preferred_helicity_domain(int(pdg))
        if {-1, 1}.issubset(helicities):
            helicity = -1 if index % 2 else 1
        elif 0 in helicities:
            helicity = 0
        else:
            helicity = sorted(helicities)[0]
        source_helicities[str(index)] = int(helicity)
    return {
        "mode": "fixed-source-helicity",
        "source_helicities": dict(source_helicities),
        "source_helicities_cli": ",".join(
            f"{label}={helicity}"
            for label, helicity in sorted(
                source_helicities.items(), key=lambda item: int(item[0])
            )
        ),
        "amplicol_helicities": [
            int(helicity)
            for _label, helicity in sorted(
                source_helicities.items(), key=lambda item: int(item[0])
            )
        ],
        "value_validation_enabled": False,
        "validation_note": (
            "fixed source-helicity selection is used for timing; "
            "selected-flow spin-summed validation remains authoritative"
        ),
    }


def _chiral_fermion_fixed_source_helicity_choice(
    process: str,
) -> dict[str, object] | None:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    source_helicities: dict[str, int] = {}
    changed = False
    for index, pdg in enumerate(legacy_amplicol.process_pdgs(process), start=1):
        helicities = _preferred_helicity_domain(int(pdg))
        if {-1, 1}.issubset(helicities) and 1 <= abs(int(pdg)) <= 16:
            helicity = -1 if int(pdg) > 0 else 1
            old = -1 if index % 2 else 1
            changed = changed or helicity != old
        elif 0 in helicities:
            helicity = 0
        else:
            helicity = sorted(helicities)[0]
        source_helicities[str(index)] = int(helicity)
    if not changed:
        return None
    return _source_helicity_choice_payload(
        process,
        source_helicities,
        selection_source="pyamplicol-dag-validated-chiral-fermion-candidate",
        validation_note=(
            "DAG-validated chiral fermion source-helicity selection is used "
            "for timing; selected-flow spin-summed validation remains authoritative"
        ),
    )


def _source_helicity_choice_payload(
    process: str,
    source_helicities: Mapping[str, object],
    *,
    selection_source: str,
    validation_note: str,
) -> dict[str, object]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    expected = len(legacy_amplicol.process_pdgs(process))
    normalized = {
        str(label): int(value)
        for label, value in sorted(
            source_helicities.items(),
            key=lambda item: int(item[0]),
        )
    }
    if sorted(int(label) for label in normalized) != list(range(1, expected + 1)):
        raise ValueError(
            "fixed source-helicity choice is not aligned with process labels "
            f"1..{expected}: {normalized}"
        )
    return {
        "mode": "fixed-source-helicity",
        "selection_source": selection_source,
        "source_helicities": dict(normalized),
        "source_helicities_cli": ",".join(
            f"{label}={helicity}" for label, helicity in normalized.items()
        ),
        "amplicol_helicities": [int(value) for value in normalized.values()],
        "value_validation_enabled": False,
        "validation_note": validation_note,
    }


def _source_helicity_choice_has_amplitudes(
    process: str,
    source_helicities: Mapping[str, object],
    *,
    spec: MatrixSpec | LadderSpec,
    artifact_root: Path,
) -> bool:
    try:
        _compile_dag_for_fixed_helicity_choice(
            process,
            spec=spec,
            artifact_root=artifact_root,
            source_helicities=source_helicities,
        )
    except Exception as exc:
        message = str(exc)
        if "has no model-supported amplitudes" in message:
            return False
        raise
    return True


def _first_model_supported_source_helicity_choice(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec,
    artifact_root: Path,
    preferred: Mapping[str, object],
) -> dict[str, object] | None:
    from pyamplicol.generation.dag_algorithms import (
        _root_source_helicity_mapping,
        _source_helicity_signature_by_bit,
    )

    dag = _compile_dag_for_fixed_helicity_choice(
        process,
        spec=spec,
        artifact_root=artifact_root,
        source_helicities=None,
    )
    source_by_bit = _source_helicity_signature_by_bit(dag)
    choices: set[tuple[tuple[str, int], ...]] = set()
    for root in dag.amplitude_roots:
        mapping = _root_source_helicity_mapping(dag, root, source_by_bit)
        choices.add(
            tuple(
                (str(label), int(helicity))
                for label, helicity in sorted(mapping.items())
            )
        )
    if not choices:
        return None
    preferred_choice = tuple(
        (str(label), int(value))
        for label, value in sorted(preferred.items(), key=lambda item: int(item[0]))
    )
    selected = preferred_choice if preferred_choice in choices else sorted(choices)[0]
    return _source_helicity_choice_payload(
        process,
        dict(selected),
        selection_source="pyamplicol-dag-amplitude-root-selection",
        validation_note=(
            "DAG-supported source-helicity selection is used for timing; "
            "selected-flow spin-summed validation remains authoritative"
        ),
    )


def _compile_dag_for_fixed_helicity_choice(
    process: str,
    *,
    spec: MatrixSpec | LadderSpec,
    artifact_root: Path,
    source_helicities: Mapping[str, object] | None,
) -> object:
    from pyamplicol.api import ProcessSet
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import resolve_config
    from pyamplicol.generation.service import GenerationBackend

    GenerationSlice, _generate_slice = _generation_slice_tools()

    generation_slice = None
    if source_helicities is not None:
        generation_slice = GenerationSlice(
            selected_source_helicities={
                int(label): int(value) for label, value in source_helicities.items()
            }
        )
    config = _run_config_values(
        model=spec.model,
        color_accuracy="lc",
        variant_overrides={
            "evaluator.backend": "jit",
            "evaluator.jit.optimization_level": 3,
        },
        artifact_root=artifact_root,
        target_runtime=1.0,
        cell_cores=1,
    )
    resolution = resolve_config(
        config,
        action=Action.GENERATE,
        base_dir=_repo_root(),
    )
    backend = GenerationBackend(
        resolution,
        None,
        process_selection=(
            None if generation_slice is None else generation_slice._selection()
        ),
    )
    resolved = backend._resolve_model(_model_source_for_api(spec.model))
    if resolved.model is None:
        raise RuntimeError("generation model resolution produced no model")
    expanded = backend._expand_process_set(
        ProcessSet.from_expressions((process,)),
        resolved,
        _NullGenerationPhase(),
    )
    dag, _coverage = backend._compile_concrete_process(
        expanded[0].process_ir,
        resolved.model,
    )
    return dag


def _preferred_helicity_domain(pdg: int) -> set[int]:
    absolute = abs(int(pdg))
    if absolute in {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16}:
        return {-1, 1}
    if absolute in {21, 22}:
        return {-1, 1}
    if absolute in {23, 24}:
        return {-1, 0, 1}
    return {0}


def _legacy_quark_line_count(pdgs: Sequence[int]) -> int:
    quark_legs = sum(1 for pdg in pdgs if 1 <= abs(int(pdg)) <= 6)
    return quark_legs // 2


def _legacy_lc_color_probe_supported(pdgs: Sequence[int]) -> bool:
    return _legacy_quark_line_count(pdgs) <= 2


def _legacy_direct_color_probe_supported(pdgs: Sequence[int]) -> bool:
    return _legacy_quark_line_count(pdgs) <= 3


def _lc_all_flow_supported(process: str) -> bool:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    return _legacy_direct_color_probe_supported(legacy_amplicol.process_pdgs(process))


def _legacy_probe_scope_limited(message: object) -> bool:
    text = str(message).lower()
    return "more than two quarks" in text or "quark lines exceed" in text


def _legacy_command_record(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], str]:
    rendered = [os.fspath(arg) for arg in args]
    started = time.perf_counter()
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        env=None if env is None else {**os.environ, **dict(env)},
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + "\n" + completed.stderr
    record = {
        "args": rendered,
        "cwd": os.fspath(cwd.resolve(strict=False)),
        "elapsed_s": elapsed,
        **(
            {"env": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}}
            if env is not None and "LD_LIBRARY_PATH" in env
            else {}
        ),
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with {completed.returncode}: {' '.join(rendered)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return record, output


def _legacy_library_environment(repository: Path) -> dict[str, str]:
    existing = os.environ.get("LD_LIBRARY_PATH")
    path = os.fspath(repository.resolve(strict=False))
    return {"LD_LIBRARY_PATH": path if not existing else f"{path}:{existing}"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_legacy_generated_library(
    repository: Path,
    destination: Path,
    *,
    required_executables: Sequence[str],
    optional_executables: Sequence[str] = (
        "amplicol_generate",
        "amplicol_library_benchmark",
        "amplicol_color_probe",
        "amplicol_color_library_probe",
    ),
    process_file: Path | None = None,
) -> dict[str, object]:
    """Preserve and execute the generated legacy library from a cell artifact."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict[str, object]] = {}
    for source in sorted(repository.glob("libamp*.so")):
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            copied[target.name] = {
                "path": os.fspath(target),
                "source": os.fspath(source),
                "sha256": _file_sha256(target),
                "size_bytes": target.stat().st_size,
            }

    source_library = repository / "Library"
    if not source_library.is_dir():
        raise FileNotFoundError(
            f"generated legacy Library directory is missing: {source_library}"
        )
    target_library = destination / "Library"
    shutil.copytree(source_library, target_library)
    for target in sorted(path for path in target_library.rglob("*") if path.is_file()):
        source = source_library / target.relative_to(target_library)
        relative = os.fspath(target.relative_to(destination))
        copied[relative] = {
            "path": os.fspath(target),
            "source": os.fspath(source),
            "sha256": _file_sha256(target),
            "size_bytes": target.stat().st_size,
        }

    required = tuple(
        dict.fromkeys(str(executable) for executable in required_executables)
    )
    optional = tuple(
        executable
        for executable in dict.fromkeys(
            str(executable) for executable in optional_executables
        )
        if executable not in required
    )
    for executable in (*required, *optional):
        source = repository / executable
        if not source.is_file():
            if executable in required:
                raise FileNotFoundError(
                    f"generated legacy executable is missing: {source}"
                )
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        copied[target.name] = {
            "path": os.fspath(target),
            "source": os.fspath(source),
            "sha256": _file_sha256(target),
            "size_bytes": target.stat().st_size,
        }

    if process_file is not None:
        target = destination / "processes.txt"
        shutil.copy2(process_file, target)
        copied[target.name] = {
            "path": os.fspath(target),
            "source": os.fspath(process_file),
            "sha256": _file_sha256(target),
            "size_bytes": target.stat().st_size,
        }

    if not any(name.startswith("libamp") and name.endswith(".so") for name in copied):
        raise FileNotFoundError(
            f"generated legacy libraries are missing in {repository}"
        )

    return {
        "args": [
            "snapshot_legacy_generated_library",
            os.fspath(repository.resolve(strict=False)),
            os.fspath(destination.resolve(strict=False)),
        ],
        "cwd": os.fspath(repository.resolve(strict=False)),
        "elapsed_s": 0.0,
        "returncode": 0,
        "artifact_path": os.fspath(destination),
        "files": [copied[name] for name in sorted(copied)],
    }


def _legacy_timing_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    in_summary = False
    pattern = re.compile(
        r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /_-]+?)\s+"
        r"(?P<seconds>[0-9.Ee+-]+)"
        r"(?:\s+(?P<percent>[0-9.]+%))?"
        r"(?:\s+(?P<note>.*?))?\s*$"
    )
    for line in output.splitlines():
        if "Timing summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        match = pattern.match(line)
        if match is None:
            continue
        try:
            seconds = float(match.group("seconds"))
        except ValueError:
            continue
        rows.append(
            {
                "label": match.group("label").strip(),
                "seconds": seconds,
                "note": match.group("note") or "",
            }
        )
    return rows


def _legacy_timing_seconds(
    rows: Sequence[Mapping[str, object]],
    label: str,
) -> float | None:
    wanted = label.strip().lower()
    for row in rows:
        if str(row.get("label", "")).strip().lower() == wanted:
            return float(row["seconds"])
    for row in rows:
        if wanted in str(row.get("label", "")).strip().lower():
            return float(row["seconds"])
    return None


def _legacy_profile_elapsed_seconds(
    record: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *labels: str,
) -> float:
    for label in labels:
        seconds = _legacy_timing_seconds(rows, label)
        if seconds is not None and math.isfinite(seconds) and seconds > 0.0:
            return seconds
    elapsed = _optional_positive_float(record.get("elapsed_s"))
    return 0.0 if elapsed is None else elapsed


def _legacy_adaptive_profile_points(
    warmup_seconds: float,
    *,
    target_runtime: float,
    warmup_points: int = DEFAULT_LEGACY_PROFILE_WARMUP_POINTS,
    min_points: int = DEFAULT_LEGACY_PROFILE_MIN_POINTS,
    max_points: int = DEFAULT_LEGACY_PROFILE_MAX_POINTS,
) -> int:
    warmup_points = max(1, int(warmup_points))
    min_points = max(1, int(min_points))
    max_points = max(min_points, int(max_points))
    target = float(target_runtime)
    if not math.isfinite(target) or target <= 0.0:
        target = DEFAULT_REPORT_TARGET_RUNTIME_SECONDS
    if not math.isfinite(warmup_seconds) or warmup_seconds <= 0.0:
        return min_points
    estimated = math.ceil(target * warmup_points / warmup_seconds)
    return max(min_points, min(max_points, int(estimated)))


def _legacy_profile_record(
    *,
    probe: str,
    target_runtime: float,
    warmup_record: Mapping[str, object],
    warmup_rows: Sequence[Mapping[str, object]],
    warmup_seconds: float,
    measurement_record: Mapping[str, object],
    measurement_rows: Sequence[Mapping[str, object]],
    measurement_seconds: float,
    measurement_points: int,
    timing_labels: Sequence[str],
) -> dict[str, object]:
    return {
        **_legacy_profile_requested_config(target_runtime),
        "probe": probe,
        "warmup_seconds": float(warmup_seconds),
        "warmup_us_per_point": (
            1.0e6 * float(warmup_seconds) / DEFAULT_LEGACY_PROFILE_WARMUP_POINTS
        ),
        "measurement_points": int(measurement_points),
        "measurement_seconds": float(measurement_seconds),
        "measurement_us_per_point": (
            1.0e6 * float(measurement_seconds) / max(1, int(measurement_points))
        ),
        "timing_labels": [str(label) for label in timing_labels],
        "warmup_record": dict(warmup_record),
        "warmup_timing_rows": [dict(row) for row in warmup_rows],
        "measurement_record": dict(measurement_record),
        "measurement_timing_rows": [dict(row) for row in measurement_rows],
    }


def _legacy_run_command_profiled(
    command_for_points: Callable[[int], Sequence[str | os.PathLike[str]]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    target_runtime: float,
    probe: str,
    timing_labels: Sequence[str],
) -> tuple[dict[str, object], str, list[dict[str, object]], int, dict[str, object]]:
    if not callable(command_for_points):
        raise TypeError("command_for_points must be callable")
    warmup_points = DEFAULT_LEGACY_PROFILE_WARMUP_POINTS
    warmup_record, warmup_output = _legacy_command_record(
        command_for_points(warmup_points),
        cwd=cwd,
        env=env,
    )
    warmup_record = {**warmup_record, "profile_phase": "warmup"}
    warmup_rows = _legacy_timing_rows(warmup_output)
    warmup_seconds = _legacy_profile_elapsed_seconds(
        warmup_record,
        warmup_rows,
        *timing_labels,
    )
    measurement_points = _legacy_adaptive_profile_points(
        warmup_seconds,
        target_runtime=target_runtime,
    )
    measurement_record, measurement_output = _legacy_command_record(
        command_for_points(measurement_points),
        cwd=cwd,
        env=env,
    )
    measurement_record = {
        **measurement_record,
        "profile_phase": "measurement",
        "profile_warmup_points": warmup_points,
        "profile_points": measurement_points,
        "profile_target_runtime_seconds": float(target_runtime),
    }
    measurement_rows = _legacy_timing_rows(measurement_output)
    measurement_seconds = _legacy_profile_elapsed_seconds(
        measurement_record,
        measurement_rows,
        *timing_labels,
    )
    profile = _legacy_profile_record(
        probe=probe,
        target_runtime=target_runtime,
        warmup_record=warmup_record,
        warmup_rows=warmup_rows,
        warmup_seconds=warmup_seconds,
        measurement_record=measurement_record,
        measurement_rows=measurement_rows,
        measurement_seconds=measurement_seconds,
        measurement_points=measurement_points,
        timing_labels=timing_labels,
    )
    return (
        measurement_record,
        measurement_output,
        measurement_rows,
        measurement_points,
        profile,
    )


def _legacy_process_list_command(
    repository: Path,
    family: ProcessFamily | None,
    process: str,
) -> list[str]:
    return [
        sys.executable,
        os.fspath(repository / "process_list.py"),
        "--serial",
        *(family.legacy_process_list_flags() if family is not None else ()),
        process,
    ]


def _write_legacy_momenta_files(
    repository: Path,
    *,
    entries: Sequence[object],
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
) -> None:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    directory = repository / "Utilities" / "ME_checks"
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        ordered = legacy_amplicol._ordered_binary64_momenta(
            source_pdgs,
            entry.process_pdgs,
            momenta,
        )
        path = directory / f"momenta_{entry.group}_{entry.integral}.txt"
        path.write_text(
            "\n".join(
                " ".join(f"{component:.17e}" for component in vector)
                for vector in ordered
            )
            + "\n",
            encoding="utf-8",
        )


@contextmanager
def _legacy_repository_process_file(
    repository: Path,
    process_file: Path,
) -> Iterator[tuple[str, Path]]:
    target = repository / "processes.txt"
    backup = repository / f".processes.txt.pyamplicol-backup-{uuid.uuid4().hex}"
    had_existing = target.exists()
    if had_existing:
        shutil.copy2(target, backup)
    shutil.copy2(process_file, target)
    try:
        yield target.name, target
    finally:
        if had_existing:
            shutil.move(backup, target)
        else:
            target.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def _legacy_run_color_probe_timed(
    repository: Path,
    *,
    process_file: Path,
    entry: object,
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
    color_accuracy: str = "lc",
    helicities: Sequence[int] | None,
    points: int,
    executable: Path | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], object]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    ordered = legacy_amplicol._ordered_binary64_momenta(
        source_pdgs,
        entry.process_pdgs,
        momenta,
    )
    permutation = legacy_amplicol._permutation(source_pdgs, entry.process_pdgs)
    ordered_helicities = (
        ()
        if helicities is None
        else tuple(int(helicities[index]) for index in permutation)
    )
    with tempfile.TemporaryDirectory(prefix="pac-", dir="/tmp") as raw:
        work = Path(raw)
        process_copy = work / "processes.txt"
        momenta_path = work / "momenta.dat"
        shutil.copy2(process_file, process_copy)
        momenta_path.write_text(
            "\n".join(
                " ".join(format(float(component), ".17g") for component in vector)
                for vector in ordered
            )
            + "\n",
            encoding="utf-8",
        )
        record, output = _legacy_command_record(
            [
                (executable or (repository / "amplicol_color_probe")).resolve(),
                str(max(1, int(points))),
                str(entry.group),
                str(entry.integral),
                color_accuracy,
                process_copy,
                momenta_path,
                *(str(value) for value in ordered_helicities),
            ],
            cwd=work if cwd is None else cwd,
            env=env,
        )
    probe = legacy_amplicol._parse_probe_output(output)
    return record, _legacy_timing_rows(output), probe


def _legacy_run_color_probe_profiled(
    repository: Path,
    *,
    process_file: Path,
    entry: object,
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
    color_accuracy: str = "lc",
    helicities: Sequence[int] | None,
    target_runtime: float,
    executable: Path | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    object,
    int,
    dict[str, object],
]:
    warmup_record, warmup_rows, _warmup_probe = _legacy_run_color_probe_timed(
        repository,
        process_file=process_file,
        entry=entry,
        source_pdgs=source_pdgs,
        momenta=momenta,
        color_accuracy=color_accuracy,
        helicities=helicities,
        points=DEFAULT_LEGACY_PROFILE_WARMUP_POINTS,
        executable=executable,
        cwd=cwd,
        env=env,
    )
    warmup_record = {**warmup_record, "profile_phase": "warmup"}
    warmup_seconds = _legacy_profile_elapsed_seconds(
        warmup_record,
        warmup_rows,
        "total",
    )
    measurement_points = _legacy_adaptive_profile_points(
        warmup_seconds,
        target_runtime=target_runtime,
    )
    record, rows, probe = _legacy_run_color_probe_timed(
        repository,
        process_file=process_file,
        entry=entry,
        source_pdgs=source_pdgs,
        momenta=momenta,
        color_accuracy=color_accuracy,
        helicities=helicities,
        points=measurement_points,
        executable=executable,
        cwd=cwd,
        env=env,
    )
    record = {
        **record,
        "profile_phase": "measurement",
        "profile_warmup_points": DEFAULT_LEGACY_PROFILE_WARMUP_POINTS,
        "profile_points": measurement_points,
        "profile_target_runtime_seconds": float(target_runtime),
    }
    measurement_seconds = _legacy_profile_elapsed_seconds(record, rows, "total")
    profile = _legacy_profile_record(
        probe="amplicol_color_probe",
        target_runtime=target_runtime,
        warmup_record=warmup_record,
        warmup_rows=warmup_rows,
        warmup_seconds=warmup_seconds,
        measurement_record=record,
        measurement_rows=rows,
        measurement_seconds=measurement_seconds,
        measurement_points=measurement_points,
        timing_labels=("total",),
    )
    return record, rows, probe, measurement_points, profile


def _legacy_lc_partition_matrix_element(
    probe: object,
    *,
    reference_color_order: Sequence[int],
    reference_color_order_candidates: Sequence[Sequence[int]] | None = None,
    source_to_row_permutation: Sequence[int],
) -> tuple[float, dict[str, object]]:
    partitions = getattr(probe, "lc_row_partitions", ())
    if not partitions:
        raise RuntimeError("AmpliCol LC probe did not emit row partitions")
    source_partitions: list[tuple[object, tuple[int, ...], float]] = []
    for partition in partitions:
        raw_word = tuple(int(label) for label in getattr(partition, "permutation", ()))
        if any(label < 1 for label in raw_word):
            raise RuntimeError(
                "AmpliCol LC probe emitted an invalid row partition permutation"
            )
        try:
            source_word = tuple(
                int(source_to_row_permutation[label - 1]) + 1 for label in raw_word
            )
        except IndexError as error:
            raise RuntimeError(
                "AmpliCol LC probe emitted a row partition outside the process legs"
            ) from error
        source_partitions.append(
            (partition, source_word, _real_nonnegative_scalar(partition.value))
    )
    coloured_labels = {
        label
        for _partition, source_word, _value in source_partitions
        for label in source_word
    }
    raw_targets = reference_color_order_candidates or (reference_color_order,)
    targets = tuple(
        dict.fromkeys(
            tuple(int(label) for label in target if int(label) in coloured_labels)
            for target in raw_targets
        )
    )
    targets = tuple(target for target in targets if target)
    if not targets:
        raise RuntimeError(
            "AmpliCol LC reference colour order has no coloured labels to match"
        )
    matches = [
        (partition, source_word, value, target)
        for partition, source_word, value in source_partitions
        for target in targets
        if source_word == target
    ]
    if len(matches) != 1:
        available = [
            list(source_word)
            for _partition, source_word, _value in source_partitions
        ]
        raise RuntimeError(
            "AmpliCol LC row partition could not be matched uniquely: "
            f"targets={[list(target) for target in targets]}, available={available}"
        )
    partition, source_word, value, target = matches[0]
    return value, {
        "reference_color_order_coloured": list(target),
        "reference_color_order_candidates": [list(target) for target in targets],
        "reference_lc_partition_row": int(partition.row),
        "reference_lc_partition_permutation": list(source_word),
        "lc_row_partitions": [
            {
                "row": int(item.row),
                "value": float(item_value),
                "permutation": list(item_word),
            }
            for item, item_word, item_value in source_partitions
        ],
    }


def _source_helicity_combinations(
    source_pdgs: Sequence[int],
) -> Iterator[tuple[int, ...]]:
    domains = [
        tuple(sorted(_preferred_helicity_domain(int(pdg)))) for pdg in source_pdgs
    ]
    yield from itertools.product(*domains)


def _legacy_lc_selected_flow_matrix_element(
    repository: Path,
    *,
    process_file: Path,
    entry: object,
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
    reference_color_order: Sequence[int],
    reference_color_order_candidates: Sequence[Sequence[int]] | None = None,
) -> tuple[
    float,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    source_to_row = legacy_amplicol._permutation(source_pdgs, entry.process_pdgs)
    commands: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    helicity_records: list[dict[str, object]] = []
    total = 0.0
    reference_metadata: dict[str, object] | None = None
    for helicities in _source_helicity_combinations(source_pdgs):
        record, rows, probe = _legacy_run_color_probe_timed(
            repository,
            process_file=process_file,
            entry=entry,
            source_pdgs=source_pdgs,
            momenta=momenta,
            helicities=helicities,
            points=1,
        )
        value, metadata = _legacy_lc_partition_matrix_element(
            probe,
            reference_color_order=reference_color_order,
            reference_color_order_candidates=reference_color_order_candidates,
            source_to_row_permutation=source_to_row,
        )
        total += value
        commands.append(record)
        timing_rows.extend(
            {**row, "source_helicities": list(helicities)} for row in rows
        )
        helicity_records.append(
            {
                "source_helicities": list(helicities),
                "value": float(value),
                "reference_lc_partition_row": metadata[
                    "reference_lc_partition_row"
                ],
            }
        )
        if reference_metadata is None:
            reference_metadata = dict(metadata)
    if reference_metadata is None:
        raise RuntimeError("AmpliCol LC selected-flow helicity sum is empty")
    return _real_nonnegative_scalar(total), commands, timing_rows, {
        **reference_metadata,
        "selected_flow_helicity_sum": helicity_records,
        "selected_flow_helicity_count": len(helicity_records),
    }


_LEGACY_GENERATED_LIBRARY_PROBE_RE = re.compile(
    r"^AMPICOL_PROBE_VALUE\s+"
    r"(?P<point>\d+)\s+(?P<group>\d+)\s+(?P<integral>\d+)\s+"
    r"(?P<value>[+\-0-9.Ee]+)$",
    re.MULTILINE,
)


def _parse_legacy_generated_library_probe_value(
    output: str,
    *,
    expected_group: int,
    expected_integral: int,
) -> float:
    matches = list(_LEGACY_GENERATED_LIBRARY_PROBE_RE.finditer(output))
    if len(matches) != 1:
        raise RuntimeError(
            "AmpliCol generated-library probe must emit exactly one "
            f"AMPICOL_PROBE_VALUE record, got {len(matches)}"
        )
    match = matches[0]
    group = int(match.group("group"))
    integral = int(match.group("integral"))
    if group != int(expected_group) or integral != int(expected_integral):
        raise RuntimeError(
            "AmpliCol generated-library probe row mismatch: "
            f"got group={group} integral={integral}, expected "
            f"group={int(expected_group)} integral={int(expected_integral)}"
        )
    return _real_nonnegative_scalar(float(match.group("value")))


def _legacy_run_generated_library_probe(
    repository: Path,
    *,
    process_file_arg: str,
    entry: object,
    output_path: Path,
) -> tuple[dict[str, object], float]:
    record, output = _legacy_command_record(
        [
            "./amplicol_generate",
            "--library=use",
            f"--process={process_file_arg}",
            "--amplicol_momenta_probe=1",
            "--timing=none",
        ],
        cwd=repository,
        env=_legacy_library_environment(repository),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    value = _parse_legacy_generated_library_probe_value(
        output,
        expected_group=int(entry.group),
        expected_integral=int(entry.integral),
    )
    record["output_path"] = os.fspath(output_path)
    return record, value


def _legacy_build_selected_flow_library_probe_record(
    repository: Path,
    *,
    jobs: int,
) -> dict[str, object]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    started = time.perf_counter()
    legacy_amplicol.build_selected_flow_library_probe(
        repository,
        jobs=max(1, int(jobs)),
    )
    return {
        "args": [
            "legacy_amplicol.build_selected_flow_library_probe",
            os.fspath(repository),
            f"jobs={max(1, int(jobs))}",
        ],
        "elapsed_s": time.perf_counter() - started,
        "returncode": 0,
    }


def _legacy_selected_flow_probe_payload(result: object) -> dict[str, object]:
    return {
        "value": float(result.value),
        "value_decimal": str(result.value_decimal),
        "group": int(result.group),
        "integral": int(result.integral),
        "process_pdgs": [int(pdg) for pdg in result.process_pdgs],
        "color_order": [int(label) for label in result.color_order],
        "amplitudes": int(result.amplitudes),
        "color_factor": int(result.color_factor),
        "identical_factor": int(result.identical_factor),
        "singlet_vertices": int(result.singlet_vertices),
        "normalization": float(result.normalization),
        "normalization_decimal": str(result.normalization_decimal),
    }


def _legacy_run_selected_flow_library_probe_record(
    repository: Path,
    *,
    entry: object,
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
    points: int,
    output_path: Path,
) -> tuple[dict[str, object], float, dict[str, object]]:
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    started = time.perf_counter()
    probe_environment = _legacy_library_environment(repository)
    saved_environment = {
        key: os.environ.get(key) for key in probe_environment
    }
    try:
        os.environ.update(probe_environment)
        result = legacy_amplicol.run_selected_flow_library_probe(
            repository,
            entry=entry,
            source_pdgs=source_pdgs,
            momenta=momenta,
            points=max(1, int(points)),
        )
    finally:
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    payload = _legacy_selected_flow_probe_payload(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload), encoding="utf-8")
    record = {
        "args": [
            "legacy_amplicol.run_selected_flow_library_probe",
            os.fspath(repository),
            f"group={int(entry.group)}",
            f"integral={int(entry.integral)}",
            f"points={max(1, int(points))}",
        ],
        "cwd": os.fspath(repository.resolve(strict=False)),
        "elapsed_s": time.perf_counter() - started,
        "env": {"LD_LIBRARY_PATH": probe_environment["LD_LIBRARY_PATH"]},
        "executable": os.fspath(
            (repository / "amplicol_library_benchmark").resolve(strict=False)
        ),
        "returncode": 0,
        "output_path": os.fspath(output_path),
    }
    return record, _real_nonnegative_scalar(result.value), payload


def _measure_legacy_amplicol(
    *,
    cell: CampaignCell,
    color_accuracy: str,
    artifact_root: Path,
    points: object | None,
    limit_gib: float,
    target_runtime: float,
    jobs: int = 1,
    fixed_helicity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cell_root = artifact_root / "cells" / cell.cell_id
    _ensure_repo_root_on_path()
    from tools.developer import legacy_amplicol

    legacy_revision = legacy_amplicol.expected_revision()
    legacy_root = cell_root / f"legacy-amplicol-{legacy_revision[:12]}"
    log_path = cell_root / "logs" / f"legacy-amplicol-{legacy_revision[:12]}.log"
    manifest_path = legacy_root / "manifest.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_root.mkdir(parents=True, exist_ok=True)
    family = _process_family_by_key(cell.process_key)
    command = ["legacy-amplicol-generated-library", cell.process, color_accuracy]
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# legacy AmpliCol cell {cell.cell_id} started {_utc_now()}\n")
            log.flush()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                repository = legacy_amplicol.DEFAULT_REPOSITORY
                if not repository.exists():
                    raise FileNotFoundError(
                        f"legacy AmpliCol checkout is missing: {repository}"
                    )
                legacy_amplicol.validate_checkout(repository)
                build_lock = (
                    Path(tempfile.gettempdir()) / "pyamplicol-legacy-build.lock"
                )
                with build_lock.open("a+b") as stream:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    try:
                        source_pdgs = legacy_amplicol.process_pdgs(cell.process)
                        legacy_amplicol.validate_selected_flow_quark_line_scope(
                            source_pdgs,
                            context=cell.process,
                        )
                        momenta = _legacy_momenta_from_pyamplicol(points)
                        point_source = "pyamplicol-shared-validation-momenta"
                        if momenta is None:
                            particles = _shared_validation_particles(cell.process)
                            momenta = _legacy_momenta_from_particles(particles)
                            source_pdgs = _legacy_pdgs_from_particles(particles)
                            point_source = "generic_validation_point"
                        process_command = _legacy_process_list_command(
                            repository,
                            family,
                            cell.process,
                        )
                        process_record, process_output = _legacy_command_record(
                            process_command,
                            cwd=legacy_root,
                        )
                        process_file = legacy_root / "processes.txt"
                        if not process_file.is_file():
                            raise legacy_amplicol.LegacyOracleError(
                                "legacy process_list.py did not produce processes.txt "
                                f"for {cell.process!r}; command={process_command!r}; "
                                f"output={process_output[-2000:]!r}"
                            )
                        entries = legacy_amplicol.parse_process_file(process_file)
                        (
                            entry,
                            matches,
                        ) = legacy_amplicol.select_generated_process_entry(
                            entries,
                            generated_process=cell.process,
                            wanted_pdgs=source_pdgs,
                        )
                        mapped_color_order = legacy_amplicol.source_mapped_color_order(
                            entry,
                            source_pdgs=source_pdgs,
                        )
                        selected_lc_reference_words: tuple[tuple[int, ...], ...] = (
                            (tuple(int(label) for label in mapped_color_order),)
                        )
                        if color_accuracy == "lc":
                            try:
                                spec = _spec_by_dataset()[cell.dataset_id]
                                selected_lc_reference_words = (
                                    _selected_lc_reference_partition_words(
                                        cell.process,
                                        spec=spec,
                                        reference_order=mapped_color_order,
                                        artifact_root=artifact_root,
                                    )
                                )
                            except Exception:
                                selected_lc_reference_words = (
                                    (tuple(int(label) for label in mapped_color_order),)
                                )
                        _write_legacy_momenta_files(
                            repository,
                            entries=entries,
                            source_pdgs=source_pdgs,
                            momenta=momenta,
                        )
                        generation_records: list[dict[str, object]] = [process_record]
                        library_mode = (
                            "create-raw" if color_accuracy != "lc" else "create"
                        )
                        make_jobs = max(1, int(jobs))
                        selected_generated_probe_record: dict[str, object] | None = (
                            None
                        )
                        selected_generated_probe_value: float | None = None
                        selected_generated_probe_payload: dict[str, object] | None = (
                            None
                        )
                        with _legacy_repository_process_file(
                            repository,
                            process_file,
                        ) as (process_file_arg, _staged_process_file):
                            for args in (
                                ["make", "cleanlib"],
                                [
                                    "make",
                                    f"-j{make_jobs}",
                                    "amplicol_generate",
                                ],
                                [
                                    "./amplicol_generate",
                                    f"--library={library_mode}",
                                    f"--process={process_file_arg}",
                                    "--amplicol_momenta_probe=10",
                                    "--amplicol_probe_quiet",
                                    "--timing=none",
                                ],
                                [
                                    "make",
                                    f"-j{make_jobs}",
                                    "amplicol_generate_library",
                                ],
                            ):
                                record, _output = _legacy_command_record(
                                    args,
                                    cwd=repository,
                                )
                                generation_records.append(record)
                        generation_seconds = math.fsum(
                            float(record["elapsed_s"])
                            for record in generation_records
                            if record["args"] != process_record["args"]
                        )
                        if color_accuracy == "lc":
                            build_benchmark = (
                                _legacy_build_selected_flow_library_probe_record(
                                    repository,
                                    jobs=make_jobs,
                                )
                            )
                            generated_library_snapshot = (
                                _snapshot_legacy_generated_library(
                                    repository,
                                    legacy_root / "generated-library",
                                    required_executables=(
                                        "amplicol_library_benchmark",
                                    ),
                                    process_file=process_file,
                                )
                            )
                            generated_library_root = Path(
                                os.fspath(
                                    generated_library_snapshot["artifact_path"]
                                )
                            )
                            generated_library_environment = (
                                _legacy_library_environment(
                                    generated_library_root,
                                )
                            )
                            (
                                benchmark_record,
                                _benchmark_output,
                                timing_rows,
                                runtime_sample_count,
                                runtime_profile,
                            ) = _legacy_run_command_profiled(
                                lambda count: [
                                    "./amplicol_library_benchmark",
                                    str(max(1, int(count))),
                                    str(entry.group),
                                    str(entry.integral),
                                ],
                                cwd=generated_library_root,
                                env=generated_library_environment,
                                target_runtime=target_runtime,
                                probe="amplicol_library_benchmark",
                                timing_labels=("amplitude evaluation", "total"),
                            )
                            runtime_seconds = _legacy_timing_seconds(
                                timing_rows,
                                "amplitude evaluation",
                            )
                            if runtime_seconds is None:
                                runtime_seconds = _legacy_timing_seconds(
                                    timing_rows,
                                    "total",
                                )
                            if runtime_seconds is None:
                                raise legacy_amplicol.LegacyOracleError(
                                    "amplicol_library_benchmark did not report "
                                    "an amplitude-evaluation timing row"
                                )
                            runtime_seconds /= runtime_sample_count
                            runtime_probe = "direct_generated_library_benchmark"
                            (
                                selected_generated_probe_record,
                                selected_generated_probe_value,
                                selected_generated_probe_payload,
                            ) = _legacy_run_selected_flow_library_probe_record(
                                generated_library_root,
                                entry=entry,
                                source_pdgs=source_pdgs,
                                momenta=momenta,
                                points=1,
                                output_path=(
                                    legacy_root / "selected-flow-library-probe.json"
                                ),
                            )
                            if selected_generated_probe_record is None:
                                raise legacy_amplicol.LegacyOracleError(
                                    "generated LC library probe did not run"
                                )
                            matrix_element = _real_nonnegative_scalar(
                                selected_generated_probe_value
                            )
                            matrix_element_probe = (
                                "amplicol_library_benchmark_selected_flow"
                            )
                            selected_probe_rows = []
                            selected_probe_metadata = {
                                "selected_flow_probe": matrix_element_probe,
                                "selected_flow_probe_output": (
                                    selected_generated_probe_record.get("output_path")
                                ),
                                "selected_flow_library_probe": (
                                    selected_generated_probe_payload
                                ),
                                "selected_flow_partition_status": (
                                    ResultStatus.UNSUPPORTED.value
                                    if not _legacy_lc_color_probe_supported(source_pdgs)
                                    else NA_STATUS
                                ),
                                "selected_flow_partition_note": (
                                    "legacy color-probe row partitions are limited "
                                    "to at most two quark lines; selected scalar "
                                    "uses the generated-library indexed probe"
                                    if not _legacy_lc_color_probe_supported(source_pdgs)
                                    else None
                                ),
                            }
                            timing_commands = [
                                build_benchmark,
                                generated_library_snapshot,
                                benchmark_record,
                                selected_generated_probe_record,
                            ]
                            if _legacy_lc_color_probe_supported(source_pdgs):
                                build_selected_probe, _build_output = (
                                    _legacy_command_record(
                                        [
                                            "make",
                                            f"-j{make_jobs}",
                                            "amplicol_color_probe",
                                        ],
                                        cwd=repository,
                                    )
                                )
                                (
                                    partition_value,
                                    selected_probe_commands,
                                    selected_probe_rows,
                                    selected_probe_metadata,
                                ) = _legacy_lc_selected_flow_matrix_element(
                                    repository,
                                    process_file=process_file,
                                    entry=entry,
                                    source_pdgs=source_pdgs,
                                    momenta=momenta,
                                    reference_color_order=mapped_color_order,
                                    reference_color_order_candidates=(
                                        selected_lc_reference_words
                                    ),
                                )
                                (
                                    partition_absolute,
                                    partition_relative,
                                    partition_status,
                                ) = _matrix_element_difference(
                                    matrix_element,
                                    partition_value,
                                )
                                if partition_status != ResultStatus.OK.value:
                                    raise RuntimeError(
                                        "AmpliCol generated-library LC scalar and "
                                        "LC row-partition scalar disagree: "
                                        f"generated={matrix_element}, "
                                        f"partition={partition_value}, "
                                        f"relative={partition_relative}"
                                    )
                                selected_probe_metadata = {
                                    **selected_probe_metadata,
                                    "selected_flow_probe": matrix_element_probe,
                                    "selected_flow_probe_output": (
                                        selected_generated_probe_record.get(
                                            "output_path"
                                        )
                                    ),
                                    "selected_flow_partition_status": (
                                        ResultStatus.OK.value
                                    ),
                                    "selected_flow_partition_value": float(
                                        partition_value
                                    ),
                                    "selected_flow_partition_absolute_difference": (
                                        partition_absolute
                                    ),
                                    "selected_flow_partition_relative_difference": (
                                        partition_relative
                                    ),
                                }
                                timing_commands.extend(
                                    [
                                        build_selected_probe,
                                        *selected_probe_commands,
                                    ]
                                )
                        else:
                            use_direct_color_probe = (
                                _legacy_direct_color_probe_supported(source_pdgs)
                                and not _legacy_lc_color_probe_supported(source_pdgs)
                            )
                            if use_direct_color_probe:
                                build_probe, _build_output = _legacy_command_record(
                                    [
                                        "make",
                                        f"-j{make_jobs}",
                                        "amplicol_color_probe",
                                    ],
                                    cwd=repository,
                                )
                                generated_library_snapshot = (
                                    _snapshot_legacy_generated_library(
                                        repository,
                                        legacy_root / "generated-library",
                                        required_executables=(
                                            "amplicol_color_probe",
                                        ),
                                        process_file=process_file,
                                    )
                                )
                                generated_library_root = Path(
                                    os.fspath(
                                        generated_library_snapshot["artifact_path"]
                                    )
                                )
                                generated_library_environment = (
                                    _legacy_library_environment(
                                        generated_library_root,
                                    )
                                )
                                (
                                    probe_record,
                                    timing_rows,
                                    probe,
                                    runtime_sample_count,
                                    runtime_profile,
                                ) = _legacy_run_color_probe_profiled(
                                    repository,
                                    process_file=process_file,
                                    entry=entry,
                                    source_pdgs=source_pdgs,
                                    momenta=momenta,
                                    color_accuracy=color_accuracy,
                                    helicities=None,
                                    target_runtime=target_runtime,
                                    executable=(
                                        generated_library_root
                                        / "amplicol_color_probe"
                                    ),
                                    cwd=generated_library_root,
                                    env=generated_library_environment,
                                )
                                runtime_seconds = _legacy_timing_seconds(
                                    timing_rows,
                                    "total",
                                )
                                if runtime_seconds is None:
                                    raise legacy_amplicol.LegacyOracleError(
                                        "amplicol_color_probe did not report "
                                        "a total timing row"
                                    )
                                runtime_seconds /= runtime_sample_count
                                runtime_probe = "amplicol_color_probe"
                                timing_commands = [
                                    build_probe,
                                    generated_library_snapshot,
                                    probe_record,
                                ]
                                matrix_element = _real_nonnegative_scalar(probe.value)
                                matrix_element_probe = "amplicol_color_probe"
                                selected_probe_rows = []
                                selected_probe_metadata = {}
                            else:
                                build_probe, _build_output = _legacy_command_record(
                                    [
                                        "make",
                                        f"-j{make_jobs}",
                                        "amplicol_color_library_probe",
                                    ],
                                    cwd=repository,
                                )
                                generated_library_snapshot = (
                                    _snapshot_legacy_generated_library(
                                        repository,
                                        legacy_root / "generated-library",
                                        required_executables=(
                                            "amplicol_color_library_probe",
                                        ),
                                        process_file=process_file,
                                    )
                                )
                                generated_library_root = Path(
                                    os.fspath(
                                        generated_library_snapshot["artifact_path"]
                                    )
                                )
                                generated_library_environment = (
                                    _legacy_library_environment(
                                        generated_library_root,
                                    )
                                )
                                (
                                    probe_record,
                                    _probe_output,
                                    timing_rows,
                                    runtime_sample_count,
                                    runtime_profile,
                                ) = _legacy_run_command_profiled(
                                    lambda count: [
                                        "./amplicol_color_library_probe",
                                        str(max(1, int(count))),
                                        str(entry.group),
                                        str(entry.integral),
                                        color_accuracy,
                                        repository
                                        / "Utilities"
                                        / "ME_checks"
                                        / f"momenta_{entry.group}_{entry.integral}.txt",
                                    ],
                                    cwd=generated_library_root,
                                    env=generated_library_environment,
                                    target_runtime=target_runtime,
                                    probe="amplicol_color_library_probe",
                                    timing_labels=("total",),
                                )
                                runtime_seconds = _legacy_timing_seconds(
                                    timing_rows,
                                    "total",
                                )
                                if runtime_seconds is None:
                                    raise legacy_amplicol.LegacyOracleError(
                                        "amplicol_color_library_probe did not report "
                                        "a total timing row"
                                    )
                                runtime_seconds /= runtime_sample_count
                                runtime_probe = "amplicol_color_library_probe"
                                timing_commands = [
                                    build_probe,
                                    generated_library_snapshot,
                                    probe_record,
                                ]
                                if not (repository / "amplicol_color_probe").is_file():
                                    legacy_amplicol.build_color_probe(
                                        repository,
                                        jobs=make_jobs,
                                    )
                                probe = legacy_amplicol.run_color_probe(
                                    repository,
                                    process_file=process_file,
                                    entry=entry,
                                    source_pdgs=source_pdgs,
                                    momenta=momenta,
                                    color_accuracy=color_accuracy,
                                )
                                matrix_element = _real_nonnegative_scalar(probe.value)
                                matrix_element_probe = "amplicol_color_probe"
                                selected_probe_rows = []
                                selected_probe_metadata = {}
                        all_flow_payload: dict[str, object] = {
                            "all_flow_status": None,
                            "all_flow_generation_s": None,
                            "all_flow_generation_source": None,
                            "all_flow_runtime_us_per_point": None,
                            "all_flow_runtime_probe_points": None,
                            "all_flow_reference_value": None,
                            "all_flow_reference_probe": None,
                            "all_flow_timing_rows": [],
                        }
                        all_flow_commands: list[dict[str, object]] = []
                        if color_accuracy != "lc":
                            fixed_helicity = None
                        if color_accuracy == "lc" and fixed_helicity is None:
                            fixed_helicity = _fixed_source_helicity_choice(
                                cell.process,
                            )
                        if color_accuracy == "lc" and fixed_helicity is not None:
                            if _lc_all_flow_supported(cell.process):
                                build_color_probe, _build_output = (
                                    _legacy_command_record(
                                        [
                                            "make",
                                            f"-j{make_jobs}",
                                            "amplicol_color_probe",
                                        ],
                                        cwd=repository,
                                    )
                                )
                                all_flow_snapshot = (
                                    _snapshot_legacy_generated_library(
                                        repository,
                                        legacy_root / "all-flow-generated-library",
                                        required_executables=(
                                            "amplicol_color_probe",
                                        ),
                                        process_file=process_file,
                                    )
                                )
                                all_flow_root = Path(
                                    os.fspath(all_flow_snapshot["artifact_path"])
                                )
                                all_flow_environment = _legacy_library_environment(
                                    all_flow_root
                                )
                                (
                                    all_flow_record,
                                    all_flow_rows,
                                    all_flow_probe,
                                    all_flow_sample_count,
                                    all_flow_profile,
                                ) = _legacy_run_color_probe_profiled(
                                    repository,
                                    process_file=process_file,
                                    entry=entry,
                                    source_pdgs=source_pdgs,
                                    momenta=momenta,
                                    helicities=fixed_helicity["amplicol_helicities"],  # type: ignore[arg-type]
                                    target_runtime=target_runtime,
                                    executable=(
                                        all_flow_root / "amplicol_color_probe"
                                    ),
                                    cwd=all_flow_root,
                                    env=all_flow_environment,
                                )
                                all_flow_total = _legacy_timing_seconds(
                                    all_flow_rows,
                                    "total",
                                )
                                all_flow_probe_setup = _legacy_timing_seconds(
                                    all_flow_rows,
                                    "generation setup",
                                )
                                all_flow_payload.update(
                                    {
                                        "all_flow_status": ResultStatus.OK.value,
                                        "all_flow_generation_s": generation_seconds,
                                        "all_flow_generation_source": (
                                            LEGACY_LC_ALL_FLOW_GENERATION_SOURCE
                                        ),
                                        "all_flow_probe_setup_s": all_flow_probe_setup,
                                        "all_flow_runtime_us_per_point": (
                                            None
                                            if all_flow_total is None
                                            else 1.0e6
                                            * all_flow_total
                                            / all_flow_sample_count
                                        ),
                                        "all_flow_runtime_probe_points": (
                                            all_flow_sample_count
                                        ),
                                        "all_flow_profile": all_flow_profile,
                                        "all_flow_reference_value": (
                                            _real_nonnegative_scalar(
                                                all_flow_probe.value
                                            )
                                        ),
                                        "all_flow_reference_probe": (
                                            "amplicol_color_probe_fixed_helicity_all_flows"
                                        ),
                                        "all_flow_timing_rows": all_flow_rows,
                                        "all_flow_source_helicities": (
                                            fixed_helicity["source_helicities"]
                                        ),
                                        "all_flow_amplicol_helicities": (
                                            fixed_helicity["amplicol_helicities"]
                                        ),
                                        "all_flow_helicity_mode": (
                                            fixed_helicity["mode"]
                                        ),
                                        "all_flow_helicity_selection_source": (
                                            fixed_helicity.get("selection_source")
                                        ),
                                        "all_flow_validation_note": (
                                            fixed_helicity["validation_note"]
                                        ),
                                    }
                                )
                                all_flow_commands.extend(
                                    [
                                        build_color_probe,
                                        all_flow_snapshot,
                                        all_flow_record,
                                    ]
                                )
                            else:
                                all_flow_payload.update(
                                    {
                                        "all_flow_status": (
                                            ResultStatus.UNSUPPORTED.value
                                        ),
                                        "all_flow_failure_message": (
                                            "AmpliCol fixed-helicity all-flow probe "
                                            "is unsupported for more than three "
                                            "quark lines"
                                        ),
                                    }
                                )
                    finally:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        measurement = {
            **_empty_measurement(),
            "status": ResultStatus.OK.value,
            "generation_seconds": generation_seconds,
            "sample_count": runtime_sample_count,
            "wall_seconds_per_point": runtime_seconds,
            "evaluator_seconds_per_point": None,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": matrix_element,
            "requested_config": {
                "method": "legacy_amplicol_generated_library",
                "color_accuracy": color_accuracy,
                "point_source": point_source,
                "jobs": max(1, int(jobs)),
                **_legacy_profile_requested_config(target_runtime),
            },
            "effective_config": {
                "process_file": os.fspath(process_file),
                "matching_row_count": len(matches),
                "row_id": f"group:{entry.group}:integral:{entry.integral}",
                "row_selection_policy": (
                    legacy_amplicol.GENERATED_PROCESS_ROW_SELECTION_POLICY
                ),
                "legacy_process_list_flags": (
                    [] if family is None else family.legacy_process_list_flags()
                ),
            },
            "environment": {
                "repository": os.fspath(repository),
                "revision": legacy_amplicol.expected_revision(),
                "report_source": _report_source_provenance(),
            },
            "artifact_path": os.fspath(legacy_root),
            "log_path": os.fspath(log_path),
            "manifest_path": os.fspath(manifest_path),
            "limit_gib": limit_gib,
            "command": command,
            "metadata": {
                "cell": cell.as_json(),
                "timing_method": runtime_probe,
                "runtime_profile": runtime_profile,
                "matrix_element_probe": matrix_element_probe,
                "point_source": point_source,
                "source_provenance": _report_source_provenance(),
                "old_matrix_format": {
                    "status": ResultStatus.OK.value,
                    "generation_s": generation_seconds,
                    "runtime_us_per_point": 1.0e6 * runtime_seconds,
                    "reference_probe": matrix_element_probe,
                    "runtime_probe": runtime_probe,
                    "runtime_profile": runtime_profile,
                    "runtime_probe_points": runtime_sample_count,
                    "process_file": os.fspath(process_file),
                    "process_list_backend": "legacy",
                    "reference_color_order": list(mapped_color_order),
                    "reference_color_order_process_file": list(entry.color_order),
                    **selected_probe_metadata,
                    "row_selection_policy": (
                        legacy_amplicol.GENERATED_PROCESS_ROW_SELECTION_POLICY
                    ),
                    "timing_rows": timing_rows,
                    "selected_flow_probe_timing_rows": selected_probe_rows,
                    "commands": [
                        *generation_records,
                        *timing_commands,
                        *all_flow_commands,
                    ],
                    **all_flow_payload,
                },
            },
        }
        manifest = {
            "cell": cell.as_json(),
            "measurement": measurement,
            "source_provenance": _report_source_provenance(),
            "captured_at": _utc_now(),
        }
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")
        return measurement
    except Exception as exc:
        status = (
            ResultStatus.UNSUPPORTED
            if (
                type(exc).__name__ == "LegacyOracleError"
                or _legacy_probe_scope_limited(exc)
            )
            else ResultStatus.ERROR
        )
        return _failure_measurement(
            status,
            str(exc),
            failure_kind=type(exc).__name__,
            artifact_path=legacy_root,
            log_path=log_path,
            manifest_path=manifest_path,
            limit_gib=limit_gib,
            command=command,
                metadata={
                    "cell": cell.as_json(),
                    "source_provenance": _report_source_provenance(),
                    "old_matrix_format": {
                        "status": status.value,
                        "all_flow_status": status.value,
                },
            },
        )


def _pointwise_validation(
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
    *,
    require_all_flow: bool = False,
) -> dict[str, object]:
    payload = _empty_validation()
    if not _measurement_ok(legacy) or not _measurement_ok(pyamplicol):
        return payload
    reference = legacy.get("matrix_element")
    observed = pyamplicol.get("matrix_element")
    if reference is None or observed is None:
        payload.update(
            {
                "status": ResultStatus.ERROR.value,
                "message": "missing matrix element for pointwise validation",
            }
        )
        return payload
    absolute, relative, status = _matrix_element_difference(reference, observed)
    payload.update(
        {
            "status": status,
            "reference_matrix_element": float(reference),
            "pyamplicol_matrix_element": float(observed),
            "absolute_difference": absolute,
            "relative_difference": relative,
            "point_source": "shared validation point",
            "message": (
                None if status == ResultStatus.OK.value else "pointwise mismatch"
            ),
        }
    )
    if not require_all_flow:
        return payload

    legacy_fields = _measurement_old_matrix_fields(legacy)
    pyamplicol_fields = _measurement_old_matrix_fields(pyamplicol)
    legacy_status = str(legacy_fields.get("all_flow_status", NA_STATUS))
    pyamplicol_status = str(pyamplicol_fields.get("all_flow_status", NA_STATUS))
    if (
        legacy_status != ResultStatus.OK.value
        or pyamplicol_status != ResultStatus.OK.value
    ):
        payload.update(
            {
                "status": ResultStatus.ERROR.value,
                "all_flow_status": ResultStatus.ERROR.value,
                "message": (
                    "all-flow validation measurement is unavailable: "
                    f"AmpliCol={legacy_status}, pyAmpliCol={pyamplicol_status}"
                ),
            }
        )
        return payload
    all_flow_reference = legacy_fields.get("all_flow_reference_value")
    all_flow_observed = pyamplicol_fields.get("all_flow_matrix_element")
    if all_flow_reference is None or all_flow_observed is None:
        payload.update(
            {
                "status": ResultStatus.ERROR.value,
                "all_flow_status": ResultStatus.ERROR.value,
                "message": "missing matrix element for all-flow validation",
            }
        )
        return payload
    all_flow_absolute, all_flow_relative, all_flow_status = _matrix_element_difference(
        all_flow_reference, all_flow_observed
    )
    payload.update(
        {
            "all_flow_status": all_flow_status,
            "all_flow_reference_matrix_element": float(all_flow_reference),
            "all_flow_pyamplicol_matrix_element": float(all_flow_observed),
            "all_flow_absolute_difference": all_flow_absolute,
            "all_flow_relative_difference": all_flow_relative,
        }
    )
    if all_flow_status != ResultStatus.OK.value:
        payload.update(
            {
                "status": all_flow_status,
                "message": "all-flow pointwise mismatch",
            }
        )
    return payload


def _matrix_element_difference(
    reference: object,
    observed: object,
) -> tuple[float, float, str]:
    absolute = abs(float(reference) - float(observed))
    relative = absolute / max(abs(float(reference)), 1.0e-300)
    status = (
        ResultStatus.OK.value
        if (
            absolute <= VALIDATION_ABSOLUTE_TOLERANCE
            or relative <= VALIDATION_RELATIVE_TOLERANCE
        )
        else ResultStatus.VALIDATION_FAILED.value
    )
    return absolute, relative, status


def _parameter_alignment_for(
    spec: MatrixSpec,
    artifact_root: Path,
    cell: CampaignCell,
) -> dict[str, object]:
    payload = _empty_parameter_alignment()
    snapshot_path = (
        artifact_root
        / "cells"
        / cell.cell_id
        / "inputs"
        / "sm-parameter-alignment.json"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    ufo_source = _model_source_path(EXTERNAL_SM)
    snapshot = {
        "status": ResultStatus.OK.value,
        "built_in_sm_source": "built-in-sm",
        "ufo_sm_source": None if ufo_source is None else os.fspath(ufo_source),
        "comparison_reference": "original Fortran AmpliCol",
        "model_profile": spec.model.profile,
        "source_provenance": _report_source_provenance(),
        "captured_at": _utc_now(),
    }
    snapshot_path.write_text(_json_text(snapshot), encoding="utf-8")
    payload.update(
        {
            "status": ResultStatus.OK.value,
            "ufo_sm_source": None if ufo_source is None else os.fspath(ufo_source),
            "snapshot_path": os.fspath(snapshot_path),
            "message": (
                "built-in SM and UFO SM use the dev-install packaged numerical inputs"
            ),
        }
    )
    return payload


def _entry_from_measurements(
    *,
    base_entry: Mapping[str, object],
    legacy: Mapping[str, object],
    pyamplicol: Mapping[str, object],
    validation: Mapping[str, object],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    entry = dict(base_entry)
    entry.update(
        {
            "legacy_amplicol": dict(legacy),
            "pyamplicol_jit_o3": dict(pyamplicol),
            "pointwise_validation": dict(validation),
            "parameter_alignment": dict(alignment),
        }
    )
    _refresh_matrix_derived_fields(entry)
    return entry


def _measure_cell_payload(
    cell: CampaignCell,
    *,
    artifact_root: Path,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
    limit_gib: float,
) -> dict[str, object]:
    spec = _spec_by_dataset()[cell.dataset_id]
    if isinstance(spec, MatrixSpec):
        previous_entry = _previous_cache_entry_for_cell(cell)
        previous_pyamplicol = (
            previous_entry.get("pyamplicol_jit_o3")
            if isinstance(previous_entry, Mapping)
            else None
        )
        base_entry = {
            "process_key": cell.process_key,
            "n_final": cell.n_final,
            "process": cell.process,
            "applicable": True,
            "status": NA_STATUS,
            "legacy_amplicol": _empty_measurement(),
            "pyamplicol_jit_o3": _empty_measurement(),
            "reference": _empty_measurement(),
            "pyamplicol": _empty_measurement(),
            "generation_multiplier": None,
            "runtime_multiplier": None,
            "pointwise_validation": _empty_validation(),
            "parameter_alignment": _empty_parameter_alignment(),
            "relative_difference": None,
        }
        shared_particles = _shared_validation_particles(cell.process)
        points = _pyamplicol_points_from_particles(shared_particles)
        fixed_helicity = (
            _fixed_source_helicity_choice(
                cell.process,
                spec=spec,
                artifact_root=artifact_root,
            )
            if spec.color_accuracy == "lc"
            else None
        )
        legacy = _measure_legacy_amplicol(
            cell=cell,
            color_accuracy=spec.color_accuracy,
            artifact_root=artifact_root,
            points=points,
            limit_gib=limit_gib,
            target_runtime=target_runtime,
            jobs=cell_cores,
            fixed_helicity=fixed_helicity,
        )
        pyamplicol, points = _measure_pyamplicol_matrix_jit_o3(
            cell=cell,
            spec=spec,
            legacy=legacy,
            artifact_root=artifact_root,
            generation_timeout_seconds=generation_timeout_seconds,
            target_runtime=target_runtime,
            cell_cores=cell_cores,
            points=points,
            fixed_helicity=fixed_helicity,
            previous_measurement=(
                previous_pyamplicol
                if isinstance(previous_pyamplicol, Mapping)
                else None
            ),
        )
        validation = _pointwise_validation(
            legacy,
            pyamplicol,
            require_all_flow=(
                spec.color_accuracy == "lc" and _lc_all_flow_supported(cell.process)
            ),
        )
        alignment = _parameter_alignment_for(spec, artifact_root, cell)
        return {
            "cell": cell.as_json(),
            "cache_name": cell.cache_name,
            "entry": _entry_from_measurements(
                base_entry=base_entry,
                legacy=legacy,
                pyamplicol=pyamplicol,
                validation=validation,
                alignment=alignment,
            ),
        }
    assert isinstance(spec, LadderSpec)
    if spec.kind == CacheKind.PERFORMANCE_LADDER:
        previous_entry = _previous_cache_entry_for_cell(cell)
        previous_measurement = (
            previous_entry.get("measurement")
            if isinstance(previous_entry, Mapping)
            else None
        )
        shared_particles = _shared_validation_particles(cell.process)
        points = _pyamplicol_points_from_particles(shared_particles)
        variant = next(item for item in spec.variants if item.key == cell.variant)
        if variant.key == "reference":
            measurement = _measure_legacy_amplicol(
                cell=cell,
                color_accuracy="lc",
                artifact_root=artifact_root,
                points=points,
                limit_gib=limit_gib,
                target_runtime=target_runtime,
                jobs=cell_cores,
                fixed_helicity=_fixed_source_helicity_choice(
                    cell.process,
                    spec=spec,
                    artifact_root=artifact_root,
                ),
            )
        else:
            measurement, _points = _measure_pyamplicol_lc_two_workloads(
                cell=cell,
                spec=spec,
                variant_overrides=variant.config_overrides,
                legacy=None,
                artifact_root=artifact_root,
                generation_timeout_seconds=generation_timeout_seconds,
                target_runtime=target_runtime,
                cell_cores=cell_cores,
                points=points,
                previous_measurement=(
                    previous_measurement
                    if isinstance(previous_measurement, Mapping)
                    else None
                ),
            )
        return {
            "cell": cell.as_json(),
            "cache_name": cell.cache_name,
            "entry": {
                "n_final": cell.n_final,
                "process": cell.process,
                "variant": cell.variant,
                "status": measurement["status"],
                "measurement": measurement,
            },
        }
    previous_model_entry = _previous_cache_entry_for_cell(cell)
    previous_model_measurement = (
        previous_model_entry.get("measurement")
        if isinstance(previous_model_entry, Mapping)
        else None
    )
    measurement, _points = _measure_pyamplicol(
        cell=cell,
        spec=spec,
        color_accuracy="lc",
        variant_overrides={},
        artifact_root=artifact_root,
        generation_timeout_seconds=generation_timeout_seconds,
        target_runtime=target_runtime,
        cell_cores=cell_cores,
        high_precision=True,
        previous_measurement=(
            previous_model_measurement
            if isinstance(previous_model_measurement, Mapping)
            else None
        ),
    )
    metadata = measurement.get("metadata", {})
    high_precision_value = None
    relative_difference = None
    if isinstance(metadata, Mapping):
        high_precision_value = metadata.get("high_precision_matrix_element")
        relative_difference = metadata.get("high_precision_relative_difference")
    return {
        "cell": cell.as_json(),
        "cache_name": cell.cache_name,
        "entry": {
            "n_final": cell.n_final,
            "process": cell.process,
            "status": measurement["status"],
            "measurement": measurement,
            "high_precision_matrix_element": high_precision_value,
            "relative_difference": relative_difference,
        },
    }


def _cell_from_json(payload: Mapping[str, object]) -> CampaignCell:
    kind = str(payload["kind"])
    if kind not in {"matrix", "performance_ladder", "model_ladder"}:
        raise ValueError(f"invalid campaign cell kind {kind!r}")
    return CampaignCell(
        kind=kind,  # type: ignore[arg-type]
        cache_name=str(payload["cache_name"]),
        dataset_id=str(payload["dataset_id"]),
        n_final=int(payload["n_final"]),
        process=str(payload["process"]),
        process_key=(
            None if payload.get("process_key") is None else str(payload["process_key"])
        ),
        variant=None if payload.get("variant") is None else str(payload["variant"]),
    )


def _failure_entry_for_cell(
    cell: CampaignCell,
    *,
    status: ResultStatus,
    message: str,
    artifact_root: Path,
    limit_gib: float,
    timeout_seconds: float,
) -> dict[str, object]:
    measurement = _failure_measurement(
        status,
        message,
        artifact_path=artifact_root / "cells" / cell.cell_id,
        log_path=artifact_root / "cells" / cell.cell_id / "logs" / "worker.log",
        limit_gib=limit_gib,
        timeout_seconds=timeout_seconds,
        metadata={
            "cell": cell.as_json(),
            "source_provenance": _report_source_provenance(),
        },
    )
    spec = _spec_by_dataset()[cell.dataset_id]
    if isinstance(spec, MatrixSpec):
        base_entry = {
            "process_key": cell.process_key,
            "n_final": cell.n_final,
            "process": cell.process,
            "applicable": True,
            "status": status.value,
            "legacy_amplicol": measurement,
            "pyamplicol_jit_o3": measurement,
            "reference": measurement,
            "pyamplicol": measurement,
            "generation_multiplier": None,
            "runtime_multiplier": None,
            "pointwise_validation": _empty_validation(),
            "parameter_alignment": _empty_parameter_alignment(),
            "relative_difference": None,
        }
        _refresh_matrix_derived_fields(base_entry)
        return base_entry
    if isinstance(spec, LadderSpec) and spec.kind == CacheKind.PERFORMANCE_LADDER:
        return {
            "n_final": cell.n_final,
            "process": cell.process,
            "variant": cell.variant,
            "status": status.value,
            "measurement": measurement,
        }
    return {
        "n_final": cell.n_final,
        "process": cell.process,
        "status": status.value,
        "measurement": measurement,
        "high_precision_matrix_element": None,
        "relative_difference": None,
    }


def _merge_cell_entry(
    caches: dict[str, dict[str, object]],
    *,
    cell: CampaignCell,
    entry: Mapping[str, object],
) -> None:
    payload = caches[cell.cache_name]
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise TypeError("cache entries must be a list")
    for index, existing in enumerate(entries):
        if not isinstance(existing, Mapping):
            continue
        if cell.kind == "matrix":
            if (
                existing.get("process_key") == cell.process_key
                and existing.get("n_final") == cell.n_final
            ):
                entries[index] = dict(entry)
                break
        elif cell.kind == "performance_ladder":
            if (
                existing.get("n_final") == cell.n_final
                and existing.get("variant") == cell.variant
            ):
                entries[index] = dict(entry)
                break
        else:
            if existing.get("n_final") == cell.n_final:
                entries[index] = dict(entry)
                break
    else:
        raise ValueError(f"could not find cache entry for {cell.cell_id}")
    if cell.kind == "performance_ladder":
        _refresh_performance_ladder_validation(payload, n_final=cell.n_final)
    payload["updated_at"] = _utc_now()
    normalized = normalize_cache_payload(payload)
    validate_cache(normalized)
    caches[cell.cache_name] = normalized


def _refresh_performance_ladder_validation(
    payload: dict[str, object],
    *,
    n_final: int,
) -> None:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return
    reference: Mapping[str, object] | None = None
    for candidate in entries:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("n_final") == n_final
            and candidate.get("variant") == "reference"
            and isinstance(candidate.get("measurement"), Mapping)
        ):
            reference = candidate["measurement"]  # type: ignore[assignment]
            break
    if reference is None or not _measurement_ok(reference):
        return
    for index, candidate in enumerate(entries):
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("n_final") != n_final
            or candidate.get("variant") == "reference"
        ):
            continue
        measurement_value = candidate.get("measurement")
        if not isinstance(measurement_value, Mapping):
            continue
        measurement = dict(measurement_value)
        validation = _pointwise_validation(
            reference,
            measurement,
            require_all_flow=True,
        )
        metadata_value = measurement.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        metadata["pointwise_validation"] = validation
        measurement["metadata"] = metadata
        updated = dict(candidate)
        updated["measurement"] = measurement
        if (
            validation.get("status") == ResultStatus.VALIDATION_FAILED.value
            or validation.get("all_flow_status")
            == ResultStatus.VALIDATION_FAILED.value
        ):
            updated["status"] = ResultStatus.VALIDATION_FAILED.value
        else:
            updated["status"] = measurement.get("status", NA_STATUS)
        entries[index] = updated


def _worker_command(
    *,
    python: Path,
    cell: CampaignCell,
    result_json: Path,
    artifact_root: Path,
    limit_gib: float,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
) -> list[str]:
    return [
        os.fspath(python),
        "tools/ci/memory_watchdog.py",
        "--limit-gib",
        f"{limit_gib:g}",
        "--",
        os.fspath(python),
        "docs/result_tables.py",
        "measure-cell",
        "--cell-json",
        json.dumps(cell.as_json(), sort_keys=True),
        "--result-json",
        os.fspath(result_json),
        "--artifact-root",
        os.fspath(artifact_root),
        "--limit-gib",
        f"{limit_gib:g}",
        "--generation-timeout-seconds",
        f"{generation_timeout_seconds:g}",
        "--target-runtime",
        f"{target_runtime:g}",
        "--cell-cores",
        str(max(1, cell_cores)),
    ]


def _terminate_worker_process(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 30.0,
) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _terminate_active_worker_processes(
    *,
    grace_seconds: float = 30.0,
) -> int:
    with _ACTIVE_WORKER_LOCK:
        processes = tuple(_ACTIVE_WORKER_PROCESSES)
    for process in processes:
        _terminate_worker_process(process, grace_seconds=grace_seconds)
    return len(processes)


def _run_worker_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float | None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"# worker command started {_utc_now()}\n".encode())
        log.write((" ".join(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with _ACTIVE_WORKER_LOCK:
            _ACTIVE_WORKER_PROCESSES.add(process)
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_worker_process(process)
            return 124
        finally:
            with _ACTIVE_WORKER_LOCK:
                _ACTIVE_WORKER_PROCESSES.discard(process)


def _campaign_worker_timeout_seconds(
    cell: CampaignCell,
    generation_timeout_seconds: float,
) -> float | None:
    if generation_timeout_seconds <= 0:
        return None
    if cell.kind == "matrix":
        spec = _spec_by_dataset()[cell.dataset_id]
        assert isinstance(spec, MatrixSpec)
        generation_workloads = 3 if spec.color_accuracy == "lc" else 2
    elif cell.kind == "performance_ladder":
        generation_workloads = 1 if cell.variant == "reference" else 2
    else:
        generation_workloads = 1
    return generation_workloads * generation_timeout_seconds + 900.0


def _worker_log_reports_memory_limit(log_path: Path) -> bool:
    try:
        return "memory-watchdog: RSS limit exceeded:" in log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False


def _execute_campaign_cell(
    cell: CampaignCell,
    *,
    python: Path,
    artifact_root: Path,
    limit_gib: float,
    generation_timeout_seconds: float,
    target_runtime: float,
    cell_cores: int,
) -> dict[str, object]:
    cell_root = artifact_root / "cells" / cell.cell_id
    result_json = cell_root / "result.json"
    worker_log = cell_root / "logs" / "worker.log"
    with contextlib.suppress(FileNotFoundError):
        result_json.unlink()
    command = _worker_command(
        python=python,
        cell=cell,
        result_json=result_json,
        artifact_root=artifact_root,
        limit_gib=limit_gib,
        generation_timeout_seconds=generation_timeout_seconds,
        target_runtime=target_runtime,
        cell_cores=cell_cores,
    )
    worker_timeout_seconds = _campaign_worker_timeout_seconds(
        cell,
        generation_timeout_seconds,
    )
    code = _run_worker_command(
        command,
        cwd=_repo_root(),
        log_path=worker_log,
        timeout_seconds=worker_timeout_seconds,
    )
    if code == 0 and result_json.is_file():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{result_json} must contain an object")
        return payload
    if code == 137 and _worker_log_reports_memory_limit(worker_log):
        status = ResultStatus.MEMORY_LIMIT
        message = f"memory watchdog exceeded {limit_gib:g} GiB"
    elif code == 124:
        status = ResultStatus.TIMEOUT
        message = (
            "worker exceeded its aggregate supervision budget"
            if worker_timeout_seconds is None
            else f"worker exceeded {worker_timeout_seconds:g} second aggregate budget"
        )
    else:
        status = ResultStatus.ERROR
        message = (
            "worker exited with code 137 without a memory-watchdog limit marker"
            if code == 137
            else f"worker exited with code {code}"
        )
    return {
        "cell": cell.as_json(),
        "cache_name": cell.cache_name,
        "entry": _failure_entry_for_cell(
            cell,
            status=status,
            message=message,
            artifact_root=artifact_root,
            limit_gib=limit_gib,
            timeout_seconds=(
                generation_timeout_seconds
                if worker_timeout_seconds is None
                else worker_timeout_seconds
            ),
        ),
    }


def _truthy_environment_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _campaign_worker_selection(
    requested_workers: int,
    cell_count: int,
    *,
    allow_symbolica_parallel: bool,
) -> tuple[int, int, int, str | None]:
    requested = max(1, int(requested_workers))
    scheduled_cap = min(requested, max(1, int(cell_count)))
    if scheduled_cap <= 1 or allow_symbolica_parallel:
        return requested, scheduled_cap, scheduled_cap, None
    reason = (
        "parallel worker processes are disabled for this dev-install campaign "
        "because these cells instantiate Symbolica and the current host only "
        "permits one unlicensed Symbolica instance at a time"
    )
    return requested, scheduled_cap, 1, reason


@contextmanager
def _report_lock(paths: ReportPaths) -> Iterable[None]:
    digest = hashlib.sha256(os.fspath(paths.docs_dir).encode("utf-8")).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"pyamplicol-report-{digest}.lock"
    stream: BinaryIO | None = None
    try:
        stream = lock_path.open("a+b")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if stream is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _write_staged(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs: dict[str, object] = {}
    if isinstance(content, str):
        kwargs["encoding"] = "utf-8"
        kwargs["newline"] = "\n"
    with path.open(mode, **kwargs) as stream:  # type: ignore[arg-type]
        stream.write(content)  # type: ignore[arg-type]
        stream.flush()
        os.fsync(stream.fileno())


def _compile_staged_pdf(paths: ReportPaths, staging: Path) -> bytes:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is required for --compile")
    shutil.copy2(paths.report_tex, staging / paths.report_tex.name)
    for name in REPORT_TEX_INPUTS:
        source = paths.docs_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"report TeX input is missing: {name}")
        shutil.copy2(source, staging / name)
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
    completed = subprocess.run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            paths.report_tex.name,
        ],
        cwd=staging,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"latexmk failed with exit {completed.returncode}:\n{tail}")
    pdf_path = staging / paths.report_pdf.name
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("latexmk did not produce a non-empty report PDF")
    return pdf_path.read_bytes()


def _publish_files(
    paths: ReportPaths,
    files: Mapping[Path, str | bytes],
) -> tuple[Path, ...]:
    """Publish a staged file set with rollback if any replacement fails."""

    for relative in files:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"report output must be relative: {relative}")
    staging = paths.docs_dir / f".performance-report-staging-{uuid.uuid4().hex}"
    backup_root = staging / ".backup"
    staging.mkdir(parents=False)
    try:
        for relative, content in files.items():
            _write_staged(staging / relative, content)
        published: list[Path] = []
        backed_up: list[Path] = []
        try:
            for relative in files:
                source = staging / relative
                destination = paths.docs_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = backup_root / relative
                if destination.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, backup)
                    backed_up.append(relative)
                os.replace(source, destination)
                published.append(relative)
        except BaseException:
            for relative in reversed(published):
                destination = paths.docs_dir / relative
                if destination.exists():
                    destination.unlink()
            for relative in reversed(backed_up):
                backup = backup_root / relative
                if backup.exists():
                    destination = paths.docs_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, destination)
            raise
        return tuple(paths.docs_dir / relative for relative in files)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class ReportService:
    """Validate, render, and publish the complete report result set."""

    def __init__(self, paths: ReportPaths | None = None) -> None:
        self.paths = paths or ReportPaths.default()

    def validate(self) -> dict[str, dict[str, object]]:
        with _report_lock(self.paths):
            caches = load_caches(self.paths)
            expected_schema = schema_document()
            actual_schema = json.loads(
                self.paths.schema_path.read_text(encoding="utf-8")
            )
            if actual_schema != expected_schema:
                raise ValueError("checked-in report-cache.schema.json is stale")
            expected_tables = render_tables(caches)
            for name, expected in expected_tables.items():
                actual = (self.paths.docs_dir / name).read_text(encoding="utf-8")
                if actual != expected:
                    raise ValueError(f"checked-in generated table is stale: {name}")
            return caches

    def reset(self, *, compile_pdf: bool = False) -> tuple[Path, ...]:
        return self._refresh(build_reset_caches(), compile_pdf=compile_pdf)

    def render(self, *, compile_pdf: bool = False) -> tuple[Path, ...]:
        return self._refresh(load_caches(self.paths), compile_pdf=compile_pdf)

    def populate(
        self,
        cells: Sequence[CampaignCell],
        *,
        workers: int,
        python: Path,
        artifact_root: Path,
        limit_gib: float,
        generation_timeout_seconds: float,
        target_runtime: float,
        cell_cores: int,
        refresh_pdf: Literal["always", "never"],
        allow_symbolica_parallel: bool = False,
    ) -> None:
        artifact_root = artifact_root.expanduser().resolve(strict=False)
        artifact_root.mkdir(parents=True, exist_ok=True)
        run_log = artifact_root / "runs" / f"{_utc_now().replace(':', '')}.jsonl"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        requested_workers, scheduled_worker_cap, effective_workers, limit_reason = (
            _campaign_worker_selection(
                workers,
                len(cells),
                allow_symbolica_parallel=allow_symbolica_parallel,
            )
        )
        if limit_reason is not None:
            print(
                "campaign worker limit: "
                f"requested={requested_workers} "
                f"scheduled_cap={scheduled_worker_cap} "
                f"effective={effective_workers}; {limit_reason}",
                file=sys.stderr,
            )
        with run_log.open("a", encoding="utf-8") as events:
            events.write(
                json.dumps(
                    {
                        "event": "campaign-start",
                        "started_at": _utc_now(),
                        "requested_workers": requested_workers,
                        "scheduled_worker_cap": scheduled_worker_cap,
                        "workers": effective_workers,
                        "effective_workers": effective_workers,
                        "limit_gib": limit_gib,
                        "generation_timeout_seconds": generation_timeout_seconds,
                        "target_runtime": target_runtime,
                        "cell_count": len(cells),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if limit_reason is not None:
                events.write(
                    json.dumps(
                        {
                            "event": "campaign-worker-limit",
                            "recorded_at": _utc_now(),
                            "requested_workers": requested_workers,
                            "scheduled_worker_cap": scheduled_worker_cap,
                            "effective_workers": effective_workers,
                            "reason": limit_reason,
                            "override_cli": "--allow-symbolica-parallel",
                            "override_env": ALLOW_PARALLEL_SYMBOLICA_ENV,
                        },
                        sort_keys=True,
                    )
                    + "\n"
            )
            events.flush()
            print(
                "campaign start: "
                f"cells={len(cells)} workers={effective_workers} "
                f"target_runtime={target_runtime:g}s "
                f"artifact_root={artifact_root}",
                file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            completed = 0
            status_counts: dict[str, int] = {}
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=effective_workers
            )
            futures: dict[
                concurrent.futures.Future[dict[str, object]], CampaignCell
            ] = {}
            try:
                futures = {
                    executor.submit(
                        _execute_campaign_cell,
                        cell,
                        python=python,
                        artifact_root=artifact_root,
                        limit_gib=limit_gib,
                        generation_timeout_seconds=generation_timeout_seconds,
                        target_runtime=target_runtime,
                        cell_cores=cell_cores,
                    ): cell
                    for cell in cells
                }
                for future in concurrent.futures.as_completed(futures):
                    cell = futures[future]
                    try:
                        payload = future.result()
                    except Exception as exc:
                        payload = {
                            "cell": cell.as_json(),
                            "cache_name": cell.cache_name,
                            "entry": _failure_entry_for_cell(
                                cell,
                                status=ResultStatus.ERROR,
                                message=str(exc),
                                artifact_root=artifact_root,
                                limit_gib=limit_gib,
                                timeout_seconds=generation_timeout_seconds,
                            ),
                        }
                    raw_cell = payload.get("cell")
                    if not isinstance(raw_cell, Mapping):
                        raise TypeError("worker payload is missing cell metadata")
                    completed_cell = _cell_from_json(raw_cell)
                    raw_entry = payload.get("entry")
                    if not isinstance(raw_entry, Mapping):
                        raise TypeError("worker payload is missing entry data")
                    self._merge_and_refresh(
                        completed_cell,
                        raw_entry,
                        compile_pdf=(refresh_pdf == "always"),
                    )
                    completed += 1
                    status = str(raw_entry.get("status", NA_STATUS))
                    status_counts[status] = status_counts.get(status, 0) + 1
                    elapsed = time.perf_counter() - started
                    events.write(
                        json.dumps(
                            {
                                "event": "cell-merged",
                                "merged_at": _utc_now(),
                                "cell": completed_cell.as_json(),
                                "status": status,
                                "completed": completed,
                                "cell_count": len(cells),
                                "elapsed_seconds": elapsed,
                                "status_counts": dict(sorted(status_counts.items())),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    events.flush()
                    print(
                        "campaign progress: "
                        f"{completed}/{len(cells)} merged "
                        f"elapsed={elapsed / 60.0:.1f}m "
                        f"status={status} "
                        f"counts={dict(sorted(status_counts.items()))} "
                        f"cell={completed_cell.cell_id}",
                        file=sys.stderr,
                        flush=True,
                    )
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                terminated = _terminate_active_worker_processes()
                for future in futures:
                    future.cancel()
                elapsed = time.perf_counter() - started
                events.write(
                    json.dumps(
                        {
                            "event": "campaign-interrupted",
                            "interrupted_at": _utc_now(),
                            "completed": completed,
                            "cell_count": len(cells),
                            "elapsed_seconds": elapsed,
                            "active_workers_terminated": terminated,
                            "status_counts": dict(sorted(status_counts.items())),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                events.flush()
                print(
                    "campaign interrupted: "
                    f"completed={completed}/{len(cells)} "
                    f"terminated_active_workers={terminated}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    def _merge_and_refresh(
        self,
        cell: CampaignCell,
        entry: Mapping[str, object],
        *,
        compile_pdf: bool,
    ) -> tuple[Path, ...]:
        """Merge against the latest on-disk caches while holding the writer lock."""

        with _report_lock(self.paths):
            caches = load_caches(self.paths)
            _merge_cell_entry(caches, cell=cell, entry=entry)
            return self._refresh_locked(caches, compile_pdf=compile_pdf)

    def _refresh(
        self,
        caches: Mapping[str, Mapping[str, object]],
        *,
        compile_pdf: bool,
    ) -> tuple[Path, ...]:
        with _report_lock(self.paths):
            return self._refresh_locked(caches, compile_pdf=compile_pdf)

    def _refresh_locked(
        self,
        caches: Mapping[str, Mapping[str, object]],
        *,
        compile_pdf: bool,
    ) -> tuple[Path, ...]:
        for payload in caches.values():
            validate_cache(payload)
        tables = render_tables(caches)
        files: dict[Path, str | bytes] = {
            Path("results/report-cache.schema.json"): _json_text(schema_document()),
        }
        files.update(
            {
                Path("results") / name: _json_text(payload)
                for name, payload in caches.items()
            }
        )
        files.update({Path(name): table for name, table in tables.items()})
        if compile_pdf:
            compile_stage = self.paths.docs_dir / (
                f".performance-report-compile-{uuid.uuid4().hex}"
            )
            compile_stage.mkdir()
            try:
                for relative, content in files.items():
                    _write_staged(compile_stage / relative, content)
                pdf = _compile_staged_pdf(self.paths, compile_stage)
                files[Path(self.paths.report_pdf.name)] = pdf
            finally:
                shutil.rmtree(compile_stage, ignore_errors=True)
        return _publish_files(self.paths, files)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or explicitly refresh the standalone pyAmpliCol "
            "performance report."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate caches and generated tables.")
    for command, help_text in (
        ("reset", "Replace all result caches with canonical N/A entries."),
        ("render", "Render the current validated caches."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--compile",
            action="store_true",
            help="Compile and publish pyAmpliCol.pdf in the same transaction.",
        )
    populate = subparsers.add_parser(
        "populate",
        help="Run or dry-run the diagnostics/performance campaign.",
    )
    populate.add_argument("--dry-run", action="store_true")
    populate.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    populate.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    populate.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=DEFAULT_GENERATION_TIMEOUT_SECONDS,
    )
    populate.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_REPORT_TARGET_RUNTIME_SECONDS,
    )
    populate.add_argument("--cell-cores", type=int, default=DEFAULT_PARALLEL_CELL_CORES)
    populate.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    populate.add_argument("--python", type=Path, default=DEFAULT_DEV_PYTHON)
    populate.add_argument(
        "--refresh-pdf",
        choices=("always", "never"),
        default="always",
    )
    populate.add_argument(
        "--allow-symbolica-parallel",
        action="store_true",
        help=(
            "Permit concurrent worker processes. By default populate serializes "
            "worker subprocesses because the dev-install Symbolica runtime on "
            "this host allows only one unlicensed instance."
        ),
    )
    populate.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Restrict to one dataset_id; repeat for multiple datasets.",
    )
    populate.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Restrict to one exact campaign cell_id from populate --dry-run; "
            "repeat for multiple cells."
        ),
    )
    populate.add_argument(
        "--process",
        action="append",
        default=None,
        help=(
            "Restrict to one exact generated process expression, for example "
            "'d d~ > z g g'; repeat for multiple expressions."
        ),
    )
    populate.add_argument(
        "--process-key",
        action="append",
        default=None,
        help="Restrict matrix cells to one process key; repeat for multiple keys.",
    )
    populate.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Restrict performance-ladder cells to one variant; repeat for multiple.",
    )
    populate.add_argument(
        "--n-final",
        action="append",
        type=int,
        default=None,
        help="Restrict to one final-state multiplicity; repeat for multiple.",
    )
    populate.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="Run only the first N fast-first cells; useful for smoke tests.",
    )
    populate.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "Schedule only cells that are N/A or stale for the current report "
            "schema. Old generic Z rows without selected/all-flow metadata are "
            "treated as stale."
        ),
    )
    worker = subparsers.add_parser(
        "measure-cell",
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--cell-json", required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    worker.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    worker.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=DEFAULT_GENERATION_TIMEOUT_SECONDS,
    )
    worker.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_REPORT_TARGET_RUNTIME_SECONDS,
    )
    worker.add_argument("--cell-cores", type=int, default=DEFAULT_PARALLEL_CELL_CORES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    service = ReportService()
    if args.command == "validate":
        caches = service.validate()
        print(f"validated {len(caches)} caches and {len(TABLE_INPUTS)} tables")
        return 0
    if args.command == "populate":
        if args.cell_id is not None:
            unknown_cell_ids = sorted(set(args.cell_id) - _known_cell_ids())
            if unknown_cell_ids:
                parser.error(
                    f"unknown --cell-id value(s): {', '.join(unknown_cell_ids)}"
                )
        if args.process is not None:
            requested_processes = {
                _normalized_process_expression(process) for process in args.process
            }
            unknown_processes = sorted(
                requested_processes - _known_process_expressions()
            )
            if unknown_processes:
                parser.error(
                    f"unknown --process expression(s): {', '.join(unknown_processes)}"
                )
        caches = load_caches(service.paths) if args.missing_only else None
        cells = _select_cells(
            datasets=None if args.dataset is None else set(args.dataset),
            cell_ids=None if args.cell_id is None else set(args.cell_id),
            processes=None if args.process is None else set(args.process),
            process_keys=(None if args.process_key is None else set(args.process_key)),
            variants=None if args.variant is None else set(args.variant),
            n_values=None if args.n_final is None else set(args.n_final),
            limit=args.limit_cells,
            missing_only=bool(args.missing_only),
            caches=caches,
        )
        if args.dry_run:
            for cell in cells:
                print(json.dumps(cell.as_json(), sort_keys=True))
            print(f"planned {len(cells)} cells", file=sys.stderr)
            return 0
        try:
            service.populate(
                cells,
                workers=args.workers,
                python=args.python,
                artifact_root=args.artifact_root,
                limit_gib=args.limit_gib,
                generation_timeout_seconds=args.generation_timeout_seconds,
                target_runtime=args.target_runtime,
                cell_cores=args.cell_cores,
                refresh_pdf=args.refresh_pdf,
                allow_symbolica_parallel=(
                    args.allow_symbolica_parallel
                    or _truthy_environment_flag(ALLOW_PARALLEL_SYMBOLICA_ENV)
                ),
            )
        except KeyboardInterrupt:
            print(
                "populate interrupted; active worker processes were terminated",
                file=sys.stderr,
            )
            return 130
        print(f"populated {len(cells)} scheduled cells")
        return 0
    if args.command == "measure-cell":
        cell_payload = json.loads(args.cell_json)
        if not isinstance(cell_payload, Mapping):
            raise TypeError("--cell-json must decode to an object")
        cell = _cell_from_json(cell_payload)
        result = _measure_cell_payload(
            cell,
            artifact_root=args.artifact_root,
            generation_timeout_seconds=args.generation_timeout_seconds,
            target_runtime=args.target_runtime,
            cell_cores=args.cell_cores,
            limit_gib=args.limit_gib,
        )
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(_json_text(result), encoding="utf-8")
        print(args.result_json)
        return 0
    compile_pdf = bool(args.compile)
    paths = (
        service.reset(compile_pdf=compile_pdf)
        if args.command == "reset"
        else service.render(compile_pdf=compile_pdf)
    )
    for path in paths:
        print(path.relative_to(service.paths.docs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
