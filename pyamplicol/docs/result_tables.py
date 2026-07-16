# SPDX-License-Identifier: 0BSD
"""Typed cache, table, and PDF services for the public performance report.

This module never generates physics artifacts on import or during a normal
render. Measurements are supplied explicitly by a caller after using the
public ``Generator`` and ``BenchmarkRunner`` APIs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol

if TYPE_CHECKING:
    pass


REPORT_VERSION = "0.1.0"
CACHE_SCHEMA_VERSION = 1
SPDX_LICENSE = "0BSD"
NA_STATUS = "not_available"

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
    "benchmark.target_runtime": 10.0,
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
        }


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


BUILTIN_SM = ModelSpec("built-in-sm", "Built-in Standard Model", "built-in-sm")
EXTERNAL_SM = ModelSpec(
    "external-sm",
    "External Standard Model (UFO/JSON)",
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
    ProcessFamily(2, "ud_w_jets", r"$u\bar d\to W^++(n-1)g$", ("u", "d~"), ("w+",), 9),
    ProcessFamily(
        3, "dd_epem_jets", r"$d\bar d\to e^+e^-+(n-2)g$", ("d", "d~"), ("e+", "e-"), 9
    ),
    ProcessFamily(
        4, "ud_epve_jets", r"$u\bar d\to e^+\nu_e+(n-2)g$", ("u", "d~"), ("e+", "ve"), 9
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
    ),
    ProcessFamily(
        14,
        "dd_4q_lines",
        r"$d\bar d\to u\bar u\,s\bar s\,c\bar c+(n-6)g$",
        ("d", "d~"),
        ("u", "u~", "s", "s~", "c", "c~"),
        8,
    ),
)


def _matrix_spec(
    model_key: str,
    model: ModelSpec,
    accuracy: Literal["lc", "nlc", "full"],
) -> MatrixSpec:
    accuracy_label = {"lc": "LC", "nlc": "NLC", "full": "full-colour"}[accuracy]
    model_title = "Built-in SM" if model is BUILTIN_SM else "External SM"
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
    VariantSpec(
        "jit_o3",
        "JIT level 3",
        {"evaluator.backend": "jit", "evaluator.jit.optimization_level": 3},
    ),
    VariantSpec("asm_o3", "ASM O3", {"evaluator.backend": "asm"}),
    VariantSpec(
        "cpp_o3",
        "C++ O3",
        {"evaluator.backend": "cpp", "evaluator.cpp.optimization": "O3"},
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
            r"External SM \texorpdfstring{$d\bar d\to Z+(n-1)g$}"
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
        "scalar_0 scalar_0 > X*scalar_0",
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
        "scalar_0 scalar_0 > X*graviton",
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
    }


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
        "target_runtime_seconds": 10.0,
        "batch_size": 128,
        "warmup_runs": 2,
        "minimum_samples": 5,
        "config_overrides": dict(REPORT_CONFIG_OVERRIDES),
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
                    "reference": _empty_measurement(),
                    "pyamplicol": _empty_measurement(),
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
    )
    _require_exact_keys(
        measurement,
        (
            "status",
            "sample_count",
            "requested_config",
            "effective_config",
            "environment",
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
                "reference",
                "pyamplicol",
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
        _validate_measurement(entry["reference"], f"entries[{index}].reference")
        _validate_measurement(entry["pyamplicol"], f"entries[{index}].pyamplicol")
        relative_difference = entry["relative_difference"]
        if relative_difference is not None and (
            isinstance(relative_difference, bool)
            or not isinstance(relative_difference, (int, float))
            or not math.isfinite(float(relative_difference))
            or float(relative_difference) < 0.0
        ):
            raise ValueError(
                f"entries[{index}].relative_difference must be non-negative or null"
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


def _measurement_cell(measurement: Mapping[str, object]) -> str:
    if measurement["status"] != ResultStatus.OK.value:
        return r"\ReportNA"
    return (
        r"\ReportMetricCell"
        f"{{{_format_number(measurement['generation_seconds'])}}}"
        f"{{{_format_number(measurement['wall_seconds_per_point'])}}}"
    )


def render_matrix_table(spec: MatrixSpec, payload: Mapping[str, object]) -> str:
    validate_cache(payload)
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    entries = {
        (str(entry["process_key"]), int(entry["n_final"])): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    column_spec = "@{}rL{2.70in}" + "c" * len(spec.multiplicities) + "@{}"
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
        rf"\subsection{{{spec.title}}}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\textbf{ID} & \textbf{process family} & "
        + " & ".join(rf"\textbf{{$n={n}$}}" for n in spec.multiplicities)
        + r" \\",
        r"\midrule",
    ]
    for family in PROCESS_FAMILIES:
        cells: list[str] = []
        for n_final in spec.multiplicities:
            entry = entries[(family.key, n_final)]
            measurement = entry["pyamplicol"]
            assert isinstance(measurement, Mapping)
            cells.append(_measurement_cell(measurement))
        lines.append(
            f"{family.identifier} & {family.label_tex} & " + " & ".join(cells) + r" \\"
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


def render_performance_ladder(spec: LadderSpec, payload: Mapping[str, object]) -> str:
    validate_cache(payload)
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    entries = {
        (int(entry["n_final"]), str(entry["variant"])): entry
        for entry in raw_entries
        if isinstance(entry, Mapping)
    }
    column_spec = "@{}L{1.55in}" + "c" * len(spec.multiplicities) + "@{}"
    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
        rf"\subsection{{{spec.title}}}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\textbf{evaluator} & "
        + " & ".join(rf"\textbf{{$n={n}$}}" for n in spec.multiplicities)
        + r" \\",
        r"\midrule",
    ]
    for variant in spec.variants:
        cells: list[str] = []
        for n_final in spec.multiplicities:
            entry = entries[(n_final, variant.key)]
            measurement = entry["measurement"]
            assert isinstance(measurement, Mapping)
            cells.append(_measurement_cell(measurement))
        lines.append(f"{_tex_escape(variant.label)} & " + " & ".join(cells) + r" \\")
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
    column_spec = "@{}L{1.65in}" + "c" * multiplicity_count + "@{}"
    process_family = _tex_escape(spec.process_family)
    measurements = {
        n_final: entries[n_final]["measurement"]
        for n_final in spec.multiplicities
    }
    if not all(
        isinstance(measurement, Mapping) for measurement in measurements.values()
    ):
        raise TypeError("model-ladder measurements must be objects")

    def measurement_row(label: str, field: str) -> str:
        values = (
            _format_number(measurements[n_final][field])  # type: ignore[index]
            for n_final in spec.multiplicities
        )
        return f"{label} & " + " & ".join(values) + r" \\"

    lines = [
        "% SPDX-License-Identifier: 0BSD",
        "% Generated by docs/result_tables.py; edit the JSON cache, then render.",
        rf"\subsection{{{spec.title}}}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            rf"\multicolumn{{{multiplicity_count + 1}}}{{c}}"
            rf"{{\texttt{{{process_family}}}}}"
            + r" \\"
        ),
        r"\textbf{metric} & "
        + " & ".join(rf"\textbf{{$X={n}$}}" for n in spec.multiplicities)
        + r" \\",
        r"\midrule",
        r"status & "
        + " & ".join(r"\ReportNA" for _ in spec.multiplicities)
        + r" \\",
        measurement_row("generation [s]", "generation_seconds"),
        measurement_row("wall [s/pt]", "wall_seconds_per_point"),
        measurement_row("evaluator [s/pt]", "evaluator_seconds_per_point"),
        measurement_row("matrix element", "matrix_element"),
        "relative difference & "
        + " & ".join(
            _format_number(entries[n_final]["relative_difference"])
            for n_final in spec.multiplicities
        )
        + r" \\",
    ]
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
            tables[spec.table_name] = render_performance_ladder(spec, payload)
        else:
            tables[spec.table_name] = render_model_ladder(spec, payload)
    return tables


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
        validate_cache(payload)
        caches[name] = payload
    return caches


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
        caches = load_caches(self.paths)
        expected_schema = schema_document()
        actual_schema = json.loads(self.paths.schema_path.read_text(encoding="utf-8"))
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

    def _refresh(
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
        with _report_lock(self.paths):
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
            "performance report. No physics generation is run."
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service = ReportService()
    if args.command == "validate":
        caches = service.validate()
        print(f"validated {len(caches)} caches and {len(TABLE_INPUTS)} tables")
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
