# SPDX-License-Identifier: 0BSD
"""Typed contracts shared by the report catalog, runner, cache, and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

LEGACY_AMPLICOL_MAX_OPEN_QUARK_LINES = 3
_QUARK_TOKENS = frozenset({"d", "u", "s", "c", "b", "t"})
_ANTIQUARK_TOKENS = frozenset(f"{token}~" for token in _QUARK_TOKENS)


def open_quark_line_count(process: str) -> int:
    """Return the number of open quark/antiquark lines in a concrete process."""

    initial, separator, final = process.partition(">")
    if not separator:
        raise ValueError(f"process has no initial/final separator: {process!r}")
    tokens = (*initial.split(), *final.split())
    quarks = sum(token in _QUARK_TOKENS for token in tokens)
    antiquarks = sum(token in _ANTIQUARK_TOKENS for token in tokens)
    if quarks != antiquarks:
        raise ValueError(
            "concrete report process has unpaired quark content: "
            f"{process!r} ({quarks} quarks, {antiquarks} antiquarks)"
        )
    return quarks


class Accuracy(StrEnum):
    LC = "lc"
    NLC = "nlc"
    FULL = "full"


class ExecutionMode(StrEnum):
    AMPLICOL = "amplicol"
    MADGRAPH = "madgraph"
    RECURRENCE = "recurrence"
    COMPILED = "compiled"
    EAGER = "eager"
    ON_THE_FLY = "on-the-fly"


class ModelKey(StrEnum):
    BUILTIN_SM = "builtin_sm"
    UFO_SM = "ufo_sm"
    SCALAR_CONTACT = "scalar_contact"
    SCALAR_GRAVITY = "scalar_gravity"


class Workload(StrEnum):
    SELECTED_FLOW = "selected-flow"
    ALL_FLOW = "all-flow"
    CONTRACTED = "contracted"


class ArtifactPolicy(StrEnum):
    REUSE = "reuse"
    RETIME = "retime"
    REGENERATE = "regenerate"


class ResultStatus(StrEnum):
    NOT_AVAILABLE = "not_available"
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    SKIP = "skip"
    VALIDATION_FAILED = "validation_failed"
    UNVERIFIED = "unverified"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: ModelKey
    profile: str
    label: str
    source_kind: Literal["built-in-sm", "json"]


@dataclass(frozen=True, slots=True)
class ProcessFamily:
    identifier: int
    key: str
    label_tex: str
    initial_state: tuple[str, ...]
    base_final_state: tuple[str, ...]
    maximum_lc_n: int
    maximum_contracted_n: int = 5
    include_3qqbar: bool = False
    include_cc: bool = False
    include_resonance: bool = False

    @property
    def minimum_n(self) -> int:
        return len(self.base_final_state)

    def maximum_n(self, accuracy: Accuracy) -> int:
        return (
            self.maximum_lc_n
            if accuracy is Accuracy.LC
            else self.maximum_contracted_n
        )

    def process(self, n_final: int) -> str | None:
        extra_gluons = n_final - self.minimum_n
        if extra_gluons < 0:
            return None
        final_state = (*self.base_final_state, *("g" for _ in range(extra_gluons)))
        return f"{' '.join(self.initial_state)} > {' '.join(final_state)}"


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    execution_mode: ExecutionMode
    model: ModelKey | None
    accuracy: Accuracy
    backend: str
    jit_optimization_level: int | None


@dataclass(frozen=True, slots=True)
class MatrixDataset:
    dataset_id: str
    title: str
    table_name: str
    cache_name: str
    candidate: MeasurementSpec
    baseline: MeasurementSpec
    multiplicities: tuple[int, ...]
    static_na_reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class MatrixComparisonView:
    """One rendered matrix comparison backed by shared measurement datasets.

    A view deliberately identifies its candidate and reference datasets by ID
    instead of duplicating either measurement surface.  In particular, the
    MadGraph recurrence view can reuse the existing UFO-SM/full recurrence
    profile while presenting it against the independent MadGraph reference.
    """

    comparison_id: str
    candidate_dataset_id: str
    baseline_dataset_id: str
    title: str
    table_name: str


@dataclass(frozen=True, slots=True)
class ScalarDataset:
    dataset_id: str
    title: str
    table_name: str
    cache_name: str
    model: ModelKey
    process_template: str
    final_particle: str
    multiplicities: tuple[int, ...]
    measurement: MeasurementSpec

    def process(self, n_final: int) -> str:
        final_state = " ".join(self.final_particle for _ in range(n_final))
        return f"scalar_0 scalar_0 > {final_state}"


@dataclass(frozen=True, slots=True)
class ZVariant:
    key: str
    label: str
    execution_mode: ExecutionMode
    backend: str
    jit_optimization_level: int | None = None
    cpp_optimization: str | None = None
    maximum_generation_n_final: int | None = None
    static_na_reason_code: str | None = None

    def __post_init__(self) -> None:
        maximum = self.maximum_generation_n_final
        reason = self.static_na_reason_code
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("maximum_generation_n_final must be a positive integer")
        if maximum is None and reason is not None:
            raise ValueError(
                "static_na_reason_code requires maximum_generation_n_final"
            )
        if maximum is not None and (
            not isinstance(reason, str) or not reason.strip()
        ):
            raise ValueError(
                "maximum_generation_n_final requires a static_na_reason_code"
            )


@dataclass(frozen=True, slots=True)
class CellSpec:
    dataset_id: str
    process: str
    n_final: int
    process_key: str | None
    measurement: MeasurementSpec
    workload: Workload
    variant: str | None = None

    @property
    def cell_id(self) -> str:
        parts = [
            self.dataset_id,
            f"n{self.n_final}",
            self.process_key,
            self.variant,
            self.workload.value,
        ]
        return "-".join(
            part.replace("_", "-") for part in parts if part is not None
        )
