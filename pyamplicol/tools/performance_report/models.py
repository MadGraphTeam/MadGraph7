"""Typed contracts shared by the report catalog, runner, cache, and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Accuracy(StrEnum):
    LC = "lc"
    NLC = "nlc"
    FULL = "full"


class ExecutionMode(StrEnum):
    AMPLICOL = "amplicol"
    RECURRENCE = "recurrence"
    COMPILED = "compiled"
    EAGER = "eager"


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
    include_3qqbar: bool = False
    include_cc: bool = False
    include_resonance: bool = False

    @property
    def minimum_n(self) -> int:
        return len(self.base_final_state)

    def maximum_n(self, accuracy: Accuracy) -> int:
        return self.maximum_lc_n if accuracy is Accuracy.LC else 5

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


@dataclass(frozen=True, slots=True)
class ZVariant:
    key: str
    label: str
    execution_mode: ExecutionMode
    backend: str
    jit_optimization_level: int | None = None
    cpp_optimization: str | None = None


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
