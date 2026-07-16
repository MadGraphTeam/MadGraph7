# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..models import Model
from .dag_types import GenericDAG

LC_SECTOR_SELECTOR_PARAMETER = "runtime.lc_sector_id"
EXPRESSION_SCHEMA_CONTRACT_VERSION = 2
_REQUIRED_RUNTIME_SCHEMA_KEYS = frozenset(
    {
        "amplitude_stage",
        "current_storage",
        "momentum_slots",
        "parameter_layout",
        "stages",
        "value_storage",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeExpressionSchema:
    """Canonical, dependency-neutral stage compiler input schema."""

    canonical_json: str
    contract_version: int = EXPRESSION_SCHEMA_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != EXPRESSION_SCHEMA_CONTRACT_VERSION:
            raise ValueError(
                "unsupported runtime expression-schema contract version "
                f"{self.contract_version}"
            )
        payload = self.to_mapping()
        missing = sorted(_REQUIRED_RUNTIME_SCHEMA_KEYS - payload.keys())
        if missing:
            raise ValueError(
                "runtime expression schema is missing: " + ", ".join(missing)
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> RuntimeExpressionSchema:
        try:
            canonical = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "runtime expression schema must contain canonical JSON values"
            ) from exc
        return cls(canonical_json=canonical)

    def to_mapping(self) -> dict[str, object]:
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("runtime expression schema is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("runtime expression schema root must be an object")
        return {str(key): value for key, value in payload.items()}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StageCompilationInput:
    dag: GenericDAG
    model: Model
    runtime_schema: RuntimeExpressionSchema

    def __post_init__(self) -> None:
        if not isinstance(self.dag, GenericDAG):
            raise TypeError("stage compilation input requires a GenericDAG")
        if not isinstance(self.model, Model):
            raise TypeError("stage compilation input requires a Model")
        if not isinstance(self.runtime_schema, RuntimeExpressionSchema):
            raise TypeError(
                "stage compilation input requires a RuntimeExpressionSchema"
            )


def runtime_coupling_parameter_names(
    vertex_kind: int,
    vertex_particles: Sequence[int],
    coupling: Sequence[object],
    *,
    model: Model | None = None,
) -> list[str | None]:
    runtime_names = getattr(model, "runtime_parameter_names_for_vertex", None)
    if callable(runtime_names):
        return [str(name) for name in runtime_names(int(vertex_kind))]
    particles = "_".join(str(int(pdg)) for pdg in vertex_particles)
    base = f"coupling.{int(vertex_kind)}.{particles}"
    names: list[str | None] = []
    for component, value in enumerate(coupling):
        if not isinstance(value, str | int | float):
            raise TypeError(
                f"runtime coupling components must be numeric, got {value!r}"
            )
        numeric_value = float(value)
        if component == 1 and numeric_value == -10.0:
            names.append(None)
        else:
            names.append(f"{base}.component_{component}")
    return names


__all__ = [
    "EXPRESSION_SCHEMA_CONTRACT_VERSION",
    "LC_SECTOR_SELECTOR_PARAMETER",
    "RuntimeExpressionSchema",
    "StageCompilationInput",
    "runtime_coupling_parameter_names",
]
