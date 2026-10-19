# SPDX-License-Identifier: 0BSD
"""Explicit loop declarations for the ordinary transactional artifact writer.

These records describe a saved execution lane. Scheme names are supplied by the
caller and remain declarations, never model-name-derived physics certificates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pyamplicol._internal.versions import LOOP_RUNTIME_CAPABILITY
from pyamplicol.loops.physical_orders import PhysicalOrderLayout
from pyamplicol.models.loop_declaration import (
    LoopSchemeDeclaration as LoopSchemeDeclaration,
)

from .validation import ValidationPointRecord

LOOP_RUNTIME_KIND = "pyamplicol-runtime-loop-execution"
LOOP_PHYSICS_KIND = "pyamplicol-loop-physics"
LOOP_RUNTIME_PATH = "loop-runtime.pacbin"
EPSILON_CONVENTION = "Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon)"


@dataclass(frozen=True, slots=True)
class LoopExecutionMetadata:
    scheme: LoopSchemeDeclaration
    orders: PhysicalOrderLayout
    r2_strategy: Literal["model", "direct"]
    available_r2_strategies: tuple[str, ...]
    arithmetic: tuple[str, ...]
    source_owned: bool
    contribution_kind: Literal["virtual"] = "virtual"
    execution_mode: Literal["loop_recurrence"] = "loop_recurrence"
    epsilon_powers: tuple[int, ...] = (0, -1, -2)
    epsilon_convention: str = EPSILON_CONVENTION
    dimension: str = "D=4-2*epsilon"
    interference: str = "2*Re(conj(Born)*loop)"
    color_accuracy: str = "full"
    scheme_status: str = "declared-not-certified"
    correlations: bool = False
    resolved: bool = False
    partial_results: bool = False

    def __post_init__(self):
        if not isinstance(self.scheme, LoopSchemeDeclaration) or not isinstance(
            self.orders, PhysicalOrderLayout
        ):
            raise TypeError("loop metadata requires typed scheme and order records")
        if self.r2_strategy != "model" or (
            self.available_r2_strategies != (self.r2_strategy,)
        ):
            raise ValueError(
                "public loop artifact currently requires model-supplied R2"
            )
        if self.arithmetic != ("double", "double-double", "arbitrary"):
            raise ValueError("loop artifact must declare its three supported carriers")
        if self.source_owned is not True:
            raise TypeError("public loop artifacts require source-owned scalar data")
        if (
            self.contribution_kind != "virtual"
            or self.execution_mode != "loop_recurrence"
            or self.epsilon_powers != (0, -1, -2)
            or self.epsilon_convention != EPSILON_CONVENTION
            or self.dimension != "D=4-2*epsilon"
            or self.interference != "2*Re(conj(Born)*loop)"
            or self.color_accuracy != "full"
            or self.scheme_status != "declared-not-certified"
        ):
            raise ValueError("unsupported loop contribution or convention declaration")
        if type(self.resolved) is not bool:
            raise TypeError("loop resolved capability must be a boolean")
        if any(
            value is not False for value in (self.correlations, self.partial_results)
        ):
            raise ValueError(
                "loop artifact correlations/partial results are unavailable"
            )

    def validate_plan(self, plan):
        if type(self.resolved) is not bool or self.resolved != (
            plan.get("helicity_projection") is not None
        ):
            raise ValueError(
                "loop artifact resolved capability differs from the validated plan"
            )
        # JSON normalization only removes tuple/list representation differences.
        orders = json.loads(json.dumps(asdict(self.orders)))
        if orders != plan["order_layout"]:
            raise ValueError("loop artifact order axes differ from the validated plan")
        if self.r2_strategy != plan["r2_strategy"]:
            raise ValueError(
                "loop artifact R2 strategy differs from the validated plan"
            )
        if self.source_owned != (plan["model_binding"]["kind"] == "source"):
            raise ValueError(
                "loop artifact scalar ownership differs from the validated plan"
            )


@dataclass(frozen=True, slots=True)
class LoopProcessArtifact:
    """One already compiled loop PACBIN and explicit publication declarations."""

    process_id: str
    expression: str
    external_pdgs: tuple[int, ...]
    loop_runtime_path: Path
    metadata: LoopExecutionMetadata
    validation_point: ValidationPointRecord
    generation_filters: dict[str, object]
    color_accuracy: str = "full"
    aliases: tuple = ()

    def __post_init__(self):
        if not isinstance(self.metadata, LoopExecutionMetadata):
            raise TypeError("loop process requires explicit typed execution metadata")
        if self.color_accuracy != self.metadata.color_accuracy or self.aliases:
            raise ValueError(
                "loop artifacts currently require full colour and no aliases"
            )


def loop_process_payloads(process: LoopProcessArtifact, staged_plan: Path):
    """Validate the staged snapshot through core before publishing its metadata."""
    import _loop_reduction

    plan = json.loads(
        _loop_reduction.PortableLoopPlan(str(staged_plan)).metadata_json()
    )
    process.metadata.validate_plan(plan)
    if list(process.external_pdgs) != plan["external_pdgs"]:
        raise ValueError(
            "loop artifact external particles differ from the validated plan"
        )
    physics = {
        "schema_version": 1,
        "kind": LOOP_PHYSICS_KIND,
        "process_id": process.process_id,
        "metadata": asdict(process.metadata),
        "initial_average": plan["initial_average"],
        "state_legs": plan["state_legs"],
        "helicities": plan["helicities"],
    }
    execution = {
        "schema_version": 3,
        "kind": LOOP_RUNTIME_KIND,
        "process_id": process.process_id,
        "contribution_kind": "virtual",
        "execution_mode": "loop_recurrence",
        "required_runtime_capabilities": [LOOP_RUNTIME_CAPABILITY],
        "runtime_path": LOOP_RUNTIME_PATH,
    }
    return physics, execution
