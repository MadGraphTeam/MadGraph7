# SPDX-License-Identifier: 0BSD
"""Explicit loop declarations for the ordinary transactional artifact writer.

These records describe a saved execution lane. Scheme names are supplied by the
caller and remain declarations, never model-name-derived physics certificates.

Two engines publish virtual processes: the legacy per-diagram engine
(``rusticol.loop-runtime.v3``, kept as the oracle until gate R3) and the
shared-recursion engine v2 (``rusticol.loop-runtime.v4``, plan v4; design
§12.1), whose metadata is ``LoopV2ExecutionMetadata``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pyamplicol._internal.versions import (
    LOOP_RUNTIME_CAPABILITY,
    LOOP_V2_RUNTIME_CAPABILITY,
)
from pyamplicol.config import PoleSelection
from pyamplicol.loops.orders import VirtualOrderSector
from pyamplicol.loops.physical_orders import PhysicalOrderLayout
from pyamplicol.models.base import CouplingOrders
from pyamplicol.models.loop_declaration import (
    LoopSchemeDeclaration as LoopSchemeDeclaration,
)

from .validation import ValidationPointRecord

LOOP_RUNTIME_KIND = "pyamplicol-runtime-loop-execution"
LOOP_PHYSICS_KIND = "pyamplicol-loop-physics"
LOOP_RUNTIME_PATH = "loop-runtime.pacbin"
EPSILON_CONVENTION = "Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon)"
ORDINARY_INTERFERENCE = "2*Re(conj(Born)*loop)"
DIRECTED_INTERFERENCE = "Bdagger K_BL L + Ldagger K_LB B"
LOOP_V2_EXECUTION_SCHEMA = 4
LOOP_V2_PHYSICS_SCHEMA = 2
# How a plan v4 process obtains its per-point inputs (trees, Born amplitudes,
# counterterm scalars): rebuilt at load from the artifact's compiled model and
# generation request until WP3.1 stores native programs for them under a new
# value (rusticol-core ``loop_artifact::LOOP_V2_POINT_INPUTS``).
LOOP_V2_POINT_INPUTS = "regenerated-at-load"


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
    interference: str | None = None
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
        if type(self.resolved) is not bool or type(self.correlations) is not bool:
            raise TypeError("loop resolved/correlation capabilities must be booleans")
        interference = (
            DIRECTED_INTERFERENCE if self.correlations else ORDINARY_INTERFERENCE
        )
        if self.interference is None:
            object.__setattr__(self, "interference", interference)
        elif self.interference != interference:
            raise ValueError(
                "loop interference declaration must match correlation capability"
            )
        if (
            self.contribution_kind != "virtual"
            or self.execution_mode != "loop_recurrence"
            or self.epsilon_powers != (0, -1, -2)
            or self.epsilon_convention != EPSILON_CONVENTION
            or self.dimension != "D=4-2*epsilon"
            or self.color_accuracy not in ("lc", "nlc", "full")
            or self.scheme_status != "declared-not-certified"
        ):
            raise ValueError("unsupported loop contribution or convention declaration")
        if self.partial_results is not False:
            raise ValueError("loop artifact public partial results are unavailable")
        if self.correlations and self.resolved:
            raise ValueError("directed loop artifacts require inclusive projection")

    def validate_plan(self, plan):
        if self.correlations != (plan.get("directed_projection") is not None):
            raise ValueError(
                "loop artifact correlation capability differs from the validated plan"
            )
        projection = plan.get("color_projection")
        accuracy = "full" if projection is None else projection["accuracy"]
        if self.color_accuracy != accuracy:
            raise ValueError(
                "loop artifact colour accuracy differs from the validated plan"
            )
        from pyamplicol.loops.physical_correlations import (
            PROJECTED_FAMILY_CONVENTION,
            PROJECTED_IDENTITY_CONVENTION,
        )

        directed = plan.get("directed_projection")
        projected_directed = (
            self.correlations
            and projection is not None
            and directed["kind"] == "directed-complex"
            and bool(directed["histories"])
            and (
                (
                    directed["convention"] == PROJECTED_IDENTITY_CONVENTION
                    and directed.get("color_family_projection") is None
                    and all(
                        not item["bra"] and not item["ket"]
                        for item in directed["histories"]
                    )
                )
                or (
                    directed["convention"] == PROJECTED_FAMILY_CONVENTION
                    and directed.get("color_family_projection") is not None
                    and directed["color_family_projection"]["kind"]
                    == "complete-typed-history-families-v1"
                    and directed["color_family_projection"]["accuracy"] == accuracy
                )
            )
        )
        if self.correlations and projection is not None and not projected_directed:
            raise ValueError("projected correlation requires a certified descriptor")
        partition = plan.get("born_color_components")
        if self.color_accuracy == "lc" and not projected_directed:
            if partition is None or not self.resolved:
                raise ValueError(
                    "LC loop artifacts require resolved canonical Born components"
                )
        elif partition is not None:
            raise ValueError(
                "Born colour components require LC ordinary resolved output"
            )
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
class LoopV2ExecutionMetadata:
    """Declarations of one plan v4 process (design §12.1).

    ``born_orders`` are the squared Born orders and ``sectors`` the squared
    coupling-order sectors of the virtual with their contributing (Born, loop)
    amplitude orders: every sector is a separately returned partial result.
    ``buckets`` lists (propagators N, rank R, families) per reduction bucket;
    ``color_degrees`` the leading N_c degree P kept per sector at LC/NLC.
    ``color_correlations`` are the declared directed colour histories (ids;
    the Born metric ``"born"`` is always available) and ``spin_correlations``
    the declared spin classes (one-based legs); ``resolved`` is whether
    helicity/colour-resolved output is served: always, since observable
    columns are contracted directly (``color_contraction`` is "auto" or
    "direct"; generation refuses the forced seed FFT). ``poles`` lists the
    pole selections an evaluation may request (``PoleSelection``: every pole,
    or none for the finite part only); stability estimates come from the
    requested checks, and escalating the arithmetic is the caller's decision
    (``rescue_policy``).
    """

    scheme: LoopSchemeDeclaration
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    families: int
    buckets: tuple[tuple[int, int, int], ...]
    color_accuracy: Literal["lc", "nlc", "full"]
    color_degrees: dict[str, int | None] | None
    color_contraction: Literal["auto", "direct"]
    engine: Literal["v2"] = "v2"
    schedule: Literal["backward"] = "backward"
    r2_strategy: Literal["model"] = "model"
    arithmetic: tuple[str, ...] = ("double", "double-double", "arbitrary")
    color_correlations: tuple[str, ...] = ()
    spin_correlations: tuple[tuple[int, ...], ...] = ()
    resolved: bool = True
    poles: tuple[str, ...] = tuple(PoleSelection)
    stability_checks: tuple[str, ...] = ("rotation", "reduction-direction")
    accuracy_retest: Literal["reduction-direction"] = "reduction-direction"
    rescue_policy: Literal["caller"] = "caller"
    contribution_kind: Literal["virtual"] = "virtual"
    execution_mode: Literal["loop_recurrence"] = "loop_recurrence"
    epsilon_powers: tuple[int, ...] = (0, -1, -2)
    epsilon_convention: str = EPSILON_CONVENTION
    dimension: str = "D=4-2*epsilon"
    interference: str = ORDINARY_INTERFERENCE
    scheme_status: str = "declared-not-certified"

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, LoopSchemeDeclaration):
            raise TypeError("loop metadata requires a typed scheme declaration")
        if not self.born_orders or not self.sectors or self.families < 1:
            raise ValueError("a v2 virtual needs Born orders, sectors and families")
        if self.color_accuracy not in ("lc", "nlc", "full") or (
            (self.color_degrees is None) != (self.color_accuracy == "full")
        ):
            raise ValueError("colour degrees exist exactly at LC and NLC")
        if sum(count for _n, _r, count in self.buckets) != self.families:
            raise ValueError("reduction buckets must cover every family once")
        if (
            self.color_contraction not in ("auto", "direct")
            or self.resolved is not True
        ):
            raise ValueError(
                "v2 processes contract seed columns directly and serve resolved output"
            )
        if tuple(self.poles) != tuple(PoleSelection):
            raise ValueError(
                "v2 processes serve the pole selections "
                f"{[item.value for item in PoleSelection]}"
            )
        if len(set(self.color_correlations)) != len(self.color_correlations) or (
            "born" in self.color_correlations
        ):
            raise ValueError("declared colour histories are distinct non-Born ids")
        if any(
            not legs or list(legs) != sorted(set(legs)) or legs[0] < 1
            for legs in self.spin_correlations
        ):
            raise ValueError("spin classes are increasing one-based leg labels")


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
                "loop process colour accuracy must match its metadata; aliases "
                "are unavailable"
            )

    @property
    def runtime_capability(self) -> str:
        return LOOP_RUNTIME_CAPABILITY


@dataclass(frozen=True, slots=True)
class LoopV2ProcessArtifact(LoopProcessArtifact):
    """One plan v4 process: its PACBIN, generation request and public catalogue.

    ``request`` is the slice request record (``loops.v2.artifact``) from which
    the runtime rebuilds the Python per-point inputs; ``plan_sha256`` is the
    digest of the stored plan JSON.
    """

    request: dict[str, object] | None = None
    plan_sha256: str = ""
    state_legs: tuple[dict[str, object], ...] = ()
    helicities: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, LoopV2ExecutionMetadata):
            raise TypeError("v2 loop processes require LoopV2ExecutionMetadata")
        if self.color_accuracy != self.metadata.color_accuracy or self.aliases:
            raise ValueError(
                "loop process colour accuracy must match its metadata; aliases "
                "are unavailable"
            )
        if self.request is None or len(self.plan_sha256) != 64:
            raise ValueError("v2 loop processes record their request and plan digest")
        if len(self.state_legs) != len(self.external_pdgs) or not self.helicities:
            raise ValueError("v2 loop processes declare their legs and helicities")

    @property
    def runtime_capability(self) -> str:
        return LOOP_V2_RUNTIME_CAPABILITY


def _v2_process_payloads(
    process: LoopV2ProcessArtifact,
) -> tuple[dict[str, object], dict[str, object]]:
    # The staged plan is written from the JSON whose digest is
    # ``plan_sha256``; the native admission of the staged validation checks
    # the published member against it, so it is not re-read here.
    physics = {
        "schema_version": LOOP_V2_PHYSICS_SCHEMA,
        "kind": LOOP_PHYSICS_KIND,
        "process_id": process.process_id,
        "metadata": asdict(process.metadata),
        "external_pdgs": list(process.external_pdgs),
        "state_legs": [dict(leg) for leg in process.state_legs],
        "helicities": [list(row) for row in process.helicities],
    }
    execution = {
        "schema_version": LOOP_V2_EXECUTION_SCHEMA,
        "kind": LOOP_RUNTIME_KIND,
        "process_id": process.process_id,
        "contribution_kind": "virtual",
        "execution_mode": "loop_recurrence",
        "engine": "v2",
        "required_runtime_capabilities": [LOOP_V2_RUNTIME_CAPABILITY],
        "runtime_path": LOOP_RUNTIME_PATH,
        "plan_sha256": process.plan_sha256,
        "point_inputs": LOOP_V2_POINT_INPUTS,
        "request": process.request,
    }
    return physics, execution


def loop_process_payloads(process: LoopProcessArtifact, staged_plan: Path):
    """Validate the staged snapshot through core before publishing its metadata."""
    if isinstance(process, LoopV2ProcessArtifact):
        return _v2_process_payloads(process)

    from pyamplicol.loops.native import load_loop_native

    backend = load_loop_native()

    plan = json.loads(backend.PortableLoopPlan(str(staged_plan)).metadata_json())
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
