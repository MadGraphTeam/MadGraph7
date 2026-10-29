# SPDX-License-Identifier: 0BSD
"""Explicit loop declarations for the ordinary transactional artifact writer.

These records describe a saved execution lane. Scheme names are supplied by the
caller and remain declarations, never model-name-derived physics certificates.

Virtual processes are published by the shared-recursion engine v2
(``rusticol.loop-runtime.v4``, plan v4; design §12.1), whose metadata is
``LoopV2ExecutionMetadata``. The removed legacy engine's format
(``rusticol.loop-runtime.v3``) is refused at load (``runtime.backend``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pyamplicol._internal.versions import LOOP_V2_RUNTIME_CAPABILITY
from pyamplicol.config import PoleSelection
from pyamplicol.loops.orders import VirtualOrderSector
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
# counterterm and field components): from native programs stored as further
# PACBIN members (``loops.v2.native_process``, WP3.1; rusticol-core
# ``loop_artifact::LOOP_V2_NATIVE_POINT_INPUTS``). "native-v2": the native
# programs with the binary64 SymJIT plane application of every saved
# evaluator (licence-free binary64, DESIGN §3.7). A process without them is
# refused at generation: nothing of a stored process regenerates at load
# (REQUIREMENTS_ADDENDUM G).
LOOP_V2_NATIVE_POINT_INPUTS = "native-v2"
# Former declarations, refused with a request to regenerate (internal artifact
# formats are not kept backward compatible, AGENTS.md): "native-v1" stored no
# plane applications, "regenerated-at-load" rebuilt the inputs at load from
# the artifact's compiled model and generation request.
LOOP_V2_RETIRED_POINT_INPUTS = ("native-v1", "regenerated-at-load")


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
    columns are contracted directly. ``color_contraction`` is the requested
    seed contraction ("auto", "direct" or "fft", design §7.3): the certified
    symmetric-group orbit pairs of the stored seed Fourier record are
    contracted by the FFT under "auto" and "fft". ``poles`` lists the
    pole selections an evaluation may request (``PoleSelection``: every pole,
    none for the finite part only, or their ultraviolet or infrared parts;
    design §12.4); stability estimates come from the requested checks, and
    escalating the arithmetic is the caller's decision (``rescue_policy``).
    """

    scheme: LoopSchemeDeclaration
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    families: int
    buckets: tuple[tuple[int, int, int], ...]
    color_accuracy: Literal["lc", "nlc", "full"]
    color_degrees: dict[str, int | None] | None
    color_contraction: Literal["auto", "direct", "fft"]
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
        if self.color_contraction not in ("auto", "direct", "fft"):
            raise ValueError("unknown v2 seed contraction policy")
        if self.resolved is not True:
            raise ValueError("v2 processes serve resolved output")
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
class LoopV2ProcessArtifact:
    """One plan v4 process: its PACBIN, generation request and public catalogue.

    ``request`` is the slice request record (``loops.v2.artifact``, the
    generation record; a load never regenerates from it); ``plan_sha256`` is the
    digest of the stored plan JSON and ``scaleless_plan_sha256`` that of the
    stored companion plan of the certified-scaleless families (None without
    them; ``loops.v2.artifact.SCALELESS_PLAN_MEMBER``).
    """

    process_id: str
    expression: str
    external_pdgs: tuple[int, ...]
    loop_runtime_path: Path
    metadata: LoopV2ExecutionMetadata
    validation_point: ValidationPointRecord
    generation_filters: dict[str, object]
    color_accuracy: str = "full"
    aliases: tuple = ()
    request: dict[str, object] | None = None
    plan_sha256: str = ""
    scaleless_plan_sha256: str | None = None
    state_legs: tuple[dict[str, object], ...] = ()
    helicities: tuple[tuple[int, ...], ...] = ()
    point_inputs: str = LOOP_V2_NATIVE_POINT_INPUTS

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
        if (
            self.scaleless_plan_sha256 is not None
            and len(self.scaleless_plan_sha256) != 64
        ):
            raise ValueError("a companion plan digest is a SHA-256")
        if len(self.state_legs) != len(self.external_pdgs) or not self.helicities:
            raise ValueError("v2 loop processes declare their legs and helicities")
        if self.point_inputs in LOOP_V2_RETIRED_POINT_INPUTS:
            raise ValueError(
                f"point inputs {self.point_inputs!r} are no longer supported "
                f"(this build stores {LOOP_V2_NATIVE_POINT_INPUTS!r}); regenerate "
                "the artifact"
            )
        if self.point_inputs != LOOP_V2_NATIVE_POINT_INPUTS:
            raise ValueError(f"unknown point inputs {self.point_inputs!r}")

    @property
    def runtime_capability(self) -> str:
        return LOOP_V2_RUNTIME_CAPABILITY


def recorded_conventions(
    extensions: Mapping[str, object], process_id: str
) -> dict[str, object] | None:
    """The conventions of process ``process_id``'s electroweak virtual as the
    artifact's generation extension records them (the process's generation
    filters, ``conventions``: WP3.2 EWI-7, ``loops.v2.ew_policy.
    scheme_conventions``), None when none is recorded. The physics metadata
    keeps its schema (the native admission reads it)."""

    generation = extensions.get("generation")
    if not isinstance(generation, Mapping):
        return None
    records = generation.get("concrete_processes")
    if not isinstance(records, (list, tuple)):
        return None
    for record in records:
        if isinstance(record, Mapping) and record.get("id") == process_id:
            filters = record.get("filters")
            found = filters.get("conventions") if isinstance(filters, Mapping) else None
            return dict(found) if isinstance(found, Mapping) else None
    return None


def _v2_process_payloads(
    process: LoopV2ProcessArtifact,
) -> tuple[dict[str, object], dict[str, object]]:
    # The staged plan is written from the JSON whose digest is
    # ``plan_sha256``, so it is not re-read here. The native admission
    # checks the published member against that digest at every
    # ``Runtime.load``, and also at write time when post-build
    # validation is enabled (its staged validation loads the artifact).
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
        "scaleless_plan_sha256": process.scaleless_plan_sha256,
        "point_inputs": process.point_inputs,
        "request": process.request,
    }
    return physics, execution


def loop_process_payloads(process: LoopV2ProcessArtifact, staged_plan: Path):
    """The physics and execution records of one published plan v4 process."""
    if not isinstance(process, LoopV2ProcessArtifact):
        raise TypeError("loop processes are plan v4 (LoopV2ProcessArtifact)")
    return _v2_process_payloads(process)
