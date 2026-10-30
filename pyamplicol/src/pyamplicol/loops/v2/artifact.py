# SPDX-License-Identifier: 0BSD
"""Plan v4 loop artifacts and their public runtime (design §12, WP2.6).

Format. A v2 virtual process is published in the ordinary schema-v3 artifact
envelope under the runtime capability ``rusticol.loop-runtime.v4``:

* ``processes/<id>/loop-runtime.pacbin`` holds a member of kind
  ``LOOP_RUNTIME_PLAN``, ``schedule/loop-plan-v4.json.zlib``: the plan v4
  program of ``plan_export`` (zlib-compressed JSON, schedule B; the forward
  oracle F is not stored) that the native ``V2Executor`` validates and runs.
  Its kernels are exact polynomial term lists interpreted natively
  (``loop_v2/dress.rs``): no compiled code, so nothing depends on the build
  host's CPU features or paths. A process with certified-scaleless families
  also stores their companion plan (``LoopSlice.scaleless``),
  ``schedule/loop-plan-v4-scaleless.json.zlib``, which only the ``"uv"``
  and ``"ir"`` pole selections run (design §12.4): one artifact serves all
  four selections.
* ``processes/<id>/execution.json`` (schema 4) records the plan's SHA-256
  (and the companion's, or null), ``point_inputs`` and the generation
  request (process, coupling orders,
  corrections, fixed parameters, colour accuracy and continuation, seed
  contraction, the resolved correlator declarations);
  ``processes/<id>/physics.json`` (schema 2) the public catalogue
  (``generation.loop_artifact.LoopV2ExecutionMetadata``: Born orders,
  sectors, families and reduction buckets, colour degrees, declared colour
  histories and spin classes, carriers, legs and helicity rows).
* ``model/compiled-model.json`` is the compiled loop model; flavour members
  served by a representative are listed in ``process-groups.json``
  (basis ``exact-loop-flavor``).

The artifact is relocatable: every reference is relative to its root, and no
absolute path, user or host name is stored (``tests/unit/test_loop_v2_artifact
.py`` scans it).

Point inputs. Every point needs the block rows of the trees, the Born
amplitudes and the counterterm and external-field components. With
``point_inputs = "native-v2"`` (WP3.1) the PACBIN also holds the process's
native program (``native_process.NativeProcess``): its assembly record and
tree program (``point-inputs/*.json.zlib``: the interaction program, the Born
and counterterm closures, the interferences), its saved exact evaluators
(``point-inputs/*.state``: the parameter program and the interaction
kernels, run by DoubleFloat and arbitrary precision) and their binary64
SymJIT plane applications (``point-inputs/*.symjit``: storage-v4 IR,
recompiled for the loading CPU; binary64 then touches no Symbolica state,
DESIGN §3.7). A load then regenerates nothing: the native executor binds
every parameter-dependent value from the external parameters and forms all
of it per point. Correlated, spin-correlated and resolved calls run the
stored observable columns (``native_observables.StoredObservables``) and a
flavour member served by a representative its stored identity-port
substitution, so no call of a loaded process regenerates anything
(REQUIREMENTS_ADDENDUM G). A process whose point inputs or observable
columns cannot be exported natively is refused at generation, never stored
for a regeneration at load; the former ``point_inputs =
"regenerated-at-load"`` declaration is refused at load like ``native-v1``.
Generation does not depend on the interpreter's Symbolica symbol history:
tree stages use canonical symbols (``assemble._canonical_stage``) and kernel
terms a canonical order (``reference_executor.ExactKernel``).

Evaluation returns the public result types: Born per squared Born order, the
complete virtual per squared coupling-order sector with its three Laurent
coefficients (finite only on request), and, when stability checks were asked
for, one ``VirtualStabilityEstimate`` per point. Correlated (colour and spin)
and resolved outputs and flavour members run through the observables package
(``runtime.virtual.V2ObservablesRuntime`` over the stored columns,
``native_observables.StoredObservableSlice``). Binary64,
double-double and arbitrary precision run natively. Escalation of the
arithmetic is the caller's decision (requirements addendum C).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import ModelParameters, Momenta, ScalarInput
from pyamplicol.api.results import (
    BornAnchoredColorComponent,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
)
from pyamplicol.api.virtual_results import (
    POLE_COMPONENTS,
    POLE_ORIGINS,
    LaurentCoefficients,
    VirtualColorContraction,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualPoleParts,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
    VirtualStabilityPolicy,
    computes_poles,
    splits_poles,
)
from pyamplicol.config import PoleSelection, R2Strategy
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin,
)
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection

from .families import Orders, add_orders

PLAN_MEMBER = "schedule/loop-plan-v4.json.zlib"
# The native point inputs (``native_process.NativeProcess``): its assembly
# record and tree program (zlib JSON) and saved exact evaluators.
NATIVE_RECORD_MEMBER = "point-inputs/native-process.json.zlib"
NATIVE_INPUTS_PREFIX = "point-inputs/"
# Under ``NATIVE_INPUTS_PREFIX`` (and ``NATIVE_SCALELESS_PREFIX``, the
# certified-scaleless companion's tree program, ``native_process.
# NativeCompanion``): the tree program, the parameter program's saved exact
# evaluator and the kernels', and their binary64 SymJIT plane applications
# (storage-v4 IR).
NATIVE_TREES_MEMBER = "tree-program.json.zlib"
NATIVE_PARAMETERS_MEMBER = "parameters.state"
NATIVE_PARAMETERS_PLANE_MEMBER = "parameters.symjit"
NATIVE_KERNEL_MEMBER = "kernel-{}.state"
NATIVE_KERNEL_PLANE_MEMBER = "kernel-{}.symjit"
NATIVE_SCALELESS_PREFIX = "point-inputs/scaleless/"
# The stored observable columns (``native_observables.StoredObservables``):
# zlib JSON members ``point-inputs/observables*.json.zlib``.
NATIVE_OBSERVABLES_PREFIX = "point-inputs/"
# The companion plan of the certified-scaleless families (rusticol-core
# ``loop_runtime::LOOP_PROGRAM_V4_SCALELESS_MEMBER``).
SCALELESS_PLAN_MEMBER = "schedule/loop-plan-v4-scaleless.json.zlib"
ENGINE = "v2"


# --------------------------------------------------------------------- request


def _orders_record(orders: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(orders.items()))


def _encode_field(name: str, value: Any) -> Any:
    """One ``SliceRequest`` field as JSON: mappings sorted, tuples as lists,
    the colour declaration as its dictionary; anything else must already be
    a JSON scalar (a new field of another type fails here, not silently)."""

    if name == "colour_declaration":
        return None if value is None else value.to_dict()
    if name == OPTIONAL_REQUEST_FIELD:
        return value.record()
    if name == "loop_orders" and not isinstance(value, Mapping):
        return [_orders_record(orders) for orders in value]
    if isinstance(value, Mapping):
        return _orders_record(value)
    if isinstance(value, tuple):
        return list(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    raise TypeError(f"SliceRequest field {name!r} has no JSON record ({value!r})")


def _decode_field(name: str, value: Any) -> Any:
    """The inverse of ``_encode_field``."""

    if name == "colour_declaration":
        from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

        return None if value is None else LoopColorDeclaration.from_dict(value)
    if name == OPTIONAL_REQUEST_FIELD:
        from .selection import Selection

        return Selection.from_record(value)
    if name == "loop_orders" and isinstance(value, list):
        return tuple(dict(orders) for orders in value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return tuple(value)
    return value


# The request field recorded only when set: a leading-colour selection
# (``selection.Selection``), so requests without one keep their records'
# bytes.
OPTIONAL_REQUEST_FIELD = "selection"
# The request fields recorded only when they differ from their default
# (``request_record``), with that default: the selection and the external
# photons' tags (MS-PHOTON), so records of requests that keep them keep their
# bytes, and records written before a field existed keep their meaning.
OPTIONAL_REQUEST_DEFAULTS: dict[str, object] = {
    OPTIONAL_REQUEST_FIELD: None,
    "external_photons": "final",
    "alpha0_conversion": "eCoup_DGmua0_UV_EW",
    "alpha0_inverse": None,
}


def request_record(request: Any, correlators: Any = None) -> dict[str, object]:
    """The JSON record of a ``SliceRequest`` (every dataclass field, so a new
    request field is recorded and restored without an edit here; the
    ``OPTIONAL_REQUEST_DEFAULTS`` fields only when not at their default) and
    its resolved correlator declarations: the artifact's generation recipe."""

    from dataclasses import fields

    record: dict[str, object] = {
        item.name: _encode_field(item.name, getattr(request, item.name))
        for item in fields(request)
        if item.name not in OPTIONAL_REQUEST_DEFAULTS
        or getattr(request, item.name) != OPTIONAL_REQUEST_DEFAULTS[item.name]
    }
    record["correlators"] = None if correlators is None else correlators.to_json_dict()
    return record


def request_from_record(record: Mapping[str, Any]) -> Any:
    """The ``SliceRequest`` of a request record, which must name exactly the
    request's fields (a missing field would silently take its default) except
    the ``OPTIONAL_REQUEST_DEFAULTS`` ones."""

    from dataclasses import fields

    from .assemble import SliceRequest

    known = {item.name for item in fields(SliceRequest)}
    for name in OPTIONAL_REQUEST_DEFAULTS:
        if name not in record:
            known.discard(name)
    if set(record) != known | {"correlators"}:
        raise ArtifactError(
            "the loop request record does not match this pyAmpliCol build "
            f"(unknown {sorted(set(record) - known - {'correlators'})}, missing "
            f"{sorted(known - set(record))}); regenerate the artifact"
        )
    return SliceRequest(
        **{name: _decode_field(name, record[name]) for name in sorted(known)}
    )


def correlators_from_record(record: Mapping[str, Any]) -> Any:
    """The resolved ``CorrelatorConfig`` of a request record, or ``None``."""

    from pyamplicol.correlators import CorrelatorConfig

    value = record.get("correlators")
    return None if value is None else CorrelatorConfig.from_json_dict(value)


# --------------------------------------------------------------------- preflight


def born_amplitude_orders(
    compiled: Any,
    process: str,
    bounds: Mapping[str, int],
    particle_veto: Sequence[int] = (),
) -> tuple[tuple[Orders, ...], Any]:
    """The Born amplitude orders of ``process`` within the coupling-order
    ``bounds`` and the Born DAG they are read from, before any loop
    generation: one tree DAG, compiled as ``blocks.build_tree_blocks``
    compiles the Born of a slice (without the vetoed |PDG| codes
    ``particle_veto``; a loop Born DAG, ``loop_born``: a Dirac current
    closing on a Weyl sink is a root, ``dag_ordering.DIRAC_WEYL_CLOSURE``)."""

    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    from .blocks import _born_roots, vetoed_particle_ids

    model = CompiledUFOModel(compiled)
    dag = compile_generic_dag(
        build_model_process_ir(process, compiled.ir, color_accuracy="full"),
        model=model,
        max_coupling_orders=dict(bounds),
        ignored_particle_ids=vetoed_particle_ids(particle_veto),
        loop_born=True,
    )
    orders = {root.coupling_orders for root in _born_roots(dag, model)}
    return tuple(sorted(orders)), dag


def coloured_widths(
    dags: Sequence[Any], compiled: Any, fixed_parameters: Mapping[str, str]
) -> dict[str, str]:
    """``{particle: width parameter}`` of the coloured particles that a tree
    current of ``dags`` propagates with a width that is not zero: neither
    ``ZERO``, a compile-time zero nor fixed to zero by ``fixed_parameters``.
    Source currents (external legs), auxiliary currents and currents of all
    legs but one do not count: the latter carry an external momentum and are
    amputated (a Born root contracts them with the remaining leg; a loop
    attached to one would be an external self-energy). The counterterm
    recursion (WP2.2) chooses full tree kernels for coloured widths by a
    criterion of its own that also counts model-supplied propagators; the
    two are to be one criterion (DESIGN WP2.6, "widths")."""

    from pyamplicol.models.compiler_kernels import _is_compile_time_zero_parameter

    from .blocks import propagated_species

    particles = {int(record.pdg_code): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}
    out: dict[str, str] = {}
    for dag in dags:
        # The current scan (native for a ``dag_input.NativeDag``: no decode).
        for particle_id, amputated in propagated_species(dag):
            if amputated:
                continue
            record = particles.get(particle_id)
            if record is None or abs(int(record.color)) == 1:
                continue
            width = str(record.width)
            if _is_compile_time_zero_parameter(width, parameters):
                continue
            fixed = fixed_parameters.get(width)
            if fixed is not None and Decimal(str(fixed)) == 0:
                continue
            out[str(record.name)] = width
    return out


def require_zero_coloured_widths(
    process: str,
    dags: Sequence[Any],
    compiled: Any,
    fixed_parameters: Mapping[str, str],
) -> None:
    """Refuse a request whose tree currents propagate a coloured particle
    with a width (``coloured_widths``).

    v2 virtuals are real on-shell: loop propagators carry no width and the
    model counterterms renormalise the real mass, so a width in the tree
    currents (Born and the blocks feeding the loop) would mix two schemes;
    for g g > t t~ at loop_sm's default top width the single pole then no
    longer follows the colour-correlated Borns (O(Gamma), not O(Gamma^2)).
    Fixing the width to zero is the legacy engine's convention (its
    propagators carry no width); a resonant coloured particle needs the
    complex-mass scheme (``loops.model_cms``: its model propagates the
    particle with its complex pole mass and width zero, which passes here)."""

    from .refusals import LoopRequestUnsupported

    widths = coloured_widths(dags, compiled, fixed_parameters)
    if not widths:
        return
    named = ", ".join(f"{name} ({width})" for name, width in sorted(widths.items()))
    fixed = ", ".join(f'{width} = "0"' for width in sorted(set(widths.values())))
    raise LoopRequestUnsupported(
        "coloured-width",
        f"{process}: tree currents propagate coloured particles whose width "
        f"is not fixed to zero: {named}; fix it in [loop.preparation] "
        f"fixed_parameters ({fixed}), take a restriction whose widths are "
        'zero (model.restriction = "no_widths"), or take the complex-mass '
        'scheme ([loop.model.scheme] mass_scheme = "complex-mass")',
    )


# --------------------------------------------------------------------- plan file


def plan_digest(plan_json: str) -> str:
    return hashlib.sha256(plan_json.encode()).hexdigest()


def write_plan(
    plan_json: str,
    destination: Path,
    native: Any = None,
    scaleless_json: str | None = None,
) -> None:
    """Write the plan v4 PACBIN (its digest is ``GeneratedProcess.plan_sha256``,
    computed once from the same JSON), with the companion plan of the
    certified-scaleless families when the process has one and, with
    ``native`` (a ``native_process.NativeProcess`` of that plan), its native
    point inputs."""

    from io import BytesIO

    plan = PacbinMemberKind.LOOP_RUNTIME_PLAN
    state = PacbinMemberKind.SYMBOLICA_EXACT_STATE
    symjit = PacbinMemberKind.SYMJIT_APPLICATION
    # (logical path, kind, bytes); JSON members are zlib streams.
    members = [(PLAN_MEMBER, plan, zlib.compress(plan_json.encode(), 6))]
    if scaleless_json is not None:
        members.append(
            (SCALELESS_PLAN_MEMBER, plan, zlib.compress(scaleless_json.encode(), 6))
        )
    if native is not None:
        if native.plan_json != plan_json:
            raise ValueError("the native point inputs belong to another plan")
        companion = native.scaleless
        if (companion is None) != (scaleless_json is None) or (
            companion is not None and companion.plan_json != scaleless_json
        ):
            raise ValueError(
                "the native point inputs store the companion of the stored "
                "companion plan"
            )
        stored = native.members()
        members.append(
            (
                NATIVE_RECORD_MEMBER,
                plan,
                zlib.compress(stored["native-process.json"], 6),
            )
        )
        prefixes = [("", NATIVE_INPUTS_PREFIX)]
        if companion is not None:
            from .native_process import SCALELESS

            prefixes.append((SCALELESS, NATIVE_SCALELESS_PREFIX))
        for prefix, path in prefixes:
            members += [
                (
                    f"{path}{NATIVE_TREES_MEMBER}",
                    plan,
                    zlib.compress(stored[f"{prefix}tree-program.json"], 6),
                ),
                (
                    f"{path}{NATIVE_PARAMETERS_MEMBER}",
                    state,
                    stored[f"{prefix}parameters.bin"],
                ),
                (
                    f"{path}{NATIVE_PARAMETERS_PLANE_MEMBER}",
                    symjit,
                    stored[f"{prefix}parameters.symjit"],
                ),
            ]
            k = 0
            while f"{prefix}kernel-{k}.bin" in stored:
                members.append(
                    (
                        path + NATIVE_KERNEL_MEMBER.format(k),
                        state,
                        stored[f"{prefix}kernel-{k}.bin"],
                    )
                )
                members.append(
                    (
                        path + NATIVE_KERNEL_PLANE_MEMBER.format(k),
                        symjit,
                        stored[f"{prefix}kernel-{k}.symjit"],
                    )
                )
                k += 1
        for name in sorted(stored):
            if name.startswith("observables"):
                members.append(
                    (
                        f"{NATIVE_OBSERVABLES_PREFIX}{name}.zlib",
                        plan,
                        zlib.compress(stored[name], 6),
                    )
                )
    write_pacbin(
        destination,
        tuple(
            PacbinMemberSource(name, kind, BytesIO(payload))
            for name, kind, payload in members
        ),
    )


def read_plan(path: Path, member: str = PLAN_MEMBER) -> str | None:
    """A plan v4 JSON of a ``loop-runtime.pacbin`` (payload digests checked):
    the process plan, or with ``member=SCALELESS_PLAN_MEMBER`` the companion
    plan (None when absent)."""

    with PacbinReader.open(path) as reader:
        if member != PLAN_MEMBER and member not in {
            item.logical_path for item in reader.members
        }:
            return None
        found = reader.member(member)
        if found.kind != PacbinMemberKind.LOOP_RUNTIME_PLAN:
            raise ArtifactError("loop runtime member is not a plan v4")
        compressed = reader.read_member(member, length=found.length)
    return zlib.decompress(compressed).decode()


def read_native_process(
    path: Path, plan_json: str, scaleless_json: str | None = None
) -> Any:
    """The native point inputs of a ``loop-runtime.pacbin`` written with them
    (``write_plan(..., native)``), for its plan ``plan_json`` and, when the
    process has certified-scaleless families, its companion plan
    ``scaleless_json`` (whose native tree program it also holds; read from
    the PACBIN when not given)."""

    from .native_process import SCALELESS, NativeProcess

    if scaleless_json is None:
        scaleless_json = read_plan(path, SCALELESS_PLAN_MEMBER)

    def read(reader: Any, name: str, kind: PacbinMemberKind) -> bytes:
        member = reader.member(name)
        if member.kind != kind:
            raise ArtifactError(f"loop runtime member {name!r} has the wrong kind")
        return bytes(reader.read_member(name, length=member.length))

    members: dict[str, bytes] = {}
    with PacbinReader.open(path) as reader:
        names = {member.logical_path for member in reader.index.members}
        json_kind = PacbinMemberKind.LOOP_RUNTIME_PLAN
        state = PacbinMemberKind.SYMBOLICA_EXACT_STATE
        symjit = PacbinMemberKind.SYMJIT_APPLICATION
        members["native-process.json"] = zlib.decompress(
            read(reader, NATIVE_RECORD_MEMBER, json_kind)
        )
        prefixes = [("", NATIVE_INPUTS_PREFIX)]
        if scaleless_json is not None:
            prefixes.append((SCALELESS, NATIVE_SCALELESS_PREFIX))
        for prefix, base in prefixes:
            members[f"{prefix}tree-program.json"] = zlib.decompress(
                read(reader, base + NATIVE_TREES_MEMBER, json_kind)
            )
            members[f"{prefix}parameters.bin"] = read(
                reader, base + NATIVE_PARAMETERS_MEMBER, state
            )
            members[f"{prefix}parameters.symjit"] = read(
                reader, base + NATIVE_PARAMETERS_PLANE_MEMBER, symjit
            )
            k = 0
            while base + NATIVE_KERNEL_MEMBER.format(k) in names:
                members[f"{prefix}kernel-{k}.bin"] = read(
                    reader, base + NATIVE_KERNEL_MEMBER.format(k), state
                )
                members[f"{prefix}kernel-{k}.symjit"] = read(
                    reader, base + NATIVE_KERNEL_PLANE_MEMBER.format(k), symjit
                )
                k += 1
        start = f"{NATIVE_OBSERVABLES_PREFIX}observables"
        for name in sorted(names):
            if name.startswith(start) and name.endswith(".json.zlib"):
                members[name[len(NATIVE_OBSERVABLES_PREFIX) : -len(".zlib")]] = (
                    zlib.decompress(read(reader, name, json_kind))
                )
    try:
        return NativeProcess.from_members(plan_json, members, scaleless_json)
    except ValueError as error:
        raise ArtifactError(f"unreadable native point inputs: {error}") from error


# --------------------------------------------------------------------- generation


@dataclass(frozen=True)
class GeneratedProcess:
    """One process generated through v2: its observables (and slice), plan
    and public catalogue, and the companion plan of its certified-scaleless
    families (``LoopSlice.scaleless``; None without them). Frozen: the
    digests are cached from the plan JSON."""

    observables: Any
    exported: Any
    request: Mapping[str, Any]
    plan_json: str
    # The native point inputs (``native_process.NativeProcess``; None only for
    # generation objects that are never stored).
    native: Any = None
    scaleless_exported: Any = None
    scaleless_plan_json: str | None = None

    @property
    def slice(self) -> Any:
        return self.observables.slice

    @property
    def point_inputs(self) -> str:
        from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS

        if self.native is None:
            raise ArtifactError(
                "the generated process has no native point inputs; it cannot be stored"
            )
        return LOOP_V2_NATIVE_POINT_INPUTS

    @cached_property
    def plan_sha256(self) -> str:
        return plan_digest(self.plan_json)

    @cached_property
    def scaleless_plan_sha256(self) -> str | None:
        if self.scaleless_plan_json is None:
            return None
        return plan_digest(self.scaleless_plan_json)

    def executor(self, inputs: str = "python") -> Any:
        """A native executor of the slice on the stored plans; ``inputs=
        "native"`` forms the point inputs from the plan's tree program (the
        stored native process's lane, ``native_process``)."""

        from .plan_export import NativeExecutor

        scaleless = None
        if self.scaleless_exported is not None:
            scaleless = (self.scaleless_exported, cast(str, self.scaleless_plan_json))
        return NativeExecutor(
            self.slice,
            exported=self.exported,
            plan_json=self.plan_json,
            scaleless=scaleless,
            inputs=inputs,
        )

    def born_orders(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        amplitudes = tuple(self.slice.blocks.born_orders)
        return tuple(
            sorted(
                {
                    canonical_order(add_orders(a, b))
                    for a in amplitudes
                    for b in amplitudes
                }
            )
        )

    def sectors(self) -> tuple[VirtualOrderSector, ...]:
        return tuple(
            VirtualOrderSector(
                canonical_order(sector),
                tuple(BornLoopPair(born, loop) for born, loop in sorted(pairs)),
            )
            for sector, pairs in self.slice.sectors
        )

    def buckets(self) -> list[list[int]]:
        """(N, R, families) per reduction bucket of the plan."""

        counts: dict[tuple[int, int], int] = {}
        for family in self.exported.program["families"]:
            key = (len(family["sequence"]), family["rank"])
            counts[key] = counts.get(key, 0) + 1
        return [[n, r, count] for (n, r), count in sorted(counts.items())]

    def colour_degrees(self) -> dict[str, int | None] | None:
        if self.slice.request.accuracy == "full":
            return None
        degrees = self.slice.degrees()["virtual"]
        return {
            _order_text(sector): degrees[sector]
            for sector, _pairs in self.slice.sectors
        }

    def state_legs(self) -> list[dict[str, object]]:
        return [
            {"pdg": int(leg.pdg), "incoming": bool(leg.is_initial)}
            for leg in sorted(self.slice.blocks.process.legs, key=lambda x: x.label)
        ]

    def helicities(self) -> list[list[int]]:
        return [[h for _leg, h in row] for row in self.exported.rows]

    def color_correlations(self) -> tuple[str, ...]:
        return tuple(history.id for history in self.observables.histories)

    def spin_correlations(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(legs) for legs in self.observables.spin_classes)

    def inert_counterterms(self) -> dict[str, str]:
        """The CT rules the request admits that contribute nothing, by name,
        with the reason (``CountertermRecursion.inert``, REQUIREMENTS_ADDENDUM
        J): the generation record's ``counterterms_inert``. Rule names and
        reasons come from the model, so the record holds no path."""

        program = self.slice.counterterm_program
        if program is None:
            return {}
        return dict(sorted(program.recursion.inert.items()))

    def anomalous_veto_counterterms(self) -> dict[str, list[str]]:
        """The CT rules of the request with a Levi-Civita cell its
        ``particle_veto`` keeps from cancelling that cancels without it
        (``ct_recursion``, "Anomalous vetoes"; [U2]), by name, with those
        cells: lowered under ``allow_anomalous_veto``, the generation
        record's ``counterterms_anomalous_veto``."""

        from .ct_recursion import rule_key

        program = self.slice.counterterm_program
        if program is None:
            return {}
        recursion = program.recursion
        anomalous = recursion.model.anomalous_veto
        return {
            name: sorted(str(cell) for cell in anomalous[rule_key(rule)])
            for name, rule in sorted(recursion.rules.items())
            if rule_key(rule) in anomalous
        }

    def conventions(self) -> dict[str, object] | None:
        """The conventions of an electroweak request's virtual (WP3.2 EWI-7,
        ``ew_policy.scheme_conventions``: regularisation and lambda_HV, the
        gamma-five reading, gauge, UV, mass and electroweak input schemes,
        the external-photon convention): the generation record's
        ``conventions`` (``Runtime.inspect`` and the artifact inspection read
        it there); None for a request that is not electroweak (its record
        then has no entry, so such artifacts keep their bytes)."""

        from .ew_policy import scheme_conventions

        return scheme_conventions(self.slice)

    def native_fallbacks(self) -> dict[str, str]:
        """The generation stages that took the tree engine instead of the
        native loop-request DAG builder, each with the listed feature it
        does not cover (``dag_input.fallback_record``, RUST-MIGRATION rule
        2): the generation record's ``native_fallbacks``; empty when every
        stage was native (the record then has no entry)."""

        from .dag_input import fallback_record

        return fallback_record(self.slice)

    def binary64_exact_engine(self) -> dict[str, Any]:
        """The routing of light masses in binary64 (genericity audit 2,
        G2-M3; DESIGN §11.2 'Absolute thresholds'): the masses of the process
        (loop lines of both plans, external legs) whose binary64 square at the
        generation parameters (the model's defaults and the request's fixed
        parameters) lies strictly between 0 and the fast engine's absolute
        threshold in the model's units (``V2_FAST_ENGINE_THRESHOLD`` of the
        loop module), by mass parameter, with that square and the threshold.
        Binary64 lanes with such a mass take the exact generic engine
        (``v2_reduce::reaches_fast_threshold``); the runtime decides per lane
        again, on the point's invariants and any parameter override. The
        generation record's ``binary64_exact_engine``; empty when no mass is
        that light (the record then has no entry, so such artifacts keep
        their bytes)."""

        from .assemble import Precision, _leg_masses, _mass_values

        slice_ = self.slice
        threshold = float(slice_.native.V2_FAST_ENGINE_THRESHOLD)
        parameters = slice_.request.fixed_parameters
        digits = Precision.double().working_digits
        lines = {
            mass
            for plan in (self.exported, self.scaleless_exported)
            if plan is not None
            for mass in plan.masses
            if not mass.massless
        }
        squares = {
            mass.mass: _threshold_size(value)
            for mass, value in _mass_values(
                slice_.compiled, parameters, lines, digits
            ).items()
        }
        for leg, value in zip(
            slice_.families.legs, _leg_masses(slice_, parameters, digits), strict=True
        ):
            squares[leg.mass.mass] = float(value)
        light = {
            name: repr(value)
            for name, value in sorted(squares.items())
            if 0 < abs(value) < threshold
        }
        if not light:
            return {}
        return {"threshold": repr(threshold), "squared_masses": light}


def _threshold_size(square: Any) -> float:
    """A squared mass's size as the binary64 fast engine's threshold reads
    it (``v2_reduce::squared_mass_reaches_fast_threshold``): |re| + |im| in
    binary64, the value of a real (non-negative) square; a complex-mass
    scheme's pole mass squared is complex."""

    if isinstance(square, int | Fraction):
        return float(square)
    from .reference_executor import exact

    value = exact(square)
    return abs(float(value.real)) + abs(float(value.imag))


def _order_text(order: Orders) -> str:
    return " ".join(f"{name}^{power}" for name, power in canonical_order(order))


def generate_process(
    compiled: Any,
    request: Any,
    native: Any,
    correlators: Any = None,
) -> GeneratedProcess:
    """Generate one process through v2 with its declared observables. The
    returned ``GeneratedProcess`` holds the generation objects (the slice,
    the exported plans); nothing keeps them after the caller drops it, and
    no load reads them (a stored process evaluates from its native program).

    Without colour histories or spin classes this is the inclusive slice
    (``LoopSlice.generate``, colour-pruned at LC/NLC for the inclusive table);
    with them, ``ObservableSlice.generate`` (full tree kernels for spin
    sources; LC/NLC pruning keeps every skeleton live in the declared
    histories' complete-family tables). The stored plan is the inclusive plan
    of that slice; with native point inputs the correlated and resolved
    column sets are exported here too and stored with the native process
    (``native_observables.StoredObservables``), so no call of a loaded
    process regenerates.
    """

    from pyamplicol.correlators import CorrelatorConfig

    from .assemble import LoopSlice, resolve_request
    from .observables import ObservableSlice
    from .plan_export import export_plan

    if request.counterterms != "recursion":
        # Site closure (with its loop matching) is the O6 test oracle only.
        raise ArtifactError(
            "plan v4 processes take their counterterms from the tree recursion; "
            f"counterterms={request.counterterms!r} is a test oracle"
        )
    # The recorded recipe names the corrections the slice takes (a request
    # that names none takes the model's strong correction).
    request = resolve_request(compiled, request)
    declared = correlators is not None and bool(
        correlators.color_correlations
        or correlators.spin_correlations
        or correlators.all_color_through_order is not None
    )
    from .native_trees import NativeExportUnsupported

    try:
        # A listed feature the native point inputs do not cover is refused as
        # soon as the generation knows it (``native_inputs``), before the
        # trees are built.
        if declared:
            observables = ObservableSlice.generate(
                compiled, request, native, correlators=correlators, native_inputs=True
            )
        else:
            observables = ObservableSlice.of(
                LoopSlice.generate(compiled, request, native, native_inputs=True)
            )
    except NativeExportUnsupported as error:
        raise _not_storable(request, "native point inputs", error) from error
    # Schedule F (the forward oracle) is not stored: the runtime dresses by
    # schedule B, and F's section is over half of the plan (gg -> gggg: 52 of
    # 98 MB of JSON).
    blocks = observables.slice.blocks
    require_zero_coloured_widths(
        repr(request.process),
        (blocks.blocks_dag, blocks.born_dag),
        observables.slice.compiled,
        request.fixed_parameters,
    )
    # The plan carries the native point inputs (``native_process``): the tree
    # program with the Born, counterterm and field components, so a load
    # needs no regeneration. A process whose tree program cannot be exported
    # natively is refused: a stored process never regenerates at load
    # (REQUIREMENTS_ADDENDUM G).
    native = None
    # The observable column sets (``native_observables``): their seed records
    # are formed on the plan's own export (``column_sets``) and stored with
    # the native process, so no correlated, resolved or catalogue call of a
    # loaded process regenerates.
    from .native_observables import column_plans

    plans = column_plans(observables)
    try:
        # With declared observables every counterterm ket component stays in
        # the tree program (``native_observables``: a colour history can read
        # a component the inclusive metric does not).
        exported = export_plan(
            observables.slice,
            forward=False,
            trees=True,
            born_squared=True,
            keep_kets=declared,
            column_sets={kind: plan.seed_columns for kind, plan in plans.items()},
        )
    except NativeExportUnsupported as error:
        raise _not_storable(request, "native point inputs", error) from error
    plan_json = exported.json()
    # The companion plan of the certified-scaleless families, which the "uv"
    # and "ir" pole selections run (design §12.4). With native point inputs
    # it is exported with its own tree program (no Born or counterterm
    # outputs: it adds only lanes), stored with the process, so no pole
    # selection of a loaded process regenerates; with the seed records of the
    # column plans' twins on it.
    companion = observables.slice.scaleless
    scaleless = None
    if companion is not None:
        twins = {}
        if exported.trees is not None:
            for kind, plan in plans.items():
                twin = plan.scaleless()
                if twin is not None and twin.slice is companion:
                    twins[kind] = twin.seed_columns
        try:
            scaleless = export_plan(
                companion,
                forward=False,
                trees=exported.trees is not None,
                born_squared=False,
                counterterms=False,
                column_sets=twins or None,
            )
        except NativeExportUnsupported as error:
            raise _not_storable(request, "companion point inputs", error) from error
    scaleless_json = None if scaleless is None else scaleless.json()
    if exported.trees is not None:
        from .native_process import NativeProcess

        native = NativeProcess.export(
            observables.slice,
            exported,
            plan_json,
            None if scaleless is None else (scaleless, cast(str, scaleless_json)),
        )
        # A column set the native trees cannot form (no native counterterm
        # components) is refused like the point inputs: a correlated or
        # resolved call of a stored process never regenerates.
        from dataclasses import replace

        from .native_observables import StoredObservables

        try:
            stored_columns = StoredObservables.export(
                observables, exported, native, plans, scaleless
            )
        except NativeExportUnsupported as error:
            raise _not_storable(request, "observable columns", error) from error
        native = replace(native, observables=stored_columns)
    resolved = CorrelatorConfig(
        tuple(observables.histories), tuple(observables.spin_classes)
    )
    record = request_record(request, resolved if declared else None)
    generated = GeneratedProcess(
        observables,
        exported,
        record,
        plan_json,
        native=native,
        scaleless_exported=scaleless,
        scaleless_plan_json=scaleless_json,
    )
    return generated


def _not_storable(request: Any, what: str, error: Exception) -> Exception:
    """The generation refusal of a process whose ``what`` cannot be exported
    natively (a listed feature, ``native_trees.NATIVE_EXPORT_REFUSALS``, its
    key first): a plan v4 process is stored with them, so no call of a
    loaded process regenerates it. There is no plan-only downgrade; any other
    exception of the export propagates (fail closed)."""

    from .refusals import NativeExportRefused

    # Typed with the export refusal's key (m-sm MS-MODEL), not as text
    # inside a plain GenerationError.
    feature = getattr(error, "feature", None) or "native-export"
    return NativeExportRefused(
        feature,
        f"{feature}: {request.process}: its {what} cannot be exported natively "
        f"({error}); a stored plan v4 process evaluates from them and never "
        "regenerates at load (REQUIREMENTS_ADDENDUM G)",
    )


# --------------------------------------------------------------------- runtime


def admit(native: Any, root: Path, process_id: str) -> dict[str, Any]:
    """Admit one plan v4 process through the native artifact loader.

    ``rusticol_core::loop_artifact::load_loop_v2_artifact`` checks the
    envelope, the execution and physics declarations, the plan member's
    digest, and validates the program, whose families, reduction buckets,
    legs, helicity rows and sectors must equal the declarations. Returns its
    description (identity, plan digest and size, buckets, program shape,
    metadata and request).
    """

    try:
        return cast(
            dict[str, Any], json.loads(native.admit_v2_artifact(str(root), process_id))
        )
    except ValueError as error:
        raise ArtifactError(f"plan v4 admission failed: {error}") from error


def _select(manifest: Any, process: str | None) -> tuple[Mapping[str, Any], Any]:
    """The selected process record and, for a flavour member served by a
    representative (``process-groups.json``, basis ``exact-loop-flavor``),
    its group member (else ``None``)."""

    records = cast(tuple[Mapping[str, Any], ...], manifest.processes)
    if process is None:
        if manifest.default_process_id is not None:
            process = manifest.default_process_id
        elif len(records) == 1:
            return records[0], None
        else:
            raise EvaluationError(
                "this artifact holds several processes; select one with process="
            )
    for record in records:
        if record["id"] == process:
            return record, None
    from pyamplicol.artifacts.process_groups import read_process_groups

    groups = read_process_groups(manifest.root)
    # The public spelling rule of requests (processes.model.
    # resolve_model_particle_name): the exact spelling first, then a unique
    # letter-case-insensitive match, as the tree artifacts' selector compares
    # lower-cased tokens (rusticol-core normalize_process_expression). The
    # artifact records the model's spelling (loop_sm's gluon is ``G``), so
    # ``g g > t t~`` selects ``G G > t t~``.
    for fold in (False, True):
        wanted = _selector_text(process, fold=fold)
        matches = [
            record
            for record in records
            if _selector_text(str(record["expression"]), fold=fold) == wanted
        ]
        if fold and len(matches) > 1:
            choices = ", ".join(sorted(str(record["id"]) for record in matches))
            raise EvaluationError(
                f"process expression {process!r} is ambiguous; select one of "
                f"these stable ids: {choices}"
            )
        if matches:
            return matches[0], None
        for representative, group in (groups.groups if groups else {}).items():
            for member in group.members:
                if member.process_id != representative and (
                    member.process_id == process
                    or _selector_text(member.expression, fold=fold) == wanted
                ):
                    if str(member.basis) != "exact-loop-flavor":
                        break
                    for record in records:
                        if record["id"] == representative:
                            return record, member
    raise EvaluationError(f"artifact has no process {process!r}")


def _selector_text(expression: str, *, fold: bool) -> str:
    """A process expression as ``_select`` compares it: whitespace-normalised
    and, with ``fold``, lower-cased per token."""

    tokens = expression.split()
    return " ".join(token.lower() for token in tokens) if fold else " ".join(tokens)


def _json_payload(manifest: Any, path: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads((manifest.root / path).read_text()))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(
            f"unreadable loop artifact payload {path}: {error}"
        ) from error


def _real_parameters(mapping: ModelParameters | None) -> dict[str, str]:
    from pyamplicol.runtime.virtual import _parameters

    parameters = _parameters(mapping)
    if parameters is None:
        return {}
    out = {}
    for name, (real, imaginary) in parameters.items():
        if Decimal(imaginary) != 0:
            raise CompatibilityError(
                f"parameter {name!r}: v2 loop runtimes take real external parameters"
            )
        out[name] = real
    return out


def _precision_bits(carrier: str, digits: int | None) -> int:
    if carrier == "double":
        return 53
    if carrier == "double-double":
        return 106
    return math.ceil(cast(int, digits) * math.log2(10))


class V2VirtualRuntime:
    """Runtime of one v2 virtual process (the public ``Runtime`` backend).

    ``process`` selects a process record or a flavour member that the
    artifact serves through its representative (``process-groups.json``):
    the member's momenta, colour histories, spin legs and helicity rows are
    read on the representative (generation proved the substitution,
    ``observables.admit_member``; ``_stored_member``).
    """

    contribution_kind = "virtual"
    execution_mode = "loop_recurrence"

    def __init__(
        self,
        path: Path,
        *,
        process: str | None = None,
        model_parameters: ModelParameters | None = None,
        mute_warnings: bool = False,
        arithmetic: str | None = None,
        precision: int | None = None,
        r2_strategy: str | None = None,
        stability: VirtualStabilityPolicy | None = None,
    ) -> None:
        from pyamplicol.artifacts import MANIFEST_NAME, load_manifest
        from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS
        from pyamplicol.loops.native import load_loop_native
        from pyamplicol.models.loading import CompiledModel
        from pyamplicol.runtime.virtual import _NO_RESCUE, v2_r2_strategy

        if stability is not None:
            raise CompatibilityError(_NO_RESCUE)
        self._default_r2 = v2_r2_strategy(r2_strategy)
        self.default_precision = 16 if precision is None else _digits(precision)
        self.default_arithmetic = _arithmetic(arithmetic) or (
            "double" if self.default_precision == 16 else "arbitrary"
        )
        root = Path(path)
        if root.name == MANIFEST_NAME and root.is_file():
            root = root.parent
        started = time.perf_counter()
        # Payload digests are not recomputed on the runtime path (as for every
        # artifact, ``runtime.backend``): the native admission below checks
        # the execution and physics declarations and the plan member against
        # its declared digest.
        manifest = load_manifest(root)
        selected, group_member = _select(manifest, process)
        process_id = str(selected["id"])
        native = load_loop_native()
        admission = admit(native, root, process_id)
        execution = _json_payload(manifest, f"processes/{process_id}/execution.json")
        physics = _json_payload(manifest, str(selected["physics_path"]))
        model_payload = _json_payload(manifest, "model/compiled-model.json")
        self._lock = RLock()
        self._native = native
        self._model_payload = model_payload
        self._plan_sha256 = str(admission["plan_sha256"])
        self._scaleless_plan_sha256 = admission["scaleless_plan_sha256"]
        self._group_member = group_member
        self._member = None
        self._observables: Any = None
        self._execution = execution
        compiled = CompiledModel.from_dict(model_payload, validate_fingerprint=False)
        self._externals = {
            record.name
            for record in compiled.ir.parameters
            if record.nature == "external"
        }
        if execution["point_inputs"] != LOOP_V2_NATIVE_POINT_INPUTS:
            # The native admission refuses every other declaration first.
            raise ArtifactError(
                f"plan v4 point inputs {execution['point_inputs']!r} are not "
                "supported; regenerate the artifact"
            )
        # The stored native program evaluates every point in every pole
        # selection (with the stored companion of the certified-scaleless
        # families): no slice is regenerated (WP3.1 I6). A flavour member is
        # an identity-port member of its representative, proved at generation
        # (``process-groups.json``); correlated and resolved calls run the
        # stored observable columns (``native_observables``). Nothing of a
        # loaded process regenerates (REQUIREMENTS_ADDENDUM G).
        runtime = root / "processes" / process_id / str(execution["runtime_path"])
        self._native_process = read_native_process(
            runtime,
            read_plan(runtime),
            read_plan(runtime, SCALELESS_PLAN_MEMBER),
        )
        if self._native_process.observables is None:
            raise ArtifactError(
                "the stored process has no observable columns (an older "
                "build's artifact); regenerate the artifact"
            )
        if group_member is not None:
            self._member = _stored_member(group_member, str(selected["expression"]))
        record = self._native_process.record
        self._fixed = dict(record["fixed_parameters"])
        self._prepared_fixed = frozenset(record.get("preparation_fixed_parameters", ()))
        self._scale_parameter = record["scale_parameter"]
        self._executor = self._native_process.executor(
            native, externals=self._externals
        )
        self._parameters = self._checked_parameters(model_parameters)
        self._load_seconds = time.perf_counter() - started
        self._admission = admission
        self._manifest = manifest
        # WP3.2 EWI-7: the conventions of an electroweak virtual, from the
        # generation record (None when none is recorded).
        from pyamplicol.generation.loop_artifact import recorded_conventions

        self._conventions = recorded_conventions(
            manifest.extensions, str(selected["id"])
        )
        self._process_id = process_id
        self._expression = str(selected["expression"])
        self._physics = physics
        self._execution = execution
        self._metadata = physics["metadata"]
        self._born_orders = tuple(
            canonical_order(tuple((n, p) for n, p in order))
            for order in self._metadata["born_orders"]
        )
        self._sectors = tuple(
            VirtualOrderSector(
                canonical_order(tuple((n, p) for n, p in item["order"])),
                tuple(
                    BornLoopPair(
                        tuple((n, p) for n, p in pair["born"]),
                        tuple((n, p) for n, p in pair["loop"]),
                    )
                    for pair in item["pairs"]
                ),
            )
            for item in self._metadata["sectors"]
        )

    def _executor_for(self, poles: PoleSelection) -> Any:
        """The executor of a pole selection: one executor serves all of them
        (a stored native process holds the companion of its
        certified-scaleless families, which "uv" and "ir" also dress)."""

        del poles
        return self._executor

    def _observable_runtime(self) -> Any:
        """The correlated and resolved outputs' runtime (``runtime.virtual.
        V2ObservablesRuntime``), built on first use on the stored observable
        columns of the native process (nothing regenerates)."""

        from pyamplicol.runtime.virtual import V2ObservablesRuntime

        from .native_observables import StoredObservableSlice
        from .observables import FORCED_FFT_OBSERVABLES

        if self._execution["request"]["colour_contraction"] == "fft":
            # The stored observable columns are contracted directly.
            raise CompatibilityError(FORCED_FFT_OBSERVABLES)
        with self._lock:
            if self._observables is None:
                correlators = correlators_from_record(self._execution["request"])
                observables = StoredObservableSlice(
                    self._native_process,
                    self._native_process.observables,
                    () if correlators is None else correlators.color_correlations,
                    self._native,
                )
                runtime = V2ObservablesRuntime(
                    observables,
                    member=self._member,
                    arithmetic=self.default_arithmetic,
                    precision=self.default_precision,
                )
                runtime.set_model_parameters(self._parameters)
                self._observables = runtime
            return self._observables

    # -- identity and metadata ------------------------------------------------

    @property
    def artifact_id(self) -> str:
        return self._manifest.artifact_id

    @property
    def representative_process_key(self) -> str:
        return self._process_id

    @property
    def _public_id(self) -> str:
        member = self._group_member
        return self._process_id if member is None else str(member.process_id)

    @property
    def external_count(self) -> int:
        return len(self._physics["state_legs"])

    @property
    def external_permutation(self) -> tuple[int, ...]:
        return tuple(range(self.external_count))

    @property
    def physics(self) -> ProcessPhysics:
        accuracy = cast(Literal["lc", "nlc", "full"], self._metadata["color_accuracy"])
        member = self._group_member
        anchors = (
            tuple(
                BornAnchoredColorComponent(
                    flow.id,
                    index,
                    (flow.basis,),
                    f"leading-colour flow {flow.word} (Born basis element "
                    f"{flow.basis})",
                )
                for index, flow in enumerate(self._flows())
                if flow.basis is not None
            )
            if accuracy == "lc" and member is None
            else ()
        )
        kind = "born-anchored-interference" if anchors else "contracted-color"
        # A generation-time selection (the request's ``selection``) keeps
        # some helicity rows or flows: coverage 'selected', as the tree
        # engine reports it.
        chosen = self._execution["request"].get(OPTIONAL_REQUEST_FIELD) or {}
        helicity_coverage = "selected" if chosen.get("helicities") else "all"
        color_coverage = "selected" if chosen.get("flows") is not None else "all"
        pdgs = (
            [int(leg["pdg"]) for leg in self._physics["state_legs"]]
            if member is None
            else list(member.pdgs)
        )
        return ProcessPhysics(
            process_id=self._public_id,
            process=self._expression if member is None else str(member.expression),
            color_accuracy=accuracy,
            helicity_coverage=helicity_coverage,
            color_coverage=color_coverage,
            color_kind=kind,
            structural_zero_helicity_count=0,
            external_particles=tuple(
                ExternalParticle(
                    index=i,
                    label=i + 1,
                    name=str(pdg),
                    pdg_id=pdg,
                    state="incoming" if leg["incoming"] else "outgoing",
                    momentum_slot=i,
                )
                for i, (leg, pdg) in enumerate(
                    zip(self._physics["state_legs"], pdgs, strict=True)
                )
            ),
            helicities=tuple(
                HelicityConfiguration(f"h{i}", i, tuple(h), True, False, f"h{i}", 1.0)
                for i, h in enumerate(self._physics["helicities"])
            ),
            color_flows=(),
            contracted_color_components=()
            if anchors
            else (
                ContractedColorComponent(
                    accuracy, 0, f"{accuracy} colour interference"
                ),
            ),
            # Leading colour: the flows ``evaluate(color_flows=...)`` selects
            # (``selection``: tree-engine IDs and order), each anchored to
            # its Born basis element (``evaluate_resolved``'s ``basis-<i>``).
            born_anchored_color_components=anchors,
            contribution_kind="virtual",
            reduction=PhysicsReduction(kind, ()),
            model_parameters=(),
            selector_capabilities=("helicity", "color_flow") if anchors else (),
        )

    def _flows(self) -> tuple[Any, ...]:
        """The stored leading-colour flows (none below leading colour)."""

        from .selection import Flow

        record = self._native_process.observables.record
        return tuple(Flow.from_record(flow) for flow in record.get("color_flows", ()))

    def inspect(self) -> Mapping[str, object]:
        with self._lock:
            return deepcopy(
                {
                    "artifact_id": self.artifact_id,
                    "process_id": self._public_id,
                    "representative_process_id": self._process_id,
                    "process": self._expression,
                    "engine": ENGINE,
                    **self._metadata,
                    "plan_sha256": self._execution["plan_sha256"],
                    "scaleless_plan_sha256": self._admission["scaleless_plan_sha256"],
                    "plan_bytes": self._admission["plan_bytes"],
                    "point_inputs": self._execution["point_inputs"],
                    # The certified symmetric-group FFT of the stored seeds
                    # (design §7.3): its orbit pairs, 0 without a record.
                    "seed_fourier_pairs": sum(
                        len(job["pairs"])
                        for job in (
                            self._native_process.record.get("fourier") or {}
                        ).get("jobs", ())
                    ),
                    "native_admission": {
                        key: self._admission[key]
                        for key in ("runtime_capability", "legs", "rows", "columns")
                    },
                    "request": self._execution["request"],
                    "load_seconds": self._load_seconds,
                    "default_arithmetic": self.default_arithmetic,
                    "default_precision": self.default_precision,
                    "requested_r2_strategy": self._default_r2.value,
                    "effective_r2_strategy": "model",
                    "parameter_overrides": dict(self._parameters),
                    "scale_input": "explicit positive mu_r2, scalar or per point",
                    "stability_checks": (
                        "rotations (count or exact rotations) and the "
                        "reduction-direction check, optionally rescaled; none "
                        "by default"
                    ),
                    "rescue_available": False,
                    "accuracy_established": False,
                    **(
                        {"conventions": self._conventions}
                        if self._conventions is not None
                        else {}
                    ),
                }
            )

    # -- evaluation ---------------------------------------------------------------

    def _carrier(
        self, arithmetic: str | None, precision: int | None
    ) -> tuple[str, Any]:
        """(public arithmetic, v2 ``Precision``) of one call: an explicit
        arithmetic, else the load-time default at the default precision, else
        binary64 at 16 digits and arbitrary precision otherwise."""

        from .assemble import Precision

        digits = self.default_precision if precision is None else _digits(precision)
        carrier = _arithmetic(arithmetic) or (
            self.default_arithmetic if digits == self.default_precision else None
        )
        carrier = carrier or ("double" if digits == 16 else "arbitrary")
        if carrier == "double":
            return carrier, Precision.double()
        if carrier == "double-double":
            return carrier, Precision.double_double()
        return carrier, Precision.arbitrary(digits)

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
        on_failure: Literal["raise", "return"] = "raise",
    ) -> VirtualEvaluation:
        from pyamplicol.runtime.virtual import _inputs, v2_options

        if helicities or color_flows:
            # A leading-colour selection (``selection``): one summed output
            # of the stored resolved columns.
            return self._observable_runtime().evaluate(
                momenta,
                helicities=helicities,
                color_flows=color_flows,
                mu_r2=mu_r2,
                r2_strategy=r2_strategy,
                stability=stability,
                poles=poles,
                stability_checks=stability_checks,
                on_failure=on_failure,
                **self._observable_options(precision, arithmetic, {}),
            )
        selection, checks = v2_options(
            r2_strategy=r2_strategy,
            stability=stability,
            poles=poles,
            stability_checks=stability_checks,
            on_failure=on_failure,
        )
        # v2_options refused direct R2.
        r2 = self._default_r2 if r2_strategy is None else R2Strategy(r2_strategy)
        carrier, setting = self._carrier(arithmetic, precision)
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        if self._member is not None:
            points = [self._member.point(point) for point in points]
        with self._lock:
            parameters = dict(self._parameters)
            results, failures = self._run(
                points, scales, parameters, setting, selection, checks
            )
        if failures and on_failure == "raise":
            index = min(failures)
            raise EvaluationError(
                f"v2 virtual evaluation failed at point {index}: "
                + "; ".join(failures[index])
            )
        return self._result(results, failures, setting, carrier, r2, selection, checks)

    def _run(
        self,
        points: list[list[list[str]]],
        scales: list[str],
        parameters: dict[str, str],
        setting: Any,
        poles: PoleSelection,
        checks: VirtualStabilityChecks,
    ) -> tuple[list[Any], dict[int, list[str]]]:
        """One batch through the native executor in the pole mode ``poles``
        (``loops.v2.poles``): the per-point results (``None`` for a failed
        point) and the failures (a failed point fails alone,
        ``plan_export.FailedPoints``)."""

        from .plan_export import FailedPoints

        if not points:
            return [], {}
        if len({Fraction(scale) for scale in scales}) > 1 and self._scale_partitioned(
            poles
        ):
            return self._run_per_scale(
                points, scales, parameters, setting, poles, checks
            )
        try:
            return (
                list(
                    self._executor_for(poles).evaluate(
                        points,
                        scales,
                        parameters=parameters,
                        precision=setting,
                        poles=poles.value,
                        stability=checks.to_checks() if checks.enabled else None,
                    )
                ),
                {},
            )
        except FailedPoints as failed:
            return list(failed.results), dict(failed.failures)

    def _scale_partitioned(self, poles: PoleSelection) -> bool:
        """Whether a batch of mixed mu_R^2 runs per scale: the stored tree
        program (or, for "uv" and "ir", its certified-scaleless companion's)
        has Born couplings, kernel constants or masses that depend on the
        renormalisation scale (``native_trees.scale_dependent_shared``, a
        running-coupling or MSbar-mass model), as the SDK's
        ``V2Runtime::evaluate`` partitions."""

        from .native_trees import scale_dependent_shared

        process = self._native_process
        trees = [process.trees]
        if splits_poles(poles) and process.scaleless is not None:
            trees.append(process.scaleless.trees)
        return any(scale_dependent_shared(item) for item in trees)

    def _run_per_scale(
        self,
        points: list[list[list[str]]],
        scales: list[str],
        parameters: dict[str, str],
        setting: Any,
        poles: PoleSelection,
        checks: VirtualStabilityChecks,
    ) -> tuple[list[Any], dict[int, list[str]]]:
        """``_run`` per exact mu_R^2 (first-seen order), each part a batch of
        one scale; results and failures in the caller's point order."""

        parts: dict[Fraction, list[int]] = {}
        for index, scale in enumerate(scales):
            parts.setdefault(Fraction(scale), []).append(index)
        results: list[Any] = [None] * len(points)
        failures: dict[int, list[str]] = {}
        for indices in parts.values():
            found, failed = self._run(
                [points[i] for i in indices],
                [scales[i] for i in indices],
                parameters,
                setting,
                poles,
                checks,
            )
            for local, index in enumerate(indices):
                results[index] = found[local]
            failures.update({indices[local]: v for local, v in failed.items()})
        return results, failures

    def _result(
        self,
        results: list[Any],
        failures: Mapping[int, Sequence[str]],
        setting: Any,
        carrier: str,
        r2: R2Strategy,
        poles: PoleSelection,
        checks: VirtualStabilityChecks,
    ) -> VirtualEvaluation:
        from pyamplicol.runtime.virtual import v2_public_estimate

        with_poles = computes_poles(poles)
        bits = _precision_bits(carrier, setting.digits)
        arithmetic = cast(Literal["double", "double-double", "arbitrary"], carrier)
        contraction = self._contraction(results)
        born_values: list[tuple[Any, ...] | None] = []
        values: list[tuple[LaurentCoefficients, ...] | None] = []
        diagnostics: list[VirtualLaneDiagnostic] = []
        estimates: list[VirtualStabilityEstimate | None] = []
        parts: list[VirtualPoleParts | None] = []
        for index, result in enumerate(results):
            if result is None:
                reason = "; ".join(failures.get(index, ["failed"]))
                born_values.append(None)
                values.append(None)
                estimates.append(None)
                parts.append(None)
                diagnostics.append(
                    VirtualLaneDiagnostic(
                        "failed",
                        arithmetic,
                        bits,
                        False,
                        reason,
                        failures=(
                            VirtualFailureDiagnostic(
                                index, None, None, None, "v2-evaluation", reason
                            ),
                        ),
                        primary_color_contraction=contraction,
                    )
                )
                continue
            born = {
                canonical_order(order): value for order, value in result.born.items()
            }
            virtual = {
                canonical_order(sector): laurent
                for sector, laurent in result.components["virtual"].items()
            }
            born_values.append(
                tuple(born.get(order, _zero(carrier)) for order in self._born_orders)
            )
            values.append(
                tuple(
                    LaurentCoefficients(*virtual[sector.order].coefficients)
                    if with_poles
                    else LaurentCoefficients(
                        virtual[sector.order].finite, *_ABSENT_POLES
                    )
                    for sector in self._sectors
                )
            )
            estimates.append(v2_public_estimate(result.stability, with_poles))
            parts.append(self._pole_parts(result) if splits_poles(poles) else None)
            diagnostics.append(
                VirtualLaneDiagnostic(
                    "success",
                    arithmetic,
                    bits,
                    False,
                    "accuracy is estimated only by the requested stability checks",
                    primary_color_contraction=contraction,
                )
            )
        return VirtualEvaluation(
            tuple(born_values),
            tuple(values),
            self._born_orders,
            self._sectors,
            tuple(diagnostics),
            R2Selection(r2, R2Strategy.MODEL, "v2 artifacts carry model R2 rules"),
            color_accuracy=cast(
                Literal["lc", "nlc", "full"], self._metadata["color_accuracy"]
            ),
            poles=poles,
            stability=tuple(estimates) if checks.enabled else None,
            pole_parts=tuple(parts) if splits_poles(poles) else None,
        )

    def _contraction(self, results: Sequence[Any]) -> VirtualColorContraction:
        """The seed contraction a batch ran (design §7.3): the certified
        symmetric-group orbit pairs of the stored seed Fourier record by the
        FFT (``PointResult.colour_contraction``), or every weight directly."""

        from .colour import SYMMETRIC_GROUP_FFT

        requested = self._execution["request"]["colour_contraction"]
        requested = {"fft": SYMMETRIC_GROUP_FFT}.get(requested, requested)
        algorithm = next(
            (result.colour_contraction for result in results if result is not None),
            "direct",
        )
        fourier = self._native_process.record.get("fourier")
        if algorithm == SYMMETRIC_GROUP_FFT and fourier is not None:
            return VirtualColorContraction(
                requested,
                SYMMETRIC_GROUP_FFT,
                "the certified symmetric-group orbit pairs of the native seeds "
                "are contracted by the FFT in the carrier, the other colour "
                "weights K[c, b] directly",
                regular_kernels=sum(len(job["pairs"]) for job in fourier["jobs"]),
            )
        reason = (
            "native seeds contract the exact colour weights K[c, b] directly"
            if fourier is None
            else "the requested contraction is direct"
        )
        return VirtualColorContraction(requested, "direct", reason)

    def _pole_parts(self, result: Any) -> VirtualPoleParts:
        """Both origins' parts of a split result by component
        (``PointResult.pole_origins``), per public sector."""

        origins = result.pole_origins

        def sectors(origin: str, name: str) -> tuple[LaurentCoefficients, ...]:
            laurents = {
                canonical_order(sector): laurent
                for sector, laurent in origins[origin][name].items()
            }
            return tuple(
                LaurentCoefficients(*laurents[sector.order].coefficients)
                for sector in self._sectors
            )

        return VirtualPoleParts(
            *(
                {name: sectors(origin, name) for name in POLE_COMPONENTS}
                for origin in POLE_ORIGINS
            )
        )

    # -- observables (``runtime.virtual.V2ObservablesRuntime``) -----------------

    def _observable_options(
        self, precision: int | None, arithmetic: str | None, options: dict[str, Any]
    ) -> dict[str, Any]:
        """The arithmetic of one observable call (``_carrier``'s rule) and the
        options ``V2ObservablesRuntime`` validates itself."""

        carrier, setting = self._carrier(arithmetic, precision)
        digits = None if carrier != "arbitrary" else setting.digits
        return {
            **options,
            "arithmetic": carrier,
            "precision": 16 if digits is None else digits,
        }

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        **options: Any,
    ) -> Any:
        """Per Born helicity row; the colour axis holds one entry per Born
        basis element at LC (``basis-<i>``) and one contracted component at
        NLC and full colour (design §8.5)."""

        return self._observable_runtime().evaluate_resolved(
            momenta, **self._observable_options(precision, arithmetic, options)
        )

    def evaluate_correlated(
        self,
        momenta: Momenta,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        **options: Any,
    ) -> VirtualEvaluation:
        return self._observable_runtime().evaluate_correlated(
            momenta, **self._observable_options(precision, arithmetic, options)
        )

    def evaluate_correlated_many(
        self,
        momenta: Momenta,
        requests: Any,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        **options: Any,
    ) -> dict[str, VirtualEvaluation]:
        return self._observable_runtime().evaluate_correlated_many(
            momenta,
            requests,
            **self._observable_options(precision, arithmetic, options),
        )

    def available_color_correlations(self) -> tuple[Any, ...]:
        return self._observable_runtime().available_color_correlations()

    def set_spin_correlation_vectors(self, vectors: Any) -> None:
        self._observable_runtime().set_spin_correlation_vectors(vectors)

    def _checked_parameters(self, mapping: ModelParameters | None) -> dict[str, str]:
        """Real values of external model parameters, read and checked by
        name (the SDK's order, ``v2_runtime::checked_parameters``: values
        first, then names). Unknown names, the renormalisation scale
        (``mu_r2`` sets it per point) and a value other than the one a
        parameter was fixed to at generation (the record's
        ``fixed_parameters``: the request's and those the model preparation
        fixes) are refused. Other known parameters are accepted even where
        this process does not use them, as by the legacy engine: e.g. a width
        that only loop propagators carry does not enter (v2 virtuals are real
        on-shell)."""

        from .assemble import check_model_parameters

        if isinstance(mapping, Mapping) and all(isinstance(n, str) for n in mapping):
            mapping = {name: mapping[name] for name in sorted(mapping)}
        return check_model_parameters(
            _real_parameters(mapping),
            self._externals,
            self._scale_parameter,
            self._fixed,
            self._prepared_fixed,
        )

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        parameters = self._checked_parameters(mapping)
        with self._lock:
            self._parameters.update(parameters)
            if self._observables is not None:
                self._observables.set_model_parameters(parameters)

    def clear(self) -> None:
        """Drop the native executors (and their master caches)."""
        with self._lock:
            self._executor = self._native_process.executor(
                self._native, externals=self._externals
            )
            if self._observables is not None:
                self._observables.clear()

    def mute_warnings(self) -> None:
        """No effect: v2 runtimes emit no runtime warnings."""

    def unmute_warnings(self) -> None:
        """No effect: v2 runtimes emit no runtime warnings."""


def _stored_member(group_member: Any, representative: str) -> Any:
    """The identity-port flavour member of a stored process group
    (``process-groups.json``, basis ``exact-loop-flavor``): generation proved
    the species substitution on the representative (``observables.
    admit_member``, ``loop_service._v2_member``) and lists only identity-port
    members, so a load reads it without regenerating the representative."""

    from .observables import FlavourMember

    legs = tuple(int(leg) for leg in group_member.permutation)
    if legs != tuple(range(len(legs))):
        raise ArtifactError(
            f"flavour member {group_member.process_id!r} is not an identity-port "
            "member of its representative"
        )
    return FlavourMember(str(group_member.expression), representative, legs, {})


def _zero(carrier: str) -> Any:
    return 0.0 if carrier == "double" else Decimal(0)


def _digits(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("precision must be a positive decimal digit count")
    return value


def _arithmetic(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ("double", "double-double", "arbitrary"):
        raise ValueError("unknown virtual arithmetic")
    return value


# Finite-only Laurent coefficients carry explicitly absent poles.
_ABSENT_POLES = cast(tuple[Any, Any], (None, None))
