# SPDX-License-Identifier: 0BSD
"""Plan v4 loop artifacts and their public runtime (design §12, WP2.6).

Format. A v2 virtual process is published in the ordinary schema-v3 artifact
envelope under the runtime capability ``rusticol.loop-runtime.v4``:

* ``processes/<id>/loop-runtime.pacbin`` holds a member of kind
  ``LOOP_RUNTIME_PLAN``, ``schedule/loop-plan-v4.bin.zlib``: the plan v4
  program of ``plan_export`` (its binary columnar member, zlib-compressed,
  M-FINAL MF-8G-1 item 5; schedule B; the forward oracle F is not stored)
  that the native ``V2Executor`` validates and runs. The oracle switch
  ``native_lowering.BINARY_PLAN = False`` stores the plan JSON instead,
  ``schedule/loop-plan-v4.json.zlib``, which the readers therefore also read
  (the oracle's form, not a compatibility promise: AGENTS.md keeps no old
  internal artifact formats).
  Its kernels are exact polynomial term lists interpreted natively
  (``loop_v2/dress.rs``): no compiled code, so nothing depends on the build
  host's CPU features or paths. A process with certified-scaleless families
  also stores their companion plan (``LoopSlice.scaleless``),
  ``schedule/loop-plan-v4-scaleless.bin.zlib`` (or ``.json.zlib``), which
  only the ``"uv"``
  and ``"ir"`` pole selections run (design §12.4): one artifact serves all
  four selections.
* ``processes/<id>/execution.json`` (schema 4) records the plan's SHA-256
  (of the stored plan's bytes, the member or the JSON; and the companion's,
  or null), ``point_inputs`` and the generation
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
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    EvaluationError,
    ReductionUnavailable,
)
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
    VirtualReduction,
    VirtualRescuePolicy,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
    computes_poles,
    splits_poles,
)
from pyamplicol.config import PoleSelection, R2Strategy
from pyamplicol.config.models import reduction_request
from pyamplicol.config.models import reduction_target as reduction_target_value
from pyamplicol.config.models import reduction_unchecked as reduction_unchecked_value
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin,
)
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection

# ``reduction.SUMMED`` / ``RESOLVED`` are read through the module at call
# time (its PEP 562 attributes load the native policy on first use, not at
# this module's import).
from . import reduction as _routes
from .families import Orders, add_orders
from .native_trees import Program, StoredMember, program_bytes
from .reduction import Route
from .stability import CROSS

PLAN_MEMBER = "schedule/loop-plan-v4.json.zlib"
# The binary plan member (``ExportedPlan.binary()``, M-FINAL MF-8G-1 item 5),
# which a process stores instead of the plan JSON (zlib stream; rusticol-core
# ``loop_runtime::LOOP_PROGRAM_V4_BINARY_MEMBER``); likewise the companion's.
PLAN_BINARY_MEMBER = "schedule/loop-plan-v4.bin.zlib"
SCALELESS_PLAN_BINARY_MEMBER = "schedule/loop-plan-v4-scaleless.bin.zlib"
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
# The binary tree program (``native_trees.NativeTrees.binary``, M-FINAL
# MF-8G-1), which a process stores instead of its JSON (zlib stream).
NATIVE_TREES_BINARY_MEMBER = "tree-program.bin.zlib"
# C4 (M-FINAL MF-8G-1 item 7): the IR dipole operators per relabelling orbit
# (``native_process.NativeProcess.dipoles``; zlib JSON, ``dipoles.FORMAT``).
NATIVE_DIPOLES_MEMBER = "dipoles.json.zlib"
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
    # m-lit LIT-REQ: the served contribution, named only for a loop-induced
    # request (Born-interfered records keep their bytes).
    "contribution": "virtual",
    # m-lit LIT-API: the generation default of the reduction backend (Q17),
    # named only when not "auto".
    "reduction": "auto",
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
    compiles the Born of a slice (``blocks.loop_born_dag``, shared with it
    in an open ``blocks.born_dag_scope``) (without the vetoed |PDG| codes
    ``particle_veto``; a loop Born DAG, ``loop_born``: a Dirac current
    closing on a Weyl sink is a root, ``dag_ordering.DIRAC_WEYL_CLOSURE``)."""

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    from .blocks import _born_roots, loop_born_dag, vetoed_particle_ids

    model = CompiledUFOModel(compiled)
    # In an open ``blocks.born_dag_scope`` (the public route's generation)
    # the slice's ``build_tree_blocks`` reuses this DAG.
    dag = loop_born_dag(
        build_model_process_ir(process, compiled.ir, color_accuracy="full"),
        model,
        max_coupling_orders=dict(bounds),
        ignored_particle_ids=vetoed_particle_ids(particle_veto),
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


def _plan_bytes(plan: Program) -> bytes:
    """A stored plan's bytes: a JSON's text encoded, a binary member's bytes,
    or a loaded process's stored member inflated (``StoredMember``)."""

    return program_bytes(plan)


def plan_digest(plan: Program) -> str:
    """SHA-256 of a stored plan's bytes: its binary plan member, or its JSON
    (text) for a plan stored as JSON."""

    return hashlib.sha256(_plan_bytes(plan)).hexdigest()


def _plan_member(plan: Program, scaleless: bool = False) -> tuple[str, bytes]:
    """(logical path, bytes) of a stored plan: the binary member's path for a
    binary plan member, the JSON's for a JSON."""

    from .native_lowering import is_member

    if isinstance(plan, StoredMember):
        plan = plan.bytes()
    if is_member(plan):
        name = SCALELESS_PLAN_BINARY_MEMBER if scaleless else PLAN_BINARY_MEMBER
    else:
        name = SCALELESS_PLAN_MEMBER if scaleless else PLAN_MEMBER
    return name, _plan_bytes(plan)


def write_plan(
    plan: str | bytes,
    destination: Path,
    native: Any = None,
    scaleless: str | bytes | None = None,
) -> None:
    """Write the plan v4 PACBIN (its digest is ``GeneratedProcess.plan_sha256``,
    computed once from the same bytes), with the companion plan of the
    certified-scaleless families when the process has one and, with
    ``native`` (a ``native_process.NativeProcess`` of that plan), its native
    point inputs. A plan is its binary plan member (``ExportedPlan.
    stored()``, stored as ``PLAN_BINARY_MEMBER``) or a plan JSON (text,
    ``PLAN_MEMBER``)."""

    from io import BytesIO

    if native is not None:
        if _plan_member(native.plan) != _plan_member(plan):
            raise ValueError("the native point inputs belong to another plan")
        companion = native.scaleless
        if (companion is None) != (scaleless is None) or (
            companion is not None
            and _plan_member(companion.plan, scaleless=True)
            != _plan_member(cast(Any, scaleless), scaleless=True)
        ):
            raise ValueError(
                "the native point inputs store the companion of the stored "
                "companion plan"
            )
    plan_kind = PacbinMemberKind.LOOP_RUNTIME_PLAN
    state = PacbinMemberKind.SYMBOLICA_EXACT_STATE
    symjit = PacbinMemberKind.SYMJIT_APPLICATION
    # (logical path, kind, bytes); plan and JSON members are zlib streams.
    name, data = _plan_member(plan)
    members = [(name, plan_kind, zlib.compress(data, 6))]
    if scaleless is not None:
        name, data = _plan_member(scaleless, scaleless=True)
        members.append((name, plan_kind, zlib.compress(data, 6)))
    del data
    if native is not None:
        stored = native.members()
        members.append(
            (
                NATIVE_RECORD_MEMBER,
                plan_kind,
                zlib.compress(stored["native-process.json"], 6),
            )
        )
        prefixes = [("", NATIVE_INPUTS_PREFIX)]
        if companion is not None:
            from .native_process import SCALELESS

            prefixes.append((SCALELESS, NATIVE_SCALELESS_PREFIX))
        for prefix, path in prefixes:
            binary = stored.get(f"{prefix}tree-program.bin")
            members += [
                (
                    f"{path}{NATIVE_TREES_MEMBER}",
                    plan_kind,
                    zlib.compress(stored[f"{prefix}tree-program.json"], 6),
                )
                if binary is None
                else (
                    f"{path}{NATIVE_TREES_BINARY_MEMBER}",
                    plan_kind,
                    zlib.compress(binary, 6),
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
        if "dipoles.json" in stored:
            members.append(
                (
                    f"{NATIVE_INPUTS_PREFIX}{NATIVE_DIPOLES_MEMBER}",
                    plan_kind,
                    zlib.compress(stored["dipoles.json"], 6),
                )
            )
        for name in sorted(stored):
            if name.startswith("observables"):
                members.append(
                    (
                        f"{NATIVE_OBSERVABLES_PREFIX}{name}.zlib",
                        plan_kind,
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


def read_plan(path: Path, member: str = PLAN_MEMBER) -> str | bytes | None:
    """The stored plan v4 of a ``loop-runtime.pacbin`` (payload digests
    checked): the process plan, or with ``member=SCALELESS_PLAN_MEMBER`` the
    companion plan (None when absent); its binary plan member (bytes) where
    the PACBIN stores one (``PLAN_BINARY_MEMBER``), else its JSON (text).
    ``read_plan_json`` gives the JSON of either."""

    binary = {
        PLAN_MEMBER: PLAN_BINARY_MEMBER,
        SCALELESS_PLAN_MEMBER: SCALELESS_PLAN_BINARY_MEMBER,
    }.get(member)
    with PacbinReader.open(path) as reader:
        names = {item.logical_path for item in reader.members}
        if binary in names:
            found_name = cast(str, binary)
        elif member != PLAN_MEMBER and member not in names:
            return None
        else:
            found_name = member
        found = reader.member(found_name)
        if found.kind != PacbinMemberKind.LOOP_RUNTIME_PLAN:
            raise ArtifactError("loop runtime member is not a plan v4")
        compressed = reader.read_member(found_name, length=found.length)
    data = zlib.decompress(compressed)
    del compressed
    if found_name == binary:
        return data
    return data.decode()


def read_plan_json(path: Path, member: str = PLAN_MEMBER) -> str | None:
    """``read_plan``'s plan as its JSON (a binary plan member's written by the
    native model, ``native_lowering.member_json``)."""

    from .native_lowering import member_json

    found = read_plan(path, member)
    return None if found is None else member_json(found)


def _read_member(reader: Any, name: str, kind: PacbinMemberKind) -> bytes:
    """The payload of member ``name`` of an open container, of kind ``kind``."""

    member = reader.member(name)
    if member.kind != kind:
        raise ArtifactError(f"loop runtime member {name!r} has the wrong kind")
    return bytes(reader.read_member(name, length=member.length))


def _native_members(
    reader: Any,
    companion: bool,
    binary_trees: Callable[[str], bytes | StoredMember],
) -> dict[str, bytes | StoredMember]:
    """The native point-input members of an open ``loop-runtime.pacbin``
    (``NativeProcess.from_members``), the one enumeration of both readers
    (``read_native_process``, ``load_native_process``): the assembly record;
    per prefix (the process, and its certified-scaleless companion when
    ``companion``) the tree program (a binary one as ``binary_trees`` reads
    it, a JSON one, an older build's, inflated), the parameter program and
    the kernels with their plane applications; the dipole operators and the
    observable columns."""

    from .native_process import SCALELESS

    names = {member.logical_path for member in reader.index.members}
    json_kind = PacbinMemberKind.LOOP_RUNTIME_PLAN
    state = PacbinMemberKind.SYMBOLICA_EXACT_STATE
    symjit = PacbinMemberKind.SYMJIT_APPLICATION
    members: dict[str, bytes | StoredMember] = {}
    members["native-process.json"] = zlib.decompress(
        _read_member(reader, NATIVE_RECORD_MEMBER, json_kind)
    )
    prefixes = [("", NATIVE_INPUTS_PREFIX)]
    if companion:
        prefixes.append((SCALELESS, NATIVE_SCALELESS_PREFIX))
    for prefix, base in prefixes:
        if base + NATIVE_TREES_BINARY_MEMBER in names:
            members[f"{prefix}tree-program.bin"] = binary_trees(
                base + NATIVE_TREES_BINARY_MEMBER
            )
        else:
            members[f"{prefix}tree-program.json"] = zlib.decompress(
                _read_member(reader, base + NATIVE_TREES_MEMBER, json_kind)
            )
        members[f"{prefix}parameters.bin"] = _read_member(
            reader, base + NATIVE_PARAMETERS_MEMBER, state
        )
        members[f"{prefix}parameters.symjit"] = _read_member(
            reader, base + NATIVE_PARAMETERS_PLANE_MEMBER, symjit
        )
        k = 0
        while base + NATIVE_KERNEL_MEMBER.format(k) in names:
            members[f"{prefix}kernel-{k}.bin"] = _read_member(
                reader, base + NATIVE_KERNEL_MEMBER.format(k), state
            )
            members[f"{prefix}kernel-{k}.symjit"] = _read_member(
                reader, base + NATIVE_KERNEL_PLANE_MEMBER.format(k), symjit
            )
            k += 1
    dipoles = f"{NATIVE_INPUTS_PREFIX}{NATIVE_DIPOLES_MEMBER}"
    if dipoles in names:
        members["dipoles.json"] = zlib.decompress(
            _read_member(reader, dipoles, json_kind)
        )
    start = f"{NATIVE_OBSERVABLES_PREFIX}observables"
    for name in sorted(names):
        if name.startswith(start) and name.endswith(".json.zlib"):
            members[name[len(NATIVE_OBSERVABLES_PREFIX) : -len(".zlib")]] = (
                zlib.decompress(_read_member(reader, name, json_kind))
            )
    return members


def read_native_process(
    path: Path, plan: str | bytes, scaleless: str | bytes | None = None
) -> Any:
    """The native point inputs of a ``loop-runtime.pacbin`` written with them
    (``write_plan(..., native)``), for its stored plan ``plan`` (``read_plan``)
    and, when the process has certified-scaleless families, its companion
    plan ``scaleless`` (whose native tree program it also holds; read from
    the PACBIN when not given), every member inflated in memory (the oracle
    of ``load_native_process``)."""

    from .native_process import NativeProcess

    if scaleless is None:
        scaleless = read_plan(path, SCALELESS_PLAN_MEMBER)

    with PacbinReader.open(path) as reader:
        members = _native_members(
            reader,
            scaleless is not None,
            lambda name: zlib.decompress(
                _read_member(reader, name, PacbinMemberKind.LOOP_RUNTIME_PLAN)
            ),
        )
    try:
        return NativeProcess.from_members(plan, members, scaleless)
    except ValueError as error:
        raise ArtifactError(f"unreadable native point inputs: {error}") from error


def load_native_process(path: Path) -> Any:
    """The native point inputs of a ``loop-runtime.pacbin`` written with them,
    as ``read_native_process`` reads them (``_native_members``), but the
    plan, the companion plan and the binary tree programs are read on demand
    (``StoredMember``, M-FINAL MF-8G-2 item 2): the executors inflate them
    natively, and the process's Python view inflates the small members only
    (assembly record, saved evaluators, plane applications, observable
    columns, dipole operators) and each tree program's head. The container
    is opened once, here: its payload digests are verified through that
    handle, which the stored members keep for the process's lifetime and
    read through (digest checked), so the loaded process does not depend on
    ``path`` (moving, renaming, deleting or replacing the artifact changes no
    result; a file rewritten in place is refused). Without positional reads
    (``os.pread``) the stored members are read here. A JSON tree program (an
    older build's store) is inflated as ``read_native_process`` inflates
    it."""

    import os

    from .native_process import NativeProcess

    handle = Path(path).open("rb")  # noqa: SIM115 - kept by the stored members
    try:
        with PacbinReader.open(handle) as reader:
            index = {member.logical_path: member for member in reader.index.members}

            def stored(name: str) -> StoredMember:
                member = index[name]
                if member.kind != PacbinMemberKind.LOOP_RUNTIME_PLAN:
                    raise ArtifactError(
                        f"loop runtime member {name!r} has the wrong kind"
                    )
                if not hasattr(os, "pread"):
                    kind = PacbinMemberKind.LOOP_RUNTIME_PLAN
                    data = _read_member(reader, name, kind)
                    return StoredMember(name, member.sha256, data=data)
                return StoredMember(
                    name, member.sha256, handle, member.offset, member.length
                )

            # The binary plan members where stored (M-FINAL MF-8G-1 item 5),
            # else the JSON ones (as ``read_plan``).
            plan_name = (
                PLAN_BINARY_MEMBER if PLAN_BINARY_MEMBER in index else PLAN_MEMBER
            )
            if plan_name not in index:
                raise ArtifactError("the loop runtime has no plan v4")
            plan = stored(plan_name)
            scaleless_name = (
                SCALELESS_PLAN_BINARY_MEMBER
                if SCALELESS_PLAN_BINARY_MEMBER in index
                else SCALELESS_PLAN_MEMBER
            )
            scaleless = stored(scaleless_name) if scaleless_name in index else None
            members = _native_members(reader, scaleless is not None, stored)
    except BaseException:
        handle.close()
        raise
    try:
        return NativeProcess.from_members(plan, members, scaleless)
    except ValueError as error:
        raise ArtifactError(f"unreadable native point inputs: {error}") from error


# --------------------------------------------------------------------- generation


@dataclass(frozen=True)
class GeneratedProcess:
    """One process generated through v2: its observables (and slice), plan
    and public catalogue, and the companion plan of its certified-scaleless
    families (``LoopSlice.scaleless``; None without them). Frozen: the
    digests are cached from the stored plans. A stored plan (``plan``,
    ``scaleless_plan``) is its binary plan member (``ExportedPlan.stored()``,
    M-FINAL MF-8G-1 item 5) or a plan JSON (text); ``plan_json`` and
    ``scaleless_plan_json`` give their JSON."""

    observables: Any
    exported: Any
    request: Mapping[str, Any]
    plan: str | bytes
    # The native point inputs (``native_process.NativeProcess``; None only for
    # generation objects that are never stored).
    native: Any = None
    scaleless_exported: Any = None
    scaleless_plan: str | bytes | None = None

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

    @property
    def plan_json(self) -> str:
        """The plan's JSON (formed from a binary member on each call)."""

        from .native_lowering import member_json

        return member_json(self.plan)

    @property
    def scaleless_plan_json(self) -> str | None:
        """The companion plan's JSON, or None without one."""

        from .native_lowering import member_json

        return None if self.scaleless_plan is None else member_json(self.scaleless_plan)

    @cached_property
    def plan_sha256(self) -> str:
        return plan_digest(self.plan)

    @cached_property
    def scaleless_plan_sha256(self) -> str | None:
        if self.scaleless_plan is None:
            return None
        return plan_digest(self.scaleless_plan)

    def executor(self, inputs: str = "python") -> Any:
        """A native executor of the slice on the stored plans; ``inputs=
        "native"`` forms the point inputs from the plan's tree program (the
        stored native process's lane, ``native_process``)."""

        from .plan_export import NativeExecutor

        scaleless = None
        if self.scaleless_exported is not None:
            scaleless = (self.scaleless_exported, cast(Any, self.scaleless_plan))
        return NativeExecutor(
            self.slice,
            exported=self.exported,
            plan_json=self.plan,
            scaleless=scaleless,
            inputs=inputs,
        )

    def contribution(self) -> dict[str, Any]:
        """The metadata of a loop-induced process (``contribution_kind``,
        ``interference`` and the pole selections it serves; m-lit PLAN
        §2.a.9), empty for a Born-interfered one (its metadata keeps the
        defaults and its bytes)."""

        if getattr(self.slice, "amplitude", None) is None:
            return {}
        from pyamplicol.generation.loop_artifact import (
            LOOP_INDUCED_INTERFERENCE,
            LOOP_INDUCED_POLES,
        )

        return {
            "contribution_kind": "loop-induced",
            "interference": LOOP_INDUCED_INTERFERENCE,
            "poles": LOOP_INDUCED_POLES,
        }

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

    def reductions(self) -> tuple[str, ...] | None:
        """The reduction backends the process admits when it is TIR-only
        (``("tirust",)``: a family beyond NinjaRS's rank <= N + 2, m-lit
        lit-eft; generated only where TIRust reduces it,
        ``families.family_rank_refusal``), else None (every backend). A
        generation default naming NinjaRS (``[loop] reduction``) cannot serve
        a TIR-only process and is refused typed."""

        from .families import tir_only

        if not tir_only(self.buckets()):
            return None
        if self.slice.request.reduction == "ninjars":
            raise ReductionUnavailable(
                f"{self.slice.request.process}: the generation default reduction "
                "'ninjars' cannot serve this process: a family is beyond NinjaRS's "
                "rank <= N+2, so the artifact admits the reduction 'tirust' only "
                "(m-lit lit-eft); generate with reduction 'auto' or 'tirust'"
            )
        return ("tirust",)

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
        """The conventions of the request's virtual (WP3.2 EWI-7,
        ``ew_policy.scheme_conventions``: for an electroweak request
        regularisation and lambda_HV, the gamma-five reading, gauge, UV, mass
        and electroweak input schemes, the external-photon convention; the
        complex-mass record; a loop-induced request's width policy, m-lit
        Q7): the generation record's ``conventions`` (``Runtime.inspect`` and
        the artifact inspection read it there); None for a Born-interfered
        request that is neither electroweak nor complex-mass (its record then
        has no entry, so such artifacts keep their bytes)."""

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
    *,
    memo: Any = None,
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
    process regenerates. ``memo`` (a generation's
    ``preparation_cache.PreparationMemo``) shares the model preparation.
    """

    from .assemble import LoopSlice, resolve_request
    from .observables import ObservableSlice

    if request.counterterms != "recursion":
        # Site closure (with its loop matching) is the O6 test oracle only.
        raise ArtifactError(
            "plan v4 processes take their counterterms from the tree recursion; "
            f"counterterms={request.counterterms!r} is a test oracle"
        )
    # The recorded recipe names the corrections the slice takes (a request
    # that names none takes the model's strong correction).
    request = resolve_request(compiled, request)
    declared = _declared(correlators)
    from .native_trees import NativeExportUnsupported

    try:
        # A listed feature the native point inputs do not cover is refused as
        # soon as the generation knows it (``native_inputs``), before the
        # trees are built.
        if declared and request.contribution == "loop-induced":
            # Loop-induced observables are quadratic forms on the amplitude
            # vector (m-lit LIT-OBS, ``induced_observables``): the slice is
            # the inclusive one, its declared histories' matrices are stored
            # with its amplitude basis.
            from .induced_observables import declare

            slice_ = LoopSlice.generate(
                compiled, request, native, native_inputs=True, memo=memo
            )
            declare(slice_, compiled, request, correlators)
            observables = ObservableSlice.of(slice_)
        elif declared:
            observables = ObservableSlice.generate(
                compiled,
                request,
                native,
                correlators=correlators,
                native_inputs=True,
                memo=memo,
            )
        else:
            observables = ObservableSlice.of(
                LoopSlice.generate(
                    compiled, request, native, native_inputs=True, memo=memo
                )
            )
    except NativeExportUnsupported as error:
        raise _not_storable(request, "native point inputs", error) from error
    return export_process(observables, request, declared)


def reexport_process(
    process: GeneratedProcess, compiled: Any, request: Any, correlators: Any = None
) -> GeneratedProcess:
    """``process`` exported again from its own slice (``export_process``), as
    ``generate_process`` exported it for ``request`` and ``correlators``."""

    from .assemble import resolve_request

    return export_process(
        process.observables, resolve_request(compiled, request), _declared(correlators)
    )


def _declared(correlators: Any) -> bool:
    """Whether ``correlators`` declares colour or spin observables."""

    return correlators is not None and bool(
        correlators.color_correlations
        or correlators.spin_correlations
        or correlators.all_color_through_order is not None
    )


def export_process(observables: Any, request: Any, declared: bool) -> GeneratedProcess:
    """The stored plans and native point inputs of a generated slice
    (``generate_process``'s export): exported again, on the same slice, when
    the exact helicity-zero filter of a loop-induced process drops rows
    (``LoopSlice.zero_rows``, ``helicity_filter``)."""

    from pyamplicol.correlators import CorrelatorConfig

    from .native_trees import NativeExportUnsupported
    from .plan_export import export_plan

    # Schedule F (the forward oracle) is not stored: the runtime dresses by
    # schedule B, and F's section is over half of the plan (gg -> gggg: 52 of
    # 98 MB of JSON).
    blocks = observables.slice.blocks
    if request.contribution != "loop-induced":
        # A loop-induced request admits fixed widths in its tree propagators
        # (m-lit Q7: no Born, so no Catani mismatch; the UV counterterm
        # vertices multiply the same widthful tree propagators).
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

    # A loop-induced process has no Born and stores no observable columns:
    # its colour and spin correlations and resolved outputs are quadratic
    # forms on the amplitude vector (``induced_observables``).
    induced = getattr(observables.slice, "amplitude", None) is not None
    plans = {} if induced else column_plans(observables)
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
            column_sets=None
            if induced
            else {kind: plan.seed_columns for kind, plan in plans.items()},
        )
    except NativeExportUnsupported as error:
        raise _not_storable(request, "native point inputs", error) from error
    # The stored plan (``ExportedPlan.stored()``: the binary plan member).
    stored_plan = exported.stored()
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
    scaleless_plan = None if scaleless is None else scaleless.stored()
    if exported.trees is not None:
        from .native_process import NativeProcess

        native = NativeProcess.export(
            observables.slice,
            exported,
            stored_plan,
            None if scaleless is None else (scaleless, cast(Any, scaleless_plan)),
        )
        # A column set the native trees cannot form (no native counterterm
        # components) is refused like the point inputs: a correlated or
        # resolved call of a stored process never regenerates.
        from dataclasses import replace

        from .native_observables import StoredObservables

        if not induced:
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
        stored_plan,
        native=native,
        scaleless_exported=scaleless,
        scaleless_plan=scaleless_plan,
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
    """The selected stored process record and, for a process it serves
    through a leg mapping (``mapping``: a permutation alias or a process-group
    member), that process as a ``GroupMember`` (its id, expression, PDGs and
    representative-to-public leg permutation; else ``None``).

    The precedence is the artifact selector's (rusticol-core
    ``select_process_with_groups``): a stable process id, an alias id, a
    stored expression, an alias expression, the unique stored expression
    equal up to a permutation within each side (the least displaced among
    several), then a process-group member (basis ``exact-loop-flavor``, a
    relation proved exactly at generation): its id, its expression, the
    unique permutation of it within each side. Expressions compare
    whitespace-normalised, exactly and then letter-case-insensitively."""

    from dataclasses import replace

    from pyamplicol.artifacts.process_groups import read_process_groups

    from .mapping import permuted_expression, side_permutation

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
        for alias in record.get("aliases", ()):
            if alias["id"] == process:
                return record, _alias_member(record, alias)
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
        aliases = [
            (record, alias)
            for record in records
            for alias in record.get("aliases", ())
            if _selector_text(str(alias["expression"]), fold=fold) == wanted
        ]
        if len(aliases) > 1:
            choices = ", ".join(sorted(str(alias["id"]) for _r, alias in aliases))
            raise EvaluationError(
                f"process expression {process!r} is ambiguous; select one of "
                f"these alias ids: {choices}"
            )
        if aliases:
            return aliases[0][0], _alias_member(*aliases[0])
    permuted = [
        (record, permutation)
        for record in records
        if (permutation := side_permutation(str(record["expression"]), process))
        is not None
    ]
    if len(permuted) > 1:
        # As the artifact selector: the representative that already agrees
        # with the requested order in the most slots.
        least = min(_displacement(permutation) for _r, permutation in permuted)
        permuted = [item for item in permuted if _displacement(item[1]) == least]
    if len(permuted) > 1:
        choices = ", ".join(sorted(str(record["id"]) for record, _p in permuted))
        raise EvaluationError(
            f"process expression {process!r} is permutation-ambiguous; select "
            f"one of these stable ids: {choices}"
        )
    if permuted:
        return permuted[0][0], _permuted_member(*permuted[0])
    groups = read_process_groups(manifest.root)
    members = [
        (representative, member)
        for representative, group in (groups.groups if groups else {}).items()
        for member in group.members
        if member.process_id != representative and not member.generated
    ]
    for fold in (None, False, True):
        for representative, member in members:
            if (
                member.process_id == process
                if fold is None
                else _selector_text(member.expression, fold=fold)
                == _selector_text(process, fold=fold)
            ):
                return _group_record(records, representative, member, process)
    for representative, member in members:
        permutation = side_permutation(member.expression, process)
        if permutation is not None:
            # Representative leg r -> member leg -> public leg.
            composed = replace(
                member,
                expression=permuted_expression(member.expression, permutation),
                pdgs=_permuted(member.pdgs, permutation),
                permutation=tuple(permutation[m] for m in member.permutation),
            )
            return _group_record(records, representative, composed, process)
    raise EvaluationError(f"artifact has no process {process!r}")


def _group_record(
    records: Sequence[Mapping[str, Any]], representative: str, member: Any, process: str
) -> tuple[Mapping[str, Any], Any]:
    """The stored representative of a process-group member a loop artifact
    serves: a relation proved exactly at generation (basis
    ``exact-loop-flavor``, factor 1, no charge conjugation). A relation of
    any other basis (a numerical process-group analysis) is refused."""

    if (
        str(member.basis) != "exact-loop-flavor"
        or member.factor != 1
        or (member.conjugated)
    ):
        raise EvaluationError(
            f"process {process!r} is a process-group member of basis "
            f"{str(member.basis)!r} (factor {member.factor}"
            + (", charge-conjugated" if member.conjugated else "")
            + "); a loop artifact serves exact species-substitution members "
            "(basis 'exact-loop-flavor', process.flavor_blind) only, never a "
            "numerical process-group relation; generate the process itself"
        )
    for record in records:
        if record["id"] == representative:
            return record, member
    raise ArtifactError(
        f"process-group representative {representative!r} is not stored"
    )


def _alias_member(record: Mapping[str, Any], alias: Mapping[str, Any]) -> Any:
    """A stored process's permutation alias (its manifest alias record) as a
    group member of basis identity."""

    from pyamplicol.generation.process_grouping import GroupMember

    expression = str(alias["expression"])
    return GroupMember(
        process_id=str(alias["id"]),
        expression=expression,
        pdgs=tuple(int(pdg) for pdg in alias["external_pdgs"]),
        initial_count=len(expression.split(">")[0].split()),
        permutation=tuple(int(leg) for leg in alias["external_permutation"]),
        generated=False,
    )


def _permuted_member(record: Mapping[str, Any], permutation: Sequence[int]) -> Any:
    """A stored process in another order within each side: no member for the
    identity, else a group member of basis identity that keeps the stored
    process's id (as the artifact selector's inferred permutation)."""

    from pyamplicol.generation.process_grouping import GroupMember

    from .mapping import permuted_expression

    if tuple(permutation) == tuple(range(len(permutation))):
        return None
    expression = permuted_expression(str(record["expression"]), permutation)
    return GroupMember(
        process_id=str(record["id"]),
        expression=expression,
        pdgs=_permuted(record["external_pdgs"], permutation),
        initial_count=len(expression.split(">")[0].split()),
        permutation=tuple(permutation),
        generated=False,
    )


def _permuted(values: Sequence[Any], permutation: Sequence[int]) -> tuple[int, ...]:
    out = [0] * len(values)
    for representative, public in enumerate(permutation):
        out[public] = int(values[representative])
    return tuple(out)


def _displacement(permutation: Sequence[int]) -> int:
    return sum(1 for index, image in enumerate(permutation) if index != image)


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

    ``process`` selects a process record or a process the artifact serves
    through a stored representative (``mapping``: a permutation alias, a
    side-permuted expression or a process-group member of
    ``process-groups.json``): its momenta, colour histories, spin legs and
    helicity rows are read on the representative (generation proved a
    species substitution, ``observables.admit_member``; ``_stored_member``).

    ``stability`` (a ``VirtualRescuePolicy``) is the default rescue of
    ``evaluate`` (``loops.v2.rescue``; off by default), used by every call
    that names neither its own policy nor an arithmetic, a precision or
    stability checks.

    ``reduction`` (``"auto"``, ``"ninjars"`` or ``"tirust"``) is the
    reduction backend every call requests that names none, by default the
    artifact's generation default (``[loop] reduction``, the request
    record's ``reduction``; ``"auto"`` when absent). Each call resolves it
    once per carrier (``loops.v2.reduction.resolve``) and records the
    requested and effective backends and the reason in its lane
    diagnostics.
    """

    # The class default; a loaded runtime reads its process's kind from the
    # artifact metadata ("virtual" or "loop-induced", m-lit Q36).
    contribution_kind: Literal["virtual", "loop-induced"] = "virtual"
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
        stability: VirtualRescuePolicy | None = None,
        reduction: str | None = None,
        reduction_target: float | None = None,
        reduction_unchecked: bool | None = None,
    ) -> None:
        from pyamplicol.artifacts import MANIFEST_NAME, load_manifest
        from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS
        from pyamplicol.loops.native import load_loop_native
        from pyamplicol.models.loading import CompiledModel
        from pyamplicol.runtime.virtual import v2_r2_strategy

        from .rescue import rescue_policy

        # A default policy owns the carriers: refused with load-time ones.
        self._rescue = rescue_policy(
            stability, arithmetic=arithmetic, precision=precision
        )
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
        self._mapping = (
            None if group_member is None else _mapping_kind(selected, group_member)
        )
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
        # families): no slice is regenerated (WP3.1 I6). A mapped process is
        # evaluated by its representative at permuted momenta (``mapping``);
        # correlated and resolved calls run the stored observable columns
        # (``native_observables``). Nothing of a loaded process regenerates
        # (REQUIREMENTS_ADDENDUM G).
        runtime = root / "processes" / process_id / str(execution["runtime_path"])
        # File-backed (M-FINAL MF-8G-2 item 2): the plans and binary tree
        # programs stay in the PACBIN, read by the native executors.
        self._native_process = load_native_process(runtime)
        # A loop-induced process stores no observable columns: its
        # observables are quadratic forms on the amplitude vector
        # (``induced_observables``).
        self._induced = "amplitude" in self._native_process.record
        # A loop-induced process's colour-zero record (m-lit Q13: the one
        # exact zero, by colour algebra alone: no colour coefficient of the
        # loop amplitude and no counterterm colour tensor is nonzero at
        # N_c = 3, ``induced.bind``).
        self._colour_zero: dict[str, object] | None = None
        if self._induced:
            from .induced import StoredAmplitude

            stored = StoredAmplitude.from_record(
                self._native_process.record["amplitude"]
            )
            self._colour_zero = {
                "exact_colour_zero": stored.colour_zero,
                "colour_zero_loop_orders": [
                    dict(order) for order in sorted(stored.colour_zero_orders)
                ],
                # M-LIT lit-fold: the fold zero's record (USER ~22:39: an
                # exact zero proven by the exact test), stored only with one.
                **(
                    {
                        "exact_fold_zero": stored.fold_zero["exact_fold_zero"],
                        "fold_zero_loop_orders": [
                            dict(order)
                            for order in stored.fold_zero["fold_zero_loop_orders"]
                        ],
                        "fold_certificate": dict(stored.fold_zero["certificate"]),
                    }
                    if stored.fold_zero is not None
                    else {}
                ),
            }
        if self._native_process.observables is None and not self._induced:
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
        self.contribution_kind = cast(
            Literal["virtual", "loop-induced"], self._metadata["contribution_kind"]
        )
        # The reduction backends the artifact admits: those its declarations
        # name (a TIR-only process, m-lit lit-eft: TIRust), else every one.
        admitted = self._metadata.get("reductions")
        self._admitted: frozenset[str] | None = (
            None if admitted is None else frozenset(admitted)
        )
        # The reduction backend a call requests when it names none: the
        # load-time one, else the artifact's generation default (m-lit Q17).
        self._reduction_request = reduction_request(
            execution["request"].get("reduction", "auto")
            if reduction is None
            else reduction
        )
        self._reductions: dict[tuple[str, str, Route], VirtualReduction] = {}
        # The target accuracy of a call that sets none (m-lit F3 (a); None:
        # the native policy's default).
        self._reduction_target = (
            None
            if reduction_target is None
            else reduction_target_value(reduction_target)
        )
        # The unchecked setting of a call that sets none (m-lit lit-unchecked;
        # None: the native policy's default, off).
        self._reduction_unchecked = (
            None
            if reduction_unchecked is None
            else reduction_unchecked_value(reduction_unchecked)
        )
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
        columns of the native process (nothing regenerates). A loop-induced
        process's (m-lit LIT-OBS; ``V2InducedObservablesRuntime``) are
        quadratic forms on the amplitude vectors of the process's own
        executor, with the declared histories' stored matrices; they run the
        runtime's current executor (``clear`` replaces it) under the
        runtime's lock (the inclusive calls share the executor). A permuted
        mapped process reads the histories and spin classes on its labels."""

        from pyamplicol.runtime.virtual import (
            V2InducedObservablesRuntime,
            V2ObservablesRuntime,
        )

        from .induced_observables import InducedObservables
        from .native_observables import StoredObservableSlice
        from .observables import FORCED_FFT_OBSERVABLES

        if self._induced:
            kind: Any = V2InducedObservablesRuntime

            def observables() -> Any:
                return InducedObservables(
                    self._executor, current=lambda: self._executor, lock=self._lock
                )

        else:
            if self._execution["request"]["colour_contraction"] == "fft":
                # The stored observable columns are contracted directly.
                raise CompatibilityError(FORCED_FFT_OBSERVABLES)
            kind = V2ObservablesRuntime

            def observables() -> Any:
                correlators = correlators_from_record(self._execution["request"])
                return StoredObservableSlice(
                    self._native_process,
                    self._native_process.observables,
                    () if correlators is None else correlators.color_correlations,
                    self._native,
                )

        with self._lock:
            if self._observables is None:
                found = observables()
                if self._member is not None and self._member.alias:
                    from .mapping import MappedObservables

                    found = MappedObservables(found, self._member)
                runtime = kind(
                    found,
                    member=self._member,
                    mapping=self._mapping,
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
        member = self._group_member
        if member is None:
            return tuple(range(self.external_count))
        return tuple(int(leg) for leg in member.permutation)

    @property
    def physics(self) -> ProcessPhysics:
        accuracy = cast(Literal["lc", "nlc", "full"], self._metadata["color_accuracy"])
        member = self._group_member
        # A mapped process's resolved components are the stored basis
        # elements; it is anchored to the stored flows read on its own legs.
        anchors = (
            tuple(
                BornAnchoredColorComponent(
                    flow.id,
                    index,
                    (flow.basis,),
                    f"leading-colour flow {flow.word} (Born basis element "
                    f"{flow.basis})",
                )
                for index, flow in enumerate(self._public_flows())
                if flow.basis is not None
            )
            if accuracy == "lc"
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
                for i, h in enumerate(self._helicity_rows())
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
            contribution_kind=self.contribution_kind,
            reduction=PhysicsReduction(kind, ()),
            model_parameters=(),
            # Selection of a mapped process is refused (``_refuse_member_
            # selection``).
            selector_capabilities=("helicity", "color_flow")
            if anchors and member is None
            else (),
        )

    def _helicity_rows(self) -> list[tuple[int, ...]]:
        """The stored helicity rows, on a mapped process's legs in its own
        sorted order (as its resolved rows, ``V2ObservablesRuntime``)."""

        rows = [tuple(int(h) for h in row) for row in self._physics["helicities"]]
        if self._member is None or not self._member.alias:
            return rows
        legs = self._member.legs
        return sorted(tuple(row[leg] for leg in legs) for row in rows)

    def _flows(self) -> tuple[Any, ...]:
        """The stored leading-colour flows (none below leading colour)."""

        from .selection import Flow

        record = self._native_process.observables.record
        return tuple(Flow.from_record(flow) for flow in record.get("color_flows", ()))

    def _public_flows(self) -> tuple[Any, ...]:
        """The stored leading-colour flows on the selected process's legs: a
        permuted mapped process reads each word through its member legs
        (representative label ``legs[i] + 1`` is its label ``i + 1``)."""

        from dataclasses import replace

        from .selection import flow_id

        flows = self._flows()
        if self._member is None or not self._member.alias:
            return flows
        public = {rep + 1: leg + 1 for leg, rep in enumerate(self._member.legs)}
        out = []
        for flow in flows:
            word = tuple(public.get(label, label) for label in flow.word)
            out.append(replace(flow, id=flow_id(word), word=word))
        return tuple(out)

    def inspect(self) -> Mapping[str, object]:
        with self._lock:
            return deepcopy(
                {
                    "artifact_id": self.artifact_id,
                    "process_id": self._public_id,
                    "representative_process_id": self._process_id,
                    **(
                        {"external_permutation": list(self.external_permutation)}
                        if self._group_member is not None
                        else {}
                    ),
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
                    # The reduction backend (m-lit Q17 / Q18): the request of
                    # a call that names none, the artifact's generation
                    # default, the per-type default of "auto" and what the
                    # native extension serves (backend -> carriers).
                    "reduction": self._reduction_summary(),
                    **({} if self._colour_zero is None else self._colour_zero),
                    "parameter_overrides": dict(self._parameters),
                    "scale_input": "explicit positive mu_r2, scalar or per point",
                    "stability_checks": (
                        "rotations (count or exact rotations) and the "
                        "reduction-direction check, optionally rescaled; none "
                        "by default"
                    ),
                    # The built-in rescue (``loops.v2.rescue``): a
                    # VirtualRescuePolicy per evaluate call or at load.
                    "rescue_available": True,
                    "default_rescue_levels": (
                        None if self._rescue is None else len(self._rescue.levels)
                    ),
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
        stability: VirtualRescuePolicy | None = None,
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        reduction: str | None = None,
        reduction_target: float | None = None,
        reduction_unchecked: bool | None = None,
    ) -> VirtualEvaluation:
        """The virtual of every point in one carrier, or under a rescue
        policy (``stability``, else the load-time default; ``loops.v2.
        rescue``) level by level from the original exact inputs.
        ``reduction`` overrides the runtime's reduction-backend request for
        the call (resolved once per carrier, before any point is
        evaluated), ``reduction_target`` its target accuracy (a rescue
        level's own target drives its level), ``reduction_unchecked`` its
        unchecked setting (every level's and the per-point fallback's
        reductions; m-lit lit-unchecked). With the default policy
        (``VirtualRescuePolicy()``) a loop-induced call requesting ``"auto"``
        runs the loop-induced default stack (``rescue.default_stack``)."""

        from pyamplicol.runtime.virtual import _inputs, v2_options

        from .rescue import default_stack, rescue, rescue_policy

        policy = rescue_policy(
            stability,
            default=self._rescue,
            arithmetic=arithmetic,
            precision=precision,
            default_precision=self.default_precision,
            stability_checks=stability_checks,
        )
        selected = bool(helicities or color_flows)
        if policy is not None:
            requested = self._requested(reduction)
            policy = default_stack(
                policy,
                self.contribution_kind,
                requested,
                self._reduction(
                    requested,
                    "double",
                    _routes.RESOLVED if selected else _routes.SUMMED,
                ).fallback,
            )
        choices = self._reduction_choices(
            reduction,
            arithmetic,
            precision,
            policy,
            _routes.RESOLVED if selected else _routes.SUMMED,
            stability_checks,
            reduction_target,
            reduction_unchecked,
        )
        if selected:
            # A leading-colour selection (``selection``): one summed output
            # of the stored resolved columns.
            carrier_options = (
                self._observable_options(precision, arithmetic, {})
                if policy is None
                else {}
            )
            return cast(
                VirtualEvaluation,
                self._observable_runtime().evaluate(
                    momenta,
                    helicities=helicities,
                    color_flows=color_flows,
                    mu_r2=mu_r2,
                    r2_strategy=r2_strategy,
                    stability=policy,
                    poles=poles,
                    stability_checks=stability_checks,
                    on_failure=on_failure,
                    reductions=choices,
                    **carrier_options,
                ),
            )
        selection, checks = v2_options(
            r2_strategy=r2_strategy,
            poles=poles,
            stability_checks=stability_checks,
            on_failure=on_failure,
        )
        if selection.value not in self._metadata["poles"]:
            # A loop-induced process serves "all" and "none" (its UV and IR
            # parts are not additive in |A|^2, m-lit PLAN §2.a.7).
            raise CompatibilityError(
                f"the pole selection {selection.value!r} is not served by this "
                f"{self._metadata['contribution_kind']} process (it serves "
                f"{list(self._metadata['poles'])})"
            )
        # v2_options refused direct R2.
        r2 = self._default_r2 if r2_strategy is None else R2Strategy(r2_strategy)
        carrier, setting = (
            self._carrier(arithmetic, precision) if policy is None else (None, None)
        )
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        if self._member is not None:
            points = [self._member.point(point) for point in points]
        with self._lock:
            parameters = dict(self._parameters)
            if policy is None:
                return self._evaluate_level(
                    points,
                    scales,
                    parameters,
                    cast(str, carrier),
                    setting,
                    r2,
                    selection,
                    checks,
                    on_failure,
                    choices[cast(str, carrier)],
                )

            def level(indices: tuple[int, ...], rung: Any) -> VirtualEvaluation:
                carrier, setting = self._carrier(rung.arithmetic, rung.precision)
                return self._evaluate_level(
                    [points[i] for i in indices],
                    [scales[i] for i in indices],
                    parameters,
                    carrier,
                    setting,
                    r2,
                    selection,
                    rung.checks,
                    "return",
                    choices[rung],
                )

            return rescue(policy, len(points), level, on_failure=on_failure)

    def _reduction(
        self, requested: str, carrier: str, route: Route
    ) -> VirtualReduction:
        """The reduction backend of ``requested`` at ``carrier`` on ``route``
        (``loops.v2.reduction.resolve`` for this process's contribution kind,
        the native extension and the backends the route's executor runs;
        fixed per runtime, so cached)."""

        key = (requested, carrier, route)
        found = self._reductions.get(key)
        if found is None:
            from .reduction import resolve

            found = resolve(
                requested,
                contribution=self.contribution_kind,
                carrier=carrier,
                native=self._native,
                route=route,
                admitted=self._admitted,
            )
            self._reductions[key] = found
        return found

    def _reduction_summary(self) -> dict[str, object]:
        """``inspect``'s reduction-backend record."""

        from .reduction import (
            BACKENDS,
            auto_default,
            default_target,
            default_unchecked,
            native_backends,
        )

        return {
            "requested": self._reduction_request,
            "artifact_default": self._execution["request"].get("reduction", "auto"),
            # The one helper (``reduction_policy::auto_default``).
            "auto_default": auto_default(self.contribution_kind, self._admitted),
            # The target accuracy of a call that sets none (m-lit F3).
            "target": (
                default_target()
                if self._reduction_target is None
                else self._reduction_target
            ),
            # The unchecked setting of a call that sets none (m-lit
            # lit-unchecked).
            "unchecked": (
                default_unchecked()
                if self._reduction_unchecked is None
                else self._reduction_unchecked
            ),
            # The backends the artifact admits (m-lit lit-eft: TIR-only).
            "admitted": sorted(BACKENDS if self._admitted is None else self._admitted),
            "native": {
                name: sorted(carriers)
                for name, carriers in native_backends(self._native).items()
            },
        }

    def _requested(self, reduction: str | None) -> str:
        """A call's reduction request: its own, else the runtime's."""

        return (
            self._reduction_request
            if reduction is None
            else reduction_request(reduction)
        )

    def _reduction_choices(
        self,
        reduction: str | None,
        arithmetic: str | None,
        precision: int | None,
        policy: VirtualRescuePolicy | None,
        route: Route,
        checks: VirtualStabilityChecks | None = None,
        target: float | None = None,
        unchecked: bool | None = None,
    ) -> dict[Any, VirtualReduction]:
        """The reduction backend of a call on ``route``, resolved before any
        point is evaluated: keyed by the call's carrier, or under a rescue
        policy by each ``VirtualRescueLevel``, resolved once per level for
        its carrier from the level's own request (``VirtualRescueLevel.
        reduction``; ``"auto"`` takes the call's; m-lit Q17 / Q22). The
        effective backend is one the route's executor runs (the guard of
        every route, ``reduction.Route``): an explicit backend that cannot
        serve one of them is refused typed, ``auto`` falls back with its
        reason (m-lit Q17 / Q18; never a silent switch). With the
        cross-backend check (``checks``) the other backend must serve the
        carrier as well (refused typed otherwise). Each choice carries its
        target accuracy (``VirtualReduction.target``): a level's own target
        under a policy, else the call's ``target``, else the runtime's
        (m-lit F3 (a)), and the unchecked setting (``VirtualReduction.
        unchecked``): the call's ``unchecked``, else the runtime's, else
        off, for every level (m-lit lit-unchecked)."""

        from .reduction import default_unchecked

        requested = self._requested(reduction)
        call_unchecked = (
            reduction_unchecked_value(unchecked)
            if unchecked is not None
            else self._reduction_unchecked
            if self._reduction_unchecked is not None
            else default_unchecked()
        )
        if policy is not None:
            # The levels own their checks (``rescue.RESCUE_OWNS_CARRIER``)
            # and their targets.
            levels = {
                level: replace(
                    self._reduction(
                        level.backend_request(requested),
                        self._carrier(level.arithmetic, level.precision)[0],
                        route,
                    ),
                    target=level.target,
                    unchecked=call_unchecked,
                )
                for level in policy.levels
            }
            # The policy refuses a repeated (precision bits, reduction) pair
            # of its levels' requests; an "auto" level may resolve to the
            # backend another level at its precision names (e.g. a binary64
            # "auto" and a binary64 "ninjars" level on a Born-interfered
            # call), which would evaluate the same points again on the same
            # backend and carrier: refused the same way, before any point.
            pairs = [
                (level.precision_bits, choice.effective)
                for level, choice in levels.items()
            ]
            if len(set(pairs)) != len(pairs):
                raise ValueError(
                    "two rescue levels at the same precision differ in their "
                    "reduction backend: repeated (precision bits, backend) pair "
                    f"in this call's resolved levels {pairs}"
                )
            return levels
        carrier = self._carrier(arithmetic, precision)[0]
        from .reduction import default_target

        # The call's target, else the runtime's, else the policy's default.
        call_target = (
            reduction_target_value(target)
            if target is not None
            else self._reduction_target
            if self._reduction_target is not None
            else default_target()
        )
        choice = replace(
            self._reduction(requested, carrier, route),
            target=call_target,
            unchecked=call_unchecked,
        )
        # (A checks value of another type is refused by ``v2_options``.)
        if getattr(checks, "cross_backend", False) is True:
            from .reduction import cross, text

            try:
                self._reduction(cross(choice.effective), carrier, route)
            except ReductionUnavailable as error:
                raise ReductionUnavailable(text("cross", detail=str(error))) from error
        return {carrier: choice}

    def _evaluate_level(
        self,
        points: list[list[list[str]]],
        scales: list[str],
        parameters: dict[str, str],
        carrier: str,
        setting: Any,
        r2: R2Strategy,
        selection: PoleSelection,
        checks: VirtualStabilityChecks,
        on_failure: str,
        reduction: VirtualReduction,
    ) -> VirtualEvaluation:
        """One batch of exact points in one carrier with one set of checks
        (the whole ``evaluate`` without a rescue policy, and each level of
        one), on the ``reduction`` backend (one the native executor runs,
        ``_reduction_choices`` on ``reduction.SUMMED``) at its target, with
        its per-point fallback (``reduction.with_fallback``: the points it
        refuses or fails again on NinjaRS at ``carrier``, each served one
        guarded by its reduction-direction check against the target,
        recorded per point; a fallback point's cross-backend check is not
        run, m-lit R6)."""

        from .reduction import with_fallback

        def run(
            indices: list[int] | None, own: VirtualStabilityChecks
        ) -> tuple[list[Any], dict[int, list[str]]]:
            if indices is None:
                return self._run(points, scales, parameters, setting, selection, own)
            return self._run(
                [points[i] for i in indices],
                [scales[i] for i in indices],
                parameters,
                setting,
                selection,
                own,
            )

        results, failures, served = with_fallback(
            run, reduction, len(points), checks, carrier
        )
        if failures and on_failure == "raise":
            index = min(failures)
            raise EvaluationError(
                f"v2 virtual evaluation failed at point {index}: "
                + "; ".join(failures[index])
            )
        return self._result(
            results,
            failures,
            setting,
            carrier,
            r2,
            selection,
            checks,
            reduction,
            served,
        )

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
        ``plan_export.FailedPoints``; so do the points of a mu_R^2 at which a
        counterterm coupling is not finite, ``isolate_scale_failures``)."""

        from pyamplicol.runtime.virtual import isolate_scale_failures

        from .plan_export import FailedPoints

        if not points:
            return [], {}
        if len({Fraction(scale) for scale in scales}) > 1 and self._scale_partitioned(
            poles
        ):
            return self._run_per_scale(
                points, scales, parameters, setting, poles, checks
            )

        def batch(indices: list[int]) -> tuple[list[Any], dict[int, list[str]]]:
            try:
                return (
                    list(
                        self._executor_for(poles).evaluate(
                            [points[i] for i in indices],
                            [scales[i] for i in indices],
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

        return isolate_scale_failures(batch, scales)

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
        reduction: VirtualReduction | None = None,
        served: Mapping[int, VirtualReduction] | None = None,
    ) -> VirtualEvaluation:
        """The public result of a batch; ``served`` maps the points the
        per-point fallback evaluated again to their own reduction record."""

        from pyamplicol.runtime.virtual import v2_public_estimate

        with_poles = computes_poles(poles)
        bits = _precision_bits(carrier, setting.digits)
        detail = "accuracy is estimated only by the requested stability checks"
        if self._colour_zero is not None and self._colour_zero["exact_colour_zero"]:
            # The colour-zero record (m-lit Q13): exactly 0, not a numerical
            # zero.
            detail = (
                "exact colour zero: every colour coefficient of the loop "
                "amplitude vanishes at N_c = 3 (colour algebra), the value is "
                "exactly 0"
            )
        elif self._colour_zero is not None and self._colour_zero.get("exact_fold_zero"):
            # The fold zero (M-LIT lit-fold, USER ~22:39): exactly 0 by the
            # engine's exact algebra, with the certificate's failure bound.
            certificate = self._colour_zero["fold_certificate"]
            detail = (
                "exact fold zero: the exact test certified the loop chains "
                "proportional and their weights cancel exactly (F_p, p = "
                f"{certificate['prime']}, {certificate['points']} point(s), "
                f"{certificate['identities']} identities, failure bound "
                f"{certificate['failure_bound']:.3g} <= "
                f"{certificate['failure_probability']:.3g}), the value is "
                "exactly 0"
            )
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
                        reduction=(served or {}).get(index, reduction),
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
            own = (served or {}).get(index)
            estimate = v2_public_estimate(
                result.stability,
                with_poles,
                contribution=self.contribution_kind,
                sectors=[sector.order for sector in self._sectors],
            )
            if own is not None and estimate is not None and checks.cross_backend:
                # The fallback point's cross-backend check is not run (R6).
                estimate = replace(
                    estimate, not_run=(*estimate.not_run, (CROSS, own.reason))
                )
            estimates.append(estimate)
            parts.append(self._pole_parts(result) if splits_poles(poles) else None)
            diagnostics.append(
                VirtualLaneDiagnostic(
                    "success",
                    arithmetic,
                    bits,
                    False,
                    detail,
                    primary_color_contraction=contraction,
                    reduction=(served or {}).get(index, reduction),
                    finiteness=result.finiteness,
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
            contribution=self.contribution_kind,
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

    def _observable_call(
        self,
        name: str,
        args: tuple[Any, ...],
        precision: int | None,
        arithmetic: str | None,
        reduction: str | None,
        options: dict[str, Any],
    ) -> Any:
        """One observable call (``V2ObservablesRuntime``) on the call's
        reduction backend, resolved before it runs (``_reduction_choices`` on
        ``reduction.RESOLVED``, with the call's target accuracy) and recorded
        in its lane diagnostics as they are built (``reductions``)."""

        target = options.pop("reduction_target", None)
        unchecked = options.pop("reduction_unchecked", None)
        choices = self._reduction_choices(
            reduction,
            arithmetic,
            precision,
            options.get("stability"),
            _routes.RESOLVED,
            options.get("stability_checks"),
            target,
            unchecked,
        )
        operation = getattr(self._observable_runtime(), name)
        return operation(
            *args,
            reductions=choices,
            **self._observable_options(precision, arithmetic, options),
        )

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        reduction: str | None = None,
        **options: Any,
    ) -> Any:
        """Per Born helicity row; the colour axis holds one entry per Born
        basis element at LC (``basis-<i>``) and one contracted component at
        NLC and full colour (design §8.5). A loop-induced process resolves
        its own helicity rows with one contracted component (``"full"``) or,
        with ``color_resolution="amplitude-basis"``, its amplitude-basis
        anchors (``amplitude-<t>``; ``runtime.virtual.
        V2InducedObservablesRuntime``)."""

        return self._observable_call(
            "evaluate_resolved", (momenta,), precision, arithmetic, reduction, options
        )

    def evaluate_correlated(
        self,
        momenta: Momenta,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        reduction: str | None = None,
        **options: Any,
    ) -> VirtualEvaluation:
        return cast(
            VirtualEvaluation,
            self._observable_call(
                "evaluate_correlated",
                (momenta,),
                precision,
                arithmetic,
                reduction,
                options,
            ),
        )

    def evaluate_correlated_many(
        self,
        momenta: Momenta,
        requests: Any,
        *,
        precision: int | None = None,
        arithmetic: str | None = None,
        reduction: str | None = None,
        **options: Any,
    ) -> dict[str, VirtualEvaluation]:
        return cast(
            dict[str, VirtualEvaluation],
            self._observable_call(
                "evaluate_correlated_many",
                (momenta, requests),
                precision,
                arithmetic,
                reduction,
                options,
            ),
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


def _mapping_kind(record: Mapping[str, Any], group_member: Any) -> str:
    """How the stored process ``record`` serves the selected mapped process
    (``_select``), for messages: a manifest permutation alias, the stored
    process selected in another leg order within each side, or a
    process-group (flavour) member, possibly in permuted leg order."""

    process_id = str(group_member.process_id)
    if process_id == str(record["id"]):
        return "a side permutation of the stored process"
    if any(str(alias["id"]) == process_id for alias in record.get("aliases", ())):
        return "a permutation alias"
    permutation = tuple(int(leg) for leg in group_member.permutation)
    permuted = permutation != tuple(range(len(permutation)))
    return "a process-group flavour member" + (
        " in permuted leg order" if permuted else ""
    )


def _stored_member(group_member: Any, representative: str) -> Any:
    """The mapped process (``mapping``) of a selected alias, side-permuted
    expression or process-group member (``_select``) as a ``FlavourMember``
    of its stored representative: ``legs`` inverts the group member's
    representative-to-public permutation. A species substitution was proved
    on the representative at generation (``observables.admit_member``,
    ``loop_service._v2_member``), so a load reads it without regenerating."""

    from .mapping import inverse
    from .observables import FlavourMember

    legs = inverse(tuple(int(leg) for leg in group_member.permutation))
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
