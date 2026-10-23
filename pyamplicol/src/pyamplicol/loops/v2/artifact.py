# SPDX-License-Identifier: 0BSD
"""Plan v4 loop artifacts and their public runtime (design §12, WP2.6).

Format. A v2 virtual process is published in the ordinary schema-v3 artifact
envelope under the runtime capability ``rusticol.loop-runtime.v4``:

* ``processes/<id>/loop-runtime.pacbin`` holds one member of kind
  ``LOOP_RUNTIME_PLAN``, ``schedule/loop-plan-v4.json.zlib``: the plan v4
  program of ``plan_export`` (zlib-compressed JSON, schedule B; the forward
  oracle F is not stored) that the native ``V2Executor`` validates and runs.
  Its kernels are exact polynomial term lists interpreted natively
  (``loop_v2/dress.rs``): no compiled code, so nothing depends on the build
  host's CPU features or paths.
* ``processes/<id>/execution.json`` (schema 4) records the plan's SHA-256,
  ``point_inputs`` and the generation request (process, coupling orders,
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

Point inputs (``point_inputs = "regenerated-at-load"``). Trees, Born
amplitudes and counterterm scalars are still evaluated per point in Python
(design §5.5), from the generation objects of ``LoopSlice``. Until the
loop-request mode of the Rust recurrence (WP3.1) moves them native, the
runtime rebuilds those objects at load time from the artifact's own compiled
model and the saved request (the recursive generator, no diagram is listed;
the SymJIT tree kernels are compiled in memory at -O2 and never stored) and
requires the regenerated plan to have the stored plan's digest (byte
equality): the Python input tables (block rows, Born components, kernel
constants) must match the stored program's numbering, or the native
executor would read wrong inputs. Regeneration does not depend on the
interpreter's Symbolica symbol history: tree stages use canonical symbols
(``assemble._canonical_stage``) and kernel terms a canonical order
(``reference_executor.ExactKernel``), so plan and binary64 values are
identical after unrelated symbols were created (tested). The same process
reuses generation objects already built in this interpreter (generation
and its staged validation). WP3.1 adds native
programs for the point inputs as further PACBIN members under a new
``point_inputs`` value; the plan member and the declarations stay as they
are, so loading needs no regeneration then (requirements addendum G).

Evaluation returns the public result types: Born per squared Born order, the
complete virtual per squared coupling-order sector with its three Laurent
coefficients (finite only on request), and, when stability checks were asked
for, one ``VirtualStabilityEstimate`` per point. Correlated (colour and spin)
and resolved outputs and flavour members run through the observables package
(``runtime.virtual.V2ObservablesRuntime`` over ``observables.ObservableSlice``),
whose column plans are exported from the same regenerated slice. Binary64,
double-double and arbitrary precision run natively. Escalation of the
arithmetic is the caller's decision (requirements addendum C).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import ModelParameters, Momenta, ScalarInput
from pyamplicol.api.results import (
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
)
from pyamplicol.api.virtual_results import (
    LaurentCoefficients,
    VirtualColorContraction,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
    VirtualStabilityPolicy,
)
from pyamplicol.config import R2Strategy
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
ENGINE = "v2"
# Generation objects of recently generated or loaded processes, keyed by the
# compiled-model payload digest, the request record and the plan digest.
_SLICE_CACHE_SIZE = 4
_slices: OrderedDict[tuple[str, str, str], tuple[Any, Any]] = OrderedDict()
_slices_lock = RLock()


# --------------------------------------------------------------------- request


def _orders_record(orders: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(orders.items()))


def _encode_field(name: str, value: Any) -> Any:
    """One ``SliceRequest`` field as JSON: mappings sorted, tuples as lists,
    the colour declaration as its dictionary; anything else must already be
    a JSON scalar (a new field of another type fails here, not silently)."""

    if name == "colour_declaration":
        return None if value is None else value.to_dict()
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
    if name == "loop_orders" and isinstance(value, list):
        return tuple(dict(orders) for orders in value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def request_record(request: Any, correlators: Any = None) -> dict[str, object]:
    """The JSON record of a ``SliceRequest`` (every dataclass field, so a new
    request field is recorded and restored without an edit here) and its
    resolved correlator declarations: the artifact's generation recipe."""

    from dataclasses import fields

    record: dict[str, object] = {
        item.name: _encode_field(item.name, getattr(request, item.name))
        for item in fields(request)
    }
    record["correlators"] = None if correlators is None else correlators.to_json_dict()
    return record


def request_from_record(record: Mapping[str, Any]) -> Any:
    """The ``SliceRequest`` of a request record, which must name exactly the
    request's fields (a missing field would silently take its default)."""

    from dataclasses import fields

    from .assemble import SliceRequest

    known = {item.name for item in fields(SliceRequest)}
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


def _record_key(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def model_digest(payload: Mapping[str, Any]) -> str:
    """Digest of a compiled-model payload (``CompiledModel.to_dict``)."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------- plan file


def plan_digest(plan_json: str) -> str:
    return hashlib.sha256(plan_json.encode()).hexdigest()


def write_plan(plan_json: str, destination: Path) -> str:
    """Write the plan v4 PACBIN; returns the SHA-256 of the plan JSON."""

    payload = zlib.compress(plan_json.encode(), 6)
    staged = destination.with_name(destination.name + ".member")
    staged.write_bytes(payload)
    try:
        write_pacbin(
            destination,
            (
                PacbinMemberSource(
                    PLAN_MEMBER, PacbinMemberKind.LOOP_RUNTIME_PLAN, staged
                ),
            ),
        )
    finally:
        staged.unlink()
    return plan_digest(plan_json)


def read_plan(path: Path) -> str:
    """The plan v4 JSON of a ``loop-runtime.pacbin`` (payload digests checked)."""

    with PacbinReader.open(path) as reader:
        member = reader.member(PLAN_MEMBER)
        if member.kind != PacbinMemberKind.LOOP_RUNTIME_PLAN:
            raise ArtifactError("loop runtime member is not a plan v4")
        compressed = reader.read_member(PLAN_MEMBER, length=member.length)
    return zlib.decompress(compressed).decode()


# --------------------------------------------------------------------- generation


@dataclass
class GeneratedProcess:
    """One process generated through v2: its observables (and slice), plan
    and public catalogue."""

    observables: Any
    exported: Any
    request: Mapping[str, Any]
    plan_json: str

    @property
    def slice(self) -> Any:
        return self.observables.slice

    @property
    def plan_sha256(self) -> str:
        return plan_digest(self.plan_json)

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


def _order_text(order: Orders) -> str:
    return " ".join(f"{name}^{power}" for name, power in canonical_order(order))


def generate_process(
    compiled: Any, request: Any, native: Any, correlators: Any = None
) -> GeneratedProcess:
    """Generate one process through v2 with its declared observables and
    remember its generation objects.

    Without colour histories or spin classes this is the inclusive slice
    (``LoopSlice.generate``, colour-pruned at LC/NLC for the inclusive table);
    with them, ``ObservableSlice.generate`` (full tree kernels for spin
    sources; LC/NLC pruning keeps every skeleton live in the declared
    histories' complete-family tables). The stored plan is the inclusive plan
    of that slice; correlated and resolved column plans are exported from the
    same slice at run time.
    """

    from pyamplicol.correlators import CorrelatorConfig

    from .assemble import LoopSlice
    from .observables import ObservableSlice
    from .plan_export import export_plan

    declared = correlators is not None and bool(
        correlators.color_correlations
        or correlators.spin_correlations
        or correlators.all_color_through_order is not None
    )
    if declared:
        observables = ObservableSlice.generate(
            compiled, request, native, correlators=correlators
        )
    else:
        observables = ObservableSlice.of(LoopSlice.generate(compiled, request, native))
    # Schedule F (the forward oracle) is not stored: the runtime dresses by
    # schedule B, and F's section is over half of the plan (gg -> gggg: 52 of
    # 98 MB of JSON).
    exported = export_plan(observables.slice, forward=False)
    resolved = CorrelatorConfig(
        tuple(observables.histories), tuple(observables.spin_classes)
    )
    record = request_record(request, resolved if declared else None)
    generated = GeneratedProcess(observables, exported, record, exported.json())
    _remember(model_digest(compiled.to_dict()), generated)
    return generated


def _remember(model: str, generated: GeneratedProcess) -> None:
    key = (model, _record_key(generated.request), generated.plan_sha256)
    with _slices_lock:
        _slices[key] = (generated.observables, generated.exported)
        _slices.move_to_end(key)
        while len(_slices) > _SLICE_CACHE_SIZE:
            _slices.popitem(last=False)


def _recall(model: str, record: Mapping[str, Any], plan: str) -> Any:
    key = (model, _record_key(record), plan)
    with _slices_lock:
        found = _slices.get(key)
        if found is not None:
            _slices.move_to_end(key)
        return found


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
    normalized = " ".join(process.split())
    for record in records:
        if record["id"] == process:
            return record, None
    for record in records:
        if " ".join(str(record["expression"]).split()) == normalized:
            return record, None
    from pyamplicol.artifacts.process_groups import read_process_groups

    groups = read_process_groups(manifest.root)
    for representative, group in (groups.groups if groups else {}).items():
        for member in group.members:
            if member.process_id != representative and (
                member.process_id == process
                or " ".join(member.expression.split()) == normalized
            ):
                if str(member.basis) != "exact-loop-flavor":
                    break
                for record in records:
                    if record["id"] == representative:
                        return record, member
    raise EvaluationError(f"artifact has no process {process!r}")


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
    read on the representative (``observables.admit_member`` re-proves the
    substitution at load).
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
        from pyamplicol.generation.loop_artifact import (
            LOOP_PHYSICS_KIND,
            LOOP_RUNTIME_KIND,
            LOOP_V2_EXECUTION_SCHEMA,
            LOOP_V2_POINT_INPUTS,
        )
        from pyamplicol.loops.native import load_loop_native
        from pyamplicol.models.loading import CompiledModel
        from pyamplicol.runtime.virtual import _NO_RESCUE, V2ObservablesRuntime

        from .observables import admit_member
        from .plan_export import NativeExecutor

        if stability is not None:
            raise CompatibilityError(_NO_RESCUE)
        self._default_r2 = _r2(r2_strategy)
        self.default_precision = 16 if precision is None else _digits(precision)
        self.default_arithmetic = _arithmetic(arithmetic) or (
            "double" if self.default_precision == 16 else "arbitrary"
        )
        root = Path(path)
        if root.name == MANIFEST_NAME and root.is_file():
            root = root.parent
        started = time.perf_counter()
        manifest = load_manifest(root, verify_payloads=True)
        selected, group_member = _select(manifest, process)
        process_id = str(selected["id"])
        execution = _json_payload(manifest, f"processes/{process_id}/execution.json")
        physics = _json_payload(manifest, str(selected["physics_path"]))
        if (
            execution.get("kind") != LOOP_RUNTIME_KIND
            or execution.get("schema_version") != LOOP_V2_EXECUTION_SCHEMA
            or execution.get("engine") != ENGINE
            or execution.get("point_inputs") != LOOP_V2_POINT_INPUTS
            or physics.get("kind") != LOOP_PHYSICS_KIND
            or physics.get("process_id") != process_id
        ):
            raise ArtifactError(f"process {process_id!r} is not a v2 virtual process")
        native = load_loop_native()
        # The native loader checks the plan member against the declared
        # digest and validates the program against the declarations once;
        # the regenerated plan is then compared by that digest.
        admission = admit(native, root, process_id)
        plan_sha256 = str(admission["plan_sha256"])
        model_payload = _json_payload(manifest, "model/compiled-model.json")
        model = model_digest(model_payload)
        request = execution["request"]
        found = _recall(model, request, plan_sha256)
        if found is None:
            compiled = CompiledModel.from_dict(
                model_payload, validate_fingerprint=False
            )
            generated = generate_process(
                compiled,
                request_from_record(request),
                native,
                correlators_from_record(request),
            )
            if generated.plan_sha256 != plan_sha256:
                raise ArtifactError(
                    "the plan v4 regenerated from this artifact differs from the "
                    "stored one: it was written by a different pyAmpliCol build; "
                    "regenerate the artifact"
                )
            found = generated.observables, generated.exported
        observables, exported = found
        member = None
        if group_member is not None:
            try:
                member = admit_member(observables.slice, group_member.expression)
            except ValueError as error:
                raise ArtifactError(
                    f"flavour member {group_member.process_id!r} is not admitted "
                    f"by its representative: {error}"
                ) from error
            if member.alias or tuple(member.legs) != tuple(group_member.permutation):
                raise ArtifactError(
                    f"flavour member {group_member.process_id!r} is not an "
                    "identity-port member of its representative"
                )
        self._generated = observables, exported
        self._member = member
        self._group_member = group_member
        self._executor = NativeExecutor(observables.slice, exported=exported)
        self._parameters = _real_parameters(model_parameters)
        self._observables = V2ObservablesRuntime(
            observables,
            member=member,
            arithmetic=self.default_arithmetic,
            precision=self.default_precision,
        )
        self._observables.set_model_parameters(self._parameters)
        self._load_seconds = time.perf_counter() - started
        self._admission = admission
        self._manifest = manifest
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
        self._lock = RLock()

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
        pdgs = (
            [int(leg["pdg"]) for leg in self._physics["state_legs"]]
            if member is None
            else list(member.pdgs)
        )
        return ProcessPhysics(
            process_id=self._public_id,
            process=self._expression if member is None else str(member.expression),
            color_accuracy=accuracy,
            helicity_coverage="all",
            color_coverage="all",
            color_kind="contracted-color",
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
            contracted_color_components=(
                ContractedColorComponent(
                    accuracy, 0, f"{accuracy} colour interference"
                ),
            ),
            born_anchored_color_components=(),
            contribution_kind="virtual",
            reduction=PhysicsReduction("contracted-color", ()),
            model_parameters=(),
            selector_capabilities=(),
        )

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
                    "plan_bytes": self._admission["plan_bytes"],
                    "point_inputs": self._execution["point_inputs"],
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
                    "poles": "computed unless poles=False (finite-only)",
                    "stability_checks": (
                        "rotations (count or exact rotations) and the "
                        "reduction-direction check; none by default"
                    ),
                    "rescue_available": False,
                    "accuracy_established": False,
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
        poles: bool | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
        on_failure: Literal["raise", "return"] = "raise",
    ) -> VirtualEvaluation:
        from pyamplicol.runtime.virtual import _inputs, v2_options

        poles, checks = v2_options(
            helicities=helicities,
            color_flows=color_flows,
            r2_strategy=r2_strategy,
            stability=stability,
            poles=poles,
            stability_checks=stability_checks,
            on_failure=on_failure,
        )
        r2 = self._default_r2 if r2_strategy is None else _r2(r2_strategy)
        carrier, setting = self._carrier(arithmetic, precision)
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        if self._member is not None:
            points = [self._member.point(point) for point in points]
        with self._lock:
            parameters = dict(self._parameters)
            results, failures = self._run(
                points, scales, parameters, setting, poles, checks
            )
        if failures and on_failure == "raise":
            index = min(failures)
            raise EvaluationError(
                f"v2 virtual evaluation failed at point {index}: "
                + "; ".join(failures[index])
            )
        return self._result(results, failures, setting, carrier, r2, poles, checks)

    def _run(
        self,
        points: list[list[list[str]]],
        scales: list[str],
        parameters: dict[str, str],
        setting: Any,
        poles: bool,
        checks: VirtualStabilityChecks,
    ) -> tuple[list[Any], dict[int, list[str]]]:
        """One batch through the native executor: the per-point results
        (``None`` for a failed point) and the failures (a failed point fails
        alone, ``plan_export.FailedPoints``)."""

        from .plan_export import FailedPoints

        if not points:
            return [], {}
        try:
            return (
                list(
                    self._executor.evaluate(
                        points,
                        scales,
                        parameters=parameters,
                        precision=setting,
                        poles=poles,
                        stability=checks.to_checks() if checks.enabled else None,
                    )
                ),
                {},
            )
        except FailedPoints as failed:
            return list(failed.results), dict(failed.failures)

    def _result(
        self,
        results: list[Any],
        failures: Mapping[int, Sequence[str]],
        setting: Any,
        carrier: str,
        r2: R2Strategy,
        poles: bool,
        checks: VirtualStabilityChecks,
    ) -> VirtualEvaluation:
        from pyamplicol.runtime.virtual import v2_public_estimate

        bits = _precision_bits(carrier, setting.digits)
        arithmetic = cast(Literal["double", "double-double", "arbitrary"], carrier)
        contraction = VirtualColorContraction(
            cast(
                Literal["auto", "direct"],
                self._execution["request"]["colour_contraction"],
            ),
            "direct",
            "native seeds contract the exact colour weights K[c, b] directly",
        )
        born_values: list[tuple[Any, ...] | None] = []
        values: list[tuple[LaurentCoefficients, ...] | None] = []
        diagnostics: list[VirtualLaneDiagnostic] = []
        estimates: list[VirtualStabilityEstimate | None] = []
        for index, result in enumerate(results):
            if result is None:
                reason = "; ".join(failures.get(index, ["failed"]))
                born_values.append(None)
                values.append(None)
                estimates.append(None)
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
                    if poles
                    else LaurentCoefficients(
                        virtual[sector.order].finite, *_ABSENT_POLES
                    )
                    for sector in self._sectors
                )
            )
            estimates.append(v2_public_estimate(result.stability, poles))
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

        return self._observables.evaluate_resolved(
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
        return self._observables.evaluate_correlated(
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
        return self._observables.evaluate_correlated_many(
            momenta,
            requests,
            **self._observable_options(precision, arithmetic, options),
        )

    def available_color_correlations(self) -> tuple[Any, ...]:
        return self._observables.available_color_correlations()

    def set_spin_correlation_vectors(self, vectors: Any) -> None:
        self._observables.set_spin_correlation_vectors(vectors)

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        parameters = _real_parameters(mapping)
        with self._lock:
            self._parameters.update(parameters)
            self._observables.set_model_parameters(parameters)

    def clear(self) -> None:
        """Drop the native executors (and their master caches)."""
        with self._lock:
            from .plan_export import NativeExecutor

            observables, exported = self._generated
            self._executor = NativeExecutor(observables.slice, exported=exported)
            self._observables.clear()

    def mute_warnings(self) -> None:
        """No effect: v2 runtimes emit no runtime warnings."""

    def unmute_warnings(self) -> None:
        """No effect: v2 runtimes emit no runtime warnings."""


def _zero(carrier: str) -> Any:
    return 0.0 if carrier == "double" else Decimal(0)


def _r2(value: str | None) -> R2Strategy:
    strategy = R2Strategy.AUTO if value is None else R2Strategy(value)
    if strategy is R2Strategy.DIRECT:
        raise CompatibilityError("v2 virtual artifacts carry model-supplied R2 only")
    return strategy


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
