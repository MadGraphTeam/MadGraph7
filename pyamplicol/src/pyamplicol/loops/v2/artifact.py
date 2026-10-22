# SPDX-License-Identifier: 0BSD
"""Plan v4 loop artifacts and their public runtime (design §12, WP2.6).

A v2 virtual process is published in the ordinary schema-v3 artifact
envelope under the runtime capability ``rusticol.loop-runtime.v4``. Its
``loop-runtime.pacbin`` holds one member of kind ``LOOP_RUNTIME_PLAN``: the
plan v4 program of ``plan_export`` (zlib-compressed JSON), which the native
``V2Executor`` validates and runs. The execution record keeps the slice
request (process, coupling orders, corrections, fixed parameters, colour
accuracy and continuation, seed contraction) and the plan's SHA-256.

Trees, Born amplitudes and counterterm scalars are still evaluated per point
in Python (design §5.5), from the generation objects of ``LoopSlice``. Until
the loop-request mode of the Rust recurrence (WP3.1) moves them native, the
runtime rebuilds those objects at load time from the artifact's own compiled
model and the saved request (the recursive generator, no diagram is listed)
and requires the regenerated plan to equal the saved one byte for byte: the
Python input tables (block rows, Born components, kernel constants) must
match the stored program's numbering, or the native executor would read
wrong inputs. The same process reuses generation objects already built in
this interpreter (generation and its staged validation).

Evaluation returns the public ``VirtualEvaluation``: Born per squared Born
order, the complete virtual per squared coupling-order sector with its three
Laurent coefficients (finite only on request), and, when stability checks
were asked for, one ``VirtualStabilityEstimate`` per point. Escalation of the
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

ContractionPolicy = Literal["auto", "direct", "fft"]


# --------------------------------------------------------------------- request


def request_record(request: Any) -> dict[str, object]:
    """The JSON record of a ``SliceRequest`` (the artifact's generation recipe)."""

    declaration = request.colour_declaration
    return {
        "process": request.process,
        "born_orders": dict(sorted(request.born_orders.items())),
        "loop_orders": dict(sorted(request.loop_orders.items())),
        "corrections": list(request.corrections),
        "fixed_parameters": dict(sorted(request.fixed_parameters.items())),
        "ignored_particle_ids": list(request.ignored_particle_ids),
        "accuracy": request.accuracy,
        "colour_declaration": None if declaration is None else declaration.to_dict(),
        "scale_parameter": request.scale_parameter,
        "alpha_s_parameter": request.alpha_s_parameter,
        "compress_colour": request.compress_colour,
        "colour_contraction": request.colour_contraction,
    }


def request_from_record(record: Mapping[str, Any]) -> Any:
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    from .assemble import SliceRequest

    declaration = record["colour_declaration"]
    return SliceRequest(
        record["process"],
        dict(record["born_orders"]),
        dict(record["loop_orders"]),
        corrections=tuple(record["corrections"]),
        fixed_parameters=dict(record["fixed_parameters"]),
        ignored_particle_ids=tuple(record["ignored_particle_ids"]),
        accuracy=record["accuracy"],
        colour_declaration=None
        if declaration is None
        else LoopColorDeclaration.from_dict(declaration),
        scale_parameter=record["scale_parameter"],
        alpha_s_parameter=record["alpha_s_parameter"],
        compress_colour=record["compress_colour"],
        colour_contraction=record["colour_contraction"],
    )


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
    """One process generated through v2: its slice, plan and public catalogue."""

    slice: Any
    exported: Any
    request: Mapping[str, Any]
    plan_json: str

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


def _order_text(order: Orders) -> str:
    return " ".join(f"{name}^{power}" for name, power in canonical_order(order))


def generate_process(compiled: Any, request: Any, native: Any) -> GeneratedProcess:
    """Generate one process through v2 and remember its generation objects."""

    from .assemble import LoopSlice
    from .plan_export import export_plan

    slice_ = LoopSlice.generate(compiled, request, native)
    exported = export_plan(slice_)
    record = request_record(request)
    generated = GeneratedProcess(slice_, exported, record, exported.json())
    _remember(model_digest(compiled.to_dict()), generated)
    return generated


def _remember(model: str, generated: GeneratedProcess) -> None:
    key = (model, _record_key(generated.request), generated.plan_sha256)
    with _slices_lock:
        _slices[key] = (generated.slice, generated.exported)
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


def _select(manifest: Any, process: str | None) -> Mapping[str, Any]:
    records = cast(tuple[Mapping[str, Any], ...], manifest.processes)
    if process is None:
        if manifest.default_process_id is not None:
            process = manifest.default_process_id
        elif len(records) == 1:
            return records[0]
        else:
            raise EvaluationError(
                "this artifact holds several processes; select one with process="
            )
    for record in records:
        if record["id"] == process:
            return record
    normalized = " ".join(process.split())
    for record in records:
        if " ".join(str(record["expression"]).split()) == normalized:
            return record
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


def _precision_bits(digits: int | None) -> int:
    return 53 if digits is None else math.ceil(digits * math.log2(10))


class V2VirtualRuntime:
    """Runtime of one v2 virtual process (the public ``Runtime`` backend)."""

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
        )
        from pyamplicol.loops.native import load_loop_native
        from pyamplicol.models.loading import CompiledModel

        from .plan_export import NativeExecutor

        if stability is not None:
            raise CompatibilityError(_NO_RESCUE)
        self._default_r2 = _r2(r2_strategy)
        self.default_precision = 16 if precision is None else _digits(precision)
        self.default_arithmetic = _arithmetic(arithmetic)
        root = Path(path)
        if root.name == MANIFEST_NAME and root.is_file():
            root = root.parent
        started = time.perf_counter()
        manifest = load_manifest(root, verify_payloads=True)
        selected = _select(manifest, process)
        process_id = str(selected["id"])
        execution = _json_payload(manifest, f"processes/{process_id}/execution.json")
        physics = _json_payload(manifest, str(selected["physics_path"]))
        if (
            execution.get("kind") != LOOP_RUNTIME_KIND
            or execution.get("schema_version") != LOOP_V2_EXECUTION_SCHEMA
            or execution.get("engine") != ENGINE
            or physics.get("kind") != LOOP_PHYSICS_KIND
            or physics.get("process_id") != process_id
        ):
            raise ArtifactError(f"process {process_id!r} is not a v2 virtual process")
        native = load_loop_native()
        admission = admit(native, root, process_id)
        if admission["plan_sha256"] != execution["plan_sha256"]:
            raise ArtifactError("the native loader admitted a different plan v4")
        plan_json = read_plan(
            root / f"processes/{process_id}/{execution['runtime_path']}"
        )
        if plan_digest(plan_json) != execution["plan_sha256"]:
            raise ArtifactError("the stored plan v4 differs from its declared digest")
        model_payload = _json_payload(manifest, "model/compiled-model.json")
        model = model_digest(model_payload)
        request = execution["request"]
        found = _recall(model, request, execution["plan_sha256"])
        if found is None:
            compiled = CompiledModel.from_dict(
                model_payload, validate_fingerprint=False
            )
            generated = generate_process(compiled, request_from_record(request), native)
            if generated.plan_json != plan_json:
                raise ArtifactError(
                    "the plan v4 regenerated from this artifact differs from the "
                    "stored one: it was written by a different pyAmpliCol build; "
                    "regenerate the artifact"
                )
            found = generated.slice, generated.exported
        slice_, exported = found
        self._generated = slice_, exported
        self._executor = NativeExecutor(slice_, exported=exported)
        # Constructing the f64 executor validates the stored program natively.
        self._executor.native("double")
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
        self._parameters = _real_parameters(model_parameters)
        self._muted = bool(mute_warnings)

    # -- identity and metadata ------------------------------------------------

    @property
    def artifact_id(self) -> str:
        return self._manifest.artifact_id

    @property
    def representative_process_key(self) -> str:
        return self._process_id

    @property
    def external_count(self) -> int:
        return len(self._physics["state_legs"])

    @property
    def external_permutation(self) -> tuple[int, ...]:
        return tuple(range(self.external_count))

    @property
    def physics(self) -> ProcessPhysics:
        accuracy = cast(Literal["lc", "nlc", "full"], self._metadata["color_accuracy"])
        return ProcessPhysics(
            process_id=self._process_id,
            process=self._expression,
            color_accuracy=accuracy,
            helicity_coverage="all",
            color_coverage="all",
            color_kind="contracted-color",
            structural_zero_helicity_count=0,
            external_particles=tuple(
                ExternalParticle(
                    index=i,
                    label=i + 1,
                    name=str(leg["pdg"]),
                    pdg_id=leg["pdg"],
                    state="incoming" if leg["incoming"] else "outgoing",
                    momentum_slot=i,
                )
                for i, leg in enumerate(self._physics["state_legs"])
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
                    "process_id": self._process_id,
                    "process": self._expression,
                    "engine": ENGINE,
                    **self._metadata,
                    "plan_sha256": self._execution["plan_sha256"],
                    "plan_bytes": self._admission["plan_bytes"],
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
        from pyamplicol.runtime.virtual import _inputs

        from .assemble import Precision

        if on_failure not in ("raise", "return"):
            raise ValueError("on_failure must be 'raise' or 'return'")
        if helicities or color_flows:
            raise CompatibilityError(
                "helicity and colour selectors of v2 virtuals await the "
                "observables package (WP2.5)"
            )
        if stability is not None:
            raise CompatibilityError(_NO_RESCUE)
        r2 = self._default_r2 if r2_strategy is None else _r2(r2_strategy)
        poles = True if poles is None else poles
        if type(poles) is not bool:
            raise TypeError("poles must be a boolean")
        checks = stability_checks or VirtualStabilityChecks()
        if not isinstance(checks, VirtualStabilityChecks):
            raise TypeError("stability_checks requires VirtualStabilityChecks")
        digits = self.default_precision if precision is None else _digits(precision)
        carrier = _arithmetic(arithmetic) or self.default_arithmetic
        carrier = carrier or ("double" if digits == 16 else "arbitrary")
        setting = (
            Precision.double()
            if carrier == "double"
            else Precision("arbitrary", digits)
        )
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        with self._lock:
            parameters = dict(self._parameters)
            results = self._run(points, scales, parameters, setting, poles, checks)
        failed = [
            index for index, result in enumerate(results) if isinstance(result, str)
        ]
        if failed and on_failure == "raise":
            raise EvaluationError(
                f"v2 virtual evaluation failed at point {failed[0]}: "
                f"{results[failed[0]]}"
            )
        return self._result(
            results,
            setting,
            cast(Literal["double", "arbitrary"], carrier),
            r2,
            poles,
            checks,
        )

    def _run(
        self,
        points: list[list[list[str]]],
        scales: list[str],
        parameters: dict[str, str],
        setting: Any,
        poles: bool,
        checks: VirtualStabilityChecks,
    ) -> list[Any]:
        """One batch through the native executor; on a failed lane, every point
        alone, so a failure stays with its point. A failed point is its error
        text."""

        def batch(
            selected_points: list[list[list[str]]], selected_scales: list[str]
        ) -> list[Any]:
            return self._executor.evaluate(
                selected_points,
                selected_scales,
                parameters=parameters,
                precision=setting,
                poles=poles,
                stability=checks.to_checks(),
            )

        if not points:
            return []
        try:
            return list(batch(points, scales))
        except ArithmeticError as error:
            if len(points) == 1:
                return [str(error)]
        out: list[Any] = []
        for point, scale in zip(points, scales, strict=True):
            try:
                out.extend(batch([point], [scale]))
            except ArithmeticError as error:
                out.append(str(error))
        return out

    def _result(
        self,
        results: list[Any],
        setting: Any,
        carrier: Literal["double", "arbitrary"],
        r2: R2Strategy,
        poles: bool,
        checks: VirtualStabilityChecks,
    ) -> VirtualEvaluation:
        bits = _precision_bits(None if carrier == "double" else setting.working_digits)
        contraction = VirtualColorContraction(
            _requested_contraction(self._execution["request"]["colour_contraction"]),
            "direct",
            "native seeds contract the exact colour weights K[c, b] directly",
        )
        born_values: list[tuple[Any, ...] | None] = []
        values: list[tuple[LaurentCoefficients, ...] | None] = []
        diagnostics: list[VirtualLaneDiagnostic] = []
        estimates: list[VirtualStabilityEstimate | None] = []
        for index, result in enumerate(results):
            if isinstance(result, str):
                born_values.append(None)
                values.append(None)
                estimates.append(None)
                diagnostics.append(
                    VirtualLaneDiagnostic(
                        "failed",
                        carrier,
                        bits,
                        False,
                        result,
                        failures=(
                            VirtualFailureDiagnostic(
                                index, None, None, None, "v2-evaluation", result
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
            estimates.append(_estimate(result.stability, poles))
            diagnostics.append(
                VirtualLaneDiagnostic(
                    "success",
                    carrier,
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

    def evaluate_resolved(self, momenta: Momenta, **kwargs: Any) -> Any:
        raise CompatibilityError(_AWAITS_OBSERVABLES.format("resolved output"))

    def evaluate_correlated(self, momenta: Momenta, **kwargs: Any) -> Any:
        raise CompatibilityError(_AWAITS_OBSERVABLES.format("colour/spin correlations"))

    def evaluate_correlated_many(
        self, momenta: Momenta, requests: Any, **kwargs: Any
    ) -> Any:
        raise CompatibilityError(_AWAITS_OBSERVABLES.format("colour/spin correlations"))

    def available_color_correlations(self) -> tuple[Any, ...]:
        raise CompatibilityError(_AWAITS_OBSERVABLES.format("colour/spin correlations"))

    def set_spin_correlation_vectors(self, vectors: Any) -> None:
        raise CompatibilityError(_AWAITS_OBSERVABLES.format("colour/spin correlations"))

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        parameters = _real_parameters(mapping)
        with self._lock:
            self._parameters.update(parameters)

    def clear(self) -> None:
        """Drop the native executors (and their master caches)."""
        with self._lock:
            from .plan_export import NativeExecutor

            slice_, exported = self._generated
            self._executor = NativeExecutor(slice_, exported=exported)

    def mute_warnings(self) -> None:
        self._muted = True

    def unmute_warnings(self) -> None:
        self._muted = False


_NO_RESCUE = (
    "v2 loop runtimes return stability estimates (stability_checks) and leave "
    "precision escalation to the caller; a VirtualStabilityPolicy rescue is "
    "available with the legacy engine only"
)
_AWAITS_OBSERVABLES = (
    "{} of v2 virtuals await the observables package (WP2.5); generate with "
    "loop.engine='legacy' for them"
)


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
    if value == "double-double":
        raise CompatibilityError(
            "double-double v2 evaluations await the precision package (WP2.3); "
            "use arithmetic='arbitrary' with precision=32"
        )
    return value


_CONTRACTIONS: dict[str, Literal["auto", "direct", "symmetric-group-fft"]] = {
    "auto": "auto",
    "direct": "direct",
    "fft": "symmetric-group-fft",
}
# Finite-only Laurent coefficients carry explicitly absent poles.
_ABSENT_POLES = cast(tuple[Any, Any], (None, None))


def _requested_contraction(
    policy: str,
) -> Literal["auto", "direct", "symmetric-group-fft"]:
    return _CONTRACTIONS[policy]


def _estimate(estimate: Any, poles: bool) -> VirtualStabilityEstimate | None:
    if estimate is None:
        return None
    coefficients = tuple(float(x) for x in estimate.coefficients)
    return VirtualStabilityEstimate(
        float(estimate.relative),
        coefficients[0],
        coefficients[1] if poles else None,
        coefficients[2] if poles else None,
        tuple((name, float(value)) for name, value in estimate.checks.items()),
    )
