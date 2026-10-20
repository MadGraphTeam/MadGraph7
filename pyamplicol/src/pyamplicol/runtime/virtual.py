# SPDX-License-Identifier: 0BSD
"""Typed Python transport for the native, source-owned virtual runtime.

Python formats inputs and immutable outputs. The native coordinator evaluates
one complete batch, including source parameters, counterterms and reduction.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Literal, TypeVar, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    EvaluationError,
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
    LaurentCoefficients,
    ResolvedVirtualEvaluation,
    VirtualArithmeticSpec,
    VirtualColorContraction,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualScalar,
    VirtualStabilityPolicy,
    VirtualVerificationAttempt,
    VirtualVerificationDiagnostic,
)
from pyamplicol.config import R2Strategy
from pyamplicol.correlators import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatedValue,
    _emission_from_json,
)
from pyamplicol.loops.correlation_labels import public_color_history
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.runtime.correlated_exact import (
    _copy_spin_vectors,
    _prepare_spin_vectors,
)


def _verification_diagnostic(value: dict[str, Any]) -> VirtualVerificationDiagnostic:
    return VirtualVerificationDiagnostic(
        point_id=value["point_id"],
        status=value["status"],
        accuracy_established=value["accuracy_established"],
        agreeing_bits=tuple(value["agreeing_bits"]) if value["agreeing_bits"] else None,
        attempts=tuple(
            VirtualVerificationAttempt(
                VirtualArithmeticSpec(**attempt["arithmetic"]),
                attempt["precision_bits"],
                attempt["numerical_success"],
                attempt["compared_precision_bits"],
                attempt["agreement"],
                attempt["conversion_agreement"],
                tuple(VirtualFailureDiagnostic(**item) for item in attempt["failures"]),
                VirtualColorContraction(**attempt["color_contraction"]),
            )
            for attempt in value["attempts"]
        ),
        detail=value["detail"],
    )


def _stability_input(
    policy: VirtualStabilityPolicy | None,
) -> tuple[list[tuple[str, int | None]], str, str, bool] | None:
    if policy is None:
        return None
    if not isinstance(policy, VirtualStabilityPolicy):
        raise TypeError("stability requires a VirtualStabilityPolicy")
    return (
        [(item.arithmetic, item.precision) for item in policy.higher],
        policy.relative_tolerance,
        policy.absolute_tolerance,
        policy.verify_successful,
    )


def _number(value: object, description: str) -> str:
    """Validate a real decimal input without a binary64 round trip."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral, Real, Decimal)):
        raise TypeError(f"{description} must be a real number or decimal string")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{description} is not a decimal number") from error
    if not decimal.is_finite():
        raise ValueError(f"{description} must be finite")
    return str(decimal)


def _parameters(mapping: ModelParameters | None) -> dict[str, list[str]] | None:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise TypeError("model parameters must be a mapping")
    result = {}
    for name, value in mapping.items():
        real: object
        imaginary: object
        if not isinstance(name, str) or not name:
            raise TypeError("model parameter names must be nonempty strings")
        if isinstance(value, complex):
            real, imaginary = value.real, value.imag
        elif isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"parameter {name!r} needs a real/imaginary pair")
            real, imaginary = value
        else:
            real, imaginary = value, 0
        result[name] = [
            _number(real, f"parameter {name!r} real part"),
            _number(imaginary, f"parameter {name!r} imaginary part"),
        ]
    return result


def _inputs(
    momenta: Momenta, mu_r2: object, legs: int
) -> tuple[list[list[list[str]]], list[str]]:
    points = []
    for point in momenta:
        if len(point) != legs or any(len(vector) != 4 for vector in point):
            raise ValueError(f"momenta require {legs} four-vectors per point")
        points.append(
            [
                [_number(value, "momentum component") for value in vector]
                for vector in point
            ]
        )
    if mu_r2 is None:
        raise CompatibilityError(
            "virtual evaluation currently requires explicit positive mu_r2 "
            "(the renormalization scale squared)"
        )
    if isinstance(mu_r2, Mapping):
        raise TypeError("mu_r2 must be a scalar or ordered sequence, not a mapping")
    if not isinstance(mu_r2, (str, Integral, Real, Decimal)):
        try:
            scales: list[object] = list(cast(Iterable[object], mu_r2))
        except TypeError as error:
            raise TypeError(
                "mu_r2 must be a real scalar or one-dimensional sequence"
            ) from error
        if len(scales) != len(points):
            raise ValueError("mu_r2 must have one entry per point")
    else:
        # Validate even for an empty batch.
        scalar = _number(mu_r2, "mu_r2")
        if Decimal(scalar) <= 0:
            raise ValueError("mu_r2 must be positive")
        scales = [scalar] * len(points)
    formatted_scales = [_number(value, "mu_r2") for value in scales]
    if any(Decimal(value) <= 0 for value in formatted_scales):
        raise ValueError("mu_r2 must be positive")
    return points, formatted_scales


def _invoke(
    module: Any, operation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    try:
        return operation(*args, **kwargs)
    except Exception as error:
        for name, public in (
            ("VirtualArtifactError", ArtifactError),
            ("VirtualCompatibilityError", CompatibilityError),
            ("VirtualEvaluationError", EvaluationError),
        ):
            native = getattr(module, name, None)
            if isinstance(native, type) and isinstance(error, native):
                raise public(str(error)) from error
        raise


def _rows(
    batch: Any, count: int, width: int, arithmetic: str, *, correlated: bool = False
) -> tuple[tuple[VirtualScalar, ...], ...]:
    if (
        batch.rows != count
        or batch.columns != width
        or list(batch.row_ids) != list(range(count))
    ):
        raise EvaluationError(
            "native virtual output has inconsistent shape or lane IDs"
        )
    flat = batch.to_decimal()
    if len(flat) != count * width:
        raise EvaluationError("native virtual output has inconsistent scalar count")
    values: list[VirtualScalar] = []
    for real, imaginary in flat:
        real, imaginary = Decimal(real), Decimal(imaginary)
        if not real.is_finite() or not imaginary.is_finite():
            raise EvaluationError("virtual output must be finite")
        if correlated:
            values.append(
                complex(float(real), float(imaginary))
                if arithmetic == "double"
                else CorrelatedValue(real, imaginary)
            )
        else:
            if imaginary != 0:
                raise EvaluationError("ordinary virtual output must be finite and real")
            values.append(float(real) if arithmetic == "double" else real)
    return tuple(
        tuple(values[start : start + width]) for start in range(0, len(values), width)
    )


_PointValue = TypeVar("_PointValue")


def _scatter_points(
    values: tuple[_PointValue, ...], point_ids: tuple[int, ...], count: int
) -> tuple[_PointValue | None, ...]:
    """Assemble output positions; numerical execution remains one native batch."""
    result: list[_PointValue | None] = [None] * count
    for point, value in zip(point_ids, values, strict=True):
        result[point] = value
    return tuple(result)


class VirtualRuntimeBackend:
    """One native handle, with atomic exact rebinding when arithmetic changes."""

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
        if precision is not None and (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise ValueError("precision must be a positive decimal digit count")
        if arithmetic is not None and arithmetic not in {
            "double",
            "double-double",
            "arbitrary",
        }:
            raise ValueError("unknown virtual arithmetic")
        self.default_precision = precision if precision is not None else 16
        self.default_arithmetic = arithmetic
        policy_input = _stability_input(stability)
        self._default_stability = stability
        self._default_r2 = (
            R2Strategy.AUTO if r2_strategy is None else R2Strategy(r2_strategy)
        )
        initial_arithmetic = arithmetic or (
            "double" if self.default_precision == 16 else "arbitrary"
        )
        module = load_loop_native()
        self._module = module
        self._lock = RLock()
        self._native = _invoke(
            module,
            module.NativeVirtualRuntime,
            str(path),
            process=process,
            parameters=_parameters(model_parameters),
            arithmetic=initial_arithmetic,
            precision=self.default_precision
            if initial_arithmetic == "arbitrary"
            else None,
            r2_strategy=self._default_r2.value,
            **({"stability": policy_input} if policy_input is not None else {}),
        )
        self._metadata: dict[str, Any] = json.loads(self._native.metadata_json())
        self._color_accuracy = cast(
            Literal["lc", "nlc", "full"], self._metadata["physics"]["color_accuracy"]
        )
        partition = self._metadata["plan"].get("born_color_components")
        self._color_ids: tuple[str, ...] = (
            tuple(partition["ids"])
            if partition is not None
            else (self._color_accuracy,)
        )
        self._muted = bool(mute_warnings)
        self._last_requested = self._default_r2
        orders = self._metadata["physics"]["orders"]
        self._born_orders = tuple(
            canonical_order(item["order"]) for item in orders["born"]
        )
        self._sectors = tuple(
            VirtualOrderSector(
                canonical_order(item["order"]),
                tuple(
                    BornLoopPair(pair["born"], pair["loop"]) for pair in item["pairs"]
                ),
            )
            for item in orders["virtual"]
        )
        self._spin_vectors: dict[int, object] = {}

    def _spin_classes(self) -> tuple[tuple[int, ...], ...]:
        if not self._metadata["physics"]["correlations"]:
            return ()
        return tuple(
            tuple(item["legs"])
            for item in self._metadata["plan"].get("spin_projections", ()) or ()
        )

    def _saved_color_correlations(self) -> tuple[ColorCorrelator, ...]:
        descriptor = self._metadata["plan"].get("directed_projection")
        if not self._metadata["physics"]["correlations"] or descriptor is None:
            raise CompatibilityError(
                "this selected process has no admitted directed colour histories"
            )
        return tuple(
            ColorCorrelator(
                item["id"],
                tuple(_emission_from_json(step) for step in item["bra"]),
                tuple(_emission_from_json(step) for step in item["ket"]),
            )
            for item in descriptor["histories"]
        )

    def _color_history_aliases(self) -> dict[str, str]:
        histories = self._saved_color_correlations()
        aliases = {item.id: item.id for item in histories}
        born = next((item for item in histories if item.id == "born"), None)
        if born is not None and (born.bra or born.ket):
            raise CompatibilityError(
                "reserved 'born' history must have empty bra and ket"
            )
        if born is None:
            identities = [
                item.id for item in histories if not item.bra and not item.ket
            ]
            if len(identities) == 1:
                aliases["born"] = identities[0]
        return aliases

    def available_color_correlations(self) -> tuple[ColorCorrelator, ...]:
        histories = tuple(map(public_color_history, self._saved_color_correlations()))
        aliases = self._color_history_aliases()
        if "born" in aliases and aliases["born"] != "born":
            return (*histories, ColorCorrelator("born"))
        return histories

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        self.available_color_correlations()
        copied = self._copy_spin_vectors(vectors)
        with self._lock:
            self._spin_vectors = copied

    def _copy_spin_vectors(
        self, vectors: Mapping[int, object] | None, *, point_count: int | None = None
    ) -> dict[int, object]:
        if vectors and not self._spin_classes():
            raise CompatibilityError("this artifact has no saved joint spin sources")
        return _copy_spin_vectors(
            vectors, self._spin_classes(), point_count=point_count
        )

    def evaluate_correlated(
        self, momenta: Momenta, *, color_correlation: str = "born", **kwargs: Any
    ) -> VirtualEvaluation:
        return self.evaluate_correlated_many(
            momenta, {"selected": CorrelatedRequest(color_correlation)}, **kwargs
        )["selected"]

    def evaluate_correlated_many(
        self, momenta: Momenta, requests: Mapping[str, CorrelatedRequest], **kwargs: Any
    ) -> dict[str, VirtualEvaluation]:
        aliases = self._color_history_aliases()
        if not isinstance(requests, Mapping):
            raise TypeError("correlated requests must be a mapping")
        selected = dict(requests)
        with self._lock:
            snapshot = dict(self._spin_vectors)
        assignments: dict[tuple[object, ...], dict[int, list[list[list[str]]]]] = {}
        routes: dict[str, tuple[object, ...]] = {}
        classes = self._spin_classes()
        allowed = tuple(sorted({leg for group in classes for leg in group}))
        for label, request in selected.items():
            if not isinstance(label, str) or not label:
                raise ValueError("correlated request labels must be nonempty strings")
            if not isinstance(request, CorrelatedRequest):
                raise TypeError("correlated requests require CorrelatedRequest records")
            if request.color_correlation not in aliases:
                raise CompatibilityError(
                    "undeclared virtual colour correlation "
                    f"{request.color_correlation!r}"
                )
            vectors = self._copy_spin_vectors(
                snapshot if request.spin_vectors is None else request.spin_vectors,
                point_count=len(momenta),
            )
            prepared = _prepare_spin_vectors(
                vectors, allowed_legs=allowed, point_count=len(momenta)
            )
            # Keep class identity even for an empty batch. Decimal tuples retain
            # exact input values, signed zero and requested literal precision.
            key = (
                tuple(sorted(vectors)),
                tuple(
                    tuple(
                        (leg, tuple((a.as_tuple(), b.as_tuple()) for a, b in wave))
                        for leg, wave in sorted(point.items())
                    )
                    for point in prepared
                ),
            )
            assignments.setdefault(
                key,
                {
                    leg: [
                        [[str(real), str(imag)] for real, imag in point[leg]]
                        for point in prepared
                    ]
                    for leg in sorted(vectors)
                },
            )
            routes[label] = key
        if not selected:
            return {}
        # Each distinct source assignment executes a full native event batch,
        # sharing all requested histories. No Python event-evaluation loop.
        with self._lock:
            results = {
                key: cast(
                    dict[str, VirtualEvaluation],
                    self._evaluate(
                        momenta,
                        resolved=False,
                        directed=True,
                        spin_vectors=vectors,
                        **kwargs,
                    ),
                )
                for key, vectors in assignments.items()
            }
        return {
            label: results[routes[label]][aliases[request.color_correlation]]
            for label, request in selected.items()
        }

    @property
    def artifact_id(self) -> str:
        return cast(str, self._metadata["artifact_id"])

    @property
    def representative_process_key(self) -> str:
        return cast(str, self._metadata["representative_process_id"])

    @property
    def external_count(self) -> int:
        return len(self._metadata["external_pdgs"])

    @property
    def external_permutation(self) -> tuple[int, ...]:
        """The native loader's representative-to-selected leg mapping."""
        return tuple(self._metadata["external_permutation"])

    @property
    def physics(self) -> ProcessPhysics:
        metadata, plan = self._metadata, self._metadata["plan"]
        partition = plan.get("born_color_components")
        kind: Literal["born-anchored-interference", "contracted-color"] = (
            "born-anchored-interference"
            if partition is not None
            else "contracted-color"
        )
        return ProcessPhysics(
            process_id=metadata["process_id"],
            process=metadata["expression"],
            color_accuracy=self._color_accuracy,
            helicity_coverage="all",
            color_coverage="all",
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
                for i, (pdg, leg) in enumerate(
                    zip(metadata["external_pdgs"], plan["state_legs"], strict=True)
                )
            ),
            helicities=tuple(
                HelicityConfiguration(f"h{i}", i, tuple(h), True, False, f"h{i}", 1.0)
                for i, h in enumerate(plan["helicities"])
            ),
            color_flows=(),
            contracted_color_components=()
            if partition is not None
            else (
                ContractedColorComponent(
                    self._color_accuracy,
                    0,
                    f"{self._color_accuracy} colour interference",
                ),
            ),
            born_anchored_color_components=()
            if partition is None
            else tuple(
                BornAnchoredColorComponent(
                    identifier,
                    index,
                    tuple(members),
                    "signed interference anchored to a canonical Born tensor group",
                )
                for index, (identifier, members) in enumerate(
                    zip(partition["ids"], partition["members"], strict=True)
                )
            ),
            contribution_kind="virtual",
            reduction=PhysicsReduction(kind, ()),
            # Evaluator-derived defaults are not decimal literals in the plan.
            # Do not mislabel rounded/current snapshot values as exact defaults.
            model_parameters=(),
            selector_capabilities=(),
        )

    def inspect(self) -> Mapping[str, object]:
        with self._lock:
            metadata = self._metadata
            source = metadata["plan"]["model_binding"]["owner"]
            return deepcopy(
                {
                    "artifact_id": self.artifact_id,
                    "process_id": metadata["process_id"],
                    "process": metadata["expression"],
                    **metadata["physics"],
                    "physics_scope": metadata["physics_scope"],
                    "external_permutation": metadata["external_permutation"],
                    "representative_physics": metadata["representative_physics"],
                    "initial_average": metadata["initial_average"],
                    "color_correlations": [
                        item.to_json_dict()
                        for item in self.available_color_correlations()
                    ]
                    if self._metadata["physics"].get("correlations")
                    else [],
                    "correlation_semantics": (
                        "Bdagger K_BL L + Ldagger K_LB B; "
                        "ordered histories may be complex"
                    ),
                    "spin_correlations": self._spin_classes(),
                    "resolved_color_components": list(self._color_ids)
                    if metadata["physics"].get("resolved")
                    else [],
                    "resolved_normalization": (
                        "each helicity retains the inclusive incoming spin average"
                    ),
                    "resolved_color_semantics": (
                        "signed, potentially overcomplete canonical Born tensor "
                        "groups; "
                        "not sampling probabilities"
                        if metadata["plan"].get("born_color_components") is not None
                        else "one contracted colour component"
                    ),
                    "source_revision": self._native.source_revision,
                    "arithmetic": self._native.arithmetic,
                    "color_contraction": json.loads(
                        self._native.color_contraction_json
                    ),
                    "supported_arithmetics": metadata["physics"]["arithmetic"],
                    "default_arithmetic": self.default_arithmetic,
                    "default_precision": self.default_precision,
                    "stability": asdict(self._default_stability)
                    if self._default_stability is not None
                    else None,
                    "precision_bits": self._native.precision_bits,
                    "requested_r2_strategy": self._last_requested.value,
                    "effective_r2_strategy": self._native.r2_strategy,
                    "r2_selection_reason": (
                        "explicit strategy"
                        if self._last_requested is R2Strategy.MODEL
                        else "only complete eligible strategy"
                    ),
                    "parameter_names": source["external_names"],
                    "parameter_types": source["external_types"],
                    "fixed_parameters": source["fixed_parameters"],
                    "parameter_overrides": self._native.parameter_overrides,
                    "parameter_defaults": (
                        "retained native source evaluator; not materialized as metadata"
                    ),
                    "scale_input": "explicit positive mu_r2, scalar or per point",
                    "rescue_available": True,
                    "rescue_policy": "explicit higher-precision agreement",
                    "rescue_limitation": (
                        "numerical source-binding rescue requires a load-time policy; "
                        "successful source admission alone does not establish accuracy"
                    ),
                    "accuracy_established": False,
                }
            )

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
        on_failure: Literal["raise", "return"] = "raise",
    ) -> VirtualEvaluation:
        return cast(
            VirtualEvaluation,
            self._evaluate(
                momenta,
                helicities=helicities,
                color_flows=color_flows,
                precision=precision,
                arithmetic=arithmetic,
                mu_r2=mu_r2,
                r2_strategy=r2_strategy,
                stability=stability,
                on_failure=on_failure,
                resolved=False,
            ),
        )

    def _evaluate(
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
        on_failure: Literal["raise", "return"] = "raise",
        resolved: bool,
        directed: bool = False,
        spin_vectors: dict[int, list[list[list[str]]]] | None = None,
    ) -> VirtualEvaluation | ResolvedVirtualEvaluation | dict[str, VirtualEvaluation]:
        if on_failure not in {"raise", "return"}:
            raise ValueError("on_failure must be 'raise' or 'return'")
        histories = self._saved_color_correlations() if directed else ()
        if directed and resolved:
            raise CompatibilityError("resolved directed virtual output is unavailable")
        if resolved and not self._metadata["physics"].get("resolved", False):
            raise CompatibilityError(
                "this artifact has no resolved virtual projection kernels"
            )
        if helicities or color_flows:
            raise CompatibilityError("virtual artifact selectors are not yet available")
        precision = self.default_precision if precision is None else precision
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise ValueError("precision must be a positive decimal digit count")
        arithmetic = (
            arithmetic
            or self.default_arithmetic
            or ("double" if precision == 16 else "arbitrary")
        )
        if arithmetic not in {"double", "double-double", "arbitrary"}:
            raise ValueError("unknown virtual arithmetic")
        requested = self._default_r2 if r2_strategy is None else R2Strategy(r2_strategy)
        if requested is R2Strategy.DIRECT:
            raise CompatibilityError(
                "this virtual artifact only contains model-supplied R2"
            )
        digits = precision if arithmetic == "arbitrary" else None
        effective_policy = self._default_stability if stability is None else stability
        policy_input = _stability_input(effective_policy)
        points, scales = _inputs(momenta, mu_r2, len(self._metadata["external_pdgs"]))
        with self._lock:
            native = self._native
            if (arithmetic, digits) != (
                native.arithmetic,
                native.precision,
            ):
                candidate = _invoke(
                    self._module,
                    native.rebind,
                    arithmetic=arithmetic,
                    precision=digits,
                    r2_strategy=requested.value,
                    **({"stability": policy_input} if stability is not None else {}),
                )
                native = candidate
                # A call-level replacement policy stays local to this call.
                # Pass it before rebinding so an incompatible retained policy
                # cannot obstruct a valid new primary/verification pair.
                if stability is None:
                    self._native = candidate
            verification: tuple[VirtualVerificationDiagnostic, ...] = ()
            if policy_input is None:
                result = _invoke(
                    self._module,
                    native.evaluate_directed if directed else native.evaluate_partial,
                    points,
                    scales,
                    r2_strategy=requested.value,
                    **(
                        {"partial": True, "spin_vectors": spin_vectors}
                        if directed
                        else {"resolved": resolved}
                    ),
                )
            else:
                verified = _invoke(
                    self._module,
                    native.evaluate_directed_verified
                    if directed
                    else native.evaluate_verified,
                    points,
                    scales,
                    policy_input[0],
                    policy_input[1],
                    policy_input[2],
                    r2_strategy=requested.value,
                    **(
                        {"spin_vectors": spin_vectors}
                        if directed
                        else {"resolved": resolved}
                    ),
                    verify_successful=policy_input[3],
                )
                result = verified.result
                verification = tuple(
                    _verification_diagnostic(x)
                    for x in json.loads(verified.diagnostics_json)
                )
                if any(type(x.point_id) is not int for x in verification) or tuple(
                    x.point_id for x in verification
                ) != tuple(range(len(points))):
                    raise EvaluationError("native verification identities differ")
            failures = tuple(
                VirtualFailureDiagnostic(
                    item.point_id,
                    item.helicity,
                    item.order_index,
                    item.family_index,
                    item.stage,
                    item.detail,
                    item.external_leg,
                    item.color_component_index,
                    item.history_index,
                    item.direction,
                )
                for item in result.failures
            )
            if failures and on_failure == "raise":
                failed_points = sorted({item.point_id for item in failures})
                raise EvaluationError(
                    "unrecovered virtual numerical failures at points "
                    f"{failed_points}; "
                    "inspect error.failures for stage, helicity, "
                    "colour component, order and family details",
                    failures=failures,
                    verification=verification,
                )
            helicity_count = len(self._metadata["plan"]["helicities"])
            color_count = len(self._color_ids)
            requested_ids = tuple(result.requested_point_ids)
            successful_ids = tuple(result.point_ids)
            successful = set(successful_ids)
            failures_by_point: dict[int, list[VirtualFailureDiagnostic]] = {}
            for failure in failures:
                failures_by_point.setdefault(failure.point_id, []).append(failure)
            if (
                any(type(point) is not int for point in requested_ids + successful_ids)
                or requested_ids != tuple(range(len(points)))
                or successful_ids
                != tuple(point for point in requested_ids if point in successful)
                or set(failures_by_point) != set(requested_ids) - successful
                or result.r2_strategy != "model"
            ):
                raise EvaluationError(
                    "native virtual result has inconsistent lane or strategy metadata"
                )
            if verification and any(
                item.status
                not in {"success", "rescued", "failed", "accuracy-not-established"}
                or (item.status in {"success", "rescued"})
                != (item.point_id in successful)
                for item in verification
            ):
                raise EvaluationError("native verification acceptance differs")
            rows_per_point = helicity_count * color_count if resolved else 1
            count = len(successful_ids) * rows_per_point
            if len(result.statuses) != count:
                raise EvaluationError("native virtual reduction status count differs")
            failed = [
                (
                    {
                        "point": successful_ids[lane // rows_per_point],
                        "helicity": f"h{lane // color_count % helicity_count}",
                        "color_component": self._color_ids[lane % color_count],
                    }
                    if resolved
                    else {"point": successful_ids[lane]},
                    statuses,
                )
                for lane, statuses in enumerate(result.statuses)
                if any(status != "success" for _, status in statuses)
            ]
            if failed:
                raise EvaluationError(
                    f"unrecovered virtual reduction failures: {failed}"
                )
            if directed and (
                result.observable_kind != "directed-complex"
                or result.history_ids != [item.id for item in histories]
                or result.spin_legs != sorted(spin_vectors or {})
            ):
                raise EvaluationError("native directed history catalogue differs")
            history_count = len(histories) if directed else 1
            born = _rows(
                result.born,
                count,
                history_count * len(self._born_orders),
                arithmetic,
                correlated=directed,
            )
            components = result.component_batches
            if len(components) != 6:
                raise EvaluationError("native complete virtual result omits components")
            virtual = _rows(
                components[0],
                count,
                history_count * 3 * len(self._sectors),
                arithmetic,
                correlated=directed,
            )
            values = tuple(
                tuple(
                    LaurentCoefficients(*row[i : i + 3]) for i in range(0, len(row), 3)
                )
                for row in virtual
            )
            color_contraction = VirtualColorContraction(
                **json.loads(result.primary_color_contraction_json)
            )
            diagnostics = tuple(
                VirtualLaneDiagnostic(
                    verification[i].status
                    if verification
                    else "success"
                    if i in successful
                    else "failed",
                    cast(Literal["double", "double-double", "arbitrary"], arithmetic),
                    native.precision_bits,
                    verification[i].accuracy_established if verification else False,
                    verification[i].detail
                    if verification
                    else (
                        "successful initial evaluation; accuracy not established"
                        if i in successful
                        else "unrecovered virtual numerical failure"
                    ),
                    verification[i] if verification else None,
                    tuple(failures_by_point.get(i, ())),
                    color_contraction,
                )
                for i in range(len(points))
            )
            selection = R2Selection(
                requested,
                R2Strategy.MODEL,
                "explicit strategy"
                if requested is R2Strategy.MODEL
                else "only complete eligible strategy",
            )
            if directed:
                self._last_requested = requested
                return {
                    history.id: VirtualEvaluation(
                        _scatter_points(
                            tuple(
                                row[
                                    j * len(self._born_orders) : (j + 1)
                                    * len(self._born_orders)
                                ]
                                for row in born
                            ),
                            successful_ids,
                            len(points),
                        ),
                        _scatter_points(
                            tuple(
                                row[
                                    j * len(self._sectors) : (j + 1)
                                    * len(self._sectors)
                                ]
                                for row in values
                            ),
                            successful_ids,
                            len(points),
                        ),
                        self._born_orders,
                        self._sectors,
                        diagnostics,
                        selection,
                        correlated=True,
                        color_accuracy=self._color_accuracy,
                    )
                    for j, history in enumerate(histories)
                }
            if resolved:
                output: VirtualEvaluation | ResolvedVirtualEvaluation = (
                    ResolvedVirtualEvaluation(
                        _scatter_points(
                            tuple(
                                tuple(
                                    tuple(
                                        tuple(
                                            born[
                                                (p * helicity_count + h) * color_count
                                                + c
                                            ][order]
                                            for c in range(color_count)
                                        )
                                        for h in range(helicity_count)
                                    )
                                    for order in range(len(self._born_orders))
                                )
                                for p in range(len(successful_ids))
                            ),
                            successful_ids,
                            len(points),
                        ),
                        _scatter_points(
                            tuple(
                                tuple(
                                    tuple(
                                        tuple(
                                            values[
                                                (p * helicity_count + h) * color_count
                                                + c
                                            ][order]
                                            for c in range(color_count)
                                        )
                                        for h in range(helicity_count)
                                    )
                                    for order in range(len(self._sectors))
                                )
                                for p in range(len(successful_ids))
                            ),
                            successful_ids,
                            len(points),
                        ),
                        self._born_orders,
                        self._sectors,
                        diagnostics,
                        selection,
                        tuple(f"h{i}" for i in range(helicity_count)),
                        self._color_ids,
                        self._color_ids,
                        self._color_accuracy,
                    )
                )
            else:
                output = VirtualEvaluation(
                    _scatter_points(born, successful_ids, len(points)),
                    _scatter_points(values, successful_ids, len(points)),
                    self._born_orders,
                    self._sectors,
                    diagnostics,
                    selection,
                    color_accuracy=self._color_accuracy,
                )
            self._last_requested = requested
            return output

    def evaluate_resolved(
        self, momenta: Momenta, **kwargs: Any
    ) -> ResolvedVirtualEvaluation:
        return cast(
            ResolvedVirtualEvaluation, self._evaluate(momenta, resolved=True, **kwargs)
        )

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        parameters = _parameters(mapping)
        with self._lock:
            _invoke(self._module, self._native.set_model_parameters, parameters)

    def clear(self) -> None:
        with self._lock:
            _invoke(self._module, self._native.clear)

    def mute_warnings(self) -> None:
        self._muted = True

    def unmute_warnings(self) -> None:
        self._muted = False
