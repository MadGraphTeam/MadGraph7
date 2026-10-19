# SPDX-License-Identifier: 0BSD
"""Typed Python transport for the native, source-owned virtual runtime.

Python formats inputs and immutable outputs. The native coordinator evaluates
one complete batch, including source parameters, counterterms and reduction.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    DependencyError,
    EvaluationError,
)
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
    ResolvedVirtualEvaluation,
    VirtualArithmeticSpec,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualStabilityPolicy,
    VirtualVerificationAttempt,
    VirtualVerificationDiagnostic,
)
from pyamplicol.config import R2Strategy
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection


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
            )
            for attempt in value["attempts"]
        ),
        detail=value["detail"],
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
    batch: Any, count: int, width: int, arithmetic: str
) -> tuple[tuple[float | Decimal, ...], ...]:
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
    values = []
    for real, imaginary in flat:
        real, imaginary = Decimal(real), Decimal(imaginary)
        if not real.is_finite() or not imaginary.is_finite() or imaginary != 0:
            raise EvaluationError("ordinary virtual output must be finite and real")
        values.append(float(real) if arithmetic == "double" else real)
    return tuple(
        tuple(values[start : start + width]) for start in range(0, len(values), width)
    )


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
        self._default_r2 = (
            R2Strategy.AUTO if r2_strategy is None else R2Strategy(r2_strategy)
        )
        initial_arithmetic = arithmetic or (
            "double" if self.default_precision == 16 else "arbitrary"
        )
        try:
            module = importlib.import_module("_loop_reduction")
        except ImportError as error:
            raise DependencyError(
                "the native _loop_reduction extension is unavailable"
            ) from error
        if not hasattr(module, "NativeVirtualRuntime"):
            raise DependencyError(
                "the native _loop_reduction extension lacks source virtual artifact "
                "support; "
                "rebuild it with NativeVirtualRuntime"
            )
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
        )
        self._metadata: dict[str, Any] = json.loads(self._native.metadata_json())
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

    @property
    def artifact_id(self) -> str:
        return cast(str, self._metadata["artifact_id"])

    @property
    def physics(self) -> ProcessPhysics:
        metadata, plan = self._metadata, self._metadata["plan"]
        return ProcessPhysics(
            process_id=metadata["process_id"],
            process=metadata["expression"],
            color_accuracy="full",
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
                for i, (pdg, leg) in enumerate(
                    zip(plan["external_pdgs"], plan["state_legs"], strict=True)
                )
            ),
            helicities=tuple(
                HelicityConfiguration(f"h{i}", i, tuple(h), True, False, f"h{i}", 1.0)
                for i, h in enumerate(plan["helicities"])
            ),
            color_flows=(),
            contracted_color_components=(
                ContractedColorComponent("full", 0, "full colour interference"),
            ),
            reduction=PhysicsReduction("contracted-color", ()),
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
                    "initial_average": metadata["initial_average"],
                    "resolved_color_components": ["full"]
                    if metadata["physics"].get("resolved")
                    else [],
                    "resolved_normalization": (
                        "each helicity retains the inclusive incoming spin average"
                    ),
                    "source_revision": self._native.source_revision,
                    "arithmetic": self._native.arithmetic,
                    "supported_arithmetics": metadata["physics"]["arithmetic"],
                    "default_arithmetic": self.default_arithmetic,
                    "default_precision": self.default_precision,
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
                        "source-global binding must succeed in the requested carrier"
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
        resolved: bool,
    ) -> VirtualEvaluation | ResolvedVirtualEvaluation:
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
        points, scales = _inputs(momenta, mu_r2, len(self._metadata["external_pdgs"]))
        with self._lock:
            if (arithmetic, digits) != (
                self._native.arithmetic,
                self._native.precision,
            ):
                candidate = _invoke(
                    self._module,
                    self._native.rebind,
                    arithmetic=arithmetic,
                    precision=digits,
                    r2_strategy=requested.value,
                )
                self._native = candidate
            verification: tuple[VirtualVerificationDiagnostic, ...] = ()
            if stability is None:
                result = _invoke(
                    self._module,
                    self._native.evaluate_partial,
                    points,
                    scales,
                    r2_strategy=requested.value,
                    resolved=resolved,
                )
            else:
                if not isinstance(stability, VirtualStabilityPolicy):
                    raise TypeError("stability requires a VirtualStabilityPolicy")
                verified = _invoke(
                    self._module,
                    self._native.evaluate_verified,
                    points,
                    scales,
                    [(x.arithmetic, x.precision) for x in stability.higher],
                    stability.relative_tolerance,
                    stability.absolute_tolerance,
                    r2_strategy=requested.value,
                    resolved=resolved,
                    verify_successful=stability.verify_successful,
                )
                result = verified.result
                verification = tuple(
                    _verification_diagnostic(x)
                    for x in json.loads(verified.diagnostics_json)
                )
                if tuple(x.point_id for x in verification) != tuple(range(len(points))):
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
                )
                for item in result.failures
            )
            if failures:
                failed_points = sorted({item.point_id for item in failures})
                raise EvaluationError(
                    "unrecovered virtual numerical failures at points "
                    f"{failed_points}; "
                    "inspect error.failures for stage, helicity, "
                    "order and family details",
                    failures=failures,
                    verification=verification,
                )
            helicity_count = len(self._metadata["plan"]["helicities"])
            count = len(points) * (helicity_count if resolved else 1)
            if (
                list(result.point_ids) != list(range(count))
                or result.r2_strategy != "model"
            ):
                raise EvaluationError(
                    "native virtual result has inconsistent lane or strategy metadata"
                )
            failed = [
                (
                    {
                        "point": lane // helicity_count,
                        "helicity": f"h{lane % helicity_count}",
                    }
                    if resolved
                    else {"point": lane},
                    statuses,
                )
                for lane, statuses in enumerate(result.statuses)
                if any(status != "success" for _, status in statuses)
            ]
            if len(result.statuses) != count or failed:
                raise EvaluationError(
                    f"unrecovered virtual reduction failures: {failed}"
                )
            born = _rows(result.born, count, len(self._born_orders), arithmetic)
            components = result.component_batches
            if len(components) != 6:
                raise EvaluationError("native complete virtual result omits components")
            virtual = _rows(components[0], count, 3 * len(self._sectors), arithmetic)
            values = tuple(
                tuple(
                    LaurentCoefficients(*row[i : i + 3]) for i in range(0, len(row), 3)
                )
                for row in virtual
            )
            diagnostics = tuple(
                VirtualLaneDiagnostic(
                    cast(Literal["success", "rescued"], verification[i].status)
                    if verification
                    else "success",
                    cast(Literal["double", "double-double", "arbitrary"], arithmetic),
                    self._native.precision_bits,
                    verification[i].accuracy_established if verification else False,
                    verification[i].detail
                    if verification
                    else "successful initial evaluation; accuracy not established",
                    verification[i] if verification else None,
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
            if resolved:
                output: VirtualEvaluation | ResolvedVirtualEvaluation = (
                    ResolvedVirtualEvaluation(
                        tuple(
                            tuple(
                                tuple(
                                    (born[p * helicity_count + h][order],)
                                    for h in range(helicity_count)
                                )
                                for order in range(len(self._born_orders))
                            )
                            for p in range(len(points))
                        ),
                        tuple(
                            tuple(
                                tuple(
                                    (values[p * helicity_count + h][order],)
                                    for h in range(helicity_count)
                                )
                                for order in range(len(self._sectors))
                            )
                            for p in range(len(points))
                        ),
                        self._born_orders,
                        self._sectors,
                        diagnostics,
                        selection,
                        tuple(f"h{i}" for i in range(helicity_count)),
                        ("full",),
                        ("full",),
                        "full",
                    )
                )
            else:
                output = VirtualEvaluation(
                    born,
                    values,
                    self._born_orders,
                    self._sectors,
                    diagnostics,
                    selection,
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
