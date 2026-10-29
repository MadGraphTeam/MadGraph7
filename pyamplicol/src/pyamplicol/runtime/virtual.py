# SPDX-License-Identifier: 0BSD
"""Typed Python transport for the native, source-owned virtual runtime.

Python formats inputs and immutable outputs. The native coordinator evaluates
one complete batch, including source parameters, counterterms and reduction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from threading import RLock
from typing import Any, Literal, TypeVar, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    EvaluationError,
)
from pyamplicol.api.protocols import ModelParameters, Momenta, ScalarInput
from pyamplicol.api.virtual_results import (
    POLE_COMPONENTS,
    POLE_ORIGINS,
    LaurentCoefficients,
    ResolvedVirtualEvaluation,
    VirtualArithmeticSpec,
    VirtualColorContraction,
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
    VirtualPoleParts,
    VirtualScalar,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
    VirtualStabilityPolicy,
    VirtualVerificationAttempt,
    VirtualVerificationDiagnostic,
    computes_poles,
    pole_selection,
    splits_poles,
)
from pyamplicol.config import PoleSelection, R2Strategy
from pyamplicol.correlators import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatedValue,
)
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
    """Validate a real decimal input without a binary64 round trip: a number
    or ASCII decimal text (``Decimal``'s syntax; other digits and whitespace
    than ASCII are refused, as by the native SDKs'
    ``v2_runtime::decimal_parameter``)."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral, Real, Decimal)):
        raise TypeError(f"{description} must be a real number or decimal string")
    text = str(value)
    if not text.isascii():
        raise ValueError(f"{description} is not a decimal number")
    try:
        decimal = Decimal(text)
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


_NO_RESCUE = (
    "v2 loop runtimes return stability estimates (stability_checks) and leave "
    "precision escalation to the caller; a VirtualStabilityPolicy rescue is "
    "not available"
)


def v2_public_estimate(
    estimate: Any, with_poles: bool
) -> VirtualStabilityEstimate | None:
    """The public record of a v2 ``stability.StabilityEstimate``."""

    if estimate is None:
        return None
    coefficients = tuple(float(x) for x in estimate.coefficients)
    return VirtualStabilityEstimate(
        float(estimate.relative),
        coefficients[0],
        coefficients[1] if with_poles else None,
        coefficients[2] if with_poles else None,
        tuple((name, float(value)) for name, value in estimate.checks.items()),
    )


def v2_r2_strategy(value: R2Strategy | str | None) -> R2Strategy:
    """The R2 strategy of a v2 virtual runtime: v2 artifacts carry the
    model-supplied R2 rules only, so ``direct`` is refused."""

    strategy = R2Strategy.AUTO if value is None else R2Strategy(value)
    if strategy is R2Strategy.DIRECT:
        raise CompatibilityError("v2 virtual artifacts carry model-supplied R2 only")
    return strategy


def v2_options(
    *,
    helicities: Sequence[str] | None = None,
    color_flows: Sequence[str] | None = None,
    r2_strategy: str | None = None,
    stability: VirtualStabilityPolicy | None = None,
    poles: PoleSelection | str | None = None,
    stability_checks: VirtualStabilityChecks | None = None,
    on_failure: str = "raise",
) -> tuple[PoleSelection, VirtualStabilityChecks]:
    """Validate the public options a v2 virtual runtime does not forward
    unchanged; returns (pole selection, stability checks)."""

    if on_failure not in ("raise", "return"):
        raise ValueError("on_failure must be 'raise' or 'return'")
    if helicities or color_flows:
        raise CompatibilityError(
            "v2 virtual runtimes evaluate every helicity and colour; use "
            "evaluate_resolved for per-helicity and per-colour outputs"
        )
    v2_r2_strategy(r2_strategy)
    if stability is not None:
        raise CompatibilityError(_NO_RESCUE)
    selection = PoleSelection.ALL if poles is None else pole_selection(poles)
    checks = stability_checks or VirtualStabilityChecks()
    if not isinstance(checks, VirtualStabilityChecks):
        raise TypeError("stability_checks requires VirtualStabilityChecks")
    return selection, checks


class V2ObservablesRuntime:
    """The public virtual-runtime calls over a generated v2 slice's
    observables; the artifact runtime (``loops.v2.artifact.V2VirtualRuntime``)
    delegates its correlated, resolved and flavour-member calls here.

    It answers ``evaluate``, ``evaluate_resolved``, ``evaluate_correlated`` and
    ``evaluate_correlated_many`` with the public result types from an
    ``loops.v2.observables.ObservableSlice``: the colour histories and spin
    classes declared at its generation, optionally for an admitted flavour
    ``member`` (``loops.v2.observables.admit_member``) evaluated by the
    representative. ``executor="native"`` (the default) dresses and reduces
    every batch in the native v2 executor (binary64, ``double-double`` or
    MPFR arbitrary precision; one native plan per output column set, built
    once); ``"python"`` runs the Python reference executor (binary64 or
    arbitrary precision). Resolved output is per Born helicity row; at
    leading colour also per Born colour component, a Born basis element
    (``born-basis-anchored-interference``, design §1.6 decision 3) with ID
    ``basis-<index>`` (a member's are its representative's elements on the
    member's labels). At NLC and full colour the colour is contracted: one
    component per row, with the accuracy as its ID (the documented public
    shape of ``Runtime.evaluate_resolved``).

    Every call takes ``poles`` (a ``PoleSelection``; ``"none"`` is the finite
    part only, poles explicitly absent; ``"uv"`` and ``"ir"`` return the
    ultraviolet or infrared parts, with both origins by component in
    ``pole_parts``) and ``stability_checks`` (rotations
    and/or the reduction-direction check; requirements addendum C): a
    correlated evaluation then carries one estimate per point and output, a
    resolved one per point and entry (helicity row and colour component);
    without checks no estimate exists. ``on_failure="return"`` keeps failed
    points as ``None`` with their reasons. Accuracy is not otherwise
    established and nothing is escalated automatically.
    """

    contribution_kind = "virtual"
    execution_mode = "loop_v2"

    def __init__(
        self,
        observables: Any,
        *,
        model_parameters: ModelParameters | None = None,
        arithmetic: str | None = None,
        precision: int | None = None,
        executor: Literal["native", "python"] = "native",
        member: Any = None,
        lanes: int | None = None,
        threads: int | None = None,
    ) -> None:
        if executor not in ("native", "python"):
            raise ValueError(f"unknown v2 executor {executor!r}")
        self._observables = observables
        self._member = member
        self._executor = executor
        self._native_options = {
            key: value
            for key, value in (("lanes", lanes), ("threads", threads))
            if value is not None
        }
        self._native: dict[tuple[Any, ...], Any] = {}
        self._parameters = {
            name: real
            for name, (real, _imaginary) in (
                _parameters(model_parameters) or {}
            ).items()
        }
        self.default_arithmetic = arithmetic or (
            "double" if precision in (None, 16) else "arbitrary"
        )
        self.default_precision = precision if precision is not None else 16
        self._spin_vectors: dict[int, object] = {}
        self._lock = RLock()
        slice_ = observables.slice
        self._sectors = tuple(
            VirtualOrderSector(
                canonical_order(sector),
                tuple(BornLoopPair(born, loop) for born, loop in sorted(pairs)),
            )
            for sector, pairs in slice_.sectors
        )

    # -- declarations ---------------------------------------------------------

    @property
    def external_count(self) -> int:
        return len(self._observables.slice.blocks.process.legs)

    def available_color_correlations(self) -> tuple[ColorCorrelator, ...]:
        return (ColorCorrelator("born"), *self._observables.histories)

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        copied = _copy_spin_vectors(vectors, self._observables.spin_classes)
        with self._lock:
            self._spin_vectors = copied

    def set_model_parameters(self, parameters: Mapping[str, str]) -> None:
        """Update real external parameter overrides (decimal text)."""

        with self._lock:
            self._parameters.update(parameters)

    def clear(self) -> None:
        with self._lock:
            self._spin_vectors = {}
            self._native = {}

    # -- executors --------------------------------------------------------------

    def _runner(self) -> Any:
        """The Python reference evaluation of the slice or of the member."""

        from pyamplicol.loops.v2.observables import MemberObservables

        if self._member is None:
            return self._observables
        return MemberObservables(self._observables, self._member)

    def _native_columns(self, key: tuple[Any, ...]) -> Any:
        """The native executor of one output column set (built once):
        ``("correlated", histories)`` or ``("resolved", resolution)``."""

        from pyamplicol.loops.v2.observables import NativeObservables

        with self._lock:
            found = self._native.get(key)
            if found is None:
                kind, value = key
                options = {"member": self._member, **self._native_options}
                stored = getattr(self._observables, "native_columns", None)
                if stored is not None:
                    # A loaded process's stored columns (no slice).
                    found = self._native[key] = stored(kind, value, **options)
                    return found
                found = self._native[key] = (
                    NativeObservables.correlated(self._observables, value, **options)
                    if kind == "correlated"
                    else NativeObservables.resolved(self._observables, value, **options)
                )
            return found

    def _run(
        self,
        native_key: tuple[Any, ...],
        python: Callable[[Any, Any, Any], Any],
        points: Sequence[Any],
        scales: Sequence[Any],
        spins: Sequence[Any],
        options: Mapping[str, Any],
    ) -> tuple[list[Any], dict[int, list[str]]]:
        """Per-point outputs (``None`` for a failed point) and the failures."""

        from pyamplicol.loops.v2.plan_export import FailedPoints

        if self._executor == "native":
            try:
                return (
                    self._native_columns(native_key).evaluate(
                        points, scales, spin=list(spins), **options
                    ),
                    {},
                )
            except FailedPoints as failed:
                return list(failed.results), dict(failed.failures)
        out: list[Any] = []
        failures: dict[int, list[str]] = {}
        for index, (point, scale, spin) in enumerate(
            zip(points, scales, spins, strict=True)
        ):
            try:
                out.append(python(point, scale, spin))
            except ArithmeticError as error:
                out.append(None)
                failures[index] = [str(error)]
        return out, failures

    # -- evaluation -----------------------------------------------------------

    def _precision(
        self, arithmetic: str | None, precision: int | None
    ) -> tuple[str, Any]:
        """(public arithmetic, v2 ``Precision`` carrier)."""

        from pyamplicol.loops.v2.assemble import Precision

        precision = self.default_precision if precision is None else precision
        arithmetic = arithmetic or (
            self.default_arithmetic if precision == self.default_precision else None
        )
        arithmetic = arithmetic or ("double" if precision == 16 else "arbitrary")
        if arithmetic == "double":
            return arithmetic, Precision.double()
        if arithmetic == "arbitrary":
            return arithmetic, Precision.arbitrary(precision)
        if arithmetic == "double-double" and self._executor == "native":
            return arithmetic, Precision.double_double()
        raise CompatibilityError(
            f"the v2 {self._executor} executor does not evaluate in {arithmetic!r}"
        )

    def _scalar(self, value: Any, arithmetic: str, correlated: bool) -> VirtualScalar:
        if arithmetic == "double":
            return complex(value) if correlated else float(value)
        if correlated:
            real, imaginary = (
                (value.real, value.imag)
                if not isinstance(value, Decimal)
                else (value, Decimal(0))
            )
            return CorrelatedValue(Decimal(real), Decimal(imaginary))
        return Decimal(value)

    def _laurent(
        self, laurent: Any, arithmetic: str, correlated: bool, with_poles: bool = True
    ) -> LaurentCoefficients:
        if not with_poles:
            absent = cast(VirtualScalar, None)
            return LaurentCoefficients(
                self._scalar(laurent.finite, arithmetic, correlated), absent, absent
            )
        if laurent.single_pole is None:
            raise CompatibilityError("public virtual results carry all three poles")
        return LaurentCoefficients(
            *(
                self._scalar(value, arithmetic, correlated)
                for value in laurent.coefficients
            )
        )

    def _diagnostic(
        self,
        arithmetic: str,
        precision: Any,
        checked: bool,
        index: int = 0,
        reasons: Sequence[str] | None = None,
    ) -> VirtualLaneDiagnostic:
        bits = {"double": 53, "double-double": 106}.get(arithmetic) or math.ceil(
            precision.digits * math.log2(10)
        )
        executor = "native" if self._executor == "native" else "Python reference"
        if reasons is not None:
            reason = "; ".join(reasons)
            return VirtualLaneDiagnostic(
                "failed",
                arithmetic,  # type: ignore[arg-type]
                bits,
                False,
                f"v2 {executor} executor: {reason}",
                failures=(
                    VirtualFailureDiagnostic(
                        index, None, None, None, "v2-evaluation", reason
                    ),
                ),
            )
        return VirtualLaneDiagnostic(
            "success",
            arithmetic,  # type: ignore[arg-type]
            bits,
            False,
            f"v2 {executor} executor; "
            + (
                "accuracy is estimated only by the requested stability checks"
                if checked
                else "no stability check requested"
            ),
        )

    def _failed(self, failures: Mapping[int, Sequence[str]], on_failure: str) -> None:
        if failures and on_failure == "raise":
            index = min(failures)
            raise EvaluationError(
                f"v2 virtual evaluation failed at point {index}: "
                + "; ".join(failures[index])
            )

    def _evaluation(
        self,
        results: Sequence[Any],
        failures: Mapping[int, Sequence[str]],
        arithmetic: str,
        precision: Any,
        correlated: bool,
        poles: PoleSelection = PoleSelection.ALL,
        checks: VirtualStabilityChecks | None = None,
    ) -> VirtualEvaluation:

        born_orders = self._default_born_orders()
        checked = checks is not None and checks.enabled
        with_poles = computes_poles(poles)
        born_values: list[Any] = []
        values: list[Any] = []
        diagnostics: list[VirtualLaneDiagnostic] = []
        estimates: list[VirtualStabilityEstimate | None] = []
        split = splits_poles(poles)
        parts: list[VirtualPoleParts | None] = []
        for index, result in enumerate(results):
            if result is None:
                born_values.append(None)
                values.append(None)
                estimates.append(None)
                parts.append(None)
                diagnostics.append(
                    self._diagnostic(
                        arithmetic,
                        precision,
                        checked,
                        index,
                        failures.get(index, ["failed"]),
                    )
                )
                continue
            born_values.append(
                tuple(
                    self._scalar(result.born.get(order, 0), arithmetic, correlated)
                    for order in born_orders
                )
            )
            values.append(
                tuple(
                    self._laurent(
                        result.components["virtual"][sector.order],
                        arithmetic,
                        correlated,
                        with_poles,
                    )
                    for sector in self._sectors
                )
            )
            estimates.append(v2_public_estimate(result.stability, with_poles))
            if split:
                origins = result.pole_origins
                parts.append(
                    VirtualPoleParts(
                        *(
                            {
                                name: tuple(
                                    self._laurent(
                                        origins[origin][name][sector.order],
                                        arithmetic,
                                        correlated,
                                    )
                                    for sector in self._sectors
                                )
                                for name in POLE_COMPONENTS
                            }
                            for origin in POLE_ORIGINS
                        )
                    )
                )
            diagnostics.append(self._diagnostic(arithmetic, precision, checked))
        return VirtualEvaluation(
            born_values=tuple(born_values),
            values=tuple(values),
            born_orders=born_orders,
            sectors=self._sectors,
            diagnostics=tuple(diagnostics),
            r2=R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "model-supplied R2"),
            correlated=correlated,
            color_accuracy=self._observables.slice.request.accuracy,
            poles=poles,
            stability=tuple(estimates) if checked else None,
            pole_parts=tuple(parts) if split else None,
        )

    def _default_born_orders(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        """The squared Born orders (the keys of ``PointResult.born``)."""

        from pyamplicol.loops.v2.families import add_orders

        orders = self._observables.slice.blocks.born_orders
        return tuple(
            sorted({canonical_order(add_orders(a, b)) for a in orders for b in orders})
        )

    def evaluate(
        self,
        momenta: Momenta,
        *,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        **options: Any,
    ) -> VirtualEvaluation:
        return self.evaluate_correlated_many(
            momenta,
            {"born": CorrelatedRequest("born", {})},
            mu_r2=mu_r2,
            precision=precision,
            arithmetic=arithmetic,
            _correlated=False,
            **options,
        )["born"]

    def evaluate_correlated(
        self, momenta: Momenta, *, color_correlation: str = "born", **kwargs: Any
    ) -> VirtualEvaluation:
        return self.evaluate_correlated_many(
            momenta, {"selected": CorrelatedRequest(color_correlation)}, **kwargs
        )["selected"]

    def evaluate_correlated_many(
        self,
        momenta: Momenta,
        requests: Mapping[str, CorrelatedRequest],
        *,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        _correlated: bool = True,
        **options: Any,
    ) -> dict[str, VirtualEvaluation]:
        """Every request on one batch; requests with equal spin sources share
        their seed columns (one dressing per point and source assignment; one
        native batch per source assignment)."""

        selection, checks = v2_options(on_failure=on_failure, **options)
        observables = self._observables
        arithmetic, carrier = self._precision(arithmetic, precision)
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        known = {item.id for item in self.available_color_correlations()}
        with self._lock:
            inherited = dict(self._spin_vectors)
            parameters = dict(self._parameters)
        classes = observables.spin_classes
        allowed = tuple(sorted({leg for group in classes for leg in group}))
        groups: dict[tuple[Any, ...], list[str]] = {}
        sources: dict[tuple[Any, ...], tuple[dict[int, Any], ...]] = {}
        for label, request in requests.items():
            if not isinstance(request, CorrelatedRequest):
                raise TypeError("correlated requests require CorrelatedRequest records")
            if request.color_correlation not in known:
                raise CompatibilityError(
                    "undeclared virtual colour correlation "
                    f"{request.color_correlation!r}"
                )
            vectors = _copy_spin_vectors(
                inherited if request.spin_vectors is None else request.spin_vectors,
                classes,
                point_count=len(points),
            )
            prepared = _prepare_spin_vectors(
                vectors, allowed_legs=allowed, point_count=len(points)
            )
            key = tuple(
                tuple(sorted((leg, wave) for leg, wave in point.items()))
                for point in prepared
            )
            groups.setdefault(key, []).append(label)
            sources[key] = prepared
        settings = {
            "parameters": parameters,
            "precision": carrier,
            "poles": selection.value,
            "stability": checks.to_checks() if checks.enabled else None,
        }
        out: dict[str, VirtualEvaluation] = {}
        for key, labels in groups.items():
            histories = tuple(
                dict.fromkeys(requests[label].color_correlation for label in labels)
            )
            spins = [spin or None for spin in sources[key]]
            runner: Any = self._runner() if self._executor != "native" else None

            def python(
                point: Any,
                scale: Any,
                spin: Any,
                runner: Any = runner,
                histories: Any = histories,
            ) -> Any:
                return runner.evaluate(
                    point, scale, correlations=histories, spin=spin, **settings
                )

            per_point, failures = self._run(
                ("correlated", histories),
                python,
                points,
                scales,
                spins,
                settings,
            )
            self._failed(failures, on_failure)
            for label in labels:
                history = requests[label].color_correlation
                out[label] = self._evaluation(
                    [
                        None if results is None else results[history]
                        for results in per_point
                    ],
                    failures,
                    arithmetic,
                    carrier,
                    _correlated,
                    selection,
                    checks,
                )
        return {label: out[label] for label in requests}

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        **options: Any,
    ) -> ResolvedVirtualEvaluation:
        """Per Born helicity row and Born basis element (design §8.5)."""


        selection, checks = v2_options(on_failure=on_failure, **options)
        with_poles = computes_poles(selection)
        split = splits_poles(selection)
        observables = self._observables
        arithmetic, carrier = self._precision(arithmetic, precision)
        points, scales = _inputs(momenta, mu_r2, self.external_count)
        rows = observables.rows
        if self._member is not None:
            # The member's rows in its own catalogue order.
            rows = tuple(sorted(self._member.row(row) for row in rows))
        accuracy = observables.slice.request.accuracy
        resolution = "helicity-colour" if accuracy == "lc" else "helicity"
        components: tuple[Any, ...] = (
            observables.components if accuracy == "lc" else (None,)
        )
        with self._lock:
            parameters = dict(self._parameters)
        settings = {
            "parameters": parameters,
            "precision": carrier,
            "poles": selection.value,
            "stability": checks.to_checks() if checks.enabled else None,
        }
        runner: Any = self._runner() if self._executor != "native" else None

        def python(point: Any, scale: Any, _spin: Any) -> Any:
            return runner.evaluate_resolved(
                point, scale, resolution=resolution, **settings
            ).entries

        entries, failures = self._run(
            ("resolved", resolution),
            python,
            points,
            scales,
            [None] * len(points),
            settings,
        )
        self._failed(failures, on_failure)
        born_orders = self._default_born_orders()

        def born_value(entry: Any, order: Any) -> VirtualScalar:
            value = entry.born.get(order, 0) if entry is not None else 0
            return self._scalar(value, arithmetic, False)

        born_values: list[Any] = []
        values: list[Any] = []
        diagnostics: list[VirtualLaneDiagnostic] = []
        estimates: list[Any] = []
        parts: list[VirtualPoleParts | None] = []
        for index, result in enumerate(entries):
            if result is None:
                born_values.append(None)
                values.append(None)
                estimates.append(None)
                parts.append(None)
                diagnostics.append(
                    self._diagnostic(
                        arithmetic,
                        carrier,
                        checks.enabled,
                        index,
                        failures.get(index, ["failed"]),
                    )
                )
                continue
            born_values.append(
                tuple(
                    tuple(
                        tuple(
                            born_value(result.get((row, b)), order) for b in components
                        )
                        for row in rows
                    )
                    for order in born_orders
                )
            )
            values.append(
                tuple(
                    tuple(
                        tuple(
                            self._laurent(
                                result[(row, b)].components["virtual"][sector.order],
                                arithmetic,
                                False,
                                with_poles,
                            )
                            for b in components
                        )
                        for row in rows
                    )
                    for sector in self._sectors
                )
            )
            diagnostics.append(self._diagnostic(arithmetic, carrier, checks.enabled))
            estimates.append(
                tuple(
                    tuple(
                        v2_public_estimate(result[(row, b)].stability, with_poles)
                        for b in components
                    )
                    for row in rows
                )
                if checks.enabled
                else None
            )
            if split:

                def entry(
                    origin: str, name: str, sector: Any, result: Any = result
                ) -> Any:
                    return tuple(
                        tuple(
                            self._laurent(
                                result[(row, b)].pole_origins[origin][name][sector],
                                arithmetic,
                                False,
                            )
                            for b in components
                        )
                        for row in rows
                    )

                parts.append(
                    VirtualPoleParts(
                        *(
                            {
                                name: tuple(
                                    entry(origin, name, sector.order)
                                    for sector in self._sectors
                                )
                                for name in POLE_COMPONENTS
                            }
                            for origin in POLE_ORIGINS
                        )
                    )
                )
        ids = (
            tuple(f"basis-{b}" for b in components)
            if accuracy == "lc"
            else (str(accuracy),)
        )
        return ResolvedVirtualEvaluation(
            born_values=tuple(born_values),
            values=tuple(values),
            born_orders=born_orders,
            sectors=self._sectors,
            diagnostics=tuple(diagnostics),
            r2=R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "model-supplied R2"),
            helicity_ids=tuple(f"h{i}" for i in range(len(rows))),
            born_color_ids=ids,
            virtual_color_ids=ids,
            color_accuracy=observables.slice.request.accuracy,
            poles=selection,
            stability=tuple(estimates) if checks.enabled else None,
            pole_parts=tuple(parts) if split else None,
        )
