# SPDX-License-Identifier: 0BSD
"""Immutable Born-loop results; no implicit order sum or finite-only coercion."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, TypeAlias, cast

from pyamplicol.config import PoleSelection, R2Strategy
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.loops.orders import VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.loops.v2.poles import ORIGINS
from pyamplicol.loops.v2.stability import (
    DEFAULT_RESCUE,
    RESCUE_ROTATION,
    RESCUE_TARGET,
    Rotation,
    StabilityChecks,
)
from pyamplicol.loops.v2.stability import default_rotation as _default_rotation
from pyamplicol.models.base import CouplingOrders

from .results import _exact_decimal_sum

VirtualScalar: TypeAlias = float | complex | Decimal | CorrelatedValue
EPSILON_POWERS = (0, -1, -2)
# The contributions a virtual result holds: the Born-virtual interference and
# |A_loop|^2 of a loop-induced process (m-lit Q36).
CONTRIBUTIONS = ("virtual", "loop-induced")


class LaurentCoefficients(NamedTuple):
    """Physical coefficients in the declared C_epsilon convention.

    The poles are ``None`` in a finite-only evaluation (``poles="none"``),
    which never computed them. Under ``poles="uv"`` they are the bare loop's
    UV poles, under ``"ir"`` the renormalised virtual's IR poles
    (:class:`PoleSelection`); the finite part is the renormalised virtual's
    under every selection.
    """

    finite: VirtualScalar
    single_pole: VirtualScalar
    double_pole: VirtualScalar


# Axes are explicit even for unavailable lanes, which retain their None entry.
OrderCoefficients: TypeAlias = tuple[tuple[VirtualScalar, ...] | None, ...]
VirtualValues: TypeAlias = tuple[tuple[LaurentCoefficients, ...] | None, ...]
ResolvedCoefficients: TypeAlias = tuple[
    tuple[tuple[tuple[VirtualScalar, ...], ...], ...] | None, ...
]
ResolvedVirtualValues: TypeAlias = tuple[
    tuple[tuple[tuple[LaurentCoefficients, ...], ...], ...] | None, ...
]
_FrozenAxis: TypeAlias = VirtualScalar | tuple["_FrozenAxis", ...]


@dataclass(frozen=True, slots=True)
class VirtualFailureDiagnostic:
    """One failed native numerical channel with original physical identity."""

    point_id: int
    helicity: int | None
    order_index: int | None
    family_index: int | None
    stage: str
    detail: str
    external_leg: int | None = None
    color_component_index: int | None = None
    history_index: int | None = None
    direction: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "point_id",
            "helicity",
            "order_index",
            "family_index",
            "external_leg",
            "color_component_index",
            "history_index",
            "direction",
        ):
            value = getattr(self, name)
            if value is None and name != "point_id":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.direction not in (None, 0, 1):
            raise ValueError("directed failure direction must be 0 or 1")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("failure stage must be a nonempty string")
        if not isinstance(self.detail, str):
            raise TypeError("failure detail must be a string")


@dataclass(frozen=True, slots=True)
class VirtualArithmeticSpec:
    """Explicit carrier; decimal precision applies only to arbitrary arithmetic."""

    arithmetic: Literal["double", "double-double", "arbitrary"]
    precision: int | None = None

    def __post_init__(self) -> None:
        if self.arithmetic not in {"double", "double-double", "arbitrary"}:
            raise ValueError("unknown virtual arithmetic")
        if self.arithmetic == "arbitrary":
            if type(self.precision) is not int or self.precision <= 0:
                raise ValueError(
                    "arbitrary arithmetic requires positive decimal precision"
                )
        elif self.precision is not None:
            raise ValueError("precision applies only to arbitrary arithmetic")


def parse_rotation(specification: str) -> Rotation:
    """One rotation check from text.

    ``"<x|y|z>:<cos>:<sin>"`` is exact (rational cos and sin with
    cos^2 + sin^2 = 1, e.g. ``"z:3/5:4/5"``); ``"<ax>,<ay>,<az>:<radians>"``
    rotates about an axis by an angle, rounded to an exact rotation
    (``Rotation.from_axis_angle``); ``"euler:<alpha>,<beta>,<gamma>"``
    (radians) is gammaloop's Euler rotation Rz(gamma) Ry(beta) Rx(alpha),
    rounded once to an exact rotation (``Rotation.from_euler``).
    """

    if not isinstance(specification, str):
        raise TypeError("a rotation specification is text")
    parts = specification.split(":")
    if len(parts) == 3 and parts[0] in ("x", "y", "z"):
        return Rotation.about(parts[0], parts[1], parts[2])
    if len(parts) == 2 and parts[0] == "euler":
        angles = tuple(float(value) for value in parts[1].split(","))
        if len(angles) != 3:
            raise ValueError(f"rotation {specification!r} needs three Euler angles")
        return Rotation.from_euler(*angles)
    if len(parts) == 2:
        axis = tuple(float(value) for value in parts[0].split(","))
        return Rotation.from_axis_angle(axis, float(parts[1]))
    raise ValueError(
        f"rotation {specification!r} is neither '<x|y|z>:<cos>:<sin>', "
        "'<ax>,<ay>,<az>:<radians>' nor 'euler:<alpha>,<beta>,<gamma>'"
    )


@dataclass(frozen=True, slots=True)
class VirtualStabilityChecks:
    """Stability checks of one evaluation (requirements addendum C).

    ``rotations`` re-evaluates every point with its momenta rotated by each
    rotation (exact ``Rotation`` records, specification text for
    :func:`parse_rotation`, or a count of default rotations); the
    reduction-direction check reduces every loop numerator a second time in a
    reversed frame, reusing its coefficients and the cached master integrals;
    with ``rescaled`` that re-reduction runs at rescaled inputs, so the master
    integrals are evaluated again and their own rounding is measured too
    (requires ``reduction_direction``). They combine. Without a check no
    estimate is returned. Escalating the arithmetic is the caller's decision.
    ``VirtualStabilityChecks(1, True, rescaled=True)`` is the recommended
    configuration (``loops.v2.stability.RECOMMENDED_CHECKS``).
    """

    # Normalised to exact ``Rotation`` records by ``__post_init__``.
    rotations: tuple[Rotation | str, ...] | int = ()
    reduction_direction: bool = False
    rescaled: bool = False

    def __post_init__(self) -> None:
        given = self.rotations
        if isinstance(given, bool):
            raise TypeError("rotations take a count or rotation records")
        rotations: tuple[Rotation, ...]
        if isinstance(given, int):
            if given < 0:
                raise ValueError("the rotation count is nonnegative")
            rotations = tuple(_default_rotation(k) for k in range(given))
        else:
            rotations = tuple(
                parse_rotation(item) if isinstance(item, str) else item
                for item in given
            )
        if type(self.reduction_direction) is not bool:
            raise TypeError("reduction_direction must be a boolean")
        if type(self.rescaled) is not bool:
            raise TypeError("rescaled must be a boolean")
        object.__setattr__(
            self,
            "rotations",
            StabilityChecks(
                rotations, self.reduction_direction, self.rescaled
            ).rotations,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.rotations) or self.reduction_direction

    def to_checks(self) -> StabilityChecks:
        return StabilityChecks(
            cast(tuple[Rotation, ...], self.rotations),
            self.reduction_direction,
            self.rescaled,
        )


def _nonnegative(value: object, name: str) -> None:
    if not isinstance(value, float) or math.isnan(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative float")


def _absolute_axis(
    value: object, name: str, with_poles: bool
) -> tuple[LaurentCoefficients, ...]:
    """Per-sector absolute Laurent floats (``VirtualStabilityEstimate``'s
    ``absolute`` and ``magnitude``): poles ``None`` exactly when absent."""

    out = []
    for sector in tuple(cast(Sequence[object], value)):
        laurent = tuple(cast(Sequence[object], sector))
        if len(laurent) != 3:
            raise ValueError(f"{name} holds Laurent triples per sector")
        _nonnegative(laurent[0], name)
        for pole in laurent[1:]:
            if not with_poles:
                if pole is not None:
                    raise ValueError(f"{name} has no pole entries without poles")
                continue
            _nonnegative(pole, name)
        out.append(LaurentCoefficients(*cast(tuple[float, float, float], laurent)))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class VirtualStabilityEstimate:
    """Estimated accuracy of one point's virtual.

    ``relative`` bounds each sector's error relative to that sector's largest
    Laurent coefficient; ``finite``, ``single_pole`` and ``double_pole``
    bound each coefficient's error relative to the coefficient itself (the
    poles are ``None`` in finite-only evaluations; ``inf`` marks an error
    estimate on an exact zero). ``checks`` names every check that ran with
    its own relative budget (``rotation-<k>``, ``reduction-direction`` or
    ``reduction-rescaled``).

    The absolute estimate E and the scale S it is measured against (v2 loop
    runtimes fill them; ``None`` otherwise): ``absolute[s]`` holds E per
    sector and Laurent coefficient (:class:`LaurentCoefficients` of floats,
    poles ``None`` when not computed), ``magnitude[s]`` the scale S of the
    same entries (the absolute sum of the contributions before they cancel;
    for a loop-induced result the pre-cancellation scale of its amplitudes,
    m-lit PLAN §2.a.8), and ``scale`` the largest E / S over them: the
    accuracy against the size of what cancelled (the ``"scale"`` metric of
    a :class:`VirtualRescueLevel`). A value with ``|value| <= E`` is zero
    within the estimate (m-lit Q13: zeros are numerical, never declared by
    a physics rule). For a loop-induced result
    (``contribution="loop-induced"``) the pole estimates ``single_pole`` and
    ``double_pole`` are measured against S as well (max over sectors of E /
    S): its poles vanish, so their own size is no scale (m-lit PLAN §2.a.7,
    "Pole shape").
    """

    relative: float
    finite: float
    single_pole: float | None
    double_pole: float | None
    checks: tuple[tuple[str, float], ...]
    scale: float | None = None
    absolute: tuple[LaurentCoefficients, ...] | None = None
    magnitude: tuple[LaurentCoefficients, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("relative", "finite", "single_pole", "double_pole", "scale"):
            value = getattr(self, name)
            if value is None and name in ("single_pole", "double_pole", "scale"):
                continue
            _nonnegative(value, name)
        if (self.single_pole is None) != (self.double_pole is None):
            raise ValueError("both pole estimates are present or absent")
        checks = tuple((str(name), float(value)) for name, value in self.checks)
        if not checks or len({name for name, _ in checks}) != len(checks):
            raise ValueError("an estimate names its distinct checks")
        object.__setattr__(self, "checks", checks)
        given = [item is None for item in (self.scale, self.absolute, self.magnitude)]
        if any(given) != all(given):
            raise ValueError("scale, absolute and magnitude are given together")
        if self.absolute is not None:
            with_poles = self.single_pole is not None
            absolute = _absolute_axis(self.absolute, "absolute", with_poles)
            magnitude = _absolute_axis(self.magnitude, "magnitude", with_poles)
            if len(absolute) != len(magnitude) or not absolute:
                raise ValueError("absolute and magnitude follow the sectors")
            object.__setattr__(self, "absolute", absolute)
            object.__setattr__(self, "magnitude", magnitude)


REDUCTION_CHECKS = ("none", "reversed", "rescaled")
RESCUE_METRICS = ("relative", "finite", "scale")
# The reduction backends of a v2 loop evaluation (m-lit Q17 / Q18): "auto"
# takes the per-type default (``loops.v2.reduction``).
REDUCTION_BACKENDS = ("auto", "ninjars", "tirust")


@dataclass(frozen=True, slots=True)
class VirtualReduction:
    """The reduction backend of one evaluation (m-lit PLAN §2.b.4, RULED
    Q17 / Q18): the ``requested`` one (``"auto"``, ``"ninjars"`` or
    ``"tirust"``), the ``effective`` one that reduced every loop numerator
    of the call, and the ``reason`` of the choice. ``"auto"`` takes the
    per-type default (loop-induced: TIRust; Born-interfered: NinjaRS) and
    falls back to NinjaRS, with the reason, where TIRust cannot serve the
    call; an explicitly requested backend that cannot serve it is refused
    instead. The choice is made once per call and carrier, never per point
    (no silent switch)."""

    requested: Literal["auto", "ninjars", "tirust"]
    effective: Literal["ninjars", "tirust"]
    reason: str

    def __post_init__(self) -> None:
        if self.requested not in REDUCTION_BACKENDS:
            raise ValueError(f"requested reduction must be one of {REDUCTION_BACKENDS}")
        if self.effective not in REDUCTION_BACKENDS[1:]:
            raise ValueError("effective reduction must be 'ninjars' or 'tirust'")
        if self.requested not in ("auto", self.effective):
            raise ValueError("an explicit reduction request is served as requested")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("a reduction choice states its reason")


def _carrier_bits(arithmetic: str, precision: int | None) -> int:
    """The precision bits a carrier delivers (``VirtualLaneDiagnostic``)."""

    if arithmetic == "double":
        return 53
    if arithmetic == "double-double":
        return 106
    return math.ceil(cast(int, precision) * math.log2(10))


@dataclass(frozen=True, slots=True)
class VirtualRescueLevel:
    """One level of a :class:`VirtualRescuePolicy`: a carrier, the stability
    checks every point of the level runs, and the accuracy it must reach.

    ``arithmetic`` and ``precision`` (decimal digits, arbitrary arithmetic
    only) as in :class:`VirtualArithmeticSpec`. ``rotations`` re-evaluates
    every point rotated (a count of default rotations, exact ``Rotation``
    records or rotation text, :func:`parse_rotation`, e.g. gammaloop's
    ``"euler:0.1,0.2,0.3"``); ``reduction_check`` re-reduces every loop
    numerator in the ``"reversed"`` frame (cached masters) or the
    ``"rescaled"`` one (masters evaluated again), or ``"none"``
    (:class:`VirtualStabilityChecks`). A point is accepted at the level when
    it evaluates and its estimate's ``metric`` (``"relative"``:
    ``VirtualStabilityEstimate.relative``, each sector against its largest
    Laurent coefficient; ``"finite"``: the finite part against itself;
    ``"scale"``: ``VirtualStabilityEstimate.scale``, E <= target x S against
    the scale S of what cancelled, an opt-in acceptance rule for values that
    may vanish, such as loop-induced zeros, m-lit PLAN §2.a.8) is at most
    ``target``. A level without any check accepts no point by its
    estimate; it may only be the last level, which keeps every point it
    evaluates.
    """

    arithmetic: Literal["double", "double-double", "arbitrary"]
    precision: int | None = None
    rotations: tuple[Rotation | str, ...] | int = (RESCUE_ROTATION,)
    reduction_check: Literal["none", "reversed", "rescaled"] = "rescaled"
    target: float = RESCUE_TARGET
    metric: Literal["relative", "finite", "scale"] = "relative"

    def __post_init__(self) -> None:
        VirtualArithmeticSpec(self.arithmetic, self.precision)
        if self.reduction_check not in REDUCTION_CHECKS:
            raise ValueError(f"reduction_check must be one of {REDUCTION_CHECKS}")
        if self.metric not in RESCUE_METRICS:
            raise ValueError(f"metric must be one of {RESCUE_METRICS}")
        target = self.target
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("a rescue target is a number")
        if not math.isfinite(target) or target <= 0:
            raise ValueError("a rescue target is finite and positive")
        checks = VirtualStabilityChecks(
            self.rotations,
            self.reduction_check != "none",
            rescaled=self.reduction_check == "rescaled",
        )
        object.__setattr__(self, "rotations", checks.rotations)
        object.__setattr__(self, "target", float(target))

    @property
    def checks(self) -> VirtualStabilityChecks:
        """The stability checks of the level's evaluation."""

        return VirtualStabilityChecks(
            self.rotations,
            self.reduction_check != "none",
            rescaled=self.reduction_check == "rescaled",
        )

    @property
    def precision_bits(self) -> int:
        return _carrier_bits(self.arithmetic, self.precision)


def _default_levels() -> tuple[VirtualRescueLevel, ...]:
    return tuple(
        VirtualRescueLevel(cast(Any, arithmetic), digits)
        for arithmetic, digits in DEFAULT_RESCUE
    )


@dataclass(frozen=True, slots=True)
class VirtualRescuePolicy:
    """The built-in stability rescue of a virtual ``evaluate`` (Python API
    only; off unless passed): an ordered ladder of :class:`VirtualRescueLevel`
    in strictly increasing precision bits, modelled on gammaloop's stability
    levels.

    Level 0 evaluates the whole batch. A point that fails (also in one of its
    rotated evaluations) or misses the level's target is evaluated again at
    the next level from its original exact inputs (no value is promoted); an
    accepted point is never evaluated again. At the last level a point that
    evaluates keeps its value, with the target marked as not met; a point
    that fails at every level is a failed lane (raised under
    ``on_failure="raise"``). ``VirtualEvaluation.rescue`` records per point
    the level reached and every attempt.

    The default ``VirtualRescuePolicy()`` runs binary64, then double-double,
    then 57-digit arbitrary precision, each with target 1e-5 on the relative
    estimate, one rotation ``"euler:0.1,0.2,0.3"`` and the rescaled reduction
    check (``loops.v2.stability.DEFAULT_RESCUE``).
    """

    levels: tuple[VirtualRescueLevel, ...] = field(default_factory=_default_levels)

    def __post_init__(self) -> None:
        if isinstance(self.levels, VirtualRescueLevel):
            raise TypeError("levels is a sequence of VirtualRescueLevel records")
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("a rescue policy has at least one level")
        if any(not isinstance(level, VirtualRescueLevel) for level in levels):
            raise TypeError("rescue levels are VirtualRescueLevel records")
        bits = [level.precision_bits for level in levels]
        if any(later <= earlier for earlier, later in pairwise(bits)):
            raise ValueError(
                f"rescue levels need strictly increasing precision bits (got {bits})"
            )
        if any(not level.checks.enabled for level in levels[:-1]):
            raise ValueError(
                "only the last rescue level may run without a stability check"
            )
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True, slots=True)
class VirtualRescueAttempt:
    """One level's evaluation of one point: its carrier, the value of the
    level's metric (``None`` when the point failed or the level runs no
    check) or the failure."""

    arithmetic: Literal["double", "double-double", "arbitrary"]
    precision_bits: int
    estimate: float | None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.failure is not None:
            if not isinstance(self.failure, str) or self.estimate is not None:
                raise ValueError("a failed attempt names its failure, no estimate")
        elif self.estimate is not None and (
            not isinstance(self.estimate, float)
            or math.isnan(self.estimate)
            or self.estimate < 0
        ):
            raise ValueError("an attempt's estimate is a nonnegative float")


@dataclass(frozen=True, slots=True)
class VirtualRescueRecord:
    """The rescue of one point: the level whose evaluation it returns (its
    carrier and target), whether that evaluation's estimate met the target,
    and every attempt from level 0 on. ``met`` is the estimate's verdict
    (the stability estimate's calibrated miss rates apply), not a
    certificate."""

    level: int
    arithmetic: Literal["double", "double-double", "arbitrary"]
    precision_bits: int
    target: float
    met: bool
    attempts: tuple[VirtualRescueAttempt, ...]

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if type(self.level) is not int or self.level < 0:
            raise ValueError("a rescue level is a nonnegative integer")
        if len(attempts) != self.level + 1 or any(
            not isinstance(item, VirtualRescueAttempt) for item in attempts
        ):
            raise ValueError("a rescue record holds one attempt per level reached")
        last = attempts[-1]
        if (last.arithmetic, last.precision_bits) != (
            self.arithmetic,
            self.precision_bits,
        ):
            raise ValueError("a rescue record's carrier is its last attempt's")
        if type(self.met) is not bool:
            raise TypeError("met must be a boolean")
        if self.met and (last.estimate is None or last.estimate > self.target):
            raise ValueError("a met target needs an estimate within it")
        object.__setattr__(self, "attempts", attempts)

    @property
    def failed(self) -> bool:
        """Whether the point failed at every level."""

        return self.attempts[-1].failure is not None


@dataclass(frozen=True, slots=True)
class VirtualColorContraction:
    """Selected bare-loop dual-weight operation in one arithmetic attempt.

    Born-square and counterterm contractions remain direct. A failed attempt
    need not reach the selected stage; this is not an execution-time counter.
    """

    requested: Literal["auto", "direct", "symmetric-group-fft"]
    algorithm: Literal["direct", "symmetric-group-fft"]
    reason: str
    regular_kernels: int = 0
    direct_residual_entries: int = 0

    def __post_init__(self) -> None:
        if self.requested not in {"auto", "direct", "symmetric-group-fft"}:
            raise ValueError("unknown requested colour contraction")
        if self.algorithm not in {"direct", "symmetric-group-fft"}:
            raise ValueError("unknown effective colour contraction")
        if any(
            type(n) is not int or n < 0
            for n in (self.regular_kernels, self.direct_residual_entries)
        ):
            raise ValueError("colour operation counts must be nonnegative integers")
        if self.algorithm == "symmetric-group-fft" and self.regular_kernels == 0:
            raise ValueError("FFT colour operation requires actual regular kernels")


@dataclass(frozen=True, slots=True)
class VirtualVerificationAttempt:
    arithmetic: VirtualArithmeticSpec
    precision_bits: int
    numerical_success: bool
    compared_precision_bits: int | None = None
    agreement: bool | None = None
    conversion_agreement: bool | None = None
    failures: tuple[VirtualFailureDiagnostic, ...] = ()
    color_contraction: VirtualColorContraction | None = None


@dataclass(frozen=True, slots=True)
class VirtualVerificationDiagnostic:
    """Empirical numerical agreement, not independent physics certification."""

    point_id: int
    status: Literal["success", "rescued", "failed", "accuracy-not-established"]
    accuracy_established: bool
    agreeing_bits: tuple[int, int] | None
    attempts: tuple[VirtualVerificationAttempt, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class VirtualLaneDiagnostic:
    """Accuracy is explicit; a finite output is not an accuracy certificate."""

    status: Literal[
        "success", "rescued", "unsupported", "failed", "accuracy-not-established"
    ]
    arithmetic: Literal["double", "double-double", "arbitrary"]
    precision_bits: int
    accuracy_established: bool
    detail: str = ""
    verification: VirtualVerificationDiagnostic | None = None
    failures: tuple[VirtualFailureDiagnostic, ...] = ()
    # The initial attempt; rescued evaluations retain their actual later
    # operations in verification.attempts.
    primary_color_contraction: VirtualColorContraction | None = None
    # The reduction backend of the evaluation (m-lit Q17 / Q18).
    reduction: VirtualReduction | None = None
    # A loop-induced point's amplitude-level finiteness ratio (m-lit PLAN
    # §2.a.7, L3): the largest ||a_{-k}||_G over loop orders, helicity rows
    # and pole coefficients against the row's pre-cancellation term scale,
    # of the order of the carrier's unit roundoff for a complete amplitude.
    # Formed with the poles and the stability checks (their term scale);
    # None otherwise and for Born-interfered results.
    finiteness: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "success",
            "rescued",
            "unsupported",
            "failed",
            "accuracy-not-established",
        }:
            raise ValueError("unknown virtual lane status")
        if self.arithmetic not in {"double", "double-double", "arbitrary"}:
            raise ValueError("unknown virtual arithmetic")
        if (
            isinstance(self.precision_bits, bool)
            or not isinstance(self.precision_bits, int)
            or self.precision_bits < 2
        ):
            raise ValueError("precision_bits must be an integer of at least two")
        if not isinstance(self.accuracy_established, bool):
            raise TypeError("accuracy_established must be a boolean")
        fixed_bits = {"double": 53, "double-double": 106}.get(self.arithmetic)
        if fixed_bits is not None and self.precision_bits != fixed_bits:
            raise ValueError(f"{self.arithmetic} delivers {fixed_bits} precision bits")
        if not self.valid and self.accuracy_established:
            raise ValueError("a failed or unsupported lane cannot establish accuracy")
        failures = tuple(self.failures)
        if any(not isinstance(item, VirtualFailureDiagnostic) for item in failures):
            raise TypeError("lane failures require VirtualFailureDiagnostic records")
        if self.valid and failures:
            raise ValueError("valid lanes cannot retain unrecovered failures")
        object.__setattr__(self, "failures", failures)
        if self.reduction is not None and not isinstance(
            self.reduction, VirtualReduction
        ):
            raise TypeError("a lane's reduction is a VirtualReduction record")
        if self.finiteness is not None:
            _nonnegative(self.finiteness, "finiteness")
            if not self.valid:
                raise ValueError("failed lanes carry no finiteness ratio")

    @property
    def valid(self) -> bool:
        return self.status in {"success", "rescued"}


def _scalar(value: object, correlated: bool, arithmetic: str) -> VirtualScalar:
    if isinstance(value, bool):
        raise TypeError("virtual values must be numerical coefficients")
    if arithmetic != "double" and not isinstance(value, (Decimal, CorrelatedValue)):
        raise TypeError(
            "high-precision results require precision-preserving containers"
        )
    if isinstance(value, Decimal):
        finite = value.is_finite()
    elif isinstance(value, CorrelatedValue):
        if not correlated:
            raise TypeError("ordinary interference must be real")
        if not isinstance(value.real, Decimal) or not isinstance(value.imag, Decimal):
            raise TypeError("complex high-precision values require Decimal components")
        finite = value.real.is_finite() and value.imag.is_finite()
    elif isinstance(value, complex):
        if not correlated:
            raise TypeError("ordinary interference must be real")
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
    elif isinstance(value, (float, int)):
        value = float(value)
        finite = math.isfinite(value)
    else:
        raise TypeError("unsupported virtual coefficient container")
    if not finite:
        raise ValueError("failed lanes must omit values, not return nonfinite data")
    return value


def pole_selection(value: object) -> PoleSelection:
    """The :class:`PoleSelection` of a public ``poles`` argument (the enum or
    its value text: ``"all"``, ``"none"``, ``"uv"`` or ``"ir"``)."""

    if isinstance(value, PoleSelection):
        return value
    choices = ", ".join(repr(item.value) for item in PoleSelection)
    if not isinstance(value, str):
        raise TypeError(f"poles takes a PoleSelection ({choices}); got {value!r}")
    try:
        return PoleSelection(value)
    except ValueError:
        message = f"unknown pole selection {value!r}; one of {choices}"
        raise ValueError(message) from None


def computes_poles(selection: PoleSelection) -> bool:
    """Whether an evaluation under ``selection`` returns pole coefficients."""

    return selection is not PoleSelection.NONE


def splits_poles(selection: PoleSelection) -> bool:
    """Whether an evaluation under ``selection`` splits every pole by origin
    (``"uv"`` and ``"ir"``; :class:`VirtualPoleParts`)."""

    return selection in (PoleSelection.UV, PoleSelection.IR)


# The origins of a split pole and the components whose parts a split
# evaluation reports (:class:`VirtualPoleParts`).
POLE_ORIGINS = ORIGINS
POLE_COMPONENTS = ("virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields")


@dataclass(frozen=True, slots=True)
class VirtualPoleParts:
    """The ultraviolet and infrared parts of one point's poles by component
    (evaluations with ``poles="uv"`` or ``"ir"``; design §12.4).

    ``uv[component]`` and ``ir[component]`` hold the component's
    :class:`LaurentCoefficients` on the evaluation's axes (per virtual
    sector; for resolved output per sector, helicity and virtual colour
    component): its finite part (the same under every pole selection) and
    that origin's single and double poles. Every pole is split as P/eps =
    P_UV/eps_UV + P_IR/eps_IR, so the two origins of each component add up
    to its poles under ``poles="all"`` (to rounding); the double pole is
    infrared. Both origins cost the same, so both are returned whichever of
    ``"uv"`` and ``"ir"`` was selected. The values of ``poles="uv"`` are
    ``uv["bare"]``'s poles (the unrenormalised amplitude), those of
    ``"ir"`` are ``ir["virtual"]``'s; the counterterms' UV poles are
    ``uv["uv"]`` (``uv_local`` plus ``uv_external_fields``) and ``uv["r2"]``.
    Components:

    * ``virtual``: the renormalised virtual, ``bare + uv + r2``. Its
      ultraviolet part is the bare loop's UV poles plus the counterterms'
      (zero up to rounding where renormalisation is complete); its infrared
      poles are those of ``poles="all"``, Catani's operator.
    * ``bare``: the bare loop. Its ultraviolet part is each reduction lane's
      UV residue (scaleless integrals included, B0(0;0,0) = 1/eps_UV -
      1/eps_IR), the infrared part the rest.
    * ``r2``: the rational terms (ultraviolet by origin; finite in
      ``loop_sm``).
    * ``uv_local``: the model's ultraviolet counterterms of couplings and
      masses (ultraviolet by origin).
    * ``uv_external_fields``: the on-shell external-field factors. A model
      writes them with one eps (a massless leg's Z_UV (1/eps_UV - 1/eps_IR)
      vanishes, a massive quark's mixes 1/eps_UV and 2/eps_IR) and does not
      state the origin, so their ultraviolet part is fixed by the
      ultraviolet finiteness of the renormalised virtual, ``-(bare +
      uv_local + r2)``: the bare-plus-counterterm UV cancellation holds by
      construction, and the field UV part is what to compare with a
      textbook field renormalisation (design §12.4).
    * A model that does state the origin, with pole regulators (MadLoop's
      electroweak UFOs: ``epsUV`` and ``epsIR`` at UFO/SLHA TECHNICAL 1 and
      2): the UV parts of ``r2``, ``uv_local`` and ``uv_external_fields``
      are their values with the IR regulator at 0 (MadLoop's ``epsIR = 0``
      reading) and their IR parts the rest, so the bare-plus-counterterm UV
      cancellation is a check, not a construction.
    * ``uv``: ``uv_local + uv_external_fields``.
    """

    uv: Mapping[str, tuple[Any, ...]]
    ir: Mapping[str, tuple[Any, ...]]

    def origin(self, origin: str) -> Mapping[str, tuple[Any, ...]]:
        if origin not in POLE_ORIGINS:
            raise ValueError(f"pole origin must be one of {POLE_ORIGINS}")
        return cast(Mapping[str, tuple[Any, ...]], getattr(self, origin))

    def mapped(self, transform: Callable[[tuple[Any, ...]], Any]) -> VirtualPoleParts:
        """The parts with every component's value replaced by
        ``transform(value)`` (e.g. one sector, or a sum over axes)."""

        return VirtualPoleParts(
            *(
                {name: transform(value) for name, value in self.origin(o).items()}
                for o in POLE_ORIGINS
            )
        )


def _pole_parts(
    parts: object,
    selection: PoleSelection,
    diagnostics: tuple[VirtualLaneDiagnostic, ...],
    dimensions: tuple[int, ...],
    correlated: bool,
) -> tuple[VirtualPoleParts | None, ...] | None:
    """Validated, frozen per-lane pole parts: present exactly for a split
    selection, ``None`` for failed lanes."""

    if not splits_poles(selection):
        if parts is not None:
            raise ValueError("pole parts exist exactly under poles 'uv' and 'ir'")
        return None
    if parts is None:
        raise ValueError("a uv or ir evaluation carries its pole parts")
    lanes = tuple(cast(Sequence[object], parts))
    if len(lanes) != len(diagnostics):
        raise ValueError("pole parts and diagnostics differ in lane count")
    out: list[VirtualPoleParts | None] = []
    for lane, diagnostic in zip(lanes, diagnostics, strict=True):
        if not diagnostic.valid:
            if lane is not None:
                raise ValueError("failed lanes carry no pole parts")
            out.append(None)
            continue
        if not isinstance(lane, VirtualPoleParts):
            raise TypeError("pole parts are VirtualPoleParts records")
        frozen = {}
        for origin in POLE_ORIGINS:
            components = lane.origin(origin)
            if set(components) != set(POLE_COMPONENTS):
                raise ValueError(f"pole parts name the components {POLE_COMPONENTS}")
            frozen[origin] = MappingProxyType(
                {
                    name: cast(
                        tuple[Any, ...],
                        _freeze_axes(
                            components[name],
                            dimensions,
                            correlated,
                            diagnostic.arithmetic,
                            True,
                        ),
                    )
                    for name in POLE_COMPONENTS
                }
            )
        out.append(VirtualPoleParts(frozen["uv"], frozen["ir"]))
    return tuple(out)


def _pole_part(
    parts: tuple[VirtualPoleParts | None, ...] | None, origin: str, component: str
) -> tuple[Any, ...]:
    if parts is None:
        raise ValueError(
            "only evaluations with poles 'uv' or 'ir' split the poles by origin"
        )
    if component not in POLE_COMPONENTS:
        raise ValueError(f"pole component must be one of {POLE_COMPONENTS}")
    return tuple(
        None if lane is None else lane.origin(origin)[component] for lane in parts
    )


def _one_sector(
    parts: tuple[VirtualPoleParts | None, ...] | None, index: int
) -> tuple[VirtualPoleParts | None, ...] | None:
    """Per-lane pole parts restricted to the sector ``index`` (``for_order``)."""

    if parts is None:
        return None
    return tuple(
        None if lane is None else lane.mapped(lambda value: (value[index],))
        for lane in parts
    )


def _freeze_axes(
    values: object,
    dimensions: tuple[int, ...],
    correlated: bool,
    arithmetic: str,
    laurent: bool = False,
    with_poles: bool = True,
) -> _FrozenAxis:
    if not dimensions:
        return _scalar(values, correlated, arithmetic)
    if not isinstance(values, (tuple, list)) or len(values) != dimensions[0]:
        raise ValueError("virtual values do not match their declared dimensions")
    if laurent and dimensions == (3,) and not with_poles:
        # Finite-only evaluations never computed the poles: explicitly absent.
        if values[1] is not None or values[2] is not None:
            raise ValueError("finite-only results carry no pole coefficients")
        absent = cast(VirtualScalar, None)
        return LaurentCoefficients(
            _scalar(values[0], correlated, arithmetic), absent, absent
        )
    frozen = tuple(
        _freeze_axes(value, dimensions[1:], correlated, arithmetic, laurent, with_poles)
        for value in values
    )
    # The terminal dimension was checked above and every child has passed the
    # scalar validator; only this validated terminal tuple becomes Laurent data.
    return (
        LaurentCoefficients(
            *cast(tuple[VirtualScalar, VirtualScalar, VirtualScalar], frozen)
        )
        if laurent and dimensions == (3,)
        else frozen
    )


def _sum(values: tuple[VirtualScalar, ...]) -> VirtualScalar:
    """Only output aggregation; numerical evaluation remains in native batches."""
    if not values:
        return 0.0
    kind = type(values[0])
    if any(type(value) is not kind for value in values):
        raise TypeError("cannot sum mixed arithmetic containers within one lane")
    if kind is Decimal:
        return _exact_decimal_sum(cast(tuple[Decimal, ...], values))
    if kind is CorrelatedValue:
        correlated = cast(tuple[CorrelatedValue, ...], values)
        return CorrelatedValue(
            _exact_decimal_sum(tuple(value.real for value in correlated)),
            _exact_decimal_sum(tuple(value.imag for value in correlated)),
        )
    if kind is complex:
        complex_values = cast(tuple[complex, ...], values)
        return complex(
            math.fsum(value.real for value in complex_values),
            math.fsum(value.imag for value in complex_values),
        )
    return math.fsum(cast(tuple[float, ...], values))


def _orders(
    born_orders: Sequence[CouplingOrders],
    sectors: Sequence[VirtualOrderSector],
    contribution: str = "virtual",
) -> tuple[tuple[CouplingOrders, ...], tuple[VirtualOrderSector, ...]]:
    born_orders = tuple(canonical_order(order) for order in born_orders)
    sectors = tuple(sectors)
    if contribution not in CONTRIBUTIONS:
        raise ValueError(f"unknown contribution {contribution!r}")
    if contribution == "loop-induced":
        # |A_loop|^2 (m-lit Q36): no Born, loop x loop sectors.
        if born_orders:
            raise ValueError("a loop-induced result has an empty Born catalogue")
        if not sectors:
            raise ValueError("virtual results require an interference catalogue")
    elif not born_orders or not sectors:
        raise ValueError("virtual results require Born and interference catalogues")
    if len(set(born_orders)) != len(born_orders):
        raise ValueError("duplicate Born squared orders")
    for sector in sectors:
        if not isinstance(sector, VirtualOrderSector) or not sector.pairs:
            raise ValueError("virtual sectors require contributing Born/loop pairs")
        if canonical_order(sector.order) != sector.order:
            raise ValueError("virtual sector orders must be canonical")
        if any(pair.squared != sector.order for pair in sector.pairs):
            raise ValueError("Born/loop pair does not match its squared sector")
    if len({sector.order for sector in sectors}) != len(sectors):
        raise ValueError("duplicate virtual squared orders")
    return born_orders, tuple(
        VirtualOrderSector(sector.order, tuple(sector.pairs)) for sector in sectors
    )


def _scalar_kinds(value: object) -> set[type[object]]:
    if isinstance(value, tuple):
        return set().union(*(_scalar_kinds(item) for item in value if item is not None))
    return {type(value)}


def _common(
    result: VirtualEvaluation | ResolvedVirtualEvaluation,
    born_dimensions: tuple[int, ...],
    virtual_dimensions: tuple[int, ...],
) -> None:
    if result.color_accuracy not in {"lc", "nlc", "full"}:
        raise ValueError("unknown colour accuracy")
    if not isinstance(result.correlated, bool):
        raise TypeError("correlated must be a boolean")
    object.__setattr__(result, "poles", pole_selection(result.poles))
    contribution = result.contribution
    born_orders, sectors = _orders(result.born_orders, result.sectors, contribution)
    if contribution == "loop-induced" and splits_poles(result.poles):
        # The UV and IR parts of |A_loop|^2 are not additive (m-lit PLAN
        # §2.a.7): a loop-induced result has every pole or none.
        raise ValueError(
            "a loop-induced result has the pole selection 'all' or 'none' "
            f"(got {result.poles.value!r})"
        )
    diagnostics = tuple(result.diagnostics)
    if not all(isinstance(item, VirtualLaneDiagnostic) for item in diagnostics):
        raise TypeError("virtual results require per-lane diagnostics")
    if len(result.born_values) != len(diagnostics) or len(result.values) != len(
        diagnostics
    ):
        raise ValueError(
            "virtual results and diagnostics must have the same lane count"
        )
    if not isinstance(result.r2, R2Selection) or result.r2.effective not in {
        R2Strategy.MODEL,
        R2Strategy.DIRECT,
    }:
        raise ValueError("virtual results require a resolved R2 strategy")
    frozen_born: list[_FrozenAxis | None] = []
    frozen_values: list[_FrozenAxis | None] = []
    for born, virtual, diagnostic in zip(
        result.born_values, result.values, diagnostics, strict=True
    ):
        if not diagnostic.valid:
            if born is not None or virtual is not None:
                raise ValueError("failed lanes must not expose stale numerical values")
            frozen_born.append(None)
            frozen_values.append(None)
            continue
        frozen_born.append(
            _freeze_axes(
                born, born_dimensions, result.correlated, diagnostic.arithmetic
            )
        )
        frozen_values.append(
            _freeze_axes(
                virtual,
                virtual_dimensions,
                result.correlated,
                diagnostic.arithmetic,
                True,
                computes_poles(result.poles),
            )
        )
        # (A loop-induced result's Born axes have zero length: no container.)
        checked = (
            (frozen_values[-1],)
            if contribution == "loop-induced"
            else (frozen_born[-1], frozen_values[-1])
        )
        if any(len(_scalar_kinds(value)) != 1 for value in checked):
            raise TypeError("a lane must use consistent Born and virtual containers")
    for name, value in (
        ("born_orders", born_orders),
        ("sectors", sectors),
        ("diagnostics", diagnostics),
        ("born_values", tuple(frozen_born)),
        ("values", tuple(frozen_values)),
    ):
        object.__setattr__(result, name, value)


def _stability(
    estimates: object,
    diagnostics: tuple[VirtualLaneDiagnostic, ...],
    with_poles: bool,
    rescue: tuple[VirtualRescueRecord, ...] | None = None,
) -> tuple[VirtualStabilityEstimate | None, ...] | None:
    """Per-lane estimates, or None when the evaluation ran no check (a
    rescued lane returned by a check-free last level has none either)."""
    if estimates is None:
        return None
    estimates = tuple(cast(Sequence[object], estimates))
    if len(estimates) != len(diagnostics):
        raise ValueError("stability estimates and diagnostics differ in lane count")
    unchecked = (
        [False] * len(diagnostics)
        if rescue is None
        else [record.attempts[-1].estimate is None for record in rescue]
    )
    for estimate, diagnostic, free in zip(
        estimates, diagnostics, unchecked, strict=True
    ):
        if estimate is None:
            if diagnostic.valid and not free:
                raise ValueError("a checked successful lane has an estimate")
            continue
        if not isinstance(estimate, VirtualStabilityEstimate):
            raise TypeError("stability estimates are VirtualStabilityEstimate records")
        if not diagnostic.valid:
            raise ValueError("failed lanes carry no stability estimate")
        if (estimate.single_pole is not None) != with_poles:
            raise ValueError("pole estimates exist exactly when poles were computed")
    return cast(tuple[VirtualStabilityEstimate | None, ...], estimates)


def _rescue(
    records: object, diagnostics: tuple[VirtualLaneDiagnostic, ...]
) -> tuple[VirtualRescueRecord, ...] | None:
    """Per-lane rescue records (one per lane), or None without a policy."""

    if records is None:
        return None
    records = tuple(cast(Sequence[object], records))
    if len(records) != len(diagnostics):
        raise ValueError("rescue records and diagnostics differ in lane count")
    for record, diagnostic in zip(records, diagnostics, strict=True):
        if not isinstance(record, VirtualRescueRecord):
            raise TypeError("rescue records are VirtualRescueRecord records")
        if record.failed == diagnostic.valid:
            raise ValueError("a rescue record fails exactly with its lane")
        if (record.arithmetic, record.precision_bits) != (
            diagnostic.arithmetic,
            diagnostic.precision_bits,
        ):
            raise ValueError("a lane's carrier is its rescue level's")
    return cast(tuple[VirtualRescueRecord, ...], records)


@dataclass(frozen=True, slots=True)
class VirtualEvaluation:
    """Born [point, order] and interference [point, order, epsilon].

    Ordinary interference is real and directed correlations may be complex.
    Failed lanes have ``None`` values, available only under the runtime's explicit
    diagnostic-return policy. Each returned coefficient includes physical
    couplings and normalization, without division by the Born amplitude.

    ``poles`` is the :class:`PoleSelection` of the evaluation; under
    ``"none"`` (finite only) the pole coefficients were never computed and
    are ``None``; under ``"uv"`` and ``"ir"`` they are the ultraviolet or
    infrared parts of the poles, and ``pole_parts`` holds per lane both
    origins' parts by component (:class:`VirtualPoleParts`; ``None`` for
    failed lanes and under ``"all"`` and ``"none"``). ``stability`` holds one
    :class:`VirtualStabilityEstimate` per lane (``None`` for failed lanes)
    when stability checks ran, and is ``None`` when none were requested.
    ``rescue`` holds one :class:`VirtualRescueRecord` per lane when the call
    ran a :class:`VirtualRescuePolicy` (``None`` otherwise); each lane's
    values, diagnostic, estimate and pole parts are then those of the level
    that returned it, and ``rescue_counts`` summarises the levels.

    ``contribution`` is ``"virtual"`` (the Born-virtual interference) or
    ``"loop-induced"`` (|A_loop|^2 of a process without a Born amplitude,
    m-lit Q36): its Born fields are zeroed (``born_orders`` empty, every
    per-point Born entry of zero length), its sectors pair loop orders (the
    pair's ``born`` slot holds the conjugated loop order) and its computed
    poles vanish up to rounding (a finiteness cross-check).
    """

    born_values: OrderCoefficients
    values: VirtualValues
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    diagnostics: tuple[VirtualLaneDiagnostic, ...]
    r2: R2Selection
    correlated: bool = False
    color_accuracy: Literal["lc", "nlc", "full"] = "full"
    poles: PoleSelection = PoleSelection.ALL
    stability: tuple[VirtualStabilityEstimate | None, ...] | None = None
    pole_parts: tuple[VirtualPoleParts | None, ...] | None = None
    rescue: tuple[VirtualRescueRecord, ...] | None = None
    contribution: Literal["virtual", "loop-induced"] = "virtual"

    def __post_init__(self) -> None:
        _common(self, (len(self.born_orders),), (len(self.sectors), 3))
        object.__setattr__(self, "rescue", _rescue(self.rescue, self.diagnostics))
        object.__setattr__(
            self,
            "stability",
            _stability(
                self.stability,
                self.diagnostics,
                computes_poles(self.poles),
                self.rescue,
            ),
        )
        object.__setattr__(
            self,
            "pole_parts",
            _pole_parts(
                self.pole_parts,
                self.poles,
                self.diagnostics,
                (len(self.sectors), 3),
                self.correlated,
            ),
        )

    @property
    def epsilon_powers(self) -> tuple[int, int, int]:
        return EPSILON_POWERS

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.values), len(self.sectors), 3

    @property
    def born_shape(self) -> tuple[int, int]:
        return len(self.born_values), len(self.born_orders)

    @property
    def valid(self) -> tuple[bool, ...]:
        return tuple(item.valid for item in self.diagnostics)

    @property
    def rescue_counts(self) -> dict[int | str, int] | None:
        """Lanes per rescue level whose estimate met the level's target
        (levels 0 to the highest reached), plus ``"not met"`` (lanes kept
        at the last level with the target missed) and ``"failed"`` (lanes
        that failed at every level); ``None`` without a rescue policy."""

        if self.rescue is None:
            return None
        top = max((record.level for record in self.rescue), default=0)
        counts: dict[int | str, int] = dict.fromkeys(range(top + 1), 0)
        counts["not met"] = counts["failed"] = 0
        for record in self.rescue:
            key: int | str = (
                "failed" if record.failed else record.level if record.met else "not met"
            )
            counts[key] += 1
        return counts

    def coefficient(self, epsilon_power: int) -> OrderCoefficients:
        if (
            isinstance(epsilon_power, bool)
            or not isinstance(epsilon_power, int)
            or epsilon_power not in EPSILON_POWERS
        ):
            raise ValueError("epsilon power must be 0, -1, or -2")
        if epsilon_power and not computes_poles(self.poles):
            raise ValueError("this finite-only evaluation computed no poles")
        index = EPSILON_POWERS.index(epsilon_power)
        return tuple(
            None if point is None else tuple(sector[index] for sector in point)
            for point in self.values
        )

    @property
    def finite(self) -> OrderCoefficients:
        return self.coefficient(0)

    @property
    def single_pole(self) -> OrderCoefficients:
        return self.coefficient(-1)

    @property
    def double_pole(self) -> OrderCoefficients:
        return self.coefficient(-2)

    def pole_part(self, origin: str, component: str = "virtual") -> VirtualValues:
        """One origin's (``"uv"`` or ``"ir"``) Laurent coefficients of one
        component (``POLE_COMPONENTS``) per lane and sector, from an
        evaluation with ``poles="uv"`` or ``"ir"`` (``None`` for failed
        lanes). ``pole_part("uv", "bare")`` holds the poles of the
        ``"uv"`` values, ``pole_part("uv", "uv")`` the counterterms' UV
        poles, ``pole_part("ir")`` the poles of the ``"ir"`` values."""

        return cast(VirtualValues, _pole_part(self.pole_parts, origin, component))

    def for_order(self, order: CouplingOrders) -> VirtualEvaluation:
        wanted = canonical_order(order)
        for index, sector in enumerate(self.sectors):
            if sector.order == wanted:
                return replace(
                    self,
                    sectors=(sector,),
                    values=tuple(
                        None if point is None else (point[index],)
                        for point in self.values
                    ),
                    pole_parts=_one_sector(self.pole_parts, index),
                )
        raise ValueError(
            f"squared coupling order {wanted!r} is absent from the catalogue"
        )

    def sum_orders(self) -> tuple[LaurentCoefficients | None, ...]:
        """Explicitly sum only the virtual order axis; keep all coefficients."""
        return tuple(
            None
            if point is None
            else LaurentCoefficients(
                *(
                    _sum(tuple(sector[epsilon] for sector in point))
                    if epsilon == 0 or computes_poles(self.poles)
                    else cast(VirtualScalar, None)
                    for epsilon in range(3)
                )
            )
            for point in self.values
        )


@dataclass(frozen=True, slots=True)
class ResolvedVirtualEvaluation:
    """Interference [point, order, helicity, virtual colour component, epsilon].

    Born colour components have their own independent axis. LC components are
    signed interference contributions, never sampling probabilities.

    ``poles`` and ``pole_parts`` follow :class:`VirtualEvaluation` (a
    finite-only evaluation has explicitly absent poles; under ``"uv"`` and
    ``"ir"`` each lane's :class:`VirtualPoleParts` hold every component per
    sector, helicity and virtual colour component). When stability checks
    ran, ``stability``
    holds per lane one :class:`VirtualStabilityEstimate` per resolved entry,
    indexed ``[helicity][virtual colour component]``, each relative to its own
    entry (``None`` for failed lanes); it is ``None`` when no check was
    requested. An entry that vanishes structurally (a helicity row whose Born
    amplitude is zero) holds rounding noise, so its relative estimate is
    large (e.g. 1e4): compare such entries on the point's scale.

    ``contribution`` follows :class:`VirtualEvaluation`: a loop-induced
    result (``"loop-induced"``) has no Born colour axis (``born_color_ids``
    empty, every Born entry of zero length) and its ``virtual_color_ids``
    resolve the colour-contracted |A_loop|^2 (m-lit Q36).
    """

    born_values: ResolvedCoefficients
    values: ResolvedVirtualValues
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    diagnostics: tuple[VirtualLaneDiagnostic, ...]
    r2: R2Selection
    helicity_ids: tuple[str, ...]
    born_color_ids: tuple[str, ...]
    virtual_color_ids: tuple[str, ...]
    color_accuracy: Literal["lc", "nlc", "full"]
    correlated: bool = False
    poles: PoleSelection = PoleSelection.ALL
    stability: (
        tuple[tuple[tuple[VirtualStabilityEstimate, ...], ...] | None, ...] | None
    ) = None
    pole_parts: tuple[VirtualPoleParts | None, ...] | None = None
    contribution: Literal["virtual", "loop-induced"] = "virtual"

    def __post_init__(self) -> None:
        if self.color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError("unknown colour accuracy")
        for name in ("helicity_ids", "born_color_ids", "virtual_color_ids"):
            ids = tuple(getattr(self, name))
            # A loop-induced result has no Born colour axis (m-lit Q36).
            empty = name == "born_color_ids" and self.contribution == "loop-induced"
            if (
                bool(ids) == empty
                or len(set(ids)) != len(ids)
                or not all(isinstance(i, str) and i for i in ids)
            ):
                raise ValueError(
                    "a loop-induced result has no Born colour IDs"
                    if empty
                    else "resolved axis IDs must be nonempty and unique"
                )
            object.__setattr__(self, name, ids)
        _common(
            self,
            (len(self.born_orders), len(self.helicity_ids), len(self.born_color_ids)),
            (len(self.sectors), len(self.helicity_ids), len(self.virtual_color_ids), 3),
        )
        object.__setattr__(self, "stability", self._resolved_stability())
        object.__setattr__(
            self,
            "pole_parts",
            _pole_parts(
                self.pole_parts,
                self.poles,
                self.diagnostics,
                (
                    len(self.sectors),
                    len(self.helicity_ids),
                    len(self.virtual_color_ids),
                    3,
                ),
                self.correlated,
            ),
        )

    def _resolved_stability(
        self,
    ) -> tuple[tuple[tuple[VirtualStabilityEstimate, ...], ...] | None, ...] | None:
        """Per-lane [helicity][colour] estimates, or None without checks."""
        if self.stability is None:
            return None
        lanes = tuple(cast(Sequence[object], self.stability))
        if len(lanes) != len(self.diagnostics):
            raise ValueError("stability estimates and diagnostics differ in lane count")
        out: list[tuple[tuple[VirtualStabilityEstimate, ...], ...] | None] = []
        for lane, diagnostic in zip(lanes, self.diagnostics, strict=True):
            if lane is None:
                if diagnostic.valid:
                    raise ValueError("a checked successful lane has estimates")
                out.append(None)
                continue
            if not diagnostic.valid:
                raise ValueError("failed lanes carry no stability estimate")
            rows = tuple(tuple(row) for row in cast(Sequence[Sequence[object]], lane))
            if len(rows) != len(self.helicity_ids) or any(
                len(row) != len(self.virtual_color_ids) for row in rows
            ):
                raise ValueError("resolved estimates follow the helicity/colour axes")
            for row in rows:
                for estimate in row:
                    _stability((estimate,), (diagnostic,), computes_poles(self.poles))
            out.append(cast(tuple[tuple[VirtualStabilityEstimate, ...], ...], rows))
        return tuple(out)

    @property
    def epsilon_powers(self) -> tuple[int, int, int]:
        return EPSILON_POWERS

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (
            len(self.values),
            len(self.sectors),
            len(self.helicity_ids),
            len(self.virtual_color_ids),
            3,
        )

    @property
    def born_shape(self) -> tuple[int, int, int, int]:
        return (
            len(self.born_values),
            len(self.born_orders),
            len(self.helicity_ids),
            len(self.born_color_ids),
        )

    @property
    def valid(self) -> tuple[bool, ...]:
        return tuple(item.valid for item in self.diagnostics)

    def coefficient(self, epsilon_power: int) -> ResolvedCoefficients:
        if (
            isinstance(epsilon_power, bool)
            or not isinstance(epsilon_power, int)
            or epsilon_power not in EPSILON_POWERS
        ):
            raise ValueError("epsilon power must be 0, -1, or -2")
        if epsilon_power and not computes_poles(self.poles):
            raise ValueError("this finite-only evaluation computed no poles")
        index = EPSILON_POWERS.index(epsilon_power)
        return tuple(
            None
            if point is None
            else tuple(
                tuple(
                    tuple(colour[index] for colour in helicity) for helicity in sector
                )
                for sector in point
            )
            for point in self.values
        )

    @property
    def finite(self) -> ResolvedCoefficients:
        return self.coefficient(0)

    @property
    def single_pole(self) -> ResolvedCoefficients:
        return self.coefficient(-1)

    @property
    def double_pole(self) -> ResolvedCoefficients:
        return self.coefficient(-2)

    def pole_part(
        self, origin: str, component: str = "virtual"
    ) -> ResolvedVirtualValues:
        """One origin's Laurent coefficients of one component per lane,
        sector, helicity and virtual colour component (``poles="uv"`` or
        ``"ir"``; :meth:`VirtualEvaluation.pole_part`)."""

        return cast(
            ResolvedVirtualValues, _pole_part(self.pole_parts, origin, component)
        )

    def for_order(self, order: CouplingOrders) -> ResolvedVirtualEvaluation:
        wanted = canonical_order(order)
        for index, sector in enumerate(self.sectors):
            if sector.order == wanted:
                return replace(
                    self,
                    sectors=(sector,),
                    values=tuple(
                        None if point is None else (point[index],)
                        for point in self.values
                    ),
                    pole_parts=_one_sector(self.pole_parts, index),
                )
        raise ValueError(
            f"squared coupling order {wanted!r} is absent from the catalogue"
        )

    def total(self) -> VirtualEvaluation:
        """Sum helicities and colour components, preserving orders and epsilon.

        The sum carries no stability estimate: per-entry estimates do not
        bound a sum whose entries cancel (the inclusive evaluation returns
        its own estimate).
        """
        born = tuple(
            None
            if point is None
            else tuple(
                _sum(tuple(value for helicity in sector for value in helicity))
                for sector in point
            )
            for point in self.born_values
        )
        with_poles = computes_poles(self.poles)

        def summed(
            point: tuple[tuple[tuple[LaurentCoefficients, ...], ...], ...],
        ) -> tuple[LaurentCoefficients, ...]:
            return tuple(
                LaurentCoefficients(
                    *(
                        _sum(
                            tuple(
                                colour[epsilon]
                                for helicity in sector
                                for colour in helicity
                            )
                        )
                        if epsilon == 0 or with_poles
                        else cast(VirtualScalar, None)
                        for epsilon in range(3)
                    )
                )
                for sector in point
            )

        values = tuple(
            None if point is None else summed(point) for point in self.values
        )
        return VirtualEvaluation(
            born,
            values,
            self.born_orders,
            self.sectors,
            self.diagnostics,
            self.r2,
            self.correlated,
            color_accuracy=self.color_accuracy,
            poles=self.poles,
            pole_parts=None
            if self.pole_parts is None
            else tuple(
                None if parts is None else parts.mapped(summed)
                for parts in self.pole_parts
            ),
            contribution=self.contribution,
        )
