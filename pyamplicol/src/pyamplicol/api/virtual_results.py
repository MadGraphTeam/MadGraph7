# SPDX-License-Identifier: 0BSD
"""Immutable Born-loop results; no implicit order sum or finite-only coercion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal, NamedTuple, TypeAlias, cast

from pyamplicol.config import R2Strategy
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.loops.orders import VirtualOrderSector, canonical_order
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
from pyamplicol.models.base import CouplingOrders

from .results import _exact_decimal_sum

VirtualScalar: TypeAlias = float | complex | Decimal | CorrelatedValue
EPSILON_POWERS = (0, -1, -2)


class LaurentCoefficients(NamedTuple):
    """Physical coefficients in the declared C_epsilon convention."""

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


@dataclass(frozen=True, slots=True)
class VirtualStabilityPolicy:
    """Opt-in native verification/rescue with exact decimal agreement bounds.

    The native boundary checks strictly increasing actual carrier bit precision.
    Rescue requires two successful levels and verified output conversion.
    """

    higher: tuple[VirtualArithmeticSpec, ...]
    relative_tolerance: str
    absolute_tolerance: str
    verify_successful: bool = False

    def __post_init__(self) -> None:
        levels = tuple(self.higher)
        if not levels or any(not isinstance(x, VirtualArithmeticSpec) for x in levels):
            raise ValueError("higher requires explicit arithmetic specifications")
        object.__setattr__(self, "higher", levels)
        for value in (self.relative_tolerance, self.absolute_tolerance):
            if not isinstance(value, str):
                raise TypeError("agreement tolerances must be decimal strings")
            number = Decimal(value)
            if not number.is_finite() or number < 0:
                raise ValueError("agreement tolerances must be finite and nonnegative")
        if Decimal(self.relative_tolerance) == Decimal(self.absolute_tolerance) == 0:
            raise ValueError("agreement tolerances cannot both be zero")
        if type(self.verify_successful) is not bool:
            raise TypeError("verify_successful must be a boolean")


def _default_rotation(index: int) -> Rotation:
    """The ``index``-th default rotation check: exact, deterministic, generic.

    Axes follow a spherical Fibonacci sequence and angles golden-ratio steps
    of a turn, so no two checks share an axis or a rotation angle and none is
    a symmetry of a typical point. ``Rotation.from_axis_angle`` rounds the
    quaternion to rationals, which keeps every rotated point exactly on shell.
    """

    golden = (1 + math.sqrt(5)) / 2
    z = 1 - (2 * index + 1) / (2 * (index + 2))
    phi = 2 * math.pi * index / golden
    radius = math.sqrt(1 - z * z)
    angle = 2 * math.pi * (((index + 1) / golden) % 1)
    return Rotation.from_axis_angle(
        (radius * math.cos(phi), radius * math.sin(phi), z), max(angle, 0.5)
    )


def parse_rotation(specification: str) -> Rotation:
    """One rotation check from text.

    ``"<x|y|z>:<cos>:<sin>"`` is exact (rational cos and sin with
    cos^2 + sin^2 = 1, e.g. ``"z:3/5:4/5"``); ``"<ax>,<ay>,<az>:<radians>"``
    rotates about an axis by an angle, rounded to an exact rotation
    (``Rotation.from_axis_angle``).
    """

    if not isinstance(specification, str):
        raise TypeError("a rotation specification is text")
    parts = specification.split(":")
    if len(parts) == 3 and parts[0] in ("x", "y", "z"):
        return Rotation.about(parts[0], parts[1], parts[2])
    if len(parts) == 2:
        axis = tuple(float(value) for value in parts[0].split(","))
        return Rotation.from_axis_angle(axis, float(parts[1]))
    raise ValueError(
        f"rotation {specification!r} is neither '<x|y|z>:<cos>:<sin>' "
        "nor '<ax>,<ay>,<az>:<radians>'"
    )


@dataclass(frozen=True, slots=True)
class VirtualStabilityChecks:
    """Stability checks of one evaluation (requirements addendum C).

    ``rotations`` re-evaluates every point with its momenta rotated by each
    rotation (exact ``Rotation`` records, specification text for
    :func:`parse_rotation`, or a count of default rotations); the
    reduction-direction check reduces every loop numerator a second time in a
    reversed frame, reusing its coefficients and the cached master integrals.
    Both combine. Without a check no estimate is returned. Escalating the
    arithmetic is the caller's decision.
    """

    rotations: tuple[Rotation, ...] | int = ()
    reduction_direction: bool = False

    def __post_init__(self) -> None:
        rotations = self.rotations
        if isinstance(rotations, bool):
            raise TypeError("rotations take a count or rotation records")
        if isinstance(rotations, int):
            if rotations < 0:
                raise ValueError("the rotation count is nonnegative")
            rotations = tuple(_default_rotation(k) for k in range(rotations))
        else:
            rotations = tuple(
                parse_rotation(item) if isinstance(item, str) else item
                for item in rotations
            )
        if type(self.reduction_direction) is not bool:
            raise TypeError("reduction_direction must be a boolean")
        object.__setattr__(
            self,
            "rotations",
            StabilityChecks(rotations, self.reduction_direction).rotations,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.rotations) or self.reduction_direction

    def to_checks(self) -> StabilityChecks:
        return StabilityChecks(
            cast(tuple[Rotation, ...], self.rotations), self.reduction_direction
        )


@dataclass(frozen=True, slots=True)
class VirtualStabilityEstimate:
    """Estimated relative accuracy of one point's virtual.

    ``relative`` bounds each sector's error relative to that sector's largest
    Laurent coefficient; ``finite``, ``single_pole`` and ``double_pole``
    bound each coefficient's error relative to the coefficient itself (the
    poles are ``None`` in finite-only evaluations; ``inf`` marks an error
    estimate on an exact zero). ``checks`` names every check that ran with
    its own relative budget (``rotation-<k>``, ``reduction-direction``).
    """

    relative: float
    finite: float
    single_pole: float | None
    double_pole: float | None
    checks: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for name in ("relative", "finite", "single_pole", "double_pole"):
            value = getattr(self, name)
            if value is None and name in ("single_pole", "double_pole"):
                continue
            if not isinstance(value, float) or math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative float")
        if (self.single_pole is None) != (self.double_pole is None):
            raise ValueError("both pole estimates are present or absent")
        checks = tuple((str(name), float(value)) for name, value in self.checks)
        if not checks or len({name for name, _ in checks}) != len(checks):
            raise ValueError("an estimate names its distinct checks")
        object.__setattr__(self, "checks", checks)


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


def _freeze_axes(
    values: object,
    dimensions: tuple[int, ...],
    correlated: bool,
    arithmetic: str,
    laurent: bool = False,
    poles: bool = True,
) -> _FrozenAxis:
    if not dimensions:
        return _scalar(values, correlated, arithmetic)
    if not isinstance(values, (tuple, list)) or len(values) != dimensions[0]:
        raise ValueError("virtual values do not match their declared dimensions")
    if laurent and dimensions == (3,) and not poles:
        # Finite-only evaluations never computed the poles: explicitly absent.
        if values[1] is not None or values[2] is not None:
            raise ValueError("finite-only results carry no pole coefficients")
        return LaurentCoefficients(
            _scalar(values[0], correlated, arithmetic), None, None
        )
    frozen = tuple(
        _freeze_axes(value, dimensions[1:], correlated, arithmetic, laurent, poles)
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
    born_orders: Sequence[CouplingOrders], sectors: Sequence[VirtualOrderSector]
) -> tuple[tuple[CouplingOrders, ...], tuple[VirtualOrderSector, ...]]:
    born_orders = tuple(canonical_order(order) for order in born_orders)
    sectors = tuple(sectors)
    if not born_orders or not sectors:
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
    if type(result.poles) is not bool:
        raise TypeError("poles must be a boolean")
    born_orders, sectors = _orders(result.born_orders, result.sectors)
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
                result.poles,
            )
        )
        if any(
            len(_scalar_kinds(value)) != 1
            for value in (frozen_born[-1], frozen_values[-1])
        ):
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
    estimates: object, diagnostics: tuple[VirtualLaneDiagnostic, ...], poles: bool
) -> tuple[VirtualStabilityEstimate | None, ...] | None:
    """Per-lane estimates, or None when the evaluation ran no check."""
    if estimates is None:
        return None
    estimates = tuple(cast(Sequence[object], estimates))
    if len(estimates) != len(diagnostics):
        raise ValueError("stability estimates and diagnostics differ in lane count")
    for estimate, diagnostic in zip(estimates, diagnostics, strict=True):
        if estimate is None:
            if diagnostic.valid:
                raise ValueError("a checked successful lane has an estimate")
            continue
        if not isinstance(estimate, VirtualStabilityEstimate):
            raise TypeError("stability estimates are VirtualStabilityEstimate records")
        if not diagnostic.valid:
            raise ValueError("failed lanes carry no stability estimate")
        if (estimate.single_pole is not None) != poles:
            raise ValueError("pole estimates exist exactly when poles were computed")
    return cast(tuple[VirtualStabilityEstimate | None, ...], estimates)


@dataclass(frozen=True, slots=True)
class VirtualEvaluation:
    """Born [point, order] and interference [point, order, epsilon].

    Ordinary interference is real and directed correlations may be complex.
    Failed lanes have ``None`` values, available only under the runtime's explicit
    diagnostic-return policy. Each returned coefficient includes physical
    couplings and normalization, without division by the Born amplitude.

    ``poles`` is False for a finite-only evaluation: its pole coefficients
    were never computed and are ``None``. ``stability`` holds one
    :class:`VirtualStabilityEstimate` per lane (``None`` for failed lanes)
    when stability checks ran, and is ``None`` when none were requested.
    """

    born_values: OrderCoefficients
    values: VirtualValues
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    diagnostics: tuple[VirtualLaneDiagnostic, ...]
    r2: R2Selection
    correlated: bool = False
    color_accuracy: Literal["lc", "nlc", "full"] = "full"
    poles: bool = True
    stability: tuple[VirtualStabilityEstimate | None, ...] | None = None

    def __post_init__(self) -> None:
        _common(self, (len(self.born_orders),), (len(self.sectors), 3))
        object.__setattr__(
            self, "stability", _stability(self.stability, self.diagnostics, self.poles)
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

    def coefficient(self, epsilon_power: int) -> OrderCoefficients:
        if (
            isinstance(epsilon_power, bool)
            or not isinstance(epsilon_power, int)
            or epsilon_power not in EPSILON_POWERS
        ):
            raise ValueError("epsilon power must be 0, -1, or -2")
        if epsilon_power and not self.poles:
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
                    if epsilon == 0 or self.poles
                    else None
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

    def __post_init__(self) -> None:
        if self.color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError("unknown colour accuracy")
        for name in ("helicity_ids", "born_color_ids", "virtual_color_ids"):
            ids = tuple(getattr(self, name))
            if (
                not ids
                or len(set(ids)) != len(ids)
                or not all(isinstance(i, str) and i for i in ids)
            ):
                raise ValueError("resolved axis IDs must be nonempty and unique")
            object.__setattr__(self, name, ids)
        _common(
            self,
            (len(self.born_orders), len(self.helicity_ids), len(self.born_color_ids)),
            (len(self.sectors), len(self.helicity_ids), len(self.virtual_color_ids), 3),
        )

    @property
    def poles(self) -> bool:
        """Resolved evaluations always carry every Laurent coefficient."""
        return True

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
                )
        raise ValueError(
            f"squared coupling order {wanted!r} is absent from the catalogue"
        )

    def total(self) -> VirtualEvaluation:
        """Sum helicities and colour components, preserving orders and epsilon."""
        born = tuple(
            None
            if point is None
            else tuple(
                _sum(tuple(value for helicity in sector for value in helicity))
                for sector in point
            )
            for point in self.born_values
        )
        values = tuple(
            None
            if point is None
            else tuple(
                LaurentCoefficients(
                    *(
                        _sum(
                            tuple(
                                colour[epsilon]
                                for helicity in sector
                                for colour in helicity
                            )
                        )
                        for epsilon in range(3)
                    )
                )
                for sector in point
            )
            for point in self.values
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
        )
