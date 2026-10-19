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


@dataclass(frozen=True, slots=True)
class VirtualVerificationAttempt:
    arithmetic: VirtualArithmeticSpec
    precision_bits: int
    numerical_success: bool
    compared_precision_bits: int | None = None
    agreement: bool | None = None
    conversion_agreement: bool | None = None
    failures: tuple[VirtualFailureDiagnostic, ...] = ()


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
) -> _FrozenAxis:
    if not dimensions:
        return _scalar(values, correlated, arithmetic)
    if not isinstance(values, (tuple, list)) or len(values) != dimensions[0]:
        raise ValueError("virtual values do not match their declared dimensions")
    frozen = tuple(
        _freeze_axes(value, dimensions[1:], correlated, arithmetic, laurent)
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
        return set().union(*(_scalar_kinds(item) for item in value))
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


@dataclass(frozen=True, slots=True)
class VirtualEvaluation:
    """Born [point, order] and interference [point, order, epsilon].

    Ordinary interference is real and directed correlations may be complex.
    Failed lanes have ``None`` values, available only under the runtime's explicit
    diagnostic-return policy. Each returned coefficient includes physical
    couplings and normalization, without division by the Born amplitude.
    """

    born_values: OrderCoefficients
    values: VirtualValues
    born_orders: tuple[CouplingOrders, ...]
    sectors: tuple[VirtualOrderSector, ...]
    diagnostics: tuple[VirtualLaneDiagnostic, ...]
    r2: R2Selection
    correlated: bool = False
    color_accuracy: Literal["lc", "nlc", "full"] = "full"

    def __post_init__(self) -> None:
        _common(self, (len(self.born_orders),), (len(self.sectors), 3))

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
