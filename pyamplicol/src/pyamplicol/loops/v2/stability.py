# SPDX-License-Identifier: 0BSD
"""Stability estimates returned with an evaluation (REQUIREMENTS_ADDENDUM C).

Two MadLoop-style checks, individually configurable and combinable:

* **Rotation checks.** The point is evaluated again with its external momenta
  rotated by each user-supplied rotation (any number, including none). A
  rotation is an exact rational orthogonal matrix, so the rotated point is
  exactly on shell and every reduction lane's integral (a Lorentz scalar) is
  unchanged in exact arithmetic; the differences measure the rounding of the
  whole evaluation (trees, dressing, reduction, counterterms).
* **Reduction-direction check.** Every family lane is reduced a second time
  with the loop direction reversed (``V2Reducer.reduce(check=True)``, the
  NinjaRS adapter's reversed frame), reusing the numerator coefficients and
  the cached masters. It measures the reduction alone, at the cost of a
  second reduction pass.
* **Rescaled reduction check** (``StabilityChecks(rescaled=True)``). The
  same re-reduction with every dimensionful input scaled by a non-dyadic
  factor (``ReductionFrame::RESCALED``): the masters are evaluated anew at
  differently rounded arguments instead of hitting the cache, so a master
  integral's own rounding (e.g. a binary64 box near a vanishing Gram
  determinant) is measured too. Neither the cached re-reduction nor a
  rotation can see it: both keep every invariant, so every master returns
  the same value. It costs the master evaluations of a second pass.

Every check yields an absolute error budget per squared-order sector and
Laurent coefficient that lets no two contributions cancel (design §10.2):
sum_f |Omega_f - Omega'_f| over the reduction lanes (bare part), plus, for a
rotation, the deviations of the local components (R2, UV vertex, fields). A
single difference of two evaluations can be accidentally small (its ratio to
the true error has a heavy tail), the budget over many lanes cannot.

The estimate of a configuration combines its checks per sector and Laurent
coefficient as

    E = SAFETY x (max over rotations of their budgets + reduction budget)
        + FLOOR_UNITS x u x S,

with u the unit roundoff of the carrier and S the absolute sum of the
contributions (sum_f |bare_f| plus the absolute local components): the
factor 2 covers a difference of two evaluations with comparable rounding,
which understates either one's error by up to about 2, and the floor covers
roundings no check samples (couplings, kernel constants, the final sums, and
the master integrals unless the rescaled check runs).

Calibration against Float(256) (DESIGN §10.4, O15: 2000 binary64 and 400
double-double points each of gg -> ggg and uu~ -> e-e+dd~g) gives the rate at
which the true error exceeds E in binary64: about 2.5 % for the cached
reduction check alone (it cannot see the masters' own rounding), up to 0.6 %
for one rotation, 0.2 % for the rescaled check and at most 0.1 % for the
rescaled check with a rotation, the recommended configuration (none of the
800 double-double points).

``StabilityEstimate.relative`` measures E against each sector's largest
Laurent coefficient (the comparison metric of the oracles); ``coefficients``
measures it against each coefficient itself, so ``finite`` is the relative
accuracy of the finite part. Without a requested check no estimate exists:
``LoopSlice.evaluate`` then returns ``stability=None``. Escalating the
precision is the caller's decision.

``RECOMMENDED_CHECKS`` is that configuration with ``default_rotation(0)``
(a generic axis, as calibrated); it is the estimate of the O1 rule. Two
weaker choices miss errors by more than ten times their estimate (package
mq-stab, binary64 against double-double):

* A rotation about the beam axis (z for the collision-frame points of a
  2 -> n process) leaves both incoming momenta and the energy and p_z of
  every momentum bitwise unchanged. With the cached reduction check it
  reported 7.6e-14 for the double pole of u u~ > u u~ e+ e- at the fixture's
  point 2, whose error is 1.7e-12 (22.8 x); any rotation moving the beams
  brackets it (error 0.5 x the estimate). The masters are not the cause
  there: every rotation keeps their exact invariants and so their binary64
  inputs bitwise.
* Without the rescaled check no master is evaluated twice: at
  g g > b b~ h (4FS, m_b = 4.7) the finite part's binary64 error sits in
  the masters with the light b and is 23-96 x the estimate of every
  rotation with the cached check; the rescaled check brings it to
  0.14-0.24 x (7.5 x at the worst point).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

SAFETY = 2
FLOOR_UNITS = 16
ROTATION = "rotation-{}"
REDUCTION = "reduction-direction"
RESCALED = "reduction-rescaled"
# The re-reduction frame of each reduction check (``V2Reducer.reduce``).
FRAMES = {REDUCTION: "reversed", RESCALED: "rescaled"}

_AXES = {"x": 1, "y": 2, "z": 3}


@dataclass(frozen=True, slots=True)
class Rotation:
    """A spatial rotation as an exact rational orthogonal 3x3 matrix."""

    matrix: tuple[tuple[Fraction, Fraction, Fraction], ...]

    def __post_init__(self) -> None:
        matrix = tuple(tuple(Fraction(x) for x in row) for row in self.matrix)
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("a rotation is a 3x3 matrix")
        identity = tuple(
            tuple(sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        if identity != ((1, 0, 0), (0, 1, 0), (0, 0, 1)) or _determinant(matrix) != 1:
            raise ValueError("a rotation matrix must be exactly orthogonal, det +1")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def about(cls, axis: str, cos: Any, sin: Any) -> Rotation:
        """Rotation about ``axis`` ('x', 'y' or 'z') with exact cos and sin
        (e.g. 3/5 and 4/5); cos^2 + sin^2 must be exactly 1."""

        c, s = Fraction(cos), Fraction(sin)
        if c * c + s * s != 1:
            raise ValueError("cos^2 + sin^2 must be exactly 1")
        index = _AXES[axis] - 1
        a, b = [k for k in range(3) if k != index]
        rows = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
        rows[a][a], rows[a][b], rows[b][a], rows[b][b] = c, -s, s, c
        return cls(tuple(tuple(row) for row in rows))

    @classmethod
    def from_axis_angle(
        cls, axis: Sequence[float], angle: float, *, resolution: int = 10**6
    ) -> Rotation:
        """An exact rotation within ~1/``resolution`` of ``angle`` (radians,
        right-handed) about the direction ``axis``.

        The unit quaternion (cos(angle/2), sin(angle/2) n) is rounded to
        rationals with denominators up to ``resolution``; the Euler-Rodrigues
        matrix of any rational quaternion is exactly orthogonal, so the rotated
        point stays exactly on shell.
        """

        norm = math.sqrt(sum(float(x) ** 2 for x in axis))
        if len(axis) != 3 or not norm or not math.isfinite(angle):
            raise ValueError("a rotation needs a non-zero 3-axis and a finite angle")
        half = angle / 2
        a, b, c, d = (
            Fraction(value).limit_denominator(resolution)
            for value in (
                math.cos(half),
                *(math.sin(half) * float(x) / norm for x in axis),
            )
        )
        n = a * a + b * b + c * c + d * d
        if not n:
            raise ValueError("the rounded quaternion vanishes; raise the resolution")
        rows = (
            (a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),
            (2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)),
            (2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d),
        )
        return cls(tuple(tuple(x / n for x in row) for row in rows))

    @property
    def is_identity(self) -> bool:
        return self.matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __matmul__(self, other: Rotation) -> Rotation:
        return Rotation(
            tuple(
                tuple(
                    sum(self.matrix[i][k] * other.matrix[k][j] for k in range(3))
                    for j in range(3)
                )
                for i in range(3)
            )
        )

    def apply(
        self, point: Sequence[Sequence[Fraction]]
    ) -> tuple[tuple[Fraction, ...], ...]:
        """Rotate the spatial parts of every momentum (E, px, py, pz)."""

        out = []
        for momentum in point:
            energy, *space = (Fraction(x) for x in momentum)
            rotated = [
                sum(self.matrix[i][k] * space[k] for k in range(3)) for i in range(3)
            ]
            out.append((energy, *rotated))
        return tuple(out)


def default_rotation(index: int) -> Rotation:
    """The ``index``-th default rotation check: exact, deterministic, generic.

    Axes follow a spherical Fibonacci sequence and angles golden-ratio steps
    of a turn, so no two checks share an axis or a rotation angle, none is a
    coordinate axis (the beam axis included: module docstring) and none is
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


def _determinant(m) -> Fraction:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


@dataclass(frozen=True, slots=True)
class StabilityChecks:
    """Which checks an evaluation runs: rotations (any number) and/or one
    reduction check, the reduction-direction check (cached masters) or, with
    ``rescaled``, its rescaled variant (masters recomputed). The default runs
    none."""

    rotations: tuple[Rotation, ...] = ()
    reduction_direction: bool = False
    rescaled: bool = False

    def __post_init__(self) -> None:
        rotations = tuple(self.rotations)
        if any(not isinstance(r, Rotation) for r in rotations):
            raise TypeError("rotation checks take Rotation records")
        if any(r.is_identity for r in rotations):
            raise ValueError("the identity is no rotation check")
        if self.rescaled and not self.reduction_direction:
            raise ValueError("rescaled is a variant of the reduction-direction check")
        object.__setattr__(self, "rotations", rotations)

    @property
    def enabled(self) -> bool:
        return bool(self.rotations) or self.reduction_direction

    @property
    def reduction(self) -> str | None:
        """The name of the reduction check that runs (``REDUCTION`` or
        ``RESCALED``), or None."""
        if not self.reduction_direction:
            return None
        return RESCALED if self.rescaled else REDUCTION

    @property
    def frame(self) -> str | None:
        """The re-reduction frame of the reduction check, or None."""
        return FRAMES.get(self.reduction)


# The recommended configuration (DESIGN §10.4) and the O1 rule's estimate:
# one generic rotation and the rescaled reduction check (module docstring).
RECOMMENDED_CHECKS = StabilityChecks(
    (default_rotation(0),), reduction_direction=True, rescaled=True
)


def relative(reference: Mapping[Any, Any], absolute: Mapping[Any, Any]) -> float:
    """Largest absolute deviation relative to the size of its Laurent vector.

    ``reference`` maps sectors to ``Laurent`` values and ``absolute`` to
    ``Laurent`` absolute deviations. Each sector's deviations are measured
    against its largest coefficient (the comparison metric of the oracles), so
    a pole that nearly cancels between components does not turn rounding of
    its parts into an arbitrarily large relative error.
    """

    worst = 0.0
    for sector, laurent in reference.items():
        scale = max(abs(float(x)) for x in laurent.coefficients)
        errors = absolute[sector].coefficients
        if scale:
            worst = max(worst, max(abs(float(x)) for x in errors) / scale)
    return worst


def per_coefficient(
    reference: Mapping[Any, Any], absolute: Mapping[Any, Any]
) -> tuple[float, ...]:
    """Largest absolute deviation of each Laurent coefficient relative to the
    coefficient itself, over sectors (``inf`` for a deviation of an exact 0)."""

    width = len(next(iter(reference.values())).coefficients)
    worst = [0.0] * width
    for sector, laurent in reference.items():
        for k, (value, error) in enumerate(
            zip(laurent.coefficients, absolute[sector].coefficients, strict=True)
        ):
            value, error = abs(float(value)), abs(float(error))
            if error:
                worst[k] = max(worst[k], error / value if value else math.inf)
    return tuple(worst)


def deviation(reference: Mapping[Any, Any], other: Mapping[Any, Any]) -> float:
    """Relative deviation of ``other``'s Laurent coefficients from the
    ``reference`` ones over sectors (``relative`` of their differences)."""

    return relative(reference, differences(reference, other))


def differences(reference: Mapping[Any, Any], other: Mapping[Any, Any]) -> dict:
    """|other - reference| per sector and Laurent coefficient."""

    return {
        sector: Budget(
            tuple(
                abs(b - a)
                for a, b in zip(
                    laurent.coefficients, other[sector].coefficients, strict=True
                )
            )
        )
        for sector, laurent in reference.items()
    }


@dataclass(frozen=True, slots=True)
class Budget:
    """Absolute errors per Laurent coefficient (a ``Laurent``-like record)."""

    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class StabilityEstimate:
    """The relative accuracy estimate of one evaluation (module docstring).

    ``relative`` is E against each sector's largest Laurent coefficient,
    ``coefficients`` E per Laurent coefficient against that coefficient
    (finite, then the poles when computed), ``absolute`` E itself per sector
    and Laurent coefficient. ``checks`` holds each check's own budget against
    the Laurent vector, without safety factor and floor. ``budgets``
    (absolute, per check and sector) and ``scale`` (the absolute sums S) let
    ``subset`` re-combine fewer checks; ``families`` holds |Omega - Omega'|
    per reduction lane when the reduction check ran.

    A relative estimate of order 1 or more says that the value has no
    correct digit: it is then measured against a value that is itself
    wrong, and only ``absolute`` keeps its meaning.
    """

    relative: float
    coefficients: tuple[float, ...]
    checks: Mapping[str, float]
    absolute: Mapping[Any, Budget] = field(repr=False)
    budgets: Mapping[str, Mapping[Any, Budget]] = field(repr=False)
    reference: Mapping[Any, Any] = field(repr=False)
    scale: Mapping[Any, Budget] = field(repr=False)
    roundoff: float = field(repr=False)
    families: Mapping[Any, Any] | None = field(default=None, repr=False)

    @property
    def finite(self) -> float:
        """The relative accuracy estimate of the finite part."""
        return self.coefficients[0]

    @classmethod
    def from_checks(
        cls,
        reference: Mapping[Any, Any],
        budgets: Mapping[str, Mapping[Any, Any]],
        *,
        scale: Mapping[Any, Any],
        roundoff: float,
        families: Mapping[Any, Any] | None = None,
    ) -> StabilityEstimate:
        """Combine the absolute budgets of named checks (``ROTATION`` and
        ``REDUCTION`` names) for the reference Laurent values per sector;
        ``scale`` are the absolute sums S and ``roundoff`` the carrier's unit
        roundoff."""

        if not budgets:
            raise ValueError("a stability estimate needs at least one check")
        unknown = [
            name
            for name in budgets
            if name not in FRAMES and not name.startswith(ROTATION.format(""))
        ]
        if unknown:
            raise ValueError(f"unknown stability checks {unknown}")
        rotations = [b for name, b in budgets.items() if name not in FRAMES]
        reductions = [b for name, b in budgets.items() if name in FRAMES]
        combined = {}
        for sector, laurent in reference.items():
            width = len(laurent.coefficients)

            def sampled(k: int, sector=sector) -> float:
                rotated = max(
                    (_at(r[sector], k) for r in rotations),
                    default=0.0,
                )
                return rotated + sum(_at(r[sector], k) for r in reductions)

            combined[sector] = Budget(
                tuple(
                    SAFETY * sampled(k) + FLOOR_UNITS * roundoff * _at(scale[sector], k)
                    for k in range(width)
                )
            )
        return cls(
            relative(reference, combined),
            per_coefficient(reference, combined),
            {name: relative(reference, item) for name, item in budgets.items()},
            combined,
            dict(budgets),
            reference,
            scale,
            roundoff,
            families,
        )

    def subset(self, names: Iterable[str]) -> StabilityEstimate:
        """The estimate an evaluation with only the checks ``names`` returns."""

        names = tuple(names)
        missing = [name for name in names if name not in self.budgets]
        if missing:
            raise KeyError(f"checks {missing} did not run")
        return StabilityEstimate.from_checks(
            self.reference,
            {name: self.budgets[name] for name in names},
            scale=self.scale,
            roundoff=self.roundoff,
            families=self.families if set(names) & set(FRAMES) else None,
        )


def _at(budget: Any, k: int) -> float:
    return abs(float(budget.coefficients[k]))


__all__ = [
    "FLOOR_UNITS",
    "FRAMES",
    "RECOMMENDED_CHECKS",
    "REDUCTION",
    "RESCALED",
    "ROTATION",
    "SAFETY",
    "Budget",
    "Rotation",
    "StabilityChecks",
    "StabilityEstimate",
    "default_rotation",
    "deviation",
    "differences",
    "per_coefficient",
    "relative",
]
