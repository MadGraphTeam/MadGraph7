# SPDX-License-Identifier: 0BSD
"""Stability estimates returned with an evaluation (REQUIREMENTS_ADDENDUM C).

Two MadLoop-style checks, individually configurable and combinable:

* **Rotation checks.** The point is evaluated again with its external momenta
  rotated by each user-supplied rotation (any number, including none). A
  rotation is an exact rational orthogonal matrix, so the rotated point is
  exactly on shell and the virtual is unchanged in exact arithmetic; the
  difference measures the rounding of the whole evaluation (trees, dressing,
  reduction, counterterms).
* **Reduction-direction check.** Every family lane is reduced a second time
  with the loop direction reversed (``V2Reducer.reduce(check=True)``, the
  NinjaRS adapter's reversed frame), reusing the numerator coefficients and
  the cached masters. It measures the reduction alone, at the cost of a
  second reduction pass.

The estimate of a check is the largest deviation of the virtual's Laurent
coefficients (the finite part only, in finite-only mode) relative to the
sector's largest coefficient, over every squared-order sector: |V_rotated - V|
for a rotation, and for the reduction direction the error budget
sum_f |Omega_f - Omega'_f| over the families (design §10.2). The combined
estimate adds, coefficient by coefficient, the largest rotation deviation and
the reduction budget: the rotations sample the rounding of the whole
evaluation and the budget bounds the reduction's, and one difference of two
evaluations with comparable rounding can underestimate either evaluation's
error (NinjaRS's reversed frame tracks the true error within a factor of
about 2). Roundings common to every evaluation (couplings, kernel constants,
the final sums) are sampled by no check, so the combined estimate also carries
a floor of a few units in the last place of the carrier. Without a requested
check no estimate exists: ``LoopSlice.evaluate`` then returns
``stability=None``.
Escalating the precision is the caller's decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

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


def _determinant(m) -> Fraction:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


@dataclass(frozen=True, slots=True)
class StabilityChecks:
    """Which checks an evaluation runs: rotations (any number) and/or the
    reduction-direction check. The default runs none."""

    rotations: tuple[Rotation, ...] = ()
    reduction_direction: bool = False

    def __post_init__(self) -> None:
        rotations = tuple(self.rotations)
        if any(not isinstance(r, Rotation) for r in rotations):
            raise TypeError("rotation checks take Rotation records")
        object.__setattr__(self, "rotations", rotations)

    @property
    def enabled(self) -> bool:
        return bool(self.rotations) or self.reduction_direction


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


def deviation(reference: Mapping[Any, Any], other: Mapping[Any, Any]) -> float:
    """Relative deviation of ``other``'s Laurent coefficients from the
    ``reference`` ones over sectors (``relative`` of their differences)."""

    return relative(
        reference,
        {
            sector: _Deviation(
                tuple(
                    b - a
                    for a, b in zip(
                        laurent.coefficients, other[sector].coefficients, strict=True
                    )
                )
            )
            for sector, laurent in reference.items()
        },
    )


@dataclass(frozen=True, slots=True)
class _Deviation:
    coefficients: tuple[Any, ...]


@dataclass(frozen=True)
class StabilityEstimate:
    """The relative accuracy estimate of one evaluation.

    ``relative`` is the combined estimate (module docstring) and ``checks``
    the individual ones, each the largest absolute deviation of the virtual's
    Laurent coefficients relative to the sector's largest one (``relative``):
    for a rotation, |V_rotated - V|; for the reduction direction, the error
    budget sum_f |Omega_f - Omega'_f| of the families' contributions (design
    §10.2), which lets no two families' errors cancel. ``families`` holds
    |Omega - Omega'| per reduction lane when that check ran.
    """

    relative: float
    checks: Mapping[str, float]
    families: Mapping[Any, Any] | None = field(default=None, repr=False)

    @classmethod
    def from_checks(
        cls,
        reference: Mapping[Any, Any],
        rotations: Sequence[Mapping[Any, Any]] = (),
        reduction: Mapping[Any, Any] | None = None,
        families: Mapping[Any, Any] | None = None,
        floor: float = 0.0,
    ) -> StabilityEstimate:
        """Combine absolute Laurent deviations per sector: one mapping per
        rotation and the reduction budget; ``floor`` is the carrier's
        relative rounding floor."""

        checks = {f"rotation-{i}": item for i, item in enumerate(rotations)}
        if reduction is not None:
            checks["reduction-direction"] = reduction
        if not checks:
            raise ValueError("a stability estimate needs at least one check")
        combined = {}
        for sector, laurent in reference.items():
            width = len(laurent.coefficients)
            rotated = [
                max(
                    (abs(float(r[sector].coefficients[k])) for r in rotations),
                    default=0.0,
                )
                for k in range(width)
            ]
            budget = (
                [abs(float(x)) for x in reduction[sector].coefficients]
                if reduction is not None
                else [0.0] * width
            )
            combined[sector] = _Deviation(
                tuple(a + b for a, b in zip(rotated, budget, strict=True))
            )
        return cls(
            relative(reference, combined) + floor,
            {name: relative(reference, item) for name, item in checks.items()},
            families,
        )


__all__ = [
    "Rotation",
    "StabilityChecks",
    "StabilityEstimate",
    "deviation",
    "relative",
]
