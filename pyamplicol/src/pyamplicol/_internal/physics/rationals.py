# SPDX-License-Identifier: 0BSD
"""Simplest rationals (Stern-Brocot) without a denominator cap.

A binary64 number written by a model (``2/3`` -> ``0.6666666666666666``) or a
measured ratio is read as the simplest rational -- the smallest denominator,
then the smallest numerator -- inside the interval of reals it stands for:
the float's rounding interval, or a tolerance interval. The interval alone
decides; no fixed bound on the numerator or the denominator.
"""

from __future__ import annotations

import math
from fractions import Fraction


def simplest_rational_between(low: Fraction, high: Fraction) -> Fraction:
    """The simplest rational in the closed interval ``[low, high]``."""

    low, high = Fraction(low), Fraction(high)
    if low > high:
        raise ValueError(f"empty rational interval [{low}, {high}]")
    if low <= 0 <= high:
        return Fraction(0)
    if high < 0:
        return -_simplest_positive(-high, -low)
    return _simplest_positive(low, high)


def simplest_rational_for_float(value: float) -> Fraction:
    """The simplest rational strictly inside the rounding interval of the
    finite binary64 ``value`` (between the midpoints to the neighbouring
    floats), so ``float(result) == value`` always holds; an integer-valued
    float is that integer (beyond 2^53 the interval holds several)."""

    if not math.isfinite(value):
        raise ValueError(f"{value!r} is not a finite binary64 number")
    magnitude = abs(value)
    exact = Fraction(magnitude)
    if exact.denominator == 1:
        return Fraction(int(value))
    below = math.nextafter(magnitude, 0.0)
    above = math.nextafter(magnitude, math.inf)
    low = (exact + Fraction(below)) / 2
    high = (
        (exact + Fraction(above)) / 2
        if math.isfinite(above)
        else exact + Fraction(math.ulp(magnitude)) / 2
    )
    result = _simplest_inside(low, high)
    if float(result) != magnitude:  # pragma: no cover - interval arithmetic
        raise ArithmeticError(f"no rational inside the rounding interval of {value!r}")
    return result if value > 0 else -result


def _simplest_positive(low: Fraction, high: Fraction) -> Fraction:
    """The simplest rational of the closed ``[low, high]``, ``0 < low``."""

    # x = a0 + 1/(a1 + 1/(...)): the common partial quotients of both
    # endpoints, closed by the smallest integer between them.
    terms: list[int] = []
    while True:
        floor = low.numerator // low.denominator
        if floor == low:
            terms.append(floor)
            break
        if floor + 1 <= high:
            terms.append(floor + 1)
            break
        terms.append(floor)
        low, high = 1 / (high - floor), 1 / (low - floor)
    return _from_terms(terms)


def _simplest_inside(low: Fraction, high: Fraction | None) -> Fraction:
    """The simplest rational of the open ``(low, high)``, ``0 <= low < high``;
    ``high`` None is infinity."""

    terms: list[int] = []
    while True:
        floor = low.numerator // low.denominator
        if high is None or floor + 1 < high:
            terms.append(floor + 1)
            break
        # No integer inside: x = floor + 1/y with y in the image interval.
        terms.append(floor)
        low, high = 1 / (high - floor), None if low == floor else 1 / (low - floor)
    return _from_terms(terms)


def _from_terms(terms: list[int]) -> Fraction:
    result = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        result = term + 1 / result
    return result


__all__ = ["simplest_rational_between", "simplest_rational_for_float"]
