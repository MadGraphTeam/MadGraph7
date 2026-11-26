# SPDX-License-Identifier: 0BSD
"""Exponent-safe magnitudes of the stability estimates of arbitrary precision.

A stability estimate compares sizes: absolute budgets E, the scale S of what
cancelled and the value itself, against each other and against a target.
In binary64 and double-double those sizes are binary64 floats. An
arbitrary-precision carrier reaches far below binary64's exponent range: at
the default 500 digits (``loops.v2.reduction.default_arbitrary_digits``) the
unit roundoff is about 1e-500 and a rounding-level budget is of that order,
so as floats both are 0.0, every estimate reads 0.0 and an integrated zero
(g g > z: |A|^2 ~ 1e-1039, E ~ 1e-500 S) reads "met" through 0 / 0.

``Wide`` is a float mantissa with an unbounded integer binary exponent
(``m * 2**e``, ``0.5 <= |m| < 1`` or ``m == 0``): binary64's 53 bits of
significance, which is all a size needs, at any exponent. It mixes with
floats and ints in ``+ - * /``, ``**`` (a real exponent) and comparisons, so
the estimate code is shared between the carriers: binary64 and double-double
keep their floats (bitwise unchanged), an arbitrary-precision carrier forms
its magnitudes as ``Wide`` (``magnitude``) and every ratio compared with a
target is formed exactly before ``float`` rounds it (a ratio below binary64's
range is 0.0, one above it inf: both read right against a target).
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import Any

# log2(10), to place a Decimal's binary exponent before its exact rescaling.
_LOG2_10 = math.log2(10.0)
# The smallest normal binary64: inside [_NORMAL, inf) a value's float is
# correctly rounded, and ``Wide`` takes it as it is (bitwise the float path).
_NORMAL = 2.0**-1022


def _normal(value: float) -> bool:
    return math.isfinite(value) and abs(value) >= _NORMAL


class Wide:
    """``m * 2**e``: a binary64 mantissa with an unbounded exponent (module
    docstring)."""

    __slots__ = ("e", "m")

    def __init__(self, m: float = 0.0, e: int = 0) -> None:
        if m == 0.0 or not math.isfinite(m):
            self.m, self.e = float(m), 0
            return
        mantissa, shift = math.frexp(m)
        self.m, self.e = mantissa, e + shift

    # -- construction --------------------------------------------------------

    @classmethod
    def of(cls, value: Any) -> Wide:
        """A ``Wide`` of a float, int, Decimal, numeric text or ``Wide``."""

        if isinstance(value, Wide):
            return value
        if isinstance(value, (float, int)):
            return cls(float(value))
        if isinstance(value, str):
            value = Decimal(value)
        if isinstance(value, Decimal):
            if not value.is_finite():
                return cls(float(value))
            if not value:
                return cls(0.0)
            rounded = float(value)
            if _normal(rounded):
                # Inside binary64's range: its correctly rounded float.
                return cls(rounded)
            shift = math.floor(value.adjusted() * _LOG2_10)
            with localcontext() as context:
                context.prec = 30
                context.Emax = 10**9
                context.Emin = -(10**9)
                scaled = value.scaleb(0) / (Decimal(2) ** shift)
            return cls(float(scaled), shift)
        return cls(float(value))

    # -- conversion ----------------------------------------------------------

    def __float__(self) -> float:
        if self.m == 0.0 or not math.isfinite(self.m):
            return self.m
        try:
            return math.ldexp(self.m, self.e)
        except OverflowError:
            return math.copysign(math.inf, self.m)

    def __bool__(self) -> bool:
        return self.m != 0.0

    def __repr__(self) -> str:
        if self.m == 0.0 or not math.isfinite(self.m):
            return f"Wide({self.m!r})"
        exponent = self.e * math.log10(2.0) + math.log10(abs(self.m))
        whole = math.floor(exponent)
        lead = math.copysign(10.0 ** (exponent - whole), self.m)
        return f"Wide({lead:.6g}e{whole:+d})"

    def __format__(self, spec: str) -> str:
        return format(float(self), spec)

    # -- arithmetic ----------------------------------------------------------

    def __neg__(self) -> Wide:
        out = Wide.__new__(Wide)
        out.m, out.e = -self.m, self.e
        return out

    def __abs__(self) -> Wide:
        out = Wide.__new__(Wide)
        out.m, out.e = abs(self.m), self.e
        return out

    def __add__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        if not other.m:
            return self
        if not self.m:
            return other
        if not (math.isfinite(self.m) and math.isfinite(other.m)):
            return Wide(self.m + other.m)
        if self.e >= other.e:
            return Wide(self.m + math.ldexp(other.m, other.e - self.e), self.e)
        return Wide(other.m + math.ldexp(self.m, self.e - other.e), other.e)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        if not (math.isfinite(self.m) and math.isfinite(other.m)):
            return Wide(self.m * other.m)
        return Wide(self.m * other.m, self.e + other.e)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        if not other.m:
            raise ZeroDivisionError("Wide division by zero")
        if not (math.isfinite(self.m) and math.isfinite(other.m)):
            return Wide(self.m / other.m)
        return Wide(self.m / other.m, self.e - other.e)

    def __rtruediv__(self, other: Any) -> Wide:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self

    def __pow__(self, power: float) -> Wide:
        if not self.m or not math.isfinite(self.m):
            return Wide(self.m**power)
        if self.m < 0:
            # A negative base: an integer power only (magnitudes never need
            # one; float would give a complex number).
            if not float(power).is_integer():
                raise ValueError("a negative Wide takes an integer power only")
            out = abs(self) ** power
            return -out if int(power) % 2 else out
        rounded = float(self)
        if _normal(rounded):
            try:
                found = rounded**power
            except OverflowError:
                # (binary64's power overflows: the general path below.)
                found = math.inf
            if _normal(found):
                # Inside binary64's range: binary64's own power (libm pow),
                # bitwise the float path.
                return Wide(found)
        if power == 0.5:
            # The square root, correctly rounded (an even exponent halves
            # exactly).
            if self.e % 2:
                return Wide(math.sqrt(2.0 * self.m), (self.e - 1) // 2)
            return Wide(math.sqrt(self.m), self.e // 2)
        # log2 of the value, split so that no float overflows.
        exponent = (self.e + math.log2(self.m)) * float(power)
        whole = math.floor(exponent)
        return Wide(2.0 ** (exponent - whole), whole)

    # -- comparison ----------------------------------------------------------

    def _key(self) -> tuple[int, float, float]:
        """An order key: sign, then (for nonzero) exponent and mantissa."""

        if self.m == 0.0:
            return (0, 0.0, 0.0)
        if not math.isfinite(self.m):
            return (1 if self.m > 0 else -1, self.m, 0.0)
        sign = 1 if self.m > 0 else -1
        return (sign, sign * float(self.e), sign * abs(self.m))

    def _compare(self, other: Any) -> int:
        other = _wide(other)
        if other is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        if math.isnan(self.m) or math.isnan(other.m):
            return 2  # unordered
        a, b = self._key(), other._key()
        if a[0] != b[0]:
            return -1 if a[0] < b[0] else 1
        if a == b:
            return 0
        return -1 if a < b else 1

    def __eq__(self, other: object) -> bool:
        found = self._compare(other)
        if found is NotImplemented:
            return NotImplemented
        return found == 0

    def __hash__(self) -> int:
        # Equal values hash equal: a ``Wide`` inside binary64's range as its
        # float.
        rounded = float(self)
        if Wide(rounded) == self or not math.isfinite(self.m):
            return hash(rounded)
        return hash((self.m, self.e))

    def __lt__(self, other: Any) -> bool:
        found = self._compare(other)
        if found is NotImplemented:
            return NotImplemented
        return found == -1

    def __le__(self, other: Any) -> bool:
        found = self._compare(other)
        if found is NotImplemented:
            return NotImplemented
        return found in (-1, 0)

    def __gt__(self, other: Any) -> bool:
        found = self._compare(other)
        if found is NotImplemented:
            return NotImplemented
        return found == 1

    def __ge__(self, other: Any) -> bool:
        found = self._compare(other)
        if found is NotImplemented:
            return NotImplemented
        return found in (1, 0)


def _wide(value: Any) -> Any:
    if isinstance(value, Wide):
        return value
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return Wide(float(value))
    return NotImplemented


def magnitude(value: Any, wide: bool) -> Any:
    """|value| as a stability estimate holds it: a binary64 float (``wide``
    False: binary64 and double-double, ``float(abs(value))`` as before) or a
    ``Wide`` (arbitrary precision). A complex Decimal pair (``rx.Complex``,
    whose ``abs`` is a binary64 float) is measured as sqrt(re^2 + im^2) in
    ``Wide``."""

    if not wide:
        return float(abs(value))
    if isinstance(value, Wide):
        return abs(value)
    real = getattr(value, "real", None)
    imag = getattr(value, "imag", None)
    if (
        not isinstance(value, (Decimal, float, int, str))
        and real is not None
        and imag is not None
    ):
        rounded = float(abs(value))
        if _normal(rounded):
            # Inside binary64's range: the carrier's own modulus, as before.
            return Wide(rounded)
        re, im = Wide.of(real), Wide.of(imag)
        if not im:
            return abs(re)
        if not re:
            return abs(im)
        return (re * re + im * im) ** 0.5
    return abs(Wide.of(value))


def ratio(error: Any, size: Any) -> float:
    """``error / size`` as a binary64 float (0 for an exact zero ``error``,
    inf for ``error > 0`` at ``size == 0``), formed exactly before rounding
    when either is a ``Wide``."""

    if not error:
        return 0.0
    if not size:
        return math.inf
    return float(error / size)


def is_wide(values: Any) -> bool:
    """Whether any of ``values`` is a ``Wide``."""

    return any(isinstance(value, Wide) for value in values)


__all__ = ["Wide", "is_wide", "magnitude", "ratio"]
