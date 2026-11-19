# SPDX-License-Identifier: 0BSD
"""Exponent-safe stability magnitudes (``loops.v2.wide``): a 500-digit
estimate's sizes lie far below binary64's range; ``Wide`` keeps them, mixes
with floats, and every ratio rounds to binary64 last. Binary64 and
double-double keep their floats (``magnitude(x, False)`` is
``float(abs(x))``)."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.v2.wide import Wide, is_wide, magnitude, ratio


@pytest.mark.loop_fast
def test_wide_keeps_sizes_beyond_binary64():
    with localcontext() as context:
        context.prec = 50
        tiny = Decimal("5.2277113914463723585e-1039")
        error = Decimal("1.0e-500")
    a, e = Wide.of(tiny), Wide.of(error)
    assert float(a) == 0.0 and float(e) == 0.0  # binary64 loses both
    assert a and e and a < e and e > a and a != e
    # Their ratio is formed exactly, then rounded: above binary64's range it
    # is inf, below it 0.0, inside it the float.
    assert ratio(e, a) == math.inf and ratio(a, e) == 0.0
    assert math.isclose(float(e / Wide.of(Decimal("2e-500"))), 0.5)
    assert math.isclose(float((e * 1e300) / (a * 1e300) * a / e), 1.0)
    # Mixed with floats and ints, as the estimate code mixes them.
    floor = 16 * Wide.of(Decimal("5e-500")) * 0.25
    assert math.isclose(float(floor / Wide.of(Decimal("2e-499"))), 1.0)
    total = sum([0.0, e, 2 * e])
    assert isinstance(total, Wide) and math.isclose(float(total / e), 3.0)
    assert max(0.0, e) is e and e > 0.0 and e <= e and not (e < 0)
    assert math.isclose(float((e * e) ** 0.5 / e), 1.0)
    assert float(Wide.of(1e300) * Wide.of(1e300)) == math.inf
    assert format(Wide.of(0.25), ".2e") == "2.50e-01"
    assert ratio(0.0, e) == 0.0 and ratio(e, 0.0) == math.inf
    assert is_wide([1.0, e]) and not is_wide([1.0, 2.0])


@pytest.mark.loop_fast
def test_magnitude_is_float_abs_outside_arbitrary_precision():
    for value in (3.25, -1e-300, Decimal("-2.5e-17"), 7):
        found = magnitude(value, False)
        assert type(found) is float and found == float(abs(value))
    found = magnitude(Decimal("-2.5e-700"), True)
    assert isinstance(found, Wide) and math.isclose(
        float(found / Wide.of(Decimal("1e-700"))), 2.5
    )

    class Pair:  # a complex Decimal pair (``rx.Complex``: a binary64 abs)
        real = Decimal("3e-800")
        imag = Decimal("-4e-800")

        def __abs__(self) -> float:
            return 0.0  # below binary64's range

    modulus = magnitude(Pair(), True)
    assert math.isclose(float(modulus / Wide.of(Decimal("1e-800"))), 5.0)
    # Inside binary64's range ``Wide`` is the float path, bitwise.
    for value in (Decimal("0.1"), Decimal("-3.3e-200"), Decimal("7e250")):
        assert Wide.of(value) == float(value)
        assert float(Wide.of(value)) == float(value)
    x, y = Decimal("0.1"), Decimal("3.7e-5")
    assert float(Wide.of(x) * Wide.of(y) + 2.5 * Wide.of(x)) == (
        float(x) * float(y) + 2.5 * float(x)
    )
    assert float(Wide.of(y) ** 0.5) == math.sqrt(float(y))


@pytest.mark.loop_fast
def test_the_wide_guard_mirror_is_the_native_rule(loop_native):
    """The fallback guard's exponent-safe mirror (``reduction.
    _wide_fallback_estimate``, every arbitrary-precision carrier) is the
    native rule (``reduction_policy::fallback_estimate``) wherever binary64
    holds the inputs: bitwise on random outputs (ragged sectors, vanishing
    outputs at the floor, zeros, NaN)."""

    import random

    from pyamplicol.loops.v2.reduction import _wide_fallback_estimate

    native = loop_native.reduction_fallback_estimate
    rng = random.Random(5)
    for _ in range(4000):
        u = rng.choice([2.0**-53, 2.0**-104, 0.5e-56])
        outputs = []
        for _o in range(rng.randint(1, 4)):
            sectors = []
            for _s in range(rng.randint(1, 3)):
                scale = 10 ** rng.uniform(-30, 5)
                kind = rng.random()
                size = (
                    0.0
                    if kind < 0.1
                    else scale * u * rng.uniform(0, 20)
                    if kind < 0.3
                    else 10 ** rng.uniform(-40, 5)
                )
                error = 0.0 if rng.random() < 0.1 else 10 ** rng.uniform(-60, 0) * scale
                if rng.random() < 0.01:
                    error = math.nan
                sectors.append((size, error, scale))
            outputs.append(sectors)
        expected = float(native(outputs, u))
        found = _wide_fallback_estimate(
            [[tuple(Wide(x) for x in sector) for sector in out] for out in outputs],
            Wide(u),
        )
        assert (math.isnan(expected) and math.isnan(found)) or found == expected


@pytest.mark.loop_fast
def test_wide_edges():
    """Powers match binary64's own inside its range (libm pow, as the float
    path), a negative base takes an integer power with its exponent, a
    foreign type is not compared silently, equal values hash equal."""

    # Values where libm pow and the correctly rounded sqrt differ by 1 ulp.
    for value in (8.522513224083059e-17, 629422423279058.1):
        assert math.sqrt(value) != value**0.5
        assert float(Wide(value) ** 0.5) == value**0.5
    # binary64's power overflows: the exponent-safe path takes over.
    assert math.isclose(float(Wide(1e200) ** 2 / Wide(1e200) / Wide(1e200)), 1.0)
    assert math.isclose(float(Wide(2.0) ** 2000 / Wide(2.0) ** 1999), 2.0)
    assert hash(Wide(1e-310)) == hash(1e-310) and Wide(1e-310) == 1e-310
    assert math.isclose(float(Wide(-1.0, -3000) ** 2 / Wide(1.0, -6000)), 1.0)
    assert float(Wide(-2.0) ** 3) == -8.0
    with pytest.raises(ValueError):
        Wide(-4.0) ** 0.5
    with pytest.raises(TypeError):
        Wide(1.0) < Decimal(2)  # noqa: B015
    assert hash(Wide(1.0)) == hash(1.0) and hash(Wide(0.0)) == hash(0.0)
    assert {Wide(0.25): 1}[0.25] == 1
