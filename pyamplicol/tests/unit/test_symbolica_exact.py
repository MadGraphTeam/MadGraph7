# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import ROUND_UP, Decimal, localcontext

from pyamplicol.runtime.symbolica_exact import (
    _ExactEvaluator,
    _upcast_decimal,
    _working_precision,
)


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.values: object = None
        self.precision: int | None = None

    def evaluate_complex_with_prec(
        self, values: object, precision: int
    ) -> list[tuple[Decimal, Decimal]]:
        self.values = values
        self.precision = precision
        return [(Decimal("1.25"), Decimal("-0.5"))]


def test_upcast_decimal_preserves_value_and_carries_requested_precision() -> None:
    cases = (
        Decimal("500"),
        Decimal("0.00123"),
        Decimal("1e100"),
        Decimal("-7.25"),
    )

    for value in cases:
        upcast = _upcast_decimal(value, 40)
        assert upcast == value
        assert len(upcast.as_tuple().digits) == 40

    zero = _upcast_decimal(Decimal("0"), 40)
    assert zero == 0
    assert zero.as_tuple().exponent == -40


def test_exact_evaluator_upcasts_every_complex_input() -> None:
    recording = _RecordingEvaluator()
    evaluator = _ExactEvaluator((recording,))

    result = evaluator.evaluate(
        ((Decimal("500"), Decimal("0")), (Decimal("0.125"), Decimal("-2"))),
        80,
    )

    assert result == ((Decimal("1.25"), Decimal("-0.5")),)
    assert recording.precision == 80
    values = recording.values
    assert isinstance(values, tuple)
    for real, imaginary in values:
        assert isinstance(real, Decimal)
        assert isinstance(imaginary, Decimal)
        assert len(real.as_tuple().digits) == 80 or real.is_zero()
        assert len(imaginary.as_tuple().digits) == 80 or imaginary.is_zero()


def test_upcast_rounding_does_not_depend_on_ambient_decimal_context() -> None:
    with localcontext() as context:
        context.rounding = ROUND_UP
        assert _upcast_decimal(Decimal("1.25"), 2) == Decimal("1.2")


def test_every_exact_request_stays_above_symbolica_binary64_shortcut() -> None:
    assert all(_working_precision(precision) >= 40 for precision in range(1, 40))
    assert _working_precision(80) == 88
