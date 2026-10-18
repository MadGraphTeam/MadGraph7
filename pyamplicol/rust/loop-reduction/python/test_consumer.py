# SPDX-License-Identifier: 0BSD
"""Real Python -> shared decoder -> native batch -> NinjaRS conformance."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import CoefficientEvaluator, PolynomialPlan
from symbolica import E, S


def saved_coefficient():
    x = S("loop_native_coefficient_x")
    return (x - E("1")).evaluator([x], n_cores=1, jit_compile=False).save()


@pytest.mark.parametrize("precision,bits", [(16, 53), (32, 106), (80, 266)])
def test_saved_coefficients_reduce_in_native_batches(precision, bits):
    coefficients = CoefficientEvaluator(saved_coefficient(), precision)
    plan = PolynomialPlan(1, ["c", "2", "0"], ["c"], precision)
    assert coefficients.precision_bits == plan.precision_bits == bits
    with localcontext() as context:
        context.prec = 180
        tiny = Decimal(2) ** -(80 if precision == 32 else 180)
        tolerance = Decimal(10) ** -(12 if precision == 16 else precision - 5)
        for size in [1, 33, 8, 0, 1]:
            expected = [
                Decimal(row + 1) / 4 if precision == 16 else tiny * (row + 1)
                for row in range(size)
            ]
            inputs = [[str(1 + value), "0"] for value in expected]
            values = coefficients.evaluate_decimal(inputs, size)
            ids = [700 - row * 3 for row in range(size)]
            result = plan.evaluate_decimal(
                ids,
                [[["0", "0"] for _ in range(4)] for _ in ids],
                [["1", "0"] for _ in ids],
                [["1", "0"] for _ in ids],
                values,
            )
            assert [row[0] for row in result] == ids
            for (_, status, poles), value in zip(result, expected, strict=True):
                assert status == "success"
                assert poles is not None
                # A0(1) = 1 + 1/epsilon, hence N0=c,N1=2 gives
                # finite=c+2, single pole=c. No physical prefactors are implied.
                finite, single, double = [Decimal(pair[0]) for pair in poles]
                assert abs(finite - value - 2) < tolerance
                assert abs(single - value) < tolerance
                if precision != 16:
                    # Detect any binary64 conversion of 1+tiny at any boundary.
                    assert single != 0
                    assert abs(single / value - 1) < Decimal("0.00001")
                assert double == 0
                assert all(Decimal(pair[1]) == 0 for pair in poles)


@pytest.mark.parametrize("precision", [16, 32, 80])
def test_explicit_mu2_triangle_crosses_python_boundary(precision):
    plan = PolynomialPlan(3, ["2*mu2", "0", "0"], [], precision)
    shifts = [
        [["0", "0"] for _ in range(4)],
        [["-0.5", "0"], ["0", "0"], ["0", "0"], ["-0.5", "0"]],
        [["-1", "0"], ["0", "0"], ["0", "0"], ["0", "0"]],
    ]
    result = plan.evaluate_decimal(
        [91, 13], shifts * 2, [["4", "0"]] * 6, [["1", "0"]] * 2, []
    )
    for lane_id, status, values in result:
        assert lane_id in [91, 13]
        assert status == "success"
        tolerance = Decimal(10) ** -(precision - 4)
        for actual, expected in zip(values, [[1, 0], [0, 0], [0, 0]], strict=True):
            assert all(
                abs(Decimal(part) - target) < tolerance
                for part, target in zip(actual, expected, strict=True)
            )


def test_invalid_shapes_serialization_and_precision_fail_at_boundary():
    state = saved_coefficient()
    with pytest.raises(ValueError, match="trailing"):
        CoefficientEvaluator(state + b"\0")
    with pytest.raises(ValueError, match="precision"):
        CoefficientEvaluator(state, 20)
    with pytest.raises(ValueError, match="higher-precision"):
        CoefficientEvaluator(state, 32).evaluate_f64([[2, 0]], 1)
    evaluator = CoefficientEvaluator(state)
    assert evaluator.evaluate_f64([[2, 0], [4, 1]], 2) == [[1, 0], [3, 1]]
    with pytest.raises(ValueError):
        evaluator.evaluate_decimal([], 2)
    with pytest.raises(ValueError, match="at most six"):
        PolynomialPlan(7, ["1", "0", "0"], [])
    plan = PolynomialPlan(1, ["c", "0", "0"], ["c"])
    with pytest.raises(ValueError):
        plan.evaluate_decimal([10], [[["0", "0"]] * 4], [["1", "0"]], [["1", "0"]], [])
