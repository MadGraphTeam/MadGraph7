# SPDX-License-Identifier: 0BSD
"""Exact invariant ingress: no rounding until the polynomial result exists."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import ExactPolynomialEvaluator
from symbolica import E, Expression, S


def saved(expressions, parameters):
    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


@pytest.mark.parametrize(
    "arithmetic,precision,digits",
    [("double", None, 16), ("double-double", None, 32), ("arbitrary", 80, 80)],
)
def test_exact_invariant_zero_and_small_nonzero_survive_decimal_ingress(
    arithmetic, precision, digits
):
    energy, x, y = S("native_exact_e", "native_exact_x", "native_exact_y")
    evaluator = ExactPolynomialEvaluator(
        saved([energy**2 - (x + y) ** 2, energy**2 - x**2], [energy, x, y])
    )
    assert (evaluator.input_count, evaluator.output_count) == (3, 2)
    for rows in [2, 34, 0, 2]:
        inputs = [
            ["0.3", "0"],
            ["0.1", "0"],
            ["0.2", "0"],
            ["1.0000000000000000001", "0"],
            ["1", "0"],
            ["0", "0"],
        ] * (rows // 2)
        result = evaluator.evaluate_decimal(
            inputs, rows, arithmetic, precision, [1000 - i for i in range(rows)]
        )
        assert result.row_ids == [1000 - i for i in range(rows)]
        with localcontext() as context:
            context.prec = 160
            expected = [
                Decimal(0),
                Decimal("0.08"),
                Decimal("2e-19") + Decimal("1e-38"),
                Decimal("2e-19") + Decimal("1e-38"),
            ] * (rows // 2)
            for (re, im), target in zip(result.to_decimal(), expected, strict=True):
                assert Decimal(im) == 0
                if target == 0:
                    assert Decimal(re) == 0
                else:
                    assert abs(Decimal(re) / target - 1) < Decimal(10) ** -(digits - 3)


def test_decimal_literal_parser_matches_standard_decimal_exact_ratios():
    x = S("native_exact_identity_x")
    evaluator = ExactPolynomialEvaluator(saved([x], [x]))
    literals = [
        "0",
        "-0",
        "+0",
        "1.",
        ".125",
        "-.125",
        "+1.25e+2",
        "-1.25E-2",
        "00012.3400",
        "1e-400",
        "-987654321.00123456789",
    ]
    result = evaluator.evaluate_decimal(
        [[value, value] for value in literals], len(literals), "arbitrary", 160
    )
    with localcontext() as context:
        context.prec = 200
        for literal, pair in zip(literals, result.to_decimal(), strict=True):
            numerator, denominator = Decimal(literal).as_integer_ratio()
            expected = Decimal(numerator) / Decimal(denominator)
            for component in pair:
                value = Decimal(component)
                assert (
                    value == 0
                    if expected == 0
                    else abs(value / expected - 1) < Decimal("1e-155")
                )
    for literal in [
        "",
        " ",
        "1 2",
        " 1",
        "1 ",
        "+",
        "-",
        ".",
        "1..0",
        "1e",
        "1e+",
        "1e2e3",
        "--1",
        "1/2",
        "NaN",
        "inf",
        "-Infinity",
        "1_000",
        "1e9223372036854775808",
    ]:
        with pytest.raises(ValueError):
            evaluator.evaluate_decimal([[literal, "0"]], 1)


def test_exact_complex_coefficients_and_inputs_use_dependency_rational_ring():
    x, y = S("native_exact_complex_x", "native_exact_complex_y")
    evaluator = ExactPolynomialEvaluator(saved([E("1/3") * x * x + y], [x, y]))
    result = evaluator.evaluate_decimal(
        [["1.2", "-0.3"], ["-4.5", "2"]], 1, "arbitrary", 80
    )
    re, im = map(Decimal, result.to_decimal()[0])
    assert abs(re - Decimal("-4.05")) < Decimal("1e-75")
    assert abs(im - Decimal("1.76")) < Decimal("1e-75")


@pytest.mark.parametrize(
    "expression",
    [
        "native_exact_reject_x^-1",
        "sqrt(native_exact_reject_x)",
        "sin(native_exact_reject_x)",
        "native_exact_reject_x+pi",
        "native_exact_reject_x^native_exact_reject_x",
        "if(native_exact_reject_x,1,0)",
    ],
)
def test_exact_polynomial_boundary_rejects_other_operations(expression):
    x = S("native_exact_reject_x")
    state = saved([E(expression)], [x])
    with pytest.raises(ValueError, match="rational polynomial"):
        ExactPolynomialEvaluator(state)


def test_exact_constant_batches_and_shape_rejection():
    evaluator = ExactPolynomialEvaluator(saved([E("2/3")], []))
    result = evaluator.evaluate_decimal([], 3, "arbitrary", 80, [7, 2, 5])
    assert result.row_ids == [7, 2, 5]
    assert result.rows == 3
    with pytest.raises(ValueError):
        evaluator.evaluate_decimal([["0", "0"]], 1)
    with pytest.raises(ValueError):
        evaluator.evaluate_decimal([], 1, row_ids=[])
