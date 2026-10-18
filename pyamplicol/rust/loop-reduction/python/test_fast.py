# SPDX-License-Identifier: 0BSD
"""Parity and domain boundaries for the upstream binary64 fast reducer."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import NativeBatch, PolynomialPlan


def batch(values, rows, columns, *, arithmetic="double", precision=None):
    return NativeBatch.from_decimal(
        values, rows, columns, arithmetic, precision, [419 - 3 * i for i in range(rows)]
    )


def test_fast_explicit_mu2_and_three_epsilon_sectors_match_analytic_triangle():
    plans = [
        PolynomialPlan(3, ["a*mu2", "b", "c"], ["a", "b", "c"], engine=engine)
        for engine in ["generic", "fast"]
    ]
    shifts = (
        [["0", "0"]] * 4
        + [["1", "0"], ["0", "0"], ["0", "0"], ["1", "0"]]
        + [["0", "0"], ["0", "0"], ["0", "0"], ["2", "0"]]
    )
    matrix = [[str(x), "0"] for x in [0, 0, -4, 0, 0, 0, -4, 0, 0]]
    for rows in [1, 33, 0, 1]:
        inputs = [
            batch(shifts * rows, rows, 12),
            batch([["0", "0"]] * (rows * 3), rows, 3),
            batch([["1", "0"]] * rows, rows, 1),
            batch([["2", "0"], ["3", "0"], ["5", "0"]] * rows, rows, 3),
        ]
        for plan in plans:
            result = plan.evaluate_batch(
                *inputs, s_matrix=batch(matrix * rows, rows, 9)
            )
            assert result.statuses == [(419 - 3 * i, "success") for i in range(rows)]
            values = result.values.to_decimal()
            with localcontext() as context:
                context.prec = 50
                expected = [
                    1 + 3 * Decimal(4).ln() / 4 - Decimal(5) / 4,
                    Decimal("-0.75"),
                    Decimal(0),
                ]
                for actual, target in zip(values, expected * rows, strict=True):
                    assert abs(Decimal(actual[0]) - target) < Decimal("1e-11")
                    assert abs(Decimal(actual[1])) < Decimal("1e-11")


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "generic"),
        ("double", None, "fast"),
        ("double-double", None, "generic"),
        ("arbitrary", 80, "generic"),
    ],
)
def test_symbolic_zero_sectors_match_parameter_zero_and_validate_inputs(
    arithmetic, precision, engine
):
    options = dict(arithmetic=arithmetic, precision=precision, engine=engine)
    optimized = PolynomialPlan(1, ["c", "0", "0"], ["c", "z"], **options)
    reference = PolynomialPlan(1, ["c", "z", "z"], ["c", "z"], **options)

    def b(values, rows, columns):
        return batch(values, rows, columns, arithmetic=arithmetic, precision=precision)

    for rows in [3, 0, 1]:
        inputs = [
            b([["0", "0"]] * (rows * 4), rows, 4),
            b([["1", "-0.25"]] * rows, rows, 1),
            b([["1", "0"]] * rows, rows, 1),
            b([["2", "0"], ["0", "0"]] * rows, rows, 2),
        ]
        left, right = [
            p.evaluate_batch(*inputs).values.to_decimal()
            for p in [optimized, reference]
        ]
        assert left == right
    zero = PolynomialPlan(1, ["0", "0", "0"], [], **options)
    with pytest.raises(ValueError):
        zero.evaluate_decimal([0], [], [], [["1", "0"]], [])


def test_fast_engine_is_explicit_about_arithmetic_and_real_kinematics():
    with pytest.raises(ValueError, match="only double"):
        PolynomialPlan(
            1, ["1", "0", "0"], [], arithmetic="double-double", engine="fast"
        )
    with pytest.raises(ValueError, match="engine"):
        PolynomialPlan(1, ["1", "0", "0"], [], engine="automatic")
    plan = PolynomialPlan(1, ["1", "0", "0"], [], engine="fast")
    with pytest.raises(ValueError, match="real shifts"):
        plan.evaluate_batch(
            batch([["0", "1"]] + [["0", "0"]] * 3, 1, 4),
            batch([["1", "0"]], 1, 1),
            batch([["1", "0"]], 1, 1),
            batch([], 1, 0),
        )


def test_fast_master_threshold_is_distinct_from_exact_zero_generic_policy():
    shifts = batch([["0", "0"]] * 4, 1, 4)
    masses = batch([["1e-12", "0"]], 1, 1)
    scales = batch([["1", "0"]], 1, 1)
    parameters = batch([], 1, 0)
    values = [
        PolynomialPlan(1, ["1", "0", "0"], [], engine=engine)
        .evaluate_batch(shifts, masses, scales, parameters)
        .values.to_decimal()
        for engine in ["generic", "fast"]
    ]
    assert Decimal(values[0][1][0]) > 0
    assert all(Decimal(component) == 0 for pair in values[1] for component in pair)
