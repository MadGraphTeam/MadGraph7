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
        for engine in ["generic", "fast", "fast-tensor"]
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
        ("double", None, "fast-tensor"),
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


@pytest.mark.parametrize("engine", ["fast", "fast-tensor"])
def test_fast_engine_is_explicit_about_arithmetic_and_real_kinematics(engine):
    with pytest.raises(ValueError, match="only double"):
        PolynomialPlan(
            1, ["1", "0", "0"], [], arithmetic="double-double", engine=engine
        )
    with pytest.raises(ValueError, match="engine"):
        PolynomialPlan(1, ["1", "0", "0"], [], engine="automatic")
    plan = PolynomialPlan(1, ["1", "0", "0"], [], engine=engine)
    with pytest.raises(ValueError, match="real shifts"):
        plan.evaluate_batch(
            batch([["0", "1"]] + [["0", "0"]] * 3, 1, 4),
            batch([["1", "0"]], 1, 1),
            batch([["1", "0"]], 1, 1),
            batch([], 1, 0),
        )


@pytest.mark.parametrize(
    "expression",
    ["c*mu2^2", "c*(q0*q1*q2*q3*q0 + mu2*q0*q2*q3 + mu2^2*q1) + d*(q2 + mu2)"],
)
def test_dense_mu_blocks_weighted_rank_and_overwrite_match_callback(expression):
    plans = [
        PolynomialPlan(4, [expression, "d*mu2", "c"], ["c", "d"], engine=e)
        for e in ["fast", "fast-tensor"]
    ]
    for rows in [3, 33, 1, 0, 8, 1]:
        shifts = batch(
            [
                [str(x), "0"]
                for _ in range(rows)
                for p in [(0, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0), (4, 0, 0, 1)]
                for x in p
            ],
            rows,
            16,
        )
        masses = batch([["1", "-0.1"]] * (4 * rows), rows, 4)
        scales = batch([["1", "0"]] * rows, rows, 1)
        parameters = batch([[str(i - rows), "0.25"] for i in range(2 * rows)], rows, 2)
        results = [p.evaluate_batch(shifts, masses, scales, parameters) for p in plans]
        assert (
            results[0].statuses
            == results[1].statuses
            == [(419 - 3 * i, "success") for i in range(rows)]
        )
        assert all(
            ok
            for _, ok in results[0].values.agreement_rows(
                results[1].values, "2e-9", "2e-9"
            )
        )
        # An actual failed lane must not expose the previous successful value.
        if rows:
            invalid = batch(
                [["NaN", "0"]] * 4 + [["1", "-0.1"]] * (4 * (rows - 1)), rows, 4
            )
            failed = plans[1].evaluate_batch(shifts, invalid, scales, parameters)
            assert failed.statuses[0][0] == 419 and failed.statuses[0][1] != "success"
            assert failed.successful_values().row_ids == [
                419 - 3 * i for i in range(1, rows)
            ]


def test_dense_tensor_rejects_rank_and_nonpolynomial_without_truncation():
    for expression in ["mu2^3", "q0^6"]:
        with pytest.raises(ValueError, match="rank"):
            PolynomialPlan(4, [expression, "0", "0"], [], engine="fast-tensor")
    with pytest.raises(ValueError, match="polynomial"):
        PolynomialPlan(4, ["1/q0", "0", "0"], [], engine="fast-tensor")
    with pytest.raises(ValueError, match="six propagators"):
        PolynomialPlan(7, ["1", "0", "0"], [], engine="fast-tensor")


def test_dense_head_overflow_is_explicit_and_next_batch_rebinds():
    plan = PolynomialPlan(1, ["c^2", "0", "0"], ["c"], engine="fast-tensor")
    inputs = [
        batch([["0", "0"]] * 8, 2, 4),
        batch([["1", "0"]] * 2, 2, 1),
        batch([["1", "0"]] * 2, 2, 1),
    ]
    good = batch([["2", "0"], ["3", "0"]], 2, 1)
    before = plan.evaluate_batch(*inputs, good).values
    with pytest.raises(ValueError, match="nonfinite polynomial coefficient"):
        plan.evaluate_batch(*inputs, batch([["1e200", "0"], ["3", "0"]], 2, 1))
    after = plan.evaluate_batch(*inputs, good).values
    assert before.to_decimal() == after.to_decimal()


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
