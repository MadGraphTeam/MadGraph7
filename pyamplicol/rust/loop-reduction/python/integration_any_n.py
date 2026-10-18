# SPDX-License-Identifier: 0BSD
"""A rank-n+1 seven-line numerator cancels to an analytic scalar triangle."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import NativeBatch, PolynomialPlan

SHIFTS = (
    ("0", "0", "0", "0"),
    ("1", "0", "0", "1"),
    ("0", "0", "0", "2"),
    ("3.25", "0.125", "-0.375", "0.25"),
    ("5.5", "-0.75", "0.625", "-0.125"),
    ("8.125", "0.375", "0.25", "-0.875"),
    ("11.75", "-0.5", "-0.875", "0.625"),
)
MASSES_SQUARED = ("0", "0", "0", "1", "2", "3", "4")


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "generic", "1e-6"),
        ("double", None, "fast-tensor", "1e-6"),
        ("double-double", None, "generic", "1e-22"),
        ("arbitrary", 80, "generic", "1e-60"),
    ],
)
def test_seven_propagator_rank_eight_cancellation_uses_real_masters(
    arithmetic, precision, engine, tolerance
):
    # D_i=(q+V_i)^2-m_i^2-mu^2. The exact numerator D3 D4 D5 D6
    # cancels four denominators, leaving C0(0,0,-4;0,0,0). Keeping -mu^2
    # is essential; a four-dimensional-only product is a different integral.
    denominators = [
        "("
        + "+".join(
            ("" if i == 0 else "-") + f"(q{i}+({v}))^2" for i, v in enumerate(shift)
        )
        + f"-({mass})-mu2)"
        for shift, mass in zip(SHIFTS[3:], MASSES_SQUARED[3:], strict=True)
    ]
    expression = "a*" + "*".join(denominators)
    plan = PolynomialPlan(
        7,
        [expression, "0", "0"],
        ["a"],
        arithmetic=arithmetic,
        precision=precision,
        engine=engine,
    )
    with localcontext() as context:
        context.prec = 110
        vectors = [tuple(map(Decimal, row)) for row in SHIFTS]
        matrix = [
            str((a[0] - b[0]) ** 2 - sum((a[i] - b[i]) ** 2 for i in range(1, 4)))
            for a in vectors
            for b in vectors
        ]
        # The coefficients are supplied anew for each real native batch.
        for ids, coefficients, scales in (
            ([71, 9], ["1", "-2"], ["1", "4"]),
            ([9], ["3"], ["2"]),
            ([], [], []),
            ([71], ["1"], ["1"]),
        ):
            rows = len(ids)

            def batch(values, columns, rows=rows, ids=ids):
                return NativeBatch.from_exact_decimal(
                    [[value, "0"] for value in values],
                    rows,
                    columns,
                    arithmetic=arithmetic,
                    precision=precision,
                    row_ids=ids,
                )

            result = plan.evaluate_batch(
                batch([x for v in SHIFTS for x in v] * rows, 28),
                batch(list(MASSES_SQUARED) * rows, 7),
                batch(scales, 1),
                batch(coefficients, 1),
                s_matrix=batch(matrix * rows, 49),
            )
            assert result.statuses == [(point, "success") for point in ids]
            assert result.values.row_ids == ids
            actual = result.values.to_decimal()
            for row, (coefficient, scale) in enumerate(
                zip(coefficients, scales, strict=True)
            ):
                logarithm = (Decimal(4) / Decimal(scale)).ln()
                expected = [-(logarithm**2) / 8, logarithm / 4, -Decimal(1) / 4]
                for pair, value in zip(
                    actual[3 * row : 3 * row + 3], expected, strict=True
                ):
                    target = Decimal(coefficient) * value
                    assert abs(Decimal(pair[0]) - target) < Decimal(tolerance)
                    assert abs(Decimal(pair[1])) < Decimal(tolerance)
