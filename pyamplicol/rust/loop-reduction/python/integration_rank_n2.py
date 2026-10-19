# SPDX-License-Identifier: 0BSD
"""Independent massless rank-four bubble, including its finite epsilon mixing."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import NativeBatch, PolynomialPlan


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "generic", "2e-12"),
        ("double", None, "fast", "2e-12"),
        ("double", None, "fast-tensor", "2e-12"),
        ("double-double", None, "generic", "2e-25"),
        ("arbitrary", 80, "generic", "1e-60"),
    ],
)
def test_rank_four_lightlike_projection_bubble(
    arithmetic, precision, engine, tolerance
):
    # Metric (+---), p=(0,1,0,0), n=(1,1,0,0): p²=-1, n²=0, p.n=-1.
    # With D0=q²-mu², D1=(q+p)²-mu², Feynman parameters give
    # I[(q.n)^4] / B0 = Beta(5-eps,1-eps)/Beta(1-eps,1-eps)
    #                = 1/5 + 29*eps/300 + O(eps²).
    # Thus the finite answer includes the explicit 29/300 pole mixing.
    plan = PolynomialPlan(
        2,
        ["a*(q0-q1)^4", "0", "0"],
        ["a"],
        arithmetic=arithmetic,
        precision=precision,
        engine=engine,
    )
    with localcontext() as context:
        context.prec = 110
        for ids, coefficients, scales in (
            ([73, 9], ["1", "-2"], ["1", "4"]),
            ([9], ["3"], ["2"]),
            ([], [], []),
            ([73], ["1"], ["1"]),
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
                batch(["0", "0", "0", "0", "0", "1", "0", "0"] * rows, 8),
                batch(["0", "0"] * rows, 2),
                batch(scales, 1),
                batch(coefficients, 1),
                s_matrix=batch(["0", "-1", "-1", "0"] * rows, 4),
            )
            assert result.statuses == [(point, "success") for point in ids]
            assert result.values.row_ids == ids
            values = result.values.to_decimal()
            for row, (coefficient, scale) in enumerate(
                zip(coefficients, scales, strict=True)
            ):
                expected = (
                    Decimal(149) / 300 + Decimal(scale).ln() / 5,
                    Decimal(1) / 5,
                    Decimal(0),
                )
                for pair, value in zip(
                    values[3 * row : 3 * row + 3], expected, strict=True
                ):
                    assert abs(
                        Decimal(pair[0]) - Decimal(coefficient) * value
                    ) < Decimal(tolerance)
                    assert abs(Decimal(pair[1])) < Decimal(tolerance)
    # The actual selected reducer owns its weighted-rank limit and diagnostics.
    with pytest.raises(ValueError, match="rank"):
        PolynomialPlan(
            2,
            ["q0^5", "0", "0"],
            [],
            arithmetic=arithmetic,
            precision=precision,
            engine=engine,
        )
